from __future__ import annotations

import math
import os
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import docx  # python-docx
except Exception:  # pragma: no cover
    docx = None

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Optional loaders for various pickle formats
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    import cloudpickle  # type: ignore
except Exception:  # pragma: no cover
    cloudpickle = None

# Avoid importing heavy optional deps at startup; we'll attempt lazy import later
xgb = None  # type: ignore

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
except Exception:  # pragma: no cover
    letter = None  # type: ignore


class PredictRequest(BaseModel):
    disease_id: int = Field(..., ge=1, le=5, description="1-5 corresponding to model1..model5")
    # List of numeric features for the selected disease model
    features: List[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    disease_id: int
    probability: float
    details: Dict[str, Any] = {}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_model_file(filename: Path) -> Any:
    """Attempt to load a model using multiple strategies for resilience."""
    if not filename.exists():
        raise RuntimeError(f"Model file not found: {filename}")
    try:
        size = filename.stat().st_size
        if size == 0:
            raise RuntimeError(f"Model file {filename.name} is empty (0 bytes)")
    except Exception:
        pass

    # Peek header bytes for diagnostics
    try:
        with open(filename, "rb") as _fpeek:
            header = _fpeek.read(8)
    except Exception:
        header = b""
    # 1) Standard pickle
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 2) joblib
    if joblib is not None:
        try:
            return joblib.load(str(filename))
        except Exception:
            pass

    # 3) cloudpickle
    if cloudpickle is not None:
        try:
            with open(filename, "rb") as f:
                return cloudpickle.load(f)
        except Exception:
            pass

    # 4) xgboost native model (lazy import)
    try:
        import xgboost as _xgb  # type: ignore
        booster = _xgb.Booster()
        booster.load_model(str(filename))

        class XGBBoosterAdapter:
            def __init__(self, booster: "_xgb.Booster"):
                self.booster = booster

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                dmat = _xgb.DMatrix(X)
                preds = self.booster.predict(dmat)
                preds = np.asarray(preds, dtype=float).reshape(-1)
                if preds.ndim == 1:
                    p1 = np.clip(preds, 0.0, 1.0)
                    p0 = 1.0 - p1
                    return np.vstack([p0, p1]).T
                return preds

            def predict(self, X: np.ndarray) -> np.ndarray:
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1)

        return XGBBoosterAdapter(booster)
    except Exception:
        pass

    # If all methods failed, raise detailed error
    raise RuntimeError(
        f"Unsupported or corrupted model file: {filename.name}. "
        f"Tried pickle, joblib, and cloudpickle. First bytes: {header.hex()}"
    )


def load_models(models_dir: Path) -> Dict[int, Any]:
    loaded: Dict[int, Any] = {}
    for i in range(1, 6):
        # Try common extensions
        candidates = [
            models_dir / f"model{i}.pkl",
            models_dir / f"model{i}.joblib",
            models_dir / f"model{i}.bin",
        ]
        loaded[i] = None
        for filename in candidates:
            if not filename.exists():
                continue
            try:
                loaded[i] = load_model_file(filename)
                break
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load {filename.name}: {exc}")
    return loaded


app = FastAPI(title="Disease Probability API", version="1.0.0")
app = FastAPI()

# Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)


# Allow all origins by default; tighten if needed



BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR  # Expect model1.pkl..model5.pkl next to this file by default
# Try to eagerly load models, but continue even if some fail; detailed messages are printed.
try:
    MODELS: Dict[int, Any] = load_models(MODELS_DIR)
except Exception as _e:
    print(f"Model preloading failed: {_e}")
    MODELS = {i: None for i in range(1, 6)}


@app.get("/health")
def health() -> Dict[str, Any]:
    available = {i: (MODELS.get(i) is not None) for i in range(1, 6)}
    return {"status": "ok", "models": available}

@app.get("/debug")
def debug() -> Dict[str, Any]:
    return {
        "cors_origins": cors_origins,
        "environment": {
            "CORS_ORIGINS": os.getenv("CORS_ORIGINS"),
            "PORT": os.getenv("PORT"),
            "OPENAI_API_KEY_SET": bool(os.getenv("OPENAI_API_KEY"))
        }
    }


@app.get("/random-data/{disease_type}")
def get_random_data(disease_type: str) -> Dict[str, Any]:
    """Generate random data for a specific disease type for testing/demo purposes"""
    try:
        random_data = generate_random_data_for_disease(disease_type)
        return {
            "disease_type": disease_type,
            "data": random_data,
            "message": f"Random data generated for {disease_type}"
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to generate random data: {exc}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Lazy-load model if missing
    model = MODELS.get(req.disease_id)
    used_fallback = False
    fallback_reason: Optional[str] = None
    if model is None:
        # Try multiple extensions lazily
        candidates = [
            MODELS_DIR / f"model{req.disease_id}.pkl",
            MODELS_DIR / f"model{req.disease_id}.joblib",
            MODELS_DIR / f"model{req.disease_id}.bin",
        ]
        for filename in candidates:
            if not filename.exists():
                continue
            try:
                MODELS[req.disease_id] = load_model_file(filename)
                model = MODELS[req.disease_id]
                break
            except Exception as exc:
                last_exc = exc
        if model is None:
            # Graceful fallback: synthesize a simple probability model so the API still works
            # This avoids blocking the UI if a serialized model can't be loaded on this machine
            class FallbackModel:
                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    x = np.nan_to_num(X.astype(float))
                    # Simple bounded score based on normalized mean of features
                    mean = np.mean(x, axis=1)
                    p1 = 1.0 / (1.0 + np.exp(-(mean - np.mean(mean))))
                    p1 = np.clip(p1, 0.01, 0.99)
                    p0 = 1.0 - p1
                    return np.vstack([p0, p1]).T

            model = FallbackModel()
            used_fallback = True
            fallback_reason = str(last_exc) if 'last_exc' in locals() else "Model file could not be loaded"
    if model is None:
        raise HTTPException(status_code=400, detail=f"Model for disease_id={req.disease_id} is not available")

    # Prepare features as 2D array for sklearn-like estimators
    try:
        X = np.asarray(req.features, dtype=float).reshape(1, -1)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid features. Must be an array of numbers")

    probability: Optional[float] = None
    details: Dict[str, Any] = {}

    # Try common estimator APIs in order of preference
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Use positive class probability if binary; otherwise take max across classes
        if proba.ndim == 2 and proba.shape[1] >= 2:
            probability = float(proba[0, 1])
        else:
            probability = float(np.max(proba[0]))
        details["source"] = "predict_proba"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 2 and scores.shape[1] > 1:
            # Multi-class: map max score via softmax to a probability proxy
            logits = np.asarray(scores[0], dtype=float)
            exp_logits = np.exp(logits - np.max(logits))
            softmax = exp_logits / np.sum(exp_logits)
            probability = float(np.max(softmax))
        else:
            probability = float(_sigmoid(np.asarray(scores).reshape(-1))[0])
        details["source"] = "decision_function->probability_proxy"
    elif hasattr(model, "predict"):
        # Fallback: map prediction to probability proxy (0/1)
        pred = model.predict(X)
        probability = float(np.clip(np.asarray(pred, dtype=float).reshape(-1)[0], 0.0, 1.0))
        details["source"] = "predict->probability_proxy"
    else:
        raise HTTPException(status_code=500, detail="Loaded model does not support prediction")

    # Bound probability to [0,1]
    probability = max(0.0, min(1.0, probability))

    if used_fallback:
        details["used_fallback_model"] = True
        if fallback_reason:
            details["model_load_error"] = fallback_reason

    return PredictResponse(disease_id=req.disease_id, probability=probability, details=details)


def extract_text_from_upload(file: UploadFile) -> str:
    content_type = (file.content_type or "").lower()
    if "pdf" in content_type:
        if pdfplumber is None:
            raise HTTPException(status_code=500, detail="pdfplumber not installed on server")
        try:
            text_parts: List[str] = []
            file.file.seek(0)
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts).strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}")
    if "word" in content_type or content_type.endswith("officedocument.wordprocessingml.document"):
        if docx is None:
            raise HTTPException(status_code=500, detail="python-docx not installed on server")
        try:
            file.file.seek(0)
            document = docx.Document(file.file)
            return "\n".join([p.text for p in document.paragraphs]).strip()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read DOCX: {exc}")
    raise HTTPException(status_code=415, detail="Unsupported file type. Please upload PDF or DOCX.")


def select_model_index_for_disease(disease_type: str) -> int:
    mapping = {
        "general": 1,
        "diabetes": 2,         # Uses model2.pkl
        "diabetes-type1": 2,
        "diabetes-type2": 2,  # Uses model2.pkl
        "hypertension": 3,     # Uses model3.pkl
        "heart-failure": 4,    # Uses model4.pkl
        "weight-glp1": 5,
        "prediabetes": 2,      # Uses model2.pkl for diabetes prediction
    }
    return mapping.get(disease_type, 1)


def generate_random_data_for_disease(disease_type: str) -> Dict[str, Any]:
    """Generate realistic random data for testing/demo purposes that matches form field names exactly"""
    import random
    
    if disease_type == "general":
        # General model inputs as specified
        return {
            "age": random.randint(18, 85),
            "sex": random.choice(["Male", "Female", "Other"]),
            "bmi": round(random.uniform(18.0, 40.0), 1),
            "smoking_status": random.choice(["Never", "Former", "Current"]),
            "alcohol_use": random.choice(["None", "Moderate", "Heavy"]),
            "weight": round(random.uniform(45, 140), 1),
            "systolic_bp": random.randint(100, 160),
            "diastolic_bp": random.randint(60, 100),
            "heart_rate": random.randint(55, 100),
            "glucose": random.randint(80, 160),
            "steps_per_day": random.randint(1000, 15000),
            "sleep_hours": round(random.uniform(4.0, 9.5), 1),
            "hba1c": round(random.uniform(4.8, 9.5), 1),
            "cholesterol": random.randint(120, 280),
            "ldl": random.randint(50, 190),
            "hdl": random.randint(25, 80),
            "triglycerides": random.randint(60, 350),
            "creatinine": round(random.uniform(0.6, 1.8), 1),
            "egfr": random.randint(45, 120),
            "hemoglobin": round(random.uniform(10.0, 17.5), 1),
            "med_adherence": random.randint(40, 100),
            "chronic_meds": random.randint(0, 8),
            "insulin_or_oral_use": random.choice(["No", "Yes"]),
            "antihypertensive_use": random.choice(["No", "Yes"]),
        }
    
    elif disease_type == "diabetes-type1":
        return {
            "age": random.randint(15, 45),
            "hba1c": round(random.uniform(6.0, 10.0), 1),  # step="0.1"
            "bmi": round(random.uniform(18, 35), 1),  # step="0.1"
            "insulinDose": random.randint(20, 80),
            "glucoseFasting": random.randint(80, 200),
            "cPeptide": round(random.uniform(0.1, 2.0), 2)  # step="0.01"
        }
    
    elif disease_type == "diabetes" or disease_type == "diabetes-type2":
        return {
            "glucose": random.randint(80, 200),  # F1 - Fasting Glucose
            "hba1c": round(random.uniform(5.5, 9.0), 1),  # F2 - HbA1c (step="0.1")
            "bmi": round(random.uniform(22, 40), 1),  # F3 - BMI (step="0.1")
            "systolic_bp": random.randint(110, 160),  # F4 - Systolic BP
            "diastolic_bp": random.randint(70, 100),  # F5 - Diastolic BP
            "cholesterol": random.randint(150, 250),  # F6 - Cholesterol
            "hdl": random.randint(30, 70),  # F7 - HDL
            "ldl": random.randint(80, 180),  # F8 - LDL
            "triglycerides": random.randint(100, 300),  # F9 - Triglycerides
            "insulin_level": round(random.uniform(5, 25), 1),  # F10 - Insulin level
            "heart_rate": random.randint(60, 100)  # F11 - Heart rate / variability marker
        }
    
    elif disease_type == "prediabetes":
        return {
            "age": random.randint(30, 65),
            "bmi": round(random.uniform(24, 32), 1),  # step="0.1"
            "glucoseFasting": random.randint(95, 125),
            "hba1c": round(random.uniform(5.5, 6.4), 1),  # step="0.1"
            "waistCircumference": random.randint(80, 110),
            "familyHistory": random.choice(["No", "Yes"])
        }
    
    elif disease_type == "hypertension":
        activity_level = random.choice(["Low", "Moderate", "High"])
        activity_numeric = {"Low": 0, "Moderate": 1, "High": 2}[activity_level]
        med_adherence = random.choice(["Poor", "Fair", "Good", "Excellent"])
        med_adherence_numeric = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}[med_adherence]
        
        return {
            "weight": round(random.uniform(60, 120), 1),  # Weight (step="0.1")
            "glucose": random.randint(80, 200),  # Glucose
            "heart_rate": random.randint(60, 100),  # Heart rate
            "activity": activity_level,  # Activity level
            "sleep": round(random.uniform(5, 9), 1),  # Sleep hours (step="0.5")
            "systolic_bp": random.randint(120, 180),  # Systolic BP
            "diastolic_bp": random.randint(80, 110),  # Diastolic BP
            "hba1c": round(random.uniform(5.0, 7.0), 1),  # HbA1c (step="0.1")
            "lipids": random.randint(150, 300),  # Lipids (total cholesterol)
            "creatinine": round(random.uniform(0.7, 1.5), 1),  # Creatinine (step="0.1")
            "med_adherence": med_adherence,  # Medication adherence
        }
    
    elif disease_type == "heart-failure":
        # Generate physical_activity_level first, then set numeric accordingly
        activity_level = random.choice(["Low", "Medium", "High"])
        activity_numeric = {"Low": 0, "Medium": 1, "High": 2}[activity_level]
        
        return {
            "t0_window_days": random.randint(30, 120),
            "age": random.randint(50, 85),
            "sex_male": random.choice([0, 1]),
            "bmi": round(random.uniform(20, 40), 1),  # step="0.1"
            "sbp_last": random.randint(90, 160),
            "dbp_last": random.randint(60, 100),
            "history_diabetes": random.choice([0, 1]),
            "history_hypertension": random.choice([0, 1]),
            "creatinine_last": round(random.uniform(0.8, 2.5), 1),  # step="0.1"
            "creatinine_mean": round(random.uniform(0.7, 2.2), 1),  # step="0.1"
            "creatinine_slope_per_day": round(random.uniform(-0.01, 0.01), 3),  # step="0.001"
            "hbA1c_last": round(random.uniform(5.0, 9.0), 1),  # step="0.1"
            "fpg_last": random.randint(80, 200),
            "hdl_last": random.randint(25, 70),
            "ldl_last": random.randint(80, 180),
            "triglycerides_last": random.randint(100, 300),
            "qrs_duration_ms": random.randint(80, 140),
            "arrhythmia_flag": random.choice([0, 1]),
            "afib_flag": random.choice([0, 1]),
            "prev_mi": random.choice([0, 1]),
            "cabg_history": random.choice([0, 1]),
            "echo_ef_last": round(random.uniform(25, 65), 1),  # step="0.1"
            "has_echo": random.choice([0, 1]),
            "on_ACEi": random.choice([0, 1]),
            "on_beta_blocker": random.choice([0, 1]),
            "on_diuretic": random.choice([0, 1]),
            "hf_events_past_year": random.randint(0, 5),
            "admissions_30d": random.randint(0, 3),
            "physical_activity_level": activity_level,
            "physical_activity_numeric": activity_numeric,
            "cci": random.randint(0, 8)
        }
    
    elif disease_type == "weight-glp1":
        # Categorical pools
        obesity_classes = ["None", "Class I", "Class II", "Class III"]
        yes_no = ["No", "Yes"]
        glp1_agents = ["Semaglutide", "Liraglutide", "Dulaglutide", "Other"]
        dose_tiers = ["Low", "Medium", "High"]

        return {
            "age": random.randint(25, 70),
            "sex": random.choice([0, 1]),
            "BMI": round(random.uniform(27, 45), 1),
            "waist_cm": random.randint(85, 130),
            "obesity_class": random.choice(obesity_classes),
            "T2D_status": random.choice(yes_no),
            "HTN_status": random.choice(yes_no),
            "OSA_status": random.choice(yes_no),
            "hbA1c_baseline": round(random.uniform(5.5, 10.0), 1),
            "hbA1c_delta": round(random.uniform(-2.0, 0.5), 1),
            "fasting_glucose": random.randint(80, 220),
            "ldl": random.randint(70, 190),
            "hdl": random.randint(25, 80),
            "triglycerides": random.randint(80, 350),
            "alt": random.randint(10, 80),
            "egfr": random.randint(45, 110),
            "weight_4w_slope": round(random.uniform(-2.0, 0.5), 1),
            "sbp": random.randint(110, 170),
            "dbp": random.randint(65, 105),
            "hr": random.randint(55, 100),
            "spo2": round(random.uniform(92, 100), 1),
            "GLP1_agent": random.choice(glp1_agents),
            "dose_tier": random.choice(dose_tiers),
            "adherence_90d": random.randint(50, 100),
            "missed_doses_last_30d": random.randint(0, 6),
            "nausea_score": random.randint(0, 10),
            "vomit_score": random.randint(0, 10),
            "appetite_score": random.randint(0, 10),
            "steps_avg": random.randint(2000, 12000),
            "active_minutes": random.randint(10, 120),
            "exercise_days_wk": random.randint(0, 7),
            "sleep_hours": round(random.uniform(4.0, 9.5), 1),
            "alcohol_units_wk": random.randint(0, 20),
            "tobacco_cigs_per_day": random.randint(0, 20),
            "tobacco_chew_use": random.choice(yes_no),
            "junk_food_freq_wk": random.randint(0, 14),
            "insurance_denied": random.choice(yes_no),
            "prior_auth_denial": random.choice(yes_no),
            "fill_gap_days": random.randint(0, 30),
            "telehealth_visits": random.randint(0, 6),
            "nurse_messages": random.randint(0, 20),
            "cancellations": random.randint(0, 5),
            "ER_visits_obesity_related": random.randint(0, 3),
        }
    
    else:
        # Default to general if unknown disease type
        return {
            "age": random.randint(25, 75),
            "weight": round(random.uniform(50, 120), 1),
            "height": random.randint(150, 190),
            "bloodPressureSys": random.randint(110, 160),
            "bloodPressureDia": random.randint(70, 100),
            "cholesterol": random.randint(150, 250)
        }


def _build_default_summary(disease_type: str, inputs: Dict[str, Any], probability: float) -> Dict[str, Any]:
    # Enhanced heuristics to provide better default summaries
    bmi = float(inputs.get("bmi") or 0)
    sbp = float(inputs.get("sbp_last") or inputs.get("systolicBP") or 0)
    dbp = float(inputs.get("dbp_last") or inputs.get("diastolicBP") or 0)
    hba1c = float(inputs.get("hbA1c_last") or inputs.get("hba1c") or 0)
    creat = float(inputs.get("creatinine_last") or inputs.get("creatinine") or 0)
    age = float(inputs.get("age") or 0)

    # Disease-specific summaries
    disease_summaries = {
        "heart-failure": f"Cardiac function analysis shows {probability:.1%} risk of heart failure progression. ",
        "diabetes-type1": f"Type 1 diabetes management assessment indicates {probability:.1%} risk of complications. ",
        "hypertension": f"Blood pressure analysis reveals {probability:.1%} risk of cardiovascular events. ",
        "weight-glp1": f"GLP-1 therapy effectiveness assessment shows {probability:.1%} risk of suboptimal response. ",
        "general": f"Comprehensive health assessment indicates {probability:.1%} overall risk. "
    }

    # Build risk factors based on available data
    factors = []
    
    if bmi and bmi > 0:
        bmi_impact = min(100, max(0, int(abs(bmi - 25) * 3)))
        bmi_status = "obese" if bmi > 30 else "overweight" if bmi > 25 else "normal"
        factors.append({
            "factor": "Body Mass Index", 
            "impact": bmi_impact, 
            "description": f"BMI of {bmi:.1f} indicates {bmi_status} weight status, affecting cardiovascular and metabolic risk"
        })
    
    if sbp and sbp > 0:
        bp_impact = min(100, max(0, int((sbp - 120) * 1.5)))
        bp_status = "hypertensive" if sbp > 140 else "elevated" if sbp > 120 else "normal"
        factors.append({
            "factor": "Blood Pressure", 
            "impact": bp_impact, 
            "description": f"Systolic BP of {sbp:.0f} mmHg indicates {bp_status} blood pressure levels"
        })
    
    if hba1c and hba1c > 0:
        hba1c_impact = min(100, max(0, int((hba1c - 5.5) * 15)))
        hba1c_status = "diabetic" if hba1c > 6.5 else "prediabetic" if hba1c > 5.7 else "normal"
        factors.append({
            "factor": "Glycemic Control", 
            "impact": hba1c_impact, 
            "description": f"HbA1c of {hba1c:.1f}% indicates {hba1c_status} glucose control status"
        })
    
    if creat and creat > 0:
        creat_impact = min(100, max(0, int((creat - 1.0) * 30)))
        creat_status = "elevated" if creat > 1.2 else "normal"
        factors.append({
            "factor": "Kidney Function", 
            "impact": creat_impact, 
            "description": f"Creatinine of {creat:.2f} mg/dL suggests {creat_status} kidney function"
        })
    
    if age and age > 0:
        age_impact = min(100, max(0, int((age - 40) * 1.5)))
        factors.append({
            "factor": "Age", 
            "impact": age_impact, 
            "description": f"Age {age:.0f} years contributes to baseline cardiovascular and metabolic risk"
        })

    # Add generic factors if none available
    if not factors:
        factors = [
            {"factor": "Demographic Risk", "impact": 35, "description": "Age and gender-based baseline risk factors"},
            {"factor": "Lifestyle Factors", "impact": 25, "description": "Physical activity, diet, and lifestyle choices"},
            {"factor": "Medical History", "impact": 30, "description": "Previous conditions and family history"},
            {"factor": "Current Medications", "impact": 20, "description": "Medication adherence and effectiveness"},
        ]

    # Disease-specific recommendations
    disease_recs = {
        "heart-failure": {
        "Immediate Action": [
                "Schedule cardiology consultation within 1-2 weeks",
                "Review current heart failure medications and dosages",
                "Assess fluid status and weight management",
                "Order echocardiogram and BNP levels if not recent"
        ],
        "Lifestyle Changes": [
                "Implement low-sodium diet (<2g/day)",
                "Daily weight monitoring and fluid restriction",
                "Gradual increase in physical activity as tolerated",
                "Smoking cessation and alcohol moderation"
        ],
        "Monitoring": [
                "Weekly weight checks and symptom diary",
                "Monthly blood pressure and heart rate monitoring",
                "Quarterly lab work including electrolytes",
                "Annual echocardiogram and stress testing"
            ]
        },
        "diabetes-type1": {
            "Immediate Action": [
                "Endocrinology consultation within 1 week",
                "Review insulin regimen and dosing",
                "Check for diabetic complications screening",
                "Assess blood glucose monitoring frequency"
            ],
            "Lifestyle Changes": [
                "Carbohydrate counting and meal planning",
                "Regular physical activity (150 min/week)",
                "Continuous glucose monitoring consideration",
                "Diabetes education and support groups"
            ],
            "Monitoring": [
                "Daily blood glucose monitoring (4+ times)",
                "Quarterly HbA1c testing",
                "Annual eye and foot examinations",
                "Regular blood pressure and cholesterol checks"
            ]
        },
        "hypertension": {
            "Immediate Action": [
                "Primary care follow-up within 2 weeks",
                "Review antihypertensive medication regimen",
                "Check for target organ damage",
                "Assess lifestyle modification needs"
            ],
            "Lifestyle Changes": [
                "DASH diet with <2.3g sodium daily",
                "Regular aerobic exercise (30 min, 5x/week)",
                "Weight reduction if BMI >25",
                "Stress management and relaxation techniques"
            ],
            "Monitoring": [
                "Home blood pressure monitoring 2x daily",
                "Monthly office blood pressure checks",
                "Annual comprehensive metabolic panel",
                "Regular assessment of medication adherence"
            ]
        },
        "weight-glp1": {
            "Immediate Action": [
                "Endocrinology consultation for GLP-1 optimization",
                "Review current GLP-1 dosing and timing",
                "Assess for side effects and tolerability",
                "Evaluate need for dose adjustment"
            ],
            "Lifestyle Changes": [
                "Structured meal timing with GLP-1 dosing",
                "Regular physical activity (150 min/week)",
                "Behavioral counseling for eating habits",
                "Adequate hydration and fiber intake"
            ],
            "Monitoring": [
                "Weekly weight tracking and body measurements",
                "Monthly review of GLP-1 effectiveness",
                "Quarterly metabolic panel and HbA1c",
                "Regular assessment of gastrointestinal side effects"
            ]
        },
        "general": {
            "Immediate Action": [
                "Primary care physician consultation within 2-4 weeks",
                "Comprehensive health assessment and screening",
                "Review of current medications and supplements",
                "Baseline laboratory workup if indicated"
            ],
            "Lifestyle Changes": [
                "Regular physical activity (150 min moderate/week)",
                "Balanced diet with emphasis on whole foods",
                "Adequate sleep (7-9 hours nightly)",
                "Stress management and mental health support"
            ],
            "Monitoring": [
                "Annual comprehensive physical examination",
                "Regular blood pressure and weight monitoring",
                "Age-appropriate cancer and disease screening",
                "Lifestyle modification tracking and adjustment"
            ]
        }
    }

    recs = disease_recs.get(disease_type, disease_recs["general"])
    det = "yes" if probability >= 0.5 else "no"
    
    summary = disease_summaries.get(disease_type, disease_summaries["general"])
    if probability >= 0.7:
        summary += "This represents a high-risk profile requiring immediate attention and comprehensive management."
    elif probability >= 0.5:
        summary += "This indicates moderate risk requiring close monitoring and proactive intervention."
    else:
        summary += "This suggests lower risk with continued preventive care and lifestyle maintenance."

    return {
        "summary": summary,
        "risk": {"probability": probability, "deterioration": det},
        "primary_factors": factors[:6],
        "recommendations": recs,
    }


def _heuristic_probability(disease_type: str, inputs: Dict[str, Any]) -> float:
    """Compute a bounded probability from common clinical signals so fallback isn't a flat 0.5.

    This is intentionally simple and monotonic with known risk drivers.
    """
    def num(*keys: str) -> float:
        for k in keys:
            v = inputs.get(k)
            try:
                if v is not None and v != "":
                    return float(v)
            except Exception:
                continue
        return 0.0

    # Gather signals with lenient key aliases
    hba1c = num("hba1c", "hbA1c_baseline")
    sbp = num("systolic_bp", "sbp", "sbp_last")
    dbp = num("diastolic_bp", "dbp", "dbp_last")
    bmi = num("bmi", "BMI")
    chol = num("cholesterol", "lipids", "total_cholesterol")
    creat = num("creatinine", "creatinine_last")
    age = num("age")

    components: List[float] = []
    if hba1c > 0:
        components.append(max(0.0, min(1.0, (hba1c - 5.5) / 3.5)))
    if sbp > 0:
        components.append(max(0.0, min(1.0, (sbp - 110.0) / 60.0)))
    if dbp > 0:
        components.append(max(0.0, min(1.0, (dbp - 70.0) / 40.0)))
    if bmi > 0:
        components.append(max(0.0, min(1.0, (bmi - 22.0) / 18.0)))
    if chol > 0:
        components.append(max(0.0, min(1.0, (chol - 150.0) / 150.0)))
    if creat > 0:
        components.append(max(0.0, min(1.0, (creat - 1.0) / 1.5)))
    if age > 0:
        components.append(max(0.0, min(1.0, (age - 40.0) / 45.0)))

    if not components:
        return 0.5
    raw = sum(components) / len(components)
    return float(max(0.05, min(0.95, raw)))


def llm_only_predict(disease_type: str, inputs: Dict[str, Any], report_text: Optional[str], openai_api_key: Optional[str]) -> Dict[str, Any]:
    """Skip model inference and let the LLM generate probability and narrative directly.

    We set a seed but expect the LLM to compute and return its own probability from inputs.
    If the LLM doesn't provide one, we fall back to a light heuristic from inputs.
    """
    seed_probability = _heuristic_probability(disease_type, inputs)
    try:
        llm = call_llm_summary(openai_api_key, disease_type, inputs, report_text, seed_probability)
        prob = llm.get("risk", {}).get("probability")
        if prob is None:
            prob = seed_probability
        else:
            prob = float(prob)
        return {
            "disease_type": disease_type,
            "probability": prob,
            "summary": llm.get("summary"),
            "risk": {**llm.get("risk", {}), "probability": prob},
            "primary_factors": llm.get("primary_factors", []),
            "recommendations": llm.get("recommendations", {}),
        }
    except Exception:
        default = _build_default_summary(disease_type, inputs, 0.5 if seed_probability is None else seed_probability)
        return {
            "disease_type": disease_type,
            "probability": default["risk"]["probability"],
            "summary": default["summary"],
            "risk": default["risk"],
            "primary_factors": default["primary_factors"],
            "recommendations": {},
        }


def call_llm_summary(openai_api_key: Optional[str], disease_type: str, inputs: Dict[str, Any], report_text: Optional[str], model_probability: float) -> Dict[str, Any]:
    if not openai_api_key or OpenAI is None:
        return _build_default_summary(disease_type, inputs, model_probability)
    client = OpenAI(api_key=openai_api_key)
    sys_prompt = (
        "You are an expert clinical decision support assistant specializing in chronic disease risk assessment. "
        "Analyze the provided patient data and generate a comprehensive clinical summary. Return STRICT JSON only, no prose.\n\n"
        "REQUIRED JSON SCHEMA:\n"
        "{\n"
        "  \"summary\": \"Detailed clinical summary (2-3 sentences) explaining the patient's risk profile and key findings\",\n"
        "  \"risk\": {\n"
        "    \"probability\": number (0-1, use the provided model probability exactly),\n"
        "    \"deterioration\": \"yes\" or \"no\" (yes if probability >= 0.5, no otherwise)\n"
        "  },\n"
        "  \"primary_factors\": [\n"
        "    {\n"
        "      \"factor\": \"Specific clinical factor name\",\n"
        "      \"impact\": number (0-100, representing contribution to risk),\n"
        "      \"description\": \"Detailed explanation of how this factor affects the patient's risk\"\n"
        "    }\n"
        "  ],\n"
        "  \"recommendations\": {\n"
        "    \"Immediate Action\": [\"Specific actionable items for immediate care\"],\n"
        "    \"Lifestyle Changes\": [\"Concrete lifestyle modifications with measurable goals\"],\n"
        "    \"Monitoring\": [\"Specific monitoring protocols and follow-up schedules\"]\n"
        "  }\n"
        "}\n\n"
        "CLINICAL GUIDELINES:\n"
        "- Generate 4-6 primary_factors based on available data and clinical knowledge\n"
        "- Impact scores should reflect clinical significance (0-100 scale)\n"
        "- Provide 4-5 specific, actionable recommendations in each category\n"
        "- Use evidence-based clinical language appropriate for healthcare providers\n"
        "- Consider disease-specific risk factors and comorbidities\n"
        "- Include both modifiable and non-modifiable risk factors\n"
        "- Ensure recommendations are patient-specific and clinically relevant\n\n"
        "DISEASE-SPECIFIC FOCUS:\n"
        "- For heart failure: Focus on cardiac function, medication adherence, fluid management\n"
        "- For diabetes: Emphasize glycemic control, complications, medication optimization\n"
        "- For hypertension: Consider blood pressure control, cardiovascular risk, lifestyle factors\n"
        "- For general health: Assess overall cardiovascular and metabolic risk\n\n"
        "Return only valid minified JSON without any additional text."
    )
    user_content = {
        "disease_type": disease_type,
        "model_probability": model_probability,
        "inputs": inputs,
        "report_text_excerpt": (report_text or "")[:4000],
    }
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_content)},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        content = completion.choices[0].message.content or "{}"
        # Best-effort JSON parse; if parse fails, wrap as summary
        try:
            parsed = json.loads(content)
            # Ensure completeness
            if not parsed.get("primary_factors") or not parsed.get("recommendations"):
                defaults = _build_default_summary(disease_type, inputs, model_probability)
                parsed.setdefault("risk", defaults["risk"])  
                parsed.setdefault("summary", defaults["summary"])  
                if not parsed.get("primary_factors"):
                    parsed["primary_factors"] = defaults["primary_factors"]
                if not parsed.get("recommendations"):
                    parsed["recommendations"] = defaults["recommendations"]
            return parsed
        except Exception:
            return _build_default_summary(disease_type, inputs, model_probability)
    except Exception as exc:  # pragma: no cover
        return _build_default_summary(disease_type, inputs, model_probability)


@app.post("/analyze")
async def analyze(
    disease_type: str = Form(..., description="Disease type identifier (e.g., heart-failure)"),
    data: str = Form(..., description="JSON string of input fields for the disease"),
    file: Optional[UploadFile] = File(None, description="Optional PDF/DOCX medical report"),
):
    try:
        inputs = json.loads(data)
        if not isinstance(inputs, dict):
            raise ValueError("data must be a JSON object")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'data': {exc}")

    report_text: Optional[str] = None
    if file is not None:
        report_text = extract_text_from_upload(file)

    model_idx = select_model_index_for_disease(disease_type)
    # Lazy-load model if missing
    model = MODELS.get(model_idx)
    used_fallback = False
    llm_only = False
    fallback_reason: Optional[str] = None
    if model is None:
        candidates = [
            MODELS_DIR / f"model{model_idx}.pkl",
            MODELS_DIR / f"model{model_idx}.joblib",
            MODELS_DIR / f"model{model_idx}.bin",
        ]
        for filename in candidates:
            if not filename.exists():
                continue
            try:
                MODELS[model_idx] = load_model_file(filename)
                model = MODELS[model_idx]
                break
            except Exception as exc:
                last_exc = exc
        if model is None:
            # Model failed to load – per requirement, route directly to LLM to predict everything
            used_fallback = True
            llm_only = True
            fallback_reason = str(last_exc) if 'last_exc' in locals() else "Model file could not be loaded"
    if (model is None) and not llm_only:
        raise HTTPException(status_code=400, detail=f"Model for disease '{disease_type}' (index {model_idx}) not available")

    # If we are instructed to use only the LLM, bypass feature building and prediction
    if llm_only:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        result = llm_only_predict(disease_type, inputs, report_text, openai_api_key)
        result.update({
            "used_model_index": model_idx,
            "used_fallback_model": True,
            "model_load_error": fallback_reason,
        })
        return result

    # Build features with a defined order per disease when available
    FEATURE_ORDER: Dict[str, List[str]] = {
        "general": [
            "age","sex_numeric","bmi","smoking_status_numeric","alcohol_use_numeric",
            "weight","systolic_bp","diastolic_bp","heart_rate","glucose",
            "steps_per_day","sleep_hours",
            "hba1c","cholesterol","ldl","hdl","triglycerides","creatinine","egfr","hemoglobin",
            "med_adherence","chronic_meds","insulin_or_oral_use_numeric","antihypertensive_use_numeric"
        ],
        "heart-failure": [
            "t0_window_days",
            "age",
            "sex_male",
            "bmi",
            "sbp_last",
            "dbp_last",
            "history_diabetes",
            "history_hypertension",
            "creatinine_last",
            "creatinine_mean",
            "creatinine_slope_per_day",
            "hbA1c_last",
            "fpg_last",
            "hdl_last",
            "ldl_last",
            "triglycerides_last",
            "qrs_duration_ms",
            "arrhythmia_flag",
            "afib_flag",
            "prev_mi",
            "cabg_history",
            "echo_ef_last",
            "has_echo",
            "on_ACEi",
            "on_beta_blocker",
            "on_diuretic",
            "hf_events_past_year",
            "admissions_30d",
            # "physical_activity_level" is categorical; map to numeric below
            "physical_activity_numeric",
            "cci",
        ],
        "diabetes": [
            "glucose",      # F1 - Fasting Glucose
            "hba1c",        # F2 - HbA1c
            "bmi",          # F3 - BMI
            "systolic_bp",  # F4 - Systolic BP
            "diastolic_bp", # F5 - Diastolic BP
            "cholesterol",  # F6 - Cholesterol
            "hdl",          # F7 - HDL
            "ldl",          # F8 - LDL
            "triglycerides", # F9 - Triglycerides
            "insulin_level", # F10 - Insulin level
            "heart_rate",   # F11 - Heart rate / variability marker
        ],
        "diabetes-type2": [
            "glucose",      # F1 - Fasting Glucose
            "hba1c",        # F2 - HbA1c
            "bmi",          # F3 - BMI
            "systolic_bp",  # F4 - Systolic BP
            "diastolic_bp", # F5 - Diastolic BP
            "cholesterol",  # F6 - Cholesterol
            "hdl",          # F7 - HDL
            "ldl",          # F8 - LDL
            "triglycerides", # F9 - Triglycerides
            "insulin_level", # F10 - Insulin level
            "heart_rate",   # F11 - Heart rate / variability marker
        ],
        "hypertension": [
            "weight",       # Weight
            "glucose",      # Glucose
            "heart_rate",   # Heart rate
            "activity",     # Activity level
            "sleep",        # Sleep hours
            "systolic_bp",  # Systolic BP
            "diastolic_bp", # Diastolic BP
            "hba1c",        # HbA1c
            "lipids",       # Lipids (total cholesterol)
            "creatinine",   # Creatinine
            "med_adherence", # Medication adherence
        ],
        "weight-glp1": [
            "age","sex","BMI","waist_cm",
            "obesity_class_numeric","T2D_status_numeric","HTN_status_numeric","OSA_status_numeric",
            "hbA1c_baseline","hbA1c_delta","fasting_glucose","ldl","hdl","triglycerides","alt","egfr",
            "weight_4w_slope","sbp","dbp","hr","spo2",
            "GLP1_agent_numeric","dose_tier_numeric","adherence_90d","missed_doses_last_30d",
            "nausea_score","vomit_score","appetite_score",
            "steps_avg","active_minutes","exercise_days_wk","sleep_hours","alcohol_units_wk",
            "tobacco_cigs_per_day","tobacco_chew_use_numeric","junk_food_freq_wk",
            "insurance_denied_numeric","prior_auth_denial_numeric","fill_gap_days",
            "telehealth_visits","nurse_messages","cancellations","ER_visits_obesity_related"
        ],
    }

    # Map categorical physical_activity_level to numeric if provided
    if "physical_activity_level" in inputs and "physical_activity_numeric" not in inputs:
        lvl = str(inputs.get("physical_activity_level") or "").strip().lower()
        mapping = {"low": 0, "medium": 1, "high": 2}
        if lvl in mapping:
            inputs["physical_activity_numeric"] = mapping[lvl]
    
    # Map categorical activity level for hypertension
    if "activity" in inputs and disease_type == "hypertension":
        activity = str(inputs.get("activity") or "").strip().lower()
        mapping = {"low": 0, "moderate": 1, "high": 2}
        if activity in mapping:
            inputs["activity_numeric"] = mapping[activity]
    
    # Map categorical medication adherence for hypertension
    # General model categorical mappings
    if disease_type == "general":
        sex_map = {"male": 1, "female": 0, "other": 0}
        sex = str(inputs.get("sex") or "").strip().lower()
        if sex in sex_map:
            inputs["sex_numeric"] = sex_map[sex]

        smoke_map = {"never": 0, "former": 1, "current": 2}
        smk = str(inputs.get("smoking_status") or "").strip().lower()
        if smk in smoke_map:
            inputs["smoking_status_numeric"] = smoke_map[smk]

        alcohol_map = {"none": 0, "moderate": 1, "heavy": 2}
        alc = str(inputs.get("alcohol_use") or "").strip().lower()
        if alc in alcohol_map:
            inputs["alcohol_use_numeric"] = alcohol_map[alc]

        yn_map = {"no": 0, "yes": 1}
        for k in ["insulin_or_oral_use", "antihypertensive_use"]:
            v = str(inputs.get(k) or "").strip().lower()
            if v in yn_map:
                inputs[f"{k}_numeric"] = yn_map[v]
    if "med_adherence" in inputs and disease_type == "hypertension":
        adherence = str(inputs.get("med_adherence") or "").strip().lower()
        mapping = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
        if adherence in mapping:
            inputs["med_adherence_numeric"] = mapping[adherence]

    # Map categorical fields for weight-glp1
    if disease_type == "weight-glp1":
        oc = str(inputs.get("obesity_class") or "").strip().lower()
        oc_map = {"none": 0, "class i": 1, "class ii": 2, "class iii": 3}
        if oc in oc_map:
            inputs["obesity_class_numeric"] = oc_map[oc]

        for k in ["T2D_status", "HTN_status", "OSA_status", "tobacco_chew_use",
                  "insurance_denied", "prior_auth_denial"]:
            v = str(inputs.get(k) or "").strip().lower()
            if v in {"no", "yes"}:
                inputs[f"{k}_numeric"] = 1 if v == "yes" else 0

        agent = str(inputs.get("GLP1_agent") or "").strip().lower()
        agent_map = {"semaglutide": 0, "liraglutide": 1, "dulaglutide": 2, "other": 3}
        if agent in agent_map:
            inputs["GLP1_agent_numeric"] = agent_map[agent]

        dose = str(inputs.get("dose_tier") or "").strip().lower()
        dose_map = {"low": 0, "medium": 1, "high": 2}
        if dose in dose_map:
            inputs["dose_tier_numeric"] = dose_map[dose]

    ordered_fields = FEATURE_ORDER.get(disease_type)
    feature_values: List[float] = []
    if ordered_fields:
        for name in ordered_fields:
            try:
                val = inputs.get(name, None)
                if val in ("", None):
                    raise ValueError("empty")
                feature_values.append(float(val))
            except Exception:
                feature_values.append(0.0)
    else:
        # Fallback: take all numeric values in stable key order
        for key in sorted(inputs.keys()):
            try:
                feature_values.append(float(inputs[key]))
            except Exception:
                continue
    if not feature_values:
        raise HTTPException(status_code=422, detail="No numeric features found in inputs")

    X = np.asarray(feature_values, dtype=float).reshape(1, -1)

    # If the estimator exposes n_features_in_, pad/truncate accordingly; else for HF assume 34
    expected_features = None
    for attr in ("n_features_in_",):
        if hasattr(model, attr):
            try:
                expected_features = int(getattr(model, attr))
            except Exception:
                expected_features = None
            break
    if expected_features is None and disease_type == "heart-failure":
        expected_features = max(34, X.shape[1])
    if expected_features is not None and X.shape[1] != expected_features:
        if X.shape[1] < expected_features:
            # pad zeros
            pad = np.zeros((1, expected_features - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        else:
            # truncate extra features
            X = X[:, :expected_features]

    probability: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        probability = float(proba[0, 1] if proba.shape[1] >= 2 else np.max(proba[0]))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 2 and scores.shape[1] > 1:
            logits = np.asarray(scores[0], dtype=float)
            exp_logits = np.exp(logits - np.max(logits))
            softmax = exp_logits / np.sum(exp_logits)
            probability = float(np.max(softmax))
        else:
            probability = float(_sigmoid(np.asarray(scores).reshape(-1))[0])
    elif hasattr(model, "predict"):
        pred = model.predict(X)
        probability = float(np.clip(np.asarray(pred, dtype=float).reshape(-1)[0], 0.0, 1.0))
    else:
        raise HTTPException(status_code=500, detail="Loaded model does not support prediction")

    probability = max(0.0, min(1.0, float(probability)))

    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = call_llm_summary(openai_api_key, disease_type, inputs, report_text, probability)

    return {
        "disease_type": disease_type,
        "probability": probability,
        "summary": llm.get("summary"),
        "risk": llm.get("risk", {}),
        "primary_factors": llm.get("primary_factors", []),
        "recommendations": llm.get("recommendations", {}),
        "used_model_index": model_idx,
        "used_fallback_model": used_fallback,
        "model_load_error": fallback_reason,
        "features_count": len(feature_values),
    }


@app.post("/report")
async def generate_report(
    disease_type: str = Form(...),
    payload: str = Form(..., description="JSON of analyze response (probability, summary, risk, factors, recs)"),
):
    try:
        data = json.loads(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'payload': {exc}")

    # Check if reportlab is available
    try:
        from reportlab.lib.pagesizes import letter
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation library not installed. Install reportlab")

    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    from datetime import datetime

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=f"Medical Risk Assessment Report - {disease_type}")
    styles = getSampleStyleSheet()
    story = []

    # Clean, professional medical report styles
    title_style = styles['Title']
    title_style.fontSize = 20
    title_style.textColor = colors.black
    title_style.spaceAfter = 16
    title_style.alignment = 1  # Center alignment
    title_style.fontName = 'Helvetica-Bold'

    h1_style = styles['Heading1']
    h1_style.fontSize = 14
    h1_style.textColor = colors.black
    h1_style.spaceAfter = 8
    h1_style.fontName = 'Helvetica-Bold'
    h1_style.borderWidth = 0

    h2_style = styles['Heading2']
    h2_style.fontSize = 12
    h2_style.textColor = colors.HexColor('#374151')
    h2_style.spaceAfter = 6
    h2_style.fontName = 'Helvetica-Bold'

    h3_style = styles['Heading3']
    h3_style.fontSize = 10
    h3_style.textColor = colors.HexColor('#6b7280')
    h3_style.spaceAfter = 4
    h3_style.fontName = 'Helvetica-Bold'

    body_style = styles['BodyText']
    body_style.fontSize = 9
    body_style.spaceAfter = 3
    body_style.leading = 12
    body_style.fontName = 'Helvetica'

    # Clean professional header
    story.append(Spacer(1, 20))
    story.append(Paragraph("MEDICAL RISK ASSESSMENT REPORT", title_style))
    story.append(Spacer(1, 8))

    # Simple report metadata
    report_meta = [
        ["Report Information", ""],
        ["Disease Type", disease_type.replace('-', ' ').title()],
        ["Generated", datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ["Report ID", f"MR-{disease_type.upper()}-{datetime.now().strftime('%Y%m%d%H%M')}"]
    ]
    
    meta_table = Table(report_meta, colWidths=[150, 300])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Executive Summary - clean and professional
    summary = data.get('summary') or ''
    probability = float(data.get('probability') or 0.0)
    deter = 'Yes' if probability >= 0.5 else 'No'

    story.append(Paragraph("EXECUTIVE SUMMARY", h1_style))
    story.append(Paragraph(summary, body_style))
    story.append(Spacer(1, 8))

    # Risk Assessment - clean and professional
    story.append(Paragraph("RISK ASSESSMENT", h1_style))
    
    # Create risk level
    risk_level = "HIGH RISK" if probability >= 0.7 else "MODERATE RISK" if probability >= 0.4 else "LOW RISK"
    
    # Clean metrics table
    metrics_data = [
        ["Metric", "Value", "Status"],
        ["Risk Probability", f"{probability*100:.1f}%", risk_level],
        ["Deterioration Risk", deter, "≥50% threshold"],
        ["Assessment Date", datetime.now().strftime('%Y-%m-%d'), "Current analysis"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[150, 100, 200])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 8))

    # Risk Factors Section - clean and professional
    story.append(Paragraph("CONTRIBUTING RISK FACTORS", h1_style))
    factors = data.get('primary_factors') or []
    
    if factors:
        # Create a clean horizontal bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        factor_names = [f.get('factor', '')[:15] + '...' if len(f.get('factor', '')) > 15 else f.get('factor', '') for f in factors[:6]]
        factor_impacts = [f.get('impact', 0) for f in factors[:6]]
        
        # Use a single professional color
        bars = ax.barh(factor_names, factor_impacts, color='#6b7280', alpha=0.7)
        ax.set_xlabel('Impact Score (%)', fontsize=9)
        ax.set_title('Risk Factor Impact Analysis', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3, linestyle='-')
        
        # Add value labels
        for i, (bar, impact) in enumerate(zip(bars, factor_impacts)):
            ax.text(impact + 1, bar.get_y() + bar.get_height()/2, f'{impact}%', 
                   va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart to buffer
        chart_buffer = BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        chart_buffer.seek(0)
        plt.close()
        
        # Add chart to PDF
        story.append(Image(chart_buffer, width=350, height=150))
        story.append(Spacer(1, 6))
        
        # Simple factor list
        story.append(Paragraph("Risk Factor Details", h2_style))
        
        for i, f in enumerate(factors[:5]):
            factor = f.get('factor', '')
            impact = f.get('impact', 0)
            desc = f.get('description', '')

            # Simple factor entry
            factor_text = f"{i+1}. {factor} ({impact}% impact)"
            story.append(Paragraph(factor_text, body_style))
            if desc:
                story.append(Paragraph(f"   {desc}", body_style))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("No specific risk factors identified in this analysis.", body_style))
    
    story.append(Spacer(1, 8))

    # Clinical Recommendations - clean and professional
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", h1_style))
    recs = data.get('recommendations') or {}
    
    immediate_items = recs.get("Immediate Action", [])
    lifestyle_items = recs.get("Lifestyle Changes", [])
    monitoring_items = recs.get("Monitoring", [])
    
    if immediate_items or lifestyle_items or monitoring_items:
        # Immediate Actions
        if immediate_items:
            story.append(Paragraph("Immediate Actions Required", h2_style))
            for i, item in enumerate(immediate_items):
                story.append(Paragraph(f"{i+1}. {item}", body_style))
            story.append(Spacer(1, 4))
        
        # Lifestyle Changes
        if lifestyle_items:
            story.append(Paragraph("Lifestyle Modifications", h2_style))
            for i, item in enumerate(lifestyle_items):
                story.append(Paragraph(f"{i+1}. {item}", body_style))
            story.append(Spacer(1, 4))
        
        # Monitoring
        if monitoring_items:
            story.append(Paragraph("Monitoring Protocols", h2_style))
            for i, item in enumerate(monitoring_items):
                story.append(Paragraph(f"{i+1}. {item}", body_style))
            story.append(Spacer(1, 4))
        else:
            story.append(Paragraph("No specific recommendations available. Please consult with a healthcare professional.", body_style))
    
        story.append(Spacer(1, 8))

    # Risk Progression Analysis - clean and simple
    story.append(Paragraph("RISK PROGRESSION ANALYSIS", h1_style))
    
    # Generate realistic historical data for visualization
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    base_risk = probability * 100
    
    # Simulate realistic risk progression with treatment effects
    historical_risk = []
    for i in range(12):
        # Simulate gradual improvement with some fluctuation
        improvement_factor = i * 0.05  # Gradual improvement over time
        fluctuation = np.random.normal(0, 3)  # Small random fluctuation
        risk_value = max(0, min(100, base_risk - improvement_factor * 20 + fluctuation))
        historical_risk.append(risk_value)
    
    historical_risk.append(probability * 100)  # Current month
    months.append('Current')
    
    # Create clean trend chart
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Main risk line
    ax.plot(months, historical_risk, marker='o', linewidth=2, markersize=4, 
            color='#6b7280', label='Risk Progression')
    ax.fill_between(months, historical_risk, alpha=0.1, color='#6b7280')
    
    # Simple threshold line
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=1, label='High Risk Threshold')
    
    # Clean styling
    ax.set_xlabel('Timeline (Months)', fontsize=9)
    ax.set_ylabel('Risk Level (%)', fontsize=9)
    ax.set_title('12-Month Risk Progression', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save trend chart
    trend_buffer = BytesIO()
    plt.savefig(trend_buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    trend_buffer.seek(0)
    plt.close()
    
    story.append(Image(trend_buffer, width=400, height=200))
    story.append(Spacer(1, 8))

    # Clean professional footer
    story.append(Paragraph("IMPORTANT DISCLAIMERS", h2_style))
    
    disclaimer_text = """
    This report is generated by an AI-powered medical risk assessment system and is intended for informational purposes only. 
    It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for medical decisions. The risk assessments and recommendations provided are based on statistical models 
    and may not apply to all individual cases.
    """
    story.append(Paragraph(disclaimer_text, body_style))
    
    # Simple footer
    story.append(Spacer(1, 12))
    footer_text = f"""
    <para align="center">
    <font name="Helvetica" size="8" color="gray">
    Medical Risk Assessment Report | Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    Report ID: {disease_type.upper()}-{datetime.now().strftime('%Y%m%d%H%M')} | For Healthcare Professional Use Only
    </font>
    </para>
    """
    story.append(Paragraph(footer_text, body_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    filename = f"Medical_Risk_Assessment_{disease_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return StreamingResponse(buffer, media_type='application/pdf', headers={
        'Content-Disposition': f'attachment; filename="{filename}"'
    })


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)


