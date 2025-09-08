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

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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

    # 4) xgboost native model
    if xgb is not None:
        try:
            booster = xgb.Booster()
            booster.load_model(str(filename))

            class XGBBoosterAdapter:
                def __init__(self, booster: "xgb.Booster"):
                    self.booster = booster

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    dmat = xgb.DMatrix(X)
                    preds = self.booster.predict(dmat)
                    preds = np.asarray(preds, dtype=float).reshape(-1)
                    # If output is probability for positive class (binary), build 2-column proba
                    if preds.ndim == 1:
                        p1 = np.clip(preds, 0.0, 1.0)
                        p0 = 1.0 - p1
                        return np.vstack([p0, p1]).T
                    # If already multi-class probabilities
                    return preds

                def predict(self, X: np.ndarray) -> np.ndarray:
                    proba = self.predict_proba(X)
                    # Argmax for class prediction
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

# Allow all origins by default; tighten if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR  # Expect model1.pkl..model5.pkl next to this file by default
MODELS: Dict[int, Any] = load_models(MODELS_DIR)


@app.get("/health")
def health() -> Dict[str, Any]:
    available = {i: (MODELS.get(i) is not None) for i in range(1, 6)}
    return {"status": "ok", "models": available}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Lazy-load model if missing
    model = MODELS.get(req.disease_id)
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
            detail = f"Model files not loadable for id {req.disease_id} (tried .pkl/.joblib/.bin)"
            if 'last_exc' in locals():
                detail += f": {last_exc}"
            raise HTTPException(status_code=500, detail=detail)
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
        "diabetes-type1": 2,
        "hypertension": 3,
        "heart-failure": 4,
        "weight-glp1": 5,
    }
    return mapping.get(disease_type, 1)


def _build_default_summary(disease_type: str, inputs: Dict[str, Any], probability: float) -> Dict[str, Any]:
    # Simple heuristics to ensure non-empty UI even without LLM
    bmi = float(inputs.get("bmi") or 0)
    sbp = float(inputs.get("sbp_last") or inputs.get("systolicBP") or 0)
    dbp = float(inputs.get("dbp_last") or inputs.get("diastolicBP") or 0)
    hba1c = float(inputs.get("hbA1c_last") or inputs.get("hba1c") or 0)
    creat = float(inputs.get("creatinine_last") or inputs.get("creatinine") or 0)

    factors = []
    if bmi:
        factors.append({"factor": "BMI", "impact": min(100, int(abs(bmi - 25) * 5)), "description": f"BMI {bmi:.1f}"})
    if sbp:
        factors.append({"factor": "Systolic BP", "impact": min(100, max(0, int((sbp - 120) * 2))), "description": f"SBP {sbp:.0f} mmHg"})
    if hba1c:
        factors.append({"factor": "HbA1c", "impact": min(100, max(0, int((hba1c - 5.5) * 20))), "description": f"HbA1c {hba1c:.1f}%"})
    if creat:
        factors.append({"factor": "Creatinine", "impact": min(100, max(0, int((creat - 1.0) * 40))), "description": f"Creatinine {creat:.2f} mg/dL"})
    if not factors:
        factors = [
            {"factor": "Age", "impact": 30, "description": "Demographic risk"},
            {"factor": "Vital signs", "impact": 25, "description": "Recent measurements"},
            {"factor": "Comorbidities", "impact": 20, "description": "Chronic conditions"},
        ]

    recs = {
        "Immediate Action": [
            "Schedule follow-up within 2–4 weeks",
            "Review current medications and adherence",
            "Order baseline labs and ECG if indicated",
        ],
        "Lifestyle Changes": [
            "Target 150 minutes/week moderate activity",
            "Adopt DASH-style, low-sodium diet",
            "Weight management with dietitian support",
        ],
        "Monitoring": [
            "Home BP/weight logs and symptom diary",
            "Repeat labs in 8–12 weeks",
            "Set alerts for worsening symptoms",
        ],
    }
    det = "yes" if probability >= 0.5 else "no"
    return {
        "summary": f"Disease: {disease_type}. Estimated risk {probability:.2%}.",
        "risk": {"probability": probability, "deterioration": det},
        "primary_factors": factors[:5],
        "recommendations": recs,
    }


def call_llm_summary(openai_api_key: Optional[str], disease_type: str, inputs: Dict[str, Any], report_text: Optional[str], model_probability: float) -> Dict[str, Any]:
    if not openai_api_key or OpenAI is None:
        return _build_default_summary(disease_type, inputs, model_probability)
    client = OpenAI(api_key=openai_api_key)
    sys_prompt = (
        "You are a clinical decision support assistant. Return STRICT JSON only, no prose. "
        "Use this schema: {\n"
        "  \"summary\": string,\n"
        "  \"risk\": { \"probability\": number (0-1), \"deterioration\": \"yes\"|\"no\" },\n"
        "  \"primary_factors\": [ { \"factor\": string, \"impact\": number (0-100), \"description\": string } ],\n"
        "  \"recommendations\": {\n"
        "    \"Immediate Action\": string[],\n"
        "    \"Lifestyle Changes\": string[],\n"
        "    \"Monitoring\": string[]\n"
        "  }\n"
        "}.\n"
        "- Set risk.probability to the provided model probability (do not invent).\n"
        "- Set risk.deterioration = 'yes' if probability >= 0.5 else 'no'.\n"
        "- If some inputs are missing or empty, infer reasonable generic factors and recommendations based on the disease type.\n"
        "- Always produce 3-5 primary_factors grounded on available inputs/report; if limited, use general clinical knowledge.\n"
        "- Always provide 3-5 items in each recommendations section, concise and clinical.\n"
        "Reply with only minified JSON."
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
            temperature=0.3,
            max_tokens=500,
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
            detail = f"Model files not loadable for disease '{disease_type}' (index {model_idx})"
            if 'last_exc' in locals():
                detail += f": {last_exc}"
            raise HTTPException(status_code=500, detail=detail)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Model for disease '{disease_type}' (index {model_idx}) not available")

    # Build features with a defined order per disease when available
    FEATURE_ORDER: Dict[str, List[str]] = {
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
            "pred_prob_90d",
        ],
    }

    # Map categorical physical_activity_level to numeric if provided
    if "physical_activity_level" in inputs and "physical_activity_numeric" not in inputs:
        lvl = str(inputs.get("physical_activity_level") or "").strip().lower()
        mapping = {"low": 0, "medium": 1, "high": 2}
        if lvl in mapping:
            inputs["physical_activity_numeric"] = mapping[lvl]

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

    if letter is None:
        raise HTTPException(status_code=500, detail="PDF generation library not installed. Install reportlab")

    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=f"Analysis Report - {disease_type}")
    styles = getSampleStyleSheet()
    story = []

    title = styles['Title']
    h2 = styles['Heading2']
    body = styles['BodyText']

    story.append(Paragraph(f"Analysis Report - {disease_type}", title))
    story.append(Spacer(1, 8))

    summary = data.get('summary') or ''
    probability = float(data.get('probability') or 0.0)
    deter = 'Yes' if probability >= 0.5 else 'No'

    story.append(Paragraph("Summary", h2))
    story.append(Paragraph(summary, body))
    story.append(Spacer(1, 12))

    metrics = [["Probability", f"{probability*100:.1f}%"], ["Deterioration (≥50%)", deter]]
    table = Table(metrics, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Contributing Risk Factors", h2))
    factors = data.get('primary_factors') or []
    for f in factors:
        factor = f.get('factor', '')
        impact = f.get('impact', 0)
        desc = f.get('description', '')
        story.append(Paragraph(f"- {factor} ({impact}%) — {desc}", body))
    if not factors:
        story.append(Paragraph("- None provided", body))
    story.append(Spacer(1, 12))

    recs = data.get('recommendations') or {}
    for section in ["Immediate Action", "Lifestyle Changes", "Monitoring"]:
        story.append(Paragraph(section, h2))
        items = recs.get(section) or []
        if items:
            for item in items:
                story.append(Paragraph(f"- {item}", body))
        else:
            story.append(Paragraph("- None provided", body))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    filename = f"analysis_report_{disease_type.replace(' ', '_')}.pdf"
    return StreamingResponse(buffer, media_type='application/pdf', headers={
        'Content-Disposition': f'attachment; filename="{filename}"'
    })


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)


