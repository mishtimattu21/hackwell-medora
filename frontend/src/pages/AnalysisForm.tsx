import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Upload, FileText, Loader2, ArrowRight, Shuffle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const AnalysisForm = () => {
  const { diseaseType } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const diseaseConfig = {
    general: {
      title: "General Health Analysis",
      fields: [
        // Demographics & Baseline
        { name: "age", label: "Age (years)", type: "number", placeholder: "35" },
        { name: "sex", label: "Sex", type: "select", options: ["Male", "Female", "Other"] },
        { name: "bmi", label: "BMI (kg/m²)", type: "number", placeholder: "24.5", step: "0.1" },
        { name: "smoking_status", label: "Smoking Status", type: "select", options: ["Never", "Former", "Current"] },
        { name: "alcohol_use", label: "Alcohol Use", type: "select", options: ["None", "Moderate", "Heavy"] },

        // Vitals & Logs
        { name: "weight", label: "Weight (kg)", type: "number", placeholder: "72", step: "0.1" },
        { name: "systolic_bp", label: "Blood Pressure – Systolic (mmHg)", type: "number", placeholder: "120" },
        { name: "diastolic_bp", label: "Blood Pressure – Diastolic (mmHg)", type: "number", placeholder: "80" },
        { name: "heart_rate", label: "Heart Rate (bpm)", type: "number", placeholder: "72" },
        { name: "glucose", label: "Daily Glucose (mg/dL)", type: "number", placeholder: "100" },
        { name: "steps_per_day", label: "Activity Level (steps/day)", type: "number", placeholder: "6000" },
        { name: "sleep_hours", label: "Sleep Duration (hours/night)", type: "number", placeholder: "7.5", step: "0.1" },

        // Laboratory Values
        { name: "hba1c", label: "HbA1c (%)", type: "number", placeholder: "5.8", step: "0.1" },
        { name: "cholesterol", label: "Total Cholesterol (mg/dL)", type: "number", placeholder: "180" },
        { name: "ldl", label: "LDL (mg/dL)", type: "number", placeholder: "110" },
        { name: "hdl", label: "HDL (mg/dL)", type: "number", placeholder: "50" },
        { name: "triglycerides", label: "Triglycerides (mg/dL)", type: "number", placeholder: "150" },
        { name: "creatinine", label: "Creatinine (mg/dL)", type: "number", placeholder: "1.0", step: "0.1" },
        { name: "egfr", label: "eGFR (ml/min/1.73 m²)", type: "number", placeholder: "95" },
        { name: "hemoglobin", label: "Hemoglobin (g/dL)", type: "number", placeholder: "14.0", step: "0.1" },

        // Medications
        { name: "med_adherence", label: "Medication Adherence (%)", type: "number", placeholder: "85" },
        { name: "chronic_meds", label: "Number of Prescribed Chronic Meds", type: "number", placeholder: "2" },
        { name: "insulin_or_oral_use", label: "Insulin/Oral Hypoglycemic Use", type: "select", options: ["No", "Yes"] },
        { name: "antihypertensive_use", label: "Antihypertensive Use", type: "select", options: ["No", "Yes"] },
      ]
    },
    "diabetes": {
      title: "Diabetes Analysis",
      fields: [
        { name: "glucose", label: "Fasting Glucose (mg/dL)", type: "number", placeholder: "100" },
        { name: "hba1c", label: "HbA1c (%)", type: "number", placeholder: "7.0", step: "0.1" },
        { name: "bmi", label: "BMI (kg/m²)", type: "number", placeholder: "22.5", step: "0.1" },
        { name: "systolic_bp", label: "Systolic BP (mmHg)", type: "number", placeholder: "120" },
        { name: "diastolic_bp", label: "Diastolic BP (mmHg)", type: "number", placeholder: "80" },
        { name: "cholesterol", label: "Cholesterol (mg/dL)", type: "number", placeholder: "180" },
        { name: "hdl", label: "HDL (mg/dL)", type: "number", placeholder: "45" },
        { name: "ldl", label: "LDL (mg/dL)", type: "number", placeholder: "120" },
        { name: "triglycerides", label: "Triglycerides (mg/dL)", type: "number", placeholder: "150" },
        { name: "insulin_level", label: "Insulin Level (μU/mL)", type: "number", placeholder: "15", step: "0.1" },
        { name: "heart_rate", label: "Heart Rate (bpm)", type: "number", placeholder: "72" },
      ]
    },
    
    hypertension: {
      title: "Hypertension Analysis",
      fields: [
        { name: "weight", label: "Weight (kg)", type: "number", placeholder: "75", step: "0.1" },
        { name: "glucose", label: "Glucose (mg/dL)", type: "number", placeholder: "100" },
        { name: "heart_rate", label: "Heart Rate (bpm)", type: "number", placeholder: "72" },
        { name: "activity", label: "Activity Level", type: "select", options: ["Low", "Moderate", "High"] },
        { name: "sleep", label: "Sleep Hours/Night", type: "number", placeholder: "7", step: "0.1" },
        { name: "systolic_bp", label: "Systolic BP (mmHg)", type: "number", placeholder: "140" },
        { name: "diastolic_bp", label: "Diastolic BP (mmHg)", type: "number", placeholder: "90" },
        { name: "hba1c", label: "HbA1c (%)", type: "number", placeholder: "6.5", step: "0.1" },
        { name: "lipids", label: "Total Lipids (mg/dL)", type: "number", placeholder: "200" },
        { name: "creatinine", label: "Creatinine (mg/dL)", type: "number", placeholder: "1.0", step: "0.1" },
        { name: "med_adherence", label: "Medication Adherence", type: "select", options: ["Poor", "Fair", "Good", "Excellent"] },
      ]
    },
    "heart-failure": {
      title: "Heart Failure Analysis",
      fields: [
        { name: "t0_window_days", label: "Time Window (days)", type: "number", placeholder: "90" },
        { name: "age", label: "Age", type: "number", placeholder: "65" },
        { name: "sex_male", label: "Sex (1=Male, 0=Female)", type: "number", placeholder: "1" },
        { name: "bmi", label: "BMI (kg/m²)", type: "number", placeholder: "27.5", step: "0.1" },
        { name: "sbp_last", label: "Last Systolic BP (mmHg)", type: "number", placeholder: "120" },
        { name: "dbp_last", label: "Last Diastolic BP (mmHg)", type: "number", placeholder: "80" },
        { name: "history_diabetes", label: "History of Diabetes (0/1)", type: "number", placeholder: "0" },
        { name: "history_hypertension", label: "History of Hypertension (0/1)", type: "number", placeholder: "1" },
        { name: "creatinine_last", label: "Creatinine Last (mg/dL)", type: "number", placeholder: "1.2", step: "0.1" },
        { name: "creatinine_mean", label: "Creatinine Mean (mg/dL)", type: "number", placeholder: "1.1", step: "0.1" },
        { name: "creatinine_slope_per_day", label: "Creatinine Slope per Day", type: "number", placeholder: "0.0", step: "0.001" },
        { name: "hbA1c_last", label: "HbA1c Last (%)", type: "number", placeholder: "6.5", step: "0.1" },
        { name: "fpg_last", label: "Fasting Plasma Glucose (mg/dL)", type: "number", placeholder: "110" },
        { name: "hdl_last", label: "HDL (mg/dL)", type: "number", placeholder: "45" },
        { name: "ldl_last", label: "LDL (mg/dL)", type: "number", placeholder: "120" },
        { name: "triglycerides_last", label: "Triglycerides (mg/dL)", type: "number", placeholder: "150" },
        { name: "qrs_duration_ms", label: "QRS Duration (ms)", type: "number", placeholder: "100" },
        { name: "arrhythmia_flag", label: "Arrhythmia Present (0/1)", type: "number", placeholder: "0" },
        { name: "afib_flag", label: "Atrial Fibrillation (0/1)", type: "number", placeholder: "0" },
        { name: "prev_mi", label: "Previous MI (0/1)", type: "number", placeholder: "0" },
        { name: "cabg_history", label: "CABG History (0/1)", type: "number", placeholder: "0" },
        { name: "echo_ef_last", label: "Echo EF Last (%)", type: "number", placeholder: "45", step: "0.1" },
        { name: "has_echo", label: "Echo Available (0/1)", type: "number", placeholder: "1" },
        { name: "on_ACEi", label: "On ACE Inhibitor (0/1)", type: "number", placeholder: "1" },
        { name: "on_beta_blocker", label: "On Beta-blocker (0/1)", type: "number", placeholder: "1" },
        { name: "on_diuretic", label: "On Diuretic (0/1)", type: "number", placeholder: "1" },
        { name: "hf_events_past_year", label: "HF Admissions Past Year", type: "number", placeholder: "0" },
        { name: "admissions_30d", label: "Admissions Last 30 Days", type: "number", placeholder: "0" },
        { name: "physical_activity_level", label: "Physical Activity Level", type: "select", options: ["Low", "Medium", "High"] },
        { name: "physical_activity_numeric", label: "Physical Activity (0/1/2)", type: "number", placeholder: "1" },
        { name: "cci", label: "Charlson Comorbidity Index", type: "number", placeholder: "3" }
      ]
    },
    "weight-glp1": {
      title: "Weight & GLP-1 Analysis",
      fields: [
        { name: "age", label: "Age", type: "number", placeholder: "45" },
        { name: "sex", label: "Sex (0=Female,1=Male)", type: "number", placeholder: "1" },
        { name: "BMI", label: "BMI (kg/m²)", type: "number", placeholder: "32.5", step: "0.1" },
        { name: "waist_cm", label: "Waist (cm)", type: "number", placeholder: "100" },
        { name: "obesity_class", label: "Obesity Class", type: "select", options: ["None", "Class I", "Class II", "Class III"] },
        { name: "T2D_status", label: "Type 2 Diabetes", type: "select", options: ["No", "Yes"] },
        { name: "HTN_status", label: "Hypertension", type: "select", options: ["No", "Yes"] },
        { name: "OSA_status", label: "Sleep Apnea (OSA)", type: "select", options: ["No", "Yes"] },
        { name: "hbA1c_baseline", label: "Baseline HbA1c (%)", type: "number", placeholder: "6.8", step: "0.1" },
        { name: "hbA1c_delta", label: "HbA1c Change (%)", type: "number", placeholder: "-0.6", step: "0.1" },
        { name: "fasting_glucose", label: "Fasting Glucose (mg/dL)", type: "number", placeholder: "110" },
        { name: "ldl", label: "LDL (mg/dL)", type: "number", placeholder: "120" },
        { name: "hdl", label: "HDL (mg/dL)", type: "number", placeholder: "45" },
        { name: "triglycerides", label: "Triglycerides (mg/dL)", type: "number", placeholder: "180" },
        { name: "alt", label: "ALT (U/L)", type: "number", placeholder: "28" },
        { name: "egfr", label: "eGFR (mL/min/1.73m²)", type: "number", placeholder: "90" },
        { name: "weight_4w_slope", label: "Weight 4-week Slope (kg/wk)", type: "number", placeholder: "-0.6", step: "0.1" },
        { name: "sbp", label: "Systolic BP (mmHg)", type: "number", placeholder: "130" },
        { name: "dbp", label: "Diastolic BP (mmHg)", type: "number", placeholder: "85" },
        { name: "hr", label: "Heart Rate (bpm)", type: "number", placeholder: "72" },
        { name: "spo2", label: "SpO₂ (%)", type: "number", placeholder: "98", step: "0.1" },
        { name: "GLP1_agent", label: "GLP-1 Agent", type: "select", options: ["Semaglutide", "Liraglutide", "Dulaglutide", "Other"] },
        { name: "dose_tier", label: "Dose Tier", type: "select", options: ["Low", "Medium", "High"] },
        { name: "adherence_90d", label: "Adherence (90d, %)", type: "number", placeholder: "85", step: "1" },
        { name: "missed_doses_last_30d", label: "Missed Doses (last 30d)", type: "number", placeholder: "1" },
        { name: "nausea_score", label: "Nausea Score (0-10)", type: "number", placeholder: "2", step: "1" },
        { name: "vomit_score", label: "Vomiting Score (0-10)", type: "number", placeholder: "1", step: "1" },
        { name: "appetite_score", label: "Appetite Score (0-10)", type: "number", placeholder: "3", step: "1" },
        { name: "steps_avg", label: "Daily Steps (avg)", type: "number", placeholder: "6000" },
        { name: "active_minutes", label: "Active Minutes/Day", type: "number", placeholder: "40" },
        { name: "exercise_days_wk", label: "Exercise Days/Week", type: "number", placeholder: "4" },
        { name: "sleep_hours", label: "Sleep Hours/Night", type: "number", placeholder: "7.5", step: "0.1" },
        { name: "alcohol_units_wk", label: "Alcohol Units/Week", type: "number", placeholder: "2" },
        { name: "tobacco_cigs_per_day", label: "Cigarettes/Day", type: "number", placeholder: "0" },
        { name: "tobacco_chew_use", label: "Tobacco Chew Use", type: "select", options: ["No", "Yes"] },
        { name: "junk_food_freq_wk", label: "Junk Food Meals/Week", type: "number", placeholder: "2" },
        { name: "insurance_denied", label: "Insurance Denied", type: "select", options: ["No", "Yes"] },
        { name: "prior_auth_denial", label: "Prior Auth Denial", type: "select", options: ["No", "Yes"] },
        { name: "fill_gap_days", label: "Rx Fill Gap (days)", type: "number", placeholder: "0" },
        { name: "telehealth_visits", label: "Telehealth Visits (90d)", type: "number", placeholder: "1" },
        { name: "nurse_messages", label: "Nurse Messages (90d)", type: "number", placeholder: "2" },
        { name: "cancellations", label: "Appointment Cancellations (90d)", type: "number", placeholder: "0" },
        { name: "ER_visits_obesity_related", label: "ER Visits (obesity-related)", type: "number", placeholder: "0" }
      ]
    }
  };

  const config = diseaseConfig[diseaseType as keyof typeof diseaseConfig] || diseaseConfig.general;

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      toast({
        title: "File uploaded successfully",
        description: `${file.name} has been uploaded.`,
      });
    }
  };

  const handleRandomData = async () => {
    setIsLoading(true);
    try {
      const RAW_BASE = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
      const API_BASE = String(RAW_BASE).replace(/\/*$/, "");
      const res = await fetch(`${API_BASE}/random-data/${diseaseType}`, { method: "GET" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to generate random data");
      }
      const result = await res.json();
      
      // Fill the form with random data
      const form = document.querySelector('form') as HTMLFormElement;
      if (form) {
        console.log('Filling form with random data for', diseaseType, ':', result.data);
        Object.entries(result.data).forEach(([key, value]) => {
          const input = form.querySelector(`[name="${key}"]`) as HTMLInputElement | HTMLSelectElement;
          if (input) {
            if (input.tagName === 'SELECT') {
              // For select elements, find the option with matching value
              const select = input as HTMLSelectElement;
              const option = Array.from(select.options).find(opt => opt.value === String(value));
              if (option) {
                select.value = String(value);
                // Trigger change event for React
                select.dispatchEvent(new Event('change', { bubbles: true }));
                select.dispatchEvent(new Event('input', { bubbles: true }));
                console.log(`Set select ${key} to ${value}`);
              } else {
                console.warn(`No option found for ${key} with value ${value}`);
              }
            } else {
              // For input elements
              (input as HTMLInputElement).value = String(value);
              // Trigger change event for React
              input.dispatchEvent(new Event('change', { bubbles: true }));
              input.dispatchEvent(new Event('input', { bubbles: true }));
              console.log(`Set input ${key} to ${value}`);
            }
          } else {
            console.warn(`No input found for field ${key}`);
          }
        });
      }
      
      toast({
        title: "Random data generated",
        description: `Random ${diseaseType} data has been filled in the form.`,
      });
    } catch (err: any) {
      toast({ 
        title: "Random data generation failed", 
        description: String(err?.message || err), 
        variant: "destructive" 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    try {
      const formEl = event.currentTarget as HTMLFormElement;
      const formData = new FormData(formEl);
      const config = diseaseConfig[diseaseType as keyof typeof diseaseConfig] || diseaseConfig.general;

      // Collect values from the form based on configured fields
      const values: Record<string, any> = {};
      config.fields.forEach((field: any) => {
        const raw = formData.get(field.name);
        if (raw === null || raw === "") return;
        if (field.type === "number") {
          const num = Number(raw);
          values[field.name] = Number.isNaN(num) ? raw : num;
        } else {
          values[field.name] = raw;
        }
      });

      // Attach file and disease_type
      const payload = new FormData();
      payload.append("disease_type", diseaseType || "general");
      payload.append("data", JSON.stringify(values));
      if (uploadedFile) {
        payload.append("file", uploadedFile);
      }

      const RAW_BASE = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
      const API_BASE = String(RAW_BASE).replace(/\/*$/, ""); // trim trailing slashes
      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: payload });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to analyze");
      }
      const result = await res.json();
      try {
        sessionStorage.setItem("analysis_result", JSON.stringify(result));
      } catch {}
      toast({ title: "Analysis complete", description: `Probability ${(result?.probability * 100).toFixed(1)}%` });

      navigate(`/results/${diseaseType}`);
    } catch (err: any) {
      toast({ title: "Submission failed", description: String(err?.message || err), variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-20">
      <div className="container mx-auto px-4 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            {config.title}
          </h1>
          <p className="text-xl text-muted-foreground">
            Upload your medical reports and fill in the required parameters for analysis
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* File Upload Section */}
          <Card className="shadow-soft border-0 frosted">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5 text-primary" />
                Upload Medical Reports
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  accept=".pdf,.doc,.docx,.jpg,.jpeg,.png"
                  onChange={handleFileUpload}
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium text-foreground mb-2">
                    {uploadedFile ? uploadedFile.name : "Click to upload medical reports"}
                  </p>
                  <p className="text-muted-foreground">
                    Supports PDF, Word documents, and images
                  </p>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Parameters Section */}
          <Card className="shadow-soft border-0 frosted">
            <CardHeader>
              <CardTitle>Health Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {config.fields.map((field) => (
                  <div key={field.name} className="space-y-2">
                    <Label htmlFor={field.name}>{field.label}</Label>
                    {field.type === "select" ? (
                      <select
                        id={field.name}
                        name={field.name}
                        className="flex h-10 w-full rounded-md border border-input bg-background/50 px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                      >
                        <option value="">Select...</option>
                        {field.options?.map((option) => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    ) : (
                      <Input
                        id={field.name}
                        name={field.name}
                        type={field.type}
                        placeholder={field.placeholder}
                        step={field.step}
                        className="bg-background/50"
                      />
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Additional Notes */}
          <Card className="shadow-soft border-0 frosted">
            <CardHeader>
              <CardTitle>Additional Notes (Optional)</CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Any additional symptoms, medications, or relevant medical history..."
                className="min-h-[100px] bg-background/50"
              />
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="text-center space-y-4">
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button
                type="button"
                variant="outline"
                size="lg"
                onClick={handleRandomData}
                disabled={isLoading}
                className="px-8 py-6 text-lg border-primary/20 hover:bg-primary/5"
              >
                <Shuffle className="h-5 w-5 mr-2" />
                Generate Random Data
              </Button>
              
            <Button
              type="submit"
              size="lg"
              disabled={isLoading}
              className="medical-gradient text-white hover:shadow-glow transition-all duration-300 px-12 py-6 text-lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin mr-2" />
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Now
                  <ArrowRight className="h-5 w-5 ml-2" />
                </>
              )}
            </Button>
            </div>
            
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              Use "Generate Random Data" to quickly test the analysis with realistic sample data, 
              or fill in your own health parameters for a personalized assessment.
            </p>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AnalysisForm;