import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Upload, FileText, Loader2, ArrowRight } from "lucide-react";
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
        { name: "age", label: "Age", type: "number", placeholder: "30" },
        { name: "weight", label: "Weight (kg)", type: "number", placeholder: "70" },
        { name: "height", label: "Height (cm)", type: "number", placeholder: "175" },
        { name: "bloodPressureSys", label: "Systolic BP (mmHg)", type: "number", placeholder: "120" },
        { name: "bloodPressureDia", label: "Diastolic BP (mmHg)", type: "number", placeholder: "80" },
        { name: "cholesterol", label: "Total Cholesterol (mg/dL)", type: "number", placeholder: "180" },
      ]
    },
    "diabetes-type1": {
      title: "Type-1 Diabetes Analysis",
      fields: [
        { name: "age", label: "Age", type: "number", placeholder: "25" },
        { name: "hba1c", label: "HbA1c (%)", type: "number", placeholder: "7.0", step: "0.1" },
        { name: "bmi", label: "BMI", type: "number", placeholder: "22.5", step: "0.1" },
        { name: "insulinDose", label: "Daily Insulin (units)", type: "number", placeholder: "40" },
        { name: "glucoseFasting", label: "Fasting Glucose (mg/dL)", type: "number", placeholder: "100" },
        { name: "cPeptide", label: "C-Peptide (ng/mL)", type: "number", placeholder: "0.1", step: "0.01" },
      ]
    },
    "diabetes-type2": {
      title: "Type-2 Diabetes Analysis", 
      fields: [
        { name: "age", label: "Age", type: "number", placeholder: "45" },
        { name: "hba1c", label: "HbA1c (%)", type: "number", placeholder: "6.8", step: "0.1" },
        { name: "bmi", label: "BMI", type: "number", placeholder: "28.5", step: "0.1" },
        { name: "glucoseFasting", label: "Fasting Glucose (mg/dL)", type: "number", placeholder: "110" },
        { name: "triglycerides", label: "Triglycerides (mg/dL)", type: "number", placeholder: "150" },
        { name: "hdl", label: "HDL Cholesterol (mg/dL)", type: "number", placeholder: "45" },
      ]
    },
    prediabetes: {
      title: "Prediabetes Analysis",
      fields: [
        { name: "age", label: "Age", type: "number", placeholder: "40" },
        { name: "bmi", label: "BMI", type: "number", placeholder: "26.0", step: "0.1" },
        { name: "glucoseFasting", label: "Fasting Glucose (mg/dL)", type: "number", placeholder: "105" },
        { name: "hba1c", label: "HbA1c (%)", type: "number", placeholder: "5.8", step: "0.1" },
        { name: "waistCircumference", label: "Waist Circumference (cm)", type: "number", placeholder: "90" },
        { name: "familyHistory", label: "Family History of Diabetes", type: "select", options: ["No", "Yes"] },
      ]
    },
    hypertension: {
      title: "Hypertension Analysis",
      fields: [
        { name: "age", label: "Age", type: "number", placeholder: "50" },
        { name: "systolicBP", label: "Systolic BP (mmHg)", type: "number", placeholder: "140" },
        { name: "diastolicBP", label: "Diastolic BP (mmHg)", type: "number", placeholder: "90" },
        { name: "bmi", label: "BMI", type: "number", placeholder: "27.0", step: "0.1" },
        { name: "sodium", label: "Daily Sodium Intake (mg)", type: "number", placeholder: "2300" },
        { name: "exerciseHours", label: "Exercise Hours/Week", type: "number", placeholder: "3" },
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
        { name: "cci", label: "Charlson Comorbidity Index", type: "number", placeholder: "3" },
        { name: "pred_prob_90d", label: "Predicted Prob (90d)", type: "number", placeholder: "0.25", step: "0.01" }
      ]
    },
    "weight-glp1": {
      title: "Weight & GLP-1 Analysis",
      fields: [
        { name: "age", label: "Age", type: "number", placeholder: "35" },
        { name: "currentWeight", label: "Current Weight (kg)", type: "number", placeholder: "85" },
        { name: "height", label: "Height (cm)", type: "number", placeholder: "170" },
        { name: "startingWeight", label: "Weight Before GLP-1 (kg)", type: "number", placeholder: "95" },
        { name: "glp1Duration", label: "Months on GLP-1", type: "number", placeholder: "6" },
        { name: "glp1Type", label: "GLP-1 Medication", type: "select", options: ["Semaglutide", "Liraglutide", "Dulaglutide", "Other"] },
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

          {/* Submit Button */}
          <div className="text-center">
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
        </form>
      </div>
    </div>
  );
};

export default AnalysisForm;