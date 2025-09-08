import { useParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  AlertTriangle, 
  TrendingUp, 
  Heart, 
  Activity,
  Download,
  Share,
  Calendar,
  Target,
  Lightbulb,
  Shield
} from "lucide-react";

const ResultsDashboard = () => {
  const { diseaseType } = useParams();

  // Mock data - in real app this would come from API
  const results = {
    riskScore: 67,
    deteriorationRisk: 23,
    timeframe: "90 days",
    confidence: 89,
    primaryFactors: [
      { factor: "HbA1c Level", impact: 85, description: "Above target range" },
      { factor: "BMI", impact: 72, description: "Overweight category" },
      { factor: "Age", impact: 45, description: "Moderate risk factor" },
      { factor: "Blood Pressure", impact: 38, description: "Slightly elevated" }
    ],
    recommendations: [
      {
        category: "Immediate Action",
        items: [
          "Schedule follow-up with endocrinologist within 2 weeks",
          "Begin continuous glucose monitoring",
          "Review current medication dosages"
        ]
      },
      {
        category: "Lifestyle Changes", 
        items: [
          "Reduce carbohydrate intake by 20-30%",
          "Increase physical activity to 150 minutes/week",
          "Implement stress management techniques"
        ]
      },
      {
        category: "Monitoring",
        items: [
          "Check blood glucose 4x daily",
          "Weekly weight measurements",
          "Monthly HbA1c testing"
        ]
      }
    ]
  };

  const getRiskLevel = (score: number) => {
    if (score >= 70) return { level: "High", color: "text-red-500", bg: "bg-red-500/10" };
    if (score >= 40) return { level: "Moderate", color: "text-yellow-500", bg: "bg-yellow-500/10" };
    return { level: "Low", color: "text-green-500", bg: "bg-green-500/10" };
  };

  const riskLevel = getRiskLevel(results.riskScore);
  const deteriorationLevel = getRiskLevel(results.deteriorationRisk);

  return (
    <div className="min-h-screen pt-24 pb-20">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Header */}
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-foreground mb-2">
              Analysis Results
            </h1>
            <p className="text-muted-foreground">
              Generated on {new Date().toLocaleDateString()} • Disease Type: {diseaseType}
            </p>
          </div>
          <div className="flex gap-3 mt-4 lg:mt-0">
            <Button variant="outline" className="flex items-center gap-2">
              <Share className="h-4 w-4" />
              Share
            </Button>
            <Button className="medical-gradient text-white flex items-center gap-2">
              <Download className="h-4 w-4" />
              Download Report
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Risk Scores */}
          <div className="lg:col-span-2 space-y-6">
            {/* Risk Score Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="shadow-soft border-0 frosted">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <AlertTriangle className="h-5 w-5 text-primary" />
                    Overall Risk Score
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-end gap-2">
                      <span className="text-4xl font-bold text-foreground">{results.riskScore}%</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${riskLevel.bg} ${riskLevel.color}`}>
                        {riskLevel.level} Risk
                      </span>
                    </div>
                    <Progress value={results.riskScore} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      Confidence: {results.confidence}%
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-soft border-0 frosted">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    Deterioration Risk
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-end gap-2">
                      <span className="text-4xl font-bold text-foreground">{results.deteriorationRisk}%</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${deteriorationLevel.bg} ${deteriorationLevel.color}`}>
                        {deteriorationLevel.level} Risk
                      </span>
                    </div>
                    <Progress value={results.deteriorationRisk} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      Within {results.timeframe}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Risk Factors Chart */}
            <Card className="shadow-soft border-0 frosted">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Contributing Risk Factors
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.primaryFactors.map((factor, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium text-foreground">{factor.factor}</span>
                        <span className="text-sm font-medium text-primary">{factor.impact}%</span>
                      </div>
                      <Progress value={factor.impact} className="h-2" />
                      <p className="text-xs text-muted-foreground">{factor.description}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Risk Progression Chart Placeholder */}
            <Card className="shadow-soft border-0 frosted">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Heart className="h-5 w-5 text-primary" />
                  Risk Progression
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <TrendingUp className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                    <p className="text-muted-foreground">Interactive chart showing risk trends over time</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations Sidebar */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <Card className="shadow-soft border-0 frosted">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Target className="h-5 w-5 text-primary" />
                  Quick Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  <Calendar className="h-4 w-4 mr-2" />
                  Schedule Check-up
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Shield className="h-4 w-4 mr-2" />
                  Update Insurance
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Get Lab Orders
                </Button>
              </CardContent>
            </Card>

            {/* Recommendations */}
            {results.recommendations.map((section, index) => (
              <Card key={index} className="shadow-soft border-0 frosted">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Lightbulb className="h-5 w-5 text-primary" />
                    {section.category}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {section.items.map((item, itemIndex) => (
                      <li key={itemIndex} className="flex items-start gap-2">
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                        <span className="text-sm text-foreground leading-relaxed">{item}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}

            {/* Clinical Notes */}
            <Card className="shadow-soft border-0 frosted">
              <CardHeader>
                <CardTitle className="text-lg">Clinical Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="p-3 bg-muted/30 rounded-lg">
                    <h4 className="font-medium text-foreground mb-1">For Healthcare Provider:</h4>
                    <p className="text-sm text-muted-foreground">
                      Patient shows elevated risk markers requiring immediate attention. 
                      Consider adjustment of current treatment protocol.
                    </p>
                  </div>
                  <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <h4 className="font-medium text-foreground mb-1">Patient Education:</h4>
                    <p className="text-sm text-muted-foreground">
                      Focus on medication adherence and lifestyle modifications. 
                      Provide diabetes education resources.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;