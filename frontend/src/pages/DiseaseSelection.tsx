import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import Reveal from "@/components/Reveal";
import { 
  Activity, 
  Droplets, 
  Heart, 
  TrendingDown, 
  Weight, 
  ArrowRight 
} from "lucide-react";

const DiseaseSelection = () => {
  const diseases = [
    {
      id: "general",
      title: "General Model",
      description: "Comprehensive health risk assessment without specific disease focus",
      icon: Activity,
      color: "bg-blue-500/10 hover:bg-blue-500/20",
      iconColor: "text-blue-500"
    },
    {
      id: "diabetes-type1",
      title: "Diabetes",
      description: "Insulin-dependent diabetes risk analysis and management insights",
      icon: Droplets,
      color: "bg-red-500/10 hover:bg-red-500/20",
      iconColor: "text-red-500"
    },
    {
      id: "hypertension",
      title: "Hypertension",
      description: "Blood pressure risk assessment and cardiovascular health",
      icon: Heart,
      color: "bg-purple-500/10 hover:bg-purple-500/20",
      iconColor: "text-purple-500"
    },
    {
      id: "heart-failure",
      title: "Heart Failure",
      description: "Cardiac function analysis and deterioration prediction",
      icon: Heart,
      color: "bg-pink-500/10 hover:bg-pink-500/20",
      iconColor: "text-pink-500"
    },
    {
      id: "weight-glp1",
      title: "Weight & GLP-1 Support",
      description: "Weight management and GLP-1 medication effectiveness tracking",
      icon: Weight,
      color: "bg-green-500/10 hover:bg-green-500/20",
      iconColor: "text-green-500"
    }
  ];

  return (
    <div className="min-h-screen pt-24 pb-20 fade-in">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-16 slide-up">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
            Select Your Health Focus
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Choose the specific health condition you'd like to analyze, or select our general model 
            for comprehensive risk assessment across multiple conditions.
          </p>
        </div>

        {/* Disease Cards Grid - 3 per row on md+ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {diseases.map((disease, index) => (
            <Link key={disease.id} to={`/analysis/${disease.id}`}>
              <Reveal delayMs={index * 80}>
              <Card className={`
                shadow-soft hover:shadow-glow transition-all duration-300 
                border-0 frosted group cursor-pointer h-full
                hover:scale-105 transform
              `}>
                <CardContent className="p-8 text-center h-full flex flex-col">
                  {/* Icon */}
                  <div className={`
                    rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6
                    ${disease.color} transition-colors duration-300
                  `}>
                    <disease.icon className={`h-10 w-10 ${disease.iconColor}`} />
                  </div>

                  {/* Title */}
                  <h3 className="text-xl font-semibold text-foreground mb-3 group-hover:text-primary transition-colors">
                    {disease.title}
                  </h3>

                  {/* Description */}
                  <p className="text-muted-foreground leading-relaxed mb-6 flex-grow">
                    {disease.description}
                  </p>

                  {/* Arrow */}
                  <div className="flex justify-center">
                    <ArrowRight className="h-5 w-5 text-primary opacity-0 group-hover:opacity-100 transform translate-x-0 group-hover:translate-x-2 transition-all duration-300" />
                  </div>
                </CardContent>
              </Card>
              </Reveal>
            </Link>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-16">
          <p className="text-muted-foreground mb-4">
            Not sure which model to choose?
          </p>
          <Link 
            to="/analysis/general" 
            className="inline-flex items-center text-primary hover:text-primary-glow transition-colors font-medium"
          >
            Start with General Model
            <ArrowRight className="h-4 w-4 ml-2" />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default DiseaseSelection;