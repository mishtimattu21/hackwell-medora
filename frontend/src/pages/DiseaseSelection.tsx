import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import Reveal from "@/components/Reveal";
import { 
  ArrowRight 
} from "lucide-react";

const DiseaseSelection = () => {
  const diseases = [
    {
      id: "general",
      title: "General Model",
      description: "Comprehensive health risk assessment without specific disease focus",
      image: "/general.png"
    },
    {
      id: "diabetes",
      title: "Diabetes",
      description: "Type 2 diabetes risk analysis with F1-F11 parameters",
      image: "/diabetes.png"
    },
    {
      id: "hypertension",
      title: "Hypertension",
      description: "Blood pressure risk assessment and cardiovascular health",
      image: "/hypertension.png"
    },
    {
      id: "heart-failure",
      title: "Heart Failure",
      description: "Cardiac function analysis and deterioration prediction",
      image: "/heart.png"
    },
    {
      id: "weight-glp1",
      title: "Weight & GLP-1 Support",
      description: "Weight management and GLP-1 medication effectiveness tracking",
      image: "/weight.png"
    }
  ];

  return (
    <div className="min-h-screen pt-24 pb-20 fade-in">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8 slide-up">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Select Your <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-glow">Health Focus</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Choose the specific health condition you'd like to analyze, or select our general model 
            for comprehensive risk assessment across multiple conditions.
          </p>
        </div>

        {/* Disease Cards Grid */}
        <div className="max-w-5xl mx-auto">
          {/* First 3 cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            {diseases.slice(0, 3).map((disease, index) => (
              <Link key={disease.id} to={`/analysis/${disease.id}`}>
                <Reveal delayMs={index * 80}>
                <Card className={`
                  shadow-soft hover:shadow-glow transition-all duration-300 
                  border-0 frosted group cursor-pointer h-full
                  hover:scale-105 transform
                `}>
                  <CardContent className="p-4 text-center h-full flex flex-col">
                    {/* Icon */}
                    <div className="mx-auto mb-2">
                      <img src={disease.image} alt={disease.title} className="h-12 w-auto md:h-14 object-contain" />
                    </div>

                    {/* Title */}
                    <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                      {disease.title}
                    </h3>

                    {/* Description */}
                    <p className="text-sm text-muted-foreground leading-relaxed mb-4 flex-grow">
                      {disease.description}
                    </p>

                    {/* Arrow */}
                    <div className="flex justify-center">
                      <ArrowRight className="h-4 w-4 text-primary opacity-0 group-hover:opacity-100 transform translate-x-0 group-hover:translate-x-2 transition-all duration-300" />
                    </div>
                  </CardContent>
                </Card>
                </Reveal>
              </Link>
            ))}
          </div>

          {/* Last 2 cards - centered */}
          <div className="flex justify-center">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl">
              {diseases.slice(3).map((disease, index) => (
                <Link key={disease.id} to={`/analysis/${disease.id}`}>
                  <Reveal delayMs={(index + 3) * 80}>
                  <Card className={`
                    shadow-soft hover:shadow-glow transition-all duration-300 
                    border-0 frosted group cursor-pointer h-full
                    hover:scale-105 transform
                  `}>
                    <CardContent className="p-4 text-center h-full flex flex-col">
                      {/* Icon */}
                      <div className="mx-auto mb-2">
                        <img src={disease.image} alt={disease.title} className="h-12 w-auto md:h-14 object-contain" />
                      </div>

                      {/* Title */}
                      <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                        {disease.title}
                      </h3>

                      {/* Description */}
                      <p className="text-sm text-muted-foreground leading-relaxed mb-4 flex-grow">
                        {disease.description}
                      </p>

                      {/* Arrow */}
                      <div className="flex justify-center">
                        <ArrowRight className="h-4 w-4 text-primary opacity-0 group-hover:opacity-100 transform translate-x-0 group-hover:translate-x-2 transition-all duration-300" />
                      </div>
                    </CardContent>
                  </Card>
                  </Reveal>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-8">
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