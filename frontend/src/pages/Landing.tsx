import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import Reveal from "@/components/Reveal";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { supabase } from "@/lib/supabaseClient";
import { 
  Upload, 
  FileText, 
  BarChart3, 
  Shield, 
  Users, 
  TrendingUp,
  Heart,
  Activity,
  Star,
  ArrowRight
} from "lucide-react";

const Landing = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section 
        className="relative min-h-screen flex items-center overflow-hidden bg-background fade-in"
        style={{
          backgroundImage: "url(/bg-2.jpg)",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat"
        }}
      >
        <div className="absolute inset-0 bg-background/70" />
        
        <div className="relative z-10 container mx-auto px-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left: Copy */}
            <div className="space-y-8 text-left slide-up">
              

              <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-glow leading-tight">
                AI-Powered Health Risk Prediction
              </h1>
              
              <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl leading-relaxed">
                Analyze chronic disease risks with patient-friendly insights. 
                Get accurate predictions and personalized recommendations in minutes.
              </p>

              <div className="flex flex-col sm:flex-row gap-4">
                <Button 
                  asChild
                  size="lg"
                  className="medical-gradient text-white hover:shadow-glow transition-all duration-300 text-lg px-8 py-6"
                >
                  <Link to="/disease-selection" className="flex items-center gap-2">
                    Analyze Probability & Deterioration
                    <ArrowRight className="h-5 w-5" />
                  </Link>
                </Button>
                
                <Button 
                  variant="outline" 
                  size="lg"
                  className="text-lg px-8 py-6 hover:bg-primary/5 border-primary/20"
                >
                  Watch Demo
                </Button>
              </div>
            </div>

            {/* Right: Hero image from public */}
            <div className="relative slide-in flex justify-end pr-9 md:pr-8 lg:pr-12">
              <div className="relative w-full max-w-xl lg:max-w-2xl ml-auto">
                <div className="aspect-[4/3] flex items-center justify-center">
                  <img src="/hero-1.png" alt="Medical hero" className="w-full h-full object-contain" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-foreground mb-4">How it <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-glow">Works</span></h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Simple 3-step process to get comprehensive health risk analysis
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              {
                icon: Upload,
                title: "Upload Report",
                description: "Upload your medical reports, lab results, or input key health parameters"
              },
              {
                icon: FileText,
                title: "Fill Parameters",
                description: "Complete any missing health information for accurate analysis"
              },
              {
                icon: BarChart3,
                title: "Get Predictions",
                description: "Receive detailed risk analysis with actionable recommendations"
              }
            ].map((item, index) => (
              <Reveal key={index} delayMs={index * 120}>
                <Card className="relative overflow-hidden shadow-soft hover:shadow-glow transition-all duration-300 border-0 frosted h-full">
                  <CardContent className="p-8 text-center h-full flex flex-col">
                    <div className="relative mb-6">
                      <div className="bg-primary/10 rounded-full w-16 h-16 flex items-center justify-center mx-auto">
                        <item.icon className="h-8 w-8 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-xl font-semibold text-foreground mb-3">{item.title}</h3>
                    <p className="text-muted-foreground leading-relaxed flex-1">{item.description}</p>
                  </CardContent>
                </Card>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-muted/60 dark:bg-muted/40">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-foreground mb-4">Platform <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-glow">Features</span></h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Comprehensive health analysis tools designed for patients and clinicians
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
            {[
              {
                icon: Heart,
                title: "Multi-disease Prediction",
                description: "Advanced models for diabetes, hypertension, heart failure, and more"
              },
              {
                icon: BarChart3,
                title: "Risk Scoring & Probabilities",
                description: "Clear probability scores with confidence intervals"
              },
              {
                icon: Users,
                title: "Clinician-friendly Insights", 
                description: "Professional reports formatted for healthcare providers"
              },
              {
                icon: FileText,
                title: "Patient Recommendations",
                description: "Personalized lifestyle and treatment suggestions"
              },
              {
                icon: TrendingUp,
                title: "Dashboard with Charts",
                description: "Interactive visualizations of health trends and risks"
              },
              {
                icon: Shield,
                title: "Secure & Private",
                description: "HIPAA-compliant data handling and encryption"
              }
            ].map((feature, index) => (
              <Reveal key={index} delayMs={index * 100}>
                <Card className="shadow-soft hover:shadow-glow transition-all duration-300 border-0 frosted group h-full">
                  <CardContent className="p-6 h-full flex flex-col">
                  <div className="bg-primary/10 rounded-lg w-12 h-12 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground text-sm leading-relaxed flex-1">{feature.description}</p>
                  </CardContent>
                </Card>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="feedback" className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-foreground mb-4">What Our Users <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-glow">Say</span></h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Trusted by healthcare professionals and patients worldwide
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {[
              {
                name: "Dr. Sarah Johnson",
                role: "Cardiologist",
                content: "MedPredict has transformed how I assess patient risk. The accuracy and clarity of reports save significant time in consultations.",
                rating: 5
              },
              {
                name: "Michael Chen",
                role: "Patient",
                content: "Finally, a tool that explains my health risks in terms I can understand. The recommendations are practical and easy to follow.",
                rating: 5
              },
              {
                name: "Dr. Emily Rodriguez",
                role: "Endocrinologist", 
                content: "The diabetes prediction models are incredibly accurate. It's become an essential part of my practice workflow.",
                rating: 5
              }
            ].map((testimonial, index) => (
              <Reveal key={index} delayMs={index * 120}>
                <Card className="shadow-soft border-0 frosted h-full">
                  <CardContent className="p-6 h-full flex flex-col">
                  <div className="flex mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
                    ))}
                  </div>
                  <p className="text-muted-foreground mb-6 leading-relaxed italic flex-1">
                    "{testimonial.content}"
                  </p>
                  <div>
                    <p className="font-semibold text-foreground">{testimonial.name}</p>
                    <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                  </div>
                  </CardContent>
                </Card>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-foreground mb-4">Get in <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-glow">Touch</span></h2>
              <p className="text-xl text-muted-foreground">
                Have questions? We'd love to hear from you.
              </p>
            </div>

            <Card className="shadow-soft border-0 frosted">
              <CardContent className="p-8">
                <form className="space-y-6" onSubmit={async (e) => {
                  e.preventDefault();
                  const form = e.currentTarget as HTMLFormElement;
                  const formData = new FormData(form);
                  const name = String(formData.get('name') || '').trim();
                  const email = String(formData.get('email') || '').trim();
                  const message = String(formData.get('message') || '').trim();
                  if (!name || !email || !message) {
                    alert('Please fill in name, email and message.');
                    return;
                  }
                  const { error } = await supabase
                    .from('contacts')
                    .insert([{ name, email, message, source: 'landing' }]);
                  if (error) {
                    alert('Failed to send. Please try again.');
                    return;
                  }
                  alert('Message sent! We will get back to you soon.');
                  form.reset();
                }}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Name
                      </label>
                      <Input name="name" placeholder="Your name" className="bg-background/50" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Email
                      </label>
                      <Input name="email" type="email" placeholder="your.email@example.com" className="bg-background/50" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Message
                    </label>
                    <Textarea 
                      name="message"
                      placeholder="Tell us how we can help..."
                      className="min-h-[120px] bg-background/50"
                    />
                  </div>
                  <Button type="submit" className="medical-gradient text-white hover:shadow-glow transition-all duration-300 w-full">
                    Send Message
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Landing;