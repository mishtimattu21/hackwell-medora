import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import Reveal from "@/components/Reveal";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { supabase } from "@/lib/supabaseClient";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { 
  Star,
  ArrowRight,
  Loader2,
  CheckCircle,
  AlertCircle
} from "lucide-react";

const Landing = () => {
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [messageLength, setMessageLength] = useState(0);
  const [isDemoOpen, setIsDemoOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Pause video when dialog closes
  useEffect(() => {
    if (!isDemoOpen && videoRef.current) {
      try {
        videoRef.current.pause();
      } catch (_) {}
    }
  }, [isDemoOpen]);

  

  const handleContactSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const form = e.currentTarget as HTMLFormElement;
      const formData = new FormData(form);
      const name = String(formData.get('name') || '').trim();
      const email = String(formData.get('email') || '').trim();
      const message = String(formData.get('message') || '').trim();
      
      // Validation
      if (!name || !email || !message) {
        toast({
          title: "Missing Information",
          description: "Please fill in all required fields (name, email, and message).",
          variant: "destructive",
        });
        return;
      }

      // Message length validation
      if (message.length < 10) {
        toast({
          title: "Message Too Short",
          description: "Please provide a more detailed message (at least 10 characters).",
          variant: "destructive",
        });
        return;
      }

      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        toast({
          title: "Invalid Email",
          description: "Please enter a valid email address.",
          variant: "destructive",
        });
        return;
      }

      // Submit to Supabase
      const { error } = await supabase
        .from('contacts')
        .insert([{ name, email, message, source: 'landing' }]);
      
      if (error) {
        throw error;
      }

      // Success
      toast({
        title: "Message Sent Successfully!",
        description: "Thank you for your message. We'll get back to you within 24 hours.",
        duration: 5000,
      });
      
      form.reset();
    } catch (error: any) {
      console.error('Contact form error:', error);
      toast({
        title: "Failed to Send Message",
        description: error.message || "An unexpected error occurred. Please try again later.",
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

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
        
        <div className="relative z-10 container mx-auto px-4 md:translate-x-2 lg:translate-x-4">
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
                
                <Dialog open={isDemoOpen} onOpenChange={setIsDemoOpen}>
                  <DialogTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="lg"
                      className="text-lg px-8 py-6 hover:bg-primary/5 border-primary/20"
                    >
                      Watch Demo
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="w-full max-w-4xl p-0 overflow-hidden border-primary/20">
                    <DialogHeader className="px-4 pt-4">
                      <DialogTitle className="text-xl">Platform Demo</DialogTitle>
                    </DialogHeader>
                    <div className="relative px-4 pb-4">
                      <div className="relative w-full rounded-md overflow-hidden border border-border bg-background">
                        <div className="aspect-video w-full">
                        <video
                          ref={videoRef}
                          src="https://auzwwcdtxdafwimpxvil.supabase.co/storage/v1/object/public/demo/demo.mp4"
                          controls
                          className="h-full w-full object-contain bg-black"
                          />
                        </div>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>

            {/* Right: Hero image from public */}
            <div className="relative slide-in flex justify-end pr-9 md:pr-8 lg:pr-12">
              <div className="relative w-full max-w-xl lg:max-w-2xl ml-auto">
                <div className="aspect-[4/3] flex items-center justify-center">
                  <img src="/hero-2.png" alt="Medical hero" className="w-full h-full object-contain" />
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

          <div className="relative grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto items-start">
            {[
              {
                image: "/report.png",
                title: "Upload Report",
                description: "Upload your medical reports, lab results, or key health parameters"
              },
              {
                image: "/parameters.png",
                title: "Fill Parameters",
                description: "Complete any missing health information for accurate analysis"
              },
              {
                image: "/predictions.png",
                title: "Get Predictions",
                description: "Receive detailed risk analysis with actionable recommendations"
              }
            ].map((item, index) => (
              <Reveal key={index} delayMs={index * 120}>
                <Card className="relative overflow-hidden shadow-soft hover:shadow-glow transition-all duration-300 border-0 frosted h-full">
                  <CardContent className="p-8 text-center h-full flex flex-col">
                    <div className="relative mb-6">
                      <img src={item.image} alt={item.title} className="h-12 w-auto md:h-14 object-contain mx-auto" />
                    </div>
                    <h3 className="text-xl font-semibold text-foreground mb-3">{item.title}</h3>
                    <p className="text-muted-foreground leading-relaxed flex-1">{item.description}</p>
                  </CardContent>
                </Card>
              </Reveal>
            ))}
            {/* Arrows between steps for md+ screens */}
            <img src="/arrow1.png" alt="to step 2" className="hidden md:block absolute top-1/2 -translate-y-1/2 left-1/3 -ml-6 h-8 w-auto" />
            <img src="/arrow2.png" alt="to step 3" className="hidden md:block absolute top-1/2 -translate-y-1/2 left-2/3 -ml-6 h-8 w-auto" />
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
                image: "/multi_disease.png",
                title: "Multi-disease Prediction",
                description: "Advanced models for diabetes, hypertension, heart failure, and more"
              },
              {
                image: "/risk_scoring.png",
                title: "Risk Scoring & Probabilities",
                description: "Clear probability scores with confidence intervals"
              },
              {
                image: "/clinic_friendly.png",
                title: "Clinician-friendly Insights", 
                description: "Professional reports formatted for healthcare providers"
              },
              {
                image: "/patient_recommend.png",
                title: "Patient Recommend",
                description: "Personalized lifestyle and treatment suggestions"
              },
              {
                image: "/dashboardd.png",
                title: "Dashboard with Charts",
                description: "Interactive visualizations of health trends and risks"
              },
              {
                image: "/password.png",
                title: "Secure & Private",
                description: "HIPAA-compliant data handling and encryption"
              }
            ].map((feature, index) => (
              <Reveal key={index} delayMs={index * 100}>
                <Card className="shadow-soft hover:shadow-glow transition-all duration-300 border frosted group h-full border-transparent hover:border-primary hover:scale-[1.02]">
                  <CardContent className="p-6 h-full flex flex-col">
                  <div className="mb-4">
                    <img src={feature.image} alt={feature.title} className="h-12 w-auto md:h-14 object-contain" />
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
      {/* <section id="feedback" className="py-20 bg-muted/30">
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
                content: "Medora has transformed how I assess patient risk. The accuracy and clarity of reports save significant time in consultations.",
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
      </section> */}

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
                <form className="space-y-6" onSubmit={handleContactSubmit}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Name <span className="text-red-500">*</span>
                      </label>
                      <Input 
                        name="name" 
                        placeholder="Your name" 
                        className="bg-background/50" 
                        disabled={isSubmitting}
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">
                        Email <span className="text-red-500">*</span>
                      </label>
                      <Input 
                        name="email" 
                        type="email" 
                        placeholder="your.email@example.com" 
                        className="bg-background/50" 
                        disabled={isSubmitting}
                        required
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Message <span className="text-red-500">*</span>
                    </label>
                    <Textarea 
                      name="message"
                      placeholder="Tell us how we can help..."
                      className="min-h-[120px] bg-background/50"
                      disabled={isSubmitting}
                      required
                      onChange={(e) => setMessageLength(e.target.value.length)}
                    />
                    <div className="flex justify-between items-center mt-1">
                      <span className={`text-xs ${messageLength > 0 ? 'text-foreground' : 'text-muted-foreground'}`}>
                        {messageLength > 0 && `${messageLength} characters`}
                      </span>
                      <span className={`text-xs ${messageLength < 10 && messageLength > 0 ? 'text-red-500' : 'text-muted-foreground'}`}>
                        {messageLength < 10 && messageLength > 0 ? 'Minimum 10 characters required' : 'Minimum 10 characters'}
                      </span>
                    </div>
                  </div>
                  <Button 
                    type="submit" 
                    className="medical-gradient text-white hover:shadow-glow transition-all duration-300 w-full"
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        Sending Message...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Send Message
                      </>
                    )}
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