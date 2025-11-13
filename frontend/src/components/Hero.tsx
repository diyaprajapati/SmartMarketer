import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-image.jpg";
import { ArrowRight, Sparkles } from "lucide-react";

export const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-secondary/20 to-accent/20" />

      {/* Hero image overlay */}
      <div className="absolute inset-0 opacity-10">
        <img src={heroImage} alt="SmartMarketer Platform" className="w-full h-full object-cover" />
      </div>

      <div className="relative z-10 container mx-auto px-6 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-gradient-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-8 backdrop-blur-sm border border-primary/20">
          <Sparkles className="w-4 h-4" />
          ML-Powered Dynamic Pricing for Ride-Sharing
        </div>

        {/* Main headline */}
        <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
          Smart Pricing, <span className="bg-gradient-primary bg-clip-text text-transparent">Real-Time</span>
          <br />
          for Ride-Sharing Platforms
        </h1>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto leading-relaxed">
          Advanced machine learning algorithms that adjust ride prices in real-time based on supply, demand, and market conditions.
          Maximize revenue while ensuring fair pricing for riders and drivers.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
          <Button variant="hero" size="lg" className="text-lg px-8 py-6 h-auto" asChild>
            <a href="/pricing">
              Try Live Pricing
              <ArrowRight className="w-5 h-5" />
            </a>
          </Button>
          <Button variant="outline" size="lg" className="text-lg px-8 py-6 h-auto" asChild>
            <a href="/dashboard">ML Dashboard</a>
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-primary mb-2">19</div>
            <div className="text-muted-foreground">Cities Supported</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-primary mb-2">96.7%</div>
            <div className="text-muted-foreground">Model Accuracy (R²)</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-primary mb-2">&lt;150ms</div>
            <div className="text-muted-foreground">Real-Time Updates</div>
          </div>
        </div>
      </div>

      {/* Floating elements */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-primary/20 rounded-full blur-xl animate-pulse" />
      <div className="absolute bottom-20 right-10 w-32 h-32 bg-accent/20 rounded-full blur-xl animate-pulse delay-1000" />
    </section>
  );
};
