import { Card } from "@/components/ui/card";
import { Users, Brain, Zap } from "lucide-react";

export const HowItWorks = () => {
  const steps = [
    {
      icon: Users,
      title: "Register & Connect",
      description: "Drivers and riders register in their city and area. The system tracks real-time supply (drivers) and demand (riders) across 19 cities.",
      step: "01"
    },
    {
      icon: Brain,
      title: "ML Calculates Price",
      description: "Our ensemble ML model analyzes supply/demand ratio, city tier, peak hours, and area to calculate optimal dynamic pricing with 96.7% accuracy.",
      step: "02"
    },
    {
      icon: Zap,
      title: "Real-Time Updates",
      description: "Prices update instantly via WebSocket when new drivers or riders join. Event-driven architecture ensures efficient, responsive pricing.",
      step: "03"
    }
  ];

  return (
    <section className="py-24 bg-secondary/30">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            How It <span className="bg-gradient-primary bg-clip-text text-transparent">Works</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Three simple steps to understand how our ML-powered dynamic pricing system works for ride-sharing platforms.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              <Card className="p-8 text-center h-full bg-card/50 backdrop-blur-sm border-border/50 hover:shadow-soft transition-all duration-300">
                {/* Step number */}
                <div className="absolute -top-4 left-8 bg-gradient-primary text-white w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold">
                  {step.step}
                </div>
                
                {/* Icon */}
                <div className="w-16 h-16 bg-gradient-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <step.icon className="w-8 h-8 text-primary" />
                </div>

                {/* Content */}
                <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{step.description}</p>
              </Card>

              {/* Arrow connector */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-4 w-8 h-0.5 bg-gradient-primary transform -translate-y-1/2" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};