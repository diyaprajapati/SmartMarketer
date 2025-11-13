import { Card } from "@/components/ui/card";
import { TrendingUp, Zap, Shield, BarChart3, Globe, Users } from "lucide-react";

export const Features = () => {
  const features = [
    {
      icon: TrendingUp,
      title: "ML-Powered Pricing",
      description: "Ensemble learning model combining Random Forest, XGBoost, LightGBM, and Neural Networks with 96.7% accuracy (R² score).",
      highlight: "96.7% Accuracy"
    },
    {
      icon: BarChart3,
      title: "City-Tier System",
      description: "Intelligent pricing across 19 cities categorized into Tier A (Metropolitan), Tier B (Major), and Tier C (Developing) cities.",
      highlight: "19 Cities"
    },
    {
      icon: Zap,
      title: "Event-Driven Updates",
      description: "Real-time price updates triggered only when new drivers or riders join, reducing computational overhead by 80%.",
      highlight: "Real-Time"
    },
    {
      icon: Shield,
      title: "Supply/Demand Balance",
      description: "Dynamic pricing that automatically adjusts based on real-time supply (drivers) and demand (riders) ratios.",
      highlight: "Auto-Balance"
    },
    {
      icon: Globe,
      title: "WebSocket Communication",
      description: "Instant price updates delivered via WebSocket connections with sub-150ms latency for seamless user experience.",
      highlight: "Instant Updates"
    },
    {
      icon: Users,
      title: "Fair Pricing",
      description: "All riders see the same base price in the same area, ensuring transparency and fairness in pricing.",
      highlight: "Transparent"
    }
  ];

  return (
    <section className="py-24">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Powerful <span className="bg-gradient-primary bg-clip-text text-transparent">Features</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Advanced machine learning features that power intelligent, real-time pricing for ride-sharing platforms.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card key={index} className="p-6 group hover:shadow-glow transition-all duration-300 border-border/50 bg-card/50 backdrop-blur-sm">
              {/* Icon and highlight */}
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 bg-gradient-primary/10 rounded-lg flex items-center justify-center group-hover:bg-gradient-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <span className="text-xs font-medium text-accent bg-accent/10 px-2 py-1 rounded-full">
                  {feature.highlight}
                </span>
              </div>

              {/* Content */}
              <h3 className="text-xl font-bold mb-3 group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};