import { Card } from "@/components/ui/card";
import { TrendingUp, Zap, Shield, BarChart3, Globe, Users } from "lucide-react";

export const Features = () => {
  const features = [
    {
      icon: TrendingUp,
      title: "Dynamic Pricing",
      description: "AI-powered pricing that adapts to market conditions in real-time, ensuring optimal profit margins.",
      highlight: "Smart Algorithms"
    },
    {
      icon: BarChart3,
      title: "Commission Logic",
      description: "Transparent fee structure that scales with transaction value, aligning our success with yours.",
      highlight: "Fair & Transparent"
    },
    {
      icon: Zap,
      title: "Easy Transactions",
      description: "Streamlined checkout process with multiple payment options and instant settlement.",
      highlight: "Lightning Fast"
    },
    {
      icon: Shield,
      title: "Secure Platform",
      description: "Enterprise-grade security with end-to-end encryption and fraud protection.",
      highlight: "Bank-Level Security"
    },
    {
      icon: Globe,
      title: "Global Reach",
      description: "Connect with buyers and sellers worldwide with multi-currency support.",
      highlight: "Worldwide Access"
    },
    {
      icon: Users,
      title: "Community Driven",
      description: "Join a thriving marketplace with verified users and quality assurance.",
      highlight: "Trusted Network"
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
            Everything you need to succeed in the modern marketplace, powered by cutting-edge technology.
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