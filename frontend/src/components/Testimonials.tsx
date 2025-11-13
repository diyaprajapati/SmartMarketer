import { Card } from "@/components/ui/card";
import { Star, Quote } from "lucide-react";

export const Testimonials = () => {
  const testimonials = [
    {
      name: "Rajesh Kumar",
      role: "Ride-Sharing Platform CEO",
      company: "CityRide India",
      content: "SmartMarketer's ML-powered pricing system increased our revenue by 25% while maintaining rider satisfaction. The event-driven updates are incredibly efficient.",
      rating: 5,
      avatar: "RK"
    },
    {
      name: "Priya Sharma",
      role: "Platform Operations Manager",
      company: "UrbanMobility",
      content: "The city-tier system and real-time supply/demand balancing have transformed our pricing strategy. The 96.7% model accuracy is impressive.",
      rating: 5,
      avatar: "PS"
    },
    {
      name: "Amit Patel",
      role: "CTO",
      company: "RideShare Pro",
      content: "The WebSocket integration provides instant price updates with minimal latency. Our drivers and riders love the transparency and fairness of the pricing.",
      rating: 5,
      avatar: "AP"
    }
  ];

  const useCases = [
    {
      title: "Metropolitan Cities (Tier A)",
      description: "High-demand cities like Mumbai, Delhi, and Bangalore with premium pricing and complex peak hour patterns.",
      metric: "1.1-1.5x base multiplier"
    },
    {
      title: "Major Cities (Tier B)",
      description: "Cities like Hyderabad, Pune, and Ahmedabad with moderate demand and standard pricing strategies.",
      metric: "0.7-1.0x base multiplier"
    },
    {
      title: "Developing Cities (Tier C)",
      description: "Emerging markets with affordable pricing to encourage platform adoption and growth.",
      metric: "0.4-0.7x base multiplier"
    }
  ];

  return (
    <section className="py-24 bg-secondary/30">
      <div className="container mx-auto px-6">
        {/* Testimonials Section */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            What Our <span className="bg-gradient-primary bg-clip-text text-transparent">Users Say</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Trusted by ride-sharing platforms across India for intelligent, real-time dynamic pricing.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-20">
          {testimonials.map((testimonial, index) => (
            <Card key={index} className="p-6 bg-card/70 backdrop-blur-sm border-border/50 hover:shadow-soft transition-all duration-300">
              {/* Quote icon */}
              <Quote className="w-8 h-8 text-primary/20 mb-4" />
              
              {/* Stars */}
              <div className="flex gap-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-4 h-4 fill-primary text-primary" />
                ))}
              </div>

              {/* Content */}
              <p className="text-foreground leading-relaxed mb-6 italic">
                "{testimonial.content}"
              </p>

              {/* Author */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-primary rounded-full flex items-center justify-center text-white font-medium text-sm">
                  {testimonial.avatar}
                </div>
                <div>
                  <div className="font-semibold">{testimonial.name}</div>
                  <div className="text-sm text-muted-foreground">{testimonial.role}</div>
                  <div className="text-xs text-primary">{testimonial.company}</div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Use Cases Section */}
        <div className="text-center mb-12">
          <h3 className="text-3xl font-bold mb-4">Use Cases</h3>
          <p className="text-muted-foreground max-w-xl mx-auto">
            See how our city-tier system adapts pricing across different market segments.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {useCases.map((useCase, index) => (
            <Card key={index} className="p-6 text-center bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/30 transition-all duration-300">
              <h4 className="text-xl font-bold mb-3">{useCase.title}</h4>
              <p className="text-muted-foreground mb-4 leading-relaxed">{useCase.description}</p>
              <div className="text-primary font-semibold text-sm bg-primary/10 px-3 py-1 rounded-full inline-block">
                {useCase.metric}
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};