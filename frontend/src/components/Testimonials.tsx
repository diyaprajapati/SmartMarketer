import { Card } from "@/components/ui/card";
import { Star, Quote } from "lucide-react";

export const Testimonials = () => {
  const testimonials = [
    {
      name: "Sarah Chen",
      role: "E-commerce Entrepreneur",
      company: "TechGear Plus",
      content: "SmartMarketer's dynamic pricing increased our profit margins by 35% while keeping our customers happy. The platform is intuitive and the results speak for themselves.",
      rating: 5,
      avatar: "SC"
    },
    {
      name: "Marcus Rodriguez",
      role: "Marketplace Seller",
      company: "Artisan Crafts Co.",
      content: "Finally, a platform that understands the value of my handmade products. The commission structure is fair and the buyer quality is exceptional.",
      rating: 5,
      avatar: "MR"
    },
    {
      name: "Emily Watson",
      role: "Digital Products Seller",
      company: "Creative Studios",
      content: "The transaction process is seamless and the pricing intelligence has helped me scale from $10K to $100K in monthly sales within 8 months.",
      rating: 5,
      avatar: "EW"
    }
  ];

  const useCases = [
    {
      title: "B2B Software Sales",
      description: "Enterprise software companies use our platform to optimize pricing for different market segments.",
      metric: "Average 28% revenue increase"
    },
    {
      title: "Digital Marketplace",
      description: "Online creators and digital product sellers leverage our dynamic pricing for maximum profitability.",
      metric: "95% seller satisfaction"
    },
    {
      title: "Physical Goods Trading",
      description: "Manufacturers and retailers optimize inventory turnover with intelligent pricing strategies.",
      metric: "42% faster sales cycles"
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
            Join thousands of successful entrepreneurs who have transformed their business with SmartMarketer.
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
            See how different industries leverage SmartMarketer to drive growth.
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