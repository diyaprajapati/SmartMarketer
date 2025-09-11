import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Car, Users, MapPin, Star, Phone, User } from "lucide-react";
import { toast } from "@/hooks/use-toast";

interface City {
  [key: string]: string[];
}

interface CitiesResponse {
  cities: City;
  tiers: {
    [key: string]: string;
  };
}

interface AreaResponse {
  areas: string[];
}

interface UserData {
  user_type: "driver" | "rider";
  user_id: string;
  name: string;
  phone: string;
  city: string;
  area: string;
  rating: number;
  trips_completed: number;
}

interface UserRegistrationProps {
  onRegistrationComplete: (userData: UserData) => void;
}

export const UserRegistration: React.FC<UserRegistrationProps> = ({ onRegistrationComplete }) => {
  const [userType, setUserType] = useState<"driver" | "rider">("rider");
  const [formData, setFormData] = useState({
    name: "",
    phone: "",
    city: "",
    area: "",
    rating: 4.0,
    trips_completed: 0,
  });
  const [cities, setCities] = useState<City>({});
  const [tiers, setTiers] = useState<{ [key: string]: string }>({});
  const [areas, setAreas] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingAreas, setLoadingAreas] = useState(false);

  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    fetchCities();
  }, []);

  useEffect(() => {
    if (formData.city) {
      fetchAreas(formData.city);
    }
  }, [formData.city]);

  const fetchCities = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/cities`);
      if (response.ok) {
        const data: CitiesResponse = await response.json();
        setCities(data.cities);
        setTiers(data.tiers);
      }
    } catch (error) {
      console.error("Error fetching cities:", error);
      toast({
        title: "Error",
        description: "Failed to load cities. Please try again.",
        variant: "destructive",
      });
    }
  };

  const fetchAreas = async (city: string) => {
    setLoadingAreas(true);
    try {
      const response = await fetch(`${API_BASE}/api/cities/${city}/areas`);
      if (response.ok) {
        const data: AreaResponse = await response.json();
        setAreas(data.areas);
        // Reset area selection when city changes
        setFormData((prev) => ({ ...prev, area: "" }));
      }
    } catch (error) {
      console.error("Error fetching areas:", error);
      setAreas([]);
    } finally {
      setLoadingAreas(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const userData: UserData = {
        user_type: userType,
        user_id: `${userType}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: formData.name,
        phone: formData.phone,
        city: formData.city,
        area: formData.area,
        rating: formData.rating,
        trips_completed: formData.trips_completed,
      };

      const response = await fetch(`${API_BASE}/api/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        const result = await response.json();
        toast({
          title: "Registration Successful! 🎉",
          description: `Welcome ${formData.name}! You're registered as a ${userType} in ${formData.city}.`,
        });
        onRegistrationComplete(userData);
      } else {
        const error = await response.json();
        throw new Error(error.detail || "Registration failed");
      }
    } catch (error) {
      console.error("Registration error:", error);
      toast({
        title: "Registration Failed",
        description: error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const getTierBadgeColor = (tier: string) => {
    switch (tier) {
      case "A":
        return "bg-purple-500 hover:bg-purple-600";
      case "B":
        return "bg-blue-500 hover:bg-blue-600";
      case "C":
        return "bg-green-500 hover:bg-green-600";
      default:
        return "bg-gray-500 hover:bg-gray-600";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-2xl mx-auto">
        <Card className="shadow-xl border-0">
          <CardHeader className="text-center bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg">
            <CardTitle className="text-3xl font-bold">Join SmartRide</CardTitle>
            <CardDescription className="text-blue-100">
              Register as a driver or rider to get started with dynamic pricing
            </CardDescription>
          </CardHeader>
          <CardContent className="p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* User Type Selection */}
              <div className="space-y-3">
                <Label className="text-lg font-semibold">I am a</Label>
                <RadioGroup
                  value={userType}
                  onValueChange={(value: "driver" | "rider") => setUserType(value)}
                  className="flex space-x-6"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="rider" id="rider" />
                    <Label htmlFor="rider" className="flex items-center space-x-2 cursor-pointer">
                      <Users className="w-5 h-5" />
                      <span>Rider</span>
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="driver" id="driver" />
                    <Label htmlFor="driver" className="flex items-center space-x-2 cursor-pointer">
                      <Car className="w-5 h-5" />
                      <span>Driver</span>
                    </Label>
                  </div>
                </RadioGroup>
              </div>

              {/* Personal Information */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name" className="flex items-center space-x-2">
                    <User className="w-4 h-4" />
                    <span>Full Name</span>
                  </Label>
                  <Input
                    id="name"
                    type="text"
                    placeholder="Enter your full name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone" className="flex items-center space-x-2">
                    <Phone className="w-4 h-4" />
                    <span>Phone Number</span>
                  </Label>
                  <Input
                    id="phone"
                    type="tel"
                    placeholder="10-digit phone number"
                    value={formData.phone}
                    onChange={(e) =>
                      setFormData({ ...formData, phone: e.target.value.replace(/\D/g, "").slice(0, 10) })
                    }
                    required
                  />
                </div>
              </div>

              {/* City Selection */}
              <div className="space-y-3">
                <Label className="flex items-center space-x-2">
                  <MapPin className="w-4 h-4" />
                  <span>Select Your City</span>
                </Label>
                <div className="space-y-4">
                  {Object.entries(cities).map(([tier, cityList]) => (
                    <div key={tier} className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Badge className={`${getTierBadgeColor(tier)} text-white`}>Tier {tier}</Badge>
                        <span className="text-sm text-gray-600">{tiers[tier]}</span>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {cityList.map((city) => (
                          <Button
                            key={city}
                            type="button"
                            variant={formData.city === city ? "default" : "outline"}
                            size="sm"
                            onClick={() => setFormData({ ...formData, city: city })}
                            className="justify-start"
                          >
                            {city}
                          </Button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Area Selection */}
              {formData.city && (
                <div className="space-y-2">
                  <Label htmlFor="area">Select Area in {formData.city}</Label>
                  <Select
                    value={formData.area}
                    onValueChange={(value) => setFormData({ ...formData, area: value })}
                    disabled={loadingAreas}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={loadingAreas ? "Loading areas..." : "Choose your area"} />
                    </SelectTrigger>
                    <SelectContent>
                      {areas.map((area) => (
                        <SelectItem key={area} value={area}>
                          {area}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* Experience Fields */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="rating" className="flex items-center space-x-2">
                    <Star className="w-4 h-4" />
                    <span>Rating (1-5)</span>
                  </Label>
                  <Input
                    id="rating"
                    type="number"
                    min="1"
                    max="5"
                    step="0.1"
                    value={formData.rating}
                    onChange={(e) => setFormData({ ...formData, rating: parseFloat(e.target.value) || 4.0 })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="trips">Trips Completed</Label>
                  <Input
                    id="trips"
                    type="number"
                    min="0"
                    value={formData.trips_completed}
                    onChange={(e) => setFormData({ ...formData, trips_completed: parseInt(e.target.value) || 0 })}
                  />
                </div>
              </div>

              {/* Submit Button */}
              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3"
                disabled={loading || !formData.name || !formData.phone || !formData.city || !formData.area}
              >
                {loading ? "Registering..." : `Register as ${userType.charAt(0).toUpperCase() + userType.slice(1)}`}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
