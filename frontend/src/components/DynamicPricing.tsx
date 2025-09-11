import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Car,
  Users,
  MapPin,
  Clock,
  TrendingUp,
  Activity,
  Zap,
  RefreshCw,
  DollarSign,
  BarChart3,
  Timer,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

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

interface PriceData {
  predicted_price: number;
  base_price: number;
  city_tier: string;
  surge_multiplier: number;
  surge_level: string;
  is_peak_hour: boolean;
  demand_supply_ratio: number;
  timestamp: string;
  city_info: {
    name: string;
    tier: string;
    base_multiplier: number;
  };
  user_type: string;
  area: string;
}

interface SupplyDemandData {
  riders: number;
  drivers: number;
  ratio: number;
}

interface WebSocketMessage {
  type: string;
  city: string;
  area: string;
  timestamp: string;
  supply_demand: SupplyDemandData;
  pricing: {
    rider: PriceData;
    driver: PriceData;
  };
}

interface DynamicPricingProps {
  userData: UserData;
  onBack: () => void;
}

export const DynamicPricing: React.FC<DynamicPricingProps> = ({ userData, onBack }) => {
  const [currentPrice, setCurrentPrice] = useState<PriceData | null>(null);
  const [supplyDemand, setSupplyDemand] = useState<SupplyDemandData>({ riders: 50, drivers: 30, ratio: 1.67 });
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string>("");
  const [priceHistory, setPriceHistory] = useState<number[]>([]);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [countdown, setCountdown] = useState(10);

  const wsRef = useRef<WebSocket | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  const API_BASE = "http://localhost:8000";
  const WS_BASE = "ws://localhost:8000";

  useEffect(() => {
    // Initial price fetch
    fetchCurrentPrice();

    // Setup WebSocket connection
    connectWebSocket();

    // Setup countdown timer
    startCountdown();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, [userData.city]);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`${WS_BASE}/ws/${userData.city}`);

      ws.onopen = () => {
        setIsConnected(true);
        console.log("WebSocket connected");
        toast({
          title: "Real-time Updates Connected! 🚀",
          description: "You'll receive live price updates every 10 seconds.",
        });
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          if (data.type === "price_update" && data.city === userData.city) {
            const relevantPrice = userData.user_type === "rider" ? data.pricing.rider : data.pricing.driver;
            setCurrentPrice(relevantPrice);
            setSupplyDemand(data.supply_demand);
            setLastUpdate(data.timestamp);

            // Update price history (keep last 20 points)
            setPriceHistory((prev) => [...prev.slice(-19), relevantPrice.predicted_price]);

            // Reset countdown
            setCountdown(10);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log("WebSocket disconnected");
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("Failed to connect WebSocket:", error);
    }
  };

  const startCountdown = () => {
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          return 10; // Reset to 10
        }
        return prev - 1;
      });
    }, 1000);
  };

  const fetchCurrentPrice = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/price`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          city: userData.city,
          user_type: userData.user_type,
          area: userData.area,
          current_riders: supplyDemand.riders,
          current_drivers: supplyDemand.drivers,
          user_rating: userData.rating,
          trips_completed: userData.trips_completed,
        }),
      });

      if (response.ok) {
        const priceData: PriceData = await response.json();
        setCurrentPrice(priceData);
        setLastUpdate(priceData.timestamp);
        setPriceHistory((prev) => [...prev.slice(-19), priceData.predicted_price]);
      }
    } catch (error) {
      console.error("Error fetching price:", error);
      toast({
        title: "Price Fetch Failed",
        description: "Could not get current price. Please try again.",
        variant: "destructive",
      });
    }
  };

  const getSurgeColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "low":
        return "bg-green-500";
      case "medium":
        return "bg-yellow-500";
      case "high":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case "A":
        return "bg-purple-500";
      case "B":
        return "bg-blue-500";
      case "C":
        return "bg-green-500";
      default:
        return "bg-gray-500";
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const priceChange =
    priceHistory.length >= 2 ? priceHistory[priceHistory.length - 1] - priceHistory[priceHistory.length - 2] : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <Card className="shadow-lg">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
            <div className="flex justify-between items-center">
              <div>
                <CardTitle className="text-2xl font-bold">Welcome, {userData.name}! 👋</CardTitle>
                <CardDescription className="text-blue-100">
                  {userData.user_type === "rider" ? "Book your ride" : "Accept ride requests"} with real-time pricing
                </CardDescription>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${isConnected ? "bg-green-400" : "bg-red-400"}`}></div>
                  <span className="text-sm">{isConnected ? "Live" : "Disconnected"}</span>
                </div>
                <Button
                  onClick={onBack}
                  variant="outline"
                  className="text-white border-white hover:bg-white hover:text-blue-600"
                >
                  Change User
                </Button>
              </div>
            </div>
          </CardHeader>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Price Display */}
          <div className="lg:col-span-2 space-y-6">
            {/* Current Price Card */}
            <Card className="shadow-lg border-0">
              <CardHeader className="pb-4">
                <div className="flex justify-between items-center">
                  <CardTitle className="flex items-center space-x-2">
                    <DollarSign className="w-6 h-6 text-green-600" />
                    <span>Current Price</span>
                  </CardTitle>
                  <div className="flex items-center space-x-2">
                    <Timer className="w-4 h-4 text-gray-500" />
                    <span className="text-sm text-gray-500">Next update in {countdown}s</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {currentPrice ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className="text-5xl font-bold text-gray-900 mb-2">₹{currentPrice.predicted_price}</div>
                      <div className="flex items-center justify-center space-x-2">
                        {priceChange !== 0 && (
                          <Badge variant={priceChange > 0 ? "destructive" : "default"} className="text-sm">
                            {priceChange > 0 ? "+" : ""}₹{priceChange.toFixed(2)}
                          </Badge>
                        )}
                        <span className="text-gray-500">from base ₹{currentPrice.base_price}</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="space-y-1">
                        <div className="text-2xl font-semibold text-blue-600">{currentPrice.surge_multiplier}x</div>
                        <div className="text-sm text-gray-500">Surge</div>
                      </div>
                      <div className="space-y-1">
                        <div className="text-2xl font-semibold text-purple-600">
                          {currentPrice.demand_supply_ratio.toFixed(1)}
                        </div>
                        <div className="text-sm text-gray-500">Demand Ratio</div>
                      </div>
                      <div className="space-y-1">
                        <Badge className={getTierColor(currentPrice.city_tier)}>Tier {currentPrice.city_tier}</Badge>
                        <div className="text-sm text-gray-500">City Tier</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <RefreshCw className="w-12 h-12 text-gray-400 mx-auto mb-4 animate-spin" />
                    <p className="text-gray-500">Loading current price...</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Market Conditions */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-orange-600" />
                  <span>Market Conditions</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Surge Level</span>
                      {currentPrice && (
                        <Badge className={`${getSurgeColor(currentPrice.surge_level)} text-white`}>
                          {currentPrice.surge_level}
                        </Badge>
                      )}
                    </div>
                    <div className="text-2xl font-bold">{currentPrice?.surge_level || "Loading..."}</div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Peak Hour</span>
                      <Badge variant={currentPrice?.is_peak_hour ? "destructive" : "default"}>
                        {currentPrice?.is_peak_hour ? "Yes" : "No"}
                      </Badge>
                    </div>
                    <div className="text-2xl font-bold">{currentPrice?.is_peak_hour ? "🔥 Peak" : "✅ Normal"}</div>
                  </div>

                  <div className="space-y-2">
                    <span className="text-sm font-medium">Last Updated</span>
                    <div className="text-sm font-mono">{lastUpdate ? formatTime(lastUpdate) : "Never"}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Price History */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-green-600" />
                  <span>Price Trend</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-24 flex items-end space-x-1">
                  {priceHistory.slice(-15).map((price, index) => (
                    <div
                      key={index}
                      className="bg-blue-500 rounded-t"
                      style={{
                        height: `${(price / Math.max(...priceHistory.slice(-15))) * 100}%`,
                        minHeight: "8px",
                        width: "100%",
                      }}
                      title={`₹${price}`}
                    />
                  ))}
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  Last 15 price updates • {priceHistory.length} total updates
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* User Info */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  {userData.user_type === "rider" ? <Users className="w-5 h-5" /> : <Car className="w-5 h-5" />}
                  <span>Your Profile</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Type</span>
                  <Badge>{userData.user_type.charAt(0).toUpperCase() + userData.user_type.slice(1)}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Location</span>
                  <div className="text-right">
                    <div className="font-medium">{userData.city}</div>
                    <div className="text-sm text-gray-500">{userData.area}</div>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Rating</span>
                  <span className="font-medium">⭐ {userData.rating}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Trips</span>
                  <span className="font-medium">{userData.trips_completed}</span>
                </div>
              </CardContent>
            </Card>

            {/* Supply & Demand */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  <span>Live Supply & Demand</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Riders Active</span>
                    <span className="text-lg font-bold text-orange-600">{supplyDemand.riders}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-orange-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min((supplyDemand.riders / 200) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Drivers Online</span>
                    <span className="text-lg font-bold text-green-600">{supplyDemand.drivers}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min((supplyDemand.drivers / 100) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                <Separator />

                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{supplyDemand.ratio.toFixed(2)}:1</div>
                  <div className="text-sm text-gray-500">Demand-to-Supply Ratio</div>
                </div>
              </CardContent>
            </Card>

            {/* Action Button */}
            <Card className="shadow-lg">
              <CardContent className="p-6">
                <Button
                  className="w-full bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white py-3"
                  size="lg"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  {userData.user_type === "rider" ? "Book Ride Now" : "Go Online"}
                </Button>
                <p className="text-xs text-gray-500 mt-2 text-center">
                  {userData.user_type === "rider" ? "Price updates every 10 seconds" : "Start accepting ride requests"}
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};
