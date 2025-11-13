import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  TrendingUp,
  Users,
  MapPin,
  BarChart3,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Target,
  Brain,
  Zap,
  Car,
  Navigation,
  Building2,
} from "lucide-react";

interface CityStats {
  riders: number;
  drivers: number;
  ratio: number;
  last_updated?: string;
}

interface CityData {
  name: string;
  tier: string;
  areas: string[];
  stats: CityStats;
  samplePrice?: {
    rider: number;
    driver: number;
    surge_level: string;
  };
}

interface DashboardData {
  total_cities: number;
  cities_by_tier: {
    A: string[];
    B: string[];
    C: string[];
  };
  city_stats: Record<string, CityStats>;
  system_info: {
    version: string;
    model_status: string;
    event_driven: boolean;
  };
}

export const MLDashboard = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [citiesData, setCitiesData] = useState<CityData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState("overview");
  const [selectedCity, setSelectedCity] = useState<string | null>(null);

  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    fetchDashboardData();
    // Refresh more frequently to catch real-time updates
    const interval = setInterval(fetchDashboardData, 3000); // Refresh every 3 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (dashboardData) {
      loadCitiesDetails();
    }
  }, [dashboardData]);

  const fetchDashboardData = async () => {
    try {
      // Fetch cities data
      const citiesResponse = await fetch(`${API_BASE}/api/cities`);
      if (!citiesResponse.ok) throw new Error("Failed to fetch cities");
      const citiesData = await citiesResponse.json();

      // Fetch city stats
      const statsResponse = await fetch(`${API_BASE}/api/city-stats`);
      if (!statsResponse.ok) throw new Error("Failed to fetch city stats");
      const statsData = await statsResponse.json();

      setDashboardData({
        total_cities: citiesData.total_cities,
        cities_by_tier: citiesData.cities,
        city_stats: statsData.stats || {},
        system_info: {
          version: "1.0.0",
          model_status: "Active",
          event_driven: true,
        },
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const loadCitiesDetails = async () => {
    if (!dashboardData) return;

    // Get all cities that have actual stats (registered users)
    const citiesWithStats = Object.keys(dashboardData.city_stats || {});
    
    // If no cities have stats yet, show empty state
    if (citiesWithStats.length === 0) {
      setCitiesData([]);
      return;
    }

    const citiesDetails: CityData[] = [];

    // Process only cities that have actual registered users
    for (const cityName of citiesWithStats) {
      try {
        // Get areas for city
        const areasResponse = await fetch(`${API_BASE}/api/cities/${encodeURIComponent(cityName)}/areas`);
        const areasData = areasResponse.ok ? await areasResponse.json() : { areas: [] };

        // Get stats for city (we know it exists since we're iterating over citiesWithStats)
        const cityStats = dashboardData.city_stats[cityName];
        const riders = cityStats?.riders || 0;
        const drivers = cityStats?.drivers || 0;
        const ratio = riders > 0 && drivers > 0 ? Math.round((riders / drivers) * 100) / 100 : 0;

        // Determine tier
        let tier = "C";
        if (dashboardData.cities_by_tier.A.includes(cityName)) tier = "A";
        else if (dashboardData.cities_by_tier.B.includes(cityName)) tier = "B";

        // Calculate real price based on actual rider/driver counts
        let samplePrice = undefined;
        if (riders > 0 || drivers > 0) {
          try {
            const priceResponse = await fetch(`${API_BASE}/api/price`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                city: cityName,
                user_type: "rider",
                area: areasData.areas[0] || "",
                current_riders: riders,
                current_drivers: drivers,
              }),
            });
            if (priceResponse.ok) {
              const priceData = await priceResponse.json();
              samplePrice = {
                rider: priceData.predicted_price,
                driver: priceData.predicted_price * 0.7, // Approximate driver earnings
                surge_level: priceData.surge_level,
              };
            }
          } catch (e) {
            console.error(`Error fetching price for ${cityName}:`, e);
          }
        }

        citiesDetails.push({
          name: cityName,
          tier,
          areas: areasData.areas || [],
          stats: {
            riders,
            drivers,
            ratio,
            last_updated: cityStats?.last_updated,
          },
          samplePrice,
        });
      } catch (e) {
        console.error(`Error loading details for ${cityName}:`, e);
      }
    }

    // Sort by total users (riders + drivers) descending
    citiesDetails.sort((a, b) => (b.stats.riders + b.stats.drivers) - (a.stats.riders + a.stats.drivers));

    setCitiesData(citiesDetails);
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case "A":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200";
      case "B":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "C":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  const getSurgeColor = (level: string) => {
    switch (level) {
      case "High":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      case "Medium":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      default:
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Brain className="w-16 h-16 mx-auto mb-4 animate-pulse text-primary" />
          <p className="text-lg font-medium">Loading SmartMarketer Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-6 py-8">
        <Alert className="border-red-200 bg-red-50 dark:bg-red-950">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Failed to load dashboard: {error}. Make sure the API is running on {API_BASE}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // Calculate totals from real data - only count cities with actual users
  const totalRiders = Object.values(dashboardData?.city_stats || {}).reduce(
    (sum, stat) => sum + (stat.riders || 0),
    0
  );
  const totalDrivers = Object.values(dashboardData?.city_stats || {}).reduce(
    (sum, stat) => sum + (stat.drivers || 0),
    0
  );
  // Only count cities that have at least one registered user
  const activeCities = Object.values(dashboardData?.city_stats || {}).filter(
    (stat) => (stat.riders || 0) > 0 || (stat.drivers || 0) > 0
  ).length;

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">SmartMarketer Dynamic Pricing Dashboard</h1>
        <p className="text-muted-foreground">Real-time pricing analytics and city-based market monitoring</p>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="cities">Cities</TabsTrigger>
          <TabsTrigger value="pricing">Pricing</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Cities</CardTitle>
                <Building2 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{dashboardData?.total_cities || 0}</div>
                <p className="text-xs text-muted-foreground">
                  {dashboardData?.cities_by_tier.A.length || 0} Tier A, {dashboardData?.cities_by_tier.B.length || 0}{" "}
                  Tier B, {dashboardData?.cities_by_tier.C.length || 0} Tier C
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Riders</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{totalRiders}</div>
                <p className="text-xs text-muted-foreground">Across {activeCities} cities</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Drivers</CardTitle>
                <Car className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{totalDrivers}</div>
                <p className="text-xs text-muted-foreground">
                  Supply/Demand Ratio: {totalRiders > 0 && totalDrivers > 0 ? (totalRiders / totalDrivers).toFixed(2) : "0.00"}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">System Status</CardTitle>
                <CheckCircle className="h-4 w-4 text-green-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">Active</div>
                <p className="text-xs text-muted-foreground">Event-driven pricing enabled</p>
              </CardContent>
            </Card>
          </div>

          {/* City Tiers Overview */}
          <Card>
            <CardHeader>
              <CardTitle>City Tiers Distribution</CardTitle>
              <CardDescription>Cities categorized by market characteristics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">Tier A - Metropolitan</span>
                    <Badge className={getTierColor("A")}>Premium</Badge>
                  </div>
                  <p className="text-2xl font-bold">{dashboardData?.cities_by_tier.A.length || 0}</p>
                  <p className="text-sm text-muted-foreground">High demand, premium pricing</p>
                  <p className="text-xs text-muted-foreground mt-2">
                    {dashboardData?.cities_by_tier.A.slice(0, 3).join(", ") || "None"}
                  </p>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">Tier B - Major</span>
                    <Badge className={getTierColor("B")}>Standard</Badge>
                  </div>
                  <p className="text-2xl font-bold">{dashboardData?.cities_by_tier.B.length || 0}</p>
                  <p className="text-sm text-muted-foreground">Moderate demand, standard pricing</p>
                  <p className="text-xs text-muted-foreground mt-2">
                    {dashboardData?.cities_by_tier.B.slice(0, 3).join(", ") || "None"}
                  </p>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">Tier C - Developing</span>
                    <Badge className={getTierColor("C")}>Affordable</Badge>
                  </div>
                  <p className="text-2xl font-bold">{dashboardData?.cities_by_tier.C.length || 0}</p>
                  <p className="text-sm text-muted-foreground">Lower demand, affordable pricing</p>
                  <p className="text-xs text-muted-foreground mt-2">
                    {dashboardData?.cities_by_tier.C.slice(0, 3).join(", ") || "None"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cities" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>City Statistics & Supply/Demand</CardTitle>
              <CardDescription>Real-time market conditions across all cities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {citiesData.length > 0 ? (
                  citiesData.map((city) => {
                    // Only show cities with actual users
                    const hasUsers = city.stats.riders > 0 || city.stats.drivers > 0;
                    if (!hasUsers) return null;
                    
                    return (
                    <div key={city.name} className="p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <MapPin className="w-4 h-4 text-primary" />
                            <h3 className="font-semibold text-lg">{city.name}</h3>
                            <Badge className={getTierColor(city.tier)}>Tier {city.tier}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {city.areas.length} areas: {city.areas.slice(0, 3).join(", ")}
                            {city.areas.length > 3 && "..."}
                          </p>
                        </div>
                        {city.samplePrice && (
                          <Badge className={getSurgeColor(city.samplePrice.surge_level)}>
                            {city.samplePrice.surge_level} Surge
                          </Badge>
                        )}
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                        <div>
                          <p className="text-xs text-muted-foreground">Riders</p>
                          <p className="text-xl font-bold">{city.stats.riders}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Drivers</p>
                          <p className="text-xl font-bold">{city.stats.drivers}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Ratio</p>
                          <p className="text-xl font-bold">{city.stats.ratio.toFixed(2)}</p>
                        </div>
                        {city.samplePrice && (
                          <div>
                            <p className="text-xs text-muted-foreground">Sample Price</p>
                            <p className="text-xl font-bold">₹{city.samplePrice.rider.toFixed(0)}</p>
                          </div>
                        )}
                      </div>

                      {city.stats.last_updated && (
                        <p className="text-xs text-muted-foreground mt-2">
                          Last updated: {new Date(city.stats.last_updated).toLocaleString()}
                        </p>
                      )}
                    </div>
                    );
                  })
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <MapPin className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-lg font-medium mb-2">No Active Cities</p>
                    <p>Register drivers or riders in the pricing app to see real-time data here.</p>
                    <Button 
                      className="mt-4" 
                      variant="outline"
                      onClick={() => window.location.href = "/pricing"}
                    >
                      Go to Pricing App
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pricing" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pricing Model Information</CardTitle>
              <CardDescription>ML model details and pricing factors</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="font-semibold mb-3">Model Architecture</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="w-5 h-5 text-primary" />
                        <span className="font-medium">Ensemble Model</span>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Combines Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks, and Elastic Net
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Target className="w-5 h-5 text-primary" />
                        <span className="font-medium">R² Score: 0.967</span>
                      </div>
                      <p className="text-sm text-muted-foreground">High accuracy in price prediction</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-3">Pricing Factors</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="p-3 border rounded-lg text-center">
                      <MapPin className="w-6 h-6 mx-auto mb-1 text-primary" />
                      <p className="text-xs font-medium">City Tier</p>
                      <p className="text-xs text-muted-foreground">A, B, or C</p>
                    </div>
                    <div className="p-3 border rounded-lg text-center">
                      <Users className="w-6 h-6 mx-auto mb-1 text-primary" />
                      <p className="text-xs font-medium">Supply/Demand</p>
                      <p className="text-xs text-muted-foreground">Riders/Drivers</p>
                    </div>
                    <div className="p-3 border rounded-lg text-center">
                      <Clock className="w-6 h-6 mx-auto mb-1 text-primary" />
                      <p className="text-xs font-medium">Time Factors</p>
                      <p className="text-xs text-muted-foreground">Peak hours, day</p>
                    </div>
                    <div className="p-3 border rounded-lg text-center">
                      <Navigation className="w-6 h-6 mx-auto mb-1 text-primary" />
                      <p className="text-xs font-medium">Area</p>
                      <p className="text-xs text-muted-foreground">Location within city</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-3">Event-Driven Updates</h3>
                  <Alert>
                    <Zap className="h-4 w-4" />
                    <AlertDescription>
                      Prices update automatically when new drivers or riders are added to the system. This ensures
                      real-time market responsiveness while minimizing computational overhead.
                    </AlertDescription>
                  </Alert>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>System Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">API Version</span>
                  <span className="font-medium">{dashboardData?.system_info.version || "1.0.0"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model Status</span>
                  <Badge variant="default">Active</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Update Mode</span>
                  <Badge variant="outline">Event-Driven</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Cities</span>
                  <span className="font-medium">{dashboardData?.total_cities || 0}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full" variant="outline" onClick={() => window.open("/pricing", "_blank")}>
                  <DollarSign className="w-4 h-4 mr-2" />
                  Open Pricing App
                </Button>
                <Button className="w-full" variant="outline" onClick={fetchDashboardData}>
                  <Activity className="w-4 h-4 mr-2" />
                  Refresh Data
                </Button>
                <Button className="w-full" variant="outline" onClick={() => window.open(`${API_BASE}/docs`, "_blank")}>
                  <BarChart3 className="w-4 h-4 mr-2" />
                  API Documentation
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>API Endpoints</CardTitle>
              <CardDescription>Available endpoints for the pricing system</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 font-mono text-sm">
                <div className="flex items-center gap-2 p-2 bg-muted rounded">
                  <span className="text-green-600">GET</span>
                  <span>/api/cities</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-muted rounded">
                  <span className="text-green-600">GET</span>
                  <span>/api/city-stats</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-muted rounded">
                  <span className="text-blue-600">POST</span>
                  <span>/api/register</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-muted rounded">
                  <span className="text-blue-600">POST</span>
                  <span>/api/price</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-muted rounded">
                  <span className="text-purple-600">WS</span>
                  <span>/ws/{"{city}"}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
