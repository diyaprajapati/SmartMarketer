import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  TrendingUp,
  Users,
  Shield,
  BarChart3,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  DollarSign,
  Target,
  Brain,
  Zap,
} from "lucide-react";

interface ModelStatus {
  status: "active" | "inactive";
  type: string;
  last_updated: string | null;
  performance: string;
}

interface DashboardData {
  system_status: {
    uptime: string;
    models_active: number;
    total_models: number;
    system_health: "healthy" | "degraded";
  };
  request_statistics: {
    total_requests: number;
    requests_24h: number;
    success_rate: number;
    avg_response_size: number;
  };
  endpoint_usage: Record<string, number>;
  model_status: Record<string, ModelStatus>;
  performance_metrics: Record<string, string>;
}

export const MLDashboard = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState("overview");
  const [testResults, setTestResults] = useState<any>(null);

  // API Base URL - update this to match your backend
  const API_BASE = "http://localhost:5000/api";

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE}/analytics/dashboard`);
      if (!response.ok) throw new Error("Failed to fetch dashboard data");
      const data = await response.json();
      setDashboardData(data.dashboard);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const testPricingModel = async () => {
    try {
      const testData = {
        features: {
          Number_of_Riders: 42,
          Number_of_Drivers: 31,
          Expected_Ride_Duration: 76,
          Vehicle_Type_encoded: 1,
          hour: 14,
          day_of_week: 2,
          month: 3,
          is_weekend: 0,
          is_peak_hour: 0,
        },
      };

      const response = await fetch(`${API_BASE}/pricing/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testData),
      });

      if (!response.ok) throw new Error("Pricing prediction failed");
      const result = await response.json();
      setTestResults({ type: "pricing", data: result });
    } catch (err) {
      setTestResults({ type: "error", data: err instanceof Error ? err.message : "Unknown error" });
    }
  };

  const testFraudDetection = async () => {
    try {
      const testData = {
        transaction: {
          transaction_id: "TEST_001",
          amount: 2500.0,
          hour: 2,
          merchant_category: "ATM",
          location_type: "Travel",
          days_since_last_transaction: 0.1,
          transactions_last_hour: 3,
          transactions_last_day: 8,
          is_new_device: 1,
          is_new_ip: 1,
          distance_from_home_km: 200,
          is_weekend: 1,
        },
      };

      const response = await fetch(`${API_BASE}/fraud/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testData),
      });

      if (!response.ok) throw new Error("Fraud analysis failed");
      const result = await response.json();
      setTestResults({ type: "fraud", data: result });
    } catch (err) {
      setTestResults({ type: "error", data: err instanceof Error ? err.message : "Unknown error" });
    }
  };

  const testDemandForecasting = async () => {
    try {
      const testData = {
        steps: 24,
        method: "ensemble",
      };

      const response = await fetch(`${API_BASE}/demand/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testData),
      });

      if (!response.ok) throw new Error("Demand forecasting failed");
      const result = await response.json();
      setTestResults({ type: "demand", data: result });
    } catch (err) {
      setTestResults({ type: "error", data: err instanceof Error ? err.message : "Unknown error" });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Brain className="w-16 h-16 mx-auto mb-4 animate-pulse text-primary" />
          <p className="text-lg font-medium">Loading ML Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-6 py-8">
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Failed to load dashboard: {error}. Make sure the ML API is running on http://localhost:5000
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">🧠 SmartMarketer ML Dashboard</h1>
        <p className="text-muted-foreground">Real-time analytics and machine learning model monitoring</p>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="testing">Testing</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* System Status */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">System Health</CardTitle>
                {dashboardData?.system_status.system_health === "healthy" ? (
                  <CheckCircle className="h-4 w-4 text-green-600" />
                ) : (
                  <XCircle className="h-4 w-4 text-red-600" />
                )}
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {dashboardData?.system_status.system_health === "healthy" ? "✅" : "⚠️"}
                </div>
                <p className="text-xs text-muted-foreground">
                  {dashboardData?.system_status.models_active}/{dashboardData?.system_status.total_models} models active
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {dashboardData?.request_statistics.total_requests.toLocaleString()}
                </div>
                <p className="text-xs text-muted-foreground">
                  {dashboardData?.request_statistics.requests_24h} in last 24h
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{dashboardData?.request_statistics.success_rate.toFixed(1)}%</div>
                <p className="text-xs text-muted-foreground">Last 24 hours</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Uptime</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{dashboardData?.system_status.uptime.split(",")[0]}</div>
                <p className="text-xs text-muted-foreground">System running</p>
              </CardContent>
            </Card>
          </div>

          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>Real-time model performance indicators</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {dashboardData?.performance_metrics &&
                  Object.entries(dashboardData.performance_metrics).map(([key, value]) => (
                    <div key={key} className="text-center p-4 border rounded-lg">
                      <div className="text-lg font-semibold text-primary">{value}</div>
                      <div className="text-sm text-muted-foreground capitalize">{key.replace(/_/g, " ")}</div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="models" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {dashboardData?.model_status &&
              Object.entries(dashboardData.model_status).map(([modelName, status]) => (
                <Card key={modelName}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="capitalize">{modelName.replace(/_/g, " ")}</CardTitle>
                      <Badge variant={status.status === "active" ? "default" : "destructive"}>{status.status}</Badge>
                    </div>
                    <CardDescription>{status.type}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Performance:</span>
                        <span className="text-sm font-medium">{status.performance}</span>
                      </div>
                      {status.last_updated && (
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">Last Updated:</span>
                          <span className="text-sm font-medium">
                            {new Date(status.last_updated).toLocaleDateString()}
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
          </div>
        </TabsContent>

        <TabsContent value="testing" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Testing</CardTitle>
              <CardDescription>Test individual ML models with sample data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <Button onClick={testPricingModel} className="flex items-center gap-2">
                  <DollarSign className="w-4 h-4" />
                  Test Pricing Model
                </Button>
                <Button onClick={testFraudDetection} className="flex items-center gap-2">
                  <Shield className="w-4 h-4" />
                  Test Fraud Detection
                </Button>
                <Button onClick={testDemandForecasting} className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Test Demand Forecast
                </Button>
              </div>

              {testResults && (
                <Card className="bg-muted/50">
                  <CardHeader>
                    <CardTitle className="text-lg">Test Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-sm overflow-auto max-h-96 bg-background p-4 rounded border">
                      {JSON.stringify(testResults.data, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          {/* Endpoint Usage */}
          <Card>
            <CardHeader>
              <CardTitle>API Endpoint Usage</CardTitle>
              <CardDescription>Requests in the last 24 hours</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {dashboardData?.endpoint_usage &&
                  Object.entries(dashboardData.endpoint_usage)
                    .sort(([, a], [, b]) => b - a)
                    .map(([endpoint, count]) => (
                      <div key={endpoint} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Zap className="w-4 h-4 text-primary" />
                          <span className="font-mono text-sm">{endpoint}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{count} requests</Badge>
                          <div className="w-24 bg-muted rounded h-2">
                            <div
                              className="bg-primary h-2 rounded"
                              style={{
                                width: `${(count / Math.max(...Object.values(dashboardData.endpoint_usage))) * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Avg Response</p>
                    <p className="text-2xl font-bold">
                      {dashboardData?.request_statistics.avg_response_size
                        ? (dashboardData.request_statistics.avg_response_size / 1024).toFixed(1) + "KB"
                        : "0KB"}
                    </p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Models Active</p>
                    <p className="text-2xl font-bold">
                      {dashboardData?.system_status.models_active}/{dashboardData?.system_status.total_models}
                    </p>
                  </div>
                  <Brain className="h-8 w-8 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Health Status</p>
                    <p className="text-2xl font-bold">
                      {dashboardData?.system_status.system_health === "healthy" ? "🟢" : "🟡"}
                    </p>
                  </div>
                  <Activity className="h-8 w-8 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">API Version</p>
                    <p className="text-2xl font-bold">v2.0</p>
                  </div>
                  <Zap className="h-8 w-8 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
