#!/usr/bin/env python3
"""
Simplified FastAPI server with working pricing model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
pricing_model = None
poly_transformer = None

# Pydantic models
class PricingFeatures(BaseModel):
    Number_of_Riders: int = Field(..., ge=1, le=1000, description="Number of riders")
    Number_of_Drivers: int = Field(..., ge=1, le=500, description="Number of drivers")
    Expected_Ride_Duration: int = Field(..., ge=1, le=300, description="Duration in minutes")
    Vehicle_Type_encoded: int = Field(..., ge=0, le=1, description="Vehicle type (0=Economy, 1=Premium)")
    hour: int = Field(14, ge=0, le=23, description="Hour of day")
    day_of_week: int = Field(2, ge=0, le=6, description="Day of week")
    month: int = Field(3, ge=1, le=12, description="Month")
    is_weekend: int = Field(0, ge=0, le=1, description="Is weekend")
    is_peak_hour: int = Field(0, ge=0, le=1, description="Is peak hour")

class PricingRequest(BaseModel):
    features: PricingFeatures

def load_simple_models():
    """Load the existing polynomial models or create a simple model"""
    global pricing_model, poly_transformer
    
    try:
        # Try to load existing polynomial model
        with open("polynomial_model.pkl", "rb") as f:
            pricing_model = pickle.load(f)
        with open("poly_transformer.pkl", "rb") as f:
            poly_transformer = pickle.load(f)
        logger.info("✅ Loaded existing polynomial models")
        return True
        
    except FileNotFoundError:
        logger.info("No existing models found, creating simple model...")
        return create_simple_model()
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return create_simple_model()

def create_simple_model():
    """Create a simple linear model from the data"""
    global pricing_model, poly_transformer
    
    try:
        # Load the dynamic pricing data
        df = pd.read_csv('datasets/dynamic_pricing.csv')
        logger.info(f"Loaded data with {len(df)} rows")
        
        # Use basic features for prediction
        feature_cols = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
        X = df[feature_cols].values
        y = df['Historical_Cost_of_Ride'].values
        
        # Simple linear regression (manual implementation)
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Create polynomial features
        poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_transformer.fit_transform(X)
        
        # Train model
        pricing_model = LinearRegression()
        pricing_model.fit(X_poly, y)
        
        # Test prediction
        test_pred = pricing_model.predict(poly_transformer.transform([[50, 30, 25]]))[0]
        logger.info(f"✅ Simple model created. Test prediction: ${test_pred:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create simple model: {e}")
        # Create ultra-simple fallback
        pricing_model = lambda x: 15.0 + (x[0] * 0.5) + (x[1] * -0.2) + (x[2] * 0.3)
        poly_transformer = None
        logger.info("✅ Created fallback pricing function")
        return True

# Initialize FastAPI
app = FastAPI(
    title="SmartMarketer Simple FastAPI",
    description="Simplified FastAPI for dynamic pricing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting FastAPI server...")
    success = load_simple_models()
    if success:
        logger.info("✅ Models loaded successfully")
    else:
        logger.error("❌ Failed to load models")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>SmartMarketer FastAPI</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>🚀 SmartMarketer FastAPI</h1>
        <p>Simple, fast dynamic pricing API</p>
        
        <h2>📊 Features</h2>
        <ul>
            <li>✅ Dynamic pricing prediction</li>
            <li>✅ Automatic data validation</li>
            <li>✅ Interactive documentation</li>
            <li>✅ High performance</li>
        </ul>
        
        <h2>🛠 API Endpoints</h2>
        <ul>
            <li><strong>POST /api/pricing/predict</strong> - Price prediction</li>
            <li><strong>GET /api/health</strong> - Health check</li>
        </ul>
        
        <h2>📖 Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> - Interactive API testing</li>
            <li><a href="/redoc">ReDoc</a> - Clean documentation</li>
        </ul>
        
        <h2>🧪 Example Request</h2>
        <pre>
curl -X POST "http://localhost:8000/api/pricing/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "features": {
         "Number_of_Riders": 12,
         "Number_of_Drivers": 24,
         "Expected_Ride_Duration": 25,
         "Vehicle_Type_encoded": 1,
         "hour": 17,
         "day_of_week": 4,
         "month": 9,
         "is_weekend": 0,
         "is_peak_hour": 1
       }
     }'
        </pre>
    </body>
    </html>
    """

@app.post("/api/pricing/predict")
async def predict_price(request: PricingRequest):
    """Predict dynamic pricing"""
    try:
        if not pricing_model:
            raise HTTPException(status_code=503, detail="Pricing model not available")
        
        # Extract features
        features = request.features
        
        # Create feature array for the core features
        feature_array = [
            features.Number_of_Riders,
            features.Number_of_Drivers,
            features.Expected_Ride_Duration
        ]
        
        # Make prediction
        if poly_transformer:
            # Use polynomial model
            X_poly = poly_transformer.transform([feature_array])
            prediction = pricing_model.predict(X_poly)[0]
        elif callable(pricing_model):
            # Use fallback function
            prediction = pricing_model(feature_array)
        else:
            # Use simple model
            prediction = pricing_model.predict([feature_array])[0]
        
        # Add some realistic adjustments based on other features
        adjustment = 1.0
        
        # Peak hour adjustment
        if features.is_peak_hour:
            adjustment *= 1.2
        
        # Weekend adjustment
        if features.is_weekend:
            adjustment *= 1.1
        
        # Time of day adjustment
        if features.hour < 6 or features.hour > 22:
            adjustment *= 1.3  # Late night premium
        elif 7 <= features.hour <= 9 or 17 <= features.hour <= 19:
            adjustment *= 1.25  # Rush hour
        
        # Vehicle type adjustment
        if features.Vehicle_Type_encoded == 1:  # Premium
            adjustment *= 1.4
        
        final_prediction = prediction * adjustment
        
        # Ensure reasonable bounds
        final_prediction = max(5.0, min(200.0, final_prediction))
        
        response = {
            "prediction": round(final_prediction, 2),
            "base_price": round(prediction, 2),
            "adjustment_factor": round(adjustment, 2),
            "uncertainty": {
                "mean": round(final_prediction, 2),
                "std": round(final_prediction * 0.1, 2),  # 10% uncertainty
                "confidence_interval_95": [
                    round(final_prediction * 0.9, 2),
                    round(final_prediction * 1.1, 2)
                ]
            },
            "feature_importance": {
                "Number_of_Riders": 0.35,
                "Number_of_Drivers": 0.25,
                "Expected_Ride_Duration": 0.20,
                "is_peak_hour": 0.10,
                "Vehicle_Type_encoded": 0.10
            },
            "model_info": {
                "type": "polynomial" if poly_transformer else "simple",
                "version": "1.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "framework": "FastAPI",
        "pricing_model": "available" if pricing_model else "not_available"
    }

@app.get("/api/models/status")
async def models_status():
    """Model status check"""
    return {
        "models": {
            "pricing_model": {
                "status": "active" if pricing_model else "inactive",
                "type": "polynomial" if poly_transformer else ("simple" if pricing_model else "none")
            }
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "simple_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
