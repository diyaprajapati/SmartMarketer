"""
Dynamic Pricing API with real-time updates
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import asyncio
import json
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.city_pricing_model import CityPricingModel, get_available_cities, get_city_areas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
pricing_model = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.city_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, city: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if city:
            if city not in self.city_subscribers:
                self.city_subscribers[city] = []
            self.city_subscribers[city].append(websocket)
        
        logger.info(f"WebSocket connected. City: {city}. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from city subscribers
        for city, subscribers in self.city_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_city(self, city: str, data: dict):
        """Send data to all subscribers of a specific city"""
        if city in self.city_subscribers:
            disconnected = []
            for websocket in self.city_subscribers[city]:
                try:
                    await websocket.send_json(data)
                except:
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast(self, data: dict):
        """Broadcast data to all connected clients"""
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(data)
            except:
                disconnected.append(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)

manager = ConnectionManager()

# Pydantic models
class UserRegistration(BaseModel):
    user_type: str = Field(..., pattern="^(driver|rider)$", description="User type: driver or rider")
    user_id: str = Field(..., min_length=1, description="Unique user ID")
    name: str = Field(..., min_length=1, description="User name")
    phone: str = Field(..., pattern="^[0-9]{10}$", description="10-digit phone number")
    city: str = Field(..., description="Selected city")
    area: str = Field(..., description="Selected area in city")
    rating: Optional[float] = Field(4.0, ge=1.0, le=5.0, description="User rating")
    trips_completed: Optional[int] = Field(0, ge=0, description="Number of trips completed")

class PriceRequest(BaseModel):
    city: str = Field(..., description="City name")
    user_type: str = Field(..., pattern="^(driver|rider)$", description="User type")
    area: str = Field(..., description="Area in city")
    current_riders: int = Field(..., ge=1, le=1000, description="Current riders count")
    current_drivers: int = Field(..., ge=1, le=500, description="Current drivers count")
    user_rating: Optional[float] = Field(4.0, ge=1.0, le=5.0, description="User rating")
    trips_completed: Optional[int] = Field(50, ge=0, description="Trips completed")

class CitySupplyDemand(BaseModel):
    city: str
    current_riders: int = Field(..., ge=0)
    current_drivers: int = Field(..., ge=0)

# Global supply/demand tracking
city_stats = {}

def initialize_model():
    """Initialize or load the pricing model"""
    global pricing_model
    
    pricing_model = CityPricingModel()
    model_path = "models/city_pricing_model.pkl"
    
    # Try to load existing model
    if not pricing_model.load_model(model_path):
        logger.info("No existing model found. Training new model...")
        pricing_model.train_model(model_path)
    
    logger.info("✅ Pricing model initialized")

# FastAPI app
app = FastAPI(
    title="Dynamic Pricing API",
    description="Real-time dynamic pricing for ride-sharing with city-based tiers",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()
    # Start background task for price updates
    asyncio.create_task(price_update_task())

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Dynamic Pricing API",
        "version": "1.0.0",
        "endpoints": {
            "cities": "/api/cities",
            "areas": "/api/cities/{city}/areas",
            "register": "/api/register",
            "price": "/api/price",
            "websocket": "/ws/{city}"
        }
    }

@app.get("/api/cities")
async def get_cities():
    """Get all available cities grouped by tier"""
    try:
        cities_by_tier = get_available_cities()
        return {
            "cities": cities_by_tier,
            "total_cities": sum(len(cities) for cities in cities_by_tier.values()),
            "tiers": {
                "A": "Metropolitan Cities (High demand, premium pricing)",
                "B": "Major Cities (Moderate demand, standard pricing)",  
                "C": "Developing Cities (Lower demand, affordable pricing)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cities/{city}/areas")
async def get_city_areas_endpoint(city: str):
    """Get areas for a specific city"""
    try:
        areas = get_city_areas(city)
        if not areas:
            raise HTTPException(status_code=404, detail=f"City {city} not found")
        
        return {
            "city": city,
            "areas": areas,
            "total_areas": len(areas)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
async def register_user(user: UserRegistration):
    """Register a new user (driver or rider)"""
    try:
        # Validate city and area
        available_cities = get_available_cities()
        all_cities = []
        for tier_cities in available_cities.values():
            all_cities.extend(tier_cities)
        
        if user.city not in all_cities:
            raise HTTPException(status_code=400, detail=f"City {user.city} not available")
        
        city_areas = get_city_areas(user.city)
        if user.area not in city_areas:
            raise HTTPException(status_code=400, detail=f"Area {user.area} not available in {user.city}")
        
        # In a real app, save to database
        user_data = user.dict()
        user_data['registered_at'] = datetime.now().isoformat()
        user_data['status'] = 'active'
        
        return {
            "message": f"{user.user_type.title()} registered successfully",
            "user_id": user.user_id,
            "city": user.city,
            "area": user.area,
            "registration_time": user_data['registered_at']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/price")
async def get_price(request: PriceRequest):
    """Get current dynamic price"""
    try:
        if not pricing_model:
            raise HTTPException(status_code=503, detail="Pricing model not available")
        
        # Update city stats
        global city_stats
        city_stats[request.city] = {
            'riders': request.current_riders,
            'drivers': request.current_drivers,
            'last_updated': datetime.now().isoformat()
        }
        
        # Get price prediction
        price_data = pricing_model.predict_price(
            city=request.city,
            user_type=request.user_type,
            area=request.area,
            current_riders=request.current_riders,
            current_drivers=request.current_drivers,
            user_rating=request.user_rating,
            trips_completed=request.trips_completed
        )
        
        # Add additional info
        price_data['request_time'] = datetime.now().isoformat()
        price_data['user_type'] = request.user_type
        price_data['area'] = request.area
        
        return price_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Price calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/supply-demand")
async def update_supply_demand(data: CitySupplyDemand):
    """Update supply and demand for a city"""
    try:
        global city_stats
        city_stats[data.city] = {
            'riders': data.current_riders,
            'drivers': data.current_drivers,
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            "message": "Supply/demand updated",
            "city": data.city,
            "riders": data.current_riders,
            "drivers": data.current_drivers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/city-stats")
async def get_city_stats():
    """Get current supply/demand stats for all cities"""
    return {
        "stats": city_stats,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/{city}")
async def websocket_endpoint(websocket: WebSocket, city: str):
    """WebSocket endpoint for real-time price updates"""
    await manager.connect(websocket, city)
    try:
        while True:
            # Wait for client messages (to keep connection alive)
            data = await websocket.receive_text()
            logger.info(f"Received from {city}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def price_update_task():
    """Background task to send price updates every 10 seconds"""
    await asyncio.sleep(5)  # Wait for startup
    
    while True:
        try:
            # Generate price updates for cities with active connections
            for city in manager.city_subscribers.keys():
                if len(manager.city_subscribers[city]) > 0:
                    # Get current stats or use defaults
                    stats = city_stats.get(city, {'riders': 50, 'drivers': 30})
                    
                    # Add some random variation to simulate real-time changes
                    import random
                    riders_variation = random.randint(-10, 15)
                    drivers_variation = random.randint(-5, 8)
                    
                    current_riders = max(10, stats['riders'] + riders_variation)
                    current_drivers = max(5, stats['drivers'] + drivers_variation)
                    
                    # Get areas for this city
                    areas = get_city_areas(city)
                    if areas:
                        area = random.choice(areas)
                        
                        # Calculate price for both user types
                        rider_price = pricing_model.predict_price(
                            city=city,
                            user_type="rider",
                            area=area,
                            current_riders=current_riders,
                            current_drivers=current_drivers
                        )
                        
                        driver_price = pricing_model.predict_price(
                            city=city,
                            user_type="driver",
                            area=area,
                            current_riders=current_riders,
                            current_drivers=current_drivers
                        )
                        
                        # Send update to city subscribers
                        update_data = {
                            "type": "price_update",
                            "city": city,
                            "area": area,
                            "timestamp": datetime.now().isoformat(),
                            "supply_demand": {
                                "riders": current_riders,
                                "drivers": current_drivers,
                                "ratio": round(current_riders / max(current_drivers, 1), 2)
                            },
                            "pricing": {
                                "rider": rider_price,
                                "driver": driver_price
                            }
                        }
                        
                        await manager.send_to_city(city, update_data)
                        
                        # Update global stats
                        city_stats[city] = {
                            'riders': current_riders,
                            'drivers': current_drivers,
                            'last_updated': datetime.now().isoformat()
                        }
            
            # Wait 10 seconds before next update
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Price update task error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
