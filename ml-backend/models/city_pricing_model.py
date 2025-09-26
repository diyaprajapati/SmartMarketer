"""
City-based Dynamic Pricing Model
Handles pricing for different cities with tiers
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

@dataclass
class CityInfo:
    """City information with pricing tier"""
    name: str
    tier: str  # 'A', 'B', 'C'
    base_multiplier: float
    peak_hours: List[int]
    surge_areas: List[str]

@dataclass
class UserProfile:
    """User profile for pricing calculation"""
    user_type: str  # 'driver' or 'rider'
    user_id: str
    city: str
    location_area: str
    rating: float
    trips_completed: int

class CityPricingModel:
    """Dynamic pricing model for city-based ride sharing"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        self.area_encoder = LabelEncoder()
        self.user_type_encoder = LabelEncoder()
        self.is_trained = False
        
        # Define cities with tiers
        self.cities = {
            # Tier A - Metropolitan cities
            'Mumbai': CityInfo('Mumbai', 'A', 1.5, [8, 9, 18, 19, 20], ['Bandra', 'Andheri', 'Lower Parel']),
            'Delhi': CityInfo('Delhi', 'A', 1.4, [8, 9, 18, 19, 20], ['CP', 'Gurgaon', 'Noida']),
            'Bangalore': CityInfo('Bangalore', 'A', 1.3, [8, 9, 18, 19, 20], ['Koramangala', 'Whitefield', 'Electronic City']),
            'Chennai': CityInfo('Chennai', 'A', 1.2, [8, 9, 18, 19, 20], ['T Nagar', 'Anna Nagar', 'OMR']),
            'Kolkata': CityInfo('Kolkata', 'A', 1.1, [8, 9, 18, 19, 20], ['Salt Lake', 'Park Street', 'New Market']),
            
            # Tier B - Major cities
            'Hyderabad': CityInfo('Hyderabad', 'B', 1.0, [8, 9, 18, 19], ['Hitech City', 'Gachibowli', 'Jubilee Hills']),
            'Pune': CityInfo('Pune', 'B', 0.9, [8, 9, 18, 19], ['Koregaon Park', 'Hinjewadi', 'Viman Nagar']),
            'Ahmedabad': CityInfo('Ahmedabad', 'B', 0.8, [8, 9, 18, 19], ['SG Highway', 'Satellite', 'Navrangpura']),
            'Jaipur': CityInfo('Jaipur', 'B', 0.8, [8, 9, 18, 19], ['Pink City', 'Malviya Nagar', 'Vaishali Nagar']),
            'Lucknow': CityInfo('Lucknow', 'B', 0.7, [8, 9, 18, 19], ['Hazratganj', 'Gomti Nagar', 'Indira Nagar']),
            
            # Tier C - Developing cities
            'Bhopal': CityInfo('Bhopal', 'C', 0.6, [8, 9, 18], ['New Market', 'MP Nagar', 'Arera Colony']),
            'Indore': CityInfo('Indore', 'C', 0.6, [8, 9, 18], ['Vijay Nagar', 'Palasia', 'Rajwada']),
            'Chandigarh': CityInfo('Chandigarh', 'C', 0.7, [8, 9, 18], ['Sector 17', 'Sector 35', 'Elante Mall']),
            'Coimbatore': CityInfo('Coimbatore', 'C', 0.5, [8, 9, 18], ['RS Puram', 'Peelamedu', 'Saibaba Colony']),
            'Kochi': CityInfo('Kochi', 'C', 0.6, [8, 9, 18], ['Marine Drive', 'Kakkanad', 'Edapally']),
            'Nagpur': CityInfo('Nagpur', 'C', 0.5, [8, 9, 18], ['Sitabuldi', 'Dharampeth', 'Sadar']),
            'Vadodara': CityInfo('Vadodara', 'C', 0.5, [8, 9, 18], ['Alkapuri', 'Fatehgunj', 'Sayajigunj']),
            'Surat': CityInfo('Surat', 'C', 0.5, [8, 9, 18], ['Adajan', 'Vesu', 'Citylight']),
            'Visakhapatnam': CityInfo('Visakhapatnam', 'C', 0.5, [8, 9, 18], ['MVP Colony', 'Dwaraka Nagar', 'Gajuwaka']),
            'Agra': CityInfo('Agra', 'C', 0.4, [8, 9, 18], ['Taj Ganj', 'Sadar Bazaar', 'Civil Lines'])
        }
        
        self.base_price = 50.0  # Base price in INR
        
    def get_cities_by_tier(self) -> Dict[str, List[str]]:
        """Get cities grouped by tier"""
        tiers = {'A': [], 'B': [], 'C': []}
        for city_name, city_info in self.cities.items():
            tiers[city_info.tier].append(city_name)
        return tiers
    
    def get_city_areas(self, city_name: str) -> List[str]:
        """Get areas for a specific city"""
        if city_name in self.cities:
            return self.cities[city_name].surge_areas
        return []
    
    def create_training_data(self) -> pd.DataFrame:
        """Create synthetic training data for the model"""
        np.random.seed(42)
        data = []
        
        cities = list(self.cities.keys())
        user_types = ['driver', 'rider']
        
        # Generate 5000 training samples
        for _ in range(5000):
            city = np.random.choice(cities)
            city_info = self.cities[city]
            user_type = np.random.choice(user_types)
            area = np.random.choice(city_info.surge_areas)
            
            # Time features
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            is_weekend = 1 if day_of_week >= 5 else 0
            is_peak_hour = 1 if hour in city_info.peak_hours else 0
            
            # Supply and demand
            riders_count = np.random.randint(10, 200)
            drivers_count = np.random.randint(5, 100)
            demand_supply_ratio = riders_count / max(drivers_count, 1)
            
            # User features
            user_rating = np.random.uniform(3.0, 5.0)
            trips_completed = np.random.randint(0, 1000)
            
            # Weather and events (synthetic)
            weather_impact = np.random.uniform(0.8, 1.3)
            event_multiplier = np.random.choice([1.0, 1.2, 1.5, 2.0], p=[0.7, 0.15, 0.1, 0.05])
            
            # Calculate price
            base_price = self.base_price
            city_multiplier = city_info.base_multiplier
            peak_multiplier = 1.3 if is_peak_hour else 1.0
            weekend_multiplier = 1.1 if is_weekend else 1.0
            demand_multiplier = min(3.0, max(0.5, demand_supply_ratio / 2))
            
            final_price = (base_price * city_multiplier * peak_multiplier * 
                          weekend_multiplier * demand_multiplier * 
                          weather_impact * event_multiplier)
            
            data.append({
                'city': city,
                'user_type': user_type,
                'area': area,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak_hour,
                'riders_count': riders_count,
                'drivers_count': drivers_count,
                'demand_supply_ratio': demand_supply_ratio,
                'user_rating': user_rating,
                'trips_completed': trips_completed,
                'weather_impact': weather_impact,
                'event_multiplier': event_multiplier,
                'price': final_price
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, save_path: str = "models/city_pricing_model.pkl"):
        """Train the pricing model and save to pickle file"""
        print("🔧 Training city pricing model...")
        
        # Create training data
        df = self.create_training_data()
        
        # Prepare features
        categorical_features = ['city', 'user_type', 'area']
        
        # Encode categorical features
        df_encoded = df.copy()
        df_encoded['city_encoded'] = self.city_encoder.fit_transform(df['city'])
        df_encoded['user_type_encoded'] = self.user_type_encoder.fit_transform(df['user_type'])
        df_encoded['area_encoded'] = self.area_encoder.fit_transform(df['area'])
        
        # Select features for training
        feature_columns = [
            'city_encoded', 'user_type_encoded', 'area_encoded', 'hour', 
            'day_of_week', 'is_weekend', 'is_peak_hour', 'riders_count', 
            'drivers_count', 'demand_supply_ratio', 'user_rating', 
            'trips_completed', 'weather_impact', 'event_multiplier'
        ]
        
        X = df_encoded[feature_columns]
        y = df_encoded['price']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=15
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Model trained successfully!")
        print(f"   Train R²: {train_score:.3f}")
        print(f"   Test R²: {test_score:.3f}")
        
        self.is_trained = True
        
        # Save model
        self.save_model(save_path)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_columns, self.model.feature_importances_))
        }
    
    def save_model(self, file_path: str):
        """Save the trained model to pickle file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'city_encoder': self.city_encoder,
            'area_encoder': self.area_encoder,
            'user_type_encoder': self.user_type_encoder,
            'cities': self.cities,
            'base_price': self.base_price,
            'is_trained': self.is_trained,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {file_path}")
    
    def load_model(self, file_path: str) -> bool:
        """Load model from pickle file"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ Model file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.city_encoder = model_data['city_encoder']
            self.area_encoder = model_data['area_encoder']
            self.user_type_encoder = model_data['user_type_encoder']
            self.cities = model_data['cities']
            self.base_price = model_data['base_price']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ Model loaded from {file_path}")
            print(f"   Trained at: {model_data.get('trained_at', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_price(self, 
                     city: str,
                     user_type: str,
                     area: str,
                     current_riders: int,
                     current_drivers: int,
                     user_rating: float = 4.0,
                     trips_completed: int = 50) -> Dict:
        """Predict dynamic price for given parameters"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Please train or load a model first.")
        
        # Get current time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Check if peak hour for this city
        city_info = self.cities.get(city)
        if not city_info:
            raise ValueError(f"City {city} not found")
        
        is_peak_hour = 1 if hour in city_info.peak_hours else 0
        
        # Calculate demand supply ratio
        demand_supply_ratio = current_riders / max(current_drivers, 1)
        
        # Create feature array
        try:
            city_encoded = self.city_encoder.transform([city])[0]
            user_type_encoded = self.user_type_encoder.transform([user_type])[0]
            area_encoded = self.area_encoder.transform([area])[0]
        except ValueError as e:
            # Handle unseen categories
            city_encoded = 0
            user_type_encoded = 0 if user_type == 'driver' else 1
            area_encoded = 0
        
        # Synthetic features (fixed to ensure deterministic pricing per counts)
        weather_impact = 1.0
        event_multiplier = 1.0
        
        features = np.array([[
            city_encoded, user_type_encoded, area_encoded, hour,
            day_of_week, is_weekend, is_peak_hour, current_riders,
            current_drivers, demand_supply_ratio, user_rating,
            trips_completed, weather_impact, event_multiplier
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predicted_price = self.model.predict(features_scaled)[0]
        
        # Calculate confidence and surge info
        surge_multiplier = demand_supply_ratio / 2
        surge_level = "Low" if surge_multiplier < 1.2 else "Medium" if surge_multiplier < 2.0 else "High"
        
        return {
            'predicted_price': round(predicted_price, 2),
            'base_price': self.base_price,
            'city_tier': city_info.tier,
            'surge_multiplier': round(surge_multiplier, 2),
            'surge_level': surge_level,
            'is_peak_hour': bool(is_peak_hour),
            'demand_supply_ratio': round(demand_supply_ratio, 2),
            'timestamp': datetime.now().isoformat(),
            'city_info': {
                'name': city,
                'tier': city_info.tier,
                'base_multiplier': city_info.base_multiplier
            }
        }

# Utility functions
def get_available_cities() -> Dict[str, List[str]]:
    """Get all available cities grouped by tier"""
    model = CityPricingModel()
    return model.get_cities_by_tier()

def get_city_areas(city_name: str) -> List[str]:
    """Get areas for a specific city"""
    model = CityPricingModel()
    return model.get_city_areas(city_name)

if __name__ == "__main__":
    # Train and save model if run directly
    model = CityPricingModel()
    
    # Try to load existing model
    model_path = "models/city_pricing_model.pkl"
    if not model.load_model(model_path):
        print("No existing model found. Training new model...")
        model.train_model(model_path)
    
    # Test prediction
    test_prediction = model.predict_price(
        city="Mumbai",
        user_type="rider",
        area="Bandra",
        current_riders=120,
        current_drivers=45,
        user_rating=4.2,
        trips_completed=75
    )
    
    print(f"\n🧪 Test Prediction:")
    print(f"   Price: ₹{test_prediction['predicted_price']}")
    print(f"   Surge: {test_prediction['surge_level']} ({test_prediction['surge_multiplier']}x)")
    print(f"   Peak Hour: {test_prediction['is_peak_hour']}")
