#!/usr/bin/env python3
"""
Script to train and save the city pricing model
Run this once to create the PKL file, then the API will use the saved model
"""

import os
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.city_pricing_model import CityPricingModel

def main():
    print("🚀 Training City Pricing Model")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"📁 Created {models_dir} directory")
    
    # Initialize model
    model = CityPricingModel()
    
    # Model file path
    model_path = os.path.join(models_dir, "city_pricing_model.pkl")
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"📋 Existing model found at {model_path}")
        choice = input("Do you want to retrain the model? (y/N): ").strip().lower()
        if choice != 'y':
            print("✅ Using existing model")
            return
    
    # Train the model
    print("🔧 Training model... (this may take a few moments)")
    results = model.train_model(model_path)
    
    print("\n🎯 Training Results:")
    print(f"   📊 Train R²: {results['train_score']:.3f}")
    print(f"   📈 Test R²: {results['test_score']:.3f}")
    
    print("\n🔝 Top Feature Importances:")
    sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"   {feature}: {importance:.3f}")
    
    # Test the model with a few predictions
    print("\n🧪 Testing Model Predictions:")
    test_cases = [
        {"city": "Mumbai", "user_type": "rider", "area": "Bandra", "riders": 120, "drivers": 45},
        {"city": "Delhi", "user_type": "driver", "area": "CP", "riders": 80, "drivers": 60},
        {"city": "Bangalore", "user_type": "rider", "area": "Koramangala", "riders": 100, "drivers": 40},
        {"city": "Agra", "user_type": "rider", "area": "Taj Ganj", "riders": 30, "drivers": 25},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            prediction = model.predict_price(
                city=test_case["city"],
                user_type=test_case["user_type"],
                area=test_case["area"],
                current_riders=test_case["riders"],
                current_drivers=test_case["drivers"]
            )
            
            print(f"   {i}. {test_case['city']} ({test_case['user_type']}): ₹{prediction['predicted_price']:.2f}")
            print(f"      Surge: {prediction['surge_level']} ({prediction['surge_multiplier']:.1f}x)")
            print(f"      Tier: {prediction['city_tier']}")
            
        except Exception as e:
            print(f"   {i}. {test_case['city']}: Error - {e}")
    
    print(f"\n✅ Model successfully trained and saved to {model_path}")
    print("🚀 You can now start the API server with: python api/pricing_api.py")
    
    # Show available cities
    print(f"\n🌍 Available Cities by Tier:")
    cities_by_tier = model.get_cities_by_tier()
    for tier, cities in cities_by_tier.items():
        tier_desc = {
            'A': 'Metropolitan Cities',
            'B': 'Major Cities', 
            'C': 'Developing Cities'
        }
        print(f"   Tier {tier} ({tier_desc[tier]}): {len(cities)} cities")
        print(f"      {', '.join(cities)}")

if __name__ == "__main__":
    main()
