#!/usr/bin/env python3
"""
Example script to demonstrate model training via API
"""

import requests
import json
import time

# API configuration
API_BASE = "http://localhost:5000/api"

def train_all_models():
    """Train all models via API"""
    print("🚀 Starting model training via API...")
    
    # Training request
    training_data = {
        "models": ["all"],  # Train all models
        "use_synthetic_data": True,
        "config": {
            "random_state": 42
        }
    }
    
    try:
        # Send training request
        response = requests.post(
            f"{API_BASE}/models/train",
            json=training_data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Training request successful!")
            print(f"📊 Training Results:")
            print(f"   Started: {result.get('started_at')}")
            print(f"   Completed: {result.get('completed_at')}")
            print(f"   Duration: {result.get('total_duration')}")
            print(f"   Success Rate: {result.get('success_rate')}")
            
            print(f"\n🎯 Models Trained:")
            for model in result.get('models_trained', []):
                print(f"   ✅ {model}")
            
            print(f"\n📈 Performance Metrics:")
            for model, metrics in result.get('performance_metrics', {}).items():
                print(f"   {model}:")
                for metric, value in metrics.items():
                    print(f"      {metric}: {value}")
            
            return result
        else:
            print(f"❌ Training failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Training request timed out (this is normal for large models)")
        return None
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

def save_models():
    """Save trained models"""
    print("\n💾 Saving models...")
    
    save_data = {
        "models": ["all"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/models/save",
            json=save_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Models saved successfully!")
            print(f"   Saved models: {result.get('saved_models', [])}")
            print(f"   Total saved: {result.get('total_saved', 0)}")
            return result
        else:
            print(f"❌ Save failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return None

def check_model_status():
    """Check current model status"""
    print("\n📋 Checking model status...")
    
    try:
        response = requests.get(f"{API_BASE}/models/status")
        
        if response.status_code == 200:
            result = response.json()
            print("📊 Model Status:")
            print(f"   Overall Health: {result.get('overall_health')}")
            print(f"   Active Models: {result.get('active_models')}/{result.get('total_models')}")
            
            for model_name, status in result.get('models', {}).items():
                print(f"   {model_name}: {status.get('status')} ({status.get('type')})")
            
            return result
        else:
            print(f"❌ Status check failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def test_trained_models():
    """Test some of the trained models"""
    print("\n🧪 Testing trained models...")
    
    # Test pricing model
    print("   Testing pricing model...")
    pricing_data = {
        "features": {
            "Number_of_Riders": 42,
            "Number_of_Drivers": 31,
            "Expected_Ride_Duration": 76,
            "Vehicle_Type_encoded": 1,
            "hour": 14,
            "is_peak_hour": 0
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/pricing/predict", json=pricing_data)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ Predicted price: ${result['prediction']:.2f}")
        else:
            print(f"      ❌ Pricing test failed")
    except Exception as e:
        print(f"      ❌ Pricing test error: {e}")
    
    # Test fraud detection
    print("   Testing fraud detection...")
    fraud_data = {
        "transaction": {
            "transaction_id": "TEST_001",
            "amount": 2500.0,
            "hour": 2,
            "merchant_category": "ATM",
            "location_type": "Travel",
            "is_new_device": 1,
            "distance_from_home_km": 200
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/fraud/analyze", json=fraud_data)
        if response.status_code == 200:
            result = response.json()
            analysis = result['fraud_analysis']
            print(f"      ✅ Fraud score: {analysis['fraud_score']:.3f}")
            print(f"      ✅ Risk level: {analysis['risk_level']}")
            print(f"      ✅ Action: {analysis['recommended_action']}")
        else:
            print(f"      ❌ Fraud test failed")
    except Exception as e:
        print(f"      ❌ Fraud test error: {e}")

def main():
    """Main execution"""
    print("🧠 SmartMarketer Model Training Demo")
    print("=" * 50)
    
    # Check initial status
    check_model_status()
    
    # Train models
    training_result = train_all_models()
    
    if training_result:
        # Save models
        save_models()
        
        # Check status after training
        time.sleep(2)
        check_model_status()
        
        # Test models
        test_trained_models()
        
        print(f"\n🎉 Model training and testing complete!")
        print(f"💡 You can now use the ML Dashboard at: http://localhost:5173/dashboard")
    else:
        print(f"\n❌ Training failed. Make sure the ML API is running at {API_BASE}")
        print(f"💡 Start the API with: cd ml-backend && python advanced_api.py")

if __name__ == "__main__":
    main()
