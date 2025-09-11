#!/usr/bin/env python3
"""
Quick test script for FastAPI server
"""

import requests
import json

def test_pricing_prediction():
    """Test the pricing prediction endpoint that was failing"""
    
    print("🧪 Testing FastAPI Pricing Prediction...")
    
    # The exact request that was failing
    test_data = {
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
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/pricing/predict",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            json=test_data,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Pricing prediction working")
            print(f"   Predicted Price: ${result['prediction']:.2f}")
            print(f"   Uncertainty: ±{result['uncertainty']['std']:.2f}")
            print(f"   Model Type: {result['model_info']['type']}")
            return True
        else:
            print("❌ FAILED!")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI server")
        print("💡 Make sure to start FastAPI with: python advanced_fastapi.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_check():
    """Test basic health check"""
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ FastAPI Health Check: {health.get('status')}")
            print(f"   Models loaded: {health.get('models_loaded', 0)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except:
        return False

def test_interactive_docs():
    """Test if interactive docs are available"""
    try:
        # Test Swagger UI
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print("✅ Swagger UI available at: http://localhost:8000/docs")
        
        # Test ReDoc
        response = requests.get("http://localhost:8000/redoc")
        if response.status_code == 200:
            print("✅ ReDoc available at: http://localhost:8000/redoc")
        
        return True
    except:
        print("❌ Interactive docs not accessible")
        return False

def main():
    print("⚡ FastAPI Quick Test")
    print("=" * 40)
    
    # Test health first
    if not test_health_check():
        print("❌ FastAPI server is not running")
        print("💡 Start with: python advanced_fastapi.py")
        return
    
    # Test interactive docs
    test_interactive_docs()
    
    # Test the main pricing endpoint
    success = test_pricing_prediction()
    
    if success:
        print("\n🎉 FastAPI is working correctly!")
        print("🔗 Try these URLs:")
        print("   📖 API Docs: http://localhost:8000/docs")
        print("   🎯 Dashboard: http://localhost:5173/dashboard")
    else:
        print("\n❌ FastAPI needs troubleshooting")

if __name__ == "__main__":
    main()
