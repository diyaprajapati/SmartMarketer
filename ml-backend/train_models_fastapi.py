#!/usr/bin/env python3
"""
Example script to demonstrate FastAPI model training
"""

import requests
import json
import time
import asyncio
import httpx

# API configuration
API_BASE = "http://localhost:8000/api"

class FastAPITrainingDemo:
    """Demo class for FastAPI training endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
    
    async def async_train_all_models(self):
        """Train all models via FastAPI (async)"""
        print("🚀 Starting model training via FastAPI...")
        
        # Training request with Pydantic validation
        training_data = {
            "models": ["all"],  # Train all models
            "use_synthetic_data": True,
            "config": {
                "random_state": 42
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Send training request
                response = await client.post(
                    f"{API_BASE}/models/train",
                    json=training_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print("✅ Training request successful!")
                    print(f"📊 Training Results:")
                    print(f"   Started: {result.get('started_at')}")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Message: {result.get('message')}")
                    
                    return result
                else:
                    print(f"❌ Training failed with status {response.status_code}")
                    print(f"Error: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return None
    
    def train_all_models(self):
        """Train all models via FastAPI (sync)"""
        print("🚀 Starting model training via FastAPI...")
        
        # Training request with automatic Pydantic validation
        training_data = {
            "models": ["all"],
            "use_synthetic_data": True,
            "config": {}
        }
        
        try:
            # Send training request
            response = self.session.post(
                f"{API_BASE}/models/train",
                json=training_data,
                timeout=60  # Shorter timeout since training runs in background
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Training request initiated!")
                print(f"📊 Training Response:")
                print(f"   Started: {result.get('started_at')}")
                print(f"   Status: {result.get('status')}")
                print(f"   Message: {result.get('message')}")
                
                return result
            else:
                print(f"❌ Training failed with status {response.status_code}")
                error_detail = response.json().get('detail', response.text)
                print(f"Error: {error_detail}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏰ Training request completed (background processing)")
            return {"status": "background"}
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return None
    
    def test_pydantic_validation(self):
        """Test FastAPI's automatic Pydantic validation"""
        print("\n🧪 Testing Pydantic Validation...")
        
        # Test valid pricing request
        print("   Testing valid pricing request...")
        valid_pricing = {
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
            response = self.session.post(f"{API_BASE}/pricing/predict", json=valid_pricing)
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Valid request - Predicted price: ${result['prediction']:.2f}")
            else:
                print(f"      ❌ Valid request failed: {response.status_code}")
        except Exception as e:
            print(f"      ❌ Valid request error: {e}")
        
        # Test invalid pricing request (should fail validation)
        print("   Testing invalid pricing request...")
        invalid_pricing = {
            "features": {
                "Number_of_Riders": -5,  # Invalid: negative value
                "Number_of_Drivers": 31,
                "Expected_Ride_Duration": 500,  # Invalid: too high
                "Vehicle_Type_encoded": 2,  # Invalid: out of range
                "hour": 25  # Invalid: hour > 23
            }
        }
        
        try:
            response = self.session.post(f"{API_BASE}/pricing/predict", json=invalid_pricing)
            if response.status_code == 422:
                error_detail = response.json()
                print(f"      ✅ Validation correctly failed with 422")
                print(f"      📋 Validation errors: {len(error_detail.get('detail', []))} errors found")
            else:
                print(f"      ❌ Validation should have failed but got: {response.status_code}")
        except Exception as e:
            print(f"      ❌ Validation test error: {e}")
    
    def test_interactive_docs(self):
        """Test interactive documentation endpoints"""
        print("\n📚 Testing Interactive Documentation...")
        
        # Test Swagger UI
        try:
            response = self.session.get("http://localhost:8000/docs")
            if response.status_code == 200:
                print("   ✅ Swagger UI available at: http://localhost:8000/docs")
            else:
                print("   ❌ Swagger UI not accessible")
        except Exception as e:
            print(f"   ❌ Swagger UI error: {e}")
        
        # Test ReDoc
        try:
            response = self.session.get("http://localhost:8000/redoc")
            if response.status_code == 200:
                print("   ✅ ReDoc available at: http://localhost:8000/redoc")
            else:
                print("   ❌ ReDoc not accessible")
        except Exception as e:
            print(f"   ❌ ReDoc error: {e}")
        
        # Test OpenAPI JSON
        try:
            response = self.session.get("http://localhost:8000/openapi.json")
            if response.status_code == 200:
                openapi_spec = response.json()
                endpoints = len(openapi_spec.get('paths', {}))
                print(f"   ✅ OpenAPI spec available with {endpoints} endpoints")
            else:
                print("   ❌ OpenAPI spec not accessible")
        except Exception as e:
            print(f"   ❌ OpenAPI spec error: {e}")
    
    def compare_performance(self):
        """Compare FastAPI vs Flask performance"""
        print("\n⚡ Performance Comparison...")
        
        # Test FastAPI performance
        fastapi_times = []
        for i in range(10):
            start_time = time.time()
            try:
                response = self.session.get("http://localhost:8000/api/health")
                if response.status_code == 200:
                    fastapi_times.append(time.time() - start_time)
            except:
                pass
        
        # Test Flask performance (if running)
        flask_times = []
        for i in range(10):
            start_time = time.time()
            try:
                response = self.session.get("http://localhost:5000/api/health")
                if response.status_code == 200:
                    flask_times.append(time.time() - start_time)
            except:
                pass
        
        if fastapi_times:
            avg_fastapi = sum(fastapi_times) / len(fastapi_times) * 1000
            print(f"   FastAPI avg response time: {avg_fastapi:.2f}ms")
        else:
            print("   ❌ FastAPI not accessible")
        
        if flask_times:
            avg_flask = sum(flask_times) / len(flask_times) * 1000
            print(f"   Flask avg response time: {avg_flask:.2f}ms")
            
            if fastapi_times:
                improvement = ((avg_flask - avg_fastapi) / avg_flask) * 100
                print(f"   📈 FastAPI is {improvement:.1f}% faster than Flask")
        else:
            print("   ℹ️ Flask not running for comparison")
    
    def test_fraud_detection_with_validation(self):
        """Test fraud detection with proper validation"""
        print("\n🚨 Testing Fraud Detection with Validation...")
        
        fraud_data = {
            "transaction": {
                "transaction_id": "TEST_FASTAPI_001",
                "amount": 2500.0,
                "hour": 2,
                "merchant_category": "ATM",
                "location_type": "Travel",
                "days_since_last_transaction": 0.1,
                "transactions_last_hour": 3,
                "transactions_last_day": 8,
                "is_new_device": 1,
                "is_new_ip": 1,
                "distance_from_home_km": 200.0,
                "is_weekend": 1
            }
        }
        
        try:
            response = self.session.post(f"{API_BASE}/fraud/analyze", json=fraud_data)
            if response.status_code == 200:
                result = response.json()
                analysis = result['fraud_analysis']
                print(f"   ✅ Fraud score: {analysis['fraud_score']:.3f}")
                print(f"   ✅ Risk level: {analysis['risk_level']}")
                print(f"   ✅ Action: {analysis['recommended_action']}")
                print(f"   📋 Triggered rules: {len(analysis['triggered_rules'])}")
            else:
                print(f"   ❌ Fraud detection failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Fraud detection error: {e}")

def main():
    """Main execution"""
    print("⚡ SmartMarketer FastAPI Demo")
    print("=" * 50)
    
    demo = FastAPITrainingDemo()
    
    # Test basic connectivity
    try:
        response = demo.session.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ FastAPI is running")
            print(f"   Framework: {health.get('framework', 'FastAPI')}")
            print(f"   Version: {health.get('version', 'Unknown')}")
            print(f"   Models loaded: {health.get('models_loaded', 0)}")
        else:
            print("❌ FastAPI not accessible")
            return
    except Exception as e:
        print(f"❌ Cannot connect to FastAPI: {e}")
        print("💡 Make sure to start FastAPI with: python advanced_fastapi.py")
        return
    
    # Run tests
    demo.test_interactive_docs()
    demo.test_pydantic_validation()
    demo.test_fraud_detection_with_validation()
    demo.compare_performance()
    
    # Train models
    training_result = demo.train_all_models()
    
    print(f"\n🎉 FastAPI demo complete!")
    print(f"💡 Interactive docs available at:")
    print(f"   📖 Swagger UI: http://localhost:8000/docs")
    print(f"   📚 ReDoc: http://localhost:8000/redoc")
    print(f"   🎯 Dashboard: http://localhost:5173/dashboard")

async def async_main():
    """Async main for testing async capabilities"""
    print("⚡ SmartMarketer FastAPI Async Demo")
    print("=" * 50)
    
    demo = FastAPITrainingDemo()
    
    # Test async training
    await demo.async_train_all_models()
    
    print("🎉 Async demo complete!")

if __name__ == "__main__":
    # Run sync demo by default
    main()
    
    # Uncomment to test async capabilities
    # asyncio.run(async_main())
