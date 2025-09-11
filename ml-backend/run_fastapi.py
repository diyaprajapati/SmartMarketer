#!/usr/bin/env python3
"""
FastAPI Server Runner with automatic testing
"""

import subprocess
import sys
import time
import requests
import threading
import os

def check_server_health(port=8000, max_attempts=30):
    """Check if the FastAPI server is running and healthy"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/api/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
        print(f"Waiting for server... ({attempt + 1}/{max_attempts})")
    return False

def test_pricing_endpoint(port=8000):
    """Test the pricing prediction endpoint"""
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
            f"http://localhost:{port}/api/pricing/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! Predicted Price: ${result['prediction']:.2f}")
            return True
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_advanced_fastapi():
    """Run the advanced FastAPI server"""
    print("🚀 Starting Advanced FastAPI Server...")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "advanced_fastapi:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("⏳ Waiting for server to start...")
        
        # Wait for server to be ready
        if check_server_health():
            print("✅ Server is healthy!")
            
            # Test the pricing endpoint
            print("🧪 Testing pricing prediction...")
            if test_pricing_endpoint():
                print("🎉 All tests passed!")
                print()
                print("📍 Server URLs:")
                print("   🏠 Home: http://localhost:8000")
                print("   📖 Swagger UI: http://localhost:8000/docs")
                print("   📚 ReDoc: http://localhost:8000/redoc")
                print("   🎯 Dashboard: http://localhost:5173/dashboard")
                print()
                print("Press Ctrl+C to stop the server...")
                
                # Keep the server running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping server...")
                    process.terminate()
                    process.wait()
            else:
                print("❌ Tests failed, stopping server...")
                process.terminate()
        else:
            print("❌ Server failed to start properly")
            process.terminate()
            
    except FileNotFoundError:
        print("❌ uvicorn not found. Install with: pip install uvicorn")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def run_simple_fastapi():
    """Run the simple FastAPI server"""
    print("🚀 Starting Simple FastAPI Server...")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "simple_fastapi.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("⏳ Waiting for server to start...")
        
        # Wait for server to be ready
        if check_server_health():
            print("✅ Simple server is healthy!")
            
            # Test the pricing endpoint
            print("🧪 Testing pricing prediction...")
            if test_pricing_endpoint():
                print("🎉 All tests passed!")
                print()
                print("📍 Server URLs:")
                print("   🏠 Home: http://localhost:8000")
                print("   📖 Swagger UI: http://localhost:8000/docs")
                print("   📚 ReDoc: http://localhost:8000/redoc")
                print()
                print("Press Ctrl+C to stop the server...")
                
                # Keep the server running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping server...")
                    process.terminate()
                    process.wait()
            else:
                print("❌ Tests failed, stopping server...")
                process.terminate()
        else:
            print("❌ Server failed to start properly")
            process.terminate()
            
    except Exception as e:
        print(f"❌ Error starting simple server: {e}")

def main():
    print("⚡ FastAPI Server Launcher")
    print("=" * 50)
    
    # Change to the ml-backend directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Which FastAPI server would you like to run?")
    print("1. Advanced FastAPI (full features)")
    print("2. Simple FastAPI (basic pricing only)")
    print("3. Auto-select (try advanced, fallback to simple)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_advanced_fastapi()
    elif choice == "2":
        run_simple_fastapi()
    elif choice == "3":
        print("🔄 Trying advanced FastAPI first...")
        try:
            # Test if advanced dependencies are available
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            print("✅ Advanced dependencies available")
            run_advanced_fastapi()
        except ImportError as e:
            print(f"⚠️ Advanced dependencies missing: {e}")
            print("🔄 Falling back to simple FastAPI...")
            run_simple_fastapi()
    else:
        print("❌ Invalid choice. Use 1, 2, or 3")

if __name__ == "__main__":
    main()
