#!/usr/bin/env python3
"""
Quick start script for the SmartRide dynamic pricing system
"""

import os
import sys
import subprocess
import time

def check_model_exists():
    """Check if the pricing model exists"""
    model_path = "models/city_pricing_model.pkl"
    return os.path.exists(model_path)

def train_model():
    """Train the model if it doesn't exist"""
    print("🔧 Training pricing model...")
    try:
        subprocess.run([sys.executable, "train_and_save_model.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        return False

def start_api():
    """Start the pricing API"""
    print("🚀 Starting pricing API server...")
    try:
        subprocess.run([sys.executable, "api/pricing_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ API server failed: {e}")

def main():
    print("🎯 SmartRide Dynamic Pricing System")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if model exists
    if not check_model_exists():
        print("📋 No trained model found")
        print("🔧 Training model first...")
        if not train_model():
            print("❌ Failed to train model. Please check dependencies.")
            return
    else:
        print("✅ Trained model found")
    
    print("\n🌟 System Features:")
    print("   🏙️  20 Cities across 3 tiers (A, B, C)")
    print("   👥 Driver and Rider user types")
    print("   ⚡ Real-time price updates every 10 seconds")
    print("   📱 WebSocket connections for live updates")
    print("   🎯 Dynamic surge pricing based on supply/demand")
    
    print("\n📍 API will be available at:")
    print("   🌐 Main API: http://localhost:8000")
    print("   📖 API Docs: http://localhost:8000/docs")
    print("   🎛️ Frontend: http://localhost:5173/pricing")
    
    print("\n🚀 Starting API server...")
    print("   Press Ctrl+C to stop")
    time.sleep(2)
    
    start_api()

if __name__ == "__main__":
    main()
