#!/usr/bin/env python3
"""
Startup script for SmartMarketer API servers
Supports both Flask and FastAPI with easy switching
"""

import argparse
import sys
import os
import subprocess
import time

def start_flask():
    """Start Flask server"""
    print("🚀 Starting Flask API Server...")
    print("📍 URL: http://localhost:5000")
    print("📖 Documentation: http://localhost:5000")
    print("🎯 Dashboard: http://localhost:5173/dashboard")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "advanced_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Flask server stopped")
    except Exception as e:
        print(f"❌ Flask server error: {e}")

def start_fastapi():
    """Start FastAPI server"""
    print("⚡ Starting FastAPI Server...")
    print("📍 URL: http://localhost:8000")
    print("📖 Swagger UI: http://localhost:8000/docs")
    print("📚 ReDoc: http://localhost:8000/redoc")
    print("🎯 Dashboard: http://localhost:5173/dashboard (update API_BASE to port 8000)")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "advanced_fastapi:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
    except Exception as e:
        print(f"❌ FastAPI server error: {e}")

def start_both():
    """Start both Flask and FastAPI servers"""
    print("🔥 Starting Both Flask and FastAPI Servers...")
    print("📍 Flask: http://localhost:5000")
    print("📍 FastAPI: http://localhost:8000")
    print("📖 FastAPI Docs: http://localhost:8000/docs")
    print("-" * 50)
    
    import threading
    import time
    
    def run_flask():
        try:
            subprocess.run([sys.executable, "advanced_api.py"], check=True)
        except Exception as e:
            print(f"❌ Flask error: {e}")
    
    def run_fastapi():
        time.sleep(2)  # Small delay to avoid port conflicts during startup
        try:
            subprocess.run([
                sys.executable, "-m", "uvicorn", 
                "advanced_fastapi:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload",
                "--log-level", "info"
            ], check=True)
        except Exception as e:
            print(f"❌ FastAPI error: {e}")
    
    # Start both servers in threads
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    
    flask_thread.start()
    fastapi_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Both servers stopped")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'fastapi', 'uvicorn', 'pydantic', 
        'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def show_comparison():
    """Show Flask vs FastAPI comparison"""
    print("\n📊 Flask vs FastAPI Comparison")
    print("=" * 50)
    print("🐍 Flask:")
    print("   ✅ Mature and stable")
    print("   ✅ Large ecosystem")
    print("   ✅ Simple to understand")
    print("   ⚠️  Synchronous by default")
    print("   ⚠️  Manual API documentation")
    print("   📍 Port: 5000")
    
    print("\n⚡ FastAPI:")
    print("   ✅ High performance (2-3x faster)")
    print("   ✅ Automatic API documentation")
    print("   ✅ Built-in data validation")
    print("   ✅ Async/await support")
    print("   ✅ Modern Python type hints")
    print("   📍 Port: 8000")
    
    print("\n🎯 Choose based on your needs:")
    print("   - Use Flask for traditional web apps")
    print("   - Use FastAPI for high-performance APIs")
    print("   - Both support the same ML models!")

def main():
    parser = argparse.ArgumentParser(
        description="SmartMarketer API Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_api.py --flask          # Start Flask server
  python start_api.py --fastapi        # Start FastAPI server  
  python start_api.py --both           # Start both servers
  python start_api.py --compare        # Show comparison
        """
    )
    
    parser.add_argument('--flask', action='store_true', 
                       help='Start Flask server (port 5000)')
    parser.add_argument('--fastapi', action='store_true', 
                       help='Start FastAPI server (port 8000)')
    parser.add_argument('--both', action='store_true', 
                       help='Start both Flask and FastAPI servers')
    parser.add_argument('--compare', action='store_true', 
                       help='Show Flask vs FastAPI comparison')
    parser.add_argument('--check', action='store_true', 
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    print("🧠 SmartMarketer API Server Launcher")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    if args.check:
        print("✅ Dependency check complete")
        return
    
    if args.compare:
        show_comparison()
        return
    
    # Change to the ml-backend directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if args.flask:
        start_flask()
    elif args.fastapi:
        start_fastapi()
    elif args.both:
        start_both()
    else:
        # Default: show options
        print("🤔 Which server would you like to start?")
        print("   1. Flask (Traditional, Port 5000)")
        print("   2. FastAPI (Modern, Port 8000)")
        print("   3. Both (Comparison)")
        print("   4. Show comparison")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            start_flask()
        elif choice == "2":
            start_fastapi()
        elif choice == "3":
            start_both()
        elif choice == "4":
            show_comparison()
        else:
            print("❌ Invalid choice")
            print("💡 Use --help for usage information")

if __name__ == "__main__":
    main()
