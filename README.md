# SmartMarketer - ML-Powered Dynamic Pricing System

> **A real-time dynamic pricing platform for ride-sharing with machine learning-powered price optimization**

SmartMarketer is an intelligent pricing system that uses machine learning to dynamically adjust ride prices based on real-time supply and demand, city tiers, time patterns, and market conditions. The system features a modern React frontend and a FastAPI backend with WebSocket support for real-time price updates.

## 🎯 Project Overview

This project implements a dynamic pricing system with:

- **🧠 ML Model**: Random Forest Regressor for price prediction
- **📊 Real-time Dashboard**: Live monitoring of city statistics, supply/demand ratios, and pricing
- **⚡ WebSocket Updates**: Event-driven price updates when riders or drivers register
- **🏙️ City-Tier System**: 20 cities across 3 tiers (A, B, C) with different pricing strategies
- **📱 Modern UI**: React + TypeScript frontend with Shadcn UI components
- **🔌 RESTful API**: FastAPI backend with automatic API documentation

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   ML Backend    │    │   ML Models     │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (RandomForest)│
│                 │    │                 │    │                 │
│ • Dashboard     │    │ • REST API      │    │ • Pricing Model │
│ • Live Pricing  │    │ • WebSocket     │    │ • City Tiers    │
│ • Registration  │    │ • Real-time     │    │ • PKL Files     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (Python 3.10+ recommended)
- **Node.js 16+** and npm
- **uv** (Python package manager) - [Install uv](https://github.com/astral-sh/uv)
- **Git** (for cloning the repository)

### Installing uv

If you don't have `uv` installed:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd SmartMarketer
```

### Step 2: Backend Setup

Navigate to the backend directory and set up the Python environment:

```bash
cd ml-backend

# Create virtual environment using uv
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# Install Python dependencies
uv pip install -r requirements.txt

# Install additional dependencies (if needed)
uv pip install scikit-learn logger llvmlite db-sqlite3 pyod
```

### Step 3: Train the ML Model

Before running the API, you need to train and save the pricing model:

```bash
# Make sure you're in the ml-backend directory
python train_and_save_model.py
```

This will:

- Train the Random Forest pricing model
- Save it to `models/city_pricing_model.pkl`
- Display training metrics and model performance

**Note**: Training may take a few minutes depending on your system. The model will be saved and reused on subsequent API starts.

### Step 4: Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
cd frontend

# Install Node.js dependencies
npm install
```

## 🎮 Running the Application

You need to run both the backend API and the frontend development server.

### Terminal 1: Start the Backend API

```bash
cd ml-backend

# Make sure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Start the pricing API server
python start_pricing_system.py
```

The API server will start on **http://localhost:8000**

You should see:

```
🚀 Starting pricing API server...
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start the Frontend

```bash
cd frontend

# Start the development server
npm run dev
```

The frontend will start on **http://localhost:5173**

You should see:

```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

## 🌐 Accessing the Application

Once both servers are running:

- **Landing Page**: http://localhost:5173/
- **Live Pricing App**: http://localhost:5173/pricing
- **ML Dashboard**: http://localhost:5173/dashboard
- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **API ReDoc**: http://localhost:8000/redoc (Alternative API docs)

## 📁 Project Structure

```
SmartMarketer/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── MLDashboard.tsx      # Main dashboard
│   │   │   ├── DynamicPricing.tsx   # Pricing interface
│   │   │   ├── UserRegistration.tsx # User registration
│   │   │   └── ui/          # Shadcn UI components
│   │   ├── pages/           # Page components
│   │   └── App.tsx          # Main app router
│   ├── package.json
│   └── vite.config.ts
│
├── ml-backend/              # Python backend
│   ├── api/
│   │   └── pricing_api.py   # FastAPI application
│   ├── models/
│   │   ├── city_pricing_model.py    # ML model class
│   │   └── city_pricing_model.pkl   # Trained model (generated)
│   ├── datasets/            # Training data
│   ├── train_and_save_model.py      # Model training script
│   ├── start_pricing_system.py      # API startup script
│   └── requirements.txt     # Python dependencies
│
└── README.md
```

## 🔌 API Endpoints

### Core Endpoints

#### Get Available Cities

```http
GET /api/cities
```

Returns list of all supported cities grouped by tier.

#### Get City Areas

```http
GET /api/cities/{city}/areas
```

Returns available areas for a specific city.

#### Register User (Rider/Driver)

```http
POST /api/register
Content-Type: application/json

{
  "user_id": "user123",
  "user_type": "rider",  // or "driver"
  "name": "John Doe",
  "phone": "1234567890",
  "city": "Mumbai",
  "area": "Bandra",
  "rating": 4.5,
  "trips_completed": 50
}
```

#### Get Price Prediction

```http
POST /api/price
Content-Type: application/json

{
  "city": "Mumbai",
  "user_type": "rider",
  "area": "Bandra",
  "current_riders": 50,
  "current_drivers": 30,
  "user_rating": 4.5,
  "trips_completed": 50
}
```

**Note**: The endpoint uses actual city statistics from registered users (maintained via `/api/register` or `/api/supply-demand`), not the values in the request body. The request values are required but ignored to ensure consistent pricing based on real-time supply/demand.

#### Get City Statistics

```http
GET /api/city-stats
```

Returns current supply/demand statistics for all cities.

#### Get City Statistics (Single City)

```http
GET /api/city-stats/{city}
```

Returns statistics for a specific city.

#### Update Supply/Demand

```http
POST /api/supply-demand
Content-Type: application/json

{
  "city": "Mumbai",
  "current_riders": 50,
  "current_drivers": 30
}
```

#### Reset City Statistics

```http
POST /api/city-stats/reset
```

Clears all city statistics (useful for testing).

### WebSocket Endpoint

#### Real-time Price Updates

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/{city}");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Price update:", data);
};
```

## 🎯 Key Features

### 1. Dynamic Pricing

- Real-time price calculation based on supply/demand ratio
- City-tier based pricing (Tier A, B, C cities have different base prices)
- Time-based pricing (peak hours, weekends)
- Event-driven updates when users register

### 2. ML-Powered Predictions

- Random Forest Regressor model for price prediction
- Features include:
  - Current riders and drivers count
  - Demand/supply ratio
  - Time of day and day of week
  - Peak hour detection
  - City tier and area
  - User rating and trip history

### 3. Real-time Dashboard

- Live city statistics
- Supply/demand ratios
- Sample prices for each city
- Surge level indicators
- Last updated timestamps

### 4. User Registration

- Register riders and drivers
- Automatic price updates on registration
- WebSocket notifications to connected clients

## 🛠️ Development

### Backend Development

The backend uses FastAPI with the following key files:

- **`api/pricing_api.py`**: Main FastAPI application with all endpoints
- **`models/city_pricing_model.py`**: ML model class and prediction logic
- **`train_and_save_model.py`**: Script to train and save the model

### Frontend Development

The frontend uses React + TypeScript + Vite:

- **`src/components/MLDashboard.tsx`**: Main dashboard component
- **`src/components/DynamicPricing.tsx`**: Pricing interface with WebSocket
- **`src/components/UserRegistration.tsx`**: User registration form

### Making Changes

1. **Backend Changes**:

   - Edit files in `ml-backend/`
   - Restart the API server to apply changes

2. **Frontend Changes**:

   - Edit files in `frontend/src/`
   - Changes are hot-reloaded automatically by Vite

3. **Model Retraining**:
   - Modify training logic in `train_and_save_model.py`
   - Run `python train_and_save_model.py` to retrain
   - Restart API server to load new model

## 🐛 Troubleshooting

### Issue: Model file not found

**Error**: `FileNotFoundError: models/city_pricing_model.pkl`

**Solution**:

```bash
cd ml-backend
python train_and_save_model.py
```

### Issue: Port already in use

**Error**: `Address already in use` or `Port 8000 is already in use`

**Solution**:

- Stop any process using port 8000 (backend) or 5173 (frontend)
- Or change ports in:
  - Backend: `start_pricing_system.py` or `pricing_api.py`
  - Frontend: `vite.config.ts`

### Issue: Module not found errors

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:

```bash
cd ml-backend
source .venv/bin/activate  # Activate venv
uv pip install -r requirements.txt
```

### Issue: Frontend can't connect to API

**Error**: `Failed to fetch` or CORS errors

**Solution**:

- Ensure backend is running on http://localhost:8000
- Check `API_BASE` in frontend components matches backend URL
- Verify CORS is enabled in `pricing_api.py`

### Issue: WebSocket connection fails

**Error**: `WebSocket connection failed`

**Solution**:

- Ensure backend is running
- Check WebSocket URL format: `ws://localhost:8000/ws/{city}`
- Verify city name is valid (use cities from `/api/cities`)

### Issue: Dashboard shows no data

**Solution**:

- Register some users (riders/drivers) via the pricing app
- Check that `/api/city-stats` returns data
- Verify backend is running and accessible

## 📊 Model Performance

The Random Forest pricing model achieves:

- **R² Score**: ~0.90+ (90% variance explained)
- **Prediction Time**: <150ms per request
- **Features**: 14+ engineered features
- **Model**: Random Forest Regressor with feature engineering

## 🏙️ Supported Cities

### Tier A Cities (Premium Pricing)

- Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata, Pune

### Tier B Cities (Standard Pricing)

- Ahmedabad, Jaipur, Surat, Lucknow, Kanpur, Nagpur, Indore

### Tier C Cities (Economy Pricing)

- Thane, Bhopal, Visakhapatnam, Patna, Vadodara, Ghaziabad

Each city has multiple areas with specific pricing zones.

## 🔒 Environment Variables

Currently, the project uses default configurations. For production, consider adding:

- `API_PORT`: Backend port (default: 8000)
- `FRONTEND_PORT`: Frontend port (default: 5173)
- `MODEL_PATH`: Path to model file
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

## 📝 Notes

- The model is trained on synthetic data for demonstration purposes
- City statistics are stored in-memory and reset on server restart
- WebSocket connections are per-city (subscribe to specific city updates)
- Prices are calculated in Indian Rupees (₹)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes as part of a B.Tech project.

## 👨‍💻 Authors

- **Diya** - 22ITUON070
- **Tisha** - 22ITUBS036

**Under the guidance of**: Prof. Dr. Harshadkumar B. Prajapati 

**Institution**: Dharmsinh Desai University, Nadiad

---

## 🚀 Quick Start Summary

```bash
# 1. Clone and navigate
git clone <repo-url>
cd SmartMarketer

# 2. Backend setup
cd ml-backend
uv venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
uv pip install -r requirements.txt
uv pip install scikit-learn logger llvmlite db-sqlite3 pyod
python train_and_save_model.py

# 3. Frontend setup (new terminal)
cd frontend
npm install

# 4. Run backend (Terminal 1)
cd ml-backend
source .venv/bin/activate
python start_pricing_system.py

# 5. Run frontend (Terminal 2)
cd frontend
npm run dev

# 6. Open browser
# http://localhost:5173
```

---

**⭐ If you find this project helpful, please star the repository!**
