"""
Advanced Ensemble Pricing Model
Combines multiple ML algorithms for robust dynamic pricing
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin

# Advanced ML models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

# Time series and forecasting
from sklearn.cluster import KMeans
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

warnings.filterwarnings('ignore')

class AdvancedEnsemblePricer(BaseEstimator, RegressorMixin):
    """
    Advanced ensemble model that combines multiple algorithms and adapts based on market conditions
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.meta_model = None
        self.scalers = {}
        self.feature_engineeer = None
        self.market_regimes = None
        self.is_fitted = False
        
        # Initialize base models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the ensemble of models"""
        self.models = {
            'catboost': CatBoostRegressor(
                iterations=1000,
                depth=8,
                learning_rate=0.03,
                l2_leaf_reg=3,
                subsample=0.8,
                verbose=False,
                random_seed=self.random_state
            ),
            'xgboost': XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=-1
            ),
            'neural_net': MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=self.random_state
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=self.random_state
            )
        }
        
        # Meta-learner for stacking
        self.meta_model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_seed=self.random_state
        )
    
    def _advanced_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better predictions"""
        X_enhanced = X.copy()
        
        # Time-based features if timestamp is available
        current_time = datetime.now()
        X_enhanced['hour'] = current_time.hour
        X_enhanced['day_of_week'] = current_time.weekday()
        X_enhanced['month'] = current_time.month
        X_enhanced['is_weekend'] = (current_time.weekday() >= 5).astype(int)
        X_enhanced['is_peak_hour'] = ((current_time.hour >= 8) & (current_time.hour <= 10) | 
                                    (current_time.hour >= 17) & (current_time.hour <= 19)).astype(int)
        
        # Supply-demand ratio features
        if 'Number_of_Riders' in X.columns and 'Number_of_Drivers' in X.columns:
            X_enhanced['supply_demand_ratio'] = X['Number_of_Drivers'] / (X['Number_of_Riders'] + 1)
            X_enhanced['demand_pressure'] = X['Number_of_Riders'] / (X['Number_of_Drivers'] + 1)
            X_enhanced['market_saturation'] = X['Number_of_Drivers'] + X['Number_of_Riders']
        
        # Polynomial features for key interactions
        if len(X.columns) > 2:
            poly_features = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
            poly_features = [col for col in poly_features if col in X.columns]
            
            if len(poly_features) >= 2:
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                poly_data = poly.fit_transform(X[poly_features])
                poly_names = poly.get_feature_names_out(poly_features)
                
                for i, name in enumerate(poly_names):
                    if name not in X.columns:  # Avoid duplicates
                        X_enhanced[f'poly_{name}'] = poly_data[:, i]
        
        # Statistical features
        numeric_cols = X_enhanced.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            X_enhanced['feature_mean'] = X_enhanced[numeric_cols].mean(axis=1)
            X_enhanced['feature_std'] = X_enhanced[numeric_cols].std(axis=1)
            X_enhanced['feature_skew'] = X_enhanced[numeric_cols].skew(axis=1)
        
        return X_enhanced
    
    def _detect_market_regime(self, X: pd.DataFrame) -> np.ndarray:
        """Detect market regimes for adaptive pricing"""
        if self.market_regimes is None:
            # Use clustering to identify market conditions
            features_for_clustering = []
            if 'Number_of_Riders' in X.columns:
                features_for_clustering.append('Number_of_Riders')
            if 'Number_of_Drivers' in X.columns:
                features_for_clustering.append('Number_of_Drivers')
            if 'supply_demand_ratio' in X.columns:
                features_for_clustering.append('supply_demand_ratio')
            
            if features_for_clustering:
                self.market_regimes = KMeans(n_clusters=3, random_state=self.random_state)
                regime_data = X[features_for_clustering].fillna(0)
                return self.market_regimes.fit_predict(regime_data)
            else:
                return np.zeros(len(X))
        else:
            features_for_clustering = []
            if 'Number_of_Riders' in X.columns:
                features_for_clustering.append('Number_of_Riders')
            if 'Number_of_Drivers' in X.columns:
                features_for_clustering.append('Number_of_Drivers')
            if 'supply_demand_ratio' in X.columns:
                features_for_clustering.append('supply_demand_ratio')
            
            if features_for_clustering:
                regime_data = X[features_for_clustering].fillna(0)
                return self.market_regimes.predict(regime_data)
            else:
                return np.zeros(len(X))
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdvancedEnsemblePricer':
        """Fit the ensemble model"""
        print("🚀 Training Advanced Ensemble Pricing Model...")
        
        # Feature engineering
        X_enhanced = self._advanced_feature_engineering(X)
        
        # Detect market regimes
        market_regimes = self._detect_market_regime(X_enhanced)
        X_enhanced['market_regime'] = market_regimes
        
        # Handle missing values
        X_enhanced = X_enhanced.fillna(X_enhanced.median())
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X_enhanced)
        X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced.columns)
        
        # Split for meta-learning
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train base models
        base_predictions = np.zeros((len(X_val), len(self.models)))
        
        print("Training base models...")
        for i, (name, model) in enumerate(self.models.items()):
            print(f"  📊 Training {name}...")
            
            if name == 'neural_net':
                # Special handling for neural network
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            # Get validation predictions for meta-learning
            val_pred = model.predict(X_val)
            base_predictions[:, i] = val_pred
            
            # Calculate individual model performance
            val_score = r2_score(y_val, val_pred)
            print(f"    ✅ {name} R² score: {val_score:.4f}")
        
        # Train meta-model
        print("  🧠 Training meta-learner...")
        self.meta_model.fit(base_predictions, y_val)
        
        # Final ensemble prediction on validation set
        meta_pred = self.meta_model.predict(base_predictions)
        ensemble_score = r2_score(y_val, meta_pred)
        print(f"  🎯 Ensemble R² score: {ensemble_score:.4f}")
        
        self.is_fitted = True
        self.feature_names_ = X_enhanced.columns.tolist()
        
        print("✅ Advanced Ensemble Model Training Complete!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Feature engineering
        X_enhanced = self._advanced_feature_engineering(X)
        
        # Detect market regimes
        market_regimes = self._detect_market_regime(X_enhanced)
        X_enhanced['market_regime'] = market_regimes
        
        # Handle missing values
        X_enhanced = X_enhanced.fillna(0)
        
        # Ensure all columns are present
        for col in self.feature_names_:
            if col not in X_enhanced.columns:
                X_enhanced[col] = 0
        
        # Reorder columns to match training
        X_enhanced = X_enhanced[self.feature_names_]
        
        # Scale features
        X_scaled = self.scalers['main'].transform(X_enhanced)
        X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced.columns)
        
        # Get base model predictions
        base_predictions = np.zeros((len(X_scaled), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            pred = model.predict(X_scaled)
            base_predictions[:, i] = pred
        
        # Meta-model prediction
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get importance from tree-based models
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                model_importance = dict(zip(self.feature_names_, model.feature_importances_))
                importances[name] = model_importance
        
        # Average importance across models
        avg_importance = {}
        for feature in self.feature_names_:
            feature_scores = [imp.get(feature, 0) for imp in importances.values()]
            avg_importance[feature] = np.mean(feature_scores)
        
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get predictions from all base models
        X_enhanced = self._advanced_feature_engineering(X)
        market_regimes = self._detect_market_regime(X_enhanced)
        X_enhanced['market_regime'] = market_regimes
        X_enhanced = X_enhanced.fillna(0)
        
        for col in self.feature_names_:
            if col not in X_enhanced.columns:
                X_enhanced[col] = 0
        
        X_enhanced = X_enhanced[self.feature_names_]
        X_scaled = self.scalers['main'].transform(X_enhanced)
        X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced.columns)
        
        all_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions).T
        
        # Calculate mean and std across models
        mean_pred = np.mean(all_predictions, axis=1)
        std_pred = np.std(all_predictions, axis=1)
        
        return mean_pred, std_pred
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'scalers': self.scalers,
            'market_regimes': self.market_regimes,
            'feature_names_': self.feature_names_ if hasattr(self, 'feature_names_') else None,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.models = model_data['models']
        instance.meta_model = model_data['meta_model']
        instance.scalers = model_data['scalers']
        instance.market_regimes = model_data['market_regimes']
        instance.feature_names_ = model_data.get('feature_names_', [])
        instance.is_fitted = model_data['is_fitted']
        
        print(f"📥 Model loaded from {filepath}")
        return instance


class PriceOptimizer:
    """Advanced price optimization using economic principles"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.elasticity_model = None
        
    def calculate_price_elasticity(self, X: pd.DataFrame, price_range: Tuple[float, float], steps: int = 20):
        """Calculate price elasticity of demand"""
        if not hasattr(self.base_model, 'predict'):
            raise ValueError("Base model must have predict method")
        
        prices = np.linspace(price_range[0], price_range[1], steps)
        elasticities = []
        
        base_X = X.copy()
        
        for i in range(len(prices) - 1):
            # Simulate demand at different price points
            X_low = base_X.copy()
            X_high = base_X.copy()
            
            # Assume price affects demand (inverse relationship)
            demand_factor_low = 1.0 + (price_range[1] - prices[i]) / price_range[1] * 0.5
            demand_factor_high = 1.0 + (price_range[1] - prices[i+1]) / price_range[1] * 0.5
            
            if 'Number_of_Riders' in X.columns:
                X_low['Number_of_Riders'] = X['Number_of_Riders'] * demand_factor_low
                X_high['Number_of_Riders'] = X['Number_of_Riders'] * demand_factor_high
            
            # Predict demand
            demand_low = np.mean(self.base_model.predict(X_low))
            demand_high = np.mean(self.base_model.predict(X_high))
            
            # Calculate elasticity
            price_change = (prices[i+1] - prices[i]) / prices[i]
            demand_change = (demand_high - demand_low) / demand_low if demand_low != 0 else 0
            
            elasticity = demand_change / price_change if price_change != 0 else 0
            elasticities.append(elasticity)
        
        return np.mean(elasticities)
    
    def optimize_price(self, X: pd.DataFrame, target_margin: float = 0.2) -> Dict:
        """Optimize price for maximum revenue or target margin"""
        base_prediction = self.base_model.predict(X)
        
        # Simulate different price multipliers
        multipliers = np.arange(0.8, 1.5, 0.05)
        revenues = []
        
        for mult in multipliers:
            # Simulate demand response to price change
            X_sim = X.copy()
            if 'Number_of_Riders' in X.columns:
                # Inverse relationship: higher price = lower demand
                demand_factor = 1.0 / (1.0 + (mult - 1.0) * 2.0)
                X_sim['Number_of_Riders'] = X['Number_of_Riders'] * demand_factor
            
            predicted_demand = np.mean(self.base_model.predict(X_sim))
            price = np.mean(base_prediction) * mult
            revenue = predicted_demand * price
            revenues.append(revenue)
        
        # Find optimal multiplier
        optimal_idx = np.argmax(revenues)
        optimal_multiplier = multipliers[optimal_idx]
        optimal_price = np.mean(base_prediction) * optimal_multiplier
        
        return {
            'base_price': np.mean(base_prediction),
            'optimal_price': optimal_price,
            'optimal_multiplier': optimal_multiplier,
            'expected_revenue': revenues[optimal_idx],
            'elasticity': self.calculate_price_elasticity(X, (optimal_price * 0.8, optimal_price * 1.2))
        }


def train_and_evaluate_ensemble(data_path: str = None):
    """Complete training and evaluation pipeline"""
    
    # Load data (using the existing dataset)
    if data_path is None:
        # Load the existing dynamic pricing data
        df = pd.read_csv('/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/datasets/dynamic_pricing.csv')
    else:
        df = pd.read_csv(data_path)
    
    print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare features and target
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    
    # Select features
    feature_cols = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 
                   'Average_Ratings', 'Expected_Ride_Duration']
    
    # Add encoded categorical features
    for col in categorical_cols:
        if f'{col}_encoded' in df.columns:
            feature_cols.append(f'{col}_encoded')
    
    # Remove any columns that don't exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Historical_Cost_of_Ride']
    
    # Apply log transformation to target
    y_transformed = np.log1p(y)
    
    print(f"🎯 Features: {feature_cols}")
    print(f"📈 Target: Historical_Cost_of_Ride (log-transformed)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42
    )
    
    # Train ensemble model
    ensemble = AdvancedEnsemblePricer(random_state=42)
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    y_pred = ensemble.predict(X_test)
    
    # Transform back to original scale
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    # Evaluate
    r2 = r2_score(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    
    print(f"\n🎯 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    # Feature importance
    importance = ensemble.get_feature_importance()
    print(f"\n🔍 Top 5 Feature Importances:")
    for i, (feature, imp) in enumerate(list(importance.items())[:5]):
        print(f"   {i+1}. {feature}: {imp:.4f}")
    
    # Uncertainty estimation
    pred_mean, pred_std = ensemble.predict_with_uncertainty(X_test.head(5))
    print(f"\n🎲 Prediction Uncertainty (first 5 samples):")
    for i in range(5):
        print(f"   Sample {i+1}: ${np.expm1(pred_mean[i]):.2f} ± ${np.expm1(pred_std[i]):.2f}")
    
    # Price optimization
    optimizer = PriceOptimizer(ensemble)
    optimization_result = optimizer.optimize_price(X_test.head(1))
    
    print(f"\n💰 Price Optimization Example:")
    print(f"   Base Price: ${optimization_result['base_price']:.2f}")
    print(f"   Optimal Price: ${optimization_result['optimal_price']:.2f}")
    print(f"   Price Multiplier: {optimization_result['optimal_multiplier']:.2f}x")
    print(f"   Expected Revenue: ${optimization_result['expected_revenue']:.2f}")
    print(f"   Price Elasticity: {optimization_result['elasticity']:.3f}")
    
    # Save model
    model_path = '/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/advanced_models/ensemble_model.joblib'
    ensemble.save_model(model_path)
    
    return ensemble, {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'feature_importance': importance,
        'optimization_example': optimization_result
    }


if __name__ == "__main__":
    # Run the complete training pipeline
    model, results = train_and_evaluate_ensemble()
    print("\n🚀 Advanced Ensemble Pricing Model is ready for deployment!")
