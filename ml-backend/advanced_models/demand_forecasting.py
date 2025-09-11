"""
Advanced Demand Forecasting System
Uses LSTM and time series analysis for predicting future demand patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
from typing import Dict, List, Tuple, Optional

# Time series libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

# Deep learning for time series
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class DemandForecaster:
    """Advanced demand forecasting using multiple techniques"""
    
    def __init__(self, sequence_length=24, forecast_horizon=12):
        self.sequence_length = sequence_length  # Hours to look back
        self.forecast_horizon = forecast_horizon  # Hours to forecast ahead
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
    def _generate_synthetic_time_series(self, start_date='2023-01-01', periods=8760):
        """Generate synthetic hourly demand data for demonstration"""
        date_range = pd.date_range(start=start_date, periods=periods, freq='H')
        
        # Base demand with trends and seasonality
        t = np.arange(periods)
        
        # Trend component
        trend = 100 + 0.01 * t
        
        # Daily seasonality (higher demand during day, peak at rush hours)
        daily_seasonal = 20 * np.sin(2 * np.pi * t / 24) + \
                        30 * np.sin(2 * np.pi * (t - 8) / 24) * (np.sin(2 * np.pi * (t - 8) / 24) > 0) + \
                        40 * np.sin(2 * np.pi * (t - 18) / 24) * (np.sin(2 * np.pi * (t - 18) / 24) > 0)
        
        # Weekly seasonality (higher demand on weekdays)
        weekly_seasonal = 15 * np.sin(2 * np.pi * t / (24 * 7))
        
        # Monthly seasonality
        monthly_seasonal = 10 * np.sin(2 * np.pi * t / (24 * 30))
        
        # Weather effect (simplified)
        weather_effect = 5 * np.sin(2 * np.pi * t / (24 * 365)) + np.random.normal(0, 3, periods)
        
        # Special events (random spikes)
        special_events = np.random.exponential(0.1, periods) * np.random.binomial(1, 0.05, periods) * 50
        
        # Random noise
        noise = np.random.normal(0, 5, periods)
        
        # Combine all components
        demand = trend + daily_seasonal + weekly_seasonal + monthly_seasonal + weather_effect + special_events + noise
        demand = np.maximum(demand, 10)  # Ensure positive demand
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'demand': demand,
            'hour': date_range.hour,
            'day_of_week': date_range.dayofweek,
            'month': date_range.month,
            'is_weekend': (date_range.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.02, periods),  # 2% chance of holiday
        })
        
        return df
    
    def _create_sequences(self, data, target_col='demand'):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.forecast_horizon + 1):
            # Features: past sequence_length observations
            X.append(data[i - self.sequence_length:i])
            # Target: next forecast_horizon observations
            y.append(data[target_col].iloc[i:i + self.forecast_horizon].values)
        
        return np.array(X), np.array(y)
    
    def _build_lstm_model(self, input_shape, output_length):
        """Build LSTM model for demand forecasting"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_length, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_cnn_lstm_model(self, input_shape, output_length):
        """Build CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(output_length, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, data: pd.DataFrame, target_col: str = 'demand'):
        """Fit the forecasting models"""
        print("🔮 Training Advanced Demand Forecasting Models...")
        
        # Ensure timestamp column is datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp').sort_index()
        
        # Feature engineering
        data_enhanced = self._add_time_features(data.copy())
        
        # Prepare data for different models
        self._prepare_data(data_enhanced, target_col)
        
        # Fit traditional time series models
        self._fit_traditional_models(data_enhanced[target_col])
        
        # Fit deep learning models
        self._fit_deep_learning_models(data_enhanced, target_col)
        
        self.is_fitted = True
        print("✅ Demand Forecasting Models Training Complete!")
    
    def _add_time_features(self, data):
        """Add time-based features"""
        if data.index.name == 'timestamp' or isinstance(data.index, pd.DatetimeIndex):
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['day_of_year'] = data.index.dayofyear
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            data['quarter'] = data.index.quarter
            
            # Cyclical encoding
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        return data
    
    def _prepare_data(self, data, target_col):
        """Prepare data for model training"""
        # Select features for deep learning
        feature_cols = [col for col in data.columns if col != target_col]
        
        # Scale the data
        self.scalers['features'] = StandardScaler()
        self.scalers['target'] = MinMaxScaler()
        
        features_scaled = self.scalers['features'].fit_transform(data[feature_cols])
        target_scaled = self.scalers['target'].fit_transform(data[[target_col]])
        
        # Create DataFrame with scaled data
        self.scaled_data = pd.DataFrame(
            np.column_stack([features_scaled, target_scaled.flatten()]),
            columns=feature_cols + [target_col],
            index=data.index
        )
    
    def _fit_traditional_models(self, ts_data):
        """Fit traditional time series models"""
        print("  📈 Training traditional time series models...")
        
        # ARIMA model
        try:
            # Determine ARIMA order automatically
            arima_order = self._auto_arima_order(ts_data)
            self.models['arima'] = ARIMA(ts_data, order=arima_order).fit()
            print(f"    ✅ ARIMA{arima_order} fitted")
        except Exception as e:
            print(f"    ❌ ARIMA failed: {str(e)}")
        
        # Exponential Smoothing
        try:
            self.models['exponential_smoothing'] = ExponentialSmoothing(
                ts_data, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=24  # Daily seasonality for hourly data
            ).fit()
            print("    ✅ Exponential Smoothing fitted")
        except Exception as e:
            print(f"    ❌ Exponential Smoothing failed: {str(e)}")
    
    def _auto_arima_order(self, ts_data, max_p=3, max_d=2, max_q=3):
        """Automatically determine ARIMA order"""
        # Check stationarity
        adf_result = adfuller(ts_data)
        d = 0 if adf_result[1] <= 0.05 else 1
        
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        return best_order
    
    def _fit_deep_learning_models(self, data, target_col):
        """Fit deep learning models"""
        print("  🧠 Training deep learning models...")
        
        # Create sequences
        feature_cols = [col for col in data.columns if col != target_col]
        X, y = self._create_sequences(self.scaled_data)
        
        if len(X) == 0:
            print("    ❌ Not enough data for sequence creation")
            return
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # LSTM Model
        try:
            input_shape = (X.shape[1], X.shape[2])
            self.models['lstm'] = self._build_lstm_model(input_shape, self.forecast_horizon)
            
            history = self.models['lstm'].fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            print("    ✅ LSTM model fitted")
        except Exception as e:
            print(f"    ❌ LSTM failed: {str(e)}")
        
        # CNN-LSTM Model
        try:
            self.models['cnn_lstm'] = self._build_cnn_lstm_model(input_shape, self.forecast_horizon)
            
            history = self.models['cnn_lstm'].fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            print("    ✅ CNN-LSTM model fitted")
        except Exception as e:
            print(f"    ❌ CNN-LSTM failed: {str(e)}")
    
    def forecast(self, steps: int, data: pd.DataFrame = None, method: str = 'ensemble') -> pd.DataFrame:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first")
        
        forecasts = {}
        
        # Traditional model forecasts
        if 'arima' in self.models:
            try:
                arima_forecast = self.models['arima'].forecast(steps=steps)
                forecasts['arima'] = arima_forecast
            except Exception as e:
                print(f"ARIMA forecast failed: {e}")
        
        if 'exponential_smoothing' in self.models:
            try:
                es_forecast = self.models['exponential_smoothing'].forecast(steps=steps)
                forecasts['exponential_smoothing'] = es_forecast
            except Exception as e:
                print(f"Exponential Smoothing forecast failed: {e}")
        
        # Deep learning forecasts
        if data is not None and ('lstm' in self.models or 'cnn_lstm' in self.models):
            try:
                # Prepare last sequence for prediction
                data_enhanced = self._add_time_features(data.copy())
                data_scaled = self.scalers['features'].transform(
                    data_enhanced.drop(columns=['demand'] if 'demand' in data_enhanced.columns else [])
                )
                
                last_sequence = data_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                
                if 'lstm' in self.models:
                    lstm_pred = self.models['lstm'].predict(last_sequence, verbose=0)[0]
                    lstm_pred = self.scalers['target'].inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
                    forecasts['lstm'] = lstm_pred[:steps]
                
                if 'cnn_lstm' in self.models:
                    cnn_lstm_pred = self.models['cnn_lstm'].predict(last_sequence, verbose=0)[0]
                    cnn_lstm_pred = self.scalers['target'].inverse_transform(cnn_lstm_pred.reshape(-1, 1)).flatten()
                    forecasts['cnn_lstm'] = cnn_lstm_pred[:steps]
                    
            except Exception as e:
                print(f"Deep learning forecast failed: {e}")
        
        # Ensemble forecast
        if len(forecasts) > 1 and method == 'ensemble':
            # Simple average ensemble
            min_length = min([len(f) for f in forecasts.values()])
            ensemble_forecast = np.mean([f[:min_length] for f in forecasts.values()], axis=0)
            forecasts['ensemble'] = ensemble_forecast
        
        # Create forecast DataFrame
        if data is not None and hasattr(data.index, 'freq'):
            future_dates = pd.date_range(
                start=data.index[-1] + data.index.freq,
                periods=steps,
                freq=data.index.freq
            )
        else:
            future_dates = range(len(data), len(data) + steps) if data is not None else range(steps)
        
        forecast_df = pd.DataFrame(forecasts, index=future_dates)
        
        return forecast_df
    
    def evaluate_models(self, test_data: pd.DataFrame, target_col: str = 'demand') -> Dict:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first")
        
        results = {}
        
        # Generate forecasts for test period
        forecast_steps = len(test_data)
        forecasts = self.forecast(steps=forecast_steps, data=test_data)
        
        actual = test_data[target_col].values
        
        for model_name in forecasts.columns:
            predicted = forecasts[model_name].values[:len(actual)]
            
            if len(predicted) == len(actual):
                mse = mean_squared_error(actual, predicted)
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                
                results[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                }
        
        return results
    
    def plot_forecasts(self, historical_data: pd.DataFrame, forecasts: pd.DataFrame, 
                      target_col: str = 'demand', figsize: Tuple[int, int] = (15, 8)):
        """Plot historical data and forecasts"""
        plt.figure(figsize=figsize)
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data[target_col], 
                label='Historical', color='blue', alpha=0.7)
        
        # Plot forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, model in enumerate(forecasts.columns):
            plt.plot(forecasts.index, forecasts[model], 
                    label=f'{model.title()} Forecast', 
                    color=colors[i % len(colors)], 
                    linestyle='--', alpha=0.8)
        
        plt.title('Demand Forecasting Results', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Demand', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_seasonality(self, data: pd.DataFrame, target_col: str = 'demand'):
        """Analyze seasonal patterns in demand"""
        # Seasonal decomposition
        decomposition = seasonal_decompose(
            data[target_col], 
            model='additive', 
            period=24  # Daily seasonality for hourly data
        )
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Plot components
        decomposition.observed.plot(ax=axes[0], title='Original Data')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        
        # Hourly and daily patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hourly pattern
        hourly_avg = data.groupby(data.index.hour)[target_col].mean()
        hourly_avg.plot(ax=axes[0, 0], kind='bar', title='Average Demand by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        
        # Daily pattern
        daily_avg = data.groupby(data.index.dayofweek)[target_col].mean()
        daily_avg.plot(ax=axes[0, 1], kind='bar', title='Average Demand by Day of Week')
        axes[0, 1].set_xlabel('Day of Week (0=Monday)')
        
        # Monthly pattern
        monthly_avg = data.groupby(data.index.month)[target_col].mean()
        monthly_avg.plot(ax=axes[1, 0], kind='bar', title='Average Demand by Month')
        axes[1, 0].set_xlabel('Month')
        
        # Heatmap of hour vs day of week
        pivot_data = data.pivot_table(
            values=target_col, 
            index=data.index.hour, 
            columns=data.index.dayofweek, 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, ax=axes[1, 1], cmap='YlOrRd', cbar_kws={'label': 'Demand'})
        axes[1, 1].set_title('Demand Heatmap: Hour vs Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Hour of Day')
        
        plt.tight_layout()
        
        return decomposition
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': {k: v for k, v in self.models.items() if k not in ['lstm', 'cnn_lstm']},
            'scalers': self.scalers,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }
        
        # Save traditional models
        joblib.dump(model_data, f"{filepath}_traditional.joblib")
        
        # Save deep learning models separately
        for model_name in ['lstm', 'cnn_lstm']:
            if model_name in self.models:
                self.models[model_name].save(f"{filepath}_{model_name}.h5")
        
        print(f"💾 Forecasting models saved to {filepath}")
    
    @classmethod
    def load_models(cls, filepath: str):
        """Load trained models"""
        # Load traditional models
        model_data = joblib.load(f"{filepath}_traditional.joblib")
        
        instance = cls(
            sequence_length=model_data['sequence_length'],
            forecast_horizon=model_data['forecast_horizon']
        )
        
        instance.models = model_data['models']
        instance.scalers = model_data['scalers']
        instance.is_fitted = model_data['is_fitted']
        
        # Load deep learning models
        for model_name in ['lstm', 'cnn_lstm']:
            try:
                model_path = f"{filepath}_{model_name}.h5"
                instance.models[model_name] = tf.keras.models.load_model(model_path)
            except:
                pass  # Model doesn't exist
        
        print(f"📥 Forecasting models loaded from {filepath}")
        return instance


def train_demand_forecasting_system():
    """Complete training pipeline for demand forecasting"""
    print("🔮 Starting Advanced Demand Forecasting System Training...")
    
    # Create forecaster instance
    forecaster = DemandForecaster(sequence_length=24, forecast_horizon=12)
    
    # Generate synthetic data (in real scenario, you'd load actual data)
    print("📊 Generating synthetic demand data...")
    data = forecaster._generate_synthetic_time_series(periods=8760)  # 1 year of hourly data
    
    # Split data for training and testing
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point].set_index('timestamp')
    test_data = data[split_point:].set_index('timestamp')
    
    print(f"   Training data: {len(train_data)} samples")
    print(f"   Test data: {len(test_data)} samples")
    
    # Train models
    forecaster.fit(train_data, target_col='demand')
    
    # Generate forecasts
    forecast_steps = 168  # 1 week ahead
    forecasts = forecaster.forecast(steps=forecast_steps, data=train_data, method='ensemble')
    
    print(f"\n🔮 Generated forecasts for next {forecast_steps} hours:")
    print(forecasts.head())
    
    # Evaluate models
    evaluation_results = forecaster.evaluate_models(test_data[:forecast_steps], target_col='demand')
    
    print(f"\n📊 Model Performance Evaluation:")
    for model_name, metrics in evaluation_results.items():
        print(f"   {model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value:.3f}")
    
    # Analyze seasonality
    print(f"\n📈 Analyzing seasonal patterns...")
    seasonality_analysis = forecaster.analyze_seasonality(train_data, target_col='demand')
    
    # Save models
    model_path = '/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/advanced_models/demand_forecaster'
    forecaster.save_models(model_path)
    
    return forecaster, forecasts, evaluation_results


if __name__ == "__main__":
    # Run the complete training pipeline
    forecaster, forecasts, results = train_demand_forecasting_system()
    print("\n🚀 Advanced Demand Forecasting System is ready for deployment!")
