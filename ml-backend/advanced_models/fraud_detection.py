"""
Advanced Fraud Detection System
Real-time anomaly detection and fraud prevention using multiple ML techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Union

# Core ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Network analysis
import networkx as nx

# Statistical methods
from scipy import stats
from scipy.spatial.distance import mahalanobis
import pyod
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF

warnings.filterwarnings('ignore')

class TransactionFraudDetector:
    """Advanced fraud detection system for transactions"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.fraud_rules = []
        self.is_fitted = False
        self.fraud_threshold = 0.5
        
    def _generate_transaction_data(self, n_transactions=10000):
        """Generate synthetic transaction data with fraud cases"""
        np.random.seed(self.random_state)
        
        # Normal transactions (90%)
        n_normal = int(n_transactions * 0.9)
        
        # Transaction amounts (log-normal distribution)
        normal_amounts = np.random.lognormal(4, 1.5, n_normal)
        normal_amounts = np.clip(normal_amounts, 1, 10000)
        
        # Transaction times (business hours weighted)
        hours = np.random.choice(24, n_normal, p=self._create_hourly_weights())
        
        # Merchant categories
        merchant_categories = np.random.choice(
            ['Grocery', 'Gas', 'Restaurant', 'Online', 'Retail', 'ATM'], 
            n_normal, 
            p=[0.25, 0.15, 0.20, 0.15, 0.20, 0.05]
        )
        
        # Customer locations
        locations = np.random.choice(['Home', 'Work', 'Travel'], n_normal, p=[0.6, 0.3, 0.1])
        
        # Days since last transaction
        days_since_last = np.random.exponential(2, n_normal)
        
        # Velocity features
        transactions_last_hour = np.random.poisson(0.5, n_normal)
        transactions_last_day = np.random.poisson(3, n_normal)
        
        # Device/IP features
        is_new_device = np.random.binomial(1, 0.1, n_normal)
        is_new_ip = np.random.binomial(1, 0.15, n_normal)
        
        # Geographic features
        distance_from_home = np.random.exponential(10, n_normal)  # km
        
        # Normal transaction DataFrame
        normal_df = pd.DataFrame({
            'transaction_id': range(1, n_normal + 1),
            'amount': normal_amounts,
            'hour': hours,
            'merchant_category': merchant_categories,
            'location_type': locations,
            'days_since_last_transaction': days_since_last,
            'transactions_last_hour': transactions_last_hour,
            'transactions_last_day': transactions_last_day,
            'is_new_device': is_new_device,
            'is_new_ip': is_new_ip,
            'distance_from_home_km': distance_from_home,
            'is_weekend': np.random.binomial(1, 2/7, n_normal),
            'is_fraud': 0
        })
        
        # Fraudulent transactions (10%)
        n_fraud = n_transactions - n_normal
        
        # Fraud patterns
        # 1. Large amounts
        fraud_amounts = np.random.lognormal(6, 1, n_fraud)
        fraud_amounts = np.clip(fraud_amounts, 500, 50000)
        
        # 2. Unusual hours (late night/early morning)
        fraud_hours = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], n_fraud)
        
        # 3. Suspicious merchant categories
        fraud_merchants = np.random.choice(
            ['Online', 'ATM', 'Cash_Advance', 'Unknown'], 
            n_fraud, 
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        # 4. Travel/unknown locations
        fraud_locations = np.random.choice(['Travel', 'Unknown'], n_fraud, p=[0.7, 0.3])
        
        # 5. Rapid transactions
        fraud_velocity_hour = np.random.poisson(5, n_fraud)
        fraud_velocity_day = np.random.poisson(15, n_fraud)
        
        # 6. New devices/IPs
        fraud_new_device = np.random.binomial(1, 0.8, n_fraud)
        fraud_new_ip = np.random.binomial(1, 0.9, n_fraud)
        
        # 7. Far from home
        fraud_distance = np.random.exponential(100, n_fraud)
        
        # Fraudulent transaction DataFrame
        fraud_df = pd.DataFrame({
            'transaction_id': range(n_normal + 1, n_transactions + 1),
            'amount': fraud_amounts,
            'hour': fraud_hours,
            'merchant_category': fraud_merchants,
            'location_type': fraud_locations,
            'days_since_last_transaction': np.random.exponential(0.1, n_fraud),  # Recent activity
            'transactions_last_hour': fraud_velocity_hour,
            'transactions_last_day': fraud_velocity_day,
            'is_new_device': fraud_new_device,
            'is_new_ip': fraud_new_ip,
            'distance_from_home_km': fraud_distance,
            'is_weekend': np.random.binomial(1, 0.5, n_fraud),
            'is_fraud': 1
        })
        
        # Combine and shuffle
        transaction_data = pd.concat([normal_df, fraud_df], ignore_index=True)
        transaction_data = transaction_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return transaction_data
    
    def _create_hourly_weights(self):
        """Create realistic hourly transaction weights"""
        # Business hours have higher probability
        weights = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM
            0.03, 0.05, 0.08, 0.10, 0.09, 0.08,  # 6-11 AM
            0.09, 0.08, 0.07, 0.06, 0.07, 0.08,  # 12-5 PM
            0.09, 0.08, 0.06, 0.04, 0.03, 0.02   # 6-11 PM
        ])
        return weights / weights.sum()
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for fraud detection"""
        df = data.copy()
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_zscore'] = stats.zscore(df['amount'])
        df['is_large_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_small_amount'] = (df['amount'] < df['amount'].quantile(0.05)).astype(int)
        
        # Time-based features
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Velocity features
        df['velocity_ratio'] = df['transactions_last_hour'] / (df['transactions_last_day'] + 1)
        df['is_high_velocity'] = (df['transactions_last_hour'] > 3).astype(int)
        
        # Location features
        df['is_travel_location'] = (df['location_type'] == 'Travel').astype(int)
        df['is_far_from_home'] = (df['distance_from_home_km'] > 50).astype(int)
        
        # Device/IP features
        df['new_device_ip'] = df['is_new_device'] & df['is_new_ip']
        
        # Recency features
        df['is_recent_activity'] = (df['days_since_last_transaction'] < 1).astype(int)
        df['is_dormant_reactivation'] = (df['days_since_last_transaction'] > 30).astype(int)
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['merchant_category', 'location_type']
        
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = le.fit_transform(df[col])
        
        return df
    
    def _create_fraud_rules(self, data: pd.DataFrame) -> List[Dict]:
        """Create rule-based fraud detection rules"""
        rules = [
            {
                'name': 'Large Amount + New Device',
                'condition': lambda df: (df['amount'] > 1000) & (df['is_new_device'] == 1),
                'risk_score': 0.8
            },
            {
                'name': 'Night Transaction + High Velocity',
                'condition': lambda df: (df['is_night_time'] == 1) & (df['transactions_last_hour'] > 2),
                'risk_score': 0.7
            },
            {
                'name': 'Travel Location + Large Amount',
                'condition': lambda df: (df['is_travel_location'] == 1) & (df['amount'] > 500),
                'risk_score': 0.6
            },
            {
                'name': 'New IP + ATM Transaction',
                'condition': lambda df: (df['is_new_ip'] == 1) & (df['merchant_category'] == 'ATM'),
                'risk_score': 0.9
            },
            {
                'name': 'Multiple Cards Same Hour',
                'condition': lambda df: df['transactions_last_hour'] > 5,
                'risk_score': 0.8
            },
            {
                'name': 'Dormant Account Reactivation + Large Amount',
                'condition': lambda df: (df['is_dormant_reactivation'] == 1) & (df['amount'] > 200),
                'risk_score': 0.5
            },
            {
                'name': 'Far From Home + New Device',
                'condition': lambda df: (df['is_far_from_home'] == 1) & (df['is_new_device'] == 1),
                'risk_score': 0.7
            }
        ]
        
        return rules
    
    def fit(self, data: pd.DataFrame = None):
        """Fit fraud detection models"""
        print("🚨 Training Advanced Fraud Detection System...")
        
        # Generate data if not provided
        if data is None:
            print("📊 Generating synthetic transaction data...")
            data = self._generate_transaction_data()
        
        # Engineer features
        data_enhanced = self._engineer_features(data)
        
        # Create fraud rules
        self.fraud_rules = self._create_fraud_rules(data_enhanced)
        
        # Prepare features for ML models
        feature_cols = [
            'amount_log', 'hour', 'transactions_last_hour', 'transactions_last_day',
            'is_new_device', 'is_new_ip', 'distance_from_home_km', 'is_weekend',
            'is_business_hours', 'is_night_time', 'velocity_ratio', 'is_high_velocity',
            'is_travel_location', 'is_far_from_home', 'new_device_ip',
            'is_recent_activity', 'is_dormant_reactivation',
            'merchant_category_encoded', 'location_type_encoded'
        ]
        
        # Ensure all features exist
        feature_cols = [col for col in feature_cols if col in data_enhanced.columns]
        
        X = data_enhanced[feature_cols]
        y = data_enhanced['is_fraud']
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_std = self.scalers['standard'].fit_transform(X_train)
        X_test_std = self.scalers['standard'].transform(X_test)
        
        X_train_robust = self.scalers['robust'].fit_transform(X_train)
        X_test_robust = self.scalers['robust'].transform(X_test)
        
        # Train multiple anomaly detection models
        self._train_anomaly_models(X_train_std, X_train_robust, y_train)
        
        # Train supervised models
        self._train_supervised_models(X_train_std, y_train)
        
        # Train neural network
        self._train_neural_network(X_train_std, y_train, X_test_std, y_test)
        
        # Evaluate models
        self._evaluate_models(X_test_std, y_test)
        
        self.is_fitted = True
        print("✅ Fraud Detection System Training Complete!")
        
        return self
    
    def _train_anomaly_models(self, X_train_std, X_train_robust, y_train):
        """Train unsupervised anomaly detection models"""
        print("  🔍 Training anomaly detection models...")
        
        # Use only normal transactions for unsupervised training
        normal_mask = y_train == 0
        X_normal_std = X_train_std[normal_mask]
        X_normal_robust = X_train_robust[normal_mask]
        
        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=self.random_state,
            n_estimators=200
        )
        self.models['isolation_forest'].fit(X_normal_std)
        
        # One-Class SVM
        self.models['one_class_svm'] = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        self.models['one_class_svm'].fit(X_normal_std)
        
        # Local Outlier Factor
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        self.models['lof'].fit(X_normal_std)
        
        # Elliptic Envelope
        self.models['elliptic_envelope'] = EllipticEnvelope(
            contamination=0.1,
            random_state=self.random_state
        )
        self.models['elliptic_envelope'].fit(X_normal_robust)
        
        # AutoEncoder (PyOD)
        try:
            self.models['autoencoder'] = AutoEncoder(
                hidden_neurons=[32, 16, 8, 16, 32],
                contamination=0.1,
                epochs=100,
                verbose=0
            )
            self.models['autoencoder'].fit(X_normal_std)
            print("    ✅ AutoEncoder fitted")
        except Exception as e:
            print(f"    ❌ AutoEncoder failed: {e}")
        
        print("    ✅ Anomaly detection models fitted")
    
    def _train_supervised_models(self, X_train_std, y_train):
        """Train supervised fraud detection models"""
        print("  🎯 Training supervised models...")
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.models['random_forest'].fit(X_train_std, y_train)
        
        print("    ✅ Supervised models fitted")
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network for fraud detection"""
        print("  🧠 Training neural network...")
        
        try:
            # Build model
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Train
            callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            self.models['neural_network'] = model
            print("    ✅ Neural network fitted")
            
        except Exception as e:
            print(f"    ❌ Neural network failed: {e}")
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("  📊 Evaluating models...")
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network':
                    y_pred_proba = model.predict(X_test).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                elif name in ['isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope']:
                    # Anomaly models return -1 for outliers, 1 for inliers
                    y_pred_anomaly = model.predict(X_test)
                    y_pred = (y_pred_anomaly == -1).astype(int)
                    y_pred_proba = None
                elif name == 'autoencoder':
                    y_pred_proba = model.decision_function(X_test)
                    y_pred = (y_pred_proba > model.threshold_).astype(int)
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = np.mean(y_pred == y_test)
                
                if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        print(f"    📈 {name}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
                    except:
                        print(f"    📈 {name}: Accuracy={accuracy:.3f}")
                else:
                    print(f"    📈 {name}: Accuracy={accuracy:.3f}")
                    
            except Exception as e:
                print(f"    ❌ {name} evaluation failed: {e}")
    
    def predict_fraud_probability(self, transaction_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict fraud probability using ensemble of models"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first")
        
        # Engineer features
        data_enhanced = self._engineer_features(transaction_data)
        
        # Prepare features
        X = data_enhanced[self.feature_names]
        X_std = self.scalers['standard'].transform(X)
        
        predictions = {}
        
        # Rule-based predictions
        rule_scores = self._apply_fraud_rules(data_enhanced)
        predictions['rules'] = rule_scores
        
        # Model predictions
        for name, model in self.models.items():
            try:
                if name == 'neural_network':
                    pred = model.predict(X_std).flatten()
                elif name in ['isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope']:
                    # Convert anomaly scores to probabilities
                    anomaly_pred = model.predict(X_std)
                    pred = (anomaly_pred == -1).astype(float)
                elif name == 'autoencoder':
                    scores = model.decision_function(X_std)
                    # Normalize scores to 0-1 range
                    pred = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    pred = model.predict_proba(X_std)[:, 1]
                
                predictions[name] = pred
                
            except Exception as e:
                print(f"Prediction failed for {name}: {e}")
        
        # Ensemble prediction (weighted average)
        ensemble_weights = {
            'rules': 0.3,
            'random_forest': 0.25,
            'neural_network': 0.2,
            'isolation_forest': 0.15,
            'autoencoder': 0.1
        }
        
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, weight in ensemble_weights.items():
            if model_name in predictions:
                ensemble_pred += predictions[model_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def _apply_fraud_rules(self, data: pd.DataFrame) -> np.ndarray:
        """Apply rule-based fraud detection"""
        rule_scores = np.zeros(len(data))
        
        for rule in self.fraud_rules:
            try:
                condition_mask = rule['condition'](data)
                rule_scores[condition_mask] = np.maximum(
                    rule_scores[condition_mask], 
                    rule['risk_score']
                )
            except Exception as e:
                print(f"Rule '{rule['name']}' failed: {e}")
        
        return rule_scores
    
    def analyze_transaction(self, transaction: Dict) -> Dict:
        """Analyze a single transaction for fraud indicators"""
        # Convert to DataFrame
        transaction_df = pd.DataFrame([transaction])
        
        # Get fraud predictions
        predictions = self.predict_fraud_probability(transaction_df)
        ensemble_score = predictions['ensemble'][0]
        
        # Determine risk level
        if ensemble_score >= 0.8:
            risk_level = 'HIGH'
            action = 'BLOCK'
        elif ensemble_score >= 0.6:
            risk_level = 'MEDIUM'
            action = 'REVIEW'
        elif ensemble_score >= 0.3:
            risk_level = 'LOW'
            action = 'MONITOR'
        else:
            risk_level = 'MINIMAL'
            action = 'APPROVE'
        
        # Find triggered rules
        data_enhanced = self._engineer_features(transaction_df)
        triggered_rules = []
        
        for rule in self.fraud_rules:
            try:
                if rule['condition'](data_enhanced).iloc[0]:
                    triggered_rules.append({
                        'rule': rule['name'],
                        'risk_score': rule['risk_score']
                    })
            except:
                pass
        
        return {
            'transaction_id': transaction.get('transaction_id', 'Unknown'),
            'fraud_score': float(ensemble_score),
            'risk_level': risk_level,
            'recommended_action': action,
            'triggered_rules': triggered_rules,
            'model_scores': {k: float(v[0]) for k, v in predictions.items()},
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_fraud_insights(self, transaction_data: pd.DataFrame) -> Dict:
        """Get comprehensive fraud insights"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first")
        
        predictions = self.predict_fraud_probability(transaction_data)
        fraud_scores = predictions['ensemble']
        
        # Risk distribution
        high_risk = np.sum(fraud_scores >= 0.8)
        medium_risk = np.sum((fraud_scores >= 0.6) & (fraud_scores < 0.8))
        low_risk = np.sum((fraud_scores >= 0.3) & (fraud_scores < 0.6))
        minimal_risk = np.sum(fraud_scores < 0.3)
        
        # Feature analysis
        data_enhanced = self._engineer_features(transaction_data)
        
        insights = {
            'total_transactions': len(transaction_data),
            'risk_distribution': {
                'high_risk': int(high_risk),
                'medium_risk': int(medium_risk),
                'low_risk': int(low_risk),
                'minimal_risk': int(minimal_risk)
            },
            'average_fraud_score': float(np.mean(fraud_scores)),
            'fraud_rate_estimate': float(np.mean(fraud_scores > 0.5)),
            'top_risk_factors': self._identify_risk_factors(data_enhanced, fraud_scores),
            'hourly_risk_pattern': self._analyze_hourly_patterns(data_enhanced, fraud_scores),
            'merchant_risk_analysis': self._analyze_merchant_risk(data_enhanced, fraud_scores)
        }
        
        return insights
    
    def _identify_risk_factors(self, data: pd.DataFrame, fraud_scores: np.ndarray) -> List[Dict]:
        """Identify top risk factors"""
        risk_factors = []
        
        # Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                try:
                    correlation = np.corrcoef(data[col], fraud_scores)[0, 1]
                    if not np.isnan(correlation):
                        risk_factors.append({
                            'factor': col,
                            'correlation': abs(correlation),
                            'direction': 'positive' if correlation > 0 else 'negative'
                        })
                except:
                    pass
        
        # Sort by correlation strength
        risk_factors.sort(key=lambda x: x['correlation'], reverse=True)
        
        return risk_factors[:10]
    
    def _analyze_hourly_patterns(self, data: pd.DataFrame, fraud_scores: np.ndarray) -> Dict:
        """Analyze fraud risk by hour"""
        hourly_risk = {}
        
        if 'hour' in data.columns:
            for hour in range(24):
                hour_mask = data['hour'] == hour
                if np.sum(hour_mask) > 0:
                    avg_risk = np.mean(fraud_scores[hour_mask])
                    hourly_risk[str(hour)] = float(avg_risk)
        
        return hourly_risk
    
    def _analyze_merchant_risk(self, data: pd.DataFrame, fraud_scores: np.ndarray) -> Dict:
        """Analyze fraud risk by merchant category"""
        merchant_risk = {}
        
        if 'merchant_category' in data.columns:
            for category in data['merchant_category'].unique():
                category_mask = data['merchant_category'] == category
                if np.sum(category_mask) > 0:
                    avg_risk = np.mean(fraud_scores[category_mask])
                    merchant_risk[category] = float(avg_risk)
        
        return merchant_risk
    
    def save_models(self, filepath: str):
        """Save fraud detection models"""
        model_data = {
            'models': {k: v for k, v in self.models.items() if k != 'neural_network'},
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'fraud_rules': self.fraud_rules,
            'fraud_threshold': self.fraud_threshold,
            'is_fitted': self.is_fitted
        }
        
        # Save traditional models
        joblib.dump(model_data, f"{filepath}_fraud_models.joblib")
        
        # Save neural network separately
        if 'neural_network' in self.models:
            self.models['neural_network'].save(f"{filepath}_neural_network.h5")
        
        print(f"💾 Fraud detection models saved to {filepath}")
    
    @classmethod
    def load_models(cls, filepath: str):
        """Load trained models"""
        # Load traditional models
        model_data = joblib.load(f"{filepath}_fraud_models.joblib")
        
        instance = cls()
        instance.models = model_data['models']
        instance.scalers = model_data['scalers']
        instance.feature_names = model_data['feature_names']
        instance.fraud_rules = model_data['fraud_rules']
        instance.fraud_threshold = model_data['fraud_threshold']
        instance.is_fitted = model_data['is_fitted']
        
        # Load neural network
        try:
            neural_network_path = f"{filepath}_neural_network.h5"
            instance.models['neural_network'] = tf.keras.models.load_model(neural_network_path)
        except:
            pass  # Neural network doesn't exist
        
        print(f"📥 Fraud detection models loaded from {filepath}")
        return instance


class NetworkAnalysisFraud:
    """Network-based fraud detection using graph analysis"""
    
    def __init__(self):
        self.transaction_graph = None
        self.community_detector = None
        self.anomaly_threshold = 2.0
        
    def build_transaction_network(self, transaction_data: pd.DataFrame):
        """Build transaction network graph"""
        # Create graph from transactions
        G = nx.Graph()
        
        # Add nodes (customers/merchants)
        customers = transaction_data['customer_id'].unique() if 'customer_id' in transaction_data.columns else []
        merchants = transaction_data['merchant_id'].unique() if 'merchant_id' in transaction_data.columns else []
        
        # Add customer nodes
        for customer in customers:
            customer_data = transaction_data[transaction_data['customer_id'] == customer]
            G.add_node(f"C_{customer}", 
                      type='customer',
                      total_amount=customer_data['amount'].sum(),
                      transaction_count=len(customer_data))
        
        # Add merchant nodes
        for merchant in merchants:
            merchant_data = transaction_data[transaction_data['merchant_id'] == merchant]
            G.add_node(f"M_{merchant}",
                      type='merchant',
                      total_amount=merchant_data['amount'].sum(),
                      transaction_count=len(merchant_data))
        
        # Add edges (transactions)
        if 'customer_id' in transaction_data.columns and 'merchant_id' in transaction_data.columns:
            for _, transaction in transaction_data.iterrows():
                customer_node = f"C_{transaction['customer_id']}"
                merchant_node = f"M_{transaction['merchant_id']}"
                
                if G.has_edge(customer_node, merchant_node):
                    # Update existing edge
                    G[customer_node][merchant_node]['weight'] += transaction['amount']
                    G[customer_node][merchant_node]['count'] += 1
                else:
                    # Add new edge
                    G.add_edge(customer_node, merchant_node,
                              weight=transaction['amount'],
                              count=1)
        
        self.transaction_graph = G
        return G
    
    def detect_network_anomalies(self) -> List[Dict]:
        """Detect anomalies in the transaction network"""
        if self.transaction_graph is None:
            raise ValueError("Transaction network must be built first")
        
        anomalies = []
        G = self.transaction_graph
        
        # Calculate network metrics
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        
        # Detect anomalies based on network properties
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # High degree anomaly (too many connections)
            degree_zscore = (degrees[node] - np.mean(list(degrees.values()))) / np.std(list(degrees.values()))
            
            # High betweenness centrality (potential money laundering hub)
            betweenness_zscore = (betweenness[node] - np.mean(list(betweenness.values()))) / np.std(list(betweenness.values()))
            
            # Low clustering coefficient (unusual connection pattern)
            clustering_zscore = (clustering[node] - np.mean(list(clustering.values()))) / np.std(list(clustering.values()))
            
            # Check for anomalies
            anomaly_scores = []
            
            if abs(degree_zscore) > self.anomaly_threshold:
                anomaly_scores.append(('high_degree', abs(degree_zscore)))
            
            if abs(betweenness_zscore) > self.anomaly_threshold:
                anomaly_scores.append(('high_betweenness', abs(betweenness_zscore)))
            
            if abs(clustering_zscore) > self.anomaly_threshold:
                anomaly_scores.append(('unusual_clustering', abs(clustering_zscore)))
            
            if anomaly_scores:
                anomalies.append({
                    'node': node,
                    'type': node_data.get('type', 'unknown'),
                    'anomalies': anomaly_scores,
                    'degree': degrees[node],
                    'betweenness_centrality': betweenness[node],
                    'clustering_coefficient': clustering[node]
                })
        
        return sorted(anomalies, key=lambda x: sum([score for _, score in x['anomalies']]), reverse=True)


def create_fraud_detection_system():
    """Create and train the complete fraud detection system"""
    print("🚨 Building Advanced Fraud Detection System...")
    
    # Initialize fraud detector
    fraud_detector = TransactionFraudDetector(random_state=42)
    
    # Train the system
    fraud_detector.fit()
    
    # Test with sample transactions
    print(f"\n🧪 Testing Fraud Detection System:")
    
    # Normal transaction
    normal_transaction = {
        'transaction_id': 'T001',
        'amount': 85.50,
        'hour': 14,
        'merchant_category': 'Grocery',
        'location_type': 'Home',
        'days_since_last_transaction': 2,
        'transactions_last_hour': 0,
        'transactions_last_day': 1,
        'is_new_device': 0,
        'is_new_ip': 0,
        'distance_from_home_km': 5,
        'is_weekend': 0
    }
    
    # Suspicious transaction
    suspicious_transaction = {
        'transaction_id': 'T002',
        'amount': 2500.00,
        'hour': 2,
        'merchant_category': 'ATM',
        'location_type': 'Travel',
        'days_since_last_transaction': 0.1,
        'transactions_last_hour': 3,
        'transactions_last_day': 8,
        'is_new_device': 1,
        'is_new_ip': 1,
        'distance_from_home_km': 200,
        'is_weekend': 1
    }
    
    # Analyze transactions
    normal_analysis = fraud_detector.analyze_transaction(normal_transaction)
    suspicious_analysis = fraud_detector.analyze_transaction(suspicious_transaction)
    
    print(f"\n📊 Normal Transaction Analysis:")
    print(f"   Fraud Score: {normal_analysis['fraud_score']:.3f}")
    print(f"   Risk Level: {normal_analysis['risk_level']}")
    print(f"   Action: {normal_analysis['recommended_action']}")
    print(f"   Triggered Rules: {len(normal_analysis['triggered_rules'])}")
    
    print(f"\n🚨 Suspicious Transaction Analysis:")
    print(f"   Fraud Score: {suspicious_analysis['fraud_score']:.3f}")
    print(f"   Risk Level: {suspicious_analysis['risk_level']}")
    print(f"   Action: {suspicious_analysis['recommended_action']}")
    print(f"   Triggered Rules: {len(suspicious_analysis['triggered_rules'])}")
    for rule in suspicious_analysis['triggered_rules'][:3]:
        print(f"     - {rule['rule']} (Score: {rule['risk_score']})")
    
    # Generate insights from sample data
    sample_data = fraud_detector._generate_transaction_data(1000)
    insights = fraud_detector.get_fraud_insights(sample_data)
    
    print(f"\n📈 Fraud Detection Insights:")
    print(f"   Total Transactions: {insights['total_transactions']}")
    print(f"   High Risk: {insights['risk_distribution']['high_risk']}")
    print(f"   Medium Risk: {insights['risk_distribution']['medium_risk']}")
    print(f"   Average Fraud Score: {insights['average_fraud_score']:.3f}")
    print(f"   Estimated Fraud Rate: {insights['fraud_rate_estimate']:.2%}")
    
    print(f"\n🔍 Top Risk Factors:")
    for i, factor in enumerate(insights['top_risk_factors'][:5]):
        print(f"   {i+1}. {factor['factor']}: {factor['correlation']:.3f} correlation")
    
    # Save models
    model_path = '/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/advanced_models/fraud_detection'
    fraud_detector.save_models(model_path)
    
    return fraud_detector


if __name__ == "__main__":
    # Run the complete fraud detection system
    fraud_system = create_fraud_detection_system()
    print("\n🚀 Advanced Fraud Detection System is ready for deployment!")
