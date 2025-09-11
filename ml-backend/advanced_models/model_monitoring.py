"""
Advanced Model Monitoring and Auto-Retraining System
MLOps pipeline for model performance monitoring, drift detection, and automated retraining
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import sqlite3
import threading
import time
from pathlib import Path

# Core ML libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Statistical tests for drift detection
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import scipy.stats as scipy_stats

# Advanced drift detection
try:
    from alibi_detect import KSDrift, MMDDrift, TabularDrift
except ImportError:
    print("Warning: alibi-detect not installed. Some drift detection features may not work.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceMonitor:
    """Monitor model performance metrics over time"""
    
    def __init__(self, model_name: str, model_type: str = 'regression'):
        self.model_name = model_name
        self.model_type = model_type  # 'regression' or 'classification'
        self.performance_history = []
        self.alert_thresholds = self._set_default_thresholds()
        self.db_path = f'/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/monitoring/{model_name}_monitoring.db'
        self._init_database()
    
    def _set_default_thresholds(self) -> Dict[str, float]:
        """Set default performance alert thresholds"""
        if self.model_type == 'regression':
            return {
                'r2_score_min': 0.7,
                'rmse_max': 100.0,
                'mae_max': 50.0,
                'performance_drop_threshold': 0.1  # 10% drop in R2
            }
        else:  # classification
            return {
                'accuracy_min': 0.8,
                'precision_min': 0.75,
                'recall_min': 0.75,
                'f1_min': 0.75,
                'performance_drop_threshold': 0.1  # 10% drop in accuracy
            }
    
    def _init_database(self):
        """Initialize SQLite database for storing monitoring data"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                prediction_count INTEGER,
                data_size INTEGER,
                notes TEXT
            )
        ''')
        
        # Create drift detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                drift_type TEXT NOT NULL,
                p_value REAL,
                statistic REAL,
                is_drift BOOLEAN,
                feature_name TEXT,
                drift_magnitude REAL,
                notes TEXT
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_value REAL,
                threshold REAL,
                is_resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       prediction_count: int = None, data_size: int = None, notes: str = None):
        """Log performance metrics"""
        timestamp = datetime.now()
        
        if self.model_type == 'regression':
            metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred)
            }
        else:  # classification
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        
        # Store in memory
        performance_record = {
            'timestamp': timestamp,
            'metrics': metrics,
            'prediction_count': prediction_count,
            'data_size': data_size,
            'notes': notes
        }
        self.performance_history.append(performance_record)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            cursor.execute('''
                INSERT INTO performance_metrics 
                (metric_name, metric_value, prediction_count, data_size, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (metric_name, metric_value, prediction_count, data_size, notes))
        
        conn.commit()
        conn.close()
        
        # Check for alerts
        self._check_performance_alerts(metrics)
        
        logger.info(f"Logged performance for {self.model_name}: {metrics}")
    
    def _check_performance_alerts(self, metrics: Dict[str, float]):
        """Check if performance metrics trigger alerts"""
        alerts = []
        
        for metric_name, metric_value in metrics.items():
            threshold_key = f"{metric_name}_min" if metric_name in ['r2_score', 'accuracy', 'precision', 'recall', 'f1_score'] else f"{metric_name}_max"
            
            if threshold_key in self.alert_thresholds:
                threshold = self.alert_thresholds[threshold_key]
                
                if 'min' in threshold_key and metric_value < threshold:
                    alerts.append({
                        'type': 'performance_degradation',
                        'severity': 'high' if metric_value < threshold * 0.9 else 'medium',
                        'message': f"{metric_name} ({metric_value:.3f}) below threshold ({threshold})",
                        'metric_value': metric_value,
                        'threshold': threshold
                    })
                elif 'max' in threshold_key and metric_value > threshold:
                    alerts.append({
                        'type': 'performance_degradation',
                        'severity': 'high' if metric_value > threshold * 1.1 else 'medium',
                        'message': f"{metric_name} ({metric_value:.3f}) above threshold ({threshold})",
                        'metric_value': metric_value,
                        'threshold': threshold
                    })
        
        # Check for performance drops compared to recent history
        if len(self.performance_history) > 5:
            self._check_performance_trends(metrics, alerts)
        
        # Log alerts
        if alerts:
            self._log_alerts(alerts)
    
    def _check_performance_trends(self, current_metrics: Dict[str, float], alerts: List[Dict]):
        """Check for performance trends and drops"""
        recent_records = self.performance_history[-5:]  # Last 5 records
        
        for metric_name, current_value in current_metrics.items():
            recent_values = [record['metrics'].get(metric_name, 0) for record in recent_records]
            recent_avg = np.mean(recent_values)
            
            # Check for significant drop
            if metric_name in ['r2_score', 'accuracy', 'precision', 'recall', 'f1_score']:
                # For metrics where higher is better
                drop_ratio = (recent_avg - current_value) / recent_avg if recent_avg > 0 else 0
                if drop_ratio > self.alert_thresholds['performance_drop_threshold']:
                    alerts.append({
                        'type': 'performance_trend',
                        'severity': 'high',
                        'message': f"{metric_name} dropped {drop_ratio:.1%} from recent average ({recent_avg:.3f} to {current_value:.3f})",
                        'metric_value': current_value,
                        'threshold': recent_avg
                    })
            else:
                # For metrics where lower is better (RMSE, MAE)
                increase_ratio = (current_value - recent_avg) / recent_avg if recent_avg > 0 else 0
                if increase_ratio > self.alert_thresholds['performance_drop_threshold']:
                    alerts.append({
                        'type': 'performance_trend',
                        'severity': 'high',
                        'message': f"{metric_name} increased {increase_ratio:.1%} from recent average ({recent_avg:.3f} to {current_value:.3f})",
                        'metric_value': current_value,
                        'threshold': recent_avg
                    })
    
    def _log_alerts(self, alerts: List[Dict]):
        """Log alerts to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for alert in alerts:
            cursor.execute('''
                INSERT INTO alerts (alert_type, severity, message, metric_value, threshold)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert['type'], alert['severity'], alert['message'], 
                  alert.get('metric_value'), alert.get('threshold')))
        
        conn.commit()
        conn.close()
        
        # Log to console
        for alert in alerts:
            logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for the last N days"""
        conn = sqlite3.connect(self.db_path)
        
        # Query performance metrics
        metrics_df = pd.read_sql_query('''
            SELECT * FROM performance_metrics 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        
        # Query alerts
        alerts_df = pd.read_sql_query('''
            SELECT * FROM alerts 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        
        conn.close()
        
        if len(metrics_df) == 0:
            return {'error': 'No performance data available'}
        
        # Calculate summary statistics
        summary = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'period_days': days,
            'total_evaluations': len(metrics_df['timestamp'].unique()),
            'latest_metrics': {},
            'average_metrics': {},
            'trend_analysis': {},
            'alert_summary': {
                'total_alerts': len(alerts_df),
                'high_severity': len(alerts_df[alerts_df['severity'] == 'high']),
                'medium_severity': len(alerts_df[alerts_df['severity'] == 'medium']),
                'recent_alerts': alerts_df.head(5).to_dict('records') if len(alerts_df) > 0 else []
            }
        }
        
        # Latest metrics
        latest_timestamp = metrics_df['timestamp'].max()
        latest_metrics = metrics_df[metrics_df['timestamp'] == latest_timestamp]
        for _, row in latest_metrics.iterrows():
            summary['latest_metrics'][row['metric_name']] = row['metric_value']
        
        # Average metrics
        for metric in metrics_df['metric_name'].unique():
            metric_data = metrics_df[metrics_df['metric_name'] == metric]
            summary['average_metrics'][metric] = {
                'mean': metric_data['metric_value'].mean(),
                'std': metric_data['metric_value'].std(),
                'min': metric_data['metric_value'].min(),
                'max': metric_data['metric_value'].max()
            }
        
        return summary


class DataDriftDetector:
    """Detect data drift in model inputs"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.reference_data = None
        self.feature_names = None
        self.drift_detectors = {}
        self.db_path = f'/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/monitoring/{model_name}_drift.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize database for drift detection"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feature_name TEXT,
                drift_method TEXT,
                p_value REAL,
                statistic REAL,
                is_drift BOOLEAN,
                drift_magnitude REAL,
                sample_size INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def set_reference_data(self, X_reference: np.ndarray, feature_names: List[str] = None):
        """Set reference data for drift detection"""
        self.reference_data = X_reference
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_reference.shape[1])]
        
        logger.info(f"Set reference data with {X_reference.shape[0]} samples and {X_reference.shape[1]} features")
    
    def detect_drift(self, X_new: np.ndarray, method: str = 'ks_test', 
                    significance_level: float = 0.05) -> Dict:
        """Detect drift in new data"""
        if self.reference_data is None:
            raise ValueError("Reference data must be set before drift detection")
        
        if X_new.shape[1] != self.reference_data.shape[1]:
            raise ValueError("New data must have same number of features as reference data")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'significance_level': significance_level,
            'overall_drift': False,
            'feature_drift': {},
            'summary': {}
        }
        
        n_features = X_new.shape[1]
        drift_count = 0
        
        for i in range(n_features):
            feature_name = self.feature_names[i]
            ref_feature = self.reference_data[:, i]
            new_feature = X_new[:, i]
            
            if method == 'ks_test':
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(ref_feature, new_feature)
                is_drift = p_value < significance_level
                
                # Calculate drift magnitude (effect size)
                drift_magnitude = abs(np.mean(new_feature) - np.mean(ref_feature)) / np.std(ref_feature)
                
            elif method == 'chi2_test' and self._is_categorical_feature(ref_feature, new_feature):
                # Chi-square test for categorical features
                try:
                    # Create contingency table
                    ref_counts = pd.Series(ref_feature).value_counts()
                    new_counts = pd.Series(new_feature).value_counts()
                    
                    # Align indices
                    all_values = set(ref_counts.index) | set(new_counts.index)
                    ref_aligned = [ref_counts.get(v, 0) for v in all_values]
                    new_aligned = [new_counts.get(v, 0) for v in all_values]
                    
                    contingency_table = np.array([ref_aligned, new_aligned])
                    statistic, p_value, _, _ = chi2_contingency(contingency_table)
                    is_drift = p_value < significance_level
                    drift_magnitude = np.sqrt(statistic / (np.sum(contingency_table)))
                    
                except Exception as e:
                    # Fallback to KS test
                    statistic, p_value = ks_2samp(ref_feature, new_feature)
                    is_drift = p_value < significance_level
                    drift_magnitude = abs(np.mean(new_feature) - np.mean(ref_feature)) / np.std(ref_feature)
            
            elif method == 'wasserstein':
                # Wasserstein distance
                from scipy.stats import wasserstein_distance
                statistic = wasserstein_distance(ref_feature, new_feature)
                
                # Bootstrap to get p-value
                p_value = self._bootstrap_wasserstein_test(ref_feature, new_feature, statistic)
                is_drift = p_value < significance_level
                drift_magnitude = statistic / (np.std(ref_feature) + 1e-8)
            
            else:
                # Default to KS test
                statistic, p_value = ks_2samp(ref_feature, new_feature)
                is_drift = p_value < significance_level
                drift_magnitude = abs(np.mean(new_feature) - np.mean(ref_feature)) / np.std(ref_feature)
            
            # Store feature drift result
            drift_results['feature_drift'][feature_name] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_drift': is_drift,
                'drift_magnitude': float(drift_magnitude),
                'drift_severity': self._classify_drift_severity(drift_magnitude)
            }
            
            if is_drift:
                drift_count += 1
            
            # Log to database
            self._log_drift_result(feature_name, method, p_value, statistic, 
                                 is_drift, drift_magnitude, len(X_new))
        
        # Overall drift assessment
        drift_ratio = drift_count / n_features
        drift_results['overall_drift'] = drift_ratio > 0.2  # If more than 20% of features show drift
        
        drift_results['summary'] = {
            'total_features': n_features,
            'features_with_drift': drift_count,
            'drift_ratio': drift_ratio,
            'avg_drift_magnitude': np.mean([
                result['drift_magnitude'] for result in drift_results['feature_drift'].values()
            ])
        }
        
        logger.info(f"Drift detection completed: {drift_count}/{n_features} features show drift")
        
        return drift_results
    
    def _is_categorical_feature(self, ref_feature: np.ndarray, new_feature: np.ndarray) -> bool:
        """Check if feature appears to be categorical"""
        # Simple heuristic: if number of unique values is less than 10% of samples
        ref_unique = len(np.unique(ref_feature))
        new_unique = len(np.unique(new_feature))
        ref_ratio = ref_unique / len(ref_feature)
        new_ratio = new_unique / len(new_feature)
        
        return ref_ratio < 0.1 and new_ratio < 0.1 and ref_unique < 20
    
    def _bootstrap_wasserstein_test(self, ref_data: np.ndarray, new_data: np.ndarray, 
                                   observed_stat: float, n_bootstrap: int = 1000) -> float:
        """Bootstrap test for Wasserstein distance"""
        from scipy.stats import wasserstein_distance
        
        # Combine data and shuffle
        combined = np.concatenate([ref_data, new_data])
        n_ref = len(ref_data)
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            shuffled = np.random.permutation(combined)
            boot_ref = shuffled[:n_ref]
            boot_new = shuffled[n_ref:]
            boot_stat = wasserstein_distance(boot_ref, boot_new)
            bootstrap_stats.append(boot_stat)
        
        # P-value is the proportion of bootstrap statistics >= observed
        p_value = np.mean(np.array(bootstrap_stats) >= observed_stat)
        return p_value
    
    def _classify_drift_severity(self, drift_magnitude: float) -> str:
        """Classify drift severity based on magnitude"""
        if drift_magnitude < 0.1:
            return 'low'
        elif drift_magnitude < 0.5:
            return 'medium'
        else:
            return 'high'
    
    def _log_drift_result(self, feature_name: str, method: str, p_value: float, 
                         statistic: float, is_drift: bool, drift_magnitude: float, sample_size: int):
        """Log drift result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_results 
            (feature_name, drift_method, p_value, statistic, is_drift, drift_magnitude, sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (feature_name, method, p_value, statistic, is_drift, drift_magnitude, sample_size))
        
        conn.commit()
        conn.close()


class AutoRetrainingManager:
    """Manage automatic model retraining"""
    
    def __init__(self, model_name: str, retraining_strategy: str = 'performance_based'):
        self.model_name = model_name
        self.retraining_strategy = retraining_strategy
        self.retraining_config = self._get_default_config()
        self.performance_monitor = None
        self.drift_detector = None
        self.is_running = False
        self.monitoring_thread = None
        
    def _get_default_config(self) -> Dict:
        """Get default retraining configuration"""
        return {
            'performance_threshold': 0.1,  # Trigger if performance drops by 10%
            'drift_threshold': 0.3,  # Trigger if 30% of features show drift
            'min_samples_for_retraining': 1000,
            'max_retraining_frequency': timedelta(days=7),  # At most once per week
            'evaluation_window': timedelta(days=30),
            'auto_approve_retraining': False,  # Require manual approval
            'backup_model_versions': 5
        }
    
    def set_monitors(self, performance_monitor: ModelPerformanceMonitor, 
                    drift_detector: DataDriftDetector):
        """Set monitoring components"""
        self.performance_monitor = performance_monitor
        self.drift_detector = drift_detector
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Started monitoring for {self.model_name}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info(f"Stopped monitoring for {self.model_name}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        check_interval = 3600  # Check every hour
        
        while self.is_running:
            try:
                self._check_retraining_conditions()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _check_retraining_conditions(self):
        """Check if retraining should be triggered"""
        if not self.performance_monitor:
            return
        
        # Get recent performance summary
        summary = self.performance_monitor.get_performance_summary(days=7)
        
        if 'error' in summary:
            return
        
        # Check performance degradation
        performance_trigger = self._check_performance_trigger(summary)
        
        # Check data drift (if drift detector is available)
        drift_trigger = False
        if self.drift_detector:
            # This would require new data to check drift
            # In practice, you'd have a data pipeline feeding new data
            pass
        
        # Check sample size
        sample_trigger = self._check_sample_trigger()
        
        # Evaluate retraining decision
        if performance_trigger or drift_trigger or sample_trigger:
            self._initiate_retraining({
                'performance_trigger': performance_trigger,
                'drift_trigger': drift_trigger,
                'sample_trigger': sample_trigger,
                'timestamp': datetime.now().isoformat()
            })
    
    def _check_performance_trigger(self, summary: Dict) -> bool:
        """Check if performance degradation triggers retraining"""
        if not summary.get('latest_metrics'):
            return False
        
        # Compare latest performance with average
        for metric_name, latest_value in summary['latest_metrics'].items():
            if metric_name in summary['average_metrics']:
                avg_info = summary['average_metrics'][metric_name]
                avg_value = avg_info['mean']
                
                # For metrics where higher is better
                if metric_name in ['r2_score', 'accuracy', 'precision', 'recall', 'f1_score']:
                    drop_ratio = (avg_value - latest_value) / avg_value if avg_value > 0 else 0
                    if drop_ratio > self.retraining_config['performance_threshold']:
                        logger.warning(f"Performance degradation detected in {metric_name}: {drop_ratio:.2%}")
                        return True
                
                # For metrics where lower is better
                elif metric_name in ['rmse', 'mae', 'mse']:
                    increase_ratio = (latest_value - avg_value) / avg_value if avg_value > 0 else 0
                    if increase_ratio > self.retraining_config['performance_threshold']:
                        logger.warning(f"Performance degradation detected in {metric_name}: {increase_ratio:.2%}")
                        return True
        
        return False
    
    def _check_sample_trigger(self) -> bool:
        """Check if enough new samples are available for retraining"""
        # This would check your data storage for new samples
        # For now, return False as we don't have a real data pipeline
        return False
    
    def _initiate_retraining(self, trigger_info: Dict):
        """Initiate model retraining process"""
        logger.info(f"Retraining triggered for {self.model_name}: {trigger_info}")
        
        # In a real system, this would:
        # 1. Fetch new training data
        # 2. Prepare data pipeline
        # 3. Train new model
        # 4. Evaluate new model
        # 5. Compare with current model
        # 6. Deploy if better (or queue for approval)
        
        # For now, just log the trigger
        retraining_record = {
            'model_name': self.model_name,
            'trigger_info': trigger_info,
            'status': 'triggered',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save retraining record
        self._save_retraining_record(retraining_record)
    
    def _save_retraining_record(self, record: Dict):
        """Save retraining record to database"""
        db_path = f'/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/monitoring/{self.model_name}_retraining.db'
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trigger_info TEXT,
                status TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            INSERT INTO retraining_log (trigger_info, status)
            VALUES (?, ?)
        ''', (json.dumps(record['trigger_info']), record['status']))
        
        conn.commit()
        conn.close()


class MLOpsManager:
    """Central MLOps management system"""
    
    def __init__(self):
        self.monitors = {}
        self.drift_detectors = {}
        self.retraining_managers = {}
        self.global_config = {
            'monitoring_enabled': True,
            'auto_retraining_enabled': False,
            'alert_email': None,
            'dashboard_port': 8080
        }
    
    def register_model(self, model_name: str, model_type: str = 'regression'):
        """Register a model for monitoring"""
        # Create monitors
        performance_monitor = ModelPerformanceMonitor(model_name, model_type)
        drift_detector = DataDriftDetector(model_name)
        retraining_manager = AutoRetrainingManager(model_name)
        
        # Link components
        retraining_manager.set_monitors(performance_monitor, drift_detector)
        
        # Store components
        self.monitors[model_name] = performance_monitor
        self.drift_detectors[model_name] = drift_detector
        self.retraining_managers[model_name] = retraining_manager
        
        logger.info(f"Registered model for monitoring: {model_name}")
    
    def start_monitoring(self, model_name: str):
        """Start monitoring for a specific model"""
        if model_name not in self.retraining_managers:
            raise ValueError(f"Model {model_name} not registered")
        
        if self.global_config['monitoring_enabled']:
            self.retraining_managers[model_name].start_monitoring()
    
    def stop_monitoring(self, model_name: str):
        """Stop monitoring for a specific model"""
        if model_name in self.retraining_managers:
            self.retraining_managers[model_name].stop_monitoring()
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.monitors),
            'monitoring_enabled': self.global_config['monitoring_enabled'],
            'auto_retraining_enabled': self.global_config['auto_retraining_enabled'],
            'models': {}
        }
        
        for model_name in self.monitors:
            model_status = {
                'performance_summary': self.monitors[model_name].get_performance_summary(days=7),
                'monitoring_active': self.retraining_managers[model_name].is_running,
                'last_evaluation': 'N/A'  # Would track last evaluation time
            }
            status['models'][model_name] = model_status
        
        return status
    
    def simulate_monitoring_demo(self):
        """Simulate monitoring for demonstration"""
        print("🔍 MLOps Monitoring System Demo")
        print("=" * 50)
        
        # Register demo models
        models = [
            ('pricing_model', 'regression'),
            ('fraud_detector', 'classification'),
            ('demand_forecaster', 'regression')
        ]
        
        for model_name, model_type in models:
            self.register_model(model_name, model_type)
            print(f"✅ Registered {model_name} ({model_type})")
        
        # Simulate performance logging
        print(f"\n📊 Simulating Performance Monitoring:")
        
        for model_name, model_type in models:
            monitor = self.monitors[model_name]
            
            # Generate synthetic performance data
            if model_type == 'regression':
                # Simulate declining performance
                for i in range(5):
                    y_true = np.random.normal(100, 20, 100)
                    noise_factor = 1 + i * 0.1  # Increasing noise
                    y_pred = y_true + np.random.normal(0, 10 * noise_factor, 100)
                    
                    monitor.log_performance(y_true, y_pred, prediction_count=100, 
                                          notes=f"Simulation batch {i+1}")
            else:  # classification
                # Simulate declining accuracy
                for i in range(5):
                    y_true = np.random.binomial(1, 0.5, 100)
                    accuracy = 0.9 - i * 0.05  # Declining accuracy
                    y_pred = np.where(np.random.random(100) < accuracy, y_true, 1 - y_true)
                    
                    monitor.log_performance(y_true, y_pred, prediction_count=100,
                                          notes=f"Simulation batch {i+1}")
            
            print(f"   📈 {model_name}: Logged 5 performance evaluations")
        
        # Simulate drift detection
        print(f"\n🚨 Simulating Data Drift Detection:")
        
        for model_name, _ in models:
            drift_detector = self.drift_detectors[model_name]
            
            # Set reference data
            X_reference = np.random.normal(0, 1, (1000, 5))
            drift_detector.set_reference_data(X_reference, [f'feature_{i}' for i in range(5)])
            
            # Simulate drifted data
            X_new = np.random.normal(0.5, 1.2, (500, 5))  # Mean shift and scale change
            drift_results = drift_detector.detect_drift(X_new)
            
            drift_count = sum([result['is_drift'] for result in drift_results['feature_drift'].values()])
            print(f"   🔍 {model_name}: {drift_count}/5 features show drift")
        
        # Get system status
        print(f"\n📋 System Status Summary:")
        status = self.get_system_status()
        
        print(f"   Total Models: {status['total_models']}")
        print(f"   Monitoring Enabled: {status['monitoring_enabled']}")
        
        for model_name, model_status in status['models'].items():
            perf_summary = model_status['performance_summary']
            if 'error' not in perf_summary:
                print(f"   📊 {model_name}:")
                print(f"      - Evaluations: {perf_summary['total_evaluations']}")
                print(f"      - Alerts: {perf_summary['alert_summary']['total_alerts']}")
                if perf_summary['latest_metrics']:
                    for metric, value in list(perf_summary['latest_metrics'].items())[:2]:
                        print(f"      - {metric}: {value:.3f}")
        
        print(f"\n🎯 MLOps Demo Complete!")
        print(f"💾 Monitoring data saved to: /Users/dhruvdabhi/temp/SmartMarketer/ml-backend/monitoring/")


def create_mlops_system():
    """Create and demonstrate the MLOps system"""
    print("🔧 Building Advanced MLOps System...")
    
    # Create MLOps manager
    mlops = MLOpsManager()
    
    # Run demonstration
    mlops.simulate_monitoring_demo()
    
    return mlops


if __name__ == "__main__":
    # Run the MLOps system demo
    mlops_system = create_mlops_system()
    print("\n🚀 Advanced MLOps System is ready for deployment!")
