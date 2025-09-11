"""
Advanced Customer Intelligence System
Customer segmentation, personalized pricing, and behavior analysis
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Advanced analytics
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency, pearsonr
import networkx as nx

# Time series for customer lifecycle
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """Advanced customer segmentation using multiple clustering algorithms"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.segment_profiles = {}
        self.is_fitted = False
    
    def _generate_customer_data(self, n_customers=5000):
        """Generate synthetic customer data for demonstration"""
        np.random.seed(self.random_state)
        
        # Customer demographics
        age = np.random.normal(35, 12, n_customers)
        age = np.clip(age, 18, 80)
        
        # Income based on age with some noise
        income = 25000 + age * 1200 + np.random.normal(0, 15000, n_customers)
        income = np.clip(income, 20000, 200000)
        
        # Transaction behavior
        avg_order_value = np.random.lognormal(4, 0.8, n_customers)
        transaction_frequency = np.random.gamma(2, 2, n_customers)
        days_since_last_purchase = np.random.exponential(30, n_customers)
        
        # Product preferences (categorical)
        preferred_categories = np.random.choice(['Electronics', 'Fashion', 'Home', 'Books', 'Sports'], 
                                              n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Loyalty and engagement
        loyalty_score = np.random.beta(2, 3, n_customers) * 100
        app_engagement = np.random.gamma(1.5, 2, n_customers)
        
        # Customer lifetime value (calculated)
        months_as_customer = np.random.exponential(18, n_customers)
        total_spent = avg_order_value * transaction_frequency * (months_as_customer / 12)
        
        # Churn indicators
        support_tickets = np.random.poisson(2, n_customers)
        returns_rate = np.random.beta(1, 10, n_customers)
        
        # Geographic data
        cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
                                 n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Create customer segments naturally
        # Segment 1: High-value customers (20%)
        high_value_mask = np.random.choice([True, False], n_customers, p=[0.2, 0.8])
        income[high_value_mask] *= 1.5
        avg_order_value[high_value_mask] *= 2
        loyalty_score[high_value_mask] = np.clip(loyalty_score[high_value_mask] + 30, 0, 100)
        
        # Segment 2: Frequent buyers (25%)
        frequent_mask = np.random.choice([True, False], n_customers, p=[0.25, 0.75])
        transaction_frequency[frequent_mask] *= 2
        app_engagement[frequent_mask] *= 1.5
        
        # Segment 3: Price-sensitive (30%)
        price_sensitive_mask = np.random.choice([True, False], n_customers, p=[0.3, 0.7])
        avg_order_value[price_sensitive_mask] *= 0.7
        days_since_last_purchase[price_sensitive_mask] *= 1.3
        
        # Create DataFrame
        customer_data = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': age.astype(int),
            'income': income.round(2),
            'avg_order_value': avg_order_value.round(2),
            'transaction_frequency': transaction_frequency.round(2),
            'days_since_last_purchase': days_since_last_purchase.round(0).astype(int),
            'preferred_category': preferred_categories,
            'loyalty_score': loyalty_score.round(2),
            'app_engagement': app_engagement.round(2),
            'months_as_customer': months_as_customer.round(1),
            'total_spent': total_spent.round(2),
            'support_tickets': support_tickets,
            'returns_rate': returns_rate.round(3),
            'city': cities
        })
        
        # Calculate additional metrics
        customer_data['clv'] = customer_data['total_spent'] / np.maximum(customer_data['months_as_customer'], 1) * 24  # 2-year CLV
        customer_data['recency_score'] = 100 - np.clip(customer_data['days_since_last_purchase'], 0, 100)
        customer_data['frequency_score'] = np.clip(customer_data['transaction_frequency'] * 10, 0, 100)
        customer_data['monetary_score'] = np.clip(customer_data['avg_order_value'] / 10, 0, 100)
        
        # RFM Score
        customer_data['rfm_score'] = (
            customer_data['recency_score'] * 0.3 + 
            customer_data['frequency_score'] * 0.4 + 
            customer_data['monetary_score'] * 0.3
        )
        
        return customer_data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for customer segmentation"""
        df = data.copy()
        
        # Behavioral features
        df['purchase_intensity'] = df['transaction_frequency'] / np.maximum(df['months_as_customer'], 1)
        df['avg_monthly_spend'] = df['total_spent'] / np.maximum(df['months_as_customer'], 1)
        df['engagement_per_transaction'] = df['app_engagement'] / np.maximum(df['transaction_frequency'], 1)
        
        # Risk features
        df['churn_risk'] = (
            (df['days_since_last_purchase'] > 60).astype(int) * 0.4 +
            (df['returns_rate'] > 0.1).astype(int) * 0.3 +
            (df['support_tickets'] > 5).astype(int) * 0.3
        )
        
        # Value tiers
        df['value_tier'] = pd.qcut(df['clv'], q=5, labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'])
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in ['preferred_category', 'city', 'value_tier']:
            df[f'{col}_encoded'] = le.fit_transform(df[col])
        
        return df
    
    def fit(self, data: pd.DataFrame = None, features: List[str] = None):
        """Fit customer segmentation models"""
        print("👥 Training Customer Segmentation Models...")
        
        # Generate data if not provided
        if data is None:
            print("📊 Generating synthetic customer data...")
            data = self._generate_customer_data()
        
        # Engineer features
        data_enhanced = self._engineer_features(data)
        
        # Select features for clustering
        if features is None:
            features = [
                'age', 'income', 'avg_order_value', 'transaction_frequency',
                'loyalty_score', 'app_engagement', 'clv', 'rfm_score',
                'purchase_intensity', 'avg_monthly_spend', 'churn_risk',
                'preferred_category_encoded', 'city_encoded'
            ]
        
        # Ensure all features exist
        features = [f for f in features if f in data_enhanced.columns]
        
        X = data_enhanced[features]
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Store feature names
        self.feature_names = features
        self.raw_data = data_enhanced
        
        # Fit multiple clustering models
        self._fit_clustering_models(X_scaled)
        
        # Analyze segments
        self._analyze_segments(data_enhanced, X_scaled)
        
        self.is_fitted = True
        print("✅ Customer Segmentation Training Complete!")
        
        return self
    
    def _fit_clustering_models(self, X_scaled):
        """Fit various clustering algorithms"""
        print("  🔄 Training clustering models...")
        
        # K-Means with optimal number of clusters
        optimal_k = self._find_optimal_clusters(X_scaled)
        self.models['kmeans'] = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        kmeans_labels = self.models['kmeans'].fit_predict(X_scaled)
        
        print(f"    ✅ K-Means with {optimal_k} clusters (Silhouette: {silhouette_score(X_scaled, kmeans_labels):.3f})")
        
        # DBSCAN for density-based clustering
        self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = self.models['dbscan'].fit_predict(X_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        if n_clusters_dbscan > 1:
            # Filter out noise points for silhouette calculation
            mask = dbscan_labels != -1
            if np.sum(mask) > 1:
                dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
                print(f"    ✅ DBSCAN with {n_clusters_dbscan} clusters (Silhouette: {dbscan_silhouette:.3f})")
            else:
                print(f"    ✅ DBSCAN with {n_clusters_dbscan} clusters")
        else:
            print(f"    ⚠️ DBSCAN found {n_clusters_dbscan} clusters")
        
        # Agglomerative Clustering
        self.models['hierarchical'] = AgglomerativeClustering(n_clusters=optimal_k)
        hierarchical_labels = self.models['hierarchical'].fit_predict(X_scaled)
        
        print(f"    ✅ Hierarchical with {optimal_k} clusters (Silhouette: {silhouette_score(X_scaled, hierarchical_labels):.3f})")
        
        # Store labels
        self.cluster_labels = {
            'kmeans': kmeans_labels,
            'dbscan': dbscan_labels,
            'hierarchical': hierarchical_labels
        }
    
    def _find_optimal_clusters(self, X_scaled, max_k=10):
        """Find optimal number of clusters using silhouette analysis"""
        silhouette_scores = []
        K_range = range(2, min(max_k + 1, len(X_scaled) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        return optimal_k
    
    def _analyze_segments(self, data: pd.DataFrame, X_scaled: np.ndarray):
        """Analyze characteristics of each segment"""
        print("  📊 Analyzing segment characteristics...")
        
        # Use K-Means labels as primary segmentation
        labels = self.cluster_labels['kmeans']
        data_with_segments = data.copy()
        data_with_segments['segment'] = labels
        
        # Calculate segment profiles
        profiles = {}
        for segment in np.unique(labels):
            segment_data = data_with_segments[data_with_segments['segment'] == segment]
            
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(data) * 100,
                'avg_age': segment_data['age'].mean(),
                'avg_income': segment_data['income'].mean(),
                'avg_clv': segment_data['clv'].mean(),
                'avg_loyalty': segment_data['loyalty_score'].mean(),
                'avg_order_value': segment_data['avg_order_value'].mean(),
                'transaction_frequency': segment_data['transaction_frequency'].mean(),
                'churn_risk': segment_data['churn_risk'].mean(),
                'top_category': segment_data['preferred_category'].mode().iloc[0] if len(segment_data) > 0 else 'Unknown',
                'top_city': segment_data['city'].mode().iloc[0] if len(segment_data) > 0 else 'Unknown'
            }
            
            profiles[f'Segment_{segment}'] = profile
        
        self.segment_profiles['kmeans'] = profiles
        
        # Print segment summary
        for segment_name, profile in profiles.items():
            print(f"    🎯 {segment_name}: {profile['size']} customers ({profile['percentage']:.1f}%)")
            print(f"       Avg CLV: ${profile['avg_clv']:.0f}, Loyalty: {profile['avg_loyalty']:.0f}")
    
    def predict_segment(self, customer_data: pd.DataFrame, method: str = 'kmeans') -> np.ndarray:
        """Predict segment for new customers"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Engineer features
        data_enhanced = self._engineer_features(customer_data)
        
        # Select and scale features
        X = data_enhanced[self.feature_names]
        X_scaled = self.scalers['standard'].transform(X)
        
        # Predict using specified method
        if method == 'kmeans':
            return self.models['kmeans'].predict(X_scaled)
        elif method == 'dbscan':
            # DBSCAN doesn't have predict method, use fit_predict
            return self.models['dbscan'].fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_segment_profiles(self, method: str = 'kmeans') -> Dict:
        """Get detailed segment profiles"""
        if method not in self.segment_profiles:
            raise ValueError(f"Profiles not available for method: {method}")
        
        return self.segment_profiles[method]
    
    def visualize_segments(self, method: str = 'kmeans', technique: str = 'pca', 
                          figsize: Tuple[int, int] = (12, 8)):
        """Visualize customer segments"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get scaled data
        X_scaled = self.scalers['standard'].transform(self.raw_data[self.feature_names])
        labels = self.cluster_labels[method]
        
        # Dimensionality reduction
        if technique == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
            X_reduced = reducer.fit_transform(X_scaled)
            title_suffix = f"(PCA - {reducer.explained_variance_ratio_.sum():.2%} variance explained)"
        elif technique == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.random_state)
            X_reduced = reducer.fit_transform(X_scaled)
            title_suffix = "(t-SNE)"
        else:
            raise ValueError("Technique must be 'pca' or 'tsne'")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Segment scatter plot
        scatter = axes[0, 0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[0, 0].set_title(f'Customer Segments - {method.upper()} {title_suffix}')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Segment size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        axes[0, 1].pie(counts, labels=[f'Segment {i}' for i in unique_labels], autopct='%1.1f%%')
        axes[0, 1].set_title('Segment Size Distribution')
        
        # CLV by segment
        clv_by_segment = [self.raw_data[labels == label]['clv'].mean() for label in unique_labels]
        axes[1, 0].bar(range(len(unique_labels)), clv_by_segment, color=plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
        axes[1, 0].set_title('Average CLV by Segment')
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Average CLV ($)')
        axes[1, 0].set_xticks(range(len(unique_labels)))
        axes[1, 0].set_xticklabels([f'Segment {i}' for i in unique_labels])
        
        # Feature importance heatmap
        segment_means = []
        for label in unique_labels:
            segment_data = self.raw_data[labels == label][self.feature_names]
            segment_means.append(segment_data.mean().values)
        
        segment_means = np.array(segment_means)
        im = axes[1, 1].imshow(segment_means.T, cmap='RdYlBu', aspect='auto')
        axes[1, 1].set_title('Feature Profiles by Segment')
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('Features')
        axes[1, 1].set_xticks(range(len(unique_labels)))
        axes[1, 1].set_xticklabels([f'S{i}' for i in unique_labels])
        axes[1, 1].set_yticks(range(len(self.feature_names)))
        axes[1, 1].set_yticklabels([name[:10] + '...' if len(name) > 10 else name for name in self.feature_names], fontsize=8)
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig


class PersonalizedPricing:
    """Personalized pricing based on customer segments and behavior"""
    
    def __init__(self, base_pricing_model, customer_segmentation):
        self.base_model = base_pricing_model
        self.segmentation = customer_segmentation
        self.pricing_strategies = {}
        self.elasticity_models = {}
        self.is_fitted = False
    
    def fit(self, transaction_data: pd.DataFrame):
        """Fit personalized pricing models"""
        print("💰 Training Personalized Pricing Models...")
        
        # Get customer segments
        if hasattr(self.segmentation, 'raw_data'):
            customer_data = self.segmentation.raw_data
            segments = self.segmentation.cluster_labels['kmeans']
            
            # Define pricing strategies for each segment
            self._define_pricing_strategies(customer_data, segments)
            
            # Train price elasticity models
            self._train_elasticity_models(transaction_data, customer_data, segments)
            
            self.is_fitted = True
            print("✅ Personalized Pricing Training Complete!")
        else:
            raise ValueError("Customer segmentation model must be fitted first")
    
    def _define_pricing_strategies(self, customer_data: pd.DataFrame, segments: np.ndarray):
        """Define pricing strategies for each customer segment"""
        unique_segments = np.unique(segments)
        
        for segment in unique_segments:
            segment_data = customer_data[segments == segment]
            
            # Calculate segment characteristics
            avg_clv = segment_data['clv'].mean()
            avg_loyalty = segment_data['loyalty_score'].mean()
            price_sensitivity = 1 / (segment_data['avg_order_value'].mean() / 100)  # Inverse relationship
            
            # Define pricing strategy
            if avg_clv > customer_data['clv'].quantile(0.8):
                # High-value customers: premium pricing acceptable
                strategy = {
                    'base_multiplier': 1.1,
                    'loyalty_discount': 0.05,
                    'volume_discount': 0.02,
                    'price_sensitivity': 0.3,
                    'strategy_name': 'Premium'
                }
            elif avg_loyalty > customer_data['loyalty_score'].quantile(0.7):
                # Loyal customers: moderate pricing with rewards
                strategy = {
                    'base_multiplier': 1.0,
                    'loyalty_discount': 0.08,
                    'volume_discount': 0.03,
                    'price_sensitivity': 0.5,
                    'strategy_name': 'Loyalty Rewards'
                }
            elif price_sensitivity > 1.5:
                # Price-sensitive customers: competitive pricing
                strategy = {
                    'base_multiplier': 0.95,
                    'loyalty_discount': 0.03,
                    'volume_discount': 0.05,
                    'price_sensitivity': 0.8,
                    'strategy_name': 'Value'
                }
            else:
                # Standard customers: standard pricing
                strategy = {
                    'base_multiplier': 1.0,
                    'loyalty_discount': 0.05,
                    'volume_discount': 0.02,
                    'price_sensitivity': 0.6,
                    'strategy_name': 'Standard'
                }
            
            self.pricing_strategies[segment] = strategy
            print(f"    🎯 Segment {segment}: {strategy['strategy_name']} Strategy")
    
    def _train_elasticity_models(self, transaction_data: pd.DataFrame, 
                                customer_data: pd.DataFrame, segments: np.ndarray):
        """Train price elasticity models for each segment"""
        # For demonstration, create synthetic elasticity models
        # In practice, you'd use historical price/demand data
        
        for segment in np.unique(segments):
            # Simple elasticity model based on segment characteristics
            segment_customers = customer_data[segments == segment]
            
            # Price elasticity tends to be higher for price-sensitive segments
            base_elasticity = -0.5  # Base elasticity
            sensitivity_factor = self.pricing_strategies[segment]['price_sensitivity']
            
            # Adjust elasticity based on segment characteristics
            elasticity = base_elasticity * sensitivity_factor
            
            self.elasticity_models[segment] = {
                'elasticity': elasticity,
                'base_price_acceptance': segment_customers['avg_order_value'].mean()
            }
    
    def calculate_personalized_price(self, base_price: float, customer_id: int, 
                                   customer_features: Dict, context: Dict = None) -> Dict:
        """Calculate personalized price for a specific customer"""
        if not self.is_fitted:
            raise ValueError("Personalized pricing model must be fitted first")
        
        # Convert customer features to DataFrame
        customer_df = pd.DataFrame([customer_features])
        
        # Get customer segment
        segment = self.segmentation.predict_segment(customer_df)[0]
        strategy = self.pricing_strategies[segment]
        
        # Start with base price
        personalized_price = base_price * strategy['base_multiplier']
        
        # Apply loyalty discount
        if customer_features.get('loyalty_score', 0) > 70:
            personalized_price *= (1 - strategy['loyalty_discount'])
        
        # Apply volume discount for frequent buyers
        if customer_features.get('transaction_frequency', 0) > 10:
            personalized_price *= (1 - strategy['volume_discount'])
        
        # Context-based adjustments
        if context:
            # Time-based pricing
            if context.get('is_peak_time', False):
                personalized_price *= 1.05
            
            # Inventory-based pricing
            if context.get('inventory_level', 1.0) < 0.2:
                personalized_price *= 1.1  # Low inventory = higher price
            
            # Competition-based pricing
            competitor_price = context.get('competitor_price')
            if competitor_price:
                price_difference = (personalized_price - competitor_price) / competitor_price
                if price_difference > 0.15:  # If more than 15% higher than competitor
                    personalized_price = competitor_price * 1.1  # Stay competitive
        
        # Calculate expected demand using elasticity
        elasticity_info = self.elasticity_models[segment]
        price_change = (personalized_price - base_price) / base_price
        demand_change = elasticity_info['elasticity'] * price_change
        expected_demand_multiplier = 1 + demand_change
        
        # Estimate revenue impact
        revenue_impact = personalized_price * expected_demand_multiplier
        base_revenue = base_price * 1.0  # Baseline demand = 1
        revenue_lift = (revenue_impact - base_revenue) / base_revenue
        
        return {
            'base_price': base_price,
            'personalized_price': round(personalized_price, 2),
            'price_adjustment': round((personalized_price - base_price) / base_price * 100, 2),
            'customer_segment': int(segment),
            'pricing_strategy': strategy['strategy_name'],
            'expected_demand_change': round(demand_change * 100, 2),
            'revenue_lift': round(revenue_lift * 100, 2),
            'applied_discounts': {
                'loyalty': strategy['loyalty_discount'] if customer_features.get('loyalty_score', 0) > 70 else 0,
                'volume': strategy['volume_discount'] if customer_features.get('transaction_frequency', 0) > 10 else 0
            }
        }
    
    def optimize_segment_pricing(self, segment: int, base_price: float, 
                               demand_data: Dict = None) -> Dict:
        """Optimize pricing for an entire customer segment"""
        if segment not in self.pricing_strategies:
            raise ValueError(f"Unknown segment: {segment}")
        
        strategy = self.pricing_strategies[segment]
        elasticity_info = self.elasticity_models[segment]
        
        # Test different price points
        price_multipliers = np.arange(0.8, 1.3, 0.05)
        results = []
        
        for multiplier in price_multipliers:
            test_price = base_price * multiplier
            price_change = (test_price - base_price) / base_price
            demand_change = elasticity_info['elasticity'] * price_change
            expected_demand = 1 + demand_change  # Baseline demand = 1
            
            revenue = test_price * max(expected_demand, 0.1)  # Minimum demand of 0.1
            
            results.append({
                'price_multiplier': multiplier,
                'price': test_price,
                'demand_multiplier': expected_demand,
                'revenue': revenue
            })
        
        # Find optimal price
        optimal_result = max(results, key=lambda x: x['revenue'])
        
        return {
            'segment': segment,
            'strategy_name': strategy['strategy_name'],
            'optimal_price_multiplier': optimal_result['price_multiplier'],
            'optimal_price': optimal_result['price'],
            'optimal_revenue': optimal_result['revenue'],
            'base_price': base_price,
            'revenue_improvement': (optimal_result['revenue'] / base_price - 1) * 100
        }
    
    def analyze_pricing_impact(self, customer_sample: pd.DataFrame, 
                             base_prices: List[float]) -> pd.DataFrame:
        """Analyze pricing impact across customer segments"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        results = []
        
        for idx, (_, customer) in enumerate(customer_sample.iterrows()):
            if idx < len(base_prices):
                base_price = base_prices[idx]
                customer_dict = customer.to_dict()
                
                pricing_result = self.calculate_personalized_price(
                    base_price=base_price,
                    customer_id=customer.get('customer_id', idx),
                    customer_features=customer_dict
                )
                
                results.append({
                    'customer_id': customer.get('customer_id', idx),
                    'segment': pricing_result['customer_segment'],
                    'strategy': pricing_result['pricing_strategy'],
                    'base_price': pricing_result['base_price'],
                    'personalized_price': pricing_result['personalized_price'],
                    'price_adjustment_pct': pricing_result['price_adjustment'],
                    'revenue_lift_pct': pricing_result['revenue_lift']
                })
        
        return pd.DataFrame(results)


class ChurnPrediction:
    """Customer churn prediction model"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.is_fitted = False
    
    def _create_churn_labels(self, customer_data: pd.DataFrame) -> np.ndarray:
        """Create churn labels based on customer behavior"""
        # Define churn based on multiple factors
        churn_conditions = (
            (customer_data['days_since_last_purchase'] > 90) |
            (customer_data['returns_rate'] > 0.2) |
            (customer_data['support_tickets'] > 10) |
            ((customer_data['transaction_frequency'] < 1) & (customer_data['months_as_customer'] > 6))
        )
        
        # Add some randomness to make it more realistic
        random_churn = np.random.binomial(1, 0.05, len(customer_data))  # 5% random churn
        
        churn_labels = (churn_conditions | random_churn.astype(bool)).astype(int)
        
        return churn_labels
    
    def fit(self, customer_data: pd.DataFrame, churn_labels: np.ndarray = None):
        """Fit churn prediction model"""
        print("⚠️ Training Churn Prediction Model...")
        
        # Create churn labels if not provided
        if churn_labels is None:
            churn_labels = self._create_churn_labels(customer_data)
        
        # Select features for churn prediction
        churn_features = [
            'age', 'income', 'avg_order_value', 'transaction_frequency',
            'days_since_last_purchase', 'loyalty_score', 'app_engagement',
            'months_as_customer', 'total_spent', 'support_tickets',
            'returns_rate', 'rfm_score', 'churn_risk'
        ]
        
        # Ensure features exist
        churn_features = [f for f in churn_features if f in customer_data.columns]
        
        X = customer_data[churn_features]
        y = churn_labels
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Feature importance
        self.feature_importance = dict(zip(churn_features, self.model.feature_importances_))
        
        print(f"    ✅ Churn Model - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
        print(f"    📊 Churn Rate: {np.mean(y):.2%}")
        
        self.feature_names = churn_features
        self.is_fitted = True
        
        return self
    
    def predict_churn_probability(self, customer_data: pd.DataFrame) -> np.ndarray:
        """Predict churn probability for customers"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = customer_data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_churn_insights(self, customer_data: pd.DataFrame) -> Dict:
        """Get insights about churn risk factors"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get churn probabilities
        churn_probs = self.predict_churn_probability(customer_data)
        
        # Analyze high-risk customers
        high_risk_threshold = 0.7
        high_risk_customers = customer_data[churn_probs > high_risk_threshold]
        
        insights = {
            'total_customers': len(customer_data),
            'high_risk_customers': len(high_risk_customers),
            'high_risk_percentage': len(high_risk_customers) / len(customer_data) * 100,
            'avg_churn_probability': np.mean(churn_probs),
            'top_risk_factors': dict(sorted(self.feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]),
            'high_risk_characteristics': {
                'avg_days_since_purchase': high_risk_customers['days_since_last_purchase'].mean() if len(high_risk_customers) > 0 else 0,
                'avg_loyalty_score': high_risk_customers['loyalty_score'].mean() if len(high_risk_customers) > 0 else 0,
                'avg_support_tickets': high_risk_customers['support_tickets'].mean() if len(high_risk_customers) > 0 else 0
            }
        }
        
        return insights


def create_customer_intelligence_system():
    """Create and train the complete customer intelligence system"""
    print("🧠 Building Advanced Customer Intelligence System...")
    
    # Initialize components
    segmentation = CustomerSegmentation(random_state=42)
    
    # Train customer segmentation
    segmentation.fit()
    
    # Get sample customer data for other models
    customer_data = segmentation.raw_data
    
    # Train churn prediction
    churn_model = ChurnPrediction(random_state=42)
    churn_model.fit(customer_data)
    
    # Create personalized pricing (assuming we have a base pricing model)
    class MockBasePricingModel:
        def predict(self, X):
            # Mock base pricing model
            return np.random.uniform(50, 500, len(X))
    
    base_pricing_model = MockBasePricingModel()
    personalized_pricing = PersonalizedPricing(base_pricing_model, segmentation)
    
    # Create mock transaction data for pricing model
    mock_transaction_data = pd.DataFrame({
        'customer_id': range(1, 101),
        'price': np.random.uniform(50, 500, 100),
        'quantity': np.random.poisson(2, 100)
    })
    
    personalized_pricing.fit(mock_transaction_data)
    
    # Demonstrate the system
    print(f"\n🎯 Customer Intelligence System Summary:")
    
    # Segmentation results
    print(f"\n👥 Customer Segmentation:")
    profiles = segmentation.get_segment_profiles()
    for segment, profile in profiles.items():
        print(f"   {segment}: {profile['size']} customers ({profile['percentage']:.1f}%)")
        print(f"      Strategy: {profile['top_category']} focus, Avg CLV: ${profile['avg_clv']:.0f}")
    
    # Churn insights
    churn_insights = churn_model.get_churn_insights(customer_data)
    print(f"\n⚠️ Churn Analysis:")
    print(f"   High-risk customers: {churn_insights['high_risk_customers']} ({churn_insights['high_risk_percentage']:.1f}%)")
    print(f"   Top risk factors: {list(churn_insights['top_risk_factors'].keys())[:3]}")
    
    # Personalized pricing example
    print(f"\n💰 Personalized Pricing Example:")
    sample_customer = {
        'age': 35,
        'income': 75000,
        'avg_order_value': 150,
        'transaction_frequency': 8,
        'loyalty_score': 75,
        'app_engagement': 5.2,
        'days_since_last_purchase': 10
    }
    
    pricing_result = personalized_pricing.calculate_personalized_price(
        base_price=100,
        customer_id=1,
        customer_features=sample_customer
    )
    
    print(f"   Base Price: ${pricing_result['base_price']:.2f}")
    print(f"   Personalized Price: ${pricing_result['personalized_price']:.2f}")
    print(f"   Strategy: {pricing_result['pricing_strategy']}")
    print(f"   Expected Revenue Lift: {pricing_result['revenue_lift']:.1f}%")
    
    # Save models
    models_path = '/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/advanced_models/'
    
    # Save segmentation model
    segmentation_data = {
        'models': segmentation.models,
        'scalers': segmentation.scalers,
        'feature_names': segmentation.feature_names,
        'segment_profiles': segmentation.segment_profiles,
        'cluster_labels': segmentation.cluster_labels,
        'is_fitted': segmentation.is_fitted
    }
    joblib.dump(segmentation_data, f"{models_path}customer_segmentation.joblib")
    
    # Save churn model
    churn_data = {
        'model': churn_model.model,
        'scaler': churn_model.scaler,
        'feature_names': churn_model.feature_names,
        'feature_importance': churn_model.feature_importance,
        'is_fitted': churn_model.is_fitted
    }
    joblib.dump(churn_data, f"{models_path}churn_prediction.joblib")
    
    print(f"\n💾 Models saved to {models_path}")
    
    return {
        'segmentation': segmentation,
        'churn_model': churn_model,
        'personalized_pricing': personalized_pricing,
        'customer_data': customer_data
    }


if __name__ == "__main__":
    # Run the complete customer intelligence system
    system = create_customer_intelligence_system()
    print("\n🚀 Advanced Customer Intelligence System is ready for deployment!")
