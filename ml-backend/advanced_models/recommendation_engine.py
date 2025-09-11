"""
Advanced Recommendation Engine
Collaborative filtering, content-based filtering, and hybrid recommendation system
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
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Matrix factorization
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import pdist, squareform

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Network analysis
import networkx as nx

warnings.filterwarnings('ignore')

class RecommendationEngine:
    """Advanced recommendation system with multiple algorithms"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.is_fitted = False
        
    def _generate_recommendation_data(self, n_users=1000, n_items=500, n_interactions=10000):
        """Generate synthetic recommendation data"""
        np.random.seed(self.random_state)
        
        # User data
        users = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'age': np.random.normal(35, 12, n_users).astype(int),
            'gender': np.random.choice(['M', 'F'], n_users),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_users, p=[0.5, 0.3, 0.2]),
            'income_level': np.random.choice(['Low', 'Medium', 'High'], n_users, p=[0.3, 0.5, 0.2]),
            'registration_date': pd.date_range('2020-01-01', periods=n_users, freq='D')[:n_users]
        })
        
        # Item data (products)
        categories = ['Electronics', 'Fashion', 'Home', 'Books', 'Sports', 'Beauty', 'Automotive']
        items = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'category': np.random.choice(categories, n_items),
            'price': np.random.lognormal(4, 1, n_items),
            'brand': np.random.choice([f'Brand_{i}' for i in range(1, 51)], n_items),
            'rating': np.random.normal(4.0, 0.8, n_items),
            'popularity_score': np.random.exponential(2, n_items),
            'release_date': pd.date_range('2020-01-01', periods=n_items, freq='W')[:n_items]
        })
        
        # Create realistic item descriptions
        item_descriptions = []
        for _, item in items.iterrows():
            desc_parts = [
                item['category'].lower(),
                item['brand'].lower(),
                f"price_{int(item['price']//100)*100}",
                f"rating_{int(item['rating'])}"
            ]
            item_descriptions.append(' '.join(desc_parts))
        
        items['description'] = item_descriptions
        
        # Generate interactions with realistic patterns
        interactions = []
        
        for _ in range(n_interactions):
            # Select user (some users more active)
            user_weights = np.random.exponential(1, n_users)
            user_weights = user_weights / user_weights.sum()
            user_id = np.random.choice(users['user_id'], p=user_weights)
            
            # Select item based on popularity and user preferences
            user_data = users[users['user_id'] == user_id].iloc[0]
            
            # Create preference bias based on user demographics
            item_weights = items['popularity_score'].copy()
            
            # Age-based preferences
            if user_data['age'] < 30:
                item_weights[items['category'].isin(['Electronics', 'Fashion'])] *= 2
            elif user_data['age'] > 50:
                item_weights[items['category'].isin(['Home', 'Books'])] *= 2
            
            # Gender-based preferences
            if user_data['gender'] == 'F':
                item_weights[items['category'].isin(['Fashion', 'Beauty'])] *= 1.5
            else:
                item_weights[items['category'].isin(['Electronics', 'Sports', 'Automotive'])] *= 1.5
            
            # Income-based preferences
            if user_data['income_level'] == 'High':
                item_weights[items['price'] > items['price'].quantile(0.7)] *= 1.3
            elif user_data['income_level'] == 'Low':
                item_weights[items['price'] < items['price'].quantile(0.4)] *= 1.3
            
            item_weights = item_weights / item_weights.sum()
            item_id = np.random.choice(items['item_id'], p=item_weights)
            
            # Generate rating (influenced by item's average rating)
            item_avg_rating = items[items['item_id'] == item_id]['rating'].iloc[0]
            rating = np.random.normal(item_avg_rating, 0.5)
            rating = np.clip(rating, 1, 5)
            
            # Generate interaction type
            interaction_type = np.random.choice(['view', 'purchase', 'cart_add', 'favorite'], 
                                              p=[0.6, 0.2, 0.15, 0.05])
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'interaction_type': interaction_type,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Remove duplicates (keep latest)
        interactions_df = interactions_df.sort_values('timestamp').drop_duplicates(['user_id', 'item_id'], keep='last')
        
        return users, items, interactions_df
    
    def fit(self, users: pd.DataFrame = None, items: pd.DataFrame = None, 
            interactions: pd.DataFrame = None):
        """Fit the recommendation models"""
        print("🎯 Training Advanced Recommendation Engine...")
        
        # Generate data if not provided
        if users is None or items is None or interactions is None:
            print("📊 Generating synthetic recommendation data...")
            users, items, interactions = self._generate_recommendation_data()
        
        self.users_df = users
        self.items_df = items
        self.interactions_df = interactions
        
        # Prepare data
        self._prepare_data()
        
        # Train collaborative filtering models
        self._train_collaborative_filtering()
        
        # Train content-based filtering
        self._train_content_based_filtering()
        
        # Train neural collaborative filtering
        self._train_neural_collaborative_filtering()
        
        # Train hybrid model
        self._train_hybrid_model()
        
        self.is_fitted = True
        print("✅ Recommendation Engine Training Complete!")
        
        return self
    
    def _prepare_data(self):
        """Prepare data for different recommendation algorithms"""
        print("  📋 Preparing recommendation data...")
        
        # Create user-item interaction matrix
        user_item_pivot = self.interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # Convert to sparse matrix for memory efficiency
        self.user_item_matrix = csr_matrix(user_item_pivot.values)
        self.user_ids = user_item_pivot.index.tolist()
        self.item_ids = user_item_pivot.columns.tolist()
        
        # Create user-item mapping
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Prepare item features for content-based filtering
        self._prepare_item_features()
        
        # Prepare user features
        self._prepare_user_features()
    
    def _prepare_item_features(self):
        """Prepare item features for content-based filtering"""
        # Encode categorical features
        le_category = LabelEncoder()
        le_brand = LabelEncoder()
        
        item_features = self.items_df.copy()
        item_features['category_encoded'] = le_category.fit_transform(item_features['category'])
        item_features['brand_encoded'] = le_brand.fit_transform(item_features['brand'])
        
        # Create TF-IDF features from descriptions
        self.vectorizers['tfidf'] = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_features = self.vectorizers['tfidf'].fit_transform(item_features['description'])
        
        # Combine numerical and categorical features
        numerical_features = ['price', 'rating', 'popularity_score', 'category_encoded', 'brand_encoded']
        
        # Scale numerical features
        self.scalers['item_features'] = StandardScaler()
        scaled_numerical = self.scalers['item_features'].fit_transform(item_features[numerical_features])
        
        # Combine features
        self.item_features = np.hstack([scaled_numerical, tfidf_features.toarray()])
        
        # Store encoders for later use
        self.label_encoders = {
            'category': le_category,
            'brand': le_brand
        }
    
    def _prepare_user_features(self):
        """Prepare user features"""
        le_gender = LabelEncoder()
        le_location = LabelEncoder()
        le_income = LabelEncoder()
        
        user_features = self.users_df.copy()
        user_features['gender_encoded'] = le_gender.fit_transform(user_features['gender'])
        user_features['location_encoded'] = le_location.fit_transform(user_features['location'])
        user_features['income_encoded'] = le_income.fit_transform(user_features['income_level'])
        
        # Calculate days since registration
        user_features['days_since_registration'] = (
            datetime.now() - pd.to_datetime(user_features['registration_date'])
        ).dt.days
        
        # Select features
        feature_cols = ['age', 'gender_encoded', 'location_encoded', 'income_encoded', 'days_since_registration']
        
        # Scale features
        self.scalers['user_features'] = StandardScaler()
        self.user_features = self.scalers['user_features'].fit_transform(user_features[feature_cols])
        
        # Store encoders
        self.user_label_encoders = {
            'gender': le_gender,
            'location': le_location,
            'income': le_income
        }
    
    def _train_collaborative_filtering(self):
        """Train collaborative filtering models"""
        print("  🤝 Training collaborative filtering models...")
        
        # Matrix Factorization using SVD
        self.models['svd'] = TruncatedSVD(n_components=50, random_state=self.random_state)
        self.user_factors = self.models['svd'].fit_transform(self.user_item_matrix)
        self.item_factors = self.models['svd'].components_.T
        
        # Non-negative Matrix Factorization
        self.models['nmf'] = NMF(n_components=50, random_state=self.random_state, max_iter=200)
        self.user_factors_nmf = self.models['nmf'].fit_transform(self.user_item_matrix)
        self.item_factors_nmf = self.models['nmf'].components_.T
        
        # User-based collaborative filtering
        user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_matrix = user_similarity
        
        # Item-based collaborative filtering
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_matrix = item_similarity
        
        print("    ✅ Collaborative filtering models trained")
    
    def _train_content_based_filtering(self):
        """Train content-based filtering models"""
        print("  📝 Training content-based filtering...")
        
        # Item similarity based on features
        self.item_content_similarity = cosine_similarity(self.item_features)
        
        # K-Means clustering for item categories
        self.models['item_clusters'] = KMeans(n_clusters=10, random_state=self.random_state)
        self.item_cluster_labels = self.models['item_clusters'].fit_predict(self.item_features)
        
        # Nearest neighbors for content-based recommendations
        self.models['content_knn'] = NearestNeighbors(n_neighbors=20, metric='cosine')
        self.models['content_knn'].fit(self.item_features)
        
        print("    ✅ Content-based filtering trained")
    
    def _train_neural_collaborative_filtering(self):
        """Train neural collaborative filtering model"""
        print("  🧠 Training neural collaborative filtering...")
        
        try:
            # Prepare data for neural network
            interactions = self.interactions_df.copy()
            
            # Map user and item IDs to indices
            interactions['user_idx'] = interactions['user_id'].map(self.user_to_idx)
            interactions['item_idx'] = interactions['item_id'].map(self.item_to_idx)
            
            # Remove interactions with unmapped IDs
            interactions = interactions.dropna(subset=['user_idx', 'item_idx'])
            
            if len(interactions) == 0:
                print("    ❌ No valid interactions for neural model")
                return
            
            # Split data
            train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=self.random_state)
            
            # Build neural collaborative filtering model
            n_users = len(self.user_ids)
            n_items = len(self.item_ids)
            embedding_size = 50
            
            # User and item inputs
            user_input = Input(shape=(), name='user_id')
            item_input = Input(shape=(), name='item_id')
            
            # Embeddings
            user_embedding = Embedding(n_users, embedding_size, name='user_embedding')(user_input)
            item_embedding = Embedding(n_items, embedding_size, name='item_embedding')(item_input)
            
            # Flatten embeddings
            user_vec = Flatten()(user_embedding)
            item_vec = Flatten()(item_embedding)
            
            # Neural MF path
            concat = Concatenate()([user_vec, item_vec])
            dense1 = Dense(128, activation='relu')(concat)
            dropout1 = Dropout(0.2)(dense1)
            dense2 = Dense(64, activation='relu')(dropout1)
            dropout2 = Dropout(0.2)(dense2)
            
            # GMF path (Generalized Matrix Factorization)
            gmf = Multiply()([user_vec, item_vec])
            
            # Combine paths
            combined = Concatenate()([dense2, gmf])
            output = Dense(1, activation='linear', name='rating')(combined)
            
            # Create and compile model
            model = Model(inputs=[user_input, item_input], outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
            
            history = model.fit(
                [train_data['user_idx'].values, train_data['item_idx'].values],
                train_data['rating'].values,
                validation_data=(
                    [test_data['user_idx'].values, test_data['item_idx'].values],
                    test_data['rating'].values
                ),
                epochs=50,
                batch_size=256,
                callbacks=callbacks,
                verbose=0
            )
            
            self.models['neural_cf'] = model
            print("    ✅ Neural collaborative filtering trained")
            
        except Exception as e:
            print(f"    ❌ Neural collaborative filtering failed: {e}")
    
    def _train_hybrid_model(self):
        """Train hybrid recommendation model"""
        print("  🔀 Training hybrid model...")
        
        # Simple weighted ensemble approach
        self.hybrid_weights = {
            'collaborative': 0.4,
            'content': 0.3,
            'neural': 0.2,
            'popularity': 0.1
        }
        
        # Calculate item popularity scores
        item_popularity = self.interactions_df.groupby('item_id').size().reset_index(name='popularity')
        self.item_popularity = dict(zip(item_popularity['item_id'], item_popularity['popularity']))
        
        print("    ✅ Hybrid model configured")
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10, 
                       method: str = 'hybrid') -> List[Dict]:
        """Generate recommendations for a user"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.user_to_idx:
            # Handle cold start - recommend popular items
            return self._get_popular_recommendations(n_recommendations)
        
        user_idx = self.user_to_idx[user_id]
        
        if method == 'collaborative':
            return self._collaborative_recommendations(user_idx, n_recommendations)
        elif method == 'content':
            return self._content_based_recommendations(user_id, n_recommendations)
        elif method == 'neural':
            return self._neural_recommendations(user_idx, n_recommendations)
        elif method == 'hybrid':
            return self._hybrid_recommendations(user_idx, n_recommendations)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _collaborative_recommendations(self, user_idx: int, n_recommendations: int) -> List[Dict]:
        """Generate collaborative filtering recommendations"""
        # SVD-based recommendations
        user_profile = self.user_factors[user_idx]
        scores = np.dot(self.item_factors, user_profile)
        
        # Get user's already rated items
        user_rated_items = set(self.user_item_matrix[user_idx].nonzero()[1])
        
        # Filter out already rated items
        recommendations = []
        for item_idx in np.argsort(scores)[::-1]:
            if item_idx not in user_rated_items and len(recommendations) < n_recommendations:
                item_id = self.idx_to_item[item_idx]
                item_info = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
                
                recommendations.append({
                    'item_id': item_id,
                    'score': float(scores[item_idx]),
                    'method': 'collaborative_svd',
                    'item_name': f"{item_info['category']} by {item_info['brand']}",
                    'price': float(item_info['price']),
                    'category': item_info['category']
                })
        
        return recommendations
    
    def _content_based_recommendations(self, user_id: int, n_recommendations: int) -> List[Dict]:
        """Generate content-based recommendations"""
        # Get user's interaction history
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return self._get_popular_recommendations(n_recommendations)
        
        # Get user's preferred items
        preferred_items = user_interactions[user_interactions['rating'] >= 4]['item_id'].tolist()
        
        if not preferred_items:
            preferred_items = user_interactions['item_id'].tolist()
        
        # Calculate average feature vector for preferred items
        preferred_item_indices = [self.item_to_idx.get(item) for item in preferred_items if item in self.item_to_idx]
        preferred_item_indices = [idx for idx in preferred_item_indices if idx is not None]
        
        if not preferred_item_indices:
            return self._get_popular_recommendations(n_recommendations)
        
        user_profile = np.mean(self.item_features[preferred_item_indices], axis=0)
        
        # Calculate similarity to all items
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get recommendations
        recommendations = []
        user_rated_items = set(user_interactions['item_id'])
        
        for item_idx in np.argsort(similarities)[::-1]:
            item_id = self.idx_to_item[item_idx]
            if item_id not in user_rated_items and len(recommendations) < n_recommendations:
                item_info = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
                
                recommendations.append({
                    'item_id': item_id,
                    'score': float(similarities[item_idx]),
                    'method': 'content_based',
                    'item_name': f"{item_info['category']} by {item_info['brand']}",
                    'price': float(item_info['price']),
                    'category': item_info['category']
                })
        
        return recommendations
    
    def _neural_recommendations(self, user_idx: int, n_recommendations: int) -> List[Dict]:
        """Generate neural collaborative filtering recommendations"""
        if 'neural_cf' not in self.models:
            return self._collaborative_recommendations(user_idx, n_recommendations)
        
        model = self.models['neural_cf']
        
        # Get all items for this user
        all_items = np.arange(len(self.item_ids))
        user_array = np.full(len(all_items), user_idx)
        
        # Predict ratings
        predictions = model.predict([user_array, all_items], verbose=0).flatten()
        
        # Get user's already rated items
        user_id = self.idx_to_user[user_idx]
        user_rated_items = set(self.interactions_df[self.interactions_df['user_id'] == user_id]['item_id'])
        
        # Filter and sort recommendations
        recommendations = []
        for item_idx in np.argsort(predictions)[::-1]:
            item_id = self.idx_to_item[item_idx]
            if item_id not in user_rated_items and len(recommendations) < n_recommendations:
                item_info = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
                
                recommendations.append({
                    'item_id': item_id,
                    'score': float(predictions[item_idx]),
                    'method': 'neural_cf',
                    'item_name': f"{item_info['category']} by {item_info['brand']}",
                    'price': float(item_info['price']),
                    'category': item_info['category']
                })
        
        return recommendations
    
    def _hybrid_recommendations(self, user_idx: int, n_recommendations: int) -> List[Dict]:
        """Generate hybrid recommendations"""
        # Get recommendations from different methods
        collab_recs = self._collaborative_recommendations(user_idx, n_recommendations * 2)
        content_recs = self._content_based_recommendations(self.idx_to_user[user_idx], n_recommendations * 2)
        neural_recs = self._neural_recommendations(user_idx, n_recommendations * 2)
        
        # Combine scores
        item_scores = {}
        
        # Collaborative filtering scores
        for rec in collab_recs:
            item_id = rec['item_id']
            item_scores[item_id] = item_scores.get(item_id, 0) + rec['score'] * self.hybrid_weights['collaborative']
        
        # Content-based scores
        for rec in content_recs:
            item_id = rec['item_id']
            item_scores[item_id] = item_scores.get(item_id, 0) + rec['score'] * self.hybrid_weights['content']
        
        # Neural CF scores
        for rec in neural_recs:
            item_id = rec['item_id']
            item_scores[item_id] = item_scores.get(item_id, 0) + rec['score'] * self.hybrid_weights['neural']
        
        # Add popularity bonus
        for item_id in item_scores:
            popularity = self.item_popularity.get(item_id, 0)
            normalized_popularity = popularity / max(self.item_popularity.values()) if self.item_popularity else 0
            item_scores[item_id] += normalized_popularity * self.hybrid_weights['popularity']
        
        # Sort and format recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in sorted_items[:n_recommendations]:
            item_info = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
            
            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'method': 'hybrid',
                'item_name': f"{item_info['category']} by {item_info['brand']}",
                'price': float(item_info['price']),
                'category': item_info['category']
            })
        
        return recommendations
    
    def _get_popular_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Get popular item recommendations (for cold start)"""
        popular_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, popularity in popular_items[:n_recommendations]:
            item_info = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
            
            recommendations.append({
                'item_id': item_id,
                'score': float(popularity),
                'method': 'popularity',
                'item_name': f"{item_info['category']} by {item_info['brand']}",
                'price': float(item_info['price']),
                'category': item_info['category']
            })
        
        return recommendations
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Dict]:
        """Get items similar to a given item"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if item_id not in self.item_to_idx:
            return []
        
        item_idx = self.item_to_idx[item_id]
        
        # Use content-based similarity
        similarities = self.item_content_similarity[item_idx]
        
        similar_items = []
        for idx in np.argsort(similarities)[::-1][1:n_similar+1]:  # Exclude the item itself
            similar_item_id = self.idx_to_item[idx]
            item_info = self.items_df[self.items_df['item_id'] == similar_item_id].iloc[0]
            
            similar_items.append({
                'item_id': similar_item_id,
                'similarity_score': float(similarities[idx]),
                'item_name': f"{item_info['category']} by {item_info['brand']}",
                'price': float(item_info['price']),
                'category': item_info['category']
            })
        
        return similar_items
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """Analyze user preferences"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return {'error': 'No interactions found for user'}
        
        # Category preferences
        category_ratings = user_interactions.groupby('category')['rating'].agg(['mean', 'count']).reset_index()
        category_ratings = category_ratings.merge(
            user_interactions.groupby('category')['item_id'].nunique().reset_index().rename(columns={'item_id': 'unique_items'}),
            on='category'
        )
        
        # Price preferences
        price_stats = user_interactions.merge(self.items_df[['item_id', 'price']], on='item_id')
        avg_price = price_stats['price'].mean()
        price_range = (price_stats['price'].min(), price_stats['price'].max())
        
        # Rating patterns
        avg_rating = user_interactions['rating'].mean()
        rating_std = user_interactions['rating'].std()
        
        # Interaction patterns
        interaction_types = user_interactions['interaction_type'].value_counts().to_dict()
        
        return {
            'user_id': user_id,
            'total_interactions': len(user_interactions),
            'unique_items': user_interactions['item_id'].nunique(),
            'average_rating': float(avg_rating),
            'rating_std': float(rating_std) if not pd.isna(rating_std) else 0,
            'average_price': float(avg_price),
            'price_range': [float(price_range[0]), float(price_range[1])],
            'category_preferences': category_ratings.to_dict('records'),
            'interaction_patterns': interaction_types,
            'most_recent_interaction': user_interactions['timestamp'].max().isoformat()
        }
    
    def save_models(self, filepath: str):
        """Save recommendation models"""
        model_data = {
            'models': {k: v for k, v in self.models.items() if k != 'neural_cf'},
            'scalers': self.scalers,
            'vectorizers': self.vectorizers,
            'user_item_matrix': self.user_item_matrix,
            'item_features': self.item_features,
            'user_features': self.user_features,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'item_similarity_matrix': self.item_similarity_matrix,
            'item_content_similarity': self.item_content_similarity,
            'hybrid_weights': self.hybrid_weights,
            'item_popularity': self.item_popularity,
            'label_encoders': self.label_encoders,
            'user_label_encoders': self.user_label_encoders,
            'is_fitted': self.is_fitted
        }
        
        # Save traditional models
        joblib.dump(model_data, f"{filepath}_recommendation_models.joblib")
        
        # Save neural network separately
        if 'neural_cf' in self.models:
            self.models['neural_cf'].save(f"{filepath}_neural_cf.h5")
        
        # Save data
        self.users_df.to_csv(f"{filepath}_users.csv", index=False)
        self.items_df.to_csv(f"{filepath}_items.csv", index=False)
        self.interactions_df.to_csv(f"{filepath}_interactions.csv", index=False)
        
        print(f"💾 Recommendation models saved to {filepath}")
    
    @classmethod
    def load_models(cls, filepath: str):
        """Load trained models"""
        # Load traditional models
        model_data = joblib.load(f"{filepath}_recommendation_models.joblib")
        
        instance = cls()
        for key, value in model_data.items():
            setattr(instance, key, value)
        
        # Load neural network
        try:
            neural_cf_path = f"{filepath}_neural_cf.h5"
            instance.models['neural_cf'] = tf.keras.models.load_model(neural_cf_path)
        except:
            pass  # Neural network doesn't exist
        
        # Load data
        instance.users_df = pd.read_csv(f"{filepath}_users.csv")
        instance.items_df = pd.read_csv(f"{filepath}_items.csv")
        instance.interactions_df = pd.read_csv(f"{filepath}_interactions.csv")
        
        print(f"📥 Recommendation models loaded from {filepath}")
        return instance


def create_recommendation_system():
    """Create and train the recommendation system"""
    print("🎯 Building Advanced Recommendation Engine...")
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine(random_state=42)
    
    # Train the system
    rec_engine.fit()
    
    # Test the system
    print(f"\n🧪 Testing Recommendation System:")
    
    # Test user recommendations
    test_user_id = rec_engine.user_ids[0]  # Get first user
    
    print(f"\n👤 Recommendations for User {test_user_id}:")
    
    # Test different methods
    methods = ['collaborative', 'content', 'neural', 'hybrid']
    
    for method in methods:
        try:
            recommendations = rec_engine.recommend_items(test_user_id, n_recommendations=5, method=method)
            print(f"\n📋 {method.title()} Recommendations:")
            for i, rec in enumerate(recommendations[:3]):
                print(f"   {i+1}. {rec['item_name']} - Score: {rec['score']:.3f} - ${rec['price']:.2f}")
        except Exception as e:
            print(f"   ❌ {method} failed: {e}")
    
    # Test item similarity
    test_item_id = rec_engine.item_ids[0]
    similar_items = rec_engine.get_similar_items(test_item_id, n_similar=3)
    
    print(f"\n🔍 Items similar to Item {test_item_id}:")
    for i, item in enumerate(similar_items):
        print(f"   {i+1}. {item['item_name']} - Similarity: {item['similarity_score']:.3f}")
    
    # Test user preferences analysis
    user_prefs = rec_engine.get_user_preferences(test_user_id)
    print(f"\n📊 User {test_user_id} Preferences:")
    print(f"   Total Interactions: {user_prefs['total_interactions']}")
    print(f"   Average Rating: {user_prefs['average_rating']:.2f}")
    print(f"   Average Price: ${user_prefs['average_price']:.2f}")
    print(f"   Top Categories: {[cat['category'] for cat in user_prefs['category_preferences'][:3]]}")
    
    # System statistics
    print(f"\n📈 System Statistics:")
    print(f"   Total Users: {len(rec_engine.user_ids)}")
    print(f"   Total Items: {len(rec_engine.item_ids)}")
    print(f"   Total Interactions: {len(rec_engine.interactions_df)}")
    print(f"   Sparsity: {(1 - len(rec_engine.interactions_df) / (len(rec_engine.user_ids) * len(rec_engine.item_ids))) * 100:.2f}%")
    
    # Save models
    model_path = '/Users/dhruvdabhi/temp/SmartMarketer/ml-backend/advanced_models/recommendation_engine'
    rec_engine.save_models(model_path)
    
    return rec_engine


if __name__ == "__main__":
    # Run the complete recommendation system
    rec_system = create_recommendation_system()
    print("\n🚀 Advanced Recommendation Engine is ready for deployment!")
