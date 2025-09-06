#!/usr/bin/env python3
"""
T3SS Project
File: core/querying/ranking/ml_ranker.py
(c) 2025 Qiss Labs. All Rights Reserved.
Unauthorized copying or distribution of this file is strictly prohibited.
For internal use only.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import hashlib

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from sentence_transformers import SentenceTransformer

# Async and Performance
import asyncio
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configuration
@dataclass
class RankingConfig:
    """Configuration for ML ranking system"""
    # Model settings
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    enable_random_forest: bool = True
    enable_linear_models: bool = True
    enable_deep_learning: bool = True
    
    # Feature settings
    enable_text_features: bool = True
    enable_link_features: bool = True
    enable_user_features: bool = True
    enable_temporal_features: bool = True
    enable_semantic_features: bool = True
    
    # Performance settings
    max_features: int = 1000
    batch_size: int = 1000
    max_workers: int = 4
    cache_size: int = 10000
    model_update_interval: int = 3600  # 1 hour
    
    # Model parameters
    xgb_params: Dict = None
    lgb_params: Dict = None
    rf_params: Dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.lgb_params is None:
            self.lgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }

@dataclass
class Document:
    """Document representation for ranking"""
    doc_id: str
    url: str
    title: str
    content: str
    domain: str
    content_type: str
    language: str
    timestamp: int
    content_length: int
    in_links: int = 0
    out_links: int = 0
    page_rank: float = 0.0
    click_count: int = 0
    impression_count: int = 0
    bounce_rate: float = 0.0
    dwell_time: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    authority_score: float = 0.0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class Query:
    """Query representation for ranking"""
    query_id: str
    text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: int = None
    intent: str = "informational"
    location: Optional[str] = None
    device_type: str = "desktop"
    language: str = "en"
    filters: Dict[str, Any] = None

@dataclass
class RankingResult:
    """Ranking result with score and explanation"""
    doc_id: str
    score: float
    model_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    explanation: str
    confidence: float

class FeatureExtractor:
    """Extracts features for ML ranking models"""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def extract_features(self, query: Query, documents: List[Document]) -> np.ndarray:
        """Extract features for query-document pairs"""
        features = []
        
        for doc in documents:
            doc_features = []
            
            # Text features
            if self.config.enable_text_features:
                doc_features.extend(self._extract_text_features(query, doc))
            
            # Link features
            if self.config.enable_link_features:
                doc_features.extend(self._extract_link_features(doc))
            
            # User features
            if self.config.enable_user_features:
                doc_features.extend(self._extract_user_features(query, doc))
            
            # Temporal features
            if self.config.enable_temporal_features:
                doc_features.extend(self._extract_temporal_features(query, doc))
            
            # Semantic features
            if self.config.enable_semantic_features:
                doc_features.extend(self._extract_semantic_features(query, doc))
            
            features.append(doc_features)
        
        return np.array(features)
    
    def _extract_text_features(self, query: Query, doc: Document) -> List[float]:
        """Extract text-based features"""
        features = []
        
        # Query-document similarity
        query_terms = set(query.text.lower().split())
        doc_terms = set((doc.title + " " + doc.content).lower().split())
        
        # Term overlap
        overlap = len(query_terms.intersection(doc_terms))
        features.append(overlap / len(query_terms) if query_terms else 0)
        
        # Title match
        title_match = any(term in doc.title.lower() for term in query_terms)
        features.append(1.0 if title_match else 0.0)
        
        # Content length
        features.append(np.log1p(doc.content_length))
        
        # Language match
        features.append(1.0 if query.language == doc.language else 0.0)
        
        # Content type relevance
        content_type_scores = {
            'text/html': 1.0,
            'application/pdf': 0.8,
            'text/plain': 0.6,
            'application/json': 0.4
        }
        features.append(content_type_scores.get(doc.content_type, 0.5))
        
        return features
    
    def _extract_link_features(self, doc: Document) -> List[float]:
        """Extract link-based features"""
        features = []
        
        # PageRank
        features.append(doc.page_rank)
        
        # In-link count
        features.append(np.log1p(doc.in_links))
        
        # Out-link count
        features.append(np.log1p(doc.out_links))
        
        # Authority score
        features.append(doc.authority_score)
        
        return features
    
    def _extract_user_features(self, query: Query, doc: Document) -> List[float]:
        """Extract user behavior features"""
        features = []
        
        # Click-through rate
        ctr = doc.click_count / max(doc.impression_count, 1)
        features.append(ctr)
        
        # Bounce rate (inverted)
        features.append(1.0 - doc.bounce_rate)
        
        # Dwell time
        features.append(np.log1p(doc.dwell_time))
        
        # Quality score
        features.append(doc.quality_score)
        
        return features
    
    def _extract_temporal_features(self, query: Query, doc: Document) -> List[float]:
        """Extract temporal features"""
        features = []
        
        # Document age
        current_time = time.time()
        doc_age = (current_time - doc.timestamp) / (24 * 3600)  # days
        features.append(np.log1p(doc_age))
        
        # Freshness score
        features.append(doc.freshness_score)
        
        # Time-based relevance (e.g., news articles)
        if doc.content_type == 'news':
            # Recent news gets higher score
            freshness = max(0, 1 - doc_age / 7)  # 7 days decay
            features.append(freshness)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_semantic_features(self, query: Query, doc: Document) -> List[float]:
        """Extract semantic features using embeddings"""
        features = []
        
        try:
            # Generate embeddings
            query_embedding = self.sentence_transformer.encode([query.text])
            doc_embedding = self.sentence_transformer.encode([doc.title + " " + doc.content])
            
            # Cosine similarity
            similarity = np.dot(query_embedding[0], doc_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding[0])
            )
            features.append(similarity)
            
            # Relevance score
            features.append(doc.relevance_score)
            
        except Exception as e:
            logging.warning(f"Failed to extract semantic features: {e}")
            features.extend([0.0, 0.0])
        
        return features

class DeepLearningRanker(nn.Module):
    """Deep learning model for ranking"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(DeepLearningRanker, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

class MLRanker:
    """Main ML ranking system with multiple models"""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        self.cache = {}
        self.redis_client = None
        self.lock = threading.Lock()
        
        # Metrics
        self.ranking_counter = Counter('ranking_requests_total', 'Total ranking requests')
        self.ranking_duration = Histogram('ranking_duration_seconds', 'Ranking duration')
        self.model_accuracy = Gauge('model_accuracy', 'Model accuracy', ['model_name'])
        
        # Initialize models
        self._initialize_models()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        if self.config.enable_xgboost:
            self.models['xgboost'] = xgb.XGBRegressor(**self.config.xgb_params)
            self.model_weights['xgboost'] = 0.3
        
        if self.config.enable_lightgbm:
            self.models['lightgbm'] = lgb.LGBMRegressor(**self.config.lgb_params)
            self.model_weights['lightgbm'] = 0.3
        
        if self.config.enable_random_forest:
            self.models['random_forest'] = RandomForestRegressor(**self.config.rf_params)
            self.model_weights['random_forest'] = 0.2
        
        if self.config.enable_linear_models:
            self.models['ridge'] = Ridge(alpha=1.0)
            self.models['lasso'] = Lasso(alpha=0.1)
            self.model_weights['ridge'] = 0.1
            self.model_weights['lasso'] = 0.1
        
        if self.config.enable_deep_learning:
            # Will be initialized after feature extraction
            self.model_weights['deep_learning'] = 0.2
    
    def _start_background_tasks(self):
        """Start background tasks for model updates and monitoring"""
        def update_models():
            while True:
                time.sleep(self.config.model_update_interval)
                self._update_models()
        
        def monitor_performance():
            while True:
                time.sleep(300)  # 5 minutes
                self._monitor_performance()
        
        threading.Thread(target=update_models, daemon=True).start()
        threading.Thread(target=monitor_performance, daemon=True).start()
    
    async def rank_documents(self, query: Query, documents: List[Document]) -> List[RankingResult]:
        """Rank documents for a given query"""
        start_time = time.time()
        self.ranking_counter.inc()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, documents)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Extract features
            features = self.feature_extractor.extract_features(query, documents)
            
            if features.size == 0:
                return []
            
            # Get predictions from all models
            predictions = {}
            model_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(features)
                        predictions[model_name] = pred
                        model_scores[model_name] = pred
                except Exception as e:
                    logging.warning(f"Model {model_name} prediction failed: {e}")
                    continue
            
            # Ensemble prediction
            final_scores = self._ensemble_predictions(predictions)
            
            # Create ranking results
            results = []
            for i, (doc, score) in enumerate(zip(documents, final_scores)):
                result = RankingResult(
                    doc_id=doc.doc_id,
                    score=float(score),
                    model_scores={name: float(scores[i]) for name, scores in model_scores.items()},
                    feature_importance=self._get_feature_importance(documents[i]),
                    explanation=self._generate_explanation(query, doc, score),
                    confidence=self._calculate_confidence(predictions, i)
                )
                results.append(result)
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Cache results
            self.cache[cache_key] = results
            if len(self.cache) > self.config.cache_size:
                # Remove oldest entries
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            # Record metrics
            duration = time.time() - start_time
            self.ranking_duration.observe(duration)
            
            return results
            
        except Exception as e:
            logging.error(f"Ranking failed: {e}")
            return []
    
    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models"""
        if not predictions:
            return np.array([])
        
        # Weighted average
        total_weight = sum(self.model_weights.get(name, 0) for name in predictions.keys())
        if total_weight == 0:
            return np.array([])
        
        ensemble_score = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0) / total_weight
            ensemble_score += weight * pred
        
        return ensemble_score
    
    def _get_feature_importance(self, doc: Document) -> Dict[str, float]:
        """Get feature importance for a document"""
        # This would be calculated based on the model's feature importance
        # For now, return a simplified version
        return {
            'text_similarity': 0.3,
            'page_rank': 0.2,
            'click_through_rate': 0.2,
            'freshness': 0.15,
            'authority': 0.15
        }
    
    def _generate_explanation(self, query: Query, doc: Document, score: float) -> str:
        """Generate human-readable explanation for ranking"""
        explanations = []
        
        if doc.page_rank > 0.5:
            explanations.append("High PageRank authority")
        
        if any(term in doc.title.lower() for term in query.text.lower().split()):
            explanations.append("Title matches query")
        
        if doc.click_count > 100:
            explanations.append("Popular content")
        
        if doc.freshness_score > 0.8:
            explanations.append("Recent content")
        
        return "; ".join(explanations) if explanations else "Standard relevance"
    
    def _calculate_confidence(self, predictions: Dict[str, np.ndarray], index: int) -> float:
        """Calculate confidence in the ranking"""
        if not predictions:
            return 0.0
        
        # Calculate variance across models
        scores = [pred[index] for pred in predictions.values()]
        if len(scores) < 2:
            return 1.0
        
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Higher variance = lower confidence
        confidence = 1.0 / (1.0 + variance)
        return float(confidence)
    
    def _generate_cache_key(self, query: Query, documents: List[Document]) -> str:
        """Generate cache key for query-document pairs"""
        doc_ids = sorted([doc.doc_id for doc in documents])
        key_data = f"{query.query_id}:{':'.join(doc_ids)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def train_models(self, training_data: List[Tuple[Query, List[Document], List[float]]]):
        """Train all models with labeled data"""
        logging.info(f"Training models with {len(training_data)} examples")
        
        # Prepare training data
        X, y = [], []
        
        for query, docs, labels in training_data:
            features = self.feature_extractor.extract_features(query, docs)
            X.extend(features)
            y.extend(labels)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_extractor.scaler.transform(X_test)
        
        # Train models
        for model_name, model in self.models.items():
            try:
                logging.info(f"Training {model_name}")
                
                if model_name in ['ridge', 'lasso']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
                # Update model accuracy metric
                self.model_accuracy.labels(model_name=model_name).set(r2)
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                
            except Exception as e:
                logging.error(f"Failed to train {model_name}: {e}")
        
        # Train deep learning model
        if self.config.enable_deep_learning:
            self._train_deep_learning_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    def _train_deep_learning_model(self, X_train, y_train, X_test, y_test):
        """Train deep learning model"""
        try:
            input_dim = X_train.shape[1]
            model = DeepLearningRanker(input_dim)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logging.info(f"Deep Learning Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor).numpy()
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"Deep Learning - MSE: {mse:.4f}, R²: {r2:.4f}")
                self.model_accuracy.labels(model_name='deep_learning').set(r2)
            
            self.models['deep_learning'] = model
            
        except Exception as e:
            logging.error(f"Failed to train deep learning model: {e}")
    
    def _update_models(self):
        """Update models with new data"""
        # This would implement online learning or model retraining
        logging.info("Updating models with new data")
    
    def _monitor_performance(self):
        """Monitor model performance and update weights"""
        # This would implement performance monitoring and weight adjustment
        logging.info("Monitoring model performance")
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'scaler': self.feature_extractor.scaler,
            'config': asdict(self.config)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.model_weights = model_data['model_weights']
        self.feature_importance = model_data['feature_importance']
        self.feature_extractor.scaler = model_data['scaler']
        
        logging.info(f"Models loaded from {filepath}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the ranking system"""
        return {
            'models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'cache_size': len(self.cache),
            'feature_importance': self.feature_importance,
            'config': asdict(self.config)
        }

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = RankingConfig()
    
    # Create ranker
    ranker = MLRanker(config)
    
    # Example query and documents
    query = Query(
        query_id="q1",
        text="machine learning algorithms",
        user_id="user1",
        timestamp=int(time.time())
    )
    
    documents = [
        Document(
            doc_id="doc1",
            url="https://example.com/ml-intro",
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence...",
            domain="example.com",
            content_type="text/html",
            language="en",
            timestamp=int(time.time() - 86400),
            content_length=5000,
            page_rank=0.8,
            quality_score=0.9
        ),
        Document(
            doc_id="doc2",
            url="https://example.com/algorithms",
            title="Algorithm Design",
            content="Algorithms are step-by-step procedures...",
            domain="example.com",
            content_type="text/html",
            language="en",
            timestamp=int(time.time() - 172800),
            content_length=3000,
            page_rank=0.6,
            quality_score=0.7
        )
    ]
    
    # Test ranking
    async def test_ranking():
        results = await ranker.rank_documents(query, documents)
        for result in results:
            print(f"Doc {result.doc_id}: Score {result.score:.4f}, Confidence {result.confidence:.4f}")
            print(f"Explanation: {result.explanation}")
            print()
    
    # Run test
    asyncio.run(test_ranking())