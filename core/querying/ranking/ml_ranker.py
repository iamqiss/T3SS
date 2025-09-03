# T3SS Project
# File: core/querying/ranking/ml_ranker.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class RankingFeatures:
    """Features used for ML-based ranking"""
    # Query features
    query_length: int
    query_complexity: float
    query_type: str  # 'navigational', 'informational', 'transactional'
    
    # Document features
    doc_id: str
    title_relevance: float
    content_relevance: float
    url_relevance: float
    domain_authority: float
    page_rank: float
    content_freshness: float
    content_length: int
    content_quality: float
    
    # User interaction features
    click_through_rate: float
    dwell_time: float
    bounce_rate: float
    user_satisfaction: float
    
    # Contextual features
    user_location: str
    user_device: str
    time_of_day: int
    day_of_week: int
    search_history: List[str] = field(default_factory=list)
    
    # Advanced features
    semantic_similarity: float
    entity_relevance: float
    sentiment_score: float
    language_match: float

@dataclass
class RankingResult:
    """Result of ML ranking"""
    doc_id: str
    score: float
    confidence: float
    feature_importance: Dict[str, float]
    explanation: str

class MLRanker:
    """
    Advanced ML-based ranking system with real-time learning capabilities.
    
    Features:
    - Multiple ML models (XGBoost, LightGBM, Random Forest)
    - Real-time feature engineering
    - Online learning and model updates
    - Feature importance tracking
    - A/B testing framework
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.feature_scalers = {}
        self.feature_importance = defaultdict(float)
        self.training_data = deque(maxlen=config.get('max_training_samples', 100000))
        self.performance_metrics = defaultdict(list)
        self.model_lock = threading.RLock()
        self.is_training = False
        
        # Initialize models
        self._initialize_models()
        
        # Start background training thread
        self._start_background_training()
    
    def _initialize_models(self):
        """Initialize ML models"""
        model_config = self.config.get('models', {})
        
        # XGBoost model
        if model_config.get('enable_xgboost', True):
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=model_config.get('xgboost_estimators', 100),
                max_depth=model_config.get('xgboost_depth', 6),
                learning_rate=model_config.get('xgboost_lr', 0.1),
                subsample=model_config.get('xgboost_subsample', 0.8),
                colsample_bytree=model_config.get('xgboost_colsample', 0.8),
                random_state=42
            )
        
        # LightGBM model
        if model_config.get('enable_lightgbm', True):
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=model_config.get('lightgbm_estimators', 100),
                max_depth=model_config.get('lightgbm_depth', 6),
                learning_rate=model_config.get('lightgbm_lr', 0.1),
                subsample=model_config.get('lightgbm_subsample', 0.8),
                colsample_bytree=model_config.get('lightgbm_colsample', 0.8),
                random_state=42,
                verbose=-1
            )
        
        # Random Forest model
        if model_config.get('enable_random_forest', True):
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=model_config.get('rf_estimators', 100),
                max_depth=model_config.get('rf_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        
        # Linear model for baseline
        if model_config.get('enable_linear', True):
            self.models['linear'] = Ridge(alpha=1.0)
        
        # Initialize feature scalers
        for model_name in self.models.keys():
            self.feature_scalers[model_name] = StandardScaler()
    
    def _start_background_training(self):
        """Start background thread for continuous model training"""
        def training_loop():
            while True:
                try:
                    if len(self.training_data) >= self.config.get('min_training_samples', 1000):
                        self._train_models()
                    time.sleep(self.config.get('training_interval', 300))  # 5 minutes
                except Exception as e:
                    logger.error(f"Error in background training: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()
    
    async def rank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        user_context: Dict[str, Any] = None
    ) -> List[RankingResult]:
        """
        Rank documents using ML models with real-time feature engineering
        """
        if not documents:
            return []
        
        # Extract features for all documents
        features_list = []
        for doc in documents:
            features = self._extract_features(query, doc, user_context)
            features_list.append(features)
        
        # Convert to feature matrix
        feature_matrix = self._features_to_matrix(features_list)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    # Scale features
                    scaled_features = self.feature_scalers[model_name].transform(feature_matrix)
                    pred = model.predict(scaled_features)
                    predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                continue
        
        # Ensemble predictions
        ensemble_scores = self._ensemble_predictions(predictions)
        
        # Create ranking results
        results = []
        for i, (doc, score) in enumerate(zip(documents, ensemble_scores)):
            result = RankingResult(
                doc_id=doc.get('id', str(i)),
                score=float(score),
                confidence=self._calculate_confidence(predictions, i),
                feature_importance=self._get_feature_importance(features_list[i]),
                explanation=self._generate_explanation(features_list[i], score)
            )
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _extract_features(
        self, 
        query: str, 
        document: Dict[str, Any], 
        user_context: Dict[str, Any] = None
    ) -> RankingFeatures:
        """Extract comprehensive features for ranking"""
        
        # Query features
        query_length = len(query.split())
        query_complexity = self._calculate_query_complexity(query)
        query_type = self._classify_query_type(query)
        
        # Document features
        title = document.get('title', '')
        content = document.get('content', '')
        url = document.get('url', '')
        
        title_relevance = self._calculate_text_relevance(query, title)
        content_relevance = self._calculate_text_relevance(query, content)
        url_relevance = self._calculate_text_relevance(query, url)
        
        domain_authority = document.get('domain_authority', 0.0)
        page_rank = document.get('page_rank', 0.0)
        content_freshness = self._calculate_content_freshness(document.get('timestamp', 0))
        content_length = len(content)
        content_quality = self._calculate_content_quality(content)
        
        # User interaction features (from historical data)
        doc_id = document.get('id', '')
        click_through_rate = self._get_historical_ctr(doc_id)
        dwell_time = self._get_historical_dwell_time(doc_id)
        bounce_rate = self._get_historical_bounce_rate(doc_id)
        user_satisfaction = self._get_historical_satisfaction(doc_id)
        
        # Contextual features
        user_context = user_context or {}
        user_location = user_context.get('location', 'unknown')
        user_device = user_context.get('device', 'unknown')
        time_of_day = time.localtime().tm_hour
        day_of_week = time.localtime().tm_wday
        search_history = user_context.get('search_history', [])
        
        # Advanced features
        semantic_similarity = self._calculate_semantic_similarity(query, content)
        entity_relevance = self._calculate_entity_relevance(query, content)
        sentiment_score = self._calculate_sentiment_score(content)
        language_match = self._calculate_language_match(query, content)
        
        return RankingFeatures(
            query_length=query_length,
            query_complexity=query_complexity,
            query_type=query_type,
            doc_id=doc_id,
            title_relevance=title_relevance,
            content_relevance=content_relevance,
            url_relevance=url_relevance,
            domain_authority=domain_authority,
            page_rank=page_rank,
            content_freshness=content_freshness,
            content_length=content_length,
            content_quality=content_quality,
            click_through_rate=click_through_rate,
            dwell_time=dwell_time,
            bounce_rate=bounce_rate,
            user_satisfaction=user_satisfaction,
            user_location=user_location,
            user_device=user_device,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            search_history=search_history,
            semantic_similarity=semantic_similarity,
            entity_relevance=entity_relevance,
            sentiment_score=sentiment_score,
            language_match=language_match
        )
    
    def _features_to_matrix(self, features_list: List[RankingFeatures]) -> np.ndarray:
        """Convert features to numpy matrix"""
        feature_vectors = []
        
        for features in features_list:
            vector = [
                features.query_length,
                features.query_complexity,
                self._encode_query_type(features.query_type),
                features.title_relevance,
                features.content_relevance,
                features.url_relevance,
                features.domain_authority,
                features.page_rank,
                features.content_freshness,
                features.content_length,
                features.content_quality,
                features.click_through_rate,
                features.dwell_time,
                features.bounce_rate,
                features.user_satisfaction,
                self._encode_location(features.user_location),
                self._encode_device(features.user_device),
                features.time_of_day,
                features.day_of_week,
                features.semantic_similarity,
                features.entity_relevance,
                features.sentiment_score,
                features.language_match
            ]
            feature_vectors.append(vector)
        
        return np.array(feature_vectors)
    
    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models"""
        if not predictions:
            return np.array([])
        
        # Weighted ensemble (can be learned from validation data)
        weights = {
            'xgboost': 0.4,
            'lightgbm': 0.3,
            'random_forest': 0.2,
            'linear': 0.1
        }
        
        ensemble_scores = np.zeros(len(next(iter(predictions.values()))))
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.0)
            ensemble_scores += weight * pred
        
        return ensemble_scores
    
    def add_training_sample(
        self, 
        query: str, 
        document: Dict[str, Any], 
        user_context: Dict[str, Any],
        relevance_score: float
    ):
        """Add a training sample for online learning"""
        features = self._extract_features(query, document, user_context)
        self.training_data.append((features, relevance_score))
    
    def _train_models(self):
        """Train all models with current training data"""
        if self.is_training or len(self.training_data) < 100:
            return
        
        with self.model_lock:
            self.is_training = True
        
        try:
            # Prepare training data
            features_list, labels = zip(*self.training_data)
            X = self._features_to_matrix(list(features_list))
            y = np.array(labels)
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    X_scaled = self.feature_scalers[model_name].fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                    # Update feature importance
                    if hasattr(model, 'feature_importances_'):
                        for i, importance in enumerate(model.feature_importances_):
                            feature_name = self._get_feature_name(i)
                            self.feature_importance[feature_name] = importance
                    
                    logger.info(f"Trained {model_name} model with {len(X)} samples")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
            
            # Evaluate models
            self._evaluate_models(X, y)
            
        finally:
            with self.model_lock:
                self.is_training = False
    
    def _evaluate_models(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance"""
        # Split data for evaluation
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for model_name, model in self.models.items():
            try:
                X_train_scaled = self.feature_scalers[model_name].transform(X_train)
                X_test_scaled = self.feature_scalers[model_name].transform(X_test)
                
                # Train on training set
                model.fit(X_train_scaled, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                self.performance_metrics[model_name].append({
                    'mse': mse,
                    'mae': mae,
                    'timestamp': time.time()
                })
                
                logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
    
    # Helper methods for feature extraction
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        words = query.split()
        if not words:
            return 0.0
        
        # Simple complexity based on word count and special characters
        complexity = len(words) * 0.1
        complexity += len([c for c in query if c in '?!"@#$%^&*()']) * 0.2
        return min(complexity, 1.0)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return 'informational'
        elif any(word in query_lower for word in ['buy', 'purchase', 'shop', 'price']):
            return 'transactional'
        else:
            return 'navigational'
    
    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Calculate text relevance score"""
        if not text:
            return 0.0
        
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = len(query_words.intersection(text_words))
        return intersection / len(query_words)
    
    def _calculate_content_freshness(self, timestamp: int) -> float:
        """Calculate content freshness score"""
        if timestamp == 0:
            return 0.5  # Default for unknown timestamps
        
        current_time = time.time()
        age_days = (current_time - timestamp) / (24 * 3600)
        
        # Exponential decay: fresher content gets higher scores
        return np.exp(-age_days / 365)  # Decay over a year
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score"""
        if not content:
            return 0.0
        
        # Simple quality metrics
        word_count = len(content.split())
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        
        if sentence_count == 0:
            return 0.0
        
        avg_sentence_length = word_count / sentence_count
        
        # Quality score based on length and structure
        quality = min(word_count / 100, 1.0)  # Normalize by 100 words
        quality *= min(avg_sentence_length / 20, 1.0)  # Normalize by 20 words per sentence
        
        return quality
    
    def _get_historical_ctr(self, doc_id: str) -> float:
        """Get historical click-through rate for document"""
        # Placeholder - would query historical data
        return 0.1  # Default CTR
    
    def _get_historical_dwell_time(self, doc_id: str) -> float:
        """Get historical dwell time for document"""
        # Placeholder - would query historical data
        return 30.0  # Default dwell time in seconds
    
    def _get_historical_bounce_rate(self, doc_id: str) -> float:
        """Get historical bounce rate for document"""
        # Placeholder - would query historical data
        return 0.3  # Default bounce rate
    
    def _get_historical_satisfaction(self, doc_id: str) -> float:
        """Get historical user satisfaction for document"""
        # Placeholder - would query historical data
        return 0.7  # Default satisfaction score
    
    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """Calculate semantic similarity between query and content"""
        # Placeholder - would use embedding models
        return 0.5
    
    def _calculate_entity_relevance(self, query: str, content: str) -> float:
        """Calculate entity relevance score"""
        # Placeholder - would use NER models
        return 0.5
    
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate sentiment score of content"""
        # Placeholder - would use sentiment analysis
        return 0.0  # Neutral sentiment
    
    def _calculate_language_match(self, query: str, content: str) -> float:
        """Calculate language match score"""
        # Placeholder - would use language detection
        return 1.0  # Assume same language
    
    def _encode_query_type(self, query_type: str) -> float:
        """Encode query type as numeric value"""
        encoding = {'navigational': 0.0, 'informational': 0.5, 'transactional': 1.0}
        return encoding.get(query_type, 0.5)
    
    def _encode_location(self, location: str) -> float:
        """Encode location as numeric value"""
        # Placeholder - would use proper location encoding
        return 0.5
    
    def _encode_device(self, device: str) -> float:
        """Encode device type as numeric value"""
        encoding = {'mobile': 0.0, 'tablet': 0.5, 'desktop': 1.0}
        return encoding.get(device, 0.5)
    
    def _get_feature_name(self, index: int) -> str:
        """Get feature name by index"""
        feature_names = [
            'query_length', 'query_complexity', 'query_type', 'title_relevance',
            'content_relevance', 'url_relevance', 'domain_authority', 'page_rank',
            'content_freshness', 'content_length', 'content_quality', 'ctr',
            'dwell_time', 'bounce_rate', 'satisfaction', 'location', 'device',
            'time_of_day', 'day_of_week', 'semantic_similarity', 'entity_relevance',
            'sentiment_score', 'language_match'
        ]
        return feature_names[index] if index < len(feature_names) else f'feature_{index}'
    
    def _calculate_confidence(self, predictions: Dict[str, np.ndarray], index: int) -> float:
        """Calculate confidence score for prediction"""
        if not predictions:
            return 0.0
        
        # Calculate variance across models as confidence measure
        scores = [pred[index] for pred in predictions.values()]
        if len(scores) < 2:
            return 0.5
        
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + variance)
        return confidence
    
    def _get_feature_importance(self, features: RankingFeatures) -> Dict[str, float]:
        """Get feature importance for a specific ranking"""
        # Return top features based on global importance
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return dict(top_features)
    
    def _generate_explanation(self, features: RankingFeatures, score: float) -> str:
        """Generate human-readable explanation for ranking"""
        explanations = []
        
        if features.title_relevance > 0.8:
            explanations.append("High title relevance")
        if features.content_relevance > 0.7:
            explanations.append("Strong content match")
        if features.domain_authority > 0.8:
            explanations.append("High domain authority")
        if features.content_freshness > 0.8:
            explanations.append("Fresh content")
        
        if not explanations:
            explanations.append("Standard relevance")
        
        return "; ".join(explanations)
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        performance = {}
        
        for model_name, metrics in self.performance_metrics.items():
            if metrics:
                latest = metrics[-1]
                performance[model_name] = {
                    'mse': latest['mse'],
                    'mae': latest['mae'],
                    'last_updated': latest['timestamp']
                }
        
        return performance
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.feature_scalers,
            'feature_importance': dict(self.feature_importance),
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.feature_scalers = model_data['scalers']
        self.feature_importance = defaultdict(float, model_data['feature_importance'])
        self.config.update(model_data['config'])