# T3SS Project
# File: core/indexing/spam_detector/ml_classifier.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import asyncio
import logging
import time
import pickle
import json
from collections import defaultdict, Counter
import re
import hashlib
from urllib.parse import urlparse
import threading
from concurrent.futures import ThreadPoolExecutor

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

@dataclass
class SpamFeatures:
    """Features extracted from content for spam detection"""
    # Text features
    text_length: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    capital_ratio: float
    digit_ratio: float
    special_char_ratio: float
    
    # URL features
    url_count: int
    external_url_count: int
    suspicious_domain_count: int
    url_length_avg: float
    
    # Content features
    spam_word_count: int
    commercial_word_count: int
    repetition_ratio: float
    language_consistency: float
    
    # Metadata features
    domain_age: float
    domain_authority: float
    content_freshness: float
    user_engagement: float
    
    # Behavioral features
    click_through_rate: float
    bounce_rate: float
    time_on_page: float
    user_feedback_score: float

@dataclass
class SpamResult:
    """Result of spam classification"""
    is_spam: bool
    confidence: float
    spam_score: float
    ham_score: float
    features_used: List[str]
    explanation: str
    model_used: str
    processing_time: float

class SpamDataset(Dataset):
    """PyTorch dataset for spam detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SpamBERTModel(nn.Module):
    """BERT-based spam detection model"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(SpamBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class MLSpamClassifier:
    """
    Advanced ML-based spam detection system with multiple models and ensemble learning.
    
    Features:
    - Multiple ML algorithms (Random Forest, XGBoost, LightGBM, BERT)
    - Comprehensive feature engineering
    - Real-time learning and model updates
    - Ensemble voting for improved accuracy
    - Performance monitoring and A/B testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.feature_extractors = {}
        self.scalers = {}
        self.label_encoders = {}
        self.training_data = []
        self.model_lock = threading.RLock()
        self.is_training = False
        
        # Spam indicators
        self.spam_keywords = {
            'viagra', 'cialis', 'casino', 'poker', 'lottery', 'winner', 'congratulations',
            'free money', 'click here', 'buy now', 'limited time', 'act now', 'urgent',
            'guaranteed', 'no risk', 'make money', 'work from home', 'get rich'
        }
        
        self.commercial_keywords = {
            'buy', 'purchase', 'sale', 'discount', 'offer', 'deal', 'price', 'cost',
            'order', 'shop', 'store', 'product', 'service', 'company', 'business'
        }
        
        self.suspicious_domains = {
            '.tk', '.ml', '.ga', '.cf', '.click', '.download', '.exe', '.zip'
        }
        
        # Initialize models
        self._initialize_models()
        
        # Start background training
        self._start_background_training()
    
    def _initialize_models(self):
        """Initialize ML models"""
        model_config = self.config.get('models', {})
        
        # Traditional ML models
        if model_config.get('enable_random_forest', True):
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        if model_config.get('enable_xgboost', True):
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        if model_config.get('enable_lightgbm', True):
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        if model_config.get('enable_svm', True):
            self.models['svm'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        
        if model_config.get('enable_naive_bayes', True):
            self.models['naive_bayes'] = MultinomialNB()
        
        # Initialize feature extractors
        self.feature_extractors['tfidf'] = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.feature_extractors['count'] = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
        
        # Initialize BERT model if enabled
        if model_config.get('enable_bert', True):
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = SpamBERTModel()
                self.models['bert'] = self.bert_model
            except Exception as e:
                logger.warning(f"Failed to initialize BERT model: {e}")
    
    def _start_background_training(self):
        """Start background thread for continuous model training"""
        def training_loop():
            while True:
                try:
                    if len(self.training_data) >= self.config.get('min_training_samples', 1000):
                        self._train_models()
                    time.sleep(self.config.get('training_interval', 3600))  # 1 hour
                except Exception as e:
                    logger.error(f"Error in background training: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()
    
    async def classify(self, content: str, metadata: Dict[str, Any] = None) -> SpamResult:
        """Classify content as spam or ham"""
        start_time = time.time()
        
        try:
            # Extract features
            features = self._extract_features(content, metadata or {})
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'bert':
                        pred, conf = await self._predict_bert(content)
                    else:
                        pred, conf = self._predict_traditional(model_name, features)
                    
                    predictions[model_name] = pred
                    confidences[model_name] = conf
                    
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    continue
            
            # Ensemble voting
            final_prediction, final_confidence = self._ensemble_vote(predictions, confidences)
            
            # Generate explanation
            explanation = self._generate_explanation(features, final_prediction)
            
            processing_time = time.time() - start_time
            
            return SpamResult(
                is_spam=final_prediction,
                confidence=final_confidence,
                spam_score=final_confidence if final_prediction else 1.0 - final_confidence,
                ham_score=1.0 - final_confidence if final_prediction else final_confidence,
                features_used=list(features.__dict__.keys()),
                explanation=explanation,
                model_used='ensemble',
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Spam classification failed: {e}")
            return SpamResult(
                is_spam=False,
                confidence=0.5,
                spam_score=0.5,
                ham_score=0.5,
                features_used=[],
                explanation=f"Classification failed: {str(e)}",
                model_used='error',
                processing_time=time.time() - start_time
            )
    
    def _extract_features(self, content: str, metadata: Dict[str, Any]) -> SpamFeatures:
        """Extract comprehensive features from content"""
        
        # Text features
        text_length = len(content)
        words = content.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        capital_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        digit_ratio = sum(1 for c in content if c.isdigit()) / len(content) if content else 0
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content) if content else 0
        
        # URL features
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        url_count = len(urls)
        
        external_url_count = 0
        suspicious_domain_count = 0
        url_length_avg = 0
        
        if urls:
            for url in urls:
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    
                    # Check for external domains
                    if not any(trusted in domain for trusted in ['google.com', 'wikipedia.org', 'github.com']):
                        external_url_count += 1
                    
                    # Check for suspicious domains
                    if any(suspicious in domain for suspicious in self.suspicious_domains):
                        suspicious_domain_count += 1
                    
                except:
                    pass
            
            url_length_avg = np.mean([len(url) for url in urls])
        
        # Content features
        content_lower = content.lower()
        spam_word_count = sum(1 for word in self.spam_keywords if word in content_lower)
        commercial_word_count = sum(1 for word in self.commercial_keywords if word in content_lower)
        
        # Repetition ratio
        word_counts = Counter(words)
        repetition_ratio = sum(1 for count in word_counts.values() if count > 1) / len(word_counts) if word_counts else 0
        
        # Language consistency (simplified)
        language_consistency = 1.0  # Placeholder
        
        # Metadata features
        domain_age = metadata.get('domain_age', 0.0)
        domain_authority = metadata.get('domain_authority', 0.0)
        content_freshness = metadata.get('content_freshness', 0.0)
        user_engagement = metadata.get('user_engagement', 0.0)
        
        # Behavioral features
        click_through_rate = metadata.get('click_through_rate', 0.0)
        bounce_rate = metadata.get('bounce_rate', 0.0)
        time_on_page = metadata.get('time_on_page', 0.0)
        user_feedback_score = metadata.get('user_feedback_score', 0.0)
        
        return SpamFeatures(
            text_length=text_length,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            capital_ratio=capital_ratio,
            digit_ratio=digit_ratio,
            special_char_ratio=special_char_ratio,
            url_count=url_count,
            external_url_count=external_url_count,
            suspicious_domain_count=suspicious_domain_count,
            url_length_avg=url_length_avg,
            spam_word_count=spam_word_count,
            commercial_word_count=commercial_word_count,
            repetition_ratio=repetition_ratio,
            language_consistency=language_consistency,
            domain_age=domain_age,
            domain_authority=domain_authority,
            content_freshness=content_freshness,
            user_engagement=user_engagement,
            click_through_rate=click_through_rate,
            bounce_rate=bounce_rate,
            time_on_page=time_on_page,
            user_feedback_score=user_feedback_score
        )
    
    def _predict_traditional(self, model_name: str, features: SpamFeatures) -> Tuple[bool, float]:
        """Predict using traditional ML models"""
        if model_name not in self.models:
            return False, 0.5
        
        model = self.models[model_name]
        
        # Convert features to array
        feature_array = np.array([
            features.text_length, features.word_count, features.sentence_count,
            features.avg_word_length, features.capital_ratio, features.digit_ratio,
            features.special_char_ratio, features.url_count, features.external_url_count,
            features.suspicious_domain_count, features.url_length_avg,
            features.spam_word_count, features.commercial_word_count,
            features.repetition_ratio, features.language_consistency,
            features.domain_age, features.domain_authority, features.content_freshness,
            features.user_engagement, features.click_through_rate, features.bounce_rate,
            features.time_on_page, features.user_feedback_score
        ]).reshape(1, -1)
        
        # Scale features
        if model_name in self.scalers:
            feature_array = self.scalers[model_name].transform(feature_array)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_array)[0]
            prediction = proba[1] > 0.5  # Assuming class 1 is spam
            confidence = max(proba)
        else:
            prediction = model.predict(feature_array)[0]
            confidence = 0.8  # Default confidence
        
        return bool(prediction), float(confidence)
    
    async def _predict_bert(self, content: str) -> Tuple[bool, float]:
        """Predict using BERT model"""
        if 'bert' not in self.models:
            return False, 0.5
        
        try:
            # Tokenize
            encoding = self.bert_tokenizer(
                content,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                probabilities = torch.softmax(outputs, dim=1)
                prediction = probabilities[0][1] > 0.5
                confidence = float(torch.max(probabilities[0]))
            
            return bool(prediction), confidence
            
        except Exception as e:
            logger.error(f"BERT prediction failed: {e}")
            return False, 0.5
    
    def _ensemble_vote(self, predictions: Dict[str, bool], confidences: Dict[str, float]) -> Tuple[bool, float]:
        """Combine predictions from multiple models using weighted voting"""
        if not predictions:
            return False, 0.5
        
        # Model weights (can be learned from validation data)
        weights = {
            'random_forest': 0.2,
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'svm': 0.15,
            'naive_bayes': 0.1,
            'bert': 0.05
        }
        
        spam_score = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.1)
            confidence = confidences.get(model_name, 0.5)
            
            if prediction:  # If model predicts spam
                spam_score += weight * confidence
            else:  # If model predicts ham
                spam_score += weight * (1.0 - confidence)
            
            total_weight += weight
        
        if total_weight == 0:
            return False, 0.5
        
        final_score = spam_score / total_weight
        final_prediction = final_score > 0.5
        final_confidence = max(final_score, 1.0 - final_score)
        
        return final_prediction, final_confidence
    
    def _generate_explanation(self, features: SpamFeatures, prediction: bool) -> str:
        """Generate human-readable explanation for the prediction"""
        explanations = []
        
        if features.spam_word_count > 0:
            explanations.append(f"Contains {features.spam_word_count} spam keywords")
        
        if features.suspicious_domain_count > 0:
            explanations.append(f"Has {features.suspicious_domain_count} suspicious domains")
        
        if features.capital_ratio > 0.3:
            explanations.append("High ratio of capital letters")
        
        if features.special_char_ratio > 0.2:
            explanations.append("High ratio of special characters")
        
        if features.url_count > 5:
            explanations.append("Contains many URLs")
        
        if features.repetition_ratio > 0.5:
            explanations.append("High word repetition")
        
        if features.domain_authority < 0.3:
            explanations.append("Low domain authority")
        
        if not explanations:
            explanations.append("Standard content analysis")
        
        if prediction:
            return f"Classified as spam: {'; '.join(explanations)}"
        else:
            return f"Classified as legitimate: {'; '.join(explanations)}"
    
    def add_training_sample(self, content: str, is_spam: bool, metadata: Dict[str, Any] = None):
        """Add a training sample for online learning"""
        features = self._extract_features(content, metadata or {})
        self.training_data.append((features, is_spam))
    
    def _train_models(self):
        """Train all models with current training data"""
        if self.is_training or len(self.training_data) < 100:
            return
        
        with self.model_lock:
            self.is_training = True
        
        try:
            # Prepare training data
            features_list, labels = zip(*self.training_data)
            
            # Convert features to matrix
            X = np.array([[getattr(f, attr) for attr in SpamFeatures.__annotations__.keys()] 
                         for f in features_list])
            y = np.array(labels)
            
            # Train each model
            for model_name, model in self.models.items():
                if model_name == 'bert':
                    continue  # Skip BERT for now (requires more complex training)
                
                try:
                    # Scale features
                    X_scaled = self.scalers[model_name].fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                    logger.info(f"Trained {model_name} model with {len(X)} samples")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
            
        finally:
            with self.model_lock:
                self.is_training = False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        # This would return performance metrics from validation
        return {
            'total_samples': len(self.training_data),
            'models_trained': len([m for m in self.models.values() if hasattr(m, 'predict')]),
            'last_training': time.time()
        }
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'models': {name: model for name, model in self.models.items() if name != 'bert'},
            'scalers': self.scalers,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models.update(model_data['models'])
        self.scalers.update(model_data['scalers'])
        self.config.update(model_data['config'])

# Example usage
async def main():
    """Example usage of the ML spam classifier"""
    config = {
        'models': {
            'enable_random_forest': True,
            'enable_xgboost': True,
            'enable_lightgbm': True,
            'enable_svm': True,
            'enable_naive_bayes': True,
            'enable_bert': False  # Disable BERT for demo
        },
        'min_training_samples': 100,
        'training_interval': 3600
    }
    
    classifier = MLSpamClassifier(config)
    
    # Test samples
    test_samples = [
        ("Buy Viagra now! Limited time offer! Click here!", True),
        ("This is a legitimate article about machine learning.", False),
        ("Congratulations! You've won $1000! Click to claim!", True),
        ("The weather is nice today.", False),
        ("Free money! No risk! Guaranteed income!", True)
    ]
    
    print("Testing spam classifier...")
    for content, expected in test_samples:
        result = await classifier.classify(content)
        print(f"\nContent: {content}")
        print(f"Predicted: {'SPAM' if result.is_spam else 'HAM'}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Explanation: {result.explanation}")
        print(f"Expected: {'SPAM' if expected else 'HAM'}")
        print(f"Correct: {result.is_spam == expected}")

if __name__ == "__main__":
    asyncio.run(main())
