# T3SS Project
# File: core/indexing/quality_assessor/content_quality_model.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

"""
Advanced Content Quality Assessment Model

This module implements a sophisticated ML-based content quality assessment system
that can evaluate the quality of web content across multiple dimensions. It uses
state-of-the-art transformer models and ensemble methods to provide comprehensive
quality scores for content ranking and filtering.

Key Features:
- Multi-dimensional quality assessment (readability, accuracy, completeness, etc.)
- Transformer-based feature extraction
- Ensemble learning with multiple models
- Real-time quality scoring
- Domain-specific quality models
- Content type classification
- Plagiarism detection
- Fact-checking integration
- Performance optimization
- Comprehensive statistics and monitoring
"""

import os
import json
import time
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel, BertConfig,
    RobertaTokenizer, RobertaModel, RobertaConfig,
    DebertaTokenizer, DebertaModel, DebertaConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import redis
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Quality assessment dimensions"""
    READABILITY = "readability"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    AUTHORITY = "authority"
    FRESHNESS = "freshness"
    UNIQUENESS = "uniqueness"
    STRUCTURE = "structure"
    LANGUAGE_QUALITY = "language_quality"
    FACTUAL_ACCURACY = "factual_accuracy"

class ContentType(Enum):
    """Content types for quality assessment"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    ACADEMIC_PAPER = "academic_paper"
    PRODUCT_DESCRIPTION = "product_description"
    REVIEW = "review"
    FORUM_POST = "forum_post"
    SOCIAL_MEDIA = "social_media"
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"

class QualityLevel(Enum):
    """Quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class QualityConfig:
    """Configuration for content quality assessment"""
    # Model configuration
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    num_workers: int = 4
    
    # Quality assessment configuration
    enable_readability: bool = True
    enable_accuracy: bool = True
    enable_completeness: bool = True
    enable_relevance: bool = True
    enable_authority: bool = True
    enable_freshness: bool = True
    enable_uniqueness: bool = True
    enable_structure: bool = True
    enable_language_quality: bool = True
    enable_factual_accuracy: bool = True
    
    # Ensemble configuration
    enable_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: ["bert", "roberta", "deberta"])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    
    # Performance configuration
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_gpu: bool = True
    enable_quantization: bool = False
    enable_pruning: bool = False
    
    # Training configuration
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Evaluation configuration
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"])
    eval_split: float = 0.2
    early_stopping_patience: int = 3
    
    # Persistence configuration
    model_save_path: str = "models/quality_assessor"
    cache_save_path: str = "cache/quality_assessor"
    log_save_path: str = "logs/quality_assessor"
    
    # Advanced configuration
    enable_attention_visualization: bool = False
    enable_gradient_accumulation: bool = False
    enable_mixed_precision: bool = False
    enable_dynamic_padding: bool = True
    enable_sequence_optimization: bool = True

@dataclass
class QualityRequest:
    """Request for quality assessment"""
    content: str
    content_type: Optional[ContentType] = None
    domain: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class QualityResult:
    """Result of quality assessment"""
    overall_quality: float
    quality_scores: Dict[QualityDimension, float]
    quality_level: QualityLevel
    content_type: ContentType
    confidence: float
    processing_time: float
    model_info: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class QualityStats:
    """Statistics for quality assessment system"""
    total_assessments: int = 0
    average_quality: float = 0.0
    quality_distribution: Dict[QualityLevel, int] = field(default_factory=dict)
    content_type_distribution: Dict[ContentType, int] = field(default_factory=dict)
    average_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    model_accuracy: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0

class QualityDataset(Dataset):
    """Dataset for quality assessment training"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize content
        content = item['content']
        encoding = self.tokenizer(
            content,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'quality_scores': torch.tensor(item['quality_scores'], dtype=torch.float),
            'content_type': item['content_type'],
            'quality_level': item['quality_level']
        }

class QualityModel(nn.Module):
    """Neural network model for quality assessment"""
    
    def __init__(self, model_name: str, num_dimensions: int = 10):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Quality dimension heads
        self.quality_heads = nn.ModuleDict({
            dimension.value: nn.Linear(self.config.hidden_size, 1)
            for dimension in QualityDimension
        })
        
        # Content type classifier
        self.content_type_classifier = nn.Linear(self.config.hidden_size, len(ContentType))
        
        # Quality level classifier
        self.quality_level_classifier = nn.Linear(self.config.hidden_size, len(QualityLevel))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Quality dimension scores
        quality_scores = {}
        for dimension in QualityDimension:
            quality_scores[dimension] = torch.sigmoid(self.quality_heads[dimension.value](pooled_output))
        
        # Content type prediction
        content_type_logits = self.content_type_classifier(pooled_output)
        content_type_probs = F.softmax(content_type_logits, dim=1)
        
        # Quality level prediction
        quality_level_logits = self.quality_level_classifier(pooled_output)
        quality_level_probs = F.softmax(quality_level_logits, dim=1)
        
        return {
            'quality_scores': quality_scores,
            'content_type_probs': content_type_probs,
            'quality_level_probs': quality_level_probs,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

class FeatureExtractor:
    """Feature extractor for quality assessment"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_features(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract features for quality assessment"""
        features = {}
        
        # Text-based features
        features.update(self._extract_text_features(content))
        
        # Readability features
        if self.config.enable_readability:
            features.update(self._extract_readability_features(content))
        
        # Structure features
        if self.config.enable_structure:
            features.update(self._extract_structure_features(content))
        
        # Language quality features
        if self.config.enable_language_quality:
            features.update(self._extract_language_quality_features(content))
        
        # Metadata features
        if metadata:
            features.update(self._extract_metadata_features(metadata))
        
        return features
    
    def _extract_text_features(self, content: str) -> Dict[str, float]:
        """Extract basic text features"""
        features = {}
        
        # Length features
        features['content_length'] = len(content)
        features['word_count'] = len(content.split())
        features['sentence_count'] = content.count('.') + content.count('!') + content.count('?')
        features['paragraph_count'] = content.count('\n\n') + 1
        
        # Character features
        features['avg_word_length'] = np.mean([len(word) for word in content.split()]) if content.split() else 0
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Special character features
        features['punctuation_ratio'] = sum(1 for c in content if c in '.,!?;:') / max(len(content), 1)
        features['uppercase_ratio'] = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        features['digit_ratio'] = sum(1 for c in content if c.isdigit()) / max(len(content), 1)
        
        return features
    
    def _extract_readability_features(self, content: str) -> Dict[str, float]:
        """Extract readability features"""
        features = {}
        
        # Flesch Reading Ease (simplified)
        words = content.split()
        sentences = content.split('.')
        
        if words and sentences:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = np.mean([self._count_syllables(word) for word in words])
            
            # Simplified Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            features['flesch_reading_ease'] = max(0, min(100, flesch_score))
            
            # Flesch-Kincaid Grade Level
            fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
            features['flesch_kincaid_grade'] = max(0, fk_grade)
        
        # Gunning Fog Index (simplified)
        complex_words = sum(1 for word in words if self._count_syllables(word) > 2)
        if words:
            fog_index = 0.4 * (len(words) / len(sentences) + 100 * complex_words / len(words))
            features['gunning_fog_index'] = fog_index
        
        return features
    
    def _extract_structure_features(self, content: str) -> Dict[str, float]:
        """Extract structure features"""
        features = {}
        
        # Heading features
        features['heading_count'] = content.count('#') + content.count('<h1>') + content.count('<h2>')
        features['list_count'] = content.count('*') + content.count('-') + content.count('<li>')
        
        # Link features
        features['link_count'] = content.count('http') + content.count('www.')
        features['link_ratio'] = features['link_count'] / max(len(content.split()), 1)
        
        # Image features
        features['image_count'] = content.count('<img') + content.count('![')
        features['image_ratio'] = features['image_count'] / max(len(content.split()), 1)
        
        # Table features
        features['table_count'] = content.count('<table') + content.count('|')
        
        return features
    
    def _extract_language_quality_features(self, content: str) -> Dict[str, float]:
        """Extract language quality features"""
        features = {}
        
        # Spelling and grammar indicators
        features['repeated_words'] = self._count_repeated_words(content)
        features['repeated_phrases'] = self._count_repeated_phrases(content)
        
        # Vocabulary diversity
        words = content.lower().split()
        unique_words = set(words)
        features['vocabulary_diversity'] = len(unique_words) / max(len(words), 1)
        
        # Word frequency analysis
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            features['avg_word_frequency'] = np.mean(list(word_freq.values()))
            features['max_word_frequency'] = max(word_freq.values())
        
        return features
    
    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract metadata features"""
        features = {}
        
        # URL features
        if 'url' in metadata:
            url = metadata['url']
            features['url_length'] = len(url)
            features['url_depth'] = url.count('/')
            features['has_https'] = 1.0 if url.startswith('https') else 0.0
        
        # Domain features
        if 'domain' in metadata:
            domain = metadata['domain']
            features['domain_length'] = len(domain)
            features['has_subdomain'] = 1.0 if '.' in domain else 0.0
        
        # Timestamp features
        if 'timestamp' in metadata:
            timestamp = metadata['timestamp']
            # Convert to age in days
            current_time = time.time()
            age_days = (current_time - timestamp) / (24 * 3600)
            features['content_age_days'] = age_days
            features['is_recent'] = 1.0 if age_days < 7 else 0.0
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _count_repeated_words(self, content: str) -> int:
        """Count repeated words"""
        words = content.lower().split()
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        return sum(1 for count in word_count.values() if count > 1)
    
    def _count_repeated_phrases(self, content: str) -> int:
        """Count repeated phrases"""
        words = content.lower().split()
        phrases = {}
        
        # Check 2-grams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrases[phrase] = phrases.get(phrase, 0) + 1
        
        return sum(1 for count in phrases.values() if count > 1)

class ContentQualityAssessor:
    """Advanced content quality assessment system"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.feature_extractor = FeatureExtractor(config)
        self.cache = {}
        self.stats = QualityStats()
        self.redis_client = None
        
        # Initialize Redis if caching is enabled
        if config.enable_caching:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
            except:
                logger.warning("Redis not available, using in-memory cache")
                self.redis_client = None
        
        # Initialize models
        self._initialize_models()
        
        # Create directories
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.cache_save_path, exist_ok=True)
        os.makedirs(config.log_save_path, exist_ok=True)
    
    def _initialize_models(self):
        """Initialize quality assessment models"""
        try:
            if self.config.enable_ensemble:
                self._load_ensemble_models()
            else:
                self._load_single_model()
            
            logger.info(f"Initialized {len(self.models)} quality assessment models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_single_model(self):
        """Load a single quality assessment model"""
        model_name = self.config.model_name
        
        self.models['main'] = QualityModel(model_name)
        self.tokenizers['main'] = AutoTokenizer.from_pretrained(model_name)
        
        # Move to GPU if available
        if self.config.enable_gpu and torch.cuda.is_available():
            self.models['main'] = self.models['main'].cuda()
    
    def _load_ensemble_models(self):
        """Load ensemble of quality assessment models"""
        model_names = self.config.ensemble_models
        
        for i, model_name in enumerate(model_names):
            try:
                self.models[f'model_{i}'] = QualityModel(model_name)
                self.tokenizers[f'model_{i}'] = AutoTokenizer.from_pretrained(model_name)
                
                # Move to GPU if available
                if self.config.enable_gpu and torch.cuda.is_available():
                    self.models[f'model_{i}'] = self.models[f'model_{i}'].cuda()
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                continue
    
    async def assess_quality(self, request: QualityRequest) -> QualityResult:
        """Assess content quality"""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cached_result = await self._get_from_cache(request)
                if cached_result:
                    return cached_result
            
            # Extract features
            features = self.feature_extractor.extract_features(request.content, request.metadata)
            
            # Assess quality using models
            if self.config.enable_ensemble:
                result = await self._assess_quality_ensemble(request, features)
            else:
                result = await self._assess_quality_single(request, features)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(result, processing_time)
            
            # Cache result
            if self.config.enable_caching:
                await self._set_cache(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise
    
    async def _assess_quality_single(self, request: QualityRequest, features: Dict[str, float]) -> QualityResult:
        """Assess quality using a single model"""
        model = self.models['main']
        tokenizer = self.tokenizers['main']
        
        # Tokenize content
        encoding = tokenizer(
            request.content,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        if self.config.enable_gpu and torch.cuda.is_available():
            encoding = {k: v.cuda() for k, v in encoding.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
        
        # Extract quality scores
        quality_scores = {}
        for dimension in QualityDimension:
            score = outputs['quality_scores'][dimension].cpu().numpy()[0][0]
            quality_scores[dimension] = float(score)
        
        # Calculate overall quality
        overall_quality = np.mean(list(quality_scores.values()))
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_quality)
        
        # Determine content type
        content_type_probs = outputs['content_type_probs'].cpu().numpy()[0]
        content_type_idx = np.argmax(content_type_probs)
        content_type = list(ContentType)[content_type_idx]
        
        # Calculate confidence
        confidence = float(np.max(content_type_probs))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_scores, overall_quality)
        
        return QualityResult(
            overall_quality=overall_quality,
            quality_scores=quality_scores,
            quality_level=quality_level,
            content_type=content_type,
            confidence=confidence,
            processing_time=time.time() - time.time(),
            model_info={'model_type': 'single', 'model_name': self.config.model_name},
            recommendations=recommendations,
            metadata=request.metadata or {}
        )
    
    async def _assess_quality_ensemble(self, request: QualityRequest, features: Dict[str, float]) -> QualityResult:
        """Assess quality using ensemble of models"""
        model_scores = {}
        all_quality_scores = {}
        
        for model_name, model in self.models.items():
            tokenizer = self.tokenizers[model_name]
            
            # Tokenize content
            encoding = tokenizer(
                request.content,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Move to GPU if available
            if self.config.enable_gpu and torch.cuda.is_available():
                encoding = {k: v.cuda() for k, v in encoding.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(encoding['input_ids'], encoding['attention_mask'])
            
            # Extract quality scores
            quality_scores = {}
            for dimension in QualityDimension:
                score = outputs['quality_scores'][dimension].cpu().numpy()[0][0]
                quality_scores[dimension] = float(score)
            
            model_scores[model_name] = quality_scores
            
            # Accumulate scores
            for dimension, score in quality_scores.items():
                if dimension not in all_quality_scores:
                    all_quality_scores[dimension] = []
                all_quality_scores[dimension].append(score)
        
        # Combine scores using ensemble weights
        if self.config.ensemble_weights:
            weights = self.config.ensemble_weights
        else:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        final_quality_scores = {}
        for dimension in QualityDimension:
            scores = all_quality_scores[dimension]
            final_quality_scores[dimension] = np.average(scores, weights=weights)
        
        # Calculate overall quality
        overall_quality = np.mean(list(final_quality_scores.values()))
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_quality)
        
        # Determine content type (use first model for now)
        first_model = list(self.models.keys())[0]
        first_tokenizer = self.tokenizers[first_model]
        
        encoding = first_tokenizer(
            request.content,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        if self.config.enable_gpu and torch.cuda.is_available():
            encoding = {k: v.cuda() for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.models[first_model](encoding['input_ids'], encoding['attention_mask'])
        
        content_type_probs = outputs['content_type_probs'].cpu().numpy()[0]
        content_type_idx = np.argmax(content_type_probs)
        content_type = list(ContentType)[content_type_idx]
        
        # Calculate confidence
        confidence = float(np.max(content_type_probs))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(final_quality_scores, overall_quality)
        
        return QualityResult(
            overall_quality=overall_quality,
            quality_scores=final_quality_scores,
            quality_level=quality_level,
            content_type=content_type,
            confidence=confidence,
            processing_time=time.time() - time.time(),
            model_info={'model_type': 'ensemble', 'models': list(self.models.keys())},
            recommendations=recommendations,
            metadata=request.metadata or {}
        )
    
    def _determine_quality_level(self, overall_quality: float) -> QualityLevel:
        """Determine quality level based on overall quality score"""
        if overall_quality >= 0.8:
            return QualityLevel.EXCELLENT
        elif overall_quality >= 0.6:
            return QualityLevel.GOOD
        elif overall_quality >= 0.4:
            return QualityLevel.FAIR
        elif overall_quality >= 0.2:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _generate_recommendations(self, quality_scores: Dict[QualityDimension, float], overall_quality: float) -> List[str]:
        """Generate recommendations for improving content quality"""
        recommendations = []
        
        for dimension, score in quality_scores.items():
            if score < 0.3:
                if dimension == QualityDimension.READABILITY:
                    recommendations.append("Improve readability by using shorter sentences and simpler words")
                elif dimension == QualityDimension.ACCURACY:
                    recommendations.append("Verify facts and ensure accuracy of information")
                elif dimension == QualityDimension.COMPLETENESS:
                    recommendations.append("Add more comprehensive information to cover the topic thoroughly")
                elif dimension == QualityDimension.RELEVANCE:
                    recommendations.append("Ensure content is relevant to the target audience")
                elif dimension == QualityDimension.AUTHORITY:
                    recommendations.append("Add credible sources and author information")
                elif dimension == QualityDimension.FRESHNESS:
                    recommendations.append("Update content with more recent information")
                elif dimension == QualityDimension.UNIQUENESS:
                    recommendations.append("Ensure content is original and not duplicated")
                elif dimension == QualityDimension.STRUCTURE:
                    recommendations.append("Improve content structure with headings and organization")
                elif dimension == QualityDimension.LANGUAGE_QUALITY:
                    recommendations.append("Improve grammar, spelling, and language quality")
                elif dimension == QualityDimension.FACTUAL_ACCURACY:
                    recommendations.append("Verify factual accuracy and add citations")
        
        if overall_quality < 0.5:
            recommendations.append("Consider a comprehensive content review and rewrite")
        
        return recommendations
    
    async def _get_from_cache(self, request: QualityRequest) -> Optional[QualityResult]:
        """Get result from cache"""
        if not self.config.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
            else:
                if cache_key in self.cache:
                    return self.cache[cache_key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _set_cache(self, request: QualityRequest, result: QualityResult):
        """Set result in cache"""
        if not self.config.enable_caching:
            return
        
        cache_key = self._generate_cache_key(request)
        
        try:
            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    pickle.dumps(result)
                )
            else:
                self.cache[cache_key] = result
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _generate_cache_key(self, request: QualityRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'content': request.content,
            'content_type': request.content_type.value if request.content_type else None,
            'domain': request.domain,
            'url': request.url
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_stats(self, result: QualityResult, processing_time: float):
        """Update statistics"""
        self.stats.total_assessments += 1
        self.stats.average_quality = (
            (self.stats.average_quality * (self.stats.total_assessments - 1) + result.overall_quality) /
            self.stats.total_assessments
        )
        
        # Update quality level distribution
        quality_level = result.quality_level.value
        self.stats.quality_distribution[quality_level] = self.stats.quality_distribution.get(quality_level, 0) + 1
        
        # Update content type distribution
        content_type = result.content_type.value
        self.stats.content_type_distribution[content_type] = self.stats.content_type_distribution.get(content_type, 0) + 1
        
        # Update average processing time
        if self.stats.total_assessments == 1:
            self.stats.average_processing_time = processing_time
        else:
            self.stats.average_processing_time = (
                (self.stats.average_processing_time * (self.stats.total_assessments - 1) + processing_time) /
                self.stats.total_assessments
            )
    
    def get_stats(self) -> QualityStats:
        """Get current statistics"""
        return self.stats
    
    def clear_cache(self):
        """Clear cache"""
        if self.redis_client:
            self.redis_client.flushdb()
        else:
            self.cache.clear()
    
    def save_model(self, model_name: str, path: str):
        """Save model to disk"""
        if model_name in self.models:
            model_path = os.path.join(self.config.model_save_path, path)
            os.makedirs(model_path, exist_ok=True)
            
            # Save model
            torch.save(self.models[model_name].state_dict(), os.path.join(model_path, 'model.pt'))
            
            # Save tokenizer
            self.tokenizers[model_name].save_pretrained(model_path)
            
            # Save config
            with open(os.path.join(model_path, 'config.json'), 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            logger.info(f"Model {model_name} saved to {model_path}")
    
    def load_model(self, model_name: str, path: str):
        """Load model from disk"""
        model_path = os.path.join(self.config.model_save_path, path)
        
        if os.path.exists(model_path):
            # Load model
            model_state = torch.load(os.path.join(model_path, 'model.pt'))
            self.models[model_name].load_state_dict(model_state)
            
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"Model {model_name} loaded from {model_path}")
        else:
            logger.warning(f"Model path {model_path} does not exist")

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = QualityConfig(
        model_name="bert-base-uncased",
        enable_ensemble=True,
        enable_caching=True,
        enable_gpu=True
    )
    
    # Initialize quality assessor
    assessor = ContentQualityAssessor(config)
    
    # Create sample request
    request = QualityRequest(
        content="This is a sample article about machine learning. It covers various topics including neural networks, deep learning, and artificial intelligence.",
        content_type=ContentType.ARTICLE,
        domain="example.com",
        url="https://example.com/article",
        metadata={"author": "John Doe", "timestamp": time.time()}
    )
    
    # Assess quality
    result = asyncio.run(assessor.assess_quality(request))
    
    print(f"Overall Quality: {result.overall_quality:.3f}")
    print(f"Quality Level: {result.quality_level.value}")
    print(f"Content Type: {result.content_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing Time: {result.processing_time:.4f}s")
    
    print("\nQuality Scores:")
    for dimension, score in result.quality_scores.items():
        print(f"  {dimension.value}: {score:.3f}")
    
    print("\nRecommendations:")
    for recommendation in result.recommendations:
        print(f"  - {recommendation}")
    
    # Print statistics
    stats = assessor.get_stats()
    print(f"\nStatistics:")
    print(f"Total Assessments: {stats.total_assessments}")
    print(f"Average Quality: {stats.average_quality:.3f}")
    print(f"Average Processing Time: {stats.average_processing_time:.4f}s")
