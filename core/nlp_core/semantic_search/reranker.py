# T3SS Project
# File: core/nlp_core/semantic_search/reranker.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

"""
Advanced Neural Reranking System

This module implements a sophisticated neural reranking system that can improve
search result quality by reordering candidates based on learned relevance scores.
It supports multiple reranking models, ensemble methods, and advanced features
like query-document interaction modeling, cross-encoder architectures, and
real-time adaptation.

Key Features:
- Multiple reranking models (BERT, RoBERTa, DeBERTa, etc.)
- Ensemble reranking with multiple models
- Query-document interaction modeling
- Cross-encoder architectures
- Real-time model adaptation
- Advanced feature engineering
- Performance optimization
- Caching and persistence
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
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
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

class RerankingModelType(Enum):
    """Supported reranking model types"""
    BERT = "bert"
    ROBERTA = "roberta"
    DEBERTA = "deberta"
    CROSS_ENCODER = "cross_encoder"
    BI_ENCODER = "bi_encoder"
    CUSTOM = "custom"

class RerankingStrategy(Enum):
    """Reranking strategies"""
    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"
    CASCADE = "cascade"
    ADAPTIVE = "adaptive"

class FeatureType(Enum):
    """Feature types for reranking"""
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"
    USER_BEHAVIOR = "user_behavior"

@dataclass
class RerankingConfig:
    """Configuration for the reranking system"""
    # Model configuration
    model_type: RerankingModelType = RerankingModelType.BERT
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    num_workers: int = 4
    
    # Reranking configuration
    strategy: RerankingStrategy = RerankingStrategy.SINGLE_MODEL
    top_k: int = 100
    rerank_k: int = 20
    ensemble_models: List[str] = field(default_factory=list)
    ensemble_weights: List[float] = field(default_factory=list)
    
    # Feature configuration
    enable_semantic_features: bool = True
    enable_lexical_features: bool = True
    enable_statistical_features: bool = True
    enable_contextual_features: bool = True
    enable_temporal_features: bool = False
    enable_user_behavior_features: bool = False
    
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
    eval_metrics: List[str] = field(default_factory=lambda: ["ndcg", "precision", "recall", "f1"])
    eval_split: float = 0.2
    early_stopping_patience: int = 3
    
    # Persistence configuration
    model_save_path: str = "models/reranker"
    cache_save_path: str = "cache/reranker"
    log_save_path: str = "logs/reranker"
    
    # Advanced configuration
    enable_attention_visualization: bool = False
    enable_gradient_accumulation: bool = False
    enable_mixed_precision: bool = False
    enable_dynamic_padding: bool = True
    enable_sequence_optimization: bool = True

@dataclass
class RerankingRequest:
    """Request for reranking"""
    query: str
    documents: List[Dict[str, Any]]
    query_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RerankingResult:
    """Result of reranking"""
    query: str
    reranked_documents: List[Dict[str, Any]]
    scores: List[float]
    model_scores: Optional[Dict[str, List[float]]] = None
    features: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    model_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RerankingStats:
    """Statistics for reranking system"""
    total_requests: int = 0
    total_documents_processed: int = 0
    average_processing_time: float = 0.0
    average_reranking_time: float = 0.0
    cache_hit_rate: float = 0.0
    model_accuracy: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0

class RerankingDataset(Dataset):
    """Dataset for reranking training"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query and document
        query = item['query']
        document = item['document']
        
        # Create input for cross-encoder
        input_text = f"{query} [SEP] {document}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.float),
            'query_id': item.get('query_id', ''),
            'document_id': item.get('document_id', ''),
            'relevance_score': item.get('relevance_score', 0.0)
        }

class CrossEncoderReranker(nn.Module):
    """Cross-encoder model for reranking"""
    
    def __init__(self, model_name: str, num_labels: int = 1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

class BiEncoderReranker(nn.Module):
    """Bi-encoder model for reranking"""
    
    def __init__(self, model_name: str, embedding_dim: int = 768):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(self.config.hidden_size, embedding_dim)
        
    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        # Encode query
        query_outputs = self.bert(input_ids=query_ids, attention_mask=query_mask)
        query_embeddings = self.projection(query_outputs.pooler_output)
        
        # Encode document
        doc_outputs = self.bert(input_ids=doc_ids, attention_mask=doc_mask)
        doc_embeddings = self.projection(doc_outputs.pooler_output)
        
        # Compute similarity
        similarity = F.cosine_similarity(query_embeddings, doc_embeddings, dim=1)
        
        return {
            'query_embeddings': query_embeddings,
            'doc_embeddings': doc_embeddings,
            'similarity': similarity
        }

class FeatureExtractor:
    """Feature extractor for reranking"""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_features(self, query: str, document: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract features for reranking"""
        features = {}
        
        if self.config.enable_semantic_features:
            features.update(self._extract_semantic_features(query, document))
        
        if self.config.enable_lexical_features:
            features.update(self._extract_lexical_features(query, document))
        
        if self.config.enable_statistical_features:
            features.update(self._extract_statistical_features(query, document))
        
        if self.config.enable_contextual_features:
            features.update(self._extract_contextual_features(query, document, context))
        
        if self.config.enable_temporal_features:
            features.update(self._extract_temporal_features(query, document, context))
        
        if self.config.enable_user_behavior_features:
            features.update(self._extract_user_behavior_features(query, document, context))
        
        return features
    
    def _extract_semantic_features(self, query: str, document: str) -> Dict[str, float]:
        """Extract semantic features"""
        features = {}
        
        # Basic text similarity
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if len(query_words) > 0 and len(doc_words) > 0:
            features['jaccard_similarity'] = len(query_words & doc_words) / len(query_words | doc_words)
            features['overlap_ratio'] = len(query_words & doc_words) / len(query_words)
        else:
            features['jaccard_similarity'] = 0.0
            features['overlap_ratio'] = 0.0
        
        # Length features
        features['query_length'] = len(query.split())
        features['document_length'] = len(document.split())
        features['length_ratio'] = features['query_length'] / max(features['document_length'], 1)
        
        return features
    
    def _extract_lexical_features(self, query: str, document: str) -> Dict[str, float]:
        """Extract lexical features"""
        features = {}
        
        # Character-level features
        features['char_overlap'] = len(set(query.lower()) & set(document.lower())) / max(len(set(query.lower()) | set(document.lower())), 1)
        
        # N-gram features
        query_bigrams = set(zip(query.lower().split()[:-1], query.lower().split()[1:]))
        doc_bigrams = set(zip(document.lower().split()[:-1], document.lower().split()[1:]))
        
        if len(query_bigrams) > 0 and len(doc_bigrams) > 0:
            features['bigram_overlap'] = len(query_bigrams & doc_bigrams) / len(query_bigrams | doc_bigrams)
        else:
            features['bigram_overlap'] = 0.0
        
        return features
    
    def _extract_statistical_features(self, query: str, document: str) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        # Term frequency features
        query_tf = {}
        doc_tf = {}
        
        for word in query.lower().split():
            query_tf[word] = query_tf.get(word, 0) + 1
        
        for word in document.lower().split():
            doc_tf[word] = doc_tf.get(word, 0) + 1
        
        # TF-IDF-like features
        common_terms = set(query_tf.keys()) & set(doc_tf.keys())
        if common_terms:
            features['common_term_ratio'] = len(common_terms) / len(set(query_tf.keys()) | set(doc_tf.keys()))
            features['avg_term_frequency'] = np.mean([query_tf[term] + doc_tf[term] for term in common_terms])
        else:
            features['common_term_ratio'] = 0.0
            features['avg_term_frequency'] = 0.0
        
        return features
    
    def _extract_contextual_features(self, query: str, document: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract contextual features"""
        features = {}
        
        if context:
            # User context features
            features['user_query_history_length'] = context.get('query_history_length', 0)
            features['user_click_history_length'] = context.get('click_history_length', 0)
            features['user_session_duration'] = context.get('session_duration', 0)
            
            # Query context features
            features['query_position_in_session'] = context.get('query_position', 0)
            features['time_since_last_query'] = context.get('time_since_last_query', 0)
        else:
            features['user_query_history_length'] = 0
            features['user_click_history_length'] = 0
            features['user_session_duration'] = 0
            features['query_position_in_session'] = 0
            features['time_since_last_query'] = 0
        
        return features
    
    def _extract_temporal_features(self, query: str, document: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract temporal features"""
        features = {}
        
        if context:
            # Time-based features
            features['hour_of_day'] = context.get('hour_of_day', 12)
            features['day_of_week'] = context.get('day_of_week', 1)
            features['month'] = context.get('month', 1)
            
            # Document age features
            features['document_age_days'] = context.get('document_age_days', 0)
            features['document_freshness'] = context.get('document_freshness', 0.5)
        else:
            features['hour_of_day'] = 12
            features['day_of_week'] = 1
            features['month'] = 1
            features['document_age_days'] = 0
            features['document_freshness'] = 0.5
        
        return features
    
    def _extract_user_behavior_features(self, query: str, document: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract user behavior features"""
        features = {}
        
        if context:
            # Click behavior features
            features['user_click_rate'] = context.get('click_rate', 0.0)
            features['user_dwell_time'] = context.get('avg_dwell_time', 0.0)
            features['user_bounce_rate'] = context.get('bounce_rate', 0.0)
            
            # Query behavior features
            features['user_query_frequency'] = context.get('query_frequency', 0.0)
            features['user_query_diversity'] = context.get('query_diversity', 0.0)
        else:
            features['user_click_rate'] = 0.0
            features['user_dwell_time'] = 0.0
            features['user_bounce_rate'] = 0.0
            features['user_query_frequency'] = 0.0
            features['user_query_diversity'] = 0.0
        
        return features

class RerankingSystem:
    """Advanced neural reranking system"""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.feature_extractor = FeatureExtractor(config)
        self.cache = {}
        self.stats = RerankingStats()
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
        """Initialize reranking models"""
        try:
            if self.config.strategy == RerankingStrategy.SINGLE_MODEL:
                self._load_single_model()
            elif self.config.strategy == RerankingStrategy.ENSEMBLE:
                self._load_ensemble_models()
            elif self.config.strategy == RerankingStrategy.CASCADE:
                self._load_cascade_models()
            elif self.config.strategy == RerankingStrategy.ADAPTIVE:
                self._load_adaptive_models()
            
            logger.info(f"Initialized {len(self.models)} reranking models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_single_model(self):
        """Load a single reranking model"""
        model_name = self.config.model_name
        
        if self.config.model_type == RerankingModelType.CROSS_ENCODER:
            self.models['main'] = CrossEncoderReranker(model_name)
        elif self.config.model_type == RerankingModelType.BI_ENCODER:
            self.models['main'] = BiEncoderReranker(model_name)
        else:
            # Use pre-trained model
            self.models['main'] = AutoModel.from_pretrained(model_name)
        
        self.tokenizers['main'] = AutoTokenizer.from_pretrained(model_name)
        
        # Move to GPU if available
        if self.config.enable_gpu and torch.cuda.is_available():
            self.models['main'] = self.models['main'].cuda()
    
    def _load_ensemble_models(self):
        """Load ensemble of reranking models"""
        model_names = self.config.ensemble_models or [self.config.model_name]
        
        for i, model_name in enumerate(model_names):
            try:
                if self.config.model_type == RerankingModelType.CROSS_ENCODER:
                    self.models[f'model_{i}'] = CrossEncoderReranker(model_name)
                elif self.config.model_type == RerankingModelType.BI_ENCODER:
                    self.models[f'model_{i}'] = BiEncoderReranker(model_name)
                else:
                    self.models[f'model_{i}'] = AutoModel.from_pretrained(model_name)
                
                self.tokenizers[f'model_{i}'] = AutoTokenizer.from_pretrained(model_name)
                
                # Move to GPU if available
                if self.config.enable_gpu and torch.cuda.is_available():
                    self.models[f'model_{i}'] = self.models[f'model_{i}'].cuda()
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                continue
    
    def _load_cascade_models(self):
        """Load cascade of reranking models"""
        # First stage: fast model
        self.models['fast'] = AutoModel.from_pretrained('distilbert-base-uncased')
        self.tokenizers['fast'] = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Second stage: accurate model
        self.models['accurate'] = AutoModel.from_pretrained(self.config.model_name)
        self.tokenizers['accurate'] = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Move to GPU if available
        if self.config.enable_gpu and torch.cuda.is_available():
            self.models['fast'] = self.models['fast'].cuda()
            self.models['accurate'] = self.models['accurate'].cuda()
    
    def _load_adaptive_models(self):
        """Load adaptive reranking models"""
        # Load multiple models for different scenarios
        self.models['general'] = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizers['general'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.models['domain_specific'] = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizers['domain_specific'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Move to GPU if available
        if self.config.enable_gpu and torch.cuda.is_available():
            self.models['general'] = self.models['general'].cuda()
            self.models['domain_specific'] = self.models['domain_specific'].cuda()
    
    async def rerank(self, request: RerankingRequest) -> RerankingResult:
        """Rerank documents for a query"""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cached_result = await self._get_from_cache(request)
                if cached_result:
                    return cached_result
            
            # Extract features
            features = self._extract_features(request)
            
            # Rerank documents
            if self.config.strategy == RerankingStrategy.SINGLE_MODEL:
                result = await self._rerank_single_model(request, features)
            elif self.config.strategy == RerankingStrategy.ENSEMBLE:
                result = await self._rerank_ensemble(request, features)
            elif self.config.strategy == RerankingStrategy.CASCADE:
                result = await self._rerank_cascade(request, features)
            elif self.config.strategy == RerankingStrategy.ADAPTIVE:
                result = await self._rerank_adaptive(request, features)
            else:
                raise ValueError(f"Unknown reranking strategy: {self.config.strategy}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(request.documents))
            
            # Cache result
            if self.config.enable_caching:
                await self._set_cache(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise
    
    async def _rerank_single_model(self, request: RerankingRequest, features: Dict[str, Any]) -> RerankingResult:
        """Rerank using a single model"""
        model = self.models['main']
        tokenizer = self.tokenizers['main']
        
        # Prepare inputs
        inputs = []
        for doc in request.documents:
            input_text = f"{request.query} [SEP] {doc.get('text', '')}"
            encoding = tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            inputs.append(encoding)
        
        # Batch process
        batch_size = self.config.batch_size
        scores = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Prepare batch tensors
            input_ids = torch.cat([inp['input_ids'] for inp in batch])
            attention_mask = torch.cat([inp['attention_mask'] for inp in batch])
            
            # Move to GPU if available
            if self.config.enable_gpu and torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            # Get model predictions
            with torch.no_grad():
                if self.config.model_type == RerankingModelType.CROSS_ENCODER:
                    outputs = model(input_ids, attention_mask)
                    batch_scores = outputs['logits'].squeeze().cpu().numpy()
                else:
                    outputs = model(input_ids, attention_mask)
                    batch_scores = outputs.pooler_output.mean(dim=1).cpu().numpy()
            
            scores.extend(batch_scores.tolist())
        
        # Sort documents by scores
        doc_scores = list(zip(request.documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_scores]
        reranked_scores = [score for doc, score in doc_scores]
        
        return RerankingResult(
            query=request.query,
            reranked_documents=reranked_docs,
            scores=reranked_scores,
            features=features,
            processing_time=time.time() - time.time(),
            model_info={'model_type': self.config.model_type.value}
        )
    
    async def _rerank_ensemble(self, request: RerankingRequest, features: Dict[str, Any]) -> RerankingResult:
        """Rerank using ensemble of models"""
        model_scores = {}
        all_scores = []
        
        for model_name, model in self.models.items():
            tokenizer = self.tokenizers[model_name]
            
            # Get scores from this model
            scores = await self._get_model_scores(request, model, tokenizer)
            model_scores[model_name] = scores
            all_scores.append(scores)
        
        # Combine scores
        if self.config.ensemble_weights:
            weights = self.config.ensemble_weights
        else:
            weights = [1.0 / len(all_scores)] * len(all_scores)
        
        combined_scores = np.average(all_scores, axis=0, weights=weights)
        
        # Sort documents by combined scores
        doc_scores = list(zip(request.documents, combined_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_scores]
        reranked_scores = [score for doc, score in doc_scores]
        
        return RerankingResult(
            query=request.query,
            reranked_documents=reranked_docs,
            scores=reranked_scores,
            model_scores=model_scores,
            features=features,
            processing_time=time.time() - time.time()
        )
    
    async def _rerank_cascade(self, request: RerankingRequest, features: Dict[str, Any]) -> RerankingResult:
        """Rerank using cascade of models"""
        # First stage: fast model
        fast_model = self.models['fast']
        fast_tokenizer = self.tokenizers['fast']
        
        fast_scores = await self._get_model_scores(request, fast_model, fast_tokenizer)
        
        # Select top documents for second stage
        doc_scores = list(zip(request.documents, fast_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in doc_scores[:self.config.top_k]]
        
        # Second stage: accurate model
        accurate_model = self.models['accurate']
        accurate_tokenizer = self.tokenizers['accurate']
        
        # Create new request with top documents
        top_request = RerankingRequest(
            query=request.query,
            documents=top_docs,
            query_id=request.query_id,
            user_id=request.user_id,
            session_id=request.session_id,
            context=request.context,
            features=request.features,
            metadata=request.metadata
        )
        
        accurate_scores = await self._get_model_scores(top_request, accurate_model, accurate_tokenizer)
        
        # Combine results
        final_docs = top_docs
        final_scores = accurate_scores
        
        return RerankingResult(
            query=request.query,
            reranked_documents=final_docs,
            scores=final_scores,
            features=features,
            processing_time=time.time() - time.time()
        )
    
    async def _rerank_adaptive(self, request: RerankingRequest, features: Dict[str, Any]) -> RerankingResult:
        """Rerank using adaptive model selection"""
        # Select model based on query characteristics
        if self._is_domain_specific_query(request.query):
            model_name = 'domain_specific'
        else:
            model_name = 'general'
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        scores = await self._get_model_scores(request, model, tokenizer)
        
        # Sort documents by scores
        doc_scores = list(zip(request.documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_scores]
        reranked_scores = [score for doc, score in doc_scores]
        
        return RerankingResult(
            query=request.query,
            reranked_documents=reranked_docs,
            scores=reranked_scores,
            features=features,
            processing_time=time.time() - time.time(),
            model_info={'selected_model': model_name}
        )
    
    async def _get_model_scores(self, request: RerankingRequest, model, tokenizer) -> List[float]:
        """Get scores from a model"""
        inputs = []
        for doc in request.documents:
            input_text = f"{request.query} [SEP] {doc.get('text', '')}"
            encoding = tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            inputs.append(encoding)
        
        # Batch process
        batch_size = self.config.batch_size
        scores = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Prepare batch tensors
            input_ids = torch.cat([inp['input_ids'] for inp in batch])
            attention_mask = torch.cat([inp['attention_mask'] for inp in batch])
            
            # Move to GPU if available
            if self.config.enable_gpu and torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            # Get model predictions
            with torch.no_grad():
                if hasattr(model, 'forward') and 'logits' in model.forward(input_ids, attention_mask):
                    outputs = model(input_ids, attention_mask)
                    batch_scores = outputs['logits'].squeeze().cpu().numpy()
                else:
                    outputs = model(input_ids, attention_mask)
                    batch_scores = outputs.pooler_output.mean(dim=1).cpu().numpy()
            
            scores.extend(batch_scores.tolist())
        
        return scores
    
    def _extract_features(self, request: RerankingRequest) -> Dict[str, Any]:
        """Extract features for reranking"""
        features = {}
        
        for i, doc in enumerate(request.documents):
            doc_features = self.feature_extractor.extract_features(
                request.query,
                doc.get('text', ''),
                request.context
            )
            features[f'doc_{i}'] = doc_features
        
        return features
    
    def _is_domain_specific_query(self, query: str) -> bool:
        """Check if query is domain-specific"""
        # Simple heuristic - can be improved with ML
        domain_keywords = ['technical', 'medical', 'legal', 'scientific', 'academic']
        return any(keyword in query.lower() for keyword in domain_keywords)
    
    async def _get_from_cache(self, request: RerankingRequest) -> Optional[RerankingResult]:
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
    
    async def _set_cache(self, request: RerankingRequest, result: RerankingResult):
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
    
    def _generate_cache_key(self, request: RerankingRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'query': request.query,
            'documents': [doc.get('id', '') for doc in request.documents],
            'user_id': request.user_id,
            'session_id': request.session_id
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_stats(self, processing_time: float, num_documents: int):
        """Update statistics"""
        self.stats.total_requests += 1
        self.stats.total_documents_processed += num_documents
        
        # Update average processing time
        if self.stats.total_requests == 1:
            self.stats.average_processing_time = processing_time
        else:
            self.stats.average_processing_time = (
                (self.stats.average_processing_time * (self.stats.total_requests - 1) + processing_time) /
                self.stats.total_requests
            )
        
        # Update throughput
        self.stats.throughput = self.stats.total_documents_processed / max(self.stats.average_processing_time, 1e-6)
    
    def get_stats(self) -> RerankingStats:
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
    config = RerankingConfig(
        model_type=RerankingModelType.BERT,
        model_name="bert-base-uncased",
        strategy=RerankingStrategy.SINGLE_MODEL,
        enable_caching=True,
        enable_gpu=True
    )
    
    # Initialize reranking system
    reranker = RerankingSystem(config)
    
    # Create sample request
    request = RerankingRequest(
        query="machine learning algorithms",
        documents=[
            {"id": "1", "text": "Introduction to machine learning and neural networks"},
            {"id": "2", "text": "Deep learning for computer vision applications"},
            {"id": "3", "text": "Statistical methods for data analysis"},
            {"id": "4", "text": "Natural language processing with transformers"}
        ]
    )
    
    # Rerank documents
    result = asyncio.run(reranker.rerank(request))
    
    print(f"Query: {result.query}")
    print(f"Reranked documents: {len(result.reranked_documents)}")
    print(f"Scores: {result.scores}")
    print(f"Processing time: {result.processing_time:.4f}s")
    
    # Print statistics
    stats = reranker.get_stats()
    print(f"Total requests: {stats.total_requests}")
    print(f"Average processing time: {stats.average_processing_time:.4f}s")
    print(f"Throughput: {stats.throughput:.2f} docs/s")
