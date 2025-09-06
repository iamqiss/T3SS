# T3SS Project
# File: core/nlp_core/sentiment_analyzer.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import pickle
import hashlib
from datetime import datetime, timedelta
import redis
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import re
import string
from collections import defaultdict, Counter
import math
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class EmotionLabel(Enum):
    """Emotion classification labels"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    sentiment: SentimentLabel
    confidence: float
    emotions: Dict[EmotionLabel, float]
    polarity_score: float  # -1.0 to 1.0
    subjectivity_score: float  # 0.0 to 1.0
    processing_time: float
    model_used: str
    language: str
    metadata: Dict[str, Any]

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    # Model settings
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    use_ensemble: bool = True
    ensemble_models: List[str] = None
    
    # Performance settings
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = True
    num_workers: int = 4
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Text preprocessing
    enable_preprocessing: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    normalize_emojis: bool = True
    handle_negations: bool = True
    
    # Analysis settings
    enable_emotion_analysis: bool = True
    enable_subjectivity_analysis: bool = True
    enable_confidence_scoring: bool = True
    confidence_threshold: float = 0.7
    
    # Language detection
    enable_language_detection: bool = True
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "ProsusAI/finbert",
                "distilbert-base-uncased-finetuned-sst-2-english"
            ]
        
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|'
            r'[\U0001F1E0-\U0001F1FF]|[\U0002600-\U000026FF]|[\U0002700-\U000027BF]'
        )
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody', 'neither',
            'nor', 'cannot', "can't", "won't", "wouldn't", "shouldn't", "couldn't",
            "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
            "hasn't", "haven't", "hadn't", "do", "does", "did", "is", "are", "was",
            "were", "has", "have", "had"
        }
        
        # Emoji sentiment mapping
        self.emoji_sentiment = {
            '😀': 0.8, '😃': 0.8, '😄': 0.9, '😁': 0.8, '😆': 0.7, '😅': 0.6,
            '😂': 0.8, '🤣': 0.9, '😊': 0.7, '😇': 0.6, '🙂': 0.5, '🙃': 0.3,
            '😉': 0.4, '😌': 0.3, '😍': 0.9, '🥰': 0.9, '😘': 0.8, '😗': 0.6,
            '😙': 0.6, '😚': 0.7, '😋': 0.6, '😛': 0.4, '😜': 0.3, '🤪': 0.2,
            '😝': 0.1, '🤑': 0.3, '🤗': 0.7, '🤭': 0.4, '🤫': 0.2, '🤔': 0.0,
            '🤐': -0.2, '🤨': -0.3, '😐': -0.1, '😑': -0.2, '😶': -0.3, '😏': 0.1,
            '😒': -0.4, '🙄': -0.3, '😬': -0.2, '🤥': -0.6, '😔': -0.5, '😕': -0.3,
            '🙁': -0.4, '☹️': -0.5, '😣': -0.4, '😖': -0.5, '😫': -0.6, '😩': -0.5,
            '🥺': -0.3, '😢': -0.6, '😭': -0.7, '😤': -0.4, '😠': -0.6, '😡': -0.8,
            '🤬': -0.9, '🤯': -0.2, '😳': -0.1, '🥵': -0.2, '🥶': -0.3, '😱': -0.7,
            '😨': -0.6, '😰': -0.5, '😥': -0.4, '😓': -0.3, '🤗': 0.7, '🤔': 0.0
        }
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not self.config.enable_preprocessing:
            return text
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.config.remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags
        if self.config.remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        
        # Normalize emojis
        if self.config.normalize_emojis:
            text = self._normalize_emojis(text)
        
        # Handle negations
        if self.config.handle_negations:
            text = self._handle_negations(text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _normalize_emojis(self, text: str) -> str:
        """Normalize emojis to sentiment indicators"""
        def replace_emoji(match):
            emoji = match.group()
            if emoji in self.emoji_sentiment:
                sentiment = self.emoji_sentiment[emoji]
                if sentiment > 0.5:
                    return " [POSITIVE_EMOJI] "
                elif sentiment < -0.5:
                    return " [NEGATIVE_EMOJI] "
                else:
                    return " [NEUTRAL_EMOJI] "
            return " [EMOJI] "
        
        return self.emoji_pattern.sub(replace_emoji, text)
    
    def _handle_negations(self, text: str) -> str:
        """Handle negation words to improve sentiment analysis"""
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            if word in self.negation_words and i + 1 < len(words):
                # Mark the next few words as negated
                result.append(f"NOT_{words[i+1]}")
                i += 2
                # Continue marking words until punctuation or another negation
                while i < len(words) and words[i] not in string.punctuation and words[i] not in self.negation_words:
                    result.append(f"NOT_{words[i]}")
                    i += 1
            else:
                result.append(word)
                i += 1
        
        return ' '.join(result)

class SentimentModel:
    """Base class for sentiment analysis models"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text"""
        try:
            result = self.pipeline(text)
            return result[0] if isinstance(result, list) else result
        except Exception as e:
            logger.error(f"Prediction failed for model {self.model_name}: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for a batch of texts"""
        try:
            results = self.pipeline(texts)
            return results
        except Exception as e:
            logger.error(f"Batch prediction failed for model {self.model_name}: {e}")
            return [{"label": "NEUTRAL", "score": 0.5} for _ in texts]

class EnsembleSentimentModel:
    """Ensemble model for improved sentiment analysis"""
    
    def __init__(self, model_names: List[str], device: str = "auto"):
        self.models = []
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        for model_name in model_names:
            try:
                model = SentimentModel(model_name, self.device)
                self.models.append(model)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using ensemble voting"""
        if not self.models:
            return {"label": "NEUTRAL", "score": 0.5}
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(text)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if not predictions:
            return {"label": "NEUTRAL", "score": 0.5}
        
        # Weighted voting based on confidence scores
        label_scores = defaultdict(float)
        total_weight = 0.0
        
        for pred in predictions:
            label = pred.get("label", "NEUTRAL")
            score = pred.get("score", 0.5)
            weight = score  # Use confidence as weight
            label_scores[label] += weight
            total_weight += weight
        
        if total_weight == 0:
            return {"label": "NEUTRAL", "score": 0.5}
        
        # Normalize scores
        for label in label_scores:
            label_scores[label] /= total_weight
        
        # Get the best label
        best_label = max(label_scores.items(), key=lambda x: x[1])
        
        return {
            "label": best_label[0],
            "score": best_label[1],
            "all_predictions": predictions,
            "ensemble_confidence": best_label[1]
        }

class EmotionAnalyzer:
    """Emotion analysis using transformer models"""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
    
    def analyze_emotions(self, text: str) -> Dict[EmotionLabel, float]:
        """Analyze emotions in text"""
        try:
            result = self.pipeline(text)
            emotions = {}
            
            for item in result:
                label = item["label"].lower()
                score = item["score"]
                
                # Map to our emotion labels
                if "joy" in label:
                    emotions[EmotionLabel.JOY] = score
                elif "sadness" in label:
                    emotions[EmotionLabel.SADNESS] = score
                elif "anger" in label:
                    emotions[EmotionLabel.ANGER] = score
                elif "fear" in label:
                    emotions[EmotionLabel.FEAR] = score
                elif "surprise" in label:
                    emotions[EmotionLabel.SURPRISE] = score
                elif "disgust" in label:
                    emotions[EmotionLabel.DISGUST] = score
                else:
                    emotions[EmotionLabel.NEUTRAL] = score
            
            return emotions
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {EmotionLabel.NEUTRAL: 1.0}

class SubjectivityAnalyzer:
    """Subjectivity analysis for text"""
    
    def __init__(self):
        # Simple rule-based subjectivity analysis
        self.subjective_words = {
            'think', 'believe', 'feel', 'opinion', 'personal', 'subjective',
            'probably', 'maybe', 'perhaps', 'possibly', 'likely', 'unlikely',
            'seem', 'appear', 'look', 'sound', 'taste', 'smell', 'feel',
            'wonder', 'doubt', 'suspect', 'assume', 'presume', 'suppose',
            'guess', 'estimate', 'approximate', 'roughly', 'about', 'around'
        }
        
        self.objective_indicators = {
            'fact', 'data', 'statistics', 'research', 'study', 'analysis',
            'evidence', 'proof', 'confirmed', 'verified', 'measured',
            'calculated', 'computed', 'determined', 'established'
        }
    
    def analyze_subjectivity(self, text: str) -> float:
        """Analyze subjectivity of text (0.0 = objective, 1.0 = subjective)"""
        words = text.lower().split()
        subjective_count = sum(1 for word in words if word in self.subjective_words)
        objective_count = sum(1 for word in words if word in self.objective_indicators)
        
        total_words = len(words)
        if total_words == 0:
            return 0.5
        
        # Calculate subjectivity score
        subjective_ratio = subjective_count / total_words
        objective_ratio = objective_count / total_words
        
        # Normalize to 0-1 range
        subjectivity = subjective_ratio / (subjective_ratio + objective_ratio + 1e-6)
        
        return min(1.0, max(0.0, subjectivity))

class LanguageDetector:
    """Simple language detection"""
    
    def __init__(self):
        # Simple character-based language detection
        self.language_patterns = {
            'en': re.compile(r'[a-zA-Z]'),
            'zh': re.compile(r'[\u4e00-\u9fff]'),
            'ja': re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),
            'ko': re.compile(r'[\uac00-\ud7af]'),
            'ar': re.compile(r'[\u0600-\u06ff]'),
            'ru': re.compile(r'[\u0400-\u04ff]'),
            'es': re.compile(r'[ñáéíóúü]'),
            'fr': re.compile(r'[àâäéèêëïîôöùûüÿç]'),
            'de': re.compile(r'[äöüß]'),
            'it': re.compile(r'[àèéìíîòóù]')
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        text_lower = text.lower()
        scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = pattern.findall(text_lower)
            scores[lang] = len(matches)
        
        if not scores or max(scores.values()) == 0:
            return 'en'  # Default to English
        
        return max(scores.items(), key=lambda x: x[1])[0]

class SentimentAnalyzer:
    """Advanced sentiment analysis system"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.preprocessor = TextPreprocessor(self.config)
        
        # Initialize models
        if self.config.use_ensemble:
            self.sentiment_model = EnsembleSentimentModel(
                self.config.ensemble_models,
                "cuda" if self.config.use_gpu else "cpu"
            )
        else:
            self.sentiment_model = SentimentModel(
                self.config.model_name,
                "cuda" if self.config.use_gpu else "cpu"
            )
        
        # Initialize additional analyzers
        if self.config.enable_emotion_analysis:
            self.emotion_analyzer = EmotionAnalyzer()
        
        if self.config.enable_subjectivity_analysis:
            self.subjectivity_analyzer = SubjectivityAnalyzer()
        
        if self.config.enable_language_detection:
            self.language_detector = LanguageDetector()
        
        # Initialize caching
        if self.config.enable_caching:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
        else:
            self.redis_client = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0,
            'error_count': 0
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[SentimentResult]:
        """Get result from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(f"sentiment:{cache_key}")
            if cached:
                data = json.loads(cached)
                return SentimentResult(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _set_cache(self, cache_key: str, result: SentimentResult):
        """Set result in cache"""
        if not self.redis_client:
            return
        
        try:
            data = json.dumps(result.__dict__, default=str)
            self.redis_client.setex(f"sentiment:{cache_key}", self.config.cache_ttl, data)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _map_sentiment_label(self, label: str) -> SentimentLabel:
        """Map model label to our sentiment label"""
        label_lower = label.lower()
        
        if 'very negative' in label_lower or 'very_negative' in label_lower:
            return SentimentLabel.VERY_NEGATIVE
        elif 'negative' in label_lower:
            return SentimentLabel.NEGATIVE
        elif 'neutral' in label_lower:
            return SentimentLabel.NEUTRAL
        elif 'positive' in label_lower and 'very' not in label_lower:
            return SentimentLabel.POSITIVE
        elif 'very positive' in label_lower or 'very_positive' in label_lower:
            return SentimentLabel.VERY_POSITIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _calculate_polarity_score(self, sentiment: SentimentLabel, confidence: float) -> float:
        """Calculate polarity score from -1.0 to 1.0"""
        polarity_map = {
            SentimentLabel.VERY_NEGATIVE: -1.0,
            SentimentLabel.NEGATIVE: -0.5,
            SentimentLabel.NEUTRAL: 0.0,
            SentimentLabel.POSITIVE: 0.5,
            SentimentLabel.VERY_POSITIVE: 1.0
        }
        
        base_polarity = polarity_map.get(sentiment, 0.0)
        return base_polarity * confidence
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)
            
            # Detect language
            language = "en"
            if self.config.enable_language_detection:
                language = self.language_detector.detect_language(processed_text)
            
            # Analyze sentiment
            sentiment_pred = self.sentiment_model.predict(processed_text)
            sentiment = self._map_sentiment_label(sentiment_pred["label"])
            confidence = sentiment_pred["score"]
            
            # Analyze emotions
            emotions = {}
            if self.config.enable_emotion_analysis:
                emotions = self.emotion_analyzer.analyze_emotions(processed_text)
            
            # Analyze subjectivity
            subjectivity_score = 0.5
            if self.config.enable_subjectivity_analysis:
                subjectivity_score = self.subjectivity_analyzer.analyze_subjectivity(processed_text)
            
            # Calculate polarity score
            polarity_score = self._calculate_polarity_score(sentiment, confidence)
            
            # Create result
            result = SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                emotions=emotions,
                polarity_score=polarity_score,
                subjectivity_score=subjectivity_score,
                processing_time=time.time() - start_time,
                model_used=self.config.model_name,
                language=language,
                metadata={
                    "processed_text": processed_text,
                    "raw_prediction": sentiment_pred,
                    "ensemble_used": self.config.use_ensemble
                }
            )
            
            # Cache result
            self._set_cache(cache_key, result)
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['total_requests'] - 1) + 
                 result.processing_time) / self.stats['total_requests']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            self.stats['error_count'] += 1
            
            # Return neutral result on error
            return SentimentResult(
                text=text,
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
                emotions={EmotionLabel.NEUTRAL: 1.0},
                polarity_score=0.0,
                subjectivity_score=0.5,
                processing_time=time.time() - start_time,
                model_used=self.config.model_name,
                language="en",
                metadata={"error": str(e)}
            )
    
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts"""
        tasks = [self.analyze_sentiment(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        cache_hit_rate = 0.0
        if self.stats['total_requests'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_requests']
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': self.stats['error_count'] / max(1, self.stats['total_requests'])
        }
    
    def clear_cache(self):
        """Clear the cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")

# Example usage and testing
async def main():
    """Example usage of the sentiment analyzer"""
    config = SentimentConfig(
        use_ensemble=True,
        enable_emotion_analysis=True,
        enable_subjectivity_analysis=True,
        enable_caching=True
    )
    
    analyzer = SentimentAnalyzer(config)
    
    # Test texts
    test_texts = [
        "I love this product! It's amazing! 😍",
        "This is terrible. I hate it.",
        "The weather is okay today.",
        "I'm not sure about this decision.",
        "This is absolutely fantastic! Best thing ever! 🎉"
    ]
    
    print("Analyzing sentiment...")
    for text in test_texts:
        result = await analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.sentiment.value}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Polarity: {result.polarity_score:.3f}")
        print(f"Subjectivity: {result.subjectivity_score:.3f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        if result.emotions:
            print("Emotions:")
            for emotion, score in result.emotions.items():
                print(f"  {emotion.value}: {score:.3f}")
    
    # Batch analysis
    print("\n" + "="*50)
    print("Batch analysis...")
    batch_results = await analyzer.analyze_batch(test_texts)
    
    for i, result in enumerate(batch_results):
        print(f"Text {i+1}: {result.sentiment.value} ({result.confidence:.3f})")
    
    # Statistics
    print("\n" + "="*50)
    print("Statistics:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())