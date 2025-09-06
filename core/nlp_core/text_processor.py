#!/usr/bin/env python3
"""
T3SS Project
File: core/nlp_core/text_processor.py
(c) 2025 Qiss Labs. All Rights Reserved.
Unauthorized copying or distribution of this file is strictly prohibited.
For internal use only.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import re
import time
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# NLP Libraries
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec, Doc2Vec

# ML Libraries
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel,
    BertTokenizer, BertModel,
    pipeline
)

@dataclass
class TextProcessingConfig:
    """Configuration for text processing"""
    enable_ner: bool = True
    enable_sentiment: bool = True
    enable_keywords: bool = True
    enable_summarization: bool = True
    enable_language_detection: bool = True
    max_text_length: int = 10000
    batch_size: int = 32
    enable_gpu: bool = True

class TextProcessor:
    """Advanced text processing with NLP capabilities"""
    
    def __init__(self, config: TextProcessingConfig):
        self.config = config
        self.nlp = None
        self.stop_words = set()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_pipeline = None
        self.summarization_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Download NLTK data
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            
            # Load stop words
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize sentiment analysis
            if self.config.enable_sentiment:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() and self.config.enable_gpu else -1
                )
            
            # Initialize summarization
            if self.config.enable_summarization:
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() and self.config.enable_gpu else -1
                )
            
        except Exception as e:
            logging.warning(f"Failed to initialize some NLP models: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better analysis"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove stop words
        if self.stop_words:
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not self.nlp or not self.config.enable_ner:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0
            })
        
        return entities
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords from text using TF-IDF"""
        if not text or not self.config.enable_keywords:
            return []
        
        vectorizer = TfidfVectorizer(
            max_features=top_k,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            keywords = list(zip(feature_names, scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords
        except Exception as e:
            logging.warning(f"Keyword extraction failed: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not text or not self.config.enable_sentiment or not self.sentiment_pipeline:
            return {'label': 'neutral', 'score': 0.5}
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
            
            result = self.sentiment_pipeline(text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logging.warning(f"Sentiment analysis failed: {e}")
            return {'label': 'neutral', 'score': 0.5}
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text"""
        if not text or not self.config.enable_summarization or not self.summarization_pipeline:
            return self._simple_summarize(text)
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
            
            result = self.summarization_pipeline(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            
            return result[0]['summary_text']
        except Exception as e:
            logging.warning(f"Summarization failed: {e}")
            return self._simple_summarize(text)
    
    def _simple_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization"""
        if not text:
            return ""
        
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        return '. '.join(sentences[:max_sentences]) + '.'
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not text or not self.config.enable_language_detection:
            return "en"
        
        # Simple language detection based on common words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = set(text.lower().split())
        
        if len(words.intersection(english_words)) > 0:
            return "en"
        else:
            return "unknown"
    
    def extract_phrases(self, text: str, min_count: int = 2) -> List[str]:
        """Extract meaningful phrases from text"""
        if not text:
            return []
        
        # Simple phrase extraction using NLTK
        sentences = sent_tokenize(text)
        phrases = []
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            # Extract bigrams and trigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram.split()) == 2:
                    phrases.append(bigram)
            
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(trigram.split()) == 3:
                    phrases.append(trigram)
        
        # Count phrase frequency
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Return phrases that appear at least min_count times
        return [phrase for phrase, count in phrase_counts.items() if count >= min_count]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logging.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def extract_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Extract part-of-speech tags"""
        if not text:
            return []
        
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            return pos_tags
        except Exception as e:
            logging.warning(f"POS tagging failed: {e}")
            return []
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text"""
        if not text:
            return ""
        
        try:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except Exception as e:
            logging.warning(f"Lemmatization failed: {e}")
            return text
    
    def stem_text(self, text: str) -> str:
        """Stem text"""
        if not text:
            return ""
        
        try:
            words = word_tokenize(text)
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        except Exception as e:
            logging.warning(f"Stemming failed: {e}")
            return text
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """Process a document and extract all features"""
        if not text:
            return {}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract features
        result = {
            'original_text': text,
            'processed_text': processed_text,
            'entities': self.extract_entities(text),
            'keywords': self.extract_keywords(text),
            'sentiment': self.analyze_sentiment(text),
            'summary': self.summarize_text(text),
            'language': self.detect_language(text),
            'phrases': self.extract_phrases(text),
            'pos_tags': self.extract_pos_tags(text),
            'lemmatized': self.lemmatize_text(text),
            'stemmed': self.stem_text(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'character_count': len(text)
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = TextProcessingConfig()
    
    # Create processor
    processor = TextProcessor(config)
    
    # Example text
    text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.
    It enables computers to learn and improve from experience without being explicitly programmed.
    Deep learning, a subset of machine learning, uses neural networks with multiple layers.
    """
    
    # Process text
    result = processor.process_document(text)
    
    print("Text Processing Results:")
    print(f"Language: {result['language']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Keywords: {result['keywords'][:5]}")
    print(f"Entities: {result['entities']}")
    print(f"Summary: {result['summary']}")
    print(f"Word Count: {result['word_count']}")