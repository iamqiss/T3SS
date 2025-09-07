# T3SS Project
# File: core/querying/searcher/result_aggregator.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import time
from collections import defaultdict, Counter
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggregationStrategy(Enum):
    """Strategies for result aggregation"""
    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX_SCORE = "max_score"
    MIN_SCORE = "min_score"
    HARMONIC_MEAN = "harmonic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    CUSTOM_FUNCTION = "custom_function"

class DeduplicationMethod(Enum):
    """Methods for deduplicating results"""
    EXACT_MATCH = "exact_match"
    SIMILARITY_THRESHOLD = "similarity_threshold"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CONTENT_HASH = "content_hash"

@dataclass
class SearchResult:
    """Individual search result"""
    result_id: str
    title: str
    content: str
    url: str
    score: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    language: str = "en"
    quality_score: float = 0.0
    freshness_score: float = 0.0
    authority_score: float = 0.0
    relevance_score: float = 0.0

@dataclass
class AggregationConfig:
    """Configuration for result aggregation"""
    # Scoring weights
    title_weight: float = 0.3
    content_weight: float = 0.4
    metadata_weight: float = 0.1
    quality_weight: float = 0.1
    freshness_weight: float = 0.05
    authority_weight: float = 0.05
    
    # Deduplication settings
    deduplication_method: DeduplicationMethod = DeduplicationMethod.SIMILARITY_THRESHOLD
    similarity_threshold: float = 0.8
    max_results: int = 1000
    min_score_threshold: float = 0.1
    
    # Aggregation settings
    aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_SUM
    diversity_penalty: float = 0.1
    freshness_decay_factor: float = 0.1
    quality_boost_factor: float = 1.2
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    max_workers: int = 4
    
    # Quality settings
    enable_quality_filtering: bool = True
    min_quality_score: float = 0.3
    enable_freshness_boost: bool = True
    freshness_window_days: int = 30

@dataclass
class AggregatedResults:
    """Aggregated search results"""
    results: List[SearchResult]
    total_count: int
    aggregated_score: float
    diversity_score: float
    quality_score: float
    freshness_score: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SimilarityCalculator:
    """Calculate similarity between search results"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._fitted = False
    
    def _fit_vectorizer(self, texts: List[str]):
        """Fit TF-IDF vectorizer on texts"""
        if not self._fitted and texts:
            self.tfidf_vectorizer.fit(texts)
            self._fitted = True
    
    def calculate_similarity(self, result1: SearchResult, result2: SearchResult) -> float:
        """Calculate similarity between two results"""
        # Combine title and content for similarity calculation
        text1 = f"{result1.title} {result1.content}"
        text2 = f"{result2.title} {result2.content}"
        
        # Use TF-IDF cosine similarity
        try:
            if not self._fitted:
                self._fit_vectorizer([text1, text2])
            
            vectors = self.tfidf_vectorizer.transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, result1: SearchResult, result2: SearchResult) -> float:
        """Calculate semantic similarity using embeddings"""
        # This would use actual embeddings in production
        # For now, use a simplified approach
        title_sim = self._text_similarity(result1.title, result2.title)
        content_sim = self._text_similarity(result1.content, result2.content)
        
        return (title_sim * 0.6 + content_sim * 0.4)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

class QualityAssessor:
    """Assess quality of search results"""
    
    def __init__(self):
        self.quality_indicators = {
            'title_length': (10, 100),  # Optimal title length range
            'content_length': (100, 10000),  # Optimal content length range
            'url_quality': True,  # URL structure quality
            'metadata_completeness': True,  # Metadata completeness
            'language_consistency': True  # Language consistency
        }
    
    def assess_quality(self, result: SearchResult) -> float:
        """Assess quality of a search result"""
        quality_scores = []
        
        # Title quality
        title_score = self._assess_title_quality(result.title)
        quality_scores.append(title_score)
        
        # Content quality
        content_score = self._assess_content_quality(result.content)
        quality_scores.append(content_score)
        
        # URL quality
        url_score = self._assess_url_quality(result.url)
        quality_scores.append(url_score)
        
        # Metadata quality
        metadata_score = self._assess_metadata_quality(result.metadata)
        quality_scores.append(metadata_score)
        
        # Language consistency
        language_score = self._assess_language_consistency(result)
        quality_scores.append(language_score)
        
        # Calculate weighted average
        weights = [0.3, 0.4, 0.1, 0.1, 0.1]
        weighted_score = sum(score * weight for score, weight in zip(quality_scores, weights))
        
        return min(1.0, max(0.0, weighted_score))
    
    def _assess_title_quality(self, title: str) -> float:
        """Assess title quality"""
        if not title:
            return 0.0
        
        length = len(title)
        optimal_min, optimal_max = self.quality_indicators['title_length']
        
        if optimal_min <= length <= optimal_max:
            return 1.0
        elif length < optimal_min:
            return length / optimal_min
        else:
            return max(0.5, 1.0 - (length - optimal_max) / optimal_max)
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality"""
        if not content:
            return 0.0
        
        length = len(content)
        optimal_min, optimal_max = self.quality_indicators['content_length']
        
        if optimal_min <= length <= optimal_max:
            return 1.0
        elif length < optimal_min:
            return length / optimal_min
        else:
            return max(0.5, 1.0 - (length - optimal_max) / optimal_max)
    
    def _assess_url_quality(self, url: str) -> float:
        """Assess URL quality"""
        if not url:
            return 0.0
        
        # Check for common quality indicators
        quality_indicators = [
            'https://' in url,
            len(url) < 200,  # Not too long
            '.' in url,  # Has domain
            not any(char in url for char in ['?', '#', '&']) or url.count('?') <= 2  # Not too many parameters
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _assess_metadata_quality(self, metadata: Dict[str, Any]) -> float:
        """Assess metadata quality"""
        if not metadata:
            return 0.0
        
        # Check for important metadata fields
        important_fields = ['description', 'keywords', 'author', 'published_date']
        present_fields = sum(1 for field in important_fields if field in metadata and metadata[field])
        
        return present_fields / len(important_fields)
    
    def _assess_language_consistency(self, result: SearchResult) -> float:
        """Assess language consistency"""
        # Simple language consistency check
        if result.language == "en":
            # Check if content appears to be in English
            english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            content_words = set(result.content.lower().split())
            english_ratio = len(content_words.intersection(english_words)) / max(1, len(content_words))
            return min(1.0, english_ratio * 2)  # Scale up the ratio
        
        return 1.0  # Assume consistency for non-English content

class FreshnessCalculator:
    """Calculate freshness scores for search results"""
    
    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor
    
    def calculate_freshness(self, result: SearchResult, reference_time: datetime = None) -> float:
        """Calculate freshness score for a result"""
        if reference_time is None:
            reference_time = datetime.now()
        
        # Calculate age in days
        age_days = (reference_time - result.timestamp).total_seconds() / (24 * 3600)
        
        # Apply exponential decay
        freshness = np.exp(-self.decay_factor * age_days)
        
        return min(1.0, max(0.0, freshness))
    
    def calculate_freshness_boost(self, result: SearchResult, window_days: int = 30) -> float:
        """Calculate freshness boost for recent results"""
        age_days = (datetime.now() - result.timestamp).total_seconds() / (24 * 3600)
        
        if age_days <= window_days:
            # Linear boost for recent results
            boost = 1.0 - (age_days / window_days)
            return 1.0 + (boost * 0.5)  # Up to 50% boost
        else:
            return 1.0

class DiversityCalculator:
    """Calculate diversity scores for result sets"""
    
    def __init__(self):
        self.similarity_calculator = SimilarityCalculator()
    
    def calculate_diversity(self, results: List[SearchResult]) -> float:
        """Calculate diversity score for a set of results"""
        if len(results) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                sim = self.similarity_calculator.calculate_similarity(results[i], results[j])
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
    
    def calculate_category_diversity(self, results: List[SearchResult]) -> float:
        """Calculate diversity based on categories"""
        if not results:
            return 0.0
        
        # Count categories
        category_counts = Counter()
        for result in results:
            for category in result.categories:
                category_counts[category] += 1
        
        # Calculate entropy
        total_results = len(results)
        entropy = 0.0
        
        for count in category_counts.values():
            if count > 0:
                p = count / total_results
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(category_counts)) if category_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy

class ResultAggregator:
    """Advanced result aggregation system"""
    
    def __init__(self, config: AggregationConfig = None):
        self.config = config or AggregationConfig()
        self.similarity_calculator = SimilarityCalculator()
        self.quality_assessor = QualityAssessor()
        self.freshness_calculator = FreshnessCalculator(self.config.freshness_decay_factor)
        self.diversity_calculator = DiversityCalculator()
        
        # Redis connection for caching
        self.redis_client = None
        if self.config.enable_caching:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
        
        # Statistics
        self.stats = {
            'total_aggregations': 0,
            'total_results_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0
        }
    
    def _get_cache_key(self, results: List[SearchResult]) -> str:
        """Generate cache key for results"""
        # Create a hash of result IDs and scores
        result_data = [(r.result_id, r.score) for r in results]
        data_str = json.dumps(sorted(result_data), default=str)
        return f"aggregation:{hashlib.md5(data_str.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[AggregatedResults]:
        """Get aggregated results from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                # Reconstruct SearchResult objects
                results = []
                for result_data in data['results']:
                    result = SearchResult(**result_data)
                    results.append(result)
                
                aggregated = AggregatedResults(
                    results=results,
                    total_count=data['total_count'],
                    aggregated_score=data['aggregated_score'],
                    diversity_score=data['diversity_score'],
                    quality_score=data['quality_score'],
                    freshness_score=data['freshness_score'],
                    processing_time=data['processing_time'],
                    metadata=data.get('metadata', {})
                )
                return aggregated
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _set_cache(self, cache_key: str, aggregated: AggregatedResults):
        """Set aggregated results in cache"""
        if not self.redis_client:
            return
        
        try:
            # Convert SearchResult objects to dictionaries
            results_data = []
            for result in aggregated.results:
                result_dict = {
                    'result_id': result.result_id,
                    'title': result.title,
                    'content': result.content,
                    'url': result.url,
                    'score': result.score,
                    'source': result.source,
                    'timestamp': result.timestamp.isoformat(),
                    'metadata': result.metadata,
                    'features': result.features,
                    'categories': result.categories,
                    'language': result.language,
                    'quality_score': result.quality_score,
                    'freshness_score': result.freshness_score,
                    'authority_score': result.authority_score,
                    'relevance_score': result.relevance_score
                }
                results_data.append(result_dict)
            
            data = {
                'results': results_data,
                'total_count': aggregated.total_count,
                'aggregated_score': aggregated.aggregated_score,
                'diversity_score': aggregated.diversity_score,
                'quality_score': aggregated.quality_score,
                'freshness_score': aggregated.freshness_score,
                'processing_time': aggregated.processing_time,
                'metadata': aggregated.metadata
            }
            
            self.redis_client.setex(cache_key, self.config.cache_ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def aggregate_results(self, result_sets: List[List[SearchResult]]) -> AggregatedResults:
        """Aggregate multiple result sets into a single ranked list"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key([result for result_set in result_sets for result in result_set])
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        try:
            # Flatten all results
            all_results = []
            for result_set in result_sets:
                all_results.extend(result_set)
            
            if not all_results:
                return AggregatedResults(
                    results=[],
                    total_count=0,
                    aggregated_score=0.0,
                    diversity_score=0.0,
                    quality_score=0.0,
                    freshness_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Deduplicate results
            deduplicated_results = await self._deduplicate_results(all_results)
            
            # Calculate quality scores
            for result in deduplicated_results:
                result.quality_score = self.quality_assessor.assess_quality(result)
            
            # Calculate freshness scores
            for result in deduplicated_results:
                result.freshness_score = self.freshness_calculator.calculate_freshness(result)
            
            # Apply quality filtering
            if self.config.enable_quality_filtering:
                deduplicated_results = [
                    r for r in deduplicated_results 
                    if r.quality_score >= self.config.min_quality_score
                ]
            
            # Apply freshness boost
            if self.config.enable_freshness_boost:
                for result in deduplicated_results:
                    boost = self.freshness_calculator.calculate_freshness_boost(
                        result, self.config.freshness_window_days
                    )
                    result.score *= boost
            
            # Calculate aggregated scores
            for result in deduplicated_results:
                result.relevance_score = self._calculate_relevance_score(result)
            
            # Sort by aggregated score
            deduplicated_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply max results limit
            if len(deduplicated_results) > self.config.max_results:
                deduplicated_results = deduplicated_results[:self.config.max_results]
            
            # Calculate diversity score
            diversity_score = self.diversity_calculator.calculate_diversity(deduplicated_results)
            
            # Calculate quality score
            quality_score = np.mean([r.quality_score for r in deduplicated_results]) if deduplicated_results else 0.0
            
            # Calculate freshness score
            freshness_score = np.mean([r.freshness_score for r in deduplicated_results]) if deduplicated_results else 0.0
            
            # Calculate aggregated score
            aggregated_score = np.mean([r.relevance_score for r in deduplicated_results]) if deduplicated_results else 0.0
            
            # Create aggregated results
            aggregated = AggregatedResults(
                results=deduplicated_results,
                total_count=len(deduplicated_results),
                aggregated_score=aggregated_score,
                diversity_score=diversity_score,
                quality_score=quality_score,
                freshness_score=freshness_score,
                processing_time=time.time() - start_time,
                metadata={
                    'original_count': len(all_results),
                    'deduplication_ratio': len(deduplicated_results) / len(all_results) if all_results else 0.0,
                    'quality_filtered': self.config.enable_quality_filtering,
                    'freshness_boosted': self.config.enable_freshness_boost
                }
            )
            
            # Cache result
            self._set_cache(cache_key, aggregated)
            
            # Update statistics
            self.stats['total_aggregations'] += 1
            self.stats['total_results_processed'] += len(all_results)
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['total_aggregations'] - 1) + 
                 aggregated.processing_time) / self.stats['total_aggregations']
            )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            return AggregatedResults(
                results=[],
                total_count=0,
                aggregated_score=0.0,
                diversity_score=0.0,
                quality_score=0.0,
                freshness_score=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _calculate_relevance_score(self, result: SearchResult) -> float:
        """Calculate relevance score for a result"""
        # Base score from the result
        base_score = result.score
        
        # Apply quality weight
        quality_score = result.quality_score * self.config.quality_weight
        
        # Apply freshness weight
        freshness_score = result.freshness_score * self.config.freshness_weight
        
        # Apply authority weight
        authority_score = result.authority_score * self.config.authority_weight
        
        # Calculate weighted sum
        relevance_score = (
            base_score * 0.5 +  # Base score gets 50% weight
            quality_score +
            freshness_score +
            authority_score
        )
        
        return min(1.0, max(0.0, relevance_score))
    
    async def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate results based on configured method"""
        if not results:
            return []
        
        if self.config.deduplication_method == DeduplicationMethod.EXACT_MATCH:
            return self._deduplicate_exact_match(results)
        elif self.config.deduplication_method == DeduplicationMethod.SIMILARITY_THRESHOLD:
            return await self._deduplicate_similarity_threshold(results)
        elif self.config.deduplication_method == DeduplicationMethod.SEMANTIC_SIMILARITY:
            return await self._deduplicate_semantic_similarity(results)
        elif self.config.deduplication_method == DeduplicationMethod.CONTENT_HASH:
            return self._deduplicate_content_hash(results)
        else:
            return results
    
    def _deduplicate_exact_match(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate using exact URL matching"""
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _deduplicate_similarity_threshold(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate using similarity threshold"""
        if len(results) <= 1:
            return results
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        deduplicated = []
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self.similarity_calculator.calculate_similarity(result, existing)
                if similarity >= self.config.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    async def _deduplicate_semantic_similarity(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate using semantic similarity"""
        if len(results) <= 1:
            return results
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        deduplicated = []
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self.similarity_calculator.calculate_semantic_similarity(result, existing)
                if similarity >= self.config.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _deduplicate_content_hash(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate using content hash"""
        seen_hashes = set()
        deduplicated = []
        
        for result in results:
            # Create content hash
            content_hash = hashlib.md5(f"{result.title}{result.content}".encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(result)
        
        return deduplicated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear the cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")

# Example usage
async def main():
    """Example usage of the result aggregator"""
    config = AggregationConfig(
        max_results=100,
        similarity_threshold=0.8,
        enable_quality_filtering=True,
        enable_freshness_boost=True
    )
    
    aggregator = ResultAggregator(config)
    
    # Create sample results
    results1 = [
        SearchResult(
            result_id="1",
            title="Machine Learning Tutorial",
            content="A comprehensive guide to machine learning algorithms and techniques.",
            url="https://example.com/ml-tutorial",
            score=0.9,
            source="search_engine_1",
            timestamp=datetime.now() - timedelta(days=1),
            categories=["technology", "education"]
        ),
        SearchResult(
            result_id="2",
            title="Deep Learning Basics",
            content="Introduction to deep learning and neural networks.",
            url="https://example.com/dl-basics",
            score=0.8,
            source="search_engine_1",
            timestamp=datetime.now() - timedelta(days=2),
            categories=["technology", "education"]
        )
    ]
    
    results2 = [
        SearchResult(
            result_id="3",
            title="ML Tutorial for Beginners",
            content="A beginner-friendly guide to machine learning concepts.",
            url="https://example.com/ml-beginners",
            score=0.85,
            source="search_engine_2",
            timestamp=datetime.now() - timedelta(days=1),
            categories=["technology", "education"]
        ),
        SearchResult(
            result_id="4",
            title="Data Science Guide",
            content="Complete guide to data science and analytics.",
            url="https://example.com/data-science",
            score=0.7,
            source="search_engine_2",
            timestamp=datetime.now() - timedelta(days=5),
            categories=["technology", "data"]
        )
    ]
    
    # Aggregate results
    aggregated = await aggregator.aggregate_results([results1, results2])
    
    print(f"Aggregated {aggregated.total_count} results")
    print(f"Aggregated score: {aggregated.aggregated_score:.3f}")
    print(f"Diversity score: {aggregated.diversity_score:.3f}")
    print(f"Quality score: {aggregated.quality_score:.3f}")
    print(f"Freshness score: {aggregated.freshness_score:.3f}")
    print(f"Processing time: {aggregated.processing_time:.3f}s")
    
    print("\nTop results:")
    for i, result in enumerate(aggregated.results[:5]):
        print(f"{i+1}. {result.title} (score: {result.relevance_score:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
