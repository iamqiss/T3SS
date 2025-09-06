"""
T3SS Project - News Search Vertical
Advanced news processing and analysis for search functionality
(c) 2025 Qiss Labs. All Rights Reserved.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from datetime import datetime, timedelta
import hashlib

import aiohttp
import asyncpg
import redis.asyncio as redis
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import spacy
from textblob import TextBlob
import yake
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import feedparser
import dateutil.parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCategory(Enum):
    """News categories"""
    POLITICS = "politics"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    SCIENCE = "science"
    WORLD = "world"
    LOCAL = "local"
    OPINION = "opinion"
    WEATHER = "weather"
    CRIME = "crime"
    EDUCATION = "education"
    ENVIRONMENT = "environment"
    TRAVEL = "travel"
    FOOD = "food"
    FASHION = "fashion"
    AUTOMOTIVE = "automotive"
    REAL_ESTATE = "real_estate"
    FINANCE = "finance"

class NewsSentiment(Enum):
    """News sentiment"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class NewsCredibility(Enum):
    """News credibility levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class NewsMetadata:
    """News article metadata"""
    title: str
    author: str
    publication: str
    published_date: datetime
    modified_date: Optional[datetime]
    url: str
    source_url: str
    language: str
    category: NewsCategory
    subcategory: str
    tags: List[str]
    keywords: List[str]
    entities: List[Dict[str, Any]]
    sentiment: NewsSentiment
    sentiment_score: float
    credibility: NewsCredibility
    word_count: int
    reading_time: int
    has_images: bool
    has_videos: bool
    has_audio: bool
    is_breaking: bool
    is_opinion: bool
    is_satire: bool
    is_press_release: bool
    content_hash: str
    duplicate_count: int

@dataclass
class NewsContent:
    """News article content"""
    headline: str
    summary: str
    body: str
    lead_paragraph: str
    conclusion: str
    quotes: List[str]
    statistics: List[str]
    locations: List[str]
    people: List[str]
    organizations: List[str]
    topics: List[str]
    key_phrases: List[str]
    named_entities: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    readability_score: float
    content_quality_score: float

@dataclass
class NewsSearchResult:
    """News search result"""
    article_id: str
    title: str
    summary: str
    url: str
    source: str
    author: str
    published_date: datetime
    category: NewsCategory
    sentiment: NewsSentiment
    credibility: NewsCredibility
    relevance_score: float
    freshness_score: float
    quality_score: float
    metadata: NewsMetadata
    content: NewsContent
    related_articles: List[str]
    trending_score: float
    social_engagement: Dict[str, int]

class NewsProcessor:
    """Advanced news processing and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.keyword_extractor = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            # Load spaCy model
            self.nlp_model = spacy.load("en_core_web_sm")
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize keyword extractor
            self.keyword_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.7,
                top=20
            )
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('taggers/averaged_perceptron_tagger')
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.data.find('taggers/averaged_perceptron_tagger')
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            
            logger.info("News processor models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing news processor models: {e}")
            raise
    
    async def process_news_article(self, article_data: Dict[str, Any]) -> NewsSearchResult:
        """Process news article and extract all features"""
        try:
            # Extract metadata
            metadata = await self._extract_metadata(article_data)
            
            # Extract content
            content = await self._extract_content(article_data)
            
            # Generate article ID
            article_id = self._generate_article_id(article_data)
            
            # Calculate scores
            relevance_score = await self._calculate_relevance_score(metadata, content)
            freshness_score = await self._calculate_freshness_score(metadata)
            quality_score = await self._calculate_quality_score(metadata, content)
            trending_score = await self._calculate_trending_score(article_id)
            
            # Get related articles
            related_articles = await self._find_related_articles(article_id, content)
            
            # Get social engagement
            social_engagement = await self._get_social_engagement(article_data)
            
            # Create search result
            result = NewsSearchResult(
                article_id=article_id,
                title=metadata.title,
                summary=content.summary,
                url=metadata.url,
                source=metadata.publication,
                author=metadata.author,
                published_date=metadata.published_date,
                category=metadata.category,
                sentiment=metadata.sentiment,
                credibility=metadata.credibility,
                relevance_score=relevance_score,
                freshness_score=freshness_score,
                quality_score=quality_score,
                metadata=metadata,
                content=content,
                related_articles=related_articles,
                trending_score=trending_score,
                social_engagement=social_engagement
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing news article: {e}")
            raise
    
    async def _extract_metadata(self, article_data: Dict[str, Any]) -> NewsMetadata:
        """Extract comprehensive article metadata"""
        try:
            # Basic metadata
            title = article_data.get('title', '')
            author = article_data.get('author', '')
            publication = article_data.get('publication', '')
            url = article_data.get('url', '')
            source_url = article_data.get('source_url', '')
            
            # Parse dates
            published_date = self._parse_date(article_data.get('published_date', ''))
            modified_date = self._parse_date(article_data.get('modified_date', ''))
            
            # Language detection
            language = await self._detect_language(article_data.get('content', ''))
            
            # Category classification
            category = await self._classify_category(article_data)
            
            # Extract tags and keywords
            tags = article_data.get('tags', [])
            keywords = await self._extract_keywords(article_data.get('content', ''))
            
            # Named entity recognition
            entities = await self._extract_entities(article_data.get('content', ''))
            
            # Sentiment analysis
            sentiment, sentiment_score = await self._analyze_sentiment(article_data.get('content', ''))
            
            # Credibility assessment
            credibility = await self._assess_credibility(article_data)
            
            # Content analysis
            word_count = len(article_data.get('content', '').split())
            reading_time = max(1, word_count // 200)  # Average reading speed: 200 words per minute
            
            # Media analysis
            has_images = bool(article_data.get('images', []))
            has_videos = bool(article_data.get('videos', []))
            has_audio = bool(article_data.get('audio', []))
            
            # Content type analysis
            is_breaking = await self._is_breaking_news(article_data)
            is_opinion = await self._is_opinion_piece(article_data)
            is_satire = await self._is_satire(article_data)
            is_press_release = await self._is_press_release(article_data)
            
            # Generate content hash
            content_hash = hashlib.sha256(article_data.get('content', '').encode()).hexdigest()
            
            # Check for duplicates
            duplicate_count = await self._count_duplicates(content_hash)
            
            return NewsMetadata(
                title=title,
                author=author,
                publication=publication,
                published_date=published_date,
                modified_date=modified_date,
                url=url,
                source_url=source_url,
                language=language,
                category=category,
                subcategory=article_data.get('subcategory', ''),
                tags=tags,
                keywords=keywords,
                entities=entities,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                credibility=credibility,
                word_count=word_count,
                reading_time=reading_time,
                has_images=has_images,
                has_videos=has_videos,
                has_audio=has_audio,
                is_breaking=is_breaking,
                is_opinion=is_opinion,
                is_satire=is_satire,
                is_press_release=is_press_release,
                content_hash=content_hash,
                duplicate_count=duplicate_count
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            raise
    
    async def _extract_content(self, article_data: Dict[str, Any]) -> NewsContent:
        """Extract and analyze article content"""
        try:
            content_text = article_data.get('content', '')
            
            # Basic content extraction
            headline = article_data.get('title', '')
            summary = await self._generate_summary(content_text)
            body = content_text
            lead_paragraph = await self._extract_lead_paragraph(content_text)
            conclusion = await self._extract_conclusion(content_text)
            
            # Extract quotes
            quotes = await self._extract_quotes(content_text)
            
            # Extract statistics
            statistics = await self._extract_statistics(content_text)
            
            # Extract locations
            locations = await self._extract_locations(content_text)
            
            # Extract people
            people = await self._extract_people(content_text)
            
            # Extract organizations
            organizations = await self._extract_organizations(content_text)
            
            # Extract topics
            topics = await self._extract_topics(content_text)
            
            # Extract key phrases
            key_phrases = await self._extract_key_phrases(content_text)
            
            # Named entity recognition
            named_entities = await self._extract_named_entities(content_text)
            
            # Sentiment analysis
            sentiment_analysis = await self._analyze_detailed_sentiment(content_text)
            
            # Readability score
            readability_score = await self._calculate_readability(content_text)
            
            # Content quality score
            content_quality_score = await self._calculate_content_quality(content_text)
            
            return NewsContent(
                headline=headline,
                summary=summary,
                body=body,
                lead_paragraph=lead_paragraph,
                conclusion=conclusion,
                quotes=quotes,
                statistics=statistics,
                locations=locations,
                people=people,
                organizations=organizations,
                topics=topics,
                key_phrases=key_phrases,
                named_entities=named_entities,
                sentiment_analysis=sentiment_analysis,
                readability_score=readability_score,
                content_quality_score=content_quality_score
            )
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            raise
    
    async def _classify_category(self, article_data: Dict[str, Any]) -> NewsCategory:
        """Classify article category using ML and heuristics"""
        try:
            title = article_data.get('title', '').lower()
            content = article_data.get('content', '').lower()
            tags = [tag.lower() for tag in article_data.get('tags', [])]
            
            # Category keywords
            category_keywords = {
                NewsCategory.POLITICS: ['politics', 'government', 'election', 'president', 'congress', 'senate', 'vote', 'campaign'],
                NewsCategory.BUSINESS: ['business', 'economy', 'market', 'stock', 'company', 'corporate', 'finance', 'investment'],
                NewsCategory.TECHNOLOGY: ['technology', 'tech', 'software', 'ai', 'artificial intelligence', 'computer', 'internet', 'digital'],
                NewsCategory.SPORTS: ['sports', 'football', 'basketball', 'baseball', 'soccer', 'tennis', 'olympics', 'championship'],
                NewsCategory.ENTERTAINMENT: ['entertainment', 'movie', 'music', 'celebrity', 'hollywood', 'tv', 'show', 'actor'],
                NewsCategory.HEALTH: ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment', 'medicine', 'healthcare'],
                NewsCategory.SCIENCE: ['science', 'research', 'study', 'scientist', 'discovery', 'experiment', 'laboratory'],
                NewsCategory.WORLD: ['world', 'international', 'global', 'foreign', 'country', 'nation', 'diplomatic'],
                NewsCategory.LOCAL: ['local', 'city', 'town', 'community', 'neighborhood', 'regional'],
                NewsCategory.OPINION: ['opinion', 'editorial', 'commentary', 'analysis', 'perspective', 'viewpoint'],
                NewsCategory.WEATHER: ['weather', 'climate', 'temperature', 'rain', 'snow', 'storm', 'forecast'],
                NewsCategory.CRIME: ['crime', 'police', 'arrest', 'investigation', 'court', 'trial', 'criminal'],
                NewsCategory.EDUCATION: ['education', 'school', 'university', 'student', 'teacher', 'academic', 'learning'],
                NewsCategory.ENVIRONMENT: ['environment', 'climate change', 'pollution', 'conservation', 'sustainability', 'green'],
                NewsCategory.TRAVEL: ['travel', 'tourism', 'vacation', 'hotel', 'airline', 'destination', 'trip'],
                NewsCategory.FOOD: ['food', 'restaurant', 'cooking', 'recipe', 'chef', 'cuisine', 'dining'],
                NewsCategory.FASHION: ['fashion', 'style', 'clothing', 'designer', 'trend', 'beauty', 'cosmetics'],
                NewsCategory.AUTOMOTIVE: ['automotive', 'car', 'vehicle', 'auto', 'driving', 'transportation', 'automobile'],
                NewsCategory.REAL_ESTATE: ['real estate', 'property', 'housing', 'home', 'mortgage', 'rent', 'construction'],
                NewsCategory.FINANCE: ['finance', 'banking', 'money', 'currency', 'trading', 'investment', 'financial']
            }
            
            # Score each category
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in title:
                        score += 3  # Title matches are weighted higher
                    if keyword in content:
                        score += 1
                    if keyword in tags:
                        score += 2  # Tag matches are weighted higher
                
                category_scores[category] = score
            
            # Return category with highest score
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                if category_scores[best_category] > 0:
                    return best_category
            
            # Default to world news if no clear category
            return NewsCategory.WORLD
            
        except Exception as e:
            logger.error(f"Error classifying category: {e}")
            return NewsCategory.WORLD
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        try:
            if not content:
                return []
            
            # Use YAKE for keyword extraction
            keywords = self.keyword_extractor.extract_keywords(content)
            
            # Extract top keywords
            top_keywords = [kw[1] for kw in keywords[:10]]
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content"""
        try:
            if not content:
                return []
            
            # Use spaCy for entity recognition
            doc = self.nlp_model(content)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def _analyze_sentiment(self, content: str) -> Tuple[NewsSentiment, float]:
        """Analyze sentiment of content"""
        try:
            if not content:
                return NewsSentiment.NEUTRAL, 0.0
            
            # Use transformer model for sentiment analysis
            results = self.sentiment_analyzer(content[:512])  # Limit to 512 tokens
            
            # Get the highest scoring sentiment
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Map to our sentiment enum
            sentiment_map = {
                'LABEL_0': NewsSentiment.NEGATIVE,
                'LABEL_1': NewsSentiment.NEUTRAL,
                'LABEL_2': NewsSentiment.POSITIVE
            }
            
            sentiment = sentiment_map.get(best_result['label'], NewsSentiment.NEUTRAL)
            score = best_result['score']
            
            return sentiment, score
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return NewsSentiment.NEUTRAL, 0.0
    
    async def _assess_credibility(self, article_data: Dict[str, Any]) -> NewsCredibility:
        """Assess article credibility"""
        try:
            publication = article_data.get('publication', '').lower()
            url = article_data.get('url', '').lower()
            
            # Known credible sources
            credible_sources = [
                'bbc', 'cnn', 'reuters', 'ap', 'associated press', 'new york times',
                'washington post', 'wall street journal', 'npr', 'pbs', 'propublica',
                'the guardian', 'financial times', 'bloomberg', 'forbes', 'time',
                'newsweek', 'usatoday', 'abc news', 'cbs news', 'nbc news'
            ]
            
            # Known less credible sources
            less_credible_sources = [
                'infowars', 'breitbart', 'daily stormer', 'stormfront'
            ]
            
            # Check publication credibility
            for source in credible_sources:
                if source in publication or source in url:
                    return NewsCredibility.HIGH
            
            for source in less_credible_sources:
                if source in publication or source in url:
                    return NewsCredibility.LOW
            
            # Check for other credibility indicators
            content = article_data.get('content', '')
            
            # Check for author information
            has_author = bool(article_data.get('author', ''))
            
            # Check for publication date
            has_date = bool(article_data.get('published_date', ''))
            
            # Check for source citations
            has_citations = bool(re.search(r'\[source\]|\[citation\]|according to', content, re.IGNORECASE))
            
            # Check for balanced reporting
            has_quotes = bool(re.search(r'"[^"]*"', content))
            
            # Calculate credibility score
            credibility_score = 0
            if has_author:
                credibility_score += 1
            if has_date:
                credibility_score += 1
            if has_citations:
                credibility_score += 1
            if has_quotes:
                credibility_score += 1
            
            # Map score to credibility level
            if credibility_score >= 3:
                return NewsCredibility.HIGH
            elif credibility_score >= 2:
                return NewsCredibility.MEDIUM
            else:
                return NewsCredibility.LOW
            
        except Exception as e:
            logger.error(f"Error assessing credibility: {e}")
            return NewsCredibility.UNKNOWN
    
    async def _generate_summary(self, content: str) -> str:
        """Generate article summary"""
        try:
            if not content:
                return ""
            
            # Simple extractive summarization
            sentences = sent_tokenize(content)
            
            if len(sentences) <= 3:
                return content
            
            # Use TF-IDF to score sentences
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences
            top_sentences = sorted(
                enumerate(sentence_scores),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Sort by original order
            top_sentences.sort(key=lambda x: x[0])
            
            # Create summary
            summary = ' '.join([sentences[i] for i, _ in top_sentences])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    async def _extract_lead_paragraph(self, content: str) -> str:
        """Extract lead paragraph"""
        try:
            if not content:
                return ""
            
            # Get first paragraph
            paragraphs = content.split('\n\n')
            if paragraphs:
                return paragraphs[0].strip()
            
            # If no paragraphs, get first sentence
            sentences = sent_tokenize(content)
            if sentences:
                return sentences[0]
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting lead paragraph: {e}")
            return ""
    
    async def _extract_conclusion(self, content: str) -> str:
        """Extract conclusion paragraph"""
        try:
            if not content:
                return ""
            
            # Get last paragraph
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                return paragraphs[-1].strip()
            
            # If no paragraphs, get last sentence
            sentences = sent_tokenize(content)
            if len(sentences) > 1:
                return sentences[-1]
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting conclusion: {e}")
            return ""
    
    async def _extract_quotes(self, content: str) -> List[str]:
        """Extract quotes from content"""
        try:
            if not content:
                return []
            
            # Find quoted text
            quotes = re.findall(r'"([^"]*)"', content)
            
            # Filter out very short quotes
            quotes = [quote for quote in quotes if len(quote) > 10]
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error extracting quotes: {e}")
            return []
    
    async def _extract_statistics(self, content: str) -> List[str]:
        """Extract statistics from content"""
        try:
            if not content:
                return []
            
            # Find percentage patterns
            percentages = re.findall(r'\d+\.?\d*%', content)
            
            # Find number patterns
            numbers = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', content)
            
            # Find ratio patterns
            ratios = re.findall(r'\d+:\d+', content)
            
            statistics = percentages + numbers + ratios
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error extracting statistics: {e}")
            return []
    
    async def _extract_locations(self, content: str) -> List[str]:
        """Extract locations from content"""
        try:
            if not content:
                return []
            
            # Use spaCy for location extraction
            doc = self.nlp_model(content)
            
            locations = []
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entity or location
                    locations.append(ent.text)
            
            return list(set(locations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting locations: {e}")
            return []
    
    async def _extract_people(self, content: str) -> List[str]:
        """Extract people from content"""
        try:
            if not content:
                return []
            
            # Use spaCy for person extraction
            doc = self.nlp_model(content)
            
            people = []
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    people.append(ent.text)
            
            return list(set(people))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting people: {e}")
            return []
    
    async def _extract_organizations(self, content: str) -> List[str]:
        """Extract organizations from content"""
        try:
            if not content:
                return []
            
            # Use spaCy for organization extraction
            doc = self.nlp_model(content)
            
            organizations = []
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    organizations.append(ent.text)
            
            return list(set(organizations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting organizations: {e}")
            return []
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        try:
            if not content:
                return []
            
            # Use keywords as topics
            keywords = await self._extract_keywords(content)
            
            # Also extract noun phrases
            doc = self.nlp_model(content)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Combine and deduplicate
            topics = list(set(keywords + noun_phrases))
            
            return topics[:20]  # Limit to top 20 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        try:
            if not content:
                return []
            
            # Use YAKE for key phrase extraction
            key_phrases = self.keyword_extractor.extract_keywords(content)
            
            # Extract top key phrases
            top_phrases = [phrase[1] for phrase in key_phrases[:15]]
            
            return top_phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    async def _extract_named_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities with detailed information"""
        try:
            if not content:
                return []
            
            # Use spaCy for named entity recognition
            doc = self.nlp_model(content)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_)
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting named entities: {e}")
            return []
    
    async def _analyze_detailed_sentiment(self, content: str) -> Dict[str, Any]:
        """Perform detailed sentiment analysis"""
        try:
            if not content:
                return {'overall': 'neutral', 'score': 0.0, 'breakdown': {}}
            
            # Analyze overall sentiment
            sentiment, score = await self._analyze_sentiment(content)
            
            # Analyze sentiment by sentences
            sentences = sent_tokenize(content)
            sentence_sentiments = []
            
            for sentence in sentences[:10]:  # Limit to first 10 sentences
                sent_sentiment, sent_score = await self._analyze_sentiment(sentence)
                sentence_sentiments.append({
                    'sentence': sentence,
                    'sentiment': sent_sentiment.value,
                    'score': sent_score
                })
            
            return {
                'overall': sentiment.value,
                'score': score,
                'breakdown': sentence_sentiments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing detailed sentiment: {e}")
            return {'overall': 'neutral', 'score': 0.0, 'breakdown': {}}
    
    async def _calculate_readability(self, content: str) -> float:
        """Calculate readability score"""
        try:
            if not content:
                return 0.0
            
            # Use TextBlob for readability calculation
            blob = TextBlob(content)
            
            # Simple readability score based on sentence and word length
            sentences = blob.sentences
            words = blob.words
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability formula
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            
            # Normalize to 0-1 scale
            readability = max(0, min(1, readability / 100))
            
            return readability
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.5
    
    async def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score"""
        try:
            if not content:
                return 0.0
            
            quality_score = 0.0
            
            # Length score (optimal length is 300-800 words)
            word_count = len(content.split())
            if 300 <= word_count <= 800:
                quality_score += 0.3
            elif 200 <= word_count < 300 or 800 < word_count <= 1200:
                quality_score += 0.2
            else:
                quality_score += 0.1
            
            # Structure score (has paragraphs)
            paragraphs = content.split('\n\n')
            if len(paragraphs) >= 3:
                quality_score += 0.2
            
            # Quote score (has quotes)
            quotes = re.findall(r'"[^"]*"', content)
            if quotes:
                quality_score += 0.1
            
            # Statistics score (has numbers/statistics)
            numbers = re.findall(r'\d+', content)
            if numbers:
                quality_score += 0.1
            
            # Named entities score (has people, places, organizations)
            entities = await self._extract_entities(content)
            if entities:
                quality_score += 0.1
            
            # Readability score
            readability = await self._calculate_readability(content)
            quality_score += readability * 0.2
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating content quality: {e}")
            return 0.5
    
    async def _calculate_relevance_score(self, metadata: NewsMetadata, content: NewsContent) -> float:
        """Calculate relevance score"""
        try:
            relevance_score = 0.0
            
            # Title relevance
            if metadata.title:
                relevance_score += 0.2
            
            # Content quality
            relevance_score += content.content_quality_score * 0.3
            
            # Entity richness
            if content.named_entities:
                relevance_score += min(0.2, len(content.named_entities) * 0.01)
            
            # Keyword density
            if content.key_phrases:
                relevance_score += min(0.2, len(content.key_phrases) * 0.01)
            
            # Credibility
            if metadata.credibility == NewsCredibility.HIGH:
                relevance_score += 0.1
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5
    
    async def _calculate_freshness_score(self, metadata: NewsMetadata) -> float:
        """Calculate freshness score"""
        try:
            if not metadata.published_date:
                return 0.0
            
            # Calculate age in hours
            age_hours = (datetime.now() - metadata.published_date).total_seconds() / 3600
            
            # Freshness score decreases with age
            if age_hours <= 1:
                return 1.0
            elif age_hours <= 6:
                return 0.9
            elif age_hours <= 24:
                return 0.7
            elif age_hours <= 72:
                return 0.5
            elif age_hours <= 168:  # 1 week
                return 0.3
            else:
                return 0.1
            
        except Exception as e:
            logger.error(f"Error calculating freshness score: {e}")
            return 0.5
    
    async def _calculate_quality_score(self, metadata: NewsMetadata, content: NewsContent) -> float:
        """Calculate overall quality score"""
        try:
            quality_score = 0.0
            
            # Content quality
            quality_score += content.content_quality_score * 0.4
            
            # Credibility
            if metadata.credibility == NewsCredibility.HIGH:
                quality_score += 0.3
            elif metadata.credibility == NewsCredibility.MEDIUM:
                quality_score += 0.2
            else:
                quality_score += 0.1
            
            # Completeness
            if metadata.author:
                quality_score += 0.1
            if metadata.published_date:
                quality_score += 0.1
            if content.summary:
                quality_score += 0.1
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    async def _calculate_trending_score(self, article_id: str) -> float:
        """Calculate trending score"""
        try:
            # This would typically query a database for social engagement metrics
            # For now, return a random score
            import random
            return random.uniform(0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating trending score: {e}")
            return 0.0
    
    async def _find_related_articles(self, article_id: str, content: NewsContent) -> List[str]:
        """Find related articles"""
        try:
            # This would typically use semantic similarity or shared entities
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error finding related articles: {e}")
            return []
    
    async def _get_social_engagement(self, article_data: Dict[str, Any]) -> Dict[str, int]:
        """Get social engagement metrics"""
        try:
            # This would typically query social media APIs
            # For now, return empty dict
            return {
                'shares': 0,
                'likes': 0,
                'comments': 0,
                'views': 0
            }
            
        except Exception as e:
            logger.error(f"Error getting social engagement: {e}")
            return {}
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string"""
        try:
            if not date_str:
                return None
            
            # Try different date formats
            try:
                return dateutil.parser.parse(date_str)
            except:
                # Try common formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%B %d, %Y',
                    '%b %d, %Y'
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
                
                return None
                
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return None
    
    async def _detect_language(self, content: str) -> str:
        """Detect content language"""
        try:
            if not content:
                return 'en'
            
            # Use TextBlob for language detection
            blob = TextBlob(content)
            language = blob.detect_language()
            
            return language if language else 'en'
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'en'
    
    async def _is_breaking_news(self, article_data: Dict[str, Any]) -> bool:
        """Check if article is breaking news"""
        try:
            title = article_data.get('title', '').lower()
            content = article_data.get('content', '').lower()
            
            breaking_keywords = [
                'breaking', 'urgent', 'alert', 'just in', 'developing',
                'live updates', 'emergency', 'crisis', 'disaster'
            ]
            
            for keyword in breaking_keywords:
                if keyword in title or keyword in content:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking breaking news: {e}")
            return False
    
    async def _is_opinion_piece(self, article_data: Dict[str, Any]) -> bool:
        """Check if article is an opinion piece"""
        try:
            title = article_data.get('title', '').lower()
            content = article_data.get('content', '').lower()
            
            opinion_keywords = [
                'opinion', 'editorial', 'commentary', 'analysis',
                'perspective', 'viewpoint', 'i think', 'i believe',
                'in my opinion', 'according to me'
            ]
            
            for keyword in opinion_keywords:
                if keyword in title or keyword in content:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking opinion piece: {e}")
            return False
    
    async def _is_satire(self, article_data: Dict[str, Any]) -> bool:
        """Check if article is satire"""
        try:
            title = article_data.get('title', '').lower()
            content = article_data.get('content', '').lower()
            
            satire_keywords = [
                'satire', 'parody', 'humor', 'comedy', 'joke',
                'fake news', 'not real', 'fictional'
            ]
            
            for keyword in satire_keywords:
                if keyword in title or keyword in content:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking satire: {e}")
            return False
    
    async def _is_press_release(self, article_data: Dict[str, Any]) -> bool:
        """Check if article is a press release"""
        try:
            title = article_data.get('title', '').lower()
            content = article_data.get('content', '').lower()
            
            press_release_keywords = [
                'press release', 'announces', 'announced', 'launches',
                'launched', 'introduces', 'introduced', 'partnership',
                'collaboration', 'agreement', 'contract'
            ]
            
            for keyword in press_release_keywords:
                if keyword in title or keyword in content:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking press release: {e}")
            return False
    
    async def _count_duplicates(self, content_hash: str) -> int:
        """Count duplicate articles"""
        try:
            # This would typically query a database
            # For now, return 0
            return 0
            
        except Exception as e:
            logger.error(f"Error counting duplicates: {e}")
            return 0
    
    def _generate_article_id(self, article_data: Dict[str, Any]) -> str:
        """Generate unique article ID"""
        try:
            url = article_data.get('url', '')
            if url:
                return hashlib.sha256(url.encode()).hexdigest()
            
            title = article_data.get('title', '')
            content = article_data.get('content', '')
            return hashlib.sha256((title + content).encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating article ID: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()