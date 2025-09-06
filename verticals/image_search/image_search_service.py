"""
T3SS Project - Image Search Service
Advanced image search functionality with similarity matching
(c) 2025 Qiss Labs. All Rights Reserved.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import io
import base64
import hashlib

import numpy as np
import faiss
from PIL import Image
import aiohttp
import asyncpg
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import torch
import clip

from .image_processor import ImageProcessor, ImageSearchResult, ImageMetadata, ImageFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Image search types"""
    SIMILARITY = "similarity"
    REVERSE = "reverse"
    SEMANTIC = "semantic"
    COLOR = "color"
    TEXT = "text"
    FACE = "face"
    OBJECT = "object"

class SortOrder(Enum):
    """Sort order options"""
    RELEVANCE = "relevance"
    DATE = "date"
    SIZE = "size"
    QUALITY = "quality"
    POPULARITY = "popularity"

@dataclass
class ImageSearchQuery:
    """Image search query structure"""
    query_type: SearchType
    query_text: str = ""
    query_image: bytes = b""
    query_url: str = ""
    filters: Dict[str, Any] = None
    sort_order: SortOrder = SortOrder.RELEVANCE
    limit: int = 20
    offset: int = 0
    user_id: str = ""
    session_id: str = ""

@dataclass
class ImageSearchResponse:
    """Image search response structure"""
    results: List[ImageSearchResult]
    total_count: int
    query_time: float
    suggestions: List[str]
    filters_applied: Dict[str, Any]
    next_offset: Optional[int] = None
    prev_offset: Optional[int] = None

class ImageSearchService:
    """Advanced image search service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.db_pool = None
        self.redis_client = None
        self.faiss_index = None
        self.sentence_transformer = None
        self.clip_model = None
        self.clip_preprocess = None
        self._initialize_services()
    
    async def _initialize_services(self):
        """Initialize database and search services"""
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                host=self.config['db_host'],
                port=self.config['db_port'],
                database=self.config['db_name'],
                user=self.config['db_user'],
                password=self.config['db_password'],
                min_size=5,
                max_size=20
            )
            
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config['redis_password'],
                db=self.config['redis_db']
            )
            
            # Initialize FAISS index
            self._initialize_faiss_index()
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('clip-ViT-B-32')
            
            # Initialize CLIP model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            
            logger.info("Image search service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing image search service: {e}")
            raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for similarity search"""
        try:
            # Create FAISS index for CLIP features (512 dimensions)
            dimension = 512
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Load existing index if available
            index_path = self.config.get('faiss_index_path', '/tmp/image_index.faiss')
            try:
                self.faiss_index = faiss.read_index(index_path)
                logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            except:
                logger.info("Creating new FAISS index")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise
    
    async def search_images(self, query: ImageSearchQuery) -> ImageSearchResponse:
        """Perform image search based on query"""
        try:
            start_time = time.time()
            
            # Validate query
            if not self._validate_query(query):
                raise ValueError("Invalid search query")
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached search result")
                return cached_result
            
            # Perform search based on query type
            if query.query_type == SearchType.SIMILARITY:
                results = await self._similarity_search(query)
            elif query.query_type == SearchType.REVERSE:
                results = await self._reverse_image_search(query)
            elif query.query_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query)
            elif query.query_type == SearchType.COLOR:
                results = await self._color_search(query)
            elif query.query_type == SearchType.TEXT:
                results = await self._text_search(query)
            elif query.query_type == SearchType.FACE:
                results = await self._face_search(query)
            elif query.query_type == SearchType.OBJECT:
                results = await self._object_search(query)
            else:
                raise ValueError(f"Unsupported search type: {query.query_type}")
            
            # Apply filters
            if query.filters:
                results = await self._apply_filters(results, query.filters)
            
            # Sort results
            results = await self._sort_results(results, query.sort_order)
            
            # Apply pagination
            total_count = len(results)
            results = results[query.offset:query.offset + query.limit]
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(query)
            
            # Calculate query time
            query_time = time.time() - start_time
            
            # Create response
            response = ImageSearchResponse(
                results=results,
                total_count=total_count,
                query_time=query_time,
                suggestions=suggestions,
                filters_applied=query.filters or {}
            )
            
            # Set pagination info
            if query.offset + query.limit < total_count:
                response.next_offset = query.offset + query.limit
            if query.offset > 0:
                response.prev_offset = max(0, query.offset - query.limit)
            
            # Cache result
            await self._cache_result(cache_key, response)
            
            # Log search
            await self._log_search(query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error performing image search: {e}")
            raise
    
    async def _similarity_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform similarity search using FAISS index"""
        try:
            # Get query image features
            if query.query_image:
                query_image = Image.open(io.BytesIO(query.query_image))
                features = await self.image_processor._extract_features(query_image)
                query_vector = features.clip_features.reshape(1, -1)
            elif query.query_url:
                # Download image from URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(query.query_url) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            query_image = Image.open(io.BytesIO(image_data))
                            features = await self.image_processor._extract_features(query_image)
                            query_vector = features.clip_features.reshape(1, -1)
                        else:
                            raise ValueError(f"Failed to download image from URL: {query.query_url}")
            else:
                raise ValueError("No query image provided for similarity search")
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_vector, min(1000, query.limit * 10))
            
            # Get results from database
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                # Get image data from database
                image_data = await self._get_image_by_index(idx)
                if image_data:
                    result = ImageSearchResult(
                        image_id=image_data['image_id'],
                        url=image_data['url'],
                        title=image_data['title'],
                        description=image_data['description'],
                        thumbnail_url=image_data['thumbnail_url'],
                        source_url=image_data['source_url'],
                        metadata=ImageMetadata(**image_data['metadata']),
                        features=ImageFeatures(**image_data['features']),
                        similarity_score=float(score),
                        relevance_score=float(score),
                        tags=image_data['tags'],
                        categories=image_data['categories'],
                        license=image_data['license'],
                        author=image_data['author'],
                        created_date=image_data['created_date'],
                        modified_date=image_data['modified_date']
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def _reverse_image_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform reverse image search"""
        try:
            # This is similar to similarity search but with additional processing
            results = await self._similarity_search(query)
            
            # Add reverse search specific features
            for result in results:
                # Calculate additional similarity metrics
                if query.query_image:
                    query_image = Image.open(io.BytesIO(query.query_image))
                    similarity = await self.image_processor.calculate_similarity(
                        query_image, 
                        Image.open(io.BytesIO(await self._get_image_data(result.image_id)))
                    )
                    result.similarity_score = similarity
            
            return results
            
        except Exception as e:
            logger.error(f"Error in reverse image search: {e}")
            return []
    
    async def _semantic_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform semantic search using text queries"""
        try:
            if not query.query_text:
                raise ValueError("No query text provided for semantic search")
            
            # Encode query text using CLIP
            text_tokens = clip.tokenize([query.query_text])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_vector = text_features.numpy()
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_vector, min(1000, query.limit * 10))
            
            # Get results from database
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                image_data = await self._get_image_by_index(idx)
                if image_data:
                    result = ImageSearchResult(
                        image_id=image_data['image_id'],
                        url=image_data['url'],
                        title=image_data['title'],
                        description=image_data['description'],
                        thumbnail_url=image_data['thumbnail_url'],
                        source_url=image_data['source_url'],
                        metadata=ImageMetadata(**image_data['metadata']),
                        features=ImageFeatures(**image_data['features']),
                        similarity_score=float(score),
                        relevance_score=float(score),
                        tags=image_data['tags'],
                        categories=image_data['categories'],
                        license=image_data['license'],
                        author=image_data['author'],
                        created_date=image_data['created_date'],
                        modified_date=image_data['modified_date']
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _color_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform color-based search"""
        try:
            # Extract color information from query
            if query.query_image:
                query_image = Image.open(io.BytesIO(query.query_image))
                query_colors = await self.image_processor._extract_dominant_colors(query_image)
            else:
                # Parse color from query text
                query_colors = self._parse_color_from_text(query.query_text)
            
            if not query_colors:
                return []
            
            # Search database for images with similar colors
            async with self.db_pool.acquire() as conn:
                query_sql = """
                    SELECT * FROM images 
                    WHERE metadata->>'dominant_colors' IS NOT NULL
                    ORDER BY (
                        SELECT MIN(
                            SQRT(
                                POWER((metadata->>'dominant_colors'->>0)::int - $1, 2) +
                                POWER((metadata->>'dominant_colors'->>1)::int - $2, 2) +
                                POWER((metadata->>'dominant_colors'->>2)::int - $3, 2)
                            )
                        )
                        FROM jsonb_array_elements(metadata->'dominant_colors')
                    )
                    LIMIT $4
                """
                
                results = []
                for color in query_colors:
                    rows = await conn.fetch(query_sql, color[0], color[1], color[2], query.limit)
                    for row in rows:
                        result = await self._row_to_image_result(row)
                        if result:
                            results.append(result)
                
                return results
            
        except Exception as e:
            logger.error(f"Error in color search: {e}")
            return []
    
    async def _text_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform text-based search"""
        try:
            if not query.query_text:
                return []
            
            # Search database for images with matching text
            async with self.db_pool.acquire() as conn:
                query_sql = """
                    SELECT * FROM images 
                    WHERE to_tsvector('english', title || ' ' || description || ' ' || array_to_string(tags, ' ')) 
                    @@ plainto_tsquery('english', $1)
                    ORDER BY ts_rank(
                        to_tsvector('english', title || ' ' || description || ' ' || array_to_string(tags, ' ')),
                        plainto_tsquery('english', $1)
                    ) DESC
                    LIMIT $2
                """
                
                rows = await conn.fetch(query_sql, query.query_text, query.limit)
                results = []
                for row in rows:
                    result = await self._row_to_image_result(row)
                    if result:
                        results.append(result)
                
                return results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    async def _face_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform face-based search"""
        try:
            if not query.query_image:
                return []
            
            # Detect faces in query image
            query_image = Image.open(io.BytesIO(query.query_image))
            query_faces = await self.image_processor.detect_faces(query_image)
            
            if not query_faces:
                return []
            
            # Search for images with similar face characteristics
            # This is a simplified implementation
            results = await self._similarity_search(query)
            
            # Filter results that contain faces
            face_results = []
            for result in results:
                # Check if result image has faces
                image_data = await self._get_image_data(result.image_id)
                if image_data:
                    result_image = Image.open(io.BytesIO(image_data))
                    result_faces = await self.image_processor.detect_faces(result_image)
                    if result_faces:
                        face_results.append(result)
            
            return face_results
            
        except Exception as e:
            logger.error(f"Error in face search: {e}")
            return []
    
    async def _object_search(self, query: ImageSearchQuery) -> List[ImageSearchResult]:
        """Perform object-based search"""
        try:
            # Use semantic search for object detection
            if query.query_text:
                # Enhance query with object-related terms
                enhanced_query = f"object {query.query_text} thing item"
                query.query_text = enhanced_query
            
            results = await self._semantic_search(query)
            
            # Filter results based on object detection confidence
            object_results = []
            for result in results:
                # This would typically use an object detection model
                # For now, we'll use the similarity score as a proxy
                if result.similarity_score > 0.3:
                    object_results.append(result)
            
            return object_results
            
        except Exception as e:
            logger.error(f"Error in object search: {e}")
            return []
    
    async def _apply_filters(self, results: List[ImageSearchResult], filters: Dict[str, Any]) -> List[ImageSearchResult]:
        """Apply filters to search results"""
        try:
            filtered_results = results.copy()
            
            # Filter by image type
            if 'image_type' in filters:
                image_type = filters['image_type']
                filtered_results = [r for r in filtered_results if r.metadata.image_type.value == image_type]
            
            # Filter by size
            if 'min_width' in filters:
                min_width = filters['min_width']
                filtered_results = [r for r in filtered_results if r.metadata.width >= min_width]
            
            if 'min_height' in filters:
                min_height = filters['min_height']
                filtered_results = [r for r in filtered_results if r.metadata.height >= min_height]
            
            # Filter by quality
            if 'min_quality' in filters:
                min_quality = filters['min_quality']
                filtered_results = [r for r in filtered_results if r.metadata.quality_score >= min_quality]
            
            # Filter by license
            if 'license' in filters:
                license_type = filters['license']
                filtered_results = [r for r in filtered_results if r.license == license_type]
            
            # Filter by date range
            if 'date_from' in filters or 'date_to' in filters:
                date_from = filters.get('date_from')
                date_to = filters.get('date_to')
                filtered_results = [r for r in filtered_results if self._is_date_in_range(r.created_date, date_from, date_to)]
            
            # Filter by color
            if 'color' in filters:
                target_color = filters['color']
                filtered_results = [r for r in filtered_results if self._has_similar_color(r.metadata.dominant_colors, target_color)]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return results
    
    async def _sort_results(self, results: List[ImageSearchResult], sort_order: SortOrder) -> List[ImageSearchResult]:
        """Sort search results"""
        try:
            if sort_order == SortOrder.RELEVANCE:
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
            elif sort_order == SortOrder.DATE:
                return sorted(results, key=lambda x: x.created_date, reverse=True)
            elif sort_order == SortOrder.SIZE:
                return sorted(results, key=lambda x: x.metadata.size_bytes, reverse=True)
            elif sort_order == SortOrder.QUALITY:
                return sorted(results, key=lambda x: x.metadata.quality_score, reverse=True)
            elif sort_order == SortOrder.POPULARITY:
                # This would typically use view count or download count
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
            else:
                return results
                
        except Exception as e:
            logger.error(f"Error sorting results: {e}")
            return results
    
    async def _generate_suggestions(self, query: ImageSearchQuery) -> List[str]:
        """Generate search suggestions"""
        try:
            suggestions = []
            
            # Get popular searches
            async with self.db_pool.acquire() as conn:
                popular_searches = await conn.fetch(
                    "SELECT query_text FROM search_logs WHERE query_type = 'image' GROUP BY query_text ORDER BY COUNT(*) DESC LIMIT 5"
                )
                suggestions.extend([row['query_text'] for row in popular_searches])
            
            # Get related tags
            if query.query_text:
                async with self.db_pool.acquire() as conn:
                    related_tags = await conn.fetch(
                        "SELECT DISTINCT unnest(tags) as tag FROM images WHERE tags && $1 LIMIT 5",
                        [query.query_text]
                    )
                    suggestions.extend([row['tag'] for row in related_tags])
            
            return suggestions[:10]  # Limit to 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ImageSearchResponse]:
        """Get cached search result"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return ImageSearchResponse(**json.loads(cached_data))
            return None
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: ImageSearchResponse):
        """Cache search result"""
        try:
            cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
            await self.redis_client.setex(
                cache_key, 
                cache_ttl, 
                json.dumps(asdict(result), default=str)
            )
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def _generate_cache_key(self, query: ImageSearchQuery) -> str:
        """Generate cache key for query"""
        query_data = {
            'type': query.query_type.value,
            'text': query.query_text,
            'url': query.query_url,
            'filters': query.filters,
            'sort': query.sort_order.value,
            'limit': query.limit,
            'offset': query.offset
        }
        if query.query_image:
            query_data['image_hash'] = hashlib.sha256(query.query_image).hexdigest()
        
        return f"image_search:{hashlib.sha256(json.dumps(query_data, sort_keys=True).encode()).hexdigest()}"
    
    async def _log_search(self, query: ImageSearchQuery, response: ImageSearchResponse):
        """Log search query and results"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO search_logs (query_type, query_text, query_data, result_count, query_time, user_id, session_id, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """,
                    'image',
                    query.query_text,
                    json.dumps(asdict(query), default=str),
                    response.total_count,
                    response.query_time,
                    query.user_id,
                    query.session_id
                )
        except Exception as e:
            logger.error(f"Error logging search: {e}")
    
    def _validate_query(self, query: ImageSearchQuery) -> bool:
        """Validate search query"""
        if query.query_type == SearchType.SIMILARITY and not query.query_image and not query.query_url:
            return False
        if query.query_type == SearchType.SEMANTIC and not query.query_text:
            return False
        if query.limit <= 0 or query.limit > 100:
            return False
        if query.offset < 0:
            return False
        return True
    
    def _parse_color_from_text(self, text: str) -> List[Tuple[int, int, int]]:
        """Parse color information from text"""
        # Simple color mapping
        color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128)
        }
        
        colors = []
        text_lower = text.lower()
        for color_name, color_value in color_map.items():
            if color_name in text_lower:
                colors.append(color_value)
        
        return colors
    
    def _is_date_in_range(self, date_str: str, date_from: str = None, date_to: str = None) -> bool:
        """Check if date is in specified range"""
        try:
            from datetime import datetime
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            if date_from:
                from_date = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                if date < from_date:
                    return False
            
            if date_to:
                to_date = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                if date > to_date:
                    return False
            
            return True
        except:
            return True
    
    def _has_similar_color(self, dominant_colors: List[Tuple[int, int, int]], target_color: str) -> bool:
        """Check if image has similar color to target"""
        try:
            target_rgb = self._parse_color_from_text(target_color)
            if not target_rgb:
                return True
            
            target_color = target_rgb[0]
            
            for color in dominant_colors:
                # Calculate color distance
                distance = sum((a - b) ** 2 for a, b in zip(color, target_color)) ** 0.5
                if distance < 100:  # Threshold for color similarity
                    return True
            
            return False
        except:
            return True
    
    async def _get_image_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get image data by FAISS index"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM images WHERE faiss_index = $1",
                    index
                )
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting image by index: {e}")
            return None
    
    async def _get_image_data(self, image_id: str) -> Optional[bytes]:
        """Get image data by ID"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT image_data FROM images WHERE image_id = $1",
                    image_id
                )
                return row['image_data'] if row else None
        except Exception as e:
            logger.error(f"Error getting image data: {e}")
            return None
    
    async def _row_to_image_result(self, row: Dict[str, Any]) -> Optional[ImageSearchResult]:
        """Convert database row to ImageSearchResult"""
        try:
            return ImageSearchResult(
                image_id=row['image_id'],
                url=row['url'],
                title=row['title'],
                description=row['description'],
                thumbnail_url=row['thumbnail_url'],
                source_url=row['source_url'],
                metadata=ImageMetadata(**row['metadata']),
                features=ImageFeatures(**row['features']),
                similarity_score=0.0,
                relevance_score=0.0,
                tags=row['tags'],
                categories=row['categories'],
                license=row['license'],
                author=row['author'],
                created_date=row['created_date'],
                modified_date=row['modified_date']
            )
        except Exception as e:
            logger.error(f"Error converting row to image result: {e}")
            return None
    
    async def add_image(self, image_data: bytes, metadata: Dict[str, Any]) -> str:
        """Add image to search index"""
        try:
            # Process image
            result = await self.image_processor.process_image(image_data, metadata.get('url'))
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                # Insert image record
                image_id = await conn.fetchval(
                    """
                    INSERT INTO images (image_id, url, title, description, thumbnail_url, source_url, 
                                      metadata, features, tags, categories, license, author, 
                                      created_date, modified_date, image_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING image_id
                    """,
                    result.image_id,
                    result.url,
                    metadata.get('title', ''),
                    metadata.get('description', ''),
                    metadata.get('thumbnail_url', ''),
                    result.source_url,
                    asdict(result.metadata),
                    asdict(result.features),
                    metadata.get('tags', []),
                    metadata.get('categories', []),
                    metadata.get('license', ''),
                    metadata.get('author', ''),
                    metadata.get('created_date', ''),
                    metadata.get('modified_date', ''),
                    image_data
                )
                
                # Add to FAISS index
                faiss_index = self.faiss_index.ntotal
                self.faiss_index.add(result.features.clip_features.reshape(1, -1))
                
                # Update database with FAISS index
                await conn.execute(
                    "UPDATE images SET faiss_index = $1 WHERE image_id = $2",
                    faiss_index,
                    image_id
                )
                
                # Save FAISS index
                index_path = self.config.get('faiss_index_path', '/tmp/image_index.faiss')
                faiss.write_index(self.faiss_index, index_path)
                
                return image_id
                
        except Exception as e:
            logger.error(f"Error adding image: {e}")
            raise
    
    async def remove_image(self, image_id: str):
        """Remove image from search index"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get FAISS index
                row = await conn.fetchrow(
                    "SELECT faiss_index FROM images WHERE image_id = $1",
                    image_id
                )
                
                if row:
                    faiss_index = row['faiss_index']
                    
                    # Remove from database
                    await conn.execute(
                        "DELETE FROM images WHERE image_id = $1",
                        image_id
                    )
                    
                    # Remove from FAISS index (this is complex and may require rebuilding)
                    # For now, we'll mark it as removed
                    logger.warning(f"Image {image_id} removed from database but FAISS index may need rebuilding")
                
        except Exception as e:
            logger.error(f"Error removing image: {e}")
            raise
    
    async def get_image_by_id(self, image_id: str) -> Optional[ImageSearchResult]:
        """Get image by ID"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM images WHERE image_id = $1",
                    image_id
                )
                
                if row:
                    return await self._row_to_image_result(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting image by ID: {e}")
            return None
    
    async def get_trending_images(self, limit: int = 20) -> List[ImageSearchResult]:
        """Get trending images"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT i.* FROM images i
                    JOIN search_logs sl ON i.image_id = sl.result_data->>'image_id'
                    WHERE sl.created_at > NOW() - INTERVAL '7 days'
                    GROUP BY i.image_id
                    ORDER BY COUNT(*) DESC
                    LIMIT $1
                    """,
                    limit
                )
                
                results = []
                for row in rows:
                    result = await self._row_to_image_result(row)
                    if result:
                        results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting trending images: {e}")
            return []
    
    async def get_related_images(self, image_id: str, limit: int = 10) -> List[ImageSearchResult]:
        """Get related images"""
        try:
            # Get the target image
            target_image = await self.get_image_by_id(image_id)
            if not target_image:
                return []
            
            # Perform similarity search
            query = ImageSearchQuery(
                query_type=SearchType.SIMILARITY,
                query_image=await self._get_image_data(image_id),
                limit=limit + 1  # +1 to exclude the original image
            )
            
            results = await self._similarity_search(query)
            
            # Remove the original image from results
            related_results = [r for r in results if r.image_id != image_id]
            
            return related_results[:limit]
            
        except Exception as e:
            logger.error(f"Error getting related images: {e}")
            return []
    
    async def close(self):
        """Close connections and cleanup"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            # Save FAISS index
            if self.faiss_index:
                index_path = self.config.get('faiss_index_path', '/tmp/image_index.faiss')
                faiss.write_index(self.faiss_index, index_path)
            
            logger.info("Image search service closed")
            
        except Exception as e:
            logger.error(f"Error closing image search service: {e}")