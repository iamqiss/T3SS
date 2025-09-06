"""
T3SS Project - Image Search Vertical
Advanced image processing and analysis for search functionality
(c) 2025 Qiss Labs. All Rights Reserved.
"""

import asyncio
import hashlib
import io
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

import aiohttp
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import clip
import faiss
from sentence_transformers import SentenceTransformer
import exifread
import imagehash
from skimage import feature, color, segmentation
from skimage.metrics import structural_similarity as ssim
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageType(Enum):
    """Image type classification"""
    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    DIAGRAM = "diagram"
    CHART = "chart"
    LOGO = "logo"
    SCREENSHOT = "screenshot"
    MEME = "meme"
    GIF = "gif"
    ANIMATED = "animated"

class ImageQuality(Enum):
    """Image quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ImageMetadata:
    """Image metadata structure"""
    width: int
    height: int
    format: str
    mode: str
    size_bytes: int
    dpi: Tuple[int, int]
    color_space: str
    has_transparency: bool
    exif_data: Dict[str, Any]
    dominant_colors: List[Tuple[int, int, int]]
    quality_score: float
    image_type: ImageType
    content_hash: str
    perceptual_hash: str
    average_hash: str
    dhash: str
    phash: str
    whash: str

@dataclass
class ImageFeatures:
    """Image feature vectors"""
    resnet_features: np.ndarray
    efficientnet_features: np.ndarray
    clip_features: np.ndarray
    color_histogram: np.ndarray
    texture_features: np.ndarray
    edge_features: np.ndarray
    sift_features: np.ndarray
    orb_features: np.ndarray
    semantic_embedding: np.ndarray

@dataclass
class ImageSearchResult:
    """Image search result"""
    image_id: str
    url: str
    title: str
    description: str
    thumbnail_url: str
    source_url: str
    metadata: ImageMetadata
    features: ImageFeatures
    similarity_score: float
    relevance_score: float
    tags: List[str]
    categories: List[str]
    license: str
    author: str
    created_date: str
    modified_date: str

class ImageProcessor:
    """Advanced image processing and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.feature_extractors = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for feature extraction"""
        try:
            # Load ResNet50 for general image features
            self.models['resnet'] = resnet50(pretrained=True)
            self.models['resnet'].eval()
            self.models['resnet'].to(self.device)
            
            # Load EfficientNet for efficient feature extraction
            self.models['efficientnet'] = efficientnet_b0(pretrained=True)
            self.models['efficientnet'].eval()
            self.models['efficientnet'].to(self.device)
            
            # Load CLIP for semantic understanding
            self.models['clip'], self.models['clip_preprocess'] = clip.load("ViT-B/32", device=self.device)
            
            # Load SentenceTransformer for text-image similarity
            self.feature_extractors['sentence_transformer'] = SentenceTransformer('clip-ViT-B-32')
            
            # Initialize SIFT and ORB feature detectors
            self.feature_extractors['sift'] = cv2.SIFT_create()
            self.feature_extractors['orb'] = cv2.ORB_create()
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def process_image(self, image_data: bytes, url: str = None) -> ImageSearchResult:
        """Process image and extract all features"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Extract metadata
            metadata = await self._extract_metadata(image, image_data)
            
            # Extract features
            features = await self._extract_features(image)
            
            # Generate image ID
            image_id = self._generate_image_id(image_data, url)
            
            # Create search result
            result = ImageSearchResult(
                image_id=image_id,
                url=url or "",
                title="",
                description="",
                thumbnail_url="",
                source_url=url or "",
                metadata=metadata,
                features=features,
                similarity_score=0.0,
                relevance_score=0.0,
                tags=[],
                categories=[],
                license="",
                author="",
                created_date="",
                modified_date=""
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    async def _extract_metadata(self, image: Image.Image, image_data: bytes) -> ImageMetadata:
        """Extract comprehensive image metadata"""
        try:
            # Basic image properties
            width, height = image.size
            format_name = image.format or "unknown"
            mode = image.mode
            size_bytes = len(image_data)
            
            # DPI information
            dpi = image.info.get('dpi', (72, 72))
            
            # Color space
            color_space = self._detect_color_space(image)
            
            # Transparency
            has_transparency = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            
            # EXIF data
            exif_data = await self._extract_exif_data(image_data)
            
            # Dominant colors
            dominant_colors = await self._extract_dominant_colors(image)
            
            # Quality assessment
            quality_score = await self._assess_image_quality(image)
            
            # Image type classification
            image_type = await self._classify_image_type(image)
            
            # Generate hashes
            content_hash = hashlib.sha256(image_data).hexdigest()
            perceptual_hash = str(imagehash.phash(image))
            average_hash = str(imagehash.average_hash(image))
            dhash = str(imagehash.dhash(image))
            phash = str(imagehash.phash(image))
            whash = str(imagehash.whash(image))
            
            return ImageMetadata(
                width=width,
                height=height,
                format=format_name,
                mode=mode,
                size_bytes=size_bytes,
                dpi=dpi,
                color_space=color_space,
                has_transparency=has_transparency,
                exif_data=exif_data,
                dominant_colors=dominant_colors,
                quality_score=quality_score,
                image_type=image_type,
                content_hash=content_hash,
                perceptual_hash=perceptual_hash,
                average_hash=average_hash,
                dhash=dhash,
                phash=phash,
                whash=whash
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            raise
    
    async def _extract_features(self, image: Image.Image) -> ImageFeatures:
        """Extract comprehensive image features"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # ResNet features
            resnet_features = await self._extract_resnet_features(image)
            
            # EfficientNet features
            efficientnet_features = await self._extract_efficientnet_features(image)
            
            # CLIP features
            clip_features = await self._extract_clip_features(image)
            
            # Color histogram
            color_histogram = await self._extract_color_histogram(image)
            
            # Texture features
            texture_features = await self._extract_texture_features(image)
            
            # Edge features
            edge_features = await self._extract_edge_features(image)
            
            # SIFT features
            sift_features = await self._extract_sift_features(image)
            
            # ORB features
            orb_features = await self._extract_orb_features(image)
            
            # Semantic embedding
            semantic_embedding = await self._extract_semantic_embedding(image)
            
            return ImageFeatures(
                resnet_features=resnet_features,
                efficientnet_features=efficientnet_features,
                clip_features=clip_features,
                color_histogram=color_histogram,
                texture_features=texture_features,
                edge_features=edge_features,
                sift_features=sift_features,
                orb_features=orb_features,
                semantic_embedding=semantic_embedding
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    async def _extract_resnet_features(self, image: Image.Image) -> np.ndarray:
        """Extract ResNet50 features"""
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.models['resnet'](input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting ResNet features: {e}")
            return np.zeros(1000)  # ResNet50 output size
    
    async def _extract_efficientnet_features(self, image: Image.Image) -> np.ndarray:
        """Extract EfficientNet features"""
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.models['efficientnet'](input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting EfficientNet features: {e}")
            return np.zeros(1000)  # EfficientNet output size
    
    async def _extract_clip_features(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP features"""
        try:
            # Preprocess image
            input_tensor = self.models['clip_preprocess'](image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.models['clip'].encode_image(input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting CLIP features: {e}")
            return np.zeros(512)  # CLIP output size
    
    async def _extract_color_histogram(self, image: Image.Image) -> np.ndarray:
        """Extract color histogram features"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate histograms for each channel
            hist_r = np.histogram(img_array[:, :, 0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:, :, 1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:, :, 2], bins=32, range=(0, 256))[0]
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            # Concatenate histograms
            color_histogram = np.concatenate([hist_r, hist_g, hist_b])
            
            return color_histogram
            
        except Exception as e:
            logger.error(f"Error extracting color histogram: {e}")
            return np.zeros(96)  # 32 * 3 channels
    
    async def _extract_texture_features(self, image: Image.Image) -> np.ndarray:
        """Extract texture features using Local Binary Patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Extract LBP features
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            return hist
            
        except Exception as e:
            logger.error(f"Error extracting texture features: {e}")
            return np.zeros(10)
    
    async def _extract_edge_features(self, image: Image.Image) -> np.ndarray:
        """Extract edge features using Canny edge detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate edge orientation histogram
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(sobel_y, sobel_x)
            
            hist, _ = np.histogram(orientation, bins=8, range=(-np.pi, np.pi))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Combine features
            edge_features = np.concatenate([[edge_density], hist])
            
            return edge_features
            
        except Exception as e:
            logger.error(f"Error extracting edge features: {e}")
            return np.zeros(9)
    
    async def _extract_sift_features(self, image: Image.Image) -> np.ndarray:
        """Extract SIFT features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Detect and compute SIFT features
            keypoints, descriptors = self.feature_extractors['sift'].detectAndCompute(gray, None)
            
            if descriptors is not None:
                # Use mean of descriptors as feature vector
                sift_features = np.mean(descriptors, axis=0)
            else:
                sift_features = np.zeros(128)  # SIFT descriptor size
            
            return sift_features
            
        except Exception as e:
            logger.error(f"Error extracting SIFT features: {e}")
            return np.zeros(128)
    
    async def _extract_orb_features(self, image: Image.Image) -> np.ndarray:
        """Extract ORB features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Detect and compute ORB features
            keypoints, descriptors = self.feature_extractors['orb'].detectAndCompute(gray, None)
            
            if descriptors is not None:
                # Use mean of descriptors as feature vector
                orb_features = np.mean(descriptors, axis=0)
            else:
                orb_features = np.zeros(32)  # ORB descriptor size
            
            return orb_features
            
        except Exception as e:
            logger.error(f"Error extracting ORB features: {e}")
            return np.zeros(32)
    
    async def _extract_semantic_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract semantic embedding using CLIP"""
        try:
            # Use CLIP features as semantic embedding
            clip_features = await self._extract_clip_features(image)
            return clip_features
            
        except Exception as e:
            logger.error(f"Error extracting semantic embedding: {e}")
            return np.zeros(512)
    
    async def _extract_exif_data(self, image_data: bytes) -> Dict[str, Any]:
        """Extract EXIF data from image"""
        try:
            exif_data = {}
            
            # Parse EXIF data
            tags = exifread.process_file(io.BytesIO(image_data))
            
            for tag in tags.keys():
                if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
                    exif_data[tag] = str(tags[tag])
            
            return exif_data
            
        except Exception as e:
            logger.error(f"Error extracting EXIF data: {e}")
            return {}
    
    async def _extract_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image"""
        try:
            # Resize image for faster processing
            image = image.resize((150, 150))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use K-means clustering to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in dominant_colors]
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return [(0, 0, 0)] * num_colors
    
    async def _assess_image_quality(self, image: Image.Image) -> float:
        """Assess image quality using multiple metrics"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate sharpness using Laplacian variance
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Calculate noise level
            noise = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            
            # Normalize metrics
            sharpness_score = min(sharpness / 1000, 1.0)  # Normalize sharpness
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal brightness around 128
            contrast_score = min(contrast / 64, 1.0)  # Normalize contrast
            noise_score = max(0, 1.0 - noise / 100)  # Lower noise is better
            
            # Combine scores
            quality_score = (sharpness_score + brightness_score + contrast_score + noise_score) / 4
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return 0.5
    
    async def _classify_image_type(self, image: Image.Image) -> ImageType:
        """Classify image type using heuristics and ML"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check for transparency (likely illustration or logo)
            if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                return ImageType.ILLUSTRATION
            
            # Check aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            # Check for common screenshot characteristics
            if aspect_ratio > 1.5 and width > 1000:
                return ImageType.SCREENSHOT
            
            # Check for chart/diagram characteristics
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.1:  # High edge density suggests diagram/chart
                return ImageType.DIAGRAM
            
            # Check for logo characteristics (small, centered, high contrast)
            if width < 500 and height < 500:
                contrast = np.std(gray)
                if contrast > 50:
                    return ImageType.LOGO
            
            # Default to photo
            return ImageType.PHOTO
            
        except Exception as e:
            logger.error(f"Error classifying image type: {e}")
            return ImageType.PHOTO
    
    def _detect_color_space(self, image: Image.Image) -> str:
        """Detect color space of image"""
        if image.mode == 'RGB':
            return 'sRGB'
        elif image.mode == 'RGBA':
            return 'sRGB with Alpha'
        elif image.mode == 'CMYK':
            return 'CMYK'
        elif image.mode == 'LAB':
            return 'LAB'
        elif image.mode == 'HSV':
            return 'HSV'
        else:
            return 'Unknown'
    
    def _generate_image_id(self, image_data: bytes, url: str = None) -> str:
        """Generate unique image ID"""
        if url:
            return hashlib.sha256(url.encode()).hexdigest()
        else:
            return hashlib.sha256(image_data).hexdigest()
    
    async def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """Create thumbnail of image"""
        try:
            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create square thumbnail with padding if needed
            if thumbnail.size != size:
                new_image = Image.new('RGB', size, (255, 255, 255))
                new_image.paste(thumbnail, ((size[0] - thumbnail.size[0]) // 2, 
                                          (size[1] - thumbnail.size[1]) // 2))
                thumbnail = new_image
            
            return thumbnail
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return image
    
    async def detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        try:
            # Convert to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Convert to list of dictionaries
            face_data = []
            for (x, y, w, h) in faces:
                face_data.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 1.0  # OpenCV doesn't provide confidence scores
                })
            
            return face_data
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    async def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # This would typically use Tesseract or similar OCR engine
            # For now, return empty string
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    async def calculate_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """Calculate similarity between two images"""
        try:
            # Extract features from both images
            features1 = await self._extract_features(image1)
            features2 = await self._extract_features(image2)
            
            # Calculate cosine similarity using CLIP features
            similarity = np.dot(features1.clip_features, features2.clip_features) / (
                np.linalg.norm(features1.clip_features) * np.linalg.norm(features2.clip_features)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0