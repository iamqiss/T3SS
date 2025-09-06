# T3SS Project
# File: core/indexing/crawler/parser/pdf_parser.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import hashlib
import json

# PDF processing libraries
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pdfminer
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextBox, LTTextLine
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage

# Image processing for OCR
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Document analysis
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Async processing
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class PDFMetadata:
    """PDF document metadata"""
    title: str = ""
    author: str = ""
    subject: str = ""
    creator: str = ""
    producer: str = ""
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    pdf_version: str = ""
    encryption: bool = False
    linearized: bool = False
    tagged: bool = False
    custom_properties: Dict[str, str] = field(default_factory=dict)

@dataclass
class PDFPageInfo:
    """Information about a PDF page"""
    page_number: int
    width: float
    height: float
    rotation: int
    text_content: str = ""
    image_count: int = 0
    annotation_count: int = 0
    form_field_count: int = 0
    reading_order: List[str] = field(default_factory=list)
    layout_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PDFStructure:
    """PDF document structure analysis"""
    has_outline: bool = False
    outline_items: List[Dict[str, Any]] = field(default_factory=list)
    has_bookmarks: bool = False
    bookmark_count: int = 0
    has_forms: bool = False
    form_field_count: int = 0
    has_annotations: bool = False
    annotation_count: int = 0
    has_images: bool = False
    image_count: int = 0
    has_tables: bool = False
    table_count: int = 0
    reading_order: List[str] = field(default_factory=list)
    toc_structure: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PDFContent:
    """Extracted PDF content"""
    raw_text: str = ""
    structured_text: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)

@dataclass
class PDFAnalysis:
    """Comprehensive PDF analysis results"""
    metadata: PDFMetadata
    structure: PDFStructure
    content: PDFContent
    pages: List[PDFPageInfo]
    
    # Text analysis
    language: str = "en"
    readability_score: float = 0.0
    complexity_score: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    
    # Quality metrics
    text_quality: float = 0.0
    structure_quality: float = 0.0
    overall_quality: float = 0.0
    
    # Processing info
    processing_time: float = 0.0
    extraction_method: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class PDFParser:
    """Advanced PDF parser with comprehensive content extraction and analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        self._init_nlp_models()
        
        # Processing options
        self.enable_ocr = self.config.get('enable_ocr', True)
        self.enable_image_extraction = self.config.get('enable_image_extraction', True)
        self.enable_table_extraction = self.config.get('enable_table_extraction', True)
        self.enable_structure_analysis = self.config.get('enable_structure_analysis', True)
        self.enable_text_analysis = self.config.get('enable_text_analysis', True)
        
        # Performance settings
        self.max_pages = self.config.get('max_pages', 1000)
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.thread_pool_size = self.config.get('thread_pool_size', 4)
        self.process_pool_size = self.config.get('process_pool_size', 2)
        
        # Quality thresholds
        self.min_text_quality = self.config.get('min_text_quality', 0.3)
        self.min_image_quality = self.config.get('min_image_quality', 0.5)
        
        # Initialize processing pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=self.process_pool_size)
        
    def _init_nlp_models(self):
        """Initialize NLP models for text analysis"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy English model not found, using basic tokenization")
                self.nlp = None
            
            # Initialize stemmer
            self.stemmer = PorterStemmer()
            
            # Get stop words
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP models: {e}")
            self.nlp = None
            self.stemmer = None
            self.stop_words = set()
    
    async def parse_pdf(self, file_path: Union[str, Path, bytes]) -> PDFAnalysis:
        """Parse PDF file and return comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Validate input
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                
                file_size = file_path.stat().st_size
                if file_size > self.max_file_size:
                    raise ValueError(f"PDF file too large: {file_size} bytes")
                
                # Read file
                async with aiofiles.open(file_path, 'rb') as f:
                    pdf_data = await f.read()
            else:
                pdf_data = file_path
                file_size = len(pdf_data)
            
            # Create analysis object
            analysis = PDFAnalysis(
                metadata=PDFMetadata(),
                structure=PDFStructure(),
                content=PDFContent(),
                pages=[]
            )
            
            # Extract metadata
            analysis.metadata = await self._extract_metadata(pdf_data, file_size)
            
            # Extract structure
            if self.enable_structure_analysis:
                analysis.structure = await self._extract_structure(pdf_data)
            
            # Extract content
            analysis.content = await self._extract_content(pdf_data)
            
            # Extract page information
            analysis.pages = await self._extract_pages(pdf_data)
            
            # Perform text analysis
            if self.enable_text_analysis:
                await self._analyze_text(analysis)
            
            # Calculate quality metrics
            await self._calculate_quality_metrics(analysis)
            
            # Set processing info
            analysis.processing_time = time.time() - start_time
            analysis.extraction_method = "comprehensive"
            
            self.logger.info(f"Successfully parsed PDF in {analysis.processing_time:.2f}s")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to parse PDF: {e}")
            raise
    
    async def _extract_metadata(self, pdf_data: bytes, file_size: int) -> PDFMetadata:
        """Extract PDF metadata"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            metadata = PDFMetadata()
            metadata.file_size = file_size
            metadata.page_count = len(pdf_reader.pages)
            
            # Extract document info
            if pdf_reader.metadata:
                info = pdf_reader.metadata
                metadata.title = info.get('/Title', '') or ''
                metadata.author = info.get('/Author', '') or ''
                metadata.subject = info.get('/Subject', '') or ''
                metadata.creator = info.get('/Creator', '') or ''
                metadata.producer = info.get('/Producer', '') or ''
                metadata.creation_date = str(info.get('/CreationDate', ''))
                metadata.modification_date = str(info.get('/ModDate', ''))
            
            # Extract PDF version
            metadata.pdf_version = pdf_reader.pdf_version
            
            # Check for encryption
            metadata.encryption = pdf_reader.is_encrypted
            
            # Check for linearization
            metadata.linearized = pdf_reader.is_linearized
            
            # Extract custom properties
            if hasattr(pdf_reader, 'custom_properties'):
                metadata.custom_properties = pdf_reader.custom_properties
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata: {e}")
            return PDFMetadata(file_size=file_size)
    
    async def _extract_structure(self, pdf_data: bytes) -> PDFStructure:
        """Extract PDF structure information"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            structure = PDFStructure()
            
            # Check for outline/bookmarks
            outline = pdf_document.get_toc()
            if outline:
                structure.has_outline = True
                structure.outline_items = outline
                structure.bookmark_count = len(outline)
            
            # Check for forms
            form_fields = pdf_document.get_form_fields()
            if form_fields:
                structure.has_forms = True
                structure.form_field_count = len(form_fields)
            
            # Check for annotations
            annotation_count = 0
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                annotations = page.get_annotations()
                annotation_count += len(annotations)
            
            if annotation_count > 0:
                structure.has_annotations = True
                structure.annotation_count = annotation_count
            
            # Check for images
            image_count = 0
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images()
                image_count += len(image_list)
            
            if image_count > 0:
                structure.has_images = True
                structure.image_count = image_count
            
            # Extract table of contents structure
            structure.toc_structure = self._extract_toc_structure(outline)
            
            pdf_document.close()
            return structure
            
        except Exception as e:
            self.logger.error(f"Failed to extract structure: {e}")
            return PDFStructure()
    
    async def _extract_content(self, pdf_data: bytes) -> PDFContent:
        """Extract comprehensive PDF content"""
        try:
            content = PDFContent()
            
            # Extract text using multiple methods
            content.raw_text = await self._extract_text_comprehensive(pdf_data)
            
            # Extract structured text
            content.structured_text = await self._extract_structured_text(pdf_data)
            
            # Extract images if enabled
            if self.enable_image_extraction:
                content.images = await self._extract_images(pdf_data)
            
            # Extract tables if enabled
            if self.enable_table_extraction:
                content.tables = await self._extract_tables(pdf_data)
            
            # Extract links
            content.links = await self._extract_links(pdf_data)
            
            # Extract forms
            content.forms = await self._extract_forms(pdf_data)
            
            # Extract annotations
            content.annotations = await self._extract_annotations(pdf_data)
            
            # Process text into blocks, paragraphs, sentences, and words
            await self._process_text_content(content)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to extract content: {e}")
            return PDFContent()
    
    async def _extract_text_comprehensive(self, pdf_data: bytes) -> str:
        """Extract text using multiple methods for best results"""
        try:
            # Method 1: PyMuPDF (fitz)
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            text_parts = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            pdf_document.close()
            
            if text_parts:
                return '\n'.join(text_parts)
            
            # Method 2: pdfplumber (fallback)
            pdf_stream.seek(0)
            with pdfplumber.open(pdf_stream) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                if text_parts:
                    return '\n'.join(text_parts)
            
            # Method 3: pdfminer (fallback)
            pdf_stream.seek(0)
            text = extract_text(pdf_stream)
            return text or ""
            
        except Exception as e:
            self.logger.error(f"Failed to extract text: {e}")
            return ""
    
    async def _extract_structured_text(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract text with structure information"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            structured_text = []
            
            with pdfplumber.open(pdf_stream) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with layout information
                    text_objects = page.extract_text_simple()
                    
                    if text_objects:
                        structured_text.append({
                            'page': page_num + 1,
                            'text': text_objects,
                            'bbox': page.bbox,
                            'rotation': page.rotation
                        })
                    
                    # Extract text blocks
                    text_blocks = page.extract_text_blocks()
                    for block in text_blocks:
                        if block.get('text'):
                            structured_text.append({
                                'page': page_num + 1,
                                'text': block['text'],
                                'bbox': block['bbox'],
                                'font_size': block.get('font_size'),
                                'font_name': block.get('font_name'),
                                'type': 'text_block'
                            })
            
            return structured_text
            
        except Exception as e:
            self.logger.error(f"Failed to extract structured text: {e}")
            return []
    
    async def _extract_images(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            images = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Calculate image hash
                            img_hash = hashlib.md5(img_data).hexdigest()
                            
                            images.append({
                                'page': page_num + 1,
                                'index': img_index,
                                'xref': xref,
                                'width': pix.width,
                                'height': pix.height,
                                'colorspace': pix.colorspace.name if pix.colorspace else 'unknown',
                                'size': len(img_data),
                                'hash': img_hash,
                                'data': img_data,
                                'bbox': page.get_image_bbox(xref)
                            })
                        
                        pix = None
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
            
            pdf_document.close()
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to extract images: {e}")
            return []
    
    async def _extract_tables(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            tables = []
            
            with pdfplumber.open(pdf_stream) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_index, table in enumerate(page_tables):
                        if table:
                            tables.append({
                                'page': page_num + 1,
                                'index': table_index,
                                'data': table,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0,
                                'bbox': page.bbox
                            })
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to extract tables: {e}")
            return []
    
    async def _extract_links(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract links from PDF"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            links = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                link_list = page.get_links()
                
                for link in link_list:
                    links.append({
                        'page': page_num + 1,
                        'url': link.get('uri', ''),
                        'bbox': link.get('rect'),
                        'type': link.get('kind', 'unknown')
                    })
            
            pdf_document.close()
            return links
            
        except Exception as e:
            self.logger.error(f"Failed to extract links: {e}")
            return []
    
    async def _extract_forms(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract form fields from PDF"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            forms = []
            form_fields = pdf_document.get_form_fields()
            
            for field_name, field_info in form_fields.items():
                forms.append({
                    'name': field_name,
                    'type': field_info.get('field_type', 'unknown'),
                    'value': field_info.get('field_value', ''),
                    'bbox': field_info.get('rect'),
                    'page': field_info.get('page', 0)
                })
            
            pdf_document.close()
            return forms
            
        except Exception as e:
            self.logger.error(f"Failed to extract forms: {e}")
            return []
    
    async def _extract_annotations(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract annotations from PDF"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            annotations = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                page_annotations = page.get_annotations()
                
                for ann in page_annotations:
                    annotations.append({
                        'page': page_num + 1,
                        'type': ann.get('type', 'unknown'),
                        'content': ann.get('content', ''),
                        'bbox': ann.get('rect'),
                        'author': ann.get('author', ''),
                        'created': ann.get('creationDate', '')
                    })
            
            pdf_document.close()
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to extract annotations: {e}")
            return []
    
    async def _extract_pages(self, pdf_data: bytes) -> List[PDFPageInfo]:
        """Extract detailed page information"""
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            
            pages = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract page text
                text_content = page.get_text()
                
                # Count images, annotations, and form fields
                image_count = len(page.get_images())
                annotation_count = len(page.get_annotations())
                form_field_count = len(page.get_form_fields())
                
                page_info = PDFPageInfo(
                    page_number=page_num + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    rotation=page.rotation,
                    text_content=text_content,
                    image_count=image_count,
                    annotation_count=annotation_count,
                    form_field_count=form_field_count
                )
                
                pages.append(page_info)
            
            pdf_document.close()
            return pages
            
        except Exception as e:
            self.logger.error(f"Failed to extract pages: {e}")
            return []
    
    async def _process_text_content(self, content: PDFContent):
        """Process extracted text into structured components"""
        try:
            if not content.raw_text:
                return
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in content.raw_text.split('\n\n') if p.strip()]
            content.paragraphs = paragraphs
            
            # Split into sentences
            sentences = sent_tokenize(content.raw_text)
            content.sentences = sentences
            
            # Split into words
            words = word_tokenize(content.raw_text.lower())
            content.words = words
            
            # Extract text blocks with layout information
            content.text_blocks = self._extract_text_blocks(content.raw_text)
            
        except Exception as e:
            self.logger.error(f"Failed to process text content: {e}")
    
    def _extract_text_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract text blocks with layout analysis"""
        try:
            blocks = []
            lines = text.split('\n')
            
            current_block = []
            current_font_size = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_block:
                        blocks.append({
                            'text': '\n'.join(current_block),
                            'lines': len(current_block),
                            'type': 'paragraph'
                        })
                        current_block = []
                    continue
                
                # Simple font size detection (based on line length and content)
                estimated_font_size = self._estimate_font_size(line)
                
                if current_font_size is None or abs(estimated_font_size - current_font_size) < 2:
                    current_block.append(line)
                    current_font_size = estimated_font_size
                else:
                    if current_block:
                        blocks.append({
                            'text': '\n'.join(current_block),
                            'lines': len(current_block),
                            'type': 'paragraph',
                            'estimated_font_size': current_font_size
                        })
                    current_block = [line]
                    current_font_size = estimated_font_size
            
            if current_block:
                blocks.append({
                    'text': '\n'.join(current_block),
                    'lines': len(current_block),
                    'type': 'paragraph',
                    'estimated_font_size': current_font_size
                })
            
            return blocks
            
        except Exception as e:
            self.logger.error(f"Failed to extract text blocks: {e}")
            return []
    
    def _estimate_font_size(self, line: str) -> float:
        """Estimate font size based on line characteristics"""
        # Simple heuristic based on line length and content
        if len(line) < 20:
            return 14.0  # Likely heading
        elif len(line) < 50:
            return 12.0  # Likely subheading
        else:
            return 10.0  # Likely body text
    
    def _extract_toc_structure(self, outline: List[Tuple[int, str, int]]) -> List[Dict[str, Any]]:
        """Extract table of contents structure"""
        try:
            toc_structure = []
            for level, title, page in outline:
                toc_structure.append({
                    'level': level,
                    'title': title,
                    'page': page,
                    'type': 'heading'
                })
            return toc_structure
        except Exception as e:
            self.logger.error(f"Failed to extract TOC structure: {e}")
            return []
    
    async def _analyze_text(self, analysis: PDFAnalysis):
        """Perform comprehensive text analysis"""
        try:
            if not analysis.content.raw_text:
                return
            
            text = analysis.content.raw_text
            
            # Basic counts
            analysis.word_count = len(analysis.content.words)
            analysis.sentence_count = len(analysis.content.sentences)
            analysis.paragraph_count = len(analysis.content.paragraphs)
            
            # Language detection (simplified)
            analysis.language = self._detect_language(text)
            
            # Readability analysis
            if analysis.sentence_count > 0 and analysis.word_count > 0:
                try:
                    analysis.readability_score = flesch_reading_ease(text)
                    analysis.complexity_score = flesch_kincaid_grade(text)
                except:
                    analysis.readability_score = 0.0
                    analysis.complexity_score = 0.0
            
            # Advanced analysis with spaCy
            if self.nlp:
                await self._advanced_text_analysis(analysis)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze text: {e}")
    
    async def _advanced_text_analysis(self, analysis: PDFAnalysis):
        """Perform advanced text analysis using spaCy"""
        try:
            if not self.nlp:
                return
            
            text = analysis.content.raw_text
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            # Extract key phrases
            key_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:
                    key_phrases.append(chunk.text)
            
            # Analyze sentiment (if available)
            sentiment_score = 0.0
            if hasattr(doc, 'sentiment'):
                sentiment_score = doc.sentiment
            
            # Store advanced analysis results
            analysis.content.advanced_analysis = {
                'entities': entities,
                'key_phrases': key_phrases,
                'sentiment_score': sentiment_score,
                'pos_tags': [(token.text, token.pos_) for token in doc],
                'lemmas': [token.lemma_ for token in doc if not token.is_stop]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to perform advanced text analysis: {e}")
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Simple heuristic based on common words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = set(text.lower().split())
        
        english_count = len(words.intersection(english_words))
        if english_count > 5:
            return 'en'
        
        return 'unknown'
    
    async def _calculate_quality_metrics(self, analysis: PDFAnalysis):
        """Calculate quality metrics for the PDF"""
        try:
            # Text quality metrics
            text_quality = 0.0
            
            if analysis.content.raw_text:
                # Check text length
                if len(analysis.content.raw_text) > 100:
                    text_quality += 0.3
                
                # Check for structured content
                if analysis.content.paragraphs:
                    text_quality += 0.2
                
                # Check for readable text
                if analysis.readability_score > 30:
                    text_quality += 0.2
                
                # Check for proper formatting
                if analysis.content.structured_text:
                    text_quality += 0.3
            
            analysis.text_quality = min(text_quality, 1.0)
            
            # Structure quality metrics
            structure_quality = 0.0
            
            if analysis.structure.has_outline:
                structure_quality += 0.3
            
            if analysis.structure.has_bookmarks:
                structure_quality += 0.2
            
            if analysis.structure.toc_structure:
                structure_quality += 0.2
            
            if analysis.structure.has_forms:
                structure_quality += 0.1
            
            if analysis.structure.has_annotations:
                structure_quality += 0.1
            
            if analysis.structure.has_images:
                structure_quality += 0.1
            
            analysis.structure_quality = min(structure_quality, 1.0)
            
            # Overall quality
            analysis.overall_quality = (analysis.text_quality + analysis.structure_quality) / 2
            
        except Exception as e:
            self.logger.error(f"Failed to calculate quality metrics: {e}")
    
    async def extract_text_only(self, file_path: Union[str, Path, bytes]) -> str:
        """Extract only text content from PDF"""
        try:
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
                async with aiofiles.open(file_path, 'rb') as f:
                    pdf_data = await f.read()
            else:
                pdf_data = file_path
            
            return await self._extract_text_comprehensive(pdf_data)
            
        except Exception as e:
            self.logger.error(f"Failed to extract text only: {e}")
            return ""
    
    async def extract_metadata_only(self, file_path: Union[str, Path, bytes]) -> PDFMetadata:
        """Extract only metadata from PDF"""
        try:
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
                async with aiofiles.open(file_path, 'rb') as f:
                    pdf_data = await f.read()
            else:
                pdf_data = file_path
            
            file_size = len(pdf_data)
            return await self._extract_metadata(pdf_data, file_size)
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata only: {e}")
            return PDFMetadata()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported PDF formats"""
        return ['pdf']
    
    def get_processing_capabilities(self) -> Dict[str, bool]:
        """Get processing capabilities"""
        return {
            'text_extraction': True,
            'metadata_extraction': True,
            'image_extraction': self.enable_image_extraction,
            'table_extraction': self.enable_table_extraction,
            'structure_analysis': self.enable_structure_analysis,
            'text_analysis': self.enable_text_analysis,
            'ocr': self.enable_ocr,
            'form_extraction': True,
            'annotation_extraction': True,
            'link_extraction': True
        }
    
    def close(self):
        """Close the parser and cleanup resources"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Failed to close parser: {e}")

# Utility functions
def create_pdf_parser(config: Optional[Dict[str, Any]] = None) -> PDFParser:
    """Create a new PDF parser instance"""
    return PDFParser(config)

async def parse_pdf_file(file_path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> PDFAnalysis:
    """Convenience function to parse a PDF file"""
    parser = PDFParser(config)
    try:
        return await parser.parse_pdf(file_path)
    finally:
        parser.close()

async def extract_pdf_text(file_path: Union[str, Path]) -> str:
    """Convenience function to extract only text from PDF"""
    parser = PDFParser()
    try:
        return await parser.extract_text_only(file_path)
    finally:
        parser.close()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example usage
        parser = PDFParser({
            'enable_ocr': True,
            'enable_image_extraction': True,
            'enable_table_extraction': True,
            'max_pages': 100
        })
        
        try:
            # Parse PDF file
            analysis = await parser.parse_pdf("example.pdf")
            
            print(f"Title: {analysis.metadata.title}")
            print(f"Author: {analysis.metadata.author}")
            print(f"Pages: {analysis.metadata.page_count}")
            print(f"Text Quality: {analysis.text_quality:.2f}")
            print(f"Structure Quality: {analysis.structure_quality:.2f}")
            print(f"Overall Quality: {analysis.overall_quality:.2f}")
            print(f"Processing Time: {analysis.processing_time:.2f}s")
            
        finally:
            parser.close()
    
    asyncio.run(main())
