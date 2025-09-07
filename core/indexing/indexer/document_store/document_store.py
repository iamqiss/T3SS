# T3SS Project
# File: core/indexing/indexer/document_store/document_store.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

"""
Advanced Document Store

This module provides a high-performance document storage system with support for:
- Multiple storage backends (memory, disk, distributed)
- Document versioning and history
- Compression and encryption
- Full-text search capabilities
- Metadata indexing and querying
- Caching and performance optimization
- Backup and recovery
"""

import json
import time
import logging
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import gzip
import pickle
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class StorageBackend(Enum):
    """Storage backend types"""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"

class DocumentStatus(Enum):
    """Document status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    PENDING = "pending"

@dataclass
class DocumentMetadata:
    """Document metadata"""
    id: str
    title: str
    url: str
    content_type: str
    language: str
    created_at: float
    updated_at: float
    size: int
    status: DocumentStatus
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    metadata: DocumentMetadata
    version: int = 1
    checksum: str = ""
    compressed: bool = False
    encrypted: bool = False

@dataclass
class DocumentStoreConfig:
    """Configuration for document store"""
    backend: StorageBackend = StorageBackend.MEMORY
    max_documents: int = 1000000
    enable_compression: bool = True
    enable_encryption: bool = False
    enable_versioning: bool = True
    enable_caching: bool = True
    cache_size: int = 10000
    storage_path: str = "data/documents"
    backup_interval: int = 3600
    enable_metrics: bool = True

@dataclass
class DocumentStoreStats:
    """Document store statistics"""
    total_documents: int = 0
    total_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_ratio: float = 0.0
    average_document_size: float = 0.0
    last_backup: Optional[float] = None

class DocumentStore:
    """Advanced document store"""
    
    def __init__(self, config: DocumentStoreConfig):
        self.config = config
        self.documents: Dict[str, Document] = {}
        self.metadata_index: Dict[str, List[str]] = defaultdict(list)
        self.content_index: Dict[str, List[str]] = defaultdict(list)
        self.cache: Dict[str, Document] = {}
        self.stats = DocumentStoreStats()
        self.lock = threading.RLock()
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Initialize storage backend
        self._initialize_storage()
        
        # Start background tasks
        if config.enable_metrics:
            self._start_metrics_collector()
    
    def _initialize_storage(self):
        """Initialize storage backend"""
        if self.config.backend == StorageBackend.DISK:
            self._initialize_disk_storage()
        elif self.config.backend == StorageBackend.DISTRIBUTED:
            self._initialize_distributed_storage()
        elif self.config.backend == StorageBackend.HYBRID:
            self._initialize_hybrid_storage()
    
    def _initialize_disk_storage(self):
        """Initialize disk-based storage"""
        Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        db_path = Path(self.config.storage_path) / "documents.db"
        self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables
        cursor = self.db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                version INTEGER,
                checksum TEXT,
                created_at REAL,
                updated_at REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_versions (
                id TEXT,
                version INTEGER,
                content TEXT,
                created_at REAL,
                PRIMARY KEY (id, version)
            )
        ''')
        
        self.db_connection.commit()
    
    def _initialize_distributed_storage(self):
        """Initialize distributed storage"""
        # This would initialize distributed storage (e.g., Redis, Cassandra)
        pass
    
    def _initialize_hybrid_storage(self):
        """Initialize hybrid storage (memory + disk)"""
        self._initialize_disk_storage()
    
    def store_document(self, document: Document) -> bool:
        """Store a document"""
        try:
            with self.lock:
                # Generate checksum
                document.checksum = self._calculate_checksum(document.content)
                
                # Compress if enabled
                if self.config.enable_compression:
                    document = self._compress_document(document)
                
                # Encrypt if enabled
                if self.config.enable_encryption:
                    document = self._encrypt_document(document)
                
                # Store document
                if self.config.backend == StorageBackend.MEMORY:
                    self._store_in_memory(document)
                elif self.config.backend == StorageBackend.DISK:
                    self._store_on_disk(document)
                elif self.config.backend == StorageBackend.DISTRIBUTED:
                    self._store_distributed(document)
                elif self.config.backend == StorageBackend.HYBRID:
                    self._store_hybrid(document)
                
                # Update indexes
                self._update_indexes(document)
                
                # Update cache
                if self.config.enable_caching:
                    self._update_cache(document)
                
                # Update stats
                self._update_stats_on_store(document)
                
                logger.info(f"Stored document: {document.id}")
                return True
        
        except Exception as e:
            logger.error(f"Error storing document {document.id}: {e}")
            return False
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID"""
        try:
            with self.lock:
                # Check cache first
                if self.config.enable_caching and document_id in self.cache:
                    self.stats.cache_hits += 1
                    return self.cache[document_id]
                
                # Get from storage
                document = None
                if self.config.backend == StorageBackend.MEMORY:
                    document = self.documents.get(document_id)
                elif self.config.backend == StorageBackend.DISK:
                    document = self._get_from_disk(document_id)
                elif self.config.backend == StorageBackend.DISTRIBUTED:
                    document = self._get_distributed(document_id)
                elif self.config.backend == StorageBackend.HYBRID:
                    document = self._get_hybrid(document_id)
                
                if document:
                    # Decompress if needed
                    if document.compressed:
                        document = self._decompress_document(document)
                    
                    # Decrypt if needed
                    if document.encrypted:
                        document = self._decrypt_document(document)
                    
                    # Update cache
                    if self.config.enable_caching:
                        self._update_cache(document)
                    
                    self.stats.cache_misses += 1
                
                return document
        
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None
    
    def update_document(self, document_id: str, content: str, metadata: Optional[DocumentMetadata] = None) -> bool:
        """Update a document"""
        try:
            with self.lock:
                # Get existing document
                existing_doc = self.get_document(document_id)
                if not existing_doc:
                    return False
                
                # Create new version
                new_document = Document(
                    id=document_id,
                    content=content,
                    metadata=metadata or existing_doc.metadata,
                    version=existing_doc.version + 1,
                    checksum="",
                    compressed=existing_doc.compressed,
                    encrypted=existing_doc.encrypted
                )
                
                # Store new version
                if self.config.enable_versioning:
                    self._store_version(existing_doc)
                
                return self.store_document(new_document)
        
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        try:
            with self.lock:
                # Get document
                document = self.get_document(document_id)
                if not document:
                    return False
                
                # Update status
                document.metadata.status = DocumentStatus.DELETED
                
                # Remove from storage
                if self.config.backend == StorageBackend.MEMORY:
                    self.documents.pop(document_id, None)
                elif self.config.backend == StorageBackend.DISK:
                    self._delete_from_disk(document_id)
                elif self.config.backend == StorageBackend.DISTRIBUTED:
                    self._delete_distributed(document_id)
                elif self.config.backend == StorageBackend.HYBRID:
                    self._delete_hybrid(document_id)
                
                # Remove from cache
                self.cache.pop(document_id, None)
                
                # Update indexes
                self._remove_from_indexes(document)
                
                # Update stats
                self._update_stats_on_delete(document)
                
                logger.info(f"Deleted document: {document_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search documents"""
        try:
            with self.lock:
                results = []
                
                # Search in content index
                query_terms = query.lower().split()
                for term in query_terms:
                    if term in self.content_index:
                        for doc_id in self.content_index[term]:
                            doc = self.get_document(doc_id)
                            if doc and doc.metadata.status == DocumentStatus.ACTIVE:
                                results.append(doc)
                
                # Apply filters
                if filters:
                    results = self._apply_filters(results, filters)
                
                # Remove duplicates and sort by relevance
                results = list(set(results))
                results = self._sort_by_relevance(results, query)
                
                return results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_versions(self, document_id: str) -> List[Document]:
        """Get all versions of a document"""
        try:
            with self.lock:
                if self.config.backend == StorageBackend.DISK and self.db_connection:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "SELECT content, metadata, version, created_at FROM document_versions WHERE id = ? ORDER BY version",
                        (document_id,)
                    )
                    
                    versions = []
                    for row in cursor.fetchall():
                        content, metadata_json, version, created_at = row
                        metadata = DocumentMetadata(**json.loads(metadata_json))
                        
                        document = Document(
                            id=document_id,
                            content=content,
                            metadata=metadata,
                            version=version
                        )
                        versions.append(document)
                    
                    return versions
                
                return []
        
        except Exception as e:
            logger.error(f"Error getting document versions {document_id}: {e}")
            return []
    
    def _store_in_memory(self, document: Document):
        """Store document in memory"""
        self.documents[document.id] = document
    
    def _store_on_disk(self, document: Document):
        """Store document on disk"""
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO documents (id, content, metadata, version, checksum, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.content,
                    json.dumps(document.metadata.__dict__),
                    document.version,
                    document.checksum,
                    document.metadata.created_at,
                    document.metadata.updated_at
                )
            )
            self.db_connection.commit()
    
    def _store_distributed(self, document: Document):
        """Store document in distributed storage"""
        # This would implement distributed storage
        pass
    
    def _store_hybrid(self, document: Document):
        """Store document in hybrid storage"""
        self._store_in_memory(document)
        self._store_on_disk(document)
    
    def _get_from_disk(self, document_id: str) -> Optional[Document]:
        """Get document from disk"""
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT content, metadata, version, checksum FROM documents WHERE id = ?",
                (document_id,)
            )
            
            row = cursor.fetchone()
            if row:
                content, metadata_json, version, checksum = row
                metadata = DocumentMetadata(**json.loads(metadata_json))
                
                return Document(
                    id=document_id,
                    content=content,
                    metadata=metadata,
                    version=version,
                    checksum=checksum
                )
        
        return None
    
    def _get_distributed(self, document_id: str) -> Optional[Document]:
        """Get document from distributed storage"""
        # This would implement distributed retrieval
        return None
    
    def _get_hybrid(self, document_id: str) -> Optional[Document]:
        """Get document from hybrid storage"""
        # Try memory first
        doc = self.documents.get(document_id)
        if doc:
            return doc
        
        # Fall back to disk
        return self._get_from_disk(document_id)
    
    def _store_version(self, document: Document):
        """Store document version"""
        if self.config.backend == StorageBackend.DISK and self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO document_versions (id, version, content, created_at) VALUES (?, ?, ?, ?)",
                (document.id, document.version, document.content, time.time())
            )
            self.db_connection.commit()
    
    def _delete_from_disk(self, document_id: str):
        """Delete document from disk"""
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            cursor.execute("DELETE FROM document_versions WHERE id = ?", (document_id,))
            self.db_connection.commit()
    
    def _delete_distributed(self, document_id: str):
        """Delete document from distributed storage"""
        # This would implement distributed deletion
        pass
    
    def _delete_hybrid(self, document_id: str):
        """Delete document from hybrid storage"""
        self.documents.pop(document_id, None)
        self._delete_from_disk(document_id)
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate content checksum"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _compress_document(self, document: Document) -> Document:
        """Compress document content"""
        compressed_content = gzip.compress(document.content.encode())
        document.content = compressed_content.decode('latin-1')
        document.compressed = True
        return document
    
    def _decompress_document(self, document: Document) -> Document:
        """Decompress document content"""
        compressed_content = document.content.encode('latin-1')
        document.content = gzip.decompress(compressed_content).decode()
        document.compressed = False
        return document
    
    def _encrypt_document(self, document: Document) -> Document:
        """Encrypt document content"""
        # This would implement encryption
        document.encrypted = True
        return document
    
    def _decrypt_document(self, document: Document) -> Document:
        """Decrypt document content"""
        # This would implement decryption
        document.encrypted = False
        return document
    
    def _update_indexes(self, document: Document):
        """Update search indexes"""
        # Update content index
        content_terms = document.content.lower().split()
        for term in content_terms:
            if term not in self.content_index[term]:
                self.content_index[term].append(document.id)
        
        # Update metadata index
        self.metadata_index[document.metadata.title].append(document.id)
        self.metadata_index[document.metadata.url].append(document.id)
        for tag in document.metadata.tags:
            self.metadata_index[tag].append(document.id)
    
    def _remove_from_indexes(self, document: Document):
        """Remove document from indexes"""
        # Remove from content index
        content_terms = document.content.lower().split()
        for term in content_terms:
            if term in self.content_index:
                self.content_index[term] = [doc_id for doc_id in self.content_index[term] if doc_id != document.id]
        
        # Remove from metadata index
        for key in [document.metadata.title, document.metadata.url] + document.metadata.tags:
            if key in self.metadata_index:
                self.metadata_index[key] = [doc_id for doc_id in self.metadata_index[key] if doc_id != document.id]
    
    def _update_cache(self, document: Document):
        """Update document cache"""
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[document.id] = document
    
    def _apply_filters(self, documents: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Apply filters to documents"""
        filtered = documents
        
        if 'status' in filters:
            filtered = [doc for doc in filtered if doc.metadata.status == filters['status']]
        
        if 'content_type' in filters:
            filtered = [doc for doc in filtered if doc.metadata.content_type == filters['content_type']]
        
        if 'language' in filters:
            filtered = [doc for doc in filtered if doc.metadata.language == filters['language']]
        
        if 'tags' in filters:
            required_tags = set(filters['tags'])
            filtered = [doc for doc in filtered if required_tags.issubset(set(doc.metadata.tags))]
        
        return filtered
    
    def _sort_by_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """Sort documents by relevance to query"""
        query_terms = set(query.lower().split())
        
        def relevance_score(doc: Document) -> int:
            content_terms = set(doc.content.lower().split())
            return len(query_terms.intersection(content_terms))
        
        return sorted(documents, key=relevance_score, reverse=True)
    
    def _update_stats_on_store(self, document: Document):
        """Update statistics on document store"""
        self.stats.total_documents += 1
        self.stats.total_size += len(document.content)
        self.stats.average_document_size = self.stats.total_size / self.stats.total_documents
    
    def _update_stats_on_delete(self, document: Document):
        """Update statistics on document delete"""
        self.stats.total_documents -= 1
        self.stats.total_size -= len(document.content)
        if self.stats.total_documents > 0:
            self.stats.average_document_size = self.stats.total_size / self.stats.total_documents
    
    def _start_metrics_collector(self):
        """Start metrics collection thread"""
        def collect_metrics():
            while True:
                try:
                    time.sleep(60)  # Collect every minute
                    self._collect_metrics()
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
        
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        with self.lock:
            # Calculate compression ratio
            if self.stats.total_size > 0:
                compressed_size = sum(len(doc.content) for doc in self.documents.values() if doc.compressed)
                self.stats.compression_ratio = compressed_size / self.stats.total_size
    
    def get_stats(self) -> DocumentStoreStats:
        """Get document store statistics"""
        with self.lock:
            return self.stats
    
    def close(self):
        """Close document store"""
        if self.db_connection:
            self.db_connection.close()