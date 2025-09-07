# T3SS Project
# File: core/indexing/indexer/document_store/document_store_manager.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

"""
Document Store Manager

This module provides centralized management of multiple document stores with support for:
- Load balancing across stores
- Failover and redundancy
- Performance monitoring
- Store health checking
- Automatic scaling
- Data replication
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import random
import statistics

from .document_store import DocumentStore, DocumentStoreConfig, DocumentStoreStats
from .sharding_manager import ShardingManager, ShardConfig, ShardingStrategy

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

class StoreStatus(Enum):
    """Store status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class StoreInfo:
    """Information about a document store"""
    id: str
    name: str
    store: DocumentStore
    status: StoreStatus
    weight: int = 1
    last_health_check: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    connection_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentStoreManagerConfig:
    """Configuration for document store manager"""
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: int = 30
    enable_failover: bool = True
    enable_replication: bool = True
    replication_factor: int = 2
    enable_auto_scaling: bool = True
    max_stores: int = 10
    min_stores: int = 1
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    enable_metrics: bool = True

@dataclass
class DocumentStoreManagerStats:
    """Statistics for document store manager"""
    total_stores: int = 0
    healthy_stores: int = 0
    failed_stores: int = 0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    load_balance_efficiency: float = 0.0

class DocumentStoreManager:
    """Centralized document store manager"""
    
    def __init__(self, config: DocumentStoreManagerConfig):
        self.config = config
        self.stores: Dict[str, StoreInfo] = {}
        self.stats = DocumentStoreManagerStats()
        self.lock = threading.RLock()
        self.round_robin_index = 0
        self.health_checker = None
        self.scaler = None
        
        # Start background tasks
        self._start_health_checker()
        if config.enable_auto_scaling:
            self._start_scaler()
    
    def add_store(self, store_id: str, store: DocumentStore, weight: int = 1) -> bool:
        """Add a document store"""
        try:
            with self.lock:
                if store_id in self.stores:
                    logger.warning(f"Store {store_id} already exists")
                    return False
                
                store_info = StoreInfo(
                    id=store_id,
                    name=f"Store {store_id}",
                    store=store,
                    status=StoreStatus.HEALTHY,
                    weight=weight,
                    last_health_check=time.time()
                )
                
                self.stores[store_id] = store_info
                self.stats.total_stores += 1
                self.stats.healthy_stores += 1
                
                logger.info(f"Added store: {store_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error adding store {store_id}: {e}")
            return False
    
    def remove_store(self, store_id: str) -> bool:
        """Remove a document store"""
        try:
            with self.lock:
                if store_id in self.stores:
                    store_info = self.stores[store_id]
                    
                    # Close store if needed
                    if hasattr(store_info.store, 'close'):
                        store_info.store.close()
                    
                    del self.stores[store_id]
                    self.stats.total_stores -= 1
                    
                    if store_info.status == StoreStatus.HEALTHY:
                        self.stats.healthy_stores -= 1
                    elif store_info.status == StoreStatus.FAILED:
                        self.stats.failed_stores -= 1
                    
                    logger.info(f"Removed store: {store_id}")
                    return True
                
                return False
        
        except Exception as e:
            logger.error(f"Error removing store {store_id}: {e}")
            return False
    
    def get_store(self, store_id: str) -> Optional[StoreInfo]:
        """Get store information"""
        with self.lock:
            return self.stores.get(store_id)
    
    def get_healthy_stores(self) -> List[StoreInfo]:
        """Get all healthy stores"""
        with self.lock:
            return [store for store in self.stores.values() if store.status == StoreStatus.HEALTHY]
    
    def select_store(self, document_id: Optional[str] = None) -> Optional[StoreInfo]:
        """Select a store based on load balancing strategy"""
        try:
            with self.lock:
                healthy_stores = self.get_healthy_stores()
                if not healthy_stores:
                    return None
                
                if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
                    return self._select_round_robin(healthy_stores)
                elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                    return self._select_least_connections(healthy_stores)
                elif self.config.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                    return self._select_weighted_round_robin(healthy_stores)
                elif self.config.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
                    return self._select_random(healthy_stores)
                elif self.config.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    return self._select_consistent_hash(healthy_stores, document_id)
                else:
                    return self._select_round_robin(healthy_stores)
        
        except Exception as e:
            logger.error(f"Error selecting store: {e}")
            return None
    
    def _select_round_robin(self, stores: List[StoreInfo]) -> StoreInfo:
        """Select store using round-robin"""
        store = stores[self.round_robin_index % len(stores)]
        self.round_robin_index = (self.round_robin_index + 1) % len(stores)
        return store
    
    def _select_least_connections(self, stores: List[StoreInfo]) -> StoreInfo:
        """Select store with least connections"""
        return min(stores, key=lambda s: s.connection_count)
    
    def _select_weighted_round_robin(self, stores: List[StoreInfo]) -> StoreInfo:
        """Select store using weighted round-robin"""
        total_weight = sum(store.weight for store in stores)
        if total_weight == 0:
            return stores[0]
        
        # Simple weighted selection
        target_weight = self.round_robin_index % total_weight
        current_weight = 0
        
        for store in stores:
            current_weight += store.weight
            if current_weight > target_weight:
                self.round_robin_index = (self.round_robin_index + 1) % total_weight
                return store
        
        return stores[0]
    
    def _select_random(self, stores: List[StoreInfo]) -> StoreInfo:
        """Select store randomly"""
        return random.choice(stores)
    
    def _select_consistent_hash(self, stores: List[StoreInfo], document_id: Optional[str]) -> StoreInfo:
        """Select store using consistent hashing"""
        if not document_id:
            return self._select_round_robin(stores)
        
        # Simple hash-based selection
        hash_value = hash(document_id)
        store_index = hash_value % len(stores)
        return stores[store_index]
    
    def store_document(self, document_id: str, content: str, metadata: Any) -> bool:
        """Store a document"""
        try:
            store_info = self.select_store(document_id)
            if not store_info:
                logger.error("No healthy stores available")
                return False
            
            # Increment connection count
            store_info.connection_count += 1
            
            # Store document
            start_time = time.time()
            success = store_info.store.store_document(document_id, content, metadata)
            response_time = time.time() - start_time
            
            # Update store stats
            store_info.response_time = response_time
            store_info.connection_count -= 1
            
            # Update manager stats
            self._update_stats(success, response_time)
            
            if success:
                logger.debug(f"Stored document {document_id} in store {store_info.id}")
            else:
                logger.error(f"Failed to store document {document_id} in store {store_info.id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error storing document {document_id}: {e}")
            self._update_stats(False, 0.0)
            return False
    
    def get_document(self, document_id: str) -> Optional[Any]:
        """Get a document"""
        try:
            store_info = self.select_store(document_id)
            if not store_info:
                logger.error("No healthy stores available")
                return None
            
            # Increment connection count
            store_info.connection_count += 1
            
            # Get document
            start_time = time.time()
            document = store_info.store.get_document(document_id)
            response_time = time.time() - start_time
            
            # Update store stats
            store_info.response_time = response_time
            store_info.connection_count -= 1
            
            # Update manager stats
            self._update_stats(document is not None, response_time)
            
            if document:
                logger.debug(f"Retrieved document {document_id} from store {store_info.id}")
            else:
                logger.debug(f"Document {document_id} not found in store {store_info.id}")
            
            return document
        
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            self._update_stats(False, 0.0)
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        try:
            store_info = self.select_store(document_id)
            if not store_info:
                logger.error("No healthy stores available")
                return False
            
            # Increment connection count
            store_info.connection_count += 1
            
            # Delete document
            start_time = time.time()
            success = store_info.store.delete_document(document_id)
            response_time = time.time() - start_time
            
            # Update store stats
            store_info.response_time = response_time
            store_info.connection_count -= 1
            
            # Update manager stats
            self._update_stats(success, response_time)
            
            if success:
                logger.debug(f"Deleted document {document_id} from store {store_info.id}")
            else:
                logger.error(f"Failed to delete document {document_id} from store {store_info.id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            self._update_stats(False, 0.0)
            return False
    
    def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Search documents across all stores"""
        try:
            results = []
            
            with self.lock:
                for store_info in self.stores.values():
                    if store_info.status == StoreStatus.HEALTHY:
                        try:
                            store_results = store_info.store.search_documents(query, filters)
                            results.extend(store_results)
                        except Exception as e:
                            logger.error(f"Error searching in store {store_info.id}: {e}")
            
            # Remove duplicates and sort by relevance
            results = list(set(results))
            results = self._sort_by_relevance(results, query)
            
            logger.debug(f"Found {len(results)} documents for query: {query}")
            return results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _sort_by_relevance(self, documents: List[Any], query: str) -> List[Any]:
        """Sort documents by relevance to query"""
        query_terms = set(query.lower().split())
        
        def relevance_score(doc: Any) -> int:
            if hasattr(doc, 'content'):
                content_terms = set(doc.content.lower().split())
                return len(query_terms.intersection(content_terms))
            return 0
        
        return sorted(documents, key=relevance_score, reverse=True)
    
    def _update_stats(self, success: bool, response_time: float):
        """Update manager statistics"""
        with self.lock:
            self.stats.total_operations += 1
            
            if success:
                self.stats.successful_operations += 1
            else:
                self.stats.failed_operations += 1
            
            # Update average response time
            if self.stats.total_operations == 1:
                self.stats.average_response_time = response_time
            else:
                self.stats.average_response_time = (self.stats.average_response_time + response_time) / 2
    
    def _start_health_checker(self):
        """Start health checker thread"""
        def health_check_loop():
            while True:
                try:
                    self._check_store_health()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health checker: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.health_checker = threading.Thread(target=health_check_loop, daemon=True)
        self.health_checker.start()
    
    def _check_store_health(self):
        """Check health of all stores"""
        with self.lock:
            for store_info in self.stores.values():
                try:
                    # Simple health check - try to get a test document
                    start_time = time.time()
                    store_info.store.get_document("__health_check__")
                    response_time = time.time() - start_time
                    
                    # Update store stats
                    store_info.response_time = response_time
                    store_info.last_health_check = time.time()
                    
                    # Determine status based on response time and error rate
                    if response_time > 5.0:  # 5 seconds timeout
                        store_info.status = StoreStatus.DEGRADED
                    else:
                        store_info.status = StoreStatus.HEALTHY
                    
                except Exception as e:
                    logger.warning(f"Store {store_info.id} health check failed: {e}")
                    store_info.status = StoreStatus.FAILED
                    store_info.error_rate = 1.0
            
            # Update manager stats
            self.stats.healthy_stores = len([s for s in self.stores.values() if s.status == StoreStatus.HEALTHY])
            self.stats.failed_stores = len([s for s in self.stores.values() if s.status == StoreStatus.FAILED])
    
    def _start_scaler(self):
        """Start auto-scaling thread"""
        def scale_loop():
            while True:
                try:
                    self._check_scaling_needs()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Error in scaler: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.scaler = threading.Thread(target=scale_loop, daemon=True)
        self.scaler.start()
    
    def _check_scaling_needs(self):
        """Check if scaling is needed"""
        with self.lock:
            if not self.config.enable_auto_scaling:
                return
            
            healthy_stores = self.get_healthy_stores()
            if not healthy_stores:
                return
            
            # Calculate average load
            total_connections = sum(store.connection_count for store in healthy_stores)
            avg_connections = total_connections / len(healthy_stores)
            
            # Check if we need to scale up
            if len(healthy_stores) < self.config.max_stores and avg_connections > self.config.scale_up_threshold:
                logger.info("Scaling up stores")
                self._scale_up()
            
            # Check if we need to scale down
            elif len(healthy_stores) > self.config.min_stores and avg_connections < self.config.scale_down_threshold:
                logger.info("Scaling down stores")
                self._scale_down()
    
    def _scale_up(self):
        """Scale up by adding a new store"""
        # This would implement actual scaling logic
        logger.info("Scale up not implemented")
    
    def _scale_down(self):
        """Scale down by removing a store"""
        # This would implement actual scaling logic
        logger.info("Scale down not implemented")
    
    def get_stats(self) -> DocumentStoreManagerStats:
        """Get manager statistics"""
        with self.lock:
            return self.stats
    
    def get_store_stats(self, store_id: str) -> Optional[DocumentStoreStats]:
        """Get statistics for a specific store"""
        with self.lock:
            store_info = self.stores.get(store_id)
            if store_info:
                return store_info.store.get_stats()
            return None
    
    def get_all_store_stats(self) -> Dict[str, DocumentStoreStats]:
        """Get statistics for all stores"""
        with self.lock:
            stats = {}
            for store_id, store_info in self.stores.items():
                stats[store_id] = store_info.store.get_stats()
            return stats
    
    def close(self):
        """Close all stores"""
        with self.lock:
            for store_info in self.stores.values():
                if hasattr(store_info.store, 'close'):
                    store_info.store.close()
            self.stores.clear()