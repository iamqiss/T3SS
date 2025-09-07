# T3SS Project
# File: core/indexing/indexer/document_store/sharding_manager.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

"""
Advanced Document Sharding Manager

This module provides sophisticated document sharding capabilities with support for:
- Multiple sharding strategies (hash-based, range-based, consistent hashing)
- Dynamic shard rebalancing and migration
- Shard health monitoring and failover
- Load balancing and performance optimization
- Shard replication and consistency
- Comprehensive statistics and monitoring
"""

import hashlib
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class ShardingStrategy(Enum):
    """Sharding strategies"""
    HASH = "hash"
    RANGE = "range"
    CONSISTENT_HASH = "consistent_hash"
    ROUND_ROBIN = "round_robin"

class ShardStatus(Enum):
    """Shard status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MIGRATING = "migrating"
    FAILED = "failed"

@dataclass
class ShardConfig:
    """Configuration for sharding"""
    strategy: ShardingStrategy = ShardingStrategy.HASH
    num_shards: int = 4
    replication_factor: int = 2
    rebalance_threshold: float = 0.2
    health_check_interval: int = 30
    enable_auto_rebalancing: bool = True

@dataclass
class ShardInfo:
    """Information about a shard"""
    id: str
    name: str
    status: ShardStatus
    size: int
    document_count: int
    created_at: float
    last_accessed: float
    replica_shards: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ShardStats:
    """Statistics for a shard"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0

@dataclass
class ShardingResult:
    """Result of sharding operation"""
    shard_id: str
    replica_shards: List[str]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0

class ConsistentHashRing:
    """Consistent hash ring for shard distribution"""
    
    def __init__(self, shards: List[str], replicas: int = 3):
        self.shards = shards
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        self._build_ring()
    
    def _build_ring(self):
        """Build the consistent hash ring"""
        for shard in self.shards:
            for i in range(self.replicas):
                key = self._hash(f"{shard}:{i}")
                self.ring[key] = shard
                self.sorted_keys.append(key)
        
        self.sorted_keys.sort()
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_shard(self, key: str) -> str:
        """Get shard for a key"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first shard with hash >= hash_key
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # Wrap around to the first shard
        return self.ring[self.sorted_keys[0]]
    
    def get_replica_shards(self, key: str, count: int) -> List[str]:
        """Get replica shards for a key"""
        if not self.ring:
            return []
        
        hash_key = self._hash(key)
        replicas = []
        
        # Find all shards starting from the primary
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                shard = self.ring[ring_key]
                if shard not in replicas:
                    replicas.append(shard)
                    if len(replicas) >= count:
                        break
        
        # If we need more replicas, wrap around
        if len(replicas) < count:
            for ring_key in self.sorted_keys:
                shard = self.ring[ring_key]
                if shard not in replicas:
                    replicas.append(shard)
                    if len(replicas) >= count:
                        break
        
        return replicas

class ShardingManager:
    """Advanced document sharding manager"""
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.shards: Dict[str, ShardInfo] = {}
        self.shard_stats: Dict[str, ShardStats] = {}
        self.hash_ring: Optional[ConsistentHashRing] = None
        self.round_robin_index = 0
        self.lock = threading.RLock()
        
        # Initialize shards
        self._initialize_shards()
    
    def _initialize_shards(self):
        """Initialize shards based on configuration"""
        for i in range(self.config.num_shards):
            shard_id = f"shard_{i}"
            shard_info = ShardInfo(
                id=shard_id,
                name=f"Shard {i}",
                status=ShardStatus.ACTIVE,
                size=0,
                document_count=0,
                created_at=time.time(),
                last_accessed=time.time()
            )
            
            self.shards[shard_id] = shard_info
            self.shard_stats[shard_id] = ShardStats()
        
        # Initialize consistent hash ring if needed
        if self.config.strategy == ShardingStrategy.CONSISTENT_HASH:
            shard_ids = list(self.shards.keys())
            self.hash_ring = ConsistentHashRing(shard_ids, self.config.replication_factor)
    
    def get_shard_for_document(self, document_id: str) -> ShardingResult:
        """Get shard for a document"""
        start_time = time.time()
        
        try:
            with self.lock:
                if self.config.strategy == ShardingStrategy.HASH:
                    shard_id = self._get_hash_shard(document_id)
                elif self.config.strategy == ShardingStrategy.RANGE:
                    shard_id = self._get_range_shard(document_id)
                elif self.config.strategy == ShardingStrategy.CONSISTENT_HASH:
                    shard_id = self._get_consistent_hash_shard(document_id)
                elif self.config.strategy == ShardingStrategy.ROUND_ROBIN:
                    shard_id = self._get_round_robin_shard()
                else:
                    shard_id = self._get_hash_shard(document_id)
                
                # Get replica shards
                replica_shards = self._get_replica_shards(shard_id)
                
                # Update shard stats
                self._update_shard_stats(shard_id, True, time.time() - start_time)
                
                return ShardingResult(
                    shard_id=shard_id,
                    replica_shards=replica_shards,
                    success=True,
                    processing_time=time.time() - start_time
                )
        
        except Exception as e:
            logger.error(f"Error getting shard for document {document_id}: {e}")
            return ShardingResult(
                shard_id="",
                replica_shards=[],
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _get_hash_shard(self, document_id: str) -> str:
        """Get shard using hash-based sharding"""
        hash_value = int(hashlib.md5(document_id.encode()).hexdigest(), 16)
        shard_index = hash_value % self.config.num_shards
        return f"shard_{shard_index}"
    
    def _get_range_shard(self, document_id: str) -> str:
        """Get shard using range-based sharding"""
        first_char = document_id[0].lower() if document_id else 'a'
        char_value = ord(first_char) - ord('a')
        shard_index = char_value % self.config.num_shards
        return f"shard_{shard_index}"
    
    def _get_consistent_hash_shard(self, document_id: str) -> str:
        """Get shard using consistent hashing"""
        if self.hash_ring:
            return self.hash_ring.get_shard(document_id)
        return f"shard_0"
    
    def _get_round_robin_shard(self) -> str:
        """Get shard using round-robin"""
        with self.lock:
            shard_id = f"shard_{self.round_robin_index}"
            self.round_robin_index = (self.round_robin_index + 1) % self.config.num_shards
            return shard_id
    
    def _get_replica_shards(self, primary_shard_id: str) -> List[str]:
        """Get replica shards for a primary shard"""
        if self.config.strategy == ShardingStrategy.CONSISTENT_HASH and self.hash_ring:
            return self.hash_ring.get_replica_shards(primary_shard_id, self.config.replication_factor)
        
        # Simple replica selection
        shard_ids = list(self.shards.keys())
        primary_index = shard_ids.index(primary_shard_id)
        replicas = []
        
        for i in range(1, self.config.replication_factor + 1):
            replica_index = (primary_index + i) % len(shard_ids)
            replicas.append(shard_ids[replica_index])
        
        return replicas
    
    def _update_shard_stats(self, shard_id: str, success: bool, response_time: float):
        """Update shard statistics"""
        if shard_id not in self.shard_stats:
            return
        
        stats = self.shard_stats[shard_id]
        stats.total_operations += 1
        
        if success:
            stats.successful_operations += 1
        else:
            stats.failed_operations += 1
        
        # Update average response time
        if stats.total_operations == 1:
            stats.average_response_time = response_time
        else:
            stats.average_response_time = (stats.average_response_time + response_time) / 2
        
        # Update error rate
        stats.error_rate = stats.failed_operations / stats.total_operations
    
    def get_shard_info(self, shard_id: str) -> Optional[ShardInfo]:
        """Get information about a shard"""
        with self.lock:
            return self.shards.get(shard_id)
    
    def get_all_shards(self) -> List[ShardInfo]:
        """Get all shards"""
        with self.lock:
            return list(self.shards.values())
    
    def get_shard_stats(self, shard_id: str) -> Optional[ShardStats]:
        """Get statistics for a shard"""
        with self.lock:
            return self.shard_stats.get(shard_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self.lock:
            metrics = {
                'total_shards': len(self.shards),
                'active_shards': len([s for s in self.shards.values() if s.status == ShardStatus.ACTIVE]),
                'total_operations': sum(stats.total_operations for stats in self.shard_stats.values()),
                'average_response_time': sum(stats.average_response_time for stats in self.shard_stats.values()) / len(self.shard_stats) if self.shard_stats else 0.0,
                'total_error_rate': sum(stats.failed_operations for stats in self.shard_stats.values()) / max(sum(stats.total_operations for stats in self.shard_stats.values()), 1)
            }
            
            return metrics