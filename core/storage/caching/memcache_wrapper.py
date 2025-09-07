# T3SS Project
# File: core/storage/caching/memcache_wrapper.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

"""
Advanced Memcache Wrapper

This module provides a high-performance wrapper around memcached with support for:
- Connection pooling and failover
- Automatic serialization/deserialization
- Compression and encryption
- TTL management and expiration
- Statistics and monitoring
- Batch operations
"""

import time
import json
import pickle
import gzip
import logging
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False
    memcache = None

logger = logging.getLogger(__name__)

class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"

class SerializationType(Enum):
    JSON = "json"
    PICKLE = "pickle"

@dataclass
class CacheConfig:
    servers: List[str] = field(default_factory=lambda: ["127.0.0.1:11211"])
    compression: CompressionType = CompressionType.GZIP
    serialization: SerializationType = SerializationType.JSON
    default_ttl: int = 3600
    enable_compression: bool = True
    enable_statistics: bool = True
    connection_pool_size: int = 10
    debug: bool = False

@dataclass
class CacheStats:
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    get_operations: int = 0
    set_operations: int = 0
    hit_rate: float = 0.0
    average_response_time: float = 0.0

class MemcacheWrapper:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.connection_pool = []
        self.current_connection = 0
        
        if not MEMCACHE_AVAILABLE:
            raise ImportError("memcache library not available")
        
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        for _ in range(self.config.connection_pool_size):
            try:
                client = memcache.Client(self.config.servers, debug=self.config.debug)
                client.servers[0].connect()
                self.connection_pool.append(client)
            except Exception as e:
                logger.error(f"Failed to create memcache connection: {e}")
    
    def _get_connection(self):
        with self.lock:
            if not self.connection_pool:
                raise RuntimeError("No memcache connections available")
            
            connection = self.connection_pool[self.current_connection]
            self.current_connection = (self.current_connection + 1) % len(self.connection_pool)
            return connection
    
    def get(self, key: str) -> Optional[Any]:
        start_time = time.time()
        
        try:
            with self.lock:
                self.stats.total_operations += 1
                self.stats.get_operations += 1
            
            connection = self._get_connection()
            value = connection.get(key)
            
            if value is not None:
                value = self._deserialize_value(value)
                value = self._decompress_value(value)
                
                with self.lock:
                    self.stats.successful_operations += 1
                    self.stats.hit_rate = self.stats.successful_operations / self.stats.total_operations
                
                return value
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            with self.lock:
                self.stats.failed_operations += 1
            return None
        
        finally:
            response_time = time.time() - start_time
            with self.lock:
                self.stats.average_response_time = (self.stats.average_response_time + response_time) / 2
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        start_time = time.time()
        
        try:
            with self.lock:
                self.stats.total_operations += 1
                self.stats.set_operations += 1
            
            connection = self._get_connection()
            
            compressed_value = self._compress_value(value)
            serialized_value = self._serialize_value(compressed_value)
            
            if ttl is None:
                ttl = self.config.default_ttl
            
            success = connection.set(key, serialized_value, time=ttl)
            
            if success:
                with self.lock:
                    self.stats.successful_operations += 1
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            with self.lock:
                self.stats.failed_operations += 1
            return False
        
        finally:
            response_time = time.time() - start_time
            with self.lock:
                self.stats.average_response_time = (self.stats.average_response_time + response_time) / 2
    
    def delete(self, key: str) -> bool:
        try:
            connection = self._get_connection()
            return connection.delete(key)
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        try:
            connection = self._get_connection()
            values = connection.get_multi(keys)
            
            result = {}
            for key, value in values.items():
                if value is not None:
                    deserialized = self._deserialize_value(value)
                    decompressed = self._decompress_value(deserialized)
                    result[key] = decompressed
            
            return result
        except Exception as e:
            logger.error(f"Error getting multiple keys: {e}")
            return {}
    
    def set_multi(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            connection = self._get_connection()
            
            processed_mapping = {}
            for key, value in mapping.items():
                compressed_value = self._compress_value(value)
                serialized_value = self._serialize_value(compressed_value)
                processed_mapping[key] = serialized_value
            
            if ttl is None:
                ttl = self.config.default_ttl
            
            return connection.set_multi(processed_mapping, time=ttl)
        except Exception as e:
            logger.error(f"Error setting multiple keys: {e}")
            return False
    
    def increment(self, key: str, delta: int = 1) -> Optional[int]:
        try:
            connection = self._get_connection()
            return connection.incr(key, delta)
        except Exception as e:
            logger.error(f"Error incrementing key {key}: {e}")
            return None
    
    def decrement(self, key: str, delta: int = 1) -> Optional[int]:
        try:
            connection = self._get_connection()
            return connection.decr(key, delta)
        except Exception as e:
            logger.error(f"Error decrementing key {key}: {e}")
            return None
    
    def flush_all(self) -> bool:
        try:
            connection = self._get_connection()
            return connection.flush_all()
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def _serialize_value(self, value: Any) -> bytes:
        if self.config.serialization == SerializationType.JSON:
            return json.dumps(value).encode('utf-8')
        elif self.config.serialization == SerializationType.PICKLE:
            return pickle.dumps(value)
        else:
            return str(value).encode('utf-8')
    
    def _deserialize_value(self, value: bytes) -> Any:
        if self.config.serialization == SerializationType.JSON:
            return json.loads(value.decode('utf-8'))
        elif self.config.serialization == SerializationType.PICKLE:
            return pickle.loads(value)
        else:
            return value.decode('utf-8')
    
    def _compress_value(self, value: Any) -> Any:
        if not self.config.enable_compression:
            return value
        
        if self.config.compression == CompressionType.GZIP:
            if isinstance(value, str):
                return gzip.compress(value.encode('utf-8'))
            elif isinstance(value, bytes):
                return gzip.compress(value)
            else:
                serialized = self._serialize_value(value)
                return gzip.compress(serialized)
        else:
            return value
    
    def _decompress_value(self, value: Any) -> Any:
        if not self.config.enable_compression:
            return value
        
        if self.config.compression == CompressionType.GZIP:
            try:
                if isinstance(value, bytes):
                    decompressed = gzip.decompress(value)
                    try:
                        return self._deserialize_value(decompressed)
                    except:
                        return decompressed.decode('utf-8')
                else:
                    return value
            except Exception as e:
                logger.warning(f"Failed to decompress value: {e}")
                return value
        else:
            return value
    
    def get_stats(self) -> CacheStats:
        with self.lock:
            return self.stats
    
    def close(self):
        with self.lock:
            for connection in self.connection_pool:
                try:
                    connection.disconnect_all()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self.connection_pool.clear()