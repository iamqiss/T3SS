// T3SS Project
// File: core/querying/cache/multi_layer_cache.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use lru::LruCache;
use dashmap::DashMap;

/// Multi-layer cache for ultra-fast search responses
pub struct MultiLayerCache {
    l1_cache: Arc<Mutex<LruCache<String, CacheEntry>>>,
    l2_cache: Arc<DashMap<String, CacheEntry>>,
    l3_cache: Arc<AsyncRwLock<HashMap<String, CacheEntry>>>,
    config: CacheConfig,
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub data: Vec<u8>,
    pub created_at: u64,
    pub accessed_at: u64,
    pub access_count: u64,
    pub ttl: Duration,
    pub size: usize,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub default_ttl: Duration,
    pub enable_compression: bool,
    pub enable_metrics: bool,
    pub max_key_length: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 1000,      // L1: In-memory LRU cache
            l2_size: 10000,     // L2: Concurrent hash map
            l3_size: 100000,    // L3: Async hash map
            default_ttl: Duration::from_secs(300), // 5 minutes
            enable_compression: true,
            enable_metrics: true,
            max_key_length: 255,
        }
    }
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub total_requests: u64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub average_response_time: Duration,
    pub cache_size: usize,
    pub memory_usage: u64,
}

impl MultiLayerCache {
    /// Create a new multi-layer cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            l1_cache: Arc::new(Mutex::new(LruCache::new(config.l1_size))),
            l2_cache: Arc::new(DashMap::with_capacity(config.l2_size)),
            l3_cache: Arc::new(AsyncRwLock::new(HashMap::with_capacity(config.l3_size))),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get a value from the cache with sub-millisecond performance
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let start_time = Instant::now();
        let mut stats = self.stats.lock().unwrap();
        stats.total_requests += 1;

        // L1 Cache (fastest - in-memory LRU)
        if let Some(entry) = self.get_from_l1(key).await {
            if !self.is_expired(&entry) {
                stats.l1_hits += 1;
                stats.total_hits += 1;
                self.update_access_time(&entry);
                stats.average_response_time = self.update_avg_time(stats.average_response_time, start_time.elapsed());
                return Some(entry.data);
            }
        }
        stats.l1_misses += 1;

        // L2 Cache (fast - concurrent hash map)
        if let Some(entry) = self.get_from_l2(key).await {
            if !self.is_expired(&entry) {
                stats.l2_hits += 1;
                stats.total_hits += 1;
                self.update_access_time(&entry);
                // Promote to L1
                self.set_l1(key, entry.clone()).await;
                stats.average_response_time = self.update_avg_time(stats.average_response_time, start_time.elapsed());
                return Some(entry.data);
            }
        }
        stats.l2_misses += 1;

        // L3 Cache (slower - async hash map)
        if let Some(entry) = self.get_from_l3(key).await {
            if !self.is_expired(&entry) {
                stats.l3_hits += 1;
                stats.total_hits += 1;
                self.update_access_time(&entry);
                // Promote to L2 and L1
                self.set_l2(key, entry.clone()).await;
                self.set_l1(key, entry.clone()).await;
                stats.average_response_time = self.update_avg_time(stats.average_response_time, start_time.elapsed());
                return Some(entry.data);
            }
        }
        stats.l3_misses += 1;
        stats.total_misses += 1;
        stats.average_response_time = self.update_avg_time(stats.average_response_time, start_time.elapsed());

        None
    }

    /// Set a value in the cache
    pub async fn set(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<(), String> {
        if key.len() > self.config.max_key_length {
            return Err("Key too long".to_string());
        }

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        
        let entry = CacheEntry {
            data: if self.config.enable_compression {
                self.compress_data(&value)?
            } else {
                value
            },
            created_at: now,
            accessed_at: now,
            access_count: 1,
            ttl,
            size: value.len(),
        };

        // Set in all cache layers
        self.set_l1(key, entry.clone()).await;
        self.set_l2(key, entry.clone()).await;
        self.set_l3(key, entry).await;

        Ok(())
    }

    /// Get from L1 cache (LRU)
    async fn get_from_l1(&self, key: &str) -> Option<CacheEntry> {
        let mut cache = self.l1_cache.lock().unwrap();
        cache.get(key).cloned()
    }

    /// Get from L2 cache (concurrent)
    async fn get_from_l2(&self, key: &str) -> Option<CacheEntry> {
        self.l2_cache.get(key).map(|entry| entry.clone())
    }

    /// Get from L3 cache (async)
    async fn get_from_l3(&self, key: &str) -> Option<CacheEntry> {
        let cache = self.l3_cache.read().await;
        cache.get(key).cloned()
    }

    /// Set in L1 cache
    async fn set_l1(&self, key: &str, entry: CacheEntry) {
        let mut cache = self.l1_cache.lock().unwrap();
        cache.put(key.to_string(), entry);
    }

    /// Set in L2 cache
    async fn set_l2(&self, key: &str, entry: CacheEntry) {
        self.l2_cache.insert(key.to_string(), entry);
    }

    /// Set in L3 cache
    async fn set_l3(&self, key: &str, entry: CacheEntry) {
        let mut cache = self.l3_cache.write().await;
        cache.insert(key.to_string(), entry);
    }

    /// Check if cache entry is expired
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now - entry.created_at > entry.ttl.as_secs()
    }

    /// Update access time for an entry
    fn update_access_time(&self, entry: &CacheEntry) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        // This would update the entry in place in a real implementation
    }

    /// Compress data for storage
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Simplified compression - in production, use a proper compression library
        if data.len() < 100 {
            Ok(data.to_vec())
        } else {
            // Placeholder for actual compression
            Ok(data.to_vec())
        }
    }

    /// Decompress data for retrieval
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // Simplified decompression - in production, use a proper decompression library
        Ok(data.to_vec())
    }

    /// Update average response time
    fn update_avg_time(&self, current: Duration, new: Duration) -> Duration {
        // Simple moving average
        Duration::from_nanos((current.as_nanos() + new.as_nanos()) / 2)
    }

    /// Clear all cache layers
    pub async fn clear(&self) {
        {
            let mut l1 = self.l1_cache.lock().unwrap();
            l1.clear();
        }
        self.l2_cache.clear();
        {
            let mut l3 = self.l3_cache.write().await;
            l3.clear();
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.lock().unwrap().clone();
        
        // Update current cache size
        stats.cache_size = self.l1_cache.lock().unwrap().len() + 
                          self.l2_cache.len() + 
                          self.l3_cache.try_read().map(|c| c.len()).unwrap_or(0);
        
        stats
    }

    /// Get cache hit rate
    pub fn get_hit_rate(&self) -> f32 {
        let stats = self.stats.lock().unwrap();
        if stats.total_requests == 0 {
            0.0
        } else {
            stats.total_hits as f32 / stats.total_requests as f32
        }
    }

    /// Warm up cache with frequently accessed data
    pub async fn warm_up(&self, warm_data: HashMap<String, Vec<u8>>) -> Result<(), String> {
        for (key, value) in warm_data {
            self.set(&key, value, None).await?;
        }
        Ok(())
    }

    /// Evict expired entries from all layers
    pub async fn evict_expired(&self) {
        // L1 cache (LRU handles eviction automatically)
        
        // L2 cache
        self.l2_cache.retain(|_, entry| !self.is_expired(entry));
        
        // L3 cache
        {
            let mut l3 = self.l3_cache.write().await;
            l3.retain(|_, entry| !self.is_expired(entry));
        }
    }

    /// Get memory usage estimate
    pub fn get_memory_usage(&self) -> u64 {
        let mut total = 0u64;
        
        // Estimate L1 memory usage
        {
            let l1 = self.l1_cache.lock().unwrap();
            total += l1.len() as u64 * 1024; // Rough estimate
        }
        
        // Estimate L2 memory usage
        total += self.l2_cache.len() as u64 * 1024;
        
        // Estimate L3 memory usage
        if let Ok(l3) = self.l3_cache.try_read() {
            total += l3.len() as u64 * 1024;
        }
        
        total
    }
}

/// Cache manager for coordinating multiple cache instances
pub struct CacheManager {
    caches: HashMap<String, Arc<MultiLayerCache>>,
    default_cache: Arc<MultiLayerCache>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new() -> Self {
        let default_config = CacheConfig::default();
        let default_cache = Arc::new(MultiLayerCache::new(default_config));
        
        Self {
            caches: HashMap::new(),
            default_cache,
        }
    }

    /// Get a cache by name
    pub fn get_cache(&self, name: &str) -> Arc<MultiLayerCache> {
        self.caches.get(name).cloned().unwrap_or_else(|| self.default_cache.clone())
    }

    /// Add a new cache
    pub fn add_cache(&mut self, name: String, cache: Arc<MultiLayerCache>) {
        self.caches.insert(name, cache);
    }

    /// Get default cache
    pub fn get_default_cache(&self) -> Arc<MultiLayerCache> {
        self.default_cache.clone()
    }

    /// Get aggregated statistics from all caches
    pub fn get_aggregated_stats(&self) -> CacheStats {
        let mut aggregated = CacheStats::default();
        
        for cache in self.caches.values() {
            let stats = cache.get_stats();
            aggregated.l1_hits += stats.l1_hits;
            aggregated.l1_misses += stats.l1_misses;
            aggregated.l2_hits += stats.l2_hits;
            aggregated.l2_misses += stats.l2_misses;
            aggregated.l3_hits += stats.l3_hits;
            aggregated.l3_misses += stats.l3_misses;
            aggregated.total_requests += stats.total_requests;
            aggregated.total_hits += stats.total_hits;
            aggregated.total_misses += stats.total_misses;
            aggregated.cache_size += stats.cache_size;
            aggregated.memory_usage += stats.memory_usage;
        }
        
        aggregated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_layer_cache() {
        let config = CacheConfig::default();
        let cache = MultiLayerCache::new(config);
        
        // Test set and get
        let key = "test_key";
        let value = b"test_value".to_vec();
        
        cache.set(key, value.clone(), None).await.unwrap();
        let retrieved = cache.get(key).await.unwrap();
        assert_eq!(retrieved, value);
        
        // Test cache hit rate
        let hit_rate = cache.get_hit_rate();
        assert!(hit_rate > 0.0);
        
        // Test statistics
        let stats = cache.get_stats();
        assert!(stats.total_requests > 0);
        assert!(stats.total_hits > 0);
    }

    #[tokio::test]
    async fn test_cache_manager() {
        let mut manager = CacheManager::new();
        
        let config = CacheConfig::default();
        let cache = Arc::new(MultiLayerCache::new(config));
        manager.add_cache("test_cache".to_string(), cache);
        
        let retrieved_cache = manager.get_cache("test_cache");
        assert!(Arc::ptr_eq(&retrieved_cache, &manager.caches["test_cache"]));
    }
}