// T3SS Project
// File: core/storage/caching/cache_invalidator.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use log::{info, warn, error, debug};
use tokio::sync::RwLock as AsyncRwLock;

/// Cache invalidation strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InvalidationStrategy {
    TimeBased,
    EventBased,
    DependencyBased,
    Manual,
    Hybrid,
}

/// Cache invalidation event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InvalidationEvent {
    DocumentUpdated(String),
    DocumentDeleted(String),
    IndexRebuilt,
    ConfigurationChanged,
    Custom(String),
}

/// Cache entry metadata for invalidation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntryMetadata {
    pub key: String,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub ttl: Option<Duration>,
    pub dependencies: HashSet<String>,
    pub tags: HashSet<String>,
    pub size: usize,
    pub priority: u8,
}

/// Configuration for cache invalidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInvalidatorConfig {
    pub strategy: InvalidationStrategy,
    pub default_ttl: Duration,
    pub max_entries: usize,
    pub cleanup_interval: Duration,
    pub enable_dependency_tracking: bool,
    pub enable_event_listening: bool,
    pub enable_metrics: bool,
}

impl Default for CacheInvalidatorConfig {
    fn default() -> Self {
        Self {
            strategy: InvalidationStrategy::Hybrid,
            default_ttl: Duration::from_secs(3600),
            max_entries: 100000,
            cleanup_interval: Duration::from_secs(300),
            enable_dependency_tracking: true,
            enable_event_listening: true,
            enable_metrics: true,
        }
    }
}

/// Cache invalidation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationRule {
    pub id: String,
    pub pattern: String,
    pub strategy: InvalidationStrategy,
    pub ttl: Option<Duration>,
    pub dependencies: Vec<String>,
    pub tags: Vec<String>,
    pub enabled: bool,
    pub priority: u8,
    pub created_at: Instant,
}

/// Cache invalidation statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InvalidationStats {
    pub total_invalidations: u64,
    pub successful_invalidations: u64,
    pub failed_invalidations: u64,
    pub time_based_invalidations: u64,
    pub event_based_invalidations: u64,
    pub dependency_invalidations: u64,
    pub manual_invalidations: u64,
    pub average_invalidation_time: Duration,
    pub cache_hit_rate: f64,
    pub total_entries_tracked: u64,
    pub active_rules: u64,
}

/// Advanced cache invalidator with multiple strategies
pub struct CacheInvalidator {
    config: CacheInvalidatorConfig,
    entries: Arc<AsyncRwLock<HashMap<String, CacheEntryMetadata>>>,
    rules: Arc<AsyncRwLock<Vec<InvalidationRule>>>,
    dependencies: Arc<AsyncRwLock<HashMap<String, HashSet<String>>>>,
    event_listeners: Arc<AsyncRwLock<Vec<Box<dyn Fn(InvalidationEvent) + Send + Sync>>>>,
    stats: Arc<AsyncRwLock<InvalidationStats>>,
}

impl CacheInvalidator {
    /// Creates a new cache invalidator
    pub fn new(config: CacheInvalidatorConfig) -> Self {
        Self {
            config,
            entries: Arc::new(AsyncRwLock::new(HashMap::new())),
            rules: Arc::new(AsyncRwLock::new(Vec::new())),
            dependencies: Arc::new(AsyncRwLock::new(HashMap::new())),
            event_listeners: Arc::new(AsyncRwLock::new(Vec::new())),
            stats: Arc::new(AsyncRwLock::new(InvalidationStats::default())),
        }
    }

    /// Initializes the cache invalidator
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing cache invalidator...");
        self.load_default_rules().await?;
        info!("Cache invalidator initialized successfully");
        Ok(())
    }

    /// Registers a cache entry for tracking
    pub async fn register_entry(&self, key: String, metadata: CacheEntryMetadata) -> Result<()> {
        let mut entries = self.entries.write().await;
        
        if entries.len() >= self.config.max_entries {
            self.evict_entries(&mut entries).await?;
        }
        
        entries.insert(key, metadata);
        
        let mut stats = self.stats.write().await;
        stats.total_entries_tracked += 1;
        
        debug!("Registered cache entry: {}", key);
        Ok(())
    }

    /// Invalidates cache entries based on strategy
    pub async fn invalidate(&self, pattern: &str, strategy: Option<InvalidationStrategy>) -> Result<u64> {
        let start_time = Instant::now();
        let strategy = strategy.unwrap_or(self.config.strategy.clone());
        
        let mut invalidated_count = 0;
        
        match strategy {
            InvalidationStrategy::TimeBased => {
                invalidated_count = self.invalidate_by_time().await?;
            }
            InvalidationStrategy::EventBased => {
                invalidated_count = self.invalidate_by_event(pattern).await?;
            }
            InvalidationStrategy::DependencyBased => {
                invalidated_count = self.invalidate_by_dependency(pattern).await?;
            }
            InvalidationStrategy::Manual => {
                invalidated_count = self.invalidate_manual(pattern).await?;
            }
            InvalidationStrategy::Hybrid => {
                invalidated_count = self.invalidate_hybrid(pattern).await?;
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.total_invalidations += 1;
        stats.successful_invalidations += 1;
        stats.average_invalidation_time = (stats.average_invalidation_time + start_time.elapsed()) / 2;
        
        match strategy {
            InvalidationStrategy::TimeBased => stats.time_based_invalidations += 1,
            InvalidationStrategy::EventBased => stats.event_based_invalidations += 1,
            InvalidationStrategy::DependencyBased => stats.dependency_invalidations += 1,
            InvalidationStrategy::Manual => stats.manual_invalidations += 1,
            InvalidationStrategy::Hybrid => {
                stats.time_based_invalidations += 1;
                stats.event_based_invalidations += 1;
            }
        }
        
        info!("Invalidated {} entries using {:?} strategy", invalidated_count, strategy);
        Ok(invalidated_count)
    }

    /// Invalidates entries by time (TTL)
    async fn invalidate_by_time(&self) -> Result<u64> {
        let mut entries = self.entries.write().await;
        let now = Instant::now();
        let mut to_remove = Vec::new();
        
        for (key, metadata) in entries.iter() {
            if let Some(ttl) = metadata.ttl {
                if now.duration_since(metadata.created_at) > ttl {
                    to_remove.push(key.clone());
                }
            } else if now.duration_since(metadata.last_accessed) > self.config.default_ttl {
                to_remove.push(key.clone());
            }
        }
        
        let count = to_remove.len() as u64;
        for key in to_remove {
            entries.remove(&key);
        }
        
        Ok(count)
    }

    /// Invalidates entries by event
    async fn invalidate_by_event(&self, pattern: &str) -> Result<u64> {
        let mut entries = self.entries.write().await;
        let mut to_remove = Vec::new();
        
        for (key, _metadata) in entries.iter() {
            if self.matches_pattern(key, pattern) {
                to_remove.push(key.clone());
            }
        }
        
        let count = to_remove.len() as u64;
        for key in to_remove {
            entries.remove(&key);
        }
        
        Ok(count)
    }

    /// Invalidates entries by dependency
    async fn invalidate_by_dependency(&self, pattern: &str) -> Result<u64> {
        let mut entries = self.entries.write().await;
        let mut to_remove = Vec::new();
        
        let dependencies = self.dependencies.read().await;
        if let Some(dependent_keys) = dependencies.get(pattern) {
            for key in dependent_keys {
                if entries.contains_key(key) {
                    to_remove.push(key.clone());
                }
            }
        }
        
        let count = to_remove.len() as u64;
        for key in to_remove {
            entries.remove(&key);
        }
        
        Ok(count)
    }

    /// Manual invalidation by pattern
    async fn invalidate_manual(&self, pattern: &str) -> Result<u64> {
        let mut entries = self.entries.write().await;
        let mut to_remove = Vec::new();
        
        for (key, _metadata) in entries.iter() {
            if self.matches_pattern(key, pattern) {
                to_remove.push(key.clone());
            }
        }
        
        let count = to_remove.len() as u64;
        for key in to_remove {
            entries.remove(&key);
        }
        
        Ok(count)
    }

    /// Hybrid invalidation combining multiple strategies
    async fn invalidate_hybrid(&self, pattern: &str) -> Result<u64> {
        let mut total_invalidated = 0;
        
        total_invalidated += self.invalidate_by_time().await?;
        total_invalidated += self.invalidate_by_event(pattern).await?;
        total_invalidated += self.invalidate_by_dependency(pattern).await?;
        
        Ok(total_invalidated)
    }

    /// Adds an invalidation rule
    pub async fn add_rule(&self, rule: InvalidationRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.push(rule);
        rules.sort_by_key(|r| r.priority);
        
        let mut stats = self.stats.write().await;
        stats.active_rules += 1;
        
        info!("Added invalidation rule: {}", rule.id);
        Ok(())
    }

    /// Adds a dependency relationship
    pub async fn add_dependency(&self, key: String, dependent_key: String) -> Result<()> {
        let mut dependencies = self.dependencies.write().await;
        dependencies.entry(key).or_insert_with(HashSet::new).insert(dependent_key);
        Ok(())
    }

    /// Evicts entries based on LRU and priority
    async fn evict_entries(&self, entries: &mut HashMap<String, CacheEntryMetadata>) -> Result<()> {
        let mut entries_vec: Vec<_> = entries.iter().collect();
        
        entries_vec.sort_by(|a, b| {
            a.1.priority.cmp(&b.1.priority)
                .then(a.1.last_accessed.cmp(&b.1.last_accessed))
        });
        
        let to_remove = entries_vec.len() / 10;
        for (key, _) in entries_vec.iter().take(to_remove) {
            entries.remove(*key);
        }
        
        Ok(())
    }

    /// Checks if a key matches a pattern
    fn matches_pattern(&self, key: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        
        if pattern.starts_with("*") && pattern.ends_with("*") {
            let inner = &pattern[1..pattern.len()-1];
            return key.contains(inner);
        }
        
        if pattern.starts_with("*") {
            return key.ends_with(&pattern[1..]);
        }
        
        if pattern.ends_with("*") {
            return key.starts_with(&pattern[..pattern.len()-1]);
        }
        
        key == pattern
    }

    /// Loads default invalidation rules
    async fn load_default_rules(&self) -> Result<()> {
        let default_rules = vec![
            InvalidationRule {
                id: "default_ttl".to_string(),
                pattern: "*".to_string(),
                strategy: InvalidationStrategy::TimeBased,
                ttl: Some(self.config.default_ttl),
                dependencies: vec![],
                tags: vec![],
                enabled: true,
                priority: 1,
                created_at: Instant::now(),
            },
        ];
        
        let mut rules = self.rules.write().await;
        rules.extend(default_rules);
        
        Ok(())
    }

    /// Gets invalidation statistics
    pub async fn get_stats(&self) -> InvalidationStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Gets the number of tracked entries
    pub async fn get_entry_count(&self) -> usize {
        let entries = self.entries.read().await;
        entries.len()
    }
}