// T3SS Project
// File: core/storage/distributed_storage_engine.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, BTreeMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write, Seek, SeekFrom};
use std::fs::{File, OpenOptions, create_dir_all};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock as AsyncRwLock, Mutex as AsyncMutex};
use tokio::time::{sleep, interval};
use rayon::prelude::*;
use ring::digest;
use ring::rand::{SecureRandom, SystemRandom};
use uuid::Uuid;
use async_trait::async_trait;
use etcd_rs::{Client, ClientConfig, PutRequest, GetRequest, DeleteRequest, WatchRequest};
use redis::aio::ConnectionManager;
use redis::{Client as RedisClient, Commands};
use sled::{Db, Tree};
use lz4_flex::{compress, decompress};
use brotli::{enc::BrotliEncoderParams, CompressorWriter, Decompressor};
use flate2::{Compression, write::GzEncoder, read::GzDecoder};

/// Storage node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageNode {
    pub id: String,
    pub address: SocketAddr,
    pub capacity: u64,
    pub used_space: u64,
    pub status: NodeStatus,
    pub last_heartbeat: u64,
    pub shards: Vec<u32>,
    pub metadata: HashMap<String, String>,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Inactive,
    Maintenance,
    Failed,
}

/// Data shard information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub id: u32,
    pub nodes: Vec<String>,
    pub replica_count: u32,
    pub status: ShardStatus,
    pub size: u64,
    pub last_updated: u64,
}

/// Shard status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardStatus {
    Healthy,
    Degraded,
    Failed,
    Rebalancing,
}

/// Storage operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageResult<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub latency_ms: u64,
    pub node_id: Option<String>,
}

/// Data item with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    pub key: String,
    pub value: Vec<u8>,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
    pub ttl: Option<u64>,
    pub version: u64,
    pub checksum: String,
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub cluster_name: String,
    pub node_id: String,
    pub listen_address: SocketAddr,
    pub etcd_endpoints: Vec<String>,
    pub redis_endpoints: Vec<String>,
    pub data_directory: PathBuf,
    pub max_shard_size: u64,
    pub replication_factor: u32,
    pub consistency_level: ConsistencyLevel,
    pub compression_enabled: bool,
    pub compression_algorithm: CompressionAlgorithm,
    pub enable_encryption: bool,
    pub encryption_key: Option<Vec<u8>>,
    pub heartbeat_interval: Duration,
    pub rebalance_threshold: f64,
    pub max_concurrent_operations: usize,
    pub cache_size: usize,
    pub enable_wal: bool,
    pub wal_directory: PathBuf,
}

/// Consistency levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    One,
    Quorum,
    All,
    Strong,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    LZ4,
    Brotli,
    Gzip,
    Snappy,
    None,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            cluster_name: "t3ss-storage".to_string(),
            node_id: Uuid::new_v4().to_string(),
            listen_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            etcd_endpoints: vec!["http://127.0.0.1:2379".to_string()],
            redis_endpoints: vec!["redis://127.0.0.1:6379".to_string()],
            data_directory: PathBuf::from("./data"),
            max_shard_size: 1024 * 1024 * 1024, // 1GB
            replication_factor: 3,
            consistency_level: ConsistencyLevel::Quorum,
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::LZ4,
            enable_encryption: false,
            encryption_key: None,
            heartbeat_interval: Duration::from_secs(30),
            rebalance_threshold: 0.8,
            max_concurrent_operations: 1000,
            cache_size: 10000,
            enable_wal: true,
            wal_directory: PathBuf::from("./wal"),
        }
    }
}

/// Consistent hashing ring
#[derive(Debug)]
pub struct ConsistentHashRing {
    ring: BTreeMap<u64, String>,
    virtual_nodes: u32,
    nodes: HashMap<String, StorageNode>,
}

impl ConsistentHashRing {
    pub fn new(virtual_nodes: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
            nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: StorageNode) {
        self.nodes.insert(node.id.clone(), node.clone());
        
        for i in 0..self.virtual_nodes {
            let hash_key = format!("{}:{}", node.id, i);
            let hash = self.hash(&hash_key);
            self.ring.insert(hash, node.id.clone());
        }
    }

    pub fn remove_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
        
        let mut to_remove = Vec::new();
        for (hash, id) in &self.ring {
            if id == node_id {
                to_remove.push(*hash);
            }
        }
        
        for hash in to_remove {
            self.ring.remove(&hash);
        }
    }

    pub fn get_nodes(&self, key: &str, count: usize) -> Vec<String> {
        let hash = self.hash(key);
        let mut nodes = Vec::new();
        let mut seen = HashSet::new();
        
        // Find the first node
        for (_, node_id) in self.ring.range(hash..) {
            if !seen.contains(node_id) {
                nodes.push(node_id.clone());
                seen.insert(node_id.clone());
                if nodes.len() >= count {
                    break;
                }
            }
        }
        
        // If we need more nodes, wrap around
        if nodes.len() < count {
            for (_, node_id) in self.ring.iter() {
                if !seen.contains(node_id) {
                    nodes.push(node_id.clone());
                    seen.insert(node_id.clone());
                    if nodes.len() >= count {
                        break;
                    }
                }
            }
        }
        
        nodes
    }

    fn hash(&self, key: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// Distributed storage engine
pub struct DistributedStorageEngine {
    config: StorageConfig,
    node: StorageNode,
    hash_ring: Arc<Mutex<ConsistentHashRing>>,
    shards: Arc<AsyncRwLock<HashMap<u32, Shard>>>,
    etcd_client: Option<Client>,
    redis_client: Option<ConnectionManager>,
    sled_db: Option<Db>,
    cache: Arc<Mutex<HashMap<String, DataItem>>>,
    wal: Arc<Mutex<Vec<WalEntry>>>,
    stats: Arc<Mutex<StorageStats>>,
    compression_engine: Arc<CompressionEngine>,
    encryption_engine: Arc<EncryptionEngine>,
}

/// Write-ahead log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WalEntry {
    pub operation: WalOperation,
    pub key: String,
    pub value: Option<Vec<u8>>,
    pub timestamp: u64,
    pub sequence: u64,
}

/// WAL operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum WalOperation {
    Put,
    Delete,
    Update,
}

/// Storage statistics
#[derive(Debug, Default)]
struct StorageStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_bytes_stored: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_latency_ms: f64,
    pub active_connections: u32,
}

/// Compression engine
struct CompressionEngine {
    algorithm: CompressionAlgorithm,
}

impl CompressionEngine {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self.algorithm {
            CompressionAlgorithm::LZ4 => {
                Ok(compress(data))
            },
            CompressionAlgorithm::Brotli => {
                let mut compressed = Vec::new();
                let mut encoder = CompressorWriter::new(&mut compressed, 4096, &BrotliEncoderParams::default());
                encoder.write_all(data)
                    .map_err(|e| format!("Brotli compression failed: {}", e))?;
                encoder.flush()
                    .map_err(|e| format!("Brotli flush failed: {}", e))?;
                Ok(compressed)
            },
            CompressionAlgorithm::Gzip => {
                let mut compressed = Vec::new();
                let mut encoder = GzEncoder::new(&mut compressed, Compression::default());
                encoder.write_all(data)
                    .map_err(|e| format!("Gzip compression failed: {}", e))?;
                encoder.finish()
                    .map_err(|e| format!("Gzip finish failed: {}", e))?;
                Ok(compressed)
            },
            CompressionAlgorithm::None => Ok(data.to_vec()),
            _ => Ok(data.to_vec()),
        }
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self.algorithm {
            CompressionAlgorithm::LZ4 => {
                decompress(data)
                    .map_err(|e| format!("LZ4 decompression failed: {}", e))
            },
            CompressionAlgorithm::Brotli => {
                let mut decompressed = Vec::new();
                let mut decompressor = Decompressor::new(data, 4096);
                decompressor.read_to_end(&mut decompressed)
                    .map_err(|e| format!("Brotli decompression failed: {}", e))?;
                Ok(decompressed)
            },
            CompressionAlgorithm::Gzip => {
                let mut decompressed = Vec::new();
                let mut decoder = GzDecoder::new(data);
                decoder.read_to_end(&mut decompressed)
                    .map_err(|e| format!("Gzip decompression failed: {}", e))?;
                Ok(decompressed)
            },
            CompressionAlgorithm::None => Ok(data.to_vec()),
            _ => Ok(data.to_vec()),
        }
    }
}

/// Encryption engine
struct EncryptionEngine {
    enabled: bool,
    key: Option<Vec<u8>>,
}

impl EncryptionEngine {
    fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        if !self.enabled {
            return Ok(data.to_vec());
        }
        
        // Simplified encryption - in production, use proper encryption
        let mut encrypted = Vec::new();
        if let Some(key) = &self.key {
            for (i, byte) in data.iter().enumerate() {
                encrypted.push(byte ^ key[i % key.len()]);
            }
        }
        Ok(encrypted)
    }

    fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        if !self.enabled {
            return Ok(data.to_vec());
        }
        
        // Simplified decryption - in production, use proper decryption
        let mut decrypted = Vec::new();
        if let Some(key) = &self.key {
            for (i, byte) in data.iter().enumerate() {
                decrypted.push(byte ^ key[i % key.len()]);
            }
        }
        Ok(decrypted)
    }
}

impl DistributedStorageEngine {
    /// Create a new distributed storage engine
    pub async fn new(config: StorageConfig) -> Result<Self, String> {
        // Create data directory
        create_dir_all(&config.data_directory)
            .map_err(|e| format!("Failed to create data directory: {}", e))?;

        if config.enable_wal {
            create_dir_all(&config.wal_directory)
                .map_err(|e| format!("Failed to create WAL directory: {}", e))?;
        }

        // Initialize storage node
        let node = StorageNode {
            id: config.node_id.clone(),
            address: config.listen_address,
            capacity: 1024 * 1024 * 1024 * 1024, // 1TB default
            used_space: 0,
            status: NodeStatus::Active,
            last_heartbeat: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            shards: Vec::new(),
            metadata: HashMap::new(),
        };

        // Initialize etcd client
        let etcd_client = if !config.etcd_endpoints.is_empty() {
            let etcd_config = ClientConfig::new(config.etcd_endpoints.clone());
            Some(Client::connect(etcd_config).await
                .map_err(|e| format!("Failed to connect to etcd: {}", e))?)
        } else {
            None
        };

        // Initialize Redis client
        let redis_client = if !config.redis_endpoints.is_empty() {
            let redis_url = &config.redis_endpoints[0];
            let client = RedisClient::open(redis_url)
                .map_err(|e| format!("Failed to create Redis client: {}", e))?;
            Some(client.get_connection_manager().await
                .map_err(|e| format!("Failed to connect to Redis: {}", e))?)
        } else {
            None
        };

        // Initialize Sled database
        let sled_db = Some(Db::open(&config.data_directory.join("sled"))
            .map_err(|e| format!("Failed to open Sled database: {}", e))?);

        // Initialize hash ring
        let hash_ring = Arc::new(Mutex::new(ConsistentHashRing::new(150)));

        // Initialize compression engine
        let compression_engine = Arc::new(CompressionEngine {
            algorithm: config.compression_algorithm.clone(),
        });

        // Initialize encryption engine
        let encryption_engine = Arc::new(EncryptionEngine {
            enabled: config.enable_encryption,
            key: config.encryption_key.clone(),
        });

        let engine = Self {
            config,
            node,
            hash_ring,
            shards: Arc::new(AsyncRwLock::new(HashMap::new())),
            etcd_client,
            redis_client,
            sled_db,
            cache: Arc::new(Mutex::new(HashMap::new())),
            wal: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(StorageStats::default())),
            compression_engine,
            encryption_engine,
        };

        // Register node with cluster
        engine.register_node().await?;

        // Start background tasks
        engine.start_background_tasks().await;

        Ok(engine)
    }

    /// Put data into storage
    pub async fn put(&self, key: String, value: Vec<u8>, metadata: HashMap<String, String>) -> Result<StorageResult<()>, String> {
        let start_time = Instant::now();
        
        // Create data item
        let checksum = self.calculate_checksum(&value);
        let mut data_item = DataItem {
            key: key.clone(),
            value: value.clone(),
            metadata,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            ttl: None,
            version: 1,
            checksum,
        };

        // Compress data if enabled
        if self.config.compression_enabled {
            data_item.value = self.compression_engine.compress(&data_item.value)?;
        }

        // Encrypt data if enabled
        if self.config.enable_encryption {
            data_item.value = self.encryption_engine.encrypt(&data_item.value)?;
        }

        // Get nodes for this key
        let nodes = self.get_nodes_for_key(&key).await;
        
        // Store on multiple nodes for replication
        let mut success_count = 0;
        let mut errors = Vec::new();

        for node_id in nodes {
            match self.store_on_node(&node_id, &data_item).await {
                Ok(_) => success_count += 1,
                Err(e) => errors.push(e),
            }
        }

        // Check consistency requirements
        let required_successes = self.get_required_successes();
        let success = success_count >= required_successes;

        // Update cache
        if success {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key.clone(), data_item);
        }

        // Write to WAL
        if self.config.enable_wal {
            self.write_to_wal(WalOperation::Put, &key, Some(value)).await;
        }

        // Update statistics
        self.update_stats(success, start_time.elapsed()).await;

        Ok(StorageResult {
            success,
            data: if success { Some(()) } else { None },
            error: if success { None } else { Some(errors.join("; ")) },
            latency_ms: start_time.elapsed().as_millis() as u64,
            node_id: Some(self.node.id.clone()),
        })
    }

    /// Get data from storage
    pub async fn get(&self, key: &str) -> Result<StorageResult<DataItem>, String> {
        let start_time = Instant::now();

        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(item) = cache.get(key) {
                self.update_cache_stats(true).await;
                return Ok(StorageResult {
                    success: true,
                    data: Some(item.clone()),
                    error: None,
                    latency_ms: start_time.elapsed().as_millis() as u64,
                    node_id: Some(self.node.id.clone()),
                });
            }
        }

        self.update_cache_stats(false).await;

        // Get nodes for this key
        let nodes = self.get_nodes_for_key(key).await;
        
        // Try to get from nodes
        for node_id in nodes {
            match self.get_from_node(&node_id, key).await {
                Ok(Some(mut item)) => {
                    // Decrypt if needed
                    if self.config.enable_encryption {
                        item.value = self.encryption_engine.decrypt(&item.value)?;
                    }

                    // Decompress if needed
                    if self.config.compression_enabled {
                        item.value = self.compression_engine.decompress(&item.value)?;
                    }

                    // Verify checksum
                    let calculated_checksum = self.calculate_checksum(&item.value);
                    if calculated_checksum != item.checksum {
                        continue; // Try next node
                    }

                    // Update cache
                    {
                        let mut cache = self.cache.lock().unwrap();
                        cache.insert(key.to_string(), item.clone());
                    }

                    // Update statistics
                    self.update_stats(true, start_time.elapsed()).await;

                    return Ok(StorageResult {
                        success: true,
                        data: Some(item),
                        error: None,
                        latency_ms: start_time.elapsed().as_millis() as u64,
                        node_id: Some(self.node.id.clone()),
                    });
                },
                Ok(None) => continue,
                Err(_) => continue,
            }
        }

        // Update statistics
        self.update_stats(false, start_time.elapsed()).await;

        Ok(StorageResult {
            success: false,
            data: None,
            error: Some("Key not found".to_string()),
            latency_ms: start_time.elapsed().as_millis() as u64,
            node_id: Some(self.node.id.clone()),
        })
    }

    /// Delete data from storage
    pub async fn delete(&self, key: &str) -> Result<StorageResult<()>, String> {
        let start_time = Instant::now();

        // Get nodes for this key
        let nodes = self.get_nodes_for_key(key).await;
        
        // Delete from nodes
        let mut success_count = 0;
        let mut errors = Vec::new();

        for node_id in nodes {
            match self.delete_from_node(&node_id, key).await {
                Ok(_) => success_count += 1,
                Err(e) => errors.push(e),
            }
        }

        // Check consistency requirements
        let required_successes = self.get_required_successes();
        let success = success_count >= required_successes;

        // Remove from cache
        if success {
            let mut cache = self.cache.lock().unwrap();
            cache.remove(key);
        }

        // Write to WAL
        if self.config.enable_wal {
            self.write_to_wal(WalOperation::Delete, key, None).await;
        }

        // Update statistics
        self.update_stats(success, start_time.elapsed()).await;

        Ok(StorageResult {
            success,
            data: if success { Some(()) } else { None },
            error: if success { None } else { Some(errors.join("; ")) },
            latency_ms: start_time.elapsed().as_millis() as u64,
            node_id: Some(self.node.id.clone()),
        })
    }

    /// Get nodes responsible for a key
    async fn get_nodes_for_key(&self, key: &str) -> Vec<String> {
        let hash_ring = self.hash_ring.lock().unwrap();
        hash_ring.get_nodes(key, self.config.replication_factor as usize)
    }

    /// Get required number of successful operations based on consistency level
    fn get_required_successes(&self) -> usize {
        match self.config.consistency_level {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => (self.config.replication_factor / 2 + 1) as usize,
            ConsistencyLevel::All => self.config.replication_factor as usize,
            ConsistencyLevel::Strong => self.config.replication_factor as usize,
        }
    }

    /// Store data on a specific node
    async fn store_on_node(&self, node_id: &str, item: &DataItem) -> Result<(), String> {
        if node_id == &self.node.id {
            // Store locally
            self.store_locally(item).await
        } else {
            // Store on remote node (would implement RPC call)
            self.store_remote(node_id, item).await
        }
    }

    /// Store data locally
    async fn store_locally(&self, item: &DataItem) -> Result<(), String> {
        // Store in Sled database
        if let Some(db) = &self.sled_db {
            let serialized = bincode::serialize(item)
                .map_err(|e| format!("Serialization failed: {}", e))?;
            db.insert(&item.key, serialized)
                .map_err(|e| format!("Sled insert failed: {}", e))?;
        }

        // Store in Redis if available
        if let Some(redis) = &self.redis_client {
            let serialized = bincode::serialize(item)
                .map_err(|e| format!("Serialization failed: {}", e))?;
            let _: () = redis.set(&item.key, serialized).await
                .map_err(|e| format!("Redis set failed: {}", e))?;
        }

        Ok(())
    }

    /// Store data on remote node
    async fn store_remote(&self, _node_id: &str, _item: &DataItem) -> Result<(), String> {
        // In production, this would make an RPC call to the remote node
        // For now, we'll simulate success
        Ok(())
    }

    /// Get data from a specific node
    async fn get_from_node(&self, node_id: &str, key: &str) -> Result<Option<DataItem>, String> {
        if node_id == &self.node.id {
            // Get from local storage
            self.get_locally(key).await
        } else {
            // Get from remote node (would implement RPC call)
            self.get_remote(node_id, key).await
        }
    }

    /// Get data from local storage
    async fn get_locally(&self, key: &str) -> Result<Option<DataItem>, String> {
        // Try Sled first
        if let Some(db) = &self.sled_db {
            if let Some(data) = db.get(key)
                .map_err(|e| format!("Sled get failed: {}", e))? {
                let item: DataItem = bincode::deserialize(&data)
                    .map_err(|e| format!("Deserialization failed: {}", e))?;
                return Ok(Some(item));
            }
        }

        // Try Redis
        if let Some(redis) = &self.redis_client {
            if let Ok(data) = redis.get::<_, Vec<u8>>(key).await {
                let item: DataItem = bincode::deserialize(&data)
                    .map_err(|e| format!("Deserialization failed: {}", e))?;
                return Ok(Some(item));
            }
        }

        Ok(None)
    }

    /// Get data from remote node
    async fn get_remote(&self, _node_id: &str, _key: &str) -> Result<Option<DataItem>, String> {
        // In production, this would make an RPC call to the remote node
        // For now, we'll return None
        Ok(None)
    }

    /// Delete data from a specific node
    async fn delete_from_node(&self, node_id: &str, key: &str) -> Result<(), String> {
        if node_id == &self.node.id {
            // Delete from local storage
            self.delete_locally(key).await
        } else {
            // Delete from remote node (would implement RPC call)
            self.delete_remote(node_id, key).await
        }
    }

    /// Delete data from local storage
    async fn delete_locally(&self, key: &str) -> Result<(), String> {
        // Delete from Sled
        if let Some(db) = &self.sled_db {
            db.remove(key)
                .map_err(|e| format!("Sled remove failed: {}", e))?;
        }

        // Delete from Redis
        if let Some(redis) = &self.redis_client {
            let _: () = redis.del(key).await
                .map_err(|e| format!("Redis del failed: {}", e))?;
        }

        Ok(())
    }

    /// Delete data from remote node
    async fn delete_remote(&self, _node_id: &str, _key: &str) -> Result<(), String> {
        // In production, this would make an RPC call to the remote node
        // For now, we'll simulate success
        Ok(())
    }

    /// Calculate checksum for data
    fn calculate_checksum(&self, data: &[u8]) -> String {
        let hash = digest::digest(&digest::SHA256, data);
        hex::encode(hash.as_ref())
    }

    /// Write to write-ahead log
    async fn write_to_wal(&self, operation: WalOperation, key: &str, value: Option<Vec<u8>>) {
        let entry = WalEntry {
            operation,
            key: key.to_string(),
            value,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            sequence: 0, // Would be incremented in production
        };

        let mut wal = self.wal.lock().unwrap();
        wal.push(entry);
    }

    /// Register node with cluster
    async fn register_node(&self) -> Result<(), String> {
        if let Some(etcd) = &self.etcd_client {
            let node_data = serde_json::to_string(&self.node)
                .map_err(|e| format!("Failed to serialize node: {}", e))?;
            
            let key = format!("/nodes/{}", self.node.id);
            let _ = etcd.put(PutRequest::new(key, node_data)).await
                .map_err(|e| format!("Failed to register node: {}", e))?;
        }
        Ok(())
    }

    /// Start background tasks
    async fn start_background_tasks(&self) {
        // Heartbeat task
        let node_id = self.node.id.clone();
        let etcd_client = self.etcd_client.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                // Send heartbeat
                if let Some(etcd) = &etcd_client {
                    let heartbeat_data = serde_json::json!({
                        "node_id": node_id,
                        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        "status": "active"
                    });
                    let key = format!("/heartbeats/{}", node_id);
                    let _ = etcd.put(PutRequest::new(key, heartbeat_data.to_string())).await;
                }
            }
        });

        // Rebalancing task
        let hash_ring = self.hash_ring.clone();
        let shards = self.shards.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // 5 minutes
            loop {
                interval.tick().await;
                // Check for rebalancing needs
                // This would implement rebalancing logic
            }
        });
    }

    /// Update statistics
    async fn update_stats(&self, success: bool, duration: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_operations += 1;
        if success {
            stats.successful_operations += 1;
        } else {
            stats.failed_operations += 1;
        }
        
        // Update average latency
        let latency_ms = duration.as_millis() as f64;
        stats.average_latency_ms = (stats.average_latency_ms * (stats.total_operations - 1) as f64 + latency_ms) / stats.total_operations as f64;
    }

    /// Update cache statistics
    async fn update_cache_stats(&self, hit: bool) {
        let mut stats = self.stats.lock().unwrap();
        if hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cluster information
    pub async fn get_cluster_info(&self) -> Result<HashMap<String, StorageNode>, String> {
        // In production, this would fetch from etcd
        let mut nodes = HashMap::new();
        nodes.insert(self.node.id.clone(), self.node.clone());
        Ok(nodes)
    }

    /// Shutdown the storage engine
    pub async fn shutdown(&self) -> Result<(), String> {
        // Flush WAL
        if self.config.enable_wal {
            self.flush_wal().await?;
        }

        // Unregister node
        if let Some(etcd) = &self.etcd_client {
            let key = format!("/nodes/{}", self.node.id);
            let _ = etcd.delete(DeleteRequest::new(key)).await;
        }

        Ok(())
    }

    /// Flush write-ahead log
    async fn flush_wal(&self) -> Result<(), String> {
        let wal = self.wal.lock().unwrap();
        if wal.is_empty() {
            return Ok(());
        }

        // In production, this would flush WAL entries to persistent storage
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_engine() {
        let config = StorageConfig::default();
        let engine = DistributedStorageEngine::new(config).await.unwrap();
        
        let key = "test_key".to_string();
        let value = b"test_value".to_vec();
        let metadata = HashMap::new();
        
        // Test put
        let result = engine.put(key.clone(), value, metadata).await.unwrap();
        assert!(result.success);
        
        // Test get
        let result = engine.get(&key).await.unwrap();
        assert!(result.success);
        assert!(result.data.is_some());
        
        // Test delete
        let result = engine.delete(&key).await.unwrap();
        assert!(result.success);
    }
}