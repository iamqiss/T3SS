// T3SS Project
// File: core/storage/distributed_storage_engine.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use uuid::Uuid;
use ring::digest;
use consistent_hash::ConsistentHash;
use sled::{Db, Tree};
use bytes::Bytes;

/// Represents a storage node in the distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageNode {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub capacity: u64,
    pub used_space: u64,
    pub status: NodeStatus,
    pub last_heartbeat: u64,
    pub replication_factor: u8,
    pub consistency_level: ConsistencyLevel,
}

/// Node status in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Inactive,
    Maintenance,
    Failed,
}

/// Consistency levels for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    One,        // At least one replica
    Quorum,     // Majority of replicas
    All,        // All replicas
    Strong,     // Strong consistency with consensus
}

/// Represents a data item in the distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    pub key: String,
    pub value: Vec<u8>,
    pub version: u64,
    pub timestamp: u64,
    pub checksum: String,
    pub replication_nodes: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Configuration for the distributed storage engine
#[derive(Debug, Clone)]
pub struct DistributedStorageConfig {
    pub replication_factor: u8,
    pub default_consistency_level: ConsistencyLevel,
    pub heartbeat_interval: Duration,
    pub failure_detection_timeout: Duration,
    pub max_value_size: usize,
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub compression_level: u8,
    pub enable_sharding: bool,
    pub shard_count: usize,
    pub enable_auto_rebalancing: bool,
    pub enable_quorum_reads: bool,
    pub enable_quorum_writes: bool,
}

impl Default for DistributedStorageConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            default_consistency_level: ConsistencyLevel::Quorum,
            heartbeat_interval: Duration::from_secs(5),
            failure_detection_timeout: Duration::from_secs(30),
            max_value_size: 10 * 1024 * 1024, // 10MB
            enable_compression: true,
            enable_encryption: false,
            compression_level: 6,
            enable_sharding: true,
            shard_count: 16,
            enable_auto_rebalancing: true,
            enable_quorum_reads: true,
            enable_quorum_writes: true,
        }
    }
}

/// High-performance distributed storage engine
pub struct DistributedStorageEngine {
    config: DistributedStorageConfig,
    nodes: Arc<RwLock<HashMap<String, StorageNode>>>,
    consistent_hash: Arc<RwLock<ConsistentHash<String>>>,
    local_db: Arc<Db>,
    replication_manager: Arc<Mutex<ReplicationManager>>,
    failure_detector: Arc<Mutex<FailureDetector>>,
    stats: Arc<Mutex<StorageStats>>,
    shard_manager: Arc<Mutex<ShardManager>>,
}

/// Manages replication across nodes
struct ReplicationManager {
    replication_queue: Vec<ReplicationTask>,
    active_replications: HashMap<String, ReplicationTask>,
    replication_stats: ReplicationStats,
}

/// Replication task
#[derive(Debug, Clone)]
struct ReplicationTask {
    pub id: String,
    pub key: String,
    pub value: Vec<u8>,
    pub source_node: String,
    pub target_nodes: Vec<String>,
    pub status: ReplicationStatus,
    pub created_at: u64,
    pub retry_count: u8,
}

/// Replication status
#[derive(Debug, Clone)]
enum ReplicationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Replication statistics
#[derive(Debug, Default)]
struct ReplicationStats {
    pub total_replications: u64,
    pub successful_replications: u64,
    pub failed_replications: u64,
    pub average_replication_time: Duration,
}

/// Detects node failures
struct FailureDetector {
    node_heartbeats: HashMap<String, u64>,
    failed_nodes: HashSet<String>,
    detection_threshold: u64,
}

/// Manages data sharding
struct ShardManager {
    shards: HashMap<u32, ShardInfo>,
    shard_assignments: HashMap<String, Vec<u32>>,
}

/// Information about a data shard
#[derive(Debug, Clone)]
struct ShardInfo {
    pub id: u32,
    pub size: u64,
    pub node_assignments: Vec<String>,
    pub status: ShardStatus,
}

/// Shard status
#[derive(Debug, Clone)]
enum ShardStatus {
    Active,
    Migrating,
    Failed,
}

/// Storage statistics
#[derive(Debug, Default)]
pub struct StorageStats {
    pub total_keys: u64,
    pub total_size: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub delete_operations: u64,
    pub replication_operations: u64,
    pub failed_operations: u64,
    pub average_read_latency: Duration,
    pub average_write_latency: Duration,
    pub active_nodes: u32,
    pub failed_nodes: u32,
}

impl DistributedStorageEngine {
    /// Create a new distributed storage engine
    pub fn new(config: DistributedStorageConfig) -> Result<Self, String> {
        // Initialize local database
        let local_db = sled::open("distributed_storage").map_err(|e| format!("Failed to open database: {}", e))?;
        
        // Initialize consistent hash ring
        let consistent_hash = Arc::new(RwLock::new(ConsistentHash::new()));
        
        // Initialize replication manager
        let replication_manager = Arc::new(Mutex::new(ReplicationManager {
            replication_queue: Vec::new(),
            active_replications: HashMap::new(),
            replication_stats: ReplicationStats::default(),
        }));
        
        // Initialize failure detector
        let failure_detector = Arc::new(Mutex::new(FailureDetector {
            node_heartbeats: HashMap::new(),
            failed_nodes: HashSet::new(),
            detection_threshold: config.failure_detection_timeout.as_secs(),
        }));
        
        // Initialize shard manager
        let shard_manager = Arc::new(Mutex::new(ShardManager {
            shards: HashMap::new(),
            shard_assignments: HashMap::new(),
        }));
        
        let stats = Arc::new(Mutex::new(StorageStats::default()));

        Ok(Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            consistent_hash,
            local_db: Arc::new(local_db),
            replication_manager,
            failure_detector,
            stats,
            shard_manager,
        })
    }

    /// Add a storage node to the cluster
    pub async fn add_node(&self, node: StorageNode) -> Result<(), String> {
        let mut nodes = self.nodes.write().unwrap();
        let mut hash_ring = self.consistent_hash.write().unwrap();
        
        // Add node to hash ring
        hash_ring.add_node(&node.id, node.capacity);
        
        // Store node information
        nodes.insert(node.id.clone(), node);
        
        // Update shard assignments
        self.update_shard_assignments().await?;
        
        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: &str) -> Result<(), String> {
        let mut nodes = self.nodes.write().unwrap();
        let mut hash_ring = self.consistent_hash.write().unwrap();
        
        // Remove from hash ring
        hash_ring.remove_node(node_id);
        
        // Remove node
        nodes.remove(node_id);
        
        // Rebalance data
        self.rebalance_data(node_id).await?;
        
        Ok(())
    }

    /// Store a key-value pair with replication
    pub async fn put(&self, key: String, value: Vec<u8>) -> Result<(), String> {
        let start_time = Instant::now();
        
        // Validate input
        if value.len() > self.config.max_value_size {
            return Err(format!("Value too large: {} bytes", value.len()));
        }

        // Generate checksum
        let checksum = self.calculate_checksum(&value);
        
        // Determine replication nodes
        let replication_nodes = self.get_replication_nodes(&key).await?;
        
        // Create data item
        let data_item = DataItem {
            key: key.clone(),
            value: value.clone(),
            version: self.get_next_version(&key).await,
            timestamp: self.current_timestamp(),
            checksum,
            replication_nodes: replication_nodes.clone(),
            metadata: HashMap::new(),
        };

        // Store locally first
        self.store_locally(&data_item).await?;
        
        // Replicate to other nodes
        self.replicate_data(&data_item).await?;
        
        // Update statistics
        self.update_write_stats(start_time.elapsed());
        
        Ok(())
    }

    /// Retrieve a value by key with consistency guarantees
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let start_time = Instant::now();
        
        // Try local storage first
        if let Some(data_item) = self.get_locally(key).await? {
            // Verify checksum
            if self.verify_checksum(&data_item.value, &data_item.checksum) {
                self.update_read_stats(start_time.elapsed());
                return Ok(Some(data_item.value));
            }
        }
        
        // If not found locally or checksum mismatch, try other nodes
        let replication_nodes = self.get_replication_nodes(key).await?;
        
        for node_id in replication_nodes {
            if let Some(data_item) = self.get_from_node(key, &node_id).await? {
                if self.verify_checksum(&data_item.value, &data_item.checksum) {
                    // Repair local copy if needed
                    self.store_locally(&data_item).await?;
                    self.update_read_stats(start_time.elapsed());
                    return Ok(Some(data_item.value));
                }
            }
        }
        
        self.update_read_stats(start_time.elapsed());
        Ok(None)
    }

    /// Delete a key from all replicas
    pub async fn delete(&self, key: &str) -> Result<(), String> {
        let start_time = Instant::now();
        
        // Delete locally
        self.delete_locally(key).await?;
        
        // Delete from replication nodes
        let replication_nodes = self.get_replication_nodes(key).await?;
        for node_id in replication_nodes {
            self.delete_from_node(key, &node_id).await?;
        }
        
        self.update_delete_stats(start_time.elapsed());
        Ok(())
    }

    /// Get replication nodes for a key
    async fn get_replication_nodes(&self, key: &str) -> Result<Vec<String>, String> {
        let hash_ring = self.consistent_hash.read().unwrap();
        let nodes = self.nodes.read().unwrap();
        
        // Get primary node
        let primary_node = hash_ring.get_node(key)
            .ok_or_else(|| "No nodes available".to_string())?;
        
        // Get additional replica nodes
        let mut replica_nodes = Vec::new();
        replica_nodes.push(primary_node.clone());
        
        // Add additional replicas
        let mut current_key = key.to_string();
        for _ in 1..self.config.replication_factor {
            current_key = format!("{}:{}", current_key, "replica");
            if let Some(node_id) = hash_ring.get_node(&current_key) {
                if !replica_nodes.contains(&node_id) {
                    replica_nodes.push(node_id);
                }
            }
        }
        
        // Filter out failed nodes
        let failed_nodes = self.get_failed_nodes().await;
        replica_nodes.retain(|node_id| !failed_nodes.contains(node_id));
        
        Ok(replica_nodes)
    }

    /// Store data locally
    async fn store_locally(&self, data_item: &DataItem) -> Result<(), String> {
        let serialized = bincode::serialize(data_item)
            .map_err(|e| format!("Serialization error: {}", e))?;
        
        self.local_db.insert(&data_item.key, serialized)
            .map_err(|e| format!("Database error: {}", e))?;
        
        Ok(())
    }

    /// Get data from local storage
    async fn get_locally(&self, key: &str) -> Result<Option<DataItem>, String> {
        if let Some(data) = self.local_db.get(key)
            .map_err(|e| format!("Database error: {}", e))? {
            let data_item: DataItem = bincode::deserialize(&data)
                .map_err(|e| format!("Deserialization error: {}", e))?;
            Ok(Some(data_item))
        } else {
            Ok(None)
        }
    }

    /// Delete data from local storage
    async fn delete_locally(&self, key: &str) -> Result<(), String> {
        self.local_db.remove(key)
            .map_err(|e| format!("Database error: {}", e))?;
        Ok(())
    }

    /// Replicate data to other nodes
    async fn replicate_data(&self, data_item: &DataItem) -> Result<(), String> {
        let mut replication_manager = self.replication_manager.lock().unwrap();
        
        for node_id in &data_item.replication_nodes {
            let replication_task = ReplicationTask {
                id: Uuid::new_v4().to_string(),
                key: data_item.key.clone(),
                value: data_item.value.clone(),
                source_node: "local".to_string(),
                target_nodes: vec![node_id.clone()],
                status: ReplicationStatus::Pending,
                created_at: self.current_timestamp(),
                retry_count: 0,
            };
            
            replication_manager.replication_queue.push(replication_task);
        }
        
        // Process replication queue
        self.process_replication_queue().await?;
        
        Ok(())
    }

    /// Process replication queue
    async fn process_replication_queue(&self) -> Result<(), String> {
        let mut replication_manager = self.replication_manager.lock().unwrap();
        let mut completed_tasks = Vec::new();
        
        for task in &mut replication_manager.replication_queue {
            if task.status == ReplicationStatus::Pending {
                task.status = ReplicationStatus::InProgress;
                
                // Simulate replication to target node
                // In production, this would make actual network calls
                if self.replicate_to_node(&task.key, &task.value, &task.target_nodes[0]).await? {
                    task.status = ReplicationStatus::Completed;
                    completed_tasks.push(task.id.clone());
                } else {
                    task.status = ReplicationStatus::Failed;
                    task.retry_count += 1;
                    
                    if task.retry_count < 3 {
                        task.status = ReplicationStatus::Pending;
                    }
                }
            }
        }
        
        // Remove completed tasks
        replication_manager.replication_queue.retain(|task| !completed_tasks.contains(&task.id));
        
        Ok(())
    }

    /// Replicate data to a specific node
    async fn replicate_to_node(&self, key: &str, value: &[u8], node_id: &str) -> Result<bool, String> {
        // In production, this would make an HTTP/gRPC call to the target node
        // For now, simulate success
        Ok(true)
    }

    /// Get data from a specific node
    async fn get_from_node(&self, key: &str, node_id: &str) -> Result<Option<DataItem>, String> {
        // In production, this would make an HTTP/gRPC call to the target node
        // For now, return None to simulate remote fetch
        Ok(None)
    }

    /// Delete data from a specific node
    async fn delete_from_node(&self, key: &str, node_id: &str) -> Result<(), String> {
        // In production, this would make an HTTP/gRPC call to the target node
        Ok(())
    }

    /// Calculate checksum for data
    fn calculate_checksum(&self, data: &[u8]) -> String {
        let hash = digest::digest(&digest::SHA256, data);
        hex::encode(hash.as_ref())
    }

    /// Verify checksum
    fn verify_checksum(&self, data: &[u8], checksum: &str) -> bool {
        let calculated_checksum = self.calculate_checksum(data);
        calculated_checksum == checksum
    }

    /// Get next version number for a key
    async fn get_next_version(&self, key: &str) -> u64 {
        // In production, this would use atomic counters
        self.current_timestamp()
    }

    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Get failed nodes
    async fn get_failed_nodes(&self) -> HashSet<String> {
        let failure_detector = self.failure_detector.lock().unwrap();
        failure_detector.failed_nodes.clone()
    }

    /// Update shard assignments
    async fn update_shard_assignments(&self) -> Result<(), String> {
        let mut shard_manager = self.shard_manager.lock().unwrap();
        let nodes = self.nodes.read().unwrap();
        
        // Clear existing assignments
        shard_manager.shard_assignments.clear();
        
        // Assign shards to nodes
        for (node_id, node) in nodes.iter() {
            if node.status == NodeStatus::Active {
                let mut node_shards = Vec::new();
                for shard_id in 0..self.config.shard_count {
                    if shard_id % nodes.len() == node_id.len() % nodes.len() {
                        node_shards.push(shard_id as u32);
                    }
                }
                shard_manager.shard_assignments.insert(node_id.clone(), node_shards);
            }
        }
        
        Ok(())
    }

    /// Rebalance data after node removal
    async fn rebalance_data(&self, removed_node_id: &str) -> Result<(), String> {
        // In production, this would:
        // 1. Identify data that was stored on the removed node
        // 2. Replicate that data to other nodes
        // 3. Update metadata and routing information
        
        Ok(())
    }

    /// Update write statistics
    fn update_write_stats(&self, latency: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.write_operations += 1;
        
        if stats.average_write_latency == Duration::default() {
            stats.average_write_latency = latency;
        } else {
            stats.average_write_latency = (stats.average_write_latency + latency) / 2;
        }
    }

    /// Update read statistics
    fn update_read_stats(&self, latency: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.read_operations += 1;
        
        if stats.average_read_latency == Duration::default() {
            stats.average_read_latency = latency;
        } else {
            stats.average_read_latency = (stats.average_read_latency + latency) / 2;
        }
    }

    /// Update delete statistics
    fn update_delete_stats(&self, latency: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.delete_operations += 1;
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cluster information
    pub async fn get_cluster_info(&self) -> ClusterInfo {
        let nodes = self.nodes.read().unwrap();
        let mut active_nodes = 0;
        let mut total_capacity = 0;
        let mut used_capacity = 0;
        
        for node in nodes.values() {
            if node.status == NodeStatus::Active {
                active_nodes += 1;
                total_capacity += node.capacity;
                used_capacity += node.used_space;
            }
        }
        
        ClusterInfo {
            total_nodes: nodes.len() as u32,
            active_nodes,
            total_capacity,
            used_capacity,
            replication_factor: self.config.replication_factor,
        }
    }
}

/// Cluster information
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub total_nodes: u32,
    pub active_nodes: u32,
    pub total_capacity: u64,
    pub used_capacity: u64,
    pub replication_factor: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_distributed_storage() {
        let config = DistributedStorageConfig::default();
        let storage = DistributedStorageEngine::new(config).unwrap();
        
        // Add a test node
        let node = StorageNode {
            id: "node1".to_string(),
            address: "127.0.0.1".to_string(),
            port: 8080,
            capacity: 1000000,
            used_space: 0,
            status: NodeStatus::Active,
            last_heartbeat: 1234567890,
            replication_factor: 3,
            consistency_level: ConsistencyLevel::Quorum,
        };
        
        storage.add_node(node).await.unwrap();
        
        // Test put operation
        let key = "test_key".to_string();
        let value = b"test_value".to_vec();
        storage.put(key.clone(), value.clone()).await.unwrap();
        
        // Test get operation
        let retrieved = storage.get(&key).await.unwrap();
        assert_eq!(retrieved, Some(value));
        
        // Test delete operation
        storage.delete(&key).await.unwrap();
        let retrieved_after_delete = storage.get(&key).await.unwrap();
        assert_eq!(retrieved_after_delete, None);
    }
}