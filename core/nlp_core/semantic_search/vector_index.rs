// T3SS Project
// File: core/nlp_core/semantic_search/vector_index.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use uuid::Uuid;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use faiss::{Index, IndexFlat, IndexIVFFlat, IndexHNSW, MetricType};
use hnsw_rs::{Hnsw, HnswParams, DistEnum};
use rand::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use memmap2::{Mmap, MmapOptions};

/// Configuration for vector indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexConfig {
    pub dimension: usize,
    pub max_vectors: usize,
    pub index_type: IndexType,
    pub metric: DistanceMetric,
    pub enable_compression: bool,
    pub compression_ratio: f32,
    pub enable_quantization: bool,
    pub quantization_bits: u8,
    pub enable_hnsw: bool,
    pub hnsw_params: HnswParams,
    pub enable_faiss: bool,
    pub faiss_params: FaissParams,
    pub enable_memory_mapping: bool,
    pub memory_limit: usize,
    pub enable_parallel_indexing: bool,
    pub batch_size: usize,
    pub enable_caching: bool,
    pub cache_size: usize,
    pub enable_persistence: bool,
    pub persistence_interval: Duration,
    pub index_file_path: String,
}

/// Types of vector indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Flat,           // Linear search
    IVF,            // Inverted File
    HNSW,           // Hierarchical Navigable Small World
    LSH,            // Locality Sensitive Hashing
    PQ,             // Product Quantization
    Hybrid,         // Combination of multiple indices
}

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    DotProduct,
    Jaccard,
    Hamming,
}

/// FAISS-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissParams {
    pub nlist: usize,           // Number of clusters for IVF
    pub nprobe: usize,          // Number of clusters to search
    pub ef_construction: usize, // HNSW construction parameter
    pub ef_search: usize,       // HNSW search parameter
    pub m: usize,               // HNSW connectivity parameter
}

impl Default for FaissParams {
    fn default() -> Self {
        Self {
            nlist: 1024,
            nprobe: 32,
            ef_construction: 200,
            ef_search: 50,
            m: 16,
        }
    }
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            max_vectors: 1_000_000,
            index_type: IndexType::HNSW,
            metric: DistanceMetric::Cosine,
            enable_compression: true,
            compression_ratio: 0.5,
            enable_quantization: true,
            quantization_bits: 8,
            enable_hnsw: true,
            hnsw_params: HnswParams::new(768, 16, 200, 50, DistEnum::Cosine),
            enable_faiss: true,
            faiss_params: FaissParams::default(),
            enable_memory_mapping: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            enable_parallel_indexing: true,
            batch_size: 1000,
            enable_caching: true,
            cache_size: 10000,
            enable_persistence: true,
            persistence_interval: Duration::from_secs(300), // 5 minutes
            index_file_path: "vector_index.bin".to_string(),
        }
    }
}

/// Represents a vector document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
    pub quality_score: f32,
    pub embedding_model: String,
}

/// Search result from vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub document: VectorDocument,
    pub score: f32,
    pub distance: f32,
    pub rank: usize,
}

/// Vector index statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VectorIndexStats {
    pub total_vectors: u64,
    pub index_size_bytes: u64,
    pub memory_usage: u64,
    pub average_vector_dimension: f32,
    pub index_build_time: Duration,
    pub last_updated: u64,
    pub search_count: u64,
    pub average_search_time: Duration,
    pub cache_hit_rate: f32,
}

/// High-performance vector index with multiple backends
pub struct VectorIndex {
    config: VectorIndexConfig,
    documents: Arc<AsyncRwLock<HashMap<String, VectorDocument>>>,
    vectors: Arc<AsyncRwLock<Vec<Vec<f32>>>>,
    vector_ids: Arc<AsyncRwLock<Vec<String>>>,
    
    // Index backends
    faiss_index: Option<Arc<Mutex<Box<dyn Index + Send + Sync>>>>,
    hnsw_index: Option<Arc<Mutex<Hnsw<f32, DistEnum>>>>,
    
    // Caching
    search_cache: Arc<Mutex<HashMap<String, Vec<VectorSearchResult>>>>,
    
    // Statistics
    stats: Arc<Mutex<VectorIndexStats>>,
    
    // Persistence
    index_file: Option<File>,
    memory_map: Option<Mmap>,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(config: VectorIndexConfig) -> Result<Self, String> {
        let documents = Arc::new(AsyncRwLock::new(HashMap::new()));
        let vectors = Arc::new(AsyncRwLock::new(Vec::new()));
        let vector_ids = Arc::new(AsyncRwLock::new(Vec::new()));
        
        // Initialize FAISS index if enabled
        let faiss_index = if config.enable_faiss {
            Some(Arc::new(Mutex::new(Self::create_faiss_index(&config)?)))
        } else {
            None
        };
        
        // Initialize HNSW index if enabled
        let hnsw_index = if config.enable_hnsw {
            Some(Arc::new(Mutex::new(Self::create_hnsw_index(&config)?)))
        } else {
            None
        };
        
        // Initialize search cache
        let search_cache = Arc::new(Mutex::new(HashMap::new()));
        
        // Initialize statistics
        let stats = Arc::new(Mutex::new(VectorIndexStats::default()));
        
        // Initialize persistence
        let index_file = if config.enable_persistence {
            Some(OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(&config.index_file_path)
                .map_err(|e| format!("Failed to open index file: {}", e))?)
        } else {
            None
        };
        
        // Load existing index if available
        let memory_map = if config.enable_memory_mapping && index_file.is_some() {
            Some(unsafe {
                MmapOptions::new()
                    .map(&index_file.as_ref().unwrap())
                    .map_err(|e| format!("Failed to create memory map: {}", e))?
            })
        } else {
            None
        };
        
        Ok(Self {
            config,
            documents,
            vectors,
            vector_ids,
            faiss_index,
            hnsw_index,
            search_cache,
            stats,
            index_file,
            memory_map,
        })
    }
    
    /// Add a vector document to the index
    pub async fn add_document(&self, doc: VectorDocument) -> Result<(), String> {
        // Validate vector dimension
        if doc.vector.len() != self.config.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                doc.vector.len()
            ));
        }
        
        // Add to document store
        {
            let mut documents = self.documents.write().await;
            documents.insert(doc.id.clone(), doc.clone());
        }
        
        // Add to vector storage
        {
            let mut vectors = self.vectors.write().await;
            let mut vector_ids = self.vector_ids.write().await;
            
            vectors.push(doc.vector.clone());
            vector_ids.push(doc.id.clone());
        }
        
        // Add to FAISS index
        if let Some(faiss_index) = &self.faiss_index {
            let mut index = faiss_index.lock().unwrap();
            let vector_array = Array2::from_shape_vec((1, self.config.dimension), doc.vector.clone())
                .map_err(|e| format!("Failed to create vector array: {}", e))?;
            
            index.add(&vector_array).map_err(|e| format!("FAISS add failed: {}", e))?;
        }
        
        // Add to HNSW index
        if let Some(hnsw_index) = &self.hnsw_index {
            let mut index = hnsw_index.lock().unwrap();
            index.insert(&doc.vector, &doc.id).map_err(|e| format!("HNSW insert failed: {}", e))?;
        }
        
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_vectors += 1;
            stats.last_updated = doc.timestamp;
        }
        
        Ok(())
    }
    
    /// Add multiple documents in batch
    pub async fn add_documents_batch(&self, docs: Vec<VectorDocument>) -> Result<(), String> {
        if self.config.enable_parallel_indexing {
            self.add_documents_parallel(docs).await
        } else {
            self.add_documents_sequential(docs).await
        }
    }
    
    /// Add documents sequentially
    async fn add_documents_sequential(&self, docs: Vec<VectorDocument>) -> Result<(), String> {
        for doc in docs {
            self.add_document(doc).await?;
        }
        Ok(())
    }
    
    /// Add documents in parallel
    async fn add_documents_parallel(&self, docs: Vec<VectorDocument>) -> Result<(), String> {
        let chunks: Vec<Vec<VectorDocument>> = docs
            .chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let results: Result<Vec<_>, _> = chunks
            .into_par_iter()
            .map(|chunk| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    for doc in chunk {
                        self.add_document(doc).await?;
                    }
                    Ok::<(), String>(())
                })
            })
            .collect();
        
        results?;
        Ok(())
    }
    
    /// Search for similar vectors
    pub async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<VectorSearchResult>, String> {
        let start_time = Instant::now();
        
        // Validate query vector
        if query_vector.len() != self.config.dimension {
            return Err(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                query_vector.len()
            ));
        }
        
        // Check cache first
        let cache_key = self.generate_cache_key(query_vector, k);
        if self.config.enable_caching {
            if let Some(cached_results) = self.get_from_cache(&cache_key).await {
                self.update_search_stats(true, start_time.elapsed());
                return Ok(cached_results);
            }
        }
        
        let mut results = Vec::new();
        
        // Search using FAISS if available
        if let Some(faiss_index) = &self.faiss_index {
            let faiss_results = self.search_faiss(faiss_index, query_vector, k).await?;
            results.extend(faiss_results);
        }
        
        // Search using HNSW if available
        if let Some(hnsw_index) = &self.hnsw_index {
            let hnsw_results = self.search_hnsw(hnsw_index, query_vector, k).await?;
            results.extend(hnsw_results);
        }
        
        // If no specialized indices, use linear search
        if results.is_empty() {
            results = self.search_linear(query_vector, k).await?;
        }
        
        // Sort by score and limit results
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        
        // Assign ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }
        
        // Cache results
        if self.config.enable_caching {
            self.set_cache(&cache_key, &results).await;
        }
        
        self.update_search_stats(false, start_time.elapsed());
        Ok(results)
    }
    
    /// Search using FAISS index
    async fn search_faiss(&self, faiss_index: &Arc<Mutex<Box<dyn Index + Send + Sync>>>, 
                         query_vector: &[f32], k: usize) -> Result<Vec<VectorSearchResult>, String> {
        let index = faiss_index.lock().unwrap();
        let query_array = Array2::from_shape_vec((1, self.config.dimension), query_vector.to_vec())
            .map_err(|e| format!("Failed to create query array: {}", e))?;
        
        let (distances, indices) = index.search(&query_array, k)
            .map_err(|e| format!("FAISS search failed: {}", e))?;
        
        let mut results = Vec::new();
        let documents = self.documents.read().await;
        let vector_ids = self.vector_ids.read().await;
        
        for (i, &idx) in indices.iter().enumerate() {
            if let Some(doc_id) = vector_ids.get(idx as usize) {
                if let Some(doc) = documents.get(doc_id) {
                    let distance = distances[i];
                    let score = self.distance_to_score(distance);
                    
                    results.push(VectorSearchResult {
                        document: doc.clone(),
                        score,
                        distance,
                        rank: 0, // Will be set later
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Search using HNSW index
    async fn search_hnsw(&self, hnsw_index: &Arc<Mutex<Hnsw<f32, DistEnum>>>, 
                        query_vector: &[f32], k: usize) -> Result<Vec<VectorSearchResult>, String> {
        let index = hnsw_index.lock().unwrap();
        let search_results = index.search(query_vector, k, 0)
            .map_err(|e| format!("HNSW search failed: {}", e))?;
        
        let mut results = Vec::new();
        let documents = self.documents.read().await;
        
        for (i, (doc_id, distance)) in search_results.iter().enumerate() {
            if let Some(doc) = documents.get(doc_id) {
                let score = self.distance_to_score(*distance);
                
                results.push(VectorSearchResult {
                    document: doc.clone(),
                    score,
                    distance: *distance,
                    rank: 0, // Will be set later
                });
            }
        }
        
        Ok(results)
    }
    
    /// Linear search fallback
    async fn search_linear(&self, query_vector: &[f32], k: usize) -> Result<Vec<VectorSearchResult>, String> {
        let vectors = self.vectors.read().await;
        let vector_ids = self.vector_ids.read().await;
        let documents = self.documents.read().await;
        
        let mut results = Vec::new();
        
        for (i, vector) in vectors.iter().enumerate() {
            if let Some(doc_id) = vector_ids.get(i) {
                if let Some(doc) = documents.get(doc_id) {
                    let distance = self.calculate_distance(query_vector, vector);
                    let score = self.distance_to_score(distance);
                    
                    results.push(VectorSearchResult {
                        document: doc.clone(),
                        score,
                        distance,
                        rank: 0, // Will be set later
                    });
                }
            }
        }
        
        // Sort by distance and take top k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Calculate distance between two vectors
    fn calculate_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => self.cosine_distance(vec1, vec2),
            DistanceMetric::Euclidean => self.euclidean_distance(vec1, vec2),
            DistanceMetric::Manhattan => self.manhattan_distance(vec1, vec2),
            DistanceMetric::DotProduct => self.dot_product_distance(vec1, vec2),
            DistanceMetric::Jaccard => self.jaccard_distance(vec1, vec2),
            DistanceMetric::Hamming => self.hamming_distance(vec1, vec2),
        }
    }
    
    /// Cosine distance
    fn cosine_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|a| a * a).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            1.0
        } else {
            1.0 - (dot_product / (norm1 * norm2))
        }
    }
    
    /// Euclidean distance
    fn euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Manhattan distance
    fn manhattan_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }
    
    /// Dot product distance
    fn dot_product_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        -vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum::<f32>()
    }
    
    /// Jaccard distance
    fn jaccard_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        let set1: HashSet<u32> = vec1.iter().map(|&x| (x * 1000.0) as u32).collect();
        let set2: HashSet<u32> = vec2.iter().map(|&x| (x * 1000.0) as u32).collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            1.0
        } else {
            1.0 - (intersection as f32 / union as f32)
        }
    }
    
    /// Hamming distance
    fn hamming_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| if (a - b).abs() < 1e-6 { 0.0 } else { 1.0 })
            .sum()
    }
    
    /// Convert distance to similarity score
    fn distance_to_score(&self, distance: f32) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Jaccard => 1.0 - distance,
            DistanceMetric::Hamming => 1.0 - distance,
        }
    }
    
    /// Create FAISS index
    fn create_faiss_index(config: &VectorIndexConfig) -> Result<Box<dyn Index + Send + Sync>, String> {
        let metric = match config.metric {
            DistanceMetric::Cosine => MetricType::METRIC_INNER_PRODUCT,
            DistanceMetric::Euclidean => MetricType::METRIC_L2,
            DistanceMetric::Manhattan => MetricType::METRIC_L1,
            DistanceMetric::DotProduct => MetricType::METRIC_INNER_PRODUCT,
            _ => MetricType::METRIC_L2,
        };
        
        match config.index_type {
            IndexType::Flat => {
                let index = IndexFlat::new(config.dimension, metric)
                    .map_err(|e| format!("Failed to create FAISS flat index: {}", e))?;
                Ok(Box::new(index))
            },
            IndexType::IVF => {
                let quantizer = IndexFlat::new(config.dimension, metric)
                    .map_err(|e| format!("Failed to create FAISS quantizer: {}", e))?;
                let index = IndexIVFFlat::new(quantizer, config.dimension, config.faiss_params.nlist, metric)
                    .map_err(|e| format!("Failed to create FAISS IVF index: {}", e))?;
                Ok(Box::new(index))
            },
            _ => {
                // Fallback to flat index
                let index = IndexFlat::new(config.dimension, metric)
                    .map_err(|e| format!("Failed to create FAISS index: {}", e))?;
                Ok(Box::new(index))
            }
        }
    }
    
    /// Create HNSW index
    fn create_hnsw_index(config: &VectorIndexConfig) -> Result<Hnsw<f32, DistEnum>, String> {
        let dist_enum = match config.metric {
            DistanceMetric::Cosine => DistEnum::Cosine,
            DistanceMetric::Euclidean => DistEnum::L2,
            DistanceMetric::Manhattan => DistEnum::L1,
            _ => DistEnum::L2,
        };
        
        let params = HnswParams::new(
            config.dimension,
            config.hnsw_params.m,
            config.hnsw_params.ef_construction,
            config.hnsw_params.ef_search,
            dist_enum,
        );
        
        Hnsw::new(&params, config.max_vectors)
            .map_err(|e| format!("Failed to create HNSW index: {}", e))
    }
    
    /// Generate cache key for search
    fn generate_cache_key(&self, query_vector: &[f32], k: usize) -> String {
        let vector_hash = query_vector.iter()
            .map(|&x| (x * 1000.0) as i32)
            .collect::<Vec<_>>();
        format!("search_{:?}_{}", vector_hash, k)
    }
    
    /// Get results from cache
    async fn get_from_cache(&self, key: &str) -> Option<Vec<VectorSearchResult>> {
        let cache = self.search_cache.lock().unwrap();
        cache.get(key).cloned()
    }
    
    /// Set results in cache
    async fn set_cache(&self, key: &str, results: &[VectorSearchResult]) {
        let mut cache = self.search_cache.lock().unwrap();
        if cache.len() >= self.config.cache_size {
            // Simple LRU eviction - remove oldest entry
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }
        cache.insert(key.to_string(), results.to_vec());
    }
    
    /// Update search statistics
    fn update_search_stats(&self, cache_hit: bool, search_time: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.search_count += 1;
        
        if cache_hit {
            // Update cache hit rate
            let total_searches = stats.search_count;
            let current_hits = (stats.cache_hit_rate * (total_searches - 1) as f32) as u64;
            let new_hits = current_hits + 1;
            stats.cache_hit_rate = new_hits as f32 / total_searches as f32;
        }
        
        // Update average search time
        if stats.average_search_time == Duration::default() {
            stats.average_search_time = search_time;
        } else {
            stats.average_search_time = (stats.average_search_time + search_time) / 2;
        }
    }
    
    /// Get document by ID
    pub async fn get_document(&self, doc_id: &str) -> Option<VectorDocument> {
        let documents = self.documents.read().await;
        documents.get(doc_id).cloned()
    }
    
    /// Get index statistics
    pub fn get_stats(&self) -> VectorIndexStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Optimize the index
    pub async fn optimize(&self) -> Result<(), String> {
        // Rebuild FAISS index if needed
        if let Some(faiss_index) = &self.faiss_index {
            let mut index = faiss_index.lock().unwrap();
            index.train().map_err(|e| format!("FAISS training failed: {}", e))?;
        }
        
        // Optimize HNSW index
        if let Some(hnsw_index) = &self.hnsw_index {
            let _index = hnsw_index.lock().unwrap();
            // HNSW doesn't need explicit optimization
        }
        
        Ok(())
    }
    
    /// Persist index to disk
    pub async fn persist(&self) -> Result<(), String> {
        if !self.config.enable_persistence {
            return Ok(());
        }
        
        // Serialize index data
        let documents = self.documents.read().await;
        let vectors = self.vectors.read().await;
        let vector_ids = self.vector_ids.read().await;
        
        let index_data = IndexData {
            documents: documents.clone(),
            vectors: vectors.clone(),
            vector_ids: vector_ids.clone(),
            config: self.config.clone(),
        };
        
        let serialized = bincode::serialize(&index_data)
            .map_err(|e| format!("Failed to serialize index: {}", e))?;
        
        // Write to file
        if let Some(file) = &self.index_file {
            let mut writer = BufWriter::new(file);
            writer.write_all(&serialized)
                .map_err(|e| format!("Failed to write index: {}", e))?;
        }
        
        Ok(())
    }
    
    /// Load index from disk
    pub async fn load(&self) -> Result<(), String> {
        if !self.config.enable_persistence {
            return Ok(());
        }
        
        if let Some(file) = &self.index_file {
            let mut reader = BufReader::new(file);
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer)
                .map_err(|e| format!("Failed to read index: {}", e))?;
            
            let index_data: IndexData = bincode::deserialize(&buffer)
                .map_err(|e| format!("Failed to deserialize index: {}", e))?;
            
            // Restore data
            {
                let mut documents = self.documents.write().await;
                *documents = index_data.documents;
            }
            
            {
                let mut vectors = self.vectors.write().await;
                *vectors = index_data.vectors;
            }
            
            {
                let mut vector_ids = self.vector_ids.write().await;
                *vector_ids = index_data.vector_ids;
            }
        }
        
        Ok(())
    }
}

/// Index data for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexData {
    documents: HashMap<String, VectorDocument>,
    vectors: Vec<Vec<f32>>,
    vector_ids: Vec<String>,
    config: VectorIndexConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_vector_index_creation() {
        let config = VectorIndexConfig::default();
        let index = VectorIndex::new(config).unwrap();
        
        let stats = index.get_stats();
        assert_eq!(stats.total_vectors, 0);
    }
    
    #[tokio::test]
    async fn test_vector_index_add_and_search() {
        let config = VectorIndexConfig {
            dimension: 3,
            max_vectors: 1000,
            enable_faiss: false,
            enable_hnsw: false,
            ..Default::default()
        };
        
        let index = VectorIndex::new(config).unwrap();
        
        // Add a test document
        let doc = VectorDocument {
            id: "test1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: HashMap::new(),
            timestamp: 1234567890,
            quality_score: 0.9,
            embedding_model: "test".to_string(),
        };
        
        index.add_document(doc).await.unwrap();
        
        // Search for similar vectors
        let query = vec![1.1, 2.1, 3.1];
        let results = index.search(&query, 1).await.unwrap();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "test1");
    }
}