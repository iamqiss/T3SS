// T3SS Project
// File: core/querying/searcher/query_executor.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use lru::LruCache;

/// Represents a search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub filters: HashMap<String, String>,
    pub limit: usize,
    pub offset: usize,
    pub boost_fields: HashMap<String, f32>,
}

/// Represents a search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: u64,
    pub score: f32,
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub metadata: HashMap<String, String>,
}

/// High-performance query executor
pub struct QueryExecutor {
    index: Arc<AsyncRwLock<InvertedIndex>>,
    cache: Arc<Mutex<LruCache<String, Vec<SearchResult>>>>,
    config: QueryExecutorConfig,
    stats: Arc<Mutex<QueryExecutorStats>>,
}

/// Configuration for the query executor
#[derive(Debug, Clone)]
pub struct QueryExecutorConfig {
    pub cache_size: usize,
    pub max_results: usize,
    pub enable_parallel_execution: bool,
    pub enable_result_caching: bool,
}

impl Default for QueryExecutorConfig {
    fn default() -> Self {
        Self {
            cache_size: 10000,
            max_results: 1000,
            enable_parallel_execution: true,
            enable_result_caching: true,
        }
    }
}

/// Statistics for the query executor
#[derive(Debug, Default)]
pub struct QueryExecutorStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_execution_time: Duration,
    pub queries_per_second: f32,
}

impl QueryExecutor {
    /// Create a new query executor
    pub fn new(
        index: Arc<AsyncRwLock<InvertedIndex>>,
        config: QueryExecutorConfig,
    ) -> Self {
        let cache = Arc::new(Mutex::new(LruCache::new(config.cache_size)));
        
        Self {
            index,
            cache,
            config,
            stats: Arc::new(Mutex::new(QueryExecutorStats::default())),
        }
    }

    /// Execute a search query with maximum performance
    pub async fn execute_query(&self, query: SearchQuery) -> Result<Vec<SearchResult>, String> {
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.enable_result_caching {
            if let Some(cached_results) = self.get_cached_results(&query).await {
                self.update_stats(true, start_time.elapsed());
                return Ok(cached_results);
            }
        }

        // Extract terms from query
        let terms: Vec<String> = query.query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();

        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Execute search
        let results = if self.config.enable_parallel_execution {
            self.execute_parallel_search(&terms).await?
        } else {
            self.execute_sequential_search(&terms).await?
        };

        // Apply pagination
        let paginated_results = self.apply_pagination(results, &query);
        
        // Cache results if enabled
        if self.config.enable_result_caching {
            self.cache_results(&query, &paginated_results).await;
        }
        
        self.update_stats(false, start_time.elapsed());
        Ok(paginated_results)
    }

    /// Execute search in parallel for maximum performance
    async fn execute_parallel_search(&self, terms: &[String]) -> Result<Vec<SearchResult>, String> {
        let index = self.index.read().await;
        
        // Execute term lookups in parallel
        let term_results: Result<Vec<_>, _> = terms
            .par_iter()
            .map(|term| {
                index.term_postings.get(term)
                    .map(|postings| (term.clone(), postings.clone()))
                    .ok_or_else(|| format!("Term not found: {}", term))
            })
            .collect();

        let term_results = term_results?;
        
        // Combine results
        let mut combined_results = Vec::new();
        for (term, postings) in term_results {
            for posting in postings {
                if let Some(doc) = index.doc_metadata.get(&posting.doc_id) {
                    let result = SearchResult {
                        doc_id: posting.doc_id,
                        score: posting.term_frequency as f32,
                        title: doc.title.clone(),
                        url: doc.url.clone(),
                        snippet: self.generate_snippet(&doc.content),
                        metadata: HashMap::new(),
                    };
                    combined_results.push(result);
                }
            }
        }
        
        // Remove duplicates and sort by score
        combined_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        combined_results.dedup_by(|a, b| a.doc_id == b.doc_id);
        
        Ok(combined_results)
    }

    /// Execute search sequentially
    async fn execute_sequential_search(&self, terms: &[String]) -> Result<Vec<SearchResult>, String> {
        let index = self.index.read().await;
        let mut results = Vec::new();
        
        for term in terms {
            if let Some(postings) = index.term_postings.get(term) {
                for posting in postings {
                    if let Some(doc) = index.doc_metadata.get(&posting.doc_id) {
                        let result = SearchResult {
                            doc_id: posting.doc_id,
                            score: posting.term_frequency as f32,
                            title: doc.title.clone(),
                            url: doc.url.clone(),
                            snippet: self.generate_snippet(&doc.content),
                            metadata: HashMap::new(),
                        };
                        results.push(result);
                    }
                }
            }
        }
        
        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.dedup_by(|a, b| a.doc_id == b.doc_id);
        
        Ok(results)
    }

    /// Apply pagination to results
    fn apply_pagination(&self, results: Vec<SearchResult>, query: &SearchQuery) -> Vec<SearchResult> {
        let start = query.offset;
        let end = start + query.limit;
        
        if start >= results.len() {
            return Vec::new();
        }
        
        let end = end.min(results.len());
        results[start..end].to_vec()
    }

    /// Generate snippet for search result
    fn generate_snippet(&self, content: &str) -> String {
        let max_length = 200;
        if content.len() <= max_length {
            content.to_string()
        } else {
            format!("{}...", &content[..max_length])
        }
    }

    /// Get cached results for a query
    async fn get_cached_results(&self, query: &SearchQuery) -> Option<Vec<SearchResult>> {
        let cache_key = self.generate_cache_key(query);
        let mut cache = self.cache.lock().unwrap();
        cache.get(&cache_key).cloned()
    }

    /// Cache results for a query
    async fn cache_results(&self, query: &SearchQuery, results: &[SearchResult]) {
        let cache_key = self.generate_cache_key(query);
        let mut cache = self.cache.lock().unwrap();
        cache.put(cache_key, results.to_vec());
    }

    /// Generate cache key for a query
    fn generate_cache_key(&self, query: &SearchQuery) -> String {
        format!("{}:{}:{}", query.query, query.limit, query.offset)
    }

    /// Update executor statistics
    fn update_stats(&self, cache_hit: bool, execution_time: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_queries += 1;
        
        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
        
        stats.average_execution_time = (stats.average_execution_time + execution_time) / 2;
        
        if execution_time.as_secs() > 0 {
            stats.queries_per_second = 1.0 / execution_time.as_secs_f32();
        }
    }

    /// Get executor statistics
    pub fn get_stats(&self) -> QueryExecutorStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear query cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
}

// Import the InvertedIndex from the indexer module
use crate::indexing::indexer::inverted_index_builder::{InvertedIndex, Posting};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_executor() {
        let config = QueryExecutorConfig::default();
        let index = Arc::new(AsyncRwLock::new(InvertedIndex {
            term_postings: HashMap::new(),
            doc_metadata: HashMap::new(),
            term_stats: HashMap::new(),
            total_docs: 0,
            total_terms: 0,
        }));
        
        let executor = QueryExecutor::new(index, config);
        
        let query = SearchQuery {
            query: "test query".to_string(),
            filters: HashMap::new(),
            limit: 10,
            offset: 0,
            boost_fields: HashMap::new(),
        };
        
        // Test basic functionality
        let results = executor.execute_query(query).await.unwrap();
        assert!(results.is_empty()); // Empty because no documents in test index
    }
}