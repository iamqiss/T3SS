// T3SS Project
// File: core/indexing/indexer/inverted_index_builder.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;

/// Represents a document to be indexed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDocument {
    pub id: u64,
    pub url: String,
    pub title: String,
    pub content: String,
    pub timestamp: u64,
    pub domain: String,
    pub content_type: String,
    pub size: u64,
}

/// Represents a posting in the inverted index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Posting {
    pub doc_id: u64,
    pub term_frequency: u32,
    pub positions: Vec<u32>,
    pub field_weights: HashMap<String, f32>,
}

/// Represents the inverted index structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndex {
    pub term_postings: HashMap<String, Vec<Posting>>,
    pub doc_metadata: HashMap<u64, IndexDocument>,
    pub term_stats: HashMap<String, TermStats>,
    pub total_docs: u64,
    pub total_terms: u64,
}

/// Statistics for a term in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStats {
    pub document_frequency: u32,
    pub total_frequency: u64,
    pub avg_term_frequency: f32,
}

/// Configuration for the index builder
#[derive(Debug, Clone)]
pub struct IndexBuilderConfig {
    pub max_documents: usize,
    pub batch_size: usize,
    pub enable_compression: bool,
    pub enable_parallel_processing: bool,
    pub max_term_length: usize,
    pub min_term_frequency: u32,
    pub field_weights: HashMap<String, f32>,
}

impl Default for IndexBuilderConfig {
    fn default() -> Self {
        let mut field_weights = HashMap::new();
        field_weights.insert("title".to_string(), 3.0);
        field_weights.insert("content".to_string(), 1.0);
        field_weights.insert("url".to_string(), 2.0);
        
        Self {
            max_documents: 1_000_000,
            batch_size: 1000,
            enable_compression: true,
            enable_parallel_processing: true,
            max_term_length: 50,
            min_term_frequency: 1,
            field_weights,
        }
    }
}

/// High-performance inverted index builder
pub struct InvertedIndexBuilder {
    config: IndexBuilderConfig,
    index: Arc<AsyncRwLock<InvertedIndex>>,
    term_buffer: Arc<Mutex<HashMap<String, Vec<Posting>>>>,
    doc_counter: Arc<Mutex<u64>>,
    stats: Arc<Mutex<IndexBuilderStats>>,
}

/// Statistics for the index builder
#[derive(Debug, Default)]
pub struct IndexBuilderStats {
    pub documents_processed: u64,
    pub terms_indexed: u64,
    pub build_time: Duration,
    pub memory_usage: u64,
    pub compression_ratio: f32,
}

impl InvertedIndexBuilder {
    /// Create a new index builder
    pub fn new(config: IndexBuilderConfig) -> Self {
        let index = InvertedIndex {
            term_postings: HashMap::new(),
            doc_metadata: HashMap::new(),
            term_stats: HashMap::new(),
            total_docs: 0,
            total_terms: 0,
        };

        Self {
            config,
            index: Arc::new(AsyncRwLock::new(index)),
            term_buffer: Arc::new(Mutex::new(HashMap::new())),
            doc_counter: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(IndexBuilderStats::default())),
        }
    }

    /// Add a single document to the index
    pub async fn add_document(&self, doc: IndexDocument) -> Result<(), String> {
        let start_time = Instant::now();
        
        // Generate document ID
        let doc_id = {
            let mut counter = self.doc_counter.lock().unwrap();
            *counter += 1;
            *counter
        };

        // Tokenize and process the document
        let terms = self.tokenize_document(&doc)?;
        
        // Create postings for each term
        let mut term_postings = HashMap::new();
        for (term, positions) in terms {
            if term.len() > self.config.max_term_length {
                continue;
            }

            let posting = Posting {
                doc_id,
                term_frequency: positions.len() as u32,
                positions,
                field_weights: self.calculate_field_weights(&term, &doc),
            };

            term_postings.entry(term).or_insert_with(Vec::new).push(posting);
        }

        // Add to buffer
        {
            let mut buffer = self.term_buffer.lock().unwrap();
            for (term, postings) in term_postings {
                buffer.entry(term).or_insert_with(Vec::new).extend(postings);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.documents_processed += 1;
            stats.terms_indexed += terms.len() as u64;
        }

        // Flush buffer if it's getting large
        if self.should_flush_buffer().await {
            self.flush_buffer().await?;
        }

        Ok(())
    }

    /// Add multiple documents in batch for better performance
    pub async fn add_documents_batch(&self, docs: Vec<IndexDocument>) -> Result<(), String> {
        if self.config.enable_parallel_processing {
            self.add_documents_parallel(docs).await
        } else {
            self.add_documents_sequential(docs).await
        }
    }

    /// Add documents sequentially
    async fn add_documents_sequential(&self, docs: Vec<IndexDocument>) -> Result<(), String> {
        for doc in docs {
            self.add_document(doc).await?;
        }
        Ok(())
    }

    /// Add documents in parallel for maximum performance
    async fn add_documents_parallel(&self, docs: Vec<IndexDocument>) -> Result<(), String> {
        let chunks: Vec<Vec<IndexDocument>> = docs
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

    /// Tokenize a document into terms with positions
    fn tokenize_document(&self, doc: &IndexDocument) -> Result<HashMap<String, Vec<u32>>, String> {
        let mut terms = HashMap::new();
        let mut position = 0u32;

        // Tokenize title
        for term in self.tokenize_text(&doc.title) {
            terms.entry(term).or_insert_with(Vec::new).push(position);
            position += 1;
        }

        // Tokenize content
        for term in self.tokenize_text(&doc.content) {
            terms.entry(term).or_insert_with(Vec::new).push(position);
            position += 1;
        }

        // Tokenize URL
        for term in self.tokenize_text(&doc.url) {
            terms.entry(term).or_insert_with(Vec::new).push(position);
            position += 1;
        }

        Ok(terms)
    }

    /// Tokenize text into terms
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        text
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .filter(|term| term.len() >= 2 && term.len() <= self.config.max_term_length)
            .map(|term| term.to_string())
            .collect()
    }

    /// Calculate field weights for a term
    fn calculate_field_weights(&self, term: &str, doc: &IndexDocument) -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        
        // Check if term appears in title
        if doc.title.to_lowercase().contains(&term.to_lowercase()) {
            weights.insert("title".to_string(), 
                self.config.field_weights.get("title").unwrap_or(&1.0) * 2.0);
        }
        
        // Check if term appears in content
        if doc.content.to_lowercase().contains(&term.to_lowercase()) {
            weights.insert("content".to_string(), 
                self.config.field_weights.get("content").unwrap_or(&1.0));
        }
        
        // Check if term appears in URL
        if doc.url.to_lowercase().contains(&term.to_lowercase()) {
            weights.insert("url".to_string(), 
                self.config.field_weights.get("url").unwrap_or(&1.0) * 1.5);
        }
        
        weights
    }

    /// Check if buffer should be flushed
    async fn should_flush_buffer(&self) -> bool {
        let buffer = self.term_buffer.lock().unwrap();
        buffer.len() >= self.config.batch_size
    }

    /// Flush the term buffer to the main index
    async fn flush_buffer(&self) -> Result<(), String> {
        let buffer_terms = {
            let mut buffer = self.term_buffer.lock().unwrap();
            std::mem::take(&mut *buffer)
        };

        if buffer_terms.is_empty() {
            return Ok(());
        }

        let mut index = self.index.write().await;
        
        for (term, postings) in buffer_terms {
            // Update term statistics
            let doc_freq = postings.len() as u32;
            let total_freq: u64 = postings.iter().map(|p| p.term_frequency as u64).sum();
            let avg_freq = total_freq as f32 / doc_freq as f32;
            
            let term_stats = TermStats {
                document_frequency: doc_freq,
                total_frequency: total_freq,
                avg_term_frequency: avg_freq,
            };
            
            index.term_stats.insert(term.clone(), term_stats);
            
            // Add postings to index
            index.term_postings.entry(term).or_insert_with(Vec::new).extend(postings);
        }
        
        index.total_terms += buffer_terms.len() as u64;
        
        Ok(())
    }

    /// Finalize the index and apply optimizations
    pub async fn finalize(&self) -> Result<(), String> {
        // Flush any remaining buffer
        self.flush_buffer().await?;
        
        let mut index = self.index.write().await;
        
        // Sort postings by document ID for better compression
        for postings in index.term_postings.values_mut() {
            postings.sort_by_key(|p| p.doc_id);
        }
        
        // Apply compression if enabled
        if self.config.enable_compression {
            self.compress_index(&mut index).await?;
        }
        
        // Update final statistics
        index.total_docs = *self.doc_counter.lock().unwrap();
        
        Ok(())
    }

    /// Apply compression to the index
    async fn compress_index(&self, index: &mut InvertedIndex) -> Result<(), String> {
        // This is a simplified compression - in production, you'd use more sophisticated methods
        let mut compressed_postings = HashMap::new();
        
        for (term, postings) in index.term_postings.drain() {
            if postings.len() >= self.config.min_term_frequency as usize {
                compressed_postings.insert(term, postings);
            }
        }
        
        index.term_postings = compressed_postings;
        Ok(())
    }

    /// Get the built index
    pub async fn get_index(&self) -> InvertedIndex {
        self.index.read().await.clone()
    }

    /// Get builder statistics
    pub fn get_stats(&self) -> IndexBuilderStats {
        self.stats.lock().unwrap().clone()
    }

    /// Search the index for a term
    pub async fn search_term(&self, term: &str) -> Option<Vec<Posting>> {
        let index = self.index.read().await;
        index.term_postings.get(&term.to_lowercase()).cloned()
    }

    /// Get document by ID
    pub async fn get_document(&self, doc_id: u64) -> Option<IndexDocument> {
        let index = self.index.read().await;
        index.doc_metadata.get(&doc_id).cloned()
    }

    /// Get index statistics
    pub async fn get_index_stats(&self) -> HashMap<String, u64> {
        let index = self.index.read().await;
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), index.total_docs);
        stats.insert("total_terms".to_string(), index.total_terms);
        stats.insert("unique_terms".to_string(), index.term_postings.len() as u64);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_builder() {
        let config = IndexBuilderConfig::default();
        let builder = InvertedIndexBuilder::new(config);
        
        let doc = IndexDocument {
            id: 1,
            url: "https://example.com".to_string(),
            title: "Example Title".to_string(),
            content: "This is example content".to_string(),
            timestamp: 1234567890,
            domain: "example.com".to_string(),
            content_type: "text/html".to_string(),
            size: 100,
        };
        
        builder.add_document(doc).await.unwrap();
        builder.finalize().await.unwrap();
        
        let stats = builder.get_index_stats().await;
        assert!(stats.get("total_documents").unwrap() > &0);
    }
}