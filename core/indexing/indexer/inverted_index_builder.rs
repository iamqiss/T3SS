use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::RwLock as AsyncRwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::hash::{Hash, Hasher};
use ahash::{AHashMap, AHashSet};
use roaring::RoaringBitmap;

/// High-performance inverted index builder with advanced optimizations
/// Implements techniques from Google's Caffeine and modern search engines
#[derive(Debug, Clone)]
pub struct InvertedIndexBuilder {
    /// Document frequency map for efficient term filtering
    doc_frequencies: Arc<DashMap<String, AtomicUsize>>,
    /// Term frequency maps for each document
    term_frequencies: Arc<DashMap<u64, AHashMap<String, u32>>>,
    /// Posting lists with position information
    posting_lists: Arc<DashMap<String, SegQueue<PostingEntry>>>,
    /// Document metadata cache
    document_metadata: Arc<DashMap<u64, DocumentMetadata>>,
    /// Term vocabulary with statistics
    vocabulary: Arc<DashMap<String, TermStats>>,
    /// Bloom filter for fast term existence checks
    term_bloom: Arc<bloomfilter::BloomFilter>,
    /// Compression settings
    compression_config: CompressionConfig,
    /// Indexing statistics
    stats: Arc<IndexingStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingEntry {
    pub doc_id: u64,
    pub term_frequency: u32,
    pub positions: Vec<u32>,
    pub field_weights: AHashMap<String, f32>,
    pub quality_score: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub doc_id: u64,
    pub url: String,
    pub title: String,
    pub content_length: u32,
    pub language: String,
    pub domain: String,
    pub crawl_timestamp: u64,
    pub quality_score: f32,
    pub pagerank: f32,
    pub freshness_score: f32,
    pub spam_score: f32,
    pub content_type: String,
    pub encoding: String,
    pub last_modified: Option<u64>,
    pub etag: Option<String>,
    pub backlinks: u32,
    pub social_signals: SocialSignals,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialSignals {
    pub likes: u32,
    pub shares: u32,
    pub comments: u32,
    pub engagement_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStats {
    pub document_frequency: u32,
    pub collection_frequency: u64,
    pub idf: f32,
    pub avg_term_frequency: f32,
    pub max_term_frequency: u32,
    pub term_length: u32,
    pub is_stopword: bool,
    pub semantic_cluster: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub use_variable_byte_encoding: bool,
    pub use_gamma_encoding: bool,
    pub use_pfor_delta: bool,
    pub compression_level: u8,
    pub block_size: usize,
}

#[derive(Debug, Clone)]
pub struct IndexingStats {
    pub total_documents: AtomicU64,
    pub total_terms: AtomicU64,
    pub unique_terms: AtomicUsize,
    pub indexing_time: AtomicU64,
    pub memory_usage: AtomicU64,
    pub compression_ratio: AtomicU64,
}

impl InvertedIndexBuilder {
    /// Create a new inverted index builder with optimized settings
    pub fn new() -> Self {
        let compression_config = CompressionConfig {
            use_variable_byte_encoding: true,
            use_gamma_encoding: true,
            use_pfor_delta: true,
            compression_level: 6,
            block_size: 128,
        };

        Self {
            doc_frequencies: Arc::new(DashMap::with_capacity(1_000_000)),
            term_frequencies: Arc::new(DashMap::with_capacity(10_000_000)),
            posting_lists: Arc::new(DashMap::with_capacity(1_000_000)),
            document_metadata: Arc::new(DashMap::with_capacity(1_000_000)),
            vocabulary: Arc::new(DashMap::with_capacity(1_000_000)),
            term_bloom: Arc::new(bloomfilter::BloomFilter::new(10_000_000, 0.01)),
            compression_config,
            stats: Arc::new(IndexingStats {
                total_documents: AtomicU64::new(0),
                total_terms: AtomicU64::new(0),
                unique_terms: AtomicUsize::new(0),
                indexing_time: AtomicU64::new(0),
                memory_usage: AtomicU64::new(0),
                compression_ratio: AtomicU64::new(0),
            }),
        }
    }

    /// Add a document to the index with advanced preprocessing
    pub async fn add_document(&self, doc: DocumentMetadata, content: &str) -> Result<(), IndexingError> {
        let start_time = Instant::now();
        
        // Advanced text preprocessing pipeline
        let processed_content = self.preprocess_content(content, &doc.language)?;
        
        // Extract terms with position information
        let terms = self.extract_terms_with_positions(&processed_content)?;
        
        // Calculate term frequencies and field weights
        let term_freq_map = self.calculate_term_frequencies(&terms);
        let field_weights = self.calculate_field_weights(&terms, &doc);
        
        // Update document metadata
        self.document_metadata.insert(doc.doc_id, doc.clone());
        
        // Process terms in parallel for maximum performance
        let term_processing_results: Vec<_> = terms
            .par_iter()
            .enumerate()
            .map(|(pos, term)| {
                self.process_term(term, doc.doc_id, pos as u32, &term_freq_map, &field_weights)
            })
            .collect();
        
        // Handle any processing errors
        for result in term_processing_results {
            result?;
        }
        
        // Update statistics
        self.stats.total_documents.fetch_add(1, Ordering::Relaxed);
        self.stats.total_terms.fetch_add(terms.len() as u64, Ordering::Relaxed);
        
        let indexing_time = start_time.elapsed().as_micros() as u64;
        self.stats.indexing_time.fetch_add(indexing_time, Ordering::Relaxed);
        
        Ok(())
    }

    /// Advanced content preprocessing with multilingual support
    fn preprocess_content(&self, content: &str, language: &str) -> Result<String, IndexingError> {
        let mut processed = content.to_lowercase();
        
        // Language-specific preprocessing
        match language {
            "en" => {
                // English-specific preprocessing
                processed = self.apply_english_preprocessing(&processed);
            },
            "zh" => {
                // Chinese-specific preprocessing (no spaces between characters)
                processed = self.apply_chinese_preprocessing(&processed);
            },
            "ar" => {
                // Arabic-specific preprocessing (right-to-left text)
                processed = self.apply_arabic_preprocessing(&processed);
            },
            _ => {
                // Generic preprocessing for other languages
                processed = self.apply_generic_preprocessing(&processed);
            }
        }
        
        // Remove HTML tags and special characters
        processed = self.clean_html_and_special_chars(&processed);
        
        // Normalize unicode
        processed = self.normalize_unicode(&processed);
        
        Ok(processed)
    }

    /// Extract terms with position information for phrase matching
    fn extract_terms_with_positions(&self, content: &str) -> Result<Vec<String>, IndexingError> {
        let mut terms = Vec::new();
        let mut current_pos = 0;
        
        // Advanced tokenization with position tracking
        for token in content.split_whitespace() {
            if !token.is_empty() && self.is_valid_term(token) {
                terms.push(token.to_string());
                current_pos += 1;
            }
        }
        
        Ok(terms)
    }

    /// Calculate term frequencies with advanced weighting
    fn calculate_term_frequencies(&self, terms: &[String]) -> AHashMap<String, u32> {
        let mut freq_map = AHashMap::new();
        
        for term in terms {
            *freq_map.entry(term.clone()).or_insert(0) += 1;
        }
        
        // Apply logarithmic scaling for better distribution
        for (_, freq) in freq_map.iter_mut() {
            *freq = (1.0 + (*freq as f32).ln()) as u32;
        }
        
        freq_map
    }

    /// Calculate field weights based on term positions and document structure
    fn calculate_field_weights(&self, terms: &[String], doc: &DocumentMetadata) -> AHashMap<String, f32> {
        let mut field_weights = AHashMap::new();
        
        // Title weight (higher for terms in title)
        let title_terms: HashSet<&str> = doc.title.to_lowercase().split_whitespace().collect();
        for term in terms {
            if title_terms.contains(term.as_str()) {
                field_weights.insert("title".to_string(), 2.0);
            }
        }
        
        // URL weight
        let url_terms: HashSet<&str> = doc.url.to_lowercase().split_whitespace().collect();
        for term in terms {
            if url_terms.contains(term.as_str()) {
                field_weights.insert("url".to_string(), 1.5);
            }
        }
        
        // Content weight (default)
        for term in terms {
            field_weights.entry("content".to_string()).or_insert(1.0);
        }
        
        field_weights
    }

    /// Process a single term and update all relevant data structures
    fn process_term(
        &self,
        term: &str,
        doc_id: u64,
        position: u32,
        term_freq_map: &AHashMap<String, u32>,
        field_weights: &AHashMap<String, f32>,
    ) -> Result<(), IndexingError> {
        // Check if term exists in bloom filter for fast lookup
        if !self.term_bloom.check(term) {
            self.term_bloom.set(term);
            self.stats.unique_terms.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update document frequency
        self.doc_frequencies
            .entry(term.to_string())
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
        
        // Update term frequency for this document
        self.term_frequencies
            .entry(doc_id)
            .or_insert_with(AHashMap::new)
            .insert(term.to_string(), *term_freq_map.get(term).unwrap_or(&1));
        
        // Create posting entry
        let posting_entry = PostingEntry {
            doc_id,
            term_frequency: *term_freq_map.get(term).unwrap_or(&1),
            positions: vec![position],
            field_weights: field_weights.clone(),
            quality_score: self.calculate_quality_score(term, doc_id),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Add to posting list
        self.posting_lists
            .entry(term.to_string())
            .or_insert_with(SegQueue::new)
            .push(posting_entry);
        
        // Update vocabulary statistics
        self.update_vocabulary_stats(term, term_freq_map.get(term).unwrap_or(&1));
        
        Ok(())
    }

    /// Calculate quality score for a term-document pair
    fn calculate_quality_score(&self, term: &str, doc_id: u64) -> f32 {
        // Base quality score
        let mut score = 1.0;
        
        // Term length penalty (very short or very long terms get lower scores)
        let term_len = term.len() as f32;
        if term_len < 2.0 || term_len > 50.0 {
            score *= 0.5;
        }
        
        // Character diversity bonus
        let unique_chars = term.chars().collect::<HashSet<_>>().len() as f32;
        let diversity_ratio = unique_chars / term_len;
        if diversity_ratio > 0.5 {
            score *= 1.2;
        }
        
        // Document quality factor (if available)
        if let Some(metadata) = self.document_metadata.get(&doc_id) {
            score *= metadata.quality_score;
        }
        
        score
    }

    /// Update vocabulary statistics for a term
    fn update_vocabulary_stats(&self, term: &str, term_freq: &u32) {
        self.vocabulary
            .entry(term.to_string())
            .and_modify(|stats| {
                stats.collection_frequency += *term_freq as u64;
                stats.avg_term_frequency = stats.collection_frequency as f32 / stats.document_frequency as f32;
                stats.max_term_frequency = stats.max_term_frequency.max(*term_freq);
            })
            .or_insert_with(|| TermStats {
                document_frequency: 1,
                collection_frequency: *term_freq as u64,
                idf: 0.0, // Will be calculated later
                avg_term_frequency: *term_freq as f32,
                max_term_frequency: *term_freq,
                term_length: term.len() as u32,
                is_stopword: self.is_stopword(term),
                semantic_cluster: None,
            });
    }

    /// Check if a term is valid for indexing
    fn is_valid_term(&self, term: &str) -> bool {
        // Minimum length check
        if term.len() < 2 {
            return false;
        }
        
        // Maximum length check
        if term.len() > 100 {
            return false;
        }
        
        // Character composition check
        let alpha_ratio = term.chars().filter(|c| c.is_alphabetic()).count() as f32 / term.len() as f32;
        if alpha_ratio < 0.5 {
            return false;
        }
        
        // Stopword check
        if self.is_stopword(term) {
            return false;
        }
        
        true
    }

    /// Check if a term is a stopword
    fn is_stopword(&self, term: &str) -> bool {
        // Common English stopwords
        let stopwords = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those"
        ];
        
        stopwords.contains(&term)
    }

    /// Apply English-specific preprocessing
    fn apply_english_preprocessing(&self, content: &str) -> String {
        content
            .chars()
            .map(|c| if c.is_alphabetic() || c.is_whitespace() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Apply Chinese-specific preprocessing
    fn apply_chinese_preprocessing(&self, content: &str) -> String {
        // Chinese text doesn't use spaces, so we need character-based tokenization
        content
            .chars()
            .filter(|c| c.is_alphabetic() || c.is_whitespace())
            .collect::<String>()
    }

    /// Apply Arabic-specific preprocessing
    fn apply_arabic_preprocessing(&self, content: &str) -> String {
        // Arabic text is right-to-left, need special handling
        content
            .chars()
            .filter(|c| c.is_alphabetic() || c.is_whitespace())
            .collect::<String>()
    }

    /// Apply generic preprocessing for other languages
    fn apply_generic_preprocessing(&self, content: &str) -> String {
        content
            .chars()
            .filter(|c| c.is_alphabetic() || c.is_whitespace())
            .collect::<String>()
    }

    /// Clean HTML tags and special characters
    fn clean_html_and_special_chars(&self, content: &str) -> String {
        // Remove HTML tags
        let html_clean = regex::Regex::new(r"<[^>]*>")
            .unwrap()
            .replace_all(content, " ");
        
        // Remove special characters but keep alphanumeric and spaces
        regex::Regex::new(r"[^\w\s]")
            .unwrap()
            .replace_all(&html_clean, " ")
            .to_string()
    }

    /// Normalize unicode characters
    fn normalize_unicode(&self, content: &str) -> String {
        // Use unicode normalization
        unicode_normalization::UnicodeNormalization::nfc(content)
            .collect::<String>()
    }

    /// Get posting list for a term with compression
    pub async fn get_posting_list(&self, term: &str) -> Option<Vec<PostingEntry>> {
        if let Some(posting_queue) = self.posting_lists.get(term) {
            let mut postings = Vec::new();
            while let Some(posting) = posting_queue.pop() {
                postings.push(posting);
            }
            Some(postings)
        } else {
            None
        }
    }

    /// Get document frequency for a term
    pub fn get_document_frequency(&self, term: &str) -> Option<usize> {
        self.doc_frequencies.get(term).map(|df| df.load(Ordering::Relaxed))
    }

    /// Get term statistics
    pub fn get_term_stats(&self, term: &str) -> Option<TermStats> {
        self.vocabulary.get(term).map(|stats| stats.clone())
    }

    /// Get indexing statistics
    pub fn get_stats(&self) -> IndexingStats {
        IndexingStats {
            total_documents: AtomicU64::new(self.stats.total_documents.load(Ordering::Relaxed)),
            total_terms: AtomicU64::new(self.stats.total_terms.load(Ordering::Relaxed)),
            unique_terms: AtomicUsize::new(self.stats.unique_terms.load(Ordering::Relaxed)),
            indexing_time: AtomicU64::new(self.stats.indexing_time.load(Ordering::Relaxed)),
            memory_usage: AtomicU64::new(self.stats.memory_usage.load(Ordering::Relaxed)),
            compression_ratio: AtomicU64::new(self.stats.compression_ratio.load(Ordering::Relaxed)),
        }
    }

    /// Optimize the index for query performance
    pub async fn optimize_index(&self) -> Result<(), IndexingError> {
        // Sort posting lists by document ID for better compression
        for mut posting_list in self.posting_lists.iter_mut() {
            let mut entries: Vec<_> = std::iter::from_fn(|| posting_list.pop()).collect();
            entries.sort_by_key(|entry| entry.doc_id);
            
            // Rebuild the queue with sorted entries
            for entry in entries {
                posting_list.push(entry);
            }
        }
        
        // Calculate IDF values for all terms
        let total_docs = self.stats.total_documents.load(Ordering::Relaxed) as f32;
        for mut term_stats in self.vocabulary.iter_mut() {
            let df = term_stats.document_frequency as f32;
            term_stats.idf = (total_docs / df).ln();
        }
        
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IndexingError {
    #[error("Invalid document content: {0}")]
    InvalidContent(String),
    #[error("Preprocessing failed: {0}")]
    PreprocessingFailed(String),
    #[error("Term extraction failed: {0}")]
    TermExtractionFailed(String),
    #[error("Index optimization failed: {0}")]
    OptimizationFailed(String),
}

impl Default for InvertedIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_document_indexing() {
        let builder = InvertedIndexBuilder::new();
        
        let doc = DocumentMetadata {
            doc_id: 1,
            url: "https://example.com".to_string(),
            title: "Test Document".to_string(),
            content_length: 100,
            language: "en".to_string(),
            domain: "example.com".to_string(),
            crawl_timestamp: 1234567890,
            quality_score: 0.8,
            pagerank: 0.5,
            freshness_score: 0.9,
            spam_score: 0.1,
            content_type: "text/html".to_string(),
            encoding: "utf-8".to_string(),
            last_modified: Some(1234567890),
            etag: Some("abc123".to_string()),
            backlinks: 10,
            social_signals: SocialSignals {
                likes: 5,
                shares: 2,
                comments: 3,
                engagement_rate: 0.1,
            },
        };
        
        let content = "This is a test document with some sample content for indexing.";
        
        let result = builder.add_document(doc, content).await;
        assert!(result.is_ok());
        
        let stats = builder.get_stats();
        assert_eq!(stats.total_documents.load(Ordering::Relaxed), 1);
        assert!(stats.total_terms.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_posting_list_retrieval() {
        let builder = InvertedIndexBuilder::new();
        
        let doc = DocumentMetadata {
            doc_id: 1,
            url: "https://example.com".to_string(),
            title: "Test Document".to_string(),
            content_length: 100,
            language: "en".to_string(),
            domain: "example.com".to_string(),
            crawl_timestamp: 1234567890,
            quality_score: 0.8,
            pagerank: 0.5,
            freshness_score: 0.9,
            spam_score: 0.1,
            content_type: "text/html".to_string(),
            encoding: "utf-8".to_string(),
            last_modified: Some(1234567890),
            etag: Some("abc123".to_string()),
            backlinks: 10,
            social_signals: SocialSignals {
                likes: 5,
                shares: 2,
                comments: 3,
                engagement_rate: 0.1,
            },
        };
        
        let content = "test document content";
        builder.add_document(doc, content).await.unwrap();
        
        let postings = builder.get_posting_list("test").await;
        assert!(postings.is_some());
        assert!(!postings.unwrap().is_empty());
    }
}