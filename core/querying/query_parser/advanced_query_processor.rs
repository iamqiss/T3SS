// T3SS Project
// File: core/querying/query_parser/advanced_query_processor.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};
use levenshtein::levenshtein;
use uuid::Uuid;

/// Represents a parsed and processed query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedQuery {
    pub id: String,
    pub original_query: String,
    pub terms: Vec<QueryTerm>,
    pub intent: QueryIntent,
    pub filters: HashMap<String, String>,
    pub boost_fields: HashMap<String, f32>,
    pub expansion_terms: Vec<String>,
    pub corrected_query: Option<String>,
    pub confidence_score: f32,
    pub processing_time: Duration,
}

/// Represents a term in the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTerm {
    pub text: String,
    pub stem: String,
    pub position: usize,
    pub weight: f32,
    pub term_type: TermType,
    pub synonyms: Vec<String>,
    pub related_terms: Vec<String>,
}

/// Types of query terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TermType {
    Keyword,
    Phrase,
    Boolean,
    Wildcard,
    Regex,
    Entity,
    Number,
    Date,
    Location,
}

/// Query intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    Informational,    // Seeking information
    Navigational,     // Looking for specific site/page
    Transactional,    // Want to perform action
    Commercial,       // Shopping/commercial intent
    Local,           // Local business/service
    Question,         // Question-answering
    Multimedia,       // Looking for images/videos
    Unknown,
}

/// Configuration for the query processor
#[derive(Debug, Clone)]
pub struct QueryProcessorConfig {
    pub enable_spell_correction: bool,
    pub enable_query_expansion: bool,
    pub enable_intent_classification: bool,
    pub enable_entity_recognition: bool,
    pub enable_synonym_expansion: bool,
    pub enable_related_terms: bool,
    pub max_expansion_terms: usize,
    pub confidence_threshold: f32,
    pub enable_parallel_processing: bool,
    pub cache_size: usize,
    pub enable_query_logging: bool,
}

impl Default for QueryProcessorConfig {
    fn default() -> Self {
        Self {
            enable_spell_correction: true,
            enable_query_expansion: true,
            enable_intent_classification: true,
            enable_entity_recognition: true,
            enable_synonym_expansion: true,
            enable_related_terms: true,
            max_expansion_terms: 10,
            confidence_threshold: 0.7,
            enable_parallel_processing: true,
            cache_size: 10000,
            enable_query_logging: true,
        }
    }
}

/// Advanced query processor with NLP capabilities
pub struct AdvancedQueryProcessor {
    config: QueryProcessorConfig,
    stemmer: Stemmer,
    spell_checker: Arc<Mutex<SpellChecker>>,
    synonym_db: Arc<RwLock<HashMap<String, Vec<String>>>>,
    entity_db: Arc<RwLock<HashMap<String, EntityInfo>>>,
    intent_classifier: Arc<Mutex<IntentClassifier>>,
    query_cache: Arc<Mutex<HashMap<String, ProcessedQuery>>>,
    stats: Arc<Mutex<QueryProcessorStats>>,
}

/// Spell checker implementation
struct SpellChecker {
    dictionary: HashSet<String>,
    bigram_model: HashMap<String, HashMap<String, f32>>,
    edit_distance_threshold: usize,
}

/// Entity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    pub entity_type: String,
    pub confidence: f32,
    pub aliases: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Intent classifier
struct IntentClassifier {
    patterns: HashMap<QueryIntent, Vec<Regex>>,
    keywords: HashMap<QueryIntent, Vec<String>>,
    trained_model: Option<IntentModel>,
}

/// Trained intent classification model
struct IntentModel {
    features: Vec<String>,
    weights: HashMap<String, HashMap<QueryIntent, f32>>,
    bias: HashMap<QueryIntent, f32>,
}

/// Statistics for query processing
#[derive(Debug, Default)]
pub struct QueryProcessorStats {
    pub total_queries_processed: u64,
    pub spell_corrections_applied: u64,
    pub query_expansions_applied: u64,
    pub intent_classifications: u64,
    pub entity_recognitions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_processing_time: Duration,
    pub queries_per_second: f32,
}

impl AdvancedQueryProcessor {
    /// Create a new advanced query processor
    pub fn new(config: QueryProcessorConfig) -> Self {
        let stemmer = Stemmer::create(Algorithm::English);
        let spell_checker = Arc::new(Mutex::new(SpellChecker::new()));
        let synonym_db = Arc::new(RwLock::new(HashMap::new()));
        let entity_db = Arc::new(RwLock::new(HashMap::new()));
        let intent_classifier = Arc::new(Mutex::new(IntentClassifier::new()));
        let query_cache = Arc::new(Mutex::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(QueryProcessorStats::default()));

        Self {
            config,
            stemmer,
            spell_checker,
            synonym_db,
            entity_db,
            intent_classifier,
            query_cache,
            stats,
        }
    }

    /// Process a raw query into a structured, enhanced query
    pub async fn process_query(&self, raw_query: String) -> Result<ProcessedQuery, String> {
        let start_time = Instant::now();
        
        // Check cache first
        if let Some(cached_query) = self.get_cached_query(&raw_query).await {
            self.update_stats(true, start_time.elapsed());
            return Ok(cached_query);
        }

        let query_id = Uuid::new_v4().to_string();
        
        // Initialize processed query
        let mut processed_query = ProcessedQuery {
            id: query_id,
            original_query: raw_query.clone(),
            terms: Vec::new(),
            intent: QueryIntent::Unknown,
            filters: HashMap::new(),
            boost_fields: HashMap::new(),
            expansion_terms: Vec::new(),
            corrected_query: None,
            confidence_score: 0.0,
            processing_time: Duration::default(),
        };

        // Step 1: Parse and tokenize query
        let tokens = self.tokenize_query(&raw_query);
        
        // Step 2: Spell correction
        if self.config.enable_spell_correction {
            processed_query.corrected_query = self.correct_spelling(&tokens).await;
        }

        // Step 3: Extract terms with analysis
        processed_query.terms = self.extract_and_analyze_terms(&tokens).await;

        // Step 4: Intent classification
        if self.config.enable_intent_classification {
            processed_query.intent = self.classify_intent(&processed_query).await;
        }

        // Step 5: Entity recognition
        if self.config.enable_entity_recognition {
            self.recognize_entities(&mut processed_query).await;
        }

        // Step 6: Query expansion
        if self.config.enable_query_expansion {
            processed_query.expansion_terms = self.expand_query(&processed_query).await;
        }

        // Step 7: Calculate confidence score
        processed_query.confidence_score = self.calculate_confidence(&processed_query);

        // Step 8: Apply filters and boosts
        self.apply_filters_and_boosts(&mut processed_query).await;

        processed_query.processing_time = start_time.elapsed();

        // Cache the result
        self.cache_query(&raw_query, &processed_query).await;

        self.update_stats(false, processed_query.processing_time);
        Ok(processed_query)
    }

    /// Tokenize the query into individual terms
    fn tokenize_query(&self, query: &str) -> Vec<String> {
        // Remove special characters and split on whitespace
        let cleaned_query = query
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '"' || *c == '\'')
            .collect::<String>();

        // Split on whitespace and filter empty strings
        cleaned_query
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Extract and analyze terms from tokens
    async fn extract_and_analyze_terms(&self, tokens: &[String]) -> Vec<QueryTerm> {
        let mut terms = Vec::new();

        for (position, token) in tokens.iter().enumerate() {
            let stem = self.stemmer.stem(token).to_string();
            let term_type = self.classify_term_type(token);
            
            // Get synonyms if enabled
            let synonyms = if self.config.enable_synonym_expansion {
                self.get_synonyms(token).await
            } else {
                Vec::new()
            };

            // Get related terms if enabled
            let related_terms = if self.config.enable_related_terms {
                self.get_related_terms(token).await
            } else {
                Vec::new()
            };

            let weight = self.calculate_term_weight(token, &term_type);

            terms.push(QueryTerm {
                text: token.clone(),
                stem,
                position,
                weight,
                term_type,
                synonyms,
                related_terms,
            });
        }

        terms
    }

    /// Classify the type of a term
    fn classify_term_type(&self, term: &str) -> TermType {
        // Check for numbers
        if term.parse::<f64>().is_ok() {
            return TermType::Number;
        }

        // Check for dates (simplified)
        if self.is_date_pattern(term) {
            return TermType::Date;
        }

        // Check for boolean operators
        if matches!(term, "and" | "or" | "not" | "AND" | "OR" | "NOT") {
            return TermType::Boolean;
        }

        // Check for wildcards
        if term.contains('*') || term.contains('?') {
            return TermType::Wildcard;
        }

        // Check for regex patterns
        if term.starts_with('/') && term.ends_with('/') {
            return TermType::Regex;
        }

        // Check for phrases (quoted strings)
        if term.starts_with('"') && term.ends_with('"') {
            return TermType::Phrase;
        }

        // Default to keyword
        TermType::Keyword
    }

    /// Check if a term matches a date pattern
    fn is_date_pattern(&self, term: &str) -> bool {
        // Simplified date pattern matching
        let date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",           // YYYY-MM-DD
            r"^\d{2}/\d{2}/\d{4}$",           // MM/DD/YYYY
            r"^\d{1,2}/\d{1,2}/\d{2,4}$",    // M/D/YY or MM/DD/YYYY
        ];

        for pattern in &date_patterns {
            if Regex::new(pattern).unwrap().is_match(term) {
                return true;
            }
        }

        false
    }

    /// Calculate weight for a term based on its characteristics
    fn calculate_term_weight(&self, term: &str, term_type: &TermType) -> f32 {
        let mut weight = 1.0;

        // Adjust weight based on term type
        match term_type {
            TermType::Phrase => weight *= 2.0,
            TermType::Entity => weight *= 1.5,
            TermType::Number => weight *= 1.2,
            TermType::Date => weight *= 1.3,
            TermType::Boolean => weight *= 0.5,
            _ => weight *= 1.0,
        }

        // Adjust weight based on term length
        if term.len() > 10 {
            weight *= 1.1; // Longer terms might be more specific
        } else if term.len() < 3 {
            weight *= 0.8; // Very short terms might be less meaningful
        }

        weight
    }

    /// Correct spelling in the query
    async fn correct_spelling(&self, tokens: &[String]) -> Option<String> {
        let mut corrected_tokens = Vec::new();
        let mut has_corrections = false;

        for token in tokens {
            if let Some(corrected) = self.spell_checker.lock().unwrap().correct(token) {
                corrected_tokens.push(corrected);
                has_corrections = true;
            } else {
                corrected_tokens.push(token.clone());
            }
        }

        if has_corrections {
            Some(corrected_tokens.join(" "))
        } else {
            None
        }
    }

    /// Classify query intent
    async fn classify_intent(&self, query: &ProcessedQuery) -> QueryIntent {
        let classifier = self.intent_classifier.lock().unwrap();
        
        // Pattern-based classification
        for (intent, patterns) in &classifier.patterns {
            for pattern in patterns {
                if pattern.is_match(&query.original_query) {
                    return intent.clone();
                }
            }
        }

        // Keyword-based classification
        for (intent, keywords) in &classifier.keywords {
            for keyword in keywords {
                if query.original_query.to_lowercase().contains(keyword) {
                    return intent.clone();
                }
            }
        }

        // Machine learning-based classification (if model is available)
        if let Some(model) = &classifier.trained_model {
            return self.ml_classify_intent(query, model).await;
        }

        QueryIntent::Unknown
    }

    /// Machine learning-based intent classification
    async fn ml_classify_intent(&self, query: &ProcessedQuery, model: &IntentModel) -> QueryIntent {
        let mut scores = HashMap::new();

        // Extract features from query
        let features = self.extract_features(query);

        // Calculate scores for each intent
        for intent in &[
            QueryIntent::Informational,
            QueryIntent::Navigational,
            QueryIntent::Transactional,
            QueryIntent::Commercial,
            QueryIntent::Local,
            QueryIntent::Question,
            QueryIntent::Multimedia,
        ] {
            let mut score = model.bias.get(intent).copied().unwrap_or(0.0);

            for feature in &features {
                if let Some(weights) = model.weights.get(feature) {
                    if let Some(weight) = weights.get(intent) {
                        score += weight;
                    }
                }
            }

            scores.insert(intent.clone(), score);
        }

        // Return intent with highest score
        scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(intent, _)| intent)
            .unwrap_or(QueryIntent::Unknown)
    }

    /// Extract features for ML classification
    fn extract_features(&self, query: &ProcessedQuery) -> Vec<String> {
        let mut features = Vec::new();

        // Query length features
        features.push(format!("length_{}", query.original_query.len()));
        features.push(format!("word_count_{}", query.terms.len()));

        // Term type features
        let mut type_counts = HashMap::new();
        for term in &query.terms {
            let count = type_counts.entry(&term.term_type).or_insert(0);
            *count += 1;
        }

        for (term_type, count) in type_counts {
            features.push(format!("{:?}_{}", term_type, count));
        }

        // Specific word features
        for term in &query.terms {
            features.push(format!("word_{}", term.text));
        }

        features
    }

    /// Recognize entities in the query
    async fn recognize_entities(&self, query: &mut ProcessedQuery) {
        let entity_db = self.entity_db.read().unwrap();
        
        for term in &mut query.terms {
            if let Some(entity_info) = entity_db.get(&term.text) {
                term.term_type = TermType::Entity;
                term.weight *= entity_info.confidence;
            }
        }
    }

    /// Expand query with synonyms and related terms
    async fn expand_query(&self, query: &ProcessedQuery) -> Vec<String> {
        let mut expansion_terms = Vec::new();

        for term in &query.terms {
            // Add synonyms
            expansion_terms.extend(term.synonyms.clone());

            // Add related terms
            expansion_terms.extend(term.related_terms.clone());
        }

        // Remove duplicates and limit
        expansion_terms.sort();
        expansion_terms.dedup();
        expansion_terms.truncate(self.config.max_expansion_terms);

        expansion_terms
    }

    /// Calculate confidence score for the processed query
    fn calculate_confidence(&self, query: &ProcessedQuery) -> f32 {
        let mut confidence = 1.0;

        // Reduce confidence for very short queries
        if query.terms.len() < 2 {
            confidence *= 0.7;
        }

        // Reduce confidence for queries with many corrections
        if query.corrected_query.is_some() {
            confidence *= 0.9;
        }

        // Reduce confidence for unknown intent
        if matches!(query.intent, QueryIntent::Unknown) {
            confidence *= 0.8;
        }

        // Increase confidence for queries with entities
        let entity_count = query.terms.iter()
            .filter(|t| matches!(t.term_type, TermType::Entity))
            .count();
        
        if entity_count > 0 {
            confidence *= 1.0 + (entity_count as f32 * 0.1);
        }

        confidence.min(1.0)
    }

    /// Apply filters and field boosts based on query analysis
    async fn apply_filters_and_boosts(&self, query: &mut ProcessedQuery) {
        // Apply intent-based boosts
        match query.intent {
            QueryIntent::Navigational => {
                query.boost_fields.insert("title".to_string(), 3.0);
                query.boost_fields.insert("url".to_string(), 2.0);
            },
            QueryIntent::Informational => {
                query.boost_fields.insert("content".to_string(), 2.0);
            },
            QueryIntent::Commercial => {
                query.boost_fields.insert("title".to_string(), 2.0);
                query.filters.insert("content_type".to_string(), "product".to_string());
            },
            QueryIntent::Local => {
                query.filters.insert("location".to_string(), "local".to_string());
            },
            QueryIntent::Multimedia => {
                query.filters.insert("content_type".to_string(), "multimedia".to_string());
            },
            _ => {},
        }

        // Apply term-type based boosts
        for term in &query.terms {
            match term.term_type {
                TermType::Entity => {
                    query.boost_fields.insert("title".to_string(), 2.0);
                },
                TermType::Number => {
                    query.boost_fields.insert("content".to_string(), 1.5);
                },
                TermType::Date => {
                    query.filters.insert("date_range".to_string(), term.text.clone());
                },
                _ => {},
            }
        }
    }

    /// Get synonyms for a term
    async fn get_synonyms(&self, term: &str) -> Vec<String> {
        let synonym_db = self.synonym_db.read().unwrap();
        synonym_db.get(term).cloned().unwrap_or_default()
    }

    /// Get related terms for a term
    async fn get_related_terms(&self, term: &str) -> Vec<String> {
        // Simplified related terms - in production, use co-occurrence analysis
        let related_terms_db = HashMap::new(); // Would be populated from training data
        related_terms_db.get(term).cloned().unwrap_or_default()
    }

    /// Cache a processed query
    async fn cache_query(&self, raw_query: &str, processed_query: &ProcessedQuery) {
        let mut cache = self.query_cache.lock().unwrap();
        cache.insert(raw_query.to_string(), processed_query.clone());
    }

    /// Get cached query
    async fn get_cached_query(&self, raw_query: &str) -> Option<ProcessedQuery> {
        let cache = self.query_cache.lock().unwrap();
        cache.get(raw_query).cloned()
    }

    /// Update processor statistics
    fn update_stats(&self, cache_hit: bool, processing_time: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_queries_processed += 1;
        
        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }

        // Update average processing time
        if stats.average_processing_time == Duration::default() {
            stats.average_processing_time = processing_time;
        } else {
            stats.average_processing_time = (stats.average_processing_time + processing_time) / 2;
        }

        // Calculate queries per second
        if processing_time.as_secs() > 0 {
            stats.queries_per_second = 1.0 / processing_time.as_secs_f32();
        }
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> QueryProcessorStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear query cache
    pub fn clear_cache(&self) {
        let mut cache = self.query_cache.lock().unwrap();
        cache.clear();
    }
}

impl SpellChecker {
    fn new() -> Self {
        Self {
            dictionary: HashSet::new(),
            bigram_model: HashMap::new(),
            edit_distance_threshold: 2,
        }
    }

    fn correct(&self, word: &str) -> Option<String> {
        // If word is in dictionary, no correction needed
        if self.dictionary.contains(word) {
            return None;
        }

        // Find candidates with edit distance <= threshold
        let mut candidates = Vec::new();
        for dict_word in &self.dictionary {
            let distance = levenshtein(word, dict_word);
            if distance <= self.edit_distance_threshold {
                candidates.push((dict_word.clone(), distance));
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Sort by edit distance and return best candidate
        candidates.sort_by_key(|(_, distance)| *distance);
        Some(candidates[0].0.clone())
    }
}

impl IntentClassifier {
    fn new() -> Self {
        let mut patterns = HashMap::new();
        let mut keywords = HashMap::new();

        // Question patterns
        patterns.insert(QueryIntent::Question, vec![
            Regex::new(r"^(what|how|when|where|why|who|which|can|could|would|should|is|are|do|does|did)\s").unwrap(),
        ]);

        // Navigational keywords
        keywords.insert(QueryIntent::Navigational, vec![
            "facebook".to_string(),
            "youtube".to_string(),
            "google".to_string(),
            "amazon".to_string(),
            "wikipedia".to_string(),
        ]);

        // Commercial keywords
        keywords.insert(QueryIntent::Commercial, vec![
            "buy".to_string(),
            "purchase".to_string(),
            "price".to_string(),
            "cost".to_string(),
            "shop".to_string(),
            "store".to_string(),
        ]);

        // Local keywords
        keywords.insert(QueryIntent::Local, vec![
            "near me".to_string(),
            "nearby".to_string(),
            "local".to_string(),
            "restaurant".to_string(),
            "hotel".to_string(),
        ]);

        Self {
            patterns,
            keywords,
            trained_model: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_processing() {
        let config = QueryProcessorConfig::default();
        let processor = AdvancedQueryProcessor::new(config);
        
        let result = processor.process_query("machine learning algorithms".to_string()).await.unwrap();
        
        assert_eq!(result.original_query, "machine learning algorithms");
        assert_eq!(result.terms.len(), 2);
        assert_eq!(result.terms[0].text, "machine");
        assert_eq!(result.terms[1].text, "learning");
    }

    #[tokio::test]
    async fn test_intent_classification() {
        let config = QueryProcessorConfig::default();
        let processor = AdvancedQueryProcessor::new(config);
        
        let result = processor.process_query("what is machine learning?".to_string()).await.unwrap();
        assert!(matches!(result.intent, QueryIntent::Question));
    }
}