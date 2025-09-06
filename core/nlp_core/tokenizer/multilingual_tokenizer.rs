// T3SS Project
// File: core/nlp_core/tokenizer/multilingual_tokenizer.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;
use unicode_normalization::UnicodeNormalization;
use whatlang::{detect, Lang, Script};
use rust_stemmers::{Algorithm, Stemmer};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::Path;

/// Language codes supported by the tokenizer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Russian,
    Chinese,
    Japanese,
    Korean,
    Arabic,
    Hindi,
    Dutch,
    Swedish,
    Norwegian,
    Danish,
    Finnish,
    Polish,
    Czech,
    Hungarian,
    Romanian,
    Bulgarian,
    Croatian,
    Serbian,
    Slovak,
    Slovenian,
    Estonian,
    Latvian,
    Lithuanian,
    Greek,
    Turkish,
    Hebrew,
    Thai,
    Vietnamese,
    Indonesian,
    Malay,
    Tagalog,
    Ukrainian,
    Belarusian,
    Unknown,
}

impl Language {
    /// Convert from ISO 639-1 code
    pub fn from_iso_code(code: &str) -> Self {
        match code.to_lowercase().as_str() {
            "en" => Language::English,
            "es" => Language::Spanish,
            "fr" => Language::French,
            "de" => Language::German,
            "it" => Language::Italian,
            "pt" => Language::Portuguese,
            "ru" => Language::Russian,
            "zh" | "zh-cn" | "zh-tw" => Language::Chinese,
            "ja" => Language::Japanese,
            "ko" => Language::Korean,
            "ar" => Language::Arabic,
            "hi" => Language::Hindi,
            "nl" => Language::Dutch,
            "sv" => Language::Swedish,
            "no" => Language::Norwegian,
            "da" => Language::Danish,
            "fi" => Language::Finnish,
            "pl" => Language::Polish,
            "cs" => Language::Czech,
            "hu" => Language::Hungarian,
            "ro" => Language::Romanian,
            "bg" => Language::Bulgarian,
            "hr" => Language::Croatian,
            "sr" => Language::Serbian,
            "sk" => Language::Slovak,
            "sl" => Language::Slovenian,
            "et" => Language::Estonian,
            "lv" => Language::Latvian,
            "lt" => Language::Lithuanian,
            "el" => Language::Greek,
            "tr" => Language::Turkish,
            "he" => Language::Hebrew,
            "th" => Language::Thai,
            "vi" => Language::Vietnamese,
            "id" => Language::Indonesian,
            "ms" => Language::Malay,
            "tl" => Language::Tagalog,
            "uk" => Language::Ukrainian,
            "be" => Language::Belarusian,
            _ => Language::Unknown,
        }
    }

    /// Convert to ISO 639-1 code
    pub fn to_iso_code(self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Hindi => "hi",
            Language::Dutch => "nl",
            Language::Swedish => "sv",
            Language::Norwegian => "no",
            Language::Danish => "da",
            Language::Finnish => "fi",
            Language::Polish => "pl",
            Language::Czech => "cs",
            Language::Hungarian => "hu",
            Language::Romanian => "ro",
            Language::Bulgarian => "bg",
            Language::Croatian => "hr",
            Language::Serbian => "sr",
            Language::Slovak => "sk",
            Language::Slovenian => "sl",
            Language::Estonian => "et",
            Language::Latvian => "lv",
            Language::Lithuanian => "lt",
            Language::Greek => "el",
            Language::Turkish => "tr",
            Language::Hebrew => "he",
            Language::Thai => "th",
            Language::Vietnamese => "vi",
            Language::Indonesian => "id",
            Language::Malay => "ms",
            Language::Tagalog => "tl",
            Language::Ukrainian => "uk",
            Language::Belarusian => "be",
            Language::Unknown => "unknown",
        }
    }
}

/// Token types for different categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    Word,
    Number,
    Punctuation,
    Whitespace,
    Symbol,
    Emoji,
    URL,
    Email,
    Hashtag,
    Mention,
    Phone,
    Date,
    Time,
    Currency,
    Unknown,
}

/// A token with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub token_type: TokenType,
    pub start: usize,
    pub end: usize,
    pub language: Language,
    pub normalized: String,
    pub stem: Option<String>,
    pub lemma: Option<String>,
    pub pos_tag: Option<String>,
    pub is_stop_word: bool,
    pub frequency: u32,
    pub confidence: f32,
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub enable_language_detection: bool,
    pub enable_normalization: bool,
    pub enable_stemming: bool,
    pub enable_lemmatization: bool,
    pub enable_pos_tagging: bool,
    pub enable_stop_word_filtering: bool,
    pub enable_emoji_handling: bool,
    pub enable_url_detection: bool,
    pub enable_email_detection: bool,
    pub enable_hashtag_detection: bool,
    pub enable_mention_detection: bool,
    pub enable_phone_detection: bool,
    pub enable_date_detection: bool,
    pub enable_time_detection: bool,
    pub enable_currency_detection: bool,
    pub case_sensitive: bool,
    pub preserve_whitespace: bool,
    pub min_token_length: usize,
    pub max_token_length: usize,
    pub supported_languages: Vec<Language>,
    pub custom_patterns: HashMap<String, Regex>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            enable_language_detection: true,
            enable_normalization: true,
            enable_stemming: true,
            enable_lemmatization: false,
            enable_pos_tagging: false,
            enable_stop_word_filtering: true,
            enable_emoji_handling: true,
            enable_url_detection: true,
            enable_email_detection: true,
            enable_hashtag_detection: true,
            enable_mention_detection: true,
            enable_phone_detection: true,
            enable_date_detection: true,
            enable_time_detection: true,
            enable_currency_detection: true,
            case_sensitive: false,
            preserve_whitespace: false,
            min_token_length: 1,
            max_token_length: 100,
            supported_languages: vec![
                Language::English,
                Language::Spanish,
                Language::French,
                Language::German,
                Language::Italian,
                Language::Portuguese,
                Language::Russian,
                Language::Chinese,
                Language::Japanese,
                Language::Korean,
                Language::Arabic,
            ],
            custom_patterns: HashMap::new(),
        }
    }
}

/// Multilingual tokenizer with advanced features
pub struct MultilingualTokenizer {
    config: TokenizerConfig,
    language_detector: LanguageDetector,
    normalizers: HashMap<Language, TextNormalizer>,
    stemmers: HashMap<Language, Stemmer>,
    stop_words: HashMap<Language, HashSet<String>>,
    patterns: PatternMatcher,
    statistics: Arc<Mutex<TokenizerStats>>,
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    pub language: Language,
    pub confidence: f32,
    pub script: Script,
    pub is_reliable: bool,
}

/// Text normalization result
#[derive(Debug, Clone)]
pub struct NormalizationResult {
    pub normalized_text: String,
    pub original_text: String,
    pub changes: Vec<NormalizationChange>,
}

/// Normalization change
#[derive(Debug, Clone)]
pub struct NormalizationChange {
    pub original: String,
    pub normalized: String,
    pub change_type: ChangeType,
    pub position: usize,
}

/// Type of normalization change
#[derive(Debug, Clone)]
pub enum ChangeType {
    CaseChange,
    DiacriticRemoval,
    PunctuationNormalization,
    WhitespaceNormalization,
    UnicodeNormalization,
    Custom,
}

/// Pattern matcher for special tokens
struct PatternMatcher {
    url_pattern: Regex,
    email_pattern: Regex,
    hashtag_pattern: Regex,
    mention_pattern: Regex,
    phone_pattern: Regex,
    date_pattern: Regex,
    time_pattern: Regex,
    currency_pattern: Regex,
    emoji_pattern: Regex,
}

/// Language detector
struct LanguageDetector {
    supported_languages: Vec<Language>,
    fallback_language: Language,
}

/// Text normalizer for specific language
struct TextNormalizer {
    language: Language,
    case_normalization: bool,
    diacritic_removal: bool,
    punctuation_normalization: bool,
    whitespace_normalization: bool,
    unicode_normalization: bool,
}

/// Tokenizer statistics
#[derive(Debug, Default)]
pub struct TokenizerStats {
    pub total_texts_processed: u64,
    pub total_tokens_generated: u64,
    pub language_detection_accuracy: f32,
    pub average_processing_time: Duration,
    pub token_frequency: HashMap<String, u32>,
    pub language_distribution: HashMap<Language, u64>,
    pub token_type_distribution: HashMap<TokenType, u64>,
}

impl MultilingualTokenizer {
    /// Create a new multilingual tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        let language_detector = LanguageDetector::new(config.supported_languages.clone());
        
        let mut normalizers = HashMap::new();
        for &lang in &config.supported_languages {
            normalizers.insert(lang, TextNormalizer::new(lang));
        }
        
        let mut stemmers = HashMap::new();
        for &lang in &config.supported_languages {
            if let Some(stemmer) = Self::create_stemmer(lang) {
                stemmers.insert(lang, stemmer);
            }
        }
        
        let stop_words = Self::load_stop_words(&config.supported_languages);
        let patterns = PatternMatcher::new();
        let statistics = Arc::new(Mutex::new(TokenizerStats::default()));
        
        Self {
            config,
            language_detector,
            normalizers,
            stemmers,
            stop_words,
            patterns,
            statistics,
        }
    }
    
    /// Tokenize text with language detection
    pub fn tokenize(&self, text: &str) -> Result<Vec<Token>, String> {
        let start_time = Instant::now();
        
        // Detect language
        let language_result = if self.config.enable_language_detection {
            self.language_detector.detect(text)
        } else {
            LanguageDetectionResult {
                language: Language::English,
                confidence: 1.0,
                script: Script::Latin,
                is_reliable: true,
            }
        };
        
        // Normalize text
        let normalized_text = if self.config.enable_normalization {
            self.normalize_text(text, language_result.language)?
        } else {
            text.to_string()
        };
        
        // Tokenize
        let tokens = self.tokenize_text(&normalized_text, language_result.language)?;
        
        // Update statistics
        self.update_statistics(&tokens, language_result.language, start_time.elapsed());
        
        Ok(tokens)
    }
    
    /// Tokenize text in batch
    pub fn tokenize_batch(&self, texts: &[String]) -> Result<Vec<Vec<Token>>, String> {
        let results: Result<Vec<_>, _> = texts
            .par_iter()
            .map(|text| self.tokenize(text))
            .collect();
        results
    }
    
    /// Normalize text for a specific language
    fn normalize_text(&self, text: &str, language: Language) -> Result<String, String> {
        if let Some(normalizer) = self.normalizers.get(&language) {
            normalizer.normalize(text)
        } else {
            Ok(text.to_string())
        }
    }
    
    /// Tokenize text into tokens
    fn tokenize_text(&self, text: &str, language: Language) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let mut position = 0;
        
        // Split by whitespace and punctuation
        let segments = self.split_text(text);
        
        for segment in segments {
            if segment.trim().is_empty() {
                if self.config.preserve_whitespace {
                    tokens.push(Token {
                        text: segment.to_string(),
                        token_type: TokenType::Whitespace,
                        start: position,
                        end: position + segment.len(),
                        language,
                        normalized: segment.to_string(),
                        stem: None,
                        lemma: None,
                        pos_tag: None,
                        is_stop_word: false,
                        frequency: 0,
                        confidence: 1.0,
                    });
                }
                position += segment.len();
                continue;
            }
            
            // Check for special patterns
            if let Some(special_token) = self.detect_special_token(segment, position, language) {
                tokens.push(special_token);
                position += segment.len();
                continue;
            }
            
            // Regular word tokenization
            let word_tokens = self.tokenize_word(segment, position, language)?;
            tokens.extend(word_tokens);
            position += segment.len();
        }
        
        // Post-process tokens
        self.post_process_tokens(&mut tokens, language)?;
        
        Ok(tokens)
    }
    
    /// Split text into segments
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current_segment = String::new();
        
        for grapheme in text.graphemes(true) {
            if grapheme.chars().all(|c| c.is_whitespace()) {
                if !current_segment.is_empty() {
                    segments.push(current_segment);
                    current_segment = String::new();
                }
                segments.push(grapheme.to_string());
            } else if grapheme.chars().any(|c| c.is_ascii_punctuation()) {
                if !current_segment.is_empty() {
                    segments.push(current_segment);
                    current_segment = String::new();
                }
                segments.push(grapheme.to_string());
            } else {
                current_segment.push_str(grapheme);
            }
        }
        
        if !current_segment.is_empty() {
            segments.push(current_segment);
        }
        
        segments
    }
    
    /// Detect special tokens (URLs, emails, etc.)
    fn detect_special_token(&self, text: &str, position: usize, language: Language) -> Option<Token> {
        if self.config.enable_url_detection && self.patterns.url_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::URL,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_email_detection && self.patterns.email_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Email,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_hashtag_detection && self.patterns.hashtag_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Hashtag,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_mention_detection && self.patterns.mention_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Mention,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_phone_detection && self.patterns.phone_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Phone,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_date_detection && self.patterns.date_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Date,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_time_detection && self.patterns.time_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Time,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_currency_detection && self.patterns.currency_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Currency,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        if self.config.enable_emoji_handling && self.patterns.emoji_pattern.is_match(text) {
            return Some(Token {
                text: text.to_string(),
                token_type: TokenType::Emoji,
                start: position,
                end: position + text.len(),
                language,
                normalized: text.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
        }
        
        None
    }
    
    /// Tokenize a word
    fn tokenize_word(&self, word: &str, position: usize, language: Language) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        
        // Check if it's a number
        if word.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',' || c == '-') {
            tokens.push(Token {
                text: word.to_string(),
                token_type: TokenType::Number,
                start: position,
                end: position + word.len(),
                language,
                normalized: word.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
            return Ok(tokens);
        }
        
        // Check if it's punctuation
        if word.chars().all(|c| c.is_ascii_punctuation()) {
            tokens.push(Token {
                text: word.to_string(),
                token_type: TokenType::Punctuation,
                start: position,
                end: position + word.len(),
                language,
                normalized: word.to_string(),
                stem: None,
                lemma: None,
                pos_tag: None,
                is_stop_word: false,
                frequency: 0,
                confidence: 1.0,
            });
            return Ok(tokens);
        }
        
        // Regular word
        let normalized = if self.config.case_sensitive {
            word.to_string()
        } else {
            word.to_lowercase()
        };
        
        let stem = if self.config.enable_stemming {
            self.stem_word(&normalized, language)
        } else {
            None
        };
        
        let is_stop_word = if self.config.enable_stop_word_filtering {
            self.is_stop_word(&normalized, language)
        } else {
            false
        };
        
        tokens.push(Token {
            text: word.to_string(),
            token_type: TokenType::Word,
            start: position,
            end: position + word.len(),
            language,
            normalized,
            stem,
            lemma: None, // TODO: Implement lemmatization
            pos_tag: None, // TODO: Implement POS tagging
            is_stop_word,
            frequency: 0,
            confidence: 1.0,
        });
        
        Ok(tokens)
    }
    
    /// Stem a word
    fn stem_word(&self, word: &str, language: Language) -> Option<String> {
        if let Some(stemmer) = self.stemmers.get(&language) {
            Some(stemmer.stem(word).to_string())
        } else {
            None
        }
    }
    
    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str, language: Language) -> bool {
        if let Some(stop_words) = self.stop_words.get(&language) {
            stop_words.contains(word)
        } else {
            false
        }
    }
    
    /// Post-process tokens
    fn post_process_tokens(&self, tokens: &mut Vec<Token>, language: Language) -> Result<(), String> {
        // Filter by length
        tokens.retain(|token| {
            token.text.len() >= self.config.min_token_length && 
            token.text.len() <= self.config.max_token_length
        });
        
        // Update frequencies
        let mut freq_map = HashMap::new();
        for token in tokens.iter() {
            *freq_map.entry(token.normalized.clone()).or_insert(0) += 1;
        }
        
        for token in tokens.iter_mut() {
            token.frequency = freq_map.get(&token.normalized).copied().unwrap_or(0);
        }
        
        Ok(())
    }
    
    /// Update statistics
    fn update_statistics(&self, tokens: &[Token], language: Language, processing_time: Duration) {
        let mut stats = self.statistics.lock().unwrap();
        stats.total_texts_processed += 1;
        stats.total_tokens_generated += tokens.len() as u64;
        
        // Update language distribution
        *stats.language_distribution.entry(language).or_insert(0) += 1;
        
        // Update token type distribution
        for token in tokens {
            *stats.token_type_distribution.entry(token.token_type.clone()).or_insert(0) += 1;
            *stats.token_frequency.entry(token.normalized.clone()).or_insert(0) += 1;
        }
        
        // Update average processing time
        if stats.average_processing_time == Duration::default() {
            stats.average_processing_time = processing_time;
        } else {
            stats.average_processing_time = (stats.average_processing_time + processing_time) / 2;
        }
    }
    
    /// Create stemmer for language
    fn create_stemmer(language: Language) -> Option<Stemmer> {
        let algorithm = match language {
            Language::English => Algorithm::English,
            Language::Spanish => Algorithm::Spanish,
            Language::French => Algorithm::French,
            Language::German => Algorithm::German,
            Language::Italian => Algorithm::Italian,
            Language::Portuguese => Algorithm::Portuguese,
            Language::Russian => Algorithm::Russian,
            Language::Dutch => Algorithm::Dutch,
            Language::Swedish => Algorithm::Swedish,
            Language::Norwegian => Algorithm::Norwegian,
            Language::Danish => Algorithm::Danish,
            Language::Finnish => Algorithm::Finnish,
            Language::Polish => Algorithm::Polish,
            Language::Czech => Algorithm::Czech,
            Language::Hungarian => Algorithm::Hungarian,
            Language::Romanian => Algorithm::Romanian,
            Language::Bulgarian => Algorithm::Bulgarian,
            Language::Croatian => Algorithm::Croatian,
            Language::Serbian => Algorithm::Serbian,
            Language::Slovak => Algorithm::Slovak,
            Language::Slovenian => Algorithm::Slovenian,
            Language::Estonian => Algorithm::Estonian,
            Language::Latvian => Algorithm::Latvian,
            Language::Lithuanian => Algorithm::Lithuanian,
            Language::Greek => Algorithm::Greek,
            Language::Turkish => Algorithm::Turkish,
            Language::Hebrew => Algorithm::Hebrew,
            Language::Thai => Algorithm::Thai,
            Language::Vietnamese => Algorithm::Vietnamese,
            Language::Indonesian => Algorithm::Indonesian,
            Language::Malay => Algorithm::Malay,
            Language::Tagalog => Algorithm::Tagalog,
            Language::Ukrainian => Algorithm::Ukrainian,
            Language::Belarusian => Algorithm::Belarusian,
            _ => return None,
        };
        
        Some(Stemmer::create(algorithm))
    }
    
    /// Load stop words for languages
    fn load_stop_words(languages: &[Language]) -> HashMap<Language, HashSet<String>> {
        let mut stop_words = HashMap::new();
        
        for &language in languages {
            let mut words = HashSet::new();
            
            // Load from built-in stop words
            let built_in_words = Self::get_built_in_stop_words(language);
            for word in built_in_words {
                words.insert(word);
            }
            
            // Try to load from file
            if let Ok(file_words) = Self::load_stop_words_from_file(language) {
                for word in file_words {
                    words.insert(word);
                }
            }
            
            stop_words.insert(language, words);
        }
        
        stop_words
    }
    
    /// Get built-in stop words for language
    fn get_built_in_stop_words(language: Language) -> Vec<String> {
        match language {
            Language::English => vec![
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
                "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those",
                "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"
            ],
            Language::Spanish => vec![
                "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "y", "a", "que",
                "es", "son", "era", "eran", "fue", "fueron", "ser", "estar", "tener", "haber", "hacer",
                "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "me", "te", "lo", "la", "nos", "os", "los", "las"
            ],
            Language::French => vec![
                "le", "la", "les", "un", "une", "des", "de", "du", "en", "et", "à", "que", "qui", "ce", "cette",
                "est", "sont", "était", "étaient", "être", "avoir", "faire", "aller", "venir",
                "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "me", "te", "le", "la", "nous", "vous", "les"
            ],
            Language::German => vec![
                "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "eines", "und", "oder", "aber",
                "ist", "sind", "war", "waren", "sein", "haben", "werden", "können", "müssen", "sollen",
                "ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "mich", "dich", "ihn", "sie", "uns", "euch", "sie"
            ],
            _ => vec![],
        }.into_iter().map(|s| s.to_string()).collect()
    }
    
    /// Load stop words from file
    fn load_stop_words_from_file(language: Language) -> Result<Vec<String>, String> {
        let filename = format!("stop_words_{}.txt", language.to_iso_code());
        let path = Path::new("data/stop_words").join(filename);
        
        if !path.exists() {
            return Ok(vec![]);
        }
        
        let file = File::open(&path).map_err(|e| format!("Failed to open stop words file: {}", e))?;
        let reader = BufReader::new(file);
        
        let mut words = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let word = line.trim().to_string();
            if !word.is_empty() {
                words.push(word);
            }
        }
        
        Ok(words)
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> TokenizerStats {
        self.statistics.lock().unwrap().clone()
    }
    
    /// Clear statistics
    pub fn clear_statistics(&self) {
        let mut stats = self.statistics.lock().unwrap();
        *stats = TokenizerStats::default();
    }
}

impl PatternMatcher {
    fn new() -> Self {
        Self {
            url_pattern: Regex::new(r"https?://[^\s]+").unwrap(),
            email_pattern: Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
            hashtag_pattern: Regex::new(r"#[a-zA-Z0-9_]+").unwrap(),
            mention_pattern: Regex::new(r"@[a-zA-Z0-9_]+").unwrap(),
            phone_pattern: Regex::new(r"\+?[\d\s\-\(\)]{10,}").unwrap(),
            date_pattern: Regex::new(r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}").unwrap(),
            time_pattern: Regex::new(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?", RegexOptions::new().case_insensitive(true)).unwrap(),
            currency_pattern: Regex::new(r"[$€£¥₹₽]\s?\d+(?:\.\d{2})?").unwrap(),
            emoji_pattern: Regex::new(r"[\p{Emoji}\p{Emoji_Modifier}\p{Emoji_Modifier_Base}\p{Emoji_Presentation}\p{Emoji_Symbol}\p{Emoji_Text_Default}]").unwrap(),
        }
    }
}

impl LanguageDetector {
    fn new(supported_languages: Vec<Language>) -> Self {
        Self {
            supported_languages,
            fallback_language: Language::English,
        }
    }
    
    fn detect(&self, text: &str) -> LanguageDetectionResult {
        // Simple language detection based on character patterns
        let script = self.detect_script(text);
        let language = self.detect_language_by_script(text, script);
        let confidence = self.calculate_confidence(text, language);
        
        LanguageDetectionResult {
            language,
            confidence,
            script,
            is_reliable: confidence > 0.8,
        }
    }
    
    fn detect_script(&self, text: &str) -> Script {
        let mut latin_count = 0;
        let mut cyrillic_count = 0;
        let mut arabic_count = 0;
        let mut chinese_count = 0;
        let mut japanese_count = 0;
        let mut korean_count = 0;
        let mut devanagari_count = 0;
        let mut thai_count = 0;
        let mut hebrew_count = 0;
        let mut greek_count = 0;
        
        for ch in text.chars() {
            match ch {
                'a'..='z' | 'A'..='Z' => latin_count += 1,
                'а'..='я' | 'А'..='Я' | 'ё' | 'Ё' => cyrillic_count += 1,
                'ا'..='ي' | 'أ'..='ي' => arabic_count += 1,
                '一'..='龯' => chinese_count += 1,
                'ひ'..='ゟ' | 'カ'..='ヿ' | '一'..='龯' => japanese_count += 1,
                'ㄱ'..='ㅿ' | '가'..='힣' => korean_count += 1,
                'अ'..='ह' | '०'..='९' => devanagari_count += 1,
                'ก'..='๙' => thai_count += 1,
                'א'..='ת' => hebrew_count += 1,
                'α'..='ω' | 'Α'..='Ω' => greek_count += 1,
                _ => {}
            }
        }
        
        let total = latin_count + cyrillic_count + arabic_count + chinese_count + 
                   japanese_count + korean_count + devanagari_count + thai_count + 
                   hebrew_count + greek_count;
        
        if total == 0 {
            return Script::Latin;
        }
        
        let mut max_count = latin_count;
        let mut max_script = Script::Latin;
        
        if cyrillic_count > max_count { max_count = cyrillic_count; max_script = Script::Cyrillic; }
        if arabic_count > max_count { max_count = arabic_count; max_script = Script::Arabic; }
        if chinese_count > max_count { max_count = chinese_count; max_script = Script::Chinese; }
        if japanese_count > max_count { max_count = japanese_count; max_script = Script::Japanese; }
        if korean_count > max_count { max_count = korean_count; max_script = Script::Korean; }
        if devanagari_count > max_count { max_count = devanagari_count; max_script = Script::Devanagari; }
        if thai_count > max_count { max_count = thai_count; max_script = Script::Thai; }
        if hebrew_count > max_count { max_count = hebrew_count; max_script = Script::Hebrew; }
        if greek_count > max_count { max_count = greek_count; max_script = Script::Greek; }
        
        max_script
    }
    
    fn detect_language_by_script(&self, text: &str, script: Script) -> Language {
        match script {
            Script::Latin => {
                // Simple heuristics for Latin-based languages
                if text.contains("the") || text.contains("and") || text.contains("of") {
                    Language::English
                } else if text.contains("el") || text.contains("la") || text.contains("de") {
                    Language::Spanish
                } else if text.contains("le") || text.contains("la") || text.contains("de") {
                    Language::French
                } else if text.contains("der") || text.contains("die") || text.contains("das") {
                    Language::German
                } else if text.contains("il") || text.contains("la") || text.contains("di") {
                    Language::Italian
                } else if text.contains("o") || text.contains("a") || text.contains("de") {
                    Language::Portuguese
                } else {
                    Language::English
                }
            },
            Script::Cyrillic => {
                if text.contains("и") || text.contains("в") || text.contains("на") {
                    Language::Russian
                } else if text.contains("і") || text.contains("в") || text.contains("на") {
                    Language::Ukrainian
                } else if text.contains("і") || text.contains("ў") || text.contains("на") {
                    Language::Belarusian
                } else {
                    Language::Russian
                }
            },
            Script::Arabic => Language::Arabic,
            Script::Chinese => Language::Chinese,
            Script::Japanese => Language::Japanese,
            Script::Korean => Language::Korean,
            Script::Devanagari => Language::Hindi,
            Script::Thai => Language::Thai,
            Script::Hebrew => Language::Hebrew,
            Script::Greek => Language::Greek,
            _ => Language::English,
        }
    }
    
    fn calculate_confidence(&self, text: &str, language: Language) -> f32 {
        // Simple confidence calculation based on character distribution
        let total_chars = text.chars().count() as f32;
        if total_chars == 0.0 {
            return 0.0;
        }
        
        let language_chars = match language {
            Language::English => text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32,
            Language::Spanish => text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32,
            Language::French => text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32,
            Language::German => text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32,
            Language::Italian => text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32,
            Language::Portuguese => text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f32,
            Language::Russian => text.chars().filter(|c| matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё')).count() as f32,
            Language::Arabic => text.chars().filter(|c| matches!(c, 'ا'..='ي' | 'أ'..='ي')).count() as f32,
            Language::Chinese => text.chars().filter(|c| matches!(c, '一'..='龯')).count() as f32,
            Language::Japanese => text.chars().filter(|c| matches!(c, 'ひ'..='ゟ' | 'カ'..='ヿ' | '一'..='龯')).count() as f32,
            Language::Korean => text.chars().filter(|c| matches!(c, 'ㄱ'..='ㅿ' | '가'..='힣')).count() as f32,
            Language::Hindi => text.chars().filter(|c| matches!(c, 'अ'..='ह' | '०'..='९')).count() as f32,
            Language::Thai => text.chars().filter(|c| matches!(c, 'ก'..='๙')).count() as f32,
            Language::Hebrew => text.chars().filter(|c| matches!(c, 'א'..='ת')).count() as f32,
            Language::Greek => text.chars().filter(|c| matches!(c, 'α'..='ω' | 'Α'..='Ω')).count() as f32,
            _ => text.chars().filter(|c| c.is_alphabetic()).count() as f32,
        };
        
        (language_chars / total_chars).min(1.0)
    }
}

impl TextNormalizer {
    fn new(language: Language) -> Self {
        Self {
            language,
            case_normalization: true,
            diacritic_removal: false,
            punctuation_normalization: true,
            whitespace_normalization: true,
            unicode_normalization: true,
        }
    }
    
    fn normalize(&self, text: &str) -> Result<String, String> {
        let mut normalized = text.to_string();
        
        if self.unicode_normalization {
            normalized = unicode_normalization::nfc(&normalized);
        }
        
        if self.whitespace_normalization {
            normalized = self.normalize_whitespace(&normalized);
        }
        
        if self.punctuation_normalization {
            normalized = self.normalize_punctuation(&normalized);
        }
        
        if self.case_normalization {
            normalized = self.normalize_case(&normalized);
        }
        
        if self.diacritic_removal {
            normalized = self.remove_diacritics(&normalized);
        }
        
        Ok(normalized)
    }
    
    fn normalize_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<&str>>().join(" ")
    }
    
    fn normalize_punctuation(&self, text: &str) -> String {
        text.chars()
            .map(|c| match c {
                '"' | '"' | '"' => '"',
                ''' | ''' | ''' => '\'',
                '–' | '—' => '-',
                '…' => '.',
                _ => c,
            })
            .collect()
    }
    
    fn normalize_case(&self, text: &str) -> String {
        text.to_lowercase()
    }
    
    fn remove_diacritics(&self, text: &str) -> String {
        text.chars()
            .map(|c| c.to_ascii_lowercase())
            .collect()
    }
}

impl Language {
    fn to_iso_code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Dutch => "nl",
            Language::Swedish => "sv",
            Language::Norwegian => "no",
            Language::Danish => "da",
            Language::Finnish => "fi",
            Language::Polish => "pl",
            Language::Czech => "cs",
            Language::Hungarian => "hu",
            Language::Romanian => "ro",
            Language::Bulgarian => "bg",
            Language::Croatian => "hr",
            Language::Serbian => "sr",
            Language::Slovak => "sk",
            Language::Slovenian => "sl",
            Language::Estonian => "et",
            Language::Latvian => "lv",
            Language::Lithuanian => "lt",
            Language::Greek => "el",
            Language::Turkish => "tr",
            Language::Hebrew => "he",
            Language::Arabic => "ar",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Hindi => "hi",
            Language::Thai => "th",
            Language::Vietnamese => "vi",
            Language::Indonesian => "id",
            Language::Malay => "ms",
            Language::Tagalog => "tl",
            Language::Ukrainian => "uk",
            Language::Belarusian => "be",
        }
    }
}

impl TokenType {
    fn to_string(&self) -> String {
        match self {
            TokenType::Word => "word".to_string(),
            TokenType::Number => "number".to_string(),
            TokenType::Punctuation => "punctuation".to_string(),
            TokenType::Whitespace => "whitespace".to_string(),
            TokenType::URL => "url".to_string(),
            TokenType::Email => "email".to_string(),
            TokenType::Hashtag => "hashtag".to_string(),
            TokenType::Mention => "mention".to_string(),
            TokenType::Phone => "phone".to_string(),
            TokenType::Date => "date".to_string(),
            TokenType::Time => "time".to_string(),
            TokenType::Currency => "currency".to_string(),
            TokenType::Emoji => "emoji".to_string(),
        }
    }
}

impl Clone for TokenType {
    fn clone(&self) -> Self {
        match self {
            TokenType::Word => TokenType::Word,
            TokenType::Number => TokenType::Number,
            TokenType::Punctuation => TokenType::Punctuation,
            TokenType::Whitespace => TokenType::Whitespace,
            TokenType::URL => TokenType::URL,
            TokenType::Email => TokenType::Email,
            TokenType::Hashtag => TokenType::Hashtag,
            TokenType::Mention => TokenType::Mention,
            TokenType::Phone => TokenType::Phone,
            TokenType::Date => TokenType::Date,
            TokenType::Time => TokenType::Time,
            TokenType::Currency => TokenType::Currency,
            TokenType::Emoji => TokenType::Emoji,
        }
    }
}

impl PartialEq for TokenType {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for TokenType {}

impl Hash for TokenType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

impl Clone for Language {
    fn clone(&self) -> Self {
        match self {
            Language::English => Language::English,
            Language::Spanish => Language::Spanish,
            Language::French => Language::French,
            Language::German => Language::German,
            Language::Italian => Language::Italian,
            Language::Portuguese => Language::Portuguese,
            Language::Russian => Language::Russian,
            Language::Dutch => Language::Dutch,
            Language::Swedish => Language::Swedish,
            Language::Norwegian => Language::Norwegian,
            Language::Danish => Language::Danish,
            Language::Finnish => Language::Finnish,
            Language::Polish => Language::Polish,
            Language::Czech => Language::Czech,
            Language::Hungarian => Language::Hungarian,
            Language::Romanian => Language::Romanian,
            Language::Bulgarian => Language::Bulgarian,
            Language::Croatian => Language::Croatian,
            Language::Serbian => Language::Serbian,
            Language::Slovak => Language::Slovak,
            Language::Slovenian => Language::Slovenian,
            Language::Estonian => Language::Estonian,
            Language::Latvian => Language::Latvian,
            Language::Lithuanian => Language::Lithuanian,
            Language::Greek => Language::Greek,
            Language::Turkish => Language::Turkish,
            Language::Hebrew => Language::Hebrew,
            Language::Arabic => Language::Arabic,
            Language::Chinese => Language::Chinese,
            Language::Japanese => Language::Japanese,
            Language::Korean => Language::Korean,
            Language::Hindi => Language::Hindi,
            Language::Thai => Language::Thai,
            Language::Vietnamese => Language::Vietnamese,
            Language::Indonesian => Language::Indonesian,
            Language::Malay => Language::Malay,
            Language::Tagalog => Language::Tagalog,
            Language::Ukrainian => Language::Ukrainian,
            Language::Belarusian => Language::Belarusian,
        }
    }
}

impl PartialEq for Language {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for Language {}

impl Hash for Language {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

impl Clone for Script {
    fn clone(&self) -> Self {
        match self {
            Script::Latin => Script::Latin,
            Script::Cyrillic => Script::Cyrillic,
            Script::Arabic => Script::Arabic,
            Script::Chinese => Script::Chinese,
            Script::Japanese => Script::Japanese,
            Script::Korean => Script::Korean,
            Script::Devanagari => Script::Devanagari,
            Script::Thai => Script::Thai,
            Script::Hebrew => Script::Hebrew,
            Script::Greek => Script::Greek,
        }
    }
}

impl PartialEq for Script {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for Script {}

impl Hash for Script {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

impl Clone for TokenizerStats {
    fn clone(&self) -> Self {
        Self {
            total_texts_processed: self.total_texts_processed,
            total_tokens_generated: self.total_tokens_generated,
            language_detection_accuracy: self.language_detection_accuracy,
            average_processing_time: self.average_processing_time,
            token_frequency: self.token_frequency.clone(),
            language_distribution: self.language_distribution.clone(),
            token_type_distribution: self.token_type_distribution.clone(),
        }
    }
}

impl Default for TokenizerStats {
    fn default() -> Self {
        Self {
            total_texts_processed: 0,
            total_tokens_generated: 0,
            language_detection_accuracy: 0.0,
            average_processing_time: Duration::default(),
            token_frequency: HashMap::new(),
            language_distribution: HashMap::new(),
            token_type_distribution: HashMap::new(),
        }
    }
}