// T3SS Project
// File: core/indexing/crawler/parser/html_parser.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use regex::Regex;
use serde::{Deserialize, Serialize};
use url::Url;
use html5ever::{
    parse_document, rcdom::{RcDom, NodeData, Handle},
    tendril::TendrilSink,
    driver::{ParseOpts, process},
};
use html5ever::tendril::StrTendril;
use html5ever::tree_builder::{TreeBuilderOpts, QuirksMode};
use html5ever::interface::QualName;
use html5ever::local_name;

/// Configuration for HTML parsing
#[derive(Debug, Clone)]
pub struct HtmlParserConfig {
    pub max_content_length: usize,
    pub extract_links: bool,
    pub extract_images: bool,
    pub extract_metadata: bool,
    pub extract_text_content: bool,
    pub normalize_urls: bool,
    pub follow_redirects: bool,
    pub max_links_per_page: usize,
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    pub min_link_text_length: usize,
    pub max_link_text_length: usize,
}

impl Default for HtmlParserConfig {
    fn default() -> Self {
        Self {
            max_content_length: 10 * 1024 * 1024, // 10MB
            extract_links: true,
            extract_images: true,
            extract_metadata: true,
            extract_text_content: true,
            normalize_urls: true,
            follow_redirects: true,
            max_links_per_page: 1000,
            allowed_domains: vec![],
            blocked_domains: vec![],
            min_link_text_length: 1,
            max_link_text_length: 200,
        }
    }
}

/// Represents a parsed HTML document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub url: String,
    pub title: String,
    pub meta_description: String,
    pub meta_keywords: Vec<String>,
    pub text_content: String,
    pub links: Vec<ExtractedLink>,
    pub images: Vec<ExtractedImage>,
    pub metadata: HashMap<String, String>,
    pub language: String,
    pub charset: String,
    pub content_length: usize,
    pub parsing_time: Duration,
    pub link_count: usize,
    pub image_count: usize,
}

/// Represents an extracted link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLink {
    pub url: String,
    pub anchor_text: String,
    pub title: String,
    pub rel: String,
    pub link_type: LinkType,
    pub confidence: f64,
}

/// Represents an extracted image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedImage {
    pub src: String,
    pub alt: String,
    pub title: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub loading: String,
}

/// Link types for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkType {
    Internal,
    External,
    Navigational,
    Content,
    Social,
    Reference,
    Unknown,
}

/// High-performance HTML parser with advanced content extraction
pub struct HtmlParser {
    config: HtmlParserConfig,
    link_regex: Regex,
    image_regex: Regex,
    stats: Arc<Mutex<ParserStats>>,
}

/// Statistics for HTML parsing
#[derive(Debug, Default)]
pub struct ParserStats {
    pub total_documents_parsed: u64,
    pub total_links_extracted: u64,
    pub total_images_extracted: u64,
    pub average_parsing_time: Duration,
    pub total_parsing_time: Duration,
    pub failed_parses: u64,
}

impl HtmlParser {
    /// Create a new HTML parser
    pub fn new(config: HtmlParserConfig) -> Self {
        Self {
            config,
            link_regex: Regex::new(r#"(?i)<a\s+[^>]*href\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>"#).unwrap(),
            image_regex: Regex::new(r#"(?i)<img\s+[^>]*src\s*=\s*["']([^"']+)["'][^>]*>"#).unwrap(),
            stats: Arc::new(Mutex::new(ParserStats::default())),
        }
    }

    /// Parse HTML content and extract structured data
    pub fn parse(&self, html_content: &str, base_url: &str) -> Result<ParsedDocument, String> {
        let start_time = Instant::now();
        
        // Validate content length
        if html_content.len() > self.config.max_content_length {
            return Err(format!("Content too large: {} bytes", html_content.len()));
        }

        // Parse HTML using html5ever
        let dom = parse_document(RcDom::default(), ParseOpts::default())
            .from_utf8()
            .read_from(&mut html_content.as_bytes())
            .map_err(|e| format!("HTML parsing error: {}", e))?;

        let mut document = ParsedDocument {
            url: base_url.to_string(),
            title: String::new(),
            meta_description: String::new(),
            meta_keywords: Vec::new(),
            text_content: String::new(),
            links: Vec::new(),
            images: Vec::new(),
            metadata: HashMap::new(),
            language: String::new(),
            charset: String::new(),
            content_length: html_content.len(),
            parsing_time: Duration::from_secs(0),
            link_count: 0,
            image_count: 0,
        };

        // Extract data from DOM
        self.extract_title(&dom.document, &mut document);
        self.extract_metadata(&dom.document, &mut document);
        self.extract_text_content(&dom.document, &mut document);
        
        if self.config.extract_links {
            self.extract_links(&dom.document, &mut document, base_url);
        }
        
        if self.config.extract_images {
            self.extract_images(&dom.document, &mut document, base_url);
        }

        document.parsing_time = start_time.elapsed();
        document.link_count = document.links.len();
        document.image_count = document.images.len();

        // Update statistics
        self.update_stats(true, document.parsing_time, document.link_count, document.image_count);

        Ok(document)
    }

    /// Extract page title
    fn extract_title(&self, document: &Handle, parsed_doc: &mut ParsedDocument) {
        if let Some(title_node) = self.find_element_by_tag(document, "title") {
            parsed_doc.title = self.extract_text_content_from_node(&title_node);
        }
    }

    /// Extract metadata (description, keywords, etc.)
    fn extract_metadata(&self, document: &Handle, parsed_doc: &mut ParsedDocument) {
        self.extract_meta_tags(document, parsed_doc);
        self.extract_language(document, parsed_doc);
        self.extract_charset(document, parsed_doc);
    }

    /// Extract meta tags
    fn extract_meta_tags(&self, document: &Handle, parsed_doc: &mut ParsedDocument) {
        let meta_nodes = self.find_elements_by_tag(document, "meta");
        
        for meta_node in meta_nodes {
            if let NodeData::Element { ref attrs, .. } = meta_node.data {
                let mut name = String::new();
                let mut content = String::new();
                
                for attr in attrs.borrow().iter() {
                    match attr.name.local.as_ref() {
                        "name" => name = attr.value.to_string(),
                        "content" => content = attr.value.to_string(),
                        _ => {}
                    }
                }
                
                match name.to_lowercase().as_str() {
                    "description" => parsed_doc.meta_description = content,
                    "keywords" => {
                        parsed_doc.meta_keywords = content
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                    _ => {
                        parsed_doc.metadata.insert(name, content);
                    }
                }
            }
        }
    }

    /// Extract language from HTML lang attribute
    fn extract_language(&self, document: &Handle, parsed_doc: &mut ParsedDocument) {
        if let NodeData::Element { ref attrs, .. } = document.data {
            for attr in attrs.borrow().iter() {
                if attr.name.local.as_ref() == "lang" {
                    parsed_doc.language = attr.value.to_string();
                    break;
                }
            }
        }
    }

    /// Extract charset from meta tag
    fn extract_charset(&self, document: &Handle, parsed_doc: &mut ParsedDocument) {
        let meta_nodes = self.find_elements_by_tag(document, "meta");
        
        for meta_node in meta_nodes {
            if let NodeData::Element { ref attrs, .. } = meta_node.data {
                for attr in attrs.borrow().iter() {
                    if attr.name.local.as_ref() == "charset" {
                        parsed_doc.charset = attr.value.to_string();
                        return;
                    }
                }
            }
        }
    }

    /// Extract text content from the document
    fn extract_text_content(&self, document: &Handle, parsed_doc: &mut ParsedDocument) {
        parsed_doc.text_content = self.extract_text_content_from_node(document);
    }

    /// Extract links from the document
    fn extract_links(&self, document: &Handle, parsed_doc: &mut ParsedDocument, base_url: &str) {
        let link_nodes = self.find_elements_by_tag(document, "a");
        
        for link_node in link_nodes {
            if let Some(link) = self.extract_link_from_node(&link_node, base_url) {
                if self.is_valid_link(&link) {
                    parsed_doc.links.push(link);
                }
            }
        }
    }

    /// Extract images from the document
    fn extract_images(&self, document: &Handle, parsed_doc: &mut ParsedDocument, base_url: &str) {
        let img_nodes = self.find_elements_by_tag(document, "img");
        
        for img_node in img_nodes {
            if let Some(image) = self.extract_image_from_node(&img_node, base_url) {
                parsed_doc.images.push(image);
            }
        }
    }

    /// Extract a single link from a node
    fn extract_link_from_node(&self, node: &Handle, base_url: &str) -> Option<ExtractedLink> {
        if let NodeData::Element { ref attrs, .. } = node.data {
            let mut href = String::new();
            let mut title = String::new();
            let mut rel = String::new();
            
            for attr in attrs.borrow().iter() {
                match attr.name.local.as_ref() {
                    "href" => href = attr.value.to_string(),
                    "title" => title = attr.value.to_string(),
                    "rel" => rel = attr.value.to_string(),
                    _ => {}
                }
            }
            
            if href.is_empty() {
                return None;
            }
            
            // Normalize URL
            let normalized_url = if self.config.normalize_urls {
                self.normalize_url(&href, base_url)
            } else {
                href
            };
            
            // Extract anchor text
            let anchor_text = self.extract_text_content_from_node(node);
            
            // Determine link type
            let link_type = self.classify_link_type(&normalized_url, base_url, &anchor_text, &rel);
            
            // Calculate confidence
            let confidence = self.calculate_link_confidence(&anchor_text, &title, &rel);
            
            Some(ExtractedLink {
                url: normalized_url,
                anchor_text,
                title,
                rel,
                link_type,
                confidence,
            })
        } else {
            None
        }
    }

    /// Extract a single image from a node
    fn extract_image_from_node(&self, node: &Handle, base_url: &str) -> Option<ExtractedImage> {
        if let NodeData::Element { ref attrs, .. } = node.data {
            let mut src = String::new();
            let mut alt = String::new();
            let mut title = String::new();
            let mut width = None;
            let mut height = None;
            let mut loading = String::new();
            
            for attr in attrs.borrow().iter() {
                match attr.name.local.as_ref() {
                    "src" => src = attr.value.to_string(),
                    "alt" => alt = attr.value.to_string(),
                    "title" => title = attr.value.to_string(),
                    "width" => width = attr.value.parse().ok(),
                    "height" => height = attr.value.parse().ok(),
                    "loading" => loading = attr.value.to_string(),
                    _ => {}
                }
            }
            
            if src.is_empty() {
                return None;
            }
            
            // Normalize URL
            let normalized_src = if self.config.normalize_urls {
                self.normalize_url(&src, base_url)
            } else {
                src
            };
            
            Some(ExtractedImage {
                src: normalized_src,
                alt,
                title,
                width,
                height,
                loading,
            })
        } else {
            None
        }
    }

    /// Normalize URL relative to base URL
    fn normalize_url(&self, url: &str, base_url: &str) -> String {
        if let Ok(base) = Url::parse(base_url) {
            if let Ok(parsed_url) = base.join(url) {
                return parsed_url.to_string();
            }
        }
        url.to_string()
    }

    /// Classify link type based on URL, anchor text, and rel attribute
    fn classify_link_type(&self, url: &str, base_url: &str, anchor_text: &str, rel: &str) -> LinkType {
        // Check rel attribute first
        if rel.contains("nofollow") {
            return LinkType::Reference;
        }
        
        // Check if internal or external
        let base_domain = self.extract_domain(base_url);
        let link_domain = self.extract_domain(url);
        
        if base_domain == link_domain {
            return LinkType::Internal;
        }
        
        // Classify based on anchor text patterns
        let anchor_lower = anchor_text.to_lowercase();
        if anchor_lower.contains("home") || anchor_lower.contains("menu") || anchor_lower.contains("navigation") {
            return LinkType::Navigational;
        }
        
        if anchor_lower.contains("share") || anchor_lower.contains("tweet") || anchor_lower.contains("facebook") {
            return LinkType::Social;
        }
        
        if anchor_lower.contains("reference") || anchor_lower.contains("source") || anchor_lower.contains("cite") {
            return LinkType::Reference;
        }
        
        LinkType::Content
    }

    /// Calculate confidence score for a link
    fn calculate_link_confidence(&self, anchor_text: &str, title: &str, rel: &str) -> f64 {
        let mut confidence = 1.0;
        
        // Penalize empty anchor text
        if anchor_text.trim().is_empty() {
            confidence *= 0.3;
        }
        
        // Penalize generic anchor text
        let generic_patterns = ["click here", "read more", "more", "here", "link"];
        for pattern in &generic_patterns {
            if anchor_text.to_lowercase().contains(pattern) {
                confidence *= 0.7;
            }
        }
        
        // Reward descriptive anchor text
        if anchor_text.len() > 10 && anchor_text.len() < 100 {
            confidence *= 1.2;
        }
        
        // Penalize nofollow links
        if rel.contains("nofollow") {
            confidence *= 0.5;
        }
        
        // Reward links with title attribute
        if !title.is_empty() {
            confidence *= 1.1;
        }
        
        confidence.min(2.0).max(0.1)
    }

    /// Check if a link is valid according to configuration
    fn is_valid_link(&self, link: &ExtractedLink) -> bool {
        // Check anchor text length
        if link.anchor_text.len() < self.config.min_link_text_length ||
           link.anchor_text.len() > self.config.max_link_text_length {
            return false;
        }
        
        // Check domain restrictions
        let domain = self.extract_domain(&link.url);
        
        // Check blocked domains
        for blocked in &self.config.blocked_domains {
            if domain.contains(blocked) {
                return false;
            }
        }
        
        // Check allowed domains (if specified)
        if !self.config.allowed_domains.is_empty() {
            let mut allowed = false;
            for allowed_domain in &self.config.allowed_domains {
                if domain.contains(allowed_domain) {
                    allowed = true;
                    break;
                }
            }
            if !allowed {
                return false;
            }
        }
        
        true
    }

    /// Extract domain from URL
    fn extract_domain(&self, url: &str) -> String {
        if let Ok(parsed_url) = Url::parse(url) {
            if let Some(host) = parsed_url.host_str() {
                return host.to_string();
            }
        }
        String::new()
    }

    /// Find element by tag name
    fn find_element_by_tag(&self, node: &Handle, tag_name: &str) -> Option<Handle> {
        self.find_elements_by_tag(node, tag_name).into_iter().next()
    }

    /// Find elements by tag name
    fn find_elements_by_tag(&self, node: &Handle, tag_name: &str) -> Vec<Handle> {
        let mut elements = Vec::new();
        self.collect_elements_by_tag(node, tag_name, &mut elements);
        elements
    }

    /// Recursively collect elements by tag name
    fn collect_elements_by_tag(&self, node: &Handle, tag_name: &str, elements: &mut Vec<Handle>) {
        if let NodeData::Element { ref name, .. } = node.data {
            if name.local.as_ref() == tag_name {
                elements.push(node.clone());
            }
        }
        
        for child in node.children.borrow().iter() {
            self.collect_elements_by_tag(child, tag_name, elements);
        }
    }

    /// Extract text content from a node
    fn extract_text_content_from_node(&self, node: &Handle) -> String {
        let mut text = String::new();
        self.collect_text_from_node(node, &mut text);
        text.trim().to_string()
    }

    /// Recursively collect text from a node
    fn collect_text_from_node(&self, node: &Handle, text: &mut String) {
        match node.data {
            NodeData::Text { ref contents } => {
                text.push_str(&contents.borrow());
            }
            NodeData::Element { ref name, .. } => {
                // Skip script and style tags
                let tag_name = name.local.as_ref();
                if tag_name != "script" && tag_name != "style" {
                    for child in node.children.borrow().iter() {
                        self.collect_text_from_node(child, text);
                    }
                }
            }
            _ => {
                for child in node.children.borrow().iter() {
                    self.collect_text_from_node(child, text);
                }
            }
        }
    }

    /// Update parser statistics
    fn update_stats(&self, success: bool, parsing_time: Duration, link_count: usize, image_count: usize) {
        let mut stats = self.stats.lock().unwrap();
        
        if success {
            stats.total_documents_parsed += 1;
            stats.total_links_extracted += link_count as u64;
            stats.total_images_extracted += image_count as u64;
            
            // Update average parsing time
            if stats.average_parsing_time == Duration::from_secs(0) {
                stats.average_parsing_time = parsing_time;
            } else {
                stats.average_parsing_time = (stats.average_parsing_time * 9 + parsing_time) / 10;
            }
            
            stats.total_parsing_time += parsing_time;
        } else {
            stats.failed_parses += 1;
        }
    }

    /// Get current parser statistics
    pub fn get_stats(&self) -> ParserStats {
        self.stats.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_parsing() {
        let config = HtmlParserConfig::default();
        let parser = HtmlParser::new(config);
        
        let html = r#"
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="A test page">
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Hello World</h1>
            <p>This is a test page with <a href="https://example.com">a link</a>.</p>
            <img src="test.jpg" alt="Test image">
        </body>
        </html>
        "#;
        
        let result = parser.parse(html, "https://example.com/page").unwrap();
        
        assert_eq!(result.title, "Test Page");
        assert_eq!(result.meta_description, "A test page");
        assert_eq!(result.links.len(), 1);
        assert_eq!(result.images.len(), 1);
        assert!(result.text_content.contains("Hello World"));
    }
}
