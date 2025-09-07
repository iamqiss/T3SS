// T3SS Project
// File: core/indexing/spam_detector/heuristic_rules.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicResult {
    pub is_spam: bool,
    pub confidence: f64,
    pub score: f64,
}

pub struct HeuristicRules {
    spam_keywords: Vec<String>,
}

impl HeuristicRules {
    pub fn new() -> Self {
        Self {
            spam_keywords: vec![
                "viagra".to_string(),
                "casino".to_string(),
                "lottery".to_string(),
            ],
        }
    }

    pub fn analyze(&self, content: &str) -> HeuristicResult {
        let mut score = 0.0;
        let content_lower = content.to_lowercase();
        
        for keyword in &self.spam_keywords {
            if content_lower.contains(keyword) {
                score += 0.3;
            }
        }
        
        HeuristicResult {
            is_spam: score > 0.5,
            confidence: score.min(1.0),
            score,
        }
    }
}