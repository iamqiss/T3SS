// T3SS Project
// File: core/indexing/quality_assessor/domain_reputation_tracker.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use log::{info, warn, error};
use chrono::{DateTime, Utc, NaiveDateTime};
use uuid::Uuid;

/// Domain Reputation Tracker for advanced domain reputation management
/// 
/// This module provides a comprehensive domain reputation tracking system with support for:
/// - Real-time reputation scoring and monitoring
/// - Historical reputation tracking and trends
/// - Multi-source reputation data aggregation
/// - Automated reputation updates and notifications
/// - Domain classification and categorization
/// - Reputation-based content filtering
/// - Performance optimization and caching
/// - Comprehensive statistics and monitoring

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    pub update_interval: Duration,
    pub cache_ttl: Duration,
    pub max_history_days: u32,
    pub enable_real_time_updates: bool,
    pub enable_notifications: bool,
    pub reputation_sources: Vec<String>,
    pub scoring_weights: HashMap<String, f32>,
    pub threshold_config: ThresholdConfig,
    pub enable_caching: bool,
    pub enable_persistence: bool,
    pub persistence_path: String,
    pub enable_analytics: bool,
    pub enable_alerts: bool,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub excellent_threshold: f32,
    pub good_threshold: f32,
    pub fair_threshold: f32,
    pub poor_threshold: f32,
    pub very_poor_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub reputation_drop_threshold: f32,
    pub reputation_rise_threshold: f32,
    pub suspicious_activity_threshold: f32,
    pub spam_detection_threshold: f32,
    pub malware_detection_threshold: f32,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(3600), // 1 hour
            cache_ttl: Duration::from_secs(7200), // 2 hours
            max_history_days: 365,
            enable_real_time_updates: true,
            enable_notifications: true,
            reputation_sources: vec![
                "internal".to_string(),
                "external_api".to_string(),
                "user_reports".to_string(),
                "automated_scanning".to_string(),
            ],
            scoring_weights: {
                let mut weights = HashMap::new();
                weights.insert("internal".to_string(), 0.4);
                weights.insert("external_api".to_string(), 0.3);
                weights.insert("user_reports".to_string(), 0.2);
                weights.insert("automated_scanning".to_string(), 0.1);
                weights
            },
            threshold_config: ThresholdConfig {
                excellent_threshold: 0.8,
                good_threshold: 0.6,
                fair_threshold: 0.4,
                poor_threshold: 0.2,
                very_poor_threshold: 0.0,
            },
            enable_caching: true,
            enable_persistence: true,
            persistence_path: "data/reputation".to_string(),
            enable_analytics: true,
            enable_alerts: true,
            alert_thresholds: AlertThresholds {
                reputation_drop_threshold: 0.2,
                reputation_rise_threshold: 0.3,
                suspicious_activity_threshold: 0.1,
                spam_detection_threshold: 0.05,
                malware_detection_threshold: 0.01,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainReputation {
    pub domain: String,
    pub current_score: f32,
    pub reputation_level: ReputationLevel,
    pub confidence: f32,
    pub last_updated: DateTime<Utc>,
    pub sources: Vec<ReputationSource>,
    pub history: Vec<ReputationHistoryEntry>,
    pub metadata: HashMap<String, String>,
    pub alerts: Vec<ReputationAlert>,
    pub trends: ReputationTrends,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationSource {
    pub name: String,
    pub score: f32,
    pub confidence: f32,
    pub last_updated: DateTime<Utc>,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub score: f32,
    pub source: String,
    pub change_reason: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationAlert {
    pub id: Uuid,
    pub domain: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    ReputationDrop,
    ReputationRise,
    SuspiciousActivity,
    SpamDetection,
    MalwareDetection,
    NewDomain,
    DomainExpired,
    CertificateExpired,
    DnsChanged,
    ContentChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReputationLevel {
    Excellent,
    Good,
    Fair,
    Poor,
    VeryPoor,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationTrends {
    pub trend_direction: TrendDirection,
    pub trend_strength: f32,
    pub volatility: f32,
    pub stability: f32,
    pub recent_changes: Vec<ReputationChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Rising,
    Falling,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationChange {
    pub timestamp: DateTime<Utc>,
    pub old_score: f32,
    pub new_score: f32,
    pub change_amount: f32,
    pub change_percentage: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationUpdate {
    pub domain: String,
    pub source: String,
    pub score: f32,
    pub confidence: f32,
    pub reason: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationQuery {
    pub domain: String,
    pub include_history: bool,
    pub include_trends: bool,
    pub include_alerts: bool,
    pub max_history_days: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationStats {
    pub total_domains: u64,
    pub total_updates: u64,
    pub average_score: f32,
    pub reputation_distribution: HashMap<ReputationLevel, u64>,
    pub source_distribution: HashMap<String, u64>,
    pub alert_count: u64,
    pub last_update: DateTime<Utc>,
    pub cache_hit_rate: f32,
    pub update_frequency: f32,
    pub error_rate: f32,
}

/// Domain Reputation Tracker
pub struct DomainReputationTracker {
    config: ReputationConfig,
    reputations: Arc<RwLock<HashMap<String, DomainReputation>>>,
    cache: Arc<RwLock<HashMap<String, DomainReputation>>>,
    stats: Arc<RwLock<ReputationStats>>,
    alert_handlers: Arc<RwLock<Vec<Box<dyn AlertHandler + Send + Sync>>>>,
    update_scheduler: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

/// Trait for alert handlers
pub trait AlertHandler {
    fn handle_alert(&self, alert: &ReputationAlert) -> Result<()>;
    fn can_handle(&self, alert_type: &AlertType) -> bool;
}

impl DomainReputationTracker {
    /// Create a new domain reputation tracker
    pub fn new(config: ReputationConfig) -> Result<Self> {
        let stats = ReputationStats {
            total_domains: 0,
            total_updates: 0,
            average_score: 0.0,
            reputation_distribution: HashMap::new(),
            source_distribution: HashMap::new(),
            alert_count: 0,
            last_update: Utc::now(),
            cache_hit_rate: 0.0,
            update_frequency: 0.0,
            error_rate: 0.0,
        };

        Ok(Self {
            config,
            reputations: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(stats)),
            alert_handlers: Arc::new(RwLock::new(Vec::new())),
            update_scheduler: Arc::new(RwLock::new(None)),
        })
    }

    /// Initialize the reputation tracker
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing domain reputation tracker");
        
        // Load existing data if persistence is enabled
        if self.config.enable_persistence {
            self.load_reputations().await?;
        }
        
        // Start update scheduler if real-time updates are enabled
        if self.config.enable_real_time_updates {
            self.start_update_scheduler().await?;
        }
        
        info!("Domain reputation tracker initialized successfully");
        Ok(())
    }

    /// Get domain reputation
    pub async fn get_reputation(&self, domain: &str) -> Result<Option<DomainReputation>> {
        // Check cache first
        if self.config.enable_caching {
            let cache = self.cache.read().await;
            if let Some(reputation) = cache.get(domain) {
                return Ok(Some(reputation.clone()));
            }
        }
        
        // Get from main storage
        let reputations = self.reputations.read().await;
        let reputation = reputations.get(domain).cloned();
        
        // Update cache
        if let Some(ref reputation) = reputation {
            if self.config.enable_caching {
                let mut cache = self.cache.write().await;
                cache.insert(domain.to_string(), reputation.clone());
            }
        }
        
        Ok(reputation)
    }

    /// Update domain reputation
    pub async fn update_reputation(&self, update: ReputationUpdate) -> Result<()> {
        let domain = update.domain.clone();
        let mut reputations = self.reputations.write().await;
        
        // Get or create domain reputation
        let reputation = reputations.entry(domain.clone()).or_insert_with(|| {
            DomainReputation {
                domain: domain.clone(),
                current_score: 0.0,
                reputation_level: ReputationLevel::Unknown,
                confidence: 0.0,
                last_updated: Utc::now(),
                sources: Vec::new(),
                history: Vec::new(),
                metadata: HashMap::new(),
                alerts: Vec::new(),
                trends: ReputationTrends {
                    trend_direction: TrendDirection::Stable,
                    trend_strength: 0.0,
                    volatility: 0.0,
                    stability: 0.0,
                    recent_changes: Vec::new(),
                },
            }
        });
        
        // Update source
        let source = ReputationSource {
            name: update.source.clone(),
            score: update.score,
            confidence: update.confidence,
            last_updated: Utc::now(),
            weight: self.config.scoring_weights.get(&update.source).copied().unwrap_or(1.0),
            metadata: update.metadata.clone(),
        };
        
        // Update or add source
        if let Some(existing_source) = reputation.sources.iter_mut().find(|s| s.name == update.source) {
            *existing_source = source;
        } else {
            reputation.sources.push(source);
        }
        
        // Calculate new overall score
        let old_score = reputation.current_score;
        reputation.current_score = self.calculate_overall_score(&reputation.sources);
        reputation.confidence = self.calculate_confidence(&reputation.sources);
        reputation.last_updated = Utc::now();
        
        // Update reputation level
        reputation.reputation_level = self.determine_reputation_level(reputation.current_score);
        
        // Add to history
        let history_entry = ReputationHistoryEntry {
            timestamp: Utc::now(),
            score: reputation.current_score,
            source: update.source,
            change_reason: update.reason,
            metadata: update.metadata,
        };
        reputation.history.push(history_entry);
        
        // Update trends
        self.update_trends(reputation).await?;
        
        // Check for alerts
        if self.config.enable_alerts {
            self.check_alerts(reputation, old_score, reputation.current_score).await?;
        }
        
        // Update cache
        if self.config.enable_caching {
            let mut cache = self.cache.write().await;
            cache.insert(domain.clone(), reputation.clone());
        }
        
        // Update statistics
        self.update_stats().await?;
        
        info!("Updated reputation for domain {}: {} -> {}", domain, old_score, reputation.current_score);
        Ok(())
    }

    /// Batch update multiple domain reputations
    pub async fn batch_update_reputations(&self, updates: Vec<ReputationUpdate>) -> Result<()> {
        for update in updates {
            if let Err(e) = self.update_reputation(update).await {
                error!("Failed to update reputation: {}", e);
            }
        }
        Ok(())
    }

    /// Get reputation statistics
    pub async fn get_stats(&self) -> Result<ReputationStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Add alert handler
    pub async fn add_alert_handler(&self, handler: Box<dyn AlertHandler + Send + Sync>) -> Result<()> {
        let mut handlers = self.alert_handlers.write().await;
        handlers.push(handler);
        Ok(())
    }

    /// Clear reputation cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Export reputation data
    pub async fn export_reputations(&self, format: ExportFormat) -> Result<Vec<u8>> {
        let reputations = self.reputations.read().await;
        
        match format {
            ExportFormat::Json => {
                let data = serde_json::to_vec(&*reputations)?;
                Ok(data)
            }
            ExportFormat::Csv => {
                let mut csv = String::new();
                csv.push_str("domain,score,level,confidence,last_updated\n");
                
                for reputation in reputations.values() {
                    csv.push_str(&format!(
                        "{},{},{:?},{},{}\n",
                        reputation.domain,
                        reputation.current_score,
                        reputation.reputation_level,
                        reputation.confidence,
                        reputation.last_updated.format("%Y-%m-%d %H:%M:%S UTC")
                    ));
                }
                
                Ok(csv.into_bytes())
            }
        }
    }

    /// Import reputation data
    pub async fn import_reputations(&self, data: &[u8], format: ExportFormat) -> Result<()> {
        match format {
            ExportFormat::Json => {
                let reputations: HashMap<String, DomainReputation> = serde_json::from_slice(data)?;
                let mut storage = self.reputations.write().await;
                for (domain, reputation) in reputations {
                    storage.insert(domain, reputation);
                }
            }
            ExportFormat::Csv => {
                let csv = String::from_utf8(data.to_vec())?;
                let lines: Vec<&str> = csv.lines().collect();
                
                for (i, line) in lines.iter().enumerate() {
                    if i == 0 { continue; } // Skip header
                    
                    let fields: Vec<&str> = line.split(',').collect();
                    if fields.len() >= 5 {
                        let domain = fields[0].to_string();
                        let score: f32 = fields[1].parse().unwrap_or(0.0);
                        let level = self.parse_reputation_level(fields[2]);
                        let confidence: f32 = fields[3].parse().unwrap_or(0.0);
                        let last_updated = DateTime::parse_from_rfc3339(fields[4])
                            .unwrap_or_else(|_| Utc::now())
                            .with_timezone(&Utc);
                        
                        let reputation = DomainReputation {
                            domain: domain.clone(),
                            current_score: score,
                            reputation_level: level,
                            confidence,
                            last_updated,
                            sources: Vec::new(),
                            history: Vec::new(),
                            metadata: HashMap::new(),
                            alerts: Vec::new(),
                            trends: ReputationTrends {
                                trend_direction: TrendDirection::Stable,
                                trend_strength: 0.0,
                                volatility: 0.0,
                                stability: 0.0,
                                recent_changes: Vec::new(),
                            },
                        };
                        
                        let mut storage = self.reputations.write().await;
                        storage.insert(domain, reputation);
                    }
                }
            }
        }
        
        Ok(())
    }

    // Private helper methods
    async fn load_reputations(&self) -> Result<()> {
        // Load from persistence storage
        // This would read from files or database
        info!("Loading reputations from persistence storage");
        Ok(())
    }

    async fn start_update_scheduler(&self) -> Result<()> {
        let config = self.config.clone();
        let reputations = self.reputations.clone();
        let stats = self.stats.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.update_interval);
            
            loop {
                interval.tick().await;
                
                // Update reputations from external sources
                if let Err(e) = Self::update_from_external_sources(&reputations, &stats, &config).await {
                    error!("Failed to update from external sources: {}", e);
                }
            }
        });
        
        let mut scheduler = self.update_scheduler.write().await;
        *scheduler = Some(handle);
        
        Ok(())
    }

    async fn update_from_external_sources(
        reputations: &Arc<RwLock<HashMap<String, DomainReputation>>>,
        stats: &Arc<RwLock<ReputationStats>>,
        config: &ReputationConfig,
    ) -> Result<()> {
        // This would fetch data from external reputation sources
        // For now, just log the action
        info!("Updating reputations from external sources");
        Ok(())
    }

    fn calculate_overall_score(&self, sources: &[ReputationSource]) -> f32 {
        if sources.is_empty() {
            return 0.0;
        }
        
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for source in sources {
            weighted_sum += source.score * source.weight;
            total_weight += source.weight;
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    fn calculate_confidence(&self, sources: &[ReputationSource]) -> f32 {
        if sources.is_empty() {
            return 0.0;
        }
        
        let mut weighted_confidence = 0.0;
        let mut total_weight = 0.0;
        
        for source in sources {
            weighted_confidence += source.confidence * source.weight;
            total_weight += source.weight;
        }
        
        if total_weight > 0.0 {
            weighted_confidence / total_weight
        } else {
            0.0
        }
    }

    fn determine_reputation_level(&self, score: f32) -> ReputationLevel {
        if score >= self.config.threshold_config.excellent_threshold {
            ReputationLevel::Excellent
        } else if score >= self.config.threshold_config.good_threshold {
            ReputationLevel::Good
        } else if score >= self.config.threshold_config.fair_threshold {
            ReputationLevel::Fair
        } else if score >= self.config.threshold_config.poor_threshold {
            ReputationLevel::Poor
        } else if score >= self.config.threshold_config.very_poor_threshold {
            ReputationLevel::VeryPoor
        } else {
            ReputationLevel::Unknown
        }
    }

    async fn update_trends(&self, reputation: &mut DomainReputation) -> Result<()> {
        if reputation.history.len() < 2 {
            return Ok(());
        }
        
        let recent_entries = &reputation.history[reputation.history.len().saturating_sub(10)..];
        let scores: Vec<f32> = recent_entries.iter().map(|e| e.score).collect();
        
        // Calculate trend direction
        let first_score = scores.first().unwrap_or(&0.0);
        let last_score = scores.last().unwrap_or(&0.0);
        let change = last_score - first_score;
        
        reputation.trends.trend_direction = if change > 0.1 {
            TrendDirection::Rising
        } else if change < -0.1 {
            TrendDirection::Falling
        } else {
            TrendDirection::Stable
        };
        
        // Calculate trend strength
        reputation.trends.trend_strength = change.abs();
        
        // Calculate volatility
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        reputation.trends.volatility = variance.sqrt();
        
        // Calculate stability
        reputation.trends.stability = 1.0 - (reputation.trends.volatility / mean.max(0.1));
        
        Ok(())
    }

    async fn check_alerts(&self, reputation: &mut DomainReputation, old_score: f32, new_score: f32) -> Result<()> {
        let change = new_score - old_score;
        let change_percentage = if old_score > 0.0 { (change / old_score).abs() } else { 0.0 };
        
        // Check for reputation drop
        if change < -self.config.alert_thresholds.reputation_drop_threshold {
            let alert = ReputationAlert {
                id: Uuid::new_v4(),
                domain: reputation.domain.clone(),
                alert_type: AlertType::ReputationDrop,
                severity: if change_percentage > 0.5 { AlertSeverity::High } else { AlertSeverity::Medium },
                message: format!("Domain reputation dropped by {:.2}%", change_percentage * 100.0),
                timestamp: Utc::now(),
                resolved: false,
                metadata: HashMap::new(),
            };
            reputation.alerts.push(alert.clone());
            self.handle_alert(&alert).await?;
        }
        
        // Check for reputation rise
        if change > self.config.alert_thresholds.reputation_rise_threshold {
            let alert = ReputationAlert {
                id: Uuid::new_v4(),
                domain: reputation.domain.clone(),
                alert_type: AlertType::ReputationRise,
                severity: AlertSeverity::Low,
                message: format!("Domain reputation rose by {:.2}%", change_percentage * 100.0),
                timestamp: Utc::now(),
                resolved: false,
                metadata: HashMap::new(),
            };
            reputation.alerts.push(alert.clone());
            self.handle_alert(&alert).await?;
        }
        
        Ok(())
    }

    async fn handle_alert(&self, alert: &ReputationAlert) -> Result<()> {
        let handlers = self.alert_handlers.read().await;
        
        for handler in handlers.iter() {
            if handler.can_handle(&alert.alert_type) {
                if let Err(e) = handler.handle_alert(alert) {
                    error!("Alert handler failed: {}", e);
                }
            }
        }
        
        Ok(())
    }

    async fn update_stats(&self) -> Result<()> {
        let reputations = self.reputations.read().await;
        let mut stats = self.stats.write().await;
        
        stats.total_domains = reputations.len() as u64;
        stats.average_score = if !reputations.is_empty() {
            reputations.values().map(|r| r.current_score).sum::<f32>() / reputations.len() as f32
        } else {
            0.0
        };
        
        // Update reputation distribution
        stats.reputation_distribution.clear();
        for reputation in reputations.values() {
            let level = format!("{:?}", reputation.reputation_level);
            *stats.reputation_distribution.entry(level).or_insert(0) += 1;
        }
        
        // Update source distribution
        stats.source_distribution.clear();
        for reputation in reputations.values() {
            for source in &reputation.sources {
                *stats.source_distribution.entry(source.name.clone()).or_insert(0) += 1;
            }
        }
        
        stats.last_update = Utc::now();
        
        Ok(())
    }

    fn parse_reputation_level(&self, level_str: &str) -> ReputationLevel {
        match level_str {
            "Excellent" => ReputationLevel::Excellent,
            "Good" => ReputationLevel::Good,
            "Fair" => ReputationLevel::Fair,
            "Poor" => ReputationLevel::Poor,
            "VeryPoor" => ReputationLevel::VeryPoor,
            _ => ReputationLevel::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
}

// Example alert handler
pub struct LoggingAlertHandler;

impl AlertHandler for LoggingAlertHandler {
    fn handle_alert(&self, alert: &ReputationAlert) -> Result<()> {
        info!("Alert: {} - {}", alert.alert_type, alert.message);
        Ok(())
    }

    fn can_handle(&self, _alert_type: &AlertType) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_domain_reputation_tracker() {
        let config = ReputationConfig::default();
        let tracker = DomainReputationTracker::new(config).unwrap();
        
        // Test initialization
        tracker.initialize().await.unwrap();
        
        // Test reputation update
        let update = ReputationUpdate {
            domain: "example.com".to_string(),
            source: "internal".to_string(),
            score: 0.8,
            confidence: 0.9,
            reason: "Test update".to_string(),
            metadata: HashMap::new(),
        };
        
        tracker.update_reputation(update).await.unwrap();
        
        // Test reputation retrieval
        let reputation = tracker.get_reputation("example.com").await.unwrap();
        assert!(reputation.is_some());
        assert_eq!(reputation.unwrap().current_score, 0.8);
    }

    #[tokio::test]
    async fn test_batch_update() {
        let config = ReputationConfig::default();
        let tracker = DomainReputationTracker::new(config).unwrap();
        tracker.initialize().await.unwrap();
        
        let updates = vec![
            ReputationUpdate {
                domain: "example1.com".to_string(),
                source: "internal".to_string(),
                score: 0.7,
                confidence: 0.8,
                reason: "Test update 1".to_string(),
                metadata: HashMap::new(),
            },
            ReputationUpdate {
                domain: "example2.com".to_string(),
                source: "internal".to_string(),
                score: 0.9,
                confidence: 0.9,
                reason: "Test update 2".to_string(),
                metadata: HashMap::new(),
            },
        ];
        
        tracker.batch_update_reputations(updates).await.unwrap();
        
        let reputation1 = tracker.get_reputation("example1.com").await.unwrap();
        let reputation2 = tracker.get_reputation("example2.com").await.unwrap();
        
        assert!(reputation1.is_some());
        assert!(reputation2.is_some());
        assert_eq!(reputation1.unwrap().current_score, 0.7);
        assert_eq!(reputation2.unwrap().current_score, 0.9);
    }
}
