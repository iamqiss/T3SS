// T3SS Project
// File: infrastructure/monitoring/metrics_collector.rs
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use uuid::Uuid;
use prometheus::{Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec, Registry};
use prometheus::core::{Collector, Desc, Opts};
use prometheus::proto::{CounterProto, GaugeProto, HistogramProto, MetricFamily};
use prometheus::Encoder;
use tokio::time::{interval, sleep};
use tokio_stream::{wrappers::IntervalStream, StreamExt};

/// Represents a metric measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMeasurement {
    pub id: String,
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub timestamp: u64,
    pub metric_type: MetricType,
}

/// Types of metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Configuration for the metrics collector
#[derive(Debug, Clone)]
pub struct MetricsCollectorConfig {
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub max_metrics_per_type: usize,
    pub enable_aggregation: bool,
    pub enable_anomaly_detection: bool,
    pub enable_alerting: bool,
    pub alert_thresholds: HashMap<String, f64>,
    pub export_formats: Vec<ExportFormat>,
    pub enable_real_time_streaming: bool,
    pub buffer_size: usize,
}

/// Export formats for metrics
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Prometheus,
    InfluxDB,
    Graphite,
    JSON,
    CSV,
}

impl Default for MetricsCollectorConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("cpu_usage".to_string(), 80.0);
        alert_thresholds.insert("memory_usage".to_string(), 85.0);
        alert_thresholds.insert("disk_usage".to_string(), 90.0);
        alert_thresholds.insert("response_time".to_string(), 1000.0);
        
        Self {
            collection_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(3600), // 1 hour
            max_metrics_per_type: 10000,
            enable_aggregation: true,
            enable_anomaly_detection: true,
            enable_alerting: true,
            alert_thresholds,
            export_formats: vec![ExportFormat::Prometheus, ExportFormat::JSON],
            enable_real_time_streaming: true,
            buffer_size: 1000,
        }
    }
}

/// Comprehensive metrics collector for the search engine
pub struct MetricsCollector {
    config: MetricsCollectorConfig,
    registry: Registry,
    metrics_buffer: Arc<Mutex<VecDeque<MetricMeasurement>>>,
    aggregated_metrics: Arc<RwLock<HashMap<String, AggregatedMetric>>>,
    anomaly_detector: Arc<Mutex<AnomalyDetector>>,
    alert_manager: Arc<Mutex<AlertManager>>,
    exporters: Vec<Box<dyn MetricExporter + Send + Sync>>,
    stats: Arc<Mutex<CollectorStats>>,
    shutdown_tx: tokio::sync::mpsc::Sender<()>,
}

/// Aggregated metric data
#[derive(Debug, Clone)]
pub struct AggregatedMetric {
    pub name: String,
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub last_updated: u64,
}

/// Anomaly detector for metrics
struct AnomalyDetector {
    models: HashMap<String, AnomalyModel>,
    threshold: f64,
    window_size: usize,
}

/// Anomaly detection model
struct AnomalyModel {
    name: String,
    historical_values: VecDeque<f64>,
    mean: f64,
    std_dev: f64,
    anomaly_count: u64,
}

/// Alert manager for metrics
struct AlertManager {
    active_alerts: HashMap<String, Alert>,
    alert_history: VecDeque<Alert>,
    notification_channels: Vec<Box<dyn NotificationChannel + Send + Sync>>,
}

/// Alert information
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub metric_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub threshold: f64,
    pub current_value: f64,
    pub timestamp: u64,
    pub status: AlertStatus,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert status
#[derive(Debug, Clone)]
pub enum AlertStatus {
    Active,
    Resolved,
    Acknowledged,
}

/// Notification channel trait
trait NotificationChannel {
    fn send_alert(&self, alert: &Alert) -> Result<(), String>;
    fn get_name(&self) -> String;
}

/// Metric exporter trait
trait MetricExporter {
    fn export_metrics(&self, metrics: &[MetricMeasurement]) -> Result<(), String>;
    fn get_name(&self) -> String;
}

/// Statistics for the metrics collector
#[derive(Debug, Default)]
pub struct CollectorStats {
    pub total_metrics_collected: u64,
    pub total_alerts_generated: u64,
    pub total_anomalies_detected: u64,
    pub export_operations: u64,
    pub export_failures: u64,
    pub buffer_overflows: u64,
    pub collection_errors: u64,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MetricsCollectorConfig) -> Self {
        let registry = Registry::new();
        let metrics_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size)));
        let aggregated_metrics = Arc::new(RwLock::new(HashMap::new()));
        let anomaly_detector = Arc::new(Mutex::new(AnomalyDetector::new()));
        let alert_manager = Arc::new(Mutex::new(AlertManager::new()));
        let stats = Arc::new(Mutex::new(CollectorStats::default()));
        
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel(1);

        let mut exporters = Vec::new();
        
        // Add Prometheus exporter
        exporters.push(Box::new(PrometheusExporter::new()));
        
        // Add JSON exporter
        exporters.push(Box::new(JSONExporter::new()));

        Self {
            config,
            registry,
            metrics_buffer,
            aggregated_metrics,
            anomaly_detector,
            alert_manager,
            exporters,
            stats,
            shutdown_tx,
        }
    }

    /// Start the metrics collector
    pub async fn start(&self) -> Result<(), String> {
        // Start collection loop
        let collection_task = self.start_collection_loop();
        
        // Start aggregation task
        let aggregation_task = self.start_aggregation_loop();
        
        // Start anomaly detection task
        let anomaly_task = self.start_anomaly_detection_loop();
        
        // Start export task
        let export_task = self.start_export_loop();
        
        // Start alert processing task
        let alert_task = self.start_alert_processing_loop();

        // Wait for shutdown signal
        tokio::select! {
            _ = collection_task => {},
            _ = aggregation_task => {},
            _ = anomaly_task => {},
            _ = export_task => {},
            _ = alert_task => {},
        }

        Ok(())
    }

    /// Record a metric measurement
    pub fn record_metric(&self, measurement: MetricMeasurement) -> Result<(), String> {
        let mut buffer = self.metrics_buffer.lock().unwrap();
        
        // Check buffer capacity
        if buffer.len() >= self.config.buffer_size {
            buffer.pop_front(); // Remove oldest metric
            self.stats.lock().unwrap().buffer_overflows += 1;
        }
        
        buffer.push_back(measurement);
        self.stats.lock().unwrap().total_metrics_collected += 1;
        
        Ok(())
    }

    /// Record a counter metric
    pub fn record_counter(&self, name: String, value: f64, labels: HashMap<String, String>) -> Result<(), String> {
        let measurement = MetricMeasurement {
            id: Uuid::new_v4().to_string(),
            name,
            value,
            labels,
            timestamp: self.current_timestamp(),
            metric_type: MetricType::Counter,
        };
        
        self.record_metric(measurement)
    }

    /// Record a gauge metric
    pub fn record_gauge(&self, name: String, value: f64, labels: HashMap<String, String>) -> Result<(), String> {
        let measurement = MetricMeasurement {
            id: Uuid::new_v4().to_string(),
            name,
            value,
            labels,
            timestamp: self.current_timestamp(),
            metric_type: MetricType::Gauge,
        };
        
        self.record_metric(measurement)
    }

    /// Record a histogram metric
    pub fn record_histogram(&self, name: String, value: f64, labels: HashMap<String, String>) -> Result<(), String> {
        let measurement = MetricMeasurement {
            id: Uuid::new_v4().to_string(),
            name,
            value,
            labels,
            timestamp: self.current_timestamp(),
            metric_type: MetricType::Histogram,
        };
        
        self.record_metric(measurement)
    }

    /// Start the collection loop
    async fn start_collection_loop(&self) {
        let mut interval = interval(self.config.collection_interval);
        let buffer = Arc::clone(&self.metrics_buffer);
        let stats = Arc::clone(&self.stats);
        
        loop {
            interval.tick().await;
            
            // Collect system metrics
            self.collect_system_metrics().await;
            
            // Collect application metrics
            self.collect_application_metrics().await;
            
            // Collect business metrics
            self.collect_business_metrics().await;
        }
    }

    /// Start the aggregation loop
    async fn start_aggregation_loop(&self) {
        let mut interval = interval(Duration::from_secs(60)); // Aggregate every minute
        let buffer = Arc::clone(&self.metrics_buffer);
        let aggregated = Arc::clone(&self.aggregated_metrics);
        
        loop {
            interval.tick().await;
            
            if self.config.enable_aggregation {
                self.aggregate_metrics().await;
            }
        }
    }

    /// Start the anomaly detection loop
    async fn start_anomaly_detection_loop(&self) {
        let mut interval = interval(Duration::from_secs(30));
        let aggregated = Arc::clone(&self.aggregated_metrics);
        let detector = Arc::clone(&self.anomaly_detector);
        let stats = Arc::clone(&self.stats);
        
        loop {
            interval.tick().await;
            
            if self.config.enable_anomaly_detection {
                self.detect_anomalies().await;
            }
        }
    }

    /// Start the export loop
    async fn start_export_loop(&self) {
        let mut interval = interval(Duration::from_secs(60)); // Export every minute
        let buffer = Arc::clone(&self.metrics_buffer);
        let exporters = &self.exporters;
        let stats = Arc::clone(&self.stats);
        
        loop {
            interval.tick().await;
            
            let metrics = {
                let buffer = buffer.lock().unwrap();
                buffer.iter().cloned().collect::<Vec<_>>()
            };
            
            if !metrics.is_empty() {
                for exporter in exporters {
                    match exporter.export_metrics(&metrics) {
                        Ok(_) => stats.lock().unwrap().export_operations += 1,
                        Err(_) => stats.lock().unwrap().export_failures += 1,
                    }
                }
            }
        }
    }

    /// Start the alert processing loop
    async fn start_alert_processing_loop(&self) {
        let mut interval = interval(Duration::from_secs(10));
        let alert_manager = Arc::clone(&self.alert_manager);
        let aggregated = Arc::clone(&self.aggregated_metrics);
        
        loop {
            interval.tick().await;
            
            if self.config.enable_alerting {
                self.process_alerts().await;
            }
        }
    }

    /// Collect system metrics
    async fn collect_system_metrics(&self) {
        // CPU usage
        if let Ok(cpu_usage) = self.get_cpu_usage().await {
            let mut labels = HashMap::new();
            labels.insert("host".to_string(), "localhost".to_string());
            self.record_gauge("cpu_usage_percent".to_string(), cpu_usage, labels).unwrap();
        }
        
        // Memory usage
        if let Ok(memory_usage) = self.get_memory_usage().await {
            let mut labels = HashMap::new();
            labels.insert("host".to_string(), "localhost".to_string());
            self.record_gauge("memory_usage_percent".to_string(), memory_usage, labels).unwrap();
        }
        
        // Disk usage
        if let Ok(disk_usage) = self.get_disk_usage().await {
            let mut labels = HashMap::new();
            labels.insert("host".to_string(), "localhost".to_string());
            self.record_gauge("disk_usage_percent".to_string(), disk_usage, labels).unwrap();
        }
        
        // Network I/O
        if let Ok((bytes_in, bytes_out)) = self.get_network_io().await {
            let mut labels = HashMap::new();
            labels.insert("host".to_string(), "localhost".to_string());
            labels.insert("direction".to_string(), "in".to_string());
            self.record_counter("network_bytes".to_string(), bytes_in, labels.clone()).unwrap();
            
            labels.insert("direction".to_string(), "out".to_string());
            self.record_counter("network_bytes".to_string(), bytes_out, labels).unwrap();
        }
    }

    /// Collect application metrics
    async fn collect_application_metrics(&self) {
        // Search requests per second
        let mut labels = HashMap::new();
        labels.insert("service".to_string(), "search".to_string());
        self.record_counter("requests_per_second".to_string(), 100.0, labels).unwrap();
        
        // Response times
        let mut labels = HashMap::new();
        labels.insert("endpoint".to_string(), "/search".to_string());
        self.record_histogram("response_time_ms".to_string(), 150.0, labels).unwrap();
        
        // Error rates
        let mut labels = HashMap::new();
        labels.insert("error_type".to_string(), "timeout".to_string());
        self.record_counter("error_rate".to_string(), 0.01, labels).unwrap();
    }

    /// Collect business metrics
    async fn collect_business_metrics(&self) {
        // Total searches
        let mut labels = HashMap::new();
        labels.insert("type".to_string(), "total".to_string());
        self.record_counter("searches_total".to_string(), 1000.0, labels).unwrap();
        
        // Unique users
        let mut labels = HashMap::new();
        labels.insert("type".to_string(), "unique_users".to_string());
        self.record_gauge("users_active".to_string(), 500.0, labels).unwrap();
        
        // Popular queries
        let mut labels = HashMap::new();
        labels.insert("query".to_string(), "machine learning".to_string());
        self.record_counter("popular_queries".to_string(), 50.0, labels).unwrap();
    }

    /// Aggregate metrics
    async fn aggregate_metrics(&self) {
        let buffer = self.metrics_buffer.lock().unwrap();
        let mut aggregated = self.aggregated_metrics.write().unwrap();
        
        // Group metrics by name
        let mut grouped_metrics: HashMap<String, Vec<&MetricMeasurement>> = HashMap::new();
        for metric in buffer.iter() {
            grouped_metrics.entry(metric.name.clone()).or_insert_with(Vec::new).push(metric);
        }
        
        // Calculate aggregations for each metric
        for (name, metrics) in grouped_metrics {
            if metrics.is_empty() {
                continue;
            }
            
            let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
            let count = values.len() as u64;
            let sum: f64 = values.iter().sum();
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = sum / count as f64;
            
            // Calculate percentiles
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = self.calculate_percentile(&sorted_values, 0.5);
            let p95 = self.calculate_percentile(&sorted_values, 0.95);
            let p99 = self.calculate_percentile(&sorted_values, 0.99);
            
            let aggregated_metric = AggregatedMetric {
                name: name.clone(),
                count,
                sum,
                min,
                max,
                avg,
                p50,
                p95,
                p99,
                last_updated: self.current_timestamp(),
            };
            
            aggregated.insert(name, aggregated_metric);
        }
    }

    /// Detect anomalies in metrics
    async fn detect_anomalies(&self) {
        let aggregated = self.aggregated_metrics.read().unwrap();
        let mut detector = self.anomaly_detector.lock().unwrap();
        
        for (name, metric) in aggregated.iter() {
            if let Some(model) = detector.models.get_mut(name) {
                if model.is_anomaly(metric.avg) {
                    // Generate alert
                    self.generate_alert(name.clone(), metric.avg, "Anomaly detected").await;
                    model.anomaly_count += 1;
                }
                
                // Update model
                model.update(metric.avg);
            } else {
                // Create new model
                detector.models.insert(name.clone(), AnomalyModel::new(name.clone()));
            }
        }
    }

    /// Process alerts
    async fn process_alerts(&self) {
        let aggregated = self.aggregated_metrics.read().unwrap();
        let mut alert_manager = self.alert_manager.lock().unwrap();
        
        for (name, metric) in aggregated.iter() {
            if let Some(threshold) = self.config.alert_thresholds.get(name) {
                if metric.avg > *threshold {
                    self.generate_alert(name.clone(), metric.avg, &format!("Threshold exceeded: {}", threshold)).await;
                }
            }
        }
    }

    /// Generate an alert
    async fn generate_alert(&self, metric_name: String, current_value: f64, message: String) {
        let alert = Alert {
            id: Uuid::new_v4().to_string(),
            metric_name,
            severity: AlertSeverity::Warning,
            message,
            threshold: 0.0,
            current_value,
            timestamp: self.current_timestamp(),
            status: AlertStatus::Active,
        };
        
        let mut alert_manager = self.alert_manager.lock().unwrap();
        alert_manager.add_alert(alert);
        self.stats.lock().unwrap().total_alerts_generated += 1;
    }

    /// Calculate percentile
    fn calculate_percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        
        let index = (percentile * (sorted_values.len() - 1) as f64) as usize;
        sorted_values[index]
    }

    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// System metric collection methods (simplified implementations)
    async fn get_cpu_usage(&self) -> Result<f64, String> {
        // In production, use proper system monitoring libraries
        Ok(45.0) // Simulated CPU usage
    }

    async fn get_memory_usage(&self) -> Result<f64, String> {
        // In production, use proper system monitoring libraries
        Ok(67.5) // Simulated memory usage
    }

    async fn get_disk_usage(&self) -> Result<f64, String> {
        // In production, use proper system monitoring libraries
        Ok(23.8) // Simulated disk usage
    }

    async fn get_network_io(&self) -> Result<(f64, f64), String> {
        // In production, use proper system monitoring libraries
        Ok((1024.0, 2048.0)) // Simulated network I/O
    }

    /// Get collector statistics
    pub fn get_stats(&self) -> CollectorStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(&self) -> HashMap<String, AggregatedMetric> {
        self.aggregated_metrics.read().unwrap().clone()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        let alert_manager = self.alert_manager.lock().unwrap();
        alert_manager.active_alerts.values().cloned().collect()
    }
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            threshold: 2.0, // 2 standard deviations
            window_size: 100,
        }
    }
}

impl AnomalyModel {
    fn new(name: String) -> Self {
        Self {
            name,
            historical_values: VecDeque::with_capacity(100),
            mean: 0.0,
            std_dev: 0.0,
            anomaly_count: 0,
        }
    }

    fn is_anomaly(&self, value: f64) -> bool {
        if self.historical_values.len() < 10 {
            return false; // Not enough data
        }
        
        let z_score = (value - self.mean) / self.std_dev;
        z_score.abs() > 2.0
    }

    fn update(&mut self, value: f64) {
        self.historical_values.push_back(value);
        if self.historical_values.len() > self.historical_values.capacity() {
            self.historical_values.pop_front();
        }
        
        // Recalculate mean and standard deviation
        let count = self.historical_values.len() as f64;
        self.mean = self.historical_values.iter().sum::<f64>() / count;
        
        let variance = self.historical_values.iter()
            .map(|v| (v - self.mean).powi(2))
            .sum::<f64>() / count;
        self.std_dev = variance.sqrt();
    }
}

impl AlertManager {
    fn new() -> Self {
        Self {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::with_capacity(1000),
            notification_channels: Vec::new(),
        }
    }

    fn add_alert(&mut self, alert: Alert) {
        self.active_alerts.insert(alert.id.clone(), alert.clone());
        self.alert_history.push_back(alert);
        
        if self.alert_history.len() > self.alert_history.capacity() {
            self.alert_history.pop_front();
        }
    }
}

// Prometheus exporter implementation
struct PrometheusExporter {
    registry: Registry,
}

impl PrometheusExporter {
    fn new() -> Self {
        Self {
            registry: Registry::new(),
        }
    }
}

impl MetricExporter for PrometheusExporter {
    fn export_metrics(&self, metrics: &[MetricMeasurement]) -> Result<(), String> {
        // Convert metrics to Prometheus format
        let mut families = Vec::new();
        
        for metric in metrics {
            let family = self.convert_to_prometheus_metric(metric);
            families.push(family);
        }
        
        // Encode to Prometheus format
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&families, &mut buffer).map_err(|e| e.to_string())?;
        
        // In production, this would write to a file or send over HTTP
        println!("Prometheus metrics: {}", String::from_utf8_lossy(&buffer));
        
        Ok(())
    }

    fn get_name(&self) -> String {
        "prometheus".to_string()
    }
}

impl PrometheusExporter {
    fn convert_to_prometheus_metric(&self, metric: &MetricMeasurement) -> MetricFamily {
        // Simplified conversion - in production, use proper Prometheus types
        let mut family = MetricFamily::new();
        family.set_name(metric.name.clone());
        family.set_field_type(prometheus::proto::MetricType::COUNTER);
        
        let mut proto_metric = prometheus::proto::Metric::new();
        let mut counter = CounterProto::new();
        counter.set_value(metric.value);
        proto_metric.set_counter(counter);
        
        family.mut_metric().push(proto_metric);
        family
    }
}

// JSON exporter implementation
struct JSONExporter;

impl JSONExporter {
    fn new() -> Self {
        Self
    }
}

impl MetricExporter for JSONExporter {
    fn export_metrics(&self, metrics: &[MetricMeasurement]) -> Result<(), String> {
        let json_data = serde_json::to_string_pretty(metrics).map_err(|e| e.to_string())?;
        
        // In production, this would write to a file or send to a service
        println!("JSON metrics: {}", json_data);
        
        Ok(())
    }

    fn get_name(&self) -> String {
        "json".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector() {
        let config = MetricsCollectorConfig::default();
        let collector = MetricsCollector::new(config);
        
        // Record a test metric
        let mut labels = HashMap::new();
        labels.insert("test".to_string(), "value".to_string());
        collector.record_gauge("test_metric".to_string(), 42.0, labels).unwrap();
        
        // Check stats
        let stats = collector.get_stats();
        assert!(stats.total_metrics_collected > 0);
    }
}