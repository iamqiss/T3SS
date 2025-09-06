// T3SS Project
// File: infrastructure/monitoring/metrics_collector.rs
// (c) 2025 Qiss Labs. All Rights Reserved.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use prometheus::{Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec, Registry};
use prometheus::Encoder;
use std::thread;
use std::sync::atomic::{AtomicU64, Ordering};

/// System metrics collector for comprehensive monitoring
pub struct MetricsCollector {
    registry: Registry,
    metrics: Arc<Mutex<SystemMetrics>>,
    exporters: Vec<Box<dyn MetricsExporter>>,
    collection_interval: Duration,
    running: Arc<AtomicU64>,
}

/// System metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_io: NetworkMetrics,
    pub process_metrics: ProcessMetrics,
    pub application_metrics: ApplicationMetrics,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub connection_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub thread_count: u32,
    pub file_descriptors: u32,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetrics {
    pub requests_total: u64,
    pub requests_per_second: f64,
    pub average_response_time: f64,
    pub error_rate: f64,
    pub active_connections: u32,
    pub cache_hit_rate: f64,
    pub queue_depth: u32,
}

/// Metrics exporter trait
pub trait MetricsExporter: Send + Sync {
    fn export(&self, metrics: &SystemMetrics) -> Result<(), String>;
    fn name(&self) -> &str;
}

/// Prometheus exporter
pub struct PrometheusExporter {
    registry: Registry,
    counters: HashMap<String, Counter>,
    gauges: HashMap<String, Gauge>,
    histograms: HashMap<String, Histogram>,
}

/// InfluxDB exporter
pub struct InfluxDBExporter {
    url: String,
    database: String,
    username: String,
    password: String,
}

/// JSON file exporter
pub struct JSONFileExporter {
    file_path: String,
}

impl MetricsCollector {
    pub fn new(collection_interval: Duration) -> Self {
        let registry = Registry::new();
        let metrics = Arc::new(Mutex::new(SystemMetrics::default()));
        let exporters = Vec::new();
        let running = Arc::new(AtomicU64::new(0));

        Self {
            registry,
            metrics,
            exporters,
            collection_interval,
            running,
        }
    }

    pub fn add_exporter(&mut self, exporter: Box<dyn MetricsExporter>) {
        self.exporters.push(exporter);
    }

    pub async fn start(&self) -> Result<(), String> {
        self.running.store(1, Ordering::SeqCst);
        
        let metrics = Arc::clone(&self.metrics);
        let exporters = self.exporters.clone();
        let interval = self.collection_interval;
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) == 1 {
                let system_metrics = Self::collect_system_metrics().await;
                
                {
                    let mut metrics_guard = metrics.lock().unwrap();
                    *metrics_guard = system_metrics;
                }

                // Export metrics
                for exporter in &exporters {
                    if let Err(e) = exporter.export(&system_metrics) {
                        eprintln!("Export failed for {}: {}", exporter.name(), e);
                    }
                }

                tokio::time::sleep(interval).await;
            }
        });

        Ok(())
    }

    pub fn stop(&self) {
        self.running.store(0, Ordering::SeqCst);
    }

    async fn collect_system_metrics() -> SystemMetrics {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        SystemMetrics {
            timestamp,
            cpu_usage: Self::get_cpu_usage(),
            memory_usage: Self::get_memory_usage(),
            disk_usage: Self::get_disk_usage(),
            network_io: Self::get_network_metrics(),
            process_metrics: Self::get_process_metrics(),
            application_metrics: Self::get_application_metrics(),
            custom_metrics: HashMap::new(),
        }
    }

    fn get_cpu_usage() -> f64 {
        // Simplified CPU usage calculation
        25.0 // Placeholder
    }

    fn get_memory_usage() -> f64 {
        // Simplified memory usage calculation
        60.0 // Placeholder
    }

    fn get_disk_usage() -> f64 {
        // Simplified disk usage calculation
        45.0 // Placeholder
    }

    fn get_network_metrics() -> NetworkMetrics {
        NetworkMetrics {
            bytes_sent: 1024000,
            bytes_received: 2048000,
            packets_sent: 1000,
            packets_received: 2000,
            connection_count: 50,
        }
    }

    fn get_process_metrics() -> ProcessMetrics {
        ProcessMetrics {
            cpu_percent: 15.0,
            memory_mb: 512.0,
            thread_count: 20,
            file_descriptors: 100,
            uptime_seconds: 3600,
        }
    }

    fn get_application_metrics() -> ApplicationMetrics {
        ApplicationMetrics {
            requests_total: 10000,
            requests_per_second: 100.0,
            average_response_time: 50.0,
            error_rate: 0.01,
            active_connections: 25,
            cache_hit_rate: 0.85,
            queue_depth: 5,
        }
    }

    pub fn get_current_metrics(&self) -> SystemMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: 0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_io: NetworkMetrics::default(),
            process_metrics: ProcessMetrics::default(),
            application_metrics: ApplicationMetrics::default(),
            custom_metrics: HashMap::new(),
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            connection_count: 0,
        }
    }
}

impl Default for ProcessMetrics {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_mb: 0.0,
            thread_count: 0,
            file_descriptors: 0,
            uptime_seconds: 0,
        }
    }
}

impl Default for ApplicationMetrics {
    fn default() -> Self {
        Self {
            requests_total: 0,
            requests_per_second: 0.0,
            average_response_time: 0.0,
            error_rate: 0.0,
            active_connections: 0,
            cache_hit_rate: 0.0,
            queue_depth: 0,
        }
    }
}

impl PrometheusExporter {
    pub fn new() -> Self {
        let registry = Registry::new();
        let mut counters = HashMap::new();
        let mut gauges = HashMap::new();
        let mut histograms = HashMap::new();

        // Initialize Prometheus metrics
        let cpu_gauge = Gauge::new("system_cpu_usage_percent", "CPU usage percentage").unwrap();
        let memory_gauge = Gauge::new("system_memory_usage_percent", "Memory usage percentage").unwrap();
        
        gauges.insert("cpu_usage".to_string(), cpu_gauge);
        gauges.insert("memory_usage".to_string(), memory_gauge);

        Self {
            registry,
            counters,
            gauges,
            histograms,
        }
    }
}

impl MetricsExporter for PrometheusExporter {
    fn export(&self, metrics: &SystemMetrics) -> Result<(), String> {
        // Update Prometheus metrics
        if let Some(gauge) = self.gauges.get("cpu_usage") {
            gauge.set(metrics.cpu_usage);
        }
        if let Some(gauge) = self.gauges.get("memory_usage") {
            gauge.set(metrics.memory_usage);
        }

        // Encode metrics
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();

        println!("Prometheus metrics exported: {} bytes", buffer.len());
        Ok(())
    }

    fn name(&self) -> &str {
        "prometheus"
    }
}

impl InfluxDBExporter {
    pub fn new(url: String, database: String, username: String, password: String) -> Self {
        Self {
            url,
            database,
            username,
            password,
        }
    }
}

impl MetricsExporter for InfluxDBExporter {
    fn export(&self, metrics: &SystemMetrics) -> Result<(), String> {
        // In production, send HTTP POST to InfluxDB
        println!("InfluxDB export: CPU={}%, Memory={}%", 
                metrics.cpu_usage, metrics.memory_usage);
        Ok(())
    }

    fn name(&self) -> &str {
        "influxdb"
    }
}

impl JSONFileExporter {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }
}

impl MetricsExporter for JSONFileExporter {
    fn export(&self, metrics: &SystemMetrics) -> Result<(), String> {
        let json = serde_json::to_string_pretty(metrics)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;
        
        std::fs::write(&self.file_path, json)
            .map_err(|e| format!("File write failed: {}", e))?;
        
        println!("Metrics exported to {}", self.file_path);
        Ok(())
    }

    fn name(&self) -> &str {
        "json_file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector() {
        let mut collector = MetricsCollector::new(Duration::from_secs(1));
        
        // Add exporters
        collector.add_exporter(Box::new(JSONFileExporter::new("test_metrics.json".to_string())));
        
        // Start collection
        collector.start().await.unwrap();
        
        // Wait a bit
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Get metrics
        let metrics = collector.get_current_metrics();
        assert!(metrics.cpu_usage >= 0.0);
        assert!(metrics.memory_usage >= 0.0);
        
        // Stop collection
        collector.stop();
    }
}