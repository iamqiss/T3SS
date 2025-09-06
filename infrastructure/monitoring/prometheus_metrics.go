// T3SS Project
// File: infrastructure/monitoring/prometheus_metrics.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package monitoring

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

// MetricsCollector handles Prometheus metrics collection
type MetricsCollector struct {
	registry *prometheus.Registry
	logger   *zap.Logger
	mu       sync.RWMutex
	
	// Search metrics
	searchRequests    *prometheus.CounterVec
	searchDuration    *prometheus.HistogramVec
	searchResults     *prometheus.HistogramVec
	searchErrors      *prometheus.CounterVec
	
	// Crawler metrics
	crawlRequests     *prometheus.CounterVec
	crawlDuration     *prometheus.HistogramVec
	crawlErrors       *prometheus.CounterVec
	pagesCrawled      *prometheus.CounterVec
	
	// Indexing metrics
	indexingDuration  *prometheus.HistogramVec
	indexingErrors    *prometheus.CounterVec
	documentsIndexed  *prometheus.CounterVec
	
	// System metrics
	cpuUsage          prometheus.Gauge
	memoryUsage       prometheus.Gauge
	diskUsage         prometheus.Gauge
	networkIO         *prometheus.CounterVec
	
	// Cache metrics
	cacheHits         *prometheus.CounterVec
	cacheMisses       *prometheus.CounterVec
	cacheSize         *prometheus.GaugeVec
	
	// Business metrics
	activeUsers       prometheus.Gauge
	queriesPerSecond  prometheus.Gauge
	responseTime      *prometheus.HistogramVec
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(logger *zap.Logger) *MetricsCollector {
	registry := prometheus.NewRegistry()
	
	collector := &MetricsCollector{
		registry: registry,
		logger:   logger,
		
		// Search metrics
		searchRequests: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_search_requests_total",
				Help: "Total number of search requests",
			},
			[]string{"query_type", "user_type", "status"},
		),
		
		searchDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "t3ss_search_duration_seconds",
				Help:    "Search request duration in seconds",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"query_type", "index_type"},
		),
		
		searchResults: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "t3ss_search_results_count",
				Help:    "Number of search results returned",
				Buckets: prometheus.ExponentialBuckets(1, 2, 12),
			},
			[]string{"query_type"},
		),
		
		searchErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_search_errors_total",
				Help: "Total number of search errors",
			},
			[]string{"error_type", "component"},
		),
		
		// Crawler metrics
		crawlRequests: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_crawl_requests_total",
				Help: "Total number of crawl requests",
			},
			[]string{"domain", "status"},
		),
		
		crawlDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "t3ss_crawl_duration_seconds",
				Help:    "Crawl request duration in seconds",
				Buckets: prometheus.ExponentialBuckets(0.1, 2, 12),
			},
			[]string{"domain", "content_type"},
		),
		
		crawlErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_crawl_errors_total",
				Help: "Total number of crawl errors",
			},
			[]string{"error_type", "domain"},
		),
		
		pagesCrawled: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_pages_crawled_total",
				Help: "Total number of pages crawled",
			},
			[]string{"domain", "content_type"},
		),
		
		// Indexing metrics
		indexingDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "t3ss_indexing_duration_seconds",
				Help:    "Document indexing duration in seconds",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"document_type", "index_type"},
		),
		
		indexingErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_indexing_errors_total",
				Help: "Total number of indexing errors",
			},
			[]string{"error_type", "document_type"},
		),
		
		documentsIndexed: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_documents_indexed_total",
				Help: "Total number of documents indexed",
			},
			[]string{"document_type", "source"},
		),
		
		// System metrics
		cpuUsage: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "t3ss_cpu_usage_percent",
				Help: "CPU usage percentage",
			},
		),
		
		memoryUsage: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "t3ss_memory_usage_percent",
				Help: "Memory usage percentage",
			},
		),
		
		diskUsage: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "t3ss_disk_usage_percent",
				Help: "Disk usage percentage",
			},
		),
		
		networkIO: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_network_io_bytes_total",
				Help: "Total network I/O in bytes",
			},
			[]string{"direction", "interface"},
		),
		
		// Cache metrics
		cacheHits: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_cache_hits_total",
				Help: "Total number of cache hits",
			},
			[]string{"cache_type", "key_type"},
		),
		
		cacheMisses: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "t3ss_cache_misses_total",
				Help: "Total number of cache misses",
			},
			[]string{"cache_type", "key_type"},
		),
		
		cacheSize: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "t3ss_cache_size_bytes",
				Help: "Cache size in bytes",
			},
			[]string{"cache_type"},
		),
		
		// Business metrics
		activeUsers: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "t3ss_active_users",
				Help: "Number of active users",
			},
		),
		
		queriesPerSecond: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "t3ss_queries_per_second",
				Help: "Queries per second",
			},
		),
		
		responseTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "t3ss_response_time_seconds",
				Help:    "Response time in seconds",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"endpoint", "method", "status_code"},
		),
	}
	
	// Register all metrics
	collector.registerMetrics()
	
	return collector
}

// registerMetrics registers all metrics with the registry
func (mc *MetricsCollector) registerMetrics() {
	mc.registry.MustRegister(
		// Search metrics
		mc.searchRequests,
		mc.searchDuration,
		mc.searchResults,
		mc.searchErrors,
		
		// Crawler metrics
		mc.crawlRequests,
		mc.crawlDuration,
		mc.crawlErrors,
		mc.pagesCrawled,
		
		// Indexing metrics
		mc.indexingDuration,
		mc.indexingErrors,
		mc.documentsIndexed,
		
		// System metrics
		mc.cpuUsage,
		mc.memoryUsage,
		mc.diskUsage,
		mc.networkIO,
		
		// Cache metrics
		mc.cacheHits,
		mc.cacheMisses,
		mc.cacheSize,
		
		// Business metrics
		mc.activeUsers,
		mc.queriesPerSecond,
		mc.responseTime,
	)
}

// RecordSearchRequest records a search request
func (mc *MetricsCollector) RecordSearchRequest(queryType, userType, status string) {
	mc.searchRequests.WithLabelValues(queryType, userType, status).Inc()
}

// RecordSearchDuration records search duration
func (mc *MetricsCollector) RecordSearchDuration(queryType, indexType string, duration time.Duration) {
	mc.searchDuration.WithLabelValues(queryType, indexType).Observe(duration.Seconds())
}

// RecordSearchResults records number of search results
func (mc *MetricsCollector) RecordSearchResults(queryType string, count int) {
	mc.searchResults.WithLabelValues(queryType).Observe(float64(count))
}

// RecordSearchError records a search error
func (mc *MetricsCollector) RecordSearchError(errorType, component string) {
	mc.searchErrors.WithLabelValues(errorType, component).Inc()
}

// RecordCrawlRequest records a crawl request
func (mc *MetricsCollector) RecordCrawlRequest(domain, status string) {
	mc.crawlRequests.WithLabelValues(domain, status).Inc()
}

// RecordCrawlDuration records crawl duration
func (mc *MetricsCollector) RecordCrawlDuration(domain, contentType string, duration time.Duration) {
	mc.crawlDuration.WithLabelValues(domain, contentType).Observe(duration.Seconds())
}

// RecordCrawlError records a crawl error
func (mc *MetricsCollector) RecordCrawlError(errorType, domain string) {
	mc.crawlErrors.WithLabelValues(errorType, domain).Inc()
}

// RecordPageCrawled records a page being crawled
func (mc *MetricsCollector) RecordPageCrawled(domain, contentType string) {
	mc.pagesCrawled.WithLabelValues(domain, contentType).Inc()
}

// RecordIndexingDuration records indexing duration
func (mc *MetricsCollector) RecordIndexingDuration(documentType, indexType string, duration time.Duration) {
	mc.indexingDuration.WithLabelValues(documentType, indexType).Observe(duration.Seconds())
}

// RecordIndexingError records an indexing error
func (mc *MetricsCollector) RecordIndexingError(errorType, documentType string) {
	mc.indexingErrors.WithLabelValues(errorType, documentType).Inc()
}

// RecordDocumentIndexed records a document being indexed
func (mc *MetricsCollector) RecordDocumentIndexed(documentType, source string) {
	mc.documentsIndexed.WithLabelValues(documentType, source).Inc()
}

// UpdateSystemMetrics updates system metrics
func (mc *MetricsCollector) UpdateSystemMetrics(cpu, memory, disk float64) {
	mc.cpuUsage.Set(cpu)
	mc.memoryUsage.Set(memory)
	mc.diskUsage.Set(disk)
}

// RecordNetworkIO records network I/O
func (mc *MetricsCollector) RecordNetworkIO(direction, interface string, bytes int64) {
	mc.networkIO.WithLabelValues(direction, interface).Add(float64(bytes))
}

// RecordCacheHit records a cache hit
func (mc *MetricsCollector) RecordCacheHit(cacheType, keyType string) {
	mc.cacheHits.WithLabelValues(cacheType, keyType).Inc()
}

// RecordCacheMiss records a cache miss
func (mc *MetricsCollector) RecordCacheMiss(cacheType, keyType string) {
	mc.cacheMisses.WithLabelValues(cacheType, keyType).Inc()
}

// UpdateCacheSize updates cache size
func (mc *MetricsCollector) UpdateCacheSize(cacheType string, size int64) {
	mc.cacheSize.WithLabelValues(cacheType).Set(float64(size))
}

// UpdateActiveUsers updates active users count
func (mc *MetricsCollector) UpdateActiveUsers(count int) {
	mc.activeUsers.Set(float64(count))
}

// UpdateQueriesPerSecond updates queries per second
func (mc *MetricsCollector) UpdateQueriesPerSecond(qps float64) {
	mc.queriesPerSecond.Set(qps)
}

// RecordResponseTime records response time
func (mc *MetricsCollector) RecordResponseTime(endpoint, method, statusCode string, duration time.Duration) {
	mc.responseTime.WithLabelValues(endpoint, method, statusCode).Observe(duration.Seconds())
}

// GetHandler returns HTTP handler for Prometheus metrics
func (mc *MetricsCollector) GetHandler() http.Handler {
	return promhttp.HandlerFor(mc.registry, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	})
}

// StartServer starts the metrics server
func (mc *MetricsCollector) StartServer(port int) error {
	http.Handle("/metrics", mc.GetHandler())
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: http.DefaultServeMux,
	}
	
	mc.logger.Info("Starting metrics server", zap.Int("port", port))
	
	return server.ListenAndServe()
}

// GetRegistry returns the Prometheus registry
func (mc *MetricsCollector) GetRegistry() *prometheus.Registry {
	return mc.registry
}

// CustomCollector for custom metrics
type CustomCollector struct {
	metrics map[string]prometheus.Metric
	mu      sync.RWMutex
}

// NewCustomCollector creates a new custom collector
func NewCustomCollector() *CustomCollector {
	return &CustomCollector{
		metrics: make(map[string]prometheus.Metric),
	}
}

// Describe implements prometheus.Collector
func (cc *CustomCollector) Describe(ch chan<- *prometheus.Desc) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	for _, metric := range cc.metrics {
		ch <- metric.Desc()
	}
}

// Collect implements prometheus.Collector
func (cc *CustomCollector) Collect(ch chan<- prometheus.Metric) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	for _, metric := range cc.metrics {
		ch <- metric
	}
}

// AddMetric adds a custom metric
func (cc *CustomCollector) AddMetric(name string, metric prometheus.Metric) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	
	cc.metrics[name] = metric
}

// RemoveMetric removes a custom metric
func (cc *CustomCollector) RemoveMetric(name string) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	
	delete(cc.metrics, name)
}