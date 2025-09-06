// T3SS Project
// File: frontend/api_gateway/main.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
	"gopkg.in/yaml.v2"
)

// APIGatewayConfig holds configuration for the API gateway
type APIGatewayConfig struct {
	Server struct {
		Host         string        `yaml:"host"`
		Port         int           `yaml:"port"`
		ReadTimeout  time.Duration `yaml:"read_timeout"`
		WriteTimeout time.Duration `yaml:"write_timeout"`
		IdleTimeout  time.Duration `yaml:"idle_timeout"`
	} `yaml:"server"`
	
	RateLimit struct {
		Enabled          bool          `yaml:"enabled"`
		RequestsPerMinute int          `yaml:"requests_per_minute"`
		BurstSize        int           `yaml:"burst_size"`
		CleanupInterval  time.Duration `yaml:"cleanup_interval"`
	} `yaml:"rate_limit"`
	
	Caching struct {
		Enabled         bool          `yaml:"enabled"`
		RedisEndpoint   string        `yaml:"redis_endpoint"`
		DefaultTTL      time.Duration `yaml:"default_ttl"`
		MaxCacheSize    int64         `yaml:"max_cache_size"`
		CleanupInterval time.Duration `yaml:"cleanup_interval"`
	} `yaml:"caching"`
	
	Security struct {
		EnableHTTPS     bool   `yaml:"enable_https"`
		CertFile        string `yaml:"cert_file"`
		KeyFile         string `yaml:"key_file"`
		EnableCORS      bool   `yaml:"enable_cors"`
		AllowedOrigins  []string `yaml:"allowed_origins"`
		EnableAPIKey    bool   `yaml:"enable_api_key"`
		APIKeyHeader    string `yaml:"api_key_header"`
	} `yaml:"security"`
	
	Services struct {
		SearchServiceURL string `yaml:"search_service_url"`
		IndexServiceURL   string `yaml:"index_service_url"`
		RankingServiceURL string `yaml:"ranking_service_url"`
		Timeout          time.Duration `yaml:"timeout"`
	} `yaml:"services"`
	
	Monitoring struct {
		EnableMetrics   bool   `yaml:"enable_metrics"`
		MetricsPort     int    `yaml:"metrics_port"`
		EnableHealthCheck bool `yaml:"enable_health_check"`
		HealthCheckPath string `yaml:"health_check_path"`
	} `yaml:"monitoring"`
}

// SearchRequest represents a search API request
type SearchRequest struct {
	Query     string            `json:"query" binding:"required"`
	Limit     int               `json:"limit,omitempty"`
	Offset    int               `json:"offset,omitempty"`
	Filters   map[string]string `json:"filters,omitempty"`
	SortBy    string            `json:"sort_by,omitempty"`
	SortOrder string            `json:"sort_order,omitempty"`
	APIKey    string            `json:"api_key,omitempty"`
}

// SearchResponse represents a search API response
type SearchResponse struct {
	Query       string        `json:"query"`
	Results     []SearchResult `json:"results"`
	TotalCount  int64         `json:"total_count"`
	Page        int           `json:"page"`
	PageSize    int           `json:"page_size"`
	ProcessingTime time.Duration `json:"processing_time"`
	CacheHit    bool          `json:"cache_hit"`
	RequestID   string        `json:"request_id"`
}

// SearchResult represents a single search result
type SearchResult struct {
	ID          string            `json:"id"`
	Title       string            `json:"title"`
	URL         string            `json:"url"`
	Snippet     string            `json:"snippet"`
	Score       float64           `json:"score"`
	Metadata    map[string]string `json:"metadata"`
	Timestamp   time.Time         `json:"timestamp"`
}

// RateLimiter manages rate limiting for API requests
type RateLimiter struct {
	limiters map[string]*rate.Limiter
	mu       sync.RWMutex
	config   APIGatewayConfig
	cleanup  *time.Ticker
}

// CacheManager manages response caching
type CacheManager struct {
	redisClient *redis.Client
	config      APIGatewayConfig
	logger      *zap.Logger
}

// MetricsCollector collects API gateway metrics
type MetricsCollector struct {
	requestDuration    prometheus.HistogramVec
	requestCount       prometheus.CounterVec
	rateLimitHits      prometheus.CounterVec
	cacheHits          prometheus.CounterVec
	cacheMisses        prometheus.CounterVec
	activeConnections  prometheus.Gauge
	errorCount         prometheus.CounterVec
}

// APIGateway is the main API gateway struct
type APIGateway struct {
	config         APIGatewayConfig
	logger         *zap.Logger
	rateLimiter    *RateLimiter
	cacheManager   *CacheManager
	metrics        *MetricsCollector
	httpClient     *http.Client
	server         *http.Server
}

// NewAPIGateway creates a new API gateway instance
func NewAPIGateway(config APIGatewayConfig) (*APIGateway, error) {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}

	// Initialize rate limiter
	rateLimiter := NewRateLimiter(config)

	// Initialize cache manager
	cacheManager, err := NewCacheManager(config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache manager: %w", err)
	}

	// Initialize metrics collector
	metrics := NewMetricsCollector()

	// Initialize HTTP client
	httpClient := &http.Client{
		Timeout: config.Services.Timeout,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: false,
			},
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	return &APIGateway{
		config:       config,
		logger:       logger,
		rateLimiter:  rateLimiter,
		cacheManager: cacheManager,
		metrics:      metrics,
		httpClient:   httpClient,
	}, nil
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(config APIGatewayConfig) *RateLimiter {
	rl := &RateLimiter{
		limiters: make(map[string]*rate.Limiter),
		config:   config,
	}

	if config.RateLimit.Enabled {
		rl.cleanup = time.NewTicker(config.RateLimit.CleanupInterval)
		go rl.cleanupRoutine()
	}

	return rl
}

// NewCacheManager creates a new cache manager
func NewCacheManager(config APIGatewayConfig, logger *zap.Logger) (*CacheManager, error) {
	var redisClient *redis.Client
	
	if config.Caching.Enabled {
		redisClient = redis.NewClient(&redis.Options{
			Addr: config.Caching.RedisEndpoint,
		})
		
		// Test connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		if err := redisClient.Ping(ctx).Err(); err != nil {
			return nil, fmt.Errorf("failed to connect to Redis: %w", err)
		}
	}

	return &CacheManager{
		redisClient: redisClient,
		config:      config,
		logger:      logger,
	}, nil
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		requestDuration: *prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "api_gateway_request_duration_seconds",
				Help:    "Request duration in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method", "endpoint", "status_code"},
		),
		requestCount: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "api_gateway_requests_total",
				Help: "Total number of requests",
			},
			[]string{"method", "endpoint", "status_code"},
		),
		rateLimitHits: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "api_gateway_rate_limit_hits_total",
				Help: "Total number of rate limit hits",
			},
			[]string{"client_ip"},
		),
		cacheHits: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "api_gateway_cache_hits_total",
				Help: "Total number of cache hits",
			},
			[]string{"endpoint"},
		),
		cacheMisses: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "api_gateway_cache_misses_total",
				Help: "Total number of cache misses",
			},
			[]string{"endpoint"},
		),
		activeConnections: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "api_gateway_active_connections",
				Help: "Number of active connections",
			},
		),
		errorCount: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "api_gateway_errors_total",
				Help: "Total number of errors",
			},
			[]string{"error_type"},
		),
	}
}

// Start starts the API gateway server
func (gw *APIGateway) Start() error {
	// Register metrics
	prometheus.MustRegister(
		gw.metrics.requestDuration,
		gw.metrics.requestCount,
		gw.metrics.rateLimitHits,
		gw.metrics.cacheHits,
		gw.metrics.cacheMisses,
		gw.metrics.activeConnections,
		gw.metrics.errorCount,
	)

	// Setup Gin router
	router := gw.setupRouter()

	// Create HTTP server
	gw.server = &http.Server{
		Addr:         fmt.Sprintf("%s:%d", gw.config.Server.Host, gw.config.Server.Port),
		Handler:      router,
		ReadTimeout:  gw.config.Server.ReadTimeout,
		WriteTimeout: gw.config.Server.WriteTimeout,
		IdleTimeout:  gw.config.Server.IdleTimeout,
	}

	gw.logger.Info("Starting API Gateway server",
		zap.String("address", gw.server.Addr),
		zap.Duration("read_timeout", gw.config.Server.ReadTimeout),
		zap.Duration("write_timeout", gw.config.Server.WriteTimeout))

	// Start server
	if gw.config.Security.EnableHTTPS {
		return gw.server.ListenAndServeTLS(gw.config.Security.CertFile, gw.config.Security.KeyFile)
	}
	return gw.server.ListenAndServe()
}

// Stop gracefully stops the API gateway
func (gw *APIGateway) Stop(ctx context.Context) error {
	gw.logger.Info("Stopping API Gateway server")
	
	if gw.rateLimiter.cleanup != nil {
		gw.rateLimiter.cleanup.Stop()
	}
	
	return gw.server.Shutdown(ctx)
}

// setupRouter sets up the Gin router with middleware and routes
func (gw *APIGateway) setupRouter() *gin.Engine {
	router := gin.New()

	// Global middleware
	router.Use(gw.loggingMiddleware())
	router.Use(gw.recoveryMiddleware())
	router.Use(gw.metricsMiddleware())
	router.Use(gw.corsMiddleware())

	// Rate limiting middleware
	if gw.config.RateLimit.Enabled {
		router.Use(gw.rateLimitMiddleware())
	}

	// API key middleware
	if gw.config.Security.EnableAPIKey {
		router.Use(gw.apiKeyMiddleware())
	}

	// Health check endpoint
	if gw.config.Monitoring.EnableHealthCheck {
		router.GET(gw.config.Monitoring.HealthCheckPath, gw.healthCheckHandler())
	}

	// Metrics endpoint
	if gw.config.Monitoring.EnableMetrics {
		router.GET("/metrics", gin.WrapH(promhttp.Handler()))
	}

	// API routes
	api := router.Group("/api/v1")
	{
		api.POST("/search", gw.searchHandler())
		api.GET("/search", gw.searchHandler())
		api.GET("/suggest", gw.suggestHandler())
		api.GET("/autocomplete", gw.autocompleteHandler())
		api.GET("/trending", gw.trendingHandler())
		api.GET("/stats", gw.statsHandler())
	}

	return router
}

// searchHandler handles search requests
func (gw *APIGateway) searchHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		startTime := time.Now()
		requestID := generateRequestID()

		// Parse request
		var req SearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			gw.logger.Error("Failed to parse search request", zap.Error(err))
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
			return
		}

		// Set defaults
		if req.Limit <= 0 {
			req.Limit = 10
		}
		if req.Offset < 0 {
			req.Offset = 0
		}

		// Check cache first
		var response *SearchResponse
		var cacheHit bool
		
		if gw.config.Caching.Enabled {
			if cachedResp, err := gw.cacheManager.GetCachedResponse(&req); err == nil && cachedResp != nil {
				response = cachedResp
				cacheHit = true
				gw.metrics.cacheHits.WithLabelValues("search").Inc()
			} else {
				gw.metrics.cacheMisses.WithLabelValues("search").Inc()
			}
		}

		// If not cached, make request to search service
		if response == nil {
			var err error
			response, err = gw.forwardSearchRequest(&req, requestID)
			if err != nil {
				gw.logger.Error("Search request failed", zap.Error(err))
				gw.metrics.errorCount.WithLabelValues("search_service_error").Inc()
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Search service unavailable"})
				return
			}

			// Cache the response
			if gw.config.Caching.Enabled {
				gw.cacheManager.CacheResponse(&req, response)
			}
		}

		// Set response metadata
		response.ProcessingTime = time.Since(startTime)
		response.CacheHit = cacheHit
		response.RequestID = requestID

		// Record metrics
		gw.metrics.requestCount.WithLabelValues("POST", "/search", "200").Inc()
		gw.metrics.requestDuration.WithLabelValues("POST", "/search", "200").Observe(time.Since(startTime).Seconds())

		c.JSON(http.StatusOK, response)
	}
}

// suggestHandler handles search suggestions
func (gw *APIGateway) suggestHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		query := c.Query("q")
		if query == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
			return
		}

		// Forward to suggestion service
		suggestions, err := gw.forwardSuggestionRequest(query)
		if err != nil {
			gw.logger.Error("Suggestion request failed", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Suggestion service unavailable"})
			return
		}

		c.JSON(http.StatusOK, gin.H{"suggestions": suggestions})
	}
}

// autocompleteHandler handles autocomplete requests
func (gw *APIGateway) autocompleteHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		query := c.Query("q")
		if query == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
			return
		}

		// Forward to autocomplete service
		completions, err := gw.forwardAutocompleteRequest(query)
		if err != nil {
			gw.logger.Error("Autocomplete request failed", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Autocomplete service unavailable"})
			return
		}

		c.JSON(http.StatusOK, gin.H{"completions": completions})
	}
}

// trendingHandler handles trending queries
func (gw *APIGateway) trendingHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Forward to trending service
		trending, err := gw.forwardTrendingRequest()
		if err != nil {
			gw.logger.Error("Trending request failed", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Trending service unavailable"})
			return
		}

		c.JSON(http.StatusOK, gin.H{"trending": trending})
	}
}

// statsHandler handles statistics requests
func (gw *APIGateway) statsHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		stats := gin.H{
			"rate_limiter": gw.rateLimiter.getStats(),
			"cache":        gw.cacheManager.getStats(),
			"gateway":      gw.getGatewayStats(),
		}

		c.JSON(http.StatusOK, stats)
	}
}

// healthCheckHandler handles health check requests
func (gw *APIGateway) healthCheckHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		health := gin.H{
			"status":    "healthy",
			"timestamp": time.Now().Unix(),
			"version":   "1.0.0",
		}

		c.JSON(http.StatusOK, health)
	}
}

// forwardSearchRequest forwards search request to search service
func (gw *APIGateway) forwardSearchRequest(req *SearchRequest, requestID string) (*SearchResponse, error) {
	// Create request body
	requestBody := map[string]interface{}{
		"query":      req.Query,
		"limit":      req.Limit,
		"offset":     req.Offset,
		"filters":    req.Filters,
		"sort_by":    req.SortBy,
		"sort_order": req.SortOrder,
		"request_id": requestID,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make HTTP request
	httpReq, err := http.NewRequest("POST", gw.config.Services.SearchServiceURL+"/search", strings.NewReader(string(jsonBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Request-ID", requestID)

	resp, err := gw.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search service returned status %d", resp.StatusCode)
	}

	// Parse response
	var searchResp SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &searchResp, nil
}

// forwardSuggestionRequest forwards suggestion request
func (gw *APIGateway) forwardSuggestionRequest(query string) ([]string, error) {
	// Implementation would make HTTP request to suggestion service
	return []string{"suggestion1", "suggestion2", "suggestion3"}, nil
}

// forwardAutocompleteRequest forwards autocomplete request
func (gw *APIGateway) forwardAutocompleteRequest(query string) ([]string, error) {
	// Implementation would make HTTP request to autocomplete service
	return []string{"completion1", "completion2", "completion3"}, nil
}

// forwardTrendingRequest forwards trending request
func (gw *APIGateway) forwardTrendingRequest() ([]string, error) {
	// Implementation would make HTTP request to trending service
	return []string{"trending1", "trending2", "trending3"}, nil
}

// Middleware functions
func (gw *APIGateway) loggingMiddleware() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		gw.logger.Info("HTTP Request",
			zap.String("method", param.Method),
			zap.String("path", param.Path),
			zap.Int("status", param.StatusCode),
			zap.Duration("latency", param.Latency),
			zap.String("client_ip", param.ClientIP),
		)
		return ""
	})
}

func (gw *APIGateway) recoveryMiddleware() gin.HandlerFunc {
	return gin.Recovery()
}

func (gw *APIGateway) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		
		duration := time.Since(start)
		status := strconv.Itoa(c.Writer.Status())
		
		gw.metrics.requestDuration.WithLabelValues(c.Request.Method, c.FullPath(), status).Observe(duration.Seconds())
		gw.metrics.requestCount.WithLabelValues(c.Request.Method, c.FullPath(), status).Inc()
	}
}

func (gw *APIGateway) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if gw.config.Security.EnableCORS {
			origin := c.Request.Header.Get("Origin")
			if gw.isAllowedOrigin(origin) {
				c.Header("Access-Control-Allow-Origin", origin)
			}
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
			c.Header("Access-Control-Allow-Credentials", "true")
		}
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		
		c.Next()
	}
}

func (gw *APIGateway) rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		clientIP := c.ClientIP()
		
		if !gw.rateLimiter.Allow(clientIP) {
			gw.metrics.rateLimitHits.WithLabelValues(clientIP).Inc()
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "Rate limit exceeded"})
			c.Abort()
			return
		}
		
		c.Next()
	}
}

func (gw *APIGateway) apiKeyMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		apiKey := c.GetHeader(gw.config.Security.APIKeyHeader)
		if apiKey == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "API key required"})
			c.Abort()
			return
		}
		
		// Validate API key (implementation would check against database)
		if !gw.validateAPIKey(apiKey) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid API key"})
			c.Abort()
			return
		}
		
		c.Next()
	}
}

// Helper functions
func (gw *APIGateway) isAllowedOrigin(origin string) bool {
	for _, allowedOrigin := range gw.config.Security.AllowedOrigins {
		if origin == allowedOrigin {
			return true
		}
	}
	return false
}

func (gw *APIGateway) validateAPIKey(apiKey string) bool {
	// Implementation would validate against database
	return len(apiKey) > 0
}

func (gw *APIGateway) getGatewayStats() gin.H {
	return gin.H{
		"uptime":      time.Since(time.Now()).String(),
		"version":     "1.0.0",
		"config":      gw.config,
	}
}

func generateRequestID() string {
	return fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), os.Getpid())
}

// RateLimiter methods
func (rl *RateLimiter) Allow(clientIP string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	
	limiter, exists := rl.limiters[clientIP]
	if !exists {
		limiter = rate.NewLimiter(
			rate.Limit(rl.config.RateLimit.RequestsPerMinute)/60,
			rl.config.RateLimit.BurstSize,
		)
		rl.limiters[clientIP] = limiter
	}
	
	return limiter.Allow()
}

func (rl *RateLimiter) cleanupRoutine() {
	for range rl.cleanup.C {
		rl.mu.Lock()
		for clientIP, limiter := range rl.limiters {
			if !limiter.Allow() {
				delete(rl.limiters, clientIP)
			}
		}
		rl.mu.Unlock()
	}
}

func (rl *RateLimiter) getStats() gin.H {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	
	return gin.H{
		"active_clients": len(rl.limiters),
		"requests_per_minute": rl.config.RateLimit.RequestsPerMinute,
		"burst_size": rl.config.RateLimit.BurstSize,
	}
}

// CacheManager methods
func (cm *CacheManager) GetCachedResponse(req *SearchRequest) (*SearchResponse, error) {
	if cm.redisClient == nil {
		return nil, fmt.Errorf("Redis not configured")
	}
	
	cacheKey := cm.generateCacheKey(req)
	ctx := context.Background()
	
	data, err := cm.redisClient.Get(ctx, cacheKey).Result()
	if err != nil {
		return nil, err
	}
	
	var response SearchResponse
	if err := json.Unmarshal([]byte(data), &response); err != nil {
		return nil, err
	}
	
	return &response, nil
}

func (cm *CacheManager) CacheResponse(req *SearchRequest, response *SearchResponse) {
	if cm.redisClient == nil {
		return
	}
	
	cacheKey := cm.generateCacheKey(req)
	ctx := context.Background()
	
	data, err := json.Marshal(response)
	if err != nil {
		cm.logger.Error("Failed to marshal response for caching", zap.Error(err))
		return
	}
	
	cm.redisClient.Set(ctx, cacheKey, data, cm.config.Caching.DefaultTTL)
}

func (cm *CacheManager) generateCacheKey(req *SearchRequest) string {
	keyData := fmt.Sprintf("%s:%d:%d:%v", req.Query, req.Limit, req.Offset, req.Filters)
	return fmt.Sprintf("search:%x", []byte(keyData))
}

func (cm *CacheManager) getStats() gin.H {
	if cm.redisClient == nil {
		return gin.H{"enabled": false}
	}
	
	ctx := context.Background()
	info, err := cm.redisClient.Info(ctx, "memory").Result()
	if err != nil {
		return gin.H{"enabled": true, "error": err.Error()}
	}
	
	return gin.H{
		"enabled": true,
		"info":    info,
	}
}

// Main function
func main() {
	// Load configuration
	configFile := os.Getenv("CONFIG_FILE")
	if configFile == "" {
		configFile = "config.yaml"
	}
	
	configData, err := os.ReadFile(configFile)
	if err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}
	
	var config APIGatewayConfig
	if err := yaml.Unmarshal(configData, &config); err != nil {
		log.Fatalf("Failed to parse config file: %v", err)
	}
	
	// Create API gateway
	gateway, err := NewAPIGateway(config)
	if err != nil {
		log.Fatalf("Failed to create API gateway: %v", err)
	}
	
	// Start server
	if err := gateway.Start(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Failed to start API gateway: %v", err)
	}
}