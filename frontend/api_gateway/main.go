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
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
	"gopkg.in/yaml.v2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/any"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/t3ss/shared_libs/proto/search"
	pbauth "github.com/t3ss/shared_libs/proto/auth"
	pbindexing "github.com/t3ss/shared_libs/proto/indexing"
	pbranking "github.com/t3ss/shared_libs/proto/ranking"
	pbml "github.com/t3ss/shared_libs/proto/ml"
)

// Configuration structure
type Config struct {
	Server   ServerConfig   `yaml:"server"`
	Redis    RedisConfig    `yaml:"redis"`
	Security SecurityConfig `yaml:"security"`
	RateLimit RateLimitConfig `yaml:"rate_limit"`
	Cache    CacheConfig    `yaml:"cache"`
	Services ServicesConfig `yaml:"services"`
	Metrics  MetricsConfig  `yaml:"metrics"`
}

type ServerConfig struct {
	Host         string `yaml:"host"`
	Port         int    `yaml:"port"`
	ReadTimeout  int    `yaml:"read_timeout"`
	WriteTimeout int    `yaml:"write_timeout"`
	IdleTimeout  int    `yaml:"idle_timeout"`
	EnableHTTPS  bool   `yaml:"enable_https"`
	CertFile     string `yaml:"cert_file"`
	KeyFile      string `yaml:"key_file"`
}

type RedisConfig struct {
	Host     string `yaml:"host"`
	Port     int    `yaml:"port"`
	Password string `yaml:"password"`
	DB       int    `yaml:"db"`
	PoolSize int    `yaml:"pool_size"`
}

type SecurityConfig struct {
	EnableCORS      bool     `yaml:"enable_cors"`
	AllowedOrigins  []string `yaml:"allowed_origins"`
	EnableAPIKeys   bool     `yaml:"enable_api_keys"`
	APIKeyHeader    string   `yaml:"api_key_header"`
	EnableJWT       bool     `yaml:"enable_jwt"`
	JWTSecret       string   `yaml:"jwt_secret"`
	EnableRateLimit bool     `yaml:"enable_rate_limit"`
}

type RateLimitConfig struct {
	RequestsPerMinute int `yaml:"requests_per_minute"`
	BurstSize         int `yaml:"burst_size"`
	EnablePerIP       bool `yaml:"enable_per_ip"`
	EnablePerAPIKey   bool `yaml:"enable_per_api_key"`
}

type CacheConfig struct {
	EnableRedis     bool          `yaml:"enable_redis"`
	DefaultTTL      time.Duration `yaml:"default_ttl"`
	MaxCacheSize    int64         `yaml:"max_cache_size"`
	EnableCompression bool        `yaml:"enable_compression"`
}

type ServicesConfig struct {
	SearchService    string `yaml:"search_service"`
	IndexingService  string `yaml:"indexing_service"`
	RankingService   string `yaml:"ranking_service"`
	MLService        string `yaml:"ml_service"`
	AuthService      string `yaml:"auth_service"`
	AnalyticsService string `yaml:"analytics_service"`
}

type MetricsConfig struct {
	EnablePrometheus bool   `yaml:"enable_prometheus"`
	MetricsPath      string `yaml:"metrics_path"`
	EnableHealthCheck bool  `yaml:"enable_health_check"`
}

// API Gateway structure
type APIGateway struct {
	config      *Config
	router      *gin.Engine
	redisClient *redis.Client
	logger      *zap.Logger
	rateLimiters map[string]*rate.Limiter
	mu          sync.RWMutex
	
	// gRPC clients
	searchClient    pb.SearchServiceClient
	authClient      pbauth.AuthServiceClient
	indexingClient  pbindexing.IndexingServiceClient
	rankingClient   pbranking.RankingServiceClient
	mlClient        pbml.MLServicesClient
	
	// Metrics
	requestCounter    *prometheus.CounterVec
	requestDuration   *prometheus.HistogramVec
	responseSize      *prometheus.HistogramVec
	activeConnections prometheus.Gauge
	cacheHits         *prometheus.CounterVec
	cacheMisses       *prometheus.CounterVec
	rateLimitHits     *prometheus.CounterVec
}

// Request/Response structures
type SearchRequest struct {
	Query     string                 `json:"query" binding:"required"`
	Filters   map[string]interface{} `json:"filters,omitempty"`
	Page      int                    `json:"page,omitempty"`
	PageSize  int                    `json:"page_size,omitempty"`
	SortBy    string                 `json:"sort_by,omitempty"`
	UserID    string                 `json:"user_id,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
}

type SearchResponse struct {
	Results    []SearchResult `json:"results"`
	Total      int64          `json:"total"`
	Page       int            `json:"page"`
	PageSize   int            `json:"page_size"`
	QueryTime  float64        `json:"query_time_ms"`
	Facets     map[string]interface{} `json:"facets,omitempty"`
	Suggestions []string      `json:"suggestions,omitempty"`
}

type SearchResult struct {
	ID          string                 `json:"id"`
	Title       string                 `json:"title"`
	URL         string                 `json:"url"`
	Snippet     string                 `json:"snippet"`
	Score       float64                `json:"score"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Highlights  []string               `json:"highlights,omitempty"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type HealthResponse struct {
	Status    string            `json:"status"`
	Timestamp time.Time         `json:"timestamp"`
	Services  map[string]string `json:"services"`
	Metrics   map[string]interface{} `json:"metrics"`
}

// Rate limiter for different types
type RateLimiter struct {
	IPLimiters    map[string]*rate.Limiter
	APIKeyLimiters map[string]*rate.Limiter
	GlobalLimiter *rate.Limiter
	mu            sync.RWMutex
}

// Cache interface
type Cache interface {
	Get(ctx context.Context, key string) (string, error)
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
}

// Redis cache implementation
type RedisCache struct {
	client *redis.Client
}

func (r *RedisCache) Get(ctx context.Context, key string) (string, error) {
	return r.client.Get(ctx, key).Result()
}

func (r *RedisCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	return r.client.Set(ctx, key, value, ttl).Err()
}

func (r *RedisCache) Delete(ctx context.Context, key string) error {
	return r.client.Del(ctx, key).Err()
}

func (r *RedisCache) Clear(ctx context.Context) error {
	return r.client.FlushDB(ctx).Err()
}

// NewAPIGateway creates a new API Gateway instance
func NewAPIGateway(config *Config) (*APIGateway, error) {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}

	// Initialize Redis client
	var redisClient *redis.Client
	if config.Cache.EnableRedis {
		redisClient = redis.NewClient(&redis.Options{
			Addr:     fmt.Sprintf("%s:%d", config.Redis.Host, config.Redis.Port),
			Password: config.Redis.Password,
			DB:       config.Redis.DB,
			PoolSize: config.Redis.PoolSize,
		})

		// Test Redis connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := redisClient.Ping(ctx).Err(); err != nil {
			return nil, fmt.Errorf("failed to connect to Redis: %w", err)
		}
	}

	// Initialize gRPC clients
	searchConn, err := grpc.Dial(config.Services.SearchService, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to search service: %w", err)
	}

	authConn, err := grpc.Dial(config.Services.AuthService, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to auth service: %w", err)
	}

	indexingConn, err := grpc.Dial(config.Services.IndexingService, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to indexing service: %w", err)
	}

	rankingConn, err := grpc.Dial(config.Services.RankingService, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to ranking service: %w", err)
	}

	mlConn, err := grpc.Dial(config.Services.MLService, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to ML service: %w", err)
	}

	// Initialize Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// Initialize metrics
	requestCounter := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_gateway_requests_total",
			Help: "Total number of API requests",
		},
		[]string{"method", "endpoint", "status_code"},
	)

	requestDuration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "api_gateway_request_duration_seconds",
			Help:    "Request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	responseSize := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "api_gateway_response_size_bytes",
			Help:    "Response size in bytes",
			Buckets: prometheus.ExponentialBuckets(100, 10, 8),
		},
		[]string{"method", "endpoint"},
	)

	activeConnections := prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "api_gateway_active_connections",
			Help: "Number of active connections",
		},
	)

	cacheHits := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_gateway_cache_hits_total",
			Help: "Total number of cache hits",
		},
		[]string{"cache_type"},
	)

	cacheMisses := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_gateway_cache_misses_total",
			Help: "Total number of cache misses",
		},
		[]string{"cache_type"},
	)

	rateLimitHits := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_gateway_rate_limit_hits_total",
			Help: "Total number of rate limit hits",
		},
		[]string{"limiter_type"},
	)

	// Register metrics
	prometheus.MustRegister(requestCounter, requestDuration, responseSize, activeConnections, cacheHits, cacheMisses, rateLimitHits)

	gateway := &APIGateway{
		config:      config,
		router:      router,
		redisClient: redisClient,
		logger:      logger,
		rateLimiters: make(map[string]*rate.Limiter),
		searchClient:    pb.NewSearchServiceClient(searchConn),
		authClient:      pbauth.NewAuthServiceClient(authConn),
		indexingClient:  pbindexing.NewIndexingServiceClient(indexingConn),
		rankingClient:   pbranking.NewRankingServiceClient(rankingConn),
		mlClient:        pbml.NewMLServicesClient(mlConn),
		requestCounter:    requestCounter,
		requestDuration:   requestDuration,
		responseSize:      responseSize,
		activeConnections: activeConnections,
		cacheHits:         cacheHits,
		cacheMisses:       cacheMisses,
		rateLimitHits:     rateLimitHits,
	}

	// Setup middleware
	gateway.setupMiddleware()

	// Setup routes
	gateway.setupRoutes()

	return gateway, nil
}

// setupMiddleware configures all middleware
func (g *APIGateway) setupMiddleware() {
	// CORS middleware
	if g.config.Security.EnableCORS {
		g.router.Use(g.corsMiddleware())
	}

	// Logging middleware
	g.router.Use(g.loggingMiddleware())

	// Metrics middleware
	g.router.Use(g.metricsMiddleware())

	// Rate limiting middleware
	if g.config.Security.EnableRateLimit {
		g.router.Use(g.rateLimitMiddleware())
	}

	// Authentication middleware
	if g.config.Security.EnableAPIKeys || g.config.Security.EnableJWT {
		g.router.Use(g.authMiddleware())
	}

	// Compression middleware
	g.router.Use(g.compressionMiddleware())

	// Request size limiting
	g.router.Use(g.requestSizeMiddleware())
}

// setupRoutes configures all API routes
func (g *APIGateway) setupRoutes() {
	// Health check endpoint
	if g.config.Metrics.EnableHealthCheck {
		g.router.GET("/health", g.healthCheck)
	}

	// Metrics endpoint
	if g.config.Metrics.EnablePrometheus {
		g.router.GET(g.config.Metrics.MetricsPath, gin.WrapH(promhttp.Handler()))
	}

	// API routes
	api := g.router.Group("/api/v1")
	{
		// Search endpoints
		api.POST("/search", g.search)
		api.GET("/search/suggest", g.searchSuggest)
		api.GET("/search/autocomplete", g.autocomplete)

		// Document endpoints
		api.GET("/documents/:id", g.getDocument)
		api.POST("/documents", g.createDocument)
		api.PUT("/documents/:id", g.updateDocument)
		api.DELETE("/documents/:id", g.deleteDocument)

		// Analytics endpoints
		api.POST("/analytics/click", g.trackClick)
		api.POST("/analytics/impression", g.trackImpression)
		api.GET("/analytics/stats", g.getAnalytics)

		// Admin endpoints
		admin := api.Group("/admin")
		admin.Use(g.adminAuthMiddleware())
		{
			admin.GET("/stats", g.getStats)
			admin.POST("/cache/clear", g.clearCache)
			admin.GET("/rate-limits", g.getRateLimits)
			admin.POST("/rate-limits/reset", g.resetRateLimits)
		}
	}
}

// Middleware implementations
func (g *APIGateway) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		
		// Check if origin is allowed
		allowed := false
		for _, allowedOrigin := range g.config.Security.AllowedOrigins {
			if allowedOrigin == "*" || allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-API-Key")
		c.Header("Access-Control-Allow-Credentials", "true")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func (g *APIGateway) loggingMiddleware() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		g.logger.Info("HTTP Request",
			zap.String("method", param.Method),
			zap.String("path", param.Path),
			zap.Int("status", param.StatusCode),
			zap.Duration("latency", param.Latency),
			zap.String("client_ip", param.ClientIP),
			zap.String("user_agent", param.Request.UserAgent()),
		)
		return ""
	})
}

func (g *APIGateway) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		// Increment active connections
		g.activeConnections.Inc()
		defer g.activeConnections.Dec()

		c.Next()

		// Record metrics
		duration := time.Since(start).Seconds()
		status := strconv.Itoa(c.Writer.Status())
		
		g.requestCounter.WithLabelValues(c.Request.Method, c.FullPath(), status).Inc()
		g.requestDuration.WithLabelValues(c.Request.Method, c.FullPath()).Observe(duration)
		g.responseSize.WithLabelValues(c.Request.Method, c.FullPath()).Observe(float64(c.Writer.Size()))
	}
}

func (g *APIGateway) rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var limiter *rate.Limiter
		var limiterType string

		// Get rate limiter based on configuration
		if g.config.RateLimit.EnablePerAPIKey {
			apiKey := c.GetHeader(g.config.Security.APIKeyHeader)
			if apiKey != "" {
				limiter = g.getRateLimiter("api_key:"+apiKey)
				limiterType = "api_key"
			}
		}

		if limiter == nil && g.config.RateLimit.EnablePerIP {
			clientIP := c.ClientIP()
			limiter = g.getRateLimiter("ip:"+clientIP)
			limiterType = "ip"
		}

		if limiter == nil {
			limiter = g.getRateLimiter("global")
			limiterType = "global"
		}

		// Check rate limit
		if !limiter.Allow() {
			g.rateLimitHits.WithLabelValues(limiterType).Inc()
			c.JSON(http.StatusTooManyRequests, ErrorResponse{
				Error:   "Rate limit exceeded",
				Code:    http.StatusTooManyRequests,
				Message: "Too many requests. Please try again later.",
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

func (g *APIGateway) authMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip auth for health check and metrics
		if c.Request.URL.Path == "/health" || c.Request.URL.Path == g.config.Metrics.MetricsPath {
			c.Next()
			return
		}

		// API Key authentication
		if g.config.Security.EnableAPIKeys {
			apiKey := c.GetHeader(g.config.Security.APIKeyHeader)
			if apiKey == "" {
				c.JSON(http.StatusUnauthorized, ErrorResponse{
					Error:   "Unauthorized",
					Code:    http.StatusUnauthorized,
					Message: "API key required",
				})
				c.Abort()
				return
			}

			// Validate API key (in production, check against database)
			if !g.validateAPIKey(apiKey) {
				c.JSON(http.StatusUnauthorized, ErrorResponse{
					Error:   "Unauthorized",
					Code:    http.StatusUnauthorized,
					Message: "Invalid API key",
				})
				c.Abort()
				return
			}
		}

		// JWT authentication
		if g.config.Security.EnableJWT {
			authHeader := c.GetHeader("Authorization")
			if authHeader == "" {
				c.JSON(http.StatusUnauthorized, ErrorResponse{
					Error:   "Unauthorized",
					Code:    http.StatusUnauthorized,
					Message: "Authorization header required",
				})
				c.Abort()
				return
			}

			token := strings.TrimPrefix(authHeader, "Bearer ")
			if !g.validateJWT(token) {
				c.JSON(http.StatusUnauthorized, ErrorResponse{
					Error:   "Unauthorized",
					Code:    http.StatusUnauthorized,
					Message: "Invalid JWT token",
				})
				c.Abort()
				return
			}
		}

		c.Next()
	}
}

func (g *APIGateway) adminAuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// In production, implement proper admin authentication
		adminKey := c.GetHeader("X-Admin-Key")
		if adminKey != "admin-secret-key" {
			c.JSON(http.StatusForbidden, ErrorResponse{
				Error:   "Forbidden",
				Code:    http.StatusForbidden,
				Message: "Admin access required",
			})
			c.Abort()
			return
		}
		c.Next()
	}
}

func (g *APIGateway) compressionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Enable gzip compression
		c.Header("Content-Encoding", "gzip")
		c.Next()
	}
}

func (g *APIGateway) requestSizeMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Limit request size to 10MB
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 10<<20)
		c.Next()
	}
}

// Rate limiter management
func (g *APIGateway) getRateLimiter(key string) *rate.Limiter {
	g.mu.Lock()
	defer g.mu.Unlock()

	limiter, exists := g.rateLimiters[key]
	if !exists {
		limiter = rate.NewLimiter(
			rate.Limit(g.config.RateLimit.RequestsPerMinute/60.0),
			g.config.RateLimit.BurstSize,
		)
		g.rateLimiters[key] = limiter
	}

	return limiter
}

// Authentication helpers
func (g *APIGateway) validateAPIKey(apiKey string) bool {
	// In production, validate against database
	validKeys := map[string]bool{
		"test-api-key-123": true,
		"prod-api-key-456": true,
	}
	return validKeys[apiKey]
}

func (g *APIGateway) validateJWT(token string) bool {
	// In production, implement proper JWT validation
	return len(token) > 10
}

// Handler implementations
func (g *APIGateway) healthCheck(c *gin.Context) {
	services := make(map[string]string)
	
	// Check Redis connection
	if g.redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := g.redisClient.Ping(ctx).Err(); err != nil {
			services["redis"] = "unhealthy"
		} else {
			services["redis"] = "healthy"
		}
	} else {
		services["redis"] = "disabled"
	}

	// Check external services
	services["search_service"] = g.checkServiceHealth(g.config.Services.SearchService)
	services["indexing_service"] = g.checkServiceHealth(g.config.Services.IndexingService)

	// Determine overall status
	status := "healthy"
	for _, serviceStatus := range services {
		if serviceStatus == "unhealthy" {
			status = "unhealthy"
			break
		}
	}

	response := HealthResponse{
		Status:    status,
		Timestamp: time.Now(),
		Services:  services,
		Metrics: map[string]interface{}{
			"active_connections": g.getActiveConnections(),
			"total_requests":     g.getTotalRequests(),
		},
	}

	if status == "healthy" {
		c.JSON(http.StatusOK, response)
	} else {
		c.JSON(http.StatusServiceUnavailable, response)
	}
}

func (g *APIGateway) search(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid request",
			Code:    http.StatusBadRequest,
			Message: err.Error(),
		})
		return
	}

	// Set defaults
	if req.Page <= 0 {
		req.Page = 1
	}
	if req.PageSize <= 0 {
		req.PageSize = 10
	}
	if req.PageSize > 100 {
		req.PageSize = 100
	}

	// Check cache first
	cacheKey := g.generateCacheKey("search", req)
	if g.config.Cache.EnableRedis && g.redisClient != nil {
		if cached, err := g.redisClient.Get(c.Request.Context(), cacheKey).Result(); err == nil {
			var response SearchResponse
			if json.Unmarshal([]byte(cached), &response) == nil {
				g.cacheHits.WithLabelValues("search").Inc()
				c.JSON(http.StatusOK, response)
				return
			}
		}
		g.cacheMisses.WithLabelValues("search").Inc()
	}

	// Convert to gRPC request
	grpcReq := &pb.SearchRequest{
		Query:     req.Query,
		Filters:   g.convertFiltersToProto(req.Filters),
		Page:      int32(req.Page),
		PageSize:  int32(req.PageSize),
		SortBy:    req.SortBy,
		UserId:    req.UserID,
		SessionId: req.SessionID,
		BoostFields: req.BoostFields,
		Timestamp: timestamppb.New(time.Now()),
	}

	// Forward request to search service via gRPC
	start := time.Now()
	grpcResp, err := g.searchClient.Search(c.Request.Context(), grpcReq)
	queryTime := time.Since(start).Seconds() * 1000

	if err != nil {
		g.logger.Error("Search request failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Search failed",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	// Convert gRPC response to HTTP response
	response := g.convertSearchResponseFromProto(grpcResp)
	response.QueryTime = queryTime

	// Cache the response
	if g.config.Cache.EnableRedis && g.redisClient != nil {
		if data, err := json.Marshal(response); err == nil {
			g.redisClient.Set(c.Request.Context(), cacheKey, data, g.config.Cache.DefaultTTL)
		}
	}

	c.JSON(http.StatusOK, response)
}

func (g *APIGateway) searchSuggest(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Missing query parameter",
			Code:    http.StatusBadRequest,
			Message: "Query parameter 'q' is required",
		})
		return
	}

	// Get suggestions from search service
	suggestions, err := g.getSearchSuggestions(query)
	if err != nil {
		g.logger.Error("Failed to get suggestions", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to get suggestions",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"suggestions": suggestions,
	})
}

func (g *APIGateway) autocomplete(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Missing query parameter",
			Code:    http.StatusBadRequest,
			Message: "Query parameter 'q' is required",
		})
		return
	}

	// Get autocomplete suggestions
	suggestions, err := g.getAutocompleteSuggestions(query)
	if err != nil {
		g.logger.Error("Failed to get autocomplete", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to get autocomplete",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"suggestions": suggestions,
	})
}

func (g *APIGateway) getDocument(c *gin.Context) {
	docID := c.Param("id")
	if docID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Missing document ID",
			Code:    http.StatusBadRequest,
			Message: "Document ID is required",
		})
		return
	}

	// Get document from indexing service
	document, err := g.getDocumentByID(docID)
	if err != nil {
		g.logger.Error("Failed to get document", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to get document",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	if document == nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Error:   "Document not found",
			Code:    http.StatusNotFound,
			Message: "Document with the specified ID was not found",
		})
		return
	}

	c.JSON(http.StatusOK, document)
}

func (g *APIGateway) createDocument(c *gin.Context) {
	var document map[string]interface{}
	if err := c.ShouldBindJSON(&document); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid document",
			Code:    http.StatusBadRequest,
			Message: err.Error(),
		})
		return
	}

	// Create document via indexing service
	docID, err := g.createDocumentInService(document)
	if err != nil {
		g.logger.Error("Failed to create document", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to create document",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusCreated, map[string]interface{}{
		"id":      docID,
		"message": "Document created successfully",
	})
}

func (g *APIGateway) updateDocument(c *gin.Context) {
	docID := c.Param("id")
	if docID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Missing document ID",
			Code:    http.StatusBadRequest,
			Message: "Document ID is required",
		})
		return
	}

	var document map[string]interface{}
	if err := c.ShouldBindJSON(&document); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid document",
			Code:    http.StatusBadRequest,
			Message: err.Error(),
		})
		return
	}

	// Update document via indexing service
	err := g.updateDocumentInService(docID, document)
	if err != nil {
		g.logger.Error("Failed to update document", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to update document",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"message": "Document updated successfully",
	})
}

func (g *APIGateway) deleteDocument(c *gin.Context) {
	docID := c.Param("id")
	if docID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Missing document ID",
			Code:    http.StatusBadRequest,
			Message: "Document ID is required",
		})
		return
	}

	// Delete document via indexing service
	err := g.deleteDocumentInService(docID)
	if err != nil {
		g.logger.Error("Failed to delete document", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to delete document",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"message": "Document deleted successfully",
	})
}

func (g *APIGateway) trackClick(c *gin.Context) {
	var clickData map[string]interface{}
	if err := c.ShouldBindJSON(&clickData); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid click data",
			Code:    http.StatusBadRequest,
			Message: err.Error(),
		})
		return
	}

	// Track click via analytics service
	err := g.trackClickInService(clickData)
	if err != nil {
		g.logger.Error("Failed to track click", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to track click",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"message": "Click tracked successfully",
	})
}

func (g *APIGateway) trackImpression(c *gin.Context) {
	var impressionData map[string]interface{}
	if err := c.ShouldBindJSON(&impressionData); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid impression data",
			Code:    http.StatusBadRequest,
			Message: err.Error(),
		})
		return
	}

	// Track impression via analytics service
	err := g.trackImpressionInService(impressionData)
	if err != nil {
		g.logger.Error("Failed to track impression", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to track impression",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"message": "Impression tracked successfully",
	})
}

func (g *APIGateway) getAnalytics(c *gin.Context) {
	// Get analytics data
	analytics, err := g.getAnalyticsFromService()
	if err != nil {
		g.logger.Error("Failed to get analytics", zap.Error(err))
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to get analytics",
			Code:    http.StatusInternalServerError,
			Message: "Internal server error",
		})
		return
	}

	c.JSON(http.StatusOK, analytics)
}

func (g *APIGateway) getStats(c *gin.Context) {
	stats := map[string]interface{}{
		"requests_total":     g.getTotalRequests(),
		"active_connections": g.getActiveConnections(),
		"cache_hits":         g.getCacheHits(),
		"cache_misses":       g.getCacheMisses(),
		"rate_limit_hits":    g.getRateLimitHits(),
		"uptime":            g.getUptime(),
	}

	c.JSON(http.StatusOK, stats)
}

func (g *APIGateway) clearCache(c *gin.Context) {
	if g.redisClient != nil {
		err := g.redisClient.FlushDB(c.Request.Context()).Err()
		if err != nil {
			g.logger.Error("Failed to clear cache", zap.Error(err))
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error:   "Failed to clear cache",
				Code:    http.StatusInternalServerError,
				Message: "Internal server error",
			})
			return
		}
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"message": "Cache cleared successfully",
	})
}

func (g *APIGateway) getRateLimits(c *gin.Context) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	limits := make(map[string]interface{})
	for key, limiter := range g.rateLimiters {
		limits[key] = map[string]interface{}{
			"limit":  limiter.Limit(),
			"burst":  limiter.Burst(),
			"tokens": limiter.Tokens(),
		}
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"rate_limits": limits,
	})
}

func (g *APIGateway) resetRateLimits(c *gin.Context) {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Reset all rate limiters
	for key := range g.rateLimiters {
		g.rateLimiters[key] = rate.NewLimiter(
			rate.Limit(g.config.RateLimit.RequestsPerMinute/60.0),
			g.config.RateLimit.BurstSize,
		)
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"message": "Rate limits reset successfully",
	})
}

// Helper methods
func (g *APIGateway) generateCacheKey(prefix string, data interface{}) string {
	jsonData, _ := json.Marshal(data)
	return fmt.Sprintf("%s:%x", prefix, jsonData)
}

func (g *APIGateway) checkServiceHealth(serviceURL string) string {
	// In production, implement proper health checks
	if serviceURL == "" {
		return "disabled"
	}
	return "healthy"
}

func (g *APIGateway) forwardSearchRequest(req SearchRequest) (*SearchResponse, error) {
	// In production, make HTTP request to search service
	// For now, return mock data
	return &SearchResponse{
		Results: []SearchResult{
			{
				ID:      "1",
				Title:   "Example Document",
				URL:     "https://example.com",
				Snippet: "This is an example document...",
				Score:   0.95,
			},
		},
		Total:    1,
		Page:     req.Page,
		PageSize: req.PageSize,
	}, nil
}

func (g *APIGateway) getSearchSuggestions(query string) ([]string, error) {
	// In production, get from search service
	return []string{"machine learning", "artificial intelligence", "deep learning"}, nil
}

func (g *APIGateway) getAutocompleteSuggestions(query string) ([]string, error) {
	// In production, get from search service
	return []string{"machine learning algorithms", "machine learning python", "machine learning tutorial"}, nil
}

func (g *APIGateway) getDocumentByID(docID string) (map[string]interface{}, error) {
	// In production, get from indexing service
	return map[string]interface{}{
		"id":      docID,
		"title":   "Example Document",
		"content": "This is example content",
		"url":     "https://example.com",
	}, nil
}

func (g *APIGateway) createDocumentInService(document map[string]interface{}) (string, error) {
	// In production, create via indexing service
	return "doc-123", nil
}

func (g *APIGateway) updateDocumentInService(docID string, document map[string]interface{}) error {
	// In production, update via indexing service
	return nil
}

func (g *APIGateway) deleteDocumentInService(docID string) error {
	// In production, delete via indexing service
	return nil
}

func (g *APIGateway) trackClickInService(clickData map[string]interface{}) error {
	// In production, track via analytics service
	return nil
}

func (g *APIGateway) trackImpressionInService(impressionData map[string]interface{}) error {
	// In production, track via analytics service
	return nil
}

func (g *APIGateway) getAnalyticsFromService() (map[string]interface{}, error) {
	// In production, get from analytics service
	return map[string]interface{}{
		"total_searches": 1000,
		"total_clicks":   100,
		"ctr":            0.1,
	}, nil
}

// Statistics helpers
func (g *APIGateway) getTotalRequests() int64 {
	// In production, get from metrics
	return 1000
}

func (g *APIGateway) getActiveConnections() float64 {
	// In production, get from metrics
	return 10
}

func (g *APIGateway) getCacheHits() int64 {
	// In production, get from metrics
	return 500
}

func (g *APIGateway) getCacheMisses() int64 {
	// In production, get from metrics
	return 500
}

func (g *APIGateway) getRateLimitHits() int64 {
	// In production, get from metrics
	return 10
}

func (g *APIGateway) getUptime() time.Duration {
	// In production, calculate actual uptime
	return time.Hour * 24
}

// Helper methods for gRPC conversion

// convertFiltersToProto converts HTTP filters to protobuf filters
func (g *APIGateway) convertFiltersToProto(filters map[string]interface{}) map[string]string {
	protoFilters := make(map[string]string)
	for k, v := range filters {
		if str, ok := v.(string); ok {
			protoFilters[k] = str
		} else {
			protoFilters[k] = fmt.Sprintf("%v", v)
		}
	}
	return protoFilters
}

// convertSearchResponseFromProto converts protobuf search response to HTTP response
func (g *APIGateway) convertSearchResponseFromProto(grpcResp *pb.SearchResponse) *SearchResponse {
	results := make([]SearchResult, len(grpcResp.Results))
	for i, grpcResult := range grpcResp.Results {
		results[i] = SearchResult{
			ID:         grpcResult.Id,
			Title:      grpcResult.Title,
			URL:        grpcResult.Url,
			Snippet:    grpcResult.Snippet,
			Score:      grpcResult.Score,
			Metadata:   g.convertMetadataFromProto(grpcResult.Metadata),
			Highlights: grpcResult.Highlights,
		}
	}

	return &SearchResponse{
		Results:        results,
		Total:          grpcResp.Total,
		Page:           int(grpcResp.Page),
		PageSize:       int(grpcResp.PageSize),
		QueryTime:      grpcResp.QueryTimeMs,
		Facets:         g.convertFacetsFromProto(grpcResp.Facets),
		Suggestions:    grpcResp.Suggestions,
		CorrectedQuery: grpcResp.CorrectedQuery,
		ConfidenceScore: grpcResp.ConfidenceScore,
	}
}

// convertMetadataFromProto converts protobuf metadata to HTTP metadata
func (g *APIGateway) convertMetadataFromProto(protoMetadata map[string]*any.Any) map[string]interface{} {
	metadata := make(map[string]interface{})
	for k, v := range protoMetadata {
		// For simplicity, convert to string
		// In production, you might want to handle different types
		metadata[k] = string(v.Value)
	}
	return metadata
}

// convertFacetsFromProto converts protobuf facets to HTTP facets
func (g *APIGateway) convertFacetsFromProto(protoFacets map[string]*any.Any) map[string]interface{} {
	facets := make(map[string]interface{})
	for k, v := range protoFacets {
		// For simplicity, convert to string
		// In production, you might want to handle different types
		facets[k] = string(v.Value)
	}
	return facets
}

// Start the API Gateway
func (g *APIGateway) Start() error {
	server := &http.Server{
		Addr:         fmt.Sprintf("%s:%d", g.config.Server.Host, g.config.Server.Port),
		Handler:      g.router,
		ReadTimeout:  time.Duration(g.config.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(g.config.Server.WriteTimeout) * time.Second,
		IdleTimeout:  time.Duration(g.config.Server.IdleTimeout) * time.Second,
	}

	// Configure TLS if enabled
	if g.config.Server.EnableHTTPS {
		server.TLSConfig = &tls.Config{
			MinVersion: tls.VersionTLS12,
		}
	}

	g.logger.Info("Starting API Gateway",
		zap.String("host", g.config.Server.Host),
		zap.Int("port", g.config.Server.Port),
		zap.Bool("https", g.config.Server.EnableHTTPS),
	)

	// Start server in goroutine
	go func() {
		var err error
		if g.config.Server.EnableHTTPS {
			err = server.ListenAndServeTLS(g.config.Server.CertFile, g.config.Server.KeyFile)
		} else {
			err = server.ListenAndServe()
		}
		if err != nil && err != http.ErrServerClosed {
			g.logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	g.logger.Info("Shutting down API Gateway...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		g.logger.Error("Server forced to shutdown", zap.Error(err))
		return err
	}

	g.logger.Info("API Gateway stopped")
	return nil
}

// Load configuration from file
func LoadConfig(configPath string) (*Config, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &config, nil
}

func main() {
	// Load configuration
	configPath := os.Getenv("CONFIG_PATH")
	if configPath == "" {
		configPath = "config.yaml"
	}

	config, err := LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Create API Gateway
	gateway, err := NewAPIGateway(config)
	if err != nil {
		log.Fatalf("Failed to create API Gateway: %v", err)
	}

	// Start the gateway
	if err := gateway.Start(); err != nil {
		log.Fatalf("Failed to start API Gateway: %v", err)
	}
}