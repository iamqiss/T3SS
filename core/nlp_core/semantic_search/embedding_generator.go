// T3SS Project
// File: core/nlp_core/semantic_search/embedding_generator.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package semantic_search

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
)

// EmbeddingConfig holds configuration for embedding generation
type EmbeddingConfig struct {
	// Model configuration
	ModelName           string        `yaml:"model_name"`
	ModelPath           string        `yaml:"model_path"`
	MaxSequenceLength   int           `yaml:"max_sequence_length"`
	EmbeddingDimension  int           `yaml:"embedding_dimension"`
	
	// Performance settings
	BatchSize           int           `yaml:"batch_size"`
	MaxConcurrentReqs   int           `yaml:"max_concurrent_requests"`
	EnableGPU           bool          `yaml:"enable_gpu"`
	GPUMemoryFraction   float64       `yaml:"gpu_memory_fraction"`
	
	// Text preprocessing
	EnableNormalization bool          `yaml:"enable_normalization"`
	EnableStemming      bool          `yaml:"enable_stemming"`
	EnableStopWords     bool          `yaml:"enable_stop_words"`
	MinTokenLength      int           `yaml:"min_token_length"`
	MaxTokenLength      int           `yaml:"max_token_length"`
	
	// Caching
	EnableCaching       bool          `yaml:"enable_caching"`
	CacheTTL            time.Duration `yaml:"cache_ttl"`
	RedisEndpoint       string        `yaml:"redis_endpoint"`
	
	// Quality control
	MinTextLength       int           `yaml:"min_text_length"`
	MaxTextLength       int           `yaml:"max_text_length"`
	QualityThreshold    float64       `yaml:"quality_threshold"`
}

// EmbeddingRequest represents a request for embedding generation
type EmbeddingRequest struct {
	ID          string            `json:"id"`
	Text        string            `json:"text"`
	TextType    TextType          `json:"text_type"`
	Language    string            `json:"language"`
	Metadata    map[string]string `json:"metadata"`
	Priority    int               `json:"priority"`
	CreatedAt   time.Time         `json:"created_at"`
}

// EmbeddingResponse represents the response from embedding generation
type EmbeddingResponse struct {
	ID              string    `json:"id"`
	Embedding       []float64 `json:"embedding"`
	Quality         float64   `json:"quality"`
	ProcessingTime  time.Duration `json:"processing_time"`
	ModelUsed       string    `json:"model_used"`
	Dimension       int       `json:"dimension"`
	Error           string    `json:"error,omitempty"`
}

// TextType represents different types of text content
type TextType string

const (
	TextTypeQuery      TextType = "query"
	TextTypeDocument   TextType = "document"
	TextTypeTitle      TextType = "title"
	TextTypeAbstract   TextType = "abstract"
	TextTypeContent    TextType = "content"
	TextTypeMetadata   TextType = "metadata"
)

// EmbeddingGenerator generates high-quality embeddings using transformer models
type EmbeddingGenerator struct {
	config     EmbeddingConfig
	logger     *zap.Logger
	redisClient *redis.Client
	
	// Model components (simplified for this example)
	model       *TransformerModel
	tokenizer   *Tokenizer
	normalizer  *TextNormalizer
	
	// Processing queues
	requestQueue chan *EmbeddingRequest
	responseQueue chan *EmbeddingResponse
	
	// Workers and synchronization
	workers     []*EmbeddingWorker
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
	
	// Statistics
	stats       *EmbeddingStats
	statsMutex  sync.RWMutex
	
	// Caching
	cache       map[string]*EmbeddingResponse
	cacheMutex  sync.RWMutex
}

// TransformerModel represents a transformer-based embedding model
type TransformerModel struct {
	name        string
	dimension   int
	maxLength   int
	vocabSize   int
	layers      int
	heads       int
	weights     map[string][]float64
}

// Tokenizer handles text tokenization
type Tokenizer struct {
	vocab       map[string]int
	unkToken    string
	padToken    string
	clsToken    string
	sepToken    string
	maxLength   int
}

// TextNormalizer handles text normalization
type TextNormalizer struct {
	enableStemming   bool
	enableStopWords  bool
	stopWords        map[string]bool
	minTokenLength   int
	maxTokenLength   int
}

// EmbeddingWorker processes embedding requests
type EmbeddingWorker struct {
	id          int
	generator   *EmbeddingGenerator
	requestQueue chan *EmbeddingRequest
	logger      *zap.Logger
}

// EmbeddingStats tracks embedding generation statistics
type EmbeddingStats struct {
	TotalRequests       int64
	SuccessfulRequests  int64
	FailedRequests      int64
	CacheHits           int64
	CacheMisses         int64
	AverageLatency      time.Duration
	TotalProcessingTime time.Duration
	AverageQuality      float64
	RequestsPerSecond   float64
}

// NewEmbeddingGenerator creates a new embedding generator
func NewEmbeddingGenerator(config EmbeddingConfig) (*EmbeddingGenerator, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}
	
	// Initialize Redis client if caching is enabled
	var redisClient *redis.Client
	if config.EnableCaching {
		redisClient = redis.NewClient(&redis.Options{
			Addr: config.RedisEndpoint,
		})
		
		// Test connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := redisClient.Ping(ctx).Err(); err != nil {
			return nil, fmt.Errorf("failed to connect to Redis: %w", err)
		}
	}
	
	// Initialize model components
	model := &TransformerModel{
		name:        config.ModelName,
		dimension:   config.EmbeddingDimension,
		maxLength:   config.MaxSequenceLength,
		vocabSize:   50000, // Simplified
		layers:      12,
		heads:       12,
		weights:     make(map[string][]float64),
	}
	
	tokenizer := &Tokenizer{
		vocab:       make(map[string]int),
		unkToken:    "[UNK]",
		padToken:    "[PAD]",
		clsToken:    "[CLS]",
		sepToken:    "[SEP]",
		maxLength:   config.MaxSequenceLength,
	}
	
	normalizer := &TextNormalizer{
		enableStemming:   config.EnableStemming,
		enableStopWords:  config.EnableStopWords,
		stopWords:        loadStopWords(),
		minTokenLength:   config.MinTokenLength,
		maxTokenLength:   config.MaxTokenLength,
	}
	
	ctx, cancel = context.WithCancel(context.Background())
	
	return &EmbeddingGenerator{
		config:     config,
		logger:     logger,
		redisClient: redisClient,
		
		model:      model,
		tokenizer:  tokenizer,
		normalizer: normalizer,
		
		requestQueue:  make(chan *EmbeddingRequest, config.BatchSize*2),
		responseQueue: make(chan *EmbeddingResponse, config.BatchSize*2),
		
		ctx:    ctx,
		cancel: cancel,
		
		stats: &EmbeddingStats{},
		cache: make(map[string]*EmbeddingResponse),
	}, nil
}

// Start begins the embedding generation service
func (eg *EmbeddingGenerator) Start() error {
	eg.logger.Info("Starting embedding generator")
	
	// Start embedding workers
	for i := 0; i < eg.config.MaxConcurrentReqs; i++ {
		worker := &EmbeddingWorker{
			id:           i,
			generator:    eg,
			requestQueue: eg.requestQueue,
			logger:       eg.logger.With(zap.Int("worker_id", i)),
		}
		eg.workers = append(eg.workers, worker)
		
		eg.wg.Add(1)
		go worker.processRequests()
	}
	
	// Start statistics collector
	eg.wg.Add(1)
	go eg.collectStats()
	
	eg.logger.Info("Embedding generator started successfully")
	return nil
}

// Stop gracefully shuts down the embedding generator
func (eg *EmbeddingGenerator) Stop() error {
	eg.logger.Info("Stopping embedding generator")
	
	eg.cancel()
	eg.wg.Wait()
	
	if eg.redisClient != nil {
		eg.redisClient.Close()
	}
	
	eg.logger.Info("Embedding generator stopped")
	return nil
}

// GenerateEmbedding generates an embedding for the given text
func (eg *EmbeddingGenerator) GenerateEmbedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	// Validate request
	if err := eg.validateRequest(req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}
	
	// Check cache first
	if eg.config.EnableCaching {
		if cached := eg.getFromCache(req); cached != nil {
			eg.updateStats(true, cached.ProcessingTime, cached.Quality, true)
			return cached, nil
		}
	}
	
	// Generate embedding
	start := time.Now()
	embedding, quality, err := eg.generateEmbeddingInternal(req)
	processingTime := time.Since(start)
	
	if err != nil {
		eg.updateStats(false, processingTime, 0.0, false)
		return nil, fmt.Errorf("embedding generation failed: %w", err)
	}
	
	response := &EmbeddingResponse{
		ID:             req.ID,
		Embedding:      embedding,
		Quality:        quality,
		ProcessingTime: processingTime,
		ModelUsed:      eg.config.ModelName,
		Dimension:      eg.config.EmbeddingDimension,
	}
	
	// Cache the result
	if eg.config.EnableCaching {
		eg.setCache(req, response)
	}
	
	eg.updateStats(true, processingTime, quality, false)
	return response, nil
}

// GenerateEmbeddingsBatch generates embeddings for multiple texts
func (eg *EmbeddingGenerator) GenerateEmbeddingsBatch(ctx context.Context, requests []*EmbeddingRequest) ([]*EmbeddingResponse, error) {
	responses := make([]*EmbeddingResponse, len(requests))
	semaphore := make(chan struct{}, eg.config.MaxConcurrentReqs)
	var wg sync.WaitGroup
	
	for i, req := range requests {
		wg.Add(1)
		go func(index int, request *EmbeddingRequest) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			resp, err := eg.GenerateEmbedding(ctx, request)
			if err != nil {
				resp = &EmbeddingResponse{
					ID:    request.ID,
					Error: err.Error(),
				}
			}
			responses[index] = resp
		}(i, req)
	}
	
	wg.Wait()
	return responses, nil
}

// validateRequest validates an embedding request
func (eg *EmbeddingGenerator) validateRequest(req *EmbeddingRequest) error {
	if req.Text == "" {
		return fmt.Errorf("empty text")
	}
	
	if len(req.Text) < eg.config.MinTextLength {
		return fmt.Errorf("text too short: %d characters", len(req.Text))
	}
	
	if len(req.Text) > eg.config.MaxTextLength {
		return fmt.Errorf("text too long: %d characters", len(req.Text))
	}
	
	return nil
}

// generateEmbeddingInternal performs the actual embedding generation
func (eg *EmbeddingGenerator) generateEmbeddingInternal(req *EmbeddingRequest) ([]float64, float64, error) {
	// Normalize text
	normalizedText := eg.normalizer.Normalize(req.Text)
	
	// Tokenize text
	tokens := eg.tokenizer.Tokenize(normalizedText)
	
	// Convert tokens to IDs
	tokenIds := eg.tokenizer.TokensToIds(tokens)
	
	// Generate embedding using transformer model
	embedding := eg.model.Forward(tokenIds)
	
	// Calculate quality score
	quality := eg.calculateQuality(req.Text, embedding)
	
	// Normalize embedding
	normalizedEmbedding := eg.normalizeEmbedding(embedding)
	
	return normalizedEmbedding, quality, nil
}

// calculateQuality calculates the quality score for an embedding
func (eg *EmbeddingGenerator) calculateQuality(text string, embedding []float64) float64 {
	quality := 1.0
	
	// Text length factor
	textLength := len(text)
	if textLength < 50 {
		quality *= 0.8
	} else if textLength > 1000 {
		quality *= 0.9
	}
	
	// Embedding magnitude factor
	magnitude := eg.calculateMagnitude(embedding)
	if magnitude < 0.1 {
		quality *= 0.7
	} else if magnitude > 10.0 {
		quality *= 0.8
	}
	
	// Text complexity factor
	complexity := eg.calculateTextComplexity(text)
	quality *= complexity
	
	return quality
}

// calculateMagnitude calculates the magnitude of a vector
func (eg *EmbeddingGenerator) calculateMagnitude(embedding []float64) float64 {
	sum := 0.0
	for _, val := range embedding {
		sum += val * val
	}
	return math.Sqrt(sum)
}

// calculateTextComplexity calculates text complexity score
func (eg *EmbeddingGenerator) calculateTextComplexity(text string) float64 {
	// Simplified complexity calculation
	words := len(strings.Fields(text))
	sentences := len(strings.Split(text, "."))
	
	if sentences == 0 {
		return 0.5
	}
	
	avgWordsPerSentence := float64(words) / float64(sentences)
	
	// Normalize to 0.5-1.5 range
	if avgWordsPerSentence < 5 {
		return 0.5
	} else if avgWordsPerSentence > 20 {
		return 1.5
	} else {
		return 0.5 + (avgWordsPerSentence-5)/15.0
	}
}

// normalizeEmbedding normalizes an embedding vector
func (eg *EmbeddingGenerator) normalizeEmbedding(embedding []float64) []float64 {
	magnitude := eg.calculateMagnitude(embedding)
	if magnitude == 0 {
		return embedding
	}
	
	normalized := make([]float64, len(embedding))
	for i, val := range embedding {
		normalized[i] = val / magnitude
	}
	
	return normalized
}

// getFromCache retrieves embedding from cache
func (eg *EmbeddingGenerator) getFromCache(req *EmbeddingRequest) *EmbeddingResponse {
	if !eg.config.EnableCaching {
		return nil
	}
	
	cacheKey := eg.generateCacheKey(req)
	
	// Try Redis first
	if eg.redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()
		
		cached, err := eg.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var response EmbeddingResponse
			if json.Unmarshal([]byte(cached), &response) == nil {
				return &response
			}
		}
	}
	
	// Try local cache
	eg.cacheMutex.RLock()
	defer eg.cacheMutex.RUnlock()
	
	if cached, exists := eg.cache[cacheKey]; exists {
		return cached
	}
	
	return nil
}

// setCache stores embedding in cache
func (eg *EmbeddingGenerator) setCache(req *EmbeddingRequest, resp *EmbeddingResponse) {
	if !eg.config.EnableCaching {
		return
	}
	
	cacheKey := eg.generateCacheKey(req)
	
	// Store in Redis
	if eg.redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()
		
		data, err := json.Marshal(resp)
		if err == nil {
			eg.redisClient.Set(ctx, cacheKey, data, eg.config.CacheTTL)
		}
	}
	
	// Store in local cache
	eg.cacheMutex.Lock()
	defer eg.cacheMutex.Unlock()
	
	eg.cache[cacheKey] = resp
}

// generateCacheKey generates a cache key for a request
func (eg *EmbeddingGenerator) generateCacheKey(req *EmbeddingRequest) string {
	return fmt.Sprintf("embedding:%s:%s:%s", req.TextType, req.Language, req.Text)
}

// updateStats updates embedding statistics
func (eg *EmbeddingGenerator) updateStats(success bool, latency time.Duration, quality float64, cacheHit bool) {
	eg.statsMutex.Lock()
	defer eg.statsMutex.Unlock()
	
	eg.stats.TotalRequests++
	if success {
		eg.stats.SuccessfulRequests++
		eg.stats.TotalProcessingTime += latency
		
		// Update average latency
		if eg.stats.AverageLatency == 0 {
			eg.stats.AverageLatency = latency
		} else {
			eg.stats.AverageLatency = (eg.stats.AverageLatency*9 + latency) / 10
		}
		
		// Update average quality
		if eg.stats.AverageQuality == 0 {
			eg.stats.AverageQuality = quality
		} else {
			eg.stats.AverageQuality = (eg.stats.AverageQuality*9 + quality) / 10
		}
		
		if cacheHit {
			eg.stats.CacheHits++
		} else {
			eg.stats.CacheMisses++
		}
	} else {
		eg.stats.FailedRequests++
	}
	
	// Calculate requests per second
	if eg.stats.TotalProcessingTime > 0 {
		eg.stats.RequestsPerSecond = float64(eg.stats.SuccessfulRequests) / eg.stats.TotalProcessingTime.Seconds()
	}
}

// GetStats returns current embedding statistics
func (eg *EmbeddingGenerator) GetStats() EmbeddingStats {
	eg.statsMutex.RLock()
	defer eg.statsMutex.RUnlock()
	return *eg.stats
}

// processRequests processes embedding requests in a worker
func (ew *EmbeddingWorker) processRequests() {
	defer ew.generator.wg.Done()
	
	ew.logger.Info("Starting embedding worker")
	
	for {
		select {
		case req := <-ew.requestQueue:
			ew.processRequest(req)
		case <-ew.generator.ctx.Done():
			ew.logger.Info("Embedding worker stopping")
			return
		}
	}
}

// processRequest processes a single embedding request
func (ew *EmbeddingWorker) processRequest(req *EmbeddingRequest) {
	start := time.Now()
	
	ew.logger.Debug("Processing embedding request",
		zap.String("id", req.ID),
		zap.String("type", string(req.TextType)),
		zap.Int("text_length", len(req.Text)))
	
	// Generate embedding
	embedding, quality, err := ew.generator.generateEmbeddingInternal(req)
	processingTime := time.Since(start)
	
	response := &EmbeddingResponse{
		ID:             req.ID,
		ProcessingTime: processingTime,
		ModelUsed:      ew.generator.config.ModelName,
		Dimension:      ew.generator.config.EmbeddingDimension,
	}
	
	if err != nil {
		response.Error = err.Error()
		ew.generator.updateStats(false, processingTime, 0.0, false)
	} else {
		response.Embedding = embedding
		response.Quality = quality
		ew.generator.updateStats(true, processingTime, quality, false)
	}
	
	// Send response
	select {
	case ew.generator.responseQueue <- response:
	case <-ew.generator.ctx.Done():
		return
	}
}

// collectStats collects and updates statistics
func (eg *EmbeddingGenerator) collectStats() {
	defer eg.wg.Done()
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			eg.logStats()
		case <-eg.ctx.Done():
			return
		}
	}
}

// logStats logs current statistics
func (eg *EmbeddingGenerator) logStats() {
	stats := eg.GetStats()
	
	eg.logger.Info("Embedding generator statistics",
		zap.Int64("total_requests", stats.TotalRequests),
		zap.Int64("successful_requests", stats.SuccessfulRequests),
		zap.Int64("failed_requests", stats.FailedRequests),
		zap.Int64("cache_hits", stats.CacheHits),
		zap.Int64("cache_misses", stats.CacheMisses),
		zap.Duration("average_latency", stats.AverageLatency),
		zap.Float64("average_quality", stats.AverageQuality),
		zap.Float64("requests_per_second", stats.RequestsPerSecond))
}

// Model methods (simplified implementations)
func (tm *TransformerModel) Forward(tokenIds []int) []float64 {
	// Simplified transformer forward pass
	embedding := make([]float64, tm.dimension)
	
	// Initialize with random values (in production, use actual model weights)
	for i := range embedding {
		embedding[i] = math.Sin(float64(i)) * 0.1
	}
	
	// Apply transformer layers (simplified)
	for layer := 0; layer < tm.layers; layer++ {
		embedding = tm.applyLayer(embedding, tokenIds)
	}
	
	return embedding
}

func (tm *TransformerModel) applyLayer(embedding []float64, tokenIds []int) []float64 {
	// Simplified layer application
	result := make([]float64, len(embedding))
	for i, val := range embedding {
		result[i] = val * 0.9 + math.Sin(float64(i+len(tokenIds))) * 0.1
	}
	return result
}

// Tokenizer methods
func (t *Tokenizer) Tokenize(text string) []string {
	// Simplified tokenization
	words := strings.Fields(text)
	tokens := make([]string, 0, len(words)+2)
	
	tokens = append(tokens, t.clsToken)
	for _, word := range words {
		if len(word) >= 2 {
			tokens = append(tokens, strings.ToLower(word))
		}
	}
	tokens = append(tokens, t.sepToken)
	
	// Truncate if too long
	if len(tokens) > t.maxLength {
		tokens = tokens[:t.maxLength]
	}
	
	return tokens
}

func (t *Tokenizer) TokensToIds(tokens []string) []int {
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		if id, exists := t.vocab[token]; exists {
			ids[i] = id
		} else {
			ids[i] = t.vocab[t.unkToken]
		}
	}
	return ids
}

// TextNormalizer methods
func (tn *TextNormalizer) Normalize(text string) string {
	// Convert to lowercase
	normalized := strings.ToLower(text)
	
	// Remove extra whitespace
	normalized = strings.Join(strings.Fields(normalized), " ")
	
	// Remove stop words if enabled
	if tn.enableStopWords {
		words := strings.Fields(normalized)
		filtered := make([]string, 0, len(words))
		for _, word := range words {
			if !tn.stopWords[word] {
				filtered = append(filtered, word)
			}
		}
		normalized = strings.Join(filtered, " ")
	}
	
	return normalized
}

// loadStopWords loads common stop words
func loadStopWords() map[string]bool {
	stopWords := []string{
		"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
		"is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
		"will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those",
	}
	
	result := make(map[string]bool)
	for _, word := range stopWords {
		result[word] = true
	}
	return result
}
