// T3SS Project
// File: core/nlp_core/semantic_search/semantic_search.go
// (c) 2025 Qiss Labs. All Rights Reserved.

package semantic

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
)

// SemanticSearchEngine provides advanced semantic search capabilities
type SemanticSearchEngine struct {
	config      SemanticConfig
	embedder    *EmbeddingGenerator
	redisClient *redis.Client
	logger      *zap.Logger
	vectorIndex VectorIndex
	stats       *SemanticStats
}

// SemanticConfig configuration for semantic search
type SemanticConfig struct {
	EmbeddingModel      string        `yaml:"embedding_model"`
	IndexType           string        `yaml:"index_type"`
	MaxResults          int           `yaml:"max_results"`
	SimilarityThreshold float32       `yaml:"similarity_threshold"`
	EnableCache         bool          `yaml:"enable_cache"`
	CacheTTL            time.Duration `yaml:"cache_ttl"`
	EnableReranking     bool          `yaml:"enable_reranking"`
	RerankTopK          int           `yaml:"rerank_top_k"`
}

// VectorIndex interface for vector similarity search
type VectorIndex interface {
	Add(id string, vector []float32) error
	Search(query []float32, topK int) ([]SearchResult, error)
	Remove(id string) error
	Size() int
	Clear() error
}

// SearchResult represents a semantic search result
type SearchResult struct {
	ID          string  `json:"id"`
	Score       float32 `json:"score"`
	Content     string  `json:"content"`
	Metadata    map[string]interface{} `json:"metadata"`
	Embedding   []float32 `json:"embedding,omitempty"`
}

// SemanticStats tracks semantic search statistics
type SemanticStats struct {
	TotalSearches      int64         `json:"total_searches"`
	TotalIndexed       int64         `json:"total_indexed"`
	CacheHits          int64         `json:"cache_hits"`
	CacheMisses        int64         `json:"cache_misses"`
	AverageLatency     time.Duration `json:"average_latency"`
	TotalLatency       time.Duration `json:"total_latency"`
	IndexSize          int           `json:"index_size"`
	LastIndexUpdate    time.Time     `json:"last_index_update"`
}

// NewSemanticSearchEngine creates a new semantic search engine
func NewSemanticSearchEngine(config SemanticConfig) (*SemanticSearchEngine, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}

	// Initialize embedding generator
	embedderConfig := EmbeddingConfig{
		ModelName:          config.EmbeddingModel,
		EmbeddingDimension: 384,
		EnableCache:        config.EnableCache,
		CacheTTL:           config.CacheTTL,
	}
	
	embedder, err := NewEmbeddingGenerator(embedderConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	// Initialize Redis client
	var redisClient *redis.Client
	if config.EnableCache {
		redisClient = redis.NewClient(&redis.Options{
			Addr: "localhost:6379",
		})
	}

	// Initialize vector index
	var vectorIndex VectorIndex
	switch config.IndexType {
	case "faiss":
		vectorIndex, err = NewFAISSIndex(embedderConfig.EmbeddingDimension)
	case "hnsw":
		vectorIndex, err = NewHNSWIndex(embedderConfig.EmbeddingDimension)
	default:
		vectorIndex, err = NewInMemoryIndex(embedderConfig.EmbeddingDimension)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create vector index: %w", err)
	}

	return &SemanticSearchEngine{
		config:      config,
		embedder:    embedder,
		redisClient: redisClient,
		logger:      logger,
		vectorIndex: vectorIndex,
		stats:       &SemanticStats{},
	}, nil
}

// IndexDocument indexes a document for semantic search
func (sse *SemanticSearchEngine) IndexDocument(id, content string, metadata map[string]interface{}) error {
	// Generate embedding
	embedding, err := sse.embedder.GenerateEmbedding(content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Add to vector index
	if err := sse.vectorIndex.Add(id, embedding.Embedding); err != nil {
		return fmt.Errorf("failed to add to vector index: %w", err)
	}

	// Store document metadata
	if sse.redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		docData := map[string]interface{}{
			"id":       id,
			"content":  content,
			"metadata": metadata,
		}

		data, err := json.Marshal(docData)
		if err == nil {
			sse.redisClient.Set(ctx, "doc:"+id, data, sse.config.CacheTTL)
		}
	}

	sse.stats.TotalIndexed++
	sse.stats.IndexSize = sse.vectorIndex.Size()
	sse.stats.LastIndexUpdate = time.Now()

	return nil
}

// Search performs semantic search
func (sse *SemanticSearchEngine) Search(query string, topK int) ([]SearchResult, error) {
	start := time.Now()

	// Check cache first
	if sse.config.EnableCache && sse.redisClient != nil {
		if cached, err := sse.getCachedResults(query); err == nil && cached != nil {
			sse.stats.CacheHits++
			return cached, nil
		}
		sse.stats.CacheMisses++
	}

	// Generate query embedding
	queryEmbedding, err := sse.embedder.GenerateEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Search vector index
	results, err := sse.vectorIndex.Search(queryEmbedding.Embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// Filter by similarity threshold
	filteredResults := make([]SearchResult, 0)
	for _, result := range results {
		if result.Score >= sse.config.SimilarityThreshold {
			filteredResults = append(filteredResults, result)
		}
	}

	// Rerank if enabled
	if sse.config.EnableReranking && len(filteredResults) > sse.config.RerankTopK {
		filteredResults = sse.rerankResults(query, filteredResults[:sse.config.RerankTopK])
	}

	// Cache results
	if sse.config.EnableCache && sse.redisClient != nil {
		sse.cacheResults(query, filteredResults)
	}

	// Update statistics
	sse.updateStats(time.Since(start))

	return filteredResults, nil
}

// rerankResults reranks results using advanced algorithms
func (sse *SemanticSearchEngine) rerankResults(query string, results []SearchResult) []SearchResult {
	// Simplified reranking - in production, use more sophisticated algorithms
	queryEmbedding, err := sse.embedder.GenerateEmbedding(query)
	if err != nil {
		return results
	}

	// Calculate enhanced scores
	for i := range results {
		// Combine semantic similarity with other factors
		contentEmbedding, err := sse.embedder.GenerateEmbedding(results[i].Content)
		if err != nil {
			continue
		}

		similarity, err := sse.embedder.CalculateSimilarity(queryEmbedding.Embedding, contentEmbedding.Embedding)
		if err != nil {
			continue
		}

		// Apply boosting factors
		boost := float32(1.0)
		if metadata, ok := results[i].Metadata["quality_score"]; ok {
			if quality, ok := metadata.(float64); ok {
				boost *= float32(quality)
			}
		}

		if metadata, ok := results[i].Metadata["freshness"]; ok {
			if freshness, ok := metadata.(float64); ok {
				boost *= float32(freshness)
			}
		}

		results[i].Score = similarity * boost
	}

	// Sort by new scores
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score < results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results
}

// BatchIndex indexes multiple documents
func (sse *SemanticSearchEngine) BatchIndex(documents []Document) error {
	for _, doc := range documents {
		if err := sse.IndexDocument(doc.ID, doc.Content, doc.Metadata); err != nil {
			sse.logger.Error("Failed to index document", 
				zap.String("id", doc.ID), 
				zap.Error(err))
		}
	}
	return nil
}

// Document represents a document to be indexed
type Document struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
}

// Cache management
func (sse *SemanticSearchEngine) getCachedResults(query string) ([]SearchResult, error) {
	if sse.redisClient == nil {
		return nil, fmt.Errorf("Redis client not available")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	cached, err := sse.redisClient.Get(ctx, "search:"+query).Result()
	if err != nil {
		return nil, err
	}

	var results []SearchResult
	if err := json.Unmarshal([]byte(cached), &results); err != nil {
		return nil, err
	}

	return results, nil
}

func (sse *SemanticSearchEngine) cacheResults(query string, results []SearchResult) {
	if sse.redisClient == nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	data, err := json.Marshal(results)
	if err == nil {
		sse.redisClient.Set(ctx, "search:"+query, data, sse.config.CacheTTL)
	}
}

// Statistics
func (sse *SemanticSearchEngine) updateStats(latency time.Duration) {
	sse.stats.TotalSearches++
	sse.stats.TotalLatency += latency
	sse.stats.AverageLatency = time.Duration(int64(sse.stats.TotalLatency) / sse.stats.TotalSearches)
}

func (sse *SemanticSearchEngine) GetStats() *SemanticStats {
	sse.stats.IndexSize = sse.vectorIndex.Size()
	return sse.stats
}

// Vector index implementations
type InMemoryIndex struct {
	vectors map[string][]float32
	dimension int
	mu      sync.RWMutex
}

func NewInMemoryIndex(dimension int) (*InMemoryIndex, error) {
	return &InMemoryIndex{
		vectors:   make(map[string][]float32),
		dimension: dimension,
	}, nil
}

func (imi *InMemoryIndex) Add(id string, vector []float32) error {
	if len(vector) != imi.dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", imi.dimension, len(vector))
	}

	imi.mu.Lock()
	imi.vectors[id] = vector
	imi.mu.Unlock()

	return nil
}

func (imi *InMemoryIndex) Search(query []float32, topK int) ([]SearchResult, error) {
	if len(query) != imi.dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", imi.dimension, len(query))
	}

	imi.mu.RLock()
	defer imi.mu.RUnlock()

	results := make([]SearchResult, 0, len(imi.vectors))

	for id, vector := range imi.vectors {
		similarity := cosineSimilarity(query, vector)
		results = append(results, SearchResult{
			ID:    id,
			Score: similarity,
		})
	}

	// Sort by score (descending)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score < results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

func (imi *InMemoryIndex) Remove(id string) error {
	imi.mu.Lock()
	delete(imi.vectors, id)
	imi.mu.Unlock()
	return nil
}

func (imi *InMemoryIndex) Size() int {
	imi.mu.RLock()
	defer imi.mu.RUnlock()
	return len(imi.vectors)
}

func (imi *InMemoryIndex) Clear() error {
	imi.mu.Lock()
	imi.vectors = make(map[string][]float32)
	imi.mu.Unlock()
	return nil
}

// FAISS and HNSW implementations would be similar but use external libraries
type FAISSIndex struct {
	dimension int
	// FAISS index implementation would go here
}

func NewFAISSIndex(dimension int) (*FAISSIndex, error) {
	return &FAISSIndex{dimension: dimension}, nil
}

func (fi *FAISSIndex) Add(id string, vector []float32) error {
	// FAISS implementation
	return nil
}

func (fi *FAISSIndex) Search(query []float32, topK int) ([]SearchResult, error) {
	// FAISS implementation
	return []SearchResult{}, nil
}

func (fi *FAISSIndex) Remove(id string) error {
	return nil
}

func (fi *FAISSIndex) Size() int {
	return 0
}

func (fi *FAISSIndex) Clear() error {
	return nil
}

type HNSWIndex struct {
	dimension int
	// HNSW implementation would go here
}

func NewHNSWIndex(dimension int) (*HNSWIndex, error) {
	return &HNSWIndex{dimension: dimension}, nil
}

func (hi *HNSWIndex) Add(id string, vector []float32) error {
	// HNSW implementation
	return nil
}

func (hi *HNSWIndex) Search(query []float32, topK int) ([]SearchResult, error) {
	// HNSW implementation
	return []SearchResult{}, nil
}

func (hi *HNSWIndex) Remove(id string) error {
	return nil
}

func (hi *HNSWIndex) Size() int {
	return 0
}

func (hi *HNSWIndex) Clear() error {
	return nil
}

// Utility functions
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// Close closes the semantic search engine
func (sse *SemanticSearchEngine) Close() error {
	if sse.redisClient != nil {
		return sse.redisClient.Close()
	}
	return sse.embedder.Close()
}