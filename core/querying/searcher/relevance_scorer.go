// T3SS Project
// File: core/querying/searcher/relevance_scorer.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package searcher

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
)

// RelevanceScore represents a document's relevance score with breakdown
type RelevanceScore struct {
	DocumentID     string                 `json:"document_id"`
	TotalScore     float64                `json:"total_score"`
	ScoreBreakdown map[string]float64     `json:"score_breakdown"`
	RankingFactors map[string]interface{} `json:"ranking_factors"`
	Timestamp      time.Time              `json:"timestamp"`
}

// RankingFactors contains all factors used in relevance scoring
type RankingFactors struct {
	// Text relevance factors
	TermFrequency    float64            `json:"term_frequency"`
	InverseDocFreq   float64            `json:"inverse_doc_freq"`
	FieldWeights     map[string]float64 `json:"field_weights"`
	QueryTermMatches int                `json:"query_term_matches"`
	TotalQueryTerms  int                `json:"total_query_terms"`

	// Document quality factors
	PageRank         float64 `json:"page_rank"`
	ClickThroughRate float64 `json:"click_through_rate"`
	BounceRate       float64 `json:"bounce_rate"`
	TimeOnPage       float64 `json:"time_on_page"`
	DocumentLength   int     `json:"document_length"`
	ContentQuality   float64 `json:"content_quality"`

	// Freshness factors
	PublishDate     time.Time `json:"publish_date"`
	LastModified    time.Time `json:"last_modified"`
	UpdateFrequency float64   `json:"update_frequency"`

	// User behavior factors
	UserEngagement  float64 `json:"user_engagement"`
	SocialSignals   float64 `json:"social_signals"`
	BacklinkCount   int     `json:"backlink_count"`
	DomainAuthority float64 `json:"domain_authority"`

	// Query-specific factors
	QueryIntent     string  `json:"query_intent"`
	QueryComplexity float64 `json:"query_complexity"`
	QueryLength     int     `json:"query_length"`

	// Contextual factors
	UserLocation string `json:"user_location"`
	UserLanguage string `json:"user_language"`
	DeviceType   string `json:"device_type"`
	TimeOfDay    int    `json:"time_of_day"`
	DayOfWeek    int    `json:"day_of_week"`
}

// ScoringWeights defines the weights for different scoring factors
type ScoringWeights struct {
	// Text relevance weights
	TermFrequencyWeight   float64 `json:"term_frequency_weight"`
	InverseDocFreqWeight  float64 `json:"inverse_doc_freq_weight"`
	FieldWeightBonus      float64 `json:"field_weight_bonus"`
	QueryCoverageWeight   float64 `json:"query_coverage_weight"`

	// Document quality weights
	PageRankWeight        float64 `json:"page_rank_weight"`
	ClickThroughWeight    float64 `json:"click_through_weight"`
	BounceRateWeight      float64 `json:"bounce_rate_weight"`
	TimeOnPageWeight      float64 `json:"time_on_page_weight"`
	ContentQualityWeight  float64 `json:"content_quality_weight"`

	// Freshness weights
	PublishDateWeight     float64 `json:"publish_date_weight"`
	LastModifiedWeight    float64 `json:"last_modified_weight"`
	UpdateFrequencyWeight float64 `json:"update_frequency_weight"`

	// User behavior weights
	UserEngagementWeight  float64 `json:"user_engagement_weight"`
	SocialSignalsWeight   float64 `json:"social_signals_weight"`
	BacklinkWeight        float64 `json:"backlink_weight"`
	DomainAuthorityWeight float64 `json:"domain_authority_weight"`

	// Query-specific weights
	QueryIntentWeight     float64 `json:"query_intent_weight"`
	QueryComplexityWeight float64 `json:"query_complexity_weight"`

	// Contextual weights
	LocationWeight        float64 `json:"location_weight"`
	LanguageWeight        float64 `json:"language_weight"`
	DeviceWeight          float64 `json:"device_weight"`
	TemporalWeight        float64 `json:"temporal_weight"`
}

// RelevanceScorerConfig contains configuration for the relevance scorer
type RelevanceScorerConfig struct {
	// Scoring weights
	Weights ScoringWeights `json:"weights"`

	// Cache configuration
	CacheEnabled bool          `json:"cache_enabled"`
	CacheTTL     time.Duration `json:"cache_ttl"`
	CacheMaxSize int           `json:"cache_max_size"`

	// Performance settings
	MaxConcurrency int           `json:"max_concurrency"`
	BatchSize      int           `json:"batch_size"`
	Timeout        time.Duration `json:"timeout"`

	// Feature flags
	EnableMLScoring      bool `json:"enable_ml_scoring"`
	EnableRealTime       bool `json:"enable_real_time"`
	EnablePersonalization bool `json:"enable_personalization"`
}

// RelevanceScorer provides advanced relevance scoring capabilities
type RelevanceScorer struct {
	config     RelevanceScorerConfig
	logger     *zap.Logger
	redis      *redis.Client
	cache      map[string]*RelevanceScore
	cacheMutex sync.RWMutex

	// Statistics
	stats struct {
		sync.RWMutex
		TotalQueries     int64   `json:"total_queries"`
		TotalDocuments   int64   `json:"total_documents"`
		CacheHits        int64   `json:"cache_hits"`
		CacheMisses      int64   `json:"cache_misses"`
		AverageScoreTime float64 `json:"average_score_time"`
		TotalScoreTime   float64 `json:"total_score_time"`
		ErrorCount       int64   `json:"error_count"`
	}
}

// NewRelevanceScorer creates a new relevance scorer instance
func NewRelevanceScorer(config RelevanceScorerConfig, logger *zap.Logger, redis *redis.Client) *RelevanceScorer {
	// Set default weights if not provided
	if config.Weights.TermFrequencyWeight == 0 {
		config.Weights = getDefaultWeights()
	}

	// Set default configuration
	if config.CacheTTL == 0 {
		config.CacheTTL = 1 * time.Hour
	}
	if config.MaxConcurrency == 0 {
		config.MaxConcurrency = 100
	}
	if config.BatchSize == 0 {
		config.BatchSize = 1000
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	return &RelevanceScorer{
		config: config,
		logger: logger,
		redis:  redis,
		cache:  make(map[string]*RelevanceScore),
	}
}

// getDefaultWeights returns default scoring weights
func getDefaultWeights() ScoringWeights {
	return ScoringWeights{
		// Text relevance weights
		TermFrequencyWeight:   0.25,
		InverseDocFreqWeight:  0.20,
		FieldWeightBonus:      0.15,
		QueryCoverageWeight:   0.10,

		// Document quality weights
		PageRankWeight:        0.15,
		ClickThroughWeight:    0.08,
		BounceRateWeight:      0.05,
		TimeOnPageWeight:      0.05,
		ContentQualityWeight:  0.10,

		// Freshness weights
		PublishDateWeight:     0.05,
		LastModifiedWeight:    0.03,
		UpdateFrequencyWeight: 0.02,

		// User behavior weights
		UserEngagementWeight:  0.08,
		SocialSignalsWeight:   0.05,
		BacklinkWeight:        0.07,
		DomainAuthorityWeight: 0.10,

		// Query-specific weights
		QueryIntentWeight:     0.12,
		QueryComplexityWeight: 0.05,

		// Contextual weights
		LocationWeight: 0.03,
		LanguageWeight: 0.02,
		DeviceWeight:   0.01,
		TemporalWeight: 0.01,
	}
}

// Document represents a document to be scored
type Document struct {
	ID              string            `json:"id"`
	Title           string            `json:"title"`
	Content         string            `json:"content"`
	ContentLength   int               `json:"content_length"`
	ContentType     string            `json:"content_type"`
	Language        string            `json:"language"`
	Location        string            `json:"location"`
	MobileOptimized bool              `json:"mobile_optimized"`
	AuthorityScore  float64           `json:"authority_score"`
	Fields          map[string]string `json:"fields"`
}

// ScoreDocuments calculates relevance scores for multiple documents
func (rs *RelevanceScorer) ScoreDocuments(ctx context.Context, query string, documents []Document, factors []RankingFactors) ([]RelevanceScore, error) {
	start := time.Now()
	defer func() {
		rs.updateStats(time.Since(start).Seconds(), len(documents))
	}()

	rs.stats.Lock()
	rs.stats.TotalQueries++
	rs.stats.TotalDocuments += int64(len(documents))
	rs.stats.Unlock()

	// Validate inputs
	if len(documents) != len(factors) {
		return nil, fmt.Errorf("documents and factors length mismatch")
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, rs.config.Timeout)
	defer cancel()

	// Score documents in parallel
	scores := make([]RelevanceScore, len(documents))
	semaphore := make(chan struct{}, rs.config.MaxConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstError error

	for i, doc := range documents {
		wg.Add(1)
		go func(index int, document Document, factor RankingFactors) {
			defer wg.Done()

			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				return
			}

			score, err := rs.scoreDocument(ctx, query, document, factor)
			if err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = err
				}
				mu.Unlock()
				return
			}

			mu.Lock()
			scores[index] = score
			mu.Unlock()
		}(i, doc, factors[i])
	}

	wg.Wait()

	if firstError != nil {
		return nil, firstError
	}

	// Sort by relevance score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].TotalScore > scores[j].TotalScore
	})

	return scores, nil
}

// scoreDocument calculates relevance score for a single document
func (rs *RelevanceScorer) scoreDocument(ctx context.Context, query string, document Document, factors RankingFactors) (RelevanceScore, error) {
	// Check cache first
	if rs.config.CacheEnabled {
		if cached := rs.getCachedScore(query, document.ID); cached != nil {
			rs.stats.Lock()
			rs.stats.CacheHits++
			rs.stats.Unlock()
			return *cached, nil
		}
		rs.stats.Lock()
		rs.stats.CacheMisses++
		rs.stats.Unlock()
	}

	scoreBreakdown := make(map[string]float64)
	rankingFactors := make(map[string]interface{})

	// Calculate text relevance score
	textScore := rs.calculateTextRelevance(query, document, factors)
	scoreBreakdown["text_relevance"] = textScore

	// Calculate document quality score
	qualityScore := rs.calculateDocumentQuality(document, factors)
	scoreBreakdown["document_quality"] = qualityScore

	// Calculate freshness score
	freshnessScore := rs.calculateFreshnessScore(document, factors)
	scoreBreakdown["freshness"] = freshnessScore

	// Calculate user behavior score
	behaviorScore := rs.calculateUserBehaviorScore(document, factors)
	scoreBreakdown["user_behavior"] = behaviorScore

	// Calculate query-specific score
	queryScore := rs.calculateQuerySpecificScore(query, document, factors)
	scoreBreakdown["query_specific"] = queryScore

	// Calculate contextual score
	contextScore := rs.calculateContextualScore(document, factors)
	scoreBreakdown["contextual"] = contextScore

	// Calculate total weighted score
	totalScore := rs.calculateWeightedScore(scoreBreakdown)

	// Store ranking factors for analysis
	rankingFactors["query"] = query
	rankingFactors["document_id"] = document.ID
	rankingFactors["factors"] = factors

	score := RelevanceScore{
		DocumentID:     document.ID,
		TotalScore:     totalScore,
		ScoreBreakdown: scoreBreakdown,
		RankingFactors: rankingFactors,
		Timestamp:      time.Now(),
	}

	// Cache the score
	if rs.config.CacheEnabled {
		rs.cacheScore(query, document.ID, &score)
	}

	return score, nil
}

// calculateTextRelevance calculates text-based relevance score
func (rs *RelevanceScorer) calculateTextRelevance(query string, document Document, factors RankingFactors) float64 {
	score := 0.0

	// Term frequency score
	tfScore := rs.calculateTermFrequencyScore(query, document, factors)
	score += tfScore * rs.config.Weights.TermFrequencyWeight

	// Inverse document frequency score
	idfScore := rs.calculateInverseDocFreqScore(query, document, factors)
	score += idfScore * rs.config.Weights.InverseDocFreqWeight

	// Field weight bonus
	fieldScore := rs.calculateFieldWeightScore(document, factors)
	score += fieldScore * rs.config.Weights.FieldWeightBonus

	// Query coverage score
	coverageScore := rs.calculateQueryCoverageScore(query, document, factors)
	score += coverageScore * rs.config.Weights.QueryCoverageWeight

	return score
}

// calculateTermFrequencyScore calculates TF-based score
func (rs *RelevanceScorer) calculateTermFrequencyScore(query string, document Document, factors RankingFactors) float64 {
	if factors.TermFrequency == 0 {
		return 0.0
	}

	// Use log normalization for TF
	return 1.0 + math.Log(factors.TermFrequency)
}

// calculateInverseDocFreqScore calculates IDF-based score
func (rs *RelevanceScorer) calculateInverseDocFreqScore(query string, document Document, factors RankingFactors) float64 {
	if factors.InverseDocFreq == 0 {
		return 0.0
	}

	return factors.InverseDocFreq
}

// calculateFieldWeightScore calculates field-specific weight score
func (rs *RelevanceScorer) calculateFieldWeightScore(document Document, factors RankingFactors) float64 {
	score := 0.0

	for field, weight := range factors.FieldWeights {
		// Apply field-specific boosting
		switch field {
		case "title":
			score += weight * 2.0 // Title is very important
		case "heading":
			score += weight * 1.5 // Headings are important
		case "content":
			score += weight * 1.0 // Content is baseline
		case "meta":
			score += weight * 0.8 // Meta tags are less important
		default:
			score += weight * 0.5 // Other fields are least important
		}
	}

	return score
}

// calculateQueryCoverageScore calculates how well the document covers the query
func (rs *RelevanceScorer) calculateQueryCoverageScore(query string, document Document, factors RankingFactors) float64 {
	if factors.TotalQueryTerms == 0 {
		return 0.0
	}

	coverage := float64(factors.QueryTermMatches) / float64(factors.TotalQueryTerms)
	return coverage
}

// calculateDocumentQuality calculates document quality score
func (rs *RelevanceScorer) calculateDocumentQuality(document Document, factors RankingFactors) float64 {
	score := 0.0

	// PageRank score
	score += factors.PageRank * rs.config.Weights.PageRankWeight

	// Click-through rate score
	score += factors.ClickThroughRate * rs.config.Weights.ClickThroughWeight

	// Bounce rate score (inverted - lower is better)
	bounceScore := 1.0 - factors.BounceRate
	score += bounceScore * rs.config.Weights.BounceRateWeight

	// Time on page score
	timeScore := math.Min(factors.TimeOnPage/300.0, 1.0) // Normalize to 5 minutes
	score += timeScore * rs.config.Weights.TimeOnPageWeight

	// Content quality score
	score += factors.ContentQuality * rs.config.Weights.ContentQualityWeight

	return score
}

// calculateFreshnessScore calculates document freshness score
func (rs *RelevanceScorer) calculateFreshnessScore(document Document, factors RankingFactors) float64 {
	score := 0.0

	// Publish date score
	publishScore := rs.calculateDateScore(factors.PublishDate)
	score += publishScore * rs.config.Weights.PublishDateWeight

	// Last modified score
	modifiedScore := rs.calculateDateScore(factors.LastModified)
	score += modifiedScore * rs.config.Weights.LastModifiedWeight

	// Update frequency score
	score += factors.UpdateFrequency * rs.config.Weights.UpdateFrequencyWeight

	return score
}

// calculateDateScore calculates score based on document date
func (rs *RelevanceScorer) calculateDateScore(date time.Time) float64 {
	if date.IsZero() {
		return 0.0
	}

	daysSince := time.Since(date).Hours() / 24.0

	// Exponential decay for freshness
	decayRate := 0.01 // 1% decay per day
	return math.Exp(-decayRate * daysSince)
}

// calculateUserBehaviorScore calculates user behavior-based score
func (rs *RelevanceScorer) calculateUserBehaviorScore(document Document, factors RankingFactors) float64 {
	score := 0.0

	// User engagement score
	score += factors.UserEngagement * rs.config.Weights.UserEngagementWeight

	// Social signals score
	score += factors.SocialSignals * rs.config.Weights.SocialSignalsWeight

	// Backlink count score (log scale)
	if factors.BacklinkCount > 0 {
		backlinkScore := math.Log(float64(factors.BacklinkCount) + 1)
		score += backlinkScore * rs.config.Weights.BacklinkWeight
	}

	// Domain authority score
	score += factors.DomainAuthority * rs.config.Weights.DomainAuthorityWeight

	return score
}

// calculateQuerySpecificScore calculates query-specific score
func (rs *RelevanceScorer) calculateQuerySpecificScore(query string, document Document, factors RankingFactors) float64 {
	score := 0.0

	// Query intent score
	intentScore := rs.calculateIntentScore(factors.QueryIntent, document)
	score += intentScore * rs.config.Weights.QueryIntentWeight

	// Query complexity score
	complexityScore := rs.calculateComplexityScore(factors.QueryComplexity, document)
	score += complexityScore * rs.config.Weights.QueryComplexityWeight

	return score
}

// calculateIntentScore calculates score based on query intent
func (rs *RelevanceScorer) calculateIntentScore(intent string, document Document) float64 {
	// Map query intent to document type relevance
	switch intent {
	case "informational":
		// Prefer longer, more detailed content
		return math.Min(float64(document.ContentLength)/1000.0, 1.0)
	case "navigational":
		// Prefer exact matches and authoritative sources
		return document.AuthorityScore
	case "transactional":
		// Prefer commercial content
		if strings.Contains(strings.ToLower(document.Content), "buy") ||
			strings.Contains(strings.ToLower(document.Content), "price") ||
			strings.Contains(strings.ToLower(document.Content), "shop") {
			return 1.0
		}
		return 0.5
	default:
		return 0.5
	}
}

// calculateComplexityScore calculates score based on query complexity
func (rs *RelevanceScorer) calculateComplexityScore(complexity float64, document Document) float64 {
	// For complex queries, prefer more comprehensive content
	if complexity > 0.7 {
		return math.Min(float64(document.ContentLength)/2000.0, 1.0)
	}
	return 0.5
}

// calculateContextualScore calculates contextual score
func (rs *RelevanceScorer) calculateContextualScore(document Document, factors RankingFactors) float64 {
	score := 0.0

	// Location score
	locationScore := rs.calculateLocationScore(factors.UserLocation, document)
	score += locationScore * rs.config.Weights.LocationWeight

	// Language score
	languageScore := rs.calculateLanguageScore(factors.UserLanguage, document)
	score += languageScore * rs.config.Weights.LanguageWeight

	// Device score
	deviceScore := rs.calculateDeviceScore(factors.DeviceType, document)
	score += deviceScore * rs.config.Weights.DeviceWeight

	// Temporal score
	temporalScore := rs.calculateTemporalScore(factors.TimeOfDay, factors.DayOfWeek, document)
	score += temporalScore * rs.config.Weights.TemporalWeight

	return score
}

// calculateLocationScore calculates location-based score
func (rs *RelevanceScorer) calculateLocationScore(userLocation string, document Document) float64 {
	// Simple location matching - in real implementation, use geolocation data
	if userLocation == "" || document.Location == "" {
		return 0.5
	}

	if strings.EqualFold(userLocation, document.Location) {
		return 1.0
	}

	return 0.3
}

// calculateLanguageScore calculates language-based score
func (rs *RelevanceScorer) calculateLanguageScore(userLanguage string, document Document) float64 {
	if userLanguage == "" || document.Language == "" {
		return 0.5
	}

	if strings.EqualFold(userLanguage, document.Language) {
		return 1.0
	}

	return 0.2
}

// calculateDeviceScore calculates device-based score
func (rs *RelevanceScorer) calculateDeviceScore(deviceType string, document Document) float64 {
	// Mobile-optimized content gets higher score for mobile users
	if deviceType == "mobile" && document.MobileOptimized {
		return 1.0
	}

	return 0.5
}

// calculateTemporalScore calculates time-based score
func (rs *RelevanceScorer) calculateTemporalScore(timeOfDay, dayOfWeek int, document Document) float64 {
	// Simple temporal scoring - in real implementation, use more sophisticated logic
	score := 0.5

	// Boost news content during business hours
	if timeOfDay >= 9 && timeOfDay <= 17 && document.ContentType == "news" {
		score += 0.3
	}

	// Boost entertainment content on weekends
	if dayOfWeek >= 6 && document.ContentType == "entertainment" {
		score += 0.2
	}

	return math.Min(score, 1.0)
}

// calculateWeightedScore calculates final weighted score
func (rs *RelevanceScorer) calculateWeightedScore(scoreBreakdown map[string]float64) float64 {
	totalScore := 0.0

	for factor, score := range scoreBreakdown {
		// Apply factor-specific weights
		weight := rs.getFactorWeight(factor)
		totalScore += score * weight
	}

	// Normalize to 0-1 range
	return math.Min(totalScore, 1.0)
}

// getFactorWeight returns weight for a specific factor
func (rs *RelevanceScorer) getFactorWeight(factor string) float64 {
	switch factor {
	case "text_relevance":
		return 0.4
	case "document_quality":
		return 0.25
	case "freshness":
		return 0.15
	case "user_behavior":
		return 0.10
	case "query_specific":
		return 0.05
	case "contextual":
		return 0.05
	default:
		return 0.0
	}
}

// getCachedScore retrieves cached score
func (rs *RelevanceScorer) getCachedScore(query, documentID string) *RelevanceScore {
	rs.cacheMutex.RLock()
	defer rs.cacheMutex.RUnlock()

	key := rs.generateCacheKey(query, documentID)
	return rs.cache[key]
}

// cacheScore stores score in cache
func (rs *RelevanceScorer) cacheScore(query, documentID string, score *RelevanceScore) {
	rs.cacheMutex.Lock()
	defer rs.cacheMutex.Unlock()

	// Check cache size limit
	if len(rs.cache) >= rs.config.CacheMaxSize {
		rs.evictOldestCacheEntry()
	}

	key := rs.generateCacheKey(query, documentID)
	rs.cache[key] = score
}

// generateCacheKey generates cache key for query and document
func (rs *RelevanceScorer) generateCacheKey(query, documentID string) string {
	return query + ":" + documentID
}

// evictOldestCacheEntry removes oldest cache entry
func (rs *RelevanceScorer) evictOldestCacheEntry() {
	var oldestKey string
	var oldestTime time.Time

	for key, score := range rs.cache {
		if oldestKey == "" || score.Timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = score.Timestamp
		}
	}

	if oldestKey != "" {
		delete(rs.cache, oldestKey)
	}
}

// updateStats updates scorer statistics
func (rs *RelevanceScorer) updateStats(scoreTime float64, documentCount int) {
	rs.stats.Lock()
	defer rs.stats.Unlock()

	rs.stats.TotalScoreTime += scoreTime
	rs.stats.AverageScoreTime = rs.stats.TotalScoreTime / float64(rs.stats.TotalQueries)
}

// GetStatistics returns scorer statistics
func (rs *RelevanceScorer) GetStatistics() map[string]interface{} {
	rs.stats.RLock()
	defer rs.stats.RUnlock()

	rs.cacheMutex.RLock()
	cacheSize := len(rs.cache)
	rs.cacheMutex.RUnlock()

	return map[string]interface{}{
		"total_queries":       rs.stats.TotalQueries,
		"total_documents":     rs.stats.TotalDocuments,
		"cache_hits":          rs.stats.CacheHits,
		"cache_misses":        rs.stats.CacheMisses,
		"cache_hit_rate":      float64(rs.stats.CacheHits) / float64(rs.stats.CacheHits+rs.stats.CacheMisses),
		"average_score_time":  rs.stats.AverageScoreTime,
		"total_score_time":    rs.stats.TotalScoreTime,
		"error_count":         rs.stats.ErrorCount,
		"cache_size":          cacheSize,
		"cache_max_size":      rs.config.CacheMaxSize,
	}
}

// ClearCache clears the score cache
func (rs *RelevanceScorer) ClearCache() {
	rs.cacheMutex.Lock()
	defer rs.cacheMutex.Unlock()

	rs.cache = make(map[string]*RelevanceScore)
}

// UpdateWeights updates scoring weights
func (rs *RelevanceScorer) UpdateWeights(weights ScoringWeights) {
	rs.config.Weights = weights
}

// GetWeights returns current scoring weights
func (rs *RelevanceScorer) GetWeights() ScoringWeights {
	return rs.config.Weights
}

