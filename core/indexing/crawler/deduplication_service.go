// T3SS Project
// File: core/indexing/crawler/deduplication_service.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package crawler

import (
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
)

// DeduplicationService handles content deduplication
type DeduplicationService struct {
	redisClient *redis.Client
	logger      *zap.Logger
	mu          sync.RWMutex
}

// ContentFingerprint represents a content fingerprint
type ContentFingerprint struct {
	Hash        string    `json:"hash"`
	URL         string    `json:"url"`
	Title       string    `json:"title"`
	Content     string    `json:"content"`
	Similarity  float64   `json:"similarity"`
	CreatedAt   time.Time `json:"created_at"`
	LastSeen    time.Time `json:"last_seen"`
	Count       int       `json:"count"`
}

// SimilarityResult represents the result of similarity comparison
type SimilarityResult struct {
	IsDuplicate bool    `json:"is_duplicate"`
	Similarity  float64 `json:"similarity"`
	OriginalURL string  `json:"original_url"`
	Confidence  float64 `json:"confidence"`
}

// NewDeduplicationService creates a new deduplication service
func NewDeduplicationService(redisClient *redis.Client, logger *zap.Logger) *DeduplicationService {
	return &DeduplicationService{
		redisClient: redisClient,
		logger:      logger,
	}
}

// CheckDuplicate checks if content is a duplicate
func (ds *DeduplicationService) CheckDuplicate(url, title, content string) (*SimilarityResult, error) {
	// Generate content hash
	contentHash := ds.generateContentHash(content)
	
	// Check exact hash match first
	if result := ds.checkExactHash(contentHash, url); result != nil {
		return result, nil
	}
	
	// Check for near-duplicates using similarity
	return ds.checkSimilarity(url, title, content, contentHash)
}

// checkExactHash checks for exact content hash matches
func (ds *DeduplicationService) checkExactHash(hash, url string) *SimilarityResult {
	ctx := context.Background()
	
	// Check if hash exists in Redis
	existing, err := ds.redisClient.HGet(ctx, "content_hashes", hash).Result()
	if err != nil || existing == "" {
		return nil
	}
	
	// Parse existing fingerprint
	var fingerprint ContentFingerprint
	if err := json.Unmarshal([]byte(existing), &fingerprint); err != nil {
		return nil
	}
	
	// Update last seen and count
	fingerprint.LastSeen = time.Now()
	fingerprint.Count++
	
	// Store updated fingerprint
	fingerprintData, _ := json.Marshal(fingerprint)
	ds.redisClient.HSet(ctx, "content_hashes", hash, fingerprintData)
	
	return &SimilarityResult{
		IsDuplicate: true,
		Similarity:  1.0,
		OriginalURL: fingerprint.URL,
		Confidence:  1.0,
	}
}

// checkSimilarity checks for similar content using various algorithms
func (ds *DeduplicationService) checkSimilarity(url, title, content, contentHash string) (*SimilarityResult, error) {
	// Generate content fingerprint
	fingerprint := ds.generateFingerprint(content)
	
	// Check against existing fingerprints
	ctx := context.Background()
	existingHashes, err := ds.redisClient.HKeys(ctx, "content_hashes").Result()
	if err != nil {
		return nil, err
	}
	
	var bestMatch *SimilarityResult
	bestSimilarity := 0.0
	
	for _, hash := range existingHashes {
		existingData, err := ds.redisClient.HGet(ctx, "content_hashes", hash).Result()
		if err != nil {
			continue
		}
		
		var existingFingerprint ContentFingerprint
		if err := json.Unmarshal([]byte(existingData), &existingFingerprint); err != nil {
			continue
		}
		
		// Calculate similarity
		similarity := ds.calculateSimilarity(fingerprint, existingFingerprint)
		
		if similarity > bestSimilarity && similarity > 0.8 { // 80% similarity threshold
			bestSimilarity = similarity
			bestMatch = &SimilarityResult{
				IsDuplicate: true,
				Similarity:  similarity,
				OriginalURL: existingFingerprint.URL,
				Confidence:  ds.calculateConfidence(similarity, title, existingFingerprint.Title),
			}
		}
	}
	
	// Store new fingerprint if not a duplicate
	if bestMatch == nil {
		ds.storeFingerprint(url, title, content, contentHash, fingerprint)
	}
	
	return bestMatch, nil
}

// generateContentHash generates a hash for content deduplication
func (ds *DeduplicationService) generateContentHash(content string) string {
	// Normalize content for hashing
	normalized := ds.normalizeContent(content)
	
	// Generate SHA256 hash
	hash := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(hash[:])
}

// generateFingerprint generates a content fingerprint for similarity comparison
func (ds *DeduplicationService) generateFingerprint(content string) ContentFingerprint {
	// Extract key features
	words := ds.extractWords(content)
	ngrams := ds.generateNGrams(words, 3)
	
	// Create fingerprint
	fingerprint := ContentFingerprint{
		Hash:      ds.generateContentHash(content),
		Content:   content,
		CreatedAt: time.Now(),
		LastSeen:  time.Now(),
		Count:     1,
	}
	
	return fingerprint
}

// normalizeContent normalizes content for comparison
func (ds *DeduplicationService) normalizeContent(content string) string {
	// Convert to lowercase
	content = strings.ToLower(content)
	
	// Remove extra whitespace
	content = regexp.MustCompile(`\s+`).ReplaceAllString(content, " ")
	
	// Remove punctuation
	content = regexp.MustCompile(`[^\w\s]`).ReplaceAllString(content, "")
	
	// Remove common stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
		"with": true, "by": true, "is": true, "are": true, "was": true, "were": true,
		"be": true, "been": true, "have": true, "has": true, "had": true, "do": true,
		"does": true, "did": true, "will": true, "would": true, "could": true, "should": true,
	}
	
	words := strings.Fields(content)
	var filteredWords []string
	for _, word := range words {
		if !stopWords[word] && len(word) > 2 {
			filteredWords = append(filteredWords, word)
		}
	}
	
	return strings.Join(filteredWords, " ")
}

// extractWords extracts words from content
func (ds *DeduplicationService) extractWords(content string) []string {
	// Normalize and split into words
	normalized := ds.normalizeContent(content)
	return strings.Fields(normalized)
}

// generateNGrams generates n-grams from words
func (ds *DeduplicationService) generateNGrams(words []string, n int) []string {
	var ngrams []string
	
	for i := 0; i <= len(words)-n; i++ {
		ngram := strings.Join(words[i:i+n], " ")
		ngrams = append(ngrams, ngram)
	}
	
	return ngrams
}

// calculateSimilarity calculates similarity between two content fingerprints
func (ds *DeduplicationService) calculateSimilarity(fp1, fp2 ContentFingerprint) float64 {
	// Use multiple similarity metrics
	contentSim := ds.calculateContentSimilarity(fp1.Content, fp2.Content)
	
	// Weight the similarity
	return contentSim
}

// calculateContentSimilarity calculates content similarity using Jaccard similarity
func (ds *DeduplicationService) calculateContentSimilarity(content1, content2 string) float64 {
	// Extract words from both contents
	words1 := ds.extractWords(content1)
	words2 := ds.extractWords(content2)
	
	// Create word sets
	set1 := make(map[string]bool)
	set2 := make(map[string]bool)
	
	for _, word := range words1 {
		set1[word] = true
	}
	
	for _, word := range words2 {
		set2[word] = true
	}
	
	// Calculate intersection and union
	intersection := 0
	union := len(set1)
	
	for word := range set2 {
		if set1[word] {
			intersection++
		} else {
			union++
		}
	}
	
	if union == 0 {
		return 0.0
	}
	
	return float64(intersection) / float64(union)
}

// calculateConfidence calculates confidence in duplicate detection
func (ds *DeduplicationService) calculateConfidence(similarity float64, title1, title2 string) float64 {
	confidence := similarity
	
	// Boost confidence if titles are similar
	if ds.calculateTitleSimilarity(title1, title2) > 0.8 {
		confidence *= 1.2
	}
	
	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}
	
	return confidence
}

// calculateTitleSimilarity calculates title similarity
func (ds *DeduplicationService) calculateTitleSimilarity(title1, title2 string) float64 {
	// Normalize titles
	title1 = strings.ToLower(strings.TrimSpace(title1))
	title2 = strings.ToLower(strings.TrimSpace(title2))
	
	if title1 == title2 {
		return 1.0
	}
	
	// Use Jaccard similarity on words
	words1 := strings.Fields(title1)
	words2 := strings.Fields(title2)
	
	set1 := make(map[string]bool)
	set2 := make(map[string]bool)
	
	for _, word := range words1 {
		set1[word] = true
	}
	
	for _, word := range words2 {
		set2[word] = true
	}
	
	intersection := 0
	union := len(set1)
	
	for word := range set2 {
		if set1[word] {
			intersection++
		} else {
			union++
		}
	}
	
	if union == 0 {
		return 0.0
	}
	
	return float64(intersection) / float64(union)
}

// storeFingerprint stores a content fingerprint
func (ds *DeduplicationService) storeFingerprint(url, title, content, hash string, fingerprint ContentFingerprint) {
	ctx := context.Background()
	
	fingerprint.URL = url
	fingerprint.Title = title
	fingerprint.Content = content
	fingerprint.Hash = hash
	
	fingerprintData, err := json.Marshal(fingerprint)
	if err != nil {
		ds.logger.Error("Failed to marshal fingerprint", zap.Error(err))
		return
	}
	
	// Store in Redis with TTL
	ds.redisClient.HSet(ctx, "content_hashes", hash, fingerprintData)
	ds.redisClient.Expire(ctx, "content_hashes", 30*24*time.Hour) // 30 days TTL
}

// GetDuplicateStats returns statistics about duplicates
func (ds *DeduplicationService) GetDuplicateStats() (map[string]interface{}, error) {
	ctx := context.Background()
	
	// Get all fingerprints
	allHashes, err := ds.redisClient.HKeys(ctx, "content_hashes").Result()
	if err != nil {
		return nil, err
	}
	
	stats := map[string]interface{}{
		"total_fingerprints": len(allHashes),
		"duplicates_found":   0,
		"similarity_threshold": 0.8,
	}
	
	// Count duplicates (simplified - in production, you'd track this separately)
	duplicateCount := 0
	for _, hash := range allHashes {
		data, err := ds.redisClient.HGet(ctx, "content_hashes", hash).Result()
		if err != nil {
			continue
		}
		
		var fingerprint ContentFingerprint
		if err := json.Unmarshal([]byte(data), &fingerprint); err != nil {
			continue
		}
		
		if fingerprint.Count > 1 {
			duplicateCount++
		}
	}
	
	stats["duplicates_found"] = duplicateCount
	
	return stats, nil
}

// ClearCache clears the deduplication cache
func (ds *DeduplicationService) ClearCache() error {
	ctx := context.Background()
	return ds.redisClient.Del(ctx, "content_hashes").Err()
}