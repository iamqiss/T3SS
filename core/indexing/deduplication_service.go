// T3SS Project
// File: core/indexing/deduplication_service.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package indexing

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"strings"
	"sync"
	"time"
)

// Document represents a document to be deduplicated
type Document struct {
	URL         string
	Content     string
	Title       string
	ContentHash string
	URLHash     string
	Timestamp   time.Time
	Size        int64
}

// DuplicateInfo contains information about duplicate detection
type DuplicateInfo struct {
	IsDuplicate bool
	OriginalURL string
	Similarity  float64
	Reason      string
}

// DeduplicationService provides high-performance document deduplication
type DeduplicationService struct {
	contentHashes map[string]string // content hash -> URL
	urlHashes     map[string]string // URL hash -> URL
	similarityCache map[string][]string // content hash -> similar URLs
	mu            sync.RWMutex
	config        DeduplicationConfig
}

// DeduplicationConfig holds configuration for deduplication
type DeduplicationConfig struct {
	MinContentLength    int     // Minimum content length to consider
	SimilarityThreshold float64 // Threshold for considering documents similar
	CacheSize          int     // Maximum number of hashes to cache
	EnableSimilarity   bool    // Enable similarity-based deduplication
}

// NewDeduplicationService creates a new deduplication service
func NewDeduplicationService(config DeduplicationConfig) *DeduplicationService {
	return &DeduplicationService{
		contentHashes:   make(map[string]string),
		urlHashes:       make(map[string]string),
		similarityCache: make(map[string][]string),
		config:          config,
	}
}

// CheckDuplicate checks if a document is a duplicate
func (ds *DeduplicationService) CheckDuplicate(doc *Document) *DuplicateInfo {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Generate hashes
	contentHash := ds.generateContentHash(doc.Content)
	urlHash := ds.generateURLHash(doc.URL)

	// Check for exact content match
	if originalURL, exists := ds.contentHashes[contentHash]; exists {
		return &DuplicateInfo{
			IsDuplicate: true,
			OriginalURL: originalURL,
			Similarity:  1.0,
			Reason:      "exact_content_match",
		}
	}

	// Check for exact URL match (redirects, etc.)
	if originalURL, exists := ds.urlHashes[urlHash]; exists {
		return &DuplicateInfo{
			IsDuplicate: true,
			OriginalURL: originalURL,
			Similarity:  1.0,
			Reason:      "exact_url_match",
		}
	}

	// Check for similar content if enabled
	if ds.config.EnableSimilarity {
		if similarURLs, exists := ds.similarityCache[contentHash]; exists {
			for _, similarURL := range similarURLs {
				similarity := ds.calculateSimilarity(doc.Content, ds.getContentByURL(similarURL))
				if similarity >= ds.config.SimilarityThreshold {
					return &DuplicateInfo{
						IsDuplicate: true,
						OriginalURL: similarURL,
						Similarity:  similarity,
						Reason:      "similar_content",
					}
				}
			}
		}
	}

	// Not a duplicate - add to cache
	ds.addToCache(doc, contentHash, urlHash)

	return &DuplicateInfo{
		IsDuplicate: false,
		Similarity:  0.0,
		Reason:      "unique",
	}
}

// generateContentHash generates a hash for document content
func (ds *DeduplicationService) generateContentHash(content string) string {
	// Use SHA-256 for content hashing
	hash := sha256.Sum256([]byte(content))
	return hex.EncodeToString(hash[:])
}

// generateURLHash generates a hash for URL
func (ds *DeduplicationService) generateURLHash(url string) string {
	// Use FNV-1a for fast URL hashing
	h := fnv.New32a()
	h.Write([]byte(url))
	return fmt.Sprintf("%x", h.Sum32())
}

// calculateSimilarity calculates content similarity using Jaccard similarity
func (ds *DeduplicationService) calculateSimilarity(content1, content2 string) float64 {
	if len(content1) == 0 || len(content2) == 0 {
		return 0.0
	}

	// Create shingle sets (3-grams)
	shingles1 := ds.createShingles(content1)
	shingles2 := ds.createShingles(content2)

	// Calculate Jaccard similarity
	intersection := 0
	union := make(map[string]bool)

	for shingle := range shingles1 {
		union[shingle] = true
		if shingles2[shingle] {
			intersection++
		}
	}

	for shingle := range shingles2 {
		union[shingle] = true
	}

	if len(union) == 0 {
		return 0.0
	}

	return float64(intersection) / float64(len(union))
}

// createShingles creates 3-gram shingles from content
func (ds *DeduplicationService) createShingles(content string) map[string]bool {
	shingles := make(map[string]bool)
	
	// Normalize content
	normalized := ds.normalizeContent(content)
	
	// Create 3-grams
	for i := 0; i <= len(normalized)-3; i++ {
		shingle := normalized[i : i+3]
		shingles[shingle] = true
	}
	
	return shingles
}

// normalizeContent normalizes content for comparison
func (ds *DeduplicationService) normalizeContent(content string) string {
	// Convert to lowercase and remove extra whitespace
	normalized := strings.ToLower(content)
	normalized = strings.ReplaceAll(normalized, "\n", " ")
	normalized = strings.ReplaceAll(normalized, "\t", " ")
	
	// Remove multiple spaces
	for strings.Contains(normalized, "  ") {
		normalized = strings.ReplaceAll(normalized, "  ", " ")
	}
	
	return strings.TrimSpace(normalized)
}

// addToCache adds a document to the deduplication cache
func (ds *DeduplicationService) addToCache(doc *Document, contentHash, urlHash string) {
	// Check cache size limits
	if len(ds.contentHashes) >= ds.config.CacheSize {
		ds.evictOldEntries()
	}

	// Add to caches
	ds.contentHashes[contentHash] = doc.URL
	ds.urlHashes[urlHash] = doc.URL

	// Add to similarity cache if enabled
	if ds.config.EnableSimilarity {
		ds.similarityCache[contentHash] = append(ds.similarityCache[contentHash], doc.URL)
	}
}

// evictOldEntries evicts old entries from cache (simple LRU approximation)
func (ds *DeduplicationService) evictOldEntries() {
	// Remove 10% of entries (simplified eviction)
	evictCount := len(ds.contentHashes) / 10
	if evictCount == 0 {
		evictCount = 1
	}

	count := 0
	for contentHash := range ds.contentHashes {
		if count >= evictCount {
			break
		}
		delete(ds.contentHashes, contentHash)
		count++
	}

	// Also clean up similarity cache
	for contentHash := range ds.similarityCache {
		if count >= evictCount {
			break
		}
		delete(ds.similarityCache, contentHash)
		count++
	}
}

// getContentByURL retrieves content by URL (simplified - in production, you'd have a document store)
func (ds *DeduplicationService) getContentByURL(url string) string {
	// This is a placeholder - in production, you'd query your document store
	// For now, return empty string
	return ""
}

// GetStats returns deduplication statistics
func (ds *DeduplicationService) GetStats() map[string]interface{} {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	return map[string]interface{}{
		"content_hashes":    len(ds.contentHashes),
		"url_hashes":        len(ds.urlHashes),
		"similarity_cache":  len(ds.similarityCache),
		"cache_size_limit":  ds.config.CacheSize,
		"similarity_enabled": ds.config.EnableSimilarity,
	}
}

// ClearCache clears all deduplication caches
func (ds *DeduplicationService) ClearCache() {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	ds.contentHashes = make(map[string]string)
	ds.urlHashes = make(map[string]string)
	ds.similarityCache = make(map[string][]string)
}

// BatchCheckDuplicates checks multiple documents for duplicates efficiently
func (ds *DeduplicationService) BatchCheckDuplicates(docs []*Document) []*DuplicateInfo {
	results := make([]*DuplicateInfo, len(docs))
	
	// Process in parallel for better performance
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 100) // Limit concurrent goroutines
	
	for i, doc := range docs {
		wg.Add(1)
		go func(index int, document *Document) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			results[index] = ds.CheckDuplicate(document)
		}(i, doc)
	}
	
	wg.Wait()
	return results
}