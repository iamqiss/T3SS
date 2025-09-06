// T3SS Project
// File: core/graph_core/link_graph_builder.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package graph_core

import (
	"context"
	"crypto/md5"
	"fmt"
	"log"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
)

// LinkGraphConfig holds configuration for link graph building
type LinkGraphConfig struct {
	// Graph construction parameters
	MaxNodes                int           `yaml:"max_nodes"`
	MaxEdges                int           `yaml:"max_edges"`
	MinLinkWeight           float64       `yaml:"min_link_weight"`
	MaxLinkWeight           float64       `yaml:"max_link_weight"`
	
	// Link analysis parameters
	EnableAnchorTextAnalysis bool         `yaml:"enable_anchor_text_analysis"`
	EnableLinkTypeDetection  bool         `yaml:"enable_link_type_detection"`
	EnableSpamDetection     bool         `yaml:"enable_spam_detection"`
	EnableDuplicateDetection bool        `yaml:"enable_duplicate_detection"`
	
	// Performance tuning
	BatchSize               int           `yaml:"batch_size"`
	WorkerCount             int           `yaml:"worker_count"`
	CacheSize               int           `yaml:"cache_size"`
	EnableParallelProcessing bool         `yaml:"enable_parallel_processing"`
	
	// Quality control
	MinAnchorTextLength     int           `yaml:"min_anchor_text_length"`
	MaxAnchorTextLength     int           `yaml:"max_anchor_text_length"`
	AllowedDomains          []string      `yaml:"allowed_domains"`
	BlockedDomains          []string      `yaml:"blocked_domains"`
	
	// Storage configuration
	RedisEndpoint           string        `yaml:"redis_endpoint"`
	EnablePersistence       bool          `yaml:"enable_persistence"`
	PersistenceInterval     time.Duration `yaml:"persistence_interval"`
}

// Link represents a link between web pages
type Link struct {
	ID              string    `json:"id"`
	FromURL         string    `json:"from_url"`
	ToURL           string    `json:"to_url"`
	AnchorText      string    `json:"anchor_text"`
	LinkType        LinkType  `json:"link_type"`
	Weight          float64   `json:"weight"`
	Confidence      float64   `json:"confidence"`
	DiscoveredAt    time.Time `json:"discovered_at"`
	LastSeen        time.Time `json:"last_seen"`
	ClickCount      int64     `json:"click_count"`
	IsSpam          bool      `json:"is_spam"`
	IsDuplicate     bool      `json:"is_duplicate"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// LinkType represents different types of links
type LinkType string

const (
	LinkTypeInternal     LinkType = "internal"
	LinkTypeExternal     LinkType = "external"
	LinkTypeNavigational LinkType = "navigational"
	LinkTypeContent      LinkType = "content"
	LinkTypeSponsored    LinkType = "sponsored"
	LinkTypeSocial       LinkType = "social"
	LinkTypeReference    LinkType = "reference"
	LinkTypeUnknown      LinkType = "unknown"
)

// WebPage represents a web page in the link graph
type WebPage struct {
	ID              string                 `json:"id"`
	URL             string                 `json:"url"`
	Domain          string                 `json:"domain"`
	Title           string                 `json:"title"`
	ContentLength   int64                  `json:"content_length"`
	LastCrawled     time.Time              `json:"last_crawled"`
	InLinkCount     int64                  `json:"in_link_count"`
	OutLinkCount    int64                  `json:"out_link_count"`
	PageRank        float64                `json:"page_rank"`
	AuthorityScore  float64                `json:"authority_score"`
	HubScore        float64                `json:"hub_score"`
	QualityScore    float64                `json:"quality_score"`
	SpamScore       float64                `json:"spam_score"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// LinkGraph represents the complete link graph
type LinkGraph struct {
	Pages           map[string]*WebPage    `json:"pages"`
	Links           map[string]*Link       `json:"links"`
	DomainGraph     map[string][]string    `json:"domain_graph"`
	AnchorTextIndex map[string][]string    `json:"anchor_text_index"`
	Stats           *LinkGraphStats        `json:"stats"`
}

// LinkGraphStats tracks link graph statistics
type LinkGraphStats struct {
	TotalPages           int64     `json:"total_pages"`
	TotalLinks           int64     `json:"total_links"`
	TotalDomains         int64     `json:"total_domains"`
	AverageInDegree      float64   `json:"average_in_degree"`
	AverageOutDegree     float64   `json:"average_out_degree"`
	MaxInDegree          int64     `json:"max_in_degree"`
	MaxOutDegree         int64     `json:"max_out_degree"`
	GraphDensity         float64   `json:"graph_density"`
	LastUpdated          time.Time `json:"last_updated"`
	ProcessingTime       time.Duration `json:"processing_time"`
	LinksProcessed       int64     `json:"links_processed"`
	SpamLinksDetected    int64     `json:"spam_links_detected"`
	DuplicateLinksFound  int64     `json:"duplicate_links_found"`
}

// LinkGraphBuilder builds and maintains the link graph
type LinkGraphBuilder struct {
	config     LinkGraphConfig
	logger     *zap.Logger
	redisClient *redis.Client
	
	// Graph data structures
	graph      *LinkGraph
	graphMutex sync.RWMutex
	
	// Processing queues
	linkQueue  chan *Link
	pageQueue  chan *WebPage
	
	// Caches and indexes
	urlCache   map[string]bool
	domainCache map[string]bool
	anchorTextCache map[string]int
	
	// Workers and synchronization
	workers    []*LinkProcessor
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
	
	// Statistics
	stats      *LinkGraphStats
	statsMutex sync.RWMutex
	
	// Spam detection
	spamDetector *SpamDetector
	duplicateDetector *DuplicateDetector
}

// LinkProcessor processes individual links
type LinkProcessor struct {
	id       int
	builder  *LinkGraphBuilder
	linkQueue chan *Link
	logger   *zap.Logger
}

// SpamDetector detects spam links
type SpamDetector struct {
	spamPatterns    []*regexp.Regexp
	suspiciousDomains map[string]bool
	threshold       float64
}

// DuplicateDetector detects duplicate links
type DuplicateDetector struct {
	linkHashes map[string]string
	mutex      sync.RWMutex
}

// NewLinkGraphBuilder creates a new link graph builder
func NewLinkGraphBuilder(config LinkGraphConfig) (*LinkGraphBuilder, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}
	
	// Initialize Redis client if persistence is enabled
	var redisClient *redis.Client
	if config.EnablePersistence {
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
	
	// Initialize spam detector
	spamDetector := &SpamDetector{
		spamPatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)(viagra|cialis|casino|poker|lottery)`),
			regexp.MustCompile(`(?i)(click here|buy now|free money)`),
			regexp.MustCompile(`(?i)(\.tk|\.ml|\.ga|\.cf)$`),
		},
		suspiciousDomains: make(map[string]bool),
		threshold: 0.7,
	}
	
	// Initialize duplicate detector
	duplicateDetector := &DuplicateDetector{
		linkHashes: make(map[string]string),
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	return &LinkGraphBuilder{
		config:     config,
		logger:     logger,
		redisClient: redisClient,
		
		graph: &LinkGraph{
			Pages:           make(map[string]*WebPage),
			Links:           make(map[string]*Link),
			DomainGraph:     make(map[string][]string),
			AnchorTextIndex: make(map[string][]string),
			Stats:           &LinkGraphStats{},
		},
		
		linkQueue:  make(chan *Link, config.BatchSize*2),
		pageQueue:  make(chan *WebPage, config.BatchSize*2),
		
		urlCache:      make(map[string]bool),
		domainCache:   make(map[string]bool),
		anchorTextCache: make(map[string]int),
		
		ctx:    ctx,
		cancel: cancel,
		
		stats: &LinkGraphStats{},
		
		spamDetector:     spamDetector,
		duplicateDetector: duplicateDetector,
	}, nil
}

// Start begins the link graph building process
func (lgb *LinkGraphBuilder) Start() error {
	lgb.logger.Info("Starting link graph builder")
	
	// Start link processors
	for i := 0; i < lgb.config.WorkerCount; i++ {
		processor := &LinkProcessor{
			id:        i,
			builder:   lgb,
			linkQueue: lgb.linkQueue,
			logger:    lgb.logger.With(zap.Int("worker_id", i)),
		}
		lgb.workers = append(lgb.workers, processor)
		
		lgb.wg.Add(1)
		go processor.processLinks()
	}
	
	// Start statistics collector
	lgb.wg.Add(1)
	go lgb.collectStats()
	
	// Start persistence worker if enabled
	if lgb.config.EnablePersistence {
		lgb.wg.Add(1)
		go lgb.persistenceWorker()
	}
	
	lgb.logger.Info("Link graph builder started successfully")
	return nil
}

// Stop gracefully shuts down the link graph builder
func (lgb *LinkGraphBuilder) Stop() error {
	lgb.logger.Info("Stopping link graph builder")
	
	lgb.cancel()
	lgb.wg.Wait()
	
	if lgb.redisClient != nil {
		lgb.redisClient.Close()
	}
	
	lgb.logger.Info("Link graph builder stopped")
	return nil
}

// AddLink adds a new link to the graph
func (lgb *LinkGraphBuilder) AddLink(link *Link) error {
	// Validate link
	if err := lgb.validateLink(link); err != nil {
		return fmt.Errorf("invalid link: %w", err)
	}
	
	// Check for duplicates
	if lgb.duplicateDetector.isDuplicate(link) {
		atomic.AddInt64(&lgb.stats.DuplicateLinksFound, 1)
		return nil
	}
	
	// Check for spam
	if lgb.spamDetector.isSpam(link) {
		link.IsSpam = true
		atomic.AddInt64(&lgb.stats.SpamLinksDetected, 1)
	}
	
	// Queue for processing
	select {
	case lgb.linkQueue <- link:
		return nil
	case <-lgb.ctx.Done():
		return fmt.Errorf("link graph builder is shutting down")
	default:
		return fmt.Errorf("link queue is full")
	}
}

// AddPage adds a new web page to the graph
func (lgb *LinkGraphBuilder) AddPage(page *WebPage) error {
	lgb.graphMutex.Lock()
	defer lgb.graphMutex.Unlock()
	
	// Check if page already exists
	if _, exists := lgb.graph.Pages[page.ID]; exists {
		return fmt.Errorf("page already exists: %s", page.ID)
	}
	
	// Add page to graph
	lgb.graph.Pages[page.ID] = page
	
	// Update domain graph
	if _, exists := lgb.graph.DomainGraph[page.Domain]; !exists {
		lgb.graph.DomainGraph[page.Domain] = make([]string, 0)
	}
	lgb.graph.DomainGraph[page.Domain] = append(lgb.graph.DomainGraph[page.Domain], page.ID)
	
	// Update statistics
	atomic.AddInt64(&lgb.stats.TotalPages, 1)
	
	return nil
}

// GetPage retrieves a page by ID
func (lgb *LinkGraphBuilder) GetPage(pageID string) (*WebPage, bool) {
	lgb.graphMutex.RLock()
	defer lgb.graphMutex.RUnlock()
	
	page, exists := lgb.graph.Pages[pageID]
	return page, exists
}

// GetLinks retrieves links for a given page
func (lgb *LinkGraphBuilder) GetLinks(pageID string, direction string) ([]*Link, error) {
	lgb.graphMutex.RLock()
	defer lgb.graphMutex.RUnlock()
	
	var links []*Link
	
	for _, link := range lgb.graph.Links {
		switch direction {
		case "incoming":
			if link.ToURL == pageID {
				links = append(links, link)
			}
		case "outgoing":
			if link.FromURL == pageID {
				links = append(links, link)
			}
		case "all":
			if link.FromURL == pageID || link.ToURL == pageID {
				links = append(links, link)
			}
		}
	}
	
	return links, nil
}

// GetTopPages returns the top pages by a given metric
func (lgb *LinkGraphBuilder) GetTopPages(metric string, limit int) ([]*WebPage, error) {
	lgb.graphMutex.RLock()
	defer lgb.graphMutex.RUnlock()
	
	var pages []*WebPage
	for _, page := range lgb.graph.Pages {
		pages = append(pages, page)
	}
	
	// Sort by metric
	switch metric {
	case "pagerank":
		sort.Slice(pages, func(i, j int) bool {
			return pages[i].PageRank > pages[j].PageRank
		})
	case "authority":
		sort.Slice(pages, func(i, j int) bool {
			return pages[i].AuthorityScore > pages[j].AuthorityScore
		})
	case "hub":
		sort.Slice(pages, func(i, j int) bool {
			return pages[i].HubScore > pages[j].HubScore
		})
	case "inlinks":
		sort.Slice(pages, func(i, j int) bool {
			return pages[i].InLinkCount > pages[j].InLinkCount
		})
	case "outlinks":
		sort.Slice(pages, func(i, j int) bool {
			return pages[i].OutLinkCount > pages[j].OutLinkCount
		})
	}
	
	if limit > 0 && limit < len(pages) {
		pages = pages[:limit]
	}
	
	return pages, nil
}

// GetStats returns current link graph statistics
func (lgb *LinkGraphBuilder) GetStats() LinkGraphStats {
	lgb.statsMutex.RLock()
	defer lgb.statsMutex.RUnlock()
	return *lgb.stats
}

// validateLink validates a link before processing
func (lgb *LinkGraphBuilder) validateLink(link *Link) error {
	if link.FromURL == "" || link.ToURL == "" {
		return fmt.Errorf("missing URL")
	}
	
	if link.FromURL == link.ToURL {
		return fmt.Errorf("self-link not allowed")
	}
	
	// Validate URLs
	if _, err := url.Parse(link.FromURL); err != nil {
		return fmt.Errorf("invalid from URL: %w", err)
	}
	
	if _, err := url.Parse(link.ToURL); err != nil {
		return fmt.Errorf("invalid to URL: %w", err)
	}
	
	// Check domain restrictions
	fromDomain := lgb.extractDomain(link.FromURL)
	toDomain := lgb.extractDomain(link.ToURL)
	
	if lgb.isDomainBlocked(fromDomain) || lgb.isDomainBlocked(toDomain) {
		return fmt.Errorf("domain blocked")
	}
	
	if len(lgb.config.AllowedDomains) > 0 {
		if !lgb.isDomainAllowed(fromDomain) || !lgb.isDomainAllowed(toDomain) {
			return fmt.Errorf("domain not allowed")
		}
	}
	
	return nil
}

// extractDomain extracts domain from URL
func (lgb *LinkGraphBuilder) extractDomain(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return u.Host
}

// isDomainBlocked checks if domain is blocked
func (lgb *LinkGraphBuilder) isDomainBlocked(domain string) bool {
	for _, blocked := range lgb.config.BlockedDomains {
		if strings.Contains(domain, blocked) {
			return true
		}
	}
	return false
}

// isDomainAllowed checks if domain is allowed
func (lgb *LinkGraphBuilder) isDomainAllowed(domain string) bool {
	if len(lgb.config.AllowedDomains) == 0 {
		return true
	}
	
	for _, allowed := range lgb.config.AllowedDomains {
		if strings.Contains(domain, allowed) {
			return true
		}
	}
	return false
}

// processLinks processes links in the worker
func (lp *LinkProcessor) processLinks() {
	defer lp.builder.wg.Done()
	
	lp.logger.Info("Starting link processor")
	
	for {
		select {
		case link := <-lp.linkQueue:
			lp.processLink(link)
		case <-lp.builder.ctx.Done():
			lp.logger.Info("Link processor stopping")
			return
		}
	}
}

// processLink processes a single link
func (lp *LinkProcessor) processLink(link *Link) {
	start := time.Now()
	
	lp.logger.Debug("Processing link",
		zap.String("from", link.FromURL),
		zap.String("to", link.ToURL),
		zap.String("anchor", link.AnchorText))
	
	// Generate link ID if not provided
	if link.ID == "" {
		link.ID = lp.generateLinkID(link)
	}
	
	// Analyze anchor text
	if lp.builder.config.EnableAnchorTextAnalysis {
		lp.analyzeAnchorText(link)
	}
	
	// Detect link type
	if lp.builder.config.EnableLinkTypeDetection {
		lp.detectLinkType(link)
	}
	
	// Calculate link weight
	lp.calculateLinkWeight(link)
	
	// Add to graph
	lp.builder.graphMutex.Lock()
	lp.builder.graph.Links[link.ID] = link
	lp.builder.graphMutex.Unlock()
	
	// Update page statistics
	lp.updatePageStatistics(link)
	
	// Update anchor text index
	if link.AnchorText != "" {
		lp.updateAnchorTextIndex(link)
	}
	
	// Update statistics
	atomic.AddInt64(&lp.builder.stats.TotalLinks, 1)
	atomic.AddInt64(&lp.builder.stats.LinksProcessed, 1)
	
	lp.logger.Debug("Link processed",
		zap.String("link_id", link.ID),
		zap.Duration("processing_time", time.Since(start)))
}

// generateLinkID generates a unique ID for a link
func (lp *LinkProcessor) generateLinkID(link *Link) string {
	data := fmt.Sprintf("%s->%s:%s", link.FromURL, link.ToURL, link.AnchorText)
	hash := md5.Sum([]byte(data))
	return fmt.Sprintf("%x", hash)
}

// analyzeAnchorText analyzes anchor text for quality and relevance
func (lp *LinkProcessor) analyzeAnchorText(link *Link) {
	anchorText := strings.TrimSpace(link.AnchorText)
	
	// Check length constraints
	if len(anchorText) < lp.builder.config.MinAnchorTextLength {
		link.Confidence *= 0.5 // Reduce confidence for short anchor text
		return
	}
	
	if len(anchorText) > lp.builder.config.MaxAnchorTextLength {
		link.Confidence *= 0.8 // Reduce confidence for very long anchor text
		return
	}
	
	// Analyze anchor text quality
	qualityScore := lp.calculateAnchorTextQuality(anchorText)
	link.Confidence *= qualityScore
	
	// Update metadata
	if link.Metadata == nil {
		link.Metadata = make(map[string]interface{})
	}
	link.Metadata["anchor_text_quality"] = qualityScore
	link.Metadata["anchor_text_length"] = len(anchorText)
}

// calculateAnchorTextQuality calculates quality score for anchor text
func (lp *LinkProcessor) calculateAnchorTextQuality(anchorText string) float64 {
	score := 1.0
	
	// Penalize generic anchor text
	genericPatterns := []string{
		"click here", "read more", "more", "here", "link",
		"www", "http", "https", "www.", ".com", ".org",
	}
	
	for _, pattern := range genericPatterns {
		if strings.Contains(strings.ToLower(anchorText), pattern) {
			score *= 0.7
		}
	}
	
	// Reward descriptive anchor text
	if len(anchorText) > 10 && len(anchorText) < 100 {
		score *= 1.2
	}
	
	// Penalize anchor text with too many special characters
	specialCharCount := 0
	for _, char := range anchorText {
		if !((char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || 
			 (char >= '0' && char <= '9') || char == ' ' || char == '-') {
			specialCharCount++
		}
	}
	
	if float64(specialCharCount)/float64(len(anchorText)) > 0.3 {
		score *= 0.8
	}
	
	return score
}

// detectLinkType detects the type of link
func (lp *LinkProcessor) detectLinkType(link *Link) {
	fromDomain := lp.builder.extractDomain(link.FromURL)
	toDomain := lp.builder.extractDomain(link.ToURL)
	
	if fromDomain == toDomain {
		link.LinkType = LinkTypeInternal
	} else {
		link.LinkType = LinkTypeExternal
	}
	
	// Further classification based on anchor text and context
	anchorText := strings.ToLower(link.AnchorText)
	
	switch {
	case strings.Contains(anchorText, "sponsored") || strings.Contains(anchorText, "ad"):
		link.LinkType = LinkTypeSponsored
	case strings.Contains(anchorText, "share") || strings.Contains(anchorText, "tweet"):
		link.LinkType = LinkTypeSocial
	case strings.Contains(anchorText, "reference") || strings.Contains(anchorText, "source"):
		link.LinkType = LinkTypeReference
	case strings.Contains(anchorText, "home") || strings.Contains(anchorText, "menu"):
		link.LinkType = LinkTypeNavigational
	default:
		if link.LinkType == LinkTypeExternal {
			link.LinkType = LinkTypeContent
		}
	}
}

// calculateLinkWeight calculates the weight of a link
func (lp *LinkProcessor) calculateLinkWeight(link *Link) {
	weight := 1.0
	
	// Base weight on link type
	switch link.LinkType {
	case LinkTypeInternal:
		weight = 1.0
	case LinkTypeExternal:
		weight = 1.5
	case LinkTypeContent:
		weight = 2.0
	case LinkTypeReference:
		weight = 2.5
	case LinkTypeNavigational:
		weight = 0.5
	case LinkTypeSponsored:
		weight = 0.3
	case LinkTypeSocial:
		weight = 1.2
	}
	
	// Adjust based on anchor text quality
	if quality, exists := link.Metadata["anchor_text_quality"]; exists {
		if qualityScore, ok := quality.(float64); ok {
			weight *= qualityScore
		}
	}
	
	// Adjust based on confidence
	weight *= link.Confidence
	
	// Apply bounds
	if weight < lp.builder.config.MinLinkWeight {
		weight = lp.builder.config.MinLinkWeight
	}
	if weight > lp.builder.config.MaxLinkWeight {
		weight = lp.builder.config.MaxLinkWeight
	}
	
	link.Weight = weight
}

// updatePageStatistics updates statistics for pages involved in the link
func (lp *LinkProcessor) updatePageStatistics(link *Link) {
	lp.builder.graphMutex.Lock()
	defer lp.builder.graphMutex.Unlock()
	
	// Update from page
	if fromPage, exists := lp.builder.graph.Pages[link.FromURL]; exists {
		fromPage.OutLinkCount++
	} else {
		// Create page if it doesn't exist
		fromPage = &WebPage{
			ID:     link.FromURL,
			URL:    link.FromURL,
			Domain: lp.builder.extractDomain(link.FromURL),
		}
		lp.builder.graph.Pages[link.FromURL] = fromPage
	}
	
	// Update to page
	if toPage, exists := lp.builder.graph.Pages[link.ToURL]; exists {
		toPage.InLinkCount++
	} else {
		// Create page if it doesn't exist
		toPage = &WebPage{
			ID:     link.ToURL,
			URL:    link.ToURL,
			Domain: lp.builder.extractDomain(link.ToURL),
		}
		lp.builder.graph.Pages[link.ToURL] = toPage
	}
}

// updateAnchorTextIndex updates the anchor text index
func (lp *LinkProcessor) updateAnchorTextIndex(link *Link) {
	lp.builder.graphMutex.Lock()
	defer lp.builder.graphMutex.Unlock()
	
	anchorText := strings.ToLower(link.AnchorText)
	words := strings.Fields(anchorText)
	
	for _, word := range words {
		if len(word) > 2 { // Only index words longer than 2 characters
			if _, exists := lp.builder.graph.AnchorTextIndex[word]; !exists {
				lp.builder.graph.AnchorTextIndex[word] = make([]string, 0)
			}
			lp.builder.graph.AnchorTextIndex[word] = append(lp.builder.graph.AnchorTextIndex[word], link.ID)
		}
	}
}

// collectStats collects and updates statistics
func (lgb *LinkGraphBuilder) collectStats() {
	defer lgb.wg.Done()
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			lgb.updateStats()
		case <-lgb.ctx.Done():
			return
		}
	}
}

// updateStats updates graph statistics
func (lgb *LinkGraphBuilder) updateStats() {
	lgb.graphMutex.RLock()
	defer lgb.graphMutex.RUnlock()
	
	lgb.statsMutex.Lock()
	defer lgb.statsMutex.Unlock()
	
	// Calculate degree statistics
	var totalInDegree, totalOutDegree int64
	var maxInDegree, maxOutDegree int64
	
	for _, page := range lgb.graph.Pages {
		totalInDegree += page.InLinkCount
		totalOutDegree += page.OutLinkCount
		
		if page.InLinkCount > maxInDegree {
			maxInDegree = page.InLinkCount
		}
		if page.OutLinkCount > maxOutDegree {
			maxOutDegree = page.OutLinkCount
		}
	}
	
	pageCount := int64(len(lgb.graph.Pages))
	if pageCount > 0 {
		lgb.stats.AverageInDegree = float64(totalInDegree) / float64(pageCount)
		lgb.stats.AverageOutDegree = float64(totalOutDegree) / float64(pageCount)
	}
	
	lgb.stats.MaxInDegree = maxInDegree
	lgb.stats.MaxOutDegree = maxOutDegree
	lgb.stats.TotalDomains = int64(len(lgb.graph.DomainGraph))
	lgb.stats.LastUpdated = time.Now()
	
	// Calculate graph density
	if pageCount > 1 {
		maxPossibleEdges := pageCount * (pageCount - 1)
		lgb.stats.GraphDensity = float64(lgb.stats.TotalLinks) / float64(maxPossibleEdges)
	}
}

// persistenceWorker handles persistence to Redis
func (lgb *LinkGraphBuilder) persistenceWorker() {
	defer lgb.wg.Done()
	
	if lgb.redisClient == nil {
		return
	}
	
	ticker := time.NewTicker(lgb.config.PersistenceInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			lgb.persistToRedis()
		case <-lgb.ctx.Done():
			return
		}
	}
}

// persistToRedis persists graph data to Redis
func (lgb *LinkGraphBuilder) persistToRedis() {
	// Implementation would serialize and store graph data
	// This is a placeholder for the actual persistence logic
	lgb.logger.Debug("Persisting graph data to Redis")
}

// Spam detection methods
func (sd *SpamDetector) isSpam(link *Link) bool {
	// Check anchor text patterns
	anchorText := strings.ToLower(link.AnchorText)
	for _, pattern := range sd.spamPatterns {
		if pattern.MatchString(anchorText) {
			return true
		}
	}
	
	// Check suspicious domains
	toDomain := extractDomain(link.ToURL)
	if sd.suspiciousDomains[toDomain] {
		return true
	}
	
	return false
}

// Duplicate detection methods
func (dd *DuplicateDetector) isDuplicate(link *Link) bool {
	linkHash := fmt.Sprintf("%s->%s", link.FromURL, link.ToURL)
	
	dd.mutex.Lock()
	defer dd.mutex.Unlock()
	
	if _, exists := dd.linkHashes[linkHash]; exists {
		return true
	}
	
	dd.linkHashes[linkHash] = link.ID
	return false
}

// Helper function to extract domain
func extractDomain(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return u.Host
}
