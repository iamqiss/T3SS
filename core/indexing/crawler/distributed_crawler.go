// T3SS Project
// File: core/indexing/crawler/distributed_crawler.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package crawler

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-redis/redis/v8"
	"go.etcd.io/etcd/clientv3"
	"go.uber.org/zap"
)

// DistributedCrawlerConfig holds configuration for the distributed crawler
type DistributedCrawlerConfig struct {
	// Cluster configuration
	NodeID              string        `yaml:"node_id"`
	EtcdEndpoints       []string      `yaml:"etcd_endpoints"`
	RedisEndpoint       string        `yaml:"redis_endpoint"`
	MaxConcurrentNodes  int           `yaml:"max_concurrent_nodes"`
	
	// Crawling configuration
	MaxConcurrentCrawls int           `yaml:"max_concurrent_crawls"`
	CrawlDelay          time.Duration `yaml:"crawl_delay"`
	MaxPagesPerDomain   int           `yaml:"max_pages_per_domain"`
	MaxDepth            int           `yaml:"max_depth"`
	
	// Performance tuning
	BatchSize           int           `yaml:"batch_size"`
	QueueBufferSize     int           `yaml:"queue_buffer_size"`
	HeartbeatInterval   time.Duration `yaml:"heartbeat_interval"`
	
	// Quality control
	MinContentLength    int           `yaml:"min_content_length"`
	MaxContentLength    int           `yaml:"max_content_length"`
	AllowedContentTypes []string      `yaml:"allowed_content_types"`
}

// CrawlJob represents a crawling job in the distributed system
type CrawlJob struct {
	ID          string    `json:"id"`
	URL         string    `json:"url"`
	Priority    int       `json:"priority"`
	Depth       int       `json:"depth"`
	Domain      string    `json:"domain"`
	AssignedTo  string    `json:"assigned_to"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	RetryCount  int       `json:"retry_count"`
	MaxRetries  int       `json:"max_retries"`
}

// CrawlResult represents the result of a crawl operation
type CrawlResult struct {
	JobID       string            `json:"job_id"`
	URL         string            `json:"url"`
	StatusCode  int               `json:"status_code"`
	Content     string            `json:"content"`
	Links       []string          `json:"links"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
	ProcessedAt time.Time         `json:"processed_at"`
}

// NodeStatus represents the status of a crawler node
type NodeStatus struct {
	NodeID      string    `json:"node_id"`
	Status      string    `json:"status"`
	ActiveJobs  int       `json:"active_jobs"`
	TotalJobs   int64     `json:"total_jobs"`
	LastSeen    time.Time `json:"last_seen"`
	Capabilities []string `json:"capabilities"`
}

// DistributedCrawler manages distributed web crawling across multiple nodes
type DistributedCrawler struct {
	config     DistributedCrawlerConfig
	nodeID     string
	etcdClient *clientv3.Client
	redisClient *redis.Client
	fetcher    *Fetcher
	logger     *zap.Logger
	
	// Job management
	jobQueue   chan *CrawlJob
	jobResults chan *CrawlResult
	activeJobs map[string]*CrawlJob
	
	// Node coordination
	nodeStatus    *NodeStatus
	otherNodes    map[string]*NodeStatus
	leaderElection *LeaderElection
	
	// Statistics
	stats        *DistributedCrawlerStats
	shutdownChan chan struct{}
	
	// Synchronization
	mu           sync.RWMutex
	wg           sync.WaitGroup
	ctx          context.Context
	cancel       context.CancelFunc
}

// DistributedCrawlerStats tracks distributed crawler performance
type DistributedCrawlerStats struct {
	TotalJobsProcessed    int64
	TotalJobsSucceeded    int64
	TotalJobsFailed       int64
	TotalPagesCrawled     int64
	TotalLinksDiscovered  int64
	AverageJobLatency     time.Duration
	JobsPerSecond         float64
	ActiveNodes           int
	QueueDepth            int
	mu                    sync.RWMutex
}

// LeaderElection handles leader election for job distribution
type LeaderElection struct {
	nodeID     string
	etcdClient *clientv3.Client
	isLeader   bool
	leaderID   string
	mu         sync.RWMutex
}

// NewDistributedCrawler creates a new distributed crawler instance
func NewDistributedCrawler(config DistributedCrawlerConfig) (*DistributedCrawler, error) {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %w", err)
	}
	
	// Initialize etcd client for coordination
	etcdClient, err := clientv3.New(clientv3.Config{
		Endpoints:   config.EtcdEndpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %w", err)
	}
	
	// Initialize Redis client for job queue
	redisClient := redis.NewClient(&redis.Options{
		Addr: config.RedisEndpoint,
	})
	
	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}
	
	// Initialize fetcher
	fetcherConfig := FetcherConfig{
		MaxConcurrentRequests: config.MaxConcurrentCrawls,
		RequestTimeout:        30 * time.Second,
		MaxRetries:            3,
		RateLimitPerSecond:    10,
		UserAgent:             "T3SS-DistributedCrawler/1.0",
		MaxResponseSize:       int64(config.MaxContentLength),
		EnableCompression:     true,
		KeepAliveTimeout:      30 * time.Second,
	}
	fetcher := NewFetcher(fetcherConfig)
	
	// Initialize leader election
	leaderElection := &LeaderElection{
		nodeID:     config.NodeID,
		etcdClient: etcdClient,
		isLeader:   false,
	}
	
	// Create context for graceful shutdown
	ctx, cancel = context.WithCancel(context.Background())
	
	return &DistributedCrawler{
		config:     config,
		nodeID:     config.NodeID,
		etcdClient: etcdClient,
		redisClient: redisClient,
		fetcher:    fetcher,
		logger:     logger,
		
		jobQueue:   make(chan *CrawlJob, config.QueueBufferSize),
		jobResults: make(chan *CrawlResult, config.QueueBufferSize),
		activeJobs: make(map[string]*CrawlJob),
		
		nodeStatus: &NodeStatus{
			NodeID:       config.NodeID,
			Status:       "starting",
			ActiveJobs:  0,
			TotalJobs:    0,
			LastSeen:     time.Now(),
			Capabilities: []string{"crawling", "parsing"},
		},
		otherNodes:    make(map[string]*NodeStatus),
		leaderElection: leaderElection,
		
		stats:        &DistributedCrawlerStats{},
		shutdownChan: make(chan struct{}),
		
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

// Start begins the distributed crawler operation
func (dc *DistributedCrawler) Start() error {
	dc.logger.Info("Starting distributed crawler", zap.String("node_id", dc.nodeID))
	
	// Start leader election
	if err := dc.leaderElection.start(); err != nil {
		return fmt.Errorf("failed to start leader election: %w", err)
	}
	
	// Start node registration and heartbeat
	dc.wg.Add(1)
	go dc.nodeHeartbeat()
	
	// Start job processing workers
	for i := 0; i < dc.config.MaxConcurrentCrawls; i++ {
		dc.wg.Add(1)
		go dc.jobWorker(i)
	}
	
	// Start job result processor
	dc.wg.Add(1)
	go dc.resultProcessor()
	
	// Start job distributor (only if leader)
	dc.wg.Add(1)
	go dc.jobDistributor()
	
	// Start statistics collector
	dc.wg.Add(1)
	go dc.statsCollector()
	
	dc.nodeStatus.Status = "running"
	dc.logger.Info("Distributed crawler started successfully")
	
	return nil
}

// Stop gracefully shuts down the distributed crawler
func (dc *DistributedCrawler) Stop() error {
	dc.logger.Info("Stopping distributed crawler")
	
	// Signal shutdown
	close(dc.shutdownChan)
	dc.cancel()
	
	// Wait for all goroutines to finish
	dc.wg.Wait()
	
	// Clean up resources
	dc.fetcher.Close()
	dc.etcdClient.Close()
	dc.redisClient.Close()
	
	dc.logger.Info("Distributed crawler stopped")
	return nil
}

// AddSeedURLs adds initial URLs to crawl
func (dc *DistributedCrawler) AddSeedURLs(urls []string) error {
	for _, urlStr := range urls {
		job := &CrawlJob{
			ID:         dc.generateJobID(urlStr),
			URL:        urlStr,
			Priority:   100, // High priority for seed URLs
			Depth:      0,
			Domain:     dc.extractDomain(urlStr),
			Status:     "pending",
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
			RetryCount: 0,
			MaxRetries: 3,
		}
		
		if err := dc.enqueueJob(job); err != nil {
			dc.logger.Error("Failed to enqueue seed URL", 
				zap.String("url", urlStr), 
				zap.Error(err))
			continue
		}
	}
	
	dc.logger.Info("Added seed URLs", zap.Int("count", len(urls)))
	return nil
}

// jobWorker processes crawl jobs
func (dc *DistributedCrawler) jobWorker(workerID int) {
	defer dc.wg.Done()
	
	dc.logger.Info("Starting job worker", zap.Int("worker_id", workerID))
	
	for {
		select {
		case job := <-dc.jobQueue:
			dc.processJob(job, workerID)
		case <-dc.ctx.Done():
			dc.logger.Info("Job worker stopping", zap.Int("worker_id", workerID))
			return
		}
	}
}

// processJob processes a single crawl job
func (dc *DistributedCrawler) processJob(job *CrawlJob, workerID int) {
	start := time.Now()
	
	dc.logger.Debug("Processing job", 
		zap.String("job_id", job.ID),
		zap.String("url", job.URL),
		zap.Int("worker_id", workerID))
	
	// Update job status
	dc.updateJobStatus(job.ID, "processing")
	
	// Perform the crawl
	result := dc.performCrawl(job)
	
	// Update statistics
	dc.updateStats(result, time.Since(start))
	
	// Send result for processing
	select {
	case dc.jobResults <- result:
	case <-dc.ctx.Done():
		return
	}
}

// performCrawl executes the actual crawling operation
func (dc *DistributedCrawler) performCrawl(job *CrawlJob) *CrawlResult {
	result := &CrawlResult{
		JobID:       job.ID,
		URL:         job.URL,
		ProcessedAt: time.Now(),
	}
	
	// Check if we should crawl this URL
	if !dc.shouldCrawl(job) {
		result.Error = "URL filtered out"
		return result
	}
	
	// Perform HTTP fetch
	fetchResult := dc.fetcher.Fetch(dc.ctx, job.URL)
	if fetchResult.Error != nil {
		result.Error = fetchResult.Error.Error()
		result.StatusCode = fetchResult.StatusCode
		return result
	}
	
	// Parse content
	links, metadata, err := dc.parseContent(fetchResult.Body, job.URL)
	if err != nil {
		result.Error = fmt.Sprintf("parsing error: %v", err)
		result.StatusCode = fetchResult.StatusCode
		return result
	}
	
	// Populate successful result
	result.StatusCode = fetchResult.StatusCode
	result.Content = string(fetchResult.Body)
	result.Links = links
	result.Metadata = metadata
	
	// Create new jobs for discovered links
	dc.createJobsForLinks(links, job.Depth+1)
	
	return result
}

// shouldCrawl determines if a URL should be crawled
func (dc *DistributedCrawler) shouldCrawl(job *CrawlJob) bool {
	// Check depth limit
	if job.Depth > dc.config.MaxDepth {
		return false
	}
	
	// Check retry limit
	if job.RetryCount >= job.MaxRetries {
		return false
	}
	
	// Check domain limits
	if dc.getDomainPageCount(job.Domain) >= dc.config.MaxPagesPerDomain {
		return false
	}
	
	return true
}

// parseContent extracts links and metadata from HTML content
func (dc *DistributedCrawler) parseContent(content []byte, baseURL string) ([]string, map[string]string, error) {
	links := make([]string, 0)
	metadata := make(map[string]string)
	
	// Extract title
	if title := dc.extractTitle(content); title != "" {
		metadata["title"] = title
	}
	
	// Extract meta description
	if desc := dc.extractMetaDescription(content); desc != "" {
		metadata["description"] = desc
	}
	
	// Extract meta keywords
	if keywords := dc.extractMetaKeywords(content); keywords != "" {
		metadata["keywords"] = keywords
	}
	
	// Extract canonical URL
	if canonical := dc.extractCanonicalURL(content); canonical != "" {
		metadata["canonical"] = canonical
	}
	
	// Extract Open Graph metadata
	ogData := dc.extractOpenGraphData(content)
	for k, v := range ogData {
		metadata["og_"+k] = v
	}
	
	// Extract structured data (JSON-LD, microdata)
	structuredData := dc.extractStructuredData(content)
	if len(structuredData) > 0 {
		metadata["structured_data"] = structuredData
	}
	
	// Extract links with proper HTML parsing
	links = dc.extractLinks(content, baseURL)
	
	// Extract content text for indexing
	textContent := dc.extractTextContent(content)
	metadata["content"] = textContent
	metadata["content_length"] = fmt.Sprintf("%d", len(textContent))
	
	// Calculate content hash for deduplication
	contentHash := dc.calculateContentHash(textContent)
	metadata["content_hash"] = contentHash
	
	return links, metadata, nil
}

// extractTitle extracts the page title from HTML content
func (dc *DistributedCrawler) extractTitle(content []byte) string {
	// Simplified title extraction - in production, use proper HTML parsing
	titleStart := []byte("<title>")
	titleEnd := []byte("</title>")
	
	start := -1
	for i := 0; i < len(content)-len(titleStart); i++ {
		if string(content[i:i+len(titleStart)]) == string(titleStart) {
			start = i + len(titleStart)
			break
		}
	}
	
	if start == -1 {
		return ""
	}
	
	end := -1
	for i := start; i < len(content)-len(titleEnd); i++ {
		if string(content[i:i+len(titleEnd)]) == string(titleEnd) {
			end = i
			break
		}
	}
	
	if end == -1 {
		return ""
	}
	
	return string(content[start:end])
}

// extractMetaDescription extracts meta description from HTML content
func (dc *DistributedCrawler) extractMetaDescription(content []byte) string {
	// Look for meta description tag
	pattern := `(?i)<meta\s+name=["']description["']\s+content=["']([^"']+)["']`
	re := regexp.MustCompile(pattern)
	matches := re.FindSubmatch(content)
	if len(matches) > 1 {
		return string(matches[1])
	}
	return ""
}

// extractMetaKeywords extracts meta keywords from HTML content
func (dc *DistributedCrawler) extractMetaKeywords(content []byte) string {
	pattern := `(?i)<meta\s+name=["']keywords["']\s+content=["']([^"']+)["']`
	re := regexp.MustCompile(pattern)
	matches := re.FindSubmatch(content)
	if len(matches) > 1 {
		return string(matches[1])
	}
	return ""
}

// extractCanonicalURL extracts canonical URL from HTML content
func (dc *DistributedCrawler) extractCanonicalURL(content []byte) string {
	pattern := `(?i)<link\s+rel=["']canonical["']\s+href=["']([^"']+)["']`
	re := regexp.MustCompile(pattern)
	matches := re.FindSubmatch(content)
	if len(matches) > 1 {
		return string(matches[1])
	}
	return ""
}

// extractOpenGraphData extracts Open Graph metadata
func (dc *DistributedCrawler) extractOpenGraphData(content []byte) map[string]string {
	ogData := make(map[string]string)
	pattern := `(?i)<meta\s+property=["']og:([^"']+)["']\s+content=["']([^"']+)["']`
	re := regexp.MustCompile(pattern)
	matches := re.FindAllSubmatch(content, -1)
	
	for _, match := range matches {
		if len(match) > 2 {
			ogData[string(match[1])] = string(match[2])
		}
	}
	
	return ogData
}

// extractStructuredData extracts structured data (JSON-LD, microdata)
func (dc *DistributedCrawler) extractStructuredData(content []byte) string {
	// Extract JSON-LD
	pattern := `(?i)<script\s+type=["']application/ld\+json["']>(.*?)</script>`
	re := regexp.MustCompile(pattern)
	matches := re.FindAllSubmatch(content, -1)
	
	var structuredData []string
	for _, match := range matches {
		if len(match) > 1 {
			structuredData = append(structuredData, string(match[1]))
		}
	}
	
	if len(structuredData) > 0 {
		return strings.Join(structuredData, "\n")
	}
	return ""
}

// extractLinks extracts all links from HTML content
func (dc *DistributedCrawler) extractLinks(content []byte, baseURL string) []string {
	var links []string
	pattern := `(?i)<a\s+[^>]*href=["']([^"']+)["'][^>]*>`
	re := regexp.MustCompile(pattern)
	matches := re.FindAllSubmatch(content, -1)
	
	base, err := url.Parse(baseURL)
	if err != nil {
		return links
	}
	
	for _, match := range matches {
		if len(match) > 1 {
			href := string(match[1])
			// Resolve relative URLs
			if resolvedURL := dc.resolveURL(base, href); resolvedURL != "" {
				links = append(links, resolvedURL)
			}
		}
	}
	
	return links
}

// extractTextContent extracts clean text content from HTML
func (dc *DistributedCrawler) extractTextContent(content []byte) string {
	// Remove script and style tags
	scriptPattern := `(?i)<script[^>]*>.*?</script>`
	stylePattern := `(?i)<style[^>]*>.*?</style>`
	content = regexp.MustCompile(scriptPattern).ReplaceAll(content, []byte(""))
	content = regexp.MustCompile(stylePattern).ReplaceAll(content, []byte(""))
	
	// Remove HTML tags
	tagPattern := `(?i)<[^>]+>`
	content = regexp.MustCompile(tagPattern).ReplaceAll(content, []byte(" "))
	
	// Clean up whitespace
	whitespacePattern := `\s+`
	content = regexp.MustCompile(whitespacePattern).ReplaceAll(content, []byte(" "))
	
	return strings.TrimSpace(string(content))
}

// calculateContentHash calculates hash for content deduplication
func (dc *DistributedCrawler) calculateContentHash(content string) string {
	hash := md5.Sum([]byte(content))
	return fmt.Sprintf("%x", hash)
}

// resolveURL resolves relative URLs against a base URL
func (dc *DistributedCrawler) resolveURL(base *url.URL, href string) string {
	parsed, err := url.Parse(href)
	if err != nil {
		return ""
	}
	
	resolved := base.ResolveReference(parsed)
	
	// Filter out unwanted URLs
	if resolved.Scheme != "http" && resolved.Scheme != "https" {
		return ""
	}
	
	// Filter out fragments and query parameters for deduplication
	resolved.Fragment = ""
	resolved.RawQuery = ""
	
	return resolved.String()
}

// createJobsForLinks creates new crawl jobs for discovered links
func (dc *DistributedCrawler) createJobsForLinks(links []string, depth int) {
	for _, link := range links {
		// Normalize URL
		normalizedURL := dc.normalizeURL(link)
		if normalizedURL == "" {
			continue
		}
		
		job := &CrawlJob{
			ID:         dc.generateJobID(normalizedURL),
			URL:        normalizedURL,
			Priority:   50, // Normal priority for discovered links
			Depth:      depth,
			Domain:     dc.extractDomain(normalizedURL),
			Status:     "pending",
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
			RetryCount: 0,
			MaxRetries: 3,
		}
		
		if err := dc.enqueueJob(job); err != nil {
			dc.logger.Debug("Failed to enqueue discovered link", 
				zap.String("url", normalizedURL), 
				zap.Error(err))
		}
	}
}

// enqueueJob adds a job to the distributed queue
func (dc *DistributedCrawler) enqueueJob(job *CrawlJob) error {
	jobData, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("failed to marshal job: %w", err)
	}
	
	// Add to Redis queue with priority
	ctx, cancel := context.WithTimeout(dc.ctx, 5*time.Second)
	defer cancel()
	
	score := float64(job.Priority) + float64(time.Now().Unix())
	return dc.redisClient.ZAdd(ctx, "crawl_queue", &redis.Z{
		Score:  score,
		Member: jobData,
	}).Err()
}

// resultProcessor processes crawl results
func (dc *DistributedCrawler) resultProcessor() {
	defer dc.wg.Done()
	
	for {
		select {
		case result := <-dc.jobResults:
			dc.handleCrawlResult(result)
		case <-dc.ctx.Done():
			return
		}
	}
}

// handleCrawlResult processes a crawl result
func (dc *DistributedCrawlResult) handleCrawlResult(result *CrawlResult) {
	// Update job status
	if result.Error != "" {
		dc.updateJobStatus(result.JobID, "failed")
		dc.logger.Error("Crawl failed", 
			zap.String("job_id", result.JobID),
			zap.String("url", result.URL),
			zap.String("error", result.Error))
	} else {
		dc.updateJobStatus(result.JobID, "completed")
		dc.logger.Debug("Crawl completed", 
			zap.String("job_id", result.JobID),
			zap.String("url", result.URL),
			zap.Int("links_found", len(result.Links)))
	}
	
	// Store result for indexing
	dc.storeCrawlResult(result)
}

// storeCrawlResult stores the crawl result for later indexing
func (dc *DistributedCrawler) storeCrawlResult(result *CrawlResult) {
	resultData, err := json.Marshal(result)
	if err != nil {
		dc.logger.Error("Failed to marshal crawl result", zap.Error(err))
		return
	}
	
	ctx, cancel := context.WithTimeout(dc.ctx, 5*time.Second)
	defer cancel()
	
	// Store in Redis with TTL
	key := fmt.Sprintf("crawl_result:%s", result.JobID)
	dc.redisClient.Set(ctx, key, resultData, 24*time.Hour)
}

// Helper methods
func (dc *DistributedCrawler) generateJobID(url string) string {
	hash := md5.Sum([]byte(url))
	return fmt.Sprintf("%x", hash)
}

func (dc *DistributedCrawler) extractDomain(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return u.Host
}

func (dc *DistributedCrawler) normalizeURL(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	
	// Basic normalization
	u.Fragment = ""
	u.RawQuery = ""
	
	return u.String()
}

func (dc *DistributedCrawler) getDomainPageCount(domain string) int {
	// In production, this would query a database
	return 0
}

func (dc *DistributedCrawler) updateJobStatus(jobID, status string) {
	dc.mu.Lock()
	defer dc.mu.Unlock()
	
	if job, exists := dc.activeJobs[jobID]; exists {
		job.Status = status
		job.UpdatedAt = time.Now()
	}
}

func (dc *DistributedCrawler) updateStats(result *CrawlResult, duration time.Duration) {
	dc.stats.mu.Lock()
	defer dc.stats.mu.Unlock()
	
	dc.stats.TotalJobsProcessed++
	if result.Error == "" {
		dc.stats.TotalJobsSucceeded++
		dc.stats.TotalPagesCrawled++
		dc.stats.TotalLinksDiscovered += int64(len(result.Links))
	} else {
		dc.stats.TotalJobsFailed++
	}
	
	// Update average latency
	if dc.stats.AverageJobLatency == 0 {
		dc.stats.AverageJobLatency = duration
	} else {
		dc.stats.AverageJobLatency = (dc.stats.AverageJobLatency + duration) / 2
	}
}

// Leader election and node coordination methods
func (le *LeaderElection) start() error {
	// Implement leader election using etcd
	// This is a simplified version - in production, use proper leader election
	return nil
}

func (dc *DistributedCrawler) nodeHeartbeat() {
	defer dc.wg.Done()
	
	ticker := time.NewTicker(dc.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			dc.sendHeartbeat()
		case <-dc.ctx.Done():
			return
		}
	}
}

func (dc *DistributedCrawler) sendHeartbeat() {
	dc.nodeStatus.LastSeen = time.Now()
	dc.nodeStatus.ActiveJobs = len(dc.activeJobs)
	
	nodeData, err := json.Marshal(dc.nodeStatus)
	if err != nil {
		dc.logger.Error("Failed to marshal node status", zap.Error(err))
		return
	}
	
	ctx, cancel := context.WithTimeout(dc.ctx, 5*time.Second)
	defer cancel()
	
	key := fmt.Sprintf("nodes/%s", dc.nodeID)
	dc.etcdClient.Put(ctx, key, string(nodeData))
}

func (dc *DistributedCrawler) jobDistributor() {
	defer dc.wg.Done()
	
	// Only the leader distributes jobs
	if !dc.leaderElection.isLeader {
		return
	}
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			dc.distributeJobs()
		case <-dc.ctx.Done():
			return
		}
	}
}

func (dc *DistributedCrawler) distributeJobs() {
	// Implement job distribution logic
	// This would pull jobs from Redis queue and assign them to available nodes
}

func (dc *DistributedCrawler) statsCollector() {
	defer dc.wg.Done()
	
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			dc.collectStats()
		case <-dc.ctx.Done():
			return
		}
	}
}

func (dc *DistributedCrawler) collectStats() {
	dc.stats.mu.Lock()
	defer dc.stats.mu.Unlock()
	
	dc.stats.QueueDepth = len(dc.jobQueue)
	dc.stats.ActiveNodes = len(dc.otherNodes) + 1
	
	// Calculate jobs per second
	if dc.stats.AverageJobLatency > 0 {
		dc.stats.JobsPerSecond = float64(time.Second) / float64(dc.stats.AverageJobLatency)
	}
}

// GetStats returns current crawler statistics
func (dc *DistributedCrawler) GetStats() DistributedCrawlerStats {
	dc.stats.mu.RLock()
	defer dc.stats.mu.RUnlock()
	return *dc.stats
}