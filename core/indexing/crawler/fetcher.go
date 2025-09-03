// T3SS Project
// File: core/indexing/crawler/fetcher.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package crawler

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// FetcherConfig holds configuration for the high-performance fetcher
type FetcherConfig struct {
	MaxConcurrentRequests int           `yaml:"max_concurrent_requests"`
	RequestTimeout        time.Duration `yaml:"request_timeout"`
	MaxRetries            int           `yaml:"max_retries"`
	RateLimitPerSecond    int           `yaml:"rate_limit_per_second"`
	UserAgent             string        `yaml:"user_agent"`
	MaxResponseSize       int64         `yaml:"max_response_size"`
	EnableCompression     bool          `yaml:"enable_compression"`
	KeepAliveTimeout      time.Duration `yaml:"keep_alive_timeout"`
}

// FetchResult represents the result of a fetch operation
type FetchResult struct {
	URL         string
	StatusCode  int
	Headers     http.Header
	Body        []byte
	FetchTime   time.Duration
	Error       error
	ContentType string
	Size        int64
}

// Fetcher is a high-performance HTTP fetcher with connection pooling and rate limiting
type Fetcher struct {
	config     FetcherConfig
	client     *http.Client
	rateLimiter *rate.Limiter
	pool       sync.Pool
	stats      *FetcherStats
}

// FetcherStats tracks performance metrics
type FetcherStats struct {
	TotalRequests    int64
	SuccessfulRequests int64
	FailedRequests   int64
	AverageLatency   time.Duration
	TotalBytes       int64
	mu               sync.RWMutex
}

// NewFetcher creates a new high-performance fetcher
func NewFetcher(config FetcherConfig) *Fetcher {
	// Create optimized HTTP transport with connection pooling
	transport := &http.Transport{
		MaxIdleConns:        config.MaxConcurrentRequests * 2,
		MaxIdleConnsPerHost: config.MaxConcurrentRequests,
		IdleConnTimeout:     config.KeepAliveTimeout,
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: false,
		},
		DisableCompression: !config.EnableCompression,
		ForceAttemptHTTP2:  true,
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   config.RequestTimeout,
	}

	rateLimiter := rate.NewLimiter(rate.Limit(config.RateLimitPerSecond), config.RateLimitPerSecond)

	return &Fetcher{
		config:      config,
		client:      client,
		rateLimiter: rateLimiter,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, 64*1024) // 64KB initial capacity
			},
		},
		stats: &FetcherStats{},
	}
}

// Fetch performs a high-performance HTTP fetch with retries and error handling
func (f *Fetcher) Fetch(ctx context.Context, url string) *FetchResult {
	start := time.Now()
	result := &FetchResult{
		URL: url,
	}

	// Rate limiting
	if err := f.rateLimiter.Wait(ctx); err != nil {
		result.Error = fmt.Errorf("rate limit error: %w", err)
		return result
	}

	// Retry logic with exponential backoff
	for attempt := 0; attempt <= f.config.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt*attempt) * 100 * time.Millisecond
			select {
			case <-ctx.Done():
				result.Error = ctx.Err()
				return result
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			result.Error = fmt.Errorf("failed to create request: %w", err)
			continue
		}

		// Set optimized headers
		req.Header.Set("User-Agent", f.config.UserAgent)
		req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
		req.Header.Set("Accept-Language", "en-US,en;q=0.5")
		req.Header.Set("Accept-Encoding", "gzip, deflate, br")
		req.Header.Set("Connection", "keep-alive")
		req.Header.Set("Upgrade-Insecure-Requests", "1")

		resp, err := f.client.Do(req)
		if err != nil {
			result.Error = fmt.Errorf("request failed: %w", err)
			continue
		}

		// Check status code
		result.StatusCode = resp.StatusCode
		if resp.StatusCode >= 400 {
			resp.Body.Close()
			result.Error = fmt.Errorf("HTTP error: %d", resp.StatusCode)
			continue
		}

		// Read response body with size limit
		body := f.pool.Get().([]byte)
		defer f.pool.Put(body[:0])

		limitedReader := io.LimitReader(resp.Body, f.config.MaxResponseSize)
		body, err = io.ReadAll(limitedReader)
		resp.Body.Close()

		if err != nil {
			result.Error = fmt.Errorf("failed to read response: %w", err)
			continue
		}

		// Success - populate result
		result.Headers = resp.Header
		result.Body = make([]byte, len(body))
		copy(result.Body, body)
		result.ContentType = resp.Header.Get("Content-Type")
		result.Size = int64(len(result.Body))
		result.FetchTime = time.Since(start)
		result.Error = nil

		// Update stats
		f.updateStats(true, result.FetchTime, result.Size)
		return result
	}

	// All retries failed
	result.FetchTime = time.Since(start)
	f.updateStats(false, result.FetchTime, 0)
	return result
}

// FetchBatch performs concurrent fetches with controlled concurrency
func (f *Fetcher) FetchBatch(ctx context.Context, urls []string) []*FetchResult {
	results := make([]*FetchResult, len(urls))
	semaphore := make(chan struct{}, f.config.MaxConcurrentRequests)
	var wg sync.WaitGroup

	for i, url := range urls {
		wg.Add(1)
		go func(index int, targetURL string) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release

			results[index] = f.Fetch(ctx, targetURL)
		}(i, url)
	}

	wg.Wait()
	return results
}

// updateStats updates the fetcher statistics in a thread-safe manner
func (f *Fetcher) updateStats(success bool, latency time.Duration, bytes int64) {
	f.stats.mu.Lock()
	defer f.stats.mu.Unlock()

	f.stats.TotalRequests++
	if success {
		f.stats.SuccessfulRequests++
		f.stats.TotalBytes += bytes
	} else {
		f.stats.FailedRequests++
	}

	// Update average latency using exponential moving average
	if f.stats.AverageLatency == 0 {
		f.stats.AverageLatency = latency
	} else {
		f.stats.AverageLatency = (f.stats.AverageLatency*9 + latency) / 10
	}
}

// GetStats returns current fetcher statistics
func (f *Fetcher) GetStats() FetcherStats {
	f.stats.mu.RLock()
	defer f.stats.mu.RUnlock()
	return *f.stats
}

// Close cleans up resources
func (f *Fetcher) Close() {
	if transport, ok := f.client.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
}