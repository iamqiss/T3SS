// T3SS Project
// File: core/indexing/crawler/politeness_enforcer.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package crawler

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// RobotsRule represents a single robots.txt rule
type RobotsRule struct {
	UserAgent string
	Allow     []*regexp.Regexp
	Disallow  []*regexp.Regexp
	CrawlDelay time.Duration
}

// RobotsCache caches robots.txt rules for domains
type RobotsCache struct {
	rules    map[string]*RobotsRule
	lastFetch map[string]time.Time
	mu       sync.RWMutex
	client   *http.Client
}

// PolitenessEnforcer ensures respectful crawling behavior
type PolitenessEnforcer struct {
	cache        *RobotsCache
	rateLimiters map[string]*rate.Limiter
	mu           sync.RWMutex
	config       PolitenessConfig
}

// PolitenessConfig holds configuration for polite crawling
type PolitenessConfig struct {
	DefaultDelay      time.Duration
	MaxConcurrentPerDomain int
	RobotsCacheTTL    time.Duration
	UserAgent         string
	RespectRobots     bool
	MaxRetries        int
}

// NewPolitenessEnforcer creates a new politeness enforcer
func NewPolitenessEnforcer(config PolitenessConfig) *PolitenessEnforcer {
	return &PolitenessEnforcer{
		cache: &RobotsCache{
			rules:      make(map[string]*RobotsRule),
			lastFetch:  make(map[string]time.Time),
			client:     &http.Client{Timeout: 10 * time.Second},
		},
		rateLimiters: make(map[string]*rate.Limiter),
		config:       config,
	}
}

// CanCrawl checks if a URL can be crawled according to robots.txt
func (pe *PolitenessEnforcer) CanCrawl(ctx context.Context, targetURL string) (bool, error) {
	if !pe.config.RespectRobots {
		return true, nil
	}

	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		return false, fmt.Errorf("invalid URL: %w", err)
	}

	domain := parsedURL.Host
	rule, err := pe.getRobotsRule(ctx, domain)
	if err != nil {
		// If we can't fetch robots.txt, be conservative and allow
		return true, nil
	}

	if rule == nil {
		return true, nil
	}

	// Check if our user agent matches
	if !pe.matchesUserAgent(rule.UserAgent) {
		return true, nil
	}

	// Check disallow rules first
	for _, disallowPattern := range rule.Disallow {
		if disallowPattern.MatchString(targetURL) {
			return false, nil
		}
	}

	// Check allow rules
	for _, allowPattern := range rule.Allow {
		if allowPattern.MatchString(targetURL) {
			return true, nil
		}
	}

	// Default to allowed if no specific rules match
	return true, nil
}

// WaitForRateLimit waits for the appropriate delay before crawling a domain
func (pe *PolitenessEnforcer) WaitForRateLimit(ctx context.Context, domain string) error {
	limiter := pe.getRateLimiter(domain)
	return limiter.Wait(ctx)
}

// getRobotsRule fetches and caches robots.txt rules for a domain
func (pe *PolitenessEnforcer) getRobotsRule(ctx context.Context, domain string) (*RobotsRule, error) {
	pe.cache.mu.Lock()
	defer pe.cache.mu.Unlock()

	// Check cache first
	if rule, exists := pe.cache.rules[domain]; exists {
		if time.Since(pe.cache.lastFetch[domain]) < pe.config.RobotsCacheTTL {
			return rule, nil
		}
	}

	// Fetch robots.txt
	robotsURL := fmt.Sprintf("https://%s/robots.txt", domain)
	req, err := http.NewRequestWithContext(ctx, "GET", robotsURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", pe.config.UserAgent)
	resp, err := pe.cache.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// No robots.txt or error - cache as nil
		pe.cache.rules[domain] = nil
		pe.cache.lastFetch[domain] = time.Now()
		return nil, nil
	}

	// Parse robots.txt
	rule, err := pe.parseRobotsTxt(resp.Body)
	if err != nil {
		return nil, err
	}

	// Cache the rule
	pe.cache.rules[domain] = rule
	pe.cache.lastFetch[domain] = time.Now()

	return rule, nil
}

// parseRobotsTxt parses robots.txt content
func (pe *PolitenessEnforcer) parseRobotsTxt(body io.Reader) (*RobotsRule, error) {
	// This is a simplified parser - in production, you'd use a more robust one
	scanner := bufio.NewScanner(body)
	
	rule := &RobotsRule{
		UserAgent: "*",
		Allow:     make([]*regexp.Regexp, 0),
		Disallow:  make([]*regexp.Regexp, 0),
		CrawlDelay: pe.config.DefaultDelay,
	}

	var currentUserAgent string
	var inOurSection bool

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}

		directive := strings.ToLower(strings.TrimSpace(parts[0]))
		value := strings.TrimSpace(parts[1])

		switch directive {
		case "user-agent":
			currentUserAgent = value
			inOurSection = pe.matchesUserAgent(currentUserAgent)
		case "disallow":
			if inOurSection {
				if pattern, err := pe.compileRobotsPattern(value); err == nil {
					rule.Disallow = append(rule.Disallow, pattern)
				}
			}
		case "allow":
			if inOurSection {
				if pattern, err := pe.compileRobotsPattern(value); err == nil {
					rule.Allow = append(rule.Allow, pattern)
				}
			}
		case "crawl-delay":
			if inOurSection {
				if delay, err := time.ParseDuration(value + "s"); err == nil {
					rule.CrawlDelay = delay
				}
			}
		}
	}

	return rule, scanner.Err()
}

// compileRobotsPattern compiles a robots.txt pattern to regex
func (pe *PolitenessEnforcer) compileRobotsPattern(pattern string) (*regexp.Regexp, error) {
	// Convert robots.txt pattern to regex
	// * matches any sequence of characters
	// $ matches end of string
	escaped := regexp.QuoteMeta(pattern)
	escaped = strings.ReplaceAll(escaped, "\\*", ".*")
	escaped = strings.ReplaceAll(escaped, "\\$", "$")
	
	// Ensure it matches from the beginning
	if !strings.HasPrefix(escaped, "^") {
		escaped = "^" + escaped
	}
	
	return regexp.Compile(escaped)
}

// matchesUserAgent checks if our user agent matches the pattern
func (pe *PolitenessEnforcer) matchesUserAgent(pattern string) bool {
	if pattern == "*" {
		return true
	}
	
	// Simple substring match - in production, you'd use proper pattern matching
	return strings.Contains(strings.ToLower(pe.config.UserAgent), strings.ToLower(pattern))
}

// getRateLimiter gets or creates a rate limiter for a domain
func (pe *PolitenessEnforcer) getRateLimiter(domain string) *rate.Limiter {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	if limiter, exists := pe.rateLimiters[domain]; exists {
		return limiter
	}

	// Create new rate limiter based on robots.txt crawl delay
	delay := pe.config.DefaultDelay
	if rule, exists := pe.cache.rules[domain]; exists && rule != nil {
		delay = rule.CrawlDelay
	}

	// Convert delay to rate (requests per second)
	rate := rate.Every(delay)
	limiter := rate.NewLimiter(rate, 1) // Burst of 1
	
	pe.rateLimiters[domain] = limiter
	return limiter
}

// UpdateDomainDelay updates the crawl delay for a specific domain
func (pe *PolitenessEnforcer) UpdateDomainDelay(domain string, delay time.Duration) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Update or create rate limiter
	rate := rate.Every(delay)
	limiter := rate.NewLimiter(rate, 1)
	pe.rateLimiters[domain] = limiter

	// Update cached rule if it exists
	pe.cache.mu.Lock()
	if rule, exists := pe.cache.rules[domain]; exists && rule != nil {
		rule.CrawlDelay = delay
	}
	pe.cache.mu.Unlock()
}

// GetDomainDelay returns the current crawl delay for a domain
func (pe *PolitenessEnforcer) GetDomainDelay(domain string) time.Duration {
	pe.cache.mu.RLock()
	defer pe.cache.mu.RUnlock()

	if rule, exists := pe.cache.rules[domain]; exists && rule != nil {
		return rule.CrawlDelay
	}
	return pe.config.DefaultDelay
}

// ClearCache clears the robots.txt cache
func (pe *PolitenessEnforcer) ClearCache() {
	pe.cache.mu.Lock()
	defer pe.cache.mu.Unlock()

	pe.cache.rules = make(map[string]*RobotsRule)
	pe.cache.lastFetch = make(map[string]time.Time)
}

// GetCacheStats returns cache statistics
func (pe *PolitenessEnforcer) GetCacheStats() map[string]interface{} {
	pe.cache.mu.RLock()
	defer pe.cache.mu.RUnlock()

	pe.mu.RLock()
	defer pe.mu.RUnlock()

	return map[string]interface{}{
		"cached_domains": len(pe.cache.rules),
		"rate_limiters":  len(pe.rateLimiters),
		"cache_ttl":      pe.config.RobotsCacheTTL,
	}
}