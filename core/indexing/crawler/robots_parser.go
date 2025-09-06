// T3SS Project
// File: core/indexing/crawler/robots_parser.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package crawler

import (
	"bufio"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

// RobotsRule represents a single robots.txt rule
type RobotsRule struct {
	UserAgent string
	Path      string
	Allow     bool
}

// RobotsTxt represents a parsed robots.txt file
type RobotsTxt struct {
	Rules       []RobotsRule
	CrawlDelay  time.Duration
	Sitemaps    []string
	LastFetched time.Time
	Expires     time.Time
}

// RobotsParser handles robots.txt parsing and caching
type RobotsParser struct {
	cache map[string]*RobotsTxt
	mu    sync.RWMutex
	client *http.Client
}

// NewRobotsParser creates a new robots.txt parser
func NewRobotsParser() *RobotsParser {
	return &RobotsParser{
		cache: make(map[string]*RobotsTxt),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// CanCrawl checks if a URL can be crawled according to robots.txt
func (rp *RobotsParser) CanCrawl(targetURL, userAgent string) (bool, time.Duration, error) {
	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		return false, 0, err
	}

	robotsURL := fmt.Sprintf("%s://%s/robots.txt", parsedURL.Scheme, parsedURL.Host)
	
	// Get or fetch robots.txt
	robots, err := rp.getRobotsTxt(robotsURL)
	if err != nil {
		// If we can't fetch robots.txt, assume we can crawl
		return true, 0, nil
	}

	// Check if robots.txt has expired
	if time.Now().After(robots.Expires) {
		// Re-fetch robots.txt
		robots, err = rp.fetchRobotsTxt(robotsURL)
		if err != nil {
			// If re-fetch fails, use cached version
			robots, _ = rp.getRobotsTxt(robotsURL)
		}
	}

	if robots == nil {
		return true, 0, nil
	}

	// Check rules
	canCrawl := rp.checkRules(robots, parsedURL.Path, userAgent)
	
	return canCrawl, robots.CrawlDelay, nil
}

// getRobotsTxt gets robots.txt from cache or fetches it
func (rp *RobotsParser) getRobotsTxt(robotsURL string) (*RobotsTxt, error) {
	rp.mu.RLock()
	robots, exists := rp.cache[robotsURL]
	rp.mu.RUnlock()

	if exists && time.Now().Before(robots.Expires) {
		return robots, nil
	}

	// Fetch if not in cache or expired
	return rp.fetchRobotsTxt(robotsURL)
}

// fetchRobotsTxt fetches and parses robots.txt from the given URL
func (rp *RobotsParser) fetchRobotsTxt(robotsURL string) (*RobotsTxt, error) {
	resp, err := rp.client.Get(robotsURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("robots.txt returned status %d", resp.StatusCode)
	}

	robots := &RobotsTxt{
		Rules:       []RobotsRule{},
		CrawlDelay:  0,
		Sitemaps:    []string{},
		LastFetched: time.Now(),
		Expires:     time.Now().Add(24 * time.Hour), // Default 24 hour cache
	}

	scanner := bufio.NewScanner(resp.Body)
	var currentUserAgent string

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse line
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}

		directive := strings.ToLower(strings.TrimSpace(parts[0]))
		value := strings.TrimSpace(parts[1])

		switch directive {
		case "user-agent":
			currentUserAgent = value
		case "disallow":
			if currentUserAgent != "" {
				robots.Rules = append(robots.Rules, RobotsRule{
					UserAgent: currentUserAgent,
					Path:      value,
					Allow:     false,
				})
			}
		case "allow":
			if currentUserAgent != "" {
				robots.Rules = append(robots.Rules, RobotsRule{
					UserAgent: currentUserAgent,
					Path:      value,
					Allow:     true,
				})
			}
		case "crawl-delay":
			if currentUserAgent != "" {
				if delay, err := time.ParseDuration(value + "s"); err == nil {
					robots.CrawlDelay = delay
				}
			}
		case "sitemap":
			robots.Sitemaps = append(robots.Sitemaps, value)
		}
	}

	// Cache the result
	rp.mu.Lock()
	rp.cache[robotsURL] = robots
	rp.mu.Unlock()

	return robots, scanner.Err()
}

// checkRules checks if a path is allowed for a user agent
func (rp *RobotsParser) checkRules(robots *RobotsTxt, path, userAgent string) bool {
	// Find applicable rules
	var applicableRules []RobotsRule
	
	for _, rule := range robots.Rules {
		if rp.matchesUserAgent(rule.UserAgent, userAgent) {
			applicableRules = append(applicableRules, rule)
		}
	}

	// If no specific rules, check for wildcard
	if len(applicableRules) == 0 {
		for _, rule := range robots.Rules {
			if rule.UserAgent == "*" {
				applicableRules = append(applicableRules, rule)
			}
		}
	}

	// If still no rules, allow crawling
	if len(applicableRules) == 0 {
		return true
	}

	// Check rules in order (most specific first)
	// Sort rules by path length (longer paths are more specific)
	for i := 0; i < len(applicableRules); i++ {
		for j := i + 1; j < len(applicableRules); j++ {
			if len(applicableRules[i].Path) < len(applicableRules[j].Path) {
				applicableRules[i], applicableRules[j] = applicableRules[j], applicableRules[i]
			}
		}
	}

	// Check each rule
	for _, rule := range applicableRules {
		if rp.matchesPath(rule.Path, path) {
			return rule.Allow
		}
	}

	// If no rules match, allow crawling
	return true
}

// matchesUserAgent checks if a user agent matches a rule
func (rp *RobotsParser) matchesUserAgent(ruleUserAgent, userAgent string) bool {
	if ruleUserAgent == "*" {
		return true
	}
	
	// Case-insensitive comparison
	return strings.EqualFold(ruleUserAgent, userAgent)
}

// matchesPath checks if a path matches a rule pattern
func (rp *RobotsParser) matchesPath(rulePath, path string) bool {
	if rulePath == "" {
		return false
	}

	// Convert robots.txt pattern to regex
	// * matches any sequence of characters
	// $ matches end of string
	pattern := regexp.QuoteMeta(rulePath)
	pattern = strings.ReplaceAll(pattern, "\\*", ".*")
	
	// Add anchor if pattern doesn't end with $
	if !strings.HasSuffix(pattern, "$") {
		pattern += ".*"
	}

	regex, err := regexp.Compile("^" + pattern)
	if err != nil {
		return false
	}

	return regex.MatchString(path)
}

// GetSitemaps returns sitemap URLs from robots.txt
func (rp *RobotsParser) GetSitemaps(domain string) ([]string, error) {
	robotsURL := fmt.Sprintf("https://%s/robots.txt", domain)
	robots, err := rp.getRobotsTxt(robotsURL)
	if err != nil {
		return nil, err
	}

	return robots.Sitemaps, nil
}

// ClearCache clears the robots.txt cache
func (rp *RobotsParser) ClearCache() {
	rp.mu.Lock()
	defer rp.mu.Unlock()
	rp.cache = make(map[string]*RobotsTxt)
}