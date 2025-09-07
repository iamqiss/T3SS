// T3SS Project
// File: core/indexing/spam_detector/blacklist_manager.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package spam_detector

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// BlacklistType represents the type of blacklist entry
type BlacklistType int

const (
	BlacklistTypeDomain BlacklistType = iota
	BlacklistTypeIP
	BlacklistTypeURL
	BlacklistTypeEmail
	BlacklistTypeKeyword
	BlacklistTypeRegex
)

// BlacklistEntry represents a single blacklist entry
type BlacklistEntry struct {
	ID          string        `json:"id"`
	Type        BlacklistType `json:"type"`
	Value       string        `json:"value"`
	Reason      string        `json:"reason"`
	Source      string        `json:"source"`
	CreatedAt   time.Time     `json:"created_at"`
	IsActive    bool          `json:"is_active"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// BlacklistConfig holds configuration for the blacklist manager
type BlacklistConfig struct {
	StoragePath     string        `json:"storage_path"`
	UpdateInterval  time.Duration `json:"update_interval"`
	EnablePersistence bool        `json:"enable_persistence"`
}

// BlacklistStats holds statistics for the blacklist manager
type BlacklistStats struct {
	TotalEntries     int64     `json:"total_entries"`
	ActiveEntries    int64     `json:"active_entries"`
	MatchCount       int64     `json:"match_count"`
}

// BlacklistManager manages multiple blacklists
type BlacklistManager struct {
	config     BlacklistConfig
	entries    map[string]*BlacklistEntry
	domains    map[string]*BlacklistEntry
	ips        map[string]*BlacklistEntry
	urls       map[string]*BlacklistEntry
	emails     map[string]*BlacklistEntry
	keywords   map[string]*BlacklistEntry
	regexes    []*BlacklistEntry
	mu         sync.RWMutex
	stats      BlacklistStats
}

// NewBlacklistManager creates a new blacklist manager
func NewBlacklistManager(config BlacklistConfig) *BlacklistManager {
	return &BlacklistManager{
		config:  config,
		entries: make(map[string]*BlacklistEntry),
		domains: make(map[string]*BlacklistEntry),
		ips:     make(map[string]*BlacklistEntry),
		urls:    make(map[string]*BlacklistEntry),
		emails:  make(map[string]*BlacklistEntry),
		keywords: make(map[string]*BlacklistEntry),
		regexes: make([]*BlacklistEntry, 0),
	}
}

// Start initializes the blacklist manager
func (bm *BlacklistManager) Start() error {
	log.Println("Starting Blacklist Manager...")
	
	// Create storage directory if it doesn't exist
	if err := os.MkdirAll(bm.config.StoragePath, 0755); err != nil {
		return fmt.Errorf("failed to create storage directory: %w", err)
	}
	
	// Load existing entries
	if err := bm.loadEntries(); err != nil {
		log.Printf("Warning: failed to load existing entries: %v", err)
	}
	
	log.Println("Blacklist Manager started successfully")
	return nil
}

// Stop stops the blacklist manager
func (bm *BlacklistManager) Stop() error {
	log.Println("Stopping Blacklist Manager...")
	
	// Save entries
	if bm.config.EnablePersistence {
		if err := bm.saveEntries(); err != nil {
			log.Printf("Warning: failed to save entries: %v", err)
		}
	}
	
	log.Println("Blacklist Manager stopped")
	return nil
}

// AddEntry adds a new blacklist entry
func (bm *BlacklistManager) AddEntry(entry *BlacklistEntry) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	// Generate ID if not provided
	if entry.ID == "" {
		entry.ID = bm.generateID(entry.Value)
	}
	
	// Set timestamps
	now := time.Now()
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = now
	}
	
	// Add to main entries map
	bm.entries[entry.ID] = entry
	
	// Add to type-specific maps
	bm.addToTypeMaps(entry)
	
	// Update statistics
	bm.updateStats()
	
	log.Printf("Added blacklist entry: %s", entry.ID)
	return nil
}

// RemoveEntry removes a blacklist entry
func (bm *BlacklistManager) RemoveEntry(id string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	entry, exists := bm.entries[id]
	if !exists {
		return fmt.Errorf("entry with ID %s not found", id)
	}
	
	// Remove from main entries map
	delete(bm.entries, id)
	
	// Remove from type-specific maps
	bm.removeFromTypeMaps(entry)
	
	// Update statistics
	bm.updateStats()
	
	log.Printf("Removed blacklist entry: %s", id)
	return nil
}

// IsBlacklisted checks if a value is blacklisted
func (bm *BlacklistManager) IsBlacklisted(value string) (bool, *BlacklistEntry, error) {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	
	// Check domains
	if entry, found := bm.checkDomain(value); found {
		bm.stats.MatchCount++
		return true, entry, nil
	}
	
	// Check IPs
	if entry, found := bm.checkIP(value); found {
		bm.stats.MatchCount++
		return true, entry, nil
	}
	
	// Check URLs
	if entry, found := bm.checkURL(value); found {
		bm.stats.MatchCount++
		return true, entry, nil
	}
	
	// Check emails
	if entry, found := bm.checkEmail(value); found {
		bm.stats.MatchCount++
		return true, entry, nil
	}
	
	// Check keywords
	if entry, found := bm.checkKeywords(value); found {
		bm.stats.MatchCount++
		return true, entry, nil
	}
	
	// Check regex patterns
	if entry, found := bm.checkRegex(value); found {
		bm.stats.MatchCount++
		return true, entry, nil
	}
	
	return false, nil, nil
}

// GetStats returns blacklist statistics
func (bm *BlacklistManager) GetStats() BlacklistStats {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	return bm.stats
}

// GetEntries returns all blacklist entries
func (bm *BlacklistManager) GetEntries() []*BlacklistEntry {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	
	entries := make([]*BlacklistEntry, 0, len(bm.entries))
	for _, entry := range bm.entries {
		entries = append(entries, entry)
	}
	
	return entries
}

// checkDomain checks if a domain is blacklisted
func (bm *BlacklistManager) checkDomain(value string) (*BlacklistEntry, bool) {
	// Extract domain from URL if needed
	domain := value
	if strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://") {
		if u, err := url.Parse(value); err == nil {
			domain = u.Host
		}
	}
	
	// Check exact match
	if entry, exists := bm.domains[domain]; exists && entry.IsActive {
		return entry, true
	}
	
	return nil, false
}

// checkIP checks if an IP is blacklisted
func (bm *BlacklistManager) checkIP(value string) (*BlacklistEntry, bool) {
	// Parse IP
	ip := net.ParseIP(value)
	if ip == nil {
		return nil, false
	}
	
	// Check exact match
	if entry, exists := bm.ips[value]; exists && entry.IsActive {
		return entry, true
	}
	
	return nil, false
}

// checkURL checks if a URL is blacklisted
func (bm *BlacklistManager) checkURL(value string) (*BlacklistEntry, bool) {
	// Normalize URL
	normalized := bm.normalizeURL(value)
	
	// Check exact match
	if entry, exists := bm.urls[normalized]; exists && entry.IsActive {
		return entry, true
	}
	
	return nil, false
}

// checkEmail checks if an email is blacklisted
func (bm *BlacklistManager) checkEmail(value string) (*BlacklistEntry, bool) {
	// Normalize email
	normalized := strings.ToLower(strings.TrimSpace(value))
	
	// Check exact match
	if entry, exists := bm.emails[normalized]; exists && entry.IsActive {
		return entry, true
	}
	
	return nil, false
}

// checkKeywords checks if any keywords are found in the value
func (bm *BlacklistManager) checkKeywords(value string) (*BlacklistEntry, bool) {
	normalized := strings.ToLower(value)
	
	for keyword, entry := range bm.keywords {
		if entry.IsActive && strings.Contains(normalized, keyword) {
			return entry, true
		}
	}
	
	return nil, false
}

// checkRegex checks if the value matches any regex patterns
func (bm *BlacklistManager) checkRegex(value string) (*BlacklistEntry, bool) {
	for _, entry := range bm.regexes {
		if !entry.IsActive {
			continue
		}
		
		regex, err := regexp.Compile(entry.Value)
		if err != nil {
			continue
		}
		
		if regex.MatchString(value) {
			return entry, true
		}
	}
	
	return nil, false
}

// addToTypeMaps adds an entry to type-specific maps
func (bm *BlacklistManager) addToTypeMaps(entry *BlacklistEntry) {
	switch entry.Type {
	case BlacklistTypeDomain:
		bm.domains[entry.Value] = entry
	case BlacklistTypeIP:
		bm.ips[entry.Value] = entry
	case BlacklistTypeURL:
		normalized := bm.normalizeURL(entry.Value)
		bm.urls[normalized] = entry
	case BlacklistTypeEmail:
		normalized := strings.ToLower(strings.TrimSpace(entry.Value))
		bm.emails[normalized] = entry
	case BlacklistTypeKeyword:
		bm.keywords[strings.ToLower(entry.Value)] = entry
	case BlacklistTypeRegex:
		bm.regexes = append(bm.regexes, entry)
	}
}

// removeFromTypeMaps removes an entry from type-specific maps
func (bm *BlacklistManager) removeFromTypeMaps(entry *BlacklistEntry) {
	switch entry.Type {
	case BlacklistTypeDomain:
		delete(bm.domains, entry.Value)
	case BlacklistTypeIP:
		delete(bm.ips, entry.Value)
	case BlacklistTypeURL:
		normalized := bm.normalizeURL(entry.Value)
		delete(bm.urls, normalized)
	case BlacklistTypeEmail:
		normalized := strings.ToLower(strings.TrimSpace(entry.Value))
		delete(bm.emails, normalized)
	case BlacklistTypeKeyword:
		delete(bm.keywords, strings.ToLower(entry.Value))
	case BlacklistTypeRegex:
		for i, regexEntry := range bm.regexes {
			if regexEntry.ID == entry.ID {
				bm.regexes = append(bm.regexes[:i], bm.regexes[i+1:]...)
				break
			}
		}
	}
}

// normalizeURL normalizes a URL
func (bm *BlacklistManager) normalizeURL(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return urlStr
	}
	
	// Remove trailing slash
	path := u.Path
	if strings.HasSuffix(path, "/") && path != "/" {
		path = path[:len(path)-1]
	}
	
	return fmt.Sprintf("%s://%s%s", u.Scheme, u.Host, path)
}

// generateID generates a unique ID for an entry
func (bm *BlacklistManager) generateID(value string) string {
	hash := sha256.Sum256([]byte(value))
	return fmt.Sprintf("%x", hash)[:16]
}

// updateStats updates the statistics
func (bm *BlacklistManager) updateStats() {
	bm.stats.TotalEntries = int64(len(bm.entries))
	bm.stats.ActiveEntries = 0
	
	for _, entry := range bm.entries {
		if entry.IsActive {
			bm.stats.ActiveEntries++
		}
	}
}

// loadEntries loads existing entries from disk
func (bm *BlacklistManager) loadEntries() error {
	entriesPath := filepath.Join(bm.config.StoragePath, "blacklist_entries.json")
	
	data, err := os.ReadFile(entriesPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No entries file yet
		}
		return err
	}
	
	var entries map[string]*BlacklistEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}
	
	bm.entries = entries
	
	// Rebuild type-specific maps
	for _, entry := range bm.entries {
		bm.addToTypeMaps(entry)
	}
	
	bm.updateStats()
	return nil
}

// saveEntries saves entries to disk
func (bm *BlacklistManager) saveEntries() error {
	entriesPath := filepath.Join(bm.config.StoragePath, "blacklist_entries.json")
	
	data, err := json.MarshalIndent(bm.entries, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(entriesPath, data, 0644)
}