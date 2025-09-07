// T3SS Project
// File: core/storage/archival/glacier_analog.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package archival

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ArchiveTier represents different storage tiers
type ArchiveTier int

const (
	TierStandard ArchiveTier = iota
	TierInfrequent
	TierArchive
	TierDeepArchive
)

// ArchiveStatus represents the status of an archive operation
type ArchiveStatus int

const (
	StatusPending ArchiveStatus = iota
	StatusInProgress
	StatusCompleted
	StatusFailed
	StatusRetrieving
	StatusRetrieved
)

// ArchiveConfig holds configuration for the glacier analog
type ArchiveConfig struct {
	StoragePath     string        `json:"storage_path"`
	MaxFileSize     int64         `json:"max_file_size"`
	CompressionLevel int          `json:"compression_level"`
	RetentionDays   int           `json:"retention_days"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
	EnableEncryption bool         `json:"enable_encryption"`
	EncryptionKey   string        `json:"encryption_key"`
}

// ArchiveEntry represents a single archived item
type ArchiveEntry struct {
	ID           string        `json:"id"`
	Path         string        `json:"path"`
	Size         int64         `json:"size"`
	Tier         ArchiveTier   `json:"tier"`
	Status       ArchiveStatus `json:"status"`
	CreatedAt    time.Time     `json:"created_at"`
	LastAccessed time.Time     `json:"last_accessed"`
	Checksum     string        `json:"checksum"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// ArchiveRequest represents a request to archive data
type ArchiveRequest struct {
	ID       string                 `json:"id"`
	Path     string                 `json:"path"`
	Tier     ArchiveTier           `json:"tier"`
	Metadata map[string]interface{} `json:"metadata"`
}

// ArchiveResponse represents the response from an archive operation
type ArchiveResponse struct {
	ID        string        `json:"id"`
	Status    ArchiveStatus `json:"status"`
	Message   string        `json:"message"`
	Checksum  string        `json:"checksum"`
	Size      int64         `json:"size"`
	CreatedAt time.Time     `json:"created_at"`
}

// RetrievalRequest represents a request to retrieve archived data
type RetrievalRequest struct {
	ID string `json:"id"`
}

// RetrievalResponse represents the response from a retrieval operation
type RetrievalResponse struct {
	ID        string        `json:"id"`
	Status    ArchiveStatus `json:"status"`
	Path      string        `json:"path"`
	Message   string        `json:"message"`
	Checksum  string        `json:"checksum"`
	Size      int64         `json:"size"`
	CreatedAt time.Time     `json:"created_at"`
}

// GlacierAnalog provides AWS Glacier-like archival functionality
type GlacierAnalog struct {
	config     ArchiveConfig
	entries    map[string]*ArchiveEntry
	mu         sync.RWMutex
	stopChan   chan struct{}
	wg         sync.WaitGroup
}

// NewGlacierAnalog creates a new glacier analog instance
func NewGlacierAnalog(config ArchiveConfig) *GlacierAnalog {
	return &GlacierAnalog{
		config:  config,
		entries: make(map[string]*ArchiveEntry),
		stopChan: make(chan struct{}),
	}
}

// Start initializes the glacier analog
func (ga *GlacierAnalog) Start() error {
	log.Println("Starting Glacier Analog...")
	
	// Create storage directory if it doesn't exist
	if err := os.MkdirAll(ga.config.StoragePath, 0755); err != nil {
		return fmt.Errorf("failed to create storage directory: %w", err)
	}
	
	// Load existing entries
	if err := ga.loadEntries(); err != nil {
		log.Printf("Warning: failed to load existing entries: %v", err)
	}
	
	// Start cleanup routine
	ga.wg.Add(1)
	go ga.cleanupRoutine()
	
	log.Println("Glacier Analog started successfully")
	return nil
}

// Stop stops the glacier analog
func (ga *GlacierAnalog) Stop() error {
	log.Println("Stopping Glacier Analog...")
	
	close(ga.stopChan)
	ga.wg.Wait()
	
	// Save entries
	if err := ga.saveEntries(); err != nil {
		log.Printf("Warning: failed to save entries: %v", err)
	}
	
	log.Println("Glacier Analog stopped")
	return nil
}

// Archive archives data to the specified tier
func (ga *GlacierAnalog) Archive(req ArchiveRequest) (*ArchiveResponse, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	
	// Check if entry already exists
	if _, exists := ga.entries[req.ID]; exists {
		return nil, fmt.Errorf("entry with ID %s already exists", req.ID)
	}
	
	// Create archive entry
	entry := &ArchiveEntry{
		ID:        req.ID,
		Path:      req.Path,
		Tier:      req.Tier,
		Status:    StatusPending,
		CreatedAt: time.Now(),
		Metadata:  req.Metadata,
	}
	
	// Get file info
	fileInfo, err := os.Stat(req.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}
	
	entry.Size = fileInfo.Size()
	
	// Check file size limit
	if entry.Size > ga.config.MaxFileSize {
		return nil, fmt.Errorf("file size %d exceeds maximum allowed size %d", entry.Size, ga.config.MaxFileSize)
	}
	
	// Calculate checksum
	checksum, err := ga.calculateChecksum(req.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate checksum: %w", err)
	}
	entry.Checksum = checksum
	
	// Archive the file
	if err := ga.archiveFile(entry); err != nil {
		return nil, fmt.Errorf("failed to archive file: %w", err)
	}
	
	entry.Status = StatusCompleted
	entry.LastAccessed = time.Now()
	
	// Store entry
	ga.entries[req.ID] = entry
	
	// Save entries
	if err := ga.saveEntries(); err != nil {
		log.Printf("Warning: failed to save entries: %v", err)
	}
	
	return &ArchiveResponse{
		ID:        entry.ID,
		Status:    entry.Status,
		Message:   "Archive completed successfully",
		Checksum:  entry.Checksum,
		Size:      entry.Size,
		CreatedAt: entry.CreatedAt,
	}, nil
}

// Retrieve retrieves archived data
func (ga *GlacierAnalog) Retrieve(req RetrievalRequest) (*RetrievalResponse, error) {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	
	entry, exists := ga.entries[req.ID]
	if !exists {
		return nil, fmt.Errorf("entry with ID %s not found", req.ID)
	}
	
	entry.Status = StatusRetrieving
	entry.LastAccessed = time.Now()
	
	// Retrieve the file
	retrievedPath, err := ga.retrieveFile(entry)
	if err != nil {
		entry.Status = StatusFailed
		return nil, fmt.Errorf("failed to retrieve file: %w", err)
	}
	
	entry.Status = StatusRetrieved
	
	// Save entries
	if err := ga.saveEntries(); err != nil {
		log.Printf("Warning: failed to save entries: %v", err)
	}
	
	return &RetrievalResponse{
		ID:        entry.ID,
		Status:    entry.Status,
		Path:      retrievedPath,
		Message:   "Retrieval completed successfully",
		Checksum:  entry.Checksum,
		Size:      entry.Size,
		CreatedAt: entry.CreatedAt,
	}, nil
}

// ListEntries lists all archived entries
func (ga *GlacierAnalog) ListEntries() ([]*ArchiveEntry, error) {
	ga.mu.RLock()
	defer ga.mu.RUnlock()
	
	entries := make([]*ArchiveEntry, 0, len(ga.entries))
	for _, entry := range ga.entries {
		entries = append(entries, entry)
	}
	
	// Sort by creation time
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].CreatedAt.Before(entries[j].CreatedAt)
	})
	
	return entries, nil
}

// GetEntry retrieves a specific entry
func (ga *GlacierAnalog) GetEntry(id string) (*ArchiveEntry, error) {
	ga.mu.RLock()
	defer ga.mu.RUnlock()
	
	entry, exists := ga.entries[id]
	if !exists {
		return nil, fmt.Errorf("entry with ID %s not found", id)
	}
	
	return entry, nil
}

// DeleteEntry deletes an archived entry
func (ga *GlacierAnalog) DeleteEntry(id string) error {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	
	entry, exists := ga.entries[id]
	if !exists {
		return fmt.Errorf("entry with ID %s not found", id)
	}
	
	// Delete the archived file
	archivePath := ga.getArchivePath(entry.ID)
	if err := os.Remove(archivePath); err != nil {
		log.Printf("Warning: failed to delete archive file %s: %v", archivePath, err)
	}
	
	// Remove entry
	delete(ga.entries, id)
	
	// Save entries
	if err := ga.saveEntries(); err != nil {
		log.Printf("Warning: failed to save entries: %v", err)
	}
	
	return nil
}

// archiveFile archives a file to the storage path
func (ga *GlacierAnalog) archiveFile(entry *ArchiveEntry) error {
	// Create archive path
	archivePath := ga.getArchivePath(entry.ID)
	
	// Create archive file
	archiveFile, err := os.Create(archivePath)
	if err != nil {
		return fmt.Errorf("failed to create archive file: %w", err)
	}
	defer archiveFile.Close()
	
	// Create gzip writer
	gzipWriter := gzip.NewWriter(archiveFile)
	defer gzipWriter.Close()
	
	// Create tar writer
	tarWriter := tar.NewWriter(gzipWriter)
	defer tarWriter.Close()
	
	// Open source file
	sourceFile, err := os.Open(entry.Path)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer sourceFile.Close()
	
	// Get file info
	fileInfo, err := sourceFile.Stat()
	if err != nil {
		return fmt.Errorf("failed to get file info: %w", err)
	}
	
	// Create tar header
	header := &tar.Header{
		Name: filepath.Base(entry.Path),
		Size: fileInfo.Size(),
		Mode: int64(fileInfo.Mode()),
		ModTime: fileInfo.ModTime(),
	}
	
	// Write tar header
	if err := tarWriter.WriteHeader(header); err != nil {
		return fmt.Errorf("failed to write tar header: %w", err)
	}
	
	// Copy file content
	if _, err := io.Copy(tarWriter, sourceFile); err != nil {
		return fmt.Errorf("failed to copy file content: %w", err)
	}
	
	return nil
}

// retrieveFile retrieves a file from the archive
func (ga *GlacierAnalog) retrieveFile(entry *ArchiveEntry) (string, error) {
	// Create retrieval path
	retrievalPath := filepath.Join(ga.config.StoragePath, "retrieved", entry.ID)
	
	// Create retrieval directory
	if err := os.MkdirAll(filepath.Dir(retrievalPath), 0755); err != nil {
		return "", fmt.Errorf("failed to create retrieval directory: %w", err)
	}
	
	// Open archive file
	archivePath := ga.getArchivePath(entry.ID)
	archiveFile, err := os.Open(archivePath)
	if err != nil {
		return "", fmt.Errorf("failed to open archive file: %w", err)
	}
	defer archiveFile.Close()
	
	// Create gzip reader
	gzipReader, err := gzip.NewReader(archiveFile)
	if err != nil {
		return "", fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzipReader.Close()
	
	// Create tar reader
	tarReader := tar.NewReader(gzipReader)
	
	// Read tar header
	header, err := tarReader.Next()
	if err != nil {
		return "", fmt.Errorf("failed to read tar header: %w", err)
	}
	
	// Create retrieved file
	retrievedFile, err := os.Create(retrievalPath)
	if err != nil {
		return "", fmt.Errorf("failed to create retrieved file: %w", err)
	}
	defer retrievedFile.Close()
	
	// Copy file content
	if _, err := io.Copy(retrievedFile, tarReader); err != nil {
		return "", fmt.Errorf("failed to copy file content: %w", err)
	}
	
	return retrievalPath, nil
}

// calculateChecksum calculates SHA256 checksum of a file
func (ga *GlacierAnalog) calculateChecksum(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	
	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// getArchivePath returns the path for an archived file
func (ga *GlacierAnalog) getArchivePath(id string) string {
	return filepath.Join(ga.config.StoragePath, "archives", id+".tar.gz")
}

// loadEntries loads existing entries from disk
func (ga *GlacierAnalog) loadEntries() error {
	entriesPath := filepath.Join(ga.config.StoragePath, "entries.json")
	
	data, err := os.ReadFile(entriesPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No entries file yet
		}
		return err
	}
	
	var entries map[string]*ArchiveEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}
	
	ga.entries = entries
	return nil
}

// saveEntries saves entries to disk
func (ga *GlacierAnalog) saveEntries() error {
	entriesPath := filepath.Join(ga.config.StoragePath, "entries.json")
	
	data, err := json.MarshalIndent(ga.entries, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(entriesPath, data, 0644)
}

// cleanupRoutine runs periodic cleanup
func (ga *GlacierAnalog) cleanupRoutine() {
	defer ga.wg.Done()
	
	ticker := time.NewTicker(ga.config.CleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ga.cleanup()
		case <-ga.stopChan:
			return
		}
	}
}

// cleanup removes old entries based on retention policy
func (ga *GlacierAnalog) cleanup() {
	ga.mu.Lock()
	defer ga.mu.Unlock()
	
	cutoff := time.Now().AddDate(0, 0, -ga.config.RetentionDays)
	
	for id, entry := range ga.entries {
		if entry.CreatedAt.Before(cutoff) {
			// Delete archive file
			archivePath := ga.getArchivePath(entry.ID)
			if err := os.Remove(archivePath); err != nil {
				log.Printf("Warning: failed to delete old archive file %s: %v", archivePath, err)
			}
			
			// Remove entry
			delete(ga.entries, id)
		}
	}
	
	// Save entries
	if err := ga.saveEntries(); err != nil {
		log.Printf("Warning: failed to save entries during cleanup: %v", err)
	}
}