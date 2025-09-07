// T3SS Project
// File: core/indexing/indexer/document_store/compression_utils.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package document_store

import (
	"bytes"
	"compress/gzip"
	"compress/lz4"
	"compress/zlib"
	"fmt"
	"io"
	"sync"

	"github.com/DataDog/zstd"
	"github.com/klauspost/compress/s2"
	"github.com/pierrec/lz4/v4"
)

// CompressionType represents different compression algorithms
type CompressionType string

const (
	CompressionTypeNone   CompressionType = "none"
	CompressionTypeGzip   CompressionType = "gzip"
	CompressionTypeLZ4    CompressionType = "lz4"
	CompressionTypeZstd   CompressionType = "zstd"
	CompressionTypeS2     CompressionType = "s2"
	CompressionTypeZlib   CompressionType = "zlib"
)

// CompressionConfig holds configuration for compression
type CompressionConfig struct {
	Type            CompressionType `yaml:"type"`
	Level           int             `yaml:"level"`
	EnableParallel  bool            `yaml:"enable_parallel"`
	MaxWorkers      int             `yaml:"max_workers"`
	ChunkSize       int             `yaml:"chunk_size"`
	EnableMetrics   bool            `yaml:"enable_metrics"`
}

// CompressionResult represents the result of compression
type CompressionResult struct {
	Data            []byte
	OriginalSize    int
	CompressedSize  int
	CompressionRatio float64
	Algorithm       CompressionType
	TimeMs          int64
}

// CompressionStats tracks compression statistics
type CompressionStats struct {
	TotalCompressions   int64
	TotalDecompressions int64
	TotalBytesIn        int64
	TotalBytesOut       int64
	AverageRatio        float64
	AverageTimeMs       float64
	ErrorCount          int64
}

// CompressionManager manages compression operations
type CompressionManager struct {
	config          CompressionConfig
	stats           *CompressionStats
	statsMutex      sync.RWMutex
	workerPool      chan struct{}
}

// NewCompressionManager creates a new compression manager
func NewCompressionManager(config CompressionConfig) *CompressionManager {
	cm := &CompressionManager{
		config:     config,
		stats:      &CompressionStats{},
		workerPool: make(chan struct{}, config.MaxWorkers),
	}
	
	if cm.config.MaxWorkers <= 0 {
		cm.config.MaxWorkers = 4
	}
	
	return cm
}

// Compress compresses data using the configured algorithm
func (cm *CompressionManager) Compress(data []byte) (*CompressionResult, error) {
	start := cm.getCurrentTime()
	
	var compressed []byte
	var err error
	
	switch cm.config.Type {
	case CompressionTypeNone:
		compressed = data
	case CompressionTypeGzip:
		compressed, err = cm.compressGzip(data)
	case CompressionTypeLZ4:
		compressed, err = cm.compressLZ4(data)
	case CompressionTypeZstd:
		compressed, err = cm.compressZstd(data)
	case CompressionTypeS2:
		compressed, err = cm.compressS2(data)
	case CompressionTypeZlib:
		compressed, err = cm.compressZlib(data)
	default:
		return nil, fmt.Errorf("unsupported compression type: %s", cm.config.Type)
	}
	
	if err != nil {
		cm.updateErrorStats()
		return nil, err
	}
	
	timeMs := cm.getCurrentTime() - start
	ratio := float64(len(compressed)) / float64(len(data))
	
	result := &CompressionResult{
		Data:            compressed,
		OriginalSize:    len(data),
		CompressedSize:  len(compressed),
		CompressionRatio: ratio,
		Algorithm:       cm.config.Type,
		TimeMs:          timeMs,
	}
	
	cm.updateCompressionStats(result)
	return result, nil
}

// Decompress decompresses data using the configured algorithm
func (cm *CompressionManager) Decompress(data []byte) ([]byte, error) {
	start := cm.getCurrentTime()
	
	var decompressed []byte
	var err error
	
	switch cm.config.Type {
	case CompressionTypeNone:
		decompressed = data
	case CompressionTypeGzip:
		decompressed, err = cm.decompressGzip(data)
	case CompressionTypeLZ4:
		decompressed, err = cm.decompressLZ4(data)
	case CompressionTypeZstd:
		decompressed, err = cm.decompressZstd(data)
	case CompressionTypeS2:
		decompressed, err = cm.decompressS2(data)
	case CompressionTypeZlib:
		decompressed, err = cm.decompressZlib(data)
	default:
		return nil, fmt.Errorf("unsupported compression type: %s", cm.config.Type)
	}
	
	if err != nil {
		cm.updateErrorStats()
		return nil, err
	}
	
	timeMs := cm.getCurrentTime() - start
	cm.updateDecompressionStats(len(data), len(decompressed), timeMs)
	
	return decompressed, nil
}

// CompressParallel compresses data in parallel chunks
func (cm *CompressionManager) CompressParallel(data []byte) (*CompressionResult, error) {
	if !cm.config.EnableParallel || len(data) < cm.config.ChunkSize {
		return cm.Compress(data)
	}
	
	chunks := cm.splitIntoChunks(data, cm.config.ChunkSize)
	results := make([]*CompressionResult, len(chunks))
	
	var wg sync.WaitGroup
	errors := make(chan error, len(chunks))
	
	for i, chunk := range chunks {
		wg.Add(1)
		go func(index int, chunkData []byte) {
			defer wg.Done()
			
			cm.workerPool <- struct{}{} // Acquire worker
			defer func() { <-cm.workerPool }() // Release worker
			
			result, err := cm.Compress(chunkData)
			if err != nil {
				errors <- err
				return
			}
			
			results[index] = result
		}(i, chunk)
	}
	
	wg.Wait()
	close(errors)
	
	// Check for errors
	for err := range errors {
		if err != nil {
			return nil, err
		}
	}
	
	// Combine results
	return cm.combineResults(results), nil
}

// DecompressParallel decompresses data in parallel chunks
func (cm *CompressionManager) DecompressParallel(data []byte) ([]byte, error) {
	if !cm.config.EnableParallel {
		return cm.Decompress(data)
	}
	
	// This would implement parallel decompression
	// For now, fall back to regular decompression
	return cm.Decompress(data)
}

// GetStats returns compression statistics
func (cm *CompressionManager) GetStats() CompressionStats {
	cm.statsMutex.RLock()
	defer cm.statsMutex.RUnlock()
	return *cm.stats
}

// ResetStats resets compression statistics
func (cm *CompressionManager) ResetStats() {
	cm.statsMutex.Lock()
	defer cm.statsMutex.Unlock()
	cm.stats = &CompressionStats{}
}

// Private methods

func (cm *CompressionManager) compressGzip(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer, err := gzip.NewWriterLevel(&buf, cm.config.Level)
	if err != nil {
		return nil, err
	}
	
	_, err = writer.Write(data)
	if err != nil {
		writer.Close()
		return nil, err
	}
	
	err = writer.Close()
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (cm *CompressionManager) decompressGzip(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	
	return io.ReadAll(reader)
}

func (cm *CompressionManager) compressLZ4(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer := lz4.NewWriter(&buf)
	
	_, err := writer.Write(data)
	if err != nil {
		writer.Close()
		return nil, err
	}
	
	err = writer.Close()
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (cm *CompressionManager) decompressLZ4(data []byte) ([]byte, error) {
	reader := lz4.NewReader(bytes.NewReader(data))
	return io.ReadAll(reader)
}

func (cm *CompressionManager) compressZstd(data []byte) ([]byte, error) {
	return zstd.Compress(nil, data)
}

func (cm *CompressionManager) decompressZstd(data []byte) ([]byte, error) {
	return zstd.Decompress(nil, data)
}

func (cm *CompressionManager) compressS2(data []byte) ([]byte, error) {
	return s2.Encode(nil, data), nil
}

func (cm *CompressionManager) decompressS2(data []byte) ([]byte, error) {
	return s2.Decode(nil, data)
}

func (cm *CompressionManager) compressZlib(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer, err := zlib.NewWriterLevel(&buf, cm.config.Level)
	if err != nil {
		return nil, err
	}
	
	_, err = writer.Write(data)
	if err != nil {
		writer.Close()
		return nil, err
	}
	
	err = writer.Close()
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (cm *CompressionManager) decompressZlib(data []byte) ([]byte, error) {
	reader, err := zlib.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	
	return io.ReadAll(reader)
}

func (cm *CompressionManager) splitIntoChunks(data []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	
	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunks = append(chunks, data[i:end])
	}
	
	return chunks
}

func (cm *CompressionManager) combineResults(results []*CompressionResult) *CompressionResult {
	if len(results) == 0 {
		return &CompressionResult{}
	}
	
	totalOriginalSize := 0
	totalCompressedSize := 0
	totalTimeMs := int64(0)
	
	var combinedData []byte
	
	for _, result := range results {
		totalOriginalSize += result.OriginalSize
		totalCompressedSize += result.CompressedSize
		totalTimeMs += result.TimeMs
		combinedData = append(combinedData, result.Data...)
	}
	
	ratio := float64(totalCompressedSize) / float64(totalOriginalSize)
	
	return &CompressionResult{
		Data:            combinedData,
		OriginalSize:    totalOriginalSize,
		CompressedSize:  totalCompressedSize,
		CompressionRatio: ratio,
		Algorithm:       results[0].Algorithm,
		TimeMs:          totalTimeMs,
	}
}

func (cm *CompressionManager) updateCompressionStats(result *CompressionResult) {
	cm.statsMutex.Lock()
	defer cm.statsMutex.Unlock()
	
	cm.stats.TotalCompressions++
	cm.stats.TotalBytesIn += int64(result.OriginalSize)
	cm.stats.TotalBytesOut += int64(result.CompressedSize)
	
	// Update average ratio
	if cm.stats.TotalCompressions > 0 {
		cm.stats.AverageRatio = float64(cm.stats.TotalBytesOut) / float64(cm.stats.TotalBytesIn)
	}
	
	// Update average time
	if cm.stats.TotalCompressions > 0 {
		cm.stats.AverageTimeMs = (cm.stats.AverageTimeMs*float64(cm.stats.TotalCompressions-1) + float64(result.TimeMs)) / float64(cm.stats.TotalCompressions)
	}
}

func (cm *CompressionManager) updateDecompressionStats(compressedSize, decompressedSize int, timeMs int64) {
	cm.statsMutex.Lock()
	defer cm.statsMutex.Unlock()
	
	cm.stats.TotalDecompressions++
	cm.stats.TotalBytesIn += int64(compressedSize)
	cm.stats.TotalBytesOut += int64(decompressedSize)
}

func (cm *CompressionManager) updateErrorStats() {
	cm.statsMutex.Lock()
	defer cm.statsMutex.Unlock()
	cm.stats.ErrorCount++
}

func (cm *CompressionManager) getCurrentTime() int64 {
	// This would return current time in milliseconds
	// For now, return 0
	return 0
}