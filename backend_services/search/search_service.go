// T3SS Project
// File: backend_services/search/search_service.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package search

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/any"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/t3ss/shared_libs/proto/search"
)

// SearchService implements the search gRPC service
type SearchService struct {
	pb.UnimplementedSearchServiceServer
	indexer    Indexer
	ranker     Ranker
	analyzer   QueryAnalyzer
	cache      Cache
	analytics  Analytics
}

// Indexer interface for document indexing
type Indexer interface {
	Search(ctx context.Context, query string, filters map[string]string, from, size int) (*SearchResults, error)
	GetDocument(ctx context.Context, docID string) (*Document, error)
	GetSuggestions(ctx context.Context, query string, limit int) ([]string, error)
	GetAutocomplete(ctx context.Context, query string, limit int) ([]string, error)
}

// Ranker interface for document ranking
type Ranker interface {
	RankDocuments(ctx context.Context, docs []*Document, query string, userID string) ([]*RankedDocument, error)
	ReRankDocuments(ctx context.Context, docs []*RankedDocument, query string, userID string) ([]*RankedDocument, error)
}

// QueryAnalyzer interface for query analysis
type QueryAnalyzer interface {
	AnalyzeQuery(ctx context.Context, query string) (*QueryAnalysis, error)
	ExpandQuery(ctx context.Context, query string) ([]string, error)
	CorrectSpelling(ctx context.Context, query string) (string, error)
}

// Cache interface for caching search results
type Cache interface {
	Get(ctx context.Context, key string) (*SearchResults, error)
	Set(ctx context.Context, key string, results *SearchResults, ttl time.Duration) error
	GetSuggestions(ctx context.Context, key string) ([]string, error)
	SetSuggestions(ctx context.Context, key string, suggestions []string, ttl time.Duration) error
}

// Analytics interface for tracking search analytics
type Analytics interface {
	TrackSearch(ctx context.Context, query string, userID, sessionID string, results []*RankedDocument) error
	TrackClick(ctx context.Context, resultID, query string, position int, userID, sessionID string) error
	TrackImpression(ctx context.Context, resultIDs []string, query, userID, sessionID string) error
}

// Document represents a searchable document
type Document struct {
	ID          string            `json:"id"`
	Title       string            `json:"title"`
	Content     string            `json:"content"`
	URL         string            `json:"url"`
	Score       float64           `json:"score"`
	Metadata    map[string]string `json:"metadata"`
	Categories  []string          `json:"categories"`
	Language    string            `json:"language"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	PageRank    float64           `json:"page_rank"`
	ContentType string            `json:"content_type"`
}

// RankedDocument represents a document with ranking information
type RankedDocument struct {
	Document      *Document `json:"document"`
	FinalScore    float64   `json:"final_score"`
	RelevanceScore float64  `json:"relevance_score"`
	AuthorityScore float64  `json:"authority_score"`
	FreshnessScore float64  `json:"freshness_score"`
	PersonalizationScore float64 `json:"personalization_score"`
	Explanation   string    `json:"explanation"`
	Rank          int       `json:"rank"`
}

// SearchResults represents search results
type SearchResults struct {
	Documents    []*RankedDocument `json:"documents"`
	Total        int64             `json:"total"`
	From         int               `json:"from"`
	Size         int               `json:"size"`
	MaxScore     float64           `json:"max_score"`
	TookMs       float64           `json:"took_ms"`
	Suggestions  []string          `json:"suggestions"`
	Facets       map[string]interface{} `json:"facets"`
	CorrectedQuery string          `json:"corrected_query"`
	ConfidenceScore float64        `json:"confidence_score"`
}

// QueryAnalysis represents analyzed query
type QueryAnalysis struct {
	OriginalQuery    string            `json:"original_query"`
	ProcessedQuery   string            `json:"processed_query"`
	Terms            []string          `json:"terms"`
	Intent           string            `json:"intent"`
	Filters          map[string]string `json:"filters"`
	BoostFields      map[string]float64 `json:"boost_fields"`
	ExpansionTerms   []string          `json:"expansion_terms"`
	CorrectedQuery   string            `json:"corrected_query"`
	ConfidenceScore  float64           `json:"confidence_score"`
	ProcessingTime   time.Duration     `json:"processing_time"`
}

// NewSearchService creates a new search service
func NewSearchService(indexer Indexer, ranker Ranker, analyzer QueryAnalyzer, cache Cache, analytics Analytics) *SearchService {
	return &SearchService{
		indexer:   indexer,
		ranker:    ranker,
		analyzer:  analyzer,
		cache:     cache,
		analytics: analytics,
	}
}

// Search performs a search query
func (s *SearchService) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	start := time.Now()
	
	// Set defaults
	if req.Page <= 0 {
		req.Page = 1
	}
	if req.PageSize <= 0 {
		req.PageSize = 10
	}
	if req.PageSize > 100 {
		req.PageSize = 100
	}

	from := (req.Page - 1) * req.PageSize

	// Generate cache key
	cacheKey := s.generateCacheKey(req)

	// Check cache first
	if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != nil {
		return s.convertSearchResultsToProto(cached), nil
	}

	// Analyze query
	analysis, err := s.analyzer.AnalyzeQuery(ctx, req.Query)
	if err != nil {
		return &pb.SearchResponse{
			Results: []*pb.SearchResult{},
			Total:   0,
			Page:    int32(req.Page),
			PageSize: int32(req.PageSize),
			QueryTimeMs: float64(time.Since(start).Nanoseconds()) / 1e6,
		}, status.Error(codes.Internal, "failed to analyze query")
	}

	// Search documents
	searchResults, err := s.indexer.Search(ctx, analysis.ProcessedQuery, analysis.Filters, from, req.PageSize)
	if err != nil {
		return &pb.SearchResponse{
			Results: []*pb.SearchResult{},
			Total:   0,
			Page:    int32(req.Page),
			PageSize: int32(req.PageSize),
			QueryTimeMs: float64(time.Since(start).Nanoseconds()) / 1e6,
		}, status.Error(codes.Internal, "failed to search documents")
	}

	// Rank documents
	rankedDocs, err := s.ranker.RankDocuments(ctx, s.convertToDocuments(searchResults.Documents), req.Query, req.UserId)
	if err != nil {
		return &pb.SearchResponse{
			Results: []*pb.SearchResult{},
			Total:   0,
			Page:    int32(req.Page),
			PageSize: int32(req.PageSize),
			QueryTimeMs: float64(time.Since(start).Nanoseconds()) / 1e6,
		}, status.Error(codes.Internal, "failed to rank documents")
	}

	// Update search results with ranked documents
	searchResults.Documents = rankedDocs
	searchResults.TookMs = float64(time.Since(start).Nanoseconds()) / 1e6
	searchResults.CorrectedQuery = analysis.CorrectedQuery
	searchResults.ConfidenceScore = analysis.ConfidenceScore

	// Cache results
	s.cache.Set(ctx, cacheKey, searchResults, 5*time.Minute)

	// Track analytics
	go s.analytics.TrackSearch(ctx, req.Query, req.UserId, req.SessionId, rankedDocs)

	// Convert to protobuf response
	response := s.convertSearchResultsToProto(searchResults)
	response.QueryTimeMs = float64(time.Since(start).Nanoseconds()) / 1e6

	return response, nil
}

// GetSuggestions gets search suggestions
func (s *SearchService) GetSuggestions(ctx context.Context, req *pb.SuggestionsRequest) (*pb.SuggestionsResponse, error) {
	// Set default limit
	limit := int(req.Limit)
	if limit <= 0 {
		limit = 10
	}
	if limit > 50 {
		limit = 50
	}

	// Generate cache key
	cacheKey := fmt.Sprintf("suggestions:%s:%d", req.Query, limit)

	// Check cache first
	if cached, err := s.cache.GetSuggestions(ctx, cacheKey); err == nil && len(cached) > 0 {
		return &pb.SuggestionsResponse{
			Suggestions:   cached,
			Confidence:    0.8,
			OriginalQuery: req.Query,
		}, nil
	}

	// Get suggestions from indexer
	suggestions, err := s.indexer.GetSuggestions(ctx, req.Query, limit)
	if err != nil {
		return &pb.SuggestionsResponse{
			Suggestions:   []string{},
			Confidence:    0.0,
			OriginalQuery: req.Query,
		}, status.Error(codes.Internal, "failed to get suggestions")
	}

	// Cache suggestions
	s.cache.SetSuggestions(ctx, cacheKey, suggestions, 10*time.Minute)

	return &pb.SuggestionsResponse{
		Suggestions:   suggestions,
		Confidence:    0.8,
		OriginalQuery: req.Query,
	}, nil
}

// GetAutocomplete gets autocomplete suggestions
func (s *SearchService) GetAutocomplete(ctx context.Context, req *pb.AutocompleteRequest) (*pb.AutocompleteResponse, error) {
	// Set default limit
	limit := int(req.Limit)
	if limit <= 0 {
		limit = 10
	}
	if limit > 50 {
		limit = 50
	}

	// Generate cache key
	cacheKey := fmt.Sprintf("autocomplete:%s:%d", req.Query, limit)

	// Check cache first
	if cached, err := s.cache.GetSuggestions(ctx, cacheKey); err == nil && len(cached) > 0 {
		return &pb.AutocompleteResponse{
			Completions:   cached,
			Confidence:    0.9,
			OriginalQuery: req.Query,
		}, nil
	}

	// Get autocomplete from indexer
	completions, err := s.indexer.GetAutocomplete(ctx, req.Query, limit)
	if err != nil {
		return &pb.AutocompleteResponse{
			Completions:   []string{},
			Confidence:    0.0,
			OriginalQuery: req.Query,
		}, status.Error(codes.Internal, "failed to get autocomplete")
	}

	// Cache completions
	s.cache.SetSuggestions(ctx, cacheKey, completions, 15*time.Minute)

	return &pb.AutocompleteResponse{
		Completions:   completions,
		Confidence:    0.9,
		OriginalQuery: req.Query,
	}, nil
}

// TrackClick tracks a click on a search result
func (s *SearchService) TrackClick(ctx context.Context, req *pb.ClickTrackingRequest) (*pb.ClickTrackingResponse, error) {
	err := s.analytics.TrackClick(ctx, req.ResultId, req.Query, int(req.Position), req.UserId, req.SessionId)
	if err != nil {
		return &pb.ClickTrackingResponse{
			Success: false,
			Message: "Failed to track click",
		}, status.Error(codes.Internal, "failed to track click")
	}

	return &pb.ClickTrackingResponse{
		Success: true,
		Message: "Click tracked successfully",
	}, nil
}

// TrackImpression tracks impressions of search results
func (s *SearchService) TrackImpression(ctx context.Context, req *pb.ImpressionTrackingRequest) (*pb.ImpressionTrackingResponse, error) {
	err := s.analytics.TrackImpression(ctx, req.ResultIds, req.Query, req.UserId, req.SessionId)
	if err != nil {
		return &pb.ImpressionTrackingResponse{
			Success: false,
			Message: "Failed to track impression",
		}, status.Error(codes.Internal, "failed to track impression")
	}

	return &pb.ImpressionTrackingResponse{
		Success: true,
		Message: "Impression tracked successfully",
	}, nil
}

// HealthCheck performs health check
func (s *SearchService) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	// Check dependencies
	status := pb.HealthCheckResponse_SERVING
	message := "Service is healthy"
	details := make(map[string]string)

	// Test indexer
	if _, err := s.indexer.Search(ctx, "test", map[string]string{}, 0, 1); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Indexer is not responding"
		details["indexer"] = "unhealthy"
	} else {
		details["indexer"] = "healthy"
	}

	// Test ranker
	if _, err := s.ranker.RankDocuments(ctx, []*Document{}, "test", ""); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Ranker is not responding"
		details["ranker"] = "unhealthy"
	} else {
		details["ranker"] = "healthy"
	}

	// Test analyzer
	if _, err := s.analyzer.AnalyzeQuery(ctx, "test"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Query analyzer is not responding"
		details["analyzer"] = "unhealthy"
	} else {
		details["analyzer"] = "healthy"
	}

	return &pb.HealthCheckResponse{
		Status:  status,
		Message: message,
		Details: details,
	}, nil
}

// Helper methods

// generateCacheKey generates a cache key for search request
func (s *SearchService) generateCacheKey(req *pb.SearchRequest) string {
	// Create a hash of the request parameters
	key := fmt.Sprintf("search:%s:%d:%d:%s:%s", 
		req.Query, req.Page, req.PageSize, req.UserId, req.SessionId)
	
	// Add filters to key
	if len(req.Filters) > 0 {
		var filterKeys []string
		for k, v := range req.Filters {
			filterKeys = append(filterKeys, fmt.Sprintf("%s:%s", k, v))
		}
		sort.Strings(filterKeys)
		key += ":" + strings.Join(filterKeys, ",")
	}

	// Add boost fields to key
	if len(req.BoostFields) > 0 {
		var boostKeys []string
		for _, field := range req.BoostFields {
			boostKeys = append(boostKeys, field)
		}
		sort.Strings(boostKeys)
		key += ":" + strings.Join(boostKeys, ",")
	}

	return key
}

// convertToDocuments converts ranked documents to documents
func (s *SearchService) convertToDocuments(rankedDocs []*RankedDocument) []*Document {
	docs := make([]*Document, len(rankedDocs))
	for i, rankedDoc := range rankedDocs {
		docs[i] = rankedDoc.Document
	}
	return docs
}

// convertSearchResultsToProto converts SearchResults to protobuf
func (s *SearchService) convertSearchResultsToProto(results *SearchResults) *pb.SearchResponse {
	protoResults := make([]*pb.SearchResult, len(results.Documents))
	for i, doc := range results.Documents {
		protoResults[i] = &pb.SearchResult{
			Id:          doc.Document.ID,
			Title:       doc.Document.Title,
			Url:         doc.Document.URL,
			Snippet:     s.generateSnippet(doc.Document.Content),
			Score:       doc.FinalScore,
			Metadata:    s.convertMetadataToProto(doc.Document.Metadata),
			Highlights:  []string{}, // TODO: Implement highlighting
			ContentType: doc.Document.ContentType,
			LastUpdated: timestamppb.New(doc.Document.UpdatedAt),
			PageRank:    doc.Document.PageRank,
			Categories:  doc.Document.Categories,
		}
	}

	return &pb.SearchResponse{
		Results:        protoResults,
		Total:          results.Total,
		Page:           int32(results.From/results.Size + 1),
		PageSize:       int32(results.Size),
		QueryTimeMs:    results.TookMs,
		Facets:         s.convertFacetsToProto(results.Facets),
		Suggestions:    results.Suggestions,
		CorrectedQuery: results.CorrectedQuery,
		ConfidenceScore: results.ConfidenceScore,
	}
}

// convertMetadataToProto converts metadata to protobuf Any
func (s *SearchService) convertMetadataToProto(metadata map[string]string) map[string]*any.Any {
	protoMetadata := make(map[string]*any.Any)
	for k, v := range metadata {
		// For simplicity, store as string values
		// In production, you might want to use proper Any types
		protoMetadata[k] = &any.Any{
			TypeUrl: "type.googleapis.com/google.protobuf.StringValue",
			Value:   []byte(fmt.Sprintf(`"%s"`, v)),
		}
	}
	return protoMetadata
}

// convertFacetsToProto converts facets to protobuf Any
func (s *SearchService) convertFacetsToProto(facets map[string]interface{}) map[string]*any.Any {
	protoFacets := make(map[string]*any.Any)
	for k, v := range facets {
		// Convert to JSON and store as Any
		if jsonData, err := json.Marshal(v); err == nil {
			protoFacets[k] = &any.Any{
				TypeUrl: "type.googleapis.com/google.protobuf.Value",
				Value:   jsonData,
			}
		}
	}
	return protoFacets
}

// generateSnippet generates a snippet from document content
func (s *SearchService) generateSnippet(content string) string {
	// Simple snippet generation - take first 200 characters
	if len(content) <= 200 {
		return content
	}
	
	// Find the last complete word within 200 characters
	snippet := content[:200]
	lastSpace := strings.LastIndex(snippet, " ")
	if lastSpace > 100 { // Only truncate if we have enough content
		snippet = snippet[:lastSpace]
	}
	
	return snippet + "..."
}

// RegisterSearchService registers the search service with gRPC server
func RegisterSearchService(s *grpc.Server, service *SearchService) {
	pb.RegisterSearchServiceServer(s, service)
}