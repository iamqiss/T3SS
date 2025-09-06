// T3SS Project
// File: backend_services/ranking/ranking_service.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package ranking

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/t3ss/shared_libs/proto/ranking"
)

// RankingService implements the ranking gRPC service
type RankingService struct {
	pb.UnimplementedRankingServiceServer
	pageRankEngine PageRankEngine
	mlRanker       MLRanker
	featureExtractor FeatureExtractor
	userProfiler   UserProfiler
	modelManager   ModelManager
}

// PageRankEngine interface for PageRank computation
type PageRankEngine interface {
	ComputePageRank(ctx context.Context, links []*Link, config *PageRankConfig) (*PageRankResults, error)
	GetPageRank(ctx context.Context, url string) (float64, error)
	UpdatePageRank(ctx context.Context, newLinks []*Link, removedLinks []string) error
}

// MLRanker interface for machine learning-based ranking
type MLRanker interface {
	RankDocuments(ctx context.Context, docs []*Document, query string, userID string, modelName string) ([]*RankedDocument, error)
	ReRankDocuments(ctx context.Context, docs []*RankedDocument, query string, userID string, modelName string) ([]*RankedDocument, error)
	TrainModel(ctx context.Context, examples []*TrainingExample, config *ModelConfig) (*ModelTrainingResult, error)
	UpdateModel(ctx context.Context, modelName string, examples []*TrainingExample) error
	GetModel(ctx context.Context, modelName string) (*RankingModel, error)
}

// FeatureExtractor interface for feature extraction
type FeatureExtractor interface {
	ExtractFeatures(ctx context.Context, docs []*Document, query string, userID string) ([]*FeatureVector, error)
	GetFeatureImportance(ctx context.Context, modelName string) (map[string]float64, error)
}

// UserProfiler interface for user profiling
type UserProfiler interface {
	GetUserProfile(ctx context.Context, userID string) (*UserProfile, error)
	UpdateUserProfile(ctx context.Context, userID string, profile *UserProfile) error
	GetPersonalizedRanking(ctx context.Context, userID string, docs []*Document, query string) ([]*RankedDocument, error)
}

// ModelManager interface for model management
type ModelManager interface {
	CreateModel(ctx context.Context, name string, config *ModelConfig) error
	DeleteModel(ctx context.Context, name string) error
	ListModels(ctx context.Context) ([]*RankingModel, error)
	GetModelStatus(ctx context.Context, name string) (string, error)
}

// Document represents a document to be ranked
type Document struct {
	ID          string            `json:"id"`
	Title       string            `json:"title"`
	Content     string            `json:"content"`
	URL         string            `json:"url"`
	Score       float64           `json:"score"`
	Features    map[string]float64 `json:"features"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Categories  []string          `json:"categories"`
	Language    string            `json:"language"`
	PageRank    float64           `json:"page_rank"`
	AuthorityScore float64        `json:"authority_score"`
	HubScore    float64           `json:"hub_score"`
}

// RankedDocument represents a ranked document
type RankedDocument struct {
	Document            *Document `json:"document"`
	FinalScore          float64   `json:"final_score"`
	RelevanceScore      float64   `json:"relevance_score"`
	AuthorityScore      float64   `json:"authority_score"`
	FreshnessScore      float64   `json:"freshness_score"`
	PersonalizationScore float64  `json:"personalization_score"`
	FeatureScores       map[string]float64 `json:"feature_scores"`
	Explanation         string    `json:"explanation"`
	Rank                int       `json:"rank"`
}

// Link represents a link between pages
type Link struct {
	FromURL      string    `json:"from_url"`
	ToURL        string    `json:"to_url"`
	Weight       float64   `json:"weight"`
	AnchorText   string    `json:"anchor_text"`
	LinkType     string    `json:"link_type"`
	DiscoveredAt time.Time `json:"discovered_at"`
}

// PageRankConfig represents PageRank configuration
type PageRankConfig struct {
	DampingFactor       float64            `json:"damping_factor"`
	ConvergenceThreshold float64           `json:"convergence_threshold"`
	MaxIterations       int                `json:"max_iterations"`
	EnableParallel      bool               `json:"enable_parallel"`
	Algorithm           string             `json:"algorithm"`
	TopicWeights        map[string]float64 `json:"topic_weights"`
	UserID              string             `json:"user_id"`
}

// PageRankResults represents PageRank computation results
type PageRankResults struct {
	Results         []*PageRankResult `json:"results"`
	Iterations      int               `json:"iterations"`
	Converged       bool              `json:"converged"`
	ComputationTime time.Duration     `json:"computation_time"`
}

// PageRankResult represents a single PageRank result
type PageRankResult struct {
	URL           string  `json:"url"`
	PageRank      float64 `json:"page_rank"`
	InLinks       int     `json:"in_links"`
	OutLinks      int     `json:"out_links"`
	AuthorityScore float64 `json:"authority_score"`
	HubScore      float64 `json:"hub_score"`
}

// TrainingExample represents a training example for ML models
type TrainingExample struct {
	Features    []float64         `json:"features"`
	Target      float64           `json:"target"`
	Query       string            `json:"query"`
	DocumentID  string            `json:"document_id"`
	UserID      string            `json:"user_id"`
	Context     map[string]string `json:"context"`
	Embedding   []float64         `json:"embedding"`
	Position    int               `json:"position"`
	Clicked     bool              `json:"clicked"`
	DwellTime   float64           `json:"dwell_time"`
}

// ModelConfig represents model configuration
type ModelConfig struct {
	InputSize              int               `json:"input_size"`
	HiddenSize             int               `json:"hidden_size"`
	OutputSize             int               `json:"output_size"`
	NumLayers              int               `json:"num_layers"`
	NumHeads               int               `json:"num_heads"`
	NumAttentionLayers     int               `json:"num_attention_layers"`
	LearningRate           float64           `json:"learning_rate"`
	BatchSize              int               `json:"batch_size"`
	Epochs                 int               `json:"epochs"`
	DropoutRate            float64           `json:"dropout_rate"`
	ActivationFunction     string            `json:"activation_function"`
	Optimizer              string            `json:"optimizer"`
	UseBatchNorm           bool              `json:"use_batch_norm"`
	UseResidualConnections bool              `json:"use_residual_connections"`
	MaxSequenceLength      int               `json:"max_sequence_length"`
	EmbeddingDim           int               `json:"embedding_dim"`
	AttentionType          string            `json:"attention_type"`
	Hyperparameters        map[string]string `json:"hyperparameters"`
}

// ModelTrainingResult represents model training results
type ModelTrainingResult struct {
	ModelID           string            `json:"model_id"`
	TrainingLoss      float64           `json:"training_loss"`
	ValidationLoss    float64           `json:"validation_loss"`
	TrainingAccuracy  float64           `json:"training_accuracy"`
	ValidationAccuracy float64          `json:"validation_accuracy"`
	TrainingExamples  int               `json:"training_examples"`
	TrainedAt         time.Time         `json:"trained_at"`
	Metrics           map[string]float64 `json:"metrics"`
	FeatureImportance map[string]float64 `json:"feature_importance"`
}

// RankingModel represents a ranking model
type RankingModel struct {
	Name           string            `json:"name"`
	Algorithm      string            `json:"algorithm"`
	Config         *ModelConfig      `json:"config"`
	Accuracy       float64           `json:"accuracy"`
	Precision      float64           `json:"precision"`
	Recall         float64           `json:"recall"`
	F1Score        float64           `json:"f1_score"`
	TrainingExamples int64           `json:"training_examples"`
	CreatedAt      time.Time         `json:"created_at"`
	LastUpdated    time.Time         `json:"last_updated"`
	FeatureWeights map[string]float64 `json:"feature_weights"`
	Status         string            `json:"status"`
}

// FeatureVector represents a feature vector
type FeatureVector struct {
	DocumentID   string            `json:"document_id"`
	Features     map[string]float64 `json:"features"`
	FeatureNames []string          `json:"feature_names"`
	VectorNorm   float64           `json:"vector_norm"`
}

// UserProfile represents a user profile for personalization
type UserProfile struct {
	UserID              string            `json:"user_id"`
	Interests           []string          `json:"interests"`
	PreferredCategories []string          `json:"preferred_categories"`
	TopicPreferences    map[string]float64 `json:"topic_preferences"`
	ClickedDocuments    []string          `json:"clicked_documents"`
	SearchHistory       []string          `json:"search_history"`
	Location            string            `json:"location"`
	Language            string            `json:"language"`
	LastUpdated         time.Time         `json:"last_updated"`
	Metadata            map[string]string `json:"metadata"`
}

// NewRankingService creates a new ranking service
func NewRankingService(
	pageRankEngine PageRankEngine,
	mlRanker MLRanker,
	featureExtractor FeatureExtractor,
	userProfiler UserProfiler,
	modelManager ModelManager,
) *RankingService {
	return &RankingService{
		pageRankEngine:   pageRankEngine,
		mlRanker:        mlRanker,
		featureExtractor: featureExtractor,
		userProfiler:    userProfiler,
		modelManager:    modelManager,
	}
}

// RankDocuments ranks documents using the specified model
func (s *RankingService) RankDocuments(ctx context.Context, req *pb.RankDocumentsRequest) (*pb.RankDocumentsResponse, error) {
	start := time.Now()

	// Convert protobuf documents to internal documents
	docs := s.convertDocumentsFromProto(req.Documents)

	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Rank documents using ML ranker
	rankedDocs, err := s.mlRanker.RankDocuments(ctx, docs, req.Query, req.UserId, modelName)
	if err != nil {
		return &pb.RankDocumentsResponse{
			Success: false,
			Message: "Failed to rank documents",
		}, status.Error(codes.Internal, "failed to rank documents")
	}

	// Apply personalization if enabled
	if req.EnablePersonalization && req.UserId != "" {
		personalizedDocs, err := s.userProfiler.GetPersonalizedRanking(ctx, req.UserId, docs, req.Query)
		if err == nil {
			// Merge personalization scores
			s.mergePersonalizationScores(rankedDocs, personalizedDocs)
		}
	}

	// Sort by final score
	sort.Slice(rankedDocs, func(i, j int) bool {
		return rankedDocs[i].FinalScore > rankedDocs[j].FinalScore
	})

	// Assign ranks
	for i, doc := range rankedDocs {
		doc.Rank = i + 1
	}

	// Convert to protobuf
	protoDocs := s.convertRankedDocumentsToProto(rankedDocs)

	// Extract feature weights
	featureWeights := make(map[string]float64)
	if model, err := s.mlRanker.GetModel(ctx, modelName); err == nil {
		featureWeights = model.FeatureWeights
	}

	return &pb.RankDocumentsResponse{
		Success:         true,
		Message:         "Documents ranked successfully",
		RankedDocuments: protoDocs,
		ProcessingTimeMs: float64(time.Since(start).Nanoseconds()) / 1e6,
		ModelUsed:       modelName,
		FeatureWeights:  featureWeights,
	}, nil
}

// ReRankDocuments re-ranks already ranked documents
func (s *RankingService) ReRankDocuments(ctx context.Context, req *pb.ReRankDocumentsRequest) (*pb.ReRankDocumentsResponse, error) {
	start := time.Now()

	// Convert protobuf documents to internal documents
	docs := s.convertRankedDocumentsFromProto(req.Documents)

	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Re-rank documents
	rankedDocs, err := s.mlRanker.ReRankDocuments(ctx, docs, req.Query, req.UserId, modelName)
	if err != nil {
		return &pb.ReRankDocumentsResponse{
			Success: false,
			Message: "Failed to re-rank documents",
		}, status.Error(codes.Internal, "failed to re-rank documents")
	}

	// Sort by final score
	sort.Slice(rankedDocs, func(i, j int) bool {
		return rankedDocs[i].FinalScore > rankedDocs[j].FinalScore
	})

	// Assign ranks
	for i, doc := range rankedDocs {
		doc.Rank = i + 1
	}

	// Convert to protobuf
	protoDocs := s.convertRankedDocumentsToProto(rankedDocs)

	return &pb.ReRankDocumentsResponse{
		Success:         true,
		Message:         "Documents re-ranked successfully",
		RankedDocuments: protoDocs,
		ProcessingTimeMs: float64(time.Since(start).Nanoseconds()) / 1e6,
		ModelUsed:       modelName,
	}, nil
}

// TrainRankingModel trains a new ranking model
func (s *RankingService) TrainRankingModel(ctx context.Context, req *pb.TrainRankingModelRequest) (*pb.TrainRankingModelResponse, error) {
	// Convert protobuf examples to internal examples
	examples := s.convertTrainingExamplesFromProto(req.Examples)

	// Convert protobuf config to internal config
	config := s.convertModelConfigFromProto(req.Config)

	// Train model
	result, err := s.mlRanker.TrainModel(ctx, examples, config)
	if err != nil {
		return &pb.TrainRankingModelResponse{
			Success: false,
			Message: "Failed to train model",
		}, status.Error(codes.Internal, "failed to train model")
	}

	return &pb.TrainRankingModelResponse{
		Success:           true,
		Message:           "Model trained successfully",
		ModelId:           result.ModelID,
		TrainingAccuracy:  result.TrainingAccuracy,
		ValidationAccuracy: result.ValidationAccuracy,
		TrainingLoss:      result.TrainingLoss,
		ValidationLoss:    result.ValidationLoss,
		TrainingExamples:  int32(result.TrainingExamples),
		TrainedAt:         timestamppb.New(result.TrainedAt),
		FeatureImportance: result.FeatureImportance,
	}, nil
}

// ComputePageRank computes PageRank for the given links
func (s *RankingService) ComputePageRank(ctx context.Context, req *pb.ComputePageRankRequest) (*pb.ComputePageRankResponse, error) {
	// Convert protobuf links to internal links
	links := s.convertLinksFromProto(req.Links)

	// Convert protobuf config to internal config
	config := &PageRankConfig{
		DampingFactor:       req.DampingFactor,
		ConvergenceThreshold: req.ConvergenceThreshold,
		MaxIterations:       int(req.MaxIterations),
		EnableParallel:      req.EnableParallel,
		Algorithm:           req.Algorithm,
		TopicWeights:        req.TopicWeights,
		UserID:              req.UserId,
	}

	// Compute PageRank
	results, err := s.pageRankEngine.ComputePageRank(ctx, links, config)
	if err != nil {
		return &pb.ComputePageRankResponse{
			Success: false,
			Message: "Failed to compute PageRank",
		}, status.Error(codes.Internal, "failed to compute PageRank")
	}

	// Convert to protobuf
	protoResults := s.convertPageRankResultsToProto(results)

	return &pb.ComputePageRankResponse{
		Success:           true,
		Message:           "PageRank computed successfully",
		Results:           protoResults,
		Iterations:        int32(results.Iterations),
		Converged:         results.Converged,
		ComputationTimeMs: float64(results.ComputationTime.Nanoseconds()) / 1e6,
	}, nil
}

// GetPersonalizedRanking gets personalized ranking for a user
func (s *RankingService) GetPersonalizedRanking(ctx context.Context, req *pb.GetPersonalizedRankingRequest) (*pb.GetPersonalizedRankingResponse, error) {
	// Convert protobuf documents to internal documents
	docs := s.convertDocumentsFromProto(req.Documents)

	// Get user profile
	profile, err := s.userProfiler.GetUserProfile(ctx, req.UserId)
	if err != nil {
		return &pb.GetPersonalizedRankingResponse{
			Success: false,
			Message: "Failed to get user profile",
		}, status.Error(codes.Internal, "failed to get user profile")
	}

	// Get personalized ranking
	rankedDocs, err := s.userProfiler.GetPersonalizedRanking(ctx, req.UserId, docs, req.Query)
	if err != nil {
		return &pb.GetPersonalizedRankingResponse{
			Success: false,
			Message: "Failed to get personalized ranking",
		}, status.Error(codes.Internal, "failed to get personalized ranking")
	}

	// Convert to protobuf
	protoDocs := s.convertRankedDocumentsToProto(rankedDocs)
	protoProfile := s.convertUserProfileToProto(profile)

	return &pb.GetPersonalizedRankingResponse{
		Success:                true,
		Message:                "Personalized ranking retrieved successfully",
		RankedDocuments:        protoDocs,
		UserProfile:            protoProfile,
		PersonalizationStrength: 0.8, // TODO: Calculate based on profile strength
	}, nil
}

// ExtractFeatures extracts features from documents
func (s *RankingService) ExtractFeatures(ctx context.Context, req *pb.ExtractFeaturesRequest) (*pb.ExtractFeaturesResponse, error) {
	// Convert protobuf documents to internal documents
	docs := s.convertDocumentsFromProto(req.Documents)

	// Extract features
	featureVectors, err := s.featureExtractor.ExtractFeatures(ctx, docs, req.Query, req.UserId)
	if err != nil {
		return &pb.ExtractFeaturesResponse{
			Success: false,
			Message: "Failed to extract features",
		}, status.Error(codes.Internal, "failed to extract features")
	}

	// Convert to protobuf
	protoVectors := s.convertFeatureVectorsToProto(featureVectors)

	// Get feature importance
	featureImportance := make(map[string]float64)
	if req.UserId != "" {
		if importance, err := s.featureExtractor.GetFeatureImportance(ctx, "default"); err == nil {
			featureImportance = importance
		}
	}

	return &pb.ExtractFeaturesResponse{
		Success:           true,
		Message:           "Features extracted successfully",
		FeatureVectors:    protoVectors,
		FeatureImportance: featureImportance,
	}, nil
}

// HealthCheck performs health check
func (s *RankingService) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	status := pb.HealthCheckResponse_SERVING
	message := "Service is healthy"
	details := make(map[string]string)

	// Test PageRank engine
	if _, err := s.pageRankEngine.GetPageRank(ctx, "https://example.com"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "PageRank engine is not responding"
		details["page_rank_engine"] = "unhealthy"
	} else {
		details["page_rank_engine"] = "healthy"
	}

	// Test ML ranker
	if _, err := s.mlRanker.GetModel(ctx, "default"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "ML ranker is not responding"
		details["ml_ranker"] = "unhealthy"
	} else {
		details["ml_ranker"] = "healthy"
	}

	// Test feature extractor
	if _, err := s.featureExtractor.GetFeatureImportance(ctx, "default"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Feature extractor is not responding"
		details["feature_extractor"] = "unhealthy"
	} else {
		details["feature_extractor"] = "healthy"
	}

	return &pb.HealthCheckResponse{
		Status:  status,
		Message: message,
		Details: details,
	}, nil
}

// Helper methods

// mergePersonalizationScores merges personalization scores with existing scores
func (s *RankingService) mergePersonalizationScores(rankedDocs []*RankedDocument, personalizedDocs []*RankedDocument) {
	// Create a map of document ID to personalized score
	personalizedScores := make(map[string]float64)
	for _, doc := range personalizedDocs {
		personalizedScores[doc.Document.ID] = doc.PersonalizationScore
	}

	// Merge scores
	for _, doc := range rankedDocs {
		if personalizedScore, exists := personalizedScores[doc.Document.ID]; exists {
			doc.PersonalizationScore = personalizedScore
			// Update final score with personalization
			doc.FinalScore = doc.FinalScore*0.7 + personalizedScore*0.3
		}
	}
}

// convertDocumentsFromProto converts protobuf documents to internal documents
func (s *RankingService) convertDocumentsFromProto(protoDocs []*pb.Document) []*Document {
	docs := make([]*Document, len(protoDocs))
	for i, protoDoc := range protoDocs {
		docs[i] = &Document{
			ID:          protoDoc.Id,
			Title:       protoDoc.Title,
			Content:     protoDoc.Content,
			URL:         protoDoc.Url,
			Score:       protoDoc.InitialScore,
			Features:    protoDoc.Features,
			Metadata:    protoDoc.Metadata,
			CreatedAt:   protoDoc.CreatedAt.AsTime(),
			UpdatedAt:   protoDoc.UpdatedAt.AsTime(),
			Categories:  protoDoc.Categories,
			Language:    protoDoc.Language,
		}
	}
	return docs
}

// convertRankedDocumentsToProto converts internal ranked documents to protobuf
func (s *RankingService) convertRankedDocumentsToProto(rankedDocs []*RankedDocument) []*pb.RankedDocument {
	protoDocs := make([]*pb.RankedDocument, len(rankedDocs))
	for i, doc := range rankedDocs {
		protoDocs[i] = &pb.RankedDocument{
			Document: &pb.Document{
				Id:          doc.Document.ID,
				Title:       doc.Document.Title,
				Content:     doc.Document.Content,
				Url:         doc.Document.URL,
				InitialScore: doc.Document.Score,
				Features:    doc.Document.Features,
				Metadata:    doc.Document.Metadata,
				CreatedAt:   timestamppb.New(doc.Document.CreatedAt),
				UpdatedAt:   timestamppb.New(doc.Document.UpdatedAt),
				Categories:  doc.Document.Categories,
				Language:    doc.Document.Language,
			},
			FinalScore:          doc.FinalScore,
			RelevanceScore:      doc.RelevanceScore,
			AuthorityScore:      doc.AuthorityScore,
			FreshnessScore:      doc.FreshnessScore,
			PersonalizationScore: doc.PersonalizationScore,
			FeatureScores:       doc.FeatureScores,
			Explanation:         doc.Explanation,
			Rank:                int32(doc.Rank),
		}
	}
	return protoDocs
}

// convertRankedDocumentsFromProto converts protobuf ranked documents to internal documents
func (s *RankingService) convertRankedDocumentsFromProto(protoDocs []*pb.RankedDocument) []*RankedDocument {
	rankedDocs := make([]*RankedDocument, len(protoDocs))
	for i, protoDoc := range protoDocs {
		rankedDocs[i] = &RankedDocument{
			Document: &Document{
				ID:          protoDoc.Document.Id,
				Title:       protoDoc.Document.Title,
				Content:     protoDoc.Document.Content,
				URL:         protoDoc.Document.Url,
				Score:       protoDoc.Document.InitialScore,
				Features:    protoDoc.Document.Features,
				Metadata:    protoDoc.Document.Metadata,
				CreatedAt:   protoDoc.Document.CreatedAt.AsTime(),
				UpdatedAt:   protoDoc.Document.UpdatedAt.AsTime(),
				Categories:  protoDoc.Document.Categories,
				Language:    protoDoc.Document.Language,
			},
			FinalScore:          protoDoc.FinalScore,
			RelevanceScore:      protoDoc.RelevanceScore,
			AuthorityScore:      protoDoc.AuthorityScore,
			FreshnessScore:      protoDoc.FreshnessScore,
			PersonalizationScore: protoDoc.PersonalizationScore,
			FeatureScores:       protoDoc.FeatureScores,
			Explanation:         protoDoc.Explanation,
			Rank:                int(protoDoc.Rank),
		}
	}
	return rankedDocs
}

// convertTrainingExamplesFromProto converts protobuf training examples to internal examples
func (s *RankingService) convertTrainingExamplesFromProto(protoExamples []*pb.TrainingExample) []*TrainingExample {
	examples := make([]*TrainingExample, len(protoExamples))
	for i, protoExample := range protoExamples {
		examples[i] = &TrainingExample{
			Features:   protoExample.Features,
			Target:     protoExample.Target,
			Query:      protoExample.Query,
			DocumentID: protoExample.DocumentId,
			UserID:     protoExample.UserId,
			Context:    protoExample.Context,
			Embedding:  protoExample.Embedding,
			Position:   int(protoExample.Position),
			Clicked:    protoExample.Clicked,
			DwellTime:  protoExample.DwellTime,
		}
	}
	return examples
}

// convertModelConfigFromProto converts protobuf model config to internal config
func (s *RankingService) convertModelConfigFromProto(protoConfig *pb.ModelConfig) *ModelConfig {
	return &ModelConfig{
		InputSize:              int(protoConfig.InputSize),
		HiddenSize:             int(protoConfig.HiddenSize),
		OutputSize:             int(protoConfig.OutputSize),
		NumLayers:              int(protoConfig.NumLayers),
		NumHeads:               int(protoConfig.NumHeads),
		NumAttentionLayers:     int(protoConfig.NumAttentionLayers),
		LearningRate:           protoConfig.LearningRate,
		BatchSize:              int(protoConfig.BatchSize),
		Epochs:                 int(protoConfig.Epochs),
		DropoutRate:            protoConfig.DropoutRate,
		ActivationFunction:     protoConfig.ActivationFunction,
		Optimizer:              protoConfig.Optimizer,
		UseBatchNorm:           protoConfig.UseBatchNorm,
		UseResidualConnections: protoConfig.UseResidualConnections,
		MaxSequenceLength:      int(protoConfig.MaxSequenceLength),
		EmbeddingDim:           int(protoConfig.EmbeddingDim),
		AttentionType:          protoConfig.AttentionType,
		Hyperparameters:        protoConfig.Hyperparameters,
	}
}

// convertLinksFromProto converts protobuf links to internal links
func (s *RankingService) convertLinksFromProto(protoLinks []*pb.Link) []*Link {
	links := make([]*Link, len(protoLinks))
	for i, protoLink := range protoLinks {
		links[i] = &Link{
			FromURL:      protoLink.FromUrl,
			ToURL:        protoLink.ToUrl,
			Weight:       protoLink.Weight,
			AnchorText:   protoLink.AnchorText,
			LinkType:     protoLink.LinkType,
			DiscoveredAt: protoLink.DiscoveredAt.AsTime(),
		}
	}
	return links
}

// convertPageRankResultsToProto converts internal PageRank results to protobuf
func (s *RankingService) convertPageRankResultsToProto(results *PageRankResults) []*pb.PageRankResult {
	protoResults := make([]*pb.PageRankResult, len(results.Results))
	for i, result := range results.Results {
		protoResults[i] = &pb.PageRankResult{
			Url:           result.URL,
			PageRank:      result.PageRank,
			InLinks:       int32(result.InLinks),
			OutLinks:      int32(result.OutLinks),
			AuthorityScore: result.AuthorityScore,
			HubScore:      result.HubScore,
		}
	}
	return protoResults
}

// convertUserProfileToProto converts internal user profile to protobuf
func (s *RankingService) convertUserProfileToProto(profile *UserProfile) *pb.UserProfile {
	return &pb.UserProfile{
		UserId:              profile.UserID,
		Interests:           profile.Interests,
		PreferredCategories: profile.PreferredCategories,
		TopicPreferences:    profile.TopicPreferences,
		ClickedDocuments:    profile.ClickedDocuments,
		SearchHistory:       profile.SearchHistory,
		Location:            profile.Location,
		Language:            profile.Language,
		LastUpdated:         timestamppb.New(profile.LastUpdated),
		Metadata:            profile.Metadata,
	}
}

// convertFeatureVectorsToProto converts internal feature vectors to protobuf
func (s *RankingService) convertFeatureVectorsToProto(vectors []*FeatureVector) []*pb.FeatureVector {
	protoVectors := make([]*pb.FeatureVector, len(vectors))
	for i, vector := range vectors {
		protoVectors[i] = &pb.FeatureVector{
			DocumentId:   vector.DocumentID,
			Features:     vector.Features,
			FeatureNames: vector.FeatureNames,
			VectorNorm:   vector.VectorNorm,
		}
	}
	return protoVectors
}

// RegisterRankingService registers the ranking service with gRPC server
func RegisterRankingService(s *grpc.Server, service *RankingService) {
	pb.RegisterRankingServiceServer(s, service)
}