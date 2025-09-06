// T3SS Project
// File: backend_services/ml/ml_services.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package ml

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/t3ss/shared_libs/proto/ml"
)

// MLServices implements the ML services gRPC service
type MLServices struct {
	pb.UnimplementedMLServicesServer
	neuralRanker      NeuralRanker
	embeddingService  EmbeddingService
	queryAnalyzer     QueryAnalyzer
	contentAnalyzer   ContentAnalyzer
	recommendationEngine RecommendationEngine
	anomalyDetector   AnomalyDetector
	modelManager      ModelManager
}

// NeuralRanker interface for neural ranking
type NeuralRanker interface {
	RankDocuments(ctx context.Context, docs []*Document, query string, userID string, modelName string) ([]*RankedDocument, error)
	TrainModel(ctx context.Context, examples []*TrainingExample, config *NeuralModelConfig) (*ModelTrainingResult, error)
	GetModel(ctx context.Context, modelName string) (*NeuralModel, error)
}

// EmbeddingService interface for embedding operations
type EmbeddingService interface {
	GenerateEmbeddings(ctx context.Context, texts []string, modelName string) ([]*Embedding, error)
	SemanticSearch(ctx context.Context, query string, docs []*Document, threshold float64) ([]*RankedDocument, error)
	TrainEmbeddingModel(ctx context.Context, data []*TrainingText, config *EmbeddingModelConfig) (*EmbeddingTrainingResult, error)
}

// QueryAnalyzer interface for query analysis
type QueryAnalyzer interface {
	ClassifyQuery(ctx context.Context, query string, modelName string) (*QueryClassification, error)
	ExtractEntities(ctx context.Context, text string, modelName string) ([]*Entity, error)
	ExpandQuery(ctx context.Context, query string, modelName string) (*QueryExpansion, error)
}

// ContentAnalyzer interface for content analysis
type ContentAnalyzer interface {
	AnalyzeContent(ctx context.Context, content string, contentType string) (*ContentAnalysis, error)
	DetectLanguage(ctx context.Context, text string) (*LanguageDetection, error)
	ExtractKeywords(ctx context.Context, text string, maxKeywords int) ([]*Keyword, error)
}

// RecommendationEngine interface for recommendations
type RecommendationEngine interface {
	GetRecommendations(ctx context.Context, userID string, itemType string, maxResults int) ([]*Recommendation, error)
	TrainModel(ctx context.Context, interactions []*UserInteraction, config *RecommendationModelConfig) (*RecommendationTrainingResult, error)
}

// AnomalyDetector interface for anomaly detection
type AnomalyDetector interface {
	DetectAnomalies(ctx context.Context, values []float64, modelName string) (*AnomalyDetection, error)
	TrainModel(ctx context.Context, data []*TrainingDataPoint, config *AnomalyModelConfig) (*AnomalyTrainingResult, error)
}

// ModelManager interface for model management
type ModelManager interface {
	CreateModel(ctx context.Context, name string, modelType string, config interface{}) error
	DeleteModel(ctx context.Context, name string) error
	ListModels(ctx context.Context) ([]*ModelInfo, error)
	GetModelStatus(ctx context.Context, name string) (string, error)
}

// Document represents a document for ML processing
type Document struct {
	ID          string            `json:"id"`
	Title       string            `json:"title"`
	Content     string            `json:"content"`
	URL         string            `json:"url"`
	Features    map[string]float64 `json:"features"`
	Metadata    map[string]string `json:"metadata"`
	Embedding   []float64         `json:"embedding"`
	Language    string            `json:"language"`
	Categories  []string          `json:"categories"`
}

// RankedDocument represents a ranked document
type RankedDocument struct {
	Document        *Document `json:"document"`
	Score           float64   `json:"score"`
	RelevanceScore  float64   `json:"relevance_score"`
	SemanticScore   float64   `json:"semantic_score"`
	AttentionScore  float64   `json:"attention_score"`
	FeatureScores   map[string]float64 `json:"feature_scores"`
	Explanation     string    `json:"explanation"`
	Rank            int       `json:"rank"`
}

// TrainingExample represents a training example
type TrainingExample struct {
	Features    []float64         `json:"features"`
	Target      float64           `json:"target"`
	Query       string            `json:"query"`
	DocumentID  string            `json:"document_id"`
	UserID      string            `json:"user_id"`
	Context     map[string]string `json:"context"`
	Embedding   []float64         `json:"embedding"`
}

// NeuralModelConfig represents neural model configuration
type NeuralModelConfig struct {
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
}

// NeuralModel represents a neural model
type NeuralModel struct {
	Name           string            `json:"name"`
	Architecture   string            `json:"architecture"`
	Config         *NeuralModelConfig `json:"config"`
	Accuracy       float64           `json:"accuracy"`
	TrainingLoss   float64           `json:"training_loss"`
	ValidationLoss float64           `json:"validation_loss"`
	TrainingExamples int64           `json:"training_examples"`
	CreatedAt      time.Time         `json:"created_at"`
	LastUpdated    time.Time         `json:"last_updated"`
	Status         string            `json:"status"`
}

// Embedding represents an embedding
type Embedding struct {
	Text      string            `json:"text"`
	Values    []float64         `json:"values"`
	Norm      float64           `json:"norm"`
	Language  string            `json:"language"`
	Metadata  map[string]string `json:"metadata"`
}

// TrainingText represents training text for embedding models
type TrainingText struct {
	Text      string            `json:"text"`
	Tokens    []string          `json:"tokens"`
	Language  string            `json:"language"`
	Metadata  map[string]string `json:"metadata"`
}

// EmbeddingModelConfig represents embedding model configuration
type EmbeddingModelConfig struct {
	EmbeddingDim           int               `json:"embedding_dim"`
	WindowSize             int               `json:"window_size"`
	MinCount               int               `json:"min_count"`
	NegativeSamples        int               `json:"negative_samples"`
	LearningRate           float64           `json:"learning_rate"`
	Epochs                 int               `json:"epochs"`
	BatchSize              int               `json:"batch_size"`
	Algorithm              string            `json:"algorithm"`
	MaxVocabSize           int               `json:"max_vocab_size"`
	UseHierarchicalSoftmax bool              `json:"use_hierarchical_softmax"`
	UseNegativeSampling    bool              `json:"use_negative_sampling"`
	Hyperparameters        map[string]string `json:"hyperparameters"`
}

// EmbeddingTrainingResult represents embedding training results
type EmbeddingTrainingResult struct {
	ModelID           string            `json:"model_id"`
	VocabularySize    int               `json:"vocabulary_size"`
	TrainingExamples  int               `json:"training_examples"`
	TrainingLoss      float64           `json:"training_loss"`
	TrainedAt         time.Time         `json:"trained_at"`
	Metrics           map[string]float64 `json:"metrics"`
}

// QueryClassification represents query classification
type QueryClassification struct {
	Intent           string            `json:"intent"`
	Categories       []string          `json:"categories"`
	Entities         []string          `json:"entities"`
	Language         string            `json:"language"`
	ComplexityScore  float64           `json:"complexity_score"`
	IsQuestion       bool              `json:"is_question"`
	IsCommercial     bool              `json:"is_commercial"`
	IsLocal          bool              `json:"is_local"`
	IntentScores     map[string]float64 `json:"intent_scores"`
}

// Entity represents an extracted entity
type Entity struct {
	Text           string            `json:"text"`
	Type           string            `json:"type"`
	Confidence     float64           `json:"confidence"`
	StartPos       int               `json:"start_pos"`
	EndPos         int               `json:"end_pos"`
	NormalizedForm string            `json:"normalized_form"`
	Properties     map[string]string `json:"properties"`
	Embedding      []float64         `json:"embedding"`
}

// QueryExpansion represents query expansion
type QueryExpansion struct {
	OriginalQuery    string            `json:"original_query"`
	Synonyms         []string          `json:"synonyms"`
	RelatedTerms     []string          `json:"related_terms"`
	ExpandedTerms    []string          `json:"expanded_terms"`
	TermWeights      map[string]float64 `json:"term_weights"`
	ExpandedQuery    string            `json:"expanded_query"`
}

// ContentAnalysis represents content analysis results
type ContentAnalysis struct {
	ContentID       string            `json:"content_id"`
	Language        string            `json:"language"`
	SentimentScore  float64           `json:"sentiment_score"`
	Topics          []string          `json:"topics"`
	Keywords        []string          `json:"keywords"`
	ReadabilityScore float64          `json:"readability_score"`
	WordCount       int               `json:"word_count"`
	Entities        []*Entity         `json:"entities"`
	TopicScores     map[string]float64 `json:"topic_scores"`
	KeywordScores   map[string]float64 `json:"keyword_scores"`
	Summary         string            `json:"summary"`
}

// LanguageDetection represents language detection results
type LanguageDetection struct {
	Language        string            `json:"language"`
	Confidence      float64           `json:"confidence"`
	LanguageScores  map[string]float64 `json:"language_scores"`
}

// Keyword represents an extracted keyword
type Keyword struct {
	Text     string  `json:"text"`
	Score    float64 `json:"score"`
	Position int     `json:"position"`
	Length   int     `json:"length"`
	Stem     string  `json:"stem"`
}

// Recommendation represents a recommendation
type Recommendation struct {
	ItemID    string            `json:"item_id"`
	Score     float64           `json:"score"`
	Reason    string            `json:"reason"`
	Metadata  map[string]string `json:"metadata"`
	ItemType  string            `json:"item_type"`
}

// UserInteraction represents a user interaction
type UserInteraction struct {
	UserID         string            `json:"user_id"`
	ItemID         string            `json:"item_id"`
	InteractionType string           `json:"interaction_type"`
	Rating         float64           `json:"rating"`
	Timestamp      time.Time         `json:"timestamp"`
	Context        map[string]string `json:"context"`
}

// RecommendationModelConfig represents recommendation model configuration
type RecommendationModelConfig struct {
	EmbeddingDim        int               `json:"embedding_dim"`
	HiddenLayers        int               `json:"hidden_layers"`
	LearningRate        float64           `json:"learning_rate"`
	Epochs              int               `json:"epochs"`
	BatchSize           int               `json:"batch_size"`
	Regularization      float64           `json:"regularization"`
	UseNegativeSampling bool              `json:"use_negative_sampling"`
	NegativeSamples     int               `json:"negative_samples"`
	Hyperparameters     map[string]string `json:"hyperparameters"`
}

// RecommendationTrainingResult represents recommendation training results
type RecommendationTrainingResult struct {
	ModelID           string            `json:"model_id"`
	TrainingLoss      float64           `json:"training_loss"`
	ValidationLoss    float64           `json:"validation_loss"`
	HitRate           float64           `json:"hit_rate"`
	NDCG              float64           `json:"ndcg"`
	TrainingExamples  int               `json:"training_examples"`
	TrainedAt         time.Time         `json:"trained_at"`
}

// AnomalyDetection represents anomaly detection results
type AnomalyDetection struct {
	Anomalies    []*Anomaly         `json:"anomalies"`
	AnomalyScore float64            `json:"anomaly_score"`
	IsAnomaly    bool               `json:"is_anomaly"`
	ModelUsed    string             `json:"model_used"`
}

// Anomaly represents an anomaly
type Anomaly struct {
	Index       int               `json:"index"`
	Score       float64           `json:"score"`
	Type        string            `json:"type"`
	Description string            `json:"description"`
	Context     map[string]string `json:"context"`
}

// TrainingDataPoint represents a training data point for anomaly detection
type TrainingDataPoint struct {
	Features  []float64         `json:"features"`
	IsAnomaly bool              `json:"is_anomaly"`
	Metadata  map[string]string `json:"metadata"`
	Timestamp time.Time         `json:"timestamp"`
}

// AnomalyModelConfig represents anomaly model configuration
type AnomalyModelConfig struct {
	InputDim           int               `json:"input_dim"`
	HiddenDim          int               `json:"hidden_dim"`
	Contamination      float64           `json:"contamination"`
	NEstimators        int               `json:"n_estimators"`
	LearningRate       float64           `json:"learning_rate"`
	Epochs             int               `json:"epochs"`
	BatchSize          int               `json:"batch_size"`
	ActivationFunction string            `json:"activation_function"`
	Hyperparameters    map[string]string `json:"hyperparameters"`
}

// AnomalyTrainingResult represents anomaly training results
type AnomalyTrainingResult struct {
	ModelID           string            `json:"model_id"`
	TrainingAccuracy  float64           `json:"training_accuracy"`
	ValidationAccuracy float64          `json:"validation_accuracy"`
	Precision         float64           `json:"precision"`
	Recall            float64           `json:"recall"`
	F1Score           float64           `json:"f1_score"`
	TrainingExamples  int               `json:"training_examples"`
	TrainedAt         time.Time         `json:"trained_at"`
}

// ModelInfo represents model information
type ModelInfo struct {
	Name           string    `json:"name"`
	Type           string    `json:"type"`
	Status         string    `json:"status"`
	CreatedAt      time.Time `json:"created_at"`
	LastUpdated    time.Time `json:"last_updated"`
	Accuracy       float64   `json:"accuracy"`
	TrainingExamples int64   `json:"training_examples"`
}

// NewMLServices creates a new ML services instance
func NewMLServices(
	neuralRanker NeuralRanker,
	embeddingService EmbeddingService,
	queryAnalyzer QueryAnalyzer,
	contentAnalyzer ContentAnalyzer,
	recommendationEngine RecommendationEngine,
	anomalyDetector AnomalyDetector,
	modelManager ModelManager,
) *MLServices {
	return &MLServices{
		neuralRanker:      neuralRanker,
		embeddingService:  embeddingService,
		queryAnalyzer:     queryAnalyzer,
		contentAnalyzer:   contentAnalyzer,
		recommendationEngine: recommendationEngine,
		anomalyDetector:   anomalyDetector,
		modelManager:      modelManager,
	}
}

// NeuralRank performs neural ranking
func (s *MLServices) NeuralRank(ctx context.Context, req *pb.NeuralRankRequest) (*pb.NeuralRankResponse, error) {
	start := time.Now()

	// Convert protobuf documents to internal documents
	docs := s.convertDocumentsFromProto(req.Documents)

	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Rank documents using neural ranker
	rankedDocs, err := s.neuralRanker.RankDocuments(ctx, docs, req.Query, req.UserId, modelName)
	if err != nil {
		return &pb.NeuralRankResponse{
			Success: false,
			Message: "Failed to rank documents",
		}, status.Error(codes.Internal, "failed to rank documents")
	}

	// Convert to protobuf
	protoDocs := s.convertRankedDocumentsToProto(rankedDocs)

	// Calculate attention weights (simplified)
	attentionWeights := make(map[string]float64)
	for i, doc := range rankedDocs {
		attentionWeights[doc.Document.ID] = doc.AttentionScore
	}

	return &pb.NeuralRankResponse{
		Success:         true,
		Message:         "Documents ranked successfully",
		RankedDocuments: protoDocs,
		ProcessingTimeMs: float64(time.Since(start).Nanoseconds()) / 1e6,
		ModelUsed:       modelName,
		AttentionWeights: attentionWeights,
		ConfidenceScore: 0.85, // TODO: Calculate actual confidence
	}, nil
}

// TrainNeuralModel trains a neural model
func (s *MLServices) TrainNeuralModel(ctx context.Context, req *pb.TrainNeuralModelRequest) (*pb.TrainNeuralModelResponse, error) {
	// Convert protobuf examples to internal examples
	examples := s.convertTrainingExamplesFromProto(req.Examples)

	// Convert protobuf config to internal config
	config := s.convertNeuralModelConfigFromProto(req.Config)

	// Train model
	result, err := s.neuralRanker.TrainModel(ctx, examples, config)
	if err != nil {
		return &pb.TrainNeuralModelResponse{
			Success: false,
			Message: "Failed to train model",
		}, status.Error(codes.Internal, "failed to train model")
	}

	return &pb.TrainNeuralModelResponse{
		Success:           true,
		Message:           "Model trained successfully",
		ModelId:           result.ModelID,
		TrainingLoss:      result.TrainingLoss,
		ValidationLoss:    result.ValidationLoss,
		TrainingAccuracy:  result.TrainingAccuracy,
		ValidationAccuracy: result.ValidationAccuracy,
		TrainingExamples:  int32(result.TrainingExamples),
		TrainedAt:         timestamppb.New(result.TrainedAt),
		Metrics:           result.Metrics,
	}, nil
}

// SemanticSearch performs semantic search
func (s *MLServices) SemanticSearch(ctx context.Context, req *pb.SemanticSearchRequest) (*pb.SemanticSearchResponse, error) {
	start := time.Now()

	// Convert protobuf documents to internal documents
	docs := s.convertDocumentsFromProto(req.Documents)

	// Set default model name
	modelName := req.EmbeddingModel
	if modelName == "" {
		modelName = "default"
	}

	// Perform semantic search
	rankedDocs, err := s.embeddingService.SemanticSearch(ctx, req.Query, docs, req.SimilarityThreshold)
	if err != nil {
		return &pb.SemanticSearchResponse{
			Success: false,
			Message: "Failed to perform semantic search",
		}, status.Error(codes.Internal, "failed to perform semantic search")
	}

	// Convert to protobuf
	protoDocs := s.convertRankedDocumentsToProto(rankedDocs)

	// Generate query embedding
	queryEmbedding := []float64{} // TODO: Generate actual query embedding

	// Calculate similarity scores
	similarityScores := make(map[string]float64)
	for _, doc := range rankedDocs {
		similarityScores[doc.Document.ID] = doc.SemanticScore
	}

	return &pb.SemanticSearchResponse{
		Success:           true,
		Message:           "Semantic search completed successfully",
		Results:           protoDocs,
		QueryEmbedding:    queryEmbedding,
		ProcessingTimeMs:  float64(time.Since(start).Nanoseconds()) / 1e6,
		ModelUsed:         modelName,
		SimilarityScores:  similarityScores,
	}, nil
}

// GenerateEmbeddings generates embeddings for texts
func (s *MLServices) GenerateEmbeddings(ctx context.Context, req *pb.GenerateEmbeddingsRequest) (*pb.GenerateEmbeddingsResponse, error) {
	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Generate embeddings
	embeddings, err := s.embeddingService.GenerateEmbeddings(ctx, req.Texts, modelName)
	if err != nil {
		return &pb.GenerateEmbeddingsResponse{
			Success: false,
			Message: "Failed to generate embeddings",
		}, status.Error(codes.Internal, "failed to generate embeddings")
	}

	// Convert to protobuf
	protoEmbeddings := s.convertEmbeddingsToProto(embeddings)

	return &pb.GenerateEmbeddingsResponse{
		Success:        true,
		Message:        "Embeddings generated successfully",
		Embeddings:     protoEmbeddings,
		ModelUsed:      modelName,
		EmbeddingDim:   int32(len(embeddings[0].Values)),
	}, nil
}

// ClassifyQuery classifies a query
func (s *MLServices) ClassifyQuery(ctx context.Context, req *pb.ClassifyQueryRequest) (*pb.ClassifyQueryResponse, error) {
	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Classify query
	classification, err := s.queryAnalyzer.ClassifyQuery(ctx, req.Query, modelName)
	if err != nil {
		return &pb.ClassifyQueryResponse{
			Success: false,
			Message: "Failed to classify query",
		}, status.Error(codes.Internal, "failed to classify query")
	}

	// Convert to protobuf
	protoClassification := s.convertQueryClassificationToProto(classification)

	return &pb.ClassifyQueryResponse{
		Success:     true,
		Message:     "Query classified successfully",
		Classification: protoClassification,
		Confidence:  0.85, // TODO: Calculate actual confidence
		Explanation: "Classification based on query patterns and keywords",
		ModelUsed:   modelName,
	}, nil
}

// ExtractEntities extracts entities from text
func (s *MLServices) ExtractEntities(ctx context.Context, req *pb.ExtractEntitiesRequest) (*pb.ExtractEntitiesResponse, error) {
	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Extract entities
	entities, err := s.queryAnalyzer.ExtractEntities(ctx, req.Text, modelName)
	if err != nil {
		return &pb.ExtractEntitiesResponse{
			Success: false,
			Message: "Failed to extract entities",
		}, status.Error(codes.Internal, "failed to extract entities")
	}

	// Convert to protobuf
	protoEntities := s.convertEntitiesToProto(entities)

	return &pb.ExtractEntitiesResponse{
		Success:   true,
		Message:   "Entities extracted successfully",
		Entities:  protoEntities,
		ModelUsed: modelName,
	}, nil
}

// AnalyzeContent analyzes content
func (s *MLServices) AnalyzeContent(ctx context.Context, req *pb.ContentAnalysisRequest) (*pb.ContentAnalysisResponse, error) {
	start := time.Now()

	// Analyze content
	analysis, err := s.contentAnalyzer.AnalyzeContent(ctx, req.Content, req.ContentType)
	if err != nil {
		return &pb.ContentAnalysisResponse{
			Success: false,
			Message: "Failed to analyze content",
		}, status.Error(codes.Internal, "failed to analyze content")
	}

	// Convert to protobuf
	protoAnalysis := s.convertContentAnalysisToProto(analysis)

	return &pb.ContentAnalysisResponse{
		Success:           true,
		Message:           "Content analyzed successfully",
		Analysis:          protoAnalysis,
		ProcessingTimeMs:  float64(time.Since(start).Nanoseconds()) / 1e6,
	}, nil
}

// GetRecommendations gets recommendations for a user
func (s *MLServices) GetRecommendations(ctx context.Context, req *pb.RecommendationRequest) (*pb.RecommendationResponse, error) {
	// Set default max results
	maxResults := int(req.MaxRecommendations)
	if maxResults <= 0 {
		maxResults = 10
	}

	// Get recommendations
	recommendations, err := s.recommendationEngine.GetRecommendations(ctx, req.UserId, req.ItemType, maxResults)
	if err != nil {
		return &pb.RecommendationResponse{
			Success: false,
			Message: "Failed to get recommendations",
		}, status.Error(codes.Internal, "failed to get recommendations")
	}

	// Convert to protobuf
	protoRecommendations := s.convertRecommendationsToProto(recommendations)

	return &pb.RecommendationResponse{
		Success:      true,
		Message:      "Recommendations retrieved successfully",
		Recommendations: protoRecommendations,
		AlgorithmUsed: "collaborative_filtering", // TODO: Get actual algorithm
		Confidence:   0.8, // TODO: Calculate actual confidence
	}, nil
}

// DetectAnomalies detects anomalies in data
func (s *MLServices) DetectAnomalies(ctx context.Context, req *pb.AnomalyDetectionRequest) (*pb.AnomalyDetectionResponse, error) {
	// Set default model name
	modelName := req.ModelName
	if modelName == "" {
		modelName = "default"
	}

	// Detect anomalies
	detection, err := s.anomalyDetector.DetectAnomalies(ctx, req.Values, modelName)
	if err != nil {
		return &pb.AnomalyDetectionResponse{
			Success: false,
			Message: "Failed to detect anomalies",
		}, status.Error(codes.Internal, "failed to detect anomalies")
	}

	// Convert to protobuf
	protoAnomalies := s.convertAnomaliesToProto(detection.Anomalies)

	return &pb.AnomalyDetectionResponse{
		Success:      true,
		Message:      "Anomaly detection completed successfully",
		Anomalies:    protoAnomalies,
		AnomalyScore: detection.AnomalyScore,
		IsAnomaly:    detection.IsAnomaly,
		ModelUsed:    detection.ModelUsed,
	}, nil
}

// HealthCheck performs health check
func (s *MLServices) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	status := pb.HealthCheckResponse_SERVING
	message := "Service is healthy"
	details := make(map[string]string)

	// Test neural ranker
	if _, err := s.neuralRanker.GetModel(ctx, "default"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Neural ranker is not responding"
		details["neural_ranker"] = "unhealthy"
	} else {
		details["neural_ranker"] = "healthy"
	}

	// Test embedding service
	if _, err := s.embeddingService.GenerateEmbeddings(ctx, []string{"test"}, "default"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Embedding service is not responding"
		details["embedding_service"] = "unhealthy"
	} else {
		details["embedding_service"] = "healthy"
	}

	// Test query analyzer
	if _, err := s.queryAnalyzer.ClassifyQuery(ctx, "test", "default"); err != nil {
		status = pb.HealthCheckResponse_NOT_SERVING
		message = "Query analyzer is not responding"
		details["query_analyzer"] = "unhealthy"
	} else {
		details["query_analyzer"] = "healthy"
	}

	return &pb.HealthCheckResponse{
		Status:  status,
		Message: message,
		Details: details,
	}, nil
}

// Helper methods

// convertDocumentsFromProto converts protobuf documents to internal documents
func (s *MLServices) convertDocumentsFromProto(protoDocs []*pb.Document) []*Document {
	docs := make([]*Document, len(protoDocs))
	for i, protoDoc := range protoDocs {
		docs[i] = &Document{
			ID:          protoDoc.Id,
			Title:       protoDoc.Title,
			Content:     protoDoc.Content,
			URL:         protoDoc.Url,
			Features:    protoDoc.Features,
			Metadata:    protoDoc.Metadata,
			Embedding:   protoDoc.Embedding,
			Language:    protoDoc.Language,
			Categories:  protoDoc.Categories,
		}
	}
	return docs
}

// convertRankedDocumentsToProto converts internal ranked documents to protobuf
func (s *MLServices) convertRankedDocumentsToProto(rankedDocs []*RankedDocument) []*pb.RankedDocument {
	protoDocs := make([]*pb.RankedDocument, len(rankedDocs))
	for i, doc := range rankedDocs {
		protoDocs[i] = &pb.RankedDocument{
			Document: &pb.Document{
				Id:          doc.Document.ID,
				Title:       doc.Document.Title,
				Content:     doc.Document.Content,
				Url:         doc.Document.URL,
				Features:    doc.Document.Features,
				Metadata:    doc.Document.Metadata,
				Embedding:   doc.Document.Embedding,
				Language:    doc.Document.Language,
				Categories:  doc.Document.Categories,
			},
			Score:           doc.Score,
			RelevanceScore:  doc.RelevanceScore,
			SemanticScore:   doc.SemanticScore,
			AttentionScore:  doc.AttentionScore,
			FeatureScores:   doc.FeatureScores,
			Explanation:     doc.Explanation,
			Rank:            int32(doc.Rank),
		}
	}
	return protoDocs
}

// convertTrainingExamplesFromProto converts protobuf training examples to internal examples
func (s *MLServices) convertTrainingExamplesFromProto(protoExamples []*pb.TrainingExample) []*TrainingExample {
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
		}
	}
	return examples
}

// convertNeuralModelConfigFromProto converts protobuf neural model config to internal config
func (s *MLServices) convertNeuralModelConfigFromProto(protoConfig *pb.NeuralModelConfig) *NeuralModelConfig {
	return &NeuralModelConfig{
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

// convertEmbeddingsToProto converts internal embeddings to protobuf
func (s *MLServices) convertEmbeddingsToProto(embeddings []*Embedding) []*pb.Embedding {
	protoEmbeddings := make([]*pb.Embedding, len(embeddings))
	for i, embedding := range embeddings {
		protoEmbeddings[i] = &pb.Embedding{
			Text:     embedding.Text,
			Values:   embedding.Values,
			Norm:     embedding.Norm,
			Language: embedding.Language,
			Metadata: embedding.Metadata,
		}
	}
	return protoEmbeddings
}

// convertQueryClassificationToProto converts internal query classification to protobuf
func (s *MLServices) convertQueryClassificationToProto(classification *QueryClassification) *pb.QueryClassification {
	return &pb.QueryClassification{
		Intent:          classification.Intent,
		Categories:      classification.Categories,
		Entities:        classification.Entities,
		Language:        classification.Language,
		ComplexityScore: classification.ComplexityScore,
		IsQuestion:      classification.IsQuestion,
		IsCommercial:    classification.IsCommercial,
		IsLocal:         classification.IsLocal,
		IntentScores:    classification.IntentScores,
	}
}

// convertEntitiesToProto converts internal entities to protobuf
func (s *MLServices) convertEntitiesToProto(entities []*Entity) []*pb.Entity {
	protoEntities := make([]*pb.Entity, len(entities))
	for i, entity := range entities {
		protoEntities[i] = &pb.Entity{
			Text:           entity.Text,
			Type:           entity.Type,
			Confidence:     entity.Confidence,
			StartPos:       int32(entity.StartPos),
			EndPos:         int32(entity.EndPos),
			NormalizedForm: entity.NormalizedForm,
			Properties:     entity.Properties,
			Embedding:      entity.Embedding,
		}
	}
	return protoEntities
}

// convertContentAnalysisToProto converts internal content analysis to protobuf
func (s *MLServices) convertContentAnalysisToProto(analysis *ContentAnalysis) *pb.ContentAnalysis {
	protoEntities := s.convertEntitiesToProto(analysis.Entities)
	
	return &pb.ContentAnalysis{
		ContentId:       analysis.ContentID,
		Language:        analysis.Language,
		SentimentScore:  analysis.SentimentScore,
		Topics:          analysis.Topics,
		Keywords:        analysis.Keywords,
		ReadabilityScore: analysis.ReadabilityScore,
		WordCount:       int32(analysis.WordCount),
		Entities:        protoEntities,
		TopicScores:     analysis.TopicScores,
		KeywordScores:   analysis.KeywordScores,
		Summary:         analysis.Summary,
	}
}

// convertRecommendationsToProto converts internal recommendations to protobuf
func (s *MLServices) convertRecommendationsToProto(recommendations []*Recommendation) []*pb.Recommendation {
	protoRecommendations := make([]*pb.Recommendation, len(recommendations))
	for i, rec := range recommendations {
		protoRecommendations[i] = &pb.Recommendation{
			ItemId:    rec.ItemID,
			Score:     rec.Score,
			Reason:    rec.Reason,
			Metadata:  rec.Metadata,
			ItemType:  rec.ItemType,
		}
	}
	return protoRecommendations
}

// convertAnomaliesToProto converts internal anomalies to protobuf
func (s *MLServices) convertAnomaliesToProto(anomalies []*Anomaly) []*pb.Anomaly {
	protoAnomalies := make([]*pb.Anomaly, len(anomalies))
	for i, anomaly := range anomalies {
		protoAnomalies[i] = &pb.Anomaly{
			Index:       int32(anomaly.Index),
			Score:       anomaly.Score,
			Type:        anomaly.Type,
			Description: anomaly.Description,
			Context:     anomaly.Context,
		}
	}
	return protoAnomalies
}

// RegisterMLServices registers the ML services with gRPC server
func RegisterMLServices(s *grpc.Server, service *MLServices) {
	pb.RegisterMLServicesServer(s, service)
}