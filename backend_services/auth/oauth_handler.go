// T3SS Project
// File: backend_services/auth/oauth_handler.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/github"
	"golang.org/x/oauth2/microsoft"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/t3ss/shared_libs/proto/auth"
)

// OAuthProvider represents different OAuth providers
type OAuthProvider string

const (
	GoogleProvider    OAuthProvider = "google"
	GitHubProvider    OAuthProvider = "github"
	MicrosoftProvider OAuthProvider = "microsoft"
)

// OAuthConfig holds OAuth configuration
type OAuthConfig struct {
	ClientID     string
	ClientSecret string
	RedirectURL  string
	Scopes       []string
	Provider     OAuthProvider
}

// OAuthHandler handles OAuth authentication
type OAuthHandler struct {
	pb.UnimplementedAuthServiceServer
	configs     map[OAuthProvider]*OAuthConfig
	jwtSecret   []byte
	userStore   UserStore
	sessionStore SessionStore
	stateStore  StateStore
}

// UserStore interface for user data operations
type UserStore interface {
	GetUserByID(ctx context.Context, userID string) (*User, error)
	GetUserByEmail(ctx context.Context, email string) (*User, error)
	CreateUser(ctx context.Context, user *User) error
	UpdateUser(ctx context.Context, user *User) error
	DeleteUser(ctx context.Context, userID string) error
}

// SessionStore interface for session management
type SessionStore interface {
	CreateSession(ctx context.Context, session *Session) error
	GetSession(ctx context.Context, sessionID string) (*Session, error)
	DeleteSession(ctx context.Context, sessionID string) error
	DeleteUserSessions(ctx context.Context, userID string) error
}

// StateStore interface for OAuth state management
type StateStore interface {
	StoreState(ctx context.Context, state string, data *StateData) error
	GetState(ctx context.Context, state string) (*StateData, error)
	DeleteState(ctx context.Context, state string) error
}

// User represents a user in the system
type User struct {
	ID            string    `json:"id"`
	Username      string    `json:"username"`
	Email         string    `json:"email"`
	FirstName     string    `json:"first_name"`
	LastName      string    `json:"last_name"`
	AvatarURL     string    `json:"avatar_url"`
	EmailVerified bool      `json:"email_verified"`
	IsActive      bool      `json:"is_active"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
	LastLogin     time.Time `json:"last_login"`
	Roles         []string  `json:"roles"`
	Metadata      map[string]string `json:"metadata"`
	Provider      string    `json:"provider"`
	ProviderID    string    `json:"provider_id"`
}

// Session represents a user session
type Session struct {
	ID        string            `json:"id"`
	UserID    string            `json:"user_id"`
	CreatedAt time.Time         `json:"created_at"`
	ExpiresAt time.Time         `json:"expires_at"`
	Metadata  map[string]string `json:"metadata"`
	IsActive  bool              `json:"is_active"`
}

// StateData represents OAuth state data
type StateData struct {
	Provider    string            `json:"provider"`
	RedirectURL string            `json:"redirect_url"`
	UserID      string            `json:"user_id"`
	CreatedAt   time.Time         `json:"created_at"`
	Metadata    map[string]string `json:"metadata"`
}

// JWTClaims represents JWT token claims
type JWTClaims struct {
	UserID    string   `json:"user_id"`
	Username  string   `json:"username"`
	Email     string   `json:"email"`
	Roles     []string `json:"roles"`
	SessionID string   `json:"session_id"`
	jwt.RegisteredClaims
}

// NewOAuthHandler creates a new OAuth handler
func NewOAuthHandler(jwtSecret []byte, userStore UserStore, sessionStore SessionStore, stateStore StateStore) *OAuthHandler {
	configs := make(map[OAuthProvider]*OAuthConfig)
	
	// Initialize OAuth configurations
	configs[GoogleProvider] = &OAuthConfig{
		ClientID:     getEnv("GOOGLE_CLIENT_ID", ""),
		ClientSecret: getEnv("GOOGLE_CLIENT_SECRET", ""),
		RedirectURL:  getEnv("GOOGLE_REDIRECT_URL", ""),
		Scopes:       []string{"openid", "profile", "email"},
		Provider:     GoogleProvider,
	}
	
	configs[GitHubProvider] = &OAuthConfig{
		ClientID:     getEnv("GITHUB_CLIENT_ID", ""),
		ClientSecret: getEnv("GITHUB_CLIENT_SECRET", ""),
		RedirectURL:  getEnv("GITHUB_REDIRECT_URL", ""),
		Scopes:       []string{"user:email"},
		Provider:     GitHubProvider,
	}
	
	configs[MicrosoftProvider] = &OAuthConfig{
		ClientID:     getEnv("MICROSOFT_CLIENT_ID", ""),
		ClientSecret: getEnv("MICROSOFT_CLIENT_SECRET", ""),
		RedirectURL:  getEnv("MICROSOFT_REDIRECT_URL", ""),
		Scopes:       []string{"openid", "profile", "email"},
		Provider:     MicrosoftProvider,
	}

	return &OAuthHandler{
		configs:     configs,
		jwtSecret:   jwtSecret,
		userStore:   userStore,
		sessionStore: sessionStore,
		stateStore:  stateStore,
	}
}

// GetOAuthURL generates OAuth URL for the specified provider
func (h *OAuthHandler) GetOAuthURL(ctx context.Context, req *pb.GetOAuthURLRequest) (*pb.GetOAuthURLResponse, error) {
	provider := OAuthProvider(req.Provider)
	config, exists := h.configs[provider]
	if !exists {
		return &pb.GetOAuthURLResponse{
			Error: "Unsupported OAuth provider",
		}, status.Error(codes.InvalidArgument, "unsupported OAuth provider")
	}

	// Generate state parameter
	state, err := h.generateState()
	if err != nil {
		return &pb.GetOAuthURLResponse{
			Error: "Failed to generate state",
		}, status.Error(codes.Internal, "failed to generate state")
	}

	// Store state data
	stateData := &StateData{
		Provider:    req.Provider,
		RedirectURL: req.RedirectUri,
		UserID:      req.ClientId,
		CreatedAt:   time.Now(),
		Metadata:    map[string]string{
			"scopes": strings.Join(req.Scopes, ","),
		},
	}

	if err := h.stateStore.StoreState(ctx, state, stateData); err != nil {
		return &pb.GetOAuthURLResponse{
			Error: "Failed to store state",
		}, status.Error(codes.Internal, "failed to store state")
	}

	// Create OAuth2 config
	oauthConfig := h.createOAuth2Config(config, req.Scopes)
	
	// Generate OAuth URL
	authURL := oauthConfig.AuthCodeURL(state, oauth2.AccessTypeOffline)

	return &pb.GetOAuthURLResponse{
		Url:   authURL,
		State: state,
	}, nil
}

// HandleOAuthCallback processes OAuth callback
func (h *OAuthHandler) HandleOAuthCallback(ctx context.Context, req *pb.HandleOAuthCallbackRequest) (*pb.HandleOAuthCallbackResponse, error) {
	// Validate state parameter
	stateData, err := h.stateStore.GetState(ctx, req.State)
	if err != nil {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "Invalid state parameter",
		}, status.Error(codes.InvalidArgument, "invalid state parameter")
	}

	// Clean up state
	defer h.stateStore.DeleteState(ctx, req.State)

	// Verify state matches request
	if stateData.Provider != req.Provider {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "State mismatch",
		}, status.Error(codes.InvalidArgument, "state mismatch")
	}

	// Get OAuth config
	provider := OAuthProvider(req.Provider)
	config, exists := h.configs[provider]
	if !exists {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "Unsupported OAuth provider",
		}, status.Error(codes.InvalidArgument, "unsupported OAuth provider")
	}

	// Create OAuth2 config
	oauthConfig := h.createOAuth2Config(config, []string{})

	// Exchange code for token
	token, err := oauthConfig.Exchange(ctx, req.Code)
	if err != nil {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "Failed to exchange code for token",
		}, status.Error(codes.Internal, "failed to exchange code for token")
	}

	// Get user info from provider
	userInfo, err := h.getUserInfoFromProvider(ctx, provider, token)
	if err != nil {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "Failed to get user info",
		}, status.Error(codes.Internal, "failed to get user info")
	}

	// Check if user exists
	user, err := h.userStore.GetUserByEmail(ctx, userInfo.Email)
	isNewUser := false
	if err != nil {
		// User doesn't exist, create new user
		user = &User{
			ID:            h.generateUserID(),
			Username:      userInfo.Username,
			Email:         userInfo.Email,
			FirstName:     userInfo.FirstName,
			LastName:      userInfo.LastName,
			AvatarURL:     userInfo.AvatarURL,
			EmailVerified: userInfo.EmailVerified,
			IsActive:      true,
			CreatedAt:     time.Now(),
			UpdatedAt:     time.Now(),
			LastLogin:     time.Now(),
			Roles:         []string{"user"},
			Metadata:      make(map[string]string),
			Provider:      req.Provider,
			ProviderID:    userInfo.ID,
		}

		if err := h.userStore.CreateUser(ctx, user); err != nil {
			return &pb.HandleOAuthCallbackResponse{
				Success: false,
				Message: "Failed to create user",
			}, status.Error(codes.Internal, "failed to create user")
		}
		isNewUser = true
	} else {
		// Update existing user
		user.LastLogin = time.Now()
		user.AvatarURL = userInfo.AvatarURL
		user.UpdatedAt = time.Now()
		
		if err := h.userStore.UpdateUser(ctx, user); err != nil {
			return &pb.HandleOAuthCallbackResponse{
				Success: false,
				Message: "Failed to update user",
			}, status.Error(codes.Internal, "failed to update user")
		}
	}

	// Create session
	sessionID := h.generateSessionID()
	session := &Session{
		ID:        sessionID,
		UserID:    user.ID,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(24 * time.Hour),
		Metadata:  make(map[string]string),
		IsActive:  true,
	}

	if err := h.sessionStore.CreateSession(ctx, session); err != nil {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "Failed to create session",
		}, status.Error(codes.Internal, "failed to create session")
	}

	// Generate JWT tokens
	accessToken, refreshToken, err := h.generateTokens(user, sessionID)
	if err != nil {
		return &pb.HandleOAuthCallbackResponse{
			Success: false,
			Message: "Failed to generate tokens",
		}, status.Error(codes.Internal, "failed to generate tokens")
	}

	return &pb.HandleOAuthCallbackResponse{
		Success:      true,
		Message:      "Authentication successful",
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    3600, // 1 hour
		User:         h.convertUserToProto(user),
		IsNewUser:    isNewUser,
	}, nil
}

// createOAuth2Config creates OAuth2 configuration for the provider
func (h *OAuthHandler) createOAuth2Config(config *OAuthConfig, scopes []string) *oauth2.Config {
	baseConfig := &oauth2.Config{
		ClientID:     config.ClientID,
		ClientSecret: config.ClientSecret,
		RedirectURL:  config.RedirectURL,
		Scopes:       scopes,
	}

	switch config.Provider {
	case GoogleProvider:
		baseConfig.Endpoint = google.Endpoint
	case GitHubProvider:
		baseConfig.Endpoint = github.Endpoint
	case MicrosoftProvider:
		baseConfig.Endpoint = microsoft.AzureADEndpoint("common")
	}

	return baseConfig
}

// getUserInfoFromProvider fetches user information from OAuth provider
func (h *OAuthHandler) getUserInfoFromProvider(ctx context.Context, provider OAuthProvider, token *oauth2.Token) (*UserInfo, error) {
	client := oauth2.NewClient(ctx, oauth2.StaticTokenSource(token))

	switch provider {
	case GoogleProvider:
		return h.getGoogleUserInfo(ctx, client)
	case GitHubProvider:
		return h.getGitHubUserInfo(ctx, client)
	case MicrosoftProvider:
		return h.getMicrosoftUserInfo(ctx, client)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// UserInfo represents user information from OAuth provider
type UserInfo struct {
	ID            string
	Username      string
	Email         string
	FirstName     string
	LastName      string
	AvatarURL     string
	EmailVerified bool
}

// getGoogleUserInfo fetches user info from Google
func (h *OAuthHandler) getGoogleUserInfo(ctx context.Context, client *http.Client) (*UserInfo, error) {
	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var userInfo struct {
		ID            string `json:"id"`
		Email         string `json:"email"`
		Name          string `json:"name"`
		GivenName     string `json:"given_name"`
		FamilyName    string `json:"family_name"`
		Picture       string `json:"picture"`
		VerifiedEmail bool   `json:"verified_email"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, err
	}

	return &UserInfo{
		ID:            userInfo.ID,
		Username:      userInfo.Email,
		Email:         userInfo.Email,
		FirstName:     userInfo.GivenName,
		LastName:      userInfo.FamilyName,
		AvatarURL:     userInfo.Picture,
		EmailVerified: userInfo.VerifiedEmail,
	}, nil
}

// getGitHubUserInfo fetches user info from GitHub
func (h *OAuthHandler) getGitHubUserInfo(ctx context.Context, client *http.Client) (*UserInfo, error) {
	resp, err := client.Get("https://api.github.com/user")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var userInfo struct {
		ID    int    `json:"id"`
		Login string `json:"login"`
		Email string `json:"email"`
		Name  string `json:"name"`
		AvatarURL string `json:"avatar_url"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, err
	}

	// Get email if not public
	if userInfo.Email == "" {
		emailsResp, err := client.Get("https://api.github.com/user/emails")
		if err == nil {
			defer emailsResp.Body.Close()
			var emails []struct {
				Email   string `json:"email"`
				Primary bool   `json:"primary"`
			}
			if json.NewDecoder(emailsResp.Body).Decode(&emails) == nil {
				for _, email := range emails {
					if email.Primary {
						userInfo.Email = email.Email
						break
					}
				}
			}
		}
	}

	return &UserInfo{
		ID:            fmt.Sprintf("%d", userInfo.ID),
		Username:      userInfo.Login,
		Email:         userInfo.Email,
		FirstName:     userInfo.Name,
		LastName:      "",
		AvatarURL:     userInfo.AvatarURL,
		EmailVerified: true, // GitHub emails are verified
	}, nil
}

// getMicrosoftUserInfo fetches user info from Microsoft
func (h *OAuthHandler) getMicrosoftUserInfo(ctx context.Context, client *http.Client) (*UserInfo, error) {
	resp, err := client.Get("https://graph.microsoft.com/v1.0/me")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var userInfo struct {
		ID                string `json:"id"`
		UserPrincipalName string `json:"userPrincipalName"`
		DisplayName       string `json:"displayName"`
		GivenName         string `json:"givenName"`
		Surname           string `json:"surname"`
		Mail              string `json:"mail"`
		JobTitle          string `json:"jobTitle"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, err
	}

	return &UserInfo{
		ID:            userInfo.ID,
		Username:      userInfo.UserPrincipalName,
		Email:         userInfo.Mail,
		FirstName:     userInfo.GivenName,
		LastName:      userInfo.Surname,
		AvatarURL:     "",
		EmailVerified: true, // Microsoft emails are verified
	}, nil
}

// generateState generates a random state parameter
func (h *OAuthHandler) generateState() (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(bytes), nil
}

// generateUserID generates a unique user ID
func (h *OAuthHandler) generateUserID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return fmt.Sprintf("user_%x", bytes)
}

// generateSessionID generates a unique session ID
func (h *OAuthHandler) generateSessionID() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	return fmt.Sprintf("session_%x", bytes)
}

// generateTokens generates JWT access and refresh tokens
func (h *OAuthHandler) generateTokens(user *User, sessionID string) (string, string, error) {
	// Access token (1 hour)
	accessClaims := &JWTClaims{
		UserID:    user.ID,
		Username:  user.Username,
		Email:     user.Email,
		Roles:     user.Roles,
		SessionID: sessionID,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "t3ss-auth",
			Subject:   user.ID,
		},
	}

	accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
	accessTokenString, err := accessToken.SignedString(h.jwtSecret)
	if err != nil {
		return "", "", err
	}

	// Refresh token (30 days)
	refreshClaims := &JWTClaims{
		UserID:    user.ID,
		SessionID: sessionID,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(30 * 24 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "t3ss-auth",
			Subject:   user.ID,
		},
	}

	refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
	refreshTokenString, err := refreshToken.SignedString(h.jwtSecret)
	if err != nil {
		return "", "", err
	}

	return accessTokenString, refreshTokenString, nil
}

// convertUserToProto converts User to protobuf User
func (h *OAuthHandler) convertUserToProto(user *User) *pb.User {
	return &pb.User{
		Id:            user.ID,
		Username:      user.Username,
		Email:         user.Email,
		FirstName:     user.FirstName,
		LastName:      user.LastName,
		AvatarUrl:     user.AvatarURL,
		EmailVerified: user.EmailVerified,
		IsActive:      user.IsActive,
		CreatedAt:     timestamppb.New(user.CreatedAt),
		UpdatedAt:     timestamppb.New(user.UpdatedAt),
		LastLogin:     timestamppb.New(user.LastLogin),
		Roles:         user.Roles,
		Metadata:      user.Metadata,
	}
}

// getEnv gets environment variable with default value
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// RegisterOAuthHandler registers OAuth handler with gRPC server
func RegisterOAuthHandler(s *grpc.Server, handler *OAuthHandler) {
	pb.RegisterAuthServiceServer(s, handler)
}
