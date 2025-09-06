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
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/github"
	"golang.org/x/oauth2/microsoft"
)

// OAuthConfig holds OAuth configuration
type OAuthConfig struct {
	GoogleClientID     string `yaml:"google_client_id"`
	GoogleClientSecret string `yaml:"google_client_secret"`
	GitHubClientID     string `yaml:"github_client_id"`
	GitHubClientSecret string `yaml:"github_client_secret"`
	MicrosoftClientID  string `yaml:"microsoft_client_id"`
	MicrosoftClientSecret string `yaml:"microsoft_client_secret"`
	RedirectURL        string `yaml:"redirect_url"`
	JWTSecret          string `yaml:"jwt_secret"`
	JWTExpiry          time.Duration `yaml:"jwt_expiry"`
	StateExpiry        time.Duration `yaml:"state_expiry"`
}

// OAuthProvider represents different OAuth providers
type OAuthProvider string

const (
	ProviderGoogle   OAuthProvider = "google"
	ProviderGitHub   OAuthProvider = "github"
	ProviderMicrosoft OAuthProvider = "microsoft"
)

// OAuthHandler handles OAuth authentication flows
type OAuthHandler struct {
	config     OAuthConfig
	redisClient *redis.Client
	logger     *zap.Logger
	providers  map[OAuthProvider]*oauth2.Config
}

// UserInfo represents user information from OAuth provider
type UserInfo struct {
	ID       string `json:"id"`
	Email    string `json:"email"`
	Name     string `json:"name"`
	Picture  string `json:"picture"`
	Provider string `json:"provider"`
}

// AuthToken represents authentication token
type AuthToken struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresAt    time.Time `json:"expires_at"`
	TokenType    string    `json:"token_type"`
	Scope        string    `json:"scope"`
}

// JWTClaims represents JWT claims
type JWTClaims struct {
	UserID   string `json:"user_id"`
	Email    string `json:"email"`
	Provider string `json:"provider"`
	jwt.RegisteredClaims
}

// NewOAuthHandler creates a new OAuth handler
func NewOAuthHandler(config OAuthConfig, redisClient *redis.Client, logger *zap.Logger) *OAuthHandler {
	handler := &OAuthHandler{
		config:     config,
		redisClient: redisClient,
		logger:     logger,
		providers:  make(map[OAuthProvider]*oauth2.Config),
	}

	// Initialize OAuth providers
	handler.initializeProviders()

	return handler
}

// initializeProviders initializes OAuth provider configurations
func (h *OAuthHandler) initializeProviders() {
	// Google OAuth
	if h.config.GoogleClientID != "" && h.config.GoogleClientSecret != "" {
		h.providers[ProviderGoogle] = &oauth2.Config{
			ClientID:     h.config.GoogleClientID,
			ClientSecret: h.config.GoogleClientSecret,
			RedirectURL:  h.config.RedirectURL + "/auth/google/callback",
			Scopes: []string{
				"https://www.googleapis.com/auth/userinfo.email",
				"https://www.googleapis.com/auth/userinfo.profile",
			},
			Endpoint: google.Endpoint,
		}
	}

	// GitHub OAuth
	if h.config.GitHubClientID != "" && h.config.GitHubClientSecret != "" {
		h.providers[ProviderGitHub] = &oauth2.Config{
			ClientID:     h.config.GitHubClientID,
			ClientSecret: h.config.GitHubClientSecret,
			RedirectURL:  h.config.RedirectURL + "/auth/github/callback",
			Scopes:       []string{"user:email"},
			Endpoint:     github.Endpoint,
		}
	}

	// Microsoft OAuth
	if h.config.MicrosoftClientID != "" && h.config.MicrosoftClientSecret != "" {
		h.providers[ProviderMicrosoft] = &oauth2.Config{
			ClientID:     h.config.MicrosoftClientID,
			ClientSecret: h.config.MicrosoftClientSecret,
			RedirectURL:  h.config.RedirectURL + "/auth/microsoft/callback",
			Scopes: []string{
				"https://graph.microsoft.com/User.Read",
				"https://graph.microsoft.com/User.ReadBasic.All",
			},
			Endpoint: microsoft.AzureADEndpoint("common"),
		}
	}
}

// GetAuthURL generates OAuth authorization URL
func (h *OAuthHandler) GetAuthURL(provider OAuthProvider, state string) (string, error) {
	config, exists := h.providers[provider]
	if !exists {
		return "", fmt.Errorf("unsupported OAuth provider: %s", provider)
	}

	// Generate and store state
	stateData := map[string]interface{}{
		"provider": string(provider),
		"timestamp": time.Now().Unix(),
		"nonce":     h.generateNonce(),
	}

	stateJSON, err := json.Marshal(stateData)
	if err != nil {
		return "", fmt.Errorf("failed to marshal state: %w", err)
	}

	// Store state in Redis
	stateKey := fmt.Sprintf("oauth_state:%s", state)
	err = h.redisClient.Set(context.Background(), stateKey, stateJSON, h.config.StateExpiry).Err()
	if err != nil {
		return "", fmt.Errorf("failed to store state: %w", err)
	}

	// Generate authorization URL
	authURL := config.AuthCodeURL(state, oauth2.AccessTypeOffline, oauth2.SetAuthURLParam("prompt", "consent"))
	
	h.logger.Info("Generated OAuth auth URL", 
		zap.String("provider", string(provider)),
		zap.String("state", state))

	return authURL, nil
}

// HandleCallback handles OAuth callback
func (h *OAuthHandler) HandleCallback(provider OAuthProvider, code, state string) (*AuthToken, *UserInfo, error) {
	// Validate state
	if err := h.validateState(state); err != nil {
		return nil, nil, fmt.Errorf("invalid state: %w", err)
	}

	config, exists := h.providers[provider]
	if !exists {
		return nil, nil, fmt.Errorf("unsupported OAuth provider: %s", provider)
	}

	// Exchange code for token
	token, err := config.Exchange(context.Background(), code)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to exchange code for token: %w", err)
	}

	// Get user info
	userInfo, err := h.getUserInfo(provider, token)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get user info: %w", err)
	}

	// Create auth token
	authToken := &AuthToken{
		AccessToken:  token.AccessToken,
		RefreshToken: token.RefreshToken,
		ExpiresAt:    token.Expiry,
		TokenType:    token.TokenType,
		Scope:        strings.Join(token.Extra("scope").([]string), " "),
	}

	// Clean up state
	h.redisClient.Del(context.Background(), fmt.Sprintf("oauth_state:%s", state))

	h.logger.Info("OAuth callback successful",
		zap.String("provider", string(provider)),
		zap.String("user_id", userInfo.ID),
		zap.String("email", userInfo.Email))

	return authToken, userInfo, nil
}

// getUserInfo retrieves user information from OAuth provider
func (h *OAuthHandler) getUserInfo(provider OAuthProvider, token *oauth2.Token) (*UserInfo, error) {
	client := oauth2.NewClient(context.Background(), oauth2.StaticTokenSource(token))

	switch provider {
	case ProviderGoogle:
		return h.getGoogleUserInfo(client)
	case ProviderGitHub:
		return h.getGitHubUserInfo(client)
	case ProviderMicrosoft:
		return h.getMicrosoftUserInfo(client)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// getGoogleUserInfo retrieves user info from Google
func (h *OAuthHandler) getGoogleUserInfo(client *http.Client) (*UserInfo, error) {
	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Google API returned status %d", resp.StatusCode)
	}

	var userInfo struct {
		ID      string `json:"id"`
		Email   string `json:"email"`
		Name    string `json:"name"`
		Picture string `json:"picture"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, err
	}

	return &UserInfo{
		ID:       userInfo.ID,
		Email:    userInfo.Email,
		Name:     userInfo.Name,
		Picture:  userInfo.Picture,
		Provider: string(ProviderGoogle),
	}, nil
}

// getGitHubUserInfo retrieves user info from GitHub
func (h *OAuthHandler) getGitHubUserInfo(client *http.Client) (*UserInfo, error) {
	// Get user info
	userResp, err := client.Get("https://api.github.com/user")
	if err != nil {
		return nil, err
	}
	defer userResp.Body.Close()

	if userResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API returned status %d", userResp.StatusCode)
	}

	var userInfo struct {
		ID       int    `json:"id"`
		Email    string `json:"email"`
		Name     string `json:"name"`
		AvatarURL string `json:"avatar_url"`
	}

	if err := json.NewDecoder(userResp.Body).Decode(&userInfo); err != nil {
		return nil, err
	}

	// If email is not public, get it from emails endpoint
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
		ID:       fmt.Sprintf("%d", userInfo.ID),
		Email:    userInfo.Email,
		Name:     userInfo.Name,
		Picture:  userInfo.AvatarURL,
		Provider: string(ProviderGitHub),
	}, nil
}

// getMicrosoftUserInfo retrieves user info from Microsoft
func (h *OAuthHandler) getMicrosoftUserInfo(client *http.Client) (*UserInfo, error) {
	resp, err := client.Get("https://graph.microsoft.com/v1.0/me")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Microsoft Graph API returned status %d", resp.StatusCode)
	}

	var userInfo struct {
		ID                string `json:"id"`
		Mail              string `json:"mail"`
		UserPrincipalName string `json:"userPrincipalName"`
		DisplayName       string `json:"displayName"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, err
	}

	// Use mail if available, otherwise use userPrincipalName
	email := userInfo.Mail
	if email == "" {
		email = userInfo.UserPrincipalName
	}

	return &UserInfo{
		ID:       userInfo.ID,
		Email:    email,
		Name:     userInfo.DisplayName,
		Picture:  "", // Microsoft Graph doesn't provide profile picture in basic profile
		Provider: string(ProviderMicrosoft),
	}, nil
}

// GenerateJWT generates a JWT token for the user
func (h *OAuthHandler) GenerateJWT(userInfo *UserInfo) (string, error) {
	claims := JWTClaims{
		UserID:   userInfo.ID,
		Email:    userInfo.Email,
		Provider: userInfo.Provider,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(h.config.JWTExpiry)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "t3ss-auth",
			Subject:   userInfo.ID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(h.config.JWTSecret))
}

// ValidateJWT validates a JWT token
func (h *OAuthHandler) ValidateJWT(tokenString string) (*JWTClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(h.config.JWTSecret), nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*JWTClaims); ok && token.Valid {
		return claims, nil
	}

	return nil, fmt.Errorf("invalid token")
}

// RefreshToken refreshes an OAuth token
func (h *OAuthHandler) RefreshToken(provider OAuthProvider, refreshToken string) (*AuthToken, error) {
	config, exists := h.providers[provider]
	if !exists {
		return nil, fmt.Errorf("unsupported OAuth provider: %s", provider)
	}

	token := &oauth2.Token{
		RefreshToken: refreshToken,
	}

	tokenSource := config.TokenSource(context.Background(), token)
	newToken, err := tokenSource.Token()
	if err != nil {
		return nil, fmt.Errorf("failed to refresh token: %w", err)
	}

	return &AuthToken{
		AccessToken:  newToken.AccessToken,
		RefreshToken: newToken.RefreshToken,
		ExpiresAt:    newToken.Expiry,
		TokenType:    newToken.TokenType,
	}, nil
}

// RevokeToken revokes an OAuth token
func (h *OAuthHandler) RevokeToken(provider OAuthProvider, token string) error {
	var revokeURL string

	switch provider {
	case ProviderGoogle:
		revokeURL = "https://oauth2.googleapis.com/revoke"
	case ProviderGitHub:
		// GitHub doesn't have a token revocation endpoint
		return nil
	case ProviderMicrosoft:
		revokeURL = "https://graph.microsoft.com/v1.0/me/revokeSignInSessions"
	default:
		return fmt.Errorf("unsupported provider: %s", provider)
	}

	// Create revocation request
	form := url.Values{}
	form.Set("token", token)

	resp, err := http.PostForm(revokeURL, form)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("token revocation failed with status %d", resp.StatusCode)
	}

	return nil
}

// validateState validates OAuth state parameter
func (h *OAuthHandler) validateState(state string) error {
	stateKey := fmt.Sprintf("oauth_state:%s", state)
	stateData, err := h.redisClient.Get(context.Background(), stateKey).Result()
	if err != nil {
		return fmt.Errorf("state not found or expired")
	}

	var stateInfo map[string]interface{}
	if err := json.Unmarshal([]byte(stateData), &stateInfo); err != nil {
		return fmt.Errorf("invalid state format")
	}

	// Check timestamp
	timestamp, ok := stateInfo["timestamp"].(float64)
	if !ok {
		return fmt.Errorf("invalid state timestamp")
	}

	if time.Now().Unix()-int64(timestamp) > int64(h.config.StateExpiry.Seconds()) {
		return fmt.Errorf("state expired")
	}

	return nil
}

// generateNonce generates a random nonce
func (h *OAuthHandler) generateNonce() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	return base64.URLEncoding.EncodeToString(bytes)
}

// GetSupportedProviders returns list of supported OAuth providers
func (h *OAuthHandler) GetSupportedProviders() []OAuthProvider {
	var providers []OAuthProvider
	for provider := range h.providers {
		providers = append(providers, provider)
	}
	return providers
}

// IsProviderSupported checks if a provider is supported
func (h *OAuthHandler) IsProviderSupported(provider OAuthProvider) bool {
	_, exists := h.providers[provider]
	return exists
}