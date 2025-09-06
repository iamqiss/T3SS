// T3SS Project
// File: backend_services/security/auth_service.go
// (c) 2025 Qiss Labs. All Rights Reserved.

package security

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/bcrypt"
)

// AuthService provides enterprise-grade authentication and authorization
type AuthService struct {
	jwtSecret     []byte
	apiKeys       map[string]*APIKey
	sessions      map[string]*Session
	rateLimits    map[string]*RateLimit
	mu            sync.RWMutex
	cleanupTicker *time.Ticker
}

// APIKey represents an API key with permissions
type APIKey struct {
	ID          string    `json:"id"`
	KeyHash     string    `json:"key_hash"`
	Permissions []string  `json:"permissions"`
	RateLimit   int       `json:"rate_limit"`
	ExpiresAt   time.Time `json:"expires_at"`
	CreatedAt   time.Time `json:"created_at"`
	LastUsed    time.Time `json:"last_used"`
	IsActive    bool      `json:"is_active"`
}

// Session represents a user session
type Session struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
	IPAddress string    `json:"ip_address"`
	UserAgent string    `json:"user_agent"`
	IsActive  bool      `json:"is_active"`
}

// RateLimit tracks rate limiting information
type RateLimit struct {
	Requests     int       `json:"requests"`
	WindowStart  time.Time `json:"window_start"`
	Limit        int       `json:"limit"`
	WindowSize   time.Duration `json:"window_size"`
}

// JWTClaims represents JWT token claims
type JWTClaims struct {
	UserID      string   `json:"user_id"`
	Permissions []string `json:"permissions"`
	jwt.RegisteredClaims
}

// NewAuthService creates a new authentication service
func NewAuthService(jwtSecret string) *AuthService {
	service := &AuthService{
		jwtSecret:  []byte(jwtSecret),
		apiKeys:    make(map[string]*APIKey),
		sessions:   make(map[string]*Session),
		rateLimits: make(map[string]*RateLimit),
	}

	// Start cleanup routine
	service.cleanupTicker = time.NewTicker(5 * time.Minute)
	go service.cleanupRoutine()

	return service
}

// HashPassword hashes a password using Argon2
func (a *AuthService) HashPassword(password string) (string, error) {
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		return "", err
	}

	hash := argon2.IDKey([]byte(password), salt, 1, 64*1024, 4, 32)
	return base64.StdEncoding.EncodeToString(append(salt, hash...)), nil
}

// VerifyPassword verifies a password against its hash
func (a *AuthService) VerifyPassword(password, hash string) bool {
	decoded, err := base64.StdEncoding.DecodeString(hash)
	if err != nil || len(decoded) < 16 {
		return false
	}

	salt := decoded[:16]
	expectedHash := decoded[16:]

	actualHash := argon2.IDKey([]byte(password), salt, 1, 64*1024, 4, 32)
	return len(actualHash) == len(expectedHash) && 
		   subtle.ConstantTimeCompare(actualHash, expectedHash) == 1
}

// GenerateJWT creates a JWT token
func (a *AuthService) GenerateJWT(userID string, permissions []string, expiresIn time.Duration) (string, error) {
	claims := JWTClaims{
		UserID:      userID,
		Permissions: permissions,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(expiresIn)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "t3ss",
			Subject:   userID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(a.jwtSecret)
}

// VerifyJWT verifies a JWT token
func (a *AuthService) VerifyJWT(tokenString string) (*JWTClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return a.jwtSecret, nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*JWTClaims); ok && token.Valid {
		return claims, nil
	}

	return nil, fmt.Errorf("invalid token")
}

// CreateAPIKey generates a new API key
func (a *AuthService) CreateAPIKey(permissions []string, rateLimit int, expiresIn *time.Duration) (string, error) {
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return "", err
	}

	apiKey := base64.StdEncoding.EncodeToString(keyBytes)
	keyHash := a.hashAPIKey(apiKey)
	keyID := hex.EncodeToString(keyHash[:8])

	expiresAt := time.Now().Add(24 * time.Hour) // Default 24 hours
	if expiresIn != nil {
		expiresAt = time.Now().Add(*expiresIn)
	}

	apiKeyInfo := &APIKey{
		ID:          keyID,
		KeyHash:     hex.EncodeToString(keyHash),
		Permissions: permissions,
		RateLimit:   rateLimit,
		ExpiresAt:   expiresAt,
		CreatedAt:   time.Now(),
		IsActive:    true,
	}

	a.mu.Lock()
	a.apiKeys[keyID] = apiKeyInfo
	a.mu.Unlock()

	return apiKey, nil
}

// ValidateAPIKey validates an API key
func (a *AuthService) ValidateAPIKey(apiKey string) (*APIKey, error) {
	keyHash := a.hashAPIKey(apiKey)
	keyID := hex.EncodeToString(keyHash[:8])

	a.mu.RLock()
	keyInfo, exists := a.apiKeys[keyID]
	a.mu.RUnlock()

	if !exists || !keyInfo.IsActive {
		return nil, fmt.Errorf("invalid API key")
	}

	if time.Now().After(keyInfo.ExpiresAt) {
		return nil, fmt.Errorf("API key expired")
	}

	// Update last used time
	a.mu.Lock()
	keyInfo.LastUsed = time.Now()
	a.mu.Unlock()

	return keyInfo, nil
}

// CreateSession creates a new user session
func (a *AuthService) CreateSession(userID, ipAddress, userAgent string, expiresIn time.Duration) (string, error) {
	sessionID := a.generateSessionID()
	now := time.Now()

	session := &Session{
		ID:        sessionID,
		UserID:    userID,
		CreatedAt: now,
		ExpiresAt: now.Add(expiresIn),
		IPAddress: ipAddress,
		UserAgent: userAgent,
		IsActive:  true,
	}

	a.mu.Lock()
	a.sessions[sessionID] = session
	a.mu.Unlock()

	return sessionID, nil
}

// ValidateSession validates a session
func (a *AuthService) ValidateSession(sessionID string) (*Session, error) {
	a.mu.RLock()
	session, exists := a.sessions[sessionID]
	a.mu.RUnlock()

	if !exists || !session.IsActive {
		return nil, fmt.Errorf("invalid session")
	}

	if time.Now().After(session.ExpiresAt) {
		return nil, fmt.Errorf("session expired")
	}

	return session, nil
}

// CheckRateLimit checks if a request is within rate limits
func (a *AuthService) CheckRateLimit(identifier string, limit int, windowSize time.Duration) bool {
	now := time.Now()

	a.mu.Lock()
	defer a.mu.Unlock()

	rateLimit, exists := a.rateLimits[identifier]
	if !exists {
		rateLimit = &RateLimit{
			Requests:    0,
			WindowStart: now,
			Limit:       limit,
			WindowSize:  windowSize,
		}
		a.rateLimits[identifier] = rateLimit
	}

	// Reset window if needed
	if now.Sub(rateLimit.WindowStart) >= windowSize {
		rateLimit.Requests = 0
		rateLimit.WindowStart = now
	}

	if rateLimit.Requests >= limit {
		return false
	}

	rateLimit.Requests++
	return true
}

// RevokeAPIKey revokes an API key
func (a *AuthService) RevokeAPIKey(apiKey string) error {
	keyHash := a.hashAPIKey(apiKey)
	keyID := hex.EncodeToString(keyHash[:8])

	a.mu.Lock()
	defer a.mu.Unlock()

	if keyInfo, exists := a.apiKeys[keyID]; exists {
		keyInfo.IsActive = false
		return nil
	}

	return fmt.Errorf("API key not found")
}

// RevokeSession revokes a session
func (a *AuthService) RevokeSession(sessionID string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if session, exists := a.sessions[sessionID]; exists {
		session.IsActive = false
		return nil
	}

	return fmt.Errorf("session not found")
}

// GetAPIKeyInfo returns API key information
func (a *AuthService) GetAPIKeyInfo(apiKey string) (*APIKey, error) {
	return a.ValidateAPIKey(apiKey)
}

// GetSessionInfo returns session information
func (a *AuthService) GetSessionInfo(sessionID string) (*Session, error) {
	return a.ValidateSession(sessionID)
}

// Helper methods
func (a *AuthService) hashAPIKey(apiKey string) []byte {
	hash := sha256.Sum256([]byte(apiKey))
	return hash[:]
}

func (a *AuthService) generateSessionID() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

func (a *AuthService) cleanupRoutine() {
	for range a.cleanupTicker.C {
		a.cleanupExpired()
	}
}

func (a *AuthService) cleanupExpired() {
	now := time.Now()

	a.mu.Lock()
	defer a.mu.Unlock()

	// Cleanup expired API keys
	for id, keyInfo := range a.apiKeys {
		if now.After(keyInfo.ExpiresAt) {
			delete(a.apiKeys, id)
		}
	}

	// Cleanup expired sessions
	for id, session := range a.sessions {
		if now.After(session.ExpiresAt) {
			delete(a.sessions, id)
		}
	}

	// Cleanup old rate limits
	for id, rateLimit := range a.rateLimits {
		if now.Sub(rateLimit.WindowStart) > rateLimit.WindowSize*2 {
			delete(a.rateLimits, id)
		}
	}
}

// Close stops the cleanup routine
func (a *AuthService) Close() {
	if a.cleanupTicker != nil {
		a.cleanupTicker.Stop()
	}
}