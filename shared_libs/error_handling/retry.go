// T3SS Project
// File: shared_libs/error_handling/retry.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package error_handling

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// RetryConfig represents retry configuration
type RetryConfig struct {
	MaxAttempts    int           `json:"max_attempts"`
	InitialDelay   time.Duration `json:"initial_delay"`
	MaxDelay       time.Duration `json:"max_delay"`
	Multiplier     float64       `json:"multiplier"`
	Jitter         bool          `json:"jitter"`
	RetryableCodes []codes.Code  `json:"retryable_codes"`
}

// DefaultRetryConfig returns a default retry configuration
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxAttempts:    3,
		InitialDelay:   100 * time.Millisecond,
		MaxDelay:       5 * time.Second,
		Multiplier:     2.0,
		Jitter:         true,
		RetryableCodes: []codes.Code{
			codes.Unavailable,
			codes.DeadlineExceeded,
			codes.ResourceExhausted,
			codes.Aborted,
			codes.Internal,
		},
	}
}

// RetryableError represents a retryable error
type RetryableError struct {
	Err        error
	RetryAfter time.Duration
}

func (e *RetryableError) Error() string {
	return fmt.Sprintf("retryable error: %v", e.Err)
}

func (e *RetryableError) Unwrap() error {
	return e.Err
}

// Retry executes a function with retry logic
func Retry(ctx context.Context, config *RetryConfig, fn func() error) error {
	if config == nil {
		config = DefaultRetryConfig()
	}

	var lastErr error
	delay := config.InitialDelay

	for attempt := 0; attempt < config.MaxAttempts; attempt++ {
		// Check if context is cancelled
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Execute function
		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err, config.RetryableCodes) {
			return err
		}

		// Don't retry on last attempt
		if attempt == config.MaxAttempts-1 {
			break
		}

		// Calculate delay with exponential backoff
		actualDelay := delay
		if config.Jitter {
			// Add jitter to prevent thundering herd
			jitter := time.Duration(rand.Float64() * float64(delay) * 0.1)
			actualDelay += jitter
		}

		// Wait before retry
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(actualDelay):
		}

		// Calculate next delay
		delay = time.Duration(float64(delay) * config.Multiplier)
		if delay > config.MaxDelay {
			delay = config.MaxDelay
		}
	}

	return &RetryableError{
		Err:        lastErr,
		RetryAfter: delay,
	}
}

// RetryWithBackoff executes a function with exponential backoff retry
func RetryWithBackoff(ctx context.Context, config *RetryConfig, fn func() error) error {
	return Retry(ctx, config, fn)
}

// RetryWithLinearBackoff executes a function with linear backoff retry
func RetryWithLinearBackoff(ctx context.Context, maxAttempts int, delay time.Duration, fn func() error) error {
	config := &RetryConfig{
		MaxAttempts:    maxAttempts,
		InitialDelay:   delay,
		MaxDelay:       delay,
		Multiplier:     1.0,
		Jitter:         false,
		RetryableCodes: DefaultRetryConfig().RetryableCodes,
	}
	return Retry(ctx, config, fn)
}

// RetryWithExponentialBackoff executes a function with exponential backoff retry
func RetryWithExponentialBackoff(ctx context.Context, maxAttempts int, initialDelay, maxDelay time.Duration, fn func() error) error {
	config := &RetryConfig{
		MaxAttempts:    maxAttempts,
		InitialDelay:   initialDelay,
		MaxDelay:       maxDelay,
		Multiplier:     2.0,
		Jitter:         true,
		RetryableCodes: DefaultRetryConfig().RetryableCodes,
	}
	return Retry(ctx, config, fn)
}

// RetryWithCustomBackoff executes a function with custom backoff strategy
func RetryWithCustomBackoff(ctx context.Context, maxAttempts int, backoffFunc func(attempt int) time.Duration, fn func() error) error {
	var lastErr error

	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Check if context is cancelled
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Execute function
		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Don't retry on last attempt
		if attempt == maxAttempts-1 {
			break
		}

		// Calculate delay using custom function
		delay := backoffFunc(attempt)

		// Wait before retry
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}

	return &RetryableError{
		Err:        lastErr,
		RetryAfter: backoffFunc(maxAttempts - 1),
	}
}

// CircuitBreaker represents a circuit breaker
type CircuitBreaker struct {
	maxFailures   int
	resetTimeout  time.Duration
	state         CircuitState
	failureCount  int
	lastFailTime  time.Time
	mu            sync.RWMutex
}

// CircuitState represents circuit breaker state
type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        StateClosed,
	}
}

// Execute executes a function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() error) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Check circuit state
	switch cb.state {
	case StateOpen:
		if time.Since(cb.lastFailTime) > cb.resetTimeout {
			cb.state = StateHalfOpen
		} else {
			return fmt.Errorf("circuit breaker is open")
		}
	case StateHalfOpen:
		// Allow one request to test if service is back
	case StateClosed:
		// Normal operation
	}

	// Execute function
	err := fn()

	// Update circuit state based on result
	if err != nil {
		cb.failureCount++
		cb.lastFailTime = time.Now()

		if cb.failureCount >= cb.maxFailures {
			cb.state = StateOpen
		}
	} else {
		// Reset on success
		cb.failureCount = 0
		cb.state = StateClosed
	}

	return err
}

// GetState returns the current circuit breaker state
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Bulkhead represents a bulkhead pattern for resource isolation
type Bulkhead struct {
	maxConcurrency int
	semaphore     chan struct{}
}

// NewBulkhead creates a new bulkhead
func NewBulkhead(maxConcurrency int) *Bulkhead {
	return &Bulkhead{
		maxConcurrency: maxConcurrency,
		semaphore:     make(chan struct{}, maxConcurrency),
	}
}

// Execute executes a function with bulkhead protection
func (b *Bulkhead) Execute(fn func() error) error {
	// Acquire semaphore
	select {
	case b.semaphore <- struct{}{}:
		defer func() { <-b.semaphore }()
	default:
		return fmt.Errorf("bulkhead is full")
	}

	return fn()
}

// Timeout represents a timeout wrapper
type Timeout struct {
	timeout time.Duration
}

// NewTimeout creates a new timeout wrapper
func NewTimeout(timeout time.Duration) *Timeout {
	return &Timeout{timeout: timeout}
}

// Execute executes a function with timeout
func (t *Timeout) Execute(fn func() error) error {
	ctx, cancel := context.WithTimeout(context.Background(), t.timeout)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- fn()
	}()

	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// RateLimiter represents a rate limiter
type RateLimiter struct {
	tokens   int
	capacity int
	rate     time.Duration
	lastRefill time.Time
	mu       sync.Mutex
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(capacity int, rate time.Duration) *RateLimiter {
	return &RateLimiter{
		tokens:     capacity,
		capacity:   capacity,
		rate:       rate,
		lastRefill: time.Now(),
	}
}

// Execute executes a function with rate limiting
func (rl *RateLimiter) Execute(fn func() error) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	// Refill tokens based on elapsed time
	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	tokensToAdd := int(elapsed / rl.rate)
	
	if tokensToAdd > 0 {
		rl.tokens = min(rl.capacity, rl.tokens+tokensToAdd)
		rl.lastRefill = now
	}

	// Check if we have tokens available
	if rl.tokens <= 0 {
		return fmt.Errorf("rate limit exceeded")
	}

	// Consume token
	rl.tokens--

	return fn()
}

// Helper functions

// isRetryableError checks if an error is retryable
func isRetryableError(err error, retryableCodes []codes.Code) bool {
	if err == nil {
		return false
	}

	// Check gRPC status codes
	if st, ok := status.FromError(err); ok {
		for _, code := range retryableCodes {
			if st.Code() == code {
				return true
			}
		}
	}

	// Check for specific error types
	switch err.(type) {
	case *RetryableError:
		return true
	default:
		return false
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ExponentialBackoff calculates exponential backoff delay
func ExponentialBackoff(attempt int, baseDelay time.Duration, maxDelay time.Duration) time.Duration {
	delay := time.Duration(float64(baseDelay) * math.Pow(2, float64(attempt)))
	if delay > maxDelay {
		delay = maxDelay
	}
	return delay
}

// LinearBackoff calculates linear backoff delay
func LinearBackoff(attempt int, baseDelay time.Duration, maxDelay time.Duration) time.Duration {
	delay := baseDelay * time.Duration(attempt+1)
	if delay > maxDelay {
		delay = maxDelay
	}
	return delay
}

// FibonacciBackoff calculates Fibonacci backoff delay
func FibonacciBackoff(attempt int, baseDelay time.Duration, maxDelay time.Duration) time.Duration {
	fib := fibonacci(attempt + 1)
	delay := baseDelay * time.Duration(fib)
	if delay > maxDelay {
		delay = maxDelay
	}
	return delay
}

// fibonacci calculates the nth Fibonacci number
func fibonacci(n int) int {
	if n <= 1 {
		return n
	}
	return fibonacci(n-1) + fibonacci(n-2)
}

// Jitter adds random jitter to a delay
func Jitter(delay time.Duration, jitterPercent float64) time.Duration {
	if jitterPercent <= 0 {
		return delay
	}
	
	jitter := time.Duration(float64(delay) * jitterPercent * rand.Float64())
	return delay + jitter
}