#!/bin/bash

# T3SS Deployment Script
# (c) 2025 Qiss Labs. All Rights Reserved.

set -euo pipefail

# Configuration
NAMESPACE="t3ss"
ENVIRONMENT="${ENVIRONMENT:-production}"
REGISTRY="${REGISTRY:-gcr.io/t3ss}"
VERSION="${VERSION:-latest}"
REPLICAS="${REPLICAS:-3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check if protoc is installed
    if ! command -v protoc &> /dev/null; then
        log_error "protoc is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Generate protobuf files
generate_protobuf() {
    log_info "Generating protobuf files..."
    
    # Create output directories
    mkdir -p shared_libs/proto/generated/go
    mkdir -p shared_libs/proto/generated/python
    mkdir -p shared_libs/proto/generated/rust
    
    # Generate Go protobuf files
    for proto_file in shared_libs/proto/*.proto; do
        if [ -f "$proto_file" ]; then
            log_info "Generating Go code for $(basename "$proto_file")"
            protoc --go_out=shared_libs/proto/generated/go \
                   --go-grpc_out=shared_libs/proto/generated/go \
                   --go_opt=paths=source_relative \
                   --go-grpc_opt=paths=source_relative \
                   "$proto_file"
        fi
    done
    
    # Generate Python protobuf files
    for proto_file in shared_libs/proto/*.proto; do
        if [ -f "$proto_file" ]; then
            log_info "Generating Python code for $(basename "$proto_file")"
            protoc --python_out=shared_libs/proto/generated/python \
                   --grpc_python_out=shared_libs/proto/generated/python \
                   "$proto_file"
        fi
    done
    
    # Generate Rust protobuf files
    for proto_file in shared_libs/proto/*.proto; do
        if [ -f "$proto_file" ]; then
            log_info "Generating Rust code for $(basename "$proto_file")"
            protoc --rust_out=shared_libs/proto/generated/rust \
                   --grpc-rust_out=shared_libs/proto/generated/rust \
                   "$proto_file"
        fi
    done
    
    log_success "Protobuf files generated"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build API Gateway
    log_info "Building API Gateway image..."
    docker build -t "$REGISTRY/api-gateway:$VERSION" -f frontend/api_gateway/Dockerfile .
    
    # Build Search Service
    log_info "Building Search Service image..."
    docker build -t "$REGISTRY/search-service:$VERSION" -f backend_services/search/Dockerfile .
    
    # Build Indexing Service
    log_info "Building Indexing Service image..."
    docker build -t "$REGISTRY/indexing-service:$VERSION" -f backend_services/indexing/Dockerfile .
    
    # Build Ranking Service
    log_info "Building Ranking Service image..."
    docker build -t "$REGISTRY/ranking-service:$VERSION" -f backend_services/ranking/Dockerfile .
    
    # Build ML Services
    log_info "Building ML Services image..."
    docker build -t "$REGISTRY/ml-services:$VERSION" -f backend_services/ml/Dockerfile .
    
    # Build Auth Service
    log_info "Building Auth Service image..."
    docker build -t "$REGISTRY/auth-service:$VERSION" -f backend_services/auth/Dockerfile .
    
    # Build Analytics Service
    log_info "Building Analytics Service image..."
    docker build -t "$REGISTRY/analytics-service:$VERSION" -f backend_services/analytics/Dockerfile .
    
    log_success "Docker images built"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    # Push all images
    docker push "$REGISTRY/api-gateway:$VERSION"
    docker push "$REGISTRY/search-service:$VERSION"
    docker push "$REGISTRY/indexing-service:$VERSION"
    docker push "$REGISTRY/ranking-service:$VERSION"
    docker push "$REGISTRY/ml-services:$VERSION"
    docker push "$REGISTRY/auth-service:$VERSION"
    docker push "$REGISTRY/analytics-service:$VERSION"
    
    log_success "Images pushed to registry"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps
    log_info "Applying ConfigMaps..."
    kubectl apply -f infrastructure/deployment/kubernetes/configmaps/ -n "$NAMESPACE"
    
    # Apply Secrets
    log_info "Applying Secrets..."
    kubectl apply -f infrastructure/deployment/kubernetes/secrets/ -n "$NAMESPACE"
    
    # Apply Services
    log_info "Applying Services..."
    kubectl apply -f infrastructure/deployment/kubernetes/services/ -n "$NAMESPACE"
    
    # Apply Deployments
    log_info "Applying Deployments..."
    envsubst < infrastructure/deployment/kubernetes/deployments/api-gateway.yaml | kubectl apply -f - -n "$NAMESPACE"
    envsubst < infrastructure/deployment/kubernetes/deployments/search-service.yaml | kubectl apply -f - -n "$NAMESPACE"
    envsubst < infrastructure/deployment/kubernetes/deployments/indexing-service.yaml | kubectl apply -f - -n "$NAMESPACE"
    envsubst < infrastructure/deployment/kubernetes/deployments/ranking-service.yaml | kubectl apply -f - -n "$NAMESPACE"
    envsubst < infrastructure/deployment/kubernetes/deployments/ml-services.yaml | kubectl apply -f - -n "$NAMESPACE"
    envsubst < infrastructure/deployment/kubernetes/deployments/auth-service.yaml | kubectl apply -f - -n "$NAMESPACE"
    envsubst < infrastructure/deployment/kubernetes/deployments/analytics-service.yaml | kubectl apply -f - -n "$NAMESPACE"
    
    # Apply Ingress
    log_info "Applying Ingress..."
    kubectl apply -f infrastructure/deployment/kubernetes/ingress/ -n "$NAMESPACE"
    
    # Apply HPA
    log_info "Applying Horizontal Pod Autoscalers..."
    kubectl apply -f infrastructure/deployment/kubernetes/hpa/ -n "$NAMESPACE"
    
    log_success "Kubernetes deployment completed"
}

# Wait for deployment
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for all deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/api-gateway -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/search-service -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/indexing-service -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/ranking-service -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/ml-services -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/auth-service -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=300s deployment/analytics-service -n "$NAMESPACE"
    
    log_success "All deployments are ready"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get API Gateway service URL
    API_GATEWAY_URL=$(kubectl get service api-gateway -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$API_GATEWAY_URL" ]; then
        API_GATEWAY_URL=$(kubectl get service api-gateway -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Wait for API Gateway to be ready
    log_info "Waiting for API Gateway to be ready..."
    for i in {1..30}; do
        if curl -f "http://$API_GATEWAY_URL:8080/health" > /dev/null 2>&1; then
            log_success "API Gateway is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "API Gateway failed to become ready"
            exit 1
        fi
        sleep 10
    done
    
    # Run comprehensive health checks
    log_info "Running comprehensive health checks..."
    
    # Check API Gateway health
    if ! curl -f "http://$API_GATEWAY_URL:8080/health" > /dev/null 2>&1; then
        log_error "API Gateway health check failed"
        exit 1
    fi
    
    # Check search endpoint
    if ! curl -f "http://$API_GATEWAY_URL:8080/api/v1/search" -X POST \
         -H "Content-Type: application/json" \
         -d '{"query": "test"}' > /dev/null 2>&1; then
        log_error "Search endpoint health check failed"
        exit 1
    fi
    
    # Check metrics endpoint
    if ! curl -f "http://$API_GATEWAY_URL:8080/metrics" > /dev/null 2>&1; then
        log_error "Metrics endpoint health check failed"
        exit 1
    fi
    
    log_success "All health checks passed"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Get API Gateway service URL
    API_GATEWAY_URL=$(kubectl get service api-gateway -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$API_GATEWAY_URL" ]; then
        API_GATEWAY_URL=$(kubectl get service api-gateway -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Run integration tests
    cd shared_libs/testing
    python -m pytest integration_tests/ -v --api-url="http://$API_GATEWAY_URL:8080"
    
    log_success "Integration tests passed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files
    rm -rf shared_libs/proto/generated/
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting T3SS deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Registry: $REGISTRY"
    log_info "Version: $VERSION"
    log_info "Replicas: $REPLICAS"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    generate_protobuf
    build_images
    push_images
    deploy_kubernetes
    wait_for_deployment
    run_health_checks
    run_integration_tests
    
    log_success "T3SS deployment completed successfully!"
    log_info "API Gateway URL: http://$API_GATEWAY_URL:8080"
    log_info "Health Check URL: http://$API_GATEWAY_URL:8080/health"
    log_info "Metrics URL: http://$API_GATEWAY_URL:8080/metrics"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --environment ENV    Set environment (default: production)"
            echo "  --namespace NS       Set namespace (default: t3ss)"
            echo "  --registry REG       Set registry (default: gcr.io/t3ss)"
            echo "  --version VER        Set version (default: latest)"
            echo "  --replicas REP       Set replicas (default: 3)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main