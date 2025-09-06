# T3SS Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the T3SS (Triple-S Search System) - a Google-level search engine built with polyglot microservices architecture.

## Prerequisites

### System Requirements
- **Kubernetes Cluster**: Version 1.20+ with at least 3 nodes
- **CPU**: 8+ cores per node
- **Memory**: 32+ GB RAM per node
- **Storage**: 100+ GB SSD per node
- **Network**: 1+ Gbps bandwidth

### Required Tools
- `kubectl` (v1.20+)
- `helm` (v3.0+)
- `docker` (v20.0+)
- `protoc` (v3.0+)
- `envsubst` (GNU gettext)

### External Dependencies
- **Redis**: 6.0+ (for caching and job queue)
- **PostgreSQL**: 13+ (for persistent storage)
- **etcd**: 3.5+ (for service discovery)
- **Prometheus**: 2.30+ (for metrics)
- **Grafana**: 8.0+ (for visualization)
- **Jaeger**: 1.20+ (for tracing)

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/qisslabs/t3ss.git
cd t3ss
chmod +x scripts/deploy.sh
```

### 2. Configure Environment
```bash
export ENVIRONMENT=production
export NAMESPACE=t3ss
export REGISTRY=gcr.io/your-project
export VERSION=latest
export REPLICAS=3
```

### 3. Deploy
```bash
./scripts/deploy.sh
```

## Detailed Deployment

### 1. Generate Protobuf Files
```bash
# Generate Go protobuf files
protoc --go_out=shared_libs/proto/generated/go \
       --go-grpc_out=shared_libs/proto/generated/go \
       --go_opt=paths=source_relative \
       --go-grpc_opt=paths=source_relative \
       shared_libs/proto/*.proto

# Generate Python protobuf files
protoc --python_out=shared_libs/proto/generated/python \
       --grpc_python_out=shared_libs/proto/generated/python \
       shared_libs/proto/*.proto

# Generate Rust protobuf files
protoc --rust_out=shared_libs/proto/generated/rust \
       --grpc-rust_out=shared_libs/proto/generated/rust \
       shared_libs/proto/*.proto
```

### 2. Build Docker Images
```bash
# Build all services
docker build -t $REGISTRY/api-gateway:$VERSION -f frontend/api_gateway/Dockerfile .
docker build -t $REGISTRY/search-service:$VERSION -f backend_services/search/Dockerfile .
docker build -t $REGISTRY/indexing-service:$VERSION -f backend_services/indexing/Dockerfile .
docker build -t $REGISTRY/ranking-service:$VERSION -f backend_services/ranking/Dockerfile .
docker build -t $REGISTRY/ml-services:$VERSION -f backend_services/ml/Dockerfile .
docker build -t $REGISTRY/auth-service:$VERSION -f backend_services/auth/Dockerfile .
docker build -t $REGISTRY/analytics-service:$VERSION -f backend_services/analytics/Dockerfile .
```

### 3. Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace $NAMESPACE

# Apply configurations
kubectl apply -f infrastructure/deployment/kubernetes/configmaps/ -n $NAMESPACE
kubectl apply -f infrastructure/deployment/kubernetes/secrets/ -n $NAMESPACE
kubectl apply -f infrastructure/deployment/kubernetes/services/ -n $NAMESPACE

# Deploy services
kubectl apply -f infrastructure/deployment/kubernetes/deployments/ -n $NAMESPACE
kubectl apply -f infrastructure/deployment/kubernetes/ingress/ -n $NAMESPACE
kubectl apply -f infrastructure/deployment/kubernetes/hpa/ -n $NAMESPACE
```

### 4. Deploy Monitoring Stack
```bash
# Deploy Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values infrastructure/monitoring/prometheus/values.yaml

# Deploy Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values infrastructure/monitoring/grafana/values.yaml

# Deploy Jaeger
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm install jaeger jaegertracing/jaeger \
  --namespace monitoring \
  --values infrastructure/monitoring/jaeger/values.yaml
```

## Configuration

### Environment Variables
```bash
# API Gateway
API_GATEWAY_PORT=8080
API_GATEWAY_HOST=0.0.0.0

# Search Service
SEARCH_SERVICE_PORT=8081
SEARCH_SERVICE_HOST=0.0.0.0

# Indexing Service
INDEXING_SERVICE_PORT=8082
INDEXING_SERVICE_HOST=0.0.0.0

# Ranking Service
RANKING_SERVICE_PORT=8083
RANKING_SERVICE_HOST=0.0.0.0

# ML Services
ML_SERVICES_PORT=8084
ML_SERVICES_HOST=0.0.0.0

# Auth Service
AUTH_SERVICE_PORT=8085
AUTH_SERVICE_HOST=0.0.0.0

# Analytics Service
ANALYTICS_SERVICE_PORT=8086
ANALYTICS_SERVICE_HOST=0.0.0.0

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# PostgreSQL
POSTGRES_HOST=postgresql
POSTGRES_PORT=5432
POSTGRES_USER=t3ss
POSTGRES_PASSWORD=your-postgres-password
POSTGRES_DB=t3ss

# etcd
ETCD_ENDPOINTS=etcd:2379
ETCD_USERNAME=root
ETCD_PASSWORD=your-etcd-password

# OAuth Providers
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
MICROSOFT_CLIENT_ID=your-microsoft-client-id
MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret

# JWT
JWT_SECRET=your-jwt-secret
JWT_EXPIRY=24h

# Rate Limiting
RATE_LIMIT_RPS=100
RATE_LIMIT_BURST=200

# Caching
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Logging
LOG_LEVEL=info
LOG_FORMAT=json

# Metrics
METRICS_ENABLED=true
METRICS_PORT=9090

# Tracing
TRACING_ENABLED=true
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

### Configuration Files
- `infrastructure/config/production_config.yaml` - Main configuration
- `infrastructure/deployment/kubernetes/configmaps/` - Kubernetes ConfigMaps
- `infrastructure/deployment/kubernetes/secrets/` - Kubernetes Secrets

## Testing

### Integration Tests
```bash
# Run integration tests
python shared_libs/testing/integration_tests/test_search_integration.py \
  --api-url http://your-api-gateway:8080 \
  --timeout 30 \
  --concurrent 10
```

### Load Tests
```bash
# Run load tests
python shared_libs/testing/load_tests/load_test.py \
  --api-url http://your-api-gateway:8080 \
  --duration 300 \
  --concurrent 100 \
  --rps 50
```

### Stress Tests
```bash
# Run stress tests
python shared_libs/testing/load_tests/load_test.py \
  --api-url http://your-api-gateway:8080 \
  --stress
```

## Monitoring

### Accessing Dashboards
- **Grafana**: http://your-grafana:3000 (admin/admin)
- **Prometheus**: http://your-prometheus:9090
- **Jaeger**: http://your-jaeger:16686

### Key Metrics
- **Request Rate**: `sum(rate(http_requests_total[5m]))`
- **Response Time**: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`
- **Error Rate**: `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100`
- **Memory Usage**: `sum(go_memstats_alloc_bytes) by (service)`
- **CPU Usage**: `rate(process_cpu_seconds_total[5m]) * 100`

### Alerts
- High error rate (>5%)
- High response time (>1s)
- High memory usage (>80%)
- High CPU usage (>80%)
- Service down

## Troubleshooting

### Common Issues

#### 1. Service Discovery Issues
```bash
# Check etcd connectivity
kubectl exec -it etcd-0 -- etcdctl endpoint health

# Check service registration
kubectl exec -it etcd-0 -- etcdctl get --prefix /t3ss/services/
```

#### 2. gRPC Communication Issues
```bash
# Check service endpoints
kubectl get endpoints -n $NAMESPACE

# Check service logs
kubectl logs -f deployment/api-gateway -n $NAMESPACE
kubectl logs -f deployment/search-service -n $NAMESPACE
```

#### 3. Database Connection Issues
```bash
# Check PostgreSQL connectivity
kubectl exec -it postgresql-0 -- psql -U t3ss -d t3ss -c "SELECT 1;"

# Check Redis connectivity
kubectl exec -it redis-0 -- redis-cli ping
```

#### 4. Performance Issues
```bash
# Check resource usage
kubectl top pods -n $NAMESPACE

# Check HPA status
kubectl get hpa -n $NAMESPACE

# Check node resources
kubectl top nodes
```

### Debug Commands
```bash
# Get all resources
kubectl get all -n $NAMESPACE

# Describe problematic pod
kubectl describe pod <pod-name> -n $NAMESPACE

# Check events
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'

# Port forward for debugging
kubectl port-forward service/api-gateway 8080:8080 -n $NAMESPACE
```

## Scaling

### Horizontal Scaling
```bash
# Scale API Gateway
kubectl scale deployment api-gateway --replicas=5 -n $NAMESPACE

# Scale Search Service
kubectl scale deployment search-service --replicas=3 -n $NAMESPACE

# Scale Indexing Service
kubectl scale deployment indexing-service --replicas=2 -n $NAMESPACE
```

### Vertical Scaling
```yaml
# Update resource limits in deployment YAML
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Security

### Network Policies
```bash
# Apply network policies
kubectl apply -f infrastructure/deployment/kubernetes/network-policies/ -n $NAMESPACE
```

### RBAC
```bash
# Apply RBAC rules
kubectl apply -f infrastructure/deployment/kubernetes/rbac/ -n $NAMESPACE
```

### Secrets Management
```bash
# Create secrets
kubectl create secret generic t3ss-secrets \
  --from-literal=redis-password=your-redis-password \
  --from-literal=postgres-password=your-postgres-password \
  --from-literal=jwt-secret=your-jwt-secret \
  -n $NAMESPACE
```

## Backup and Recovery

### Database Backup
```bash
# Backup PostgreSQL
kubectl exec -it postgresql-0 -- pg_dump -U t3ss t3ss > backup.sql

# Restore PostgreSQL
kubectl exec -i postgresql-0 -- psql -U t3ss t3ss < backup.sql
```

### Configuration Backup
```bash
# Backup configurations
kubectl get configmaps -n $NAMESPACE -o yaml > configmaps-backup.yaml
kubectl get secrets -n $NAMESPACE -o yaml > secrets-backup.yaml
```

## Maintenance

### Rolling Updates
```bash
# Update API Gateway
kubectl set image deployment/api-gateway api-gateway=$REGISTRY/api-gateway:$NEW_VERSION -n $NAMESPACE

# Check rollout status
kubectl rollout status deployment/api-gateway -n $NAMESPACE

# Rollback if needed
kubectl rollout undo deployment/api-gateway -n $NAMESPACE
```

### Health Checks
```bash
# Check all deployments
kubectl get deployments -n $NAMESPACE

# Check pod health
kubectl get pods -n $NAMESPACE

# Check service health
kubectl get services -n $NAMESPACE
```

## Support

For support and questions:
- **Documentation**: https://docs.t3ss.qisslabs.com
- **Issues**: https://github.com/qisslabs/t3ss/issues
- **Discussions**: https://github.com/qisslabs/t3ss/discussions
- **Email**: support@qisslabs.com

## License

Copyright (c) 2025 Qiss Labs. All Rights Reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.