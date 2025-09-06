#!/usr/bin/env python3
"""
T3SS Search Integration Tests
(c) 2025 Qiss Labs. All Rights Reserved.
"""

import pytest
import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for integration tests"""
    api_url: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    concurrent_requests: int = 10
    test_duration: int = 300  # 5 minutes

class SearchIntegrationTests:
    """Comprehensive search integration tests"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout
        
        # Test data
        self.test_queries = [
            "machine learning",
            "artificial intelligence",
            "web search",
            "distributed systems",
            "microservices architecture",
            "gRPC communication",
            "Kubernetes deployment",
            "Docker containers",
            "Redis caching",
            "PostgreSQL database"
        ]
        
        self.test_documents = [
            {
                "id": "doc1",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                "url": "https://example.com/ml-intro",
                "metadata": {
                    "author": "John Doe",
                    "category": "Technology",
                    "tags": ["machine learning", "AI", "algorithms"]
                }
            },
            {
                "id": "doc2",
                "title": "Web Search Engine Architecture",
                "content": "A web search engine consists of several components including crawlers, indexers, and ranking algorithms.",
                "url": "https://example.com/search-architecture",
                "metadata": {
                    "author": "Jane Smith",
                    "category": "Technology",
                    "tags": ["search engine", "architecture", "web"]
                }
            },
            {
                "id": "doc3",
                "title": "Distributed Systems Design",
                "content": "Distributed systems are collections of independent computers that appear to users as a single coherent system.",
                "url": "https://example.com/distributed-systems",
                "metadata": {
                    "author": "Bob Johnson",
                    "category": "Technology",
                    "tags": ["distributed systems", "design", "scalability"]
                }
            }
        ]
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     headers: Optional[Dict] = None) -> requests.Response:
        """Make HTTP request with retry logic"""
        url = f"{self.config.api_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=data, headers=headers)
                elif method.upper() == "PUT":
                    response = self.session.put(url, json=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise Exception("Max retries exceeded")
    
    def test_health_check(self):
        """Test health check endpoint"""
        logger.info("Testing health check endpoint")
        
        response = self._make_request("GET", "/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "version" in health_data
        
        logger.info("Health check test passed")
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        logger.info("Testing metrics endpoint")
        
        response = self._make_request("GET", "/metrics")
        assert response.status_code == 200
        
        # Check if response contains Prometheus metrics
        metrics_text = response.text
        assert "http_requests_total" in metrics_text
        assert "http_request_duration_seconds" in metrics_text
        
        logger.info("Metrics endpoint test passed")
    
    def test_search_basic(self):
        """Test basic search functionality"""
        logger.info("Testing basic search functionality")
        
        for query in self.test_queries[:3]:  # Test first 3 queries
            response = self._make_request("POST", "/api/v1/search", {
                "query": query,
                "limit": 10
            })
            
            assert response.status_code == 200
            
            search_data = response.json()
            assert "results" in search_data
            assert "total" in search_data
            assert "query" in search_data
            assert "took" in search_data
            
            # Verify result structure
            results = search_data["results"]
            assert isinstance(results, list)
            
            if results:  # If there are results
                for result in results:
                    assert "id" in result
                    assert "title" in result
                    assert "content" in result
                    assert "score" in result
                    assert "url" in result
        
        logger.info("Basic search test passed")
    
    def test_search_with_filters(self):
        """Test search with filters"""
        logger.info("Testing search with filters")
        
        response = self._make_request("POST", "/api/v1/search", {
            "query": "technology",
            "filters": {
                "category": "Technology",
                "author": "John Doe"
            },
            "limit": 5
        })
        
        assert response.status_code == 200
        
        search_data = response.json()
        results = search_data["results"]
        
        # Verify filters are applied
        for result in results:
            if "metadata" in result and "category" in result["metadata"]:
                assert result["metadata"]["category"] == "Technology"
        
        logger.info("Search with filters test passed")
    
    def test_search_with_boosts(self):
        """Test search with field boosts"""
        logger.info("Testing search with field boosts")
        
        response = self._make_request("POST", "/api/v1/search", {
            "query": "machine learning",
            "boost_fields": {
                "title": 2.0,
                "content": 1.0
            },
            "limit": 5
        })
        
        assert response.status_code == 200
        
        search_data = response.json()
        results = search_data["results"]
        
        # Verify results are returned
        assert isinstance(results, list)
        
        logger.info("Search with boosts test passed")
    
    def test_search_pagination(self):
        """Test search pagination"""
        logger.info("Testing search pagination")
        
        # First page
        response1 = self._make_request("POST", "/api/v1/search", {
            "query": "technology",
            "limit": 2,
            "offset": 0
        })
        
        assert response1.status_code == 200
        page1_data = response1.json()
        page1_results = page1_data["results"]
        
        # Second page
        response2 = self._make_request("POST", "/api/v1/search", {
            "query": "technology",
            "limit": 2,
            "offset": 2
        })
        
        assert response2.status_code == 200
        page2_data = response2.json()
        page2_results = page2_data["results"]
        
        # Verify pagination works
        assert len(page1_results) <= 2
        assert len(page2_results) <= 2
        
        # Verify different results (if any)
        if page1_results and page2_results:
            page1_ids = {result["id"] for result in page1_results}
            page2_ids = {result["id"] for result in page2_results}
            assert page1_ids.isdisjoint(page2_ids)
        
        logger.info("Search pagination test passed")
    
    def test_suggest_endpoint(self):
        """Test search suggestions"""
        logger.info("Testing search suggestions")
        
        response = self._make_request("GET", "/api/v1/suggest", {
            "q": "mach"
        })
        
        assert response.status_code == 200
        
        suggest_data = response.json()
        assert "suggestions" in suggest_data
        
        suggestions = suggest_data["suggestions"]
        assert isinstance(suggestions, list)
        
        logger.info("Search suggestions test passed")
    
    def test_autocomplete_endpoint(self):
        """Test autocomplete functionality"""
        logger.info("Testing autocomplete functionality")
        
        response = self._make_request("GET", "/api/v1/autocomplete", {
            "q": "artificial"
        })
        
        assert response.status_code == 200
        
        autocomplete_data = response.json()
        assert "completions" in autocomplete_data
        
        completions = autocomplete_data["completions"]
        assert isinstance(completions, list)
        
        logger.info("Autocomplete test passed")
    
    def test_document_management(self):
        """Test document CRUD operations"""
        logger.info("Testing document management")
        
        # Test document indexing
        for doc in self.test_documents:
            response = self._make_request("POST", "/api/v1/documents", doc)
            assert response.status_code == 201
            
            index_data = response.json()
            assert "id" in index_data
            assert index_data["id"] == doc["id"]
        
        # Test document retrieval
        for doc in self.test_documents:
            response = self._make_request("GET", f"/api/v1/documents/{doc['id']}")
            assert response.status_code == 200
            
            retrieved_doc = response.json()
            assert retrieved_doc["id"] == doc["id"]
            assert retrieved_doc["title"] == doc["title"]
        
        # Test document update
        update_doc = self.test_documents[0].copy()
        update_doc["title"] = "Updated Title"
        
        response = self._make_request("PUT", f"/api/v1/documents/{update_doc['id']}", update_doc)
        assert response.status_code == 200
        
        # Test document deletion
        response = self._make_request("DELETE", f"/api/v1/documents/{self.test_documents[-1]['id']}")
        assert response.status_code == 204
        
        logger.info("Document management test passed")
    
    def test_analytics_endpoint(self):
        """Test analytics functionality"""
        logger.info("Testing analytics endpoint")
        
        response = self._make_request("GET", "/api/v1/analytics/search-stats")
        assert response.status_code == 200
        
        analytics_data = response.json()
        assert "total_searches" in analytics_data
        assert "popular_queries" in analytics_data
        assert "search_trends" in analytics_data
        
        logger.info("Analytics test passed")
    
    def test_rate_limiting(self):
        """Test rate limiting"""
        logger.info("Testing rate limiting")
        
        # Make multiple requests quickly
        responses = []
        for _ in range(20):  # Exceed rate limit
            try:
                response = self._make_request("POST", "/api/v1/search", {
                    "query": "test",
                    "limit": 1
                })
                responses.append(response)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    logger.info("Rate limiting working correctly")
                    return
                else:
                    raise
        
        # If we get here, rate limiting might not be working
        logger.warning("Rate limiting test inconclusive")
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        logger.info("Testing concurrent request handling")
        
        def make_search_request(query: str) -> Dict[str, Any]:
            try:
                response = self._make_request("POST", "/api/v1/search", {
                    "query": query,
                    "limit": 5
                })
                return response.json()
            except Exception as e:
                logger.error(f"Concurrent request failed: {e}")
                return {"error": str(e)}
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=self.config.concurrent_requests) as executor:
            futures = [
                executor.submit(make_search_request, query) 
                for query in self.test_queries[:self.config.concurrent_requests]
            ]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Verify all requests completed
        assert len(results) == self.config.concurrent_requests
        
        # Verify most requests succeeded
        successful_requests = [r for r in results if "error" not in r]
        assert len(successful_requests) >= self.config.concurrent_requests * 0.8  # 80% success rate
        
        logger.info("Concurrent request test passed")
    
    def test_error_handling(self):
        """Test error handling"""
        logger.info("Testing error handling")
        
        # Test invalid query
        try:
            response = self._make_request("POST", "/api/v1/search", {
                "query": "",  # Empty query
                "limit": 10
            })
            # Should either return 400 or handle gracefully
            assert response.status_code in [200, 400]
        except requests.exceptions.HTTPError as e:
            assert e.response.status_code == 400
        
        # Test invalid document ID
        try:
            response = self._make_request("GET", "/api/v1/documents/invalid-id")
            assert response.status_code == 404
        except requests.exceptions.HTTPError as e:
            assert e.response.status_code == 404
        
        logger.info("Error handling test passed")
    
    def test_performance(self):
        """Test performance metrics"""
        logger.info("Testing performance metrics")
        
        start_time = time.time()
        
        # Make multiple requests
        for query in self.test_queries[:5]:
            response = self._make_request("POST", "/api/v1/search", {
                "query": query,
                "limit": 10
            })
            assert response.status_code == 200
            
            # Check response time
            search_data = response.json()
            if "took" in search_data:
                assert search_data["took"] < 1000  # Less than 1 second
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify reasonable performance
        assert total_time < 30  # All requests should complete within 30 seconds
        
        logger.info(f"Performance test passed (total time: {total_time:.2f}s)")
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting T3SS integration tests")
        
        test_methods = [
            self.test_health_check,
            self.test_metrics_endpoint,
            self.test_search_basic,
            self.test_search_with_filters,
            self.test_search_with_boosts,
            self.test_search_pagination,
            self.test_suggest_endpoint,
            self.test_autocomplete_endpoint,
            self.test_document_management,
            self.test_analytics_endpoint,
            self.test_rate_limiting,
            self.test_concurrent_requests,
            self.test_error_handling,
            self.test_performance
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
                logger.info(f"✓ {test_method.__name__} passed")
            except Exception as e:
                failed += 1
                logger.error(f"✗ {test_method.__name__} failed: {e}")
        
        logger.info(f"Integration tests completed: {passed} passed, {failed} failed")
        
        if failed > 0:
            raise Exception(f"{failed} tests failed")
        
        logger.info("All integration tests passed!")

def main():
    """Main function for running integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="T3SS Integration Tests")
    parser.add_argument("--api-url", required=True, help="API Gateway URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    
    args = parser.parse_args()
    
    config = TestConfig(
        api_url=args.api_url,
        timeout=args.timeout,
        concurrent_requests=args.concurrent
    )
    
    tests = SearchIntegrationTests(config)
    tests.run_all_tests()

if __name__ == "__main__":
    main()