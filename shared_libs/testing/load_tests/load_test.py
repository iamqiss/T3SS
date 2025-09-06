#!/usr/bin/env python3
"""
T3SS Load Testing Suite
(c) 2025 Qiss Labs. All Rights Reserved.
"""

import asyncio
import aiohttp
import time
import json
import logging
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    api_url: str
    duration: int = 300  # 5 minutes
    concurrent_users: int = 100
    requests_per_second: int = 50
    ramp_up_time: int = 60  # 1 minute
    test_queries: List[str] = field(default_factory=lambda: [
        "machine learning",
        "artificial intelligence",
        "web search",
        "distributed systems",
        "microservices",
        "gRPC",
        "Kubernetes",
        "Docker",
        "Redis",
        "PostgreSQL"
    ])

@dataclass
class RequestResult:
    """Result of a single request"""
    success: bool
    status_code: int
    response_time: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class LoadTestResults:
    """Results of load testing"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    response_times: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second"""
        if self.total_duration == 0:
            return 0.0
        return self.total_requests / self.total_duration
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def median_response_time(self) -> float:
        """Calculate median response time"""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time"""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
    
    @property
    def p99_response_time(self) -> float:
        """Calculate 99th percentile response time"""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile

class LoadTester:
    """Load testing implementation"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = LoadTestResults()
        self.start_time = None
        self.end_time = None
        
    async def make_search_request(self, session: aiohttp.ClientSession, query: str) -> RequestResult:
        """Make a single search request"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.config.api_url}/api/v1/search",
                json={"query": query, "limit": 10},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                # Read response body
                await response.text()
                
                return RequestResult(
                    success=response.status == 200,
                    status_code=response.status,
                    response_time=response_time
                )
                
        except asyncio.TimeoutError:
            return RequestResult(
                success=False,
                status_code=0,
                response_time=time.time() - start_time,
                error="Timeout"
            )
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def make_health_request(self, session: aiohttp.ClientSession) -> RequestResult:
        """Make a health check request"""
        start_time = time.time()
        
        try:
            async with session.get(
                f"{self.config.api_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_time = time.time() - start_time
                
                await response.text()
                
                return RequestResult(
                    success=response.status == 200,
                    status_code=response.status,
                    response_time=response_time
                )
                
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def make_metrics_request(self, session: aiohttp.ClientSession) -> RequestResult:
        """Make a metrics request"""
        start_time = time.time()
        
        try:
            async with session.get(
                f"{self.config.api_url}/metrics",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_time = time.time() - start_time
                
                await response.text()
                
                return RequestResult(
                    success=response.status == 200,
                    status_code=response.status,
                    response_time=response_time
                )
                
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def worker(self, session: aiohttp.ClientSession, worker_id: int) -> List[RequestResult]:
        """Worker coroutine for making requests"""
        results = []
        query_index = 0
        
        while time.time() - self.start_time < self.config.duration:
            # Select query
            query = self.config.test_queries[query_index % len(self.config.test_queries)]
            query_index += 1
            
            # Make request
            result = await self.make_search_request(session, query)
            results.append(result)
            
            # Update results
            self.results.total_requests += 1
            if result.success:
                self.results.successful_requests += 1
            else:
                self.results.failed_requests += 1
                if result.error:
                    self.results.errors.append(result.error)
            
            self.results.response_times.append(result.response_time)
            self.results.status_codes[result.status_code] = self.results.status_codes.get(result.status_code, 0) + 1
            
            # Rate limiting
            await asyncio.sleep(1.0 / self.config.requests_per_second)
        
        return results
    
    async def run_load_test(self):
        """Run the load test"""
        logger.info(f"Starting load test for {self.config.duration} seconds")
        logger.info(f"Concurrent users: {self.config.concurrent_users}")
        logger.info(f"Target RPS: {self.config.requests_per_second}")
        
        self.start_time = time.time()
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create worker tasks
            tasks = []
            for i in range(self.config.concurrent_users):
                task = asyncio.create_task(self.worker(session, i))
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
        
        self.end_time = time.time()
        self.results.total_duration = self.end_time - self.start_time
        
        logger.info("Load test completed")
    
    def print_results(self):
        """Print load test results"""
        print("\n" + "="*60)
        print("T3SS LOAD TEST RESULTS")
        print("="*60)
        
        print(f"Total Duration: {self.results.total_duration:.2f} seconds")
        print(f"Total Requests: {self.results.total_requests}")
        print(f"Successful Requests: {self.results.successful_requests}")
        print(f"Failed Requests: {self.results.failed_requests}")
        print(f"Success Rate: {self.results.success_rate:.2f}%")
        print(f"Requests per Second: {self.results.requests_per_second:.2f}")
        
        print(f"\nResponse Times:")
        print(f"  Average: {self.results.average_response_time:.3f}s")
        print(f"  Median: {self.results.median_response_time:.3f}s")
        print(f"  95th Percentile: {self.results.p95_response_time:.3f}s")
        print(f"  99th Percentile: {self.results.p99_response_time:.3f}s")
        
        print(f"\nStatus Codes:")
        for status_code, count in sorted(self.results.status_codes.items()):
            print(f"  {status_code}: {count}")
        
        if self.results.errors:
            print(f"\nErrors ({len(self.results.errors)}):")
            error_counts = {}
            for error in self.results.errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error}: {count}")
        
        print("="*60)
    
    def save_results(self, filename: str):
        """Save results to JSON file"""
        results_data = {
            "config": {
                "api_url": self.config.api_url,
                "duration": self.config.duration,
                "concurrent_users": self.config.concurrent_users,
                "requests_per_second": self.config.requests_per_second
            },
            "results": {
                "total_requests": self.results.total_requests,
                "successful_requests": self.results.successful_requests,
                "failed_requests": self.results.failed_requests,
                "success_rate": self.results.success_rate,
                "requests_per_second": self.results.requests_per_second,
                "total_duration": self.results.total_duration,
                "average_response_time": self.results.average_response_time,
                "median_response_time": self.results.median_response_time,
                "p95_response_time": self.results.p95_response_time,
                "p99_response_time": self.results.p99_response_time,
                "status_codes": self.results.status_codes,
                "errors": self.results.errors
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

class StressTester:
    """Stress testing implementation"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = LoadTestResults()
        self.start_time = None
        self.end_time = None
    
    async def stress_test_worker(self, session: aiohttp.ClientSession, worker_id: int, 
                                target_rps: float) -> List[RequestResult]:
        """Stress test worker that tries to maintain target RPS"""
        results = []
        query_index = 0
        last_request_time = 0
        
        while time.time() - self.start_time < self.config.duration:
            current_time = time.time()
            
            # Calculate if we should make a request now
            if current_time - last_request_time >= 1.0 / target_rps:
                query = self.config.test_queries[query_index % len(self.config.test_queries)]
                query_index += 1
                
                # Make request
                result = await self.make_search_request(session, query)
                results.append(result)
                
                # Update results
                self.results.total_requests += 1
                if result.success:
                    self.results.successful_requests += 1
                else:
                    self.results.failed_requests += 1
                    if result.error:
                        self.results.errors.append(result.error)
                
                self.results.response_times.append(result.response_time)
                self.results.status_codes[result.status_code] = self.results.status_codes.get(result.status_code, 0) + 1
                
                last_request_time = current_time
            else:
                # Wait a bit before next request
                await asyncio.sleep(0.001)
        
        return results
    
    async def make_search_request(self, session: aiohttp.ClientSession, query: str) -> RequestResult:
        """Make a single search request"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.config.api_url}/api/v1/search",
                json={"query": query, "limit": 10},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                await response.text()
                
                return RequestResult(
                    success=response.status == 200,
                    status_code=response.status,
                    response_time=response_time
                )
                
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def run_stress_test(self, max_rps: int = 1000):
        """Run stress test to find maximum sustainable RPS"""
        logger.info(f"Starting stress test up to {max_rps} RPS")
        
        self.start_time = time.time()
        
        # Test different RPS levels
        rps_levels = [50, 100, 200, 500, 1000]
        best_rps = 0
        
        for target_rps in rps_levels:
            if target_rps > max_rps:
                break
                
            logger.info(f"Testing {target_rps} RPS")
            
            # Reset results for this level
            self.results = LoadTestResults()
            test_start = time.time()
            
            # Run test for 60 seconds
            test_duration = 60
            connector = aiohttp.TCPConnector(limit=1000, limit_per_host=100)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = []
                for i in range(min(self.config.concurrent_users, target_rps)):
                    task = asyncio.create_task(
                        self.stress_test_worker(session, i, target_rps / self.config.concurrent_users)
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            test_end = time.time()
            actual_duration = test_end - test_start
            
            # Calculate actual RPS
            actual_rps = self.results.total_requests / actual_duration
            success_rate = self.results.success_rate
            
            logger.info(f"Target RPS: {target_rps}, Actual RPS: {actual_rps:.2f}, Success Rate: {success_rate:.2f}%")
            
            # If success rate is good, update best RPS
            if success_rate >= 95.0 and actual_rps >= target_rps * 0.9:
                best_rps = target_rps
            else:
                logger.info(f"Breaking at {target_rps} RPS due to poor performance")
                break
        
        self.end_time = time.time()
        self.results.total_duration = self.end_time - self.start_time
        
        logger.info(f"Maximum sustainable RPS: {best_rps}")
        return best_rps

def main():
    """Main function for running load tests"""
    parser = argparse.ArgumentParser(description="T3SS Load Testing Suite")
    parser.add_argument("--api-url", required=True, help="API Gateway URL")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--concurrent", type=int, default=100, help="Concurrent users")
    parser.add_argument("--rps", type=int, default=50, help="Target requests per second")
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        api_url=args.api_url,
        duration=args.duration,
        concurrent_users=args.concurrent,
        requests_per_second=args.rps
    )
    
    if args.stress:
        # Run stress test
        stress_tester = StressTester(config)
        max_rps = asyncio.run(stress_tester.run_stress_test())
        print(f"Maximum sustainable RPS: {max_rps}")
    else:
        # Run load test
        load_tester = LoadTester(config)
        asyncio.run(load_tester.run_load_test())
        load_tester.print_results()
        
        if args.output:
            load_tester.save_results(args.output)

if __name__ == "__main__":
    main()