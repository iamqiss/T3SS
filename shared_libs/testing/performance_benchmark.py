# T3SS Project
# File: shared_libs/testing/performance_benchmark.py
# (c) 2025 Qiss Labs. All Rights Reserved.

import asyncio
import aiohttp
import time
import statistics
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import psutil
import threading
import queue
import signal
import sys

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    base_url: str = "http://localhost:8080"
    api_key: str = "test-api-key-123"
    max_concurrent_requests: int = 100
    test_duration_seconds: int = 300
    ramp_up_seconds: int = 60
    ramp_down_seconds: int = 30
    request_timeout: int = 30
    enable_ssl: bool = False
    enable_compression: bool = True
    user_agents: List[str] = field(default_factory=lambda: [
        "T3SS-Benchmark/1.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ])
    test_queries: List[str] = field(default_factory=lambda: [
        "machine learning algorithms",
        "artificial intelligence",
        "deep learning neural networks",
        "natural language processing",
        "computer vision",
        "data science",
        "python programming",
        "javascript development",
        "web development",
        "mobile app development"
    ])

@dataclass
class RequestResult:
    """Result of a single request"""
    timestamp: float
    method: str
    url: str
    status_code: int
    response_time: float
    response_size: int
    error: Optional[str] = None
    thread_id: int = 0

@dataclass
class BenchmarkResult:
    """Results of a benchmark test"""
    test_name: str
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_response_time: float
    min_response_time: float
    max_response_time: float
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    throughput_mbps: float
    error_rate: float
    status_codes: Dict[int, int] = field(default_factory=dict)
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class SystemMonitor:
    """Monitor system resources during benchmarks"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                # Network I/O
                network = psutil.net_io_counters()
                
                # Process info
                process = psutil.Process()
                process_cpu = process.cpu_percent()
                process_memory = process.memory_info().rss / (1024**2)  # MB
                
                metric = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_used_gb': memory_used_gb,
                    'disk_percent': disk_percent,
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv,
                    'process_cpu_percent': process_cpu,
                    'process_memory_mb': process_memory
                }
                
                self.metrics.append(metric)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.interval)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics"""
        return self.metrics.copy()

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.system_monitor = SystemMonitor()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        if self.session:
            asyncio.create_task(self.session.close())
        sys.exit(0)
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            limit_per_host=self.config.max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'X-API-Key': self.config.api_key,
                'Accept-Encoding': 'gzip, deflate' if self.config.enable_compression else 'identity',
                'User-Agent': self.config.user_agents[0]
            }
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def run_search_benchmark(self) -> BenchmarkResult:
        """Run search API benchmark"""
        logger.info("Starting search API benchmark")
        
        # Start system monitoring
        self.system_monitor.start()
        
        start_time = time.time()
        results_queue = queue.Queue()
        
        # Create tasks for concurrent requests
        tasks = []
        for i in range(self.config.max_concurrent_requests):
            task = asyncio.create_task(
                self._search_worker(results_queue, i)
            )
            tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(self.config.test_duration_seconds)
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Stop system monitoring
        self.system_monitor.stop()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Calculate benchmark result
        benchmark_result = self._calculate_benchmark_result(
            "Search API Benchmark",
            start_time,
            end_time,
            results
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    async def _search_worker(self, results_queue: queue.Queue, worker_id: int):
        """Worker for search requests"""
        try:
            while True:
                # Select random query
                query = np.random.choice(self.config.test_queries)
                
                # Make search request
                result = await self._make_search_request(query, worker_id)
                results_queue.put(result)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Search worker {worker_id} error: {e}")
    
    async def _make_search_request(self, query: str, worker_id: int) -> RequestResult:
        """Make a single search request"""
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}/api/v1/search"
            payload = {
                "query": query,
                "page": 1,
                "page_size": 10
            }
            
            async with self.session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                response_data = await response.read()
                
                return RequestResult(
                    timestamp=start_time,
                    method="POST",
                    url=url,
                    status_code=response.status,
                    response_time=response_time,
                    response_size=len(response_data),
                    thread_id=worker_id
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                timestamp=start_time,
                method="POST",
                url=url,
                status_code=0,
                response_time=response_time,
                response_size=0,
                error=str(e),
                thread_id=worker_id
            )
    
    async def run_load_test(self, target_rps: int) -> BenchmarkResult:
        """Run load test with target requests per second"""
        logger.info(f"Starting load test with target RPS: {target_rps}")
        
        self.system_monitor.start()
        start_time = time.time()
        results_queue = queue.Queue()
        
        # Calculate request interval
        request_interval = 1.0 / target_rps
        
        # Create tasks
        tasks = []
        for i in range(self.config.max_concurrent_requests):
            task = asyncio.create_task(
                self._load_test_worker(results_queue, i, request_interval)
            )
            tasks.append(task)
        
        # Run for test duration
        await asyncio.sleep(self.config.test_duration_seconds)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        self.system_monitor.stop()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        benchmark_result = self._calculate_benchmark_result(
            f"Load Test ({target_rps} RPS)",
            start_time,
            end_time,
            results
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    async def _load_test_worker(self, results_queue: queue.Queue, worker_id: int, interval: float):
        """Worker for load testing"""
        try:
            while True:
                query = np.random.choice(self.config.test_queries)
                result = await self._make_search_request(query, worker_id)
                results_queue.put(result)
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Load test worker {worker_id} error: {e}")
    
    async def run_stress_test(self) -> BenchmarkResult:
        """Run stress test to find breaking point"""
        logger.info("Starting stress test")
        
        # Gradually increase load until system breaks
        target_rps_values = [10, 50, 100, 200, 500, 1000, 2000]
        stress_results = []
        
        for target_rps in target_rps_values:
            logger.info(f"Testing {target_rps} RPS")
            
            result = await self.run_load_test(target_rps)
            stress_results.append(result)
            
            # Check if system is still responsive
            if result.error_rate > 0.1:  # 10% error rate threshold
                logger.warning(f"High error rate at {target_rps} RPS: {result.error_rate:.2%}")
                break
            
            # Wait between tests
            await asyncio.sleep(30)
        
        # Return the last successful result
        return stress_results[-1] if stress_results else None
    
    async def run_endurance_test(self) -> BenchmarkResult:
        """Run endurance test for extended period"""
        logger.info("Starting endurance test")
        
        # Run for longer duration
        original_duration = self.config.test_duration_seconds
        self.config.test_duration_seconds = 3600  # 1 hour
        
        result = await self.run_search_benchmark()
        
        # Restore original duration
        self.config.test_duration_seconds = original_duration
        
        return result
    
    def _calculate_benchmark_result(
        self, 
        test_name: str, 
        start_time: float, 
        end_time: float, 
        results: List[RequestResult]
    ) -> BenchmarkResult:
        """Calculate benchmark statistics"""
        
        if not results:
            return BenchmarkResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_response_time=0,
                min_response_time=0,
                max_response_time=0,
                avg_response_time=0,
                median_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                throughput_mbps=0,
                error_rate=0
            )
        
        # Filter successful requests
        successful_results = [r for r in results if r.status_code == 200]
        failed_results = [r for r in results if r.status_code != 200]
        
        # Response times
        response_times = [r.response_time for r in successful_results]
        
        # Status codes
        status_codes = {}
        for result in results:
            status_codes[result.status_code] = status_codes.get(result.status_code, 0) + 1
        
        # Calculate statistics
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        
        if response_times:
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            min_response_time = max_response_time = avg_response_time = 0
            median_response_time = p95_response_time = p99_response_time = 0
        
        # Throughput
        test_duration = end_time - start_time
        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        
        # Calculate throughput in MB/s
        total_bytes = sum(r.response_size for r in results)
        throughput_mbps = (total_bytes / (1024 * 1024)) / test_duration if test_duration > 0 else 0
        
        # Error rate
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        return BenchmarkResult(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_response_time=sum(response_times),
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            avg_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            throughput_mbps=throughput_mbps,
            error_rate=error_rate,
            status_codes=status_codes,
            response_times=response_times,
            errors=[r.error for r in failed_results if r.error]
        )
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive benchmark report"""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        report = {
            "benchmark_summary": {
                "total_tests": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "base_url": self.config.base_url,
                    "max_concurrent_requests": self.config.max_concurrent_requests,
                    "test_duration_seconds": self.config.test_duration_seconds
                }
            },
            "results": []
        }
        
        for result in self.results:
            result_dict = {
                "test_name": result.test_name,
                "duration_seconds": result.end_time - result.start_time,
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "requests_per_second": result.requests_per_second,
                "throughput_mbps": result.throughput_mbps,
                "error_rate": result.error_rate,
                "response_times": {
                    "min_ms": result.min_response_time * 1000,
                    "max_ms": result.max_response_time * 1000,
                    "avg_ms": result.avg_response_time * 1000,
                    "median_ms": result.median_response_time * 1000,
                    "p95_ms": result.p95_response_time * 1000,
                    "p99_ms": result.p99_response_time * 1000
                },
                "status_codes": result.status_codes
            }
            report["results"].append(result_dict)
        
        # Add system metrics
        system_metrics = self.system_monitor.get_metrics()
        if system_metrics:
            report["system_metrics"] = {
                "avg_cpu_percent": statistics.mean([m['cpu_percent'] for m in system_metrics]),
                "max_cpu_percent": max([m['cpu_percent'] for m in system_metrics]),
                "avg_memory_percent": statistics.mean([m['memory_percent'] for m in system_metrics]),
                "max_memory_percent": max([m['memory_percent'] for m in system_metrics]),
                "avg_memory_used_gb": statistics.mean([m['memory_used_gb'] for m in system_metrics]),
                "max_memory_used_gb": max([m['memory_used_gb'] for m in system_metrics])
            }
        
        # Save report
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Benchmark report saved to {output_file}")
        else:
            print(json.dumps(report, indent=2))
    
    def plot_results(self, output_dir: str = "benchmark_plots"):
        """Generate performance plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for result in self.results:
            if not result.response_times:
                continue
            
            # Response time distribution
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(result.response_times, bins=50, alpha=0.7)
            plt.title(f"{result.test_name} - Response Time Distribution")
            plt.xlabel("Response Time (seconds)")
            plt.ylabel("Frequency")
            
            # Response time over time
            plt.subplot(2, 2, 2)
            plt.plot(result.response_times)
            plt.title(f"{result.test_name} - Response Time Over Time")
            plt.xlabel("Request Number")
            plt.ylabel("Response Time (seconds)")
            
            # Status codes
            plt.subplot(2, 2, 3)
            status_codes = list(result.status_codes.keys())
            counts = list(result.status_codes.values())
            plt.bar([str(code) for code in status_codes], counts)
            plt.title(f"{result.test_name} - Status Codes")
            plt.xlabel("Status Code")
            plt.ylabel("Count")
            
            # Performance metrics
            plt.subplot(2, 2, 4)
            metrics = ['RPS', 'Avg RT (ms)', 'P95 RT (ms)', 'Error Rate (%)']
            values = [
                result.requests_per_second,
                result.avg_response_time * 1000,
                result.p95_response_time * 1000,
                result.error_rate * 100
            ]
            plt.bar(metrics, values)
            plt.title(f"{result.test_name} - Key Metrics")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{result.test_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Performance plots saved to {output_dir}")

async def main():
    """Main benchmark execution"""
    config = BenchmarkConfig(
        base_url="http://localhost:8080",
        max_concurrent_requests=50,
        test_duration_seconds=60,
        ramp_up_seconds=10,
        ramp_down_seconds=5
    )
    
    async with PerformanceBenchmark(config) as benchmark:
        # Run different types of benchmarks
        logger.info("Running comprehensive performance benchmarks")
        
        # Search benchmark
        search_result = await benchmark.run_search_benchmark()
        logger.info(f"Search benchmark completed: {search_result.requests_per_second:.2f} RPS")
        
        # Load test
        load_result = await benchmark.run_load_test(target_rps=100)
        logger.info(f"Load test completed: {load_result.requests_per_second:.2f} RPS")
        
        # Generate report
        benchmark.generate_report("benchmark_report.json")
        benchmark.plot_results()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())