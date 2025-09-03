# T3SS Project
# File: core/indexing/crawler/scheduler.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

@dataclass
class CrawlJob:
    """Represents a single crawl job with priority and metadata"""
    url: str
    priority: float = 1.0
    depth: int = 0
    parent_url: Optional[str] = None
    scheduled_time: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    domain: str = field(init=False)
    last_crawled: Optional[float] = None
    
    def __post_init__(self):
        self.domain = urlparse(self.url).netloc.lower()
    
    def __lt__(self, other):
        # Higher priority first, then earlier scheduled time
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.scheduled_time < other.scheduled_time

@dataclass
class DomainStats:
    """Tracks statistics for a domain"""
    requests_made: int = 0
    last_request_time: float = 0
    success_rate: float = 1.0
    average_response_time: float = 0.0
    robots_delay: float = 1.0
    max_concurrent: int = 1

class HighPerformanceScheduler:
    """
    Ultra-fast crawl scheduler with intelligent prioritization and domain management.
    Features:
    - Priority-based scheduling with heap
    - Domain-aware rate limiting
    - Adaptive crawling based on success rates
    - Memory-efficient job management
    - Real-time statistics tracking
    """
    
    def __init__(self, 
                 max_concurrent_domains: int = 1000,
                 max_jobs_per_domain: int = 10,
                 default_delay: float = 1.0,
                 max_queue_size: int = 1000000):
        self.max_concurrent_domains = max_concurrent_domains
        self.max_jobs_per_domain = max_jobs_per_domain
        self.default_delay = default_delay
        self.max_queue_size = max_queue_size
        
        # Core data structures
        self.job_queue: List[CrawlJob] = []  # Min-heap for priority scheduling
        self.domain_queues: Dict[str, deque] = defaultdict(deque)
        self.domain_stats: Dict[str, DomainStats] = defaultdict(DomainStats)
        self.active_domains: Set[str] = set()
        self.completed_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
        # Thread safety
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        
        # Statistics
        self.total_jobs_scheduled = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0
        self.start_time = time.time()
        
        # Performance optimization
        self._url_cache = set()  # Prevent duplicate URLs
        self._domain_priorities = defaultdict(float)  # Dynamic domain prioritization
        
    def add_job(self, job: CrawlJob) -> bool:
        """Add a crawl job to the scheduler with intelligent deduplication"""
        with self.lock:
            # Check queue size limit
            if len(self.job_queue) >= self.max_queue_size:
                logger.warning(f"Queue size limit reached ({self.max_queue_size})")
                return False
            
            # Deduplication
            if job.url in self.completed_urls or job.url in self.failed_urls:
                return False
            
            if job.url in self._url_cache:
                return False
            
            # Add to cache
            self._url_cache.add(job.url)
            
            # Add to priority queue
            heapq.heappush(self.job_queue, job)
            self.domain_queues[job.domain].append(job)
            self.total_jobs_scheduled += 1
            
            # Update domain priority based on content quality signals
            self._update_domain_priority(job.domain, job.priority)
            
            # Notify waiting threads
            self.condition.notify()
            return True
    
    def add_urls(self, urls: List[str], priority: float = 1.0, depth: int = 0) -> int:
        """Bulk add URLs with the same priority and depth"""
        added_count = 0
        for url in urls:
            job = CrawlJob(url=url, priority=priority, depth=depth)
            if self.add_job(job):
                added_count += 1
        return added_count
    
    def get_next_job(self, timeout: float = 1.0) -> Optional[CrawlJob]:
        """Get the next job to crawl with domain-aware scheduling"""
        with self.condition:
            start_time = time.time()
            
            while True:
                # Check if we have jobs
                if not self.job_queue:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        return None
                    self.condition.wait(remaining_time)
                    continue
                
                # Find the best job considering domain constraints
                best_job = None
                current_time = time.time()
                
                # Try to find a job from a domain that's ready
                temp_queue = []
                
                while self.job_queue:
                    job = heapq.heappop(self.job_queue)
                    temp_queue.append(job)
                    
                    domain_stats = self.domain_stats[job.domain]
                    
                    # Check if domain is ready (respects delay)
                    time_since_last = current_time - domain_stats.last_request_time
                    if time_since_last >= domain_stats.robots_delay:
                        # Check if we haven't exceeded domain limits
                        if (len(self.active_domains) < self.max_concurrent_domains or 
                            job.domain in self.active_domains):
                            best_job = job
                            break
                
                # Put remaining jobs back
                for job in temp_queue:
                    if job != best_job:
                        heapq.heappush(self.job_queue, job)
                
                if best_job:
                    # Mark domain as active
                    self.active_domains.add(best_job.domain)
                    self.domain_stats[best_job.domain].last_request_time = current_time
                    self.domain_stats[best_job.domain].requests_made += 1
                    
                    # Remove from domain queue
                    if best_job in self.domain_queues[best_job.domain]:
                        self.domain_queues[best_job.domain].remove(best_job)
                    
                    return best_job
                
                # No job ready, wait a bit
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    return None
                self.condition.wait(min(0.1, remaining_time))
    
    def complete_job(self, job: CrawlJob, success: bool, response_time: float = 0.0):
        """Mark a job as completed and update domain statistics"""
        with self.lock:
            # Remove from active domains
            self.active_domains.discard(job.domain)
            
            # Update domain statistics
            domain_stats = self.domain_stats[job.domain]
            if success:
                self.completed_urls.add(job.url)
                self.total_jobs_completed += 1
                
                # Update success rate using exponential moving average
                domain_stats.success_rate = (domain_stats.success_rate * 0.9 + 0.1)
                
                # Update average response time
                if domain_stats.average_response_time == 0:
                    domain_stats.average_response_time = response_time
                else:
                    domain_stats.average_response_time = (
                        domain_stats.average_response_time * 0.9 + response_time * 0.1
                    )
            else:
                self.failed_urls.add(job.url)
                self.total_jobs_failed += 1
                
                # Update success rate
                domain_stats.success_rate = (domain_stats.success_rate * 0.9)
                
                # Retry logic
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.scheduled_time = time.time() + (2 ** job.retry_count)  # Exponential backoff
                    heapq.heappush(self.job_queue, job)
                    self.domain_queues[job.domain].append(job)
    
    def update_domain_delay(self, domain: str, delay: float):
        """Update the crawl delay for a specific domain"""
        with self.lock:
            self.domain_stats[domain].robots_delay = delay
    
    def _update_domain_priority(self, domain: str, priority: float):
        """Update domain priority based on content quality"""
        # Use exponential moving average for domain priority
        if domain in self._domain_priorities:
            self._domain_priorities[domain] = (
                self._domain_priorities[domain] * 0.9 + priority * 0.1
            )
        else:
            self._domain_priorities[domain] = priority
    
    def get_statistics(self) -> Dict:
        """Get comprehensive scheduler statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            return {
                "uptime_seconds": uptime,
                "total_jobs_scheduled": self.total_jobs_scheduled,
                "total_jobs_completed": self.total_jobs_completed,
                "total_jobs_failed": self.total_jobs_failed,
                "queue_size": len(self.job_queue),
                "active_domains": len(self.active_domains),
                "completed_urls": len(self.completed_urls),
                "failed_urls": len(self.failed_urls),
                "jobs_per_second": self.total_jobs_completed / uptime if uptime > 0 else 0,
                "success_rate": (
                    self.total_jobs_completed / 
                    (self.total_jobs_completed + self.total_jobs_failed)
                    if (self.total_jobs_completed + self.total_jobs_failed) > 0 else 0
                ),
                "top_domains": sorted(
                    self.domain_stats.items(),
                    key=lambda x: x[1].requests_made,
                    reverse=True
                )[:10]
            }
    
    def clear_completed_urls(self, max_age_seconds: float = 3600):
        """Clear old completed URLs to free memory"""
        with self.lock:
            current_time = time.time()
            # This is a simplified version - in production, you'd track timestamps
            if len(self.completed_urls) > 100000:  # Arbitrary limit
                # Keep only recent URLs (simplified)
                self.completed_urls.clear()
                self._url_cache.clear()
    
    def export_state(self) -> str:
        """Export scheduler state for persistence"""
        with self.lock:
            state = {
                "domain_stats": {
                    domain: {
                        "requests_made": stats.requests_made,
                        "success_rate": stats.success_rate,
                        "average_response_time": stats.average_response_time,
                        "robots_delay": stats.robots_delay
                    }
                    for domain, stats in self.domain_stats.items()
                },
                "domain_priorities": dict(self._domain_priorities),
                "statistics": self.get_statistics()
            }
            return json.dumps(state, indent=2)
    
    def import_state(self, state_json: str):
        """Import scheduler state from persistence"""
        try:
            state = json.loads(state_json)
            with self.lock:
                # Restore domain stats
                for domain, stats_data in state.get("domain_stats", {}).items():
                    self.domain_stats[domain] = DomainStats(**stats_data)
                
                # Restore domain priorities
                self._domain_priorities.update(state.get("domain_priorities", {}))
        except Exception as e:
            logger.error(f"Failed to import state: {e}")