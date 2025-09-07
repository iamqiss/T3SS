# T3SS Project
# File: core/indexing/crawler/async_crawl_jobs/worker_orchestrator.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import weakref
import gc

# Async processing
from asyncio import Queue, Semaphore, Event, Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Distributed computing
import redis.asyncio as redis
from aioredis import Redis

# Monitoring and metrics
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"

class WorkerType(Enum):
    """Types of workers"""
    CRAWLER = "crawler"
    PARSER = "parser"
    INDEXER = "indexer"
    ANALYZER = "analyzer"
    VALIDATOR = "validator"

class Priority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class JobConfig:
    """Configuration for a job"""
    job_id: str
    job_type: str
    priority: Priority = Priority.NORMAL
    timeout: int = 300  # seconds
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    dependencies: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

@dataclass
class JobResult:
    """Result of job execution"""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    completed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerInfo:
    """Information about a worker"""
    worker_id: str
    worker_type: WorkerType
    status: str = "idle"
    current_job: Optional[str] = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    total_execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    capabilities: Set[str] = field(default_factory=set)
    max_concurrent_jobs: int = 1
    current_jobs: int = 0

class JobQueue:
    """High-performance job queue with priority support"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            Priority.CRITICAL: Queue(maxsize=max_size // 4),
            Priority.HIGH: Queue(maxsize=max_size // 4),
            Priority.NORMAL: Queue(maxsize=max_size // 2),
            Priority.LOW: Queue(maxsize=max_size // 4)
        }
        self.job_registry: Dict[str, JobConfig] = {}
        self.lock = Lock()
        
    async def enqueue(self, job_config: JobConfig) -> bool:
        """Enqueue a job with priority"""
        async with self.lock:
            if len(self.job_registry) >= self.max_size:
                return False
            
            self.job_registry[job_config.job_id] = job_config
            
            if job_config.scheduled_at and job_config.scheduled_at > datetime.utcnow():
                asyncio.create_task(self._schedule_job(job_config))
                return True
            
            try:
                await self.queues[job_config.priority].put(job_config)
                return True
            except asyncio.QueueFull:
                return False
    
    async def dequeue(self, worker_type: WorkerType) -> Optional[JobConfig]:
        """Dequeue a job for a specific worker type"""
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            try:
                job_config = self.queues[priority].get_nowait()
                if self._is_job_compatible(job_config, worker_type):
                    return job_config
                else:
                    await self.queues[priority].put(job_config)
            except asyncio.QueueEmpty:
                continue
        return None
    
    def _is_job_compatible(self, job_config: JobConfig, worker_type: WorkerType) -> bool:
        """Check if job is compatible with worker type"""
        return True
    
    async def _schedule_job(self, job_config: JobConfig):
        """Schedule a job for later execution"""
        delay = (job_config.scheduled_at - datetime.utcnow()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        await self.enqueue(job_config)
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a job"""
        async with self.lock:
            if job_id in self.job_registry:
                return JobStatus.PENDING
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        async with self.lock:
            if job_id in self.job_registry:
                del self.job_registry[job_id]
                return True
        return False

class WorkerPool:
    """Manages a pool of workers for job execution"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_semaphore = Semaphore(max_workers)
        self.worker_lock = Lock()
        self.metrics = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'active_workers': 0,
            'queue_size': 0
        }
    
    async def register_worker(self, worker_info: WorkerInfo) -> bool:
        """Register a new worker"""
        async with self.worker_lock:
            if len(self.workers) >= self.max_workers:
                return False
            
            self.workers[worker_info.worker_id] = worker_info
            self.metrics['active_workers'] = len(self.workers)
            return True
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker"""
        async with self.worker_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                self.metrics['active_workers'] = len(self.workers)
                return True
        return False
    
    async def get_available_worker(self, worker_type: WorkerType) -> Optional[WorkerInfo]:
        """Get an available worker of specific type"""
        async with self.worker_lock:
            for worker in self.workers.values():
                if (worker.worker_type == worker_type and 
                    worker.status == "idle" and 
                    worker.current_jobs < worker.max_concurrent_jobs):
                    return worker
        return None
    
    async def update_worker_status(self, worker_id: str, status: str, 
                                 current_job: Optional[str] = None,
                                 memory_usage: int = 0,
                                 cpu_usage: float = 0.0):
        """Update worker status and metrics"""
        async with self.worker_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.status = status
                worker.current_job = current_job
                worker.memory_usage = memory_usage
                worker.cpu_usage = cpu_usage
                worker.last_heartbeat = datetime.utcnow()

class JobExecutor:
    """Executes jobs with monitoring and error handling"""
    
    def __init__(self, worker_pool: WorkerPool, job_queue: JobQueue):
        self.worker_pool = worker_pool
        self.job_queue = job_queue
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_results: Dict[str, JobResult] = {}
        self.execution_lock = Lock()
        
        # Metrics
        self.job_counter = Counter('jobs_total', 'Total jobs processed', ['status'])
        self.job_duration = Histogram('job_duration_seconds', 'Job execution time')
        self.active_jobs = Gauge('active_jobs', 'Currently active jobs')
        
    async def execute_job(self, job_config: JobConfig, worker_info: WorkerInfo) -> JobResult:
        """Execute a single job"""
        start_time = time.time()
        job_id = job_config.job_id
        
        try:
            await self.worker_pool.update_worker_status(
                worker_info.worker_id, "running", job_id
            )
            
            result = JobResult(job_id=job_id, status=JobStatus.RUNNING)
            
            try:
                job_result = await asyncio.wait_for(
                    self._run_job(job_config, worker_info),
                    timeout=job_config.timeout
                )
                result.result = job_result
                result.status = JobStatus.COMPLETED
                
            except asyncio.TimeoutError:
                result.status = JobStatus.TIMEOUT
                result.error = f"Job timed out after {job_config.timeout} seconds"
                
            except Exception as e:
                result.status = JobStatus.FAILED
                result.error = str(e)
                logger.error(f"Job {job_id} failed: {e}")
            
            result.execution_time = time.time() - start_time
            result.completed_at = datetime.utcnow()
            
            process = psutil.Process()
            result.memory_usage = process.memory_info().rss
            result.cpu_usage = process.cpu_percent()
            
            self.job_counter.labels(status=result.status.value).inc()
            self.job_duration.observe(result.execution_time)
            
            return result
            
        finally:
            await self.worker_pool.update_worker_status(
                worker_info.worker_id, "idle", None
            )
            
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    async def _run_job(self, job_config: JobConfig, worker_info: WorkerInfo) -> Any:
        """Run the actual job logic"""
        job_type = job_config.job_type
        worker_type = worker_info.worker_type
        
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            'job_id': job_config.job_id,
            'job_type': job_type,
            'worker_type': worker_type.value,
            'execution_time': time.time(),
            'status': 'completed'
        }
    
    async def submit_job(self, job_config: JobConfig) -> str:
        """Submit a job for execution"""
        success = await self.job_queue.enqueue(job_config)
        if not success:
            raise Exception("Failed to enqueue job - queue full")
        return job_config.job_id
    
    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get result of a completed job"""
        return self.job_results.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
            return True
        return await self.job_queue.cancel_job(job_id)

class WorkerOrchestrator:
    """Main orchestrator for managing workers and job execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.job_queue = JobQueue(config.get('max_queue_size', 10000))
        self.worker_pool = WorkerPool(config.get('max_workers', 10))
        self.job_executor = JobExecutor(self.worker_pool, self.job_queue)
        
        self.redis_client: Optional[Redis] = None
        self.redis_config = config.get('redis', {})
        
        self.monitoring_enabled = config.get('monitoring', True)
        self.metrics_port = config.get('metrics_port', 8000)
        
        self.running = False
        self.shutdown_event = Event()
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def start(self):
        """Start the orchestrator"""
        logger.info("Starting Worker Orchestrator")
        
        if self.redis_config:
            await self._init_redis()
        
        if self.monitoring_enabled:
            await self._start_monitoring()
        
        self.running = True
        self._start_background_tasks()
        
        logger.info("Worker Orchestrator started successfully")
    
    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping Worker Orchestrator")
        
        self.running = False
        self.shutdown_event.set()
        
        for task in self.running_jobs.values():
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Worker Orchestrator stopped")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.from_url(
                self.redis_config.get('url', 'redis://localhost:6379'),
                encoding='utf-8',
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def _start_monitoring(self):
        """Start monitoring server"""
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Monitoring server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start monitoring server: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        task = asyncio.create_task(self._job_dispatcher())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        task = asyncio.create_task(self._worker_health_checker())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        task = asyncio.create_task(self._metrics_collector())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def _job_dispatcher(self):
        """Background task to dispatch jobs to workers"""
        while self.running:
            try:
                for worker_type in WorkerType:
                    worker = await self.worker_pool.get_available_worker(worker_type)
                    if worker:
                        job_config = await self.job_queue.dequeue(worker_type)
                        if job_config:
                            task = asyncio.create_task(
                                self.job_executor.execute_job(job_config, worker)
                            )
                            self.running_jobs[job_config.job_id] = task
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in job dispatcher: {e}")
                await asyncio.sleep(1)
    
    async def _worker_health_checker(self):
        """Background task to check worker health"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                unhealthy_workers = []
                
                for worker_id, worker in self.worker_pool.workers.items():
                    if (current_time - worker.last_heartbeat).total_seconds() > 30:
                        unhealthy_workers.append(worker_id)
                
                for worker_id in unhealthy_workers:
                    await self.worker_pool.unregister_worker(worker_id)
                    logger.warning(f"Removed unhealthy worker: {worker_id}")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in worker health checker: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collector(self):
        """Background task to collect and report metrics"""
        while self.running:
            try:
                self.active_jobs.set(len(self.running_jobs))
                
                worker_metrics = await self.worker_pool.get_worker_metrics()
                
                logger.info(f"Active jobs: {len(self.running_jobs)}, "
                          f"Workers: {worker_metrics['active_workers']}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(30)
    
    async def submit_job(self, job_type: str, priority: Priority = Priority.NORMAL,
                        timeout: int = 300, max_retries: int = 3,
                        dependencies: List[str] = None,
                        resources: Dict[str, Any] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """Submit a new job"""
        job_id = str(uuid.uuid4())
        
        job_config = JobConfig(
            job_id=job_id,
            job_type=job_type,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            dependencies=dependencies or [],
            resources=resources or {},
            metadata=metadata or {}
        )
        
        return await self.job_executor.submit_job(job_config)
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a job"""
        if job_id in self.running_jobs:
            return JobStatus.RUNNING
        
        result = await self.job_executor.get_job_result(job_id)
        if result:
            return result.status
        
        return await self.job_queue.get_job_status(job_id)
    
    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get result of a completed job"""
        return await self.job_executor.get_job_result(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        return await self.job_executor.cancel_job(job_id)
    
    async def register_worker(self, worker_id: str, worker_type: WorkerType,
                            capabilities: Set[str] = None,
                            max_concurrent_jobs: int = 1) -> bool:
        """Register a new worker"""
        worker_info = WorkerInfo(
            worker_id=worker_id,
            worker_type=worker_type,
            capabilities=capabilities or set(),
            max_concurrent_jobs=max_concurrent_jobs
        )
        
        return await self.worker_pool.register_worker(worker_info)
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker"""
        return await self.worker_pool.unregister_worker(worker_id)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        worker_metrics = await self.worker_pool.get_worker_metrics()
        
        return {
            'orchestrator': {
                'running': self.running,
                'active_jobs': len(self.running_jobs),
                'background_tasks': len(self.background_tasks)
            },
            'workers': worker_metrics,
            'queue': {
                'size': len(self.job_queue.job_registry)
            }
        }

# Factory functions
def create_orchestrator(config: Dict[str, Any]) -> WorkerOrchestrator:
    """Create a new worker orchestrator"""
    return WorkerOrchestrator(config)

async def create_and_start_orchestrator(config: Dict[str, Any]) -> WorkerOrchestrator:
    """Create and start a worker orchestrator"""
    orchestrator = create_orchestrator(config)
    await orchestrator.start()
    return orchestrator

# Example usage
async def main():
    """Example usage of the worker orchestrator"""
    config = {
        'max_workers': 5,
        'max_queue_size': 1000,
        'monitoring': True,
        'metrics_port': 8000,
        'redis': {
            'url': 'redis://localhost:6379'
        }
    }
    
    orchestrator = await create_and_start_orchestrator(config)
    
    try:
        await orchestrator.register_worker("worker-1", WorkerType.CRAWLER)
        await orchestrator.register_worker("worker-2", WorkerType.PARSER)
        await orchestrator.register_worker("worker-3", WorkerType.INDEXER)
        
        job_ids = []
        for i in range(10):
            job_id = await orchestrator.submit_job(
                job_type="crawl",
                priority=Priority.NORMAL,
                metadata={"url": f"https://example.com/page{i}"}
            )
            job_ids.append(job_id)
        
        for job_id in job_ids:
            status = await orchestrator.get_job_status(job_id)
            print(f"Job {job_id}: {status}")
        
        await asyncio.sleep(5)
        
        for job_id in job_ids:
            result = await orchestrator.get_job_result(job_id)
            if result:
                print(f"Job {job_id} result: {result.status}")
        
        metrics = await orchestrator.get_metrics()
        print(f"Orchestrator metrics: {metrics}")
        
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())
