# T3SS Project
# File: core/querying/query_logging/anomaly_flagger.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import redis
import time
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    QUERY_FREQUENCY = "query_frequency"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    PATTERN_DEVIATION = "pattern_deviation"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"
    BEHAVIORAL_CHANGE = "behavioral_change"

class SeverityLevel(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    # Detection thresholds
    frequency_threshold: float = 3.0  # Standard deviations
    response_time_threshold: float = 2.5
    error_rate_threshold: float = 2.0
    resource_threshold: float = 2.5
    
    # Time windows
    short_window: int = 300  # 5 minutes
    medium_window: int = 3600  # 1 hour
    long_window: int = 86400  # 24 hours
    
    # Model parameters
    isolation_forest_contamination: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown: int = 300  # 5 minutes
    max_alerts_per_hour: int = 100

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_id: str
    timestamp: datetime
    response_time: float
    status_code: int
    user_id: Optional[str]
    query_type: str
    result_count: int
    cpu_usage: float
    memory_usage: float
    cache_hit: bool
    error_message: Optional[str] = None

@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    timestamp: datetime
    query_id: str
    description: str
    confidence: float
    metrics: Dict[str, Any]
    baseline_value: float
    actual_value: float
    deviation: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyStats:
    """Statistics for anomaly detection"""
    total_anomalies: int = 0
    anomalies_by_type: Dict[AnomalyType, int] = field(default_factory=dict)
    anomalies_by_severity: Dict[SeverityLevel, int] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    detection_accuracy: float = 0.0
    average_processing_time: float = 0.0

class StatisticalAnalyzer:
    """Statistical analysis for anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def detect_outliers_zscore(self, data: List[float], threshold: float = 3.0) -> List[bool]:
        """Detect outliers using Z-score method"""
        if len(data) < 3:
            return [False] * len(data)
        
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold
    
    def detect_outliers_iqr(self, data: List[float]) -> List[bool]:
        """Detect outliers using IQR method"""
        if len(data) < 4:
            return [False] * len(data)
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return [(x < lower_bound or x > upper_bound) for x in data]
    
    def detect_outliers_isolation_forest(self, data: List[List[float]], contamination: float = 0.1) -> List[bool]:
        """Detect outliers using Isolation Forest"""
        if len(data) < 10:
            return [False] * len(data)
        
        try:
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(data)
            return predictions == -1
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
            return [False] * len(data)
    
    def calculate_rolling_statistics(self, data: List[float], window: int = 10) -> Dict[str, List[float]]:
        """Calculate rolling statistics for time series data"""
        if len(data) < window:
            return {"mean": data, "std": [0.0] * len(data), "min": data, "max": data}
        
        df = pd.Series(data)
        rolling = df.rolling(window=window, min_periods=1)
        
        return {
            "mean": rolling.mean().tolist(),
            "std": rolling.std().fillna(0).tolist(),
            "min": rolling.min().tolist(),
            "max": rolling.max().tolist()
        }

class PatternAnalyzer:
    """Pattern analysis for behavioral anomaly detection"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.cluster_models = {}
    
    def extract_query_patterns(self, queries: List[QueryMetrics]) -> Dict[str, Any]:
        """Extract patterns from query data"""
        patterns = {
            "hourly_distribution": self._get_hourly_distribution(queries),
            "daily_distribution": self._get_daily_distribution(queries),
            "query_type_distribution": self._get_query_type_distribution(queries),
            "response_time_patterns": self._get_response_time_patterns(queries),
            "user_behavior": self._get_user_behavior_patterns(queries)
        }
        return patterns
    
    def _get_hourly_distribution(self, queries: List[QueryMetrics]) -> Dict[int, int]:
        """Get hourly distribution of queries"""
        hourly = defaultdict(int)
        for query in queries:
            hour = query.timestamp.hour
            hourly[hour] += 1
        return dict(hourly)
    
    def _get_daily_distribution(self, queries: List[QueryMetrics]) -> Dict[int, int]:
        """Get daily distribution of queries"""
        daily = defaultdict(int)
        for query in queries:
            weekday = query.timestamp.weekday()
            daily[weekday] += 1
        return dict(daily)
    
    def _get_query_type_distribution(self, queries: List[QueryMetrics]) -> Dict[str, int]:
        """Get query type distribution"""
        types = defaultdict(int)
        for query in queries:
            types[query.query_type] += 1
        return dict(types)
    
    def _get_response_time_patterns(self, queries: List[QueryMetrics]) -> Dict[str, float]:
        """Get response time patterns"""
        response_times = [q.response_time for q in queries]
        if not response_times:
            return {}
        
        return {
            "mean": np.mean(response_times),
            "median": np.median(response_times),
            "std": np.std(response_times),
            "p95": np.percentile(response_times, 95),
            "p99": np.percentile(response_times, 99)
        }
    
    def _get_user_behavior_patterns(self, queries: List[QueryMetrics]) -> Dict[str, Any]:
        """Get user behavior patterns"""
        user_queries = defaultdict(list)
        for query in queries:
            if query.user_id:
                user_queries[query.user_id].append(query)
        
        patterns = {}
        for user_id, user_query_list in user_queries.items():
            if len(user_query_list) < 5:  # Need minimum queries for pattern analysis
                continue
            
            response_times = [q.response_time for q in user_query_list]
            query_types = [q.query_type for q in user_query_list]
            
            patterns[user_id] = {
                "avg_response_time": np.mean(response_times),
                "query_frequency": len(user_query_list),
                "preferred_query_types": dict(pd.Series(query_types).value_counts().head(3))
            }
        
        return patterns
    
    def detect_pattern_deviations(self, current_patterns: Dict[str, Any], 
                                historical_patterns: Dict[str, Any]) -> List[AnomalyType]:
        """Detect deviations from historical patterns"""
        deviations = []
        
        # Check hourly distribution deviation
        if "hourly_distribution" in current_patterns and "hourly_distribution" in historical_patterns:
            current_hourly = current_patterns["hourly_distribution"]
            historical_hourly = historical_patterns["hourly_distribution"]
            
            if self._is_distribution_deviant(current_hourly, historical_hourly):
                deviations.append(AnomalyType.PATTERN_DEVIATION)
        
        # Check response time pattern deviation
        if "response_time_patterns" in current_patterns and "response_time_patterns" in historical_patterns:
            current_rt = current_patterns["response_time_patterns"]
            historical_rt = historical_patterns["response_time_patterns"]
            
            if abs(current_rt.get("mean", 0) - historical_rt.get("mean", 0)) > historical_rt.get("std", 1) * 2:
                deviations.append(AnomalyType.PATTERN_DEVIATION)
        
        return deviations
    
    def _is_distribution_deviant(self, current: Dict[int, int], historical: Dict[int, int]) -> bool:
        """Check if current distribution deviates significantly from historical"""
        if not current or not historical:
            return False
        
        # Normalize distributions
        current_total = sum(current.values())
        historical_total = sum(historical.values())
        
        if current_total == 0 or historical_total == 0:
            return False
        
        current_norm = {k: v/current_total for k, v in current.items()}
        historical_norm = {k: v/historical_total for k, v in historical.items()}
        
        # Calculate chi-square test
        all_keys = set(current_norm.keys()) | set(historical_norm.keys())
        observed = [current_norm.get(k, 0) for k in all_keys]
        expected = [historical_norm.get(k, 0) for k in all_keys]
        
        if sum(expected) == 0:
            return False
        
        chi2, p_value = stats.chisquare(observed, expected)
        return p_value < 0.05  # Significant deviation

class AnomalyFlagger:
    """Advanced anomaly detection system for query logging"""
    
    def __init__(self, config: AnomalyConfig = None):
        self.config = config or AnomalyConfig()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        
        # Data storage
        self.query_history = deque(maxlen=10000)  # Keep last 10k queries
        self.anomaly_history = deque(maxlen=1000)  # Keep last 1k anomalies
        self.baseline_metrics = {}
        self.pattern_baselines = {}
        
        # Redis connection for caching
        self.redis_client = None
        if self.config.enable_caching:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
        
        # Statistics
        self.stats = AnomalyStats()
        self.alert_cooldowns = {}  # Track alert cooldowns
        self.alert_counts = defaultdict(int)  # Track alert frequency
        
        # Initialize baselines
        self._initialize_baselines()
    
    def _initialize_baselines(self):
        """Initialize baseline metrics from historical data"""
        # This would typically load from a database or file
        self.baseline_metrics = {
            "response_time": {"mean": 100.0, "std": 50.0, "p95": 200.0, "p99": 500.0},
            "error_rate": {"mean": 0.01, "std": 0.005},
            "query_frequency": {"mean": 100.0, "std": 20.0},
            "cpu_usage": {"mean": 50.0, "std": 15.0},
            "memory_usage": {"mean": 60.0, "std": 10.0}
        }
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key"""
        return f"anomaly:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(key, self.config.cache_ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def add_query_metrics(self, metrics: QueryMetrics):
        """Add query metrics for analysis"""
        self.query_history.append(metrics)
        
        # Trigger real-time anomaly detection
        asyncio.create_task(self._detect_realtime_anomalies(metrics))
    
    async def _detect_realtime_anomalies(self, metrics: QueryMetrics):
        """Detect anomalies in real-time"""
        anomalies = []
        
        # Check response time anomaly
        if self._is_response_time_anomaly(metrics):
            anomaly = self._create_anomaly(
                AnomalyType.RESPONSE_TIME,
                metrics,
                f"Response time {metrics.response_time}ms exceeds threshold",
                self.baseline_metrics["response_time"]["p95"],
                metrics.response_time
            )
            anomalies.append(anomaly)
        
        # Check error rate anomaly
        if self._is_error_anomaly(metrics):
            anomaly = self._create_anomaly(
                AnomalyType.ERROR_RATE,
                metrics,
                f"Error detected: {metrics.error_message}",
                0.0,
                1.0
            )
            anomalies.append(anomaly)
        
        # Check resource usage anomaly
        if self._is_resource_anomaly(metrics):
            anomaly = self._create_anomaly(
                AnomalyType.RESOURCE_USAGE,
                metrics,
                f"High resource usage: CPU {metrics.cpu_usage}%, Memory {metrics.memory_usage}%",
                self.baseline_metrics["cpu_usage"]["mean"],
                metrics.cpu_usage
            )
            anomalies.append(anomaly)
        
        # Process detected anomalies
        for anomaly in anomalies:
            await self._process_anomaly(anomaly)
    
    def _is_response_time_anomaly(self, metrics: QueryMetrics) -> bool:
        """Check if response time is anomalous"""
        baseline = self.baseline_metrics["response_time"]
        threshold = baseline["p95"] + (baseline["std"] * self.config.response_time_threshold)
        return metrics.response_time > threshold
    
    def _is_error_anomaly(self, metrics: QueryMetrics) -> bool:
        """Check if error is anomalous"""
        return metrics.status_code >= 400 or metrics.error_message is not None
    
    def _is_resource_anomaly(self, metrics: QueryMetrics) -> bool:
        """Check if resource usage is anomalous"""
        cpu_baseline = self.baseline_metrics["cpu_usage"]
        memory_baseline = self.baseline_metrics["memory_usage"]
        
        cpu_threshold = cpu_baseline["mean"] + (cpu_baseline["std"] * self.config.resource_threshold)
        memory_threshold = memory_baseline["mean"] + (memory_baseline["std"] * self.config.resource_threshold)
        
        return (metrics.cpu_usage > cpu_threshold or 
                metrics.memory_usage > memory_threshold)
    
    def _create_anomaly(self, anomaly_type: AnomalyType, metrics: QueryMetrics, 
                       description: str, baseline_value: float, actual_value: float) -> Anomaly:
        """Create an anomaly object"""
        anomaly_id = hashlib.md5(f"{metrics.query_id}_{anomaly_type.value}_{metrics.timestamp.isoformat()}".encode()).hexdigest()
        
        deviation = abs(actual_value - baseline_value) / baseline_value if baseline_value != 0 else 0
        severity = self._calculate_severity(deviation, anomaly_type)
        confidence = min(1.0, deviation / 2.0)  # Simple confidence calculation
        
        return Anomaly(
            anomaly_id=anomaly_id,
            anomaly_type=anomaly_type,
            severity=severity,
            timestamp=metrics.timestamp,
            query_id=metrics.query_id,
            description=description,
            confidence=confidence,
            metrics={
                "response_time": metrics.response_time,
                "status_code": metrics.status_code,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "result_count": metrics.result_count
            },
            baseline_value=baseline_value,
            actual_value=actual_value,
            deviation=deviation,
            context={
                "user_id": metrics.user_id,
                "query_type": metrics.query_type,
                "cache_hit": metrics.cache_hit
            }
        )
    
    def _calculate_severity(self, deviation: float, anomaly_type: AnomalyType) -> SeverityLevel:
        """Calculate severity level based on deviation"""
        if deviation > 5.0:
            return SeverityLevel.CRITICAL
        elif deviation > 3.0:
            return SeverityLevel.HIGH
        elif deviation > 2.0:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    async def _process_anomaly(self, anomaly: Anomaly):
        """Process a detected anomaly"""
        # Add to anomaly history
        self.anomaly_history.append(anomaly)
        
        # Update statistics
        self.stats.total_anomalies += 1
        self.stats.anomalies_by_type[anomaly.anomaly_type] = \
            self.stats.anomalies_by_type.get(anomaly.anomaly_type, 0) + 1
        self.stats.anomalies_by_severity[anomaly.severity] = \
            self.stats.anomalies_by_severity.get(anomaly.severity, 0) + 1
        
        # Send alert if enabled
        if self.config.enable_alerts:
            await self._send_alert(anomaly)
        
        # Log anomaly
        logger.warning(f"Anomaly detected: {anomaly.anomaly_type.value} - {anomaly.description}")
    
    async def _send_alert(self, anomaly: Anomaly):
        """Send alert for anomaly"""
        # Check cooldown
        alert_key = f"{anomaly.anomaly_type.value}_{anomaly.severity.value}"
        now = time.time()
        
        if alert_key in self.alert_cooldowns:
            if now - self.alert_cooldowns[alert_key] < self.config.alert_cooldown:
                return
        
        # Check rate limiting
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        if self.alert_counts[hour_key] >= self.config.max_alerts_per_hour:
            logger.warning("Alert rate limit exceeded")
            return
        
        # Send alert (placeholder - would integrate with actual alerting system)
        alert_data = {
            "anomaly_id": anomaly.anomaly_id,
            "type": anomaly.anomaly_type.value,
            "severity": anomaly.severity.value,
            "timestamp": anomaly.timestamp.isoformat(),
            "description": anomaly.description,
            "confidence": anomaly.confidence,
            "metrics": anomaly.metrics
        }
        
        logger.critical(f"ALERT: {json.dumps(alert_data, indent=2)}")
        
        # Update cooldown and count
        self.alert_cooldowns[alert_key] = now
        self.alert_counts[hour_key] += 1
    
    async def detect_batch_anomalies(self, queries: List[QueryMetrics]) -> List[Anomaly]:
        """Detect anomalies in a batch of queries"""
        if not queries:
            return []
        
        anomalies = []
        
        # Extract features for analysis
        features = self._extract_features(queries)
        
        # Statistical anomaly detection
        statistical_anomalies = await self._detect_statistical_anomalies(features, queries)
        anomalies.extend(statistical_anomalies)
        
        # Pattern-based anomaly detection
        pattern_anomalies = await self._detect_pattern_anomalies(queries)
        anomalies.extend(pattern_anomalies)
        
        # Seasonal anomaly detection
        seasonal_anomalies = await self._detect_seasonal_anomalies(queries)
        anomalies.extend(seasonal_anomalies)
        
        return anomalies
    
    def _extract_features(self, queries: List[QueryMetrics]) -> List[List[float]]:
        """Extract features for machine learning analysis"""
        features = []
        
        for query in queries:
            feature_vector = [
                query.response_time,
                query.status_code,
                query.result_count,
                query.cpu_usage,
                query.memory_usage,
                float(query.cache_hit),
                query.timestamp.hour,
                query.timestamp.weekday()
            ]
            features.append(feature_vector)
        
        return features
    
    async def _detect_statistical_anomalies(self, features: List[List[float]], 
                                          queries: List[QueryMetrics]) -> List[Anomaly]:
        """Detect statistical anomalies using machine learning"""
        if len(features) < 10:
            return []
        
        anomalies = []
        
        try:
            # Use Isolation Forest for outlier detection
            outlier_mask = self.statistical_analyzer.detect_outliers_isolation_forest(
                features, self.config.isolation_forest_contamination
            )
            
            for i, is_outlier in enumerate(outlier_mask):
                if is_outlier and i < len(queries):
                    query = queries[i]
                    anomaly = self._create_anomaly(
                        AnomalyType.STATISTICAL_OUTLIER,
                        query,
                        f"Statistical outlier detected in query pattern",
                        np.mean([f[i] for f in features]),
                        features[i][0]  # Use response time as primary metric
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_pattern_anomalies(self, queries: List[QueryMetrics]) -> List[Anomaly]:
        """Detect pattern-based anomalies"""
        if len(queries) < 50:  # Need sufficient data for pattern analysis
            return []
        
        anomalies = []
        
        try:
            # Extract current patterns
            current_patterns = self.pattern_analyzer.extract_query_patterns(queries)
            
            # Get historical patterns (would typically come from database)
            historical_patterns = self.pattern_baselines
            
            # Detect pattern deviations
            deviations = self.pattern_analyzer.detect_pattern_deviations(
                current_patterns, historical_patterns
            )
            
            for deviation_type in deviations:
                # Create anomaly for pattern deviation
                representative_query = queries[len(queries) // 2]  # Use middle query as representative
                anomaly = self._create_anomaly(
                    deviation_type,
                    representative_query,
                    f"Pattern deviation detected: {deviation_type.value}",
                    0.0,  # No specific baseline for pattern deviations
                    1.0   # Pattern deviation detected
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_seasonal_anomalies(self, queries: List[QueryMetrics]) -> List[Anomaly]:
        """Detect seasonal anomalies"""
        if len(queries) < 100:  # Need sufficient data for seasonal analysis
            return []
        
        anomalies = []
        
        try:
            # Group queries by hour
            hourly_queries = defaultdict(list)
            for query in queries:
                hourly_queries[query.timestamp.hour].append(query)
            
            # Check for unusual patterns in each hour
            for hour, hour_queries in hourly_queries.items():
                if len(hour_queries) < 5:
                    continue
                
                response_times = [q.response_time for q in hour_queries]
                avg_response_time = np.mean(response_times)
                
                # Compare with historical average for this hour
                historical_key = f"hour_{hour}_response_time"
                if historical_key in self.baseline_metrics:
                    historical_avg = self.baseline_metrics[historical_key]["mean"]
                    historical_std = self.baseline_metrics[historical_key]["std"]
                    
                    if abs(avg_response_time - historical_avg) > historical_std * 2:
                        # Create anomaly for seasonal deviation
                        representative_query = hour_queries[0]
                        anomaly = self._create_anomaly(
                            AnomalyType.SEASONAL_ANOMALY,
                            representative_query,
                            f"Seasonal anomaly detected at hour {hour}",
                            historical_avg,
                            avg_response_time
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Seasonal anomaly detection failed: {e}")
        
        return anomalies
    
    def update_baselines(self, queries: List[QueryMetrics]):
        """Update baseline metrics from recent data"""
        if not queries:
            return
        
        # Update response time baselines
        response_times = [q.response_time for q in queries]
        self.baseline_metrics["response_time"] = {
            "mean": np.mean(response_times),
            "std": np.std(response_times),
            "p95": np.percentile(response_times, 95),
            "p99": np.percentile(response_times, 99)
        }
        
        # Update error rate baselines
        error_count = sum(1 for q in queries if q.status_code >= 400)
        error_rate = error_count / len(queries)
        self.baseline_metrics["error_rate"] = {
            "mean": error_rate,
            "std": np.std([1 if q.status_code >= 400 else 0 for q in queries])
        }
        
        # Update resource usage baselines
        cpu_usage = [q.cpu_usage for q in queries]
        memory_usage = [q.memory_usage for q in queries]
        
        self.baseline_metrics["cpu_usage"] = {
            "mean": np.mean(cpu_usage),
            "std": np.std(cpu_usage)
        }
        
        self.baseline_metrics["memory_usage"] = {
            "mean": np.mean(memory_usage),
            "std": np.std(memory_usage)
        }
        
        # Update hourly baselines
        hourly_response_times = defaultdict(list)
        for query in queries:
            hourly_response_times[query.timestamp.hour].append(query.response_time)
        
        for hour, times in hourly_response_times.items():
            if len(times) > 5:  # Need minimum data points
                self.baseline_metrics[f"hour_{hour}_response_time"] = {
                    "mean": np.mean(times),
                    "std": np.std(times)
                }
        
        # Update pattern baselines
        self.pattern_baselines = self.pattern_analyzer.extract_query_patterns(queries)
    
    def get_anomaly_stats(self) -> AnomalyStats:
        """Get anomaly detection statistics"""
        return self.stats
    
    def get_recent_anomalies(self, limit: int = 100) -> List[Anomaly]:
        """Get recent anomalies"""
        return list(self.anomaly_history)[-limit:]
    
    def get_anomalies_by_type(self, anomaly_type: AnomalyType) -> List[Anomaly]:
        """Get anomalies by type"""
        return [a for a in self.anomaly_history if a.anomaly_type == anomaly_type]
    
    def get_anomalies_by_severity(self, severity: SeverityLevel) -> List[Anomaly]:
        """Get anomalies by severity"""
        return [a for a in self.anomaly_history if a.severity == severity]
    
    def clear_history(self):
        """Clear anomaly history"""
        self.anomaly_history.clear()
        self.query_history.clear()
        self.stats = AnomalyStats()

# Example usage
async def main():
    """Example usage of the anomaly flagger"""
    config = AnomalyConfig(
        enable_alerts=True,
        enable_caching=True
    )
    
    flagger = AnomalyFlagger(config)
    
    # Simulate some query metrics
    queries = []
    base_time = datetime.now()
    
    for i in range(100):
        # Normal queries
        query = QueryMetrics(
            query_id=f"query_{i}",
            timestamp=base_time + timedelta(minutes=i),
            response_time=100 + np.random.normal(0, 20),
            status_code=200,
            user_id=f"user_{i % 10}",
            query_type="SELECT",
            result_count=np.random.randint(1, 100),
            cpu_usage=50 + np.random.normal(0, 10),
            memory_usage=60 + np.random.normal(0, 5),
            cache_hit=np.random.choice([True, False])
        )
        queries.append(query)
    
    # Add some anomalous queries
    for i in range(10):
        query = QueryMetrics(
            query_id=f"anomaly_query_{i}",
            timestamp=base_time + timedelta(minutes=100 + i),
            response_time=500 + np.random.normal(0, 100),  # High response time
            status_code=500 if i % 3 == 0 else 200,  # Some errors
            user_id=f"user_{i % 10}",
            query_type="SELECT",
            result_count=np.random.randint(1, 100),
            cpu_usage=90 + np.random.normal(0, 5),  # High CPU
            memory_usage=85 + np.random.normal(0, 5),  # High memory
            cache_hit=False,
            error_message="Database timeout" if i % 3 == 0 else None
        )
        queries.append(query)
    
    # Add queries to flagger
    for query in queries:
        flagger.add_query_metrics(query)
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Detect batch anomalies
    batch_anomalies = await flagger.detect_batch_anomalies(queries)
    
    print(f"Detected {len(batch_anomalies)} anomalies in batch analysis")
    
    # Print recent anomalies
    recent_anomalies = flagger.get_recent_anomalies(10)
    print(f"\nRecent anomalies:")
    for anomaly in recent_anomalies:
        print(f"- {anomaly.anomaly_type.value}: {anomaly.description} (severity: {anomaly.severity.value})")
    
    # Print statistics
    stats = flagger.get_anomaly_stats()
    print(f"\nAnomaly Statistics:")
    print(f"Total anomalies: {stats.total_anomalies}")
    print(f"By type: {dict(stats.anomalies_by_type)}")
    print(f"By severity: {dict(stats.anomalies_by_severity)}")

if __name__ == "__main__":
    asyncio.run(main())
