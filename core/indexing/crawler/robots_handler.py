# T3SS Project
# File: core/indexing/crawler/robots_handler.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import aiohttp
import re
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from collections import defaultdict
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RobotsRule:
    """Represents a single robots.txt rule"""
    user_agent: str
    allow_patterns: List[str] = field(default_factory=list)
    disallow_patterns: List[str] = field(default_factory=list)
    crawl_delay: Optional[float] = None
    sitemap_urls: List[str] = field(default_factory=list)
    host: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class RobotsCache:
    """Caches robots.txt rules for domains"""
    domain: str
    rules: List[RobotsRule]
    last_fetched: datetime
    expires_at: datetime
    content_hash: str
    fetch_errors: int = 0
    is_valid: bool = True

@dataclass
class CrawlRequest:
    """Represents a crawl request to be checked against robots.txt"""
    url: str
    user_agent: str
    referer: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class AdvancedRobotsHandler:
    """
    Ultra-advanced robots.txt handler with intelligent caching, parsing, and compliance.
    
    Features:
    - Multi-user-agent support with fallback rules
    - Intelligent caching with TTL and invalidation
    - Advanced pattern matching with wildcards and regex
    - Sitemap discovery and parsing
    - Rate limiting and politeness enforcement
    - Error handling and retry logic
    - Performance optimization with async operations
    - Comprehensive logging and statistics
    """
    
    def __init__(self, 
                 cache_ttl: int = 3600,  # 1 hour
                 max_cache_size: int = 10000,
                 request_timeout: int = 10,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_sitemap_discovery: bool = True,
                 enable_advanced_patterns: bool = True):
        
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_sitemap_discovery = enable_sitemap_discovery
        self.enable_advanced_patterns = enable_advanced_patterns
        
        # Cache management
        self.robots_cache: Dict[str, RobotsCache] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'robots_fetched': 0,
            'fetch_errors': 0,
            'allowed_requests': 0,
            'blocked_requests': 0,
            'sitemaps_discovered': 0,
            'average_fetch_time': 0.0,
            'total_fetch_time': 0.0
        }
        
        # HTTP session for efficient connection reuse
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Pattern compilation cache
        self.pattern_cache: Dict[str, re.Pattern] = {}
        
        # Common user agents for fallback
        self.common_user_agents = [
            '*',  # Wildcard - matches all
            'Googlebot',
            'Bingbot',
            'Slurp',
            'DuckDuckBot',
            'Baiduspider',
            'YandexBot',
            'facebookexternalhit',
            'Twitterbot',
            'LinkedInBot'
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'T3SS-RobotsHandler/1.0 (+https://example.com/bot)',
                'Accept': 'text/plain, text/html, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def can_crawl(self, request: CrawlRequest) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if a URL can be crawled according to robots.txt rules.
        
        Returns:
            Tuple of (can_crawl, reason, crawl_delay)
        """
        self.stats['total_requests'] += 1
        
        try:
            parsed_url = urlparse(request.url)
            domain = parsed_url.netloc.lower()
            
            # Get robots.txt rules for domain
            robots_cache = await self.get_robots_for_domain(domain, request.user_agent)
            
            if not robots_cache or not robots_cache.is_valid:
                # If we can't get robots.txt, be conservative but allow
                logger.warning(f"Could not fetch robots.txt for {domain}, allowing crawl")
                self.stats['allowed_requests'] += 1
                return True, "No robots.txt available", None
            
            # Check rules for the specific user agent
            applicable_rules = self.get_applicable_rules(robots_cache.rules, request.user_agent)
            
            if not applicable_rules:
                # No specific rules for this user agent, allow
                self.stats['allowed_requests'] += 1
                return True, "No rules for user agent", None
            
            # Check if URL is allowed
            can_crawl, reason = self.check_url_against_rules(request.url, applicable_rules)
            
            if can_crawl:
                self.stats['allowed_requests'] += 1
                # Get crawl delay from rules
                crawl_delay = self.get_crawl_delay(applicable_rules)
                return True, reason, crawl_delay
            else:
                self.stats['blocked_requests'] += 1
                return False, reason, None
                
        except Exception as e:
            logger.error(f"Error checking robots.txt for {request.url}: {e}")
            # On error, be conservative but allow
            self.stats['allowed_requests'] += 1
            return True, f"Error checking robots.txt: {e}", None
    
    async def get_robots_for_domain(self, domain: str, user_agent: str) -> Optional[RobotsCache]:
        """Get robots.txt rules for a domain with intelligent caching"""
        async with self.cache_lock:
            # Check cache first
            if domain in self.robots_cache:
                cache_entry = self.robots_cache[domain]
                
                # Check if cache is still valid
                if datetime.now() < cache_entry.expires_at:
                    self.stats['cache_hits'] += 1
                    self.cache_access_times[domain] = time.time()
                    return cache_entry
                else:
                    # Cache expired, remove it
                    del self.robots_cache[domain]
                    if domain in self.cache_access_times:
                        del self.cache_access_times[domain]
            
            self.stats['cache_misses'] += 1
            
            # Fetch robots.txt
            robots_cache = await self.fetch_robots_txt(domain, user_agent)
            
            if robots_cache:
                # Add to cache
                await self.add_to_cache(domain, robots_cache)
                return robots_cache
            
            return None
    
    async def fetch_robots_txt(self, domain: str, user_agent: str) -> Optional[RobotsCache]:
        """Fetch and parse robots.txt for a domain"""
        start_time = time.time()
        
        try:
            # Try HTTPS first, then HTTP
            robots_urls = [
                f"https://{domain}/robots.txt",
                f"http://{domain}/robots.txt"
            ]
            
            for robots_url in robots_urls:
                try:
                    async with self.session.get(robots_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            content_hash = hashlib.md5(content.encode()).hexdigest()
                            
                            # Parse robots.txt content
                            rules = self.parse_robots_txt(content, domain)
                            
                            fetch_time = time.time() - start_time
                            self.stats['robots_fetched'] += 1
                            self.stats['total_fetch_time'] += fetch_time
                            self.stats['average_fetch_time'] = (
                                self.stats['total_fetch_time'] / self.stats['robots_fetched']
                            )
                            
                            # Discover sitemaps if enabled
                            sitemap_urls = []
                            if self.enable_sitemap_discovery:
                                sitemap_urls = self.extract_sitemap_urls(content)
                                self.stats['sitemaps_discovered'] += len(sitemap_urls)
                            
                            return RobotsCache(
                                domain=domain,
                                rules=rules,
                                last_fetched=datetime.now(),
                                expires_at=datetime.now() + timedelta(seconds=self.cache_ttl),
                                content_hash=content_hash,
                                is_valid=True
                            )
                        
                        elif response.status == 404:
                            # No robots.txt file - this is allowed
                            logger.info(f"No robots.txt found for {domain}")
                            return RobotsCache(
                                domain=domain,
                                rules=[],
                                last_fetched=datetime.now(),
                                expires_at=datetime.now() + timedelta(seconds=self.cache_ttl),
                                content_hash="",
                                is_valid=True
                            )
                        
                        elif response.status >= 500:
                            # Server error - retry with backoff
                            logger.warning(f"Server error {response.status} for {robots_url}")
                            continue
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching robots.txt from {robots_url}")
                    continue
                except Exception as e:
                    logger.warning(f"Error fetching robots.txt from {robots_url}: {e}")
                    continue
            
            # All attempts failed
            self.stats['fetch_errors'] += 1
            logger.error(f"Failed to fetch robots.txt for {domain} after trying all URLs")
            return None
            
        except Exception as e:
            self.stats['fetch_errors'] += 1
            logger.error(f"Unexpected error fetching robots.txt for {domain}: {e}")
            return None
    
    def parse_robots_txt(self, content: str, domain: str) -> List[RobotsRule]:
        """Parse robots.txt content into structured rules"""
        rules = []
        current_rule = None
        current_user_agents = []
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Handle line continuation (lines starting with space)
            if line.startswith(' ') and current_rule:
                # This is a continuation of the previous directive
                directive, value = self.parse_directive(line.strip())
                if directive and value:
                    self.add_directive_to_rule(current_rule, directive, value)
                continue
            
            # Parse directive
            directive, value = self.parse_directive(line)
            if not directive or not value:
                continue
            
            directive = directive.lower()
            
            if directive == 'user-agent':
                # Save previous rule if it exists
                if current_rule and current_user_agents:
                    for user_agent in current_user_agents:
                        rule_copy = self.copy_rule_for_user_agent(current_rule, user_agent)
                        rules.append(rule_copy)
                
                # Start new rule
                current_user_agents = [ua.strip() for ua in value.split(',')]
                current_rule = RobotsRule(
                    user_agent=current_user_agents[0],  # Primary user agent
                    comment=f"Parsed from line {line_num}"
                )
            
            elif directive in ['allow', 'disallow'] and current_rule:
                self.add_directive_to_rule(current_rule, directive, value)
            
            elif directive == 'crawl-delay' and current_rule:
                try:
                    delay = float(value)
                    current_rule.crawl_delay = delay
                except ValueError:
                    logger.warning(f"Invalid crawl-delay value: {value}")
            
            elif directive == 'sitemap' and current_rule:
                current_rule.sitemap_urls.append(value.strip())
            
            elif directive == 'host' and current_rule:
                current_rule.host = value.strip()
        
        # Save the last rule
        if current_rule and current_user_agents:
            for user_agent in current_user_agents:
                rule_copy = self.copy_rule_for_user_agent(current_rule, user_agent)
                rules.append(rule_copy)
        
        return rules
    
    def parse_directive(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse a robots.txt directive line"""
        if ':' not in line:
            return None, None
        
        directive, value = line.split(':', 1)
        directive = directive.strip().lower()
        value = value.strip()
        
        return directive, value
    
    def add_directive_to_rule(self, rule: RobotsRule, directive: str, value: str):
        """Add a directive to a robots rule"""
        if directive == 'allow':
            rule.allow_patterns.append(value)
        elif directive == 'disallow':
            rule.disallow_patterns.append(value)
    
    def copy_rule_for_user_agent(self, rule: RobotsRule, user_agent: str) -> RobotsRule:
        """Create a copy of a rule for a specific user agent"""
        return RobotsRule(
            user_agent=user_agent,
            allow_patterns=rule.allow_patterns.copy(),
            disallow_patterns=rule.disallow_patterns.copy(),
            crawl_delay=rule.crawl_delay,
            sitemap_urls=rule.sitemap_urls.copy(),
            host=rule.host,
            comment=rule.comment
        )
    
    def get_applicable_rules(self, rules: List[RobotsRule], user_agent: str) -> List[RobotsRule]:
        """Get rules that apply to a specific user agent"""
        applicable = []
        
        # First, look for exact user agent matches
        for rule in rules:
            if rule.user_agent.lower() == user_agent.lower():
                applicable.append(rule)
        
        # If no exact matches, look for wildcard rules
        if not applicable:
            for rule in rules:
                if rule.user_agent == '*':
                    applicable.append(rule)
        
        # If still no matches, look for partial matches (common user agents)
        if not applicable:
            user_agent_lower = user_agent.lower()
            for rule in rules:
                if rule.user_agent.lower() in user_agent_lower or user_agent_lower in rule.user_agent.lower():
                    applicable.append(rule)
        
        return applicable
    
    def check_url_against_rules(self, url: str, rules: List[RobotsRule]) -> Tuple[bool, str]:
        """Check if a URL is allowed by the given rules"""
        parsed_url = urlparse(url)
        path = parsed_url.path or '/'
        
        # Combine all rules into single lists
        all_allow_patterns = []
        all_disallow_patterns = []
        
        for rule in rules:
            all_allow_patterns.extend(rule.allow_patterns)
            all_disallow_patterns.extend(rule.disallow_patterns)
        
        # Check disallow patterns first (more restrictive)
        for pattern in all_disallow_patterns:
            if self.matches_pattern(path, pattern):
                return False, f"Blocked by disallow pattern: {pattern}"
        
        # Check allow patterns (more permissive)
        for pattern in all_allow_patterns:
            if self.matches_pattern(path, pattern):
                return True, f"Allowed by allow pattern: {pattern}"
        
        # Default behavior: if there are any disallow patterns, be restrictive
        if all_disallow_patterns:
            return False, "No matching allow pattern found"
        
        # If no patterns at all, allow
        return True, "No restrictions found"
    
    def matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a robots.txt pattern"""
        if not pattern:
            return False
        
        # Handle empty pattern (disallow: with no value means allow all)
        if pattern == '':
            return False
        
        # Normalize path
        if not path.startswith('/'):
            path = '/' + path
        
        # Handle exact matches
        if pattern == path:
            return True
        
        # Handle wildcard patterns
        if '*' in pattern or '$' in pattern:
            return self.matches_wildcard_pattern(path, pattern)
        
        # Handle prefix matches
        if pattern.endswith('*'):
            pattern = pattern[:-1]
            return path.startswith(pattern)
        
        # Handle exact path matches
        return path.startswith(pattern)
    
    def matches_wildcard_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a wildcard pattern"""
        # Convert robots.txt pattern to regex
        regex_pattern = self.convert_to_regex(pattern)
        
        # Use cached compiled pattern if available
        if regex_pattern in self.pattern_cache:
            compiled_pattern = self.pattern_cache[regex_pattern]
        else:
            try:
                compiled_pattern = re.compile(regex_pattern)
                self.pattern_cache[regex_pattern] = compiled_pattern
            except re.error:
                logger.warning(f"Invalid regex pattern: {regex_pattern}")
                return False
        
        return bool(compiled_pattern.match(path))
    
    def convert_to_regex(self, pattern: str) -> str:
        """Convert robots.txt pattern to regex"""
        # Escape special regex characters except * and $
        escaped = re.escape(pattern)
        
        # Replace escaped \* with .*
        escaped = escaped.replace(r'\*', '.*')
        
        # Handle $ (end of string)
        if pattern.endswith('$'):
            escaped = escaped[:-2] + '$'  # Remove \$, add $
        else:
            escaped += '.*'  # Allow anything after the pattern
        
        return '^' + escaped
    
    def get_crawl_delay(self, rules: List[RobotsRule]) -> Optional[float]:
        """Get the crawl delay from applicable rules"""
        for rule in rules:
            if rule.crawl_delay is not None:
                return rule.crawl_delay
        return None
    
    def extract_sitemap_urls(self, content: str) -> List[str]:
        """Extract sitemap URLs from robots.txt content"""
        sitemap_urls = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                if sitemap_url:
                    sitemap_urls.append(sitemap_url)
        
        return sitemap_urls
    
    async def add_to_cache(self, domain: str, robots_cache: RobotsCache):
        """Add robots.txt cache entry with LRU eviction"""
        async with self.cache_lock:
            # Check if we need to evict entries
            if len(self.robots_cache) >= self.max_cache_size:
                await self.evict_oldest_entries()
            
            self.robots_cache[domain] = robots_cache
            self.cache_access_times[domain] = time.time()
    
    async def evict_oldest_entries(self):
        """Evict oldest cache entries to make room"""
        if not self.cache_access_times:
            return
        
        # Sort by access time and remove oldest 10%
        sorted_domains = sorted(
            self.cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        evict_count = max(1, len(sorted_domains) // 10)
        
        for domain, _ in sorted_domains[:evict_count]:
            if domain in self.robots_cache:
                del self.robots_cache[domain]
            if domain in self.cache_access_times:
                del self.cache_access_times[domain]
    
    async def get_sitemaps_for_domain(self, domain: str, user_agent: str = '*') -> List[str]:
        """Get sitemap URLs for a domain"""
        robots_cache = await self.get_robots_for_domain(domain, user_agent)
        
        if not robots_cache:
            return []
        
        sitemap_urls = []
        for rule in robots_cache.rules:
            sitemap_urls.extend(rule.sitemap_urls)
        
        return list(set(sitemap_urls))  # Remove duplicates
    
    def get_statistics(self) -> Dict:
        """Get handler statistics"""
        return {
            **self.stats,
            'cache_size': len(self.robots_cache),
            'pattern_cache_size': len(self.pattern_cache),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            )
        }
    
    async def clear_cache(self):
        """Clear all cached robots.txt data"""
        async with self.cache_lock:
            self.robots_cache.clear()
            self.cache_access_times.clear()
            self.pattern_cache.clear()
    
    async def preload_robots_for_domains(self, domains: List[str], user_agent: str = '*'):
        """Preload robots.txt for a list of domains"""
        tasks = []
        for domain in domains:
            task = self.get_robots_for_domain(domain, user_agent)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def export_cache(self) -> Dict:
        """Export cache data for persistence"""
        cache_data = {}
        
        for domain, robots_cache in self.robots_cache.items():
            cache_data[domain] = {
                'domain': robots_cache.domain,
                'rules': [
                    {
                        'user_agent': rule.user_agent,
                        'allow_patterns': rule.allow_patterns,
                        'disallow_patterns': rule.disallow_patterns,
                        'crawl_delay': rule.crawl_delay,
                        'sitemap_urls': rule.sitemap_urls,
                        'host': rule.host,
                        'comment': rule.comment
                    }
                    for rule in robots_cache.rules
                ],
                'last_fetched': robots_cache.last_fetched.isoformat(),
                'expires_at': robots_cache.expires_at.isoformat(),
                'content_hash': robots_cache.content_hash,
                'fetch_errors': robots_cache.fetch_errors,
                'is_valid': robots_cache.is_valid
            }
        
        return cache_data
    
    def import_cache(self, cache_data: Dict):
        """Import cache data from persistence"""
        for domain, data in cache_data.items():
            rules = []
            for rule_data in data['rules']:
                rule = RobotsRule(
                    user_agent=rule_data['user_agent'],
                    allow_patterns=rule_data['allow_patterns'],
                    disallow_patterns=rule_data['disallow_patterns'],
                    crawl_delay=rule_data['crawl_delay'],
                    sitemap_urls=rule_data['sitemap_urls'],
                    host=rule_data['host'],
                    comment=rule_data['comment']
                )
                rules.append(rule)
            
            robots_cache = RobotsCache(
                domain=data['domain'],
                rules=rules,
                last_fetched=datetime.fromisoformat(data['last_fetched']),
                expires_at=datetime.fromisoformat(data['expires_at']),
                content_hash=data['content_hash'],
                fetch_errors=data['fetch_errors'],
                is_valid=data['is_valid']
            )
            
            self.robots_cache[domain] = robots_cache
            self.cache_access_times[domain] = time.time()

