# cache_config.py
"""
Caching and Rate Limiting Configuration for Fall Detection System
Implements Redis-based caching and Flask-Limiter for API rate limiting
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from functools import wraps

try:
    import redis
    from flask_caching import Cache
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("Caching libraries not installed. Install with: pip install Flask-Caching Flask-Limiter redis")

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for Firebase queries and API responses"""
    
    def __init__(self, app=None, redis_url=None):
        self.cache = None
        self.redis_client = None
        self.initialized = False
        
        if not CACHE_AVAILABLE:
            logger.warning("Cache libraries not available")
            return
            
        try:
            # Initialize Redis connection
            redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test Redis connection
            self.redis_client.ping()
            logger.info(f"Redis connected successfully: {redis_url}")
            
            if app:
                self.init_app(app)
                
        except Exception as e:
            logger.warning(f"Redis connection failed, using simple cache: {e}")
            self.redis_client = None
            if app:
                self.init_app(app, use_simple=True)
    
    def init_app(self, app, use_simple=False):
        """Initialize Flask-Caching with the app"""
        try:
            if use_simple or not self.redis_client:
                # Use simple in-memory cache as fallback
                app.config['CACHE_TYPE'] = 'SimpleCache'
                app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
                logger.info("Using SimpleCache (in-memory) for caching")
            else:
                # Use Redis cache
                app.config['CACHE_TYPE'] = 'RedisCache'
                app.config['CACHE_REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
                logger.info("Using Redis for caching")
            
            self.cache = Cache(app)
            self.initialized = True
            logger.info("Cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self.initialized = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.initialized:
            return None
            
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, timeout: int = 300) -> bool:
        """Set value in cache"""
        if not self.initialized:
            return False
            
        try:
            return self.cache.set(key, value, timeout=timeout)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.initialized:
            return False
            
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        if not self.initialized:
            return False
            
        try:
            return self.cache.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        # Filter out non-serializable objects and only use safe parameters
        safe_params = {}
        for key, value in kwargs.items():
            try:
                # Only include serializable values
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    safe_params[key] = value
                elif hasattr(value, '__name__'):  # Function names
                    safe_params[key] = value.__name__
                else:
                    # Skip non-serializable objects like FirebaseHandler
                    continue
            except:
                continue
        
        # Create a hash from the safe parameters
        params_str = json.dumps(safe_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{prefix}:{params_hash}"

class RateLimitManager:
    """Manages rate limiting for API endpoints"""
    
    def __init__(self, app=None):
        self.limiter = None
        self.initialized = False
        
        if not CACHE_AVAILABLE:
            logger.warning("Rate limiting libraries not available")
            return
            
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize Flask-Limiter with the app"""
        try:
            # Configure rate limiting storage
            redis_url = os.getenv('REDIS_URL')
            if redis_url:
                storage_uri = redis_url
            else:
                storage_uri = "memory://"
            
            self.limiter = Limiter(
                key_func=get_remote_address,
                default_limits=["1000 per day", "100 per hour"],
                storage_uri=storage_uri
            )
            self.limiter.init_app(app)
            
            self.initialized = True
            logger.info(f"Rate limiter initialized with storage: {storage_uri}")
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            self.initialized = False
    
    def limit(self, rate_limit: str):
        """Decorator for rate limiting endpoints"""
        if not self.initialized:
            # Return a no-op decorator if not initialized
            def decorator(f):
                return f
            return decorator
        
        return self.limiter.limit(rate_limit)

def cached_firebase_query(timeout: int = 300, key_prefix: str = "firebase"):
    """Decorator for caching Firebase queries with lazy cache manager evaluation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager at runtime, not at import time
            cache_manager = get_cache_manager()
            
            if not cache_manager or not cache_manager.initialized:
                # If cache is not available, execute function directly
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cache_manager.generate_key(key_prefix, func=func.__name__, args=args, kwargs=kwargs)
            
            # Try to get from cache first
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            try:
                result = func(*args, **kwargs)
                
                # Only cache serializable results
                try:
                    json.dumps(result, default=str)  # Test if result is serializable
                    cache_manager.set(cache_key, result, timeout=timeout)
                    logger.debug(f"Cache miss, stored result for key: {cache_key}")
                except (TypeError, ValueError) as json_error:
                    logger.warning(f"Result not cacheable for {func.__name__}: {json_error}")
                
                return result
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                # Return empty result on error to prevent cascading failures
                return [] if 'get' in func.__name__.lower() else None
        
        return wrapper
    return decorator

def optimized_firebase_query(original_query_func):
    """Decorator to optimize Firebase queries"""
    @wraps(original_query_func)
    def wrapper(*args, **kwargs):
        # Add query optimizations
        if 'limit' not in kwargs:
            kwargs['limit'] = 100  # Default limit to prevent large queries
        
        # Add timestamp filtering for recent queries
        if 'recent' in original_query_func.__name__ and 'hours' not in kwargs:
            kwargs['hours'] = 24  # Default to last 24 hours for recent queries
        
        return original_query_func(*args, **kwargs)
    
    return wrapper

# Global instances
cache_manager = None
rate_limit_manager = None

def initialize_cache_and_rate_limiting(app, redis_url=None):
    """Initialize global cache and rate limiting managers"""
    global cache_manager, rate_limit_manager
    
    cache_manager = CacheManager(app, redis_url)
    rate_limit_manager = RateLimitManager(app)
    
    return cache_manager, rate_limit_manager

def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager instance"""
    return cache_manager

def get_rate_limit_manager() -> Optional[RateLimitManager]:
    """Get global rate limit manager instance"""
    return rate_limit_manager