import functools
import json
import logging
import redis
import hashlib
from typing import Any, Callable, Optional
from config import settings

logger = logging.getLogger("CacheUtils")

# Initialize Redis connection
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis cache initialized successfully.")
except Exception as e:
    logger.warning(f"Redis cache initialization failed: {e}. Caching will be disabled.")
    redis_client = None

def redis_cache(ttl: int = 86400, prefix: str = "medicortex:cache"):
    """
    Decorator for caching function results in Redis.
    Args:
        ttl: Time to live in seconds (default: 24 hours).
        prefix: Prefix for the Redis key.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not redis_client:
                return func(*args, **kwargs)

            # Create a unique key based on function name and arguments
            key_data = f"{func.__name__}:{args}:{kwargs}"
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            cache_key = f"{prefix}:{func.__name__}:{key_hash}"

            try:
                # Check cache
                cached_val = redis_client.get(cache_key)
                if cached_val:
                    logger.info(f"Cache HIT for {func.__name__} (key: {cache_key[:30]}...)")
                    return json.loads(cached_val)
            except Exception as e:
                logger.warning(f"Cache lookup failed for {func.__name__}: {e}")

            # Execute function
            result = func(*args, **kwargs)

            try:
                # Save to cache
                if result is not None:
                    redis_client.setex(cache_key, ttl, json.dumps(result))
                    logger.info(f"Cache MISS for {func.__name__}. Saved result to cache.")
            except Exception as e:
                logger.warning(f"Cache save failed for {func.__name__}: {e}")

            return result
        return wrapper
    return decorator
