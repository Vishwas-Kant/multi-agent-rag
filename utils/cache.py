"""
Caching utilities — LRU + TTL caches for reducing redundant computation.
"""

from __future__ import annotations

import functools
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional


class TTLCache:
    """
    Thread-safe dictionary cache with per-entry TTL expiration.
    """

    def __init__(self, max_size: int = 256, default_ttl: float = 300.0) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if time.time() > expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._store[key] = (value, time.time() + ttl)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def cached(ttl: float = 300.0, max_size: int = 128) -> Callable:
    _cache = TTLCache(max_size=max_size, default_ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            raw_key = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
            key = hashlib.sha256(raw_key.encode()).hexdigest()
            hit = _cache.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            _cache.set(key, result)
            return result

        wrapper.cache = _cache
        return wrapper

    return decorator
