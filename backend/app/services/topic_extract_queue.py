import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional

from app.config import get_settings

try:
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover
    redis_async = None

logger = logging.getLogger(__name__)


class TopicExtractionQueueError(Exception):
    """Base exception for topic extraction queue issues."""


class TopicExtractionQueueFullError(TopicExtractionQueueError):
    """Raised when incoming requests exceed queue capacity."""


class TopicExtractionQueueTimeoutError(TopicExtractionQueueError):
    """Raised when waiting for an available worker times out."""


class InMemoryTopicExtractionQueueManager:
    """Simple in-memory queue controller for topic extraction workloads."""

    def __init__(
        self,
        max_workers: int,
        queue_limit: Optional[int],
        timeout_seconds: Optional[int],
    ) -> None:
        if max_workers < 1:
            max_workers = 1
        self.max_workers = max_workers
        self.queue_limit = queue_limit if queue_limit and queue_limit > 0 else None
        self.timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None

        self._worker_semaphore = asyncio.Semaphore(self.max_workers)
        self._lock = asyncio.Lock()
        self._waiting_requests = 0
        self._active_workers = 0

    async def _register_waiting(self) -> None:
        async with self._lock:
            if self.queue_limit is not None and self._waiting_requests >= self.queue_limit:
                raise TopicExtractionQueueFullError(
                    "Topic extraction queue capacity exhausted."
                )
            self._waiting_requests += 1

    async def _unregister_waiting(self) -> None:
        async with self._lock:
            if self._waiting_requests > 0:
                self._waiting_requests -= 1

    async def _set_active(self, delta: int) -> None:
        async with self._lock:
            self._active_workers = max(0, self._active_workers + delta)

    async def _wait_for_worker(self) -> None:
        if self.timeout_seconds is None:
            await self._worker_semaphore.acquire()
            return
        try:
            await asyncio.wait_for(self._worker_semaphore.acquire(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TopicExtractionQueueTimeoutError(
                "Timed out while waiting for topic extraction worker."
            ) from exc

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        await self._register_waiting()
        try:
            await self._wait_for_worker()
        except Exception:
            await self._unregister_waiting()
            raise
        await self._unregister_waiting()

        await self._set_active(1)
        try:
            yield
        finally:
            await self._set_active(-1)
            self._worker_semaphore.release()

    async def get_metrics(self) -> Dict[str, int]:
        async with self._lock:
            return {
                "active_workers": self._active_workers,
                "max_workers": self.max_workers,
                "waiting_requests": self._waiting_requests,
                "queue_limit": self.queue_limit or 0,
            }


class RedisTopicExtractionQueueManager:
    """Distributed queue controller backed by Redis sorted sets."""

    _ACQUIRE_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2])
local active = redis.call('ZCARD', KEYS[1])
if active < tonumber(ARGV[1]) then
    redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
    return 1
end
return 0
"""
    _RELEASE_LUA = "return redis.call('ZREM', KEYS[1], ARGV[1])"

    def __init__(
        self,
        redis_client: "redis_async.Redis",
        *,
        max_workers: int,
        poll_interval_ms: int,
        lease_seconds: int,
        namespace: str = "topic_extraction",
    ) -> None:
        if max_workers < 1:
            max_workers = 1
        if lease_seconds <= 0:
            lease_seconds = 900
        self.redis = redis_client
        self.max_workers = max_workers
        self.poll_interval = max(0.05, poll_interval_ms / 1000)
        self.lease_seconds = lease_seconds
        self.active_key = f"{namespace}:active_workers"

    async def _acquire_token(self, token: str) -> bool:
        now = time.time()
        expire_before = now - self.lease_seconds
        try:
            result = await self.redis.eval(
                self._ACQUIRE_LUA,
                1,
                self.active_key,
                str(self.max_workers),
                str(expire_before),
                str(now),
                token,
            )
            return int(result or 0) == 1
        except Exception as exc:  # pragma: no cover
            logger.error("Redis queue acquire failed: %s", exc)
            raise TopicExtractionQueueError("Failed to acquire Redis queue slot") from exc

    async def _release_token(self, token: str) -> None:
        try:
            await self.redis.eval(self._RELEASE_LUA, 1, self.active_key, token)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to release Redis queue token %s: %s", token, exc)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        token = uuid.uuid4().hex
        acquired = False
        while not acquired:
            acquired = await self._acquire_token(token)
            if not acquired:
                await asyncio.sleep(self.poll_interval)
        try:
            yield
        finally:
            await self._release_token(token)

    async def get_metrics(self) -> Dict[str, int]:
        try:
            active = await self.redis.zcard(self.active_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Redis queue metrics failed: %s", exc)
            active = -1
        return {
            "active_workers": max(0, int(active)),
            "max_workers": self.max_workers,
            "waiting_requests": 0,
            "queue_limit": 0,
        }


def _create_redis_client():
    if redis_async is None:
        raise RuntimeError("redis package is not installed")
    settings = get_settings()
    if settings.redis_url:
        return redis_async.Redis.from_url(
            settings.redis_url,
            encoding=None,
            decode_responses=False,
        )
    connection_kwargs = {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "db": settings.redis_db,
        "password": settings.redis_password,
        "encoding": None,
        "decode_responses": False,
    }
    if settings.redis_ssl:
        try:
            ssl_connection_cls = redis_async.connection.SSLConnection
        except AttributeError as exc:
            raise RuntimeError("Installed redis client does not support SSL connections") from exc
        connection_kwargs["connection_class"] = ssl_connection_cls
    return redis_async.Redis(**connection_kwargs)


def _build_queue_manager():
    settings = get_settings()
    backend = (settings.topic_extract_queue_backend or "memory").lower()
    if backend == "redis":
        if redis_async is None:
            logger.warning(
                "Topic extraction queue configured for Redis backend, but redis package is not installed. "
                "Falling back to in-memory queue."
            )
        else:
            try:
                redis_client = _create_redis_client()
                logger.info(
                    "Topic extraction queue using Redis backend at %s:%s (db=%s)",
                    settings.redis_host if not settings.redis_url else "url",
                    settings.redis_port,
                    settings.redis_db,
                )
                return RedisTopicExtractionQueueManager(
                    redis_client,
                    max_workers=settings.topic_extract_max_workers,
                    poll_interval_ms=settings.topic_extract_queue_poll_interval_ms,
                    lease_seconds=settings.topic_extract_queue_lease_seconds,
                )
            except Exception as exc:
                logger.exception("Failed to initialize Redis queue. Falling back to in-memory queue: %s", exc)

    logger.info("Topic extraction queue using in-memory backend")
    return InMemoryTopicExtractionQueueManager(
        max_workers=settings.topic_extract_max_workers,
        queue_limit=settings.topic_extract_queue_limit,
        timeout_seconds=settings.topic_extract_queue_timeout_seconds,
    )


topic_extraction_queue = _build_queue_manager()