"""Asynchronous Redis client wrapper."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import redis.asyncio as redis
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnError

from app.core.exceptions import RedisConnectionError

logger = logging.getLogger(__name__)


class RedisClient:
    """Asynchronous Redis client for the application."""

    def __init__(self, redis_url: str) -> None:
        """Initialize a Redis client.

        Args:
            redis_url: Base URL of the Redis API (e.g. `redis://localhost:6379`).
        """
        self._url = redis_url
        self._client = redis.from_url(self._url, decode_responses=True)
        logger.info("RedisClient initialized for %s", self._url)

    async def connect(
        self, connection_timeout: int = 10, interval: float = 1.0,
    ) -> None:
        """Wait until the Redis server becomes responsive.

        Args:
            connection_timeout: Maximum wait time in seconds before giving up.
            interval: Delay between retry attempts in seconds.

        Raises:
            RedisConnectionError: If Redis is not ready before the timeout.
        """
        start = datetime.now(UTC)
        deadline = start + timedelta(seconds=connection_timeout)

        logger.info("Waiting for Redis readiness (%s)", self._url)

        while datetime.now(UTC) < deadline:
            if await self.is_ready():
                return

            await asyncio.sleep(interval)

        msg = (
            f"Timeout waiting for Redis readiness after "
            f"{connection_timeout}s ({self._url})"
        )
        logger.error(msg)
        raise RedisConnectionError(msg)

    async def disconnect(self) -> None:
        """Close the Redis connection gracefully."""
        if not self._client:
            logger.warning("Redis disconnect called but client is not initialized.")
            return
        await self._client.close()
        logger.info("Redis disconnected from %s", self._url)

    async def is_ready(self) -> bool:
        """Check if Redis is responsive.

        Returns:
            bool: True if Redis responds to PING, False otherwise.
        """
        try:
            return bool(await self._client.ping())
        except RedisConnError:
            return False

    @property
    def client(self) -> Redis:
        """Expose the underlying Redis client.

        Returns:
            Redis: The active Redis client instance.

        Raises:
            RedisConnectionError: If the client is not initialized.
        """
        if not self._client:
            msg = "Redis client not initialized."
            raise RedisConnectionError(msg)
        return self._client
