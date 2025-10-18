"""Business logic for conversation persistence using Redis."""

import json
import logging
from typing import Any

from app.connections.redis_client import RedisClient
from app.core.exceptions import ConversationHistoryError

logger = logging.getLogger(__name__)


class ConversationService:
    """High-level service for managing persisted chat conversations."""

    def __init__(self, redis_client: RedisClient, ttl_seconds: int) -> None:
        """Initialize the conversation service.

        Args:
            redis_client: Connected Redis client instance.
            ttl_seconds: Expiration time for stored conversations (in seconds).
        """
        self.redis_client = redis_client
        self.ttl = ttl_seconds

    def _key(self, user_id: str, session_id: str) -> str:
        """Generate a Redis key for the given user and session."""
        return f"user:{user_id}:session:{session_id}"

    async def load(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """Retrieve a user's conversation from Redis.

        Args:
            user_id: Unique identifier for the user.
            session_id: Unique identifier for the conversation session.

        Returns:
            A list of message dictionaries, or an empty list if not found.

        Raises:
            ConversationHistoryError: If retrieval fails.
        """
        try:
            data = await self.redis_client.client.get(self._key(user_id, session_id))
            return json.loads(data) if data else []
        except Exception as e:
            logger.error(
                "Failed to load conversation (user=%s, session=%s): %s",
                user_id,
                session_id,
                e,
            )
            raise ConversationHistoryError(str(e)) from e

    async def save(
        self, user_id: str, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Save or update a user's conversation in Redis.

        Args:
            user_id: Unique identifier for the user.
            session_id: Unique identifier for the conversation.
            messages: List of message objects representing the chat history.

        Raises:
            ConversationHistoryError: If saving fails.
        """
        try:
            await self.redis_client.client.set(
                self._key(user_id, session_id),
                json.dumps(messages),
                ex=self.ttl,
            )
            logger.debug(
                "Conversation saved (user=%s, session=%s)", user_id, session_id
            )
        except Exception as e:
            logger.error(
                "Failed to save conversation (user=%s, session=%s): %s",
                user_id,
                session_id,
                e,
            )
            raise ConversationHistoryError(str(e)) from e
