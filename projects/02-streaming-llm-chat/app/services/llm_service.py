"""Business logic for LLM chat interactions."""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from app.connections.llm_client import LLMClient
from app.core.exceptions import (
    ConversationHistoryError,
    LLMGenerationError,
    LLMServiceError,
)
from app.services.conversation_service import ConversationService

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class LLMService:
    """Service for handling LLM chat requests with persisted conversation context."""

    def __init__(
        self, llm_client: LLMClient, conversation_service: ConversationService,
    ) -> None:
        """Initialize the LLM chat service.

        Args:
            llm_client: LLM client used to communicate with the inference backend.
            conversation_service: Service managing conversation history storage.
        """
        self.llm_client = llm_client
        self.conversation_service = conversation_service
        self.system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        }

    async def chat_stream(  # noqa: PLR0913
        self,
        user_id: str,
        message: str,
        model: str,
        *,
        session_id: str | None = None,
        temperature: float= 0.7,
        top_p: float = 1.0,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion while managing persisted conversation state.

        Args:
            user_id (str): Unique identifier for the user.
            message (str): User input message.
            model(str): Model to used.
            session_id (str | None): Optional session ID for conversation continuity.
            temperature (float, optional): Sampling temperature controlling response randomness.
            top_p (float, optional): Nucleus sampling parameter that limits sampling
                to tokens with a cumulative probability ≤ `top_p`.
            stream (bool, optional): Whether to enable token-by-token streaming of the response.

        Yields:
            str: Each streamed token (SSE chunk) from the LLM.

        Raises:
            LLMServiceError: If an error occurs during chat generation or persistence.
        """
        session_id = session_id or str(uuid.uuid4())
        logger.info("Chat session started (user=%s, session=%s)", user_id, session_id)

        try:
            messages = await self.conversation_service.load(user_id, session_id)
        except ConversationHistoryError as e:
            logger.warning("Failed to load conversation (user=%s): %s", user_id, e)
            messages = []

        if not messages:
            messages = [self.system_message]

        messages.append({"role": "user", "content": message})

        full_response = ""
        try:
            async for token in self.llm_client.generate_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            ):
                full_response += token
                yield token
        except LLMGenerationError as e:
            logger.error(
                "LLM generation error (user=%s, session=%s): %s", user_id, session_id, e,
            )
            msg = "LLM failed to generate response."
            raise LLMServiceError(msg) from e
        except Exception as e:
            logger.exception(
                "Unexpected error during chat stream (user=%s, session=%s)",
                user_id,
                session_id,
            )
            msg = "Unexpected error during chat stream."
            raise LLMServiceError(msg) from e

        messages.append({"role": "assistant", "content": full_response})

        try:
            await self.conversation_service.save(user_id, session_id, messages)
            logger.debug(
                "Conversation updated (user=%s, session=%s)", user_id, session_id,
            )
        except ConversationHistoryError as e:
            logger.error(
                "Failed to persist conversation (user=%s, session=%s): %s",
                user_id,
                session_id,
                e,
            )
