"""LLM inference server client wrapper (OpenAI-compatible)."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from app.core.exceptions import LLMConnectionError, LLMGenerationError

logger = logging.getLogger(__name__)

class LLMClient:
    """Async wrapper around an OpenAI-compatible inference backend (vLLM, Ollama, OpenAI)."""

    def __init__(self, base_url: str, api_key: str) -> None:
        """Initialize an LLM client.

        Args:
            base_url: Base URL of the LLM API (e.g. `http://localhost:8000/v1`).
            api_key: API key of the LLM API
        """
        self._url = base_url
        self._client = AsyncOpenAI(base_url=self._url, api_key=api_key)
        logger.info("LLM client initialized (base_url=%s)", self._url)

    async def generate_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """Stream the LLM's response in OpenAI-compatible format.

        Sends the given chat messages to the model and yields the output incrementally
        as SSE-formatted JSON chunks.

        Args:
            messages: Ordered list of chat message dictionaries following the
                OpenAI chat schema (`role` and `content` keys required).
            model: Model identifier. If omitted, the default backend model is used.
            temperature: Sampling temperature (controls randomness).
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling probability threshold.

        Yields:
            str: JSON-formatted SSE data lines, each representing a partial delta
            from the model, followed by a final ``[DONE]`` marker.

        Raises:
            LLMGenerationError: If an API or internal error occurs during streaming.
        """
        try:
            stream_obj = await self._client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
            )

            async for chunk in stream_obj:
                if chunk.choices and any(c.delta for c in chunk.choices):
                    yield f"data: {json.dumps(chunk.to_dict())}\n\n"
            yield "data: [DONE]\n\n"

        except OpenAIError as e:
            msg = f"LLM streaming failed: {e}"
            logger.error(msg)
            raise LLMGenerationError(msg) from e
        except Exception as e:
            msg = f"Unexpected error in LLM stream: {e}"
            logger.exception(msg)
            raise LLMGenerationError(msg) from e


    async def connect(self, connection_timeout: int = 10, interval: float = 1.0) -> None:
        """Wait until the LLM server becomes responsive.

        Args:
            connection_timeout: Maximum wait time in seconds before failing.
            interval: Delay between retry attempts in seconds.

        Raises:
            LLMConnectionError: If the LLM server is not ready before timeout.
        """
        start = datetime.now(UTC)
        deadline = start + timedelta(seconds=connection_timeout)
        logger.info("Waiting for LLM readiness (%s)", self._url)

        while datetime.now(UTC) < deadline:
            if await self.is_ready():
                logger.info("LLM server ready at %s", self._url)
                return

            logger.debug("LLM not ready yet, retrying in %.1fs...", interval)
            await asyncio.sleep(interval)

        msg = f"Timeout waiting for LLM readiness after {connection_timeout}s ({self._url})"
        logger.error(msg)
        raise LLMConnectionError(msg)

    async def is_ready(self) -> bool:
        """Check if the LLM backend is responsive.

        Returns:
            bool: True if the LLM responds successfully to a model listing query.
        """
        try:
            models = await self._client.models.list()
            return bool(models.data)
        except (OpenAIError, APIConnectionError, APIStatusError) as e:
            logger.debug("LLM readiness check failed: %s", e)
            return False
