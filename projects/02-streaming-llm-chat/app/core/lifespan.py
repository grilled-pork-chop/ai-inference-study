"""FastAPI lifespan management for initializing and closing resources."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.connections.llm_client import LLMClient
from app.connections.redis_client import RedisClient
from app.core.config import settings
from app.services.conversation_service import ConversationService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context for FastAPI application."""

    redis_client = RedisClient(redis_url=settings.REDIS_URL)
    conversation_service = ConversationService(
        redis_client=redis_client, ttl_seconds=settings.CONVERSATION_TTL_SECONDS,
    )
    llm_client = LLMClient(
        base_url=settings.LLM_URL,
    )
    llm_service = LLMService(
        conversation_service=conversation_service, llm_client=llm_client,
    )

    await asyncio.gather(
        llm_client.connect(),
        redis_client.connect(),
    )


    app.state.redis_client = redis_client
    app.state.conversation_service = llm_client
    app.state.llm_client = llm_client
    app.state.llm_service = llm_service

    logger.info("Lifespan startup completed")

    yield

    if hasattr(app.state, "redis_client"):
        await app.state.redis_client.disconnect()

    logger.info("Lifespan shutdown completed")
