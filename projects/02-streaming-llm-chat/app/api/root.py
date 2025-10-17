import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.connections.llm_client import LLMClient
from app.connections.redis_client import RedisClient
from app.core.dependencies import get_llm_client, get_redis_client

logger = logging.getLogger(__name__)

root_router = APIRouter(tags=["Root"])

@root_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Return a plain text 'OK' message if the service is healthy.",
    responses={
        status.HTTP_200_OK: {"description": "Service is healthy"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def health_check() -> PlainTextResponse:
    return PlainTextResponse(content="OK")


@root_router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Return a plain text 'OK' message if the service is ready.",
    responses={
        status.HTTP_200_OK: {"description": "Service is ready"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service is not ready"},
    },
)
async def ready(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
    redis_client: Annotated[RedisClient, Depends(get_redis_client)],
) -> PlainTextResponse:
    llm_ready, redis_ready = await asyncio.gather(
        llm_client.is_ready(),
        redis_client.is_ready(),
    )

    if not all([llm_ready, redis_ready]):
        logger.warning(
            "Readiness check failed (llm_ready=%s, redis_ready=%s)",
            llm_ready,
            redis_ready,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    return PlainTextResponse(content="OK")

