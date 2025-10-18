"""SSE chat streaming route."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest
from app.core.dependencies import get_llm_service
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["Chat"])


@chat_router.post(
    "/chat",
    responses={
        200: {"description": "Streaming chat response"},
        500: {"description": "Internal Server Error."},
    },
    summary="Stream chat responses token-by-token",
    description="""
    Streams responses from the LLM in real time using Server-Sent Events (SSE).
    Partial format from: https://platform.openai.com/docs/api-reference/chat-streaming/streaming
    """,
)
async def chat(
    chat_request: ChatRequest,
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> StreamingResponse:
    """Stream LLM responses via SSE following OpenAI format."""
    token_stream = llm_service.chat_stream(
        user_id=chat_request.user_id,
        message=chat_request.message,
        model=chat_request.model,
        **chat_request.model_dump(exclude=["user_id", "message", "model"]),
    )

    logger.info(
        "Stream started (user=%s, session=%s)",
        chat_request.user_id,
        chat_request.session_id,
    )

    return StreamingResponse(
        token_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
