"""Centralized exception handlers for the FastAPI application."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


async def llm_exception_handler(
    request: Request, exc: LLMError,
) -> JSONResponse:
    """Handles custom LLMBaseException and returns a JSON response."""
    logger.error(
        "Handled API error: %s - %s (Request URL: %s)",
        exc.status_code,
        exc.message,
        request.url,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


async def unhandled_exception_handler(request: Request, _: Exception) -> JSONResponse:
    """Handles any unhandled exceptions and returns a generic 500 error."""
    logger.exception(
        "An unhandled exception occurred during request processing (Request URL: %s)",
        request.url,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers all custom exception handlers with the FastAPI application."""
    app.add_exception_handler(LLMError, llm_exception_handler)
    app.add_exception_handler(
        Exception, unhandled_exception_handler,
    )  # Register catch-all handler
    logger.info("Custom exception handlers registered.")
