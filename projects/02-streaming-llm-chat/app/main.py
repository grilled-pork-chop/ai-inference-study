"""FastAPI application entrypoint."""

import uvicorn
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import api_router, root_router
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.logger import configure_logger


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logger(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        lifespan=lifespan,
    )


    register_exception_handlers(app)

    app.include_router(root_router)
    app.include_router(api_router)

    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    )

    instrumentator.instrument(app).expose(app, endpoint="/metrics")

    return app


def run() -> None:
    """Main entry point for the application."""
    uvicorn.run(
        "app.main:create_app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:create_app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
        factory=True,
    )
