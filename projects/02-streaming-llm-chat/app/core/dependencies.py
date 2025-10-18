"""FastAPI dependencies."""

from fastapi import Request

from app.connections.llm_client import LLMClient
from app.connections.redis_client import RedisClient
from app.services.llm_service import LLMService


def get_llm_client(request: Request) -> LLMClient:
    """Dependency to get the initialized LLM client."""
    return request.app.state.llm_client

def get_redis_client(request: Request) -> RedisClient:
    """Dependency to get the initialized Redis client."""
    return request.app.state.redis_client

def get_llm_service(request: Request) -> LLMService:
    """Dependency to get the initialized LLM service."""
    return request.app.state.llm_service
