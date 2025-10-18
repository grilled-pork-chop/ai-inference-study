from fastapi import APIRouter

from app.api.chat import chat_router
from app.api.root import root_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat_router)

__all__ = ["api_router", "root_router"]