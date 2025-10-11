from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.connections.inference_client import InferenceClient
from app.core.dependencies import get_inference_client

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
    triton_client: Annotated[InferenceClient, Depends(get_inference_client)],
) -> PlainTextResponse:
    model_ready = await triton_client.is_model_ready()
    server_ready = await triton_client.is_server_ready()
    if not (model_ready and server_ready):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        )
    return PlainTextResponse(content="OK")
