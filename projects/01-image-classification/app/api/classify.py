"""FastAPI routes for image classification API."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.schemas import ClassificationResponse
from app.core.dependencies import get_classification_service
from app.services.classification_service import ClassificationService
from app.utils.validators import validate_image

classify_router = APIRouter(tags=["Classify"])

@classify_router.post(
    "/classify",
    responses={
        400: {"description": "Bad Request - Invalid or empty image file."},
        413: {"description": "Payload Too Large."},
        415: {"description": "Unsupported Media Type."},
        500: {"description": "Internal Server Error."},
    },
    summary="Classify an uploaded image",
    description="Uploads an image and returns the top-k class predictions from the model.",
)
async def classify(
    classification_service: Annotated[ClassificationService, Depends(get_classification_service)],
    file: Annotated[UploadFile, File()] = ...,
    top_k: Annotated[int, Query(ge=1, le=10)] = 5,
) -> ClassificationResponse:
    image_data = await validate_image(file)
    predictions = await classification_service.classify(image_data, top_k=top_k)
    return ClassificationResponse(predictions=predictions)
