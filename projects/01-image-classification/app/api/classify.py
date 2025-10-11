"""FastAPI routes for image classification API."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.schemas import ClassificationResponse
from app.core.dependencies import get_classification_service
from app.core.utils import preprocess_image, validate_image
from app.services.classification_service import ClassificationService

classify_router = APIRouter(tags=["Classify"])

@classify_router.post("/classify")
async def classify(
    classification_service: Annotated[ClassificationService, Depends(get_classification_service)],
    file: Annotated[UploadFile, File()] = ...,
    top_k: Annotated[int, Query(ge=1, le=10)] = 5,
) -> ClassificationResponse:
    """Classify an uploaded image.

    Args:
        file (UploadFile): The image to classify.
        top_k (int): Number of top predictions to return.

    Returns:
        ClassificationResponse: List of predictions.
    """
    validate_image(file)
    img_array = preprocess_image(file.file)
    predictions = await classification_service.classify(img_array, top_k=top_k)
    return ClassificationResponse(predictions=predictions)
