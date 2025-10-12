"""FastAPI dependencies."""


from fastapi import Request

from app.connections.inference_client import InferenceClient
from app.services.classification_service import ClassificationService


def get_inference_client(request: Request) -> InferenceClient:
    """Dependency to get the initialized Inference client."""
    return request.app.state.inference_client

def get_classification_service(request: Request) -> ClassificationService:
    """Dependency to get the initialized classification service."""
    return request.app.state.classification_service

def get_imagenet_classes(request: Request) -> list[str]:
    """Dependency to get ImageNet class labels."""
    return request.app.state.imagenet_classes
