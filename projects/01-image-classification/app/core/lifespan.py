"""FastAPI lifespan management for initializing and closing resources."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI

from app.connections.inference_client import InferenceClient
from app.services.classification_service import ClassificationService

logger = logging.getLogger(__name__)

async def load_imagenet_classes() -> list[str]:
    """Load ImageNet class labels from a local JSON file asynchronously.

    Returns:
        list[str]: A list of class labels.

    Fallback:
        Returns generic class names if the file is missing or invalid.
    """
    file_path = Path(__file__).parent.parent / "static" / "imagenet_classes.json"

    if not file_path.exists():
        logger.warning("ImageNet classes file not found at %d", file_path)
        return [f"class_{i}" for i in range(1000)]

    async with aiofiles.open(file_path) as f:
        data = await f.read()
        classes = json.loads(data)
        if isinstance(classes, list) and len(classes):
            logger.info("Loaded %d ImageNet class labels.", len(classes))
            return classes
        msg = "Invalid format in imagenet_classes.json"
        raise ValueError(msg)

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """ Lifespan context for FastAPI application. """
    imagenet_classes = await load_imagenet_classes()

    inference_client = InferenceClient()
    await inference_client.connect()

    classification_service = ClassificationService(inference_client, imagenet_classes)

    app.state.inference_client = inference_client
    app.state.classification_service = classification_service
    app.state.imagenet_classes = imagenet_classes

    logger.info("Lifespan startup completed")

    yield

    if hasattr(app.state, "inference_client"):
        app.state.inference_client.disconnect()

    logger.info("Lifespan shutdown completed")
