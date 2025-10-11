"""Utility functions for image preprocessing and validation."""

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.core.config import settings


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for ResNet50 inference.

    Args:
        image (Image.Image): Input PIL image.

    Returns:
        np.ndarray: Preprocessed image array with shape (1, 3, 224, 224).
    """
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file.

    Args:
        file (UploadFile): Uploaded image file.

    Raises:
        HTTPException: If file is invalid, wrong type, or too large.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = file.filename.split(".")[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size} > {settings.MAX_IMAGE_SIZE})",
        )
