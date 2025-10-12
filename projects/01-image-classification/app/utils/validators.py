"""Async image validation utilities."""

from fastapi import HTTPException, UploadFile

from app.core.config import settings


async def validate_image(file: UploadFile) -> bytes:
    """Validate and read uploaded image bytes.

    Raises:
        HTTPException: If file is invalid.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(data) > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_IMAGE_SIZE} bytes allowed.",
        )

    return data
