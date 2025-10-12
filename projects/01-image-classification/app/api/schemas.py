"""Pydantic response models for the API."""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Represents a single prediction result."""
    class_id: int = Field(..., description="Predicted class index")
    class_name: str = Field(..., description="Human-readable class label")
    confidence: float = Field(..., description="Prediction confidence score (0.0–1.0)")

class ClassificationResponse(BaseModel):
    """Represents model classification output."""
    predictions: list[Prediction] = Field(..., description="List of class predictions")
