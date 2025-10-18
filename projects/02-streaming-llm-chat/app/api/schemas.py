"""Pydantic response models for the API."""


from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Represents an incoming chat message."""
    user_id: str = Field(..., description="Unique user identifier.")
    message: str = Field(..., description="User message input.")
    session_id: str | None = Field(None, description="Conversation session identifier.")

    # Runtime-tunable parameters
    model: str = Field(..., description="Model for this request.", example="tinyllama:1.1b")
    temperature: float | None = Field(0.7, ge=0, le=2, description="Sampling temperature.")
    top_p: float | None = Field(1.0, ge=0, le=1, description="Nucleus sampling parameter.")
