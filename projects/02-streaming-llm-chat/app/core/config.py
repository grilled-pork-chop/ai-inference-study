from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ##### Global settings ######

    LOG_LEVEL: str = Field(
        "INFO",
        description="Minimum log level for the application (e.g., DEBUG, INFO, WARNING, ERROR).",
    )
    CORS_ORIGINS: str = Field(
        "*",
        description="Comma-separated list of allowed CORS origins, or '*' for all origins.",
    )


    ##### Application settings ######

    APP_TITLE: str = Field(
        "STREAMING LLM CHAT API",
        description="The title of the FastAPI application, displayed in documentation.",
    )
    APP_VERSION: str = Field(
        "0.1.0",
        description="The version of the FastAPI application.",
    )
    APP_DESCRIPTION: str = Field(
        (
            "A REST API for streaming LLM chat responses using "
            "vLLM and Redis for conversation context."
        ),
        description="A detailed description of the FastAPI application.",
    )
    APP_HOST: str = Field(
        "0.0.0.0",  # noqa: S104
        description="The host IP address on which the FastAPI application will listen.",
    )
    APP_PORT: int = Field(
        8080,
        description="The port number on which the FastAPI application will listen.",
    )

    ##### Services settings ######
    LLM_URL: str = Field("http://localhost:8000/v1", description="URL of the LLM inference server.")


    REDIS_URL: str = Field("redis://localhost:6379", description="URL of the Redis server.")
    CONVERSATION_TTL_SECONDS: int = Field(
        3600,
        description="Time-to-live (in seconds) for conversation history in Redis.",
    )

settings = Settings()
