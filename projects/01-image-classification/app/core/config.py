from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    ##### Global settings ######

    LOG_LEVEL: str = "DEBUG"
    CORS_ORIGINS: str = "*"

    ##### Application settings ######

    APP_TITLE: str = "IMAGE CLASSIFICATION API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = (
        "A REST API for real-time image classification using "
        "a pre-trained ResNet50 model served via Triton Inference Server."
    )
    APP_HOST: str = "0.0.0.0"  # noqa: S104
    APP_PORT: int = 8080
    APP_WORKERS: int = 1

    ##### Services settings ######
    TRITON_HOST: str = "triton"
    TRITON_GRPC_PORT: int = 8001
    MODEL_NAME: str = "resnet50"
    MAX_IMAGE_SIZE: int = 5_000_000
    ALLOWED_EXTENSIONS: list[str] = ["jpg", "jpeg", "png", "webp"]

settings = Settings()
