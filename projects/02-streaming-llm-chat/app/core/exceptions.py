"""Custom exceptions for the LLM Chat API."""


class LLMError(Exception):
    """Base exception for all custom LLM Chat API errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMConnectionError(LLMError):
    """Raised when there's an issue connecting to or communicating with the LLM server."""

    def __init__(
        self,
        message: str = "Could not connect to LLM inference server.",
        status_code: int = 503,
    ) -> None:
        super().__init__(message, status_code)


class LLMGenerationError(LLMError):
    """Raised when LLM fails to generate a response."""

    def __init__(
        self, message: str = "LLM generation failed.", status_code: int = 500,
    ) -> None:
        super().__init__(message, status_code)


class RedisConnectionError(LLMError):
    """Raised when there's an issue connecting to or communicating with Redis."""

    def __init__(
        self, message: str = "Could not connect to Redis.", status_code: int = 503,
    ) -> None:
        super().__init__(message, status_code)


class ConversationHistoryError(LLMError):
    """Raised when there's an issue saving or retrieving conversation history."""

    def __init__(
        self,
        message: str = "Failed to manage conversation history.",
        status_code: int = 500,
    ) -> None:
        super().__init__(message, status_code)


class LLMServiceError(LLMError):
    """A general service-level error for LLM operations."""

    def __init__(
        self,
        message: str = "An unexpected error occurred in the LLM service.",
        status_code: int = 500,
    ) -> None:
        super().__init__(message, status_code)
