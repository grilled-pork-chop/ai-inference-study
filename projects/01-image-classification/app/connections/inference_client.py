"""Async Triton Inference Server client wrapper."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput, InferResult
from tritonclient.utils import InferenceServerException

from app.core.config import settings

logger = logging.getLogger(__name__)

class InferenceClient:
    """Async wrapper around Triton HTTP client."""

    def __init__(self) -> None:
        """Initialize Triton HTTP async client."""
        self.client = InferenceServerClient(
            url=f"{settings.TRITON_HOST}:{settings.TRITON_HTTP_PORT}", verbose=False,
        )
        self.model_name = settings.MODEL_NAME

    async def infer(
        self,
        inputs: list[InferInput],
        outputs: list[InferRequestedOutput],
    ) -> InferResult:
        """Perform inference.

        Args:
            inputs (List[InferInput]): Model inputs.
            outputs (List[InferRequestedOutput]): Requested outputs.

        Returns:
            InferResult: Inference response.

        Raises:
            RuntimeError: If inference fails.
        """
        try:
            return await self.client.async_infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
            )
        except InferenceServerException as e:
            msg = f"Triton inference failed: {e}"
            raise RuntimeError(msg) from e

    async def is_server_ready(self) -> bool:
        """Check if Triton server is ready.

        Returns:
            bool: True if ready, False otherwise.
        """
        return await self.client.is_server_ready()

    async def is_model_ready(self) -> bool:
        """Check if model is loaded and ready.

        Returns:
            bool: True if model is ready.
        """
        return self.client.is_model_ready(self.model_name)

    async def connect(self, connection_timeout: int = 30, interval: float = 1.0) -> None:
        """Wait until Triton server and model are ready.

        Args:
            connection_timeout: Max seconds to wait before failing.
            interval: Seconds between checks.

        Raises:
            TimeoutError: If the server or model is not ready in time.
        """
        start = datetime.now(UTC)
        deadline = start + timedelta(seconds=connection_timeout)

        logger.info("Connecting to Triton at %s:%d", settings.TRITON_HOST, settings.TRITON_HTTP_PORT)

        while datetime.now(UTC) < deadline:
            if await self.is_server_ready() and await self.is_model_ready():
                logger.info("Inference server and model is ready.")
                return

            logger.debug("Ingference server not ready yet.")
            await asyncio.sleep(interval)

        msg = (
            f"Timeout waiting for inference server/model to become ready "
            f"after {connection_timeout} seconds."
        )
        raise TimeoutError(msg)

    async def disconnect(self) -> None:
        """Disconnect to inference server."""
        self.client.close()
