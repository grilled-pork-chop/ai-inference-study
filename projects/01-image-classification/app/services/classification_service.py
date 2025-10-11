"""Business logic for image classification."""

import logging

import numpy as np
from tritonclient.http import InferInput, InferRequestedOutput

from app.connections.inference_client import InferenceClient

logger = logging.getLogger(__name__)

class ClassificationService:
    """Service for handling image classification requests."""

    def __init__(self, inference_client: InferenceClient, imagenet_classes: list[str]) -> None:
        """
        Args:
            inference_client (InferenceClient): Inference client.
            imagenet_classes (List[str]): ImageNet class labels.
        """
        self.inference_client = inference_client
        self.imagenet_classes = imagenet_classes

    async def classify(self, img_array: np.ndarray, top_k: int = 5) -> tuple[list[dict], float]:
        """Classify image array using Triton server.

        Args:
            img_array (np.ndarray): Preprocessed image array.
            top_k (int, optional): Number of top predictions. Defaults to 5.

        Returns:
            Tuple[List[dict], float]: List of predictions and inference time (ms).

        Raises:
            Exception: Propagates any Triton or service errors.
        """
        inputs = [InferInput("input", img_array.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_array)
        outputs = [InferRequestedOutput("output")]

        response = await self.triton_client.async_infer(inputs, outputs)
        logits = response.as_numpy("output")[0]

        top_indices = np.argsort(logits)[-top_k:][::-1]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return [
            {
                "class_name": self.imagenet_classes[idx] if idx < len(self.imagenet_classes) else f"class_{idx}",
                "confidence": float(probs[idx]),
                "class_id": int(idx),
            }
            for idx in top_indices
        ]

