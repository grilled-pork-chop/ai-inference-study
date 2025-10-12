"""Business logic for image classification."""

import logging

from tritonclient.grpc.aio import InferInput, InferRequestedOutput

from app.connections.inference_client import InferenceClient
from app.pipelines.classification_pipeline import postprocess_logits, preprocess_image

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

    async def classify(self, image_data: bytes, top_k: int = 5) -> tuple[list[dict], float]:
        """Classify image array using Triton server.

        Args:
            image_data: Raw image bytes
            top_k (int, optional): Number of top predictions. Defaults to 5.

        Returns:
            Tuple[List[dict], float]: List of predictions and inference time (ms).
        """
        img_array = preprocess_image(image_data)

        inputs = [InferInput("input", img_array.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_array)
        outputs = [InferRequestedOutput("output")]
        response = await self.inference_client.infer(inputs, outputs)

        logits = response.as_numpy("output")
        return postprocess_logits(logits, self.imagenet_classes, top_k)

