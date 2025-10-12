import io

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw image bytes for ResNet50 inference.

    Args:
        image_bytes (bytes): Raw image data.

    Returns:
        np.ndarray: Array of shape (1, 3, 224, 224), FP32.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img_convert = img.convert("RGB").resize((224, 224))

    arr = np.asarray(img_convert, dtype=np.float32)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def postprocess_logits(logits: np.ndarray, class_names: list[str], top_k: int = 5) -> list[dict]:
    """Convert model logits into top-k predictions.

    Args:
        logits (np.ndarray): Model output tensor.
        class_names (list[str]): List of label names.
        top_k (int): Number of predictions to return.

    Returns:
        list[dict]: List of predictions.
    """
    logits = logits[0] if logits.ndim == 2 else logits
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    top_indices = np.argsort(probs)[-top_k:][::-1]

    return [
        {
            "class_id": int(idx),
            "class_name": class_names[idx] if idx < len(class_names) else f"class_{idx}",
            "confidence": float(probs[idx]),
        }
        for idx in top_indices
    ]
