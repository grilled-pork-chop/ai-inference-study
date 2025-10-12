"""
Export a pretrained ResNet50 model to ONNX format for Triton Inference Server.

This script:
  - Loads a pretrained torchvision ResNet50
  - Exports it to ONNX (opset 17)
  - Creates a Triton-ready model repository structure:
        model_repository/
            resnet50/
                1/
                    model.onnx
                config.pbtxt
"""

import os
from pathlib import Path

import torch
from torchvision import models

MODEL_NAME = "resnet50"
MODEL_VERSION = "1"
MODEL_REPO_DIR = "model"
MODEL_PATH = os.path.join(MODEL_REPO_DIR, MODEL_NAME, MODEL_VERSION, "model.onnx")


def export_resnet50_to_onnx() -> None:
    """Export pretrained ResNet50 model to ONNX format for Triton."""
    print("🔹 Loading pretrained ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print(f"🔹 Exporting model to {MODEL_PATH}...")

    torch.onnx.export(
        model,
        dummy_input,
        MODEL_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model exported successfully ({size_mb:.2f} MB)")

    _write_triton_config()


def _write_triton_config() -> None:
    """Write Triton Inference Server model config.pbtxt."""
    config_path = os.path.join(MODEL_REPO_DIR, MODEL_NAME, "config.pbtxt")
    print(f"🔹 Generating Triton config: {config_path}")

    config = f"""
name: "{MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

# Uncomment for GPU
# instance_group [
#   {{
#     count: 1
#     kind: KIND_GPU
#     gpus: [ 0 ]
#   }}
# ]

dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}}
"""

    with Path.open(config_path, "w", encoding="utf-8") as f:
        f.write(config.strip() + "\n")

    print("✅ Triton config.pbtxt created.")


if __name__ == "__main__":
    try:
        export_resnet50_to_onnx()
    except Exception as e:
        print(f"❌ Error exporting model: {e}")
        print("\n💡 Make sure you have the required packages installed:")
        print("   pip install torch torchvision onnx")
        raise SystemExit(1)
