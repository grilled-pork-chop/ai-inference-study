# Model Quantization

## Overview
Quantization reduces model size and speeds up inference by using lower-precision numerical representations for weights and activations. Instead of 32-bit floating-point numbers, models can use 16-bit, 8-bit, or even lower precision.  

!!! tip "Think of quantization as compression"
    Like compressing a 4K video to 1080p — you trade some quality for smaller file size and faster inference.

### Benefits

| Benefit            | Impact                                        |
| ------------------ | --------------------------------------------- |
| Smaller file size  | 50-75% reduction (FP16), 75% reduction (INT8) |
| Faster inference   | 2-4x speedup typical                          |
| Lower memory usage | Deploy larger models on same hardware         |
| Reduced bandwidth  | Faster downloads, cheaper cloud storage       |
| Energy efficiency  | Critical for mobile/edge devices              |

**Trade-off:** Slight accuracy loss (typically 1–3% on well-implemented quantization)

---

## Quantization Types

### By Precision

| Type           | Bits | Size vs FP32 | Typical Accuracy Loss | Use Case                                |
| -------------- | ---- | ------------ | --------------------- | --------------------------------------- |
| FP32           | 32   | 100%         | 0%                    | Training, high-precision inference      |
| FP16           | 16   | 50%          | ~0.1%                 | GPU inference, mixed precision training |
| INT8           | 8    | 25%          | 1–3%                  | Production inference, edge devices      |
| INT4           | 4    | 12.5%        | 3–5%                  | Large models, extreme compression       |
| Binary/Ternary | 1-2  | 3–6%         | 5–10%                 | Research, extreme edge                  |

### By Method

- **Post-Training Quantization (PTQ):** No retraining; fast; minor accuracy loss.
- **Quantization-Aware Training (QAT):** Simulate quantization during training; minimal accuracy loss (<1%); requires retraining.
- **Mixed Precision:** Different precisions per layer; balance size, speed, and accuracy; hardware-dependent.

---

## Framework Notes

| Framework   | Methods                               | Output Format          |
| ----------- | ------------------------------------- | ---------------------- |
| PyTorch     | Dynamic, Static, QAT                  | .pt, ONNX, TorchScript |
| TensorFlow  | PTQ, Full INT8                        | .tflite                |
| ONNX        | Dynamic, Static, QAT (via conversion) | .onnx                  |
| HuggingFace | GPTQ, AWQ, bitsandbytes               | INT4/INT8 support      |

---

## Quantization Impact by Model Type

| Model Type         | Tolerance   | Recommended | Notes                      |
| ------------------ | ----------- | ----------- | -------------------------- |
| CNNs (Image)       | High        | INT8        | Minimal accuracy loss      |
| Object Detection   | Medium-High | INT8        | May need calibration       |
| Transformers (NLP) | Medium      | INT8, INT4  | Attention layers sensitive |
| LLMs               | Medium      | INT4, INT8  | Use GPTQ/AWQ for INT4      |
| Speech Recognition | High        | INT8        | Works well                 |
| Embeddings         | Low-Medium  | FP16, INT8  | Affects similarity scores  |
| Generative Models  | Low-Medium  | FP16, INT8  | Quality-sensitive          |

---

## Hardware Support

| Hardware                 | Best Precision | Notes                   |
| ------------------------ | -------------- | ----------------------- |
| NVIDIA GPU (recent)      | FP16, INT8     | TensorRT optimized      |
| NVIDIA GPU (older)       | FP32, FP16     | Limited INT8 support    |
| Apple Silicon (M1/M2/M3) | FP16, INT8     | Neural Engine optimized |
| Intel CPU                | INT8           | VNNI instructions       |
| ARM CPU                  | INT8           | NEON instructions       |
| Mobile GPU               | FP16, INT8     | Varies by chipset       |
| TPU                      | BF16, INT8     | Google Cloud TPU        |

---

## Selection Guide

| Precision | Deployment          | Notes                                     |
| --------- | ------------------- | ----------------------------------------- |
| FP16      | GPU                 | 2x speedup, 50% size reduction            |
| INT8      | CPU / Edge / Mobile | 4x speedup, 75% size reduction, <2% loss  |
| INT4      | Huge LLMs           | Extreme compression, GPTQ/AWQ recommended |
| QAT       | Production          | Accuracy-critical, retraining feasible    |
| PTQ       | Quick experiments   | FP16/INT8 targets, no retraining needed   |

---

## Example File Size Impact (7B LLM)

| Precision   | Size   | Reduction | Accuracy |
| ----------- | ------ | --------- | -------- |
| FP32        | 28 GB  | 0%        | 100%     |
| FP16        | 14 GB  | 50%       | 99.9%    |
| INT8        | 7 GB   | 75%       | 98–99%   |
| INT4 (GPTQ) | 3.5 GB | 87.5%     | 96–98%   |

---

## Visualization: Quantization Workflow
```
Full Precision Model (FP32)
├─ Post-Training Quantization (PTQ)
│  ├─ FP16 Model → GPU Deployment
│  └─ INT8 Model → Edge/CPU Deployment
├─ Quantization-Aware Training (QAT)
│  ├─ FP16 Model → GPU Deployment
│  └─ INT8 Model → Edge/CPU Deployment
└─ Extreme Compression INT4 → Large LLM Deployment
```