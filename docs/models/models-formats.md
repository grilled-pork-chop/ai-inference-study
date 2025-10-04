# AI Model Formats

## Overview
AI models are stored in different formats — standardized ways to save model architecture, weights, and metadata. The format you choose impacts **performance, compatibility, and deployment**.

!!! tip "Think of formats like file types"
    Documents can be `.pdf`, `.docx`, `.txt`. AI models can be `.onnx`, `.pt`, `.safetensors`, etc.

---

## Common Model Formats

| Format      | Extension            | Framework          | Key Characteristics                       | Best For                     |
| ----------- | -------------------- | ------------------ | ----------------------------------------- | ---------------------------- |
| ONNX        | .onnx                | Framework-agnostic | Cross-platform, optimized inference       | Production, edge devices     |
| PyTorch     | .pt, .pth, .bin      | PyTorch            | Native training, includes optimizer state | PyTorch training & inference |
| TensorFlow  | .pb, .h5, SavedModel | TensorFlow/Keras   | Multiple formats, TF ecosystem            | TensorFlow apps              |
| SafeTensors | .safetensors         | Framework-agnostic | Fast, secure, memory-efficient            | HuggingFace, safe loading    |
| TorchScript | .pt, .pth            | PyTorch            | Optimized, serialized PyTorch             | Production PyTorch inference |
| TensorRT    | .plan, .engine       | NVIDIA             | GPU-optimized, platform-specific          | NVIDIA GPU inference         |
| Core ML     | .mlmodel, .mlpackage | Apple              | iOS/macOS optimized                       | Apple devices                |
| TFLite      | .tflite              | TensorFlow         | Mobile/embedded optimized                 | Mobile & edge                |
| OpenVINO    | .xml + .bin          | Intel              | CPU/VPU optimized                         | Intel hardware               |

---

## Detailed Format Breakdown

### ONNX (Open Neural Network Exchange)

**What it is**: Universal format for representing neural networks, enabling interoperability between frameworks.

**Extensions**: `.onnx`

**Key features**:

- Framework-agnostic (train in PyTorch, deploy anywhere)
- Optimized for inference (no training overhead)
- Wide hardware support (CPU, GPU, mobile, edge)
- Smaller file sizes than native formats
- Runtime optimization support

**Advantages**:

- ✅ Cross-platform compatibility
- ✅ Production-ready optimization
- ✅ Hardware acceleration support
- ✅ Extensive runtime support
- ✅ Model compression tools

**Limitations**:

- ❌ Export can fail for complex operations
- ❌ Some loss of flexibility vs native format
- ❌ Debugging harder than native format

!!! info "Use cases"
    - Production deployment across different platforms
    - Edge device inference
    - Model serving at scale
    - Cross-framework model sharing

**File size**: Typically 10-30% smaller than native formats

### PyTorch (.pt, .pth, .bin)

**What it is**: Native PyTorch model serialization format.

**Extensions**: `.pt`, `.pth`, `.bin`

**Key features**:

- Can save full model or just weights
- Includes optimizer state (for training)
- Dynamic computation graph
- Pythonic and flexible
- Easy debugging

**Save types**:

- **State dict only**: Just model weights (smaller, recommended)
- **Entire model**: Architecture + weights (larger, less portable)
- **Checkpoint**: Weights + optimizer + training state

**Advantages**:

- ✅ Native PyTorch support
- ✅ Easy to use and debug
- ✅ Saves training state
- ✅ Flexible and dynamic

**Limitations**:

- ❌ PyTorch-only (not portable)
- ❌ Larger file sizes
- ❌ Security risks (pickle-based)
- ❌ Slower loading

!!! info "Use cases"
    - PyTorch research and development
    - Model checkpointing during training
    - PyTorch-only deployment
    - Quick prototyping

**File size**: Typically largest format (uncompressed)

### SafeTensors

**What it is**: Modern, secure format designed for safe and fast model loading.

**Extensions**: `.safetensors`

**Key features**:

- Memory-mapped loading (instant)
- Zero-copy deserialization
- Secure (no code execution)
- Framework-agnostic
- Smaller than pickle-based formats

**Technical advantages**:

- Loads without unpickling (safe from attacks)
- Lazy loading (load only needed tensors)
- No Python overhead
- Deterministic file format

**Advantages**:

- ✅ Extremely fast loading
- ✅ Secure (no arbitrary code execution)
- ✅ Memory efficient
- ✅ Cross-framework compatible
- ✅ HuggingFace standard

**Limitations**:

- ❌ Weights only (no architecture)
- ❌ Requires architecture separately
- ❌ Newer format (less widespread support)

!!! info "Use cases"
    - HuggingFace model distribution
    - Large model loading (LLMs)
    - Production deployment
    - Secure model serving

**File size**: 5-15% smaller than PyTorch .bin

### TensorFlow (SavedModel, .pb, .h5)

**What it is**: TensorFlow's native formats for model serialization.

**Extensions**: `SavedModel/` (directory), `.pb`, `.h5`

**Format types**:

- **SavedModel**: Complete model + metadata (recommended)
- **Frozen Graph (.pb)**: Optimized for inference
- **HDF5 (.h5)**: Keras-specific, weights or full model

**Advantages**:

- ✅ TensorFlow ecosystem integration
- ✅ Complete model + metadata
- ✅ TensorFlow Serving ready
- ✅ Multiple optimization options

**Limitations**:

- ❌ TensorFlow-specific
- ❌ Larger file sizes (SavedModel)
- ❌ Complex format structure
- ❌ Version compatibility issues

!!! info "Use cases"
    - TensorFlow/Keras applications
    - TensorFlow Serving deployment
    - TensorFlow Lite conversion
    - Google Cloud deployment

**File size**: SavedModel typically 20-40% larger than raw weights

### TorchScript

**What it is**: Optimized, production-ready serialization of PyTorch models.

**Extensions**: `.pt`, `.pth`

**Key features**:

- Static graph (vs PyTorch's dynamic)
- C++ inference (no Python required)
- Optimization passes
- Mobile deployment support

**Creation methods**:

- **Tracing**: Record operations during execution
- **Scripting**: Analyze code directly

**Advantages**:

- ✅ Production-optimized
- ✅ C++ deployment (no Python)
- ✅ Mobile support
- ✅ Faster inference

**Limitations**:

- ❌ Less flexible than PyTorch
- ❌ Complex models may fail conversion
- ❌ Still PyTorch ecosystem

!!! info "Use cases"
    - Production PyTorch inference
    - Mobile deployment
    - C++ applications
    - Latency-critical systems

**File size**: Similar to PyTorch format

### TensorRT (.plan, .engine)

**What it is**: NVIDIA's high-performance inference optimizer and runtime.

**Extensions**: `.plan`, `.engine`

**Key features**:

- GPU-optimized inference
- Layer fusion and kernel auto-tuning
- Mixed precision (FP32/FP16/INT8)
- Platform-specific (compiled per GPU)

**Optimization techniques**:

- Precision calibration (INT8 quantization)
- Kernel auto-tuning
- Dynamic tensor memory
- Multi-stream execution

**Advantages**:

- ✅ Extreme GPU performance
- ✅ Automatic optimization
- ✅ Quantization support
- ✅ Production deployment features

**Limitations**:

- ❌ NVIDIA GPUs only
- ❌ Platform-specific (must rebuild per GPU)
- ❌ Complex setup
- ❌ Not portable

!!! info "Use cases"
    - NVIDIA GPU inference at scale
    - Real-time inference (autonomous vehicles)
    - Video processing
    - Edge inference (Jetson)

**File size**: Comparable to original model

### Core ML (.mlmodel, .mlpackage)
**What it is**: Apple's machine learning format for iOS, macOS, watchOS, tvOS.

**Extensions**: `.mlmodel` (older), `.mlpackage` (newer)

**Key features**:

- Apple Neural Engine optimization
- On-device inference
- Privacy-preserving (no cloud needed)
- Xcode integration

**Supported operations**:

- Neural networks (CNN, RNN, Transformer)
- Classical ML (trees, SVM)
- Custom layers

**Advantages**:

- ✅ Optimized for Apple hardware
- ✅ Neural Engine acceleration
- ✅ Integrated with Apple ecosystem
- ✅ Privacy-preserving

**Limitations**:

- ❌ Apple devices only
- ❌ Conversion can be lossy
- ❌ Limited operation support
- ❌ No training capability

!!! info "Use cases"
    - iOS/macOS apps
    - On-device ML
    - Privacy-sensitive applications
    - Apple ecosystem deployment

**File size**: Typically 10-20% larger than source

### TensorFlow Lite (.tflite)

**What it is**: Lightweight TensorFlow for mobile and embedded devices.

**Extensions**: `.tflite`

**Key features**:

- Small binary size (~300KB runtime)
- Fast inference on mobile/edge
- Quantization support (INT8, FP16)
- Hardware acceleration (GPU, NPU)

**Optimization levels**:

- Dynamic range quantization (weights only)
- Full integer quantization (weights + activations)
- Float16 quantization

**Advantages**:

- ✅ Extremely lightweight
- ✅ Mobile/edge optimized
- ✅ Aggressive quantization
- ✅ Cross-platform mobile

**Limitations**:

- ❌ Limited operation support
- ❌ Conversion can be complex
- ❌ Inference-only
- ❌ Some accuracy loss with quantization

!!! info "Use cases"
    - Mobile apps (Android/iOS)
    - Embedded devices (Raspberry Pi)
    - IoT devices
    - Resource-constrained environments

**File size**: Typically 50-75% smaller with quantization

### OpenVINO (.xml + .bin)

**What it is**: Intel's toolkit for optimized inference on Intel hardware.

**Extensions**: `.xml` (architecture) + `.bin` (weights)

**Key features**:

- CPU/integrated GPU/VPU optimization
- Intel hardware acceleration
- Model optimization toolkit
- Heterogeneous execution

**Supported hardware**:

- Intel CPUs (x86)
- Intel integrated GPUs
- Intel Movidius VPUs
- Intel FPGAs

**Advantages**:

- ✅ Excellent CPU performance
- ✅ Intel hardware optimized
- ✅ Broad operation support
- ✅ Heterogeneous deployment

**Limitations**:

- ❌ Best on Intel hardware
- ❌ Two-file format
- ❌ Complex ecosystem
- ❌ Less common than ONNX

!!! info "Use cases"
    - Intel CPU inference
    - Edge computing (VPU)
    - Industrial applications
    - Retail/surveillance systems

**File size**: Comparable to original model

## Format Selection Guide

### By Use Case

| Use Case                    | Recommended Format | Alternative                      |
| --------------------------- | ------------------ | -------------------------------- |
| Training (PyTorch)          | .pt (state dict)   | .safetensors                     |
| Training (TensorFlow)       | SavedModel, .h5    | -                                |
| Production (cross-platform) | ONNX               | SafeTensors                      |
| Production (PyTorch only)   | TorchScript        | ONNX                             |
| Production (NVIDIA GPU)     | TensorRT           | ONNX                             |
| Mobile (iOS/macOS)          | Core ML            | TFLite                           |
| Mobile (Android)            | TFLite             | ONNX Mobile                      |
| Edge (general)              | ONNX, TFLite       | OpenVINO                         |
| Edge (Intel)                | OpenVINO           | ONNX                             |
| Edge (NVIDIA)               | TensorRT           | ONNX                             |
| HuggingFace distribution    | SafeTensors        | Model sharing: ONNX, SafeTensors |

---

### By Priority

| Priority               | Format Choice                                |
| ---------------------- | -------------------------------------------- |
| Speed (loading)        | SafeTensors > ONNX > PyTorch                 |
| Speed (inference, GPU) | TensorRT > ONNX > TorchScript > PyTorch      |
| Speed (inference, CPU) | OpenVINO > ONNX > TensorRT                   |
| Portability            | ONNX > SafeTensors > others                  |
| Security               | SafeTensors > ONNX > PyTorch                 |
| File size              | TFLite (INT8) > ONNX > SafeTensors > PyTorch |
| Flexibility            | PyTorch > TensorFlow > others                |
| Ecosystem support      | PyTorch, TensorFlow > ONNX > others          |

---

### Conversion Paths

```
PyTorch (.pt) ─┬→ ONNX ─┬→ TensorRT (NVIDIA)
               │        ├→ OpenVINO (Intel)
               │        └→ TFLite
               ├→ TorchScript
               ├→ SafeTensors
               └→ Core ML

TensorFlow ────┬→ ONNX ─→ [same as above]
               ├→ TFLite
               └→ Core ML
```
