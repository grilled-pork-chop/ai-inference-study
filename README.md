# AI Inference Study (Backend Developer’s View)

A practical guide to **putting AI models into production**:  
covering models, inference interfaces, serving frameworks, monitoring, and benchmarking.  
This repository combines **documentation (with MkDocs)** and **hands-on projects** to help backend developers understand and test modern inference stacks.

## Documentation

The full documentation is built with [MkDocs](https://www.mkdocs.org/) using the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme and hosted on **GitHub Pages**.

It covers:
- **AI Models** – types (LLMs, vision, speech, multimodal) and specializations  
- **Interfaces** – REST, gRPC, WebSocket, streaming APIs, MCP.  
- **Inference Servers** – Triton, TorchServe, BentoML, KServe, vLLM, etc.  
- **Projects** – concrete stacks for benchmarking  
- **Benchmarks** – methodology, metrics, and results  
- **Production Concerns** – scaling, monitoring, security  

## Installation

### Prerequisites
- Python **3.12+**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management

### Setup
```bash
uv sync --frozen
```

### Run Docs Locally

```bash
uv run mkdocs serve --livereload --open
```

### Build Static Site

```bash
uv run mkdocs build
```

## Benchmark Groups

This study is organized into groups of common AI workloads.
Each group is benchmarked across multiple serving stacks and interfaces to provide fair and comparable results.

### Group 1: Image Classification (ResNet50)
- Models: ResNet50 (ONNX/PyTorch)
- Focus: Latency, throughput, GPU utilization
- Stacks: FastAPI + ONNXRuntime, Triton, TorchServe

### Group 2: Text Generation (LLMs)
- Models: GPT-2 small, OPT-125M
- Focus: Time-to-first-token, streaming latency, throughput
- Stacks: FastAPI + PyTorch, vLLM, TGI

### Group 3: Embeddings
- Models: MiniLM, Sentence-BERT
- Focus: High-throughput embedding generation
- Stacks: FastAPI + PyTorch, Triton, TEI (Text Embedding Inference)

### Group 4: Multimodal Inference
- Models: CLIP (text-image similarity), LLaVA (vision + language assistant)
- Focus: Image-text encoding, end-to-end multimodal reasoning
- Stacks: FastAPI + PyTorch, Triton ensembles, vLLM (for LLaVA)

### Group 5: Speech (STT/TTS)
- Models: Whisper (speech-to-text), Coqui TTS / OpenVoice (text-to-speech)
- Focus: Real-time factor, streaming latency, throughput
- Stacks: FastAPI + Whisper, Triton (Whisper ONNX/TensorRT), NVIDIA Riva

### Group 6: Batch Serving
- Models: Any model that benefits from batching (LLMs, embeddings)
- Focus: Dynamic batching, throughput scaling, tail latency
- Stacks: FastAPI + custom batching, Triton, vLLM, TGI, TorchServe

## Projects

Each group includes comparable projects where only one variable (stack) changes.
This enables apples-to-apples benchmarking.

1. FastAPI + PyTorch/ONNX Stack
2. Triton Inference Server Stack
3. TorchServe Stack
4. vLLM Stack
5. TGI (Text Generation Inference) Stack
6. TEI (Text Embedding Inference) Stack

## Benchmarking Methodology

- Load generation: `k6`, `locust`, or Python async benchmarking tools
- Metrics: Latency (p50, p95, p99), throughput, time-to-first-token, GPU utilization
- Monitoring: Prometheus + Grafana
- Result storage: JSON logs processed into visual reports with Pandas and Matplotlib

## Roadmap

- Add quantization benchmarks (FP32, FP16, INT8)
- Add multi-GPU scaling (Triton, vLLM, TGI)
- Add Kubernetes deployment (KServe, BentoML)
- Compare cost-efficiency (throughput per $/hour on cloud GPUs)