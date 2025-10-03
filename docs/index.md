---
hide:
  navigation: true
  toc: true
---

# AI Inference Study

A comprehensive guide to deploying AI models in production environments, covering everything from model formats to serving infrastructure and real-world benchmark projects.

## 🎯 Overview

This documentation provides practical guidance for backend engineers and MLOps practitioners on:

- **Models**: Understanding different AI domains, architectures, and formats for production deployment
- **Inference**: Comparing serving interfaces (REST, gRPC, WebSocket) and inference servers (Triton, TorchServe, vLLM)
- **Projects**: Three complete benchmark stacks with performance comparisons and deployment guides

## 🚀 Quick Start

1. **Choose your model domain**: [Computer Vision](models/models-domains.md#computer-vision), [NLP/LLMs](models/models-domains.md#nlp-large-language-models), [Speech](models/models-domains.md#speech-audio-processing), or [Multimodal](models/models-domains.md#multimodal-ai)
2. **Select model format**: [ONNX, TorchScript, SavedModel](models/models-formats.md) based on your infrastructure
3. **Pick serving interface**: [REST, gRPC, WebSocket](inference/inference-interfaces.md) for your latency requirements
4. **Choose inference server**: [Triton, TorchServe, vLLM](inference/inference-servers.md) for your scale and complexity needs
5. **Deploy with examples**: Use our [benchmark projects](benchmarks/index.md) as starting templates

## 📊 Benchmark Groups

TODO

## 🏗️ Architecture Philosophy

Our approach focuses on **production-ready solutions** with:

- **Performance**: Detailed benchmarks with real hardware metrics
- **Scalability**: Kubernetes-native deployments with auto-scaling
- **Observability**: Prometheus/Grafana monitoring out-of-the-box
- **Reliability**: Health checks, circuit breakers, and graceful degradation

## 📚 Navigation

### [Models](models/index.md)
- [Domains](models/models-domains.md): Computer Vision, NLP/LLMs, Speech, Multimodal
- [Formats](models/models-formats.md): ONNX, TorchScript, SavedModel, GGUF, TensorRT

### [Inference](inference/index.md)
- [Interfaces](inference/inference-interfaces.md): REST, gRPC, WebSocket, SSE comparison
- [Servers](inference/inference-servers.md): Triton, TorchServe, vLLM, KServe analysis

### [Benchmarks](benchmarks/index.md)
- [Image Classification](benchmarks/group1-image-classification.md): Measure latency and throughput for vision models on single images and batched inputs.
- [Text Generation](benchmarks/group2-text-generation.md): Benchmark large language models (LLMs) for text completion and conversational tasks.
- [Embeddings](benchmarks/group3-embeddings.md): Evaluate throughput and scalability for generating vector embeddings.