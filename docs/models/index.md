# Models

Understanding AI models is crucial for production deployment decisions. This section covers the four major domains of AI models and their production characteristics.

## 🎯 Quick Navigation

- **[Domains](models-domains.md)**: Computer Vision, NLP/LLMs, Speech/Audio, Multimodal AI
- **[Formats](models-formats.md)**: ONNX, TorchScript, SavedModel, GGUF, TensorRT, and more

## 📊 Model Selection Matrix

| Domain              | Best Models                  | Typical Latency | Hardware Needs   | Production Complexity |
| ------------------- | ---------------------------- | --------------- | ---------------- | --------------------- |
| **Computer Vision** | YOLOv8, ResNet, EfficientNet | 1-50ms          | GPU recommended  | Medium                |
| **NLP/LLMs**        | BERT, GPT, LLaMA             | 10ms-5s         | GPU/TPU required | High                  |
| **Speech/Audio**    | Whisper, Wav2Vec2            | 30-200ms        | CPU/GPU hybrid   | Medium                |
| **Multimodal**      | CLIP, GPT-4V, Flamingo       | 50ms-10s        | High-end GPU     | Very High             |

## 🚀 Production Readiness Checklist

### Model Characteristics
- [ ] **Latency**: Meets SLA requirements (p95 < target)
- [ ] **Throughput**: Handles expected concurrent requests  
- [ ] **Accuracy**: Meets business requirements
- [ ] **Size**: Fits in target deployment environment

### Format Compatibility
- [ ] **Framework**: Compatible with serving infrastructure
- [ ] **Optimization**: Supports quantization if needed
- [ ] **Portability**: Can deploy across environments
- [ ] **Versioning**: Supports model updates and rollbacks

### Operational Requirements
- [ ] **Monitoring**: Integrates with observability stack
- [ ] **Scaling**: Supports horizontal/vertical scaling
- [ ] **Security**: Meets compliance requirements
- [ ] **Dependencies**: Minimal and well-documented

## 🎪 Common Pitfalls

**Choosing the Wrong Format**
- Using research formats (`.pkl`) in production
- Ignoring optimization opportunities (TensorRT, quantization)

**Underestimating Hardware**
- Not accounting for concurrent requests
- Ignoring memory bandwidth limitations

**Overlooking Operational Concerns**
- No model versioning strategy
- Missing monitoring and alerting
