# Glossary

> Glossary and key concepts for AI, ML, and LLM systems. Covers model domains, optimizations, deployment, scaling, and observability.

---

## 1. Model Domains & Subdomains

| Domain                        | Subdomain / Example                                   | Description                                          |
| ----------------------------- | ----------------------------------------------------- | ---------------------------------------------------- |
| **Computer Vision**           | Classification, Detection, Segmentation, Generation   | Models that process images/videos                    |
| **NLP**                       | Text Classification, Summarization, Translation, LLMs | Text understanding and generation                    |
| **Speech / Audio**            | ASR, TTS, Speaker ID, Audio Generation                | Convert speech to text or vice versa, audio analysis |
| **Reinforcement Learning**    | Policy Gradients, Q-Learning, Actor-Critic            | Agents learn from environment feedback               |
| **Graph ML**                  | GCN, Graph Attention Networks                         | Models on graph-structured data                      |
| **Recommendation Systems**    | Collaborative Filtering, Embedding Models             | Predict user preferences                             |
| **Time Series / Forecasting** | ARIMA, LSTM, Transformer-based                        | Predict future values from sequences                 |

---

## 2. Model Formats

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

## 3. Training Optimizations

| Term                         | Description                                                             |
| ---------------------------- | ----------------------------------------------------------------------- |
| **Gradient Clipping**        | Limit gradient magnitude to avoid exploding gradients                   |
| **Learning Rate Scheduling** | Adjust learning rate during training (warmup, cosine decay, step decay) |
| **Weight Decay**             | Regularization to prevent overfitting                                   |
| **Dropout**                  | Randomly deactivate neurons to improve generalization                   |
| **Label Smoothing**          | Prevent overconfident predictions by adjusting target labels            |
| **Curriculum Learning**      | Train on easy examples first, then harder examples                      |
| **Mixed Precision / AMP**    | Use FP16 + FP32 for speed and memory efficiency                         |
| **Gradient Accumulation**    | Simulate larger batch sizes by accumulating gradients                   |
| **Checkpointing**            | Save model weights and optimizer state periodically                     |

---

## 4. Model Compression & Deployment

| Term                           | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| **Pruning**                    | Remove unimportant weights to reduce model size              |
| **Quantization**               | Reduce numeric precision (FP32 → FP16 / INT8)                |
| **Distillation**               | Train a smaller “student” model from a larger “teacher”      |
| **Knowledge Transfer**         | Transfer learned representations across tasks/models         |
| **Low-Rank Approximation**     | Reduce weight matrix rank to save memory and speed inference |
| **Sparse Models**              | Only store/compute non-zero weights                          |
| **LoRA (Low-Rank Adaptation)** | Fine-tuning large models with fewer parameters               |
| **Adapters**                   | Small trainable layers added to pre-trained models           |

---

## 5. Inference & Serving Optimizations

| Term                            | Description                                        |
| ------------------------------- | -------------------------------------------------- |
| **Batching**                    | Combine requests for GPU efficiency                |
| **Caching**                     | Store intermediate outputs like embeddings         |
| **Operator Fusion**             | Merge layers/ops to reduce kernel calls            |
| **Pipeline Parallelism**        | Split model across devices for memory-bound models |
| **Model Sharding**              | Partition large models across GPUs or nodes        |
| **Async / Streaming Inference** | Return partial outputs for responsiveness          |
| **Latency Class**               | Categorize requests for routing (fast vs slow)     |

---

## 6. Scaling & System-Level

| Term                      | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| **Horizontal Scaling**    | Add replicas of service to increase throughput                 |
| **Vertical Scaling**      | Increase hardware resources on a single node                   |
| **GPU Pooling**           | Share GPUs among models safely                                 |
| **Elastic / Autoscaling** | Dynamic scaling based on load or metrics                       |
| **Load Balancing**        | Distribute requests across replicas                            |
| **Backpressure**          | Limit or queue requests to avoid overload                      |
| **Throughput vs Latency** | Tradeoff: larger batches increase throughput but latency grows |

---

## 7. Observability & Monitoring

| Term                     | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| **SLO / SLA**            | Service level objectives / agreements (latency, availability) |
| **Telemetry**            | Data collected from the system for analysis                   |
| **Metrics**              | Numeric measurements (latency, throughput, GPU usage)         |
| **Traces**               | Request path and timing across multiple services              |
| **Logging**              | Structured records of events, errors, and operations          |
| **Alerts**               | Notification of anomalies or thresholds breached              |
| **GPU Utilization**      | Real usage of hardware resources                              |
| **Memory Fragmentation** | Inefficient memory usage leading to OOM risks                 |

---

## 8. LLM & Advanced Concepts

| Term                                        | Description                                                |
| ------------------------------------------- | ---------------------------------------------------------- |
| **Context Window**                          | Number of tokens the model can attend to at once           |
| **KV Cache**                                | Memory for previously generated tokens for faster decoding |
| **Few-Shot / Zero-Shot Learning**           | Model adapts to new tasks with few or no examples          |
| **Fine-Tuning / Instruction Tuning / RLHF** | Techniques to specialize or align models                   |
| **Mixture-of-Experts (MoE)**                | Conditional execution of model parts for efficiency        |
| **Attention Mechanisms**                    | Core of Transformers; can be sparse or efficient           |
| **Semantic Memory / Vector DB**             | Store embeddings for retrieval-augmented generation        |
| **Adapter Modules**                         | Lightweight trainable layers for pre-trained models        |

---

## 9. Low-Level Optimization Techniques

| Term                         | Description                                             |
| ---------------------------- | ------------------------------------------------------- |
| **Gradient Checkpointing**   | Save memory by recomputing activations during backprop  |
| **Tensor Parallelism**       | Split computations across multiple GPUs                 |
| **Operator-level Fusion**    | Merge small ops to reduce kernel launches               |
| **Mixed Precision Training** | FP16/FP32 mixed precision to speed up training          |
| **Activation Offloading**    | Move intermediate activations to CPU to save GPU memory |
| **Memory Profiling**         | Monitor memory usage to avoid OOM                       |
