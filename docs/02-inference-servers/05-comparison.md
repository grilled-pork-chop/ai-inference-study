# Comparison & Decision Guide

> Choose the right inference server based on model type, size, and latency goals.

---

| Server        | Best For               | Frameworks               | Streaming | Scaling    | Notes                        |
| ------------- | ---------------------- | ------------------------ | --------- | ---------- | ---------------------------- |
| **Triton**    | Multi-model pipelines  | PyTorch, ONNX, TensorRT  | No        | Excellent  | Industry standard            |
| **vLLM**      | LLM chat & generation  | Transformers             | Yes       | Good       | OpenAI-style API             |
| **TGI**       | LLMs (HF ecosystem)    | Transformers             | Yes       | Medium     | Hugging Face optimized       |
| **Ollama**    | Local LLMs             | GGUF, GGML, Transformers | Yes       | Local-only | Simple, portable, private    |
| **Ray Serve** | Dynamic workflows      | Any (Python)             | Partial   | Excellent  | Autoscaling & orchestration  |
| **BentoML**   | Lightweight deployment | Any (Sklearn, PyTorch)   | No        | Medium     | Easy REST packaging          |
| **OVMS**      | CPU / Edge inference   | ONNX, TensorFlow, IR     | No        | Good       | Optimized for Intel hardware |

---

## Decision Guide

| Goal                      | Recommended Runtime         |
| ------------------------- | --------------------------- |
| Small model / REST API    | **Triton** or **BentoML**   |
| Text generation / chatbot | **vLLM** or **TGI**         |
| Local or on-prem LLM      | **Ollama**                  |
| Multi-stage pipeline      | **Triton** or **Ray Serve** |
| CPU or edge inference     | **OVMS**                    |
| Autoscaling workflows     | **Ray Serve + KEDA**        |

---

## 🚀 Emerging Servers

| Project              | Description                                         |
| -------------------- | --------------------------------------------------- |
| **SGLang**           | Modular, high-performance LLM runtime for streaming |
| **vLLM + Ray Serve** | Hybrid for large-scale distributed inference        |
| **LMDeploy**         | Efficient multi-GPU LLM runtime (Alibaba)           |

---

## Key Takeaways

* **TorchServe is deprecated — use Triton instead**
* Match runtime to **model type** and **scaling goals**
* Use **Ollama** for local/private setups, **vLLM** for scalable production
* Integrate with **FastAPI, Ray Serve, and Prometheus** for orchestration and observability
