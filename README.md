# AI Inference Study

**AI Inference Study** is a guide for backend engineers building production-grade AI inference systems.
It focuses on real-world deployments, serving frameworks, observability, and scaling patterns.

---

## Overview

This study explores how to turn trained AI models into reliable, maintainable inference services.
The documentation covers the full production lifecycle — from model usage to deployment and monitoring.

**Main chapters:**

1. **Fundamentals** – Model lifecycles, GPU and workload concepts
2. **Model Usage & Patterns** – Single models, pipelines, RAG, agentic flows
3. **Inference Servers** – Triton, vLLM, TorchServe, TGI, Ollama
4. **Architecture Patterns** – API layout, scaling, streaming, caching
5. **Deployment** – Docker, Kubernetes, Helm, CI/CD, model updates
6. **Observability** – Metrics, tracing, logging, alerting
7. **Projects** – End-to-end production-ready examples

---

## Small AI Projects

Each project represents a deployable pattern:

### [1. Image Classification API](https://github.com/grilled-pork-chop/ai-inference-study/tree/main/projects/01-image-classification)

* **Concepts:** Single model, Triton, batching
* **Stack:** FastAPI + Triton + ResNet50
* **Use Case:** Real-time product categorization
* **Highlights:** Dynamic batching, Prometheus metrics, sub-100ms latency

### 2. Streaming LLM Chat

* **Concepts:** vLLM, token streaming
* **Stack:** FastAPI + vLLM + Llama 3
* **Use Case:** Customer support chatbot
* **Highlights:** SSE streaming, Redis context, backpressure handling

### 3. RAG Document Q&A

* **Concepts:** Retrieval pipeline, embeddings
* **Stack:** FastAPI + Triton (embeddings) + vLLM + Milvus
* **Use Case:** Knowledge base search
* **Highlights:** Chunking, semantic retrieval, source citations

### 4. Batch Video Analysis

* **Concepts:** Async jobs, batch inference
* **Stack:** FastAPI + Celery + Redis + Triton
* **Use Case:** Content moderation
* **Highlights:** Job priority, progress tracking, webhooks

### 5. Multi-Model Ensemble

* **Concepts:** Parallel inference, aggregation
* **Stack:** FastAPI + Triton (3 models)
* **Use Case:** Fraud detection
* **Highlights:** Weighted voting, fallback handling, low latency

### 6. Agentic Email Assistant

* **Concepts:** Agents, tools, reasoning loops
* **Stack:** FastAPI + vLLM + LangGraph + Redis + PostgreSQL
* **Use Case:** Automated email triage
* **Highlights:** Persistent memory, structured reasoning logs

### 7. Cost-Optimized Inference

* **Concepts:** Quantization, caching, autoscaling
* **Stack:** FastAPI + ONNX Runtime + Redis + S3
* **Use Case:** Translation service
* **Highlights:** INT8 models, response caching, spot GPU scaling

### 8. Monitored CV Pipeline

* **Concepts:** Observability, tracing
* **Stack:** FastAPI + Triton + Prometheus + Grafana + Loki
* **Use Case:** OCR with SLA monitoring
* **Highlights:** Full tracing, GPU dashboards, latency alerts

### 9. Model A/B Testing Platform

* **Concepts:** Canary deployment, rollout control
* **Stack:** FastAPI + Triton + PostgreSQL
* **Use Case:** Sentiment model rollout
* **Highlights:** Traffic split, rollback logic, accuracy comparison

### 10. Edge Inference Gateway

* **Concepts:** Edge deployment, offline mode
* **Stack:** FastAPI + ONNX Runtime + SQLite
* **Use Case:** IoT image classification
* **Highlights:** Quantized models, local cache, low memory use

### 11. Secure Multi-Tenant Inference

* **Concepts:** Auth, rate limiting, isolation
* **Stack:** FastAPI + Keycloak + Triton + PostgreSQL
* **Use Case:** SaaS inference platform
* **Highlights:** OAuth2, per-tenant models, audit logs

### 12. Auto-Scaling LLM Service

* **Concepts:** HPA, GPU pooling
* **Stack:** Kubernetes + vLLM + KEDA + Prometheus
* **Use Case:** Elastic text generation
* **Highlights:** Queue-based scaling, GPU sharing, zero-idle cost

---

## Setup

**Requirements**

* Python 3.12+
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

**Install**

```bash
uv sync --frozen
```

**Run Docs**

```bash
uv run mkdocs serve --livereload --open
```

**Build**

```bash
uv run mkdocs build
```

---

## Goals

* Share clear production patterns for AI inference
* Compare serving frameworks by reliability, not benchmarks
* Provide ready-to-deploy project templates
* Encourage observability, versioning, and maintainability

---

Full documentation: [https://grilled-pork-chop.github.io/ai-inference-study/](https://grilled-pork-chop.github.io/ai-inference-study/)
