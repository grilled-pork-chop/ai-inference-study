---
hide:
  navigation: true
---

# Small AI Project Examples

## 1. 📸 Image Classification API
**Concepts:** Single model, Triton, batching, health checks  
**Stack:** FastAPI + Triton + ResNet50  
**Use Case:** Real-time product categorization for e-commerce  
**Key Features:**

- Dynamic batching for efficiency
- Prometheus metrics integration
- Health/readiness probes
- Sub-100ms latency

---

## 2. 💬 Streaming LLM Chat
**Concepts:** vLLM, WebSocket streaming, token generation  
**Stack:** FastAPI + vLLM + Llama 3  
**Use Case:** Customer support chatbot  
**Key Features:**

- Server-Sent Events for streaming
- Conversation context in Redis
- Token-by-token response
- Backpressure handling

---

## 3. 📚 RAG Document Q&A
**Concepts:** Pipeline, embeddings, vector DB, retrieval  
**Stack:** FastAPI + Triton (embeddings) + vLLM + Milvus  
**Use Case:** Internal knowledge base query system  
**Key Features:**

- Document chunking & indexing
- Semantic search with vector DB
- Context injection for LLM
- Source citation tracking

---

## 4. 🎥 Batch Video Analysis
**Concepts:** Async jobs, Celery, queuing, batch inference  
**Stack:** FastAPI + Celery + Redis + Triton (CV models)  
**Use Case:** Content moderation pipeline  
**Key Features:**

- Job queue with priority
- Frame extraction & batching
- Progress tracking via Redis
- Webhook notifications

---

## 5. 🔀 Multi-Model Ensemble
**Concepts:** Ensemble, parallel inference, aggregation  
**Stack:** FastAPI + Triton (3 models) + aggregation layer  
**Use Case:** Fraud detection scoring  
**Key Features:**

- Parallel model execution
- Weighted voting aggregation
- Fallback logic if model fails
- Sub-200ms combined latency

---

## 6. 🤖 Agentic Email Assistant
**Concepts:** Agents, tools, memory, reasoning loops  
**Stack:** FastAPI + vLLM + LangGraph + Redis + PostgreSQL  
**Use Case:** Automated email triage and response  
**Key Features:**

- Tool calling (search, send, schedule)
- Persistent memory across sessions
- Structured reasoning logs
- Human-in-the-loop approval

---

## 7. 💰 Cost-Optimized Inference
**Concepts:** Caching, quantization, spot instances, batching  
**Stack:** FastAPI + ONNX Runtime + Redis + S3  
**Use Case:** High-volume translation service  
**Key Features:**

- INT8 quantized models
- Response caching (70% hit rate)
- Spot GPU autoscaling
- Cost tracking per request

---

## 8. 📊 Monitored CV Pipeline
**Concepts:** Observability, metrics, tracing, alerts  
**Stack:** FastAPI + Triton + Prometheus + Grafana + Loki + OpenTelemetry  
**Use Case:** Production OCR service with SLA monitoring  
**Key Features:**

- Full request tracing
- GPU utilization dashboards
- Latency percentile tracking
- PagerDuty alerting

---

## 9. 🔄 Model A/B Testing Platform
**Concepts:** Traffic splitting, canary deployment, metrics comparison  
**Stack:** FastAPI + Triton (2 model versions) + PostgreSQL  
**Use Case:** Gradual rollout of improved sentiment model  
**Key Features:**

- 90/10 traffic split
- Per-model accuracy tracking
- Automatic rollback logic
- Statistical significance tests

---

## 10. 🌐 Edge Inference Gateway
**Concepts:** Model quantization, edge deployment, offline mode  
**Stack:** FastAPI + ONNX Runtime + SQLite  
**Use Case:** IoT device classification at retail stores  
**Key Features:**

- 4-bit quantized models
- Local SQLite for cache
- Sync to cloud when online
- <50MB memory footprint

---

## 11. 🔐 Secure Multi-Tenant Inference
**Concepts:** Auth, rate limiting, tenant isolation, encryption  
**Stack:** FastAPI + Keycloak + Triton + PostgreSQL  
**Use Case:** SaaS platform with per-customer models  
**Key Features:**

- OAuth2 + JWT validation
- Per-tenant rate limits
- Model namespace isolation
- Audit logging

---

## 12. 🚀 Auto-Scaling LLM Service
**Concepts:** HPA, queue-based scaling, GPU pooling  
**Stack:** Kubernetes + vLLM + KEDA + Prometheus  
**Use Case:** Variable-load text generation API  
**Key Features:**

- Scale 0→5 replicas based on queue
- GPU sharing across pods
- Graceful pod shutdown
- Cost = $0 when idle
