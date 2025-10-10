# Single Model API

> The simplest production pattern: one model per endpoint.

---

## What is it?

A **single model API** exposes one trained model for predictions via REST/gRPC.

!!! tip
    Start with single model endpoints; pipelines are only needed when tasks become complex.

**Example:** Image classifier that outputs "cat" or "dog".

---

## Why Use It?

* **Simplicity** — minimal orchestration, easy to monitor  
* **Low latency** — no multi-model coordination  
* **Predictable GPU/CPU usage** — one model per compute unit

---

## Backend Considerations

| Aspect        | Recommendation                                 |
| ------------- | ---------------------------------------------- |
| GPU           | Pin model to GPU, use warmup requests          |
| Scaling       | Horizontal replicas with load balancing        |
| Caching       | Cache frequent predictions to reduce GPU calls |
| Observability | Track latency, errors, throughput              |

---

## Visual Diagram

```mermaid
flowchart LR
    Client --> API[FastAPI / gRPC Endpoint]
    API --> Model[Single Model Inference]
    Model --> Output[Prediction]
```
