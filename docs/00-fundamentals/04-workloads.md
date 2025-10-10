# AI Workload Types

> Different AI workloads behave differently in production.
> Understanding their characteristics is critical for proper design.

---

## Types of Workloads

Different workloads require different infrastructure patterns and optimizations.

## Online / Real-Time

* Requests come **one at a time**, often from web or mobile clients  
* Requires **low latency** (<100ms–3s depending on SLA)  
* Common examples: intent detection, real-time image classification, chatbots

!!! tip "Backend optimization"
    Use async request handling and GPU pooling to ensure responsiveness.

---

### Batch / Offline

* Requests are collected and processed together  
* Focus on **throughput, not latency**  
* Often used for analytics, report generation, or embedding calculations

!!! info "Backend design"
    Batching enables **GPU utilization efficiency** but requires a queueing system (e.g., Celery, Redis, Ray).

---


### Streaming / Real-Time

* Continuous input → continuous output  
* Examples: LLM token streaming, ASR, TTS  
* Backend challenges: **persistent connections, flow control, partial results**

!!! warning
    Treat streaming separately from batch/interactive endpoints to avoid queue congestion and high latency.

---

### Edge / Mobile

* Inference happens on device  
* Must consider **memory, CPU/GPU constraints, battery**  
* Offline operation possible, but model size must be small

!!! tip
    Quantize models and profile across multiple device types. Latency can vary widely.

---

## Latency Classes

Latency defines **how quickly a model must respond** to meet user expectations.
Different applications tolerate different delays — some must feel instant, others can take a few seconds.

| Class                             | SLA (Target Latency) | Typical Use                         |
| --------------------------------- | -------------------- | ----------------------------------- |
| **Sub-100 ms (Interactive)**      | < 0.1 s              | Real-time CV, ranking, autocomplete |
| **100 ms – 3 s (Conversational)** | 0.1–3 s              | LLM chat, summarization             |
| **3 – 10 s (Generative)**         | few s                | Image generation, TTS               |
| **> 10 s (Multi-step)**           | 10 s – min           | AI agents, document workflows       |

!!! tip "Why it matters"
    Each latency class implies **different architectural choices**:

    * **Interactive** → synchronous APIs, GPU always warm
    * **Conversational** → streaming responses (WebSocket/SSE)
    * **Generative** → async job queue or request batching
    * **Multi-step** → background pipelines or orchestration (Celery, Ray)

!!! info "What is SLA?"
    **SLA (Service Level Agreement)** defines the **expected latency and reliability** for an API.
    Example: “99% of requests under 200 ms.”
    It helps you size compute, design autoscaling, and choose between sync or async execution.
