# AI Schema

> Concept maps connecting AI training, optimization, deployment, and serving.  
> Use this as a high-level guide for system architecture and learning flow.

---

## 1. AI Lifecycle Overview

```mermaid
flowchart LR
    A[Data Collection] --> B[Preprocessing]
    B --> C[Model Training]
    C --> D[Evaluation]
    D --> E[Compression & Optimization]
    E --> F[Deployment]
    F --> G[Monitoring & Feedback]
    G --> B
```

*Feedback loops drive model iteration and continuous improvement.*

---

## 2. Model Optimization Path

```mermaid
flowchart LR
    Train[Training] --> Grad[Gradient Control]
    Grad --> Reg[Regularization]
    Reg --> Mix[Mixed Precision]
    Mix --> Comp[Compression]
    Comp --> Quant[Quantization]
    Comp --> Prune[Pruning]
    Comp --> Distill[Distillation]
    Quant --> Deploy[Deployment]
    Distill --> Deploy
    Prune --> Deploy
```

**Goal:** Improve efficiency without sacrificing performance.

---

## 3. Scaling Hierarchy

```mermaid
flowchart TD
    Scale[Scaling] --> Hor[Horizontal Scaling]
    Scale --> Ver[Vertical Scaling]
    Hor --> Rep[Replica Deployment]
    Hor --> Dist[Distributed Serving]
    Ver --> GPU[GPU/TPU Upgrade]
    Ver --> Mem[Memory Optimization]
```

**Horizontal = throughput**
**Vertical = capacity**

---

## 4. Inference Architecture Map

```mermaid
flowchart LR
    Client --> Gateway[API Gateway]
    Gateway --> Router[Router / Load Balancer]
    Router --> Service[Inference Service]
    Service --> Runtime[Triton / vLLM / Ollama]
    Runtime --> GPU[GPU Runtime]
    GPU --> Monitor[Telemetry & Observability]
    Monitor --> Store[Metrics / Logs / Traces]
```

**Layers:**

* Gateway → Authentication & Throttling
* Router → Request dispatch & routing
* Runtime → Model execution backend
* Monitor → Metrics & feedback

---

## 5. Training Stack Abstraction

```mermaid
flowchart TD
    Data[Data Pipeline] --> Loader[Dataloader / Augmentation]
    Loader --> Model[Model Architecture]
    Model --> Optimizer[Optimizer + LR Scheduler]
    Optimizer --> GPU[GPU Runtime / Accelerator]
    GPU --> Checkpoint[Checkpoint & Logging]
```

Each layer can be **profiled, optimized, or replaced** independently.

---

## 6. Observability Ecosystem

```mermaid
flowchart LR
    Metrics --> Prometheus
    Traces --> Jaeger
    Logs --> Loki
    Prometheus --> Grafana
    Jaeger --> Grafana
    Loki --> Grafana
```

**Unified dashboards** give end-to-end visibility from model inference to user latency.

---

## 7. Complete AI System Map

```mermaid
flowchart TD
    subgraph Dev[Development]
        A[Data Collection]
        B[Model Training]
        C[Evaluation]
        D[Optimization]
    end

    subgraph Deploy[Deployment]
        E[Serving Runtime]
        F[API Gateway]
        G[Autoscaler]
    end

    subgraph Obs[Observability]
        H[Metrics]
        I[Logs]
        J[Tracing]
    end

    A --> B --> C --> D --> E --> F
    F --> G
    E --> H
    F --> I
    G --> J
    H --> A
```
