# Deployment Targets
> Where you run inference directly impacts cost, latency, and scalability.  
> Choosing the right target is a balance of **performance, cost, and operational complexity**.

---

## Cloud GPU

Cloud GPUs provide **flexibility and scalability** for large models.

**Backend considerations:**

* Ideal for LLMs, large CV pipelines, or batch jobs  
* Autoscale instances to match demand  
* Use **spot instances** for non-critical batch inference to save costs  
* Monitor GPU memory usage and request queue length

!!! tip "Backend best practice"
    Always separate **interactive endpoints** from **batch jobs**, even in the cloud.  
    This prevents high-latency batch jobs from impacting real-time services.

---

## On-Prem GPU

On-prem GPUs offer **full control** and **data privacy**, but require maintenance.

**When to use:**

* Enterprise with sensitive or private data  
* Workloads that need guaranteed low latency  
* Predictable, steady traffic patterns

!!! warning
    Requires **hardware maintenance**, driver updates, and careful cluster management.

---

## CPU

CPUs are **cheaper and easier** to scale horizontally for small models.

**Characteristics:**

* Works for small NLP, CV, or embedding models  
* Lower throughput and higher latency for large models  
* Combine **batching + caching** to maximize efficiency

!!! tip
    Consider CPUs for **analytics pipelines** or models <2GB where latency is not critical.

---

## Edge / Mobile

Running inference directly on devices enables **offline capabilities**.

**Requirements:**

* Quantization and memory optimization  
* Careful profiling: performance differs across device types  
* Limited throughput — often single-user or small batch

!!! info "Use cases"
    Mobile apps, IoT devices, robotics, and real-time sensor processing.

---

## Decision Factors

When choosing a deployment target, always consider:

| Factor                    | Why it matters                              |
| ------------------------- | ------------------------------------------- |
| **Model size**            | Must fit in memory (GPU/CPU/Edge)           |
| **Throughput**            | Requests/sec, batch efficiency              |
| **Latency**               | Interactive vs batch, real-time constraints |
| **Cost**                  | GPU vs CPU, cloud vs on-prem                |
| **Security & compliance** | Private data may restrict cloud deployment  |

!!! tip
    Map **model size + latency requirements** first, then optimize for cost and security.
