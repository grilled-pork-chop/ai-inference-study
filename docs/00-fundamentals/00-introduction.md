# Introduction

> A practical guide for backend engineers entering AI systems — from understanding models to deploying inference at scale.

---

## What This Guide Covers

This documentation bridges the gap between **traditional backend development** and **AI production systems**.  
You’ll learn how AI workloads differ from APIs and microservices, and how to deploy, scale, and observe them safely.

By the end, you’ll understand **how to go from model file → running inference service → production system**.

---

## Why This Guide Exists

Modern AI systems combine **software engineering**, **infrastructure**, and **machine learning**.  
Yet, most backend developers face one of two issues:

1. ML guides are written for **data scientists**, not engineers.  
2. Infra guides skip the **model-specific constraints** (GPU memory, batching, latency, etc.).

This documentation focuses on the backend perspective:  
how to **serve**, **optimize**, and **scale** models reliably — not how to train them.

---

## Document Structure

Each section builds on the previous one, following the lifecycle of deploying an AI model.

| #      | File                                               | Description                                                                           |
| ------ | -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **01** | [Models and Checkpoints](01-models-checkpoints.md) | How models are structured, saved, and optimized for inference.                        |
| **02** | [Inference](02-inference.md)                       | What inference really is, how it differs from training, and how to make it efficient. |
| **03** | [Inference Lifecycle](03-inference-lifecycle.md)   | The complete path from model load → request → output → monitoring.                    |
| **04** | [Workloads](04-workloads.md)                       | Common inference workloads (LLMs, CV, TTS) and their architectural implications.      |
| **05** | [Deployment Targets](05-deployment-targets.md)     | How inference runs on CPU, GPU, or distributed clusters.                              |
| **06** | [GPU Fundamentals](06-gpu-fundamentals.md)         | What backend engineers must know about GPUs, CUDA, and VRAM management.               |
| **07** | [Common Pitfalls](07-common-pitfalls.md)           | Typical mistakes when serving AI models and how to avoid them.                        |

---

## Audience

This guide is for **backend engineers**, **DevOps**, and **infra leads** who:

* Want to deploy or integrate models (LLMs, CV, etc.)  
* Work with inference servers like **Triton**, **vLLM**, or **TGI**  
* Need to manage GPU resources, latency, and scaling  
* Care about observability, efficiency, and production readiness  

!!! tip "If you already know FastAPI or Kubernetes"
    You’ll feel at home.  
    The main difference with AI workloads is **resource predictability** and **latency sensitivity**.

---

## Core Principles

AI inference follows the same production principles as backend development — with **different constraints**.

| Principle              | Backend Systems        | AI Inference                                 |
| ---------------------- | ---------------------- | -------------------------------------------- |
| **Stateless requests** | Typical REST API calls | Often need session or context (LLMs, agents) |
| **CPU-bound**          | Mostly CPU threads     | Often GPU or mixed CPU+GPU                   |
| **Scalability**        | Horizontal scaling     | GPU memory and batching are bottlenecks      |
| **Startup**            | Fast (seconds)         | Slow model load (tens of seconds)            |
| **Monitoring**         | Logs, metrics          | Latency, throughput, GPU utilization         |

!!! info "Key takeaway"
    AI inference is not a new paradigm — it’s backend engineering with **hardware-aware constraints** and **data-aware optimizations**.

---
