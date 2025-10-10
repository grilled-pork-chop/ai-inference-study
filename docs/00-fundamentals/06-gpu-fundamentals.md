# GPU Fundamentals
> Most production AI inference is GPU-bound. Understanding memory, cores, and precision is crucial for reliable performance.

---

## GPU Memory & Model Size

A GPU must hold **model weights** + **activations** for each batch.  

* Small CV models → 2–8GB VRAM  
* Medium LLMs (7B–13B) → 8–24GB VRAM  
* Large LLMs (70B+) → 40–80GB VRAM  

!!! warning "OOM Risk"
    Batch size × activation memory must not exceed GPU VRAM.  
    Otherwise, inference will crash with out-of-memory errors.

!!! tip
    Always measure **memory per batch** before scaling to multiple GPUs.

---

## CUDA & Tensor Cores

* **CUDA cores** → Parallel compute for matrix operations  
* **Tensor cores** → Accelerate mixed-precision (FP16 / INT8) calculations  
* Mixed precision reduces memory footprint and increases throughput  

!!! tip
    Don’t just monitor latency per request — **throughput per GPU** is the critical metric for scaling decisions.

---

## Best Practices

1. **Warmup** models with dummy input to avoid first-inference delay.  
2. **Batch small requests** to fully utilize cores without increasing latency.  
3. **Monitor GPU metrics continuously**: utilization, memory, temperature.  
4. **Separate large models across GPUs** to prevent contention.  

!!! info "Backend mindset"
    A backend engineer doesn’t just serve requests — they **orchestrate GPU resources** efficiently.

---

## Visual Summary

```mermaid
flowchart LR
    Input -->|Batch| GPU[GPU VRAM & Cores]
    GPU --> Output[Predictions]
```