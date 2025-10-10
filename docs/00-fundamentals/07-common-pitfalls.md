# Common Pitfalls in Inference Systems

> Most production inference issues don’t come from the model — they come from how it’s served.

Inference looks simple (“call model, return output”) but it’s an **engineering pipeline** that mixes CPU, GPU, memory, and network behavior.
Below are the most frequent traps backend teams fall into — and how to avoid them.

---

## Resource Mismanagement

### Running Models in API Threads

Putting inference directly in your FastAPI or Flask route handler blocks the event loop and kills concurrency.

**Why it happens:**
Teams underestimate how long model inference takes (especially with transformers) and forget Python’s GIL.

**Fix:**

* Offload inference to async background workers
* Or use a dedicated inference server (Triton, vLLM, TGI)

??? example
    ```python
    # Bad
    @app.post("/infer")
    def infer(req: Request):
    return model(req.json())

    # Good
    @app.post("/infer")
    async def infer(req: Request):
        data = await req.json()
        result = await inference_pool.run_inference(data)
        return result
    ```

---

### Overloading a Single GPU

Running multiple large models (or too many concurrent jobs) on one GPU leads to out-of-memory crashes and context thrashing.

**Fix:**

* Pin models to specific GPUs (`CUDA_VISIBLE_DEVICES`)
* Use load balancers or model pools
* Monitor VRAM and model residency time

!!! tip
    **One model = one GPU** is usually safer unless using a managed runtime like Triton that handles GPU partitioning.

---

## Cold Starts & Latency Spikes

### Ignoring Model Warmup

The first inference after model load triggers JIT compilation, memory allocation, or graph optimization — all slow.

**Fix:**
Run warmup requests on startup.

**Extra:**
For containerized deployments, keep models loaded via a readiness probe before routing traffic.

??? example
    ```python
    @app.on_event("startup")
    async def warmup():
        dummy = tokenizer("hello world", return_tensors="pt")
        _ = model(**dummy)
    ```

---

### Missing Adaptive Batching

Static batch sizes or poor timeout settings either underutilize GPUs or increase latency.

**Fix:**

* Implement *dynamic batching* (e.g., Triton `max_queue_delay_microseconds`)
* Adjust batch timeout dynamically under load

---

## Reliability & Backpressure

### No Timeout or Backpressure

Requests that hang pile up and saturate your queue or GPU.

**Fix:**

* Enforce per-request timeouts
* Cancel slow jobs
* Drop or defer excess requests under load

??? example
    ```python
    try:
        result = await asyncio.wait_for(run_inference(), timeout=3)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Inference timeout")
    ```

---

### Model Reload Storms

Deployments that reload the same large model on each request waste minutes and memory.

**Fix:**

* Keep models persistent in memory
* Use connection pooling or model registries
* Hot-reload models only when weights actually change

---

## Input / Output Inconsistency

### Tokenizer or Preprocessing Mismatch

Inference often fails silently when the preprocessing pipeline differs from the one used during training.

**Symptoms:**

* Unexpected accuracy drop
* Misaligned embeddings
* Garbage outputs from LLMs

**Fix:**

* Version both model **and tokenizer**
* Store preprocessing config in model artifacts (Hugging Face `config.json` or ONNX metadata)

---

### Overly Heavy Payloads

Passing entire files (images, documents) inline in JSON causes memory bloat.

**Fix:**

* Use binary uploads or pre-signed URLs
* Stream large payloads in chunks

---

## Observability & Debugging Gaps

### No Visibility or Metrics

Without telemetry, you’re flying blind — especially under load.

**Fix:**

* Log `model_name`, `latency_ms`, `request_id`, `gpu_util`
* Use Prometheus / OpenTelemetry
* Set up per-stage metrics (preprocess, execute, postprocess)

---

### Missing Request Correlation

When errors occur in distributed setups, logs are useless without consistent IDs.

**Fix:**

* Propagate `X-Request-ID` or trace context headers end-to-end
* Attach them to logs, traces, and responses

---

## Architecture Anti-Patterns

!!! warning "Backend Anti-Patterns"
    - Flask/Django sync routes calling heavy GPU models
    - Running inference in the same container as business logic
    - Ignoring batching or queue configuration
    - Serving multiple latency classes (LLM + CV + ASR) on the same instance
    - Deploying without readiness probes or health checks
    - Using blocking tokenizers inside async loops

---

## Quick Summary

**Inference is not just prediction — it’s orchestration.**

* Keep model loading **persistent and isolated**
* Treat GPUs as **shared compute resources**, not threads
* Always have **timeouts, metrics, and warmups**
* Design for **backpressure and cancellation**
* Separate **business API** from **inference runtime**

!!! info
    Backend developers turn *research models* into *production systems*.

    Failures rarely come from the model itself — they come from **architecture debt** and **missing observability**.
