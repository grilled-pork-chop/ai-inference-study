# Multi-Model / Pipelines

> When multiple models are chained or combined to produce richer results.

---

## When to Use

* Multi-stage processing (e.g., OCR → NLP → database lookup)
* RAG or classification + generation workflows
* Ensemble or modular AI pipelines

---

## Concept

Pipeline = **directed graph of models**.
Each model consumes previous outputs, adds new context, and forwards results.

```mermaid
flowchart LR
    A[Input Data] --> B[Model 1 - Embeddings]
    B --> C[Model 2 - Classifier]
    C --> D[Aggregator / Post-Processor]
```

---

## Why Pipelines Matter

* Reuse specialized models for modularity
* Easier to scale individual stages
* Can mix different frameworks (ONNX, PyTorch, vLLM)

---

### Backend Example (FastAPI + Triton)

```python
async def classify_document(file: UploadFile):
    text = await extract_text(file)
    embedding = await triton_infer("embedder", text)
    category = await triton_infer("classifier", embedding)
    return {"category": category}
```

!!! warning "Pipeline Latency"
    Each hop adds latency — prefer **parallelization** and **async orchestration**.
