# Containerization

> Isolate dependencies, reproduce environments, and simplify scaling.

---

## Why Containerize?

- Every model has **unique Python packages, CUDA versions, config files**  
- Containers ensure **consistent environments** across dev, staging, and production  
- Simplifies deployment to **cloud, edge, or hybrid**  

---

## Best Practices

| Practice                  | Reason                                  |
| ------------------------- | --------------------------------------- |
| Minimal base image        | Smaller attack surface & faster builds  |
| Runtime-only dependencies | Avoid training libraries in prod images |
| Multi-stage builds        | Reduce final image size                 |
| Tag with model version    | Reproducibility and rollback            |

---

## Docker Build Flow

```mermaid
flowchart LR
    Base[Python + OS] --> Stage1[Install Dependencies]
    Stage1 --> Stage2[Copy Model Weights]
    Stage2 --> Runtime[Inference-ready Docker Image]
```

---

### Example Dockerfile

```dockerfile
FROM python:3.12-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_weights/ ./model_weights/
COPY src/ ./src/
CMD ["python", "src/server.py"]
```

---

### Tips & Warnings

!!! tip
    * Keep models **outside image** for easier updates
    * Use **GPU-compatible base images** (CUDA) only when needed

!!! warning
    * Avoid embedding secrets in images
    * Don’t install unnecessary packages — security and size impact