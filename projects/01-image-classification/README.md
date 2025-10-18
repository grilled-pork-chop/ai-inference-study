# 01 - Image Classification API

An image classification API using **FastAPI** and **Triton Inference Server**. Includes **Prometheus metrics** and **GPU support**.

## Feature

- Triton inference
- Prometheus metrics: request count, latency, error count, prediction confidence
- Health and readiness endpoints
- Docker

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (if using GPU Triton)
- Python 3.12 (for local dev)

## Development

```bash
# Install dependencies
make install-dev

# Run locally
make run
```