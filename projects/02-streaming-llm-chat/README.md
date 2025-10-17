# 02 - Streaming LLM Chat API


A **FastAPI-based streaming chat server** compatible with any **OpenAI-like backend**  
(vLLM, Ollama, or OpenAI). It supports **token-by-token SSE streaming** and  
**conversation persistence in Redis**.

---

## Quick Start

### 1. Choose a backend

#### Option A — vLLM

```bash
cp .env.vllm .env
docker compose -f docker-compose.vllm.yaml up --build
```

#### Option B — Ollama

```bash
cp .env.ollama .env
docker compose -f docker-compose.ollama.yaml up --build
```

---

## 📡 API Endpoints

| Method | Path      | Description                    |
| ------ | --------- | ------------------------------ |
| `GET`  | `/health` | Liveness probe                 |
| `GET`  | `/ready`  | Readiness probe (LLM + Redis)  |
| `POST` | `/chat`   | Stream chat completion via SSE |

### Example request

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Hello, who are you?"
  }'
```

### Example stream output

```json
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

---

## ⚙️ Environment Variables

| Variable       | Description                             | Example                |
| -------------- | --------------------------------------- | ---------------------- |
| `LLM_BASE_URL` | Base URL of backend (OpenAI-compatible) | `http://vllm:8000/v1`  |
| `LLM_MODEL`    | Model name to use                       | `facebook/opt-125m`    |
| `REDIS_URL`    | Redis connection string                 | `redis://redis:6379/0` |
| `APP_PORT`     | Application port                        | `8000`                 |
| `LOG_LEVEL`    | Log level                               | `info`                 |

---
## Development

### Prerequisites

Before running the app locally, make sure the following services are available:

| Service         | Description                                                            | Local Setup                                                                 |
| --------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **LLM Backend** | Any OpenAI-compatible server (e.g., `vLLM`, `Ollama`, or `OpenAI API`) | Start via Docker Compose: `docker compose -f docker-compose.vllm.yml up -d` |
| **Redis**       | Used to persist conversation history and session state                 | Start via Docker Compose: `docker compose up redis -d`                      |

> 💡 **Note:** If you’re not using Docker, you’ll need to run these manually.
> Example for Redis:
>
> ```bash
> docker run -d -p 6379:6379 redis:7-alpine
> ```

---

### Local Setup

```bash
make install-dev

make run
```