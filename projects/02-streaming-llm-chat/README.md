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
curl -X 'POST' \
  'http://localhost:8080/api/v1/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "user",
  "message": "Hello, who are you?"
  "session_id": "session_id",
  "model": "tinyllama:1.1b",
  "temperature": 0.7,
  "top_p": 1
}'
```

### Example stream output

```json
data: {"id": "chatcmpl-40", "choices": [{"delta": {"content": "Hey!", "role": "assistant"}, "finish_reason": null, "index": 0}], "created": 1760795323, "model": "tinyllama:1.1b", "object": "chat.completion.chunk", "system_fingerprint": "fp_ollama"}

data: {"id": "chatcmpl-40", "choices": [{"delta": {"content": " great", "role": "assistant"}, "finish_reason": null, "index": 0}], "created": 1760795324, "model": "tinyllama:1.1b", "object": "chat.completion.chunk", "system_fingerprint": "fp_ollama"}

data: [DONE]
```

---

## ⚙️ Environment Variables

| Variable      | Description                             | Example                |
| ------------- | --------------------------------------- | ---------------------- |
| `LLM_URL`     | Base URL of backend (OpenAI-compatible) | `http://vllm:8000/v1`  |
| `LLM_MODEL`   | Model name to use                       | `facebook/opt-125m`    |
| `LLM_API_KEY` | API KEY to use                          | `dummy`                |
| `REDIS_URL`   | Redis connection string                 | `redis://redis:6379/0` |
| `APP_PORT`    | Application port                        | `8000`                 |
| `LOG_LEVEL`   | Log level                               | `info`                 |

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