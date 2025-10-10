# State Management

> Inference is stateless by default — until you need context.

---

## Why It Matters

Most AI inference calls are independent. But:

- Conversational AI requires **chat memory**
- RAG pipelines require **retrieved context**
- Pipelines require **partial outputs**

Without explicit state management, systems risk:

- Lost context  
- Memory overload  
- Inconsistent outputs  

---

## State Types

| Type                     | Example                                | Backend                     |
| ------------------------ | -------------------------------------- | --------------------------- |
| **Session**              | Chat conversation                      | Redis / PostgreSQL          |
| **Context cache**        | RAG embeddings, FAISS/Milvus indexes   | Redis / Milvus              |
| **Intermediate results** | Partial outputs between pipeline steps | Local cache / shared volume |


---

## Recommended Pattern

Keep inference **stateless per request**. Store state externally:

```mermaid
flowchart LR
    Client --> API
    API --> Cache[Redis / DB]
    Cache --> Inference[Model Runtime]
    Inference --> Cache
```

??? example "Python Example"
    ```python
    # Save session state in Redis
    redis.hset(f"chat:{session_id}", "messages", json.dumps(messages))
    redis.expire(f"chat:{session_id}", 600)  # auto-expire in 10min
    ```

### Best Practices

!!! tip
    * Use **fast backend** (Redis) for transient state
    * Set **TTL** to avoid memory leaks
    * Avoid storing user data in GPU memory
    * Keep inference workers **stateless**

!!! warning
    * Storing context in GPU memory → blocks other requests, memory leaks
    * Large unbounded session caches → OOM errors

---

## Takeaway

> Let GPU workers focus on tensor computation; external storage handles session and pipeline state.
