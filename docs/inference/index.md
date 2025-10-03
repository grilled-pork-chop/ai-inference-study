# Inference

Inference infrastructure determines how your models serve predictions in production. This section covers interfaces (how clients communicate) and servers (how models are hosted and scaled).

## 🎯 Quick Navigation

- **[Interfaces](inference-interfaces.md)**: REST, gRPC, WebSocket, Server-Sent Events
- **[Servers](inference-servers.md)**: Triton, TorchServe, vLLM, KServe, BentoML

## 📊 Interface vs Server Decision Matrix

| Use Case              | Recommended Interface | Recommended Server      | Reason                                     |
| --------------------- | --------------------- | ----------------------- | ------------------------------------------ |
| **Simple API**        | REST                  | FastAPI + PyTorch       | Easy development and debugging             |
| **High Performance**  | gRPC                  | Triton Inference Server | Lowest latency, highest throughput         |
| **LLM Streaming**     | WebSocket/SSE         | vLLM                    | Token streaming, real-time interaction     |
| **Multi-Model**       | REST + gRPC           | Triton                  | Framework flexibility, enterprise features |
| **Kubernetes Native** | REST                  | KServe                  | Cloud-native scaling, GitOps integration   |

## 🚀 Performance Characteristics

### Interface Latency Overhead

| Protocol          | Serialization | Connection | Total Overhead |
| ----------------- | ------------- | ---------- | -------------- |
| **REST/JSON**     | 2-5ms         | 1-3ms      | 3-8ms          |
| **gRPC/Protobuf** | 0.5-1ms       | 0.5-1ms    | 1-2ms          |
| **WebSocket**     | 1-2ms         | 0ms*       | 1-2ms          |
| **SSE**           | 1-2ms         | 0ms*       | 1-2ms          |

*After initial connection establishment

### Server Performance Comparison

| Server         | Startup Time | Memory Overhead | GPU Utilization | Concurrent Users |
| -------------- | ------------ | --------------- | --------------- | ---------------- |
| **FastAPI**    | <1s          | 100-200MB       | 60-70%          | 100-500          |
| **TorchServe** | 5-15s        | 200-500MB       | 70-85%          | 500-2000         |
| **Triton**     | 10-30s       | 300-800MB       | 85-95%          | 1000-5000        |
| **vLLM**       | 30-60s       | 500MB-2GB       | 80-90%          | 50-200*          |

*Depends heavily on model size and sequence length

## 🏗️ Architecture Patterns

### Pattern 1: Direct Model Serving
```
Client → REST API → Model → Response
```
- **Best for**: Simple use cases, prototyping
- **Limitations**: No batching, limited scalability

### Pattern 2: Inference Server
```
Client → Load Balancer → Inference Server → Model Pool → Response
```
- **Best for**: Production workloads, multiple models
- **Benefits**: Batching, monitoring, scaling

### Pattern 3: Streaming Architecture
```
Client ↔ WebSocket/SSE ↔ Stream Server ↔ Model → Token Stream
```
- **Best for**: LLMs, real-time AI interactions
- **Benefits**: Progressive responses, better UX

### Pattern 4: Microservices
```
Client → API Gateway → Service Mesh → Multiple AI Services
```
- **Best for**: Complex AI pipelines, enterprise
- **Benefits**: Service isolation, independent scaling

## 🔧 Integration Considerations

### Client-Side Integration

**JavaScript/Browser:**
```javascript
// REST API
const response = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({input: data})
});

// WebSocket streaming
const ws = new WebSocket('ws://localhost:8000/stream');
ws.onmessage = (event) => {
    const token = JSON.parse(event.data);
    updateUI(token);
};
```

**Python Client:**
```python
# gRPC client
stub = InferenceServiceStub(channel)
request = ModelInferRequest(model_name="my_model", inputs=[...])
response = stub.ModelInfer(request)

# Streaming client
async def stream_tokens():
    async for token in client.stream_generate(prompt):
        yield token
```

### Server-Side Patterns

**Batching Strategy:**
```python
# Dynamic batching implementation
class BatchProcessor:
    def __init__(self, max_batch_size=8, max_wait_time=50):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
    
    async def process_batch(self, requests):
        # Combine requests into batch
        batch_input = torch.cat([req.input for req in requests])
        batch_output = await self.model.forward(batch_input)
        # Split results back to individual requests
        return self.split_batch_output(batch_output, requests)
```

## 📊 Monitoring and Observability

### Key Metrics to Track

**Interface Metrics:**
- Request/response latency (p50, p90, p95, p99)
- Connection establishment time
- Payload serialization/deserialization time
- Error rates by status code

**Server Metrics:**
- Model loading time and memory usage
- GPU utilization and memory allocation
- Batch processing efficiency
- Request queue depth and wait times

**Business Metrics:**
- Predictions per second
- Model accuracy/quality scores
- Cost per inference
- User satisfaction (for streaming apps)

### Prometheus Integration Example

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
REQUEST_LATENCY = Histogram('inference_duration_seconds', 'Inference latency', ['model'])
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])

# Instrument your code
@REQUEST_LATENCY.labels(model=model_name).time()
def predict(input_data):
    result = model.forward(input_data)
    REQUEST_COUNT.labels(model=model_name, status='success').inc()
    return result
```

## ⚠️ Common Pitfalls

**Interface Selection:**
- Using REST for streaming use cases (poor UX)
- Using WebSocket for simple request/response (unnecessary complexity)
- Not implementing proper error handling for streaming connections

**Server Configuration:**
- Inadequate batching configuration (poor GPU utilization)
- Wrong memory allocation (OOM errors)
- Missing health checks and graceful shutdown

**Scaling Issues:**
- Not accounting for model loading time in scaling decisions
- Cold start problems in serverless deployments
- Connection pooling misconfiguration

## 🎯 Best Practices

### Interface Design
1. **Use appropriate protocol** for your use case pattern
2. **Implement proper error handling** and status codes
3. **Add request/response validation** and sanitization
4. **Design for retries** and circuit breaker patterns

### Server Optimization
1. **Configure batching** based on your latency/throughput requirements
2. **Monitor resource usage** and set appropriate limits
3. **Implement health checks** for orchestration systems
4. **Use model caching** for frequently accessed models

### Operational Excellence
1. **Set up comprehensive monitoring** from day one
2. **Implement proper logging** for debugging and auditing
3. **Plan for scaling** both up and down
4. **Test failure scenarios** and recovery procedures
