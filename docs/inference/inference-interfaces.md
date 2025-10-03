# Inference Interfaces

Communication interfaces determine how clients interact with your AI models. The choice impacts latency, scalability, development complexity, and user experience.

## 🎯 Interface Comparison Matrix

| Protocol      | Latency | Throughput | Streaming | Complexity | Browser Support | Best For                        |
| ------------- | ------- | ---------- | --------- | ---------- | --------------- | ------------------------------- |
| **REST/HTTP** | Medium  | High       | ❌         | Low        | ✅               | Simple APIs, CRUD operations    |
| **gRPC**      | Low     | Very High  | ✅         | Medium     | Limited*        | High-performance, microservices |
| **WebSocket** | Low     | Medium     | ✅         | Medium     | ✅               | Real-time, bidirectional        |
| **SSE**       | Low     | Medium     | ✅         | Low        | ✅               | Server-to-client streaming      |

*Requires gRPC-Web proxy for browsers

## 🚀 REST/HTTP APIs

### When to Use REST
- **Simple prediction APIs** with request/response pattern
- **Maximum compatibility** across clients and tools
- **Easy debugging** and testing with standard HTTP tools
- **Caching requirements** with HTTP cache headers

### Technical Implementation

**FastAPI Example:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import time
from prometheus_client import Counter, Histogram

app = FastAPI(title="AI Inference API", version="1.0.0")

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

class PredictionRequest(BaseModel):
    image_data: str  # base64 encoded
    model_name: str = "default"
    confidence_threshold: float = 0.5

class PredictionResponse(BaseModel):
    predictions: list[dict]
    confidence_scores: list[float]
    processing_time_ms: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # Load and preprocess image
        image_tensor = preprocess_image(request.image_data)
        
        # Model inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = postprocess_outputs(outputs, request.confidence_threshold)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.observe(processing_time / 1000)
        
        return PredictionResponse(
            predictions=predictions["boxes"],
            confidence_scores=predictions["scores"],
            processing_time_ms=processing_time,
            model_version="yolov8n-1.0.0"
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": time.time()
    }

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    batch_results = []
    
    # Process in batches for efficiency
    batch_size = 8
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        
        # Batch preprocessing
        batch_tensors = torch.stack([preprocess_image(req.image_data) for req in batch])
        
        # Batch inference
        with torch.no_grad():
            batch_outputs = model(batch_tensors)
        
        # Individual postprocessing
        for j, req in enumerate(batch):
            predictions = postprocess_outputs(batch_outputs[j], req.confidence_threshold)
            batch_results.append(predictions)
    
    return {"results": batch_results, "batch_size": len(requests)}
```

**Performance Optimizations:**
```python
# HTTP/2 with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --http h2

# Connection pooling for clients
import httpx

class AIClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            timeout=httpx.Timeout(30.0)
        )
    
    async def predict(self, image_data: str):
        response = await self.client.post(
            "/predict",
            json={"image_data": image_data}
        )
        return response.json()
```

### REST Performance Characteristics
- **Latency Overhead**: 3-8ms (JSON serialization + HTTP)
- **Throughput**: 1,000-10,000 RPS (depending on payload size)
- **Payload Limits**: Typically 1-100MB per request
- **Caching**: Excellent with HTTP cache headers

## ⚡ gRPC

### When to Use gRPC
- **High-performance microservices** communication
- **Strong typing** requirements with Protocol Buffers
- **Bidirectional streaming** for real-time inference
- **Language interoperability** with type safety

### Technical Implementation

**Protocol Buffer Definition (`inference.proto`):**
```protobuf
syntax = "proto3";

package inference;

service InferenceService {
    // Unary prediction
    rpc Predict(PredictRequest) returns (PredictResponse);
    
    // Server streaming for batch results
    rpc PredictStream(PredictRequest) returns (stream PredictResponse);
    
    // Bidirectional streaming for real-time inference
    rpc PredictRealtime(stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
    string model_name = 1;
    bytes image_data = 2;
    float confidence_threshold = 3;
    map<string, string> metadata = 4;
}

message PredictResponse {
    repeated BoundingBox predictions = 1;
    repeated float confidence_scores = 2;
    float processing_time_ms = 3;
    string model_version = 4;
    string request_id = 5;
}

message BoundingBox {
    float x = 1;
    float y = 2;
    float width = 3;
    float height = 4;
    string class_name = 5;
}
```

**Server Implementation:**
```python
import grpc
from concurrent import futures
import inference_pb2_grpc
import inference_pb2
import torch
import asyncio

class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, model):
        self.model = model
        self.request_count = 0
    
    def Predict(self, request, context):
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Convert protobuf to tensor
            image_tensor = self.bytes_to_tensor(request.image_data)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                predictions = self.postprocess_outputs(outputs, request.confidence_threshold)
            
            # Build response
            response = inference_pb2.PredictResponse()
            response.processing_time_ms = (time.time() - start_time) * 1000
            response.model_version = "yolov8n-1.0.0"
            response.request_id = f"req-{self.request_count}"
            
            for box, score in zip(predictions["boxes"], predictions["scores"]):
                bbox = response.predictions.add()
                bbox.x, bbox.y, bbox.width, bbox.height = box
                bbox.class_name = predictions["classes"][len(response.predictions) - 1]
                response.confidence_scores.append(score)
            
            return response
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()
    
    def PredictStream(self, request, context):
        """Server streaming - useful for batch processing"""
        batch_size = 32
        
        # Process in batches and stream results
        for i in range(0, len(request.batch_data), batch_size):
            batch = request.batch_data[i:i + batch_size]
            
            # Batch inference
            batch_results = self.process_batch(batch)
            
            # Stream each result
            for result in batch_results:
                yield result
    
    def PredictRealtime(self, request_iterator, context):
        """Bidirectional streaming for real-time inference"""
        for request in request_iterator:
            # Process each request as it arrives
            response = self.Predict(request, context)
            yield response

# Server setup
def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000)
        ]
    )
    
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(model), server
    )
    
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    print(f"Starting gRPC server on {listen_addr}")
    server.start()
    server.wait_for_termination()
```

**Client Implementation:**
```python
import grpc
import inference_pb2_grpc
import inference_pb2

class GRPCInferenceClient:
    def __init__(self, server_address: str):
        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000)
            ]
        )
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
    
    def predict(self, image_data: bytes, model_name: str = "default"):
        request = inference_pb2.PredictRequest(
            model_name=model_name,
            image_data=image_data,
            confidence_threshold=0.5
        )
        
        response = self.stub.Predict(request)
        return response
    
    def predict_stream(self, requests):
        """Use server streaming for batch requests"""
        for response in self.stub.PredictStream(requests):
            yield response
    
    def predict_realtime(self, request_generator):
        """Bidirectional streaming"""
        response_iterator = self.stub.PredictRealtime(request_generator)
        for response in response_iterator:
            yield response
```

### gRPC Performance Characteristics
- **Latency Overhead**: 1-2ms (binary protobuf + HTTP/2)
- **Throughput**: 5,000-50,000 RPS (binary serialization advantage)
- **Connection Efficiency**: Multiplexing over single connection
- **Payload Size**: More efficient for large payloads

## 🔄 WebSocket

### When to Use WebSocket
- **Real-time bidirectional communication** (chat, gaming)
- **Streaming inference** with user interaction
- **Long-lived connections** with state management
- **Browser-based real-time applications**

### Technical Implementation

**FastAPI WebSocket Server:**
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import asyncio
import torch
from typing import Dict, List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_contexts: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.user_contexts[client_id] = {"session_id": client_id, "request_count": 0}
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.user_contexts[client_id]
    
    async def send_personal_message(self, message: dict, client_id: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Update context
            context = manager.user_contexts[client_id]
            context["request_count"] += 1
            
            # Process based on message type
            if message["type"] == "predict":
                await handle_prediction(message, client_id, context)
            elif message["type"] == "stream_predict":
                await handle_streaming_prediction(message, client_id, context)
            elif message["type"] == "batch_predict":
                await handle_batch_prediction(message, client_id, context)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def handle_prediction(message: dict, client_id: str, context: dict):
    """Handle single prediction request"""
    try:
        # Process image
        image_data = message["data"]["image"]
        image_tensor = preprocess_image_base64(image_data)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = postprocess_outputs(outputs)
        
        # Send response
        response = {
            "type": "prediction_result",
            "request_id": message.get("request_id"),
            "predictions": predictions,
            "processing_time_ms": predictions["processing_time"],
            "session_request_count": context["request_count"]
        }
        
        await manager.send_personal_message(response, client_id)
        
    except Exception as e:
        error_response = {
            "type": "error",
            "request_id": message.get("request_id"),
            "error": str(e)
        }
        await manager.send_personal_message(error_response, client_id)

async def handle_streaming_prediction(message: dict, client_id: str, context: dict):
    """Handle streaming prediction with progressive results"""
    request_id = message.get("request_id")
    
    try:
        # Send processing started notification
        await manager.send_personal_message({
            "type": "stream_start",
            "request_id": request_id
        }, client_id)
        
        # Simulate progressive processing (useful for complex models)
        image_data = message["data"]["image"]
        
        # Stage 1: Preprocessing
        await manager.send_personal_message({
            "type": "stream_progress",
            "request_id": request_id,
            "stage": "preprocessing",
            "progress": 0.2
        }, client_id)
        
        image_tensor = preprocess_image_base64(image_data)
        
        # Stage 2: Inference
        await manager.send_personal_message({
            "type": "stream_progress", 
            "request_id": request_id,
            "stage": "inference",
            "progress": 0.6
        }, client_id)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Stage 3: Postprocessing
        await manager.send_personal_message({
            "type": "stream_progress",
            "request_id": request_id, 
            "stage": "postprocessing",
            "progress": 0.9
        }, client_id)
        
        predictions = postprocess_outputs(outputs)
        
        # Send final result
        await manager.send_personal_message({
            "type": "stream_complete",
            "request_id": request_id,
            "predictions": predictions,
            "progress": 1.0
        }, client_id)
        
    except Exception as e:
        await manager.send_personal_message({
            "type": "stream_error",
            "request_id": request_id,
            "error": str(e)
        }, client_id)
```

**JavaScript Client:**
```javascript
class AIWebSocketClient {
    constructor(serverUrl, clientId) {
        this.serverUrl = serverUrl;
        this.clientId = clientId;
        this.socket = null;
        this.requestCallbacks = new Map();
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            this.socket = new WebSocket(`${this.serverUrl}/ws/${this.clientId}`);
            
            this.socket.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            
            this.socket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.socket.onclose = () => {
                console.log('WebSocket disconnected');
                // Implement reconnection logic
                setTimeout(() => this.connect(), 5000);
            };
        });
    }
    
    predict(imageData, requestId = null) {
        requestId = requestId || Date.now().toString();
        
        return new Promise((resolve, reject) => {
            this.requestCallbacks.set(requestId, { resolve, reject });
            
            this.socket.send(JSON.stringify({
                type: 'predict',
                request_id: requestId,
                data: { image: imageData }
            }));
        });
    }
    
    streamPredict(imageData, onProgress, requestId = null) {
        requestId = requestId || Date.now().toString();
        
        return new Promise((resolve, reject) => {
            this.requestCallbacks.set(requestId, { 
                resolve, 
                reject, 
                onProgress,
                isStream: true 
            });
            
            this.socket.send(JSON.stringify({
                type: 'stream_predict',
                request_id: requestId,
                data: { image: imageData }
            }));
        });
    }
    
    handleMessage(message) {
        const callback = this.requestCallbacks.get(message.request_id);
        if (!callback) return;
        
        switch (message.type) {
            case 'prediction_result':
                callback.resolve(message);
                this.requestCallbacks.delete(message.request_id);
                break;
                
            case 'stream_progress':
                if (callback.onProgress) {
                    callback.onProgress(message);
                }
                break;
                
            case 'stream_complete':
                callback.resolve(message);
                this.requestCallbacks.delete(message.request_id);
                break;
                
            case 'error':
            case 'stream_error':
                callback.reject(new Error(message.error));
                this.requestCallbacks.delete(message.request_id);
                break;
        }
    }
}

// Usage example
const client = new AIWebSocketClient('ws://localhost:8000', 'user123');

async function runInference() {
    await client.connect();
    
    // Single prediction
    const result = await client.predict(imageBase64);
    console.log('Prediction:', result.predictions);
    
    // Streaming prediction with progress
    const streamResult = await client.streamPredict(
        imageBase64,
        (progress) => {
            console.log(`Progress: ${progress.stage} - ${progress.progress * 100}%`);
            updateProgressBar(progress.progress);
        }
    );
    console.log('Stream complete:', streamResult.predictions);
}
```

### WebSocket Performance Characteristics
- **Connection Overhead**: Initial handshake ~5-10ms, then 0ms
- **Message Latency**: <1ms for small messages
- **Concurrent Connections**: 1,000-10,000 per server
- **Memory Usage**: ~2-5KB per connection

## 📡 Server-Sent Events (SSE)

### When to Use SSE
- **Server-to-client streaming** only (no client uploads during stream)
- **LLM token streaming** for chat applications
- **Simple streaming** with HTTP compatibility
- **Progressive result delivery**

### Technical Implementation

**FastAPI SSE Server:**
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
import torch
from typing import AsyncIterator

app = FastAPI()

async def generate_tokens(prompt: str, model_name: str = "gpt-3.5") -> AsyncIterator[str]:
    """Simulate LLM token generation"""
    
    # Initialize generation
    yield f"data: {json.dumps({'type': 'start', 'model': model_name})}\n\n"
    
    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate tokens one by one
    generated = inputs.clone()
    
    for step in range(100):  # Max 100 tokens
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            
        # Decode token
        token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        
        # Send token
        yield f"data: {json.dumps({'type': 'token', 'text': token_text, 'step': step})}\n\n"
        
        # Add to generated sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Small delay to simulate realistic generation speed
        await asyncio.sleep(0.05)
        
        # Check for stop conditions
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Send completion
    final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    yield f"data: {json.dumps({'type': 'complete', 'full_text': final_text})}\n\n"

@app.get("/generate")
async def generate_stream(prompt: str, model: str = "gpt-3.5"):
    """Stream LLM token generation"""
    
    return StreamingResponse(
        generate_tokens(prompt, model),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/predict_stream")
async def predict_stream(image_url: str):
    """Stream computer vision results progressively"""
    
    async def generate_predictions():
        # Download and preprocess
        yield f"data: {json.dumps({'stage': 'downloading', 'progress': 0.1})}\n\n"
        image = download_image(image_url)
        
        yield f"data: {json.dumps({'stage': 'preprocessing', 'progress': 0.3})}\n\n"
        image_tensor = preprocess_image(image)
        
        # Multiple model passes for progressive detail
        models = ['fast_detector', 'accurate_detector', 'classifier']
        all_predictions = {}
        
        for i, model_name in enumerate(models):
            progress = 0.3 + (i + 1) * 0.2
            yield f"data: {json.dumps({'stage': f'inference_{model_name}', 'progress': progress})}\n\n"
            
            model = load_model(model_name)
            with torch.no_grad():
                outputs = model(image_tensor)
                predictions = postprocess_outputs(outputs)
            
            all_predictions[model_name] = predictions
            
            # Send intermediate results
            yield f"data: {json.dumps({'type': 'partial_result', 'model': model_name, 'predictions': predictions})}\n\n"
        
        # Final aggregated result
        final_predictions = aggregate_predictions(all_predictions)
        yield f"data: {json.dumps({'type': 'final_result', 'predictions': final_predictions, 'progress': 1.0})}\n\n"
    
    return StreamingResponse(
        generate_predictions(),
        media_type="text/plain"
    )
```

**JavaScript Client:**
```javascript
class SSEClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    
    streamGeneration(prompt, onToken, onComplete, onError) {
        const eventSource = new EventSource(
            `${this.baseUrl}/generate?prompt=${encodeURIComponent(prompt)}`
        );
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'start':
                    console.log(`Starting generation with ${data.model}`);
                    break;
                    
                case 'token':
                    onToken(data.text, data.step);
                    break;
                    
                case 'complete':
                    onComplete(data.full_text);
                    eventSource.close();
                    break;
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            onError(error);
            eventSource.close();
        };
        
        return eventSource;  // Return for manual closing if needed
    }
    
    streamPrediction(imageUrl, onProgress, onResult, onComplete) {
        const eventSource = new EventSource(
            `${this.baseUrl}/predict_stream?image_url=${encodeURIComponent(imageUrl)}`
        );
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.stage) {
                onProgress(data.stage, data.progress);
            } else if (data.type === 'partial_result') {
                onResult(data.model, data.predictions);
            } else if (data.type === 'final_result') {
                onComplete(data.predictions);
                eventSource.close();
            }
        };
        
        return eventSource;
    }
}

// Usage example
const sseClient = new SSEClient('http://localhost:8000');

// Stream text generation
const eventSource = sseClient.streamGeneration(
    "Write a story about AI",
    (token, step) => {
        // Append each token to UI
        document.getElementById('output').textContent += token;
    },
    (fullText) => {
        console.log('Generation complete:', fullText);
    },
    (error) => {
        console.error('Generation failed:', error);
    }
);

// Stream image prediction
sseClient.streamPrediction(
    "https://example.com/image.jpg",
    (stage, progress) => {
        updateProgressBar(stage, progress);
    },
    (model, predictions) => {
        displayPartialResults(model, predictions);
    },
    (finalPredictions) => {
        displayFinalResults(finalPredictions);
    }
);
```

### SSE Performance Characteristics
- **Latency**: <1ms per message after connection
- **Browser Compatibility**: Native EventSource API
- **Automatic Reconnection**: Built-in reconnection on connection loss
- **Memory Usage**: Very low, event-driven

## 📊 Performance Benchmarks

### Latency Comparison (Single Request)

| Protocol      | Serialization | Network | Processing | Total   |
| ------------- | ------------- | ------- | ---------- | ------- |
| **REST/HTTP** | 2-5ms         | 1-3ms   | 50ms       | 53-58ms |
| **gRPC**      | 0.5-1ms       | 0.5-1ms | 50ms       | 51-52ms |
| **WebSocket** | 1ms           | 0ms*    | 50ms       | 51ms    |
| **SSE**       | 1ms           | 0ms*    | 50ms       | 51ms    |

*After initial connection

### Throughput Comparison (RPS)

| Protocol      | Small Payload | Large Payload | Concurrent Connections |
| ------------- | ------------- | ------------- | ---------------------- |
| **REST/HTTP** | 10,000        | 1,000         | 1,000-5,000            |
| **gRPC**      | 25,000        | 5,000         | 5,000-20,000           |
| **WebSocket** | 15,000        | 2,000         | 1,000-10,000           |
| **SSE**       | 12,000        | 1,500         | 1,000-8,000            |

## 🎯 Decision Framework

### Choose REST when:
- ✅ Simple request/response pattern
- ✅ Maximum client compatibility needed
- ✅ Caching requirements important
- ✅ Easy debugging and testing required

### Choose gRPC when:
- ✅ High performance and low latency critical
- ✅ Strong typing and validation needed
- ✅ Microservices architecture
- ✅ Bidirectional streaming required

### Choose WebSocket when:
- ✅ Real-time bidirectional communication
- ✅ Stateful connections beneficial
- ✅ Browser-based real-time apps
- ✅ Interactive AI applications

### Choose SSE when:
- ✅ Server-to-client streaming only
- ✅ Progressive result delivery
- ✅ LLM token streaming
- ✅ Simple streaming with HTTP compatibility
