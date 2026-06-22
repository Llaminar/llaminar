# Llaminar REST API Server - Implementation Plan

**Status**: Planning  
**Author**: David Sanftenberg  
**Date**: November 2025

## Overview

This document outlines the implementation plan for adding an OpenAI-compatible REST API server to Llaminar. This addresses the architectural limitation where interactive chat modes cannot work with MPI tensor parallelism (all ranks must participate in forward passes synchronously).

## Problem Statement

The current `--chat` interactive mode has a fundamental incompatibility with MPI tensor parallelism:

1. **Tensor Parallelism Requirement**: All MPI ranks must call `pipeline->forward()` together - attention computations use MPI collectives internally
2. **Interactive Blocking**: The chat UI on rank 0 blocks waiting for user input
3. **Deadlock Result**: Non-rank-0 processes wait at MPI barriers while rank 0 is blocked on stdin

The `--chat-single` mode works because it's non-interactive (single prompt → single response), but true interactive chat requires decoupling the UI from the inference engine.

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ curl/CLI │  │ FTXUI    │  │ Web UI   │  │ LangChain/etc    │ │
│  │          │  │ Chat App │  │ (future) │  │ (OpenAI compat)  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │             │                  │           │
└───────┼─────────────┼─────────────┼──────────────────┼───────────┘
        │             │             │                  │
        └─────────────┴──────┬──────┴──────────────────┘
                             │ HTTP (localhost:8080)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Llaminar Server (--serve)                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   HTTP Server (Rank 0 only)                 ││
│  │  • Accepts connections                                      ││
│  │  • Parses OpenAI-format requests                            ││
│  │  • Queues inference requests                                ││
│  │  • Streams responses (SSE)                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Request Queue (thread-safe)               ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Inference Coordinator (all ranks)              ││
│  │  • Rank 0: Dequeues requests, broadcasts to other ranks     ││
│  │  • All ranks: Execute forward passes together               ││
│  │  • Rank 0: Samples tokens, streams back to HTTP handler     ││
│  └─────────────────────────────────────────────────────────────┘│
│       │              │              │              │             │
│       ▼              ▼              ▼              ▼             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│  │ Rank 0  │   │ Rank 1  │   │ Rank 2  │   │ Rank N  │          │
│  │ (NUMA0) │   │ (NUMA1) │   │ (NUMA2) │   │ (NUMA N)│          │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘          │
│       │              │              │              │             │
│       └──────────────┴──────────────┴──────────────┘             │
│                    MPI_COMM_WORLD                                │
└─────────────────────────────────────────────────────────────────┘
```

## API Specification

### Endpoints

Following the OpenAI API format for maximum compatibility:

#### 1. Chat Completions

```
POST /v1/chat/completions
```

**Request**:
```json
{
  "model": "qwen2.5-0.5b-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 256,
  "stream": true
}
```

**Response (non-streaming)**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1732900000,
  "model": "qwen2.5-0.5b-instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

**Response (streaming via SSE)**:
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{}}],"finish_reason":"stop"}

data: [DONE]
```

#### 2. Text Completions (Legacy)

```
POST /v1/completions
```

For raw text completion without chat formatting.

#### 3. Models List

```
GET /v1/models
```

Returns the currently loaded model.

#### 4. Health Check

```
GET /health
```

Returns server status and basic metrics.

## Implementation Phases

### Phase 1: Core Infrastructure

**Goal**: Basic HTTP server with non-streaming completions

**Components**:
1. **HTTP Library Selection**
   - Option A: [cpp-httplib](https://github.com/yhirose/cpp-httplib) - Header-only, simple
   - Option B: [Crow](https://github.com/CrowCpp/Crow) - Flask-like, more features
   - Option C: [Boost.Beast](https://www.boost.org/doc/libs/release/libs/beast/) - Part of Boost, robust
   - **Recommendation**: cpp-httplib for simplicity (single header, no dependencies)

2. **Server Mode Entry Point**
   - New `--serve` flag in ArgumentParser
   - `--port` option (default: 8080)
   - `--host` option (default: 127.0.0.1)

3. **Request/Response Types**
   - `ChatCompletionRequest` struct
   - `ChatCompletionResponse` struct
   - JSON serialization (nlohmann/json or rapidjson)

4. **MPI Coordination**
   - Rank 0: Runs HTTP server + inference coordinator
   - Other ranks: Run inference loop, wait for broadcast commands
   - Command types: `INFERENCE`, `SHUTDOWN`

**Files to Create**:
```
src/v2/server/
├── HttpServer.h/.cpp       # HTTP server wrapper
├── ApiTypes.h/.cpp         # Request/response structs
├── RequestQueue.h/.cpp     # Thread-safe queue
├── InferenceWorker.h/.cpp  # MPI-aware inference loop
└── ServerMain.cpp          # --serve entry point (or integrate into Main.cpp)
```

**Estimated Effort**: 2-3 days

### Phase 2: Streaming Support

**Goal**: Server-Sent Events (SSE) for token-by-token streaming

**Components**:
1. **SSE Response Handler**
   - Chunked transfer encoding
   - `text/event-stream` content type
   - Proper connection keep-alive

2. **Token Callback Integration**
   - Modify decode loop to call back per-token
   - Queue tokens for HTTP response thread

3. **Cancellation Support**
   - Client disconnect detection
   - Graceful generation abort

**Estimated Effort**: 1-2 days

### Phase 3: Production Features

**Goal**: Robustness and observability

**Components**:
1. **Request Queuing & Batching** (optional)
   - Multiple concurrent requests
   - Dynamic batching for throughput

2. **Metrics & Logging**
   - Prometheus-compatible `/metrics` endpoint
   - Request latency, tokens/sec, queue depth

3. **Error Handling**
   - Proper HTTP error codes
   - Graceful degradation

4. **Configuration**
   - Max concurrent requests
   - Request timeout
   - Rate limiting

**Estimated Effort**: 2-3 days

### Phase 4: Chat CLI Client

**Goal**: Standalone chat application using the API

**Components**:
1. **Separate Executable**: `llaminar-chat`
2. **FTXUI-based UI** (reuse existing ChatUI patterns)
3. **Connects to** `http://localhost:8080`
4. **Features**:
   - Conversation history
   - System prompt configuration
   - Streaming display
   - Multi-turn chat

**Files to Create**:
```
src/v2/clients/
├── ChatClient.h/.cpp       # HTTP client for API
└── ChatApp.cpp             # FTXUI chat application main
```

**Estimated Effort**: 1-2 days

## Dependencies

### Required
- **JSON Library**: nlohmann/json (header-only, already common in C++ projects)
- **HTTP Library**: cpp-httplib (header-only, ~15KB)

### Optional
- **HTTP Client** (for chat CLI): cpp-httplib also provides client functionality

## Command Line Interface

### Server Mode
```bash
# Start API server
./run_llaminar.sh --serve -m models/qwen2.5-0.5b-instruct-q4_0.gguf

# With options
./run_llaminar.sh --serve -m model.gguf --port 8080 --host 0.0.0.0

# With sampling defaults
./run_llaminar.sh --serve -m model.gguf -t 0.7 --top-k 40 --top-p 0.9
```

### Chat Client
```bash
# Connect to local server (default)
./llaminar-chat

# Connect to specific server
./llaminar-chat --server http://localhost:8080

# With system prompt
./llaminar-chat --system "You are a helpful coding assistant."
```

### API Usage Examples
```bash
# Simple completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

## Thread Model (Rank 0)

```
┌─────────────────────────────────────────────────────────┐
│                       Rank 0                            │
│                                                         │
│  ┌─────────────────┐     ┌─────────────────────────┐   │
│  │  HTTP Thread    │     │  Inference Thread       │   │
│  │                 │     │                         │   │
│  │  • Accept       │     │  • Dequeue request      │   │
│  │  • Parse JSON   │────▶│  • Broadcast to ranks   │   │
│  │  • Queue req    │     │  • Run forward pass     │   │
│  │  • Wait result  │◀────│  • Sample token         │   │
│  │  • Stream resp  │     │  • Queue result         │   │
│  └─────────────────┘     └─────────────────────────┘   │
│           │                         │                   │
│           ▼                         ▼                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Thread-Safe Request Queue             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## MPI Communication Protocol

### Command Broadcast (Rank 0 → All)
```cpp
enum class ServerCommand : int {
    INFERENCE = 1,    // Run inference with broadcast tokens
    SHUTDOWN = 2      // Graceful shutdown
};

struct InferenceCommand {
    int token_count;
    int max_new_tokens;
    float temperature;
    // ... other sampling params
};
```

### Inference Loop (Non-Rank-0)
```cpp
void worker_loop() {
    while (true) {
        ServerCommand cmd;
        MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (cmd == ServerCommand::SHUTDOWN) break;
        
        if (cmd == ServerCommand::INFERENCE) {
            InferenceCommand params;
            MPI_Bcast(&params, sizeof(params), MPI_BYTE, 0, MPI_COMM_WORLD);
            
            std::vector<int> tokens(params.token_count);
            MPI_Bcast(tokens.data(), params.token_count, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Prefill
            pipeline->forward(tokens.data(), params.token_count);
            
            // Decode loop
            for (int i = 0; i < params.max_new_tokens; ++i) {
                int next_token;
                MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);
                
                if (next_token < 0) break;  // EOS or abort signal
                
                pipeline->forward(&next_token, 1);
            }
        }
    }
}
```

## Testing Plan

### Unit Tests
- Request/response JSON parsing
- Queue operations
- SSE formatting

### Integration Tests
- Single request completion
- Streaming response
- Concurrent requests
- Client disconnect handling
- MPI coordination

### Manual Testing
- curl commands
- Chat CLI interaction
- Load testing with multiple clients

## Future Enhancements

1. **Continuous Batching**: Process multiple requests efficiently
2. **KV Cache Management**: Handle multiple conversations
3. **Model Hot-Swap**: Load different models without restart
4. **Authentication**: API key support
5. **TLS/HTTPS**: Secure connections
6. **WebSocket**: Alternative to SSE for bidirectional communication
7. **Web UI**: Simple browser-based chat interface

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HTTP library complexity | Low | Medium | Use cpp-httplib (simple, header-only) |
| MPI thread safety issues | Medium | High | Careful locking, MPI_THREAD_MULTIPLE |
| Streaming reliability | Medium | Medium | Proper SSE implementation, testing |
| Performance overhead | Low | Low | HTTP overhead minimal for LLM inference |

## Success Criteria

1. ✅ `--serve` mode starts HTTP server on rank 0
2. ✅ All MPI ranks participate in inference correctly
3. ✅ OpenAI-compatible `/v1/chat/completions` endpoint works
4. ✅ Streaming responses work with `stream: true`
5. ✅ `llaminar-chat` CLI provides interactive experience
6. ✅ No deadlocks under normal operation
7. ✅ Graceful shutdown on SIGINT/SIGTERM

## References

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [llama.cpp server implementation](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
