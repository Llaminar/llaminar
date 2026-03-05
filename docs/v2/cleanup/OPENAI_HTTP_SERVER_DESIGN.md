# OpenAI-Compatible HTTP Server Design

**Date**: 2025-03-05  
**Status**: Planned  
**Depends On**: [Main.cpp Refactor Plan](MAIN_CPP_REFACTOR_PLAN.md) (Phase 5: `ServerMode`)

---

## Overview

Llaminar will serve an OpenAI-compatible HTTP API on Rank 0 whenever the engine boots without an explicit one-off prompt. This becomes the **default launch mode**: load the model, start serving requests over HTTP, and run until `SIGTERM` / `CTRL-C`.

**Non-goals for V1**: TLS termination (use a reverse proxy), multi-model serving, batched continuous batching (single-request-at-a-time queue first).

---

## HTTP Library Selection

### Requirements

- Header-only or easily vendored via `FetchContent`
- Mature, actively maintained, production-hardened
- Async or threaded listener (we only need Rank 0 to serve; inference is synchronous)
- Plain HTTP (no TLS requirement, but nice to have for future)
- Minimal dependencies (no Boost requirement)
- Permissive license (MIT/BSD/Apache-2.0)

### Candidates Evaluated

| Library | License | Header-Only | Async | Maturity | Verdict |
|---------|---------|-------------|-------|----------|---------|
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | MIT | Yes (single header) | Thread-pool | 6k+ stars, actively maintained | **Selected** |
| Boost.Beast | BSL-1.0 | No (needs Boost.Asio) | Full async | Very mature | Too heavy; adds Boost dependency |
| Drogon | MIT | No | Full async | Production-grade | Overkill; pulls in trantor, jsoncpp |
| Crow | BSD | Header-only | Thread-pool | Good | Less active than cpp-httplib |
| Pistache | Apache-2.0 | No | Async | Linux-only | Not cross-platform |

### Why cpp-httplib

- **Single header** (~30K lines, but zero build complexity) — fits our existing `FetchContent` pattern
- **Thread-pool listener** — perfect for our model: Rank 0 accepts HTTP, queues to inference thread
- **Streaming response support** — `ContentProvider` callback for SSE (Server-Sent Events)
- **11 years of development**, regular releases, well-tested against edge cases
- **No dependencies** beyond `<thread>`, `<mutex>`, `<condition_variable>` (all C++17 stdlib)
- Optional OpenSSL for HTTPS if ever needed

### CMake Integration

```cmake
# =============================================================================
# cpp-httplib (for OpenAI-compatible HTTP server)
# =============================================================================
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
    GIT_TAG v0.18.3
    GIT_SHALLOW TRUE
)
set(HTTPLIB_COMPILE OFF CACHE BOOL "Header-only mode" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "No TLS requirement" FORCE)
FetchContent_MakeAvailable(httplib)
message(STATUS "V2: cpp-httplib v0.18.3 configured for HTTP server")
```

### JSON Library

We need JSON serialization for request/response bodies. Options:

| Library | Approach | Notes |
|---------|----------|-------|
| [nlohmann/json](https://github.com/nlohmann/json) | Single header, MIT | De facto standard for C++ JSON |
| [simdjson](https://github.com/simdjson/simdjson) | Parse-only, fast | No serialization |
| [rapidjson](https://github.com/Tencent/rapidjson) | SAX/DOM, fast | More verbose API |

**Selected: nlohmann/json** — ubiquitous, excellent ergonomics, header-only, MIT.

```cmake
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "Disable json tests" FORCE)
FetchContent_MakeAvailable(nlohmann_json)
```

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              ServerMode                   │
                    │         (IExecutionMode impl)             │
                    └─────────┬───────────────────────────────┘
                              │ owns
                    ┌─────────▼───────────────────────────────┐
                    │            HttpServer                     │
                    │  cpp-httplib listener (thread pool)       │
                    │  Routes → handler functions               │
                    └─────────┬───────────────────────────────┘
                              │ dispatches to
                    ┌─────────▼───────────────────────────────┐
                    │         RequestQueue                      │
                    │  std::deque + mutex + condition_variable  │
                    │  Bounded capacity with backpressure       │
                    └─────────┬───────────────────────────────┘
                              │ consumed by
                    ┌─────────▼───────────────────────────────┐
                    │       InferenceWorker                     │
                    │  Single thread: dequeue → prefill →      │
                    │  decode loop → write response             │
                    │  Owns IOrchestrationRunner exclusively    │
                    └─────────────────────────────────────────┘
```

### Key Design Decisions

1. **Single inference thread**: `IOrchestrationRunner` is not thread-safe. One dedicated thread drains the queue and runs prefill/decode. The HTTP listener thread pool accepts connections and enqueues.

2. **Request queue with backpressure**: Bounded queue (configurable, default 64). When full, new requests get `HTTP 503 Service Unavailable` immediately.

3. **Streaming via SSE**: For `stream: true` requests, the inference worker writes tokens to a per-request pipe/ring-buffer. The HTTP handler reads from it using `httplib::ContentProvider`.

4. **Rank 0 only**: The HTTP server runs exclusively on Rank 0. Other MPI ranks participate in inference via the normal MPI collective path — they don't know about HTTP at all. The inference worker thread calls `runner->prefill()` and `runner->decodeStep()` which internally trigger MPI collectives that non-rank-0 processes participate in.

---

## File Structure

```
src/v2/app/modes/server/
├── ServerMode.h/cpp              # IExecutionMode: startup, signal handling, shutdown
├── HttpServer.h/cpp              # Route registration, listener lifecycle
├── RequestQueue.h/cpp            # Thread-safe bounded queue
├── InferenceWorker.h/cpp         # Inference loop: dequeue → generate → respond
├── handlers/
│   ├── ChatCompletionHandler.h/cpp    # POST /v1/chat/completions
│   ├── CompletionHandler.h/cpp        # POST /v1/completions
│   ├── ModelHandler.h/cpp             # GET /v1/models
│   └── HealthHandler.h/cpp            # GET /health, GET /v1/health
├── types/
│   ├── ChatCompletionRequest.h        # Request DTOs with JSON de/serialization
│   ├── ChatCompletionResponse.h       # Response DTOs
│   ├── CompletionRequest.h
│   ├── CompletionResponse.h
│   ├── ModelInfo.h
│   ├── ErrorResponse.h
│   └── Usage.h                        # Token usage tracking
├── middleware/
│   ├── ApiKeyAuth.h/cpp               # Optional Bearer token validation
│   └── RequestLogger.h/cpp            # Access logging
└── streaming/
    └── SSEWriter.h/cpp                # Server-Sent Events formatting
```

---

## API Endpoints

### `GET /health`

Health check for load balancers and monitoring.

```json
// Response 200
{
  "status": "ok",
  "model": "qwen2.5-0.5b-instruct-q4_0",
  "ready": true,
  "uptime_seconds": 3642,
  "queue_depth": 2,
  "queue_capacity": 64
}
```

Returns `503` with `"ready": false` during model loading or shutdown.

### `GET /v1/models`

OpenAI-compatible model listing.

```json
// Response 200
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-0.5b-instruct-q4_0",
      "object": "model",
      "created": 1709654400,
      "owned_by": "llaminar"
    }
  ]
}
```

The model `id` is derived from the GGUF filename (minus `.gguf` extension).

### `POST /v1/chat/completions`

OpenAI-compatible chat completions.

**Request**:
```json
{
  "model": "qwen2.5-0.5b-instruct-q4_0",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": false,
  "stop": ["<|im_end|>"],
  "seed": 42,
  "logprobs": false,
  "top_logprobs": null,
  "n": 1,
  "reasoning_effort": "medium"
}
```

**Non-streaming Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709654400,
  "model": "qwen2.5-0.5b-instruct-q4_0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

**Streaming Response** (`stream: true`):

Each chunk is an SSE event:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1709654400,"model":"qwen2.5-0.5b-instruct-q4_0","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1709654400,"model":"qwen2.5-0.5b-instruct-q4_0","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1709654400,"model":"qwen2.5-0.5b-instruct-q4_0","choices":[{"index":0,"delta":{"content":" +"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1709654400,"model":"qwen2.5-0.5b-instruct-q4_0","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":8,"total_tokens":32}}

data: [DONE]
```

### `POST /v1/completions`

Legacy text completions endpoint.

**Request**:
```json
{
  "model": "qwen2.5-0.5b-instruct-q4_0",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "stream": false
}
```

**Response**: Same structure as OpenAI's completions API with `"object": "text_completion"`.

---

## Supported Request Parameters

### Standard OpenAI Parameters

| Parameter | Type | Default | Supported | Notes |
|-----------|------|---------|-----------|-------|
| `model` | string | — | Yes | Validated against loaded model name |
| `messages` | array | — | Yes | Chat completions only |
| `prompt` | string | — | Yes | Text completions only |
| `max_tokens` | int | model max | Yes | Maps to `n_predict` |
| `temperature` | float | 0.8 | Yes | Maps to `SamplingParams::temperature` |
| `top_p` | float | 0.9 | Yes | Maps to `SamplingParams::top_p` |
| `top_k` | int | 40 | Yes | Extension: not in OpenAI spec, but widely used |
| `n` | int | 1 | Partial | Only `n=1` initially; >1 requires KV cache branching |
| `stream` | bool | false | Yes | SSE streaming |
| `stop` | string/array | null | Yes | Additional stop sequences (tokenized and checked) |
| `seed` | int | null | Yes | Deterministic sampling when set |
| `logprobs` | bool | false | Planned | Requires logit extraction per token |
| `top_logprobs` | int | null | Planned | Top-N log probabilities per token |
| `presence_penalty` | float | 0.0 | Planned | Not in V1 |
| `frequency_penalty` | float | 0.0 | Planned | Not in V1 |
| `user` | string | null | Yes | Logged for audit, not used for inference |

### Reasoning Parameters (Extended)

For reasoning models (e.g., DeepSeek-R1, QwQ), the API supports OpenAI's reasoning parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_effort` | string | `"medium"` | `"low"`, `"medium"`, `"high"` — controls reasoning depth |

**Implementation**: `reasoning_effort` maps to inference behavior:

| Effort | Behavior |
|--------|----------|
| `"low"` | `max_tokens` capped to 1024, temperature lowered to 0.3 |
| `"medium"` | Default sampling parameters, no modifications |
| `"high"` | `max_tokens` uncapped (model maximum), temperature raised to 1.0 for diverse reasoning chains |

For models that emit `<think>...</think>` blocks, the response includes both reasoning and final content. The API does **not** strip reasoning tokens — the client decides how to handle them.

### Unsupported Parameters (Silently Ignored)

These OpenAI API parameters are accepted but have no effect in V1:

- `tools` / `tool_choice` (function calling)
- `response_format` (JSON mode)
- `service_tier`

---

## Authentication

### API Key Authentication (Optional)

Enabled via CLI flag or environment variable:

```bash
# Via CLI
./llaminar2 --api-key "sk-my-secret-key" -m model.gguf

# Via environment variable
LLAMINAR_API_KEY="sk-my-secret-key" ./llaminar2 -m model.gguf
```

When enabled, all `/v1/*` endpoints require:

```
Authorization: Bearer sk-my-secret-key
```

**Behavior**:
- `/health` endpoint is **always unauthenticated** (for health checks / load balancers)
- Missing or invalid key returns `HTTP 401 Unauthorized`:
  ```json
  {
    "error": {
      "message": "Invalid API key",
      "type": "authentication_error",
      "code": "invalid_api_key"
    }
  }
  ```
- Multiple API keys can be specified (comma-separated):
  ```bash
  LLAMINAR_API_KEY="sk-key1,sk-key2" ./llaminar2 -m model.gguf
  ```

### Implementation

```cpp
class ApiKeyAuth {
public:
    explicit ApiKeyAuth(const std::string& keys_csv);

    /// Returns true if auth is disabled (no keys configured)
    bool isDisabled() const;

    /// Validate request. Returns empty string on success, error message on failure.
    std::string validate(const httplib::Request& req) const;

private:
    std::set<std::string> valid_keys_;
};
```

Middleware is installed on the `httplib::Server` as a pre-routing handler.

---

## Request Queue and Inference Worker

### RequestQueue

```cpp
struct InferenceRequest {
    std::string id;                           // "chatcmpl-<uuid>"
    RequestType type;                         // CHAT_COMPLETION or COMPLETION
    std::vector<ChatMessage> messages;        // For chat completions
    std::string prompt;                       // For text completions
    SamplingParams sampling;
    int max_tokens;
    std::vector<std::string> stop_sequences;  // Additional stop strings
    bool stream;
    std::string model;                        // Requested model name
    std::string user;                         // Optional user identifier

    // Reasoning
    std::string reasoning_effort;             // "low", "medium", "high"

    // Response channel: inference worker writes here, HTTP handler reads
    std::shared_ptr<ResponseChannel> response;
};

class RequestQueue {
public:
    explicit RequestQueue(size_t capacity = 64);

    /// Enqueue a request. Returns false if queue is full (backpressure).
    bool enqueue(InferenceRequest req);

    /// Dequeue next request (blocks until available or shutdown).
    std::optional<InferenceRequest> dequeue();

    /// Signal shutdown (unblocks waiting dequeue).
    void shutdown();

    /// Current queue depth.
    size_t depth() const;

    /// Maximum capacity.
    size_t capacity() const;

private:
    std::deque<InferenceRequest> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t capacity_;
    bool shutting_down_ = false;
};
```

### ResponseChannel

Bridge between the inference worker (producer) and the HTTP handler (consumer):

```cpp
class ResponseChannel {
public:
    /// Write a token to the channel (called by inference worker)
    void writeToken(const std::string& text, int32_t token_id);

    /// Signal completion with finish reason and usage
    void complete(const std::string& finish_reason, const Usage& usage);

    /// Signal error
    void error(const std::string& message);

    /// Read next event (blocks until available). Used by HTTP handler for streaming.
    ResponseEvent readEvent();

    /// Get full response (blocks until complete). Used for non-streaming.
    FullResponse waitForCompletion();

private:
    std::queue<ResponseEvent> events_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool completed_ = false;
};
```

### InferenceWorker

```cpp
class InferenceWorker {
public:
    InferenceWorker(
        IOrchestrationRunner* runner,
        std::shared_ptr<ITokenizer> tokenizer,
        RequestQueue& queue);

    /// Start the worker thread
    void start();

    /// Stop the worker (waits for current request to finish + drain)
    void stop();

private:
    void workerLoop();
    void processRequest(InferenceRequest& req);
    void processChatCompletion(InferenceRequest& req);
    void processCompletion(InferenceRequest& req);

    IOrchestrationRunner* runner_;
    std::shared_ptr<ITokenizer> tokenizer_;
    RequestQueue& queue_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};
```

---

## Signal Handling and Graceful Shutdown

### Shutdown Sequence

```
SIGTERM / SIGINT (CTRL-C)
    │
    ▼
ServerMode::signalHandler()
    │  sets atomic shutdown_requested_ = true
    │
    ▼
HttpServer::stop()
    │  httplib::Server::stop() — stops accepting new connections
    │  In-flight HTTP handlers finish naturally
    │
    ▼
RequestQueue::shutdown()
    │  Rejects new enqueues
    │  Unblocks dequeue() with nullopt
    │
    ▼
InferenceWorker::stop()
    │  Finishes current request (does NOT abort mid-generation)
    │  Drains remaining queue (or discards with 503)
    │  Joins worker thread
    │
    ▼
IOrchestrationRunner::shutdown()
    │  Frees device memory, MPI teardown
    │
    ▼
MPI_Finalize() → exit(0)
```

### Implementation

```cpp
// In ServerMode::execute()
static std::atomic<bool> shutdown_requested{false};

auto signal_handler = [](int sig) {
    shutdown_requested.store(true);
};

std::signal(SIGTERM, signal_handler);
std::signal(SIGINT, signal_handler);

// Main loop on Rank 0
http_server.listenInBackground(host, port);

while (!shutdown_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Graceful shutdown
LOG_INFO("[Server] Shutdown signal received, draining...");
http_server.stop();
inference_worker.stop();
```

### Timeout Behavior

- **Current request**: Allowed to finish (up to `max_tokens` generation). No mid-token abort.
- **Queued requests**: On shutdown, remaining queued requests receive `503 Service Unavailable` via their `ResponseChannel::error()`.
- **Shutdown timeout**: If the current request doesn't finish within 30 seconds after signal, force-terminate.

---

## MPI Rank Coordination

The HTTP server runs only on Rank 0. Non-rank-0 processes need to know when to participate in inference and when to shut down.

### Approach: Inference Worker Drives MPI

The existing `IOrchestrationRunner::prefill()` and `decodeStep()` internally trigger MPI collectives (AllReduce, AllGather) that all ranks must participate in. This means:

1. **Rank 0** (HTTP server): The `InferenceWorker` thread calls `prefill()` / `decodeStep()` — these block until all ranks complete.
2. **Non-rank-0**: Must be in a matching loop that calls the same collective operations.

This is already how the CLI modes work. The `ServerMode` on non-rank-0 processes will enter a **follower loop**:

```cpp
// Non-rank-0 processes in ServerMode
void ServerMode::followerLoop(AppContext& ctx) {
    while (true) {
        // Wait for rank 0 to broadcast command (PREFILL, DECODE, SHUTDOWN)
        Command cmd = receiveCommand(ctx.mpi_ctx);

        switch (cmd.type) {
            case CommandType::PREFILL:
                ctx.runner->prefill(cmd.tokens);
                break;
            case CommandType::DECODE_STEP:
                ctx.runner->decodeStep();
                break;
            case CommandType::CLEAR_CACHE:
                ctx.runner->clearCache();
                break;
            case CommandType::SHUTDOWN:
                return;
        }
    }
}
```

**Command broadcasting** uses a lightweight MPI protocol:
- Rank 0 broadcasts a small header (command type + payload size)
- Followed by payload (token IDs for prefill, nothing for decode/shutdown)
- This is the same pattern used by the existing `--chat-single` mode (MPI_Bcast of token count + tokens)

---

## CLI Flags

New flags added to `OrchestrationConfig` / `OrchestrationConfigParser`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--serve` | bool | false | Explicitly enable server mode |
| `--host` | string | `"0.0.0.0"` | Listen address |
| `--port` | int | `8080` | Listen port |
| `--api-key` | string | `""` | API key(s), comma-separated |
| `--queue-size` | int | `64` | Max queued requests |
| `--request-timeout` | int | `300` | Per-request timeout in seconds |

**Default mode behavior**: When no `-p` prompt, `--chat`, `--chat-single`, or `--benchmark` flag is given, `ServerMode::matches()` returns true and the server starts automatically. The `--serve` flag is for explicitly requesting server mode even when other flags are present.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_API_KEY` | API key(s), comma-separated | (none — auth disabled) |
| `LLAMINAR_SERVER_HOST` | Listen address | `0.0.0.0` |
| `LLAMINAR_SERVER_PORT` | Listen port | `8080` |
| `LLAMINAR_SERVER_QUEUE_SIZE` | Max queued requests | `64` |

CLI flags take precedence over environment variables.

---

## Startup Banner

When the server starts, Rank 0 prints:

```
╔══════════════════════════════════════════════════════════════╗
║                    Llaminar Inference Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  Model:    qwen2.5-0.5b-instruct-q4_0.gguf                  ║
║  Device:   cuda:0 (NVIDIA RTX 4090, 24GB)                   ║
║  TP:       2-way LOCAL (NCCL)                                ║
║  API:      http://0.0.0.0:8080                               ║
║  Auth:     API key required                                  ║
║  Queue:    0/64                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /v1/chat/completions                                 ║
║    POST /v1/completions                                      ║
║    GET  /v1/models                                           ║
║    GET  /health                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Press Ctrl-C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
```

(Rendered with libfort, per project guidelines.)

---

## Error Responses

All error responses follow the OpenAI error format:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "error_type",
    "param": null,
    "code": "error_code"
  }
}
```

| HTTP Code | `type` | `code` | When |
|-----------|--------|--------|------|
| 400 | `invalid_request_error` | `invalid_request` | Malformed JSON, missing required fields |
| 401 | `authentication_error` | `invalid_api_key` | Missing or invalid API key |
| 404 | `not_found` | `model_not_found` | Requested model doesn't match loaded model |
| 422 | `invalid_request_error` | `invalid_parameter` | Parameter validation failure (e.g., `temperature < 0`) |
| 429 | `rate_limit_error` | `queue_full` | Request queue at capacity |
| 500 | `server_error` | `inference_error` | Prefill or decode failure |
| 503 | `server_error` | `service_unavailable` | Shutting down or not ready |

---

## Request ID Generation

Each request gets a unique ID in the format `chatcmpl-<uuid>` (chat) or `cmpl-<uuid>` (text).

UUID generation uses a simple counter + timestamp approach (no external UUID library needed):

```cpp
std::string generateRequestId(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << prefix << std::hex << ms << "-" << seq;
    return oss.str();
}
```

---

## Token Usage Tracking

```cpp
struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens() const { return prompt_tokens + completion_tokens; }
};
```

- `prompt_tokens`: Counted after tokenization (before prefill)
- `completion_tokens`: Incremented in the decode loop
- Included in both streaming (final chunk only) and non-streaming responses

---

## Implementation Phases

### Phase 5a: Infrastructure

1. Add `cpp-httplib` and `nlohmann/json` to CMake via `FetchContent`
2. Create `src/v2/app/modes/server/` directory structure
3. Implement `RequestQueue`, `ResponseChannel`
4. Implement `ApiKeyAuth` middleware
5. Add `--serve`, `--host`, `--port`, `--api-key`, `--queue-size` to `OrchestrationConfigParser`
6. Add corresponding fields to `OrchestrationConfig`

### Phase 5b: Core Server

1. Implement `HttpServer` with route registration
2. Implement `InferenceWorker` (single-thread queue drain, prefill/decode)
3. Implement `HealthHandler`, `ModelHandler`
4. Implement `ChatCompletionHandler` (non-streaming first)
5. Implement `CompletionHandler` (non-streaming)
6. Wire into `ServerMode::execute()` on Rank 0
7. Implement MPI follower loop for non-rank-0

### Phase 5c: Streaming

1. Implement `SSEWriter` for Server-Sent Events formatting
2. Add streaming path to `ChatCompletionHandler` using `httplib::ContentProvider`
3. Add streaming path to `CompletionHandler`

### Phase 5d: Polish

1. Startup banner (libfort)
2. Signal handling (`SIGTERM`, `SIGINT`)
3. Graceful shutdown with drain timeout
4. Request logging middleware
5. `reasoning_effort` parameter support

### Phase 5e: Testing

1. Unit tests for `RequestQueue`, `ResponseChannel`, `ApiKeyAuth`
2. Unit tests for request/response JSON serialization
3. Integration test: start server, `curl` against endpoints, verify responses
4. Streaming integration test with SSE parsing
5. Shutdown test: send SIGTERM during active generation

---

## Future Enhancements (Not in V1)

- **Continuous batching**: Process multiple requests concurrently with dynamic batching
- **KV cache management**: Per-request KV cache slots, prefix caching for common system prompts
- **Function calling**: `tools` / `tool_choice` parameter support
- **JSON mode**: `response_format: { type: "json_object" }` with constrained decoding
- **Logprobs**: Per-token log probabilities in responses
- **Prometheus metrics**: `/metrics` endpoint for monitoring
- **Rate limiting**: Per-key rate limits
- **Request cancellation**: Client disconnect detection → abort generation
- **Multi-model**: Hot-swap models via admin API
