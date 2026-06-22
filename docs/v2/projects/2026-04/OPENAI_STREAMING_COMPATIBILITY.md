# OpenAI API Streaming & Compatibility Project Plan

**Date**: April 7, 2026
**Status**: In Progress
**Goal**: Full OpenAI Chat Completions API compatibility with SSE streaming and reasoning model support, enabling integration with clients like Open WebUI.

---

## Table of Contents
- [Motivation](#motivation)
- [Gap Analysis](#gap-analysis)
- [Implementation Plan](#implementation-plan)
  - [Phase 1: Response Metadata](#phase-1-response-metadata)
  - [Phase 2: Request Parameter Parsing](#phase-2-request-parameter-parsing)
  - [Phase 3: SSE Streaming Infrastructure](#phase-3-sse-streaming-infrastructure)
  - [Phase 4: Streaming Think/Content Split](#phase-4-streaming-thinkingcontent-split)
- [File Changes](#file-changes)
- [Testing Strategy](#testing-strategy)
- [Open WebUI Compatibility Notes](#open-webui-compatibility-notes)

---

## Motivation

Llaminar's HTTP server (`/v1/chat/completions`) supports non-streaming chat completions with `reasoning_content` extraction, but lacks:
1. **SSE streaming** — the primary mode used by clients like Open WebUI
2. **Complete response metadata** — missing `model`, `created`, unique `id`
3. **Reasoning control parameters** — `stream`, `enable_thinking`

Without streaming, Open WebUI shows no output until the entire response is generated, and cannot display reasoning content in its collapsible thinking UI (which relies on streaming `reasoning_content` deltas).

---

## Gap Analysis

| Feature | Current | Target (OpenAI-compatible) |
|---|---|---|
| Non-streaming `reasoning_content` | ✅ Works | ✅ Keep |
| `model` field in response | ❌ Missing | ✅ Model filename |
| `created` timestamp | ❌ Missing | ✅ Unix epoch seconds |
| Unique `id` per request | ❌ Hardcoded `"chatcmpl-llaminar"` | ✅ `chatcmpl-<uuid>` |
| `system_fingerprint` | ❌ Missing | ✅ `"llaminar-v2"` |
| `stream` request parameter | ❌ Not parsed | ✅ Parsed, routes to streaming |
| `enable_thinking` parameter | ❌ Not parsed | ✅ Parsed, controls `<think>` tag injection |
| SSE streaming response | ❌ Not implemented | ✅ Per-token SSE with `reasoning_content` in deltas |
| `[DONE]` sentinel | ❌ N/A | ✅ Sent after final chunk |

---

## Implementation Plan

### Phase 1: Response Metadata

**Goal**: Fix non-streaming response to include all required OpenAI fields.

**Changes**:
- Add `model` field to `ChatCompletionRequest` (extracted from request body or set from config)
- Add unique ID generation: `chatcmpl-<hex-random>` using `<random>`
- Add `created` field: `std::time(nullptr)` (Unix epoch seconds)
- Add `system_fingerprint`: `"llaminar-v2"`
- Pass model name from `OrchestrationConfig::model_path` through to handler

**Files**:
- `src/v2/app/modes/ChatCompletionHandler.h` — Add `model_name` to handler constructor
- `src/v2/app/modes/ChatCompletionHandler.cpp` — Generate metadata in response JSON
- `src/v2/app/modes/ServerMode.cpp` — Pass model name from config

### Phase 2: Request Parameter Parsing

**Goal**: Parse `stream` and `enable_thinking` from request body.

**Changes**:
- Add `bool stream` field to `ChatCompletionRequest` (default: `false`)
- Add `bool enable_thinking` field (default: `true` — matches OpenAI convention)
- Parse both in `parseRequest()`
- Wire `enable_thinking` through to chat template application

**Files**:
- `src/v2/app/modes/ChatCompletionHandler.h` — Extend `ChatCompletionRequest` struct
- `src/v2/app/modes/ChatCompletionHandler.cpp` — Parse new fields

### Phase 3: SSE Streaming Infrastructure

**Goal**: Implement Server-Sent Events streaming response path.

**Architecture**:
```
Client                    ServerMode                ChatCompletionHandler
  |--- POST /v1/chat ------>|                               |
  |                          |--- handleStreamingRequest --->|
  |                          |                               |
  |<-- "data: {...}\n\n" -- DataSink.write <-- per-token ----|
  |<-- "data: {...}\n\n" -- DataSink.write <-- per-token ----|
  |<-- "data: [DONE]\n\n" - DataSink.write <-- done ---------|
```

**cpp-httplib streaming**: Use `res.set_chunked_content_provider()` with `ContentProviderWithoutLength`. The provider callback receives a `DataSink` and writes SSE chunks synchronously. The inference loop runs inside the provider callback.

**Streaming chunk format** (OpenAI-compatible):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3.5-4b","system_fingerprint":"llaminar-v2","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3.5-4b","system_fingerprint":"llaminar-v2","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3.5-4b","system_fingerprint":"llaminar-v2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**New method**: `ChatCompletionHandler::handleStreamingRequest()` that takes a callback `std::function<bool(const std::string& sse_data)>` for emitting each SSE chunk.

**Files**:
- `src/v2/app/modes/ChatCompletionHandler.h` — Add streaming method and callback type
- `src/v2/app/modes/ChatCompletionHandler.cpp` — Implement streaming decode loop
- `src/v2/app/modes/ServerMode.cpp` — Route to streaming when `stream=true`

### Phase 4: Streaming Think/Content Split

**Goal**: During streaming, emit `reasoning_content` in deltas while inside `<think>` tags, then switch to `content` after `</think>`.

**Design**:
- Track a `ThinkingState` enum: `NOT_STARTED`, `IN_THINKING`, `AFTER_THINKING`
- Each decoded token's text is checked against the thinking end tag
- Buffer partial tag matches (the end tag may span multiple tokens)
- While `IN_THINKING`: emit `{"delta": {"reasoning_content": "..."}}`
- After `</think>`: emit `{"delta": {"content": "..."}}`
- First chunk always includes `{"delta": {"role": "assistant"}}` only

**Streaming delta field mapping** (Open WebUI compatible):

| Thinking State | Delta Field | Example |
|---|---|---|
| First chunk | `role` only | `{"role": "assistant"}` |
| IN_THINKING | `reasoning_content` | `{"reasoning_content": "Let me think..."}` |
| AFTER_THINKING | `content` | `{"content": "Paris is the capital."}` |

**Files**:
- `src/v2/app/modes/ChatCompletionHandler.cpp` — Implement `StreamingThinkSplitter` class

---

## File Changes

| File | Change Type | Description |
|---|---|---|
| `src/v2/app/modes/ChatCompletionHandler.h` | Modified | Add streaming types, extend request struct, add model_name |
| `src/v2/app/modes/ChatCompletionHandler.cpp` | Modified | Streaming implementation, metadata, think splitter |
| `src/v2/app/modes/ServerMode.cpp` | Modified | Route streaming requests, pass model name |
| `tests/v2/unit/app/modes/Test__ChatCompletionHandler.cpp` | Modified | New tests for all features |

---

## Testing Strategy

All features are unit-testable via the existing mock infrastructure (`MockOrchestrationRunner`, `MockTokenizer`).

### Test Categories

**Response Metadata Tests**:
- Unique ID format (`chatcmpl-` prefix, hex suffix)
- Two requests produce different IDs
- `model` field present and correct
- `created` timestamp is a recent Unix epoch
- `system_fingerprint` is `"llaminar-v2"`

**Request Parsing Tests**:
- `stream: true` parsed correctly
- `stream: false` is default
- `enable_thinking: false` parsed
- `enable_thinking: true` is default

**Streaming Tests**:
- First chunk has `role: "assistant"` in delta
- Token-by-token content in delta chunks
- Final chunk has `finish_reason: "stop"` or `"length"`
- `[DONE]` sentinel emitted last
- Chunk `object` is `"chat.completion.chunk"`
- Chunk ID is consistent across all chunks
- Stop token terminates stream with `finish_reason: "stop"`
- Max tokens terminates stream with `finish_reason: "length"`
- Prefill failure calls error callback
- Decode failure terminates stream gracefully

**Streaming Think/Content Split Tests**:
- Thinking model: `reasoning_content` in deltas during `<think>` phase
- Thinking model: `content` in deltas after `</think>`
- Think end tag spanning multiple tokens handled correctly
- Non-thinking model: all content in `content` field
- Think split with `enable_thinking: false` disables splitting

---

## Open WebUI Compatibility Notes

Open WebUI detects reasoning content via two mechanisms:
1. **Streaming delta fields**: `reasoning_content`, `reasoning`, or `thinking` field in streaming delta objects
2. **Text-based parsing**: `<think>...</think>` tags in accumulated `content`

Our implementation uses mechanism (1) — the `reasoning_content` field in streaming deltas — which is the recommended approach for OpenAI-compatible backends. This avoids double-rendering of thinking content in Open WebUI's UI.

Open WebUI defaults to `stream: true` for all providers, making streaming support the primary compatibility requirement.

**CORS**: Open WebUI may send preflight OPTIONS requests. cpp-httplib handles these automatically when CORS headers are set. We add `Access-Control-Allow-Origin: *` and related headers.
