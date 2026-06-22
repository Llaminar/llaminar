# Main.cpp Refactor Plan

**Date**: 2025-03-05  
**Status**: Planned  
**Motivation**: `Main.cpp` is 1,495 lines and growing. An OpenAI-compatible HTTP server mode is planned as the default launch mode, which will be impractical to splice into the current monolith.

---

## Current State Analysis

`src/v2/Main.cpp` contains six distinct responsibilities in a single file:

| Section | Approx Lines | Responsibility |
|---------|-------------|----------------|
| Anonymous helpers | ~280 | NUMA detection, CPU list parsing, affinity verification |
| `list_devices()` | ~40 | Device enumeration output |
| Pre-MPI bootstrap | ~270 | Topology planning, NUMA resolution, MPI self-launch |
| Post-MPI runtime init | ~200 | MPI init, NUMA verify, DeviceManager, runner creation |
| Chat template override | ~50 | String→enum mapping |
| Mode dispatch (chat/single-shot/benchmark/inference) | ~520 | Four modes with two duplicate adapter classes |

### Key Pain Points

1. **Duplicate adapters**: `OrchestrationRunnerAdapter` (chat mode, ~line 1003) and `BenchmarkRunnerAdapter` (benchmark mode, ~line 1237) are nearly identical `IOrchestrationRunner → IInferenceRunner` wrappers defined as local classes inside `main()`.
2. **Monolithic `main()`**: ~1,100 lines of sequential logic with deeply nested MPI rank-gating (`if (mpi_ctx->rank() == 0)`).
3. **No extension point**: Adding a new execution mode (e.g., HTTP server) means splicing another 200+ line block into the middle of `main()`.
4. **Untestable**: The bootstrap/init sequence can't be unit tested because it lives entirely in `main()`.

---

## Target Structure

```
src/v2/
├── Main.cpp                              # ~30 lines: parse → AppLifecycle::run()
├── app/
│   ├── AppLifecycle.h/cpp                # Owns full startup: bootstrap → init → dispatch
│   ├── AppContext.h                      # Shared state struct passed to modes
│   ├── MPIBootstrapPhase.h/cpp           # Pre-MPI: NUMA resolution, topology, self-launch
│   ├── RuntimeInitPhase.h/cpp            # Post-MPI: affinity verify, DeviceManager, runner
│   ├── InferenceRunnerAdapter.h/cpp      # One adapter: IOrchestrationRunner → IInferenceRunner
│   ├── ChatTemplateResolver.h/cpp        # string → ChatTemplateType mapping
│   └── modes/
│       ├── IExecutionMode.h              # Interface for pluggable modes
│       ├── CompletionMode.h/cpp          # Standard one-shot inference (-p "...")
│       ├── InteractiveChatMode.h/cpp     # --chat
│       ├── SingleShotChatMode.h/cpp      # --chat-single
│       ├── BenchmarkMode.h/cpp           # --benchmark
│       └── ServerMode.h/cpp              # Future: OpenAI-compatible HTTP API
```

---

## Core Interfaces

### `IExecutionMode` — Pluggable Mode Dispatch

```cpp
class IExecutionMode {
public:
    virtual ~IExecutionMode() = default;

    /// Human-readable mode name (for logging/diagnostics)
    virtual const char* name() const = 0;

    /// Does this mode handle the given config?
    virtual bool matches(const OrchestrationConfig& config) const = 0;

    /// Execute the mode. Returns process exit code.
    virtual int execute(AppContext& ctx) = 0;
};
```

### `AppContext` — Shared State Passed to Modes

Replaces the loose local variables currently scattered through `main()`:

```cpp
struct AppContext {
    OrchestrationConfig config;
    std::shared_ptr<MPIContext> mpi_ctx;
    std::unique_ptr<IOrchestrationRunner> runner;
    std::shared_ptr<ITokenizer> tokenizer;

    /// Convenience: is this rank 0?
    bool isRootRank() const { return mpi_ctx->rank() == 0; }
};
```

### `AppLifecycle` — Top-Level Orchestrator

```cpp
class AppLifecycle {
public:
    int run(int argc, char* argv[]);

private:
    // Phase 1: Pre-MPI (may execvp into mpirun and never return)
    int bootstrapMPI(OrchestrationConfig& config);

    // Phase 2: Post-MPI init (returns nullopt on failure)
    std::optional<AppContext> initializeRuntime(OrchestrationConfig config);

    // Phase 3: Mode dispatch (first matching mode wins)
    int dispatch(AppContext& ctx);

    // Registered modes in priority order
    std::vector<std::unique_ptr<IExecutionMode>> modes_;
};
```

---

## File-by-File Extraction Map

| New File | Extracted From | Content |
|----------|---------------|---------|
| `AppLifecycle` | `main()` skeleton | Top-level `run()` calling bootstrap → init → dispatch |
| `AppContext` | Loose locals in `main()` | Struct holding config, mpi_ctx, runner, tokenizer |
| `MPIBootstrapPhase` | Anonymous namespace + pre-MPI `if` block | `resolveInferenceNUMANodes`, `parseCpuList`, `detectCpuNumaNode`, `physicalRepresentativeForCpu`, `verifyStartupThreadAffinity`, and the ~270-line bootstrap block (topology planning, cpu-set resolution, MPI self-launch) |
| `RuntimeInitPhase` | Post-MPI init block | `MPI_Init_thread`, NUMA detection, affinity verification, CPU shorthand runtime mapping, DeviceManager init, `OrchestrationRunner` creation, tokenizer acquisition |
| `InferenceRunnerAdapter` | Two duplicate local classes | Single shared `IOrchestrationRunner → IInferenceRunner` adapter replacing both `OrchestrationRunnerAdapter` and `BenchmarkRunnerAdapter`. Superset of both (includes `executorStats`, `sampleGreedyOnDevice`, etc.) |
| `ChatTemplateResolver` | Chat template override block | The `if/else if` chain mapping strings to `ChatTemplateType` enums |
| `CompletionMode` | Standard inference block (~L1330-1495) | Tokenize → prefill → decode loop |
| `InteractiveChatMode` | Chat mode block (~L980-1070) | Chat UI setup and run |
| `SingleShotChatMode` | Single-shot chat block (~L1073-1220) | Conversation encoding, MPI broadcast, prefill, decode loop |
| `BenchmarkMode` | Benchmark block (~L1225-1325) | Adapter creation, `BenchmarkRunner` invocation |
| `ServerMode` | **New** (future) | OpenAI-compatible HTTP API |

---

## Resulting Main.cpp (~30 lines)

```cpp
#include "app/AppLifecycle.h"

int main(int argc, char* argv[]) {
    llaminar2::AppLifecycle app;
    return app.run(argc, argv);
}
```

---

## Server Mode Sketch (Future)

With the `IExecutionMode` interface, the HTTP server becomes a natural addition:

```cpp
class ServerMode : public IExecutionMode {
public:
    const char* name() const override { return "server"; }

    bool matches(const OrchestrationConfig& config) const override {
        // Default mode when no prompt given and not in chat/benchmark mode
        return config.prompt.empty()
            && !config.chat_mode
            && !config.benchmark_mode
            && !config.single_shot_chat;
    }

    int execute(AppContext& ctx) override {
        HttpServer server(ctx.runner, ctx.tokenizer);
        server.registerRoute("POST", "/v1/chat/completions", ...);
        server.registerRoute("POST", "/v1/completions", ...);
        server.registerRoute("GET",  "/v1/models", ...);
        return server.listen(ctx.config.port);
    }
};
```

---

## Migration Phases

Each phase is independently compilable and testable. No phase changes external behavior.

### Phase 1: Extract `InferenceRunnerAdapter`

- Move to `src/v2/app/InferenceRunnerAdapter.h/cpp`
- Superset of both existing local adapter classes (include `executorStats`, `sampleOnDevice`, `setSkipLogitsGather*`)
- Update both usage sites in `main()` to use the shared adapter
- Delete both local class definitions
- **Validation**: All existing tests pass unchanged

### Phase 2: Extract Bootstrap & Init Phases

- Move anonymous namespace helpers to `MPIBootstrapPhase`
- Move pre-MPI bootstrap block to `MPIBootstrapPhase::execute()`
- Move post-MPI init block to `RuntimeInitPhase::execute()`
- Move chat template parsing to `ChatTemplateResolver::resolve()`
- `main()` calls `MPIBootstrapPhase` → `RuntimeInitPhase` → mode dispatch
- **Validation**: All existing tests pass; manual `--dry-run` and `--show-topology` still work

### Phase 3: Extract Execution Modes

- Create `IExecutionMode` interface
- Extract `CompletionMode`, `InteractiveChatMode`, `SingleShotChatMode`, `BenchmarkMode`
- Each mode gets its own `.h/.cpp` in `src/v2/app/modes/`
- `main()` iterates modes, first match wins
- **Validation**: All inference modes produce identical output

### Phase 4: Wire `AppLifecycle`

- Create `AppLifecycle` class that assembles all phases
- Create `AppContext` struct
- Shrink `Main.cpp` to the ~30-line stub
- **Validation**: Full regression test pass

### Phase 5: Add `ServerMode`

- Implement OpenAI-compatible HTTP API
- Register as default mode (matches when no prompt/chat/benchmark flags present)
- Requires HTTP library selection (e.g., cpp-httplib, Boost.Beast, or custom)
- **Validation**: `curl` against `/v1/chat/completions` produces correct streaming responses

---

## CMake Integration Notes

- New `src/v2/app/` directory needs to be added to `llaminar2_core` sources in `CMakeLists.txt`
- No new external dependencies until Phase 5 (HTTP library)
- `app/modes/ServerMode.cpp` can be conditionally compiled behind a `HAVE_HTTP_SERVER` option

---

## Testing Strategy

- **Phase 1-4**: Zero new tests needed — all existing tests validate behavior preservation
- **Phase 5**: New integration tests for HTTP endpoints; can use `ctest` + `curl` or a lightweight HTTP client in C++
- `AppLifecycle` and individual modes become independently unit-testable by mocking `AppContext` members
