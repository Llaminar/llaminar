> STATUS NOTE (2025-10): The Abstract Pipeline system described here has been implemented and the legacy `MPITransformerPipeline` is now formally DEPRECATED. References below to “keeping the existing MPITransformerPipeline” reflect the historical state at the time this plan was authored. The generic compute graph has been removed from the codebase. This document is retained as a historical design record and for any still‑pending future phases (multi‑architecture expansion, execution tape). Please do not add new code paths to `MPITransformerPipeline`; implement new behavior via concrete `AbstractPipeline` adapters instead.

We can absolutely introduce a Pipeline base abstraction to support multiple model architectures (Qwen, LLaMA, Mistral, Mixtral/MoE, GPT-J style, etc.) without destabilizing the existing (now legacy) `MPITransformerPipeline`. Below is the pragmatic, incremental design that preserved behavior while opening the door for future architectures.

## Goals
1. Allow multiple pipeline implementations (different block structure, attention variants, norms, MoE).
2. Keep existing Qwen path working with minimal churn.
3. Avoid premature over-generalization (don’t abstract every weight now).
4. Centralize cross‑cutting services: tensor factory, backend selection, MPI context, debugEnv snapshot.
5. Enable partial reuse: shared kernels where shapes align; override where architecture diverges.
6. Support evolution toward a more dynamic execution graph later (optionally reviving or replacing the dormant compute graph).

## High-Level Layering

```
+--------------------------------------------------+
| PipelineFactory / Registration                   |
+----------------------+---------------------------+
| AbstractPipelineBase | (pure virtual interface)  |
+----------------------+---------------------------+
|  Concrete Pipelines:                             
|   - QwenPipeline   (wraps current MPITransformerPipeline logic)
|   - LlamaPipeline
|   - MixtralPipeline (MoE gating + experts)
|   - (Future architectures)                       |
+--------------------------------------------------+
| Shared Services: BackendSelector, TensorFactory, |
| DebugEnv snapshot, MPIContext, KVCacheManager    |
+--------------------------------------------------+
| Kernels: Attention, RMSNorm, MLP, MatMul, Rope   |
+--------------------------------------------------+
```

## Core Abstractions

### 1. Pipeline Config Unification

Create a superset `ModelConfig` (replaces or wraps `LayerConfig`):

```cpp
struct ModelConfig {
    std::string architecture;     // "qwen2.5", "llama3", "mixtral", etc.
    int n_layers;
    int d_model;
    int n_head;
    int head_dim;
    int kv_head_dim;              // optional override
    int vocab_size;
    int d_ff;                     // optional / infer
    bool uses_gqa = false;
    bool uses_moe = false;
    int n_experts = 0;
    int experts_per_token = 0;
    // Extensible capability flags
    struct {
        bool rope = true;
        bool rmsnorm = true;
        bool alibi = false;
        bool grouped_query_attention = false;
        bool sliding_window = false;
    } features;
};
```

### 2. Model Weights Interface

Polymorphic access without requiring every architecture to expose identical internals:

```cpp
class IModelWeights {
public:
    virtual ~IModelWeights() = default;
    virtual const std::shared_ptr<TensorBase>& token_embedding() const = 0;
    virtual int layers() const = 0;
    virtual std::string arch() const = 0;
    // Optional typed queries (return nullptr if unsupported)
    virtual const std::shared_ptr<TensorBase>& lm_head() const = 0;
    // Provide extension hook for architecture-specific queries:
    virtual std::shared_ptr<TensorBase> get(const std::string& key) const = 0;
};
```

Your existing `ModelWeights` becomes `QwenModelWeights : public IModelWeights`.

### 3. Pipeline Base

```cpp
class AbstractPipeline {
public:
    virtual ~AbstractPipeline() = default;
    virtual const ModelConfig& config() const = 0;

    // Core inference entry-point (prefill+decode could be split later)
    virtual bool execute(
        const std::vector<int>& input_tokens,
        const IModelWeights& weights,
        std::shared_ptr<TensorBase>& output_logits) = 0;

    // Optional streaming decode step (for future token-by-token API)
    virtual bool decode_next(
        int next_token,
        const IModelWeights& weights,
        std::shared_ptr<TensorBase>& output_logits)
    { (void)next_token; (void)weights; (void)output_logits; return false; }

    // Hook for architecture-specific KV cache (allocate / resize)
    virtual void ensure_kv_capacity(size_t seq_len) = 0;

    // Introspection
    virtual std::string name() const = 0;

    // Optional: architecture-specific weight loader (kept separate first for minimal risk)
    virtual std::unique_ptr<IModelWeights> loadWeights(const std::string& path) = 0;
};
```

### 4. Pipeline Factory

Simple registry avoids switch statements everywhere:

```cpp
class PipelineFactory {
public:
    using Creator = std::function<std::unique_ptr<AbstractPipeline>(const ModelConfig&)>;
    static PipelineFactory& instance();
    void register_creator(const std::string& arch, Creator c);
    std::unique_ptr<AbstractPipeline> create(const ModelConfig& cfg);

private:
    std::unordered_map<std::string, Creator> map_;
};
```

Registration macro:

```cpp
#define REGISTER_PIPELINE(ARCH, TYPE) \
    static bool _reg_##TYPE = [](){ \
        PipelineFactory::instance().register_creator(ARCH, \
            [](const ModelConfig& c){ return std::make_unique<TYPE>(c); }); \
        return true; }();
```

### 5. Adapting Current Implementation

Current `MPITransformerPipeline` can become `QwenPipeline : public AbstractPipeline`:

Minimal steps:
1. Rename or wrap: keep existing file, add inheritance.
2. Replace the public `execute()` signature with override (internal logic unchanged).
3. Move `ModelWeights loadModelWeights(...)` into `QwenPipeline::loadWeights`.
4. Deprecate the free function loaders (they can now delegate to the pipeline’s method).
5. Introduce thin `QwenModelWeights` implementing `IModelWeights`.

Incremental trick: Keep old struct `ModelWeights` untouched, and create an adapter:

```cpp
class QwenModelWeightsAdapter : public IModelWeights {
    MPITransformerPipeline::ModelWeights inner_;
public:
    // implement interface by forwarding
};
```

This lets you defer refactoring all weight member naming immediately.

### 6. Backend & MPI Context Decoupling

Refactor MPI utilities into a small `MPIContext`:

```cpp
struct MPIContext {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    // future: local_rank, node_id, numa_node
    static MPIContext capture(MPI_Comm c = MPI_COMM_WORLD) {
        MPIContext ctx; MPI_Comm_rank(c,&ctx.rank); MPI_Comm_size(c,&ctx.size); ctx.comm=c; return ctx;
    }
};
```

Pass it into pipeline constructor or capture internally at construction to avoid repeated calls.

### 7. Handling Architectural Divergence

| Feature | Variation Strategy |
|---------|--------------------|
| Attention (GQA, MQA, standard) | Virtual method `prepare_attention()` or a small strategy object |
| MoE gating & experts | Add optional `execute_ffn(layer, state)` override in derived pipeline |
| Positional encoding (RoPE vs ALiBi) | Capability flags + virtual `apply_position_encoding()` |
| Norm type differences (RMSNorm vs LayerNorm) | Kernel dispatch keyed off config flag |
| Weight tying / separate head | Derived pipeline chooses `lm_head()` source |
| Tensor parallel vs sharded variant | Choose appropriate creation calls in weight loader |

### 8. Execution Flow Hooks

Break up the existing monolith:

```cpp
struct LayerState {
    // Pointers into weights / caches
};

virtual bool run_layer(
    int layer_index,
    LayerState& state,
    const IModelWeights& weights,
    std::shared_ptr<TensorBase>& residual);
```

Derived pipelines override just the parts that differ (e.g., Mixtral splits FFN into gating + expert selection).

### 9. (Historical) Execution Tape Concept

The legacy generic compute graph has since been removed. A future lightweight “execution tape” (vector of semantic ops) remains an optional enhancement for advanced scheduling or replay, but is not currently prioritized. This section is preserved only to capture the earlier rationale.

### 10. Migration Phases

| Phase | Scope | Risk |
|-------|-------|------|
| 0 | Add `ModelConfig`, keep using old `LayerConfig` (wrap) | None |
| 1 | Add `AbstractPipeline`, adapter for current pipeline | Low |
| 2 | Introduce factory + register Qwen pipeline | Low |
| 3 | Move weight loading behind virtual interface | Moderate (log parity) |
| 4 | Spin up prototype `LlamaPipeline` (even if stubs) | Moderate |
| 5 | Extract backend selector & MPIContext | Low |
| 6 | (Optional) Delete now-dead free loader functions | Low |
| 7 | Introduce execution tape / remove old compute graph | Moderate |

Each phase should pass: build + existing tests + log substring checks (where sensitive).

### 11. Testing Strategy

- Add new tests: `test_pipeline_factory.cpp`, `test_qwen_adapter.cpp`.
- Parity test: old vs new pipeline produce identical logits for a fixed seed & small prompt.
- Negative test: request unknown architecture → factory throws (assert error path).
- Future: model-specific tests (LLaMA rotary dims, Mixtral gating correctness).

### 12. Potential Pitfalls

| Pitfall | Mitigation |
|---------|------------|
| Log differences break existing tests | Keep adapter logs identical; only swap implementation behind interface. |
| Weight memory duplication | Use shared_ptr inside adapter; avoid deep copies. |
| Inference latency regression | No additional virtual calls in inner loops (invoke high-level virtual once per layer). |
| Explosion of architecture-specific if/else | Push logic into derived classes early. |
| Over-abstraction (YAGNI) | Start with minimal pure virtual set (execute, loadWeights, config, ensure_kv_capacity). |

### 13. Example Skeleton (Concise)

```cpp
// pipeline_base.h
class AbstractPipeline {
public:
    virtual ~AbstractPipeline() = default;
    virtual const ModelConfig& config() const = 0;
    virtual std::unique_ptr<IModelWeights> loadWeights(const std::string& path) = 0;
    virtual bool execute(const std::vector<int>& tokens,
                         const IModelWeights& weights,
                         std::shared_ptr<TensorBase>& logits) = 0;
    virtual void ensure_kv_capacity(size_t seq_len) = 0;
    virtual std::string name() const = 0;
};

// qwen_pipeline.h
class QwenPipeline : public AbstractPipeline {
public:
    explicit QwenPipeline(ModelConfig cfg);
    const ModelConfig& config() const override { return cfg_; }
    std::unique_ptr<IModelWeights> loadWeights(const std::string& path) override;
    bool execute(const std::vector<int>& tokens,
                 const IModelWeights& weights,
                 std::shared_ptr<TensorBase>& logits) override;
    void ensure_kv_capacity(size_t seq_len) override;
    std::string name() const override { return \"QwenPipeline\"; }
private:
    ModelConfig cfg_;
    // reuse existing internal members
};
```

### 14. Incremental Code Touch Zones

- Add new headers: `pipeline_base.h`, `pipeline_factory.h`.
- Modify main.cpp: replace direct `createMPITransformerPipeline` with factory create.
- Wrap existing `loadModelWeights` paths (already unified) inside `QwenPipeline::loadWeights`.
- Keep old APIs temporarily with DEPRECATED log shim.

### 15. When to Introduce MoE / GQA Differences?

Wait until after base abstraction stabilized (Phase 4). Do not anticipate all gating logic now—just ensure the base interface does not preclude adding per-layer dynamic ops.

---

## Recommended Immediate First Step (Low-Risk)

1. Add `ModelConfig` (thin adapter of current `LayerConfig`).
2. Introduce `AbstractPipeline` + `PipelineFactory` with only Qwen registered.
3. Wrap existing pipeline inside `QwenPipeline` with an adapter for weights (no functional changes).
4. Switch main.cpp to factory creation if `config.architecture` is set; else fallback to old path for safety (temporary dual path).
5. Add a parity test ensuring identical first token logits old vs new pipeline.

Below is a unified architectural view of how “prefill” is handled today, why it’s fragmented, and how to fold it cleanly into the proposed `AbstractPipeline` without redundancy.

## 1. What “Prefill” Means Here
Prefill = the initial large sequence (prompt) pass where:
- Sequence length is long (threshold-driven) so Q/K/V and FFN matmuls are large enough to justify different backends (COSMA / distributed).
- We populate KV cache for future decode steps.
- We may choose different matmul backends (COSMA vs OpenBLAS) to exploit better scaling.

Decode (a.k.a. “inference” or “generation” step) = short sequence extension (usually 1 token at a time) favoring low latency, small matmuls, local OpenBLAS.

## 2. Where Prefill Logic Lives Today

| Layer | Current Responsibility | Location(s) |
|-------|------------------------|-------------|
| Pipeline orchestration | Calls layer loops, decides to use COSMA path in attention | `MPITransformerPipeline::execute()` (monolithic), `shouldUseCosmaPrefill`, `executePrefillAttentionCosma` |
| Backend decision large matmuls | Adaptive heuristics (size & is_prefill flag) | `adaptive_matmul` + quantitative threshold via `debugEnv().cosma.prefill_threshold` |
| COSMA prefill gating | Sequence length threshold + env overrides | `shouldUseCosmaPrefill()` |
| Prefill attention fused path | Specialized function populating Q/K/V & output with COSMA | `executePrefillAttentionCosma()` (heavy branching) |
| Prefill diag & baselines | Multiple `[Prefill*]` logs + `PrefillBaselineRegistry` snapshots | Large spans in mpi_transformer_pipeline.cpp |
| Kernel-level prefill aware paths | Separate “prefill-like” branch using `PrefillBackendFactory` | MPIAttentionKernel.cpp, MPIMLPKernel.cpp |
| Prefill matmul backend selection | Separate call into `PrefillBackendInterface` then fallback to `adaptive_matmul` | `prefill_backend.{h,cpp}` |
| COSMA manager for large GEMMs | Weight staging, tile orchestration | `cosma_prefill_manager.{h,cpp}` |
| Performance counters (prefill stats) | Aggregating time/flops by backend with is_prefill flag | perf_counters.h |

This set is cohesive in intention but scattered across:
- Pipeline class
- Independent backend layer
- Per-kernel special cases
- Global managers + debug instrumentation
- Adaptive matmul fallback

## 3. Duplication / Fragmentation Issues
1. Two parallel notions of “prefill”:
   - Pipeline-level COSMA enabling (`shouldUseCosmaPrefill`).
   - Kernel-level `is_prefill_like = seq_len >= threshold` repeated in Attention + MLP kernels.

2. Two “abstraction” layers:
   - `PrefillBackend` interface (currently CPU stub, acts as an alternate adapter).
   - Proposed `AbstractPipeline` would re-encapsulate stage awareness—risk of redundancy if we keep both unchanged.

3. Instrumentation entangled with logic:
   - Prefill diagnostics are embedded in the execution loops (difficult to reuse or toggle).
   - Baseline comparison logic couples computation and logging.

4. Fallback and decision stacking:
   - Path: PrefillBackend → (maybe) adaptive_matmul → inside which we again branch on `is_prefill`.
   - This leads to repeated size checks.

## 4. Architectural Vision: Unify Under AbstractPipeline

Introduce three explicit stages in the pipeline interface:

```cpp
enum class InferenceStage { Prefill, Decode };

struct StageContext {
    InferenceStage stage;
    int seq_len;        // current effective sequence length
    int new_tokens;     // tokens being added this call (prefill: seq_len, decode: 1)
    bool distributed_allowed;
};
```

### AbstractPipeline Additions

```cpp
class AbstractPipeline {
public:
    virtual bool prefill(const std::vector<int>& prompt_tokens,
                         const IModelWeights& weights,
                         StageContext& ctx,
                         std::shared_ptr<TensorBase>& logits) = 0;

    virtual bool decode(const std::vector<int>& next_tokens,   // usually size 1
                        const IModelWeights& weights,
                        StageContext& ctx,
                        std::shared_ptr<TensorBase>& logits) = 0;

    // Convenience unified execute that dispatches:
    bool execute(const std::vector<int>& tokens,
                 const IModelWeights& weights,
                 std::shared_ptr<TensorBase>& logits) {
        StageContext ctx;
        // detect stage
        if (kv_cache_empty()) {
            ctx.stage = InferenceStage::Prefill;
            ctx.seq_len = (int)tokens.size();
            ctx.new_tokens = ctx.seq_len;
            return prefill(tokens, weights, ctx, logits);
        } else {
            ctx.stage = InferenceStage::Decode;
            ctx.seq_len = existing_seq_len() + (int)tokens.size();
            ctx.new_tokens = (int)tokens.size();
            return decode(tokens, weights, ctx, logits);
        }
    }
};
```

### Backend Selection Consolidation

Replace scattered logic with one function:

```cpp
MatMulBackend select_backend(int m, int n, int k, const StageContext& ctx);
```

Implementation consumes:
- `ctx.stage`
- volume thresholds
- environment flags (COSMA force/disable)
- memory residency guard
- (future) model architecture hints (MoE, grouped heads)

### COSMA Manager Integration

`CosmaPrefillManager` becomes a strategy used only when:
- `ctx.stage == Prefill`
- `select_backend` chooses a COSMA category
- Required shapes (Q/K/V fused) are above threshold

Expose a neutral handle:

```cpp
struct LargeMatmulPlan {
    bool use_cosma;
    bool fused_qkv;
    // scratch sizes, memory reservations, etc.
};
LargeMatmulPlan plan_attention_prefill(int seq_len, int d_model, int n_heads);
```

Pipeline calls plan, then executes via manager, isolating this from kernel code.

### Kernel Simplification

Current kernels embed prefill-vs-decode branching. Instead:
- Kernels trust the pipeline to hand them already-partitioned local slices and chosen backend.
- Remove local `is_prefill_like` checks.
- Surface a parameter: `ExecutionMode mode` (Prefill vs Decode) only if algorithmic difference (e.g., attention window or rope offset handling), not for matmul backend choice.

### PrefillBackend De-Duplication

Retire or repurpose `PrefillBackendInterface`:

Option A (Immediate): Remove interface; pipeline directly calls `adaptive_matmul` or COSMA plan based on centralized `select_backend`.

Option B (Later extension): Rename to `StageBackendInterface` to allow plugging GPU specialization per stage. For now, Option A keeps code simpler.

## 5. Migration Path (Incremental, Low Risk)

| Step | Change | Notes |
|------|--------|-------|
| 1 | Introduce `InferenceStage`, `StageContext` enums/structs | No existing code removal |
| 2 | Add central `select_backend(.., StageContext)` and switch `adaptive_matmul` calls to pass stage | Keep old signature shim temporarily |
| 3 | Modify `MPITransformerPipeline` to split `execute()` into `prefill()` + `decode()` small wrappers calling existing internal functions | Most logic stays, just relocated |
| 4 | Remove kernel-local `is_prefill_like` if condition duplicates (replace with parameter passed from pipeline) | Attention + MLP |
| 5 | Collapse `PrefillBackendFactory` usage: pipeline chooses, kernels just do the operation | Remove factory code after pipeline path stable |
| 6 | Integrate COSMA prefill path behind a single function `execute_attention_prefill(...)` called only in `prefill()` | Reuse existing cosma manager code |
| 7 | Extract prefill diagnostics from main compute loops into `log_prefill_snapshot(Stage, Tag, TensorView)` utilities | Reduces noise in core path |
| 8 | Add new pipeline abstraction (from earlier discussion) so new architectures implement both prefill/decode symmetrically | Avoids repeating adjustments later |
| 9 | Cleanup: delete `PrefillBackendInterface` if fully superseded | After tests stable |

## 6. Unifying Diagnostics

Create a dedicated module `prefill_diagnostics.h/.cpp`:

```cpp
struct PrefillDiagConfig {
    bool capture_baseline;
    bool compare_baseline;
    bool row_stats;
    bool value_scan;
};

void prefill_log_tensor(const char* tag,
                        int rank,
                        const float* data,
                        int rows, int cols,
                        const PrefillDiagConfig& cfg,
                        BaselineStore* store);
```

Replace scattered inline loops with calls, decreasing code footprint in pipeline by hundreds of lines.

## 7. Handling Sequence Continuity

Current code implicitly assumes:
- Prefill populates full KV cache
- Decode appends one token

Abstract explicitly:

```cpp
struct KVCacheState {
    size_t allocated_tokens;
    size_t used_tokens;
    // shape metadata for heads, head_dim, etc.
};
```

`prefill()` sets `used_tokens = prompt_len`. `decode()` increments by 1 per step. Future multi-token decode batching trivially fits.

## 8. Where AbstractPipeline Adds Value (Non-Redundant)

The existing Prefill/Inference split is:
- Tactical, embedded, and duplicated.
- Lacks a contract interface for new architectures.

`AbstractPipeline` gives:
- Formal stage gating
- Centralizes backend/mode decisions
- Removes branching from each kernel
- Surfaces prefill vs decode as a first-class design dimension (same for MoE routing later)

Thus it does NOT duplicate; it replaces an implicit pattern with an explicit, enforceable one.

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing log-based tests | Keep original log substrings; wrap new diag functions to emit identical tags |
| Performance regression (extra virtual calls) | Only stage dispatch is virtual; inner loops unchanged |
| COSMA path subtle bugs during relocation | Move code verbatim first (mechanical), refactor after passing tests |
| Over-abstraction early | Limit initial interface to: `prefill()`, `decode()`, `execute()`, `select_backend()`; postpone more until a second architecture lands |

## 10. Concrete First Patch (What I Can Do Next If Approved)
1. Add `inference_stage.h` with `InferenceStage` + `StageContext`.
2. Introduce `select_backend` central helper (migrate logic from scattered calls).
3. Refactor `MPITransformerPipeline::execute` to:
   - Detect stage
   - Call `prefill_impl` (existing body moved) or `decode_impl` (currently a reduced path for token extension; create placeholder if decode path isn’t separate yet).
4. Replace local `is_prefill_like` calculations by handing down a boolean parameter derived from `StageContext`.
5. Ensure build + run minimal tests (parity of logs).

(Full AbstractPipeline & factory layering could be second patch to keep changeset reviewable.)

---

## Summary Answer to Your Direct Questions

1. How do we handle prefill right now?
   - Through ad hoc sequence length threshold checks scattered across pipeline and kernels, plus a distinct COSMA prefill manager path for large attention/FFN operations. Prefill detection = `seq_len >= env_threshold`. Diagnostics and baselines heavily interleaved in the main pipeline source.

2. Does the MPITransformerPipeline do this?
   - Yes. It orchestrates prefill vs decode implicitly inside `execute()` and calls specialized COSMA routines when `shouldUseCosmaPrefill(seq_len)` returns true.

3. Are existing Prefill / Inference abstractions redundant with proposed `AbstractPipelineBase`?
   - They will become redundant. The current “abstraction” is informal. A formal `AbstractPipeline` with explicit `prefill()` and `decode()` methods subsumes the existing patterns and removes duplication.

4. Need for unified architectural vision?
   - Yes: central stage context, single backend decision layer, pipeline-level stage methods, extraction of diagnostics, and eventual pluggable architectures.

---