# Qwen3-Next Gated DeltaNet (GDN) Project Plan

**Date**: 2025-03-05  
**Status**: Research Complete — Implementation Planned  
**Motivation**: Qwen3-Next (September 2025) introduces **Gated DeltaNet (GDN)** as a hybrid attention replacement. GDN layers use a linear recurrence instead of softmax attention, enabling O(n·d_k·d_v) complexity and fixed-size recurrent state instead of a growing KV cache. Supporting this architecture is required for Llaminar to run Qwen3-Next models (both dense and MoE variants).

---

## Table of Contents

- [Background](#background)
- [Current State Analysis](#current-state-analysis)
- [Architecture Overview: Gated DeltaNet](#architecture-overview-gated-deltanet)
- [Hybrid Architecture: Per-Layer Dispatch](#hybrid-architecture-per-layer-dispatch)
- [Phase 1: Python Reference and Parity Infrastructure](#phase-1-python-reference-and-parity-infrastructure)
- [Phase 2: GGUF Weight Loading](#phase-2-gguf-weight-loading)
- [Phase 3: New Tensor Types and Kernel Interfaces](#phase-3-new-tensor-types-and-kernel-interfaces)
- [Phase 4: Graph Schema and Stages](#phase-4-graph-schema-and-stages)
- [Phase 5: CPU Reference Kernels](#phase-5-cpu-reference-kernels)
- [Phase 6: Optimized Kernels (CPU JIT, CUDA, ROCm)](#phase-6-optimized-kernels-cpu-jit-cuda-rocm)
- [Phase 7: MoE Integration](#phase-7-moe-integration)
- [Phase 8: End-to-End Validation](#phase-8-end-to-end-validation)
- [Risk Assessment](#risk-assessment)
- [External References](#external-references)

---

## Background

### Naming Clarification

The model commonly referred to as "Qwen 3.5" is officially **Qwen3-Next** (September 2025). Key facts:

| Model Family | Release | Attention Type | Status in Llaminar |
|-------------|---------|----------------|-------------------|
| Qwen2.5 | Late 2024 | Standard GQA | ✅ Fully supported |
| Qwen3 | April–July 2025 | Standard GQA (+ thinking mode) | ✅ Architecturally identical to Qwen2.5 |
| **Qwen3-Next** | **September 2025** | **Hybrid GDN + sliding-window softmax** | **❌ Not supported — this project** |

### What is Gated DeltaNet?

Gated DeltaNet (GDN) was introduced in the ICLR 2025 paper *"Gated Delta Networks: Improving Mamba2 with Delta Rule"* (arXiv: 2412.06464). It replaces softmax attention with a **linear recurrence**:

**Core recurrence**:
```
S_t = diag(exp(g_t)) · S_{t-1} + β_t · k_t · (v_t − S_{t-1}^T k_t)^T
o_t = S_t^T · q_t
```

Where:
- `S_t` is a fixed-size state matrix (d_k × d_v), replacing the growing KV cache
- `g_t` is a gating vector (controls state decay per dimension)
- `β_t` is a scalar gate (controls write strength)
- The delta rule term `(v_t − S_{t-1}^T k_t)` provides error-corrective updates

**Key Properties**:
- **Linear complexity**: O(n · d_k · d_v) versus O(n² · d_k) for softmax attention
- **Fixed memory**: State size is constant regardless of sequence length
- **No KV cache**: GDN layers maintain a recurrent state, not key/value caches
- **Chunk-parallel training/prefill**: Can process sequences in chunks with parallel scan

### Hybrid Architecture

Qwen3-Next is a **hybrid** model: some layers use GDN, some use sliding-window softmax attention. This is configured per-layer in the model config (e.g., `attn_type: [0, 0, 1, 0, 1, ...]` where 0=GDN, 1=softmax).

---

## Current State Analysis

### Existing Qwen2 Support

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Graph Schema | `src/v2/models/qwen/Qwen2Schema.h` | ~407 | Declarative stage/buffer/weight spec |
| Graph Builder | `src/v2/models/qwen/Qwen2Graph.h` | ~941 | Materializes schema into ComputeGraph |
| Graph Config | `src/v2/models/qwen/Qwen2Graph.h` | struct | `Qwen2GraphConfig` with all model dimensions |
| Attention Kernels | `kernels/cpu/attention/` | ~12K+ | FlashAttention (JIT AVX-512), tiled, reference |
| CUDA Attention | `kernels/cuda/attention/` | ~720 | FA2 prefill + Flash Decoding |
| ROCm Attention | `kernels/rocm/attention/` | ~762 | AMD MI50 targeting |
| Python Reference | `python/reference/qwen.py` | ~369 | HF monkey-patched snapshot hooks |
| Attention Interface | `tensors/TensorKernels.h` | — | `ITensorAttention`, `ITensorFusedAttentionWo` |

### What Must Change

GDN is **not attention** — it is a fundamentally different computation. The existing attention kernel interfaces (`ITensorAttention`, `ITensorFusedAttentionWo`) cannot be extended to cover GDN. A new parallel interface is required.

| Existing Component | GDN Impact |
|-------------------|-----------|
| `ITensorAttention` | **New interface needed** — `ITensorGatedDeltaNet` |
| KV cache (`RingKVCache`) | **Not used by GDN layers** — GDN uses fixed-size state matrix |
| RoPE stage | **Still used** for both GDN and softmax layers |
| QKV projection stage | **Modified** — GDN needs additional projections (a, b, g) |
| FlashAttention kernels | **Unchanged** — still used for softmax layers in hybrid model |
| `Qwen2Schema` | **New schema** — `Qwen3NextSchema` with per-layer type dispatch |

---

## Architecture Overview: Gated DeltaNet

### GDN Layer Computation Flow

```
Input (d_model)
    │
    ├─→ q_proj ──→ ShortConv1d(k=4) ──→ SiLU ──→ Q (n_heads × d_k)
    ├─→ k_proj ──→ ShortConv1d(k=4) ──→ SiLU ──→ K (n_kv_heads × d_k)
    ├─→ v_proj ──→ ShortConv1d(k=4) ──→ SiLU ──→ V (n_kv_heads × d_v)  ← d_v = expand_v × d_k
    ├─→ a_proj ──────────────────────────────────→ A (n_heads,)         ← beta gate (sigmoid)
    ├─→ b_proj ──────────────────────────────────→ B (n_heads,)         ← output gate
    └─→ g_proj ──────────────────────────────────→ G (n_heads × d_k)   ← state decay
    │
    │   A_log parameter → exp(−softplus(A_log + A)) → β (write gate)
    │   dt_bias parameter → clamp → decay computation
    │
    ├─→ RoPE(Q, K)  ← rotary embeddings still applied
    │
    ├─→ GDN Recurrence:
    │     S_t = diag(exp(g_t)) · S_{t-1} + β_t · k_t · (v_t − S_{t-1}^T k_t)^T
    │     o_t = S_t^T · q_t
    │
    ├─→ GatedRMSNorm(o) ──→ Wo projection
    │
    └─→ Residual add
```

### New Projections (vs Standard Attention)

| Projection | Shape | Purpose | Analogous to |
|-----------|-------|---------|-------------|
| `q_proj` | (d_model, n_heads × d_k) | Query vectors | Same as GQA |
| `k_proj` | (d_model, n_kv_heads × d_k) | Key vectors | Same as GQA |
| `v_proj` | (d_model, n_kv_heads × d_v) | Value vectors | Same shape class, but d_v ≠ d_k |
| **`a_proj`** | (d_model, n_heads) | Beta gate input | **New** |
| **`b_proj`** | (d_model, n_heads) | Output gate input | **New** |
| **`g_proj`** | (d_model, n_heads × d_k) | State decay gate | **New** |
| **`A_log`** | (n_heads,) | Learnable log-space gate parameter | **New** (not a projection — a parameter) |
| **`dt_bias`** | (n_heads,) | Decay time-step bias | **New** (not a projection — a parameter) |

### Short Convolution (Conv1d)

GDN applies a 1D causal convolution (kernel_size=4) to Q, K, V **before** the recurrence:

| Parameter | Shape | Notes |
|-----------|-------|-------|
| `q_conv_weight` | (n_heads × d_k, 1, 4) | Depthwise Conv1d |
| `k_conv_weight` | (n_kv_heads × d_k, 1, 4) | Depthwise Conv1d |
| `v_conv_weight` | (n_kv_heads × d_v, 1, 4) | Depthwise Conv1d |
| `q_conv_bias` | (n_heads × d_k,) | Optional |
| `k_conv_bias` | (n_kv_heads × d_k,) | Optional |
| `v_conv_bias` | (n_kv_heads × d_v,) | Optional |

During decode, the convolution requires maintaining a small sliding-window state buffer of the last 3 (kernel_size − 1) values.

### Value Expansion

GDN typically uses `expand_v = 2.0`, meaning:
- `d_v = expand_v × d_k` (e.g., d_k=64 → d_v=128)
- V tensors and the state matrix S are wider than in standard attention
- `Wo` projection shape changes: (n_heads × d_v, d_model) instead of (n_heads × d_k, d_model)

### Recurrent State (replaces KV cache)

| Property | KV Cache (GQA) | GDN State |
|----------|---------------|-----------|
| Size per layer | 2 × seq_len × n_kv_heads × d_k | n_heads × d_k × d_v |
| Growth | Linear in sequence length | **Fixed** |
| Memory for 128K context, 32 heads, d_k=128 | ~4 GB per layer | ~2 MB per layer |
| Update cost | Append | Matrix multiply + element-wise ops |

---

## Hybrid Architecture: Per-Layer Dispatch

The Qwen3-Next model config specifies which layers use GDN vs softmax:

```json
{
  "attn_type": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, ...],
  "sliding_window": 4096
}
```

- `0` = GDN layer (linear recurrence, no KV cache)
- `1` = Sliding-window softmax attention layer (standard FlashAttention with window)

This means the **graph schema must support per-layer stage selection**. Some layers instantiate GDN stages; others instantiate the existing softmax attention stages.

---

## Phase 1: Python Reference and Parity Infrastructure

**Goal**: Get a PyTorch reference implementation producing snapshot-compatible outputs for parity testing.

### 1.1 Add fla Library Dependency

The [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library (`fla`) contains complete PyTorch implementations of GDN. Add it as a dependency.

**File**: `requirements.txt` (add `flash-linear-attention`)

### 1.2 Create Python Reference Model

**File**: `python/reference/qwen3next.py`

Follow the pattern of `python/reference/qwen.py`:
- Load the Qwen3-Next GGUF model weights
- Monkey-patch the HuggingFace `GatedDeltaNetForCausalLM` from fla to capture per-layer snapshots
- Expose hooks for: post-embedding, per-layer (pre-norm, post-QKV-proj, post-conv, post-recurrence, post-Wo, post-FFN), final-norm, logits
- Handle hybrid layers: softmax layers use the existing `qwen.py` hooks; GDN layers use new hooks

### 1.3 Snapshot Stages for GDN

**File**: `python/reference/pipeline_stages.py`

Add new `PipelineStage` enum entries:
```python
# GDN-specific stages
GDN_SHORT_CONV_Q = auto()
GDN_SHORT_CONV_K = auto()
GDN_SHORT_CONV_V = auto()
GDN_GATE_COMPUTE = auto()      # a_proj → β, b_proj → output gate, g_proj → decay
GDN_RECURRENCE = auto()        # Core S_t update + output
GDN_GATED_RMSNORM = auto()     # Gated normalization before Wo
GDN_WO_PROJ = auto()           # Output projection
```

### 1.4 Generate Reference Snapshots

**File**: `python/reference/generate_gdn_snapshots.py`

Script to generate layer-by-layer snapshots for a Qwen3-Next model, writing `.npy` files in the format expected by the C++ parity test infrastructure.

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| Python reference | `python/reference/qwen3next.py` | HF-compatible reference with snapshot hooks |
| Pipeline stages | `python/reference/pipeline_stages.py` | New GDN stage enums |
| Snapshot generator | `python/reference/generate_gdn_snapshots.py` | Reference snapshot generation |
| Requirements update | `requirements.txt` | `flash-linear-attention` dependency |

---

## Phase 2: GGUF Weight Loading

**Goal**: Extend the GGUF loader and `WeightManager` to recognize and load Qwen3-Next weight tensors.

### 2.1 New Weight Tensor Names

GDN layers introduce new weight tensors in GGUF format. Expected GGUF names (following llama.cpp conventions):

| GGUF Tensor Name | Shape | Per-Layer | Purpose |
|-----------------|-------|-----------|---------|
| `blk.{i}.attn_a.weight` | (d_model, n_heads) | Yes | Beta gate projection |
| `blk.{i}.attn_b.weight` | (d_model, n_heads) | Yes | Output gate projection |
| `blk.{i}.attn_g.weight` | (d_model, n_heads × d_k) | Yes | State decay projection |
| `blk.{i}.attn_A_log` | (n_heads,) | Yes | Learnable log gate parameter |
| `blk.{i}.attn_dt_bias` | (n_heads,) | Yes | Decay time-step bias |
| `blk.{i}.attn_q_conv.weight` | (n_heads × d_k, 1, 4) | Yes | Q short convolution |
| `blk.{i}.attn_k_conv.weight` | (n_kv_heads × d_k, 1, 4) | Yes | K short convolution |
| `blk.{i}.attn_v_conv.weight` | (n_kv_heads × d_v, 1, 4) | Yes | V short convolution |
| `blk.{i}.attn_q_conv.bias` | (n_heads × d_k,) | Yes | Q conv bias (optional) |
| `blk.{i}.attn_k_conv.bias` | (n_kv_heads × d_k,) | Yes | K conv bias (optional) |
| `blk.{i}.attn_v_conv.bias` | (n_kv_heads × d_v,) | Yes | V conv bias (optional) |
| `blk.{i}.attn_gated_rmsnorm.weight` | (n_heads × d_v,) | Yes | Gated RMSNorm scale |

> **Note**: Exact GGUF tensor names depend on the llama.cpp conversion tool. These names may need adjustment once official GGUF conversion for Qwen3-Next is available. The loader should be flexible.

### 2.2 Model Config Metadata

The GGUF file must contain:
- `qwen3next.attn_type` — array of per-layer attention type (0=GDN, 1=softmax)
- `qwen3next.expand_v` — value expansion factor (default 2.0)
- `qwen3next.conv_kernel_size` — short convolution kernel size (default 4)
- `qwen3next.sliding_window` — sliding window size for softmax layers

### 2.3 WeightManager Extension

**Files to modify**:
- `src/v2/loaders/WeightManager.h/cpp` — Register new tensor name patterns
- `src/v2/loaders/GGUFLoader.h/cpp` — Parse new metadata fields

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| Weight name registry | `WeightManager.h/cpp` | New tensor name patterns for GDN layers |
| GGUF metadata parsing | `GGUFLoader.h/cpp` | `attn_type` array, `expand_v`, `conv_kernel_size` |
| Unit tests | `tests/v2/unit/Test__Qwen3NextWeightLoading.cpp` | Verify weight discovery and sharding |

---

## Phase 3: New Tensor Types and Kernel Interfaces

**Goal**: Define the kernel interfaces and any new tensor types needed for GDN computation.

### 3.1 ITensorGatedDeltaNet Interface

**File**: `src/v2/tensors/TensorKernels.h` (extend existing)

```cpp
/// Kernel interface for Gated DeltaNet linear recurrence.
class ITensorGatedDeltaNet {
public:
    virtual ~ITensorGatedDeltaNet() = default;

    /// Prefill: process a full sequence in chunk-parallel mode.
    /// Q: (seq_len, n_heads, d_k)
    /// K: (seq_len, n_kv_heads, d_k)
    /// V: (seq_len, n_kv_heads, d_v)
    /// beta: (seq_len, n_heads)        — write gate
    /// decay: (seq_len, n_heads, d_k)  — state decay (exp(g))
    /// state_in: (n_heads, d_k, d_v)   — prior state (nullptr for first call)
    /// output: (seq_len, n_heads, d_v)
    /// state_out: (n_heads, d_k, d_v)  — updated state
    virtual bool compute_prefill(
        const float* Q, const float* K, const float* V,
        const float* beta, const float* decay,
        const float* state_in, float* output, float* state_out,
        int seq_len, int n_heads, int n_kv_heads,
        int d_k, int d_v, int chunk_size) = 0;

    /// Decode: process a single new token, updating state in-place.
    /// Q: (1, n_heads, d_k)
    /// K: (1, n_kv_heads, d_k)
    /// V: (1, n_kv_heads, d_v)
    /// beta: (1, n_heads)
    /// decay: (1, n_heads, d_k)
    /// state: (n_heads, d_k, d_v) — updated in-place
    /// output: (1, n_heads, d_v)
    virtual bool compute_decode(
        const float* Q, const float* K, const float* V,
        const float* beta, const float* decay,
        float* state, float* output,
        int n_heads, int n_kv_heads, int d_k, int d_v) = 0;
};
```

### 3.2 ITensorShortConvolution Interface

**File**: `src/v2/tensors/TensorKernels.h` (extend existing)

```cpp
/// Kernel interface for causal depthwise Conv1d.
class ITensorShortConvolution {
public:
    virtual ~ITensorShortConvolution() = default;

    /// Prefill: full-sequence causal convolution.
    /// input: (seq_len, channels)
    /// weight: (channels, 1, kernel_size)  — depthwise
    /// bias: (channels,) or nullptr
    /// output: (seq_len, channels)
    /// conv_state_out: (channels, kernel_size-1)  — saved for decode
    virtual bool compute_prefill(
        const float* input, const float* weight, const float* bias,
        float* output, float* conv_state_out,
        int seq_len, int channels, int kernel_size) = 0;

    /// Decode: single-step convolution with state update.
    /// input: (1, channels)
    /// conv_state: (channels, kernel_size-1) — updated in-place (shift + insert)
    /// output: (1, channels)
    virtual bool compute_decode(
        const float* input, const float* weight, const float* bias,
        float* conv_state, float* output,
        int channels, int kernel_size) = 0;
};
```

### 3.3 Gated RMSNorm

GDN uses a **gated** variant of RMSNorm where the output is element-wise multiplied by a gate vector before normalization:

```
GatedRMSNorm(x, gate) = RMSNorm(x ⊙ gate)
```

This can be implemented as a variant of the existing RMSNorm kernel.

### 3.4 Recurrent State Tensor

A new buffer type for the fixed-size GDN state matrix:

**File**: `src/v2/tensors/GDNStateTensor.h` (new)

```cpp
/// Fixed-size recurrent state for GDN layers.
/// Shape: (n_heads, d_k, d_v) per layer.
/// Unlike KV cache, this does NOT grow with sequence length.
class GDNStateTensor : public FP32Tensor {
public:
    GDNStateTensor(int n_heads, int d_k, int d_v);
    void reset();  // Zero the state (between sequences)

    int n_heads() const;
    int d_k() const;
    int d_v() const;
};
```

### 3.5 Convolution State Buffer

Small per-layer buffer for the short convolution sliding window:

```cpp
/// Sliding window state for short convolution.
/// Shape: (channels, kernel_size - 1) per projection per layer.
class ConvStateTensor : public FP32Tensor {
public:
    ConvStateTensor(int channels, int kernel_size);
    void reset();
};
```

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| GDN kernel interface | `TensorKernels.h` | `ITensorGatedDeltaNet` |
| Conv kernel interface | `TensorKernels.h` | `ITensorShortConvolution` |
| GDN state tensor | `tensors/GDNStateTensor.h` | Fixed-size recurrent state |
| Conv state buffer | `tensors/ConvStateTensor.h` | Sliding window for decode |
| Gated RMSNorm | `kernels/cpu/primitives/` | Element-wise gated normalization |

---

## Phase 4: Graph Schema and Stages

**Goal**: Design the `Qwen3NextSchema` that supports per-layer dispatch between GDN and softmax attention.

### 4.1 Qwen3NextGraphConfig

**File**: `src/v2/models/qwen3next/Qwen3NextGraph.h` (new)

```cpp
struct Qwen3NextGraphConfig {
    // Inherited from Qwen2
    int n_layers;
    int d_model;
    int n_heads;
    int n_kv_heads;
    int head_dim;       // d_k
    int d_ff;
    int vocab_size;

    // GDN-specific
    std::vector<int> attn_type;   // Per-layer: 0=GDN, 1=softmax
    float expand_v;               // Value expansion (default 2.0)
    int conv_kernel_size;         // Short conv kernel size (default 4)
    int sliding_window;           // For softmax layers (default 4096)
    int d_v;                      // Computed: head_dim * expand_v

    // TP config (inherited)
    int tp_degree = 1;
    int tp_rank = 0;
};
```

### 4.2 Per-Layer Stage Selection in Schema

The schema factory must generate different stage specs per layer depending on `attn_type[i]`:

```
For each layer i:
    ├── attn_norm (RMSNorm)                   — SAME for both types
    │
    ├── IF attn_type[i] == 0 (GDN):
    │   ├── gdn_qkv_proj                     — Projects Q,K,V + a,b,g
    │   ├── gdn_short_conv_q                  — Conv1d on Q
    │   ├── gdn_short_conv_k                  — Conv1d on K
    │   ├── gdn_short_conv_v                  — Conv1d on V
    │   ├── gdn_gate_compute                  — β = sigmoid(−softplus(A_log + a))
    │   ├── gdn_rope                          — RoPE on Q, K
    │   ├── gdn_recurrence                    — Core state update + output
    │   ├── gdn_gated_rmsnorm                 — Gated normalization
    │   └── gdn_wo_proj                       — Output projection
    │
    ├── IF attn_type[i] == 1 (Softmax):
    │   ├── qkv_proj                          — Standard Q,K,V projection
    │   ├── rope                              — RoPE on Q, K
    │   ├── kv_append                         — Append to KV cache
    │   ├── attention                         — Sliding-window FlashAttention
    │   └── wo_proj                           — Output projection
    │
    ├── attn_residual                         — SAME for both types
    ├── ffn_norm (RMSNorm)                    — SAME for both types
    ├── gate_up_proj                          — SAME for both types
    ├── swiglu                                — SAME for both types
    ├── down_proj                             — SAME for both types
    └── ffn_residual                          — SAME for both types
```

### 4.3 New ComputeStage Types

| Stage | Type | Description |
|-------|------|-------------|
| `GDNProjectionStage` | GEMM (fused) | Projects input → Q, K, V, a, b, g in one or two fused GEMMs |
| `ShortConvolutionStage` | NEW | Causal depthwise Conv1d with state management |
| `GDNGateComputeStage` | NEW | Computes β (write gate) and decay from a_proj, g_proj, A_log, dt_bias |
| `GDNRecurrenceStage` | NEW | Core linear recurrence: state update + output computation |
| `GatedRMSNormStage` | NEW | Gated variant of RMSNorm |
| `SlidingWindowAttentionStage` | Modified | Existing attention with window mask parameter |

### 4.4 Buffer Specifications

New buffers needed **per GDN layer**:

| Buffer | Shape | Lifetime | Notes |
|--------|-------|----------|-------|
| `gdn_state_{i}` | (n_heads, d_k, d_v) | Persistent (across tokens) | Replaces KV cache for GDN layers |
| `conv_state_q_{i}` | (n_heads × d_k, conv_k − 1) | Persistent | Q convolution sliding window |
| `conv_state_k_{i}` | (n_kv_heads × d_k, conv_k − 1) | Persistent | K convolution sliding window |
| `conv_state_v_{i}` | (n_kv_heads × d_v, conv_k − 1) | Persistent | V convolution sliding window |
| `a_proj_out_{i}` | (seq_len, n_heads) | Transient | Can alias scratch buffer |
| `b_proj_out_{i}` | (seq_len, n_heads) | Transient | Can alias scratch buffer |
| `g_proj_out_{i}` | (seq_len, n_heads × d_k) | Transient | Can alias scratch buffer |
| `beta_{i}` | (seq_len, n_heads) | Transient | Write gate after sigmoid |
| `decay_{i}` | (seq_len, n_heads, d_k) | Transient | State decay after exp(g) |
| `gdn_output_{i}` | (seq_len, n_heads, d_v) | Transient | Recurrence output |

**Memory savings**: GDN layers replace KV cache (grows with seq_len) with fixed-size state + conv buffers. For a 128K context model, GDN state is ~2 MB/layer vs ~4 GB/layer for KV cache. Transient buffers can reuse the existing buffer aliasing system.

### 4.5 Weight Sharding for TP

| Weight | Sharding Mode | Notes |
|--------|--------------|-------|
| `a_proj` | COLUMN_PARALLEL | Shard across heads |
| `b_proj` | COLUMN_PARALLEL | Shard across heads |
| `g_proj` | COLUMN_PARALLEL | Shard across heads |
| `A_log` | COLUMN_PARALLEL | Shard across heads |
| `dt_bias` | COLUMN_PARALLEL | Shard across heads |
| `q_conv_weight` | COLUMN_PARALLEL | Shard with Q heads |
| `k_conv_weight` | COLUMN_PARALLEL | Shard with KV heads |
| `v_conv_weight` | COLUMN_PARALLEL | Shard with KV heads |
| `gdn_wo` | ROW_PARALLEL + allreduce | Same pattern as standard Wo |
| `gated_rmsnorm_weight` | REPLICATE | Small, replicate everywhere |

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| Graph config | `models/qwen3next/Qwen3NextGraph.h` | Config struct with per-layer attn_type |
| Schema factory | `models/qwen3next/Qwen3NextSchema.h` | Per-layer stage dispatch |
| GDN stages | `execution/stages/gdn/` | All new GDN compute stages |
| Sliding-window support | `execution/stages/` | Window mask for softmax layers |
| State management | `execution/` | GDN state and conv state lifetime management |

---

## Phase 5: CPU Reference Kernels

**Goal**: Implement correct (not fast) CPU kernels for GDN to validate against the Python reference.

### 5.1 GDN Recurrence Reference Kernel

**File**: `src/v2/kernels/cpu/gdn/CPUGDNReferenceKernel.h/cpp`

Direct translation of the fla naive implementation:

```cpp
// Scalar recurrence — correct, not optimized
for (int h = 0; h < n_heads; ++h) {
    for (int t = 0; t < seq_len; ++t) {
        float beta_t = beta[t * n_heads + h];
        
        // Retrieve = S^T @ k_t
        for (int j = 0; j < d_v; ++j) {
            float retrieve = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                retrieve += state[h * d_k * d_v + i * d_v + j] * K[t * d_k + i];
            }
            
            // Delta = v_t - retrieve
            float delta = V[t * d_v + j] - retrieve;
            
            // State update: S = diag(decay) * S + beta * k * delta^T
            for (int i = 0; i < d_k; ++i) {
                state[h * d_k * d_v + i * d_v + j] *= decay[t * d_k + i];
                state[h * d_k * d_v + i * d_v + j] += beta_t * K[t * d_k + i] * delta;
            }
        }
        
        // Output: o_t = S^T @ q_t
        for (int j = 0; j < d_v; ++j) {
            float out = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                out += state[h * d_k * d_v + i * d_v + j] * Q[t * d_k + i];
            }
            output[t * n_heads * d_v + h * d_v + j] = out;
        }
    }
}
```

### 5.2 Short Convolution Reference Kernel

**File**: `src/v2/kernels/cpu/gdn/CPUShortConvReferenceKernel.h/cpp`

Simple causal depthwise convolution:

```cpp
// Prefill: causal conv1d (kernel_size=4)
for (int t = 0; t < seq_len; ++t) {
    for (int c = 0; c < channels; ++c) {
        float sum = bias ? bias[c] : 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int src_t = t - k;
            if (src_t >= 0) {
                sum += input[src_t * channels + c] * weight[c * kernel_size + k];
            }
        }
        output[t * channels + c] = sum;
    }
}
```

### 5.3 Gate Computation Reference

**File**: `src/v2/kernels/cpu/gdn/CPUGDNGateKernel.h/cpp`

```cpp
// β = sigmoid(-softplus(A_log + a_proj_out))
// decay = exp(g_proj_out)  (with clamping)
```

### 5.4 KernelFactory Extension

**File**: `src/v2/kernels/KernelFactory.h/cpp`

Add factory methods:
```cpp
static std::unique_ptr<ITensorGatedDeltaNet> createGDN(DeviceType device);
static std::unique_ptr<ITensorShortConvolution> createShortConv(DeviceType device);
```

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| GDN reference kernel | `kernels/cpu/gdn/CPUGDNReferenceKernel.h/cpp` | Scalar recurrence |
| Short conv kernel | `kernels/cpu/gdn/CPUShortConvReferenceKernel.h/cpp` | Causal depthwise conv |
| Gate computation | `kernels/cpu/gdn/CPUGDNGateKernel.h/cpp` | Beta/decay computation |
| KernelFactory | `kernels/KernelFactory.h/cpp` | Factory methods for GDN kernels |
| Unit tests | `tests/v2/unit/Test__GDNReferenceKernel.cpp` | Correctness vs Python reference |

---

## Phase 6: Optimized Kernels (CPU JIT, CUDA, ROCm)

**Goal**: High-performance GDN kernels for production inference.

> **Note**: This phase can proceed incrementally. The reference kernels from Phase 5 are sufficient for correctness validation. Optimized kernels are for competitive throughput.

### 6.1 CPU Optimized (AVX-512 / AVX2)

**Key optimizations**:

| Operation | Optimization Strategy |
|-----------|----------------------|
| State update (d_k × d_v matrix) | SIMD outer product: `state[i][j] = decay[i] * state[i][j] + beta * k[i] * delta[j]` |
| Output (S^T @ q) | SIMD GEMV with vectorized dot products |
| Short convolution | SIMD depthwise conv (kernel_size=4 fits in one register) |
| Gate computation | Vectorized sigmoid, softplus, exp |
| Chunk-parallel prefill | OpenMP over heads, SIMD within head |

**Potential JIT approach**: For decode (single token), the recurrence is a tight loop over heads that can be JIT-compiled similarly to the existing attention JIT microkernels.

### 6.2 CUDA Kernels

**Key considerations**:
- The fla library provides **Triton kernels** for GDN. These serve as the algorithmic reference.
- For CUDA C++, the chunk-parallel algorithm processes the sequence in chunks of size C, with intra-chunk parallel scan and inter-chunk sequential state propagation.
- Prefill: chunk-parallel with shared memory for state tiles
- Decode: single-token kernel, one thread block per head

### 6.3 ROCm Kernels

Follow the same pattern as CUDA kernels, adapted for HIP. The ROCm GDN kernel can be a HIPified version of the CUDA kernel.

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| CPU optimized | `kernels/cpu/gdn/CPUGDNOptimizedKernel.h/cpp` | AVX-512 vectorized |
| CPU JIT (optional) | `kernels/cpu/gdn/jit/` | JIT decode microkernel |
| CUDA kernel | `kernels/cuda/gdn/CUDAGDNKernel.h/cu` | Chunk-parallel prefill + fused decode |
| ROCm kernel | `kernels/rocm/gdn/ROCmGDNKernel.h/hip` | HIP port of CUDA kernel |
| Parity tests | `tests/v2/integration/parity/` | Optimized vs reference kernel parity |
| Perf benchmarks | `tests/v2/performance/` | Head-to-head vs softmax attention |

---

## Phase 7: MoE Integration

**Goal**: Support Qwen3-Next MoE variants (some models use Mixture-of-Experts FFN blocks).

### 7.1 MoE Architecture

MoE models replace the dense FFN with:
```
Input → Router (top-k expert selection) → Expert FFN × N → Weighted combine → Output
```

Each expert is a standard SwiGLU FFN (gate_up_proj + swiglu + down_proj). The router selects top-k experts per token.

### 7.2 New Stages

| Stage | Description |
|-------|-------------|
| `MoERouterStage` | Computes router logits, selects top-k experts |
| `MoEDispatchStage` | Gathers tokens per expert, forms per-expert batches |
| `MoEExpertFFNStage` | Runs selected expert FFNs (batched GEMM) |
| `MoECombineStage` | Weighted scatter-add of expert outputs |

### 7.3 Schema Extension

```cpp
struct Qwen3NextMoeSchemaFactory {
    // Layers identical to dense except FFN block:
    //   ffn_norm → moe_router → moe_dispatch → moe_expert_ffn → moe_combine → ffn_residual
    // GDN/softmax attention dispatch is identical to dense variant
};
```

### 7.4 Weight Sharding for MoE

| Weight | Sharding | Notes |
|--------|----------|-------|
| Router weights | REPLICATE | Small, all ranks need full router |
| Expert FFN weights | EXPERT_PARALLEL | Different experts on different devices |
| Shared expert (if any) | Same as dense FFN | Standard column/row parallel |

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| MoE stages | `execution/stages/moe/` | Router, dispatch, expert FFN, combine |
| MoE schema | `models/qwen3next/Qwen3NextMoeSchema.h` | MoE variant schema factory |
| Expert parallelism | `WeightManager` extension | Expert-level sharding |
| Unit tests | `tests/v2/unit/Test__MoERouter.cpp` etc. | Router correctness, dispatch/combine |

---

## Phase 8: End-to-End Validation

**Goal**: Confirm that the full Qwen3-Next inference pipeline produces correct output.

### 8.1 Parity Tests

| Test | Description | Tolerance |
|------|-------------|-----------|
| Layer-by-layer GDN parity | C++ GDN vs Python fla reference per layer | Cosine sim > 0.999 |
| Layer-by-layer softmax parity | C++ softmax layers vs Python (same as Qwen2) | Cosine sim > 0.999 |
| Full-model logit parity | Final logits: C++ vs Python | KL divergence < 0.01 |
| Top-K token overlap | Greedy decode token match | 100% for first 50 tokens |
| TP parity (2-way) | TP=2 vs TP=1 output match | Cosine sim > 0.9999 |

### 8.2 Integration Tests

| Test | Description |
|------|-------------|
| Hybrid layer dispatch | Verify correct stage selection per layer type |
| GDN state persistence | State correctly maintained across decode steps |
| Conv state persistence | Conv sliding window maintained across decode steps |
| State reset | State correctly zeroed between sequences |
| Long-sequence GDN | Verify fixed memory usage at 128K+ tokens |
| MoE routing | Correct expert selection and output combination |

### 8.3 Performance Benchmarks

| Benchmark | Metric | Comparison |
|-----------|--------|------------|
| Prefill throughput | tok/s | vs Qwen2.5 equivalent size |
| Decode throughput | tok/s | vs Qwen2.5 equivalent size |
| Memory usage (GDN vs KV cache) | MB/layer | Quantify savings at various seq_len |
| TP scaling | Linear speedup | 1/2/4-way TP efficiency |

### Deliverables

| Item | File/Location | Description |
|------|--------------|-------------|
| GDN parity test | `tests/v2/integration/parity/Test__Qwen3Next_GDN_Parity.cpp` | Layer-by-layer |
| E2E parity test | `tests/v2/integration/parity/Test__Qwen3Next_E2E_Parity.cpp` | Full model |
| Integration tests | `tests/v2/integration/Test__Qwen3NextHybridDispatch.cpp` | Hybrid layer dispatch |
| Perf benchmarks | `tests/v2/performance/Test__Qwen3NextBenchmark.cpp` | Throughput comparison |

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GGUF format not yet standardized for Qwen3-Next | Blocks Phase 2 | Medium | Monitor llama.cpp repo; implement flexible tensor name mapping |
| GDN numerical stability (exp overflow in decay) | Incorrect output | Medium | Use log-space computation throughout; clamp decay values |
| Chunk-parallel prefill complexity | Slow/incorrect prefill | Medium | Start with sequential reference; validate against fla Triton kernels |
| MoE expert parallelism complexity | Blocks Phase 7 | Medium | Implement dense variant first; MoE is additive |
| Value expansion (d_v ≠ d_k) breaks assumptions | Subtle bugs | High | Audit all code paths that assume d_v == d_k (currently true for Qwen2) |
| Per-layer dispatch adds graph complexity | Maintenance burden | Low | Clean schema abstraction; test both layer types in isolation |
| Model not yet publicly released | Cannot validate end-to-end | Medium | Build infrastructure now; validate with synthetic weights + small test models from fla |

---

## Implementation Priority

| Priority | Phase | Rationale |
|----------|-------|-----------|
| 1 | Phase 1 (Python reference) | Establishes ground truth; unblocks all parity testing |
| 2 | Phase 2 (GGUF loading) | Required to load any model weights |
| 3 | Phase 3 (Interfaces + types) | Defines the API contract for all kernels |
| 4 | Phase 4 (Schema + stages) | Core graph infrastructure for hybrid dispatch |
| 5 | Phase 5 (CPU reference) | Correctness validation — must pass parity before optimization |
| 6 | Phase 8 (Validation) | Runs in parallel with Phase 5; confirms correctness |
| 7 | Phase 6 (Optimized kernels) | Performance — only after correctness is proven |
| 8 | Phase 7 (MoE) | Additive feature; independent of GDN correctness |

---

## External References

| Resource | URL | Notes |
|----------|-----|-------|
| GDN Paper (ICLR 2025) | https://arxiv.org/abs/2412.06464 | *Gated Delta Networks: Improving Mamba2 with Delta Rule* |
| flash-linear-attention | https://github.com/fla-org/flash-linear-attention | PyTorch + Triton reference implementations |
| fla GDN layer | `fla/layers/gated_deltanet.py` | Complete GDN layer (~318 lines) |
| fla GDN naive reference | `fla/ops/gated_delta_rule/naive.py` | Scalar reference (~156 lines) |
| fla GDN model | `fla/models/gated_deltanet/modeling_gated_deltanet.py` | HuggingFace-compatible model (~381 lines) |
| fla GDN config | `fla/models/gated_deltanet/configuration_gated_deltanet.py` | Model configuration class |
| Qwen3 Blog | https://qwenlm.github.io/blog/qwen3/ | Qwen3 architecture reference (GQA, not GDN) |
| Qwen3 GitHub | https://github.com/QwenLM/Qwen3 | Official repo |
