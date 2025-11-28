# Llaminar V2 Architecture (Operator-Free Design)

This document is a high-level but concrete map of the Llaminar V2 stack: tensors, kernels, devices, pipelines, MPI orchestration, and attention. It is intended as a **quick-start reference for future agents** so they can safely modify V2 without re-deriving the architecture from scratch.

---

## 1. Design Goals

V2 is a **kernel-centric, operator-free** architecture:

- **Per-tensor device affinity** – each tensor knows which device it lives on.
- **Direct kernel calls from pipelines** – no heavyweight operator graph; pipelines orchestrate kernels explicitly.
- **Heterogeneous execution** – CPU / CUDA / ROCm / (future) backends can be mixed in one run.
- **Quantization-aware kernels** – unified GEMM/attention interfaces that work with FP32/BF16 and quantized formats.
- **MPI-aware orchestration** – multi-rank inference (tensor/sequence/pipeline parallel) lives in a small set of orchestrators.

The mental model:

> **Pipeline** (per model) calls into **device-aware tensors**, which expose **ITensor* interfaces** to create the right **kernels** for GEMM, attention, etc. MPI and multi-device routing is handled by **orchestrators**, not by kernels.

---

## 2. Tensors and Tensor Interfaces

### 2.1 Core Tensor Types

Location: `src/v2/tensors/`

Key concepts:

- `TensorBase` – abstract base: shape, dtype, device, and virtual hooks to create kernels.
- Concrete tensors:
  - `FP32Tensor`, `BF16Tensor`, `FP16Tensor` – dense float tensors (CPU today, future device-aware variants).
  - Quantized tensors: `IQ4_NLTensor`, `Q4_0Tensor`, `Q6_KTensor`, `Q8_0Tensor`, etc.
  - View/alias tensors (e.g., *views* over quantized buffers) for cheap slicing/reinterpretation.

Each tensor type exposes methods to construct appropriate kernels via interfaces, for example:

- `std::unique_ptr<ITensorGemm> createGemm() const;`
- `std::unique_ptr<ITensorAttention> createAttention() const;`

These are **factory hooks**: pipelines never instantiate kernels directly. They always go through tensors, which encapsulate device + layout details.

### 2.2 Tensor Kernel Interfaces

Location: `src/v2/tensors/TensorKernels.h`

Canonical interfaces:

- `ITensorGemm`
  - GEMM between activations and/or weights.
  - Supports activation-activation and activation-weight paths, plus transposition flags.
- `ITensorAttention`
  - Attention over Q/K/V (single sequence or batched), with optional mask and MPI context.
- Other per-op interfaces (e.g. RMSNorm, SwiGLU, RoPE) follow the same pattern: **one narrow interface per logical op**, created from tensors.

Example (conceptual):

```cpp
class ITensorGemm {
public:
    virtual bool multiply_activations(
        const float* A,
        const float* B,
        float* C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta) = 0;
};

class ITensorAttention {
public:
    virtual bool compute(
        const float* Q,
        const float* K,
        const float* V,
        float* output,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        TensorBase* workspace_scores,
        TensorBase* workspace_qkv,
        TensorBase* workspace_context,
        TensorBase* mask,               // optional
        bool use_bf16,
        MPIContext* mpi_ctx,           // optional
        int layer_index) = 0;

    virtual bool compute_batch(
        const float* Q,
        const float* K,
        const float* V,
        float* output,
        int batch_size,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        TensorBase* workspace_scores,
        TensorBase* workspace_qkv,
        TensorBase* workspace_context,
        TensorBase* mask,
        bool use_bf16,
        MPIContext* mpi_ctx,
        int layer_index) = 0;
};
```

Tensors decide which **concrete kernel implementation** backs these interfaces (e.g. CPU OpenBLAS/oneDNN, GPU kernel, quantized path, etc.).

---

## 3. Kernels and Device Backends

### 3.1 CPU Kernels

Location: `src/v2/kernels/cpu/`

- GEMM: oneDNN / OpenBLAS-backed GEMM kernels accessible via `ITensorGemm`.
- Attention: `CpuAttentionKernelT<T>` implements `ITensorAttention` for FP32/BF16 activations.
- Vectorized primitives live in `primitives/` (e.g. `SoftmaxPrimitives.h`) and are used by kernels.

CPU kernels are **MPI-agnostic**: they operate on local buffers only. Any MPI-related coordination (allreduce, scatter/gather) is done *outside* in orchestrators.

### 3.2 Quantized GEMM and Block Decoders

Location: `src/v2/tensors/`, `src/v2/kernels/`

Quantized formats share a **strategy-style decoder** interface:

- `IBlockDecoder`
  - `decode_block_at(row_idx, k_block_offset, float* out)` – decode a quantized block to FP32.
  - `block_size()` – elements per block.

Quantized tensors implement both `TensorBase` and `IBlockDecoder`:

- `IQ4_NLTensor : public TensorBase, public IBlockDecoder`
- `Q6_KTensor : public TensorBase, public IBlockDecoder`

A **generic quantized GEMM kernel** (e.g. `QuantizedGemmKernel`) takes an `IBlockDecoder*` and implements `ITensorGemm`. `decode_block_at` is marked `always_inline`, so the virtual call is de-virtualized in practice.

This yields:

- One generic quantized GEMM implementation.
- New formats are added by implementing `IBlockDecoder` and returning `QuantizedGemmKernel` from `createGemm()`.

### 3.3 Device Backends

Location: `src/v2/backends/`

- `Device` / `DeviceManager` abstractions encapsulate **where** a tensor lives and how to allocate kernels.
- Pipelines ask the `DeviceManager` to pick devices for weights/activations and then rely on tensors to materialize the right kernels.
- Today, CPU is the primary backend; the design is forward-compatible with CUDA/ROCm/Vulkan backends.

---

## 4. MPI Layer and Orchestrators

### 4.1 MPI Context

Location: `src/v2/utils/MPIContext.h`, `src/v2/utils/MPIStager.h`

`MPIContext` wraps raw MPI calls and exposes:

- `rank()`, `world_size()`
- `allreduce_sum(...)`, `broadcast(...)`, `barrier()`, etc.
- Helper methods like `get_local_slice(total_heads)` to compute per-rank head ranges for tensor parallelism.

`MPIStager` handles staging buffers to/from host memory when tensors live on GPUs (today a no-op for CPU tensors).

### 4.2 MPIStrategy

Location: `src/v2/pipelines/MPIStrategy.h`

Defines how work is split across ranks:

- `MPIStrategy::None` – single-rank.
- `MPIStrategy::TensorParallel` – split heads across ranks.
- `MPIStrategy::SequenceParallel` – split sequence dimension (future).
- `MPIStrategy::PipelineParallel` – split layers across ranks.
- `MPIStrategy::Hybrid` – mix of the above.

### 4.3 MpiAttentionOrchestrator

Location: `src/v2/pipelines/attention/MpiAttentionOrchestrator.*`

This is the **only place** that knows how to route attention across ranks.

Responsibilities:

- Interpret `MPIStrategy` and `MPIContext`.
- For tensor-parallel:
  - Compute `[start_head, local_n_heads)` per rank.
  - Slice local Q/K/V head chunks.
  - Call into the shared attention front-end (`GQAAttention`) or directly into `ITensorAttention` for the local slice.
  - Allreduce outputs across ranks when needed.
- For sequence/pipeline/hybrid: use similar patterns (barriers + collectives), but on sequence or layer dimensions.

Kernels stay oblivious to MPI; they only see local pointers.

---

## 5. Attention Abstractions

### 5.1 ITensorAttention and CpuAttentionKernelT

Location:

- Interface: `src/v2/tensors/TensorKernels.h`
- CPU implementation: `src/v2/kernels/cpu/CpuAttentionKernelT.h` (and related files)

`CpuAttentionKernelT<T>` implements `ITensorAttention` for `T=fp32`, `T=bf16`, etc. It:

- Interprets Q, K, V as `[seq_len, n_heads * head_dim]` and `[seq_len, n_kv_heads * head_dim]`.
- Performs:
  - Optional GQA-like broadcasting of K/V (or assumes already broadcasted by caller, depending on path).
  - QK^T matmul via `ITensorGemm`.
  - Scaling + masking (mask is provided as a tensor by caller).
  - Softmax via vectorized primitives.
  - Matmul with V to produce context.

All of that is **per-rank local**; any cross-rank behavior is layered above.

### 5.2 GQAAttention (Front-End)

Location: `src/v2/pipelines/attention/GQAAttention.*`

`GQAAttention` is a small, static-only front-end responsible for **GQA/MQA/MHA semantics**:

- Inputs:
  - Q, K, V tensors (`TensorBase*`), flattened as `[batch * seq_len, n_heads * head_dim]` and `[batch * seq_len, n_kv_heads * head_dim]`.
  - `GQAAttentionConfig`:
    - `n_heads`, `n_kv_heads`, `head_dim`.
    - `causal`, `window_size`.
    - `precision` (FP32/BF16).
    - MPI-related fields (`mpi_ctx`, `mpi_strategy`, `verbose_logging`).
    - Workspace tensors for scores, qkv, context, mask.

- Public entry points:
  - `compute(Q, K, V, output, config, batch_size, sequence_lengths)` – single-sequence or batched.
  - `compute_batch(Q, K, V, output, actual_lengths, batch_size, seq_len, config)` – explicitly batched.
  - `compute_mpi(...)` – today a thin wrapper that forwards to `compute` because MPI routing is handled by `MpiAttentionOrchestrator`.

- Responsibilities:
  - **Validation**: shapes, head counts, divisibility, non-null pointers.
  - **Mask construction** (via local helpers):
    - Single-sequence causal + window mask.
    - Batched block-diagonal masks (per-sequence lengths).
    - Combined causal + padding masks.
  - **Kernel dispatch**:
    - Downcast `output` to `IActivationTensor*`.
    - `auto attention_kernel = activation_output->createAttention();`
    - Call `attention_kernel->compute` or `compute_batch` with proper arguments.

All heavy math lives behind `ITensorAttention`; `GQAAttention` is mostly glue + masking.

> For future work (e.g. DeepSeek-style multi-head latent attention), this is the pattern to follow: add a new, small attention front-end (`LatentAttention`, etc.) that encodes the semantics and calls into shared kernels or specialized kernels.

---

## 6. Pipelines and Model Orchestration

### 6.1 Pipeline Overview

Location: `src/v2/pipelines/`

Each model family (e.g. Qwen 2.5) has a **pipeline** that orchestrates the entire forward pass:

- Owns model weights (tensors), KV cache, and configuration.
- Builds per-layer execution steps: attention, MLP, norms, projections.
- Interacts with `DeviceManager` to place weights and activations on devices.
- Uses tensor factories and `ITensor*` interfaces to create kernels.

Key responsibilities:

1. **Initialization**
   - Load GGUF weights via `ModelLoader` into appropriate tensor types (FP32/BF16/quantized).
   - Initialize KV cache tensors and any per-layer workspaces.

2. **Prefill / Decode Execution**
   - Prefill: run model over prompt sequence (`seq_len > 1`), filling KV cache.
   - Decode: step-by-step generation (`seq_len = 1` per token), reading from KV cache.

3. **Layer Loop (per step)**
   - Apply embedding + positional encoding (e.g., RoPE via `ITensorRoPE`).
   - Run attention:
     - Build/configure `GQAAttentionConfig` (or future attention front-ends).
     - If MPI is enabled, call into `MpiAttentionOrchestrator`, which will:
       - Use `GQAAttention`/`ITensorAttention` for local computation.
       - Handle collectives.
     - Else, call `GQAAttention::compute` directly.
   - Run MLP (+ SwiGLU) via GEMM + activation kernels.
   - Run RMSNorm/LayerNorm via dedicated kernels.

4. **Logits and Sampling**
   - Final projection to vocab via GEMM.
   - Sampling done via a sampler component (e.g. greedy, top-k, nucleus) built on top of logits.

### 6.2 GEMM Usage in Pipelines

Pipelines use GEMM for:

- Q/K/V projections.
- Output projections.
- MLP layers (up-project and down-project).
- Attention context matmuls (depending on the kernel path).

Pattern:

```cpp
// Example inside a pipeline layer
auto W = /* weight tensor (FP32, BF16, or quantized) */;
auto X = /* activation tensor */;

auto gemm = W->createGemm();
bool ok = gemm->multiply_activations(
    /*A=*/X.data(),
    /*B=*/W.data(),
    /*C=*/Y.mutable_data(),
    /*m=*/batch_tokens,
    /*n=*/out_dim,
    /*k=*/in_dim,
    /*transpose_B=*/true,
    /*alpha=*/1.0f,
    /*beta=*/0.0f);
```

The pipeline does **no backend-specific logic**. The tensor + kernel implementation decide how to map this onto oneDNN/OpenBLAS/quantized kernels.

### 6.3 Pipeline Operations (Ops Layer)

Location: `src/v2/pipelines/ops/`

Pipeline operations encapsulate the **validate-execute-capture** pattern that repeats throughout pipelines:

1. Validate inputs (null checks, dimension checks)
2. Create kernel (from appropriate tensor interface)
3. Execute kernel (with error handling)
4. Capture snapshot (if enabled for E2E testing)

**Available Operations:**

| Op Class | Purpose | Key Method |
|----------|---------|------------|
| `RMSNormOp` | RMS normalization (pre-attention, pre-FFN, final) | `operator()(input, weight, output, rows, cols, eps, snapshot, mpi, device)` |
| `GemmOp` | Weight projections (Q/K/V, output, FFN) | `operator()(A, W, C, m, n, k, snapshot, mpi, device)` |
| `GemmOp::activations` | Activation matmul (attention scores) | `activations(A, B, C, m, n, k, transpose, alpha, beta, snapshot, mpi, device)` |
| `SwiGLUOp` | SwiGLU activation (FFN) | `operator()(gate, up, output, rows, cols, snapshot, mpi, device)` |
| `RoPEOp` | Rotary position embeddings | `operator()(Q, K, positions, seq_len, heads, kv_heads, dim, theta, prefix, mpi, device)` |
| `ResidualOp` | Residual connections | `operator()(residual, input, output, rows, cols, snapshot)` |
| `ResidualOp::batched` | Batched residual with padding mask | `batched(residual, input, output, batch, seq, cols, lengths, snapshot)` |

### 6.4 PipelineBase Composite Operations

Location: `src/v2/pipelines/PipelineBase.h` (lines 880-1045)

**Design Philosophy**: Ops are private members of `PipelineBase`, exposed through high-level **composite operations** that encapsulate device placement, MPI context, snapshot capture, and validation. This makes child pipelines read like **declarative compute graphs**.

**Composite Operations in PipelineBase:**

| Method | Purpose | Internal Op |
|--------|---------|-------------|
| `rms_norm()` | RMSNorm with snapshot | `rmsnorm_op_` |
| `project()` | Weight projection (GEMM) with snapshot | `gemm_op_` |
| `add_residual()` | Batch-aware residual connection | `residual_op_` |
| `swiglu()` | SwiGLU activation with snapshot | `swiglu_op_` |
| `apply_rope()` | RoPE to Q/K with snapshots | `rope_op_` |
| `compute_attention()` | GQA attention with snapshot | Direct kernel call |
| `save_residual()` | Copy tensor for residual | memcpy wrapper |
| `capture_snapshot()` | Manual snapshot capture | Direct macro call |

**PipelineBase Private Members:**

```cpp
class PipelineBase {
private:
    // Reusable operations (stateless, self-validating)
    RMSNormOp rmsnorm_op_;
    GemmOp gemm_op_;
    SwiGLUOp swiglu_op_;
    RoPEOp rope_op_;
    ResidualOp residual_op_;

protected:
    // High-level composite operations for child pipelines
    bool rms_norm(TensorBase* input, const TensorBase* gamma, TensorBase* output,
                  int rows, int cols, float eps,
                  const std::string& snapshot_key, int device = -1);

    bool project(const TensorBase* input, TensorBase* weight, TensorBase* output,
                 int m, int n, int k,
                 const std::string& snapshot_key, int device = -1);

    bool add_residual(const TensorBase* residual, const TensorBase* input, TensorBase* output,
                      int batch_size, int seq_len, int hidden_dim,
                      const std::vector<int>& sequence_lengths,
                      const std::string& snapshot_key);

    bool swiglu(TensorBase* gate, TensorBase* up, TensorBase* output,
                int rows, int cols,
                const std::string& snapshot_key, int device = -1);

    bool apply_rope(TensorBase* Q, TensorBase* K, const int* position_ids,
                    int seq_len, int n_heads, int n_kv_heads, int head_dim,
                    float theta, const std::string& snapshot_prefix, int device = -1);

    bool compute_attention(TensorBase* Q, TensorBase* K, TensorBase* V, TensorBase* output,
                           int seq_len, int n_heads, int n_kv_heads, int head_dim,
                           int batch_size, const std::vector<int>& sequence_lengths,
                           int padded_seq_len, bool causal, const std::string& snapshot_key);

    bool save_residual(const TensorBase* input, TensorBase* residual_buffer,
                       int seq_len, int hidden_dim);

    void capture_snapshot(const std::string& key, TensorBase* tensor, int rows, int cols);
};
```

### 6.5 Fused Kernels

Location: `src/v2/kernels/cpu/fused/`

Fused kernels optimize multi-projection patterns by quantizing activations once:

**FusedGEMM** (`FusedGEMM.h`):
- Fuses N GEMM projections with shared activation quantization
- Use cases: FFN gate/up (2 GEMMs), Attention Q/K/V (3 GEMMs)
- Performance: Saves N-1 activation quantization passes, better cache locality

```cpp
// FFN: Gate/Up fused (2 projections)
layer.gate_up_fused = std::make_unique<FusedGEMM>(
    layer.gate_proj.get(), layer.up_proj.get());

layer.gate_up_fused->execute(
    normalized->data(),
    {{gate->mutable_data(), nullptr, d_ff, "gate"},
     {up->mutable_data(), nullptr, d_ff, "up", nullptr, false}},
    seq_len, d_model, mpi_ctx, device);

// Attention: Q/K/V fused (3 projections)
layer.qkv_fused = std::make_unique<FusedGEMM>(
    layer.wq.get(), layer.wk.get(), layer.wv.get());

layer.qkv_fused->execute(
    normalized->data(),
    Q->mutable_data(), K->mutable_data(), V->mutable_data(),
    nullptr, nullptr, nullptr,  // biases
    seq_len,
    n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim,
    d_model, mpi_ctx, device);
```

### 6.6 Declarative Pipeline Pattern

With composite operations in PipelineBase, child pipelines now read like **declarative compute graphs**:

```cpp
bool Qwen2Pipeline::attention_block(const LayerWeights& layer, int layer_idx, int seq_len) {
    auto& buffers = activation_buffers_;
    std::string prefix = "layer" + std::to_string(layer_idx);

    // Save residual for later
    TRY_OP(save_residual(input_hidden, buffers.residual.get(), seq_len, d_model_));

    // 1. Pre-attention RMSNorm
    TRY_OP(rms_norm(buffers.residual.get(), layer.attn_norm.get(), buffers.normalized.get(),
                    seq_len, d_model_, 1e-6f, prefix + "_ATTENTION_NORM", device));

    // 2. Fused Q/K/V projections
    VALIDATE_OP(layer.qkv_fused->execute(...), "Fused Q/K/V projection");

    // Capture Q/K/V
    capture_snapshot(prefix + "_Q_PROJECTION", buffers.Q.get(), seq_len, n_heads_ * head_dim_);
    capture_snapshot(prefix + "_K_PROJECTION", buffers.K.get(), seq_len, n_kv_heads_ * head_dim_);
    capture_snapshot(prefix + "_V_PROJECTION", buffers.V.get(), seq_len, n_kv_heads_ * head_dim_);

    // 3. RoPE
    TRY_OP(apply_rope(buffers.Q.get(), buffers.K.get(), position_ids.data(),
                      seq_len, n_heads_, n_kv_heads_, head_dim_,
                      rope_theta_, prefix, device));

    // 4. GQA attention
    TRY_OP(compute_attention(buffers.Q.get(), buffers.K.get(), buffers.V.get(),
                             buffers.attn_output.get(), seq_len, n_heads_, n_kv_heads_,
                             head_dim_, batch_size_, sequence_lengths_, padded_seq_len_,
                             false, prefix + "_ATTENTION_CONTEXT"));

    // 5. Output projection
    TRY_OP(project(buffers.attn_output.get(), layer.wo.get(), buffers.attn_proj.get(),
                   seq_len, d_model_, n_heads_ * head_dim_,
                   prefix + "_ATTENTION_OUTPUT", device));

    // 6. Residual
    TRY_OP(add_residual(buffers.residual.get(), buffers.attn_proj.get(), current_hidden_.get(),
                        batch_size_, padded_seq_len_, d_model_, sequence_lengths_,
                        prefix + "_ATTENTION_RESIDUAL"));

    return true;
}

bool Qwen2Pipeline::ffn_block(const LayerWeights& layer, int layer_idx, int seq_len) {
    std::string prefix = "layer" + std::to_string(layer_idx);

    TRY_OP(save_residual(input, buffers.residual.get(), seq_len, d_model_));

    // 1. Pre-FFN RMSNorm
    TRY_OP(rms_norm(buffers.residual.get(), layer.ffn_norm.get(), buffers.normalized.get(),
                    seq_len, d_model_, 1e-6f, prefix + "_FFN_NORM", device));

    // 2. Fused Gate/Up
    VALIDATE_OP(layer.gate_up_fused->execute(...), "Fused Gate/Up projection");
    capture_snapshot(prefix + "_FFN_GATE", buffers.gate.get(), seq_len, d_ff_);
    capture_snapshot(prefix + "_FFN_UP", buffers.up.get(), seq_len, d_ff_);

    // 3. SwiGLU
    TRY_OP(swiglu(buffers.gate.get(), buffers.up.get(), buffers.up.get(),
                  seq_len, d_ff_, prefix + "_FFN_SWIGLU", device));

    // 4. Down projection
    TRY_OP(project(buffers.up.get(), layer.down_proj.get(), buffers.ffn_output.get(),
                   seq_len, d_model_, d_ff_, prefix + "_FFN_DOWN", device));

    // 5. Residual
    TRY_OP(add_residual(buffers.residual.get(), buffers.ffn_output.get(), current_hidden_.get(),
                        batch_size_, padded_seq_len_, d_model_, sequence_lengths_,
                        prefix + "_FFN_RESIDUAL"));

    return true;
}
```

**Benefits:**
- **~50 lines per block** vs ~150+ lines with manual validation/capture
- **Self-documenting**: Each step is clearly named
- **Reusable**: Same patterns work across Qwen2/Qwen3/MoE
- **Testable**: Ops can be unit tested in isolation
- **Snapshot capture built-in**: Every operation captures its output

---

## 7. Execution Modes and Testing

### 7.1 Execution Modes

- **Single-rank, single-device** (default in tests): simplest environment, no MPI.
- **MPI tensor-parallel**: multiple ranks per layer; orchestrator splits heads and synchronizes via collectives.
- **Future**: sequence-parallel/pipeline-parallel/hybrid via `MPIStrategy` and orchestrators.

### 7.2 Tests

Location: `tests/v2/`

- **Unit tests** – `tests/v2/unit/`
  - Focused on individual components (tensors, kernels, orchestrators, attention front-ends).
  - Notably: `V2_Unit_MpiAttentionOrchestrator`, `V2_Unit_CpuAttentionKernelT`, quantized tensor tests, etc.

- **Integration tests** – `tests/v2/integration/`
  - Full pipeline runs with real models; validate end-to-end behavior without parity checks.

- **E2E/Parity tests** – `tests/v2/e2e/`
  - Compare V2 outputs to PyTorch reference snapshots.

- **Performance tests** – `tests/v2/performance/`
  - Benchmarks for GEMM, attention, quantized paths, MPI tensor-parallel, etc.

V2 tests heavily rely on the interfaces described above. If you modify any of:

- `TensorBase` or tensor subclasses
- `ITensorGemm`, `ITensorAttention` (or their implementers)
- `GQAAttention`, `MpiAttentionOrchestrator`
- Pipelines

…you should run at least:

```bash
cd /workspaces/llaminar
ctest --test-dir build_v2 -R '^V2_Unit_' --output-on-failure
ctest --test-dir build_v2_release -R '^V2_Integration_' --output-on-failure
```

---

## 8. How to Safely Extend V2

When adding new features, follow these patterns:

1. **New attention style (e.g., DeepSeek-style multi-head latent attention)**
   - Add a new front-end class alongside `GQAAttention` (e.g., `LatentAttention`).
   - Define a config struct describing its semantics (visible vs latent heads, routing, extra masks).
   - Implement `compute` / `compute_batch` as small wrappers that:
     - Validate shapes.
     - Build masks and routing structures into shared workspaces.
     - Call `IActivationTensor::createAttention()->compute[_batch]` or a specialized kernel.
   - Update `MpiAttentionOrchestrator` to choose between `GQAAttention` and `LatentAttention` based on model/pipeline config.

2. **New quantized format**
   - Implement a new quantized tensor (`MyFormatTensor`) with `IBlockDecoder`.
   - Implement `decode_block_at` and `block_size`.
   - Return `QuantizedGemmKernel` (or equivalent) from `createGemm()`.
   - Add unit + performance tests.

3. **New backend (e.g., CUDA)**
   - Add a new `Device` implementation and tensor variants that allocate on that device.
   - Implement `ITensorGemm` / `ITensorAttention` kernels for that backend.
   - Keep MPI logic in orchestrators; kernels stay local-device only.

4. **Modifying MPI behavior**
   - Confine changes to `MPIContext`, `MPIStager`, and orchestrators (e.g., `MpiAttentionOrchestrator`).
   - Never bake rank/world-size assumptions into kernels.

---

## 9. Snapshot System (Pipeline Instrumentation & Debugging)

### 9.1 Overview

The **snapshot system** is a built-in instrumentation framework for capturing intermediate tensor states during pipeline execution. It enables:

- **Layer-by-layer parity testing** against reference implementations (PyTorch, llama.cpp)
- **Root cause analysis** of numerical divergence in complex pipelines
- **Performance profiling** checkpoints (future: capture + timing metadata)
- **Zero overhead in production** (compile-time conditional, optimizes away in Release builds)

**Key Design Principles:**

1. **Compile-time conditional** – Only enabled in Debug/E2ERelease builds via `ENABLE_PIPELINE_SNAPSHOTS`
2. **In-memory storage** – Snapshots stored in `std::map<std::string, std::vector<float>>` during execution
3. **Optional file output** – Can specify output directory, but default is memory-only for test assertions
4. **Minimal instrumentation overhead** – Capture points use inline macros that compile away to NOOPs in Release

### 9.2 Build Configuration

Location: `src/v2/CMakeLists.txt` (lines 540-560)

```cmake
# Enable pipeline snapshot capture (for parity testing and debugging)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (Debug build)")
elseif(CMAKE_BUILD_TYPE MATCHES "E2ERelease")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (E2ERelease build)")
else()
    option(ENABLE_SNAPSHOTS "Enable pipeline snapshot capture" OFF)
    if(ENABLE_SNAPSHOTS)
        target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    endif()
endif()
```

**Build Types:**

- **Debug**: Snapshots always enabled (development/debugging)
- **E2ERelease**: Snapshots enabled with `-O3` optimizations (optimized E2E testing)
- **Release**: Snapshots disabled by default (production), enable with `-DENABLE_SNAPSHOTS=ON`

**Creating E2ERelease build:**

```bash
cmake -B build_v2_e2e -S src/v2 -DCMAKE_BUILD_TYPE=E2ERelease
cmake --build build_v2_e2e --parallel
```

### 9.3 PipelineBase Integration

Location: `src/v2/pipelines/PipelineBase.h` (lines 665-765)

**API Methods:**

```cpp
// Enable snapshot capture (call before forward pass)
void enableSnapshotCapture(const std::string& output_dir = "");

// Disable and clear snapshots
void disableSnapshotCapture();

// Retrieve captured snapshot by key
const float* getSnapshot(const std::string& key, size_t& out_size) const;

// Get all snapshot keys
std::vector<std::string> getSnapshotKeys() const;
```

**Internal Capture Method (used by pipelines):**

```cpp
inline void captureSnapshot(const std::string& key,
                            const float* data,
                            size_t size)
{
#ifdef ENABLE_PIPELINE_SNAPSHOTS
    if (snapshot_capture_enabled_) {
        snapshots_[key].assign(data, data + size);
    }
#endif
    // In Release builds: entire function is NOOP, optimizes away
}
```

### 9.4 Instrumentation in Qwen2Pipeline

Location: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` (lines 30-750)

**Capture Macros:**

```cpp
// Full tensor capture
#define CAPTURE_SNAPSHOT(key, tensor_ptr) \
    do { \
        size_t _numel = (tensor_ptr)->element_count(); \
        captureSnapshot((key), (tensor_ptr)->data(), _numel); \
    } while(0)

// View capture (for buffers larger than actual data)
#define CAPTURE_SNAPSHOT_VIEW(key, tensor_ptr, rows, cols) \
    do { \
        auto _view = (tensor_ptr)->view({rows, cols}); \
        size_t _numel = (rows) * (cols); \
        captureSnapshot((key), _view->data(), _numel); \
    } while(0)
```

**Instrumentation Points (Qwen2Pipeline):**

```cpp
// Embedding output
CAPTURE_SNAPSHOT("EMBEDDING", current_hidden_.get());

// Per-layer captures (for each layer 0..N-1):
CAPTURE_SNAPSHOT_VIEW("layer{i}_ATTENTION_NORM", normalized, seq_len, d_model);
CAPTURE_SNAPSHOT_VIEW("layer{i}_Q_PROJECTION", Q_buffer, seq_len, n_heads * head_dim);
CAPTURE_SNAPSHOT_VIEW("layer{i}_K_PROJECTION", K_buffer, seq_len, n_kv_heads * head_dim);
CAPTURE_SNAPSHOT_VIEW("layer{i}_V_PROJECTION", V_buffer, seq_len, n_kv_heads * head_dim);
CAPTURE_SNAPSHOT_VIEW("layer{i}_Q_ROPE", Q_buffer, seq_len, n_heads * head_dim);
CAPTURE_SNAPSHOT_VIEW("layer{i}_K_ROPE", K_buffer, seq_len, n_kv_heads * head_dim);
CAPTURE_SNAPSHOT_VIEW("layer{i}_ATTENTION_CONTEXT", attn_output, seq_len, n_heads * head_dim);
CAPTURE_SNAPSHOT_VIEW("layer{i}_ATTENTION_OUTPUT", attn_proj, seq_len, d_model);
CAPTURE_SNAPSHOT("layer{i}_ATTENTION_RESIDUAL", current_hidden_.get());
CAPTURE_SNAPSHOT_VIEW("layer{i}_FFN_NORM", normalized, seq_len, d_model);
CAPTURE_SNAPSHOT_VIEW("layer{i}_FFN_GATE", gate_buffer, seq_len, d_ff);
CAPTURE_SNAPSHOT_VIEW("layer{i}_FFN_UP", up_buffer, seq_len, d_ff);
CAPTURE_SNAPSHOT_VIEW("layer{i}_FFN_SWIGLU", up_buffer, seq_len, d_ff);
CAPTURE_SNAPSHOT_VIEW("layer{i}_FFN_DOWN", down_output, seq_len, d_model);
CAPTURE_SNAPSHOT("layer{i}_FFN_RESIDUAL", current_hidden_.get());

// Final output
CAPTURE_SNAPSHOT("FINAL_NORM", current_hidden_.get());
```

**Key Naming Convention:**

- **Global stages**: `EMBEDDING`, `FINAL_NORM`
- **Layer-specific**: `layer{i}_{STAGE}` where stage is:
  - Attention: `ATTENTION_NORM`, `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`, `Q_ROPE`, `K_ROPE`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL`
  - FFN: `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_SWIGLU`, `FFN_DOWN`, `FFN_RESIDUAL`

### 9.5 Using Snapshots in Tests

**Example: Layer-by-Layer Parity Testing**

Location: `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp`

```cpp
TEST_F(Qwen2E2ECorrectness, MultiSequenceBatch) {
    // ... setup ...
    
    // Sequential execution (baseline)
    std::vector<std::unique_ptr<Qwen2Pipeline>> pipelines_seq(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        pipelines_seq[i] = std::make_unique<Qwen2Pipeline>(
            model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, 1);
        
        // Enable snapshot capture
        pipelines_seq[i]->enableSnapshotCapture("/tmp/snapshots_seq_" + std::to_string(i));
        
        pipelines_seq[i]->forward(batch[i].data(), batch[i].size());
    }
    
    // Batched execution
    auto pipeline_batch = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, PipelineConfig{}, batch_size);
    
    pipeline_batch->enableSnapshotCapture("/tmp/snapshots_batch");
    pipeline_batch->forward_batch(batch);
    
    // Compare snapshots layer-by-layer
    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<std::string> keys = {
            "layer" + std::to_string(layer) + "_Q_ROPE",
            "layer" + std::to_string(layer) + "_K_ROPE",
            "layer" + std::to_string(layer) + "_ATTENTION_CONTEXT"
        };
        
        for (const auto& key : keys) {
            size_t seq_size = 0, batch_size_snap = 0;
            const float* seq_data = pipelines_seq[1]->getSnapshot(key, seq_size);
            const float* batch_data = pipeline_batch->getSnapshot(key, batch_size_snap);
            
            if (seq_data && batch_data) {
                // Extract sequence portion from batched snapshot
                size_t feature_dim = seq_size / seq_len;
                size_t batch_offset = padded_seq_len * feature_dim * sequence_index;
                const float* batch_seq_data = batch_data + batch_offset;
                
                auto result = compareTensors(seq_data, batch_seq_data, seq_size, tolerance);
                
                LOG_INFO("Layer " << layer << " " << key << ": "
                         << "max_diff=" << result.max_abs_diff << ", "
                         << "mean=" << result.mean_abs_diff << ", "
                         << "status=" << (result.passed ? "PASS" : "FAIL"));
            }
        }
    }
}
```

**Real-World Example Output:**

```
[INFO] Layer 0 Q_ROPE: max_diff=0, mean=0, status=PASS
[INFO] Layer 0 K_ROPE: max_diff=0, mean=0, status=PASS
[INFO] Layer 0 ATTENTION_CONTEXT: max_diff=0.269798, mean=0.0187709, status=FAIL
[INFO] Layer 1 Q_ROPE: max_diff=6.35003, mean=0.773988, status=FAIL
[INFO] Layer 1 K_ROPE: max_diff=6.21053, mean=1.07118, status=FAIL
```

**Interpretation:**

- Layer 0 RoPE is perfect → RoPE implementation is correct
- Layer 0 ATTENTION_CONTEXT diverges → **Root cause is in attention computation**
- Layer 1 diverges because Layer 0 output was wrong (error propagation)

### 9.6 Troubleshooting Workflow

**Step 1: Enable Snapshots**

Use E2ERelease build for optimized testing:

```bash
cmake -B build_v2_e2e -S src/v2 -DCMAKE_BUILD_TYPE=E2ERelease
cmake --build build_v2_e2e --parallel
```

**Step 2: Instrument Test**

Enable capture in both baseline and test execution:

```cpp
pipeline_baseline->enableSnapshotCapture();
pipeline_test->enableSnapshotCapture();
```

**Step 3: Run Test with INFO Logging**

```bash
export LLAMINAR_LOG_LEVEL=INFO
mpirun -np 2 ./build_v2_e2e/tests/v2/v2_test_qwen2_e2e_correctness \
    --gtest_filter="YourTest" 2>&1 | grep -E "(layer|Snapshot)"
```

**Step 4: Analyze Divergence Points**

- Compare snapshots layer-by-layer
- Identify first layer where divergence occurs
- Check preceding operations (e.g., if ATTENTION_CONTEXT diverges, check Q_ROPE/K_ROPE first)
- Narrow down to specific operation within that layer

**Step 5: Add Fine-Grained Captures**

If needed, add intermediate captures within failing operation:

```cpp
// In GQAAttention.cpp
CAPTURE_SNAPSHOT_VIEW("layer{i}_ATTENTION_SCORES_PRE_SOFTMAX", scores, ...);
CAPTURE_SNAPSHOT_VIEW("layer{i}_ATTENTION_SCORES_POST_SOFTMAX", scores, ...);
CAPTURE_SNAPSHOT_VIEW("layer{i}_ATTENTION_MASK", mask, ...);
```

**Step 6: Root Cause**

Once you identify the diverging operation:

1. Review implementation for batched vs sequential differences
2. Check buffer indexing (especially with padding)
3. Verify mask generation/application
4. Check MPI communication boundaries (tensor-parallel attention)

### 9.7 Performance Considerations

**Memory Overhead:**

- Each snapshot stores `size * sizeof(float)` bytes
- Typical model: ~50 snapshots × 1MB each = ~50MB per forward pass
- Batched execution: multiply by batch size (for per-sequence comparison)
- **Recommendation**: Only enable for specific failing tests, not full test suites

**Execution Overhead:**

- `std::vector::assign()` copies data → O(N) per capture
- For large tensors (>10MB), adds ~1-5% runtime overhead
- **Mitigation**: E2ERelease build uses `-O3`, minimizes copy overhead

**Best Practices:**

1. **Use E2ERelease** for snapshot testing (optimized + snapshots)
2. **Limit capture scope** to failing layers/sequences when debugging
3. **Clear snapshots** between test iterations (`disableSnapshotCapture()`)
4. **Compare selectively** – only check snapshots for failing test cases

### 9.8 Future Enhancements

**Planned Features:**

1. **File Output**: Automatic writing of snapshots to `.npy` files for external analysis
2. **Metadata**: Capture timing, memory usage, device placement alongside tensors
3. **Selective Capture**: Environment variable to enable only specific stages (e.g., `LLAMINAR_SNAPSHOT_STAGES=ATTENTION,FFN`)
4. **Compression**: Optional lossy compression for large snapshots (e.g., FP16 storage)
5. **Diff Tool**: Standalone utility to compare snapshot directories and generate reports

**Example Future Usage:**

```bash
# Capture only attention stages
export LLAMINAR_SNAPSHOT_STAGES="ATTENTION"
./build_v2_e2e/tests/v2/v2_test_qwen2_e2e_correctness --gtest_filter="MyTest"

# Compare snapshot directories
./tools/compare_snapshots.py /tmp/baseline /tmp/test --tolerance 1e-3
```

---

## 10. Quick Checklist for Future Agents

When you need to work on V2:

1. **Find the right layer:**
   - Tensors / quantization → `src/v2/tensors/`
   - Kernels (GEMM, attention) → `src/v2/kernels/`
   - Fused kernels (FusedGEMM) → `src/v2/kernels/cpu/fused/`
   - Operations (Ops layer) → `src/v2/pipelines/ops/`
   - MPI + attention routing → `src/v2/pipelines/attention/`
   - Pipeline base class → `src/v2/pipelines/PipelineBase.{h,cpp}`
   - Model pipelines → `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

2. **Understand the operation hierarchy:**
   - **Ops** (`src/v2/pipelines/ops/`) – Low-level, stateless operations (RMSNormOp, GemmOp, etc.)
   - **PipelineBase composite methods** – High-level wrappers (rms_norm, project, swiglu, etc.)
   - **Fused kernels** – Multi-projection optimizations (FusedGEMM for Q/K/V, gate/up)
   - **Child pipelines** – Model-specific orchestration using PipelineBase methods

3. **Use the composite operations:**
   - Child pipelines (Qwen2Pipeline) should use PipelineBase methods (`rms_norm()`, `project()`, etc.)
   - These handle device placement, MPI context, snapshot capture, and validation
   - Use `TRY_OP()` macro for clean chaining

4. **Use fused kernels where applicable:**
   - `FusedGEMM` for Q/K/V projections (3 GEMMs sharing activation quantization)
   - `FusedGEMM` for gate/up projections (2 GEMMs sharing activation quantization)
   - Fused kernels are created lazily on first use, stored in LayerWeights

5. **Keep MPI out of kernels:**
   - MPI must live in orchestrators + `MPIContext`, not in `CpuAttentionKernelT` or GEMM kernels.

6. **Add tests:**
   - For any new feature or significant refactor, add or adjust unit tests in `tests/v2/unit/` and run the relevant suites.

7. **Debug with snapshots:**
   - When encountering numerical divergence or parity failures, use E2ERelease build with `enableSnapshotCapture()`
   - Compare layer-by-layer to identify first diverging operation
   - Use `capture_snapshot()` in PipelineBase for manual captures
   - See Section 9 for complete snapshot system documentation

8. **Document decisions:**
   - If you change architectural boundaries (e.g. how operations are layered, how fused kernels work), update this file and any relevant `.md` plans under `.github/instructions/` and `docs/`.

This architecture is intentionally modular: small, focused abstractions connected by narrow interfaces. If you keep those boundaries sharp, V2 remains easy to extend and reason about—even with advanced attention types and heterogeneous backends.