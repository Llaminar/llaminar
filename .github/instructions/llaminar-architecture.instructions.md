# Llaminar LLM Inference Engine - Architecture Documentation

*Last Updated: October 20, 2025*

## Overview

Llaminar is a high-performance, MPI-first LLM inference engine focused on low‑latency decode and scalable prefill. The architecture is built on a **multi-architecture pipeline abstraction** with pluggable model-family adapters, **refactored Template Method prefill providers** (58-65% code reduction), **unified backend-agnostic attention kernel** (86% code reduction), **MPI-aware provider abstractions** for weights and KV cache, and comprehensive observability.

### Recent Milestones (October 2025)

🎉 **MPI-Aware Provider Pattern** ✨ *NEW OCTOBER 11, 2025*
- **ModelWeightsProvider**: Type-safe structured access to model weights with MPI slicing metadata
- **KVCacheProvider**: Clean interface for KV cache population during prefill and consumption during decode
- **Separation of Concerns**: Loading → Verification → Serving as distinct responsibilities
- **Rank-Aware Access**: Automatic handling of column-sliced weights (Q/K/V, FFN) vs replicated weights (embeddings, norms)
- **Testing Support**: Built-in metadata queries for validation (isWeightSliced, getLocalSliceInfo)

🎉 **Prefill Provider Refactoring Complete** - Template Method pattern achieves:
- **69% code reduction** in OpenBLAS provider (865 → 280 lines)
- **70% code reduction** in COSMA provider (758 → 260 lines)
- **1,083 net lines eliminated** via shared base implementation (680 lines)
- **100% PyTorch parity maintained** (387/387 tests passing)
- **Zero performance regression** - identical benchmark timing
- **Production deployed** - old implementations removed, refactored code now standard

🎉 **Unified Backend-Agnostic Attention** - MPIAttentionKernel consolidation achieves:
- **86% code reduction** (185 lines of duplicated COSMA logic eliminated)
- **100% PyTorch parity** (387/387 tests passing with e-05 to e-06 precision)
- **Runtime backend injection** via `setCosmaManager()` for COSMA vs OpenBLAS switching
- **Systematic transpose fixes** across all weight projections (attention, FFN, LM_HEAD)

🎉 **OpenBLAS BF16 Emulation Verified** ✨ *NEW OCTOBER 20, 2025*
- **Investigation Complete**: OpenBLAS v0.3.26 `cblas_sbgemm` verified working on Cascade Lake CPUs
- **Original Bug Report**: January 2025 report claimed NaN outputs - **NOT REPRODUCED** in October testing
- **Comprehensive Testing**: All matrix sizes pass (2×2, 64×64, 64×896×896) with no NaN or numerical errors
- **CPU Feature Check Removed**: Defensive fallback to FP32 eliminated after verification
- **Production Status**: OpenBLAS BF16 emulation now trusted on all CPUs (software emulation works)
- **Test Suite**: 4/4 BF16 tests passing, end-to-end inference successful (1.22 tok/s decode)
- See: `changelog/2025-10-20-openblas-bf16-bug-investigation.md`, `changelog/2025-10-20-openblas-bf16-cpu-check-removed.md`

🎉 **Intel MKL BF16 Backend Integration** ✨ *OCTOBER 20, 2025*
- **Production-Quality BF16 GEMM**: Intel MKL `cblas_sbgemm` as default BF16 backend when `HAVE_MKL` defined
- **Performance Optimization**: Hardware acceleration on Ice Lake+ (AVX512_BF16), graceful software fallback otherwise
- **100% PyTorch Parity**: 387/387 tests passing with < 1e-4 relative L2 error
- **Graceful Fallback Chain**: MKL → OpenBLAS (verified working) → FP32 expansion with automatic degradation
- **2× Memory Savings**: BF16 storage vs FP32 with same exponent range (bfloat16 format)
- **Build Configuration**: Optional `-DUSE_MKL=ON` flag, links against Intel oneAPI MKL
- See: `src/backends/MKLBackend.{h,cpp}`, `src/AdaptiveMatmul.{h,cpp}`, `changelog/2025-10-20-mkl-default-backend.md`

### Core Architecture Pillars

1. **Abstract Pipeline System** ✨
   - **Multi-Architecture Support**: Factory pattern for Qwen, LLaMA, and future model families
   - **Clean Abstraction**: `AbstractPipeline` interface with `prefill()` / `decode()` lifecycle
   - **Concrete Implementations**: `QwenPipeline` (production), `LlamaPipeline` (prototype)
   - **Adapters**: `QwenPipelineAdapter`, `LlamaPipelineAdapter` implement standard interface
   - **Factory Registration**: Automatic model-family selection via `PipelineFactory`

2. **MPI-Aware Provider Pattern** ✨ *NEW OCTOBER 11, 2025*
   - **ModelWeightsProvider**: Structured, type-safe access to model weights with rank-aware slicing metadata
   - **KVCacheProvider**: Clean interface for KV cache lifecycle (populate during prefill, consume during decode)
   - **Separation of Concerns**: Loading, verification, and serving as distinct responsibilities
   - **Testing Support**: Built-in metadata queries for validation frameworks
   - **Backward Compatibility**: Raw ModelWeights struct still accessible for legacy code
   - See: `src/model_weights_provider.h`, `src/kv_cache_provider.h`

3. **Weight Contract System** ✨
   - **Load-Time Validation**: Declarative contracts validate GGUF weight dimensions/orientations
   - **Contract-Driven Loading** ✨ *NEW OCTOBER 12, 2025*: Automatic MPI-aware weight loading
   - **150 Lines Eliminated**: One-line `contract.load()` replaces manual MPI slicing logic
   - **Automatic Dimension Handling**: GGUF ↔ PyTorch conversion, transpose detection, slice strategy
   - **Zero Bugs**: Fixes K/V weight rank-specific errors (12.3 max error on rank 1)
   - **Fail Fast**: Clear error messages before inference if format mismatches
   - **Simplified Kernels**: No runtime shape detection needed
   - **Canonical Format**: Single source of truth for `[out_features, in_features]` convention
   - **Test Consistency**: Ensures synthetic test data matches production GGUF format

4. **Prefill Provider Abstraction** ✨ *REFACTORED OCTOBER 2025*
   - **Strategy + Template Method**: Swappable backends with shared execution flow (680 lines base)
   - **58-65% Code Reduction**: Per-provider savings via PrefillProviderBaseImpl
   - **Built-in Snapshot Capture**: Base class provides 387 parity testing points for all providers
   - **Runtime Selection**: `PrefillProviderFactory` chooses optimal provider based on sequence length
   - **Isolated Testing**: Each provider testable in isolation with unified metrics
   - **Clean Separation**: Pipeline orchestrates, providers execute (3 virtual methods each)
   - **Production Deployed**: Old implementations removed, refactored code now standard

5. **Unified Attention Kernel** ✨ *COMPLETED OCTOBER 2025*
   - **Backend-Agnostic**: Single `MPIAttentionKernel` implementation for both OpenBLAS and COSMA
   - **86% Code Reduction**: Eliminated 185 lines of duplicated COSMA attention logic
   - **100% PyTorch Parity**: 387/387 tests passing with micro-precision accuracy (e-05 to e-06 range)
   - **Runtime Injection**: `setCosmaManager()` switches between local and distributed matmul
   - **Systematic Transpose Handling**: All weights correctly handled via `transposeW` / `transposed_b` flags
   - **Complete Observability**: Unified snapshot capture across both backends
   - See: `src/kernels/MPIAttentionKernel.{h,cpp}`, `COSMA_PYTORCH_PARITY_STATUS.md`

6. **Tensor Sharding**
   - Current: 1D column partition for linear projections
   - MPI-aware: ModelWeightsProvider exposes slicing metadata per weight type
   - Planned: Hybrid 1D→2D sharding for multi‑node scaling

7. **Centralized Environment Snapshot** ✨
   - `debugEnv()`: Structured, typed access to all configuration flags
   - Eliminates repeated `getenv()` calls in hot loops
   - Single source of truth for tuning parameters

8. **Comprehensive Observability**
   - Structured perf counters and stage timers
   - Provider-integrated snapshot capture for parity testing (387 consistent points)
   - COSMA tile validation and distributed GEMM diagnostics
   - Prefill diagnostics module for baseline comparison

## Core Design Principles

1. **MPI-First Architecture**: All compute kernels derive from `MPIKernelBase`
2. **Multi-Architecture Pipeline**: Explicit model-family adapters (Qwen, LLaMA) behind unified interface
3. **Strategy Pattern for Backends**: Prefill providers encapsulate execution logic with runtime selection
4. **Stage-Driven Execution**: Semantic transformer stages (prefill/decode) replace generic graph scheduling
5. **Data Locality First**: Column partition weight shards + fused prefill minimize communication
6. **Graceful Degradation**: Provider factory falls back to OpenBLAS with diagnostics
7. **Deterministic Debugging**: Environment snapshot + opt‑in validation for surgical diagnostics
8. **Clean Abstraction Boundaries**: Pipeline orchestrates, kernels compute, providers execute

## Recent Architectural Improvements (October 2025)

### Unified Backend-Agnostic Attention Kernel ✨ *COMPLETED OCTOBER 10, 2025*

**Achievement**: 100% PyTorch parity (387/387 tests passing) with unified OpenBLAS/COSMA implementation.

**Motivation**: The COSMA and OpenBLAS prefill paths had duplicated attention logic (~185 lines) with inconsistent weight transpose handling, causing 99.3% PyTorch parity divergence.

**Solution**: Refactored `MPIAttentionKernel` to be backend-agnostic with runtime strategy injection:
- ✅ Eliminated code duplication (86% reduction in COSMA-specific logic)
- ✅ Systematic transpose fixes (attention, FFN, LM_HEAD projections)
- ✅ Perfect numerical parity (all errors in e-05 to e-06 range)
- ✅ Complete observability (387 snapshot points captured)

**Impact**: Both backends now use identical attention algorithm with different matmul primitives. See "COSMA Prefill Manager Refactoring" below for detailed implementation.

### Prefill Provider Refactoring ✨ *COMPLETED OCTOBER 10, 2025*

**Status**: ✅ **PRODUCTION DEPLOYED** - Template Method pattern with 58-65% code reduction per provider

**Motivation**: The original `QwenPipeline` contained monolithic prefill logic with scattered backend selection, making it difficult to:
- Test COSMA vs OpenBLAS execution paths in isolation
- Capture consistent snapshots across backends for parity testing
- Switch backends without modifying pipeline code
- Add new execution backends (GPU) without tangled dependencies

**Solution**: Extracted prefill execution into **Strategy + Template Method pattern** with pluggable providers:

#### Before (Monolithic Pipeline)
```cpp
class QwenPipeline {
    bool prefill(...) {
        // 500+ lines of prefill logic
        if (use_cosma) {
            // COSMA-specific execution (185 lines duplicated)
            executePrefillAttentionCosma(...);
        } else {
            // OpenBLAS-specific execution  
            executeAttentionLocal(...);
        }
        // Ad-hoc snapshot capture scattered throughout
    }
};
```

**Problems**:
- ❌ Backend logic tangled with pipeline orchestration
- ❌ Hard to test backends in isolation (need full pipeline)
- ❌ Snapshot capture inconsistent between backends
- ❌ Adding GPU backend requires pipeline modifications
- ❌ No unified metrics across backends
- ❌ 1,623 lines of duplicated execution scaffolding

#### After (Provider Abstraction with Template Method)
```cpp
// Shared base implementation (680 lines - execution flow + snapshot capture)
class PrefillProviderBaseImpl : public PrefillProvider {
public:
    // Template method - defines execution flow
    bool execute(tokens, weights, output, ctx, metrics) final;
    
protected:
    // Virtual methods for backend-specific operations
    virtual bool executeEmbedding(...) = 0;
    virtual bool executeLinearProjection(...) = 0;
    virtual bool executeAttentionBlock(...) = 0;
    
    // Shared implementation
    bool executeTransformerLayer(...);
    bool executeFfnBlock(...);
    void captureSnapshot(...);  // Consistent across all providers
};

// Concrete implementations (only 3 virtual methods!)
class OpenBLASPrefillProvider : public PrefillProviderBaseImpl {
    // 280 lines total (69% reduction from 865 lines)
    bool executeEmbedding(...) override;
    bool executeLinearProjection(...) override;
    bool executeAttentionBlock(...) override;
};

class COSMAPrefillProvider : public PrefillProviderBaseImpl {
    // 260 lines total (70% reduction from 758 lines)
    bool executeEmbedding(...) override;
    bool executeLinearProjection(...) override;
    bool executeAttentionBlock(...) override;
};

// Factory selection (unchanged)
auto provider = PrefillProviderFactory::create(config, mpi_ctx, seq_len);
bool success = provider->execute(tokens, weights, output, ctx, metrics);
```

**Code Reduction Metrics**:
- **OpenBLAS Provider**: 865 → 280 lines (69% reduction, 585 lines eliminated)
- **COSMA Provider**: 758 → 260 lines (70% reduction, 498 lines eliminated)
- **Total Savings**: 1,623 lines removed, 540 lines added = **1,083 net lines eliminated**
- **Shared Base**: 680 lines (single source of truth for execution flow)

**Benefits**:
- ✅ **Separation of Concerns**: Pipeline orchestrates, providers execute
- ✅ **Template Method Pattern**: Base class defines flow, derived override 3 methods
- ✅ **Isolated Testing**: Each provider testable independently with mocked weights
- ✅ **Consistent Snapshots**: Base class provides 387 capture points for all providers
- ✅ **Runtime Selection**: Factory chooses optimal provider based on workload
- ✅ **Extensible**: GPU provider needs only 3 virtual methods (~200 lines)
- ✅ **Unified Metrics**: `PrefillMetrics` struct tracks timing/FLOPS/snapshots consistently
- ✅ **Parity Testing**: Both providers capture at identical stages for A/B comparison
- ✅ **Zero Performance Regression**: 100% PyTorch parity maintained (387/387 tests passing)

#### Migration Impact

**Files Created**:
- **New**: `src/prefill_provider.{h,cpp}` - Base abstraction and factory
- **New**: `src/prefill_provider_base_impl.{h,cpp}` - Template Method base (680 lines)
- **New**: `src/openblas_prefill_provider.{h,cpp}` - Baseline CPU provider (280 lines)
- **New**: `src/cosma_prefill_provider.{h,cpp}` - Distributed COSMA provider (260 lines)
- **New**: `src/cublas_prefill_provider.{h,cpp}` - GPU stub (placeholder)
- **New**: `src/rocblas_prefill_provider.{h,cpp}` - AMD GPU stub (placeholder)

**Files Removed**:
- **Deleted**: Old monolithic implementations (1,623 lines removed)
- **Eliminated**: Duplicated execution scaffolding, snapshot capture, metrics

**Modified**:
- `src/qwen_pipeline.cpp` - Now delegates to provider factory
- `CMakeLists.txt` - Updated source file list

**Testing Coverage**:
- ✅ `ParityFrameworkTest`: 387/387 tests passing (100% PyTorch parity)
- ✅ All smoke tests passing (14 core tests in 1.16s)
- ✅ All unit tests passing (60+ tests in 2m30s)
- ✅ Integration tests passing

**No Breaking Changes**:
- External API unchanged (`AbstractPipeline::prefill()` signature preserved)
- Environment variables honored (`ADAPTIVE_DISABLE_COSMA`, etc.)
- Backend selection logic preserved (sequence length thresholds)
- Performance characteristics identical (0ms difference in benchmarks)

#### Future Extensions Enabled

This refactoring makes the following additions straightforward:

1. **GPU Provider** (stubs already created):
   ```cpp
   class CuBLASPrefillProvider : public PrefillProviderBaseImpl {
       // Only need to implement 3 virtual methods (~200 lines)
       bool executeEmbedding(...) override { /* cuBLAS */ }
       bool executeLinearProjection(...) override { /* cuBLAS GEMM */ }
       bool executeAttentionBlock(...) override { /* cuBLAS attention */ }
   };
   ```
   - No pipeline changes needed
   - Factory adds GPU detection logic
   - Inherits all 387 snapshot points automatically
   - Inherits all metrics/timing infrastructure

2. **Decode Provider** (future):
   ```cpp
   class DecodeProviderBaseImpl {
       virtual bool execute(token, weights, output, ctx, metrics) = 0;
   };
   ```
   - Mirror prefill pattern for decode phase
   - Enables latency-optimized GPU decode

3. **Fused Kernels**:
   - Providers can use specialized fused ops (FlashAttention, etc.)
   - Interface unchanged, implementation swapped in virtual methods

4. **Multi-Model Support**:
   - Factory can select provider based on model architecture
   - Different strategies for different model families

### Weight Contract System ✨ *UPDATED OCTOBER 12, 2025*

**Motivation**: Before weight contracts, we had runtime shape detection scattered throughout kernels and manual MPI slicing code duplicated in every pipeline:
- Kernels had to guess weight orientation at runtime (`[d_model, heads]` vs `[heads, d_model]`)
- Test fixtures created synthetic weights in inconsistent formats
- Silent shape mismatches led to subtle numerical bugs
- No single source of truth for GGUF format expectations
- **~150 lines of manual MPI slicing per pipeline** (offset calculations, tensor creation, error-prone)
- **Rank-specific bugs**: K/V weights had 12.3 max error on rank 1 due to slicing mistakes

**Problem Example** (MPIAttentionKernel before contracts):
```cpp
// Runtime guessing - which dimension is heads?
const int wq_cols = wq_global->shape()[1];
const bool weights_are_sharded = (wq_cols == local_head_dim);
// Complex logic to handle both orientations...
```

**Problem Example** (qwen_pipeline before contract-driven loading):
```cpp
// Manual MPI slicing - repeated for Q, K, V, O, FFN_gate, FFN_up, FFN_down
const int q_row_offset = mpi_rank * q_heads_per_rank * config.head_dim;
const int q_row_count = q_heads_per_rank * config.head_dim;
auto q_tensor = TensorFactory::create_simple({q_row_count, config.d_model});
bool success = loader.loadTensorRowShard(q_name, q_row_offset, q_row_count, ...);
// ~150 lines of duplicate code per pipeline!
```

**Solution**: Declarative contracts validated at model load time + automatic contract-driven loading:

#### Architecture

**Core Components**:
1. **`WeightShapeContract`**: Specification for single tensor with symbolic dimensions + MPI slicing metadata
2. **`ModelWeightContracts`**: Architecture-specific collection (global + per-layer)
3. **`getQwenWeightContracts()`**: Canonical GGUF format for Qwen/Qwen2 with slicing strategies
4. **`QwenModelWeights::validate()`**: Called during `loadWeights()` - fails fast
5. **`mpi_slicing::load_with_contract()`**: ✨ **NEW** Automatic MPI-aware weight loading
6. **`WeightShapeContract::load()`**: ✨ **NEW** High-level contract-driven loading interface

**Contract Definition** (with MPI slicing metadata):
```cpp
WeightShapeContract("blk.{layer}.attn_k.weight", 
    {"n_head_kv*head_dim", "d_model"},              // PyTorch dimensions
    "Key projection (row-sliced by KV heads)",      // Human description
    false,                                          // Not optional
    WeightSliceType::ROW_SLICED,                   // MPI slicing strategy
    "n_head_kv",                                   // Slice parameter (which dimension)
    {"d_model", "n_head_kv*head_dim"},             // GGUF dimensions (may differ!)
    true)                                          // Needs transpose after loading
```

**Contract-Driven Loading** (one line replaces manual slicing):
```cpp
// Before: ~150 lines of manual offset calculations
// After: One line per weight
auto wk = contracts.layer_weights[IDX_K].load(
    loader, config, mpi_rank, mpi_size, layer);
// Automatically handles: dimension evaluation, transpose detection, 
// MPI slicing, loading, transposition, and validation!
```

**Validation Flow**:
```cpp
// In QwenPipelineAdapter::loadWeights()
auto weights = std::make_unique<QwenModelWeights>();
weights->inner = loadModelWeights_impl_bridge(loader, cfg_);

// CRITICAL: Validate before returning to pipeline
try {
    weights->validate(cfg_.getLayerConfig());
    LOG_INFO("✓ All weights validated against canonical GGUF format");
} catch (const std::exception& e) {
    LOG_ERROR("Weight validation failed: " << e.what());
    throw;  // Fail fast with clear error
}
```

#### Canonical GGUF Format

**Universal Rule**: ALL weights are `[out_features, in_features]` (PyTorch `nn.Linear` convention)

```cpp
// Attention
attn_q/k/v.weight:  [n_head*head_dim, d_model]      // [896, 896], [128, 896]
attn_output.weight: [d_model, n_head*head_dim]       // [896, 896]

// FFN
ffn_gate/up.weight: [d_ff, d_model]                  // [4864, 896]
ffn_down.weight:    [d_model, d_ff]                  // [896, 4864]

// Global
token_embedding:    [vocab_size, d_model]            // [151669, 896]
lm_head:            [vocab_size, d_model]            // [151669, 896]
```

#### Error Messages

**Before (silent failure)**:
```
// Kernel silently uses wrong orientation → subtle numerical drift
```

**After (clear validation error)**:
```
[ERROR] Weight contract validation failed for 'attn_k.weight' (layer 0):
  Description: Key projection (GGUF format: [out, in])
  Reason: Dimension 0 mismatch
  Expected shape: [128, 896] (from [n_head_kv*head_dim, d_model])
  Actual shape:   [896, 128]
```

Immediately shows:
- **Which weight** and **which layer**
- **Expected** vs **actual** shapes
- **Symbolic expression** for clarity
- **Human description** for context

#### Kernel Benefits

**Before (MPIAttentionKernel)**:
```cpp
// Runtime orientation detection
const int wq_cols = wq_global->shape()[1];
const bool weights_are_sharded = (wq_cols == local_head_dim);
// Branch on orientation...
```

**After**:
```cpp
// WEIGHT FORMAT CONTRACT:
// All weights guaranteed by QwenModelWeights to be canonical GGUF format:
// - wq, wk, wv: [out_features, in_features] = [n_head*head_dim, d_model]
// - wo: [in_features, out_features] = [d_model, n_head*head_dim]

const int wq_rows = wq_global->shape()[0];
const bool weights_are_sharded = (wq_rows == local_head_dim);  // Simple!
```

#### Test Fixture Requirements

All tests **must** match GGUF canonical format:

**Before** (test_abstract_pipeline_parity.cpp):
```cpp
// Inconsistent - didn't match GGUF!
w.wq.push_back(randTensor({cfg.d_model, cfg.n_head * cfg.head_dim}));  // WRONG
w.lm_head = randTensor({cfg.d_model, cfg.vocab_size});                 // WRONG
```

**After** (enforced by validation):
```cpp
// Matches GGUF canonical format
w.wq.push_back(randTensor({cfg.n_head * cfg.head_dim, cfg.d_model}));  // ✓
w.lm_head = randTensor({cfg.vocab_size, cfg.d_model});                 // ✓
```

#### Benefits

1. **✅ Fail Fast**: Errors at load time, not during inference
2. **✅ Clear Diagnostics**: Symbolic expressions pinpoint exact mismatch
3. **✅ Simplified Kernels**: No runtime shape detection overhead
4. **✅ Self-Documenting**: Contracts are executable specification
5. **✅ Test Consistency**: Synthetic data forced to match production
6. **✅ Multi-Architecture**: Easy to add `getLlamaWeightContracts()`, etc.

#### Files

- **Core**: `src/weight_contracts.h` (contract definitions)
- **Integration**: `src/qwen_pipeline_adapter.{h,cpp}` (validation in `loadWeights()`)
- **Reference**: `src/model_loader.h` (GGUF format documentation)
- **Documentation**: `docs/WEIGHT_CONTRACTS.md` (full specification)
- **Tests**: All test fixtures updated to match GGUF format

#### Future Extensions

```cpp
// Quantization validation
struct WeightShapeContract {
    QuantizationType expected_quant = QuantizationType::F32;
};

// Value range checks (corruption detection)
struct WeightShapeContract {
    std::optional<float> expected_min, expected_max;
};

// Additional architectures
inline ModelWeightContracts getLlamaWeightContracts() { ... }
inline ModelWeightContracts getGPTWeightContracts() { ... }
```

### Stage Contracts for MPIAttentionKernel ✨

**Motivation**: Complex attention pipelines with multiple transformation stages (Q/K/V projections, RoPE, GQA replication, attention, output projection) are prone to dimension mismatches and transpose bugs that manifest as cryptic segfaults or numerical errors.

**Solution**: Introduced **explicit stage contracts** that define PRE and POST conditions for each of the 5 internal stages in MPIAttentionKernel. Contracts validate tensor shapes, layouts, and semantics at runtime with clear error messages.

**Benefits**:
- ✅ Catches dimension/transpose bugs before execution (fail-fast)
- ✅ Clear error messages: "Expected [896, 128], got [128, 896] → transposed"
- ✅ Documents expected shapes in code (self-documenting architecture)
- ✅ Validates all 10 input tensors (input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache)
- ✅ Works in single-rank and multi-rank MPI contexts
- ✅ Handles dynamic dimensions (variable seq_len)

**Testing**: 100% pass rate on 7 smoke tests validating contract infrastructure (see `tests/test_attention_stage_contracts.cpp` and `tests/ATTENTION_STAGE_CONTRACTS_TESTS.md`).

**See Section 7**: Detailed explanation under "Attention Implementation → Stage Contracts"

### COSMA Prefill Manager Refactoring ✨ *OCTOBER 2025*

**Status**: ✅ **COMPLETED** - 100% PyTorch parity achieved (387/387 tests passing)

**Motivation**: The original COSMA prefill path had duplicated attention logic and weight transpose handling inconsistencies:
- ❌ COSMA and OpenBLAS paths implemented attention independently (~185 lines duplicated)
- ❌ Manual weight transpose in COSMA path vs implicit `CblasTrans` in OpenBLAS
- ❌ 99.3% PyTorch parity divergence due to weight orientation mismatch
- ❌ Missing FFN intermediate snapshots (72 missing: FFN_GATE, FFN_UP, FFN_SWIGLU)
- ❌ Difficult to maintain consistency between backends

**Strategic Decision**: Instead of maintaining separate attention implementations, **unify both paths** by making `MPIAttentionKernel` backend-agnostic.

**Solution**: Two-phase refactoring combining architectural unification with systematic transpose fixes:

**Solution**: Two-phase refactoring combining architectural unification with systematic transpose fixes:

#### Phase 1: Architectural Unification (MPIAttentionKernel Backend-Agnostic)

**Before (Duplicated Logic)**:
```cpp
// OpenBLAS path in QwenPipeline
bool executeAttentionLocal(...) {
    // ~100 lines of attention logic using cblas_sgemm
}

// COSMA path in COSMAPrefillProvider  
bool executePrefillAttentionCosma(...) {
    // ~185 lines of DUPLICATED attention logic using CosmaPrefillManager
    // Manual transpose, different API, inconsistent snapshot capture
}
```

**After (Unified via MPIAttentionKernel)**:
```cpp
// Both paths now use the SAME kernel
class MPIAttentionKernel {
public:
    // Backend-agnostic interface
    bool execute(inputs, outputs) override;
    
    // Inject backend at runtime
    void setCosmaManager(CosmaPrefillManager* mgr);
    
private:
    // Internal matmul routing
    bool matmul_with_bias(A, W, bias, out, ...) {
        if (cosma_manager_) {
            // COSMA distributed path
            return cosma_matmul_transpose(...);
        } else {
            // OpenBLAS local path  
            return cblas_sgemm_transpose(...);
        }
    }
};

// OpenBLAS provider
auto result = attention_kernel->execute(inputs, outputs);

// COSMA provider  
attention_kernel->setCosmaManager(cosma_manager_.get());
auto result = attention_kernel->execute(inputs, outputs);  // Same API!
```

**Benefits**:
- ✅ **86% Code Reduction**: Eliminated 185 lines of duplicated COSMA attention logic
- ✅ **Single Source of Truth**: Both backends use identical attention algorithm
- ✅ **Consistent Snapshots**: Capture points now identical across backends
- ✅ **Maintainability**: Bug fixes and optimizations apply to both paths
- ✅ **Type Safety**: Unified handling of weight dimensions and transpose flags

#### Phase 2: Systematic Weight Transpose Fixes

**Root Cause Discovery**: 
- GGUF stores ALL weights as `[output_dim, input_dim]` (PyTorch `nn.Linear` convention)
- OpenBLAS uses `CblasTrans` flag to implicitly transpose during matmul
- COSMA has NO transpose support in API - requires explicit parameter

**Three-Phase Systematic Fix**:

**Phase 2.1: Attention Q/K/V Projections** (2/387 → 12/387 tests passing)
```cpp
// COSMA MPIAttentionKernel matmul_with_bias()
WeightDescriptor weight_desc{
    weight_data,
    original_rows,     // Keep [N, K] format
    original_cols,
    original_cols      // row_stride = K (no transpose)
};
cosma_manager_->matmul(..., weight_desc, ..., 
    /*transposeW=*/true);  // ← CRITICAL FIX
```

**Phase 2.2: FFN Projections** (12/387 → 314/387 tests passing, +2517% improvement!)
```cpp
// COSMAPrefillProvider FFN operations
adaptiveMatMul(...,
    weights.w_gate[layer_idx]->data(),  // [d_ff, d_model]
    ...,
    /*transposed_b=*/true);  // ← Fixed: was false

adaptiveMatMul(...,
    weights.w_up[layer_idx]->data(),    // [d_ff, d_model]
    ...,
    /*transposed_b=*/true);  // ← Fixed: was false

adaptiveMatMul(...,
    weights.w_down[layer_idx]->data(),  // [d_model, d_ff]
    ...,
    /*transposed_b=*/true);  // ← Fixed: was false
```

**Phase 2.3: LM_HEAD Projection** (314/387 → 315/387 tests passing)
```cpp
// Final vocabulary projection
adaptiveMatMul(...,
    weights.lm_head->data(),  // [vocab_size, d_model]  
    ...,
    /*transposed_b=*/true);  // ← Fixed: was false
```

**Phase 2.4: FFN Intermediate Snapshots** (315/387 → 387/387 tests passing - **100% PARITY!**)
```cpp
// COSMAPrefillProvider - added missing snapshot captures
captureSnapshot(PipelineStage::FFN_GATE, layer_idx, gate_out->data(), seq_len, d_ff);
captureSnapshot(PipelineStage::FFN_UP, layer_idx, up_out->data(), seq_len, d_ff);
captureSnapshot(PipelineStage::FFN_SWIGLU, layer_idx, swiglu_out->data(), seq_len, d_ff);
```

#### Final Results: Perfect PyTorch Parity

```
[COSMA_PYTORCH] Summary:
  ✓ Passed:  387/387
  ✗ Failed:  0/387  
  ? Missing: 0/387

[  PASSED  ] ParityFramework.COSMAPrefillVsPyTorch
```

**Sample Precision** (all operations in e-05 to e-06 range):
```
Q_PROJECTION_layer0:   max_abs=1.5e-05  rel_l2=7.3e-07  ✓ PASS
K_PROJECTION_layer0:   max_abs=3.0e-05  rel_l2=9.6e-07  ✓ PASS  
V_PROJECTION_layer0:   max_abs=8.2e-08  rel_l2=2.3e-06  ✓ PASS
FFN_GATE_layer0:       max_abs=2.4e-05  rel_l2=1.3e-06  ✓ PASS
FFN_UP_layer0:         max_abs=2.1e-05  rel_l2=1.8e-06  ✓ PASS
FFN_SWIGLU_layer0:     max_abs=4.6e-05  rel_l2=2.6e-06  ✓ PASS
LM_HEAD:               max_abs=7.0e-05  rel_l2=2.4e-06  ✓ PASS
```

#### Benefits of Unified Architecture

1. **✅ 100% PyTorch Parity**: All 387 operations match reference implementation
2. **✅ Code Reduction**: Eliminated 185 lines of duplicated attention logic  
3. **✅ Systematic Correctness**: All weight transposes now handled consistently
4. **✅ Complete Observability**: All intermediate activations captured for validation
5. **✅ Maintainability**: Single attention implementation for both backends
6. **✅ Type Safety**: Explicit transpose handling at COSMA boundaries
7. **✅ Future-Proof**: GPU backend can reuse same MPIAttentionKernel infrastructure

#### Trade-offs Acknowledged

**Complexity**: COSMA path requires explicit transpose parameters
- **Rationale**: COSMA API has no implicit transpose support (unlike BLAS)
- **Mitigation**: Clear documentation, weight contract validation, comprehensive tests

**Complexity**: COSMA path requires explicit transpose parameters
- **Rationale**: COSMA API has no implicit transpose support (unlike BLAS)
- **Mitigation**: Clear documentation, weight contract validation, comprehensive tests

**Performance**: COSMA path still ~134x slower than OpenBLAS for single-token decode
- **Rationale**: Communication overhead dominates small operations
- **Mitigation**: Adaptive backend selection uses OpenBLAS for decode, COSMA for prefill
- **Threshold**: COSMA becomes competitive at ≥8K tokens, superior at ≥64K tokens

#### Migration Status

- ✅ **Created**: Backend-agnostic MPIAttentionKernel with COSMA injection
- ✅ **Unified**: Both OpenBLAS and COSMA paths use same attention implementation
- ✅ **Fixed**: All weight transpose issues (attention, FFN, LM_HEAD)
- ✅ **Validated**: 387/387 tests passing (100% PyTorch parity)
- ✅ **Documented**: Comprehensive Doxygen comments + architecture updates
- ✅ **Tested**: Full parity test suite, smoke tests, integration tests all passing

**Key Files**:
- `src/kernels/MPIAttentionKernel.{h,cpp}` - Unified backend-agnostic attention kernel
- `src/cosma_prefill_provider.{h,cpp}` - COSMA provider using unified kernel
- `src/openblas_prefill_provider.{h,cpp}` - OpenBLAS provider using unified kernel  
- `tests/test_parity_framework.cpp` - 387-point validation test

**Legacy Code Removed**:
- ❌ `executePrefillAttentionCosma()` - 185 lines of duplicated COSMA attention (deleted)
- ❌ Manual weight transpose logic - replaced with systematic `transposeW=true` / `transposed_b=true`

**Documentation Files**:
- `COSMA_PYTORCH_PARITY_STATUS.md` - Parity achievement summary
- `.github/copilot-instructions.md` - Updated with unified architecture

### MPI-Aware Provider Pattern ✨ *NEW OCTOBER 11, 2025*

**Status**: ✅ **PRODUCTION** - Clean separation of model weight serving and KV cache lifecycle management

**Motivation**: The original architecture had several cross-cutting concerns that made testing and validation difficult:
- ❌ Direct access to raw `ModelWeights` struct required kernels to understand MPI slicing
- ❌ No metadata about which weights were sliced vs replicated (verification hard)
- ❌ KV cache population during prefill tightly coupled to pipeline internals
- ❌ Validation code had to duplicate slicing logic to verify correctness

**Solution**: Introduced two provider abstractions with clean interfaces and separation of concerns:

#### 1. ModelWeightsProvider - Structured Weight Access

**Files**: `src/model_weights_provider.{h,cpp}` (320 lines)

**Design Philosophy**:
- **Provider OWNS** weights (via `unique_ptr<ModelWeights>`)
- **Provider SERVES** weights (via const `shared_ptr` getters)
- **Provider DOCUMENTS** slicing (via metadata methods)
- **Provider does NOT** load or verify weights (separate concerns)

**Key Features**:
- ✅ **Type-Safe Getters**: Named methods for each weight category (`getQueryWeight()`, `getKeyWeight()`, etc.)
- ✅ **MPI Metadata**: Query methods to determine slicing behavior per weight type
- ✅ **Bounds Checking**: Layer index validation with clear error messages
- ✅ **Backward Compatibility**: `rawWeights()` method for legacy code
- ✅ **Testing Support**: `isWeightSliced()` and `getLocalSliceInfo()` for verification frameworks

**Architecture**:
```cpp
class QwenModelWeightsProvider {
public:
    // Constructor takes ownership of weights
    QwenModelWeightsProvider(
        std::unique_ptr<QwenPipeline::ModelWeights> weights,
        const MPIContext& mpi_ctx,
        const TransformerLayerConfig& config
    );

    // === GLOBAL WEIGHTS (Replicated) ===
    std::shared_ptr<TensorBase> getTokenEmbedding() const;
    std::shared_ptr<TensorBase> getOutputNormWeight() const;
    std::shared_ptr<TensorBase> getLMHead() const;

    // === ATTENTION WEIGHTS (Q/K/V column-sliced, O replicated) ===
    std::shared_ptr<TensorBase> getQueryWeight(int layer_idx) const;
    std::shared_ptr<TensorBase> getKeyWeight(int layer_idx) const;
    std::shared_ptr<TensorBase> getValueWeight(int layer_idx) const;
    std::shared_ptr<TensorBase> getOutputWeight(int layer_idx) const;

    // === ATTENTION BIASES (Q/K/V column-sliced) ===
    std::shared_ptr<TensorBase> getQueryBias(int layer_idx) const;
    std::shared_ptr<TensorBase> getKeyBias(int layer_idx) const;
    std::shared_ptr<TensorBase> getValueBias(int layer_idx) const;

    // === FFN WEIGHTS (Gate/Up column-sliced, Down row-sliced) ===
    std::shared_ptr<TensorBase> getFfnGateWeight(int layer_idx) const;
    std::shared_ptr<TensorBase> getFfnUpWeight(int layer_idx) const;
    std::shared_ptr<TensorBase> getFfnDownWeight(int layer_idx) const;

    // === NORMALIZATION WEIGHTS (Replicated) ===
    std::shared_ptr<TensorBase> getAttentionNormWeight(int layer_idx) const;
    std::shared_ptr<TensorBase> getFfnNormWeight(int layer_idx) const;

    // === MPI METADATA QUERIES ===
    bool isWeightSliced(const std::string& weight_type) const;
    std::pair<int, int> getLocalSliceInfo(const std::string& weight_type) const;
    int getRank() const { return rank_; }
    int getWorldSize() const { return world_size_; }

    // === BACKWARD COMPATIBILITY ===
    const QwenPipeline::ModelWeights& rawWeights() const;

private:
    std::unique_ptr<QwenPipeline::ModelWeights> weights_;  // Owned
    int rank_;
    int world_size_;
    const TransformerLayerConfig& config_;
};
```

**Weight Slicing Behavior** (Automatic via ModelLoader):

The `ModelLoader` automatically slices certain weights across MPI ranks during GGUF loading:

**Column-Sliced Weights** (each rank gets subset of output features):
- **W_Q** (`attn_q.weight`): Sliced by query heads
  - Full: `[n_head * head_dim, d_model]` = `[896, 896]`
  - Local: `[local_n_head * head_dim, d_model]` = `[448, 896]` (rank 0 gets heads 0-6, rank 1 gets heads 7-13)
- **W_K, W_V** (`attn_k.weight`, `attn_v.weight`): Sliced by KV heads  
  - Full: `[n_head_kv * head_dim, d_model]` = `[128, 896]`
  - Local: `[local_n_head_kv * head_dim, d_model]` = `[64, 896]` (rank 0 gets head 0, rank 1 gets head 1)
- **W_GATE, W_UP** (`ffn_gate.weight`, `ffn_up.weight`): Sliced by FFN hidden dimension
  - Full: `[d_ff, d_model]` = `[4864, 896]`
  - Local: `[d_ff/2, d_model]` = `[2432, 896]`

**Row-Sliced Weights** (transposed column partitioning):
- **W_DOWN** (`ffn_down.weight`): Row-sliced for output gather
  - Full: `[d_model, d_ff]` = `[896, 4864]`
  - Local: `[d_model, d_ff/2]` = `[896, 2432]`

**Replicated Weights** (full copy on each rank):
- **Token Embedding** (`token_embd.weight`): `[vocab_size, d_model]` = `[151669, 896]`
- **W_O** (`attn_output.weight`): `[d_model, n_head * head_dim]` = `[896, 896]`
- **All Normalization Weights**: RMSNorm gammas
- **LM Head**: `[vocab_size, d_model]` = `[151669, 896]`

**Usage Example (Kernel Access)**:
```cpp
// In MPIAttentionKernel::execute()
auto wq = provider.getQueryWeight(layer_idx);
auto wk = provider.getKeyWeight(layer_idx);

// Query metadata for verification
if (provider.isWeightSliced("Q")) {
    auto [offset, count] = provider.getLocalSliceInfo("Q");
    LOG_DEBUG("Rank " << provider.getRank() 
              << " has Q heads " << offset << " to " << (offset + count - 1));
}

// Backward compatibility
const auto& raw = provider.rawWeights();
```

**Usage Example (Weight Verification)**:
```cpp
// In WeightVerifier class
QwenModelWeightsProvider provider(std::move(weights), mpi_ctx, config);

for (int layer = 0; layer < num_layers; ++layer) {
    auto pytorch_q = load_pytorch_weight("layer" + std::to_string(layer) + "_Q");
    auto llaminar_q = provider.getQueryWeight(layer);
    
    // Extract the slice this rank should have
    if (provider.isWeightSliced("Q")) {
        auto [offset, count] = provider.getLocalSliceInfo("Q");
        auto pytorch_q_slice = extract_rows(pytorch_q, offset, count);
        compare(pytorch_q_slice, llaminar_q);
    } else {
        compare(pytorch_q, llaminar_q);  // Full comparison
    }
}
```

#### 2. KVCacheProvider - Cache Lifecycle Management

**Files**: `src/kv_cache_provider.{h,cpp}` (192 lines)

**Design Philosophy**:
- **Interface-based**: Abstract base class + simple concrete implementation
- **Single Responsibility**: Only manages cache storage/retrieval
- **Pipeline Agnostic**: Works with any prefill provider
- **MPI Aware**: Each rank owns subset of KV heads (head parallelism)

**Key Features**:
- ✅ **Clean Interface**: 6 virtual methods (`getKCache()`, `setKCache()`, `getVCache()`, `setVCache()`, `reserve()`, `clear()`)
- ✅ **Prefill Integration**: Providers populate cache during execution
- ✅ **Decode Consumption**: Pipeline retrieves cache for iterative decode
- ✅ **Memory Management**: Optional `reserve()` for upfront allocation
- ✅ **Testing Support**: Clear() method for test isolation

**Architecture**:
```cpp
// Abstract interface
class KVCacheProvider {
public:
    virtual ~KVCacheProvider() = default;

    // Retrieval (pipeline uses these after prefill)
    virtual const std::vector<std::shared_ptr<TensorBase>>& getKCache() const = 0;
    virtual const std::vector<std::shared_ptr<TensorBase>>& getVCache() const = 0;

    // Population (provider uses these during execution)
    virtual void setKCache(int layer_idx, std::shared_ptr<TensorBase> k_cache) = 0;
    virtual void setVCache(int layer_idx, std::shared_ptr<TensorBase> v_cache) = 0;

    // Lifecycle
    virtual void reserve(int n_layers, int seq_len, int kv_head_dim) = 0;
    virtual void clear() = 0;
};

// Simple concrete implementation
class SimpleKVCacheProvider : public KVCacheProvider {
public:
    const std::vector<std::shared_ptr<TensorBase>>& getKCache() const override {
        return k_cache_;
    }
    
    const std::vector<std::shared_ptr<TensorBase>>& getVCache() const override {
        return v_cache_;
    }
    
    void setKCache(int layer_idx, std::shared_ptr<TensorBase> k_cache) override {
        if (layer_idx >= k_cache_.size()) {
            k_cache_.resize(layer_idx + 1);
        }
        k_cache_[layer_idx] = k_cache;
    }
    
    void setVCache(int layer_idx, std::shared_ptr<TensorBase> v_cache) override {
        if (layer_idx >= v_cache_.size()) {
            v_cache_.resize(layer_idx + 1);
        }
        v_cache_[layer_idx] = v_cache;
    }
    
    void reserve(int n_layers, int seq_len, int kv_head_dim) override {
        k_cache_.reserve(n_layers);
        v_cache_.reserve(n_layers);
    }
    
    void clear() override {
        k_cache_.clear();
        v_cache_.clear();
    }

private:
    std::vector<std::shared_ptr<TensorBase>> k_cache_;
    std::vector<std::shared_ptr<TensorBase>> v_cache_;
};
```

**Cache Layout Per Rank**:
```cpp
// Each rank owns a subset of KV heads (head parallelism)
// For n_head_kv=2, world_size=2:
//   Rank 0: KV head 0  (local_kv_head_dim = 1 * head_dim = 64)
//   Rank 1: KV head 1  (local_kv_head_dim = 1 * head_dim = 64)

K cache per layer: [seq_len, local_kv_head_dim]
V cache per layer: [seq_len, local_kv_head_dim]

// Example for prefill with seq_len=5, 24 layers:
k_cache_ = vector of 24 tensors, each shape [5, 64]
v_cache_ = vector of 24 tensors, each shape [5, 64]
```

**Usage Example (Prefill Provider)**:
```cpp
// In OpenBLASPrefillProvider::execute()
bool OpenBLASPrefillProvider::execute(
    const std::vector<int>& tokens,
    const IModelWeights& weights,
    std::shared_ptr<TensorBase>& output,
    StageContext& ctx,
    PrefillMetrics& metrics,
    KVCacheProvider* kv_cache_provider  // Optional KV cache output
) {
    // Reserve cache upfront
    if (kv_cache_provider) {
        int seq_len = tokens.size();
        int local_kv_head_dim = (n_head_kv_ / world_size_) * head_dim_;
        kv_cache_provider->reserve(n_layers_, seq_len, local_kv_head_dim);
    }

    for (int layer = 0; layer < n_layers_; ++layer) {
        // Execute attention...
        
        // Populate KV cache after attention
        if (kv_cache_provider) {
            kv_cache_provider->setKCache(layer, k_projected);  // [seq_len, local_kv_head_dim]
            kv_cache_provider->setVCache(layer, v_projected);
        }
    }
}
```

**Usage Example (Pipeline Prefill → Decode Transition)**:
```cpp
// In QwenPipeline::prefill()
SimpleKVCacheProvider cache_provider;
bool success = prefill_provider->execute(tokens, weights, output, ctx, metrics, &cache_provider);

if (success) {
    // Retrieve cache for decode phase
    const auto& k_caches = cache_provider.getKCache();
    const auto& v_caches = cache_provider.getVCache();
    
    for (int i = 0; i < n_layers_; ++i) {
        k_cache_[i] = k_caches[i];  // Store in pipeline state
        v_cache_[i] = v_caches[i];
    }
}

// In QwenPipeline::decode() - use cached K/V
for (int layer = 0; layer < n_layers_; ++layer) {
    auto k_cached = k_cache_[layer];  // [past_seq_len, local_kv_head_dim]
    auto v_cached = v_cache_[layer];
    
    // Append new token's K/V and perform attention...
}
```

#### Benefits of Provider Pattern

**1. Separation of Concerns**:
- ✅ **Loading** (`ModelLoader`): Reads GGUF, handles quantization, performs MPI slicing
- ✅ **Verification** (`WeightVerifier`): Compares against PyTorch reference using metadata
- ✅ **Serving** (`ModelWeightsProvider`): Provides type-safe access with metadata
- ✅ **Execution** (Kernels): Focus on computation, trust provider contracts

**2. Testing & Validation**:
```cpp
// Before: Verification had to duplicate slicing logic
// After: Query provider metadata
TEST(WeightVerification, QueryWeightsSlicedCorrectly) {
    QwenModelWeightsProvider provider(weights, mpi_ctx, config);
    
    EXPECT_TRUE(provider.isWeightSliced("Q"));
    EXPECT_FALSE(provider.isWeightSliced("O"));
    
    auto [offset, count] = provider.getLocalSliceInfo("Q");
    if (rank == 0) {
        EXPECT_EQ(offset, 0);
        EXPECT_EQ(count, 7);  // Heads 0-6
    } else {
        EXPECT_EQ(offset, 7);
        EXPECT_EQ(count, 7);  // Heads 7-13
    }
}
```

**3. Extensibility**:
```cpp
// Easy to add new provider implementations
class OptimizedKVCacheProvider : public KVCacheProvider {
    // Could use memory pools, GPU buffers, etc.
};

class FP8ModelWeightsProvider {
    // Could add FP8 quantization support
    // getKeyWeightFP8() returns quantized tensor
};
```

**4. Backward Compatibility**:
```cpp
// Legacy code still works via rawWeights()
const auto& raw = provider.rawWeights();
auto wq = raw.wq[layer_idx];  // Old direct access
```

#### Migration Impact

**Files Created**:
- **New**: `src/model_weights_provider.{h,cpp}` - Type-safe weight serving (320 lines)
- **New**: `src/kv_cache_provider.{h,cpp}` - KV cache lifecycle management (192 lines)
- **New**: `src/simple_kv_cache_provider.{h,cpp}` - Simple concrete implementation (included in header)

**Files Modified**:
- `src/qwen_pipeline.{h,cpp}` - Uses `ModelWeightsProvider` and `KVCacheProvider`
- `src/openblas_prefill_provider.{h,cpp}` - Populates KV cache via provider
- `src/cosma_prefill_provider.{h,cpp}` - Populates KV cache via provider
- `src/weight_verifier.{h,cpp}` - Uses provider metadata for slice extraction
- `CMakeLists.txt` - Updated source file list

**Testing Coverage**:
- ✅ `test_model_weights_provider.cpp` - Provider API correctness
- ✅ `test_kv_cache_provider.cpp` - Cache lifecycle validation
- ✅ `test_weight_verifier.cpp` - Verification with provider metadata
- ✅ Integration tests updated to use providers

**No Breaking Changes**:
- `rawWeights()` method maintains compatibility
- Existing direct access patterns still work
- Gradual migration path (providers optional initially)

## Architecture Components

### 1. Entry Point & Orchestration

**File**: `src/main.cpp` (219 lines)
- **Purpose**: Application entry point and execution orchestration
- **Key Features**:
  - MPI initialization with thread support
  - 7-stage execution pipeline
  - Exception handling and graceful shutdown
  - Performance measurement and reporting

**Execution Flow**:
1. **Initialization**: Parse CLI → build `LlaminarParams` (includes `--kv-stats`, verbosity, model path)
2. **Environment Setup**: Initialize logging + `debugEnv()` snapshot
3. **Topology Detection**: Detect NUMA, cores; optionally print system info
4. **Pipeline Registration**: Register Qwen and LLaMA pipeline creators with `PipelineFactory`
5. **Model Loading**: Load GGUF model → weight tensors → wrap in architecture-specific `IModelWeights`
6. **Pipeline Creation**: Use `PipelineFactory::create(model_config)` to instantiate appropriate pipeline
7. **Inference**: Execute prefill (prompt) then iterative decode until completion
8. **Summary**: Emit KV cache stats, performance counters, backend statistics

### 2. Command Line Interface

**Files**: `src/argument_parser.h/cpp`
- **Class**: `ArgumentParser`
- **Structure**: `LlaminarParams`
- **Features**:
  - POSIX-style argument parsing (`-v`, `--verbose`, etc.)
  - Multi-level verbosity (`-v`, `-vv`, `-vvv`)
  - Model file specification (`-m`, `--model`)
  - System configuration flags
  - Legacy COSMA parameters
  - Comprehensive help and version information

**Supported Arguments**:
```bash
# Model and Logging
-m, --model <file>     # GGUF model file path
-v/-vv/-vvv           # Verbosity levels (INFO/DEBUG/TRACE)

# System Configuration  
--print-topology      # Display system topology
--enable-hyperthreading # Use HT cores
--detect-gpus         # Enable GPU detection

# Performance
--profile             # Enable kernel profiling
--validate            # Enable result validation
--matrix-size <size>  # Matrix dimensions for benchmarks
--repeat <num>        # Benchmark iterations
```

### 3. Logging System

**Files**: `src/logger.h`, `src/log_level.h`
- **Pattern**: Header-only singleton with macros
- **Levels**: ERROR(0), WARN(1), INFO(2), DEBUG(3), TRACE(4)
- **Features**:
  - Timestamped output with millisecond precision
  - File and line number tracking
  - Environment variable configuration
  - Thread-safe singleton pattern
  - Convenient macros (`LOG_INFO`, `LOG_DEBUG`, etc.)

**Usage Example**:
```cpp
LOG_INFO("Model loaded successfully");
LOG_DEBUG("Processing iteration " << i << "/" << total);
LOG_ERROR("Failed to load model: " << filename);
```

### 4. System Topology Detection

**Files**: `src/topology_manager.h/cpp`, `src/common.h/cpp`
- **Classes**: `TopologyManager`, `CPUTopology`, `SystemTopology`
- **Features**:
  - CPU architecture detection (sockets, cores, hyperthreading)
  - NUMA topology mapping
  - GPU device enumeration (CUDA/ROCm support planned)
  - Memory capacity reporting
  - MPI environment information

**Detection Capabilities**:
- **CPU**: Socket count, cores per socket, hyperthreading status
- **NUMA**: Node count, CPU affinity, memory distribution  
- **Memory**: Total and per-NUMA-node capacity
- **GPU**: Device enumeration (framework-agnostic detection)
- **MPI**: Rank, size, process distribution

**Example Output**:
```
=== CPU Topology ===
Total CPUs: 112, Physical cores: 56, Sockets: 2
Hyperthreading: Yes, Using: No

=== NUMA Topology ===  
Node 0: 56 CPUs, 376 GB memory
Node 1: 56 CPUs, 377 GB memory
```

### 2. Hybrid / Sharded Tensor Architecture

**Files**: `src/tensors/tensor_base.h`, `src/tensors/tensor_factory.cpp`, `src/tensor.h`
- **Classes**: `TensorBase`, `SimpleTensor`, `COSMATensor`, `TensorFactory`
- **Pattern**: Abstract interface with concrete implementations
- **Purpose**: Zero-copy COSMA optimization with legacy compatibility

**Key Components**:

#### TensorBase Abstract Interface
```cpp
class TensorBase {
public:
    virtual const std::vector<int>& shape() const = 0;
    virtual float* data() = 0;
    virtual const float* data() const = 0;
    virtual std::string type_name() const = 0;
    virtual bool is_distributed() const = 0;
};
```

#### SimpleTensor (Minimal Host Row-Major)
- Keeps row‑major buffers for small ops & latency‑critical decode.
- Still used for many activation intermediates (seq_len × hidden_size) to minimize transformation overhead.

#### COSMATensor (COSMA Optimization)
- **Purpose**: Direct integration with `cosma::CosmaMatrix<float>`
- **Data Storage**: COSMA-optimized distributed layout
- **Use Cases**: Large matrices (≥256×256), multi-process operations
- **Performance**: Zero-copy access to COSMA operations

#### TensorFactory (Smart Selection)
- **Auto-Selection Logic**:
  - **Matrix Size**: COSMATensor for matrices ≥256×256 elements
  - **MPI Context**: COSMATensor when multiple processes available
  - **Operation Type**: COSMATensor preferred for matmul/attention
  - **Fallback**: SimpleTensor for compatibility

**Usage Examples**:
```cpp
// Automatic tensor selection
auto tensor = TensorFactory::create_auto({1024, 1024});
// → COSMATensor in MPI environment

// Explicit type creation
auto simple = TensorFactory::create_simple({512, 512});
auto cosma = TensorFactory::create_cosma({2048, 2048}, "matmul_A", mpi_rank);

// Legacy compatibility
std::shared_ptr<Tensor> legacy = std::make_shared<Tensor>({256, 256});
auto upgraded = TensorFactory::from_tensor(legacy);
```

**Performance Benefits**:
- **Zero-Copy COSMA**: Direct `cosma::multiply()` calls without data copying
- **Automatic Optimization**: Large matrices automatically use COSMA layout
- **Legacy Compatibility**: No performance penalty for existing code
- **Smart Conversion**: Only copies data when necessary

### 3. Multi-Architecture Pipeline System ✨

The abstract pipeline architecture provides clean separation between model-family-specific logic and the core inference engine.

#### Architecture Components

**Files**: `src/abstract_pipeline.h`, `src/pipeline_base.h`, `src/pipeline_factory.h`

**Core Classes**:
- **`AbstractPipeline`**: Pure virtual interface defining pipeline lifecycle
- **`PipelineBase`**: MPI-aware base with kernel composition and tensor utilities  
- **`PipelineFactory`**: Singleton registry for architecture-specific creators
- **`IModelWeights`**: Polymorphic weight access interface
- **`StageContext`**: Metadata for prefill/decode stage tracking
- **`KVCacheState`**: KV cache capacity and usage tracking

#### Pipeline Interface

```cpp
class AbstractPipeline {
public:
    virtual const ModelConfig& config() const = 0;
    virtual bool prefill(const std::vector<int>& tokens,
                        const IModelWeights& weights,
                        StageContext& ctx) = 0;
    virtual bool decode(int next_token,
                       const IModelWeights& weights,
                       StageContext& ctx) = 0;
    virtual bool logits(std::shared_ptr<TensorBase>& out_logits) = 0;
    virtual std::unique_ptr<IModelWeights> loadWeights(const std::string& path) = 0;
    virtual std::string name() const = 0;
};
```

#### Concrete Implementations

**QwenPipeline** (`src/qwen_pipeline.{h,cpp}`):
- Production implementation for Qwen 2.5 model family
- Formerly `DistributedTransformerPipeline` (renamed for clarity)
- Implements complete transformer execution with MPI distribution
- Supports both prefill and decode with adaptive backend selection

**QwenPipelineAdapter** (`src/qwen_pipeline_adapter.{h,cpp}`):
- Wraps `QwenPipeline` behind `AbstractPipeline` interface
- Provides `QwenModelWeights` implementing `IModelWeights`
- Handles stage context management and logits caching
- Implements `loadWeights()` for Qwen-specific GGUF parsing

**LlamaPipelineAdapter** (`src/llama_pipeline_adapter.{h,cpp}`):
- Prototype implementation for LLaMA model family
- Currently delegates to `QwenPipeline` (similar architecture)
- Demonstrates multi-architecture extensibility
- Future: Specialized for LLaMA-specific features (GQA variations, etc.)

#### Pipeline Factory Pattern

```cpp
// Registration (done at startup)
PipelineFactory::instance().registerCreator("qwen", 
    [](const ModelConfig& cfg) {
        return std::make_unique<QwenPipelineAdapter>(cfg);
    });

// Usage in main.cpp
auto pipeline = PipelineFactory::instance().create(model_config);
```

#### Stage-Driven Execution

**Prefill Phase**:
1. Embedding lookup
2. Layer loop:
   - RMSNorm (attention)
   - QKV projection (may use fused COSMA path)
   - Attention primitives (RoPE, scaled dot-product, output projection)
   - Residual connection
   - RMSNorm (FFN)
   - MLP (Gate/Up/SwiGLU/Down with adaptive backend)
   - Residual connection
3. Output RMSNorm + LM head projection

**Decode Phase**:
1. Embedding lookup (single token)
2. Layer loop (same structure, optimized for single token):
   - Local OpenBLAS for all matmuls (latency-optimized)
   - KV cache append
   - Causal attention over growing context
3. Output projection → logits

#### Benefits Over Legacy Compute Graph

- ✅ **Eliminated**: Generic node scheduling overhead
- ✅ **Simplified**: Fixed semantic stages vs dynamic DAG
- ✅ **Centralized**: Environment controls via `debugEnv()` snapshot
- ✅ **Modular**: Clean adapter boundaries for new architectures
- ✅ **Testable**: Explicit stage transitions and parity validation


### 4. Weight Contract System ✨ *UPDATED OCTOBER 12, 2025*

The weight contract system provides **load-time validation** and **contract-driven loading** of model weight dimensions and orientations, eliminating runtime shape detection, manual MPI slicing logic, and providing clear error messages when GGUF files don't match expected formats.

**Files**: `src/weight_contracts.h`, `src/mpi_slicing_helper.{h,cpp}`, `src/qwen_pipeline_adapter.{h,cpp}`, `docs/WEIGHT_CONTRACTS.md`

#### Architecture Components

**Core Classes**:
- **`WeightShapeContract`**: Specification for a single weight tensor with symbolic dimension expressions, GGUF layout metadata, and **MPI slicing strategy**
- **`ModelWeightContracts`**: Collection of contracts for a complete architecture (global + per-layer weights)
- **`getQwenWeightContracts()`**: Qwen/Qwen2 canonical GGUF format specification with MPI slicing metadata
- **`QwenModelWeights::validate()`**: Validation method called during model loading
- **`mpi_slicing::load_with_contract()`**: ✨ **NEW** Automatic MPI-aware weight loading with dimension handling
- **`WeightShapeContract::load()`**: ✨ **NEW** High-level interface for contract-driven loading

#### Design Philosophy

**Problem Statement**: Before weight contracts, we had:
- ❌ Runtime shape detection in kernels (performance overhead)
- ❌ Inconsistent test fixtures (synthetic data didn't match GGUF format)
- ❌ Silent shape mismatches leading to subtle bugs
- ❌ No single source of truth for weight format expectations
- ❌ **Manual MPI slicing logic duplicated in every pipeline** (150+ lines per pipeline)
- ❌ **Manual offset calculations** prone to rank-specific bugs (K/V weights on rank 1)
- ❌ **Dimension transpose handling** scattered across loading code

**Solution**: Declarative contracts validated at load time + automatic MPI-aware loading:
```cpp
// Define expected format with symbolic expressions + MPI slicing metadata
WeightShapeContract("blk.{layer}.attn_k.weight", 
    {"n_head_kv*head_dim", "d_model"},              // PyTorch dimensions
    "Key projection (row-sliced by KV heads)",      // Human description
    false,                                          // Not optional
    WeightSliceType::ROW_SLICED,                   // MPI slicing strategy
    "n_head_kv",                                   // Slice parameter
    {"d_model", "n_head_kv*head_dim"},             // GGUF dimensions (transposed!)
    true)                                          // Needs data transpose
```

#### Canonical GGUF Format

**Universal Rule**: ALL weight matrices are `[out_features, in_features]` matching PyTorch `nn.Linear` convention.

**Attention Weights**:
```cpp
attn_q.weight:      [n_head*head_dim, d_model]       // [896, 896]
attn_k.weight:      [n_head_kv*head_dim, d_model]    // [128, 896] for GQA
attn_v.weight:      [n_head_kv*head_dim, d_model]    // [128, 896] for GQA
attn_output.weight: [d_model, n_head*head_dim]       // [896, 896]

attn_q.bias:        [n_head*head_dim]                // Critical: must be loaded!
attn_k.bias:        [n_head_kv*head_dim]
attn_v.bias:        [n_head_kv*head_dim]
```

**FFN Weights**:
```cpp
ffn_gate.weight:    [d_ff, d_model]                  // [4864, 896]
ffn_up.weight:      [d_ff, d_model]                  // [4864, 896]
ffn_down.weight:    [d_model, d_ff]                  // [896, 4864]
```

**Global Weights**:
```cpp
token_embedding:    [vocab_size, d_model]            // [151669, 896]
output_norm.weight: [d_model]                        // RMSNorm gamma
lm_head:            [vocab_size, d_model]            // [151669, 896]
```

#### Contract-Driven Loading ✨ *NEW OCTOBER 12, 2025*

**Motivation**: Manual MPI weight loading was error-prone and duplicated across pipelines:
- ❌ ~150 lines of manual offset calculations per pipeline
- ❌ Subtle bugs in rank-specific slicing (K/V weights had 12.3 max error on rank 1)
- ❌ Dimension transpose logic scattered throughout loading code
- ❌ No automatic handling of GGUF vs PyTorch dimension conventions

**Solution**: One-line loading with automatic dimension handling, MPI slicing, and transposition:

**Before (qwen_pipeline.cpp - Manual Loading)**:
```cpp
// Manual offset calculations for each rank
const int q_row_offset = mpi_rank * q_heads_per_rank * config.head_dim;
const int q_row_count = q_heads_per_rank * config.head_dim;

// Create tensor
auto q_tensor = TensorFactory::create_simple({q_row_count, config.d_model});

// Load with manual parameters
bool success = loader.loadTensorRowShard(
    q_name, q_row_offset, q_row_count,
    const_cast<float*>(q_tensor->data()));

// Repeat for K, V, O, FFN_gate, FFN_up, FFN_down...
// Total: ~150 lines of duplicate slicing logic
```

**After (qwen_pipeline.cpp - Contract-Driven)**:
```cpp
// Get weight contracts
auto contracts = llaminar::getQwenWeightContracts();

// One line per weight - automatic everything!
auto wq = contracts.layer_weights[IDX_Q].load(
    loader, config, mpi_rank, mpi_size, layer);
auto wk = contracts.layer_weights[IDX_K].load(
    loader, config, mpi_rank, mpi_size, layer);
auto wv = contracts.layer_weights[IDX_V].load(
    loader, config, mpi_rank, mpi_size, layer);
auto wo = contracts.layer_weights[IDX_O].load(
    loader, config, mpi_rank, mpi_size, layer);

// Total: ~30 lines (5x reduction, 120 lines eliminated)
```

**How It Works**:

1. **Dimension Evaluation**: Contract evaluates symbolic expressions
   - `n_head_kv=2, head_dim=64 → n_head_kv*head_dim = 128`

2. **Transpose Detection**: Compares GGUF vs PyTorch dimensions
   - GGUF: `[d_model, n_head_kv*head_dim]` = `[896, 128]`
   - PyTorch: `[n_head_kv*head_dim, d_model]` = `[128, 896]`
   - **Transposed!** → Use column slice in GGUF, transpose data after loading

3. **MPI Slicing**: Calculates rank-specific offsets
   - Row-sliced in PyTorch = Column-sliced in GGUF (when transposed)
   - Rank 1 of 2 → columns `[64:128]` from GGUF `[896, 128]`

4. **Loading**: Calls appropriate ModelLoader method
   - `loadTensorColumnShard()` for column slices
   - `loadTensorRowShard()` for row slices

5. **Transposition**: If `needs_transpose_data=true`
   - Load: `[896, 64]` (GGUF columns 64:128)
   - Transpose: `[64, 896]` (PyTorch layout)

6. **Validation**: Checks shape matches contract (sliced dimensions)

**Validation Results** (test_contract_loading with np=2):
```
Testing contract-driven loading with 2 ranks
Model: qwen2.5-0.5b-instruct-q2_k.gguf

[Rank 0] Q weight shape: [448, 896]   // 14 heads * 64 / 2 ranks = 448 rows
[Rank 1] Q weight shape: [448, 896]
[Rank 0] K weight shape: [64, 896]    // 2 KV heads * 64 / 2 ranks = 64 rows  
[Rank 1] K weight shape: [64, 896]    // ✓ Correctly loaded on rank 1!
[Rank 0] V weight shape: [64, 896]
[Rank 1] V weight shape: [64, 896]
[Rank 0] O weight shape: [896, 448]   // 896 in, 896 out / 2 ranks = 448 cols
[Rank 1] O weight shape: [896, 448]

✅ All weights loaded successfully!
```

**Benefits**:
- ✅ **150 lines eliminated** from qwen_pipeline.cpp (5x code reduction)
- ✅ **Zero bugs**: Automatic dimension handling prevents rank-specific errors
- ✅ **Self-documenting**: Contract specifies slice strategy and transpose needs
- ✅ **Reusable**: Same system for llama_pipeline, future architectures
- ✅ **Testable**: Simple validation test confirms correctness

**Implementation Files**:
- `src/mpi_slicing_helper.{h,cpp}`: Core slicing logic with dimension handling
- `src/weight_contracts.h`: Enhanced with `load()` method and GGUF metadata
- `src/qwen_pipeline.cpp`: Migrated to contract-driven loading
- `tests/test_contract_loading.cpp`: Validation test for contract loading


#### Validation Flow

**Load-Time Validation** (`QwenPipelineAdapter::loadWeights()`):
```cpp
std::unique_ptr<IModelWeights> QwenPipelineAdapter::loadWeights(const std::string& path) {
    // 1. Load GGUF file via ModelLoader
    ModelLoader loader;
    loader.loadModel(path);
    auto loaded = loadModelWeights_impl_bridge(loader, cfg_);
    
    // 2. Wrap in QwenModelWeights
    auto weights = std::make_unique<QwenModelWeights>();
    weights->inner = std::move(loaded);
    
    // 3. VALIDATE against contracts - fails fast with clear errors
    try {
        weights->validate(cfg_.getLayerConfig());
        LOG_INFO("✓ All weights validated against canonical GGUF format");
    } catch (const std::exception& e) {
        LOG_ERROR("Weight validation failed: " << e.what());
        throw;  // Fail fast before inference
    }
    
    return weights;
}
```

**Validation Logic** (`QwenModelWeights::validate()`):
```cpp
void QwenModelWeights::validate(const TransformerLayerConfig& cfg) const {
    auto contracts = getQwenWeightContracts();
    
    // Validate global weights
    contracts.validate_global(inner.token_embedding, inner.output_norm_weight,
                             inner.lm_head, cfg);
    
    // Validate each layer
    for (int layer = 0; layer < layer_count(); ++layer) {
        contracts.validate_layer(layer,
            inner.attn_norm_weight[layer],
            inner.wq[layer], inner.wk[layer], inner.wv[layer], inner.wo[layer],
            inner.ffn_norm_weight[layer],
            inner.w_gate[layer], inner.w_up[layer], inner.w_down[layer],
            cfg);
    }
}
```

#### Error Messages

When validation fails, you get **immediate, actionable errors**:

```
[ERROR] Weight validation failed: Weight contract validation failed for 'attn_k.weight' (layer 0):
  Description: Key projection (GGUF format: [out, in])
  Reason: Dimension 0 mismatch
  Expected shape: [128, 896] (from [n_head_kv*head_dim, d_model])
  Actual shape:   [896, 128]
```

This clearly identifies:
- **Which weight** (`attn_k.weight`)
- **Which layer** (layer 0)
- **What was expected** (`[128, 896]` from symbolic `[n_head_kv*head_dim, d_model]`)
- **What was found** (`[896, 128]`)
- **Human description** (helps understand purpose)

#### Kernel Benefits

Kernels can now **trust** the weight format without runtime detection:

**Before (Runtime Detection)**:
```cpp
// MPIAttentionKernel had to guess orientation
const int wq_cols = wq_global->shape()[1];
const bool weights_are_sharded = (wq_cols == local_head_dim);
// Complex logic to handle both [d_model, heads] and [heads, d_model]
```

**After (Trust Contracts)**:
```cpp
// MPIAttentionKernel knows GGUF format is guaranteed
// WEIGHT FORMAT CONTRACT:
// All weights guaranteed by QwenModelWeights to be in canonical GGUF format:
// - wq, wk, wv: [out_features, in_features] = [n_head*head_dim, d_model] or [local_head_dim, d_model] if sharded
// - wo: [in_features, out_features] = [d_model, n_head*head_dim] or [d_model, local_head_dim] if sharded

const int wq_rows = wq_global->shape()[0];
const bool weights_are_sharded = (wq_rows == local_head_dim);  // Simple check!
```

#### Test Fixture Requirements

All test fixtures **must** create weights matching GGUF canonical format:

**Example** (`test_abstract_pipeline_parity.cpp`):
```cpp
struct RandomWeightBuilder {
    QwenPipeline::ModelWeights build() {
        QwenPipeline::ModelWeights w;
        
        // Global weights
        w.token_embedding = randTensor({cfg.vocab_size, cfg.d_model});
        w.lm_head = randTensor({cfg.vocab_size, cfg.d_model});  // [vocab, d_model]!
        
        for (int i = 0; i < cfg.n_layers; ++i) {
            // Attention: ALL [out_features, in_features]
            w.wq.push_back(randTensor({cfg.n_head * cfg.head_dim, cfg.d_model}));
            w.wk.push_back(randTensor({cfg.n_head_kv * cfg.head_dim, cfg.d_model}));
            w.wv.push_back(randTensor({cfg.n_head_kv * cfg.head_dim, cfg.d_model}));
            w.wo.push_back(randTensor({cfg.d_model, cfg.n_head * cfg.head_dim}));
            
            // FFN: ALL [out_features, in_features]
            w.w_gate.push_back(randTensor({cfg.d_ff, cfg.d_model}));
            w.w_up.push_back(randTensor({cfg.d_ff, cfg.d_model}));
            w.w_down.push_back(randTensor({cfg.d_model, cfg.d_ff}));
        }
        return w;
    }
};
```

#### Benefits

1. **✅ Fail Fast**: Validation at model load, not mid-inference
2. **✅ Clear Errors**: Symbolic expressions + descriptions pinpoint exact issue
3. **✅ Simplified Kernels**: No runtime shape detection needed
4. **✅ Self-Documenting**: Contracts serve as executable spec
5. **✅ Test Consistency**: Synthetic data forced to match production format
6. **✅ Multi-Architecture**: Easy to add contracts for LLaMA, GPT, etc.
7. **✅ Massive Code Reduction**: ✨ **NEW** 150 lines eliminated per pipeline via contract-driven loading
8. **✅ Zero MPI Slicing Bugs**: ✨ **NEW** Automatic dimension handling prevents rank-specific errors
9. **✅ Automatic Transpose**: ✨ **NEW** GGUF ↔ PyTorch conversion handled transparently
10. **✅ Reusable Infrastructure**: ✨ **NEW** Same loading system for all future model architectures

#### Future Extensions

**Quantization Validation**:
```cpp
struct WeightShapeContract {
    QuantizationType expected_quant = QuantizationType::F32;
    // Validate quantization type matches
};
```

**Value Range Checks**:
```cpp
struct WeightShapeContract {
    std::optional<float> expected_min;
    std::optional<float> expected_max;
    // Sanity check for corruption detection
};
```

**Additional Architectures**:
```cpp
inline ModelWeightContracts getLlamaWeightContracts() { /* ... */ }
inline ModelWeightContracts getGPTWeightContracts() { /* ... */ }
// Contract-driven loading works for all architectures automatically!
```

**Advanced Slicing Strategies**:
```cpp
enum class WeightSliceType {
    REPLICATED,       // Full weight on every rank
    ROW_SLICED,       // Row-wise slicing (first dimension)
    COL_SLICED,       // Column-wise slicing (second dimension)
    HEAD_SLICED,      // Future: head-wise slicing for attention
    BLOCK_2D_SLICED   // Future: 2D block slicing for multi-node
};
```

#### Related Documentation

- **Full Specification**: `docs/WEIGHT_CONTRACTS.md`
- **GGUF Format Reference**: `src/model_loader.h` (lines 75-110)
- **Runtime Validation**: `src/kernels/attention/AttentionStageContracts.h`


### 5. Prefill Provider Abstraction ✨ *REFACTORED OCTOBER 2025*

**Files**: 
- `src/prefill_provider.{h,cpp}` - Base interface and factory
- `src/prefill_provider_base_impl.{h,cpp}` - Template Method base (680 lines)
- `src/openblas_prefill_provider.{h,cpp}` - OpenBLAS provider (280 lines, 69% reduction)
- `src/cosma_prefill_provider.{h,cpp}` - COSMA provider (260 lines, 70% reduction)
- `src/cublas_prefill_provider.{h,cpp}` - GPU stub (placeholder)
- `src/rocblas_prefill_provider.{h,cpp}` - AMD GPU stub (placeholder)

The prefill provider architecture implements **Strategy + Template Method patterns** for swappable prefill execution backends, achieving:
- **58-65% code reduction** per provider via shared base implementation
- Multiple prefill implementations (OpenBLAS, COSMA, future GPU)
- Built-in snapshot capture for parity testing (387 consistent capture points)
- Runtime backend selection based on workload characteristics
- Isolated testing of individual providers
- Stage-by-stage instrumentation and validation

#### Architecture Hierarchy

```
AbstractPipeline::prefill()
  └─> QwenPipeline::prefill()
       └─> PrefillProvider::execute() [interface]
            └─> PrefillProviderBaseImpl::execute() [template method - 680 lines shared]
                 ├─> OpenBLASPrefillProvider  (280 lines: 3 virtual methods)
                 ├─> COSMAPrefillProvider     (260 lines: 3 virtual methods)
                 ├─> CuBLASPrefillProvider    (stub: ready for GPU)
                 └─> ROCmPrefillProvider      (stub: ready for AMD)
```

#### PrefillProvider Base Interface

**Purpose**: Abstract interface defining provider contract

**Core Interface**:
```cpp
class PrefillProvider {
public:
    virtual bool execute(
        const std::vector<int>& tokens,
        const IModelWeights& weights,
        std::shared_ptr<TensorBase>& output,
        StageContext& ctx,
        PrefillMetrics& metrics) = 0;
    
    virtual std::string name() const = 0;
    
protected:
    void captureSnapshot(PipelineStage stage, int layer_index,
                        const float* data, int seq_len, int feature_dim);
    bool isSnapshotEnabled() const;
};
```

#### PrefillProviderBaseImpl (Template Method)

**Files**: `src/prefill_provider_base_impl.{h,cpp}` (680 lines)

**Purpose**: Shared execution flow and snapshot capture logic for all providers

**Design Pattern**: Template Method - base defines algorithm structure, derived override steps

**What's Shared** (70-80% of original code):
- ✅ Main `execute()` implementation (embedding → layers → norm → LM head)
- ✅ `executeTransformerLayer()` (attention + FFN + residuals)
- ✅ `executeFfnBlock()` (norm → gate/up → swiglu → down → residual)
- ✅ All 387 snapshot capture points (identical across backends)
- ✅ Timing/metrics collection infrastructure
- ✅ Error handling patterns
- ✅ Kernel registration scaffolding

**What's Delegated** (3 virtual methods per provider):
```cpp
class PrefillProviderBaseImpl : public PrefillProvider {
public:
    // Final implementation - shared flow
    bool execute(...) final;
    
protected:
    // Virtual methods - backend-specific operations
    virtual bool executeEmbedding(
        const std::vector<int>& tokens,
        const std::shared_ptr<TensorBase>& embedding_weights,
        std::shared_ptr<TensorBase>& output) = 0;
    
    virtual bool executeLinearProjection(
        const std::shared_ptr<TensorBase>& input,
        const std::shared_ptr<TensorBase>& weight,
        const std::shared_ptr<TensorBase>& bias,
        std::shared_ptr<TensorBase>& output,
        const std::string& op_name) = 0;
    
    virtual bool executeAttentionBlock(
        const std::shared_ptr<TensorBase>& input,
        const AttentionWeights& attn_weights,
        std::shared_ptr<TensorBase>& output,
        StageContext& ctx,
        int layer_idx) = 0;
    
    // Shared implementation methods
    bool executeTransformerLayer(...);
    bool executeFfnBlock(...);
};
```

**Key Features**:
- **Single Source of Truth**: Bug fixes and optimizations apply to all backends
- **Consistent Snapshots**: All 387 capture points identical across providers
- **DRY Principle**: Eliminated 1,083 net lines of duplicated code
- **Zero Overhead**: Snapshots compiled out in release builds
- **MPI-Aware**: Handles distributed execution and rank coordination
- **Extensible**: New backends need only 3 virtual methods (~200 lines)

#### OpenBLASPrefillProvider (Production Baseline)

**File**: `src/openblas_prefill_provider.{h,cpp}` (280 lines total)

**Code Reduction**: 865 → 280 lines (69% reduction, 585 lines eliminated)

**Purpose**: CPU-based prefill using OpenBLAS for matrix multiplications

**Characteristics**:
- **Baseline Implementation**: Well-tested, predictable behavior
- **Optimal For**: Small to medium sequences (< 4K tokens), single-node setups
- **MatMul Backend**: OpenBLAS (single/multi-threaded based on operation size)
- **Kernel Architecture**: Uses existing MPI kernel infrastructure
  - `MPIEmbeddingKernel`: Token embedding lookup
  - `MPIRMSNormKernel`: Layer normalization (sequence-wise distribution)
  - `MPIAttentionKernel`: Complete attention (Q/K/V/O projections + primitives)
  - `MPILinearKernel`: FFN linear projections
  - `MPISwiGLUKernel`: SwiGLU activation

**Implementation** (3 virtual methods only):
```cpp
class OpenBLASPrefillProvider : public PrefillProviderBaseImpl {
public:
    std::string name() const override { return "OpenBLAS"; }
    
protected:
    // Virtual method 1: Embedding lookup
    bool executeEmbedding(...) override {
        return embedding_kernel_->execute(...);
    }
    
    // Virtual method 2: Linear projections (FFN, LM head)
    bool executeLinearProjection(...) override {
        return linear_kernel_->execute(...);
    }
    
    // Virtual method 3: Attention block
    bool executeAttentionBlock(...) override {
        return attention_kernel_->execute(...);
    }
};
```

**What's Gone** (moved to PrefillProviderBaseImpl):
- ❌ Main execute() method (~180 lines)
- ❌ executeTransformerLayer() (~120 lines)
- ❌ executeFfnBlock() (~100 lines)
- ❌ All snapshot capture logic (387 capture points)
- ❌ All timing/metrics collection
- ❌ Kernel registration infrastructure

**Stage Flow** (inherited from base):
1. **EMBEDDING**: Token embedding lookup
2. **Per Layer**:
   - ATTENTION_NORM → Attention (Q/K/V/RoPE/scores/softmax/context/output) → ATTENTION_RESIDUAL
   - FFN_NORM → FFN_GATE/UP → FFN_SWIGLU → FFN_DOWN → FFN_RESIDUAL
3. **FINAL_NORM** → **LM_HEAD**

**Snapshot Capture**: All 387 standardized stages automatically captured by base class

#### COSMAPrefillProvider (Distributed Execution)

**File**: `src/cosma_prefill_provider.{h,cpp}` (260 lines total)

**Code Reduction**: 758 → 260 lines (70% reduction, 498 lines eliminated)

**Purpose**: Distributed prefill using COSMA for large-scale matrix multiplications

**Characteristics**:
- **Optimal For**: Large sequences (≥ 4K tokens), multi-node setups (2+ MPI ranks)
- **Performance**: Up to 3.6x faster than OpenBLAS for large operations (≥64K tokens)
- **MatMul Backend**: COSMA for distributed compute, with adaptive fallback
- **Fused Operations**: Combines RMSNorm + QKV projection for efficiency
- **Backend Integration**: Uses unified MPIAttentionKernel with COSMA injection

**Implementation** (3 virtual methods only):
```cpp
class COSMAPrefillProvider : public PrefillProviderBaseImpl {
public:
    std::string name() const override { return "COSMA"; }
    
protected:
    // Virtual method 1: Embedding lookup (replicated)
    bool executeEmbedding(...) override {
        // Simple memcpy - no COSMA for small embedding table
        memcpy(output->data(), &embedding_weights[token_id], ...);
        return true;
    }
    
    // Virtual method 2: Linear projections via adaptive matmul
    bool executeLinearProjection(...) override {
        return adaptiveMatMul(input, weight, output, /*transposed_b=*/true);
    }
    
    // Virtual method 3: Attention via unified MPIAttentionKernel + COSMA injection
    bool executeAttentionBlock(...) override {
        attention_kernel_->setCosmaManager(cosma_manager_.get());
        return attention_kernel_->execute(...);
    }
};
```

**What's Gone** (moved to PrefillProviderBaseImpl):
- ❌ Main execute() method (~200 lines)
- ❌ executeTransformerLayer() (~150 lines)
- ❌ executeFfnBlock() (~110 lines)
- ❌ Duplicated attention logic (185 lines - now unified via MPIAttentionKernel)
- ❌ All snapshot capture logic (387 capture points)
- ❌ All timing/metrics collection

**Key Differences from OpenBLAS**:
- **Attention**: Uses unified MPIAttentionKernel with COSMA manager injection
- **FFN**: Uses `adaptiveMatMul` (may use COSMA for gate/up/down based on size)
- **Memory**: Distributed weight layout, higher communication overhead
- **Tradeoff**: Better throughput for large ops, worse for small ops (<4K tokens)

**Snapshot Alignment**:
- Captures at **same 387 stages** as OpenBLASPrefillProvider (base class guarantees)
- Enables A/B testing: run both providers, compare snapshots stage-by-stage
- Identifies divergence source (COSMA matmul vs attention primitives vs etc.)

#### GPU Provider Stubs (Ready for Implementation)

**Files**: 
- `src/cublas_prefill_provider.{h,cpp}` - NVIDIA GPU stub
- `src/rocblas_prefill_provider.{h,cpp}` - AMD GPU stub

**Current Status**: Placeholder implementations (~100 lines each)

**Implementation Template**:
```cpp
class CuBLASPrefillProvider : public PrefillProviderBaseImpl {
public:
    std::string name() const override { return "cuBLAS"; }
    
protected:
    bool executeEmbedding(...) override {
        // TODO: cudaMemcpy, device embedding lookup
        return false;  // Not implemented
    }
    
    bool executeLinearProjection(...) override {
        // TODO: cublasSgemm for Y = XW
        return false;  // Not implemented
    }
    
    bool executeAttentionBlock(...) override {
        // TODO: cuBLAS attention or FlashAttention
        return false;  // Not implemented
    }
};
```

**When Implemented** (~200-300 lines total):
- ✅ Inherits all 387 snapshot points automatically
- ✅ Inherits all timing/metrics infrastructure
- ✅ Inherits execution flow (embedding → layers → norm → LM head)
- ✅ Only needs device-specific matmul/attention primitives
- ✅ Factory can auto-detect GPU and select this provider

#### PrefillProviderFactory

**File**: `src/prefill_provider.cpp`

**Purpose**: Automatic provider selection based on configuration and workload

**Selection Logic**:
```cpp
std::unique_ptr<PrefillProvider> PrefillProviderFactory::create(
    const ModelConfig& config,
    const MPIContext& mpi_ctx,
    int seq_len) {
    
    const auto& env = debugEnv();
    
    // Check global COSMA disable
    if (env.adaptive.disable_cosma) {
        return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
    }
    
    // Check for forced COSMA execution (debug/validation)
    if (env.cosma.force_direct || env.cosma.force_replicated) {
        return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
    }
    
    // Sequence length-based decision
    int threshold = env.cosma.prefill_threshold;  // default: 4096
    if (seq_len >= threshold && mpi_ctx.size > 1) {
        return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
    }
    
    // Default: OpenBLAS for small/medium sequences
    return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
}
```

**Decision Criteria**:
1. **Environment Override**: `ADAPTIVE_DISABLE_COSMA=1` → always OpenBLAS
2. **Debug Forcing**: `LLAMINAR_COSMA_FORCE_DIRECT=1` → always COSMA
3. **Sequence Length**: `seq_len >= threshold` (default 4096) → COSMA (if multi-rank)
4. **Default Fallback**: OpenBLAS for small sequences or single-rank

#### PrefillMetrics

**Purpose**: Comprehensive instrumentation for prefill execution

**Tracked Metrics**:
- **Timing**: Per-stage breakdown (embedding, attention, FFN, norm, LM head)
- **Compute**: Total FLOP count and GFLOPS throughput
- **Execution**: Layers executed, snapshots captured
- **Backend**: Provider name for performance attribution

**Usage Example**:
```cpp
PrefillMetrics metrics;
auto provider = PrefillProviderFactory::create(config, mpi_ctx, seq_len);
bool success = provider->execute(tokens, weights, output, ctx, metrics);

if (success) {
    LOG_INFO("Prefill completed: " << metrics.total_ms() << "ms, "
             << metrics.gflops() << " GFLOPS, "
             << "backend=" << metrics.backend_name);
}
```

#### Environment Controls

**Provider Selection**:
- `ADAPTIVE_DISABLE_COSMA=1`: Force OpenBLAS provider globally
- `LLAMINAR_COSMA_PREFILL_THRESHOLD=<tokens>`: Sequence length threshold (default: 4096)
- `LLAMINAR_COSMA_FORCE_DIRECT=1`: Force COSMA provider (debug)
- `LLAMINAR_COSMA_FORCE_REPLICATED=1`: Force COSMA replicated path (validation)

**COSMA-Specific Controls**:
- `LLAMINAR_COSMA_MAX_RESIDENT_MB=<mb>`: Memory budget (default: 2048)
- `LLAMINAR_COSMA_VALIDATE_TILE=<size>`: Enable correctness validation (debug)
- `LLAMINAR_COSMA_COMPARE_REPLICATED=1`: Full replicated comparison (expensive)
- `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE=1`: Automatic orientation correction

**Snapshot Controls**:
- `LLAMINAR_PARITY_CAPTURE=1`: Enable snapshot capture in debug builds
- `LLAMINAR_COSMA_TEST_TRACE=1`: Escalate COSMA test verbosity

#### Integration with Pipeline

The pipeline uses the factory for automatic provider selection:

```cpp
// In QwenPipeline::prefill()
PrefillMetrics metrics;
auto provider = PrefillProviderFactory::create(config_, mpi_ctx_, tokens.size());

bool success = provider->execute(tokens, weights, hidden_states, ctx, metrics);

if (success) {
    // Cache logits, update metrics, etc.
    ctx.backend_stats.prefill_backend = metrics.backend_name;
    ctx.backend_stats.prefill_gflops = metrics.gflops();
}
```

#### Benefits Over Previous Architecture

**Before (Monolithic QwenPipeline)**:
- ✗ Prefill logic scattered across pipeline class
- ✗ Hard to test backends in isolation
- ✗ Snapshot capture ad-hoc and inconsistent
- ✗ Backend switching required code changes

**After (Provider Abstraction)**:
- ✓ Clean separation: pipeline orchestrates, providers execute
- ✓ Testable in isolation: unit tests for each provider
- ✓ Consistent snapshots: base class provides utilities
- ✓ Runtime selection: factory chooses optimal provider
- ✓ Extensible: GPU provider can be added without pipeline changes

### 5. COSMA Prefill Manager

**Files**: `src/cosma_prefill_manager.{h,cpp}`, `src/prefill_diagnostics.{h,cpp}`

The COSMA prefill manager provides distributed matrix multiplication for large prefill operations using the COSMA (Communication-Optimal Matrix Algorithm) library.

#### Current Implementation (Production - October 2025)

**Status**: ✅ **DEPLOYED** - Integrated via MPIAttentionKernel with COSMA injection

**Architecture**: COSMA manager is now a backend injected into the unified MPIAttentionKernel, not a standalone attention implementation.

**Key Features**:
- **Backend Injection Pattern**: MPIAttentionKernel accepts `setCosmaManager()` to switch matmul backend
- **Unified Attention Logic**: Same attention algorithm for both OpenBLAS and COSMA (86% code reduction)
- **Communication-Optimal**: COSMA handles matrix multiplication, CPU primitives handle attention
- **Host-Owned Results**: Returns simple host-owned buffers instead of complex distributed layouts
- **Production Stable**: 100% PyTorch parity (387/387 tests passing)

**Integration Flow**:

1. **Provider Setup** (COSMAPrefillProvider):
   ```cpp
   // Inject COSMA manager into unified attention kernel
   attention_kernel_->setCosmaManager(cosma_manager_.get());
   ```

2. **Attention Execution** (MPIAttentionKernel):
   ```cpp
   bool MPIAttentionKernel::matmul_with_bias(...) {
       if (cosma_manager_) {
           // COSMA distributed path
           return cosma_manager_->matmul(..., transposeW=true);
       } else {
           // OpenBLAS local path
           return cblas_sgemm(..., CblasTrans);
       }
   }
   ```

3. **RMSNorm + QKV Fusion** (CosmaPrefillManager):
   ```cpp
   // Proven primitives for RMSNorm
   llaminar::kernels::rmsnorm_t5_forward(
       activation_row_major, gamma, normalized->data(),
       seq_len, hidden_size, eps, true /* use_parallel */
   );
   
   // COSMA distributed matmul for Q/K/V projections
   auto q_view = matmul(norm_view, wq_handle, S, H, Oq, false);
   auto k_view = matmul(norm_view, wk_handle, S, H, Ok, false);
   auto v_view = matmul(norm_view, wv_handle, S, H, Ov, false);
   
   // Gather to host-owned buffers
   reconstruct_matrix(q_view, q_buf->data(), false);
   ```

**Benefits**:
- ✅ **Code Unification**: Eliminated 185 lines of duplicated COSMA attention logic
- ✅ **100% Parity**: All 387 tests passing with micro-precision accuracy (e-05 to e-06)
- ✅ **Maintainable**: Single attention implementation for both backends
- ✅ **Systematic Correctness**: All weight transposes handled consistently via flags
- ✅ **No Segfaults**: Proper RAII, no manual memory management
- ✅ **Flexible**: Easy to optimize later without breaking everything

**Trade-offs**:
- Small gather overhead (~5-10%) vs keeping data in COSMA layout
- Acceptable for correctness and maintainability
- Still provides COSMA performance benefits for large operations (≥4K tokens)

#### Core Functionality

**Orientation & Layout Management**:
- Automatic weight transpose handling via `transposeW` and `transposed_b` flags
- GGUF canonical format validation via weight contracts
- Systematic transpose fixes across all projection types (attention, FFN, LM_HEAD)

**Distributed Execution**:
- Coordinates across MPI ranks with barriers
- Handles partial matrix reconstruction and gathering
- Manages COSMA buffer allocation lifecycle
- Integration via MPIAttentionKernel backend injection

#### Diagnostics Integration

**Prefill Diagnostics Module** (`src/prefill_diagnostics.{h,cpp}`):
- **BufferStats**: Min/max/mean/L2 norm computation
- **DiffSummary**: Relative L2 error and maximum absolute difference
- **PrefillBaselineRegistry**: Reference execution for validation

**Usage Pattern**:
```cpp
// Optional baseline comparison
if (env.prefill_debug.compare_baseline) {
    auto baseline_result = PrefillBaselineRegistry::computeBaseline(...);
    auto diff = PrefillBaselineRegistry::compareResults(
        cosma_result, baseline_result);
    LOG_INFO("Prefill diff: rel_l2=" << diff.rel_l2 
             << ", max_abs=" << diff.max_abs);
}
```

#### Performance Counters

Tracks all GEMM operations with:
- Operation dimensions (m × n × k)
- Backend used (COSMA vs OpenBLAS)
- Execution time
- GFLOPS achieved

Post-run summary shows aggregate statistics for optimization.

#### Fallback Behavior

1. **Attempt COSMA Path**: Try distributed execution via COSMA
2. **Validation Check**: Optionally verify results against OpenBLAS tile
3. **On Failure**: Log warning and fall back to OpenBLAS provider
4. **Transparent Recovery**: Pipeline continues without user intervention (factory selects provider)

#### Environment Controls

**COSMA Execution**:

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Seq length threshold for COSMA | 4096 | Only use COSMA for large prompts |
| `ADAPTIVE_DISABLE_COSMA` | Force OpenBLAS path | Unset | Override for A/B testing |
| `LLAMINAR_COSMA_FORCE_DIRECT` | Force COSMA direct path | Unset | Bypass fast path heuristics |
| `LLAMINAR_COSMA_LOG_LEVEL` | Prefill logging verbosity | info | trace/debug/info/warn/error |

**COSMA Validation** (debug only):

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LLAMINAR_COSMA_COMPARE_REPLICATED` | Full OpenBLAS validation | Unset | Expensive - use for debugging only |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Small tile validation | 0 (off) | Set to 64 for spot checks |
| `LLAMINAR_COSMA_DEBUG_RECON` | Verbose reconstruction logs | Unset | Diagnose gather issues |

**Memory Management**:

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Soft memory budget | 2048 | Fallback if exceeded |

**Dequant Diagnostics**:

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LLAMINAR_DEQUANT_STATS` | Log dequant statistics | Unset | Min/max/mean per tensor |
| `LLAMINAR_DEQUANT_ANOMALIES` | Warn on NaN/Inf values | Unset | Safety diagnostic |

#### Integration Points

- Called from `COSMAPrefillProvider` via MPIAttentionKernel injection
- Used for attention QKV projection in large prefill (≥4K tokens)
- Coordinates with `PrefillProviderFactory` for automatic backend selection
- Reports statistics via `debugEnv().prefill_debug` snapshot

### 6. Distributed Linear Projection

**File**: `src/kernels/MPILinearKernel.cpp`

Provides column-partitioned dense layer implementation for distributed weight storage with local OpenBLAS computation.

#### Architecture

**Weight Distribution**:
- Weights partitioned by columns across MPI ranks
- Each rank holds `[hidden_size, proj_size / num_ranks]` slice
- Input activations replicated across all ranks

**Execution Flow**:
1. **Local GEMM**: Each rank computes `input @ local_weight_slice`
2. **Gather Results**: `MPI_Allgatherv` assembles full output across ranks  
3. **Optional Bias**: Add bias vector if provided

**Backend Integration**:
- Explicitly uses OpenBLAS for local matmul
- Used by `OpenBLASPrefillProvider` for all linear projections
- Ensures consistent execution with provider-based architecture

#### Use Cases

- Distributed weight storage when model exceeds single-node memory
- Balanced computation across MPI ranks
- Compatible with both prefill and decode phases

#### Performance Characteristics

**Advantages**:
- Distributes memory footprint across ranks
- Balances computation load
- Leverages fast local OpenBLAS for small/medium operations

**Overhead**:
- `MPI_Allgatherv` communication cost
- Beneficial when communication < local GEMM time savings

#### Interaction with Adaptive Backend

The kernel ensures COSMA is not applied to partial weight slices:

```cpp
// In MPILinearKernel::execute()
adaptiveMatMul(input, local_weight, local_output,
               /* distributed_partition= */ true);
// Flag prevents COSMA selection for partial matrix
```

This maintains correctness while allowing adaptive selection at higher pipeline levels.

### 7. Attention Implementation

**Files**: `src/kernels/MPIAttentionKernel.{h,cpp}`, `src/kernels/attention_primitives.{h,cpp}`

The attention system provides multi-head attention with RoPE positional encoding, using a **unified backend-agnostic kernel** that supports both OpenBLAS and COSMA execution paths.

#### Unified Architecture ✨ *NEW*

**Key Innovation**: `MPIAttentionKernel` is now backend-agnostic, eliminating code duplication between OpenBLAS and COSMA paths.

**Design Pattern**: Strategy injection via `setCosmaManager()`
```cpp
class MPIAttentionKernel : public MPIKernelBase {
public:
    // Unified interface for both backends
    bool execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                 std::vector<std::shared_ptr<TensorBase>>& outputs) override;
    
    // Runtime backend injection
    void setCosmaManager(CosmaPrefillManager* mgr) { cosma_manager_ = mgr; }
    
private:
    // Backend routing with identical semantics
    bool matmul_with_bias(const float* A, const float* weight_data,
                         const float* bias, float* output, ...);
    
    CosmaPrefillManager* cosma_manager_ = nullptr;  // nullptr = OpenBLAS
};
```

**Backend Selection**:
```cpp
// OpenBLAS path (prefill_provider or pipeline)
auto attention_kernel = std::make_unique<MPIAttentionKernel>(...);
bool success = attention_kernel->execute(inputs, outputs);  // Uses cblas_sgemm

// COSMA path (cosma_prefill_provider)
attention_kernel->setCosmaManager(cosma_manager_.get());
bool success = attention_kernel->execute(inputs, outputs);  // Uses COSMA distributed
```

**Benefits**:
- ✅ **Single Implementation**: Both backends execute identical attention algorithm
- ✅ **Consistent Snapshots**: Capture points synchronized across backends
- ✅ **86% Code Reduction**: Eliminated 185 lines of duplicated COSMA attention logic
- ✅ **Maintainable**: Bug fixes automatically apply to both paths
- ✅ **Type Safe**: Unified weight transpose handling

#### Execution Paths

**Small Operations (OpenBLAS Path)**:
1. RMSNorm normalization
2. Q/K/V projection via `cblas_sgemm` with `CblasTrans` (implicit transpose)
3. Reshape to multi-head format: `[batch, seq_len, num_heads, head_dim]`
4. Apply RoPE positional encoding per head
5. Scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
6. Output projection (local OpenBLAS)

**Large Prefill (COSMA Path)**:
1. RMSNorm normalization (same as OpenBLAS)
2. Q/K/V projection via `CosmaPrefillManager::matmul` with `transposeW=true` (explicit transpose)
3. Reshape to multi-head format (same algorithm as OpenBLAS)
4. Apply RoPE positional encoding (same algorithm)
5. Scaled dot-product attention (same algorithm)
6. Output projection (may use distributed COSMA)

**Critical Difference**: Only the matmul primitive changes; all attention logic is identical.

#### Weight Transpose Handling

**GGUF Storage Format**: ALL weights are `[output_dim, input_dim]` (PyTorch convention)

**OpenBLAS Transpose**:
```cpp
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,  // ← Implicit transpose
    m, n, k, 1.0f,
    activation, k,
    weight, k,     // weight is [n, k], transposed to [k, n] during matmul
    0.0f, output, n);
```

**COSMA Transpose**:
```cpp
WeightDescriptor weight_desc{
    weight_data,
    original_rows,     // [N, K] format preserved
    original_cols,
    original_cols      // row_stride = K (no manual transpose)
};
cosma_manager_->matmul(..., weight_desc, ..., 
    /*transposeW=*/true);  // ← Explicit transpose parameter
```

**Result**: Both paths compute identical `activation @ weight.T` operation, just with different API conventions.

#### RoPE (Rotary Position Embedding)

**Implementation**: Direct per-head loop with sin/cos computation
**Format**: Applied to Q and K tensors before attention
**Future Optimization**: Vectorization, LUT-based sin/cos reuse

```cpp
// Simplified RoPE application
for (int head = 0; head < num_heads; ++head) {
    for (int pos = 0; pos < seq_len; ++pos) {
        float theta = pos / pow(10000.0, 2.0 * dim / head_dim);
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        // Apply rotation to Q[head][pos] and K[head][pos]
    }
}
```

#### Attention Mechanism

**Scaled Dot-Product**:
- Query-Key product: `scores = Q @ K^T`
- Scaling: `scores /= sqrt(head_dim)`
- Causal masking: Set future positions to -inf
- Softmax: `probs = softmax(scores)` (row-wise normalization)
- Apply to values: `output = probs @ V`

**Execution**:
- All reductions and softmax executed locally (single rank)
- Causal masking enforced row-wise
- Future enhancement: Distributed softmax for large multi-rank KV spans

#### Multi-Head Attention

**Head Configuration**:
- Number of heads from `ModelConfig::n_heads`
- Head dimension: `hidden_size / n_heads`
- Supports Grouped Query Attention (GQA) variations

**Reshaping**:
- Input: `[batch, seq_len, hidden_size]`
- Multi-head: `[batch, seq_len, num_heads, head_dim]`
- Transposed for attention: `[batch, num_heads, seq_len, head_dim]`

#### Performance Optimizations

- **RoPE Caching**: Future LUT-based sin/cos precomputation
- **Fused Kernels**: Combined operations reduce memory traffic
- **Adaptive Backend**: Large operations use COSMA, small use OpenBLAS
- **KV Cache**: Avoid recomputing past positions in decode

#### Stage Contracts: Preventing Dimension/Transpose Bugs ✨

**Files**: `src/kernels/AttentionStageContracts.h`, `src/kernels/MPIAttentionKernel.cpp`  
**Tests**: `tests/test_attention_stage_contracts.cpp` (100% pass rate, 7 test cases)

To prevent dimension mismatches and transpose bugs in the complex attention pipeline, MPIAttentionKernel uses **explicit stage contracts** that define PRE and POST conditions between the 5 internal transformation stages.

**The 5 Pipeline Stages**:
1. **Q/K/V Projections**: Linear transformations with weight/bias application
2. **RoPE Application**: Rotary position embedding (shape-invariant)
3. **GQA Replication**: K/V head expansion for Grouped Query Attention
4. **Attention Computation**: Scaled dot-product with softmax
5. **Output Projection**: Final linear transformation

**Contract System**:
```cpp
struct TensorContract {
    ShapeSpec expected_shape;       // e.g., [seq_len, n_heads * head_dim]
    TensorLayout expected_layout;   // RowMajor, HeadInterleaved, Transposed
    TensorSemantic semantic;        // Activation, Weight, Bias, AttentionScores
    std::string name;               // Human-readable identifier
};

struct StageContract {
    std::vector<TensorContract> pre_conditions;   // Expected inputs
    std::vector<TensorContract> post_conditions;  // Guaranteed outputs
    std::string stage_name;
    std::function<bool()> custom_validator;       // Optional validation logic
};
```

**How It Works**:
- Each stage declares what it expects (PRE) and what it produces (POST)
- Contracts are validated at runtime with clear error messages
- Shape specs support dynamic dimensions (e.g., `-1` for variable seq_len)
- Validation happens before execution (fail-fast)
- Errors include expected vs actual shapes, semantic context, and stage information

**Example Contract Violation**:
```
ERROR: Stage 'Q/K/V Projections' input[2] (local_wk): Contract violation!
  Expected: [896 /*out_features*/, 128 /*in_features*/]
  Actual:   [128, 896]
  → Weight tensor is transposed incorrectly
```

**Benefits**:
- ✅ Catches dimension mismatches before execution (not during crash)
- ✅ Clear, actionable error messages with context
- ✅ Documents expected tensor shapes in code
- ✅ Validates both single-rank and multi-rank MPI execution
- ✅ Handles variable-length sequences (dynamic seq_len dimension)
- ✅ Prevents costly debugging sessions tracking down transpose bugs

**Testing**: The contract system is validated via smoke tests that verify:
1. Valid tensor shapes pass validation
2. Invalid tensor shapes trigger clear contract violations
3. All 10 input tensors are properly validated (input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache)
4. Contracts work in both single-rank and multi-rank contexts

See `tests/ATTENTION_STAGE_CONTRACTS_TESTS.md` for full test documentation.

### 8. KV Cache Management ✨ *UPDATED OCTOBER 2025*

**Files**: `src/abstract_pipeline.h`, `src/qwen_pipeline.cpp`, `src/kernels/MPIAttentionKernel.cpp`

The KV cache system stores past key and value tensors to avoid recomputation during autoregressive decode. **As of October 2025**, the MPIAttentionKernel has been updated to properly use KV cache for incremental decode, fixing previous issues where attention was only computed against the current token.

#### KVCacheState Structure

```cpp
struct KVCacheState {
    int max_seq_len;      // Maximum capacity
    int current_pos;      // Current fill position (n_past)
    bool initialized;     // Cache allocation status
    
    // Per-layer storage (distributed across MPI ranks for KV heads)
    std::vector<std::shared_ptr<TensorBase>> key_cache;   
    std::vector<std::shared_ptr<TensorBase>> value_cache;
};
```

#### Lifecycle

**Initialization**: During first prefill
- Allocate `[num_layers, max_seq_len, n_head_kv * head_dim]` tensors (partitioned by KV heads across MPI ranks)
- Set `max_seq_len` from config or CLI `--max-seq-len`
- Mark `initialized = true`

**Prefill**: Write initial sequence
- Store computed K and V for all positions: `[seq_len, n_head_kv * head_dim]`
- Set `current_pos = prefill_length`
- Cache passed to MPIAttentionKernel as inputs[8] (k_cache) and inputs[9] (v_cache)

**Decode**: Append new tokens ✨ *FIXED OCTOBER 2025*
1. Check capacity: `current_pos + 1 <= max_seq_len`
2. MPIAttentionKernel receives cache as inputs[8] and inputs[9]
3. Kernel detects decode mode: `is_decode_mode = (cache present && cache_size > 0)`
4. After RoPE, kernel **appends** new K/V to cache:
   ```cpp
   // Create new cache: [n_past + 1, local_kv_head_dim]
   local_k_cache = TensorFactory::create_simple({cache_seq_len + seq_len, local_kv_head_dim});
   
   // Copy old cache + append new K/V
   memcpy(local_k_cache->data(), k_cache_in->data(), old_cache_bytes);
   memcpy(local_k_cache->data() + old_cache_size, local_k->data(), new_kv_bytes);
   ```
5. Attention computed against **full cache** (not just current token):
   ```cpp
   int attn_seq_len = cache_seq_len + seq_len;  // n_past + 1 for decode
   scores_size = local_heads * seq_len * attn_seq_len;  // [heads, 1, n_past+1]
   
   compute_qk_scores(local_q, local_k_cache, scores, 
                     seq_len, attn_seq_len, ...);  // Q@K_cache^T
   ```
6. Updated cache returned as outputs[1] (k_cache) and outputs[2] (v_cache)
7. Pipeline updates cache state: `k_cache_[layer] = attn_outputs[1]`
8. Increment `current_pos`

#### Key Fixes (October 2025)

**Before** (Bug):
- Attention kernel received KV cache but **didn't use it**
- Attention scores computed as `Q @ K^T` where K is only current token `[1, dim]`
- Result: Wrong attention shape `[heads, 1, 1]` instead of `[heads, 1, n_past+1]`
- Numerical error: Attention only looked at current token, not full context

**After** (Fixed):
- Kernel appends new K/V to cache after RoPE
- `attn_seq_len` tracks full cache length (n_past + seq_len)
- Attention scores computed as `Q @ K_cache^T` where K_cache is `[n_past+1, dim]`
- Result: Correct attention shape `[heads, 1, n_past+1]`
- GQA expansion operates on full cache, not just current K/V
- Updated cache returned to pipeline for next decode step

**Impact**:
- ✅ Fixed size mismatch in parity tests: "expected 896 but got 14" → now 112 elements correct
- ✅ Incremental decode now computes attention over full context
- ✅ KV cache grows correctly: [5] → [6] → [7] → [8] as expected
- ✅ Numerical parity with PyTorch reference improves significantly

#### Kernel Integration

**MPIAttentionKernel Input Format** (10 inputs):
```cpp
auto input       = inputs[0];  // Attention input [seq_len, hidden_size]
auto wq          = inputs[1];  // Q weight
auto wk          = inputs[2];  // K weight  
auto wv          = inputs[3];  // V weight
auto wo          = inputs[4];  // Output weight
auto bq          = inputs[5];  // Q bias
auto bk          = inputs[6];  // K bias
auto bv          = inputs[7];  // V bias
auto k_cache     = inputs[8];  // ✅ KV cache (key) - used in decode mode
auto v_cache     = inputs[9];  // ✅ KV cache (value) - used in decode mode
```

**MPIAttentionKernel Output Format** (3 outputs):
```cpp
outputs[0] = attention_output;  // [seq_len, d_model]
outputs[1] = updated_k_cache;   // ✅ [n_past+seq_len, local_kv_head_dim]
outputs[2] = updated_v_cache;   // ✅ [n_past+seq_len, local_kv_head_dim]
```

**Pipeline Cache Update**:
```cpp
// QwenPipeline::prefill() / decode()
std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out, nullptr, nullptr};
if (!executeKernel("attention", attn_inputs, attn_outputs)) {
    return false;
}

// Update cache with kernel outputs
if (use_kv_cache_ && attn_outputs.size() >= 3) {
    k_cache_[layer_idx] = attn_outputs[1];  // ✅ Updated K cache
    v_cache_[layer_idx] = attn_outputs[2];  // ✅ Updated V cache
}
```

#### Observability

**CLI Flag**: `--kv-stats`
- Prints cache usage and capacity after each operation
- Shows per-layer memory footprint
- Reports fill percentage

**Example Output**:
```
KV Cache Stats:
  Position: 127/512 (24.8%)
  Layers: 24
  Memory per layer: 2.1 MB
  Total KV cache: 50.4 MB
```

**Debug Logging** (when enabled):
```cpp
LOG_DEBUG("[KVCacheReturn] layer=" << layer_index_ 
          << " k_cache_shape=[" << local_k_cache->shape()[0] << "," << local_k_cache->shape()[1] << "]"
          << " attn_seq_len=" << attn_seq_len);
```

#### Testing

**Parity Tests**: `test_parity_framework.cpp`
- Validates KV cache state after prefill
- Compares incremental decode vs full replay
- Ensures `current_pos` consistency
- ✅ **NEW**: Verifies attention scores shape matches PyTorch (e.g., 112 elements for decode step 2)
- ✅ **NEW**: Validates KV cache growth: [5]→[6]→[7]→[8]

**Snapshot Comparison**:
- `ATTENTION_SCORES_layerN`: Now captures `[n_head * seq_len, attn_seq_len]` for incremental decode
- Expected shape for decode step 2 with 14 heads: `[14, 1, 8]` = 112 elements
- Before fix: Only 14 elements (1 per head) - wrong!
- After fix: 112 elements (14 heads × 1 query × 8 keys) - correct!

#### Future Enhancements

**Planned Features**:
- **Eviction Policies**: LRU, sliding window, selective retention
- **Compaction**: Remove unused sequence segments
- **Distribution**: Shard KV cache across MPI ranks (owner-aware) - **currently each rank stores local KV heads**
- **Quantization**: Store K/V in reduced precision (INT8, FP16)

**Pluggable Strategy**:
```cpp
class KVCacheEvictionPolicy {
public:
    virtual void compact(KVCacheState& cache, int min_retained) = 0;
};
```

Integration point: `AbstractPipeline::decode()` invokes policy before cache append.

**Current Architecture Notes**:
- KV cache is **partitioned by KV heads** across MPI ranks (each rank owns subset of KV heads)
- For GQA models (Qwen 2.5): 2 KV heads distributed across ranks
- Cache gathering happens in MPIAttentionKernel via MPI_Allgatherv when expanding for GQA
- This enables memory-efficient distributed inference while maintaining correctness

### 9. GGUF Format & Model Loading

**Files**: `src/model_loader.{h,cpp}`, `src/qwen_pipeline.cpp`

The ModelLoader system handles loading quantized models from GGUF (GPT-Generated Unified Format) files, performing dequantization, and distributing weights across MPI ranks. Understanding this system is critical for debugging weight-related issues and ensuring correct tensor initialization.

#### GGUF Format Overview

**Purpose**: GGUF is a binary format designed by the llama.cpp project for storing quantized LLM weights with metadata.

**File Structure**:
```
┌─────────────────────────────────────┐
│ Magic Number (0x46554747 = "GGUF") │
├─────────────────────────────────────┤
│ Version (uint32)                    │
├─────────────────────────────────────┤
│ Tensor Count (uint64)               │
├─────────────────────────────────────┤
│ Metadata KV Count (uint64)          │
├─────────────────────────────────────┤
│ Metadata Key-Value Pairs            │
│   - Model architecture              │
│   - Hyperparameters (n_layers, etc) │
│   - Tokenizer config                │
│   - Custom metadata                 │
├─────────────────────────────────────┤
│ Tensor Metadata (per tensor)        │
│   - Name (string)                   │
│   - Dimensions (array of uint64)    │
│   - Type (quantization format)      │
│   - Offset (uint64)                 │
├─────────────────────────────────────┤
│ Padding (alignment to 32 bytes)     │
├─────────────────────────────────────┤
│ Tensor Data (binary blobs)          │
│   - Quantized weights               │
│   - Stored in order of metadata     │
└─────────────────────────────────────┘
```

**Supported Quantization Formats**:
- `F32`: Full precision (4 bytes per element)
- `F16`: Half precision (2 bytes per element)
- `Q4_0`: 4-bit quantization (32 values per block + FP16 scale)
- `Q4_1`: 4-bit with min/max (32 values + 2 FP16 parameters)
- `Q5_0`, `Q5_1`: 5-bit variants
- `Q6_K`: 6-bit with K-quants (improved quality)
- `Q8_0`: 8-bit quantization

#### ModelLoader Architecture

**Class Hierarchy**:
```cpp
class ModelLoader {
public:
    explicit ModelLoader(const std::string& path);
    
    // Core loading API
    std::shared_ptr<TensorBase> loadTensor(const std::string& name);
    ModelConfig parseConfig();
    
    // Metadata access
    bool hasTensor(const std::string& name) const;
    std::vector<std::string> getTensorNames() const;
    
private:
    std::string file_path_;
    std::ifstream file_stream_;
    std::unordered_map<std::string, TensorInfo> tensor_info_map_;
    ModelConfig config_;
};
```

**Loading Pipeline**:
1. **Open GGUF file** and parse header (magic, version, counts)
2. **Parse metadata** key-value pairs into `ModelConfig`
3. **Parse tensor metadata** (names, shapes, types, offsets)
4. **On-demand tensor loading** via `loadTensor()`:
   - Seek to tensor offset in file
   - Read quantized binary data
   - **Dequantize** to FP32 format
   - Return as `SimpleTensor` or `COSMATensor`

#### Dequantization Process

**Why Dequantization?**
- GGUF stores weights in compressed formats (4-bit, 6-bit, etc.) to reduce file size
- Llaminar kernels expect FP32 data for matmuls (OpenBLAS, COSMA)
- Dequantization happens **once at load time**, not during inference

**Dequantization Example (Q4_0)**:
```cpp
// Q4_0 format: 32 values packed into 16 bytes + 1 FP16 scale
struct BlockQ4_0 {
    uint8_t values[16];  // 32 4-bit values packed
    float16_t scale;     // Dequantization scale factor
};

float dequantize_q4_0(const BlockQ4_0* block, int index) {
    int byte_idx = index / 2;
    int nibble_idx = index % 2;
    
    // Extract 4-bit value (0-15)
    uint8_t quant_val = (block->values[byte_idx] >> (nibble_idx * 4)) & 0x0F;
    
    // Dequantize: subtract 8 to center around 0, then scale
    float dequant_val = (float)(quant_val - 8) * fp16_to_fp32(block->scale);
    
    return dequant_val;
}
```

**Performance Characteristics**:
- Dequantization is CPU-bound but only happens **once** per tensor
- Typical dequantization time: ~50-200ms per large weight tensor
- Total model load time: 1-5 seconds for 0.5B parameter model

**Validation Flags**:
- `LLAMINAR_DEQUANT_STATS=1`: Log per-tensor statistics (min/max/mean/samples)
- `LLAMINAR_DEQUANT_ANOMALIES=1`: Warn on NaN/Inf/huge values (safety check)

**Example Output with Stats**:
```
[DEQUANT] blk.0.attn_q.weight: shape=[896,896], quant=Q6_K
  min=-0.2415, max=0.2891, mean=0.0001, mean_abs=0.0234
  samples: [0.0151, -0.0234, 0.0089, ...]
```

#### Weight Tensor Naming Convention

GGUF uses hierarchical naming for tensors, following llama.cpp conventions:

**Embedding Layer**:
- `token_embd.weight`: Token embedding matrix `[vocab_size, hidden_size]`

**Transformer Blocks** (per-layer, `blk.{layer_idx}.`):
- `attn_norm.weight`: Pre-attention RMSNorm gamma `[hidden_size]`
- `attn_q.weight`: Query projection `[n_head * head_dim, hidden_size]` = `[hidden_size, hidden_size]` (for non-GQA)
- `attn_k.weight`: Key projection `[n_head_kv * head_dim, hidden_size]`
- `attn_v.weight`: Value projection `[n_head_kv * head_dim, hidden_size]`
- `attn_output.weight`: Output projection `[hidden_size, hidden_size]`
- **`attn_q.bias`**: ⚠️ Query projection bias `[n_head * head_dim]` **CRITICAL**
- **`attn_k.bias`**: Key projection bias `[n_head_kv * head_dim]` **CRITICAL**
- **`attn_v.bias`**: Value projection bias `[n_head_kv * head_dim]` **CRITICAL**
- `ffn_norm.weight`: Pre-FFN RMSNorm gamma `[hidden_size]`
- `ffn_gate.weight`: FFN gate projection `[d_ff, hidden_size]`
- `ffn_up.weight`: FFN up projection `[d_ff, hidden_size]`
- `ffn_down.weight`: FFN down projection `[hidden_size, d_ff]`

**Output Layer**:
- `output_norm.weight`: Final RMSNorm gamma `[hidden_size]`
- `output.weight`: LM head projection `[vocab_size, hidden_size]`

**Weight Shape Convention** (PyTorch nn.Linear format):
- Weights stored as `[out_features, in_features]`
- For matmul: `output = input @ weight.T` (transpose required)
- This is why `transpose_B=true` in kernel matmul calls

#### Bias Tensor Loading (CRITICAL)

**⚠️ Historical Bug**: Llaminar originally **only loaded weights**, never bias tensors, causing massive divergence.

**Root Cause of 79.9x Divergence**:
- PyTorch `nn.Linear` layers include bias: `output = input @ weight.T + bias`
- GGUF stores bias tensors: `blk.{layer}.attn_{q,k,v}.bias`
- Llaminar was loading weights but **ignoring biases**
- **Large bias values** (e.g., Q bias range `[-79.0, +47.75]`) caused huge errors

**Example Large Bias Values**:
```python
# Layer 0 Q projection bias (first 15 values)
[-0.0150, 0.0255, -0.1035, -0.1357, -14.4375,  # ← Large!
  0.2656, 0.3242, 0.1240, -15.4375, -34.0,     # ← Very large!
 -15.1250, 6.8125, -0.4785, 7.9062, 0.6289]

# Statistics:
#   - 94 out of 896 Q bias values > ±10.0
#   - Range: [-79.0, +47.75]
#   - These bias values are ESSENTIAL for correct projections
```

**Corrected Loading Code** (`src/qwen_pipeline.cpp`):
```cpp
// Load weights (original code)
auto wq = loader.loadTensor(prefix + "attn_q.weight");
auto wk = loader.loadTensor(prefix + "attn_k.weight");
auto wv = loader.loadTensor(prefix + "attn_v.weight");
auto wo = loader.loadTensor(prefix + "attn_output.weight");

// ✅ NOW: Load biases (added fix)
auto bq = loader.loadTensor(prefix + "attn_q.bias");
auto bk = loader.loadTensor(prefix + "attn_k.bias");
auto bv = loader.loadTensor(prefix + "attn_v.bias");

// Store in ModelWeights structure
weights.wq.push_back(wq);
weights.wk.push_back(wk);
weights.wv.push_back(wv);
weights.wo.push_back(wo);
weights.bq.push_back(bq);  // ✅ NEW
weights.bk.push_back(bk);  // ✅ NEW
weights.bv.push_back(bv);  // ✅ NEW
```

**ModelWeights Structure Update**:
```cpp
struct ModelWeights {
    std::shared_ptr<TensorBase> token_embedding;
    std::vector<std::shared_ptr<TensorBase>> attn_norm_weight;
    
    // Attention weights
    std::vector<std::shared_ptr<TensorBase>> wq, wk, wv, wo;
    
    // ✅ Attention biases (CRITICAL - added October 2025)
    std::vector<std::shared_ptr<TensorBase>> bq, bk, bv;
    
    // FFN weights
    std::vector<std::shared_ptr<TensorBase>> ffn_norm_weight;
    std::vector<std::shared_ptr<TensorBase>> w_gate, w_up, w_down;
    
    // Output weights
    std::shared_ptr<TensorBase> output_norm_weight;
    std::shared_ptr<TensorBase> lm_head;
};
```

**Bias Application in Kernels** (`src/kernels/MPIAttentionKernel.cpp`):
```cpp
// After matrix multiplication: OUT = input @ W.T
// ✅ Apply bias (broadcast across sequence dimension)
if (bias && bias->data()) {
    const float* bias_data = bias->data();
    float* out_data = OUT->data();
    
    for (size_t row = 0; row < seq_len; ++row) {
        for (size_t col = 0; col < feature_dim; ++col) {
            out_data[row * feature_dim + col] += bias_data[col];
        }
    }
}
```

**Kernel Input Order** (MPIAttentionKernel - 10 inputs total):
```cpp
auto input       = inputs[0];  // Attention input [seq_len, hidden_size]
auto wq          = inputs[1];  // Q weight [n_head*head_dim, hidden_size]
auto wk          = inputs[2];  // K weight [n_head_kv*head_dim, hidden_size]
auto wv          = inputs[3];  // V weight [n_head_kv*head_dim, hidden_size]
auto wo          = inputs[4];  // Output weight [hidden_size, n_head*head_dim]
auto bq          = inputs[5];  // ✅ Q bias [n_head*head_dim]
auto bk          = inputs[6];  // ✅ K bias [n_head_kv*head_dim]
auto bv          = inputs[7];  // ✅ V bias [n_head_kv*head_dim]
auto k_cache     = inputs[8];  // KV cache (key)
auto v_cache     = inputs[9];  // KV cache (value)
```

#### Weight Distribution for MPI

**Single-Rank Execution**:
- All weights/biases used directly without distribution
- `local_wq = global_wq` (no slicing)
- `local_bq = global_bq` (no slicing)

**Multi-Rank Execution** (Tensor Parallel):
- **Weights**: Sliced by heads (column partition)
  - Rank 0 gets heads 0-6, Rank 1 gets heads 7-13 (for 14-head model)
  - Each rank holds slice: `[local_heads * head_dim, hidden_size]`
- **Biases**: Sliced correspondingly
  - Each rank gets bias slice: `[local_heads * head_dim]`
  - Bias values only for locally-owned heads

**Distribution Example** (2 ranks, 14 Q heads):
```cpp
// Rank 0:
//   local_wq: [448, 896]  (7 heads * 64 dim = 448)
//   local_bq: [448]       (corresponding bias slice)

// Rank 1:
//   local_wq: [448, 896]  (7 heads * 64 dim = 448)
//   local_bq: [448]       (corresponding bias slice)

// After projection + bias, MPI_Allreduce assembles full [seq_len, 896] output
```

#### Embedding Tensor Special Case

**Token Embedding** (`token_embd.weight`):
- Shape: `[vocab_size, hidden_size]` (e.g., `[151936, 896]`)
- **Largest tensor in model** (typically 100-500 MB)
- **Not quantized** in most GGUF files (stored as F16 or F32)
- Loaded once and **replicated** across all MPI ranks
- Accessed via index lookup: `embedding[token_id]` → `[hidden_size]` vector

**Embedding Kernel**:
```cpp
// MPIEmbeddingKernel
// Input: token IDs [seq_len]
// Output: embeddings [seq_len, hidden_size]
for (int i = 0; i < seq_len; ++i) {
    int token_id = token_ids[i];
    memcpy(output + i * hidden_size, 
           embedding_weights + token_id * hidden_size,
           hidden_size * sizeof(float));
}
```

#### RMSNorm Gamma Weights

**RMSNorm Weights** (`.weight` suffix, no bias):
- Shape: `[hidden_size]` (1D vector)
- Applied element-wise after normalization
- **No bias term** in RMSNorm (unlike LayerNorm)

**RMSNorm Application**:
```cpp
// Step 1: Compute RMS (root mean square)
float rms = 0.0f;
for (int i = 0; i < hidden_size; ++i) {
    rms += input[i] * input[i];
}
rms = sqrt(rms / hidden_size + epsilon);

// Step 2: Normalize and apply gamma weight
for (int i = 0; i < hidden_size; ++i) {
    output[i] = (input[i] / rms) * gamma[i];
}
```

**Important**: RMSNorm gamma is multiplicative scale, **not additive bias**.

#### ModelConfig Parsing

**Metadata Extraction**:
```cpp
ModelConfig parseConfig() {
    ModelConfig config;
    
    // Read from GGUF metadata KV pairs
    config.n_layers = getMetadataInt("llama.block_count");
    config.n_heads = getMetadataInt("llama.attention.head_count");
    config.n_head_kv = getMetadataInt("llama.attention.head_count_kv");
    config.hidden_size = getMetadataInt("llama.embedding_length");
    config.d_ff = getMetadataInt("llama.feed_forward_length");
    config.vocab_size = getMetadataInt("llama.vocab_size");
    config.rope_freq_base = getMetadataFloat("llama.rope.freq_base", 10000.0f);
    config.context_length = getMetadataInt("llama.context_length");
    
    return config;
}
```

**Validation**:
- Check for required metadata keys
- Validate dimension consistency (e.g., `hidden_size % n_heads == 0`)
- Ensure all expected tensors are present in file

#### Error Handling & Diagnostics

**Missing Tensor Detection**:
```cpp
auto tensor = loader.loadTensor("blk.0.attn_q.bias");
if (!tensor) {
    throw std::runtime_error("Required tensor 'blk.0.attn_q.bias' not found in GGUF");
}
```

**Shape Validation**:
```cpp
if (wq->shape()[0] != n_head * head_dim || wq->shape()[1] != hidden_size) {
    LOG_ERROR("Q weight shape mismatch! Expected [" 
              << (n_head * head_dim) << ", " << hidden_size << "], got ["
              << wq->shape()[0] << ", " << wq->shape()[1] << "]");
    return false;
}
```

**Dequantization Anomalies**:
```bash
# Enable diagnostics
export LLAMINAR_DEQUANT_ANOMALIES=1

# Output on anomaly:
[WARN] blk.5.ffn_gate.weight contains 3 NaN values after dequantization!
[WARN] blk.12.attn_k.weight: max_abs=1e8 (suspiciously large)
```

#### Best Practices for Weight Loading

1. **Always load biases** if present in GGUF (check with `hasTensor()`)
2. **Validate shapes** after loading before passing to kernels
3. **Enable dequant stats** during development to catch corruption
4. **Log tensor names** when debugging load failures
5. **Check tensor types** - not all tensors are quantized (embeddings often F16)
6. **Handle missing tensors gracefully** - some models may omit biases

#### Common Pitfalls

❌ **Forgetting to load biases** → Massive divergence (79.9x error)
❌ **Incorrect shape assumptions** → Matmul dimension mismatches
❌ **Skipping transpose** → Wrong orientation for nn.Linear weights
❌ **Ignoring dequant anomalies** → Silent NaN/Inf propagation
❌ **Hardcoded tensor names** → Breaks on model architecture changes

✅ **Use `loadTensor()` for all weights AND biases**
✅ **Validate shapes immediately after loading**
✅ **Enable diagnostics during initial model testing**
✅ **Follow GGUF naming conventions from llama.cpp**
✅ **Test with multiple model sizes** to catch dimension bugs

### 10. Environment & Observability

**Files**: `src/debug_env.{h,cpp}`

Centralized environment configuration system eliminating hot-path `std::getenv` calls with structured, typed snapshots.

#### Debug Environment Snapshot

**Access Pattern**:
```cpp
const auto& env = debugEnv();
if (env.attention.micro_trace && rank == 0) {
    LOG_TRACE("Attention Q shape: " << q_shape);
}
```

**Configuration Groups**:
- `attention`: Attention kernel diagnostics and tracing
- `baseline`: Reference execution and comparison
- `cosma`: COSMA-specific controls and validation
- `adaptive`: Backend selection overrides
- `pipeline`: Pipeline-level instrumentation
- `linear`: Linear kernel diagnostics
- `embedding`: Embedding layer controls
- `layer_capture`: Per-layer forensic capture
- `prefill_debug`: Prefill diagnostics and validation

#### Design Principles

**Mandatory Rules**:
1. **No Hot-Path getenv**: All environment variables parsed once at initialization
2. **Typed Fields**: Boolean, integer, and string fields with proper defaults
3. **Single Registry**: All flags documented in `DebugEnv.h`
4. **Lazy Initialization**: Snapshot created on first `debugEnv()` call

**Migration Guidance**:
- New flags MUST be added to appropriate group in `DebugEnv.h`
- Existing raw `std::getenv` calls should be migrated on file touch
- Experimental flags staged in snapshot early to prevent drift

#### Diagnostic Capabilities

**Per-Stage Validation**:
- Optional reference recomputation for correctness checking
- Relative L2 error computation
- Top sample value logging
- Per-layer diff summaries

**Forensic Modes**:
- Row capture for RMSNorm intermediate values
- FFN gate/up/down activation dumps
- COSMA orientation and reconstruction debugging
- Attention QKV tensor inspection

**Performance Instrumentation**:
- Per-operation timing (via performance counters)
- Adaptive backend decision logging
- GEMM statistics aggregation
- Post-run summary reports

#### Common Environment Variables

| Variable | Purpose | Group | Type |
|----------|---------|-------|------|
| `LLAMINAR_ATTN_MICRO_TRACE` | Detailed attention tracing | attention | bool |
| `ADAPTIVE_DISABLE_COSMA` | Force OpenBLAS path | adaptive | bool |
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Sequence length threshold | cosma | int |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Tile validation size | cosma | int |
| `LLAMINAR_DEQUANT_STATS` | Dequantization statistics | pipeline | bool |
| `LLAMINAR_COSMA_LOG_LEVEL` | COSMA log verbosity | cosma | string |

#### Usage Examples

```cpp
// Attention tracing
const auto& env = debugEnv();
if (env.attention.micro_trace) {
    LOG_TRACE("QK product shape: " << qk_shape);
}

// Backend override
if (env.adaptive.disable_cosma) {
    return MatMulBackend::MULTI_THREADED_OPENBLAS;
}

// Validation control
if (env.cosma.validate_tile > 0) {
    performTileValidation(env.cosma.validate_tile);
}
```

#### Future Extensions

- **Hot Reload**: Dynamic snapshot refresh without restart
- **Remote Control**: External configuration channel
- **Profiling Modes**: Predefined diagnostic profiles
- **Conditional Compilation**: Debug-only snapshot fields

### 10. Model Loading & Weight Management

**Files**: `src/model_loader.{h,cpp}`, `src/qwen_pipeline_adapter.cpp`, `src/llama_pipeline_adapter.cpp`

Model loading system parses GGUF format files and instantiates architecture-specific weight implementations.

#### Loading Pipeline

**1. Format Detection**:
- Read GGUF header and magic bytes
- Validate file version compatibility
- Parse metadata section

**2. Configuration Extraction**:
- Model dimensions (vocab size, hidden size, layer count)
- Architecture parameters (attention heads, MLP ratio)
- Tokenizer vocabulary
- Create `ModelConfig` structure

**3. Tensor Enumeration**:
- Scan tensor directory
- Map tensor names to pipeline weight keys
- Record shapes and data types

**4. Quantization Handling**:
- Detect quantization format (Q4_K, Q6_K, etc.)
- Dequantize to FP32 on load (per-tensor)
- Optional statistics logging via `LLAMINAR_DEQUANT_STATS`
- Anomaly detection via `LLAMINAR_DEQUANT_ANOMALIES`

**5. MPI Distribution**:
- Rank 0 reads GGUF file
- Broadcast metadata to all ranks
- For distributed tensors: partition and distribute slices
- For replicated tensors: broadcast full tensor

**6. Tensor Instantiation**:
- Select tensor type (SimpleTensor vs COSMATensor) based on size
- Apply environment-driven selection policy
- Populate weight containers

#### IModelWeights Implementations

**QwenModelWeights**:
```cpp
class QwenModelWeights : public IModelWeights {
    std::shared_ptr<TensorBase> getEmbedding(int token_id) const override;
    std::shared_ptr<TensorBase> getWeight(const std::string& key) const override;
    bool hasWeight(const std::string& key) const override;
    ModelConfig config() const override;
private:
    std::unordered_map<std::string, std::shared_ptr<TensorBase>> weights_;
    std::shared_ptr<TensorBase> embedding_table_;
};
```

**LlamaModelWeights**:
- Currently delegates to QwenModelWeights (similar architecture)
- Future: Handle LLaMA-specific weight layouts

#### Weight Naming Conventions

**Standard Keys**:
- Embedding: `token_embd.weight`
- QKV projections: `layers.{L}.attention.{q,k,v}_weight`
- Attention output: `layers.{L}.attention.wo_weight`
- MLP gates: `layers.{L}.mlp.{gate,up,down}_proj`
- Layer norms: `layers.{L}.{attn,mlp}_norm.weight`
- Output projection: `output.weight`

#### Quantization Support

**Supported Formats**:
- **Q4_K**: 4-bit with block-wise quantization
- **Q6_K**: 6-bit with block-wise quantization  
- **FP32**: Full precision (no quantization)

**Dequantization**:
- Performed on load (eager dequantization)
- Per-tensor statistics logged if enabled
- Anomaly detection: NaN, Inf, extreme values

**Statistics Example**:
```
Dequant [layers.0.attention.q_weight]:
  Min: -0.234, Max: 0.198
  Mean: 0.003, Std: 0.067
  Samples: [-0.012, 0.045, -0.023, ...]
```

#### Layout Adaptation

**Repacker**: Performs layout transformations for fused kernels
- Contiguous block ordering for QKV concatenation
- COSMA-compatible memory layouts
- Alignment requirements for vectorization

#### Future Enhancements

- **Lazy Dequantization**: Keep quantized format, dequant on-the-fly
- **Mixed Precision**: FP16/BF16 activation buffers
- **Streaming Load**: Memory-mapped incremental loading
- **Weight Sharding**: Automatic partitioning for large models

### 11. Testing Infrastructure

**Files**: `tests/test_*.cpp`, `tests/parity_*.cpp`

Comprehensive test suite validating pipeline correctness, backend selection, and distributed execution.

#### Test Categories

**Pipeline Tests**:
- **`test_pipeline_factory.cpp`**: Factory registration and creation
- **`test_abstract_pipeline_parity.cpp`**: Prefill vs incremental decode equivalence
- **`test_qwen_pipeline.cpp`**: Qwen-specific pipeline functionality (4 test cases)

**Backend & COSMA Tests**:
- **`test_cosma_prefill_*.cpp`**: Fused COSMA correctness and statistics
- **`test_adaptive_matmul*.cpp`**: Backend decision logic validation
- **`test_cosma.cpp`**: Core COSMA integration

**Primitive Kernel Tests**:
- **`test_rmsnorm_*.cpp`**: RMSNorm parity and edge cases
- **`test_attention_*.cpp`**: Attention mechanism validation
- **`test_rope_*.cpp`**: RoPE positional encoding
- **`test_softmax_*.cpp`**: Softmax numerical stability

**Distributed Execution Tests**:
- **`test_tp_*.cpp`**: Tensor partition correctness
- **`test_mlp_tp_parity.cpp`**: MLP distributed parity
- **`test_mpi_linear_kernel.cpp`**: MPILinearKernel validation

**Infrastructure Tests**:
- **`test_basic.cpp`**: MPI initialization and basic functionality
- **`test_numa.cpp`**: NUMA topology detection and affinity
- **`test_kv_cache_growth*.cpp`**: KV cache capacity management

#### Removed Tests

Historical tests no longer needed after architecture refactor:
- ❌ `test_graph.cpp`: Generic compute graph (removed architecture)
- ❌ `LinearKernelTest`: Legacy non-MPI linear kernel (retired)

#### Parity Testing Framework

**Purpose**: Ensure mathematical equivalence across execution paths

**Key Parity Tests**:
1. **Prefill vs Incremental Decode**: 
   - Full sequence prefill should match token-by-token decode
   - KV cache state must be identical
   - Logits must match within numerical tolerance

2. **COSMA vs OpenBLAS**:
   - Large operations should produce equivalent results
   - Relative L2 error < 1e-4 threshold
   - Maximum absolute difference tracking

3. **Distributed vs Single-Rank**:
   - Multi-rank execution matches single-rank reference
   - Gather/reduction correctness
   - No data loss or corruption

**Example Parity Check**:
```cpp
TEST_CASE("Prefill vs Incremental Parity") {
    // Full sequence prefill
    pipeline->prefill(all_tokens, weights, prefill_ctx);
    auto prefill_logits = pipeline->getLogits();
    
    // Incremental decode
    pipeline->prefill({all_tokens[0]}, weights, decode_ctx);
    for (int i = 1; i < all_tokens.size(); ++i) {
        pipeline->decode(all_tokens[i], weights, decode_ctx);
    }
    auto decode_logits = pipeline->getLogits();
    
    // Validate equivalence
    float rel_error = computeRelativeL2(prefill_logits, decode_logits);
    REQUIRE(rel_error < 1e-4);
}
```

#### Test Execution

**Run All Tests**:
```bash
ctest --test-dir build --output-on-failure --parallel
```

**Core Tests Only** (recommended during development):
```bash
ctest --test-dir build -R "^(BasicTest|PipelineFactoryTest|QwenPipelineTest)$"
```

**With Verbose Output**:
```bash
ctest --test-dir build --output-on-failure --verbose
```

**MPI Tests**:
```bash
mpirun -np 2 ./build/test_abstract_pipeline_parity
```

#### Current Test Status

**Passing Tests**:
- ✅ BasicTest (MPI initialization)
- ✅ PipelineFactoryTest (factory mechanics)  
- ✅ QwenPipelineTest (3/4 subtests)

**Known Issues**:
- ⚠️ QwenPipelineTest.ValidationTests (pre-existing precision issue, unrelated to refactor)
- ⚠️ Some COSMA tests have numerical precision edge cases

#### Testing Best Practices

1. **Always run tests after kernel changes**
2. **Use parity tests to validate optimizations**
3. **Check both single-rank and multi-rank execution**
4. **Verify KV cache state consistency**
5. **Enable validation flags during development** (`LLAMINAR_COSMA_VALIDATE_TILE`, etc.)
6. **Disable heavy validation for benchmarking**

### 12. Performance Strategy Summary

Empirically-tuned backend selection based on operation characteristics and execution phase.

#### Backend Selection by Phase

| Phase | Sequence Length | Matrix Dims | Backend Policy | Rationale |
|-------|----------------|-------------|----------------|-----------|
| **Prefill (short)** | < 4K tokens | Many small/medium | OpenBLAS (single/multi-thread) | COSMA overhead not amortized |
| **Prefill (large)** | ≥ 4K tokens | seq_len × hidden → projections | COSMA | Collective reuse + fused QKV |
| **Decode** | 1 token | 1 × hidden → projections | OpenBLAS single-thread | Min latency, avoid collectives |
| **FFN (large prefill)** | ≥ 4K tokens | seq_len × hidden → d_ff | COSMA candidate | Throughput scaling |
| **LM Head** | Any | seq_len × hidden → vocab | Local (policy TBD) | Avoid huge all-gather unless 2D shard |

#### Empirical Performance Data

**Small Operations (< 8K elements)**:
- OpenBLAS single-threaded: **134x faster** than COSMA for 1×896×896
- Communication overhead dominates COSMA performance
- Recommendation: Always use local OpenBLAS

**Medium Operations (8K - 8M elements)**:
- OpenBLAS multi-threaded competitive for < 64 tokens
- COSMA overhead still significant
- Recommendation: Multi-threaded OpenBLAS

**Large Operations (≥ 8M elements)**:
- COSMA becomes competitive at ≥ 8K tokens
- **COSMA 3.6x faster** at 64K tokens vs single-rank OpenBLAS
- Distributed memory bandwidth advantage
- Recommendation: COSMA for large prefill

#### Threading Strategy

**Small Operations (< 8K elements)**:
```cpp
openblas_set_num_threads(1);  // Minimize overhead
```

**Medium Operations (8K - 1M elements)**:
```cpp
openblas_set_num_threads(omp_get_max_threads());  // Full socket
```

**Large Distributed Operations**:
```cpp
openblas_set_num_threads(cores_per_numa_node);  // Per-rank threading
// + COSMA distributed execution
```

#### Memory Considerations

**COSMA Buffer Allocation**:
- Soft limit: 2048 MB per rank (configurable via `LLAMINAR_COSMA_MAX_RESIDENT_MB`)
- Fallback to OpenBLAS if allocation exceeds budget
- Prevents memory exhaustion on large models

**Tensor Type Selection**:
- **SimpleTensor**: < 256×256 elements or latency-critical
- **COSMATensor**: ≥ 256×256 and distributed execution

#### Optimization Priorities

1. **Latency** (Decode): Single-threaded OpenBLAS, minimal overhead
2. **Throughput** (Large Prefill): COSMA distributed, fused kernels
3. **Memory Efficiency**: Adaptive tensor types, buffer reuse
4. **Correctness**: Parity validation, fallback paths

### 13. Tensor Sharding & Future Roadmap

Current distributed execution capabilities and planned enhancements for scaling to larger models and longer contexts.

#### Current State (1D Column Sharding)

**Weight Distribution**:
- Linear weights partitioned by columns across MPI ranks
- Each rank holds `[hidden_size, proj_size / num_ranks]` slice
- Activations replicated across all ranks

**Communication Pattern**:
- Local GEMM on weight slice
- `MPI_Allgatherv` to assemble full output
- Works well for moderate model sizes

**Limitations**:
- Activations fully replicated (memory scaling bottleneck)
- All-gather communication cost grows with output size
- KV cache duplicated across ranks

#### Planned Enhancements

**Phase 1: Activation Micro-Sharding**
- **Goal**: Reduce per-rank memory for very large batch prefill
- **Approach**: Partition activation tensors across sequence dimension
- **Benefit**: Linear memory scaling with rank count
- **Challenges**: Attention computation requires gather or ring-reduce

**Phase 2: 2D Block-Cyclic Distribution**
- **Goal**: Eliminate all-gather for extreme vocab or d_ff expansions
- **Approach**: Align 2D tensor distribution with COSMA tiling strategy
- **Benefit**: Zero-copy distributed matmul without post-gather
- **Integration**: Seamless with COSMA's `BlockCyclicMatrix` layout

**Phase 3: Distributed KV Cache**
- **Goal**: Shard past sequence to reduce memory duplication
- **Components**:
  - Owner-aware KV cache distribution
  - Distributed attention softmax (ring-reduce pattern)
  - Partial sequence ownership per rank
- **Benefit**: Support longer contexts with same memory budget

**Phase 4: Mixed Precision Execution**
- **Goal**: Reduce activation memory footprint by 50%
- **Approach**: 
  - FP16/BF16 activation buffers
  - On-the-fly dequantization for matmuls
  - Selective FP32 accumulation for numerical stability
- **Benefit**: 2x memory reduction, potential speedup on modern hardware

#### Implementation Strategy

**Incremental Rollout**:
1. Implement and validate each phase independently
2. Maintain backward compatibility with existing 1D path
3. Add environment flags to enable/disable features
4. Comprehensive parity testing at each phase

**Environment Controls** (Future):
- `LLAMINAR_ACTIVATION_SHARDING={none,1d,2d}`: Activation distribution mode
- `LLAMINAR_KV_CACHE_DISTRIBUTED=1`: Enable distributed KV cache
- `LLAMINAR_MIXED_PRECISION={fp32,fp16,bf16}`: Activation precision
- `LLAMINAR_2D_SHARD_THRESHOLD=<size>`: Minimum size for 2D sharding

#### Research Directions

**Adaptive Sharding**:
- Automatically select 1D vs 2D based on model dimensions
- Hybrid strategies for different layers (e.g., 2D for MLP, 1D for attention)

**Memory-Optimal Schedules**:
- Recompute vs cache trade-offs for activations
- Gradient checkpointing patterns (if extending to fine-tuning)

**Communication Optimization**:
- Overlapped computation and communication
- Hierarchical reduction trees for large rank counts
- NCCL/RCCL integration for GPU backends

### 14. Additional Future Enhancements

Beyond tensor sharding, several high-impact features are planned for production deployment and advanced use cases.

#### Runtime Reconfiguration

**Dynamic Environment Snapshot**:
- Reload `debugEnv()` snapshot without process restart
- Hot-swap backend policies during long-running inference
- Remote control channel for runtime configuration

**Use Cases**:
- A/B testing different backend strategies
- Adaptive optimization based on workload
- Debug flag toggle without restart

**Implementation**:
```cpp
// Future API
debugEnv().reload();
debugEnv().setRemoteChannel("tcp://controller:5555");
```

#### Adaptive Decode Micro-Batching

**Goal**: Process multiple next-token candidates in parallel

**Approach**:
- Beam search or speculative decoding
- Batch multiple decode operations
- Latency guardrails to prevent slowdown

**Challenges**:
- KV cache management for multiple hypotheses
- Memory overhead for parallel paths
- Efficient pruning and selection

#### Streaming KV Cache Management

**Compaction Strategies**:
- **LRU**: Evict least-recently-used sequence segments
- **Sliding Window**: Fixed-size context window
- **Selective Retention**: Keep important tokens (e.g., system prompt)

**Eviction Policies**:
```cpp
class StreamingKVCache {
    void compact(int target_size);
    void evictLRU(int num_tokens);
    void slidingWindow(int window_size);
    void selectiveRetain(const std::vector<int>& important_positions);
};
```

**Integration**: Pluggable policy at `AbstractPipeline::decode()` level

#### Quantized Fused Prefill

**Goal**: Direct dequantization into COSMA layout

**Current**: 
1. Load quantized weights → dequantize to FP32 → SimpleTensor
2. Convert to COSMATensor for COSMA operations

**Optimized**:
1. Load quantized weights → direct dequant into COSMA layout
2. Skip intermediate FP32 buffer allocation
3. Fuse dequant kernel with COSMA distribution

**Benefits**:
- 50%+ memory reduction during load
- Faster initialization
- Reduced allocation pressure

#### GPU Backend Support

**Multi-Backend Architecture**:
- Unified `TensorBase` interface supports CPU and GPU
- CUDA/HIP kernels for attention, matmul, softmax
- NCCL for GPU-to-GPU communication
- Hybrid CPU/GPU execution

**Challenges**:
- Unified memory management
- CPU ↔ GPU data transfer overhead
- NUMA awareness for PCIe topology

#### Production Deployment Features

**Model Serving**:
- gRPC/REST API for inference requests
- Request batching and queue management
- Dynamic model loading/unloading
- Multi-tenancy support

**Observability**:
- Prometheus metrics export
- OpenTelemetry tracing integration
- Performance profiling hooks
- Health check endpoints

**Fault Tolerance**:
- Checkpoint/resume for long sequences
- Rank failure recovery (MPI fault tolerance)
- Graceful degradation on resource exhaustion

#### Research Integration

**Experimental Features**:
- Flash Attention integration
- PagedAttention for KV cache
- Mixture-of-Experts (MoE) support
- Continuous batching (Orca-style)

**Plugin System**:
```cpp
class InferencePlugin {
    virtual bool shouldIntercept(const Operation& op) = 0;
    virtual bool execute(const Operation& op) = 0;
};

// Register custom optimizations
PipelineFactory::registerPlugin(std::make_unique<FlashAttentionPlugin>());
```

---

### 15. Parity Test Framework Integration ✨

**Overview**: The parity test framework provides comprehensive snapshot capture and comparison capabilities for validating pipeline execution correctness. It's now deeply integrated into the core pipeline architecture for automatic, zero-overhead parity testing.

#### Architecture

**Core Components**:

1. **PipelineStage Enum** (`src/pipeline_stages.h`)
   - Standardized 22-stage enumeration covering all transformer operations
   - Shared between production code and tests
   - Stages: EMBEDDING, ATTENTION_NORM, QKV_PROJECTION, ROPE_APPLICATION, ATTENTION_SCORES, ATTENTION_SOFTMAX, ATTENTION_CONTEXT, ATTENTION_OUTPUT, ATTENTION_RESIDUAL, FFN_NORM, FFN_GATE, FFN_UP, FFN_SWIGLU, FFN_DOWN, FFN_RESIDUAL, FINAL_NORM, LM_HEAD, CUSTOM
   - Utility functions: `stage_to_string()`, `string_to_stage()` (inline, zero overhead)

2. **Parity Hooks** (`src/parity_hooks.h/cpp`)
   - Production-safe interface with default no-op implementation
   - Environment-driven activation via `LLAMINAR_PARITY_CAPTURE`
   - Tests provide real implementation via `parity_test_framework.cpp`
   - Zero overhead when disabled (functions inline to empty)

3. **Pipeline Integration**
   - **AbstractPipeline**: Virtual `captureStageSnapshot()` and `isParityEnabled()` methods
   - **PipelineBase**: Convenience `captureIfEnabled()` helpers with rank filtering
   - **QwenPipeline**: 8 strategic capture points at key computation stages

#### Capture Points in QwenPipeline

**Prefill/Decode Path** (automatic capture when `LLAMINAR_PARITY_CAPTURE=1`):

| Stage | Location | Description |
|-------|----------|-------------|
| `EMBEDDING` | After token embedding | Input to first transformer layer |
| `ATTENTION_NORM` | Pre-attention RMSNorm | Input to QKV projection |
| `ATTENTION_OUTPUT` | After W_o projection | Attention block output |
| `ATTENTION_RESIDUAL` | After attention residual add | Input to FFN block |
| `FFN_NORM` | Pre-FFN RMSNorm | Input to gate/up projections |
| `FFN_DOWN` | After down projection | FFN block output |
| `FFN_RESIDUAL` | After FFN residual add | Output of transformer layer |
| `FINAL_NORM` | After final RMSNorm | Input to LM head |
| `LM_HEAD` | Language model head output | Final logits |

**Key Design Features**:
- **Rank 0 Only**: Captures happen on rank 0 to avoid MPI duplication
- **Automatic Shape Extraction**: Captures sequence length and feature dimension from tensors
- **Layer-Aware**: Each capture includes layer index (or -1 for non-layer stages)
- **Zero Overhead When Disabled**: `isParityEnabled()` check inlines to false

#### Usage Patterns

**Basic Parity Testing**:
```bash
# Enable automatic snapshot capture
export LLAMINAR_PARITY_CAPTURE=1

# Run Llaminar inference (captures will be stored in SnapshotRegistry)
mpirun -np 2 ./build/llaminar -m model.gguf -v

# Compare with reference implementation in test
./build/test_parity_framework
```

**Test Integration**:
```cpp
#include "parity_test_framework.h"

TEST(ParityTest, QwenPrefillVsReference) {
    // Clear previous captures
    parity::SnapshotRegistry::instance().clear();
    
    // Enable parity capture
    parity::LlaminarSnapshotHook::set_enabled(true);
    
    // Run Llaminar pipeline (automatic capture via captureIfEnabled calls)
    auto pipeline = PipelineFactory::create(config);
    pipeline->prefill(tokens, weights, ctx);
    
    // Run reference implementation and capture
    // ... reference execution ...
    
    // Compare snapshots
    auto tolerance = parity::ComparisonTolerance(1e-3f, 1e-4);
    for (int layer = 0; layer < num_layers; ++layer) {
        auto key_llama = registry.make_key("llaminar", PipelineStage::ATTENTION_OUTPUT, layer);
        auto key_ref = registry.make_key("reference", PipelineStage::ATTENTION_OUTPUT, layer);
        
        TensorSnapshot snap_llama, snap_ref;
        ASSERT_TRUE(registry.get_snapshot(key_llama, snap_llama));
        ASSERT_TRUE(registry.get_snapshot(key_ref, snap_ref));
        
        auto result = SnapshotComparator::compare(snap_ref, snap_llama, tolerance);
        EXPECT_TRUE(result.passed()) << "Layer " << layer << " failed parity";
    }
}
```

**Custom Capture Points** (extending to new architectures):
```cpp
class MyCustomPipeline : public PipelineBase, public AbstractPipeline {
    bool execute(...) override {
        // ... computation ...
        
        // Capture at custom stage
        captureIfEnabled(PipelineStage::CUSTOM, layer_idx, my_tensor);
        
        // ... more computation ...
        return true;
    }
};
```

#### Comparison Metrics

**Supported Metrics**:
- **Max Absolute Difference**: `max(|expected - actual|)`
- **Mean Absolute Difference**: `mean(|expected - actual|)`
- **Relative L2 Norm**: `||expected - actual||₂ / ||expected||₂`
- **Worst Element Tracking**: Index and values of maximum difference

**Configurable Tolerances**:
```cpp
// Strict tolerance for early layers
auto strict = ComparisonTolerance(1e-4f, 1e-5);

// Relaxed tolerance for final logits
auto relaxed = ComparisonTolerance(1e-2f, 1e-3);
```

#### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLAMINAR_PARITY_CAPTURE` | Enable automatic snapshot capture | Disabled (0) |
| `LLAMINAR_LAYER_TOKEN_DIFF` | Legacy layer diff diagnostics | Disabled (0) |

**Note**: The new parity framework (`LLAMINAR_PARITY_CAPTURE`) is complementary to the legacy layer token diff system. They can coexist but serve different purposes:
- **Parity Framework**: Cross-implementation validation (Llaminar vs llama.cpp)
- **Layer Token Diff**: Incremental decode vs replay validation within Llaminar

#### Integration with Existing Diagnostics

**Relationship to Baseline Comparison**:
- **Baseline**: Per-stage FP32 snapshots for prefill GEMM validation
- **Parity**: Cross-pipeline snapshot comparison for correctness validation
- Both use `handle_prefill_stage_snapshot()` infrastructure
- Can be used together for comprehensive validation

**Relationship to Layer Token Diff**:
- **Layer Token Diff**: Legacy diagnostic for incremental decode parity
- **Parity Framework**: Modern unified approach for all validation
- Migration path: Gradually replace layer token diff with parity framework

#### Performance Considerations

**Overhead When Disabled**:
- **Zero Overhead**: `isParityEnabled()` compiles to constant `false`
- Capture calls eliminated by compiler dead code elimination
- No runtime checks in hot paths

**Overhead When Enabled**:
- **Memory**: Stores full tensor snapshots (can be large for long sequences)
- **Compute**: Tensor copy per capture point (~microseconds for typical sizes)
- **Recommendation**: Use only for validation, not performance benchmarks

#### Future Enhancements

**Planned Features**:
1. **Selective Stage Capture**: Environment variable to capture only specific stages
2. **Streaming Snapshots**: Write captures to disk instead of memory for long sequences
3. **Differential Snapshots**: Only capture changed regions for incremental decode
4. **Automatic Tolerance Tuning**: Learn tolerances from successful runs
5. **Cross-Rank Comparison**: Validate distributed state consistency across MPI ranks
6. **Integration with CI/CD**: Automated parity regression tests on every commit

**Plugin Architecture** (future):
```cpp
class ParityPlugin {
    virtual bool shouldCapture(PipelineStage stage, int layer) = 0;
    virtual void onCapture(const TensorSnapshot& snapshot) = 0;
};

// Register custom parity validation logic
parity::registerPlugin(std::make_unique<MyCustomValidator>());
```

#### Files

**Core**:
- `src/pipeline_stages.h`: PipelineStage enum and conversion utilities
- `src/parity_hooks.h/cpp`: Production-safe hook interface
- `src/abstract_pipeline.h`: Virtual parity methods (base interface)
- `src/pipeline_base.h/cpp`: Convenience helpers (`captureIfEnabled`)
- `src/qwen_pipeline.cpp`: 8 capture points in production pipeline

**Tests**:
- `tests/parity_test_framework.h/cpp`: Full snapshot capture and comparison implementation
- `tests/test_parity_framework.cpp`: Parity framework unit tests

**Total LOC**: ~500 lines (framework), ~10 lines added to QwenPipeline

---

## Additional Technical Details

### Build System & Dependencies

**File**: `CMakeLists.txt`
- **Pattern**: Modern CMake with submodules
- **Library Structure**: Core library + executables
- **Dependencies**:
  - **COSMA**: High-performance matrix operations
  - **GGML/LLaMA.cpp**: Model format support and inference kernels
  - **MPI**: Distributed computing (OpenMPI)
  - **OpenMP**: Shared-memory parallelism
  - **NUMA**: Memory affinity management
  - **CUDA/ROCm**: GPU acceleration (optional)

**Build Targets**:
```bash
llaminar_core    # Core library with all components
llaminar         # Main executable
test_*          # Unit test executables
```

## Data Flow Architecture

### Inference Pipeline (Planned)

1. **Model Loading**: GGUF → Parsed tensors → Distributed placement
2. **Input Processing**: Tokenization → Embeddings → Attention preparation  
3. **Forward Pass**: Transformer blocks → Matrix operations → Activations
4. **Output Generation**: Logits → Sampling → Token generation
5. **Result Collection**: Distributed gather → Response formatting

## Execution Flow

### Runtime Lifecycle

1. **Initialization**: 
   - MPI setup (`MPI_Init_thread` with `MPI_THREAD_MULTIPLE`)
   - NUMA topology detection
   - Environment snapshot creation (`debugEnv()`)
   - OpenMP thread configuration

2. **Model Loading**:
   - GGUF file parsing (metadata + tensor directory)
   - Weight tensor instantiation (auto SimpleTensor/COSMATensor selection)
   - MPI broadcast/distribution of weights
   - Pipeline creation via `PipelineFactory::create(config)`

3. **Prefill Phase**:
   - Token sequence → embedding lookup
   - Per-layer execution:
     - RMSNorm (attention)
     - QKV projection (COSMA path if threshold met via `cosma_prefill_manager`)
     - Multi-head attention (RoPE + scaled dot-product)
     - Residual connection
     - RMSNorm (MLP)
     - MLP (Gate/Up/SwiGLU/Down with adaptive backend)
     - Residual connection
   - Output RMSNorm + LM head projection
   - KV cache initialization

4. **Decode Loop** (Autoregressive Generation):
   - Single token → embedding lookup
   - Per-layer execution (same structure, latency-optimized):
     - Local OpenBLAS for all matmuls
     - KV cache append
     - Causal attention over growing context
   - Output projection → logits
   - External sampling/chat interface
   - Repeat until stopping condition

5. **Completion**:
   - Optional performance summary (if perf counters enabled)
   - Optional KV cache statistics (if `--kv-stats` flag set)
   - Validation logging (if diagnostics enabled)
   - MPI cleanup (`MPI_Finalize`)

## Performance Characteristics

### Scalability
- **MPI Distributed**: Multi-node prefill + optional future distributed decode
- **NUMA Aware**: Memory locality & process pinning still handled before pipeline creation
- **Thread Parallel**: OpenMP within each rank; adaptive single vs multi-thread based on op size

### Memory Management
- **Hybrid Tensor System**: Automatic Simple vs COSMA tensor selection remains intact
- **Prefill Working Set Control**: Environment-governed memory caps (e.g., `LLAMINAR_COSMA_MAX_RESIDENT_MB`) consulted by prefill manager
- **KV Cache Growth**: Managed outside generic graph; pipeline directly appends to caches with validation tests ensuring shape parity
- **Format Optimization**: COSMA layout for optimal cache utilization
- **Legacy Compatibility**: SimpleTensor maintains existing memory patterns
- **Smart Allocation**: TensorFactory selects optimal memory layout
- **Quantization**: Reduced precision for memory efficiency

### Compute Optimization
- **COSMA Integration**: State-of-the-art matrix multiplication algorithms
- **Kernel Registration**: Pluggable optimization for different operations
- **Topology Awareness**: Hardware-specific optimizations

## Testing Strategy

**Directory**: `tests/`

**Core Infrastructure Tests**:
- `test_basic.cpp`: MPI initialization and basic functionality
- `test_numa.cpp`: NUMA topology detection and affinity  
- `test_pipeline_factory.cpp`: Pipeline factory registration and creation

**Pipeline & Parity Tests**:
- `test_qwen_pipeline.cpp`: Qwen pipeline functionality (4 test cases)
- `test_abstract_pipeline_parity.cpp`: Prefill vs incremental decode equivalence
- `test_kv_cache_growth*.cpp`: KV cache capacity management

**Backend & Kernel Tests**:
- `test_cosma_prefill_*.cpp`: Fused COSMA correctness and statistics
- `test_adaptive_matmul*.cpp`: Backend selection validation
- `test_mpi_linear_kernel.cpp`: Distributed linear projection
- `test_cosma.cpp`: Core COSMA integration

**Primitive Tests**:
- `test_rmsnorm_*.cpp`: RMSNorm correctness and parity
- `test_attention_*.cpp`: Attention mechanism validation
- `test_rope_*.cpp`: RoPE positional encoding
- `test_softmax_*.cpp`: Softmax numerical stability
- `test_mlp_tp_parity.cpp`: MLP distributed parity
- `test_tp_*.cpp`: Tensor partition correctness

**Test Coverage Focus**:
- ✅ Mathematical equivalence across execution paths (parity tests)
- ✅ MPI communication correctness and coordination
- ✅ Backend selection policy validation
- ✅ KV cache state management
- ✅ NUMA topology detection accuracy
- ✅ Distributed execution correctness

**Removed Historical Tests**:
- ❌ `test_graph.cpp`: Generic compute graph (architecture removed)
- ❌ `LinearKernelTest`: Legacy non-MPI kernel (retired after MPI migration)

## Development Patterns

### Error Handling
- Exception-based error propagation
- MPI-aware error coordination
- Graceful degradation for optional features
- Comprehensive logging for debugging

### Memory Management
- RAII patterns for resource management
- Smart pointers for automatic cleanup
- MPI memory coordination
- NUMA-aware allocation strategies

### Extensibility Points
- **Kernel Registration**: Add new operations via inheritance
- **Model Formats**: Extend ModelLoader for new formats
- **Topology Detection**: Platform-specific detection modules
- **Communication**: Custom MPI communication patterns

## Configuration & Environment

### Environment Variables
```bash
LLAMINAR_LOG_LEVEL    # Override default log level
MPI_THREAD_MULTIPLE   # Enable MPI threading support
OMP_NUM_THREADS       # Control OpenMP parallelism
CUDA_VISIBLE_DEVICES  # GPU device selection
```

### CMake Configuration
```bash
-DCMAKE_BUILD_TYPE=Debug|Release
-DENABLE_CUDA=ON|OFF
-DENABLE_ROCM=ON|OFF
-DCOSMA_SCALAPACK_LINK_LIBRARIES=<path>
```

## Future Architecture Enhancements

### Planned Components
1. **Attention Kernels**: Multi-head attention with hybrid tensor optimization
2. **Transformer Blocks**: Complete layer implementations using TensorBase interface
3. **Advanced Tensor Operations**: Extend hybrid system to all kernel types
4. **Communication Layer**: Optimized MPI communication patterns
5. **Inference Engine**: Complete LLM inference pipeline with zero-copy optimization
6. **Model Zoo**: Pre-trained model repository integration

### Performance Optimizations
1. **GPU Acceleration**: CUDA/ROCm kernel implementations
2. **Mixed Precision**: FP16/BF16 computation paths
3. **Pipeline Parallelism**: Layer-wise distribution
4. **Tensor Parallelism**: Within-layer distribution
5. **Memory Optimization**: KV-cache management

### Scalability Improvements
1. **Dynamic Load Balancing**: Adaptive work distribution
2. **Hierarchical Communication**: Optimized MPI topologies
3. **Asynchronous Execution**: Overlapped computation/communication
4. **Elastic Scaling**: Runtime process addition/removal

## Canonical Runtime Configuration

### Optimal Launch Settings

Llaminar includes a canonical launch script with empirically-optimized settings:

```bash
# Always use the canonical launcher
./run_llaminar.sh [arguments]

# The script automatically configures:
# - OpenMP: Auto-detected cores per socket, socket placement, close binding
# - MPI: 1 process per socket, memory pinning, NUMA-aware binding  
# - Threading: Single-threaded for small ops, multi-threaded for medium, distributed for large
# - Topology Detection: Mirrors C++ logic from src/common.cpp for consistent results
```

### Environment Configuration

**OpenMP Settings** (automatically configured):
```bash
OMP_NUM_THREADS=<detected>       # Auto-detected physical cores per socket
OMP_PLACES=sockets               # Thread placement strategy
OMP_PROC_BIND=close              # Bind threads close together
KMP_AFFINITY=granularity=fine,compact,1,0  # Intel threading
KMP_BLOCKTIME=0                  # Minimize thread blocking
```

**MPI Settings** (automatically configured):
```bash
OMPI_MCA_mpi_leave_pinned=1                     # Memory pinning
OMPI_MCA_btl_vader_single_copy_mechanism=none   # NUMA optimization
OMPI_MCA_btl_openib_allow_ib=1                  # InfiniBand support
```

### System Requirements

- **CPU**: Multi-socket x86_64 with NUMA support
- **Memory**: 16GB+ RAM, preferably balanced across NUMA nodes
- **MPI**: OpenMPI 4.0+ with thread support
- **OpenMP**: libgomp or Intel OpenMP runtime
- **Optimal**: 2-socket system, 1 MPI process per socket

## Usage Examples

### Basic Execution
```bash
# Topology detection and system info
./run_llaminar.sh -v --print-topology

# Performance benchmarking
./run_llaminar.sh -vv --matrix-size 2048

# GPU detection and profiling
./run_llaminar.sh --detect-gpus --profile --trace
```

### Model Inference
```bash
# Load and run inference with Qwen 2.5 model
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v

# Verbose inference with profiling
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -vv --profile
```

### Advanced Usage (Manual MPI)
```bash
# Manual MPI execution (if canonical script unavailable)
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 --report-bindings \
  ./build/llaminar -m models/qwen2.5-0.5b-instruct-q4_0.gguf
```

This architecture provides a solid foundation for high-performance, distributed LLM inference while maintaining modularity, extensibility, and observability throughout the system.

---

## Backend Abstraction Layer (Prefill & Inference)

Llaminar supports three matrix multiplication backends for different operation profiles: **OpenBLAS** (baseline CPU BLAS), **COSMA** (distributed MPI), and **Intel MKL** (BF16 hardware acceleration).

### Goals
1. Decouple kernel call sites from concrete matmul implementation details (OpenBLAS vs COSMA vs Intel MKL vs future GPU)
2. Preserve existing adaptive CPU heuristics (single-thread / multi-thread / distributed / quantized) without duplication
3. Provide zero-cost indirection today (simple inline / single virtual hop) with fast fallback to existing `adaptive_matmul`
4. Centralize prefill-vs-inference path decisions for consistent logging and experimentation
5. Enable BF16 quantization with production-quality GEMM (Intel MKL default, OpenBLAS fallback)

### Supported Backends (October 2025)

| Backend | Optimal Use Case | Availability | Status |
|---------|------------------|--------------|--------|
| **OpenBLAS** | Decode, small/medium prefill (<4K tokens) | Always available | ✅ Production |
| **COSMA** | Large prefill (≥4K tokens), multi-rank MPI | Requires world_size > 1 | ✅ Production |
| **Intel MKL** | BF16 quantized inference (2× memory savings) | Optional (`-DUSE_MKL=ON`) | ✅ Production (Oct 20, 2025) |
| **GPU** (future) | Large batch, multi-GPU inference | Not yet implemented | 🔄 Planned |

### Intel MKL BF16 Backend ✨ *OCTOBER 20, 2025*

**Motivation**: Provide production-quality BF16 GEMM with hardware acceleration on modern CPUs (Ice Lake+) and robust software fallback. Intel MKL offers optimized `cblas_sbgemm` implementation with better performance characteristics than generic OpenBLAS on supported hardware.

**Note**: Earlier concerns about OpenBLAS BF16 emulation were resolved in October 2025 - OpenBLAS v0.3.26 verified working correctly on Cascade Lake without NaN issues. MKL remains valuable for hardware-accelerated paths and performance optimization.

**Key Features**:
- **Default BF16 Backend**: When `HAVE_MKL` defined and `LLAMINAR_QUANT_BF16_GEMM=1`, Intel MKL is tried first
- **100% Parity**: 387/387 tests passing vs PyTorch (< 1e-4 relative L2 error)
- **Graceful Fallback**: MKL → OpenBLAS → FP32 expansion (automatic degradation)
- **2× Memory Savings**: bfloat16 storage (16-bit) vs FP32 (32-bit) with same exponent range
- **Hardware Acceleration**: AVX512_BF16 on Ice Lake, Sapphire Rapids (when available)

**Implementation**:
- `src/backends/MKLBackend.{h,cpp}`: Intel MKL cblas_sbgemm wrapper
- `src/AdaptiveMatmul.h` `multiplyBF16()`: Backend selection logic (lines ~315-380)
- `LLAMINAR_QUANT_BF16_PREFER_MKL`: **DEPRECATED** - MKL now default (set =0 to force OpenBLAS)

**Build Configuration**:
```bash
cmake -B build_mkl -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/intel/oneapi/mkl/latest" \
  -DUSE_MKL=ON
cmake --build build_mkl --parallel
```

**Runtime Activation**:
```bash
export LLAMINAR_QUANT_BF16_GEMM=1  # Enable BF16 path (MKL becomes default)
./run_llaminar.sh -m model.gguf -v
```

### Components
**Current Architecture**: `src/prefill_provider.{h,cpp}`, `src/openblas_prefill_provider.{h,cpp}`, `src/cosma_prefill_provider.{h,cpp}`, `src/backends/MKLBackend.{h,cpp}`

| Element | Description |
|---------|-------------|
| `PrefillProvider` | Abstract base class for prefill execution strategies |
| `OpenBLASPrefillProvider` | CPU-based prefill using OpenBLAS (baseline, optimal for <4K tokens) |
| `COSMAPrefillProvider` | Distributed prefill using COSMA (optimal for ≥4K tokens, multi-rank) |
| `MKLBackend` | Intel MKL BF16 GEMM backend (cblas_sbgemm, default when HAVE_MKL defined) |
| `AdaptiveMatmul` | Backend selection and fallback logic (FP32, BF16 quantized paths) |
| `PrefillProviderFactory` | Automatic provider selection based on sequence length and MPI context |
| `PrefillMetrics` | Comprehensive instrumentation (timing, FLOPS, snapshots) |
| `GpuPrefillProvider` (future) | GPU-accelerated prefill with device memory management |

**Decode Path**: Currently uses local OpenBLAS in `QwenPipeline::decode()` for latency optimization. BF16 quantized decode may use Intel MKL when `LLAMINAR_QUANT_BF16_GEMM=1`. Future work may extract decode provider abstraction.

### Invocation Pattern (from QwenPipeline)
```cpp
// In QwenPipeline::prefill()
PrefillMetrics metrics;
auto provider = PrefillProviderFactory::create(config_, mpi_ctx_, tokens.size());
bool success = provider->execute(tokens, weights, output, ctx, metrics);

if (success) {
    // Cache results, update context
    ctx.backend_stats.prefill_backend = metrics.backend_name;
    ctx.backend_stats.prefill_gflops = metrics.gflops();
}

// BF16 quantized path (via AdaptiveMatmul::multiplyBF16)
if (debugEnv().quant.bf16_gemm && weight_is_bf16) {
    #ifdef HAVE_MKL
        success = MKLBackend::multiply_bf16(act, weight_bf16, output, m, n, k);
    #else
        success = OpenBLASBackend::multiply_bf16(act, weight_bf16, output, m, n, k);
    #endif
    // Automatic FP32 fallback if both fail
}
```

Provider selection logic:
1. **BF16 Quantized Path**: `LLAMINAR_QUANT_BF16_GEMM=1` + `HAVE_MKL` → Intel MKL (default)
2. **Environment Override**: `ADAPTIVE_DISABLE_COSMA=1` → OpenBLAS
3. **Debug Forcing**: `LLAMINAR_COSMA_FORCE_DIRECT=1` → COSMA
3. **Sequence Length**: `seq_len >= threshold` (default 4096) + multi-rank → COSMA
4. **Default**: OpenBLAS for small sequences or single-rank

### Descriptor Structures
| Provider Metrics | Captured Data |
|------------------|---------------|
| `PrefillMetrics::embedding_ms` | Embedding lookup time |
| `PrefillMetrics::attention_ms` | Total attention time across layers |
| `PrefillMetrics::ffn_ms` | Total FFN time across layers |
| `PrefillMetrics::norm_ms` | Normalization time |
| `PrefillMetrics::total_flops` | Floating point operations count |
| `PrefillMetrics::backend_name` | Provider identifier ("OpenBLAS", "COSMA") |

All providers capture snapshots at standardized stages for parity testing when `LLAMINAR_PARITY_CAPTURE=1`.

### Logging & Observability
Providers emit structured metrics on completion:
```
[INFO] OpenBLASPrefillProvider completed: 347.2ms, 12.3 GFLOPS, 24 layers, 147 snapshots
[INFO] COSMAPrefillProvider completed: 96.5ms, 44.1 GFLOPS, 24 layers, 147 snapshots
```

Each provider captures at identical stages for A/B comparison:
- EMBEDDING, ATTENTION_NORM, Q/K/V_PROJECTION, ROPE_APPLICATION
- ATTENTION_SCORES, ATTENTION_SOFTMAX, ATTENTION_CONTEXT, ATTENTION_OUTPUT
- FFN_NORM, FFN_GATE, FFN_UP, FFN_SWIGLU, FFN_DOWN
- FINAL_NORM, LM_HEAD

### Relationship to COSMA Prefill Manager
`COSMAPrefillProvider` delegates to `CosmaPrefillManager` for distributed matmul coordination, maintaining separation of concerns:
- **Provider**: Orchestrates transformer layer execution, snapshot capture, metrics
- **Manager**: Handles COSMA-specific matmul, orientation validation, memory management

### Future Extensions (Non-Breaking)
| Extension | Impact |
|-----------|--------|
| GPU Provider | Implement `GpuPrefillProvider` with cuBLAS/rocBLAS, no pipeline changes |
| Decode Provider | Extract `DecodeProvider` abstraction mirroring prefill pattern |
| Fused Kernels | Providers can use specialized fused ops without interface changes |
| Auto-tuning | Providers collect perf stats for heuristic refinement |
| Multi-Model | Factory can select provider based on model architecture

---

## Distribution Modes & Tensor Parallel Simplification

The legacy intra-rank tensor parallel splitter logic has been removed. Llaminar now treats **Tensor Parallel (TP)** strictly as *inter-socket / inter-process* sharding. Two high-level deployment modes are defined:

| Mode | Default Threshold | Characteristics |
|------|-------------------|-----------------|
| `ReplicatedDataParallel` | < ~32B params (configurable) | All weights replicated per rank; simpler, lower latency for small/medium models |
| `ShardedTensorParallel` | ≥ threshold | Parameter matrices partitioned across ranks; reduces memory footprint |

Runtime selection is driven by environment snapshot fields parsed once in `debug_env` (`DistributionEnvConfig`). This centralization:
1. Eliminates repeated `getenv()` overhead in hot loops
2. Provides a discoverable registry of supported knobs
3. Enables consistent experiment repro (single summary line on startup)

### Rationale for Simplification
Previous intra-rank column/row splitter heuristics added complexity and branch misprediction overhead for marginal gains on small decode shapes. Inter-rank sharding (true TP) captures the meaningful memory scaling while leaving per-rank math contiguous and cache-friendly.

### Interaction with Backends
- Prefill path: large prompt → may trigger distributed COSMA via `adaptive_matmul` once thresholds hit
- Inference path: small decode batches remain local; sharding only affects weight residency & gather/scatter steps external to backend invocation

---

## Environment Snapshot (`debugEnv`) Consolidation

Hot-path code (kernels, backend selection) consumes a pre-parsed immutable snapshot from `debug_env.{h,cpp}` instead of raw `std::getenv` calls.

Benefits:
1. Reduced libc call overhead on small matmuls
2. Single source of truth for defaults & validation
3. Easier documentation & test harness override

Key groups include:
| Group | Example Fields |
|-------|----------------|
| `cosma` | `prefill_threshold`, `max_resident_mb`, validation toggles |
| `distribution` | Mode overrides, sharding thresholds |
| `attention` | Trace / micro diagnostics flags |

---

## PyTorch Reference Implementation for Ground-Truth Validation 🐍

**Location**: `python/reference/`  
**Status**: Production-ready infrastructure (test integration pending API alignment)  
**Purpose**: Stage-by-stage numerical validation against HuggingFace transformer models

### Overview

The PyTorch reference implementation provides **21 granular capture stages** (vs llama.cpp's 2-stage capture) for comprehensive parity testing. It enables debugging of quantization errors, operator implementation issues, and numerical drift by comparing Llaminar's execution against ground-truth PyTorch/HuggingFace models.

### Architecture

**Three-Component Design**:

1. **Python Reference Implementation** (`python/reference/`)
   - Abstract base class: `AbstractReferenceModel`
   - Concrete implementations: `QwenReferenceModel` (production), `LlamaReferenceModel` (prototype)
   - Stage-by-stage hook system for capturing intermediate activations
   - Quantization simulation (Q4_0, Q6_K) matching GGUF formats
   - CLI tool: `run_reference.py` for snapshot generation

2. **Snapshot Bridge** (`tests/npz_loader.h`, `tests/npz_to_npy.py`)
   - Header-only C++ .npy parser (zero external dependencies)
   - Python extraction helper (.npz → individual .npy files)
   - Cross-language compatibility layer

3. **C++ Parity Integration** (`tests/test_parity_framework.cpp`)
   - Test template for PyTorch snapshot comparison
   - Configurable tolerances for FP32/FP16/Q6_K/Q4_0
   - Stage-by-stage comparison with detailed diagnostics
   - Status: Infrastructure complete, awaiting API refactor (see §15.7)

### 21 Capture Stages

Granular snapshots covering the full transformer pipeline:

| Category | Stages | Purpose |
|----------|--------|---------|
| **Input** | `EMBEDDING`, `POSITIONAL_ENCODING` | Input validation, vocabulary alignment |
| **Attention (per-layer)** | `ATTENTION_NORM`, `QKV_PROJECTION`, `ROPE_APPLICATION`, `ATTENTION_SCORES`, `ATTENTION_PROBS`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL` | Attention mechanism correctness |
| **FFN (per-layer)** | `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_ACTIVATION`, `FFN_DOWN`, `FFN_RESIDUAL` | Feed-forward network validation |
| **Per-Layer** | `LAYER_OUTPUT` | Layer-wise correctness |
| **Output** | `FINAL_NORM`, `LM_HEAD`, `FINAL_LOGITS`, `PROBABILITIES` | Final output validation |

**Note**: Llaminar's `QwenPipeline` currently captures 8 strategic stages (see Parity Test Framework §13). The PyTorch reference provides all 21 for comprehensive debugging.

### Quick Start Workflow

**1. Generate PyTorch Reference Snapshots**:
```bash
# Install dependencies (pre-installed in devcontainer)
pip install -r python/reference/requirements.txt

# Generate FP32 snapshots
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --output pytorch_snapshots.npz \
    --verbose

# Generate quantized snapshots for Q4_0 validation
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --quantization q4_0 \
    --output pytorch_q4_snapshots.npz
```

**2. Extract Snapshots to C++-Compatible Format**:
```bash
# Extract .npz archive to individual .npy files
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/

# Directory structure after extraction:
# pytorch_snapshots/
#   EMBEDDING_-1.npy              # Non-layer stage (layer_idx = -1)
#   ATTENTION_OUTPUT_0.npy        # Layer 0 attention output
#   FFN_DOWN_0.npy                # Layer 0 FFN down projection
#   FINAL_NORM_-1.npy             # Final normalization
#   LM_HEAD_-1.npy                # Language model head
```

**3. Load and Compare in C++**:
```cpp
#include "npz_loader.h"
using namespace llaminar::parity;

// Load PyTorch reference snapshot
PyTorchSnapshotLoader pytorch_loader("pytorch_snapshots/");
NpyArray pytorch_embedding;
if (pytorch_loader.load_snapshot("EMBEDDING", -1, pytorch_embedding)) {
    // pytorch_embedding.shape = {1, seq_len, hidden_dim}
    // pytorch_embedding.data = std::vector<float>
    
    // Compare with Llaminar snapshot from SnapshotRegistry
    auto llaminar_snapshot = SnapshotRegistry::get("EMBEDDING", -1);
    auto metrics = SnapshotComparator::compare(pytorch_embedding, llaminar_snapshot);
    
    LOG_INFO("EMBEDDING max_abs_diff: " << metrics.max_abs_diff 
             << ", rel_l2: " << metrics.rel_l2);
}
```

### File Format & Naming Convention

**NumPy .npy Format** (header-only parser in `npz_loader.h`):
- Magic bytes: `\x93NUMPY`
- Version: 1-3 (2 bytes)
- Header: Python dict with `descr` (dtype), `fortran_order`, `shape`
- Data: Raw float32 blob (little-endian)

**Naming Convention**:
```
{STAGE_NAME}_{LAYER_INDEX}.npy

Examples:
  EMBEDDING_-1.npy           # Global stage (no layer)
  ATTENTION_OUTPUT_0.npy     # Layer 0 attention output
  FFN_DOWN_5.npy             # Layer 5 FFN down projection
  FINAL_NORM_-1.npy          # Final norm (no layer)
```

### Comparison Metrics & Tolerances

**Standard Metrics**:
- `max_abs_diff`: max|A - B| across all elements
- `mean_abs_diff`: mean|A - B|
- `rel_l2`: ||A - B||₂ / ||A||₂ (relative L2 norm)

**Recommended Tolerances**:

| Precision | max_abs_diff | rel_l2 | Use Case |
|-----------|--------------|--------|----------|
| FP32 | 1e-4 | 1e-5 | Numerical precision only |
| Q6_K | 5e-3 | 1e-2 | High-quality quantization |
| Q4_0 | 1e-2 | 5e-2 | Aggressive quantization |

**Adaptive Tolerance Strategy** (recommended):
```cpp
// Stricter for early stages (error doesn't accumulate)
if (stage == "EMBEDDING" || stage.find("NORM") != npos) {
    return is_quantized ? 5e-3f : 1e-4f;
}

// Relaxed for late stages (accumulated error)
if (stage == "FINAL_LOGITS") {
    return is_quantized ? 5e-2f : 1e-3f;
}
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYTORCH_SNAPSHOT_DIR` | Directory with extracted .npy files | `pytorch_snapshots/` |
| `PYTORCH_SNAPSHOT_TOKENS` | Token IDs used for generation | `1,2,3,4,5` |
| `PYTORCH_MODEL_PATH` | HuggingFace checkpoint path | `Qwen/Qwen2-0.5B-Instruct` |
| `PYTORCH_QUANTIZATION` | Quantization format | `q4_0` or `q6_k` |

### Model Family Support

**Production**:
- **Qwen**: `QwenReferenceModel` with full 21-stage capture
  - Supports Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B
  - Quantization: Q4_0, Q6_K matching GGUF
  - RoPE, SwiGLU FFN, RMSNorm validation

**Prototype**:
- **LLaMA**: `LlamaReferenceModel` with basic structure
  - Needs attention implementation completion
  - Quantization hooks in place

**Future** (extensible via `AbstractReferenceModel`):
- DeepSeek (similar to LLaMA)
- Mistral (sliding window attention)
- GPT-2 (learned positional embeddings)

### Integration Status

**✅ Complete**:
- Python reference implementation (800+ lines)
- CLI tool for snapshot generation
- Header-only C++ .npy parser (303 lines, zero dependencies)
- Python extraction helper (100 lines)
- Comprehensive documentation (850+ lines total)

**⚠️ Pending** (see §15.7):
- Test case needs API refactoring (~1 hour)
  - Current: Uses old `GGUFContext` API
  - Target: New `ModelConfig`-based `AbstractPipeline` API
- CI/CD integration (GitHub Actions workflow)

### Advanced Usage Patterns

**Quantization Error Analysis**:
```bash
# Generate FP32 baseline
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3 \
    --output fp32.npz

# Generate Q6_K quantized
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3 \
    --quantization q6_k --output q6k.npz

# Generate Q4_0 quantized
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3 \
    --quantization q4_0 --output q4.npz

# Compare all three in C++ to isolate quantization vs implementation errors
```

**Selective Stage Testing**:
```python
# Only capture specific stages for faster iteration
python python/reference/run_reference.py \
    --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --stages EMBEDDING ATTENTION_OUTPUT FINAL_NORM \
    --output selective.npz
```

**Long Sequence Validation**:
```bash
# Test with sequences up to context length
python python/reference/run_reference.py \
    --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens $(seq 1 100 | tr '\n' ',') \
    --output long_seq.npz
```

### Troubleshooting Guide

**Shape Mismatch**:
- **Symptom**: `Shape mismatch: PyTorch [1,5,512] vs Llaminar [5,512]`
- **Cause**: Different batch dimension handling
- **Fix**: Ensure consistent token count (`PYTORCH_SNAPSHOT_TOKENS` matches input)

**High Numeric Drift**:
- **Symptom**: `rel_l2 = 0.5` (expected < 0.01)
- **Causes**: Quantization errors, operator bugs, different RNG seeds
- **Debug**: Compare stage-by-stage to isolate error source

**File Not Found**:
- **Symptom**: `FileNotFoundError: EMBEDDING_-1.npy`
- **Fix**: Run extraction step: `python tests/npz_to_npy.py snapshots.npz output/`
- **Verify**: `ls -la $PYTORCH_SNAPSHOT_DIR`

**Parser Error**:
- **Symptom**: `Invalid .npy header`
- **Cause**: Corrupted file or unsupported NumPy version
- **Fix**: Regenerate with latest NumPy: `pip install -U numpy`

### API Migration Path (§15.7)

**Current State**: Test infrastructure complete, awaiting `ModelConfig` API alignment

**Required Changes** (~1 hour):

```cpp
// 1. Model Loading (OLD → NEW)
// auto gguf_ctx = loader.load(model_path);
ModelConfig config = createConfigFromGGUF(model_path);

// 2. Pipeline Creation (OLD → NEW)
// auto pipeline = std::make_unique<QwenPipelineAdapter>(gguf_ctx);
auto pipeline = PipelineFactory::create(config);

// 3. Prefill Execution (OLD → NEW)
// auto result = pipeline->prefill(token_ids);
IModelWeights* weights = /* load from model */;
StageContext ctx = /* create context */;
bool success = pipeline->prefill(token_ids, weights, ctx);

// 4. Enable Test
// Remove DISABLED_ prefix from test name
TEST(ParityFramework, DistributedPipelineVsPyTorchReference) { ... }
```

### Documentation References

- **Python Reference README**: `python/reference/README.md` (300+ lines comprehensive guide)
- **C++ Integration Guide**: `tests/PYTORCH_INTEGRATION.md` (450+ lines with troubleshooting)
- **Test Framework Guide**: `tests/AGENTS.md` §14 (PyTorch reference usage)
- **Status Report**: `PYTORCH_INTEGRATION_STATUS.md` (implementation summary)
- **Parity Framework**: `tests/AGENTS.md` §13 (Llaminar snapshot capture system)

### Best Practices

1. **Version Control Snapshots**: Commit .npz files or document HuggingFace checkpoint versions
2. **Test Incrementally**: Validate embedding → layer 0 → layer N → final output
3. **Document Tolerances**: Add comments justifying relaxed tolerances for quantized models
4. **Isolate Errors**: Compare FP32 first, then add quantization progressively
5. **Cache Models**: Use `~/.cache/huggingface` to avoid re-downloading multi-GB checkpoints
6. **Validate Shapes First**: Shape mismatch indicates fundamental API issues
7. **Use Verbose Logging**: Enable `-vvv` and `--verbose` for debugging

### Future Enhancements

1. **Full .npz Support**: Eliminate extraction step by adding ZIP parsing to `npz_loader.h`
2. **Model Coverage**: Add DeepSeek, Mistral, GPT-2 reference implementations
3. **Automatic Tolerance Learning**: Adaptive tolerances based on successful runs
4. **GPU Validation**: Compare CUDA/ROCm kernels against PyTorch GPU execution
5. **Visual Diff Reports**: HTML output with per-stage heatmaps and statistics
6. **Streaming Snapshots**: Disk-backed storage for very long sequences
7. **CI/CD Integration**: Automated snapshot generation and comparison on every PR

---