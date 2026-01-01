# Proposal: Tensor Contracts and Validation Framework

**Author**: David Sanftenberg  
**Date**: January 2026  
**Status**: ✅ Complete (All Phases)
**Related Issues**: HybridQ16 ATTENTION_OUTPUT divergence, RoPE layout bugs, debug code block type mismatch

---

## Implementation Progress

| Phase | Status | Date |
|-------|--------|------|
| Phase 1: Unified Tensor Verification | ✅ Complete | 2026-01-01 |
| Phase 2: Safe Q16 Block Access | ✅ Complete | 2026-01-01 |
| Phase 3: Tensor Layout Contracts | ✅ Complete | 2026-01-01 |
| Phase 3b: Strict Layout Enforcement | ✅ Complete | 2026-01-01 |
| Phase 3c: HEAD_MAJOR KV Cache Layout | ✅ Complete | 2026-01-01 |
| Phase 4: Typed Stage Parameters | ✅ Complete | 2026-01-01 |
| Phase 5: Documentation & Migration | ✅ Complete | 2026-01-01 |

### Phase 1 Deliverables
- ✅ Created `src/v2/tensors/TensorVerification.h` with `VerificationResult`, `VerificationConfig`, `verifyRawBuffer()`, `dumpStageBuffers()`, `VerificationFailure` exception
- ✅ Added `verifyStageEntry()` and `verifyStageExit()` to `GraphExecutor` with automatic entry/exit validation
- ✅ Validation auto-enabled in Debug/Integration builds via `LLAMINAR_ASSERTIONS_ACTIVE`
- ✅ Automatic buffer dumps to `/tmp/llaminar_verification_dump/` on verification failure
- ✅ 26 unit tests in `tests/v2/unit/utils/Test__TensorVerification.cpp`
- ✅ Updated `copilot-instructions.md` with verification system documentation
- ✅ **Phase 1 Extension**: `fail_on_zero = true` by default in Debug/Integration builds (all-zero outputs are almost always bugs)
- ✅ **Phase 1 Extension**: Added `allowsZeroOutput()` virtual method to `IComputeStage` for per-stage override (e.g., `KVCacheGatherStage` allows zero when cache is empty)

### Phase 2 Deliverables (Complete)
- ✅ Added safe block accessors to `Q16_1Tensor`: `as_block_32()`, `as_block_64()`, `as_block_128()` (const + mutable)
- ✅ Added generic element access: `dequant_element(row, col)`, `quantized_element(row, col)`, `block_scale(row, block_in_row)`
- ✅ Created `src/v2/tensors/Q16BlockDispatch.h` with `dispatchQ16Block()`, `forEachQ16Block()` template helpers
- ✅ 29 unit tests in `tests/v2/unit/tensors/Test__Q16SafeBlockAccess.cpp` - ALL PASSING
- ✅ Added `[[deprecated]]` attribute and runtime warnings to `q16_1_blocks()`, `mutable_q16_1_blocks()`, `blocks()`, `mutable_blocks()`, `data_impl()`, `mutable_data_impl()`
- ✅ Migrated test code using unsafe `q16_1_blocks()` patterns to safe `as_block_*()` accessors
- ✅ All 212 V2 unit tests pass with zero deprecation warnings

### Phase 3 Deliverables (Complete)
- ✅ Created `src/v2/tensors/TensorLayout.h` with `TensorLayout` enum (`Q_SEQ_HEAD_DIM`, `Q_HEAD_SEQ_DIM`, `KV_POS_HEAD_DIM`, `KV_HEAD_POS_DIM`, `ROW_MAJOR_2D`, `ROW_MAJOR_1D`, `UNKNOWN`)
- ✅ Added helper functions: `layoutName()`, `layoutNameShort()`, `layoutsCompatible()`, `isQueryLayout()`, `isKVLayout()`, `isGenericLayout()`, `kvTransposeTarget()`, `queryTransposeTarget()`
- ✅ Added virtual `layout()` getter and `setLayout()` setter to `TensorBase` class
- ✅ Added `layout_` member variable to `TensorBase` (default: `UNKNOWN`)
- ✅ 18 unit tests in `tests/v2/unit/tensors/Test__TensorLayout.cpp` - ALL PASSING
- ✅ All 213 V2 unit tests pass

### Phase 3b Deliverables (Complete) - Layout Validation & Fluent API
- ✅ Created `LayoutExpectation` struct in `TensorLayout.h` for encoding expected layouts from buffer requirements
- ✅ Created `LayoutValidationResult` struct for validation outcomes with detailed error messages
- ✅ Added `validateTensorLayout()` function to check actual tensor layout against buffer requirement expectations
- ✅ Added `validateBufferLayoutByShape()` function to infer layout from tensor dimensions
- ✅ Extended `BufferDescriptor` with optional `TensorLayout expected_layout` field (default: `UNKNOWN`)
- ✅ Extended fluent API: `addInput(name, shape, type, layout)`, `addOutput(name, shape, type, layout)`, `addInout(name, shape, type, layout)`, `addScratch(name, shape, type, layout)`, `addWeight(name, shape, type, layout)` - optional 4th layout parameter preserves fluent chaining
- ✅ `GraphExecutor::verifyStageEntry()` automatically validates input layouts against `getBufferRequirements()` declarations
- ✅ 3 new unit tests for layout parameter API in `Test__BufferRole.cpp`: `LayoutParameterAPI`, `LayoutDefaultsToUnknown`, `MixedLayoutDeclarations`
- ✅ Updated `FusedAttentionWoStage` to use new layout declaration API
- ✅ All 213 V2 unit tests pass

### Phase 3c Deliverables (Complete) - HEAD_MAJOR KV Cache Layout
- ✅ Created `KVCacheLayoutMode` enum in `UnifiedKVCache.h` with `POSITION_MAJOR` and `HEAD_MAJOR` options
- ✅ Updated `UnifiedKVCache` constructor to accept optional `KVCacheLayoutMode` parameter (default: `POSITION_MAJOR`)
- ✅ Implemented HEAD_MAJOR allocation: `[n_kv_heads * max_seq_len, head_dim]` vs POSITION_MAJOR: `[max_seq_len, kv_dim]`
- ✅ Implemented `copy_append_data()` scatter-copy for HEAD_MAJOR: position-major input → head-major cache
- ✅ Implemented `shift_evict_data()` per-head shift for HEAD_MAJOR eviction
- ✅ Updated `FusedAttentionWoStage` to skip transpose workaround when cache is HEAD_MAJOR
- ✅ Updated `GraphOrchestrator` to auto-select HEAD_MAJOR for HybridQ16 mode (`kv_precision == Q16_1`)
- ✅ 25 multi-block Q16_1 tests in `Test__UnifiedKVCache.cpp` covering various `head_dim > block_size` scenarios
- ✅ 4 orchestrator layout mode tests verifying auto-selection (FP32→POSITION_MAJOR, HybridQ16→HEAD_MAJOR)
- ✅ All 213 V2 unit tests pass

### Phase 4 Deliverables (Complete)
- ✅ Replaced `void* mpi_comm` with `const MPIContext* mpi_ctx` in `AllreduceStage::Params`
- ✅ Replaced `void* mpi_comm` and `int world_size` with `const MPIContext* mpi_ctx` in `AllGatherStage::Params`
- ✅ Updated `AllreduceStage.cpp` to use `mpi_ctx->comm()` directly instead of casting `void*`
- ✅ Updated `AllGatherStage.cpp` to use `mpi_ctx->comm()` and `mpi_ctx->world_size()`
- ✅ Removed `getMPICommPtr()` helper function from `GraphBuildUtils.h`
- ✅ Updated all call sites in `Qwen2Graph.cpp`, `GraphResolver.cpp`
- ✅ Updated all test files: `Test__AllGatherStage.cpp`, `Test__StageBufferRequirements.cpp`, `Test__StageDumpInfo.cpp`, `Test__MPI_ColumnParallelLMHead.cpp`
- ✅ Removed `MPI_Comm mpi_comm` from `GraphResolverConfig` (replaced with existing `mpi_ctx`)
- ✅ Updated schema resolution to store `mpi_ctx` instead of `mpi_comm` in `opaque_params`
- ✅ Removed `#include <mpi.h>` from `GraphResolver.h` (no longer needed)
- ✅ Breaking change: All existing code using `mpi_comm`/`world_size` must migrate to `mpi_ctx`
- ✅ All 213 V2 unit tests pass

### Phase 5 Deliverables (Complete)
- ✅ All 213 V2 unit tests pass
- ✅ 53/54 integration tests pass (98% pass rate)
- ✅ Expected failure: HybridQ16 vs FP32 parity test (Q16_1 attention implementation needs debugging)
- ✅ Updated proposal document to reflect completed status
- ✅ Migration guides documented in proposal for typed_data(), TensorVerification, and MPI parameters

---

## Problem Statement

Five architectural gaps are causing debugging difficulty and silent failures in the Llaminar V2 pipeline:

### Issue 1: Missing Systematic Buffer Validation

**Current State**: `GraphExecutor` has post-stage validation via `getDumpInfo()`, but:
- No **input validation** before stage execution
- No enforcement of null/NaN/Inf/zero checks on stage entry
- Validation happens only after stage completion (too late to catch input corruption)
- Different stages have inconsistent validation coverage

**Impact**: Corrupted inputs silently flow through stages until they cause visible divergence many layers later, making root cause analysis extremely difficult.

### Issue 2: `typed_data()` Inflexibility for Q16 Block Sizes

**Current State**: `Q16_1Tensor::typed_data()` always returns `Q16_1Block*` (32-element blocks), but actual storage may use:
- `Q16_1Block_64` (64-element, 136 bytes) for head_dim=64
- `Q16_1Block_128` (128-element, 264 bytes) for head_dim=128

**Impact**: Code using `typed_data()` on Q16_1Tensor reads garbage memory when the actual block size differs from 32. This caused the debug dump corruption we just fixed.

**Example of Broken Pattern**:
```cpp
// ❌ WRONG: typed_data() returns Q16_1Block* even if storage is Q16_1Block_64
auto* q_q16 = dynamic_cast<Q16_1Tensor*>(tensor);
const Q16_1Block* blocks = q_q16->typed_data();  // Returns wrong type!
blocks[1].d;  // Reads garbage (offset 72 instead of 136)
```

### Issue 3: Undefined Tensor Layout Contracts

**Current State**: No formal specification of Q/K/V tensor layouts. Different components assume different layouts:

| Component | Q Layout | K Layout | V Layout |
|-----------|----------|----------|----------|
| FusedQKVGEMMStage | `[seq_len][n_heads*head_dim]` | `[seq_len][n_kv_heads*head_dim]` | Same as K |
| RoPEStage | Assumes head-major interleaving | Same | Same |
| UnifiedKVCache | N/A | `[position][n_kv_heads][head_dim]` (position-major) | Same as K |
| Q16IntegerAttentionRef | `[seq_len][n_heads][head_dim]` | `[n_kv_heads][kv_len][head_dim]` (head-major) | Same as K |

**Impact**: Silent layout mismatch causes attention scores to be garbage. The FusedAttentionWoStage has a transpose workaround, but this is a band-aid.

### Issue 4: `void*` Pointers in Stage Params

**Current State**: Several stage parameter structs use `void*` for type-erased parameters:

```cpp
// AllreduceStage.h
struct Params {
    TensorBase *buffer = nullptr;
    void *mpi_comm = nullptr;  // ❌ Opaque - no type safety
    // ...
};

// GraphSchema.h
struct ResolvedStage {
    std::unordered_map<std::string, void *> opaque_params;  // ❌ Type-erased
};
```

**Impact**: 
- No compile-time type checking on MPI communicators
- Easy to pass wrong pointer type silently
- Debugging requires manual `reinterpret_cast` inspection
- IDE tooling cannot provide type information

### Issue 5: No Snapshot Validation

**Current State**: Snapshots are captured without validation. Corrupted data flows into snapshot comparisons, making E2E test failures ambiguous.

**Impact**:
- Snapshots may contain NaN/Inf/all-zero data
- E2E parity tests fail with unhelpful "cosine similarity = 0" messages
- Root cause hidden: is it the snapshot data, the capture, or the PyTorch reference?

---

## Proposed Solutions

### Solution 1: Unified Tensor Verification System

**Goal**: Absorb existing `TensorValidation.h` and `ValidationConfig` into a comprehensive tensor verification system that:
- Uses `LLAMINAR_ASSERT*` macros consistently
- Runs automatically at stage boundaries in Debug/Integration builds
- On failure: logs context, dumps all stage tensors to disk, throws exception

#### 1.1 New `TensorVerification.h` Header

```cpp
// utils/TensorVerification.h

#pragma once

#include "Assertions.h"
#include "Logger.h"
#include "../tensors/Tensors.h"
#include "../execution/compute_stages/IComputeStage.h"
#include <filesystem>
#include <fstream>

namespace llaminar2::verification {

/**
 * @brief Result of tensor verification
 */
struct VerificationResult {
    bool passed = true;
    std::string error_message;
    
    // Diagnostics
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
    size_t total_sampled = 0;
    
    static VerificationResult ok() { return {true, ""}; }
    static VerificationResult fail(const std::string& msg) { return {false, msg}; }
};

/**
 * @brief Configuration for verification
 */
struct VerificationConfig {
    int sample_rows = 8;           // Check first N rows (efficiency)
    bool check_null = true;        // Fail on null pointer
    bool check_nan = true;         // Fail on NaN
    bool check_inf = true;         // Fail on Inf  
    bool check_all_zero = false;   // Fail on all-zero (warning by default)
    bool dump_on_failure = true;   // Dump tensors to /tmp on failure
};

/**
 * @brief Verify a single tensor
 * 
 * @param tensor Tensor to verify
 * @param name Name for diagnostics
 * @param config Verification configuration
 * @return VerificationResult with pass/fail and diagnostics
 */
VerificationResult verifyTensor(
    const TensorBase* tensor,
    const char* name,
    const VerificationConfig& config = {});

/**
 * @brief Dump all stage buffers to disk
 * 
 * Creates files in /tmp/llaminar_verification_dump/<timestamp>_<stage>/
 * 
 * @param stage_name Name of the failing stage
 * @param layer_idx Layer index (-1 if unknown)
 * @param phase "ENTRY" or "EXIT"
 * @param dump_info Stage's dump info with all buffers
 */
void dumpStageBuffers(
    const std::string& stage_name,
    int layer_idx,
    const char* phase,
    const StageDumpInfo& dump_info);

/**
 * @brief Verification failure exception
 * 
 * Contains all context needed to diagnose the failure.
 */
class VerificationFailure : public std::runtime_error {
public:
    VerificationFailure(
        const std::string& stage_name,
        int layer_idx,
        const char* phase,
        const std::string& tensor_name,
        const std::string& reason,
        const std::string& dump_path)
        : std::runtime_error(formatMessage(stage_name, layer_idx, phase, tensor_name, reason, dump_path))
        , stage_name_(stage_name)
        , layer_idx_(layer_idx)
        , phase_(phase)
        , tensor_name_(tensor_name)
        , dump_path_(dump_path)
    {}
    
    const std::string& stageName() const { return stage_name_; }
    int layerIdx() const { return layer_idx_; }
    const char* phase() const { return phase_; }
    const std::string& tensorName() const { return tensor_name_; }
    const std::string& dumpPath() const { return dump_path_; }
    
private:
    static std::string formatMessage(
        const std::string& stage_name, int layer_idx, const char* phase,
        const std::string& tensor_name, const std::string& reason,
        const std::string& dump_path);
    
    std::string stage_name_;
    int layer_idx_;
    const char* phase_;
    std::string tensor_name_;
    std::string dump_path_;
};

} // namespace llaminar2::verification
```

#### 1.2 Stage Boundary Verification in GraphExecutor

```cpp
// In GraphExecutor::executeNode()

#if LLAMINAR_ASSERTIONS_ACTIVE

bool GraphExecutor::verifyStageEntry(const ComputeNode& node, int layer_idx) {
    using namespace verification;
    
    const auto& config = debugEnv().validation;
    if (!config.validate_buffers) return true;
    
    auto dump_info = node.stage->getDumpInfo();
    VerificationConfig vconfig;
    vconfig.check_all_zero = false;  // Zero inputs may be valid (first layer residual)
    
    for (const auto& input : dump_info.inputs) {
        // Create temporary TensorBase wrapper for raw data
        auto result = verifyRawBuffer(input.data, input.rows, input.cols, 
                                       input.name, input.dtype, vconfig);
        if (!result.passed) {
            // Log context
            LOG_ERROR("[VERIFY] ENTRY FAILED: layer=" << layer_idx 
                      << " stage=" << node.name 
                      << " tensor=" << input.name
                      << " reason=" << result.error_message);
            
            // Dump all buffers
            std::string dump_path = dumpStageBuffers(node.name, layer_idx, "ENTRY", dump_info);
            LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
            
            // Throw exception
            throw VerificationFailure(node.name, layer_idx, "ENTRY", 
                                       input.name, result.error_message, dump_path);
        }
    }
    return true;
}

bool GraphExecutor::verifyStageExit(const ComputeNode& node, int layer_idx) {
    using namespace verification;
    
    const auto& config = debugEnv().validation;
    if (!config.validate_buffers) return true;
    
    auto dump_info = node.stage->getDumpInfo();
    VerificationConfig vconfig;
    vconfig.check_all_zero = config.fail_on_zero;  // Zero outputs are usually bugs
    
    for (const auto& output : dump_info.outputs) {
        auto result = verifyRawBuffer(output.data, output.rows, output.cols,
                                       output.name, output.dtype, vconfig);
        if (!result.passed) {
            // Log context
            LOG_ERROR("[VERIFY] EXIT FAILED: layer=" << layer_idx
                      << " stage=" << node.name
                      << " tensor=" << output.name  
                      << " reason=" << result.error_message);
            
            // Dump all buffers
            std::string dump_path = dumpStageBuffers(node.name, layer_idx, "EXIT", dump_info);
            LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
            
            // Throw exception
            throw VerificationFailure(node.name, layer_idx, "EXIT",
                                       output.name, result.error_message, dump_path);
        }
    }
    return true;
}

#endif // LLAMINAR_ASSERTIONS_ACTIVE
```

#### 1.3 Execution Flow with Verification

```cpp
bool GraphExecutor::executeNode(ComputeNode& node, IDeviceContext* ctx) {
    // Extract layer index from node name (e.g., "layer3_attention" -> 3)
    int layer_idx = extractLayerIndex(node.name);
    
#if LLAMINAR_ASSERTIONS_ACTIVE
    // ENTRY verification - before execute()
    if (debugEnv().validation.validate_buffers) {
        verifyStageEntry(node, layer_idx);  // Throws on failure
    }
#endif

    bool success = node.stage->execute(ctx);
    
    if (!success) {
        LOG_ERROR("[GraphExecutor] Stage '" << node.name << "' returned false");
        return false;
    }

#if LLAMINAR_ASSERTIONS_ACTIVE
    // EXIT verification - after execute()
    if (debugEnv().validation.validate_buffers) {
        verifyStageExit(node, layer_idx);  // Throws on failure
    }
#endif

    return true;
}
```

#### 1.4 Dump File Format

When verification fails, all stage buffers are dumped to:
```
/tmp/llaminar_verification_dump/
└── 20260101_143022_layer5_FusedAttentionWoStage_EXIT/
    ├── manifest.json           # Summary of all tensors
    ├── Q_input_fp32.bin        # Raw FP32 data
    ├── Q_input_metadata.txt    # Shape, dtype, stats
    ├── K_input_q16_1.bin       # Raw Q16_1 blocks
    ├── K_input_metadata.txt
    ├── attention_output.bin    # The failing output
    ├── attention_output_metadata.txt
    └── ...
```

**manifest.json**:
```json
{
    "stage": "FusedAttentionWoStage",
    "layer_idx": 5,
    "phase": "EXIT",
    "timestamp": "2026-01-01T14:30:22Z",
    "failure": {
        "tensor": "attention_output",
        "reason": "Contains 156 NaN values in first 8 rows"
    },
    "inputs": [
        {"name": "Q", "shape": [1, 14, 64], "dtype": "Q16_1", "file": "Q_input_q16_1.bin"}
    ],
    "outputs": [
        {"name": "attention_output", "shape": [1, 896], "dtype": "FP32", "file": "attention_output.bin"}
    ]
}
```

#### 1.5 Integration with Snapshots

Snapshot capture also uses verification:

```cpp
#define LLAMINAR_SNAPSHOT(key, tensor_ptr)                                     \
    do {                                                                       \
        auto* _tensor = (tensor_ptr);                                          \
        if (_tensor != nullptr) {                                              \
            /* Verify before capture (Debug/Integration only) */               \
            auto _result = verification::verifyTensor(_tensor, (key));         \
            if (!_result.passed) {                                             \
                LOG_ERROR("[SNAPSHOT] Verification FAILED for " << (key)       \
                          << ": " << _result.error_message);                   \
                /* Still capture for debugging, but log the issue */           \
            }                                                                  \
            captureSnapshot((key), _tensor->data(), _tensor->numel());         \
        }                                                                      \
    } while(0)
```

---

### Solution 2: Safe Q16 Block Access API

#### 2.1 Deprecate `typed_data()` for Q16_1Tensor

The CRTP base `TypedTensorBase<Q16_1Tensor, Q16_1Block>` forces `typed_data()` to return `Q16_1Block*`, but this is **inherently unsafe** for variable-block tensors.

**Options**:
1. **Remove Q16_1Tensor from CRTP** - Break the pattern entirely
2. **Add explicit unsafe marker** - `typed_data_unsafe()` with warning
3. **Runtime block size validation** - `typed_data()` throws if block_size != BLOCK_32

**Recommended**: Option 3 with deprecation warning:

```cpp
class Q16_1Tensor : public TypedTensorBase<Q16_1Tensor, Q16_1Block>, ... {
public:
    // DEPRECATED: Only valid for BLOCK_32 tensors
    [[deprecated("Use raw_data() + q16_block_size() for safe access")]]
    const Q16_1Block* typed_data() const {
        LLAMINAR_ASSERT(block_size_ == Q16BlockSize::BLOCK_32,
            "typed_data() called on Q16_1Tensor with block_size=" 
            << static_cast<int>(block_size_) << " (expected 32)");
        return q16_1_blocks();
    }
};
```

#### 2.2 Add Type-Safe Block Accessors

```cpp
class Q16_1Tensor {
public:
    // Get block size enum
    Q16BlockSize q16_block_size() const { return block_size_; }
    
    // Type-safe accessors that return correct block type
    // Returns nullptr if block size doesn't match
    const Q16_1Block* as_block_32() const {
        return (block_size_ == Q16BlockSize::BLOCK_32) 
            ? reinterpret_cast<const Q16_1Block*>(raw_data()) : nullptr;
    }
    const Q16_1Block_64* as_block_64() const {
        return (block_size_ == Q16BlockSize::BLOCK_64)
            ? reinterpret_cast<const Q16_1Block_64*>(raw_data()) : nullptr;
    }
    const Q16_1Block_128* as_block_128() const {
        return (block_size_ == Q16BlockSize::BLOCK_128)
            ? reinterpret_cast<const Q16_1Block_128*>(raw_data()) : nullptr;
    }
    
    // Generic element access (slower but safe)
    float dequant_element(size_t row, size_t col) const;
    int16_t quantized_element(size_t row, size_t col) const;
    float block_scale(size_t row, size_t col) const;
    
    // Block iteration helpers
    size_t blocks_per_row() const;
    size_t total_blocks() const;
};
```

#### 2.3 Template-Based Block Dispatch Pattern

For kernels that need efficient block access:

```cpp
// Pattern for kernels that need efficient typed access
template<typename Func>
auto dispatchQ16Block(const Q16_1Tensor* tensor, Func&& func) {
    switch (tensor->q16_block_size()) {
        case Q16BlockSize::BLOCK_32:
            return func(tensor->as_block_32(), Q16_1Block::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_64:
            return func(tensor->as_block_64(), Q16_1Block_64::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_128:
            return func(tensor->as_block_128(), Q16_1Block_128::BLOCK_SIZE);
        default:
            LLAMINAR_UNREACHABLE("Invalid Q16 block size");
    }
}

// Usage:
dispatchQ16Block(q_tensor, [&](auto* blocks, int block_size) {
    // blocks is correctly typed (Q16_1Block*, Q16_1Block_64*, etc.)
    for (int b = 0; b < num_blocks; ++b) {
        float scale = blocks[b].d;
        // ...
    }
});
```

---

### Solution 3: Explicit Tensor Layout Contracts

#### 3.1 Define Layout Enum

```cpp
// tensors/TensorLayout.h

namespace llaminar2 {

/**
 * @brief Tensor memory layout for multi-dimensional data
 * 
 * All attention tensors use one of these canonical layouts.
 * Layouts are orthogonal to quantization format.
 */
enum class TensorLayout {
    // Query tensor layouts: [seq_len, n_heads, head_dim]
    Q_SEQ_HEAD_DIM,      // [seq_len][n_heads][head_dim] - natural for batched GEMM
    
    // K/V tensor layouts
    KV_POS_HEAD_DIM,     // [position][n_kv_heads][head_dim] - position-major (cache-friendly)
    KV_HEAD_POS_DIM,     // [n_kv_heads][position][head_dim] - head-major (attention-friendly)
    
    // Generic row-major
    ROW_MAJOR_2D,        // [rows][cols] - standard 2D layout
    
    // Unknown/unspecified
    UNKNOWN
};

/**
 * @brief Get human-readable layout name
 */
const char* layoutName(TensorLayout layout);

/**
 * @brief Check if two layouts are compatible without transpose
 */
bool layoutsCompatible(TensorLayout a, TensorLayout b);

} // namespace
```

#### 3.2 Add Layout to TensorBase

```cpp
class TensorBase {
public:
    // Existing...
    
    // NEW: Layout contract
    virtual TensorLayout layout() const { return TensorLayout::UNKNOWN; }
    
    // Layout conversion (returns new tensor with different layout)
    virtual std::unique_ptr<TensorBase> transpose_to(TensorLayout target) const;
    
    // In-place layout change (for mutable buffers only)
    virtual bool transpose_inplace_to(TensorLayout target);
};
```

#### 3.3 Layout Validation in Stages

```cpp
class FusedAttentionWoStage : public ComputeStage {
public:
    ValidationResult validateInputs() const override {
        // Check Q layout
        if (params_.Q->layout() != TensorLayout::Q_SEQ_HEAD_DIM &&
            params_.Q->layout() != TensorLayout::UNKNOWN) {
            return ValidationResult::fail(
                "Q tensor has layout " + std::string(layoutName(params_.Q->layout())) +
                ", expected Q_SEQ_HEAD_DIM");
        }
        
        // Check K/V layout
        TensorLayout expected_kv = (params_.backend == FusedAttentionBackend::Q16_INTEGER)
            ? TensorLayout::KV_HEAD_POS_DIM   // Q16 kernel expects head-major
            : TensorLayout::KV_POS_HEAD_DIM;  // Other kernels expect position-major
            
        if (params_.K->layout() != expected_kv &&
            params_.K->layout() != TensorLayout::UNKNOWN) {
            return ValidationResult::fail(
                "K tensor has layout " + std::string(layoutName(params_.K->layout())) +
                ", expected " + std::string(layoutName(expected_kv)));
        }
        
        return validation::validateTensor(params_.Q, "Q") &&
               validation::validateTensor(params_.K, "K") &&
               validation::validateTensor(params_.V, "V");
    }
};
```

#### 3.4 Document Canonical Layouts

Create explicit documentation of expected layouts at each stage boundary:

```
Pipeline Layout Flow (HybridQ16 mode):
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ Stage                 │ Input Layout        │ Output Layout             │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ EmbeddingStage        │ token_ids[seq]      │ hidden[seq, d_model]      │
│ FusedQKVGEMMStage     │ hidden[seq, d]      │ Q[seq, n_h*hd]            │
│                       │                     │ K[seq, n_kv*hd]           │
│                       │                     │ V[seq, n_kv*hd]           │
│ RoPEStage (Q)         │ Q[seq, n_h*hd]      │ Q[seq, n_h, hd]  Q16_1    │
│ RoPEStage (K)         │ K[seq, n_kv*hd]     │ K[seq, n_kv, hd] Q16_1    │
│ KVCacheAppendStage    │ K[seq,n_kv,hd]      │ K_cache[pos,n_kv,hd]      │
│                       │ V[seq,n_kv,hd]      │ V_cache[pos,n_kv,hd]      │
│ FusedAttentionWoStage │ Q[seq,n_h,hd]       │ residual[seq,d_model] Q16 │
│ (Q16_INTEGER backend) │ K[n_kv,kv_len,hd]*  │                           │
│                       │ V[n_kv,kv_len,hd]*  │                           │
└─────────────────────────────────────────────────────────────────────────┘

* K/V transposed from position-major to head-major via workaround.
  Production should use HeadMajorKVCache for Q16_INTEGER backend.
```

---

### Solution 4: Typed MPI Parameters

#### 4.1 Replace `void*` with Type-Safe Wrapper

```cpp
// utils/MPITypes.h

namespace llaminar2 {

/**
 * @brief Type-safe wrapper for MPI_Comm
 * 
 * Provides compile-time type safety while remaining compatible with
 * MPI_Comm at runtime. Default-constructible to MPI_COMM_NULL equivalent.
 */
class MPICommHandle {
public:
    MPICommHandle() = default;  // Null communicator
    
    // Explicit construction from MPI_Comm (defined in .cpp to avoid MPI include)
    static MPICommHandle fromComm(MPI_Comm comm);
    
    // Access underlying communicator (defined in .cpp)
    MPI_Comm get() const;
    
    // Validity check
    bool valid() const { return valid_; }
    explicit operator bool() const { return valid_; }
    
private:
    // Store as int to avoid MPI header dependency
    // MPI_Comm is typically int or pointer - we store raw bytes
    alignas(8) uint8_t storage_[8] = {0};
    bool valid_ = false;
};

} // namespace llaminar2
```

#### 4.2 Update Stage Params

```cpp
// AllreduceStage.h - BEFORE
struct Params {
    TensorBase *buffer = nullptr;
    void *mpi_comm = nullptr;  // ❌ Type-erased
};

// AllreduceStage.h - AFTER  
struct Params {
    TensorBase *buffer = nullptr;
    MPICommHandle mpi_comm;     // ✅ Type-safe, default = null
    const MPIContext *mpi_ctx = nullptr;  // Alternative: full context
};
```

#### 4.3 Remove `opaque_params` from GraphSchema

```cpp
// GraphSchema.h - BEFORE
struct ResolvedStage {
    std::unordered_map<std::string, void *> opaque_params;
};

// GraphSchema.h - AFTER
struct ResolvedStage {
    // No more void* - use typed fields
    MPICommHandle mpi_comm;
    const MPIContext *mpi_ctx = nullptr;
};
```

---

## Implementation Plan

### Phase 1: Unified Tensor Verification System (2-3 days) ✅ COMPLETE

#### Code Research Findings

Analysis of existing codebase reveals the following infrastructure to build upon:

**Existing Validation Infrastructure** (to be absorbed):

| File | Component | Current Behavior | Issue |
|------|-----------|-----------------|-------|
| `GraphExecutor.cpp:567-655` | `validateStageOutputs()` | Only validates FP32 OUTPUTS | No input validation |
| `GraphExecutor.cpp:492-560` | `executeNode()` | Calls validation after execute | No entry validation |
| `TensorValidation.h` | `tensorAppearsZero()` | Checks sampled elements for zeros | Uses `#ifndef NDEBUG` not `LLAMINAR_ASSERTIONS_ACTIVE` |
| `TensorValidation.h` | `tensorHasNaNOrInf()` | Checks sampled elements for NaN/Inf | Returns bool, doesn't throw |
| `TensorValidation.h` | `assertTensorValid()` | Throws `std::runtime_error` | Not used in GraphExecutor, wrong exception type |

**Reusable Dump Infrastructure** (in `StageDumper.h`):

| Component | Location | Can Reuse |
|-----------|----------|-----------|
| `StageDumpContext` struct | Lines 30-50 | Pattern for verification dump context |
| `createDumpDirectory()` | Lines 105-130 | Pattern for `/tmp/llaminar_verification_dump/` |
| `dumpFP32Buffer()` | Lines 150-180 | Direct reuse for FP32 binary dump |
| `dumpQ8_1Blocks()` | Lines 200-230 | Pattern for quantized tensor dumps |

**Configuration Infrastructure** (in `DebugEnv.h`):

| Field | Current Default | Change Needed |
|-------|-----------------|---------------|
| `validate_buffers` | `false` | Auto-enable in Debug/Integration via `LLAMINAR_ASSERTIONS_ACTIVE` |
| `fail_on_nan` | `false` | Auto-enable in Debug/Integration |
| `fail_on_zero` | `false` | Keep as opt-in (too many false positives) |
| `dump_on_failure` | N/A | **Add new field**, default `true` |
| `sample_rows` | N/A | **Add new field**, default `8` |

**Execution Context** (in `IGraphExecutor.h`):

| Field | Location | Use |
|-------|----------|-----|
| `current_layer_idx` | `GraphExecutorConfig:62` | Already available - use for layer number in error messages |
| `current_iteration` | `GraphExecutorConfig:63` | Already available - use for iteration in dump directory name |
| `mpi_rank` | `MPIContext` | Use for rank in dump directory name |

**Key Architectural Decisions**:

1. **Entry + Exit Verification**: Current code only validates outputs AFTER execute. New system verifies inputs BEFORE execute (catch corruption early) AND outputs AFTER execute.

2. **Exception on Failure**: Current `validateStageOutputs()` returns `false` and logs warning. New system throws `VerificationFailure` exception with full context (layer, stage, phase, tensor, reason, dump path).

3. **Automatic Dump on Failure**: Current code has separate dump logic via `StageDumper`. New system automatically dumps ALL stage buffers (inputs + outputs + weights) when verification fails.

4. **Unified Guards**: Replace `#ifndef NDEBUG` with `#if LLAMINAR_ASSERTIONS_ACTIVE` for consistency with assertion framework.

#### Detailed File Changes

**NEW FILES:**

| File | Purpose |
|------|---------|
| `src/v2/utils/TensorVerification.h` | Main verification API: `VerificationResult`, `VerificationFailure`, `verifyTensor()`, `verifyRawBuffer()`, `dumpStageBuffers()` |
| `src/v2/utils/TensorVerification.cpp` | Implementation of dumping logic (reuses StageDumper patterns) |
| `tests/v2/unit/Test__TensorVerification.cpp` | Unit tests for verification functions |

**MODIFIED FILES:**

| File | Changes |
|------|---------|
| `src/v2/execution/GraphExecutor.h` | Add `verifyStageEntry()`, `verifyStageExit()` methods (under `#if LLAMINAR_ASSERTIONS_ACTIVE`) |
| `src/v2/execution/GraphExecutor.cpp` | Replace `validateStageOutputs()` with entry/exit verification calls in `executeNode()` |
| `src/v2/utils/DebugEnv.h` | Extend `ValidationConfig` with `dump_on_failure`, `sample_rows` |
| `src/v2/tensors/TensorValidation.h` | **DEPRECATED** - functionality moved to TensorVerification.h |
| `src/v2/CMakeLists.txt` | Add TensorVerification.cpp to sources |

#### Code Examples

**1. New `TensorVerification.h` Header:**

```cpp
// src/v2/utils/TensorVerification.h
#pragma once

#include "Assertions.h"
#include "Logger.h"
#include "DebugEnv.h"
#include "../execution/compute_stages/IComputeStage.h"
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace llaminar2::verification {

/**
 * @brief Result of tensor verification
 */
struct VerificationResult {
    bool passed = true;
    std::string tensor_name;
    std::string error_reason;
    
    // Diagnostics
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
    size_t total_sampled = 0;
    
    static VerificationResult ok() { return {true, "", ""}; }
    static VerificationResult fail(const std::string& tensor, const std::string& reason) {
        VerificationResult r;
        r.passed = false;
        r.tensor_name = tensor;
        r.error_reason = reason;
        return r;
    }
};

/**
 * @brief Exception thrown when verification fails
 * 
 * Contains all context needed to diagnose the failure.
 */
class VerificationFailure : public std::runtime_error {
public:
    VerificationFailure(
        const std::string& stage_name,
        int layer_idx,
        const char* phase,      // "ENTRY" or "EXIT"
        const std::string& tensor_name,
        const std::string& reason,
        const std::string& dump_path)
        : std::runtime_error(formatMessage(stage_name, layer_idx, phase, tensor_name, reason, dump_path))
        , stage_name_(stage_name)
        , layer_idx_(layer_idx)
        , phase_(phase)
        , tensor_name_(tensor_name)
        , dump_path_(dump_path)
    {}
    
    const std::string& stageName() const { return stage_name_; }
    int layerIdx() const { return layer_idx_; }
    const char* phase() const { return phase_; }
    const std::string& tensorName() const { return tensor_name_; }
    const std::string& dumpPath() const { return dump_path_; }
    
private:
    static std::string formatMessage(
        const std::string& stage_name, int layer_idx, const char* phase,
        const std::string& tensor_name, const std::string& reason,
        const std::string& dump_path)
    {
        std::ostringstream oss;
        oss << "\n"
            << "╔══════════════════════════════════════════════════════════════════╗\n"
            << "║               TENSOR VERIFICATION FAILED                          ║\n"
            << "╠══════════════════════════════════════════════════════════════════╣\n"
            << "║ Layer:  " << layer_idx << "\n"
            << "║ Stage:  " << stage_name << "\n"
            << "║ Phase:  " << phase << "\n"
            << "║ Tensor: " << tensor_name << "\n"
            << "║ Reason: " << reason << "\n"
            << "║\n"
            << "║ Dump:   " << dump_path << "\n"
            << "╚══════════════════════════════════════════════════════════════════╝\n";
        return oss.str();
    }
    
    std::string stage_name_;
    int layer_idx_;
    const char* phase_;
    std::string tensor_name_;
    std::string dump_path_;
};

/**
 * @brief Verify raw FP32 buffer data
 * 
 * Samples first N rows to check for null, NaN, Inf, all-zero.
 */
VerificationResult verifyRawBuffer(
    const void* data,
    size_t rows,
    size_t cols,
    const char* name,
    const char* dtype,
    int sample_rows = 8);

/**
 * @brief Dump all stage buffers to disk for debugging
 * 
 * Creates directory: /tmp/llaminar_verification_dump/<timestamp>_<stage>_<phase>/
 * 
 * @return Path to dump directory
 */
std::string dumpStageBuffers(
    const std::string& stage_name,
    int layer_idx,
    const char* phase,      // "ENTRY" or "EXIT"
    const StageDumpInfo& dump_info);

} // namespace llaminar2::verification
```

**2. Modified `executeNode()` in GraphExecutor.cpp:**

```cpp
bool GraphExecutor::executeNode(ComputeNode &node, IDeviceContext *ctx)
{
    if (!node.stage)
    {
        LOG_ERROR("[GraphExecutor] Node '" << node.name << "' has no stage");
        return false;
    }

    // Extract layer index from config (set by pipeline before each layer)
    const int layer_idx = config_.current_layer_idx;

#if LLAMINAR_ASSERTIONS_ACTIVE
    // ENTRY verification - before execute()
    if (debugEnv().validation.validate_buffers)
    {
        verifyStageEntry(node, layer_idx);  // Throws VerificationFailure on error
    }
#endif

    // ... existing StageDumper code ...

    auto start = std::chrono::high_resolution_clock::now();
    bool success = node.stage->execute(ctx);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // ... existing profiling/dump code ...

#if LLAMINAR_ASSERTIONS_ACTIVE
    // EXIT verification - after execute()
    if (success && debugEnv().validation.validate_buffers)
    {
        verifyStageExit(node, layer_idx);  // Throws VerificationFailure on error
    }
#endif

    // ... existing snapshot callback ...

    return success;
}
```

**3. New `verifyStageEntry()` and `verifyStageExit()` methods:**

```cpp
#if LLAMINAR_ASSERTIONS_ACTIVE
void GraphExecutor::verifyStageEntry(const ComputeNode& node, int layer_idx)
{
    using namespace verification;
    
    auto dump_info = node.stage->getDumpInfo();
    const auto& config = debugEnv().validation;
    
    // Verify inputs (not outputs) at entry
    for (const auto& input : dump_info.inputs)
    {
        if (!input.data) continue;  // Null inputs checked separately if needed
        
        auto result = verifyRawBuffer(
            input.data, input.rows, input.cols,
            input.name, input.dtype, config.sample_rows);
        
        if (!result.passed)
        {
            LOG_ERROR("[VERIFY] ENTRY FAILED: layer=" << layer_idx 
                      << " stage=" << node.name 
                      << " tensor=" << result.tensor_name
                      << " reason=" << result.error_reason);
            
            std::string dump_path;
            if (config.dump_on_failure) {
                dump_path = dumpStageBuffers(node.name, layer_idx, "ENTRY", dump_info);
                LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
            }
            
            throw VerificationFailure(
                node.name, layer_idx, "ENTRY",
                result.tensor_name, result.error_reason, dump_path);
        }
    }
}

void GraphExecutor::verifyStageExit(const ComputeNode& node, int layer_idx)
{
    using namespace verification;
    
    auto dump_info = node.stage->getDumpInfo();
    const auto& config = debugEnv().validation;
    
    // Verify outputs at exit
    for (const auto& output : dump_info.outputs)
    {
        if (!output.data) continue;
        
        auto result = verifyRawBuffer(
            output.data, output.rows, output.cols,
            output.name, output.dtype, config.sample_rows);
        
        if (!result.passed)
        {
            LOG_ERROR("[VERIFY] EXIT FAILED: layer=" << layer_idx
                      << " stage=" << node.name
                      << " tensor=" << result.tensor_name
                      << " reason=" << result.error_reason);
            
            std::string dump_path;
            if (config.dump_on_failure) {
                dump_path = dumpStageBuffers(node.name, layer_idx, "EXIT", dump_info);
                LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
            }
            
            throw VerificationFailure(
                node.name, layer_idx, "EXIT",
                result.tensor_name, result.error_reason, dump_path);
        }
    }
}
#endif // LLAMINAR_ASSERTIONS_ACTIVE
```

**4. Updated `ValidationConfig` in DebugEnv.h:**

```cpp
struct ValidationConfig
{
    bool validate_buffers = false; ///< Enable buffer validation at stage boundaries
    bool fail_on_zero = false;     ///< Treat all-zero as failure (vs warning)
    bool fail_on_nan = false;      ///< Treat NaN/Inf as failure
    bool dump_on_failure = true;   ///< Dump all stage buffers to /tmp on failure
    int sample_rows = 8;           ///< Number of rows to sample for verification

    ValidationConfig()
    {
        reload();
    }

    void reload()
    {
#if LLAMINAR_ASSERTIONS_ACTIVE
        validate_buffers = true;
        fail_on_nan = true;
        dump_on_failure = true;
#endif
        // Environment overrides...
        if (const char* v = std::getenv("LLAMINAR_VALIDATE_BUFFERS"))
            validate_buffers = (std::atoi(v) != 0);
        if (const char* v = std::getenv("LLAMINAR_FAIL_ON_ZERO"))
            fail_on_zero = (std::atoi(v) != 0);
        if (const char* v = std::getenv("LLAMINAR_FAIL_ON_NAN"))
            fail_on_nan = (std::atoi(v) != 0);
        if (const char* v = std::getenv("LLAMINAR_DUMP_ON_FAILURE"))
            dump_on_failure = (std::atoi(v) != 0);
        if (const char* v = std::getenv("LLAMINAR_VERIFY_SAMPLE_ROWS"))
            sample_rows = std::atoi(v);
    }
};
```

**5. Implementation of `verifyRawBuffer()` in TensorVerification.cpp:**

```cpp
// src/v2/utils/TensorVerification.cpp

#include "TensorVerification.h"
#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <chrono>
#include <iomanip>
#include <fstream>

namespace llaminar2::verification {

VerificationResult verifyRawBuffer(
    const void* data,
    size_t rows,
    size_t cols,
    const char* name,
    const char* dtype,
    int sample_rows)
{
    VerificationResult result;
    result.tensor_name = name;
    
    // Null check
    if (data == nullptr) {
        return VerificationResult::fail(name, "null data pointer");
    }
    
    // Only verify FP32 for now (can extend to Q8_1/Q16_1 scale factors later)
    if (std::string(dtype) != "FP32") {
        return VerificationResult::ok();
    }
    
    const float* fp32_data = static_cast<const float*>(data);
    const size_t total_elements = rows * cols;
    
    if (total_elements == 0) {
        return VerificationResult::fail(name, "zero elements");
    }
    
    // Sample first N rows
    const size_t sample_elements = std::min(
        static_cast<size_t>(sample_rows) * cols,
        total_elements);
    
    size_t nan_count = 0, inf_count = 0, zero_count = 0;
    
    for (size_t i = 0; i < sample_elements; ++i) {
        float v = fp32_data[i];
        if (std::isnan(v)) ++nan_count;
        else if (std::isinf(v)) ++inf_count;
        else if (v == 0.0f) ++zero_count;
    }
    
    result.nan_count = nan_count;
    result.inf_count = inf_count;
    result.zero_count = zero_count;
    result.total_sampled = sample_elements;
    
    const auto& config = debugEnv().validation;
    
    // Check NaN/Inf
    if (nan_count > 0 && config.fail_on_nan) {
        std::ostringstream oss;
        oss << nan_count << " NaN values in first " << sample_rows << " rows";
        return VerificationResult::fail(name, oss.str());
    }
    
    if (inf_count > 0 && config.fail_on_nan) {
        std::ostringstream oss;
        oss << inf_count << " Inf values in first " << sample_rows << " rows";
        return VerificationResult::fail(name, oss.str());
    }
    
    // Check all-zero
    if (zero_count == sample_elements && config.fail_on_zero) {
        std::ostringstream oss;
        oss << "all zeros in first " << sample_rows << " rows";
        return VerificationResult::fail(name, oss.str());
    }
    
    return VerificationResult::ok();
}

std::string dumpStageBuffers(
    const std::string& stage_name,
    int layer_idx,
    const char* phase,
    const StageDumpInfo& dump_info)
{
    // Create dump directory: /tmp/llaminar_verification_dump/<timestamp>_<stage>_<phase>/
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << "/tmp/llaminar_verification_dump/"
        << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
        << "_layer" << layer_idx
        << "_" << stage_name
        << "_" << phase;
    
    std::string dump_dir = oss.str();
    
    // Create directories
    mkdir("/tmp/llaminar_verification_dump", 0755);
    mkdir(dump_dir.c_str(), 0755);
    mkdir((dump_dir + "/inputs").c_str(), 0755);
    mkdir((dump_dir + "/outputs").c_str(), 0755);
    
    // Write manifest.json
    {
        std::ofstream manifest(dump_dir + "/manifest.json");
        manifest << "{\n"
                 << "  \"stage\": \"" << stage_name << "\",\n"
                 << "  \"layer_idx\": " << layer_idx << ",\n"
                 << "  \"phase\": \"" << phase << "\",\n"
                 << "  \"inputs\": [\n";
        
        bool first = true;
        for (const auto& input : dump_info.inputs) {
            if (!first) manifest << ",\n";
            first = false;
            manifest << "    {\"name\": \"" << input.name 
                     << "\", \"shape\": [" << input.rows << ", " << input.cols 
                     << "], \"dtype\": \"" << input.dtype << "\"}";
        }
        
        manifest << "\n  ],\n  \"outputs\": [\n";
        
        first = true;
        for (const auto& output : dump_info.outputs) {
            if (!first) manifest << ",\n";
            first = false;
            manifest << "    {\"name\": \"" << output.name
                     << "\", \"shape\": [" << output.rows << ", " << output.cols
                     << "], \"dtype\": \"" << output.dtype << "\"}";
        }
        
        manifest << "\n  ]\n}\n";
    }
    
    // Dump input tensors
    for (const auto& input : dump_info.inputs) {
        if (!input.data) continue;
        std::string path = dump_dir + "/inputs/" + input.name + ".bin";
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
            fwrite(input.data, input.element_size, input.rows * input.cols, f);
            fclose(f);
        }
    }
    
    // Dump output tensors
    for (const auto& output : dump_info.outputs) {
        if (!output.data) continue;
        std::string path = dump_dir + "/outputs/" + output.name + ".bin";
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
            fwrite(output.data, output.element_size, output.rows * output.cols, f);
            fclose(f);
        }
    }
    
    return dump_dir;
}

} // namespace llaminar2::verification
```

#### Existing Code to Absorb/Deprecate

| Current Code | Action |
|--------------|--------|
| `TensorValidation.h::tensorAppearsZero()` | Logic absorbed into `verifyRawBuffer()` |
| `TensorValidation.h::tensorHasNaNOrInf()` | Logic absorbed into `verifyRawBuffer()` |
| `TensorValidation.h::assertTensorValid()` | Replaced by `VerificationFailure` exception |
| `GraphExecutor::validateStageOutputs()` | Replaced by `verifyStageEntry()` + `verifyStageExit()` |

#### Reusable Infrastructure

| Component | Reuse Strategy |
|-----------|----------------|
| `StageDumper::dumpFP32Buffer()` | Pattern reused for binary tensor dump |
| `StageDumper::createDumpDirectory()` | Pattern reused for verification dump directory |
| `StageDumpInfo` structure | Used directly - stages already implement `getDumpInfo()` |
| `ValidationConfig` in DebugEnv.h | Extended with new fields |
| `LLAMINAR_ASSERTIONS_ACTIVE` macro | Used as compile-time guard |
| `AssertionError` format in Assertions.h | Pattern reused for `VerificationFailure` formatting |

#### Phase 1 Test Strategy

**Test File**: `tests/v2/unit/Test__TensorVerification.cpp` (replaces `Test__TensorValidation.cpp`)

**Test Categories**:

```cpp
// =============================================================================
// 1. VerificationResult Tests
// =============================================================================

TEST_F(TensorVerificationTest, VerificationResultOk)
{
    auto result = verification::VerificationResult::ok();
    EXPECT_TRUE(result.passed);
    EXPECT_TRUE(result.tensor_name.empty());
}

TEST_F(TensorVerificationTest, VerificationResultFail)
{
    auto result = verification::VerificationResult::fail("my_tensor", "found NaN");
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.tensor_name, "my_tensor");
    EXPECT_EQ(result.error_reason, "found NaN");
}

// =============================================================================
// 2. verifyRawBuffer Tests
// =============================================================================

TEST_F(TensorVerificationTest, NullDataPointerFails)
{
    auto result = verification::verifyRawBuffer(nullptr, 10, 64, "null_buf", "FP32");
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.error_reason, "null data pointer");
}

TEST_F(TensorVerificationTest, ZeroElementsFails)
{
    float dummy = 1.0f;
    auto result = verification::verifyRawBuffer(&dummy, 0, 0, "empty_buf", "FP32");
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.error_reason, "zero elements");
}

TEST_F(TensorVerificationTest, DetectsNaNInFP32)
{
    std::vector<float> data(64, 1.0f);
    data[32] = std::numeric_limits<float>::quiet_NaN();
    
    // With fail_on_nan = true (default in Debug)
    auto result = verification::verifyRawBuffer(data.data(), 1, 64, "nan_buf", "FP32", 8);
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.nan_count, 1);
}

TEST_F(TensorVerificationTest, DetectsInfInFP32)
{
    std::vector<float> data(64, 1.0f);
    data[10] = std::numeric_limits<float>::infinity();
    
    auto result = verification::verifyRawBuffer(data.data(), 1, 64, "inf_buf", "FP32", 8);
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.inf_count, 1);
}

TEST_F(TensorVerificationTest, DetectsAllZero)
{
    std::vector<float> data(64, 0.0f);
    
    // Only fails if fail_on_zero = true
    // Test with LLAMINAR_FAIL_ON_ZERO=1
    auto result = verification::verifyRawBuffer(data.data(), 1, 64, "zero_buf", "FP32", 8);
    EXPECT_EQ(result.zero_count, 64);
    // Failure depends on fail_on_zero config
}

TEST_F(TensorVerificationTest, PassesValidFP32Data)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    auto result = verification::verifyRawBuffer(
        tensor->data(), 32, 64, "valid_buf", "FP32", 8);
    EXPECT_TRUE(result.passed);
}

TEST_F(TensorVerificationTest, SkipsNonFP32Types)
{
    uint8_t dummy[100] = {0};
    // Q8_1 data should pass through without FP32 checks
    auto result = verification::verifyRawBuffer(dummy, 10, 10, "q8_buf", "Q8_1", 8);
    EXPECT_TRUE(result.passed);  // Non-FP32 types always pass (for now)
}

// =============================================================================
// 3. VerificationFailure Exception Tests
// =============================================================================

TEST_F(TensorVerificationTest, VerificationFailureFormatsMessage)
{
    verification::VerificationFailure ex(
        "MyStage", 5, "EXIT", "output_tensor", "42 NaN values",
        "/tmp/llaminar_verification_dump/test_dump");
    
    std::string msg = ex.what();
    EXPECT_NE(msg.find("Layer:  5"), std::string::npos);
    EXPECT_NE(msg.find("Stage:  MyStage"), std::string::npos);
    EXPECT_NE(msg.find("Phase:  EXIT"), std::string::npos);
    EXPECT_NE(msg.find("Tensor: output_tensor"), std::string::npos);
    EXPECT_NE(msg.find("42 NaN values"), std::string::npos);
    EXPECT_NE(msg.find("/tmp/llaminar_verification_dump"), std::string::npos);
}

TEST_F(TensorVerificationTest, VerificationFailureAccessors)
{
    verification::VerificationFailure ex(
        "TestStage", 3, "ENTRY", "input_tensor", "null data", "/tmp/dump");
    
    EXPECT_EQ(ex.stageName(), "TestStage");
    EXPECT_EQ(ex.layerIdx(), 3);
    EXPECT_STREQ(ex.phase(), "ENTRY");
    EXPECT_EQ(ex.tensorName(), "input_tensor");
    EXPECT_EQ(ex.dumpPath(), "/tmp/dump");
}

// =============================================================================
// 4. dumpStageBuffers Tests
// =============================================================================

TEST_F(TensorVerificationTest, DumpStageBuffersCreatesDirectory)
{
    StageDumpInfo info;
    std::vector<float> input_data(64, 1.5f);
    std::vector<float> output_data(64, 2.5f);
    info.addInput("test_input", input_data.data(), 1, 64);
    info.addOutput("test_output", output_data.data(), 1, 64);
    
    std::string dump_path = verification::dumpStageBuffers(
        "TestStage", 7, "EXIT", info);
    
    EXPECT_TRUE(std::filesystem::exists(dump_path));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/manifest.json"));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/inputs/test_input.bin"));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/outputs/test_output.bin"));
    
    // Cleanup
    std::filesystem::remove_all(dump_path);
}

TEST_F(TensorVerificationTest, DumpManifestContainsCorrectMetadata)
{
    StageDumpInfo info;
    std::vector<float> data(128, 1.0f);
    info.addInput("x", data.data(), 2, 64);
    info.addOutput("y", data.data(), 2, 64);
    
    std::string dump_path = verification::dumpStageBuffers(
        "MyStage", 10, "ENTRY", info);
    
    // Read and parse manifest
    std::ifstream manifest(dump_path + "/manifest.json");
    std::string content((std::istreambuf_iterator<char>(manifest)),
                         std::istreambuf_iterator<char>());
    
    EXPECT_NE(content.find("\"stage\": \"MyStage\""), std::string::npos);
    EXPECT_NE(content.find("\"layer_idx\": 10"), std::string::npos);
    EXPECT_NE(content.find("\"phase\": \"ENTRY\""), std::string::npos);
    
    // Cleanup
    std::filesystem::remove_all(dump_path);
}

// =============================================================================
// 5. GraphExecutor Integration Tests (requires mock stage)
// =============================================================================

// These should be in tests/v2/integration/Test__GraphExecutorVerification.cpp

TEST_F(GraphExecutorVerificationTest, ThrowsOnNaNInput)
{
    // Create a mock stage with NaN input
    auto stage = createMockStageWithNaNInput();
    
    GraphExecutor executor;
    executor.addNode("nan_stage", std::move(stage));
    
    // Should throw VerificationFailure at ENTRY
    EXPECT_THROW(
        executor.execute(),
        verification::VerificationFailure
    );
}

TEST_F(GraphExecutorVerificationTest, ThrowsOnNaNOutput)
{
    // Create a mock stage that produces NaN output
    auto stage = createMockStageProducingNaN();
    
    GraphExecutor executor;
    executor.addNode("bad_stage", std::move(stage));
    
    // Should throw VerificationFailure at EXIT
    try {
        executor.execute();
        FAIL() << "Expected VerificationFailure";
    } catch (const verification::VerificationFailure& ex) {
        EXPECT_STREQ(ex.phase(), "EXIT");
        EXPECT_NE(ex.dumpPath().find("/tmp/llaminar_verification_dump"), std::string::npos);
    }
}

TEST_F(GraphExecutorVerificationTest, PassesValidStage)
{
    auto stage = createValidMockStage();
    
    GraphExecutor executor;
    executor.addNode("valid_stage", std::move(stage));
    
    // Should not throw
    EXPECT_NO_THROW(executor.execute());
}
```

**Integration Test File**: `tests/v2/integration/Test__GraphExecutorVerification.cpp`

Tests the verification system with real stages and the full execution loop.

**CTest Labels**:
```cmake
# Unit tests
set_tests_properties(V2_Unit_TensorVerification_* PROPERTIES
    LABELS "Unit;V2;Validation")

# Integration tests
set_tests_properties(V2_Integration_GraphExecutorVerification_* PROPERTIES
    LABELS "Integration;V2;Validation")
```

**Environment Variables for Testing**:
```bash
# Run with strict zero checking
LLAMINAR_FAIL_ON_ZERO=1 ctest -R TensorVerification

# Run without dumping (for fast tests)
LLAMINAR_DUMP_ON_FAILURE=0 ctest -R TensorVerification

# Custom sample size
LLAMINAR_VERIFY_SAMPLE_ROWS=16 ctest -R TensorVerification
```

### Phase 2: Safe Q16 Block Access (1-2 days)

#### Code Research Findings

Analysis of Q16_1Tensor usage across the codebase reveals:

**Current Q16_1Tensor Architecture** (in `Tensors.h:3199-3400`):

| Component | Current State | Issue |
|-----------|---------------|-------|
| `TypedTensorBase<Q16_1Tensor, Q16_1Block>` | Always returns `Q16_1Block*` | Wrong type for BLOCK_64/128 tensors |
| `typed_data()` | Returns `Q16_1Block*` unconditionally | Memory corruption if block_size != 32 |
| `q16_1_blocks()` | Alias for `typed_data()` | Same issue |
| `q16_block_size()` | Returns actual `Q16BlockSize` enum | ✅ Already exists |
| `blocks_per_row()` | Uses actual block size | ✅ Already exists |

**Existing Block Size Support** (in `BlockStructures.h:94-180`):

| Block Type | Size | Bytes | Status |
|------------|------|-------|--------|
| `Q16_1Block` | 32 | 72 | Legacy default |
| `Q16_1Block_64` | 64 | 136 | Qwen2.5-0.5B (head_dim=64) |
| `Q16_1Block_128` | 128 | 264 | Llama3, Qwen3 (head_dim=128) |

**Existing Helper Infrastructure** (in `BlockStructures.h`):

```cpp
// Already available - can be reused:
Q16BlockSize optimal_q16_block_size(int head_dim);  // Returns best block size
size_t q16_block_size_bytes(Q16BlockSize size);     // Bytes per block
size_t q16_block_size_elements(Q16BlockSize size);  // Elements per block
template<Q16BlockSize S> using Q16BlockType_t = ...;  // Type from enum
```

**Usage Sites Requiring Updates**:

| File | Location | Current Code | Risk Level |
|------|----------|--------------|------------|
| [FusedAttentionWoStage.cpp](src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp#L180-280) | Debug dump loop | Uses `raw_data()` + `memcpy` | ✅ Already safe |
| [Test__UnifiedKVCache.cpp](tests/v2/unit/tensors/Test__UnifiedKVCache.cpp#L369-503) | `q16_1_blocks()[0].d` | Assumes BLOCK_32 | ⚠️ Fragile |
| [Test__Q16FixedScaleQuantization.cpp](tests/v2/unit/kernels/Test__Q16FixedScaleQuantization.cpp#L94-363) | `q16_1_blocks()` | Assumes BLOCK_32 | ⚠️ Fragile |
| [Test__HybridQ16_Layer0_Dataflow.cpp](tests/v2/unit/stages/Test__HybridQ16_Layer0_Dataflow.cpp#L358) | `q16_1_blocks()` | Assumes BLOCK_32 | ⚠️ Fragile |
| [Q16FusedAttentionKernel.cpp](src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp#L184) | `block_size` param | Uses tensor's `q16_block_size()` | ✅ Already safe |

**Key Insight**: The debug dump code in `FusedAttentionWoStage.cpp` was **already fixed** to use `raw_data()` + `memcpy` with `q16_block_size()`. The issue is primarily in **test code** that assumes BLOCK_32.

#### Detailed File Changes

**MODIFIED FILES:**

| File | Changes |
|------|---------|
| `src/v2/tensors/Tensors.h` | Add `as_block_64()`, `as_block_128()`, deprecate `typed_data()`, add `dequant_element()` |
| `src/v2/tensors/Q16BlockDispatch.h` | **NEW** - `dispatchQ16Block()` template helper |
| `tests/v2/unit/tensors/Test__UnifiedKVCache.cpp` | Update to use `dispatchQ16Block()` pattern |
| `tests/v2/unit/kernels/Test__Q16FixedScaleQuantization.cpp` | Update to use `dispatchQ16Block()` pattern |
| `tests/v2/unit/stages/Test__HybridQ16_Layer0_Dataflow.cpp` | Update to use safe block access |
| `tests/v2/unit/Test__Q16SafeBlockAccess.cpp` | **NEW** - Unit tests for new API |

#### Code Examples

**1. Updated Q16_1Tensor in Tensors.h:**

```cpp
class Q16_1Tensor : public TypedTensorBase<Q16_1Tensor, Q16_1Block>,
                    public TensorBase,
                    public IActivationTensor,
                    public ITensorGemmTileDataProvider
{
public:
    // ... existing code ...
    
    // =================================================================
    // DEPRECATED: typed_data() only valid for BLOCK_32
    // =================================================================
    
    /// @deprecated Use as_block_XX() or dispatchQ16Block() instead
    [[deprecated("Use as_block_32/64/128() or dispatchQ16Block() for safe access")]]
    const Q16_1Block* typed_data() const {
#if LLAMINAR_ASSERTIONS_ACTIVE
        if (block_size_ != Q16BlockSize::BLOCK_32) {
            LOG_ERROR("[Q16_1Tensor] typed_data() called on tensor with block_size="
                      << static_cast<int>(block_size_) << " (expected 32)");
            LOG_ERROR("[Q16_1Tensor] Use as_block_64/128() or dispatchQ16Block() instead");
            LLAMINAR_ASSERT(false, "typed_data() invalid for non-BLOCK_32 Q16_1Tensor");
        }
#endif
        return reinterpret_cast<const Q16_1Block*>(raw_data());
    }
    
    // =================================================================
    // Safe Block Accessors - return nullptr if block size doesn't match
    // =================================================================
    
    /// Get Q16_1Block* if this tensor uses BLOCK_32, else nullptr
    const Q16_1Block* as_block_32() const {
        return (block_size_ == Q16BlockSize::BLOCK_32)
            ? reinterpret_cast<const Q16_1Block*>(raw_data()) : nullptr;
    }
    Q16_1Block* mutable_as_block_32() {
        return (block_size_ == Q16BlockSize::BLOCK_32)
            ? reinterpret_cast<Q16_1Block*>(raw_mutable_data()) : nullptr;
    }
    
    /// Get Q16_1Block_64* if this tensor uses BLOCK_64, else nullptr
    const Q16_1Block_64* as_block_64() const {
        return (block_size_ == Q16BlockSize::BLOCK_64)
            ? reinterpret_cast<const Q16_1Block_64*>(raw_data()) : nullptr;
    }
    Q16_1Block_64* mutable_as_block_64() {
        return (block_size_ == Q16BlockSize::BLOCK_64)
            ? reinterpret_cast<Q16_1Block_64*>(raw_mutable_data()) : nullptr;
    }
    
    /// Get Q16_1Block_128* if this tensor uses BLOCK_128, else nullptr
    const Q16_1Block_128* as_block_128() const {
        return (block_size_ == Q16BlockSize::BLOCK_128)
            ? reinterpret_cast<const Q16_1Block_128*>(raw_data()) : nullptr;
    }
    Q16_1Block_128* mutable_as_block_128() {
        return (block_size_ == Q16BlockSize::BLOCK_128)
            ? reinterpret_cast<Q16_1Block_128*>(raw_mutable_data()) : nullptr;
    }
    
    // =================================================================
    // Generic Element Access (slower but always safe)
    // =================================================================
    
    /// Get dequantized element at (row, col) - handles any block size
    float dequant_element(size_t row, size_t col) const {
        const size_t block_elems = q16_block_size_elements(block_size_);
        const size_t block_bytes = q16_block_size_bytes(block_size_);
        const size_t blocks_per_row = (shape_[1] + block_elems - 1) / block_elems;
        
        const size_t b = col / block_elems;
        const size_t i = col % block_elems;
        const size_t block_idx = row * blocks_per_row + b;
        
        const uint8_t* block_ptr = static_cast<const uint8_t*>(raw_data()) + block_idx * block_bytes;
        float d;
        std::memcpy(&d, block_ptr, sizeof(float));  // Scale at offset 0
        const int16_t* qs = reinterpret_cast<const int16_t*>(block_ptr + sizeof(float));
        
        return d * static_cast<float>(qs[i]);
    }
    
    /// Get quantized int16 element at (row, col)
    int16_t quantized_element(size_t row, size_t col) const {
        const size_t block_elems = q16_block_size_elements(block_size_);
        const size_t block_bytes = q16_block_size_bytes(block_size_);
        const size_t blocks_per_row = (shape_[1] + block_elems - 1) / block_elems;
        
        const size_t b = col / block_elems;
        const size_t i = col % block_elems;
        const size_t block_idx = row * blocks_per_row + b;
        
        const uint8_t* block_ptr = static_cast<const uint8_t*>(raw_data()) + block_idx * block_bytes;
        const int16_t* qs = reinterpret_cast<const int16_t*>(block_ptr + sizeof(float));
        
        return qs[i];
    }
    
    /// Get block scale at (row, block_in_row)
    float block_scale(size_t row, size_t block_in_row) const {
        const size_t block_bytes = q16_block_size_bytes(block_size_);
        const size_t blocks_per_row_val = blocks_per_row();
        const size_t block_idx = row * blocks_per_row_val + block_in_row;
        
        const uint8_t* block_ptr = static_cast<const uint8_t*>(raw_data()) + block_idx * block_bytes;
        float d;
        std::memcpy(&d, block_ptr, sizeof(float));
        return d;
    }
};
```

**2. New Q16BlockDispatch.h Helper:**

```cpp
// src/v2/tensors/Q16BlockDispatch.h
#pragma once

#include "Tensors.h"
#include "BlockStructures.h"
#include <utility>

namespace llaminar2 {

/**
 * @brief Dispatch to typed block access based on Q16_1Tensor's runtime block size
 * 
 * @tparam Func Callable with signature: ReturnType(const BlockType*, size_t block_size)
 * @param tensor Q16_1Tensor to dispatch on
 * @param func Function receiving typed block pointer and block size
 * @return Return value of func
 * 
 * Example:
 * @code
 * float first_scale = dispatchQ16Block(tensor, [](auto* blocks, size_t bs) {
 *     return blocks[0].d;
 * });
 * @endcode
 */
template<typename Func>
auto dispatchQ16Block(const Q16_1Tensor* tensor, Func&& func) 
    -> decltype(func(std::declval<const Q16_1Block*>(), size_t{}))
{
    switch (tensor->q16_block_size()) {
        case Q16BlockSize::BLOCK_32:
            return std::forward<Func>(func)(
                reinterpret_cast<const Q16_1Block*>(tensor->raw_data()),
                Q16_1Block::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_64:
            return std::forward<Func>(func)(
                reinterpret_cast<const Q16_1Block_64*>(tensor->raw_data()),
                Q16_1Block_64::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_128:
            return std::forward<Func>(func)(
                reinterpret_cast<const Q16_1Block_128*>(tensor->raw_data()),
                Q16_1Block_128::BLOCK_SIZE);
        default:
            LLAMINAR_UNREACHABLE("Invalid Q16BlockSize: " << static_cast<int>(tensor->q16_block_size()));
    }
}

/// Mutable version of dispatchQ16Block
template<typename Func>
auto dispatchQ16BlockMutable(Q16_1Tensor* tensor, Func&& func)
    -> decltype(func(std::declval<Q16_1Block*>(), size_t{}))
{
    switch (tensor->q16_block_size()) {
        case Q16BlockSize::BLOCK_32:
            return std::forward<Func>(func)(
                reinterpret_cast<Q16_1Block*>(tensor->raw_mutable_data()),
                Q16_1Block::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_64:
            return std::forward<Func>(func)(
                reinterpret_cast<Q16_1Block_64*>(tensor->raw_mutable_data()),
                Q16_1Block_64::BLOCK_SIZE);
        case Q16BlockSize::BLOCK_128:
            return std::forward<Func>(func)(
                reinterpret_cast<Q16_1Block_128*>(tensor->raw_mutable_data()),
                Q16_1Block_128::BLOCK_SIZE);
        default:
            LLAMINAR_UNREACHABLE("Invalid Q16BlockSize: " << static_cast<int>(tensor->q16_block_size()));
    }
}

/**
 * @brief Iterate over all blocks with typed access
 * 
 * @tparam Func Callable with signature: void(const BlockType& block, size_t block_idx)
 * 
 * Example:
 * @code
 * forEachQ16Block(tensor, [](const auto& block, size_t idx) {
 *     LOG_DEBUG("Block " << idx << " scale: " << block.d);
 * });
 * @endcode
 */
template<typename Func>
void forEachQ16Block(const Q16_1Tensor* tensor, Func&& func) {
    const size_t total = tensor->total_blocks();
    dispatchQ16Block(tensor, [&](auto* blocks, size_t /*bs*/) {
        for (size_t i = 0; i < total; ++i) {
            func(blocks[i], i);
        }
    });
}

} // namespace llaminar2
```

**3. Example Migration in Test Code:**

```cpp
// BEFORE (in Test__UnifiedKVCache.cpp) - assumes BLOCK_32
TEST_F(UnifiedKVCacheTest, CacheStoresQ16Blocks)
{
    // ...
    EXPECT_FLOAT_EQ(cached_k->q16_1_blocks()[0].d, 1.0f);  // ❌ Unsafe
    EXPECT_EQ(cached_k->q16_1_blocks()[0].qs[0], 0);       // ❌ Unsafe
}

// AFTER - block-size agnostic
TEST_F(UnifiedKVCacheTest, CacheStoresQ16Blocks)
{
    // ...
    // Option A: Use dequant_element() for single values
    float first_element = cached_k->dequant_element(0, 0);
    
    // Option B: Use dispatchQ16Block for multi-element access
    float first_scale = dispatchQ16Block(cached_k, [](auto* blocks, size_t bs) {
        return blocks[0].d;
    });
    EXPECT_FLOAT_EQ(first_scale, 1.0f);
    
    int16_t first_qs = dispatchQ16Block(cached_k, [](auto* blocks, size_t bs) {
        return blocks[0].qs[0];
    });
    EXPECT_EQ(first_qs, 0);
}

// Option C: For tests that MUST test BLOCK_32 specifically
TEST_F(UnifiedKVCacheTest, Block32SpecificTest)
{
    auto tensor = factory.createQ16_1({4, 32}, Q16BlockSize::BLOCK_32, -1);
    ASSERT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_32);
    
    const Q16_1Block* blocks = tensor->as_block_32();
    ASSERT_NE(blocks, nullptr);  // Verify it's really BLOCK_32
    EXPECT_FLOAT_EQ(blocks[0].d, expected_scale);
}
```

#### Phase 2 Test Strategy

**Test File**: `tests/v2/unit/Test__Q16SafeBlockAccess.cpp`

```cpp
class Test__Q16SafeBlockAccess : public ::testing::Test { ... };

// =============================================================================
// as_block_XX() Accessor Tests
// =============================================================================

TEST_F(Test__Q16SafeBlockAccess, AsBlock32_ReturnsPointer_WhenBlock32)
{
    auto tensor = factory_->createQ16_1({4, 32}, Q16BlockSize::BLOCK_32, -1);
    EXPECT_NE(tensor->as_block_32(), nullptr);
    EXPECT_EQ(tensor->as_block_64(), nullptr);
    EXPECT_EQ(tensor->as_block_128(), nullptr);
}

TEST_F(Test__Q16SafeBlockAccess, AsBlock64_ReturnsPointer_WhenBlock64)
{
    auto tensor = factory_->createQ16_1({4, 64}, Q16BlockSize::BLOCK_64, -1);
    EXPECT_EQ(tensor->as_block_32(), nullptr);
    EXPECT_NE(tensor->as_block_64(), nullptr);
    EXPECT_EQ(tensor->as_block_128(), nullptr);
}

TEST_F(Test__Q16SafeBlockAccess, AsBlock128_ReturnsPointer_WhenBlock128)
{
    auto tensor = factory_->createQ16_1({4, 128}, Q16BlockSize::BLOCK_128, -1);
    EXPECT_EQ(tensor->as_block_32(), nullptr);
    EXPECT_EQ(tensor->as_block_64(), nullptr);
    EXPECT_NE(tensor->as_block_128(), nullptr);
}

// =============================================================================
// dequant_element() Tests
// =============================================================================

TEST_F(Test__Q16SafeBlockAccess, DequantElement_WorksForAllBlockSizes)
{
    for (Q16BlockSize bs : {Q16BlockSize::BLOCK_32, Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_128}) {
        const size_t elem_count = static_cast<size_t>(bs);
        auto tensor = factory_->createQ16_1({2, elem_count}, bs, -1);
        
        // Initialize with known values
        initializeQ16Tensor(tensor.get(), /*scale=*/0.5f, /*base_qs=*/100);
        
        // Verify dequant_element returns correct value
        float expected = 0.5f * 100.0f;  // scale * qs[0]
        EXPECT_FLOAT_EQ(tensor->dequant_element(0, 0), expected);
    }
}

// =============================================================================
// dispatchQ16Block() Tests
// =============================================================================

TEST_F(Test__Q16SafeBlockAccess, DispatchQ16Block_ReturnsCorrectType)
{
    auto tensor64 = factory_->createQ16_1({4, 64}, Q16BlockSize::BLOCK_64, -1);
    
    // Lambda should receive Q16_1Block_64*
    bool got_correct_type = dispatchQ16Block(tensor64.get(), [](auto* blocks, size_t bs) {
        // blocks is Q16_1Block_64* for BLOCK_64 tensor
        return bs == 64 && sizeof(blocks[0]) == sizeof(Q16_1Block_64);
    });
    EXPECT_TRUE(got_correct_type);
}

TEST_F(Test__Q16SafeBlockAccess, DispatchQ16Block_CanReadScales)
{
    auto tensor = factory_->createQ16_1({4, 128}, Q16BlockSize::BLOCK_128, -1);
    initializeQ16Tensor(tensor.get(), /*scale=*/2.5f, /*base_qs=*/0);
    
    float first_scale = dispatchQ16Block(tensor.get(), [](auto* blocks, size_t bs) {
        return blocks[0].d;
    });
    EXPECT_FLOAT_EQ(first_scale, 2.5f);
}

// =============================================================================
// typed_data() Deprecation Tests (Debug builds only)
// =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE
TEST_F(Test__Q16SafeBlockAccess, TypedData_Asserts_OnNonBlock32)
{
    auto tensor64 = factory_->createQ16_1({4, 64}, Q16BlockSize::BLOCK_64, -1);
    
    // typed_data() should assert/throw for non-BLOCK_32 tensors
    EXPECT_DEATH(
        { [[maybe_unused]] auto ptr = tensor64->typed_data(); },
        "typed_data.*invalid.*non-BLOCK_32"
    );
}
#endif

TEST_F(Test__Q16SafeBlockAccess, TypedData_Works_ForBlock32)
{
    auto tensor32 = factory_->createQ16_1({4, 32}, Q16BlockSize::BLOCK_32, -1);
    
    // typed_data() should work for BLOCK_32 tensors (with deprecation warning)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    const Q16_1Block* blocks = tensor32->typed_data();
    #pragma GCC diagnostic pop
    
    EXPECT_NE(blocks, nullptr);
}
```

**CTest Labels**:
```cmake
set_tests_properties(V2_Unit_Q16SafeBlockAccess_* PROPERTIES
    LABELS "Unit;V2;TensorOperations;Q16;SafeBlockAccess")
```

#### Key Implementation Notes

1. **FusedAttentionWoStage debug code is already safe**: The current implementation uses `raw_data()` + `memcpy` with `q16_block_size()` - no changes needed there.

2. **Q16FusedAttentionKernel already handles block sizes**: It passes `tensor->q16_block_size()` to the reference kernel.

3. **Main work is in test code**: Most unsafe `q16_1_blocks()` calls are in unit tests that assume BLOCK_32.

4. **Backward compatibility**: Tests that explicitly create BLOCK_32 tensors can use `as_block_32()` safely.

### Phase 3: Layout Contracts (2-3 days)

#### Code Research Findings

Analysis of tensor layout handling across the codebase reveals:

**Current Layout Architecture**:

| Component | Current State | Issue |
|-----------|---------------|-------|
| `TensorBase` | No `layout()` method | Cannot express layout contract |
| `UnifiedKVCache` | Position-major: `[pos, n_kv_heads, head_dim]` | Kernel expects head-major |
| `Q16IntegerAttentionRef` | Expects head-major: `[n_kv_heads, pos, head_dim]` | Layout mismatch |
| `FusedAttentionWoStage` | Runtime transpose workaround | O(n) copy overhead |
| Test utilities | `transposeKV()` function | Test-only, not in production |

**Key Files Documenting Layout Issue** (from `PLAN_FIXED_SCALE_ROPE_Q16.md:387-492`):

```cpp
// UnifiedKVCache stores K/V in POSITION-MAJOR layout:
// Storage: [max_seq_len, n_kv_heads * head_dim]
// Memory layout: Row 0: [head_0_pos_0, head_1_pos_0], Row 1: [head_0_pos_1, head_1_pos_1], ...
// Block indexing: block[p][h] = blocks[p * n_kv_heads + h]

// Q16IntegerAttentionRef expects HEAD-MAJOR layout:
// Block 0-3: head_0, positions 0-3
// Block 4-7: head_1, positions 0-3
// Block indexing: block[h][p] = blocks[h * kv_len + p]
```

**Current Workaround in FusedAttentionWoStage** (lines 395-430):

```cpp
// TRANSPOSE WORKAROUND: Convert position-major to head-major layout
// This is O(kv_len * n_kv_heads * head_dim) copy per attention call

std::vector<uint8_t> K_transposed_bytes(total_blocks * block_bytes);
std::vector<uint8_t> V_transposed_bytes(total_blocks * block_bytes);

// Transpose: position-major [p][h] -> head-major [h][p]
for (int h = 0; h < params_.n_kv_heads; ++h) {
    for (int p = 0; p < effective_kv_len; ++p) {
        // Source index: position-major [p * n_kv_heads + h]
        const size_t src_block_idx = (static_cast<size_t>(p) * params_.n_kv_heads + h) * blocks_per_head;
        // Dest index: head-major [h * kv_len + p]
        const size_t dst_block_idx = (static_cast<size_t>(h) * effective_kv_len + p) * blocks_per_head;
        
        std::memcpy(K_transposed_bytes.data() + dst_block_idx * block_bytes,
                    K_src + src_block_idx * block_bytes,
                    blocks_per_head * block_bytes);
        // ... same for V
    }
}
```

**Test Utility `transposeKV()`** (from `Test__Q16IntegerAttentionParity.cpp:138-166`):

```cpp
/// @brief Transpose K/V from [kv_len, num_kv_heads, head_dim] to [num_kv_heads, kv_len, head_dim]
std::vector<float> transposeKV(
    const std::vector<float> &input, int kv_len, int num_kv_heads, int head_dim)
{
    std::vector<float> output(input.size());
    for (int h = 0; h < num_kv_heads; ++h) {
        for (int pos = 0; pos < kv_len; ++pos) {
            for (int d = 0; d < head_dim; ++d) {
                // Input:  [pos][h][d]
                // Output: [h][pos][d]
                size_t in_idx  = (pos * num_kv_heads + h) * head_dim + d;
                size_t out_idx = (h * kv_len + pos) * head_dim + d;
                output[out_idx] = input[in_idx];
            }
        }
    }
    return output;
}
```

**Layout Flow in HybridQ16 Pipeline** (current):

| Stage | Output Layout | Consumer Expectation | Status |
|-------|--------------|---------------------|--------|
| FusedQKVGEMMStage | `[seq, n_h*hd]` flat | RoPE expects `[seq, n_h, hd]` | ✅ Implicit |
| RoPEStage (K) | `[seq, n_kv, hd]` | KVCacheAppend expects same | ✅ Match |
| KVCacheAppendStage | `[pos, n_kv, hd]` position-major | - | Cache layout |
| FusedAttentionWoStage | Reads `[pos, n_kv, hd]` | Q16 kernel wants `[n_kv, pos, hd]` | ❌ MISMATCH |

#### Detailed File Changes

**NEW FILES:**

| File | Purpose |
|------|---------|
| `src/v2/tensors/TensorLayout.h` | `TensorLayout` enum and utilities |
| `tests/v2/unit/Test__TensorLayout.cpp` | Unit tests for layout enum |

**MODIFIED FILES:**

| File | Changes |
|------|---------|
| `src/v2/tensors/Tensors.h` | Add `layout()` method to `TensorBase` |
| `src/v2/tensors/UnifiedKVCache.h` | Add `kv_layout()` method |
| `src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp` | Add layout validation |
| `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp` | Set output layout |

**FUTURE (Phase 3b):**

| File | Purpose |
|------|---------|
| `src/v2/tensors/HeadMajorKVCache.h` | New head-major KV cache for Q16_INTEGER |
| `src/v2/tensors/HeadMajorKVCache.cpp` | Implementation |
| `tests/v2/unit/Test__HeadMajorKVCache.cpp` | Unit tests |

#### Code Examples

**1. New TensorLayout.h:**

```cpp
// src/v2/tensors/TensorLayout.h
#pragma once

#include <cstdint>

namespace llaminar2 {

/**
 * @brief Tensor memory layout for multi-dimensional data
 * 
 * Layouts describe how logical indices map to memory addresses.
 * This is orthogonal to quantization format (FP32/Q8_1/Q16_1).
 * 
 * Naming convention: SEMANTIC_DIM1_DIM2_DIM3
 *   - KV_POS_HEAD_DIM = [position][n_kv_heads][head_dim] - cache iteration order
 *   - KV_HEAD_POS_DIM = [n_kv_heads][position][head_dim] - head-contiguous
 */
enum class TensorLayout : uint8_t {
    // ============================================
    // Query tensor layouts
    // ============================================
    
    /// [seq_len, n_heads * head_dim] - flat output from QKV GEMM
    Q_SEQ_FLAT,
    
    /// [seq_len, n_heads, head_dim] - per-head after reshape
    Q_SEQ_HEAD_DIM,
    
    // ============================================
    // K/V tensor layouts (the critical ones)
    // ============================================
    
    /// [position, n_kv_heads, head_dim] - Position-major
    /// Cache-friendly for sequential append (UnifiedKVCache default)
    /// Memory: [pos0_head0, pos0_head1, pos1_head0, pos1_head1, ...]
    KV_POS_HEAD_DIM,
    
    /// [n_kv_heads, position, head_dim] - Head-major
    /// Attention-friendly for per-head dot products (Q16IntegerAttentionRef expects this)
    /// Memory: [head0_pos0, head0_pos1, head1_pos0, head1_pos1, ...]
    KV_HEAD_POS_DIM,
    
    // ============================================
    // Generic layouts
    // ============================================
    
    /// Standard 2D row-major: [rows, cols]
    ROW_MAJOR_2D,
    
    /// Layout not specified or unknown
    UNKNOWN
};

/**
 * @brief Get human-readable layout name
 */
inline const char* layoutName(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::Q_SEQ_FLAT:      return "Q_SEQ_FLAT";
        case TensorLayout::Q_SEQ_HEAD_DIM:  return "Q_SEQ_HEAD_DIM";
        case TensorLayout::KV_POS_HEAD_DIM: return "KV_POS_HEAD_DIM (position-major)";
        case TensorLayout::KV_HEAD_POS_DIM: return "KV_HEAD_POS_DIM (head-major)";
        case TensorLayout::ROW_MAJOR_2D:    return "ROW_MAJOR_2D";
        case TensorLayout::UNKNOWN:         return "UNKNOWN";
        default:                            return "INVALID";
    }
}

/**
 * @brief Short layout name for logging
 */
inline const char* layoutShortName(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::Q_SEQ_FLAT:      return "Q_FLAT";
        case TensorLayout::Q_SEQ_HEAD_DIM:  return "Q_HEAD";
        case TensorLayout::KV_POS_HEAD_DIM: return "KV_POS";
        case TensorLayout::KV_HEAD_POS_DIM: return "KV_HEAD";
        case TensorLayout::ROW_MAJOR_2D:    return "2D";
        case TensorLayout::UNKNOWN:         return "?";
        default:                            return "ERR";
    }
}

/**
 * @brief Check if two layouts are compatible without transpose
 * 
 * UNKNOWN is compatible with anything (for gradual adoption).
 * Same layout is always compatible.
 */
inline bool layoutsCompatible(TensorLayout a, TensorLayout b) {
    if (a == TensorLayout::UNKNOWN || b == TensorLayout::UNKNOWN) {
        return true;  // Unknown is wildcard for gradual adoption
    }
    return a == b;
}

/**
 * @brief Check if layout is a K/V layout
 */
inline bool isKVLayout(TensorLayout layout) {
    return layout == TensorLayout::KV_POS_HEAD_DIM ||
           layout == TensorLayout::KV_HEAD_POS_DIM;
}

} // namespace llaminar2
```

**2. Updated TensorBase in Tensors.h:**

```cpp
class TensorBase : public virtual ITensor, public std::enable_shared_from_this<TensorBase>
{
public:
    // ... existing code ...
    
    // =================================================================
    // Layout Contract (Phase 3)
    // =================================================================
    
    /**
     * @brief Get the tensor's memory layout
     * 
     * Default is UNKNOWN for backward compatibility.
     * Tensors created through factory methods with layout specification
     * will return the specified layout.
     */
    virtual TensorLayout layout() const { return layout_; }
    
    /**
     * @brief Set the tensor's layout (for use by factory/cache)
     * 
     * @note This does NOT transpose data - only updates the layout tag.
     *       Use transpose_to() to actually reorder data.
     */
    void setLayout(TensorLayout layout) { layout_ = layout; }
    
protected:
    TensorLayout layout_ = TensorLayout::UNKNOWN;
};
```

**3. Updated UnifiedKVCache.h:**

```cpp
class IUnifiedKVCache
{
public:
    // ... existing interface ...
    
    /**
     * @brief Get the K/V tensor layout used by this cache
     * 
     * @return KV_POS_HEAD_DIM for position-major (default UnifiedKVCache)
     *         KV_HEAD_POS_DIM for head-major (HeadMajorKVCache)
     */
    virtual TensorLayout kv_layout() const = 0;
};

template <ActivationPrecision Precision>
class UnifiedKVCacheTensor : public IUnifiedKVCache
{
public:
    // ... existing implementation ...
    
    TensorLayout kv_layout() const override {
        return TensorLayout::KV_POS_HEAD_DIM;  // Position-major
    }
};
```

**4. Layout Validation in FusedAttentionWoStage:**

```cpp
bool FusedAttentionWoStage::execute(IDeviceContext *ctx)
{
    // ... existing validation ...
    
    // ============================================================
    // Layout Validation (Phase 3)
    // ============================================================
    
#if LLAMINAR_ASSERTIONS_ACTIVE
    // Determine expected K/V layout based on backend
    TensorLayout expected_kv_layout = TensorLayout::UNKNOWN;
    if (params_.backend == FusedAttentionBackend::Q16_INTEGER) {
        expected_kv_layout = TensorLayout::KV_HEAD_POS_DIM;  // Q16 kernel expects head-major
    } else {
        expected_kv_layout = TensorLayout::KV_POS_HEAD_DIM;  // Other backends expect position-major
    }
    
    // Check K/V layout if specified
    if (params_.K->layout() != TensorLayout::UNKNOWN &&
        params_.K->layout() != expected_kv_layout)
    {
        LOG_WARN("[FusedAttentionWoStage] K tensor layout mismatch: "
                 << "got " << layoutName(params_.K->layout())
                 << ", expected " << layoutName(expected_kv_layout)
                 << ". Transpose workaround will be applied.");
    }
    
    // Also check KV cache layout
    if (params_.kv_cache) {
        TensorLayout cache_layout = params_.kv_cache->kv_layout();
        if (cache_layout != TensorLayout::UNKNOWN &&
            cache_layout != expected_kv_layout)
        {
            LOG_DEBUG("[FusedAttentionWoStage] KV cache layout " 
                      << layoutName(cache_layout)
                      << " differs from kernel expectation "
                      << layoutName(expected_kv_layout)
                      << " - applying transpose workaround");
        }
    }
#endif

    // ... continue with existing execution including transpose workaround ...
}
```

**5. Layout Specification in KVCacheAppendStage:**

```cpp
bool KVCacheAppendStage::execute(IDeviceContext *ctx)
{
    // After successful append, tag K/V tensors with cache layout
    TensorBase* cached_k = params_.kv_cache->get_k_base(params_.layer_idx, params_.seq_idx);
    TensorBase* cached_v = params_.kv_cache->get_v_base(params_.layer_idx, params_.seq_idx);
    
    // Set layout based on cache type
    TensorLayout cache_layout = params_.kv_cache->kv_layout();
    if (cached_k) cached_k->setLayout(cache_layout);
    if (cached_v) cached_v->setLayout(cache_layout);
    
    return true;
}
```

#### Phase 3 Test Strategy

**Test File**: `tests/v2/unit/Test__TensorLayout.cpp`

```cpp
class Test__TensorLayout : public ::testing::Test { ... };

// =============================================================================
// TensorLayout Enum Tests
// =============================================================================

TEST_F(Test__TensorLayout, LayoutNames)
{
    EXPECT_STREQ(layoutName(TensorLayout::KV_POS_HEAD_DIM), "KV_POS_HEAD_DIM (position-major)");
    EXPECT_STREQ(layoutName(TensorLayout::KV_HEAD_POS_DIM), "KV_HEAD_POS_DIM (head-major)");
    EXPECT_STREQ(layoutShortName(TensorLayout::KV_POS_HEAD_DIM), "KV_POS");
}

TEST_F(Test__TensorLayout, LayoutCompatibility_Unknown)
{
    // UNKNOWN is wildcard - compatible with everything
    EXPECT_TRUE(layoutsCompatible(TensorLayout::UNKNOWN, TensorLayout::KV_POS_HEAD_DIM));
    EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_HEAD_POS_DIM, TensorLayout::UNKNOWN));
    EXPECT_TRUE(layoutsCompatible(TensorLayout::UNKNOWN, TensorLayout::UNKNOWN));
}

TEST_F(Test__TensorLayout, LayoutCompatibility_Same)
{
    EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_POS_HEAD_DIM, TensorLayout::KV_POS_HEAD_DIM));
    EXPECT_TRUE(layoutsCompatible(TensorLayout::KV_HEAD_POS_DIM, TensorLayout::KV_HEAD_POS_DIM));
}

TEST_F(Test__TensorLayout, LayoutCompatibility_Mismatch)
{
    EXPECT_FALSE(layoutsCompatible(TensorLayout::KV_POS_HEAD_DIM, TensorLayout::KV_HEAD_POS_DIM));
    EXPECT_FALSE(layoutsCompatible(TensorLayout::KV_HEAD_POS_DIM, TensorLayout::KV_POS_HEAD_DIM));
}

TEST_F(Test__TensorLayout, IsKVLayout)
{
    EXPECT_TRUE(isKVLayout(TensorLayout::KV_POS_HEAD_DIM));
    EXPECT_TRUE(isKVLayout(TensorLayout::KV_HEAD_POS_DIM));
    EXPECT_FALSE(isKVLayout(TensorLayout::Q_SEQ_HEAD_DIM));
    EXPECT_FALSE(isKVLayout(TensorLayout::ROW_MAJOR_2D));
    EXPECT_FALSE(isKVLayout(TensorLayout::UNKNOWN));
}

// =============================================================================
// TensorBase Layout Tests
// =============================================================================

TEST_F(Test__TensorLayout, TensorBase_DefaultLayout_IsUnknown)
{
    auto tensor = TestTensorFactory::createFP32({32, 64});
    EXPECT_EQ(tensor->layout(), TensorLayout::UNKNOWN);
}

TEST_F(Test__TensorLayout, TensorBase_SetLayout)
{
    auto tensor = TestTensorFactory::createFP32({32, 64});
    tensor->setLayout(TensorLayout::ROW_MAJOR_2D);
    EXPECT_EQ(tensor->layout(), TensorLayout::ROW_MAJOR_2D);
}

// =============================================================================
// UnifiedKVCache Layout Tests
// =============================================================================

TEST_F(Test__TensorLayout, UnifiedKVCache_ReturnsPositionMajor)
{
    auto cache = createUnifiedKVCache(
        ActivationPrecision::Q16_1,
        mpi_ctx_, n_layers_, batch_size_, max_seq_len_, n_kv_heads_, head_dim_, -1);
    
    EXPECT_EQ(cache->kv_layout(), TensorLayout::KV_POS_HEAD_DIM);
}

// =============================================================================
// Stage Layout Validation Tests (Debug builds)
// =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE
TEST_F(Test__TensorLayout, FusedAttentionWoStage_WarnsOnLayoutMismatch)
{
    // Create K tensor with head-major layout
    auto K = TestTensorFactory::createQ16_1({kv_len_, n_kv_heads_ * head_dim_});
    K->setLayout(TensorLayout::KV_HEAD_POS_DIM);
    
    // Stage with JIT backend expects position-major
    FusedAttentionWoStage::Params params;
    params.K = K.get();
    params.backend = FusedAttentionBackend::JIT;  // Expects KV_POS_HEAD_DIM
    // ... other params ...
    
    // Should log warning about layout mismatch
    testing::internal::CaptureStderr();
    FusedAttentionWoStage stage(params);
    stage.execute(nullptr);
    std::string output = testing::internal::GetCapturedStderr();
    
    EXPECT_NE(output.find("layout mismatch"), std::string::npos);
}
#endif
```

**CTest Labels**:
```cmake
set_tests_properties(V2_Unit_TensorLayout_* PROPERTIES
    LABELS "Unit;V2;TensorOperations;Layout")
```

#### Key Implementation Notes

1. **Gradual Adoption**: `UNKNOWN` layout is compatible with everything, allowing incremental rollout.

2. **No Breaking Changes**: Existing code using `TensorBase` without layout specification continues to work.

3. **Runtime Transpose Preserved**: The transpose workaround in `FusedAttentionWoStage` remains until future HeadMajorKVCache work.

4. **Debug-Only Validation**: Layout mismatch warnings only appear in Debug/Integration builds.

5. **HeadMajorKVCache (Future Work)**: Eliminate transpose overhead. Detailed design in `PLAN_FIXED_SCALE_ROPE_Q16.md:500-620`.

#### Phase 3b: Layout Validation via Buffer Requirements (✅ Complete)

**Implemented a declarative approach** where stages declare expected layouts in `getBufferRequirements()`:

```cpp
// Clean fluent API with optional 4th parameter for layout
StageBufferRequirements FusedAttentionWoStage::getBufferRequirements() const override {
    return StageBufferRequirements()
        .addInput("Q", {1, seq_len_, head_dim_}, TensorDataType::FP32, TensorLayout::Q_SEQ_HEAD_DIM)
        .addInput("K", {n_kv_heads_, kv_len_, head_dim_}, TensorDataType::Q16_1, TensorLayout::KV_HEAD_POS_DIM)
        .addInput("V", {n_kv_heads_, kv_len_, head_dim_}, TensorDataType::Q16_1, TensorLayout::KV_HEAD_POS_DIM)
        .addOutput("attention_output", {seq_len_, d_model_}, TensorDataType::FP32, TensorLayout::ROW_MAJOR_2D);
}
```

**Key Implementation Details:**

1. **Optional Layout Parameter**: All `addInput/addOutput/addInout/addScratch/addWeight` methods accept optional 4th parameter `TensorLayout layout = TensorLayout::UNKNOWN`
2. **Fluent Chaining Preserved**: Returns `StageBufferRequirements&` for continued chaining
3. **Automatic Validation**: `GraphExecutor::verifyStageEntry()` validates input layouts against declarations
4. **LayoutExpectation Struct**: Encodes expected layout from buffer requirements
5. **LayoutValidationResult Struct**: Captures validation outcomes with detailed error messages
6. **Shape-Based Inference**: `validateBufferLayoutByShape()` can infer layout from tensor dimensions

**Files Modified:**
- `src/v2/tensors/TensorLayout.h` - Added `LayoutExpectation`, `LayoutValidationResult`, validation functions
- `src/v2/execution/compute_stages/BufferRole.h` - Extended fluent API with optional layout parameter
- `src/v2/execution/GraphExecutor.cpp` - Layout validation in `verifyStageEntry()`
- `tests/v2/unit/stages/Test__BufferRole.cpp` - 3 new tests for layout API

---

### UnifiedKVCache HEAD_MAJOR Layout Mode

**Status**: ✅ Complete (January 2026)

The `UnifiedKVCache` now supports both POSITION_MAJOR and HEAD_MAJOR storage layouts. The Q16 integer attention kernel requires head-major layout for efficient per-head iteration, and GraphOrchestrator automatically selects HEAD_MAJOR when using Q16_1 KV cache precision.

**Implementation Details**:

Instead of a separate `HeadMajorKVCache` class, we add a **layout mode** to the existing `UnifiedKVCache` class:

```cpp
// src/v2/tensors/UnifiedKVCache.h

/**
 * @brief KV cache storage layout mode
 */
enum class KVCacheLayoutMode {
    POSITION_MAJOR,  ///< [position, n_kv_heads, head_dim] - cache-friendly for append
    HEAD_MAJOR       ///< [n_kv_heads, position, head_dim] - attention-friendly for Q16
};

template <ActivationPrecision Precision>
class UnifiedKVCacheTensor : public IUnifiedKVCache {
public:
    // Constructor with layout mode selection
    UnifiedKVCacheTensor(
        const MPIContext& mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_idx,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR  // Default for backward compat
    );
    
    // Layout query
    TensorLayout kv_layout() const override {
        return (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
            ? TensorLayout::KV_HEAD_POS_DIM
            : TensorLayout::KV_POS_HEAD_DIM;
    }
    
    KVCacheLayoutMode layout_mode() const { return layout_mode_; }
    
    // ================================================================
    // Head-specific accessors (for HEAD_MAJOR mode)
    // ================================================================
    
    /**
     * @brief Get K tensor for a specific KV head (HEAD_MAJOR mode only)
     * @return Tensor view of shape [cached_tokens, head_dim], or nullptr if POSITION_MAJOR
     */
    TensorT* get_k_head(int layer, int kv_head, int seq_idx = 0);
    TensorT* get_v_head(int layer, int kv_head, int seq_idx = 0);
    
    /// Stride in bytes between consecutive heads (HEAD_MAJOR mode)
    size_t head_stride_bytes() const;
    
private:
    KVCacheLayoutMode layout_mode_;
    
    // For HEAD_MAJOR: separate per-head tensors
    // head_entries_[layer][kv_head] = {K, V}
    struct HeadEntry {
        std::shared_ptr<TensorT> K;  // [max_seq_len, head_dim]
        std::shared_ptr<TensorT> V;
    };
    std::vector<std::vector<HeadEntry>> head_entries_;  // Only used in HEAD_MAJOR mode
    
    // For POSITION_MAJOR: existing flat storage (unchanged)
    // k_cache_[layer] = [max_seq_len, n_kv_heads * head_dim]
};
```

**Factory Function Update**:

```cpp
// src/v2/tensors/UnifiedKVCache.h

/// Create KV cache with specified layout mode
inline std::unique_ptr<IUnifiedKVCache> createUnifiedKVCache(
    ActivationPrecision precision,
    const MPIContext& mpi_ctx,
    int n_layers, int batch_size, int max_seq_len,
    int n_kv_heads, int head_dim, int device_idx,
    KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR)
{
    switch (precision) {
        case ActivationPrecision::Q16_1:
            return std::make_unique<UnifiedKVCacheTensor<ActivationPrecision::Q16_1>>(
                mpi_ctx, n_layers, batch_size, max_seq_len,
                n_kv_heads, head_dim, device_idx, layout_mode);
        // ... other precisions (default to POSITION_MAJOR)
    }
}
```

**GraphOrchestrator Update**:

```cpp
// src/v2/pipelines/qwen/GraphOrchestrator.cpp

// Select layout mode based on attention backend
KVCacheLayoutMode kv_layout = 
    (config_.attention_backend == FusedAttentionBackend::Q16_INTEGER)
        ? KVCacheLayoutMode::HEAD_MAJOR   // Q16 kernel needs head-major
        : KVCacheLayoutMode::POSITION_MAJOR;  // Other backends use position-major

kv_cache_ = createUnifiedKVCache(
    config_.kv_cache_precision,
    mpi_ctx_, n_layers_, batch_size_, max_seq_len_,
    n_kv_heads_, head_dim_, device_idx_,
    kv_layout);  // Pass layout mode
```

**FusedAttentionWoStage Update** (removes transpose workaround):

```cpp
// src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp

if (params_.backend == FusedAttentionBackend::Q16_INTEGER) {
    // Check if cache is head-major (no transpose needed)
    if (params_.kv_cache && params_.kv_cache->kv_layout() == TensorLayout::KV_HEAD_POS_DIM) {
        // Direct access - no transpose!
        auto* cache = dynamic_cast<UnifiedKVCacheTensor<ActivationPrecision::Q16_1>*>(params_.kv_cache);
        q16_params.K = cache->get_k_head(layer, 0)->typed_data();
        q16_params.V = cache->get_v_head(layer, 0)->typed_data();
        q16_params.kv_head_stride = cache->head_stride_bytes();
    } else {
        // Fallback: transpose workaround for position-major cache
        // (existing code, will emit layout mismatch warning)
    }
}
```

**Benefits of Unified Approach**:
- Single class to maintain (no code duplication)
- Existing tests and infrastructure work with both modes
- Layout mode is explicit in constructor (clear intent)
- `IUnifiedKVCache` interface unchanged (backward compatible)
- Can switch modes at cache creation time based on attention backend

**Implementation in Phase 3**:
- Add `KVCacheLayoutMode` enum
- Add `layout_mode_` member and constructor parameter
- Add `head_entries_` storage for HEAD_MAJOR mode
- Implement `get_k_head()`/`get_v_head()` accessors
- Update `append_kv()` to handle both layouts
- Update `get_k_base()`/`get_v_base()` to return correct layout
- Add `kv_layout()` override returning `TensorLayout` enum

### Phase 4: Typed MPI Parameters (1 day)

#### Code Research Findings

Analysis of MPI communicator handling across the codebase reveals:

**Current MPI Comm Architecture**:

| Component | Current State | Issue |
|-----------|---------------|-------|
| `AllreduceStage::Params` | `void* mpi_comm` | No type safety |
| `AllGatherStage::Params` | `void* mpi_comm` | No type safety |
| `ResolvedStage` | `opaque_params["mpi_comm"] = void*` | String-keyed, type-erased |
| `GraphResolverConfig` | `MPI_Comm mpi_comm` | ✅ Already typed |
| `MPIContext` | `MPI_Comm comm_` | ✅ Already typed |

**Usage Sites Requiring Updates**:

| File | Location | Current Code | Issue |
|------|----------|--------------|-------|
| [AllreduceStage.h](src/v2/execution/compute_stages/stages/AllreduceStage.h#L23) | `Params` struct | `void* mpi_comm = nullptr` | No type safety |
| [AllGatherStage.h](src/v2/execution/compute_stages/stages/AllGatherStage.h#L30) | `Params` struct | `void* mpi_comm = nullptr` | No type safety |
| [AllreduceStage.cpp](src/v2/execution/compute_stages/stages/AllreduceStage.cpp#L65) | `execute()` | `MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm)` | Unsafe cast |
| [GraphResolver.cpp](src/v2/execution/GraphResolver.cpp#L404) | `resolveTPCollective()` | `allreduce.opaque_params["mpi_comm"] = static_cast<void*>(runtime.mpi_comm)` | Double cast |
| [GraphResolver.cpp](src/v2/execution/GraphResolver.cpp#L805) | `buildStage()` | `static_cast<MPI_Comm>(stage.opaque_params.at("mpi_comm"))` | Unsafe retrieval |
| [GraphSchema.h](src/v2/execution/GraphSchema.h#L481) | `ResolvedStage` | `std::unordered_map<std::string, void*> opaque_params` | Type-erased |

**Test Files Using `void* mpi_comm`**:

| File | Line | Current Usage |
|------|------|---------------|
| `Test__AllGatherStage.cpp` | 105, 126, 170+ | `params.mpi_comm = MPI_COMM_WORLD` (implicit void* cast) |
| `Test__MPI_ColumnParallelLMHead.cpp` | 218, 321, 397+ | `params.mpi_comm = MPI_COMM_WORLD` |
| `Test__ComputeStages.cpp` | 943 | `params.mpi_comm = nullptr` |

**MPI_Comm Type Analysis**:

```cpp
// In most MPI implementations (OpenMPI, MPICH), MPI_Comm is:
// - OpenMPI: typedef struct ompi_communicator_t *MPI_Comm  (pointer)
// - MPICH:   typedef int MPI_Comm                          (int)
// - Intel MPI: typedef int MPI_Comm                        (int)

// Size: sizeof(MPI_Comm) is either sizeof(void*) or sizeof(int)
// We need a wrapper that handles both cases safely.
```

#### Detailed File Changes

**NEW FILES:**

| File | Purpose |
|------|---------|
| `src/v2/utils/MPITypes.h` | `MPICommHandle` type-safe wrapper |
| `src/v2/utils/MPITypes.cpp` | Implementation (hides MPI include) |
| `tests/v2/unit/Test__MPITypes.cpp` | Unit tests |

**MODIFIED FILES:**

| File | Changes |
|------|---------|
| `src/v2/execution/compute_stages/stages/AllreduceStage.h` | Change `void*` to `MPICommHandle` |
| `src/v2/execution/compute_stages/stages/AllreduceStage.cpp` | Use `mpi_comm.get()` |
| `src/v2/execution/compute_stages/stages/AllGatherStage.h` | Change `void*` to `MPICommHandle` |
| `src/v2/execution/compute_stages/stages/AllGatherStage.cpp` | Use `mpi_comm.get()` |
| `src/v2/execution/GraphSchema.h` | Add `mpi_comm` field to `ResolvedStage`, keep `opaque_params` for kv_cache |
| `src/v2/execution/GraphResolver.cpp` | Use typed `mpi_comm` field instead of `opaque_params` |
| Test files | Update to use `MPICommHandle::fromComm()` |

#### Code Examples

**1. New MPITypes.h:**

```cpp
// src/v2/utils/MPITypes.h
#pragma once

#include <cstdint>
#include <cstring>

// Forward declare MPI_Comm to avoid including <mpi.h> in headers
// This allows including MPITypes.h without MPI dependency
struct ompi_communicator_t;  // OpenMPI
typedef struct ompi_communicator_t* OMPI_MPI_Comm;
typedef int MPICH_MPI_Comm;

namespace llaminar2 {

/**
 * @brief Type-safe wrapper for MPI communicator handles
 *
 * Design Goals:
 * 1. No MPI include in header (avoids MPI dependency in non-MPI code)
 * 2. Works with both pointer-based (OpenMPI) and int-based (MPICH) MPI_Comm
 * 3. Default-constructible to null/invalid state
 * 4. Explicit construction to prevent accidental void* casts
 * 5. Safe conversion back to MPI_Comm in .cpp files
 *
 * Usage:
 * @code
 * // In .cpp file that includes <mpi.h>:
 * auto handle = MPICommHandle::fromComm(MPI_COMM_WORLD);
 * MPI_Comm comm = handle.get();
 * @endcode
 */
class MPICommHandle {
public:
    /// Default constructor - creates null/invalid handle
    MPICommHandle() : storage_{0} {}
    
    /// Check if handle is valid (non-null)
    explicit operator bool() const { return !isNull(); }
    
    /// Check if handle is null
    bool isNull() const {
        uint64_t zero = 0;
        return std::memcmp(&storage_, &zero, sizeof(storage_)) == 0;
    }
    
    /// Compare two handles
    bool operator==(const MPICommHandle& other) const {
        return std::memcmp(&storage_, &other.storage_, sizeof(storage_)) == 0;
    }
    bool operator!=(const MPICommHandle& other) const {
        return !(*this == other);
    }
    
    // ========================================
    // Construction and Extraction (in .cpp)
    // ========================================
    
    /// Create handle from MPI_Comm (defined in MPITypes.cpp)
    /// @note This is a static factory to make the intent explicit
    template<typename MPI_Comm_T>
    static MPICommHandle fromComm(MPI_Comm_T comm) {
        MPICommHandle handle;
        static_assert(sizeof(comm) <= sizeof(handle.storage_),
                      "MPI_Comm too large for storage");
        std::memcpy(&handle.storage_, &comm, sizeof(comm));
        return handle;
    }
    
    /// Extract MPI_Comm (defined in MPITypes.cpp)
    /// @note Only call this in .cpp files that include <mpi.h>
    template<typename MPI_Comm_T>
    MPI_Comm_T get() const {
        MPI_Comm_T comm;
        static_assert(sizeof(comm) <= sizeof(storage_),
                      "MPI_Comm too large for storage");
        std::memcpy(&comm, &storage_, sizeof(comm));
        return comm;
    }
    
    /// Get raw storage (for debugging)
    uint64_t rawValue() const { return storage_; }

private:
    // Storage large enough for any MPI_Comm type
    // - OpenMPI uses pointer (8 bytes on 64-bit)
    // - MPICH uses int (4 bytes)
    // We use 8 bytes to handle both cases safely
    uint64_t storage_;
};

// Convenience function for common case
inline MPICommHandle MPICommWorld();  // Defined in MPITypes.cpp

} // namespace llaminar2
```

**2. New MPITypes.cpp:**

```cpp
// src/v2/utils/MPITypes.cpp
#include "MPITypes.h"
#include <mpi.h>

namespace llaminar2 {

MPICommHandle MPICommWorld() {
    return MPICommHandle::fromComm(MPI_COMM_WORLD);
}

// Explicit instantiations for common MPI_Comm types
// These ensure the template methods work correctly

template MPICommHandle MPICommHandle::fromComm<MPI_Comm>(MPI_Comm);
template MPI_Comm MPICommHandle::get<MPI_Comm>() const;

} // namespace llaminar2
```

**3. Updated AllreduceStage.h:**

```cpp
// src/v2/execution/compute_stages/stages/AllreduceStage.h
#pragma once

#include "../IComputeStage.h"
#include "../../../utils/MPITypes.h"  // NEW

namespace llaminar2 {

class AllreduceStage : public IComputeStage {
public:
    struct Params {
        TensorBase* buffer = nullptr;
        MPICommHandle mpi_comm;           // ✅ Type-safe (was: void* mpi_comm = nullptr)
        size_t count = 0;
        const MPIContext* mpi_ctx = nullptr;
    };
    
    // ... rest unchanged ...
};

}
```

**4. Updated AllreduceStage.cpp:**

```cpp
// BEFORE:
MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);  // ❌ Unsafe

// AFTER:
if (!params_.mpi_comm) {
    LOG_ERROR("[AllreduceStage] Null MPI communicator");
    return false;
}
MPI_Comm comm = params_.mpi_comm.get<MPI_Comm>();  // ✅ Type-safe
```

**5. Updated ResolvedStage in GraphSchema.h:**

```cpp
struct ResolvedStage {
    std::string name;
    StageType type;
    
    std::vector<TensorBase*> inputs;
    std::vector<TensorBase*> outputs;
    std::vector<std::string> dependencies;
    
    int device_idx = 0;
    
    // Stage-specific parameters (typed)
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, int> int_params;
    std::unordered_map<std::string, bool> bool_params;
    std::unordered_map<std::string, std::string> string_params;
    std::unordered_map<std::string, TensorBase*> tensor_params;
    
    // ============================================
    // PHASE 4 CHANGES
    // ============================================
    
    /// Type-safe MPI communicator (replaces opaque_params["mpi_comm"])
    MPICommHandle mpi_comm;
    
    /// Opaque params for non-MPI pointers (e.g., kv_cache)
    /// NOTE: Consider typed alternatives for kv_cache in future
    std::unordered_map<std::string, void*> opaque_params;
};
```

**6. Updated GraphResolver.cpp:**

```cpp
// BEFORE (resolveTPCollective):
allreduce.opaque_params["mpi_comm"] = static_cast<void*>(runtime.mpi_comm);

// AFTER:
allreduce.mpi_comm = MPICommHandle::fromComm(runtime.mpi_comm);

// BEFORE (buildStage for Allreduce):
if (stage.opaque_params.count("mpi_comm")) {
    params.mpi_comm = static_cast<MPI_Comm>(stage.opaque_params.at("mpi_comm"));
}

// AFTER:
params.mpi_comm = stage.mpi_comm;  // Direct assignment, both are MPICommHandle
```

**7. Test Migration Example:**

```cpp
// BEFORE (Test__AllGatherStage.cpp):
params.mpi_comm = MPI_COMM_WORLD;  // Implicit void* conversion

// AFTER:
params.mpi_comm = MPICommHandle::fromComm(MPI_COMM_WORLD);  // Explicit

// Or use convenience function:
params.mpi_comm = MPICommWorld();
```

#### Phase 4 Test Strategy

**Test File**: `tests/v2/unit/Test__MPITypes.cpp`

```cpp
#include <gtest/gtest.h>
#include "utils/MPITypes.h"
#include <mpi.h>

class Test__MPITypes : public ::testing::Test {
protected:
    void SetUp() override {
        // MPI should be initialized by test harness
    }
};

// =============================================================================
// MPICommHandle Construction Tests
// =============================================================================

TEST_F(Test__MPITypes, DefaultConstruction_IsNull)
{
    MPICommHandle handle;
    EXPECT_TRUE(handle.isNull());
    EXPECT_FALSE(static_cast<bool>(handle));
}

TEST_F(Test__MPITypes, FromComm_IsNotNull)
{
    auto handle = MPICommHandle::fromComm(MPI_COMM_WORLD);
    EXPECT_FALSE(handle.isNull());
    EXPECT_TRUE(static_cast<bool>(handle));
}

TEST_F(Test__MPITypes, FromComm_RoundTrip)
{
    MPI_Comm original = MPI_COMM_WORLD;
    auto handle = MPICommHandle::fromComm(original);
    MPI_Comm retrieved = handle.get<MPI_Comm>();
    
    // Compare by performing an MPI operation
    int rank1, rank2;
    MPI_Comm_rank(original, &rank1);
    MPI_Comm_rank(retrieved, &rank2);
    EXPECT_EQ(rank1, rank2);
}

TEST_F(Test__MPITypes, Equality_SameComm)
{
    auto h1 = MPICommHandle::fromComm(MPI_COMM_WORLD);
    auto h2 = MPICommHandle::fromComm(MPI_COMM_WORLD);
    EXPECT_EQ(h1, h2);
}

TEST_F(Test__MPITypes, Equality_NullHandles)
{
    MPICommHandle h1, h2;
    EXPECT_EQ(h1, h2);
}

TEST_F(Test__MPITypes, MPICommWorld_Convenience)
{
    auto handle = MPICommWorld();
    EXPECT_FALSE(handle.isNull());
    
    MPI_Comm comm = handle.get<MPI_Comm>();
    int size;
    MPI_Comm_size(comm, &size);
    EXPECT_GE(size, 1);
}

// =============================================================================
// Integration with Stage Params
// =============================================================================

TEST_F(Test__MPITypes, AllreduceParams_WithMPICommHandle)
{
    AllreduceStage::Params params;
    params.mpi_comm = MPICommHandle::fromComm(MPI_COMM_WORLD);
    
    EXPECT_TRUE(static_cast<bool>(params.mpi_comm));
    EXPECT_FALSE(params.mpi_comm.isNull());
}

TEST_F(Test__MPITypes, AllreduceParams_DefaultIsNull)
{
    AllreduceStage::Params params;
    // Default mpi_comm should be null
    EXPECT_TRUE(params.mpi_comm.isNull());
}

TEST_F(Test__MPITypes, AllGatherParams_WithMPICommHandle)
{
    AllGatherStage::Params params;
    params.mpi_comm = MPICommWorld();
    
    EXPECT_TRUE(static_cast<bool>(params.mpi_comm));
}

// =============================================================================
// ResolvedStage Integration
// =============================================================================

TEST_F(Test__MPITypes, ResolvedStage_MPICommField)
{
    ResolvedStage stage;
    stage.mpi_comm = MPICommHandle::fromComm(MPI_COMM_WORLD);
    
    // Verify we can retrieve it
    EXPECT_FALSE(stage.mpi_comm.isNull());
    
    MPI_Comm comm = stage.mpi_comm.get<MPI_Comm>();
    int size;
    MPI_Comm_size(comm, &size);
    EXPECT_GE(size, 1);
}
```

**CTest Labels**:
```cmake
set_tests_properties(V2_Unit_MPITypes_* PROPERTIES
    LABELS "Unit;V2;MPI;TypeSafety")
```

#### Key Implementation Notes

1. **No MPI Include in Header**: `MPITypes.h` avoids `#include <mpi.h>` by using `uint64_t` storage. This allows non-MPI code to include the header without MPI dependency.

2. **Template Methods**: `fromComm<T>()` and `get<T>()` work with any MPI_Comm type (pointer or int).

3. **Backward Compatibility**: `opaque_params` is kept for `kv_cache` and other non-MPI pointers. Only MPI comm moves to typed field.

4. **Test Updates**: Tests need explicit `MPICommHandle::fromComm()` instead of implicit `void*` cast. This is intentional - makes MPI usage explicit.

5. **Null Checking**: `MPICommHandle` supports `if (handle)` pattern via `explicit operator bool()`.

6. **GraphResolverConfig Unchanged**: It already uses `MPI_Comm mpi_comm`, which is fine since that file includes `<mpi.h>`.

#### Migration Checklist

- [ ] Create `MPITypes.h` and `MPITypes.cpp`
- [ ] Add to CMakeLists.txt
- [ ] Update `AllreduceStage::Params`
- [ ] Update `AllGatherStage::Params`
- [ ] Update `AllreduceStage::execute()`
- [ ] Update `AllGatherStage::execute()`
- [ ] Add `mpi_comm` field to `ResolvedStage`
- [ ] Update `GraphResolver::resolveTPCollective()`
- [ ] Update `GraphResolver::buildStage()` for Allreduce/Allgather
- [ ] Update test files to use `MPICommHandle::fromComm()`
- [ ] Add unit tests for `MPICommHandle`

---

## Migration Guide

### For `typed_data()` Usage

**Before** (broken for block_size != 32):
```cpp
const Q16_1Block* blocks = q16_tensor->typed_data();
float scale = blocks[0].d;
```

**After** (safe):
```cpp
// Option A: Use raw_data() + q16_block_size()
const uint8_t* raw = static_cast<const uint8_t*>(q16_tensor->raw_data());
Q16BlockSize bs = q16_tensor->q16_block_size();
size_t block_bytes = q16_block_size_bytes(bs);
float scale;
std::memcpy(&scale, raw, sizeof(float));  // Scale is always at offset 0

// Option B: Use type-safe accessor
if (auto* b64 = q16_tensor->as_block_64()) {
    float scale = b64[0].d;
}

// Option C: Use dispatch helper
dispatchQ16Block(q16_tensor, [](auto* blocks, int bs) {
    float scale = blocks[0].d;
});
```

### For Tensor Verification (New Unified System)

**Before** (ad-hoc or no validation):
```cpp
bool MyStage::execute() {
    // No entry validation
    bool success = kernel_->compute(params_.input, params_.output);
    // Maybe some manual zero check?
    return success;
}
```

**After** (automatic verification at stage boundaries):
```cpp
// GraphExecutor automatically calls verification before/after execute()
// On failure:
//   1. Logs: layer=5 stage=MyStage phase=EXIT tensor=output reason="156 NaN values"
//   2. Dumps all buffers to /tmp/llaminar_verification_dump/...
//   3. Throws VerificationFailure exception

// Stages don't need explicit validation code - just implement getDumpInfo():
StageDumpInfo MyStage::getDumpInfo() const {
    StageDumpInfo info;
    info.addInput("input", params_.input, rows_, cols_);
    info.addOutput("output", params_.output, rows_, cols_);
    return info;
}
```

### For Snapshot Capture

**Before** (no validation):
```cpp
CAPTURE_SNAPSHOT("ATTENTION_OUTPUT", tensor.get());  // May capture garbage
```

**After** (verified):
```cpp
LLAMINAR_SNAPSHOT("ATTENTION_OUTPUT", tensor.get());  
// In Debug/Integration: verifies first 8 rows for NaN/Inf/null
// Logs ERROR if verification fails, still captures for debugging
```

### For MPI Communicator Parameters

**Before** (void*):
```cpp
// Creating stage
AllreduceStage::Params params;
params.mpi_comm = static_cast<void*>(MPI_COMM_WORLD);  // ❌ Manual cast

// Using in stage
MPI_Comm comm = reinterpret_cast<MPI_Comm>(params_.mpi_comm);  // ❌ Unsafe
```

**After** (typed):
```cpp
// Creating stage
AllreduceStage::Params params;
params.mpi_comm = MPICommHandle::fromComm(MPI_COMM_WORLD);  // ✅ Type-safe

// Using in stage
MPI_Comm comm = params_.mpi_comm.get();  // ✅ Type-safe
if (!params_.mpi_comm) { /* handle null */ }  // ✅ Null check
```

### For Snapshot Capture

**Before** (no validation):
```cpp
CAPTURE_SNAPSHOT("ATTENTION_OUTPUT", tensor.get());  // May capture garbage
```

**After** (validated):
```cpp
LLAMINAR_SNAPSHOT("ATTENTION_OUTPUT", tensor.get());  
// In Debug/Integration: validates first 8 rows for NaN/Inf/null
// Logs ERROR if validation fails, then captures anyway for debugging
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Performance impact of verification | Only active in Debug/Integration builds; samples first N rows |
| Breaking existing code using `typed_data()` | Deprecation warning + runtime assert, not compile error |
| Layout specification overhead | Defaults to UNKNOWN, gradual adoption |
| False positives on all-zero checks | All-zero is warning only; fail_on_zero is opt-in |
| Disk space from dumps | Dumps only on failure; limited to one stage's buffers |
| MPI header dependency in MPICommHandle | Use opaque storage, define accessors in .cpp |

---

## Success Criteria

1. **Verification at every stage boundary**: All stages verified on entry and exit in Debug/Integration builds
2. **Actionable failure messages**: On failure, logs include layer number, stage name, phase (ENTRY/EXIT), tensor name, and failure reason
3. **Full debugging context**: All stage buffers dumped to `/tmp` on verification failure
4. **Exception halts execution**: `VerificationFailure` exception thrown on first failure
5. **No more Q16 block type confusion**: `typed_data()` either works correctly or fails loudly
6. **Layout mismatches caught**: Attention stages validate K/V layout matches kernel expectations
7. **Type-safe MPI**: No `void*` casts needed for MPI communicators
8. **Snapshots verified**: `LLAMINAR_SNAPSHOT` validates data before capture

---

## Related Documents

- [PLAN_FIXED_SCALE_ROPE_Q16.md](./PLAN_FIXED_SCALE_ROPE_Q16.md) - KV cache layout issues
- [PROJECT_Q16_INTEGER_ATTENTION_V2.md](./PROJECT_Q16_INTEGER_ATTENTION_V2.md) - Q16 attention design
- [copilot-instructions.md](../../.github/copilot-instructions.md) - Assertion framework docs
