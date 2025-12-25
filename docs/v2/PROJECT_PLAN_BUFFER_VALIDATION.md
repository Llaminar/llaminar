# Project Plan: Buffer Contract Validation System

**Created:** December 25, 2025  
**Status:** In Progress  
**Branch:** feature/typed-residuals

## Problem Statement

During Hybrid mode debugging, we discovered that `V_dequant` was **allocated but never populated**, causing attention output to be all zeros. This bug went undetected because:

1. **Silent zero propagation** - Attention accepted zeros and produced zeros silently
2. **No producer/consumer contract** - Buffer allocation was decoupled from population
3. **Scattered Hybrid logic** - Mode-specific code spread across 3+ files
4. **No graph-level validation** - `StageBufferRequirements` exists but isn't enforced

## Goals

1. **Fail-fast detection** of uninitialized/zero buffers in debug builds
2. **Explicit producer declarations** for critical buffers
3. **Graph-level validation** at build time
4. **Centralized mode context** to reduce scattered conditionals

---

## Phase 1: Zero/NaN Detection (Quick Wins)

### 1.1 Tensor Validation Utilities

**File:** `src/v2/tensors/TensorValidation.h` (new)

Add debug-only validation functions:

```cpp
namespace llaminar2 {

#ifndef NDEBUG
/// Check if tensor appears to be uninitialized (all zeros)
bool tensorAppearsZero(const TensorBase* t, size_t sample_count = 1000);

/// Check if tensor contains NaN or Inf
bool tensorHasNaNOrInf(const TensorBase* t);

/// Validate tensor and log warning if problematic
void validateTensorNotZero(const TensorBase* t, const std::string& name,
                           const std::string& stage_name);
#endif

}
```

### 1.2 Attention Kernel Guards

**File:** `src/v2/kernels/cpu/attention/CPUAttentionKernelT.h`

Add assertion before attention computation:

```cpp
#ifndef NDEBUG
validateTensorNotZero(V_ptr, "V", "AttentionCompute");
#endif
```

### Unit Tests

| Test | File | Description |
|------|------|-------------|
| `TensorValidation_DetectsZeroTensor` | `Test__TensorValidation.cpp` | Verify zero detection |
| `TensorValidation_DetectsNaN` | `Test__TensorValidation.cpp` | Verify NaN detection |
| `TensorValidation_PassesValidTensor` | `Test__TensorValidation.cpp` | No false positives |

---

## Phase 2: Buffer Producer Contracts

### 2.1 Enhanced BufferDescriptor

**File:** `src/v2/execution/BufferRole.h`

Extend `BufferDescriptor` with producer declaration:

```cpp
struct BufferDescriptor {
    // ... existing fields ...
    
    /// Stage name that produces this buffer (empty = external/pre-allocated)
    std::string producer_stage;
    
    /// Whether this buffer must be non-zero after producer executes
    bool validate_populated = false;
    
    // Builder extension
    BufferDescriptor& withProducer(const std::string& stage) {
        producer_stage = stage;
        return *this;
    }
    
    BufferDescriptor& validatePopulated() {
        validate_populated = true;
        return *this;
    }
};
```

### 2.2 Graph Validation Pass

**File:** `src/v2/execution/GraphValidator.h` (new)

```cpp
class GraphValidator {
public:
    struct ValidationResult {
        bool valid = true;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    /// Validate buffer producer/consumer relationships
    ValidationResult validateBufferFlow(const ComputeGraph& graph);
    
    /// Check for unconnected outputs (allocated but never consumed)
    ValidationResult validateNoOrphanBuffers(const ComputeGraph& graph,
                                             const Qwen2ActivationBuffers& buffers);
};
```

### Unit Tests

| Test | File | Description |
|------|------|-------------|
| `GraphValidator_DetectsMissingProducer` | `Test__GraphValidator.cpp` | Buffer with no producer stage |
| `GraphValidator_DetectsOrphanBuffer` | `Test__GraphValidator.cpp` | Allocated but unused buffer |
| `GraphValidator_ValidGraphPasses` | `Test__GraphValidator.cpp` | Correct graph validates |
| `GraphValidator_HybridModeBufferFlow` | `Test__GraphValidator.cpp` | V_dequant has producer |

---

## Phase 3: Stage Output Registration

### 3.1 Stage Output Manifest

**File:** `src/v2/execution/ComputeStage.h`

Add output registration to stages:

```cpp
class IComputeStage {
public:
    // ... existing interface ...
    
    /// Declare outputs this stage will produce
    virtual std::vector<BufferDescriptor> getDeclaredOutputs() const {
        return {}; // Default: derive from Params
    }
    
    /// Called after execute to register produced buffers
    virtual void registerProducedBuffers(GraphBufferManager& manager) const {}
};
```

### 3.2 KVCacheAppendStage Output Declaration

```cpp
std::vector<BufferDescriptor> KVCacheAppendStage::getDeclaredOutputs() const {
    std::vector<BufferDescriptor> outputs;
    // Always produces K/V cache entries
    outputs.push_back(BufferDescriptor::output("K_cached", {}, BufferTensorType::FP32));
    outputs.push_back(BufferDescriptor::output("V_cached", {}, BufferTensorType::FP32));
    
    // Conditionally produces V_dequant
    if (params_.V_dequant_out) {
        outputs.push_back(BufferDescriptor::output("V_dequant", {}, BufferTensorType::FP32)
                          .validatePopulated());
    }
    return outputs;
}
```

### Unit Tests

| Test | File | Description |
|------|------|-------------|
| `KVCacheAppend_DeclaresVDequantOutput` | `Test__KVCacheAppendStage.cpp` | V_dequant in outputs when configured |
| `StageOutputs_MatchActualProduction` | `Test__StageOutputManifest.cpp` | Declared = actual |

---

## Phase 4: Centralized Mode Context

### 4.1 InferenceMode Struct

**File:** `src/v2/execution/InferenceMode.h` (new)

```cpp
struct InferenceMode {
    enum class Precision { FP32, Q8_1, Hybrid };
    Precision precision = Precision::Q8_1;
    
    // Derived properties
    bool needsKRope() const { return precision == Precision::Hybrid; }
    bool needsVDequant() const { return precision == Precision::Hybrid; }
    bool usesFusedAttention() const { return precision == Precision::Q8_1; }
    bool usesDecomposedAttention() const { return precision != Precision::Q8_1; }
    
    // Buffer requirements
    std::vector<std::string> requiredBuffers() const {
        std::vector<std::string> buffers = {"Q", "K", "V", "attn_output"};
        if (needsKRope()) buffers.push_back("K_rope");
        if (needsVDequant()) buffers.push_back("V_dequant");
        return buffers;
    }
    
    // Validate buffer availability
    bool validateBuffers(const Qwen2ActivationBuffers& buffers) const;
};
```

### 4.2 Refactor Qwen2Graph to Use InferenceMode

Replace scattered `use_hybrid_rope` checks with:

```cpp
InferenceMode mode{precision};
if (mode.needsVDequant()) {
    kv_append_params.V_dequant_out = buffers.V_dequant;
}
```

### Unit Tests

| Test | File | Description |
|------|------|-------------|
| `InferenceMode_HybridRequiresVDequant` | `Test__InferenceMode.cpp` | Property checks |
| `InferenceMode_Q8_1NoExtraBuffers` | `Test__InferenceMode.cpp` | Minimal buffers |
| `InferenceMode_ValidatesMissingBuffer` | `Test__InferenceMode.cpp` | Detects missing |

---

## Phase 5: Debug Validation Flag

### 5.1 DebugEnv Extension

**File:** `src/v2/utils/DebugEnv.h`

Add new flag:

```cpp
struct ValidationEnv {
    bool validate_buffers = false;      // LLAMINAR_VALIDATE_BUFFERS
    bool fail_on_zero_tensor = false;   // LLAMINAR_FAIL_ON_ZERO
};

struct DebugEnv {
    // ... existing ...
    ValidationEnv validation;
};
```

### 5.2 GraphExecutor Integration

**File:** `src/v2/execution/GraphExecutor.cpp`

```cpp
void GraphExecutor::executeNode(const ComputeNode& node) {
    bool success = node.stage->execute(device_ctx);
    
    #ifndef NDEBUG
    if (debugEnv().validation.validate_buffers) {
        validateStageOutputs(node);
    }
    #endif
}
```

---

## Implementation Order

| Priority | Phase | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| P0 | 1.1 TensorValidation utilities | 1 hr | Immediate detection | ✅ **DONE** |
| P0 | 1.2 Attention kernel guards | 30 min | Would have caught V=0 | ✅ **DONE** |
| P1 | 2.1 BufferDescriptor.producer_stage | 1 hr | Explicit contracts | ✅ **DONE** |
| P1 | 2.2 GraphValidator | 2 hr | Build-time validation | ✅ **DONE** |
| P2 | 3.1-3.2 Stage output manifest | 2 hr | Complete tracking | ✅ **DONE** |
| P2 | 4.1-4.2 InferenceMode | 3 hr | Cleaner code | ✅ **DONE** |
| P3 | 5.1-5.2 Debug flag | 1 hr | Runtime toggle | ✅ **DONE** |

**Total Estimated Effort:** 10-12 hours

---

## Completed Work

### Phase 1.1: TensorValidation Utilities ✅

**Files Created:**
- `src/v2/tensors/TensorValidation.h` - Debug-only validation functions
- `tests/v2/unit/Test__TensorValidation.cpp` - 20 unit tests

**Functions:**
- `tensorAppearsZero()` - Detects uninitialized (all-zero) FP32 tensors
- `tensorHasNaNOrInf()` - Detects NaN/Inf corruption
- `validateTensorNotZero()` - Logs warnings for problematic tensors
- `assertTensorValid()` - Throws for critical invariant violations

**Test Results:** 20/20 tests pass

### Phase 1.2: Attention Kernel Guards ✅

**Files Modified:**
- `src/v2/kernels/cpu/attention/CPUAttentionKernelT.h`

**Change:** Added FP32-only zero-tensor detection in `compute_typed()`:
```cpp
#ifndef NDEBUG
if constexpr (std::is_same_v<ElementType, float>) {
    // Sample first/last elements for quick zero check
    if (v_zero)
        LOG_WARN("[CPUAttentionKernelT] V tensor appears to be all zeros (likely uninitialized)!");
}
#endif
```

This would have immediately flagged the V_dequant=0 bug.

### Phase 2.1: BufferDescriptor.producer_stage ✅

**Files Modified:**
- `src/v2/execution/BufferRole.h`

**Changes:**
- Added `producer_stage` field to `BufferDescriptor` for explicit producer declaration
- Added `validate_populated` field for runtime validation flag
- Added fluent builder methods: `withProducer()`, `validatePopulated()`, `hasProducer()`

**Example Usage:**
```cpp
auto v_dequant_desc = BufferDescriptor::output("V_dequant", {9, 128}, BufferTensorType::FP32);
v_dequant_desc.withProducer("kv_append").validatePopulated();
```

### Phase 2.2: GraphValidator ✅

**Files Created:**
- `src/v2/execution/GraphValidator.h` - Graph-level buffer flow validation
- `tests/v2/unit/Test__GraphValidator.cpp` - 17 unit tests

**Classes/Structs:**
- `GraphValidationResult` - Holds errors/warnings with merge/log helpers
- `BufferFlowEntry` - Tracks producer/consumer relationships
- `GraphValidator` - Main validation class

**Key Methods:**
- `validateBufferFlow()` - Validates producer/consumer ordering and relationships
- `validateNoOrphanAllocations()` - Detects allocated-but-unpopulated buffers
- `validateBufferCompatibility()` - Checks type/shape matching between producer/consumer
- `validateAll()` - Runs all validations

**Test Results:** 17/17 tests pass

**Test Cases:**
- Empty and single-stage graphs validate correctly
- Producer/consumer pairs validate correctly
- Missing producer detection (declared but nonexistent stage)
- Orphan buffer detection (allocated but never consumed)
- Type mismatch detection (FP32 producer vs Q8_1 consumer)
- Multi-stage chain validation
- HybridMode V_dequant pattern validation

### Phase 3: Stage Output Manifest ✅

**Files Modified:**
- `src/v2/execution/ComputeStage.h` - Added `getDeclaredOutputs()` interface
- `src/v2/execution/ComputeStage.cpp` - Implemented for `KVCacheAppendStage`

**Files Created:**
- `tests/v2/unit/Test__StageOutputManifest.cpp` - 7 unit tests

**New Interface:**
```cpp
class IComputeStage {
    // ...existing interface...
    
    /// Get declared outputs this stage will produce (with producer contracts)
    virtual std::vector<BufferDescriptor> getDeclaredOutputs() const {
        return {}; // Default: no declared outputs
    }
};
```

**KVCacheAppendStage Implementation:**
- `getDeclaredOutputs()` returns V_dequant with producer contract when in Hybrid mode
- `producesVDequant()` helper for checking mode
- `getBufferRequirements()` also includes V_dequant as OUTPUT

**Key Distinction:**
- `getBufferRequirements()`: "What buffers do I need to run?" (all inputs + outputs)
- `getDeclaredOutputs()`: "What buffers do I promise to populate?" (outputs with contracts)

**Test Results:** 7/7 tests pass

**Test Cases:**
- Standard mode: no declared outputs
- Hybrid mode: V_dequant declared with producer contract
- Shape propagation from V_dequant_out tensor
- Buffer requirements includes V_dequant as OUTPUT
- Declared outputs are subset of buffer requirements
- Default stage behavior (empty outputs)

### Phase 4: InferenceMode Context ✅

**Files Created:**
- `src/v2/execution/InferenceMode.h` - Centralized mode context class
- `src/v2/execution/InferenceMode.cpp` - Implementation
- `tests/v2/unit/Test__InferenceMode.cpp` - 21 unit tests

**Files Modified:**
- `src/v2/CMakeLists.txt` - Added InferenceMode.cpp to build

**InferenceMode Class:**
```cpp
class InferenceMode {
public:
    // Constructors
    explicit InferenceMode(ActivationPrecision precision);
    static InferenceMode FP32();
    static InferenceMode Q8_1();
    static InferenceMode Hybrid();
    
    // Mode identification
    bool isFP32() const;
    bool isQ8_1() const;
    bool isHybrid() const;
    const char* name() const;
    
    // Hybrid-specific buffer requirements
    bool needsQRope() const;    // True for Hybrid
    bool needsKRope() const;    // True for Hybrid
    bool needsVDequant() const; // True for Hybrid
    
    // Attention strategy
    bool usesFusedAttention() const;      // Q8_1 only
    bool usesDecomposedAttention() const; // FP32 and Hybrid
    
    // Extra buffers needed beyond base Q/K/V/attn_output
    std::vector<std::string> extraRequiredBuffers() const;
    
    // Buffer validation
    struct ValidationResult {
        bool valid;
        std::vector<std::string> missing_buffers;
        operator bool() const { return valid; }
    };
    ValidationResult validateBuffers(bool has_Q, bool has_K, bool has_V, 
                                     bool has_attn_output, bool has_Q_rope,
                                     bool has_K_rope, bool has_V_dequant) const;
};
```

**Helper Function (replaces scattered pattern):**
```cpp
// Before:
bool use_hybrid_rope = (config_.activation_precision == ActivationPrecision::Hybrid) &&
                       buffers.Q_rope && buffers.K_rope;

// After:
bool use_hybrid_mode = isHybridModeActive(mode, buffers);
```

**Test Results:** 21/21 tests pass

**Test Cases:**
- Mode identification (FP32, Q8_1, Hybrid)
- Factory methods
- Buffer requirements (FP32/Q8_1 = none, Hybrid = Q_rope/K_rope/V_dequant)
- Attention strategy (FP32/Hybrid = decomposed, Q8_1 = fused)
- Buffer validation (core buffers, Hybrid-specific buffers)
- Missing buffer detection
- isHybridModeActive helper (replaces scattered pattern)

### Phase 5: Debug Validation Flag ✅

**Files Created:**
- `tests/v2/unit/Test__ValidationConfig.cpp` - 13 unit tests

**Files Modified:**
- `src/v2/utils/DebugEnv.h` - Added ValidationConfig struct
- `src/v2/execution/GraphExecutor.h` - Added validateStageOutputs() declaration
- `src/v2/execution/GraphExecutor.cpp` - Added validation hook and implementation

**ValidationConfig Struct:**
```cpp
struct ValidationConfig
{
    bool validate_buffers = false; ///< Enable buffer validation (LLAMINAR_VALIDATE_BUFFERS)
    bool fail_on_zero = false;     ///< Fail on zero tensor (LLAMINAR_FAIL_ON_ZERO)
    bool fail_on_nan = false;      ///< Fail on NaN/Inf (LLAMINAR_FAIL_ON_NAN)
};
```

**GraphExecutor Integration:**
```cpp
// In executeNode(), after stage execution:
#ifndef NDEBUG
if (success && debugEnv().validation.validate_buffers) {
    success = validateStageOutputs(node);
}
#endif
```

**Usage:**
```bash
# Enable buffer validation with fail-fast on issues
LLAMINAR_VALIDATE_BUFFERS=1 LLAMINAR_FAIL_ON_ZERO=1 LLAMINAR_FAIL_ON_NAN=1 \
./build_v2/llaminar2 -m model.gguf -p "test"
```

**Test Results:** 13/13 tests pass

**Test Cases:**
- Default config: all flags disabled
- LLAMINAR_VALIDATE_BUFFERS enables validation
- LLAMINAR_FAIL_ON_ZERO enables fail-fast on zeros
- LLAMINAR_FAIL_ON_NAN enables fail-fast on NaN/Inf
- All flags can be enabled together
- Config reload updates from environment
- DebugEnv includes validation config
- Valid output passes validation
- Zero output warns but passes (fail disabled)
- Zero output fails (fail enabled)
- NaN output warns but passes (fail disabled)
- NaN output fails (fail enabled)
- Validation disabled skips checks

---

## Test Matrix

### New Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `Test__TensorValidation.cpp` | 20 | Zero/NaN detection |
| `Test__GraphValidator.cpp` | 17 | Buffer flow validation |
| `Test__StageOutputManifest.cpp` | 7 | Stage output contracts |
| `Test__InferenceMode.cpp` | 21 | Mode context |
| `Test__ValidationConfig.cpp` | 13 | Env vars + executor integration |

### Extended Tests

| File | New Tests | Purpose |
|------|-----------|---------|
| `Test__KVCacheAppendStage.cpp` | 2 | V_dequant output |
| `Test__HybridPipeline*.cpp` | 1 | Zero detection in E2E |

---

## Success Criteria

1. **Debug builds fail immediately** when attention receives zero V
2. **Graph validation** catches missing producer for V_dequant
3. **No false positives** - valid graphs still build and run
4. **Hybrid mode logic** consolidated in InferenceMode
5. **All new tests pass** in CI

---

## Files to Create/Modify

### New Files
- `src/v2/tensors/TensorValidation.h`
- `src/v2/execution/GraphValidator.h`
- `src/v2/execution/GraphValidator.cpp`
- `src/v2/execution/InferenceMode.h`
- `tests/v2/unit/Test__TensorValidation.cpp`
- `tests/v2/unit/Test__GraphValidator.cpp`
- `tests/v2/unit/Test__InferenceMode.cpp`

### Modified Files
- `src/v2/execution/BufferRole.h` - Add producer_stage
- `src/v2/execution/ComputeStage.h` - Add getDeclaredOutputs()
- `src/v2/execution/ComputeStage.cpp` - KVCacheAppendStage outputs
- `src/v2/execution/GraphExecutor.cpp` - Validation hook
- `src/v2/utils/DebugEnv.h` - Validation flags
- `src/v2/kernels/cpu/attention/CPUAttentionKernelT.h` - Guards
