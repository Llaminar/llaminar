# Action Plan: Tensor Parallelism Infrastructure Improvements

**Created**: December 22, 2025  
**Status**: Planning  
**Priority**: High (prevents silent failures in TP and multi-device code)

---

## Overview

This document provides implementation plans for 8 architectural improvements identified during tensor parallelism debugging. Each section includes:
- Problem statement
- Proposed solution with code samples
- Files to modify
- Estimated effort
- Dependencies

---

## Task 1: Unified Tensor Data Access Interface

### Problem
`TensorBase`, `FP32Tensor`, `TensorSlice`, and quantized tensors all have different ways to access raw data. This leads to fragile `dynamic_cast` chains that fail silently.

### Solution
Add universal data access methods to `TensorBase`.

### Files to Modify
- `src/v2/tensors/TensorBase.h`
- `src/v2/tensors/FP32Tensor.h`
- `src/v2/tensors/TensorSlice.h`
- `src/v2/tensors/Q4_0Tensor.h` (and other quantized tensors)

### Code Sample

**TensorBase.h** - Add virtual methods:
```cpp
class TensorBase {
public:
    // ... existing code ...

    //=========================================================================
    // Unified Data Access Interface
    //=========================================================================
    
    /**
     * @brief Returns pointer to FP32 data if tensor is FP32-backed.
     * 
     * For FP32Tensor and TensorSlice (of FP32): returns direct pointer.
     * For quantized tensors: returns nullptr (use dequantize() instead).
     * 
     * @return const float* Pointer to FP32 data, or nullptr if not FP32-backed.
     */
    virtual const float* fp32_data() const { return nullptr; }
    
    /**
     * @brief Returns mutable pointer to FP32 data for activation tensors.
     * 
     * @return float* Mutable pointer, or nullptr if not mutable/not FP32.
     */
    virtual float* mutable_fp32_data() { return nullptr; }
    
    /**
     * @brief Returns true if this tensor can provide fp32_data() directly.
     */
    virtual bool is_fp32_backed() const { return false; }
    
    /**
     * @brief Returns true if this tensor represents a slice/view of another tensor.
     */
    virtual bool is_view() const { return false; }
};
```

**FP32Tensor.h** - Implement methods:
```cpp
class FP32Tensor : public TensorBase {
public:
    // ... existing code ...
    
    const float* fp32_data() const override { return data_; }
    float* mutable_fp32_data() override { return data_; }
    bool is_fp32_backed() const override { return true; }
    bool is_view() const override { return false; }
    
    // Keep existing data() for backward compatibility
    const float* data() const { return data_; }
    float* data() { return data_; }
};
```

**TensorSlice.h** - Implement methods:
```cpp
class TensorSlice : public TensorBase {
public:
    // ... existing code ...
    
    const float* fp32_data() const override { return slice_data_; }
    float* mutable_fp32_data() override { return slice_data_; }
    bool is_fp32_backed() const override { return true; }
    bool is_view() const override { return true; }
    
    // Keep existing data() for backward compatibility
    const float* data() const { return slice_data_; }
};
```

**Quantized Tensors** - Return nullptr:
```cpp
class Q4_0Tensor : public TensorBase {
public:
    // fp32_data() inherited as nullptr (correct - quantized tensors need dequantization)
    bool is_fp32_backed() const override { return false; }
    bool is_view() const override { return false; }
};
```

### Migration Pattern
Replace all instances of:
```cpp
// OLD (fragile)
auto* fp32 = dynamic_cast<FP32Tensor*>(tensor);
const float* data = fp32 ? fp32->data() : nullptr;

// NEW (safe)
const float* data = tensor->fp32_data();
```

### Estimated Effort
- **Implementation**: 2 hours
- **Migration**: 4 hours (find and replace all dynamic_cast patterns)
- **Testing**: 2 hours

---

## Task 2: Explicit Bias Handling in GEMM Stages

### Problem
Bias is optionally passed to GEMM, and it's easy to forget or mishandle it. The current pattern silently produces wrong results.

### Solution
Create a `GEMMStageConfig` struct with explicit bias requirements and fail-fast validation.

### Files to Modify
- `src/v2/execution/ComputeStage.h`
- `src/v2/execution/ComputeStage.cpp`
- `src/v2/pipelines/qwen/Qwen2Graph.cpp`

### Code Sample

**ComputeStage.h** - Add config struct:
```cpp
/**
 * @brief Configuration for GEMM-based compute stages.
 * 
 * Provides explicit control over bias handling with fail-fast validation.
 */
struct GEMMStageConfig {
    // Required parameters
    TensorBase* weight = nullptr;
    std::string stage_name;
    
    // Bias configuration
    TensorBase* bias = nullptr;
    bool bias_required = false;  // If true, throws if bias is nullptr
    
    // Shape parameters
    size_t m = 0;  // Batch/sequence dimension
    size_t n = 0;  // Output dimension
    size_t k = 0;  // Input dimension (must match weight rows if row-major)
    
    // Optional parameters
    float alpha = 1.0f;
    float beta = 0.0f;
    
    /**
     * @brief Validates configuration and throws on error.
     */
    void validate() const {
        if (!weight) {
            throw std::runtime_error("GEMMStageConfig: weight is required for stage '" + stage_name + "'");
        }
        if (bias_required && !bias) {
            throw std::runtime_error("GEMMStageConfig: bias is required but not provided for stage '" + stage_name + "'");
        }
        if (m == 0 || n == 0 || k == 0) {
            throw std::runtime_error("GEMMStageConfig: dimensions (m,n,k) must be non-zero for stage '" + stage_name + "'");
        }
    }
    
    /**
     * @brief Returns bias data pointer, or nullptr if no bias.
     */
    const float* bias_data() const {
        return bias ? bias->fp32_data() : nullptr;
    }
};
```

**GEMMStage constructor** - Use config:
```cpp
class GEMMStage : public ComputeStage {
public:
    explicit GEMMStage(GEMMStageConfig config) 
        : config_(std::move(config)) 
    {
        config_.validate();  // Fail fast!
    }
    
    void execute() override {
        // Use config_.bias_data() - guaranteed safe after validate()
        kernel_->multiply(
            input_->fp32_data(),
            output_->mutable_fp32_data(),
            config_.m, config_.n, config_.k,
            config_.alpha, config_.beta,
            config_.bias_data()
        );
    }
    
private:
    GEMMStageConfig config_;
};
```

**Usage in Qwen2Graph.cpp**:
```cpp
// OLD (error-prone)
auto gemm_stage = std::make_unique<GEMMStage>(
    weight, bias, m, n, k, "wo_proj"
);

// NEW (explicit, validated)
GEMMStageConfig config;
config.stage_name = "layer" + std::to_string(layer_idx) + "_wo_proj";
config.weight = wo_weight;
config.bias = wo_bias;
config.bias_required = (wo_bias != nullptr);  // Or: = true if model always has bias
config.m = seq_len;
config.n = d_model;
config.k = local_n_heads * head_dim;
config.validate();  // Fail fast before stage creation

auto gemm_stage = std::make_unique<GEMMStage>(config);
```

### Estimated Effort
- **Implementation**: 3 hours
- **Migration**: 4 hours (update all GEMMStage instantiations)
- **Testing**: 2 hours

---

## Task 3: Stage Execution Tracing Infrastructure

### Problem
Debugging required adding ad-hoc logging, then removing it. No systematic way to trace tensor values through pipeline.

### Solution
Built-in tracing infrastructure controlled by `DebugEnv`, with zero cost when disabled.

### Files to Modify
- `src/v2/utils/DebugEnv.h`
- `src/v2/utils/DebugEnv.cpp`
- `src/v2/execution/ComputeStage.h`
- `src/v2/execution/ComputeStage.cpp`

### Code Sample

**DebugEnv.h** - Add tracing config:
```cpp
struct ExecutionConfig {
    bool trace_stages = false;          // LLAMINAR_TRACE_STAGES
    bool trace_shapes = false;          // LLAMINAR_TRACE_SHAPES
    int trace_sample_count = 8;         // Number of elements to print
    std::string trace_filter = "";      // Only trace stages matching this substring
};

struct DebugEnvSnapshot {
    // ... existing fields ...
    ExecutionConfig execution;
};
```

**DebugEnv.cpp** - Parse new env vars:
```cpp
void DebugEnvSnapshot::loadFromEnvironment() {
    // ... existing code ...
    
    // Execution tracing
    execution.trace_stages = getEnvBool("LLAMINAR_TRACE_STAGES", false);
    execution.trace_shapes = getEnvBool("LLAMINAR_TRACE_SHAPES", false);
    execution.trace_sample_count = getEnvInt("LLAMINAR_TRACE_SAMPLE_COUNT", 8);
    execution.trace_filter = getEnvString("LLAMINAR_TRACE_FILTER", "");
}
```

**ComputeStage.h** - Add tracing helpers:
```cpp
class ComputeStage {
public:
    // ... existing code ...
    
protected:
    /**
     * @brief Trace input tensor values (only if tracing enabled).
     */
    void traceInput(const std::string& name, const TensorBase* tensor) {
        if (!shouldTrace()) return;
        traceImpl("INPUT", name, tensor);
    }
    
    /**
     * @brief Trace output tensor values (only if tracing enabled).
     */
    void traceOutput(const std::string& name, const TensorBase* tensor) {
        if (!shouldTrace()) return;
        traceImpl("OUTPUT", name, tensor);
    }
    
    /**
     * @brief Trace intermediate values.
     */
    void traceIntermediate(const std::string& name, const float* data, size_t count) {
        if (!shouldTrace()) return;
        const auto& env = debugEnv().execution;
        size_t n = std::min(count, static_cast<size_t>(env.trace_sample_count));
        LOG_TRACE("[" << stageName() << "] " << name << "[0:" << n << "]=" 
                  << formatFloatArray(data, n));
    }
    
private:
    bool shouldTrace() const {
        const auto& env = debugEnv().execution;
        if (!env.trace_stages) return false;
        if (!env.trace_filter.empty() && 
            stageName().find(env.trace_filter) == std::string::npos) {
            return false;
        }
        return true;
    }
    
    void traceImpl(const char* direction, const std::string& name, const TensorBase* tensor) {
        const auto& env = debugEnv().execution;
        
        std::ostringstream ss;
        ss << "[" << stageName() << "] " << direction << " " << name;
        
        if (env.trace_shapes) {
            ss << " shape=" << shapeToString(tensor->shape());
        }
        
        if (tensor->fp32_data()) {
            size_t n = std::min(tensor->numel(), static_cast<size_t>(env.trace_sample_count));
            ss << " data[0:" << n << "]=" << formatFloatArray(tensor->fp32_data(), n);
        } else {
            ss << " (non-FP32 tensor)";
        }
        
        LOG_TRACE(ss.str());
    }
};
```

**Usage in stage implementations**:
```cpp
void FusedQKVGEMMStage::execute() {
    traceInput("hidden_states", input_);
    
    // ... computation ...
    
    traceOutput("Q", q_output_);
    traceOutput("K", k_output_);
    traceOutput("V", v_output_);
}
```

### Usage
```bash
# Enable all stage tracing
LLAMINAR_TRACE_STAGES=1 ./run_llaminar.sh -m model.gguf -p "test"

# Enable with shape info
LLAMINAR_TRACE_STAGES=1 LLAMINAR_TRACE_SHAPES=1 ./run_llaminar.sh ...

# Filter to specific stages
LLAMINAR_TRACE_STAGES=1 LLAMINAR_TRACE_FILTER="layer0" ./run_llaminar.sh ...

# More detailed samples
LLAMINAR_TRACE_STAGES=1 LLAMINAR_TRACE_SAMPLE_COUNT=16 ./run_llaminar.sh ...
```

### Estimated Effort
- **Implementation**: 3 hours
- **Instrumenting stages**: 4 hours (add trace calls to all stages)
- **Testing**: 1 hour

---

## Task 4: Tensor Validity Assertions

### Problem
Use-after-move/use-after-free bugs are silent until they segfault.

### Solution
Debug-mode validity tracking with assertions on access.

### Files to Modify
- `src/v2/tensors/TensorBase.h`
- `src/v2/tensors/TensorBase.cpp` (if exists)
- `src/v2/execution/DeviceGraphBufferManager.h`

### Code Sample

**TensorBase.h** - Add validity tracking:
```cpp
class TensorBase {
public:
    TensorBase() = default;
    virtual ~TensorBase() = default;
    
    // ... existing code ...
    
    //=========================================================================
    // Debug: Validity Tracking (only in debug builds)
    //=========================================================================
    
#ifndef NDEBUG
    /**
     * @brief Mark this tensor as invalidated (e.g., after buffer reuse).
     * 
     * Once invalidated, any access to data() will assert.
     */
    void invalidate(const std::string& reason = "") {
        valid_ = false;
        invalidation_reason_ = reason;
    }
    
    /**
     * @brief Check if tensor is still valid for access.
     */
    bool isValid() const { return valid_; }
    
    /**
     * @brief Get reason for invalidation (empty if still valid).
     */
    const std::string& invalidationReason() const { return invalidation_reason_; }
    
protected:
    /**
     * @brief Assert that tensor is valid. Called before data access.
     */
    void assertValid(const char* operation) const {
        if (!valid_) {
            std::ostringstream ss;
            ss << "Accessing invalidated tensor during " << operation;
            if (!invalidation_reason_.empty()) {
                ss << ". Reason: " << invalidation_reason_;
            }
            ss << ". Tensor: " << debugName();
            
            // Log before assert for better debugging
            LOG_ERROR(ss.str());
            assert(false && "Accessing invalidated tensor");
        }
    }
    
private:
    bool valid_ = true;
    std::string invalidation_reason_;
    
#else
    // No-op in release builds
    void invalidate(const std::string& = "") {}
    bool isValid() const { return true; }
    const std::string& invalidationReason() const { static std::string empty; return empty; }
    
protected:
    void assertValid(const char*) const {}
#endif

    // Optional debug name for better error messages
    std::string debug_name_;
    
public:
    void setDebugName(const std::string& name) { debug_name_ = name; }
    const std::string& debugName() const { return debug_name_; }
};
```

**FP32Tensor.h** - Add assertions:
```cpp
class FP32Tensor : public TensorBase {
public:
    const float* data() const { 
        assertValid("data() const");
        return data_; 
    }
    
    float* data() { 
        assertValid("data() mutable");
        return data_; 
    }
    
    const float* fp32_data() const override {
        assertValid("fp32_data()");
        return data_;
    }
    
    float* mutable_fp32_data() override {
        assertValid("mutable_fp32_data()");
        return data_;
    }
};
```

**DeviceGraphBufferManager** - Invalidate on release:
```cpp
void DeviceGraphBufferManager::releaseBuffer(TensorBase* tensor) {
    tensor->invalidate("Released back to buffer pool by DeviceGraphBufferManager");
    // ... return to pool ...
}

void DeviceGraphBufferManager::aliasBuffer(TensorBase* original, TensorBase* alias) {
    original->invalidate("Aliased to " + alias->debugName());
    // ... setup alias ...
}
```

### Estimated Effort
- **Implementation**: 2 hours
- **Adding invalidation calls**: 2 hours
- **Testing**: 2 hours

---

## Task 5: Deterministic Testing Mode

### Problem
Non-deterministic sampling makes debugging difficult.

### Solution
Add `--deterministic` flag and `DeterministicConfig` in DebugEnv.

### Files to Modify
- `src/v2/utils/ArgParser.h`
- `src/v2/utils/ArgParser.cpp`
- `src/v2/Main.cpp`

### Code Sample

**ArgParser.h** - Add flag:
```cpp
struct Args {
    // ... existing fields ...
    
    bool deterministic = false;  // --deterministic flag
};
```

**ArgParser.cpp** - Parse flag:
```cpp
void ArgParser::parse(int argc, char** argv) {
    // ... existing code ...
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--deterministic") {
            args_.deterministic = true;
            continue;
        }
        
        // ... existing arg handling ...
    }
    
    // Apply deterministic settings after parsing
    if (args_.deterministic) {
        applyDeterministicSettings();
    }
}

void ArgParser::applyDeterministicSettings() {
    // Force greedy sampling
    if (args_.temperature != 0.0f) {
        LOG_INFO("Deterministic mode: forcing temperature=0 (was " << args_.temperature << ")");
        args_.temperature = 0.0f;
    }
    
    // Force fixed seed
    if (args_.seed == 0) {
        args_.seed = 42;
        LOG_INFO("Deterministic mode: setting seed=42");
    }
    
    // Disable any other stochastic features
    args_.top_p = 1.0f;
    args_.top_k = 0;  // 0 = disabled
    
    LOG_INFO("Deterministic mode enabled: temperature=0, seed=" << args_.seed 
             << ", top_p=1.0, top_k=disabled");
}
```

**Main.cpp** - Add to help:
```cpp
void printHelp() {
    std::cout << R"(
Options:
  ...existing options...
  
  --deterministic     Enable deterministic mode for reproducible outputs.
                      Forces: temperature=0, seed=42, top_p=1.0, top_k=disabled.
                      Useful for debugging and comparison testing.
)";
}
```

### Usage
```bash
# Deterministic inference
./run_llaminar.sh -m model.gguf -p "test" --deterministic

# Explicit (equivalent)
./run_llaminar.sh -m model.gguf -p "test" -t 0 --seed 42
```

### Estimated Effort
- **Implementation**: 1 hour
- **Testing**: 30 minutes

---

## Task 6: MPI Operation Logging

### Problem
Hard to verify MPI collective operations are happening correctly.

### Solution
Add MPI logging config to DebugEnv and instrument MPI stages.

### Files to Modify
- `src/v2/utils/DebugEnv.h`
- `src/v2/utils/DebugEnv.cpp`
- `src/v2/execution/ComputeStage.cpp` (AllreduceStage, AllGatherStage)

### Code Sample

**DebugEnv.h** - Add MPI config:
```cpp
struct MPIConfig {
    bool log_collectives = false;       // LLAMINAR_MPI_LOG_COLLECTIVES
    bool verify_checksums = false;      // LLAMINAR_MPI_VERIFY_CHECKSUMS
    bool log_timing = false;            // LLAMINAR_MPI_LOG_TIMING
};

struct DebugEnvSnapshot {
    // ... existing fields ...
    MPIConfig mpi;
};
```

**DebugEnv.cpp** - Parse:
```cpp
void DebugEnvSnapshot::loadFromEnvironment() {
    // ... existing code ...
    
    mpi.log_collectives = getEnvBool("LLAMINAR_MPI_LOG_COLLECTIVES", false);
    mpi.verify_checksums = getEnvBool("LLAMINAR_MPI_VERIFY_CHECKSUMS", false);
    mpi.log_timing = getEnvBool("LLAMINAR_MPI_LOG_TIMING", false);
}
```

**ComputeStage.cpp** - Instrument AllreduceStage:
```cpp
void AllreduceStage::execute() {
    const auto& mpi_env = debugEnv().mpi;
    
    // Pre-operation logging
    if (mpi_env.log_collectives) {
        LOG_DEBUG("[MPI] AllReduce START: stage=" << stageName() 
                  << " count=" << count_ << " buffer=" << static_cast<void*>(buffer_));
    }
    
    // Checksum before (expensive, only if enabled)
    float checksum_before = 0.0f;
    if (mpi_env.verify_checksums) {
        for (size_t i = 0; i < count_; ++i) {
            checksum_before += buffer_[i];
        }
    }
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Actual MPI call
    int result = MPI_Allreduce(MPI_IN_PLACE, buffer_, count_, MPI_FLOAT, MPI_SUM, comm_);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Post-operation logging
    if (mpi_env.log_collectives) {
        LOG_DEBUG("[MPI] AllReduce END: result=" << result);
    }
    
    if (mpi_env.log_timing) {
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        LOG_DEBUG("[MPI] AllReduce timing: " << ms << " ms for " << count_ << " elements");
    }
    
    if (mpi_env.verify_checksums) {
        float checksum_after = 0.0f;
        for (size_t i = 0; i < count_; ++i) {
            checksum_after += buffer_[i];
        }
        LOG_DEBUG("[MPI] AllReduce checksums: before=" << checksum_before 
                  << " after=" << checksum_after);
    }
    
    if (result != MPI_SUCCESS) {
        LOG_ERROR("[MPI] AllReduce FAILED with code " << result);
        throw std::runtime_error("MPI_Allreduce failed");
    }
}
```

### Usage
```bash
# Log all MPI collectives
LLAMINAR_MPI_LOG_COLLECTIVES=1 mpirun -np 2 ./run_llaminar.sh ...

# With timing info
LLAMINAR_MPI_LOG_COLLECTIVES=1 LLAMINAR_MPI_LOG_TIMING=1 mpirun -np 2 ...

# With checksum verification (slow, for debugging)
LLAMINAR_MPI_VERIFY_CHECKSUMS=1 mpirun -np 2 ...
```

### Estimated Effort
- **Implementation**: 2 hours
- **Testing**: 1 hour

---

## Task 7: Tensor Shape Validation at Stage Boundaries

### Problem
Shape mismatches between stages cause subtle bugs.

### Solution
Add shape validation infrastructure to ComputeStage base class.

### Files to Modify
- `src/v2/execution/ComputeStage.h`
- `src/v2/execution/ComputeStage.cpp`

### Code Sample

**ComputeStage.h** - Add validation infrastructure:
```cpp
class ComputeStage {
protected:
    //=========================================================================
    // Shape Validation
    //=========================================================================
    
    /**
     * @brief Contract for a single tensor's expected shape.
     */
    struct TensorShapeContract {
        std::string name;
        const TensorBase* tensor;
        std::vector<size_t> expected_shape;
        bool allow_broadcast = false;  // If true, trailing dims of 1 are OK
    };
    
    /**
     * @brief Validate tensor shapes match expected contracts.
     * 
     * @throws std::runtime_error if any shape doesn't match.
     */
    void validateShapes(std::initializer_list<TensorShapeContract> contracts) {
        for (const auto& c : contracts) {
            if (!c.tensor) {
                throw std::runtime_error(
                    stageName() + ": " + c.name + " is nullptr"
                );
            }
            
            const auto& actual = c.tensor->shape();
            const auto& expected = c.expected_shape;
            
            bool match = shapesMatch(actual, expected, c.allow_broadcast);
            
            if (!match) {
                std::ostringstream ss;
                ss << stageName() << ": " << c.name << " shape mismatch. "
                   << "Expected " << shapeToString(expected) 
                   << " but got " << shapeToString(actual);
                throw std::runtime_error(ss.str());
            }
        }
    }
    
    /**
     * @brief Validate a single tensor's shape.
     */
    void validateShape(const std::string& name, const TensorBase* tensor,
                       const std::vector<size_t>& expected) {
        validateShapes({{name, tensor, expected, false}});
    }
    
    /**
     * @brief Validate two tensors have compatible shapes for matmul.
     * 
     * For A @ B where A is (m, k) and B is (k, n), validates k dimensions match.
     */
    void validateMatmulShapes(const std::string& a_name, const TensorBase* a,
                               const std::string& b_name, const TensorBase* b) {
        if (!a || !b) {
            throw std::runtime_error(stageName() + ": null tensor in matmul");
        }
        
        const auto& a_shape = a->shape();
        const auto& b_shape = b->shape();
        
        if (a_shape.size() < 2 || b_shape.size() < 2) {
            throw std::runtime_error(
                stageName() + ": matmul requires 2D tensors. " +
                a_name + " has " + std::to_string(a_shape.size()) + " dims, " +
                b_name + " has " + std::to_string(b_shape.size()) + " dims"
            );
        }
        
        size_t a_k = a_shape[a_shape.size() - 1];  // Last dim of A
        size_t b_k = b_shape[b_shape.size() - 2];  // Second-to-last dim of B
        
        if (a_k != b_k) {
            std::ostringstream ss;
            ss << stageName() << ": matmul shape mismatch. "
               << a_name << " has k=" << a_k << " (shape " << shapeToString(a_shape) << "), "
               << b_name << " has k=" << b_k << " (shape " << shapeToString(b_shape) << ")";
            throw std::runtime_error(ss.str());
        }
    }
    
private:
    static bool shapesMatch(const std::vector<size_t>& actual,
                            const std::vector<size_t>& expected,
                            bool allow_broadcast) {
        if (actual.size() != expected.size()) {
            return false;
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) {
                if (allow_broadcast && (actual[i] == 1 || expected[i] == 1)) {
                    continue;
                }
                return false;
            }
        }
        return true;
    }
    
    static std::string shapeToString(const std::vector<size_t>& shape) {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << shape[i];
        }
        ss << ")";
        return ss.str();
    }
};
```

**Usage in stages**:
```cpp
void FusedQKVGEMMStage::execute() {
    // Validate all input shapes at start
    validateShapes({
        {"input", input_, {batch_size_, d_model_}},
        {"wq", wq_, {n_heads_ * head_dim_, d_model_}},
        {"wk", wk_, {n_kv_heads_ * head_dim_, d_model_}},
        {"wv", wv_, {n_kv_heads_ * head_dim_, d_model_}},
    });
    
    // Validate matmul compatibility
    validateMatmulShapes("input", input_, "wq", wq_);
    
    // ... rest of execution ...
}
```

### Estimated Effort
- **Implementation**: 2 hours
- **Adding validation to stages**: 3 hours
- **Testing**: 1 hour

---

## Task 8: Comparison Testing Infrastructure

### Problem
Had to manually compare single-rank vs 2-rank outputs.

### Solution
Add `--compare-ranks` mode that automatically runs and compares different configurations.

### Files to Modify
- `src/v2/utils/ArgParser.h`
- `src/v2/utils/ArgParser.cpp`
- `src/v2/Main.cpp` (or new `src/v2/tools/CompareRanks.cpp`)
- `scripts/compare_ranks.sh` (shell wrapper)

### Code Sample

**scripts/compare_ranks.sh** - Shell script approach (simpler):
```bash
#!/bin/bash
# compare_ranks.sh - Compare inference output between different rank configurations
#
# Usage: ./compare_ranks.sh -m MODEL -p PROMPT [-n TOKENS] [--ranks1 N1] [--ranks2 N2]

set -e

# Defaults
MODEL=""
PROMPT=""
TOKENS=20
RANKS1=1
RANKS2=2
VERBOSE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -n|--tokens)
            TOKENS="$2"
            shift 2
            ;;
        --ranks1)
            RANKS1="$2"
            shift 2
            ;;
        --ranks2)
            RANKS2="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 -m MODEL -p PROMPT [-n TOKENS] [--ranks1 N1] [--ranks2 N2]"
            echo ""
            echo "Options:"
            echo "  -m, --model    Path to GGUF model file (required)"
            echo "  -p, --prompt   Inference prompt (required)"
            echo "  -n, --tokens   Number of tokens to generate (default: 20)"
            echo "  --ranks1       First rank configuration (default: 1)"
            echo "  --ranks2       Second rank configuration (default: 2)"
            echo "  -v, --verbose  Show detailed output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$PROMPT" ]]; then
    echo "Error: -m MODEL and -p PROMPT are required"
    exit 1
fi

# Temp files for output
TMPDIR=$(mktemp -d)
OUTPUT1="$TMPDIR/output_ranks${RANKS1}.txt"
OUTPUT2="$TMPDIR/output_ranks${RANKS2}.txt"
TOKENS1="$TMPDIR/tokens_ranks${RANKS1}.txt"
TOKENS2="$TMPDIR/tokens_ranks${RANKS2}.txt"

cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "=========================================="
echo "Comparison Test: $RANKS1 rank(s) vs $RANKS2 rank(s)"
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $TOKENS"
echo "=========================================="

# Run with first rank configuration
echo ""
echo "Running with $RANKS1 rank(s)..."
if [[ $RANKS1 -eq 1 ]]; then
    LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/llaminar2 \
        -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
        2>&1 | tee "$OUTPUT1" | grep "Sampled token" | awk '{print $NF}' > "$TOKENS1"
else
    LLAMINAR_LOG_LEVEL=DEBUG mpirun --allow-run-as-root -np $RANKS1 \
        ./build_v2/llaminar2 \
        -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
        2>&1 | tee "$OUTPUT1" | grep "Sampled token" | awk '{print $NF}' > "$TOKENS1"
fi

# Run with second rank configuration
echo ""
echo "Running with $RANKS2 rank(s)..."
if [[ $RANKS2 -eq 1 ]]; then
    LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/llaminar2 \
        -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
        2>&1 | tee "$OUTPUT2" | grep "Sampled token" | awk '{print $NF}' > "$TOKENS2"
else
    LLAMINAR_LOG_LEVEL=DEBUG mpirun --allow-run-as-root -np $RANKS2 \
        ./build_v2/llaminar2 \
        -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
        2>&1 | tee "$OUTPUT2" | grep "Sampled token" | awk '{print $NF}' > "$TOKENS2"
fi

# Compare tokens
echo ""
echo "=========================================="
echo "COMPARISON RESULTS"
echo "=========================================="

TOKENS1_LIST=$(cat "$TOKENS1" | tr '\n' ' ')
TOKENS2_LIST=$(cat "$TOKENS2" | tr '\n' ' ')

echo "Tokens ($RANKS1 rank):  $TOKENS1_LIST"
echo "Tokens ($RANKS2 ranks): $TOKENS2_LIST"
echo ""

if diff -q "$TOKENS1" "$TOKENS2" > /dev/null 2>&1; then
    echo "✅ PASS: Token sequences are IDENTICAL"
    exit 0
else
    echo "❌ FAIL: Token sequences DIFFER"
    echo ""
    echo "Diff:"
    diff "$TOKENS1" "$TOKENS2" || true
    
    # Find first divergence point
    echo ""
    echo "First divergence:"
    paste "$TOKENS1" "$TOKENS2" | awk -F'\t' 'BEGIN{i=0} {i++; if($1!=$2){print "Token "i": "$1" vs "$2; exit}}'
    
    exit 1
fi
```

### Usage
```bash
# Basic comparison
./scripts/compare_ranks.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello world" -n 20

# Custom rank configurations
./scripts/compare_ranks.sh -m model.gguf -p "test" --ranks1 1 --ranks2 4

# Verbose output
./scripts/compare_ranks.sh -m model.gguf -p "test" -v
```

### Estimated Effort
- **Implementation**: 2 hours (shell script)
- **Testing**: 1 hour
- **CI Integration**: 1 hour (optional)

---

## Implementation Priority & Timeline

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| **Task 1: Unified Data Access** | P0 (Critical) | 8h | None |
| **Task 2: Explicit Bias Handling** | P0 (Critical) | 9h | Task 1 |
| **Task 5: Deterministic Mode** | P1 (High) | 1.5h | None |
| **Task 3: Stage Tracing** | P1 (High) | 8h | None |
| **Task 4: Validity Assertions** | P1 (High) | 6h | Task 1 |
| **Task 6: MPI Logging** | P2 (Medium) | 3h | None |
| **Task 7: Shape Validation** | P2 (Medium) | 6h | Task 1 |
| **Task 8: Comparison Testing** | P2 (Medium) | 4h | Task 5 |

**Total Estimated Effort**: ~45.5 hours

### Recommended Implementation Order

**Phase 1: Foundation (P0)** - 17 hours
1. Task 1: Unified Data Access
2. Task 2: Explicit Bias Handling

**Phase 2: Debugging Infrastructure (P1)** - 15.5 hours
3. Task 5: Deterministic Mode (quick win)
4. Task 3: Stage Tracing
5. Task 4: Validity Assertions

**Phase 3: Enhanced Tooling (P2)** - 13 hours
6. Task 6: MPI Logging
7. Task 7: Shape Validation
8. Task 8: Comparison Testing

---

## Testing Strategy

After implementing each task:

1. **Unit Tests**: Add tests for new interfaces
2. **Regression Tests**: Ensure existing tests still pass
3. **TP Comparison Test**: Run `compare_ranks.sh` to verify identical output
4. **Performance Check**: Ensure debug infrastructure has zero cost when disabled

---

## Definition of Done

Each task is complete when:
- [ ] Code implemented and compiles
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] TP comparison test passes (single-rank == multi-rank output)
- [ ] Documentation updated (copilot-instructions.md if applicable)
- [ ] No performance regression when feature disabled
