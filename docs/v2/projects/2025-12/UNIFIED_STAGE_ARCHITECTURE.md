# Unified Stage Architecture Proposal

## Current Problem

The current ComputeStage design has separate classes for CPU and GPU:

```cpp
// Current: Two separate classes
class GEMMStage : public IComputeStage { ... };      // CPU only
class GPUGEMMStage : public IComputeStage { ... };   // GPU only

// Factory selects at construction time
switch (target_backend) {
    case CPU_OPENBLAS: return make_unique<GEMMStage>(params);
    case GPU_CUDA:     return make_unique<GPUGEMMStage>(params, backend);
}
```

**Problems:**
1. **Code duplication**: Same params struct, same estimatedFlops(), same getDumpInfo()
2. **Backend locked at construction**: Can't move stage between devices at runtime
3. **Inconsistent patterns**: Some stages use KernelFactory, others have inline implementations
4. **GPU stages are stubs**: Most GPU*Stage classes just delegate to IBackend anyway

## Target Architecture

**Single stage class per operation** that delegates to KernelFactory at execute() time:

```cpp
// Unified: One class, multi-backend
class GEMMStage : public IComputeStage {
public:
    bool execute(IDeviceContext *ctx) override {
        // Get device type from context at runtime
        DeviceType dev = ctx->deviceType();  // CPU, CUDA, ROCm
        
        // Create kernel for this device type
        auto kernel = KernelFactory::createGemm(params_.B, dev);
        
        // Execute on the device
        return kernel->multiply(A_data, C_data, m, n, k, alpha, beta);
    }
};
```

### Key Changes

1. **IDeviceContext gains deviceType()**:
```cpp
class IDeviceContext {
public:
    virtual DeviceType deviceType() const = 0;
    // ...
};

class CPUDeviceContext : public IDeviceContext {
    DeviceType deviceType() const override { return DeviceType::CPU; }
};

class CUDADeviceContext : public IDeviceContext {
    DeviceType deviceType() const override { return DeviceType::CUDA; }
};
```

2. **Stages use KernelFactory for all operations**:
```cpp
class RMSNormStage : public IComputeStage {
    bool execute(IDeviceContext *ctx) override {
        DeviceType dev = ctx->deviceType();
        auto kernel = KernelFactory::createRMSNorm(params_.input, dev);
        return kernel->apply(params_.gamma, params_.output, params_.eps);
    }
};
```

3. **Remove GPU*Stage classes entirely**:
- ~~GPUGEMMStage~~ → GEMMStage handles all backends
- ~~GPURMSNormStage~~ → RMSNormStage handles all backends
- etc.

4. **ComputeStageFactory simplifies**:
```cpp
// Before: Factory had switch on backend
std::unique_ptr<IComputeStage> ComputeStageFactory::createGEMM(
    const GEMMStage::Params &params,
    ComputeBackendType target_backend)  // <-- No longer needed
{
    // Just create the stage - it handles all backends
    return std::make_unique<GEMMStage>(params);
}
```

5. **supportsBackend() becomes a capability query**:
```cpp
bool GEMMStage::supportsBackend(ComputeBackendType backend) const {
    DeviceType dev = toDeviceType(backend);
    // Ask KernelFactory if it can create a kernel for this tensor+device
    return KernelFactory::canCreateGemm(params_.B, dev);
}
```

## Migration Plan

### Phase 1: Add deviceType() to IDeviceContext ✅ Complete

```cpp
// In DeviceContext.h
class IDeviceContext {
public:
    virtual DeviceType deviceType() const = 0;
};
```

### Phase 2: Unify GEMMStage (template for others) ✅ Complete

1. GEMMStage already uses KernelFactory - verify it works with GPU context
2. Remove GPUGEMMStage 
3. Update tests

### Phase 3: Unify remaining stages ✅ Complete

All stages now use KernelFactory at execute-time for device dispatch.

### Phase 4: Remove GPU*Stage classes ✅ Complete

Deleted ~430 lines of GPU stage code:
- ~~GPUGEMMStage~~
- ~~GPURMSNormStage~~
- ~~GPUSwiGLUStage~~
- ~~GPUResidualAddStage~~
- ~~GPURoPEStage~~
- ~~GPUAttentionStage~~

### Phase 5: Simplify ComputeStageFactory ✅ Complete

Removed `target_backend` parameter from all factory methods - stages are backend-agnostic.

**Before:**
```cpp
static std::unique_ptr<IComputeStage> createGEMM(
    const GEMMStage::Params &params,
    ComputeBackendType target_backend);  // No longer needed
```

**After:**
```cpp
static std::unique_ptr<IComputeStage> createGEMM(
    const GEMMStage::Params &params);  // Backend-agnostic
```

## Benefits

1. **Single source of truth** per operation type
2. **Runtime device selection** - can move work between devices dynamically
3. **Easier maintenance** - fix once, works everywhere
4. **Consistent with KernelFactory pattern** - already proven to work
5. **Simpler factory** - no backend switch needed
6. **Better for heterogeneous execution** - same stage can run on CPU or GPU

## Example: Unified GEMMStage

```cpp
class GEMMStage : public IComputeStage {
public:
    struct Params {
        TensorBase* A;       // Activation [M, K]
        TensorBase* B;       // Weight [K, N] (quantized)
        TensorBase* C;       // Output [M, N]
        float alpha = 1.0f;
        float beta = 0.0f;
    };

    bool execute(IDeviceContext *ctx) override {
        if (!params_.A || !params_.B || !params_.C) {
            LOG_ERROR("[GEMMStage] Null tensor");
            return false;
        }

        // Get device type from execution context
        DeviceType dev = ctx->deviceType();
        
        // Get cached kernel for this weight tensor + device
        ITensorGemm* kernel = KernelFactory::getOrCreateGemm(params_.B, dev);
        if (!kernel) {
            LOG_ERROR("[GEMMStage] Failed to create kernel for device " << to_string(dev));
            return false;
        }

        // Get dimensions from tensors (self-describing!)
        int M = static_cast<int>(params_.A->rows());
        int K = static_cast<int>(params_.A->cols());
        int N = static_cast<int>(params_.B->cols());

        // Execute - kernel handles FP32 vs Q8_1 activations internally
        return kernel->multiply(
            params_.A->data(), params_.C->mutable_data(),
            M, N, K, params_.alpha, params_.beta
        );
    }

    bool supportsBackend(ComputeBackendType backend) const override {
        DeviceType dev = toDeviceType(backend);
        return KernelFactory::canCreateGemm(params_.B, dev);
    }
    
    // ... estimatedFlops(), etc. unchanged
};
```

## Open Questions

1. **Kernel caching**: Should stages cache their kernels, or rely on KernelFactory's cache?
   - Recommendation: Use KernelFactory cache - it already handles invalidation

2. **Device context lifetime**: Who owns the IDeviceContext?
   - Current: LayerExecutor creates and owns contexts
   - Keep this - contexts are per-execution, stages are reusable

3. **Multi-GPU dispatch**: How to handle when tensor is on different GPU than context?
   - Stage should check tensor device vs context device
   - Cross-device copy if needed (or error if not allowed)
