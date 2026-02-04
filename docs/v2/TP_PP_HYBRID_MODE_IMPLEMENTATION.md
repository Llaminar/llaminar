# TP_PP Hybrid Mode Implementation Plan

**Date**: February 4, 2026  
**Status**: In Progress  
**Author**: Copilot + David Sanftenberg  
**Effort Estimate**: ~2 hours

## Executive Summary

Enable **Pipeline Parallelism with Tensor Parallel domains** (TP_PP mode) in `MultiDeviceOrchestrator`. This allows topologies like:

```
Stage 0: TP(rocm:0, rocm:1)  ← 2 ROCm GPUs with RCCL allreduce
    ↓ PCIeBAR transfer
Stage 1: cuda:0              ← Single CUDA GPU
```

The implementation leverages existing infrastructure rather than creating new abstraction layers.

---

## Architecture

### Current State

```
MultiDeviceOrchestrator
├── mode_ = TP_PP (detected, but returns false)
├── pp_stage_runners_: vector<unique_ptr<IInferenceRunner>>
│   └── Currently: DeviceGraphOrchestrator only (ignores TP domains)
└── forwardPP() works via IInferenceRunner interface
```

### Target State

```
MultiDeviceOrchestrator (outer, mode=TP_PP)
├── pp_stage_runners_[0]: MultiDeviceOrchestrator (nested, mode=TP)
│   ├── device_runners_[0]: DeviceGraphOrchestrator[rocm:0]
│   ├── device_runners_[1]: DeviceGraphOrchestrator[rocm:1]
│   └── tp_ctx_: LocalTPContext (RCCL)
├── pp_stage_runners_[1]: DeviceGraphOrchestrator[cuda:0]
└── pp_ctx_: LocalPPContext (PCIeBAR transfers)
```

### Key Insight

The existing `forwardPP()` method works through `IInferenceRunner` interface:
- `forward(tokens, seq_len)`
- `getHiddenState()`
- `setHiddenState(hidden_state)`
- `clearHiddenStateInput()`

If `MultiDeviceOrchestrator` implements these methods (delegating to its primary device runner), it can be nested as a PP stage without any new abstraction layer.

---

## Implementation Tasks

### Task 1: PP Interface Methods in MultiDeviceOrchestrator

**File**: `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h/cpp`

Add overrides for the PP hidden state methods:

```cpp
// In header (.h)
TensorBase* getHiddenState() override;
const TensorBase* getHiddenState() const override;
void setHiddenState(TensorBase* hidden_state) override;
bool hasHiddenStateInput() const override;
void clearHiddenStateInput() override;
```

Implementation delegates to primary runner:
- For TP mode: `device_runners_[0]`
- For PP mode: `pp_stage_runners_.back()` (last stage has final hidden state)

**Tests**: Unit tests in `Test__MultiDeviceOrchestrator_PPInterface.cpp`

---

### Task 2: Create Nested MDO for TP Domains

**File**: `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.cpp`

Modify `initializePPDeviceRunners()` to create nested `MultiDeviceOrchestrator` when `stage_config.isTPDomain()`:

```cpp
if (stage_config.isTPDomain())
{
    // Create nested MultiDeviceOrchestrator for this TP stage
    Config nested_config;
    nested_config.mode = ParallelismMode::TP;
    nested_config.devices = stage_config.stage_devices;
    nested_config.weights = stage_config.tp_weights;
    nested_config.backend = stage_config.tp_backend;
    nested_config.max_seq_len = config_.max_seq_len;
    nested_config.batch_size = config_.batch_size;
    nested_config.activation_precision = config_.activation_precision;
    
    auto nested_mdo = std::make_unique<MultiDeviceOrchestrator>(stage_ctx, nested_config);
    pp_stage_runners_.push_back(std::move(nested_mdo));
}
else
{
    // Existing single-device path
    auto runner = createPPStageRunner(...);
    pp_stage_runners_.push_back(std::move(runner));
}
```

**Tests**: Unit tests verifying nested MDO creation

---

### Task 3: Unify forward() Dispatch

**File**: `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.cpp`

Simplify `forward()` to use `forwardPP()` for both PP and TP_PP modes:

```cpp
bool MultiDeviceOrchestrator::forward(const int *tokens, int seq_len)
{
    switch (mode_)
    {
    case ParallelismMode::TP:
        return forwardTP(tokens, seq_len);
    case ParallelismMode::PP:
    case ParallelismMode::TP_PP:  // TP_PP uses same flow as PP
        return forwardPP(tokens, seq_len);
    default:
        LOG_ERROR("MultiDeviceOrchestrator::forward: Unknown parallelism mode");
        return false;
    }
}
```

---

### Task 4: Integration Tests

**File**: `tests/v2/integration/parity/qwen2/Test__Qwen2_ParityMatrix.cpp`

The existing `LocalPP_TP2xROCm_CUDA` test should pass after implementation. Verify:
- Prefill parity
- Decode parity  
- Top-K overlap

---

## Test Plan

### Unit Tests (Task 1)

| Test | Description |
|------|-------------|
| `TPMode_GetHiddenState_DelegatesToPrimaryRunner` | Verify delegation in TP mode |
| `PPMode_GetHiddenState_DelegatesToLastStage` | Verify delegation in PP mode |
| `SetHiddenState_PropagatestoRunner` | Verify set/clear cycle |
| `HasHiddenStateInput_ReflectsRunnerState` | Verify state tracking |

### Unit Tests (Task 2)

| Test | Description |
|------|-------------|
| `InitializePP_TPDomain_CreatesNestedMDO` | Verify nested MDO creation |
| `InitializePP_TPDomain_ConfiguresTPBackend` | Verify backend propagation |
| `InitializePP_MixedStages_CreatesCorrectRunnerTypes` | Stage 0 = MDO, Stage 1 = DGO |

### Integration Tests (Task 4)

| Test | Description |
|------|-------------|
| `LocalPP_TP2xROCm_CUDA/PrefillParity` | Full prefill with hybrid topology |
| `LocalPP_TP2xROCm_CUDA/DecodeParity` | Autoregressive decode |
| `LocalPP_TP2xROCm_CUDA/TopKOverlap` | Token prediction quality |

---

## File Changes Summary

| File | Change Type | Lines |
|------|-------------|-------|
| `MultiDeviceOrchestrator.h` | Modify | +15 |
| `MultiDeviceOrchestrator.cpp` | Modify | +80 |
| `Test__MultiDeviceOrchestrator_PPInterface.cpp` | New | ~150 |
| `Test__MultiDeviceOrchestrator_NestedTP.cpp` | New | ~200 |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Hidden state tensor ownership | Use existing coherence patterns; don't transfer ownership |
| Nested MDO model context sharing | Create stage-specific ModelContext as already done |
| PP context with heterogeneous stages | LocalPPContext already handles device-to-device transfers |

---

## Success Criteria

1. ✅ `v2_unit_multi_device_orchestrator_pp_interface` passes (new tests)
2. ✅ `v2_unit_multi_device_orchestrator_nested_tp` passes (new tests)
3. ✅ `LocalPP_TP2xROCm_CUDA/PrefillParity` passes
4. ✅ `LocalPP_TP2xROCm_CUDA/DecodeParity` passes
5. ✅ `LocalPP_TP2xROCm_CUDA/TopKOverlap` passes
6. ✅ All existing `MultiDeviceOrchestrator` tests still pass

---

## Implementation Order

1. **Task 1**: PP Interface Methods (enables nesting)
2. **Task 2**: Nested MDO Creation (creates the topology)
3. **Task 3**: Forward Dispatch (removes TODO)
4. **Task 4**: Integration Test Verification
