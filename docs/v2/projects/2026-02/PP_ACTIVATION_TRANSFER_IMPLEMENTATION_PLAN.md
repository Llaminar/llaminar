# Pipeline Parallelism Activation Transfer Implementation Plan

## Executive Summary

This document outlines the implementation plan for enabling **real end-to-end Pipeline Parallelism (PP)** in Llaminar V2 by implementing activation transfer between PP stages. The core problem is that stages > 0 need to receive hidden state activations from previous stages, but currently there's no API to get/set these activations.

**Current Status:**
- ✅ `LocalPPOrchestrator` exists and creates per-stage runners
- ✅ `ILocalPPContext` has `transfer()` / `transferAsync()` APIs for data movement  
- ✅ `LocalPPContext` has backend selection logic (NCCL, RCCL, PCIeBAR, HOST)
- ✅ `DeviceGraphOrchestrator` builds partial graphs for PP stages
- 🔴 Stage >0 receives `nullptr` for tokens but has no activation input
- 🔴 `transferActivations()` is a placeholder that does nothing
- 🔴 No API to extract hidden state from stage N-1 or inject into stage N

**Goal:** Enable PP stages to pass activations between each other using the same collective backends already used for Tensor Parallelism (NCCL, RCCL, PCIeBAR, HOST).

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         LocalPPOrchestrator                                │
│                                                                           │
│  ┌────────────────┐   transferActivations()   ┌────────────────┐          │
│  │    Stage 0     │  ───────────────────────→ │    Stage 1     │          │
│  │   [cuda:0]     │   via ILocalPPContext     │   [cuda:1]     │          │
│  │ embed+L0-13    │   (NCCL P2P send/recv)    │   L14-27       │          │
│  │                │                           │                │          │
│  │ Output:        │                           │ Input:         │          │
│  │  hidden_state  │ ─────────────────────────→│  hidden_state  │          │
│  │  [seq, d_model]│                           │  [seq, d_model]│          │
│  └────────────────┘                           └────────────────┘          │
│         ▼                                             ▼                   │
│  getHiddenState()                             setHiddenState()            │
│  (new API on                                  (bypass embedding,          │
│   IInferenceRunner)                            use activations)           │
└───────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The hidden state is stored in `InferenceState::hidden` after each forward pass. We need APIs to:
1. **Extract** this hidden state from the source stage
2. **Transfer** it via `ILocalPPContext::transfer()` (already implemented!)
3. **Inject** it into the destination stage as input (bypass embedding)

---

## Implementation Phases

### Phase 1: Hidden State Access API

**Files to modify:**
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

**Changes:**

#### 1.1 Add to `IInferenceRunner.h`:

```cpp
// After the snapshot API section:

// =====================================================================
// Hidden State API (for Pipeline Parallelism)
// =====================================================================

/**
 * @brief Get final hidden state from last forward pass
 *
 * Returns the hidden state tensor after all transformer layers have
 * executed. This is used for Pipeline Parallelism to transfer
 * activations between stages.
 *
 * @return Pointer to hidden state tensor [seq_len, d_model], or nullptr
 */
virtual TensorBase* getHiddenState() { return nullptr; }
virtual const TensorBase* getHiddenState() const { return nullptr; }

/**
 * @brief Set initial hidden state for forward pass
 *
 * For PP stages that don't have embedding (middle/final stages),
 * this sets the hidden state that would normally come from embedding.
 * The forward pass will skip embedding and use this tensor directly.
 *
 * @param hidden_state Tensor containing hidden state [seq_len, d_model]
 */
virtual void setHiddenState(TensorBase* hidden_state) { (void)hidden_state; }

/**
 * @brief Check if this runner has hidden state set for next forward
 */
virtual bool hasHiddenStateInput() const { return false; }

/**
 * @brief Clear hidden state input (reset to normal embedding mode)
 */
virtual void clearHiddenStateInput() {}
```

#### 1.2 Implement in `DeviceGraphOrchestrator.h`:

Add member variable:
```cpp
/// External hidden state input for PP middle/final stages
TensorBase* external_hidden_state_input_ = nullptr;
```

Add method declarations:
```cpp
TensorBase* getHiddenState() override;
const TensorBase* getHiddenState() const override;
void setHiddenState(TensorBase* hidden_state) override;
bool hasHiddenStateInput() const override;
void clearHiddenStateInput() override;
```

#### 1.3 Implement in `DeviceGraphOrchestrator.cpp`:

```cpp
TensorBase* DeviceGraphOrchestrator::getHiddenState()
{
    if (!inference_state_.hidden)
    {
        LOG_WARN("[DeviceGraphOrchestrator] getHiddenState: no hidden state available");
        return nullptr;
    }
    return inference_state_.hidden.get();
}

const TensorBase* DeviceGraphOrchestrator::getHiddenState() const
{
    return inference_state_.hidden.get();
}

void DeviceGraphOrchestrator::setHiddenState(TensorBase* hidden_state)
{
    external_hidden_state_input_ = hidden_state;
    LOG_DEBUG("[DeviceGraphOrchestrator] setHiddenState: "
              << (hidden_state ? "set" : "cleared")
              << " external hidden state input");
}

bool DeviceGraphOrchestrator::hasHiddenStateInput() const
{
    return external_hidden_state_input_ != nullptr;
}

void DeviceGraphOrchestrator::clearHiddenStateInput()
{
    external_hidden_state_input_ = nullptr;
}
```

---

### Phase 2: Modify Forward Graph Building for Activation Input

**Files to modify:**
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/models/qwen/Qwen2Graph.h`
- `src/v2/models/qwen/Qwen2Graph.cpp`

**Changes:**

#### 2.1 Update `Qwen2ForwardInput` in `Qwen2Graph.h`:

Add activation input field:
```cpp
struct Qwen2ForwardInput
{
    // ... existing fields ...
    
    /// External hidden state input (for PP middle stages that don't have embedding)
    /// When set, embedding is skipped and this tensor is used as initial hidden state.
    TensorBase* external_hidden_state = nullptr;
};
```

#### 2.2 Update `buildPartialForwardGraph` in `Qwen2Graph.cpp`:

Change the embedding section to handle external hidden state:

```cpp
// Stage 1: Embedding Lookup (optional - only for first PP stage)
if (has_embedding)
{
    // Existing embedding code...
}
else if (input.external_hidden_state)
{
    // PP middle/final stage: Use external hidden state as starting point
    //
    // The external_hidden_state tensor contains activations from the previous
    // PP stage. We need to copy it to our working buffer if they're different.
    //
    // For HybridQ16 mode: external hidden is Q16_1, copy to residual buffer
    // For FP32 mode: external hidden is FP32, copy to current_hidden buffer
    
    InferenceMode full_pass_inference_mode(config_.activation_precision);
    TensorBase* working_buffer = buffers_.layer_buffers.residual &&
                                 full_pass_inference_mode.isHybridQ16()
                                     ? buffers_.layer_buffers.residual
                                     : buffers_.current_hidden;
    
    // Add a copy stage if external buffer differs from working buffer
    if (input.external_hidden_state != working_buffer)
    {
        // Use TensorCopyStage to copy external → working
        TensorCopyStage::Params copy_params{
            .device_id = config_.default_device,
            .src = input.external_hidden_state,
            .dst = working_buffer,
            .count = static_cast<size_t>(total_tokens * config_.d_model)
        };
        graph.addNode("pp_activation_inject",
                      ComputeStageFactory::createTensorCopy(copy_params),
                      device);
        prev_node = "pp_activation_inject";
    }
    // else: external_hidden_state IS our working buffer, no copy needed
}
else if (!has_embedding)
{
    LOG_ERROR("[Qwen2Graph] PP stage without embedding requires external hidden state");
    throw std::runtime_error("PP stage requires external hidden state input");
}
```

#### 2.3 Update `executeForward` in `DeviceGraphOrchestrator.cpp`:

Modify the input validation to accept activation input for PP stages:

```cpp
bool DeviceGraphOrchestrator::executeForward(
    const Qwen2ForwardInput &input,
    Qwen2ForwardOutput &output)
{
    ScopedDeviceLog device_log(input.device);

    // Token input OR activation input is required
    bool has_token_input = input.token_ids || input.batches;
    bool has_activation_input = external_hidden_state_input_ != nullptr;
    
    // For PP stages without embedding, activation input is required
    bool needs_activation_input = pp_stage_config_.has_value() && 
                                  !pp_stage_config_.value().has_embedding;
    
    if (!has_token_input && !has_activation_input)
    {
        LOG_ERROR("[DeviceGraphOrchestrator] No token or activation input provided");
        return false;
    }
    
    if (needs_activation_input && !has_activation_input)
    {
        LOG_ERROR("[DeviceGraphOrchestrator] PP stage without embedding requires "
                  "activation input via setHiddenState()");
        return false;
    }

    // ... rest of the method ...
    
    // Pass external hidden state to input
    Qwen2ForwardInput effective_input = input;
    effective_input.external_hidden_state = external_hidden_state_input_;
    
    // Clear for next invocation (single-use)
    external_hidden_state_input_ = nullptr;
    
    // ... rest of graph building and execution ...
}
```

---

### Phase 3: Implement LocalPPOrchestrator Activation Transfer

**Files to modify:**
- `src/v2/execution/local_execution/pp/LocalPPOrchestrator.cpp`

**Changes:**

#### 3.1 Update `transferActivations()`:

```cpp
bool LocalPPOrchestrator::transferActivations(int from_stage, int to_stage)
{
    if (!config_.pp_context)
    {
        LOG_ERROR("[LocalPPOrchestrator] No PP context available for transfer");
        return false;
    }

    // Get hidden state from source stage
    TensorBase* hidden_state = stage_runners_[from_stage]->getHiddenState();
    if (!hidden_state)
    {
        LOG_ERROR("[LocalPPOrchestrator] Stage " << from_stage 
                  << " has no hidden state to transfer");
        return false;
    }

    LOG_DEBUG("[LocalPPOrchestrator] Transferring activations: stage " 
              << from_stage << " -> " << to_stage
              << " (" << hidden_state->numel() << " elements)");

    // Transfer via ILocalPPContext (uses NCCL/RCCL/PCIeBAR/HOST as appropriate)
    bool success = config_.pp_context->transfer(hidden_state, from_stage, to_stage);
    if (!success)
    {
        LOG_ERROR("[LocalPPOrchestrator] Activation transfer failed");
        return false;
    }

    // Set the hidden state as input for the destination stage
    stage_runners_[to_stage]->setHiddenState(hidden_state);

    LOG_DEBUG("[LocalPPOrchestrator] Activation transfer complete");
    return true;
}
```

#### 3.2 Update `executeStage()`:

```cpp
bool LocalPPOrchestrator::executeStage(int stage, const int *tokens, int seq_len)
{
    if (stage < 0 || stage >= static_cast<int>(stage_runners_.size()))
    {
        LOG_ERROR("[LocalPPOrchestrator] Invalid stage index: " << stage);
        return false;
    }

    auto &runner = stage_runners_[stage];

    // Stage 0 uses token IDs for embedding lookup
    if (stage == 0)
    {
        LOG_DEBUG("[LocalPPOrchestrator] Executing stage 0 with token input");
        return runner->forward(tokens, seq_len);
    }

    // Subsequent stages should have activations set via setHiddenState()
    // from a prior transferActivations() call
    if (!runner->hasHiddenStateInput())
    {
        LOG_ERROR("[LocalPPOrchestrator] Stage " << stage 
                  << " has no activation input set. "
                  << "Call transferActivations() before executeStage()");
        return false;
    }

    LOG_DEBUG("[LocalPPOrchestrator] Executing stage " << stage 
              << " with activation input");
    
    // Forward with nullptr tokens - runner will use external hidden state
    bool success = runner->forward(nullptr, seq_len);
    
    // Clear hidden state input after use (single-use semantics)
    runner->clearHiddenStateInput();
    
    return success;
}
```

#### 3.3 Update `forward()` method to wire it all together:

```cpp
bool LocalPPOrchestrator::forward(const int *tokens, int seq_len)
{
    LOG_INFO("[LocalPPOrchestrator] Forward pass: seq_len=" << seq_len
             << ", stages=" << numStages());

    // Execute each stage in sequence with activation transfers between
    for (int stage = 0; stage < numStages(); ++stage)
    {
        // Transfer activations from previous stage (except for stage 0)
        if (stage > 0)
        {
            if (!transferActivations(stage - 1, stage))
            {
                LOG_ERROR("[LocalPPOrchestrator] Activation transfer failed: "
                          << (stage - 1) << " -> " << stage);
                return false;
            }
        }

        // Execute stage
        if (!executeStage(stage, stage == 0 ? tokens : nullptr, seq_len))
        {
            LOG_ERROR("[LocalPPOrchestrator] Stage " << stage << " execution failed");
            return false;
        }
    }

    LOG_DEBUG("[LocalPPOrchestrator] All stages completed successfully");
    return true;
}
```

---

### Phase 4: Activation Buffer Management

**Problem:** The activation transfer uses the hidden state buffer from stage N-1. But where does stage N receive it?

**Options:**

1. **Shared Buffer (Same Device):** If stages are on the same device, they can share the same hidden buffer.

2. **Separate Buffers (Different Devices):** Each stage has its own hidden buffer. Transfer copies src → dst buffer.

3. **Double Buffering (Async):** For overlapping compute and transfer, use two buffers per stage.

**Recommended Approach:** Use `activation_buffer_` in LocalPPOrchestrator as a ping-pong buffer:

```cpp
// In LocalPPOrchestrator.h, update members:

/// Primary activation buffer for inter-stage transfer
/// Size: [max_seq_len * max_batch_size, d_model]
std::unique_ptr<TensorBase> activation_buffer_;

/// Secondary buffer for double-buffering (future async transfers)
std::unique_ptr<TensorBase> activation_buffer_alt_;

/// Current buffer index (0 or 1) for ping-pong
int current_activation_buffer_ = 0;
```

The `transferActivations()` implementation then becomes:

```cpp
bool LocalPPOrchestrator::transferActivations(int from_stage, int to_stage)
{
    // Get source hidden state
    TensorBase* src_hidden = stage_runners_[from_stage]->getHiddenState();
    if (!src_hidden)
    {
        LOG_ERROR("[LocalPPOrchestrator] No hidden state from stage " << from_stage);
        return false;
    }

    // Destination buffer - use our activation_buffer_ which is allocated
    // on the destination device during initializeStages()
    TensorBase* dst_buffer = activation_buffer_.get();
    
    // Ensure destination buffer exists and is sized correctly
    if (!dst_buffer || dst_buffer->numel() < src_hidden->numel())
    {
        LOG_ERROR("[LocalPPOrchestrator] Activation buffer too small or missing");
        return false;
    }

    // Use ILocalPPContext::transfer() to move data
    // This handles NCCL P2P, PCIeBAR, or host memcpy as appropriate
    bool success = config_.pp_context->transfer(src_hidden, from_stage, to_stage);
    
    if (!success)
    {
        return false;
    }

    // Now the data is in activation_buffer_ on destination device.
    // Set it as hidden state input for the destination stage.
    //
    // Note: The transfer() should have updated src_hidden to be on dst device,
    // OR we need to use dst_buffer explicitly. Check LocalPPContext semantics.
    stage_runners_[to_stage]->setHiddenState(dst_buffer);

    return true;
}
```

---

### Phase 5: Snapshot Support for PP Parity Tests

**Files to modify:**
- `src/v2/execution/local_execution/pp/LocalPPOrchestrator.cpp`

**Key Insight:** Snapshots are already captured per-stage. The `LocalPPOrchestrator` aggregates them via `getSnapshotKeys()` and `getSnapshot()`.

The current implementation already handles this (see lines 230-295 in `LocalPPOrchestrator.h`):

```cpp
const float *getSnapshot(const std::string &key, size_t &out_size) const override;
std::vector<std::string> getSnapshotKeys() const override;
```

**Issue:** The `getSnapshot()` implementation needs to search across all stage runners to find the correct snapshot based on layer index in the key.

**Verification:** Run parity tests after implementing Phases 1-4 and verify snapshots are properly collected.

---

## Implementation Order

| Phase | Priority | Estimated Effort | Dependencies |
|-------|----------|------------------|--------------|
| Phase 1 | HIGH | 2-3 hours | None |
| Phase 2 | HIGH | 3-4 hours | Phase 1 |
| Phase 3 | HIGH | 2-3 hours | Phase 1, Phase 2 |
| Phase 4 | MEDIUM | 2-3 hours | Phase 3 |
| Phase 5 | LOW | 1-2 hours | Phase 4 |

**Total Estimated Effort:** 10-15 hours

---

## Testing Plan

### Unit Tests

1. **Hidden State API Tests:**
   - `Test__DeviceGraphOrchestrator_GetHiddenState`
   - `Test__DeviceGraphOrchestrator_SetHiddenState`

2. **Partial Graph Building Tests:**
   - `Test__Qwen2Graph_PartialForward_WithActivationInput`
   - `Test__Qwen2Graph_PartialForward_MiddleStage`

### Integration Tests

1. **PP Orchestrator Tests:**
   - `V2_Integration_LocalPP_ActivationTransfer_SameDevice`
   - `V2_Integration_LocalPP_ActivationTransfer_NCCL`
   - `V2_Integration_LocalPP_ActivationTransfer_PCIeBAR`

2. **PP Parity Tests (existing, should pass after implementation):**
   - `V2_Integration_Parity_LocalPP_PrefillParity_*` (6 configs)
   - `V2_Integration_Parity_LocalPP_DecodeParity_*` (6 configs)
   - `V2_Integration_Parity_LocalPP_SnapshotInfrastructure_*` (6 configs) - **Currently failing**

### End-to-End Tests

1. **Full PP Inference:**
   - Run `llaminar2 -m model.gguf -pp 2` with 2 GPUs
   - Compare output to single-GPU baseline

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Buffer allocation on wrong device | Medium | High | Add explicit device checks in `ensureOnDevice()` |
| Hidden state tensor format mismatch (Q16 vs FP32) | Medium | High | Use InferenceMode to determine correct format |
| Async transfer race conditions | Low | High | Start with sync transfers, add async later |
| Memory pressure from double buffering | Low | Medium | Make double buffer optional |

---

## Success Criteria

1. ✅ All 25 LocalPP parity tests pass (including 6 currently failing SnapshotInfrastructure tests)
2. ✅ `llaminar2 --pp 2` produces identical output to `--pp 1` for same model
3. ✅ Activation transfer uses NCCL/RCCL when both stages are on same GPU vendor
4. ✅ Activation transfer uses PCIeBAR for heterogeneous setups
5. ✅ No memory leaks or double-frees with ASAN enabled

---

## Future Enhancements

1. **Async Pipeline:** Overlap stage N compute with stage N-1 → N transfer
2. **Micro-batching:** Split batch across pipeline for better utilization
3. **1F1B Schedule:** Interleaved forward/backward for training
4. **Gradient Checkpointing:** Memory optimization for training

---

## References

- [LocalPPOrchestrator.h](src/v2/execution/local_execution/pp/LocalPPOrchestrator.h) - Current PP orchestrator
- [ILocalPPContext.h](../../../../src/v2/collective/ILocalPPContext.h) - Transfer API interface
- [LocalPPContext.cpp](../../../../src/v2/collective/LocalPPContext.cpp) - Backend selection logic
- [DeviceGraphOrchestrator.cpp](../../../../src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp) - Graph execution
- [Qwen2Graph.cpp](src/v2/models/qwen/Qwen2Graph.cpp) - Graph building
