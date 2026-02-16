# Snapshot Robustness Proposal

## Current Issues

### 1. Silent Missing Snapshots
The `IComputeStage::getDumpInfo()` returns empty `StageDumpInfo{}` by default. If a stage forgets to override this method, it silently produces no snapshot data. This leads to debugging confusion when comparing against PyTorch reference.

**Current Code:**
```cpp
virtual StageDumpInfo getDumpInfo() const { return StageDumpInfo{}; }
```

### 2. Missing Implementations
Several stages don't implement `getDumpInfo()`:
- `AllreduceStage` - No getDumpInfo
- `AllGatherStage` - No getDumpInfo  
- `MoERouterStage`, `MoEExpertStage`, `MoECombineStage` - No getDumpInfo
- `QuantizeToQ16_1Stage` - No getDumpInfo (has declaration but no implementation)
- `KVCacheAppendStage`, `KVCacheGatherStage` - No getDumpInfo

### 3. No Compile-Time Enforcement
Nothing enforces that new stages implement snapshot support. It's easy to add a stage and forget about debugging support.

### 4. No Runtime Validation
When `snapshot_callback` fires, there's no validation that the dump info is actually populated. Silent empty snapshots are passed through.

### 5. Fused Stages Hide Intermediates
Fused stages like `FusedAttentionWoStage` combine multiple logical operations. The intermediate outputs (e.g., attention context before Wo projection) are hidden unless explicitly captured via special buffers like `context_snapshot`.

### 6. Quantized Tensor Handling is Incomplete
`StageDumpInfo::addOutput()` for quantized tensors needs proper dtype handling to dequantize for comparison against FP32 PyTorch reference.

---

## Proposed Solutions

### Solution 1: Make getDumpInfo() Pure Virtual with Opt-Out

Instead of default implementation, make it pure virtual. Stages that truly have nothing to dump can explicitly return empty:

```cpp
class IComputeStage {
public:
    /// @brief Get dump info - MUST be implemented by all stages
    /// Stages with no data to dump should explicitly return StageDumpInfo{}
    virtual StageDumpInfo getDumpInfo() const = 0;
};

// Example: AllreduceStage explicitly opts out
StageDumpInfo AllreduceStage::getDumpInfo() const {
    // MPI collective - buffer is same as input, no separate snapshot needed
    return StageDumpInfo{};
}
```

**Pros:**
- Compile-time enforcement - won't compile until implemented
- Makes snapshot support a first-class requirement

**Cons:**
- Breaking change - all stages need implementation
- Noise in stages that have nothing to dump

### Solution 2: Snapshot Validation Mode

Add runtime validation when `LLAMINAR_SNAPSHOT_VALIDATE=1`:

```cpp
if (success && config_.snapshot_callback) {
    auto dump_info = node.stage->getDumpInfo();
    
    // Validate in debug mode
    if (debugEnv().snapshot.validate_enabled) {
        validateSnapshot(node.name, dump_info);
    }
    
    config_.snapshot_callback(node.name, dump_info);
}

void validateSnapshot(const std::string& node_name, const StageDumpInfo& info) {
    // Warn if stage produces nothing
    if (info.outputs.empty() && info.inputs.empty()) {
        LOG_WARN("[Snapshot] Stage '" << node_name << "' has no snapshot data!");
    }
    
    // Validate output buffers
    for (const auto& output : info.outputs) {
        if (!output.data) {
            LOG_ERROR("[Snapshot] Stage '" << node_name 
                      << "' output '" << output.name << "' has NULL data!");
        }
        if (output.rows == 0 || output.cols == 0) {
            LOG_ERROR("[Snapshot] Stage '" << node_name 
                      << "' output '" << output.name << "' has zero dimensions!");
        }
    }
}
```

### Solution 3: Snapshot Manifest / Registry

Create a static registry of expected snapshots that can be validated:

```cpp
// In IComputeStage.h
struct SnapshotManifest {
    std::vector<std::string> expected_outputs;  // ["C", "attn_output"]
    std::vector<std::string> expected_inputs;   // ["A", "B"]
    bool allows_empty = false;  // True for MPI stages
};

class IComputeStage {
public:
    /// Declare expected snapshot outputs (compile-time documentation)
    virtual SnapshotManifest getSnapshotManifest() const = 0;
};
```

### Solution 4: Typed Snapshot Keys (Enum-based)

Replace string-based snapshot keys with an enum for compile-time safety:

```cpp
enum class SnapshotKey {
    // Embedding
    EMBEDDING,
    
    // Attention path
    ATTENTION_NORM,
    Q_PROJECTION,
    K_PROJECTION,
    V_PROJECTION,
    Q_AFTER_ROPE,
    K_AFTER_ROPE,
    ATTENTION_SCORES,
    ATTENTION_SOFTMAX,
    ATTENTION_CONTEXT,
    ATTENTION_OUTPUT,
    ATTENTION_RESIDUAL,
    
    // FFN path
    FFN_NORM,
    FFN_GATE,
    FFN_UP,
    FFN_SWIGLU,
    FFN_DOWN,
    FFN_RESIDUAL,
    
    // Final
    FINAL_NORM,
    LM_HEAD,
    
    // Custom/Debug
    CUSTOM
};

// Usage in callback
config_.snapshot_callback(SnapshotKey::Q_PROJECTION, layer_idx, dump_info);
```

### Solution 5: Stage-Level Snapshot Configuration

Allow stages to declare their snapshot behavior at construction:

```cpp
struct StageSnapshotConfig {
    bool capture_inputs = true;
    bool capture_outputs = true;
    bool capture_intermediates = false;  // For fused stages
    std::vector<std::string> intermediate_names;  // ["context", "scores"]
};

class IComputeStage {
public:
    virtual StageSnapshotConfig getSnapshotConfig() const {
        return StageSnapshotConfig{};  // Default: capture inputs/outputs
    }
};
```

---

## Recommended Approach: Combination of Solutions

### Phase 1: Immediate (Non-Breaking)
1. **Solution 2**: Add runtime validation via `LLAMINAR_SNAPSHOT_VALIDATE=1`
2. **Implement missing getDumpInfo()**: Add implementations for all stages

### Phase 2: Medium-Term  
3. **Solution 4**: Introduce typed snapshot keys enum
4. **Solution 5**: Add StageSnapshotConfig for fused stage intermediate capture

### Phase 3: Optional (Breaking Change)
5. **Solution 1**: Make getDumpInfo() pure virtual (requires touching all stages)

---

## Implementation Checklist

### Missing getDumpInfo() Implementations
- [x] `AllreduceStage::getDumpInfo()` - Return input buffer (same as output)
- [x] `AllGatherStage::getDumpInfo()` - Return local input and full output
- [x] `MoERouterStage::getDumpInfo()` - Return hidden states, gate weights, router logits
- [x] `MoEExpertStage::getDumpInfo()` - Return expert I/O
- [x] `MoECombineStage::getDumpInfo()` - Return combined output
- [ ] `QuantizeToQ16_1Stage::getDumpInfo()` - SKIPPED: Stage not fully implemented yet
- [x] `KVCacheAppendStage::getDumpInfo()` - Return K/V being appended
- [x] `KVCacheGatherStage::getDumpInfo()` - Return gathered K/V

### Header Declarations Added
- [x] `AllreduceStage.h` - Added `getDumpInfo() const override`
- [x] `AllGatherStage.h` - Added `getDumpInfo() const override`
- [x] `MoEStages.h` - Added `getDumpInfo() const override` for Router, Expert, Combine
- [x] `KVCacheAppendStage.h` - Added `getDumpInfo() const override`
- [x] `KVCacheGatherStage.h` - Added `getDumpInfo() const override`

### Q16 Fused Attention Kernel Snapshot Support
- [x] Added `context_snapshot` parameter to `Q16FusedAttentionWoResidualParams`
- [x] Added INT32→FP32 context snapshot capture in decode path
- [x] Added INT32→FP32 context snapshot capture in prefill path (per-query)
- [x] Added unit test `DecodeContextSnapshotCapture` validating snapshot capture

The Q16 fused attention kernel now supports capturing the attention context (post-softmax, pre-Wo projection) as FP32 for debugging and parity testing. Usage:

```cpp
Q16FusedAttentionWoResidualParams params;
// ... set other params ...

// Allocate snapshot buffer: [seq_len_q × num_heads × head_dim]
std::vector<float> context_snapshot(seq_len_q * num_heads * head_dim);
params.context_snapshot = context_snapshot.data();

q16_fused_attention_wo_residual_reference(params);
// context_snapshot now contains FP32 attention context
```

### New Infrastructure
- [ ] Add `SnapshotValidationConfig` to `DebugEnv`
- [ ] Add validation logic to `DeviceGraphExecutor::executeNode()`
- [ ] Add `SnapshotKey` enum with layer-indexed helpers
- [ ] Add tests for snapshot validation

---

## Example: Complete getDumpInfo() Implementation

```cpp
StageDumpInfo AllGatherStage::getDumpInfo() const {
    StageDumpInfo info;
    
    // Local input slice (what this rank contributes)
    if (params_.local_input) {
        info.addInput("local_input", params_.local_input,
                      params_.actual_seq_len,
                      params_.local_input->cols());
    }
    
    // Full gathered output (replicated on all ranks)
    if (params_.full_output) {
        info.addOutput("full_output", params_.full_output,
                       params_.actual_seq_len,
                       params_.full_output->cols());
    }
    
    // Scalars
    info.addScalarInt("world_size", params_.world_size);
    info.addScalarInt("actual_seq_len", static_cast<int>(params_.actual_seq_len));
    
    return info;
}
```
