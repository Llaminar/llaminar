# Tensor-Parallel Snapshot Parity System Design

**Date:** 2025-01-27  
**Author:** Copilot  
**Status:** Design Phase

## Problem Statement

The current snapshot/parity testing system was designed before tensor parallelism (TP) was implemented. It does not properly handle:

1. **Per-device snapshots**: Each device in a TP scenario computes partial results
2. **Shard combination**: Column-parallel stages need concatenation; row-parallel need validation
3. **Proper comparison**: Sharded outputs must be compared against correct slices of PyTorch reference

Currently, `MultiDeviceOrchestrator::getSnapshot()` only returns device 0's data (except for LM_HEAD), and `compareTensors()` silently returns cosine=0.0 on size mismatches. This makes parity results unreliable.

## Requirements

### 1. Per-Device Snapshot Storage

Each device must dump its snapshots with a globally unique identifier:
```
{hostname}_{rank}_{devicetype}{ordinal}_{stage}_{key}.bin
```

Example:
```
localhost_rank0_cuda0_layer0_ATTENTION_CONTEXT.bin
localhost_rank0_rocm0_layer0_ATTENTION_CONTEXT.bin
```

### 2. Sharding Pattern Awareness

The parity system must understand which stages have sharded outputs:

| Stage Type | Sharding Mode | TP Behavior |
|------------|---------------|-------------|
| EMBEDDING | Replicated | Full output on each device |
| Q/K/V_PROJECTION | Column-parallel | Output split on heads dimension |
| ATTENTION_CONTEXT | Column-parallel | Output split on heads dimension |
| ATTENTION_OUTPUT (Wo) | Row-parallel | Full output after AllReduce |
| FFN_GATE, FFN_UP | Column-parallel | Output split on d_ff dimension |
| FFN_SWIGLU | Column-parallel | Output split on d_ff dimension |
| FFN_DOWN | Row-parallel | Full output after AllReduce |
| FFN_RESIDUAL | Replicated | Full output after AllReduce |
| *_NORM stages | Replicated | Full output |
| LM_HEAD | Column-parallel | Output split on vocab, then AllGather |

### 3. Result Display

The parity table should show:

1. **Per-device breakdown**: Each device's partial result vs correct PyTorch slice
2. **Combined layer parity**: All TP ranks combined vs full PyTorch reference
3. **Final combined result**: Full logits comparison

## Architecture

### Component Changes

```
┌─────────────────────────────────────────────────────────────────┐
│                     MultiDeviceOrchestrator                      │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ DeviceRunner[0]  │  │ DeviceRunner[1]  │  (LOCAL TP)         │
│  │   CUDA:0         │  │   ROCm:0         │                     │
│  │ snapshots_[0]    │  │ snapshots_[1]    │                     │
│  └────────┬─────────┘  └────────┬─────────┘                     │
│           │                     │                                │
│           ▼                     ▼                                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │           NEW: getTPSnapshot() API                    │       │
│  │  Returns: map<GlobalDeviceId, SnapshotData>          │       │
│  │  - Per-device data for column-parallel stages        │       │
│  │  - Combined data for row-parallel stages             │       │
│  │  - Metadata about sharding pattern                   │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ParityTestBase                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │           NEW: TPAwareComparison                      │       │
│  │  1. Detect stage sharding pattern from metadata       │       │
│  │  2. For column-parallel: compare each device slice    │       │
│  │  3. Combine slices and compare against full PyTorch   │       │
│  │  4. Report per-device AND combined metrics            │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### New Data Structures

```cpp
// In MultiDeviceOrchestrator.h or a new TPSnapshot.h

/**
 * @brief Sharding pattern for a snapshot
 */
enum class SnapshotShardingMode {
    REPLICATED,      // Full output on each device (norms, residuals after AllReduce)
    COLUMN_PARALLEL, // Split on output dimension (Q/K/V, FFN_GATE, FFN_UP)
    ROW_PARALLEL,    // Split on input dimension, combined after AllReduce
    GATHERED         // Like LM_HEAD - column-parallel then gathered
};

/**
 * @brief Per-device snapshot data
 */
struct DeviceSnapshotData {
    GlobalDeviceId device_id;           ///< Unique device identifier
    std::vector<float> data;            ///< Tensor data (may be partial)
    size_t rows = 0;                    ///< Logical rows
    size_t cols = 0;                    ///< Logical cols (may be partial for column-parallel)
    size_t global_start_col = 0;        ///< For column-parallel: which col this shard starts at
    size_t global_total_cols = 0;       ///< Full column count across all devices
};

/**
 * @brief Complete TP-aware snapshot for a stage
 */
struct TPSnapshot {
    std::string key;                              ///< Stage key (e.g., "layer0_ATTENTION_CONTEXT")
    SnapshotShardingMode mode;                    ///< How the output is sharded
    std::vector<DeviceSnapshotData> device_data;  ///< Per-device snapshots
    
    // Combined view (computed lazily)
    bool combined_valid = false;
    std::vector<float> combined_data;             ///< Concatenated/verified combined result
    size_t combined_rows = 0;
    size_t combined_cols = 0;
};
```

### API Changes

```cpp
// IInferenceRunner interface additions
class IInferenceRunner {
public:
    // Existing
    virtual const float* getSnapshot(const std::string& key, size_t& size) const = 0;
    
    // NEW: TP-aware snapshot retrieval
    virtual TPSnapshot getTPSnapshot(const std::string& key) const = 0;
    
    // NEW: Get all snapshot keys with their sharding metadata
    virtual std::vector<std::pair<std::string, SnapshotShardingMode>> 
        getSnapshotKeysWithSharding() const = 0;
};
```

## Implementation Plan

### Phase 1: GlobalDeviceId Enhancement

1. Implement `GlobalDeviceId::toString()` with format:
   ```
   {hostname}_rank{N}_{type}{ordinal}
   ```
   Example: `localhost_rank0_cuda0`, `node01_rank2_rocm1`

2. Add hostname detection to GlobalDeviceId

### Phase 2: Snapshot Capture Enhancement

1. Modify `DeviceGraphOrchestrator` to include `GlobalDeviceId` in snapshot storage
2. Add sharding mode metadata to each captured snapshot
3. Update `MultiDeviceOrchestrator::getSnapshot()` to return TPSnapshot

### Phase 3: ParityTestBase Updates

1. Add `compareTPSnapshot()` method that:
   - Loads PyTorch reference for the stage
   - For column-parallel stages:
     - Compare each device's shard to corresponding PyTorch slice
     - Concatenate all shards and compare to full PyTorch
   - For row-parallel/replicated stages:
     - Verify all devices have same data
     - Compare to full PyTorch
   
2. Update `renderParityTable()` to show:
   - Per-device columns in TP mode
   - Combined result column
   - Sharding mode indicator

### Phase 4: Stage Metadata

1. Define static mapping from stage type to sharding mode
2. Alternative: store sharding mode when snapshot is captured

## Stage Sharding Mode Registry

```cpp
// Static registry of stage sharding modes
static const std::unordered_map<std::string, SnapshotShardingMode> kStageShardingModes = {
    // Embedding - replicated across devices
    {"EMBEDDING", SnapshotShardingMode::REPLICATED},
    
    // Attention projections - column-parallel (split on heads)
    {"Q_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
    {"K_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
    {"V_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
    {"QKV_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
    
    // Attention context - column-parallel (split on heads)
    {"ATTENTION_CONTEXT", SnapshotShardingMode::COLUMN_PARALLEL},
    
    // Attention output (Wo) - row-parallel, combined after AllReduce
    {"ATTENTION_OUTPUT", SnapshotShardingMode::ROW_PARALLEL},
    
    // Attention norms - replicated
    {"ATTENTION_NORM", SnapshotShardingMode::REPLICATED},
    
    // FFN projections - column-parallel
    {"FFN_GATE", SnapshotShardingMode::COLUMN_PARALLEL},
    {"FFN_UP", SnapshotShardingMode::COLUMN_PARALLEL},
    {"FFN_SWIGLU", SnapshotShardingMode::COLUMN_PARALLEL},
    
    // FFN down - row-parallel
    {"FFN_DOWN", SnapshotShardingMode::ROW_PARALLEL},
    
    // FFN output/residual - replicated (after AllReduce)
    {"FFN_RESIDUAL", SnapshotShardingMode::REPLICATED},
    {"FFN_NORM", SnapshotShardingMode::REPLICATED},
    
    // Final stages
    {"FINAL_NORM", SnapshotShardingMode::REPLICATED},
    {"LM_HEAD", SnapshotShardingMode::GATHERED},
};
```

## PyTorch Slice Computation

For column-parallel stages, need to compute which slice of PyTorch output maps to which device:

```cpp
// For column-parallel with TP degree=2:
// Device 0: cols [0, total_cols/2)
// Device 1: cols [total_cols/2, total_cols)

size_t computeSliceStart(int device_idx, int tp_degree, size_t total_cols) {
    return device_idx * (total_cols / tp_degree);
}

size_t computeSliceSize(int device_idx, int tp_degree, size_t total_cols) {
    size_t base = total_cols / tp_degree;
    // Last device gets remainder
    if (device_idx == tp_degree - 1) {
        return total_cols - device_idx * base;
    }
    return base;
}
```

## Result Table Format

### Per-Layer TP Breakdown (NEW)
```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                       PCIeBAR (CUDA↔ROCm) vs PyTorch LAYER-BY-LAYER PARITY                        ║
║                           (2-way LOCAL TP, Threshold: avg cosine >= 0.940)                        ║
╠═══════════╦═══════════════════════════════════════════════════════════════════════╦══════════════╣
║   Layer   ║                    Per-Device Cosine Similarity                       ║   Combined   ║
║           ╠═════════════════════════════════╦═════════════════════════════════════╣     Cosine   ║
║           ║    cuda:0 vs PyTorch[0:N/2]     ║    rocm:0 vs PyTorch[N/2:N]        ║              ║
╠═══════════╬═════════════════════════════════╬═════════════════════════════════════╬══════════════╣
║ EMBEDDING ║           0.999912              ║           0.999912 (replicated)     ║    0.999912  ║
╠═══════════╬═════════════════════════════════╬═════════════════════════════════════╬══════════════╣
║  Layer 0  ║                                 ║                                     ║              ║
║  ATTN_CTX ║           0.987234              ║           0.986892                  ║    0.987063  ║
║  ATTN_OUT ║           0.992111              ║           0.992111 (row-parallel)   ║    0.992111  ║
║  FFN_GATE ║           0.983456              ║           0.984123                  ║    0.983790  ║
║  FFN_DOWN ║           0.991234              ║           0.991234 (row-parallel)   ║    0.991234  ║
╠═══════════╬═════════════════════════════════╬═════════════════════════════════════╬══════════════╣
...
╠═══════════╬═════════════════════════════════╬═════════════════════════════════════╬══════════════╣
║  LM_HEAD  ║      [gathered across TP]       ║                                     ║    0.995678  ║
╚═══════════╩═════════════════════════════════╩═════════════════════════════════════╩══════════════╝
```

## Implementation Order

1. **Implement `GlobalDeviceId::toString()`** - Required for device identification
2. **Add `SnapshotShardingMode` enum and registry** - Define stage sharding patterns
3. **Modify `DeviceGraphOrchestrator` snapshot capture** - Include device ID and sharding mode
4. **Add `TPSnapshot` struct and `getTPSnapshot()` API** - New retrieval interface
5. **Update `MultiDeviceOrchestrator` to populate `TPSnapshot`** - Combine device data
6. **Add `compareTPSnapshot()` to ParityTestBase** - TP-aware comparison logic
7. **Update `renderParityTable()`** - Show per-device and combined results
8. **Update PCIeBAR parity test** - Use new TP-aware comparison

## Files to Modify

| File | Changes |
|------|---------|
| `src/v2/execution/DeviceInventory.h` | Add `toString()` implementation, add hostname |
| `src/v2/execution/TPSnapshot.h` (NEW) | `SnapshotShardingMode`, `DeviceSnapshotData`, `TPSnapshot` |
| `src/v2/execution/DeviceGraphOrchestrator.h` | Store `GlobalDeviceId` with snapshots |
| `src/v2/execution/MultiDeviceOrchestrator.h/cpp` | Implement `getTPSnapshot()` |
| `src/v2/execution/IInferenceRunner.h` | Add `getTPSnapshot()` to interface |
| `tests/v2/integration/parity/ParityTestBase.h` | Add `compareTPSnapshot()`, update table |
| `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp` | Use new API |

## Testing Plan

1. **Unit test for `GlobalDeviceId::toString()`** - Verify format
2. **Unit test for slice computation** - Verify correct PyTorch slices
3. **Integration test with column-parallel stage** - Verify device slices concatenate correctly
4. **Full parity test** - Verify combined results match PyTorch

## Success Criteria

1. Per-device snapshots are captured with unique identifiers
2. Column-parallel stages show per-device cosine similarity
3. Combined results match PyTorch with expected tolerance
4. Parity table clearly shows TP breakdown
5. No silent failures (size mismatches must be reported)
