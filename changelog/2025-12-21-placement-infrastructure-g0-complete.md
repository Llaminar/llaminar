# Placement Infrastructure (Phase G0) - Complete

**Date**: December 21, 2025  
**Author**: David Sanftenberg

## Summary

Implemented the complete placement infrastructure skeleton (Phase G0) as a foundation for all future GPU integration work. This provides the framework for computing device placement decisions and applying them to weight loading.

## Motivation

The placement infrastructure was identified as a prerequisite for reliable implementation and testing of distributed execution features. As noted: "the full skeleton needs to be in place for us to reliably implement/test the rest."

## Implementation

### New Files Created

| File | Purpose |
|------|---------|
| `src/v2/execution/PlacementPlan.h` | Data structures for placement decisions |
| `src/v2/execution/PlacementPlan.cpp` | `toString()` implementation for debugging |
| `src/v2/execution/PlacementStrategy.h` | Strategy interface and factory |
| `src/v2/execution/PlacementStrategy.cpp` | Strategy implementations |
| `tests/v2/unit/Test__PlacementStrategy.cpp` | 34 unit tests |

### Files Modified

| File | Changes |
|------|---------|
| `src/v2/loaders/WeightPlacementMap.h` | Added `applyPlan()`, `hasPlan()`, `appliedStrategyName()` |
| `src/v2/loaders/WeightPlacementMap.cpp` | Implemented `applyPlan()` |
| `src/v2/utils/MPITopology.h` | Added `computePlacement()` methods |
| `src/v2/utils/MPITopology.cpp` | Implemented `computePlacement()` |
| `src/v2/CMakeLists.txt` | Added new source files |
| `tests/v2/CMakeLists.txt` | Added test target |

### Key Types

```cpp
// PlacementDevice - Where computation runs
enum class PlacementDevice : uint8_t {
    CPU = 0,        // Primary CPU/socket
    GPU_0 = 1,      // First GPU
    GPU_1, GPU_2, GPU_3,
    GPU_ANY = 254,  // Let system choose GPU
    REPLICATED = 255 // On all devices (norms, embeddings)
};

// LayerPlacement - Per-layer device assignment
struct LayerPlacement {
    int layer_idx;
    int owner_rank;        // For tensor parallelism
    PlacementDevice device;
    bool split_attention_ffn;  // Different devices for attention vs FFN
    PlacementDevice attention_device;
    PlacementDevice ffn_device;
};

// PlacementPlan - Complete model placement
struct PlacementPlan {
    std::string model_architecture;
    int n_layers;
    int world_size;
    std::vector<LayerPlacement> layer_placements;
    GlobalPlacement global_placement;
    std::string strategy_name;
    // Helper methods: isValid(), usesGPU(), getLayerPlacement(), toString()
};
```

### Strategy Interface

```cpp
// Input to strategy computation
struct PlacementInput {
    // Model info
    std::string architecture;
    int n_layers;
    int d_model;
    int n_heads;
    int d_ff;
    int vocab_size;
    ggml_type quant_type;
    
    // Topology info (from MPITopology)
    int world_size;
    int ranks_per_node;
    std::vector<DeviceCapability> device_capabilities;
};

// Base class for placement algorithms
class PlacementStrategy {
public:
    virtual PlacementPlan compute(const PlacementInput& input) const = 0;
    virtual std::string name() const = 0;
    virtual bool isApplicable(const PlacementInput& input) const = 0;
};
```

### Available Strategies

| Strategy | Description | Status |
|----------|-------------|--------|
| `CPUOnlyStrategy` | All layers → CPU with optional vocab sharding | ✅ Working |
| `GPUFirstStrategy` | Fill GPU memory first, overflow to CPU | Placeholder (falls back to CPU) |

### Integration Points

```cpp
// MPITopology integration
auto topology = std::make_shared<MPITopology>(rank, world_size, MPI_COMM_WORLD);
topology->exchangeCapabilities();

PlacementInput input;
// ... fill model info ...
PlacementPlan plan = topology->computePlacement(input);

// WeightPlacementMap integration
auto placement_map = std::make_shared<WeightPlacementMap>();
placement_map->applyPlan(plan);
```

## Design Decisions

1. **Deterministic computation**: All ranks compute the same plan locally (no MPI broadcast needed)
2. **GPU disabled by default**: `GPUFirstStrategy` falls back to CPU until GPU work begins
3. **Vocab sharding**: Large vocab (>100k) auto-shards embedding/lm_head when multi-rank
4. **Split support**: `LayerPlacement` can assign attention and FFN to different devices

## Test Results

- 34 new placement tests: All passing
- 175 total V2 unit tests: All passing

## Future Work

1. **Wire into model loading**: Call `computePlacement()` after model metadata is known
2. **Implement GPU strategies**: `GPUFirstStrategy` and `LayerBalancedStrategy`
3. **Phase G1+**: Single-GPU per rank, GPU attention kernels, cross-device transfers

## Documentation Updated

- `docs/v2/DISTRIBUTED_ARCHITECTURE_IMPLEMENTATION.md`: Updated Phase G0 status and API examples
