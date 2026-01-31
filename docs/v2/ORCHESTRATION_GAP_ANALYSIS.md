# Orchestration Architecture Gap Analysis

## Executive Summary

The Llaminar V2 orchestration architecture has **three distinct layers** designed to support complex compositions like `PipelineParallel(TensorParallel(CUDA) + TensorParallel(ROCm) + TensorParallel(CPU))`. However, there are significant gaps between design and implementation:

| Layer | Component | Status | Notes |
|-------|-----------|--------|-------|
| **Layer 1** | `MPIBootstrap` | ⚠️ Partial | Socket-based world_size, no model awareness |
| **Layer 2** | `ExecutionPlanBuilder` | ✅ Implemented | Parses domains, assigns layers, computes shards |
| **Layer 2** | `OrchestrationRunner` | ✅ Implemented | But NOT wired to Main.cpp |
| **Layer 2** | `TPPPValidator` | ✅ Implemented | But NOT called in Main.cpp path |
| **Layer 3** | `DeviceGraphOrchestrator` | ✅ Implemented | Single-device graph execution |
| **Layer 3** | `MultiDeviceOrchestrator` | ✅ Implemented | LOCAL TP across devices |
| **Layer 3** | `MultiDomainOrchestrator` | ⚠️ Stub | PP composition unimplemented |
| **Main.cpp** | Integration | ❌ Broken | Uses old path, ignores new orchestration |

## Current Flow (Main.cpp - BROKEN)

```
                    System Topology
                          │
                          ▼

              MPIBootstrap                        │
   num_procs = topology.num_sockets (e.g., 2)    │
   No model awareness, no device-type preference │

                          │
                          ▼ mpirun -np 2
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
    ┌─────────────┐              ┌─────────────┐
    │   Rank 0    │              │   Rank 1    │
    │  NUMA 0     │              │  NUMA 1     │
    └─────────────┘              └─────────────┘
           │                             │
           ▼ parse_device("auto")        ▼ parse_device("auto")
    ┌─────────────┐              ┌─────────────┐
    │  CUDA:0     │              │  ROCm:0     │  ← Round-robin picks wrong type!
    └─────────────┘              └─────────────┘
           │                             │
           ▼                             ▼
    ┌───────────────────────────────────────────┐
    │      BackendRouter detects heterogeneous   │
    │      Selects MPI (host-staged) backend     │
    │      Result: 0.06 tok/s instead of 50+    │
    └───────────────────────────────────────────┘
```

## Designed Flow (OrchestrationRunner - NOT WIRED)

```
                    Model Metadata
                          │
                          ▼

          TPPPValidator (Phase 1)                │
   max_tp = min(n_heads, n_kv_heads)             │
   Validates: tp_degree <= max_tp                │
   Fails early with clear error message          │

                          │
                          ▼

       ExecutionPlanBuilder (Phase 2)            │
   Resolves domains → devices                    │
   Computes layer assignments for PP             │
   Computes weight shards for TP                 │
   Outputs: RankExecutionPlan per rank           │

                          │
                          ▼

       OrchestrationRunner (Phase 3)             │
   Creates appropriate orchestrator:             │
   - DeviceGraphOrchestrator (single device)     │
   - MultiDeviceOrchestrator (LOCAL TP)          │
   - MultiDomainOrchestrator (PP + TP)           │

```

## Gap 1: Model-Unaware MPI Bootstrap

**Problem**: `MPIBootstrap::getDefaultConfig()` sets `num_procs = topology.num_sockets` without knowing:
- Model's `n_kv_heads` (constrains max TP degree)
- Available GPU types (CUDA vs ROCm)
- Desired parallelism mode (TP vs PP vs both)

**Current Code** (`MPIBootstrap.cpp:300`):
```cpp
MPILaunchConfig MPIBootstrap::getDefaultConfig(const CPUTopology &topology)
{
    MPILaunchConfig config;
    config.num_procs = topology.num_sockets;  // ← No model awareness!
    ...
}
```

**Required Fix**:
```cpp
// New Phase 0: Pre-launch analysis
struct BootstrapDecision {
    int optimal_world_size;
    DeviceType preferred_device_type;  // CUDA or ROCm
    std::string reason;
};

BootstrapDecision analyzeModelAndTopology(
    const std::string& model_path,      // GGUF file
    const SystemTopology& topology,     // Sockets, GPUs
    const ClusterInventory& inventory)  // CUDA/ROCm counts
{
    // Read model metadata (lightweight, just header)
    auto metadata = GGUFMetadata::readHeader(model_path);
    int max_tp = std::min(metadata.n_heads, metadata.n_kv_heads);
    
    // Prefer homogeneous GPU groups
    int cuda_count = inventory.cuda_gpus;
    int rocm_count = inventory.rocm_gpus;
    
    if (cuda_count >= 2 && cuda_count <= max_tp) {
        return {cuda_count, DeviceType::CUDA, "All CUDA for NCCL"};
    }
    if (rocm_count >= 2 && rocm_count <= max_tp) {
        return {rocm_count, DeviceType::ROCm, "All ROCm for RCCL"};
    }
    if (cuda_count >= 1 || rocm_count >= 1) {
        return {1, cuda_count > 0 ? DeviceType::CUDA : DeviceType::ROCm,
                "Single GPU (model too small for TP)"};
    }
    return {topology.num_sockets, DeviceType::CPU, "CPU-only"};
}
```

## Gap 2: Device Selection Ignores Collective Performance

**Problem**: `parse_device("auto")` does round-robin without considering:
- Device type matching (CUDA+CUDA vs CUDA+ROCm)
- Collective backend performance (NCCL >> MPI)

**Current Code** (`Main.cpp:88`):
```cpp
if (device_str == "auto") {
    // Count available GPUs (skip device 0 which is CPU)
    std::vector<size_t> gpu_indices;
    for (size_t i = 1; i < devices.size(); ++i) {
        const auto &dev = devices[i];
        if (dev.type == ComputeBackendType::GPU_CUDA ||
            dev.type == ComputeBackendType::GPU_ROCM) {
            gpu_indices.push_back(i);  // ← Types mixed together!
        }
    }
    // Round-robin GPU assignment based on local_rank
    size_t gpu_idx = local_rank % gpu_indices.size();  // ← Type-blind!
}
```

**Required Fix**:
```cpp
DeviceId selectDeviceForRank(int rank, int world_size, 
                             DeviceType preferred_type,
                             const DeviceManager& dm) {
    // Group devices by type
    auto cuda_devices = dm.devicesOfType(ComputeBackendType::GPU_CUDA);
    auto rocm_devices = dm.devicesOfType(ComputeBackendType::GPU_ROCM);
    
    if (preferred_type == DeviceType::CUDA && 
        rank < static_cast<int>(cuda_devices.size())) {
        return DeviceId::cuda(cuda_devices[rank].device_id);
    }
    if (preferred_type == DeviceType::ROCm && 
        rank < static_cast<int>(rocm_devices.size())) {
        return DeviceId::rocm(rocm_devices[rank].device_id);
    }
    // Fallback to best available
    return selectBestAvailable(rank, dm);
}
```

## Gap 3: Main.cpp Bypasses New Orchestration

**Problem**: `Main.cpp` uses `createInferenceRunner()` which goes directly to `InferenceRunnerFactory`, completely ignoring:
- `OrchestrationConfig` parsing
- `ExecutionPlanBuilder` 
- `RankExecutionPlan`
- `TPPPValidator`
- `OrchestrationRunner`

**Current Flow**:
```
Main.cpp → DeviceOrchestrator (legacy) → InferenceRunnerFactory → DeviceGraphOrchestrator
```

**Required Flow**:
```
Main.cpp → OrchestrationConfigParser → ExecutionPlanBuilder → OrchestrationRunner 
         → (MultiDomainOrchestrator | MultiDeviceOrchestrator | DeviceGraphOrchestrator)
```

## Gap 4: Pipeline Parallel Unfinished

**Problem**: `MultiDomainOrchestrator` exists as a header but is largely unimplemented.

**Current State** (`MultiDomainOrchestrator.h`):
```cpp
class MultiDomainOrchestrator : public IInferenceRunner
{
    // Header declares interface but implementation is minimal
};
```

**Missing**:
- PP stage execution ordering
- Inter-stage activation transfer (Send/Recv stages)
- Micro-batch scheduling
- Memory optimization (1F1B schedule)

## Gap 5: OrchestrationRunner Not Integrated

**Files Exist**:
- `OrchestrationRunner.h/cpp` - Full implementation
- `OrchestrationRunnerFactory.h/cpp` - Factory
- `IOrchestrationRunner.h` - Interface

**But Not Used**:
- `Main.cpp` doesn't include or call these
- No CLI path to activate new orchestration

## Implementation Plan

### Phase 1: Model-Aware Bootstrap (Priority: HIGH)

**Goal**: Determine optimal world_size from model metadata before MPI launch.

**Tasks**:
1. Add `GGUFMetadataReader::readHeaderOnly()` for lightweight model probing
2. Add `BootstrapDecision analyzeModelAndTopology()` function
3. Modify `MPIBootstrap::selfLaunch()` to call analysis before `-np` decision
4. Add `--force-world-size N` CLI override for testing

**LOE**: ~200 lines of code, 2-3 days

### Phase 2: Type-Aware Device Selection (Priority: HIGH)

**Goal**: Prefer homogeneous device groups for collective performance.

**Tasks**:
1. Refactor `parse_device()` to accept preferred device type
2. Group devices by type in DeviceManager
3. Implement type-priority selection algorithm
4. Add `--prefer-device-type cuda|rocm|any` CLI option

**LOE**: ~100 lines of code, 1-2 days

### Phase 3: Wire OrchestrationRunner to Main.cpp (Priority: MEDIUM)

**Goal**: Enable new orchestration path via CLI flag.

**Tasks**:
1. Add `--use-orchestration-runner` flag
2. Integrate OrchestrationConfigParser for domain/stage args
3. Call TPPPValidator before weight loading
4. Use OrchestrationRunnerFactory to create runner

**LOE**: ~150 lines of code, 2-3 days

### Phase 4: Pipeline Parallel Implementation (Priority: LOW)

**Goal**: Implement PP execution in MultiDomainOrchestrator.

**Tasks**:
1. Implement `SendActivationsStage` and `RecvActivationsStage`
2. Add PP stage ordering in MultiDomainOrchestrator
3. Implement 1F1B micro-batch schedule
4. Add PP parity tests

**LOE**: ~500+ lines of code, 1-2 weeks

## Test Scenarios

### Scenario 1: Single Device (Baseline)
```bash
./llaminar2 --no-mpi-bootstrap -m model.gguf
# Expected: Uses single GPU, ~50 tok/s
```

### Scenario 2: Homogeneous LOCAL TP (CUDA)
```bash
./llaminar2 --tp-devices cuda:0,cuda:1 -m model.gguf
# Expected: Uses NCCL, 2-way TP, ~80 tok/s
```

### Scenario 3: Homogeneous LOCAL TP (ROCm)
```bash
./llaminar2 --tp-devices rocm:0,rocm:1 -m model.gguf
# Expected: Uses RCCL, 2-way TP, ~60 tok/s
```

### Scenario 4: Model-Constrained TP
```bash
./llaminar2 --tp 4 -m qwen2.5-0.5b.gguf  # Model has only 2 KV heads
# Expected: Error: "TP degree 4 exceeds model maximum 2 (n_kv_heads=2)"
```

### Scenario 5: Heterogeneous PP (Future)
```bash
./llaminar2 --define-domain "gpu=cuda:0,cuda:1" \
            --define-domain "rocm=rocm:0,rocm:1" \
            --pp-stage "0=gpu:0-11" --pp-stage "1=rocm:12-23" \
            -m llama3-70b.gguf
# Expected: PP across CUDA (NCCL) and ROCm (RCCL) domains
```

## Files to Modify

| File | Change |
|------|--------|
| `src/v2/utils/MPIBootstrap.cpp` | Add model-aware world_size calculation |
| `src/v2/utils/MPIBootstrap.h` | Add BootstrapDecision struct |
| `src/v2/Main.cpp` | Wire OrchestrationRunner, add CLI flags |
| `src/v2/loaders/GGUFMetadataReader.h/cpp` | Add lightweight header-only reader |
| `tests/v2/integration/...` | Add orchestration integration tests |

## Success Metrics

1. **Single device**: No performance regression
2. **Homogeneous TP**: NCCL/RCCL achieves >1.5x speedup over single device
3. **Model validation**: Clear error messages for impossible configurations
4. **Heterogeneous PP**: Correct execution across mixed device types (future)

---

## Immediate Fix: Homogeneous Device Preference

### Root Cause Recap

The current `parse_device("auto")` mixes CUDA and ROCm devices in a single `gpu_indices` vector, then does round-robin assignment. On a 2-socket system with 1 CUDA + 1 ROCm:

```
Rank 0: gpu_indices[0 % 2] = CUDA:0  ✓
Rank 1: gpu_indices[1 % 2] = ROCm:0  ✗ (heterogeneous!)
```

### Proposed Fix (Minimal Invasive)

Modify `parse_device()` in Main.cpp to **prefer homogeneous groups**:

```cpp
DeviceId parse_device(const std::string &device_str, DeviceManager &dm, 
                      int local_rank, int local_size)
{
    if (device_str == "auto")
    {
        const auto &devices = dm.devices();

        // Separate GPUs by type
        std::vector<size_t> cuda_indices, rocm_indices;
        for (size_t i = 1; i < devices.size(); ++i)
        {
            const auto &dev = devices[i];
            if (dev.type == ComputeBackendType::GPU_CUDA)
                cuda_indices.push_back(i);
            else if (dev.type == ComputeBackendType::GPU_ROCM)
                rocm_indices.push_back(i);
        }

        // Choose the best homogeneous group based on world size
        std::vector<size_t>* chosen = nullptr;
        const char* type_name = nullptr;
        
        // Preference order: CUDA (NCCL) > ROCm (RCCL) > mixed
        if (!cuda_indices.empty() && local_size <= static_cast<int>(cuda_indices.size()))
        {
            chosen = &cuda_indices;
            type_name = "CUDA (NCCL-capable)";
        }
        else if (!rocm_indices.empty() && local_size <= static_cast<int>(rocm_indices.size()))
        {
            chosen = &rocm_indices;
            type_name = "ROCm (RCCL-capable)";
        }
        else
        {
            // Fallback: best single GPU
            if (!cuda_indices.empty())
            {
                chosen = &cuda_indices;
                type_name = "CUDA (single)";
            }
            else if (!rocm_indices.empty())
            {
                chosen = &rocm_indices;
                type_name = "ROCm (single)";
            }
        }

        if (!chosen || chosen->empty())
        {
            LOG_DEBUG("No GPUs available, falling back to CPU");
            return DeviceId::cpu();
        }

        // Assign within the homogeneous group
        size_t gpu_idx = local_rank % chosen->size();
        size_t dm_idx = (*chosen)[gpu_idx];
        const auto &dev = devices[dm_idx];
        
        LOG_INFO("Homogeneous auto-select: " << type_name 
                 << " device " << dev.device_id 
                 << " for local_rank " << local_rank << "/" << local_size);

        switch (dev.type)
        {
        case ComputeBackendType::GPU_CUDA:
            return DeviceId::cuda(dev.device_id);
        case ComputeBackendType::GPU_ROCM:
            return DeviceId::rocm(dev.device_id);
        default:
            return DeviceId::cpu();
        }
    }
    // ... rest of function unchanged
}
```

### Bootstrap World Size Fix

Additionally, modify `MPIBootstrap::getDefaultConfig()` to check GPU counts:

```cpp
MPILaunchConfig MPIBootstrap::getDefaultConfig(const CPUTopology &topology)
{
    MPILaunchConfig config;
    
    // Check available GPUs (requires DeviceManager query)
    // For now, prefer single-process unless user specifies -np
    // The user can request multi-process via --mpi-procs N
    
    // Conservative default: 1 process (single-device mode)
    // This avoids the heterogeneous device trap
    config.num_procs = 1;  // Changed from topology.num_sockets
    
    // If user wants multi-process, they should use --mpi-procs
    // Future: Add --auto-tp to enable model-aware auto-scaling
    
    // Per-socket threading still makes sense
    config.omp_threads_per_rank = topology.cores_per_socket;
    config.bind_to_socket = true;
    config.map_by_socket = true;
    config.omp_places = "sockets";
    config.omp_proc_bind = "close";

    return config;
}
```

### Backward Compatibility

Add `--auto-tp` flag to enable the old behavior (multi-process based on sockets):

```bash
# Old behavior (potentially heterogeneous, risky)
./llaminar2 --auto-tp -m model.gguf

# New default behavior (single-process, safe)
./llaminar2 -m model.gguf

# Explicit multi-GPU (user knows what they're doing)
./llaminar2 --mpi-procs 2 -d cuda:0 -m model.gguf
```

### Implementation Checklist

- [ ] Modify `parse_device()` to prefer homogeneous groups
- [ ] Change default `num_procs` to 1 in `getDefaultConfig()`
- [ ] Add `--auto-tp` flag to CLI args
- [ ] Add logging when heterogeneous would have been selected
- [ ] Update README with new device selection behavior
- [ ] Add test for homogeneous preference logic
