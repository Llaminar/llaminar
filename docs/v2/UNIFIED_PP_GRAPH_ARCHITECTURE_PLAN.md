# Unified Pipeline Parallel Graph Architecture Plan

**Author:** Claude (GitHub Copilot)  
**Date:** February 1, 2026  
**Status:** Planning  
**Epic:** Unified Parallelism Graph System

---

## Executive Summary

This document outlines the plan to unify Pipeline Parallelism (PP) into the same graph-based execution system currently used by Tensor Parallelism (TP). The goal is to enable complex parallelism compositions like `PP(TP(cudas) + TP(rocms) + TP(cpus))` through a single `ComputeGraph` with explicit device assignments and transfer stages.

### Key Outcomes

1. **Single execution model** - All parallelism (TP, PP, future MoE) uses `ComputeGraph` + `DeviceGraphExecutor`
2. **Composable domains** - TP domains can be nested within PP stages
3. **Explicit transfers** - `LocalPPTransferStage` is a first-class graph node
4. **Sunset `LocalPPOrchestrator`** - Remove the separate orchestrator pattern

---

## Current State vs Target State

### Current State (Dual Execution Models)

```
┌─────────────────────────────────────────────────────────────────┐
│ Tensor Parallelism (TP)                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ComputeGraph → DeviceGraphExecutor                                │ │
│ │   - Stages have device_id                                   │ │
│ │   - LocalTPAllreduceStage at sync points                    │ │
│ │   - Single graph for full model                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Pipeline Parallelism (PP) - SEPARATE SYSTEM                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ LocalPPOrchestrator                                         │ │
│ │   - Multiple IInferenceRunner (each with own graph)         │ │
│ │   - Manual transferActivations() between runners            │ │
│ │   - Orchestrator manages stage sequencing                   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Target State (Unified Graph)

```
┌─────────────────────────────────────────────────────────────────┐
│ All Parallelism: Single ComputeGraph                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ComputeGraph → DeviceGraphExecutor::executeMultiDevice()          │ │
│ │                                                             │ │
│ │   [PP Stage 0: TP Domain A (cuda:0, cuda:1)]                │ │
│ │     ├── embedding (device: cuda:0)                          │ │
│ │     ├── layers 0-9 (with LocalTPAllreduceStage)             │ │
│ │     └── layer9_ffn_residual                                 │ │
│ │                                                             │ │
│ │   [LocalPPTransferStage: Stage 0 → Stage 1]                 │ │
│ │                                                             │ │
│ │   [PP Stage 1: TP Domain B (rocm:0, rocm:1)]                │ │
│ │     ├── layers 10-19 (with LocalTPAllreduceStage)           │ │
│ │     └── layer19_ffn_residual                                │ │
│ │                                                             │ │
│ │   [LocalPPTransferStage: Stage 1 → Stage 2]                 │ │
│ │                                                             │ │
│ │   [PP Stage 2: TP Domain C (cpu:socket0, cpu:socket1)]      │ │
│ │     ├── layers 20-27                                        │ │
│ │     ├── final_norm                                          │ │
│ │     └── lm_head                                             │ │
│ │                                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Design

### 1. Configuration Data Structures

#### 1.1 TPDomainConfig (New)

```cpp
// Location: src/v2/config/TPDomainConfig.h

/**
 * @brief Configuration for a Tensor Parallel domain
 * 
 * A TP domain is a group of devices that work together on sharded operations.
 * Within a domain, devices use a collective backend (NCCL, RCCL, PCIeBAR, etc.)
 * for allreduce operations.
 */
struct TPDomainConfig {
    std::string name;                          // e.g., "gpu_tp_nvidia", "cpu_tp"
    std::vector<DeviceId> devices;             // Devices in this domain
    CollectiveBackendType tp_backend;          // Backend for TP collectives
    std::vector<float> device_weights;         // Optional: proportional weights
    
    // Factory method to create the appropriate ILocalTPContext
    std::unique_ptr<ILocalTPContext> createTPContext() const;
    
    // Helpers
    DeviceId primaryDevice() const { return devices.empty() ? DeviceId::cpu() : devices[0]; }
    int degree() const { return static_cast<int>(devices.size()); }
    bool isHomogeneous() const;  // All same device type?
};
```

#### 1.2 PPStageConfig (New)

```cpp
// Location: src/v2/config/PPStageConfig.h

/**
 * @brief Configuration for a Pipeline Parallel stage
 * 
 * A PP stage is a contiguous range of layers executed on a specific TP domain.
 */
struct PPStageConfig {
    int stage_id;                              // 0, 1, 2, ...
    std::string domain_name;                   // Which TPDomainConfig
    int first_layer;                           // Inclusive
    int last_layer;                            // Exclusive
    bool has_embedding;                        // True for first stage
    bool has_lm_head;                          // True for last stage
    
    int numLayers() const { return last_layer - first_layer; }
};
```

#### 1.3 PipelineConfig (New)

```cpp
// Location: src/v2/config/PipelineConfig.h

/**
 * @brief Complete pipeline configuration for PP + TP composition
 */
struct PipelineConfig {
    // TP Domains
    std::vector<TPDomainConfig> tp_domains;
    
    // PP Stages (ordered)
    std::vector<PPStageConfig> pp_stages;
    
    // PP Transfer backends (key: {from_stage, to_stage})
    std::map<std::pair<int,int>, CollectiveBackendType> pp_transfer_backends;
    
    // Model info
    int total_layers = 0;
    
    // Lookup helpers
    const TPDomainConfig* getDomain(const std::string& name) const;
    const TPDomainConfig* getDomainForStage(int stage_id) const;
    const PPStageConfig* getStageForLayer(int layer_idx) const;
    DeviceId getDeviceForLayer(int layer_idx) const;
    bool needsPPTransfer(int from_layer, int to_layer) const;
    CollectiveBackendType getTransferBackend(int from_stage, int to_stage) const;
    
    // Validation
    bool validate(std::string* error_msg = nullptr) const;
    
    // Factory: Create from OrchestrationConfig
    static PipelineConfig fromOrchestrationConfig(const OrchestrationConfig& orch);
};
```

### 2. Graph Building Changes

#### 2.1 Qwen2GraphConfig Extensions

```cpp
// Add to existing Qwen2GraphConfig in Qwen2Graph.h

// =================================================================
// Unified Pipeline Parallel Configuration
// =================================================================

/// Pipeline configuration for PP + TP composition
/// When set, graph building uses multi-stage PP-aware logic
std::shared_ptr<PipelineConfig> pipeline_config = nullptr;

/// PP contexts for inter-stage transfers (one per stage boundary)
/// Key: {from_stage, to_stage}
std::map<std::pair<int,int>, ILocalPPContext*> pp_contexts;

/// TP contexts for each domain (one per domain)
std::map<std::string, ILocalTPContext*> domain_tp_contexts;
```

#### 2.2 New Graph Building Method

```cpp
// Add to Qwen2Graph class

/**
 * @brief Build unified forward graph with PP + TP composition
 * 
 * Creates a single ComputeGraph spanning all PP stages with:
 * - Per-layer device assignment based on PP stage → TP domain mapping
 * - LocalTPAllreduceStage nodes within each TP domain
 * - LocalPPTransferStage nodes at PP stage boundaries
 * 
 * @param input Forward pass input
 * @param output Forward pass output (populated)
 * @return ComputeGraph spanning all stages
 */
ComputeGraph buildUnifiedPipelineGraph(
    const Qwen2ForwardInput& input,
    Qwen2ForwardOutput& output);
```

### 3. LocalPPTransferStage Integration

The `LocalPPTransferStage` we created in Phase 5/6 will be used as a first-class graph node:

```cpp
// Already exists: src/v2/execution/compute_stages/stages/LocalPPTransferStage.h

// Usage in graph building:
LocalPPTransferStage::Params transfer_params;
transfer_params.pp_ctx = pp_contexts_.at({from_stage, to_stage});
transfer_params.tensor = hidden_state_buffer;
transfer_params.stage_from = from_stage;
transfer_params.stage_to = to_stage;

graph.addNode("pp_transfer_" + std::to_string(from_stage) + "_to_" + std::to_string(to_stage),
    std::make_unique<LocalPPTransferStage>(transfer_params),
    DeviceId::cpu());  // Transfer stage is device-agnostic
```

---

## Implementation Phases

### Phase 1: Configuration Infrastructure (2-3 days)

**Goal:** Create the data structures for expressing PP + TP compositions.

| Task | File | Description | Tests |
|------|------|-------------|-------|
| 1.1 | `TPDomainConfig.h/cpp` | TP domain configuration struct | Unit: `Test__TPDomainConfig.cpp` |
| 1.2 | `PPStageConfig.h/cpp` | PP stage configuration struct | Unit: `Test__PPStageConfig.cpp` |
| 1.3 | `PipelineConfig.h/cpp` | Complete pipeline config with validation | Unit: `Test__PipelineConfig.cpp` |
| 1.4 | `PipelineConfigParser.cpp` | Parse from CLI/YAML | Unit: `Test__PipelineConfigParser.cpp` |
| 1.5 | Integration | Wire into `OrchestrationConfig` | Integration: existing orchestration tests |

**Acceptance Criteria:**
- [ ] Can express `PP(TP + TP + TP)` configurations declaratively
- [ ] Validation catches invalid configs (overlapping layers, missing domains)
- [ ] CLI parsing works for `--define-domain` and `--pp-stage` flags
- [ ] All existing orchestration tests pass

### Phase 2: Graph Building with PP (3-4 days)

**Goal:** Extend `Qwen2Graph` to build unified graphs with PP stages.

| Task | File | Description | Tests |
|------|------|-------------|-------|
| 2.1 | `Qwen2Graph.h` | Add `pipeline_config` and helper fields | N/A (config only) |
| 2.2 | `Qwen2Graph.cpp` | Implement `buildUnifiedPipelineGraph()` | Unit: `Test__Qwen2Graph_PP.cpp` |
| 2.3 | `Qwen2Graph.cpp` | Per-layer device assignment logic | Unit: same |
| 2.4 | `Qwen2Graph.cpp` | Insert `LocalPPTransferStage` at boundaries | Unit: same |
| 2.5 | `Qwen2Graph.cpp` | Per-domain TP context wiring | Unit: same |

**Acceptance Criteria:**
- [ ] `buildUnifiedPipelineGraph()` creates correct graph for 2-stage PP
- [ ] `buildUnifiedPipelineGraph()` creates correct graph for 3-stage PP
- [ ] TP allreduce stages use correct domain contexts
- [ ] PP transfer stages use correct PP contexts
- [ ] Graph structure matches expected node count and dependencies

### Phase 3: Execution Integration (2-3 days)

**Goal:** Execute unified PP graphs via `DeviceGraphExecutor::executeMultiDevice()`.

| Task | File | Description | Tests |
|------|------|-------------|-------|
| 3.1 | `DeviceGraphOrchestrator.cpp` | Build device contexts for all domains | Unit: `Test__DeviceGraphOrchestrator_PP.cpp` |
| 3.2 | `DeviceGraphExecutor.cpp` | Verify `executeMultiDevice()` handles PP graphs | Integration |
| 3.3 | `Qwen2Pipeline.cpp` | Option to use unified PP path | Integration |
| 3.4 | Context factories | Create `ILocalPPContext` per stage boundary | Unit |

**Acceptance Criteria:**
- [ ] `executeMultiDevice()` runs PP graph without errors
- [ ] Correct device context used for each stage
- [ ] PP transfers execute at correct points
- [ ] Hidden states flow correctly through pipeline

### Phase 4: Parity Testing (2-3 days)

**Goal:** Verify unified PP produces identical results to current orchestrator.

| Task | File | Description | Tests |
|------|------|-------------|-------|
| 4.1 | `Test__UnifiedPP_Parity.cpp` | Compare unified vs orchestrator output | Integration/Parity |
| 4.2 | Same | 2-stage PP parity (CPU+GPU) | Parity |
| 4.3 | Same | 3-stage PP parity (GPU+GPU+CPU) | Parity |
| 4.4 | Same | PP(TP+TP) parity (Scenario 7 style) | Parity |
| 4.5 | Update existing | Update `Test__Qwen2_ParityMatrix.cpp` for unified path | Parity |

**Acceptance Criteria:**
- [ ] Token-for-token match with current orchestrator for decode
- [ ] Top-5 logit match for prefill
- [ ] All existing parity tests pass with unified path

### Phase 5: Orchestrator Sunset (1-2 days) ✅ COMPLETED 2026-02-02

**Goal:** Remove `LocalPPOrchestrator` and redirect all PP to unified graph.

| Task | File | Description | Status |
|------|------|-------------|--------|
| 5.1 | `OrchestrationRunner.cpp` | Use unified path for PP | ✅ Already uses unified path |
| 5.2 | `LocalPPOrchestrator.*` | Mark as deprecated | ✅ Deleted |
| 5.3 | Parity test base | Remove orchestrator-based PP setup | ✅ Updated `Qwen2ParityTestBase.h` |
| 5.4 | Delete | Remove `LocalPPOrchestrator.h/cpp` | ✅ Deleted |
| 5.5 | Cleanup | Remove `PPSnapshotCoordinator` if no longer needed | ✅ Deleted |

**Acceptance Criteria:**
- [x] All PP tests pass without `LocalPPOrchestrator` (LocalPP tests now use stub until Phase 6 impl)
- [x] No references to removed code
- [x] Clean build with no warnings

**Files Removed:**
- `src/v2/execution/local_execution/pp/LocalPPOrchestrator.h`
- `src/v2/execution/local_execution/pp/LocalPPOrchestrator.cpp`
- `src/v2/execution/local_execution/pp/PPSnapshotCoordinator.h`
- `tests/v2/integration/parity/Test__LocalPP_MultiStageParity.cpp`
- `tests/v2/integration/parity/Test__UnifiedPP_vs_Orchestrator.cpp`

**Next:** Phase 6 requires implementing `setupLocalPPPipeline()` in `Qwen2ParityTestBase.h` using the unified graph approach.

### Phase 6: Scenario 7 Validation (2-3 days)

**Goal:** Full validation of complex compositions including Scenario 7.

| Task | File | Description | Tests |
|------|------|-------------|-------|
| 6.1 | `Test__Scenario7_Unified.cpp` | Scenario 7 with unified graph | Integration |
| 6.2 | Same | Cross-vendor TP within PP stage | Integration |
| 6.3 | Same | CPU TP domain as PP stage | Integration |
| 6.4 | Same | Proportional TP weights | Integration |
| 6.5 | Documentation | Update architecture docs | N/A |

**Acceptance Criteria:**
- [ ] Scenario 7 config parses correctly
- [ ] Unified graph has expected structure (domains, transfers)
- [ ] Mock execution validates data flow patterns
- [ ] Documentation reflects new architecture

---

## Test Strategy

### Unit Tests (per component)

| Component | Test File | Key Test Cases |
|-----------|-----------|----------------|
| `TPDomainConfig` | `Test__TPDomainConfig.cpp` | Creation, validation, context factory |
| `PPStageConfig` | `Test__PPStageConfig.cpp` | Layer ranges, embedding/LM head flags |
| `PipelineConfig` | `Test__PipelineConfig.cpp` | Composition, validation, lookups |
| `buildUnifiedPipelineGraph()` | `Test__Qwen2Graph_PP.cpp` | Graph structure, node count, dependencies |
| `LocalPPTransferStage` | `Test__LocalPPTransferStage.cpp` | ✅ Already exists (20 tests) |

### Integration Tests

| Test | Description | Location |
|------|-------------|----------|
| `Test__UnifiedPP_2Stage.cpp` | 2-stage PP with mock devices | `tests/v2/integration/pp/` |
| `Test__UnifiedPP_3Stage.cpp` | 3-stage PP with mock devices | Same |
| `Test__UnifiedPP_WithTP.cpp` | PP stages with internal TP | Same |
| `Test__UnifiedPP_Execution.cpp` | Full execution with real model | Same |

### Parity Tests

| Test | Description | Location |
|------|-------------|----------|
| `Test__UnifiedPP_vs_Orchestrator.cpp` | A/B comparison during transition | `tests/v2/integration/parity/` |
| Update `Test__Qwen2_ParityMatrix.cpp` | Add unified PP config | Existing |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing PP tests | High | Run parity tests continuously during development |
| Performance regression | Medium | Benchmark unified vs orchestrator before sunset |
| Complex multi-device debugging | Medium | Add detailed logging at PP boundaries |
| Hidden state buffer management | Medium | Careful review of buffer lifetimes |

---

## Success Metrics

1. **Functionality:** All existing PP tests pass with unified architecture
2. **Parity:** Token-for-token match with orchestrator-based PP
3. **Composability:** Scenario 7 configuration works end-to-end
4. **Simplicity:** Single code path for all parallelism modes
5. **Code reduction:** Net deletion of lines after orchestrator sunset

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Configuration | Days 1-3 | None |
| Phase 2: Graph Building | Days 4-7 | Phase 1 |
| Phase 3: Execution | Days 8-10 | Phase 2 |
| Phase 4: Parity Testing | Days 11-13 | Phase 3 |
| Phase 5: Orchestrator Sunset | Days 14-15 | Phase 4 |
| Phase 6: Scenario 7 Validation | Days 16-18 | Phase 5 |

**Total:** ~18 days (3-4 weeks with buffer)

---

## File Inventory

### New Files to Create

```
src/v2/config/
├── TPDomainConfig.h
├── TPDomainConfig.cpp
├── PPStageConfig.h
├── PPStageConfig.cpp
├── PipelineConfig.h
├── PipelineConfig.cpp
└── PipelineConfigParser.cpp

tests/v2/unit/config/
├── Test__TPDomainConfig.cpp
├── Test__PPStageConfig.cpp
└── Test__PipelineConfig.cpp

tests/v2/unit/models/qwen/
└── Test__Qwen2Graph_PP.cpp

tests/v2/integration/pp/
├── Test__UnifiedPP_2Stage.cpp
├── Test__UnifiedPP_3Stage.cpp
├── Test__UnifiedPP_WithTP.cpp
└── Test__UnifiedPP_Execution.cpp

tests/v2/integration/parity/
└── Test__UnifiedPP_vs_Orchestrator.cpp
```

### Files to Modify

```
src/v2/models/qwen/Qwen2Graph.h           # Add pipeline_config, new method
src/v2/models/qwen/Qwen2Graph.cpp         # Implement buildUnifiedPipelineGraph()
src/v2/config/OrchestrationConfig.h       # Wire in PipelineConfig
src/v2/execution/OrchestrationRunner.cpp  # Use unified path for PP
src/v2/CMakeLists.txt                     # Add new source files
tests/v2/CMakeLists.txt                   # Add new test files
```

### Files to Delete (Phase 5)

```
src/v2/execution/local_execution/pp/LocalPPOrchestrator.h
src/v2/execution/local_execution/pp/LocalPPOrchestrator.cpp
tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h  # PP orchestrator setup methods
```

---

## Appendix: Example Configurations

### A. Simple: PP(cpu, cuda, cuda)

```yaml
tp_domains:
  - name: "cpu_stage"
    devices: ["cpu"]
    backend: "host"
    
  - name: "gpu0_stage"
    devices: ["cuda:0"]
    backend: "nccl"
    
  - name: "gpu1_stage"
    devices: ["cuda:1"]
    backend: "nccl"

pp_stages:
  - stage_id: 0
    domain: "cpu_stage"
    layers: [0, 10]
    has_embedding: true
    
  - stage_id: 1
    domain: "gpu0_stage"
    layers: [10, 20]
    
  - stage_id: 2
    domain: "gpu1_stage"
    layers: [20, 28]
    has_lm_head: true

pp_transfer_backends:
  "0->1": "host"   # CPU → GPU
  "1->2": "nccl"   # GPU → GPU
```

### B. Complex: Scenario 7 (PP of TP domains)

```yaml
tp_domains:
  - name: "gpu_tp_nvidia"
    devices: ["cuda:0", "cuda:1"]
    backend: "nccl"
    weights: [0.5, 0.5]
    
  - name: "gpu_tp_amd"
    devices: ["rocm:0", "rocm:1"]
    backend: "rccl"
    weights: [0.5, 0.5]
    
  - name: "cpu_tp"
    devices: ["cpu:socket0", "cpu:socket1"]
    backend: "mpi"
    weights: [0.5, 0.5]

pp_stages:
  - stage_id: 0
    domain: "gpu_tp_nvidia"
    layers: [0, 10]
    has_embedding: true
    
  - stage_id: 1
    domain: "gpu_tp_amd"
    layers: [10, 20]
    
  - stage_id: 2
    domain: "cpu_tp"
    layers: [20, 28]
    has_lm_head: true

pp_transfer_backends:
  "0->1": "pcie_bar"  # NVIDIA → AMD
  "1->2": "host"      # AMD → CPU
```

---

## Next Steps

1. **Review this plan** - Confirm scope and approach
2. **Begin Phase 1** - Create configuration data structures
3. **Parallel track** - Update existing tests to be ready for unified path
