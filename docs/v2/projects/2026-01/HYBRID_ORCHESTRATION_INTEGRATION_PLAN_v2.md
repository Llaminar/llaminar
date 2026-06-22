# Hybrid Orchestration Integration Plan

**Version 2.0 - Consolidated**

A complete implementation plan to wire Llaminar's tested orchestration infrastructure (365+ tests) into the actual inference path, enabling Scenario 7: Heterogeneous Pipeline Cluster.

---

## Executive Summary

Llaminar has extensive orchestration infrastructure that is **fully tested but not wired to actual inference**:
- `ClusterInventory` / `DeviceInventory` - 32 tests passing
- `HeterogeneousMultiDomainStrategy` - 47 tests passing
- `TPDomain` / `MultiDomainTPConfig` - 89 tests passing
- `PipelineParallelConfig` - 32 tests passing
- All collective backends (NCCL, RCCL, PCIeBAR, MPI) - tested

**The Gap**: `InferenceRunnerFactory` ignores this infrastructure and hardcodes single-device execution with MPI-only tensor parallelism.

**The Goal**: Wire everything together so the CLI can orchestrate complex heterogeneous deployments.

**Timeline**: ~47 days (~10 weeks)

---

## Implementation Status

### Phase 6: GlobalDeviceAddress Migration

| Task ID | Task | Status | Notes |
|---------|------|--------|-------|
| 6.1 | Create `GlobalDeviceAddress` class | ✅ COMPLETE | `src/v2/backends/GlobalDeviceAddress.h/cpp` |
| 6.1a | Create `DeviceRegistry` singleton | ✅ COMPLETE | `src/v2/backends/DeviceRegistry.h/cpp` - Centralized device discovery for CPU/CUDA/ROCm |
| 6.1b | Create `DeviceAddressAdapter` utilities | ✅ COMPLETE | `src/v2/backends/DeviceAddressAdapter.h/cpp` - Legacy-to-GlobalDeviceAddress bridging |
| 6.2 | Update `DeviceInfo` to use `GlobalDeviceAddress` | 🔲 PENDING | |
| 6.3 | Update `RankInventory` and `ClusterInventory` | 🔲 PENDING | |
| 6.4 | Update `TPDomain` | 🔲 PENDING | |
| 6.5 | Update `LayerPlacement` and `PlacementPlan` | 🔲 PENDING | |
| 6.6 | Update `CollectiveContext` and `BackendRouter` | 🔲 PENDING | |
| 6.7 | Update CLI parser to produce `GlobalDeviceAddress` | 🔲 PENDING | |
| 6.8 | Update YAML/JSON config parser | 🔲 PENDING | |
| 6.9 | Update MPI serialization | 🔲 PENDING | |
| 6.10 | Unit tests | ✅ COMPLETE | 57+ tests: `Test__GlobalDeviceAddress.cpp`, `Test__DeviceRegistry.cpp`, `Test__DeviceAddressAdapter.cpp` |
| 6.11 | Migration of existing tests | 🔲 PENDING | |

#### Files Created (Phase 6)

| File | Purpose | Tests |
|------|---------|-------|
| `src/v2/backends/GlobalDeviceAddress.h` | Canonical device identifier (hostname:numa:type:ordinal) | 27 passing |
| `src/v2/backends/GlobalDeviceAddress.cpp` | Implementation | |
| `src/v2/backends/IDeviceRegistry.h` | Abstract interface for mockable testing | |
| `src/v2/backends/DeviceRegistry.h` | Singleton header for device discovery | 27 passing |
| `src/v2/backends/DeviceRegistry.cpp` | CPU/CUDA/ROCm device enumeration | |
| `src/v2/backends/DeviceAddressAdapter.h` | Legacy ↔ GlobalDeviceAddress conversions | 30+ passing |
| `src/v2/backends/DeviceAddressAdapter.cpp` | Adapter implementation | |

#### Validation Results

- **DeviceRegistry**: Discovers 6 devices (2 CPU NUMA nodes, 2 CUDA GPUs, 2 ROCm GPUs)
- **All existing backend tests**: Still passing
- **Hardware verified**: 2x NVIDIA RTX 3090 (23 GB), 2x AMD Instinct MI60 (31 GB)

---

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: ENTRY POINT                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                                  llaminar2 CLI                                               │ │
│ │  Parses: --tp, --pp, --device, --tp-scope, --placement-config, etc.                        │ │
│ │  Outputs: OrchestrationConfig (validated user intent)                                       │ │
│ └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: CLUSTER ORCHESTRATION (MPI rank 0 or all ranks collaboratively)                         │
│ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │  ClusterInventory::gatherAll()  →  HeterogeneousMultiDomainStrategy::compute()              │ │
│ │                                              │                                               │ │
│ │                                              ▼                                               │ │
│ │                                   HeterogeneousPlan                                          │ │
│ │                                   (cluster-wide decisions)                                   │ │
│ │                                              │                                               │ │
│ │                                              ▼                                               │ │
│ │                           plan.toRankExecutionPlan(my_rank)                                  │ │
│ └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                         One RankExecutionPlan per MPI rank
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: RANK-LOCAL EXECUTION (each MPI rank independently)                                      │
│ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │  InferenceRunnerFactory  →  WeightManager  →  GraphOrchestrator  →  Qwen2Graph              │ │
│ │                                                                                              │ │
│ │  Uses: RankExecutionPlan (what THIS rank should do)                                         │ │
│ │  Owns: Device initialization, weight loading, graph building, kernel execution              │ │
│ │  Scope: LOCAL devices only (no cross-rank awareness except at collective boundaries)        │ │
│ └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### RankExecutionPlan Contract

The contract between Layer 2 (cluster orchestration) and Layer 3 (rank-local execution):

```cpp
struct RankExecutionPlan {
    // Identity
    int rank;                        // This MPI rank
    std::string hostname;            // Machine we're on
    int numa_node;                   // NUMA node we're bound to

    // Pipeline Parallelism (what layers to run)
    int pp_stage_id;                 // Which PP stage this rank owns
    int first_layer;                 // First layer index to build/execute
    int last_layer;                  // Last layer index (inclusive)
    bool has_embedding;              // Build embedding stage?
    bool has_lm_head;                // Build LM head stage?
    
    // PP Communication
    std::optional<int> prev_rank;    // Rank to receive activations from
    std::optional<int> next_rank;    // Rank to send activations to

    // Tensor Parallelism (how to shard within my layers)
    TPScope tp_scope;                // LOCAL, GLOBAL, or HYBRID
    
    // LOCAL TP (devices within this rank)
    std::vector<GlobalDeviceAddress> local_tp_devices;
    std::vector<float> local_tp_weights;
    CollectiveBackendType local_tp_backend;
    
    // GLOBAL TP (participation in cross-rank TP)
    std::optional<int> global_tp_domain_id;
    int global_tp_rank_in_domain;
    int global_tp_domain_size;

    // Weight loading
    struct WeightShard {
        int shard_index;             // Which shard this rank loads
        int total_shards;            // Total shards
        float work_fraction;         // For proportional TP
    };
    WeightShard weight_shard;

    // Collective configuration
    std::vector<TPDomainSpec> my_domains;  // Domains this rank participates in
    CollectiveBackendType cross_rank_backend;
};
```

**Key Principle**: Layer 3 receives `RankExecutionPlan` and executes it **without needing cluster-wide context**. It doesn't know what other ranks are doing; it just follows its plan.

---

## Unified Device Addressing

### The Problem

The current codebase has inconsistent device addressing:
- `DeviceId` uses `type + ordinal` (e.g., CUDA:0) - no hostname, no NUMA
- `RankPlacement` uses `hostname + socket_id + numa_node` as separate strings
- CLI will need `hostname:numa:type:ordinal` for full cluster control

### The Solution: GlobalDeviceAddress

A single canonical address type used everywhere:

```cpp
struct GlobalDeviceAddress {
    std::string hostname = "localhost";   // Physical machine
    int numa_node = 0;                    // NUMA node / socket
    DeviceType device_type;               // cuda, rocm, cpu
    int device_ordinal = 0;               // Device index within type on that NUMA
    
    // Parse from string (supports shorthand)
    // Full:      "node1:0:cuda:0"
    // Shorthand: "cuda:0" → "localhost:<current_numa>:cuda:0"
    //            "0:cuda:0" → "localhost:0:cuda:0"
    static GlobalDeviceAddress parse(const std::string& spec, int current_numa = 0);
    
    // Convert to string
    std::string toString() const;  // "node1:0:cuda:0"
    
    // Convert to local DeviceId (for kernel calls)
    DeviceId toLocalDeviceId() const;
    
    // Predicates
    bool isLocal() const { return hostname == "localhost"; }
    bool sameNuma(const GlobalDeviceAddress& other) const;
    bool sameHost(const GlobalDeviceAddress& other) const;
    bool isGPU() const;
};
```

### Address Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLUSTER                                                                      │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ MACHINE (hostname: localhost)                                            │ │
│ │ ┌───────────────────────────────┐ ┌───────────────────────────────────┐ │ │
│ │ │ NUMA NODE 0 (socket 0)        │ │ NUMA NODE 1 (socket 1)            │ │ │
│ │ │ ┌─────────┐ ┌─────────┐       │ │ ┌─────────┐ ┌─────────┐          │ │ │
│ │ │ │ cuda:0  │ │ rocm:0  │ cpu:0 │ │ │ cuda:0  │ │ rocm:0  │ cpu:0    │ │ │
│ │ │ └─────────┘ └─────────┘       │ │ └─────────┘ └─────────┘          │ │ │
│ │ │ (MPI Rank 0)                  │ │ (MPI Rank 1)                      │ │ │
│ │ └───────────────────────────────┘ └───────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Convention**: One MPI rank per NUMA node (socket). `cuda:0` on NUMA 0 and `cuda:0` on NUMA 1 are **different physical GPUs**.

### Backend Selection from Addresses

```cpp
CollectiveBackendType selectBackend(const std::vector<GlobalDeviceAddress>& devices) {
    // Same host, same NUMA, same GPU vendor → NCCL or RCCL
    // Same host, same NUMA, mixed GPU vendors → PCIeBAR
    // Same host, different NUMA → PCIeBAR or UPI
    // Different hosts → MPI
}
```

---

## Phase 0: CLI and Configuration (6 days)

### Goal

Design and implement the full CLI option taxonomy with NUMA-aware device addressing.

### CLI Option Categories

#### Basic Execution
```bash
llaminar2 -m model.gguf -p "Hello" -n 50    # Single device, auto-detect
```

#### Tensor Parallelism
```bash
# TP Scope: LOCAL (single rank, multiple devices)
llaminar2 -m model.gguf --tp 2 --tp-scope local -p "Hello"

# TP Scope: GLOBAL (multiple MPI ranks)
mpirun -np 2 llaminar2 -m model.gguf --tp 2 --tp-scope global -p "Hello"

# TP Scope: HYBRID (both)
mpirun -np 2 llaminar2 -m model.gguf --tp-local 2 --tp-global 2 -p "Hello"

# Fine-grained: explicit devices with proportional weights
llaminar2 -m model.gguf --tp 2 --tp-scope local \
  --tp-devices 0:cuda:0,0:rocm:0 --tp-weights 0.73,0.27 -p "Hello"
```

#### Pipeline Parallelism
```bash
# Basic PP (equal layer split)
mpirun -np 2 llaminar2 -m model.gguf --pp 2 -p "Hello"

# Explicit layer assignment
mpirun -np 2 llaminar2 -m model.gguf --pp 2 \
  --pp-layers 0:0-13,1:14-27 -p "Hello"

# Combined PP + TP
mpirun -np 4 llaminar2 -m model.gguf --pp 2 --tp 2 -p "Hello"
```

#### Device Assignment
```bash
# Explicit device for this rank
llaminar2 -m model.gguf --device 0:cuda:0 -p "Hello"

# Per-rank device mapping
mpirun -np 2 llaminar2 -m model.gguf \
  --device-map 0=0:cuda:0,1=1:rocm:0 -p "Hello"

# Full global address
mpirun -np 4 llaminar2 -m model.gguf \
  --device-map 0=localhost:0:cuda:0,1=localhost:1:cuda:0,2=node1:0:cuda:0,3=node1:1:cuda:0 \
  -p "Hello"
```

#### Named TP Domains with PP Composition

For complex scenarios where multiple TP domains feed into a pipeline:

```bash
# Define named TP domains
--define-domain <name>:<devices>[:<backend>]

# Assign domains to PP stages with layer ranges
--pp-stage <stage_id>:<domain_name>:<first_layer>-<last_layer>
```

**Scenario 7 Full Complexity** - Dual socket, GPUs + CPUs all working:

```bash
# Socket 0 GPUs (TP via PCIeBAR), Socket 1 GPUs (TP via PCIeBAR), 
# Both CPUs (TP via UPI), all pipelined

llaminar2 -m model.gguf \
  --define-domain gpu0:0:cuda:0,0:rocm:0:pciebar \
  --define-domain gpu1:1:cuda:0,1:rocm:0:pciebar \
  --define-domain cpus:0:cpu:0,1:cpu:0:upi \
  --pp-stage 0:gpu0:0-9 \
  --pp-stage 1:gpu1:10-19 \
  --pp-stage 2:cpus:20-27 \
  -p "Hello"
```

This expresses:
```
PP(
  stage0: TP(0:cuda:0, 0:rocm:0) → layers 0-9   [PCIeBAR backend]
  stage1: TP(1:cuda:0, 1:rocm:0) → layers 10-19 [PCIeBAR backend]
  stage2: TP(0:cpu:0, 1:cpu:0)   → layers 20-27 [UPI backend]
)
```

**Simpler variant** - Just GPUs pipelined, no CPU participation:

```bash
llaminar2 -m model.gguf \
  --define-domain gpu0:0:cuda:0,0:rocm:0:pciebar \
  --define-domain gpu1:1:cuda:0,1:rocm:0:pciebar \
  --pp-stage 0:gpu0:0-13 \
  --pp-stage 1:gpu1:14-27 \
  -p "Hello"
```

**Single domain, no PP** (degenerates to pure TP):

```bash
# Equivalent to --tp 2 --tp-scope local --tp-devices 0:cuda:0,0:rocm:0
llaminar2 -m model.gguf \
  --define-domain all:0:cuda:0,0:rocm:0:pciebar \
  -p "Hello"
```

#### Domain Definition Format

```
--define-domain <name>:<device1>,<device2>[,...][:backend]

name     = alphanumeric identifier (e.g., gpu0, cpus, fast_tier)
device   = global address (hostname:numa:type:ordinal or shorthand)
backend  = pciebar | nccl | rccl | upi | mpi | auto (optional, default: auto)
```

#### PP Stage Format

```
--pp-stage <stage_id>:<domain_name>:<first_layer>-<last_layer>

stage_id    = integer (0, 1, 2, ...)
domain_name = must match a --define-domain name
layers      = inclusive range (e.g., 0-9 means layers 0 through 9)
```

#### Hybrid CPU/GPU (Simple Case)
```bash
# Last 4 layers on CPU (single domain, automatic PP insertion)
llaminar2 -m model.gguf --device cuda:0 --cpu-layers 4 -p "Hello"
```

#### Introspection
```bash
llaminar2 --show-numa                           # Print NUMA topology
llaminar2 -m model.gguf --dry-run              # Show plan without executing
llaminar2 -m model.gguf --explain-placement    # Log placement decisions
mpirun -np 4 llaminar2 -m model.gguf --show-pp-topology  # Show PP layout
```

### Configuration File Support

For complex scenarios like Scenario 7, YAML is cleaner than CLI:

```yaml
# scenario7_full.yaml
# Dual-socket machine with GPUs + CPUs all participating
#
# Physical layout:
#   Socket 0: cuda:0, rocm:0, cpu:0
#   Socket 1: cuda:0, rocm:0, cpu:0
#
# Execution:
#   PP Stage 0 (layers 0-9):   Socket 0 GPUs in TP
#   PP Stage 1 (layers 10-19): Socket 1 GPUs in TP
#   PP Stage 2 (layers 20-27): Both CPUs in TP (slower, but utilizes all hardware)

orchestration:
  domains:
    - name: gpu0
      devices:
        - 0:cuda:0    # RTX 3090 on socket 0
        - 0:rocm:0    # MI50 on socket 0
      weights: [0.6, 0.4]  # 3090 is faster
      backend: pciebar     # Cross-vendor requires PCIeBAR
      
    - name: gpu1
      devices:
        - 1:cuda:0    # RTX 3090 on socket 1
        - 1:rocm:0    # MI50 on socket 1
      weights: [0.6, 0.4]
      backend: pciebar
      
    - name: cpus
      devices:
        - 0:cpu:0     # CPU on socket 0
        - 1:cpu:0     # CPU on socket 1
      weights: [0.5, 0.5]  # Symmetric
      backend: upi         # Cross-socket CPU uses UPI

  pipeline:
    - stage: 0
      domain: gpu0
      layers: [0, 9]       # Layers 0-9 on Socket 0 GPUs
      
    - stage: 1
      domain: gpu1
      layers: [10, 19]     # Layers 10-19 on Socket 1 GPUs
      
    - stage: 2
      domain: cpus
      layers: [20, 27]     # Layers 20-27 on CPUs (memory-bound anyway)
```

```bash
llaminar2 -m model.gguf --placement-config scenario7_full.yaml -p "Hello"

# Validate without running
llaminar2 -m model.gguf --placement-config scenario7_full.yaml --validate-only

# Dry run to see computed plan
llaminar2 -m model.gguf --placement-config scenario7_full.yaml --dry-run
```

#### Simpler Configuration Examples

```yaml
# simple_tp.yaml - 2-way LOCAL TP on single socket
orchestration:
  domains:
    - name: default
      devices: [0:cuda:0, 0:cuda:1]
```

```yaml
# cross_vendor_tp.yaml - CUDA + ROCm on same socket
orchestration:
  domains:
    - name: default
      devices: [0:cuda:0, 0:rocm:0]
      weights: [0.6, 0.4]
      backend: pciebar
```

```yaml
# dual_socket_pp.yaml - Simple PP across sockets, no intra-socket TP
orchestration:
  domains:
    - name: sock0
      devices: [0:cuda:0]
    - name: sock1
      devices: [1:cuda:0]
  
  pipeline:
    - stage: 0
      domain: sock0
      layers: [0, 13]
    - stage: 1
      domain: sock1
      layers: [14, 27]
```

### OrchestrationConfig Structure

```cpp
struct OrchestrationConfig {
    // Introspection
    bool dry_run = false;
    bool explain_placement = false;
    bool show_topology = false;
    
    // Device assignment (simple cases)
    DeviceAssignmentMode device_mode = DeviceAssignmentMode::AUTO;
    GlobalDeviceAddress device_for_this_rank;
    std::vector<RankDeviceMapping> device_map;
    
    // Named Domain Definitions (--define-domain)
    struct DomainDefinition {
        std::string name;                          // "gpu0", "cpus", etc.
        std::vector<GlobalDeviceAddress> devices;  // Devices in this TP domain
        std::vector<float> weights;                // Proportional work split
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
    };
    std::vector<DomainDefinition> domain_definitions;
    
    // PP Stage Definitions (--pp-stage)
    struct PPStageDefinition {
        int stage_id;
        std::string domain_name;   // References a DomainDefinition
        int first_layer;
        int last_layer;
    };
    std::vector<PPStageDefinition> pp_stage_definitions;
    
    // Simple TP options (when not using named domains)
    int tp_degree = 1;
    TPScope tp_scope = TPScope::AUTO;
    std::vector<GlobalDeviceAddress> tp_devices;
    std::vector<float> tp_weights;
    int tp_local_degree = 1;
    int tp_global_degree = 1;
    
    // Simple PP options (when not using named domains)
    int pp_degree = 1;
    PPSplitMode pp_split = PPSplitMode::EQUAL;
    
    // Layer placement (simple hybrid CPU/GPU)
    int cpu_layers = 0;
    
    // Backend selection
    CollectiveBackendType default_backend = CollectiveBackendType::AUTO;
    
    // Config file
    std::string config_file_path;
    
    // Validation and resolution
    bool validate(const MPIContext* mpi_ctx) const;
    OrchestrationConfig resolve(const MPIContext* mpi_ctx, 
                                const ClusterInventory& cluster) const;
    
    // Check if using named domain mode vs simple mode
    bool usesNamedDomains() const { 
        return !domain_definitions.empty(); 
    }
};
```

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 0.1 | Create `GlobalDeviceAddress` class with parsing/serialization | 1 day |
| 0.2 | Design `OrchestrationConfig` struct with `DomainDefinition` and `PPStageDefinition` | 0.5 days |
| 0.3 | Implement CLI parser for `--define-domain` and `--pp-stage` | 1 day |
| 0.4 | Implement CLI parser for simple options (--tp, --pp, --device) | 0.5 days |
| 0.5 | Implement YAML config file loader (domains + pipeline schema) | 1 day |
| 0.6 | Implement `--dry-run` and `--explain-placement` | 0.5 days |
| 0.7 | Implement `--show-topology` and `--show-numa` | 0.5 days |
| 0.8 | Wire `OrchestrationConfig` into `InferenceRunnerFactory` | 1 day |
| 0.9 | Unit tests for CLI parsing and address parsing | 0.5 days |

**Phase 0 Total: 6.5 days**

### Success Criteria
- [ ] All CLI options parse correctly (including `--define-domain` and `--pp-stage`)
- [ ] `--dry-run` shows placement plan without executing
- [ ] `--show-numa` prints NUMA topology
- [ ] Config file loading works for YAML format (domains + pipeline schema)
- [ ] Scenario 7 full config validates successfully
- [ ] Existing tests still pass

---

## Phase 1: Plan-to-Graph Translation (5 days)

### Goal

Wire `HeterogeneousMultiDomainStrategy` output into graph configuration.

### Current Gap

```cpp
// HeterogeneousMultiDomainStrategy PRODUCES HeterogeneousPlan
// But InferenceRunnerFactory IGNORES it and hardcodes:
graph_config.default_device = device;  // Single device
graph_config.local_n_heads = n_heads / world_size;  // MPI-based only
```

### Implementation

```cpp
// In InferenceRunnerFactory::createInferenceRunner()

// Step 1: Build PlacementInput
PlacementInput input = buildPlacementInput(model_config, cluster_inventory);

// Step 2: Run strategy
auto strategy = PlacementStrategyFactory::create(orch_config.strategy);
HeterogeneousPlan cluster_plan = strategy->compute(input);

// Step 3: Extract this rank's execution plan
RankExecutionPlan my_plan = cluster_plan.toRankExecutionPlan(mpi_ctx->rank());

// Step 4: Configure graph from plan
Qwen2GraphConfig graph_config;
graph_config.execution_plan = my_plan;
graph_config.first_layer = my_plan.first_layer;
graph_config.last_layer = my_plan.last_layer;
graph_config.has_embedding = my_plan.has_embedding;
graph_config.has_lm_head = my_plan.has_lm_head;
// ...
```

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 1.1 | Add `RankExecutionPlan` to `Qwen2GraphConfig` | 0.5 days |
| 1.2 | Create `PlacementInput` builder from model + cluster | 1 day |
| 1.3 | Wire strategy selection in `InferenceRunnerFactory` | 1 day |
| 1.4 | Implement `HeterogeneousPlan::toRankExecutionPlan()` | 1 day |
| 1.5 | Implement `configureFromExecutionPlan()` | 1 day |
| 1.6 | Unit tests for plan-to-config translation | 0.5 days |

**Phase 1 Total: 5 days**

### Success Criteria
- [ ] `HeterogeneousMultiDomainStrategy::compute()` called during setup
- [ ] `Qwen2GraphConfig` populated from plan
- [ ] `--explain-placement` shows strategy output
- [ ] Existing single-device inference still works

---

## Phase 2: Multi-Device Graph Building (4.5 days)

### Goal

Modify `Qwen2Graph` to support per-stage device assignment.

### Current Gap

```cpp
// Current: ALL stages get same device
void Qwen2Graph::buildLayerGraph(int layer_idx) {
    params.device_id = config_.default_device;  // Same for all!
}
```

### Implementation

```cpp
// In Qwen2Graph::buildLayerGraph()

// Query device for this layer from execution plan
GlobalDeviceAddress layer_device = config_.execution_plan.deviceForLayer(layer_idx);
DeviceId local_device = layer_device.toLocalDeviceId();

// Get domain-specific head count
int local_heads = config_.execution_plan.headsForDevice(layer_device);

AttnNormStage::Params norm_params;
norm_params.device_id = local_device;   // Per-layer device
norm_params.n_heads = local_heads;      // Domain-specific
```

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 2.1 | Add `LayerPlacementConfig` to `Qwen2GraphConfig` | 0.5 days |
| 2.2 | Modify `buildLayerGraph()` to query device per layer | 1 day |
| 2.3 | Support variable `local_n_heads` based on domain | 1 day |
| 2.4 | Handle device transitions (CPU→GPU handoff) | 1 day |
| 2.5 | Unit tests for multi-device graph | 1 day |

**Phase 2 Total: 4.5 days**

### Success Criteria
- [ ] Layers can be assigned to different devices
- [ ] `--cpu-layers 4` places last 4 layers on CPU
- [ ] Graph executes correctly across device boundaries
- [ ] Coherence handles CPU↔GPU transitions

---

## Phase 3: Pipeline Parallelism Integration (6.5 days)

### Goal

Wire `PipelineParallelConfig` into graph building and insert Send/Recv stages.

### Current Gap

`PipelineParallelConfig` exists and is tested (32 tests) but:
- `Qwen2Graph` builds ALL layers regardless of rank
- `WeightManager` loads ALL weights regardless of rank
- No Send/Recv stages are inserted

### Implementation

```cpp
// In Qwen2Graph::buildFullForwardGraph()

const auto& plan = config_.execution_plan;

// Receive from previous stage (if not first)
if (plan.prev_rank.has_value()) {
    graph.addNode("recv_from_prev", 
        ComputeStageFactory::createReceiveActivations({
            .src_rank = *plan.prev_rank,
            .buffer = input_buffer,
            .mpi_ctx = mpi_ctx_
        }), DeviceId::cpu());
}

// Build ONLY my layers
for (int layer = plan.first_layer; layer <= plan.last_layer; ++layer) {
    buildLayerGraph(layer);
}

// Send to next stage (if not last)
if (plan.next_rank.has_value()) {
    graph.addNode("send_to_next",
        ComputeStageFactory::createSendActivations({
            .dest_rank = *plan.next_rank,
            .buffer = output_buffer,
            .mpi_ctx = mpi_ctx_
        }), DeviceId::cpu());
}
```

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 3.1 | Modify `buildFullForwardGraph()` to use layer range from plan | 1 day |
| 3.2 | Insert Send/Recv stages at PP boundaries | 1 day |
| 3.3 | Modify `WeightManager` for partial weight loading | 1.5 days |
| 3.4 | Handle embedding (first stage) and LM head (last stage) | 1 day |
| 3.5 | Integration test: 2-rank PP execution | 1.5 days |
| 3.6 | Handle async PP mode | 0.5 days |

**Phase 3 Total: 6.5 days**

### Success Criteria
- [ ] `--pp 2` splits layers between 2 MPI ranks
- [ ] Each rank loads only its assigned weights (~50% memory)
- [ ] Send/Recv correctly transfer activations
- [ ] End-to-end inference produces correct output with PP

---

## Phase 4: LOCAL Tensor Parallelism (7.5 days)

### Goal

Enable single-rank multi-device TP by decoupling TP degree from MPI world_size.

### Current Gap

```cpp
// In InferenceRunnerFactory:
if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded) {
    // Enable TP
} else {
    // world_size == 1 means NO TP, even with multiple GPUs!
}
```

### Implementation

```cpp
// New enablement check
bool enable_tp = (mpi_ctx && mpi_ctx->world_size() > 1)      // GLOBAL TP
              || (plan.tp_scope == TPScope::LOCAL &&          // LOCAL TP
                  plan.local_tp_devices.size() > 1);

if (enable_tp) {
    if (plan.tp_scope == TPScope::LOCAL) {
        // Shard by device index within rank
        configureLocalTP(graph_config, plan);
    } else {
        // Shard by MPI rank
        configureGlobalTP(graph_config, plan, mpi_ctx);
    }
}
```

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 4.1 | Define `LocalTPConfig` in `RankExecutionPlan` | 0.5 days |
| 4.2 | Modify TP enablement check | 0.5 days |
| 4.3 | Implement `configureLocalTP()` | 1.5 days |
| 4.4 | Modify `WeightManager` to shard by device index | 1 day |
| 4.5 | Build per-device subgraphs with local allreduce | 1.5 days |
| 4.6 | Wire `CollectiveContext` for LOCAL scope | 1 day |
| 4.7 | Integration test: single-rank 2-GPU TP | 1.5 days |

**Phase 4 Total: 7.5 days**

### Success Criteria
- [ ] `--tp 2 --tp-scope local` works with single MPI rank
- [ ] Weights sharded correctly across local devices
- [ ] PCIeBAR/NCCL used for intra-rank collectives
- [ ] Inference produces correct output with LOCAL TP

---

## Phase 5: End-to-End Integration (9.5 days)

### Goal

Full Scenario 7 validation with all components working together.

### Scenario 7: Full Hardware Utilization

**Physical Setup**: Single machine, 2 sockets, each with CUDA GPU + ROCm GPU + CPU cores

**Execution Layout**:
```
PP Stage 0: TP(0:cuda:0, 0:rocm:0) → Layers 0-9    [Socket 0 GPUs, PCIeBAR]
PP Stage 1: TP(1:cuda:0, 1:rocm:0) → Layers 10-19  [Socket 1 GPUs, PCIeBAR]
PP Stage 2: TP(0:cpu:0, 1:cpu:0)   → Layers 20-27  [Both CPUs, UPI]
```

**Why This Layout**:
- Early layers (0-9): Compute-bound, benefit from fast GPU
- Middle layers (10-19): Also compute-bound, different socket's GPU
- Late layers (20-27): More memory-bound (KV cache dominates), CPUs can handle
- Result: All hardware utilized, no idle resources

### Scenario 7 Configuration

**CLI version**:
```bash
llaminar2 -m models/qwen2.5-7b-instruct-q4_0.gguf \
  --define-domain gpu0:0:cuda:0,0:rocm:0:pciebar \
  --define-domain gpu1:1:cuda:0,1:rocm:0:pciebar \
  --define-domain cpus:0:cpu:0,1:cpu:0:upi \
  --pp-stage 0:gpu0:0-9 \
  --pp-stage 1:gpu1:10-19 \
  --pp-stage 2:cpus:20-27 \
  -p "Hello, world!"
```

**YAML version** (`scenario7_full.yaml`):
```yaml
orchestration:
  domains:
    - name: gpu0
      devices: [0:cuda:0, 0:rocm:0]
      weights: [0.6, 0.4]
      backend: pciebar
    - name: gpu1
      devices: [1:cuda:0, 1:rocm:0]
      weights: [0.6, 0.4]
      backend: pciebar
    - name: cpus
      devices: [0:cpu:0, 1:cpu:0]
      backend: upi

  pipeline:
    - stage: 0
      domain: gpu0
      layers: [0, 9]
    - stage: 1
      domain: gpu1
      layers: [10, 19]
    - stage: 2
      domain: cpus
      layers: [20, 27]
```

### Simplified Scenario 7 Variants

**Variant A**: GPUs only, no CPU participation
```bash
llaminar2 -m model.gguf \
  --define-domain gpu0:0:cuda:0,0:rocm:0:pciebar \
  --define-domain gpu1:1:cuda:0,1:rocm:0:pciebar \
  --pp-stage 0:gpu0:0-13 \
  --pp-stage 1:gpu1:14-27 \
  -p "Hello"
```

**Variant B**: Single socket, all devices (no PP, pure TP)
```bash
llaminar2 -m model.gguf \
  --define-domain all:0:cuda:0,0:rocm:0:pciebar \
  -p "Hello"
```

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 5.1 | Create Scenario 7 config file | 0.5 days |
| 5.2 | End-to-end test: 2-node, 4-GPU inference | 2 days |
| 5.3 | Parity test: Scenario 7 vs PyTorch reference | 2 days |
| 5.4 | Performance benchmarking | 1 day |
| 5.5 | Bug fixes and stabilization | 3 days |
| 5.6 | Documentation update | 1 day |

**Phase 5 Total: 9.5 days**

### Success Criteria
- [ ] Scenario 7 config loads and validates
- [ ] 2-node inference executes without crashes
- [ ] Output matches PyTorch reference (within tolerance)
- [ ] Performance meets target (>10 tok/s decode)
- [ ] All 365+ existing tests still pass

---

## Phase 6: GlobalDeviceAddress Migration (8 days)

### Goal

Migrate the entire codebase to use `GlobalDeviceAddress` consistently.

### Files to Create

| File | Purpose |
|------|---------|
| `src/v2/backends/GlobalDeviceAddress.h` | Main header |
| `src/v2/backends/GlobalDeviceAddress.cpp` | Implementation |
| `tests/v2/unit/Test__GlobalDeviceAddress.cpp` | Unit tests |

### Files to Modify

| File | Changes |
|------|---------|
| `src/v2/backends/DeviceId.h` | Add `fromGlobalAddress()` |
| `src/v2/config/TPDomain.h` | Change `devices` to `vector<GlobalDeviceAddress>` |
| `src/v2/execution/DeviceInventory.h` | Update `DeviceInfo`, `RankInventory` |
| `src/v2/execution/PlacementPlan.h` | Update `LayerPlacement` |
| `src/v2/execution/CollectiveContext.h` | Update device tracking |
| `src/v2/execution/placement/HeterogeneousMultiDomainStrategy.h` | Update domain assignments |
| `src/v2/utils/MPITopology.h` | Update `RankPlacement` |
| `src/v2/collective/BackendRouter.h` | Add address-based backend selection |

### Tasks

| ID | Task | Effort |
|----|------|--------|
| 6.1 | Create `GlobalDeviceAddress` class | 1 day |
| 6.2 | Update `DeviceInfo` to use `GlobalDeviceAddress` | 0.5 days |
| 6.3 | Update `RankInventory` and `ClusterInventory` | 0.5 days |
| 6.4 | Update `TPDomain` | 0.5 days |
| 6.5 | Update `LayerPlacement` and `PlacementPlan` | 1 day |
| 6.6 | Update `CollectiveContext` and `BackendRouter` | 1 day |
| 6.7 | Update CLI parser to produce `GlobalDeviceAddress` | 0.5 days |
| 6.8 | Update YAML/JSON config parser | 0.5 days |
| 6.9 | Update MPI serialization | 0.5 days |
| 6.10 | Unit tests | 0.5 days |
| 6.11 | Migration of existing tests | 1.5 days |

**Phase 6 Total: 8 days**

---

## Timeline Summary

| Phase | Description | Effort | Cumulative |
|-------|-------------|--------|------------|
| **0** | CLI and Configuration Architecture | 6.5 days | 6.5 days |
| **1** | Plan-to-Graph Translation | 5 days | 11.5 days |
| **2** | Multi-Device Graph Building | 4.5 days | 16 days |
| **3** | Pipeline Parallelism Integration | 6.5 days | 22.5 days |
| **4** | LOCAL Tensor Parallelism | 7.5 days | 30 days |
| **5** | End-to-End Integration | 9.5 days | 39.5 days |
| **6** | GlobalDeviceAddress Migration | 8 days | 47.5 days |

**Total: ~47.5 days (~10 weeks)**

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Coherence bugs at device boundaries | High | High | Extensive testing, transfer tracing |
| PP deadlocks from Send/Recv ordering | Medium | High | Careful ordering, timeout detection |
| Memory fragmentation with partial loading | Medium | Medium | Pre-allocation, memory pooling |
| Performance regression from overhead | Medium | Medium | Profiling each phase, optimization |
| Breaking existing single-device path | Low | High | Backward compatibility tests |
| GlobalDeviceAddress migration breaks tests | Medium | Medium | Gradual migration, dual fields |

---

## Definition of Done

The project is complete when:

1. ✅ All 365+ existing tests still pass
2. ✅ `--dry-run` shows valid placement plans for all scenarios
3. ✅ Scenario 7 config executes end-to-end
4. ✅ Parity test passes (output matches PyTorch within tolerance)
5. ✅ Performance meets target (>10 tok/s decode on Scenario 7)
6. ✅ Documentation updated with new CLI options
7. ✅ At least one real-world benchmark (7B model on heterogeneous hardware)
8. ✅ `GlobalDeviceAddress` used consistently in all orchestration code

---

## Current vs Target Architecture

### Current (365 tests unused)

```
┌───────────────────────────────────────────────┐
│           llaminar2 CLI                        │
│  (limited: -m, -p, -n, --mpi-procs)           │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│         InferenceRunnerFactory                 │
│  - Loads ALL weights                          │
│  - Single default_device                      │
│  - TP only if world_size > 1                  │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│              Qwen2Graph                        │
│  - ALL 28 layers built                        │
│  - ALL stages use same device_id              │
│  - No PP stages                               │
└───────────────────────────────────────────────┘

   ╔═════════════════════════════════════════╗
   ║          UNUSED (365 tests!)            ║
   ║  • ClusterInventory                     ║
   ║  • HeterogeneousMultiDomainStrategy     ║
   ║  • TPDomain / MultiDomainTPConfig       ║
   ║  • PipelineParallelConfig               ║
   ║  • Send/ReceiveActivationsStage         ║
   ║  • NodeTopology / NUMAAllocator         ║
   ╚═════════════════════════════════════════╝
```

### Target (Everything Wired)

```
┌───────────────────────────────────────────────┐
│           llaminar2 CLI                        │
│  --tp 2 --pp 2 --tp-scope local               │
│  --tp-devices 0:cuda:0,0:rocm:0               │
│  --placement-config cluster.yaml              │
│  --dry-run --explain-placement                │
└───────────────────────────────────────────────┘
                      │
           OrchestrationConfig
                      │
                      ▼
┌───────────────────────────────────────────────┐
│          ClusterInventory                      │
│  - Detect devices per NUMA node               │
│  - Exchange via MPI_Allgather                 │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│    HeterogeneousMultiDomainStrategy           │
│  - Compute optimal placement                  │
│  - Generate HeterogeneousPlan                 │
└───────────────────────────────────────────────┘
                      │
         plan.toRankExecutionPlan(rank)
                      │
                      ▼
┌───────────────────────────────────────────────┐
│         InferenceRunnerFactory                 │
│  - Configure from RankExecutionPlan           │
│  - Load ONLY assigned weights                 │
│  - Set up LOCAL or GLOBAL TP                  │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│              Qwen2Graph                        │
│  - Build ONLY assigned layers (PP)            │
│  - Per-stage device assignment                │
│  - Insert Send/Recv at PP boundaries          │
│  - Variable local_n_heads per domain          │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│          CollectiveContext                     │
│  - Route to correct backend per domain        │
│  - PCIeBAR for cross-vendor GPU               │
│  - NCCL/RCCL for same-vendor GPU              │
│  - MPI for cross-rank                         │
└───────────────────────────────────────────────┘
```

---

*End of Consolidated Plan*
