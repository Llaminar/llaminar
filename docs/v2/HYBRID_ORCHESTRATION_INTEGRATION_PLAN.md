# Hybrid Orchestration Integration Plan

> **Project**: Wire 365+ tested components into actual inference path
> **Goal**: Make Scenario 7 (2-Node Heterogeneous Pipeline Cluster) actually work
> **Target**: Full hybrid PP + TP inference with cross-vendor GPU support

## Executive Summary

We have extensive infrastructure (365 tests passing) for:
- Device inventory and capability exchange
- Placement strategies (HeterogeneousMultiDomainStrategy)
- TP domains (GPU_INTRA_RANK, CPU_CROSS_RANK)
- Pipeline parallelism configuration
- Send/Receive activation stages
- PCIeBAR cross-vendor backend (25μs latency!)
- NUMA-aware allocation

**The problem**: None of it is wired into the actual inference path. `InferenceRunnerFactory` → `GraphOrchestrator` → `Qwen2Graph` bypasses everything.

**This plan**: Connect the dots in 6 phases over ~4 weeks.

---

## Architectural Layers

The codebase has a clear separation into three layers:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: ENTRY POINT                                                           │
│  ═══════════════════                                                            │
│                                                                                 │
│  main.cpp                                                                       │
│    • CLI argument parsing                                                       │
│    • MPI bootstrap (--mpi-procs, hostfile)                                      │
│    • Configuration loading (YAML/JSON)                                          │
│    • Creates OrchestrationConfig                                                │
│    • Hands off to InferenceRunnerFactory                                        │
│                                                                                 │
│  Knows: User intent, file paths, CLI options                                    │
│  Doesn't know: Cluster topology, device capabilities, model architecture        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ OrchestrationConfig
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: CLUSTER ORCHESTRATION                                                 │
│  ══════════════════════════════                                                 │
│                                                                                 │
│  Runs: Once at startup (rank 0 computes, all ranks get identical result)        │
│                                                                                 │
│  Components:                                                                    │
│    • ClusterInventory - Device discovery, MPI_Allgather exchange                │
│    • MPITopology - Rank placement, node detection                               │
│    • NodeTopology - NUMA detection, socket enumeration                          │
│    • PlacementStrategy - Compute optimal placement (HeterogeneousMultiDomain)   │
│    • HeterogeneousPlan - PP stages, TP domains, layer assignments               │
│                                                                                 │
│  Uses: GlobalDeviceAddress (hostname:numa:type:ordinal)                         │
│                                                                                 │
│  Produces: RankExecutionPlan (what THIS rank should do)                         │
│    • Which layers this rank owns (PP)                                           │
│    • Which TP domains this rank participates in                                 │
│    • Which devices to use (GlobalDeviceAddress list)                            │
│    • Weight sharding fractions                                                  │
│    • Collective backend assignments                                             │
│                                                                                 │
│  Knows: Entire cluster topology, all capabilities, global optimization          │
│  Doesn't know: How to execute a single layer, kernel implementations            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ RankExecutionPlan
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: RANK-LOCAL EXECUTION                                                  │
│  ════════════════════════════                                                   │
│                                                                                 │
│  Runs: Independently on each MPI rank                                           │
│                                                                                 │
│  Components:                                                                    │
│    • WeightManager - Load ONLY assigned layers/shards from GGUF                 │
│    • GraphOrchestrator - Build compute graph for assigned layers                │
│    • Qwen2Graph - Layer implementations, stage creation                         │
│    • GraphExecutor - Execute stages, manage device memory                       │
│    • CollectiveContext - Execute collectives via BackendRouter                  │
│    • KernelFactory - Create device-specific kernels                             │
│                                                                                 │
│  Uses: DeviceId (type:ordinal) for kernel calls                                 │
│        GlobalDeviceAddress converted to DeviceId at this layer                  │
│                                                                                 │
│  Receives from Layer 2:                                                         │
│    • "You own layers 0-13"                                                      │
│    • "Use devices [localhost:0:cuda:0, localhost:0:rocm:0]"                      │
│    • "You're in TP domain 0 with 2-way parallelism"                             │
│    • "Use PCIeBAR backend for domain 0 collectives"                             │
│    • "Send activations to rank 2 after layer 13"                                │
│                                                                                 │
│  Knows: How to load weights, build graphs, execute kernels, do collectives      │
│  Doesn't know: Why this rank got these layers, what other ranks are doing       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Layer Boundaries: The Contract

The key abstraction is **RankExecutionPlan** - the contract between Layer 2 and Layer 3:

```cpp
// src/v2/execution/RankExecutionPlan.h

/**
 * @brief Everything a rank needs to know to execute its portion of inference
 *
 * Produced by: Layer 2 (Cluster Orchestration)
 * Consumed by: Layer 3 (Rank-Local Execution)
 *
 * This is the ONLY thing that flows from global planning to local execution.
 * It contains no cluster-wide state - just "what should I do?"
 */
struct RankExecutionPlan {
    // =========================================================================
    // Identity
    // =========================================================================
    int my_rank;                    ///< This rank's MPI rank ID
    int world_size;                 ///< Total MPI ranks (for validation)
    std::string my_hostname;        ///< This rank's hostname
    int my_numa_node;               ///< This rank's primary NUMA node
    
    // =========================================================================
    // Pipeline Parallelism Assignment
    // =========================================================================
    int pp_stage;                   ///< Which PP stage this rank owns (-1 if no PP)
    int first_layer;                ///< First layer index (inclusive)
    int last_layer;                 ///< Last layer index (inclusive)
    bool owns_embedding;            ///< This rank runs embedding
    bool owns_lm_head;              ///< This rank runs LM head
    bool owns_final_norm;           ///< This rank runs final RMS norm
    
    // PP communication
    int send_activations_to_rank;   ///< Rank to send to (-1 if last stage)
    int recv_activations_from_rank; ///< Rank to receive from (-1 if first stage)
    
    // =========================================================================
    // Tensor Parallelism Assignment
    // =========================================================================
    struct TPDomainAssignment {
        int domain_id;
        TPDomainType type;          ///< GPU_INTRA_RANK, CPU_CROSS_RANK, etc.
        int local_rank_in_domain;   ///< Our rank within domain (0..domain_size-1)
        int domain_size;            ///< Total participants
        
        // Devices this rank contributes to this domain
        std::vector<GlobalDeviceAddress> my_devices;
        
        // Work fractions per device (sum to our share of domain work)
        std::vector<float> device_weights;
        
        // Backend to use for this domain's collectives
        CollectiveBackendType backend;
        
        // Communicator (set at runtime by CollectiveContext)
        MPI_Comm communicator = MPI_COMM_NULL;
    };
    
    std::vector<TPDomainAssignment> tp_domains;
    
    // =========================================================================
    // Device Assignment
    // =========================================================================
    GlobalDeviceAddress primary_device;    ///< Main compute device
    std::vector<GlobalDeviceAddress> all_devices;  ///< All devices for this rank
    
    // =========================================================================
    // Weight Loading Instructions
    // =========================================================================
    struct WeightLoadInstruction {
        std::string weight_name;           ///< e.g., "layers.5.attention.wq"
        GlobalDeviceAddress target_device; ///< Where to load it
        bool is_sharded;                   ///< Is this weight sharded?
        int shard_index;                   ///< If sharded, which shard (0..n-1)
        int total_shards;                  ///< If sharded, total shard count
        ShardDimension shard_dim;          ///< ROW or COLUMN
    };
    
    std::vector<WeightLoadInstruction> weights_to_load;
    
    // =========================================================================
    // Derived Queries
    // =========================================================================
    
    /// Check if this rank participates in TP
    bool hasTP() const { return !tp_domains.empty(); }
    
    /// Check if this rank participates in PP
    bool hasPP() const { return pp_stage >= 0; }
    
    /// Get number of layers this rank executes
    int layerCount() const { 
        return (first_layer >= 0 && last_layer >= first_layer) 
               ? (last_layer - first_layer + 1) : 0; 
    }
    
    /// Check if this rank executes a specific layer
    bool ownsLayer(int layer_idx) const {
        return layer_idx >= first_layer && layer_idx <= last_layer;
    }
    
    /// Get primary device as local DeviceId (for kernel calls)
    DeviceId primaryDeviceId() const {
        return primary_device.toLocalDeviceId();
    }
    
    /// Get TP domain by ID
    const TPDomainAssignment* getTPDomain(int domain_id) const;
};
```

### Why This Separation Matters

| Concern | Layer 2 (Cluster) | Layer 3 (Local) |
|---------|-------------------|-----------------|
| **Scope** | All ranks, all devices | This rank only |
| **Timing** | Once at startup | Every inference call |
| **Communication** | MPI_Allgather, MPI_Bcast | Collectives within domains |
| **Addressing** | GlobalDeviceAddress | DeviceId |
| **Optimization** | Global (minimize total latency) | Local (maximize this rank's throughput) |
| **Testing** | Mock ClusterInventory | Mock single-rank execution |
| **Debugging** | `--dry-run`, `--explain-placement` | `LLAMINAR_STAGE_DUMP`, profiling |

### Information Flow

```
                    User
                      │
                      │ CLI args, config file
                      ▼
              ┌───────────────┐
              │   main.cpp    │  Layer 1
              └───────────────┘
                      │
                      │ OrchestrationConfig
                      ▼
              ┌───────────────┐
              │   Factory     │
              │   creates     │
              └───────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Rank 0  │  │ Rank 1  │  │ Rank N  │   (all ranks)
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        └─────────┬──┴────────────┘
                  │
                  │ MPI_Allgather (device capabilities)
                  ▼
         ┌────────────────────┐
         │  ClusterInventory  │  Layer 2
         │  (identical on     │  (runs on all ranks,
         │   all ranks)       │   produces identical result)
         └────────────────────┘
                  │
                  │ PlacementStrategy::compute()
                  ▼
         ┌────────────────────┐
         │  HeterogeneousPlan │
         │  (global view)     │
         └────────────────────┘
                  │
                  │ extractPlanForRank(my_rank)
                  ▼
         ┌────────────────────┐
         │  RankExecutionPlan │  THE CONTRACT
         │  (per-rank view)   │
         └────────────────────┘
                  │
        ┌─────────┼─────────────┐
        ▼         ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Rank 0  │  │ Rank 1  │  │ Rank N  │   Layer 3
   │ loads   │  │ loads   │  │ loads   │   (independent execution)
   │ layers  │  │ layers  │  │ layers  │
   │ 0-13    │  │ 14-27   │  │ ...     │
   └─────────┘  └─────────┘  └─────────┘
```

---

## Table of Contents

- [Phase 0: CLI and Configuration Architecture](#phase-0-cli-and-configuration-architecture)
- [Phase 1: Plan-to-Graph Translation](#phase-1-plan-to-graph-translation)
- [Phase 2: Multi-Device Graph Building](#phase-2-multi-device-graph-building)
- [Phase 3: Pipeline Parallelism Integration](#phase-3-pipeline-parallelism-integration)
- [Phase 4: LOCAL Tensor Parallelism](#phase-4-local-tensor-parallelism)
- [Phase 5: End-to-End Integration](#phase-5-end-to-end-integration)
- [Appendix A: Current vs Target Architecture](#appendix-current-vs-target-architecture)
- [Appendix B: Unified Global Device Addressing](#appendix-b-unified-global-device-addressing)
- [Appendix C: Source Files Affected](#appendix-c-source-files-affected-by-globaldeviceaddress)

---

## Phase 0: CLI and Configuration Architecture

### Objective

Design a cohesive command-line interface that provides:
1. **Sensible defaults** - Auto-detect everything when possible
2. **Coarse control** - Simple flags for common configurations
3. **Fine-grained control** - Override any aspect of orchestration
4. **Configuration files** - Complex setups via YAML/JSON
5. **Introspection** - Dry-run mode to see what would happen

### 0.1 Design Principles

| Principle | Description |
|-----------|-------------|
| **Hierarchical** | Options follow a logical tree: `--tp-*`, `--pp-*`, `--device-*` |
| **Override cascade** | Auto-detect → Config file → CLI flags (later wins) |
| **Explicit > Implicit** | When specified, flags should be unambiguous |
| **Discoverable** | `--help-orchestration` shows all options with examples |
| **Debuggable** | `--dry-run` and `--explain-placement` show decisions |

### 0.2 CLI Option Taxonomy

```
llaminar2 [MODEL_OPTIONS] [ORCHESTRATION_OPTIONS] [INFERENCE_OPTIONS]

MODEL OPTIONS (existing):
  -m, --model <path>          Path to GGUF model file
  -c, --context-length <n>    Maximum context length (default: 2048)

INFERENCE OPTIONS (existing):
  -p, --prompt <text>         Input prompt
  -n, --n-predict <n>         Number of tokens to generate
  -t, --temperature <f>       Sampling temperature (0 = greedy)
  --benchmark                 Run benchmark mode

ORCHESTRATION OPTIONS (new):
  [Auto-Detection Control]
  [Tensor Parallelism]
  [Pipeline Parallelism]
  [Device Assignment]
  [Backend Selection]
  [Placement Strategy]
  [Debugging & Introspection]
```

### 0.3 Auto-Detection Control

```
--auto                        Enable full auto-detection (default)
--no-auto                     Disable auto-detection, require explicit config

--dry-run                     Show placement plan without executing
--explain-placement           Log detailed placement decisions
--show-topology               Print detected cluster topology and exit
```

**Behavior**:
- `--auto` (default): Detect GPUs, MPI ranks, NUMA topology, select strategy
- `--no-auto`: Fail if required options not specified
- `--dry-run`: Compute placement, print plan, exit without inference

### 0.4 Tensor Parallelism Options

#### TP Scope: LOCAL vs GLOBAL

The critical distinction for TP is **scope**:

| Scope | MPI Ranks | Devices | Communication | Use Case |
|-------|-----------|---------|---------------|----------|
| **LOCAL** | 1 | Multiple GPUs within rank | PCIeBAR/NCCL/RCCL | Single-node multi-GPU |
| **GLOBAL** | Multiple | 1 GPU per rank | MPI | Multi-node or MPI-based |
| **HYBRID** | Multiple | Multiple per rank | Both | Hierarchical TP |

#### Coarse Control

```
--tensor-parallel <n>         Enable TP with n-way parallelism
                              Aliases: --tp <n>
                              
                              Interpretation depends on --tp-scope:
                              - LOCAL:  n devices within THIS rank
                              - GLOBAL: n MPI ranks, 1 device each
                              - AUTO:   If world_size==1 and n GPUs available → LOCAL
                                        If world_size==n → GLOBAL

--tp-scope <scope>            TP communication scope
                              Values: auto, local, global
                              - auto:   Infer from world_size and device count
                              - local:  Intra-rank only (single MPI rank, multiple GPUs)
                              - global: Cross-rank (multiple MPI ranks)
```

**Examples**:
```bash
# LOCAL TP: Single rank with 2 GPUs (auto-detected)
llaminar2 -m model.gguf --tp 2 -p "Hello"
# If only 1 rank but 2 GPUs available → LOCAL scope

# Explicit LOCAL: Single rank, 2 GPUs
llaminar2 -m model.gguf --tp 2 --tp-scope local -p "Hello"

# GLOBAL TP: 2 MPI ranks, 1 GPU each
mpirun -np 2 llaminar2 -m model.gguf --tp 2 --tp-scope global -p "Hello"

# GLOBAL TP (implicit): world_size matches tp degree
mpirun -np 2 llaminar2 -m model.gguf --tp 2 -p "Hello"
# world_size==2 and tp==2 → GLOBAL scope
```

#### Fine-Grained Control

```
--tp-devices <spec>           Explicit device list for LOCAL TP
                              Format: <device>[,<device>]...
                              Scope: LOCAL only (devices within this rank)

--tp-ranks <spec>             Explicit rank list for GLOBAL TP
                              Format: <rank>[,<rank>]...
                              Scope: GLOBAL only (which MPI ranks participate)

--tp-weights <spec>           Work distribution weights (proportional TP)
                              Format: <weight>[,<weight>]...
                              Must match device/rank count

--tp-domain <spec>            Define a TP domain explicitly
                              Format: <domain_id>:<type>:<participants>:<backend>
                              Type: gpu_intra_rank, cpu_cross_rank, gpu_cross_rank
                              Participants: devices (LOCAL) or ranks (GLOBAL)
                              Backend: auto, nccl, rccl, pciebar, mpi, host
```

**Examples**:
```bash
# LOCAL TP: Explicit devices within this rank
llaminar2 -m model.gguf --tp-scope local --tp-devices cuda:0,rocm:0 -p "Hello"

# LOCAL TP: Proportional split (73% to CUDA, 27% to ROCm)
llaminar2 -m model.gguf --tp-scope local \
  --tp-devices cuda:0,rocm:0 --tp-weights 0.73,0.27 -p "Hello"

# GLOBAL TP: Only ranks 0 and 2 participate (skip rank 1)
mpirun -np 3 llaminar2 -m model.gguf --tp-scope global --tp-ranks 0,2 -p "Hello"

# Explicit domain with PCIeBAR backend (LOCAL, cross-vendor)
llaminar2 -m model.gguf \
  --tp-domain 0:gpu_intra_rank:cuda:0,rocm:0:pciebar -p "Hello"

# Explicit domain with MPI backend (GLOBAL, cross-rank)
mpirun -np 2 llaminar2 -m model.gguf \
  --tp-domain 0:gpu_cross_rank:rank0,rank1:mpi -p "Hello"
```

#### Hierarchical TP (HYBRID Scope)

For large clusters, combine LOCAL + GLOBAL:

```
--tp-local <n>                LOCAL TP degree (devices per rank)
--tp-global <n>               GLOBAL TP degree (ranks participating)
                              Total TP = tp_local × tp_global
```

**Example**:
```bash
# 4-way total TP: 2 GPUs per rank × 2 ranks
mpirun -np 2 llaminar2 -m model.gguf --tp-local 2 --tp-global 2 -p "Hello"
# Each rank: cuda:0 + rocm:0 (LOCAL TP via PCIeBAR)
# Cross-rank: rank 0's result + rank 1's result (GLOBAL TP via MPI)
# Effective: 4-way TP across 4 GPUs total
```

### 0.5 Pipeline Parallelism Options

#### PP and MPI Relationship

Pipeline Parallelism is inherently **cross-rank** (GLOBAL scope). Each PP stage is owned by one MPI rank. The relationship is:

| Concept | Default | Override |
|---------|---------|----------|
| PP degree | `--pp <n>` | Number of pipeline stages |
| Stage → Rank mapping | Stage N → Rank N | `--pp-rank-map` |
| Layer → Stage mapping | Equal split | `--pp-layers` |

#### Coarse Control

```
--pipeline-parallel <n>       Enable PP with n stages
                              Aliases: --pp <n>
                              Requires: MPI world_size >= n
                              Default mapping: stage N → rank N

--pp-split <mode>             How to split layers across PP stages
                              Values: equal, weighted, manual
                              - equal: Each stage gets n_layers/pp_degree layers
                              - weighted: Split by compute weight (attention-heavy early)
                              - manual: Requires --pp-layers
```

**Examples**:
```bash
# 2-way PP: layers 0-13 on rank 0, layers 14-27 on rank 1
mpirun -np 2 llaminar2 -m model.gguf --pp 2 -p "Hello"

# 4-way PP with equal split
mpirun -np 4 llaminar2 -m model.gguf --pp 4 --pp-split equal -p "Hello"
```

#### Fine-Grained Control

```
--pp-layers <spec>            Explicit layer assignment per PP stage
                              Format: <stage>:<first>-<last>[,...]
                              
--pp-rank-map <spec>          Map PP stages to MPI ranks
                              Format: <stage>:<rank>[,...]
                              Default: stage N → rank N
                              Use case: Non-contiguous rank assignment

--pp-async                    Enable asynchronous PP (overlap compute/communicate)
                              Default: synchronous
```

**Examples**:
```bash
# Explicit layer ranges
mpirun -np 2 llaminar2 -m model.gguf --pp 2 \
  --pp-layers 0:0-13,1:14-27 -p "Hello"

# Unequal split: more layers on faster rank
mpirun -np 2 llaminar2 -m model.gguf --pp 2 \
  --pp-layers 0:0-17,1:18-27 -p "Hello"

# Custom rank mapping: rank 1 is stage 0, rank 0 is stage 1
mpirun -np 2 llaminar2 -m model.gguf --pp 2 \
  --pp-rank-map 0:1,1:0 -p "Hello"

# Skip ranks: only ranks 0 and 2 participate in PP
mpirun -np 4 llaminar2 -m model.gguf --pp 2 \
  --pp-rank-map 0:0,1:2 -p "Hello"
```

#### Combining PP and TP

PP and TP can be combined. The total parallelism is `PP × TP`:

```bash
# 2-way PP × 2-way GLOBAL TP = 4 ranks total
# Ranks 0,1: PP stage 0 with 2-way TP
# Ranks 2,3: PP stage 1 with 2-way TP
mpirun -np 4 llaminar2 -m model.gguf --pp 2 --tp 2 -p "Hello"

# 2-way PP × 2-way LOCAL TP = 2 ranks, 2 GPUs each
# Rank 0: PP stage 0, LOCAL TP across cuda:0 + rocm:0
# Rank 1: PP stage 1, LOCAL TP across cuda:0 + rocm:0
mpirun -np 2 llaminar2 -m model.gguf --pp 2 --tp 2 --tp-scope local -p "Hello"
```

#### PP Topology Visualization

```bash
# See PP stage assignments
mpirun -np 4 llaminar2 -m model.gguf --pp 2 --tp 2 --show-pp-topology

# Example output:
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                     PIPELINE PARALLEL TOPOLOGY                         ║
# ╠═════════╦═════════════╦════════════════╦═══════════════════════════════╣
# ║ PP Stage║ MPI Ranks   ║ Layers         ║ TP Configuration              ║
# ╠═════════╬═════════════╬════════════════╬═══════════════════════════════╣
# ║    0    ║ 0, 1        ║ 0-13 (14)      ║ 2-way GLOBAL (ranks 0,1)      ║
# ║    1    ║ 2, 3        ║ 14-27 (14)     ║ 2-way GLOBAL (ranks 2,3)      ║
# ╠═════════╬═════════════╬════════════════╬═══════════════════════════════╣
# ║ Total   ║ 4 ranks     ║ 28 layers      ║ PP=2 × TP=2 = 4-way parallel  ║
# ╚═════════╩═════════════╩════════════════╩═══════════════════════════════╝
```

### 0.6 Device Assignment Options

#### The Addressing Hierarchy

Devices exist in a clear physical hierarchy:

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
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ MACHINE (hostname: node1)                                                │ │
│ │ ┌───────────────────────────────┐ ┌───────────────────────────────────┐ │ │
│ │ │ NUMA NODE 0 (socket 0)        │ │ NUMA NODE 1 (socket 1)            │ │ │
│ │ │ ┌─────────┐ ┌─────────┐       │ │ ┌─────────┐ ┌─────────┐          │ │ │
│ │ │ │ cuda:0  │ │ rocm:0  │ cpu:0 │ │ │ cuda:0  │ │ rocm:0  │ cpu:0    │ │ │
│ │ │ └─────────┘ └─────────┘       │ │ └─────────┘ └─────────┘          │ │ │
│ │ │ (MPI Rank 2)                  │ │ (MPI Rank 3)                      │ │ │
│ │ └───────────────────────────────┘ └───────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: MPI ranks map to NUMA nodes (one rank per socket), not to machines.

#### Global Device Address Format

Full unambiguous address:
```
<hostname>:<numa_node>:<device_type>:<device_ordinal>
```

| Component | Description | Example |
|-----------|-------------|---------|
| `hostname` | Physical machine name | `localhost`, `node1`, `gpu-server-03` |
| `numa_node` | NUMA node / socket index | `0`, `1` |
| `device_type` | Device type | `cuda`, `rocm`, `cpu` |
| `device_ordinal` | Device index within type on that NUMA node | `0`, `1`, `2` |

**Examples**:
```
localhost:0:cuda:0    # First CUDA GPU on NUMA node 0 of localhost
localhost:1:rocm:0    # First ROCm GPU on NUMA node 1 of localhost
node1:0:cuda:0        # First CUDA GPU on NUMA node 0 of node1
node1:1:cuda:0        # First CUDA GPU on NUMA node 1 of node1 (different physical GPU!)
```

#### Shorthand Addressing

For convenience, components can be omitted from the left:

| Shorthand | Expansion | Meaning |
|-----------|-----------|---------|
| `cuda:0` | `localhost:<local_numa>:cuda:0` | CUDA GPU 0 on my NUMA node |
| `0:cuda:0` | `localhost:0:cuda:0` | CUDA GPU 0 on NUMA node 0 of localhost |
| `1:rocm:0` | `localhost:1:rocm:0` | ROCm GPU 0 on NUMA node 1 of localhost |
| `node1:0:cuda:0` | (full form) | CUDA GPU 0 on NUMA node 0 of node1 |

**Note**: `<local_numa>` is determined by which MPI rank you are (rank N typically runs on NUMA node N % numa_nodes_per_machine).

#### Device Addressing Options

```
--device <addr>               Device for THIS rank
                              Format: [hostname:]<numa>:<type>:<ordinal>
                              Or shorthand: <type>:<ordinal>
                              
--device-type <type>          Device type preference for all ranks
                              Values: auto, cpu, cuda, rocm, any-gpu
                              Each rank uses its local device of this type

--device-map <spec>           Explicit per-rank device assignment
                              Format: <rank>=<addr>[,<rank>=<addr>]...
                              Address can be full or shorthand
                              
--device-per-rank <mode>      How to assign devices when world_size > 1
                              Values: 
                                auto       - Auto-detect based on topology
                                local-gpu  - Each rank uses GPU on its NUMA node
                                round-robin - Cycle through available GPUs
                                exclusive  - Each rank gets exclusive GPU
```

#### Single-Rank Examples

```bash
# All on CPU (local NUMA node's CPU)
llaminar2 -m model.gguf --device cpu:0 -p "Hello"

# CUDA GPU 0 on my local NUMA node
llaminar2 -m model.gguf --device cuda:0 -p "Hello"

# Explicit: CUDA GPU 0 on NUMA node 0
llaminar2 -m model.gguf --device 0:cuda:0 -p "Hello"

# Explicit: ROCm GPU on NUMA node 1
llaminar2 -m model.gguf --device 1:rocm:0 -p "Hello"
```

#### Multi-Rank Examples (MPI)

```bash
# Auto-assign: each rank uses GPU on its local NUMA node
mpirun -np 2 llaminar2 -m model.gguf --device-per-rank local-gpu -p "Hello"
# Rank 0 (NUMA 0) → 0:cuda:0
# Rank 1 (NUMA 1) → 1:cuda:0 (or 1:rocm:0 if that's what's there)

# Explicit mapping with shorthand
mpirun -np 2 llaminar2 -m model.gguf \
  --device-map 0=0:cuda:0,1=1:rocm:0 -p "Hello"

# Cross-vendor on same NUMA node (LOCAL TP scenario)
llaminar2 -m model.gguf --device 0:cuda:0 \
  --tp 2 --tp-devices 0:cuda:0,0:rocm:0 -p "Hello"

# Multi-machine explicit mapping
mpirun -np 4 --hostfile hosts.txt llaminar2 -m model.gguf \
  --device-map 0=localhost:0:cuda:0,1=localhost:1:cuda:0,2=node1:0:cuda:0,3=node1:1:cuda:0 \
  -p "Hello"
```

#### NUMA Node to MPI Rank Relationship

By convention, Llaminar maps MPI ranks to NUMA nodes:

| MPI Rank | NUMA Node | Calculation |
|----------|-----------|-------------|
| 0 | Machine 0, NUMA 0 | `rank % ranks_per_machine` within machine |
| 1 | Machine 0, NUMA 1 | |
| 2 | Machine 1, NUMA 0 | |
| 3 | Machine 1, NUMA 1 | |

This mapping is controlled by MPI hostfile and binding options:
```bash
# Bind each rank to a socket (NUMA node)
mpirun -np 4 --map-by socket --bind-to socket ./llaminar2 ...
```

#### Layer Placement (Hybrid CPU/GPU)

```
--cpu-layers <n>              Number of layers to place on CPU
                              Applies to: Each rank's local CPU
                              Effect: Last N layers run on CPU

--cpu-layers-first <n>        Place first N layers on CPU (embedding + early layers)

--gpu-layers <n>              Number of layers to place on GPU (rest on CPU)
                              Alternative to --cpu-layers
```

**Examples**:
```bash
# Single rank: Most layers on GPU, last 4 on CPU
llaminar2 -m model.gguf --device cuda:0 --cpu-layers 4 -p "Hello"

# Multi-rank: Each rank runs most layers on its GPU, last 4 on its CPU
mpirun -np 2 llaminar2 -m model.gguf --device-type cuda --cpu-layers 4 -p "Hello"
```

#### Topology Introspection

```
--show-topology               Print full cluster topology and exit
--show-numa                   Print NUMA node → device mapping and exit
--validate-devices            Validate device addresses without executing
```

```bash
# See what devices exist on each NUMA node
llaminar2 --show-numa

# Example output:
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           NUMA TOPOLOGY                                       ║
# ╠════════════╦══════════╦═══════════════════════════════════════════════════════╣
# ║  Hostname  ║ NUMA Node║ Devices                                               ║
# ╠════════════╬══════════╬═══════════════════════════════════════════════════════╣
# ║ localhost  ║    0     ║ cuda:0 (RTX 3090, 24GB), rocm:0 (MI50, 16GB), cpu:0   ║
# ║ localhost  ║    1     ║ cuda:0 (RTX 3090, 24GB), rocm:0 (MI50, 16GB), cpu:0   ║
# ╚════════════╩══════════╩═══════════════════════════════════════════════════════╝
# 
# Note: cuda:0 on NUMA 0 and cuda:0 on NUMA 1 are DIFFERENT physical GPUs!

# See rank-to-device mapping for an MPI run
mpirun -np 4 llaminar2 -m model.gguf --device-per-rank auto --show-topology

# Example output:
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        MPI RANK DEVICE MAPPING                                ║
# ╠══════╦════════════╦══════════╦════════════════════════════════════════════════╣
# ║ Rank ║  Hostname  ║ NUMA Node║ Assigned Device                                ║
# ╠══════╬════════════╬══════════╬════════════════════════════════════════════════╣
# ║   0  ║ localhost  ║    0     ║ localhost:0:cuda:0 (RTX 3090, 24GB)            ║
# ║   1  ║ localhost  ║    1     ║ localhost:1:cuda:0 (RTX 3090, 24GB)            ║
# ║   2  ║ node1      ║    0     ║ node1:0:cuda:0 (RTX 3090, 24GB)                ║
# ║   3  ║ node1      ║    1     ║ node1:1:cuda:0 (RTX 3090, 24GB)                ║
# ╚══════╩════════════╩══════════╩════════════════════════════════════════════════╝
```

### 0.7 Backend Selection Options

```
--collective-backend <spec>   Override collective backend selection
                              Values: auto, nccl, rccl, pciebar, mpi, host

--backend-for-domain <spec>   Override backend for specific domain
                              Format: <domain_id>:<backend>
```

**Examples**:
```bash
# Force PCIeBAR for all collectives (cross-vendor)
llaminar2 -m model.gguf --tp 2 --collective-backend pciebar -p "Hello"

# Force MPI for domain 0, NCCL for domain 1
llaminar2 -m model.gguf --tp 4 --backend-for-domain 0:mpi,1:nccl -p "Hello"
```

### 0.8 Placement Strategy Options

```
--placement <strategy>        Placement strategy selection
                              Values: auto, cpu-only, gpu-first, hybrid, 
                                      heterogeneous-multi-domain

--placement-config <path>     Load placement from config file (YAML/JSON)
```

**Examples**:
```bash
# Force CPU-only execution
llaminar2 -m model.gguf --placement cpu-only -p "Hello"

# Force hybrid (GPU + CPU participation)
llaminar2 -m model.gguf --placement hybrid --cpu-layers 4 -p "Hello"

# Full heterogeneous multi-domain (Scenario 7 style)
llaminar2 -m model.gguf --placement heterogeneous-multi-domain -p "Hello"

# Load from config file
llaminar2 -m model.gguf --placement-config cluster.yaml -p "Hello"
```

### 0.9 Configuration File Format

For complex setups, support YAML configuration files.

#### Global Device Address Format in YAML

Device addresses use the same format as CLI:
```
hostname:numa_node:device_type:device_ordinal
```

Shorthand is allowed in config files:
- `cuda:0` → `localhost:<local_numa>:cuda:0`
- `0:cuda:0` → `localhost:0:cuda:0`

#### Full Configuration Example

```yaml
# cluster.yaml - Scenario 7 configuration
# 2 nodes × 2 sockets × (RTX 3090 + MI50) = 8 GPUs total, 4 MPI ranks
#
# Physical topology:
#   node0:
#     numa0: cuda:0, rocm:0 → MPI rank 0
#     numa1: cuda:1, rocm:1 → MPI rank 1
#   node1:
#     numa0: cuda:0, rocm:0 → MPI rank 2
#     numa1: cuda:1, rocm:1 → MPI rank 3

orchestration:
  auto_detect: false
  
  # MPI rank → device mapping (primary compute device per rank)
  # Uses full global address: hostname:numa:type:ordinal
  device_assignment:
    mode: explicit
    mapping:
      - rank: 0
        device: node0:0:cuda:0   # CUDA GPU on node0, NUMA 0
      - rank: 1
        device: node0:1:cuda:1   # CUDA GPU on node0, NUMA 1
      - rank: 2
        device: node1:0:cuda:0   # CUDA GPU on node1, NUMA 0
      - rank: 3
        device: node1:1:cuda:1   # CUDA GPU on node1, NUMA 1
  
  tensor_parallel:
    scope: local           # Each rank does LOCAL TP across its 2 GPUs
    local_degree: 2        # 2 devices per rank
    
    # Define TP domains - each domain spans devices on ONE rank
    # Devices use global addresses relative to that rank's hostname/numa
    domains:
      - id: 0
        type: gpu_intra_rank
        ranks: [0]
        # Both GPUs on node0, NUMA node 0 (same socket)
        devices:
          - node0:0:cuda:0    # RTX 3090
          - node0:0:rocm:0    # MI50
        weights: [0.6, 0.4]   # 3090 is 60% of work, MI50 is 40%
        backend: pciebar      # Cross-vendor requires PCIeBAR
        
      - id: 1
        type: gpu_intra_rank
        ranks: [1]
        devices:
          - node0:1:cuda:1    # RTX 3090 on NUMA 1
          - node0:1:rocm:1    # MI50 on NUMA 1
        weights: [0.6, 0.4]
        backend: pciebar
        
      - id: 2
        type: gpu_intra_rank
        ranks: [2]
        devices:
          - node1:0:cuda:0
          - node1:0:rocm:0
        weights: [0.6, 0.4]
        backend: pciebar
        
      - id: 3
        type: gpu_intra_rank
        ranks: [3]
        devices:
          - node1:1:cuda:1
          - node1:1:rocm:1
        weights: [0.6, 0.4]
        backend: pciebar
  
  pipeline_parallel:
    degree: 2              # 2 PP stages
    split: equal           # Each stage gets half the layers
    
    # Stage → rank mapping
    # Ranks 0,1 (node 0) = PP stage 0
    # Ranks 2,3 (node 1) = PP stage 1
    stages:
      - id: 0
        layers: [0, 13]            # Layers 0-13
        ranks: [0, 1]              # Ranks 0 and 1 share this stage (with TP)
        domains: [0, 1]            # Their TP domains
      - id: 1
        layers: [14, 27]           # Layers 14-27
        ranks: [2, 3]              # Ranks 2 and 3 share this stage (with TP)
        domains: [2, 3]
    
    # Optional: CPU participation for memory-constrained scenarios
    cpu_layers:
      enabled: false
      count: 4
      position: last       # 'first' or 'last'
    
  # Backend selection for different communication patterns
  backends:
    intra_rank_gpu: pciebar    # Cross-vendor within same NUMA node
    cross_rank_gpu: mpi        # Between ranks (could be NCCL if same vendor)
    cross_node: mpi            # Between nodes (InfiniBand)
```

**Loading**:
```bash
llaminar2 -m model.gguf --placement-config cluster.yaml -p "Hello"

# Validate without running
llaminar2 -m model.gguf --placement-config cluster.yaml --validate-only

# Dry run to see computed plan
llaminar2 -m model.gguf --placement-config cluster.yaml --dry-run
```

#### Simpler Configuration Examples

```yaml
# simple_tp.yaml - 2-way LOCAL TP on single machine
# Shorthand addresses expand to localhost:<current_numa>:...
orchestration:
  tensor_parallel:
    scope: local
    devices:
      - 0:cuda:0      # localhost:0:cuda:0
      - 0:cuda:1      # localhost:0:cuda:1
    weights: [0.5, 0.5]
```

```yaml
# simple_pp.yaml - 2-way PP across 2 nodes
# Each rank is one PP stage (no TP)
orchestration:
  pipeline_parallel:
    degree: 2
    # Implicitly: stage 0 → rank 0, stage 1 → rank 1
```

```yaml
# hybrid_cpu_gpu.yaml - GPU primary + CPU spillover on single machine
orchestration:
  device_assignment:
    mode: auto
    type: cuda    # Prefer CUDA GPUs
  layer_placement:
    cpu_layers: 4
    cpu_layers_position: last
```

```yaml
# cross_vendor_tp.yaml - LOCAL TP with CUDA + ROCm on same NUMA node
orchestration:
  tensor_parallel:
    scope: local
    devices:
      - 0:cuda:0      # RTX 3090 on NUMA 0
      - 0:rocm:0      # MI50 on NUMA 0
    weights: [0.6, 0.4]  # 3090 gets 60%, MI50 gets 40%
    backend: pciebar     # Cross-vendor requires PCIeBAR
```

### 0.10 Debugging and Introspection

```
--dry-run                     Compute placement plan, print it, exit
--explain-placement           Log detailed placement decisions at INFO level
--show-topology               Print detected topology and exit
--show-plan                   Print placement plan in human-readable format
--dump-plan <path>            Write placement plan to file (JSON)
--validate-plan               Validate placement plan without executing
```

**Examples**:
```bash
# See what would happen
llaminar2 -m model.gguf --tp 2 --pp 2 --dry-run

# Detailed explanation of why decisions were made
llaminar2 -m model.gguf --tp 2 --explain-placement -p "Hello"

# Export plan for analysis
llaminar2 -m model.gguf --tp 2 --pp 2 --dump-plan plan.json --dry-run

# Validate a config file
llaminar2 -m model.gguf --placement-config cluster.yaml --validate-plan
```

### 0.11 Help System

```
--help                        Standard help (existing options)
--help-orchestration          Detailed orchestration help with examples
--help-placement              Placement strategy documentation
--help-config                 Config file format documentation
```

### 0.12 Implementation Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **0.12.1** | Design `OrchestrationConfig` struct to hold all parsed options | 0.5 days |
| **0.12.2** | Implement CLI parser for new options (extend existing argparse) | 1 day |
| **0.12.3** | Implement YAML/JSON config file loader | 1 day |
| **0.12.4** | Implement `--dry-run` and `--explain-placement` | 0.5 days |
| **0.12.5** | Implement `--show-topology` using existing NodeTopology | 0.5 days |
| **0.12.6** | Wire `OrchestrationConfig` into `InferenceRunnerFactory` | 1 day |
| **0.12.7** | Unit tests for CLI parsing | 0.5 days |
| **0.12.8** | Integration tests for config loading | 0.5 days |
| **0.12.9** | Documentation and `--help-*` text | 0.5 days |

**Phase 0 Total: ~6 days**

### 0.13 OrchestrationConfig Structure

```cpp
// src/v2/config/OrchestrationConfig.h

// =============================================================================
// Enums
// =============================================================================

enum class TPScope {
    AUTO,      // Infer from world_size and device count
    LOCAL,     // Intra-rank (single MPI rank, multiple devices)
    GLOBAL,    // Cross-rank (multiple MPI ranks)
    HYBRID     // Both LOCAL and GLOBAL (hierarchical)
};

enum class DeviceAssignmentMode {
    AUTO,        // Auto-detect based on topology
    LOCAL_GPU,   // Each rank uses GPU on its NUMA node
    ROUND_ROBIN, // Cycle through available devices
    EXPLICIT     // Use explicit --device-map
};

enum class PPSplitMode {
    EQUAL,     // n_layers / pp_degree per stage
    WEIGHTED,  // Split by compute weight
    MANUAL     // Use explicit --pp-layers
};

// =============================================================================
// Global Device Address
// =============================================================================

/**
 * Fully-qualified device address in the cluster.
 * 
 * Format: hostname:numa_node:device_type:device_ordinal
 * 
 * Examples:
 *   - localhost:0:cuda:0    (CUDA GPU 0 on NUMA node 0 of localhost)
 *   - node1:1:rocm:0        (ROCm GPU 0 on NUMA node 1 of node1)
 *   - localhost:0:cpu:0     (CPU on NUMA node 0 of localhost)
 * 
 * Shorthand forms (expand to full form at parse time):
 *   - cuda:0          → localhost:<current_numa>:cuda:0
 *   - 0:cuda:0        → localhost:0:cuda:0
 *   - node1:0:cuda:0  → (already full form)
 */
struct GlobalDeviceAddress {
    std::string hostname = "localhost";  // Physical machine
    int numa_node = 0;                   // NUMA node / socket
    DeviceType device_type;              // cuda, rocm, cpu
    int device_ordinal = 0;              // Device index within type on that NUMA
    
    /// Parse from string (handles shorthand)
    static GlobalDeviceAddress parse(const std::string& spec);
    
    /// Convert to string (full form)
    std::string toString() const;
    
    /// Convert to local DeviceId (for use within a rank)
    DeviceId toLocalDeviceId() const;
    
    /// Check if this is the local NUMA node for the current process
    bool isLocalNuma(int current_numa_node) const;
    
    /// Equality
    bool operator==(const GlobalDeviceAddress& other) const;
};

// =============================================================================
// Device Mapping (MPI-aware)
// =============================================================================

struct RankDeviceMapping {
    int rank;                        // MPI rank
    GlobalDeviceAddress device;      // Fully-qualified device address
    
    /// Parse from string: "0=localhost:0:cuda:0" or "0=cuda:0" (shorthand)
    static RankDeviceMapping parse(const std::string& spec);
};

// =============================================================================
// TP Domain Specification
// =============================================================================

struct TPDomainSpec {
    int domain_id;
    TPDomainType type;               // GPU_INTRA_RANK, CPU_CROSS_RANK, GPU_CROSS_RANK
    
    // Devices in this domain (full addresses)
    std::vector<GlobalDeviceAddress> devices;
    
    // For GLOBAL scope: which MPI ranks participate
    std::vector<int> ranks;
    
    std::vector<float> weights;      // Proportional work split
    CollectiveBackendType backend;   // NCCL, RCCL, PCIeBAR, MPI, HOST, AUTO
};

// =============================================================================
// PP Stage Specification
// =============================================================================

struct PPStageSpec {
    int stage_id;
    int first_layer;
    int last_layer;
    std::vector<int> owning_ranks;   // MPI ranks that own this stage
    std::vector<int> domain_ids;     // TP domains active in this stage
};

// =============================================================================
// Main Configuration
// =============================================================================

struct OrchestrationConfig {
    // =========================================================================
    // Auto-Detection Control
    // =========================================================================
    bool auto_detect = true;
    bool dry_run = false;
    bool explain_placement = false;
    bool show_topology = false;
    bool show_numa = false;
    bool show_pp_topology = false;
    bool validate_only = false;
    
    // =========================================================================
    // Device Assignment (NUMA-aware global addressing)
    // =========================================================================
    DeviceAssignmentMode device_mode = DeviceAssignmentMode::AUTO;
    GlobalDeviceAddress device_for_this_rank;     // --device 
    DeviceType device_type_preference = DeviceType::ANY;  // --device-type
    std::vector<RankDeviceMapping> device_map;    // --device-map
    
    // =========================================================================
    // Tensor Parallelism
    // =========================================================================
    int tp_degree = 1;                           // --tp <n>
    TPScope tp_scope = TPScope::AUTO;            // --tp-scope
    
    // For LOCAL scope (devices within this rank's NUMA nodes):
    std::vector<GlobalDeviceAddress> tp_devices; // --tp-devices
    
    // For GLOBAL scope:
    std::vector<int> tp_ranks;                   // --tp-ranks (participating MPI ranks)
    
    // For HYBRID scope:
    int tp_local_degree = 1;                     // --tp-local
    int tp_global_degree = 1;                    // --tp-global
    
    // Common:
    std::vector<float> tp_weights;               // --tp-weights
    std::vector<TPDomainSpec> tp_domains;        // --tp-domain (explicit)
    
    // =========================================================================
    // Pipeline Parallelism
    // =========================================================================
    int pp_degree = 1;                           // --pp <n>
    PPSplitMode pp_split = PPSplitMode::EQUAL;   // --pp-split
    std::vector<PPStageSpec> pp_stages;          // --pp-layers (explicit)
    std::map<int, int> pp_stage_to_rank;         // --pp-rank-map
    bool pp_async = false;                       // --pp-async
    
    // =========================================================================
    // Layer Placement (Hybrid CPU/GPU)
    // =========================================================================
    int cpu_layers = 0;                          // --cpu-layers
    bool cpu_layers_first = false;               // --cpu-layers-first
    int gpu_layers = -1;                         // --gpu-layers (-1 = all)
    
    // =========================================================================
    // Backend Selection
    // =========================================================================
    CollectiveBackendType default_backend = CollectiveBackendType::AUTO;
    std::map<int, CollectiveBackendType> domain_backends;  // --backend-for-domain
    
    // =========================================================================
    // Placement Strategy
    // =========================================================================
    PlacementStrategyType strategy = PlacementStrategyType::AUTO;
    std::string config_file_path;                // --placement-config
    
    // =========================================================================
    // Methods
    // =========================================================================
    
    /// Validate configuration for consistency
    bool validate(const MPIContext* mpi_ctx) const;
    
    /// Get validation error messages
    std::vector<std::string> validationErrors(const MPIContext* mpi_ctx) const;
    
    /// Resolve AUTO values based on MPI context and detected topology
    OrchestrationConfig resolve(const MPIContext* mpi_ctx, 
                                const ClusterInventory& cluster) const;
    
    /// Get effective TP scope after resolution
    TPScope effectiveTPScope() const;
    
    /// Get device for a specific MPI rank
    DeviceId deviceForRank(int rank) const;
    
    /// Get all ranks participating in TP
    std::vector<int> tpParticipatingRanks() const;
    
    /// Get all ranks in a PP stage
    std::vector<int> ranksForPPStage(int stage) const;
    
    // Serialization
    static OrchestrationConfig fromYaml(const std::string& path);
    static OrchestrationConfig fromJson(const std::string& path);
    static OrchestrationConfig fromCLI(int argc, char** argv);
    std::string toYaml() const;
    std::string toJson() const;
    
    /// Print human-readable summary
    void printSummary(std::ostream& os, const MPIContext* mpi_ctx) const;
};
```

### 0.14 Success Criteria

- [ ] All new CLI options parse correctly
- [ ] `--dry-run` shows placement plan without executing
- [ ] `--explain-placement` logs decisions at INFO level
- [ ] `--show-topology` prints cluster topology
- [ ] Config file loading works for YAML format
- [ ] `OrchestrationConfig` correctly propagates to `InferenceRunnerFactory`
- [ ] Existing tests still pass (backward compatibility)
- [ ] New CLI tests cover all option combinations

---

## Phase 1: Plan-to-Graph Translation

### Objective

Wire `HeterogeneousMultiDomainStrategy` output into actual graph building.

### 1.1 Current Gap

```cpp
// HeterogeneousMultiDomainStrategy PRODUCES this:
struct HeterogeneousPlan {
    std::vector<TPDomain> tp_domains;
    std::vector<PPStage> pp_stages;
    std::vector<DomainAssignment> layer_assignments;
    // ...
};

// But InferenceRunnerFactory IGNORES it and does:
graph_config.default_device = device;  // Single device
graph_config.local_n_heads = n_heads / world_size;  // MPI-based only
```

### 1.2 Required Changes

| File | Change |
|------|--------|
| `InferenceRunnerFactory.cpp` | Accept `OrchestrationConfig`, call strategy |
| `InferenceRunnerFactory.cpp` | Extract `HeterogeneousPlan` from strategy output |
| `Qwen2GraphConfig` | Add `HeterogeneousPlan* plan` field |
| `GraphOrchestrator` | Store and use `HeterogeneousPlan` |

### 1.3 Implementation

```cpp
// In InferenceRunnerFactory::createInferenceRunner()

// Step 1: Build PlacementInput from model + cluster
PlacementInput input = buildPlacementInput(model_config, cluster_inventory);

// Step 2: Select and run strategy
auto strategy = PlacementStrategyFactory::create(orch_config.strategy);
PlacementPlan plan = strategy->compute(input);

// Step 3: Downcast to HeterogeneousPlan if applicable
auto* hetero_plan = dynamic_cast<HeterogeneousPlan*>(&plan);

// Step 4: Configure graph from plan
if (hetero_plan) {
    configureFromHeterogeneousPlan(graph_config, hetero_plan, mpi_ctx);
} else {
    // Fallback to current single-device logic
    configureFromSimplePlan(graph_config, plan, mpi_ctx);
}
```

### 1.4 Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **1.4.1** | Add `HeterogeneousPlan* plan` to `Qwen2GraphConfig` | 0.5 days |
| **1.4.2** | Create `PlacementInput` builder from model + cluster | 1 day |
| **1.4.3** | Wire strategy selection in `InferenceRunnerFactory` | 1 day |
| **1.4.4** | Implement `configureFromHeterogeneousPlan()` | 1.5 days |
| **1.4.5** | Unit tests for plan-to-config translation | 1 day |

**Phase 1 Total: ~5 days**

### 1.5 Success Criteria

- [ ] `HeterogeneousMultiDomainStrategy::compute()` called during inference setup
- [ ] `Qwen2GraphConfig` populated from plan (not hardcoded)
- [ ] `--explain-placement` shows strategy output
- [ ] Existing single-device inference still works

---

## Phase 2: Multi-Device Graph Building

### Objective

Modify `Qwen2Graph` to support per-stage device assignment.

### 2.1 Current Gap

```cpp
// Current: ALL stages get same device
void Qwen2Graph::buildLayerGraph(int layer_idx) {
    params.device_id = config_.default_device;  // Same for all!
    graph.addNode(name, stage, config_.default_device);
}
```

### 2.2 Required Changes

| File | Change |
|------|--------|
| `Qwen2GraphConfig` | Add `LayerPlacementConfig layer_placement` |
| `Qwen2Graph.cpp` | Use `layer_placement.deviceForLayer(idx)` |
| `Qwen2Graph.cpp` | Support variable `local_n_heads` per layer |

### 2.3 Implementation

```cpp
// In Qwen2Graph::buildLayerGraph()

DeviceId layer_device = config_.layer_placement.deviceForLayer(layer_idx);

// Get domain-specific head count
int local_heads = config_.plan->headsForDevice(layer_device);

AttnNormStage::Params norm_params;
norm_params.device_id = layer_device;  // Per-layer device
norm_params.n_heads = local_heads;     // Domain-specific
```

### 2.4 Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **2.4.1** | Add `LayerPlacementConfig` to `Qwen2GraphConfig` | 0.5 days |
| **2.4.2** | Modify `buildLayerGraph()` to query device per layer | 1 day |
| **2.4.3** | Support variable `local_n_heads` based on domain | 1 day |
| **2.4.4** | Handle device transitions (CPU→GPU handoff) | 1 day |
| **2.4.5** | Unit tests for multi-device graph | 1 day |

**Phase 2 Total: ~4.5 days**

### 2.5 Success Criteria

- [ ] Layers can be assigned to different devices
- [ ] `--cpu-layers 4` places last 4 layers on CPU
- [ ] Graph executes correctly across device boundaries
- [ ] Coherence handles CPU↔GPU transitions

---

## Phase 3: Pipeline Parallelism Integration

### Objective

Wire `PipelineParallelConfig` into graph building and insert Send/Recv stages.

### 3.1 Current Gap

```cpp
// PipelineParallelConfig EXISTS and is tested (32 tests) but:
// - Qwen2Graph builds ALL layers regardless of rank
// - WeightManager loads ALL weights regardless of rank
// - No Send/Recv stages are inserted
```

### 3.2 Required Changes

| File | Change |
|------|--------|
| `Qwen2Graph.cpp` | Accept `PipelineParallelConfig`, build only assigned layers |
| `Qwen2Graph.cpp` | Insert `ReceiveActivationsStage` at start (if not first stage) |
| `Qwen2Graph.cpp` | Insert `SendActivationsStage` at end (if not last stage) |
| `WeightManager.cpp` | Load only weights for assigned layers |
| `InferenceRunnerFactory.cpp` | Wire PP config |

### 3.3 Implementation

```cpp
// In Qwen2Graph::buildFullForwardGraph()

const auto& pp_config = config_.pipeline_config;
LayerRange my_range = pp_config.forRank(mpi_ctx_->rank());

// Receive from previous stage (if not first)
if (!pp_config.isFirstStage(mpi_ctx_->rank())) {
    graph.addNode("recv_from_prev", 
        ComputeStageFactory::createReceiveActivations({
            .src_rank = pp_config.previousRank(mpi_ctx_->rank()),
            .buffer = input_buffer,
            .mpi_ctx = mpi_ctx_
        }), DeviceId::cpu());
}

// Build ONLY my layers
for (int layer = my_range.first_layer; layer <= my_range.last_layer; ++layer) {
    buildLayerGraph(layer);
}

// Send to next stage (if not last)
if (!pp_config.isLastStage(mpi_ctx_->rank())) {
    graph.addNode("send_to_next",
        ComputeStageFactory::createSendActivations({
            .dest_rank = pp_config.nextRank(mpi_ctx_->rank()),
            .buffer = output_buffer,
            .mpi_ctx = mpi_ctx_
        }), DeviceId::cpu());
}
```

### 3.4 Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **3.4.1** | Add `PipelineParallelConfig` to `Qwen2GraphConfig` | 0.5 days |
| **3.4.2** | Modify `buildFullForwardGraph()` to use layer range | 1 day |
| **3.4.3** | Insert Send/Recv stages at PP boundaries | 1 day |
| **3.4.4** | Modify `WeightManager` for partial weight loading | 1.5 days |
| **3.4.5** | Handle embedding (first stage only) and LM head (last stage only) | 1 day |
| **3.4.6** | Integration test: 2-rank PP execution | 1.5 days |

**Phase 3 Total: ~6.5 days**

### 3.5 Success Criteria

- [ ] `--pp 2` splits layers between 2 MPI ranks
- [ ] Each rank loads only its assigned weights (~50% memory)
- [ ] Send/Recv correctly transfer activations between stages
- [ ] End-to-end inference produces correct output with PP

---

## Phase 4: LOCAL Tensor Parallelism

### Objective

Enable single-rank multi-device TP by decoupling TP degree from MPI world_size.

### 4.1 Current Gap

```cpp
// In InferenceRunnerFactory:
if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded) {
    // Enable TP
} else {
    // world_size == 1 means NO TP, even with multiple GPUs!
}
```

### 4.2 Required Changes

| File | Change |
|------|--------|
| `OrchestrationConfig.h` | Add `LocalTPConfig` struct |
| `InferenceRunnerFactory.cpp` | Check `local_tp_degree` alongside `world_size` |
| `WeightManager.cpp` | Shard based on device index, not MPI rank |
| `Qwen2Graph.cpp` | Build domain-specific subgraphs |

### 4.3 Implementation

```cpp
// New: LocalTPConfig for single-rank multi-device
struct LocalTPConfig {
    std::vector<DeviceId> devices;      // {cuda:0, rocm:0}
    std::vector<float> work_fractions;  // {0.73, 0.27}
    int local_tp_degree;                // 2
    
    int deviceIndex(DeviceId device) const;
    float workFraction(DeviceId device) const;
};

// In InferenceRunnerFactory:
bool enable_tp = (mpi_ctx && mpi_ctx->world_size() > 1) 
              || (local_tp_config && local_tp_config->local_tp_degree > 1);

if (enable_tp) {
    if (local_tp_config) {
        // LOCAL TP: shard by device index within rank
        configureLocalTP(graph_config, local_tp_config);
    } else {
        // GLOBAL TP: shard by MPI rank
        configureGlobalTP(graph_config, mpi_ctx);
    }
}
```

### 4.4 Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **4.4.1** | Define `LocalTPConfig` struct | 0.5 days |
| **4.4.2** | Modify TP enablement check | 0.5 days |
| **4.4.3** | Implement `configureLocalTP()` | 1.5 days |
| **4.4.4** | Modify `WeightManager` to shard by device index | 1 day |
| **4.4.5** | Build per-device subgraphs with local allreduce | 1.5 days |
| **4.4.6** | Wire `CollectiveContext` for LOCAL scope | 1 day |
| **4.4.7** | Integration test: single-rank 2-GPU TP | 1.5 days |

**Phase 4 Total: ~7.5 days**

### 4.5 Success Criteria

- [ ] `--tp 2 --tp-scope local` works with single MPI rank
- [ ] Weights sharded correctly across local devices
- [ ] PCIeBAR/NCCL used for intra-rank collectives
- [ ] Inference produces correct output with LOCAL TP

---

## Phase 5: End-to-End Integration

### Objective

Full Scenario 7 validation with all components working together.

### 5.1 Scenario 7 Configuration

```yaml
# Scenario 7: 2-Node Heterogeneous Pipeline Cluster
# - 2 nodes × 2 sockets × (RTX 3090 + MI50)
# - Cross-vendor TP via PCIeBAR
# - Cross-node PP via InfiniBand/MPI

orchestration:
  tensor_parallel:
    scope: local
    domains:
      - id: 0
        type: gpu_intra_rank
        devices: [cuda:0, rocm:0]
        weights: [0.6, 0.4]  # 3090 is faster
        backend: pciebar
  
  pipeline_parallel:
    stages: 2
    split: equal
    # Stage 0: Node 0, layers 0-13
    # Stage 1: Node 1, layers 14-27
```

### 5.2 Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **5.2.1** | Create Scenario 7 config file | 0.5 days |
| **5.2.2** | End-to-end test: 2-node, 4-GPU inference | 2 days |
| **5.2.3** | Parity test: Scenario 7 vs PyTorch reference | 2 days |
| **5.2.4** | Performance benchmarking | 1 day |
| **5.2.5** | Bug fixes and stabilization | 3 days |
| **5.2.6** | Documentation update | 1 day |

**Phase 5 Total: ~9.5 days**

### 5.3 Success Criteria

- [ ] Scenario 7 config loads and validates
- [ ] 2-node inference executes without crashes
- [ ] Output matches PyTorch reference (within tolerance)
- [ ] Performance meets target (>10 tok/s decode)
- [ ] All 365+ existing tests still pass

---

## Timeline Summary

| Phase | Description | Effort | Cumulative |
|-------|-------------|--------|------------|
| **0** | CLI and Configuration Architecture | 6 days | 6 days |
| **1** | Plan-to-Graph Translation | 5 days | 11 days |
| **2** | Multi-Device Graph Building | 4.5 days | 15.5 days |
| **3** | Pipeline Parallelism Integration | 6.5 days | 22 days |
| **4** | LOCAL Tensor Parallelism | 7.5 days | 29.5 days |
| **5** | End-to-End Integration | 9.5 days | 39 days |

**Total: ~39 days (~8 weeks)**

---

## Appendix A: Current vs Target Architecture

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     llaminar2 CLI                            │
│  (limited options: -m, -p, -n, --mpi-procs)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               InferenceRunnerFactory                         │
│  - Loads model (ALL weights)                                │
│  - Sets single default_device                               │
│  - TP only if world_size > 1                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GraphOrchestrator                          │
│  - Single device execution                                  │
│  - All layers built                                         │
│  - MPI AllReduce for TP                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Qwen2Graph                              │
│  - ALL 28 layers built                                      │
│  - ALL stages use same device_id                            │
│  - No PP stages                                             │
└─────────────────────────────────────────────────────────────┘

   ╔═══════════════════════════════════════════════════════╗
   ║                  UNUSED (365 tests!)                   ║
   ╠═══════════════════════════════════════════════════════╣
   ║  • ClusterInventory / DeviceInventory                 ║
   ║  • HeterogeneousMultiDomainStrategy                   ║
   ║  • TPDomain / MultiDomainTPConfig                     ║
   ║  • PipelineParallelConfig                             ║
   ║  • LayerPlacementConfig                               ║
   ║  • SendActivationsStage / ReceiveActivationsStage     ║
   ║  • NodeTopology / NUMAAllocator                       ║
   ║  • UPI Backend                                        ║
   ╚═══════════════════════════════════════════════════════╝
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     llaminar2 CLI                            │
│  --tp 2 --pp 2 --tp-scope local --placement heterogeneous   │
│  --tp-devices cuda:0,rocm:0 --tp-weights 0.73,0.27          │
│  --dry-run --explain-placement                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               OrchestrationConfig Parser                     │
│  - Parse CLI + config files                                 │
│  - Validate configuration                                   │
│  - Support --dry-run / --explain                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ClusterInventory                           │
│  - Detect local devices (CPU, CUDA, ROCm)                   │
│  - Exchange capabilities via MPI_Allgather                  │
│  - Build complete cluster view                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             HeterogeneousMultiDomainStrategy                 │
│  - Compute optimal placement                                │
│  - Generate HeterogeneousPlan                               │
│  - Assign layers to domains                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               InferenceRunnerFactory                         │
│  - Configure from HeterogeneousPlan                         │
│  - Load ONLY assigned weights                               │
│  - Set up LocalTPConfig OR GlobalTPConfig                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GraphOrchestrator                          │
│  - Multi-device execution                                   │
│  - Per-stage device routing                                 │
│  - Domain-aware collectives                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Qwen2Graph                              │
│  - Build ONLY assigned layers (PP)                          │
│  - Per-stage device assignment                              │
│  - Insert Send/Recv at PP boundaries                        │
│  - Variable local_n_heads per domain                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 CollectiveContext                            │
│  - Route to correct backend per domain                      │
│  - PCIeBAR for cross-vendor GPU                             │
│  - NCCL/RCCL for same-vendor GPU                            │
│  - MPI for cross-rank                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Unified Global Device Addressing

### B.1 The Problem

The current codebase has inconsistent device addressing:

| Component | How it identifies devices |
|-----------|---------------------------|
| `DeviceId` | `type + ordinal` (e.g., CUDA:0) - **no hostname, no NUMA** |
| `RankPlacement` | `hostname + socket_id + numa_node` - **hostname is string, not structured** |
| `TPDomain` | `vector<DeviceId>` - **loses hostname/NUMA context** |
| `DeviceInfo` | `numa_node` as int, **hostname on parent struct** |
| `PlacementDevice` | `Type + gpu_index` - **no vendor, no hierarchy** |
| CLI (proposed) | `hostname:numa:type:ordinal` - **new, not in code yet** |

This creates confusion:
- "cuda:0" - Is this on localhost? Which NUMA node?
- "rank 0 uses GPU 0" - Which machine? Which socket?
- TP domain with `[cuda:0, rocm:0]` - Are they on the same NUMA node?

### B.2 The Solution: GlobalDeviceAddress

Introduce a **single canonical address type** used everywhere:

```cpp
// src/v2/backends/GlobalDeviceAddress.h

#pragma once
#include "DeviceId.h"
#include <string>
#include <functional>  // for std::hash

namespace llaminar2 {

/**
 * @brief Fully-qualified global device address
 * 
 * Unambiguously identifies any device in a cluster:
 *   hostname:numa_node:device_type:device_ordinal
 * 
 * Examples:
 *   - localhost:0:cuda:0    (CUDA GPU 0 on NUMA 0 of localhost)
 *   - node1:1:rocm:0        (ROCm GPU 0 on NUMA 1 of node1)
 *   - node2:0:cpu:0         (CPU on NUMA 0 of node2)
 * 
 * This is THE canonical way to refer to devices across:
 *   - CLI arguments
 *   - Configuration files
 *   - Placement plans
 *   - TP domains
 *   - MPI rank mappings
 *   - Collective backends
 */
struct GlobalDeviceAddress {
    std::string hostname = "localhost";   ///< Physical machine hostname
    int numa_node = 0;                    ///< NUMA node / socket index
    DeviceType device_type = DeviceType::CPU;  ///< Device type
    int device_ordinal = 0;               ///< Device index within type on that NUMA
    
    // =========================================================================
    // Factory Methods
    // =========================================================================
    
    /// Create CPU address for current host
    static GlobalDeviceAddress cpu(int numa = 0) {
        return {"localhost", numa, DeviceType::CPU, 0};
    }
    
    /// Create CUDA GPU address for current host
    static GlobalDeviceAddress cuda(int ordinal, int numa = 0) {
        return {"localhost", numa, DeviceType::CUDA, ordinal};
    }
    
    /// Create ROCm GPU address for current host
    static GlobalDeviceAddress rocm(int ordinal, int numa = 0) {
        return {"localhost", numa, DeviceType::ROCm, ordinal};
    }
    
    /// Create from explicit components
    static GlobalDeviceAddress create(
        const std::string& host, int numa, DeviceType type, int ordinal) {
        return {host, numa, type, ordinal};
    }
    
    // =========================================================================
    // Parsing
    // =========================================================================
    
    /**
     * @brief Parse from string (supports shorthand)
     * 
     * Formats (longest to shortest):
     *   - "node1:0:cuda:0"     → full form
     *   - "0:cuda:0"           → localhost:0:cuda:0
     *   - "cuda:0"             → localhost:<current_numa>:cuda:0
     * 
     * @param spec Address string
     * @param current_numa NUMA node for shorthand expansion (default: 0)
     * @return Parsed address
     * @throws std::invalid_argument if format is invalid
     */
    static GlobalDeviceAddress parse(const std::string& spec, int current_numa = 0);
    
    /**
     * @brief Try to parse, return nullopt on failure
     */
    static std::optional<GlobalDeviceAddress> tryParse(
        const std::string& spec, int current_numa = 0);
    
    // =========================================================================
    // Serialization
    // =========================================================================
    
    /// Convert to full canonical string: "hostname:numa:type:ordinal"
    std::string toString() const;
    
    /// Convert to short string if possible (omit localhost, omit numa if 0)
    std::string toShortString() const;
    
    // =========================================================================
    // Conversions
    // =========================================================================
    
    /// Convert to local DeviceId (loses hostname/numa info)
    DeviceId toLocalDeviceId() const {
        return DeviceId(device_type, device_ordinal);
    }
    
    /// Create from local DeviceId + context
    static GlobalDeviceAddress fromLocalDeviceId(
        const DeviceId& local_id,
        const std::string& hostname = "localhost",
        int numa_node = 0) {
        return {hostname, numa_node, local_id.type, local_id.ordinal};
    }
    
    // =========================================================================
    // Predicates
    // =========================================================================
    
    bool isLocal() const { return hostname == "localhost"; }
    bool isCPU() const { return device_type == DeviceType::CPU; }
    bool isGPU() const { 
        return device_type == DeviceType::CUDA || 
               device_type == DeviceType::ROCm; 
    }
    bool isCUDA() const { return device_type == DeviceType::CUDA; }
    bool isROCm() const { return device_type == DeviceType::ROCm; }
    
    /// Check if this device is on the same NUMA node as another
    bool sameNuma(const GlobalDeviceAddress& other) const {
        return hostname == other.hostname && numa_node == other.numa_node;
    }
    
    /// Check if this device is on the same machine as another
    bool sameHost(const GlobalDeviceAddress& other) const {
        return hostname == other.hostname;
    }
    
    // =========================================================================
    // Comparison
    // =========================================================================
    
    bool operator==(const GlobalDeviceAddress& o) const {
        return hostname == o.hostname && 
               numa_node == o.numa_node &&
               device_type == o.device_type && 
               device_ordinal == o.device_ordinal;
    }
    
    bool operator!=(const GlobalDeviceAddress& o) const { return !(*this == o); }
    
    /// Ordering for use in std::map/set
    bool operator<(const GlobalDeviceAddress& o) const;
};

/// Stream output
std::ostream& operator<<(std::ostream& os, const GlobalDeviceAddress& addr);

} // namespace llaminar2

// Hash for use in unordered containers
namespace std {
template<>
struct hash<llaminar2::GlobalDeviceAddress> {
    size_t operator()(const llaminar2::GlobalDeviceAddress& addr) const;
};
}
```

### B.3 Migration Path

Update existing types to use `GlobalDeviceAddress`:

| Old Type | Old Field | New Field |
|----------|-----------|-----------|
| `TPDomain` | `vector<DeviceId> devices` | `vector<GlobalDeviceAddress> devices` |
| `RankInventory` | `hostname + devices[i].numa_node` | devices as `vector<GlobalDeviceAddress>` |
| `DeviceInfo` | `type + local_device_id + numa_node` | `GlobalDeviceAddress address` |
| `PlacementDevice` | `Type + gpu_index` | `GlobalDeviceAddress` (or keep for compatibility) |
| `LayerPlacement` | `PlacementDevice device` | `GlobalDeviceAddress device` |
| `CollectiveContext` | `vector<DeviceId> local_devices` | `vector<GlobalDeviceAddress> devices` |

### B.4 Updated Type Definitions

#### TPDomain (updated)

```cpp
struct TPDomain {
    TPDomainType type;
    MPI_Comm communicator;
    
    // OLD: std::vector<DeviceId> devices;
    // NEW: Full global addresses
    std::vector<GlobalDeviceAddress> devices;
    
    int local_rank_in_domain;
    int domain_size;
    std::string name;
    
    // New: Check if all devices are on same NUMA (for backend selection)
    bool allOnSameNuma() const {
        if (devices.empty()) return true;
        const auto& first = devices[0];
        for (size_t i = 1; i < devices.size(); ++i) {
            if (!devices[i].sameNuma(first)) return false;
        }
        return true;
    }
    
    // New: Check if all devices are on same host
    bool allOnSameHost() const {
        if (devices.empty()) return true;
        const auto& first = devices[0];
        for (size_t i = 1; i < devices.size(); ++i) {
            if (!devices[i].sameHost(first)) return false;
        }
        return true;
    }
    
    // New: Get unique hostnames in this domain
    std::vector<std::string> hostnames() const;
    
    // Compatibility: Get as local DeviceIds (for kernels that don't need global context)
    std::vector<DeviceId> toLocalDeviceIds() const {
        std::vector<DeviceId> result;
        result.reserve(devices.size());
        for (const auto& d : devices) {
            result.push_back(d.toLocalDeviceId());
        }
        return result;
    }
};
```

#### DeviceInfo (updated)

```cpp
struct DeviceInfo {
    // OLD: scattered fields
    // DeviceType type;
    // int local_device_id;
    // int numa_node;
    
    // NEW: Single canonical address
    GlobalDeviceAddress address;
    
    // Derived accessors (for compatibility)
    DeviceType type() const { return address.device_type; }
    int local_device_id() const { return address.device_ordinal; }
    int numa_node() const { return address.numa_node; }
    
    // Memory and compute info (unchanged)
    size_t memory_bytes = 0;
    size_t free_memory_bytes = 0;
    int compute_units = 0;
    float tflops_fp16 = 0.0f;
    float memory_bandwidth_gbps = 0.0f;
    std::string name;
    std::string uuid;
    bool supports_p2p = false;
    int pcie_bus_id = 0;
    
    bool isGPU() const { return address.isGPU(); }
    float computeWeight() const { /* unchanged */ }
};
```

#### RankInventory (updated)

```cpp
struct RankInventory {
    int rank = -1;
    int node_id = -1;
    int local_rank = -1;
    
    // OLD: std::string hostname; (separate from devices)
    // NEW: hostname derivable from devices, but also stored for convenience
    std::string hostname;
    
    // OLD: std::vector<DeviceInfo> devices;
    // NEW: devices have full addresses
    std::vector<DeviceInfo> devices;
    
    // Each device's address includes hostname:numa, so:
    // devices[i].address.hostname == this->hostname (consistency check)
    
    // Get all GPUs on a specific NUMA node
    std::vector<GlobalDeviceAddress> gpusOnNuma(int numa) const {
        std::vector<GlobalDeviceAddress> result;
        for (const auto& d : devices) {
            if (d.address.numa_node == numa && d.address.isGPU()) {
                result.push_back(d.address);
            }
        }
        return result;
    }
};
```

### B.5 Backend Selection Using GlobalDeviceAddress

The `BackendRouter` can now make intelligent decisions:

```cpp
CollectiveBackendType BackendRouter::selectBackend(
    const std::vector<GlobalDeviceAddress>& devices) {
    
    if (devices.empty()) return CollectiveBackendType::HOST;
    
    // Check if all on same NUMA node
    bool same_numa = true;
    bool same_host = true;
    bool all_cuda = true;
    bool all_rocm = true;
    bool any_gpu = false;
    
    const auto& first = devices[0];
    for (const auto& d : devices) {
        if (!d.sameNuma(first)) same_numa = false;
        if (!d.sameHost(first)) same_host = false;
        if (!d.isCUDA()) all_cuda = false;
        if (!d.isROCm()) all_rocm = false;
        if (d.isGPU()) any_gpu = true;
    }
    
    // Decision tree:
    // 1. Same host, same NUMA, all same GPU vendor → NCCL or RCCL
    // 2. Same host, same NUMA, mixed GPU vendors → PCIeBAR
    // 3. Same host, different NUMA → UPI (for CPU) or PCIeBAR (for GPU)
    // 4. Different hosts → MPI
    
    if (!same_host) {
        return CollectiveBackendType::MPI;  // Cross-node always MPI
    }
    
    if (!any_gpu) {
        return same_numa ? CollectiveBackendType::HOST : CollectiveBackendType::UPI;
    }
    
    if (same_numa) {
        if (all_cuda) return CollectiveBackendType::NCCL;
        if (all_rocm) return CollectiveBackendType::RCCL;
        // Mixed vendors on same NUMA → PCIeBAR
        return CollectiveBackendType::PCIEBAR;
    }
    
    // Same host, different NUMA, GPUs → PCIeBAR (crosses QPI/UPI)
    return CollectiveBackendType::PCIEBAR;
}
```

### B.6 Implementation Tasks

| Task | Description | Effort |
|------|-------------|--------|
| **B.6.1** | Create `GlobalDeviceAddress` class with parsing/serialization | 1 day |
| **B.6.2** | Update `DeviceInfo` to use `GlobalDeviceAddress` | 0.5 day |
| **B.6.3** | Update `RankInventory` and `ClusterInventory` | 0.5 day |
| **B.6.4** | Update `TPDomain` to use `GlobalDeviceAddress` | 0.5 day |
| **B.6.5** | Update `LayerPlacement` and `PlacementPlan` | 1 day |
| **B.6.6** | Update `CollectiveContext` and `BackendRouter` | 1 day |
| **B.6.7** | Update CLI parser to produce `GlobalDeviceAddress` | 0.5 day |
| **B.6.8** | Update YAML/JSON config parser | 0.5 day |
| **B.6.9** | Update serialization for MPI exchange | 0.5 day |
| **B.6.10** | Unit tests for `GlobalDeviceAddress` | 0.5 day |
| **B.6.11** | Migration of existing tests | 1 day |

**Total: ~8 days**

### B.7 Backward Compatibility

To avoid breaking existing code during migration:

1. **DeviceId remains** - It's still the local identifier for kernel calls
2. **Conversion methods** - `GlobalDeviceAddress::toLocalDeviceId()` and `fromLocalDeviceId()`
3. **Gradual migration** - Update components one at a time
4. **Deprecation warnings** - Mark old patterns as `[[deprecated]]`

```cpp
// Gradual migration pattern
struct TPDomain {
    // Phase 1: Add new field alongside old
    std::vector<DeviceId> devices;                    // OLD (deprecated)
    std::vector<GlobalDeviceAddress> global_devices;  // NEW
    
    // Phase 2: Remove old field, rename new
    std::vector<GlobalDeviceAddress> devices;
};
```

---

## Appendix C: Source Files Affected by GlobalDeviceAddress

### C.1 Files to Create

| File | Purpose |
|------|---------|
| `src/v2/backends/GlobalDeviceAddress.h` | Main header with struct definition |
| `src/v2/backends/GlobalDeviceAddress.cpp` | Implementation (parsing, serialization) |
| `tests/v2/unit/Test__GlobalDeviceAddress.cpp` | Unit tests for parsing, comparison, hashing |

### C.2 Files to Modify

| File | Changes |
|------|---------|
| **Backends** | |
| `src/v2/backends/DeviceId.h` | Add `fromGlobalAddress()` constructor |
| **Config** | |
| `src/v2/config/TPDomain.h` | Change `devices` to `vector<GlobalDeviceAddress>` |
| `src/v2/config/TPDomain.cpp` | Update domain creation to use global addresses |
| **Execution** | |
| `src/v2/execution/DeviceInventory.h` | Update `DeviceInfo`, `RankInventory`, `ClusterInventory` |
| `src/v2/execution/DeviceInventory.cpp` | Update discovery to populate global addresses |
| `src/v2/execution/PlacementPlan.h` | Update `LayerPlacement`, add global address support |
| `src/v2/execution/CollectiveContext.h` | Update device tracking to use global addresses |
| `src/v2/execution/CollectiveContext.cpp` | Update backend selection logic |
| `src/v2/execution/placement/HeterogeneousMultiDomainStrategy.h` | Update `DomainAssignment`, `PipelineStage` |
| `src/v2/execution/placement/HeterogeneousMultiDomainStrategy.cpp` | Update domain creation |
| **Utils** | |
| `src/v2/utils/MPITopology.h` | Update `RankPlacement` to use global addresses |
| `src/v2/utils/MPITopology.cpp` | Update serialization, hostname handling |
| `src/v2/utils/NodeTopology.h` | Add NUMA-aware device enumeration helpers |
| **Collective** | |
| `src/v2/collective/BackendRouter.h` | Add `selectBackend(vector<GlobalDeviceAddress>)` |
| `src/v2/collective/BackendRouter.cpp` | Implement smart backend selection |

### C.3 Test Files to Update

| File | Changes |
|------|---------|
| `tests/v2/unit/Test__TPDomain.cpp` | Update to use `GlobalDeviceAddress` |
| `tests/v2/unit/Test__DeviceInventory.cpp` | Update device creation |
| `tests/v2/unit/Test__BackendRouter.cpp` | Add tests for address-based selection |
| `tests/v2/integration/Test__CrossVendorTensorParallel.cpp` | Update domain setup |
| `tests/v2/integration/Test__NCCLBackend.cpp` | Update device specification |
| `tests/v2/integration/Test__RCCLBackend.cpp` | Update device specification |
| `tests/v2/integration/Test__CollectiveBackendIntegration.cpp` | Update device setup |

### C.4 Addressing Scheme Throughout Codebase

After migration, the addressing hierarchy is consistently used:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GLOBAL DEVICE ADDRESSING                              │
│                                                                              │
│  Format: hostname:numa_node:device_type:device_ordinal                      │
│                                                                              │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │   CLI / Config    │    │  Runtime Types    │    │    Kernel APIs    │   │
│  ├───────────────────┤    ├───────────────────┤    ├───────────────────┤   │
│  │ --device 0:cuda:0 │───▶│ GlobalDeviceAddr  │───▶│ DeviceId (local)  │   │
│  │ --tp-devices ...  │    │ in TPDomain       │    │ for cudaSetDevice │   │
│  │ config.yaml       │    │ in PlacementPlan  │    │ for kernel calls  │   │
│  │                   │    │ in CollectiveCtx  │    │                   │   │
│  └───────────────────┘    └───────────────────┘    └───────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│                     ┌─────────────────────────────┐                         │
│                     │    Backend Selection        │                         │
│                     │                             │                         │
│                     │  Same NUMA + same vendor    │                         │
│                     │    → NCCL/RCCL              │                         │
│                     │  Same NUMA + mixed vendor   │                         │
│                     │    → PCIeBAR                │                         │
│                     │  Same host, diff NUMA       │                         │
│                     │    → PCIeBAR or UPI         │                         │
│                     │  Different hosts            │                         │
│                     │    → MPI                    │                         │
│                     └─────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Coherence bugs at device boundaries | High | High | Extensive testing, transfer tracing |
| PP deadlocks from Send/Recv ordering | Medium | High | Careful ordering, timeout detection |
| Memory fragmentation with partial weight loading | Medium | Medium | Pre-allocation, memory pooling |
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
