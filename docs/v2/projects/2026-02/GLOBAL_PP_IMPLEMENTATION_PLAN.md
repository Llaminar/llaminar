# Hierarchical Parallelism Tree — Implementation Plan

**Date**: February 6, 2026  
**Status**: Planning → Active Implementation  
**Author**: Copilot + David Sanftenberg  
**Supersedes**: Original flat "Global PP" plan (sections below updated)

## 1. Executive Summary

This document describes the implementation plan for a **recursive parallelism tree** that composes Pipeline Parallel and Tensor Parallel at arbitrary nesting depths — across physical machines, sockets, and GPU domains.

**Design principle**: Parallelism is expressed as a tree of two node types:
- **PP (PipelineParallel)**: Children execute **sequentially**, with activations transferred between them
- **TP (TensorParallel)**: Children execute **in parallel** on the same data, with allreduce to combine

Leaves are individual compute devices (`cuda:0`, `rocm:0`, `cpu:0`).

**Target topology (4-machine, 2-socket, 4-GPU-per-socket example):**

```
PipelineParallel("global",                    ← cross-machine (InfiniBand)
    PipelineParallel("host0",                 ← cross-socket (UPI/shared memory)
        PipelineParallel("socket0",           ← within-socket (PCIeBAR/NCCL)
            TensorParallel("cuda:0","cuda:1"),← NCCL allreduce
            TensorParallel("rocm:0","rocm:1") ← RCCL allreduce
        ),
        PipelineParallel("socket1", ...)
    ),
    PipelineParallel("host1", ...),
    PipelineParallel("host2", ...),
    PipelineParallel("host3", ...)
)
```

**Layer distribution** for a 48-layer model on this topology:
```
global PP: 12 layers per machine
  host PP: 6 layers per socket
    socket PP: 3 layers per TP domain
      TP: each device handles all 3 layers (sharded weights, allreduce)
```

**Transfer mechanisms selected by tree edge type:**

| PP Edge | Relationship | Transfer | Bandwidth |
|---------|-------------|----------|-----------|
| global | host → host | MPI over InfiniBand | 25-100 GB/s |
| host | socket → socket | MPI over UPI/SHM | ~50 GB/s |
| socket | TP-domain → TP-domain | LocalPP (NCCL/PCIeBAR) | 12-25 GB/s |
| TP (leaf) | device ↔ device | NCCL/RCCL allreduce | 300+ GB/s |

### Key Insight: `IInferenceRunner` Polymorphism

The recursive tree maps directly to nested `IInferenceRunner` instances:

```
ParallelismTree node              →  IInferenceRunner implementation
──────────────────                   ─────────────────────────────
PP("global", 4 children)          →  PipelineRunner(4 stage_runners)
  PP("host0", 2 children)        →  PipelineRunner(2 stage_runners)
    PP("socket0", 2 children)    →  RankOrchestrator(PP mode)
      TP("cuda:0","cuda:1")      →  RankOrchestrator(TP, NCCL)
      TP("rocm:0","rocm:1")      →  RankOrchestrator(TP, RCCL)
    PP("socket1", ...)           →  RankOrchestrator(PP mode)
```

`RankOrchestrator` already supports `pp_stage_runners_` as a `vector<unique_ptr<IInferenceRunner>>` — allowing PP-wrapping-TP composition today. The tree simply adds more nesting levels and cross-machine edges.

### Relationship to Flat Global PP (Phases 1-2)

The Phase 1-2 implementations (`GlobalPPTransferStage`, `GlobalPPTopology`, `GlobalPPRankPlan`) remain intact as the **single-level building blocks**. The new recursive design supersedes Phase 3+ by replacing the flat `GlobalPPOrchestrator` with a tree-compiled set of nested `IInferenceRunner` instances.

| Component | Status | Role in Tree Architecture |
|-----------|--------|--------------------------|
| `GlobalPPTransferStage` | ✅ Implemented | Transfer stage for cross-rank PP edges |
| `GlobalPPTopology` | ✅ Implemented | Flat topology for single-level use; superseded by tree |
| `GlobalPPRankPlan` | ✅ Implemented | Single-level plan; superseded by tree compilation |
| `ParallelismTree` | 🆕 New | Recursive tree data structure |
| `TreeToRunnerCompiler` | 🆕 New | Tree → nested `IInferenceRunner` hierarchy |
| `HierarchicalCommBuilder` | 🆕 New | Tree → nested MPI communicators |

---

## 2. Goals and Non-Goals

### Goals

1. **Recursive parallelism tree**: Express arbitrary nesting of PP and TP as a tree of two node types
2. **Multi-machine support**: Pipeline parallel across physical machines linked by InfiniBand/Ethernet
3. **Multi-socket support**: Pipeline parallel across CPU sockets within a machine
4. **Heterogeneous TP**: Tensor parallel across mixed GPU vendors within a socket (NCCL + RCCL + PCIeBAR)
5. **Automatic transfer selection**: Tree edge type determines transfer mechanism (MPI, UPI, LocalPP, NCCL)
6. **Hierarchical communicators**: MPI communicator tree matching the parallelism tree
7. **Correct weight sharding**: Compound sharding — TP shards within PP stages within TP domains
8. **Deadlock-free**: Deterministic pipeline ordering at every tree level
9. **CLI/YAML driven**: Full topology expressible via configuration
10. **Incremental implementation**: Build on existing `IInferenceRunner`, `RankOrchestrator`, and Phase 1-2 code

### Non-Goals

- Pipeline interleaving / micro-batching (synchronous PP at each level)
- Automatic topology detection (explicit configuration required)
- Data parallelism (each tree processes the same model, not batch sharding)
- Dynamic reconfiguration (tree is fixed for the lifetime of the process)

---

## 3. Architecture

### 3.1 The Parallelism Tree

The core data structure is a recursive tree with two internal node types and one leaf type:

```
┌──────────────────────────────────────────────┐
│ ParallelismNode                               │
│   type: PP | TP | DEVICE                      │
│   name: string (e.g., "host0", "socket1")     │
│   children: vector<ParallelismNode>           │
│                                               │
│   // For DEVICE leaves:                       │
│   device: GlobalDeviceAddress                 │
│   owning_rank: int (MPI rank)                 │
│                                               │
│   // For TP nodes:                            │
│   tp_weights: vector<float>                   │
│   tp_backend: CollectiveBackendType           │
│                                               │
│   // For PP nodes:                            │
│   pp_transfer: TransferBackend (auto-derived) │
└──────────────────────────────────────────────┘
```

**Semantics:**

| Node Type | Children | Data Flow | Synchronization |
|-----------|----------|-----------|-----------------|
| **PP** | Sequential stages | Activation passed from child[i] → child[i+1] via transfer | Pipeline ordering |
| **TP** | Parallel shards | Same input broadcast; each child processes sharded weights | Allreduce after each layer |
| **DEVICE** | None (leaf) | Single-device execution via `DeviceGraphOrchestrator` | N/A |

**Invariants:**
1. PP children cover a contiguous, non-overlapping partition of layers
2. TP children all handle the same layer range (with sharded weights)
3. Every leaf DEVICE has an `owning_rank` (MPI rank that physically has the device)
4. A PP node's children may span multiple MPI ranks (cross-machine PP)
5. A TP node's children may span multiple MPI ranks (cross-rank TP / Global TP)

### 3.2 Tree-to-Runner Compilation

The `TreeToRunnerCompiler` walks the tree **bottom-up**, producing `IInferenceRunner` instances:

```
Compile(node):
  if node.type == DEVICE:
    return DeviceGraphOrchestrator(device, layers)
  
  if node.type == TP:
    runners = [Compile(child) for child in node.children]
    return RankOrchestrator(TP mode, runners, backend, weights)
  
  if node.type == PP:
    stage_runners = [Compile(child) for child in node.children]
    return PipelineRunner(stage_runners, transfer_stages)
```

The **transfer mechanism** between PP children is determined by analyzing the leaf devices:

```
DetermineTransfer(child_a, child_b):
  ranks_a = leafRanks(child_a)  // MPI ranks owning child_a's leaf devices
  ranks_b = leafRanks(child_b)  // MPI ranks owning child_b's leaf devices
  
  if ranks_a ∩ ranks_b ≠ ∅:
    // Same rank — use LocalPPTransferStage (intra-process)
    return LocalPPTransfer(backend = selectLocalBackend(devices))
  
  if sameHost(ranks_a, ranks_b):
    // Same machine, different ranks — MPI over shared memory / UPI
    return GlobalPPTransfer(backend = UPI or MPI_SHM)
  
  // Different machines — MPI over InfiniBand/Ethernet
  return GlobalPPTransfer(backend = MPI_IB)
```

### 3.3 Communicator Hierarchy

Each level of the tree that crosses MPI rank boundaries needs its own MPI communicator:

```
MPI_COMM_WORLD (all ranks across all machines)
├── split → machine_comm (ranks on same machine, color = host_id)
│   ├── split → socket0_comm (ranks on socket 0, color = numa_node)
│   │   └── (no further split — intra-rank uses NCCL/RCCL directly)
│   └── split → socket1_comm (ranks on socket 1)
└── split → global_pp_stage_comm (ranks in same PP stage across machines)
```

The `HierarchicalCommBuilder` walks the tree top-down, splitting communicators at each PP/TP node where children span different MPI ranks:

```cpp
BuildComms(node, parent_comm):
  if node.type == DEVICE:
    return  // Leaf — no communicator needed
  
  if node.type == TP:
    ranks = leafRanks(node)
    if ranks.size() > 1:
      node.comm = MPI_Comm_split(parent_comm, color=domain_id)
    for child in node.children:
      BuildComms(child, node.comm or parent_comm)
  
  if node.type == PP:
    for child in node.children:
      BuildComms(child, parent_comm)  // PP children inherit parent comm
```

### 3.4 Execution Model

**Each MPI rank** receives the full `ParallelismTree` and compiles its own subtree. Ranks that don't own any leaf devices in a subtree skip those stages but still participate in MPI collectives as needed.

**Per-rank execution flow for a 3-level tree:**

```
Rank 0 (host0, socket0, cuda:0+cuda:1):
  1. Receive activation (or embedding for first stage)
  2. Execute my TP domain (cuda:0 + cuda:1, NCCL allreduce)
  3. SEND activation to next PP stage (PCIeBAR to rocm domain, or MPI to next socket)
  4. Wait for downstream completion
  5. ...repeat per token

Rank 3 (host1, socket0, cuda:0+cuda:1):
  1. RECV activation from host0 via InfiniBand MPI
  2. Execute my TP domain
  3. SEND to host1's socket1
  4. ...
```

### 3.5 Weight Loading

Weight sharding is determined by the tree path from root to each leaf:

```
Root (PP, 48 layers)
├── host0 (PP, layers 0-11)
│   ├── socket0 (PP, layers 0-5)
│   │   ├── TP_domain_0 (TP, layers 0-2, shard 0 of 2)
│   │   │   ├── cuda:0 (shard 0 of 2 × shard 0 of 2 = shard 0 of 4 of TP_domain_0)
│   │   │   └── cuda:1 (shard 1 of 2 × shard 0 of 2 = shard 1 of 4 of TP_domain_0)
│   │   └── TP_domain_1 (TP, layers 3-5, shard 0 of 2)
│   │       └── ...
```

PP nodes partition layers across children. TP nodes shard weights across children. The compound shard descriptor at each leaf is the product of all TP sharding above it.

### 3.6 Relationship to Existing Code

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         NEW: Tree Composition Layer                                │
│                                                                                   │
│  ParallelismTree → TreeToRunnerCompiler → Nested IInferenceRunner hierarchy      │
│                    HierarchicalCommBuilder → MPI communicator tree                │
│                                                                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                         EXISTING: Execution Layer (reused)                         │
│                                                                                   │
│  ┌─────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐  │
│  │DeviceGraphOrchest.  │  │RankOrchestrator│  │PipelineRunner (new)   │  │
│  │(single device)      │  │(TP, PP, TP+PP modes)  │  │(cross-rank PP)        │  │
│  │= tree DEVICE leaf   │  │= tree TP/PP node      │  │= tree cross-rank PP   │  │
│  └─────────────────────┘  └───────────────────────┘  └───────────────────────┘  │
│                                                                                   │
│  Transfer Stages (select by tree edge type):                                     │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐                  │
│  │LocalPPTransfer   │ │GlobalPPTransfer  │ │AllreduceStage   │                  │
│  │(intra-rank GPU)  │ │(cross-rank MPI)  │ │(TP allreduce)   │                  │
│  └──────────────────┘ └──────────────────┘ └─────────────────┘                  │
│                                                                                   │
│  Backends: NCCL | RCCL | PCIeBAR | Heterogeneous | UPI | MPI | HOST             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Structures

### 4.1 ParallelismTree (NEW — replaces flat GlobalPPTopology for multi-level use)

The recursive tree data structure. Each node is one of three types.

```cpp
// execution/parallelism_tree/ParallelismTree.h

/// A node in the parallelism tree
struct ParallelismNode {
    /// Node type
    enum class Type {
        PIPELINE_PARALLEL,  ///< Children execute sequentially (layer partitioning)
        TENSOR_PARALLEL,    ///< Children execute in parallel (weight sharding + allreduce)
        DEVICE,             ///< Leaf: a single compute device
    };
    Type type;
    std::string name;  ///< Human-readable name ("global", "host0", "socket0:tp0")
    
    // ─── Tree structure ───
    std::vector<ParallelismNode> children;
    
    // ─── For DEVICE leaves ───
    GlobalDeviceAddress device;   ///< Physical device address
    int owning_rank = -1;         ///< MPI rank that physically owns this device
    
    // ─── For TENSOR_PARALLEL nodes ───
    std::vector<float> tp_weights;                ///< Proportional weights (empty = equal)
    CollectiveBackendType tp_backend = CollectiveBackendType::AUTO;  ///< Backend for allreduce
    
    // ─── For PIPELINE_PARALLEL nodes ───
    // Transfer backend is auto-derived from child rank locality
    // (cross-machine → MPI_IB, cross-socket → UPI/SHM, same-rank → LocalPP)
    
    // ─── Derived at compile time ───
    int first_layer = -1;  ///< Assigned by tree compiler (inclusive)
    int last_layer = -1;   ///< Assigned by tree compiler (inclusive)
    int total_tp_shards = 1;      ///< Compound TP degree (product of all ancestor TP nodes)
    int tp_shard_index = 0;       ///< This subtree's shard index within its parent TP
    
    // ─── Queries ───
    int layerCount() const { return last_layer - first_layer + 1; }
    bool isLeaf() const { return type == Type::DEVICE; }
    bool isCrossRank() const;        ///< Do leaf devices span multiple MPI ranks?
    std::set<int> leafRanks() const; ///< All MPI ranks owning leaf devices in subtree
    std::vector<const ParallelismNode*> leafDevices() const; ///< All leaf nodes
    int leafDeviceCount() const;     ///< Total leaf devices
    
    // ─── Validation ───
    std::vector<std::string> validate() const;
};

/// Root-level tree wrapper with model metadata
struct ParallelismTree {
    ParallelismNode root;
    int total_layers;            ///< Total transformer layers in model
    int world_size;              ///< Total MPI ranks
    bool has_embedding = true;   ///< Model has embedding (almost always true)
    bool has_lm_head = true;     ///< Model has LM head (almost always true)
    
    /// Assign layers to tree nodes (fills first_layer/last_layer throughout tree)
    void assignLayers();
    
    /// Validate entire tree
    std::vector<std::string> validate() const;
    
    /// Human-readable tree rendering
    std::string toString() const;
};
```

### 4.2 ParallelismTree Builder (fluent API)

```cpp
// Factory helpers for clean tree construction

auto tree = ParallelismTree::build(48 /*layers*/, 8 /*world_size*/,
    PP("global",
        PP("host0",
            PP("socket0",
                TP("cuda_domain", {cuda(0), cuda(1)}, CollectiveBackendType::NCCL),
                TP("rocm_domain", {rocm(0), rocm(1)}, CollectiveBackendType::RCCL)
            ),
            PP("socket1",
                TP("cuda_domain", {cuda(0), cuda(1)}, CollectiveBackendType::NCCL),
                TP("rocm_domain", {rocm(0), rocm(1)}, CollectiveBackendType::RCCL)
            )
        ),
        PP("host1", ...),
        PP("host2", ...),
        PP("host3", ...)
    )
);

// Assigns layers automatically:
// global PP: 12 layers/host, host PP: 6/socket, socket PP: 3/TP-domain
// TP: all 3 layers, sharded 2-way

tree.assignLayers();
auto errors = tree.validate();
```

### 4.3 CompoundShardInfo

Weight sharding for a device in a multi-level tree:

```cpp
/// Describes how weights are sharded at a specific leaf device
struct CompoundShardInfo {
    int layer_first;       ///< First layer this device processes
    int layer_last;        ///< Last layer this device processes
    int tp_shard_index;    ///< This device's TP shard index (0-based)
    int tp_total_shards;   ///< Total TP shards at this level
    float work_fraction;   ///< Fraction of work (for proportional TP)
    
    /// Derive from tree path: walk from root to leaf, accumulating TP shards
    static CompoundShardInfo fromTreePath(const ParallelismNode& leaf,
                                           const ParallelismTree& tree);
};
```

### 4.4 TransferSpec

Describes a transfer between two adjacent PP children:

```cpp
/// Transfer between two adjacent PP stage subtrees
struct TransferSpec {
    enum class Mechanism {
        LOCAL_PP,       ///< Same MPI rank, different devices (NCCL/PCIeBAR/host-staged)
        MPI_INTRAHOST,  ///< Same machine, different MPI ranks (UPI/shared memory)
        MPI_INTERHOST,  ///< Different machines (InfiniBand/Ethernet)
    };
    Mechanism mechanism;
    
    int sender_rank;    ///< MPI rank that sends (-1 if same-rank local)
    int receiver_rank;  ///< MPI rank that receives (-1 if same-rank local)
    int mpi_tag;        ///< Unique tag for MPI transfers
    
    CollectiveBackendType local_backend;  ///< For LOCAL_PP: which backend
    
    /// Derive mechanism from two subtrees
    static TransferSpec derive(const ParallelismNode& from,
                                const ParallelismNode& to,
                                int tag_base);
};
```

### 4.5 Flat Topology (Phase 1-2, retained)

The existing `GlobalPPTopology`, `GlobalPPRankPlan`, and `GlobalPPRankPlanBuilder` remain in place for single-level use cases (2-socket server with flat 3-stage pipeline). The tree model is a strict generalization — a single-level tree compiles down to the same flat structures internally.

See the existing headers for full API:
- `execution/global_pp/GlobalPPTopology.h` — topology + validation 
- `execution/global_pp/GlobalPPRankPlan.h` — per-rank plan with interleaved execute/transfer steps
- `execution/global_pp/GlobalPPRankPlanBuilder.h` — topology → per-rank plan derivation
- `execution/compute_stages/stages/GlobalPPTransferStage.h` — MPI send/recv compute stage

---

## 5. Implementation Phases

### Previously Completed (Phases 1-2)

| Phase | Status | What was built |
|-------|--------|----------------|
| **Phase 1**: GlobalPPTransferStage | ✅ Done | MPI send/recv compute stage, 17 unit tests |
| **Phase 2**: Flat Topology & Plan | ✅ Done | `GlobalPPTopology`, `GlobalPPRankPlan`, `GlobalPPRankPlanBuilder`, 26 unit tests |

These remain as single-level building blocks used internally by the tree compiler.

---

### Phase 3: ParallelismTree Data Structure (2-3 days)

**Goal**: Implement the recursive tree, layer assignment, validation, and fluent builder API.

**Files to Create**:
| File | Purpose |
|------|---------|
| `src/v2/execution/parallelism_tree/ParallelismTree.h` | Tree data structure + builder helpers |
| `src/v2/execution/parallelism_tree/ParallelismTree.cpp` | Layer assignment, validation, toString |
| `tests/v2/unit/execution/parallelism_tree/Test__ParallelismTree.cpp` | Tree construction and validation tests |

**Implementation Details**:

1. **`ParallelismNode`**: Recursive struct with `Type` enum (PP, TP, DEVICE), `children`, device address for leaves.

2. **`assignLayers()`**: Top-down recursive layer assignment:
   - PP node with N children and L layers: split L across children proportionally or equally
   - TP node: all children get the same layer range as the parent
   - For equal split: `child[i].layers = [base + i*chunk, base + (i+1)*chunk - 1]`

3. **`validate()`**: Comprehensive validation:
   - PP children must cover all parent layers with no gaps/overlaps
   - TP children must all have the same layer range
   - DEVICE leaves must have valid `owning_rank` within `world_size`
   - TP nodes must have ≥2 children
   - At least one DEVICE leaf must exist

4. **Fluent builder**: `PP()`, `TP()`, `Device()` helper functions returning `ParallelismNode`:
   ```cpp
   ParallelismNode PP(std::string name, std::initializer_list<ParallelismNode> children);
   ParallelismNode TP(std::string name, std::initializer_list<GlobalDeviceAddress> devices,
                       CollectiveBackendType backend = CollectiveBackendType::AUTO);
   ParallelismNode TP(std::string name, std::initializer_list<ParallelismNode> children);
   ```

5. **`leafRanks()`**: Recursively collect all MPI ranks that own leaf devices in a subtree.

6. **`isCrossRank()`**: True if `leafRanks().size() > 1`.

7. **`toString()`**: Indented tree visualization for logging.

**Unit Tests**:
- Construct 4-machine topology from user's example, verify structure
- Layer assignment for 48 layers across 4 hosts × 2 sockets × 2 TP domains
- Validate correct topology (should pass)
- Validation catches: TP with 1 child, PP with layer gap, leaf with invalid rank
- `leafRanks()` correctness across nesting levels
- `isCrossRank()` for local vs cross-rank subtrees
- Single-node degenerate case (1 device, no parallelism)
- toString output contains expected structure

---

### Phase 4: TransferSpec and CompoundShardInfo (1-2 days)

**Goal**: Derive transfer mechanisms and compound sharding from the tree.

**Files to Create**:
| File | Purpose |
|------|---------|
| `src/v2/execution/parallelism_tree/TransferSpec.h` | Transfer derivation |
| `src/v2/execution/parallelism_tree/TransferSpec.cpp` | Implementation |
| `src/v2/execution/parallelism_tree/CompoundShardInfo.h` | Compound shard descriptor |
| `src/v2/execution/parallelism_tree/CompoundShardInfo.cpp` | Implementation |
| `tests/v2/unit/execution/parallelism_tree/Test__TransferSpec.cpp` | Transfer derivation tests |
| `tests/v2/unit/execution/parallelism_tree/Test__CompoundShardInfo.cpp` | Shard derivation tests |

**Implementation Details**:

1. **`TransferSpec::derive(from, to, tag_base)`**:
   - Collect `leafRanks(from)` and `leafRanks(to)`
   - If intersection non-empty: `LOCAL_PP` (same-rank transfer)
   - Else if same hostname: `MPI_INTRAHOST`
   - Else: `MPI_INTERHOST`
   - Select sender rank = primary output rank of `from` subtree (typically the rank that computed last)
   - Select receiver rank = primary input rank of `to` subtree

2. **`CompoundShardInfo::fromTreePath(leaf, tree)`**:
   - Walk ancestors from leaf to root
   - Each TP ancestor multiplies the total_shards and adjusts shard_index
   - Each PP ancestor determines the layer range
   - Result: compound layer range + compound TP shard

**Unit Tests**:
- `TransferSpec`: same-rank (LocalPP), cross-rank same host (MPI_INTRAHOST), cross-host (MPI_INTERHOST)
- `CompoundShardInfo`: 1-level TP (2-way shard), 2-level (TP inside PP), 3-level (PP/TP/PP nesting)
- Proportional weights: non-equal TP sharding

---

### Phase 5: TreeToRunnerCompiler (3-4 days)

**Goal**: Compile the tree into nested `IInferenceRunner` instances that execute the model.

**Files to Create**:
| File | Purpose |
|------|---------|
| `src/v2/execution/parallelism_tree/TreeToRunnerCompiler.h` | Compiler header |
| `src/v2/execution/parallelism_tree/TreeToRunnerCompiler.cpp` | Recursive compilation |
| `tests/v2/unit/execution/parallelism_tree/Test__TreeToRunnerCompiler.cpp` | Compilation tests |
| `tests/v2/integration/execution/parallelism_tree/Test__TreeToRunnerCompiler_MPI.cpp` | MPI integration tests |

**Implementation Details**:

1. **`compile(tree, model_context, rank) → unique_ptr<IInferenceRunner>`**:
   - Prune the tree to only subtrees containing this rank's devices
   - Recursively compile bottom-up:
     - DEVICE → `DeviceGraphOrchestrator` configured for `[first_layer, last_layer]`
     - TP → `RankOrchestrator(TP mode)` with children as device runners
     - PP (same-rank) → `RankOrchestrator(PP mode)` with children as stage runners
     - PP (cross-rank) → `PipelineRunner` (new class, see below) with GlobalPPTransferStages

2. **`PipelineRunner`** (new `IInferenceRunner` subclass):
   - Wraps a sequence of `IInferenceRunner` stages interleaved with `GlobalPPTransferStage` transfers
   - `forward()`: for each stage in order, execute, then transfer to next
   - Only the stages owned by this rank actually compute; others are stubs
   - Handles hidden state handoff via `getHiddenState()`/`setHiddenState()`

3. **MPI communicator setup**: Each TP node that crosses ranks needs a communicator (from `HierarchicalCommBuilder`).

**Unit Tests** (single-process, mock runners):
- Compile a single device → produces `DeviceGraphOrchestrator`
- Compile a 2-device TP → produces `RankOrchestrator(TP mode)`
- Compile a 2-stage local PP → produces `RankOrchestrator(PP mode)`
- Compile a 2-level tree (PP wrapping TP) → produces nested orchestrators
- Verify layer assignment propagates to runner config

**MPI Integration Tests** (2+ processes):
- 2-rank, 2-stage cross-rank PP with CPU devices → verify hidden state transfer
- 2-rank TP with CPU → verify allreduce correctness
- Nested: 2-rank PP where each rank has 2-device TP → verify end-to-end

---

### Phase 6: HierarchicalCommBuilder (1-2 days)

**Goal**: Build MPI communicator hierarchy from the tree.

**Files to Create**:
| File | Purpose |
|------|---------|
| `src/v2/execution/parallelism_tree/HierarchicalCommBuilder.h` | Communicator builder |
| `src/v2/execution/parallelism_tree/HierarchicalCommBuilder.cpp` | MPI_Comm_split hierarchy |
| `tests/v2/unit/execution/parallelism_tree/Test__HierarchicalCommBuilder.cpp` | Unit tests (mock) |
| `tests/v2/integration/execution/parallelism_tree/Test__HierarchicalCommBuilder_MPI.cpp` | MPI integration tests |

**Implementation Details**:

1. **`build(tree, world_comm) → CommHierarchy`**:
   - Walk tree top-down
   - At each TP/PP node that crosses ranks, split the parent communicator
   - TP nodes: split by color = domain_id (ranks in same TP group get same color)
   - PP nodes: children inherit parent comm (they use P2P sends, not collectives)
   - Track lifetime: child comms freed before parent

2. **`CommHierarchy`**: RAII wrapper that frees communicators in reverse order.

3. **Integration with TPDomainBuilder**: Extend `TPDomainBuilder::splitCommunicator()` to accept parent_comm parameter.

**Unit Tests** (mock MPI):
- Verify correct split structure for 2-level tree
- Verify colors computed correctly for TP domains

**MPI Integration Tests** (2+ processes):
- Split MPI_COMM_WORLD into 2 TP domains, verify each domain communicator has correct size
- 3-level hierarchy: world → machine → socket → TP domain

---

### Phase 7: CLI / YAML Configuration Parser (2-3 days)

**Goal**: Parse tree topology from CLI arguments or YAML config file.

**Files to Create**:
| File | Purpose |
|------|---------|
| `src/v2/config/ParallelismTreeParser.h` | Parser header |
| `src/v2/config/ParallelismTreeParser.cpp` | CLI/YAML → ParallelismTree |
| `tests/v2/unit/config/Test__ParallelismTreeParser.cpp` | Parser tests |

**CLI Syntax** (hierarchical):

```bash
# Simple: 2-rank pipeline, each rank has 2-device TP
mpirun -np 2 llaminar2 -m model.gguf \
  --topology "PP(global, PP(rank0, TP(cuda:0,cuda:1)), PP(rank1, TP(rocm:0,rocm:1)))"

# Medium: 2-socket, 4 GPUs each
mpirun -np 2 llaminar2 -m model.gguf \
  --topology-file topology.yaml
```

**YAML Syntax** (the primary configuration mechanism for complex topologies):

```yaml
# topology.yaml — 4-machine, 2-socket, 4-GPU-per-socket
topology:
  type: pp
  name: global
  children:
    - type: pp
      name: host0
      children:
        - type: pp
          name: socket0
          children:
            - type: tp
              name: cuda_domain
              rank: 0
              backend: nccl
              devices: [cuda:0, cuda:1]
            - type: tp
              name: rocm_domain
              rank: 0
              backend: rccl
              devices: [rocm:0, rocm:1]
        - type: pp
          name: socket1
          children:
            - type: tp
              name: cuda_domain
              rank: 1
              backend: nccl
              devices: [cuda:0, cuda:1]
            - type: tp
              name: rocm_domain
              rank: 1
              backend: rccl
              devices: [rocm:0, rocm:1]
    - type: pp
      name: host1
      children: ...
    - type: pp
      name: host2
      children: ...
    - type: pp
      name: host3
      children: ...
```

**Unit Tests**:
- Parse YAML topology file → validate resulting tree
- Parse inline CLI topology string
- Round-trip: parse → toString → parse
- Error messages for malformed configs

---

### Phase 8: Runner Integration & E2E Testing (2-3 days)

**Goal**: Wire tree compilation into `OrchestrationRunnerFactory` and verify correctness.

**Files to Modify**:
| File | Changes |
|------|---------|
| `execution/runner/OrchestrationRunnerFactory.cpp` | Detect `--topology` → compile tree → return compiled runner |
| `config/OrchestrationConfig.h` | Add `ParallelismTree` field |

**Files to Create**:
| File | Purpose |
|------|---------|
| `tests/v2/integration/parity/qwen2/Test__Qwen2_TreePP_Parity.cpp` | Multi-level PP parity test |

**Parity Testing Strategy**:

1. **Reference**: Single-rank Qwen2 inference → capture logits
2. **Test**: 2-rank, 2-level PP (each rank has 2 TP devices) → compare logits
3. **Metrics**: Top-1 token match (greedy), cosine similarity, KL divergence

**Configurations to test** (incremental complexity):
- 1 host, 2 devices, local TP only (baseline)
- 1 host, 2 PP stages, each with 1 device (local PP)
- 2 ranks, 2 PP stages, each with 2-device TP (cross-rank PP wrapping TP)
- 4 ranks, 2 levels of PP (if test infra supports 4 MPI processes)

---

## 6. File Summary

### Phase 1-2 Files (Completed)

| File | Phase | Purpose |
|------|-------|---------|
| `execution/compute_stages/stages/GlobalPPTransferStage.h` | 1 | MPI send/recv stage header |
| `execution/compute_stages/stages/GlobalPPTransferStage.cpp` | 1 | MPI send/recv stage implementation |
| `execution/global_pp/GlobalPPTopology.h` | 2 | Flat topology description |
| `execution/global_pp/GlobalPPTopology.cpp` | 2 | Flat topology validation and helpers |
| `execution/global_pp/GlobalPPRankPlan.h` | 2 | Per-rank plan |
| `execution/global_pp/GlobalPPRankPlanBuilder.h` | 2 | Flat topology → per-rank plan |
| `execution/global_pp/GlobalPPRankPlanBuilder.cpp` | 2 | Builder implementation |

### Phase 3-8 Files (New — Tree Architecture)

| File | Phase | Purpose |
|------|-------|---------|
| `execution/parallelism_tree/ParallelismTree.h` | 3 | Recursive tree data structure + fluent builder |
| `execution/parallelism_tree/ParallelismTree.cpp` | 3 | Layer assignment, validation, toString |
| `execution/parallelism_tree/TransferSpec.h` | 4 | Transfer mechanism derivation |
| `execution/parallelism_tree/TransferSpec.cpp` | 4 | Transfer derivation implementation |
| `execution/parallelism_tree/CompoundShardInfo.h` | 4 | Multi-level shard descriptor |
| `execution/parallelism_tree/CompoundShardInfo.cpp` | 4 | Compound shard implementation |
| `execution/parallelism_tree/TreeToRunnerCompiler.h` | 5 | Tree → nested IInferenceRunner |
| `execution/parallelism_tree/TreeToRunnerCompiler.cpp` | 5 | Recursive compilation |
| `execution/parallelism_tree/PipelineRunner.h` | 5 | Cross-rank PP runner |
| `execution/parallelism_tree/PipelineRunner.cpp` | 5 | Transfer-interleaved execution |
| `execution/parallelism_tree/HierarchicalCommBuilder.h` | 6 | MPI communicator hierarchy |
| `execution/parallelism_tree/HierarchicalCommBuilder.cpp` | 6 | MPI_Comm_split tree walk |
| `config/ParallelismTreeParser.h` | 7 | CLI/YAML → ParallelismTree |
| `config/ParallelismTreeParser.cpp` | 7 | Parser implementation |

### Modified Files

| File | Phase | Changes |
|------|-------|---------|
| `src/v2/CMakeLists.txt` | 1-8 | Add new source files and test targets |
| `execution/runner/OrchestrationRunnerFactory.cpp` | 8 | Route `--topology` to TreeToRunnerCompiler |
| `config/OrchestrationConfig.h` | 7-8 | Add `ParallelismTree` field |

### Test Files

| File | Phase | Type |
|------|-------|------|
| `tests/v2/unit/.../Test__GlobalPPTransferStage.cpp` | 1 | Unit (17 tests) ✅ |
| `tests/v2/unit/.../Test__GlobalPPTopology.cpp` | 2 | Unit (26 tests) ✅ |
| `tests/v2/unit/.../Test__ParallelismTree.cpp` | 3 | Unit |
| `tests/v2/unit/.../Test__TransferSpec.cpp` | 4 | Unit |
| `tests/v2/unit/.../Test__CompoundShardInfo.cpp` | 4 | Unit |
| `tests/v2/unit/.../Test__TreeToRunnerCompiler.cpp` | 5 | Unit (mock runners) |
| `tests/v2/integration/.../Test__TreeToRunnerCompiler_MPI.cpp` | 5 | MPI (`MPI_PROCS 2`) |
| `tests/v2/unit/.../Test__HierarchicalCommBuilder.cpp` | 6 | Unit (mock MPI) |
| `tests/v2/integration/.../Test__HierarchicalCommBuilder_MPI.cpp` | 6 | MPI (`MPI_PROCS 2`) |
| `tests/v2/unit/config/Test__ParallelismTreeParser.cpp` | 7 | Unit |
| `tests/v2/integration/parity/.../Test__Qwen2_TreePP_Parity.cpp` | 8 | MPI (`MPI_PROCS 2`) |

---

## 7. Relationship to Existing Infrastructure

### What We Reuse Directly

| Component | Used By | Purpose |
|-----------|---------|---------|
| `IInferenceRunner` | `TreeToRunnerCompiler` | Polymorphic composition target |
| `RankOrchestrator` | TP / Local PP nodes | Already supports both modes with `pp_stage_runners_` |
| `DeviceGraphOrchestrator` | DEVICE leaf nodes | Runs a subset of layers on a single device |
| `GlobalPPTransferStage` | `PipelineRunner` | MPI send/recv for cross-rank PP |
| `ITPContext` / `GlobalTPContext` | Cross-rank TP nodes | Existing allreduce interface |
| `TPDomainBuilder` | `HierarchicalCommBuilder` | `splitCommunicator()` + `createCPUCrossRankDomain()` |
| `BackendRouter` | TP nodes | Domain-aware backend selection |
| `CollectiveBackendType` (8 types) | Transfer selection | AUTO/NCCL/RCCL/PCIE_BAR/HETEROGENEOUS/UPI/MPI/HOST |
| `GlobalPPTopology` / `GlobalPPRankPlan` | Flat single-level cases | Building block for simple topologies |
| `WeightShardInfo` | `CompoundShardInfo` | Extended for multi-level shard products |
| `KernelFactory` | All device nodes | Unchanged kernel dispatch |

### What We Extend

| Component | Extension |
|-----------|-----------|
| `TPDomainBuilder::splitCommunicator()` | Accept parent_comm parameter (currently hardcoded to MPI_COMM_WORLD) |
| `OrchestrationConfig` | Add `ParallelismTree` field alongside existing TP/PP config |
| `OrchestrationRunnerFactory` | Route `--topology` config to TreeToRunnerCompiler |

### What We Replace

| Old Component | Replaced By | Reason |
|--------------|-------------|--------|
| `GlobalPPOrchestrator` (Phase 3 of old plan) | `TreeToRunnerCompiler` + `PipelineRunner` | Flat orchestrator can't express recursive nesting |
| `GlobalPPConfigParser` (Phase 4 of old plan) | `ParallelismTreeParser` | Tree-aware YAML/CLI syntax |
| `HierarchicalPPContext` | No longer needed | Tree compiler handles the hierarchy directly |

### What We Keep But Don't Use

| Component | Status | Reason |
|-----------|--------|--------|
| Previous Global TP parity tests | Superseded | New tree-based parity tests are more comprehensive |
| `PPStage.GLOBAL_TP_DOMAIN` | Deprecated | Tree model natively expresses cross-rank TP |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MPI deadlock in nested send/recv | Medium | High | Deterministic pipeline ordering per tree level; all transfers have unique tags; integration tests with timeouts |
| Communicator hierarchy leak | Medium | Medium | RAII `CommHierarchy` frees child comms before parent; destructor ordering enforced |
| Compound shard miscalculation | Low | High | Extensive unit tests for 1/2/3-level shard products; parity tests catch weight loading errors |
| Performance regression (MPI overhead) | Medium | Medium | UPI transfer ~0.1ms for 2MB activation vs ~10ms compute; pipeline bubble is the real cost |
| Tree pruning edge cases | Medium | Medium | Ranks with no devices in tree → skip entirely; validate no empty runners produced |
| NCCL/RCCL init ordering with MPI | Medium | High | Init NCCL post-MPI_Comm_split; coordinated across TP domain ranks; existing GlobalTPContext handles this |
| YAML parser complexity | Low | Low | Use existing YAML library; tree depth bounded by nesting (typically 3-4 levels) |
| KV cache position drift across PP stages | Low | High | All stages see same `seq_len`; position counter is implicit from token count |
| Pipeline bubble overhead for deep trees | High | Medium | Expected: at N-way PP, bubble = (N-1)/N of compute time. Mitigated by large batch / long sequences |
| Weight streaming + tree TP incompatibility | Low | Medium | Non-goal for V1; document limitation clearly |

---

## 9. KV Cache Considerations

Each runner in the tree manages its own KV cache for its assigned layers. Cross-runner transfers carry only hidden state activations — never KV cache data.

### Layer-to-Cache Mapping

For a 48-layer model across 4 machines × 2 sockets:

```
Machine 0, Socket 0 (layers 0-5):   KV cache for layers 0-5
Machine 0, Socket 1 (layers 6-11):  KV cache for layers 6-11
Machine 1, Socket 0 (layers 12-17): KV cache for layers 12-17
Machine 1, Socket 1 (layers 18-23): KV cache for layers 18-23
Machine 2, Socket 0 (layers 24-29): KV cache for layers 24-29
Machine 2, Socket 1 (layers 30-35): KV cache for layers 30-35
Machine 3, Socket 0 (layers 36-41): KV cache for layers 36-41
Machine 3, Socket 1 (layers 42-47): KV cache for layers 42-47
```

Within a TP domain, the KV cache is **not sharded** — each device holds the full KV cache for its layers (GQA already reduces heads; further sharding rarely helps).

### Position Synchronization

The `position` counter does **not** need explicit synchronization across ranks:
- **Prefill**: all stages process the same `seq_len` (derived from input token count)
- **Decode**: all stages always process `seq_len=1`
- The pipeline ensures ordering: stage N+1 only runs after stage N completes

### Memory Implications

With tree-based PP, each rank's KV cache is proportionally smaller:
- Single rank: KV for all 48 layers (full model memory)
- 4-way PP: KV for 12 layers each (~25% per rank)
- 8-way PP: KV for 6 layers each (~12.5% per rank)

This is a key benefit — KV cache often dominates memory for long sequences.

---

## 10. Timeline

| Phase | Duration | Dependencies | Description |
|-------|----------|--------------|-------------|
| Phase 1: GlobalPPTransferStage | ✅ Done | None | MPI send/recv stage |
| Phase 2: Flat Topology & Plan | ✅ Done | None | Flat topology + plan builder |
| Phase 3: ParallelismTree | 2-3 days | None | Recursive tree + validation + builder |
| Phase 4: TransferSpec & Shard | 1-2 days | Phase 3 | Transfer derivation + compound shards |
| Phase 5: TreeToRunnerCompiler | 3-4 days | Phases 3, 4, 1 | Tree → nested runners + PipelineRunner |
| Phase 6: HierarchicalCommBuilder | 1-2 days | Phase 3 | MPI communicator tree |
| Phase 7: CLI / YAML Parser | 2-3 days | Phase 3 | Tree config parsing |
| Phase 8: Integration & Parity | 2-3 days | Phases 5, 6, 7 | OrchestrationRunnerFactory + E2E tests |

**Remaining: ~12-18 days**

**Critical path**: Phase 3 → Phase 5 → Phase 8

Phases 4 and 6 can be developed in parallel with Phase 5.
Phase 7 can be developed in parallel with Phase 6.

---

## 11. Success Criteria

1. **Recursive composition**: A `ParallelismTree` with 3+ levels compiles and executes correctly
2. **Token-level correctness**: Tree PP produces identical top-1 tokens as single-rank (greedy, temp=0)
3. **No deadlocks**: All MPI tests complete within timeout, all tree levels
4. **Automatic transfer selection**: Cross-rank transfers automatically use correct mechanism (LOCAL_PP / MPI_INTRAHOST / MPI_INTERHOST)
5. **Compound sharding**: Weights loaded correctly with multi-level TP shard products
6. **YAML-driven topology**: Full topology expressible via YAML config file
7. **Communicator correctness**: Hierarchical MPI_Comm_split produces correct domain communicators
8. **Backward compatible**: Existing `--tp`, `--pp` CLI flags continue to work (tree is opt-in via `--topology`)
9. **Test coverage**: Unit tests for all data structures; MPI integration tests for PipelineRunner and HierarchicalCommBuilder
10. **Parity passing**: At least 2-level PP×TP parity test passing against single-rank reference

---

## 12. Example: Full End-to-End Run

### Simple: 2-Rank PP with Local TP

```bash
# 2-socket server: NVIDIA GPUs on socket 0, AMD GPUs on socket 1
# Qwen2-0.5B (24 layers)
mpirun -np 2 llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --topology-file topology_2rank.yaml \
  -p "The capital of France is" -n 20 -t 0
```

```yaml
# topology_2rank.yaml
topology:
  type: pp
  name: cross_socket
  children:
    - type: tp
      name: socket0_tp
      rank: 0
      backend: nccl
      devices: [cuda:0, cuda:1]
    - type: tp
      name: socket1_tp
      rank: 1
      backend: rccl
      devices: [rocm:0, rocm:1]
```

### Complex: 4-Machine Hierarchical

```bash
# 4 machines × 2 sockets × 4 GPUs
# Llama-3 70B (80 layers)
mpirun -np 8 \
  --hostfile hosts.txt \
  --map-by socket \
  llaminar2 -m models/llama3-70b-q4_0.gguf \
  --topology-file topology_4machine.yaml \
  -p "Explain quantum computing" -n 100 -t 0
```

```yaml
# topology_4machine.yaml
topology:
  type: pp
  name: global
  children:
    - type: pp
      name: host0
      children:
        - type: tp
          name: host0_s0
          rank: 0
          backend: nccl
          devices: [cuda:0, cuda:1]
        - type: tp
          name: host0_s1
          rank: 1
          backend: nccl
          devices: [cuda:2, cuda:3]
    - type: pp
      name: host1
      children:
        - type: tp
          name: host1_s0
          rank: 2
          backend: nccl
          devices: [cuda:0, cuda:1]
        - type: tp
          name: host1_s1
          rank: 3
          backend: nccl
          devices: [cuda:2, cuda:3]
    - type: pp
      name: host2
      children:
        - type: tp
          name: host2_s0
          rank: 4
          backend: nccl
          devices: [cuda:0, cuda:1]
        - type: tp
          name: host2_s1
          rank: 5
          backend: nccl
          devices: [cuda:2, cuda:3]
    - type: pp
      name: host3
      children:
        - type: tp
          name: host3_s0
          rank: 6
          backend: nccl
          devices: [cuda:0, cuda:1]
        - type: tp
          name: host3_s1
          rank: 7
          backend: nccl
          devices: [cuda:2, cuda:3]
```

**Layer distribution** (80 layers, 4 machines × 2 sockets = 8 PP stages):
- host0_s0 (rank 0): layers 0-9
- host0_s1 (rank 1): layers 10-19
- host1_s0 (rank 2): layers 20-29
- host1_s1 (rank 3): layers 30-39
- host2_s0 (rank 4): layers 40-49
- host2_s1 (rank 5): layers 50-59
- host3_s0 (rank 6): layers 60-69
- host3_s1 (rank 7): layers 70-79

**Transfer mechanisms** (auto-derived from tree):
- host0_s0 → host0_s1: MPI over UPI (same host, different sockets)
- host0_s1 → host1_s0: MPI over InfiniBand (different hosts)
- host1_s0 → host1_s1: MPI over UPI (same host)
- ... and so on

**Expected output**: Identical to single-rank inference (greedy, temp=0).

### Mixed CUDA+ROCm TP with PCIeBAR

When a socket has both NVIDIA and AMD GPUs (e.g., PCIe slots served by the same CPU),
use `backend: pcie_bar` for the TP domain. PCIeBAR uses BAR-mapped direct P2P transfers
for allreduce, working across vendor boundaries without requiring NCCL or RCCL.

```yaml
# topology_mixed_vendor.yaml — 2-socket, mixed CUDA+ROCm per socket
topology:
  type: pp
  name: cross_socket
  children:
    - type: tp
      name: socket0_mixed
      rank: 0
      backend: pcie_bar
      devices: [cuda:0, rocm:0]
    - type: tp
      name: socket1_mixed
      rank: 1
      backend: pcie_bar
      devices: [cuda:1, rocm:1]
```

For larger mixed-vendor domains (>2 GPUs), use `backend: heterogeneous` which
orchestrates NCCL among CUDA devices, RCCL among ROCm devices, and PCIeBAR for
cross-vendor pairs:

```yaml
# topology_heterogeneous.yaml — single-rank, 4 GPUs (2 CUDA + 2 ROCm)
topology:
  type: tp
  name: hetero_tp
  rank: 0
  backend: heterogeneous
  devices: [cuda:0, cuda:1, rocm:0, rocm:1]
```

**Backend selection guide for TP domains:**

| Device Mix | Recommended Backend | Notes |
|-----------|-------------------|-------|
| All CUDA | `nccl` | Native NVIDIA collective |
| All ROCm | `rccl` | Native AMD collective |
| Mixed CUDA+ROCm (2 devices) | `pcie_bar` | Direct BAR-mapped P2P, lowest latency |
| Mixed CUDA+ROCm (>2 devices) | `heterogeneous` | Orchestrates NCCL+RCCL+PCIeBAR internally |
| CPU only | `mpi` or `host` | MPI collectives or shared memory |

**Validation**: The tree validator rejects mismatched backend-device configurations
(e.g., NCCL with ROCm devices or RCCL with CUDA devices) and will suggest the
appropriate backend.
