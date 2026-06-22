# Global TP Unified Interface Implementation Plan

**Date**: February 2, 2026  
**Status**: In Progress (Phase 5 Complete)  
**Author**: Copilot + David Sanftenberg  

## Implementation Progress

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Phase 1: Base Interface Extraction | ✅ Complete | 2026-02-02 | ITPContext, ILocalTPContext inheritance |
| Phase 2: Global TP Interface | ✅ Complete | 2026-02-02 | IGlobalTPContext, GlobalTPContext (55 tests) |
| Phase 3: Factory & Plan Wiring | ✅ Complete | 2026-02-02 | TPContextFactory (24 tests) |
| Phase 4: Stage Support | ✅ Complete | 2026-02-02 | TPAllreduceStage with ITPContext* (20 unit + 7 MPI integration tests) |
| Phase 5: HierarchicalPP Support | ✅ Complete | 2026-02-03 | PPStage global TP domain (21 unit + 13 MPI integration tests) |
| Phase 6: Integration Testing | 🔲 Next | - | End-to-end with real models |

## 1. Executive Summary

This document outlines the implementation plan for **Global Tensor Parallelism** (cross-MPI-rank TP) using a **Unified Interface** approach. The goal is to enable topologies like:

```
GlobalPipelineParallel(
  PipelineParallel(TensorParallel(0:cuda:0, 0:cuda:1), 0:rocm:0), 
  PipelineParallel(TensorParallel(1:cuda:0, 1:cuda:1), 1:rocm:0), 
  GlobalTensorParallel(0:cpu:0, 1:cpu:0)  // ← Cross-rank CPU TP via UPI
)
```

Where `GlobalTensorParallel(0:cpu:0, 1:cpu:0)` represents tensor-parallel sharding across CPUs on different MPI ranks, communicating via UPI interconnect (~50 GB/s).

---

## 2. Goals and Non-Goals

### Goals

1. **Unified TP Interface**: Create `ITPContext` as a common interface for both local and global TP
2. **Cross-Rank CPU TP**: Enable `GlobalTensorParallel` across MPI ranks using existing `UPICollectiveBackend`
3. **Hierarchical Composition**: Allow `HierarchicalPPContext` to treat `GlobalTPContext` as a stage
4. **Minimal API Changes**: Existing code using `ILocalTPContext` should require minimal changes
5. **Full Test Coverage**: Integration tests for cross-rank TP allreduce, broadcast, and allgather

### Non-Goals (Out of Scope)

- Cross-rank GPU TP (latency prohibitive; GPU TP remains intra-rank only)
- Automatic topology detection for global TP (requires explicit configuration)
- Dynamic rank joining/leaving during inference
- RDMA/InfiniBand backends (future work; MPI over UPI is sufficient for now)

---

## 3. Current State Analysis

### What Exists ✅

| Component | File | Description |
|-----------|------|-------------|
| `TPScope::GLOBAL` | `config/OrchestrationConfig.h` | Enum value for cross-rank TP |
| `RankExecutionPlan.global_tp_*` | `execution/mpi_orchestration/RankExecutionPlan.h` | Fields for global TP domain membership |
| `TPDomainType::CPU_CROSS_RANK` | `config/TPDomain.h` | Enum for cross-rank CPU domains |
| `TPDomainBuilder::createCPUCrossRankDomain()` | `config/TPDomain.cpp` | Creates MPI communicator via `MPI_Comm_split` |
| `MultiDomainTPConfig` | `config/TPDomain.h` | Manages GPU (intra-rank) + CPU (cross-rank) domains |
| `UPICollectiveBackend` | `collective/backends/UPIBackend.h` | MPI-based allreduce using domain communicator |
| `ILocalTPContext` | `collective/ILocalTPContext.h` | Interface for LOCAL TP (intra-rank only) |
| `LocalTPContext` | `collective/LocalTPContext.h` | Concrete local TP using NCCL/RCCL/PCIeBAR |
| `MPIContext::allreduce_*` | `utils/MPIContext.h` | Full set of MPI collective ops |

### What's Missing 🔧

| Gap | Severity | Description |
|-----|----------|-------------|
| **ITPContext base interface** | Major | No common interface for local vs global TP |
| **GlobalTPContext** | Major | No cross-rank TP context implementation |
| **AllreduceStage global support** | Major | Stages assume `ILocalTPContext*` |
| **HierarchicalPPContext global stages** | Medium | Can't treat cross-rank TP as a PP stage |
| **Factory for TP contexts** | Medium | No unified creation for local vs global |

### 3.1 CLI Gap Analysis

The current CLI has building blocks but **cannot express `GlobalPipelineParallel` compositions** directly.

#### What Exists ✅

| Flag | Purpose | Example |
|------|---------|---------|
| `--tp-scope local\|global\|hybrid` | Selects TP scope | `--tp-scope local` |
| `--tp-devices` | Explicit device list for LOCAL TP | `--tp-devices "cuda:0,cuda:1"` |
| `--tp-global <N>` | Global TP degree for hybrid | `--tp-global 2` |
| `--define-domain` | Named domain with devices | `--define-domain "gpu_tp=cuda:0,cuda:1"` |
| `--pp-stage` | Map PP stage to domain + layers | `--pp-stage "0=gpu_tp:0-13"` |
| `-pp` / `--pipeline-parallelism-degree` | PP degree | `-pp 2` |
| GlobalDeviceAddress | Full addressing: `hostname:numa:type:ordinal` | `"node1:0:cuda:0"` |

#### Device Address Format

```
Full:    hostname:numa_node:device_type:device_ordinal
         Example: "node1:0:cuda:0" (node1, NUMA 0, CUDA GPU 0)

Short:   type:ordinal
         Example: "cuda:0" → "localhost:0:cuda:0"

Cross-rank: rank_id:type:ordinal (NOT YET SUPPORTED)
         Example: "1:cpu:0" → CPU on MPI rank 1
```

#### What's Missing 🔧

| Gap | Severity | Description |
|-----|----------|-------------|
| **Cross-rank device addressing** | Major | No way to reference devices on other MPI ranks |
| **GlobalPipelineParallel syntax** | Major | No way to express PP across MPI ranks with global TP stages |
| **Global TP domain definition** | Medium | `--define-domain` assumes devices are LOCAL to this rank |
| **Inter-rank PP stage mapping** | Medium | `--pp-stage` can't reference cross-rank domains |

#### Target Topology (from this plan)

```
GlobalPipelineParallel(
  PipelineParallel(TensorParallel(0:cuda:0, 0:cuda:1), 0:rocm:0), 
  PipelineParallel(TensorParallel(1:cuda:0, 1:cuda:1), 1:rocm:0), 
  GlobalTensorParallel(0:cpu:0, 1:cpu:0)  // ← Cross-rank CPU TP via UPI
)
```

**Interpretation**:
- 2 MPI ranks (`0:` and `1:`)
- Each rank has local PP: GPU TP domain (CUDA) + single ROCm GPU
- Global TP stage: CPU on rank 0 + CPU on rank 1 (UPI-connected)

---

### 3.2 Proposed CLI Extensions

#### Option A: Rank-Prefixed Device Addresses (Recommended)

Extend `GlobalDeviceAddress` to support `rank:type:ordinal` format:

```bash
# Current (local only):
--tp-devices "cuda:0,cuda:1"

# Proposed (cross-rank):
--tp-devices "0:cuda:0,0:cuda:1,1:cuda:0,1:cuda:1"
#             ^rank 0^  ^rank 0^  ^rank 1^  ^rank 1^
```

**New device address format**:
```
rank_id:device_type:device_ordinal
Examples:
  0:cuda:0  → CUDA GPU 0 on MPI rank 0
  1:cpu:0   → CPU on MPI rank 1
  0:rocm:1  → ROCm GPU 1 on MPI rank 0
```

#### Option B: Domain Type Qualifier

Add domain type to `--define-domain`:

```bash
# Local domain (current):
--define-domain "gpu_tp=cuda:0,cuda:1"

# Global domain (new):
--define-domain "cpu_global=global:0:cpu:0,1:cpu:0"
#                           ^type^  ^devices^
# Or with explicit qualifier:
--define-domain "cpu_global=0:cpu:0,1:cpu:0;scope=global"
```

#### Option C: Explicit Global TP Flag

Separate flag for global TP configuration:

```bash
--global-tp-ranks "0,1"              # Ranks participating in global TP
--global-tp-devices "cpu:0"          # Device type on each rank (same for all)
# Implies: rank 0 uses cpu:0, rank 1 uses cpu:0, connected via UPI
```

---

### 3.3 Example: Full GlobalPipelineParallel CLI

**Target topology** (from Section 1):
```
GlobalPipelineParallel(
  PipelineParallel(TensorParallel(0:cuda:0, 0:cuda:1), 0:rocm:0), 
  PipelineParallel(TensorParallel(1:cuda:0, 1:cuda:1), 1:rocm:0), 
  GlobalTensorParallel(0:cpu:0, 1:cpu:0)
)
```

**Proposed CLI (using Option A + B)**:

```bash
mpirun -np 2 llaminar2 -m model.gguf \
  \
  # Define local TP domains (per-rank, same on both ranks)
  --define-domain "gpu_tp=cuda:0,cuda:1;backend=nccl" \
  \
  # Define global TP domain (cross-rank)
  --define-domain "cpu_global=0:cpu:0,1:cpu:0;scope=global;backend=upi" \
  \
  # PP stage 0: Local GPU TP (layers 0-9)
  --pp-stage "0=gpu_tp:0-9" \
  \
  # PP stage 1: Single ROCm GPU (layers 10-17) 
  --pp-stage "1=rocm:0:18-25" \
  \
  # PP stage 2: Global CPU TP (layers 26-27, cross-rank allreduce)
  --pp-stage "2=cpu_global:26-27" \
  \
  -p "Hello, world!"
```

**Simpler form for common case** (symmetric ranks):

```bash
mpirun -np 2 llaminar2 -m model.gguf \
  --pp 3 \
  --tp 2 --tp-devices "cuda:0,cuda:1" \
  --global-tp --global-tp-device cpu:0 \
  --pp-split manual \
  --pp-stage "0=local_tp:0-17" \
  --pp-stage "1=rocm:0:18-25" \
  --pp-stage "2=global_tp:26-27" \
  -p "Hello, world!"
```

---

### 3.4 Implementation Plan for CLI Extensions

**Phase 1.5** (Add after Phase 1, before Phase 2):

| Task | Description | Effort |
|------|-------------|--------|
| Extend `GlobalDeviceAddress::parse()` | Support `rank:type:ordinal` format | 0.5 day |
| Add `scope` field to `DomainDefinition` | LOCAL vs GLOBAL domain type | 0.5 day |
| Add `--global-tp` shorthand flags | `--global-tp`, `--global-tp-device`, `--global-tp-ranks` | 0.5 day |
| Update help text | Document new flags and formats | 0.5 day |
| Add CLI tests | `Test__OrchestrationConfigParser.cpp` | 1 day |

**Total CLI extension effort**: ~3 days

---

## 4. Architecture Design

### 4.1 Interface Hierarchy

```
                    ┌─────────────────┐
                    │   ITPContext    │  ← NEW: Base interface
                    │ (pure virtual)  │
                    └────────┬────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
    ┌────────▼────────┐            ┌─────────▼─────────┐
    │ ILocalTPContext │            │ IGlobalTPContext  │  ← NEW
    │  (existing)     │            │  (pure virtual)   │
    └────────┬────────┘            └─────────┬─────────┘
             │                               │
    ┌────────▼────────┐            ┌─────────▼─────────┐
    │ LocalTPContext  │            │ GlobalTPContext   │  ← NEW
    │  (existing)     │            │ (UPI + MPI)       │
    └─────────────────┘            └───────────────────┘
```

### 4.2 ITPContext Base Interface

```cpp
// collective/ITPContext.h (NEW)
class ITPContext {
public:
    virtual ~ITPContext() = default;

    // =========================================================================
    // Identity & Configuration
    // =========================================================================
    
    /// Get number of devices/ranks in this TP domain
    virtual int degree() const = 0;
    
    /// Get this participant's index within the domain (0 to degree-1)
    virtual int myIndex() const = 0;
    
    /// True if all participants are within the same MPI rank (intra-rank)
    virtual bool isLocal() const = 0;
    
    /// True if participants span multiple MPI ranks (cross-rank)
    bool isGlobal() const { return !isLocal(); }
    
    // =========================================================================
    // Core Collective Operations
    // =========================================================================
    
    /// All-reduce sum across all participants (in-place)
    virtual bool allreduce(TensorBase* tensor) = 0;
    
    /// Broadcast tensor from source to all participants
    virtual bool broadcast(TensorBase* tensor, int source_index = 0) = 0;
    
    /// All-gather: collect shards from all participants into full tensor
    virtual bool allgather(const TensorBase* local_shard, TensorBase* global_tensor) = 0;
};
```

### 4.3 ILocalTPContext Extension

```cpp
// collective/ILocalTPContext.h (MODIFIED)
class ILocalTPContext : public ITPContext {
public:
    // Existing methods remain unchanged...
    
    // ITPContext overrides
    bool isLocal() const override { return true; }
    
    // Local-specific methods (unchanged)
    virtual const std::vector<GlobalDeviceAddress>& devices() const = 0;
    virtual const std::vector<float>& weights() const = 0;
    virtual CollectiveBackendType backend() const = 0;
    
    // Extended allreduce with stage name (local optimization)
    virtual bool allreduce(TensorBase* tensor, const std::string& stage_name, size_t count = 0) = 0;
    
    // ... rest of existing interface
};
```

### 4.4 IGlobalTPContext Interface

```cpp
// collective/IGlobalTPContext.h (NEW)
class IGlobalTPContext : public ITPContext {
public:
    // ITPContext overrides
    bool isLocal() const override { return false; }
    
    // =========================================================================
    // Global-Specific Configuration
    // =========================================================================
    
    /// Get the MPI communicator for this domain
    virtual MPI_Comm communicator() const = 0;
    
    /// Get the global TP domain ID (from RankExecutionPlan)
    virtual int domainId() const = 0;
    
    /// Get all MPI ranks participating in this domain
    virtual const std::vector<int>& ranks() const = 0;
    
    // =========================================================================
    // Global-Specific Collective Operations
    // =========================================================================
    
    /// Barrier synchronization across all ranks in domain
    virtual void barrier() const = 0;
    
    /// Point-to-point send to another rank in domain
    virtual bool send(const TensorBase* tensor, int dest_index) = 0;
    
    /// Point-to-point receive from another rank in domain
    virtual bool recv(TensorBase* tensor, int source_index) = 0;
};
```

### 4.5 GlobalTPContext Implementation

```cpp
// collective/GlobalTPContext.h (NEW)
class GlobalTPContext : public IGlobalTPContext {
public:
    /// Create from TPDomain (which has MPI communicator)
    static std::unique_ptr<GlobalTPContext> create(const TPDomain& domain);
    
    /// Create from RankExecutionPlan's global_tp fields
    static std::unique_ptr<GlobalTPContext> createFromPlan(
        const RankExecutionPlan& plan, 
        MPI_Comm base_comm);

    // ITPContext implementation
    int degree() const override;
    int myIndex() const override;
    bool allreduce(TensorBase* tensor) override;
    bool broadcast(TensorBase* tensor, int source_index) override;
    bool allgather(const TensorBase* local_shard, TensorBase* global_tensor) override;
    
    // IGlobalTPContext implementation
    MPI_Comm communicator() const override;
    int domainId() const override;
    const std::vector<int>& ranks() const override;
    void barrier() const override;
    bool send(const TensorBase* tensor, int dest_index) override;
    bool recv(TensorBase* tensor, int source_index) override;

private:
    MPI_Comm domain_comm_;
    int domain_id_;
    int my_rank_in_domain_;
    int domain_size_;
    std::vector<int> world_ranks_;  // World ranks of all domain members
    std::unique_ptr<UPICollectiveBackend> backend_;
};
```

### 4.6 Factory for TP Context Creation

```cpp
// collective/TPContextFactory.h (NEW)
class TPContextFactory {
public:
    /// Create appropriate TP context based on RankExecutionPlan
    static std::unique_ptr<ITPContext> create(
        const RankExecutionPlan& plan,
        MPI_Comm base_comm);
    
    /// Create local TP context (explicit)
    static std::unique_ptr<ILocalTPContext> createLocal(
        const std::vector<DeviceId>& devices,
        const std::vector<float>& weights = {},
        CollectiveBackendType backend = CollectiveBackendType::AUTO);
    
    /// Create global TP context (explicit)
    static std::unique_ptr<IGlobalTPContext> createGlobal(
        MPI_Comm base_comm,
        int domain_id,
        int color,  // For MPI_Comm_split
        int key);   // For MPI_Comm_split
};
```

---

## 5. Implementation Phases

### Phase 1: Base Interface Extraction (1-2 days)

**Goal**: Create `ITPContext` as base interface without breaking existing code.

**Files to Create**:
- `src/v2/collective/ITPContext.h` - Base interface

**Files to Modify**:
- `src/v2/collective/ILocalTPContext.h` - Inherit from `ITPContext`, add `isLocal()` override
- `src/v2/collective/LocalTPContext.h` - Add `myIndex()` implementation

**Tests**:
- Ensure all existing `ILocalTPContext` tests pass
- Add `V2_Unit_ITPContext` tests for base interface

### Phase 2: Global TP Interface & Implementation (2-3 days)

**Goal**: Create `IGlobalTPContext` and `GlobalTPContext` implementation.

**Files to Create**:
- `src/v2/collective/IGlobalTPContext.h` - Global TP interface
- `src/v2/collective/GlobalTPContext.h` - Implementation header
- `src/v2/collective/GlobalTPContext.cpp` - Implementation

**Files to Modify**:
- `src/v2/CMakeLists.txt` - Add new source files

**Tests**:
- `tests/v2/unit/collective/Test__GlobalTPContext.cpp` - Unit tests
- `tests/v2/integration/collective/Test__GlobalTPContext_MPI.cpp` - Multi-rank tests

### Phase 3: Factory & Plan Wiring (1-2 days)

**Goal**: Create factory and wire `RankExecutionPlan` to context creation.

**Files to Create**:
- `src/v2/collective/TPContextFactory.h`
- `src/v2/collective/TPContextFactory.cpp`

**Files to Modify**:
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` - Use factory
- `src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp` - Populate global_tp fields correctly

**Tests**:
- `tests/v2/unit/collective/Test__TPContextFactory.cpp`

### Phase 4: Stage Support for Global TP (2-3 days) ✅ COMPLETE

**Goal**: Allow `LocalTPAllreduceStage` to work with both LOCAL and GLOBAL TP via `ITPContext*`.

> **Architecture Note**: There are two distinct allreduce stage types:
> - `AllreduceStage`: Uses `CollectiveContext*` + `TPDomain*` for cross-rank MPI allreduce (e.g., row-parallel weight reduction). This stage already handles global TP via MPI backends.
> - `LocalTPAllreduceStage`: Uses `ILocalTPContext*` for intra-rank LOCAL TP allreduce (e.g., NCCL/RCCL/PCIeBAR). **This is the stage we modified.**
>
> The stage was extended to accept `ITPContext*`, enabling it to work with both `LocalTPContext` (local TP) and `GlobalTPContext` (global TP).

**Implementation Decision**: The stage was NOT renamed. Instead, `LocalTPAllreduceStage` was extended with a new `setParams()` overload that accepts `ITPContext*`. A type alias `using TPAllreduceStage = LocalTPAllreduceStage;` was added for forward compatibility.

**Files Modified**:
- `src/v2/execution/compute_stages/stages/LocalTPAllreduceStage.h`:
  - Added `ITPContext* tp_ctx_base_` member for polymorphic dispatch
  - Added `setParams(input, output, ITPContext*, stage_name, count)` overload
  - Added `getTPContext()` accessor returning `ITPContext*`
  - Added `backend()` method using `ITPContext::backend()` virtual call
  - Added type alias: `using TPAllreduceStage = LocalTPAllreduceStage;`
- `src/v2/execution/compute_stages/stages/LocalTPAllreduceStage.cpp`:
  - Updated `execute()` to dispatch via `ITPContext::allreduce()` when `tp_ctx_base_` is set
  - Updated `getDumpInfo()` to include backend type from context

**Backward Compatibility**: ✅
- Existing code using `setParams(..., ILocalTPContext*)` continues to work unchanged
- `ILocalTPContext*` auto-converts to `ITPContext*` via inheritance

**Tests Created**:
- `tests/v2/unit/execution/compute_stages/stages/Test__TPAllreduceStage.cpp` - 20 unit tests (LOCAL TP only)
- `tests/v2/integration/execution/compute_stages/stages/Test__TPAllreduceStage_MPI.cpp` - 7 MPI integration tests (GLOBAL TP, `MPI_PROCS 2`)

### Phase 5: HierarchicalPPContext Global Stage Support (2-3 days) ✅ COMPLETE

**Goal**: Allow `HierarchicalPPContext` to have `GlobalTPContext` as a PP stage, enabling topologies where global (cross-MPI-rank) TP stages participate in pipeline parallel execution.

> **Architecture Note**: `HierarchicalPPContext` is a private implementation class inside `LocalPPContext.cpp`. It extends basic PP with support for stages that are TP domains (where a stage has multiple devices). We need to extend this to support stages that are **global TP domains** (where a stage spans multiple MPI ranks).

**Files to Modify**:

1. **`src/v2/collective/PPStage.h`** - Extend `PPStage` for global TP domains:
   - Add `GLOBAL_TP_DOMAIN` to `PPStageType` enum
   - Add `fromGlobalTPContext(std::shared_ptr<IGlobalTPContext>)` factory method
   - Add `std::shared_ptr<IGlobalTPContext> global_tp_context_` member
   - Add `asGlobalTPContext()` accessor (returns `IGlobalTPContext*`)
   - Add `isGlobalTPDomain()` query method
   - Update `representativeDevice()` to return local rank's device for global TP
   - Update `allDevices()` and `deviceCount()` for global TP (return local rank's device only)

2. **`src/v2/collective/PPStage.cpp`** - Implement global TP stage methods:
   - Add `GLOBAL_TP_DOMAIN` cases to all switch statements
   - `representativeDevice()`: Return local CPU device (global TP is CPU-only)
   - `allDevices()`: Return `{local_cpu_device}` (single device from local rank's perspective)
   - `deviceCount()`: Return `1` (local rank has 1 device in global TP domain)
   - `containsDevice()`: Check if device matches local rank's global TP device

3. **`src/v2/collective/LocalPPContext.cpp`** (`HierarchicalPPContext` class):
   - Update `HierarchicalPPConfig` to accept global TP stages
   - Update constructor to handle `GLOBAL_TP_DOMAIN` stage type
   - Update `transfer()` method with global TP transfer semantics (see below)
   - Update `transferAsync()` similarly

**Transfer Semantics for Global TP Stages**:

Global TP stages are CPU-only and span multiple MPI ranks. After global TP allreduce, all participating ranks have identical data on their local CPU.

| Transfer Direction | Behavior |
|--------------------|----------|
| **From Global TP → Single Device** | Local rank's CPU already has result after global TP allreduce. Transfer from local CPU to destination device. |
| **From Global TP → Local TP Domain** | Transfer from local CPU to local TP domain's representative device, then TP domain handles internal broadcast. |
| **From Single Device → Global TP** | Transfer activations to local CPU. Global TP allreduce will happen during layer computation, not during PP transfer. |
| **From Local TP → Global TP** | After local TP allreduce, transfer from local TP's representative device to local CPU. |

> **Key Insight**: PP transfers are **per-rank operations**. Each rank independently transfers data to/from its local device in the global TP domain. The global TP collective operation (allreduce) happens **during layer computation** via `TPAllreduceStage`, not during PP transfer.

**Tests**:

1. **Unit Tests** - `tests/v2/unit/collective/Test__PPStage_GlobalTP.cpp`:
   - `fromGlobalTPContext()` factory creates correct stage type
   - `isGlobalTPDomain()` returns correct value
   - `asGlobalTPContext()` returns valid pointer (or nullptr for non-global stages)
   - `representativeDevice()` returns local CPU device
   - `allDevices()` returns single-element vector with local CPU
   - `deviceCount()` returns 1
   - `describe()` includes "GlobalTP" in output

2. **MPI Integration Tests** - `tests/v2/integration/pipeline_parallel/Test__HierarchicalPP_GlobalTP.cpp`:
   - **2-rank topology**: `[LocalTP(cuda:0,cuda:1)] → [GlobalTP(0:cpu:0, 1:cpu:0)] → [SingleDevice(rocm:0)]`
   - PP transfer from local TP domain to global TP stage
   - PP transfer from global TP stage to single device
   - Full PP cycle with global TP stage in the middle
   - Verify data integrity across ranks after transfers
   - **CMake**: `MPI_PROCS 2` for cross-rank testing

**Backward Compatibility**:
- Existing code using `PPStage::fromTPContext()` continues to work unchanged
- `HierarchicalPPContext` with only local TP domains works unchanged
- No changes to `ILocalPPContext` interface

### Phase 6: Integration & End-to-End Testing (2-3 days)

**Goal**: Full integration testing with real models.

**Tests**:
- `tests/v2/integration/orchestration/Test__GlobalTP_Qwen2.cpp` - Real model with global TP
- `tests/v2/integration/parity/Test__GlobalTP_vs_LocalTP_Parity.cpp` - Verify correctness

---

## 6. File Summary

### Files Created (Phases 1-4)

| File | Phase | Purpose |
|------|-------|---------|
| `collective/ITPContext.h` | 1 | Base TP interface with `degree()`, `myIndex()`, `isLocal()`, `allreduce()` |
| `collective/IGlobalTPContext.h` | 2 | Global TP interface with `communicator()`, `domainId()`, `barrier()` |
| `collective/GlobalTPContext.h` | 2 | Global TP implementation header |
| `collective/GlobalTPContext.cpp` | 2 | Global TP implementation using UPICollectiveBackend |
| `collective/TPContextFactory.h` | 3 | Factory header for creating TP contexts |
| `collective/TPContextFactory.cpp` | 3 | Factory implementation with `createFromPlan()` |
| `tests/v2/unit/collective/Test__ITPContext.cpp` | 1 | ITPContext unit tests |
| `tests/v2/unit/collective/Test__GlobalTPContext.cpp` | 2 | GlobalTPContext unit tests (55 tests) |
| `tests/v2/unit/collective/Test__TPContextFactory.cpp` | 3 | Factory unit tests (24 tests) |
| `tests/v2/unit/execution/compute_stages/stages/Test__TPAllreduceStage.cpp` | 4 | TPAllreduceStage unit tests (20 tests, LOCAL TP only) |
| `tests/v2/integration/execution/compute_stages/stages/Test__TPAllreduceStage_MPI.cpp` | 4 | TPAllreduceStage MPI integration tests (7 tests, GLOBAL TP) |

### Files Modified (Phases 1-4)

| File | Phase | Changes |
|------|-------|---------|
| `collective/ILocalTPContext.h` | 1 | Inherit from `ITPContext`, add `isLocal()` override |
| `collective/LocalTPContext.h` | 1 | Add `myIndex()` implementation |
| `collective/LocalTPContext.cpp` | 1 | Implement `ITPContext` virtual methods |
| `execution/compute_stages/stages/LocalTPAllreduceStage.h` | 4 | Add `ITPContext*` support, keep backward compat |
| `execution/compute_stages/stages/LocalTPAllreduceStage.cpp` | 4 | Implement `ITPContext*`-based allreduce dispatch |
| `tests/v2/CMakeLists.txt` | 4 | Add MPI integration test target with `MPI_PROCS 2` |
| `collective/PPStage.h` | 5 | Add `GLOBAL_TP_DOMAIN` type, `fromGlobalTPContext()`, `isGlobalTPDomain()`, accessors |
| `collective/PPStage.cpp` | 5 | Handle global TP stage in all switch statements (representativeDevice, allDevices, etc.) |
| `collective/IGlobalTPContext.h` | 5 | Add `localDevice()` method |
| `collective/GlobalTPContext.cpp` | 5 | Implement `localDevice()` |
| `collective/LocalPPContext.cpp` | 5 | Update `HierarchicalPPContext::transfer()` for global TP stages |

### Files Created (Phase 5)

| File | Purpose |
|------|---------|
| `tests/v2/unit/collective/Test__PPStage_GlobalTP.cpp` | PPStage global TP unit tests (21 tests) |
| `tests/v2/integration/pipeline_parallel/Test__HierarchicalPP_GlobalTP.cpp` | MPI integration tests (13 tests, `MPI_PROCS 2`) |

### Phase 4 Note: No File Rename

> **Implementation Decision**: Phase 4 did NOT rename `LocalTPAllreduceStage` to `TPAllreduceStage`. Instead, the existing stage was **extended** to accept `ITPContext*` while maintaining full backward compatibility with `ILocalTPContext*`. A type alias `using TPAllreduceStage = LocalTPAllreduceStage;` was added for forward compatibility.

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MPI deadlocks in collective ops | Medium | High | Extensive barrier testing, timeout wrappers |
| Performance regression in local TP | Low | Medium | Benchmark before/after, virtual call overhead minimal |
| Interface incompatibility | Low | High | Phase 1 focuses on non-breaking changes |
| UPI bandwidth insufficient | Low | Low | Already validated at ~50 GB/s in existing tests |

---

## 8. Success Criteria

1. **Functional**: `GlobalTensorParallel(0:cpu:0, 1:cpu:0)` topology works end-to-end
2. **Performance**: Global TP allreduce within 10% of raw MPI_Allreduce
3. **Compatibility**: All existing `ILocalTPContext` code continues to work
4. **Test Coverage**: 90%+ coverage on new code paths
5. **Documentation**: All new interfaces fully documented

---

## 9. Dependencies

- Existing `UPICollectiveBackend` implementation
- Existing `TPDomainBuilder::createCPUCrossRankDomain()`
- Existing `RankExecutionPlan` global_tp fields
- MPI with `MPI_THREAD_MULTIPLE` support

---

## 10. Timeline

| Phase | Duration | Status | Notes |
|-------|----------|--------|-------|
| Phase 1: Base Interface | 1-2 days | ✅ Complete | ITPContext base interface |
| Phase 1.5: CLI Extensions | 2-3 days | 🔲 Deferred | Cross-rank device addressing (not blocking) |
| Phase 2: Global TP Impl | 2-3 days | ✅ Complete | IGlobalTPContext, GlobalTPContext (55 tests) |
| Phase 3: Factory & Wiring | 1-2 days | ✅ Complete | TPContextFactory (24 tests) |
| Phase 4: Stage Support | 2-3 days | ✅ Complete | TPAllreduceStage with ITPContext* (27 tests) |
| Phase 5: HierarchicalPP | 2-3 days | ✅ Complete | PPStage global TP domain (21 unit + 13 MPI integration tests) |
| Phase 6: Integration | 2-3 days | 🔲 Next | End-to-end model testing |

**Remaining Estimated: 2-3 days** (Phase 6)

---

## 11. Design Decisions (Resolved)

1. **Weight Distribution**: Global TP uses **equal 1/n distribution** (each participant takes equal slice). No proportional weights for global TP. Local TP retains proportional weight support for heterogeneous GPUs.

2. **Mutually Exclusive TP Types per PP Stage**: A **pipeline stage** (participant in `GlobalPipelineParallel` or `PipelineParallel`) uses **either local TP or global TP, not both**. There is no mixed local+global TP within a single pipeline stage. Different pipeline stages can use different TP types.

   > **Terminology Note**: "PP stage" / "pipeline stage" refers to a stage in a PipelineParallel context (e.g., a device or TP domain). This is distinct from "ComputeStage" which refers to nodes in the compute graph (e.g., `AllreduceStage`, `GemmStage`).

3. **Global TP Scope**: Global TP is **CPU-only, same physical machine, UPI interconnect**. No cross-machine global TP. No GPU participation in global TP (GPUs use local TP only).

### Implications

- `GlobalTPContext` does not need `weights` vector (always equal)
- `ITPContext::weights()` method stays in `ILocalTPContext` only
- No staging buffers needed in `GlobalTPContext` (CPU-to-CPU only)
- `HierarchicalPPContext` pipeline stage types are: single device, local TP domain, OR global TP domain
- `AllreduceStage` (ComputeStage) needs to accept `ITPContext*` to work with either local or global TP

---

## 12. Future Parallelism Features

This section outlines features planned after Global TP is complete. The per-rank graph architecture scales to all of these.

### 12.0 Qwen3 Kernel Requirements Analysis

Before implementing MoE support, we need to assess what kernel changes are required for Qwen3 (dense) and Qwen3-MoE models.

#### 12.0.1 Qwen3 Dense (e.g., Qwen3-8B)

**Configuration Comparison** (vs Qwen2.5):

| Parameter | Qwen2.5-0.5B | Qwen3-8B | Notes |
|-----------|--------------|----------|-------|
| `model_type` | `qwen2` | `qwen3` | Loader needs new type |
| `rope_theta` | 1,000,000 | 1,000,000 | Same (already parameterized) |
| `head_dim` | 64 | 128 | Already parameterized in attention |
| `num_hidden_layers` | 24 | 36 | Just a config value |
| `hidden_size` | 896 | 4096 | Just a config value |
| `vocab_size` | 151,936 | 151,936 | Same |

**Kernel Changes Required: NONE** ✅

The existing kernels already handle variable `head_dim` (64/128), `rope_theta`, and model dimensions. Only changes needed:

1. **GGUF Loader**: Add `model_type = "qwen3"` to architecture detection
2. **Config Parsing**: Ensure new Qwen3 keys are parsed correctly

**Estimated Effort**: 1 day (loader + config changes only)

---

#### 12.0.2 Qwen3-MoE (e.g., Qwen3-30B-A3B)

**MoE-Specific Configuration**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_type` | `qwen3_moe` | New architecture type |
| `num_experts` | 128 | Total experts per MoE layer |
| `num_experts_per_tok` | 8 | Top-k routing (8 of 128 activated) |
| `moe_intermediate_size` | 768 | FFN hidden size per expert |
| `num_hidden_layers` | 48 | Dense layers |
| `num_local_experts` | Not in config | Will be computed: `num_experts / EP_degree` |
| `shared_expert_intermediate_size` | 3584 | Shared expert FFN size |
| `decoder_sparse_step` | 1 | Every layer is MoE (not alternating) |

**New Kernels Required** 🔧

| Kernel | Purpose | Input/Output |
|--------|---------|--------------|
| `MoERouterKernel` | Compute router logits, top-k selection | `hidden_states` → `(expert_indices[B,k], expert_weights[B,k])` |
| `MoEDispatchKernel` | Permute tokens to experts | `(tokens, expert_indices)` → `permuted_tokens[E, tokens_per_expert]` |
| `MoECombineKernel` | Un-permute expert outputs, weighted sum | `(expert_outputs, weights, indices)` → `combined_output` |
| `ExpertFFNKernel` | Expert FFN (may reuse existing) | Same as `SwiGLU` but batched per-expert |

**New Collective Required** 🔧

| Collective | Purpose | Pattern |
|------------|---------|---------|
| `All-to-All` | Token ↔ Expert exchange | Each rank sends tokens to experts' owning ranks |

**Graph Stage Changes**:

```cpp
// Dense FFN (current):
GemmStage(gate_up)     → SwiGLUStage → GemmStage(down)

// MoE FFN (new):
MoERouterStage         → MoEDispatchStage
  → ExpertFFNStage(expert_0..N-1)  // Parallel expert compute
    → MoECombineStage
```

**Estimated Effort**:
- MoE Kernels: 2-3 weeks
- All-to-All Collective: 1 week
- MoE Graph/Stages: 1 week
- Testing/Parity: 1 week
- **Total: 5-6 weeks**

---

#### 12.0.3 Kernel Implementation Priority

```
Priority 1 (Required for Qwen3 dense):
 ✅ None - existing kernels work

Priority 2 (Required for Qwen3-MoE):
 🔧 MoERouterKernel - router logits + top-k selection
 🔧 MoEDispatchKernel - token permutation  
 🔧 MoECombineKernel - weighted combination
 🔧 All-to-All collective backend

Priority 3 (Optimization):
 🔧 ExpertFFNKernel - batched expert FFN (or reuse SwiGLU)
 🔧 Fused router+dispatch kernel
 🔧 Expert capacity management
```

---

### 12.1 MoE (Mixture of Experts) Parallelism

**Target Model**: Qwen3-30B-A3B (Qwen3-MoE)

**Architecture**: MoE models have sparse FFN layers where each token routes to a subset of "experts" (typically top-k of N experts). This enables larger models with less compute per token.

```
Token → Router → Select top-k experts
             ↓
     ┌───────┴───────┐
     ▼               ▼
  Expert 0        Expert 2     ← Only activated experts compute
     │               │
     └───────┬───────┘
             ▼
         Combine (weighted sum)
```

**Parallelism Strategies**:

| Strategy | Description | Communication |
|----------|-------------|---------------|
| **Expert Parallelism (EP)** | Each rank owns subset of experts | All-to-All (tokens ↔ experts) |
| **Expert + Tensor Parallelism** | Shard each expert across devices | All-to-All + AllReduce |
| **Expert Replication** | All ranks have all experts | None (but memory inefficient) |

**New Components Needed**:

| Component | Purpose |
|-----------|---------|
| `MoERouterStage` | Compute router logits, select top-k experts per token |
| `MoEDispatchStage` | All-to-All: send tokens to ranks owning selected experts |
| `MoECombineStage` | All-to-All: gather expert outputs, combine weighted |
| `IAllToAllBackend` | New collective pattern (not just AllReduce) |

**Graph Structure with EP**:
```
Rank 0 (owns Expert 0,1):       Rank 1 (owns Expert 2,3):
┌───────────────────────┐       ┌───────────────────────┐
│ MoERouterStage        │       │ MoERouterStage        │
│ MoEDispatchStage ─────┼─A2A──►│◄─ MoEDispatchStage    │
│ Expert0Stage          │       │ Expert2Stage          │
│ Expert1Stage          │       │ Expert3Stage          │
│ MoECombineStage ◄─────┼─A2A──►│── MoECombineStage     │
│ ResidualAddStage      │       │ ResidualAddStage      │
└───────────────────────┘       └───────────────────────┘
```

**Key Challenges**:
1. **All-to-All collective**: New communication primitive needed
2. **Load imbalance**: Popular experts get more tokens (capacity factor mitigation)
3. **Variable work per rank**: Different experts activated each iteration
4. **Router gradient routing**: For fine-tuning (not needed for inference)

**Estimated Effort**: 3-4 weeks

---

### 12.2 Speculative Decoding

**Goal**: Faster decode by using a small "draft" model to generate candidate tokens, verified in parallel by the main model.

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│ Draft Model (fast)    Generate K candidates            │
│ e.g., Qwen2-0.5B      [tok1, tok2, tok3, tok4, tok5]   │
└───────────────────────────┬─────────────────────────────┘
                            │ candidates
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Main Model (accurate)  Verify all K in one forward pass │
│ e.g., Qwen2-72B        Accept: [tok1, tok2, tok3] ✓     │
│                        Reject: tok4 (mismatch) ✗        │
└─────────────────────────────────────────────────────────┘
```

**Implementation Options**:

| Option | Draft Location | Main Location | Communication |
|--------|---------------|---------------|---------------|
| **Same-rank** | CPU | GPU TP domain | Host↔Device copy |
| **Separate ranks** | Dedicated rank | Other ranks | MPI_Send/Recv |
| **Same device** | Small model | Large model | Shared buffers |

**New Components Needed**:

| Component | Purpose |
|-----------|---------|
| `DraftModelRunner` | Runs draft model, generates K candidates |
| `SpeculativeVerifier` | Runs main model on K+1 positions in parallel |
| `TokenAcceptor` | Determines how many draft tokens to accept |
| `SpeculativeOrchestrator` | Coordinates draft → verify → accept loop |

**Graph Structure**:
```
No changes to compute graph structure!

Orchestrator manages two graphs:
┌─────────────────┐      ┌─────────────────┐
│ DraftGraph      │      │ MainGraph       │
│ (Qwen2-0.5B)    │ ──► │ (Qwen2-72B)     │
│ Generate K=5    │      │ Verify K+1=6    │
└─────────────────┘      └─────────────────┘
         ↑                        │
         └────── accepted count ──┘
```

**Key Challenges**:
1. **Draft model selection**: Speed vs acceptance rate tradeoff
2. **Batch size interaction**: How speculative decoding interacts with batching
3. **KV cache management**: Draft and main models need separate or shared caches
4. **Acceptance algorithm**: Tree-based vs linear speculation

**Estimated Effort**: 2-3 weeks

---

### 12.3 Dynamic Batching (Continuous Batching)

**Goal**: Maximize throughput by dynamically adding/removing sequences from the batch.

**Current (Static Batching)**:
```
Batch = [SeqA, SeqB, SeqC]
Wait until ALL complete → Return all results
```

**Dynamic Batching**:
```
Iter 1: [SeqA@5, SeqB@12, SeqC@1]
Iter 2: [SeqA@6, SeqB done!, SeqC@2, SeqD@0 ← NEW]
Iter 3: [SeqA@7, SeqC@3, SeqD@1, SeqE@0 ← NEW]
```

**New Components Needed**:

| Component | Purpose |
|-----------|---------|
| `BatchScheduler` | Manages sequence queue, decides batch composition |
| `PagedKVCache` | Paged memory management for variable-length KV cache |
| `RaggedAttention` | Attention kernel handling variable sequence lengths |
| `SequenceState` | Tracks per-sequence position, KV cache pages, completion |

**Graph Changes**:
```
Graph structure unchanged!

But stages must handle:
- Variable batch size each iteration
- Per-sequence KV cache lookup
- Ragged tensor shapes
```

**Memory Management (vLLM-style)**:
```
Physical KV Pages:     [Page0][Page1][Page2][Page3][Page4]...
                          ↑      ↑      ↑      ↑      ↑
Sequence A (len=150):  [Pg0]─[Pg1]─[Pg2]
Sequence B (len=50):            [Pg3]
Sequence C (len=200):  [Pg4]─[Pg5]─[Pg6]─[Pg7]
```

**Key Challenges**:
1. **Paged attention**: Kernel must handle non-contiguous KV cache
2. **Memory fragmentation**: Page allocation/deallocation
3. **Scheduling policy**: Which sequences to batch together
4. **Preemption**: Pause long sequences to serve short ones

**Estimated Effort**: 3-4 weeks

---

### 12.4 Feature Interaction Matrix

How these features interact with each other and existing parallelism:

| Feature | + Local TP | + Global TP | + Pipeline Parallel | + MoE EP |
|---------|-----------|-------------|---------------------|----------|
| **MoE EP** | AllReduce within expert | A2A across sockets | Experts in different PP stages | - |
| **Speculative** | Draft on CPU, verify on GPU TP | Draft with global TP | Draft and main in PP stages | Draft non-MoE, main MoE |
| **Dynamic Batch** | Same, variable batch size | Same, variable batch | Same, but scheduling complex | Capacity factor per batch |

---

### 12.5 Implementation Roadmap

```
                    2026
         Feb        Mar        Apr        May
         ├──────────┼──────────┼──────────┼──────────►
         │          │          │          │
Phase 7  │▓▓▓▓▓▓▓▓▓▓│▓▓▓       │          │  Global TP + CLI Extensions
         │          │          │          │
Phase 8  │          │  ▓▓▓▓▓▓▓▓│▓▓▓▓▓▓    │  MoE/Qwen3 Support
         │          │          │          │
Phase 9  │          │          │    ▓▓▓▓▓▓│▓▓▓▓  Dynamic Batching
         │          │          │          │
Phase 10 │          │          │          │▓▓▓▓▓  Speculative Decoding
         │          │          │          │
```

| Phase | Feature | Duration | Prerequisites |
|-------|---------|----------|---------------|
| 7 | Global TP + CLI Extensions | 2-3 weeks | (this plan: Phases 1-6 + CLI Phase 1.5) |
| 8 | MoE + Qwen3-30B-A3B | 5-6 weeks | Global TP (for CPU experts), All-to-All collective |
| 9 | Dynamic Batching | 3-4 weeks | Paged attention kernel, batch scheduler |
| 10 | Speculative Decoding | 2-3 weeks | Multi-graph orchestration |

---

### 12.6 Qwen3-MoE Specifics

**Model**: Qwen3-30B-A3B
- **Total params**: ~30B
- **Active params**: ~3B per token (hence "A3B")
- **Architecture**: MoE with top-k routing

**Expected Configuration**:
```yaml
model: qwen3-30b-a3b
moe:
  num_experts: 64        # Total experts per MoE layer
  num_experts_per_tok: 4 # top-k activated
  expert_parallelism: 8  # Experts sharded across 8 ranks
```

**Topology for Qwen3-MoE on 2-socket, 4-GPU system**:
```
GlobalPipelineParallel(
  # Attention layers: GPU TP
  PipelineParallel(
    TensorParallel(cuda:0, cuda:1),  # Attention on NVIDIA GPUs
    TensorParallel(rocm:0, rocm:1)   # Attention on AMD GPUs
  ),
  # MoE FFN layers: Expert Parallelism across all devices
  ExpertParallel(
    cuda:0: [Expert 0-7],
    cuda:1: [Expert 8-15],
    rocm:0: [Expert 16-23],
    rocm:1: [Expert 24-31],
    cpu:0:  [Expert 32-47],  # CPU handles more experts (memory)
    cpu:1:  [Expert 48-63]
  )
)
```

**Test Plan for Qwen3-MoE**:
1. Single-rank MoE (no EP) - verify correctness
2. 2-rank EP with All-to-All - verify token routing
3. Full topology with GPU TP + CPU EP - verify end-to-end
4. Parity test vs HuggingFace transformers

---

## 13. References

- [TPDomain.h](../../../../src/v2/config/TPDomain.h) - Existing cross-rank domain support
- [UPIBackend.h](../../../../src/v2/collective/backends/UPIBackend.h) - MPI-based collective backend
- [ILocalTPContext.h](../../../../src/v2/collective/ILocalTPContext.h) - Current local TP interface
- [RankExecutionPlan.h](../../../../src/v2/execution/mpi_orchestration/RankExecutionPlan.h) - Global TP fields
- [Test__TPDomainMPI.cpp](../../../../tests/v2/integration/collective/Test__TPDomainMPI.cpp) - Cross-rank domain tests

### External References

- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) - Paged KV cache management
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) - Leviathan et al.
- [GShard](https://arxiv.org/abs/2006.16668) - MoE with expert parallelism
- [Qwen3 Technical Report](https://qwenlm.github.io/) - Qwen3-MoE architecture details
