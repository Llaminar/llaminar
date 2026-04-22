# Buffer Harmonization Proposal: Unified Tensor Lifecycle Management

**Status**: Proposal  
**Author**: Generated from codebase investigation  
**Date**: 2026-03-10  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture (As-Is)](#2-current-architecture-as-is)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Proposed Architecture (To-Be)](#4-proposed-architecture-to-be)
5. [Component Design](#5-component-design)
6. [Migration Strategy](#6-migration-strategy)
7. [Testing Strategy](#7-testing-strategy)

---

## 1. Problem Statement

Tensor Parallel (TP) inference is plagued by buffer-related bugs: stale data, wrong-device access, untracked coherence, silent aliasing violations, and stream synchronization races. These stem from a fundamental design issue: **buffer management responsibilities are distributed across 6+ layers of abstraction with no single source of truth**.

Today, every compute stage can (and often does) bypass the automatic coherence system, manually call `ensureOnDevice()`, `mark_device_dirty()`, `data()`, or perform raw `memcpy` — making it impossible to reason about tensor state at any point in time.

### Pain Summary

| Category | Bug Class | Example | Severity |
|----------|-----------|---------|----------|
| **Dual coherence** | Stage does manual `ensureOnDevice()` + executor also does it | KVCacheAppendStage lines 183-615 | Critical |
| **Untracked transfers** | BAR backend does 40+ raw `cudaMemcpy` with no coherence tracking | PCIeBARBackend lines 881-2626 | Critical |
| **String-based routing** | Buffer name suffix matching determines BAR allocation | Qwen2BufferSpec lines 46-118 | High |
| **data() in hot path** | GPU→CPU sync triggered by `->data()` inside stages | KVCacheAppend (15 calls), GEMMStage, AllreduceStage | High |
| **PP outside graph** | Pipeline-parallel copies bypass ComputeGraph entirely | Qwen2Graph lines 552-579 | High |
| **No buffer namespace** | Same name → different memory per device in RankOrchestrator | logits_local on device 0 vs. device 1 | Medium |

---

## 2. Current Architecture (As-Is)

### 2.1 Ownership & Wiring Chain

```
DeviceGraphOrchestrator    ← owns InferenceState (shared_ptr<TensorBase> buffers)
    │
    ├── InferenceState     ← owns ~20 activation buffers + KV caches
    │       │
    │       └── passes raw ITensor* into ...
    │
    ├── Qwen2Graph         ← receives buffer pointers, builds Stage Params structs
    │       │
    │       └── captures ITensor* into stage Params (by-value, non-owning)
    │
    ├── DeviceGraphExecutor  ← executes ComputeGraph
    │       │
    │       ├── getDumpInfo()      ← asks stage for its tensor list (originally a debug feature!)
    │       ├── extractInputBuffers()  ← builds CoherenceBuffer from StageDumpInfo
    │       ├── cohereInputs()     ← calls ensureOnDevice() on each tensor
    │       ├── stage->execute()   ← stage may ALSO call ensureOnDevice() internally!
    │       └── markOutputsDirty() ← marks GPU as authoritative
    │
    └── DeviceGraphBufferManager ← allocates scratch/workspace buffers
            │
            └── LivenessAnalyzer ← computes aliasing for non-overlapping scratch buffers
```

### 2.2 The Seven Layers That Touch Buffers

| Layer | File | Role | Problem |
|-------|------|------|---------|
| **1. InferenceState** | DeviceGraphOrchestrator.h | Buffer ownership | Owns 20+ buffers as `shared_ptr<TensorBase>` |
| **2. Qwen2Graph** | Qwen2Graph.cpp | Buffer wiring | Captures raw `ITensor*` into Stage Params via manual wiring |
| **3. DeviceGraphExecutor** | DeviceGraphExecutor.cpp | Coherence at boundaries | Uses `getDumpInfo()` to discover tensors → coheres them |
| **4. StageCoherence** | StageCoherence.h | Coherence operations | Free functions: `cohereInputs()`, `markOutputsDirty()` |
| **5. Stages** | Various *Stage.cpp | Internal coherence | Many stages do manual `ensureOnDevice()` / `data()` internally |
| **6. KernelFactory+Kernels** | Various | Workspace management | Kernels bind workspace via `IWorkspaceConsumer` interface |
| **7. PCIeBAR/MPI backends** | PCIeBARBackend.cpp | Collective operations | Raw `cudaMemcpy` with no tensor tracking |

### 2.3 How StageDumpInfo Became Load-Bearing

`StageDumpInfo` was designed as a **debugging/inspection** mechanism — "tell me what tensors this stage uses so I can dump them." Over time it became the **primary interface for coherence**:

```cpp
// DeviceGraphExecutor.cpp — coherence depends entirely on getDumpInfo()
const StageDumpInfo &cached_dump_info = node.stage->getDumpInfo();
auto inputs = extractInputBuffers(cached_dump_info);   // ← StageDumpInfo drives coherence
cohereInputs(inputs, target_device);
auto weights = extractWeightBuffers(cached_dump_info);  // ← also through StageDumpInfo
cohereInputs(weights, target_device);
```

This means:
- If a stage omits a tensor from `buildDumpInfoImpl()`, **coherence silently skips it**
- If a stage adds a tensor to DumpInfo that it doesn't actually use, **redundant H2D/D2H transfers**
- There's no compile-time guarantee that DumpInfo matches actual usage

---

## 3. Root Cause Analysis

### 3.1 No Separation of Declaration from Execution

Stages currently have two ways to declare their buffers:
- `getBufferRequirements()` — for DeviceGraphBufferManager allocation (optional, mostly unused)
- `buildDumpInfoImpl()` — for StageDumpInfo (drives coherence, profiling, dumping)

And they have one way to actually USE buffers:
- `execute()` — where they access `params_.input`, `params_.output`, etc. directly

**There is no enforcement that the declared buffers match the used buffers.** A stage can declare `{A, B}` as inputs in DumpInfo but actually read `{A, B, C}` in execute(). Tensor C gets no coherence.

### 3.2 No Buffer Access Control

Stages hold raw `ITensor*` pointers and can call any method:
- `tensor->data()` — triggers GPU→CPU sync on hot path
- `tensor->mutable_data()` — triggers sync AND takes CPU ownership
- `tensor->ensureOnDevice(dev)` — manual H2D, bypassing executor
- `tensor->mark_device_dirty()` — manual dirty marking

There's no way to prevent a stage from calling `data()` on a GPU-authoritative tensor in a decode loop, which silently serializes execution.

### 3.3 Coherence Is Per-Tensor, But Execution Is Per-Stage

The coherence system tracks state per-tensor (`host_authoritative` vs. `device_authoritative`), but never validates the system-level invariant: **at any given point in graph execution, exactly one consumer should be reading/writing each tensor**.

Without this, aliased buffers (where scratch buffer A overlaps with scratch buffer B from a different stage) can silently corrupt data if the liveness analysis has a bug.

---

## 4. Proposed Architecture (To-Be)

### 4.1 Design Principles

1. **Stages are pure compute**: They receive typed, device-ready buffers and return results. No coherence calls, no `data()`, no `ensureOnDevice()`.
2. **Single source of truth**: One component (`BufferArena`) owns all activation buffers and tracks their state.
3. **Declarative buffer contracts**: Stages declare their I/O requirements at construction time with strongly-typed descriptors. The runtime validates these match actual usage.
4. **Device-opaque execution**: Stages operate on `BufferView`s that abstract away host/device location. The runtime handles all transfers.
5. **Collective-aware coherence**: MPI/NCCL/BAR operations are first-class coherence events, not side-channel hacks.

### 4.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BufferArena (NEW)                                     │
│                                                                              │
│  Single owner of ALL activation/scratch/workspace buffers.                   │
│  Tracks coherence state, device placement, and active borrows.               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  BufferRegistry                                                        │  │
│  │    key: BufferId (typed enum, not string)                              │  │
│  │    val: ManagedBuffer { TensorBase*, CoherenceState, ActiveBorrows }   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  CoherenceTracker                                                      │  │
│  │    - Per-buffer: { host_dirty, device_dirty, device_id, stream_event } │  │
│  │    - prepareForRead(buf, device) → ensures data on target device       │  │
│  │    - markWritten(buf, device, stream) → updates authoritative copy     │  │
│  │    - assertNoBorrows(buf) → validates no active readers on write       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  AliasingManager                                                       │  │
│  │    - Groups scratch buffers by non-overlapping lifetimes               │  │
│  │    - Enforces: no two active borrows for aliased buffers               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                   issues BufferView handles
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BufferView<T, Access> (NEW)                              │
│                                                                              │
│  RAII handle for typed, access-controlled tensor access.                     │
│  Returned by BufferArena::borrow<T, Access>(BufferId).                       │
│                                                                              │
│  template<typename T, BufferAccess Access>                                   │
│  class BufferView {                                                          │
│      const T* read_ptr() const;    // Always available                       │
│      T* write_ptr();               // Only if Access == WRITE or READWRITE   │
│      size_t rows() const;                                                    │
│      size_t cols() const;                                                    │
│      DeviceId device() const;      // Which device data is on                │
│      // No data(), no ensureOnDevice(), no mark_device_dirty()               │
│  };                                                                          │
│                                                                              │
│  Access modes:                                                               │
│    READ:       read_ptr() only, no write_ptr()                               │
│    WRITE:      write_ptr() only, data is undefined on entry                  │
│    READWRITE:  both read_ptr() and write_ptr()                               │
│                                                                              │
│  Destructor returns borrow to BufferArena.                                   │
│  BufferArena validates no overlapping writes to aliased buffers.             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                   stages receive views
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     IComputeStage (REFORMED)                                 │
│                                                                              │
│  Stages declare a StageBufferContract (replaces getDumpInfo for coherence):  │
│                                                                              │
│  struct StageBufferContract {                                                │
│      std::vector<BufferBinding> inputs;   // {BufferId, TensorType, Access}  │
│      std::vector<BufferBinding> outputs;  // {BufferId, TensorType, Access}  │
│      std::vector<BufferBinding> weights;  // {BufferId, TensorType, READ}    │
│      std::vector<WorkspaceNeed>  scratch; // {size, alignment, optional}     │
│  };                                                                          │
│                                                                              │
│  // New execute signature:                                                   │
│  virtual bool execute(const StageBoundBuffers& buffers) = 0;                │
│                                                                              │
│  // StageBoundBuffers holds the device-ready BufferViews:                    │
│  struct StageBoundBuffers {                                                  │
│      BufferView<T, READ>  input(const char* name) const;                     │
│      BufferView<T, WRITE> output(const char* name) const;                    │
│      BufferView<T, READ>  weight(const char* name) const;                    │
│      void* scratch(const char* name) const;                                  │
│  };                                                                          │
│                                                                              │
│  Stages CANNOT:                                                              │
│    - Call data() / mutable_data() / ensureOnDevice() / mark_device_dirty()  │
│    - Access any buffer they didn't declare in their contract                 │
│    - Write to READ-only views (compile-time enforced via const T*)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                    executor orchestrates
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GraphExecutor (REFORMED)                                 │
│                                                                              │
│  For each stage in topological order:                                        │
│                                                                              │
│  1. Read stage's StageBufferContract                                         │
│  2. For each input/weight:                                                   │
│       arena.prepareForRead(bufferId, target_device)                          │
│  3. For each output:                                                         │
│       arena.prepareForWrite(bufferId, target_device)                         │
│  4. Build StageBoundBuffers from arena borrows                               │
│  5. stage->execute(bound_buffers)                                            │
│  6. For each output:                                                         │
│       arena.markWritten(bufferId, target_device, stream)                     │
│  7. Release all borrows                                                      │
│                                                                              │
│  Collective stages (allreduce/allgather) are ALSO just stages:               │
│  - Contract declares buffer as READWRITE                                     │
│  - Arena handles pre/post coherence identically                              │
│  - No special CoherencePolicy::NONE needed                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 How This Solves Each Pain Point

| Current Problem | Solution |
|----------------|----------|
| **Dual coherence** (executor + stage both do it) | Stages can't do coherence — no access to raw tensors, only BufferViews |
| **Untracked BAR transfers** | BAR backend becomes a `CollectiveStage` with a proper contract; arena tracks all transfers |
| **String-based buffer routing** | `BufferId` is a typed enum, not a string. BAR-backed vs. regular is a property of the ManagedBuffer in the arena, not a name suffix |
| **data() in hot path** | `BufferView` has no `data()` method. Read access returns device-appropriate pointer |
| **PP copies outside graph** | PP transfers become explicit `PPTransferStage` nodes in the graph with proper contracts |
| **No buffer namespace** | `BufferArena` is per-device in multi-device mode. `BufferId::LOGITS_LOCAL` on device 0 and device 1 explicitly resolve to different ManagedBuffers |
| **StageDumpInfo driving coherence** | `StageBufferContract` is the single declaration. StageDumpInfo reverts to its original debugging-only role, auto-generated from the contract |

---

## 5. Component Design

### 5.1 BufferId — Typed Buffer Enumeration

Replace string-based buffer identification with a typed enum:

```cpp
// BufferIds are descriptive and self-documenting
enum class BufferId : uint32_t {
    // Cross-layer persistent buffers
    HIDDEN_STATE,           // Main hidden state [batch, d_model]
    LOGITS,                 // Final logits [batch, vocab_size]
    LOGITS_LOCAL,           // Column-parallel partial logits
    
    // Per-layer activation buffers (shared across layers)
    NORMALIZED,             // RMSNorm output
    RESIDUAL,               // Residual stream
    Q_PROJ,                 // Q projection output
    K_PROJ,                 // K projection output  
    V_PROJ,                 // V projection output
    Q_ROPE,                 // Post-RoPE Q (HybridQ16)
    K_ROPE,                 // Post-RoPE K (HybridQ16)
    ATTN_OUTPUT,            // Attention context output
    ATTN_PROJ,              // Wo projection output
    GATE_PROJ,              // FFN gate projection
    UP_PROJ,                // FFN up projection
    FFN_OUTPUT,             // FFN down projection output
    
    // Workspace buffers
    ATTN_SCORES_WORKSPACE,
    ATTN_CONTEXT_WORKSPACE,
    GEMM_WORKSPACE,
    
    // Collective staging buffers
    ALLREDUCE_STAGING,      // Staging buffer for allreduce
    ALLGATHER_STAGING,      // Staging buffer for allgather
    
    // Dynamic: KV cache handled separately (per-layer)
    // Use BufferId::kv_key(layer) and BufferId::kv_value(layer)
    
    _COUNT
};

// Per-layer KV cache uses a separate typed ID:
struct KVBufferId {
    int layer;
    enum { KEY, VALUE } type;
};
```

### 5.2 BufferArena — Single Source of Truth

```cpp
class BufferArena {
public:
    // =========================================================================
    // Construction (called once during graph setup)
    // =========================================================================
    
    /// Register a buffer with its shape, type, and device
    void registerBuffer(BufferId id, TensorShape shape, TensorType dtype, 
                        DeviceId home_device);
    
    /// Register aliasing: scratch_a and scratch_b can share storage
    /// (validated by AliasingManager based on graph liveness analysis)
    void registerAlias(BufferId a, BufferId b);
    
    /// Register a buffer backed by external memory (weights, BAR buffers)
    void registerExternalBuffer(BufferId id, ITensor* tensor);
    
    /// Allocate all registered buffers. After this, all buffers are available.
    bool allocate();
    
    // =========================================================================
    // Runtime (called per-stage by GraphExecutor)
    // =========================================================================
    
    /// Prepare buffer for reading on target device.
    /// If data is on wrong device, performs H2D/D2H transfer.
    /// Returns false on transfer failure.
    bool prepareForRead(BufferId id, DeviceId target);
    
    /// Prepare buffer for writing on target device.
    /// Allocates device buffer if needed, does NOT transfer data.
    /// Validates no active READ borrows exist (aliasing safety).
    bool prepareForWrite(BufferId id, DeviceId target);
    
    /// Mark buffer as written on device (sets device as authoritative).
    /// Optionally records stream event for async D2H later.
    void markWritten(BufferId id, DeviceId device, void* stream = nullptr);
    
    // =========================================================================
    // Typed access (compile-time type safety)
    // =========================================================================
    
    /// Get typed read-only view (for stage inputs/weights)
    template<typename T>
    BufferView<T, READ> borrowRead(BufferId id);
    
    /// Get typed write view (for stage outputs)
    template<typename T>
    BufferView<T, WRITE> borrowWrite(BufferId id);
    
    /// Get typed read-write view (for in-place operations like allreduce)
    template<typename T>
    BufferView<T, READWRITE> borrowReadWrite(BufferId id);
    
    /// Release a borrow (also called automatically by BufferView destructor)
    void releaseBorrow(BufferId id, BufferAccess access);
    
    // =========================================================================
    // Introspection (for debugging and testing)
    // =========================================================================
    
    /// Get current coherence state of a buffer
    CoherenceState getCoherenceState(BufferId id) const;
    
    /// Get the underlying tensor (for DumpInfo/debugging only, not for stages)
    ITensor* getUnderlyingTensor(BufferId id) const;
    
    /// Validate no borrows are active (call between graph executions)
    bool validateNoBorrowsActive() const;
    
private:
    struct ManagedBuffer {
        BufferId id;
        std::unique_ptr<TensorBase> owned_tensor;    // If arena owns it
        ITensor* external_tensor = nullptr;           // If externally owned (weights)
        
        CoherenceState coherence;
        int active_read_borrows = 0;
        bool active_write_borrow = false;
        
        // Aliasing group (-1 if not aliased)
        int alias_group = -1;
        
        ITensor* tensor() { return owned_tensor ? owned_tensor.get() : external_tensor; }
    };
    
    std::array<ManagedBuffer, static_cast<size_t>(BufferId::_COUNT)> buffers_;
    AliasingManager aliasing_;
    
    // Borrow safety validation (debug builds only)
    void validateBorrowSafe(BufferId id, BufferAccess access) const;
};
```

### 5.3 StageBufferContract — Declarative I/O Specification

```cpp
/// Describes one buffer binding for a stage
struct BufferBinding {
    BufferId id;                // Which buffer
    TensorType expected_dtype;  // Expected tensor type (for validation)
    BufferAccess access;        // READ, WRITE, or READWRITE
    
    // Optional: dynamic sizing (for decode mode where rows < tensor capacity)
    // If set, overrides tensor->rows() for kernel calls
    std::function<size_t()> dynamic_rows = nullptr;
};

/// Complete I/O contract for a stage
struct StageBufferContract {
    std::vector<BufferBinding> inputs;    // Read-only buffers
    std::vector<BufferBinding> outputs;   // Write-only buffers  
    std::vector<BufferBinding> weights;   // Read-only model parameters
    std::vector<BufferBinding> inouts;    // Read-write (e.g., allreduce in-place)
    
    struct WorkspaceDesc {
        const char* name;       // For lookup
        size_t size_bytes;      // How much scratch
        size_t alignment;       // Alignment requirement
        bool required;          // Fail if can't allocate?
    };
    std::vector<WorkspaceDesc> workspaces;
    
    // Builder pattern for ergonomic construction in stage constructors:
    static StageBufferContract build() { return {}; }
    StageBufferContract& addInput(BufferId id, TensorType dtype) { 
        inputs.push_back({id, dtype, BufferAccess::READ}); return *this; 
    }
    StageBufferContract& addOutput(BufferId id, TensorType dtype) { 
        outputs.push_back({id, dtype, BufferAccess::WRITE}); return *this; 
    }
    StageBufferContract& addWeight(BufferId id, TensorType dtype) { 
        weights.push_back({id, dtype, BufferAccess::READ}); return *this; 
    }
    StageBufferContract& addInOut(BufferId id, TensorType dtype) { 
        inouts.push_back({id, dtype, BufferAccess::READWRITE}); return *this; 
    }
    StageBufferContract& addWorkspace(const char* name, size_t size, 
                                       size_t align = 64, bool required = true) {
        workspaces.push_back({name, size, align, required}); return *this;
    }
};
```

**Example: GEMMStage with new contract**:

```cpp
class GEMMStage : public IComputeStage {
public:
    GEMMStage(BufferId input, BufferId weight, BufferId output,
              int m, int n, int k, DeviceId device)
        : IComputeStage(device)
        , input_id_(input), weight_id_(weight), output_id_(output)
        , m_(m), n_(n), k_(k) 
    {}
    
    StageBufferContract bufferContract() const override {
        return StageBufferContract::build()
            .addInput(input_id_, TensorType::FP32)
            .addWeight(weight_id_, TensorType::IQ4_NL)    // or whatever weight type
            .addOutput(output_id_, TensorType::FP32);
    }
    
    bool execute(const StageBoundBuffers& buffers) override {
        // Buffer views are already device-ready. No coherence calls needed.
        auto input  = buffers.input<float>(input_id_);
        auto weight = buffers.weight(weight_id_);  // returns ITensor* for kernel dispatch
        auto output = buffers.output<float>(output_id_);
        
        auto* gemm = KernelFactory::getOrCreateGemm(weight.tensor());
        return gemm->multiply(input.read_ptr(), output.write_ptr(), m_, n_, k_);
    }
    
private:
    BufferId input_id_, weight_id_, output_id_;
    int m_, n_, k_;
};
```

**Example: AllreduceStage with new contract**:

```cpp
class AllreduceStage : public IComputeStage {
public:
    AllreduceStage(BufferId buffer, CollectiveContext* ctx, DeviceId device)
        : IComputeStage(device), buffer_id_(buffer), ctx_(ctx) {}
    
    StageBufferContract bufferContract() const override {
        return StageBufferContract::build()
            .addInOut(buffer_id_, TensorType::FP32);  // In-place operation
    }
    
    bool execute(const StageBoundBuffers& buffers) override {
        auto buf = buffers.inout<float>(buffer_id_);
        return ctx_->allreduce(buf.write_ptr(), buf.rows() * buf.cols());
        // No manual coherence needed — arena handles it
    }
    
private:
    BufferId buffer_id_;
    CollectiveContext* ctx_;
};
```

### 5.4 StageBoundBuffers — Device-Ready Buffer Delivery

```cpp
/// Immutable set of buffer views bound to a stage for one execution
/// Built by GraphExecutor from BufferArena borrows
class StageBoundBuffers {
public:
    /// Get read-only typed pointer for an input buffer
    template<typename T>
    BufferView<T, READ> input(BufferId id) const;
    
    /// Get write-only typed pointer for an output buffer
    template<typename T>
    BufferView<T, WRITE> output(BufferId id) const;
    
    /// Get read-only typed pointer for a weight buffer
    template<typename T>
    BufferView<T, READ> weight(BufferId id) const;
    
    /// Get read-write typed pointer for an in-place buffer
    template<typename T>
    BufferView<T, READWRITE> inout(BufferId id) const;
    
    /// Get raw workspace pointer
    void* workspace(const char* name) const;
    
    /// Get weight as ITensor* for kernel dispatch (e.g., KernelFactory lookup)
    ITensor* weightTensor(BufferId id) const;
    
private:
    friend class GraphExecutor; // Only GraphExecutor can construct
    struct Entry {
        BufferId id;
        BufferAccess access;
        ITensor* tensor;
        void* device_ptr;       // The actual read/write pointer (on target device)
        size_t rows, cols;
    };
    std::vector<Entry> entries_;
    std::unordered_map<const char*, void*> workspaces_;
};
```

### 5.5 BufferView — RAII Typed Access Handle

```cpp
enum class BufferAccess { READ, WRITE, READWRITE };

template<typename T, BufferAccess Access>
class BufferView {
public:
    /// Read pointer (always available for READ and READWRITE)
    const T* read_ptr() const requires (Access != BufferAccess::WRITE) {
        return static_cast<const T*>(device_ptr_);
    }
    
    /// Write pointer (only for WRITE and READWRITE)
    T* write_ptr() requires (Access != BufferAccess::READ) {
        return static_cast<T*>(device_ptr_);
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t numel() const { return rows_ * cols_; }
    DeviceId device() const { return device_; }
    
    /// Get underlying ITensor* (for kernel dispatch only, not for data access)
    ITensor* tensor() const { return tensor_; }
    
    // No data(), no ensureOnDevice(), no mark_device_dirty()
    // No mutable_data(), no ensureOnHost()
    
private:
    friend class StageBoundBuffers;
    void* device_ptr_;      // Already on correct device
    ITensor* tensor_;       // For kernel type dispatch
    size_t rows_, cols_;
    DeviceId device_;
};
```

### 5.6 CoherenceTracker — Internal to BufferArena

```cpp
/// Per-buffer coherence state (internal to BufferArena)
struct CoherenceState {
    enum Authority { HOST, DEVICE, UNINITIALIZED } authority = UNINITIALIZED;
    DeviceId authoritative_device;
    void* completion_event = nullptr;   // GPU stream event for async sync
    bool event_recorded = false;
    
    bool needsTransferTo(DeviceId target) const {
        if (authority == UNINITIALIZED) return false;  // Write-only, no transfer needed
        if (authority == HOST && target.is_gpu()) return true;   // H2D needed
        if (authority == DEVICE && target.is_cpu()) return true; // D2H needed
        if (authority == DEVICE && target != authoritative_device) return true; // D2D needed
        return false;
    }
};

class CoherenceTracker {
public:
    /// Ensure buffer data is available on target device for reading
    bool prepareForRead(ManagedBuffer& buf, DeviceId target);
    
    /// Ensure buffer has allocated storage on target device for writing
    /// Does NOT transfer data (kernel will write it)
    bool prepareForWrite(ManagedBuffer& buf, DeviceId target);
    
    /// Mark buffer as written on device (device becomes authoritative)
    void markWritten(ManagedBuffer& buf, DeviceId device, void* stream);
    
    /// Wait for pending GPU writes to complete before CPU read
    void waitForCompletion(ManagedBuffer& buf);
    
private:
    IBackend* getBackendFor(DeviceId device);
};
```

### 5.7 GraphExecutor (Reformed Execute Loop)

```cpp
bool GraphExecutor::executeStage(ComputeNode& node) {
    auto* stage = node.stage.get();
    const auto& contract = stage->bufferContract();
    DeviceId target = node.device.is_valid() ? node.device : stage->device();
    
    // ─── 1. COHERE INPUTS (read-only) ────────────────────────────
    for (const auto& binding : contract.inputs) {
        if (!arena_->prepareForRead(binding.id, target)) return false;
    }
    for (const auto& binding : contract.weights) {
        if (!arena_->prepareForRead(binding.id, target)) return false;
    }
    
    // ─── 2. COHERE OUTPUTS (allocate device storage, no transfer) ─
    for (const auto& binding : contract.outputs) {
        if (!arena_->prepareForWrite(binding.id, target)) return false;
    }
    
    // ─── 3. COHERE INOUTS (need both read AND write) ─────────────
    for (const auto& binding : contract.inouts) {
        if (!arena_->prepareForRead(binding.id, target)) return false;
        // prepareForRead ensures data is on device
        // No prepareForWrite needed — data is already there
    }
    
    // ─── 4. BUILD BOUND BUFFERS (zero-alloc: reuse cached) ──────
    StageBoundBuffers bound = buildBoundBuffers(contract, target);
    
    // ─── 5. EXECUTE ──────────────────────────────────────────────
    if (!stage->execute(bound)) return false;
    
    // ─── 6. MARK OUTPUTS WRITTEN ────────────────────────────────
    void* stream = getStreamForDevice(target);
    for (const auto& binding : contract.outputs) {
        arena_->markWritten(binding.id, target, stream);
    }
    for (const auto& binding : contract.inouts) {
        arena_->markWritten(binding.id, target, stream);
    }
    
    return true;
}
```

### 5.8 How TP/PP/Single-Device Modes Become Uniform

| Mode | Current Approach | New Approach |
|------|-----------------|-------------|
| **Single device** | InferenceState owns buffers, Qwen2Graph wires them, DeviceGraphExecutor coheres | BufferArena owns all buffers. Same execute loop. |
| **Tensor Parallel** | RankOrchestrator creates N DeviceGraphOrchestrators, each with own InferenceState. BAR buffers registered manually by name. AllreduceStage has `CoherencePolicy::NONE`. | Each device gets its own BufferArena. AllreduceStage is a normal stage with `addInOut()`. BAR buffers are registered via `registerExternalBuffer()` with `BufferId::ALLREDUCE_STAGING`. |
| **Pipeline Parallel** | PP transfers happen outside the graph in Qwen2Graph::buildPartialForwardGraph(). Manual backend->copy() calls. | PPTransferStage is a first-class node in the graph with contract: `addInput(HIDDEN_STATE)` + `addOutput(HIDDEN_STATE)`. Uses CollectiveContext for the transfer. Arena handles coherence. |

**For TP specifically**, the per-device BufferArena eliminates the "same name, different memory" problem. Device 0's arena has `BufferId::LOGITS_LOCAL` pointing to VRAM on GPU 0. Device 1's arena has the same BufferId pointing to VRAM on GPU 1. The executor for each device only interacts with its own arena.

**For collective operations** (allreduce, allgather), the collective backend accesses buffers through the arena's `borrowReadWrite()`, which returns device-appropriate pointers. BAR-backed buffers are registered as external buffers with proper device affinity, so the arena tracks their coherence state just like any other buffer.

---

## 6. Migration Strategy

### Phase 1: Introduce BufferArena alongside InferenceState (Non-Breaking)

1. Create `BufferArena`, `BufferView`, `CoherenceTracker`, `BufferId` classes
2. Create `StageBufferContract` alongside existing `StageDumpInfo`
3. Add `virtual StageBufferContract bufferContract() const` to `IComputeStage` with default implementation that returns empty contract (backward compatible)
4. Unit test BufferArena independently (allocation, coherence, aliasing, borrow tracking)

**No stage changes yet. Both systems coexist.**

### Phase 2: Wire BufferArena into DeviceGraphOrchestrator

1. DeviceGraphOrchestrator creates a BufferArena and registers all buffers from InferenceState
2. DeviceGraphExecutor checks: if `stage->bufferContract()` is non-empty, use new path; otherwise fall back to old getDumpInfo path
3. Migrate one simple stage (e.g., SwiGLUStage) to the new contract as proof of concept
4. Validate single-device inference still produces correct output

**Gradual rollout: stages opt-in one at a time.**

### Phase 3: Migrate All Stages

Priority order (highest-impact first):
1. **KVCacheAppendStage** — worst offender with 15+ manual coherence calls
2. **GEMMStage** / **FusedQKVGEMMStage** — most numerous
3. **FusedAttentionWoStage** — complex multi-tensor
4. **AllreduceStage** / **AllGatherStage** — TP-critical
5. **Embedding**, **RMSNorm**, **SwiGLU**, **ResidualAdd** — straightforward
6. **LMHead**, **RoPE**, **QuantizeQ16_1** — remaining

For each stage:
- Add `bufferContract()` override
- Change `execute()` signature to accept `StageBoundBuffers`
- Remove all internal `ensureOnDevice()`, `data()`, `mark_device_dirty()` calls
- Run parity tests

### Phase 4: Wire BufferArena into PP Path (RankOrchestrator)

The PP path currently uses a completely separate allocation mechanism (`PerStageBufferPool` + `PPStageBufferSpec` + `ActivationBuffers` struct). This phase eliminates that parallel path so all inference modes use BufferArena.

1. Replace `PerStageBufferPool::allocateStageBuffers()` with per-PP-stage BufferArena registration
   - Each PP stage gets its own BufferArena instance (one per device, same as Phase 2 for single-device)
   - Buffer shapes come from model config (d_model, n_heads, intermediate_size, etc.), same as `PPStageBufferSpec` today
   - Device placement per arena matches the PP stage's assigned device
2. Replace `PPStageBufferSpec` sizing logic with BufferArena `registerBuffer()` calls
   - `PPStageBufferSpec::total_tokens()` becomes `max_seq_len * batch_size` passed to arena registration
   - BAR-backed buffers registered via `registerExternalBuffer()` instead of `Qwen2BufferSpec` suffix matching
3. Replace `ActivationBuffers` struct field access with `StageBoundBuffers` from the arena
   - Stages that currently access `activation_buffers->residual`, `activation_buffers->Q`, etc. switch to `buffers.input<float>(BufferId::RESIDUAL)`, `buffers.output<float>(BufferId::Q_PROJ)`
   - PP transfer stages (currently raw `backend->copy()` outside the graph) become `PPTransferStage` nodes with contracts: `addInput(BufferId::HIDDEN_STATE)` on source device, `addOutput(BufferId::HIDDEN_STATE)` on target device
4. Delete `PerStageBufferPool.h/.cpp`, `PPStageBufferSpec`, and `ActivationBuffers` struct
5. Validate PP inference still produces correct output (parity tests with `V2_Integration_Parity_Qwen2_LocalPP`)

**After this phase, there is exactly ONE buffer allocation mechanism (BufferArena) across all inference modes: single-device, LOCAL TP, and PP.**

### Phase 5: Remove Old Single-Device Path

1. Remove `getDumpInfo()` as coherence driver (keep for debugging/dump only)
2. Remove `CoherencePolicy` enum (all stages use contract)
3. Remove `StageCoherence.h` free functions (absorbed into BufferArena)
4. Delete InferenceState's direct buffer ownership (BufferArena is sole owner)
5. Delete `DeviceGraphBufferManager.h/.cpp` (absorbed into BufferArena)
6. Simplify Qwen2Graph to register BufferIds instead of wiring raw pointers

### Phase 6: TP Hardening

1. Per-device BufferArena in RankOrchestrator
2. BAR buffers registered as external buffers with proper device affinity
3. Collective operations use arena-issued BufferViews
4. End-to-end TP parity tests

---

## 7. Testing Strategy

### 7.1 Unit Tests for BufferArena

```cpp
TEST(Test__BufferArena, AllocateAndBorrow) {
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, {4, 896}, TensorType::FP32, DeviceId::cpu());
    ASSERT_TRUE(arena.allocate());
    
    auto view = arena.borrowRead<float>(BufferId::HIDDEN_STATE);
    EXPECT_EQ(view.rows(), 4);
    EXPECT_EQ(view.cols(), 896);
    EXPECT_NE(view.read_ptr(), nullptr);
}

TEST(Test__BufferArena, WriteBlocksOverlappingRead) {
    BufferArena arena;
    arena.registerBuffer(BufferId::Q_PROJ, {4, 896}, TensorType::FP32, DeviceId::cpu());
    arena.allocate();
    
    auto read_view = arena.borrowRead<float>(BufferId::Q_PROJ);
    
    // In debug builds, attempting to write while read borrow is active should assert
    EXPECT_DEATH(arena.borrowWrite<float>(BufferId::Q_PROJ), "active read borrow");
}

TEST(Test__BufferArena, AliasingValidation) {
    BufferArena arena;
    arena.registerBuffer(BufferId::ATTN_OUTPUT, {4, 896}, TensorType::FP32, DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, {4, 4864}, TensorType::FP32, DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();
    
    // Can borrow one at a time
    { auto a = arena.borrowWrite<float>(BufferId::ATTN_OUTPUT); }
    { auto b = arena.borrowWrite<float>(BufferId::GATE_PROJ); }
    
    // Cannot borrow both simultaneously (aliased)
    auto a = arena.borrowWrite<float>(BufferId::ATTN_OUTPUT);
    EXPECT_DEATH(arena.borrowWrite<float>(BufferId::GATE_PROJ), "aliased buffer");
}

TEST(Test__BufferArena, CoherenceTracking) {
    BufferArena arena;
    arena.registerBuffer(BufferId::FFN_OUTPUT, {4, 896}, TensorType::FP32, DeviceId::cpu());
    arena.allocate();
    
    // Write on CPU
    arena.prepareForWrite(BufferId::FFN_OUTPUT, DeviceId::cpu());
    arena.markWritten(BufferId::FFN_OUTPUT, DeviceId::cpu(), nullptr);
    
    auto state = arena.getCoherenceState(BufferId::FFN_OUTPUT);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
    
    // Now prepare for GPU read — should transfer
    // (In unit test, verify the transfer would be requested)
    EXPECT_TRUE(state.needsTransferTo(DeviceId::cuda(0)));
}
```

### 7.2 Integration Tests

- **Single-device parity**: Run existing Qwen2 parity tests with new BufferArena path
- **TP parity**: Run 2-way TP with BufferArena, compare logits vs. single-device
- **PP parity**: Run 2-way PP with PPTransferStage in graph, compare vs. single-device
- **Mixed TP+PP**: Validate heterogeneous setup still produces correct output

### 7.3 Debug-Build Validation

In debug/integration builds, BufferArena performs extra checks:
- **Borrow tracking**: Assert no overlapping writes to aliased buffers
- **Coherence validation**: Warn if buffer is read without being written first
- **Contract matching**: Validate that stage's contract matches actual BufferArena registration
- **Leak detection**: Assert all borrows are released after graph execution

---

## Appendix A: Comparison with Current System

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Buffer ownership** | InferenceState (shared_ptr) | BufferArena (single owner) |
| **Buffer identification** | String names + raw pointers | Typed BufferId enum |
| **Coherence driver** | StageDumpInfo (7 extractX functions) | StageBufferContract (declarative) |
| **Stage access** | Raw ITensor* (anything goes) | BufferView<T, Access> (restricted) |
| **Coherence enforcement** | Honor system (stages can bypass) | Structural (stages can't bypass) |
| **TP buffer routing** | String suffix matching | BufferArena per device |
| **BAR buffers** | 40+ untracked memcpy | External buffer registration |
| **PP transfers** | Outside graph, manual copy | PPTransferStage in graph |
| **Aliasing safety** | LivenessAnalyzer (no runtime check) | AliasingManager + borrow tracking |
| **Debug validation** | TensorVerification (NaN/Inf only) | Borrow conflicts, coherence, contracts |
| **Testability** | Full pipeline required | BufferArena unit-testable in isolation |

## Appendix B: Code Volume Estimate

| Component | Est. Lines | Complexity |
|-----------|-----------|------------|
| BufferId enum | ~50 | Low |
| BufferView template | ~80 | Low |
| StageBufferContract | ~100 | Low |
| StageBoundBuffers | ~120 | Low |
| CoherenceTracker | ~250 | Medium |
| AliasingManager | ~150 | Medium |
| BufferArena | ~500 | Medium |
| GraphExecutor new path | ~200 | Medium |
| Stage migrations (25 stages × ~30 lines each) | ~750 | Medium (repetitive) |
| Unit tests | ~500 | Low |
| **Total** | **~2,700** | |

Most of this is straightforward. The hardest part is not writing new code — it's carefully migrating 25 stages one at a time without breaking parity.

## Appendix C: Gap Resolutions

During Phase 1 implementation, seven gaps were identified and resolved.

### Gap 1: No Dual-Dispatch Migration Path for execute()

**Problem**: The original plan implied stages would eventually receive `StageBoundBuffers` instead of `IDeviceContext*`, but the migration path wasn't specified.

**Resolution**: Keep `execute(IDeviceContext*)` unchanged. The executor reads `bufferContract()` to drive coherence externally — stages never see it at runtime. A new `virtual StageBufferContract bufferContract() const` was added to `IComputeStage` with an empty default, so stages can be migrated incrementally.

### Gap 2: Per-Phase Graph Caching + Dynamic M

**Problem**: Prefill uses M=seq_len while decode uses M=1. How does the arena handle this?

**Resolution**: No code change needed. Both existing allocation paths (`PerStageBufferPool` and `DeviceGraphBufferManager`) already allocate at max capacity upfront. Stages carry effective M in their params. The arena allocates at `max_seq_len`; stages operate on a prefix of the buffer during decode.

### Gap 3: Weight Registration Doesn't Scale (392 Weights)

**Problem**: With 28 layers × 14 weights/layer = 392 weights, putting them all in BufferArena would bloat the `BufferId` enum and waste arena tracking overhead on immutable objects.

**Resolution**: Weights are excluded from the arena. `StageBufferContract::weights` was changed from `vector<BufferBinding>` to `vector<ITensor*> weight_tensors`. Stages register raw pointers to their weight tensors; the executor treats them as external read-only objects. `allReads()` was renamed to `allArenaReads()` to clarify it excludes weights.

### Gap 4: updateDynamicParams / onGraphReplayed Outside Contract

**Problem**: These virtual methods on `IComputeStage` are orthogonal to buffer contracts but weren't addressed.

**Resolution**: Left untouched. These methods serve different purposes (RoPE position updates, graph replay notifications) and don't interact with buffer lifecycle. They remain independent extension points.

### Gap 5: KV Cache Fundamentally Different

**Problem**: KV caches are per-layer, grow over time, and have append semantics. They don't fit the fixed-size arena model.

**Resolution**: Explicitly out of scope. KV caches remain managed by `IKVCache` / `ICPUKVCache` interfaces. The `KVBufferId` struct exists for future integration but is not used by the arena.

### Gap 6: CoherenceTracker Missing Stream Events

**Problem**: GPU graph replay needs lightweight coherence marking without per-tensor event recording overhead (~100-300µs per call on ROCm).

**Resolution**: Three `markWritten` variants now exist:
- `markWritten(state, device)` — state-only update (internal)
- `markWrittenWithEvent(tensor, state, device, stream)` — records GPU event for fine-grained sync
- `markWrittenFlagsOnly(tensor, state, device)` — lightweight flags for graph replay (Phase 3)

`BufferArena::markWritten()` gained an optional `void* stream` parameter and a new `markWrittenFlagsOnly()` method.

### Gap 7: No Auto-Generation of StageDumpInfo from Contract

**Problem**: Stages must implement both `bufferContract()` and `buildDumpInfoImpl()`, duplicating buffer declarations.

**Resolution**: A protected helper `buildDumpInfoFromContract()` was added to `IComputeStage`. It reads the contract's `weight_tensors` and populates `StageDumpInfo::weights` automatically. Stages call it from `buildDumpInfoImpl()` and append inputs/outputs with their dynamic dimensions:
```cpp
StageDumpInfo buildDumpInfoImpl() const override {
    auto info = buildDumpInfoFromContract();
    info.addInput("hidden_state", hidden_state_, M, K);
    info.addOutput("output", output_, M, N);
    return info;
}
```
