# V2 Snapshot Framework - Architecture Diagrams

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER CODE (Pipeline)                        │
│                                                                     │
│  bool Qwen2Pipeline::forward_batch(...) {                         │
│      embedding_batch(...);                                         │
│      LLAMINAR_SNAPSHOT(mgr_, GLOBAL_EMBEDDING, -1, 0, tensor);    │
│      // ↑ Macro call                                              │
│  }                                                                 │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     │ #ifndef NDEBUG (debug builds only)
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLAMINAR_SNAPSHOT Macro                          │
│  Expands to:                                                        │
│    if (mgr_ && isStageEnabled(stage)) {                            │
│        auto snap = tensor->captureSnapshot();  // ← TensorBase     │
│        snap.stage_name = formatStageName(...);                     │
│        mgr_->addSnapshot(snap);                                    │
│    }                                                                │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ├──────────────┬──────────────┐
                     ▼              ▼              ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │ TensorBase:: │  │ isStage      │  │ formatStage  │
         │ snapshot()   │  │ Enabled()    │  │ Name()       │
         └──────┬───────┘  └──────────────┘  └──────────────┘
                │
                │ Device→CPU sync, memcpy
                ▼
         ┌──────────────────────┐
         │  SnapshotData        │
         │  ----------------    │
         │  • shape             │
         │  • data (FP32)       │
         │  • native_type       │
         │  • device_index      │
         │  • stage_name        │
         │  • layer/token/seq   │
         └──────┬───────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│            PipelineSnapshotManager (per-pipeline)                   │
│                                                                     │
│  • Thread-safe storage: map<stage_name, SnapshotData>             │
│  • addSnapshot(snap)                                               │
│  • saveAll(dir) → NPY files                                        │
│  • loadAll(dir) ← NPY files                                        │
│  • getSnapshot(name) → query                                       │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     │ Test-time only
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SnapshotComparator                              │
│                                                                     │
│  • compare(llaminar, pytorch, threshold)                           │
│  • computeVarianceThreshold(pytorch_runs[])                        │
│  • loadPyTorchSnapshots(dir)                                       │
│                                                                     │
│  Returns: ComparisonMetrics {                                      │
│      max_abs_diff, rel_l2, passed, message                         │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Snapshot Capture

```
┌──────────────┐
│  Pipeline    │
│  forward()   │
└──────┬───────┘
       │
       │ 1. Compute tensor
       │    (GEMM, attention, etc.)
       ▼
┌──────────────────┐
│  TensorBase      │
│  (FP32/BF16/IQ4) │
│                  │
│  Device: GPU     │  ← May be on GPU
└──────┬───────────┘
       │
       │ 2. LLAMINAR_SNAPSHOT(mgr, stage, layer, token, tensor)
       ▼
┌────────────────────────────────────┐
│  Macro expands (debug only):       │
│                                    │
│  if (isStageEnabled(stage)) {      │
│      snap = tensor->snapshot()     │ ← 3. Calls TensorBase::snapshot()
│      mgr->addSnapshot(snap)        │
│  }                                 │
└────────────────┬───────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ TensorBase::         │
      │ captureSnapshot()    │
      │                      │
      │ 1. Sync GPU→CPU      │ ← 4. Auto-sync if on device
      │    (if needed)       │
      │ 2. memcpy data       │
      │ 3. Return snap       │
      └──────┬───────────────┘
             │
             ▼
      ┌──────────────────────┐
      │  SnapshotData {      │
      │    shape: [1024,896] │
      │    data: [...]       │
      │    type: IQ4_NL      │
      │    device: 0         │
      │  }                   │
      └──────┬───────────────┘
             │
             │ 5. Store in manager
             ▼
      ┌──────────────────────┐
      │ SnapshotManager      │
      │                      │
      │ snapshots_[          │
      │   "token_0/          │
      │    ATTN_NORM_layer3" │
      │ ] = snap             │
      └──────────────────────┘
```

---

## Data Flow: Parity Testing

```
┌──────────────────────────────────────────────────────────────┐
│                    PYTHON: Generate Reference                │
│                                                              │
│  1. Run PyTorch 3× with same inputs                         │
│  2. Capture snapshots at each stage                         │
│  3. Compute variance across runs                            │
│  4. Generate thresholds:                                    │
│     threshold = max(variance, mag×1.5%) × 5.0               │
│  5. Save:                                                    │
│     - pytorch_snapshots/*.npy                               │
│     - variance_thresholds.json                              │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   │ NPY files
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                 C++ TEST: Run Llaminar                       │
│                                                              │
│  1. Load PyTorch snapshots                                  │
│  2. Load variance thresholds                                │
│  3. Run Llaminar with snapshot capture                      │
│  4. Compare each stage:                                     │
│                                                              │
│     for each stage:                                         │
│         llaminar_snap = mgr->getSnapshot(name)              │
│         pytorch_snap = loadPyTorch(name)                    │
│         threshold = thresholds[name]                        │
│         metrics = compare(llaminar, pytorch, threshold)     │
│                                                              │
│         if (!metrics.passed):                               │
│             FAIL("Stage X diverged")                        │
│                                                              │
│  5. Report: 245 passed, 0 failed                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Memory Layout: SnapshotData

```
SnapshotData structure (typical size: ~10-50 MB depending on tensor)
┌─────────────────────────────────────────────────────────┐
│ Header                                                  │
├─────────────────────────────────────────────────────────┤
│ shape: vector<size_t>         │  [1024, 896]          │ ← 24 bytes (3×8)
│ native_type: TensorType       │  IQ4_NL               │ ← 4 bytes
│ device_index: int             │  0                    │ ← 4 bytes
│ mpi_rank: int                 │  0                    │ ← 4 bytes
│ layer_index: int              │  5                    │ ← 4 bytes
│ token_index: int              │  0                    │ ← 4 bytes
│ sequence_index: int           │  -1                   │ ← 4 bytes
│ stage_name: string            │  "token_0/..."        │ ← ~64 bytes
├─────────────────────────────────────────────────────────┤
│ Data (FP32)                                             │
├─────────────────────────────────────────────────────────┤
│ data: vector<float>           │  1024×896 = 917,504   │ ← 3.5 MB
│                               │  elements             │
│                               │  × 4 bytes = 3.5 MB   │
└─────────────────────────────────────────────────────────┘
Total: ~3.5 MB per snapshot (for [1024, 896] tensor)

Full capture (100 stages): ~350 MB
```

---

## Stage Hierarchy

```
PipelineStage enum (hierarchical organization)

GLOBAL_*  (layer = -1)
│
├── GLOBAL_EMBEDDING        # Token embedding table
├── GLOBAL_FINAL_NORM       # Final RMSNorm
└── GLOBAL_LM_HEAD          # Logits over vocabulary

LAYER_*  (layer = 0..N-1)
│
├── LAYER_INPUT             # Layer input
│
├── Attention Block
│   ├── LAYER_ATTN_NORM     # Pre-attention RMSNorm
│   ├── LAYER_ATTN_OUTPUT   # After output projection
│   └── LAYER_ATTN_RESIDUAL # After residual add
│
└── FFN Block
    ├── LAYER_FFN_NORM      # Pre-FFN RMSNorm
    ├── LAYER_FFN_OUTPUT    # After down projection
    └── LAYER_FFN_RESIDUAL  # After residual add

ATTN_*  (debug substages)
│
├── ATTN_Q_PROJECTION       # Q = x @ W_q
├── ATTN_K_PROJECTION       # K = x @ W_k
├── ATTN_V_PROJECTION       # V = x @ W_v
├── ATTN_ROPE_APPLIED       # Q, K after RoPE
├── ATTN_SCORES             # Q @ K^T
├── ATTN_WEIGHTS            # softmax(scores)
└── ATTN_CONTEXT            # weights @ V

FFN_*  (debug substages)
│
├── FFN_GATE                # gate = x @ W_gate
├── FFN_UP                  # up = x @ W_up
├── FFN_SWIGLU              # silu(gate) * up
└── FFN_DOWN                # out @ W_down (same as LAYER_FFN_OUTPUT)
```

---

## File Organization

```
src/v2/
├── utils/
│   ├── SnapshotCapture.h          # Macros, stage enum, helpers
│   ├── PipelineSnapshotManager.h  # Storage interface
│   ├── PipelineSnapshotManager.cpp
│   ├── SnapshotComparator.h       # Test utility
│   └── SnapshotComparator.cpp
│
├── tensors/
│   └── Tensors.h                  # TensorBase::captureSnapshot()
│
└── pipelines/
    └── qwen/
        ├── Qwen2Pipeline.h        # snapshot_mgr_ member
        └── Qwen2Pipeline.cpp      # LLAMINAR_SNAPSHOT calls

tests/v2/
└── parity/
    ├── Test__Qwen2_Parity.cpp     # Qwen2 parity tests
    ├── Test__LLaMA_Parity.cpp     # LLaMA parity tests (future)
    └── ParityTestUtils.h          # Shared utilities

python/reference/
├── v2_snapshot_adapter.py         # V2→V1 adapter
├── generate_v2_snapshots.py       # PyTorch snapshot generation
└── generate_v2_variance_thresholds.py  # Threshold computation

docs/v2/
├── SNAPSHOT_FRAMEWORK_DESIGN.md   # Full design spec
├── SNAPSHOT_QUICK_START.md        # Quick start guide
└── SNAPSHOT_ARCHITECTURE.md       # This file
```

---

## Macro Expansion Example

### Debug Build (CMAKE_BUILD_TYPE=Debug)

```cpp
// Source code:
LLAMINAR_SNAPSHOT(snapshot_mgr_, PipelineStage::LAYER_ATTN_NORM, 
                 layer_idx, token_idx, normalized_tensor.get());

// Preprocessor expands to:
do {
    if (snapshot_mgr_ && llaminar2::snapshot::isStageEnabled(PipelineStage::LAYER_ATTN_NORM)) {
        auto snap_data = normalized_tensor.get()->captureSnapshot();
        snap_data.stage_name = llaminar2::snapshot::formatStageName(
            PipelineStage::LAYER_ATTN_NORM, layer_idx, token_idx, -1);
        snap_data.layer_index = layer_idx;
        snap_data.token_index = token_idx;
        snap_data.sequence_index = -1;
        snapshot_mgr_->addSnapshot(snap_data);
    }
} while(0);

// Compiles to (assembly):
call    isStageEnabled
test    %al, %al
je      .skip_snapshot
call    TensorBase::captureSnapshot
call    formatStageName
call    PipelineSnapshotManager::addSnapshot
.skip_snapshot:
```

### Release Build (CMAKE_BUILD_TYPE=Release)

```cpp
// Source code (same):
LLAMINAR_SNAPSHOT(snapshot_mgr_, PipelineStage::LAYER_ATTN_NORM, 
                 layer_idx, token_idx, normalized_tensor.get());

// Preprocessor expands to:
((void)0)

// Compiles to (assembly):
# (nothing - dead code eliminated)
```

**Result**: Zero overhead in production builds!

---

## Thread Safety

```
Multiple test threads calling pipeline->forward()
│
├── Thread 1                      ├── Thread 2
│   │                            │   │
│   │ Pipeline 1                 │   │ Pipeline 2
│   │ snapshot_mgr_1             │   │ snapshot_mgr_2
│   │                            │   │
│   ├─ LLAMINAR_SNAPSHOT         │   ├─ LLAMINAR_SNAPSHOT
│   │  ├─ captureSnapshot()      │   │  ├─ captureSnapshot()
│   │  │  (thread-safe)          │   │  │  (thread-safe)
│   │  └─ mgr_1->addSnapshot()   │   │  └─ mgr_2->addSnapshot()
│   │     ├─ mutex_.lock()       │   │     ├─ mutex_.lock()
│   │     ├─ snapshots_[...] =   │   │     ├─ snapshots_[...] =
│   │     └─ mutex_.unlock()     │   │     └─ mutex_.unlock()
│   │                            │   │
│   └─ No contention!            │   └─ No contention!
    (separate managers)              (separate managers)

Key: Each pipeline has its OWN snapshot manager → no shared state
```

---

## NPY File Format

```
Snapshot saved as: pytorch_snapshots/token_0/ATTN_NORM_layer5.npy

NPY file structure (NumPy binary format):
┌─────────────────────────────────────────────────────────┐
│ Magic: "\x93NUMPY"                                      │ ← 6 bytes
│ Version: 0x01 0x00                                      │ ← 2 bytes
│ Header length: 0x0076                                   │ ← 2 bytes
├─────────────────────────────────────────────────────────┤
│ Header (dict):                                          │
│ {                                                       │
│   'descr': '<f4',        # Little-endian float32       │
│   'fortran_order': False,  # C-order (row-major)       │
│   'shape': (1024, 896),    # Tensor dimensions         │
│ }                                                       │
├─────────────────────────────────────────────────────────┤
│ Data: float32 array [1024, 896]                        │
│                                                         │
│  [0.123, 0.456, ...]  ← 1024×896 = 917,504 floats     │
│                       ← 3.5 MB                         │
└─────────────────────────────────────────────────────────┘

Compatible with:
- NumPy: np.load("snapshot.npy")
- C++: NpyArray::load("snapshot.npy")
- Python: numpy.load() / PyTorch: torch.from_numpy()
```

---

## Comparison Metrics Computation

```python
# SnapshotComparator::compare()

llaminar_data = [0.1, 0.2, 0.3, ...]  # Shape: [1024, 896]
pytorch_data = [0.1, 0.2, 0.3001, ...]

# 1. Element-wise absolute difference
abs_diff = |llaminar_data - pytorch_data|
# → [0.0, 0.0, 0.0001, ...]

# 2. Max absolute difference
max_abs_diff = max(abs_diff)
# → 0.0234

# 3. Mean absolute difference
mean_abs_diff = mean(abs_diff)
# → 0.0005

# 4. Relative L2 norm
l2_diff = sqrt(sum((llaminar - pytorch)^2))
l2_ref = sqrt(sum(pytorch^2))
rel_l2 = l2_diff / l2_ref
# → 0.0182

# 5. Num mismatches (exceeding tolerance)
num_mismatches = count(abs_diff > threshold)
# → 127 elements

# 6. Pass/fail decision
passed = (max_abs_diff <= threshold) && (rel_l2 <= threshold)
# → False (failed)

ComparisonMetrics {
    max_abs_diff: 0.0234,
    mean_abs_diff: 0.0005,
    rel_l2_norm: 0.0182,
    num_mismatches: 127,
    passed: false,
    message: "Stage failed: max_abs (0.0234) > threshold (0.0054)"
}
```

---

## Variance Threshold Computation

```python
# SnapshotComparator::computeVarianceThreshold()

# Run PyTorch 3 times with same inputs
run_0 = generate_pytorch_snapshots(tokens)  # [1024, 896]
run_1 = generate_pytorch_snapshots(tokens)  # [1024, 896]
run_2 = generate_pytorch_snapshots(tokens)  # [1024, 896]

# Stack runs: [3, 1024, 896]
stacked = np.stack([run_0, run_1, run_2], axis=0)

# Compute variance across runs (axis=0)
variance = np.var(stacked, axis=0).mean()
# → 0.000123  (very small - PyTorch is deterministic)

# Compute magnitude (mean abs of first run)
magnitude = np.abs(run_0).mean()
# → 0.456

# Compute threshold
MIN_THRESHOLD = 1e-5
MAGNITUDE_FACTOR = 0.015  # 1.5%
VARIANCE_MULTIPLIER = 5.0

threshold = max(variance, magnitude * MAGNITUDE_FACTOR) * VARIANCE_MULTIPLIER
threshold = max(threshold, MIN_THRESHOLD)
# → max(0.000123, 0.456 * 0.015) * 5.0
# → max(0.000123, 0.00684) * 5.0
# → 0.00684 * 5.0
# → 0.0342

VarianceThreshold {
    variance: 0.000123,
    magnitude: 0.456,
    threshold: 0.0342  ← Used for comparison
}
```

**Result**: Threshold automatically scales with tensor magnitude!

---

## Summary

The V2 snapshot framework provides:
- **Zero-overhead** parity testing (macros compile out in release)
- **Device-transparent** capture (auto GPU→CPU sync)
- **Hierarchical staging** (GLOBAL/LAYER/ATTN/FFN)
- **Thread-safe** per-pipeline storage
- **Automatic thresholds** from PyTorch variance

See `SNAPSHOT_FRAMEWORK_DESIGN.md` for complete specification.
