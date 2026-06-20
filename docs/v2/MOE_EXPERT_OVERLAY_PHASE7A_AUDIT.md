# MoE Expert Overlay – Bridge Phase 7A Audit

**Phase**: Bridge Phase 7A – Cross-Domain Reduce Back to Continuation Domain  
**Verdict**: ✅ Accepted  
**Date**: 2026-05-10  
**Branch**: `feat/qwen35-moe`

---

## Summary

Phase 7A was chartered to produce a production cross-domain reducer with two explicit
modes (`HostStagedCorrectness` and `ContinuationDeviceOptimized`) and a sparse
partial-row return interface.

Discovery at the start of this phase found that the core reducer
(`MoEExpertParallelReduceStage`) was **already fully implemented** and wired into
`Qwen35MoEGraph` before this phase was dispatched. Both Layout A (ROCm continuation)
and Layout B (CUDA continuation) integration tests passed on real hardware.

The primary deliverable added in Phase 7A was the **sparse partial row return interface**
(Requirement 5: prefer compact routed-row partials for non-continuation tiers).

---

## What Was Already Present (Pre-Phase 7A)

| Item | Status | Location |
|------|--------|----------|
| `MoEExpertParallelReduceMode` enum (`HostStagedCorrectness`, `ContinuationDeviceOptimized`) | ✅ Pre-existing | `MoEExpertParallelReduceStage.h` |
| `MoEExpertParallelReducePartialAccumulationPath` enum | ✅ Pre-existing | `MoEExpertParallelReduceStage.h` |
| `HostStagedCorrectness` reduce path (host-side element-wise sum + D2H/H2D) | ✅ Pre-existing | `MoEExpertParallelReduceStage.cpp` |
| `ContinuationDeviceOptimized` reduce path (H2D transfer + `vectorAddInplace`) | ✅ Pre-existing | `MoEExpertParallelReduceStage.cpp` |
| CPU-continuation fallback (auto-falls through to `HostStagedCorrectness`) | ✅ Pre-existing | `MoEExpertParallelReduceStage.cpp` |
| `MoEExpertParallelReduceDiagnostics` with per-partial accumulation-path labels | ✅ Pre-existing | `MoEExpertParallelReduceStage.h/.cpp` |
| `Qwen35MoEGraph` wiring of reduce stage with mode selection by continuation type | ✅ Pre-existing | `Qwen35MoEGraph.cpp` |
| Layout A (`LayoutAReducesCpuFallbackPartialBackToRocmContinuation`) integration test | ✅ Pre-existing, passing | `Test__MoEExpertOverlay_MultiAcceleratorTiers.cpp` |
| Layout B (`LayoutBReducesRocmAndCpuPartialsBackToCudaContinuation`) integration test | ✅ Pre-existing, passing | `Test__MoEExpertOverlay_MultiAcceleratorTiers.cpp` |
| 11 unit tests covering both modes, zero-partial, shape mismatch, type mismatch | ✅ Pre-existing, passing | `Test__MoEExpertParallelReduceStage.cpp` |

---

## What Was Added in Phase 7A

### 1. Sparse Partial Row Interface (selected_rows)

**Header (`MoEExpertParallelReduceStage.h`)**:
- `MoEExpertParallelReducePartialInfo.selected_rows: std::vector<int>` — identifies
  which full-sequence rows a compact partial covers; empty means dense layout.
- `MoEExpertParallelReducePartialDiagnostics.is_sparse: bool` — true when the partial
  used a sparse layout.
- `MoEExpertParallelReducePartialDiagnostics.sparse_row_count: size_t` — number of
  rows in the compact partial (0 for dense).
- `MoEExpertParallelReduceDiagnostics.sparse_partial_count: size_t` — total count of
  sparse partials processed in the current reduce call.

**Implementation (`MoEExpertParallelReduceStage.cpp`)**:

*HostStagedCorrectness path*:
- When `selected_rows` is non-empty, scatter-adds each compact row into the full
  output buffer at position `full_row = selected_rows[row_idx]`.
- When `selected_rows` is empty, the existing element-wise memcpy/add path is used.
- `bytes` in `PartialRuntime` is computed from `selected_rows.size()` for sparse
  partials (compact layout) vs `live_rows * live_cols` for dense.

*ContinuationDeviceOptimized path*:
- For sparse partials, expands the compact layout into a caller-provided dense
  scratch buffer via scatter-add, then uploads the expanded buffer to the
  continuation device and calls `backend->vectorAddInplace`.
- Accumulation path label for sparse in this mode: `HostStagedThenDeviceAccumulated`
  (host is used for the expand step; accumulation ownership is on the continuation
  device).
- No scratch tensor is allocated inside `execute()`. Optimized sparse reduction
  requires `Params::sparse_expansion_scratch` to be provisioned by graph planning
  or arena ownership before execution.
- Deferred: GPU-native scatter-add kernel (avoids round-trip through host). This
  would eliminate the host expand step; currently the host-staged expand is the
  bridge fallback.

*Validation in `execute()`*:
- When `selected_rows` is non-empty, validates that `partial_rows == selected_rows.size()`
  and `partial_cols == live_cols` (sparse shape check, not dense shape check).
- Follow-up audit remediation also rejects duplicate selected rows and
  out-of-range selected rows before any scatter-add.
- `PartialRuntime.is_sparse` and `.sparse_row_count` populated from
  `info.selected_rows`.

*Diagnostics*:
- `MoEExpertParallelReduceDiagnostics::clear()` now resets `sparse_partial_count = 0`.
- `logDiagnosticsIfRequested()` logs `sparse_partials=N` and per-partial
  `is_sparse`/`sparse_row_count` in transfer trace mode.

### 2. Unit Tests for Sparse Interface

Four new tests added to `Test__MoEExpertParallelReduceStage.cpp`:

| Test | Covers |
|------|--------|
| `SparsePartialScatterAddsSelectedRowsAndLeavesOtherRowsZero` | Sparse partial scatter-add in HostStagedCorrectness; non-selected rows stay zero; diagnostics report is_sparse/sparse_row_count/bytes correctly |
| `MixedSparseAndDensePartialsReduceCorrectly` | One dense + one sparse partial; values accumulate correctly; sparse_partial_count==1, dense partial has is_sparse==false |
| `SparsePartialShapeMismatchFailsClearly` | Partial claims 2 selected rows but tensor has 1 row → execute() returns false |
| `SparsePartialOutOfRangeSelectedRowFailsClearly` | Invalid selected row is rejected before scatter-add |
| `SparsePartialDuplicateSelectedRowFailsClearly` | Duplicate selected rows are rejected before scatter-add |
| `DiagnosticsDistinguishesHostStagedTransportFromHostSummedCorrectness` | Label distinction: HostSummedCorrectnessFallback (correctness bridge) vs HostStagedThenDeviceAccumulated (transport via host, GPU accumulation) |

---

## Parity Test Status

All 6 Qwen3.5 MoE Expert Overlay parity CTests (243–248) **pass at the CTest level**.
The GTest-level SKIP inside them is **not caused by Phase 7A** and was present before
this phase. The skip reason is:

> "Bridge Phase 5E accelerator LocalTP execution blocker: expert overlay tier '...'
> lowers to LocalTP TensorParallelExperts domain '...', but production prepared-weight
> TensorParallelExperts execution is not wired into Qwen3.5 parity"

This skip is checked by `overlayPlanOnlyRuntimeBlockers()` and reflects that the
production GPU LocalTP expert execution path is not wired for parity testing. The
reducer itself is correct and tested by the Layout A/B integration tests on real
hardware.

---

## Deferred Items

| Item | Priority | Notes |
|------|----------|-------|
| GPU-native scatter-add kernel for `ContinuationDeviceOptimized` + sparse | Phase 7B / performance | Eliminates host expand step; sparse partial goes directly to GPU scatter-accumulate |
| Tier stage compact return path (non-continuation tiers emit `[selected_rows.size(), d_model]`) | Phase 7B / tier stage integration | Required for non-continuation tier stages to use sparse layout; reduce stage supports it now. Graph planning must provision dense expansion scratch while the host-expand bridge remains in use. |
| Phase 5E: production LocalTP expert execution wired into Qwen3.5 parity | Phase 5E | Unrelated to Phase 7A; depends on `TensorParallelExperts` production wiring |

---

## Test Gate Results

| Gate | Command | Result |
|------|---------|--------|
| Build | `cmake --build build_v2_integration --parallel` | ✅ 776/776 |
| Unit | `ctest -R "V2_Unit_.*MoEExpertParallelReduceStage"` | ✅ 17/17 (11 pre-existing + 6 new) |
| Integration | `ctest -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers"` | ✅ 8/8 |
| Parity | `ctest -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_"` | ✅ 6/6 CTest pass (GTest SKIP is Phase 5E, not Phase 7A) |
