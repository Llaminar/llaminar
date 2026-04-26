# MoE Expert Weight Transfer — Project Plan

**Branch**: `feat/qwen35-moe`
**Status**: Phases 0–4 complete, Phase 5 deferred until GPU MoE kernels

---

## Problem Statement

When MoE expert rebalancing moves experts between NUMA sockets (MPI ranks), the current
implementation:
1. **Repacks from scratch** — newly-acquired experts run the full VNNI packing pipeline
   from raw mmap'd quantized weights, even though the departing rank already has them packed
2. **Never frees departed weights** — packed weight memory is never reclaimed after an
   expert migrates away (**fixed in Phase 0**)

For Qwen3.5-35B-A3B (256 experts × 3 projections × 40 MoE layers), unnecessary repacking
costs ~800ms per rebalance. Transferring pre-packed weights eliminates this.

---

## Phase 0: Weight Lifecycle Interface ✅

**Status**: Complete — builds, 407 unit tests pass

| Deliverable | File | Status |
|---|---|---|
| `IPackedWeights` abstract interface | `src/v2/kernels/IPackedWeights.h` | ✅ |
| `CPUPackedWeights` concrete (wraps `CPUNativeVNNIPackedWeights`) | `src/v2/kernels/cpu/native_vnni/CPUPackedWeights.h` | ✅ |
| `CPUPackedWeightsWithNativeBlocks` (deferred packing variant) | same file | ✅ |
| `ITensorGemm::detachWeights/attachWeights/releaseWeights/hasWeights` | `src/v2/tensors/TensorKernels.h` | ✅ |
| `CPUNativeVNNIGemmKernel` implementations | `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h` | ✅ |
| Departed expert weight freeing in `MoEFFNStage` | `src/v2/execution/compute_stages/stages/MoEFFNStage.cpp` | ✅ |

---

## Phase 1: Serialization — Wire Format ✅

**Status**: Complete — 408 unit tests pass (13 new serialization tests)

**Goal**: Packed weights can be serialized to a flat byte buffer and reconstructed
identically. Foundation for MPI transfer (Phase 2) and potential persistent caching.

### Wire Format (little-endian)

```
[Header — 64 bytes, fixed]
  magic:              4B  "LPWT" (Llaminar Packed Weight Transfer)
  version:            4B  uint32 = 1
  format:             4B  PackedWeightsFormat enum
  N:                  4B  int32
  K:                  4B  int32
  N_padded:           4B  int32
  blocks_per_row:     4B  int32
  codebook_id:        1B  uint8
  payload_bytes:      1B  uint8
  is_asymmetric:      1B  uint8 (0/1)
  is_nibble_lut:      1B  uint8 (0/1)
  is_superblock:      1B  uint8 (0/1)
  has_native_blocks:  1B  uint8 (0/1, deferred packing flag)
  data_stride:        4B  int32
  interleaved_block_stride: 4B  int32
  native_block_size:  4B  int32 (0 if not deferred)
  reserved:          19B  zero-fill (future use)

[Section table — 4 × uint64 = 32 bytes]
  interleaved_size:   8B  byte count of native_interleaved
  payload_size:       8B  byte count of payload
  int8_flat_size:     8B  byte count of int8_flat
  native_blocks_size: 8B  byte count of native blocks (deferred packing)

[Data sections — contiguous, in table order]
  native_interleaved bytes  (if interleaved_size > 0)
  payload bytes             (if payload_size > 0)
  int8_flat bytes           (if int8_flat_size > 0)
  native_blocks bytes       (if native_blocks_size > 0)
```

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| `PackedWeightsHeader` struct + read/write helpers | `src/v2/kernels/PackedWeightsSerialization.h` | ✅ |
| `CPUPackedWeights::serialize()` override | same file (free functions) | ✅ |
| `IPackedWeights::deserialize()` static factory | same file (free function) | ✅ |
| Unit test: round-trip serialize/deserialize | `tests/v2/unit/kernels/Test__PackedWeightsSerialization.cpp` | ✅ |
| Unit test: corrupt header detection | same file | ✅ |
| Unit test: deferred packing variant round-trip | same file | ✅ |
| CMake registration | `tests/v2/CMakeLists.txt` | ✅ |

### Design Constraints

- `serialize()` returns `std::vector<uint8_t>` — caller owns the buffer
- `deserialize()` is a static method on `IPackedWeights`, dispatches on `format` field
- Header is validated: magic, version, format range, section sizes vs total buffer size
- Deserialized weights must produce bit-identical GEMM output to the original

---

## Phase 2: Cross-Rank MPI Transfer — `ExpertWeightTransfer` ✅

**Status**: Complete — 849 total tests (14 new transfer tests)

**Goal**: Before releasing departed expert weights, send them to the acquiring rank via
MPI. The receiver attaches pre-packed weights instead of repacking from scratch.

### Protocol

1. `rebalanceLPT()` produces `old_placement[e]` → `new_placement[e]` mapping
2. Build migration manifest: for each expert where `old != new`, record
   `{expert_id, layer_id, sender_rank, receiver_rank}`
3. Each rank iterates manifest:
   - **Sender**: `detachWeights()` → `serialize()` → `MPI_Isend` (gate, up, down)
   - **Receiver**: `MPI_Irecv` → `deserialize()` → hold for attach
4. `MPI_Waitall` on all pending transfers
5. `applyExpertMasks()` uses received pre-packed weights where available

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| `ExpertWeightTransfer` class | `src/v2/execution/moe/ExpertWeightTransfer.h` | ✅ |
| MPI tag range for weight transfer | `src/v2/utils/MPITags.h` | ✅ |
| Migration manifest builder | `ExpertWeightTransfer` | ✅ |
| Non-blocking send/recv with `MPI_Waitall` | `ExpertWeightTransfer` | ✅ |
| Integration into `DeviceGraphOrchestrator::applyExpertMasks()` | `DeviceGraphOrchestrator.cpp` | ⬜ Phase 3 |
| Fallback to repack-from-raw on transfer failure | `MoEFFNStage.cpp` | ⬜ Phase 3 |
| Unit test: manifest construction | `tests/v2/unit/moe/Test__ExpertWeightTransfer.cpp` | ✅ |
| Unit test: mock MPI round-trip | same file (serialization round-trip) | ✅ |

### Design Constraints

- Non-blocking: `MPI_Isend`/`MPI_Irecv` + `MPI_Waitall` so transfers overlap
- MPI tag: `MPITags::WEIGHT_TRANSFER_BASE + layer * MAX_EXPERTS * 3 + expert * 3 + proj`
- Fallback: if recv fails or times out, fall back to current repack path (zero regression risk)
- Memory: sender frees serialization buffer after `MPI_Wait`; receiver frees after `attachWeights()`

---

## Phase 3: Attach Path in MoEFFNStage + KernelFactory ✅

**Status**: Complete — 850 total tests (7 new attach tests)

**Goal**: Received pre-packed weights bypass the full repack pipeline and integrate into
the existing `KernelFactory` cache.

### Modified `updateExpertMaskAndPrepareEngines()` Flow

```
For each newly-acquired expert e:
  if received_packed_weights[e] exists:
    1. Create CPUNativeVNNIGemmKernel (empty)
    2. attachWeights(received_packed_weights[e])
    3. Register in KernelFactory cache via new API
    4. Assign engine pointers
  else:
    repack from raw (current path, unchanged)
```

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| `KernelFactory::registerPreparedGemmFromTransfer()` | `src/v2/kernels/KernelFactory.h/.cpp` | ✅ |
| Attach path in `updateExpertMaskAndPrepareEngines()` | `MoEFFNStage.cpp` | ✅ |
| `received_weights` map plumbing (from Phase 2 transfer) | `MoEFFNStage.h` | ✅ |
| Integration test: transferred vs fresh weights parity | `tests/v2/unit/moe/Test__MoEWeightTransferAttach.cpp` | ✅ |

### Design Constraints

- `registerPreparedGemmEngine()` inserts into `prepared_gemm_registry_` with same key scheme
- Must not break existing `getOrCreatePreparedGemmWeights()` flow
- Transferred weights must produce bit-identical decode output

---

## Phase 4: GPU Stubs ✅

**Status**: Complete — 411 total unit tests (18 new stub tests)

**Goal**: Compilable but non-functional GPU packed weight types. Ensures the interface
generalizes beyond CPU before we write GPU MoE kernels.

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| `CUDAPackedWeights` stub | `src/v2/kernels/cuda/gemm/CUDAPackedWeights.h` | ✅ |
| `ROCmPackedWeights` stub | `src/v2/kernels/rocm/gemm/ROCmPackedWeights.h` | ✅ |
| `convertTo()` returns nullptr with LOG_WARN | both stubs | ✅ |
| Unit tests (18 tests) | `tests/v2/unit/kernels/Test__GPUPackedWeightsStubs.cpp` | ✅ |

---

## Phase 5: Cross-Device Conversion + NCCL/RCCL P2P (Deferred)

**Status**: Deferred until GPU MoE kernels are written

| Transfer Path | Mechanism | Notes |
|---|---|---|
| CPU → CUDA | `CPUPackedWeights` → INT8 col-major → `cudaMemcpy` | Reuse `CUDAWeightPacker` logic |
| CUDA → CPU | `cudaMemcpy` → repack to VNNI | Reverse of above |
| GPU → GPU (same vendor) | `ncclSend`/`ncclRecv` | No format change |
| GPU → GPU (cross vendor) | D2H → repack → H2D | Via host staging |

---

## Dependency Graph

```
Phase 0 (DONE) ─── IPackedWeights, detach/attach/release, departed freeing
    │
Phase 1 ────────── Serialization (wire format)
    │
Phase 2 ────────── Cross-rank MPI transfer (ExpertWeightTransfer)
    │
Phase 3 ────────── Attach path in MoEFFNStage + KernelFactory
    │
Phase 4 ────────── GPU stubs (compilable, no-op)  [independent of 2/3]
    │
Phase 5 ────────── Cross-device conversion + NCCL/RCCL P2P  [deferred]
```

Phases 1→2→3 are the critical path for CPU MoE performance.
Phase 4 is independent and can be done in parallel.
Phase 5 depends on GPU MoE kernel support.
