# Shared-Memory Spin-Wait Allreduce — Project Plan

## Problem

`MPI_Allreduce` for 8KB (2048 floats) intra-node costs ~129µs per call.
Theoretical UPI wire time for 8KB: ~0.2µs. Overhead is **645×** wire speed.

With 40 allreduce calls per decode token (36 MoE + attention TP layers),
this costs ~5.2 ms/tok — 25% of decode time on dual-socket Xeon.

## Root Cause

OpenMPI's shared-memory BTL has high per-call overhead:
- MPI runtime entry/exit (~5µs)
- Shared-memory BTL protocol negotiation (~10µs)
- Arrival synchronization (spin-wait in MPI polling loop, ~50-80µs avg with EP load imbalance)
- Internal buffer management (~5µs)

## Solution

Replace `MPI_Allreduce` with a purpose-built shared-memory spin-wait allreduce
for 2-rank intra-node CPU TP. Uses POSIX shared memory + atomic flags + AVX-512
SIMD reduction.

**Target: ~3-5µs per call** (30-40× faster than MPI).

## Architecture

### Design Principle

The backend swap happens at `GlobalTPContext` construction time — the only
change is which `ICollectiveBackend` implementation is instantiated. No graph
builder, stage, or orchestrator changes required.

```
BEFORE:
  GlobalTPContext ctor → always creates UPICollectiveBackend → MPI_Allreduce

AFTER:
  GlobalTPContext ctor → if (all_same_node_ && domain_size_ == 2)
                           → ShmemSpinBackend (spin-wait + AVX-512)
                         else
                           → UPICollectiveBackend (MPI, as before)
```

### Files Changed

| File | Change |
|------|--------|
| `src/v2/collective/backends/ShmemSpinBackend.h` | **New** — Header for shared-memory spin-wait backend |
| `src/v2/collective/backends/ShmemSpinBackend.cpp` | **New** — Implementation |
| `src/v2/collective/GlobalTPContext.h` | Change `backend_` type: `unique_ptr<UPICollectiveBackend>` → `unique_ptr<ICollectiveBackend>` |
| `src/v2/collective/GlobalTPContext.cpp` | Constructor: select backend based on topology |
| `src/v2/CMakeLists.txt` | Add `ShmemSpinBackend.cpp` to sources |
| `tests/v2/unit/collective/Test__ShmemSpinBackend.cpp` | **New** — Unit tests |
| `tests/v2/CMakeLists.txt` | Add test target |

### Files NOT Changed

- No graph builders (Qwen2Graph, Qwen35MoEGraph, etc.)
- No stages (FusedAddAllreduceStage, TPAllreduceStage, etc.)
- No orchestrators (DeviceGraphOrchestrator, RankOrchestrator)
- No ICollectiveBackend interface
- No ITPContext interface

## Shared Memory Layout

```
POSIX shm: /llaminar_shmem_ar_<domain_id>

┌───────────────────────────────────────────────────────┐
│ Header (128 bytes, 64B aligned)                       │
│   epoch[0]: atomic<uint64_t>  — rank 0 ready counter  │
│   [padding to 64B]                                     │
│   epoch[1]: atomic<uint64_t>  — rank 1 ready counter  │
│   [padding to 64B]                                     │
├───────────────────────────────────────────────────────┤
│ buffer[0]: float[MAX_COUNT] (64B aligned)             │
│   — rank 0 stages its data here                        │
├───────────────────────────────────────────────────────┤
│ buffer[1]: float[MAX_COUNT] (64B aligned)             │
│   — rank 1 stages its data here                        │
└───────────────────────────────────────────────────────┘
```

- `MAX_COUNT` = 8192 elements (32KB per rank buffer, covers d_model up to 8192)
- Total shared memory: ~64KB + 128B header
- Epoch counters on separate cache lines to avoid false sharing

## Allreduce Protocol

```
allreduce(buf, count):
  my_epoch++;

  // 1. Stage my data into shared memory
  memcpy(shmem->buffer[my_rank], buf, count * sizeof(float));

  // 2. Signal ready (store-release)
  atomic_store(&shmem->epoch[my_rank], my_epoch, memory_order_release);

  // 3. Wait for peer (load-acquire + pause spin)
  while (atomic_load(&shmem->epoch[peer_rank], memory_order_acquire) < my_epoch)
      _mm_pause();

  // 4. AVX-512 reduce both buffers into caller's buffer
  avx512_add(buf, shmem->buffer[0], shmem->buffer[1], count);
```

Both ranks execute the same protocol. Both end up with the identical reduced result
in their own `buf`.

## Fallback Strategy

`ShmemSpinBackend` delegates unsupported operations to a wrapped `UPICollectiveBackend`:

| Operation | ShmemSpinBackend | Fallback |
|-----------|-----------------|----------|
| `allreduce` (FLOAT32, SUM, count ≤ MAX) | Fast path ✓ | — |
| `allreduce` (other dtype/op/overflow) | — | UPI/MPI |
| `allgather` | — | UPI/MPI |
| `allgatherv` | — | UPI/MPI |
| `reduceScatter` | — | UPI/MPI |
| `broadcast` | — | UPI/MPI |

## Expected Performance

| Metric | MPI (current) | Shmem spin (target) | Savings |
|--------|--------------|---------------------|---------|
| Per-call latency | ~129µs | ~3-5µs | ~125µs |
| 40 calls/token | 5.2 ms | ~0.2 ms | ~5.0 ms |
| Decode throughput | 20.6 tok/s | ~25+ tok/s | +22% |

## Test Plan

1. **Single-process unit tests** (MPI_PROCS 1):
   - Shared memory allocation/cleanup lifecycle
   - AVX-512 reduce correctness (known inputs → expected output)
   - Fallback delegation for unsupported ops

2. **Two-process integration tests** (MPI_PROCS 2):
   - Basic allreduce correctness (2048 floats)
   - Repeated allreduce (epoch counter correctness)
   - Large count (MAX_COUNT boundary)
   - Mixed operations (allreduce fast path + allgather fallback)
   - Stress test (1000 consecutive allreduces)
