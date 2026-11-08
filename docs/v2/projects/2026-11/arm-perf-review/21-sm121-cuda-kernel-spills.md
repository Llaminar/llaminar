# Idea 21 — CUDA GB10 (sm_120/sm_121) kernel register spills under the zero-spill guard

Status: **Addressed** (2026-11-08 — the spill session fixed the CUDA
register-pressure issue; fix is the stacked upstream PR `Llaminar/llaminar`
**#44** (draft, base `feat/arm64-native-kernels`), linear tip **`09e14d3ae`**,
whose content equals the local merged tree `a5412af30`; spill kernel files
verified unchanged vs `be998fdec`). Remaining: upstream merge + live SM120/
SM121 and joint-pool validation once the guard is clean.

## Finding (2026-11-08; evidence: K14-UMA-SM121-SPILLS-20261108)

Tables (mangled + demangled, store/load bytes):
`~/llaminar-spark-results/K14-UMA-SM121-SPILLS-20261108/{sm121,sm120}-spill-table.json`
(18 + 45 rows).

- **sm_121 (GB10):** 18 spilling instantiations (24–276 B).
- **sm_120 is worse:** 45 spilling instantiations (SASS-compatible target).

Canonical kernel classes (from the builder evidence):

| Class | Instance | Spill bytes |
|---|---|---|
| `CUDATinyFP32ProjectionKernels` | `sharedTinyProjectionKernel<bf16,8,float>` | 4 |
| `CUDATensorPrefillKernels` | `tensor_prefill_kernel<>` | 60–88 |
| `CUDAMoEGroupedPrefillKernels` | `groupImmaPairedGateUpSwiGluKernel<13,128>` / `<13,64>` | 44 / 24 |
| `CUDAMoEOverlayActivationPacketKernels` | | 128 / 156 |
| `CUDANativeVNNIPrefillKernels` | | |
| `CUDAPooledImmutableNativeKernels` | | |

Build environment: nvcc 13.0.88, aarch64, CUTLASS v4.2.1, NCCL capture-reentry
2.28.9 — all builder probes pass; the build lane is **closed as blocked by
register pressure**. Fix = CUDA kernel register tuning for sm_121 (60–88 B
tensor-prefill instances being the largest single family).

## Implications

- GB10 GPU support **and the live UMA joint-pool validation (idea 2)** are
  **blocked** on these register-pressure fixes: the `GpuSpillGuard.cmake`
  policy rejects spills in Release/Integration, and the plan requires the
  complete canonical target set plus SM120/SM121 admission before supporting
  the Spark GPU.
- Register-pressure work on the CUDA backend (kernel restructuring,
  launch-geometry/BOM changes), separate from the ARM-CPU items in this
  review.

## Validation

- Compile the complete canonical shipped target set plus SM121 with the guard
  intact; profile the named instantiations; zero spills expected; functional
  SM121 CUDA gates and capture/KV/MTP/state/economy runs after.
- Record the target set with the evidence (spill results are
  architecture-specific).
