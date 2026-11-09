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

## Ownership/handover (2026-11-08 — spark-cuda-spil standing down)

- The CUDA spill-fix work is now owned by **Main/ARM** (per operator): he
  folds the sound parts into `feat/arm64-native-kernels` and closes PR #44.
- Canonical trace build SHA: `09e14d3ae` remains **provenance only** for the
  measured numbers; the canonical SHA for future traces = the migrated
  ARM-branch SHA, announced by Main when landed.
- **Tensor-split spec rev 2 (`83c61732d`)**: open design candidate, pending
  sign-off from arm-perf + arm-reviewer (01a1a8d9).

- Tensor-split spec rev 2 (83c61732d): **signed off by arm-perf (2026-11-08)**
  with conditions — §5 exactness proven by the mandated byte-compare +
  boundary sweep (not argument alone); fragment/lane mapping verified against
  mmaFragmentRow/Column before implementation; measured per-target values
  replace every estimate; consumer peak > 170 → stop+report (no silent 4-way
  fallback); add unprofiled before/after Release timing (one barrier per tile
  — report any slowdown with attribution); graph-capture identity includes
  the variant/generation. Implementation + measurements owned by Main/ARM
  after the branch move; arm-reviewer checks the implementation diff.

## Moved into PR #43 (2026-11-08) — most spills fixed

PR #43 = **7e709d60e** (ff from 0868a5573) now contains the CUDA commits from
closed PR #44 (branch deleted):
- a62696317 fix(cuda): sm_120/121 register budgets — SASS identical on the
  shipped archs; 12.x spills before→after: single-row dispatch 276→0, paired
  gate/up 44→0, owner phase 44→0, tiny 4→0, BK64 76→0 (inlining side effect,
  source unchanged), **BK256 328→8**.
- 7e709d60e docs: CUDA_SPILL_TENSOR_SPLIT_SPEC.md rev 2 (with the arm-perf
  conditions).

STILL OPEN on 12.x: BK256 non-ordered variants at 4–8 bytes (per-K A loads
reverted — didn't help), and `tensor_prefill` (the signed-off split spec).
Evidence: `~/llaminar-spark-results/K14-CUDA-SPILL-MIGRATION-a62696317`.

## REOPENED/INCOMPLETE (2026-11-09, arm-sparks) — fold partial for sm_121

- Building Release CUDA arch 121 at 7e709d60e SPILLS: ptxas 'Registers are
  spilled' in CUDANativeVNNIPrefillKernels.cu nativeVnniTC_BK64<...>
  (h0/h4/h24, 256-wide staging) + CUDAFlashAttentionKernels.cu (code=255,
  build stops).
- Evidence: 7e709d60e's CUDANativeVNNIPrefillKernels.cu differs from the
  spill-verified bd9dcd12a (17+/4-; TinyProjectionSharedKernel.cuh 24+/4-) —
  the fold took a DIFFERENT/PARTIAL version; bd9dcd12a's BK64/BK256
  unroll-roll fix is NOT fully present upstream.
- RECOMMENDED: fold bd9dcd12a's kernel content (or remaining rolls) into #43
  (that version probes 0 spills at sm_120+121). Current arm branch
  (302318f1e) contains 7e709d60e → Release CUDA sm_121 builds fail there
  until fixed. Idea-25 trace substitution: cff351901 (523230ab0 + bd9dcd12a;
  U2✓ U5✓) sealed.
