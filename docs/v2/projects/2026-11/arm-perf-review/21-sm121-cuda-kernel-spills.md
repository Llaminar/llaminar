# Idea 21 — CUDA GB10 (sm_120/sm_121) kernel register spills under the zero-spill guard

Status: **Open** (tracked follow-up for GB10 GPU support; CUDA side, not
ARM-CPU — filed because the operator routes all perf findings here).

## Finding (2026-11-08, evidence: K14-UMA-SM121-SPILLS-20261108)

Tables: `~/llaminar-spark-results/K14-UMA-SM121-SPILLS-20261108/
{sm121,sm120}-spill-table*.json`.

- **sm_121 (GB10):** 18 kernel instantiations spill under the zero-spill guard
  (24–276 B). Families: `tensor_prefill_kernel` family,
  `sharedTinyProjectionKernel<bf16>`, MoE grouped prefill / overlay activation
  packet, NativeVNNI prefill, pooled immutable native.
- **sm_120 is worse:** 45 spilling instantiations, adding
  `groupImmaPairedGateUpSwiGluKernel<13,128>` / `<13,64>` (44/24 B) and
  `MoEOverlayActivationPacket` (128/156 B) to the sm_121 set.

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
