# Idea 21 — CUDA sm_121 (GB10) kernel register spills under the zero-spill guard

Status: **Open** (tracked follow-up for GB10 GPU support; CUDA side, not
ARM-CPU — filed because the operator routes all perf findings here).

## Finding (2026-11-08, Evidence: K14-UMA-SM121-SPILLS-20261108)

At `sm_121` (GB10 GPU), **18 kernel instantiations spill** under the
zero-spill guard (24–276 B per instantiation). Affected families:

- `tensor_prefill_kernel` family,
- `sharedTinyProjectionKernel<bf16>`,
- MoE grouped prefill / overlay activation packet,
- NativeVNNI prefill,
- pooled immutable native.

## Implications

- GB10 GPU support cannot be qualified while spills remain: the
  `GpuSpillGuard.cmake` policy rejects them in Release/Integration, and the
  plan requires complete canonical target set + SM121 admission before
  supporting the Spark GPU.
- This is register-pressure work on the CUDA backend (kernel restructuring,
  launch-geometry/BOM changes), separate from the ARM-CPU items in this
  review.

## Validation

- Compile the complete canonical shipped target set plus SM121 with the guard
  intact; profile the named instantiations; zero spills expected; functional
  SM121 CUDA gates and capture/KV/MTP/state/economy runs after.
- Record the target set with the evidence (spill results are
  architecture-specific).
