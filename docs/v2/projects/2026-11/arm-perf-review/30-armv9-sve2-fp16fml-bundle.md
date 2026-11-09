# Idea 30 — ARMv9 ISA bundle: SVE2 (incl. SVE I8MM/BF16) + FP16/fp16fml

Status: OPEN (proposal 2026-11-09; not started — operator decision).

## Candidate 1: SVE2 (BIG)
- Features (Grace/Neoverse-V2 confirmed via DGX cpuinfo): sve sve2 svei8mm
  i8mm bf16 svebf16 fp16 …; GB10 X925 = ARMv9.2 class (VERIFY .80 cpuinfo).
  M1 = NO SVE → leaf-only (Linux/gcc), runtime HWCAP admission, exactly the
  I8MM envelope+leaf pattern; AppleClang/M1 unaffected.
- Gain hypothesis: 256-bit vectors = 2x NEON width (GEMV/GEMM/grouped rows/
  memory-bound); SVE2 I8MM smmul-class > NEON row-pair SMMLA tile; SVE2 BF16
  for BF16/F16 accum; SVE2 dot 256-bit.
- Cost: new kernel family + envelope decision (current policy: SVE2 not a
  supported envelope — reviewer), compiler/archive identity, register-gate
  evidence. = a real lane.
- Reviewer note (ISA policy answer 2026-11-09): runtime-selected default =
  feasible but different implementation (dispatcher baseline + per-ISA
  variants + isolation + per-variant gates); always-building SVE2 requires a
  capable toolchain but dispatcher could run on older CPUs.

## Candidate 2: FP16 / fp16fml (SMALL)
- FP16 FMA-long (FP16 inputs -> FP32 accumulate): matches the QFN GDN
  alpha/beta (FP16 operands, FP32 accumulation — the actual default policy)
  + FP16 attention/softmax intermediates on ARM. Small envelope add (like
  dotprod); M1/X925/Grace (verify per-core; guard if unsupported).
- NOT BF16 for alpha/beta (less mantissa — README KL-canary chose FP16).

## Candidate 3 (minor): lse/atomics, crc — no decode benefit; skip.
## Not worth: SVE2 on M1 (no HW).

## Decision points (for the operator/plan-first)
1. Order: fp16fml (cheap, first?) vs SVE2 (big, separate lane).
2. Envelope + leaf + HWCAP admission for SVE2 (same as I8MM) vs
   runtime-selected dispatcher (reviewer's alternative).
3. Verify X925 SVE2/BF16/fp16fml support on .80 cpuinfo before scoping.
4. Gate/evidence: extend the register witness + ISA census + byte oracle
   + per-ISA compile identity + ProductionTestPreflight cells.
