# Idea 6 — I8MM (SMMLA) row-pair dot-product

Status: **Open**. Not implemented; only feature observation exists
(`ARMCPUFeatures.h`: `IntegerMatMul`, HWCAP2_I8MM). M1's absence is not a
global deferral for GB10, where both core types are Armv9.2 with I8MM.

## Measured motivation (2026-11-07)

On a dense model the grouped verifier is compute-bound in SDOT. On the Mac VM,
Q4_K at the 27B gate/up shape (17408x5120, 10 threads):

- M=1 takes 1.10 ms;
- M=4 takes 2.56 ms (2.3x);
- four serial rows take 6.23 ms.

On GB10, Qwen3.6-27B MTP dynamic (4.23 tok/s) loses to MTP off (4.61 tok/s)
at 52-65% acceptance (idea 14). SMMLA's two rows per instruction targets this
directly. Start with Q4_K and Q8_0 for M >= 2, byte-authenticated against the
SDOT path on GB10.

## Context

- `CPUNativeNEONMultiScaleOps.h:24` names only `ASIMD`/`DotProd` today; the
  prepared stream is 64 output columns × 32-K blocks, four interleaved payload
  bytes per column.
- SMMLA computes a 2-row × 2-column × 8-K byte block per instruction — two
  useful result rows per instruction vs one for SDOT, with the same exact INT32
  sums (no compensation, no saturation).
- The B operand can be formed by a `vzip1q/vzip2q_s32` of two adjacent 4-byte
  group words, so no packing ABI change is required.

## Required before admission (all with real evidence)

1. Runtime capability + compiler envelope admission (HWCAP2_I8MM and
   `__ARM_FEATURE_I8MM`), separate TU/object identity with explicit target.
2. Integer sum bounds: worst-case magnitudes per format (including the
   correction/subtraction terms used by each decoder) must fit INT32 — same
   caution as `NEONAttentionMath::integerDot` (idea 11).
3. Scale subgroup handling for Q6_K dual-scale and the correction/minimum
   metadata paths.
4. Total K-bound coverage where K < 32, odd M (tail rows), odd N.
5. M=1 economy: row-pair SMMLA should not regress single-row decode (SDOT or
   a pair-proof fallback); must be proven, not assumed 2x from MAC counts.
6. All-format byte auth, MTP batch invariance, worker counts 1..20.
7. M1 must remain exact and unchanged (SMMLA never silently dispatched there).

## Exactness / risk

- Medium: the integer sums are identical, but the decode/metadata arithmetic
  around them (per-block scales, minima, Q6 planes) must be proven unchanged for
  every encoding. BFDOT/BFMMLA are explicitly out — their rounding differs from
  the FP32 contract and cannot silently replace the FP32 program.
- SVE2 is lower priority: 128-bit vector length on GB10 gives no width gain
  over NEON; needs evidence before priority.

## Validation

- Perf harness A/B (idea 1): SDOT vs SMMLA for M≥2 grouped/verifier/prefill,
  per encoding, per core class; byte-auth every timed run.

## Implementer constraints (Main/ARM base, 2026-11-08 — binding)

1. INT32 sums identical; the per-block FP program after them unchanged:
   `persistRoundedNEON` barriers, scale/min order, Q6_K dual-scale planes,
   IQ3_S / NibbleLUT metadata checks.
2. Whole output tiles with fixed full-K order; no K repartition; tile
   ownership byte-neutral under `dynamic(1)`.
3. Odd M lone tail rows and N tails go through the M=1 SDOT path —
   byte-identical to serial decode (MTP batch invariance).
4. Runtime HWCAP2_I8MM admission + compiler envelope via
   `__ARM_FEATURE_MATMUL_INT8` (canonical macro, not `__ARM_FEATURE_I8MM`) in
   its own object; M1 never dispatches SMMLA; `CPUCodegenEnvelope` /
   `cpu_isa_profile` proof extends to the new ISA level (no bypass).
5. Regressions in `ProductionTestPreflight`; ISA cells via
   `V2_CPU_NATIVE_ISA_CELLS` (add NEON_I8MM there, not a separate loop).

Base: `b2cf793bf` (upstream PR #43); rebase if the base moves (SHA-only, no
pin branches). Lane: `arm-i8mm` (01a1bbe6, codex/i8mm-typed-placement).

## Envelope resolved (2026-11-08, g++-14 probe on .80 — recheck verdict)

- Spelling: **`-march=armv8.2-a+simd+dotprod+i8mm`** — canonical (verified on
  g++-14 / llaminar-arm-engine-sdk-py:e0d6d34e7: `vmmlaq_s32`+`vusmmlaq_s32`
  compile, rc=0, emits smmla/usmmla @ -O2; AppleClang accepts too).
  `-mcpu=generic` is benign but non-canonical — `CPUInstructionSet.cmake`
  should use ONE authoritative combination (`-march` only).
- NO v8.6 requirement (GCC accepts i8mm as an additive extension with `+simd`
  present). `armv8.6-a` would pull in __ARM_FEATURE_BF16 (+ BF16 arithmetic)
  and __ARM_FEATURE_ATOMICS (LSE) — BF16 is explicitly out of scope, so
  avoid.
- Negative control: same spelling WITHOUT `+i8mm` — `target specific option
  mismatch` for the intrinsic (use in envelope/checker tests).
- Feature macros: `__ARM_FEATURE_MATMUL_INT8` (GCC does not define
  `__ARM_FEATURE_I8MM`) + `__ARM_FEATURE_DOTPROD`; checker sanity also asserts
  `!__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`.
- Startup admission: HWCAP_ASIMDDP + HWCAP2_I8MM.
- Ownership: CPUCodegenEnvelope.py NEON_I8MM + CPUInstructionSet.cmake =
  the i8mm lane; arm-sparks validates at the gate (envelope + BuildSystem
  gates + Unit/byte tests) when handed a SHA.
