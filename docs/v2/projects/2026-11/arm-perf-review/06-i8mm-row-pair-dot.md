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

- Spelling: **`-mcpu=generic -march=armv8-a+simd+dotprod+i8mm`** — canonical
  (FINAL, 2026-11-08, lane evidence; = NEON_DOTPROD's spelling + i8mm, so the
  envelope delta is exactly one feature). Verified on g++-14 /
  llaminar-arm-engine-sdk-py:e0d6d34e7 and AppleClang: `vmmlaq_s32` +
  `vusmmlaq_s32` compile, emit smmla/usmmla @ -O2, and predefine exactly
  `__ARM_FEATURE_DOTPROD` + `__ARM_FEATURE_MATMUL_INT8`.
- `armv8.2-a` REJECTED as base: gcc-14 adds `+lse+dotprod+rdma+crc+i8mm` and
  predefines `__ARM_FEATURE_ATOMICS` / QRDMX / CRC32 — the codegen probe
  deliberately #errors on `__ARM_FEATURE_ATOMICS` and none of those are
  admitted at startup. `armv8.6-a` pulled in BF16 (also rejected).
- `-mcpu=generic` RETAINED (existing canonical policy for all ARM envelopes;
  without it AppleClang keeps its implicit apple-m1 target and can
  auto-vectorize outside the envelope — that is what the envelope proof
  exists to catch). The gcc '-mcpu conflicts with -march' warning is benign
  and identical for today's NEON_DOTPROD build; dropping it for I8MM only
  would make the three envelopes inconsistent. Any reproducibility change
  applies to all three envelopes separately.
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

## Design accepted (2026-11-08) — envelope + leaf, not a new ISALevel

- I8MM is modeled as a **compiled build envelope** + **projection leaf**, NOT
  `ISALevel::NEONI8MM`: active ISA level stays `NEONDotProd` (SMMLA is
  byte-identical to SDOT by contract), so FA2/attention/RMSNorm/SwiGLU/
  qwen4exp/MPI-wire/planning are untouched.
- `LLAMINAR_CPU_ISA=NEON_I8MM` → `LLAMINAR_COMPILED_WITH_NEON_DOTPROD=1` +
  `LLAMINAR_COMPILED_WITH_NEON_I8MM=1`; `compiledNativeVNNIBuildISAName()`
  = `NEON_I8MM` (local/diagnostic only); startup hard-fails without
  HWCAP2_I8MM; `CPUCodegenEnvelope` NEON_I8MM mode (sdot + smmla);
  `V2_CPU_NATIVE_ISA_CELLS` gains `NEON_I8MM`.
- Projection leaf: `ISAPath::NEON_I8MM` (AUTO → I8MM in an I8MM build;
  explicit `NEON_DOTPROD`/`NEON` remain real, tested leaf families → same-
  binary SDOT-vs-SMMLA A/B); rejected outside an I8MM build.
- PerfStats route names: `grouped_neon_i8mm_pair_rows` / `four_rows`; tests
  assert SMMLA actually ran (path proof, maxim 26).
- Sealed identity condition: the 8-field CPU identity, MPI wire codec,
  prepared/archive identities and cross-machine cache keys keep encoding
  NEONDotProd — nothing serializes the I8MM build name into them.

- **Heterogeneous MPI condition (Main, binding):** an i8mm-built rank and a
  dotprod-built rank exchange identical CPU identities — fine only because
  SMMLA is byte-identical. The planner must never place an i8mm binary on a
  host without I8MM (startup hard-fail covers). A heterogeneous MPI job
  (i8mm rank + dotprod rank) must be **admitted with identical bytes or
  rejected** — never silently mixed with different numerics. Add a focused
  regression: CPU-identity equality + byte equality (or clean rejection)
  across i8mm/dotprod-built ranks.

## Perf window results (2026-11-08, Release c3, same-binary SDOT vs SMMLA; interleaved A,B,B,A; 31 iters + 5 warmup; raw+p95 under .80:/home/factory/llaminar-i8mm/build/results/ab-20261108T175836Z)

Kernel speedup (SDOT/SMMLA times, grouped M≥4):
- 17408x5120 (27B gate/up): X925×10 → 1.01–1.09 (Q4_K/Q8_0/IQ3_S/Q3_K 1.07–1.09; IQ4_XS/Q2_K 1.05–1.07; Q6_K 1.04–1.06; IQ2_S 1.03–1.04; Q5_K 1.01–1.03; prefill M=64 1.02–1.09). A725×10 larger: 1.06–1.19 (Q4_K 1.16–1.19, IQ4_XS 1.17–1.19, Q3_K/Q6_K/Q8_0/IQ3_S 1.09–1.13, IQ2_S/Q5_K ~1.09, Q2_K ~1.06; prefill up to 1.19). Mixed 20T: 1.03–1.10 with scheduling noise; some M=2..8 regress (Q3_K M=3 0.77, Q2_K M=4 0.86) — from the harness's CPU-id-ordered team (no capacity binding; idea 26 fixes this in the engine).
- 512x2048 (35B bundle): X925 1.09–1.17 (Q4_K/Q5_K/Q8_0/IQ3_S/IQ4_XS), 1.04–1.06 compact; Q6_K ~1.00 on X925; Q6_K mixed prefill 0.85 but bimodal (min 174 µs = SDOT) → needs a rerun.
- M=1: 0.98–1.02 everywhere (identical SDOT instantiation — guard holds by construction AND measurement). M=2/3 gain little (0.98–1.05).

Codegen (Release libllaminar2_core): SMMLA tiles have NO spill traffic — q stack st/ld Rows=4 I8MM 8/8 vs SDOT 12/12 per encoding family. The earlier Integration numbers were -O3 debug-info noise. Q6_K I8MM has ~37% more static instructions (decode-bound), matching its ~0 gain.

Engine (Release, saved plans, n=256 greedy, A,B,C,C,B,A / A,B,B,A; tokens byte-identical across builds × MTP on/off):
- 27B MTP dyn 5.455→5.545 (+1.7%, acceptance 43.4% both); no-MTP 5.28→5.25 (noise); prefill +2–3%.
- 35B MTP 33.57→34.01 (+1.3%); no-MTP 23.61→23.55 (noise).
- Binding C→B: 35B no-MTP 23.42→23.55 (no regress, ≤+0.5%), MTP 33.94→34.01; prefill noisy.

Handoffs (rebased onto afe62c5ef, fork codex/i8mm-typed-placement):
- a7f719dfb fix(runtime): capacity-ordered binding — BINDING-ONLY, parent afe62c5ef (Main cherry-pick).
- 581765ae5 build(cpu): NEON_I8MM envelope. - cd55f6b27 perf(cpu): I8MM leaf — TIP SHA for the arm-sparks gate re-run (Unit + preflight + NEON_DOTPROD/NEON_I8MM cells).
- Rebase conflict: upstream moved the fused launch recorder to recordNativeNEONFusedLaunches; I8MM ported inside the shared 21-tag schema (isa=NEON_I8MM, grouped_neon_i8mm_* routes); the extra integer_dot tag was dropped.

- Correction 2026-11-08 (Main): cd55f6b27 ALREADY contains a7f719dfb as
  ancestor — PR #43 fast-forwarded onto the binding commit, so the lane's
  rebase is a no-op and cd55f6b27 is current-base + envelope + leaf; the
  authoritative gate runs on cd55f6b27 unchanged.

- Final tip (2026-11-08): rebased onto 0868a5573 → **de4535855** (cc6ec7fd4
  envelope + de4535855 leaf; fork codex/i8mm-typed-placement). CONFLICT
  RESOLUTION: launcher keeps upstream's observer call ONLY; the I8MM
  isa/route tags (NEON_I8MM, grouped_neon_i8mm_*; lone rows NEON_DOTPROD;
  still 21 tags) live in the moved publisher recordNativeNEONFusedLaunches
  (CPUNativeVNNIGemmKernel.h). Self-check .80 Integration NEON_I8MM: 52/52
  focused tests (incl. the 5 filters that were link-broken before, GEMV
  route/isa proof assertions, fused NEON cells, codegen envelope, ISA profile,
  MPIBootstrap/OpenMP binding, CPUStartupAffinity).

## Merge candidate + window-2 results (2026-11-08)

- MERGE CANDIDATE: origin/codex/i8mm-typed-placement @ **5f4c9d4d4**
  (rebased onto 79c1f62ab: envelope d8546d184 + I8MM leaf 5f4c9d4d4; content-
  identical to the gated de4535855 pair; getenv copy dropped — in base as
  12d3f23cd). Placement commit held separately (idea 05). Review requested
  from arm-reviewer on the I8MM content.
- Q6_K mixed-20 prefill (n512 k2048 M=64, 3×ABBA, 61 iters): NOT bimodal —
  stable **−15%** (SDOT 174.8 vs SMMLA 205.5 µs, 6/6). Same shape X925-only
  1.00, A725-only 1.06, big-shape mixed 1.07; Q6_K M=2..4 mixed 1.02–1.07 →
  regression is PREFILL-ONLY and small-N+mixed. Harness uses libgomp CPU-id
  place order (A725 first) vs production capacity binding — rerun under
  production binding before any per-encoding gate.
- Small-M X925-10 (Q4_K/Q8_0/IQ4_XS, 3×ABBA, spreads <2%): M=2 1.00–1.04,
  M=3 1.02–1.06, M=4 1.07–1.09, M=5..8 1.05–1.13, prefill 1.13–1.17,
  M=1 0.98–1.00. M=2/3 low gain is STRUCTURAL (one pair saves 2 dot
  instructions per group pair vs decode work SMMLA doesn't reduce; M=3 adds
  an SDOT tail row) — no defect.
