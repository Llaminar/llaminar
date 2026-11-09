# Idea 30 — ARMv9 ISA bundle: SVE2 (incl. SVE I8MM/BF16) + FP16/fp16fml

Status: **DEPRIORITIZED** (2026-11-09) — evidence verdict below; not a lane now.

TARGET = SPARKS (operator directive 2026-11-09); Grace-specific ignored.

VERDICT: X925/GB10 VL probe (arm-sparks, .80): default=16, GET_VL=16,
  svcntb=16, SET_VL(max|ONEXEC) = **-1 REJECTED** (X925 cannot raise VL;
  DGX Grace = capped at 16 with SET returning 16) -> **SVE = 128-bit ONLY on
  BOTH targets.** The 2x-width hypothesis is DEAD. HWCAP advertises
  SVE/SVE2/SVEI8MM/SVEBF16/BF16/I8MM at VL=128 only. SVE2 residual value =
  128-bit instruction pool (dot/bf16/fp16-widen = small vs NEON: NEON
  vfmlalq_low_f16 is already default under armv9-a; SVE2 I8MM smmul at 128
  = same-class as NEON SMMLA). => SVE2 deprioritized; only a 128-bit
  ILP/SIMD-density question remains (unmeasured; low prio).

FP16fml: ALREADY AVAILABLE (NEON vfmlalq_low_f16 default at armv9-a;
  SVE2 svmlalb_f32 without +fp16fml) -> no envelope work; the actionable
  small item = verify the engine's FP16 GDN alpha/beta kernels actually
  EMIT fmlalq on X925 (objdump check), else a kernel tuning item.

QUEUE NOW: idea 31 (S=4 spillfixes) = the Sparks-directed ISA follow-up.

QUEUE ENTRY: this bundle is queued as a possible task for a lane that
finishes its current work. Natural owner: the addon lane (01a1bbe6) after
R2 completes (it owns the ISA envelope+leaf+register-witness machinery:
i8mm, panels, audit) — trigger = R2 pushed + reviewer + WSL green. If the
operator prefers, SVE2 becomes its own new lane (plan-first per the
standard). First steps on start: verify X925 SVE2/BF16/fp16fml on .80
cpuinfo; order fp16fml (cheap) vs SVE2 (big); envelope+leaf+HWCAP admission
decision; plan-first doc.

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

## DGX probe (2026-11-09) — VL = 128 CAPPED on Grace + GDN native ADMITS

- COMPILER: GCC13 + devcontainer GCC14.2 accept ALL requested -march
  spellings; census identical. Intrinsics: svdot_s32/u32 w/ SVE2; svmmla_s32
  needs +i8mm; svbfdot/svbfmmla/svbfmlalb need +bf16; SVE2 svmlalb_f32 (FP16
  widening) WITHOUT +fp16fml; NEON vfmlalq_low_f16 default under armv9-a.
- VL: Grace REALLY capped at 128-bit on this OS/device: GET_VL=16,
  svcntw=4; SET_VL 32/64/256/8192 + aligned max 65520 all return 16;
  unaligned 65535 EINVAL. HWCAP: SVE,SVE2,SVEI8MM,SVEBF16,I8MM,BF16 all 1.
  -> NO honest SVE256 timings; the 2x-width hypothesis DOES NOT HOLD on
  Grace. Verify X925/GB10 VL (arm-sparks) before any SVE2 lane.
- GDN ANSWER (user question): **quantized GDN source is ADMITTED under
  LLAMINAR_GDN_PROJECTION_ARITHMETIC=native** (plan: 'Candidates: 1 admitted
  (0 capacity-rejected)', Strategy: single, plan written) — the README
  rejection (half-operand policies) is scoped; 'native' runs quantized GDN.
  Full log retained by the DGX addon (scratch-sve2/*).
- PMU inaccessible (perf_event_paranoid=4) -> estimated cycles only.

## DGX addon FINAL (2026-11-09) — full census + sweep + X925 plan

- COMPILERS: GCC13 + GCC14.2 identical census; every spelling compiles
  (table incl. minimal `armv8-a+simd+dotprod+sve2` + optional +i8mm/bf16/
  fp16fml). `armv9-a+...` expands the WHOLE baseline (LSE etc.) — NOT a safe
  cosmetic rename; prefer minimal leaf flags. GCC14: -mcpu=generic conflicts
  with richer -march (do not change envelope to mask).
- SWEEP (Grace, VL=128, estimated cycles, cache-hot synthetic): SVE128 vs
  NEON = LOSES at small K (0.85-0.99x), near-parity at K512 (sdot 1.02x,
  smmla 1.075x); fixed128 codegen SMMLA 1.017/1.060/1.080 @K128/256/512.
  NO honest SVE256 numbers (VL capped on Grace; X925 pending). Conclusion:
  at 128-bit SVE2 = no significant win (instruction-pool only).
- X925 PLAN (if 256-bit admitted): capability gate (HWCAP bits + VALID VL
  selections; never infer 256 from name/bit/rejected prctl); minimal leaf
  envelope; leaf-first vs runtime-default; VL ownership per-worker (seal
  identity; fresh probe process); kernel priority (Q8/int8 decode
  panels/GEMV -> grouped-GEMV/MTP -> attention -> FP16/BF16 only under
  explicit contracts); evidence gates (census x envelope x route x VL x
  compiler; byte oracle; register witness to Z/P; ProductionTestPreflight);
  effort ~1-2d census / 3-5d one family / 1-3+w full. Risks: no gain@128,
  arithmetic changes, VL/layout mismatch, untracked armv9 baseline.
- GDN native ADMIT (Grace, exact cmd + diagnostics retained; 'Candidates: 1
  admitted'; NOT an executed-Sparks admission).
- Decision: if X925-256 -> resume SVE2 candidate (DGX lane); if 128 ->
  prioritize NEONDotProd/I8MM + emitted-FP16 verification (idea 31 +
  vfmlalq check).

## FINAL VERDICT (2026-11-09) — SVE2 CLOSED on SPARKS

- Corrected VALID probe (arm-sparks, saved sve-vl-probe-gb10-v2.txt):
  GET_VL=16; **SET_VL(32) -> returns 16 (errno=0), svcntb()=16** — the
  kernel ACCEPTS and CLAMPS to 16 = 128-bit HARD CEILING; SET_VL(64/128/
  8192/16) all -> 16; HWCAP SVE/SVE2/SVEI8MM/BF16/I8MM = Y (SVEBF16: v1 Y /
  v2 n — bit-10 hardcode discrepancy, note only; SVE2 class Y both);
  default sysctl 16 (RO, untouched).
- DECISION: even valid requests clamp to 16 -> **SVE2 closed on SPARKS**
  (128-bit; ~NEON parity per DGX sweep evidence; no 256-bit width play).
  Idea 30 = CLOSED/deprioritized (evidence complete). ISA queue = idea 31
  (S=4 spillfixes; census: NibbleLUT 22/22+6, IQ4_XS 31/20, Q5_K GNU clean
  = AppleClang-only, Q6_K classifier gap) + FP16/FMLAL verification.
