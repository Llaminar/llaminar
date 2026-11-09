# Idea 28 — IQ4_XS M=1 multi-thread regression after the metadata seal (R1)

Status: **Open (investigation)**. Found in addon window (b), 2026-11-09.

## Finding

Window (b): base `302318f1e` vs R1 `035d1d948` (Release NEON_I8MM, gcc-14,
production gemv path, ABBA, median of two launches per arm, 31 iters; binding
OMP_PLACES=cores OMP_PROC_BIND=close).

- **IQ4_XS 17408×5120 M=1: +6.6% @1 thread, −4.5% @10, −4.2% @20** (both
  launches consistent).
- IQ4_XS 2048×4096 M=1: +5.0 @1, **−4.1 @10**, +1.4 @20.
- IQ4_XS 512×2048 M=1: +11.7/+11.1/+7.3%.
- Everything else grouped/single-thread: **+1–12%** (IQ3_S 512×2048 M=4
  +9.0/+7.0/+7.6%; IQ4_XS M=4/M=15 positive everywhere; IQ3_S 17408×5120
  M=1 +0.9/+2.7/+2.0% etc.).

Claim boundary (lane, honest): byte-identical; **+3.1% 35B MTP-off decode**
(23.31 → 24.04; p50 41.8/42.2→40.5/40.8 ms), neutral MTP (33.94 vs 34.0,
±0.3% within noise incl. one launch outlier; acceptance 0.99 both), prefill
unchanged (~49.5–50 tok/s). NOT claiming economy for the IQ4_XS M=1 cell.

## Suspected cause (within the R1 diff)

- `noexcept` / `__builtin_unreachable` dispatch change (in-loop throw removal);
- the unchecked admitted accessors changing scheduling in the Rows=1 IQ4_XS
  nibble-source-scale path — under bandwidth pressure (17408×5120, multi-
  thread).

## Investigation plan (bounded, ~30 min)

1. Disassembly diff Rows=1 IQ4_XS nibble-source-scale path (base vs R1) —
   scheduling/inlining + dispatch edge.
2. Isolated confirmation ≥5× 31-iter at 10/20t @17408×5120 (raw+p95); if it
   holds, A/B the two candidates (noexcept dispatch vs admitted accessor).
3. Decision rule: codegen-attributable → FIX IN R1 (small amendment); clears
   as artifact/noise → record + close with the item tracked (R2 watch item).

## Evidence

`.80:build/results/w-r1-20261109T031823Z` + `build/wr1-summary.py`.

## ROOT-CAUSED + FIXED (2026-11-09) — b01d285dc (parent d1f1e8e34)

- CAUSE: R1 tile programs spill vector registers — with UNCHECKED
  admittedScales/admittedMins accessors, gcc hoists later metadata/payload
  loads into the dot program and spills accumulators/scale vectors
  (str q31,[sp]…ldr q25,[sp] once per block in Rows=1 IQ4_XS hot loop).
- Attribution (production gcc-14 -O3; gcc-16 same): base 0 spilling fns/0
  spill insns/43442 insns; R1 45/1154/38392; R1 w/ THROWING accessors 0/0/
  42866; address-dependency-only 40; memory-barrier-after-dots 31; hoisted
  strides 43; FIX (admitted accessor + FATAL NORETURN COLD bound check)
  0/0/38548. I8MM envelope: base 0, R1 48, fix 0. Compact kernels never
  spilled. MECHANISM: the old throwing check was the scheduling boundary
  stopping speculative hoisting; fatal noreturn provides it without
  exceptions — run path stays noexcept, instruction savings retained,
  in-loop reserved-bit test stays removed, arithmetic unchanged, M1 bytes
  pass.
- REGRESSION (maxim 22/24): CPUNativeTileRegisterEvidence.py — build-time
  probe of every tile (Rows1..4 × ASIMD/SDOT/SMMLA × policy × encoding) at
  -O3; ANY q/v stack traffic in run/pairBlock fails the build.
  V2_{Unit,Integration}_CPUNativeNEONTileRegisters re-verify the retained
  proof; adversarial tests V2_*_CPUNativeTileRegisterEvidence. Checker
  rejects d1f1e8e34 (120/176 spill), accepts base + fix (0/176).
- Reviewer informed (fatal accessor = small semantic delta; optional look).
- Pending: .80 build+gate at b01d285dc + 3-arm confirmation runs (base /
  R1-spilling / fix, ≥5×31, IQ4_XS+IQ3_S, 17408×5120+2048×4096, M=1/4,
  10/20t, ~45 min).

## Register-evidence gate hardening (SOURCE114 preliminary, 2026-11-09)

- Production fatal accessor + SIGABRT/terminate-77 regression = COHERENT
  (positive). Gate must upgrade before claiming 'complete' (else diagnostic):
  1) header-closure authentication: hash/bind the ACTUAL transitive
     compilation inputs (compile_commands-derived or manifest) + compiler/
     config identity; verify() rejects STALE production headers.
  2) detector width: catch stack-relative vector AND scalar FP (s/d) spills
     through ANY base (sp or fp, constant offsets); DISTINGUISH ABI
     save/restore prologue/epilogue from program spills (exclude only the
     former); include s/d widths.
  3) inventory: expected symbol/body inventory from every explicit
     instantiation (+ pairBlock), inline/call-closure accounting, partial-
     program negatives (mutate one body → reject).

## SOURCE114 FINAL (2026-11-09) — fatal accessor ACK; gate on HOLD

- Narrow source ACK for the fatal-accessor delta + focused regression
  (bounds reject before cast/pointer arithmetic; valid addresses unchanged;
  fputs+abort noexcept cold/noinline; no FP tree/dispatch/transfer change;
  fixture checks all 6 IQ3_S units × both accessors + all 8 bad-coordinate/
  accessor combos SIGABRT+diagnostic term→77; preflight registration exists).
  Direct-accessor regression — NOT a worker-team execution or rollback proof.
- REGISTER-EVIDENCE GATE: HOLD (tooling successor required, separately
  identified): exact compilation dep closure (omitted deps confirmed incl.
  tensors/NEONFP16.h + kernels/common/MoEProjectionNumericalContract.h) +
  RECORD the command EXECUTED; header-change negatives; classify-or-reject
  fp/SP-derived-base + s/d + ABI-save distinction; expected emitted-program
  inventory (rows×family×policy×noncompact encoding) + body/callee closure +
  partial/inline/outlined/empty-body negatives; never change production
  allocation to fit the scanner.
- Owner evidence (scheduling/0-176/M1 bytes/.80/3-arm) remains owner evidence.

## Update (2026-11-09, widened detector) — grouped confirmed; M=1 OPEN

- GROUPED regression confirmed + fixed: R3 SDOT 52→463→52 (base→R1→fix);
  R4 SDOT 70→666→~70; R2 30→24→30 (ASIMD); Rows=1 = 0 in all three on Linux
  gcc-14. The fix is justified by the grouped-tile spill regression.
- M=1 ATTRIBUTION NOT CONFIRMED on Linux gcc-14 (earlier spill = Darwin gcc
  output). IQ4_XS M=1 −4% cause OPEN: 3-arm run decides whether the fix
  changes M=1; else investigate separately (bind/admission per call, code
  alignment). Ratchet gate approved (idea 29): Rows=1 zero; Rows2–4 per-program
  budgets keyed by compiler id+major + envelope; unknown → fail closed.
