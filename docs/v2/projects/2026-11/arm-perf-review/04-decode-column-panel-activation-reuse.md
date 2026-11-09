# Idea 4 — Wider decode column panel + activation reuse

Status: **Open** (inner kernel unchanged since `ebe8da586`; the workshare
improvements in `33690f9b1`/`408ff90e5` address the outer task structure, not
the tile K traversal).

## Context

`CPUNativeNEONPreparedProjection.h::run()` (~:254):

- Outer loop over 4-column strips; inner loop over K blocks (32 values) and 8
  weight groups.
- `consume()` (~:219) reloads/broadcasts the 4-byte activation word **per row,
  per group, per column strip**, so the same 32-byte `Q8_1Block` payload is
  touched up to 16 times per block. The activation scale conversion
  (`nativeVNNIFP16ScaleToFP32(a.d)`) and `sum_qs` are recomputed per strip.
- For M=1 there is one int32 dot chain and one f32 accumulator per strip that
  are live at any time; the strip walks K serially, so latency of the dot
  chain (not issue width) bounds decode. (The "three registers" shorthand in the
  review is not a complete ISA/PMU finding — weights, LUT/planes, factors,
  corrections and reduced partials also live.)

Candidate changes:

1. Wider N panels (e.g. 16–32 columns) with several independent accumulator
   chains when M is small. Each column keeps its own fixed K/block/partition
   program, so arithmetic per column is unchanged; the work per byte re-read
   rises. The existing `#pragma GCC unroll 1` on Q6_K exists only to stop
   hoisting eight decoded groups into a spilling Rows=4 live set — it is not
   evidence against Rows=1 wider panels.
2. Load the 32-byte activation payload once per block into two registers and
   feed lane-indexed dot (`vdotq_laneq_s32`/ASIMD equivalent); reuse the
   metadata (scale, `sum_qs`) per panel. Note `Q8_1` has metadata in addition
   to the 32-byte payload, and Q6_K has low/high scales whose per-block FP32
   order must survive.
3. Both ASIMD (baseline) and DotProd variants need real implementations; byte
   auth per encoding.

## Exactness / risk

- Low but real: any change must keep each column's exact block/group order and
  per-block FP32 rounding (backend-native FMA vs GPU-aligned round per block).
- The Q6_K low/high scaling path and `WithMinimum` correction path are the
  sensitive ones.
- No current A/B bandwidth evidence establishes the winner; this needs the
  harness (idea 1) before and after.

## Decode M=1 thread-scaling observation (2026-11-07)

M1 Max VM: decode M=1 saturates around **8 threads**, and Q8_K **regresses to
2.96x at 10 threads** (already in `00-measured-baseline.md`). This may share a
cause with this idea and idea 12 (activation reuse / bandwidth): worth
checking on GB10 at 1..20 threads from P1, and a Q8_K regression signal would
argue for the wider-panel / activation-reuse work before touching scheduling.

## Validation

- Panel tournaments per Rows × encoding (Q4_K/Q5_K/Q6_K/Q8_0/IQ2_S/IQ3_S/
  IQ4_XS/MXFP4), M 1..16/31/33, workers 1..20, byte-auth vs serial, then
  fused bundle and engine confirmation.

## Coordination

Must be one shared change with the PR #35 MXFP4 lane (`NibbleMap::MXFP4` uses
the same tile); certify all `NibbleMap`/scale representations on the combined
source.

## Coordination update (2026-11-08, operator)

MXFP4 is UNTOUCHED and stays OPEN; the ARM branch merges FIRST. Idea 04 is
implemented within the ARM stack only (no MXFP4 coordination/wait). Keep the
Ops seam (pairDot / activation-pair API) encoding-agnostic so the MXFP4 lane
(PR #40) can adapt and adopt it later without a rewrite; note the compatibility
in the plan. Any NibbleMap touch points must remain seam-based.

## R2 in progress (2026-11-09)

- WIP rebased onto 5d4e3716a (local: f9aa1e10e C2c FA admission →
  217520f11 C4 panels → f41f0adc6 tournament harness; 6 TUs syntax-clean;
  NOT pushed yet). Plan: WIP review vs R1 (panels vs fatal-bound admittedUnit
  + register witness); C2c ISA-neutral (3/6/11% + 5/10/20% check costs in
  commit body); C3 (Ops seam) split from C4 (panels); C4 byte auth vs serial
  (every encoding × Rows × M × N tails × workers) + PanelTailsMatchSerial
  preflight; register-witness inventory → panel programs (Rows=1 zero
  spills, budgets only down); GB10 tournament + engine E2E (A1; windows via
  arm-perf — coordinate with CUDA lane on .79); self-gate → fresh reviewer
  checkpoint → WSL at R2 tip. R1 merged (5d4e3716a) — R1 DONE.

## R2 gate stop (2026-11-09) — registration finding

- C2c test V2_Integration_CPUFlashAttentionInvocationAdmission registered
  with FIXTURES_REQUIRED=V2_Models + ProductionTestPreflight → base guard at
  tests/v2/CMakeLists.txt:37501 rejects ('Preflight is deliberately
  model-free... NO_MODELS'). FIX: NO_MODELS registration (model-free by
  design), keep label. arm-sparks held (b)/(c); resume at fixed SHA; no
  local patch (tree exact). R1 configured clean at 5d4e3716a (same dir) —
  .80 fine.

## R2 gate (a) finding (2026-11-09) — correctness blocker

- 623e44ccf: CFG=0 BUILD=0 CENSUS=0 (12/12 + 9/9 budget-exact) Release both
  tips SUCCESS (after external oneDNN/cnpy seeding = env fix); CELLS 100%.
- Unit 979/14 (13 R1-known base + NEW: CPUNativeNEONPreparedProjection) +
  Pre 2363/14 (12 R1-known + NEW: CPUNativeNEONI8MMOperandGrouping +
  CPUNativeNEONPreparedProjection). VERBATIM
  (I8MMOperandGroupingMatchesSerialSDOTWalk, :634): serial.size() 864 vs
  pairs.size() 816 → **48 products MISSING (6.7%)** in the R2 grouped I8MM
  walk = REAL feature defect (grouping drops products; 816 = 864 − 48 →
  group boundary/trailing partial chunk). Lane fix-back; arm-sparks holds
  (b)/(c) until the cell is green at the fixed SHA.

## R2 grouping fix — f3b28a120 (2026-11-09)

- ROOT CAUSE = TEST ORACLE, not kernel: 'serial' side ran the INSTALLED S=2
  panel width; panel computes sealed PADDING strip terms (never stored):
  48 = Rows×blocks×8 = 2×3×8; 864 = 816 + 48; SMMLA had every live product.
- FIX (test only + public accessor): oracle pinned to STRIP WALK
  (executeWith<SDOTOps,1>); new stronger asserts (panel terms = oracle +
  exactly Rows×blocks×8×padding_strips; oracle sub-multiset via
  std::includes; compact padding_strips=0); `panelStrips<Ops>()` public
  constexpr (no code change). Kernel UNCHANGED; folded into C4 (72e234169);
  C2c/C3 range-diff `=`; harness rebased to f3b28a120.
- Lane GB10 verify (tree 2dc344326 = f3b28a120^{tree}; g++-14 Integration
  flags; scratch): 21/21 ×3 (incl. I8MMOperandGrouping, PanelWidths,
  I8MMRowPairsMatchSDOT, DotLane); scratch removed; no timing. Mac 14/14.
- Reviewer note (for R2 checkpoint): S=2 ≤1 padding strip per final chunk,
  compute-only, never stored; panel-width/tail tests prove no store past N.
- arm-sparks resume (a) gate → (b) tournament → (c) A1 at f3b28a120.

## R2 (a)+PCH clean; (b)/(c) re-booked (2026-11-09)

- f3b28a120: CFG=0 BUILD=0 CENSUS=0 (12/12); Unit 13 + Preflight 12 =
  EXACT R1 base sets; PCHBUILD=0 9/9 rerun (ParityFixture incl.). Cells
  2/199: GroupedSealedPlan (known base) + V2_Perf_CPUFlashAttentionTile
  Tournament TIMEOUT 600s (ran --parallel 20; UNCLASSIFIED; needs solo
  base-vs-R2).
- LANE SCRIPT BUG: build-release.sh pre-rename target
  (v2_perf_cpu_native_vnni_thread_scaling) → unknown target; binaries never
  built; scripts echoed EXIT/rc=0 after redirect → (b)/(c) logs read DONE
  but NO measurements. FIXED: renamed target, fail-closed, binary checks,
  real rc, r2rel.sh. arm-sparks warned: DO NOT report void numbers.
- RE-BOOK: r2rel.sh (REL_BASE=0 REL_R2=0 + 3 binaries) → window-panel.sh
  f3b28a120 → window-r2e.sh 5d4e3716a f3b28a120 → optional solo FA
  tournament A/B (else WSL/x86 classification).

## R2 (a) fix — f9f190059 (2026-11-09)

- DECISION (a) implemented: panel tournament + smoke move out of
  V2CPUArchitectureTests.cmake (early return UNCHANGED — base convention
  untouched) into tests/v2/CMakeLists.txt next to
  v2_perf_cpu_native_quantised_thread_scaling under an arm64 guard; Release
  perf-only selection admits every v2_perf_* target (V2TestTargetSelection)
  → Release produces the timed binary; same sources/flags/OpenMP/smoke
  (add_v2_test NO_MPI NO_MODELS TIMEOUT 120); reason in the commit body.
  Rebased onto #43 **7d1a94479**; r2wip; supersedes fda883839.
- Verified .80: Release + Integration CFG=0 vs 7d1a94479 guards; panel +
  FA tournament present in BOTH configs; Release target build TOURBUILD=0
  (2.4 MB); INT registrations present; smoke ran (0.08 s — pass line is
  r2gate2's evidence).
- CORRECTION: r2fa had NO gap — the FA tournament already lives in the main
  perf section (21 refs in Release); only the panel tournament was hidden.
- SINGLE-TIP sequence (relayed): r2gate2 7d1a94479 f9f190059 → window-panel
  f9f190059 → window-r2e 7d1a94479 f9f190059 (real exit codes; plan
  regen guard) → r2fa 7d1a94479 f9f190059 → push → reviewer → WSL.
