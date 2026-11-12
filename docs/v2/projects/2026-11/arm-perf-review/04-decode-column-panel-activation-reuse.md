# Idea 4 — Wider decode column panel + activation reuse

Status: **ADDRESSED** — merged via R2 C4 (8fcc0fe0b): per-encoding panel widths + the installed-tournament contract. (2026-11-10 status flip)
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

## R2 rebase — 777f7a52a on f69446e3c (2026-11-09)

- Range-diff 4/4 `=` (C2c 55282a617 / C3 444f2245a / C4 a7be52cc8 / harness
  777f7a52a); clean; only shared file tests/v2/CMakeLists.txt (hunks far
  apart: Main L1013/L28584 vs ours L7750/L19000). r2wip; basef69; trees
  seeded. FusedCompletion now builds (base guard in).
- r2fa early: BASE solo arm 507s wall + exit1 = its OWN policy check
  ('p95 regret 8.955% > 5%') → cell fails on base; solo at 85% of 600s →
  parallel-ctest timeout expected → TENTATIVE base/env, not R2 (confirmed
  when R2 arm + solo gated-cell rerun land; not claimed yet).
- Orders: r2gate2 f69446e3c 777f7a52a → window-r2e f69446e3c 777f7a52a;
  GB10 CUDA gates at f69446e3c (.79) in parallel.

## A1 COMPLETE — ALL GREEN (2026-11-10, f69446e3c/777f7a52a)

- Byte identity: generated_token_ids sha256 IDENTICAL base==R2 ABBA across
  ALL 4 cells (35B/27B x nomtp/mtp, 256 tokens each).
- ECONOMY (decode tok/s, ABBA mean): 35B-nomtp 23.2 -> 25.3 (+8.9%);
  35B-mtp 34.1 -> 34.8 (+2.0%); 27B-nomtp 5.30 -> 6.14 (+15.8%);
  27B-mtp 5.58 -> 5.78 (+3.7%). Prefill ~neutral (<=1%). ZERO regressions.
- R2 EVIDENCE COMPLETE: gate (0 new/0 gone; census 12/12, 9/9, 0 mismatches),
  r2fa (base/env, idea22), A1 (green, byte-identical, decode wins).
  remaining: #151 fix-back -> reviewer recheck -> WSL gate -> handover.

## R2 final-tip evidence COMPLETE (2026-11-10, 8fcc0fe0b vs 29351a57e)
- Short engine confirmation (w-r2es-20261110T060322Z, 35B ABBAx2): byte
  identity token sha base==R2 both cells; decode 35B-nomtp +7.9%
  (23.82->25.71), mtp +2.3% (34.11->34.88); prefill neutral.
- Gate at the final tip GATES_DONE (BUILD=0, PCH 9/9, census 12/12 + GNU-14
  9/9 Rows1=0 budgets byte-equal, proof 242/1274, REL both =0); focused 43/43
  = FIRST FA public-entry execution (owner gap closed). R2 remaining = WSL
  x86 leg only -> handover doc to Main (engine row filled; WSL pending).
- GB10 CUDA gate seal: K14-GB10-CUDA-F69446E3C-20261110 (51/64; tracked cells
  incl. RoPEGraphCaptureNoH2D GREEN; FA null-stream = single real defect,
  attribution + base control in flight).

## R2 x86 GREEN (2026-11-10, WSL; queue emptied 06:59Z)
- R2 FINAL 8fcc0fe0b vs 29351a57e: 0 new/0 gone; addon cells 14/14; C2c
  verbose 3/3 + 6/6 OK (0 skips); census +2 (admission registrations; none
  gone/label-changed; 3 ARM-only noted); Release both rc=0. 01 FINAL
  5d606a108 vs ce9ddb326: 0 new/0 gone + 4/4 extras. Base ce9ddb326 vs
  f69446e3c: 0 new/1 gone = MoEOwnedSharedBackingNativeStartup NOW PASSES
  (5.0s vs 0.44s fail) = WSL 'env' failure FIXED by the ARM-tip NUMA commits
  (real improvement). Base 29351a57e x86 gate: GREEN. R2 = ALL GATES COMPLETE
  -> handover doc to Main (lane). WSL disclosure: late reporting, no infra.

### Lane-2 residual candidate (addon plan, ~11:10Z): M=1 activation-reload headroom

Documented open observation: M=1 decode saturates ~8 threads and Q8_K
regresses 2.96× at 10 threads (00-measured-baseline). C4 widened N panels,
but candidate-2 (32-byte activation payload loaded ONCE per block +
lane-indexed dot + per-panel metadata reuse) was only partially realized —
consume()/consumePair() still walk strips with per-strip activation
reloads. Plan: M=1 decode economy window (Q8_K/Q8_0/Q4_K, threads
1/10/20, baseline shapes, ABBA) at bb893cd79 to quantify the
activation-reuse headroom; if the window shows the bandwidth/reload
signal → implement load-once-per-block + vdotq_laneq activation feeding
per panel with byte-auth vs serial, panel tails, register witness (Rows=1
zero spills, budgets only down), tournament before/after. The Ops seam
stays encoding-agnostic for the MXFP4 lane. NOTE: ideas 03/04/11 are
ADDRESSED in the tracker (R1/R2 lands) — the Lane-2 "successor scope" was
based on a stale index; the true residual is this idea-04 observation.

### **MATERIAL CORRECTION + successor plan draft v1 (arm-addon, ~14:54Z)** — execution gated on the window

Full plan: evidence/idea-04-successor-plan-draft-v1.md (also
arm-addon-notes/idea-04-successor-plan.md). **MATERIAL CORRECTION,
flagged prominently: the first interpretation guide's amplification claim
was WRONG and is RETRACTED** (idea-04-m1-window-notes.md rewritten v2).
The deeper source read shows **C4/R2 ALREADY implements candidate 2 for
the production M=1 leaves**: activationBlocks+dotLane (register-fed
vdotq_laneq_s32, no memory access) for nibble/expanded panels,
activationGroup hoisted per group for Q6_K (8×4B = the payload exactly —
NOT amplification), activation-scale conversion hoisted per block. **The
production M=1 leaf reads each activation byte exactly ONCE at
bb893cd79** — and the baseline Q8_K regression (measured 2026-11-07)
PREDATES the C4 land (2026-11-10).

PLAN: §0 what already exists (do-not-re-propose list with source lines);
§1 residual surface A — the S=1 strip walk (ASIMD leaf, SMMLA row-pair
tiles, shared-row tiles) where consume() still reloads per
(row,block,group,COLUMN): block-outer/columns-inner restructure with the
same dotLane seam, order-exact, weight stream linearized, register
pressure ~7-8/32 q-regs live at Rows=1 ASIMD (no spill risk; SMMLA +4
q-regs, witness budgets only-down); §2 residual B — Q6_K scalar→block
load unification (marginal, L1 load-port only, only if Q6_K-specific);
§3 oracle (serial reference byte-identical, PanelTailsMatchSerial +
exact-batch preflights), tournament re-runs (installed native|expert
contract, budgets only down), evidence ladder (M1 core-free → .80
focused via coordinator → engine E2E A1 → handover); **§4 THE HONEST
DECISION TREE** — regression gone → measured-and-negative, idea-04
CLOSES, no speculative implementation; regression persists uniformly →
reload hypothesis DISCONFIRMED for the production leaf, pivot by CSV
shape (K-walk latency → partition-shape; weight-stream decline →
layout/prefetch plan-first; knee cliff → scheduling owners ideas 05/26);
Q6_K-only → §2; §5 sequencing.
IMPLICATION: the M=1 window is now a DIAGNOSTIC of the baseline
regression's cause, not an activation-reuse headroom probe — the
headroom hypothesis was already implemented at bb. Window readout lands
on the §4 tree.

### IDEA-04 M=1 WINDOW BOOKED (~06:09Z): queued behind W3 + B1ATTR2 on .80

The FMLAL queue is CLEAR (PR #43 landed at 0ad09600b). The idea-04 M=1
decode economy window request: 0ad09600b Release NEON; encodings
Q8_K/Q8_0/Q4_K + Q6_K control; M=1; baseline shapes; threads 1-20;
ABBA 31 iters; OMP production binding. Decision gate = the Q8_K
10-thread regression signal + the per-encoding saturation knees; the
interpretation kit (w-m1act-summary.py + idea-04-m1-window-notes.md)
is staged. Seal K14-I04M1-0AD09600-20261112. The implementation gate
is decided by the window's signal per the integrator's rule.

### MAIN'S CONDITIONS ADOPTED (~06:09Z): one job on the host; record identities; NO profiler on the timing run; prune the build tree afterwards; results = diagnostic evidence NOT an economy admission; any kernel change follows the normal implement → validate → handback path. Seal K14-I04M1-0AD09600-20261112. .80 queue: W3 → B1ATTR2 → idea-04 M=1 → 27B rider.

### .80 SEQUENCE EVENT (~07:01Z): the M=1 window launched ahead of the B4 canonical (guard held)

Main's launch guard saw youthful_brattain (the idea-04 M=1 build +
timing runs at 0ad09600b, tree tip-i04, created 06:58:47Z) and did NOT
start main-b4c — no overlap. **Accepted sequencing: M=1 (a timing
window runs alone) → main-b4c (the B4 pre-application canonical) → 27B
rider.** Main polls for the M=1 exit; the coordinator relays the seal
(K14-I04M1-0AD09600-20261112) + the interpretation table when sparks
reports.

### M=1 WINDOW COMPLETION UNCONFIRMED (~07:19Z): the container is gone, no seal reported

Main reports youthful_brattain gone + build-i04 exists, but sparks has
NOT reported the window's completion (no seal K14-I04M1, no
interpretation table). Pressed for the M=1 state + the .79/W2 counter
re-run status (also overdue since ~04:45). Main's B4 combined-tip
canonical at 7a4790c77 (the A4-restore tip) is staged behind the
confirmations — WAIT issued.
