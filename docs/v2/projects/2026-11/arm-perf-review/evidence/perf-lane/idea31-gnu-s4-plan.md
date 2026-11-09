# Idea31 draft: GNU S4 nibble decode and Q5_K minimum epilogue

2026-11-09 — SPARKS/X925-first PERFORMANCE lane. PLAN ONLY / CODE HELD.
GNU14 first; AppleClang secondary qualification. Source-grounded
observations on detached 39c24d8b4. Required implementation base = final R2/
C4 tip 777f7a52a (on f69446e3c, incl. main-lane guards), re-confirmed by
arm-perf before any code. Do not apply this plan mechanically to a newer
seam.

## Objective / non-objectives
Reduce actual compiler FP/SIMD spills in four-row native grouped projection
while retaining production arithmetic, row sharing, layouts, source formats.
Work items = tracker GNU NibbleLUT S4 spill cells (reported 6-33) + Q5_K
additive-minimum epilogue pressure (tracker evidence, NOT a fresh census).
Committed budget = per-cell ratchet, not a target to increase.
Do NOT: row replay, float repacking, host mirrors, broad no-inline/
optimization disable, larger spill allowance, changed numerical boundaries,
implicit route change, Grace-only tuning. Do NOT mix with the separate
FP16/FMLAL projection implementation.

## Code owners / pressure hypotheses
Owners: CPUNativeNEONPreparedProjection.h (CPUNativeNEONProjectionTile<
Rows>::run/consume/consumePair/pairBlock/publishPairs/nibbles);
CPUNativeNEONContributionContract.h (accumulateSingleScaleBlockNEON /
accumulateCorrectedSingleScaleBlockNEON + rounded dependency edges);
CPUNativeNEONMultiScaleOps.h (ASIMD/SDOT/SMMLA ownership);
tests/v2/portability/CPUNativeNEONTileRegisterProbe.cpp; tests/v2/cmake/
CPUNativeTileRegisterEvidence.py + CPUNativeTileRegisterBudget.json.
At S4, run retains reduced[4] + accumulator[4] per logical partition/block;
block needs integer dots, decoded byte vectors, activation broadcasts,
weight-scale/minimum vectors, asymmetric correction temporaries. Pure SDOT
nibble walk unrolls 4 groups (low+high nibbles each); GNU hoists source
bytes, both nibble decodes + multiple row activations across the walk.
I8MM pairBlock unrolls 2 group pairs + keeps row-pair column accumulators +
odd-row support. Overlapping live ranges (not packed-tensor size) = first
hypotheses. Q5_K (unsigned 5-bit, distinct prepared encoding + additive
minimum): pressure may be in the correction epilogue even if the integer
loop is clean; do not conflate with Q4/IQ4 decode. Decode witness key to
actual encoding/map/minimum/policy before labeling a cell. Source sharing
allowed; format and arithmetic identity cannot collapse.

## Step1 — authenticate final-base problem before changing it
1. On the authorized ref (777f7a52a), verify source diff vs 39c + R2 seams;
   create isolated worktree/branch if needed (avoid another writer). Station
   GLM/GPU restrictions. No fetch/branch switch by this draft.
2. .80 GCC14: existing register-witness census on final source, exact
   production flags, NEON / NEON_DOTPROD / NEON_I8MM envelopes. Enumerate
   every S1..4 x ASIMD/SDOT/SMMLA x arithmetic policy x physical encoding/
   map/minimum cell. Save source/compiler/object + witness inventory identity.
3. For each reported S4 nibble or Q5_K cell: isolate run/consume/epilogue +
   caller/callee closure; disassemble; delimit K loop, correction epilogue,
   prologue, reduction. Classify each FP/SIMD stack access: real compiler
   spill/reload vs proven ABI d8-d15 preservation / typed intermediate
   storage / cold error edge. Reject ambiguous evidence; no source
   allowlisting.
4. Short table per target cell: key, static spill count + location, vector
   live-range contributors, expected loop instruction count, source policy,
   baseline timing. Total count alone does not identify a K-loop defect.

## Step2 — one controlled hypothesis at a time
A. Shorten nibble decode live ranges without changing dots: bounded one-group
   loop instead of blanket 4-way unrolling where witness identifies hoisting;
   load one raw16B group, decode/consume half, decode/consume other, retire.
   Same-block integer sums; no scale crossing. Check GNU still materializes
   groups/row-activation panels before assuming source scope helped.
B. Preserve grouped reuse/ILP: 4 output rows consume each weight fragment
   together with independent exact integer accumulators. Do NOT solve
   pressure by invoking the M1 program 4 times. If S4/S2 schedules differ,
   make a typed compile-time physical schedule with independent evidence;
   retain complete shape/thread dispatch. A genuine S2 grouped alternative
   may be measured explicitly but cannot be silently substituted/installed
   without overall economy + byte identity.
C. I8MM separately: keep two adjacent decoded groups per matrix operation +
   existing scale-half boundary; try reducing group-pair unroll/decoded
   lifetime, not dismantling pair ownership/panels. Retain exact row0/row1
   publication + Rows3 odd tail. SDOT evidence cannot certify SMMLA or
   mandatory ASIMD.
D. Q5_K minimum epilogue row-local retirement: consume each row's completed
   integer dot + correction as soon as its source scale/minimum is available;
   retire row-local temporaries before the next row. Do not keep all rows'
   converted dots, scaled weights, scaled minima, activation sums live
   simultaneously merely for parallelism. Shared weight metadata reused.
   Test a smaller live epilogue + a suitably ILP-preserving alternative; an
   unroll reduction that creates a dependency bottleneck is not a win.
E. No incidental codegen restrictions: scoped expressions, compile-time
   unroll policy, exact helpers. Keep required register dependency barriers
   for rounding; do NOT remove them to improve a spill count, add volatile
   memory, force values to stack, or disable vectorization globally. Inspect
   assembled results after every change, not inferred register counts.

## Arithmetic invariants — Q5 minimum cannot be reassociated
Both policies preserve increasing logical block order + existing partition
fold; integer dots exact. Preserve every persistRoundedNEON boundary:
- BackendNative: scaled_weight = round(activation_scale x weight_scale);
  value = FMA(dot_fp32, scaled_weight, accumulator);
  activation_correction = round(activation_scale x float(sum_qs));
  result = FMA(activation_correction, weight_min, value).
- GPUAlignedExpert: scaled_weight = round(activation_scale x weight_scale);
  dot_term = round(scaled_weight x dot_fp32);
  scaled_min = round(activation_scale x weight_min);
  correction = round(scaled_min x float(sum_qs));
  block = round(dot_term + correction); result = round(accumulator + block).
Ordering of two independent rows may change; arithmetic within one row may
NOT. Source scales/minima + unsigned-5-bit coefficients = exact prepared
words. Never replace the two policies by one fused/distributive expression.
Zero-valued minima are legitimate data, not permission to skip an unproved
branch.

## Gates / acceptance
1. Focused failure regression: record exact source/input/initial-state/policy/
   geometry; replay through actual production grouped implementation vs
   independent contract oracle + serial M1; add reset, partial columns/rows,
   dirty output, repeated replay. Register the production-defect replay in
   ProductionTestPreflight. A synthetic repeated call is not a recorded
   replay.
2. Total byte sweep: all formats/codebooks, every physical encoding,
   BackendNative + GPUAlignedExpert, M1..supported grouped incl. 2/3/4 + odd
   tails, N tails + panel boundaries, K block/partition boundaries, extreme/
   asymmetric minima + source scales. Verify raw dot/metadata contract AND
   completed FP32 outputs (no tolerance). Reuse existing native NEON
   numerical/prepared-projection/batch gates; extend, don't add competing
   lists.
3. Register witness: full compiler/envelope/instruction-family census +
   complete call closure. S1 stays zero-spill. New grouped candidate should
   reach zero inner-K compiler spills; do not accept a lower aggregate count
   that moved spills into a hotter loop or another format. Remaining explicit
   measured streaming exception needs the CPU skill's counter-backed
   justification + non-spilling comparison; not automatically certified by an
   old ratchet. Update committed budgets ONLY downward after authentic
   evidence; never raise or narrow inventory to get green.
4. SPARKS economy: canonical unprofiled baseline/candidate cycling in the
   existing NEON exact-overlay/production projection harness, production
   shapes/formats, representative grouped M, one core then physical-core
   scaling; preserve true prepared footprint + quantization/packing
   ownership. Report median, CV/noise, instructions/cycles/cache counters
   per exact candidate. Require less hot-loop stack traffic + end-to-end
   latency benefit without material neighbouring regressions. Grace only
   compiles/cross-checks; no Grace-specific tuning criteria.
5. Cross-compiler closure: after GNU accepted, run the same full source/ISA
   census + byte sweep on AppleClang; no silent Apple baseline widening or
   shared budget attribution. If one source schedule harms that first-class
   implementation, use an explicit compiler/codegen schedule only with
   independent gates + measured reason — not a fallback/vendor guess.
6. Canonical gate: complete Unit + ProductionTestPreflight, then .80 Release
   workload/PerfStats confirmation + applicable model E2E. No deterministic-
   mode/optimization disabling, tolerance relaxation or fresh expected-token
   baselining. Result artifacts ignored; conventional scoped commits only
   after meaningful green slices + operator approval to implement.

## Deliverables / sequencing
First: final-base authenticated S4 census + pressure/loop map + target-local
baseline. Second: one narrowly scoped GNU hypothesis (byte, register, timing;
reject if slower or spilling elsewhere). Third: Q5_K epilogue slice
independently qualified, then full combined sweep + downward ratchet update.
Fourth: AppleClang secondary + canonical gates. Effort: final-base analysis/
census ~half-1d; each slice ~1-2d incl. parity + .80 economy; full
compiler/format/ISA closure several days. Risks: stale pre-R2 seams,
confusing grouped tile names with physical row ownership, scale/minimum
reassociation, improving static spills while losing throughput.

HOLD ACTIVE: no production edits, new public modes, register-budget updates
or branch pushes. arm-perf supplies final base + approves the concrete first
slice before implementation.
