# Idea31 v2 — GNU four-strip decode-panel spill fixes

2026-11-09. SPARKS/X925-first PERFORMANCE lane. PLAN ONLY; no production code
changes. REPLACES the preliminary 39c draft's incorrect S4=four grouped rows:
**S=4 = four adjacent four-column strips = 16 output columns in the single-row
decode panel**; Rows and Strips are independent template dimensions.

## Authenticated base / exact scope
Base: 777f7a52a (R2 on main f69446e3c); fetch + detached checkout OK. FP16
chain + floating benchmark owner unchanged vs 39c. Seams verified:
ActivationBlock (one row's 32 activation bytes in low/high vector regs, one
load per K block); dotLane<Group> indexed SDOT on the selected original lane
(mandatory ASIMD has own exact impl); CPUNativeNEONProjectionBatch forwards
Strips (0 = installed width); CPUNativeNEONProjectionTile::run routes to
panel<...,S>.
INSTALLED: Rows1 SDOT (incl. I8MM single-row leaf) S4 for symmetric
ExpandedInt8 + IQ3SExpandedSourceScale; S2 for nibble, Q6_K, ExpandedInt8-
with-minimum. ASIMD + grouped/row-pair = S1. Idea31 asks whether currently
spilling forced-S4 nibble/minimum candidates can become economical
zero-spill single-row implementations; it does NOT change grouped row count
or turn an SMMLA group into serial rows.
Tracked obs to authenticate: NibbleLUT GNU S4 6-33; Q5_K min-epilogue
(AppleClang BackendNative 4). Prior tracker obs; final-base census must name
every exact program/width/compiler. No installed-width change until byte +
register + target economy gates pass.

## Owners / pressure
CPUNativeNEONPreparedProjection.h (run, panel, activationBlocks,
forEachGroup, nibbles, installed panelStrips); CPUNativeNEONMultiScaleOps.h
(ActivationBlock, dotLane); CPUNativeNEONContributionContract.h (symmetric/
asymmetric per-block arithmetic + rounded boundaries);
CPUNativeNEONTileRegisterProbe.cpp (canonical instantiations);
CPUNativeTileRegisterEvidence.py + Budget.json (complete closure +
downward-only ratchet); Perf__CPUNativeNEONPanelTournament.cpp (one candidate
per launch).
At Rows1/S4: reduced[1][4], accumulator[1][4], 4 independent integer-dot
chains + two shared activation-block registers. forEachGroup<4> expands 4
groups; strip loop unrolls 4; each raw byte vector -> low/high nibble
vectors. GNU may extend several raw/decoded strips or the lookup table/mask
across independent work — overlaps FP32 state + exact chains > 32 SIMD regs.
Q5_K: unsigned 5-bit prepared coeffs + additive minima; S4 fully unrolled
strip epilogues can overlap scale0/1, converted dots, scaled minima/partial
accs across strips; spill may be after the dot work. Decode witness tuple to
encoding/map/minimum/policy before naming a cell.

## Step1 — final-base census / pressure table
1. Final-base GCC14 + exact production envelope; source/compiler/object
   identity; discover .80 flags from the actual build (no Grace-local tuning).
2. Canonical installed-width witness SEPARATE from forced-width candidate
   inspection. The installed key certifies Strips=0; do NOT spoof that key,
   weaken inventory or claim its receipt certifies S4 nibble candidates
   installed at S2.
3. Non-installed baseline candidates: scratch instantiations through the real
   executeWith<Ops,Strips> at Rows1 for S1/S2/S4; reuse the witness's object
   parser/classifier/closure/structural rules in a clearly labeled
   NON-CERTIFYING census; keep Strips in each key; validate emitted variant
   inventory; never ignore unsupported/ambiguous forms. Production candidate
   needs explicit canonical witness extension, not ad-hoc proof.
4. Include nibble centered/nonlinear/source-scale variants + both policies;
   Q5_K unsigned-5-bit/minimum program; SDOT in NEON_DOTPROD + matching
   single-row SDOT leaf in NEON_I8MM. Installed ASIMD + grouped Rows2/3/4 S1
   = regression controls (not conflated with S4).
5. Table per exact cell: compiler/envelope, Rows, Strips, encoding/map/
   minimum/policy, spill count, K-loop vs scale/min epilogue location, live-
   register contributors, static census, .80 baseline latency/noise. Delimit
   ABI saves, cold paths, intended output/contract storage, real spills.
   Reject ambiguous stack evidence (no allowlisting).

## Step2 — narrow hypotheses (held until approval)
A. Nibble strip-local decoded lifetime: keep one raw strip's low/high decode
   live only until its two indexed dots consume it; retire before next strip
   where scheduling allows. Reuse the two activation-block regs across all 4
   strips; no return to scalar word loads or 4 independent S1 walks.
B. Group scheduling with compile-time lane indices: dotLane<Group> needs an
   immediate; a blanket runtime loop is NOT a drop-in. Vary compile-time
   group/strip scheduling + lifetime scopes while retaining indexed SDOT
   (bounded compile-time group pair / strip-local helper inlining). Preserve
   exact per-column dot + scale boundaries. Inspect emitted live ranges after
   each change.
C. Q5_K strip-local minimum completion: finish one strip's dot conversion +
   scaled weight/minimum + correction + accumulator before the next strip
   when it shortens lives. Share activation scale/metadata; do not
   materialize all strips' temporaries at once. Keep several independent dot
   chains for throughput; a serial bottleneck is not a win.
D. Retain panel economics: one activationBlock load per row/block shared;
   keep prepared source geometry, source bytes, 4-column strip ownership.
   NOT 4x S1 calls, no repacking/duplicate panels/stack forcing, keep
   persistRounded barriers, no optimization/vectorization disable. Installed
   S2 stays its own measured impl until S4 qualifies; no automatic fallback.
E. Installed policy last: change panelStrips only after target-local wins
   beyond noise + full evidence; symmetric S4 zero-spill + existing S2 +
   grouped S1 must not regress. Compiler-specific schedule = explicit
   independently gated impl, never a guessed vendor workaround.

## Arithmetic invariants (unchanged)
Same signed integer accumulation, logical block order, scale/coefficient
words, partition fold + every persistRoundedNEON dependency per output
column. Column interleave may change; arithmetic within a column may not.
- BackendNative: scaled_weight = round(a_s x w_s); value = FMA(dot_fp32,
  scaled_weight, acc); activation_correction = round(a_s x float(sum_qs));
  result = FMA(activation_correction, weight_min, value).
- GPUAlignedExpert: scaled_weight = round(a_s x w_s); dot_term =
  round(scaled_weight x dot_fp32); scaled_min = round(a_s x weight_min);
  correction = round(scaled_min x float(sum_qs)); block = round(dot_term +
  correction); result = round(acc + block).
No fused/distributive collapse. Unsigned-5-bit ints, min sign, scales, half
scale = exact words. Zero minima = data, not a shortcut permission.

## Gates
1. Recorded focused replay (exact inputs/state/width/format/policy/geometry)
   through production panel entry vs contract-ordered oracle + S1 per-column
   program; reset, dirty, partial, repeated replay; registered in
   ProductionTestPreflight. Synthetic loop alone is not a replay.
2. Byte totality: all formats/codebooks + physical encodings; both policies;
   Rows1 candidate + grouped Rows2/3/4 controls; S1/S2/S4 exact; N tails
   around 4/8/16/64 + chunk boundaries; K block/partition boundaries;
   extreme/asymmetric minima/scales; row selections; positive worker counts.
   Preserve sentinels + exact live stores; compare int dots/metadata + FP32
   bytes (no tolerance).
3. SINGLE-ROW ZERO-SPILL RULE (Rows1 S4 candidates): NO compiler FP/SIMD
   spill in the full admitted compute closure. Do NOT borrow grouped
   allowance, increase budgets, omit programs, or move spills out of the
   counted symbol. ABI saves + owned output storage proven distinct by the
   canonical analyzer. Installed inventory + source/compiler/object-bound
   proof authoritative.
4. SPARKS economy: final-base v2_perf_cpu_native_neon_panel_tournament; one
   candidate per launch; cycling with matched compiler/envelope/format/
   policy/M1/N/K/workers/input bytes; medians/CV/noise; include scale/min +
   publication costs (not SDOT throughput only); per-candidate profiling
   separate from canonical timing; repeatable latency benefit without
   neighbouring-format/shape/installed-route regressions. Grace: compile/
   cross-check only.
5. AppleClang secondary: same full width/compiler/ISA inventory + bytes +
   register closure after GNU; independently verify the Q5_K BackendNative
   S4 epilogue spills; no M1 baseline flag widening; no GNU evidence reuse.
6. Canonical closure: Unit + ProductionTestPreflight + .80 Release
   workload/PerfStats + applicable E2E. Grouped + mandatory ASIMD tests/
   contracts retained. No diagnostic deterministic mode, optimization
   disabling, tolerance relaxation, expected-token rebaseline. Scoped green
   slices committed only after implementation approval.

## Sequencing / decision
v2 approval -> final-base installed/candidate census + pressure table
(read-only/scratch) -> first slice: one measured GNU Rows1/S4 nibble
scheduling hypothesis (bytes + zero-spill closure + .80 timing) -> Q5_K
epilogue slice -> full combined matrix -> change installed widths ONLY for
qualified winners -> AppleClang + canonical closure.
Estimate: census ~0.5-1d; one candidate ~1-2d incl. parity/economy; full
closure several days. Risks: compile-time lane scheduling constraints,
defeating activation-block reuse, altered minimum rounding, static-win-
wall-loss.
HOLD: base auth passed; v2 approval + first-slice confirm + evidence precede
production edits. No implementation or installed-width change made.
