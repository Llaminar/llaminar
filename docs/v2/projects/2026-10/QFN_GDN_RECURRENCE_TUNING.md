# QFN GDN recurrence tuning

## Scope and current disposition

The active TP4 performance target remains Castagna's42.959 decode tok/s while
preserving roughly815 prefill tok/s. These experiments isolate the FP32
recurrence in `ROCmGatedDeltaNetKernels.hip`; they do not include projections,
convolution, QKV preparation, output normalization, communication or the model.
No whole-model gain is claimed from their event times.

QFN's current ordinary four-MI50 TP4 plan gives each participant12 value heads,
DK128 and DV128. Initial state is immutable and distinct from published state.
The one-row lane writes terminal state; the16-row verifier lane writes every
row snapshot and leaves the terminal bank's sentinel untouched. Both use the
real public recurrence launcher and retained HIP graphs.

## Independent baseline and regression

`ROCmGDNHistoricalRecurrentOracle.hip` preserves the literal original kernel in
a separate test translation unit. Before changing production, the native
historical tests passed against it. Coverage includes all M1..16, DK64/128,
DV64/128 and the65-column boundary, two unequal requests, full/partial/empty
live lengths, signed/zero inputs, dirty retained storage and20 M16 replays.
Every output, complete snapshot and terminal state is compared byte for byte
with both historical grouped execution and historical serial-row execution.
The explicit preflight registration is
`V2_Integration_ROCmGDNPairedQKHistorical`.

The separate Release performance fixture records64 complete recurrence nodes
inside one retained graph. After byte sanity and five warmups, twelve native
event pairs alternate production/historical order. It repeats the byte sanity
after timing. Native graph inspection proves function identity, every node's
geometry and a single ordered dependency chain. Setup, transfers and result
collection are outside the event interval. An isolated profiler process keeps
one selected family, five warmups and the sixth captured dispatch.

## Measured candidates

The unchanged control and literal historical kernel agree in timing as well as
bytes. Each candidate is paired with that same historical implementation in
its process; small changes in historical time expose the run's noise.

| Candidate | M1 production / historical µs | M16 production / historical µs | Native byte gates |
|---|---:|---:|---|
| Original256-thread recurrence | 12.639969 /12.639969 | 122.651085 /122.657336 | Pass |
|116: paired Q/K normalization | 12.305344 /12.654094 | 119.508445 /122.679703 | Pass |
|117: paired normalization,128 threads | 12.197657 /12.661414 | 115.625329 /122.685384 | Pass |

116 keeps each original square, wave64 XOR reduction, ascending four-wave
fold and final quotient/multiply expression. Independent Q/K work shares
publication edges, removing three CTA barriers per live row. K's temporary
four-wave totals borrow the existing recurrence reduction storage before that
storage's later use; the existing publication barrier retires the readers.
There is no new allocation, launch, global workspace or state authority.

117 keeps four K-row partitions and gives each CTA32 value columns instead of64.
The TP4 shape therefore launches48 rather than24 CTAs. The original norm
lanes0..127 retain their wave64 grouping; the absent old waves2/3 explicitly
publish their exact +0 totals before the same four-term fold. Per-column state,
reduction and output arithmetic are unchanged. Dynamic LDS falls13328→7184B
at DK128, and2576→1552B at DK64. External state/snapshot/workspace sizes stay
fixed. More CTAs duplicate more per-head norm/gate work, so broader shape
measurements are required before choosing this geometry globally.

The first117 build incorrectly changed the second HIP launch bound from2 to4
using a CUDA block-count interpretation. The mandatory spill guard rejected
DK64 with15 spill slots/88B and DK128 with35 slots/220B; that executable never
ran. HIP's second bound is minimum waves per execution unit, as the selected
SDK's `amd_hip_runtime.h` attributes and
[AMD's language-extension documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#launch-bounds)
confirm. Restoring2 preserves the original register constraint and compiles
without spills. The guard was not changed or bypassed.

## Evidence and remaining qualification

The baseline profiler reports124 VGPRs,96 SGPRs and no private storage for the
DK128 recurrence;116 reports128/96/0. Both retain the original grid and dynamic
LDS. The corrected117 profile reports128 allocated VGPRs,96 SGPRs,7184B
dynamic LDS and zero private storage; the native function attribute reports126
used registers. Its48 two-wave CTAs still execute96 wavefronts, matching the
original24 four-wave CTAs. These are resource and work observations, not
achieved occupancy. Counter collection is separate from unprofiled timing.

The118 breadth sweep covers32 cells: DK64/128, DV64/128,6/12/24/48 local value
heads, and M1/16. Every cell passes full output/state/snapshot byte comparisons
before and after timing. Thirty cells improve; DK128/DV64/H48/M1 is0.312%
slower, and DK128/DV128/H48/M16 is3.825% slower. The latter is a real limitation
of choosing128 threads globally, not evidence for a universal win. The current
TP4 geometry improves3.878% at M1 and5.609% at M16 in this sweep. These are
recurrence-only measurements; they do not certify entire topologies.

The unchanged185-row real-model panel also passes its required short and
held-out prefill/decode groups. All26 retained logit files are byte-identical
to the accepted observation115 control, including threshold and reset rows.
Threshold rows remain diagnostic rather than acquiring a new absolute KL
acceptance claim. The matched N512/D256 Release A/B/A is complete. Six control
samples have median 815.591867 prefill / 39.254592 decode tok/s; the three
candidate samples have median 815.849609 / 39.237071. The changes of +0.0316%
and -0.0446% are within the observed variation. All nine 256-token sequences
are identical, and every runtime, source, loader, driver and retirement check
passes. This is neutral model throughput, not a model-level improvement.

The complete ROCm GDN fixture's30 tests and the two explicitly selected
historical tests pass for both116 and corrected117, including long-prefill/
decode equivalence and captured state publication. Each CTest slice also
includes its required startup fixture. Driver windows and normal retirement
pass. The subsequent observation120 gate refresh passes776 Unit,46 affected
Integration and one setup registration, including these production GDN edits.
Full production preflight remains G60.

Local evidence under `parity-results/qwen-flash-next/`:

- `gdn116-baseline-m{1,16}-events-r1` and baseline Release build/archive.
- `gdn116-paired-native-r1`, `gdn116-paired-m{1,16}-events-r1`.
- `gdn117-128-integration-build-r1`: preserved rejected spill result.
- `gdn117-128-integration-build-r2`, `gdn117-128-native-r1` and Release build.
- `gdn117-128-m{1,16}-events-r1`.
- `gdn{116-baseline,116-paired,117-128}-m1-wavefronts-r1`: exact profile receipts.
- `gdn118-128-breadth-r1`: all32 event cells and their independently checked
  byte, native geometry, source, loader and retirement evidence.
- `gdn118-release-accuracy-r1`: canonical required groups and the complete
  `observation115-logit-byte-comparison.json`.
- `gdn118-model-summary-r2`: complete matched comparison, settings, samples
  and exact generated-token agreement. The initial wrapper failed before
  native launch because its local prompt copy was missing; the corrected run
  uses a fresh package and output directories rather than overwriting it.

The121 test-only extension adds independent historical scale-only coverage
beside L2 normalization. Both policies pass the focused padded-real-length
and historical registrations in18.34 seconds, including repeated dirty
replays. The proposed additional Q/K-load barrier removal remains unapplied.
Further GDN tuning is held while attention takes priority. The larger grouped
shape regression remains open before selecting a global geometry; no final
production promotion, refreshed full preflight or feature checkpoint follows.
