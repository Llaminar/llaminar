# QFN GDN recurrence tuning

## Scope and current disposition

Candidate177 is installed in the ordinary nonidentity M1 CUDA and ROCm paths.
It reads post-convolution merged QKV directly in the recurrence kernel, removing
one deinterleave launch while retaining the original FP32 operation order,
cache-owned live/speculative banks and snapshot publication. CPU already has a
direct merged-input path. Identity M1 and optimized multi-row prefill retain
their existing execution programs. No allocation or admission ledger is added;
the existing maximum-prefill scratch remains admitted, so this is not a peak
memory reduction.

The final focused gate passes **804 registrations: all 778 Unit tests, 25 affected
regressions and their fixture**. There are no CTest failures or skips; the exact
pre-existing intra-GTest Unit coverage limitations remain recorded separately.
The canonical selected QFN panel passes, with all **185 final-logit rows in 26
files byte-identical to accepted151**. Submission counters prove direct merged
M1 recording on all four MI50s, without claiming device replay counts.

| Final Release complete boundary | Original two-node control, µs | Direct merged M1, µs | Reduction |
|---|---:|---:|---:|
| ROCm H12 / keyH4 | 14.47875 | 12.09750 | 16.45% |
| CUDA H12 / keyH4 | 5.10400 | 3.56800 | 30.09% |
| CUDA H24 / keyH8 | 5.77600 | 4.27150 | 26.05% |

These event cells use DK128/DV128, ordinary in-place state, normalization,
64 repeated complete boundaries, five warmups and twelve alternating samples.
Native graph inspection proves 128 control nodes versus 64 candidate nodes;
byte checks run before and after the timed interval. The broader regressions
cover DK64/128, value-width tails, signed offsets, retained snapshot envelopes,
state restore, grouped equivalence and dirty captured replay.

Fresh ROCm profiles retain 96 waves, 796.5 VALU instructions per wave,
778.5 KiB fetched, zero LDS conflicts, 7184 B dynamic LDS and zero spills.
The actual loaded 172064-byte code object matches the final compiled object;
its raw and merged bodies are byte-identical to qualified177. The merged body
uses 119/120 VGPRs and 95/96 SGPRs (used/allocated). These are allocation limits,
not achieved occupancy. CUDA compilation passes the spill guard for SM80/86/89/90;
all 16 DK/state specializations have zero local storage and spills. Its nominal
static shared section is 12 B, while the loaded recurrence reports 16 B required
storage. The private collector now distinguishes those observations explicitly;
it does not infer a universal alignment rule. The original rejected H12 parser
receipt remains preserved beside its strict offline readmission. CUDA uses native
events and compiler metadata here; no new CUPTI attachment was attempted after
the previously recorded driver assertion.

| Matched full TP4 Release model throughput, tok/s | Frozen175 control | Candidate177 |
|---|---:|---:|
| Prefill median | 820.235 | 819.876 |
| Prefill range | 818.797–820.908 | 819.083–820.479 |
| Decode median | 40.151 | 40.328 |
| Decode range | 40.026–40.293 | 40.291–40.427 |

The measured changes are −0.044% prefill and +0.439% decode. Decode ranges
narrowly overlap and the second control is faster than the first: this is a small
measured gain, not a claim that the 42.959 tok/s Castagna target is reached.
All nine natural 256-token sequences match. A/B/A uses one warmup and three
measured requests per arm, the same Qwen3.8-Flash-Next UD-Q4_K_XL GGUF, full TP4
on four MI50s, sharded vocabulary, RAM ngrams, 28 physical NUMA1 workers, FP16 KV,
explicit Prefill8/Decode33 attention, prompt512/decode256/context4096, MTP/prefix
off and greedy seed42. All arms pass runtime, source, driver, resource and normal
retirement checks. Profiler durations are excluded from model throughput.

Integration caught and fixed the new merged launcher's overly strict snapshot
capacity equality: an M1 call may retain a larger verifier envelope. Both backends
now test that configuration, restore and subsequent live-state updates. The new
telemetry initially used broad enablement; the existing source guard caught it,
and it now uses the exact GDN domain. Focused regressions are explicitly in
production preflight, including `V2_Integration_ROCmGDNMergedVerifierEnvelope`,
`V2_Integration_CUDAGDNMergedDecode`, `V2_Integration_GDNMergedInputMetadata` and
`V2_Integration_ModelKLDDiagnosticTelemetryBoundary`.

Evidence is indexed by `gdn177-production-checkpoint-r1.json` under the QFN
local result root. The matched samples/settings are in
`gdn177-model-throughput-comparison-r1.json`; final gate and accuracy receipts
are `gdn177-unit-affected-r2` and `gdn177-model-fp16-r1`. This is a focused
development qualification, not a rerun of complete ProductionTestPreflight or
whole-corpus/image certification. The next experiment is an exact-order NativeQ8
producer/reducer fusion; no performance claim is made for it yet.

## Earlier component experiments

The following records describe prototype qualification before production wiring.
The176 preparation variants were rejected for scalar decode and their machinery
has been removed from the production source.

Candidate176 tests a complete existing deinterleave-plus-recurrence boundary.
Its preparation kernel publishes canonically normalized Q/K and rounded gates
once per head/row; the recurrence consumes those values with the original state
arithmetic. It adds no launch, but needs two explicitly owned gate planes before
any production installation. That prototype had no model-stage wiring.
The guarded Release build passes, and the complete-pair byte trial plus both
historical scalar/grouped tests pass. Retained full→partial→empty→full replays
preserve outputs, terminal state, snapshots and raw V bytes. This is a scoped
H12/keyH4/DK128/DV128 test, not full geometry qualification.

|176 complete two-node pair | Raw control µs | Prepared candidate µs | Latency change |
|---|---:|---:|---:|
| M1, live1 |14.47625 |14.61625 |+0.967% |
| M16, live16 |118.925 |97.0898 |−18.360% |
| M16, live15 |112.725 |92.0399 |−18.350% |
| M1, live0 |8.01123 |7.50499 |−6.319% |
| M16, live0 |14.85 |14.36495 |−3.266% |

Each cell includes five warmups, twelve alternating complete-pair event
samples and byte checks before/after timing. The one-row result does not qualify
the candidate for the decode target. Twelve exact-role profiles pass. Recurrence
VALU instructions per wave fall805.5→601.0, with96 waves and essentially unchanged
FetchSize786.5→786.375KiB. Preparation changes72→48 waves,99→171.25 VALU
instructions per wave and11.625→10.4375KiB fetched. State traffic is retained.
Prepared recurrence uses111/112 used/allocated VGPRs,7184B dynamic LDS and zero
scratch/spills. Static and dynamic LDS are checked separately: rocprof's CSV
LDS field describes the static allocation, not the runtime dynamic allocation.

The raw175/176 ISA comparison finds a real control change: the factored raw
recurrence changes126/128→124/124 used/allocated VGPRs and6612→6532 code bytes.
Its deinterleave body is identical. Historical numerical tests pass, but the
timing table is a same-source control comparison, not frozen175 performance.
A separate old-build complete-boundary control is required before promotion.
Evidence: `gdn176-component-build-r1`, `gdn176-functional-r1`, and
`gdn176-complete-pair-summary-r1.json` under the QFN local result root.

The176b wave-local four-part fold removes two barriers while preserving each
ascending reduction from positive zero. It remains byte-exact but increases
M1 complete-pair time to14.93135µs against14.47135 control.176c transposes shared
Q/K addresses, removing measured LDS conflicts2.82925→0, but takes15.0038µs
against14.4688 control. Allocated VGPRs remain112 and state traffic is unchanged.
Both retain zero spills and the same raw/control/preparation bodies as176.
The ISA shows native DPP folds and their dependency waits; it does not show
loss of packed LDS loads. Neither variant qualifies for single-token decode.
Their M16 results improve to95.74525/95.8103µs against approximately118.9µs;
this grouped result is separate from the active decode target. Evidence:
`gdn176{b,c}-*` and `/tmp/qfn-gdn176-bc-rejection/REPORT.md`.

Candidate177 instead reads original merged post-convolution QKV directly. It
retains raw normalization, gates and recurrence arithmetic, and removes the
deinterleave launch and expanded-plane traffic. The current prototype needs no
new gate planes or workspace and initially had no production-stage wiring. Its guarded
Release build and three focused native tests pass, including historical
scalar/grouped checks and exact complete-boundary outputs/state/snapshots for
M1/M16, offsets0/+5/−5 and dirty full/partial/empty replay.

|177 complete boundary | Same-source two-node control µs | One-node merged candidate µs |
|---|---:|---:|
| M1, live1 |14.4825 |12.10875 |
| M16, live16 |118.905 |115.195 |
| M16, live15 |112.745 |109.06 |
| M1, live0 |8.022555 |6.47129 |
| M16, live0 |14.8251 |12.485 |

The M1 component gain repeats at14.46125→12.10000µs. A separate captured
complete-boundary A/B/A against the immutable175 binary measures14.53395 and
14.52420µs for the old path, versus12.10125µs for177: a16.71% reduction against
the14.529075µs control median. The same-source raw control is14.475µs and is
kept distinct from that frozen baseline. These are component measurements;
whole-model throughput and production integration were still open at this point.

The extended native gate passes261 cells spanning every M1..16, DK64/128,
DV64/65/128/129, both normalization policies, three independent requests,
request holes and signed modular head offsets including INT_MIN/INT_MAX.
Four retained dirty replays per cell compare complete output, state and
snapshots against independent historical grouped and serial-row kernels.
An additional12-cell M1 test passes12 successive live/empty transitions with
the same persistent state bank as both input and output. The original extended
reporting script incorrectly expected a flattened deinterleave grid; its
offline recheck records the correction to per-request X extent and request Y
extent, preserving the original failed report and passing native run.

Exact-role rocprof joins report96 waves,796.5 VALU instructions per wave,
778.5KiB fetched and zero LDS bank conflicts for merged recurrence. Bounded
ISA inspection reports119/120 used/allocated VGPRs,95/96 SGPRs,7184B dynamic
LDS and zero private memory or spills. This retains the same register-limited
occupancy capacity as the original kernel; achieved occupancy was not measured.
CFG analysis proves signed modulo and invariant input-address setup remain
outside the causal row loop. The raw177 control body is byte-identical to176r2.

Evidence: `gdn177-frozen175-component-comparison-r1.json`,
`gdn177-extended-r2-recheck-r1.json`, `gdn177-inplace-r1`, and
`/tmp/qfn-gdn177-isa-private/REPORT.md`. No model speedup or default promotion
is claimed from these component results alone; final qualification is above.

The active TP4 performance target remains Castagna's42.959 decode tok/s while
preserving roughly815 prefill tok/s. The earlier recurrence-only candidates
below isolate the FP32 recurrence in `ROCmGatedDeltaNetKernels.hip`; their event
intervals exclude QKV preparation, projections, convolution, output normalization,
communication and the model. Candidate177's complete-boundary and model timings
are reported separately above.

QFN's ordinary four-MI50 TP4 plan gives each participant12 value heads,
DK128 and DV128. The original recurrence-only fixtures use an immutable initial
bank distinct from published state. Their one-row lane writes terminal state;
the16-row verifier lane writes every row snapshot and leaves the terminal bank's
sentinel untouched. They use the real public recurrence launcher and retained HIP
graphs. Candidate177 additionally qualifies ordinary in-place state and retained
verifier destinations as described above.

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
