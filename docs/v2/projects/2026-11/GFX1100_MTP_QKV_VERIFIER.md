# gfx1100 QKV verifier component qualification

The actual fused-stage fixture builds in Release and Integration. All85 Release
cases pass: every21-format/four-geometry combination plus the mixed-source/bias
control, each through QKV and KV-only at every physical M2–16 and live prefix.
All five registered Integration replays also pass. Ordinary-power full-extent
timing now covers all84 source/geometry cases and2,520 cells. The proposed
Automatic policy retains1,217 distinct keys with a5% qualification margin
across the candidate panel and two actual-Automatic panels. Final actual-Automatic
timing passes all2,520 cells: all1,415 native-selected cells win at the median
(geometric mean1.32030×);1,105 cells retain Dot4. All five Automatic Integration
registrations/86 cases and the focused host regressions pass. All ten affected legacy feature registrations and all64 exact-producer
geometry profiles also pass. This complete-bundle component is qualified.

## Operation and ownership

`Test__ROCmQKVVerifierTransaction.cpp` invokes `FusedQKVGEMMStage` through its
device-counted decode-equivalent verifier path. Each Dot4/WMMA candidate owns
one prepared store containing three separately seeded original sources, its
own mutable kernels, persistent workspace, outputs and retained graph. The M1
oracle has a third prepared store and independent activation quantization.

The panel uses four component geometries from the Qwen release catalog:

| Geometry | K | Q/gate width | K/V widths |
|---|---:|---:|---:|
| Qwen3.6-27B full |5120|12288|1024/1024|
| Qwen3.6-27B local TP2 |5120|6144|512/512|
| Qwen3.6-35B-A3B full |2048|8192|512/512|
| Qwen3.6-35B-A3B local TP4, replicated KV |2048|2048|512/512|

These are local arithmetic components, not serving-topology certificates.
QueryInterval original K/V slicing explicitly excludes retained MTP in the
model graph; this fixture does not enable it or invent Q-slice provenance.
Decode-replicated attention can also choose full geometry in an MTP model.

KV-only omits every Q field from the actual stage descriptor. A separate poisoned
Q output must remain untouched. The mixed-source control uses original
IQ2_S/IQ4_XS/Q8_0 weights plus nonzero per-column biases through the same stage.
The canonical21-format registry owns the homogeneous source inventory.

## Replay and timing contract

Each physical M2–16 retains both candidates through full/one/empty/regrown,
repeated and every intermediate prefix. Reset preserves graph identity. Outputs
and consumed quantization/partial banks start dirty; active bytes must equal the
independent actual M1 stages and inactive outputs must retain poison.

Native inspection authenticates the counted quantizer's original input, Q8,
scale/sum banks, grid/block and exact row-authority argument. WMMA packets bind
the same publication, original prepared source descriptors, per-role output and
bias, live-row identity, partition span and geometry. Partial planes must lie
within the candidate's dedicated workspace, remain stable across repetitions and
never overlap another role. Graph inventory rejects private scratch.

Independent quantizer and WMMA-family omissions must break nonempty results,
preserve empty output, and recover immediately on restoration. These are family
necessity controls, not individual-packet necessity claims. Submission counters
advance at recording and remain unchanged during retained replay.

Disabled paired tests retain eight complete operations per internal event
interval and collect31 alternating Dot4/WMMA pairs for each physical/live count.
Poisoning, observation and setup remain outside the interval. The collector
requires ordinary `AUTO` power before and after collection and does not mutate
power policy. The initial every-prefix paired tests cover full dense and local
MoE geometries. The new `FullExtentPaired` inventory covers all four geometries
and every source at M2–16, with prefix correctness still executed before timing.
Its full-source economy run passes84 cases/2,520 cells/78,120 paired samples.

## Current evidence

Evidence root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Build logs: `qkv-verifier-{release,integration}-build.log`, both exit0.
- Release smoke: `mtp-qkv-verifier-release-replay-1794782329305795889/receipt.json`,
  one complete case, Q4_0, local MoE, both projection sets; native execution2.055s.
- Full Release replay: `mtp-qkv-verifier-release-replay-1794782369765966142/receipt.json`,
  all85 cases,261.49s total including process lifecycle. The collector's original
  bytes are preserved by `qkv-replay-collector-snapshot.json`; its later XML
  reader change accepts explicitly executed disabled timing tests while still
  requiring every selected case to run and complete without skips.
- Integration registered replay: `qkv-verifier-integration-focused-1794782649395902692/receipt.json`,
  all five registrations/all85 cases,262.93s CTest wall time.
- First paired timing: `mtp-qkv-verifier-release-timing-1794782922878304311/receipt.json`,
  four cases/1,200 physical/live cells/37,200 pairs, all byte checks pass, ordinary
  `AUTO` before/after. The native tests complete in73.232s.
- Five focused replay registrations cover the four geometries and mixed bias;
  all five are explicitly listed in production preflight and have passed here.
- Initial source/binary/evidence snapshot: `mtp-qkv-verifier-seed-snapshot/snapshot.json`,
  three records/35 files, SHA256
  `635287bdbbd02e13dc2dc2b97b9bcb9d4b9ac1c95af88ffaeec219b5a276cbd2`.
  It predates the timing-roster extension and retains those exact fixture bytes.
- Full-source full-extent timing:
  `mtp-qkv-verifier-release-timing-1794783251594465838/receipt.json`,433.117s,
  ordinary `AUTO` before/after with unchanged source/binary bindings.
- Full-extent source/binary/evidence snapshot:
  `mtp-qkv-verifier-full-extent-snapshot/snapshot.json`,two records/29 files,
  SHA256 `2d908fe5c029daefbb9f2b85ff705da8f7c41e45527aab6053a47d21c9bd95df`.

The first full-extent paired results are:

| Geometry / operation | Q4_0 first all31-pair win | Q4_0 atM16 | IQ4_XS first all31-pair win | IQ4_XS atM16 |
|---|---:|---:|---:|---:|
| Full dense QKV |9|1.3891×|9|1.4479×|
| Full dense KV-only |5|2.0826×|5|2.2508×|
| Local MoE QKV |9|1.2141×|5|1.3642×|
| Local MoE KV-only |5|1.2569×|5|1.2876×|

Every later full extent in each row wins all31 pairs. Shorter prefixes still
favor Dot4; these results do not support selecting WMMA for an arbitrary
counted graph from capacity alone. The complete panel covers every source and
all four geometries at M2–16, reusing complete-operation timing and the preceding
all-prefix correctness controls.

## Automatic policy and final timing

The full/local MoE KV operations have identical source/geometry/operation keys;
their observations are intersected, leaving2,205 distinct keys. Every one of
the1,282 selected keys wins all31 pairs in every observed copy. Their median
gains span1.00448–2.62378×, geometric mean1.30460×. The other923 keys retain Dot4.
These are forced-candidate results, not actual-Automatic performance claims.

The first actual-Automatic panel covers all2,520 cells and selects1,499 native
cells. Of those,1,496 win at the median, with geometric mean1.30487× across the
native set. Three local-MoE QKV keys lose every pair: Q4_1/M7(0.91906×),
Q4_K/M7(0.96069×) and Q2_K/M9(0.99628×). The corrected table removes exactly
those three keys and retains1,279 distinct native keys. Both host policy and
native captured-engine regressions assert their exclusion. The initial actual
Automatic evidence remains frozen in
`mtp-qkv-verifier-automatic-snapshot-1794786090552059328/snapshot.json`, SHA256
`50f0ecaae4dd930175ac9b9547212fbf811017ade2d761b77661befc4697fb52`.

A second full Automatic panel exposes four additional local QKV ties: IQ1_S/M9,
IQ2_XS/M7, Q3_K/M8 and Q4_K/M9 lose0.04–0.61% at the median. The policy therefore
requires at least5% median gain in all three panels, with complete observations
in both Automatic runs. It retains1,217 distinct keys across every source,
removing65 original candidate selections. This margin keeps unstable crossover
points ordered; it does not promise a universal5% gain on future requests.
Both suites explicitly replay all seven observed losing source/width pairs.
Second-panel evidence is frozen in
`mtp-qkv-verifier-automatic-snapshot-1794787025443551645/snapshot.json`, SHA256
`8ef4394d29f87d26eafc0c64020dcd2e465e9b306b9b3a3956a3cab621f09b7a`.

`VerifierProjectionBundleBinding` borrows the original stage's exact descriptors,
input, device, row authority and reduction geometry before policy subgrouping.
The ROCm bridge authenticates original prepared sources, distinct outputs and
all three Q8 activation banks. Every native packet must consume an ordered,
one-to-one subset of those original members. All members must match their own
serial partition and wave policy. Mixed sources, bias, output mappings/slices,
unknown bundles and guarded capacities remain ordered. An unqualified complete
bundle cannot borrow an independently qualified scalar member.

The Automatic fixture supplies the production `ScopedCapturedVerifierExtent`
and measures its exact full-M extent. It still replays every partial prefix for
correctness without treating those observations as exact-hosted timing. Fixed
Q4_0 anchors require native M16 and Dot4 M2 independently of the selector; the
collector joins every actual captured engine to frozen full-operation winners.
Separate exclusions exercise mixed unbiased sources, homogeneous biased sources,
guarded capacities and an unknown QKV bundle containing a measured scalar shape.

Final R3 evidence:

- `mtp-qkv-verifier-automatic-release-timing-1794787247863027011/receipt.json`
  and `analysis.json`:84 cases/2,520 cells/78,120 pairs. All1,415 native cells
  win at the median, range1.00272–2.41833×, geometric mean1.32030×;
  1,368 win every pair. The qualification margin is not a promised minimum
  gain on a future run. All byte and actual-engine checks pass.
- `mtp-qkv-verifier-automatic-feature-1794787670469330226/receipt.json`:
  all five registered Automatic Integration replays/86 cases pass.
- `qkv-bundle-host-r3-receipt.json`: three focused Unit registrations plus
  the two explicit replay aliases pass, including all-member serial policy
  mismatch and seven observed native crossover exclusions.
- `mtp-qkv-verifier-release-replay-1794787235282221493/receipt.json`:
  all eight exact Q4_0/M16 native profile owners pass.
- Frozen final producer:
  `mtp-qkv-verifier-automatic-snapshot-1794787898584237310/snapshot.json`,
  three records/42 files, SHA256
  `11c2cf5f8bc09f6ccc0c245eed1b1a21ccd9d9637099f0556daa8a78173279e6`.

Independent review recomputed every final cell from its31 raw pairs and joined
all1,217 compiled keys to the generation record; no admission or scope-leakage
blocker was found. The ten affected legacy registrations pass in
`mtp-qkv-verifier-affected-feature-1794788265068151540/receipt.json`,186.70s;
LocalTP output replay uses gfx1100 plus a second real ROCm participant. Four
synthetic-source registrations now explicitly declare `NO_MODELS`; their prior
unnecessary model fixture caused a collector inventory mismatch before execution.
The final host/Automatic/affected gates retain their exact source versions in
`mtp-qkv-verifier-feature-snapshot-1794789016013468232/snapshot-versions.json`,
SHA256 `32c1670de4de9b141a1419b2b86d57e117220851bf1569ff1304b6858f41622d`.
All64 exact-producer geometry profiles pass as described below. Broader model
execution is outside this component evidence.

## Native resource reuse and geometry profiling

QKV observes one native variant, `OrderedPartial` with request-work disabled.
The complete1,002,224-byte loaded ELF from the earlier21-source/eight-metric
verifier sweep is byte-identical inside the corrected QKV core. Its168 profiles
therefore retain their ISA/resource meaning:80 VGPRs,7,040 LDS bytes, wave64,
zero private storage and memory spills. The corrected-core audit is
`qkv-native-profile-reuse-r2-audit.json`, SHA256
`8cdd099d20f3d2077bd02abb6898707b61d8d4bd19249fb8a63071230f2b7063`.
Those scalar-operation counters do not describe QKV's unequal-width grids.

Eight Q4_0 profile owners cover the four geometries and both projection sets at
M16, all independently known native keys. Each captures one complete operation
without timing events, retains the independent M1 oracle, and authenticates the
full native packet's role mask and grid. Setup, warmup, poison and observation
remain outside a scoped ROCTx region. Native counters bind that same executable
and core to fresh unprofiled timing; they remain separate diagnostic evidence.

The final64-pass sweep (eight Q4_0 geometry/operation anchors × eight metrics)
is complete and independently audited:
`qkv-automatic-profile-batch-1794788533282501321/progress.json`, SHA256
`a61ca255ba2579534e1fd5d275f997b8f7834fe72b7274252918cada793ce7d8`.
The final audit is `qkv-automatic-profile-final-audit.json`, SHA256
`bfb74f1237e08483ef8abf5df1bc43e8b1ad5e52326e621c6b07edd216498209`.
All256 raw artifact hashes,64 receipt hashes, original timing/profile-owner
bindings, frozen source/core/executable, native device and exact grids pass.
Every process retires and restores ordinary `AUTO` power before the next job.

The native producer remains80 logical VGPRs,7,040 logical LDS bytes, wave64,
zero private storage and zero memory spills. All loaded code objects retain
SHA256 `3cb215279203a6e5781dac965660e7fca1e5824f1ffaa926201cb4a4d84c896a`.
Across the eight native grids, GPUBusy is100%, MemUnitBusy15.72–59.35%,
L2CacheHit22.04–37.49%, FETCH_SIZE1,188.875–40,423.75 KiB,
Wavefronts512–7,168, SDK LDS bank-conflict ratio2.454%, average issued
VALU instructions per wave445–929.5 and OccupancyPercent4.92–39.11%. These are separate diagnostic counters, not
utilization of theoretical INT8 peak or complete-operation timing. Small KV
packets have lower occupancy while retaining their measured latency advantage.
The fresh geometry counters cover Q4_0; all21 formats retain their independent
byte/timing coverage and the earlier byte-identical ISA/resource evidence.
