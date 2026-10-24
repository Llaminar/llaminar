# QFN small-group prefix scan trial 179

## Scope and current disposition

Candidate179 passes scoped production qualification on 2026-10-24. The matched
TP4 model result is **819.648 prefill / 40.656 decode tok/s**, a **0.897% decode
gain** over the bracketed frozen177 control with prefill unchanged. The
Castagna target remains **42.959 decode tok/s**. Component measurements below
are separate from that measured whole-model gain.

The retained model trace identifies 48 legacy grouping calls per decode step
per participant, reached from `MoEProjectionPipeline::Phase::ImportDown` through
`prepareExpertGroupsAsync`. QFN has 512 experts and selects ten routes per
token. This grouping and the earlier runtime grouping publish different owned
values; neither can be removed merely because both group routes.

The legacy kernel counts routes in parallel, but each expert then traverses
all preceding expert counts to form its offset and active rank. At 512 experts
that entails 130,816 logical preceding-count visits. The candidate reuses the
existing wave/CTA integer prefix helper. It retains initialization, stable
route order, sentinels and bitwise FP32 weight copies, including valid routes
with positive or negative zero weights. Inactive expert lanes contribute zero
and still participate in every scan barrier. Integer sums are bounded by 64,
so the change cannot round or overflow. No admission, retained workspace,
transfer, allocation or route-state authority is added.

## Initial ROCm evidence

Release, gfx906 MI50 device 0, physical CPUs 28–55, exact retained tensor and
stream owners. Each paired event sample covers 32 complete grouping operations
in one retained graph, following five paired warmups. All twelve alternating
samples are retained. Transfers, native inspection and serial-oracle comparisons
run outside timing.

| Geometry | Original µs | Scan µs | Speedup |
|---|---:|---:|---:|
| E512, M1, top-k10: actual QFN decode |13.2701|4.40253|3.014×|
| E512, M8, top-k8: 64-route diagnostic |14.8351|5.95003|2.493×|

Both cases pass captured full→invalid→partial→full replay against an independent
serial oracle. Checks cover every integer publication, copied weights and
eight poison words after each plane. Every native graph node authenticates the
original nine pointers, four scalars, function, launch geometry and complete
serial dependency chain. The original compiled kernel is byte-identical to the
frozen checkpoint177 control, including resource and descriptor controls.

Eight separate rocprof launches measure one named kernel and one counter each
at the actual ten-route shape. Their instrumented durations are excluded from
the event comparison.

| Counter/resource | Original | Scan |
|---|---:|---:|
| Wavefronts per launch |8|8|
| VALU instructions per wave |1015.25|119.875|
| FetchSize, KiB |0.375|0.375|
| LDS bank conflicts |0%|0%|
| Encoded allocated VGPRs |32|20|
| Static LDS, bytes |2304|528|
| Private storage/spills |0|0|

The scan cuts SQ vector instruction issues by approximately **88.2%** with
unchanged measured fetched bytes. It has one more static barrier, so fewer
barriers do not explain this win. These counters do not measure achieved
occupancy. Actual loaded device code, extracted from its bounded profiler URI,
matches the inspected compiled object; the core and model payloads are not
hashed.

The local receipts are `parity-results/qwen-flash-next/smallgroup179-slots10-*`
and `smallgroup179-slots64-*`. Compiler, loaded-code and frozen-control binding
records live in the corresponding `/tmp/qfn-smallgroup179-*` diagnostic packets.
These are local evidence, not a published corpus or complete preflight result.

## Remaining qualification

The private wider matrices pass **5,354 ROCm** and **1,973 CUDA** cells. They
cover every admitted M/top-k combination at expert-count boundaries, full and
truncated active-list capacities, and the actual ten-route shape at every
expert count from 1 through 512. CUDA retains its M≤4 selector; ROCm admits
M×top-k≤64. Eight dirty patterns and poisoned input/output tails retain exact
publication semantics.

On RTX 3090, the ten-route operation falls **7.968→2.528 µs**, and M4/top-k16
falls **8.864→3.712 µs**, with exact bytes. SM80/86/89/90 compilation reports
zero stack/local storage. Native SM86 attributes match the compiled resources:
31 registers/2,304 B shared for the original, 34 registers/520 B for the scan.
CUDA profiler attachment remains suspended under the recorded driver-assertion
stop rule; these are unprofiled event and compiler-resource results.

Small expert counts need different scheduling. ROCm E32 is approximately flat
and E33 wins across one, ten and 64 routes. CUDA E8 is approximately flat and
E9 wins across those route extents. Production therefore uses independent
prefixes through **E32 on ROCm** and **E8 on CUDA**, and cooperative scanning
above those boundaries. Two compile-time specializations share initialization
and stable scatter; each owns only the scratch required by its policy. The
thresholds depend on expert extent, not model names or particular row buckets.

The trial entrypoints and fixtures are retired. Shared production regressions
are explicitly registered as
`V2_Integration_{CUDA,ROCm}_MoESmallGroupPrefixExactPublications`. The final
CUDA matrix adds its E7/8/9 policy boundaries for **2,354 cells**; ROCm retains
5,354. Release performance aliases compile the same fixture sources against
the Release core, avoiding a mixed snapshot ABI.

The final Release and Integration builds pass for CUDA SM80/86/89/90 and HIP
gfx906. The production functional sweeps pass all **2,354 CUDA / 5,354 ROCm**
cells. Final production event measurements, collected separately from the
earlier alternating comparison, are:

| Backend | E / M / top-k / active capacity | Selected prefix | Complete operation µs |
|---|---|---|---:|
| ROCm |32 / 1 / 10 / 10|Independent|4.9725|
| ROCm |33 / 1 / 10 / 10|Scan|4.8775|
| ROCm |512 / 1 / 10 / 10|Scan|5.0300|
| ROCm |512 / 8 / 8 / 64|Scan|6.8350|
| CUDA |8 / 1 / 10 / 8|Independent|2.2560|
| CUDA |9 / 1 / 10 / 9|Scan|2.2400|
| CUDA |512 / 1 / 10 / 10|Scan|2.5600|
| CUDA |512 / 4 / 16 / 64|Scan|3.7280|

Every row passes dirty replay, native function/geometry inspection, and the
independent join to its final compiled resources. The final scan instruction
body is byte-identical to the profiled trial on gfx906 and all four CUDA
architectures. The smaller independent-prefix specializations change their
storage as intended: 384 B LDS on HIP and 288 B shared on CUDA, versus
528 B / 520 B for the scan. All variants have zero private storage and spills.
These fresh absolute timings are not a second paired speedup measurement.
The first complete Unit/affected run finishes **794/796 CTest entries passing**
in 1,050 seconds, with a clean driver window and retired native processes.
Both new scan registrations pass. The two broad MoE suites expose 17 CUDA and
14 ROCm case failures. ROCm then aborts during its cold prepared-Q8 capture
test: an unjoined producer event throws during recording, and storage unwinds
before native capture closes. Later ROCm cases are therefore untested in this
run. The repair batch includes:

- Existing CUDA fixtures omit explicit device-input placement or the required
  producer-event join before capture. Two BF16 comparisons also pass an
  unsupported BF16 gate to the CPU reference instead of decoding the actual
  stored BF16 values into its supported FP32 representation.
- One CUDA descriptor fixture reserves 4 KB for arrays that require 147,968 B
  of NativeVNNI payload and 18,496 B of scales. The repair derives capacities
  from the actual format and retains the host upload source through completion.
- One ROCm IQ3_S descriptor omits its required secondary source-nibble plane.
  Eleven serial-reference cases omit the canonical fused-projection partial
  workspace declaration already used by production.
- Three rebalance comparisons expose a real command-representation defect:
  four unnamed padding bytes are included in the 64-byte published command.
  ABI v17 gives those bytes an explicit zero-reserved field while preserving
  existing offsets and the full byte comparison. Both backends repeat poisoned
  command publications; a device-free host-publication test also checks all
  bytes. The two GPU regressions have explicit production-preflight entries.
- The cold ROCm router fixture now prejoins input events and uses a nested
  capture owner that closes recording before stream retirement or storage free.
- One capture test per backend requires a Release build without snapshots.
  Actual Release runs expose missing explicit stage streams in both fixtures.
  Their setup repairs bind the owning stream, declare the complete workspace
  and retain stage storage until completion; both must pass in Release before
  the corresponding Integration build-mode skip can be accounted for.

These failures do not authorize a numerical tolerance change or implicit
production transfers. The repaired CUDA fifteen-case and ROCm thirteen-case
selections pass. Five CUDA capture/lifetime cases each pass twenty repeats;
the formerly aborting ROCm cold-capture case also passes twenty repeats.
A subsequent complete ROCm diagnostic reaches all 172 cases: 171 pass and the
one snapshots-disabled case still requires its exact Release counterpart.
Both explicit GPU command-byte registrations pass. The host poisoned-byte
regression passes, while an older ABI-version assertion needed updating from
16 to 17.

The Release fixture audit also finds an unpublished resident-expert bank and
obsolete geometry/counter assertions. The fixtures now publish descriptors from
their existing prepared weights through the public runtime-table API, prepare
graph readiness after stream/workspace binding, and isolate reference counters
from candidate counters. Numerical tolerances remain unchanged. These existing
cases compare eager numerical execution and capture readiness; the new grouping
regressions separately prove actual retained capture and dirty replay.

Both final fixture builds and exact Release counterparts pass. The final
Unit/affected gate passes **798/798 CTest entries** in 1,331 seconds, including
all 778 Unit registrations and both full MoE suites. The two build-mode skips
are backed by those same-source Release runs; unchanged historical Unit
limitations remain explicit in the receipt. This is not a complete
ProductionTestPreflight certificate. Sources remain unchanged, the driver
window is clean, and all owned processes retire normally.

Fresh Release sweeps again pass all 2,354 CUDA and 5,354 ROCm cells. The final
ten-route event medians are **2.5585 µs on CUDA** and **5.029985 µs on ROCm**;
the eight geometry measurements agree with the preceding table. Fresh compiler
inspection proves the scan bodies still match the trial on gfx906 and all four
CUDA targets, with zero spills. Eight separate final ROCm counter runs observe:

| Final policy / expert extent | Waves | VALU instructions per wave | FetchSize KiB | LDS bank conflict % |
|---|---:|---:|---:|---:|
| Independent / 32 |4|73.250|0.375|0|
| Scan / 512 |8|119.875|0.375|0|

Each process profiles only one selected physical dispatch, occurrence 417 of
832 grouping calls. Before timing, eight dirty patterns pass; after timing,
the retained final pattern's complete bytes and guards are checked again.
The native graph schedule is 8 dirty replays, 5 warmups, 12 timed replays and
one terminal replay, with 32 operations per graph. FetchSize is the SDK's
video-memory-fetch metric and LDSBankConflict its stalled-time percentage;
these are not achieved occupancy or a bandwidth-efficiency ratio. Achieved
occupancy is unmeasured. Profile durations are excluded from event timings.

Receipts are `smallgroup179-production-unit-affected-r2`,
`smallgroup179-final-native-r3` and `smallgroup179-final-profiles-r3` under the
local QFN result directory. Canonical model accuracy and matched unprofiled
Release model measurements and the short final model dispatch trace now pass
as recorded below.

## Final model accuracy and matched throughput

The canonical ROCm full-TP4 panel passes its unchanged required groups. All
**185 final-logit rows across 26 files**, including reset and threshold probes,
are byte-identical to both checkpoint177 and checkpoint151. Mean KL is measured
against the same-GGUF independent reference, with frozen teacher histories:

| Group | Phase | Mean KL, nats | Top-1 agreement | Top-5 overlap |
|---|---|---:|---:|---:|
| Main, 16 prompts |Prefill|0.025614|93.75%|92.50%|
| Main, 16 prompts |Decode|0.028276|95.83%|92.29%|
| Held-out, 8 prompts |Prefill|0.014911|100%|95.00%|
| Held-out, 8 prompts |Decode|0.004553|100%|91.67%|

The threshold group remains diagnostic, with unchanged mean KL of 0.000168
prefill and 0.000210 decode. It is not silently promoted to an accepted group;
this selected-topology panel is not full-corpus or image certification.

The unprofiled Release comparison uses frozen177 → final179 → frozen177.
Each arm performs one warmup and three measured requests. Settings remain:
four MI50s in full TP4, sharded vocabulary, RAM ngrams, 28 physical NUMA1
workers, a fixed 512-token prompt, 256 emitted tokens, context capacity 4096,
greedy seed 42, FP16 KV, MTP and prefix caching disabled. Decode throughput
counts the 255 forwards after the prefill-produced token. Both arms explicitly
select FP16 attention Prefill8/Decode33; this change does not promote that
experimental precision policy into the production default. Runtime/core,
HIP, RCCL and GNU OpenMP identities are verified for every arm.

| Arm | Prefill tok/s, median | Decode tok/s, median | Decode sample range |
|---|---:|---:|---:|
| Frozen177, six bracketed samples |819.6652|40.2949|40.2358–40.4117|
| Final179, three samples |819.6480|40.6564|40.5864–40.8513|

Decode improves **0.897%**, with non-overlapping observed sample ranges;
prefill changes **−0.0021%**, effectively unchanged. All nine complete
256-token sequences match. Driver windows, resource observations and native
retirement pass. The **42.959 tok/s Castagna target remains open**; the
component speedup is larger than the measured whole-model improvement.

Model receipts are `smallgroup179-model-fp16-r1` and
`smallgroup179-model-throughput-{a1,b1,a2}-r1`. The matched command packet and
aggregate are `/tmp/qfn-smallgroup179-model-benchmark-r2/`. An earlier launcher
attempt rejected the wrong inherited CPU mask before starting native inference;
its failed packet remains separate, and the accepted runs use CPUs 28–55.

The separate four-token diagnostic proves the actual model graph uses the
compiled scan: **48 calls per model graph**, 24 model-graph executions across
four GPUs, **1,152 calls total**, 288 per GPU and 144 measured calls per GPU.
Each uses one 512-thread CTA, 528 B LDS and zero private storage. All 76 graph
launches join to their original correlation IDs and complete dispatch counts.
This short observation does not claim to trace every step of the 256-token
benchmark and contributes no throughput score.

The native trace exits normally with a clean driver window. Its original
postprocessor mistakenly retains the old 6,120-call attention obligation from
the longer diagnostic. A separate immutable offline adjudication derives the
correct 72 calls per attention role/GPU from two requests, three decode forwards
and twelve attention layers. Eight negative controls reject incorrect counts,
work, participants, driver state and surviving processes. The original failed
postprocessing receipt remains unchanged; the corrected graph join and scan
census are separate records in
`/tmp/qfn-smallgroup179-short4-adjudication-r1-private/`. No native rerun or
timing substitution is used.

The first CUDA trial capture exposed an unsupported diagnostic API in the
private launch wrapper. An empty-capture reproduction proves
`cudaStreamGetDevice` returns `cudaErrorStreamCaptureUnsupported` during global
capture, followed by `cudaErrorStreamCaptureInvalidated` at closure. Ownership
inspection remains before capture; removing the hot query makes both native
screens pass without changing kernel arithmetic. The failed receipt and
successful isolated reproduction are retained. A separate fixture amendment
closes abandoned ROCm captures before its outer retirement join.

The integer metadata proof applies independently of expert weight format;
existing all-format complete-pipeline coverage remains required.
