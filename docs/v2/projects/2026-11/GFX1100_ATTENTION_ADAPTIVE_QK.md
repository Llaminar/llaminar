# gfx1100 adaptive attention Q/K qualification

Automatic production selection now installs the measured adaptive query kernel
for the immutable domain below. The device selects Dot2 for short or ineligible
live work and WMMA for eligible prefixes from1K through64K. Fresh actual-Automatic
measurements pass the fixed-query linear-scaling objective. This does not imply
linear total-prefill work when query length also grows. The earlier explicit
candidate evidence remains separately identified below.

## Implementation and admission

`FlashAttentionQKEngine::ContextAdaptive` records one native query kernel with a
typed `FlashAttentionQKBranchPolicy::ContextAdaptive`. The kernel reads the
published device parameters once and makes a workgroup-uniform choice. WMMA is
selected for live KV1024–65536, retained stride65536, position offset `live−512`,
and linear storage. Short prefixes, partial queries and ring storage execute the
same ordered Dot2 arithmetic inside this retained kernel. Neither graph topology
nor host policy changes as live context grows or resets.

Explicit recording rejects unmeasured immutable geometry: it requires actual
gfx1100, FP16 KV, B1/M512, HD256/tile16×16, retained capacity65536, Q/KV heads12/2
or8/1, causal full context, no mask/window, no head offset or replicated GQA,
query scheduling and register-owned scores. The original unconditional WMMA
candidate still supports all native KV formats and both query/context schedules.
The adaptive kernel preserves the four ordered Q/K chains and existing softmax
and FP32 P/V. It adds no allocation, transfer, synchronization or extra launch.

## Correctness and timing ownership

Focused preflight registrations are:

- `V2_Integration_ROCmAdaptiveQKPolicyReplay`: device-free live-prefix predicate,
  threshold, malformed capacity/position and ring exclusions.
- `V2_Integration_ROCmFlashAttentionAdaptiveQKReplay`: native retained replay for
  both head geometries, 1023/1024/1025 and8192/8193 boundaries,64K, partial17/1
  queries, linear/unwrapped/wrapped storage, dirty reuse and reset/regrowth.
- `V2_Integration_ROCmFlashAttentionTimingReplay`: actual singleton timeline
  events around retained production batches. It alternates both engines through
  twenty short-prefix transitions plus64K/reset and rejects stale observations.

The adaptive symbol's enum value is authenticated separately from the final
WMMA template boolean. Thus an unconditional native kernel cannot accidentally
certify the adaptive candidate. Canonical context oracles explicitly record
Dot2, and the original all-format WMMA replay also passes with all three storage
layouts.

The Q/K timing owner uses `HIPGraphCapture` and `PerStepEvents`. It captures
one or sixteen complete parameter-producer/query transactions into a single
fragment, authenticates every attention symbol/geometry, and installs exactly
two timing events around that fragment. Short contexts below2048 use sixteen
operations; longer contexts use one. All batches are retained before sampling.
A sample launches its parent once, joins at an external terminal event, consumes
one named internal interval, and divides by its operation count. Host submission
gaps are outside that interval.

Every measured batch starts with poisoned output and parameter storage. Its
publication and output bytes are observed and compared with frozen independent
reference bytes before another sample or oracle can repair them. All poison,
readback and terminal observation work remains outside the internal interval.
Error cleanup drains asynchronous host observations before their storage retires;
a failed cleanup drain is fatal. Graph parents retire before their fragments,
workspace and buffers.

Initial external-event samples are retained in `attention-adaptive-initial/`.
They measured complete graph submissions but included host submission gaps and
cannot establish a small short-context regression. The first internal epoch is
retained separately in `attention-adaptive-internal-epoch0/`; its trainer predates
the final error-cleanup change and is never rebound to the final receipt.

## Earlier explicit-candidate Release measurements

Actual gfx1100 is PCI `0000:b5:00.0`, physical HIP5, visible0. The final run keeps
M512, HD256, FP16, capacity65536 and head mapping fixed. It has26 lifecycle cells
(two geometries ×13 prefixes),31 alternating paired samples per cell, byte checks
on every measured sample and unchanged producer hashes. Counter collection is
separate and restores AUTO.

| Geometry | Dot2 at64K | Adaptive at64K | Median ratio | Paired median95% interval | Adaptive32K→64K | Affine R² |
|---|---:|---:|---:|---:|---:|---:|
| Dense Q12/KV2 |46.216ms|39.908ms|1.1581×|1.1465–1.1585|1.9892×|0.999931|
| MoE Q8/KV1 |34.855ms|30.641ms|1.1375×|1.1334–1.1400|1.9870×|0.999967|

`V2_Perf_ROCmFlashAttentionAdaptivePrefixScaling` requires affine R²≥0.995 and
last doubling≤2.2 across all seven ascending powers of two from1024 through65536.
Threshold1023/1025, non-power8193, reset512 and regrowth1024 remain separate cells.
The older unconditional WMMA scaling owner now shares this internal timing
implementation; its previous archived measurements retain their original scope.

Short contexts are close to Dot2, but this is **not a demonstrated universal
win**. Dense512/1023 median ratios are0.99784/0.99782, and reset512 is0.99599;
their paired intervals lie below parity. MoE short ratios span0.99962–1.00095,
with intervals crossing parity. The adaptive candidate removes the earlier
several-percent short-context loss, but a small dense regression remains.
At that explicit-candidate checkpoint, Automatic remained Dot2. These archived
measurements do not certify the later installed producer.

## Resources and isolated profiles

Release and Integration compile gfx906 and gfx1100 through the mandatory spill
guard. The actual gfx1100 adaptive kernel uses96 logical/allocated VGPRs,
50 SGPRs,16KiB dynamic LDS, wave64 and256 threads; private storage and scalar/vector
spills are zero. Unconditional WMMA uses92 VGPRs and Dot2 uses87. Runtime resource
queries report up to8 resident blocks per reported WGP; that is a capacity limit,
not achieved occupancy.

Separate512 and65536 profiles each select one exact attention dispatch in each
of seven counter passes. Both authenticate the same loaded native code object,
16 static FP16 WMMA instruction sites, the ordered Dot2 branch, zero scratch
instructions, and positive GPU activity and waves. A symbol containing WMMA does
not mean the512 replay executed that branch: the live device predicate selects
Dot2 there. No achieved WMMA utilization claim is inferred from static ISA.

| Live KV | GPU busy | Memory busy | L2 hit | Waves | LDS conflicts | VALU instructions |
|---|---:|---:|---:|---:|---:|---:|
|512|100%|25.1445%|86.7206%|1536|1.6491|14661|
|65536|100%|33.4027%|81.0087%|1536|19.8901|2787308.125|

Raw counter definitions and units belong to the selected ROCm SDK. Profile
durations do not supply the unprofiled latency labels above. Both profiler
receipts record original AUTO→STABLE_STD→AUTO.

## Explicit-candidate evidence

Local evidence root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence/`.

- `attention-adaptive-scaling-receipt.json`, `attention-adaptive-fixed-query-scaling.log`
  and `attention-adaptive-scaling-summary.json` bind all26 cells and806 paired samples.
- `attention-adaptive-{512,65536}-profile/` contains raw per-pass counters,
  loaded code objects, ISA, metadata and summaries. Matching power receipts
  authenticate restoration and unchanged producers.
- `attention-adaptive-native-replay.log`, `attention-adaptive-policy-replay.log`,
  `attention-adaptive-internal-final-replay.log` and
  `attention-adaptive-existing-feature-suite.log` hold the focused/feature gates.
- `attention-adaptive-resource-build.log`,
  `attention-adaptive-internal-{release,integration}-final2-build.log` retain
  guarded backend compilation and final trainer build evidence.

Final trainer SHA256:
`e4f9ffd4ab4fdb68698b068fd33727be8f24a80fb49b285efac0b884960af769`.
Loaded core SHA256:
`9d6ade3488cccbe1da190f5d7c21d70949b4fe428f35db0e8b1ba98f1a569444`.
The receipt additionally binds kernel, policy and performance-owner source bytes.

## Installed Automatic selection

`resolveFlashAttentionQKEngine` receives the actual immutable launch geometry,
resolved tile, native ISA, device-parameter authority and guarded-context limit.
Automatic selects ContextAdaptive only for the measured domain above, including
query/register-score ownership and an unguarded device publication. Other valid
geometries select ordered Dot2. Explicit unsupported native requests fail.

The resolved native function is embedded in the retained graph. Existing
engine-local graph ownership scopes model heads, native KV representation,
capacity, visibility and placement; bucket/device/domain/participant and workspace
generation complete the cache identity. No new host live-count cache is created.
A reset changes device parameters, not the topology or captured function.

`gpu_graph_inventory.rocm_fa2_qk_kernel_submissions` records successful host
submissions with resolved engine, branch policy, native ISA, head mapping and
immutable geometry. It does not claim that every adaptive execution uses WMMA.
Retained replay does not increment this host counter or download a live count.
Collection is gated by the exact record family; the test observation scope
restores both environment values and reloads the collector policy on every exit.

New focused preflight registrations are:

- `V2_Integration_ROCmAutomaticQKPolicyReplay`: M1–2048 totality for both head
  layouts plus independent immutable geometry, ISA, format, tile, parameter and
  guarded-context exclusions; malformed requests fail closed.
- `V2_Integration_ROCmFlashAttentionAutomaticQKReplay`: actual Automatic capture,
  independently expected adaptive query and ordered context kernels, nested
  diagnostic-policy isolation, threshold/reset/ring/partial replay, neighboring
  immutable exclusions and truthful counters that remain frozen during replay.
- `V2_Integration_ROCmFlashAttentionAutomaticTimingReplay`: actual Automatic
  parent timing, dirty replay and stale-observation rejection.

The full launch-policy Unit test and both predicate replays passed in2.53s.
Six focused native registrations passed in168.66s: Automatic native76.60s,
Automatic timing6.06s, original WMMA8.20s, explicit adaptive16.29s, wide-ring5.03s
and register-score lifecycle55.60s. Release and Integration both compile gfx906
and gfx1100 with the mandatory spill guard. Full preflight was not run, as
requested; these are component and feature gates.

## Actual Automatic Release qualification

The existing measurements and profiles remain accepted under the user's scope;
isolation-only refresh is canceled. Original raw observations remain unchanged.

A new producer-bound run repeats26 lifecycle cells and806 paired samples using
actual Automatic selection, an explicit Dot2 reference, internal native events,
poison-before-sample and independent byte observation. Source and binary hashes
match the recorded producer.

| Geometry | Dot2 at64K | Automatic at64K | Median ratio | Paired median95% interval | Automatic32K→64K | Affine R² |
|---|---:|---:|---:|---:|---:|---:|
| Dense Q12/KV2 |47.073ms|40.747ms|1.1553×|1.1464–1.1607|1.9982×|0.999941|
| MoE Q8/KV1 |35.571ms|31.269ms|1.1376×|1.1297–1.1422|1.9934×|0.999981|

`V2_Perf_ROCmFlashAttentionAutomaticPrefixScaling` enforces the same R²≥0.995
and final-doubling≤2.2 requirements. All measured eligible long-prefix cells
win. The device rule preserves near-Dot2 short-context cost, rather than proving
zero dispatch overhead: dense512/reset ratios are0.99962/0.99986 with intervals
crossing parity; dense1023 is0.99728 with its paired interval below parity.
MoE short cells span0.99936–1.00124 with intervals crossing parity. This small
short-prefix cost is explicit; no universal per-size speedup is claimed.

Fresh actual-Automatic512 and65536 profiles each isolate one dispatch across
seven counter passes and authenticate the same loaded gfx1100 object and PCI/UUID.
Resources remain96VGPR/50SGPR,16KiB dynamic LDS,wave64,256threads,zero private
storage/spills/scratch,16 static WMMA sites and the ordered Dot2 branch. Both have
positive activity and1536 waves. Memory activity is25.40%/33.29%, L2 hit86.65%/79.19%,
and LDS conflicts1.6491/19.8901. Both restore AUTO.
The512 kernel selects Dot2; static WMMA sites are not executed-branch counters.

Current local evidence is `attention-automatic-scaling-{receipt,summary}.json`,
`attention-automatic-fixed-query-scaling.log`,
`attention-automatic-{512,65536}-profile/summary.json` and matching power receipts.
`attention-automatic-policy-replay.log`, `attention-automatic-feature-suite.log`
and `attention-automatic-{integration-build2,release-build}.log` hold the gates.
Trainer SHA256 is
`7b6879f18d69746665312e1afb79698bb62e181170e541b3e8bbddbc621ec0e2`;
core SHA256 is
`80cb76b9fc7e6cca1f377391b457089e1ddab1b3faa9b42fe657da6745c79ffe`.
Broader M, formats, capacities and head mappings require their own measured
admission evidence; the current selector explicitly retains Dot2 there.
