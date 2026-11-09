# QFN NativeQuantised exact-overlay qualification

## Intended outcome

Refresh a declared subset of the shared production exact-overlay surface,
prioritizing QFN physical TP shards, serial decode, grouped MTP and expert
operations. Ordinary dense/BackendNative prefill collection is parked. Use actual production
operations and keep existing runtime generic behavior for unmeasured keys.
Generic fitting is retired by user direction. Rename the family to
NativeQuantised throughout code, tooling and documentation; CPU instruction-set
VNNI terminology remains accurate and unchanged.

The active model target remains 500 prefill / 40 decode tok/s for QFN on two
MI50s plus two real CPU socket ranks. The latest matched local Release result is
278.611720 / 38.800220 tok/s (38.648656 decode excluding the first prefill token)
with canonical passive counters, four/two/one same-panel CPU scheduling and
large replicated shared gathers. Versus the qualified two-pair267.049428 /
38.476294 control this is +4.33% prefill and +0.84% decode. All768 output IDs,
MTP work and initial expert selection match;
native/driver retirement and resource observation pass. The500/40 target remains
open. Detailed evidence is `/tmp/qfn-four-pair-model-r1115/`; the preceding
two-pair/shared-gather gain is `/tmp/qfn-cpu-locality-shared-prefill-r1111/`.

The preceding CPU initialization/direct-combine change measured 217.105241 /
38.502293 against 218.074697 / 38.437912 (-0.44% prefill, +0.17% decode), also
effectively unchanged. Its observer completed 225 host samples and 1117 process
memory observations. These changes remove unnecessary work and correct graph
ordering but do not establish a substantial model speedup. The expert point
table remains uninstalled. The earlier detailed-counter run was 197.243395 /
30.281351; do not use its extra instrumentation to manufacture a kernel gain.

A subsequent same-binary control uses the existing canonical passive benchmark
counter filter. It completes at 218.074697 prefill / 38.437912 decode tok/s
(38.287763 after the first prefill token), with all 768 generated IDs and actual
MTP work identical. This measures diagnostic-counter overhead, not an engine
kernel gain: throughput is 10.6%/26.9% higher with the narrower observations.
Native inference and driver retirement pass; the adapter remains red because
its final resource sample encounters missing process memory fields during exit.
The original observation journal remains red. The observer now recognizes a
same-lifetime Linux PF_EXITING witness when exit_mm removes memory fields before
the process becomes a zombie. Missing live-process fields still fail; no zero
memory values, retries or discarded errors are introduced. All 17 observer Unit
cases and the explicit preflight regression pass. Evidence and comparison
limits: `/tmp/qfn-passive-overhead-r1093/comparison.json` and
`/tmp/qfn-owned-process-exit-state-r1094/`.

The current Release candidate removes inactive-capacity CPU descriptor/prefix/LUT
initialization and combines the first two CPU participants directly into the
root result, preserving the existing vec_add operand and reduction order.
Six all-format AVX2/AVX512 lifecycle/completion tests pass in both Integration
and Release. The direct-pair change passes the scalar/AVX2/AVX512 regressions,
all-format CPU expert checks and real two-rank MPI controls. These are
correctness/economy improvements, not measured model wins. The public 131 KB plan
clear remains; no shared planning rewrite is included.

An explicit diagnostic selects ordinary-prefill mapped sparse transport without
changing decode/MTP transport or actual peer capabilities. The pure selection
tests and native CUDA/ROCm 20-epoch sparse replay pass. Both repaired
graph-lowering fixtures now pass after removing an incorrect device-free label
and binding their original model-owned epoch admission. The existing all-native
control also passes through the MPI lifetime owner. These fixture-only fixes
retain production admission checks; the default transport is unchanged. Native rooted
reduction still communicates unowned route placeholders; the diagnostic does
not certify that existing path's communication economy. Source audit confirms
the persistent CPU worker already consumes the early routing ticket concurrently
with GPU expert work; aggregate waits do not establish avoidable serialization.
The new matched experiment is `/tmp/qfn-cpu-live-prefill-transport-r1098/`;
it retains the original passive controls and requires separate admission and
actual startup-policy proof for each transport arm.

That comparison completes with identical 768 output IDs, MTP work and initial
expert selection on both CPU ranks. Mapped prefill achieves 213.958626 prefill /
39.333688 decode tok/s, versus native's 217.105241 /38.502293. Prefill is 1.45%
lower, so this run does not justify promoting mapped transport on this P2P
topology. Decode's policy is unchanged; its 2.16% difference is not evidence of
a decode transport optimization. Both original lifetimes retire with clean
driver/resource observations, and their owned disk-cache payloads are removed.
The normal-policy result remains the current baseline. Original samples and
scope limits are in `transport-comparison.json` under that experiment directory.

A subsequent candidate advances the existing ordinary-prefill CPU routing
ticket ahead of independent shared GPU FFN work. Explicit dependencies retain
the original shared-before-routed GPU computation and collective sequence;
CPU result ingress and the final join remain unchanged. The original core
fails the new order/dependency checks on both CUDA and ROCm, with no collective
sequence failure. The revised Release core passes eight focused registrations.
The actual graph tests cover both TP participants under native/native,
mapped/mapped and mapped/native policies, plus original serial and no-CPU MTP
controls. Full context admission and the matched model run pass. The latter
achieves 217.618721 prefill / 39.048121 decode tok/s, with all 768 output IDs, MTP
work and initial expert placement identical to the native-transport control.
The resource observer completes 223 host samples/1107 memory observations;
original processes and driver observations retire cleanly. Owned cache payloads
are removed. Prefill changes by only +0.24%, so the source ordering improvement
does not demonstrate useful additional overlap on this workload.
The applied patch, original failure, focused pass and fixed workload are in
`/tmp/qfn-prefill-early-cpu-ticket-order-r1100/`. The next step is current,
selective phase attribution of expert compute and peer readiness; previously
measured all-domain scopes cannot establish the passive run's critical path.

The first selective diagnostic adds only `retained_parent_transaction` and
`moe_overlay_endpoint` to the passive filter, keeping kernel histograms and all
CPU/GPU stage timers disabled. It completes with identical output/MTP work and
clean resource/driver retirement. Throughput is 221.351625 prefill / 37.818064
decode: +1.72%/-3.15% versus the same binary's passive control. The prefill
variation does not establish negative overhead or a new speedup; diagnostic
rates are not the performance baseline.

Root records contain 288 prefill layer calls: three measured requests × two
chunks × 48 layers. Their observed scopes per 512-token prefill are 817.718 ms
compute submission, 688.903 ms ticket-publication wait, 440.224 ms rooted gather,
79.592 ms fixed-order combine and 45.195 ms preweight/publication. These are
overlapping/nested host observations, not additive exclusive GPU/CPU costs.
Gather includes peer readiness, and compute submission is not isolated FFN
arithmetic. Follower records contain 480 calls from a different process-lifetime
cohort; do not divide those totals by the root's three requests. Qualified-name
timing filters restrict export but still enable sibling clocks/records in the
same domain. Existing counters suffice for this diagnostic; no new tracing
framework is added. Evidence: `/tmp/qfn-passive-phase-attribution-r1101/`.
Next: distinguish readiness from actual gather transfer cost and benchmark the
existing CPU communication regimes only if that evidence warrants it. Completed
kernel shapes remain retained, with no broad resweep prerequisite.

A second same-binary diagnostic adds the existing follower per-layer timing
and live packed-record byte counters. It passes all 768 output IDs and MTP
counts, native/driver retirement, and 222 host/1100 memory observations.
Its 220.511 prefill /37.192 decode tok/s are diagnostic observations, not a new
performance baseline. The owned disk-cache payload is removed after retirement.
Evidence: `/tmp/qfn-follower-readiness-r1103/`.

The root's 288 measured gathers contain 5,060,484,780 local payload bytes,
or 1.687 GB per measured prefill. Root gather averages 440.486 ms per request,
including peer readiness. Dividing those quantities gives about 3.83 GB/s;
this is a workload ratio, not isolated link bandwidth. Follower local execution,
joint publication and dispatch-wait totals are 4.308, 2.743 and 4.303 seconds
across its separate 480-call cohort. Its byte counters include 12 additional
calls with no endpoint-timing join, so those bytes are excluded from the root
comparison. Root resets only its own collector after setup and warmup; current
aggregated timers have no request/ticket timestamp association, so measured-only
follower durations cannot be recovered by subtracting aggregates.

Retained ready-data tests already show MPI beating the original serial
SharedSpin gather, and CMA improving isolated MPI copies without a demonstrated
matched-model gain. Do not repeat or promote those old comparisons as new work.
The concrete next candidate parallelizes the existing shared backend's two
live-copy steps using its existing workshare pattern, preserving original
NUMA/PMA buffers and ready/consumed epochs. The production protocol selector
stays unchanged pending correctness and matched economy evidence. A separate,
bounded CPU complete-FFN profile will check activation cost; the older dot-only
profile cannot answer that question. No all-shape resweep is required.

The parallel-copy candidate passes the focused degree-2/4/8 protocol tests and
the actual two-socket NUMA replay. At 28 physical workers per socket, its first
ready-data sample gives 954.642 microseconds versus MPI's 2762.948 at about
16 MiB per participant, and 242.943 versus 665.783 at about 4 MiB. It loses at
1 MiB (250.130 versus 179.168) and near the 256 KiB workshare threshold. The
root's aliased local payload is not copied. These are component medians from
one shared-first lifetime, not a full-model win or a universal selector rule.
The existing selector is unchanged. Evidence:
`/tmp/qfn-shmem-rooted-gather-parallel-copy-r1104/`.

Review also found a pre-existing abort race: a normal epoch store could replace
the terminal abort marker, and peer readiness could hide local cancellation.
The focused original-body control fails all 20 repeated publication/wait
witnesses as expected. The corrected source centralizes normal epoch advance
in one abort-absorbing CAS and checks local cancellation before peer success.
The coherent corrected Release build passes that replay and the healthy
degree-2/4/8 and NUMA protocols. The original-body negative control has been
retired. Corrected-closure component timing also passes and confirms the large
copy benefit below. Evidence:
`/tmp/qfn-shmem-absorbing-abort-r1105/`.

The existing CPU complete-FFN performance owner now carries an explicit typed
`decode|mtp|prefill` purpose into its grouped descriptors and reports the actual
policy widths. Its parser contract and native byte checks for all three purposes
pass; the serial oracle remains Decode. No production arithmetic is changed.
A bounded prefill profile uses Q4_K gate/up and Q5_1 down, physical N320/K2560
and N2560/K320, 84 authored experts/519 rows, 28 workers on socket 0 and 2,000
persistent-team transactions. This balanced row vector is a mechanism control,
not a reconstruction of the model's actual routing trace.

All phase and persistent-transaction byte oracles pass. Of 191,745 sampled
user-cycle events, nibble and non-nibble pair kernels account for 42.080% and
40.634% of period-weighted cycles, OpenMP runtime 8.931%, and named SwiGLU
routines 0.321%. Setup/oracle work is included (named random-data setup 1.483%);
these are not exclusive model phase timings. Sampling observes 28 native TIDs;
the recording lacks CPU IDs, while the launch pins the observed socket-0
physical-core set 0–27. This does not justify SwiGLU tuning. Inspect the measured
down-pair loop instead. The exact unsigned coefficient domain permits a
candidate signed-activation/unsigned-weight integer dot without bias correction;
its first three object variants spill vector state and are rejected. The fourth
specializes the immutable numerical policy while preserving the original FP
operations. Both the linked fixture and core have no hot vector spills or
ordinary hot helper calls; retained signed/AVX2 object bodies are unchanged.
All-format byte/lifecycle and complete-owner tests pass on AVX2 and AVX-512.
The first new fixture incorrectly supplied an empty source slice; correcting
it to the intended nonzero half-open interval resolves the setup failure.
Unprofiled paired economy completes with the limited leaf gain described below.
Object-only pruning avoided repeating full builds and model runs for rejected
variants. Candidate evidence: `/tmp/qfn-cpu-unsigned-weight-dot-r1108/`.

The unsigned-dot paired complete-FFN experiment retains two baseline and two
candidate lifetimes in reversed order. On authored E84/R519 prefill with 28
physical workers, aligned down latency falls 2681.205 to 2626.390 microseconds
(2.04%), but the persistent complete operation is effectively flat: 5860.715
to 5849.985 microseconds. This is a small leaf win, not a large model gain.

The next candidate gives one worker up to two adjacent pair/tail calls on one
prepared N64 panel, only for ordinary-prefill GPUAlignedExpert Pairwise work
with at least four jobs per actual worker. Existing pair arithmetic, K reduction
order and publication boundaries remain unchanged. Across reversed 2+2 native
lifetimes, the same authored complete-FFN control measures:

| Aligned operation | Dot-only control, microseconds | Same-panel candidate, microseconds | Speedup |
|---|---:|---:|---:|
| Gate/up | 2795.825 | 1880.475 | 1.487× |
| Down | 2639.380 | 1329.190 | 1.986× |
| Complete persistent FFN | 5883.985 | 3630.580 | 1.621× |

The BackendNative control is effectively flat. An authored E24/R72 MTP control
across reversed 2+2 lifetimes has 0.83% higher complete-operation latency with
overlapping samples. All-format bytes, partial/reset/lifecycle behavior and the
complete existing owner pass on both runtime ISAs. The larger compiled wrapper
has no identified new hot accumulator spills; retained pair bodies are unchanged.
A separate 2,000-transaction profile passes its serial oracle. These observations
qualify the component experiment, not model throughput or all-shape economy.
Evidence: `/tmp/qfn-cpu-same-panel-locality-next/`.

The final corrected-CAS shared-copy component confirms 957.609 versus MPI's
2878.487 microseconds near 16 MiB. A new private typed CPU-shard prefill entry
selects this path only when each replicated remote member has at least one
full 4 MiB live chunk. Original source/packet authority proves identical member
counts, so selection needs no new exchange. Generic variable counts and
decode/MTP keep their prior selection. Twelve focused registrations pass,
including degree-2/4/8 protocols, resolver/selection and all-format CPU owners.
Real-model admission at262144 tokens and diagnostic/passive comparisons pass at
`/tmp/qfn-cpu-locality-shared-prefill-r1111/`. With the same diagnostic filters,
prefill increases220.511→269.455tok/s; decode is effectively flat37.192→37.124.
The root uses shared parallel gather for247/288 prefill calls and MPI for41.
Retained-parent gather totals fall1321.459→314.904ms over three requests
(440.486→104.968ms/request), including peer readiness. Endpoint compute/packet
timers contain288 baseline versus287 current calls; those aggregates do not
form an exactly matched layer cohort and must not be normalized as one.

The passive comparison gives the267.049/38.476 rates reported above. Both runs
preserve all768 IDs, actual MTP work and initial expert placement. Dynamic
balancing can react to changed service costs; recorded live bytes are retained
rather than assumed identical. Both native lifetimes retire all GPU contexts
and pass driver observation. Diagnostic/passive observers complete224/222 host
samples and1112/1102 process-memory observations respectively, with no observer
errors; these do not certify post-warm leak bounds. Both owned disk prefix
archives are removed with metadata-only receipts (about728MB each).
Evidence: `/tmp/qfn-cpu-ffn-purpose-r1105/`; the pre-candidate native closure is
retained at `/tmp/qfn-cpu-copy-dot-baseline-r1107/`. Perf's same-inode debug-cache
hardlink changes fixture ctime only; no rebuild or payload change occurred.

Bounded Release assembly comparison against the frozen original core confirms
five 8,208-byte prefix clears and the 4,104-byte width clear disappear, LUT clears use the
live descriptor count, and batched descriptors receive only live construction
without allocation or an array cookie. Selected dot/partial and vec_add bodies
retain their instruction, register and branch structure modulo relocation and
padding. The three original root-combine memmove sites disappear. The batched
wrapper grows from 6,123 to 8,994 code bytes because the compiler unrolls live defaults;
the matched model result above shows no material throughput change. Saved original/candidate
instructions and explicit inspection limits are in
`/tmp/qfn-cpu-live-prefix-isa-r1097/`.

## Current collection strategy

The original all-model campaign is stopped by user direction. Its completed
393 M1 and 30 grouped preparation batches remain retained; the interrupted
grouped batch is archived separately and is not complete. Do not restart the
remaining broad campaign as a prerequisite for the QFN performance target.

New sweep jobs exclude Qwen2.5 and Qwen3-only geometries. Qwen3.5, Qwen3.6 and
Qwen3.8 remain in scope, including a physical geometry shared with an older
family. This is collection selection, not removal of runtime support or existing
dispatch entries. Deduplicate by the complete physical execution key. Reuse
compatible completed measurements and collect only missing cells; incomplete
or incompatible format, ISA, arithmetic, partition or actual-M observations
cannot qualify another cell. A new screening strategy alone does not invalidate
an existing complete full-grade measurement.

This selection is now the default for new direct CPU and public GPU sweep
transactions. It retains 105 of 131 canonical geometries, removing 26 old-only
shapes, while saved historical plans keep their original scope. The full CPU
plan drops 1269→1113 timing batches. Actual retained artifacts validate all 315
modern M1 jobs as installed-policy authority and 11 modern completed grouped
cells as original raw measurements, without starting inference or profiling.
The other 19 completed grouped cells remain archived outside the modern scope.
The full modern plan is an inventory, not a prerequisite to the focused QFN
slice. GPU decode/MTP now also supports importing completed shapes from an
original compatible collection or superset, preserving original observations
and launching only missing shapes. Existing raw receipts cannot establish
cross-build equivalence or missing historical routed-MoE hardware identity;
those boundaries remain explicit. The implementation and device-free evidence
are in `/tmp/qfn-native-quantised-gpu-completed-reuse-r1084/`.
Six focused scope/reuse/public-CLI CTest registrations pass after integration.
The final ten-case GPU reuse suite includes positive CUDA/ROCm strong-reader
coverage for every source format and both execution modes, plus original
timing/replay-floor rejection controls. Its explicit Unit and production
preflight registrations pass; additive evidence is in
`/tmp/qfn-native-quantised-gpu-reuse-positive-r1088/`.

Sixteen actual focused CTest registrations pass, including new explicit
preflight scope/reuse tests and the complete CUDA/ROCm analyzer regressions.
Partial CUDA M1 publication now retains unrelated installed exact rows, and
ROCm completeness checks consume the declared scope. A failed local native run
cannot be hidden by importing an older successful result. Evidence is retained
in `/tmp/qfn-native-quantised-reuse-validation-r1085/`.

The first adaptive slice targets QFN TP2 intermediate-shard experts on AVX512
with28 physical workers, initially its actual gate/up Q4_K and down Q5_1 formats.
This is a complete two-cell declared scope; the broader28-batch all-format plan
remains separately incomplete. Its remaining26 batches must not recollect those
two completed cells. The shared role catalog supplies its two geometries;
there is no new private shape table. Compatible requests reuse source packing,
buffers and worker teams. All forceable candidates retain byte/repeat checks.
Cheap interleaved screening selects the installed production baseline and
credible contenders for the original full timing floor. Screening observations
are retained but cannot publish winners. Profile the selected winner and actual
baseline separately, not every losing candidate. Two-candidate cells benefit
mainly from setup/profile reuse and scope reduction, not elimination rounds.

The adaptive CPU owner is built in Release. Its selector regressions and four
registered workflow checks pass. Both selected native cells pass:420 original
candidate/held-out observations and111 isolated winner/baseline profiles. Native
fixture time is80.1seconds for gate/up and274.9seconds for down, separately from
profiling. The staged108-key exact-point table selects Pairwise for105 keys and
WideRows for three. It is not installed or model-qualified. The older broad-owner
M3/M4 speedup below used70 projections sharing one activation matrix; the new
source-faithful contests use six projections with distinct expert inputs.
The Pairwise wrappers also differ, so these are not binary-matched comparisons.
The old gain and current small-bundle loss do not transfer between those
contracts. A controlled descriptor-count comparison on the current owner remains
useful; no broad sweep is required for it. Production submits all active expert
descriptors with their actual mixed row counts. The current execution class
distinguishes filled from underfilled work, but not the total work in a filled
bundle. The current two-cell measurements therefore need a production-sized
holdout before installation. The bounded next experiment and source ownership
map are in `/tmp/qfn-cpu-bundle-workload-audit-r1088/NOTE.md`.
The current-owner M3 control confirms a large-bundle WideRows win: 70
projections take 224.6–228.2 us versus Pairwise's 388.9–417.7 us (1.70–1.86×).
Six-projection short-warmup results change with candidate order. Two additional
six-projection lifetimes with 1024 warmups settle at Pairwise 32.85–33.42 us and
WideRows 36.11–36.49 us, retaining 30 samples and eight repeats. These are
component measurements, not model throughput or an installed policy. Only the
challenged workload needs further qualification; the paid shape matrix remains
retained. Commands, original observations and uncertainty are in
`/tmp/qfn-cpu-bundle-width-r1089/NOTE.md` and
`/tmp/qfn-cpu-p6-interleaved-r1090/NOTE.md` (the latter actually uses full-v1
timing with reversed order, as its manifests record).
The subsequent authored mixed-M holdout keeps 35 experts and 487 routed rows,
changes only its 40 matching M3 gate/up descriptors, and leaves 30 M28/M29
descriptors under Auto. Both reversed-order lifetimes pass byte, placement and
layout checks with 1024 warmups, 30 samples and eight repeats. WideRows improves
whole-bundle latency over Auto by only 2.9–3.4%; Auto and forced Pairwise select
identical physical work but have materially different sequential timings. That
unresolved timing variation prevents a strong performance conclusion. No new
workload discriminator or dispatch policy is installed from this experiment.
This is an aggregate-anchored authored sensitivity test, not a recorded routing
replay. Evidence is in `/tmp/qfn-cpu-mixed-m3-r1092/`.
The current model also bounds the value of the staged three WideRows entries:
its primary gate/up verifier observes M1–4, so the staged M31 verifier entry
does not apply. Prefill M31 and M64 together cover less than 1% of primary
gate/up member observations on either CPU rank. Observation counts are not
kernel-time shares. These entries cannot explain a substantial decode gain.
The installed M1 table is retained correctly, but its measured full-K arithmetic
does not qualify the model's partitioned expert arithmetic. Current ROCm exact
tables do not contain the relevant QFN expert geometries. No matching qualified
key is proven to be bypassed; new work must target the actual missing execution
scope rather than repeating the unrelated completed M1 sweep.
Evidence is in `/tmp/qfn-tp2-expert-actual-formats-r1084/`.
Ordinary GPU dense-prefill collection remains parked. The retained ROCm
low-bit decode collector still supports full-grade collection; its larger
resource-lifetime refactor for adaptive screening is deferred explicitly.
Planned-job counts and measured collection time must be reported separately
from inference throughput improvements.

The exact-feature model baseline admitted262144 context but failed full
initialization before inference: its CPU-cohort authority rejected the declared
whole-GPU participant-balancing axis. The narrow complete-target recovery now
passes nine focused registrations, including the complete authority and owner-map
Unit suites and the additional whole-CPU peer combination; its Release rebuild
passes. The fresh model run gets through ownership initialization and then
rejects a separate malformed GPU/CPU graph cutpoint: the CPU service can appear
before the captured route-reduction producer. The declarative DAG now closes
after that producer, with the strict lifecycle check unchanged. Six focused
native CUDA/ROCm registrations pass against the exact Release core, including
single-device and LocalTP local/remote CPU service. The next model attempt passes
that boundary but rejects a follower retained composition containing a passive
ordered-reduce stage. The root parent materializes 4202 native nodes. Capture
elides passive identities only for ordinary graphs, while this composer accepts
typed envelopes; two stale exclusions in the composer disagreed with capture.
The narrow correction requires those original passive identities and preserves
the missing/reordered/foreign-stage checks. All eight focused CUDA/ROCm graph
registrations pass in 9.54 seconds against the full Release core. The repeated
model baseline completes all serving/MTP graph preparation, then aborts during
the first request reset at `reset_committed_kv_and_gdn`, before warmup inference.
The original exception was a QFN auxiliary launch reporting a pending native
error. A focused original-code test reproduced status 704: repeated direct-peer
setup accepted `PeerAccessAlreadyEnabled` without retiring that owned status.
Subsequent successful APIs left it pending for the next launch check. Setup now
retires only its exact expected status; unrelated errors remain visible and
reject setup. CUDA/ROCm expert-transfer and collective helper boundaries receive
the same correction. Native unavailable-peer operation remains supported.
Four explicit CUDA/ROCm preflight regressions pass in 5.30 seconds, including
20 partial reset/captured replays and unrelated-error preservation. The complete
ROCm peer-lane test class also passes after its old fixtures bind destination
execution lanes through TransferEngine. The matched model now passes admission,
warmup, all three 512-prefill/256-generation measurements and process retirement.
It achieves 197.243395 prefill / 30.281351 decode tok/s. Every measured request
accepts 192 of 192 dynamic-MTP draft tokens across 64 verifier runs. Dynamic
expert balancing, 262144 context, RAM ngrams, local GPU caches and 16/32 GiB
RAM/disk prefix budgets remain enabled. The resource observer completes with
234 host samples and 1162 memory observations without errors; this is not a
full application, leak, host-accounting or image certificate. Full preflight
remains skipped this iteration by request. The expert point table is still
uninstalled.
The 512-token workload executes 448 then 64 prefill rows. This is the existing
64-token reusable-prefix snapshot boundary, not a missing M512 graph or an
arbitrary smaller bucket limit: `PrefixStateSnapshot.cpp` reserves a reusable
boundary before the terminal token, and `Runner.cpp` harvests it before the
remaining chunk. M512 is already admitted. Raising the maximum bucket alone
would not remove that checkpoint or address the measured CPU expert cost.
Source and retained failure/green evidence live under
`/tmp/qfn-joint-gpu-balancing-recovery-r1080/` and
`/tmp/qfn-heterogeneous-ticket-cutpoint-r1082/`. Six focused build-selection
registrations also pass after repairing two stale authored CMake fixtures.
The retained-composition correction and focused evidence are in
`/tmp/qfn-retained-passive-stage-coverage-r1087/`.
The peer-status original RED, correction, native GREEN and model rerun are in
`/tmp/qfn-peer-error-retirement-r1091/`; the test-only lane binding repair is in
`/tmp/qfn-native-quantised-peer-lane-fixtures-r1092/`. Reset exceptions now retain
their original native status and distinguish errors already present before
submission from those first observed afterward, without clearing unknown errors
or adding synchronization.

## Implementation and qualification status

- Eleven missing canonical QFN physical shapes are added. The resolved shared
  production inventory contains 131 geometries; QFN owns 35 physical keys,
  including 31 ordinary-prefill keys. Existing release-owned aliases remain.
- TP1/2/4/8 source intervals are explicit. F640 with native32 alignment yields
  TP8 intermediate widths96/64, not80. Query-interval KV ownership is preserved.
- A typed expert-role catalog derives H/F from named full-shape records. Its58
  participant records resolve13 physical expert geometries and retain original
  source extents, owned intervals and local policy N.
- CPU expert prefill uses the union of existing CPU evidence anchors, grouped
  verifier anchors and current physical prefill buckets. There are39 current
  anchors; future bucket changes are consumed rather than hardcoded.
- CPU exact row identity retains phase, arithmetic/source/prepared formats,
  build/runtime ISA, workers, physical and policy geometry, K partition, output
  layout and execution class. MTP is no longer collapsed into ordinary decode
  in the expert stage's scheduling metadata.
- The actual fused fixture retains separate expert activation rows and mixed
  descriptor vectors. Compatible requests share prepared storage and the real
  worker team. The exact transaction stages M1, rebuilds, measures grouped rows,
  and requires actual Auto lookup hits after final installation.
- NativeQuantised naming is applied throughout kernels, classes, tooling and
  docs. Actual VNNI instruction/packing terminology and published data schema
  IDs retain their meaning. Historical raw traces remain byte-for-byte intact.
- Generic-fit/CV/seal commands are retired. The exact emitters retain existing
  generic runtime decisions and unrelated exact rows. The MoE GPU emitter now
  explicitly preserves installed rows outside the refreshed scope.
- Twenty-nine focused CTest registrations pass after the rename (17.1 seconds),
  including native AVX2/AVX512 replay and the new batch fixture. Additional
  workflow, historical-schema and retention aliases pass separately. This is
  focused qualification, not a full preflight certificate.
- All 50 affected Python Unit registrations pass across the original run and
  focused reruns. The remaining ROCm generator validator now uses supported
  reader/exact boundaries: incomplete observations can be read without
  generating a policy, while incomplete alias contests cannot be published.
  The reader regression belongs to the explicit preflight retention entry.
- Obsolete generic tree-search and profiler-surrogate Performance campaigns
  are retired along with their registrations. Independent scorer arithmetic
  replay diagnostics and historical-data readers remain available.
- CPU Release AVX2/AVX512 and the full Release CUDA 80/86/89/90 plus HIP
  gfx906/gfx1100 build pass with the normal spill guards enabled.
- Eight real timing-batch receipts and two isolated counter receipts pass their
  strict adapters. These low-sample, two-worker controls qualify the protocol,
  not dispatch winners. Each profile authenticates its actual descriptor vector
  and one armed operation, separately from canonical timing.
- The full 131-geometry /21-format /three-ISA CPU transaction is planned as
  393 M1, 393 grouped and 483 expert preparation batches. Its first M1 batch
  exposed an invalid IQ3_S coefficient in the synthetic prepared-weight fixture;
  the fixture now writes typed source metadata through the production codecs.
  Both AVX2 and AVX512 focused regressions pass, including repeated decode and
  rejection of deliberately reserved bits. The unchanged production validator
  correctly rejected the original malformed metadata. The fresh transaction
  passed that original batch before the broad campaign was stopped in favor of
  the focused collection strategy above.
- Generated production row overlay remains empty pending authenticated native
  winner collection. No handwritten shape policy has been installed.

## Fresh component comparison

Release native owner, same binary/core,28 physical workers, retained team,
GPUAlignedExpert arithmetic. Four fresh lifetimes run Auto, WideRows, WideRows,
Auto; eight samples, four repeats and four warmups. All40 cases pass byte,
placement and counter checks. Gate/up uses Q4_K N320/K2560; down uses Q5_1
N2560/K320. Timings are microseconds for the actual measured fused bundle.

| Operation | Stats | Auto | WideRows | Auto / WideRows |
|---|---|---:|---:|---:|
| Gate/up M3 | on | 637.958 | 352.336 | 1.811× |
| Gate/up M3 | off | 640.797 | 354.522 | 1.807× |
| Gate/up M4 | on | 725.820 | 510.255 | 1.422× |
| Gate/up M4 | off | 711.713 | 511.809 | 1.391× |
| Gate/up M8 | on | 1090.480 | 1024.741 | 1.064× |
| Gate/up M8 | off | 1078.048 | 975.768 | 1.105× |
| Gate/up M64 | on | 4197.097 | 3956.500 | 1.061× |
| Gate/up M64 | off | 4160.660 | 3850.590 | 1.081× |
| Down M64 | on | 1761.770 | 2243.320 | 0.785× |
| Down M64 | off | 1771.688 | 2219.628 | 0.798× |

These are component results from the retained local broad-tree Release owner,
not an exact-feature model certificate. They justify a measured operation-aware
overlay, not a blanket WideRows default or a predicted model gain. Later losing
three/four-row kernel experiments do not invalidate the unchanged original
WideRows implementation used here.

## Coverage and evidence boundaries

A normalized prepared32-block fixture can exercise a superblock format's
prepared arithmetic at smallK. That does not prove a native source tensor or
TP cut is valid for its256-element source block. The canonical legacy strong
trainer already distinguishes this preparation. Real QFN down Q5_1/Q8_0 uses
native32 blocks; source-faithful fused cases must reject incompatible source
geometry rather than pad or requantize it.

Complete measurement scope, native source eligibility, prepared arithmetic
coverage and installable runtime coverage are separate fields. Ordinary CPU
BackendNative prefill's missing exact installation boundary remains explicit;
CPU fused expert coverage does not silently certify it.

Exact refresh must retain old generic decisions and unrelated exact entries.
CPU M1 changes are staged and the native owner rebuilt before dependent grouped
collection. Compatible requests should reuse process, preparation and team
setup while retaining independent phase/M/candidate/held-out receipts.

## Local evidence

- `/tmp/qfn-cpu-narrow-k-tuning-r1055/row-policy-comparison.json` and four native
  lifetime reports: all40 component cases green.
- `/tmp/qfn-nativevnni-qfn-shape-inventory-r1058/`: original missing-geometry
  failures, canonical resolved inventory and focused green results.
- `/tmp/qfn-nativevnni-qfn-expert-projection-r1059/`: source-role helpers,
  source-alignment distinction and shared phase counts.
- `/tmp/qfn-cpu-exact-focused-r1061/native-tests.xml`:18 registrations green.
- `/tmp/qfn-cpu-exact-release-r1060/`: coherent CPU Release AVX512/AVX2 core
  and renamed native trainer builds.
- `/tmp/qfn-native-quantised-rename-r1062/qualification.json`: renamed build
  and focused native qualification, with the explicit scope limits above.
- The same directory's `affected-python-unit.xml`, `generator-repaired.xml`
  and `retirement-followup.xml`: original 49/50, repaired missing registration,
  and both subsequently changed Unit modules passing respectively.
- `/tmp/qfn-native-quantised-full-release-r1063/state.json`: all shipped GPU
  architecture compilation; the initial configure needed the existing RCCL
  source-build proof paths, then the complete build passed.
- `/tmp/qfn-cpu-narrow-k-tuning-r1055/native-smoke-batch/adapter-result.json`
  and `native-profile-smoke/native-profile-identity-result.json`: actual native
  batch and isolated-counter identity checks. The first profiler collector
  attempt rejected the new completion marker; its original failure is retained.
- `/tmp/qfn-native-quantised-exact-sweep-r1064/`: complete declared plan and
  original IQ3_S fixture failure. It is not a completed sweep.
- `/tmp/qfn-native-quantised-source-factors-r1065/focused.xml`: both native
  source-factor regressions pass; each is registered in production preflight.
- `/tmp/qfn-native-quantised-exact-sweep-r1066/`: fresh full transaction after
  repairing the fixture, now stopped.393 M1 and30 grouped batches are complete;
  no expert batch completed. The partial next batch is archived under its
  `interrupted/` directory, with process retirement evidence in
  `/tmp/qfn-native-quantised-priority-r1078/`.

Completion of the newly declared focused scope, actual emitted Auto hits,
isolated selected-candidate profiling, matched hybrid model and OpenCode gates
remain outstanding. Full all-model sweep completion is no longer a prerequisite.

## Bounded prefill row regimes

The user approved two large-M timing anchors to keep collection manageable:
M128 for the128..511 regime and M512 for512 and above. Check M256 and M1024
against competing candidates, and split a regime only if a competitor is more
than about5% faster. These are prefill tuning regimes, not execution buckets.
Actual M must continue to own launch geometry, live data, scratch capacity,
numerical validation and graph identity. Decode and MTP keep exact actual-M
contracts. Every other dispatch discriminator remains exact, including native
architecture, source/prepared formats, N/K, arithmetic partitions, workers and
layout. Existing point observations retain their original meaning and take
precedence; no historical point-only receipt becomes a range certificate.

This is an explicit new collection scope while implementation and native
qualification are pending. The stopped r1066 plan, job identities and receipts
remain unchanged. Original collectors can measure both anchors and check points;
their complete candidate tournaments are stronger checks, not falsely labeled
two-candidate spot runs. Missing or incompatible check evidence must stop range
promotion. The selected schedule still needs actual-M native Auto/byte checks,
including odd tails and the127/128/511/512 boundaries.

The retained clean r1054 root epoch explains why this matters. Its three
measured requests contain 288 prefill FFNs and9216 verifier FFNs; gate/up
projection counts are exactly twice the corresponding down counts at every M.
The CPU expert-prefill row mix is:

| Actual expert M | Fraction of expert-row work |
|---|---:|
|1..16|17.06%|
|17..127|46.20%|
|128..511|36.73%|
|512+|0% in this short workload|

The old finite prefill anchors contain78.04% of observed projection calls but
only20.28% of row work. This is a row-membership upper bound, not full-key
coverage or an installed runtime hit rate. It ignores bundle eligibility and
other exact fields. Row work is not elapsed time; no speedup is predicted from
these fractions. Dynamic routing can change them in a fresh run. The new large-M
regime addresses a significant portion, while lower-M coverage remains explicit.
Reproducible analysis and original input paths are retained in
`/tmp/qfn-hybrid-workload-attribution-r1076/`.

A source audit found no unnecessary CPU/GPU expert serialization to remove:
the CPU service worker is armed before the retained GPU parent launch, and its
manual replay waits on the producer ticket rather than GPU expert completion.
Final CPU ticket materialization must join GPU publication because the GPU
writer also clears CPU-owned slots in the canonical route bank. Removing that
join would create a write/write race. Source permits overlap; the older
intercept-queue profiler trace does not quantify unperturbed overlap. No graph
ordering change is justified by this audit.


## Architecture-scoped GPU work and current native progress

The complete CPU M1 phase has now passed all 393 preparation batches and
20,453 isolated profiler requests with no failures or missing evidence. Its
exact entries are installed and the original AVX2/AVX512 owners rebuilt. The
first grouped batch passed natively but exposed a stale adapter requirement
for `serial_median_us` and `speedup`; these fields are no longer emitted because
serial M1 is used as the numerical oracle, not a separately timed baseline.
The adapter now accepts that native schema while continuing to require the
raw timing, serial output, repeated-byte and actual-route evidence. The adapter,
trainer and new focused production-preflight registration all pass. Original
native evidence is retained in `/tmp/qfn-native-quantised-verifier-csv-r1070/`.
Collection resumed from the M1 checkpoint without repeating that phase.

The user requires architecture-aware GPU tables and explicitly requires
inference to remain usable without exact records for the actual GPU. The
policy is exact architecture first, then nearest compatible measured
same-backend architecture for the key. Physical device identity, compiled
kernel availability and launch/resource limits remain actual-card facts.
An empty/incompatible exact domain retains the complete general selector.
Selecting another architecture's tuning parameters is not a claim of native
performance qualification on that card.

Implementation is currently private in `/tmp/qfn-gpu-architecture-exact-r1068/`
and `/tmp/qfn-native-quantised-gpu-discovery-r1069/`, so it does not change the
live CPU producer. Work is divided into:

- typed runtime selection, cache identity and consistent metadata-only
  workspace planning using the existing immutable hardware inventories;
- architecture-qualified exact keys, source-owned generic retention, and
  per-architecture native timing/profiler provenance;
- shared native discovery/selected-device receipts inside existing trainers;
- architecture-separated collection and physical profiler lane selection.

The collection/profiler controls now have device-free tests for heterogeneous
inventories, noncontiguous ordinals, conflicting visibility masks, stale or
wrong-card receipts, more than eight cards, and one complete inventory per
architecture before a shared owning rebuild. These are tooling contract
checks, not native GPU numerical or performance qualification.

Existing GPU headers do not by themselves authenticate the architecture of
historical exact rows. Do not invent `sm86` or `gfx906` provenance. Preserve the
installed source-owned generic program; native sweeps will populate qualified
architecture-specific exact rows. Historical fitting artifacts are not required
to retain unchanged versioned generic code, and exact publication makes no new
generic-policy certification claim.

After the CPU sweep, first use the frozen full Release owner at
`/tmp/qfn-native-quantised-full-release-r1063/build` for a matched hybrid
before/after CPU-table measurement. Keep GPU policy unchanged for that
comparison. Then integrate and qualify the GPU architecture changes separately
on the available `sm86`, `gfx906` and `gfx1100` hardware. Full shipped-target
compilation, native Auto/byte correctness and actual per-architecture profiling
remain required; current private scripts and empty typed draft tables are not
performance-qualified production replacements.

The private orchestration checks now cover completed-run resume against changed
timing settings, binary ownership and retained native evidence, plus completing
an M1 delta that was staged but not installed. The latter must reuse its paid
measurement rather than refusing the transition or remeasuring it. Eighteen
driver tests and eight device-selection tests pass without native work. The
selected-device integration driver exercises correct identity and wrong UUID/
architecture controls on every discovered card; its control-flow regression
uses 17 authored devices to guard against accidental fixed device-count limits.
The real CUDA/ROCm runs and CMake target integration are still pending.

ROCm profiling preparation is being made architecture-aware using the selected
SDK's actual agent metadata, available counters and schedulable counter groups.
Prepare this once per physical lane, before workloads; do not rerun workloads
to discover that an MI50 counter group is invalid on RDNA. The SDK agent index
must be bound by PCI/ISA and kept distinct from HIP's masked ordinal. The
current SDK does not expose gfx9's `VALUBusy` quantity on gfx11. Preserve that
absence explicitly, with a narrow gfx11 evidence contract requiring its real
instruction, memory and static-resource observations; instructions per wave
must never be mislabeled as a busy-time percentage. Native profiler validation
on both architectures remains required before table publication.

## Combined source qualification and remaining prefill work

The runtime, native metadata, Python architecture and orchestration packets are
assembled privately at `/tmp/qfn-native-quantised-gpu-assembly-r1072/`. Two
overlapping fixture include edits were combined without dropping either owner.
The combined Python slice passes 384 focused tests across 13 modules. Initial integration found
four synthetic selected-device receipts without the now-required compiler and
runtime facts; those fixtures are repaired, and explicit missing-fact rejection
controls remain. Original failures and repaired results are retained under that
directory. These are device-free tooling results, not compiled GPU evidence.

Independent review confirmed the four earlier runtime caller, service geometry,
empty-table and streaming-support repairs. The combined source also closes two
publication gaps before collection:

- Prefill fresh/resumed observations and decode lane resume now compare the
  child's actual compiler/runtime facts with the collection owner, rather than
  only validating physical GPU identity and nonempty metadata.
- Dense and MoE prefill require isolated profiler evidence before measured
  table promotion. Existing MoE retained-range controls are reused; both dense
  native fixtures now have scoped range hooks in staged source. Emitters
  independently reconstruct current winning obligations before publication.
  Native profiler qualification is still pending; decode profiling cannot
  certify a different prefill operation.

Ordinary CPU BackendNative prefill now composes with the existing full exact
transaction, with independent keys/includes and one final rebuild followed by
both actual-Auto validators. The ordinary-only command can collect the new
surface without repeating paid legacy/fused phases. Completing the current
decode/grouped/expert sweep does not certify that surface. Existing device-free
planning fixtures now author complete synthetic GPU facts; native integration
owners use their actual immutable hardware metadata. No expected byte constants
were changed. These C++ changes remain uncompiled.

The subsequent source review found and repaired a direct-entry consistency gap:
CUDA quantized workspace planning could accept internally valid hardware facts
for another participant or compute-unit count. One estimator-local check now
guards the three quantized sizing sites; the explicitly registered workspace
regression exercises missing metadata, another backend, both directions of
ordinal/count disagreement, and an unaffected floating projection. No new
metadata query or memory ledger was introduced. Review also repaired a malformed
ROCm test declaration from the hardware-first API migration and removed stale
`gfx906` labels from the three architecture-independent ROCm sweep owners.
Three positive planning fixture paths also needed complete synthetic facts when
constructing or remapping their ngram/head-partition participants. The unknown
ngram-owner negative keeps internally consistent hardware, so it still tests
ownership rather than passing on an unrelated metadata error.
These C++/CMake repairs are staged in `combined.v4.patch`; their native checks
remain pending. The 384 Python results cover unchanged Python source.

None of these private changes modifies the active r1066 collection or frozen
r1063 model benchmark owner. The combined patch has a clean strict whitespace
apply-check. Native build, profiler preparation, installed Auto,
per-architecture sweeps and matched model performance remain outstanding.

## Four/two/one same-panel ownership, 2026-11-09

This extends the qualified two-pair scheduler, without another exact-shape
sweep. Ordinary Prefill/GPUAlignedExpert/Pairwise/OutputTileLocal bundles may
retain four adjacent pair/tail calls on one prepared N64 panel. The original
two-pair mode remains selected when four-pair jobs would fall below four jobs
per actual worker; smaller grids retain one pair. The internal pure selector
and job decoder use the same width. Pair bodies, K partitions, reduction order,
row counts, bias and completion edges are unchanged. No allocation, packing,
communication, runtime switch or exact-table update is added.

The two-pair source is checkpointed in `a0fd16d04`, merged with remote maintenance
as `f564d784e`; both were pushed. Its native control is retained at
`/tmp/qfn-cpu-four-pair-baseline-r1112/`. Current candidate evidence is
`/tmp/qfn-cpu-same-panel-four-pair-next/`.

Six Release-owner registrations pass in 10.20 seconds, covering all 21 formats,
AVX2 and AVX-512 runtime dispatch, live prefixes, dirty/repeated execution,
retained teams, odd tails and decode/MTP controls. The existing explicit
preflight aliases include the new actual-selector boundaries for worker counts
1–64 and native single-descriptor M32/128/256 at 28 workers. These are runtime-ISA
results from the AVX512 build, not an independent AVX2 compilation certificate.
Final linked core and fixture inspection finds unchanged pair arithmetic and
no new hot-loop spills or helper calls.

The existing complete-FFN owner uses 28 physical socket0 workers, Q4_K gate/up
N320/K2560 and Q5_1 down N2560/K320. Each arm has two fresh lifetimes in reversed
order, eight warmup batches and nine samples of four transactions. All native
serial-byte oracles pass. These balanced descriptor sets are authored controls,
not captured production expert routes.

| Authored complete operation | Two-pair us | Four/two/one us | Speedup |
|---|---:|---:|---:|
| Prefill E84/R519, persistent FFN |3731.390|2308.850|1.616×|
| Same gate/up |1973.975|1162.890|1.697×|
| Same down |1385.335|671.986|2.062×|
| MTP E24/R72, persistent FFN |652.102|655.217|0.995×|
| Prefill E1/R128, persistent FFN |297.818|253.452|1.175×|
| Prefill E1/R64, persistent FFN |159.297|142.688|1.116×|

MTP's +0.48% latency change lies within the observed lifetime variation.
The same reversed comparison with forced AVX2 runtime gives 6604.335→5364.600us
for the complete prefill FFN (1.231×) and 1145.795→1099.585us for the MTP control.
MTP scheduling is unchanged; its timing difference is not a new MTP policy win.
The AVX2 samples, native byte oracles and closure receipts are in `economy-avx2/`
under the candidate evidence directory. This still uses AVX512 compiler output
with AVX2 runtime dispatch, not a separately compiled AVX2 owner.
E1/R128 has two gate/up descriptors, so its joint four-pair grid qualifies;
it is not a single-descriptor two-pair performance witness. E1/R64 supplies
that complete-FFN gate/up control (160 two-pair versus 80 four-pair jobs), while
the native single-descriptor byte test covers M128. Native-policy control
variation is larger for the tiny R64 case; do not assign its gate-only change
to the unchanged two-pair schedule.

A separate 2000-transaction profile records 83,735 samples, no lost samples and
28 native TIDs. Named nibble/non-nibble sampled periods fall from 214.808/170.272G
to 137.190/85.918G, and OpenMP from 83.046G to 46.568G. Setup/oracles are included;
this is sampled user-cycle attribution, not cache/IPC evidence or model timing.
The matched model comparison at `/tmp/qfn-four-pair-model-r1115/` passes at
278.611720 prefill /38.800220 decode tok/s, versus267.049428 /38.476294:
+4.33% prefill and +0.84% decode. The unchanged decode policy does not establish
a decode optimization. All 768 output IDs, actual MTP work, both ranks' initial
expert selection and runtime configuration match (apart from owned output
paths). Maximum context262144, dynamic MTP/balancing, RAM ngrams, local GPU
cache and 16/32 GiB prefix tiers remain fixed. The observer completes 223 host
samples and 1107 process-memory observations without error; the native process
tree and both GPU contexts retire cleanly. This is not a leak or full-resource
certificate. The 18 owned disk-cache files (727,912,502 bytes) are removed by
metadata-only cleanup. No full-preflight or release claim is made.

The bounded decode review at `/tmp/qfn-decode-root-attribution-r1114/` records
5795.255ms main-verifier CPU compute over 9110 nonempty endpoints, versus217.157ms
predictor compute over 576 endpoints, across the three measured requests.
Readiness waits overlap device/parent scopes and are not isolated GPU timing.
The conservative low-live gather selector would affect only 212/9792 root calls,
half empty, and no predictor calls. That proposed optimization is parked;
`/tmp/qfn-shmem-low-live-feasibility-r1113/` preserves its limited applicability.

## Ordered three-row tiles and live-plan initialization, 2026-11-09

The next bounded slice changes physical execution, without resweeping the
retained inventory or changing the generated row-policy table. An AVX512
WideRows tile with exactly three live rows can retain six integer-dot chains
for two sixteen-column segments and the original ordered FP32 partition fold.
The canonical prepared decoder/asymmetry metadata selects the three real
families; impossible Cartesian combinations reject before payload access.
All five nibble source formats retain their original scale/min handling,
including IQ4_XS factor restoration. The same resolved WideRows policy selects
the body for explicit requests and eventual measured Auto choices.

Linked V3 core and complete-FFN owner inspection finds no hot accumulator/LUT
spills. The original pair helpers retain identical normalized instructions.
The new native lifecycle covers all 21 formats, both runtime ISAs, original
source slices, owned/workspace banks, rows11→3→7→3→2→1→3, live-descriptor changes,
poisoned outputs and tails, immediate existing-team completion, and malformed
metadata rejection. V3's 12 focused registrations passed in16.29seconds.

The existing complete-FFN owner supplies matched Q4_K gate/up N320/K2560 and
Q5_1 down N2560/K320,28physical workers, two fresh lifetimes per arm in reversed
order,8warmup batches and9samples×4transactions. These are authored workload
vectors, not recorded model routes. Explicit WideRows comparisons are:

| Operation | Previous us | Ordered three-row us | Speedup |
|---|---:|---:|---:|
| MTP E24/R72 gate/up |204.398|191.082|1.070×|
| Same complete persistent FFN |316.505|310.716|1.019×|
| MTP E24/R168 gate/up, four-row plus three-row tail |425.548|415.127|1.025×|
| Same complete persistent FFN |788.316|766.138|1.029×|

Unchanged Auto control is645.895→639.375us; this small difference is not a new
Auto dispatch win. Separate gate/up profiles have27,252/26,024samples, zero lost
samples and28native worker TIDs. They show the intended new three-row body,
but include setup/oracles and OpenMP work; their sampled user-cycle periods
are neither canonical timings nor cache/IPC evidence. Artifacts remain at
`/tmp/qfn-cpu-ordered-three-row-next/`.

A second change removes unused plan initialization from each hot invocation.
One common fill routine serves the unchanged full-value public plan and an
internal invocation-owned live-prefix view. The latter constructs only the
actual live objects in the original fixed stack backing. Public default, copy
and assignment semantics remain intact; no heap allocation, shared planner,
cache, publication edge, arithmetic change or parallel memory ledger is added.
The focused lifecycle compares every observable public/private field, poisons
inactive backing, checks37→1→37→3→1 reuse and actual1/3/7-worker teams, and retains
the independent all-format serial-byte oracle. Fourteen final native aliases
passed in17.55seconds; both new LivePlanFill aliases belong explicitly to
production preflight.

Final core/fixture ISA confirms the hot path initializes num_descs×128bytes,
without the131,360-byte capacity clear or1026-entry defaults loop. The public
value wrapper intentionally retains those operations. The fill's frame size
is unchanged; the launcher grows64bytes. Separately emitted numerical bodies
retain identical normalized instruction sequences, and no new hot spill is
introduced. The paired complete-FFN controls against the frozen V3 owner show:

| Runtime ISA / operation | Full-value hot plan us | Live-prefix slice us | Latency change |
|---|---:|---:|---:|
| AVX512 / MTP E24/R72 |649.083|634.022|−2.32%|
| AVX512 / Prefill E84/R519 |2160.015|2106.440|−2.48%|
| AVX2 / MTP E24/R72 |1157.425|1074.775|−7.14%|
| AVX2 / Prefill E84/R519 |5378.840|5335.515|−0.81%|

AVX2 prefill overlaps the observed lifetime variation. Both runtime regimes
use the AVX512 compilation; this does not qualify independent AVX2 compiler
output. Evidence is at `/tmp/qfn-cpu-fused-live-plan-r1118/`. The matched model
comparison at `/tmp/qfn-live-plan-model-r1119/` passes at279.840919/38.838234tok/s
versus278.611720/38.800220. The +0.44%/+0.10% differences are within run variation;
this establishes no clear model gain. All768 output IDs, actual MTP work,
configuration and both ranks' initial expert selections match. Native/driver
exit and process/GPU retirement pass;221 passive host samples and1097 memory
observations have no errors, without claiming a leak/full-resource certificate.
Metadata-only cleanup removes18 owned cache files totaling727,912,510bytes.
The500/40goal remains open.

The expert-only collector now accepts a plan-owned `--expert-m` projection,
retaining the selected purposes' finite row inventory, candidate grades,
mixed-row holdouts, unchanged default plans and saved-scope resume checks.
Its Unit and explicit Scope/Evidence preflight registrations passed in1.60seconds.
The new M3 scope needs only2prepared batches/8requests; other paid cells remain
untouched. During this collection, review found that the exact owner records
projection counters inside timing intervals. The completed candidate clocks
are therefore diagnostic-only and must not promote a table. Original records
are retained at `/tmp/qfn-cpu-three-row-exact-next/campaign/`. No generated table
from that instrumented campaign has been installed.

The repaired owner suppresses recording through the existing PerfStats policy
at quiescent team boundaries, outside screening/warmup/sample clocks, and
restores the original policy for fresh route authentication. Native results
prove zero timed projection observations. Import, completed-cell reuse and new
winner publication reject instrumented clocks without discarding their original
diagnostic records or unrelated legacy/GPU collectors. The new original-code
control reproduces 134 observations instead of four untimed guards, and 126
instead of zero for adaptive timing. All four fixed native full/adaptive aliases
pass on AVX2 and AVX512 runtime routes in 2.97 seconds; fourteen focused Python
Unit/preflight registrations also pass. An authored fixture initially used an
unaligned N65 row; N80 restores its actual aligned execution class while keeping
the N64 tail. Neither production admission nor its numerical checks was relaxed.
The aliases now register beside their original executable, so explicit Release
selection reaches them as well as the Integration preflight inventory.

Fresh collection at `/tmp/qfn-cpu-exact-timing-isolation-next/campaign-clean/`
contains only two prepared M3 batches/eight requests and four separate profiler
requests. Resuming leaves all ten native timing artifacts unchanged. Gate/up
Q4_K training changes Pairwise 22.046 to WideRows 18.620 microseconds; its small
mixed holdout is effectively tied at 19.040/19.028 microseconds. Q5_1 down keeps
Pairwise (6.103 versus 7.561 microseconds). The generated two-key overlay retains
all prior includes and selects WideRows only for the measured AVX512/28-worker
MTP M3, N320/K2560 gate/up execution class; down remains Pairwise.

An independent authored E24/R42 control uses `[3,3]+[4]*3+[2]*5+[1]*14`, changing
only matching M3 keys. It is not a recorded model route. Forward/reverse fresh
lifetimes give gate/up Auto 91.408/91.973 versus WideRows 83.226/83.700
microseconds, about 9% lower whole-bundle latency. Mixed down prefers WideRows
in this larger vector, unlike its smaller training case; that observation does
not justify replacing the retained down policy. Canonical installation now
passes rebuilt Auto-dispatch authentication. Twenty focused native aliases pass
against the final Release core in 21.23 seconds. The larger post-install control
records four gate/up and two down exact hits, unchanged bytes and zero timed
observations. Across its two fresh lifetimes, Auto gate/up improves from 91.690
to 81.608 microseconds (11.00%); down is 47.615 to 46.548 microseconds, a smaller
difference that does not establish a new down-kernel gain. Final core ISA proves
identical complete numerical-helper/caller instructions versus the pre-table
core, including the inlined linear three-row body and unchanged frame/spill
behavior. The separate one-call gate profiles show Pairwise/WideRows instruction
counts 2.175/1.916 million; their perturbed durations are not canonical clocks.
The final matched model run at `/tmp/qfn-m3-overlay-model-r1121/` passes at
279.867598 prefill /38.707213 decode tok/s, versus the live-plan baseline
279.840919/38.838234. Changes are +0.01%/−0.34%, within run variation; this is
not a model-speed win. All 768 output IDs, actual MTP work, normalized runtime
configuration and both ranks' initial expert selections match. Context remains
262144, with the original dynamic MTP/balancing and RAM/local-cache placement.
Native and driver exits are zero, no new driver findings appear, and the owned
processes/contexts retire. Passive observation contains 219 host samples and
1087 individual process-memory observations without errors. Largest individual
RSS is 82.862 GB versus 82.834 GB; each selected GPU's maximum individual-client
resident VRAM differs by only 12 KiB. Shared aliases are never summed, and this
is not a full leak/resource certificate. Metadata-only cleanup removes 18 owned cache files,
727,912,513 bytes. The 500/40 model target remains open. This checkpoint contains
the measured component improvement and repaired measurement workflow; complete
production preflight and image certification are still outside this focused run.

The closed ROCm trace at `/tmp/qfn-prefill-next-attribution-r1118/analysis/`
measures one 512-token prefill after preparation and warmup. Root pure wait is
832.691 ms, of which 820.986 ms overlaps follower collectives: these are largely
the same critical interval and must not be added. Native Q8 GEMM contributes
197.290 ms root interval union, and live CPU-ticket materialization 134.484 ms.
The latter reads only live rows; no capacity-sized transfer defect is established.
The existing N64 tile for N320 prepared-Q8 projections is now measured separately
for HC K10240 and shared-FFN K2560. The original prepared-weight factory and
retained GEMMStage graphs exercise NativeCooperative, MT32/MB1/UG2 on the actual
gfx906 at PCI 8c:00.0. Two native regressions pass twenty dirty large/small/reset
histories, original weight bytes, output bytes and inactive guards. Four separate
event cases each repeat byte checks, then collect 31 alternating candidate pairs.

| Actual M / K | N64 median µs | N128 median µs | N128/N64 |
|---|---:|---:|---:|
| 448 / 2560 | 150.719 | 158.240 | 1.050 |
| 448 / 10240 | 438.560 | 461.439 | 1.052 |
| 64 / 2560 | 116.959 | 147.360 | 1.260 |
| 64 / 10240 | 333.120 | 434.400 | 1.304 |

Every pair favors N64, but M64 already uses it. These intervals include the
original activation quantizer and projection; inputs are authored deterministic
fixtures. The roughly 5% large-chunk component improvement does not establish a
model gain, and no production dispatch is changed. Native function receipts
report N64/N128 at 86/114 VGPRs, 12,800/21,504 LDS bytes and zero local storage.
These are static resources, not measured occupancy. Evidence and raw pairs are
at `/tmp/qfn-rocm-native-q8-small-n-r1122/`.

Building this original owner in selected Release mode exposed missing test-root
includes and four native helper translation units hidden under perf-only
registration conditions. They now belong to the original target independently
of registration mode; duplicate late attachments are removed. Compilation covers
CUDA 80/86/89/90 and HIP gfx906/gfx1100 with spill guards. Native byte/reset gates also pass on gfx1100 at PCI 49:00.0;
projection timing here qualifies only the selected MI50. The fixture also matches the captured function
identity directly: HIP does not promise demangled display names. Both lifecycle
regressions are explicitly registered in production preflight; the four timing
cases remain opt-in. The general Q8 variant sweep still does not exercise this
NativeCooperative producer, so unrelated variant records cannot qualify it.

Closed prefill records also narrow the next CPU question without recollecting
already-paid data. The r1118 follower's large chunk carries median 3206.5 routed
rows, versus 519 in the authored FFN control. The retained r1115 JSON contains
prefill histograms even though its text report omitted them: dominant gate/up
row-weighted median M is 72, with 71.57% of row-events at M >= 32. These are
marginal descriptor counts, not recoverable mixed route vectors or per-M time.
The larger-row policy is unchanged by the later M3-only overlay. Full analysis
is at `/tmp/qfn-cpu-current-prefill-shapes-next/`; it prevents extrapolating the
small balanced fixture's speedup to the dominant model workload.

### Prepared-Q8 row tile follow-up

One further existing candidate, N64/MT16/MB1/UG2, improves both physical row
counts without changing arithmetic or the original prepared Q8_0 planes. The
same owner captures the true Auto oracle and both candidates, proves twenty
dirty large/small/reset histories, then measures 31 alternating pairs:

| Actual M / K | N64/MT32 median µs | N64/MT16 median µs | MT32/MT16 |
|---|---:|---:|---:|
| 448 / 2560 | 150.720 | 138.560 | 1.088 |
| 448 / 10240 | 438.720 | 399.680 | 1.098 |
| 64 / 2560 | 116.960 | 97.441 | 1.200 |
| 64 / 10240 | 333.600 | 267.680 | 1.246 |

Both row-tile byte/reset registrations pass on MI50 and 7900 XTX, and belong
explicitly to production preflight. Timing cases remain opt-in. Evidence:
`/tmp/qfn-rocm-native-q8-row-tile-r1123/`. These component results are not an
installed policy or a model speedup. Native-family collection, isolated
winner/baseline profiling, architecture-scoped exact emission and post-install
Auto/model qualification are still required; expanded INT8 variant records
cannot be relabeled as this native producer's measurements. No broader shape
resweep is needed to finish those four selected cells.

### Canonical CPU-ticket ingress: two independent HIP reads

Original gfx906 materializer ISA serialized each scalar mapped load, wait and
store before the next load. The local HIP loop now issues two independent
scalar fresh reads before storing. It retains the original immutable ticket
lease, exact live source/destination indices, acquire, per-route barrier and
separate whole-grid acknowledgement. CUDA and the common copy helper remain
unchanged. This is scheduling only: no arithmetic, precision, transport,
allocation, launch-count or communication-extent change.

The original native lifecycle fixture covers twenty generations with empty,
sparse and full payloads, aborts, held publication, dirty destinations and
retained capture. QFN H2560 capacities640/4480 are added alongside existing
H67/3072; partial second-read/next-iteration widths257/513 have focused cases.
The original ready-payload performance owner adds nine H2560 live-count cases
with five warmups and fifty raw event samples each, emitted outside timing.
The authored counts are not recorded model route vectors or MTP distribution.

On MI50, baseline/candidate/baseline medians in microseconds are:

| Capacity / live rows | Baseline A1 | Two reads | Baseline A2 |
|---|---:|---:|---:|
| 640 / 3 | 61.600 | 54.960 | 61.520 |
| 4480 / 3 | 64.720 | 57.440 | 59.440 |
| 640 / 640 | 527.040 | 527.520 | 528.240 |
| 4480 / 4480 | 3467.517 | 3456.398 | 3456.157 |

Small ready cases improve about8–12%; bulk transfer remains approximately
13.2 GB/s. The same nine-case gfx1100 comparison improves the two live3 medians
by about8–12% against the mean controls, while large transfers remain within
about2% and show baseline lifetime variation. Empty cases do no payload work
and do not establish a copy win.

Both gfx906/gfx1100 ISA show two independent scalar reads before the first
wait/store, zero spills and zero private memory. Candidate compiler resources
are respectively17/18 VGPR and36/34 SGPR with wave64. Separate isolated gfx906
profiling for exactly H2560/capacity640/live3 confirms unchanged median wave,
VMEM-read/write and external-read-request counts. Native allocated VGPRs rise
16→20; compiler and allocated counts are different measurements. Achieved
occupancy is not available from the selected gfx906 counter inventory. Profiler
durations are not used as canonical clocks. The complete CUDA80/86/89/90 and
HIP gfx906/gfx1100 build retains mandatory spill guards.

The matched Release model finishes at **280.307100 prefill /38.534730 decode**
tok/s, versus279.867598 /38.707213: **+0.16% /−0.45%, effectively flat**. All768
generated IDs, MTP work, normalized configuration and initial placement match.
This remains the original 512-input/256-output, one-warmup/three-measurement
workload on two MI50 continuation cards and two physical CPU socket ranks,
context262144, dynamic MTP/balancing, RAM ngrams, local GPU caches and16/32GiB
RAM/disk prefix budgets. No profiler/build competes with canonical timing.

Driver observation reports205 snapshots and zero findings;219 passive resource
samples show no increased maximum individual RSS or per-client VRAM. RSS peaks
82.830 GB versus82.862 GB; client maxima are32.455/32.279 GB. Aliases are never
summed, and these short observations are not a long-lived leak certificate.
All processes retire. Metadata-only cleanup removes18 run-owned prefix files,
727,912,507 bytes, without reading/hashing cache payloads.

Evidence: `/tmp/qfn-canonical-ingress-two-scalar-r1123/` (frozen original core,
raw A/C/A samples, ISA and isolated counters),
`/tmp/qfn-canonical-ingress-model-r1124/` (matched model, resource and retirement
receipts). The 500/40 target remains open. Full preflight remains explicitly
outside this focused iteration.


### CPU root combine/preweight: live-entry worksharing

The original root still validates and combines every required participant in
its existing FP32 order, keeps the complete original raw output, and publishes
the same compact weighted ticket. One worker now weights each completed column
tile immediately after the unchanged `vec_add` call. The validated live-slot
list drives the task grid directly; inactive routes and per-tile binary searches
are omitted. This removes the separate whole-bank traversal/workshare, not the
raw read itself. Column partitions, multiplier authority, allocations, transfer
extents, graph edges and publication sequence remain unchanged. Borrowed sink
metadata, factor bytes, aliases and armed publication are checked before writes.
A single fused timing child prevents counting the same operation twice.

An initial dual-output arithmetic primitive was rejected: its compiled scalar
and vector tails selected different NaN payloads. It is removed completely.
Final code calls the original addition primitive, then separately multiplies
by the original finite router factor. The final linked ISA has no FMA or hot
vector spills. Same-worker reuse is intentional; arbitrary-width L1 residency
is not guaranteed. Runtime scalar/AVX2/AVX512 tests use the AVX512 build and do
not claim independent pure-AVX2 compilation.

Eight original CPU registrations pass, including 2–8 participant orders,
distinct signed quiet/signaling NaNs, subnormals, zero/Inf/cancellation, tails,
aliases, bad factor/slot metadata, held/foreign publication and repeated dirty
large→small→empty use. The original captured fixture passes prefill, decode and
verifier cases on CUDA, MI50 and 7900 XTX: nine registrations, twenty generations
each. Focused regressions are explicitly registered in production preflight;
the complete preflight was not run during this iteration.

The existing ready-record performance owner retains 4480 slots but uses the
correct logical extent: M64/640 slots with453 live routes, or M448/4480 with3208.
These are authored sparse controls, not captured model route vectors. At28
workers on one actual socket, two AVX512 lifetimes measure **122.760→60.791** and
**118.838→58.822 μs** for M64 (about2.02×); M448 measures **1887.430→1830.245** and
**1879.650→1777.090 μs** (3.1–5.8%). Runtime AVX2 gives1.93×/1.04× respectively.
All30 small-case pairs favor the candidate; large cases have substantial
first/second-arm variation and only15–16 paired wins. Earlier small-case data
that mistakenly used logical M448 is retained as diagnostic, not the M64 result.

The matched passive Release model measures **286.040438 prefill /39.074346
decode** tok/s versus280.307100 /38.534730: **+2.05% /+1.40%** in this comparison.
All768 generated IDs, MTP work, normalized configuration and initial placement
match. Controls remain512 input/256 generated, one warmup/three measurements,
two MI50 continuation cards and two physical CPU socket ranks, context262144,
dynamic MTP/balancing, RAM ngrams and16/32GiB prefix budgets. This is a modest
measured model gain; the500/40 target is still open.

All processes retire;206 driver snapshots report zero findings. The220 passive
resource samples show maximum individual RSS82.863GB and per-client VRAM
32.455/32.279GB, comparable to the previous run; aliases are never summed and
this is not a long-duration leak certificate. Metadata-only cleanup removes18
retired run-owned prefix files (727,912,509 bytes), without payload reads/hashes.
Evidence: `/tmp/qfn-cpu-root-combine-preweight-r1125/` (focused/native/ISA and raw
paired economy) and `/tmp/qfn-cpu-root-fusion-model-r1125/` (matched model,
configuration, resources, driver observation and retirement).
