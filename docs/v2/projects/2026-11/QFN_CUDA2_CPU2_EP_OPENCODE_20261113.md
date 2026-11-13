# QFN CUDA2 / CPU2 EP OpenCode qualification

This is a local Release diagnostic of the user-requested two-3090, dual-socket
whole-expert EP configuration. It is not an image, numerical-parity or release
certificate. The fresh q686 session passes the complete stress and engineering
gates: ten phases, 112 requests, 105 tool calls, one tolerated model edit mismatch
(0.95%), and 53 generated-app tests passing. Runtime, native-output, resource,
driver and retirement checks pass with unchanged admission limits. Earlier
failures remain preserved below; they are not rewritten as successful runs.
This investigation is a detailed record, not a tuning dashboard.

## Configuration

- Source: `feat/qfn-ngram-continuation`, base
  `d836a276fc556290e44942afd9fe6f5103e36c60`, with the qualified working-tree changes.
- Release build: `/tmp/qfn-rocm101-upgrade/release-build`.
- Model: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, with the
  shared Q8_0 MTP sidecar from the same installed model family.
- Native automatic request: CUDA=2, CPU=2, ExpertOverlay, Apportioned experts.
  Original MPI world contains one rank per physical socket, 28 physical workers
  per socket. Rank 0 owns both CUDA devices on NUMA 0; the CPU domain spans both
  ranks. The dense continuation uses CUDA TP2. GPU/CPU expert tiers use whole
  expert ownership and dynamic rebalancing.
- Dynamic MTP, retained capacity 15 and selected depths 1–15; stochastic
  verification for serving. FP16 router operands with FP32 accumulation,
  FP16 MoE alpha/beta, two-term Q8 HC-up and single-Q8 GDN/sparse-output policies.
- Context 262,144, configured output budget 32,768. No reduced reply reserve.
- RAM ngram backing with ConsumerLocal GPU row caches, each with a 256 MiB
  admission ceiling. Prefix cache: 16 GiB RAM per participant, 32 GiB shared disk.
- No visibility masks, topology substitution or generation deadline.

The original request, native selected plan and all local receipts are under
`/tmp/qfn-cuda2-cpu2-ep-opencode-20261113`. The native planner took 3.046 seconds;
the fixed plan admitted the model's maximum context. The selected plan is an
unchanged native artifact, not a hand-authored topology document.

## Completed checks

Root command receipts are in `/tmp/qfn-full-preflight-after-cpu-next`.

| Check | Evidence | Result |
|---|---|---|
| Automatic planning | q640, `selected-plan.json` | Passed |
| Full-context admission | q641, `context/` | Passed |
| Smoke and graceful shutdown | q649, `smoke-q649/` | Passed |
| Short Release benchmark | q650 native output, q653 retrospective validation | Passed measurement; original wrapper failure retained |
| OpenCode app | q654, `app-q654/` | Failed: completed-prefill to one-token decode boundary |
| Rebuilt admission and smoke | q660, q661 | Passed, same plan/context; native retirement and driver checks green |
| OpenCode app after slot repair | q662, `app-q662/` | Failed: finite prefill interval remained declared across suffix decode |
| Exact-interval admission and smoke | q668, q669 | Passed, clean retirement and driver checks |
| OpenCode app after both prefix fixes | q670, `app-q670/` | App/protocol/native audit passed; resource and MTP evidence checks failed |
| Diagnostic-retention rebuild admission | q684, `context-q684/` | Passed, unchanged 262,144-token Auto plan |
| Diagnostic-retention smoke | q685, `smoke-q685/` | Passed runtime/resources/driver/retirement; archive reclaimed |
| Complete rerun after diagnostic repairs | q686, `app-q686/` | Full stress and engineering pass; native retirement and archive reclamation complete |

Smoke returned exactly `ready`, stopped normally and passed attention, graph,
transfer, RSS/VRAM/cache-growth and driver checks. Native processes retired
without forced termination; the owned disk archive was reclaimed without
reading payloads.

The benchmark used 512 prompt tokens, 256 emitted tokens (255 after prefill),
greedy seed 42, dynamic MTP, no prefix cache, one warmup and three measured
iterations. Native mean throughput was **389.856 prefill / 59.660 decode-after-
prefill tok/s**. The canonical median-of-samples measure was **376.510 / 59.234**.
Prefill samples were 435.107, 376.510 and 364.845 tok/s; decode samples were
59.084, 59.234 and 60.687. All measured iterations ran MTP, with 12 controller
windows overall and no transaction/state validation failures. This short greedy
benchmark does not demonstrate stochastic activity or a depth transition.

Its actual expert banks placed 32 or 34 of 512 experts per layer in the GPU
tier, with the remaining 480 or 478 apportioned across the CPU sockets. The
full-context and MTP workspace reservations remain part of this configuration.

## Evidence-reader repairs

The smoke and benchmark evidence-reader repairs below preceded the native
lifecycle repair identified by the app session.

1. The local two-token smoke could complete between one-second stats polls.
   It now polls every 50 ms, still requiring an actually observed active request.
   The full app retains one-second polling. A negative control reproduces the
   original missed observation.
2. The local resource reader now includes the existing CUDA NVML management
   observations when CUDA is selected. It retains exact PID, UUID and PCI
   binding; a selected telemetry failure remains a failure.
3. The canonical graph validator omitted retained-parent setup from its strict
   decode-capture check. It now joins parent materialization, nodes and initial
   submission by rank/device/context. Full-app replay additionally requires
   repeat submission. Missing/foreign lifecycle records remain rejected. The
   regression is included in the explicitly registered
   `V2_Integration_HTTPNativeParentEvidence` class. q646 reproduces the original
   failure; q647 passes the focused Unit and preflight suites.
4. The benchmark reader mistook a CPU socket's topology ordinal for its native
   DeviceId ordinal. Its expert projection now preserves the global participant
   identity while matching the native CPU record spelling. Coverage includes
   1/2/4/8 sockets, apportioned and tensor-sharded banks, and missing/duplicate/
   foreign/population failures in `V2_Integration_BenchmarkRuntimeIdentity`.
   q651 reproduces the original failure; q652 passes the focused suites.

The q650 local wrapper originally inspected only the root's embedded benchmark
records. `review_benchmark.py` rechecks unchanged original inputs and joins the
second rank's 49 immutable initial-bank records from its saved native artifact.
It does not combine runtime counters from different measurement epochs or
rewrite the original failure. `benchmark-q650/revalidated-result.json` retains
the successful measurement and setup validation; graph/controller measurement-
epoch diagnostics remain distinct from full serving-lifetime qualification.

q648 passes all 51 local wrapper tests. The original failed smoke receipts are
retained, and q649 supplies the subsequent complete smoke proof. No full
production-preflight rerun or new numerical certificate is claimed here.

## App failure and focused repair

At 10:32:45 UTC, request 50 failed after restoring a prefix and executing a
prefill interval. Its 35,585-token prompt ended one token beyond the next
64-token checkpoint. The final one-row interval correctly selected the serial
decode bridge, but that bridge called strict graph-sequence admission while the
completed preceding prefill sequence still retained its protocol slots. The
coordinator rejected it with `graph sequence cannot overlap live graph slots`.
This is a deterministic lifecycle transition inside one outer PREFILL command,
not evidence of a Unicode parser or driver defect. The preceding expert
maintenance epoch does not explain this slot state.

`beginRestoredPrefixDecodeSequence` now admits either an idle command sequence
or a fully completed depth-zero MainPrefill sequence. The latter retires through
the existing authenticated sparse-return/publisher contract before decode opens.
Incomplete, live, speculative and non-prefill sequences remain fatal. It adds
no device synchronization, host state mirror or alternate execution path.
Focused tests cover 60 repeated commands with zero/one/two preceding chunks,
exact publisher/progress accounting, rejection boundaries, and an original-call
negative control reproducing the exact error. They are explicitly registered in
`V2_Integration_MoEOverlayRestoredPrefixDecodeLifecycle`. q655 rebuilt the
Integration owners; q657 passed all six selected Unit/preflight suites,
including the complete prefix-flow, transaction and collector owners. q656
rebuilt the Release server, metadata tool and native tool-evidence decoder.
q660 re-admitted the unchanged maximum-context plan on that binary.

The failed native lifetime retired all processes without forced termination;
driver diagnostics contain zero findings. Its receipts and triggering
`app/session-0000-webapp/exchange-0049.json` remain unchanged. Last observed GPU
allocations were stable near 20.37/19.69 GiB; CPU anonymous memory rose from
86.93 to 88.39 GiB on rank 0 and 48.65 to 49.72 GiB on rank 1. This growth is not
yet certified as bounded. Diagnostic collection retained geometry-dependent
records; its family filter also incorrectly admitted sibling families into the
internal map before hiding them at export. A separate focused collector repair
now applies family admission before insertion. Unfiltered collection and the
source of the complete RSS delta still require qualification.

The canonical OpenCode collector now selects serving evidence explicitly,
excluding detailed CPU per-kernel geometry/timing diagnostics. Its broad
transfer-producing domains preserve unknown semantic operation names for the
existing forbidden-transfer check. The exact 19-selector policy projects the
closed q649 smoke from 81,473 to 24,012 root records and 6,842 to 1,235 follower
records while retaining identical structured validator results. It retains and
rejects 28 injected forbidden transfers across all seven current producer
domains. This offline projection is evidence-preservation validation, not RSS
attribution or an app pass. The script/results live at
`/tmp/qfn-perfstats-family-retention-next/evidence/`.
q658 passed the canonical OpenCode Unit and explicit preflight owner; q659
passed all 51 local lifecycle-wrapper tests. No resource-growth limit changed.

The q662 rerun completed 12 requests before encountering a 12,993-token prompt.
The first repair allowed the one-row bridge to execute, exposing a second
condition: the finite aggregate prefill calibration declaration remained live
across that decode. Command completion therefore compared all transactions,
including the suffix, to the prefill-only terminal ordinal. The original focused
fixture had no aggregate declaration and missed this outer contract.

The regression now declares the actual one/two-chunk prefill workloads, covers
undershoot and overshoot, and uses the real two-participant completion-event
probe. q664 reproduces three failing assertions against the first repair.
`closeCompletedPrefillScheduleLocked` now closes that interval at the typed
suffix boundary after exact sparse-return retirement; it also owns the same
checks and field cleanup at ordinary command completion. Exact cardinality and
probe-terminal obligations remain mandatory. q665/q666 passed the subsequent
Integration/Release builds; q667 passed both complete affected Unit owners and
the explicit boundary/bounded-evidence preflight tests. Native qualification
continues on this build.

q662 again retired without forced termination, survivors or driver findings.
Follower anonymous memory reached 48.70 GiB after 12 requests (about 24 MiB above
its first active sample), and GPU allocations stayed flat. This short lifetime
is promising evidence for reduced diagnostic retention, not a full leak gate.

## Complete app run and remaining engineering checks

q670 ran the complete app workload for 4,037.1 seconds. All ten phases completed;
146 requests and 147 tool calls succeeded, with no protocol, native-output or
app-acceptance failures. The generated app's 74 tests passed, including Unicode,
validation, pagination, persistence and concurrent requests. Stochastic dynamic
MTP made 987 depth updates and 41,804 verifier runs. Its acceptance figures were
65.43% accepted/drafted and 79.96% accepted/stochastic accept tests.

Cumulative application throughput was 34.69 decode tok/s and 168.63 uncached
prefill tok/s. These are mixed, growing prompts up to roughly 142K tokens, not
the matched 512-token greedy benchmark above. Token-weighted prefix reuse was
98.636%; average TTFT was 6.729 seconds. Both GPU allocations remained fixed
near 20.375 and 19.736 GiB. RAM and disk cache tiers remained within their
configured capacities, including disk eviction and restore activity.

The full engineering result is still **failed**. The CPU follower exceeded its
unchanged initialized-PMA envelope: peak anonymous memory plus swap was
53,012,951,040 bytes against 52,532,149,431 allowed. Its final PerfStats contained
332,672 records, of which 214,116 were seven remote-projection completion
families keyed by changing epoch/expert coordinates. This establishes retained
diagnostic growth, but does not by itself attribute every byte of the RSS delta.
The scoped repair aggregates these families by stable topology/lane identity
and retains changing coordinates through the existing ordered sequence
evidence. A second shared helper aggregates distributed lifecycle summaries;
exact migration, physical-piece and retirement proof records remain unchanged.

The MTP evidence reader also filtered neighboring `forward_graph` parent proofs
while retaining that neighbor's MTP-domain sidecar records. Its CUDA:0 check
therefore reported CUDA:1 as lacking a parent despite saved native replay
evidence. The repair scopes both sides of the lifecycle join to the same owner;
the earlier complete graph/boundary validation remains unfiltered. Missing,
wrong-device and wrong-rank replay proofs remain failures. q672 reproduces this
exact validator error; q674 passes the affected complete Unit owner and both
explicit native-parent/hosted-MTP preflight entries.

q673 reproduces remote diagnostic growth with 448 records versus 42 fixed
topology records after 64 real MPI transfers. Its first distributed-summary
fixture stopped on an invalid manually edited transaction and is not accepted
as a defect reproduction. The corrected fixture uses three real authorities,
successive root-authored proposals, follower adoption, publication and retirement.
q677 reaches the intended old-code summary-cardinality/sequence failures.
q678/q679 rebuilt the affected Integration owners and Release tools. q681 passed
the complete distributed-residency Unit and MPI owners plus both explicit
diagnostic-lifetime preflight entries, with zero driver findings. No memory
allowance, transport policy or inference arithmetic changed.

q680 revalidates the unchanged q670 native records with the corrected MTP reader:
attention, graph, transfer and MTP checks all pass. It also observes two actual
restored-prefix suffix-decode transactions. This separate result preserves the
original engineering failure and does not repair its RSS failure. q676 was an
earlier report-serialization failure, not a native or validator failure.

The native q670 lifetime shut down normally without forced termination or
survivors. Driver diagnostics had zero findings. Its archive owner reclaimed
34,249,576,519 logical bytes without reading payloads. Separate authenticated
operator receipts also reclaim the retired q654/q662 archives while preserving
their failed results and logs. The original q670 failure must remain intact;
any revalidation of saved records is a separately labeled diagnostic.

The local resource reader also rescanned the complete final PerfStats once per
sample. q682 reproduces five versus twenty record traversals for two versus
seventeen otherwise-equivalent observations. The reader now parses immutable
startup namespace identities once; every sample still validates physical PID,
birth, namespace, ownership, retirement and all original resource limits. q683
passes all 55 local wrapper tests. This local tooling repair is not an engine
change, and the original q670 resource failure is unchanged.

## Fresh complete qualification after repairs

q686 completed all ten app phases in **2,355.354 seconds** (39.26 minutes).
The full owned lifetime, including startup and retirement audits, took
2,661.547 seconds. All 112 native requests completed, with zero failed or
disconnected requests. One of 105 tool attempts was a recoverable model-authored
`edit_match_not_found` error: **0.9524%**, below the unchanged strictly-less-than
5% gate. Protocol, phase coverage, stress gate, native-output validation and
independent webapp acceptance pass. The generated app's 53 tests pass. The
separate strict zero-task-error flag remains false and is preserved honestly;
it is not the configured stress acceptance policy. No compaction occurred.

The actual session produced 62,476 output tokens at **37.819 decode tok/s**.
Uncached prefill averaged **199.199 tok/s** across changing prompts; effective
prompt throughput including prefix reuse was 10,703.35 tok/s. Prefix-token
reuse was **98.1389%** and average TTFT **4.7317 seconds**. These stochastic
application measurements are not a matched code-speed comparison with q670
or with the short greedy benchmark. Dynamic MTP ran 22,887 verifiers and made
454 depth updates; accepted/drafted tokens were 70.333%, while the stochastic
accept-test rate was 84.500%.

All 457 resource samples pass. Peak CPU-follower anonymous RSS plus swap is
**52,474,941,440 bytes**, below its unchanged initialized-PMA envelope of
**52,531,440,823 bytes**. Root anonymous RSS plus swap peaks at 94,314,835,968
bytes, below 99,119,485,239 admitted. CUDA allocations remain fixed at
21,877,489,664 and 21,191,720,960 bytes. RAM/disk prefix capacity checks pass.
This qualifies the complete observed workload, not an unlimited-duration soak.

Native owners exited normally with no forced termination or survivors, and
the continuous driver window has zero findings. Shutdown persisted all 2,462
selected RAM entries: 2,248 payload writes and 214 existing-backing reuses.
The retired archive audit and cleanup pass; 2,702 regular files containing
34,237,540,760 logical bytes were reclaimed with zero payload bytes read.
All original red receipts and the unchanged admission envelopes remain retained.

This green slice satisfies the requested feature-branch checkpoint condition.
The next requested local workload is CUDA2 continuation with four MI50 expert
followers and RAM ngrams, again using actual automatic planning and dynamic MTP.
That topology requires its own admission, smoke, timing and full app evidence.
