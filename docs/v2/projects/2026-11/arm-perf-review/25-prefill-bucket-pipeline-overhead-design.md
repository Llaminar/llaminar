# Idea 25 design — Prefill bucket-pipeline overhead reduction

Status: **Design + implementation plan** (arm-p25design lane, 2026-11-12).
Source citations are pinned to the PR #43 tip `d5e04fdce` (the read-only
source reference worktree for this design); the tracker branch's `src/` tree
diverges at `2f01249c9`, so every `file:line` below must be read at that SHA.
Tracker inputs: `24-gb10-gpu-prefill-gap.md`, `25-prefill-bucket-pipeline-overhead.md`,
`25c-prefill-single-vs-async-design-notes.md`, `00-measured-baseline.md`,
`20-overlay-collection-methodology.md`, `plans/post-merge-performance-roadmap.md`.

## 1. Problem (measured, evidence-bound)

On DGX Spark (GB10, sm_121, CUDA, Release), all-CUDA:0 serving of
Qwen3.8-Flash-Next-UD-Q4_K_XL (177B MoE, ExpertOverlay runtime):

- Prefill **318.6 tok/s** at pp2160 vs llama.cpp **1063.24 tok/s** (3.3×).
- Captured stage kernel work = **0.60 ms/token** (GEMM 57% + GDN_PROJECTION
  39% of stage time) → stage-speed ceiling **~1660 tok/s**.
- Measured marginal cost **3.009 ms/token** plus a fixed **~279 ms** per-request
  intercept → ~80% of prefill wall time is not kernel math; it is the
  multi-bucket captured pipeline's per-boundary host/launch/sync overhead
  (ladder 512/448/384/256/128/64) plus the fixed intercept.
- The one completed structure trace (K14-IDEA25-TRACE-STRUCT-20261109,
  cff351901, demoted kernels — ratio is the structural point, not the
  absolute numbers): host launch per bucket replay is only **39–78 µs**
  (64→78.2, 256→47.7, 512→38.8), GPU replay per bucket 63.5/191.1/340.2 ms
  (~0.66 ms/token plus a fixed ~20 ms per replay), while prefill wall was
  ~46 s against ~3.83 s of summed GPU replay → **~91% of the wall is
  host-side work or gaps between chunk submissions**, not GPU replay and
  not the launch call itself.
- llama.cpp prefills the same prompt single-pass at 0.94 ms/token.

Goal: close the gap toward the kernel-supported ceiling by (A) letting one
captured bucket hold a long prompt and (B) removing the serial host frontier
between bucket submissions — with a canonical, non-attached measurement that
attributes every millisecond first. **Operator acceptance bar: same-or-better
performance everywhere; x86 must stay the same or improve; anything worse is
rejected** (`post-merge-performance-roadmap.md`).

## 2. How GPU prefill executes today (code map at d5e04fdce)

### 2.1 Ladder selection

- The complete captured-bucket inventory is single-sourced in
  `src/v2/utils/PrefillGraphBuckets.def`: 64…512 in fine steps, then
  544…768, 1024, 1280, 1536, 2048, 2560, 3072, **4096**. The header comment
  states the design intent verbatim: *"long prompts are chunked through these
  captures and therefore do not create larger kernel-dispatch buckets."*
- The **serving default cap is 512 rows**: `kDefaultPrefillGraphMaxBucketSize`
  (`src/v2/utils/PrefillGraphBucketDefaults.h:60`), with the comment that
  larger shapes *"remain supported and trained, but must be requested instead
  of consuming the default activation/workspace budget on every serving
  instance"* (lines 53–59). `defaultPrefillGraphBucketSizes()` =
  `prefillGraphBucketSizes(512)` = {64,128,256,384,448,512} — exactly the
  measured GB10 ladder.
- Configuration surfaces that already exist:
  - `--prefill-max-bucket-size` (`src/v2/config/OrchestrationConfigParser.cpp:1435`)
    → publishes an explicit `LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES` override
    (`src/v2/config/OrchestrationStartupPolicy.cpp:63-67`;
    `src/v2/utils/DebugEnv.h:1459-1478`).
  - `--moe-overlay-prefill-segment-rows` /
    YAML `moe.overlay_prefill_segment_rows`
    (`src/v2/config/OrchestrationConfigParser.cpp:2102-2112`, `:475-480`)
    → `RuntimeConfig.moe_routed_prefill.overlay_segment_rows`.
  - `LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS` / `prefill_graph_max_cached_buckets`
    (`src/v2/utils/DebugEnv.h:1192-1193`), default
    `kDefaultPrefillGraphMaxCachedEntries = 10`
    (`src/v2/utils/PrefillGraphBucketDefaults.h:69`) with 2 identities
    reserved for request-specific/prefix rehydration captures
    (`kExpertOverlayPrefillGraphIdentityReserve`, `src/v2/utils/PrefillGraphBucketDefaults.h:80`).
- **Admission chain (ExpertOverlay/MoE path — the measured configuration):**
  the retained ladder is derived from the *admitted segment capacity*:
  `segment_rows = resolvePrefillScheduleRowCapacity(resident_prefill_rows,
  plan.runtime.moe_routed_prefill.overlay_segment_rows)`
  (`src/v2/planning/MoEOverlayMemoryPlanInputs.cpp:360-361`), then
  `retainedRawPrefillGraphBucketLadder(configured_buckets, segment_rows,
  minimum_sequence_rows, cache_entries − reserve)` (`:366-368`). The serving
  runner publishes the same ladder at request time
  (`src/v2/execution/runner/OrchestrationRunner.cpp:22738`). Automatic
  candidate admission tries the throughput-favouring (largest) shape first
  and only reduces residency when the complete physical BOM cannot cover it
  (`src/v2/execution/local_execution/engine/PrefillBucketUtils.cpp:96-114`,
  consumed at `src/v2/planning/OrchestrationCandidateAdmission.cpp:270-274`).
  A dense/non-overlay deployment uses the ordinary ladder from
  `--prefill-max-bucket-size` instead.
- **Chunk planning:** `planPrefillChunkSchedule`
  (`src/v2/execution/local_execution/engine/PrefillBucketUtils.cpp:352-465`)
  fills with `chunk_target = max_bucket` and selects the smallest bucket ≥ the
  chunk's live rows. A 2160-token prompt on the default ladder is
  512+512+512+512+128 (tail 112 real rows padded to 128) = **5 chunks**.
  Serving routes every mathematically-prefill request through this scheduler
  (`src/v2/execution/runner/OrchestrationRunner.cpp:3406-3424`) into
  `DeviceGraphOrchestrator::forwardPrefillChunkSchedule`
  (`src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp:19204`).

### 2.2 Setup: the ladder is captured once, not per request

`materialize_prefill_family`
(`src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestratorServingGraphFamily.cpp:598-713`)
captures **every** bucket of the normalized ladder at serving setup with
`MaterializeExecutableWithoutLaunch` and counts it via
`forward_graph/serving_graph_family_materializations` (`:696-704`). Per-request
capture is therefore not part of the design; if it appears at runtime it is a
defect, detectable via the lifecycle counters (§3.3).

### 2.3 The serial per-chunk loop and its host boundary

`ForwardExecutionEngine::runPrefillChunkSchedule`
(`src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp:1543`)
validates the complete immutable chunk family (bucket identity, device/state
publication stream, sparse-query proofs, exact graph identity per chunk,
`:1573-1648`), then runs chunks **strictly in sequence** (loop at `:1650`,
`runPrefillChunk` call at `:1655`). Each
`runPrefillChunk` (`:1408`) performs, in order:

1. per-chunk snapshot-diagnostics scope (diagnostic-only, `:1441-1442`);
2. `makeBucketedPrefillInput` (`:496-570`) — host-side chunk input adapter;
3. lease begin: `host.beginPrefillChunkGraphSubmission`
   (`DeviceGraphOrchestrator.cpp:5593`). On a non-overlay deployment this
   returns trivially with no lease (`:5610-5611`). On the ExpertOverlay path
   it calls `admitSerialPrefillGraph`
   (`src/v2/execution/moe/MoEOverlayInferenceTransactionService.cpp:1735`) —
   rank-local host bookkeeping with a condition-variable wait only for LocalTP
   sibling alignment (CV at `:1773`; standard 30 s collective timeout), then binds
   `MoEOverlayPrefillChunkGraphSubmissionLease`
   (`DeviceGraphOrchestrator.cpp:3826`): it stamps wire identity — request
   generation, step id, sequence id, placement epoch, **sequence graph
   ordinal/count** (`:3884-3901`) — and arms the remote-follower ticket at the
   *executable launch edge* (`:3906-3922`), which is already an asynchronous
   launch-time callback;
4. `execute()` (`ForwardExecutionEngine.cpp:1465`) → cache-hit path
   (`:3880-4056`): `prepareLiveStateForForwardGraphExecution` (`:3914`),
   `prepareDeviceTokenInputsForForwardGraphExecution` (`:3928`),
   `updatePrefillReplayParamStages` (`:3964`, impl `:608`),
   `updateMoEOverlayCollectiveRuntimeStages` (`:3972`),
   `updateDynamicParamStages` (`:3977`) — the doc comment (`:644-652`)
   states stages *stamp their stable device parameter buffers* on an exact
   stream — then `forward_cache.graph->reset()` (`:3996`),
   `waitBeforeForwardGraphExecution` (`:4004`), and the replay:
   `executePrefillWithGraphCache` (`:4675`). The replay path
   (`:5358-5436`) binds stages to the explicit prefill capture stream, calls
   `launchPrefillGraph` → `cache.launch(key)` (`:4929`), publishes captured
   terminal state (`:5097`, `:5413`) and snapshots (`:5420`), and returns —
   **no host synchronization in the unprofiled replay path**;
5. lease finish (`:1467-1482`) — host bookkeeping/publication;
6. **maintenance boundary**: `host.prefillChunkMaintenanceState` (`:1483`,
   impl `DeviceGraphOrchestrator.cpp:5459-5477`) →
   `evaluatePrefillChunkMaintenance` (call at `:1485`,
   `PrefillBucketUtils.cpp:467-520`) → `host.onPrefillChunkMaintenance`
   (call at `:1498`, impl `DeviceGraphOrchestrator.cpp:5483`).

### 2.4 The hidden per-chunk device wait: the maintenance histogram sync

`prefillChunkMaintenanceState` always calls
`DecodeExpertHistogram::syncRuntimeHistograms()`
(`DeviceGraphOrchestrator.cpp:5470`; `src/v2/execution/moe/DecodeExpertHistogram.cpp:1324-1333`)
which runs the registered sync callbacks. The Qwen MoE graph registers
`table->syncDecodeHistogramToHost(...)` on the decode-histogram producer
stream (`src/v2/models/qwen35moe/Qwen35MoEGraph.cpp:4687-4696`). On the
GPU-mirrored runtime table this performs, **per model layer**, six D2H copies
(3 histogram sources × selected+local,
`src/v2/execution/moe/MoERuntimeTable.cpp:1997-2006`) — 48 layers → **288
small D2H copies per chunk boundary** — each through
`copyMirrorToHost → backend->deviceToHostFast`
(`MoERuntimeTable.cpp:214-220`). CUDA has no `deviceToHostFast` override, so
it falls to the base `deviceToHost`, whose contract is *"enqueue the copy,
record one exact completion event, and wait only that event before returning
host-owned bytes"* (`src/v2/backends/IBackend.h:310-329`). Because the copies
are ordered on the producer stream after the just-launched bucket replay, **the
first waited event drains the entire preceding replay**; the host then blocks
per copy. This is the structural serialization the trace measured as
"chunk wall ends long after its own launch returns" (idea 25c): it is a hidden
device sync on the prefill critical path, in tension with the events-not-syncs
maxim (AGENTS.md #11) even though each individual wait is the sanctioned
compatibility boundary for raw buffers.

Additionally, if the merged histograms make `shouldRebalance()` true,
`onPrefillChunkMaintenance` runs the heavy path: expert transfers, mask
application, **`forward_engine_->discardAllCachedGraphs()`**
(`DeviceGraphOrchestrator.cpp:5527-5545`) — the next chunk would recapture
(seconds-scale). That is a defect-class cost if it fires mid-prefill; the
lifecycle counters must prove it does not (§3.3).

### 2.5 What is already timed (zero added sync)

- `forward_graph/prefill_graph_launch` — host wall of `cache.launch()` per
  replay, with bucket/chunk/phase/launch_kind tags
  (`ForwardExecutionEngine.cpp:4955-4976`).
- `stage_gpu/prefill_graph.replay` — GPU event time per replay, **profiling
  mode only** (gated by `PerfStatsCollector::gpuStageEventTimingEnabled()`,
  `src/v2/utils/PerfStatsCollector.cpp:792`, env
  `LLAMINAR_GPU_STAGE_TIMING`/`LLAMINAR_PERF_STATS_GPU_STAGE_TIMING`,
  `src/v2/utils/DebugEnv.h:3856`, `PerfStatsCollector.cpp:361-368`); when
  enabled it synchronizes each replay
  (`synchronizeEvent`, `ForwardExecutionEngine.cpp:4989`, tagged
  `sync_scope=profiling_event_synchronized`) — usable for GPU-time
  attribution, never for economy numbers.
- `forward_graph/prefill_chunk_host_wall` + `prefill_inter_chunk_gap`
  (`recordPrefillChunkHostTiming`, `ForwardExecutionEngine.cpp:1519-1541`,
  called `:1665-1667`) and `forward_graph/prefill_family_host_wall`
  (`:1683-1691`) — host wall clocks only; the loop's comment is explicit:
  *"no event, synchronization or device read is added, so the unprofiled
  pipeline keeps its overlap"* (`:1637-1639`).
- `forward_graph/prefill_graph_lifecycle` + `prefill_graph_phase` with
  `capture_phase`, `cache_phase`, `bucket_seq_len`, `recapture_reason` tags
  (`publishPrefillGraphObservation`, `ForwardExecutionEngine.cpp:1003-1044`).
- `moe_rebalance/*` counters around histogram merges and rebalance decisions
  (`MoERuntimeTable.cpp:2053-2105`).
- CSV/JSON export: `LLAMINAR_PERF_STATS_CSV`/`LLAMINAR_PERF_STATS_JSON`
  (`src/v2/utils/PerfStatsCollector.cpp:283-298`).

## 3. Scope 1 — Canonical non-attached measurement design

**Hard constraint:** attached GPU profiling (NCU/NSYS/CUPTI) is on a safety
hold (NVRM refcount warnings observed on `.79`; nsys attachment is
candidate-clean only, not canonically certified — tracker idea 25). The
measurement must rely exclusively on PerfStats counters/timers that either add
nothing (counters) or host wall clocks (no events, no syncs, no device reads),
plus the existing profiling-gated GPU event timers used *only* for GPU-time
attribution runs.

### 3.1 Decomposition identity

For one prefill request, with the family timer bracketing the chunk loop:

```
request_prefill_wall                       (benchmark-visible wall)
 = fixed_intercept                         (§6)
 + prefill_family_host_wall                 (existing counter)
prefill_family_host_wall
 = Σ_chunks prefill_chunk_host_wall         (existing counter)
 = Σ_chunks [ lease_begin + param_staging + launch
              + terminal_publish + lease_finish
              + maintenance_state + maintenance_run ]   (§3.4 additions)
 + Σ_boundaries prefill_inter_chunk_gap      (existing counter)
GPU attribution (separate, profiling run):
 Σ_chunks stage_gpu/prefill_graph.replay    (existing, syncs per replay)
unprofiled launch host cost:
 Σ_chunks forward_graph/prefill_graph_launch (existing)
```

The per-token pipeline overhead is then
`(marginal_ms_per_token − stage_kernel_ms_per_token)` from the multi-length
benchmark regression (existing methodology, ideas 24/25: 2160/7200-token
slopes), and the counters attribute it to named host terms. The decision rule
from idea 25c applies on top: family wall ≈ Σ GPU replay → not host-bound
(kernel/occupancy work instead); gap/host-bound ≥ 50% → Fix B; large
per-chunk constants → Fix A.

### 3.2 Modes

1. **Canonical economy run (unprofiled):** Release binary, production
   settings, `LLAMINAR_PERF_STATS_CSV=...` only. This is the only mode whose
   tok/s numbers may be compared/reported (AGENTS.md #23/#27: profiler
   overhead stays out of canonical timing; PerfStats is the production truth
   #26).
2. **GPU-attribution run (profiling-gated):** add
   `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` to obtain per-replay GPU event
   times. Wall numbers from this mode are explicitly non-comparable
   (`sync_scope=profiling_event_synchronized`); it exists to measure Σ GPU
   replay and the wall−GPU split. Still no profiler attachment.
3. **Intercept run:** benchmark with ≥3 prompt lengths (e.g. 120/2160/7200
   tokens, same model/seed/temperature) — the existing slope/intercept
   regression — now with the family wall counter separating pipeline cost from
   the fixed intercept exactly (§6).

### 3.3 What the existing counters already answer

- Host-bound or not: `prefill_family_host_wall` vs
  Σ `prefill_graph_launch` vs (attribution run) Σ `prefill_graph.replay`.
- Which chunks/buckets cost: per-chunk walls with `bucket_seq_len` tags.
- Per-request capture/recapture: `prefill_graph_lifecycle`/`prefill_graph_phase`
  with `cache_phase`≠replay or non-`none` `recapture_reason` mid-request →
  defect (setup materialization §2.2 should make every request pure replay +
  `transaction_zero_after_setup`).
- Maintenance outcome: `moe_rebalance` counters (merge outcomes, slots);
  a heavy rebalance mid-prefill is visible via
  `prefill_graph_lifecycle` recapture after a chunk.

### 3.4 Required additions (host wall clocks / counters only)

All follow the existing discipline of `recordPrefillChunkHostTiming`
(`ForwardExecutionEngine.cpp:1519`): `timing_scope=host_wall`,
`sync_scope=none`, sequence words = (chunk index, token start, token end),
tags = bucket/real counts. None may add an event, sync, or device read.

| Timer/counter (domain) | Measures | Placement |
|---|---|---|
| `forward_graph/prefill_lease_begin_ns` | lease admission+bind host wall | around `beginPrefillChunkGraphSubmission` (`ForwardExecutionEngine.cpp:1448-1463`) |
| `forward_graph/prefill_param_staging_ns` | live-state prep + token inputs + replay/dynamic/overlay param staging + graph reset | the `execute()` prelude (`:3914-4004`) |
| `forward_graph/prefill_launch_publish_ns` | launch + terminal-state/snapshot publication enqueue | replay path (`:5386-5428`) |
| `forward_graph/prefill_lease_finish_ns` | lease finish host wall | `:1467-1482` |
| `forward_graph/prefill_maintenance_state_ns` | histogram sync + rebalance probe host wall (the §2.4 wait lands here) | around `prefillChunkMaintenanceState` (`:1483`) |
| `forward_graph/prefill_maintenance_run_ns` | executed maintenance (rebalance/transfer) host wall | around `onPrefillChunkMaintenance` (`:1496-1506`) |
| `moe_rebalance/runtime_histogram_sync_copies` (counter) | D2H copies + event waits per sync (288/boundary expected) | `DeviceMoERuntimeTable::syncDecodeHistogramToHost` (`MoERuntimeTable.cpp:1880`) |
| `forward_graph/prefill_request_admission_ns` | request-wall − family-wall decomposition input at the runner boundary | `OrchestrationRunner` prefill entry (§6) |

Expected outcome from the structure trace: the maintenance-state term and the
inter-chunk gaps dominate; param staging is the second candidate. The
decomposition is *complete by construction* (the seven terms + gap sum to the
family wall), so a Phase-0 gate is: **≥95% of `prefill_family_host_wall`
attributed to named terms** on the GB10 cell, unprofiled.

### 3.5 Anti-requirements

- No NCU/NSYS/CUPTI attachment anywhere in the canonical path (safety hold).
- No new device synchronization, host readback, or event in unprofiled runs
  (the timers are `steady_clock` deltas only; AGENTS.md #11/#26).
- Profiling-mode numbers never enter economy comparisons or the README chart.
- The runs execute on `.79` through arm-perf → arm-sparks scheduling only
  (no unilateral host use; merge window active).

## 4. Scope 2 — Fix A: single-bucket long-prompt path

### 4.1 What "one capture holds it" means

The bucket inventory already contains 1024…4096-row captures
(`PrefillGraphBuckets.def`). "One capture holds it" = the **admitted prefill
segment capacity** (§2.1 admission chain) is raised so a long prompt maps to
one (or two) chunk(s) instead of ~5: e.g. segment rows 2048 → a 2160-token
prompt becomes 2048+112(→128) = 2 chunks; segment rows 2560 → 1 chunk
(2160 real rows padded into a 2560-row capture). Chunk count scales as
`ceil(prompt/segment)`; every per-chunk boundary cost from §2.3–2.4
(lease, staging, launch, terminal publication, histogram sync with its
embedded replay-drain wait, maintenance evaluation) is paid per chunk, so the
host-bound fraction shrinks proportionally. The per-replay fixed GPU cost
(~20 ms/replay in the structure trace) is also amortized.

This is **primarily an admission/policy change, not new machinery**: the
captures, the ladder DP, the memory-plan inputs and the candidate search all
already support larger shapes. The work is (a) explicit larger segments
through the existing `--moe-overlay-prefill-segment-rows` /
`--prefill-max-bucket-size` surfaces, (b) making the larger segment an
*admitted* choice (PMA BOM), and (c) a measured policy decision about when it
should be selected by default.

### 4.2 Memory/BOM analysis (the real constraint)

- Activation/workspace banks scale with the segment's row capacity: the
  retained ladder enters the overlay BOM via
  `MoEOverlayMemoryPlanInputs` (`src/v2/planning/MoEOverlayMemoryPlanInputs.cpp:360-368`;
  arch doc: *"Prefill segment rows cannot shrink retained verifier capacity"*).
  Quantized projection workspace does **not** scale with rows: grouped tiles
  are bounded at 128 rows/tile and the KPAR scratch at 512 MiB
  (`kDefaultNativeQuantisedBatchInvariantTileRows`,
  `kDefaultNativeQuantisedBatchInvariantScratchBytes`,
  `src/v2/utils/PrefillGraphBucketDefaults.h:134-150`). Graph-pool memory and
  setup capture time scale with the *number and size of retained buckets*
  (`GPUGraphMemoryContract` retained families; the ladder DP bounds the count
  to `cache_entries − reserve` via `retainedPrefillGraphBucketLadder`,
  `PrefillBucketUtils.cpp:157-244`).
- KV demand is unchanged: a single 2048-row chunk appends the same KV bytes
  as 4×512; capacity comes from the context limit, not the segment.
- On GB10 (UMA, 121 GB shared, ~102 GB resident for the 177B model), the
  activation delta 512→2048 must be **measured via the BOM builder and
  admitted by `PhysicalMemoryAuthority`** (maxims #2/#15). Two possible
  outcomes, both valid: it fits → pilot measures the win; it does not fit →
  the typed `PhysicalMemoryCapacityExhausted` rejection is the honest
  boundary (no silent ladder fallback — maxim #1; exactly the honest-NoFit
  behaviour already produced by the #74 planning fix on this hardware).
- Padding economy: a padded bucket computes dead rows (2560-padded capture for
  2160 real rows wastes ~18% of that chunk's GEMM work). The ladder DP
  minimizes worst-case padding (`PrefillBucketUtils.cpp:157-244`); the
  `ExactRetained` width family (`PrefillBucketUtils.cpp:23-42`,
  `exactPrefillGraphWidths` `:44-58`) exists for exact-width pipelines but is
  scoped to the fixed-head ngram path
  (`OrchestrationRunner.cpp:3420-3424`) and requires one capture per power-of-
  two width — request-specific exact captures would violate no-recapture and
  are **not** part of this design.

### 4.3 Invariant evaluation (AGENTS.md maxims)

- **Capture identity completeness (#13):** a larger segment is a new
  `PrefillGraphCacheKey.seq_len` with unchanged domain/participant/placement/
  topology identity (`makePrefillGraphKey`, `ForwardExecutionEngine.cpp:938-948`);
  the family is captured at setup (§2.2), so requests replay — no identity
  churn.
- **No recapture per request (#14):** unchanged; setup materialization covers
  the larger ladder. `resetSessionReplayState` (`ForwardExecutionEngine.cpp:1748`)
  already separates data reset from topology.
- **Request reset resets data, not topology:** unchanged (existing replay
  param restamp per chunk, `updatePrefillReplayParams` — e.g.
  `KVCacheAppendStage.h:271-321`).
- **Arena/BOM capacity (#15):** the segment raise must flow through
  `MoEOverlayMemoryPlanInputs` (or the ordinary planner for dense) and be
  admitted; no anonymous reserve, no second ledger.
- **No host readbacks (#8/#12):** unchanged — chunk bindings and replay
  scalars are host-known at admission; the captured materializer reads device
  KV counts on-device (comment at `ForwardExecutionEngine.cpp:1570-1572`:
  *"each captured materializer still reads the very same device KV count"*).
- **Events not syncs (#11):** unchanged for Fix A itself; the histogram-sync
  wait persists once per (now fewer) chunk — Fix B removes it.
- **Communication volume / launch work:** a padded large bucket computes dead
  rows inside one capture — this is the existing, admitted padded-bucket
  contract (row-select gates, KV append gated by real counts); it does not
  communicate unused capacity (single device). The GPU launch grid covers the
  padded bucket, which is the existing bucket semantics, not an oversized
  *launch* (the capture's own geometry).

### 4.4 Maintenance-boundary decision

With one chunk, the mid-prefill rebalance boundary disappears by construction
(only the family-end boundary remains). This is a *policy simplification*, not
a lost obligation: the residency-maintenance window is token-based
(`min_rebalance_interval_tokens` = ladder max, `PrefillBucketUtils.cpp:37`)
and prefill-chunk histograms merge per chunk; a single chunk merges once.
The design decision to record: **prefill maintenance remains evaluated at
every chunk boundary and at family end; with N=1 the two coincide.** No
special case is added.

### 4.5 Design decision and gates

- Pilot (no code change): GB10 cell, `--moe-overlay-prefill-segment-rows
  {1024,2048}` (and the dense-surface analog on a dense model), same
  model/prompt/seed/temperature as idea 24. Gate: typed admission result;
  byte-identical output; prefill tok/s; BOM/memory delta recorded; PerfStats
  proves pure replay (`prefill_graph_lifecycle`) and unchanged decode.
- Production change (if the pilot wins): consider default policy (e.g.
  auto-selection weighing `--plan-workload` prefill horizon against the BOM),
  keeping the explicit override authoritative. Any default change rides the
  full x86 gate (§7) and docs/help sync (maxim #38).
- Risk register: GB10 UMA capacity (typed NoFit is acceptable and recorded);
  padding waste for adversarial lengths just above a bucket (mitigated by the
  ladder DP and by keeping mid-ladder buckets resident); setup-time growth
  from more/larger captures (bounded by the cache-entries budget); larger
  worst-case activation may reduce KV/expert headroom on smaller GPUs (the
  candidate search already trades this off, §2.1).

## 5. Scope 3 — Fix B: async multi-bucket submission

### 5.1 What is serial today vs already async

The replays are already asynchronous, single-stream and event-ordered: launch
returns without waiting (`ForwardExecutionEngine.cpp:4929`), terminal state is
published on the producer stream (`:5097`), and the unprofiled path has no
sync. What serializes the family is the **host boundary between chunks**:
param staging + graph reset (host work, no wait), lease begin/finish (host
bookkeeping), and — dominantly — the maintenance histogram sync whose first
event wait drains the previous replay (§2.4). The structure trace's ~91%
host-side fraction is exactly this window.

### 5.2 Target structure

Keep the bucket ladder and the per-chunk captured graphs; submit the whole
family back-to-back and join once at the end:

1. **Family-scoped lease.** Admit the chunk family once instead of per chunk:
   the coordinator already models ordered graph groups with per-graph
   ordinals inside one sequence (`moe_overlay_sequence_graph_ordinal/count`
   stamped at `DeviceGraphOrchestrator.cpp:3894-3901`;
   `admitSerialPrefillGraph`'s group slots,
   `MoEOverlayInferenceTransactionService.cpp:1735-1820`). Extend this to a
   typed *family admission* covering all chunk graphs of one prefill request,
   preserving LocalTP sibling alignment semantics (the CV wait exists to keep
   siblings in the same graph group — a family admission keeps that
   invariant with one alignment instead of N). Per-chunk ticket arming stays
   at the executable launch edge (already an async callback, `:3906-3922`).
2. **Deferred maintenance decision.** Do not run the blocking
   `syncRuntimeHistograms` between submissions. The histogram machinery
   already has a non-blocking drain protocol
   (`registerRuntimeHistogramDrain` / `progressRuntimeHistogramDrains`,
   `src/v2/execution/moe/DecodeExpertHistogram.cpp:1338-1404`): enqueue the
   D2H copies on the producer stream during submission (ordered after each
   chunk's replay — correct by stream order), poll/complete the drains while
   the GPU works, and evaluate **one merged maintenance decision at the
   family boundary**. Prefill-chunk histogram counts are additive per source
   (`ExpertHistogramSource::PrefillChunk`), so merging once at family end is
   arithmetically identical to per-chunk merges; the rebalance-interval
   contract (`min_rebalance_interval_tokens`, `PrefillBucketUtils.cpp:37`)
   is preserved by evaluating readiness at family end with the family's total
   tokens. This must be a **typed policy change** (prefill maintenance =
   family boundary), not a silent skip — maxim #3/#29.
3. **Submission loop restructure.** In `runPrefillChunkSchedule`, replace the
   execute→maintenance serialization with: for each chunk — validate (family
   validation already happens up front, `ForwardExecutionEngine.cpp:1573-1648`),
   stage params on the param stream, event-join param stream → replay stream,
   launch, publish terminal state; then one family-end join (existing request
   teardown joins outstanding device work — arch doc §8: *"Request reset …
   joins outstanding device work"*). Param buffers are shared per captured
   bucket; staging for chunk N+1 must be **stream-ordered after replay N**
   (single-buffer + ordered join is the design — it preserves the captured
   addresses that are part of graph identity, maxim #13; double-buffered
   publication from idea 25c is only needed if staging must overlap replay
   execution, which the param-stream/event structure already permits without
   identity churn).
4. **Failure fencing.** Host-side launch failure at chunk k stops submitting
   k+1.. (existing early-return); GPU-side failures surface at the family-end
   join via the executor's stage-failure publication, and the request
   teardown joins outstanding work before any state reuse. No partial-family
   state may be published; no retry or alternate path (fail fast, maxim #29).

### 5.3 CUDA graph feasibility

No conditional nodes are required. Multiple whole-graph replays enqueued
back-to-back on one stream are natively ordered; the per-chunk graphs are
distinct captures (or repeated replays of the same capture with restaged
params). The param-stream → replay-stream edge is an ordinary event join.
This is exactly the retained-parent/sequential-replay structure the capture
controller already manages (`executePrefillWithGraphCache`'s replay plan and
launch-dependency composition, `ForwardExecutionEngine.cpp:4675-4900`).
HIP/ROCm symmetry: the same structure applies (HIP also has no conditional
nodes here — none are needed); the `defer_final_sync` retained-parent path
(`:4881`) is the existing precedent for end-joined submission.

### 5.4 Correctness gates

- **KV position ordering:** each chunk's captured materializer reads the
  device KV count on-device (`ForwardExecutionEngine.cpp:1570-1572`); replays
  are stream-ordered; the staged `token_offset/real_count` scalars are
  host-known from the immutable schedule (validated up front, `:1573-1648`).
  A focused regression must assert exact appended KV extents per chunk and
  byte-identical full-prompt outputs vs the serial path (greedy, fixed seed).
- **Publication:** per-chunk terminal-state publication stays on the producer
  stream (last chunk's terminal row is the request result); no new host
  visibility is added mid-family.
- **Replay regressions (maxim #22):** register a ProductionTestPreflight
  replay test for the family-submission state machine (recorded inputs +
  initial state through the production implementation, reset + repeated
  replay + partial-family failure), device-free where possible (schedule +
  lease + maintenance-decision logic), with the native capture/replay
  exercised in the explicit CUDA/ROCm integration suites.
- **PerfStats proof (#26):** new counters must prove the async path ran:
  e.g. `forward_graph/prefill_family_submission` with per-chunk launch
  timestamps, plus an overlap witness (host launch time of chunk N+1 earlier
  than GPU completion of chunk N — derivable in the attribution run without
  new syncs). Unprofiled counters: chunk count, per-chunk walls, family wall.
- **Economy gate:** family wall ≈ Σ GPU replay + residual host terms; the
  §3 decomposition must show the maintenance-state term absent from the
  critical path (it moves to the family boundary).

### 5.5 Risks

- Protocol complexity in the overlay transaction service (family admission
  must not weaken the serial-alignment invariant for LocalTP siblings or the
  placement-epoch/`discardAllCachedGraphs` interaction at rebalance time:
  a family-end rebalance that invalidates graphs must still be followed by
  honest recapture accounting, never mid-family reuse of stale captures).
- Deferred rebalance changes expert-placement timing for long prompts (one
  decision point instead of ~5). Bounded by the maintenance-window policy;
  A/B on end-to-end tok/s (§7) must confirm no decode-phase regression from
  changed placement timing.
- Snapshot diagnostics per chunk (namespace sequencing,
  `DeviceGraphOrchestrator.cpp:5697-5775`) are host-RAII around *submission*,
  not execution — unchanged, but the replay test must cover
  snapshot-enabled runs (diagnostic-only).
- Any accidental new sync or readback is a hard review failure (source
  sanitizers + §3 timers make it visible).

## 6. Scope 4 — the ~279 ms fixed intercept

The intercept is everything outside the family wall. With the existing
`prefill_family_host_wall` counter, the split is direct:
`intercept = request_prefill_wall − prefill_family_host_wall − (terminal
result wait)`. The measurement design:

1. Add `forward_graph/prefill_request_admission_ns` at the runner prefill
   boundary (host wall, no sync) covering: tokenization, prefix-cache lookup
   and restore decision, request admission, `admitRequestInputsOnDevice`
   (one H2D of the whole prompt, `DeviceGraphOrchestrator.cpp:19481`), chunk
   binding construction (`makeDevicePrefillChunkGraphBinding`,
   `:19455`/`:18997`), schedule construction, and overlay request-transaction
   admission.
2. Separately time the **terminal result wait** (the one legal host boundary:
  final logits D2H + event wait) and the **first-chunk frontier** (any
  pre-launch device readiness join).
3. Benchmark methodology (already established): ≥3 prompt lengths, same
   model/seed/temperature, Release; regress wall vs tokens; report
   slope (ms/token) and intercept with the counter-based decomposition
   attached. The intercept must be classified: admission vs staging vs
   capture amortization vs terminal wait. Per-request *capture* (as opposed
   to replay) inside the intercept is a defect (§3.3) — the lifecycle
   counters decide.
4. Expected composition (to be measured, not assumed): with setup-time
   materialization (§2.2), capture amortization should be ~0 per request;
   the ~279 ms is then admission/staging/terminal dominated — a separate
   optimization target that Fix A/B must not be charged for.

## 7. Scope 5 — x86 non-regression gate

The changed code (`ForwardExecutionEngine`, `PrefillBucketUtils`,
`DeviceGraphOrchestrator`, `MoEOverlayInferenceTransactionService`,
`DecodeExpertHistogram`/`MoERuntimeTable`, ladder/BOM policy) is shared
engine code on every backend. The operator's bar: **x86 same-or-better,
anything worse rejected**; for shared-pipeline architecture the x86 GPU A/B
is part of the gate (roadmap, x86 protection classes).

Gate design (per phase, before anything lands):

- **Build/binary pair:** A = exact pre-change SHA, B = exact post-change SHA,
  alternating X/Y runs (2×2 minimum), mode-asserted via PerfStats counters
  (the existing A/B discipline from the ARM program; both arms must show
  `prefill_graph_lifecycle` replay counts and captured-path counters).
- **Topologies:**
  1. x86 CUDA single-GPU **MoE/ExpertOverlay** serving (the measured GB10
     configuration's analog — exercises the lease, maintenance deferral and
     overlay admission);
  2. x86 CUDA single-GPU **dense** (non-overlay lease path,
     `beginPrefillChunkGraphSubmission` trivial return,
     `DeviceGraphOrchestrator.cpp:5610-5611`);
  3. **CPU-only** serving cell (the chunk schedule and maintenance boundary
     also run on the stage-synchronous CPU path,
     `ForwardExecutionEngine.cpp:4071-4077` — deferred maintenance changes
     CPU rebalance cadence too);
  4. **ROCm** equivalent where a qualified cell exists (backend symmetry,
     maxim #20);
  5. multi-GPU rank-local TP overlay if available (exercises
     `admitSerialPrefillGraph` sibling alignment under the family lease).
- **Workloads:** the canonical benchmark subcommand with a saved plan
  (`--config`), fixed long-prompt prefill cell (prefill-dominated, e.g.
  ≥2048-token prompt, small `-n`), plus the standard decode cell; greedy
  (`--temperature 0 --seed 42`); ≥5 repeats, mean±sd. Metrics: prefill
  tok/s, decode tok/s, text byte-identity, memory footprint (BOM delta for
  Fix A), and the §3 counter set proving the intended path (pure replay, no
  per-request recapture, no non-terminal D2H, maintenance decisions at the
  documented boundary).
- **Bar:** prefill same-or-better on every measured cell; decode same-or-better
  (maintenance deferral must not regress decode-phase placement economics);
  byte-identical text; anything worse = rejected/reworked, no averaging away.
- **Registration:** device-free Unit tests for schedule/policy/decision logic
  (PrefillBucketUtils is already pure, `PrefillBucketUtils.cpp:8-12`); the
  focused replay regressions of §5.4 in `ProductionTestPreflight`; source
  sanitizers stay green (no new default streams/blocking syncs/host
  callbacks); canonical E2E + published-image suites and the benchmark
  phase per `docs/production-ci.md` cover the real-model cells; help/README/
  AGENTS sync for any new flag semantics (maxim #38).

## 8. Phased implementation plan

**Phase 0 — non-attached decomposition (counters only, no behavior change).**
Add the §3.4 timers/counters (+ §6 request-admission timer); device-free Unit
coverage for the timer taxonomy; run the canonical decomposition on `.79`
(via arm-perf → arm-sparks). *Gates:* ≥95% of family wall attributed;
recapture/maintenance verdicts recorded; intercept classified; the 25c
decision rule applied (A vs B vs both). *Risks:* none structural (host wall
clocks only); low.

**Phase 1 — Fix A pilot through existing surfaces (no code change).**
`--moe-overlay-prefill-segment-rows {1024,2048}` (+ dense-surface analog) on
the GB10 cell; measure admission, tok/s, byte-identity, memory. *Gates:*
typed admission outcome; prefill improvement quantified; x86 A/B on the same
surfaces neutral-or-better (config-only, but the ladder/BOM path is exercised);
docs note. *Risks:* GB10 NoFit (recorded as the hardware boundary); padding
waste. *Optional follow-up code:* default/auto policy for segment selection —
full gates of §7 before any default change.

**Phase 2 — Fix B structural change.**
Ordered sub-steps, each independently gated:
(a) **maintenance deferral** (drain-API based family-boundary merge + typed
policy) — the largest single win per §2.4; CPU A/B included;
(b) **family-scoped overlay lease** (typed coordinator extension, LocalTP
alignment preserved);
(c) **submission-loop restructure** (back-to-back submission + family-end
join + failure fencing) and the new path-proof counters.
*Gates per step:* focused replay regressions in `ProductionTestPreflight`;
byte-identical E2E on real models; the §3 decomposition shows the term moved
off the critical path; full x86 A/B (§7); source sanitizers green.
*Risks:* §5.5 (protocol complexity, placement-timing changes, failure
semantics); mitigated by step ordering — (a) alone is shippable if (b)/(c)
stall.

**Phase 3 — integration, defaults, and full evidence.**
If measured wins hold: consider default segment policy (Phase 1 follow-up),
sync all docs/help (maxim #38), re-run the complete cross-engine matrix
(CPU + both CUDA engines, ideas 24/25 validation section) and file results
into the tracker; register the final focused regressions in
`ProductionTestPreflight` (maxim #22).

**Recommended phase order (executive summary):**
1. Phase 0 counters first — cheap, no behavior change, decides A-vs-B with
   evidence (the 25c rule).
2. Phase 1 Fix A pilot immediately after (existing flags, no code) — likely
   2–3× chunk-count reduction for long prompts if GB10 admission fits.
3. Phase 2(a) maintenance-deferral next — removes the dominant hidden wait
   for every bucket configuration, CPU included.
4. Phase 2(b)+(c) family lease + back-to-back submission — completes Fix B
   toward the ~0.66–0.94 ms/token envelope.
5. Phase 3 defaults + full matrix — only after every x86 A/B is
   same-or-better.

## 9. Validation summary

- Correctness: byte-identical text vs current serial pipeline (greedy, fixed
  seed) on every measured cell; exact KV extents per chunk; snapshot/parity
  suites unchanged and green; focused replay + lifecycle regressions
  registered in `ProductionTestPreflight`.
- Economy: unprofiled Release benchmark, mean±sd, prefill and decode, with
  the §3 counter decomposition attached; profiler-gated GPU attribution kept
  out of canonical numbers.
- Non-regression: x86 GPU (MoE + dense), x86/ARM CPU, ROCm where qualified —
  same-or-better bar, alternating SHA-pair A/B, mode-asserted.
- Memory: BOM deltas admitted through `PhysicalMemoryAuthority`; typed
  capacity rejection is a valid outcome, never a fallback.

## References

- Tracker: ideas 24/25/25c/20/00 (this directory); K14 trace seals
  (`K14-IDEA25-TRACE-STRUCT-20261109`, `K14-IDEA25-Q38-TRACE-CPU4096-20261111`).
- Roadmap: `post-merge-performance-roadmap.md` (Tier 1 item 1; operator bar).
- Repo: `AGENTS.md` maxims (esp. #1–#3, #8–#16, #22, #26, #29, #38);
  `.github/instructions/llaminar-architecture-v2.instructions.md` §5–§8;
  `docs/production-ci.md` for CI/E2E/benchmark boundaries.
- Code map: §2 of this document (all citations at `d5e04fdce`).
