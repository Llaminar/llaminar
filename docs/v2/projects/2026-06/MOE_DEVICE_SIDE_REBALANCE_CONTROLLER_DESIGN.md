# GPU MoE Device-Side Rebalance Controller Design

## Goal

For homogeneous GPU LocalTP MoE domains (`cuda+nccl` or `rocm+rccl`, degree
2+), rebalance publish/apply must be graph-capturable and device-owned. The
host may initialize topology, allocate slots, prewarm graphs, and poll
diagnostics, but it must not participate in the steady-state histogram sync,
placement decision, transfer publication, or runtime-table apply path.

This design is for same-backend domains only. Cross-vendor movement and mixed
CUDA/ROCm expert domains remain out of scope for this controller.

## Policy Taxonomy

Expert ownership, rebalance strategy, routed-row assignment, and optional cache
residency are separate axes.

Base routed-expert storage for this sprint is `ApportionedExperts`: every routed
expert has one whole-expert owner in the domain. `ReplicatedExperts` means every
participant owns every expert. `ShardedExperts` means every participant owns a
shard of each expert and must combine partial outputs. These are storage
policies, not rebalance algorithms.

`Static`
: Experts are apportioned once and never moved by runtime policy. Histograms may
  still be collected for diagnostics. This is the correctness and performance
  baseline.

`Dynamic`
: One shared strategy across CPU, CUDA, and ROCm. It observes decode histogram
  windows and periodically rebalances the hottest routed experts so participant
  load is closer to even. CPU applies the policy through the host controller;
  homogeneous GPU domains apply the same policy through the device-side
  controller, async compact payload transfer, and arrival machinery. The current
  controller defaults start with a 256-token window, grow by
  `window_growth_factor=1.5`, and cap at `max_window_size=4096`. If we want a
  strict quadratic/power schedule, make it an explicit `window_schedule`
  setting. The current implementation uses the shared paired ownership-swap
  helper and rejects empty/no-payback moves.

`LLEP`
: Least-Loaded Expert Parallelism from arXiv:2601.17111. LLEP preserves the
  router's exact top-k expert choices and route weights, then assigns current
  routed row spans to least-loaded participants under capacity, minimum-chunk,
  and transfer-cost constraints. The paper-aligned prefill/batched path imports
  foreign expert payloads for the current batch/window without changing
  long-lived ownership. The current decode proxy uses the same shared LLEP
  planner to choose durable whole-expert ownership transfers when
  `HotExpertReplicaCache` is disabled, or additional resident arrivals when the
  cache layer is enabled.

`Observe`
: A measurement mode, not a rebalance strategy. It should exercise the same
  histogram and perfstat path as the selected strategy but must not mutate
  placement or cache state.

`HotExpertReplicaCache`
: An optional residency layer, not an algorithm choice. A top-10 or
  percentage-based cache can retain additional whole-expert copies after Dynamic
  or LLEP decides an expert is worth importing. The router may use resident cache
  contents to choose the least-loaded valid participant for an already-selected
  expert, but it must never change router top-k choices. Cache admission and
  retention need separate payback thresholds.

The target configuration shape is:

- `expert_rebalance_strategy = static | dynamic | llep`
- `hot_expert_cache = off | top_k:N | percent:P`
- `rebalance_observe = true|false`

## Target Graph Shape

The controller is a graph-visible state machine split across decode and
maintenance lanes.

1. `CollectLocalState`: each participant keeps histograms, resident descriptors,
   command buffers, wave state, and transfer-slot metadata in graph-stable
   device memory.
2. `AsyncWaveHistogramGather`: at a rebalance-window boundary, the maintenance
   graph packs histogram rows for the active rolling layer wave and gathers them
   on an explicit maintenance stream. The production payload is
   `[participant][wave_layer][expert]`, not the full
   `[participant][model_layer][expert]` tensor.
3. `RootPlanAssignments`: the elected root participant consumes gathered state,
   computes the selected strategy on device, and publishes epoch-stamped command
   buffers. Non-root participants execute graph-symmetric no-op/validation work.
4. `ScheduleTransferBucket`: the root/device scheduler chooses the smallest
   pre-captured payload bucket that can hold the planned arrivals. Until device
   graph launch or graph conditionals are available, host code may submit an
   already-captured bucket graph from device-published scheduler state, but it
   must not compute policy or mutate placement.
5. `StageArrivals`: sources pack only planned, non-empty NativeVNNI payloads
   into compact staging slots. NCCL/RCCL grouped collectives move the selected
   bucket lane. Destinations unpack into local preallocated transfer slots.
6. `RouteBoundaryApply`: the last local routed-expert dispatch polls ready wave
   state and publishes arrived or already-resident descriptors into the mirrored
   runtime table before later dispatch observes placement. A miss is a device
   no-op.

The steady-state decode graph owns routed expert compute and histogram updates.
The async maintenance graph owns histogram collection, root planning, and
transfer staging. The host constructs and prewarms the machine; replay advances
through device memory, explicit streams, events, kernels, and grouped
collectives.

## Transfer Contract

The active GPU packed format is backend-neutral NativeVNNI for same-backend
domains: payload, scales, mins, emins, codebook/block metadata, and matrix
dimensions. Same-backend GPU arrivals copy descriptor-described bytes directly;
they do not serialize to CPU-native blobs and do not repack on arrival.

NCCL/RCCL graph capture records concrete collective counts and participant
schedules. The transfer lane therefore uses bucketed graph variants:

- Capture one payload graph per power-of-two expert-arrival bucket up to the
  reserved staging-slot capacity.
- Use compact payload slots indexed by `[destination][payload_slot]`, not sparse
  plan capacity. A one-expert wave must not move an arena full of empty slots.
- Skip the payload graph entirely when a completed plan has no arrivals.
- Report requested arrivals, selected bucket, reserved bytes, useful bytes,
  wasted bytes, and utilization in PerfStats.

Explicit peer-copy branches are not part of the target path. NCCL/RCCL should
choose the available transport internally.

Captured maintenance transfer stream topology is aux-only in production: pack,
grouped NCCL/RCCL payload movement, unpack, and publish happen on the
device-context auxiliary transfer stream, with graph-captured event edges
to/from the maintenance capture stream. CUDA and ROCm must stay aligned here.

Perfstats tag `device_rebalance_transfer_stream_path` and should report
`auxiliary_stream` for transfer-slot maintenance work on both backends.

Current stream-overlap evidence:

- 2026-07-01 CUDA fix: captured replay previously forced a CUDA-only
  per-segment stream sync even when the caller requested deferred final sync.
  That made aux-stream metadata/payload replay enqueue block for the whole
  payload graph (`~64 ms` per launch in the earlier 2048-token diagnostic).
  `DeviceGraphCaptureController` now skips the CUDA segment sync when
  `defer_final_sync=true`, matching the ROCm deferred path.
- CUDA2 2048, seed 303, default LLEP after the fix:
  `benchmark_results/qwen36_moe_overlap_diag_after_deferred_cuda_sync_20260701T070127Z/`.
  Decode reached `124.03 tok/s`. Metadata/payload replay enqueue fell to
  `8.22 ms` total over 4 device records, with metadata/payload GPU elapsed
  `8.41 ms`; probe replay enqueue was `0.53 ms` against `34.72 ms` probe GPU
  elapsed. The large `64 ms` CUDA enqueue cliff is gone, but CUDA payload
  replay enqueue still tracks the small payload graph duration (`0.49-3.63 ms`
  in this run), so the next CUDA-specific question is whether `cudaGraphLaunch`
  of NCCL payload graphs is still effectively synchronous at this scale.
- ROCm2 2048, seed 303, default LLEP in the same run launched probes only:
  `59.05 tok/s`, no payload arrivals accepted. This is a policy-gate outcome,
  not an overlap-path measurement.
- ROCm2 2048, seed 303, forced-open movement gates:
  `benchmark_results/qwen36_moe_rocm_overlap_forced_payload_20260701T070745Z/`.
  Metadata/payload replay enqueue was `0.52 ms` total over 6 device records,
  while metadata/payload GPU elapsed was `34.63 ms`. This is the cleanest
  current proof that RCCL payload graph replay on the aux stream overlaps
  host/decode work mechanically.
- The focused CUDA integration probe
  `NCCLGroupedP2PMaintenanceGraph_AuxiliaryStream_TimingProbe` captures the
  real grouped NCCL P2P primitive through the auxiliary maintenance stream. For
  a 4 MiB bidirectional payload over six replays, the probe completes in about
  `5.26 ms`.
- The focused ROCm integration probe
  `RCCLGroupedP2PMaintenanceGraph_AuxiliaryStream_TimingProbe` captures the
  real grouped RCCL P2P primitive through the auxiliary maintenance stream. For
  a 4 MiB bidirectional payload over six replays, the probe completes in about
  `6.46 ms`.
- The captured decode/maintenance overlap probe
  `RCCLDecodeMaintenanceOverlap_BarrierAfterMaintenanceLaunch_Completes`
  replays a decode allreduce graph and maintenance collective graph on separate
  streams/contexts and completes four overlap iterations in about `1.28 ms`.

Conclusion: auxiliary-stream grouped P2P is not intrinsically slower on either
CUDA or ROCm. ROCm now demonstrates true payload replay overlap in the full
model when payload waves are forced. CUDA no longer has the explicit replay
sync bug, but its full-model NCCL payload graph launch still appears more
blocking than ROCm's at small payload sizes and needs an Nsight timeline pass
around `cudaGraphLaunch`/NCCL graph replay. Separately, the current host
probe-outcome readback/rendezvous still blocks to choose whether to launch a
payload graph; that remains below the target device-side graph scheduler design.
LLEP/Dynamic still need first-class cost gates so cheaper maintenance does not
become over-maintenance.

## LLEP Algorithm Target

LLEP is a routed-row assignment strategy, not a hot-cache policy.

1. Gather current MoE layer global expert row counts for the active batch.
   Prefill should reuse grouped-prefill routing/sort scratch. Batched decode
   should use the effective request batch rows.
2. If imbalance is below `lambda`, use standard `ApportionedExperts`.
3. Run shared LLA/LLAS logic over measured expert loads. Llaminar must use the
   runtime owner map, not a contiguous `expert / experts_per_device` assumption,
   because Dynamic may make ownership non-contiguous.
4. Emit assignment spans:
   `(layer, expert, owner_participant, destination_participant,
   route_row_begin, route_row_end, needs_foreign_weight)`.
5. Transfer only required token rows, route weights, and foreign expert payloads.
   Compute native plus foreign chunks, reverse-combine outputs to original route
   rows, and preserve exact top-k semantics.

Shared policy code should live outside CUDA/ROCm-specific files. The current
direction is `LeastLoadedExpertAssignment.h` plus a common routed-assignment
dispatch surface, with CUDA, ROCm, and CPU tests sharing semantics.

Current conformance audit, 2026-07-01:

- The shared `LeastLoadedExpertAssignment` planner matches the core LLA/LLAS
  policy shape from the paper: sort expert loads descending, reserve pending
  native work, apply `alpha` capacity, spill overload to least-loaded
  non-native participants, enforce `lambda`, `min_chunk_tokens`, and transfer
  cost gates, and use Llaminar's explicit runtime owner map instead of a
  contiguous `expert / experts_per_device` assumption.
- The decode maintenance controller currently calls the transfer-only planner
  and converts foreign expert needs into compact whole-expert movement commands.
  With `HotExpertReplicaCache=off`, those commands are durable
  `OwnershipTransfer` operations; with the cache enabled, they are additional
  resident arrivals. This is useful, graph-capturable residency movement, but it
  is not the full paper Algorithm 4 current-batch
  route-row assignment/exchange/compute/combine path.
- Prefill routing now preserves router top-k choices, plans current-batch
  LLEP spans, materializes missing foreign whole-expert arrivals through the
  compact rebalance ABI, gathers compact payload slots on the LocalTP
  collective stream, applies arrivals into transfer slots, and only then
  rewrites `route_participant_ids` from the span plan. There is no resident
  fallback in the production current-batch LLEP path; missing compact transfer
  backing is a hard configuration error.
- `LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE` defaults to `full`. The
  `resident-only` mode remains available only as an explicit diagnostic guard
  for already-resident spans; it is not the production LLEP default.
- `LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS` defaults to `8192` routed rows.
  Below that gate, graph construction stamps the prefill expert stage as
  standard `StaticOwner` apportioned-expert work and does not attach LLEP
  runtime grouping or compact transfer backing. The older in-stage skip counter
  remains as a defensive runtime guard, but production measured replay should
  avoid lowering the tiny-prefill LLEP path in the first place.
- Device-side decode movement now defaults the relative load-spread gate to
  `device_min_load_spread_improvement_divisor=15`. A wave must improve
  projected participant load spread by at least `current_total / 15`, unless
  explicitly overridden to `0` for diagnostics. This prunes whole-expert
  movement waves that have positive but too-small projected benefit.
- Repeated no-work maintenance probe completions now carry a progressive
  backoff count in `no_work_backoff_effective_periods`. Useful work resets the
  streak. In short 1024-token smokes each per-device maintenance cache usually
  only reaches streak 1, but longer no-work runs should now probe less
  aggressively.
- Therefore today's `LLEP` runtime mode is a first transfer-backed
  current-batch implementation for LocalTP apportioned grouped prefill, plus
  residency-based decode balancing. It implements the paper's load-aware
  routed-row assignment against Llaminar's runtime owner map, but the first
  production slice still imports whole foreign experts rather than exchanging
  only assigned token rows.
- PerfStats now exports the audit counters
  `device_rebalance_llep_assignment_span_count`,
  `device_rebalance_llep_weight_transfer_count`,
  `device_rebalance_llep_native_rows`,
  `device_rebalance_llep_spilled_rows`,
  `device_rebalance_llep_spilled_row_ratio`, and
  `device_rebalance_llep_spilled_rows_per_transfer`. These tell us how large
  the conceptual LLEP row-assignment opportunity was versus how much the
  current residency path actually moved.
- CUDA and ROCm now also expose a graph-capturable current-batch prefill
  planner that writes full LLEP assignment spans to
  `DeviceMoELayerRuntime::reserved_ptrs[1]` and required expert-transfer
  records to `reserved_ptrs[2]`, with capacities/counts in `reserved_u64`.
- CUDA and ROCm also expose explicit span-apply primitives:
  `NoTransfers` rewrites `route_participant_ids` only when the plan reports
  zero missing foreign weights, while `AfterTransfers` is valid only after the
  compact arrival path has populated destination descriptors. The blended
  current-batch-or-resident helper was removed so unsupported current-batch
  movement fails fast instead of silently changing strategy.
- CUDA and ROCm now expose a graph-capturable command-materialization bridge
  from current-batch LLEP transfer records to the standard compact rebalance
  ABI. It converts `reserved_ptrs[2]` weight-transfer records into bounded
  `ExpertPayloadArrival` command buffers, fills payload/destination slot ids,
  reports the same payload bucket/status counters as the maintenance transfer
  path, and is consumed by the production grouped-prefill bridge through
  graph-owned workspace buffers.
- The shared LLEP planner now accepts per-expert resident masks. A destination
  that already has a hot-cache/resident copy is treated as executable without a
  payload transfer, so LLEP and hot-cache residency are decoupled but
  composable.
- Durable no-cache LLEP ownership waves are evaluated by actual load-spread
  improvement and movement-cost floors, but they are not required to satisfy the
  hot-cache/replica absolute post-spread ceiling after a single wave. The
  absolute ceiling remains a cache/replica payback gate; otherwise useful
  durable ownership moves can be rejected simply because one compact wave cannot
  make the whole domain nearly balanced.

Paper speed-positive reading:

- The paper's speedup claim applies most directly to current-batch routed-row
  redistribution: when the active batch is large, expert routing is materially
  imbalanced, spilled chunks are large enough to form efficient GEMMs, and the
  cost of token/weight movement is amortized by reducing the slowest
  participant's MoE compute time.
- Llaminar's shared planner has the right LLA decision surface for that
  algorithm: `lambda` selects standard EP when routing is balanced enough,
  `alpha` caps per-participant routed work, `min_chunk_tokens` avoids tiny
  spill GEMMs, and the spread/foreign-row gates model useful-work floors.
- Transfer-backed grouped prefill is now the closest implementation to the
  paper: it plans current-batch spans and imports missing foreign expert
  payloads before applying the span assignment. It is still an implementation
  slice, not the full paper Algorithm 4, because it moves whole expert payloads
  through compact slots and relies on the LocalTP replicated-hidden/allreduce
  shape instead of doing a general token-row all-to-all plus reverse combine.
- Single-token decode is not the regime the paper's batch-size speedup curves
  are proving. Our decode LLEP path is a durable residency/ownership movement
  policy that uses the LLA transfer plan as an admission heuristic. It can be
  speed-positive only when routed skew persists for enough remaining decode
  tokens that future dispatch savings amortize whole-expert movement and
  maintenance overhead.
- Therefore we should expect paper-like wins first in chunky prefill and
  batched decode, not in short single-stream decode. For short decode, the
  correct behavior is often to choose standard EP and pay near-zero LLEP
  overhead.

Fresh 1024-token seed-303 audit after adding the counters:

- CUDA2 static: `124.57 tok/s` decode.
- CUDA2 LLEP: `125.45 tok/s` decode. The planner observed `577`
  conceptual assignment spans, `746` spilled rows, and `5` foreign weight
  transfers, but accepted `0` arrivals. Pre/post projected imbalance was
  unchanged (`0.0411` average), so this row is essentially a no-movement
  measurement.
- ROCm2 static: `63.62 tok/s` decode.
- ROCm2 LLEP: `65.09 tok/s` decode. The planner observed `1391`
  conceptual assignment spans, `2166` spilled rows, and `16` foreign weight
  transfers, accepting `6` planned arrivals. Pre/post projected imbalance only
  moved from `0.04898` to `0.04893`, which is far smaller than the full
  row-level LLEP opportunity.

Interpretation: current residency-based LLEP can be mildly speed-positive, but
it is not sufficiently aligned with Algorithm 4 to realize the projected
current-batch row-balancing benefit. The next speed-positive path should focus
on full prefill LLEP row assignment and exchange, while keeping decode on
residency-based balancing unless batched decode provides enough rows to
amortize current-token exchange.

Next implementation target for paper-aligned LLEP:

1. Prove the transfer-backed grouped-prefill bridge end to end with Qwen3.6
   PyTorch parity and CUDA2/ROCm2 benchmark rows.
2. Split the current in-stage bridge into first-class graph stages or a
   device-side scheduler form so transfer waves can overlap more aggressively
   with useful compute and avoid any host policy decisions.
3. Add a graph-capturable zero-arrival skip/bucket scheduler so a no-work LLEP
   plan does not pay compact payload allgather cost.
4. For the LocalTP apportioned fast path, use the fact that every participant
   already has the hidden/routing tensors and the downstream MoE allreduce
   rejoins partial outputs. The first execution slice can therefore avoid
   token-row exchange and focus on graph-capturable import of missing foreign
   expert payloads plus route assignment from the span plan.
5. For graph shapes where hidden/routing are not replicated, compact and
   exchange only the assigned token rows and route weights.
6. Import only missing foreign expert payloads needed by those spans.
7. Run grouped native plus foreign expert chunks.
8. Reverse-combine outputs to original routed rows without changing top-k
   experts or route weights.

Decode can continue to use residency-based balancing unless/until batched
decode provides enough rows per expert for current-token LLEP chunks to amortize
the token and weight exchange costs.

Important tunables:

- `lambda`: imbalance threshold for selecting standard AE instead of LLEP row
  reassignment.
- `alpha`: per-participant capacity factor.
- `min_chunk_tokens`: minimum useful rows for a spilled expert chunk.
- `max_foreign_experts_per_wave`: transfer-slot and VRAM guard.
- Cost gates: foreign expert bytes, token collective bytes, grouped-GEMM
  efficiency by row count, and maintenance graph cost.

## Dynamic Algorithm Target

Dynamic is the simpler whole-expert rebalance strategy and should stay useful
even if hot-cache or LLEP are disabled.

Target behavior:

- Start at a 256-token decode histogram window.
- Use an explicit window schedule. The current implementation uses adaptive
  `1.5x` growth to a `4096` cap; a quadratic/power schedule should be added as
  a named option if desired.
- Use the shared Dynamic ownership-swap helper to pair a heavy expert from the
  overloaded participant with a light expert from the underloaded participant.
  The same helper is called by the host `SocketAwareRebalancer` and the
  CUDA/ROCm device-side controller.
- A future hot-set variant may select the hottest `hotset_expert_count` routed
  experts or layer-expert items, defaulting to 20, but that should be an
  explicit extension of Dynamic rather than a second strategy name.
- Reassign selected non-empty experts across all domain participants with a
  load-spread balance objective.
- Reject moves that do not improve projected participant load enough to pay for
  maintenance and transfer.
- Keep ownership movement and hot-cache admission separate. Dynamic can move
  owners, create temporary resident replicas, or do no transfer at all depending
  on the configured execution mode.

Public sweep knobs now exposed through CLI, YAML, and `DebugEnv`:

- `dynamic_imbalance_threshold_permille`
  / `--moe-dynamic-imbalance-threshold-permille`
  / `LLAMINAR_MOE_DYNAMIC_IMBALANCE_THRESHOLD_PERMILLE`
- `dynamic_min_improvement_permille`
  / `--moe-dynamic-min-improvement-permille`
  / `LLAMINAR_MOE_DYNAMIC_MIN_IMPROVEMENT_PERMILLE`
- `dynamic_max_swaps_per_layer`
  / `--moe-dynamic-max-swaps-per-layer`
  / `LLAMINAR_MOE_DYNAMIC_MAX_SWAPS_PER_LAYER`
- `dynamic_max_plan_entries_per_wave`
  / `--moe-dynamic-max-plan-entries-per-wave`
  / `LLAMINAR_MOE_DYNAMIC_MAX_PLAN_ENTRIES_PER_WAVE`
- `dynamic_min_window_activations`
  / `--moe-dynamic-min-window-activations`
  / `LLAMINAR_MOE_DYNAMIC_MIN_WINDOW_ACTIVATIONS`

Device movement-cost gates are also public config now:

- `device_min_load_spread_improvement`
  / `--moe-device-rebalance-min-load-spread-improvement`
  / `LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT`
- `device_min_load_spread_improvement_divisor`
  / `--moe-device-rebalance-min-load-spread-improvement-divisor`
  / `LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR`
  (default: 15; set 0 to disable the relative floor)
- `device_min_wave_spread_improvement_per_payload_slot`
  / `--moe-device-rebalance-min-wave-spread-improvement-per-payload-slot`
  / `LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT`
- `device_min_router_spread_improvement_per_payload_slot`
  / `--moe-device-rebalance-min-router-spread-improvement-per-payload-slot`
  / `LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT`
- `device_max_post_wave_load_spread_permille`
  / `--moe-device-rebalance-max-post-wave-load-spread-permille`
  / `LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE`

Shared Dynamic defaults live in `DeviceMoERebalancePolicyShared.h` and feed CPU,
CUDA, and ROCm. Device movement-cost gate defaults live in runtime config and
`DebugEnv`, then feed the Qwen35 MoE GPU graph binding. Negative CLI/YAML values
are rejected. Environment values are clamped to non-negative values at runtime
config import.

## Current Implementation Status

Implemented or partially implemented:

- GPU packed-format contract uses backend-neutral NativeVNNI descriptors for
  same-backend arrivals.
- Production decode has a device-side rebalance ABI with command buffers, wave
  state, route-boundary apply, explicit streams, and graph-owned state.
- CUDA and ROCm share the same high-level controller shape and tests for ready
  wave apply, transfer-slot apply, and router hot-cache accounting.
- Maintenance is split into probe and metadata/payload graph bodies. No-work
  waves now run only the histogram/controller probe and skip command-buffer
  allgathers plus compact payload movement.
- Compact transfer-slot staging moves planned payload slots instead of fixed
  plan-capacity arenas.
- `LeastLoadedExpertAssignment` and the shared routed-assignment surface exist
  for offline/unit validation.
- Grouped prefill now materializes compact gate/up/down descriptor tables from
  the active device runtime bank immediately before execution. CUDA and ROCm
  share this shape, so runtime-table arrivals can be executed without rebuilding
  static host descriptor tables or repacking GPU weights.
- The GPU prefill LLEP bridge is span-aware and transfer-backed for LocalTP
  apportioned grouped prefill: repeated rows for a hot expert can be assigned
  across least-loaded participants, missing foreign experts are staged through
  compact transfer slots, and route assignments are applied only after arrivals
  are visible in the runtime table.
- The resident-span prefill bridge now uses first-class runtime-table scratch
  for a compact `[expert][participant]` split table. CUDA and ROCm use the
  shared LLEP resident water-fill helper to plan split counts in chunks, then
  fill per-route participant assignments in parallel. This removes the older
  single-thread per-row split loop from both backend kernels.
- Transfer-backed LLEP now feeds the same graph-capturable compact-arrival path
  as Dynamic. The shared transfer-only LLEP planner emits foreign whole-expert
  movement, and CUDA/ROCm controller kernels publish durable
  `OwnershipTransfer` commands when hot cache is disabled or
  `ExpertPayloadArrival` commands when the optional cache layer is enabled.
- CUDA and ROCm now distinguish LLEP candidate span economics from accepted
  runtime economics. Candidate counters report the ideal row-span plan from the
  shared LLEP helper; accepted counters and post-policy imbalance are recomputed
  from the actual whole-expert resident masks published by the live GPU proxy.
  This prevents partially accepted arrivals, duplicate-resident skips, and the
  current whole-expert proxy from overstating realized load-balance gains.
- `SocketAwareRebalancer` now delegates Dynamic ownership-swap selection to the
  same shared helper used by CUDA and ROCm device-side planning, preserving the
  host proposal/apply API while removing policy drift.
- Dynamic defaults for imbalance threshold, minimum improvement, per-layer swap
  count, command capacity, and minimum window activations are defined once in
  the shared policy header and consumed by both CPU and device configs.
- Dynamic policy knobs are now surfaced through runtime config, CLI, nested and
  flat YAML, DebugEnv, explain output, CPU controller wiring, and Qwen35 MoE GPU
  graph wiring.
- Device movement-cost gates are also surfaced through runtime config, CLI,
  nested and flat YAML, DebugEnv, explain output, and Qwen35 MoE GPU graph
  wiring. Graph construction honors typed config values by default and only
  lets `LLAMINAR_MOE_DEVICE_REBALANCE_*` values override them when those env
  vars were explicitly present at `DebugEnv` reload.
- CUDA and ROCm integration tests cover Dynamic ownership-transfer planning
  with hot-cache disabled, including root-side deferred planning, command
  buffers, load-spread stats, and unchanged active runtime state before apply.
- CUDA now preserves grouped descriptor-table host metadata across workspace
  rebinding, matching ROCm. This fixes a dynamic-path failure where the
  singleton MoE kernel could lose CUDA shared-expert grouped decode descriptor
  tables while stages still held cached table ids.
- Trace tooling exports expert loads, owner maps, apply visibility, router
  cache-use counters, payload economics, and imbalance metrics. Analyzer output
  now includes per-window imbalance ratios plus run-level avg/max/sample fields
  for pre-policy, post-policy, and post-wave load spread over total load, and
  the policy comparator can use those fields directly for threshold searches.
- The LLEP transfer planner now honors the shared relative spread-improvement
  floor (`min_spread_improvement_divisor`) in addition to the absolute and
  per-transfer floors. CUDA and ROCm pass the same device rebalance config into
  LLEP, so the existing cost-gate knob no longer silently misses the LLEP path.
- Current-batch LLEP assignment now reuses grouped-prefill route metadata instead
  of scanning every route slot per expert. The planner publishes per-expert span
  bounds in runtime scratch, and CUDA/ROCm assignment kernels apply only the
  local span slice. Transfer-command materialization also avoids dead plan-buffer
  zero-fill and uses a deterministic parallel no-overflow fast path with shared
  transfer staging and parallel status bookkeeping.
- 2026-07-02 validation after tuning grouped route scatter, Dynamic
  maintenance, current-batch LLEP assignment, and materialization:
  `V2_Perf_MoELLEPDeterminism`, `V2_Integration_CUDAMoEKernel`,
  `V2_Integration_ROCmMoEKernel`, and focused phase-split CUDA2/ROCm2 Qwen3.6
  Dynamic/LLEP `PrefillParity`, `DecodeParity`, `LongContextDecodeParity`, and
  Dynamic `SnapshotInfrastructure` passed.
- 2026-07-01 validation after exposing device movement-cost gates: integration
  and release builds passed; focused config/DebugEnv/LLEP tests passed; full
  unit suite passed (`514/514`).
- 2026-07-01 validation after fixing LLEP accepted-load accounting:
  integration build passed; `V2_Unit_LeastLoadedExpertAssignment`,
  `V2_Integration_CUDAMoEKernel`, and `V2_Integration_ROCmMoEKernel` passed.
- 2026-07-01 validation after splitting maintenance probe from metadata/payload
  replay: integration and release builds passed; `V2_Unit_MoEForbiddenDependencyScan`,
  `V2_Unit_GpuWorkspaceAllocationPolicy`, `V2_Unit_LeastLoadedExpertAssignment`,
  `V2_Integration_CUDAMoEKernel`, and `V2_Integration_ROCmMoEKernel` passed;
  full V2 unit suite passed (`514/514`).
- 2026-07-01 validation after RCCL overlap investigation: graph-captured RCCL
  decode allreduce and graph-captured RCCL maintenance raw-allgather can replay
  concurrently on dedicated explicit streams in the LocalTP overlap lab. ROCm
  does not need a drain path, and request reset must preserve captured graph
  replay instead of eagerly recapturing.
- Device maintenance completion now uses a domain-wide readiness rendezvous
  before exporting probe or metadata/payload diagnostics and before entering
  the blocking probe-outcome rendezvous. This fixes the ROCm failure where one
  worker observed its completion event, exported a root/probe row, and blocked
  while another worker had not yet observed the same maintenance wave as ready.
- 2026-07-01 validation after wiring transfer-backed current-batch LLEP spans
  into production grouped prefill: integration build passed;
  `V2_Unit_LeastLoadedExpertAssignment`,
  `V2_Unit_MoERuntimeTable`,
  `V2_Unit_Qwen35MoEGraphNativeProductionLowering`,
  `V2_Integration_CUDAMoEKernel`, and
  `V2_Integration_ROCmMoEKernel` passed. Focused CUDA/ROCm kernel tests cover
  guarded no-transfer apply, after-transfer span apply, materialized compact
  arrival commands, and resident replica destinations that avoid payload-transfer
  requirements.
- 2026-07-01 validation after adding the current-batch LLEP transfer-command
  bridge: `v2_integration_cuda_moe_kernel` and
  `v2_integration_rocm_moe_kernel` rebuilt successfully; focused
  `*RuntimePrefillLeastLoaded*` CUDA and ROCm tests passed. The planner test
  now asserts that a pending foreign expert payload materializes as a bounded
  `ExpertPayloadArrival` command with payload bucket/status counters.
- 2026-07-01 validation after removing the implicit
  current-batch-or-resident helper: integration build passed;
  `V2_Unit_LeastLoadedExpertAssignment`, `V2_Unit_MoERuntimeTable`,
  `V2_Unit_Qwen35MoEGraphNativeProductionLowering`,
  `V2_Integration_CUDAMoEKernel`, and `V2_Integration_ROCmMoEKernel` passed.
  Production grouped GPU LLEP prefill now requires compact transfer backing and
  throws if it is absent.
- 2026-07-01 Qwen3.6 ExpertOverlay math parity after the explicit-mode cleanup:
  CUDA2 and ROCm2 `PrefillParity`, `DecodeParity`, and
  `SnapshotInfrastructure` all passed. The suite also fixed its LocalTP test
  harness to create an explicit single-rank `MPIContext`, matching older
  overlay parity suites and preventing a `RankOrchestrator::Config` shared_ptr
  copy crash before graph construction.
- 2026-07-01 validation after splitting durable no-cache LLEP ownership waves
  from the hot-cache post-spread ceiling: integration build passed;
  `V2_Unit_MoEForbiddenDependencyScan`,
  `V2_Integration_CUDAMoEKernel`, and `V2_Integration_ROCmMoEKernel` passed.
  The CUDA/ROCm regression tests set a strict 10% post-spread ceiling and still
  require no-cache LLEP to publish an improving durable `OwnershipTransfer`.
- 2026-07-01 validation after fixing device-side ownership-transfer apply:
  integration build passed; focused CUDA and ROCm ownership-transfer source
  apply regressions passed; `V2_Unit_MoEForbiddenDependencyScan`,
  `V2_Integration_CUDAMoEKernel`, and `V2_Integration_ROCmMoEKernel` passed.
  Root cause: CUDA/ROCm destination-side apply set
  `owner_participant=destination`, but source/non-destination apply only
  cleared local flags. The next runtime bank rebuild OR'd the stale source
  owner back into the resident mask, resurrecting old ownership. Device apply
  now matches the host mirror and updates `owner_participant` on every
  participant for `OwnershipTransfer`.
- 2026-07-01 validation after removing grouped decode soft retries:
  integration build passed; `V2_Unit_MoEForbiddenDependencyScan`,
  `V2_Integration_CUDAMoEKernel`, and `V2_Integration_ROCmMoEKernel` passed.
  CUDA and ROCm grouped decode now treat enabled K-part / parallel fast paths
  as explicit contracts: if scratch allocation, codebook support, or the
  selected kernel launch fails, the path fails closed instead of retrying a
  serial decode path. Serial decode remains available only when the
  corresponding K-part/parallel knob is explicitly disabled. Post-cleanup
  CUDA2 and ROCm2 Qwen3.6 ExpertOverlay `DecodeParity` both passed.

Latest evidence:

- 2026-06-30 CUDA2 resident-only LLEP bridge A/B, seed 303, 1024 decode tokens,
  context 4096, `assignment=least-loaded-ep`, captured prefill required:
  `static` reached 1963.50 prefill tok/s and 126.96 decode tok/s; `dynamic_hot10`
  reached 1950.57 prefill tok/s and 126.00 decode tok/s. Both paths captured
  prefill graphs. This proves the resident-span bridge runs end to end, but it
  is not yet a speedup and should not be considered full transfer-backed LLEP.
- Matching ROCm2 A/B with the same seed/config reached `static` 517.15 prefill
  tok/s and 67.32 decode tok/s, versus `dynamic_hot10` 525.86 prefill tok/s and
  66.59 decode tok/s. ROCm also captures prefill graphs but does not show a
  resident-only LLEP decode win.

Still incomplete:

- Device-side graph scheduling still needs a host bridge for launching prewarmed
  bucket graphs on current CUDA/ROCm APIs.
- Dynamic, LLEP, and hot-cache admission are conceptually separated in code and
  tests. Dynamic now has an explicit sweep surface; LLEP and cache admission
  still need the same level of public config cleanup.
- Full paper-aligned LLEP still needs the general token-row exchange and reverse
  combine path for graph shapes where hidden/routing rows are not already
  participant-local. The current LocalTP prefill slice can prove the planner and
  compact expert import machinery, but it is not the complete Algorithm 4
  communication schedule.
- `alpha`, `lambda`, and `min_chunk_tokens` are available in the shared planner
  and backend kernels, but the production prefill policy still mostly depends on
  default planner values plus the higher-level routed-row gate. These need a
  coherent CLI/DebugEnv surface before serious LLEP sweeps.
- LLEP is mechanically transfer-backed but not yet performance-proven on the
  full Qwen3.6 parity/benchmark matrix after the explicit-mode cleanup. Current
  public plumbing should be exercised as a first-class strategy only after
  parity and repeated clean benchmark matrices.
- Hot-cache policy economics are inconsistent. The cache is mechanically visible
  to routing, but persistent hot10 alone is not a reliable speedup.

## Current Measurements

Raw run trails live in `benchmark_results/` and commit messages. This document
keeps only the current design signal:

| Scenario | Result | Takeaway |
| --- | --- | --- |
| CUDA/ROCm MoE deterministic maintenance, grouped route, and payload movement perf, 512 tokens, 256 experts, top-k 8, 4 participants, 2026-07-02 | `V2_Perf_MoELLEPDeterminism`: grouped prefill routes are CUDA `62.58 us` and ROCm `192.34 us` with hash `5063458188956773059`; Dynamic maintenance pack/controller is CUDA `156.97 us` and ROCm `521.71 us` with plan hash `10755319433736730929`; payload movement/apply improved to CUDA `281.48 us` and ROCm `512.52 us` with hash `16961506041180432709`. Component rows: CUDA pack `57.77 us`, bucket copy `39.55 us`, unpack `54.15 us`, apply `118.93 us`; ROCm pack `78.12 us`, bucket copy `39.89 us`, unpack `70.19 us`, apply `267.90 us`. | The deterministic grouped scatter uses block-wide chunked slot scans instead of one lane per expert. Plain Dynamic ownership planning has a gated fast path that bypasses the generic LLEP/hot-cache controller body while still using the shared ownership-swap helper. Payload pack/unpack no longer collapses to byte copies when the slot header leaves payload data 8/4/2-byte aligned rather than 16-byte aligned, and unpack now launches against the selected transfer-slot bucket instead of full plan capacity. Apply remains the largest remaining movement component, especially on ROCm. |
| CUDA/ROCm current-batch LLEP deterministic kernel perf, 512 tokens, 256 experts, top-k 8, 4 participants, 2026-07-02 | `V2_Perf_MoELLEPDeterminism`: CUDA assignment `4.28 us`, CUDA transfer-command materialization `5.58 us`; ROCm assignment `11.55 us`, ROCm materialization `34.70 us`. Hashes matched across backends: route assignment `9513987820616991619`, transfer plan `345862852378021593`. | Assignment no longer has the single-thread/per-expert route-slot walk (`~415 us` CUDA and `~1285 us` ROCm before tuning). ROCm materialization dropped from `~86 us` to `~35 us` by removing dead zero-fill and parallelizing the no-overflow command path, while preserving deterministic command order. |
| CUDA2 512, seed 303, static vs Dynamic no-cache, 2026-06-30 | 122.74 vs 125.80 tok/s decode; prefill 3235.87 vs 3227.29 tok/s | Dynamic path is healthy after the CUDA descriptor rebind fix. This sample had no payload movement (`payload_bucket_slots=0`, `apply_changed_layers=0`), so it proves overhead recovery, not policy benefit. |
| ROCm2 512, seed 303, static vs Dynamic no-cache, 2026-06-30 | 66.13 vs 65.63 tok/s decode; prefill 1145.22 vs 1144.90 tok/s | ROCm stayed healthy after the shared rebind guard. This sample also had no movement, so policy tuning still needs movement-positive traces. |
| CUDA2 1024, seed 303, recent clean static vs dynamic-hot10 | 126.83 vs 124.81 tok/s | Router/cache mechanics work, but hot10 maintenance was net negative in this run. |
| CUDA2 Dynamic knob sweep, seed 303, 1024/2048, 1 measured iter, perfstats on, `benchmark_results/qwen36_moe_cuda_dynamic_knob_sweep_seed303_20260630_210138/` | 1024: static 123.87, no-op Dynamic 123.49, default Dynamic 123.77, threshold1100 125.53, threshold1100+min0 123.77, aggressive 123.70 tok/s. 2048: static 124.98, no-op 125.21, default 124.68, threshold1100 124.14, threshold1100+min0 124.82, aggressive 124.23 tok/s. | No-op controller overhead was small in this run (`~4 ms` maintenance GPU elapsed at 1024, `~8 ms` at 2048, no movement). Lowering threshold to 1100 induced payload movement and produced the only 1024 positive signal (+1.34%), but movement was neutral/negative at 2048. Generated token streams diverged before the first maintenance window, so this bounds end-to-end behavior rather than replaying identical histogram inputs. Repeat candidates with more reps and/or trace replay before treating this as a policy win. |
| CUDA2 transfer-backed LLEP probe, seed 303, 1024, perfstats + trace, `benchmark_results/qwen36_moe_cuda_llep_transfer_probe_seed303_20260630_234622/` | static 123.91 vs dynamic LLEP 124.11 tok/s decode; prefill 1953.23 vs 1942.05 tok/s. Trace planned/applied two arrivals around token 323 and one arrival around token 835. | The transfer-backed path is real: `ExpertPayloadArrival` commands are planned, compact payload slots move useful bytes, runtime-table apply makes new residents visible, and router residency counters become nonzero. Payback was essentially flat in this one-run diagnostic, and the maintenance lane is still dedicated rather than fully fused into main decode collectives. |
| ROCm2 transfer-backed LLEP probe, seed 303, 1024, perfstats + trace, `benchmark_results/qwen36_moe_rocm_llep_transfer_probe_seed303_20260630_235145/` | static 66.08 vs dynamic LLEP 65.22 tok/s decode; prefill 524.69 vs 525.68 tok/s. Trace planned two reciprocal arrivals around token 324 and one later arrival around token 836. | ROCm matches CUDA mechanically but was negative in this one-run sample. This points to shared policy/cost-gate economics rather than a CUDA-only or ROCm-only wiring issue. |
| CUDA2/ROCm2 resident split chunk-planner probe, seed 303, 1024, `assignment=least-loaded-ep`, perfstats + trace, `benchmark_results/qwen36_moe_llep_chunk_split_probe_20260701_000943/` | CUDA: static 1855.37 prefill / 124.68 decode vs dynamic 1841.16 prefill / 124.51 decode. ROCm: static 537.28 prefill / 67.66 decode vs dynamic 537.47 prefill / 65.48 decode. Both dynamic runs planned 12 arrivals and requested 8 compact payload slots across the two participants; one arrival was applied before run end. Offline full-LLEP replay still predicts large spread reductions at ROI 256: CUDA spread 7015 -> 3955 with 10 transfers, ROCm 5664 -> 2518 with 11 transfers. | The shared chunk planner removes a real single-threaded kernel wart without changing semantics, but current live transfer-backed LLEP is still a coarse whole-expert-arrival/resident-assignment proxy. The remaining speed gap is implementation strategy, not absence of routing imbalance signal. |
| Compact payload capacity sweep, seed 303, 1024, dynamic LLEP only, `benchmark_results/qwen36_moe_llep_payload_slots2_probe_20260701_001956/` and `benchmark_results/qwen36_moe_llep_payload_slots4_probe_20260701_002406/` | CUDA dynamic: slots1 124.51 decode, slots2 123.86, slots4 118.80 tok/s. ROCm dynamic: slots1 65.48, slots2 64.55, slots4 64.69 tok/s. Slots2/4 planned and applied more arrivals on CUDA, but throughput fell. | Increasing whole-expert arrival capacity does not recover LLEP economics. Keep the compact payload default conservative and focus on true row-span LLEP for prefill/batched work or stronger admission gates for decode hot-cache movement. |
| CUDA2/ROCm2 current Dynamic load-ratio trace, seeds 303 and 606, 1024/2048, `assignment=least-loaded-ep`, trace mode, `benchmark_results/qwen36_moe_policy_loadratio_current_20260701T004332Z/` | CUDA 1024: Dynamic made no transfers and averaged -0.38 tok/s vs static. CUDA 2048: Dynamic reduced pre/post imbalance ratio from 0.0441 to 0.0344 but averaged -1.19 tok/s. ROCm 1024: no transfers, average -2.37 tok/s with one slow outlier. ROCm 2048: imbalance ratio 0.0409 -> 0.0319 but average -1.83 tok/s; seed 303 was positive (+3.38) and seed 606 negative (-7.05). | The controller can now quantify real imbalance reduction, but these whole-expert transfer waves are not consistently economic. A cost gate based on improvement relative to observed load is needed before more movement-capacity tuning. This run exposed that the LLEP path ignored `min_load_spread_improvement_divisor`; that is now fixed. |
| Relative LLEP cost-gate probes, seed 303, 2048, `MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR=15`, `benchmark_results/qwen36_moe_llep_relative_gate15_cuda_seed303_20260701T012906Z/` and `benchmark_results/qwen36_moe_llep_relative_gate15_rocm_seed303_20260701T013119Z/` | CUDA dynamic moved 4 arrivals, reduced imbalance 0.0434 -> 0.0341, and reached 125.67 tok/s decode versus the same-run static row from the divisor-25 A/B at 125.53 tok/s. ROCm dynamic moved 4 arrivals, reduced imbalance 0.0442 -> 0.0336, and reached 64.90 tok/s decode versus the earlier same-seed static row at 60.18 tok/s. | The relative floor is the right sweep axis: divisor 25 partially pruned CUDA arrivals (20 -> 14) and improved the delta (-1.40 -> -0.40 tok/s), while divisor 15 pruned to a high-value 4-arrival wave and recovered a positive single-row signal. This is now the device-side default so movement must scale with observed routed load; `0` remains the explicit diagnostic override. |
| CUDA2/ROCm2 no-work maintenance probe split, seed 303, 1024, divisor 15, `benchmark_results/qwen36_moe_probe_split_seed303_20260701T023641Z/` | CUDA: static 1853.63 prefill / 124.75 decode vs Dynamic 1839.74 prefill / 124.67 decode. ROCm: static 535.96 prefill / 62.78 decode vs Dynamic 537.73 prefill / 65.86 decode. Dynamic planned zero arrivals on both backends; traces exported only `probe` stages. No metadata/payload graph launched. | Splitting probe from metadata/payload replay removed command-buffer allgathers from no-work windows. No-work maintenance GPU elapsed fell versus the prior accounting-fix probe from ~45.6 ms to ~32.8 ms on CUDA and ~136.9 ms to ~100.2 ms on ROCm over six maintenance launches. CUDA no-work Dynamic is now effectively neutral to static on this seed; ROCm was positive in this run. Further reduction requires making the probe itself cheaper or less frequent when policy gates keep rejecting work. |
| RCCL/NCCL maintenance readiness fix, seed 303, divisor 15, `benchmark_results/qwen36_moe_llep_rocm1024_readiness_hardened_20260701T042229Z/`, `benchmark_results/qwen36_moe_llep_rocm2048_readiness_hardened_20260701T042542Z/`, and `benchmark_results/qwen36_moe_llep_cuda1024_readiness_sanity_20260701T041605Z/` | ROCm2 LLEP 1024 passed at 536.58 prefill / 68.16 decode tok/s after rendezvous hardening. ROCm2 LLEP 2048 passed at 538.74 prefill / 64.91 decode tok/s after rendezvous hardening. CUDA2 LLEP 1024 passed at 1850.58 prefill / 125.24 decode tok/s. All runs exited 0 and rebalance traces had paired per-device probe and metadata/payload rows. | The earlier ROCm stuck-worker failure was not an RCCL graph-capture limitation. The root cause was asymmetric host-side completion observation before blocking publish/apply rendezvous. The fix keeps RCCL collectives graph captured, keeps dedicated maintenance overlap enabled, and avoids eager recapture across request reset. |
| CUDA/ROCm 2048 economics after readiness hardening, seed 303, `benchmark_results/qwen36_moe_rocm2048_static_control_readiness_hardened_20260701T043036Z/`, `benchmark_results/qwen36_moe_llep_rocm2048_readiness_hardened_20260701T042542Z/`, `benchmark_results/qwen36_moe_cuda2048_static_llep_readiness_hardened_20260701T043422Z/`, and `benchmark_results/qwen36_moe_cuda2048_llep_divisor10_20260701T043937Z/` | ROCm static 60.70 decode tok/s vs LLEP divisor15 64.91 (+6.9%) with 15 planned arrivals, 33 transfer arrivals, and pre/post imbalance 0.0449 -> 0.0412. CUDA static 124.50 vs LLEP divisor15 124.18 (-0.25%) with 18 planned arrivals, 44 transfer arrivals, and imbalance 0.0386 -> 0.0329. CUDA divisor10 reduced movement to 3 planned / 5 transfer arrivals and reached 124.39 tok/s, but router-used events collapsed from 7430 to 10. | ROCm is now clearly speed-positive on this seed. CUDA is close to static but does not monetize the same movement policy. Simply making the spread gate stricter reduces overhead but also removes nearly all router benefit. The next CUDA lever should combine remaining-token payback, post-wave imbalance, and router-use benefit prediction rather than only per-arrival spread improvement. |
| CUDA2/ROCm2 explicit transfer-backed LLEP smoke after removing resident fallback, seed 303, 1024, one measured rep, `benchmark_results/qwen36_moe_llep_explicit_path_smoke_20260701_102100/` | CUDA static 3239.71 prefill / 126.75 decode tok/s vs LLEP 2110.41 prefill / 124.03 decode. ROCm static 1123.78 prefill / 63.12 decode vs LLEP 736.98 prefill / 63.62 decode. All runs exited 0 and required prefill graphs were captured. CUDA prefill graph nodes rose from 1915 static to 2595 LLEP; ROCm rose from 1713 to 2433. | Correctness and graph capture are now healthy on the explicit path, but transfer-backed LLEP prefill currently adds too much graph work for a one-row smoke to pay. The next optimization target is making current-batch LLEP conditional/bucketed without host policy fallback: skip zero-arrival payload work, reduce added prefill graph nodes, and gate the transfer-backed path by expected payback. |
| CUDA2/ROCm2 no-cache LLEP ownership-transfer smoke after fixing the post-spread ceiling, seed 303, 1024, perfstats on, `benchmark_results/qwen36_moe_llep_ownership_transfer_perfstats_20260701_125537/` | CUDA LLEP reached 2443.75 prefill / 124.54 decode tok/s, with prefill graph replay active, 3 planned arrivals, 3 applied windows, `1.44 MB` useful payload moved, and `660` accepted spread-improvement units. ROCm LLEP reached 814.42 prefill / 65.25 decode tok/s, with prefill graph replay active, 6 planned arrivals, 3 applied windows, `2.88 MB` useful payload moved, and `1600` accepted spread-improvement units. | The LLEP maintenance planner is no longer a no-op: improving no-cache waves now publish durable whole-expert ownership transfers and compact payload movement on both backends. One wave per backend was still rejected by cost/post-spread gates, so the next tuning target is policy economics, not missing movement plumbing. |
| CUDA2/ROCm2 static vs no-cache LLEP after source-owner apply fix, seed 303, 1024, clean rows, `benchmark_results/qwen36_moe_cuda_static_llep_after_source_owner_fix_20260701_133035/` and `benchmark_results/qwen36_moe_rocm_static_llep_after_source_owner_fix_20260701_133353/` | CUDA static 3254.50 prefill / 127.80 decode vs LLEP 2448.14 prefill / 125.77 decode (`-1.58%` decode). ROCm static 1137.83 prefill / 67.37 decode vs LLEP 789.30 prefill / 61.66 decode (`-8.48%` decode). | Fixing source ownership removed a functional correctness bug and recovered most of the prior CUDA clean-run cliff (`119.95 -> 125.77 tok/s`), but whole-expert LLEP decode movement is still not reliably economic. The current decode proxy needs stronger payback gating, likely using remaining-token/request-horizon estimates and realized post-move load benefit. Paper-aligned row-span LLEP remains the better fit for prefill/batched decode. |
| CUDA2/ROCm2 post grouped-decode fail-fast smoke, seed 303, 1024, `benchmark_results/qwen36_moe_static_llep_after_failfast_decode_20260701_140905/` and CUDA perfstats rerun `benchmark_results/qwen36_moe_cuda_llep_failfast_perfstats_20260701_141821/` | Clean smoke exited 0: CUDA static 3244.88 prefill / 126.78 decode vs LLEP 2449.22 prefill / 119.31 decode; ROCm static 1147.81 prefill / 63.21 decode vs LLEP 799.36 prefill / 65.70 decode. CUDA LLEP perfstats rerun reached 2442.06 prefill / 124.28 decode with zero planned/applied arrivals and zero payload slots. | The fail-fast cleanup did not trip production Qwen3.6 decode paths and post-cleanup CUDA2/ROCm2 decode parity passed. The one low CUDA LLEP smoke row was not reproduced under perfstats and had no movement counters; treat it as noisy end-to-end evidence, not a policy signal. |
| CUDA2/ROCm2 graph-level prefill LLEP gate, seed 303, 1024, `benchmark_results/qwen36_moe_static_llep_prefill_graph_gate_20260701_152004/` | Prefill LLEP graphs now match static node counts: CUDA `1915`, ROCm `1713`. CUDA static 3219.13 prefill / 124.57 decode vs LLEP 3208.87 prefill / 115.50 decode. ROCm static 1147.78 prefill / 66.37 decode vs LLEP 1144.06 prefill / 62.19 decode. LLEP still accepted movement: CUDA `12` plan entries / `8.65 MB` selected transport capacity; ROCm `10` plan entries / `8.65 MB`. | The prefill cliff was correctly removed by graph construction, but decode movement remained over-admitted. This separated the problem cleanly: small prefill was fixed, while no-cache whole-expert decode movement needed a stronger relative benefit gate. |
| CUDA2/ROCm2 relative movement default, seed 303, 1024, `benchmark_results/qwen36_moe_static_llep_relative_default_20260701_154152/` | CUDA static 3219.12 prefill / 124.57 decode vs LLEP 3220.32 prefill / 123.78 decode. ROCm static 1146.62 prefill / 63.58 decode vs LLEP 1145.72 prefill / 65.85 decode. Both backends planned/applied zero arrivals; `device_rebalance_llep_standard_ep_selected=12` and `device_rebalance_llep_skipped_insufficient_spread_improvement=12`. | Default `device_min_load_spread_improvement_divisor=15` pruned the negative movement waves and kept prefill static-shaped. CUDA no-work overhead remained slightly visible in this one-row smoke; ROCm was positive by noise/overhead. |
| CUDA2/ROCm2 progressive no-work backoff smoke, seed 303, 1024, `benchmark_results/qwen36_moe_static_llep_progressive_backoff_20260701_155341/` | CUDA static 3226.19 prefill / 124.21 decode vs LLEP 3217.30 prefill / 125.26 decode. ROCm static 1147.29 prefill / 62.36 decode vs LLEP 1144.38 prefill / 61.55 decode. Both backends again planned zero arrivals and moved zero payload. `device_maintenance_graph_no_work_completions=6`, `device_maintenance_graph_skipped_no_work_backoff=6`; effective backoff tags remained `1` in this short run because each maintenance cache only reached its first no-work streak. | Progressive backoff is wired and observable but needs longer no-work runs to matter. For 1024-token rows, the remaining LLEP/static delta is mostly measurement noise plus probe overhead, not payload movement. |
| ROCm2 1024, current clean static vs dynamic-hot10 | 68.99 vs 66.56 tok/s | Persistent hot10 cache is not automatically economic. |
| ROCm2 2048, current clean static vs dynamic-hot10 | 65.75 vs 58.97 tok/s | Longer generation did not rescue this cache policy sample. |
| Earlier CUDA/ROCm plumbing recovery samples | static and no-work dynamic near 125-127 CUDA tok/s, dynamic sometimes positive by noise to a few percent | The remaining problem is policy economics, not basic decode plumbing. |

Interpretation:

- Static speed has been recovered when the rebalance path is inactive or
  resident-only.
- The router can see hot-cache residency and can use it without changing top-k
  choices.
- Current hot10 maintenance can lose despite correct mechanics, so LLEP,
  Dynamic, and hot-cache admission should be evaluated independently.
- A useful policy must predict payback from controller-visible features:
  post-wave load spread, remaining tokens after apply, selected payload bucket,
  router eligible/use rates, and expected payload bytes.
- Fixed benchmark seed alone is not enough to guarantee identical expert
  histogram streams across policy variants on CUDA. Clean policy A/B work needs
  either histogram trace replay or a stronger deterministic decode harness.

## Required Perf Counters

Every rebalance strategy run should expose:

- Pre/post participant load spread and imbalance ratio.
- Selected experts, accepted/rejected moves, and rejection reason.
- Planned, copied, applied, and resident-only arrivals.
- Payload bucket slots, useful bytes, reserved bytes, wasted bytes, and
  bytes-per-accepted-spread-unit.
- Router hot-cache eligible dispatches, used dispatches, improved dispatches,
  and default-vs-actual load spread.
- Maintenance graph kind, launch count, GPU elapsed, and no-work/payload-skip
  counts.
- LLEP-specific rows: native rows, spilled rows, foreign experts, min-chunk
  rejects, lambda skips, predicted ROI, and realized spread.

## Required Tests

Correctness gates:

- Unit tests for Static, Dynamic, and LLEP shared policy helpers, including
  variable domain size, non-contiguous owner maps, capacity overflow, and no-op
  windows.
- CUDA and ROCm synthetic assignment/apply tests with identical expected command
  buffers and runtime-table state.
- CUDA2 and ROCm2 PyTorch parity suites for expert overlay prefill and decode,
  using the established parity CSV format and canonical thresholds.
- Snapshot infrastructure coverage for expert overlay.
- E2E HTTP server and prefix-cache tests after any cache or runtime-table
  change.
- Full unit suite after touching shared policy, prefix cache, graph capture, or
  device ABI.

Performance gates:

- Static, Observe, Dynamic no-cache, Dynamic plus cache, and LLEP where
  available.
- CUDA2 and ROCm2 separately.
- 512, 1024, 2048, and selected 4096-token continuations with expanded context.
- At least one warmup plus repeated measured runs for clean speed labels; use
  single-request trace mode only for policy feature training.

## Invariants

- No default streams. Every kernel, collective, and copy uses an explicit stream.
- No silent CPU fallback in homogeneous GPU device-side mode.
- No per-window host publish/apply or host policy decision in steady state.
- No new independent per-token rebalance collective.
- Do not use eager graph recapture as a correctness crutch; request reset should
  preserve captured graph replay when graph shape and bindings are still valid.
- No fixed two-card assumptions; domain size is `degree >= 2`.
- CUDA and ROCm must remain aligned in ABI, tests, and feature behavior.
- Same-backend expert arrivals use descriptor copy, not repack.
- Empty slots must not be transferred.
- If graph capture is required and unsupported, fail hard.
- If an optimized GPU decode path is enabled, it must either run or fail hard;
  do not silently retry a different path.
- Router top-k expert ids and route weights must remain exact.
- Shared experts follow dense policy. Routed experts are the only rebalanced
  experts.

## Parity Profiling Snapshot

Added an opt-in parity wall-time profiler via `LLAMINAR_PARITY_PROFILE=1`.
Focused Qwen3.6 MoE ExpertOverlay decode parity runs show the slow tests are
dominated by full two-device runner construction, not the decode/snapshot
comparison body:

- CUDA2 LLEP decode parity: 69.7s test wall; 61.9s setup, 52.0s
  `createRankOrchestrator`, 9.1s model context load, 6.0s decode parity body,
  0.9s total decode graph execution for three steps.
- ROCm2 LLEP decode parity, warm run: 104.5s test wall; 89.0s setup, 74.2s
  `createRankOrchestrator`, 9.2s model context load, 5.3s LocalTP/RCCL context,
  13.1s decode parity body, 2.3s total decode graph execution for three steps.
- ROCm2 Dynamic decode parity: 105.2s test wall; 93.0s setup, 76.8s
  `createRankOrchestrator`, 9.0s model context load, 6.9s LocalTP/RCCL context,
  10.1s decode parity body, 2.4s total decode graph execution for three steps.
- PyTorch snapshot `.npy` loads are not material at this scale: about 1,203
  loads take under 80ms total once snapshots already exist. Layer comparison is
  about 140-160ms total for three decode steps.

The main suite-speed target is therefore reuse or amortization of
`RankOrchestrator` setup and device weight finalization/upload/preparation
across parity cases, or separating kernel-level decode profiling from full
runner reconstruction when the full production graph is not required.

## Next Work

1. Clean up the remaining public policy surface so `LLEP` and
   `HotExpertReplicaCache` are explicit and independent of `Dynamic`.
2. Run transfer-backed LLEP through CUDA2 and ROCm2 parity, then clean repeated
   benchmark matrices. Include both no-cache and hot-cache-admission variants so
   LLEP, Dynamic, and cache effects stay separable.
3. Use the Dynamic CLI/DebugEnv knobs for bounded CUDA2/ROCm2 sweeps at 1024
   and 2048 tokens, starting with the now-wired relative LLEP cost gate
   (`LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR`) to
   reject low-value whole-expert arrivals. Only revisit lower imbalance
   thresholds, larger per-layer swap counts, and larger per-wave command caps
   after the high-value-only gate recovers no-movement/static speed.
4. Extend Dynamic, if needed, with an explicit hot-set variant
   (`hotset_expert_count=20`), configurable window schedule, variable domain
   size, and cost-gated non-empty moves while keeping the shared helper as the
   single policy implementation used by CPU, CUDA, and ROCm.
5. Keep LLEP modular behind the shared routed-assignment surface so additional
   algorithms can be added without backend drift.
6. Wire production prefill and batched decode to run LLEP only when row count
   and cost gates make it worthwhile.
7. Run parity, E2E prefix-cache/server tests, full unit tests, then clean
   CUDA2/ROCm2 benchmark matrices before making a policy the default.

Historical checkpoint notes were intentionally removed from this file. Use the
bench artifacts under `benchmark_results/` and commit messages for raw run
history.
