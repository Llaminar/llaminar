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
  and transfer-cost constraints. It can import foreign expert payloads for the
  current batch/window without changing long-lived ownership.

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

Important tunables:

- `lambda`: imbalance threshold for falling back to normal AE.
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
- The GPU prefill LLEP bridge is now span-aware for already-resident experts:
  repeated rows for a hot expert can be split across the least-loaded resident
  participants, preserving router top-k and route weights while finally making
  hot-cache/replica residency visible to grouped prefill load balancing.
- The resident-span prefill bridge now uses first-class runtime-table scratch
  for a compact `[expert][participant]` split table. CUDA and ROCm use the
  shared LLEP resident water-fill helper to plan split counts in chunks, then
  fill per-route participant assignments in parallel. This removes the older
  single-thread per-row split loop from both backend kernels.
- Transfer-backed LLEP now feeds the same graph-capturable compact-arrival path
  as Dynamic. The shared transfer-only LLEP planner emits foreign whole-expert
  arrivals, and CUDA/ROCm controller kernels publish `ExpertPayloadArrival`
  commands without enabling the hot-cache policy.
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
- LLEP is mechanically transfer-backed but not yet performance-proven. Current
  public plumbing still expresses it as routed assignment plus Dynamic
  maintenance; promote it to a first-class strategy only after parity and
  repeated clean benchmark matrices.
- Hot-cache policy economics are inconsistent. The cache is mechanically visible
  to routing, but persistent hot10 alone is not a reliable speedup.

## Current Measurements

Raw run trails live in `benchmark_results/` and commit messages. This document
keeps only the current design signal:

| Scenario | Result | Takeaway |
| --- | --- | --- |
| CUDA2 512, seed 303, static vs Dynamic no-cache, 2026-06-30 | 122.74 vs 125.80 tok/s decode; prefill 3235.87 vs 3227.29 tok/s | Dynamic path is healthy after the CUDA descriptor rebind fix. This sample had no payload movement (`payload_bucket_slots=0`, `apply_changed_layers=0`), so it proves overhead recovery, not policy benefit. |
| ROCm2 512, seed 303, static vs Dynamic no-cache, 2026-06-30 | 66.13 vs 65.63 tok/s decode; prefill 1145.22 vs 1144.90 tok/s | ROCm stayed healthy after the shared rebind guard. This sample also had no movement, so policy tuning still needs movement-positive traces. |
| CUDA2 1024, seed 303, recent clean static vs dynamic-hot10 | 126.83 vs 124.81 tok/s | Router/cache mechanics work, but hot10 maintenance was net negative in this run. |
| CUDA2 Dynamic knob sweep, seed 303, 1024/2048, 1 measured iter, perfstats on, `benchmark_results/qwen36_moe_cuda_dynamic_knob_sweep_seed303_20260630_210138/` | 1024: static 123.87, no-op Dynamic 123.49, default Dynamic 123.77, threshold1100 125.53, threshold1100+min0 123.77, aggressive 123.70 tok/s. 2048: static 124.98, no-op 125.21, default 124.68, threshold1100 124.14, threshold1100+min0 124.82, aggressive 124.23 tok/s. | No-op controller overhead was small in this run (`~4 ms` maintenance GPU elapsed at 1024, `~8 ms` at 2048, no movement). Lowering threshold to 1100 induced payload movement and produced the only 1024 positive signal (+1.34%), but movement was neutral/negative at 2048. Generated token streams diverged before the first maintenance window, so this bounds end-to-end behavior rather than replaying identical histogram inputs. Repeat candidates with more reps and/or trace replay before treating this as a policy win. |
| CUDA2 transfer-backed LLEP probe, seed 303, 1024, perfstats + trace, `benchmark_results/qwen36_moe_cuda_llep_transfer_probe_seed303_20260630_234622/` | static 123.91 vs dynamic LLEP 124.11 tok/s decode; prefill 1953.23 vs 1942.05 tok/s. Trace planned/applied two arrivals around token 323 and one arrival around token 835. | The transfer-backed path is real: `ExpertPayloadArrival` commands are planned, compact payload slots move useful bytes, runtime-table apply makes new residents visible, and router residency counters become nonzero. Payback was essentially flat in this one-run diagnostic, and the maintenance lane is still dedicated rather than fully fused into main decode collectives. |
| ROCm2 transfer-backed LLEP probe, seed 303, 1024, perfstats + trace, `benchmark_results/qwen36_moe_rocm_llep_transfer_probe_seed303_20260630_235145/` | static 66.08 vs dynamic LLEP 65.22 tok/s decode; prefill 524.69 vs 525.68 tok/s. Trace planned two reciprocal arrivals around token 324 and one later arrival around token 836. | ROCm matches CUDA mechanically but was negative in this one-run sample. This points to shared policy/cost-gate economics rather than a CUDA-only or ROCm-only wiring issue. |
| CUDA2/ROCm2 resident split chunk-planner probe, seed 303, 1024, `assignment=least-loaded-ep`, perfstats + trace, `benchmark_results/qwen36_moe_llep_chunk_split_probe_20260701_000943/` | CUDA: static 1855.37 prefill / 124.68 decode vs dynamic 1841.16 prefill / 124.51 decode. ROCm: static 537.28 prefill / 67.66 decode vs dynamic 537.47 prefill / 65.48 decode. Both dynamic runs planned 12 arrivals and requested 8 compact payload slots across the two participants; one arrival was applied before run end. Offline full-LLEP replay still predicts large spread reductions at ROI 256: CUDA spread 7015 -> 3955 with 10 transfers, ROCm 5664 -> 2518 with 11 transfers. | The shared chunk planner removes a real single-threaded kernel wart without changing semantics, but current live transfer-backed LLEP is still a coarse whole-expert-arrival/resident-assignment proxy. The remaining speed gap is implementation strategy, not absence of routing imbalance signal. |
| Compact payload capacity sweep, seed 303, 1024, dynamic LLEP only, `benchmark_results/qwen36_moe_llep_payload_slots2_probe_20260701_001956/` and `benchmark_results/qwen36_moe_llep_payload_slots4_probe_20260701_002406/` | CUDA dynamic: slots1 124.51 decode, slots2 123.86, slots4 118.80 tok/s. ROCm dynamic: slots1 65.48, slots2 64.55, slots4 64.69 tok/s. Slots2/4 planned and applied more arrivals on CUDA, but throughput fell. | Increasing whole-expert arrival capacity does not recover LLEP economics. Keep the compact payload default conservative and focus on true row-span LLEP for prefill/batched work or stronger admission gates for decode hot-cache movement. |
| CUDA2/ROCm2 current Dynamic load-ratio trace, seeds 303 and 606, 1024/2048, `assignment=least-loaded-ep`, trace mode, `benchmark_results/qwen36_moe_policy_loadratio_current_20260701T004332Z/` | CUDA 1024: Dynamic made no transfers and averaged -0.38 tok/s vs static. CUDA 2048: Dynamic reduced pre/post imbalance ratio from 0.0441 to 0.0344 but averaged -1.19 tok/s. ROCm 1024: no transfers, average -2.37 tok/s with one slow outlier. ROCm 2048: imbalance ratio 0.0409 -> 0.0319 but average -1.83 tok/s; seed 303 was positive (+3.38) and seed 606 negative (-7.05). | The controller can now quantify real imbalance reduction, but these whole-expert transfer waves are not consistently economic. A cost gate based on improvement relative to observed load is needed before more movement-capacity tuning. This run exposed that the LLEP path ignored `min_load_spread_improvement_divisor`; that is now fixed. |
| Relative LLEP cost-gate probes, seed 303, 2048, `MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR=15`, `benchmark_results/qwen36_moe_llep_relative_gate15_cuda_seed303_20260701T012906Z/` and `benchmark_results/qwen36_moe_llep_relative_gate15_rocm_seed303_20260701T013119Z/` | CUDA dynamic moved 4 arrivals, reduced imbalance 0.0434 -> 0.0341, and reached 125.67 tok/s decode versus the same-run static row from the divisor-25 A/B at 125.53 tok/s. ROCm dynamic moved 4 arrivals, reduced imbalance 0.0442 -> 0.0336, and reached 64.90 tok/s decode versus the earlier same-seed static row at 60.18 tok/s. | The relative floor is the right sweep axis: divisor 25 partially pruned CUDA arrivals (20 -> 14) and improved the delta (-1.40 -> -0.40 tok/s), while divisor 15 pruned to a high-value 4-arrival wave and recovered a positive single-row signal. This needs repeated CUDA/ROCm matrices before becoming a default. |
| CUDA2/ROCm2 no-work maintenance probe split, seed 303, 1024, divisor 15, `benchmark_results/qwen36_moe_probe_split_seed303_20260701T023641Z/` | CUDA: static 1853.63 prefill / 124.75 decode vs Dynamic 1839.74 prefill / 124.67 decode. ROCm: static 535.96 prefill / 62.78 decode vs Dynamic 537.73 prefill / 65.86 decode. Dynamic planned zero arrivals on both backends; traces exported only `probe` stages. No metadata/payload graph launched. | Splitting probe from metadata/payload replay removed command-buffer allgathers from no-work windows. No-work maintenance GPU elapsed fell versus the prior accounting-fix probe from ~45.6 ms to ~32.8 ms on CUDA and ~136.9 ms to ~100.2 ms on ROCm over six maintenance launches. CUDA no-work Dynamic is now effectively neutral to static on this seed; ROCm was positive in this run. Further reduction requires making the probe itself cheaper or less frequent when policy gates keep rejecting work. |
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
- No fixed two-card assumptions; domain size is `degree >= 2`.
- CUDA and ROCm must remain aligned in ABI, tests, and feature behavior.
- Same-backend expert arrivals use descriptor copy, not repack.
- Empty slots must not be transferred.
- If graph capture is required and unsupported, fail hard.
- Router top-k expert ids and route weights must remain exact.
- Shared experts follow dense policy. Routed experts are the only rebalanced
  experts.

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
