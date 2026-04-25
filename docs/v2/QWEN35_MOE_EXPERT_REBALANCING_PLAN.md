# Qwen3.5 MoE Decode Expert Utilization and Socket Rebalancing Plan

**Status:** Project plan
**Scope:** CPU dual-socket Qwen3.5 MoE decode throughput
**Related design:** `docs/v2/MOE_EXPERT_PLACEMENT_DESIGN.md`

## Problem

Qwen3.5 MoE decode currently partitions experts statically across tensor-parallel ranks or local devices. `MoEFFNStage` already skips experts outside the local expert range, and 3D expert tensors can be loaded as expert-parallel slices. The decode path does not yet use live routing statistics to correct hot-expert skew across CPU sockets, so one socket can do most of the active top-k expert work while the other waits for the MoE output reduction.

## Goals

1. Track expert utilization during decode with low overhead.
2. Build per-layer histograms that identify hot, cold, and imbalanced experts.
3. Dynamically rebalance expert placement across CPU sockets without changing model numerics.
4. Keep rebalancing deterministic, debuggable, and safe to disable.
5. Measure whether socket-aware rebalancing improves decode throughput beyond static expert parallelism.

## Non-goals

- Changing router probabilities or top-k selection.
- Introducing GPU-specific expert migration in the first implementation.
- Repacking expert weights on every decode step.
- Replacing the existing `MoEFFNStage` execution path before histogram data proves it is necessary.

## Current Code Touchpoints

- `src/v2/execution/compute_stages/stages/MoEFFNStage.cpp`
  - Single-token decode routes experts, filters by `local_expert_start/local_expert_count`, and executes local active experts.
- `src/v2/execution/moe/ExpertPlacementMap.{h,cpp}`
  - Existing implementation stores expert-to-device placement and a simple activation histogram.
- `src/v2/execution/moe/ExpertRebalancer.{h,cpp}`
  - Existing implementation proposes hot/cold expert moves from aggregate activation counts.
- `src/v2/loaders/WeightManager.cpp`
  - Supports loading expert-parallel slices for 3D expert tensors.
- `src/v2/execution/local_execution/graph/GraphResolver.cpp`
  - Graph-resolved `TPMode::ExpertParallel` is still a missing feature. The current behavior lives inside `MoEFFNStage` filtering logic; a later refactor should turn that implicit stage-local behavior into explicit graph stages and collectives.

## Proposed Architecture

### 1. Decode Expert Histogram Tracker

Add an explicit decode-oriented tracker rather than relying only on the existing flat activation count.

Tracked dimensions:

- layer id
- expert id
- socket/device id
- decode token window
- selected top-k slot
- optional routing weight sum

Core metrics:

- activations per expert per layer
- weighted activations per expert per layer
- activations per socket per layer
- top-k slot frequency per expert
- imbalance ratio between sockets
- moving-window hotness score

The first version should use fixed-size per-layer arrays for integer counts and optional routing-weight sums. Keep the fast path allocation-free after initialization.

### 2. Record Routing in `MoEFFNStage`

Record selected experts immediately after routing in decode mode:

- count every selected expert, including experts not local to the current socket
- record local-vs-remote ownership based on the current placement map
- expose debug dumps through existing stage dump or logging facilities

This preserves the current compute path while collecting the data needed to rebalance.

### 3. Socket-Aware Placement Model

Extend `ExpertPlacementMap` or add a companion placement policy for CPU sockets:

- represent two CPU sockets as distinct `DeviceId` targets, e.g. `cpu:0` and `cpu:1`
- support per-layer placement rather than one global expert-to-device vector if Qwen3.5 layers show different expert hotness
- retain a global fallback mode for simpler tests
- track placement generation so cached decode graphs can detect changes

Initial placement should continue to support static contiguous/equal splits. Rebalancing should optionally switch to histogram-derived placement after warmup.

### 4. Rebalance Algorithm

Use periodic moving-window rebalance decisions:

1. Wait until the histogram window has enough samples.
2. Compute expected socket work per layer from selected experts.
3. Identify hot experts on overloaded sockets and cold experts on underloaded sockets.
4. Propose swaps that reduce estimated imbalance while preserving equal memory footprint.
5. Apply at most a small number of moves per cycle.
6. Reset or decay the window after each accepted rebalance.

Safety constraints:

- only rebalance at decode-step boundaries
- never move an expert while a stage is executing
- avoid moves unless expected imbalance reduction exceeds a threshold
- use hysteresis to avoid oscillation
- keep a hard cap on moves per layer per cycle

### 5. Weight Residency and GEMM Engine Handling

For CPU sockets, expert movement should prefer pointer/ownership reassignment over reloading model weights:

- if both sockets have access to shared mmap-backed weights, first rebalance placement metadata and NUMA execution affinity
- if socket-local packed GEMM engines are required, prepare both old and new socket engines outside the decode critical path
- avoid destructing/recreating temporary tensors in the hot path
- keep the existing expert-parallel sliced-loading path as the memory-saving mode

Two implementation modes should be supported:

- **metadata-only rebalance:** fastest to implement; updates placement metadata and execution affinity without physically moving weight data, so it may use remote NUMA memory if packed weights remain on the original socket
- **socket-local repack rebalance:** higher payoff, requires asynchronous repack/prewarm before applying placement

### 6. Scheduling and Synchronization

Apply rebalances through the orchestration layer rather than inside low-level kernels:

- collect per-rank/socket histograms locally
- reduce histograms across the two CPU socket participants
- elect a deterministic rebalance proposal
- broadcast/apply the same placement generation to both sockets
- ensure the downstream MoE allreduce sees outputs from the same placement generation

The initial dual-socket CPU version can use existing MPI/local TP synchronization primitives. Avoid new collectives until graph-level `TPMode::ExpertParallel` is implemented.

### 7. Configuration

Add feature flags with conservative defaults:

- `--moe-decode-histogram` to enable histogram tracking
- `--moe-rebalance=off|observe|dynamic`
- `--moe-rebalance-window <tokens>`
- `--moe-rebalance-min-samples <count>`
- `--moe-rebalance-max-moves <count>`
- `--moe-rebalance-threshold <ratio>`

Recommended rollout defaults:

- tracking disabled by default until overhead is measured
- `observe` mode for logging/profiling without moving experts
- `dynamic` mode only after correctness and throughput validation

### 8. Diagnostics

Add structured logs and optional CSV output:

- per-layer top experts
- per-socket activation totals
- proposed moves
- accepted/rejected moves with reasons
- placement generation
- decode tokens/sec before and after each rebalance

Suggested environment variables:

- `LLAMINAR_MOE_HISTOGRAM_LOG=1`
- `LLAMINAR_MOE_REBALANCE_TRACE=1`
- `LLAMINAR_MOE_REBALANCE_CSV=/tmp/llaminar_moe_rebalance.csv`

## Milestones

### Milestone 1: Observe-only histograms

- Add a per-layer decode histogram data structure.
- Wire decode routing events from `MoEFFNStage`.
- Add unit tests for histogram counting, window reset, weighted counts, and top-k slot tracking.
- Add observe-mode logging.

Exit criteria:

- histogram collection is correct and allocation-free during steady-state decode
- disabled tracking has negligible overhead

### Milestone 2: Rebalance proposal engine

- Extend or replace `ExpertRebalancer` with socket-load-aware proposals.
- Support per-layer hot/cold swaps.
- Add hysteresis, thresholds, and move caps.
- Add tests for skewed, balanced, and oscillating routing distributions.

Exit criteria:

- proposals reduce estimated socket imbalance in synthetic tests
- balanced workloads produce no movement

### Milestone 3: Metadata-only dynamic placement

- Add placement generation tracking.
- Apply rebalances at decode-step boundaries.
- Update `MoEFFNStage` local expert filtering to consult current placement when dynamic placement is enabled.
- Add observe and dynamic integration tests using synthetic routing traces.

Exit criteria:

- output numerics are unchanged for deterministic routing
- dynamic placement can be enabled and disabled at runtime configuration level

### Milestone 4: Socket-local packed engine prewarming

- Prepare moved experts' GEMM engines on the target socket before applying the placement generation.
- Keep old placement active until target engines are ready.
- Add rollback if prewarming fails.

Exit criteria:

- rebalancing does not introduce first-token latency spikes after movement
- moved hot experts execute on target socket-local packed engines

### Milestone 5: Benchmark and tune

- Benchmark static EP, observe-only, metadata-only dynamic, and socket-local dynamic modes.
- Report decode tokens/sec, per-token MoE latency, allreduce time, socket imbalance, and rebalancing overhead.
- Tune default thresholds for Qwen3.5 35B-A3B and 122B-A10B.

Exit criteria:

- dynamic mode improves throughput on skewed decode workloads
- no regression on balanced or short prompts

## Test Plan

- Unit tests:
  - histogram counting and reset
  - weighted hotness scores
  - rebalance proposal generation
  - hysteresis and max-move limits
  - placement generation transitions
- Integration tests:
  - synthetic dual-socket expert routing distribution
  - deterministic decode with rebalancing disabled vs enabled
  - observe mode does not change output
- Performance tests:
  - Qwen3.5 MoE decode static EP vs dynamic rebalancing
  - measure overhead with histogram enabled but no moves
  - measure warmup and post-rebalance latency spikes

## Risks and Mitigations

- **Histogram overhead in decode:** use preallocated arrays and only record top-k integers by default.
- **Placement thrashing:** use moving windows, thresholds, cooldowns, and max moves.
- **NUMA remote-memory regressions:** separate metadata-only mode from socket-local repack mode and benchmark both.
- **Graph/cache invalidation bugs:** introduce placement generations and apply changes only at decode-step boundaries.
- **Collective mismatches:** make placement changes deterministic and synchronized across socket participants.

## Suggested Implementation Order

1. Land observe-only histogram tracking.
2. Add socket-load-aware proposal tests.
3. Add metadata-only placement rebalancing behind `observe|dynamic` config.
4. Add socket-local packed-engine prewarming.
5. Refactor stage-local expert filtering into graph-resolved `TPMode::ExpertParallel` stages and collectives once the dynamic path is stable.
