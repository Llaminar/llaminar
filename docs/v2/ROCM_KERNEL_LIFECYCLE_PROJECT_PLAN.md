# ROCm Async Weight Load/Repack Project Plan

## Goal
Reduce model startup latency (load + pack + upload) by introducing asynchronous GPU weight preparation with tensor-parallel-safe ownership and memory reclamation.

## Problem Summary
- Startup currently serializes weight load, GEMM packing, and non-GEMM uploads.
- `packGemmWeights()` and `uploadNonGemmWeights()` are synchronous and historically lock-scoped for long operations.
- In LOCAL TP, tensors are shared/cloned per device; freeing original host/unrepacked buffers too early can break consumers.

## Scope
- `WeightManager` preload path and weight packing/upload orchestration.
- `KernelFactory` packed-weight cache integration points.
- ROCm pre-packed GEMM path (`ROCmPackedWeights` upload lifecycle).

## Non-Goals (Phase 1)
- No end-to-end redesign of weight ownership model.
- No changes to TP/PP placement semantics.
- No GPU-side repack kernel introduction yet (that is Phase 3).

## Design Constraints
- Preserve current correctness first.
- Keep tensor cache ownership in existing tensor cache fields (`cache_`, `cuda_cache_`, `rocm_cache_`).
- Avoid broad API changes while proving overlap and stability.

## Phases

### Phase 1 — Async Overlap (Start Here)
Objective: overlap GEMM packing with non-GEMM uploads for the same device.

Tasks:
1. Reduce lock hold time in `WeightManager::packGemmWeights()` and `WeightManager::uploadNonGemmWeights()` so long GPU/packing work occurs outside global cache lock.
2. In runner setup, launch GEMM packing asynchronously while uploading non-GEMM weights on the main path.
3. Join/wait before inference starts so behavior remains deterministic.
4. Preserve existing fallback behavior (warnings on preload failures, lazy creation still allowed).

Exit Criteria:
- Weight preload completes correctly with no behavior regressions.
- Observed overlap between GEMM packing and non-GEMM upload in startup logs/timers.
- No new race/crash in single-device or LOCAL TP startup.

### Phase 2 — Ticketed Readiness + Safe Reclaim
Objective: formalize per-weight/device readiness and host-memory reclamation safety.

Tasks:
1. Add per-weight/device prep ticket state (`LoadedHost`, `PackedHost`, `UploadedDevice`, `Ready`).
2. Add completion event/flag synchronization for wait-free readiness checks where possible.
3. Add TP-aware completion accounting so host/unrepacked buffers are released only when all required device tickets are `Ready`.

Exit Criteria:
- No premature host-data release under LOCAL TP.
- Memory reclaim happens automatically once safe.

### Phase 3 — GPU-Side Repack Path
Objective: reduce CPU repack cost by moving repack work to GPU kernels where beneficial.

Tasks:
1. Introduce optional ROCm GPU repack for selected formats.
2. Keep compatibility with existing packed cache structures.
3. Gate by feature flag and compare against CPU repack baseline.

Exit Criteria:
- Measurable startup reduction for targeted models/devices.
- No regressions in correctness or memory safety.

## Tracking Checklist

### Phase 1 Checklist
- [x] Narrow lock scopes in `WeightManager` preload methods.
- [x] Add async overlap in `InferenceRunnerFactory` preload setup.
- [x] Validate build and startup on ROCm path.
- [x] Validate startup on non-ROCm path (CPU/CUDA as available).

### Phase 2 Checklist
- [x] Add prep ticket state model.
- [x] Add readiness synchronization.
- [x] Add TP-safe reclaim guards.

### Phase 3 Checklist
- [x] Add optional GPU repack kernels.
- [x] Add feature flag + fallback behavior.
- [x] Benchmark and compare startup latency.

### Phase 4 (Proposed) — GPU-Native GEMM Repack Pipeline (Event-Driven)
Objective: move startup GEMM repack work off CPU into GPU kernels with stream/event choreography, bounded memory, and no global `hipDeviceSynchronize()` calls.

> **Scope focus for this run:** optimize **model load + TTFT** only.
> Decode throughput is out-of-scope for Phase 4 startup work and is not used as a success metric here.

> **Architectural pivot (current):** startup CK row-major repack is disabled; startup preparation is VNNI/ratio-VNNI only.
> Rationale: current benchmark workload is decode-dominant (`M=1` GEMV fast path), where CK row-major startup repack adds cost without downstream use.

#### Why this phase
- Current startup still spends significant wall time in GEMM repack path.
- CPU-side prep plus host orchestration limits overlap and introduces serialization.
- Existing GPU repack primitives already exist for runtime CK path; Phase 4 reuses this direction for startup prepack.

#### Design principles
1. No global device syncs in startup hot path.
2. Event-based readiness between stages (`hipEventRecord` / `hipStreamWaitEvent`).
3. Bounded in-flight jobs to avoid VRAM spikes/OOM.
4. Single reusable workspace pool (ring-buffer slots), not per-weight ad hoc alloc/free.
5. Preserve eager/full upfront packing semantics (no lazy `output.weight` in this phase).

#### Target architecture
For each device preload, run a fixed-depth pipeline over 3 streams:
- `S_h2d`: host→device copies for source blocks/metadata.
- `S_repack`: GPU repack/expand kernels (format-specific).
- `S_commit`: final packed layout write/registration for GEMM engine readiness.

Each weight job advances through slot `i` in a ring-buffer pool:
1. Acquire free slot (capacity `N_slots`).
2. Enqueue H2D to slot buffers on `S_h2d`; record `E_h2d_done[i]`.
3. `S_repack` waits on `E_h2d_done[i]`, launches repack kernel(s), records `E_repack_done[i]`.
4. `S_commit` waits on `E_repack_done[i]`, publishes packed pointer/handle, records `E_commit_done[i]`.
5. Slot is reusable once commit event is observed by producer/allocator thread.

#### Synchronization model
- Replace host-side blocking waits with event polling or event chaining.
- Only synchronize at phase barrier boundaries (e.g., before graph execution starts), not per-weight.
- Use stream-local ordering + cross-stream event waits.

#### Memory safety / VRAM budget strategy
- Introduce startup repack workspace budget (MB) per device.
- Compute max slots from:
  - per-slot scratch bytes (source + destination + metadata),
  - reserved headroom, and
  - current free VRAM snapshot.
- If budget cannot fit target slots, degrade gracefully:
  - fewer slots, then
  - fewer concurrent upload/repack workers, then
  - CPU legacy repack fallback.

#### API + integration changes
1. `ROCmPackedWeights` / ROCm pack path
   - Add startup GPU-repack mode and slot-based workspace interfaces.
   - Keep existing VNNI/ratio-VNNI structures as source representation.
2. `WeightManager::packGemmWeights`
   - Add backend strategy branch: `CPU_PACK`, `GPU_REPACK_PIPELINED`.
   - Reuse current bounded producer/consumer control plane.
3. `KernelFactory`
   - Prepared-handle creation remains authoritative; readiness set when commit event completes.
4. `DebugEnv`
   - Add feature flags + tunables:
     - GPU repack pipeline is now the unconditional default for all GPU devices
     - `LLAMINAR_ROCM_REPACK_SLOTS=<n>`
     - `LLAMINAR_ROCM_REPACK_BUDGET_MB=<mb>`
     - `LLAMINAR_ROCM_REPACK_STREAMS=<n>` (optional, default fixed 3)

#### Instrumentation requirements
Add explicit startup sub-metrics:
- `weights.gemm_pack.h2d_stage`
- `weights.gemm_pack.gpu_repack_stage`
- `weights.gemm_pack.commit_stage`
- `weights.gemm_pack.slot_wait_time`
- `weights.gemm_pack.pipeline_bubble_time`

Also report effective overlap ratio:
- `overlap_efficiency = (sum_stage_times - wall_time) / sum_stage_times`.

#### Rollout plan
Step 1 — plumbing + feature flag (no behavior change)
- Add flags, stream/event wrappers, workspace slot manager.

Step 2 — single-format pilot
- Enable GPU repack pipeline for one high-impact format path first (ROCm INT8/VNNI startup path).
- Compare against current CPU-repack baseline.

Step 3 — broaden format coverage
- Add remaining quant format conversion/repack kernels needed for parity.

Step 4 — tune + harden
- Auto-slot sizing from VRAM budget.
- OOM fallback paths and stress tests.

#### Exit criteria
- Startup GEMM repack wall time reduced materially vs current pipelined CPU baseline.
- No `hipDeviceSynchronize()` in per-weight startup path.
- Stable under constrained VRAM (no OOM in standard configs).
- Correctness parity maintained (existing benchmark + prompt gate).

#### Risks
- Kernel launch overhead can negate gains for very small weights if slot sizing is poor.
- Event lifecycle bugs can cause hidden stalls/deadlocks.
- Memory fragmentation if slot allocator is not fixed-size and reusable.

#### Validation checklist (Phase 4)
- [x] Add Step 1 env/tunable plumbing and config scaffold (no behavior change).
- [x] Add flag-gated GPU startup repack pipeline.
- [x] Disable startup CK row-major repack/commit path for VNNI-only startup mode.
- [ ] Verify no per-weight global syncs in code path.
- [x] Benchmark A/B vs current baseline on `rocm:0`.
- [ ] Run constrained-budget test to validate graceful fallback.
- [ ] Confirm correctness gate output and no startup hangs.

#### Phase 4A (Begin Now) — Startup/TTFT Recovery Plan
Objective: close the gap between expected overlap and observed startup regression, then drive TTFT down with measurable overlap efficiency gains.

Current finding (Release sweep, `rocm:0`, 5x5):
- Baseline startup median (`alloc logits -> graph built`): **4.497s**
- Repack startup median: **5.075s**
- Delta: **+0.578s (+12.85%)** regression

Work items:
1. **Pinned-host H2D path for startup uploads**
   - Add pinned staging path for startup scales/payload/ratio buffers before async H2D.
   - Ensure non-pinned fallback remains safe and feature-flagged.
2. **Complete 3-stage startup pipeline (H2D -> repack -> commit)**
   - Separate commit stage from repack launch.
   - Publish readiness only after commit event, not just repack enqueue.
3. **Bounded slot lifecycle (real ring behavior)**
   - Enforce acquire/release with explicit slot occupancy accounting.
   - Add backpressure metrics for slot starvation/bubbles.
4. **Startup-only instrumentation and acceptance gates**
   - Emit and review: `weights.gemm_pack.h2d_stage`, `weights.gemm_pack.gpu_repack_stage`, `weights.gemm_pack.commit_stage`, `weights.gemm_pack.slot_wait_time`, `weights.gemm_pack.pipeline_bubble_time`.
   - Gate on startup/TTFT improvement; ignore decode tok/s in this phase.

Phase 4A checklist:
- [x] Implement pinned-host startup H2D staging.
- [x] Add flag-gated non-CK prefill scaffold branch with explicit CK fallback.
- [ ] Implement explicit commit stream/event stage.
- [ ] Implement bounded slot acquire/release accounting.
- [ ] Add startup overlap metrics + overlap efficiency report.
- [ ] Re-run Release A/B median sweep and verify startup regression is removed.
- [ ] Tune `LLAMINAR_ROCM_REPACK_STREAMS`, `LLAMINAR_ROCM_REPACK_SLOTS`, and `LLAMINAR_ROCM_REPACK_BUDGET_MB` for best startup median.

## Risks
- Overlap can expose lock-ordering/race issues that were hidden by serialized startup.
- Coarse reclaim logic can leak memory; aggressive reclaim can break TP clones.
- Additional async machinery may complicate debugging if not well-instrumented.

## Validation Plan
1. Build: `cmake --build build_v2_release --parallel`
2. Startup checks on target ROCm config (single-device and LOCAL TP where available).
3. Compare before/after for:
   - Total startup time to first token
   - GEMM pack duration vs overlapped wall-clock startup
   - Correctness/stability (no missing weights, no crashes)
4. For Phase 4A, treat startup/TTFT as the primary KPI; decode tok/s is informational only.

## Notes
- Phase 1 is intentionally low-risk and minimally invasive.
- Phase 2 introduces the primary ownership/reclaim guarantees requested for TP safety.
- Phase 3 gate (ROCm A/B, `LLAMINAR_ROCM_PACK_VNNI_ONLY`): startup total improved from 13108.2 ms to 12801.2 ms (~2.34%), with neutral/slightly noisy inference throughput delta in single-sample gate.
- Cross-backend validation passed on CUDA (`cuda:0`) with profiling-enabled benchmark startup and inference completion.
