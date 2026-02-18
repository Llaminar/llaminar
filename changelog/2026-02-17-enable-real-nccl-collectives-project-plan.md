# Enable Real NCCL Collective Operations — Project Plan

## Summary

This plan defines the work required to move LOCAL tensor parallel CUDA execution to **real NCCL collectives as the primary/required path**, instead of relying on host-staged fallback behavior when NCCL runtime setup is broken.

Scope focuses on the V2 path in `src/v2/collective/` and related LocalTP integration/parity validation.

## Why this is needed

Current behavior in `LocalTPContext::allreduceWithBarrierMultiGpu(...)` includes a host-staged FP32 fallback when NCCL allreduce fails with runtime signatures like:
- `stub library`
- `unhandled cuda error`
- `ncclAllReduce failed`

That fallback preserves correctness but hides real NCCL deployment failures and can mask performance regressions. We need a mode where NCCL is fully operational and required for CUDA LocalTP parity/perf qualification.

## Current state (as of 2026-02-17)

### Implemented foundations

- `NCCLBackend` delegates multi-buffer collectives to `NCCLCoordinator`:
  - `allreduceMultiAndSynchronize(...)`
  - `allgatherMulti(...)`
  - `broadcastMulti(...)`
  - `reduceScatterMulti(...)`
- `NCCLCoordinator` owns comms/streams/events on a dedicated coordinator thread and serializes NCCL group operations.
- LocalTP multi-device barrier flow exists and orders buffers by device index before collective launch.

### Known gap

- `LocalTPContext::allreduceWithBarrierMultiGpu(...)` currently contains host-staged fallback for NCCL runtime failures.
- In environments with broken CUDA driver/runtime wiring (e.g., stub lib), tests can pass via fallback but do not prove true NCCL execution.

## Goal definition

### Functional goal

For CUDA LocalTP (2+ GPUs), all LocalTP collectives run on NCCL and complete without host-staged fallback under supported environments.

### Non-functional goal

- Keep fallback available only as explicit opt-in emergency mode for unsupported/dev environments.
- Add observability that makes real NCCL vs fallback execution unambiguous in logs and tests.

## Out of scope

- Reworking RCCL coordinator semantics (except parity with NCCL policy where practical).
- Redesigning GLOBAL TP MPI collectives.
- Broad graph-capture redesign outside LocalTP collective boundaries.

## Key risks and constraints

1. **Environment correctness risk**
   - NCCL requires real CUDA driver + runtime compatibility (not stub libraries).
2. **Thread/stream ordering risk**
   - Device worker streams and coordinator streams/events must maintain strict ordering.
3. **Graph-capture interaction risk**
   - Decode graph capture modes may interleave with collectives; stage boundaries and stream/event synchronization must be capture-safe.
4. **Silent fallback risk**
   - Automatic fallback can conceal infrastructure failures and skew performance data.

## Work plan

## Phase 0 — Baseline and instrumentation hardening

### Deliverables
- Add/confirm explicit counters and structured logs for:
  - NCCL collective attempts/successes/failures
  - fallback attempts/successes (if fallback remains enabled)
- Emit one-line startup capability summary for NCCL runtime readiness.

### Tasks
- Add `LocalTPContext` per-stage collective telemetry fields (attempts, fallback count).
- Extend `NCCLBackend` / `NCCLCoordinator` error reporting with categorized failure codes where possible.
- Add log marker string suitable for CI grep (e.g., `LOCALTP_NCCL_PATH=REAL` vs `LOCALTP_NCCL_PATH=FALLBACK`).

### Acceptance
- Test logs can reliably distinguish real NCCL from fallback path without manual inference.

## Phase 1 — Runtime readiness gate (fail-fast for required NCCL mode)

### Deliverables
- Introduce explicit runtime mode control:
  - `LLAMINAR_LOCALTP_NCCL_MODE=required|allow_fallback|disabled`
  - default for CI parity: `required`

### Tasks
- Implement a readiness probe during LocalTP/NCCL initialization:
  - NCCL dynamic loader status
  - CUDA driver/runtime sanity checks
  - communicator initialization health
- In `required` mode:
  - disable host fallback
  - fail immediately with actionable diagnostics on NCCL runtime errors.
- In `allow_fallback` mode:
  - preserve existing fallback for developer convenience.

### Acceptance
- In broken NCCL environments, tests fail early with clear message when `required`.
- In healthy environments, no fallback is used.

## Phase 2 — Collective correctness guarantees (real NCCL path)

### Deliverables
- Harden LocalTP barrier + multi-buffer collective correctness for CUDA NCCL path.

### Tasks
- Audit `LocalTPContext::allreduceWithBarrierMultiGpu(...)` for:
  - strict buffer/device mapping invariants
  - count/shape consistency checks
  - generation/barrier timeout handling and cleanup behavior
- Verify `NCCLCoordinator` sequencing:
  - device worker event waits before collective
  - completion events visible to worker streams after collective
- Add focused integration checks for:
  - first-iteration warmup
  - repeated decode iterations
  - mixed stage ordering under LocalTP.

### Acceptance
- `V2_Integration_Parity_Qwen2_LocalTP` passes in NCCL-required mode with no fallback markers.
- `V2_Integration_TPAllreduceStage_LocalNCCL` remains green.

## Phase 3 — Graph-capture compatibility for LocalTP collectives

### Deliverables
- Explicit policy and validated behavior for LocalTP + CUDA graph capture modes.

### Tasks
- Define support matrix across:
  - `LLAMINAR_GPU_GRAPHS`
  - `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED`
  - stream-only modes
- Ensure collective boundaries are either:
  - outside captured segments, or
  - capture-safe with deterministic stream/event contract.
- Add targeted integration tests that combine LocalTP NCCL with enabled graph-capture flags.

### Acceptance
- No collective regressions when graph-capture modes are enabled in supported configuration.
- Unsupported combinations fail with explicit, documented errors.

## Phase 4 — CI policy and fallback tightening

### Deliverables
- CI enforces real NCCL path for CUDA LocalTP parity.

### Tasks
- Add CI job config for CUDA LocalTP parity with `LLAMINAR_LOCALTP_NCCL_MODE=required`.
- Gate merge on:
  - LocalTP parity pass
  - absence of fallback markers in logs.
- Optionally keep fallback mode only for non-CI/dev workflows.

### Acceptance
- CI failures clearly differentiate infra breakage vs model parity mismatch.

### Phase 4 progress (2026-02-17)

- Added required-mode qualification script: `scripts/validate_localtp_nccl_required.sh`
  - Runs `V2_Integration_Parity_Qwen2_LocalTP` with `LLAMINAR_LOCALTP_NCCL_MODE=required`
  - Hard-fails on disallowed markers: `LOCALTP_NCCL_PATH=FALLBACK`, `DISABLED`, `REQUIRED_FAIL`
  - Requires positive proof marker: `LOCALTP_NCCL_PATH=REAL`
- Added VS Code task: `V2: validate LocalTP NCCL required`
  - Entry point for local/CI parity qualification runs on NCCL-capable hosts

### Phase 4 progress (2026-02-18)

- Enhanced `scripts/validate_localtp_nccl_required.sh` with CUDA loader preflight diagnostics:
  - Emits `ldconfig` libcuda mappings
  - Emits discovered CUDA toolkit stub `libcuda.so*` locations
  - Logs original and sanitized `LD_LIBRARY_PATH`
- Added runtime path sanitization for the gate run:
  - Removes CUDA toolkit runtime paths (including stubs) from the spawned test process `LD_LIBRARY_PATH`
  - Keeps qualification runs aligned with driver-managed loader resolution (`ldconfig` path)
- Added explicit remediation output when NCCL reports CUDA stub runtime usage:
  - Detects `Cuda failure 'CUDA driver is a stub library'`
  - Emits actionable host/runtime checklist for cleanup and rerun

### Current blocker status (2026-02-18) — **RESOLVED**

- Required-mode LocalTP parity previously failed in this dev container with NCCL runtime warning:
  - `NCCL WARN Cuda failure 'CUDA driver is a stub library'`
- **Root cause**: NCCL library version incompatibility. The installed `libnccl2 2.18.5-1-2` was built against **CUDA 12.0** (`NCCL version 2.18.3+cuda12.0`), but the runtime environment runs **CUDA 13.0** (driver 580.126.09). NCCL's internal `strongstream.cc:60` stub detection fails when the CUDA driver API version (13000) is beyond what NCCL 2.18 was compiled against.
- **Fix**: Upgraded NCCL to `2.28.9-1+cuda13.0` from the NVIDIA CUDA apt repository.
  - `sudo apt-get install libnccl2=2.28.9-1+cuda13.0 libnccl-dev=2.28.9-1+cuda13.0`
- **Also fixed**: `ncclRedOp_t` enum values in `NCCLDynamicLoader.h` had `ncclMin` and `ncclMax` swapped relative to the NCCL ABI (`ncclMax=2, ncclMin=3`). This was a latent bug that would cause incorrect results for Min/Max reduction operations (Sum/Prod/Avg were unaffected).
- **Verification**: All NCCL integration tests pass:
  - Standalone `ncclAllReduce` test: PASSED
  - `V2_Integration_LocalTP_BackendBehavior`: 16/17 PASSED, 1 SKIPPED (expected)
  - `V2_Integration_NCCLBackend`: PASSED
  - `V2_Integration_TPAllreduceStage_LocalNCCL`: PASSED
  - `V2_Integration_NCCLCoordinator`: PASSED
  - `V2_Integration_LocalTP_MultiDevice`: PASSED

## Validation matrix

## Core tests

- `ctest --test-dir build_v2_integration --output-on-failure --parallel -R '^V2_Integration_Parity_Qwen2_LocalTP$'`
- `ctest --test-dir build_v2_integration --output-on-failure --parallel -R '^V2_Integration_Parity_Qwen2_SingleDevice$'`
- `ctest --test-dir build_v2_integration --output-on-failure --parallel -R '^V2_Integration_TPAllreduceStage_LocalNCCL$'`

## Mode-specific checks

1. `required` mode:
   - expect hard failure on NCCL runtime setup errors
   - expect zero fallback markers
2. `allow_fallback` mode:
   - expect correctness continuity if NCCL runtime breaks
   - fallback markers present when triggered

## Runtime diagnostics to capture in test artifacts

- NCCL loader status
- communicator init summary (device ordinals, rank/world)
- per-collective success/fail counts
- fallback count

## Implementation touchpoints (expected)

- `src/v2/collective/LocalTPContext.cpp`
- `src/v2/collective/LocalTPContext.h`
- `src/v2/collective/backends/NCCLBackend.cpp`
- `src/v2/collective/coordinators/NCCLCoordinator.h`
- `src/v2/collective/coordinators/NCCLCoordinator.cu`
- `src/v2/utils/DebugEnv.h`
- `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_Parity.cpp`
- `tests/v2/integration/execution/compute_stages/stages/Test__TPAllreduceStage_LocalNCCL.cpp`

## Rollout strategy

1. Land instrumentation + mode flag first.
2. Turn on `required` mode in local validation on known-good NCCL hosts.
3. Enable CI gating for real NCCL path.
4. Keep fallback only behind explicit opt-in for developer/unqualified environments.

## Exit criteria

Project is complete when all are true:

- CUDA LocalTP parity passes with NCCL in `required` mode.
- Logs/artifacts confirm zero fallback usage in qualifying runs.
- Graph-capture support policy is documented and validated by tests.
- Failures in broken NCCL environments are immediate, actionable, and non-silent.

## Notes

This plan intentionally separates **correctness continuity fallback** from **qualification path**. For production/perf confidence, only runs that prove `LOCALTP_NCCL_PATH=REAL` should be considered valid NCCL LocalTP results.
