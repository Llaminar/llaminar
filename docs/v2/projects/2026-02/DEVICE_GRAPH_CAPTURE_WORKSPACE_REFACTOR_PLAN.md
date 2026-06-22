# Device Graph Capture + Workspace Refactor Plan (V2)

## Scope
This plan consolidates the architectural review findings for the current device-graph orchestration path, with focus on:
- Segmented GPU graph capture/replay lifecycle
- Device workspace allocation/binding lifecycle
- Buffer manager vs orchestrator ownership boundaries
- Coherence semantics during replay/manual segments

## Execution Status
- Phase 0: **Substantially complete**
  - ✅ Segmented replay baseline test added (`V2_Integration_SegmentedGraphCaptureExecution`)
  - ✅ Workspace owner invariant guard added in orchestrator
  - ✅ Workspace-consumer binding/rebind baseline tests added (`V2_Unit_GraphBufferManager`)
  - ✅ Baseline perf/test snapshot recorded in `changelog/2026-02-16-device-graph-phase0-baseline.md`
  - ⚠️ Collective-marked segmented synthetic-case stabilization remains tracked via disabled scaffold test
- Phase 1: **In progress (started)**
  - ✅ Added model-aware workspace API in `DeviceGraphBufferManager` (`allocateDeviceWorkspaceForGraph`)
  - ✅ Routed orchestrator workspace allocation through `DeviceGraphBufferManager`
  - ✅ Removed orchestrator-owned workspace state (`device_workspaces_`, `device_workspace_allocated_`)
  - ✅ Verified focused parity suite (`V2_Unit_GraphBufferManager`, `V2_Unit_DeviceGraphOrchestratorBufferManagement`, `V2_Unit_DeviceGraphOrchestrator`, `V2_Unit_GraphResolver`, `V2_Unit_Qwen2Graph_KVCachePP`, `V2_Integration_SegmentedGraphCaptureExecution`)
  - ✅ Focused benchmark snapshot captured (prefill `7149.23 tok/s`, decode `46.34 tok/s`)
  - ✅ Phase 1.5 candidate suite established and validated (`23/23` passing)
  - 🔄 Remaining: broader regression gate deferred while legacy unrelated failures are being cleaned up
- Phase 3: **Substantially complete (implementation extraction complete; closeout validation/documentation remains)**
  - ✅ Segmented warmup/capture/replay orchestration extracted into `DeviceGraphCaptureController`
  - ✅ Executor reduced to thin coordination + fallback/policy handling for segmented path
  - 🔄 Remaining: Phase 3 closeout validation pass and doc/status cleanup

## Current-State Findings (Critical)

### 1) Split ownership of GPU workspace lifecycle
- `DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(...)` allocates and binds per-device workspace directly.
- `DeviceGraphBufferManager::allocateDeviceWorkspace(...)` independently allocates and binds workspace.
- Result: dual ownership, duplicated policy, drift risk.

### 2) Inconsistent sizing and requirement gathering
- Orchestrator path uses model-aware LM-head-heavy budget and per-consumer hint logic.
- BufferManager path uses generic defaults (`max_m=4096`) for `getWorkspaceRequirements(...)`.
- Result: potential under/over-allocation depending on code path.

### 3) Capture policy is split across layers
- Orchestrator decides segmented eligibility, backend conditions, stream application, and fallback policy.
- Executor owns segmented 3-phase state machine and replay semantics.
- Result: difficult reasoning/debugging and brittle policy coupling.

### 4) Replay semantics rely on executor compensation
- Captured segments bypass normal `executeNode(...)`; executor manually performs output-dirty marking and replay callbacks.
- Result: correctness depends on special-case replay logic instead of uniform stage contract flow.

### 5) Debug/experiment branches mixed into production replay body
- Verify/recapture/stream-only and ad-hoc logging are in the primary replay function.
- Result: larger blast radius, harder maintainability.

---

## Target Architecture (End State)

### A) Single owner for workspace lifecycle
- **Owner:** `DeviceGraphBufferManager` (or a dedicated `WorkspaceCoordinator` owned by it).
- Orchestrator does not allocate/bind workspace directly.
- One API for gather requirements, budget, allocate, bind, release.

### B) Single owner for segmented capture lifecycle
- **Owner:** `DeviceGraphCaptureController` (new component, called by `DeviceGraphExecutor`).
- Controller handles warmup/capture/replay state machine and mode policy.
- Orchestrator only passes high-level execution intent.

### C) Unified capture policy object
- Construct `CapturePolicy` once per graph/session (backend, collectives mode, max stages, diagnostics mode).
- Replace scattered env checks in orchestrator and replay loop.

### D) Uniform replay hooks contract
- Codify a single hook path for replay lifecycle:
  - pre-replay input coherence
  - post-replay output dirty marking
  - post-replay stage callback dispatch
- Keep this declarative and stage-contract-driven.

### E) Diagnostics isolation
- Move verify/recapture/stream-only into explicit strategy modules (`ReplayStrategy::Normal`, `ReplayStrategy::Verify`, etc.).

---

## Phased Implementation Plan

## Phase 0 — Baseline & Safety Net
**Goal:** lock in current behavior before refactor.

### Tasks
1. Add focused integration tests for segmented replay + collectives + workspace consumers.
2. Add invariant checks for workspace owner uniqueness (assert only one active owner).
3. Capture baseline perf and stability metrics (decode tok/s, replay failure rate).

### Deliverables
- New tests in `tests/v2/integration/` covering:
  - segmented replay with/without collectives
  - workspace consumer binding on graph rebuild
- Baseline metrics note in `changelog/`.

### Exit Criteria
- Tests green in `build_v2_integration`.
- Baseline report checked in.

---

## Phase 1 — Workspace Ownership Unification
**Goal:** remove orchestrator-owned workspace allocation path.

### Tasks
1. Introduce/expand BufferManager API to accept model-aware sizing hints.
2. Route all workspace allocation/binding through BufferManager.
3. Remove `device_workspaces_`, `device_workspace_allocated_`, and manual consumer scan from orchestrator.
4. Keep behavior parity for KV cache and attention/embedding/GEMM consumers.

### Deliverables
- `DeviceGraphOrchestrator` no longer allocates workspace directly.
- `DeviceGraphBufferManager` owns workspace lifecycle end-to-end.

### Exit Criteria
- Functional parity on unit/integration tests.
- No regression in decode throughput > 2% (same benchmark harness).

---

## Phase 2 — Capture Policy Consolidation
**Goal:** centralize capture eligibility and mode selection.

### Status (Current)
- ✅ `DeviceGraphExecutor::DecodeCapturePolicy` added and wired through `executeDecodeWithCapturePolicy(...)`.
- ✅ `DeviceGraphOrchestrator` now builds policy once (`buildDecodeCapturePolicy(...)`) and executes decode via the policy entrypoint.
- ✅ Duplicated decode-path branch logic for segmented replay vs fast decode removed from orchestrator call-site.
- ✅ Focused regression gate passed: `bash scripts/run_phase15_stable_suite.sh` (13/13 passing).

### Tasks
1. Add `CapturePolicy` struct (collective mode, backend support, max stages, diagnostics mode).
2. Build policy once in orchestrator; pass into executor/controller.
3. Remove duplicate conditional logic from replay call-sites.

### Deliverables
- One policy-construction point.
- Replay entrypoint accepts policy object.

### Exit Criteria
- Existing behavior preserved for CUDA/ROCm and collective/non-collective graphs.

---

## Phase 3 — Extract DeviceGraphCaptureController
**Goal:** isolate segmented warmup/capture/replay state machine.

### Status (Current)
- ✅ Introduced `DeviceGraphCaptureController` in `execution/local_execution/graph/` to own segmented phase transitions (warmup/capture/replay) and decode-step progression.
- ✅ `DeviceGraphExecutor::executeWithSegmentedGraphCapture(...)` now delegates phase selection to `DeviceGraphCaptureController` while preserving existing phase bodies.
- ✅ Warmup segment partitioning (`buildWarmupSegments`) and capture-phase replay callback initialization (`initializeReplayCallbacks`) now delegated to `DeviceGraphCaptureController`.
- ✅ Replay helpers delegated: stream-only replay execution (`executeStreamOnlyReplay`) and non-idempotent stage detection (`segmentHasNonIdempotentStage`).
- ✅ Manual replay segment loop delegated to `DeviceGraphCaptureController::executeManualReplaySegment(...)` with executor callback injection for `executeNode(...)` behavior parity.
- ✅ Normal capturable replay launch path delegated to `DeviceGraphCaptureController::executeCapturedReplaySegmentNormal(...)`; executor retains failure policy/fallback decisions.
- ✅ Re-capture capturable replay branch delegated to `DeviceGraphCaptureController::executeCapturedReplaySegmentRecapture(...)`.
- ✅ Verify capturable replay branch delegated to `DeviceGraphCaptureController::executeCapturedReplaySegmentVerify(...)` (including non-idempotent skip handling).
- ✅ Capture-phase capturable segment finalization delegated to `DeviceGraphCaptureController::finalizeCapturePhaseCapturableSegment(...)` (instantiate/launch vs collective phase-2 execution paths).
- ✅ Capture-phase manual (non-capturable) segment execution delegated to `DeviceGraphCaptureController::executeCapturePhaseManualSegment(...)`.
- ✅ Replay-loop capturable mode dispatch (`normal`/`verify`/`re-capture`) delegated to `DeviceGraphCaptureController::executeReplayCapturableSegment(...)` while keeping executor fallback policy for normal launch failures.
- ✅ Replay-loop per-segment dispatch (capturable + manual) delegated to `DeviceGraphCaptureController::executeReplaySegment(...)`, further reducing executor branch density.
- ✅ Replay input coherence and post-launch lifecycle helpers moved behind `DeviceGraphCaptureController::cohereReplaySegmentInputs(...)` and `DeviceGraphCaptureController::postCapturedSegmentLaunch(...)` with stage-level callbacks from executor.
- ✅ Phase-2 capture scaffolding (capture stream setup, callback init, and segment capture/manual iteration) delegated to `DeviceGraphCaptureController::executeCapturePhase(...)` with structured fallback/reset result handling.
- ✅ Replay-phase envelope (stream-only mode, per-segment replay loop, and final sync) delegated to `DeviceGraphCaptureController::executeReplayPhase(...)`, leaving executor with fallback policy handling only.
- ✅ Warmup and segmented device-prep glue moved to controller helpers (`prepareDeviceForSegmentedCapture(...)`, `executeWarmupPhase(...)`) to further slim executor phase branching.
- ✅ Top-level controller callback surface consolidated with typed `ReplayHooks` at `executeCapturePhase(...)` and `executeReplayPhase(...)`, reducing repeated lambda parameter lists at executor call-sites.
- ✅ Confidence gate passed: `bash scripts/run_phase15_candidate_suite.sh` (23/23 passing).
- ✅ Focused regression gate passed after extraction: `ctest --test-dir build_v2_integration --output-on-failure -R "^V2_Unit_GraphBufferManager$|^V2_Unit_DeviceGraphOrchestrator$|^V2_Unit_DeviceGraphOrchestratorBufferManagement$|^V2_Integration_SegmentedGraphCaptureExecution$"` (5/5 passing).
- ✅ Focused regression gate revalidated for latest slice: `ctest --test-dir /workspaces/llaminar/build_v2_integration --output-on-failure --parallel -R "V2_Unit_GraphBufferManager|V2_Unit_DeviceGraphOrchestrator|V2_Unit_DeviceGraphOrchestratorBufferManagement|V2_Integration_SegmentedGraphCaptureExecution"` (11/11 passing).

### Tasks
1. Introduce `DeviceGraphCaptureController` under `execution/local_execution/graph/`.
2. Move segmented phase logic from `DeviceGraphExecutor::executeWithSegmentedGraphCapture(...)`.
3. Keep `DeviceGraphExecutor` focused on node execution/coherence primitives.

### Deliverables
- Controller with clear API:
  - `initialize(...)`
  - `warmup(...)`
  - `capture(...)`
  - `replay(...)`
- Executor delegates segmented flow to controller.

### Exit Criteria
- All segmented replay tests pass.
- No new capture failure modes.

---

## Phase 4 — Replay Semantics Hardening
**Goal:** make replay semantics explicit and uniform.

### Tasks
1. Formalize replay lifecycle helpers (cohere inputs, mark outputs dirty, callback dispatch).
2. Ensure same semantics for captured and manual segments where contract requires.
3. Add assertions around stage coherence policy compliance in replay path.

### Deliverables
- Unified replay lifecycle module.
- Reduced ad-hoc logic inside segmented loop.

### Exit Criteria
- No coherence regressions in parity/integration tests.

---

## Phase 5 — Diagnostics Strategy Isolation
**Goal:** separate debug replay modes from production replay path.

### Tasks
1. Extract verify/recapture/stream-only branches into strategy classes.
2. Keep production path slim (`ReplayStrategy::Normal`).
3. Gate diagnostics via explicit config object rather than scattered env checks.

### Deliverables
- Strategy-based replay diagnostics.
- Cleaner production replay function.

### Exit Criteria
- Debug modes still available and validated.
- Production path complexity reduced (measurable by function size/branch count).

---

## Subagent Delegation Plan

## Delegation Model
Use one subagent per phase with strict, testable output contracts. Keep prompts scoped to one phase only.

## Packet A (Phase 0)
**Mission:** Add baseline tests + metrics harness for segmented replay/workspace ownership invariants.

**Must Return:**
- List of tests added/updated
- Exact files changed
- Command outputs for targeted ctest runs
- Baseline metric snapshot

## Packet B (Phase 1)
**Mission:** Unify workspace ownership under BufferManager and remove orchestrator allocation path.

**Must Return:**
- API changes and call-site updates
- Removed fields/functions from orchestrator
- Test and benchmark diffs (before/after)

## Packet C (Phase 2)
**Mission:** Introduce and wire `CapturePolicy` end-to-end.

**Must Return:**
- Policy definition location
- Former duplicated checks removed
- Behavior parity evidence (test results)

## Packet D (Phase 3)
**Mission:** Extract segmented state machine to `DeviceGraphCaptureController` with minimal behavior drift.

**Must Return:**
- New controller API and ownership model
- Executor simplification diff summary
- Replay/capture regression results

## Packet E (Phases 4–5)
**Mission:** Harden replay semantics and isolate diagnostics strategies.

**Must Return:**
- Replay lifecycle helper modules
- Strategy classes for diagnostics modes
- Verification that production path still matches baseline behavior

---

## Validation Matrix (Per Phase)
Run the smallest targeted checks first, then broaden:
1. Targeted unit/integration tests for touched components
  - Phase 1.5 stable suite: `bash scripts/run_phase15_stable_suite.sh`
  - Phase 1.5 stable suite + benchmark: `bash scripts/run_phase15_stable_suite_and_benchmark.sh`
  - Phase 1.5 candidate suite: `bash scripts/run_phase15_candidate_suite.sh`
  - Phase 1.5 candidate suite + benchmark: `bash scripts/run_phase15_candidate_suite_and_benchmark.sh`
2. `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel`
3. Decode benchmark sanity (`--benchmark`) for throughput drift
4. Optional parity pass for risky phases

---

## Risks & Mitigations
- **Risk:** hidden coupling with stage-specific workspace hints
  - **Mitigation:** preserve existing hints and add explicit contract tests.
- **Risk:** capture replay drift during controller extraction
  - **Mitigation:** phase-gated extraction + verify-mode comparison tests.
- **Risk:** performance regressions from additional abstractions
  - **Mitigation:** benchmark gate at each phase with fixed prompts/settings.

---

## Immediate Next Step
Close out **Phase 3** and begin **Phase 4**:
1. Run deferred broader regression gate (`V2_Integration_*`) and benchmark sanity to close remaining validation debt.
2. Update this plan status once broader gate results are captured.
3. Start Phase 4 by extracting a dedicated replay lifecycle helper module (coherence-in, dirty-mark/callback-out contract).
4. Add explicit assertions for stage coherence policy compliance in replay paths.

---

## Remaining Work Summary

### Phase 3 Closeout (Remaining)
- Execute broader regression validation beyond focused gates:
  - `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel`
- Run benchmark sanity and record drift against baseline.
- Refresh plan status sections to reflect final Phase 3 completion criteria.

### Phase 4 (Not started)
- Formalize replay lifecycle contracts into a single reusable helper:
  - pre-replay input/weight/output coherence
  - post-replay output dirty marking
  - post-replay callback dispatch
- Add replay-path assertions for stage coherence policy compliance.
- Validate with stable/candidate suites + integration gate.

### Phase 5 (Not started)
- Isolate diagnostics into strategy-style paths (`Normal`, `Verify`, `Recapture`, `StreamOnly`).
- Remove diagnostics branching from production replay body.
- Validate debug-mode parity and measure reduced production-path complexity.