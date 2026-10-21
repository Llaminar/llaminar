# DeviceGraphOrchestrator incremental source split

Date: 2026-10-19. Status: proposed; implementation has not started under this plan.

## Purpose and scope

Make DeviceGraphOrchestrator easier to navigate, review and rebuild through
small, behavior-preserving source extractions. The first delivery keeps the
existing class, interfaces, state ownership and execution architecture. Complete
method definitions move into responsibility-specific translation units; they
remain `DeviceGraphOrchestrator::...` methods with ordinary private-member access.

This is a separate, bounded project from the earlier
[DGO breakup and refactor plan](DGO_BREAKUP_REFACTOR_PLAN.md). That project moves
behavior and state into independent owners and specifies a small final facade.
This plan does not satisfy that facade's line budget or claim to remove the
class's internal coupling. It provides a useful source organization without
requiring the larger ownership migration. The earlier document remains intact.

The task creating this document authorizes documentation only. Implementation
and its scheduling belong to the main development conversation; this plan must
not interrupt ongoing QFN development or its active build/test transactions.

## Observed starting point

The source inspection on 2026-10-19 found:

| File | Physical lines |
|---|---:|
| `DeviceGraphOrchestrator.cpp` | 64,629 |
| `DeviceGraphOrchestrator.h` | 12,055 |
| `DeviceGraphOrchestratorGeneration.cpp` | 753 |
| `DeviceGraphOrchestratorMTPPublication.cpp` | 576 |
| `DeviceGraphOrchestratorMovement.cpp` | 144 |
| Total | 78,157 |

These are a dated observation, not a maintained size contract. Re-measure when
implementation begins because the active feature branch continues to change.
The existing companion `.cpp` files establish the intended mechanical pattern.

Important references are the
[DGO declaration](../../../../src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h),
[current implementation](../../../../src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp),
[V2 ownership architecture](../../../../.github/instructions/llaminar-v2-architecture.instructions.md),
[production source inventory](../../../../src/v2/CMakeLists.txt), and
[testing workflow](../../../../.agents/llaminar-testing/SKILL.md).

## Target source layout

All files below remain under `src/v2/execution/local_execution/orchestrators/`.
The suffixes are proposed destinations, not new public classes. Keep methods
with their actual dependencies; inspect the call graph before finalizing each
boundary. Split further only when a coherent responsibility warrants it.

| File or family | Responsibility |
|---|---|
| `DeviceGraphOrchestrator.cpp` | Construction, dependency wiring, facade-level entrypoints where useful, and overall initialization/teardown ordering. |
| `DeviceGraphOrchestratorSetup.cpp` | Arena/workspace binding, prepared-weight setup, KV initialization and setup validation. Existing authorities continue to own allocation and preparation. |
| `DeviceGraphOrchestratorForward.cpp` | Forward entrypoints, prefill chunking, invocation preparation and integration with `ForwardExecutionEngine`. |
| `DeviceGraphOrchestratorRequestLifecycle.cpp` | Request admission, producer-event joins, reset, terminal storage retirement, reset-ready publication and session epochs. |
| `DeviceGraphOrchestratorGeneration.cpp` | Composition and materialization of complete ordinary generation programs and their retained graph identities. Extend the existing file. |
| `DeviceGraphOrchestratorGenerationReplay.cpp` | Launch, authenticated hosted tickets, transaction submission and terminal observation for the admitted execution policy. |
| `DeviceGraphOrchestratorMTPSidecar.cpp` | Draft-model invocation and retained sidecar graph binding. |
| `DeviceGraphOrchestratorMTPVerifier.cpp` | Grouped verifier preparation, verifier graph composition, outcome handling and checkpoint/rollback coordination. |
| `DeviceGraphOrchestratorMTPPublication.cpp` | Existing speculative-state publication implementation; preserve this established boundary. |
| `DeviceGraphOrchestratorSampling.cpp` | Sampling, penalties, distributions and device token publication. |
| `DeviceGraphOrchestratorPrefixCache.cpp` | Prefix lookup, harvest, restore and cache lifecycle coordination. |
| `DeviceGraphOrchestratorPrefixCheckpoint.cpp` | Live checkpoint storage, snapshot binding and checkpoint-state publication. |
| `DeviceGraphOrchestratorMoERuntime.cpp` | ExpertOverlay leases, maintenance graph coordination and initial/runtime publication. |
| `DeviceGraphOrchestratorMovement.cpp` | Existing movement observation and archive publication, plus directly related movement methods. |
| `DeviceGraphOrchestratorDiagnostics.cpp` | Snapshot observation, completed-maintenance reporting, failure diagnostics and reporting helpers. |

Generation owns loop construction and replay. Sampling supplies its numerical
operations. MTP owns draft/verifier transactions and accepted-state publication.
Request lifecycle owns the boundaries between requests, including the ordering
required before those subsystems can reset. Prefix code uses that boundary
instead of introducing its own reset sequence.

Preserve CPU/CUDA/ROCm symmetry within each responsibility. Native APIs remain
behind the current backend and kernel interfaces. Shared generation policy
continues to distinguish a native conditional parent, authenticated retained
transaction submission, and declared heterogeneous boundaries. File placement
must not create separate CUDA and ROCm copies of the request lifecycle.

## Invariants of the mechanical extraction

1. Preserve class inheritance, public signatures, member declaration order,
   construction and destruction order, and ownership of every retained object.
   A moved method must operate on the same state it used before extraction.
2. Preserve graph topology, dependency/event edges, capture identity, launch
   policy, graph reuse, request-reset ordering and failure behavior. A source
   split cannot justify recapture, eager replay or a different backend policy.
3. Keep `PhysicalMemoryAuthority`, `BufferArena`, prepared-weight owners and
   `TransferEngine` authoritative. Moving setup code introduces no accounting
   ledger, reserve, allocator or transfer implementation.
4. Preserve device-owned execution state and exact live communication extents.
   Add no host count readbacks, hot-path allocation, callbacks or synchronization.
5. Preserve arithmetic and reduction order, compiler definitions, ISA selection,
   supported topology/precision settings and numerical acceptance thresholds.
6. Retain one definition of each helper. Move a private helper with its callers;
   use a narrow internal declaration and one implementation when multiple files
   genuinely share it. Audit anonymous-namespace state and function-local statics
   so extraction does not duplicate caches, registrations or lifecycle state.
7. Compile each `.cpp` as an ordinary translation unit. Included implementation
   files, macro-generated method bodies and a new inheritance hierarchy are
   outside this mechanical project.
8. Every touched source file and non-trivial method retains the required Doxygen
   and lifecycle explanations. Reviewability comes from responsibility boundaries,
   not compressed formatting or removed documentation.

An extraction can expose an existing defect. Record and fix it as a separately
reviewable change with a focused regression; keep the relocation diff separable
from changes to behavior. Preserve unrelated work and existing failure evidence.

## Header work and build expectations

Move substantial inline bodies out of `DeviceGraphOrchestrator.h` after their
responsibility files exist. Initial candidates are request reset and graph-cache
lifecycle methods. Retain templates and genuinely compile-time definitions where
the language requires visibility. Keep the public API and object layout stable
during this phase, and rebuild all affected consumers coherently.

Moving nested types into another header that DGO still includes improves
organization but does not reduce dependency fan-out. Do that only for a clear
source boundary. Reducing exposure of private state through opaque owners is a
later ownership change, not a prerequisite for this project.

Smaller translation units should reduce the cost of an implementation-only edit
and expose more parallel compilation. Repeated parsing and lost cross-function
inlining can also increase build or runtime costs. Measure those effects instead
of promising a speedup from file count alone. Keep Ninja, ccache, unrestricted
`cmake --build ... --parallel`, the configured compiler/SDK and spill guards.

## Delivery sequence

### P0: establish the extraction baseline

- Choose a quiescent main-thread checkpoint with known source/build identities.
  Preserve active feature edits; do not move methods while another worker edits
  the same region or while a qualification run depends on frozen source.
- Inventory methods, inline bodies, private helpers, static state and actual
  callers. Assign one destination to each moved symbol.
- Record the current test inventory and results. Carry existing failures as
  explicit baseline failures rather than attributing them to the split.
- Record representative build timings and the dependency graph described below.
- Identify source-policy tests and allowlists tied to the original filename.

### P1: extract diagnostics and prefix responsibilities

Move diagnostics, prefix-cache operations and live checkpoint methods in
separate reviewable slices. Keep teardown sequencing visible at the owning
entrypoint. Compile and run the relevant source-policy, prefix/reset and
diagnostic tests after each slice. Preserve failure cleanup as well as success.

### P2: extract sampling and forward execution

Move sampling/distribution/penalty methods, followed by forward execution and
prefill scheduling. Keep `ForwardExecutionEngine` and existing graph builders
responsible for their current work. Verify CPU and both GPU paths before
continuing into generation construction.

### P3: separate request lifecycle, generation and MTP

Move request-boundary methods and their substantial inline implementations.
Separate graph composition from replay, then sidecar from verifier behavior,
retaining the existing MTP publication file. Preserve all shared state and
retirement order. This phase requires the strongest reset, terminal, prefix,
fixed/dynamic MTP and heterogeneous execution coverage.

### P4: finish setup and MoE separation

Move setup and MoE runtime methods after earlier boundaries are visible. Audit
resource lifetimes, ExpertOverlay leases, movement publication and partial
initialization failure. Keep the core constructor/destructor sequence readable.
Remove obsolete definitions and includes as each move lands.

### P5: qualify and document the completed source split

Run complete model-free prerequisites on the final coherent tree and the
focused production generation checks selected for affected execution paths.
Measure rebuild and runtime effects. Update the architecture source map with
the actual destinations and record remaining coupling separately.

Each phase may contain several small commits. A phase boundary is a review and
validation boundary, not permission to claim a full model or image certificate.
Use the normal feature-branch workflow when implementation is scheduled.

## Functional and architecture validation

Use the existing
[CMake test inventory](../../../../tests/v2/CMakeLists.txt) and canonical testing
drivers. Select actual registered cases at implementation time; this plan does
not create a second topology matrix or invent new test names.

| Changed area | Required focused evidence |
|---|---|
| Every extraction | Coherent compile/link; original tests for moved methods; source-policy inventory and negative controls; no duplicate symbol or missing registration. |
| Setup | Device-free admission/workspace/lifetime tests and applicable native initialization, allocation-release and teardown regressions. |
| Forward and generation | CPU, CUDA and ROCm native execution; complete capture policy; repeated replay; exact graph identities and event ordering. |
| Request lifecycle | Prefill-only reset, completed-generation reset, repeated/empty reset, failure retirement, prefix replacement and next-request reuse; fresh terminal statistics published once. |
| Sampling and MTP | Greedy/stochastic policies, penalties, MTP disabled, focused fixed/dynamic depth, grouped-versus-serial byte equivalence, accepted/rejected outcomes and state publication. |
| Prefix cache | Fresh/full/partial hits, live GPU reuse, RAM/disk tiers, reset/restore equivalence, invalidation and shutdown retirement. |
| MoE and topology | Static/dynamic movement, overlay lease lifetime, native TP/PP, heterogeneous CPU/CUDA/ROCm boundaries and actual live transfer extents. |

Existing architecture tests often open `DeviceGraphOrchestrator.cpp` directly.
Update their file selection as part of each extraction. Derive the scan scope
from the canonical source inventory, authenticate that moved definitions are
present, and preserve narrow function-level exceptions. An empty scan or absent
method must fail. Retain negative controls proving forbidden behavior is still
detected in the new files; do not add whole-file exclusions to accommodate moves.

Unit remains fast and device-free. Native GPU work stays in its explicit
integration suites with the existing resource locks and driver diagnostics.
Every newly discovered production defect receives a focused regression and an
explicit `ProductionTestPreflight` registration.

Iterate with focused gates within a coherent batch, keeping the existing suite
green. Refresh complete prerequisites at the completed batch boundary and before
model qualification, commit/push or claiming the tree is green:

```sh
python3 scripts/ci/run_production_prerequisites.py \
  --build-dir build_v2_integration \
  --output parity-results/dgo-source-split/prerequisites-<unique-batch>
```

The output path above is a template; use a fresh real batch name. The canonical
driver builds and runs complete Unit and production preflight. An old receipt
cannot qualify changed source, binaries or test inventory. Reuse a valid receipt
for unchanged diagnostic cells instead of rerunning the full gate for each one.

For final production behavior, select existing generation cells covering the
changed CPU/CUDA/ROCm and composition paths, dense/MoE, prefix reuse and ordinary/
dynamic MTP behavior. Fixed-depth coverage remains a focused obligation for
changed MTP code. Required unavailable hardware is recorded as unqualified or
skipped according to the canonical contract, never counted as a pass.

Numerical drift triggers detailed diagnosis with the canonical PyTorch parity
framework and the exact same production GGUF, prompt and configuration. Preserve
all stage CSVs and find the first divergence. Full mathematical campaigns are
not a routine requirement for each mechanical move; tolerances and reference
outputs remain unchanged. No additional model or cache payload hashing is needed.

## Build and runtime measurement

Record before/after measurements using the same compiler, SDK, build type,
hardware, cache policy and concurrency. Use existing build logs and compile
commands to identify the longest compile and link steps.

- Measure an implementation-only change in an extracted file, a change to a
  remaining common implementation file, and a genuine header change. Report
  affected translation units, compile time, link time, total wall time and peak
  memory. Use isolated controlled edits and restore them exactly.
- Separate warm-ccache and cold-compile results. A no-work build is a sanity
  check, not a rebuild benchmark. Never clear shared caches or disturb another
  worker's active build to collect measurements.
- Check that ordinary implementation edits rebuild their own translation unit
  and expected linked consumers. Explain additional rebuilds through actual
  dependencies rather than assuming the split removed header coupling.
- Use repeated unprofiled Release generation measurements for affected paths,
  with matched prompts, context, precision, topology and MTP settings. Report
  prefill/decode throughput and variability; compare relevant existing graph,
  transfer, cache and allocation counters.
- Investigate a repeatable runtime regression, especially from changed inlining
  or compiler visibility. Use the repository's existing acceptance policies and
  observed noise; do not invent a percentage allowance to bless a regression.

This project adds no kernels. New kernel behavior or tuning belongs in a
separate change with its normal backend profiling and precision obligations.
Local measurements do not certify a release image or bypass CI requirements.

## Acceptance and possible later ownership work

The source split is complete when:

- The agreed responsibilities have clear implementation homes, each moved
  method/helper has one definition, and the central initialization/retirement
  order is easy to follow.
- Substantial eligible inline bodies have moved out of the header; remaining
  dependency fan-out and large methods are recorded honestly.
- Ownership, device execution policy, arithmetic, graph identity, communication
  extents and supported configurations retain their existing contracts.
- Source sanitizers cover every new translation unit with effective negative
  controls. Complete Unit/preflight and required focused execution gates pass.
- Build measurements establish the actual benefit and costs; affected Release
  execution shows no unexplained regression.
- CMake, architecture navigation and relevant tests name the installed layout.
  No generated results, private diagnostics or model payloads enter the commit.

After this bounded project, consider a separate extraction of a cohesive owner
such as request lifecycle, retained generation resources or prefix session state.
That work must move behavior together with its state and lifetime, expose narrow
typed operations, and retain a single authority. A helper accepting all DGO state
or a blanket back-reference to DGO does not establish an ownership boundary.
Independent owner extraction is optional follow-up and is not required to finish
the mechanical source split described here.
