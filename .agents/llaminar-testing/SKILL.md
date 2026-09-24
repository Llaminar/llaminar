---
name: llaminar-testing
description: Run, extend, maintain, and diagnose Llaminar's testing and certification workflow, including Unit and production-preflight gates, generation regressions, mathematical model parity, HTTP and remote-MPI E2E, and benchmark-certified Docker images. Use when selecting or running these gates, extending the canonical matrix, or investigating test failures. Kernel tuning and cross-engine performance comparisons have separate skills.
---

# Llaminar Testing

Use the canonical Llaminar testing workflow from fast model-free prerequisites
through real-weight diagnostics and Release server/image certification. Select
the gate appropriate to the task; a focused pass is not a full model or image
certificate. Reuse the existing drivers and typed matrix instead of maintaining
another model/topology list.

For mathematical model parity, prove the live inference path against an
independent reference pack generated from the declared real GGUF. Preserve every
stage comparison and canonical CSV while amortizing immutable model setup;
numerical checkpoints, not an extra whole-model hash pass, prove weight equivalence.

## Routine regression versus mathematical diagnosis

The routine production gate is **Unit → production preflight → HTTP token
generation regression → HTTP needle/long-context E2E → cross-host MPI E2E →
benchmarks → image certification**, independently for AVX512 and AVX2.
Generation projects only **MTP off (serial controls) and dynamic-depth MTP**
from the canonical model/topology definitions. Both compare against the saved
serial control outputs, requiring at least 384 continuous committed tokens.
Prompts, seeds, and each cell's explicit serial-control mapping are versioned
in the canonical definitions. Expected token payloads are versioned in
`Llaminar/corpora`, pinned by the source gitlink and ISA approval catalog.
Routine tests only read those baselines; acquisition/review/publication is a
separate explicit workflow. Never silently regenerate answers on drift.
Keep the fresh/full/partial prefix probes, exact repeatability, actual dynamic
MTP activity, movement obligations and captured production-path evidence.

The enabled `develop` GitHub workflow is intentionally narrower than that
routine certification path. It uses `scripts/ci/run_develop_image_gate.py` to
build both full-backend AVX512/AVX2 builder/runtime pairs, run the complete
Unit and ProductionTestPreflight transaction inside each builder, and publish
the two tested `develop` runtime tags **and exact source-SHA tags**. It does not run model discovery,
generation, mathematical parity, HTTP E2E, remote MPI, benchmarks, or attach a
certificate. Use it for fast shippable developer images; use the full pipeline
when requesting a certified artifact.
The `develop` → `master` PR gate is separate: its required pull-request E2E
check waits for the exact develop-head image pair, proves both ISAs on the full
HTTP matrix, then its dependent benchmark check consumes that same image-bound
E2E receipt. The benchmark check runs both ISAs even when the first has a
complete red ratchet report, comments the per-cell red/amber/green numbers on
the PR, and fails if any measured phase exceeds the high-water tolerance.
Manual-dispatch E2E or benchmark runs cannot satisfy these PR checks. Do not
commit benchmark evidence to the open PR head: `[skip ci]` would strand its
required checks. After a certified squash merge, the master release publisher compares
the merge **tree** with the tested develop image tree, revalidates both phase
artifacts, and promotes those immutable image digests to dated and master/SHA
tags. It attaches the per-ISA E2E and benchmark JSON plus chart to the GitHub
release. The first dated release uses concise bootstrap notes; later notes
cover commits since the preceding release. Its dependent job then commits the
combined upward-only AVX512/AVX2 high-water marks, result JSON, chart, and
README block to `develop` with `[skip ci]`. That fast-forward commit has the
tested develop head and certified squash-merged master commit as its two
parents, keeping the next strict PR up-to-date without rebuilding the image.
It does not change the certified master tree. See `docs/production-ci.md` before
editing the branch ruleset or release workflow. Keep the existing `develop`
deletion/force-push guard active, but do not require linear history there:
GitHub's auto-delete-on-merge setting must not remove that persistent branch
before the post-merge evidence commit.
On the production ARC scale set, the runner uses the host Docker Unix socket
rather than DIND so CI and local host work use one daemon and one safe Docker
cache. Never point two Docker daemons at the same writable data root. Its
workspace and model tmpfs mounts are explicit same-path roots declared through
`LLAMINAR_DOCKER_SHARED_ROOTS`; consult `docs/production-ci.md` before changing
the runner deployment.
Do not run fixed-depth MTP cells or mathematical HF parity by default.

Token drift or a suspected accuracy bug calls for the **specific matching
diagnostic mathematical HF parity cell**: retain its failing HTTP request and
tokens, select the same model/topology/precision/movement/MTP configuration,
and use checkpoint CSVs to locate the first divergent stage. Do not respond
by regenerating expected tokens or routinely launching the entire expensive
mathematical matrix. Fixed-depth mathematical coverage remains available for
focused diagnosis; the full mathematical suite is explicit opt-in only.
For an image pipeline, `--diagnostic-mathematical-parity` is that explicit
opt-in; `--through parity` alone is rejected. The normal piecewise boundaries
are `--through prerequisites` and `--through generation` before HTTP E2E.
The procedures below for numerical campaigns describe that diagnostic workflow,
not an additional routine CI regression gate.

## Start with the contract

Choose the workflow below before loading model-specific instructions. Treat
`tests/v2/CMakeLists.txt` plus generated CTest registration as the test inventory
source of truth; never copy an axis table into this skill or a runner script.
For model/generation/HTTP runs, read `tests/v2/integration/parity/README.md` and
the applicable model definition or fixture. For image certification and remote
infrastructure, also read `docs/production-ci.md`.

A production parity cell must:

- load real weights and execute the production runner, graph, kernels,
  collectives, streams, arenas, and KV policy;
- compare prefill and incremental decode at every published checkpoint;
- emit all six canonical numerical CSVs, `prefix_restore.csv`, and
  `production_path.csv` (plus `mtp_transactions.csv` when MTP is enabled);
- fail on missing model, backend, reference, snapshot boundary, or path proof;
- require the exact backend-native captured generation policy for a homogeneous
  GPU path: one complete parent where graph conditionals exist, or HIP's
  authenticated ticket-selected retained transaction graphs for MTP;
- keep the exact configured backend, precision, topology, and optimization;
- meet the single global 4,500-second performance target shared by the entire
  discovered matrix. Crossing the target must not stop or truncate correctness
  and artifact collection.

Do not certify eager replay, a test-only kernel, a host mirror, disabled
optimization, fewer stages, relaxed thresholds, or a synthetic model as a
substitute for the production path.

## Choose the workflow

- For the full model-free Unit and production-preflight gate, use
  `scripts/ci/run_production_prerequisites.py --build-dir BUILD --output NEW_DIR`
  or the direct CMake/CTest commands below. Refresh once after a build or test
  inventory change, not for every unchanged diagnostic cell.
- For public-surface generation regression, use
  `scripts/ci/run_model_parity_generation.py --mode regression`; follow the parity README's
  control acquisition, exact-token, prefix-restore and movement contracts.
  Diagnostic controls do not automatically certify an image.
- For HTTP needle/long-context tests, use `scripts/ci/run_model_parity_e2e.py`;
  for remote MPI, follow **Cross-host E2E certification** below. Both derive
  their eligible cells from the canonical definitions.
- For complete AVX512/AVX2 image and benchmark certification, use
  `scripts/ci/run_production_pipeline.py` and `docs/production-ci.md`. Both full
  E2E server suites precede benchmarks; one-off benchmarks cannot certify images.
- Before publishing a changed benchmark runtime, use
  `run_model_parity_benchmarks.py --diagnostic --diagnostic-binary PATH` with
  the complete exported typed E2E manifest. This local Release run preserves
  the canonical 512-token prefill/256-token decode workload and all cells but
  cannot mint an image certificate. Verify newly tagged HTTP cells locally as
  well; only the published-image PR workflow can certify the image pair.
- To explicitly inventory or run diagnostic mathematical campaigns, follow **Run a campaign**.
- For a numerical or path failure, also read
  [CSV evidence](references/csv-evidence.md).
- To regenerate, authenticate, or change the Python oracle, also read
  [Hugging Face reference](references/huggingface-reference.md).
- To add a backend, model, precision, topology, MoE policy, placement, or MTP
  depth, follow **Extend the matrix**; validate new math explicitly before
  acquiring controls, then exercise the routine generation projection.
- For a discovered defect, add a focused regression before folding the
  invariant into its production cell.

### Cross-host E2E certification

Remote MPI cases are another typed E2E projection, not a second model matrix.
The builder writes `cross-host-manifest.json` from the full inventory; the
production pipeline runs the HTTP long-needle suite first, then
`scripts/ci/run_production_cross_host_e2e.py`, and only then benchmarks. The
runner uses the existing Azure CLI login (CI performs federated login), creates
fresh owner-tagged CPU VMs through `azure_cross_host_resources.py`, stages the
immutable controller image and every declared model shard, and invokes the public HTTP
server with its MPI hostfile. The frontend owns bootstrap; MPI daemons and
inference children run inside source-tree-identical Release images on each
host. A controller may use AVX512 while a remote CPU peer uses the explicitly
admitted AVX2 sibling: the runner reads host CPU flags before transfer and
rejects an incompatible image rather than masking a SIGILL as an MPI failure. Each
scenario must execute both
the `plan-apply` and `auto-serve` routes, prove nonlocal CPU expert work and
matched transport evidence in prefill and decode, and retire its exact Azure
lease. A pipeline-owned empty projection emits an explicit not-applicable
report; a missing report, fallback to local execution, or incomplete cleanup is
fatal. Keep credentials outside image layers and reports; only the ephemeral
controller container receives the SSH key, and no container receives Azure
credentials. Use
private VM addresses for MPI and the public address only for authenticated
control-plane SSH. Fresh Ubuntu peers receive only a secret-free cloud-init
bootstrap for Docker/SSH/TUN, and the runner waits before staging bytes. The
runner owns its SSH tunnel and private Azure return route, not a pre-existing
VPN. Collect remote rank evidence before container retirement and preserve
artifacts after temporary staging cleanup. Recover interrupted leases only
with their exact receipt; see `docs/production-ci.md` for prerequisites.

### Long-context server stress

Use the mature `tests/v2/e2e/server/long_context_checks.py` needle primitive
when a live server fix needs a sustained correctness/lifetime proof. This is a
focused stress diagnostic, not a replacement for the complete eight-check E2E
certificate and not permission to maintain another model/topology matrix.

Start the stress clock only after the Release server publishes `ServerReady`.
First run one `run_needle_check` for each of `beginning`, `middle`, and `end` so
all three exact prompts are admitted to prefix cache. Take the steady-state
memory baseline after that warm cycle. Then rotate the same three placements
for the requested duration (normally 900 seconds), failing on the first recall,
HTTP, JSON, timeout, or server-process error. Use the cell's declared context,
minimum prompt length, thinking-model setting, output budget, and request
timeout; do not shorten the prompt or silently restart the server to manufacture
a pass.

Every canonical stress run also includes at least one production HTTP
generation with an exact `max_tokens: 1024` budget. It is an accuracy oracle,
not merely a long-output smoke test: use the cell's reviewed deterministic
control prompt, seed, prompt-token IDs, 1,024 completion-token IDs, and finish
reason, and compare them with `scripts/ci/generation_tokens.py`. Send
`return_token_ids: true` and `return_runtime_summary: true`, then retain the
complete response beside the per-iteration stress artifact. The request must
retain the cell's long-context needle geometry so the same transaction proves
long-prompt state and sustained decode. Structured-output and degeneration
checks may supplement this exact token comparison but cannot replace it. The
short needle answers commonly stop after only a few tokens and therefore
cannot prove a decode maintenance cadence, MTP depth behavior, movement epoch,
or post-movement arithmetic by themselves.

Publish an atomically replaced JSON artifact after every request. Retain the
iteration, placement, pass/detail, request wall time, elapsed stress time, and:

- RSS for every `llaminar2 serve` MPI process;
- CUDA compute-process memory for every participant;
- ROCm VRAM usage for every participant.

Classify the warm cycle separately: new RAM/disk prefix-cache admission may
grow memory there. Exact repeat restores after the steady-state baseline must
not show monotonic CPU or GPU growth. Inspect per-rank mappings or allocator
state rather than dismissing a slope as noise; short-lived participant threads,
unbounded diagnostic tags, retained request state, and driver allocations are
different defects. A completed stress proof requires the full requested wall
time, at least one pass at every needle placement, zero failed iterations,
stable post-warm device memory, and no unexplained post-warm host-memory trend.

Stop the server through its normal signal/HTTP shutdown path after collecting
the terminal sample so MPI followers, background maintenance, graph resources,
and PerfStats flush cleanly. Search the complete server log for asynchronous
CUDA/HIP errors and fatal/warning lifecycle diagnostics. When Dynamic
ExpertOverlay is selected, the 1,024-token response must prove device/host
authority as declared by the topology, nonzero completed decision windows, and
a complete authority-owned movement ledger. If the cell is intended to prove
movement, require at least one published movement wave, transaction, command,
and nonzero physical bytes, then validate the expected promotion/demotion and
same-priority axes. Accuracy and memory stability alone do not prove that
maintenance actually ran. A no-movement economy result is valid only for a
cell explicitly testing that outcome and must retain its typed rejection
reason.

## Run a campaign

Configure and build Integration with every available backend. Use Ninja and
unrestricted parallelism.

```bash
cmake -B build_v2_integration -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Integration
cmake --build build_v2_integration --parallel
```

Audit discovery before loading models:

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration --list
```

The audit must reject wildcard GTest filters, missing production labels,
missing `forward_graph` PerfStats, wrong timeouts, duplicate cases, and empty
selection. Record campaign and exact-cell counts when changing coverage.

The first non-list run for a build first builds the CMake-owned `v2_unit_gate`
target and runs the complete `V2_Unit_*` namespace. It then runs the
`ProductionTestPreflight` integration label, all before the model fixture or
RAM staging. Unit tests prove device-free policy and state-machine invariants;
preflight proves MPI/rank lifecycle, orchestration, explicit-stream event
ordering, graph capture/replay, retained heterogeneous tickets, prepared
ExpertOverlay weights, and CUDA/ROCm overlay epochs. Both inventories live only
in CTest registration: the Python driver discovers and validates them instead
of copying test or executable names. A preflight-label member must be an
`Integration` test, require no fixture or external file, and have a timeout no
greater than 120 seconds. Any build, unit, or preflight failure stops model
admission and is recorded by the combined preflight receipt.

Amortize that receipt across unchanged local diagnostic runs; do not rebuild or
rerun the two prerequisite suites for each cell. Numerical unseen-cell runs use
`--reuse-passed-preflight-report PATH`; the generation driver uses
`--reuse-preflight-report PATH`. Both delegate to the same receipt validator,
which checks the build directory, Ninja/CTest freshness and complete test
inventory. A stale receipt is a hard error, not permission to bypass the gate.
After a fix, run its focused regression first, finish all required executable
builds, then refresh Unit/preflight once before resuming the affected and unseen
cells. Receipt reuse does not reuse a cell's pass or certify an image.

Run the same prerequisite phases directly when developing their infrastructure:

```bash
cmake --build build_v2_integration --parallel \
  --target v2_unit_gate v2_production_test_preflight_gate
ctest --test-dir build_v2_integration \
  --output-on-failure --parallel --no-tests=error -R '^V2_Unit_'
ctest --test-dir build_v2_integration \
  --output-on-failure --parallel --no-tests=error \
  -L '^ProductionTestPreflight$'
```

Git pre-commit runs only these two complete model-free suites on every branch.
The hook builds their CMake-owned dependency targets and does not launch
numerical campaigns, E2E, containers, Release builds, or benchmarks. Register
the tracked hooks with `git config --local core.hooksPath .githooks`; see
`.githooks/README.md`. Passing pre-commit is prerequisite evidence, not a
numerical, HTTP, or performance certificate. Keep heavier manual/CI gates
separate unless the user explicitly changes that policy.

For development, run the narrow affected backend or precision first. These are
diagnostic subsets, not the final economy proof.

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --backend 'CUDA' --precision 'ALL' \
  --report parity-results/cuda-production-campaigns.json
```

Do not invoke a registered `ProductionCampaign_*` child directly with CTest.
Those entries are the driver's internal discovery/scheduling units and fail
before model mapping when the identity-bound tmpfs contract is absent. Narrow a
real campaign with the driver's `--campaign`, `--backend`, and `--precision`
selectors so staging, identity locking, artifacts, and the global timing
authority remain intact.

For one already identified red, use the driver's exact registered-cell route,
not a hand-written `mpirun` command. It validates the requested fully qualified
GTest identity against CTest discovery, retains the exact registered process
contract, runs the normal fixture/staging path and CSV validator, and records a
non-certifying focused report. It is the shortest valid feedback loop for an
HF checkpoint diagnosis; it does not update the green ledger or replace a
later unfiltered campaign. After a current prerequisite receipt exists, add
`--reuse-passed-preflight-report PATH` only when its build/CTest identity
validator accepts it.

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --backend 'CUDA' --precision 'ALL' \
  --campaign 'V2_Integration_Parity_Qwen36MoE_ExpertOverlay_ProductionCampaign_CUDA_ALL_PRECISIONS' \
  --exact-cell 'Suite/Fixture.ProductionParity/Registered_Model_Topology_Static_Ordinal_ActFP32_KVFP16_MTPOff' \
  --model-ramdisk-root /mnt/llaminar-production-parity \
  --persistent-model-cache-dir cache \
  --report parity-results/focused-exact-cell.json
```

For an explicitly requested full mathematical diagnostic, run the unfiltered
matrix. The 75-minute requirement is one wall-clock
performance target for all campaigns, backends, precision types, and fixtures
combined. It is not a per-campaign timeout or a reason to abort unfinished
cells. Each exact generated GTest cell nevertheless has a fixed 900-second
progress watchdog. A newly published per-cell `test_log.txt` transfers the
deadline to that cell; expiry kills its complete CTest/MPI process group and
records the exact identity as `exact_cell_timeout`. `GTEST_FAIL_FAST=1` is part
of every registered production campaign contract:
the first exact red ends its process-amortized aggregate. The driver publishes
that aggregate as the sole first-failure authority, cancels already-running
backend-disjoint sibling process groups, records them as cancelled rather than
additional reds, and admits no pending campaigns. Diagnose that preserved
first-failure evidence before restarting the unfiltered matrix. Use
`scripts/ci/setup_production_parity_tmpfs.sh` once for persistent
local iteration. Its named mount is independent of host-login `/dev/shm` IPC
cleanup, and repeated setup preserves the existing cache. Persistence still
ends when that mount ends: reboot, container replacement, or explicit
unmount/remount necessarily discards it. Campaign failure and successful exit
do not. Follow the parity README's explicit unmount command for manual cleanup;
never use a broad shared-memory deletion.

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --target-seconds 4500 \
  --model-ramdisk-root /mnt/llaminar-production-parity \
  --persistent-model-cache-dir cache \
  --report parity-results/production-campaigns.json
```

After an interrupted local sweep, repeat
`--prioritize-unseen-from-artifact-root PATH` for each preserved campaign
artifact root to run wholly unseen aggregates before partial aggregates and
previously complete aggregates. This is an explicit scheduling hint only. It
never removes a selected exact cell, reuses an old pass, or weakens the active
run's freshness and full CSV validation; the restarted unfiltered run must
still rerun the complete discovered matrix for certification.

Success requires exit zero, every discovered campaign completed, every exact
cell represented, `correctness_passed: true`, `global_target_met: true`,
`performance_requirements_met: true`, and `global_elapsed_seconds <= 4500`.
It also requires `preflight_return_code: 0` and a nonempty `preflight_tests`
inventory containing the complete Unit namespace and Integration preflight
label. Unit build/test and preflight elapsed time are part of that single global
wall-clock budget.
The separate completion timeout guards setup phases that cannot publish an
exact-cell transition. It is not a cumulative matrix deadline. Once inference
starts, the fixed fifteen-minute exact-cell watchdog is the sole timeout authority;
each fresh `test_log.txt` transition renews that complete budget.
Keep the JSON and CSVs as CI artifacts, but never commit local result
directories.

## Verify production-path evidence

Read `production_path.csv` before interpreting numerical output. Every
homogeneous CUDA or ROCm row needs graph execution, native decode capture or
replay, and zero segmented plan/capture/replay. A non-MTP row also needs
complete `full_graph_*` capture or replay. An MTP row must report
`device_generation_controller=true` and one of these exact policies:

- CUDA: `generation_execution_policy=native_conditional_parent`,
  `native_generation_parent=true`, `generation_loop_certified=true`, and a
  complete `full_graph_*` capture/replay.
- ROCm/HIP: `generation_execution_policy=host_scheduled_captured_transactions`,
  `native_generation_parent=false`,
  `hosted_ticket_boundary_certified=true`,
  `generation_loop_certified=true`, and a complete
  `forward_full_graph_*` transaction capture/replay. The truthful complete
  `full_graph_*` fields remain false because HIP cannot represent the outer
  conditional loop.

The ROCm certificate is not granted from the policy spelling. PerfStats must
prove the exact ABI-v2 52-byte `immutable_scheduler_snapshot` with
`state_payload=false`, authenticated observations, retained HIP graph-family
materialization, per-device captured transaction/terminal submissions, matching
controller and compact-reducer ledgers, and either exact standalone launches or
the rank launch that submits every participant before the next ticket wait.
`generation_certification_detail` gives the first failed invariant. Any other
intermediate D2H/host-materialization boundary fails closed.
CPU still needs the production graph execution path. Heterogeneous cells may
segment only at their explicitly declared backend or collective boundary.

Use `IOrchestrationRunner::moeOptimizationMovementLedger()` as the sole typed
truth for completed ExpertOverlay movement. Its immutable edges are authored
by the host or device placement authority and exported as
`expert_movement.csv`; PerfStats is lifecycle, economy, transport, and
observability evidence only. Never infer a completed objective from a proposal
counter or from physical direction: a participant-skew correction can be
folded into promotion/demotion endpoints, while a same-priority edge can merely
close a tier-residency capacity cycle. The ledger's `movement_axis` therefore
distinguishes `tier_residency`, `participant_placement`, and `combined` from
the independent physical `direction` column. Host-authoritative topologies
mirror diagnostics through `moe_overlay_residency`; device-authoritative
all-GPU topologies mirror accepted transactions and physical commits through
`moe_overlay_controller` and `moe_overlay_residency`. The required facts remain
an authority-owned typed objective, staged/copied payload, nonzero physical
bytes, and a published destination placement:

- Static: planner, copy, transport, ownership/replica, and destination-apply
  movement counters are all zero.
- Dynamic: a maintenance boundary and routing-window synchronization occurred;
  at least one ownership change or replica arrival occurred; packed expert
  bytes crossed transport and were applied at the destination.
- LLEP: the production begin/restore transaction ran; non-owner rows were
  assigned; at least one packed expert and nonzero bytes moved.

Static additionally requires an empty authoritative movement ledger. Dynamic
two-axis topologies require at least one ledger edge whose axis advances tier
residency and one whose axis advances participant placement; a `combined` edge
satisfies both without demanding a gratuitous extra transfer.

For public HTTP generation probes, validate every terminal
`runtime_summary.expert_movement` snapshot with the shared
`generation_movement_ledger.py` consumer. It checks immutable model-lifetime
history, complete physical cycles and their owner-authored economy/admission
proofs in both live and saved observations. Missing old evidence must fail,
not become an empty Static journal. Device publication can recompose several
logical objectives into one physical component; retain each command's axis
rather than requiring identical axis labels within that component. The sibling
`expert_movement_topology` comes from the runner's canonical frozen model plan,
using the same geometry projection as deep parity. Require immutable geometry,
the declared authority, and every available objective axis over a Dynamic
cohort. Static retains an empty journal even with available axes. This is a
passive evidence check, not a second planner. After shutdown the HTTP harness
independently checks published payloads and matched transport/owner publication,
then joins its committed expert identities to the HTTP journal. See the
transport evidence section of [CSV evidence](references/csv-evidence.md) before
diagnosing a missing movement witness. A journal-only saved pass is not complete
transport proof.

Generation requires its full continuous minimum even for models without MTP.
During initial acquisition, a deliberate model-owned prompt/seed search may
find a suitable long workload. Keep candidates diagnostic, freeze the chosen
workload before canonical collection, and verify it across affected topologies
and precisions. Never retry seeds inside the gate, suppress EOS, or waive exact
repeats and prefix restoration for a short answer.

Do not equate physical-movement coverage with the longer matched throughput
cohort. Every Dynamic cell owns `EconomicMovement`. A model/topology definition
may additionally select `dynamic_speedup_witness` as ordinal, random, both
owner orders, or disabled. The expander places that before/after speed proof on
only the first activation/KV pair, ordinary prefill profile, and MTP-off cell;
all remaining cells retain their complete movement, numerical, graph, prefix,
and CSV obligations. Add a representative only when a new transport class or
materially different topology economy needs one, and never infer witnesses in
the fixture from a test name or from the mere presence of CPU participants.

Use `moe_placement` separately to authenticate the requested ordinal or seeded
random physical owner map. Setup placement is not request-time movement.
ExpertOverlay cells also retain `expert_residency_diagnostics.csv` (and one
rank-suffixed file per follower) so a failed or successful run can be audited
without reconstructing cross-rank controller state from interleaved logs.

When `promoted_expert_execution.csv` proves the numerical result of a moved
expert, an independently routed row may be exactly zero in both native and
reference execution. That is an exact equality witness, recorded as
`exact_zero_route_rows`; it is not missing execution. A row that is zero on
only one side is recorded as `one_sided_zero_route_rows` and fails closed.

MTP coverage must include fixed depths 1, 2, 3, and 15 plus dynamic depth on
CPU, CUDA, and ROCm. Compare recursive sidecar checkpoints and final verifier
outputs, assert the requested/effective depth and dynamic-controller activity,
and require the same graph evidence as ordinary inference. Dynamic depth needs
one serial-token-exact, full-budget adaptive witness whose controller-window,
attempted-draft, and verifier counters increase; a physically deep transaction
that was clipped at an ExpertOverlay maintenance boundary is not adaptive-depth
proof. MoE MTP inherits the same policy and placement movement contract.

## Diagnose and fix a failure

1. Preserve the failing result directory and global report.
2. Confirm model/reference identity and production path before changing any
   tolerance.
3. Find the first failing phase, decode step, layer, and stage from the CSVs;
   later divergence is usually propagation, not the cause.
4. Compare shape/count, NaN/Inf, routing identity, cosine, relative L2, maximum
   absolute error, RMSE, SNR, distribution drift, KL, and Top-K evidence.
5. Map the first bad stage to graph wiring, tensor layout, kernel arithmetic,
   collective order, KV state, MoE movement, or MTP advancement.
6. Reproduce with one exact CTest/GTest cell. Loop a flaky case up to 20 times.
7. Fix the production implementation. Do not weaken the reference, skip a
   checkpoint, raise a timeout, or enter another path.
8. Add a focused device-free unit or backend integration regression for the
root invariant. Every `V2_Unit_*` test joins the prerequisite automatically. If
an Integration test is model-free and guards campaign infrastructure or a
production mechanism used by parity, add its existing CTest registration to
`V2_PRODUCTION_TEST_PREFLIGHT_TESTS`; do not copy either inventory into the
Python driver. Then rerun both prerequisite phases and the affected diagnostic
cell before resuming routine generation regression. Run the full mathematical
matrix only when explicitly requested.

If only the full campaign fails, inspect reuse boundaries: `ModelContext` may
retain immutable prepared weights only when its complete physical identity is
unchanged. Runner, arena, stream, graph, request snapshots, KV data, runtime
MoE policy, and controller state must be exact per cell and reset explicitly.

## Extend the matrix

Describe supported configurations with `ModelParityDefinition`: one immutable
real-model identity, one `ModelParityTopologyDefinition`, the precision matrix,
and standard feature profiles. Expand it with
`expandModelParityDefinition()` and keep the resulting `ModelParityCase` as the
test parameter. Give each topology an unambiguous backend and collective
identity. Add a `ProductionParity` test calling
`runProductionParityCampaign()`, or a narrowly specialized fused body when the
feature needs additional path evidence. Do not recover runtime policy by
parsing the generated GoogleTest name and do not build a second axis expander
inside a model fixture.

Register through `discover_v2_parity_tests()`. Supply
production binaries named `v2_integration_parity_<model>_<scope>_matrix`, with
model/scope fixture names; keep topology instances, precision, placement,
movement, and MTP axes in generated cell names. Discovery rejects production
registrations without the matrix binary role. Follow the parity README's naming
contract and compare exact parameter inventories when renaming suites; never
relabel historical CSV evidence as a new run. Supply
`PRODUCTION_PERF_STATS_FILTER` when the cell must assert domains beyond
`forward_graph`. Build the target to regenerate its CTest include, then verify
that `--list` increases by exactly the intended cells and does not lose an old
axis.

For new ExpertOverlay coverage, the standard supported matrix crosses
Static/Dynamic with ordinal/random placement and MTP off/1/2/3/15/dynamic,
asserting movement semantics on CPU, CUDA, and ROCm. Add LLEP only after its
topology is a first-class installed implementation with its own exact movement
and economy contract; never route an unfinished LLEP cell through Dynamic or
Static. Add each newly supported model, topology, activation precision, KV
precision, tensor format, sampling mode, or collective through the same typed
definition and registration source. Prefix restore is not a multiplicative
axis: every generated production cell automatically proves fresh seed, full
hit, and partial hit through the production tiered cache.

After the affected slice is green, run framework checks and routine generation
regression. The complete mathematical command below is explicit diagnosis only:

```bash
python3 tests/v2/unit/scripts/test_production_parity_campaigns.py
python3 -m pytest -q python/reference/tests/test_snapshot_metadata.py
ctest --test-dir build_v2_integration \
  -R '^V2_Unit_ProductionParityCampaigns$' --output-on-failure
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration --target-seconds 4500 \
  --persistent-model-cache-dir \
    "${LLAMINAR_PRODUCTION_PARITY_CACHE_DIR:-/dev/shm/llaminar-production-parity-cache}" \
  --report parity-results/production-campaigns.json
```

Keep `.githooks/pre-commit`, `.github/workflows/ci.yml`, this skill,
`tests/v2/integration/parity/README.md`, and the short parity references in
`AGENTS.md` synchronized when the canonical command or gate changes.
