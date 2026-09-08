---
name: model-parity-testing
description: Run, extend, maintain, and diagnose Llaminar V2 real-weight production parity campaigns against the CPU/FP32 PyTorch-Hugging Face reference. Use when changing inference mathematics, graph capture, model wiring, quantization or KV precision, backends/topologies, MoE Static/Dynamic/LLEP behavior or expert placement, MTP depth/control, parity CSVs, campaign discovery, the model-free precommit prerequisites, or the 75-minute manual/CI parity gate.
---

# Model Parity Testing

Use the production campaign system to prove the live inference path against an
independent reference pack generated from the declared real GGUF. Preserve
every stage comparison and canonical CSV while amortizing immutable model
setup; numerical checkpoints, not an extra whole-model hash pass, prove weight
equivalence.

## Start with the contract

Read `tests/v2/integration/parity/README.md` and the applicable model fixture.
Treat `tests/v2/CMakeLists.txt` plus generated CTest registration as the matrix
source of truth; never copy an axis table into this skill or a runner script.

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

- To inventory or run campaigns, follow **Run a campaign**.
- For a numerical or path failure, also read
  [CSV evidence](references/csv-evidence.md).
- To regenerate, authenticate, or change the Python oracle, also read
  [Hugging Face reference](references/huggingface-reference.md).
- To add a backend, model, precision, topology, MoE policy, placement, or MTP
  depth, follow **Extend the matrix** and then run the complete matrix.
- For a discovered defect, add a focused regression before folding the
  invariant into its production cell.

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

Every non-list campaign invocation first builds the CMake-owned `v2_unit_gate`
target and runs the complete `V2_Unit_*` namespace. It then runs the
`ProductionParityPreflight` integration label, all before the model fixture or
RAM staging. Unit tests prove device-free policy and state-machine invariants;
preflight proves MPI/rank lifecycle, orchestration, explicit-stream event
ordering, graph capture/replay, retained heterogeneous tickets, prepared
ExpertOverlay weights, and CUDA/ROCm overlay epochs. Both inventories live only
in CTest registration: the Python driver discovers and validates them instead
of copying test or executable names. A preflight-label member must be an
`Integration` test, require no fixture or external file, and have a timeout no
greater than 120 seconds. Any build, unit, or preflight failure stops model
admission and is recorded by the combined preflight receipt.

Run the same prerequisite phases directly when developing their infrastructure:

```bash
cmake --build build_v2_integration --parallel \
  --target v2_unit_gate v2_production_parity_preflight_gate
ctest --test-dir build_v2_integration \
  --output-on-failure --parallel --no-tests=error -R '^V2_Unit_'
ctest --test-dir build_v2_integration \
  --output-on-failure --parallel --no-tests=error \
  -L '^ProductionParityPreflight$'
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

Then run the unfiltered matrix. The 75-minute requirement is one wall-clock
performance target for all campaigns, backends, precision types, and fixtures
combined. It is not a per-campaign timeout or a reason to abort unfinished
cells. Each exact generated GTest cell nevertheless has a fixed 600-second
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
starts, the fixed ten-minute exact-cell watchdog is the sole timeout authority;
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
`V2_PRODUCTION_PARITY_PREFLIGHT_TESTS`; do not copy either inventory into the
Python driver. Then rerun both prerequisite phases, the affected campaign, and
the full matrix.

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

After the affected slice is green, run framework checks and the complete
campaign:

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
