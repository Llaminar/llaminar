---
name: model-parity-testing
description: Run, extend, maintain, and diagnose Llaminar V2 real-weight production parity campaigns against the CPU/FP32 PyTorch-Hugging Face reference. Use when changing inference mathematics, graph capture, model wiring, quantization or KV precision, backends/topologies, MoE Static/Dynamic/LLEP behavior or expert placement, MTP depth/control, parity CSVs, campaign discovery, or the one-hour precommit/CI parity gate.
---

# Model Parity Testing

Use the production campaign system to prove the live inference path against an
authenticated reference pack made from the exact GGUF bytes. Preserve every
stage comparison and canonical CSV while amortizing immutable model setup.

## Start with the contract

Read `tests/v2/integration/parity/README.md` and the applicable model fixture.
Treat `tests/v2/CMakeLists.txt` plus generated CTest registration as the matrix
source of truth; never copy an axis table into this skill or a runner script.

A production parity cell must:

- load real weights and execute the production runner, graph, kernels,
  collectives, streams, arenas, and KV policy;
- compare prefill and incremental decode at every published checkpoint;
- emit all six canonical numerical CSVs and `production_path.csv`;
- fail on missing model, backend, reference, snapshot boundary, or path proof;
- require the exact backend-native captured generation policy for a homogeneous
  GPU path: one complete parent where graph conditionals exist, or HIP's
  authenticated ticket-selected retained transaction graphs for MTP;
- keep the exact configured backend, precision, topology, and optimization;
- meet the single global 3,600-second performance target shared by the entire
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

For development, run the narrow affected backend or precision first. These are
diagnostic subsets, not the final economy proof.

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --backend 'CUDA' --precision 'ALL' \
  --report parity-results/cuda-production-campaigns.json
```

Then run the unfiltered matrix. The one-hour requirement is one wall-clock
performance target for all campaigns, backends, precision types, and fixtures
combined. It is not a per-campaign timeout or a reason to abort unfinished
cells. Use the persistent tmpfs cache for rapid local iteration; the driver
authenticates and reuses unchanged GGUFs and leaves cleanup to the operator.

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --target-seconds 3600 \
  --persistent-model-cache-dir \
    "${LLAMINAR_PRODUCTION_PARITY_CACHE_DIR:-/dev/shm/llaminar-production-parity-cache}" \
  --report parity-results/production-campaigns.json
```

Success requires exit zero, every discovered campaign completed, every exact
cell represented, `correctness_passed: true`, `global_target_met: true`,
`performance_requirements_met: true`, and `global_elapsed_seconds <= 3600`.
The separate completion timeout is only a hang guard and must remain generous
enough to collect the complete correctness matrix after a performance miss.
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
prove the exact 48-byte `immutable_scheduler_snapshot` with
`state_payload=false`, authenticated observations, retained HIP graph-family
materialization, per-device captured transaction/terminal submissions, matching
controller and compact-reducer ledgers, and either exact standalone launches or
the rank launch that submits every participant before the next ticket wait.
`generation_certification_detail` gives the first failed invariant. Any other
intermediate D2H/host-materialization boundary fails closed.
CPU still needs the production graph execution path. Heterogeneous cells may
segment only at their explicitly declared backend or collective boundary.

Use request-local `moe_rebalance` PerfStats for movement truth:

- Static: planner, copy, transport, ownership/replica, and destination-apply
  movement counters are all zero.
- Dynamic: a maintenance boundary and routing-window synchronization occurred;
  at least one ownership change or replica arrival occurred; packed expert
  bytes crossed transport and were applied at the destination.
- LLEP: the production begin/restore transaction ran; non-owner rows were
  assigned; at least one packed expert and nonzero bytes moved.

Use `moe_placement` separately to authenticate the requested ordinal or seeded
random physical owner map. Setup placement is not request-time movement.

MTP coverage must include fixed depths 1, 2, and 3 plus dynamic depth on CPU,
CUDA, and ROCm. Compare recursive sidecar checkpoints and final verifier
outputs, assert the requested/effective depth and dynamic-controller activity,
and require the same graph evidence as ordinary inference. MoE MTP inherits the
same policy and placement movement contract.

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
8. Add a focused device-free or backend integration regression for the root
   invariant, then rerun the affected campaign and the full matrix.

If only the full campaign fails, inspect reuse boundaries: `ModelContext` may
retain immutable prepared weights only when its complete physical identity is
unchanged. Runner, arena, stream, graph, request snapshots, KV data, runtime
MoE policy, and controller state must be exact per cell and reset explicitly.

## Extend the matrix

Add supported configurations to the existing typed `TestConfig` parameter
source. Give each case an unambiguous backend and precision identity. Add a
`ProductionParity` test calling `runProductionParityCampaign()`, or a narrowly
specialized fused body when the feature needs additional path evidence.

Register through `discover_v2_parity_tests()`. Supply
`PRODUCTION_PERF_STATS_FILTER` when the cell must assert domains beyond
`forward_graph`. Build the target to regenerate its CTest include, then verify
that `--list` increases by exactly the intended cells and does not lose an old
axis.

For new MoE coverage, cross Static/Dynamic/LLEP with ordinal/random placement
on CPU, CUDA, and ROCm and assert movement semantics. For MTP, cross all three
backends with depths 1/2/3/dynamic. Add each newly supported model, topology,
activation precision, KV precision, tensor format, prefix-cache lane, sampling
mode, or collective through the same registration source rather than a second
manifest.

After the affected slice is green, run framework checks and the complete
campaign:

```bash
python3 tests/v2/unit/scripts/test_production_parity_campaigns.py
python3 -m pytest -q python/reference/tests/test_snapshot_metadata.py
ctest --test-dir build_v2_integration \
  -R '^V2_Unit_ProductionParityCampaigns$' --output-on-failure
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration --target-seconds 3600 \
  --persistent-model-cache-dir \
    "${LLAMINAR_PRODUCTION_PARITY_CACHE_DIR:-/dev/shm/llaminar-production-parity-cache}" \
  --report parity-results/production-campaigns.json
```

Keep `.githooks/pre-commit`, `.github/workflows/ci.yml`, this skill,
`tests/v2/integration/parity/README.md`, and the short parity references in
`AGENTS.md` synchronized when the canonical command or gate changes.
