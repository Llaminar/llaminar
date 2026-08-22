# Production parity campaigns

The V2 parity framework compares the live Llaminar inference path against a
CPU/FP32 PyTorch reference generated from the same real GGUF bytes. It retains
the existing layer-by-layer mathematics and diagnostic CSVs while avoiding a
new model load for every prefill, decode, snapshot, backend, and KV-precision
case.

CTest registration in `tests/v2/CMakeLists.txt` is the matrix source of truth.
Do not maintain a second model/backend/precision manifest in scripts or docs.

## What one production cell proves

Each parameterized `ProductionParity` cell uses one production runner session
to perform this contract:

1. Authenticate or generate the CPU PyTorch reference pack.
2. Construct the configured production model, topology, collectives, kernels,
   arenas, streams, graph, and KV-cache policy.
3. Run prefill and compare embedding, every published per-layer stage, final
   norm, and LM-head distributions against PyTorch.
4. Assert that the live graph published the required snapshot boundaries.
5. Reset request-owned snapshots and cache data without rebuilding topology or
   recapturing merely to cross the request boundary.
6. Run incremental decode and compare every requested decode checkpoint and
   output distribution.
7. Export the normal numerical diagnostics plus structured production-path and
   wall-time evidence.

The comparison code remains in `ParityTestBase.h`: cosine similarity, relative
L2, KL divergence, Top-K overlap, distribution statistics, NaN/Inf checks, and
per-layer threshold assertions are unchanged. Fusing phases removes duplicate
setup; it does not weaken the oracle or compare fewer checkpoints.

Missing models, hardware, decode references, graph evidence, or snapshot
publication are failures in a production campaign. Focused developer tests may
still skip when their optional prerequisite is absent.

## Reference-pack identity

`metadata.txt` is published atomically after the NumPy files. A production
campaign accepts an existing pack only when it binds all of the following:

- supported snapshot schema;
- PyTorch on CPU using FP32;
- SHA-256 of the exact GGUF file used by the native runner;
- SHA-256 of the exact UTF-8 prompt bytes;
- nonempty tokenizer output and its SHA-256;
- sufficient decode depth, decode tokens, and boundary snapshots.

Before inference, the aggregate runner copies the complete selected GGUF corpus
into a private directory on `/dev/shm`. CTest's `REQUIRED_FILES` property is the
machine-readable model declaration for each production campaign. The runner
expands split GGUF siblings, resolves symlink aliases, rejects basename
collisions, verifies that the entire corpus plus a reserve fits on tmpfs, and
computes source SHA-256 while atomically publishing each read-only RAM copy.
Full-write and byte-count checks, source mutation detection, read-only mode, and
source/destination identity binding protect the transaction. Each production
child then performs the single canonical destination SHA-256 check against its
authenticated reference pack before inference; a shared identity-keyed digest
cache coalesces that check across campaigns. Any missing declaration,
non-memory filesystem, capacity shortfall, mutation during copy, reference
digest mismatch, or completion timeout is a hard failure. The staging and
authentication time are part of the same one-hour target; by default the
temporary corpus is removed when the run exits.

Runners with a different memory mount can pass
`--model-ramdisk-root /path/to/tmpfs`. The directory must already reside on
`tmpfs` or `ramfs`; the driver never falls back to disk or a partial corpus.

For rapid local iteration, explicitly retain the authenticated corpus and digest
records in a stable child directory:

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --persistent-model-cache-dir /dev/shm/llaminar-production-parity-cache
```

The first invocation computes source SHA-256 during the copy, atomically
publishes each read-only GGUF, and authenticates the published bytes once through
the reference-pack gate. Later invocations reuse an entry only when its source
device, inode, size, nanosecond mtime, nanosecond ctime, and read-only cached-file
identity are unchanged; the persistent digest cache binds the prior reference
authentication to that exact destination identity. Changed entries are replaced
atomically, while a changed cached identity is reauthenticated before inference.
One exclusive cache lock spans staging and every child inference process, so a
concurrent run cannot replace weights still in use. After acquiring that lock,
the driver reclaims only unpublished copy/manifest transaction files left by an
interrupted process or host reboot; published models and operator-owned files
remain untouched. The driver never deletes this directory; remove that exact
cache manually when no campaign owns it.

Every child process then re-authenticates the RAM copy against the reference
pack. Those streaming digests are coalesced through a private run-scoped cache,
or the explicitly selected persistent cache, keyed by canonical path, device,
inode, size, nanosecond mtime, and nanosecond ctime. This prevents both repeated
multi-gigabyte scans and stale pathname trust. Qwen2/Qwen3 use
`python/reference/generate_qwen_pipeline_snapshots.py`; Qwen3.5, Qwen3.6, and
their MoE variants use the architecture-specific generators under
`python/reference/`.

Reference generation has one node-local writer authority because every
backend's Hugging Face oracle consumes the same host cores and can require well
over 100 GiB for a 35B model. The ordinary fixture and architecture-specific
MTP helpers use `ReferenceGenerationLease` and follow the same double-checked
publication lifecycle:

```mermaid
flowchart LR
    A[Validate authenticated pack] -->|usable| R[Read concurrently]
    A -->|missing or stale| L[Acquire node-local writer lease]
    L --> V[Validate again]
    V -->|peer published while waiting| R
    V -->|still missing| G[Run one CPU Hugging Face generator]
    G --> P[Atomically publish and authenticate]
    P --> R
```

The lease covers only generation and publication. Validated reference reads
and production CPU/CUDA/ROCm inference still overlap. Any new custom reference
helper must join this lifecycle; spawning Python directly from a backend test
creates an unbounded RAM race and is not a supported campaign path.

## Process campaign and ownership

`discover_v2_parity_tests()` runs `--gtest_list_tests` after each parity binary
is built. Focused diagnostics remain isolated CTests. Every discovered
`ProductionParity` case is instead placed in one exact, wildcard-free GTest
filter per test type and backend signature. All precision cells for that test
type/backend execute in one process.

Across compatible non-overlay precision cells, the process may retain one
bounded `ModelContext` and its authoritative `PreparedWeightStore`. The reuse
key includes model path, weight distribution, topology, devices, collectives,
activation precision, ranks, and PP partitioning. KV precision is deliberately
excluded when it cannot affect prepared-weight capacity because KV storage is
runner-owned. Every cell still constructs an exact runner, arena, stream set,
graph identity, and KV policy; a key change evicts the prior model before
loading another, so the cache cannot grow without bound.

ExpertOverlay has a stricter boundary. A bare `ModelContext` cannot prove the
resolved rank plan, model-frozen tier quotas, owner order, or prepared expert
set. Fresh overlay cells let the production runner own loading and
certification. Reuse is legal only through a `ModelContextReuseContract`
emitted by an initialized runner and accepted after exact plan and routed-weight
identity validation. Cells that change residency policy, owner placement, or
another capacity-affecting field construct a fresh authority rather than
guessing compatibility in the fixture.

## Production-path evidence

Production campaigns force declarative graph execution and enable structured
PerfStats. Every homogeneous CUDA-only or ROCm-only cell requires:

- native decode graph capture or replay;
- no segmented plan, segmented capture, or segmented replay.

Ordinary non-MTP inference additionally requires a complete native graph
capture or replay. MTP uses a backend-specific generation proof:

- CUDA requires one native conditional parent and complete `full_graph_*`
  capture/replay.
- ROCm requires HIP's authenticated 60-byte immutable scheduler ticket and the
  retained captured transaction family it selects. The device controller owns
  every decision and mutable value; the host only submits the named branch.
  Consequently `forward_full_graph_*` proves each captured transaction while
  the complete `full_graph_*` fields truthfully remain false.

Explicitly heterogeneous placements may segment only at their declared
backend/collective boundary. CPU cells must still use the production graph
execution path. No campaign falls back to eager execution, deterministic test
kernels, serial replay, or a different backend.

Each results directory contains `production_path.csv`, recording execution
path, inner-forward versus complete graph evidence, MTP generation-controller
authority, backend policy certification, segmentation evidence, model-context
reuse, elapsed time, budget, and pass/fail status. ROCm certification requires
exact ticket ABI/provenance, participant graph-family evidence, matching
ticket/submission/controller ledgers, and a standalone or participant-complete
rank launch. `generation_certification_detail` identifies the first missing
invariant. Every mathematical production cell retains the canonical numerical
artifacts:

- `prefill_layers.csv`
- `prefill_summary.csv`
- `prefill_stages.csv`
- `decode_steps.csv`
- `decode_layers.csv`
- `decode_stages.csv`

MTP cells additionally write `mtp_sidecar_token_trace.csv`. Its committed
verifier columns (`verifier_identity_transaction_count`,
`verifier_identity_depth`, and `production_verifier_draft_tokens`) come from
the device-owned identity published by the same fused response/state commit
that advances the generation controller. They must agree with the transaction
delta and selected depth for every speculative call; reusable proposal or
verifier-input scratch is not admissible post-transaction evidence. A terminal
absorbing call may retain the preceding committed identity while selecting
depth zero, because it does not claim a new verifier transaction.

Cross-rank pipeline cells first write rank-local diagnostic fragments, then
merge them into the same canonical files. The merged result must contain every
owned layer and stage from every rank, the head-rank embedding boundary, and
the tail-rank norm/LM-head distribution; partial rank-zero CSVs are a failure.
Temporary rank fragments are removed after a successful merge.

MoE policy cells certify behavior through request-local `moe_rebalance`
PerfStats in addition to checkpoint mathematics. Fresh Dynamic and LLEP cells
must prove a payload was copied, nonzero packed expert bytes crossed the live
transport, and the destination placement was applied. Static-owner cells assert
that CPU, CUDA, and ROCm planner/copy/transport/apply movement counters all
remain zero. Setup-only `moe_placement` records separately authenticate ordinal
or random physical expert placement and are not counted as request movement.
A restored full-prefix request need not invent a redundant migration; its
preceding fresh request must already have supplied the positive movement proof.

## One-hour economy gate

The complete discovered matrix is the performance unit. All model families,
topologies, backends, precision cells, MoE policies/owner placements, and MTP
depths selected for a run share one 3,600-second wall-clock target. A CTest
campaign is only a process-resident scheduling unit used to amortize immutable
weights and claim backend resources; it does not receive a fresh hour. Missing
the target is a performance failure reported after the matrix completes, not a
reason to stop admitting work or discard later correctness evidence.

The driver stages the download fixture once, stages and authenticates the
selected real weights into RAM once, overlaps only backend-disjoint work,
runs every campaign even after the target is missed, and records the complete
correctness result. An independent 21,600-second completion timeout is a
stuck-run safety ceiling, not the economy requirement; configure it separately
with `--completion-timeout-seconds`. C++ cells write timing evidence but do not
assert the one-hour target individually. `production-campaigns.json` records
separate correctness and performance statuses, staged byte count,
source/destination digests, staging filesystem/time, and campaign coverage.

List the configured coverage without loading a model:

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration --list
```

The listing validates campaign labels, environment, timeout, declared GGUF
manifest, forward-graph PerfStats, and exact GTest filters, then reports
campaign count, exact matrix cell count, backend signatures, and precision
tags.

Run every configured campaign and write machine-readable timing evidence:

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --report parity-results/production-campaigns.json
```

Architecture slices retain the same discovery authority, backend-exclusive
scheduler, whole-slice target, and JSON evidence. Select only registered CTest
campaign names; do not maintain a copied model/topology list. For example, a
non-ExpertOverlay certification pass can omit the tier implementation slice:

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --exclude-campaign '.*(?:ExpertOverlay|MoEGraphNative).*' \
  --report /tmp/non-overlay-production-campaigns.json
```

Filter with full-match regular expressions when validating one backend or
precision subset:

```bash
python3 scripts/ci/run_production_parity_campaigns.py \
  --build-dir build_v2_integration \
  --backend 'CUDA' --precision 'ALL'
```

The report path is generated debris and must not be committed.

To run one exact campaign directly, discover its name first:

```bash
ctest --test-dir build_v2_integration -N -L Campaign
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen2_SingleDevice_ProductionCampaign_CUDA_ALL_PRECISIONS$' \
  --output-on-failure
```

## Adding or extending a matrix

Use a substantive file header and keep configurations declarative.

1. Add each supported model/topology/backend/precision configuration to the
   fixture's existing `TestConfig` parameter list. Backend identity must appear
   unambiguously in the generated case name as `CPU`, `CUDA`, or `ROCm`.
2. Implement one `ProductionParity` test that calls
   `runProductionParityCampaign()`, or an architecture-specific fused body when
   additional production evidence is required.
3. Register the binary with `discover_v2_parity_tests()` and declare every GGUF
   that all of its `ProductionParity` cells can select with `MODEL_FILES`.
   Backend-specific additions use `CPU_MODEL_FILES`, `CUDA_MODEL_FILES`, or
   `ROCM_MODEL_FILES`, which prevents a focused backend slice from staging
   unrelated large models. Registration and runtime path resolution fail closed
   if these drift. If a specialized path asserts another PerfStats domain, pass
   `PRODUCTION_PERF_STATS_FILTER "forward_graph,<domain>"`.
4. Build the target so its generated CTest include is refreshed, then run the
   campaign unit test and `--list` coverage audit.
5. Stage the real model files and run every affected backend campaign. A new
   defect needs a focused regression in addition to the full campaign cell.

Do not create separate `PrefillParity`, `DecodeParity`, and
`SnapshotInfrastructure` model-loading tests for a new matrix. Their contracts
belong in the fused production cell. Prefix-cache, MTP, stochastic sampling,
dynamic-depth, and other specialized behavioral suites remain focused gates
when they exercise a different state machine; they do not replace mathematical
production parity. MTP registration must cover fixed depths 1, 2, and 3 plus
dynamic depth on CPU, CUDA, and ROCm. MoE MTP cells inherit the same explicit
Static/Dynamic/LLEP movement contract and routed-owner placement as their
ordinary inference cells.

## Local verification

Fast, device-free framework checks:

```bash
python3 tests/v2/unit/scripts/test_production_parity_campaigns.py
python3 -m pytest -q python/reference/tests/test_snapshot_metadata.py
ctest --test-dir build_v2_integration \
  -R '^V2_Unit_ProductionParityCampaigns$' --output-on-failure
```

Build parity targets with the repository's normal unrestricted concurrency:

```bash
cmake --build build_v2_integration --parallel \
  --target v2_integration_parity_qwen2_single_device
```

## Troubleshooting

- “reference pack is stale or unauthenticated” names the exact failed identity
  field; regeneration occurs once on rank zero and all ranks wait for its
  completion.
- A production model or accelerator prerequisite failure is intentional. Stage
  the configured GGUF and schedule the campaign on its declared backend.
- A homogeneous GPU segmentation failure means the live path violated its
  backend-native captured-generation contract; fix graph construction or
  capture identity.
- A generation certification failure is diagnosed first through
  `generation_certification_detail`. Do not accept a policy name without its
  native-parent or authenticated-ticket ledger proof.
- Numerical failures retain the layer/stage/decode CSVs and test log in the
  case-specific results directory.
- Use `LLAMINAR_LOG_LEVEL=DEBUG` for lifecycle diagnostics. Stage dumping is a
  focused diagnostic and is not production-path certification.

Parity tests require Python with PyTorch/Transformers/NumPy, `cnpy`, ZLIB, the
configured MPI runtime, and the relevant backend libraries. Integration builds
must have pipeline snapshots enabled.
