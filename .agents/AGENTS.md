# Llaminar V2 Project Development Guidelines

This document provides practical guidelines for working with the **Llaminar V2** LLM inference engine, including build processes, testing, debugging, and kernel / MPI / attention development best practices.

## Non-Negotiable Engineering Maxims

These rules summarize the architecture that Llaminar is converging on. They
take precedence over stale examples or older compatibility patterns elsewhere
in the repository. **Fallbacks are forbidden.** Every supported,
user-selectable mode is a first-class implementation with its own correctness
and economy gates; no mode may be entered automatically to hide a missing,
broken, or uneconomical implementation.

### Architecture and Ownership

1. **Implement the target state directly.** Do not add serial replay, host
   mirrors, eager execution, recapture, oversized timeouts, or capability
   advertisements to postpone a required production implementation. Finish the
   implementation or fail hard with a precise diagnostic.
2. **Give every live value and physical byte one authority.** GPU execution
   state is device-owned; CPU execution state is host-owned. Do not maintain an
   informal host shadow of device state or repair coherence by downloading and
   re-uploading values. `PhysicalMemoryAuthority` is the sole canonical
   admission, reservation, materialization, and release ledger for every CPU
   and GPU allocation in Llaminar. Estimators may only contribute typed BOM
   inputs to that authority; no subsystem may keep a parallel live ledger,
   independently subtract capacity, apply an anonymous reserve, or make an
   allocation decision from duplicated accounting arithmetic.
3. **Make invalid states unrepresentable.** Put lifecycle, ownership, ordering,
   and policy in typed interfaces, enums, builders, and RAII scopes. Do not rely
   on comments, caller discipline, raw state transitions, or source scans when
   an API can reject the invalid operation.
4. **Keep model graphs declarative.** Model graph files declare typed placement,
   sharding, replication, collective, and execution policies. Reusable graph
   builders and base machinery perform the wiring; model files must not become
   imperative orchestration scripts.
5. **Keep graphs per-device and symmetric.** Compute stages are
   participant-local and cross-device movement is an explicit graph collective.
   Never create nested multi-device subgraphs or hide rank coordination inside
   an ordinary compute stage.
6. **Use one public transfer/coherence authority.** Production callers use
   `TransferEngine`, arena contracts, and event-aware publication APIs. Direct
   `transitionTo()`, ad hoc `ensureOnDevice()`, and manual dirty-state mutation
   are test-only unless an explicitly documented infrastructure boundary owns
   them.
7. **Retire replaced machinery completely.** Remove obsolete launchers,
   entrypoints, generic suites, CLI names, capability checks, and dead code when
   the replacement is installed. Do not preserve a second path "just in case."

### GPU Execution and Memory

8. **GPU inference is device-resident end to end.** After request admission,
   planning, sampling, MTP verification, state publication, and loop decisions
   stay on device. The ordinary host boundary is one small terminal result. A
   GPU graph API without conditional nodes may additionally publish one fixed-
   size immutable scheduler ticket that names a retained captured transaction;
   it may not expose mutable model, sampler, verifier, KV, or response state.
   Explicit RAM/SSD KV-cache tiers and initial/final I/O are other intentional
   exceptions, not permission for host-owned execution state.
9. **Homogeneous GPU execution uses a complete captured generation policy.**
   Ordinary inference and a conditional-capable MTP backend use one complete
   captured parent. HIP MTP uses retained complete transaction graphs selected
   by an authenticated scheduler ticket because HIP graphs have no conditional
   nodes; the device controller remains the sole decision/state authority and
   the host only submits the named branch. Segmented or eager homogeneous
   execution is a hard error. Segmentation is permitted only at an explicitly
   declared heterogeneous device/collective boundary that cannot be represented
   by one native graph.
10. **Every GPU operation has an exact non-null stream.** Kernel launches,
    libraries, copies, memset, events, and publication APIs must reject null or
    default streams. The producer publishes its exact stream/event and the
    consumer waits on that event; never guess or substitute an unrelated stream.
11. **Use events for ordering, never blocking synchronization in the hot path.**
    No per-stage, per-leaf, per-token, or per-transaction stream/device syncs.
    Keep the producer/consumer DAG visible in one timeline or fluent graph API.
12. **No hot-path allocation or transfer.** CUDA/HIP allocation, free, H2D, D2H,
    host callbacks, temporary workspace construction, and buffer rebinding are
    forbidden during captured execution. Bind arena- or instance-owned
    persistent buffers before capture and reuse them.
13. **Capture identity is complete.** Every embedded pointer, device, workspace
    generation, geometry, policy scalar, and launch mode belongs in graph-cache
    identity. Stale or partial identity is a fatal error, never a reason to
    replay a vaguely compatible graph.
14. **Request reset resets data, not topology.** Reuse proven captured graphs
    across requests and publish new device state through explicit ordered
    lifecycle operations. Do not recapture to avoid proving reset/replay
    equivalence.
15. **Share memory according to concurrency.** Graphs that cannot execute
    concurrently may share arena regions sized for the largest participant;
    concurrently runnable graphs require exclusive ownership. Validate every
    arena plan with a buffer/weight capacity BOM.
16. **Deterministic reductions have a fixed arithmetic order.** Atomics and
    partition-dependent reductions are forbidden in batch-invariant paths
    unless byte determinism is concretely proven. Prefer fixed partitions and
    fixed reduction trees that retain performance without changing arithmetic.

### Correctness and Completeness

17. **Grouped decode means serial-row byte equivalence.** MTP verifier outputs
    must be bitwise/byte identical to serial decode, not merely close. Row replay
    may be a diagnostic oracle but must never execute in production.
18. **Prove the real optimized path.** Correctness tests must exercise production
    graph capture, collectives, stochastic sampling, dynamic depth, prefix
    cache, TurboQuant, and optimized kernels. Test-only deterministic modes or
    disabled optimizations do not certify production.
19. **Coverage is total, not anecdotal.** Sweep every supported tensor
    format/codebook and backend, all grouped operations, dense and MoE models,
    MTP and non-MTP lanes, and representative geometries. Cover M from 1 through
    the supported range and dispatch totality beyond measured points; likewise
    cover N/K geometry, codebook, CPU ISA, and every positive thread count.
20. **Backend symmetry is the default.** CPU, CUDA, and ROCm receive equivalent
    features, tests, diagnostics, and economical kernels unless hardware makes
    a difference explicit. A working implementation on one backend exposes a
    gap on the others; close it rather than silently advertising less support.
21. **Correctness and economy are simultaneous invariants.** A slow serial-style
    grouped kernel is not a finished implementation. First establish exact
    arithmetic, then retain that exactness while making the grouped path
    genuinely economical.
22. **Every discovered defect gets a focused regression.** Reproduce flaky
    failures in a loop (up to 20 iterations when appropriate), reduce them to a
    focused test, then fold the invariant into the all-format/backend sweep and
    canonical integration gate.

### Performance and Observability

23. **Measure production binaries and production workloads.** Canonical server
    E2E and benchmarks use `Release`; `Integration` remains `-O3` with symbols
    and snapshots. Compare end-to-end behavior with relevant external baselines,
    not isolated kernel wins alone.
24. **Profile every new CUDA/HIP kernel.** Use `ncu`/`nsys` or
    `rocprof`/ISA inspection to report occupancy, registers/VGPRs, spills,
    throughput, memory efficiency, and launch geometry. Tune until occupancy
    and throughput are economical; zero spills is the default expectation.
25. **Profile CPU kernels with the same rigor.** Use `perf`, IPC/cache evidence,
    physical-core scaling, AVX2/AVX-512 runtime dispatch, and NUMA-aware tests.
    Work must scale sensibly from one thread through the physical cores per
    socket; hyperthreads are not the default worker budget.
26. **PerfStats is the production truth.** Add counters that prove the intended
    path actually ran, including full graph capture, MTP depth, routing mode,
    prefix-cache tier, collectives, transfers, and segmentation. E2E tests assert
    those counters, including no GPU D2H except the terminal result.
27. **Keep profiling evidence isolated.** Profile one exact kernel/ISA/shape/M
    candidate per profiler launch and attach that evidence to its timing record.
    Never contaminate candidate features with unrelated kernels or perturb the
    canonical timing run with profiler overhead.
28. **Optimize the feedback loop too.** Long preprocessing, CV, parsing,
    certification, and corpus adaptation work must use physical-core parallelism
    or efficient accelerator execution. Persistent buffers, graph capture, and
    resumable/additive corpora apply to tooling as well as inference.

### Failure, Testing, and Hygiene

29. **Fail fast and fatally.** Missing events, stale generations, null streams,
    unsupported topology, failed publication, segmented homogeneous replay, or
    incomplete state are fatal. Do not log a warning and limp onward, retry a
    different path, or disguise a deadlock with a huge timeout; collectives use
    the standard 30-second timeout unless a documented protocol requires less.
30. **Unit tests stay fast and device-free.** GPU work belongs in explicit CUDA
    and ROCm integration suites. Unit tests should finish in seconds, use shared
    tensor/arena fixtures, and test interfaces/state machines without loading
    models or occupying accelerators.
31. **Test state machines adversarially.** Transfer, KV-cache, prefix-cache,
    graph-reset, MTP advancement, promotion/demotion, eviction, and multi-stream
    publication suites should stress interleavings and lifecycle boundaries to
    destruction, not only verify a happy path.
32. **Use source sanitizers as architecture tests.** Forbid default streams,
    blocking syncs, direct coherence transitions, dynamic GPU allocation,
    host callbacks, row-replay production calls, and other structurally banned
    patterns. Keep allowlists narrow, named, and limited to true infrastructure.
33. **Logs are proportional to rarity.** Fatal state is `ERROR`, actionable
    anomalies are `WARN`, lifecycle summaries are `INFO`, diagnostics are
    `DEBUG`, and per-token/per-buffer/coherence traffic is `TRACE`. Never let
    logging create hot-path transfers or synchronization.
34. **Build and test with the repository's concurrency.** Use Ninja, ccache, and
    unrestricted `--parallel`; split very large generated kernel families into
    persistent translation-unit shards when that improves compile throughput.
    Do not cap ordinary builds/tests to an arbitrary small worker count.

### Code and Documentation

35. **Prefer typed, explicit C++ interfaces.** Use scoped enums, designated
    initializers, concepts, structured parsers, and fluent builders where they
    clarify policy. Avoid boolean soups, stringly typed wiring, duplicated mode
    names, and ad hoc pointer conventions.
36. **Document ownership and why.** Every touched source file needs a substantive
    Doxygen file header; public/protected and non-trivial private methods need
    complete Doxygen; inline comments explain lifecycle, arithmetic order,
    event edges, invariants, and performance reasoning so a junior developer can
    follow the implementation.
37. **Keep workflow docs timeless.** Skills describe stable procedures,
    commands, gates, and architecture; dashboards/project plans record current
    progress, measurements, failures, and next steps. Do not turn `SKILL.md` or
    `AGENTS.md` into a task diary.
38. **Keep names and docs synchronized.** When modes or CLI options change,
    update code, tests, help, `README.md`, `AGENTS.md`, project docs, and source
    sanitizers together. Fully disambiguate execution modes rather than carrying
    historical aliases indefinitely.
39. **Keep changes scoped and the tree clean.** Preserve unrelated user edits,
    avoid incidental rewrites, remove generated debris, and never commit local
    result directories. Checkpoint meaningful green slices; publish large
    certified corpora through Git LFS in `Llaminar/corpora`, the optional
    top-level `corpora/` submodule, never as payloads in this source repository.

**Architecture Note (V2)**: The active architecture is **Llaminar V2** in `src/v2/`, a kernel-centric design with **DeviceGraphOrchestrator** (single-device) and **RankOrchestrator** (multi-device TP/PP) as execution paths.

- For a **high-level architecture map** of tensors, kernels, attention, MPI orchestration, and graph execution, see:
    - `.github/instructions/llaminar-architecture-v2.instructions.md`
- For additional V2-specific implementation details, see:
    - `.github/instructions/llaminar-v2-architecture.instructions.md`

## Project Agent Skills

Project-local skills under `.agents/` are the canonical specialized workflows.
Framework discovery paths are symlinks to these files so their contents remain
single-source.

- `.agents/cpu-tuning/SKILL.md`: CPU profiling, Linux `perf`, ISA analysis,
  physical-core/NUMA/OpenMP scaling, and AVX2/AVX-512 tuning.
- `.agents/cuda-tuning/SKILL.md`: CUDA profiling, Nsight diagnostics, CUDA
  kernel harnesses, and parity-preserving tuning.
- `.agents/rocm-tuning/SKILL.md`: HIP/ROCm profiling, `rocprof`, LLVM ISA
  analysis, and ROCm kernel tuning.
- `.agents/mtp-tuning/SKILL.md`: speculative decode, prefix-cache interaction,
  verifier parity, grouped MTP economics, and depth-controller work.
- `.agents/model-parity-testing/SKILL.md`: real-weight production campaign
  execution, PyTorch/Hugging Face reference packs, CSV-led diagnosis, matrix
  extension, and the aggregate correctness/economy gate.
- `.agents/nativevnni-gemm-tuning/SKILL.md`: cross-backend NativeVNNI
  GEMV/GEMM candidate tuning, evidence collection, dispatch installation, and
  corpus certification.
- `.agents/llama-cpp-comparison/SKILL.md`: matched Release comparisons against
  llama.cpp, exact bucket-aligned prompts, paired phase/kernel attribution, and
  overlap-aware performance evidence.

Do not copy skill bodies into `.codex/`, `.claude/`, or `.github/`; add or
update symlinks. Backend profiler commands, metric interpretation, and tuning
workflows belong in the applicable skill, not in this file.

## Sources of Truth

Prefer the narrowest code-owned source instead of copying volatile tables into
documentation.

| Concern | Canonical source |
|---|---|
| Architecture and ownership | `.github/instructions/llaminar-v2-architecture.instructions.md` |
| Build types, options, and targets | `src/v2/CMakeLists.txt` |
| CLI flags and help | `src/v2/config/CliSpec.*`, `OrchestrationConfigParser.*` |
| CLI conflict rules | `src/v2/config/ConfigValidator.*` |
| Runtime environment variables | `src/v2/utils/DebugEnv.h` |
| Test names, labels, and registration | `tests/v2/CMakeLists.txt` |
| Parity workflow | `.agents/model-parity-testing/SKILL.md`, `tests/v2/integration/parity/README.md`, and CMake registration |
| Backend tuning procedure | `.agents/*-tuning/SKILL.md` |

Files under `docs/v2/projects/` are dated plans, investigations, and handoffs.
They provide historical context, not the current architecture, unless a live
source explicitly points to one.

### Corpus Repository Boundary

All published corpus families belong in `https://github.com/Llaminar/corpora`,
pinned by the source repository's `corpora/` gitlink. Keep collection work and
local results ignored. Installed policy source and small device-free test
fixtures remain here; corpus payloads do not.

Ordinary source checkouts, builds, container builds, and Unit/preflight gates
must not initialize this optional submodule or download its LFS objects. When
corpus work is requested, initialize metadata with
`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init -- corpora`, then use
`git -C corpora lfs pull --include '<family>/<selected-generation>/**' --exclude ''`.
Publish the corpus commit and LFS objects in the data repository first, then
commit the updated source gitlink. Never restore the old in-source corpus root.

## Architecture Orientation

The active engine is `src/v2/`. Its production execution stack is:

```text
OrchestrationRunner
  -> GlobalOrchestrator       cross-rank named-domain PP/global TP
  -> RankOrchestrator         rank-local TP or local PP
  -> DeviceGraphOrchestrator  one participant/device
  -> ForwardExecutionEngine
  -> ComputeGraph + DeviceGraphExecutor
  -> compute stages -> typed kernel interfaces -> CPU/CUDA/ROCm kernels
```

Graphs are participant-local and model-specific wiring is registered through
`GraphBuilderRegistry` and `SchemaFactoryRegistry`. `BufferArena` owns graph
activation/workspace storage, `PreparedWeightStore` owns stable prepared weight
handles, and `TransferEngine` is the public movement/coherence authority. See
the architecture document above before changing orchestration or graph
ownership.

## Build

Use an out-of-tree Ninja build and unrestricted build parallelism. The
container image and workspace environment pin the same Ninja release. Resolve
that active executable once when configuring, record it in
`CMAKE_MAKE_PROGRAM`, and invoke builds through `cmake --build`. Never
alternate Ninja executables on one build tree: incompatible command-log hashes
can make every object appear dirty. If an existing tree names another
executable, reconfigure it once with the active devcontainer path before
building.

CUDA builds use the canonical capture-reentry NCCL dependency installed by
`scripts/docker/install-nccl.sh` in both development and release images.
Outside those images, run `sudo bash scripts/docker/install-nccl.sh` before
configuring. The runtime deliberately loads its distinct SONAME, not the
distribution NCCL package, because retained parent recording requires the
corrected stream-membership lifecycle. Do not substitute a system library to
make dependency discovery pass.

RCCL runtime loading uses the exact `RCCL_LIBRARY` selected by CMake, without a
second runtime search. Select the compatible library explicitly when configuring
a local build. Container builders and Release images install the same source-
built RCCL at the same stable path; moving a checkout must not change the DSO.

```bash
LLAMINAR_NINJA_BIN="$(command -v ninja)"

# Debug: assertions/snapshots, no optimization
cmake -B build_v2 -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${LLAMINAR_NINJA_BIN}"
cmake --build build_v2 --parallel

# Release: production and performance measurements
cmake -B build_v2_release -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${LLAMINAR_NINJA_BIN}"
cmake --build build_v2_release --parallel

# Integration: -O3, symbols, assertions, and snapshots
cmake -B build_v2_integration -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Integration \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${LLAMINAR_NINJA_BIN}"
cmake --build build_v2_integration --parallel
```

The supported build types are `Debug`, `Release`, and `Integration`. CUDA and
ROCm are both enabled by default. Disable an unavailable toolchain explicitly,
for example:

```bash
cmake -B build_v2_cpu -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Integration -DHAVE_CUDA=OFF -DHAVE_ROCM=OFF \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${LLAMINAR_NINJA_BIN}"
```

`llaminar2_core` is the core implementation target and `llaminar2` is the
application. Integration and Debug enable snapshots; Release snapshots are
opt-in through the current CMake option. Do not maintain a second option table
here—inspect the CMake file when adding or changing a feature.

## Run and Inspect Configuration

Use subcommands and consult command-specific help before assuming a flag still
exists.

```bash
./build_v2_release/llaminar2 --help
./build_v2_release/llaminar2 oneshot --help
./build_v2_release/llaminar2 benchmark --help

./build_v2_release/llaminar2 oneshot \
  -m models/model.gguf -d cuda:0 -p "Hello" -n 50

./build_v2_release/llaminar2 oneshot \
  -m models/model.gguf --dry-run --explain-placement
```

The executable normally bootstraps MPI and installs rank, NUMA, OpenMP, and
BLAS placement. Do not use `--no-mpi-bootstrap` for ordinary execution. It is
only for an attaching debugger or the profiler workflow described by the
backend tuning skills.

Device selection forms and their mutual exclusions evolve with the topology
planner. Use `--help`, `ConfigValidator`, `--validate-only`, `--dry-run`, and
`--explain-placement` rather than relying on a copied flag matrix.

Production model activations currently support FP32 only. Other activation
precision requests fail as unimplemented. KV-cache precision and expert weight
formats are independent settings; their support does not imply support for
another model activation dtype.

MTP hardware defaults are selected once by `ExecutionPlanBuilder` from the
complete continuation domain and canonical device inventory. Their numeric
source is `execution/config/MTPDepthDefaults.h`. Preserve automatic versus
explicit request intent through CLI/YAML and MPI; never infer an override by
comparing its value with an old default. Other expert tiers cannot select the
continuation policy. Request admission resolves the profile into the existing
device-controller ABI without changing MTP enablement, mode, or graph capacity.

## Benchmarking

Use a Release binary and a fixed model, device, prompt bytes, decode length,
sampling policy, and runtime configuration for comparisons.

```bash
./build_v2_release/llaminar2 benchmark \
  -m models/model.gguf -d cuda:0 \
  --prompt "A fixed prompt used for every comparison."

./build_v2_release/llaminar2 benchmark \
  -m models/model.gguf -d cpu:0 \
  --prompt-file benchmarks/prompts/fixed.txt
```

Benchmark mode owns warmup, repeated measurement, cache clearing, prompt
identity, and result reporting. Read `BenchmarkMode`, `BenchmarkRunner`, and
`DebugEnv.h` for current defaults. Use PerfStats to prove production path
selection; use the relevant backend skill for kernel-level attribution and
profiling.

Never compare a Debug/Integration number with a Release number or mix profiler
overhead into the canonical timing sample.

## Testing

Tests are registered in `tests/v2/CMakeLists.txt`. Discover the current names
before choosing a regex; do not copy large generated test-name lists into this
file.

```bash
# List registered tests
ctest --test-dir build_v2_integration -N

# Run the complete configured suite
ctest --test-dir build_v2_integration --output-on-failure --parallel

# Complete Unit gate: build its CMake-owned executable inventory, then run all
# registered script and binary tests.
cmake --build build_v2_integration --parallel --target v2_unit_gate
ctest --test-dir build_v2_integration -R '^V2_Unit_' \
  --output-on-failure --parallel

# Common integration family
ctest --test-dir build_v2_integration -R '^V2_Integration_Parity_' \
  --output-on-failure --parallel

# Run one exact test after discovering its name
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_StageDumpIntegrity$' --output-on-failure
```

The registered pre-commit hook builds only `v2_unit_gate` and
`v2_production_parity_preflight_gate`, then runs the full Unit namespace and
`ProductionParityPreflight` label on every branch. It runs no model campaigns,
E2E, broader integration selections, or benchmarks. Register the tracked hooks
with `git config --local core.hooksPath .githooks`; see `.githooks/README.md`.
The full shippable-image gate is `scripts/ci/run_production_pipeline.py`:
independent AVX512 and AVX2 full CPU/CUDA/ROCm Docker builds, each running the
canonical Unit/preflight/model-parity driver and all E2E-tagged cells. Both full
E2E server suites must pass before either ISA's benchmarks run. Only complete
per-image evidence may attach a certificate; official publication requires
both images. Explicit one-off diagnostic benchmarks cannot certify images.
Official CI commits both compact ISA result JSONs and one combined upward-only
high-water proposal; local
runs never commit or publish implicitly. See `docs/production-ci.md`. Do not
add a second benchmark topology/model list or reintroduce this pipeline into
pre-commit.

Naming conventions:

- `V2_Unit_*` must remain fast and device-free.
- `V2_Integration_*` may require models, MPI, CUDA, or ROCm.
- `V2_Perf_*` and standalone performance binaries measure economy, not just
  correctness.
- Source-policy tests are architecture gates; do not weaken their allowlists to
  make a violation disappear.

Use `tests/v2/utils/TestTensorFactory.h` for shared tensor fixtures. A new
defect needs a focused regression and, when applicable, coverage in the broader
format/backend/geometry sweep.

## Diagnostics and Parity

Runtime diagnostics are centralized in `src/v2/utils/DebugEnv.h`. Production
code reads `debugEnv()` rather than scattering `getenv()` calls. When adding or
renaming a setting, update its typed configuration, parser, tests, and any
command help together.

Stage dump and output inspection are diagnostic-only because they may copy
device data to the host or synchronize execution. Their implementation and
gate are:

- `src/v2/execution/debug/StageDumper.h`
- `src/v2/execution/debug/AsyncStageDumper.h`
- `tests/v2/integration/execution/debug/Test__StageDumpIntegrity.cpp`

For model parity, use `.agents/model-parity-testing/SKILL.md`, then read
`tests/v2/integration/parity/README.md` and discover the exact registered
production campaigns. The aggregate campaign system replaces the historical
hand-picked PyTorch parity baseline: it keeps reference generation, live-path
execution, every checkpoint comparison, CSV evidence, and the shared economy
target in one registered matrix. Before model admission, a build must pass the
complete CMake-owned `V2_Unit_*` suite and the model-free
`ProductionParityPreflight` integration label. Unchanged local diagnostic runs
reuse their canonical receipt across cells; refresh it after a build or test
inventory change, not for every cell. The driver validates receipt freshness
and completeness. Device-free regressions join the
Unit phase automatically; add a focused Integration regression to preflight
when a parity defect establishes a reusable backend lifecycle, graph,
stream/event, collective, or movement invariant. Local reports and result
directories are generated debris and must not be committed.

HTTP needle and long-context certification also derives from the typed parity
definitions: `ModelParityDefinition::e2e_certifiable` selects exact existing
cells and their context profiles. Build `v2_model_parity_matrices`, then use
`scripts/ci/run_model_parity_e2e.py --list` to inspect eligibility. Its non-list
run uses the Release server and the mature HTTP harness; scripts and container
jobs must never maintain a second model/topology matrix. Numerical parity and
HTTP behavioral certificates are complementary gates, not substitutes.

For a debugger attached directly to `llaminar2`, pass
`--no-mpi-bootstrap`; otherwise it may attach to the MPI wrapper. Record any
manual affinity needed to reproduce a performance-sensitive problem, because
that direct launch does not reproduce the normal bootstrap environment.

## Project-Specific Implementation Conventions

These are mechanical repository conventions that complement the engineering
maxims above:

- Use `ASSERT`, `ASSERT_MSG`, `ASSERT_NOT_NULL`, and related helpers from
  `src/v2/utils/Assertions.h` for internal invariants. Return structured errors
  for recoverable user input.
- Use `VERIFY_TENSOR*` helpers from `src/v2/tensors/TensorVerification.h` when
  validating numerical tensors; do not add ad hoc NaN scans to hot paths.
- Use `OMP_WORKSHARE_REGION` and its variants from
  `src/v2/utils/OpenMPUtils.h` for kernels that must work both inside and
  outside an existing OpenMP team.
- Use the typed tensor APIs and `typed_data()` accessors. Do not infer element
  layout from an untyped pointer when the tensor type already encodes it.
- Register kernel implementations through `KernelFactory`; stages depend on
  operation interfaces, not concrete backend classes.
- GPU code must keep `V2_Unit_Static_NoDefaultStreamInGPUCode` green. A null
  stream is not an alias for the correct stream.
- Use `libfort` for substantial aligned terminal tables already following the
  repository's table style; ordinary logs should remain simple and structured.
- Keep logging levels proportional to rarity and avoid formatting or data
  materialization that changes hot-path behavior.

## Documentation Hygiene

Keep stable architecture and workflow in `AGENTS.md`, the architecture
instruction, and project skills. Put dated measurements, failures, decisions,
and next actions in a project plan or handoff under `docs/v2/projects/YYYY-MM/`.

When changing CLI names, execution modes, environment variables, or registered
tests, update the code-owned help/registration first and keep prose references
short. Do not add exhaustive tables that must be manually synchronized with
source.

Before handing off a change:

- preserve unrelated worktree changes;
- remove local reports, dumps, profiler captures, and parity-result folders;
- inspect the scoped diff;
- make sure every linked path exists;
- keep architecture claims aligned with installed production paths, not plans
  or compatibility aliases.
