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
    canonical integration gate. Every production defect must also have its
    focused regression registered explicitly in the `ProductionTestPreflight`
    suite; broad Unit or backend coverage does not replace that preflight entry.

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
- `.agents/llaminar-testing/SKILL.md`: Unit and production-preflight gates,
  generation regressions, real-weight HF/CSV diagnostics, canonical matrix
  extension, HTTP/remote-MPI E2E, and image/benchmark certification.
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
| Lossless config documents and startup publication | `src/v2/config/OrchestrationConfigDocument.*`, `OrchestrationStartupPolicy.*` |
| Auto selection, candidates, and measured cost estimates | `src/v2/planning/AutomaticPlanningStartup.*`, `AutomaticOrchestrationCandidates.*`, `PlanningRequestCostModel.*` |
| Domain/tier grammar and routed-expert policy | `src/v2/config/ExecutionDomainDefinition.*`, `ExpertTierDefinition.h`, `src/v2/execution/config/RoutedExpertPolicy.h` |
| Durable movement defaults and runtime policy | `src/v2/execution/config/RuntimeConfig.h`, `src/v2/execution/moe/DeviceMoERebalancePolicyShared.h` |
| Runtime environment variables | `src/v2/utils/DebugEnv.h` |
| Test names, labels, and registration | `tests/v2/CMakeLists.txt` |
| Testing workflow | `.agents/llaminar-testing/SKILL.md`, `tests/v2/integration/parity/README.md`, `docs/production-ci.md`, and CMake registration |
| Public docs and release reports | `docs/public/`, `mkdocs.public.yml`, `scripts/ci/build_public_docs.py`; published GitHub release attachments own the evidence |
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

ROCm graph execution also requires the canonical HIP runtime built by
`scripts/docker/install-hip-graph-runtime.sh`. Development and release builders
install its race-free graph-identity repair; Release images copy that exact DSO
from the builder. Outside those images, install the matching `rocm-llvm-dev`
package and run this installer before ROCm gates. Do not disable packet capture
or serialize independent graph builders to hide an unfixed system HIP runtime.
`V2_Integration_HIPConcurrentGraphIdentity` is the focused preflight proof that
every operation survives concurrent construction and replay.

Full-backend binaries also run on CPU-only cluster members. Keep CUDA Driver
API binding in `CUDADriverApi`, prepared before CUDA graph recording; do not
restore a public `CUDA::cuda_driver` dependency or inject toolkit stubs into
test discovery. Native CUDA execution still requires the real host driver.

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

### Start with auto, then measure before overriding

Use the public `plan`, `serve`, `oneshot`, and `benchmark` subcommands. Start
with the model, required context, and only the user's actual constraints.
Automatic planning is the default when no placement is declared; `--auto` is
optional. Do not preselect a device count, strategy, collective, precision,
expert quota, movement policy, or graph bucket to reproduce an old benchmark.
First inspect and measure what the installed production defaults select, then
make explicit, separately measured overrides if needed.

```bash
./build_v2_release/llaminar2 plan --help
./build_v2_release/llaminar2 serve --help

# Direct auto serving; only ROCm compute is permitted in this example.
./build_v2_release/llaminar2 serve \
  -m models/model.gguf --only-backends rocm -c 32768

# Inspect and save the selection for the same inference policy.
./build_v2_release/llaminar2 plan \
  -m models/model.gguf --only-backends rocm -c 32768 \
  --output /tmp/llaminar-plan.json

# Apply exactly that selection, without rerunning auto search.
./build_v2_release/llaminar2 serve --config /tmp/llaminar-plan.json
```

Substitute an existing GGUF and a context that admission can fit. Omit
`--only-backends` to consider all discovered compute backends. Supply the same
intended MTP, KV-cache, and prefix-cache settings to `plan` as to `serve`;
changing capacity-affecting settings afterward requires renewed admission.

Auto reads the GGUF metadata, gathers the MPI cluster inventory, admits
candidates through `PhysicalMemoryAuthority`, and ranks bounded estimates of
prefill and generation cost. These are predictions, not whole-model benchmarks
or a guarantee that every visible device will be selected. A fresh search may
select a different subset; use a saved apply document when comparing one fixed
topology across runs.

Use constraints deliberately:

- `--only-backends cpu,cuda,rocm` restricts compute, not host control/storage.
- `--auto-device-counts cuda=2,rocm=2` requires exact physical compute counts
  for the named backends while leaving endpoint identities, rank ownership,
  layer splits and tier choices to auto. CPU counts mean NUMA endpoints, not
  threads or MPI ranks. Combine with `--only-backends` to exclude other
  backends; unlisted counts are unconstrained. Use this for an actual topology
  requirement, not to reproduce an old default-performance score.
- `--only-strategies single,tp,pp,expert-overlay` restricts candidate families.
  Do not add it merely because a model is MoE. A homogeneous single-domain MoE
  candidate is currently labeled `tp` even though its runtime uses ExpertOverlay;
  an `expert-overlay`-only filter excludes that candidate. Inspect the resolved
  domains and routed-compute policy, not just the strategy label.
- `--prefer-backend` and `--prefer-strategy` are soft tie-breaking preferences;
  they do not override admission or hard constraints.
- `--plan-workload <prefill,generation>` supplies an optional expected request
  horizon for ranking. Both counts must be positive and fit the context; this
  does not limit inference output. Omit it for the initial default baseline.
- `--auto-hosts all` requires compute on every discovered physical host. The
  default permits a host subset; neither form means "use every GPU/rank."

Explicit devices/domains/tiers or an applied plan select apply mode instead.
`--planning-mode apply` may make that intent explicit; do not combine authored
placement with `--auto` or auto-search filters/hints. `plan` requires automatic
intent; inspect authored placement with `serve --dry-run --explain-placement`.
Use `--validate-only` for parsing/configuration checks, not as proof of hardware
fit, graph support, or successful inference. Current help and `ConfigValidator`
remain authoritative; accepted syntax alone is not a capability certificate.

### Execution modes and explicit placement

| Mode | What is distributed | Explicit surface when required |
|---|---|---|
| Single device | The complete model executes on one compute endpoint, without inter-device collectives. | `--device cuda:0`, `rocm:0`, or `cpu:0` |
| Dense TP | Participants cooperate on each layer using tensor shards and collectives. | `--tp-devices` with `--tp-scope`, or a named TP domain |
| Dense PP | Consecutive main-layer intervals execute in ordered domains; a domain may itself use TP. | `--define-domain` plus `--pp-stage` |
| ExpertOverlay | Routed experts of the same layer are placed across one or more domains/tiers; results return to the continuation domain. This is not a layer pipeline. | `--expert-tier` (preferred compact form) |

```bash
# Single GPU. Use rocm:0 for ROCm or cpu:0 for one CPU NUMA endpoint.
./build_v2_release/llaminar2 serve -m models/model.gguf --device cuda:0

# Dense TP within one rank; the device list supplies the degree.
./build_v2_release/llaminar2 serve -m models/dense.gguf \
  --tp-scope rank_local --tp-devices rocm:0,rocm:1

# Dense PP example ONLY for a model with 64 main transformer layers.
# Change the contiguous, inclusive intervals to the actual GGUF geometry.
./build_v2_release/llaminar2 serve -m models/dense-64-layer.gguf \
  --define-domain 'early=rocm:0;scope=rank_local' \
  --define-domain 'late=rocm:1;scope=rank_local' \
  --pp-stage '0=early:0-31' --pp-stage '1=late:32-63'
```

CPU `--device cpu` is special shorthand for all local NUMA endpoints, not one
device; `cpu:N` selects one. Do not combine `--device` with `--tp-devices` or
named domains. An optional `-tp N` must match the device-list length. Named
domains specify their own width, so do not append an unrelated `-tp` degree.

`rank_local` means multiple devices in one MPI process; `node_local` means
participants across ranks on one physical host; `global` spans the selected
MPI ranks, potentially across hosts. These scopes are not interchangeable and
do not describe expert compute distribution. A native GPU TP group uses one
vendor's collective; do not treat CUDA and ROCm as one NCCL/RCCL group. Put
distinct hardware groups in separate domains with an admitted boundary.
For dense PP, stage intervals must cover the main model without gaps/overlaps;
trailing MTP sidecar blocks are not extra pipeline layers. Automatic dense PP
is not a substitute for an MoE expert topology.

### ExpertOverlay tiers, capacity, and compute policy

All multi-device routed MoE uses one ExpertOverlay placement authority,
including a single-tier homogeneous deployment. Start with auto; use compact
tiers only when the user needs an authored placement:

```bash
# One homogeneous tier: no cross-tier promotion, but within-tier rebalance.
./build_v2_release/llaminar2 serve -m models/moe.gguf \
  --expert-tier 'accelerator=rocm:0,rocm:1;priority=0'

# GPU continuation and a two-NUMA-endpoint CPU expert tier.
./build_v2_release/llaminar2 serve -m models/moe.gguf \
  --expert-tier 'accelerator=cuda:0,cuda:1;priority=0' \
  --expert-tier 'capacity=cpu:0,cpu:1;priority=10'
```

Use observed device/NUMA IDs. Bare `cpu` is not a domain/tier participant
address, even though `--device cpu` is valid. Tier names are opaque labels,
never hard-coded "hot/warm/cold" roles. Integer priorities must be distinct:
smaller numbers are preferred and select the default continuation; the greatest
priority is the derived final-coverage tier. Do not write `fallback=true`.
That internal coverage role is neither a CPU requirement nor an error fallback.

The continuation owns the main inference path and receives the combined routed
output; additional expert tiers do not become independent samplers or MTP
controllers. Base-model and shared-expert domains default from continuation.
Scope, ownership, collective and physical expert capacity resolve through the
shared normalization/inventory and `PhysicalMemoryAuthority`, not tier names.

Omit capacity fields (or use `memory-mb=auto`) for automatic admission including
weights, KV/recurrent state, workspaces, prefix storage and movement buffers.
For a deliberate restriction, add `;memory-mb=N` or
`;max-experts-per-layer=N` to a tier. Do not hard-code expert counts to fill
VRAM or subtract another private reserve. Resolved live per-layer quotas are
authoritative: migration changes which experts occupy those slots, not a
promise that every lower-priority tier eventually drains.

Keep these policy axes separate:

- `--moe-routed-expert-compute apportioned` (default) assigns whole experts to
  participants. `replicated` places complete experts on every participant.
  `tensor-sharded` means splitting each expert's tensors, not apportionment;
  the standard Qwen3.5 MoE path currently rejects that global option as
  unimplemented. Do not infer support from its presence in help.
- `--moe-continuation-dense-policy` changes dense/shared work, not routed expert
  placement. Prefer its resolved automatic policy. Explicit choices include
  `replicated`, `tensor-parallel`, `tensor-parallel-decode-mirrored-embedding`,
  and `prefill-tensor-parallel-decode-replicated`; replication has a memory cost.
- `--moe-routed-expert-owner-order ordinal|random` chooses initial owner ordering;
  it does not enable or disable later movement. `--moe-hot-expert-cache` controls
  optional extra expert replicas, not the number of uniquely owned experts.

Advanced authored roles use `--moe-routed-expert-continuation-domain`,
`--moe-routed-expert-base-model-domain`, and `--moe-routed-expert-shared-domain`.
The expanded forms split the same intent into
`--moe-routed-expert-domain 'name=devices;...'` and
`--moe-routed-expert-tier 'tier@name;priority=N;...'`. Unlike compact tiers,
these require explicit `--moe-routed-expert-placement single-domain` or
`tiered-overlay` to enable placement. Domain options such as `scope`, `owner`,
`ranks`, `backend`, and `routed_compute` override those
particular axes; do not specify them all unless the topology actually requires
it. A routed domain accepts single/rank-local/node-local scope, not a global
TP domain; cross-host overlay connects distinct hardware domains.

### Dynamic residency maintenance and overrides

`--moe-residency-maintenance dynamic` is the default, with default residency
`rebalanced`. The sole overlay authority uses routed demand and migration
economics for cross-tier promotion/demotion and within-tier skew reduction.
A single tier still supports the second axis. Dynamic does not guarantee a
move every window or a speedup over a short request; validate completed moves,
placement epochs and amortized inference time rather than assuming them.

For a new placement without an explicit residency policy,
`--moe-residency-maintenance off` defaults residency to `static-by-id`;
`observe` records demand without publishing physical moves.
`--moe-routed-expert-residency` can explicitly override placement policy, but
does not by itself select the maintenance mode. An applied plan retains its
authored residency policy. Prefer changing just the maintenance flag for a
fixed-placement Static/Dynamic A/B rather than supplying redundant flags.

| Tuning purpose | Relevant override |
|---|---|
| Histogram cadence and adaptive growth | `--moe-residency-maintenance-window`, `--moe-residency-maintenance-max-window`, `--moe-residency-maintenance-window-growth` |
| Lifetime over which a move must repay its cost | `--moe-migration-payoff-horizon-tokens` (distinct from the histogram window) |
| Preallocated movement capacity | `--moe-migration-transfer-slots` (more slots also consume admitted memory) |
| Background GPU submission concurrency | `--moe-migration-execution-streams` (positive, no greater than slot count) |
| Active cycles admitted in a wave | `--moe-migration-cycles-per-wave` (positive, no greater than slot count) |
| Skew and benefit thresholds | `--moe-dynamic-imbalance-threshold-permille`, `--moe-dynamic-min-improvement-permille`; inspect help for device-specific economy controls |

Numeric defaults live in `RuntimeConfig.h`, `DeviceMoERebalancePolicyShared.h`
and CLI help, not a second tuning table here. Keep initial baselines at defaults;
then vary one relevant policy, include enough tokens to observe its horizon,
and compare correctness, completed movement, memory footprint and unprofiled
throughput. Do not disable maintenance silently to report a faster default.

Current-batch least-loaded assignment (LLEP) is separate from durable movement.
Domain fields `routed_prefill_assignment` / `routed_decode_assignment` select
`static-owner` or `least-loaded-resident` where supported; grouped MTP verifier
rows are decode work, not ordinary prefill. There is no `llep` value for
`--moe-residency-maintenance`. See `RoutedExpertPolicy.h` and domain validation
for supported combinations rather than treating these policies as synonyms.

### Cluster launch and shared runtime policy

The executable normally bootstraps MPI and installs rank, NUMA, OpenMP, and
BLAS placement. Do not use `--no-mpi-bootstrap` for ordinary execution; it is
only for a debugger or the profiler workflow described by the backend skills.
Use the same frontend for cluster-aware selection permitting ROCm and CPU
compute, then inspect the selected continuation and remote participants:

```bash
./build_v2_release/llaminar2 plan -m /models/moe.gguf \
  --mpi-hostfile /cluster/hosts --only-backends rocm,cpu \
  --auto-hosts all -c 32768 --output /tmp/llaminar-cluster-plan.json
./build_v2_release/llaminar2 serve --config /tmp/llaminar-cluster-plan.json
```

`--hostfile` and `--mpi-hostfile` name the same MPI cluster intent. MPI owns
hostfile slot admission; each rank supplies its actual hardware and worker
geometry. Do not infer physical hosts from rank numbers or copy launcher-local
CPU IDs/widths onto remote hosts. Images/binaries and model paths must already
be provisioned on peers; a hostfile does not distribute them. See the README's
remote-MPI recipe for container transport setup. A node-local channel remains
node-local even when a hostfile is supplied.

`plan` and `serve` share the complete runtime parser. Plan adds only checked
output options; never restore a private MTP, KV, strategy or economy table.
Automatic search runs only on discovery root through `AutomaticPlanningStartup`;
all discovery ranks participate in evidence preparation using one immutable
inventory. Publish the complete apply document before rank admission. Runner
factories remain apply-only; do not repeat search on followers or in a factory.
Saved execution selection retains discovery-rank order and process count; apply
it through `MPIContextFactory::selectRanks` before runner construction. Never
interpret execution device maps as discovery rank IDs or narrow discovery launch
from the selected inference endpoints alone.

Automatic ranking uses the shared `OrchestrationPlanningWorkload` policy
(CLI `--plan-workload`, YAML `planning.workload`). Do not substitute KV capacity
or a frontend-local horizon for that objective. Applied plans contain placement,
not another search policy. Compact tiers also adapt to existing canonical
domain/tier types; they must not acquire a second capacity or role authority.

Production model activations support FP32 only; other activation precision
requests fail as unimplemented. KV precision, allreduce wire precision and expert
weight formats are separate settings. Leave their defaults intact for the
initial auto baseline. Prefix caching is enabled by default; MTP is independently
enabled with `--mtp`, and `--mtp-depth-policy dynamic` selects adaptive depth.

MTP hardware defaults are selected once by `ExecutionPlanBuilder` from the
complete continuation domain and canonical device inventory. Their numeric
source is `execution/config/MTPDepthDefaults.h`. Preserve automatic versus
explicit request intent through CLI/YAML and MPI; never infer an override by
comparing its value with an old default. Other expert tiers cannot select the
continuation policy. Request admission resolves the profile into the existing
device-controller ABI without changing MTP enablement, mode, or graph capacity.

`--mtp-terminal-head-policy auto` is the public default: CPU continuation
domains use vocabulary-sharded heads, while CUDA/ROCm retain mirrored full
heads. Expert-only tiers do not change that choice. Explicit
`vocabulary-sharded` or `mirrored-full-vocabulary` overrides are preserved.
The rank compiler seals the policy before memory admission; graph construction,
saved plans, and physical-weight accounting consume that same resolved value.

## Benchmarking

Use a Release binary and a fixed model, device, prompt bytes, decode length,
sampling policy, and runtime configuration for comparisons.

```bash
# Measure automatic production selection first, with only a backend constraint.
./build_v2_release/llaminar2 benchmark \
  -m models/model.gguf --only-backends rocm \
  --prompt "A fixed prompt used for every comparison." \
  -n 256 --temperature 0 --seed 42

# Use a saved plan when the comparison must retain identical placement.
./build_v2_release/llaminar2 benchmark \
  --config /tmp/llaminar-plan.json \
  --prompt "A fixed prompt used for every comparison." \
  -n 256 --temperature 0 --seed 42
```

Greedy sampling (`--temperature 0 --seed 42`) is not diagnostic kernel
determinism. `--deterministic` also sets `LLAMINAR_DETERMINISTIC=1`, which can
disable optimized kernel dispatch and projection concurrency. Do not use it
as an incidental benchmark convenience or compare that result with normal
production dispatch without identifying the changed policy.

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
`v2_production_test_preflight_gate`, then runs the full Unit namespace and
`ProductionTestPreflight` label on every branch. It runs no model campaigns,
E2E, broader integration selections, or benchmarks. Register the tracked hooks
with `git config --local core.hooksPath .githooks`; see `.githooks/README.md`.
The full shippable-image gate is `scripts/ci/run_production_pipeline.py`:
independent AVX512 and AVX2 full CPU/CUDA/ROCm Docker builds, each running the
canonical Unit/preflight gate, reviewed HTTP token regressions (MTP off and
dynamic depth only), and all E2E-tagged cells. Mathematical HF parity is explicit
diagnosis via `--diagnostic-mathematical-parity`, never a routine CI dependency.
Prompts, seeds and cell/control mappings are canonical source definitions;
expected token payloads are versioned in `Llaminar/corpora` and source-pinned.
Never update expected tokens automatically. Both full
E2E server suites must pass before either ISA's benchmarks run. Only complete
per-image evidence may attach a certificate; official publication requires
both images. Explicit one-off diagnostic benchmarks cannot certify images.
An explicit protected full-certification invocation commits both compact ISA
result JSONs and one combined upward-only high-water proposal; local runs never
commit or publish implicitly. See `docs/production-ci.md`. Do not
add a second benchmark topology/model list or reintroduce this pipeline into
pre-commit.

The enabled `develop` GitHub workflow is deliberately smaller: it invokes
`scripts/ci/run_develop_image_gate.py` to build AVX512 and AVX2 full-backend
images, run only the complete Unit and `ProductionTestPreflight` gates inside
each builder, and publish the two tested `develop` runtime tags. It does not
run model/generation/parity/E2E/benchmark certification and must not be
expanded into a second production-pipeline implementation. See
`docs/production-ci.md` for the exact boundary and local invocation.

The separate manual `production-e2e.yml` and `production-benchmarks.yml`
workflows consume the branch's existing GHCR image pair. They use
`run_published_image_suite.py`, canonical typed inventory discovery and the
existing HTTP/benchmark runners. Both entire same-image E2E suites precede
either benchmark. Only complete benchmark results may update the owned README
chart block and compact JSON; this phase evidence is not full image certification.
Never substitute a rebuilt runtime for a missing published image or maintain
a workflow-local model/topology matrix. See the manual published-image section
of `docs/production-ci.md`.

For an isolated model-free gate, `scripts/ci/run_production_prerequisites.py`
delegates to the same complete Unit/preflight authority and preserves its
receipt, CTest logs and JUnit evidence. An installed-builder receipt replaces
build preparation only; it never skips tests or independently certifies an
image. See `docs/production-ci.md` for the local and installed-image invocation.

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

Use `.agents/llaminar-testing/SKILL.md` for the testing workflow. For model parity, read
`tests/v2/integration/parity/README.md` and discover the exact registered
diagnostic campaigns. On token drift or suspected accuracy errors, select the
matching HF cell and use its CSVs to find the first divergent stage; do not
launch the full mathematical matrix routinely or rebaseline the failing stream.
The aggregate campaign system replaces the historical
hand-picked PyTorch parity baseline: it keeps reference generation, live-path
execution, every checkpoint comparison, CSV evidence, and the shared economy
target in one registered matrix. Before model admission, a build must pass the
complete CMake-owned `V2_Unit_*` suite and the model-free
`ProductionTestPreflight` integration label. Unchanged local diagnostic runs
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
