---
name: mtp-tuning
description: Tune, debug, or extend Llaminar V2 MTP/speculative decoding, prefix-cache interaction, vLLM-style accepted-state publication, MTP depth control, grouped verifier kernels, dense/MoE MTP parity, and MTP benchmark dashboards. Use when Codex is asked to improve MTP speed, fix MTP correctness, compare CUDA/ROCm/CPU MTP lanes, work on decode-equivalent grouped verifier rows, remove host/device coherence issues, update the MTP project plan/dashboard, or touch MTP-related tests and perf gates.
---

# Llaminar MTP Tuning

## Purpose

Use this skill to work on Llaminar's vLLM-style MTP path without re-learning the
same hard-won rules. MTP tuning is only successful when it improves real served
inference speed while preserving strict decode equivalence across CPU, CUDA, and
ROCm, dense and MoE, greedy and stochastic.

Always keep these files current:

- `docs/v2/projects/2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md`: phase plan,
  accepted architecture, gates, and known debt.
- `docs/v2/projects/2026-06/MTP_VLLM_STYLE_TUNING_DASHBOARD.md`: compact RAG
  status and latest speed/correctness evidence. Keep it under the documented
  size limit.
- `docs/v2/projects/2026-06/PREFIX_CACHE_MTP_BENCHMARK_NOTES.md`: only for
  broader historical benchmark notes when the current dashboard is not the
  right home.

For backend kernel work, also use the relevant sibling skill:

- `.agents/cuda-tuning/SKILL.md` for CUDA profiling, Nsight, and generated
  NativeVNNI dispatch work.
- `.agents/rocm-tuning/SKILL.md` for HIP/ROCm profiling, rocprof, ISA analysis,
  and generated NativeVNNI dispatch work.

## Architecture North Star

Llaminar is moving toward vLLM-style speculative decode:

1. Draft rows live in speculative slots, not live model state.
2. The target verifier runs `draft_count + 1` rows: draft rows plus one bonus row.
3. Device-side sampler/rejection logic produces accepted counts and output tokens.
4. Only accepted speculative slots are published to live state.
5. Rejected suffix and bonus-only rows must not mutate live KV, GDN, short-conv,
   positions, terminal hidden, terminal logits, or sampler history.
6. Greedy is a deterministic specialization of the stochastic contract, not a
   separate architecture.

The canonical transaction shape is:

```text
prepare spec slots
run draft graph
run target verifier graph
run rejection sampler / accepted-count reducer
publish accepted state
discard rejected state
return tokens
```

Keep graphs per-device and symmetric. Do not introduce nested multi-device
sidecar graphs. LocalTP, LocalPP, NodeTP, and ExpertOverlay must extend the
same transaction semantics with collective coordination, not invent separate
state machines.

### Canonical MoE LLEP Policy Tuple

`LLEP` names one current-batch routed-row assignment policy. It is not a
complete execution mode, routed-expert storage policy, decode policy, durable
residency controller, or hot-cache policy. Keep these axes explicit whenever
constructing, testing, or benchmarking a MoE graph:

- dense/shared trunk: tensor parallel;
- routed-expert storage and compute: apportioned whole experts;
- routed phase: uniform;
- grouped verifier/decode assignment: static owner;
- ordinary large-prefill assignment: least-loaded resident (LLEP);
- MTP terminal norm/head: mirrored full vocabulary;
- durable expert-residency maintenance: off;
- hot expert replica cache: off;
- same-backend transport: one fully captured NCCL or RCCL graph.

The prefill row threshold is an explicit work-regime boundary, not recovery
behavior: below it, execute economical static-owner EP; at or above it, execute
the current-batch least-loaded assignment. Set the threshold to zero only in a
focused integration test that must force and prove the transfer-backed LLEP
path. Never let prefill assignment bleed into grouped verifier/decode, and
never use whole-expert decode movement as a proxy for LLEP.

Declare the tuple through `DenseParallelPolicy`,
`RoutedExpertComputePolicy`, `RoutedExpertPhasePolicy`, separate
`routed_decode_assignment` and `routed_prefill_assignment` domain fields,
`MTPTerminalHeadPolicy`, `--moe-residency-maintenance`, and
`--moe-hot-expert-cache`. Graph builders own the resulting stage, collective,
and event wiring. PerfStats must prove static-owner grouped verification,
least-loaded large prefill, mirrored terminal-head execution, full graph
capture, and zero segmented execution.

## Non-Negotiable Rules

- Never use CUDA/HIP default or null streams. Every GPU operation needs an
  explicit stream, including copies, memset, events, reductions, sampling, and
  graph-captured stages.
- Never allocate GPU memory in the hot path. Use declared graph workspace,
  `IWorkspaceConsumer`, arena buffers, or low-level sanctioned allocators.
- Never mutate tensor residency flags directly. Use `TransferEngine` unless a
  graph-stage buffer contract already provides the correct resident pointer.
- Never capture H2D copies inside GPU graphs. Upload persistent inputs before
  capture and replay device-resident buffers.
- Never leave quiet recovery paths. Every advertised MTP lane must implement
  its declared policy or fail fast with a precise diagnostic.
- Never land special codebook exceptions in production kernels. Use generic
  dispatch keyed by codebook family, M, aspect ratio, work size, and generated
  policy tables. Exact shape overlays are additive to a total general dispatch
  rule; they never substitute for that rule.
- Never accept token equality alone as correctness proof for verifier kernels.
  Require distribution and numeric gates.
- Remove dead-end code and tests once a path is abandoned. Negative tests for
  deleted approaches just confuse future work.
- Keep CUDA and ROCm structurally aligned. Backend-specific code should sit
  behind common stage/kernel interfaces.
- Write regression unit or integration tests for every crash, coherence bug,
  stream bug, dispatch bug, and parity break found during tuning.

## Correctness Gates

Use strict gates before promoting any MTP optimization:

- **Grouped verifier decode must be bitwise serial-row equivalent.** For
  grouped MTP verifier rows, the acceptance threshold is byte-for-byte identical
  FP32 output compared with running the same rows through the production M=1
  serial decode path in the same backend and tensor format. Cosine similarity,
  relative L2, symmetric KLD, and max-absolute-error are diagnostics for finding
  the first drift; they are not pass criteria for grouped decode publication.
- Relative L2: tight enough to catch drift for the tested precision/path.
- Cosine similarity: near 1.0 for logits, hidden rows, and kernel outputs.
- Symmetric KLD: required when outputs feed sampling or softmax decisions.
- Max absolute error: required for small-M kernel equivalence.
- Sampled-token equality: required, but not sufficient by itself.
- Decode-equivalent continuation: accepted-state publication must match serial
  decode after continuing for enough rows to expose KV/GDN/short-conv mistakes.

Grouped verifier APIs and kernels must accept runtime M up to the graph and
workspace capacity; never encode speculative depth as an `M=2..4` template or
admission limit. M=1 is the independent production serial-decode oracle. The
canonical grouped inventory proves every integer M=2..16 plus M=31 for every
backend and every advertised tensor codebook or floating-point tensor format.
M=31 is a deliberate deeper sentinel for speculative regimes beyond the usual
fifteen drafts; it is not a maximum. A larger configured graph capacity must
extend the contiguous sweep through that capacity rather than sample only its
endpoint. Use fixed-size physical row tiles to bound register pressure and
preserve weight reuse as M grows; do not generate one kernel specialization per
speculative depth.

Prefer dedicated integration tests before wiring a new kernel into graph
execution. Good test names to search for include:

```bash
rg -n "VerifierRows|decode_equivalent|PrefixMTP|GPUSampling|NativeVNNI|QuantisedGemmSmallM" tests/v2
```

Common focused gates:

```bash
ctest --test-dir build_v2_integration -R "^V2_Unit_PrefillDecodeTransition$|^V2_Unit_MTP|^V2_Integration_GPUSamplingKernels" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_.*MTP|^V2_Integration_.*Prefix" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_.*VerifierRows|^V2_Integration_.*QuantisedGemmSmallM|^V2_Integration_.*NativeVNNI" --output-on-failure --parallel
```

### Mandatory all-format grouped verifier sweeps

These integration suites are the first line of defense for grouped verifier
decode. They prove that production grouped rows are byte-identical to serial
decode across the backend's advertised tensor/codebook formats. Run them after
any change to MTP verifier rows, NativeVNNI/GEMV dispatch, MoE routed/shared
expert kernels, LocalTP verifier output projection, attention, RMSNorm, or GDN
projection. A failure here is a production grouped implementation bug; do not
paper over it with serial row replay, relaxed thresholds, or a backend-specific
format exception.

Treat these sweeps as mandatory architecture proof, not ordinary smoke tests.
Grouped decode is publishable only when every production grouped implementation
is both byte-exact against serial M=1 decode and genuinely grouped/economical
for every format that backend can load. If a new grouped verifier operation,
codebook family, floating-point tensor format, or backend lane is added, extend
the corresponding sweep in the same slice before tuning or claiming the lane is
complete.

The CUDA all-format gate must include the production fused projection,
large-K KPAR, fused SwiGLU/down, FP32, FP16, and BF16 runtime-M tests. Search for
`RuntimeM2To16` in `Test__CUDAGemmParity.cpp`; the low-level NativeVNNI case also
captures and replays every format/depth combination. Equivalent CPU and ROCm
lanes must use the same contiguous row inventory.

Routed MoE format sweeps must enter through the production router before the
grouped expert pass and validate the router-to-expert Q8 publication counter;
precomputed host route IDs do not prove this optimization. The canonical
counters are `cpu_moe_grouped_verifier_router_q8_reuse_calls`,
`cuda_moe_grouped_prefill_router_q8_reuse_calls`, and
`rocm_moe_grouped_prefill_router_q8_reuse_calls`. CPU serial-oracle rows must
also observe `cpu_moe_decode_router_q8_reuse_calls`. A byte-equal result without
the matching counter is a failed gate because it may only show that two
independent quantizers happened to agree for the synthetic input.

```bash
# Discover the exact registered inventory first, then run every present and
# future grouped-verifier lane. Do not replace this prefix gate with a hand list.
ctest --test-dir build_v2_integration -N -R "^V2_Integration_GroupedVerifierRows_"
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_" --output-on-failure --parallel

# Backend slices are useful while iterating, but all three remain mandatory
# before a grouped-verifier slice is accepted.
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_CPU_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_CUDA_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_ROCm_" --output-on-failure --parallel
```

As of 2026-07-13 the prefix gate discovers 51 substantive lanes: 13 CPU, 17
CUDA, and 21 ROCm (`52` CTest entries including the model fixture). The
inventory includes all-format GEMM, MoE codegroups and
expert paths, floating formats, dense QKV/GDN projections, replicated LocalTP
output projection, embedding, RMSNorm, fused residual norm, residual add,
SwiGLU, RoPE, attention, KV-cache append, GDN recurrence, short-conv,
request-batched GDN state, and device-resident stochastic sampling. The
prefix command is canonical precisely so a newly registered operation cannot be
omitted from an otherwise plausible-looking hand-maintained regex.

The backend counts are an inventory, not a symmetry waiver. Before the final
MTP gate, compare the semantic operation matrix across CPU, CUDA, and ROCm and
add every applicable missing lane. In particular, CPU currently lacks the GPU
stochastic-resident and request-batched recurrent-state lanes, while explicit
all-format/runtime-M projection and MoE codegroup discovery is not named
uniformly across the three backends.

The CUDA and ROCm `GDNRecurrence` and `ShortConv` lanes also prove captured
graph lifetime across accepted-state publication. Each backend captures the
ordinary M=1 decode graph once, runs grouped verifier M=2/3/4 into isolated
speculative state, publishes every possible accepted row by a device-owned row
index, detaches verifier bindings, and replays the original graph without
recapture. Both the continuation output and complete recurrent live state must
match serial M=1 decode in native bytes. This proof is what permits the typed
correction-boundary policy to retain single-token decode and all-position
verifier captures; a blanket CUDA/ROCm graph reset is a regression, not a
conservative fallback.

The CUDA and ROCm `KVCacheAppend` gates each enumerate the same 288 production
routes: every accepted cache/source format pair (including asymmetric prepared
TQ8-K/TQ4-V), M=2/3/4, position-major and verifier-head-major inputs,
replicated and LocalTP-sharded caches, and direct-static versus graph-captured
device-dynamic ring metadata. They require native cache byte equality, sequence
metadata equality, prefix-block export/import round trips, and exactly one
grouped route-counter observation per matrix cell.

The CPU `RoPE` gate covers FP32, BF16, FP16, pure-integer Q8_1, and Q16_1 with
32/64/128-value native blocks at M=2/3/4. It compares native bytes against
serial M=1 decode and checks the `single_grouped_row_head_workshare` route, so a
row loop hidden behind the grouped interface is not an acceptable replacement.

The explicit CUDA and ROCm `RoPE` gates mirror one 24-cell matrix per backend:
FP32 full/partial plus BF16 and FP16, M=2/3/4, and both contiguous
device-scalar and explicit device-row position owners. They compare native GPU
bytes with the same backend's production M=1 decode kernel, require non-default
streams, and reject a missing `single_grouped_launch` route counter.

The explicit CUDA and ROCm `RMSNorm` gates each cover 18 positive production
cells: FP32, BF16, and FP16 at M=2/3/4 for 128-column per-head Q/K normalization
and 4096-column hidden-state normalization. Those widths cross the narrow and
wide reduction launch policies. Every cell enters `apply_tensor` on a
non-default stream, compares native output bytes with same-backend M=1 decode,
and requires one `one_block_per_row` / `single_grouped_launch` counter. Three
additional negative cells prove that every native format rejects an unbound
default stream without publishing route telemetry.

The CPU, CUDA, and ROCm `ResidualAdd` gates each cover FP32, BF16, and FP16 at
M=2/3/4 for 128- and 4096-column rows. They require native byte equality against
same-backend M=1 decode and exactly one flat workshare/launch counter. GPU gates
also require device-only `gpu_data_ptr()` ownership and verify that every format
rejects an unbound default stream; in particular, this guards ROCm BF16's stream
setter, which must remain symmetric with FP32 and FP16.

The CPU, CUDA, and ROCm `FusedResidualNorm` gates cover the same 18 positive
cells per backend but enter `FusedResidualNormStage`, the actual graph operation
at attention and FFN residual boundaries. Both the in-place residual publication
and normalized output must match independent production M=1 stage executions in
native bytes. CUDA and ROCm additionally run one null-stream rejection cell per
format and require one device-resident fused launch; this prevents the separate
ResidualAdd/RMSNorm suites from masking broken fused type dispatch or stale host
ownership.

The standalone `SwiGLU` gates cover M=2/3/4 at 544 and 4864 columns. CUDA and
ROCm sweep FP32/BF16/FP16 with explicit streams and one flat launch; CPU adds
Q8_1 and requires one native workshare. The 544-column case is deliberately 17
Q8 blocks wide, so grouped execution cannot pair blocks across a row boundary
without the native-byte oracle noticing. These gates complement, rather than
duplicate, fused GEMM+SwiGLU-down coverage.

Use the backend-specific all-format sweep as the minimum acceptance gate for a
narrow kernel edit, then run the full backend group before claiming that grouped
decode is proven for that backend. Model-level prefix/MTP parity and
`Qwen36_MTPForwardVerifierOperationEquivalence` remain required because they
catch cross-stage amplification, but they do not replace the all-format sweeps.

Do not run only the easy backend. If CUDA has a deep PyTorch or layer-by-layer
test, ROCm and CPU need the same semantic coverage unless the plan explicitly
marks the lane as not implemented.

Refresh the inventory with `ctest --test-dir build_v2_integration -N -R
"^V2_Integration_GroupedVerifierRows_"` when adding tests. Keep every canonical
test under that namespace so the precommit, CI, and skill prefix gates include
it automatically.

## Performance Methodology

Start with real inference, then isolate:

1. Capture a Release benchmark for the relevant lane with fixed d1/d2/d3 and
   dynamic depth.
2. Enable `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` with JSON/CSV export to rank
   graph-replay bottlenecks on the production captured topology. The deprecated
   `LLAMINAR_PROFILING` alias must never disable graph capture or select eager
   execution.
3. Split host bridge accounting from producer work. A D2H timer that includes
   waiting for a producer event is not proof of copy overhead.
4. Attack the biggest real chunk first. For MTP this is usually verifier forward,
   LM head, grouped GEMV/GEMM, attention, GDN/short-conv, sampling, or state
   publication.
5. Microbench candidate kernels with strict parity against serial decode.
6. Re-run the model-level benchmark and parity suite after each concrete slice.
7. Update the dashboard with the run directory, speedup, acceptance rate, and
   current RAG.

Useful benchmark shape:

```bash
scripts/run_mtp_iteration_benchmark_matrix.sh \
  --devices cuda:0,rocm:0 \
  --models dense,moe \
  --modes greedy,stochastic \
  --variants fixed_d1,fixed_d2,fixed_d3,dynamic \
  --decode-tokens 16 \
  --perfstats \
  --output-dir benchmark_results/mtp_vllm_style/<timestamp>
```

Use `scripts/summarize_mtp_perfstats.py` to compare:

- `decode_step_ms`
- verifier forward / graph replay time
- stage GPU time by type
- sidecar time
- publish time
- sampler/reducer time
- response-ready wait versus actual D2H enqueue/wait
- acceptance, rejection, and rollback counts

### Isolated NativeVNNI profiler evidence

NativeVNNI policy training has two physically separate evidence transactions.
Canonical candidate timing runs first, with production warmups and
sample-interleaved rounds. Only after those observations and raw timing samples
are immutable may `profiler_evidence.py` derive per-candidate requests.
`profiler_collectors.py` then starts one fresh process per request and gathers
Linux `perf`, Nsight Compute, or rocprofiler metrics from an extra isolated
launch. Profiler replay duration is never a timing label.

Every supported observation needs one evidence record; unsupported candidates
need an explicit non-profile state. CUDA/ROCm candidates may be pipelines, so
retain quantization, producer, reducer, and epilogue dispatches independently
and in launch order. Optional counters unavailable on one architecture remain
typed unavailable fields. A missing tool, raw report, required counter, or
supported candidate record blocks production evidence completeness. Production
`scripts/refresh_native_vnni_dispatch_tables.sh --profile all` enables this gate
by default, and `--install --skip-profiler-evidence` is forbidden.

The required `*_profiler_features.csv` export is an authenticated join of the
canonical common-observation timing CSV, request manifest, and evidence
manifest. It retains pipeline-level canonical timing alongside one ordered row
per physical dispatch and that dispatch's own counters. Never train from a
hand-joined profiler CSV or treat profiler replay duration as canonical
latency; a corpus/request/evidence digest mismatch must fail the refresh.

### Immutable Git LFS corpora and turnkey refits

The canonical cross-backend procedure now lives in
`.agents/nativevnni-gemm-tuning/SKILL.md`. Keep this section focused on MTP's
strict grouped-row equivalence and its interaction with speculative decode.

Use `scripts/train_native_vnni_dispatch.sh --backend <backend> --install` as the
normal production entry point for `cuda`, `rocm`, `cpu`, and `cpu-prefill`.
The command rebuilds both vendor policy scorers, validates their integration
oracles, benchmarks and profiles only when no matching corpus generation is
published, fits and certifies, and atomically installs the generated include.
Additional refresh options belong after `--`.

Published generations live under
`corpora/native_vnni_dispatch/<backend>/<architecture>/<shape-digest>-<configuration-digest>/`
in the optional `Llaminar/corpora` submodule. All published corpus families
belong in that data repository; the source repository pins only its gitlink.
Initialize and selectively pull LFS through the canonical tuning skill's
corpus workflow, never during ordinary builds or Unit/preflight gates.
`corpus.manifest.json` is ordinary Git metadata there; large timing/profiler
payloads are Git LFS objects. Never hand-edit a sealed generation. The corpus verifier
must reject unresolved LFS pointers, partial files, symlinks, changed payload
digests, and stale resolved shape inventories. Fit-only replay materializes a
disposable workspace and uses `--skip-sweep --reuse-profiler-evidence`; it may
re-export authenticated profiler features but must launch no candidate kernel
and no profiler. A failed fit is mined again from the same corpus rather than
triggering another timing sweep.

The final corpus directory suffix also binds forwarded collection arguments.
Do not pass `--shapes` or `--shape-partition` through the turnkey command;
production overlays belong in the shared inventory so all backends see them.

Every exact overlay comes from the shared resolved shape inventory consumed by
the CPU, CUDA, and ROCm trainers. Backend-private shape lists are forbidden.
An overlay addition changes the inventory digest and therefore requires a new
corpus generation; exact overlays remain additive to mandatory generic rules.

### GPU-accelerated NativeVNNI policy fitting

The common policy learner can batch its first two exact leaf keys on CUDA and
ROCm. Build `v2_native_vnni_leaf_primary_scorer_cuda` and
`v2_native_vnni_leaf_primary_scorer_rocm`; production
`refresh_native_vnni_dispatch_tables.sh --profile all` auto-discovers both
vendor inventories and uses isolated one-vendor worker processes. Each physical
device defaults to eight CPU orchestration lanes because tree construction is
still host work and the scorer kernels are brief. Tune with `--policy-lanes`,
or explicitly choose `--policy-accelerators cpu` for the canonical CPU fitter.

Each GPU scorer lane must keep one uploaded regret matrix, non-default stream,
and growable scratch session alive across all leaf batches for a fit. Do not
reintroduce per-call stream creation, device allocation, or full-matrix upload.
Schedule CV/final-fit tasks through the longest-first dynamic device queue and
refill whichever lane completes first; restore task-index order before policy
reduction so scheduling cannot affect generated bytes. Static per-lane task
partitions are forbidden because heterogeneous folds create a severe idle tail.
Represent tree point subsets as arbitrary-width integer masks, cache exact
threshold membership masks, and retain ordered point tuples only for canonical
later-key/report materialization. Keep the unit invariant that leaf masks are
disjoint, cover the full point inventory, and map exactly to retained ordered
points; generated report and request bytes must match before/after any beam
representation optimization.

The GPU computes maximum regret as a diagnostic and the integer count at or
above the strict 5% boundary. Python derives p95 and mean regret and retains
candidate-name tie-breaking so report/request bytes stay identical to CPU
fitting. A policy is installable only when measured p95 regret and conservative
p95 regret upper bound are both strictly below 5%; maximum regret is never an
installation gate. Run
`V2_Integration_NativeVNNILeafPrimaryScorer_CUDA` and `_ROCm`; unit tests must
remain GPU-free. Once acceleration is explicitly requested, a missing DSO,
device, or worker is a hard failure. Never silently resume on CPU, and never
confuse policy-scoring GPU work with separately timed candidate kernels or
isolated profiler launches.

GPU scorers do not replace backend candidate measurement. In particular, CPU
kernel timing must remain on CPU. On a multisocket trainer host, split distinct
shape/source/ISA jobs across an MPMD MPI launch with one rank per socket. Never
launch the same tuple on both ranks, and give every rank distinct aggregate and
raw-timing paths. Pair only jobs with identical ordered M inventories. Enable
`LLAMINAR_CPU_NVNNI_PREFILL_MPI_ROUND_SYNC=1` so every warmup and timing round
ends in an MPI active-rank reduction; a locally converged rank must continue
unmeasured complete-round pacing until every rank finishes the same M phase.
Without that coordination, one rank can advance to a larger M and manufacture
frequency or memory-pressure drift in its peer's timing series.

Installable CPU prefill evidence must authenticate
`mpi-complete-round-v1`, MPI world size/rank, and global warmup/timing round
counts. The adapter must reject candidate/global round disagreement and any
multi-rank production corpus collected with process-local pacing. Do not fix a
cross-rank transient by raising the sample ceiling, relaxing the 2% drift gate,
or selecting a convenient stable tail.

Warmup is elapsed-evidence controlled, not round-count controlled. An M64
candidate can legitimately require more than 10,000 launches to accumulate one
second, so do not reintroduce a fixed warmup-round ceiling. Keep the 30-second
per-complete-round watchdog, require every measured launch to make finite
positive timer progress, and record total phase wall time plus maximum round
duration. Long-M phases may legitimately exceed 30 seconds across several
rounds; retain the per-candidate measured-duration gate as the promotion
requirement.

After the one-second source preconditioning cell, derive each CPU prefill
candidate's M-transition warmup from its first production-route latency probe:
`max(100 ms, min(2 s, probe_latency * 60))`. Authenticate the policy, probe,
floor, multiplier, ceiling, effective budget, and achieved duration in raw
evidence. A fixed 100 ms budget was insufficient for a real IQ3_S M4096
transition whose first roughly 45 timing launches were still elevated. Do not
repair this class of failure by filtering the timing tail or relaxing the 2%
full-history drift gate.

Use timing protocol `elapsed-stability-interleaved-v13` with ceiling policy
`samples-and-elapsed-v1`. A stable candidate may still stop at the ordinary
30-sample floor. An unstable candidate must satisfy both `sample_count >= 180`
and `timed_duration_us >= 2,000,000` before the trainer can reject it; 180 is
not an unconditional maximum. A real cheap M64 route reached 180 samples after
only about 203 ms while still settling. The forced zero-drift regression must
continue beyond 180 samples, accumulate the full measured-time budget, flush
both evidence files, and fail collectively without hanging its peer.

All MPMD ranks must make timing rejection collectively after flushing their
aggregate and timing sidecars. A rank-local GTest fatal assertion can strand a
peer in the next-M barrier or MPI finalization. Reduce the failure decision and
abort the communicator before any rank advances; retain an asymmetric direct
regression that proves the failure exits instead of hanging.

For a production CPU corpus, add `--cpu-format-shards` and bound each invocation
with `--cpu-batch-limit N`. Re-run against the same output directory with
`--resume-cpu-partials`; completed shape/format/ISA outputs are atomic and must
not be measured twice. Never treat `.inprogress` files as evidence, combine a
partial checkpoint, invoke the learner on it, or install from it.

For GPU corpora, use `--cuda-measurement-lanes N` and
`--rocm-measurement-lanes N` to assign disjoint round-robin format shards to
homogeneous physical devices. The wrapper must reject excess or heterogeneous
lanes, keep one aggregate/timing pair per lane, and merge only after every lane
is green. CUDA and ROCm timing transactions may run concurrently when they own
separate GPUs, but each concurrent command must restrict
`--policy-accelerators` to its own backend. Never let CUDA policy fitting borrow
ROCm devices, or vice versa, while the borrowed backend is collecting canonical
timing. Profiler replay is a separate launch and must never overlap canonical
timing on the same physical device.

Treat the checked-in GPU measurement plan as the physical timing authority;
the broader shape manifest is the support inventory. Common development shapes
must cover every production geometry and every reviewed feature bucket on all
formats. Backend/format-specific refinements must run only on their declared
surface: the v1 `fast-m1-cv-refinement-v5` extension is CUDA + Q4_0 + Fast M1,
is shape-sharded across CUDA lanes, and must never appear in ROCm, another
format, or grouped-verifier evidence. Before a production run, execute
`test_native_vnni_gpu_measurement_plan.py` and the refresh-wrapper regressions;
the analyzers must reject missing and unexpected
`(contract, M, shape, format, execution mode)` surfaces.

The CPU ordinary-prefill covering plan must include the minimum-N*K production
geometry for every codebook/M/aspect/ISA domain. Pairwise feature coverage alone
can choose all witnesses from one larger model group, leaving the generic
learner without enough independent groups to certify a small-shape decision.
Plan v7's generic covering array contains 1,583 runtime cells and 1,829
source-alias cells. Do not install from that base plan alone. Run the zero-kernel
production route probe in all three build/runtime ISA regimes and pass its
manifests to the planner and analyzer. Route closure expands the blessed host
plan to 1,820 runtime cells, 2,107 source cells, and 653 process jobs because the
finite native-AVX512 serial-K-part family must be timed exactly. Its
lower-boundary anchors add only about 2.65% estimated GEMM work. When rebasing a
checkpoint to a new plan contract, reuse an atomic shard only if its complete M
inventory exactly matches the new source record; otherwise omit the whole
aggregate and timing pair for recollection.

Treat the C++ route manifest as authoritative for thread count, `k_tiles`, and
the full-K versus serial-K-part arithmetic bundle. Do not reproduce the
cache-aware `computeTileConfig()` heuristic in Python. AVX2 serial-K-part has a
single forceable Pairwise schedule and needs no learned tournament. Native
AVX512 must measure Pairwise and WideRows over its complete finite production
family; the current route-certified corpus selects Pairwise for `M=2`, where
WideRows is unavailable, and measured WideRows for `M>=3`. Production `Auto`
must consume the generated table and fail hard on an unresolved surface.

Keep certification geometry inside the checked-in manifest's supported
production envelope. The current maximum is `1,271,398,400` weight elements,
set by the largest supported production projection. CPU timing has a narrower
checked-in ceiling of `778,567,680` weight elements, exactly the Qwen2.5 32B
`152064x5120` LM head; retain explicit 32B attention, QKV, FFN-up, FFN-down,
and LM-head overlays in that matrix. Do not delete larger runtime geometries
from the shared manifest merely to shorten CPU collection. Do not add larger
synthetic lattice points merely to make a sweep look exhaustive; the manifest
loader must reject them.

Keep MTP grouped-verifier measurements at `M=2..16,31`. The ordinary CPU
NativeVNNI GEMM/prefill buckets are
`M={64,256,1024,2048,4096,8192,16384}`. Cap 32B at `1024`, cap 14B at `4096`,
and run the full range through `16384` for 9B-and-smaller models. The verifier
and prefill matrices are separate contracts; do not inflate verifier collection
with prefill-only batch sizes or run giant 32B prefills merely for symmetry.
CUDA and ROCm use the same buckets but directly measure every 14B/32B
projection through `8192`; their higher ceiling is a backend-specific economy
obligation, not permission to weaken the CPU timing target.

Query the authoritative ordinary-prefill matrix with:

```bash
PYTHONPATH=tests/v2/performance/kernels \
  python3 -m native_vnni_dispatch.prefill_matrix --records
```

The production measurement test is
`CPUNativeVNNIGemvTest.TrainerCsv_StrongPrefill_AllFormats`. It must emit all
eight registry requests, retain normalized requests as unsupported evidence,
prove both launches byte-equal to serial M1 rows, and keep isolated Linux
`perf` collection outside the timing samples.

Paired refinement uses the wrapper-owned `fit-cache` directory. The compact
source/manifest/projection identity, candidate costs, and CV results are
content-addressed; candidate costs and CV are cached independently per
mode/codebook/aspect domain and keyed by only that domain's tournament evidence.
A new edge must not refit unaffected domains. Formula rows are projected lazily
only for a cost-miss or final-fit domain; CV traverses fitted trees directly
from the cost matrix. Never serialize the expanded projected corpus. The
learner requests unresolved edges from its complete evaluated CV-model frontier
in the current timing batch. Keep the cache-hit and domain-invalidation
unit regressions green whenever changing CV, candidate costs, or paired request
planning.

Ordinary NativeVNNI prefill is an independent common-policy surface. On CPU,
train only physical schedules proven by
`NativeVNNIPrefillFullKAllFormatsRuntimeMMatchesSerialDecode`: the row-chunk
grid formula and eligible two-row N-block schedules, each with exactly one
full-K tile. Reject K-tiled accumulation candidates whose intermediate FP32
stores change serial-M1 parenthesization. Require the
`cpu_native_vnni_prefill_gemm_launch` perfstats route to match the requested
candidate before admitting timing evidence.

For M=1 GEMV, report useful prepared-weight bandwidth as a fraction of an
empirically measured sustainable HBM roofline, alongside physical profiler
traffic. Keep canonical median latency as the dispatch label: launch overhead,
unpack work, and deterministic K-partition traffic can make bandwidth alone
choose the wrong winner. Grouped M>1 reports also need GOPS and arithmetic
intensity because weight reuse can move the kernel away from the HBM roofline.

## Grouped Kernel Tuning

MTP only becomes economical when grouped verifier work is genuinely grouped.
Wrapping serial decode rows under a grouped API is not enough.
Grouped decode publication also has a stricter correctness bar than ordinary
numeric parity: every grouped output row must be bitwise identical to the
backend's serial M=1 decode row for the same codebook/format. Treat relaxed
numeric metrics as failure breadcrumbs, not success thresholds.

### Verifier Kernel Mode Convention

When verifier rows can be published into live MTP/KV/GDN state, stages must request
backend verifier modes through `ITensorGemm::beginVerifierDecodeEquivalentScope()`.
Do not call backend `extern "C"` toggles or environment variables directly from
stage code. CUDA and ROCm use this shared RAII hook to select generated small-M
dispatch/reduction policies that are reproducible against rowwise serial decode;
CPU may return a no-op scope. Keep one scope per backend class alive for the whole
serial replay or grouped publication region, and let destruction restore the
previous mode.

For shared-expert MoE verifier rows, the serial replay oracle must bypass normal
grouped shared-expert decode shortcuts unless those shortcuts are themselves
strictly proven equivalent. Compare grouped publication against canonical M=1
GEMM/SwiGLU/down replay, not against another unproven grouped shortcut.

Prioritize these paths:

- Quantized GEMV/GEMM for every M through the configured verifier capacity and
  all Q, K, and IQ codebook families.
- Fused gate/up and fused SwiGLU/down.
- LM head target/bonus rows without unnecessary all-position rows.
- Attention verifier rows.
- GDN recurrence/projection and short-conv rows.
- MoE routed experts and shared expert paths, but never fuse routed/shared
  logic unless strict parity proves it.

Dispatch policy must be trained/generated, not hand hardcoded:

- Sweep representative formats and M values.
- Train/select by codebook family, M, aspect ratio, and work-size buckets.
- Use exact shape winners only as overlays above the generic policy.
- Validate generated includes before installing them.
- Keep CPU, CUDA, and ROCm trainer behavior comparable through the common
  observation schema, exact oracle, learner, and certification gates.

For CPU, build ISA and effective runtime ISA are separate policy dimensions.
Train all three supported regimes: AVX2 build/AVX2 runtime, AVX512 build/forced
AVX2 runtime, and AVX512 build/AVX512 runtime. Thread count is also part of the
runtime key. Never train only the native ISA of the build and assume forced
runtime dispatch has identical economics. Keep oneDNN artifacts isolated in
`external/onednn/build-avx2` and `external/onednn/build-avx512` so configuring
one build cannot rewrite the dependency used by the other.

The canonical CPU production refresh is:

```bash
scripts/refresh_native_vnni_dispatch_tables.sh \
  --backend cpu \
  --profile all \
  --cpu-threads 28 \
  --cpu-measurement-lanes 2 \
  --cpu-avx2-sweep-bin build_v2_release/tests/v2/v2_perf_cpu_native_vnni_gemv \
  --cpu-avx512-sweep-bin build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_gemv \
  --output-dir benchmark_results/native_vnni_dispatch/<run-id> \
  --install
```

Use the physical cores per socket for `--cpu-threads` and the physical socket
count for `--cpu-measurement-lanes` on the blessed training host. The wrapper
assigns distinct even/odd shape shards to MPI ranks mapped one per socket and
runs the AVX512 binary twice with distinct runtime dispatch,
requires the complete ISA matrix, and permits `--install` only for a production
Qwen 3.6/all-shape profile. Retain the aggregate CSV, timing sidecar, common
observation CSV, generated include, and summary from every production run.
Also retain the profiler request/evidence manifests, authenticated
`*_profiler_features.csv`, and raw artifact directory; they are bound to the
observation and timing-sample digests and cannot be regenerated from a
different run.

After a CPU refresh, rebuild both binaries and run these gates:

```bash
cmake --build build_v2_release --parallel --target v2_perf_cpu_native_vnni_gemv
cmake --build build_v2_release_avx512 --parallel --target v2_perf_cpu_native_vnni_gemv
cmake --build build_v2_integration --parallel --target v2_integration_cpu_native_vnni_gemv
ctest --test-dir build_v2_integration -R "^V2_Integration_CPUNativeVNNIVerifierISAResolver$" --output-on-failure
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_CPU_AllFormats$" --output-on-failure
ctest --test-dir build_v2_integration -R "^V2_Integration_CPUNativeVNNI_GEMV$" --output-on-failure
```

The ISA resolver gate must validate production route counters for build ISA,
effective runtime ISA, thread count, and selected grouped policy. Missing
certification is a hard failure; do not add a Pairwise or serial-row fallback.
The strong CPU trainer also executes a two-projection fused descriptor bundle
twice in every format/M cell. It requires serial-row byte equality, repeat byte
equality, and the `cpu_native_vnni_fused_verifier_rows_projection_launch`
counter with a two-row AVX2 or four-row AVX512 physical K-part tile. This proof
must remain enabled in all three CPU regimes; a byte-equal trainer candidate
alone does not prove the production fused scheduler is economical.

If user asks whether to tune a single known model shape, do the experiment, but
convert any durable result into the general training pipeline before landing it.

## State Ownership And Coherence

Most serious MTP bugs come from split host/device ownership. Prefer device-owned
live state:

- KV and shifted MTP KV publication/truncation should be explicit and atomic.
- GDN recurrence and short-conv state should publish accepted rows from device
  speculative slots.
- Host mirrors may observe state for logging/tests, but must not be the source
  of truth for the hot path.
- Reset APIs must say whether they clear session state, graph captures, prefix
  cache, KV/GDN live state, MTP sidecar state, or benchmark-only scratch.
- Request reuse and `clearCache()` need dedicated regression tests.

When fixing coherence bugs, add a test that proves continuation after publication,
not just immediate logits.

## Sampling And Stochastic MTP

Stochastic support must be production-grade:

- CPU, CUDA, and ROCm samplers should share the same math contract.
- Temperature-zero greedy must still honor penalties and other applicable
  sampling parameters.
- GPU stochastic sampling should be device-resident and graph-capturable where
  possible.
- The verifier should avoid host token uploads in the hot path. Host-visible
  tokens are outputs, not the next-step source of truth.
- Qwen chat thinking-budget phrase injection must be honored; forcing only a
  stop token is not sufficient.

Test sampler parity on synthetic hard rows and real Qwen3.6 logits. Tie-breaking
must be deterministic and documented.

## Dynamic MTP Depth Controller

The dynamic controller is a policy layer over fixed-depth economics. Do not tune
it until fixed d1/d2/d3 lanes are healthy.

Maintain or generate controller policy from real data:

- Include short repetitive prompts, long prompts, code-generation prompts, and
  default benchmark prompts.
- Compare dynamic against fixed d1, d2, and d3 for each backend/model.
- Dynamic should asymptotically approach the best fixed depth for a prompt class.
- If backend preferences differ, first rule out math/parity drift; then explain
  the performance reason in the dashboard.
- Keep generated policy files reproducible and tied to holdout data. Avoid
  one-off hand tuning.

## Multi-Device Lanes

Track these modes in the dashboard even when implementation is pending:

- SingleDevice CPU/CUDA/ROCm.
- LocalTP CUDA deg2, ROCm deg2, ROCm deg4.
- LocalPP CUDA/ROCm.
- NodeTP CPU sockets.
- ExpertOverlay cases such as GPU hot plus CPU cold and mixed CUDA/ROCm/CPU.

Multi-device MTP must use common-prefix/common-accepted-count coordination.
Every participant enters collectives in the same order. Participants with no
work no-op through the same graph/collective sequence.

## Commit And Iteration Discipline

For each slice:

1. State the targeted bottleneck or correctness bug.
2. Add or update a regression test before or with the fix.
3. Run focused tests for the touched path.
4. Run the relevant MTP/prefix parity and perf gates.
5. Update the plan and dashboard.
6. Before a non-WiP commit, run the broader unit gate requested by the project
   plan or precommit hook and fix failures rather than bypassing them.

When a test sweep is large, resume from the last failure while iterating, then
run front-to-back once every listed test has passed at least once.
