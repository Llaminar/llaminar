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
  NativeQuantised dispatch work.
- `.agents/rocm-tuning/SKILL.md` for HIP/ROCm profiling, rocprof, ISA analysis,
  and generated NativeQuantised dispatch work.

### Captured GPU stage timing

For isolated verifier/kernel experiments, use the canonical
[production kernel replay procedure](../llaminar-testing/references/kernel-replay.md).
Keep draft histories, routes and initial state fixed across candidates, and
require serial/grouped byte equivalence through retained dirty replays.

For CUDA stage attribution without CUPTI, follow the CUDA skill's
`references/native-graph-events.md`. Its optional setup-only stage observer
joins canonical model stage names to native GPU event intervals; enable
`LLAMINAR_NATIVE_EVENT_TRACE_REQUIRE_STAGE_NAMES=1` and render the exact trace
cohort with `tests/v2/performance/kernels/native_graph_stage_report.py`.
The report includes GPU spans, covered intervals, exclusive node ownership,
nested stages and unowned graph plumbing. It describes the final completed
replay of each selected executable, not every speculative transaction.

CUDA conditional bodies remain opaque. An interval around an MTP controller
or conditional parent does not identify its draft/verifier/sampler/publisher
costs or execution counts. Do not disable MTP, flatten the parent, select eager
execution or use serial row replay to manufacture that attribution. Preserve
the production depth policy and prefix behavior, and require the paired
uninstrumented run's token IDs, live rows and draft/accept/reject/verifier work
to match. Report event/scheduling overhead separately. Concurrent and nested
stage intervals overlap; summing them is not accepted-token latency. The
current standalone observer is CUDA-only; common annotation metadata alone
does not establish HIP per-stage timing support.

For launch-count audits, distinguish retained graph inventory, actual native
submissions and device-selected useful work. Correlate executable generation,
participant and setup/warmup/measured request before counting. Conditional
children establish possible work, not body execution multiplicity. A terminal
draft ledger can omit physical capacity preparation; compare first-transaction
sidecar submissions with the selected device depth on every fresh admission.
Retaining a graph must not require executing all capacity slots again.

Shifted-prefill append counts are separate from main-prompt counts. Ragged
request banks need each request's own prefix authority and checked storage
origin; `NativeAllreduceRequestRows` expresses that native sum contract without
host readbacks or rebinding arena tensors. Never pass request zero's count as
a global prefix. Focused CUDA/ROCm replay gates must check actual native
useful-byte receipts, untouched inactive rows and large/small/empty/skewed
request transitions, alongside dense and MoE graph construction.

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
`expert_decode_assignment` and `expert_prefill_assignment` domain fields,
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
rg -n "VerifierRows|decode_equivalent|PrefixMTP|GPUSampling|NativeQuantised|QuantisedGemmSmallM" tests/v2
```

Common focused gates:

```bash
ctest --test-dir build_v2_integration -R "^V2_Unit_PrefillDecodeTransition$|^V2_Unit_MTP|^V2_Integration_GPUSamplingKernels" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_.*MTP|^V2_Integration_.*Prefix" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_.*VerifierRows|^V2_Integration_.*QuantisedGemmSmallM|^V2_Integration_.*NativeQuantised" --output-on-failure --parallel
```

### Mandatory all-format grouped verifier sweeps

These integration suites are the first line of defense for grouped verifier
decode. They prove that production grouped rows are byte-identical to serial
decode across the backend's advertised tensor/codebook formats. Run them after
any change to MTP verifier rows, NativeQuantised/GEMV dispatch, MoE routed/shared
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
`RuntimeM2To16` in `Test__CUDAGemmParity.cpp`; the low-level NativeQuantised case also
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

### Isolated NativeQuantised profiler evidence

NativeQuantised exact-overlay tuning has two physically separate evidence transactions.
Canonical candidate timing runs first, with production warmups and
sample-interleaved rounds. Only after those observations and raw timing samples
are immutable may `profiler_evidence.py` derive per-candidate requests.
`profiler_collectors.py` then gathers native Linux `perf_event_open`, Nsight
Compute, or rocprofiler metrics from separate isolated launches. Compatible
requests may share prepared fixture storage and a process, but every request
owns its exact counter region and full execution identity. Profiler replay
duration is never a timing label.

Every required profiler request needs one evidence record; unsupported candidates
need an explicit non-profile state. CUDA/ROCm candidates may be pipelines, so
retain quantization, producer, reducer, and epilogue dispatches independently
and in launch order. Optional counters unavailable on one architecture remain
typed unavailable fields. A missing tool, raw report, required counter, or
supported candidate record blocks production evidence completeness. Production
`scripts/refresh_native_quantised_dispatch_tables.sh --profile all` enables this gate
by default. Missing profiler evidence blocks installation; there is no bypass flag.

The required `*_profiler_features.csv` export is an authenticated join of the
canonical common-observation timing CSV, request manifest, and evidence
manifest. It retains pipeline-level canonical timing alongside one ordered row
per physical dispatch and that dispatch's own counters. Never select an exact winner from a
hand-joined profiler CSV or treat profiler replay duration as canonical
latency; a corpus/request/evidence digest mismatch must fail the refresh.

### Exact-overlay dispatch and retained evidence

Use the shared [NativeQuantised exact-overlay workflow](../nativequantised-gemm-tuning/SKILL.md).
Generic fitting, generic CV and policy-search scorer campaigns are retired.
Keep existing runtime rules for unseen shapes; measure and emit finite exact
choices for actual source formats, physical TP shards, phases and worker/ISA
regimes. Shared inventory and CMake registration own the sweep surface.

Serial M1 changes must be staged before collecting grouped M2..16 and M31
when grouped execution inherits their arithmetic or schedule. A changed exact
choice cannot alter fixed partitions or reduction order. Verify actual Auto
hits, mixed bundles, partial rows, reset and repeated captured execution.
A single-projection timing cannot stand in for the CPU fused expert operation.

Read ordinary-prefill anchors from the shared `prefill_matrix` module and
physical buckets from `PrefillGraphBuckets.def`, rather than copying numerical
bucket lists into this skill. CPU expert-prefill anchors and dense model-tier
budgets are distinct. Runtime dispatch remains total for unmeasured positive M.

Keep all candidate raw samples and isolated profiler evidence. Report useful
prepared-weight bandwidth relative to the measured device roofline for M1;
report grouped throughput and arithmetic intensity for M>1. Canonical latency,
not bandwidth alone or profiler duration, determines an exact winner.

Reuse authenticated complete measurement cells after interruption. Never
replace source-format proof with an invalid standalone small-K source tensor;
normalized prepared execution and native source loading are separate coverage.
Publish large evidence only through the optional `Llaminar/corpora` repository,
not the source tree, and never fetch it for ordinary builds or preflight.

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

Generate measured exact choices through the shared workflow:

- Sweep the declared formats, physical geometries, phases and live M values.
- Select by the complete runtime execution key, including M and N/K geometry.
- Retain installed generic decisions for unmeasured keys.
- Validate generated includes and actual Auto hits before installation completes.
- Use common observation, correctness, exact-oracle and profiler gates across
  CPU, CUDA and ROCm. Generic fitting and cross-validation remain retired.

For CPU, build ISA and effective runtime ISA are separate policy dimensions.
Measure all three supported regimes: AVX2 build/AVX2 runtime, AVX512 build/forced
AVX2 runtime, and AVX512 build/AVX512 runtime. Thread count is also part of the
runtime key. Never measure only the native ISA of the build and assume forced
runtime dispatch has identical economics. Keep oneDNN artifacts isolated in
`external/onednn/build-avx2` and `external/onednn/build-avx512` so configuring
one build cannot rewrite the dependency used by the other.

The canonical CPU production exact refresh is:

```bash
scripts/refresh_native_quantised_dispatch_tables.sh \
  --backend cpu --profile all --stage all \
  --threads 28 --cpu-list 0-27 \
  --cpu-avx2-sweep-bin build_v2_release_avx2/tests/v2/v2_perf_cpu_native_quantised_gemv \
  --cpu-avx512-sweep-bin build_v2_release_avx512/tests/v2/v2_perf_cpu_native_quantised_gemv \
  --fused-avx2-bin build_v2_release_avx2/tests/v2/v2_perf_cpu_fused_verifier \
  --fused-avx512-bin build_v2_release_avx512/tests/v2/v2_perf_cpu_fused_verifier \
  --output-dir benchmark_results/native_quantised_dispatch/<run-id> \
  --install
```

Choose `--threads` and `--cpu-list` from the actual physical-core topology;
the example uses one 28-core socket. Collection validates actual placement.
If native counter permissions require it, add `--sudo-tool` to elevate only
the isolated counter owner through existing noninteractive sudo; do not change
host perf policy. Append `--resume` to continue the same immutable plan.

The CPU transaction measures AVX2 build/AVX2 runtime, AVX512 build/forced AVX2,
and AVX512 build/AVX512 runtime. It stages serial M1 exact choices and rebuilds
their native owners before grouped measurement. Compatible expert requests
share prepared weights/storage and a persistent timing team; each request
retains its live descriptor vector, raw samples and phase identity. Grouped
and expert includes are then rebuilt and verified through actual Auto lookup,
selected-policy and byte checks. A bounded partial checkpoint is not complete.

Retain the original native JSON/CSV and timing sidecars, common observations,
full-key/vector witnesses, generated includes, producer closures, profiler
request/evidence manifests, raw counter artifacts and resumable receipts.
The CPU plan explicitly identifies ordinary BackendNative prefill's missing
exact installation boundary; fused GPUAlignedExpert prefill evidence does not
certify that separate surface. No generic program is fitted or replaced.

After a CPU refresh, rebuild both binaries and run these gates:

```bash
cmake --build build_v2_release --parallel --target v2_perf_cpu_native_quantised_gemv
cmake --build build_v2_release_avx512 --parallel --target v2_perf_cpu_native_quantised_gemv
cmake --build build_v2_integration --parallel --target v2_integration_cpu_native_quantised_gemv
ctest --test-dir build_v2_integration -R "^V2_Integration_CPUNativeQuantisedVerifierISAResolver$" --output-on-failure
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_CPU_AllFormats$" --output-on-failure
ctest --test-dir build_v2_integration -R "^V2_Integration_CPUNativeQuantised_GEMV$" --output-on-failure
```

The ISA resolver gate must validate production route counters for build ISA,
effective runtime ISA, thread count, and selected grouped policy. Missing
certification is a hard failure; do not add a Pairwise or serial-row fallback.
The strong CPU trainer also executes a two-projection fused descriptor bundle
twice in every format/M cell. It requires serial-row byte equality, repeat byte
equality, and the `cpu_native_quantised_fused_verifier_rows_projection_launch`
counter with a two-row AVX2 or four-row AVX512 physical K-part tile. This proof
must remain enabled in all three CPU regimes; a byte-equal trainer candidate
alone does not prove the production fused scheduler is economical.

If user asks whether to tune a single known model shape, do the experiment, but
add its durable exact choice through the shared inventory and exact-overlay transaction before landing it.

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
