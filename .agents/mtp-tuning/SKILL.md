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

- `docs/v2/MTP_VLLM_STYLE_PROJECT_PLAN.md`: phase plan, accepted architecture,
  gates, and known debt.
- `docs/v2/MTP_VLLM_STYLE_TUNING_DASHBOARD.md`: compact RAG status and latest
  speed/correctness evidence. Keep it under the documented size limit.
- `docs/v2/PREFIX_CACHE_MTP_BENCHMARK_NOTES.md`: only for broader historical
  benchmark notes when the current dashboard is not the right home.

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
sidecar graphs. LocalTP, LocalPP, NodeLocalTP, and ExpertOverlay must extend the
same transaction semantics with collective coordination, not invent separate
state machines.

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
- Never leave quiet fallbacks. Unsupported MTP lanes must fail fast and loudly
  in tests or bypass with explicit counters only when the project plan says the
  lane is not implemented yet.
- Never land special codebook exceptions in production kernels. Use generic
  dispatch keyed by codebook family, M, aspect ratio, work size, and generated
  policy tables. Exact shape overlays are allowed only above a general fallback.
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

- Relative L2: tight enough to catch drift for the tested precision/path.
- Cosine similarity: near 1.0 for logits, hidden rows, and kernel outputs.
- Symmetric KLD: required when outputs feed sampling or softmax decisions.
- Max absolute error: required for small-M kernel equivalence.
- Sampled-token equality: required, but not sufficient by itself.
- Decode-equivalent continuation: accepted-state publication must match serial
  decode after continuing for enough rows to expose KV/GDN/short-conv mistakes.

For grouped verifier rows, prove M=1,2,3,4 against serial decode for every
backend that claims support. If a grouped path is not faster, keep it out of the
hot path and track the performance debt in the plan/dashboard.

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

Do not run only the easy backend. If CUDA has a deep PyTorch or layer-by-layer
test, ROCm and CPU need the same semantic coverage unless the plan explicitly
marks the lane as not implemented.

## Performance Methodology

Start with real inference, then isolate:

1. Capture a Release benchmark for the relevant lane with fixed d1/d2/d3 and
   dynamic depth.
2. Enable perfstats/stage timing only to rank bottlenecks. Profiling may disable
   graph capture, so do not quote profiled throughput as production speed.
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

## Grouped Kernel Tuning

MTP only becomes economical when grouped verifier work is genuinely grouped.
Wrapping serial decode rows under a grouped API is not enough.

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

- Quantized GEMV/GEMM M=2..4 for all Q, K, and IQ codebook families.
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
- Keep CUDA and ROCm trainer behavior comparable.

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
- NodeLocalTP CPU sockets.
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
