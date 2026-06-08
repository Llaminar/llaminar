# vLLM-Style MTP Project Plan

## Objective

Port a vLLM-style MTP/speculative decoding architecture into Llaminar for
Qwen3.6 dense and MoE models on CUDA, ROCm, and CPU. SingleDevice is the first
acceptance target. Multi-device TP/PP/ExpertParallel follows only after the
SingleDevice contract is correct, fast, and covered by repeatable parity and
benchmark gates.

This replaces the old search for verifier-row shortcuts. The target is a clean
accepted-count state machine: draft state lives in speculative slots, target
verification produces accepted counts and output tokens, and only accepted
state slots are published to live model state.

## Why vLLM Is Fast

The local vLLM source shape to port is:

- `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`
- `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `vllm/v1/sample/rejection_sampler.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/model_executor/models/qwen3_5_mtp.py`
- `vllm/model_executor/models/qwen3_next_mtp.py`

The important ideas are:

- Draft proposal is graph-shaped: draft prefill and subsequent one-token draft
  steps use persistent input/state buffers and graph-capturable routines.
- Target verification is a `draft_count + 1` target forward: draft rows plus a
  bonus row, with logits indices describing target and bonus logits.
- Rejection sampling emits output tokens and accepted counts from device-side
  metadata. Greedy is just the deterministic special case.
- GDN/short-conv attention receives `num_accepted_tokens`,
  `spec_state_indices_tensor`, non-spec state indices, query starts, and token
  indices. Speculative state is isolated from live state.
- Full graph capture works because shapes are padded into persistent device
  tensors; no hot-path CPU sync is needed to discover accepted lengths.
- Publication is explicit: accepted speculative slots become live state; rejected
  suffix and bonus-ready rows do not mutate live recurrent/KV state.

Llaminar is slow today because dense and current CUDA MoE MTP still rely on a
decode-equivalent verifier replay path. For depth-1, that means two one-token
main forwards around each speculative pair, so the sidecar is cheap but the
verifier dominates runtime.

## Target Architecture

Add a device-agnostic MTP execution contract:

```cpp
struct MTPSpecStepPlan
{
    int draft_count;
    int target_rows;      // draft_count + 1
    int accepted_count;
    bool all_accepted;
};

class IMTPSpecStateBackend
{
public:
    virtual bool prepareSpecSlots(const MTPSpecStepPlan&) = 0;
    virtual bool runDraftGraph(const MTPSpecStepPlan&) = 0;
    virtual bool runTargetVerifierGraph(const MTPSpecStepPlan&) = 0;
    virtual bool publishAcceptedState(const MTPSpecStepPlan&) = 0;
    virtual bool discardRejectedState(const MTPSpecStepPlan&) = 0;
};
```

Implementation principles:

- Per-device graphs only; no nested multi-device sidecar graph.
- Every GPU operation uses an explicit non-null stream.
- Every GPU scratch allocation uses arena/workspace declarations and
  `IWorkspaceConsumer`; no ad-hoc kernel-owned caches.
- CPU, CUDA, and ROCm share the row/state publication planner and sampler math.
  Backend code only implements kernels and buffer binding.
- TransferEngine handles host/device movement where graph-stage contracts do not
  already provide device-resident buffers.
- Greedy and stochastic use the same state publication path.
- MoE sidecar uses graph-native MoE stages and sparse/replicated expert domains,
  not dense-only fallbacks.

## Current Status

Done:

- MTP config, sidecar loading, fixed/dynamic depth controller, benchmark JSON
  counters, and per-request MTP summaries exist.
- `MTPSpecDecodeMetadata`, workspace declarations, upload guards, transaction
  counters, and accepted-count publication planning units exist.
- `MTPSpecStateContract` now materializes per-request `MTPSpecStepPlan` objects
  from metadata plus publication plans, including global speculative slot
  validation for multi-request batches.
- `MTPSpecStatePublisher` drives accepted-row publication through existing
  verifier-captured `IComputeStage` state hooks, can publish directly from a
  `ComputeGraph` in execution order, and hard-fails GPU publication without an
  explicit stream.
- `ForwardExecutionEngine` exposes the exact last cached forward graph, and
  `DeviceGraphOrchestrator` now has a runner hook that publishes only from a
  just-run all-position verifier graph with matching verifier rows.
- `MTPVerifierPolicy` has an explicit all-position state-publication path.
  `OrchestrationRunner` now selects it for greedy and device-resident
  stochastic runners that advertise accepted-state publication and do not
  require decode-equivalent GDN replay.
- `MTPSpecKVPublisher` now truncates main KV plus shifted MTP KV caches to the
  accepted-count invariant; `DeviceGraphOrchestrator` folds KV publication into
  its verifier-graph publication hook and updates logical position state.
- `MTPDecodeCatchup` now has a shared all-position greedy verifier contract that
  maps verifier rows to draft tokens, marks the accepted state publication
  prefix, and isolates rejected correction replay.
- Greedy and stochastic all-position publication have focused runner unit
  coverage for accept-all and reject-with-correction-replay cases. The runner
  now commits the first shifted MTP KV row before the all-position verifier and
  commits any additional accepted verifier prefix rows before state publication,
  so `MTPSpecKVPublisher` remains an invariant checker/truncater rather than a
  hidden state synthesizer.
- Request/session reset now invalidates request-scoped prefill graph captures
  with `PrefillGraphRejectReason::SessionReset`, fixing CUDA reused-runner
  no-MTP determinism after `clearCache()` without relying on logits gather.
- CUDA/ROCm compact stochastic sampling kernels exist for top-k/top-p tables.
- `V2_Integration_GPUSamplingKernels` now includes Qwen3.6 real-logit-style
  seeded rows with close whitespace/code-token probabilities. CUDA and ROCm
  graph-captured distribution build, direct sample, and compact sample match
  the CPU canonical sampler on that fixture.
- Dense CUDA/ROCm greedy and stochastic have parity/smoke coverage.
- Dense CPU/CUDA/ROCm SingleDevice Prefix+MTP parity now shares one declarative
  18-case test surface, including prefix restore, split prefill, dynamic/fixed
  depth, no-MTP determinism, forward-only MTP, and stochastic verifier coverage.
- CPU stochastic MTP now uses the shared sampler probability/residual math on
  host for the decode-equivalent verifier path, while GPUs still hard-fail
  without device-resident stochastic verification.
- MoE CPU/CUDA/ROCm SingleDevice Prefix+MTP parity now shares one declarative
  15-case test surface for backend-neutral behavior, including stochastic
  verifier reuse after `clearCache()`; CUDA-only fused/grouped kernel assertions
  live in a separate path-guard suite.
- ROCm MoE stochastic parity no longer crashes or diverges after runner reuse.
  The fixed root causes were stale singleton MoE scratch bindings across
  workspace-manager ABA and ROCm shared-expert gate wrappers reading host-only
  gate tensors without ensuring device residency on the explicit HIP stream.
- CUDA and ROCm dense/MoE stochastic verifier parity now pass on the same
  all-position state-publication path.
- ShortConv1d and GDN recurrence stages now refresh their shared-kernel
  verifier workspace bindings from `onGraphReplayed()`, so captured verifier
  graph replay can publish accepted rows after normal/correction graphs have
  cleared stale bindings. `V2_Unit_GDNKernels` covers this regression.
- Fresh dense GPU release benchmarks prove the accepted-count path is
  speed-positive for greedy on both CUDA and ROCm: CUDA d1 is 56.91 vs 43.82
  tok/s and ROCm d1 is 41.44 vs 30.19 tok/s. Refreshed long seeded stochastic
  evidence shows matched CUDA/ROCm acceptance at 52 accepted and 12 rejected:
  CUDA d1 is 51.29 tok/s and ROCm d1 is 31.31 tok/s. ROCm stochastic is only
  barely speed-positive and remains below the CUDA-class win target.
- CUDA MoE greedy has parity/style coverage.
- The dead verifier-row publication hooks and tests were removed.

Open gaps:

- Real-model dense CPU benchmarks have not yet refreshed after enabling the
  all-position accepted-count publication path.
- CPU stochastic accepted-count publication is not yet implemented; CPU
  stochastic currently proves correctness through the decode-equivalent host
  verifier path and still needs speed evidence.
- GDN/short-conv speculative-slot publication is available through verifier row
  capture hooks and is now used by the GPU all-position publication path; CPU
  publication and broader benchmark evidence still need to catch up.
- ROCm MoE grouped-prefill workspace sizing/binding is fixed for focused
  SingleDevice greedy and stochastic MTP parity lanes. Fresh real-model
  benchmarks are speed-negative: greedy d1 is 68.09 vs 76.23 tok/s, stochastic
  d1 is 52.57 vs 76.23 tok/s. Profiles point at verifier/catch-up replay and
  stochastic sampler overhead, not sidecar generation. The ROCm MoE
  stage-breakdown parity lane passes but currently takes about 341s, so it is a
  performance/test-duration anomaly to reduce.
- CUDA MoE has fresh release evidence after verifier replay rebinding:
  baseline 132.95 tok/s, greedy d1 113.08 tok/s at 85.94% acceptance, and
  stochastic d1 seed123 99.50 tok/s at 64.06% acceptance. Perf counters show
  `main_verifier` now reaches graph replay, so the remaining blocker is true
  verifier/catch-up cost and stochastic acceptance, not replay warmup churn.
- CPU MoE stochastic benchmark lane still needs a fresh run.
- CPU vLLM-style state publication is not implemented or benchmarked.
- CUDA MoE MTP is still speed-negative and must reduce verifier/catch-up cost
  before acceptance. Stochastic also needs acceptance-policy tuning or depth
  policy integration for the default prompt class.
- CUDA and ROCm dense stochastic MTP now match acceptance under the same seed,
  but the generated token streams still differ at a few real-model samples
  while the real-logit-style sampler fixture passes. This points at full-model
  logits/state/perf differences rather than isolated sampler math.
- TP/PP/ExpertParallel MTP is out of scope until SingleDevice is green.

## Implementation Phases

### Phase A: Spec-State Contract

- Finalize `MTPSpecStepPlan` and the accepted-count publication planner.
  Status: semantic plan builder plus vector/graph stage-state publisher units
  landed. The runner hook now combines KV and verifier-captured recurrent
  publication, and greedy/stochastic decode path integration is unit-gated for
  accept-all and rejected-correction replay.
- Add CPU reference publication for KV, GDN recurrence, short-conv state,
  terminal logits/hidden, and sampler history.
- Unit-test accept-all, reject-first, reject-after-prefix, bonus-ready, stop,
  EOS, and prefix-restore cases.

### Phase B: Dense SingleDevice Backend Publication

- CUDA: implement graph-capturable speculative state slots and accepted-state
  publication kernels for KV, GDN, and short-conv.
- ROCm: implement the same backend contract with HIP kernels and explicit stream
  binding.
- CPU: implement the same contract with deterministic reference code first, then
  optimize.
- Replace dense decode-equivalent replay with the publication path only after
  parity passes.

### Phase C: Greedy And Stochastic Sampler Unification

- Centralize sampler math so CPU/CUDA/ROCm use the same probability rules.
- Keep greedy as a deterministic specialization of the same metadata path.
- Add MoE stochastic parity lanes before enabling stochastic MoE benchmark cells.

### Phase D: MoE SingleDevice

- Fix ROCm grouped-prefill workspace sizing/binding.
- Bring dense publication into Qwen3.6 MoE MTP blocks.
- Ensure shared/routed expert paths, routing metadata, and expert workspaces are
  graph-native and workspace-declared.
- Investigate CUDA MoE acceptance regression against old 90%+ captures.
- Remove the remaining MoE decode-equivalent verifier/catch-up replay cost from
  CUDA and ROCm before marking MoE MTP speed accepted.

### Phase E: Benchmark Acceptance

- Refresh the dashboard after every iteration.
- Green requires correctness plus speed-positive MTP against same-run no-MTP
  baseline, with CUDA and ROCm posting comparable sized wins.
- Compare CUDA against llama.cpp anchors and keep ROCm within the same class of
  speedup before considering backend acceptance.

## Iteration Gates

Run these before every WiP commit.

### Required Build

```bash
cmake --build build_v2_integration --parallel
cmake --build build_v2_release --parallel
```

### Hard Commit Gate

All broader Llaminar unit tests must pass and be fixed before a WiP commit:

```bash
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
```

### MTP Unit Gate

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Unit_(PrefixMTPConfig|MTPDepthController|MTPDecodeCatchup|MTPSpecDecodeTransaction|MTPSpecDecodeMetadata|MTPSpecStateContract|MTPSpecKVPublisher|MTPStateTransaction|MTPVerifierPolicy|MTPWeightManifest|MTPGraphConstruction|PrefillDecodeTransition|PrefillGraphCacheIntegration|ForwardExecutionEngineAdvanced)" \
  --output-on-failure --parallel
```

### Functional/Parity Gate

Run the relevant available lanes for any touched backend:

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Integration_Parity_Qwen36.*(PrefixMTP|Math|GraphStreamStress)|^V2_Integration_PrefixCacheMTP_Qwen36.*(GpuGraphs|Smoke|Prefix)" \
  --output-on-failure --parallel
```

This must cover, as applicable:

- Dense CPU/CUDA/ROCm greedy MTP and prefix restore.
- Dense CPU/CUDA/ROCm stochastic MTP. The CPU lane may use the host verifier;
  CUDA/ROCm must use device-resident stochastic verification.
- Seeded stochastic sampler parity for saved real-model logits must be symmetric
  across CPU/CUDA/ROCm so backend drift cannot hide behind aggregate counters.
- Dense CUDA/ROCm GPU graph smokes.
- MoE CPU/CUDA/ROCm layer-by-layer math prefill/decode parity.
- MoE CUDA greedy MTP parity/style tests.
- MoE CPU/CUDA/ROCm stochastic verifier parity and deterministic reuse after
  `clearCache()`.
- ROCm MoE ExpertOverlay parity remains separate from SingleDevice acceptance.

### Benchmark Gate

Refresh JSON/perf evidence for every available lane touched in the iteration:

```bash
cmake --build build_v2_release --parallel
./build_v2_release/llaminar2 benchmark -m /opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf -d cuda:0 --benchmark-json-output <out>/cuda_dense_base.json
./build_v2_release/llaminar2 benchmark -m /opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf -d cuda:0 --mtp --mtp-draft-tokens 1 --mtp-verify-mode greedy --benchmark-json-output <out>/cuda_dense_greedy.json
./build_v2_release/llaminar2 benchmark -m /opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf -d cuda:0 --mtp --mtp-draft-tokens 1 --mtp-verify-mode speculative-sampling --benchmark-json-output <out>/cuda_dense_stoch.json
```

Repeat for `rocm:0`, `cpu`, and the MoE model
`/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` when that backend is part
of the slice. Capture `LLAMINAR_PERF_STATS_JSON=<out>/perfstats.json` for at
least one representative MTP run per backend.

Update `docs/v2/MTP_VLLM_STYLE_TUNING_DASHBOARD.md` after every benchmark pass.
