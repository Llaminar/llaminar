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
- Depth >1 all-position publication now accepts a target verifier with a bonus
  row beyond the accepted prefix. Focused `V2_Unit_PrefillDecodeTransition` and
  `V2_Unit_MTPGraphConstruction` coverage proves partial-prefix publication can
  commit accepted shifted rows without falling back to sequential verifier
  replay.
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
- Bounded dense iteration matrix now covers CUDA/ROCm/CPU, greedy/stochastic,
  baseline, fixed d1/d2/d3, and dynamic at 16 decode tokens. Greedy is
  speed-positive on all three backends, with best lanes CUDA d3 89.13 vs 44.60
  tok/s, ROCm d1/dynamic about 45.4 vs 31.3 tok/s, and CPU d3 8.51 vs 4.55
  tok/s. Stochastic is speed-negative on all three backends on this short
  seeded lane.
- CUDA MoE greedy has parity/style coverage.
- CUDA MoE MTP sidecar M=1 now uses the same grouped-prefill contract as
  verifier M=2..4, avoiding the fragile runtime grouped-decode chain inside
  captured MTP sidecar graphs. The former fixed-d3 Release crash repro passes,
  and compute-sanitizer reports 0 errors on that lane.
- CUDA MoE rejected-token correction replay now treats accepted-state
  publication as a graph replay-state boundary. Accept-all steps may preserve
  captured verifier replay, but steps that require correction replay reset
  captured GPU replay and kernel dynamic state before the following main decode
  graph. `Qwen36MoECUDASingleDevicePrefixMTPPathGuards.Depth1CorrectionReplayResetsCapturedStateBoundary`
  covers the former fixed-d1 crash.
- `scripts/run_mtp_iteration_benchmark_matrix.sh` now has `--decode-tokens N`
  for bounded all-device iteration sweeps. The default remains the full
  benchmark decode length.
- The dead verifier-row publication hooks and tests were removed.

Open gaps:

- Full default-length CPU dense and CPU MoE matrix refreshes remain slow
  acceptance work. Bounded CPU dense now has evidence, but the CPU lanes still
  take minutes even at 16 decode tokens.
- CPU stochastic accepted-count publication is not yet implemented; CPU
  stochastic currently proves correctness through the decode-equivalent host
  verifier path. Latest bounded evidence is speed-negative: best fixed d3 is
  3.62 vs 4.62 tok/s and still rolls back.
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
- CUDA MoE stable matrix now covers production greedy and seeded stochastic
  baselines plus fixed d1/d2/d3 and dynamic depth. Greedy baseline is 133.78
  tok/s; d1/d2/d3/dynamic are 88.49/94.39/115.94/88.75 tok/s. Stochastic
  seed123 baseline is 133.51 tok/s; d1/d2/d3/dynamic are
  91.26/93.72/86.47/97.86 tok/s. Depth >1 no longer crashes on partial-prefix
  publication or MTP sidecar graph replay, but CUDA MoE MTP remains
  speed-negative. The remaining blocker is true verifier/catch-up cost;
  dynamic depth also needs enough evidence or hysteresis tuning to promote
  during short default runs.
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

Refresh JSON/perf evidence for the SingleDevice device matrix on every tuning
iteration: CUDA, ROCm, and CPU; dense and MoE; greedy and stochastic; no-MTP
baseline, fixed d1, fixed d2, fixed d3, and dynamic depth. Greedy rows use
production runtime settings with `--temperature 0`, not `--deterministic`;
stochastic rows use a pinned seed, default `123`, so acceptance and throughput
can be compared across iterations. The generated `summary.tsv` includes
`speedup_vs_baseline` for every MTP row.

Use `cpu:0` for the SingleDevice CPU lane. Bare `cpu` auto-selects two-socket
CPU TP and belongs to a later multi-device/TP matrix, not this gate.

```bash
cmake --build build_v2_release --parallel
scripts/run_mtp_iteration_benchmark_matrix.sh --perfstats
```

The full default matrix is the acceptance capture. For inner-loop tuning, keep
the same device/model/mode/variant shape but bound the decode length so CPU and
MoE lanes remain practical:

```bash
scripts/run_mtp_iteration_benchmark_matrix.sh \
  --decode-tokens 16 --perfstats
```

For narrow diagnostic loops, keep the same variant shape while selecting the
lane under active work. These runs can guide a fix, but they do not replace the
bounded or full matrix capture for iteration evidence:

```bash
scripts/run_mtp_iteration_benchmark_matrix.sh \
  --devices cuda:0 --models moe --modes greedy,stochastic \
  --variants baseline,fixed_d1,fixed_d2,fixed_d3,dynamic --perfstats
```

Update `docs/v2/MTP_VLLM_STYLE_TUNING_DASHBOARD.md` after every benchmark pass.
