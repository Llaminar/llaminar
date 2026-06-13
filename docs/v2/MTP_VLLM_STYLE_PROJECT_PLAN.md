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

Llaminar is slow today when it diverges from that shape: dense greedy still pays
extra verifier work through batched all-position LM-head rows, stochastic evidence
still includes stepwise/decode-equivalent cost on some lanes, and MoE verifier
paths are functionally green but dominated by target-forward and condition
forward time.

## Target Architecture

The target is a first-class speculative decode transaction, not a collection of
runner fallbacks. The transaction owns all metadata, draft rows, verifier rows,
sampling decisions, and accepted-state publication for one decode step.

```cpp
struct MTPSpecPersistentMetadata
{
    DeviceBuffer<int32_t> draft_token_ids;        // [requests, max_draft]
    DeviceBuffer<int32_t> target_logits_indices;  // flattened draft rows
    DeviceBuffer<int32_t> bonus_logits_indices;   // one per request
    DeviceBuffer<int32_t> num_draft_tokens;       // [requests]
    DeviceBuffer<int32_t> num_accepted_tokens;    // [requests]
    DeviceBuffer<int32_t> spec_state_indices;     // [requests, max_draft + 1]
    DeviceBuffer<int32_t> non_spec_state_indices;
    DeviceBuffer<int32_t> token_indices;
};

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
    virtual bool runRejectionSampler(const MTPSpecStepPlan&) = 0;
    virtual bool publishAcceptedState(const MTPSpecStepPlan&) = 0;
    virtual bool discardRejectedState(const MTPSpecStepPlan&) = 0;
};
```

Subsystem boundaries:

- `MTPSpecTransactionDriver`: one device-agnostic coordinator used by CPU, CUDA,
  ROCm, dense, and MoE. It replaces decode-equivalent verifier branches.
- `MTPSpecPersistentMetadata`: vLLM-style padded buffers for draft tokens,
  target/bonus logit rows, accepted counts, state indices, sequence/query
  starts, and masks. GPU buffers live in workspace/arena allocations; CPU uses
  the same layout in host buffers.
- Draft graph: graph-shaped prefill plus one-token draft steps using persistent
  input ids, positions, hidden state, MTP KV, and optional draft probability
  output. Draft sampling is fused when the backend supports it.
- Target verifier graph: one `draft_count + 1` target forward. It produces only
  the verifier rows needed by `target_logits_indices` and `bonus_logits_indices`;
  computing a full all-position LM head is a compatibility path, not the target.
- Rejection sampler: greedy is a deterministic specialization of the stochastic
  contract. The accepted stochastic target follows vLLM's worker fast path:
  draft proposal is greedy by default, so verifier `q` is one-hot at the draft
  token (`NO_DRAFT_PROBS` in vLLM terms); target rows are processed once, the
  first rejected or bonus row is sampled on device, and full draft
  probabilities are only an optional future lane for genuinely stochastic draft
  proposal. Compact top-k/top-p tables are compatibility scaffolding, not the
  MoE performance target.
- State publication: KV, shifted MTP KV, GDN recurrence, short-conv state,
  terminal hidden/logits, sampler history, and positions are published from
  accepted speculative slots. Rejected suffix and bonus-only rows never mutate
  live state.
- Backend layer: CPU/CUDA/ROCm implement buffer binding and kernels only. The
  accepted-count planner, metadata semantics, stochastic math, and transaction
  state machine are shared.
- MoE graph layer: routed/shared expert execution is graph-native and uses the
  same metadata. Expert routing scratch is transient; only continuation state is
  publishable.

Non-negotiable invariants:

- Per-device graphs only; no nested multi-device sidecar graph.
- Every GPU operation uses an explicit non-null stream.
- Every GPU scratch allocation uses arena/workspace declarations and
  `IWorkspaceConsumer`; no ad-hoc kernel-owned caches.
- TransferEngine handles host/device movement where graph-stage contracts do not
  already provide device-resident buffers.
- Fallbacks are temporary migration scaffolding only. Once a path is replaced
  and parity/perf accepted, the dead code and tests are removed.

Current Llaminar shape versus target:

| Area | Current shape | vLLM-shaped target |
|------|---------------|--------------------|
| Greedy dense | All-position verifier publication exists and is speed-positive | Row-indexed verifier logits and graph transaction |
| Stochastic dense | Batched sampler contract is accepted for SingleDevice CPU/CUDA/ROCm; performance remains policy-sensitive | Publish compact outcome/state fully from spec slots |
| CPU verifier | Full target forward plus batched all-position LM-head rows | Target/bonus row LM head and shared metadata buffers |
| MoE | Functionally green, speed-negative | Graph-native sidecar plus batched verifier/rejection |
| State publication | Captured-stage restore works | Spec state slots are the primary live-state mechanism |
| Graph capture | Improving per backend | Draft, verify, sample, publish are captured where possible |

## Current Status

Done:

- MTP config, sidecar loading, fixed/dynamic depth controller, per-request MTP
  summaries, and benchmark JSON/table reporting exist. Benchmark mode now
  aggregates request-scoped MTP counters across measured iterations after
  warmup and emits `measurement_iterations`, so acceptance/rollback counters
  match averaged throughput.
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
- ROCm dense no-MTP fresh-runner determinism is fixed. Deterministic parity now
  bypasses ROCm flash-decode autotune trial rotation, because the autotuner is
  performance state rather than model state. Focused ROCm no-MTP determinism,
  ROCm MTP forward-only parity, and CUDA symmetry checks passed.
- CUDA/ROCm compact stochastic sampling kernels exist for top-k/top-p tables.
- `V2_Integration_GPUSamplingKernels` now includes Qwen3.6 real-logit-style
  seeded rows with close whitespace/code-token probabilities. CUDA and ROCm
  graph-captured distribution build and compact-table sample match the CPU
  canonical sampler on that fixture. Top-k ties are now a documented
  value-descending/token-id-ascending contract in CUDA, ROCm, and the CPU test
  oracle; this fixed the ROCm stochastic clear-cache repeatability drift where
  equal quantized logits could produce different draft candidate orderings.
- Dense CUDA/ROCm greedy and stochastic have parity/smoke coverage.
- Dense CPU/CUDA/ROCm SingleDevice Prefix+MTP parity now shares one declarative
  18-case test surface, including prefix restore, split prefill, dynamic/fixed
  depth, no-MTP determinism, forward-only MTP, and stochastic verifier coverage.
- Dense Qwen3.6 SingleDevice now also has a classic layer-by-layer math suite
  matching the Qwen3.5/Qwen3.6 MoE parity style. CPU/CUDA/ROCm prefill, decode,
  and snapshot infrastructure all pass with shared PyTorch snapshots, cosine
  thresholds of 0.96 prefill and 0.93 decode, and all first 8 layers gated.
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
- ROCm combined routed+shared verifier grouping now handles Qwen3.6 MoE's
  256 routed experts plus one logical shared-expert slot. The grouping kernel
  publishes counts/offsets with a strided loop, so slot 256 is not silently
  dropped by a 256-thread block. `V2_Integration_ROCmMoEKernel` and focused
  ROCm Qwen3.6 MoE greedy/stochastic/all-position parity gates pass.
- CUDA and ROCm dense/MoE stochastic verifier parity now pass on the same
  all-position state-publication path.
- vLLM-style stochastic verification is wired into the GPU SingleDevice runner
  path using processed target logits plus device-resident sampled draft tokens.
  Draft proposals follow the vLLM default greedy draft branch, so the verifier
  treats `q` as one-hot instead of allocating full draft probability rows.
  Penalty-free and penalty-sensitive rows batch through the same processed-logit
  outcome verifier; penalty-sensitive rows pre-apply the vLLM speculative branch
  history per verifier row. Host-visible sampled tokens still keep their
  arena-owned device sample-slot readiness edge, so the batch verifier consumes
  device tokens instead of re-uploading host shadows. Focused runner units,
  `V2_Unit_MTPRejectionSampler`, `V2_Integration_GPUSamplingKernels`, and
  Qwen3.6 CUDA/ROCm dense+MoE stochastic parity pass after a full relink.
- The latest GPU stochastic matrix on the vLLM greedy-draft/one-hot-q path is
  `benchmark_results/mtp_vllm_style/20260612T170149Z-gpu-stochastic-vllm-greedyq-c4096-post-moe-workspace/`.
  Dense is speed-positive on CUDA and ROCm: CUDA baseline/d1/d2/d3/dyn is
  44.7/52.6/52.6/47.7/47.7 tok/s and ROCm is
  31.3/37.0/28.7/27.0/36.9 tok/s. MoE stochastic is correctness-green but
  still performance-red: CUDA is 115.3/78.5/76.0/79.0/78.3 tok/s and ROCm is
  69.1/51.8/48.5/40.7/51.6 tok/s. The next MoE stochastic slice must reduce
  verifier/condition/sampling economics rather than returning to compact
  top-k/top-p shortcuts.
- Diagnostic probabilistic draft proposals were benchmarked in
  `benchmark_results/mtp_vllm_style/20260612T175223Z-moe-stochastic-probabilistic-draft-smoke/`.
  They improve CUDA d1 acceptance to 75% but remain speed-negative
  (CUDA 77.6 vs 115.2 tok/s baseline; ROCm 49.1 vs 67.7 tok/s baseline), so the
  accepted architecture remains vLLM's default greedy draft proposal with
  one-hot `q`. The MoE work stays focused on verifier, condition, and outcome
  costs.
- GPU stochastic MTP now follows the vLLM default draft branch: draft proposal
  uses device argmax, the processed-target verifier treats `q` as one-hot
  (`no_draft_probabilities=true`), and null draft-probability buffers hard-fail
  unless that contract is explicit. Focused coverage passed for
  `V2_Integration_GPUSamplingKernels`, `V2_Unit_MTPRejectionSampler`,
  `V2_Unit_PrefillDecodeTransition`, and Qwen3.6 CUDA/ROCm stochastic graph
  smokes. Dense and MoE CUDA/ROCm
  `MTPStochasticSamplingVerifierRuns` parity also passes on the new path. The
  production proxy in `V2_Perf_GPUSpeculativeSummary` shows rows=3 CUDA
  probability 3.96/3.81/3.69 ms improved to greedy-q 1.76/1.76/2.13 ms, and
  ROCm probability 8.91/8.93/8.96 ms improved to 4.79/4.78/5.67 ms for
  reject0/prefix1/all. Full-model matrices confirm dense speedups but not MoE
  speedups yet.
- CUDA MoE decode now declares workspace for Qwen3.6 top-k=8 gate/up fused
  fan-out. That path launches 16 logical projections but only eight active CUDA
  stream slots, so the stage reserves seven side-stream GEMV partial arenas.
  This fixes the former `[ConcurrentDecode] ... got 16` hard failure without
  reintroducing LM-head-sized global decode scratch. Regression:
  `V2_Unit_CUDAQuantisedGemmWorkspace`.
- Legacy full target/draft probability arena rows were removed from production
  `DeviceGraphOrchestrator` state after the vLLM greedy-draft/one-hot-q path
  became the accepted GPU stochastic contract. Compact distribution builds no
  longer materialize hidden full-softmax side rows, and the old scalar
  full-probability device verifier is no longer implemented by GPU runners.
  Guards passed: `V2_Unit_GpuWorkspaceAllocationPolicy`,
  `V2_Unit_PrefillDecodeTransition`, release `llaminar2`, and a CUDA Qwen3.6
  MoE d1 stochastic smoke whose perf JSON emitted processed-logit batch verifier
  counters with no full-probability rows.
- A lower-memory draft-logit proposal/verifier branch is now implemented as a
  backend/perf primitive for CUDA and ROCm, with graph-captured integration
  coverage against the CPU inverse-exponential sampler oracle. It is not
  production-promoted: focused `V2_Perf_GPUSpeculativeSummary` shows mixed CUDA
  movement and clear ROCm regression versus the existing processed-target plus
  draft-probability path, especially reject-at-prefix-0 rows. This proves the
  next vLLM-aligned win is not simply "store logits instead of q"; it must fuse
  or lazily skip target/draft probability-stat work that cannot affect the
  accepted prefix.
- A one-block prefix-stop verifier was also implemented, measured, and removed
  in the same slice. It was graph-capturable and correct, but focused perf only
  helped CUDA reject-at-prefix-0 and regressed prefix-1/all-accepted cases; ROCm
  reject-at-prefix-0 was effectively flat and deeper prefixes regressed. Do not
  reintroduce a serial prefix verifier without changing the larger target-logit
  materialization economics.
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
- Bounded dense iteration matrix covers CUDA/ROCm/CPU, greedy/stochastic,
  baseline, fixed d1/d2/d3, and dynamic at 16 decode tokens. Latest full matrix
  is `benchmark_results/mtp_vllm_style/20260609T061226Z-iteration-matrix-6753b5e7/`.
  Greedy is speed-positive on all three backends, with best lanes CUDA d3 66.7
  vs 44.6 tok/s, ROCm d1 43.9 vs 31.4 tok/s, and CPU d3 9.1 vs 4.6 tok/s.
  Stochastic is speed-negative on all three backends on this short seeded lane.
- Dynamic depth now has production-shaped exploration: standard matrix runs
  start at d1 with d1 as the adaptive floor, while d0 remains a diagnostic
  bypass lane. Demotion is stepwise, depth 1 only demotes to d0 after an all-zero
  diagnostic window, perfect probes can promote early, floor-depth windows must
  meet the promotion threshold before exploring, and a bad intermediate depth
  probes each un-rejected deeper depth once before settling downward.
  Focused controller and runner regressions cover d0 cooldown/probe,
  shifted-cache maintenance, stepwise demotion, perfect-probe promotion,
  rejected-depth hysteresis, and d2-bad/d3-untested exploration. The post-tune bounded matrix
  `benchmark_results/mtp_vllm_style/20260609T-post-hysteresis-tune-matrix/`
  completed all CUDA/ROCm/CPU dense/MoE greedy/stochastic lanes with fixed
  d1/d2/d3/dynamic. Dense greedy is still speed-positive; dynamic is less
  cliffy but remains short-run conservative versus the best fixed depth.
- `V2_Perf_MTPDepthController` now characterizes dynamic policy overhead. The
  controller is scalar counter bookkeeping, measuring about 16-26 ns per update
  in Release, so CPU dynamic-depth tuning should target verifier, condition, and
  accepted-state publication costs rather than threading or vectorizing the
  controller itself.
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
- CUDA MoE correction replay now narrows that replay reset to ordinary decode
  graphs while dirtying explicit stream bindings on preserved verifier captures.
  Bounded CUDA MoE diagnostics show verifier replay is now exercised for fixed
  d1/d2/d3/dynamic lanes, but MoE remains speed-negative because verifier plus
  correction time still dominates.
- Ordinary decode segmented captures now carry a live replay-state epoch.
  `DeviceGraphOrchestrator` advances that epoch after live-prefix/speculative
  state publication, and `ForwardExecutionEngine` recaptures stale ordinary
  decode graphs before replay while leaving all-position verifier graphs under
  their accepted-state publication contract. Focused cache/engine units plus
  CPU/CUDA/ROCm MoE stochastic verifier and depth-3 greedy parity passed; the
  release fixed-d3 sanity check
  `benchmark_results/mtp_vllm_style/20260609T082255Z-gpu-moe-d3-versioned-replay/`
  preserved acceptance instead of reproducing the stale-capture collapse.
- Replay-state mutation policy is now an explicit typed contract rather than
  an implicit `decode && !all_position` check. `ForwardExecutionEngine` exposes
  read-only replay-cache observations for tests/diagnostics, and focused units
  assert the correction boundary resets ordinary decode replay while preserving
  verifier replay only for stream rebinding. The bounded deep correctness gate
  passed on CPU/CUDA/ROCm dense and MoE depth-3 greedy plus stochastic verifier
  parity.
- `DeviceGraphOrchestrator` now exposes read-only live replay-state epoch and
  replay-cache observations, wired to the orchestrator's current epoch. A
  focused CPU unit proves live checkpoint restore advances the state-version
  contract without marking CPU graph-cache identities stale, and the same deep
  CPU/CUDA/ROCm dense+MoE depth-3 greedy/stochastic verifier parity guard passed
  after the diagnostic hook landed.
- CPU stochastic all-position publication now uses the same accepted-state
  publication contract as CUDA/ROCm, with host-side target/draft distributions
  built from the shared sampler probability and residual math. Focused runner
  units cover host accept and reject/correction cases, and the backend-symmetric
  dense+MoE CPU/CUDA/ROCm depth-3 greedy plus stochastic verifier parity gate
  passed after the parity contract was tightened to reject the old
  decode-equivalent fallback.
- CPU hybrid state export/import now builds deterministic host-copy spans and
  copies large recurrence/short-conv payloads through the existing OpenMP
  workshare pattern. CPU accepted-state publication also restores independent
  verifier-captured stages in parallel while keeping GPU publication ordered on
  the explicit stream. `V2_Unit_HybridKVCache`, `V2_Unit_MTPSpecStateContract`,
  and `V2_Unit_PrefillDecodeTransition` cover the parallel copy and publication
  contracts.
- CPU sampler top-k distribution building now uses an ISA-dispatched
  scalar/AVX2/AVX512 top-k primitive in the Qwen chat/MTP compact top-k path,
  avoiding the former full-vocab `pair` allocation plus `partial_sort`.
  `V2_Unit_Sampler` proves scalar, AVX2, and AVX512 top-k equivalence and
  distribution parity with the old partial-sort baseline. `V2_Perf_CPUSamplerTopK`
  on a 151,936-token Qwen-style vocabulary measured old/new distribution build
  times of 0.190846/0.020120 ms for top-k 20, 0.187209/0.028611 ms for top-k
  40, and 0.262855/0.225518 ms for top-k 256.
- CPU dense Qwen3.6 Prefix/MTP parity no longer spins up redundant no-MTP
  baseline runners inside prefix restore, split-prefill, fixed/dynamic MTP, or
  stochastic verifier helpers when the PyTorch decode token fixture already
  provides the correctness oracle. Focused CTest reruns show the former slow
  cells now cluster around 41-43s: split prefill 42.63s, fixed d3 MTP 41.81s,
  stochastic verifier 43.15s, and dynamic MTP 42.55s. Dedicated no-MTP and
  determinism tests remain intact.
- Rejected-token all-position publication no longer runs an expensive same-step
  correction main forward. The runner now emits the correction token, commits
  its shifted MTP row from current terminal hidden so the sidecar cache remains
  aligned, records `deferred_correction_condition_tokens`, and defers the
  correction token's main-model condition forward to the next ordinary decode
  step. Focused units and
  `Qwen36MoECUDASingleDevicePrefixMTPPathGuards.Depth1RejectedCorrectionDefersToConditionToken`
  cover this split; fresh CUDA/ROCm GPU matrices show `correction_ms=0` and
  zero rollback, but MoE remains speed-negative because verifier time dominates.
- Greedy GPU all-position verifier replay can now defer the verifier graph's
  final stream sync and hand the capture stream directly to device-side row
  sampling. The handoff is scoped by `OrchestrationRunner`, hard-fails if a
  backend cannot sample on the handed-off stream, and is deliberately disabled
  for stochastic verification until its multi-kernel distribution path has the
  same stream contract. `V2_Unit_PrefillDecodeTransition`,
  `V2_Unit_ForwardExecutionEngine`, `V2_Unit_MTPGraphConstruction`, and focused
  CUDA/ROCm Qwen3.6 MoE d3+stochastic parity passed. Release check
  `benchmark_results/mtp_vllm_style/20260609T085703Z-gpu-moe-d3-verifier-stream-handoff/`
  shows the counters firing with zero rollback/correction replay, but still no
  MoE speed-positive result: CUDA d3 70.3 vs 109.8 tok/s and ROCm d3 42.4 vs
  64.7 tok/s. The CPU Qwen3.6 MoE d3+stochastic parity cells also passed as
  the non-GPU replay-state guard.
- All-position publication no longer captures the old post-sidecar prefix
  checkpoint, because that verifier path publishes from the just-run target
  graph and never restores the sidecar checkpoint. The decode-equivalent path
  keeps its checkpoint until CPU/host verifier publication is replaced.
  `V2_Unit_PrefillDecodeTransition` covers the skip counter and absence of the
  old checkpoint timer; the fresh bounded matrix confirms GPU all-position lanes
  emit `post_sidecar_checkpoint_skipped_all_position_publication` instead.
- CUDA MoE graph-captured no-MTP baseline decode crash is fixed. Root cause was
  a split-K down-partials workspace contract mismatch plus missing expert-id
  upper-bound guards in CUDA MoE grouped k-part kernels. The focused regression
  `RuntimeRouteSelectAndFusedDecodeCaptureWithLargeExpertTable` now captures
  route selection plus fused grouped expert decode against a Qwen3.6-scale
  expert table.
- ROCm MoE shared-expert grouped prefill preparation is now graph-native enough
  for verifier/shared-expert grouped routes. It declares workspace buffers,
  prepares implicit shared-expert group metadata on the explicit HIP stream, and
  has focused `V2_Unit_PrefillGraphCapturability` plus
  `ROCmMoEKernel.SharedExpertGroupedPrefillMatchesSequentialPath` coverage.
- ROCm and CUDA MoE verifier-prefill now use graph-capturable combined
  routed/shared verifier launches for Qwen3.6-scale M=2/3/4 rows. Focused
  production-shape perf correctness is cosine 1.0 on both backends. Fresh
  focused timings from `V2_Perf_MoEVerifierPrefill` show CUDA graph M=2/3/4 at
  about 0.117/0.151/0.176 ms and ROCm graph M=2/3/4 at about
  0.237/0.266/0.303 ms, so isolated expert-prefill kernels are no longer the
  only MoE blocker.
- ROCm GDN concurrent decode is now promoted to the default outside
  deterministic mode. Focused coverage `V2_Unit_GDNKernels` and
  `V2_Unit_DeterministicMode` proves the default and deterministic override.
  No-env probe
  `benchmark_results/mtp_vllm_style/20260612T_rocm_moe_gdn_default_probe/`
  shows ROCm MoE greedy fixed d3 at 92.1 tok/s versus 77.3 baseline (1.19x)
  with 81.5% acceptance. ROCm MoE stochastic remains rejected for performance:
  dynamic is 61.7 tok/s versus 78.0 baseline and fixed d3 acceptance is only
  30.6%, so the next accepted MoE slice must reduce stochastic target
  distribution/verifier work rather than toggling more concurrency flags.
- The focused ROCm MoE parity gate exposed a loader contract regression before
  it reached math comparison: integration tests could enter `load()` with a
  GPU pool but no pinned upload ring. The fix clamps the repack stream count at
  the `WeightManager` call site and makes `LoadOrchestrator::allocate()` reject
  pinned staging with zero H2D streams. Regression coverage:
  `V2_Unit_LoadOrchestrator`, serial
  `MTPStochasticSamplingVerifierRuns`,
  `MainVerifierAllPositionRowsMatchSerialDecode`,
  `MTPGreedyDepth3MatchesBaselineTokens`, and
  `MTPBenchmarkStyleDepth3LongPromptGreedyMatchesReference`.
- ROCm stochastic small-k partial-block sweep is rejected as a performance
  fix. `20260612T_rocm_moe_stochastic_topk_partial_sweep` tested caps
  16/32/64/128; best dynamic was cap 64 at 64.0 tok/s versus a 77.7 tok/s
  baseline, and fixed d3 stayed around 0.58x. Keep the automatic/default
  partial-block policy and move to fused/lazy stochastic verifier work that
  avoids building target/bonus distributions for rows that cannot be consumed
  after an early rejection.
- Generated MTP depth policy now keys on backend plus dense/MoE model class,
  emits direct target-depth deltas and learned hold rows, and uses hold rows as
  dynamic warm starts. Focused trainer/controller units pass. Diagnostic MoE
  smoke `20260611T-moe-generated-best-depth-guard-smoke` shows ROCm greedy
  dynamic stable at depth 3 with 96.2 vs 78.8 tok/s; CUDA MoE remains
  verifier-bound at 122.2 vs 143.8 tok/s.
- Dynamic MoE greedy now gives the generated best-depth lane one
  non-catastrophic full-window grace period before demoting. This fixes the
  ROCm depth-3 churn found in
  `20260612T_moe_rocm_splitk_depth_matrix`: a first window with
  acceptance=0.395833 and zero_accept=0.375 demoted even though fixed d3 won
  the whole request. `V2_Unit_MTPDepthController` covers that exact window and
  the second-consecutive-bad-window demotion escape hatch. Focused matrix
  `20260612T_moe_gpu_greedy_dynamic_grace` shows CUDA dynamic 147.2 vs 138.5
  baseline and ROCm dynamic 94.2 vs 78.0 baseline, both with zero demotions.
- Current MoE same-run matrices are
  `benchmark_results/mtp_vllm_style/20260612T_moe_cuda_splitk_depth_matrix/`,
  `benchmark_results/mtp_vllm_style/20260612T_moe_rocm_splitk_depth_matrix/`,
  and
  `benchmark_results/mtp_vllm_style/20260612T_moe_gpu_stochastic_refresh/`.
  CUDA greedy is correct but only weakly positive: fixed d3 is 146.8 tok/s vs
  139.2 baseline (1.05x). ROCm greedy is better but still short of dense-class
  wins: fixed d3 is 97.9 tok/s vs 77.3 baseline (1.27x). CUDA stochastic is
  still negative with best dynamic at 133.1 vs 139.2 tok/s. ROCm stochastic is
  still negative with best dynamic at 66.7 vs 77.8 tok/s.
- CUDA MoE tuning has rejected the latest gate/up=32, down=16, tile-M=2 full
  model A/B even though it helped isolated shapes: real d3 fell to 144.5 tok/s.
  Do not revive one-off tile/kpart overrides without a same-run matrix win.
  CUDA's next MoE target is verifier/condition transaction economics across
  the full 40-layer graph, not the already-fast isolated combined expert
  prefill kernel.
- ROCm stochastic attribution now shows the compact D2H copy itself is small in
  isolation, about 0.04 ms. The real cost is the queued GPU work before that
  host-visible boundary, especially Qwen-sized top-k/top-p target and draft
  distribution builds. `V2_Perf_GPUSpeculativeSummary` shows ROCm stochastic
  rows 1/2/3 at about 5.13/5.79/6.50 ms, dominated by target/draft
  distribution build, while CUDA is about 1.59/1.92/2.21 ms. A durable ROCm
  stochastic MoE win likely needs a lazy or fused target-distribution verifier
  that avoids bonus/later-row top-k work after early rejection, not another
  host read tweak.
- Focused MoE GPU sprint
  `benchmark_results/mtp_vllm_style/20260612T120836Z-moe-gpu-focused-sprint/`
  confirms that compact verifier polishing is the wrong center of gravity.
  CUDA MoE stochastic is still speed-negative despite high d1 acceptance
  (90.6%), while ROCm MoE stochastic has both poor speed and very different
  acceptance (45.8/56.9/31.3% for d1/d2/d3). The follow-up architecture slice
  proved the useful vLLM idea is not persistent full target/draft probability
  rows; it is greedy draft proposal, processed target rows, one-hot `q`, and a
  batched device outcome that can sample the rejected or bonus row without host
  participation.
- The first vLLM-style recovered-token primitive is implemented as a shared
  CPU reference plus CUDA/ROCm graph-capturable backend kernels. Focused gates
  passed: `V2_Unit_MTPRejectionSampler`,
  `V2_Integration_GPUSamplingKernels`, and `V2_Perf_GPUSpeculativeSummary`.
  Direct perf for `StochasticFullProbabilityQwen36Rows` shows rejection
  recovery itself is not the remaining MoE blocker: CUDA M=1/2/3 is about
  0.146/0.152/0.223 ms and ROCm is about 1.196/1.208/1.195 ms for Qwen-sized
  rows. Graph-capturable processed-logit softmax materialization is now also
  implemented and covered in the same GPU sampler/perf gates. Direct perf for
  `ProcessedLogitSoftmaxQwen36Rows` is CUDA M=1/2/3 about
  0.425/0.425/0.426 ms and ROCm about 0.969/1.013/1.143 ms. The accepted
  production slice wires processed target logits, device-resident draft tokens,
  one-hot `q`, and the existing stochastic summary reducer instead of allocating
  persistent full-probability buffers.
- The first lazy-target proof is deliberately perf-harness-only, not a
  production path. `Perf__GPUSpeculativeSummary.StochasticLazyTargetQwen36Rows`
  uses real backend kernels with deterministic accepted-prefix fixtures. It
  shows a host-loop lazy verifier is not viable: ROCm M=3 reject-at-0 is
  slightly cheaper than eager (about 6.00 ms vs 6.50 ms), but reject-after-1 and
  accept-all are much worse (about 8.54 ms and 13.37 ms). If we pursue lazy
  target verification, it must be one fused GPU-side reducer that scans rows
  until rejection without per-row host-visible boundaries.
- A production compact lazy-bonus A/B was also rejected and removed. Diagnostic
  matrices `20260612T_lazy_bonus_off_moe_stochastic_diag` and
  `20260612T_lazy_bonus_on_moe_stochastic_diag` showed CUDA dynamic falling
  from 136.0 to 130.2 tok/s and ROCm dynamic from 61.7 to 59.6 tok/s; only ROCm
  fixed d3 moved from 43.9 to 44.8 tok/s, not enough to justify a branchy
  env-only path. The next lazy attempt must be fused GPU-side, not
  summarize-then-build-bonus on the host boundary.
- Same-run stochastic accepted-prefix histograms explain why this is primarily
  a ROCm MoE target today. CUDA fixed d3 accepts all three drafts in about 54%
  of verifier steps and averages about 2.09 accepted drafts, so eager batched
  target distributions remain reasonable. ROCm fixed d3 rejects at prefix 0 in
  about 65% of verifier steps and averages about 0.51 accepted drafts, so it
  frequently pays for target rows and a bonus row that cannot be consumed.
- A longer greedy capture-amortization check
  `benchmark_results/mtp_vllm_style/20260612T_moe_gpu_greedy_long256_capture_check/`
  shows first-use graph economics are part, but not all, of CUDA MoE's weak
  speedup. CUDA fixed d3 improves to 170.7 tok/s vs 146.7 baseline (1.16x) at
  256 decode tokens, compared with 1.05x in the decode-64 matrix. ROCm fixed d3
  is 96.2 tok/s vs 79.4 baseline (1.21x). This keeps both GPU MoE greedy lanes
  below the dense-class target and confirms that the next accepted win must
  reduce steady-state verifier/condition work, not merely hide capture warmup.
- `scripts/run_mtp_iteration_benchmark_matrix.sh` now has `--decode-tokens N`
  for bounded all-device iteration sweeps. The default remains the full
  benchmark decode length.
- The matrix runner now hard-fails dynamic-depth evidence without same-run
  `baseline,fixed_d1,fixed_d2,fixed_d3` neighbors unless
  `--allow-partial-variants` is explicitly set for local diagnostics.
- Bounded MoE iteration matrix covers CUDA/ROCm/CPU, greedy/stochastic,
  baseline, fixed d1/d2/d3, and dynamic at 16 decode tokens. Latest full matrix
  is `benchmark_results/mtp_vllm_style/20260609T061226Z-iteration-matrix-6753b5e7/`.
  All lanes are functionally green, including CUDA and ROCm stochastic. Every
  MoE MTP lane is still speed-negative against its same-run baseline. Best
  bounded greedy lanes are CUDA d3 69.5 vs 109.8 tok/s, ROCm d3 41.4 vs 64.7
  tok/s, and CPU d3 12.7 vs 17.9 tok/s. Best bounded stochastic lanes are CUDA
  dynamic 53.4 vs 109.7 tok/s, ROCm dynamic 30.6 vs 64.2 tok/s, and CPU d3 12.5
  vs 17.5 tok/s.
- The dead verifier-row publication hooks and tests were removed.

Open gaps:

- Full default-length CPU dense and CPU MoE matrix refreshes remain slow
  acceptance work. Bounded CPU dense and MoE have previous evidence, but fresh
  all-in-one iteration runs now split CPU out because a single `cpu:0` dense
  baseline can take about five minutes even at 16 decode tokens.
- CPU stochastic accepted-count publication is now implemented and
  correctness-gated, but benchmark acceptance is still open. The parity stats
  showed real CPU overhead in the publication path, including hybrid checkpoint
  export, accepted-state publication, and host sampler work. The first
  parallelization pass and top-k sampler fast path have landed, but latest
  bounded benchmark evidence is still speed-negative until refreshed after
  these changes.
- CPU dynamic dense greedy is not controller-overhead bound. In the latest
  bounded matrix, dynamic behaves like fixed d2 at 5.6 tok/s while fixed d3 hits
  9.1 tok/s; the dynamic perfstats show about 4.26s verifier time, 1.51s
  condition-forward time, and 0.51s accepted-state publication time over the
  16-token lane.
- A focused CPU dense dynamic profiler pass
  `benchmark_results/mtp_vllm_style/20260609T122350Z-cpu-dense-dynamic-verifier-profile/`
  confirms the verifier cost is model math, not controller or executor spin:
  dynamic landed at 6.56 tok/s with 4.20s verifier, 1.48s condition-forward,
  and 56.8ms publication. Host executor decode stage time was led by
  `GEMM_FUSED_GATE_UP` 30.7%, `GEMM` 27.2%, `GDN_PROJECTION` 13.9%, and
  `LM_HEAD` 12.2%. This is still less vLLM-shaped than desired because CPU
  stochastic evidence is stepwise and CPU greedy still pays a full all-position
  target forward plus batched all-position LM-head rows for verification.
- GDN/short-conv speculative-slot publication is available through verifier row
  capture hooks and is now used by the CPU/CUDA/ROCm all-position publication
  path; broader benchmark evidence still needs to catch up.
- CUDA/ROCm/CPU MoE bounded matrices are functionally green for greedy and
  stochastic, but MTP is speed-negative everywhere. The common blocker is true
  verifier/catch-up cost. Latest fixed d3 MoE greedy spends about 379 ms total
  verifier time plus 220 ms condition-forward time on CUDA, and 659 ms verifier
  plus 346 ms condition-forward time on ROCm, while correction replay remains
  0 ms. Dynamic depth is now stable across d0/d1/d2 transitions but still needs
  better short-run promotion and stochastic depth selection.
- Reusing the ordinary main-decode capture across rejected-state publication by
  merely restamping the live epoch was tested and rejected: focused parity was
  too weak to catch it, but the release MoE benchmark acceptance collapsed.
  Keep the correction boundary recapture until a stronger backend state-refresh
  contract exists.
- CPU vLLM-style state publication is implemented for the current stochastic
  SingleDevice contract but not yet benchmark-accepted.
- CPU MoE commit-replay verification now restores the post-condition
  verifier-base checkpoint. The focused CPU parity regression also proves the
  main all-position verifier rows match serial decode rows, including shifted
  cache preconditioning. The slow CPU serial LM-head verifier helper was removed
  after the batched NativeVNNI all-position path matched serial decode rows, so
  future failures should be treated as real state or publication drift rather
  than a checker-base artifact.
- Phase 3 row-indexed verifier work is accepted for dense SingleDevice:
  `HiddenStateRowsSelectStage` packs a fixed small row set into compact
  `[rows, d_model]` scratch, Qwen forward graphs can feed that scratch to one
  batched LM head, and `OrchestrationRunner` enables the row count around the
  all-position verifier. CPU copies compact rows; CUDA/ROCm use explicit-stream
  graph-workspace row-index arrays with captured replay support. The
  `HiddenStateRowsSelectStage` GPU direct-tensor path now hard-fails unless the
  caller already prepared device pointers, keeping new production tensor
  movement out of `ensureOnDevice()`. Focused slice gate passed:
  `V2_Unit_MTPGraphConstruction`, `V2_Unit_HiddenStateRowSelectStage`,
  `V2_Unit_ForwardExecutionEngineAdvanced`,
  `V2_Integration_CUDAHiddenStateRowSelectStage`, and
  `V2_Integration_ROCmHiddenStateRowSelectStage`. Dense Qwen3.6
  CPU/CUDA/ROCm fixed depth-3 greedy and stochastic verifier parity also pass
  on the row-indexed path. The accepted full bounded matrix
  `benchmark_results/mtp_vllm_style/20260609T-phase3-row-indexed-accepted-matrix/`
  covers CUDA/ROCm/CPU, dense/MoE, greedy/stochastic, baseline plus fixed
  d1/d2/d3 and dynamic, with perfstats enabled. Dense greedy is speed-positive
  on all three backends: CUDA best d1 60.9 vs 44.6 tok/s, ROCm best d1 45.7
  vs 31.3 tok/s, and CPU best d3 9.3 vs 4.7 tok/s. Dense stochastic remains
  policy-sensitive and MoE remains speed-negative, so Phase 3 is accepted for
  dense SingleDevice row-indexed verifier correctness/perf only; MoE tuning
  moves to the next slice.
  `MTPSpecDecodeVerifierInputPlan` now names the current single-request
  verifier input and compact-row layout. `OrchestrationRunner` scopes that plan
  around verifier forwards, `DeviceGraphOrchestrator` uploads its row metadata
  through `MTPSpecDecodeMetadataWorkspaceBinding` on the execution/capture
  stream, and `HiddenStateRowsSelectStage` reads that persistent row buffer
  without stage-local uploads. `V2_Unit_PrefillDecodeTransition` now proves the
  verifier plan is installed, consumed, and cleared for the all-position
  publication path, and perfstats expose `verifier_row_metadata_path`. The
  broader MTP unit gate, full `^V2_Unit_` hard commit gate, and required
  Integration/Release builds pass.
- CUDA MoE MTP is still speed-negative and must reduce verifier/catch-up cost
  before acceptance. Stochastic also needs acceptance-policy tuning or depth
  policy integration for the default prompt class.
- CUDA and ROCm dense stochastic MTP now match acceptance under the same seed,
  but the generated token streams still differ at a few real-model samples
  while the real-logit-style sampler fixture passes. This points at full-model
  logits/state/perf differences rather than isolated sampler math.
- Phase 4 sampler-contract and first device-outcome slices are implemented:
  `MTPRejectionSampler` owns threshold-driven stochastic row semantics and a
  backend-neutral batch summary contract. CUDA/ROCm now verify penalty-free
  all-position stochastic rows in one batch, sample the bonus token into an
  arena buffer, and reduce committed output tokens/accepted counts with a tiny
  shared-math summary kernel before copying compact metadata to host. Focused
  MTP unit gate, dense Qwen3.6 CPU/CUDA/ROCm stochastic parity, Integration and
  Release builds pass. Focused benchmark
  `benchmark_results/mtp_vllm_style/20260609T-phase4-device-batch-outcome-dense-stochastic/`
  shows CUDA best dynamic 38.4 vs 44.7 baseline and ROCm best d1 24.3 vs 31.4
  baseline; CPU baseline completed but the CPU d1 lane was stopped after
  excessive wall time. This is still not full Phase 4 acceptance: dense
  stochastic remains speed-negative and true GPU-resident sampling still needs
  fewer host round trips plus a speed-positive policy.
- Phase 4 penalty-free stochastic sidecar stream handoff is implemented:
  sidecar logits can remain on the explicit capture/replay stream until compact
  distribution build and device batch-outcome reduction consume them. Penalty
  paths still force the older synchronized contract because sampler history can
  mutate logits. `V2_Unit_PrefillDecodeTransition`, the broader MTP unit gate,
  dense Qwen3.6 CPU/CUDA/ROCm stochastic verifier parity, and Integration/Release
  builds passed. Focused CUDA/ROCm dense stochastic benchmark
  `benchmark_results/mtp_vllm_style/20260610T-phase4-stochastic-sidecar-stream-handoff/`
  shows the handoff counters active on both backends. CUDA now has a short-run
  stochastic win at fixed d2, 49.5 vs 44.6 tok/s (1.11x), but ROCm remains
  speed-negative, best fixed d2 29.9 vs 31.3 tok/s (0.96x). Perfstats point at
  ROCm stochastic device sampling cost as the next blocker:
  `sample_mtp_token_stochastic_device` is about 4.4 ms/sample on ROCm versus
  about 1.0 ms/sample on CUDA in this lane. A fused first-token direct sampler
  experiment was benchmark-rejected and removed; remaining Phase 4 work is
  ROCm sampler tuning plus true device-resident draft/decision plumbing, not
  another first-token sampler variant.
- Trusted compact stochastic result reads now use the backend fast D2H path for
  orchestrator-owned scratch. The MTP unit gate plus dense Qwen3.6 CPU/CUDA/ROCm
  stochastic verifier parity passed, and focused benchmark
  `benchmark_results/mtp_vllm_style/20260610T-phase4-stochastic-fast-d2h/`
  shows this was hygiene rather than the missing speedup: CUDA best fixed d2 is
  49.8 vs 44.6 tok/s (1.12x), while ROCm best fixed d2 is 29.2 vs 31.4 tok/s
  (0.93x) and `sample_mtp_token_stochastic_device` remains about 4.4-4.6
  ms/sample. The next accepted Phase 4 slice should promote draft tokens,
  target rows, and verifier bonus metadata to persistent device buffers so MTP
  does not D2H a sampled draft token between sidecar steps or before verifier
  input planning.
- Device-token batch verification and device-resident sidecar token input are
  now implemented for penalty-free CUDA/ROCm stochastic rows. Draft sample
  tokens are written into arena buffers, verifier batches consume those device
  tokens directly, and chained sidecar rows copy the prior sampled token into a
  stable arena-owned `MTP_CONDITION_TOKEN` on the explicit sidecar stream.
  ROCm gained the missing non-synchronizing `deviceCopyAsync()` backend hook
  after the first benchmark exposed a hard failure in fixed d2. Focused
  `V2_Unit_PrefillDecodeTransition`, broader MTP unit gate,
  `V2_Integration_GPUSamplingKernels`, and dense Qwen3.6 CPU/CUDA/ROCm
  stochastic parity passed. Focused benchmark
  `benchmark_results/mtp_vllm_style/20260610T-phase4-stochastic-device-sidecar-token-input-fixed/`
  shows CUDA fixed d2 at 46.6 vs 44.7 tok/s (1.04x) and ROCm fixed d2 at
  29.8 vs 31.3 tok/s (0.95x). Follow-up ROCm attribution
  `benchmark_results/mtp_vllm_style/20260610T-phase4-rocm-stochastic-sample-sync-attribution/`
  shows sampler enqueue is cheap, about 0.003 ms/sample, while the compact
  result D2H/sync costs 2.9-4.7 ms/sample because it drains deferred
  verifier/sidecar work at the host read boundary. The next Phase 4 target is
  device-resident verifier input and metadata rather than more top-k kernel
  tuning. The generic forward graph contract now carries stable device token
  IDs, Qwen embedding graphs pass the pointer through, forward-cache signatures
  distinguish host-token and device-token sources, and focused
  `V2_Unit_IGraphBuilder`, `V2_Unit_ForwardGraphTypes`, and
  `V2_Unit_ForwardExecutionEngine` pass for that slice. The target verifier
  can now compose `[first_token, draft_0, ...]` into arena-owned
  `MTP_VERIFIER_INPUT_TOKENS` on the same explicit stream used for verifier
  graph replay, then call `forwardWithDeviceTokenIds()` with the host token row
  retained only as metadata shadow. Focused
  `V2_Unit_PrefillDecodeTransition`, `V2_Unit_IGraphBuilder`,
  `V2_Unit_ForwardGraphTypes`, and `V2_Unit_ForwardExecutionEngine` pass for
  the verifier-input slice. Deferred draft-token host reads are now implemented
  for penalty-free GPU stochastic rows and guarded by explicit sample-ready
  events. Chained sidecars, verifier token input staging, and the batched
  stochastic verifier wait on those events instead of relying on the old scalar
  D2H read as an accidental synchronization point. Focused
  `V2_Unit_PrefillDecodeTransition`,
  `V2_Integration_GPUSamplingKernels`,
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke`, and
  `V2_Integration_Parity_Qwen36_ROCm_SingleDevice_Qwen36ROCmSingleDevicePrefixMTPParity_MTPStochasticSamplingVerifierRuns`
  pass. Focused ROCm benchmark
  `benchmark_results/mtp_vllm_style/20260610T-phase4-deferred-draft-host-read-ordered/`
  restores plausible acceptance and removes draft sample D2H, but fixed d2 is
  still speed-negative at 26.8 vs 31.2 tok/s; target sampling still performs a
  compact D2H at about 2.86 ms/read and verifier forward averages about
  22.7 ms/run. Later Phase 4 slices removed the token-D2H boundary and moved
  the remaining compact outcome/publication cost into the Phase 5/6 state-slot
  work.
- Penalty-free stochastic all-position verification now repairs the first
  shifted MTP KV row from the device-resident target sample when the host token
  is intentionally deferred. `HiddenStateRowSelectStage` and
  `HiddenStateRowsSelectStage` replay mutators are host-intent-only: they never
  dereference stale workspace bindings or upload GPU metadata before
  executor-owned workspace/stream binding. Graph launch preparation now uploads
  dirty row metadata through a typed `prepareGraphLaunch()` hook after the
  executor has rebound the current workspace and explicit stream, and the
  CUDA/ROCm row-select integrations exercise that contract without recapture.
  CUDA and ROCm embedding validation readbacks are skipped during graph capture
  because D2H plus stream sync is capture-illegal; ordinary validation still
  checks device token IDs outside capture. Focused gates passed:
  post-relink MTP unit gate,
  `V2_Unit_ForwardGraphTypes`,
  `V2_Unit_PrefillDecodeTransition`,
  `V2_Unit_HiddenStateRowSelectStage`,
  `V2_Unit_PrefillGraphCaptureDynamicParams`,
  CUDA/ROCm `HiddenStateRowSelectStage`,
  `V2_Integration_GPUSamplingKernels`, and CUDA/ROCm
  `Qwen36*GpuGraphsStochasticSmoke`.
- The first target token can now stay device-resident through the first MTP
  sidecar, verifier-input composition, and batched stochastic summary on
  CUDA/ROCm. Target and draft sampled-token slots use explicit sample-ready
  events, the sidecar consumes the target sample through
  `forwardMTPFromDeviceTargetForDeviceSampling()`, and the summary reducer can
  read the first token from device memory instead of a host scalar. Focused
  `V2_Unit_PrefillDecodeTransition`,
  `V2_Integration_GPUSamplingKernels`,
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke`, and
  `V2_Integration_Parity_Qwen36_ROCm_SingleDevice_Qwen36ROCmSingleDevicePrefixMTPParity_MTPStochasticSamplingVerifierRuns`
  pass. Focused ROCm benchmark
  `benchmark_results/mtp_vllm_style/20260610T-phase4-device-first-target-summary/`
  shows 15 deferred first-token reads, 15 device-first batch summaries, target
  sample ready events/waits, and only one remaining target-slot D2H sync for
  the final/budget-limited step. Fixed d2 is still speed-negative at 26.97 vs
  31.25 tok/s, so the next Phase 4 blocker has moved to shifted-prefill,
  condition-forward, and verifier-forward host-wall cost rather than token
  scalar D2H. The post-rebuild Phase 4 gate passed `^V2_Unit_` 500/500 plus
  focused ROCm stochastic parity, ROCm stochastic graph smoke, and CUDA/ROCm
  GPU sampling integrations.
- `clearCache()` now treats adaptive MTP depth as request-scoped state. The
  prior behavior preserved learned depth across benchmark iterations while
  resetting MTP counters, producing inconsistent summaries such as
  `current_depth=2` with `updates=0`. Focused
  `V2_Unit_PrefillDecodeTransition` passes after updating the regression. The
  follow-up ROCm stochastic reset check
  `benchmark_results/mtp_vllm_style/20260610T-phase4-rocm-dense-stochastic-dynamic-reset-check/`
  shows cold-request dynamic still trails baseline, 27.36 vs 30.74 tok/s,
  while follow-up policy probes rejected aggressive controller-only tuning:
  d1->d0 churn fell to 23.52 tok/s, optimistic d3 start reached 25.29-26.79
  tok/s, and a same-build fixed d3 control reached 28.33 tok/s. The standard
  matrix runner now keeps dynamic depth on a d1 floor; depth-zero bypass is a
  diagnostic/experimental lane until it is faster than d1. The accepted focused
  ROCm stochastic lane
  `benchmark_results/mtp_vllm_style/20260610T-phase4-rocm-dense-stochastic-fixed-dynamic-floor1/`
  shows baseline 30.68 tok/s, fixed d1 31.88, fixed d2 26.01, fixed d3 29.64,
  and dynamic 31.78 with 75% acceptance and no depth churn.
- Main-decode stream handoff now covers MTP condition forward, MTP depth-zero
  direct state advance, and ordinary GPU decode sampling. The forward engine
  reports actual deferred final-sync events and publishes the capture stream
  only to the immediate GPU sampler/distribution consumer. Focused units plus
  ROCm stochastic parity/smoke pass. The perfstats confirm `main_decode`
  replay syncs are gone on the corrected ROCm dynamic lane; the remaining
  shifted-prefill, sequential shifted-row commit, verifier, and
  condition-forward costs became the concrete follow-up tuning targets.
- Pending logits stream handoff is now structurally owned by private role
  slots (`MTPSidecar`, `MainDecode`, `AllPositionVerifier`) instead of raw
  nullable fields. The stream pointer is private to a one-shot handoff object:
  producers may republish the same explicit stream after an in-place logits
  mutation, but replacing an unconsumed handoff with a different stream is a
  hard logic error. The source hygiene unit strips comments/strings and fails
  if production code accesses the slot table or reintroduces a raw mutable
  slot-reference helper. The post-relink focused gate passed
  `V2_Unit_GpuWorkspaceAllocationPolicy`, row-select/graph-launch units, CUDA
  and ROCm row-select integrations, GPU sampler integration, and CUDA/ROCm
  Qwen3.6 stochastic graph smokes. `V2_Unit_DeviceGraphOrchestrator` now also
  covers the runtime one-shot rule through the public host interface: same
  stream republish is allowed, different unconsumed stream overwrite throws,
  explicit clear and `clear_cache()` reset ownership, and verifier/main roles
  are independent.
- Correction-replay publication reset now returns a typed
  `ReplayStateResetSummary` and exports cache-class counts in the
  `live_prefix_replay_state_after_mutation` perf record. Focused
  `V2_Unit_ForwardExecutionEngine` coverage proves ordinary decode cache
  identities are reset while all-position verifier identities are preserved for
  explicit-stream rebind. `scripts/summarize_mtp_perfstats.py` and the matrix
  TSV now surface reset-cache, stream-rebind, ordinary-decode, verifier, and
  other-cache counts for each benchmark lane. The next telemetry slice added
  decode-only sidecar, shifted-row, stochastic sampling, checkpoint, and
  sidecar graph hit/miss columns so ROCm/CUDA tuning can distinguish verifier
  cost from shifted-cache maintenance. The script unit, shell syntax check, a
  one-row baseline TSV sanity check, and an extended ROCm dense d1 field-count
  smoke pass.
- The long ROCm stochastic clear-cache repeatability regression split the state
  problem into two boundaries. Dense sidecar execution can now advertise
  `supportsMTPSidecarPreservesMainState()` and skip the verifier-base restore,
  after the preservation checker was corrected to compare against the
  post-condition verifier-base checkpoint. Accepted-state publication, however,
  is still a hard live-state mutation and now always resets GPU replay state.
  The previous accept-all replay preservation changed ROCm stochastic
  trajectories after `clearCache()`. A source hygiene guard locks the replay
  reset in `publishAcceptedMTPSpecState()`, and the focused gate passes:
  `V2_Unit_GpuWorkspaceAllocationPolicy`,
  `V2_Integration_Parity_Qwen36_ROCm_SingleDevice_Qwen36ROCmSingleDevicePrefixMTPParity_MTPStochasticSamplingVerifierRuns`,
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke`,
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticClearCacheRepeatabilityLong`,
  and `V2_Integration_GPUSamplingKernels`. The short CUDA/ROCm stochastic
  smokes now set `LLAMINAR_DETERMINISTIC` before asserting token equality; a
  Qwen3.6 top-k=20 repeated graph-replay sampler regression rules out compact
  sampler drift, while non-deterministic fast-path near-ties remain a benchmark
  signal rather than a repeatability assertion.
- KV-only MTP sidecar replay now has its own event-backed shifted-MTP-KV
  handoff instead of borrowing the pending-logits stream marker. The old
  ownership mix skipped both explicit stream sync and shifted-KV readiness
  events for segmented KV-only replay. The source hygiene unit now enforces
  that KV-only sidecars do not call the deferred sampling/logits handoff, and
  that accepted-state publication waits before touching shifted MTP KV. Focused
  units, CUDA/ROCm stochastic graph smokes, GPU sampler integration, and the
  release relink pass. Refreshed ROCm dense stochastic focused matrix
  `benchmark_results/mtp_vllm_style/20260610T-focused-rocm-dense-stochastic-dynamic-shifted-kv-shape-fixed/`
  shows baseline 32.55 tok/s, fixed d1 34.39, fixed d2 29.83, fixed d3 24.34,
  and dynamic 34.49 after one promotion. Follow-up condition telemetry and
  dynamic-depth tuning culminated in
  `benchmark_results/mtp_vllm_style/20260610T-focused-rocm-dense-stochastic-long64-depth-explore/`:
  baseline 30.37 tok/s, fixed d1 32.35, fixed d2 29.11, fixed d3 33.66, and
  dynamic 30.84. Dynamic now reaches d3 through d1 floor promotion plus
  `probe_higher_before_demote`, but it still trails fixed d3 because verifier
  and rejection-driven condition-forward cost dominate; sampling remains below
  13 ms/request. The added main-decode replay telemetry first showed the
  dynamic condition path did 28 warmups, 1 capture, and 0 replay, so ordinary
  decode replay preservation/rebinding after spec-state publication became
  the next concrete speed target. That slice is now implemented:
  `ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode` keeps one-token
  condition/decode captures alive across MTP accepted-state publication by
  marking them dirty for explicit-stream rebind and stamping them with the new
  live replay-state epoch. Focused real-model telemetry in
  `benchmark_results/mtp_vllm_style/20260610T-focused-rocm-dense-stochastic-long64-ordinary-replay-preserve/`
  moved the dynamic condition path to 4 warmups, 4 captures, and 65 replays,
  with `replay_ordinary_decode_resets=0`, zero transaction validation failures,
  and zero rollbacks. The next accepted slice reuses the first shifted MTP KV
  row appended by main-state-preserving sidecars instead of truncating it away
  and rerunning a KV-only depth-0 sidecar. Focused
  `V2_Unit_PrefillDecodeTransition` coverage proves all-position publication
  reuses that first row while still sequentially committing rejected correction
  rows, and the matrix schema now reports `shifted_initial_commits` plus
  `shifted_initial_reused`. Real-model ROCm telemetry in
  `benchmark_results/mtp_vllm_style/20260610T-focused-rocm-dense-stochastic-long64-shifted-first-reuse/`
  shows `shifted_initial_commits=0`, reused sidecar rows on every MTP lane,
  dynamic `shifted_row_ms` dropping from about 1006 ms to about 102 ms, zero
  transaction validation failures, and zero rollbacks. Controller probes then
  tightened `probe_higher_before_demote`, floor-promotion thresholds, and
  rejected-depth hysteresis while benchmark-rejecting early bad-probe aborts
  and stricter perfect-floor hysteresis.
- Phase 4 dense SingleDevice closeout is accepted. The final slice split
  first-sidecar host-token and device-token graph caches, preserved sidecar
  replay state across accepted-state publication, and added a GPU regression for
  alternating host/device first-sidecar calls. The focused gate passed
  `V2_Unit_MTPRejectionSampler`, `V2_Unit_PrefillDecodeTransition`,
  `V2_Unit_MTPGraphConstruction`, `V2_Unit_DeviceGraphOrchestrator`,
  `V2_Unit_GpuWorkspaceAllocationPolicy`, `V2_Integration_GPUSamplingKernels`,
  dense Qwen3.6 stochastic parity on CPU/CUDA/ROCm, and CUDA/ROCm stochastic
  graph smokes. The closeout matrix
  `benchmark_results/mtp_vllm_style/20260610T-phase4-dense-stochastic-closeout-matrix-v2/`
  shows useful speed-positive stochastic lanes on every backend: CUDA fixed d3
  59.44 vs 43.88 tok/s, ROCm fixed d1 33.48 and dynamic 32.57 vs 30.33 tok/s,
  and CPU fixed d2 5.78 vs 4.46 tok/s. ROCm fixed d2/d3 are documented
  acceptance-limited, not contract failures.
- Phase 5 has started with a typed live-state mutation ledger. Runtime probes
  and perf tags now distinguish accepted publication, rejected correction,
  prefix restore, prefix truncate, and session reset; `clear_cache()` and
  `clearInferenceState()` both advance the live-state epoch. The runner also
  skips the post-condition verifier-base checkpoint export on the all-position
  publication path when the sidecar is main-state preserving and debug replay
  checks are off. The second focused slice moved that synthetic verifier-base
  stamp into `makeLogicalMTPVerifierBaseSnapshot()`, making the checkpoint-free
  path a tested MTP transaction primitive instead of an inline runner detail.
  `V2_Unit_MTPStateTransaction` now proves the logical stamp carries
  decode-equivalent main/shifted-KV token counts and no payload blocks.
  `V2_Unit_MTPGraphConstruction` also proves accepted publication and rejected
  correction update distinct live-state mutation reasons while preserving the
  sidecar-owned first shifted-KV row contract. The benchmark summary pipeline
  now reports `publish_count` and `publish_avg_ms` beside `publish_ms`, so the
  Phase 5 closeout matrix can judge publication stability across d1/d2/d3
  instead of comparing only total wall time. A bounded dense stochastic
  publication-cost slice is green on CUDA/ROCm with 16 decode tokens and CPU
  with 8 decode tokens:
  `benchmark_results/mtp_vllm_style/20260610T-phase5-publication-cost-dense-stochastic-gpu/`
  and `...-cpu8/`. Publish cost is stable across d1/d2/d3: CUDA
  0.47-0.56 ms/publish, ROCm 0.29-0.32 ms/publish, and CPU
  3.84-3.86 ms/publish. Checkpoint export still appears as debug/prefix
  anchoring cost, but the steady slot-publication path is no longer scaling
  with depth. The next slice added a forced-reject replay oracle for the no
  ready-token case: under `LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK`, the
  all-position publication path now derives the next token by forwarding the
  rejected correction from the committed state, then compares that token and
  continuation against a full replay from the verifier base. The
  `AllPositionSpecPublicationForcedRejectReplayCheckDerivesNextToken` unit
  proves the next one-token decode consumes the rejected correction exactly
  once. The final cleanup slice removed the stale all-position
  `discarded_sidecar_checkpoint` tag; that tag remains only on the sequential
  verifier path where a post-sidecar checkpoint is still a real object.
  Phase 5 is accepted on the focused gate.
- TP/PP/ExpertParallel MTP is out of scope until SingleDevice is green.

## Implementation Phases

### Phase 1: Freeze The Spec Transaction Contract

Goal: make one transaction object describe every speculative step.

Work:

- Promote `MTPSpecStepPlan`, `MTPSpecDecodeMetadata`, accepted-count planning,
  and publication provenance into the only legal interface between
  `OrchestrationRunner` and backend publication.
- Add `MTPSpecPersistentMetadata` and a CPU reference implementation with the
  same shape as the GPU buffers.
- Encode target rows, bonus rows, accepted counts, rejected/correction rows,
  state-slot indices, and stop/EOS behavior in metadata rather than side
  channels.
- Move dynamic-depth observations to consume transaction outputs only.

Status:

- Accepted. The runner now drives speculative decode through
  `MTPSpecDecodeTransaction`, `MTPSpecStepPlan`, and
  `MTPSpecStateContract` instead of backend-local side channels. Focused
  coverage includes `V2_Unit_MTPIterationBenchmarkMatrix`,
  `V2_Unit_MTPSpecDecodeMetadata`, `V2_Unit_MTPSpecDecodeTransaction`,
  `V2_Unit_MTPDecodeCatchup`, `V2_Unit_MTPVerifierPolicy`, and
  `V2_Unit_PrefillDecodeTransition`. Dynamic-depth accounting consumes
  accepted/rejected/rollback transaction outputs rather than raw runner
  mutations.

Exit gate:

- Unit tests cover accept-all, reject-first, reject-after-prefix, bonus-ready,
  stop/EOS, prefix restore, stochastic residual, and budget-limited d0/d1 cases
  without invoking a model runner.
- Runner tests fail if a path commits tokens or state without a transaction.

### Phase 2: Persistent Metadata Buffers And Spec Slots

Goal: match vLLM's padded persistent metadata/state-buffer shape.

Work:

- Add per-backend persistent buffers for draft tokens, positions, query starts,
  target/bonus logit indices, draft probabilities, random uniforms, accepted
  counts, and GDN/short-conv/KV state-slot indices.
- GPU buffers must be declared through workspace/arena consumers and updated on
  explicit streams. CPU buffers use the same layout and can be parallel-filled.
- Replace request-local vectors in hot verifier paths with views into these
  buffers.
- Add diagnostics showing whether a lane used persistent metadata or a
  compatibility vector path.

Status:

- Accepted. Persistent metadata/workspace bindings exist for the verifier and
  sampler hot paths, including draft tokens, verifier-row positions, sampled
  device-token slots, stochastic draft sample probabilities, and accepted-count
  publication plans. GPU paths declare scratch through arena/workspace
  consumers and hard-fail missing explicit streams; CPU uses the same logical
  layout through host-side metadata. Focused coverage includes
  `V2_Unit_MTPSpecDecodeMetadata`, `V2_Unit_GpuWorkspaceAllocationPolicy`,
  `V2_Unit_PrefillDecodeTransition`, `V2_Integration_GPUSamplingKernels`, and
  the static hygiene guards for default streams and ad-hoc ROCm hot-path
  allocations.

Exit gate:

- CPU/CUDA/ROCm unit tests prove identical metadata layout for fixed d1/d2/d3,
  dynamic d0 probes, and stochastic rows.
- Perfstats show zero hot-path ad-hoc GPU allocations and no implicit-stream
  operations.

### Phase 3: Row-Indexed Target Verifier Graph

Goal: keep the target verifier as one `draft_count + 1` forward but avoid
unnecessary all-position work.

Work:

- Build verifier graph inputs from persistent metadata, not temporary vectors.
- Add row-indexed LM-head/logits production for target rows and bonus rows.
  The initial Qwen graph wiring is complete for a fixed compact row count, with
  cache-key protection for different compact depths. The current single-request
  verifier row layout is now named through `MTPSpecDecodeVerifierInputPlan`.
  `HiddenStateRowsSelectStage` consumes caller-owned device row metadata from
  `MTPSpecDecodeMetadataWorkspaceBinding`; GPU uploads happen on the same
  explicit stream as verifier graph execution. Full all-position LM head remains
  only as a guarded compatibility mode until benchmark acceptance is proven.
- Preserve verifier graph capture/replay across accept-all steps and recapture
  only when a true state-boundary invalidation occurs.
- Add CPU stage attribution for verifier rows so regressions can identify
  GEMM/GDN/LM-head cost by phase. Row metadata path diagnostics are in place;
  stage-level CPU attribution remains to be expanded if CPU verifier cost
  regresses again.

Exit gate:

- Accepted for dense SingleDevice as of
  `benchmark_results/mtp_vllm_style/20260609T-phase3-row-indexed-accepted-matrix/`.
  Dense CPU/CUDA/ROCm greedy and stochastic parity pass with row-indexed
  verifier logits; dense greedy benchmarks are speed-positive on all three
  backends.
- Remaining follow-up: keep the compatibility all-position verifier mode
  guarded until the next CI cleanup slice removes dead verifier paths.

### Phase 4: Batched Greedy/Stochastic Rejection Sampler

Goal: replace stepwise stochastic verification with a vLLM-shaped batched
sampler.

Work:

- Implement a shared rejection-sampling interface over flattened target logits,
  draft probabilities, draft tokens, and random thresholds.
  First slice complete: `MTPRejectionSampler` defines the distribution-row
  contract and all-position stochastic catch-up construction for the current
  SingleDevice path.
- Phase 4 follow-up for MoE performance: replace the compact-table stochastic
  verifier with the vLLM worker-style full-logit path. The focused
  `V2_Perf_GPUSpeculativeSummary.StochasticLazyTargetQwen36Rows` trial rejected
  two shortcuts: single-block lazy full-logit verification is far slower than
  compact tables on CUDA/ROCm, and conditional bonus sampling does not beat the
  current compact batch. Both rejected prototypes and their tests were removed
  after focused rebuild, integration, and perf-smoke guards passed. The
  accepted design now needs
  full-vocab block stats over target/draft logits, accepted-count reduction
  from draft-token probability lookups, and one-row rejected/bonus resampling.
  Shared slices complete: `MTPRejectionSampler` now has processed full-logit
  row stats, probability lookup, residual sampling, bonus sampling, and
  batch/catch-up helpers with focused `V2_Unit_MTPRejectionSampler` coverage.
  CUDA/ROCm expose graph-capturable processed-logit row verifier and bonus
  sampler kernels that match the CPU reference, reject null/default streams,
  and are covered by the verifier+bonus+summary mini-transaction in
  `V2_Integration_GPUSamplingKernels`. The direct full-vocab perf lane
  `V2_Perf_GPUSpeculativeSummary.StochasticProcessedLogitQwen36Rows` now
  exists. The first optimization slice made processed-logit verification
  row-parallel, one block per verifier row, instead of serializing rows inside
  one block. Focused correctness still passes, and release smoke now shows
  CUDA reject/all-accept cases at about 1.00/1.25 ms and ROCm at about
  2.58/2.57 ms for three verifier rows. This is correct and graph-capturable,
  but not production-promoted until the next optimization slice reduces the
  remaining full-vocab stochastic verifier cost. The next accepted slice adds
  an optional device draft-token-probability vector to the processed-logit
  verifier, matching the vLLM worker idea that the sampled draft row already
  knows `q(sampled_token)`. CUDA/ROCm now skip draft full-row stats on accepted
  rows and compute them only when residual sampling is needed after a
  rejection. Focused guards passed:
  `V2_Unit_MTPRejectionSampler|V2_Integration_GPUSamplingKernels`, and release
  `StochasticProcessedLogitQwen36Rows` reports CUDA reject/prefix1/all-accept
  0.98/0.95/0.65 ms and ROCm 2.47/2.48/1.81 ms for three Qwen3.6-sized rows.
  Follow-up plumbing complete: compact device draft sampling can now write
  `p(sampled_draft_token)` into arena-owned `STOCHASTIC_DRAFT_SAMPLE_PROBS`
  without an extra kernel or sync. The shared CPU/CUDA/ROCm sampling helper
  reports the selected probability, CUDA/ROCm graph-captured sampler tests
  prove it on Qwen3.6 top-k/top-p rows, and
  `V2_Unit_GpuWorkspaceAllocationPolicy` covers the new arena buffer.
  A fused compact target-partials verifier prototype is correct and graph
  capturable, but it is not an accepted MoE performance path: focused Release
  perf shows only noise-level CUDA movement and a clear ROCm regression
  (rows 1/2/3 compact about 5129/5792/6520 us versus fused about
  6238/6884/7597 us). Do not wire this compact fusion into production. The
  processed-logit top-k/top-p warper slice is now implemented and correct for
  CUDA/ROCm: it builds full processed-logit rows from raw Qwen3.6-sized logits,
  preserves compact top-k/top-p probability semantics, rejects null/default
  streams, graph-captures with processed-logit sampling, and can publish the
  sampled-token probability. The regression
  `TopKTopPProcessedLogits_Qwen36VocabTopK40_MatchesCPUAndCaptures` covers the
  large-vocab, top-k=40, top-p=0.95, temperature=0.6 path on both GPU backends.
  Focused guards passed:
  `V2_Unit_MTPRejectionSampler|V2_Integration_GPUSamplingKernels` and release
  `V2_Perf_GPUSpeculativeSummary`. Production wiring is now aligned with the
  vLLM worker contract instead of the earlier compact-table shortcut:
  GPU runners build processed target logits on the explicit verifier stream,
  consume sampled draft tokens from device slots, and run the batched outcome
  verifier with `no_draft_probabilities=true` so `q` is one-hot. Penalty-free
  and history-dependent penalty rows use the batched verifier; penalty rows
  first apply their deterministic vLLM speculative branch history.
  Host-visible sampled tokens keep their device sample-slot readiness edge, so
  the batch verifier still consumes device draft-token slots. Focused runner
  units, GPU sampler capture, and Qwen3.6 CUDA/ROCm dense+MoE stochastic parity
  pass. The final cleanup removed the hidden full target/draft probability arena
  rows and scalar probability-row verifier from production GPU runners; new
  tuning should reduce verifier/condition economics rather than returning to
  full-probability or compact-table dead ends.
  A fused prefix-stop verifier experiment was benchmark-rejected and removed:
  it only modestly helped ROCm reject-at-0 and was worse for
  prefix-1/all-accepted cases, so it is not an accepted path.
- GPU kernels produce output tokens plus `num_accepted_tokens` without CPU
  participation. CPU uses scalar/AVX2/AVX512 dispatch plus OpenMP where useful.
  First GPU step complete for penalty-free SingleDevice all-position verifier:
  runner batches stochastic row verification through the existing device batch
  kernel, samples the bonus token into a device arena buffer, and summarizes
  output tokens plus accepted counts through a CUDA/ROCm shared-math reduction
  kernel. The penalty-free CUDA/ROCm lane now also defers verifier final sync
  into those target distribution and batch-summary kernels, verifies draft rows
  from device token buffers, and chains sidecar inputs from a stable
  device-resident condition-token buffer. Generic forward graphs now accept a
  stable device-token input source, the target verifier input row is composed
  into arena-owned device storage on the graph execution stream, and the first
  target sample can feed both the first sidecar and batched summary without a
  host scalar read. The compact device-batch outcome now lives in the shared
  `MTPRejectionSampler` contract, and the runner no longer passes a host
  draft-token shadow into the outcome verifier; CUDA/ROCm must read sampled
  draft tokens from device slots. Focused sampler, prefill/decode transition,
  DeviceGraphOrchestrator units, GPU sampling integration, and CUDA/ROCm
  stochastic graph smokes cover that boundary. The final Phase 4 closeout slice
  also split host-token and device-token first-sidecar graph caches and
  preserves sidecar replay state across accepted-state publication. Those fixes
  let ROCm sidecar contexts reach capture/replay instead of staying in warmup.
  The accepted dense stochastic matrix
  `benchmark_results/mtp_vllm_style/20260610T-phase4-dense-stochastic-closeout-matrix-v2/`
  is speed-positive in useful bounded lanes on all three backends: CUDA fixed d3
  59.44 vs 43.88 tok/s, ROCm fixed d1 33.48 and dynamic 32.57 vs 30.33 tok/s,
  and CPU fixed d2 5.78 vs 4.46 tok/s. ROCm fixed d2/d3 are documented
  acceptance-limited on this prompt, at 28.6% and 46.2% acceptance. The runner
  still receives compact outcome metadata on the host; completing fully
  device-resident publication is Phase 5/6 work, not a Phase 4 blocker.
- Greedy uses the same buffers and output contract, with argmax equality as the
  deterministic accept test.
- Retire decode-equivalent stochastic verifier use from accepted dense
  SingleDevice lanes. The remaining compatibility code is guarded for
  unsupported future topologies/features only and must not fire in the Phase 4
  dense gates.

Exit gate:

- Accepted for dense SingleDevice CPU/CUDA/ROCm as of the closeout gate:
  `V2_Unit_MTPRejectionSampler`, `V2_Unit_PrefillDecodeTransition`,
  `V2_Unit_MTPGraphConstruction`, `V2_Unit_DeviceGraphOrchestrator`,
  `V2_Unit_GpuWorkspaceAllocationPolicy`, `V2_Integration_GPUSamplingKernels`,
  dense Qwen3.6 stochastic parity on CPU/CUDA/ROCm, and CUDA/ROCm stochastic
  graph smokes all pass.
- CPU/CUDA/ROCm sampler parity passes on synthetic and Qwen3.6 real-logit-style
  fixtures for greedy, top-k/top-p, temperature, residual sampling, and seeded
  RNG.
- Dense stochastic MTP no longer emits the retired
  `decode_equivalent_stochastic_verifier_runs` counter in accepted lanes; parity
  and prefix-cache MTP probes assert the all-position publication path instead.
- Bounded stochastic dense benchmarks are speed-positive on each backend at
  least one fixed/dynamic lane, with ROCm d2/d3 documented as
  acceptance-limited rather than contract failures.

### Phase 5: Publish From Spec Slots, Not Checkpoints

Goal: make accepted-state publication cheap, atomic, and backend-neutral.

Status:

- Accepted. Focused slices added typed state-version diagnostics and moved
  the all-position verifier-base checkpoint skip behind the tested
  `makeLogicalMTPVerifierBaseSnapshot()` transaction helper. Guarded by
  `V2_Unit_MTPStateTransaction`, `V2_Unit_MTPGraphConstruction`,
  `V2_Unit_PrefillDecodeTransition`,
  `V2_Unit_GpuWorkspaceAllocationPolicy`, and the MTP perfstats/matrix script
  unit regressions. The first publication-cost slice shows stable per-publish
  cost across fixed d1/d2/d3 on CUDA, ROCm, and CPU; forced-reject replay is
  covered by a no-ready-token oracle and the stale all-position checkpoint tag
  was removed.

Work:

- Publish KV, shifted MTP KV, GDN recurrence, short-conv state, terminal hidden,
  terminal logits, sampler history, positions, and sequence lengths from
  accepted speculative slots.
- Keep checkpoint export/import only for prefix-cache restore and debug
  verification, not the steady MTP verifier path.
- Add state-version diagnostics that distinguish accepted publication,
  rejected correction, prefix restore, and session reset.
- Ensure CPU publication uses the same slot contract as CUDA/ROCm rather than a
  host-only checkpoint path.

Exit gate:

- Perfstats show publication cost is small and stable across d1/d2/d3 on CPU,
  CUDA, and ROCm.
- Forced reject parity proves the live state equals full replay after the next
  ordinary decode step.
- Dead checkpoint-dependent MTP publication code and tests are removed.

### Phase 6: Graph-Captured Draft/Verify/Sample/Publish

Goal: make the whole SingleDevice MTP step graph-shaped where the backend can
support it.

Status:

- Accepted on 2026-06-10. The dense CUDA/ROCm stochastic graph smokes now assert the actual
  d1 vLLM-style graph lifecycle: `main_verifier`, `mtp_decode_sidecar`, and
  `mtp_decode_catchup` must warm, capture, and replay during graph warmup, then
  replay again after `clearCache()`. The smoke intentionally does not require
  an ordinary `main_decode` replay in this lane because the ready
  prefill/accepted logits feed the target verifier directly. Focused gate:
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke` and
  `V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsStochasticSmoke` pass.
- ROCm verifier attention now matches CUDA for MTP continuation rows M=2..4.
  The previous ROCm M=2 limit made fixed-depth-3 verification fall through to
  a prefill-shaped path and produce wrong verifier tokens. Focused coverage:
  `V2_Unit_AttentionComputeStage_DynamicKVLen`,
  `FlashDecode_NativeFP16KV_MultiRowContinuationMatchesSerialRows`,
  ROCm/CUDA fixed-depth-3 parity, ROCm/CUDA dynamic parity, and both CUDA/ROCm
  stochastic graph smokes pass.
- CUDA and ROCm greedy graph smokes now use the same benchmark-style
  `prefill()` plus `decodeStep()` path as stochastic MTP and require
  `main_verifier`, `mtp_decode_sidecar`, and `mtp_decode_catchup` to
  warm/capture/replay. Focused gate:
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke` and
  `V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsSmoke` pass.
- CUDA and ROCm stochastic clear-cache repeatability now use a true
  long-context 768-token prompt plus 64 decode tokens, graph capture, seeded
  stochastic sampling, and penalties. The long gate caught a CUDA lifecycle
  split where one-row catch-up captured as `mtp_decode_sequential_catchup` but
  replayed as `mtp_decode_catchup`; `DeviceGraphOrchestrator` now uses one
  canonical `kMTPDecodeCatchupContext`, guarded by
  `V2_Unit_GpuWorkspaceAllocationPolicy`. Focused long gates:
  `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticClearCacheRepeatabilityLong`
  and
  `V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsStochasticClearCacheRepeatabilityLong`
  pass. Penalty-free CUDA/ROCm stochastic smokes defer final sync for
  `main_verifier`, `mtp_decode_sidecar`, and `mtp_decode_catchup`; penalty-bearing
  long-context stochastic runs keep the verifier boundary synchronized because
  target-row penalties depend on sampler history between accepted tokens.
  Final closeout covered CUDA/ROCm graph-stream stress parity, the broad
  `V2_Unit_` gate, and both integration/release builds. Release configuration
  now guards graph-stream parity test properties when non-perf tests are skipped.

Work:

- Capture draft prefill, one-token draft decode, target verifier, greedy
  sampling, stochastic distribution/rejection, and publication helpers with
  persistent buffers.
- GPU stochastic graph capture must include Qwen chat defaults: temperature,
  top-k, top-p, penalties where supported, and seeded RNG metadata.
- CPU keeps the same transaction boundaries and uses optimized kernels rather
  than graph capture.

Exit gate:

- CUDA and ROCm dense greedy/stochastic graph stress tests pass at long context.
- No GPU lane needs final verifier sync before sampling unless explicitly
  documented.
- Perfstats expose capture/replay, stream handoff, sampler, and publication
  counters for every MTP step.

### Phase 7: Dense Performance Acceptance

Goal: make dense SingleDevice performant before MoE-specific tuning.

Work:

- Run the bounded matrix every iteration and full default matrix at acceptance
  checkpoints.
- Tune M=1..4 GEMV/GEMM, GDN/short-conv publication, row-indexed LM-head, and
  dynamic-depth hysteresis using the same evidence across CPU/CUDA/ROCm.
- Build a generated dynamic-depth policy pipeline, mirroring the GEMM/GEMV
  dispatch trainer pattern:
  - collect prompt/device/mode rows from
    `scripts/run_mtp_depth_hysteresis_sweep.sh` and the standard iteration
    matrix;
  - derive train/holdout labels from same-run fixed d1/d2/d3 throughput,
    acceptance, verifier cost, and sampling mode;
  - train a compact deterministic policy surface offline;
  - emit a checked-in C++ `.inc` table consumed by `MTPDepthController`;
  - validate generated policy decisions against holdout prompts before any table
    is accepted.
  The generated policy must stay explainable: runtime code consumes binned
  window statistics and emits promote/hold/demote decisions, not a black-box
  runtime model. The controller remains deterministic and all fixed-mode
  behavior remains untouched.
- Keep CUDA, ROCm, and CPU correctness surfaces symmetric.

Status:

- Generated dynamic-depth policy side quest is implemented. The checked-in
  trainer `scripts/train_mtp_depth_policy.py` consumes matrix/hysteresis
  `summary.tsv` rows, derives fixed-depth labels, enforces deterministic
  train/holdout gates when requested, skips low-confidence generated rules, and emits
  `src/v2/execution/mtp/MTPDepthPolicyGenerated.inc`. `MTPDepthController`
  consumes that table only in dynamic mode, keeps fixed mode untouched, and
  reports generated promote/demote reasons through normal depth-policy stats.
  The table is now verify-mode-aware: greedy and stochastic rows do not share a
  single depth-2 acceptance threshold, which avoids promoting stochastic d2
  requests into a known-poor d3 lane just because acceptance is high.
- The runtime and benchmark config surfaces expose
  `mtp_depth_generated_policy`, and the hysteresis plus iteration-matrix scripts
  report whether each dynamic lane used the generated table.
- The policy trainer now keys fixed-depth examples by source summary plus
  topology, device, model, mode, decode length, request batch, and prompt case
  when present. This prevents separate short/long or scalar/request-batched
  benchmark summaries with the same backend/model/mode from overwriting each
  other before labels are derived. The trainer also learns bounded acceptance
  intervals instead of only high-acceptance promotions, so future generated
  rules can express low-to-moderate probe regions without hand-editing the
  `.inc` table. Regression: `V2_Unit_MTPDepthPolicyTrainer`.
- Focused side-quest gates passed:
  `V2_Unit_MTPDepthController`, `V2_Unit_MTPDepthPolicyTrainer`,
  `V2_Unit_PrefillDecodeTransition`, `V2_Integration_GPUSamplingKernels`, and
  CUDA/ROCm Qwen3.6 stochastic verifier parity. Older proving-ground coverage
  also includes `V2_Unit_MTPIterationBenchmarkMatrix` and
  `V2_Perf_MTPDepthController`.
- The latest policy refresh
  `benchmark_results/mtp_depth_hysteresis/20260611T-rocm-dense-mode-aware-policy-short-code/`
  retrains from dense fixed d1/d2/d3 rows plus ROCm short/text/code prompts.
  The checked-in table uses conservative thresholds: greedy d1 promotes at
  acceptance >=0.87, greedy d2 at >=0.73, stochastic d1 at >=0.50, and
  stochastic d3 demotes at <=0.83. Stochastic d2 promotion is intentionally
  absent because the current runtime features cannot separate cases where d3 is
  best from cases where d2 should hold. A regression test now proves ambiguous
  generated rules are skipped, and the depth-zero bypass regression proves a
  generated promote cannot override an all-zero window.
- Conservative generated-policy sanity
  `benchmark_results/mtp_depth_hysteresis/20260611T-rocm-dense-conservative-policy-sanity/`
  shows generated-on is now neutral on QBF, modestly positive on C++ and tech
  prompts, and slightly negative on the Python prompt. This is accepted as a
  safe seed table; restoring stochastic depth-2 promotion requires richer live
  features than acceptance rate alone.
- Dense ROCm/CUDA catch-up slice refreshed fixed d1/d2/d3 plus dynamic for
  greedy and stochastic. ROCm greedy is in the CUDA speedup class
  (`20260611T-rocm-dense-catchup-baseline/`: fixed d3 67.6 tok/s, 2.16x).
  ROCm stochastic has now caught CUDA by speedup class after the accepted
  top-k=40 specialization, batched target/bonus top-k/top-p distribution API,
  and the latest NativeVNNI graph-capture cleanup. The batched API is a
  backend/runner contract for contiguous all-position verifier rows; it uses
  declared orchestrator scratch, explicit streams, and no allocation or
  synchronization in the kernels. ROCm NativeVNNI small-M graph capture now
  defaults to workspace split-reduce, with atomic reduce kept as an explicit
  tuning opt-in. `20260611T-rocm-dense-stochastic-split-reduce/` reports fixed
  d2 at 42.1 tok/s versus 30.2 baseline (1.394x), while
  `20260611T-rocm-dense-stochastic-explicit-atomic-ab/` reports 41.7 tok/s
  (1.374x).
  The full ROCm depth matrix
  `20260611T-rocm-dense-stochastic-full-depth-matrix/` reports d1/d2/d3 at
  37.1/41.7/35.2 tok/s over 30.2 baseline, proving d2 is the best stochastic
  lane for this prompt. The final dynamic run
  `20260611T-rocm-dense-stochastic-dynamic-generated-d3-only/` holds depth 2,
  reaches 41.9 tok/s (1.385x), and records zero depth updates. CUDA reference
  `20260611T-cuda-rocm-dense-stochastic-long-d2-dynamic/` reports fixed d2
  64.4 tok/s (1.473x) and dynamic 59.1 tok/s (1.351x), so ROCm is accepted as
  the same speedup class even though its absolute tok/s still lags.
  Fresh iteration evidence
  `20260611T124556Z-rocm-dense-stochastic-refresh` confirms the accepted
  status with the standard baseline,d1,d2,d3,dynamic lane set at 64 decode
  tokens: ROCm dynamic reaches 42.60 tok/s over 30.29 baseline (1.41x),
  accepts 108 tokens, rejects 12, records 90% acceptance, and promotes to
  depth 2 without rollbacks. The remaining dense ROCm work is absolute
  verifier/condition throughput, not a correctness or policy blocker.
  Focused gates passed: `V2_Unit_MTPDepthController`,
  `V2_Integration_ROCm_NativeVNNI_GEMV`,
  `V2_Integration_ROCmQuantisedGemmSmallM`, and ROCm Qwen3.6 stochastic
  verifier parity. Remaining ROCm dense absolute-gap evidence points at
  verifier/sidecar work drained at the all-position stochastic batch outcome
  sync, about 6.8s in the 128-token run, rather than the policy or sampler
  enqueue path.
  A one-token condition-decode replay-preservation experiment was
  benchmark-rejected and removed because the bounded lane reached
  warmup/capture but not replay, dropping ROCm d2 to 32.23 tok/s.
- ROCm batched NativeVNNI generated dispatch is not accepted for runtime use.
  The 2026-06-11 dense guard proved microbench cosine is not a sufficient
  promotion gate: generated batched entries collapsed ROCm dense d2 acceptance
  to near zero, while the restored generic path kept the expected 80%
  acceptance. The trainer now resets KB/TW overrides before building its
  canonical reference, but future batched generated entries must pass a
  model/verifier-equivalence gate before runtime promotion.
- NativeVNNI decode dispatch training is now M-aware and shared across the CUDA
  and ROCm refresh path. CUDA sweep CSVs include `m`, the CUDA tree trainer
  keys features by `(M,N,K)`, exact overlay keys pack `M`, and the CUDA small-M
  runtime path consumes generated shape/tuning for verifier rows instead of
  using an N,K-only route. ROCm decode trainer and runtime already use the same
  M-aware key shape. `scripts/refresh_native_vnni_dispatch_tables.sh` is the
  canonical sweep -> train -> validate wrapper for CUDA and ROCm; it has a
  dry-run unit guard, a stratified `family-smoke` profile that runs one bounded
  sweep per requested format before combining CSVs, and can install validated
  generated includes. The compact CUDA smoke
  `benchmark_results/native_vnni_dispatch/20260611T063908Z-cuda-m-aware-refresh-smoke/`
  produced real Q4_1 M=1..4 rows and a validated generated include. Stratified
  CUDA/ROCm smoke refreshes
  `benchmark_results/native_vnni_dispatch/20260611T065536Z-cuda-family-smoke-stratified/`
  and
  `benchmark_results/native_vnni_dispatch/20260611T065515Z-rocm-family-smoke-stratified/`
  proved actual per-format partial CSV generation, CSV combine, training, and
  generated-include validation for representative simple and IQ codebook
  families. The wrapper unit test now also guards the default `family-smoke`
  inventory so CUDA includes its `Q8_0` extension while ROCm stays on the
  supported quantized weight families. Project CUDA/ROCm tuning skills document
  `family-smoke` as the bounded proxy and `qwen36`/`all` plus parity/benchmarks
  as the only table-install acceptance path. Focused gates passed:
  `V2_Unit_NativeVNNIDispatchRefreshScript`,
  `V2_Unit_CUDAGemvDispatchGeneratorAliases`,
  `V2_Unit_CUDAGemvDispatchBaseMerge`,
  `V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator`,
  `V2_Unit_ROCmNativeVNNITrainerCsvValidator`,
  `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`, and dense CUDA Qwen3.6
  depth-3 MTP parity. The wrapper now also exposes staged strict profiles:
  `qwen36-core` for Qwen3.6 FFN/GDN projections and `qwen36-lm-head` for the
  high-cost LM-head shape. ROCm `qwen36-core` completed without installing
  tables:
  `benchmark_results/native_vnni_dispatch/20260611T072617Z-rocm-qwen36-core-refresh/`
  generated 360 entries across 15 codebook families and passed generated
  codebook validation. The post-refresh focused gate passed
  `V2_Unit_NativeVNNIDispatchRefreshScript`,
  `V2_Unit_CUDAGemvDispatchGeneratorAliases`,
  `V2_Unit_CUDAGemvDispatchBaseMerge`,
  `V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator`,
  `V2_Unit_ROCmNativeVNNITrainerCsvValidator`,
  `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`,
  `V2_Integration_ROCm_NativeVNNI_GEMV`, and
  `V2_Integration_ROCmQuantisedGemmSmallM`. A first full CUDA `qwen36-core`
  attempt was stopped after two completed cases in roughly two minutes, because
  the full strict profile is a long-running acceptance job rather than an
  inner-loop gate. That attempt exposed a trainer stream-hygiene regression:
  the CUDA sweep harness called `multiply_tensor()` without binding an explicit
  stream. `Perf__CUDABlockwiseTensorCoreGemmSweep.cpp` now creates a
  non-blocking CUDA stream, binds it with `setGPUStream()`, records timing
  events on that stream, and unbinds/destroys it on every exit path. A bounded
  CUDA qwen36-core representative refresh,
  `benchmark_results/native_vnni_dispatch/20260611T081007Z-cuda-qwen36-core-representative-stream-bound/`,
  swept Q4_0, Q4_K, IQ2_XXS, and Q8_0 on the qwen36 FFN GateUp shape for
  M=1..4, generated/validated a smoke include, and proved non-null stream
  binding in the trainer log. The strict generator threshold correctly rejects
  that partial CSV as a production table, so it is recorded as a smoke artifact
  only. CUDA `qwen36-lm-head`, full `qwen36`/`all`, and model-level
  parity/benchmarks remain pending before broad checked-in table replacement.
  Focused follow-up gates passed `V2_Unit_Static_NoDefaultStreamInGPUCode`,
  `V2_Unit_GpuWorkspaceAllocationPolicy`, `V2_Unit_NativeVNNIDispatchRefreshScript`,
  `V2_Unit_CUDAGemvDispatchGeneratorAliases`,
  `V2_Unit_CUDAGemvDispatchBaseMerge`,
  `V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator`,
  `V2_Unit_ROCmNativeVNNITrainerCsvValidator`, and
  `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`. The ROCm trainer
  also gained an explicit `LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE` mode:
  normal/core profiles keep the FP32 hipBLAS health reference, while
  `qwen36-lm-head` defaults to `native-auto` so the giant LM-head shape can
  compare candidates against a reset-AUTO native output without materializing a
  multi-GB FP32 weight mirror. A one-format LM-head smoke,
  `benchmark_results/native_vnni_dispatch/20260611T075534Z-rocm-qwen36-lm-head-native-auto-smoke/`,
  passed for Q4_0/M=1 and generated one validated entry. The trainer now treats
  already-uploaded packed weights as valid, because first-use upload clears host
  packing buffers while keeping the device upload cache authoritative.
  CUDA LM-head smoke
  `benchmark_results/native_vnni_dispatch/20260611T081631Z-cuda-qwen36-lm-head-smoke/`
  passed for Q4_0/M=1 with the stream-bound trainer and generated one
  validated entry. This proves the huge LM-head shape is tractable in the
  staged pipeline, but all-format LM-head and model-level parity still gate any
  checked-in CUDA table update. Follow-up inspection found the CUDA M=2..4
  sweep path was labelling candidates while the specialized small-M dispatcher
  still used the current generated runtime route. `CUDANativeVNNIGemvTuned.cu`
  now consumes the sweep override for real KPAR small-M candidate launches, and
  the perf harness filters out WIDE/DIRECT/ROWPAR for M=2..4 because the
  VRAM-pool prepared harness can only execute KPAR verifier kernels today.
  The standard CUDA refresh family set is therefore `wide,kpar,direct` for M=1
  and executable KPAR rows for M=2..4; ROWPAR needs a future row-major-owner
  trainer before it can appear in production generated tables. Focused smoke
  `benchmark_results/native_vnni_dispatch/20260611T091538Z-cuda-smallm-real-candidate-smoke/`
  proved the corrected path with Q4_0 Qwen3.6 GDN time projection M=2:
  648 real KPAR rows, zero small-M failure logs, generated validation passed,
  and best tile 128x1/waves4/mkg4 at 13.312 us. The CUDA sweep trainer now
  uses deterministic valid packed tensors for dispatch sweeps instead of
  per-element random quantized fixtures, prepares/uploads/repackages each
  format+shape once before candidate timing, and sizes its
  `DeviceWorkspaceManager` budget from declared `IWorkspaceConsumer`
  requirements. This keeps giant LM-head refreshes practical while retaining
  the production tensor classes and GPU preparation path. The CUDA overlay
  generator fallback is also M-aware now, matching the base tree and exact
  `(M,N,K)` overrides; `V2_Unit_CUDAGemvDispatchBaseMerge` includes a split-M
  fixture where one LM-head shape wants WIDE/DIRECT at M=1 and KPAR at M=2..4
  and an alias-conflict fixture proving Q4_1/Q4_K style source-format winners
  collapse to one codebook-level runtime dispatch row before exact thresholds.
  A strict CUDA Q4_0 LM-head refresh,
  `benchmark_results/native_vnni_dispatch/20260611T094334Z-cuda-qwen36-lm-head-q4_0-full-candidates-maware/`,
  swept the full candidate grid for M=1..4 in about 62 seconds, produced 2708
  rows, passed generated validation, and reported 100% overall/fallback
  family/exact hit rates. Full CUDA all-format LM-head refresh,
  `benchmark_results/native_vnni_dispatch/20260611T094803Z-cuda-qwen36-lm-head-all-formats/`,
  completed the full M=1..4 candidate grid in about 19.5 minutes, wrote 51,452
  sweep rows, collapsed 76 source-format winners to 64 runtime dispatch keys,
  reconciled 6 alias-conflict keys, and generated a validated include with
  100% final family/exact/fallback hit rates. Full CUDA qwen36-core refresh,
  `benchmark_results/native_vnni_dispatch/20260611T101337Z-cuda-qwen36-core-all-formats/`,
  completed the six Qwen3.6 core FFN/GDN shapes across all CUDA decode formats
  in about 10.8 minutes, wrote 308,712 sweep rows, observed KPAR as the best
  family for all 456 source-format winners, collapsed them to 384 runtime
  dispatch keys, reconciled 64 alias-conflict keys, and generated a validated
  include with 100% final family/exact hit rates. Combined CUDA qwen36 artifact,
  `benchmark_results/native_vnni_dispatch/20260611T102638Z-cuda-qwen36-combined-from-staged/`,
  was generated from the staged core plus LM-head CSVs without rerunning GPU
  sweeps. It covers 360,164 rows, 532 source-format winners, 448 runtime
  dispatch keys, 70 alias-conflict keys, and validates with 100% final
  family/exact hit rates. Full ROCm LM-head refresh
  `benchmark_results/native_vnni_dispatch/20260611T082004Z-rocm-qwen36-lm-head-full/`
  completed without installing tables. It ran all 18 ROCm text formats across
  M=1..4, produced 72 best rows, collapsed aliases into 60 generated dispatch
  entries across 15 codebook ids, and passed generated codebook validation.
  Every completed row matched the `native-auto` reference with cosine 1.0.
  IQ3_S/IQ3_XXS and IQ1_S/IQ1_M are correct but show weaker M>1 LM-head
  speedups than the Q/K/IQ2 families, so they are follow-up tuning candidates
  after model-level parity accepts any table promotion. Combined ROCm qwen36
  artifact,
  `benchmark_results/native_vnni_dispatch/20260611T102802Z-rocm-qwen36-combined-from-staged/`,
  was generated from the staged ROCm core plus LM-head CSVs without rerunning
  kernels. It covers 6048 candidate rows across 18 ROCm formats and 7 Qwen3.6
  shapes, and emits 420 generated dispatch entries across 15 codebook ids. The
  post-refresh
  generated-dispatch gate passed
  `V2_Unit_Static_NoDefaultStreamInGPUCode`,
  `V2_Unit_GpuWorkspaceAllocationPolicy`,
  `V2_Unit_NativeVNNIDispatchRefreshScript`,
  `V2_Unit_CUDAGemvDispatchGeneratorAliases`,
  `V2_Unit_CUDAGemvDispatchBaseMerge`,
  `V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator`,
  `V2_Unit_ROCmNativeVNNITrainerCsvValidator`, and
  `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`.
  The combined CUDA and ROCm generated tables are now installed in
  `CUDANativeVNNIGemvDispatchHeuristicGenerated.inc` and
  `ROCmNativeVNNIDecodeDispatchGenerated.inc`. Promotion evidence passed:
  the generated-dispatch unit/static gate, `V2_Integration_ROCm_NativeVNNI_GEMV`,
  `V2_Integration_ROCmQuantisedGemmSmallM`, and the symmetric dense Qwen3.6
  CUDA/ROCm MTP parity gate covering fixed d1/d3, dynamic depth, forward-only
  equivalence, stochastic verifier smoke, and benchmark-prompt known-window
  diagnostics. The only failure found during promotion was not a generated-table
  regression: ROCm teacher-forced benchmark-prompt parity hits a documented
  PyTorch FP32 versus quantized ROCm near-tie at decode step 6, where ROCm ranks
  token 4338 at 20.802 over PyTorch token 1092 at 20.793. The harness now keeps
  exact PyTorch-token checks before that row, asserts the near-tie remains small,
  and leaves long MTP checks comparing against the backend no-MTP baseline.
  Installed-table benchmark matrix
  `benchmark_results/mtp_vllm_style/20260611T-post-generated-dispatch-dense-cuda-rocm/`
  covered dense CUDA/ROCm greedy and stochastic baseline,d1,d2,d3,dynamic rows
  at 64 decode tokens. CUDA remains speed-positive: greedy d3 reaches
  91.4 tok/s (2.08x over 44.0 baseline) and stochastic d3 reaches
  65.7 tok/s (1.49x over 44.0 baseline). ROCm greedy is also speed-positive:
  d3 reaches 65.0 tok/s (2.14x over 30.3 baseline). Follow-up perfstats first
  exposed stochastic rejection-condition cost as the ROCm blocker; the later
  `20260611T124556Z-rocm-dense-stochastic-refresh` matrix closes that bounded
  lane with dynamic at 42.60 tok/s over 30.29 baseline (1.41x), 90%
  acceptance, and zero rollbacks. Dense CUDA/ROCm relative speedup is now
  accepted; future dense work should target ROCm absolute verifier/condition
  throughput without weakening the shared sampler or parity gates.
- ROCm default `family-smoke` now runs through the full supported decode
  codebook inventory. The first all-format attempt exposed Q4_K/M=3 and Q2_K/M=1
  as FP32 hipBLAS health-gate false negatives rather than dispatch mismatches:
  Q4_K/M=3 is covered by an expanded Qwen3.6 GDN time-projection packed native
  contract regression, and `V2_Integration_ROCm_NativeVNNI_GEMV` plus
  `V2_Integration_ROCmQuantisedGemmSmallM` remain the exact dispatch-equivalence
  gates. After documenting those ROCm trainer health gates, the all-format smoke
  `benchmark_results/native_vnni_dispatch/20260611T070818Z-rocm-family-smoke-all-formats/`
  generated 60 decode entries across 15 codebooks and passed validation.
- CUDA default `family-smoke` also runs through its full decode inventory,
  including the CUDA-only `Q8_0` codebook. The all-format smoke
  `benchmark_results/native_vnni_dispatch/20260611T071116Z-cuda-family-smoke-all-formats/`
  swept 58,235 candidate rows, trained the fallback tree, layered 76 exact
  known-shape overrides across 16 codebooks, and passed generated codebook
  validation. This exposed a policy issue rather than a kernel issue:
  `family-smoke` now uses proxy hit-rate thresholds while `qwen36`/`all` keep
  strict CUDA fallback-family/exact thresholds for production table acceptance.
  After fixing the CUDA small-M relabel/fallthrough bug, the corrected
  all-format smoke
  `benchmark_results/native_vnni_dispatch/20260611T091656Z-cuda-family-smoke-all-formats-corrected-smallm/`
  produced 51,452 executable rows, covered 16 codebook ids, generated 76
  known-shape overrides, validated the generated include, and had zero
  small-M failure logs. This corrected artifact supersedes the earlier CUDA
  family-smoke evidence for M=2..4 trainer behavior.
- Dynamic warm-start cleanup is accepted for the bounded dense stochastic lane.
  `MTPDepthController` now resolves an unset dynamic initial depth to depth 2
  when the configured range allows it, while preserving explicit depth-zero
  bypass. `scripts/run_mtp_iteration_benchmark_matrix.sh` no longer hard-pins
  dynamic rows to `--mtp-initial-draft-tokens 1`, so the matrix measures the
  runtime policy default. Runtime-default checks:
  `20260611T-rocm-dense-stochastic-dynamic-runtime-default-d2/` reports ROCm
  dynamic at 34.34 tok/s, 1.10x, 80% acceptance, zero updates; and
  `20260611T-cuda-dense-stochastic-dynamic-runtime-default-d2/` reports CUDA
  dynamic at 54.20 tok/s, 1.21x, 80% acceptance, zero updates.
- Deepest-lane dynamic policy is now generated-table only. Handwritten fallback
  promotion still handles shallow probes, but it no longer enters the maximum
  draft depth on perfect or ambiguous lower-depth windows. This keeps the
  default stochastic prompt on the proven fixed-d2 lane instead of paying d3
  probes, while preserving generated greedy d2-to-d3 promotion where the table
  has evidence.
- Prefix/MTP full-hit restore regression is fixed. Prefix harvest now refreshes
  the terminal hidden row from the just-finished prefill before storing a
  terminal MTP block, so a stored block no longer advertises MTP state while
  lacking the terminal hidden needed by the sidecar. Focused CUDA/ROCm
  Qwen3.6 dense `PrefixCacheMTPRestore` and
  `PrefixCacheMTPDynamicDepthRestore` parity tests pass.

Exit gate:

- Dense greedy and stochastic are correct on CPU/CUDA/ROCm.
- CUDA and ROCm post comparable speedup classes versus their no-MTP baselines;
  if one backend lags, it gets a tuning pass before acceptance.
- Dynamic approaches the best fixed depth for the prompt class after warmup.
- The generated dynamic-depth policy trainer has unit coverage for CSV parsing,
  holdout evaluation, and generated `.inc` output; controller unit tests prove
  generated recommendations are bounded by min/max depth and do not affect fixed
  policy mode.

### Phase 8: MoE SingleDevice Parity With Dense Contract

Goal: run Qwen3.6 MoE through the same transaction, metadata, sampler, and
publication contract as dense.

Work:

- Reuse the dense transaction driver for MoE.
- Make routed/shared expert sidecar and verifier stages graph-native with
  workspace-declared scratch.
- Persist only continuation state; expert routing payloads, histograms, and
  sparse scratch remain transient.
- Ensure CUDA and ROCm use the same MoE strategy before backend-specific tuning.

Status:

- First ROCm MoE tuning slice landed a backend parity fix with direct perf
  impact. ROCm `softmax_topk` now mirrors CUDA's block-wide parallel top-k
  selection instead of scanning all experts on thread 0 after softmax. The
  kernel preserves the previous ascending expert-id tie order, leaves router
  probability rows intact for diagnostics, rejects null/default streams and
  unsupported bounds, and is covered by
  `Test__ROCmMoEKernel.SoftmaxTopKParallelSelectionPreservesTieOrder` plus the
  existing verifier-shaped small-M router regression.
- Evidence: `20260611T-rocm-moe-parallel-topk/` reduced ROCm MoE stochastic
  fixed-d2 verifier router time from 291.8 ms to 51.9 ms and verifier total
  from 951.8 ms to 768.4 ms in the profiled lane. The non-profiled bounded
  matrix `20260611T-rocm-moe-parallel-topk-matrix/` moved fixed d2 from the
  previous 33.3 tok/s to 43.2 tok/s against a same-run 68.1 tok/s baseline.
  ROCm Qwen3.6 MoE stochastic verifier parity passed after the change.
- Full-ownership SingleDevice GPU MoE now advertises sidecar main-state
  preservation, matching the dense transaction contract. The predicate is
  intentionally ownership-based rather than enum-based: CUDA/ROCm SingleDevice
  production graphs may use the `ExpertParallel` label while still owning the
  full expert set (`local_expert_count < 0`, no overlay plan). CPU and sparse
  ExpertParallel overlays remain conservative. The focused unit
  `Test__DeviceGraphOrchestrator.SidecarMainStatePreservationIsInitializedAndTopologyBounded`
  covers this boundary, and CUDA/ROCm Qwen3.6 MoE stochastic parity passed with
  `LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE=1`.
- Evidence: `20260611T-rocm-moe-sidecar-preserve-fullowner-d2/` removes
  `all_position_verifier_base_restores`, records
  `all_position_verifier_base_restore_skipped_sidecar_preserved`, and lets
  `main_verifier` reach segmented replay. ROCm stochastic fixed d2 moved to
  46.1 tok/s. The matching CUDA lane
  `20260611T-cuda-moe-sidecar-preserve-fullowner-d2/` also skips restore and
  reaches verifier replay, with fixed d2 at 62.5 tok/s.
- Long-lane MoE evidence is now the sprint steering signal:
  `20260611T144241Z-moe-cuda-rocm-longlane` shows CUDA MoE remains
  speed-negative in greedy and stochastic even at d2/d3, while ROCm greedy can
  barely exceed baseline only through dynamic policy and ROCm stochastic remains
  negative. A backend-neutral attempt to force ROCm shared-expert verifier rows
  onto grouped prefill was benchmark-rejected:
  `20260611T145646Z-moe-rocm-shared-grouped` regressed ROCm greedy d2/d3/dynamic
  to 69.1/64.7/60.5 tok/s. The CUDA tile_m sweep also found no stable default
  promotion.
- Focused verifier-prefill perf/parity coverage now exists as
  `V2_Perf_MoEVerifierPrefill`. It exercises CUDA and ROCm M=1/2/3/4 routed
  top-k and shared expert rows at the production Qwen3.6 MoE shape
  (`d_model=2048`, `intermediate=512`, 256 routed experts), compares grouped
  verifier prefill against row-wise decode-equivalent rows, and times eager plus
  graph-replay execution. The release CTest gate passed with reduced iteration
  counts for sprint use, and the short CSV run showed graph replay is already
  sub-millisecond for these kernels: CUDA routed M1/M2/M3/M4 =
  0.154/0.168/0.183/0.192 ms, CUDA shared = 0.099/0.105/0.114/0.118 ms,
  ROCm routed = 0.242/0.266/0.337/0.339 ms, ROCm shared =
  0.144/0.158/0.212/0.227 ms, all with cosine 1.0 against decode-equivalent
  output. That shifts the next Phase 8 tuning target away from isolated
  grouped prefill itself and toward full verifier economics: routed/shared FFN
  cost across all layers, rejection condition replay, and sidecar LM-head /
  sampling work.
- Fresh clean MoE depth sweep with perfstats:
  `20260611T_moe_perfstats_depth_sprint`. CUDA greedy baseline/d1/d2/d3/dynamic
  = 136.5/83.6/97.0/106.1/81.8 tok/s; CUDA stochastic =
  137.1/79.8/84.3/85.4/78.4. ROCm greedy =
  76.5/75.6/78.2/83.1/81.0; ROCm stochastic =
  76.4/59.1/59.0/61.6/56.4. Acceptance is healthy enough that draft quality is
  not the primary blocker: CUDA greedy d3 is 84.4%, ROCm greedy d3 is 85.3%,
  and ROCm stochastic d2 is 86.3%.
- ROCm exact combined shared-gate verifier prefill now uses an IQ4_NL byte-pair
  decode table for the Qwen3.6 shared expert path. The production-shaped
  speedometer improved ROCm graph replay from about 0.702 ms to 0.506 ms with
  cosine 1.0 against the split routed+shared reference; CUDA on the same shape
  is about 0.350 ms. `V2_Integration_ROCmMoEKernel` and focused CUDA/ROCm exact
  verifier perf gates pass after the change.
- The production-shaped combined shared-gate verifier speedometer now covers
  the fixed-depth target-row counts M=2/3/4 instead of only the depth-3 M=4
  case. Reduced direct run evidence: CUDA graph replay 0.308/0.330/0.351 ms,
  ROCm graph replay 0.444/0.462/0.509 ms, all cosine 1.0 against the split
  routed+shared reference. The full `V2_Perf_MoEVerifierPrefill` CTest passed,
  so future MoE tuning can use this curve as the per-depth kernel baseline.
- Fresh post-IQ4 full MoE GPU matrix:
  `20260612T_moe_gpu_post_iq4pair_matrix`. CUDA remains speed-negative in every
  MoE lane despite high acceptance: greedy baseline/d1/d2/d3/dynamic =
  139.2/84.2/98.9/107.2/106.3 tok/s and stochastic =
  139.6/81.3/90.8/96.9/105.5. ROCm greedy dynamic is the first barely
  speed-positive GPU MoE lane at 81.6 tok/s versus 77.7 baseline, but fixed
  depths remain negative; ROCm stochastic remains negative at
  49.0/48.9/38.8/50.8 versus 77.4 baseline. Perfstats show CUDA is limited by
  verifier plus condition-forward economics, while ROCm still attributes large
  time to the compact greedy/stochastic outcome sync boundary.
- Correction-replay small-M routing was tested and rejected in
  `20260611T_moe_correction_replay_sprint`. Splitting the one-token rejected
  correction condition forward into a distinct graph signature and forcing the
  verifier-prefill MoE route regressed the same-run full matrix: CUDA greedy d3
  moved from 106.1 to 97.8 tok/s and ROCm greedy d3 from 83.1 to 71.4 tok/s.
  The experiment has been removed so future tuning does not inherit a dead-end
  graph mode.
- Stage attribution from `20260611T_moe_stage_timing_probe` shows the next
  optimization should stay on full graph economics rather than per-expert
  correctness. CUDA main-verifier d3 is dominated by routed FFN (~46 ms over
  the short probe) plus shared FFN (~43 ms), with GDN/router support work next.
  ROCm main verifier is dominated by routed FFN (~78-80 ms), then shared FFN,
  router, GDN, GEMM, and attention. Sidecar attribution shows LM head as the
  largest sidecar stage, so sidecar LM-head/sampling fusion is the next
  second-order target once verifier FFN economics are improved.
- MoE remains speed-negative, so Phase 8 is not accepted. The next bottleneck
  is verifier/condition/sidecar transaction cost, not publication, router
  correctness, isolated grouped prefill, or verifier-base restore churn. For
  stochastic MoE specifically, the vLLM processed-logit/one-hot-q verifier is
  functionally green on CUDA and ROCm, but the same-run matrix is still
  speed-negative. Continue from profiler evidence on verifier, condition, and
  queued GPU sampling work rather than reviving compact-table or full-prob row
  verifier variants.
- Fresh GPU stage-timing probe
  `20260612T225841Z-moe-stochastic-gpu-stage-timing` keeps the conclusion
  sharp: CUDA best dynamic is 81.6 versus 114.4 tok/s baseline, and ROCm best
  d1 is 51.5 versus 68.7 tok/s baseline. The new `--gpu-stage-timing` evidence
  shows routed expert FFN dominates both main-decode condition rows and
  main-verifier rows. The stochastic D2H bucket is mostly the host-visible
  synchronization boundary draining queued model work, not the primary kernel
  target. The next Phase 8 slice should therefore reduce repeated full MoE
  condition/verifier transaction work before deeper sampler tuning.
- vLLM source inspection confirms that the known-good path samples the bonus
  row up front and processes target verifier rows as a batch. Llaminar's
  processed-target, greedy-draft/one-hot-q stochastic verifier is therefore
  architecturally aligned at the sampler level. The remaining speed gap is that
  Llaminar benchmarks one request at a time, so every MoE speculative step pays
  a tiny-batch 40-layer target/condition transaction. The next Phase 8
  implementation slice should add request-batched speculative transaction
  support and a benchmark lane that measures amortized target verification,
  rather than reviving compact-table sampler shortcuts or lazy bonus sampling.
- First request-batching groundwork is in shared metadata: accepted-count
  verifier outcomes can now build one padded multi-request metadata batch with
  flattened verifier state slots, while unknown rejected device draft ids stay
  invalid instead of being synthesized on host. Focused gates:
  `V2_Unit_MTPSpecDecodeMetadata`, `V2_Unit_MTPSpecStateContract`, and the
  broader MTP unit gate passed. The next implementation step is to feed these
  batched outcomes from a runner/benchmark path instead of only unit fixtures.
- The request-batch intent knob is now explicit: `MTPRuntimeConfig` carries
  `max_request_batch`, CLI/YAML accept `--mtp-max-request-batch` /
  `max_request_batch`, benchmark JSON and the iteration matrix summary export
  it. Early revisions hard-failed values other than 1; the live greedy
  SingleDevice path now treats it as capacity/intent instead of disabling
  ordinary scalar MTP decode.
- A shared `MTPSpecTransactionDriver` now builds a full batch transaction plan
  from accepted outcomes or greedy catch-up: metadata, commit plan,
  publication plan, and per-request `MTPSpecStepPlan` are produced in one
  checked object. `OrchestrationRunner` consumes this driver for the current
  single-request all-position publication path, so the future request-batched
  scheduler path will attach to the same accepted-count semantics instead of
  cloning runner-local metadata construction. Focused gates:
  `V2_Unit_MTPSpecStateContract` and `V2_Unit_PrefillDecodeTransition`.
- The greedy all-position path now has the same batched verifier-to-publication
  adapter as accepted-count stochastic outcomes. `MTPDecodeCatchup` splits one
  compact sampled-row batch into per-request results, metadata construction
  builds one padded multi-request greedy batch, and
  `MTPSpecTransactionDriver` returns a single publication plan for mixed
  all-accepted and rejected requests. Focused gates:
  `V2_Unit_MTPDecodeCatchup`, `V2_Unit_MTPSpecDecodeMetadata`, and
  `V2_Unit_MTPSpecStateContract`.
- Compact verifier scratch now scales with request-batch intent instead of the
  historical four-row constant. `mtp_target_query_rows` resolves to
  `max(4, max_request_batch * (draft_tokens + 1))`, the Qwen/Qwen3/Qwen3.5
  schemas use it for `lm_head_input_rows`, and CPU graph construction proves a
  two-request/depth-two capacity by comparing six row-indexed verifier logits
  against full all-position logits. Focused gates:
  `V2_Unit_MTPGraphConstruction`, `V2_Unit_QwenStandardGraphSchema`,
  `V2_Unit_Qwen3BufferSizes`, and `V2_Unit_Qwen35BufferSizes`.
- Compact verifier row selection now consumes an explicit verifier row plan
  instead of assuming leading rows. `GraphConfig` and `IGraphBuilder` carry the
  selected-row vector, `DeviceGraphOrchestrator` installs it from the MTP
  verifier input plan, and Qwen graph construction validates the rows against
  the real verifier activation tensor. GPU cached graphs still read row indices
  from metadata workspace, while CPU graph construction gets the same logical
  plan through the builder. Focused gate:
  `V2_Unit_MTPGraphConstruction.RowIndexedAllPositionLogitsRespectExplicitVerifierRowsOnCPU`.
- Verifier graph execution now has an explicit logical-to-padded materializer.
  `MTPSpecDecodeVerifierGraphForwardPlan` splits flattened request verifier
  tokens into `forward_batch()` input batches, maps compact verifier and bonus
  rows into padded graph-row coordinates, and feeds those rows into both
  metadata upload and CPU graph construction. `DeviceGraphOrchestrator`
  forwards actual per-request sequence lengths during padded batch execution
  and restores cumulative request state afterwards. Focused gates:
  `V2_Unit_MTPSpecDecodeMetadata`, `V2_Unit_MTPGraphConstruction`, and the
  bounded MTP/schema unit gate.
- Runner verifier forwards now go through `MTPVerifierForwardExecutor`.
  The helper materializes graph coordinates once, routes single-request host
  tokens through `forward()`, single-request device-token rows through
  `forwardWithDeviceTokenIds()`, and multi-request host tokens through
  `forward_batch()`. Multi-request device-token rows now use the explicit
  runner-local `forwardBatchWithDeviceTokenIds()` contract: callers supply
  logical host shadows plus a padded flat device buffer, and unsupported
  topologies hard-fail rather than sharing one raw pointer across participants.
  Focused gates:
  `V2_Unit_MTPVerifierForwardExecutor` and the bounded MTP/schema unit gate.
- Greedy request-batched verifier transactions are now executable as a shared
  helper. `executeMTPGreedyVerifierBatchTransaction()` enables row-indexed
  verifier logits, installs the compact row plan, executes the padded batch
  forward, samples compact rows, cleans up row mode on success or failure, and
  returns one batched transaction/publication plan. Focused gate:
  `V2_Unit_MTPVerifierForwardExecutor`. The same helper is now proven against
  the real CPU `DeviceGraphOrchestrator` verifier graph, including paired
  all-position/row-indexed mode ownership. Focused gate:
  `V2_Unit_MTPGraphConstruction`.
- Device-reduced stochastic outcomes now feed the same request-batched
  transaction driver. `buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes()`
  validates compact CUDA/ROCm-style outcomes against request shapes, converts
  them to accepted-count metadata, and produces the same flattened
  commit/publication/step plans as greedy catch-up. Focused gate:
  `V2_Unit_MTPSpecStateContract`.
- The live single-request GPU stochastic path now consumes that device-outcome
  transaction helper instead of rebuilding accepted outcomes locally. This
  keeps today's request-1 publication semantics on the same code path future
  request-batched scheduling will use. Focused gate:
  `V2_Unit_PrefillDecodeTransition`.
- Qwen35/Qwen36 MTP sidecar graph construction now allows bounded one-token
  request batches. The graph still rejects more than four total MTP rows,
  non-KV full sidecars with `seq_len != 1`, and multi-token catchup shapes
  that are not single-request. Focused gate: `V2_Unit_MTPGraphConstruction`.
- `DeviceGraphOrchestrator` can now publish one terminal-hidden row per
  request for a one-token batch. The multi-row row-select helper accepts the
  producer stream just like the single-row selector, preventing GPU stream
  races when this path is wired into graph-captured request batching. It still
  rejects multi-token per-request batches. Focused gate:
  `V2_Unit_MTPGraphConstruction`.
- Batched verifier forwards with an installed MTP row plan now route through
  the decode graph cache instead of the ordinary batched prefill path, giving
  accepted-state publication the exact padded verifier graph it must restore
  from. `DeviceGraphOrchestrator::publishAcceptedMTPSpecStateBatch()` now
  validates common padded shape, publishes KV per request index, restores
  stage state from each request's physical verifier row, updates per-request
  positions, and packs terminal-hidden rows atomically. Mixed zero-accepted
  shifted-KV batches still hard-fail before mutation until the scheduler owns
  correction replay for those lanes. Focused gates:
  `V2_Unit_MTPGraphConstruction` and the bounded Phase 8 unit cluster.
- Live all-position publication now goes through the batch contract end to
  end for request-count one. `OrchestrationRunner` calls
  `publishAcceptedMTPSpecStateBatch()`, and `RankOrchestrator` clamps each
  request through the common-prefix helper before publishing the batch on
  every LocalTP or LocalPP participant. This removes the side-door single-step
  publication dependency from the vLLM-style path while keeping unsupported
  participants as hard failures. Focused gates:
  `V2_Unit_RankOrchestrator`, `V2_Unit_PrefillDecodeTransition`, and the
  bounded Phase 8 unit cluster.
- Request-batch admission now has a first-class scheduler contract.
  `MTPSpecRequestBatchScheduler` groups pending requests in stable order,
  admits only matching mode/topology/vocab shapes, preserves variable verifier
  token counts within the configured padded shape, and records deferred versus
  rejected reasons before any runner state mutates. `MTPVerifierForwardExecutor`
  now accepts a scheduled greedy batch
  through a narrow adapter and feeds the existing padded verifier transaction
  helper, proving scheduler output is executable without teaching the
  scheduler about runner entrypoints. Focused gates:
  `V2_Unit_MTPSpecRequestBatchScheduler`, `V2_Unit_MTPVerifierForwardExecutor`,
  and the bounded MTP metadata/transaction cluster.
- Batched device-token verifier rows now have a named SingleDevice runner
  contract. `DeviceGraphOrchestrator::forwardBatchWithDeviceTokenIds()` builds
  a padded host shadow for bookkeeping, preserves per-request sequence lengths,
  and routes embedding through the caller-owned flat device token buffer. This
  removes the executor-layer single-row limitation, while Rank/TP/PP paths
  still hard-fail until they own per-participant device-token buffers. Focused
  gates: `V2_Unit_MTPVerifierForwardExecutor` and the bounded Phase 8 unit
  cluster.
- Greedy verifier transaction helpers now carry
  `MTPVerifierForwardExecutionOptions`, so scheduled request batches can select
  the padded device-token runner contract instead of being stuck on host-token
  `forward_batch()`. Focused gate: `V2_Unit_MTPVerifierForwardExecutor`.
- Request-batch scheduler admission now treats verifier input placement as part
  of the batch shape. Homogeneous host-token and device-token batches are
  admitted, mixed placement is deferred before mutation, and the scheduled
  executor hard-fails if placement and `device_token_ids` disagree. Focused
  gates: `V2_Unit_MTPSpecRequestBatchScheduler` and
  `V2_Unit_MTPVerifierForwardExecutor`.
- Request-batch ownership now has a two-phase CPU contract. The new
  `MTPSpecRequestBatchOwner` reserves scheduled requests without removing them,
  rejects duplicate ids and in-flight mutations, commits only admitted request
  ids after publication succeeds, and releases reservations unchanged after a
  failed verifier/publication transaction. This gives benchmark/server batching
  a concrete handoff point instead of letting scheduler output silently drop
  live requests. Focused gate: `V2_Unit_MTPSpecRequestBatchOwner`.
- Owned greedy request-batched verifier execution now has a single helper that
  schedules through the owner, executes the existing padded verifier
  transaction, commits admitted requests on verifier-only success, and releases
  the reservation unchanged on forward/sampling/transaction failure. A
  publication-aware variant now takes a caller-supplied accepted-state publisher
  and commits only after that publisher succeeds; publication failure releases
  the reservation without dropping pending requests. This proves the next
  benchmark/server batch lane can use scheduler output without reimplementing
  ownership cleanup or committing before live-state publication. Focused gate:
  `V2_Unit_MTPVerifierForwardExecutor`.
- Owned stochastic request-batched publication now has the matching
  device-outcome coordinator. The owner reserves an admitted stochastic batch,
  a producer callback returns compact `MTPDeviceRejectionBatchOutcome` rows for
  exactly that batch, shared transaction planning validates accepted counts and
  publication slots, and the owner commits only after the caller's
  accepted-state publisher succeeds. Producer, planning, or publication failure
  releases the reservation unchanged. Focused gates:
  `V2_Unit_MTPVerifierForwardExecutor`, `V2_Unit_MTPSpecRequestBatchOwner`,
  `V2_Unit_MTPSpecRequestBatchScheduler`, `V2_Unit_MTPSpecStateContract`, and
  `V2_Unit_MTPRejectionSampler`.
- Request-batch intent now reserves runner capacity before runner construction.
  `RuntimeConfig::fromOrchestrationConfig()` resolves effective `batch_size`
  from the larger of `--batch-size` and enabled `--mtp-max-request-batch`, and
  named-domain runners consume the resolved value instead of the raw CLI value.
  Graph/state capacity can no longer be silently under-sized when request
  batching is enabled.
  Focused gate: `V2_Unit_PrefixMTPConfig`.
- Benchmark prefill/decode now has an explicit request-batched runner contract.
  `IInferenceRunner` exposes `supportsPrefillBatchForBenchmark()`,
  `prefillBatchForBenchmark()`, `supportsDecodeStepBatchForBenchmark()`, and
  `decodeBatchStepForBenchmark()`, and `BenchmarkRunner` uses those paths
  whenever enabled MTP requests `max_request_batch > 1`. The benchmark treats
  `n_predict` as a per-request target, reports aggregate emitted tokens across
  the admitted batch, and keeps request-0 text/tokens only for compact human
  inspection. Runners that only support request-0 prefill or single-request
  decode hard-fail instead of producing fake batched measurements. Focused
  gate: `V2_Unit_BenchmarkRunnerCPU`.
- The request-batched benchmark path now has matching orchestration-level
  landing zones. `IOrchestrationRunner` exposes `supportsPrefillBatch()`,
  `prefillBatch()`, `supportsDecodeStepBatch()`, and `decodeStepBatch()`, and
  `InferenceRunnerAdapter` forwards those results into the benchmark contract.
  The default remains an explicit unsupported result until each topology wires
  the request owner, scheduler, verifier, and publication callbacks into live
  state. SingleDevice greedy has started replacing that unsupported result.
  Focused gate: `V2_Unit_BenchmarkRunnerCPU`.
- `OrchestrationRunner::prefillBatch()` now owns the first live SingleDevice
  request-batch state boundary. It validates MTP config, prefix-cache and
  topology exclusions, MPI single-rank execution, and runner batch capacity,
  calls `forward_batch()` once for the admitted prompt rows, records per-request
  terminal-token readiness, and blocks scalar `decodeStep()` while that batched
  live state is active. `clearCache()` releases the batched state. Later Phase 8
  slices consume these slots through `decodeStepBatch()`. Focused gate:
  `V2_Unit_PrefillDecodeTransition`.
- `OrchestrationRunner::decodeStepBatch()` now consumes the ready terminal
  prefill logits for each live SingleDevice request slot without looping scalar
  `decodeStep()` or re-feeding request 0. The bridge validates per-request
  sequence metadata, samples or consumes each row's terminal token, and records
  generated token state. Later Phase 8 slices have extended this bridge into
  the request-owner sidecar/verifier/publication transaction below. Focused
  gate: `V2_Unit_PrefillDecodeTransition`.
- Variable-length request-batched prefill can now publish one stable
  terminal-hidden row per request for MTP sidecar input. `DeviceGraphOrchestrator`
  records the most recent per-request forward lengths, maps padded hidden rows
  as `request * padded_seq_len + actual_len - 1`, and uses the existing
  graph-native `HiddenStateRowsSelectStage` to gather those rows into
  `PREFIX_TERMINAL_HIDDEN`. The old multi-token batched rejection test is now a
  positive variable-length regression. Focused gates:
  `V2_Unit_MTPGraphConstruction` and the bounded Phase 8 request-batch cluster.
- `DeviceGraphOrchestrator` now exposes a real request-batched greedy MTP
  sidecar draft producer through `forwardMTPBatchAndSampleGreedy()`. The graph
  runs as `seq_len=1, batch_size=request_batch` instead of looping scalar
  sidecars, requires per-request positions, and projects every compact
  `MTP_HIDDEN` row into `MTP_LOGITS` by setting the MTP sidecar LM head to
  `compute_all_positions=true`. The focused regression proves two request rows
  produce finite hidden/logit rows and valid draft tokens; shifted-prefill
  single-request multi-token catchup remains separately covered by exact perf
  tags. Focused gates: `V2_Unit_MTPGraphConstruction` and the bounded Phase 8
  unit cluster.
- Live SingleDevice greedy request-batched continuation is now wired for
  depths 1, 2, and 3. `decodeStepBatch()` builds per-request sidecar condition rows from
  the already-emitted prompt-logit tokens, runs one true batched sidecar draft,
  then batched chained sidecar drafts for deeper fixed depths, schedules an
  owned greedy verifier batch, publishes the returned `MTPSpecStepPlanBatch`
  atomically, advances each request's sequence length, and returns only the
  newly committed suffix so the first token is not emitted twice. The next
  ready bonus token is cached per request and consumed without another verifier
  forward. `supportsDecodeStepBatch()` now advertises the same d1/d2/d3 greedy
  capability the live path executes, and request-batch states own independent
  sampler histories used by the stochastic continuation described below.
  Focused gates:
  `V2_Unit_PrefillDecodeTransition`, `V2_Unit_MTPGraphConstruction`, and the
  bounded Phase 8 unit cluster.
- Live SingleDevice stochastic request-batched continuation now executes the
  bounded vLLM-style path for depth 1 through 3. It reuses the true batched
  greedy sidecar draft producer, runs one padded target verifier forward for
  the scheduled request batch, then reduces each request's compact stochastic
  outcome through runner-owned device draft slots and the shared
  `MTPSpecTransactionDriver` publication contract. Each request owns an
  independent sampler, including the bonus-token sampler commit when the device
  summary reports a sampled terminal token, so seeded stochastic rows keep the
  same per-request semantics as scalar decode. Compact stochastic reduction is
  still delegated to the single-request reducer today, but the
  scheduler/owner/publication transaction is already batched. Focused gates:
  `V2_Unit_PrefillDecodeTransition` and the bounded Phase 8 unit cluster
  (`V2_Unit_PrefillDecodeTransition`, `V2_Unit_BenchmarkRunnerCPU`,
  `V2_Unit_MTPVerifierForwardExecutor`, `V2_Unit_MTPSpecRequestBatchOwner`,
  `V2_Unit_MTPSpecRequestBatchScheduler`, `V2_Unit_MTPSpecStateContract`,
  `V2_Unit_MTPRejectionSampler`, and `V2_Unit_MTPGraphConstruction`).
- Stochastic request-batch scratch now scales from the runtime MTP capacity
  instead of the scalar four-target/three-draft shape. GPU runners allocate
  target/bonus rows as `max(4, max_request_batch * (draft_tokens + 1))`,
  draft sample rows as `max(3, max_request_batch * draft_tokens)`, per-request
  reduced output rows as `[max_request_batch, output_fields]`, and matching
  top-k partial scratch through the arena. `decodeStepBatch()` maps target and
  bonus slots to compact verifier rows, while draft slots are packed without
  bonus gaps. Focused regressions prove a two-request depth-two stochastic
  batch uses target slots `0/3`, bonus slots `2/5`, and draft slots `0/2`
  rather than clobbering slot zero, and a GPU-gated arena-shape guard proves
  two-request depth-three scratch allocates 8 target slots, 6 draft slots, and
  two compact output rows. The implementation still reduces stochastic
  summaries once per request; the next slice should add a single multi-request
  summary/reduction kernel before benchmark acceptance. Focused gate:
  `V2_Unit_MTPGraphConstruction` and the bounded Phase 8 unit cluster.
- Stochastic request-batch outcome handoff is now atomic at the runner
  contract. `DeviceStochasticBatchOutcomeRequest` value-owns thresholds,
  stop tokens, slot coordinates, bonus rows, and vLLM rejection RNG metadata;
  `decodeStepBatch()` builds/stages every scheduled request first, then calls
  `verifyStochasticDistributionsRequestBatchOutcomesOnDevice()` once. The
  mock regression proves one runner-level outcome call for a two-request
  depth-two batch while retaining target slots `0/3`, bonus slots `2/5`, and
  draft slots `0/2`. The default runner implementation delegates to the
  existing single-request reducer, while GPU runners can override the same
  contract with compact batched output rows. Focused gate: bounded Phase 8
  unit cluster.
- `DeviceGraphOrchestrator` now overrides that request-batch handoff with a
  compact GPU path. It consumes the pending verifier stream once, enqueues each
  request's verifier, bonus sampler, and existing summary reducer into a
  distinct `[request, fields]` arena row, then performs one compact D2H copy
  for all request outcomes. This removes the repeated per-request stream drain
  while preserving the proven CUDA/ROCm summary kernels; a fused backend
  multi-request summary launch is now a benchmark-driven follow-up rather than
  a correctness prerequisite. Focused gate: bounded Phase 8 unit cluster.
- Request-batched GPU prefill and verifier metadata now share the compact row
  upload machinery without confusing their state machines. GPU prefill records
  direct terminal graph rows for compact request-batch sampling, while MTP
  verifier forwards keep using the explicit logical verifier-row plan. This
  fixed the CUDA MoE stochastic RB=2 smoke failure where a prefill row-indexed
  graph tried to consume a nonexistent MTP row plan. Focused gates:
  `V2_Unit_MTPSpecDecodeMetadata`,
  `V2_Unit_PrefillDecodeTransition`, and the bounded Phase 8 unit cluster.
- Request-batched stochastic verifier forward now distinguishes ordinary
  request-batched prefill from all-position verifier continuation before
  setting compact terminal-logit row metadata. This avoids the old hard failure
  when `forward_batch()` was called under `compute_all_position_logits=true`
  for the verifier. Focused gate: bounded Phase 8 unit cluster.
- Qwen3.5/Qwen3.6 GDN and short-conv state-capture rows now scale through
  `resolveMTPMaxTargetQueryRows(config.mtp)` instead of `draft_tokens + 1`.
  Multi-request verifier graphs therefore declare enough recurrence capture
  rows for flattened request batches such as RB=2/d1 and RB=2/d3. Focused
  gate: `V2_Unit_MTPGraphConstruction`.
- Mixed stochastic request batches now stay lockstep when one lane accepts all
  drafts and another rejects. The all-accepted lane emits its bonus-ready token
  inline but does not publish bonus recurrent/KV state; the next verifier step
  consumes that token as the condition from the accepted-prefix state, matching
  the existing deferred-ready contract without leaving half the batch in
  terminal-prefill sampling. Regression:
  `RequestBatchedStochasticMixedReadyAndRejectStaysLockstep`.
- The first real CUDA MoE stochastic request-batched benchmark smoke is green:
  `llaminar2 benchmark -m Qwen3.6-35B-A3B-UD-IQ3_S.gguf -d cuda:0
  --n-predict 16 --seed 123 --mtp --mtp-draft-tokens 1
  --mtp-depth-policy fixed --mtp-verify-mode speculative-sampling
  --mtp-max-request-batch 2 -c 4096` completed with 74.38 tok/s decode,
  14 accepted tokens, 76 rejected tokens, 90 verifier runs, and zero rollbacks.
  This proves the functional Phase 8 path for CUDA RB=2 stochastic; acceptance
  remains open until CUDA/ROCm request-batch matrices show MoE stochastic
  speedup against same-run scalar baselines.
- ROCm MoE request-batched stochastic now reaches the same functional point.
  Root cause of the previous warmup failure was a backend route gap: MTP
  request batching enters the small-M verifier router with BF16 gate weights,
  but ROCm's small-M route only accepted FP32. `ROCmMoEKernel` now dispatches
  small-M router rows for FP32, FP16, and BF16 through explicit-stream HIP
  wrappers. Regressions
  `SmallMBF16GateLogits_ModelShapeMatchesSingleTokenLaunches` and
  `SmallMBF16FusedRouter_VerifierShapeRunsWithTensorGate` prove BF16 small-M
  logits match the existing scalar BF16 path and that `routeWithTensors()` uses
  the dtype-aware small-M path. Focused `V2_Integration_ROCmMoEKernel` passes.
  The real ROCm smoke
  `llaminar2 benchmark -m Qwen3.6-35B-A3B-UD-IQ3_S.gguf -d rocm:0
  --n-predict 16 --seed 123 --mtp --mtp-draft-tokens 1
  --mtp-depth-policy fixed --mtp-verify-mode speculative-sampling
  --mtp-max-request-batch 2 -c 4096` completed with 49.53 tok/s decode,
  9 accepted tokens, 81 rejected tokens, 90 verifier runs, and zero rollbacks.
  That makes RB=2 stochastic functionally green on CUDA and ROCm, but Phase 8
  remains performance-red: both backends are slower than scalar/no-MTP
  baselines, and RB=2 acceptance is much lower than the scalar seeded lane.
- Request-batched stochastic terminal-prefill sampling now uses the same
  vLLM-style logical-position threshold contract as scalar MTP. The old GPU
  path sampled compact prefill rows through the backend RNG counter, so
  CUDA MoE RB=2 could choose a different first token than scalar MTP even when
  row logits were identical. `OrchestrationRunner::decodeStepBatch()` now
  computes per-request thresholds from each request sampler and logical
  position, and `DeviceGraphOrchestrator::sampleMainLogitsBatchRowsOnDevice()`
  builds compact top-k/top-p rows before sampling on the explicit GPU stream.
  Regression `RequestBatchedStochasticGpuPrefillUsesPositionKeyedThresholds`
  covers the handoff. Fresh evidence in
  `benchmark_results/mtp_vllm_style/20260613T_phase8_rb2_stochastic_prefill_fix/`
  shows scalar CUDA MoE stochastic `-n1` and RB=2 `-n1` both generate token
  `[271]`. CUDA/RB=2 and ROCm/RB=2 MoE stochastic `-n16` both pass with
  12 accepted, 78 rejected, and zero rollbacks at 75.90/52.12 tok/s. Same-run
  scalar d1 is 88.18/56.47 tok/s and no-MTP is 115.32/69.79 tok/s, so the
  remaining Phase 8 blocker is batching policy/transaction economics, not
  prefill-row stochastic correctness.
- Request-batched RoPE metadata is now graph-capture safe on CUDA and ROCm.
  `RoPEStage` owns its kernel instance because stream/workspace/dynamic-position
  validity is graph-node local, and `prepareGraphLaunch()` uploads either the
  scalar pos-offset buffer or explicit position-row buffer before capture/replay.
  CUDA/ROCm kernels now honor explicit position IDs as row-buffer data even when
  the current values are numerically contiguous, preventing accidental fallback
  to stale scalar metadata. Regression
  `V2_Integration_(CUDA|ROCm)RoPEGraphCaptureNoH2D` covers scalar and explicit
  row replay. The real ROCm Qwen3.6 MoE RB=2 greedy smoke
  `llaminar2 benchmark -m Qwen3.6-35B-A3B-UD-IQ3_S.gguf -d rocm:0
  --mtp --mtp-draft-tokens 1 --mtp-max-request-batch 2 --mtp-verify-mode
  greedy -t 0 --n-predict 16` completed without `MTP0_rope` segmented-capture
  fallback at 69.29 tok/s, 13 accepted, 75 rejected, and zero rollbacks.

Exit gate:

- MoE CPU/CUDA/ROCm greedy and stochastic parity passes with the same tests as
  dense plus MoE layer-by-layer math analysis.
- MoE bounded matrix is speed-positive for greedy and stochastic or has a
  measured route/acceptance bottleneck.
- No MoE path uses dense-only fallbacks.

Closure status:

- Feature/correctness gate closed on 2026-06-13 for SingleDevice
  CPU/CUDA/ROCm dense and MoE. Request-batched stochastic MoE is now covered
  by the shared vLLM-style transaction path and by CUDA/ROCm real-model RB=2
  smokes with zero rollbacks.
- Performance gate remains red and moves to the tuning dashboard: MoE
  stochastic request batching currently lowers acceptance versus scalar on
  the default prompt, and scalar MoE stochastic is still slower than no-MTP.
  Future work should tune batching policy, verifier/condition transaction
  amortization, and MoE stochastic economics without changing the Phase 8
  correctness contract.

### Phase 9: Multi-Device Promotion

Goal: extend the accepted SingleDevice contract to TP/PP/ExpertParallel without
changing its semantics.

Work:

- LocalTP/GlobalTP participants share the same draft tokens, target rows,
  accepted counts, and rollback/publication decision.
- PP stages publish only their local layer state but agree on global accepted
  token counts.
- ExpertParallel participants execute sparse no-op/dispatch/return stages in a
  symmetric sequence, with placement fingerprints included in prefix/MTP state.

Status:

- First CPU/unit slice landed the shared common-prefix contract:
  `coordinateMTPSpecCommonAcceptedPrefix()` clamps participant-local
  `MTPSpecStepPlan` publication to the minimum accepted state count and marks
  divergent participants as requiring common fallback replay. This gives
  LocalTP, GlobalTP/NodeLocalTP, LocalPP, and ExpertParallel one reusable rule:
  no participant may publish verifier state past the common accepted prefix.
  Focused gate: `V2_Unit_MTPSpecStateContract`.
- LocalTP runtime fan-out now exists for accepted spec-state publication.
  `RankOrchestrator::supportsMTPSpecStatePublication()` only advertises the
  capability when every child runner supports it, and
  `publishAcceptedMTPSpecState()` coordinates the plan through the shared
  common-prefix helper before publishing on every child via the TP worker pool.
  A failed or unsupported participant fails the rank operation instead of
  silently publishing a partial topology. Focused gate:
  `V2_Unit_RankOrchestrator`; bounded MTP gate:
  `V2_Unit_(RankOrchestrator|MTPSpecStateContract|MTPSpecDecodeMetadata|MTPSpecDecodeTransaction|MTPDecodeCatchup|MTPRejectionSampler|MTPVerifierPolicy|GpuWorkspaceAllocationPolicy|PrefillDecodeTransition)`.
- LocalTP chained sidecar drafts are no longer single-child only.
  `RankOrchestrator::supportsChainedMTPDrafts()` now requires every child to
  support depth-2/3 sidecar chaining, and `forwardMTPFromLastDraft()` fans the
  same draft token plus shifted-cache position to every participant through the
  TP worker pool. This makes fixed d2/d3 LocalTP MTP reachable under the same
  all-child hard-fail contract as rank-level publication. Focused gate:
  `V2_Unit_RankOrchestrator`; bounded MTP gate same as above.
- LocalTP shifted-prefill MTP embedding now follows the vocab-parallel contract
  on CUDA/ROCm. `EmbeddingStage` passes both the local vocab range and the
  "out-of-shard rows may be zero" permission to the GPU embedding kernels, so
  device-token validation no longer mistakes an in-process LocalTP shard for a
  broken single-device embedding. This also makes FP32 GPU embedding launches
  respect explicit vocab offsets. Focused gates:
  `V2_Unit_VocabParallelEmbeddingSharding`,
  `V2_Unit_EmbeddingStage_GraphCapture`, and
  `V2_Integration_Parity_Qwen36_LocalTP_Qwen36LocalTPPrefixMTPParity_MTPGreedyMatchesPyTorchDecodeTokens`.
- LocalTP dynamic depth is now enabled through the same rank-wide
  `OrchestrationRunner` controller used by SingleDevice. The controller chooses
  one draft depth for the request step and `RankOrchestrator` fans that depth
  out to every child, so participants do not adapt independently. PP and
  GlobalTP/MPI remain hard-gated until they have explicit scalar depth
  coordination. Focused gates: `V2_Unit_PrefillDecodeTransition`,
  `V2_Integration_Parity_Qwen36_LocalTP_Qwen36LocalTPPrefixMTPParity_MTPGreedyDepth3MatchesPyTorchDecodeTokens`,
  and
  `V2_Integration_Parity_Qwen36_LocalTP_Qwen36LocalTPPrefixMTPParity_MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens`.
- NodeLocalTP fixed d2/d3 MTP now uses the same all-participant chained
  sidecar contract. `GlobalOrchestrator` advertises chained draft support only
  when every stage runner supports it, and `forwardMTPFromLastDraft()` fans the
  same draft token plus shifted position to every rank-local participant.
  Dynamic depth now broadcasts rank 0's scalar controller decision before each
  step so all ranks execute the same sidecar/verifier shape. Focused gates:
  `V2_Unit_PrefillDecodeTransition`,
  `V2_Integration_Parity_Qwen36_NodeLocalTP_Qwen36NodeLocalTPPrefixParity_MTPGreedyDepth3MatchesPyTorchDecodeTokens`,
  `V2_Integration_Parity_Qwen36_NodeLocalTP_Qwen36NodeLocalTPPrefixParity_MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens`,
  and full `V2_Integration_Parity_Qwen36_NodeLocalTP_`, which is green for
  five real-model tests plus fixture after this slice.
- LocalTP all-position verifier sampling now consumes the verifier graph replay
  stream exactly once per child runner and reuses that handoff for every sampled
  verifier row. This closes the race where LocalTP could sample row logits on a
  child default stream before a graph-captured verifier replay had completed.
  Focused gate: `V2_Unit_RankOrchestrator`.
- ExpertOverlay Qwen3.6 MoE parity now covers ROCm2TP-hot plus CPU2LocalTP-cold
  greedy MTP and prefix-restore MTP. The fixed causes were missing ROCm
  local-expert nested workspace declarations and GPU MoE parity not enabling the
  deterministic reduction-order mode on ROCm near-tie prompts. Focused gates:
  `V2_Unit_MoELocalExpertStage_PreparedWeights`,
  `V2_Unit_RankOrchestrator`, and full
  `^V2_Integration_Parity_Qwen36MoE_ExpertOverlay_`.
- The tuning dashboard now tracks SingleDevice, LocalTP, LocalPP, NodeLocalTP,
  and ExpertOverlay separately for implementation, parity, and benchmark state.
  LocalPP fixed-depth dense MTP is now implemented through a final-stage
  sidecar delegation path: the terminal PP stage receives shifted-prefill
  tokens, owns the MTP sidecar weights plus embedding table, and builds MTP KV
  append/attention with cache-local sidecar layer ids instead of subtracting
  the main PP offset. Non-terminal PP stages reject sidecar weights. Dynamic
  depth is enabled through the same central `OrchestrationRunner` controller as
  SingleDevice/LocalTP, so PP stages do not adapt independently. LocalPP
  all-position publication is now implemented as an all-stage contract:
  non-final stages publish verifier main KV/GDN state only, while the final
  stage owns logits, stochastic device outcome verification, terminal-hidden
  row selection, and shifted sidecar KV publication. Device-token handoff
  remains gated for PP because verifier token input starts at stage 0 while
  final-stage sampler slots live on the pipeline tail. Focused gates:
  `V2_Unit_WeightManagerPPSafety`, `V2_Unit_RankOrchestrator`,
  `V2_Unit_PrefillDecodeTransition`, and full
  `^V2_Integration_Parity_Qwen36_LocalPP_`, which is green for prefix restore,
  fixed d1/d3 MTP, dynamic MTP, stochastic MTP, and prefix+MTP restore.
- The standard benchmark matrix runner now has an explicit topology axis and a
  leading `topology` summary column. `single` remains the default; opt-in
  presets generate the tested command shapes for `localtp_rocm2`,
  `localtp_cuda2`, `localpp_rocm2`, `nodelocaltp_cpu2`,
  `expert_overlay_rocm2_hot`, and `expert_overlay_rocm2_cpu2`. The script
  fails fast for unsupported model/topology pairings so multi-device evidence
  cannot accidentally mix dense-only and MoE-only lanes. Regression:
  `V2_Unit_MTPIterationBenchmarkMatrix`.
- The matrix runner now exposes `--gpu-stage-timing`, which requires
  `--perfstats` and sets `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` for MTP
  perfstats rows. Use this for bounded diagnostics when CUDA/ROCm MoE aggregate
  timers hide graph-stage GPU work behind deferred sync points. Runs now also
  emit `stage_summary.tsv`, ranking decode-domain `mtp` and `stage_gpu` timers
  per topology/device/model/mode/variant so Phase 8 and Phase 9 tuning evidence
  stays comparable across SingleDevice, TP, PP, and ExpertOverlay lanes.
  Regression: `V2_Unit_MTPIterationBenchmarkMatrix`.
- Greedy compact device-outcome verification is now an explicit runner
  capability rather than a GPU-wide assumption. Single-device and final-stage
  PP runners may use the compact path; multi-child LocalTP uses the existing
  sharded verifier-row sampler until a true cross-participant compact reducer
  exists. This keeps LocalTP from hard-failing inside an unsupported compact
  verifier while preserving the topology-hard-fail contract for genuinely
  advertised capabilities. Focused gates: `V2_Unit_PrefillDecodeTransition`
  and `V2_Unit_RankOrchestrator`. Fresh ROCm topology smoke
  `20260612T232547Z-rocm-topology-dense-greedy-smoke-capability-fix` is green:
  LocalTP d1 accepted 12/12 at 34.4 vs 36.7 tok/s, and LocalPP d1 accepted
  12/12 at 40.9 vs 31.4 tok/s.
- Fresh bounded ROCm dense greedy topology matrix
  `20260612T234446Z-iteration-matrix-3ed9c37e` is green with same-run
  baseline/fixed/dynamic evidence. LocalTP ROCm2: baseline 34.1 tok/s, d1 34.6
  (1.01x), d2 34.3 (1.00x, 80% acceptance), d3 55.4 (1.62x), dynamic 54.2
  (1.59x). LocalPP ROCm2: baseline 30.3 tok/s, d1 44.0 (1.45x), d2 39.8
  (1.32x, 80% acceptance), d3 55.5 (1.83x), dynamic 62.9 (2.08x). All MTP
  lanes completed with zero rollbacks; stage timing shows the remaining cost is
  verifier graph replay and sidecar work, not publication failure.

Exit gate:

- LocalTP, NodeLocalTP, LocalPP, and ExpertParallel parity suites pass for dense
  and MoE where hardware exists.
- Multi-device MTP never lets one participant publish a longer prefix than the
  common accepted count.

Closure status:

- Feature/correctness gate closed on 2026-06-13 for the recorded Phase 9
  topology set. Dense LocalTP, LocalPP, and NodeLocalTP parity suites are
  present and previously recorded green; ExpertOverlay MoE parity is recorded
  green for ROCm2TP-hot plus CPU2LocalTP-cold. The focused Phase 9 unit guard
  `V2_Unit_(RankOrchestrator|MTPIterationBenchmarkMatrix|MTPSpecStateContract|PrefillDecodeTransition)`
  passed after rebuilding the interface-dependent multi-device test binary.
- Performance/tuning remains dashboard-owned. ROCm dense LocalTP/LocalPP
  lanes are already speed-positive, while NodeLocalTP and ExpertOverlay
  benchmark presets still need refreshed same-run matrices before any rollout
  claim.

### Phase 10: Default-Enablement Evidence

Goal: decide rollout from measured correctness and speed, not optimism.

Work:

- Refresh the dashboard after every iteration.
- Compare CUDA against llama.cpp anchors and keep ROCm within the same class of
  speedup before considering backend acceptance.
- Capture CPU separately with realistic expectations but the same correctness
  gates.

Exit gate:

- Green requires correctness plus speed-positive MTP against same-run no-MTP
  baseline.
- Dense and MoE have separate acceptance records for greedy and stochastic.
- Any default enablement proposal names the exact backend/model/sampling lanes
  that passed parity and benchmark gates.

Current status:

- Phase 10 remains open as the active performance/default-readiness phase.
  Dense CUDA/ROCm SingleDevice and ROCm dense LocalTP/LocalPP have speed-positive
  evidence, but MoE stochastic on CUDA/ROCm and MoE CPU lanes remain red or
  amber in the dashboard.
- Fresh MoE stochastic request-batch evidence confirms correctness without
  performance acceptance. Scalar RB=1 remains speed-negative on the default
  prompt (`20260613T100145Z-moe-stochastic-single-rb1`: CUDA best d3 83.8
  versus 114.6 tok/s baseline; ROCm best dynamic 53.0 versus 69.0). RB=2 is
  worse (`20260613T100431Z-moe-stochastic-single-rb2`: CUDA best d1 70.2
  versus 115.2; ROCm best d1 46.5 versus 69.1, with d2/d3 acceptance around
  4-12%). The next Phase 10 sprint should reduce MoE stochastic
  verifier/condition-token cost in the scalar path before revisiting larger
  request batches.
- Long-lane RB=1 evidence sharpens, but does not close, the gap:
  `20260613T101458Z-moe-stochastic-long-rb1` shows CUDA fixed d3 nearly neutral
  at 136.4 versus 138.6 tok/s baseline, and ROCm fixed d2 neutral at
  77.0 versus 77.0 tok/s baseline while ROCm dynamic remains poor at 62.0. A
  generated-depth-policy refresh using that evidence was benchmark-rejected
  (`20260613T_phase10_depth_policy_refresh_rocm_moe_stoch`): ROCm dynamic still
  demoted back to d1 and reached only 60.3 versus 77.7 tok/s baseline. The
  checked-in generated table therefore remains unchanged; the promoted slice is
  only the trainer grouping/interval regression.
- No default-enable proposal is allowed until the active dashboard matrix has
  same-run parity and benchmark evidence for the exact backend/model/sampling
  lanes under consideration.

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
  -R "^V2_Unit_(PrefixMTPConfig|MTPDepthController|MTPDecodeCatchup|MTPRejectionSampler|MTPSpecDecodeTransaction|MTPSpecDecodeMetadata|MTPSpecStateContract|MTPSpecKVPublisher|MTPStateTransaction|MTPVerifierPolicy|MTPWeightManifest|MTPGraphConstruction|PrefillDecodeTransition|PrefillGraphCacheIntegration|ForwardExecutionEngineAdvanced)" \
  --output-on-failure --parallel
```

### Generated Dispatch Gate

Run after touching NativeVNNI sweep, trainer, generated include, or CUDA/ROCm
decode dispatch code:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Unit_Static_NoDefaultStreamInGPUCode|V2_Unit_GpuWorkspaceAllocationPolicy|V2_Unit_NativeVNNIDispatchRefreshScript|V2_Unit_CUDAGemvDispatchGeneratorAliases|V2_Unit_CUDAGemvDispatchBaseMerge|V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator|V2_Unit_ROCmNativeVNNITrainerCsvValidator|V2_Unit_NativeVNNIGeneratedDispatchCodebooks" \
  --output-on-failure --parallel
```

Use `scripts/refresh_native_vnni_dispatch_tables.sh --backend both --profile qwen36`
for table refreshes. Install generated tables only after model-level parity and
benchmark acceptance for the affected backend/model lanes.

### Functional/Parity Gate

Run the relevant available lanes for any touched backend:

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Integration_Parity_Qwen36.*(PrefixMTP|Math|GraphStreamStress)|^V2_Integration_PrefixCacheMTP_Qwen36.*(GpuGraphs|Smoke|Prefix)" \
  --output-on-failure --parallel
```

This must cover, as applicable:

- Dense CPU/CUDA/ROCm greedy MTP and prefix restore.
- Dense CPU/CUDA/ROCm stochastic MTP. CPU may use host kernels, but it must use
  the same batched verifier/rejection contract; CUDA/ROCm must use
  device-resident stochastic verification.
- Dense CPU/CUDA/ROCm layer-by-layer math prefill/decode parity.
- Seeded stochastic sampler parity for saved real-model logits must be symmetric
  across CPU/CUDA/ROCm so backend drift cannot hide behind aggregate counters.
- Dense CUDA/ROCm GPU graph smokes.
- MoE CPU/CUDA/ROCm layer-by-layer math prefill/decode parity.
- MoE CUDA greedy MTP parity/style tests.
- MoE CPU/CUDA/ROCm stochastic verifier parity and deterministic reuse after
  `clearCache()`.
- ROCm MoE ExpertOverlay parity remains separate from SingleDevice acceptance.

### Mandatory Benchmark Matrix Gate

Refresh JSON/perf evidence for the same SingleDevice device matrix on every
tuning iteration: CUDA, ROCm, and CPU; dense and MoE; greedy and stochastic;
no-MTP baseline, fixed d1, fixed d2, fixed d3, and dynamic depth. Multi-device
Phase 9 lanes use the same script with `--topologies` presets so LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay evidence lands in the same schema. This
matrix is the normal tuning instrument, not an occasional acceptance run. Greedy rows use
production runtime settings with `--temperature 0`, not `--deterministic`;
stochastic rows use a pinned seed, default `123`, so acceptance and throughput
can be compared across iterations. Dynamic depth must always be reported beside
the fixed d1/d2/d3 rows from the same git hash and runtime configuration; do not
tune or accept dynamic in isolation. The standard matrix dynamic lane starts at
d1 and keeps d1 as the adaptive floor; d0 bypass must be run as an explicit
diagnostic until it is proven faster than d1 on a matching benchmark. The
generated `summary.tsv` includes
`speedup_vs_baseline` for every MTP row plus perfstats-derived verifier health:
`verifier_ms`, `condition_ms/count/skipped_ready`, `rejection_no_ready`,
`correction_ms`, `publish_ms/count/avg_ms`,
`sidecar_ms`, `sidecar_depth0_decode_ms`, `shifted_*_ms`, `sampling_ms`,
`shifted_kv_ready_events/waits/syncs_deferred`, `checkpoint_ms`,
`sidecar_graph_hits/misses`,
`shifted_initial_commits/reused`,
`main_decode_warmup/capture/replay`, `main_verifier_warmup/capture/replay`,
and replay reset/preserve counts. Use
those fields to explain a speed regression before changing kernels or depth
policy.

Use `cpu:0` for the SingleDevice CPU lane. Bare `cpu` auto-selects two-socket
CPU TP and belongs to a later multi-device/TP matrix, not this gate.

For CPU dynamic-depth work, also run the focused policy-overhead perf test so
controller cost stays separated from model verifier/publication cost:

```bash
ctest --test-dir build_v2_release -R "^V2_Perf_MTPDepthController$" \
  --output-on-failure --parallel
```

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

When aggregate MTP timers are ambiguous, add graph-stage GPU event timing to the
same bounded matrix shape:

```bash
scripts/run_mtp_iteration_benchmark_matrix.sh \
  --decode-tokens 16 --perfstats --gpu-stage-timing
```

For Phase 9 multi-device evidence, select the topology preset under active
work. These rows are intentionally opt-in because they require matching local
hardware or MPI process availability:

```bash
scripts/run_mtp_iteration_benchmark_matrix.sh \
  --topologies localtp_rocm2,localpp_rocm2,nodelocaltp_cpu2 \
  --models dense --decode-tokens 16 --perfstats

scripts/run_mtp_iteration_benchmark_matrix.sh \
  --topologies expert_overlay_rocm2_cpu2 \
  --models moe --modes greedy --decode-tokens 16 --perfstats
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
If a row cannot run because of hardware availability, build failure, timeout, or
runtime crash, record that explicit reason in the dashboard instead of leaving
the row stale.
