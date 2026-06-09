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
  contract. Stochastic verification consumes target logits/probs, draft
  probabilities, uniform/residual randoms, and emits output tokens plus
  `num_accepted_tokens`.
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
| Stochastic dense | Correct, but short-lane speed-negative and CPU evidence includes stepwise cost | Batched rejection sampler over target/draft probs |
| CPU verifier | Full target forward plus batched all-position LM-head rows | Target/bonus row LM head and shared metadata buffers |
| MoE | Functionally green, speed-negative | Graph-native sidecar plus batched verifier/rejection |
| State publication | Captured-stage restore works | Spec state slots are the primary live-state mechanism |
| Graph capture | Improving per backend | Draft, verify, sample, publish are captured where possible |

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
- Bounded dense iteration matrix covers CUDA/ROCm/CPU, greedy/stochastic,
  baseline, fixed d1/d2/d3, and dynamic at 16 decode tokens. Latest full matrix
  is `benchmark_results/mtp_vllm_style/20260609T061226Z-iteration-matrix-6753b5e7/`.
  Greedy is speed-positive on all three backends, with best lanes CUDA d3 66.7
  vs 44.6 tok/s, ROCm d1 43.9 vs 31.4 tok/s, and CPU d3 9.1 vs 4.6 tok/s.
  Stochastic is speed-negative on all three backends on this short seeded lane.
- Dynamic depth now has a production-shaped depth-zero policy: dynamic matrix
  runs start at d1 with `min_depth=0`, d0 normal decode maintains shifted MTP KV
  so later probes can resume safely, demotion is stepwise, d1 only demotes to d0
  after an all-zero window, and perfect probes can promote early without waiting
  for the long default promotion hysteresis. Focused controller and runner
  regressions cover d0 cooldown/probe, shifted-cache maintenance, stepwise
  demotion, and perfect-probe promotion. The post-tune bounded matrix
  `benchmark_results/mtp_vllm_style/20260609T-post-hysteresis-tune-matrix/`
  completed all CUDA/ROCm/CPU dense/MoE greedy/stochastic lanes with fixed
  d1/d2/d3/dynamic. Dense greedy is still speed-positive; dynamic is less
  cliffy but remains short-run conservative versus the best fixed depth.
- `V2_Perf_MTPDepthController` now characterizes dynamic policy overhead. The
  controller is scalar counter bookkeeping, measuring about 11-25 ns per update
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
- GPU kernels produce output tokens plus `num_accepted_tokens` without CPU
  participation. CPU uses scalar/AVX2/AVX512 dispatch plus OpenMP where useful.
- Greedy uses the same buffers and output contract, with argmax equality as the
  deterministic accept test.
- Remove decode-equivalent stochastic verifier fallbacks after parity and
  benchmark acceptance.

Exit gate:

- CPU/CUDA/ROCm sampler parity passes on synthetic and saved Qwen3.6 real-logit
  fixtures for greedy, top-k/top-p, temperature, residual sampling, and seeded
  RNG.
- Dense stochastic MTP no longer emits `decode_equivalent_stochastic_forward_one`
  in accepted lanes.
- Bounded stochastic dense benchmarks are speed-positive or have a documented
  acceptance-limit reason.

### Phase 5: Publish From Spec Slots, Not Checkpoints

Goal: make accepted-state publication cheap, atomic, and backend-neutral.

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
- Keep CUDA, ROCm, and CPU correctness surfaces symmetric.

Exit gate:

- Dense greedy and stochastic are correct on CPU/CUDA/ROCm.
- CUDA and ROCm post comparable speedup classes versus their no-MTP baselines;
  if one backend lags, it gets a tuning pass before acceptance.
- Dynamic approaches the best fixed depth for the prompt class after warmup.

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

Exit gate:

- MoE CPU/CUDA/ROCm greedy and stochastic parity passes with the same tests as
  dense plus MoE layer-by-layer math analysis.
- MoE bounded matrix is speed-positive for greedy and stochastic or has a
  measured route/acceptance bottleneck.
- No MoE path uses dense-only fallbacks.

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

Exit gate:

- LocalTP, NodeLocalTP, LocalPP, and ExpertParallel parity suites pass for dense
  and MoE where hardware exists.
- Multi-device MTP never lets one participant publish a longer prefix than the
  common accepted count.

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
- Dense CPU/CUDA/ROCm stochastic MTP. CPU may use host kernels, but it must use
  the same batched verifier/rejection contract; CUDA/ROCm must use
  device-resident stochastic verification.
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
no-MTP baseline, fixed d1, fixed d2, fixed d3, and dynamic depth. This matrix is
the normal tuning instrument, not an occasional acceptance run. Greedy rows use
production runtime settings with `--temperature 0`, not `--deterministic`;
stochastic rows use a pinned seed, default `123`, so acceptance and throughput
can be compared across iterations. Dynamic depth must always be reported beside
the fixed d1/d2/d3 rows from the same git hash and runtime configuration; do not
tune or accept dynamic in isolation. The matrix dynamic lane starts at d1, allows
per-step d0 adaptive bypass through `--mtp-min-draft-tokens 0`, and probes back
toward d3 after cooldown. The generated `summary.tsv` includes
`speedup_vs_baseline` for every MTP row plus perfstats-derived verifier health:
`verifier_ms`, `condition_ms`, `correction_ms`,
`main_verifier_warmup/capture/replay`, and replay reset/preserve counts. Use
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
