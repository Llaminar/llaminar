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
  First slice complete: `MTPRejectionSampler` defines the distribution-row
  contract and all-position stochastic catch-up construction for the current
  SingleDevice path.
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
  -R "^V2_Unit_(PrefixMTPConfig|MTPDepthController|MTPDecodeCatchup|MTPRejectionSampler|MTPSpecDecodeTransaction|MTPSpecDecodeMetadata|MTPSpecStateContract|MTPSpecKVPublisher|MTPStateTransaction|MTPVerifierPolicy|MTPWeightManifest|MTPGraphConstruction|PrefillDecodeTransition|PrefillGraphCacheIntegration|ForwardExecutionEngineAdvanced)" \
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
tune or accept dynamic in isolation. The standard matrix dynamic lane starts at
d1 and keeps d1 as the adaptive floor; d0 bypass must be run as an explicit
diagnostic until it is proven faster than d1 on a matching benchmark. The
generated `summary.tsv` includes
`speedup_vs_baseline` for every MTP row plus perfstats-derived verifier health:
`verifier_ms`, `condition_ms/count/skipped_ready`, `rejection_no_ready`,
`correction_ms`,
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
