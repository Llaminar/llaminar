# Unified KV Prefix Cache And MTP Phased Implementation Plan

This plan implements one shared prefix-state subsystem for two features:

1. Cross-request KV prefix caching.
2. In-request MTP speculative decoding and rollback.

The central design decision is that both features use the same state contract. Full-attention KV, hybrid GDN recurrence/conv state, MTP's shifted KV cache, terminal hidden/logit rows, and MoE placement fingerprints all flow through typed snapshot, import, restore, and invalidation APIs. The live request state is still cleared at prompt boundaries; persistent prefix blocks live outside request-local `InferenceState` and can repopulate it when a request hits the cache.

The phases below are ordered to keep correctness ahead of performance. Each phase includes goal, implementation details, files, tests, and exit criteria inline.

## Current Phase Audit Status

This section is updated as phases are proven against the current worktree. A phase is marked complete only when its exit criteria have direct evidence from code plus focused tests. Synthetic/unit evidence is acceptable for CPU-side contracts; GPU, parity, and benchmark claims require focused integration, real-model parity, or benchmark evidence.

| Phase | Status | Current Evidence | Remaining Gate |
|-------|--------|------------------|----------------|
| Phase 1: Config, Feature Gates, And Fingerprints | Complete | `Test__PrefixMTPConfig` verifies disabled defaults, CLI/YAML parsing, invalid enum rejection, config propagation into `RuntimeConfig`/`RankExecutionPlan`/`GraphConfig`, and explanation output for dry-run/placement-style reporting. `Test__PrefixCacheFingerprint` verifies deterministic named-part fingerprints, field sensitivity, MoE bypass policy, rebalance epoch key changes, and histogram exclusion. Direct focused binaries passed on 2026-06-01: `v2_test_prefix_mtp_config`, `v2_test_prefix_cache_fingerprint`. | None. |
| Phase 2: IKVCache Logical Snapshot Contract | Complete | CPU logical block export/import/truncate is implemented and covered by `Test__IKVCacheLogicalBlockIO`. CUDA and ROCm logical export/import are covered by focused integration tests. Focused CTest passed on 2026-06-01: `V2_Unit_IKVCacheLogicalBlockIO`, `V2_Integration_CUDARingKVCache_LogicalBlockIO`, `V2_Integration_ROCmRingKVCache_LogicalBlockIO`. | None. |
| Phase 3: Prefix Store, Payload Layout, And Arena Staging | Complete | Prefix hash/key, RAM backend, LRU/cache stats, disk checksum round-trip, disk hydration/failure handling, and arena staging ids are implemented. Direct focused binaries passed on 2026-06-01: `v2_test_prefix_block_hash`, `v2_test_prefix_cache_key`, `v2_test_ram_prefix_storage_backend`, `v2_test_prefix_state_cache_lru`, `v2_test_disk_prefix_storage_backend`, `v2_test_prefix_arena_staging`. | None for the listed Phase 3 unit gates. Device-hot remains a later GPU integration/benchmark tier gate. |
| Phase 4: Dense Prefix Cache End To End | Complete | Dense prefix request control flow, terminal-logit full hits, final-block recompute, MTP-hidden boundary recompute, suffix chunk scheduling, and logical summary counters are covered by `V2_Unit_PrefixCachePrefillFlow`. Dense real-runner prefix reuse, GPU device-hot promotion, disk hydration, GPU cache flag preservation, and MTP probe scaffolding are covered by `V2_Integration_PrefixCacheStateProbe`. Real-model CPU prefix, partial-prefix, and split-prefill smoke parity are covered by `V2_Integration_Parity_PrefixCacheMTP_Qwen35CPUPrefixSmoke`, `V2_Integration_Parity_PrefixCacheMTP_Qwen35CPUPartialPrefixSmoke`, and `V2_Integration_Parity_PrefixCacheMTP_Qwen35CPUSplitPrefillSmoke`. Focused CTest passed on 2026-06-01. | None for dense RAM/device-hot/disk prefix-cache gates. Hybrid, MTP, TP, and MoE behavior remain later-phase gates. |
| Phase 5: MTP Loading, Sidecar Graph, And Shifted Cache | Complete | `V2_Integration_PrefixCacheStateProbe` covers real MTP model inventory, shifted MTP cache count after prefill, and one-token sidecar execution on GPU. Focused direct run passed on 2026-06-01: `MTP_ModelInventoryWhenAvailable`, `MTP_ShiftedCacheCountProbeOnGPU`, and `MTP_SidecarOneTokenExecutesOnGPU`. | None for loading, sidecar construction, and shifted-cache invariant. Decode verification, rollback, and accepted-token parity remain Phase 6/13 gates. |
| Phase 6: MTP Decode, Verification, And Rollback | Complete | `V2_Unit_MTPGraphConstruction` covers all-position LM-head contracts, sidecar execution, shifted MTP KV payload, live dense CPU snapshot restore, and verifier logits exposure. `V2_Unit_PrefillDecodeTransition` covers greedy MTP accept/reject flow, forced rollback/replay, token-budget accounting, and rollback stats. Real-model evidence passed on 2026-06-01: `V2_Integration_PrefixCacheMTP_Qwen36ROCmSmoke`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmPrefixSmoke`, `V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_MTPGreedyMatchesPyTorchDecodeTokens`, and `V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_PrefixCacheMTPRestore`. Hybrid reset regression passed through `V2_Integration_ROCmHybridKVCacheReset`. | None for single-device dense greedy MTP and rollback. Broader hybrid parity, TP, PP, MoE, and benchmark gates remain later phases. |
| Phase 7: Hybrid GDN Prefix State | Complete | `V2_Unit_HybridKVCache` covers hybrid metadata byte counts, CPU hybrid prefix-state export/import, payload layout, global FA layer ids, and kernel-object preservation. `V2_Integration_CUDAHybridKVCacheReset` and `V2_Integration_ROCmHybridKVCacheReset` cover GPU GDN/short-conv clear/import behavior and kernel-state preservation. Qwen3.5 CPU prefix, partial-prefix, and split-prefill parity passed through the Phase 4 real-model smokes. `V2_Unit_PrefillDecodeTransition` now includes forced MTP reject restoration of a checkpoint carrying hybrid payload blocks. Focused CTest passed on 2026-06-01. | None for hybrid prefix-state and rollback contract gates. Larger hybrid parity coverage remains part of Phase 13. |
| Phase 8: MoE Safety, Domain-Scoped Rebalance, And Parallelism | Complete | Focused Phase 8 unit/static tests passed on 2026-06-01: `V2_Unit_MoERebalanceController` proves reason-coded rebalance decisions, `single_participant_observe_only` dynamic downgrade, histogram retention, no placement/epoch mutation for single-participant domains, and domain-tagged replica placement; `V2_Unit_DeviceGraphOrchestrator` and `V2_Unit_RankOrchestrator` prove domain-scoped controller lookup/iteration, legacy first-controller accessor behavior, domain mismatch rejection before mask/replica mutation, GlobalTP rebalance participant id selection from rank-in-domain, and non-primary routed-overlay controller lookup; `V2_Unit_InferenceRunnerFactory_MultiDevice` proves factory creation of routed-overlay rebalance controllers from `MoEExpertOverlayRuntimePlan` plus CPU GlobalTP preservation as one `global_tp_domain_<id>` controller with legacy participant masks; `V2_Unit_MoEExpertOverlayRuntimePlan` proves routed overlay domains expose rebalance-domain attachment metadata while continuation-only dense domains do not; `V2_Unit_MoEOverlayCollectiveWorkspace`, `V2_Unit_PrefillGraphCache`, `V2_Unit_PrefillGraphCacheIntegration`, `V2_Unit_ForwardExecutionEngine`, and `V2_Unit_ForwardExecutionEngineAdvanced` cover sparse collective/manual-boundary and graph-capture maintenance gates; `V2_Unit_MoEForbiddenDependencyScan` guards rebalance call sites against reintroducing socket-mask vocabulary or socket-keyed fingerprint strings. Real-model parity passed on 2026-06-01 for Qwen3.6 dense SingleDevice, LocalTP, LocalPP, and NodeLocalTP prefix/MTP; Qwen3.6 MoE CPU/CUDA/ROCm single-device prefill/decode/snapshot parity also passed. | None for Phase 8 exit criteria. |
| Phase 9: Observability, Diagnostics, And Rollout Controls | Complete | Focused Phase 9 observability tests passed on 2026-06-01: `V2_Unit_PrefixStateCacheLRU` proves deterministic prefix stats for lookups, hits, partial hits, terminal-state hits, byte accounting, evictions, disk hydration, promotions, and disk read failures; `V2_Unit_DeviceGraphOrchestrator` proves disabled prefix cache records zero/no-op bypass stats and unsupported budget bypass details are surfaced through `PrefixRuntimeStateSnapshot`; `V2_Unit_PrefixCachePrefillFlow` and `V2_Unit_PrefillDecodeTransition` prove prefix/MTP request summaries, MTP acceptance/rejection counters, rollback counters, verifier counts, and bypass reasons; `V2_Unit_BenchmarkRunnerCPU` proves benchmark JSON/human summaries capture prefix, MTP, and prefill-chunk stats without large model files. INFO-level request summaries are emitted by `OrchestrationRunner` when prefix cache or MTP is enabled. | None for Phase 9 exit criteria. |
| Phase 10: Prefix Cache Coordination Across TP, PP, And MoE Domains | Complete | Focused Phase 10 coordination tests passed on 2026-06-01: `V2_Unit_PrefixCacheCoordinator` proves common-min token/block clamping, terminal-state AND logic, unsupported participant bypass, fingerprint mismatch handling, and aggregate runner lookup conversion; `V2_Unit_PrefixCacheCoordinatorMPI` runs with `MPI_PROCS 2` and proves GlobalTP/NodeLocalTP-style scalar reductions for common prefix, terminal-state availability, placement epoch, fingerprint mismatch, and rank-local miss clamping; `V2_Unit_RankOrchestrator` proves LocalTP child miss/common-min clamping, populate-failure clear/replay behavior, terminal restore on all children, PP stage miss clamping, and PP terminal ownership rules; `V2_Unit_PrefixCacheFingerprint` covers MoE placement/fingerprint bypass behavior. | None for Phase 10 exit criteria. |
| Phase 11: TP-Compatible MTP Sidecar Execution | Complete | Focused Phase 11 unit tests passed on 2026-06-01: `V2_Unit_MTPGraphConstruction` proves dense sidecar TP allreduce insertion, vocab-parallel embedding allreduce, column-parallel verifier/MTP logits contracts, GlobalTP MTP greedy sampling coordination, shifted MTP KV harvest/restore, and live prefix snapshot restore; `V2_Unit_RankOrchestrator` proves LocalTP `forwardMTP()` enters every child concurrently, child failures still rendezvous every participant, MTP logits/all-position verifier logits gather column-parallel shards, and live checkpoint capture/truncate/restore applies to every child; `V2_Unit_PrefillDecodeTransition` proves GlobalTP and LocalTP MTP decode paths, coordinated world-size sampling, forced reject replay, and rollback/rejection counters are recorded once per request step rather than once per participant. Real-model Qwen3.6 parity passed on 2026-06-01: LocalTP ROCm and NodeLocalTP CPU `MTPGreedyMatchesPyTorchDecodeTokens` plus `PrefixCacheMTPRestore`. | None for dense TP-compatible MTP sidecar gates. PP MTP now hard-fails before prefill until a PP-aware shifted-prefill/verifier path exists; ExpertParallel/MoE MTP remains Phase 12. |
| Phase 12: ExpertParallel Sparse MoE MTP Sidecar And Segmented Graph Capture | Complete | Focused Phase 12 slice passed on 2026-06-01: `V2_Unit_MTPGraphConstruction` proves graph-native overlay MoE MTP sidecar execution appends real shifted KV payload and marks sparse manual boundaries correctly after full graph execution; `V2_Unit_MoEOverlayCollectiveWorkspace` proves non-final sparse dispatch/return participants can complete their manual graph boundary after successful publish while final participants still require collective completion; `V2_Unit_MoEExpertWeightService` proves CPU prepared expert slabs are incrementally filled for disjoint masked participants without duplicating slabs; `V2_Unit_ForwardGraphTypes` proves named sparse/collective boundaries split segmented graph-capture plans unless graph-safe collective capture is explicitly enabled; the broader Phase 12 unit slice covering prefill graph cache, ForwardExecutionEngine chunking, MoE runtime table, decode histograms, rebalance controller, prepared-weight store resolution, and local expert prepared weights passed. Focused integration passed: `V2_Integration_MoEOverlaySparseTransport_MPI` and `V2_Integration_SegmentedGraphCaptureExecution`. Real-model graph-native overlay parity passed for Qwen3.5 MoE RocmHot/CpuCold `DecodeParity` and `PrefillParity`. Real Qwen3.6 MoE parity passes for SingleDevice `MTPGreedyMatchesPyTorchDecodeTokens` and `PrefixCacheMTPRestore`; the 2026-06-03 focused rerun also added `V2_Unit_Qwen35MoEGraph` coverage proving transient MoE runtime-table caches stay out of prefix fingerprints. Focused CTest passed on 2026-06-02 for heterogeneous ExpertOverlay ROCm2TP-hot/CPU2LocalTP-cold: `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_Qwen36MoEExpertOverlayPrefixMTPParity_MTPGreedyMatchesBaselineTokens_ROCm2TPHot_CPU2LocalTPCold` and `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_Qwen36MoEExpertOverlayPrefixMTPParity_PrefixCacheMTPRestore_ROCm2TPHot_CPU2LocalTPCold`. Additional focused slice passed on 2026-06-01: `V2_Unit_ForwardExecutionEngine`, `V2_Unit_DeviceGraphOrchestrator`, `V2_Integration_PrefillGraphCacheExecution_CUDA`, and `V2_Integration_PrefillGraphCacheExecution_ROCm` prove chunk index propagation, MoE domain/participant prefill graph identity hooks, snapshot observability for chunk id, bucket length, real-token range, domain id, participant id, placement epoch, topology signature, capture/replay phase, and structured `forward_graph.prefill_graph_lifecycle` counters while preserving padded-bucket reuse across changing real-token lengths. `V2_Unit_ForwardExecutionEngine` also proves placement-changing chunk-boundary maintenance clears the outer bucketed forward graph cache before the next chunk rebuilds under a new epoch. The CUDA/ROCm prefill graph-cache integrations now prove fixed-placement chunk schedules reach captured replay and forced chunk-boundary rebalance schedules clear, rebuild, recapture, and replay under a new placement epoch. A follow-up replay-preservation slice proves live-state restore/truncate preserves captured replay state for stable MoE placement, while MoE placement epoch participates in forward and MTP sidecar graph identity so actual placement changes miss and rebuild without a blanket live-state reset. Benchmark speedup evidence is intentionally deferred to Phase 14. | None for Phase 12 correctness and segmented-capture compatibility gates. |
| Phase 13: PyTorch Parity Acceptance Matrix | In progress | Dense and MoE parity targets are registered as normal parity-suite tests, and several Phase 13 cells have already passed as evidence for earlier phases, including single-device dense/MoE, LocalTP dense, NodeLocalTP dense, and heterogeneous ExpertOverlay MoE prefix+MTP parity. Dense Qwen3.6 SingleDevice CPU and CUDA prefix/MTP parity now have explicit CTest entries under `V2_Integration_Parity_Qwen36_CPU_SingleDevice_` and `V2_Integration_Parity_Qwen36_CUDA_SingleDevice_` rather than being inferred from the ROCm SingleDevice harness. Focused CTest passed on 2026-06-02 for all five CPU dense SingleDevice cases: `PrefixRestoreFullHit`, `PrefixRestorePartialHit`, `SplitPrefillMatchesPyTorchDecodeTokens`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Focused CTest also passed on 2026-06-02 for all five CUDA dense SingleDevice cases with the same coverage: `PrefixRestoreFullHit`, `PrefixRestorePartialHit`, `SplitPrefillMatchesPyTorchDecodeTokens`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Focused CTest also passed on 2026-06-02 for all three dense NodeLocalTP CPU cases: `PrefixRestoreFullHit`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Focused CTest passed on 2026-06-02 for `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_`: the heterogeneous ROCm2TP-hot/CPU2LocalTP-cold MTP and prefix+MTP cases passed, and the 2x ROCm hot-only ExpertOverlay tests passed via an explicit resident-VRAM prerequisite skip on the current 32 GB ROCm machine. The hot-only harness currently requires at least 40 GiB total VRAM per ROCm participant because the no-fallback plan requires all 256 experts to fit on each ROCm participant. | Audit the full Phase 13 matrix requirement-by-requirement, rerun focused missing cells, and mark complete only after every required non-skipped parity cell has direct current evidence. |
| Phase 13.5: Small-M GEMV-Many Kernel Prerequisite | In progress | ROCm now has graph-capturable native-VNNI small-M verifier routes for M=2/3/4 across Q/K/IQ codebooks, fused QKV/GateUp shared-quant dispatch, same-codebook and mixed-codebook GDN projection subgroups, heterogeneous-N Qwen3.6 qkv/z batching, fused-SwiGLU/FFN-down routing, declared-workspace FP32 GDN projection batching, a graph-capturable tiny FP32 GDN alpha/beta projection route for M=1/2/3/4, and focused ROCm small-M integration/perf coverage. The latest ROCm small-M hardening slice precomputes quantized activation block sums in declared GEMM workspace for plain/fused/batched verifier routes, fixes stale asymmetric min-correction for native-VNNI formats such as Q4_1/Q5_1, and adds `DispatchPlainAsymmetricNativeSmallMUsesFreshBlockSums` plus the `GemmWorkspaceConsumer` sums-buffer contract. The latest hardening slices added graph-capturable atomic K-partitioning for `kb>1`, preserved captured replay state across live prefix restore/truncate by keying decode/sidecar graph identity on the MoE placement epoch instead of blanket-resetting MoE replay state, preserved sidecar and verifier graph topology across request-local cache clears when workspace generation and MoE placement epoch are stable, implemented verifier-row GDN recurrence/short-conv state capture so reject rollback can restore the accepted verifier row instead of replaying accepted tokens, prevented non-verifier GDN graph paths from clearing shared verifier capture bindings, added an explicit warmup-dependent graph-capture contract so supported MoE router/expert/shared stages are planned capturable before warmup and hard-fail if they are still not capturable at capture time, then removed the legacy post-warmup resegment API/counter entirely, removed the GDN mixed-codebook batched-route bypass for native small-M-compatible subgroups, added `kernel.gdn_projection_route` counters that prove real Qwen3.6 uses qkv/z quantized pairs plus alpha/beta FP32 batched projections instead of a synthetic four-projection full mixed-codebook bundle, added `NativeVNNIGEMMPerfTest.MTP_SmallM_BatchedGDNProjectionShapes` for Qwen3.6-style heterogeneous GDN projection groups at M=2/3/4, and added direct ROCm recurrence/short-conv verifier-row restore regressions. Current same-binary long-lane ROCm evidence ratcheted again: the latest `rocm:0` depth-3 MTP run reached 54.78 decode tok/s versus a 30.93 same-binary graph-enabled baseline at `The quick brown fox`, `-c 64`, `-n 48`, with 81.02% acceptance, 111 accepted tokens, 26 rejected tokens, 27 request rollbacks, 55 verifier runs, and 220 verifier tokens. The same artifact recorded zero `forward_graph.post_warmup_resegment` records after the legacy path removal, 12 verifier-row shortcut restores over 48 GDN layers, captured `main_verifier` replay around 44.39 ms, and `mtp.verifier_forward` around 46.76 ms. The prior depth-3 ratchet reached 54.69 tok/s after batched verifier-backed shifted-cache catchup; depth-2 also reached 48.66 tok/s and depth-1 reached 47.59 tok/s on the same lane. Focused validation passed on 2026-06-03 for the latest ROCm block-sum/capture-contract slice: `V2_Unit_GemmWorkspaceConsumer`, `V2_Integration_ROCmQuantisedGemmSmallM` all-codebook and asymmetric-sums regressions, `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison`, `V2_Unit_ForwardGraphTypes`, `V2_Unit_MoERoutingStage`, `V2_Unit_PrefillGraphCapturability`, `V2_Unit_MTPGraphConstruction`, `V2_Unit_PrefillDecodeTransition`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the focused Qwen3.6 ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Earlier focused validation for the small-M/GDN route also passed: `V2_Integration_HipBLASGemm` and `V2_Integration_ROCmGDNPaddedRealLength`. CPU native VNNI now has explicit all-codebook M=2/3/4 fused projection integration coverage, includes the previously missing `Q4_K` and `Q5_K` CPU smoke/sweep entries, records `kernel.cpu_native_vnni_small_m_fused_projection_calls`, and has a focused `V2_Perf_CPUNativeVNNI_GEMV` row for the MTP small-M fused route. A CPU SingleDevice real Qwen3.6 depth-3 smoke now passes after CPU graph workspace binding for declared GDN verifier-state capture buffers, reaching 9.50 decode tok/s versus a prior 5.80 same-prompt baseline with 100% acceptance and active M=4 CPU fused-projection counters. The latest replay hardening also removes raw `TensorBase *` dirty caches from segmented replay metadata, marks replay outputs through cached stable arena write ids, and keeps the hot path flags-only, fixing a ROCm chained-draft shutdown crash after prefix/MTP state mutation while preserving and slightly improving the ROCm depth-3 speed ratchet. CUDA now records `kernel.cuda_native_vnni_small_m_fused_projection_calls`, and `V2_Integration_CUDAGemmParity` covers shared-quant CUDA native-format M=2/3/4 two-projection fused verifier parity across Q/K/IQ codebooks, including Q4_K/Q5_K aliases used by Qwen3.6. The retained CUDA production route uses the native M=2 verifier path where supported and keeps M=3/4 on row-wise GEMV; an attempted M=3/4 chunked route passed focused parity but regressed deeper-draft throughput. | Keep ROCm SingleDevice dense in Phase 13.5 until profiling shrinks the remaining verifier budget enough to approach the 2x target, especially ordinary GEMM, fused Gate/Up, GDN projection, LM head, verifier graph overhead, and deeper-draft rollback/acceptance cost. Broaden CPU real-model MTP evidence beyond the short 100%-acceptance smoke before treating CPU as Phase 14 benchmark-acceptance ready. |
| Phase 13.7: Stochastic MTP Verification And GPU Sampling | In progress | Added as the production gate for Qwen chat defaults (`temperature=0.6`, `top_p=0.95`, `top_k=20`, presence penalty). The first implementation slice is in place for SingleDevice full-logit execution: `Sampler` can build normalized penalty/temperature/top-k/top-p distributions, residual-sample from `max(p-q,0)`, and expose deterministic random thresholds; `OrchestrationRunner` now runs `MTPVerifyMode::SpeculativeSampling` without the old non-greedy bypass, captures draft/target distributions, applies `min(1,p/q)` acceptance, samples residual corrections on reject, and samples the terminal ready token after all-accepted verifier rows. CUDA and ROCm backends expose explicit-stream, graph-capturable top-k/top-p/temperature device sampling enqueue APIs, plus compact top-k/top-p distribution builders and speculative accept/residual verifier kernels for device-resident tables. A shared `src/v2/kernels/common/SamplingMath.h` now owns the SplitMix thresholds, compact top-k/top-p normalization, compact distribution sampling, and speculative accept/residual math used by CPU `Sampler`, CUDA, and ROCm compact stochastic paths. GPU SingleDevice stochastic MTP now hard-fails unsupported top-k modes or missing compact verifier hooks, and supported runners keep the first token, MTP draft tokens, verifier accept/residual tests, and terminal ready-token sampling on compact device-resident distribution tables instead of copying full logits to host. Focused validation passed on 2026-06-03: `V2_Unit_Sampler`, `V2_Unit_PrefillDecodeTransition`, `V2_Unit_PrefixMTPConfig`, `V2_Unit_ChatCompletionHandler`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, and the direct CUDA/ROCm `GPUSamplingTest.LogitPenaltyDeviceInputsAreGraphCapturable`, `GPUSamplingTest.TopKTopPSampleDeviceOutputIsGraphCapturable`, `GPUSamplingTest.TopKTopPDistributionMatchesCPUSampler`, and `GPUSamplingTest.SpeculativeVerifyDistributionsAreGraphCapturable` integration cases. Real Qwen3.6 27B ROCm and CUDA GPU-graphs stochastic smokes passed through `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke` and `V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsStochasticSmoke` with active presence penalty, proving fixed non-greedy sampling enters MTP draft/verify instead of recording the old `sampling is not greedy` bypass; the smokes assert structured counters for device first-token sampling, device MTP draft sampling, device verifier distribution build, and accept tests, while asserting host full-logit stochastic counters stay at zero. Normal Qwen3.6 dense parity-suite coverage now includes `V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_MTPStochasticSamplingVerifierRuns` and `V2_Integration_Parity_Qwen36_CUDA_SingleDevice_Qwen36CUDASingleDevicePrefixMTPParity_MTPStochasticSamplingVerifierRuns`, which run a no-MTP stochastic baseline and graph-captured speculative MTP with structured verifier-counter consistency checks and still pass after the shared sampler-math refactor. | Gather benchmark evidence before Phase 14 uses non-greedy MTP results; if exact same-seed token-stream parity is desired later, add an explicit position-stable RNG contract rather than treating speculative distribution equivalence as seed-coupled token equality. |
| Phase 14: Benchmark Acceptance And Default-Enablement Readiness | Waiting on Phase 13.5 and Phase 13.7 | `docs/v2/PREFIX_CACHE_MTP_BENCHMARK_NOTES.md` tracks real Qwen3.6 dense/MoE baseline, graph-capture status, and best observed MTP speedups by domain. A CUDA dense memory-planner sidequest now proves long-context KV capacity is charged separately from bucket-sized GPU activation arenas: Qwen3.6 27B at context 16384 initializes with `Act.Seq=4096`, 4.0 GB KV, and 1785 MB arena, while hybrid KV still reports 16 FA KV layers plus 48 GDN layers. CUDA SingleDevice dense currently has direct Qwen3.6 production-mode evidence on `The quick brown fox`, `-c 64 -n 48`: depth-1 MTP reaches 53.30 tok/s versus 40.75 tok/s baseline with 95.83% acceptance; depth-2/3 remain slower at 33.20/31.29 tok/s because verifier cost dominates. CUDA SingleDevice MoE now beats the llama.cpp MoE default-lane no-MTP and MTP decode anchors after split-K fused verifier gate/up/down, shared-expert verifier prefill, GDN qkv+z/alpha+beta fused verifier projection work, capture-safe forced verifier routing, deterministic scatter parallelization, cuBLAS FP32 router prefill, fused-runtime down warp-reduce, and IQ4_NL word decode in the shared CUDA native-VNNI helper: clean no-MTP reaches 2707.70 prefill / 119.91 decode tok/s, fixed depth-1 MTP reaches 1946.82 prefill / 148.50 decode tok/s with 71.88% acceptance, and dynamic MTP reaches 1943.69 prefill / 145.36 decode tok/s with 68.75% acceptance. ROCm SingleDevice dense now has stronger long-lane Qwen3.6 dense evidence after the tiny FP32 GDN alpha/beta route, GDN verifier-row shortcut regressions, safe arena-write replay caching, warmup-dependent capture planning, legacy resegment-path removal, batched verifier-backed shifted-cache catchup, and verifier-row batched argmax: 55.04 tok/s depth-3 MTP versus 30.64 same-binary baseline at `-c 64 -n 48`, with 85.61% acceptance. ROCm SingleDevice MoE has fresh default-lane speed-positive proof after grouped-decode runtime-table rewarm, streamful prefix terminal-state restore, capture-safe ROCm attention params, verifier-sized route grouping, and grouped/LDS small-M router logits: fixed depth 1 reached 39.94 tok/s versus 19.72 tok/s baseline at `-c 768`, `-n 64`, with 78.12% acceptance; dynamic max-depth 3 previously demoted to depth 1 and reached 37.82 tok/s. `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalPPHardFail` pins the current LocalPP matrix blocker as an early real-model hard fail with zero MTP draft/verifier counters, preventing recurrence of the old late stage-1 shifted-cache crash while the real PP-aware MTP path is still unimplemented. | Do not use Phase 14 to paper over kernel deficits. Resume benchmark acceptance after Phase 13.5 provides graph-capturable M=2/3/4 GEMV-many coverage and Phase 13.7 proves stochastic MTP correctness for non-greedy chat defaults; then continue matrix-driven benchmark work until remaining supported domains show concrete speedups or documented blockers with traces. |

Latest CUDA validation note, 2026-06-05: deep Qwen3.6 MoE CUDA math parity passed for `V2_Integration_Parity_Qwen36MoE_SingleDevice_Math_{DecodeParity,PrefillParity}_CUDA`; focused CUDA MoE prefix/MTP parity passed for `MTPGreedyMatchesPyTorchDecodeTokens`, `PrefixCacheMTPRestore`, `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`, `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, `MTPBenchmarkStyleDynamicDepthRequestStateResets`, `NoMTPBenchmarkStyleSkipGatherGreedyMatchesGatheredArgmax`, and `VerifierRowShortcutTwoRowStateMatchesFullReplay`. CUDA dense prefix/MTP parity previously passed for all six `V2_Integration_Parity_Qwen36_CUDA_SingleDevice_` cases, including stochastic verifier coverage. `V2_Integration_CUDAGemmParity` now includes `Q4_0_PrefillGraphReplaySurvivesKernelDynamicReset`, which captures the two-pass Stream-K prefill graph on an explicit nonblocking CUDA stream, resets request-scoped kernel state, then replays the graph to prove context-owned GEMM scratch is not freed while graph cache entries survive `clear_cache()`. It also includes `GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM`, which runs a CUDA GDN projection stage on an explicit nonblocking stream, captures/replays it, asserts the qkv+z native subgroup route, asserts the alpha+beta cuBLAS batched FP32 route, and compares alpha/beta outputs against CPU reference. `V2_Integration_CUDAGDNPaddedRealLength` directly covers CUDA verifier-row GDN equivalence: two-row recurrence and short-conv row-0 restore must match an independent one-row replay. `V2_Integration_CUDAMoEKernel` now covers CUDA verifier M=2/3/4 fused grouped prefill with split-K gate/up, repeated experts across rows, split/rowwise equivalence, graph replay, one-token `routeWithTensors()` graph capture without host top-k materialization, default FP32 prefill routing through cuBLAS SGEMM for `seq_len >= 16`, fused runtime MoE decode using deterministic down warp-reduce under graph replay, and capture-safe fused sub-kernel timer exports. Clean release Qwen3.6 MoE CUDA default-lane captures in `benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode` reach 2707.70 prefill / 119.91 decode tok/s without MTP, 1946.82 prefill / 148.50 decode tok/s with fixed depth-1 MTP at 71.88% acceptance, and 1943.69 prefill / 145.36 decode tok/s with dynamic MTP at 68.75% acceptance.

Latest adaptive MTP depth implementation note, 2026-06-03: `MTPDepthController` is implemented for fixed, observe, and SingleDevice dynamic modes, with CLI/config plumbing, request summaries, benchmark JSON fields, and the reusable `scripts/run_mtp_depth_hysteresis_sweep.sh` harness. The harness includes prose lanes plus Python dataclass/CLI and C++ controller/test-generation prompts. The latest tuning slice fixes controller lifetime: repeated benchmark `prefill()` cycles and request-local `clearCache()` no longer reset dynamic mode while MTP remains enabled, so the controller keeps the running tally required by the policy; disabling MTP still resets the controller. `V2_Unit_PrefillDecodeTransition.DynamicMTPDepthPersistsAcrossClearCachePrefillCycles` covers the regression. Focused validation passed: `V2_Unit_MTPDepthController`, `V2_Unit_PrefillDecodeTransition`, `V2_Unit_ForwardGraphTypes`, `V2_Unit_AttentionComputeStage_DynamicKVLen`, and the Qwen3.6 ROCm parity cells `MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens` and `PrefixCacheMTPDynamicDepthRestore`. Current ROCm Qwen3.6 dense sweeps show dynamic mode preserves depth 3 on high-acceptance `qbf_short` at 48.36 tok/s versus 48.29 tok/s fixed depth 3, demotes the default prompt to depth 1 at 45.09 tok/s versus 45.78 tok/s fixed depth 1 and 31.50 tok/s fixed depth 3, and tracks five code-generation lanes at a 44.89 tok/s mean versus 45.61 tok/s fixed depth 1 and 35.52 tok/s fixed depth 3. No additional threshold change is required for this slice; next adaptive work is promotion tuning only if a lane proves fixed depth 3 is fastest while dynamic fails to climb, then domain-common decisions for LocalTP/NodeLocalTP/ExpertParallel.

Latest explicit GPU stream note, 2026-06-03: GPU execution must never rely on the device-default null stream. `DeviceGraphExecutor` now hard-fails a GPU stage that cannot bind to a non-null worker stream, MTP deferred sampling hard-fails if final-sync deferral lacks a capture stream, and GPU greedy sampling rejects null-stream paths. Focused validation passed: `V2_Unit_GraphExecutorCollective` and `V2_Unit_Static_NoDefaultStreamInGPUCode`. The ROCm dynamic qbf-long repro that previously required stage-timing synchronization now completes without that crutch; the clean sweep artifact is `/tmp/llaminar-mtp-bench/adaptive-depth-20260603/qbf-long-explicit-stream-comparison/summary.tsv`.

Latest ROCm decode-attention graph replay note, 2026-06-03: a Qwen3.6 dense ROCm dynamic-depth long run exposed an HSA memory fault during cached HIP graph replay. Controls isolated the fault to cached replay, not ordinary explicit-stream execution: stream-only execution passed, per-step recapture passed, and the one-stage trace failed at `[GRAPH] first=layer0_attention`. The first fix restored ROCm decode-attention graph capture with bucketed graph variants: replay updates logical KV length through device params inside a stable bucket, and bucket growth recaptures instead of replaying an incompatible launch shape. A follow-up root-cause fix moved the ROCm attention `AttentionDeviceParams` H2D copy out of HIP graph capture entirely: `prepareDynamicAttnParams()` uploads on an explicit non-null stream before capture/replay, the captured stage body hard-fails if the device params are not ready, cached forward graphs bind a worker/capture stream before dynamic-param updates, and MTP sidecar graphs bind their explicit stream before sidecar dynamic-param stages update. Focused validation passed for `V2_Unit_AttentionComputeStage_DynamicKVLen`, `V2_Unit_ForwardGraphTypes`, `V2_Unit_GraphExecutorCollective`, and `V2_Unit_ForwardExecutionEngineAdvanced`. A short real Qwen3.6 ROCm dynamic-MTP run completed with 268 MTP graph replay traces, no ERROR/WARN param-copy diagnostics, 36.64 decode tok/s, and 65.85% acceptance. Phase 13.5 remains open for verifier-budget work, not because ROCm decode attention is manually segmented.

Latest ROCm verifier graph lifetime note, 2026-06-03: all-position verifier
logits now have stable per-row-count tensor owners so cached full-depth verifier
graphs can survive a smaller tail verifier and request-local `clearCache()`
without replaying a dangling LM-head output pointer. The focused Qwen3.6 ROCm
graph smoke now reproduces that benchmark crash pattern directly. The same
slice clamps MTP speculative draft depth to the remaining decode token budget,
and the Qwen3.6 dense parity harness now expects effective draft depth for
short `max_new_tokens` cases. Focused validation passed:
`V2_Unit_PrefillDecodeTransition`, `V2_Unit_MTPGraphConstruction`,
`V2_Unit_GDNKernels`, `V2_Integration_ROCmGDNPaddedRealLength`,
`V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and
`V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_MTPGreedyDepth3MatchesPyTorchDecodeTokens`.
The current ROCm dense long-lane evidence is 55.09 tok/s depth-3 MTP versus the
30.64 tok/s same-binary baseline, preserving the 1.80x ratchet while Phase 13.5
continues to target verifier GPU work before Phase 14 acceptance can claim 2x.

Latest MTP sidecar graph-cache note, 2026-06-03: depth-0 sidecar graph caches now key on MoE placement epoch only when the sidecar graph contains MoE expert weights. Dense sidecars use epoch key `0` and are no longer rebuilt because an attached rebalance controller advances an unrelated MoE epoch; real MoE sidecars still miss and rebuild on actual placement changes. Focused validation passed: `V2_Unit_MTPGraphConstruction` and `V2_Unit_PrefillDecodeTransition`.

Latest MoE warmup-dependent graph-capture note, 2026-06-03: segmented capture now plans supported MoE router, expert, shared-FFN, and shared-gate stages as capturable before warmup instead of building one cold segment plan and reconstructing it after warmup. Warmup still initializes runtime tables, kernels, descriptor banks, and scratch on the capture stream; capture now hard-fails if a stage that promised warmup-dependent graph capture is still not graph-capturable. Focused validation passed: `V2_Unit_ForwardGraphTypes`, `V2_Unit_MoERoutingStage`, `V2_Unit_PrefillGraphCapturability`, `V2_Unit_MTPGraphConstruction`, `V2_Unit_PrefillDecodeTransition`, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`.

Latest current-revision ROCm SingleDevice dense no-post-resegment rerun, 2026-06-03: Qwen3.6 dense 27B Q4_K_S on `rocm:0`, GPU graphs enabled, `The quick brown fox`, `-c 64`, `-n 48`, reached 54.78 tok/s decode and 54.75 tok/s overall at depth-3 MTP versus the 30.93 tok/s same-binary baseline, preserving and slightly improving the ROCm depth-3 ratchet at about 1.77x. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-baseline-current-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-no-postresegment-mtp-d3-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-no-postresegment-mtp-d3-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-no-postresegment-mtp-d3-c64-n48-stats.csv`. The request summary recorded 81.02% acceptance, 111 accepted tokens, 26 rejected tokens, 27 rollbacks, 55 verifier runs, and 220 verifier tokens. Structured stats contain zero `forward_graph.post_warmup_resegment` records after the legacy path removal, preserve 12 verifier-row GDN rollback shortcuts, and retain the batched `kv_cache_only` shifted-cache catchup win from the prior artifact (`mtp_decode_catchup` about 7.07 ms versus 31.66 ms before batching). Remaining Phase 13.5 pressure is captured `main_verifier` GPU work, still around 44.39 ms per measured M=4 verifier graph, plus avoiding three separate host-synchronized sidecar draft steps at depth-3.

Latest ROCm SingleDevice dense sidecar/sample sync-fusion rerun, 2026-06-03: GPU `DeviceGraphOrchestrator` now exposes a runner-native MTP sidecar forward+greedy-sample API. Dense sidecar segmented replay can defer its final capture-stream sync and enqueue argmax on the same stream, using the argmax D2H sync as the operation boundary. Qwen3.6 dense 27B Q4_K_S on `rocm:0`, GPU graphs enabled, `The quick brown fox`, `-c 64`, `-n 48`, reached 54.32 tok/s decode at depth-3 MTP on the rebuilt current code, while the immediately prior same-slice run reached 54.99 tok/s; both preserve the roughly 1.77x ROCm dense long-lane ratchet versus the 30.93 tok/s same-binary baseline. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-sidecar-samplefusion-current-mtp-d3-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-sidecar-samplefusion-current-mtp-d3-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-sidecar-samplefusion-current-mtp-d3-c64-n48-stats.csv`. Structured counters prove the path fired: 39 first-sidecar and 78 chained-sidecar `sidecar_forward_sample_sync_fusions`, 33 first-sidecar and 72 chained-sidecar `segmented_replay_final_sync_deferred` records, 117 `mtp_token_device_samples`, and no separate `sample_mtp_token_device` timer. Focused validation passed: `V2_Unit_PrefillDecodeTransition`, `V2_Unit_MTPGraphConstruction`, `V2_Unit_PrefillGraphCapturability`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the Qwen3.6 ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. This closes the avoidable sidecar/sample intermediate sync; remaining Phase 13.5 pressure is still captured `main_verifier` GPU work before Phase 14 can claim the 2x dense target.

Latest ROCm native-VNNI block-sum and K-partition diagnostic, 2026-06-03: after precomputing activation block sums in declared GEMM workspace, the focused Q4_1/Q5_1 small-M regressions, Qwen3.6 ROCm parity cells, and chained graph smoke passed. A production-path short Qwen3.6 dense `rocm:0`, `The quick brown fox`, `-c 64`, `-n 8`, depth-3 MTP smoke reached 33.09 tok/s with 80.00% acceptance and active sidecar/sample sync fusion. The matching stage-timed MTP diagnostic reached 32.88 tok/s while the no-MTP stage-timed baseline reached 31.98 tok/s. The verifier GPU stage split remains linear-heavy: `GEMM` about 17.25 ms/call, `GEMM_FUSED_GATE_UP` about 12.46 ms/call, `GDN_PROJECTION` about 7.14 ms/call, `GDN_RECURRENCE` about 2.88 ms/call, attention about 2.44 ms/call, and LM head about 1.57 ms/call. Existing graph-safe K partitioning is still needed: forcing `LLAMINAR_ROCM_NVNNI_GEMV_KB=1` collapsed the same MTP lane to 17.53 tok/s, and `KB=4` reached only 30.77 tok/s, so the next ROCm slice should not lower the default split count.

Latest ROCm SingleDevice dense stage-timed verifier split, 2026-06-03: Qwen3.6 dense 27B Q4_K_S on `rocm:0`, GPU graphs enabled, `The quick brown fox`, `-c 64`, `-n 8`, depth-3 MTP completed in 33.71 decode tok/s with 70.00% acceptance under `LLAMINAR_GPU_STAGE_TIMING=1`. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-current-stage-mtp-d3-c64-n8-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-current-stage-mtp-d3-c64-n8-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-current-stage-mtp-d3-c64-n8-stats.csv`. Decode GPU stage time averaged 22.33 ms/iteration while wall time averaged 50.72 ms/iteration. The top decode GPU buckets were ordinary `GEMM` at 7.39 ms/iteration, `GEMM_FUSED_GATE_UP` at 5.72 ms/iteration, `GDN_PROJECTION` at 3.01 ms/iteration, `FUSED_RESIDUAL_NORM` at 1.45 ms/iteration, `GDN_RECURRENCE` at 1.24 ms/iteration, and `GEMM_FUSED_QKV` at 0.83 ms/iteration. The next ROCm Phase 13.5 slice should target verifier main-graph GEMM/GateUp/GDN projection work rather than sidecar synchronization.

Latest current-revision ROCm SingleDevice dense warmup-dependent capture rerun, 2026-06-03: Qwen3.6 dense 27B Q4_K_S on `rocm:0`, GPU graphs enabled, `The quick brown fox`, `-c 64`, `-n 48`, reached 54.48 tok/s decode and 54.67 tok/s overall at depth-3 MTP versus the 30.67 tok/s same-binary baseline, preserving and slightly improving the ROCm depth-3 ratchet at about 1.78x. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-warmupdep-mtp-d3-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-warmupdep-mtp-d3-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-warmupdep-mtp-d3-c64-n48-stats.csv`. The request summary recorded 87.77% acceptance, 122 accepted tokens, 17 rejected tokens, 18 request rollbacks, 53 verifier runs, and 212 verifier tokens. Structured stats recorded `forward_graph.post_warmup_resegment{reason=not_required}=15`, 12 verifier-row shortcut restores across 48 GDN layers, no suffix/replay-forward counters, captured `main_verifier` replay around 44.42 ms, and captured sidecar-chain replay around 2.10 ms.

Latest current-revision ROCm SingleDevice dense recheck, 2026-06-03: after the sidecar epoch-key narrowing, Qwen3.6 dense 27B Q4_K_S on `rocm:0`, GPU graphs enabled, `The quick brown fox`, `-c 64`, `-n 48`, preserved the ROCm depth-3 MTP ratchet at 54.21 tok/s decode versus a 30.71 tok/s same-revision baseline, about 1.77x. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-epochkey-baseline-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-epochkey-mtp-d3-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-epochkey-mtp-d3-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-epochkey-stagegpu-mtp-d3-c64-n8-stats.json`. Focused Qwen3.6 ROCm parity passed for `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Remaining Phase 13.5 pressure is still captured verifier GPU work and the larger graph-native problem of avoiding three separate host-synchronized sidecar draft steps at depth-3.

Latest ROCm SingleDevice dense Phase 13.5/14 note, 2026-06-03: the latest long-lane recheck reached 34.75 tok/s depth-1 MTP versus a same-context 21.20 tok/s graph-enabled baseline on `rocm:1` at `The quick brown fox`, `-c 64`, `-n 48`, for a 1.64x same-binary speedup. A same-day `rocm:0` recheck preserved the retained local ratchet at 34.28 tok/s and 35.01 overall tok/s with artifacts `/tmp/llaminar-mtp-bench/dense-rocm-resegment-slice-mtp-d1-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-resegment-slice-mtp-d1-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-resegment-slice-mtp-d1-c64-n48-stats.csv`. Structured stats showed dense verifier/sidecar warmup no longer pays the MoE-only resegment scan (`forward_graph.post_warmup_resegment{reason=not_required}=12`), `main_verifier` cache hits after warmup, six measured rollbacks and six `rollback_verifier_state_row_shortcuts`, six `verifier_state_row_restores` over 48 GDN layers in the `rocm:1` run, no `replay_forward`, `mtp.verifier_forward` averaging about 51.08 ms on the `rocm:1` run and captured `main_verifier` replay averaging about 49.79 ms on the latest `rocm:0` run. Focused validation passed: `V2_Unit_ForwardGraphTypes`, `V2_Unit_MoERoutingStage`, `V2_Unit_PrefillGraphCapturability`, `V2_Unit_MTPGraphConstruction`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the focused Qwen3.6 ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens` and `PrefixCacheMTPRestore`. The ROCm native-VNNI perf harness now includes Q4_K/Q5_K real Qwen3.6 GDN `qkv+z` groups (`N={10240,6144}`) in `NativeVNNIGEMMPerfTest.MTP_SmallM_BatchedGDNProjectionShapes`, plus Q4_K/Q5_K direct-prefill route comparisons; both focused release perf tests passed. Earlier Phase 13.5 context still applies: depth-2/depth-3 are graph-captured but slower on this prompt, atomic K-partitioning is graph-safe for `kb>1`, mixed-codebook native small-M GDN projection subgroups are covered, and the remaining blocker is true captured verifier GPU work in GDN projection, ordinary GEMM, fused Gate/Up, and other verifier stages.

Latest ROCm SingleDevice dense tiny-FP32/GDN shortcut note, 2026-06-03: the latest `rocm:0` long-lane recheck reached 47.59 tok/s depth-1 MTP and 48.20 overall tok/s versus a 30.67 tok/s same-binary graph-enabled baseline at `The quick brown fox`, `-c 64`, `-n 48`, for a 1.55x speedup and the prior depth-1 ROCm dense long-lane best. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-baseline-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d1-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d1-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d1-c64-n48-stats.csv`. Structured counters proved the intended path: `rocm_fp32_tiny_batched_projection_calls` for GDN alpha/beta at M=1/2/4, `forward_graph.post_warmup_resegment{reason=not_required}=12`, measured rollback verifier-row shortcuts, and `verifier_state_row_restores` over 48 GDN layers. The request recorded 90.62% acceptance, 87 accepted tokens, 9 rejected tokens, and 9 rollbacks. Focused validation passed: `V2_Integration_HipBLASGemm`, `V2_Integration_ROCmGDNPaddedRealLength`, `V2_Unit_PrefillDecodeTransition`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the focused Qwen3.6 ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens` and `PrefixCacheMTPRestore`. Remaining Phase 13.5 work is now verifier graph overhead and dense/GDN verifier GPU work, with stage diagnostics showing GEMM, fused Gate/Up, and GDN projection as the largest GPU buckets.

Latest ROCm SingleDevice dense deeper-draft note, 2026-06-03: after the tiny-FP32/GDN shortcut slice, current `rocm:0` depth-2 and depth-3 reruns on the same Qwen3.6 dense 27B Q4_K_S lane changed the speed ratchet. Depth-2 reached 48.66 tok/s at `The quick brown fox`, `-c 64`, `-n 48`, with artifacts `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d2-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d2-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d2-c64-n48-stats.csv`. Depth-3 reached 54.32 tok/s and 54.50 overall tok/s with artifacts `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d3-c64-n48-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d3-c64-n48-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-tinyfp32-mtp-d3-c64-n48-stats.csv`, a 1.77x speedup over the 30.67 same-binary baseline. Depth-3 stats recorded 81.02% acceptance, 111 accepted tokens, 26 rejected tokens, 27 rollbacks, 55 verifier runs, and 220 verifier tokens; active counters proved M=4 tiny FP32 and native-VNNI batched routes. A new normal parity-suite cell, `V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_MTPGreedyDepth3MatchesPyTorchDecodeTokens`, passed with GPU graphs enabled, and the existing depth-1 MTP greedy plus prefix+MTP restore cells stayed green after the helper change. Phase 13.5 remains open because 1.77x is still short of the 2x Phase 14 target and CUDA/CPU M=2/3/4 coverage is not complete.

Latest ROCm SingleDevice dense both-shortcuts diagnostic note, 2026-06-03: a short stage-timed depth-3 MTP run confirmed the two current fast paths together on the real Qwen3.6 dense 27B Q4_K_S lane. Artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-noresegment-stagegpu-mtp-d3-c64-n8-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-noresegment-stagegpu-mtp-d3-c64-n8-stats.json`, and `/tmp/llaminar-mtp-bench/dense-rocm-gdnshortcut-noresegment-stagegpu-mtp-d3-c64-n8-stats.csv`. This run is diagnostic rather than a speed ratchet because `LLAMINAR_GPU_STAGE_TIMING=1` adds overhead. Structured stats recorded `forward_graph.post_warmup_resegment{reason=not_required}=15`, six `rollback_verifier_state_row_shortcuts`, six `verifier_state_row_restores` over 48 GDN layers, `restore_verifier_state_row` at about 211 us, active M=4 native-VNNI small-M batched routes, and active tiny FP32 GDN alpha/beta projection calls. Remaining stage-timed verifier pressure is still ordinary GEMM, fused Gate/Up, GDN projection/recurrence, LM head, and suffix replay when only a partial verifier row can be committed.

Latest ROCm SingleDevice dense accepted-prefix reject note, 2026-06-03: MTP reject recovery now lags the rejected correction token after any restored accepted verifier prefix, not only after a first-speculative-token reject. The CPU regression `V2_Unit_PrefillDecodeTransition` covers depth-3 MTP with one accepted speculative token, a row-1 reject, row-1 verifier state restore, no suffix replay, and the correction consumed by the next condition forward. Focused Qwen3.6 ROCm PyTorch parity stayed green for `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Real Qwen3.6 ROCm counter artifacts are `/tmp/llaminar-mtp-bench/dense-rocm-lagged-prefixreject-mtp-d3-c64-n8-stats.json` and `/tmp/llaminar-mtp-bench/dense-rocm-lagged-prefixreject-mtp-d3-c64-n48-stats.json`; the long lane reached 53.67 tok/s, below the 54.32 tok/s ratchet, but recorded 12 lagged correction tokens, zero suffix/replay counters, and verifier-row restores for rows 0/1/2 across 48 GDN layers.

Note: CTest wraps these unit and integration binaries in `mpirun -np 1` in the current build. Inside the filesystem sandbox, PMIx socket setup can fail before test assertions. The CTest evidence above was collected by running the same focused CTest commands outside the sandbox; direct focused binaries were used only as the first debugging step where noted.

## Guiding Constraints

- Graphs remain per-device and symmetric. Do not introduce nested multi-device subgraphs for prefix caching or MTP.
- Prefix blocks are runtime state, not model weights and not activation arena state. Persistent prefix storage lives outside `BufferArena`; `BufferArena` owns only request-local staging buffers.
- RAM is the primary correctness tier. Device memory is a hot promotion tier, and disk is durable backing. Device-tier eviction must never invalidate RAM/disk source blocks.
- GDN rollback is snapshot/restore/replay first. GDN recurrence and short-conv state mutate in place and cannot be treated like append-only KV rows.
- MoE decode histograms are telemetry, not prefix payload. MoE placement, masks, replicas, and rebalance epoch are key material.
- MoE rebalance is domain-scoped, not socket-scoped. Placement, masks, replicas, histograms, and epochs are keyed by ExpertParallel domain id and participant id.
- Sparse ExpertParallel graph capture is segmented around explicit dispatch/return collective boundaries. Rebalance can happen only between prefill chunks or decode steps, never during capture/replay or while a sparse collective is incomplete.
- MTP support starts with Qwen3.5/Qwen3.6 D=1 shared-embedding models and stays config-gated until greedy parity is stable.
- All config parsing follows the V2 pipeline: `OrchestrationConfig` -> `RuntimeConfig` / `RankExecutionPlan` -> `GraphConfig` -> graph builders and runners.

## Phase 0: State Probe And Scope Lock

### Goal

Build a test-backed inventory of every state item that changes during prefill/decode and is reset by `clearCache()`. This phase proves what must be persisted, what must be rolled back, and what should remain transient.

### Implementation Details

Add an integration probe target under:

```text
tests/v2/integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp
```

The probe should run representative flows without implementing the full prefix-cache manager yet:

1. `prefill(prompt_prefix)`.
2. Optional `decodeStep()`.
3. Capture state inventory.
4. `clearCache()`.
5. Confirm cleared state.
6. Prototype restore with whatever low-level hooks exist in the current phase.
7. Run suffix and compare logits/tokens against a no-clear baseline.

Classify state with this contract:

| State | Owner Today | Prefix Payload | MTP Rollback Payload | Notes |
|------|-------------|----------------|----------------------|-------|
| Full-attention K/V rows | `IKVCache` implementations | Yes | Yes | Export/import in logical token order only. |
| Ring head/count | `IKVCache` implementations | Metadata | Metadata | Restore must make `get_cached_tokens()` and graph replay head buffers consistent. |
| GDN recurrence state | `IHybridKVCache`, `HybridGDNLayerState`, GPU GDN kernel | Yes | Yes | FP32, in-place, not append-only. |
| GDN short-conv state | `IHybridKVCache`, `HybridGDNLayerState`, GPU conv kernel | Yes | Yes | Same restore contract as recurrence state. |
| MTP depth KV | New `mtp_kv_caches` | Yes when MTP enabled | Yes | Separate FA cache per MTP depth. Current target depth is 1. |
| Terminal hidden row | `InferenceState::hidden` | Yes when MTP enabled | Useful | Required to continue MTP after a full prefix hit. |
| Terminal logits row | `InferenceState::logits` | Optional for non-MTP, required for full MTP hit | Useful | Preserves first-token semantics. |
| `positions` / `sequence_lengths` | `DeviceGraphOrchestrator::InferenceState` | Metadata | Metadata | Populate sets these to the restored token count. |
| `prefill_logits_ready_` | `OrchestrationRunner` | Derived | Derived | Set only when terminal logits are restored or recomputed. |
| MoE placement | `MoERuntimeTable`, rebalance controller, stage masks | Key only | Key only | Placement changes must miss or invalidate. |
| MoE decode histogram | `MoERuntimeTable`, `DecodeExpertHistogram` | No | No | Telemetry only. |
| MoE route/grouped scratch | MoE runtime table and stages | No | No | Transient capacity and scratch. |
| Graph cache entries | `ForwardExecutionEngine`, `LayerGraphCache` | No | No | Rebuilt or reset around request boundaries. |

Dense probe:

- Use a small Qwen2.5 model.
- Record FA KV token counts, ring head/count, position values, sequence lengths, prefill logits readiness, and graph-cache reset behavior before and after `clearCache()`.
- Prototype FA KV restore if the low-level API is available in the branch.

Hybrid probe:

- Use Qwen3.5/Qwen3.6 hybrid models when available.
- Record FA KV plus GDN recurrence/conv hashes.
- Verify `clearCache()` zeroes host-side GDN vectors and GPU kernel state.

MoE probe:

- Record placement bank, active epoch, expert masks, local compute masks, and replica role values.
- Confirm decode histogram changes do not need to be restored.
- Confirm placement epoch changes alter the MoE fingerprint or force cache bypass.

MTP probe:

- Inspect `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf` and `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` when present.
- Enumerate `mtp.*` metadata and tensors.
- Confirm D=1 and shared embeddings for current target files.
- Define the shifted-cache invariant used in Phase 5.

### Files

- `tests/v2/integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp`
- `tests/v2/integration/prefix_cache/CMakeLists.txt` or the relevant test registration file.
- Existing runner/cache files are read-only in this phase unless a tiny probe-only accessor is unavoidable.

### Tests

Add these tests, model-gated where needed:

| Test | Purpose |
|------|---------|
| `DenseQwen25_ResetStateInventory` | Capture dense KV, positions, logits readiness, and graph-cache reset behavior. |
| `DenseQwen25_PrototypeKVRestoreMatchesSuffix` | Export a dense prefix, clear, import, run suffix, compare to baseline. |
| `HybridQwen35_GDNStateInventory` | Verify GDN state changes during execution and zeros after clear. |
| `HybridQwen35_PrototypeHybridRestoreMatchesSuffix` | Restore FA KV plus GDN state and compare suffix logits. |
| `MoE_RuntimeFingerprintChangesOnPlacementEpoch` | Confirm MoE placement changes alter fingerprint material. |
| `MTP_Qwen36_MetadataAndTensorInventory` | Enumerate Qwen3.6 MTP metadata/tensors, skipping when files are absent. |

### Exit Criteria

- A test log or table identifies all request-state payload and transient state.
- Dense and hybrid restore blockers are known before broad API work begins.
- MTP tensor names and metadata expectations are recorded against actual GGUF files.
- MoE histogram state is explicitly excluded from prefix payload.

## Phase 1: Config, Feature Gates, And Fingerprints

### Goal

Add disabled-by-default configuration and stable key material so prefix cache and MTP behavior can be enabled without changing normal inference.

### Implementation Details

Add explicit structs instead of sprinkling feature flags across call sites:

```cpp
enum class PrefixCacheStorageMode
{
    Disabled,
    Ram,
    Device,
    Tiered,
};

enum class PrefixCacheTerminalStateMode
{
    Off,
    Auto,
    Always,
};

enum class PrefixCacheMoEPolicy
{
    Disabled,
    PlacementFingerprint,
    InvalidateOnRebalance,
};

struct PrefixCacheRuntimeConfig
{
    bool enabled = false;
    PrefixCacheStorageMode storage_mode = PrefixCacheStorageMode::Tiered;
    int block_size = 64;
    size_t ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    size_t device_budget_bytes = 256ull * 1024ull * 1024ull;
    size_t disk_budget_bytes = 0;
    std::string disk_dir;
    PrefixCacheTerminalStateMode terminal_state = PrefixCacheTerminalStateMode::Auto;
    PrefixCacheMoEPolicy moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
};

enum class MTPVerifyMode
{
    Greedy,
    SpeculativeSampling,
};

struct MTPRuntimeConfig
{
    bool enabled = false;
    int draft_tokens = 1;
    MTPVerifyMode verify_mode = MTPVerifyMode::Greedy;
    bool require_terminal_hidden_for_full_hit = true;
};
```

Wire these through the V2 configuration pipeline:

1. `OrchestrationConfig`: raw CLI/YAML config.
2. `RuntimeConfig`: parsed values copied through plan building.
3. `RankExecutionPlan`: per-rank runtime contract.
4. `GraphConfig`: model-specific graph config and cache layout decisions.
5. `InferenceRunnerFactory`: runner construction and feature enablement.

Add CLI flags in `OrchestrationConfigParser::buildSpec()` under `Prefix Cache` and `MTP` help categories:

```text
--prefix-cache
--prefix-cache-storage ram|device|tiered
--prefix-cache-block-size <n>
--prefix-cache-vram-budget-mb <mb>
--prefix-cache-ram-budget-mb <mb>
--prefix-cache-disk-budget-mb <mb>
--prefix-cache-disk-dir <path>
--prefix-cache-terminal-state off|auto|always
--prefix-cache-moe-policy disabled|placement-fingerprint|invalidate-on-rebalance
--mtp
--mtp-draft-tokens <n>
--mtp-verify-mode greedy|speculative-sampling
```

Implement fingerprints centrally:

```text
src/v2/execution/prefix_cache/PrefixCacheFingerprint.h
src/v2/execution/prefix_cache/PrefixCacheFingerprint.cpp
```

The fingerprint should be composed from named parts:

```cpp
struct PrefixFingerprintParts
{
    uint64_t model = 0;
    uint64_t tokenizer = 0;
    uint64_t runtime = 0;
    uint64_t topology = 0;
    uint64_t hybrid = 0;
    uint64_t moe = 0;
    uint64_t mtp = 0;
};
```

Fingerprint material:

- `model`: architecture, GGUF metadata fingerprint, tensor directory fingerprint, vocab size, tied/separate LM head mode.
- `tokenizer`: tokenizer model, added tokens, chat-template identity, template override identity.
- `runtime`: activation precision, KV precision, KV layout, RoPE-on-read, TurboQuant mode, Q16 scales, fused attention backend, partial RoPE factor.
- `topology`: TP/PP degree, rank/participant id, device id, local KV-head start/count, layer range, vocab shard.
- `hybrid`: layer types, FA/GDN counts, GDN state size, group count, time-step rank, inner size, conv kernel, local head assignment.
- `moe`: expert count, top-k, expert mode, placement epoch, active bank, local compute masks, replica roles, rebalance domain id.
- `mtp`: enabled flag, depth, draft token count, shared/dedicated embedding mode, MTP tensor-name fingerprint, MTP weight shapes.

If `PrefixCacheMoEPolicy::Disabled` is active for a MoE model, bypass prefix caching rather than hashing a zero MoE part.

Environment overrides must go through `DebugEnv`. Do not add local `getenv()` calls in prefix-cache or MTP runtime code.

### Files

- `src/v2/config/OrchestrationConfig.h`
- `src/v2/config/OrchestrationConfigParser.cpp`
- `src/v2/execution/config/RuntimeConfig.h`
- `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`
- `src/v2/models/GraphTypes.h`
- `src/v2/execution/factory/InferenceRunnerFactory.*`
- `src/v2/execution/prefix_cache/PrefixCacheFingerprint.h/.cpp`
- `src/v2/utils/DebugEnv.h` if env overrides are added.

### Tests

- Parser tests for every new CLI flag, including invalid enum values.
- Runtime config copy tests that prove values survive `OrchestrationConfig` -> `RuntimeConfig` -> `GraphConfig`.
- Fingerprint tests that alter one field at a time and confirm the final key changes.
- MoE policy tests: disabled bypasses, placement-fingerprint keys, rebalance epoch changes key material.

### Exit Criteria

- Normal inference behavior is unchanged when all new flags are off.
- `--dry-run` / placement logging can report resolved prefix/MTP settings after model metadata is known.
- Fingerprints are deterministic across runs for identical config and model inputs.

## Phase 2: IKVCache Logical Snapshot Contract

### Goal

Expose logical KV export/import/truncate through `IKVCache` so prefix storage and MTP rollback never need to know raw backend ring layout.

### Implementation Details

Add narrow default-false APIs to `IKVCache`:

```cpp
struct KVCacheLogicalBlockDescriptor
{
    int layer = 0;               // Global layer index as graph stages see it.
    int seq_idx = 0;
    int logical_token_start = 0; // Oldest-to-newest logical index.
    int token_count = 0;
    void *stream = nullptr;      // cudaStream_t, hipStream_t, or nullptr.
};

struct KVCacheLogicalBlockLayout
{
    ActivationPrecision k_precision = ActivationPrecision::FP32;
    ActivationPrecision v_precision = ActivationPrecision::FP32;
    TensorLayout layout = TensorLayout::KV_POS_HEAD_DIM;
    int local_kv_heads = 0;
    int kv_head_start = 0;
    int head_dim = 0;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    bool device_resident = false;
};

struct KVCacheSequenceState
{
    int cached_tokens = 0;
    int implementation_head = 0;
    bool wrapped = false;
};

virtual KVCacheLogicalBlockLayout logicalBlockLayout(
    int global_layer,
    int token_count) const;

virtual KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const;

virtual bool exportLogicalBlock(
    const KVCacheLogicalBlockDescriptor &desc,
    void *dst_k,
    void *dst_v) const;

virtual bool importLogicalBlock(
    const KVCacheLogicalBlockDescriptor &desc,
    const void *src_k,
    const void *src_v);

virtual bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr);
```

Semantics:

- `logical_token_start` and `token_count` are logical sequence positions, not raw ring rows.
- `PrefixStateCache` never calculates raw ring offsets.
- `global_layer` uses graph-stage numbering. Hybrid caches do internal FA-index remapping.
- Export to RAM may do D2H; export to device-hot tier may do D2D. GPU paths should honor `stream`.
- Import must set head/count metadata before suffix execution.
- Import/truncate/clear must invalidate converted KV and RoPE-on-read shadow state.
- `truncateSequence()` is logical. Dense caches can update counts; hybrid rollback still uses full snapshot restore.

Backend implementation:

- `CPURingKVCache`: copy logical rows oldest-to-newest. CPU documentation treats `head` as oldest valid token.
- CUDA and ROCm bases: use existing hooks `entryHead`, `entryCount`, `setEntryHead`, `setEntryCount`, `resetEntry`, `onClearSequence`, `onEviction`, and `onAdvanceComplete`.
- Add `onRestoreSequence(layer, seq_idx)` if needed for shadow invalidation.
- CUDA/ROCm import should initially restore into an unwrapped layout starting at row 0 and set next-write head to `cached_tokens % max_seq_len`.
- TQ and Q16_1 paths must preserve native cache precision and layout. Do not dequantize for persistent payload unless a backend lacks native I/O in the first CPU-only slice.

The GPU RoPE shadow hazard is a blocker: converted/RoPE shadow caches can track count while missing content/head changes. Import, truncate, clear, and eviction must invalidate shadows by layer/sequence.

### Files

- `src/v2/kernels/IKVCache.h`
- `src/v2/kernels/cpu/CPURingKVCache.h/.cpp`
- `src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h/.cpp`
- `src/v2/kernels/cuda/kvcache/CUDARingKVCache.h/.cu`
- `src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.h/.cu`
- `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h/.cpp`
- `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.h/.cpp`
- `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.h/.hip`

### Tests

Add `tests/v2/unit/kernels/Test__IKVCacheLogicalBlockIO.cpp` with backend coverage where available:

- Empty export/import contract.
- Non-wrapped export.
- Wrapped export.
- Import into empty cache.
- Import over non-empty cache after clear.
- Truncate to zero, current length, shorter length, and invalid longer length.
- Q16_1 head-major layout.
- TQ asymmetric K/V layout.
- Sharded KV heads with nonzero `kv_head_start`.
- GPU stream import/export and synchronization.

### Exit Criteria

- CPU logical export/import/truncate passes for the formats used in unit tests.
- CUDA/ROCm either pass equivalent tests or cleanly return unsupported while feature gates keep prefix caching disabled on those backends.
- Import restores `get_cached_tokens()` and graph-captured append head behavior correctly.

## Phase 3: Prefix Store, Payload Layout, And Arena Staging

### Goal

Create the storage manager and request-local staging needed for dense prefix caching without wiring it into generation yet.

### Implementation Details

Add module layout:

```text
src/v2/execution/prefix_cache/
  BlockHash.h/.cpp
  PrefixCacheConfig.h
  PrefixCacheKey.h/.cpp
  PrefixPayloadLayout.h/.cpp
  PrefixStateBlock.h
  PrefixStateCache.h/.cpp
  PrefixStorageBackend.h
  RamPrefixStorageBackend.h/.cpp
  DeviceHotPrefixStorageBackend.h/.cpp
  DiskPrefixStorageBackend.h/.cpp
  PrefixCacheStats.h
  PrefixCacheFingerprint.h/.cpp
  PrefixStateSnapshot.h
```

Derive `PrefixPayloadLayout` once after `GraphConfig` and KV cache creation:

```cpp
struct PrefixPayloadLayout
{
    DeviceId device;
    int block_size = 64;
    int first_layer_index = 0;
    int total_layers = 0;
    int fa_layers = 0;
    int gdn_layers = 0;
    int local_kv_heads = 0;
    int kv_head_start = 0;
    int head_dim = 0;
    ActivationPrecision k_precision = ActivationPrecision::FP32;
    ActivationPrecision v_precision = ActivationPrecision::FP32;
    TensorLayout kv_layout = TensorLayout::KV_POS_HEAD_DIM;
    size_t bytes_per_fa_layer_k = 0;
    size_t bytes_per_fa_layer_v = 0;
    size_t hybrid_state_bytes = 0;
    size_t mtp_kv_bytes = 0;
    size_t terminal_hidden_bytes = 0;
    size_t terminal_logits_bytes = 0;
    bool includes_hybrid_state = false;
    bool includes_mtp_state = false;
    bool includes_terminal_hidden = false;
    bool includes_terminal_logits = false;
};
```

Dense KV bytes are:

```text
sum over local FA layers:
  block_size * local_kv_heads * head_dim * (bytes_per_k + bytes_per_v)
```

Add tier backend interfaces:

```cpp
enum class PrefixStorageTier
{
    DeviceHot,
    Ram,
    Disk,
};

struct PrefixBlockHandle
{
    PrefixCacheKey key;
    PrefixStorageTier tier = PrefixStorageTier::Ram;
    PrefixPayloadLayout layout;
    void *kv_payload = nullptr;
    void *hybrid_payload = nullptr;
    void *mtp_payload = nullptr;
    void *terminal_hidden = nullptr;
    void *terminal_logits = nullptr;
    size_t total_bytes = 0;
};

class IPrefixStorageBackend
{
public:
    virtual ~IPrefixStorageBackend() = default;
    virtual bool canStore(size_t bytes) const = 0;
    virtual PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                       const PrefixPayloadLayout &layout) = 0;
    virtual bool release(const PrefixBlockHandle &handle) = 0;
    virtual bool hydrateToRam(const PrefixBlockHandle &handle,
                              PrefixBlockHandle *ram_handle) = 0;
};
```

`PrefixStateCache` owns the hash map, block chains, LRU, ref counts, and tier promotion policy. Storage backends own memory/files only.

Tier rules:

- RAM is source of truth for early correctness.
- Device-hot blocks are promotions of RAM blocks.
- Disk blocks hydrate into RAM before request-state import.
- Device-hot eviction drops only the hot handle.
- If RAM cannot hold one complete block, disable prefix caching with a warning.
- If device-hot cannot hold one block, disable only the device tier.

Add `BufferArena` staging ids for request-local restore/harvest:

```cpp
PREFIX_K_STAGING,
PREFIX_V_STAGING,
PREFIX_HYBRID_STATE_STAGING,
PREFIX_MTP_K_STAGING,
PREFIX_MTP_V_STAGING,
PREFIX_TERMINAL_HIDDEN,
PREFIX_TERMINAL_LOGITS,
```

Update `bufferIdName()` and `BufferArena::bufferNameToId()`. Persistent blocks remain outside arena storage.

Disk format:

```text
<disk_dir>/
  manifest.json
  blocks/
    <key-hex>.meta.json
    <key-hex>.kv.bin
    <key-hex>.hybrid.bin
    <key-hex>.mtp.bin
    <key-hex>.terminal_hidden.bin
    <key-hex>.terminal_logits.bin
```

Disk metadata includes format version, key fields, payload layout, token count, block index, precision/layout, byte lengths, checksums, and all fingerprints. Disk write failure should record stats and keep the RAM block; it should not fail inference.

### Files

- `src/v2/execution/prefix_cache/*`
- `src/v2/memory/BufferId.h`
- `src/v2/memory/BufferArena.h/.cpp`
- `src/v2/CMakeLists.txt`

### Tests

- `tests/v2/unit/prefix_cache/Test__PrefixBlockHash.cpp`
- `tests/v2/unit/prefix_cache/Test__PrefixCacheKey.cpp`
- `tests/v2/unit/prefix_cache/Test__PrefixStateCacheLRU.cpp`
- `tests/v2/unit/prefix_cache/Test__RamPrefixStorageBackend.cpp`
- `tests/v2/unit/prefix_cache/Test__DiskPrefixStorageBackend.cpp`

### Exit Criteria

- Prefix blocks can be allocated, inserted, found, evicted, and hydrated in unit tests without runner integration.
- RAM LRU accounting is deterministic.
- Disk round-trip validates checksums and rejects malformed metadata.
- Arena registration supports the new staging ids without changing existing graph behavior.

## Phase 4: Dense Prefix Cache End To End

### Goal

Enable RAM-backed dense prefix caching for non-MTP, non-hybrid models first. This proves the prefix lookup, populate, suffix forward, terminal logits, and harvest control flow.

### Implementation Details

Add persistent prefix-cache ownership outside `InferenceState`:

```cpp
std::shared_ptr<PrefixStateCache> prefix_cache_;
PrefixPayloadLayout prefix_layout_;
PrefixCacheStats prefix_cache_stats_;
```

Add concrete `DeviceGraphOrchestrator` helpers before promoting them to `IInferenceRunner`:

```cpp
PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) const;
bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0);
bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count);
bool restorePrefixTerminalState(const PrefixLookupResult &hit);
PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const;
bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0);
```

`clear_cache()` continues to call `state_.clear()` and reset graph/stage dynamic state, but must not evict `prefix_cache_`.

Import order for a prefix hit:

1. Clear request-local state.
2. Import FA KV into `state_.kv_cache` or `state_.pp_kv_caches`.
3. Restore terminal hidden/logits when present.
4. Set `positions[seq_idx]` and `sequence_lengths[seq_idx]` to `hit.cached_tokens`.
5. Invalidate or refresh graph dynamic params before replay.

Update `OrchestrationRunner::prefill()` flow:

```cpp
bool OrchestrationRunner::prefill(const std::vector<int32_t> &prompt_tokens)
{
    if (!initialized_ || prompt_tokens.empty())
        return setError("Runner not initialized or prompt tokens are empty");

    broadcastPrefillToWorkers(prompt_tokens);

    PrefixLookupResult local_hit;
    if (prefix_cache_enabled_)
        local_hit = runner_->lookupPrefix(prompt_tokens);

    int matched_tokens = coordinateMinimumMatchedTokens(local_hit.cached_tokens);

    runner_->clear_cache();

    PrefixLookupResult common_hit = local_hit.clampedTo(matched_tokens);
    if (prefix_cache_enabled_ && matched_tokens > 0)
    {
        if (!runner_->populatePrefix(common_hit))
            matched_tokens = 0;
    }

    int suffix_start = matched_tokens;
    int suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;

    if (suffix_len > 0)
    {
        runner_->forward(prompt_tokens.data() + suffix_start, suffix_len);
        prefill_logits_ready_ = true;
    }
    else if (common_hit.has_terminal_logits)
    {
        runner_->restorePrefixTerminalState(common_hit);
        prefill_logits_ready_ = true;
    }
    else
    {
        matched_tokens = std::max(0, matched_tokens - prefix_block_size_);
        runner_->clear_cache();
        runner_->populatePrefix(common_hit.clampedTo(matched_tokens));
        suffix_start = matched_tokens;
        suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;
        runner_->forward(prompt_tokens.data() + suffix_start, suffix_len);
        prefill_logits_ready_ = true;
    }

    if (prefix_cache_enabled_)
        runner_->harvestPrefix(prompt_tokens, static_cast<int>(prompt_tokens.size()));

    return true;
}
```

Full-hit first-token semantics:

- If terminal logits are present, set `prefill_logits_ready_ = true` and skip refeeding the final prompt token.
- If terminal logits are absent, recompute the final block or final token before decode.
- Do not duplicate the final prompt token; doing so corrupts KV and GDN state.

### Files

- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h` after the concrete DGO helpers stabilize.
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp` only for passthrough/min coordination once single-device works.

### Tests

- Dense repeated prompt: no-cache vs RAM prefix cache logits/tokens match under greedy sampling.
- Shared prefix, different suffix: suffix logits match baseline.
- Full-hit with terminal logits: first generated token matches baseline and does not refeed prompt tail.
- Full-hit without terminal logits: final block recompute path matches baseline.
- Budget eviction: evicted block misses, non-evicted block hits.

### Exit Criteria

- Dense CPU path works end-to-end with RAM prefix cache.
- Dense GPU path is either supported or cleanly bypassed by backend support checks.
- Normal generation with prefix cache disabled remains unchanged.

## Phase 5: MTP Loading, Sidecar Graph, And Shifted Cache

### Goal

Load MTP weights and build the MTP sidecar execution path without changing decode behavior yet.

### Implementation Details

Add model-facing MTP weights:

```cpp
struct MTPDepthWeights
{
    TensorBase *fc = nullptr;
    TensorBase *pre_fc_norm_hidden = nullptr;
    TensorBase *pre_fc_norm_embedding = nullptr;
    TensorBase *final_norm = nullptr;
    LayerWeights fa_block;
};

struct MTPWeights
{
    int depth = 0;
    bool use_dedicated_embeddings = false;
    std::vector<MTPDepthWeights> depths;
};
```

Add `MTPWeights mtp;` to `ModelWeights`, plus prepared-weight bindings if graph builders need `PreparedWeightStore` refs.

Probe and support these Qwen3.5/Qwen3.6 tensor names first:

```text
mtp.fc.weight
mtp.pre_fc_norm_hidden.weight
mtp.pre_fc_norm_embedding.weight
mtp.norm.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.mlp.down_proj.weight
```

If `mtp.num_hidden_layers` metadata is absent but `mtp.fc.weight` exists, infer depth 1 and log the inference. If `--mtp` is explicit and required tensors are missing, fail loudly. If MTP is not requested, disable MTP and keep normal inference.

Add MTP buffers to `BufferId` and arena mapping:

```cpp
MTP_EMBEDDING,
MTP_NORM_HIDDEN,
MTP_NORM_EMBEDDING,
MTP_CONCAT,
MTP_PROJECTED,
MTP_HIDDEN,
MTP_Q_PROJ,
MTP_K_PROJ,
MTP_V_PROJ,
MTP_Q_ROPE,
MTP_K_ROPE,
MTP_ATTN_OUTPUT,
MTP_ATTN_PROJ,
MTP_GATE_PROJ,
MTP_UP_PROJ,
MTP_FFN_OUTPUT,
MTP_LOGITS,
```

Add request-local MTP cache state:

```cpp
std::vector<std::unique_ptr<IKVCache>> mtp_kv_caches;
std::shared_ptr<TensorBase> prefix_terminal_hidden;
std::shared_ptr<TensorBase> prefix_terminal_logits;
```

Build a sidecar graph method in Qwen35/Qwen36 graph code:

```cpp
ComputeGraph Qwen35Graph::buildMTPGraph(
    int depth_idx,
    const MTPDepthWeights &weights,
    const MTPForwardInput &input,
    MTPForwardOutput &output);
```

MTP graph sequence:

1. Embed draft token using shared embedding table unless future metadata requests dedicated embeddings.
2. RMSNorm main hidden with `mtp.pre_fc_norm_hidden.weight`.
3. RMSNorm draft embedding with `mtp.pre_fc_norm_embedding.weight`.
4. Concatenate `[norm_hidden ; norm_embedding]` into `MTP_CONCAT`.
5. Project with `mtp.fc.weight` into `MTP_PROJECTED`.
6. Run one full-attention block using existing Qwen35 FA stages: input norm, gated Q projection, Q/K norms, partial RoPE, MTP KV append/read, attention, attention output gate, Wo, residual, FFN norm, SwiGLU FFN, residual.
7. Apply `mtp.norm.weight`.
8. Run shared LM head into `MTP_LOGITS`.

Shifted-cache invariant after prompt length `N`:

```text
main KV/GDN state: state after tokens [0, N)
MTP depth-0 KV:    state after MTP inputs for pairs (hidden[i], token[i+1]) where i in [0, N-1)
terminal hidden:   main hidden row for token N-1
terminal logits:   logits predicting token N
```

Full prefix hits with MTP require terminal hidden and terminal logits. If terminal hidden is unavailable, reduce the hit by one block and recompute. Do not synthesize terminal hidden from logits.

### Files

- `src/v2/models/GraphTypes.h`
- `src/v2/models/qwen35/Qwen35GraphConfigBuilder.cpp`
- `src/v2/models/qwen35/Qwen35Graph.h/.cpp`
- Qwen36 graph/config files if separate from Qwen35.
- `src/v2/memory/BufferId.h`
- `src/v2/memory/BufferArena.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`

### Tests

- Model-gated MTP metadata and tensor inventory for Qwen3.6 dense and MoE files.
- MTP graph construction test with mocked or small fixture weights.
- MTP shifted-cache count probe: after prefill of `N`, MTP cache depth 0 has `max(0, N - 1)` logical tokens.
- MTP sidecar one-step logits smoke test against a Python/reference capture once available.

### Exit Criteria

- MTP weights load or explicitly disable with clear diagnostics.
- MTP sidecar graph can build without affecting normal graphs.
- MTP prefill cache invariant is test-backed.

## Phase 6: MTP Decode, Verification, And Rollback

### Goal

Enable greedy MTP speculative decoding behind `--mtp`, using the same snapshot/restore contract as prefix cache.

### Implementation Details

Extend `LMHeadStage::Params`:

```cpp
bool compute_all_positions = false;
```

Update LM-head row count:

```cpp
const int lm_m = params_.compute_all_positions ? params_.seq_len
               : (params_.seq_len > 1) ? 1
               : params_.seq_len;
const int lm_activation_offset = params_.compute_all_positions ? 0
                               : activationRowOffsetForLogits();
```

Update `estimatedFlops()`, `estimatedMemoryBytes()`, and `buildDumpInfoImpl()` to use the same effective row count.

Add runner hooks:

```cpp
bool forwardMTP(int32_t draft_condition_token);
const float *mtpLogits() const;
bool setComputeAllPositionLogits(bool enabled);
const float *getAllPositionLogits() const;
PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const;
bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
```

Start with conservative restore-then-replay:

```cpp
GenerationResult OrchestrationRunner::decodeStepMTP()
{
    PrefixStateSnapshot checkpoint = runner_->captureLivePrefixState();

    std::vector<int32_t> draft_tokens;

    if (prefill_logits_ready_)
    {
        draft_tokens.push_back(sampleFromCurrentLogits());
        prefill_logits_ready_ = false;
    }
    else
    {
        runner_->forward(&last_token_, 1);
        draft_tokens.push_back(sampleFromCurrentLogits());
    }

    runner_->forwardMTP(draft_tokens.back());
    draft_tokens.push_back(sampleFromMTPLogits());

    runner_->setComputeAllPositionLogits(true);
    runner_->forward(draft_tokens.data(), static_cast<int>(draft_tokens.size()));
    runner_->setComputeAllPositionLogits(false);

    AcceptanceResult accepted = verifyGreedy(draft_tokens, runner_->getAllPositionLogits());

    runner_->restoreLivePrefixState(checkpoint);
    runner_->forward(accepted.tokens.data(), static_cast<int>(accepted.tokens.size()));

    last_token_ = accepted.tokens.back();
    return accepted.toGenerationResult();
}
```

Dense-only truncate optimization can come later. Do not optimize around GDN until forced reject tests pass, because GDN state is in-place.

### Files

- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp`
- `src/v2/execution/compute_stages/stages/LMHeadStage.h/.cpp`

### Tests

- LMHead multi-position row count and dump info tests.
- MTP greedy acceptance with deterministic sampling: MTP-enabled output equals main-model greedy output.
- Forced mismatch test: rollback and replay match no-MTP state.
- Prefix + MTP full-hit path: terminal hidden/logits restored, no final-token duplication.

### Exit Criteria

- MTP greedy decode is correct but not necessarily faster.
- MTP can be enabled/disabled without changing non-MTP outputs.
- Rollback restores dense and hybrid state in tests.

## Phase 7: Hybrid GDN Prefix State

### Goal

Extend prefix cache and MTP rollback to Qwen3.5/Qwen3.6 hybrid FA+GDN models.

### Implementation Details

Add explicit hybrid state APIs to `IHybridKVCache`:

```cpp
struct HybridPrefixStateMetadata
{
    int total_layers = 0;
    int gdn_layers = 0;
    size_t host_bytes = 0;
    size_t device_bytes = 0;
    bool has_device_kernel_state = false;
};

struct HybridPrefixStateDescriptor
{
    int seq_idx = 0;
    int logical_token_count = 0;
    void *stream = nullptr;
};

virtual HybridPrefixStateMetadata hybridPrefixStateMetadata() const = 0;

virtual bool exportHybridPrefixState(
    const HybridPrefixStateDescriptor &desc,
    void *dst_host,
    void *dst_device) const = 0;

virtual bool importHybridPrefixState(
    const HybridPrefixStateDescriptor &desc,
    const void *src_host,
    const void *src_device) = 0;
```

Add kernel state methods where missing:

```cpp
class ITensorShortConvolution
{
public:
    virtual size_t stateBytes() const { return 0; }
    virtual bool exportState(void *dst_host, void *dst_device, void *stream) const { return false; }
    virtual bool importState(const void *src_host, const void *src_device, void *stream) { return false; }
};

class ITensorGatedDeltaNet
{
public:
    virtual size_t stateBytes() const { return 0; }
    virtual bool exportState(void *dst_host, void *dst_device, void *stream) const { return false; }
    virtual bool importState(const void *src_host, const void *src_device, void *stream) { return false; }
};
```

Rules:

- CPU hybrid export copies `HybridGDNLayerState::recurrence_state` and `conv_state` in global-layer order.
- GPU hybrid export/import includes device-resident short-conv and GDN recurrence kernel state.
- FA KV still goes through `IKVCache::exportLogicalBlock()` using global FA layer ids; hybrid cache implementation handles compressed FA-index remap.
- `clear_layer(global_gdn_layer)` must reset host vectors and GPU kernel state. Prefix restore uses import API, not raw vector mutation.

Hybrid payload layout adds one complete GDN-state snapshot per prefix block. This is redundant but simple and correct. Optimize delta storage later.

### Files

- `src/v2/kernels/IHybridKVCache.h`
- `src/v2/kernels/HybridKVCacheConfig.h/.cpp`
- `src/v2/kernels/cpu/CPUHybridRingKVCache.h`
- `src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h`
- `src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h`
- `src/v2/kernels/*/short_convolution*`
- `src/v2/kernels/*/gated_delta_net*`
- `src/v2/execution/prefix_cache/PrefixPayloadLayout.*`

### Tests

- Hybrid state metadata byte count tests.
- CPU/CUDA/ROCm hybrid export/import round-trip tests.
- Clear/import does not lose GDN kernel object pointers.
- Qwen3.5/Qwen3.6 suffix parity with prefix cache enabled/disabled.
- MTP forced rejection on hybrid model restores/replays to no-MTP state.

### Exit Criteria

- Hybrid prefix restore reproduces no-cache suffix logits.
- Hybrid MTP rollback is correct under forced mismatch.
- Existing CUDA/ROCm hybrid reset tests still pass.

## Phase 8: MoE Safety, Domain-Scoped Rebalance, And Parallelism

### Goal

Make prefix cache and MTP safe across MoE placement changes, dynamic rebalance, and multi-device execution. This phase incorporates the Tier 1.5 rebalance-domain cleanup from `PREFILL_HIP_GRAPH_CAPTURE_PLAN.md`: the rebalance controller must stop being a CPU-socket special case and become an ExpertParallel domain controller with explicit participants, placement epochs, masks, replicas, and histogram ownership.

### Implementation Details

Build MoE fingerprint material from domain-scoped placement state:

- `DeviceMoELayerRuntime::active_bank`
- `DeviceMoELayerRuntime::active_epoch`
- `DeviceMoEPlacementBank::expert_count`
- `DeviceMoEPlacementBank::experts[].logical_expert_id`
- `DeviceMoEPlacementBank::experts[].owner_participant`
- `DeviceMoEPlacementBank::experts[].local_slot`
- `DeviceMoEPlacementBank::experts[].flags`
- `DeviceMoEPlacementBank::local_compute_mask[]`
- `DeviceMoEPlacementBank::replica_role[]`
- ExpertParallel domain id/name.
- Domain participant ids and `GlobalDeviceAddress` metadata.
- Routed tier domain, continuation domain, and shared expert domain where an overlay plan is active.

Expose fingerprint material with either:

```cpp
virtual uint64_t placementFingerprint() const = 0;
```

or a helper over public `hostLayerState(layer_idx)` if a virtual addition is too invasive.

Add explicit rebalance domain types:

```cpp
using ExpertParallelDomainId = std::string;
using ExpertParallelParticipantId = int;

struct ExpertParallelParticipant
{
    ExpertParallelParticipantId participant_id = -1;
    GlobalDeviceAddress global_device;
    DeviceId local_device;
    int world_rank = -1;
    int rank_in_domain = -1;
    int numa_node = -1;
    std::string domain_kind; // cpu_global_tp, local_tp, routed_overlay, single, ...
};

struct MoERebalanceDomainConfig
{
    ExpertParallelDomainId domain_id;
    std::vector<ExpertParallelParticipant> participants;
    bool observe_only = false;
    bool can_rebalance = false;
    uint64_t placement_epoch = 0;
};
```

Domain model and vocabulary:

- Public rebalance APIs should use `domain`, `participant`, and `device` language.
- `computeExpertMasks(int participant_id)` replaces socket-oriented APIs.
- `owner_participant` replaces `owner_socket`; `num_participants` replaces `num_sockets`.
- Compatibility wrappers may remain temporarily, but new call sites should not introduce new public `socket_id` terminology except CPU topology adapters.
- `DecodeExpertHistogram` may remain as an implementation, but the runtime contract is a domain-scoped histogram. Add `DomainExpertHistogram` wrapper/alias if renaming is too much churn.

Wiring and ownership:

- Resolve zero or more `MoERebalanceDomainConfig` objects from the execution plan and optional overlay plan.
- Attach exactly one controller per ExpertParallel rebalance domain.
- Expose controller lookup by domain id and iteration over all active domains; do not use "first DGO controller" as the multi-domain API.
- Single-device MoE domains are observe-only: collect telemetry when requested, but never mutate masks or transfer weights.
- CPU `-d cpu` remains one CPU GlobalTP domain whose participant ids match rank-in-domain.
- LocalTP participants are local devices in plan order.
- Multi-node GlobalTP uses global/domain rank, not `local_rank`, as participant identity.
- Overlay routed expert tiers are eligible rebalance domains. Continuation/base dense domains are not expert-placement rebalance domains.

Placement, masks, replicas, and epochs:

- Treat expert masks as domain-local placement views: `mask[layer][expert] == participant owns or serves this expert`.
- Track base owner participant, replica participant(s), active compute participant, and prefill owner participant separately.
- `ExpertReplicaSet` stores a domain id and cannot be applied to a different domain.
- Weight transfer is domain-aware. CPU cross-rank paths can keep MPI send/recv; local-device and overlay transfers use domain-specific transfer code.
- `DeviceGraphOrchestrator::applyExpertMasks()` and `RankOrchestrator` mask fanout must carry domain id and participant id.
- Every successful base placement, mask, replica, owner-map, or runtime-table bank update increments only the affected domain's placement epoch.
- When a placement update occurs, either the new epoch is part of the prefix/MTP fingerprint or `PrefixStateCache::invalidateWhere(predicate)` invalidates entries for the old fingerprint.

Histogram contract:

- Add histogram source identity: `DecodeToken`, `PrefillChunk`, `SyntheticTest`, or equivalent.
- Decode remains the first production histogram source.
- Prefill histogram support is chunk-boundary only. Counts merge after a chunk completes; no mid-graph or mid-stage rebalance.
- Window accounting counts real tokens, not padded bucket tokens.
- Single-participant domains may collect histograms in observe mode, but `shouldRebalance()` returns false with reason `single_participant_observe_only`.
- Rebalance decisions expose reason codes such as `window_not_full`, `mode_off`, `dynamic_disabled_for_domain`, `single_participant_observe_only`, and `ready`.

Parallelism coordination:

- LOCAL TP: each child `DeviceGraphOrchestrator` stores local shard payload; matched token count is minimum across child runners.
- PP: each stage stores local layer range only; matched token count remains global and is minimum across stages.
- GLOBAL/NODE_LOCAL TP: use `MPI_Allreduce(MIN)` over matched token counts before populate.
- Root rank must not populate a longer prefix than any non-root rank can restore.
- Harvest can be local after successful prefill; future lookups remain safe because min coordination handles partial availability.

Graph capture:

- Prefill graph capture keys or invalidation must include MoE placement epoch.
- Tier 1 graph capture can invalidate conservatively on any mask/replica/domain placement mutation.
- Tier 2 segmented graph capture uses placement epoch and runtime-table bank flips at chunk boundaries only.
- Sparse host/collective overlay stages stay non-capturable until the segmented Tier 2 path defines explicit manual collective boundaries.
- If a MoE sparse collective has not completed, follow-on transfers must wait for `collective_complete` before continuing.

### Files

- `src/v2/execution/moe/MoERuntimeTable.h/.cpp`
- `src/v2/execution/moe/MoERebalanceController.*`
- `src/v2/execution/moe/DecodeExpertHistogram.h/.cpp`
- `src/v2/execution/moe/MoEExpertParallelPlan.h`
- `src/v2/execution/moe/MoEExpertOverlayRuntimePlan.*`
- `src/v2/config/ExecutionDomainDefinition.h/.cpp`
- `src/v2/execution/config/RuntimeConfig.h`
- `src/v2/execution/factory/InferenceRunnerFactory.cpp`
- `src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp`
- `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/execution/prefix_cache/PrefixCacheFingerprint.*`

### Tests

- Stable MoE placement hits cache.
- Rebalance epoch change misses or invalidates.
- Decode histogram is not restored as payload.
- Domain construction for CPU GlobalTP, LocalTP, single-device, synthetic multi-node GlobalTP, and overlay routed tiers.
- Controller lookup by ExpertParallel domain id and iteration over all active local controllers in single-device and composite runners.
- Legacy two-CPU-rank rebalance produces the same masks through participant APIs as the previous socket APIs.
- CPU GlobalTP controller construction produces one `global_tp_domain_<id>` controller with rank-in-domain participant masks matching the legacy socket split.
- Single-device dynamic config degrades to observe-only with a clear reason and no mask mutation.
- Domain mismatch when applying masks or replicas fails before mutation.
- Factory controller construction attaches routed-overlay domain controllers from `MoEExpertOverlayRuntimePlan` while preserving legacy primary-controller access.
- Placement epoch increments only for the affected domain.
- Prefill chunk histogram unit test merges only real-token counts.
- Static grep guard: new rebalance code should not introduce new public `socket_id` terminology outside compatibility wrappers or CPU topology adapters.
- LOCAL TP child miss clamps all children to common min prefix length.
- GLOBAL TP rank miss clamps all ranks via MPI min reduction.
- PP stage miss clamps the full pipeline to common min prefix length.
- Prefill graph-cache maintenance units block rebalance while histograms are unmerged, manual boundaries are incomplete, graph capture/replay is active, or participants are not at the same boundary.
- Sparse dispatch/return units prove graph manual-boundary completion is driven by collective completion before dependent stages continue.
- Large Qwen3.6 MoE parity tests are normal parity-suite entries. They may skip only for missing model files, metadata fixtures, required hardware, or MPI topology.

### Exit Criteria

- MoE prefix cache never reuses state across incompatible expert placement.
- Dynamic rebalance and prefix cache can coexist without silent wrong outputs.
- Multi-device prefix hits are all-or-common-prefix, never partially divergent.
- Public rebalance APIs and diagnostics use domain/participant/device language.
- Existing CPU GlobalTP `-d cpu` rebalance behavior is preserved as one domain-scoped instance.
- Single-device MoE is formalized as observe-only/no-rebalance.
- LocalTP and multi-node GlobalTP have unambiguous participant ids.
- Overlay routed expert domains have a controller attachment point before graph-captured sparse collectives are enabled.

## Phase 9: Observability, Diagnostics, And Rollout Controls

### Goal

Make prefix cache and MTP behavior explainable before expanding into TP, MoE, and performance work. Operators and tests must be able to tell whether a request hit, partially hit, missed, bypassed, rolled back, or accepted speculative tokens, and why.

### Implementation Details

Add stats and request summaries:

```cpp
struct PrefixCacheStats
{
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t partial_hits = 0;
    uint64_t matched_blocks = 0;
    uint64_t matched_tokens = 0;
    uint64_t stores = 0;
    uint64_t ram_bytes = 0;
    uint64_t device_bytes = 0;
    uint64_t disk_bytes = 0;
    uint64_t promotions = 0;
    uint64_t evictions = 0;
    uint64_t disk_hydrations = 0;
    uint64_t terminal_state_hits = 0;
    uint64_t hybrid_state_bytes = 0;
    uint64_t mtp_state_bytes = 0;
    uint64_t bypasses = 0;
    uint64_t unsupported_backend_bypasses = 0;
    uint64_t fingerprint_bypasses = 0;
    uint64_t terminal_state_bypasses = 0;
};

struct MTPStats
{
    uint64_t draft_steps = 0;
    uint64_t accepted_tokens = 0;
    uint64_t rejected_tokens = 0;
    uint64_t rollbacks = 0;
    uint64_t bypasses = 0;
    uint64_t verifier_runs = 0;
    uint64_t verifier_token_count = 0;
    uint64_t depth_policy_windows = 0;
    uint64_t depth_policy_updates = 0;
    uint64_t depth_policy_promotions = 0;
    uint64_t depth_policy_demotions = 0;
    uint64_t depth_policy_observe_recommendations = 0;
    int current_depth = 0;
    int min_depth = 0;
    int max_depth = 0;
};
```

Add structured per-request summary objects:

```cpp
struct PrefixCacheRequestSummary
{
    bool enabled = false;
    bool bypassed = false;
    std::string bypass_reason;
    bool hit = false;
    bool partial_hit = false;
    int requested_tokens = 0;
    int matched_tokens = 0;
    int matched_blocks = 0;
    bool terminal_logits_restored = false;
    bool terminal_hidden_restored = false;
    bool mtp_state_restored = false;
    bool hybrid_state_restored = false;
    std::string storage_tier; // ram, device-hot, disk-hydrated, mixed, none
};

struct MTPRequestSummary
{
    bool enabled = false;
    bool bypassed = false;
    std::string bypass_reason;
    bool adaptive_depth_enabled = false;
    std::string depth_policy_mode; // fixed, observe, dynamic
    int current_depth = 0;
    int min_depth = 0;
    int max_depth = 0;
    uint64_t depth_policy_updates = 0;
    std::string last_depth_policy_reason;
    uint64_t draft_steps = 0;
    uint64_t accepted_tokens = 0;
    uint64_t rejected_tokens = 0;
    uint64_t rollbacks = 0;
    double acceptance_rate = 0.0;
};
```

Expose summaries through:

- INFO-level per-request summary when prefix cache or MTP is enabled.
- DEBUG-level bypass details for unsupported topology, missing terminal hidden, missing MTP weights, fingerprint mismatch, RAM budget too small, or backend logical I/O unavailable.
- Existing profiling output where appropriate.
- `PrefixRuntimeStateSnapshot` so integration tests can inspect counters without parsing logs.
- Optional future server response headers/metadata.

Rollout controls:

- Keep prefix cache and MTP off by default.
- Allow dense RAM prefix cache to graduate first.
- Enable GPU/device-hot tier only after GPU logical I/O tests and focused GPU prefix integration tests pass.
- Enable dynamic MoE rebalance with prefix/MTP only after Phase 8 domain-scoped participant, placement epoch, and histogram tests pass.
- Enable LocalTP or GlobalTP prefix cache only after the Phase 10 common-prefix coordination tests pass.
- Enable MTP on single-device dense runners only after Phase 13 single-device parity passes.
- Consider MTP speedup or default-enablement claims only after the relevant backend passes Phase 13.5 small-M GEMV-many gates and Phase 13.7 stochastic-verifier gates for non-greedy chat defaults.
- Enable MTP on TP runners only after Phase 11 parity passes for the relevant TP scope.
- Enable MTP on MoE/ExpertParallel runners only after Phase 12 and Phase 13 MoE parity pass.
- Enable ExpertParallel graph-captured sparse overlay paths only after Phase 12 segmented-capture tests and Phase 14 benchmarks pass.
- Enable stochastic MTP only after Phase 13.7 proves exact speculative sampling semantics for the relevant backend/topology.
- Keep every large-model parity test registered in the parity suite, with explicit prerequisite skips for missing model files, metadata fixtures, required hardware, or MPI topology. Large-model benchmarks remain opt-in.

### Files

- `src/v2/execution/prefix_cache/PrefixCacheStats.h`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/execution/prefix_cache/PrefixCacheStateProbe.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/utils/BenchmarkRunner.cpp`
- `docs/v2/` follow-up benchmark notes or changelog files.

### Tests

- Stats counters increment deterministically in unit tests.
- Disabled feature paths produce zero/no-op stats.
- Unsupported feature paths increment bypass counters and preserve normal inference.
- INFO summaries are covered through state snapshots or log-capture tests.
- Benchmark smoke tests do not require large model files unless explicitly enabled.
- Regression filters before considering defaults:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_|^V2_Integration_PrefixCacheStateProbe" --output-on-failure --parallel
```

### Exit Criteria

- Users can tell whether a request hit, partially hit, missed, or bypassed prefix cache and why.
- MTP acceptance and rollback rates are visible.
- Default-off behavior is unchanged and visible in stats.
- Later parity and benchmark phases can consume structured counters instead of scraping logs.

## Phase 10: Prefix Cache Coordination Across TP, PP, And MoE Domains

### Goal

Make prefix-cache lookup, populate, and harvest correct for multi-device execution before enabling MTP on those topologies. A prefix hit must be an all-or-common-prefix decision across every participant that owns restorable request state.

### Implementation Details

Add a coordinator layer above per-device prefix lookup:

```cpp
struct PrefixParticipantLookup
{
    std::string domain_id;
    int participant_id = -1;
    DeviceId device;
    uint64_t placement_epoch = 0;
    uint64_t fingerprint_key = 0;
    bool cache_enabled = false;
    bool hit = false;
    int matched_tokens = 0;
    int matched_blocks = 0;
    bool has_terminal_logits = false;
    bool has_terminal_hidden = false;
    std::string bypass_reason;
};

struct PrefixCoordinationResult
{
    bool cache_enabled = false;
    std::string domain_id;
    uint64_t placement_epoch = 0;
    uint64_t fingerprint_key = 0;
    int common_matched_tokens = 0;
    int common_matched_blocks = 0;
    bool common_terminal_logits = false;
    bool common_terminal_hidden = false;
    std::string clamp_reason;
    std::vector<PrefixParticipantLookup> participants;
};
```

Coordination rules:

- Every participant computes lookup against the same token chain and fingerprint part set.
- MoE participants use the Phase 8 domain id, participant id, and placement epoch in lookup summaries and diagnostics.
- A non-zero fingerprint key mismatch across participants makes the coordinated hit unusable, even if all participants report the same token count.
- The usable hit length is the minimum matched token count across required participants.
- Terminal logits/hidden are usable only if every participant that needs them has them.
- Populate must be clamped before any participant imports state.
- If populate fails on one participant, all participants clear request-local state and replay from the common fallback point.
- Harvest happens only after successful prefill and stores each participant's local shard under the same logical block keys.
- Device-hot and disk tiers remain local to the participant that owns the payload. Promotion must not imply other participants have the same block.

Topology-specific behavior:

- SingleDevice: existing `DeviceGraphOrchestrator` lookup is the coordination result.
- LocalTP: `RankOrchestrator` queries all child `DeviceGraphOrchestrator` instances, computes the local minimum, then populates every child to that common length.
- GlobalTP / NodeLocalTP: each rank computes its local result, then a scalar domain reduction computes the minimum matched tokens and terminal-state availability. Prefer a small typed scalar coordination helper over ad hoc MPI calls in runner code.
- PP: each stage owns only its local layer range, but the logical token count is global. The pipeline uses the minimum across stages.
- ExpertParallel MoE: continuation participants own main hidden/KV/logits state. Routed expert-only participants own expert weights and sparse collective scratch, not prefix KV payload. MoE placement, domain participant ids, placement epoch, and overlay plan remain fingerprint material for all roles.

Add scalar coordination helpers:

```cpp
class IPrefixCollectiveCoordinator
{
public:
    virtual ~IPrefixCollectiveCoordinator() = default;
    virtual bool allMinInt(int local_value, int *global_value) = 0;
    virtual bool allMinUInt64(uint64_t local_value, uint64_t *global_value) = 0;
    virtual bool allMaxUInt64(uint64_t local_value, uint64_t *global_value) = 0;
    virtual bool allAndBool(bool local_value, bool *global_value) = 0;
    virtual bool allOrBool(bool local_value, bool *global_value) = 0;
};
```

Implementations can wrap:

- Local in-process reduction for LocalTP.
- Domain communicator or `IMPIContext` for GlobalTP / NodeLocalTP.
- Pipeline stage coordination where `RankOrchestrator` already owns the stage list.

The coordinator must be used for prefix-cache decisions only. Tensor data movement still goes through `TransferEngine`, `IKVCache` logical I/O, and existing collective stages.

### Files

- `src/v2/execution/prefix_cache/PrefixCacheCoordinator.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/interfaces/ICollectiveContext.h` or a narrow prefix-specific coordinator wrapper.

### Tests

- Unit tests for min-token clamping, terminal-state AND logic, and failed-populate fallback.
- LocalTP fake-runner test where one child misses and all children replay from the clamped prefix.
- GlobalTP/NodeLocalTP MPI unit or focused integration test using small CPU runners and an intentional rank-local miss.
- PP staged fake-runner test where one stage lacks a block and the whole pipeline clamps.
- Fingerprint mismatch test proving equal token counts do not permit a coordinated hit when participants restored incompatible state.
- MoE fingerprint test proving overlay placement changes still miss even when token blocks exist.

### Exit Criteria

- Multi-device prefix hits are common-prefix only, never participant-divergent.
- Prefix cache remains disabled or bypassed on any topology without a working coordinator.
- The focused multi-device prefix tests pass without running the full heavy integration suite.

## Phase 11: TP-Compatible MTP Sidecar Execution

### Goal

Promote MTP sidecar execution from a single-device helper into a normal TP-aware graph path for dense models. The sidecar must participate in the same LocalTP, NodeLocalTP, and GlobalTP collectives as the main graph.

### Implementation Details

The sidecar graph is a graph fragment, not a nested multi-device subgraph. For TP, each participant builds and executes its participant-local MTP graph using the same `GraphConfig`, `ITPContext`, sharded weights, prepared-weight store, and collective ordering as the main graph.

Required changes:

- Remove the `RankOrchestrator::forwardMTP()` single-child limitation after TP tests exist.
- Add `RankOrchestrator` MTP coordination that launches `forwardMTP()` on every child runner for LocalTP.
- For GlobalTP/NodeLocalTP, all ranks in the TP domain must enter `forwardMTP()` in identical order.
- Add domain-wide MTP checkpoint/restore wrappers around `captureLivePrefixState()` and `restoreLivePrefixState()`.
- Gather or reduce MTP logits with the same policy as main logits. Vocab-sharded LM head output cannot be sampled from only participant 0.
- Broadcast or otherwise coordinate the selected greedy draft token so every participant verifies the same sequence.
- Disable the current MPI-world-size guard in `OrchestrationRunner::canUseMTP()` only after domain-wide checkpoint, logits, sampling, and rollback tests pass.

MTP graph requirements:

- Column-parallel stages produce local shards only.
- Row-parallel stages insert `TPAllreduceStage` through `QwenGraphBase::createTPAllreduceStage()`.
- LM-head sharding follows the existing logits gather or distributed sampling path.
- MTP KV cache uses the same local KV-head assignment as the participant's attention block.
- Terminal hidden must be the replicated post-allreduce hidden row. If a topology only has sharded hidden at that point, it must allgather or bypass MTP.
- All tensor movement for terminal hidden/logits and sidecar inputs goes through `TransferEngine` or graph-stage buffer contracts.

Rollback rules:

- A checkpoint covers main KV, MTP KV, hybrid state if present, positions, terminal hidden/logits, and per-runner decode bookkeeping.
- Restore happens on every participant before replay.
- A verifier mismatch increments rollback counters once per request step, not once per participant.
- If one participant cannot restore, all participants clear and replay from the last agreed safe state.

### Files

- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/models/qwen/QwenGraphBase.cpp`
- `src/v2/models/qwen35/Qwen35Graph.cpp`
- `src/v2/execution/prefix_cache/PrefixStateSnapshot.h`

### Tests

- Unit graph-construction test proving dense MTP sidecar inserts TP allreduce when row-parallel MTP weights are sharded.
- LocalTP dense focused integration test on CPU or ROCm: MTP-enabled greedy output equals MTP-disabled greedy output.
- GlobalTP/NodeLocalTP CPU focused integration test with two ranks: domain-wide MTP draft token and rollback are consistent.
- Forced mismatch rollback test across TP participants.
- Full prefix hit plus MTP test across TP participants, including terminal hidden/logit restore.

### Exit Criteria

- Dense MTP is correct for SingleDevice CPU/CUDA/ROCm and for the first supported LocalTP/GlobalTP dense topology.
- No TP participant runs a sidecar step alone.
- The sidecar uses existing collective abstractions rather than direct MPI/NCCL/RCCL calls in MTP-specific code.

## Phase 12: ExpertParallel Sparse MoE MTP Sidecar And Segmented Graph Capture

### Goal

Make MTP compatible with MoE models and ExpertParallel overlay execution, including sparse collectives across heterogeneous devices such as 2x ROCm plus 2x CPU dual-socket. This phase also incorporates Tier 2 from `PREFILL_HIP_GRAPH_CAPTURE_PLAN.md`: ExpertParallel overlay prefill graph capture through fixed-size chunks, domain-local graph caches, placement epochs, runtime-table bank flips, and explicit manual sparse collective boundaries.

### Implementation Details

The MoE MTP sidecar must reuse graph-native MoE building blocks instead of taking a dense-only shortcut. Current dense MTP graph construction rejects MoE weights; this phase removes that limitation only after the sparse sidecar path is implemented and tested.

Graph rules:

- Keep graphs per-device and symmetric. Do not introduce nested multi-device sidecar graphs.
- Add an MTP-specific graph namespace for MoE collective keys, including request generation, decode step, MTP depth, layer, tier, participant, and direction.
- Every participant in the ExpertParallel plan executes the same logical sequence of dispatch, local expert, return-reduce, and no-op stages.
- Participants with no routed rows no-op but still participate in the collective sequence when required.
- Sparse collective completion must be observed before the continuation domain consumes returned rows.
- Failure or abort in one sparse collective aborts the sidecar step for all participants and rolls back to the domain checkpoint.
- Rebalance is allowed between graph executions only. It is never allowed during capture, replay, or while a sparse dispatch/return collective is incomplete.

Sidecar MoE sequence:

1. Build MTP projected hidden on continuation participants.
2. Run the MTP attention block and MTP KV append on the attention/continuation domain.
3. Run MTP FFN norm and MoE router using the MTP block's gate weights.
4. Build sparse dispatch payloads from MTP route indices and weights.
5. Dispatch routed rows to ExpertParallel owner participants.
6. Execute local routed experts on owner devices using resident expert GEMM engines.
7. Return/reduce sparse expert outputs to the continuation domain.
8. Run shared expert path if present and combine routed plus shared output.
9. Run final MTP norm and LM head, then gather/sample logits consistently across the continuation domain.

Persistent and rollback state:

- Persist MTP shifted KV payload for attention participants when prefix cache is enabled.
- Persist terminal hidden/logits as in dense MTP.
- Do not persist MoE route/grouped scratch, sparse payload buffers, histograms, or transient workspaces.
- Fingerprints include ExpertParallel plan id, owner map generation, active bank/epoch, expert masks, replica roles, routed tier domains, continuation domain, and shared expert domain.
- Dynamic rebalance invalidates or misses old MoE prefix blocks before they can restore MTP state.

Heterogeneous execution:

- ROCm expert participants use prepared ROCm expert GEMM engines.
- CPU expert participants use CPU expert GEMM engines and host-staged sparse transfer where needed.
- Continuation-domain transfer back to GPU must use `TransferEngine` or existing sparse-return staging, not tensor flag mutation.
- The correctness path may use host-staged sparse collectives first; optimized device-native sparse collectives can follow after parity.

Tier 2 chunked prefill and graph capture:

```text
chunk 0: captured prefill segment(s) for tokens [0, n)
         merge prefill expert histograms
         optional rebalance + placement epoch flip

chunk 1: captured prefill segment(s) for tokens [n, 2n)
         merge prefill expert histograms
         optional rebalance + placement epoch flip
```

Chunk scheduler requirements:

- Promote the bucket/chunk primitive into a reusable scheduler, not only a server padding trick.
- Add explicit chunk policy fields: fixed interval, bucket list, minimum/maximum rebalance interval, and real-token range.
- Padded bucket tokens must not contribute to expert histograms, prefix terminal logits, or rebalance windows.
- After each graph chunk completes, merge per-layer prefill counts into the domain histogram outside graph replay.
- Reset per-chunk runtime counters after successful merge so repeated syncs are idempotent.
- If `LLAMINAR_GPU_GRAPHS=1` selects this path and preflight/capture/replay cannot satisfy the contract, fail with phase/domain/stage reason rather than silently falling back.

Placement epochs and runtime-table bank flips:

- Placement epoch advances whenever expert masks, replicas, owner maps, or runtime-table banks change.
- Conservative implementation: include placement epoch in graph cache keys and recapture on epoch change.
- Target implementation: captured kernels read placement through stable `DeviceMoELayerRuntime*` pointers and double-buffered banks; cache keys include runtime-table schema/capacity, while replay observes active-bank changes.
- Use `MoERuntimeTable::prepareInactiveBank()` and `flipActiveBank()` outside capture/replay.
- Every active local expert descriptor in a new bank must have resident prepared gate/up/down payloads before the bank flip.
- Hidden stage-local placement state must not be the source of truth for captured replay once runtime-table indirection is active.

Domain-local graph caches:

- Cache graph segments per overlay domain and participant, not only per root device.
- Cache key fields include domain id/name, participant id, bucket length, real-token count, placement epoch or runtime-table schema, layer range, and graph topology signature.
- Continuation/base-domain graph segments are separate from expert-domain graph segments.
- All participants must agree on chunk id, bucket length, placement epoch, graph topology signature, and collective keys before capture/replay starts.
- Multi-node GlobalTP and overlay tests must prove domain participant id is used instead of `local_rank` aliases.

Segmented sparse collective capture:

- Keep sparse overlay collectives outside monolithic HIP/CUDA graph capture until the collective backend has a graph-safe contract.
- Split each overlay prefill chunk into explicit segments:
    - captured base/route segment
    - manual sparse dispatch collective
    - captured expert-domain local compute segment(s)
    - manual sparse return-reduce collective
    - captured continuation segment
- Preserve `MoEOverlayCollectiveResult::collective_complete` gating. No captured continuation segment or transfer may launch until the collective has completed.
- Fixed-capacity device-resident descriptors can become capturable later; host-staged sparse row metadata remains a manual boundary.
- For every manual boundary, define tensors and coherence states that must be ready before the next captured segment launches.

Chunk-boundary rebalance:

- Add a prefill-chunk maintenance hook called only after graph execution completes, histograms are merged, and all required sparse collectives complete.
- Rebalance must enforce: no active capture, no active replay, all participating domains at the same chunk boundary, and all required collectives complete.
- Transfer or prepare expert payloads before publishing a new placement epoch/runtime-table bank.
- If topology, capacity, or descriptor schema changes, invalidate and recapture the next chunk graph explicitly.
- Raw expert payload release must be delayed until no future rebalance arrival or recapture can require it.

### Files

- `src/v2/models/qwen35moe/Qwen35MoEGraph.h/.cpp`
- `src/v2/models/qwen35/Qwen35Graph.h/.cpp`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- `src/v2/execution/local_execution/engine/PrefillGraphCache.h/.cpp`
- `src/v2/execution/moe/MoEOverlaySparseCollective.h/.cpp`
- `src/v2/execution/moe/MoERuntimeTable.h/.cpp`
- `src/v2/execution/moe/MoERebalanceController.h/.cpp`
- `src/v2/execution/moe/DecodeExpertHistogram.h/.cpp`
- `src/v2/execution/moe/ExpertWeightTransfer.*`
- `src/v2/execution/compute_stages/stages/MoESparseDispatchStage.*`
- `src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.*`
- `src/v2/execution/compute_stages/stages/MoELocalExpertStage.*`
- `src/v2/execution/compute_stages/stages/MoEExpertDispatchStage.*`
- `src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.*`
- `src/v2/execution/moe/MoEExpertOverlayRuntimePlan.*`
- `src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp`
- `src/v2/execution/prefix_cache/PrefixCacheFingerprint.*`

### Tests

- Unit tests for MTP MoE collective-key uniqueness and namespace separation from main MoE layers.
- Unit tests for zero-row sparse dispatch/return no-op participation.
- Unit tests proving MoE placement fingerprint changes when owner map, masks, replicas, or active bank changes.
- Focused integration test with synthetic small MoE weights and ExpertParallel plan.
- Model-gated Qwen3.6 MoE test on available hardware for greedy MTP parity.
- Heterogeneous ExpertParallel smoke test for 2x ROCm plus 2x CPU dual-socket when the hardware is present.
- Chunk histogram merge test: two chunks with known routed experts match direct host-side counts and exclude padded tokens.
- Runtime-table bank flip capture/replay test: stable runtime-table pointer, active-bank flip between replays, output follows the new mask.
- Domain-local graph cache key test: same bucket length in different domains/participants must not alias.
- Segmented graph execution test with mock sparse collectives: captured segments replay, collectives run exactly once per chunk, continuation work waits for `collective_complete`.
- Negative test: incomplete/failed sparse collective prevents the next captured segment from launching.
- Forced chunk-boundary rebalance test: run chunk 0, rebalance, run chunk 1, compare against non-captured chunked reference with the same schedule.

### Exit Criteria

- MoE MTP sidecar no longer bypasses only because the MTP block contains MoE weights.
- ExpertParallel participants stay in lockstep through MTP sidecar sparse collectives.
- Prefix cache never reuses MoE/MTP state across incompatible expert placement.
- Tiered overlay prefill graph capture matches the non-captured overlay path for fixed and rebalanced schedules.
- Rebalance is deterministic for the same prompt, bucket policy, and interval.
- Sparse dispatch/return collectives preserve `collective_complete` ordering.
- Observability reports chunk id, bucket length, real-token range, domain id, participant id, placement epoch, capture/replay phase, and recapture reason.

## Phase 13: PyTorch Parity Acceptance Matrix

### Goal

Define the correctness gates required before claiming the plan is implemented end to end. Parity must cover prefix cache, MTP, TP collectives, hybrid state, and MoE sparse collectives.

### Implementation Details

Add a parity harness that records:

- Prompt tokens, suffix tokens, generated tokens, stop reason.
- Per-step greedy sampled token from Llaminar and PyTorch.
- Main logits for compared rows.
- MTP logits for draft rows when MTP is enabled.
- MTP acceptance/rejection trace.
- Prefix cache hit summary.
- MoE route indices and weights for compared MoE layers when deterministic enough for the model/precision.

Reference rules:

- Greedy mode only for acceptance.
- Fixed prompt fixtures and deterministic sampling.
- Compare exact generated tokens first.
- Compare logits with precision-specific tolerances. Quantized GGUF parity should use top-token equality plus bounded logit error rather than FP32-exact expectations.
- For prefix tests, PyTorch runs the full prompt while Llaminar may restore a prefix and run only the suffix. Final logits and generated tokens must match the no-cache Llaminar baseline and the PyTorch reference within tolerance.
- For MTP tests, PyTorch verifies the accepted token stream with the main model. MTP drafts may differ internally, but accepted output must equal greedy main-model output.

Required matrix:

| Topology | Model class | Prefix | MTP | Required parity |
|----------|-------------|--------|-----|-----------------|
| SingleDevice CPU | Dense Qwen3.6/Qwen3.5 fixture or small model | Off/RAM | Off | Baseline logits/tokens |
| SingleDevice CPU | Dense | RAM full/partial hit | Off | Suffix logits/tokens |
| SingleDevice CPU | Dense | RAM full/partial hit | Greedy | Accepted tokens equal main greedy |
| SingleDevice CUDA | Dense | RAM/device-hot where supported | Greedy | Same as CPU, model-gated |
| SingleDevice ROCm | Dense | RAM/device-hot where supported | Greedy | Same as CPU, model-gated |
| NodeLocalTP CPU | Dense | RAM | Greedy | Rank-wide common-prefix and MTP rollback |
| LocalTP ROCm | Dense | RAM/device-hot where supported | Greedy | LocalTP common-prefix and MTP rollback |
| SingleDevice CPU/ROCm | Hybrid Qwen3.5/Qwen3.6 | RAM | Greedy forced reject | GDN restore/replay |
| ExpertParallel CPU sockets | MoE Qwen3.6 | RAM | Off/Greedy | Placement fingerprint and sparse route correctness |
| ExpertParallel ROCm | MoE Qwen3.6 | RAM/device-hot where supported | Off/Greedy | Sparse collective parity |
| ExpertParallel ROCm+CPU | MoE Qwen3.6 | RAM | Greedy | Heterogeneous sparse sidecar parity |
| ExpertParallel ROCm+CPU graph-captured chunks | MoE Qwen3.6 | RAM | Off/Greedy | Captured fixed/rebalanced schedule equals non-captured chunked reference |

Large-model parity tests are normal parity-suite entries, not feature-gated opt-ins. They may skip only when a required model file, metadata fixture, MPI topology, or hardware backend is unavailable. Environment variables are path overrides, not enable switches:

```text
LLAMINAR_PARITY_DENSE_MODEL=/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf
LLAMINAR_PARITY_MOE_MODEL=/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
```

Small deterministic fixtures should remain in normal unit/focused integration coverage so CI can prove the contract without large model files. Large-model parity entries must still be present in the suite and report explicit prerequisite skips rather than requiring separate enable flags.

### Files

- `tests/v2/integration/parity/`
- `tests/v2/integration/parity/Test__PrefixCacheMTPParity.cpp`
- `tests/v2/integration/parity/Test__TPPrefixCacheMTPParity.cpp`
- `tests/v2/integration/parity/Test__MoEPrefixCacheMTPParity.cpp`
- Python reference helpers under the existing parity framework.
- `src/v2/testing/` helpers if a C++ parity adapter is needed.

### Tests

- Prefix disabled baseline parity.
- Prefix partial-hit parity.
- Prefix full-hit parity with terminal logits.
- Prefix full-hit plus MTP terminal-hidden restore parity.
- Forced MTP rejection parity.
- LocalTP and GlobalTP common-prefix parity.
- Hybrid GDN rollback parity.
- ExpertParallel MoE placement fingerprint and sparse-sidecar parity.
- ExpertParallel segmented graph-capture parity for fixed placement and forced chunk-boundary rebalance.

Focused command shape:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_|^V2_Integration_PrefixCache|^V2_Integration_Parity_PrefixCacheMTP" --output-on-failure --parallel
```

Large-model parity command shape:

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_.*Qwen36" --output-on-failure --parallel
```

### Exit Criteria

- All unit and focused integration parity tests pass, or skip only for explicit missing prerequisites such as model files, metadata, MPI rank count, or hardware.
- Large dense Qwen3.6 parity passes on each available SingleDevice backend requested for rollout.
- NodeLocalTP CPU and LocalTP ROCm dense parity pass before enabling TP MTP.
- ExpertParallel MoE parity passes before enabling MoE MTP.
- ExpertParallel segmented graph-capture parity passes before enabling graph-captured sparse overlay paths.
- Accepted token streams are identical to greedy main-model output for every tested MTP topology.

## Phase 13.5: Small-M GEMV-Many Kernel Prerequisite

### Goal

Make the verifier kernel path needed for Phase 14 MTP speedups a first-class backend contract instead of a benchmark-side tuning detail. Quantized GEMV must support small verifier batches `M in {2, 3, 4}` by quantizing activations once, running many projections, and staying graph-capturable on GPU.

This phase is a prerequisite to returning to Phase 14. Correctness-green MTP graph capture is already possible on some paths, but the current ROCm evidence shows that the captured verifier remains too expensive without deeper GEMV work.

This includes graph-stage fused paths, not only standalone GEMM calls. GDN projection stages, fused QKV, fused Gate/Up, fused-SwiGLU/FFN down, LM head, and MoE expert projection groups must route through the same small-M contract where they appear in verifier or sidecar execution.

### Implementation Details

Add or formalize a small-M GEMV-many path in the quantized GEMM interface:

```cpp
struct SmallMGemvProjection
{
    ITensorGemm *kernel = nullptr;
    TensorBase *output = nullptr;
    const TensorBase *bias = nullptr;
    int n = 0;
    const char *name = nullptr;
};

struct SmallMGemvManyRequest
{
    const TensorBase *input = nullptr;
    std::span<const SmallMGemvProjection> projections;
    int m = 0; // supported: 2, 3, 4
    int k = 0;
    DeviceWorkspaceManager *workspace = nullptr;
};
```

The exact API can reuse `multiply_fused_tensor()` if that remains the cleanest local shape, but the behavior must be explicit and testable:

- `M=2`, `M=3`, and `M=4` dispatch through a small-M GEMV route, not generic prefill GEMM.
- Activations are quantized once per input row block and shared across all projections.
- Fused QKV, fused Gate/Up, GDN QKV/Z/A/B projections, FFN down, LM head, and MoE expert gate/up/down shapes are covered where those projections appear in MTP verifier or sidecar paths.
- `GDNProjectionStage` must use the small-M route for its projection bundle in verifier/sidecar graph replay. If a GDN projection group is eligible for a native small-M batch, it must not silently fall back to per-projection prefill GEMM.
- GDN recurrence and short-conv kernels are not GEMV-many projections, but they are part of the same captured verifier budget. If stage timing shows they remain dominant after projection routing, they should receive adjacent graph-captured small-batch kernel work and first-class regression/perf coverage instead of being hidden under the GEMV exit criteria.
- Fused graph stages must preserve their graph-capture contracts while sharing activation quantization across the grouped projections. Batched small-M routes should expose counters by stage/projection group so real-model profiling can prove that the fused route was used.
- GPU paths are graph capturable: no hot-path allocation, no host synchronization, no pointer-shape mutation, no pageable host staging, and no fallback to uncaptured per-row streams during capture.
- Tensor movement and mapped-output staging use existing graph-stage contracts, `TransferEngine`, or preallocated workspace buffers. Do not mutate tensor placement/coherence flags directly to make a kernel path appear valid.
- Unsupported codebooks or dimensions fail loudly in focused tests and real inference instead of falling back to a misleading slower path.

Support the full quantized codebook inventory:

| Family | Codebooks |
|--------|-----------|
| Q-quants | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` where present |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` where present |
| IQ-quants | `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ1_S`, `IQ1_M` |

Backend work:

- ROCm first: extend `ROCmGemvKernel_native_VNNI.hip`, `ROCmGemvKernel_INT8_VNNI.hip`, and `ROCmQuantisedGemmKernel.cpp` so Qwen3.6 verifier shapes use graph-capturable M=2/3/4 GEMV-many for every supported codebook. The current M=2 Q4/Q4_K route and shared fused activation quantization are starting evidence, not the end state. The active ROCm sub-slice includes batched native-VNNI fused projection groups for QKV/GateUp/GDN so one quantized activation block can feed multiple same-codebook projections in a single graph-capturable launch family.
- CUDA second: generalize the existing native M=2 verifier route to M=3/4 and the full codebook set, preserving the graph-native speedup already observed for dense and MoE SingleDevice CUDA.
- CPU third: add equivalent small-M fused activation quantization and GEMV-many paths in `CPUNativeVNNIGemmKernel`/native codebook kernels so NodeLocalTP and CPU SingleDevice MTP do not rely on generic small prefill behavior.

Shape coverage should match inference, not only synthetic square cases:

- Dense Qwen3.6 27B projections: GDN projection dimensions, QKV, Gate/Up, FFN Down, and LM Head. Real-model diagnostics must report the route used for each of these buckets because GDN projection and fused projection groups have been observed as dominant verifier costs.
- MoE Qwen3.6 35B projections: router, routed expert Gate/Up/Down, shared expert paths, and LM Head.
- TP-sharded variants for LocalTP/NodeLocalTP where `N` is split.
- `M=2` for depth-1 MTP verifier, plus `M=3/4` for future deeper draft verification and grouped verifier replay.

Current active slice:

1. In progress: resume ROCm SingleDevice dense work.
2. Done for native-VNNI: expand ROCm small-M kernels from Q4-focused M=2 to all-codebook M=2/3/4.
3. Done for fused QKV/GateUp same-codebook groups: prove graph-captured fused projection paths with focused ROCm integration tests, including optional bias epilogue support and batched route counters.
4. Done for the first verifier-shaped perf expansion: extend ROCm perf rows to include Qwen3.6 hidden, GDN inner, GDN time, and GDN output projection shapes across all native Q/K/IQ codebooks and M=2/3/4.
5. Done for the shared-workspace crash slice: Qwen3.6-scale graph-captured GDN projection bundles can replay repeatedly while all projections share one `DeviceWorkspaceManager`, and focused counters prove the native batched route.
6. Done for the high split-K crash slice: real Qwen3.6 MTP reproduced an HSA fault at native small-M `KB=32`; the ROCm M=2/3/4 launchers now cap default graph-safe KB at 8 and hard-fail unsafe overrides, with focused GDN projection coverage.
7. Done for the Q4_0/Q4_1 M=2 specialization slice: single and batched ROCm native-VNNI paths have a dedicated graph-capturable two-row Q4 kernel. Real Qwen3.6 dense smoke improved only marginally from 13.96 to 14.02 tok/s, so this is regression coverage rather than Phase 14 speedup evidence.
8. Done for ROCm sidecar replay cleanup: removed the ROCm-specific MTP sidecar `force_recapture` override. `V2_Unit_MTPGraphConstruction` now requires `force_recapture=false`, and a real Qwen3.6 dense ROCm `-c 64 -n 8` smoke passed with sidecar shifted-prefill/decode replay enabled.
9. Done for the direct-prefill route diagnostic: `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison` shows direct native prefill GEMM is correctness-identical but 3.4x-7.7x slower than the current small-M route on hot Q4_1/Q5_1 FFN down and GDN output shapes at M=2/4, so ROCm should not route MTP verifier projections through generic prefill GEMM as a node-count shortcut.
10. Done for the fused-SwiGLU/FFN down route slice: ROCm `multiply_tensor_with_fused_swiglu()` now sends eligible native-VNNI `M=2/3/4` verifier shapes through the graph-native small-M path and hard-fails if that selected route cannot launch. Focused ROCm integration covers Q4_K M=2 and Q5_K M=4 route counters plus FP32-reference cosine, graph-captured Qwen3.6-scale Q4_K FFN-down M=2/M=4, and graph-captured Qwen3.6-scale GDN projection M=4. Real Qwen3.6 dense evidence is speed-positive at `-c 64 -n 16`: baseline 18.47 tok/s, MTP 25.51 tok/s, 87.5% acceptance.
11. Done for the stage-timed ROCm verifier diagnostic: after adding a timeline validity guard for MTP sidecar graph replay, the previous `LLAMINAR_GPU_STAGE_TIMING=1` HSA fault repro completed on real Qwen3.6 dense at `-c 64 -n 8`: 22.14 tok/s, 75% acceptance, `main_verifier` segmented replay about 56.15 ms per two-token graph, `mtp.verifier_forward` about 66.18 ms/call, and `main_verifier` stage GPU total about 37.87 ms. The top structured stage-GPU buckets are GDN projection (~15.73 ms/pass), generic GEMM (~8.99 ms/pass), and fused Gate/Up (~6.30 ms/pass).
12. Superseded by the retained mixed-codebook native small-M route: an earlier same-codebook subgroup batching experiment passed synthetic graph-capture tests but caused a real Qwen3.6 ROCm HSA memory access fault, so the temporary live path required an explicit `mixed_codebook` bypass. The lower-level mixed native-VNNI batched launcher now passes focused graph-capture tests and real Qwen3.6 ROCm smoke/parity, and `GraphCapturedFusedMixedCodebookGDNProjectionM4UsesMixedBatchedRoute` plus `GraphCapturedMixedCodebookQwen36GDNQkvZPairM4MatchesSeparate` require the mixed batched route instead of the old bypass. A current real Qwen3.6 ROCm `-c 64`, `-n 48` rerun reached 34.28 tok/s with 94.79% acceptance and no mixed-codebook bypass counters.
13. Done for the GDN qkv/z heterogeneous-N batching slice: Qwen3.6 GDN qkv/z `N={10240,6144}` pairs now use the generic native-VNNI small-M batched kernel instead of the unsafe Q4/M=2 specialized pair route. Focused ROCm integration captures and replays the exact shape, asserts batched projection counters, and real Qwen3.6 dense ROCm `-c 64 -n 8` completed at 22.35 tok/s with `main_verifier` GDN projection reduced from 95.90 ms to 91.65 ms total versus the earlier safe diagnostic.
14. Done for the FP32 GDN alpha/beta projection slice: ROCm FP32 fused projections now use hipBLAS batched SGEMM for same-`N` alpha/beta-style groups with device pointer arrays supplied by declared `IWorkspaceConsumer` buffers. The batched route hard-fails when workspace is missing instead of allocating ad hoc scratch, and the focused ROCm FP32 graph-capture regression binds a `DeviceWorkspaceManager`, captures/replays the batched call, and compares against CPU reference. A real Qwen3.6 dense ROCm `-c 64 -n 8` diagnostic confirmed the route active at `M=2/4`, `N=48`, and moved the short-prompt same-binary run to 24.70 tok/s MTP versus 21.70 tok/s baseline with 75% acceptance.
15. Done for the chained verifier decode-cache slice: `GraphCacheConfig::decode_seq_len` now defaults to 4, so MTP verifier continuations with `M=3/4` are decode-cache eligible instead of falling into prefill. `V2_Unit_ForwardExecutionEngineAdvanced` covers M=3 and M=4 all-position verifier cache reuse, `V2_Unit_ForwardGraphTypes` pins the default, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke` proves real Qwen3.6 ROCm GPU-graph `main_verifier` miss/hit counters for `seq_len=3`. Depth-2 graph-captured ROCm dense improved to 30.43 tok/s at `-c 64`, `-n 48`, but remains below the 33.57 tok/s depth-1 run because acceptance dropped to 87.10% and rollbacks rose to 20.
16. Done for the stage-timing stream-binding regression: ROCm `LLAMINAR_GPU_STAGE_TIMING=1` depth-1 MTP diagnostics faulted when timeline events were recorded on the worker default stream while graph-captured stages were rebound to a capture stream. `DeviceGraphExecutor` now records start/stop timing events on each stage's currently bound GPU stream, with `V2_Unit_GraphExecutorCollective.TimelineEventsUsePreBoundStageStream` covering the regression. A real Qwen3.6 dense ROCm `-c 64`, `-n 8` diagnostic completed without the old HSA fault and showed `main_verifier` remains the bottleneck: `GDN_PROJECTION` about 20.75 ms/call, ordinary `GEMM` about 14.71 ms/call, and `GEMM_FUSED_GATE_UP` about 11.62 ms/call, while `sidecar_forward` averaged about 3.47 ms/call.
17. Done for the graph-atomic K-partition slice: ROCm graph-enabled native-VNNI small-M no longer falls back to direct `kb=1` for graph safety. When `kb>1` under GPU graphs, the launcher zeroes the declared output on the graph stream, accumulates K partitions with in-kernel atomics, skips the unsafe split-reduce replay kernel, and records `path=atomic_reduce`. Focused ROCm integration covers graph-captured single and batched Qwen3.6-scale routes plus an explicit `KB=2` override, while retaining the hard fail for unsafe overrides above the graph-safe cap. A real Qwen3.6 dense ROCm graph smoke completed at 17.32 tok/s with 75% acceptance; this is a graph-safety/profiling step, not speedup evidence.
18. Done for the fresh long-lane A/B: current safe-path ROCm dense depth-1 MTP still reaches 33.25 tok/s versus 21.01 tok/s baseline at `The quick brown fox`, `-c 64`, `-n 48`, with 95.83% acceptance and fully captured `main_verifier`, `mtp_decode_sidecar`, and `mtp_decode_catchup` replay. The matching short stage diagnostic remains 17.30 tok/s because it has only 8 decode tokens, 75% acceptance, and heavy `stage_gpu,kernel` export.
19. Done for the reject-replay cleanup slice: depth-1 MTP rejects now replay only the accepted prefix token and append the correction token's shifted MTP row from that hidden state; deeper chained drafts now use the same lagged-correction shortcut when the first speculative token is rejected, and only the first post-sidecar checkpoint is captured because later chained sidecar rows are speculative scratch. Focused coverage includes `V2_Unit_PrefillDecodeTransition`, `V2_Unit_MTPGraphConstruction`, `V2_Unit_RankOrchestrator`, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`. Real ROCm depth-2 long-lane evidence confirms the counters move in the intended direction (`post_sidecar_checkpoint_skipped_speculative=51`, `lagged_rejected_correction_replays=6`, measured `replay_tokens` down from 36 to 30), but throughput did not improve because depth-2 acceptance stayed around 76-77% with 32-33 rollbacks.
20. Rejected for default behavior: a lazy-correction rollback shortcut was investigated for first-speculative-token rejects. The idea was to output only the already-restored prefix token, stash the verifier correction as the next step's ready first token, and avoid the current-step shifted-row catch-up. It was mathematically plausible, but real ROCm Qwen3.6 dense depth-2 evidence at `The quick brown fox`, `-c 64`, `-n 48` reached only 27.98 tok/s versus the 20.99 same-binary baseline and worsened exported counters relative to the retained lagged-replay path (`rejected_tokens` 12 to 15, `rollbacks` 15 to 18, `replay_tokens` 30 to 45 in the stats window). Do not reintroduce correction deferral as a default unless it also reduces verifier pressure and preserves the speed ratchet.
21. Done for live-prefix replay preservation: safe live-prefix restore/truncate no longer discards captured main/MTP replay state for stable dense or MoE placement. MoE placement epoch is now part of forward and MTP sidecar graph identity, so real placement changes miss and rebuild under the new epoch instead of paying blanket replay-state reset cost after every logical rewind.
22. Done for verifier-row GDN rollback: verifier forward captures recurrence and short-conv state rows, and reject rollback restores the accepted verifier row before falling back to checkpoint replay. Focused tests cover the state-row restore path and real Qwen3.6 ROCm chained graph smoke remains green.
23. Done for dense sidecar collective metadata caching: MTP depth-0 sidecar execution now discovers collective nodes once when the sidecar graph cache is built and reuses that cached set for capture-policy selection. Dense single-device MTP no longer rescans the sidecar graph on every token just because MoE/TP sidecars may contain collectives. `V2_Unit_MTPGraphConstruction` asserts one dense `sidecar_collective_node_scans` counter across two sidecar forwards, and the latest real Qwen3.6 ROCm depth-3 benchmark showed 224 measured decode-sidecar cache hits with zero measured sidecar graph cache misses after warmup reset.
24. Done for the first CPU M=2/3/4 all-codebook fused-projection slice: `CPUNativeVNNIGemmKernel::multiply_fused_tensor()` now records `kernel.cpu_native_vnni_small_m_fused_projection_calls` when the M>1 fused route quantizes activation rows once and runs multiple native-VNNI projections. `V2_Integration_CPUNativeVNNI_GEMV` now covers `M=2/3/4` fused-vs-separate projection parity across the full CPU native-VNNI codebook list and adds the previously missing `Q4_K`/`Q5_K` CPU smoke/sweep entries. The focused release perf binary `v2_perf_cpu_native_vnni_gemv --gtest_filter="*MTP_SmallM_FusedProjection_AllFormats"` now emits a first-class CPU MTP small-M fused projection row across all codebooks.
25. Done for the CUDA native-format shared-quant fused-projection slice: `CUDAQuantisedGemmKernel::multiply_fused_tensor()` now records `kernel.cuda_native_vnni_small_m_fused_projection_calls` when the small-M verifier route quantizes activation rows once and runs multiple native-VNNI projections. `V2_Integration_CUDAGemmParity` includes `MTP_SmallM_FusedProjection_AllNativeFormats`, covering `M=2/3/4` CUDA fused projection parity against CPU references across the CUDA native codebook inventory, plus `MTP_SmallM_MoEProjectionNamesStayRowwiseUntilBatchedRouteIsProven`, which guards MoE-style `shared_gate`/`shared_up` M=3/4 dispatch on the retained row-wise route until a batched CUDA route proves faster.
26. Done for CUDA GDN verifier-row restore parity: CUDA recurrence and short-conv kernels now capture verifier state rows into the same stage-declared verifier snapshot workspaces used by the graph system and can restore the accepted verifier row before KV truncate. `V2_Integration_CUDAGDNPaddedRealLength` covers recurrence and short-conv verifier-row snapshot/restore, while `V2_Unit_ForwardGraphTypes`, `V2_Unit_PrefillDecodeTransition`, and `V2_Unit_MTPGraphConstruction` stayed green for the dense resegment-skip and MTP graph-cache contracts.
27. Done for batched verifier-backed shifted-cache catchup: depth-0 MTP catchup now uses one multi-row `kv_cache_only` sidecar graph for contiguous verifier-backed shifted rows at `seq_len=2/3` instead of replaying one row at a time. Multi-row full sidecars hard-fail. `V2_Unit_MTPGraphConstruction` covers the allowed `kv_cache_only` graph and rejected full graph, and the real Qwen3.6 ROCm chained-draft smoke stayed green. The same-lane depth-3 benchmark preserved the speed ratchet at 54.69 tok/s and cut measured catchup replay from about 31.66 ms to 7.07 ms versus the prior ratchet artifact.
28. Done for legacy post-warmup resegment removal: segmented graph capture no longer exposes or consumes `requiresPostWarmupGraphSegmentRebuild()`, no longer stores `post_warmup_resegment_required`, and no longer emits `forward_graph.post_warmup_resegment{reason=not_required}` during dense warmup. MoE graph capture must use `supportsWarmupDependentGraphCapture()` and hard-fail at capture time if the warmed stage is still not capturable. This slice also fixed non-prefix `forwardMTP()` terminal-hidden buffer binding so sidecar graphs use `HIDDEN_STATE` when the terminal row comes from live hidden state instead of `PREFIX_TERMINAL_HIDDEN`, covering the clear-cache stale-buffer regression in `V2_Unit_MTPGraphConstruction`. Focused validation passed: `V2_Unit_ForwardGraphTypes`, `V2_Unit_MoERoutingStage`, `V2_Unit_PrefillGraphCapturability`, `V2_Unit_MTPGraphConstruction`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the Qwen3.6 ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. The real Qwen3.6 ROCm dense depth-3 lane reached 54.78 tok/s with zero `post_warmup_resegment` records.
29. Done for the sidecar/sample sync-fusion slice: GPU `DeviceGraphOrchestrator` now exposes runner-native MTP sidecar forward+greedy-sample calls. Dense sidecar segmented replay can defer its final capture-stream sync and enqueue argmax on the same stream, letting the argmax D2H sync close the operation boundary. `V2_Unit_PrefillDecodeTransition` covers the combined first/chained draft calls and proves the old separate MTP sample call is skipped. Focused validation passed with `V2_Unit_MTPGraphConstruction`, `V2_Unit_PrefillGraphCapturability`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the Qwen3.6 ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Current-code Qwen3.6 dense ROCm depth-3 evidence reached 54.32 tok/s, with a same-slice best of 54.99 tok/s; structured stats recorded 39 first-sidecar plus 78 chained-sidecar sync fusions, 117 `mtp_token_device_samples`, and no separate `sample_mtp_token_device` timer.
30. Done for ROCm asymmetric native-VNNI block-sum hardening: small-M verifier routes now compute quantized activation block sums during graph-capturable blockwise quantization into the declared `gemm_sums_a_blockwise` workspace, then feed those sums through plain, batched, mixed-codebook, fused-projection, and fused-SwiGLU/FFN-down paths. This fixes stale Q4_1/Q5_1 min-correction in the plain small-M route without ad-hoc allocation. Focused coverage passed: `V2_Unit_GemmWorkspaceConsumer`, `DispatchPlainAsymmetricNativeSmallMUsesFreshBlockSums`, `DispatchNativeSmallMAllCodebooksMatchReference`, `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison`, the Qwen3.6 ROCm parity cells `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`, plus `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`.
31. Done for ROCm verifier-row batched argmax: `IBackend` and `IInferenceRunner` now expose batched all-position verifier-row greedy sampling. ROCm implements the batched row path with one partial/finalize kernel pair over arena-declared argmax scratch; CUDA/default backends retain the row-loop interface fallback. `OrchestrationRunner` samples all verifier rows in one call and records `mtp.verifier_token_samples_batched`, while `DeviceGraphOrchestrator` records ROCm `mtp.verifier_token_device_batch_samples`. Focused coverage passed: `V2_Unit_PrefillDecodeTransition`, `V2_Integration_GPUSamplingKernels`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`, and the Qwen3.6 ROCm parity cells `MTPGreedyMatchesPyTorchDecodeTokens`, `MTPGreedyDepth3MatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Real Qwen3.6 dense ROCm depth-3 evidence reached 55.04 tok/s versus a 30.64 tok/s same-binary baseline at `The quick brown fox`, `-c 64`, `-n 48`, about 1.80x. The sampling timer is now about 70 us/call, but captured `main_verifier` replay still averages about 44.31 ms.
32. Done for bucketed ROCm decode-attention graph capture: attention stages now remain graph-capturable while reporting a split-bucket launch-topology variant. Segmented replay warms and recaptures before stale graph replay when the bucket changes, while within-bucket KV growth is handled by device-side dynamic params. Focused validation passed: `V2_Unit_AttentionComputeStage_DynamicKVLen`, `V2_Unit_ForwardGraphTypes`, `FlashDecode_FP32_GraphReplayUsesUpdatedKVLenWithinBucket`, the Qwen3.6 ROCm graph-stream stress parity test, dynamic-depth MTP parity, prefix+dynamic MTP restore parity, and the original dynamic-depth crash repro. The trace now shows `layer0_attention` as `[GRAPH]` instead of a manual fallback.
33. Done for ROCm attention device-param copy hardening: `AttentionDeviceParams` H2D upload now happens through `prepareDynamicAttnParams()` on an explicit non-null stream before HIP graph capture/replay, and the captured attention body hard-fails if pre-uploaded device params with the expected verifier row count are not already present. Cached forward graphs and MTP sidecar graphs bind worker/capture streams before dynamic-param updates. Focused validation passed: `V2_Unit_AttentionComputeStage_DynamicKVLen`, `V2_Unit_ForwardGraphTypes`, `V2_Unit_GraphExecutorCollective`, and `V2_Unit_ForwardExecutionEngineAdvanced`; a short real Qwen3.6 ROCm dynamic-MTP run completed with 268 MTP graph replay traces and no param-copy diagnostics.
34. Done for verifier-sized ROCm MoE route grouping: the async prefill grouping path now uses one explicit-stream graph-capturable HIP kernel for small verifier shapes, converting FP32 routing indices/weights directly into expert counts, offsets, grouped token ids, and grouped weights. The bridge hard-fails null streams. Focused coverage passed: `V2_Integration_ROCmMoEKernel`, direct `SmallFloatGrouping_*` regressions, `V2_Unit_Static_NoDefaultStreamInGPUCode`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke`, and isolated Qwen3.6 MoE ROCm PyTorch parity cells `MTPGreedyMatchesPyTorchDecodeTokens` and `PrefixCacheMTPRestore`. A short real Qwen3.6 MoE ROCm smoke recorded 440 `kernel.rocm_moe_small_prefill_grouping_calls`, proving the path is live; the remaining profile still points at main-verifier router/expert time as the next budget.
35. Done for the interim ROCm MoE small-M row-wise verifier routing slice: FP32 MoE router prefill for verifier-sized `M=2/3/4` bypassed the tiny hipBLAS GEMM path and launched the existing explicit-stream decode gate-logit kernel per row before the batched top-k path. Focused validation passed for `V2_Integration_ROCmMoEKernel`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, Qwen3.6 MoE ROCm `MTPGreedyMatchesPyTorchDecodeTokens`/`PrefixCacheMTPRestore`, full Qwen3.6 MoE ROCm deep math parity prefill/decode/snapshot tests, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke`. The real Qwen3.6 MoE ROCm fixed-depth-1 lane reached 39.10 tok/s versus the 19.72 baseline at `-c 768 -n 64`, about 1.98x, with 1760 row-wise-router calls and 1760 small-grouping calls recorded.
36. Done for ROCm MoE fused small-M verifier gate logits: verifier-sized FP32 MoE router prefill now computes all `M=2/3/4` gate-logit rows in one explicit-stream HIP launch before the existing batched top-k path, and the bridge hard-fails null streams. `V2_Integration_ROCmMoEKernel` now covers null-stream rejection, model-shaped fused-vs-single-token gate-logit parity, and `SmallMFusedRouter_VerifierShapeMatchesHostReference` route ids/weights against a CPU softmax/top-k reference. Focused validation passed for `V2_Integration_ROCmMoEKernel`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, Qwen3.6 MoE ROCm `MTPGreedyMatchesPyTorchDecodeTokens`/`PrefixCacheMTPRestore`, full Qwen3.6 MoE CUDA deep math parity prefill/decode/snapshot tests, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke`. The real Qwen3.6 MoE ROCm fixed-depth-1 lane reached 39.42 tok/s versus the 19.72 baseline at `-c 768 -n 64`, about 2.00x, with 1680 fused-router calls and 1680 small-grouping calls recorded in `benchmark_results/rocm_moe_mtp/20260604T025831Z-74a7043a-fused-smallm-router-n64`.
37. Done for ROCm MoE grouped/LDS small-M verifier gate logits: the same `hipMoE_gate_logits_small_m` bridge now uses the decode-router grouped pattern internally, loading each verifier hidden row into LDS once and assigning four waves to four experts per CTA. The route remains explicit-stream-only and keeps the same fused-router counter. Focused `V2_Integration_ROCmMoEKernel` regressions passed, including null-stream rejection, model-shaped fused-vs-single-token gate-logit parity, and CPU route-reference parity. Two fixed-depth Qwen3.6 MoE ROCm runs on the default lane beat the 39.42 tok/s ratchet, with the repeat reaching 39.94 tok/s versus the 19.72 baseline at `-c 768 -n 64`, about 2.03x, with 1680 fused-router calls and 1680 small-grouping calls in `benchmark_results/rocm_moe_mtp/20260604T030907Z-9da36b05-grouped-smallm-router-n64-repeat`.
38. Rejected for default behavior: token-deduplicated grouped-prefill activation quantization quantized only `seq_len` hidden rows and indexed those rows from grouped expert slots. Focused grouped-prefill correctness passed, but the real Qwen3.6 MoE ROCm fixed-depth-1 lane regressed to 38.81 tok/s versus the 39.94 grouped-router ratchet, with lower 74.22% acceptance and no reduction in average `MOE_EXPERT_FFN` pass time. The experiment was reverted; artifact `benchmark_results/rocm_moe_mtp/20260604T031900Z-d3bf0820-dedupe-prefill-quant-n64` should be used as the negative trace if this idea resurfaces.
39. Done for compact active-expert ROCm MoE grouped-prefill grids: verifier-sized grouping now emits an ascending active-expert list in declared ROCm MoE workspace, and grouped Gate/Up/Down prefill kernels map `blockIdx.z` through that compact list instead of launching all 256 expert CTAs for 16 verifier slots. The bridge and grouped-prefill pipeline hard-fail null or inconsistent streams/slots. Focused validation passed: `V2_Integration_ROCmMoEKernel`, direct `SmallFloatGrouping_*` regressions, `V2_Unit_Static_NoDefaultStreamInGPUCode`, `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke`, Qwen3.6 MoE ROCm `MTPGreedyMatchesPyTorchDecodeTokens`/`PrefixCacheMTPRestore`, full Qwen3.6 MoE ROCm deep math parity prefill/decode/snapshot tests, and full Qwen3.6 MoE CUDA deep math parity prefill/decode/snapshot tests. The real Qwen3.6 MoE ROCm fixed-depth-1 lane reached a 42.04 tok/s stable repeat versus the 19.72 baseline at `-c 768 -n 64`, about 2.13x, with active grouped-prefill grids using 16 active experts out of 256 in `benchmark_results/rocm_moe_mtp/20260604T034243Z-64e01724-active-expert-prefill-grid-n64-repeat`.
40. Rejected for default behavior: a row-batched small-M router that computed M=2/3/4 rows per expert block passed focused parity but regressed the fixed-depth-1 lane to 40.97 tok/s; a verifier-only grouped expert `TILE_M=4` specialization also passed focused tests but reached only 41.83 tok/s and did not improve `MOE_EXPERT_FFN` average time. Both experiments were reverted, while the M=2/3/4 small-router parity regression was retained.
41. Done for CUDA verifier-row batched argmax: CUDA now implements `IBackend::argmaxF32BatchedRows()` with one partial/finalize kernel pair over caller-owned scratch, matching the ROCm verifier-row sampling contract instead of using the default per-row backend loop. Focused validation passed: `V2_Integration_GPUSamplingKernels` and the CUDA Qwen3.6 dense/MoE SingleDevice prefix/MTP parity filter covering 12 model cells plus fixture. A short Qwen3.6 MoE CUDA fixed-depth-1 stats capture improved verifier sampling from 67.63 us/call to 36.98 us/call, while `mtp.verifier_forward` remained about 33 ms/call; the CUDA MoE default lane stayed correctness-green but performance-negative at 36.75 tok/s MTP versus 101.37 tok/s no-MTP.
42. Done for graph-capturable CUDA FP16KV small-M verifier attention: `AttentionComputeStage` now prepares one CUDA dynamic-attention param row per verifier row for causal FP16KV `M=2..4`, and `CUDAFlashAttentionKernelT<FP32>` dispatches those rows through the existing explicit-stream decode kernels using declared workspace buffers instead of the FA2 prefill path. The stage hard-fails missing explicit stream/workspace. Regression coverage passed: `V2_Integration_CUDAFlashAttentionParity` now includes workspace-size and CUDA graph-capture parity for `M=2`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, Qwen3.6 MoE CUDA deep math `DecodeParity`/`PrefillParity`, and the focused CUDA dense/MoE prefix/MTP parity filter. A short Qwen3.6 MoE CUDA fixed-depth-1 stats capture moved attention from 58.84 ms total before this fix to 6.66 ms total after it; the full `-c 768 -n 64` default lane improved from 36.75 tok/s to 41.92 tok/s at 32.03% acceptance, still below the 101.37 tok/s no-MTP baseline.
43. Rejected for default behavior: a CUDA small-M MoE router that staged each verifier row in shared memory and computed four experts per CTA passed focused correctness checks but regressed the short fixed-depth-1 Qwen3.6 MoE CUDA lane to 43.88 tok/s versus the 49.57 tok/s attention-only short run. The path fired 480 times, but `MOE_ROUTER` worsened from 5.18 to 5.54 ms/token and acceptance fell from 43.75% to 31.25%. The experiment was reverted; `V2_Integration_CUDAMoEKernel` now keeps the useful `RouteWithTensorsSingleTokenQwenScalePopulatesSnapshotOutputs` regression for the transient all-zero `layer40_moe_routing` dump. Focused validation passed: `V2_Integration_CUDAMoEKernel`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, Qwen3.6 MoE CUDA `PrefixCacheMTPRestore`, and full Qwen3.6 MoE CUDA deep math prefill/decode/snapshot parity.
44. Done for CUDA GDN qkv+z fused projection routing: `CUDAQuantisedGemmKernel` now advertises fused projection support, allowing `GDNProjectionStage` to route Qwen3.6 MoE verifier qkv+z as a native subgroup while alpha/beta remain FP32 single projections. The stage-level regression `Test__CUDAGemmParity.GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM` uses an explicit nonblocking CUDA stream and asserts both `kernel.gdn_projection_route` and `kernel.cuda_native_vnni_small_m_fused_projection_calls`. Focused validation passed for the nearby CUDA small-M fused projection tests, Qwen3.6 MoE CUDA `MTPGreedyMatchesPyTorchDecodeTokens`/`PrefixCacheMTPRestore`, and full Qwen3.6 MoE CUDA deep math prefill/decode parity. A short real Qwen3.6 MoE CUDA stats run recorded qkv+z native subgroups at M=1/2, and the default-lane fixed-depth-1 MTP run reached 64.72 tok/s versus the 101.37 tok/s no-MTP baseline at `-c 768 -n 64`, still performance-negative but above the prior 41.92 tok/s CUDA MoE ratchet.
45. Done for CUDA shared-expert grouped table decode: `CUDAMoEKernel` now implements the static descriptor-table gate/up and SwiGLU/down decode methods used by shared-expert FFN, backed by a persistent metadata cache that hard-fails on capture misses instead of staging host ids/weights inside a graph. `Test__CUDAMoEKernel.StaticGroupedDecodeDescriptorPathCapturesAfterWarmup` proves the path captures after explicit-stream warmup, and focused Qwen3.6 MoE CUDA deep math plus prefix/MTP parity passed with `LLAMINAR_ROCM_SHARED_EXPERT_GROUPED_DECODE=1`. The opt-in path improved the comparable GPU-stage n16 CUDA MoE MTP lane from 48.17 to 51.65 tok/s and reduced `MOE_SHARED_EXPERT_FFN` from 68.93 to 60.27 ms, but the comparable n64 run was 64.26 tok/s versus the retained 64.72 tok/s ratchet, so it stays opt-in.
46. Done for CUDA MoE north-star orientation and decode-route observability: llama.cpp was refreshed under `/tmp/llama.cpp` to `ggml-org/llama.cpp@6ddc943`, built with `GGML_CUDA=ON` for SM86, and benchmarked with `llama-bench -p 768 -n 64 -ngl 999 -r 3`. The RTX 3090 baseline is 41.82 tok/s decode for dense Qwen3.6 27B and 118.31 tok/s decode for Qwen3.6 35B A3B MoE. `CUDAMoEKernel` now emits structured `kernel.cuda_moe_grouped_decode_gateup_calls` and `kernel.cuda_moe_grouped_decode_down_calls` counters for runtime routed and static table decode paths, tagged by source, route, active slots, dimensions, codebook, and k-partitions. Focused coverage passed: `Test__CUDAMoEKernel.RuntimeGroupedDecodeDescriptorPathCapturesAfterWarmup`, `Test__CUDAMoEKernel.StaticGroupedDecodeDescriptorPathCapturesAfterWarmup`, and `V2_Unit_Static_NoDefaultStreamInGPUCode`.
47. Done for CUDA padded-prefill replay safety after north-star benchmarking exposed a dense Qwen3.6 CUDA crash: request-boundary `clear_cache()` now invalidates monolithic prefill graph entries with a structured `SessionReset` reason instead of replaying captured dynamic/kernel state after stage and kernel resets. The integration harness now binds declared `IWorkspaceConsumer` buffers for padded row-select probes and looks up bucketed forward caches with the same MoE placement epoch used by production. Focused validation passed: `V2_Unit_PrefillGraphCacheIntegration`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, and `V2_Integration_PrefillGraphCacheExecution_CUDA`. Clean CUDA no-MTP evidence after the fix: dense Qwen3.6 27B reached 671.87 tok/s prefill and 40.42 tok/s decode; MoE Qwen3.6 35B A3B reached 892.05 tok/s prefill and 103.84 tok/s decode in `benchmark_results/cuda_moe_mtp/20260604T083429Z-session-reset-prefill-invalidation-production-current`.
48. Done for CUDA MoE grouped-prefill small-M tile dispatch: CUDA grouped MoE prefill now auto-selects `TILE_M=2` or `TILE_M=4` for verifier-sized grouped expert batches and keeps larger prompt prefill on `TILE_M=16`, while explicit `LLAMINAR_CUDA_MOE_PREFILL_TILE_M` overrides remain available for diagnostics. `Test__CUDAMoEKernel.FixedTopologyRuntimeGroupedPrefillUsesCompactActiveExpertGrid` now asserts the verifier-sized tile tag, and the focused CUDA MoE capture regressions plus `V2_Unit_Static_NoDefaultStreamInGPUCode` passed. Qwen3.6 MoE CUDA deep math parity and all six CUDA SingleDevice prefix/MTP parity cells passed. The clean fixed-depth-1 default lane improved to 65.93 tok/s decode at 39.84% acceptance in `benchmark_results/cuda_moe_mtp/20260604T090428Z-smallm-prefill-tile-auto`, still performance-negative versus the 103.84 tok/s no-MTP CUDA baseline and the 118.31 tok/s llama.cpp north star.
49. Done for CUDA verifier-row shortcut restoration: CUDA now has direct recurrence and short-conv two-row GDN restore equivalence tests, plus a real Qwen3.6 MoE CUDA parity cell proving verifier row-0 shortcut restore and shifted-cache commit match a full two-token replay. The shortcut is enabled in release again with an explicit shifted-cache position override so suffix catch-up rows append at the replay checkpoint position. A release-only stale-token failure was fixed by requiring MoE expert decode to see an already-active runtime grouped-decode bank before consuming runtime route slots; a bank initialized by the expert stage is not eligible until the next pass because routing could not have populated it for the current pass. Focused validation passed for `V2_Unit_MoEForbiddenDependencyScan`, `V2_Unit_PrefillDecodeTransition`, `V2_Integration_CUDAGDNPaddedRealLength`, and the CUDA MoE parity cells `VerifierRowShortcutTwoRowStateMatchesFullReplay` and `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`. The release token-tag smoke corrected the stale-token sequence to `[271, 248046, 198, 248045]`, and the default lane improved to 83.10 tok/s fixed-depth-1 MTP with 75.78% acceptance versus a 102.03 tok/s no-MTP refresh in `benchmark_results/cuda_moe_mtp/20260604T131657Z-release-shortcut-fixed-default-n64`.
50. Done for CUDA MoE grouped-prefill production-path parity coverage: snapshot/integration builds can now execute fixed-topology grouped prefill eagerly even though they still report it non-capturable under `ENABLE_PIPELINE_SNAPSHOTS`. This fixes the root cause where the parity binary could not exercise the production fused CUDA verifier path. New Qwen3.6 MoE CUDA parity coverage, `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, resets structured counters, runs benchmark-style skip-gather MTP, and asserts the real model hits `swiglu_path=fused`, `tile_m=2`, `tile_n=64`, and `active_expert_slots=16`. Focused validation passed for `V2_Integration_CUDAMoEKernel`, the CUDA MoE prefix/MTP bundle including `NoMTPBenchmarkStyleSkipGatherGreedyMatchesGatheredArgmax`, `VerifierRowShortcutTwoRowStateMatchesFullReplay`, `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`, `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, `MTPGreedyDepth3MatchesBaselineTokens`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`, plus Qwen3.6 MoE CUDA deep math `PrefillParity` and `DecodeParity`.
51. Done for CUDA MoE verifier-grid route-shape hardening: `V2_Integration_CUDAMoEKernel.VerifierSmallMPrefillM234MatchesDecodeRowsAndCaptures` now covers production-realistic repeated experts across verifier rows while asserting top-k uniqueness within each row. This proves the fixed verifier-sized active-expert grid, including unused `-1` device slots, keeps fused grouped prefill equivalent to split prefill, rowwise decode, and CUDA graph replay for `M=2/3/4`. The deliberately impossible same-token duplicate route shape exposed divergent semantics and was not retained as a production requirement.
52. Done for CUDA MoE verifier split-K fused gate/up prefill: verifier-sized grouped MoE prefill now keeps the fused SwiGLU+quant production path while splitting the gate/up K dimension for compact active-expert M=2/3/4 grids. The bridge preallocates the split-K partial scratch before capture and hard-fails if a graph-captured verifier would require growth. `V2_Integration_CUDAMoEKernel.VerifierSmallMPrefillM234MatchesDecodeRowsAndCaptures` asserts `gateup_route=kpart_swiglu`, split-prefill equivalence, rowwise decode equivalence, repeated experts across rows, and CUDA graph replay for M=2/3/4. The focused Qwen3.6 MoE CUDA prefix/MTP parity bundle also passed, including `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, `VerifierRowShortcutTwoRowStateMatchesFullReplay`, `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`, `MTPGreedyDepth3MatchesBaselineTokens`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Clean release default-lane evidence improved fixed-depth-1 CUDA MoE MTP to 116.04 tok/s decode with 90.62% acceptance versus 101.62 tok/s baseline in `benchmark_results/cuda_moe_mtp/20260604T165605Z-kpart-verifier-gateup`, making CUDA MoE MTP speed-positive and close to the 118.31 tok/s llama.cpp MoE decode north star.
53. Done for CUDA MoE verifier replay preservation and split-K down prefill: verifier-row restore now preserves existing GPU decode/sidecar replay state instead of resetting captures, while still replaying the correction token because row-0 verifier state does not represent the post-correction state. CUDA grouped MoE verifier prefill now split-Ks the down projection for M=2/3/4 compact active grids, writing per-slot partials and reducing back into the existing deterministic scatter contract. Focused coverage passed: `V2_Integration_CUDAMoEKernel`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, and Qwen3.6 MoE CUDA parity cells `VerifierRowShortcutTwoRowStateMatchesFullReplay`, `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`, and `MTPGreedyMatchesPyTorchDecodeTokens`. Clean release evidence improved fixed-depth-1 CUDA MoE MTP to 119.65 tok/s decode with 82.81% acceptance versus 101.60 tok/s baseline in `benchmark_results/cuda_moe_mtp/20260604T172753Z-verifier-down-kpart-prefill`, beating the 118.31 tok/s llama.cpp MoE decode north star.
54. Done for CUDA shared-expert verifier grouped prefill: shared expert M=2/3/4 verifier rows now initialize an always-active single-expert grouped layout on the explicit CUDA stream and reuse the same split-K fused grouped prefill pipeline as routed experts. `V2_Integration_CUDAMoEKernel.SharedExpertVerifierSmallMPrefillM234MatchesDecodeRowsAndCaptures` proves M=2/3/4 equivalence to rowwise grouped decode and CUDA graph replay, while Qwen3.6 MoE CUDA `MTPBenchmarkStyleUsesFusedVerifierPrefillPath` asserts the production graph hits the shared `active_expert_slots=1`, `gateup_route=kpart_swiglu`, and `down_route=kpart_prefill` path. Clean release evidence improved fixed-depth-1 CUDA MoE MTP to 126.60 tok/s with 89.06% acceptance versus 101.18 tok/s baseline in `benchmark_results/cuda_moe_mtp/20260604T174640Z-shared-expert-prefill`, about 1.25x and above the 118.31 tok/s llama.cpp MoE decode north star.
55. Done for CUDA GDN alpha/beta FP32 batched projection: `CUDAFloatingPointGemmKernel` now advertises FP32 fused projection support and routes same-A alpha/beta verifier groups through cuBLAS batched SGEMM with device pointer arrays supplied by declared `IWorkspaceConsumer` buffers. `GDNProjectionStage` now labels qkv+z as the native subgroup and alpha+beta as the `same_kernel_mixed_codebook_subgroup`, instead of accidentally treating FP32 projections as native-compatible candidates or falling back to single projections. `V2_Integration_CUDAGemmParity.GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM` captures and replays the stage on an explicit CUDA stream, asserts the qkv+z native route, asserts the alpha+beta cuBLAS batched route, and compares alpha/beta outputs against CPU reference. Qwen3.6 MoE CUDA `MTPBenchmarkStyleUsesFusedVerifierPrefillPath` now asserts the production verifier records both fused GDN routes for M=2/K=2048 and no M=2 GDN fallback. Clean release validation reached 121.03 tok/s fixed-depth-1 MTP with 82.81% acceptance versus 103.40 tok/s baseline in `benchmark_results/cuda_moe_mtp/20260604T180556Z-gdn-alpha-beta-batched`; this preserves correctness and the fused path but does not replace the 126.60 tok/s best ratchet because acceptance was lower.
56. Done for CUDA MoE verifier correction replay capture hardening: correction replay now temporarily requests all-position logits and forces the CUDA MoE router, routed expert, and shared expert stages onto the grouped-prefill verifier path even for `seq_len=1`, preserving the split-K fused SwiGLU/quantized grouped-prefill route instead of falling back to decode. The MoE routing stage now separates grouped-prefill execution support from graph-capture support so integration/snapshot builds can exercise the production fused path eagerly while release builds can capture it. `CUDAMoEKernel::routeWithTensors()` now detects CUDA stream capture on the explicit stage stream and suppresses host top-k/logit materialization while capturing, fixing the root cause of the previous `cudaStreamSynchronize` capture failure. Focused validation passed: `V2_Integration_CUDAMoEKernel` now includes `RouteWithTensorsSingleTokenIsCudaGraphCapturableDeviceOnly`, `V2_Unit_PrefillGraphCapturability`, `V2_Unit_PrefillDecodeTransition`, `V2_Unit_Static_NoDefaultStreamInGPUCode`, and Qwen3.6 MoE CUDA parity cells `VerifierRowShortcutTwoRowStateMatchesFullReplay`, `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, and `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`. Clean release evidence reached 131.10 tok/s fixed-depth-1 MTP with 85.94% acceptance versus a 103.31 tok/s same-binary baseline in `benchmark_results/cuda_moe_mtp/20260604T190711Z-m1-correction-fused-routing-capture`, with no graph-capture, warmup-dependent, or stream-sync errors in the benchmark logs.
57. Done for CUDA MoE prompt-router prefill: FP32 MoE routing now uses an explicit-stream cuBLAS SGEMM path for `seq_len >= 16`, leaving the custom CUDA router for decode/small batches. `V2_Integration_CUDAMoEKernel.RouteWithTensorsCuBLASPrefillMatchesCPU` proves exact top-k index parity, close normalized weights/logits, and a structured `kernel.cuda_moe_router_cublas_prefill_calls` counter so the default path cannot silently regress; `RouteWithTensorsCuBLASPrefillCapturesAfterWarmup` proves the same route survives CUDA graph capture and replay after warmup. Focused validation passed for profiler/perf-counter coverage (`V2_Unit_ForwardGraphTypes`, `V2_Unit_StageTimeline`, `V2_Unit_PerfStatsCollector`), `V2_Integration_CUDAMoEKernel`, Qwen3.6 MoE CUDA deep math `PrefillParity` and `DecodeParity`, and the focused Qwen3.6 MoE CUDA prefix/MTP production bundle. Clean release evidence improved no-MTP prefill to 2465.72 tok/s and dynamic-MTP prefill to 1823.83 tok/s while preserving dynamic decode at 142.70 tok/s in `benchmark_results/cuda_moe_mtp/20260605T043011Z-router-cublas-ab`.
58. Done for the CUDA fused-runtime MoE down decode slice: the production fused runtime path now replaces split-K FP32 down partials plus a separate reduce with a deterministic warp-reduce down kernel that writes final routed outputs directly. `V2_Integration_CUDAMoEKernel` asserts the `fused_block_down` route under graph replay, deep Qwen3.6 MoE CUDA math parity and the focused CUDA MoE prefix/MTP parity cells stayed green, and clean release default-lane evidence in `benchmark_results/cuda_moe_mtp/20260605T050420Z-down-warp-reduce` improved no-MTP decode to 113.78 tok/s and fixed depth-1 MTP decode to 143.99 tok/s.
59. Done for CUDA MoE profiling trust: graph replay stage timing exports GPU-event `stage_gpu` rows while host replay bookkeeping stays in `forward_graph`, and the fused CUDA MoE runtime decode bridge now exports structured `kernel_cuda` sub-kernel timers for hidden quantize, gate/up k-part, SwiGLU quantize, and down warp-reduce when perf stats are enabled. The timers refuse to record while the stream is under graph capture, and `V2_Integration_CUDAMoEKernel.RuntimeGroupedDecodeFusedMatchesTwoStepAndGraphReplays` asserts both the timer rows and capture exclusion. The diagnostic export `benchmark_results/cuda_moe_mtp/20260605T065818Z-profile-fused-subkernels` shows routed down warp-reduce as the largest fused expert sub-kernel, followed by gate/up k-part.
60. Done for CUDA IQ4_NL decode in the shared native-VNNI helper: `decode_groups<4>()` and `decode_groups_vec<4>()` now reuse the packed `iq4nl_decode_word()` route instead of eight scalar table lookups per word, improving the hot Qwen3.6 MoE A3B fused down path without changing graph contracts. Focused validation passed for `V2_Integration_CUDAGemmParity`, `V2_Integration_CUDAMoEKernel`, `V2_Integration_Parity_Qwen36MoE_SingleDevice_Math_DecodeParity_CUDA`, and the CUDA MoE prefix/MTP production parity cells `MTPBenchmarkStyleDynamicDepthRequestStateResets`, `NoMTPBenchmarkStyleSkipGatherGreedyMatchesGatheredArgmax`, `VerifierRowShortcutTwoRowStateMatchesFullReplay`, `MTPGreedyMatchesPyTorchDecodeTokens`, `PrefixCacheMTPRestore`, `MTPBenchmarkStyleUsesFusedVerifierPrefillPath`, and `MTPBenchmarkStyleSkipGatherGreedyMatchesReference`. Clean release evidence in `benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode` reached 2707.70 prefill / 119.91 decode tok/s without MTP, 1946.82 prefill / 148.50 decode tok/s with fixed depth-1 MTP at 71.88% acceptance, and 1943.69 prefill / 145.36 decode tok/s with dynamic MTP at 68.75% acceptance.
61. Done for general GPU stage-profiling trust: `StageTimeline` now synchronizes the last timing event on each explicit stream before querying elapsed GPU event time, instead of assuming the last scheduled stage proves all streams complete. Structured perf stats can now request graph-captured stage timing through `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` or `LLAMINAR_PERF_STATS_FILTER=stage_gpu`, while captured graph replay rows remain clearly labeled as segment/total GPU-event timings rather than fake per-stage attribution. Focused validation passed for `V2_Unit_StageTimeline`, `V2_Unit_PerfStatsCollector`, and `V2_Unit_ForwardGraphTypes`.
62. Done for the first CUDA dense prompt-prefill selector slice: the native-VNNI prefill selector now applies a narrow Qwen3.6 Q4_K-family M-bin 512 override for FFN gate/up, FFN down, and GDN inner projection shapes based on the production-shape sweep. The perf harness now includes Qwen3.6 dense/GDN shapes plus Q4_K/Q5_K aliases, and `V2_Integration_CUDAGemmNonDeterminism.NativeVNNI_Qwen36DenseQ4KPromptPrefillUsesSweepTiles` locks the selected tile/split-K choices. Focused validation passed for the new dispatch regression, `V2_Integration_CUDAGemmParity`, and the full focused CUDA dense SingleDevice Qwen3.6 prefix/MTP parity suite. Clean release evidence in `benchmark_results/cuda_dense_mtp/20260605T075427Z-qwen36-q4k-prefill-tile-override` improved dense no-MTP prefill to 702.93 tok/s with 40.82 decode tok/s, fixed depth-1 MTP to 601.81 prefill / 56.13 decode tok/s at 96.88% acceptance, and dynamic MTP to 601.02 prefill / 55.23 decode tok/s at depth 1.
63. Done for CUDA dense NativeVNNI prefill route observability: `cudaNativeVNNIPrefill_fp32()` now emits `kernel.cuda_native_vnni_prefill_calls` counters tagged with codebook, M/N/K, tile id, split-K, BK256, and Stream-K selection, so structured JSON/CSV captures can attribute prompt-prefill routes without legacy profiling. `V2_Integration_CUDAGemmNonDeterminism.NativeVNNI_Qwen36DenseQ4KPromptPrefillUsesSweepTiles` now asserts the route counters for the Qwen3.6 FFN gate/up, FFN down, and GDN inner projection production shapes. A clean release diagnostic in `benchmark_results/cuda_dense_mtp/20260605T082625Z-qwen36-dense-kernel-route-stats` preserved the no-MTP throughput band at 698.33 prefill / 40.61 decode tok/s and recorded graph-bucket `M=600` top routes for Q4_K-family `17408x5120` tile 4, `5120x17408` tile 2, and GDN projection shapes.
64. Done for the CUDA dense GDN prompt-prefill selector slice: focused `M=600` tile sweeps for Q4_K and Q5_1 showed faster split-K winners for GDN Z/output/time, but the split-K 2/4 variants were rejected after the real Qwen3.6 27B 24GB lane OOMed during warmup from unplanned partial-buffer growth. The retained production override uses split-1 `T64x128_w4x2` for GDN Z/output on codebooks 5 and 7, preserving graph-captured memory headroom while improving no-MTP dense prefill to 709.61 tok/s. `V2_Integration_CUDAGemmNonDeterminism.NativeVNNI_Qwen36DenseQ4KPromptPrefillUsesSweepTiles` now pins the FFN, GDN inner, and GDN Z/output route counters across Q4_K/Q5_1; focused validation passed for `V2_Integration_CUDAGemmParity`, `V2_Integration_CUDAGemmNonDeterminism`, and all six `V2_Integration_Parity_Qwen36_CUDA_SingleDevice_` cases. Clean release evidence in `benchmark_results/cuda_dense_mtp/20260605T090657Z-qwen36-dense-gdn-split1-overrides` reached 709.61 prefill / 40.80 decode tok/s without MTP, 606.46 prefill / 54.32 decode tok/s for fixed depth-1 MTP at 96.88% acceptance, and 607.03 prefill / 56.15 decode tok/s for dynamic MTP at depth 1 and 96.88% acceptance.
65. Done for CUDA NativeVNNI workspace binding and allocation visibility: prefill split-K partials, prefill Stream-K fixup buffers, and decode/verifier GEMV KPAR partials now declare `IWorkspaceConsumer` requirements and bind through `DeviceWorkspaceManager` instead of allocating production hot-path scratch inside CUDA contexts. `DeviceWorkspaceManager` emits structured memory counters for workspace block/suballocation/release events, `V2_Integration_CUDAGemmNonDeterminism.NativeVNNIPrefillForcedSplitKDeclaresWorkspaceScratch` pins the exact-M versus padded-M planner fix that prevented small-M split-K launches from running without declared scratch, and CUDA GEMM parity asserts KPAR partials are required and sized by 32-wide K groups. Focused validation passed: `V2_Unit_DeviceWorkspaceManager`, `V2_Integration_CUDAGemmParity`, and `V2_Integration_CUDAGemmNonDeterminism`.
66. Next: shrink remaining CUDA dense gaps and continue hardening CUDA MoE controller/long-prompt evidence without reintroducing rejected router/projection or memory-bloating split-K experiments. CUDA MoE now beats the current llama.cpp MoE default-lane no-MTP and MTP decode anchors. ROCm MoE keeps the current 2.13x fixed-depth ratchet. After the verifier budget approaches the target, repeat longer-prompt dense/MoE evidence and run CPU real-model MTP benchmarks before returning to Phase 14 acceptance.

### Files

- `src/v2/tensors/TensorKernels.h`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h/.cpp`
- `src/v2/kernels/rocm/gemm/HipBLASGemmKernel.h/.cpp`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip`
- `src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp`
- `src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel_CUTLASS.cu`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h`
- `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/FusedGateUpGEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/FusedSwiGLUStage.cpp`
- `src/v2/execution/compute_stages/stages/GDNProjectionStage.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmSmallM.cpp`
- `tests/v2/integration/kernels/cuda/Test__CUDAGemmParity.cpp`
- `tests/v2/unit/kernels/` CPU small-M regression tests.
- `tests/v2/performance/kernels/rocm/`
- `tests/v2/performance/kernels/cuda/gemm/`
- `tests/v2/performance/kernels/cpu/native_vnni/`

### Tests

- Unit or integration correctness tests for every supported codebook at `M=2`, `M=3`, and `M=4`.
- Fused GEMV-many versus separate projection parity for QKV, Gate/Up, GDN QKV/Z/A/B, FFN Down, LM Head, and MoE expert projection shapes.
- Route regression tests must assert grouped GDN/fused projection counters for eligible small-M verifier shapes and assert hard-fail or explicit bypass counters for unsupported shapes.
- GPU graph-capture tests for ROCm and CUDA small-M fused paths: capture, instantiate, replay, compare to uncaptured output, and assert no manual fallback path was used.
- Regression tests for every crash or wrong-output issue found while running real-model smokes.
- ROCm Phase 13.5 perf must include `NativeVNNIGEMMPerfTest.MTP_SmallM_VerifierShapes_AllFormats` and `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison` so all-codebook small-M coverage and rejected prefill-route evidence remain reproducible.
- Perf tests that run as first-class `Perf__` coverage, not ad hoc benchmark notes:

```bash
cmake --build build_v2_release --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_.*(ROCm|CUDA|CPU).*QuantisedGemm.*SmallM|^V2_Perf_.*NativeVNNI.*SmallM" --output-on-failure --parallel
```

Focused correctness command shape:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_ROCmQuantisedGemmSmallM|^V2_Integration_CUDAGemmParity|^V2_Unit_.*NativeVNNI.*SmallM" --output-on-failure --parallel
```

### Exit Criteria

- ROCm, CUDA, and CPU expose a deliberate small-M GEMV-many path for `M=2/3/4`.
- Every supported Q/K/IQ codebook has correctness coverage for `M=2/3/4`, or an explicit hard-fail test documents why the codebook is not present for that backend.
- ROCm and CUDA small-M fused paths are graph-capturable and replay correctly.
- Real-model stats prove GDN projection, fused QKV/GateUp, FFN down, LM head, and MoE projection groups either use the small-M route or report a reasoned hard fail/bypass that is tracked as remaining work.
- Perf tests report latency for inference-shaped projections and prove whether each backend is using the intended small-M route.
- Phase 14 MTP benchmark work resumes only after the backend required for the next benchmark row has passed this phase's correctness and perf gates.

## Phase 13.7: Stochastic MTP Verification And GPU Sampling

### Goal

Make MTP compatible with production chat sampling defaults instead of treating non-greedy sampling as a bypass. Qwen chat defaults such as `temperature=0.6`, `top_p=0.95`, `top_k=20`, and presence penalty must run through exact speculative verification semantics, with graph-capturable GPU sampling primitives and structured acceptance telemetry.

### Implementation Details

Use the standard speculative sampling contract:

1. The main model samples the first token normally from distribution `p_0`.
2. The MTP sidecar samples draft tokens `x_i` from sidecar distributions `q_i`.
3. The verifier computes target distributions `p_i` for every drafted position.
4. For each draft token, accept with probability `min(1, p_i(x_i) / q_i(x_i))`.
5. On rejection, sample the correction token from the residual distribution `max(p_i - q_i, 0)`.
6. If every speculative token is accepted, sample the ready terminal token from the final verifier row distribution `p_{k+1}` and cache it as the next ready token.

Distribution rules:

- Apply the same additive penalties, temperature, top-k, and top-p filtering to both draft and target rows before computing probabilities.
- Penalty history for a verifier row must reflect the exact optimistic prefix consumed by that row.
- Store each draft row's sidecar logits, or an equivalent normalized distribution, until verifier acceptance finishes. Recomputing sidecar logits during rollback is not acceptable on the hot path.
- Greedy mode remains the existing exact-token comparison fast path.
- Stochastic mode must be enabled only through `MTPVerifyMode::SpeculativeSampling` and only when the runner exposes the required distribution snapshot/sampling hooks.

GPU requirements:

- Add graph-capturable device enqueue APIs for top-k/top-p/temperature sampling and speculative accept/residual verification.
- Represent draft and target stochastic rows on GPU as compact `(token_id, probability)` tables when only the filtered top-k/top-p distribution is needed.
- Device accept/residual verification consumes those compact tables, writes only the selected token, accept flag, accept probability, and threshold to scalar device buffers, and never copies full logits to host on the hot path.
- Use explicit non-null streams for every GPU sampling, penalty, probability, random, and row-copy operation.
- Keep device-resident row buffers in declared workspace or graph-owned extension buffers; do not allocate inside captured replay.
- Runtime wrappers may copy only the selected token, accept flag, and debug scalar summaries back to host.
- Cross-device TP/MoE support must coordinate the same random draws and selected tokens across participants; single-device GPU lands first.

Current implementation status, 2026-06-03:

- `src/v2/kernels/common/SamplingMath.h` now provides the shared CPU/CUDA/ROCm implementation of compact top-k/top-p normalization, splitmix random thresholds, compact distribution sampling, and speculative accept/residual verification math. CPU `Sampler`, CUDA sampling kernels, and ROCm sampling kernels all call this helper for the compact stochastic path so CPU/GPU parity tests exercise one mathematical contract instead of three hand-copied implementations.
- Host exactness is implemented for SingleDevice full-logit stochastic MTP and has unit coverage for accept, reject, residual sampling, and terminal ready-token sampling.
- `V2_Unit_Sampler` now exposes and tests `speculative_accept_probability()` and `residual_distribution()`, including a deterministic proof that the accept/reject/residual algorithm reconstructs the target distribution exactly from draft distribution `q` and verifier distribution `p`, plus compact-path checks that `Sampler::compute_distribution()` and speculative verification match `SamplingMath`.
- CUDA and ROCm have explicit-stream graph-capturable top-k/top-p sampling, compact distribution construction, and speculative accept/residual verifier enqueue APIs.
- `GPUSamplingTest.TopKTopPDistributionMatchesCPUSampler` proves CUDA and ROCm compact distribution tables match the CPU `Sampler` path, while `GPUSamplingTest.SpeculativeVerifyDistributionsAreGraphCapturable` proves both backends can capture distribution build plus accept and residual-reject verification, and that the new APIs reject null/default streams.
- Runner integration now requires compact device-resident stochastic verifier hooks on SingleDevice GPU paths. The first token, each MTP draft token, verifier accept/residual decisions, and the all-accepted terminal ready token use compact device distribution buffers rather than host full-logit snapshots.
- `V2_Unit_PrefillDecodeTransition` covers the GPU hard-fail when those hooks are missing and the supported device-resident verifier path when they are present, including active presence-penalty sampling parameters. `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke` and `V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsStochasticSmoke` pass with active presence penalty, proving the real Qwen3.6 stochastic MTP smokes still enter draft/verify and asserting structured request-summary counters plus device stochastic counters are present while host full-logit stochastic counters stay at zero. The normal Qwen3.6 ROCm and CUDA dense parity suites also include graph-captured stochastic verifier cells that run no-MTP stochastic baselines and assert speculative-MTP verifier counter consistency.
- `PrefixRuntimeStateSnapshot`, `MTPRequestSummary`, benchmark JSON, and serve summaries now expose verify mode plus stochastic accept-test, accepted-test, residual-sample, terminal-sample, and stochastic acceptance-rate fields for Phase 14 evidence.
- The current correctness contract is distribution equivalence. Same-seed token-stream equality between MTP and non-MTP stochastic decode is not asserted because the speculative path consumes draft, accept, residual, and terminal random draws that the non-MTP path does not.

Observability:

- Stochastic draft samples, verifier accept tests, residual samples, all-accepted terminal samples, and random-threshold details are emitted through profiling counters.
- Request summaries expose verify mode, stochastic acceptance rate, residual-sample count, and exactness bypass/failure reasons.

### Files

- `src/v2/utils/Sampler.h/.cpp`
- `src/v2/backends/IBackend.h`
- `src/v2/backends/cuda/CUDABackend.*`
- `src/v2/backends/rocm/ROCmBackend.*`
- `src/v2/kernels/common/SamplingMath.h`
- `src/v2/kernels/cuda/ops/CUDASamplingKernels.cu`
- `src/v2/kernels/rocm/ops/ROCmSamplingKernels.hip`
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.*`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.*`
- `src/v2/execution/runner/OrchestrationRunner.*`
- `tests/v2/unit/execution/runner/Test__PrefillDecodeTransition.cpp`
- `tests/v2/integration/backends/Test__GPUSamplingKernels.cpp`
- `tests/v2/integration/parity/qwen36/`

### Tests

- Unit tests for exact accept/reject math with scripted `p`, `q`, and random thresholds.
- Unit tests proving the CPU compact top-k/top-p and speculative verification paths match the shared `SamplingMath` contract.
- Unit tests for residual correction sampling when `p_i(x_i) < q_i(x_i)`.
- Unit tests proving non-greedy MTP no longer records a bypass when `verify_mode=SpeculativeSampling` and required hooks are present.
- GPU integration tests proving top-k/top-p/temperature sampling, compact distribution construction, and speculative verification enqueue forms are graph-capturable on CUDA and ROCm.
- GPU integration tests proving compact CUDA/ROCm distributions match the CPU `Sampler` distribution for the same logits and sampling params.
- Focused real-model stochastic coverage: normal parity-suite GPU cells must run no-MTP stochastic baselines and graph-captured stochastic MTP verifier checks with active Qwen chat-like sampling parameters; exact distribution preservation remains pinned by the unit-level speculative sampling proof unless a future position-stable RNG contract is added.
- Regression tests for Qwen chat defaults so presence penalty plus `top_k=20`/`top_p=0.95` does not fall back to full-logits CPU sampling on supported single-device GPU paths.

### Exit Criteria

- Greedy MTP behavior and parity remain unchanged.
- Non-greedy MTP uses exact speculative sampling semantics for SingleDevice CUDA and ROCm.
- CPU has either exact host-side stochastic MTP coverage or a hard-fail gate that prevents claiming CPU stochastic MTP support.
- No stochastic GPU sampling path uses a null/default stream or allocates during graph capture.
- Qwen chat defaults can run MTP without the old `sampling is not greedy` bypass on supported single-device GPU backends.
- Phase 14 benchmarks can report stochastic MTP speedups separately from greedy MTP speedups.

## Phase 14: Benchmark Acceptance And Default-Enablement Readiness

### Goal

Measure whether prefix cache and MTP provide real speedups on the supported correctness matrix, after the Phase 13.5 small-M GEMV-many prerequisite and Phase 13.7 stochastic verifier prerequisite have landed for the backend under test, and define the evidence needed before any future default enablement.

### Implementation Details

Add benchmark scenarios that emit machine-readable JSON plus human summaries:

- Initial smoke path: `llaminar2 benchmark ... --benchmark-json-output <path>` writes schema `llaminar.benchmark.v1` with timing, throughput, prefix-cache, MTP, and prefill-chunk counters.

- Prefix disabled baseline.
- RAM-only prefix cache miss and hit.
- RAM plus device-hot tier hit.
- Disk warm/hydrated path.
- Prefix partial-hit with long shared prompt and short suffix.
- Prefix full-hit with terminal logits.
- MTP-only greedy decode.
- Prefix plus MTP greedy decode.
- Hybrid Qwen3.5/Qwen3.6 prefix cache.
- Dense LocalTP/GlobalTP prefix and MTP.
- MoE Qwen3.6 ExpertParallel prefix and MTP.
- ExpertParallel overlay prefill graph capture with fixed placement.
- ExpertParallel overlay prefill graph capture with forced chunk-boundary rebalance.

Metrics:

- Prefill wall time and tokens/sec.
- Decode wall time and tokens/sec.
- Prefix lookup time, populate time, harvest time, disk hydration time, and device promotion time.
- Matched blocks/tokens and hit tier.
- MTP draft steps, verifier runs, accepted tokens, rejected tokens, rollbacks, and acceptance rate.
- Optional per-step MTP acceptance trace with condition token, first main token, sidecar draft token, verifier token, output-token count, and rollback/commit outcome.
- Sparse MoE dispatch/return bytes and dense bytes avoided.
- Graph chunk capture time, replay time, recapture count, and ineligible/fail-fast reason.
- Per-chunk histogram merge time, rebalance decision time, placement epoch flip time, and expert payload transfer/prepare time.
- Peak RAM, device-hot bytes, and disk bytes.
- Disabled-feature overhead compared with baseline.

Add an adaptive MTP depth policy after fixed-depth correctness is green:

```cpp
enum class MTPDepthPolicyMode
{
    Fixed,
    Observe,
    Dynamic,
};

enum class MTPDepthDecisionReason
{
    FixedMode,
    WindowNotReady,
    CooldownActive,
    PromoteFullAcceptRate,
    DemoteZeroAcceptRate,
    DemoteLowAcceptanceRate,
    Hold,
};

struct MTPDepthPolicyConfig
{
    MTPDepthPolicyMode mode = MTPDepthPolicyMode::Fixed;
    int min_depth = 1;
    int max_depth = 1;      // defaults to --mtp-draft-tokens
    int initial_depth = 1;  // defaults to max_depth
    int window_size = 16;   // verifier decisions, not raw tokens
    int min_samples = 4;
    int cooldown_steps = 8;
    int promote_consecutive_windows = 3;
    double promote_full_accept_rate = 1.0;
    double demote_zero_accept_rate = 0.30;
    double demote_acceptance_rate = 0.55;
};

struct MTPDepthWindow
{
    uint64_t verifier_runs = 0;
    uint64_t attempted_draft_tokens = 0;
    uint64_t accepted_draft_tokens = 0;
    uint64_t rejected_draft_tokens = 0;
    uint64_t rollbacks = 0;
    uint64_t full_accepts = 0;
    uint64_t zero_accepts = 0;
    uint64_t accepted_prefix_sum = 0;
};
```

Policy rules:

- `--mtp-draft-tokens` remains exact depth in `Fixed` mode. In `Dynamic` and `Observe` modes it becomes `max_depth`.
- The controller changes depth only at decode-step boundaries, after rollback/commit state is complete.
- The first implementation is SingleDevice. LocalTP, NodeLocalTP/GlobalTP, LocalPP, and ExpertParallel require a domain-common depth decision before enabling dynamic mode. Participant-local depth changes are a hard failure.
- `Observe` mode records the same windows and recommended decision but never changes the effective depth.
- `Dynamic` mode changes depth by at most one level per decision window and uses cooldown/hysteresis to prevent oscillation.
- Demote when a ready window has high zero-accept rate, high rollback/rejection pressure, or low accepted-draft-token rate for the current depth.
- Promote when the current depth's full-accept rate is high, zero-accept rate is low, the cooldown has elapsed, and the current depth is below `max_depth`.
- Budget clamping remains separate: the controller's chosen depth is the requested depth, and the per-step token budget may still clamp the effective depth without teaching the policy that the prompt rejected tokens.
- Graph caches must be keyed by effective verifier length and sidecar mode so changing depth reuses or builds the correct captured graph without reconstructing unrelated graph state.
- The policy must never bypass MTP correctness checks. It only chooses how many sidecar tokens to propose; the main verifier remains the source of accepted output.
- Multi-participant support uses the same coordination style as Phase 10/11: each participant contributes window counters, the domain computes one decision, then all participants apply the same next depth before the next sidecar step.
- MTP depth decisions emit structured `PerfStatsCollector` counters with tags for old depth, new depth, reason, acceptance rate, zero-accept rate, full-accept rate, and window size.

Suggested CLI/config additions:

- `--mtp-depth-policy fixed|observe|dynamic`
- `--mtp-min-draft-tokens <n>`
- `--mtp-depth-window <n>`
- `--mtp-depth-min-samples <n>`
- `--mtp-depth-cooldown <n>`
- `--mtp-depth-promote-windows <n>`
- `--mtp-depth-promote-full-accept <f>`
- `--mtp-depth-demote-zero-accept <f>`
- `--mtp-depth-demote-acceptance <f>`

Initial ROCm dense evidence motivating the policy:

| Benchmark lane | Fixed depth | Decode | Speedup vs baseline | Acceptance | Full-depth accepts | Zero accepts |
|----------------|-------------|-------:|--------------------:|-----------:|-------------------:|-------------:|
| Default prompt, 128 decode | 1 | 39.72 tok/s | 1.33x | 73.44% | 136 | 56 |
| Default prompt, 128 decode | 2 | 39.42 tok/s | 1.32x | 65.84% | 56 | 40 |
| Default prompt, 128 decode | 3 | 34.96 tok/s | 1.17x | 61.99% | 23 | 44 |
| Long lane, `The quick brown fox` | 3 | 53.81 tok/s | 1.75x | 86.33% | high enough to retain depth 3 | low |

This suggests the controller should demote the default prompt from depth 3 toward depth 1/2, while preserving the long-lane depth-3 ratchet when the recent acceptance window supports it.

Acceptance targets before considering default enablement:

- Prefix disabled overhead is within noise of baseline, target less than 2% median regression.
- RAM prefix hit shows a real prefill speedup on long shared prompts, with matched-token and timing counters proving prefill was skipped rather than hidden by measurement noise.
- Device-hot tier is faster than RAM hydrate for GPU restore or is documented as not worth enabling for that backend.
- Disk warm path improves repeated process startup or cross-process reuse workloads, or remains opt-in only.
- MTP decode benchmarks must identify the Phase 13.5 small-M GEMV-many route used for the verifier backend/codebook mix.
- Adaptive depth must not reduce correctness coverage: dynamic-depth accepted token streams must match fixed-depth greedy verifier output and the MTP-disabled greedy baseline for each parity lane.
- On acceptance-limited prompts, adaptive depth should avoid known-regressive fixed depths and stay within noise of the best fixed depth observed for that lane.
- On high-acceptance prompts, adaptive depth should promote back to the fastest safe depth and preserve the existing speedup ratchet.
- Dense MTP target is approximately 2x decode throughput versus disabled on Qwen3.6 27B for the supported GPU backend.
- MoE MTP target is approximately 1.5x decode throughput versus disabled on Qwen3.6 35B MoE for the supported ExpertParallel topology.
- Prefix plus MTP must not regress versus the faster of prefix-only and MTP-only for the benchmarked prompt class without an explicit documented reason.
- ExpertParallel graph-captured chunks should reduce host dispatch overhead inside domain-local compute segments. Sparse collective and rebalance overhead must be reported separately from graph replay time.
- Chunk-boundary rebalance should improve or preserve end-to-end long-context MoE throughput for skewed routing prompts; if it does not, the benchmark must expose whether placement transfer, recapture, or sparse collective overhead dominated.

Large-model benchmark inputs:

```text
/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf
/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
```

Required benchmark topology coverage, when hardware is present:

- SingleDevice CUDA dense.
- SingleDevice ROCm dense.
- SingleDevice CPU dense.
- NodeLocalTP CPU dense.
- LocalTP ROCm dense.
- ExpertParallel 2x socket CPU MoE.
- ExpertParallel 2x ROCm MoE.
- ExpertParallel 2x ROCm plus 2x CPU dual-socket MoE.

### Files

- `src/v2/utils/BenchmarkRunner.cpp`
- `src/v2/execution/mtp/MTPDepthController.h/.cpp`
- `src/v2/execution/config/RuntimeConfig.h`
- `src/v2/config/OrchestrationConfigParser.cpp`
- `src/v2/app/` benchmark CLI plumbing if new flags are needed.
- `tests/v2/performance/`
- `docs/v2/PREFIX_CACHE_MTP_BENCHMARK_NOTES.md`

### Tests

- Unit tests for fixed/observe/dynamic depth decisions, promotion, demotion, cooldown, min/max clamps, budget-clamp separation, and controller lifecycle across `prefill()`/`clearCache()`/disabled-MTP transitions.
- Benchmark smoke tests with tiny fixtures and no large model requirement.
- JSON schema test for benchmark output.
- Disabled-feature overhead smoke test.
- Focused SingleDevice integration test with scripted MTP accept-prefix outcomes proving dynamic depth changes without changing emitted greedy tokens.
- Qwen3.6 ROCm parity cell for dynamic-depth MTP greedy decode and prefix-cache restore.
- Future LocalTP/NodeLocalTP tests proving all participants apply the same domain-common depth decision.
- Model-gated dense Qwen3.6 benchmark.
- Model-gated MoE Qwen3.6 ExpertParallel benchmark.

Command shape:

```bash
cmake --build build_v2_release --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_PrefixCacheMTP" --output-on-failure --parallel
```

Large-model opt-in:

```bash
LLAMINAR_ENABLE_LARGE_MODEL_BENCHMARKS=1 \
ctest --test-dir build_v2_release -R "^V2_Perf_.*Qwen36" --output-on-failure --parallel
```

### Exit Criteria

- Benchmark JSON contains enough counters to explain every reported speedup.
- Dense and MoE target speedups are met or blockers are documented with traces.
- Adaptive depth decisions are visible in stats and benchmark JSON, and dynamic mode preserves greedy parity.
- Graph replay, sparse collective, histogram merge, rebalance, and recapture timings are separated in output.
- No feature is considered for default enablement until its matching parity phase and benchmark phase have passed.

## Recommended First Implementation Slice

Start with the smallest slice that proves the contract without risking GPU/hybrid complexity:

1. Phase 1 config structs and CLI parsing, disabled by default.
2. Phase 2 `IKVCache` logical block API with default false implementations.
3. CPU `CPURingKVCache` logical export/import/truncate for FP32 and the Q8/Q16 formats already used in unit tests.
4. Phase 3 `PrefixStateCache`, hash-chain keys, RAM backend, and LRU without runner integration.
5. Dense CPU unit tests for export/import and RAM cache.
6. Phase 0 probe scaffolding that records state inventory even before full restore support exists.

Then proceed in this order:

1. Dense single-runner RAM prefix cache.
2. CUDA/ROCm logical I/O and shadow invalidation.
3. Hybrid GDN state import/export.
4. Persistent MTP shifted-KV payload through prefix harvest/populate.
5. MTP sidecar loading and shifted prefill cache.
6. SingleDevice MTP greedy verification and rollback.
7. Phase 8 MoE placement fingerprint, domain-scoped rebalance controller, and rebalance invalidation.
8. Device-hot and disk tiers.
9. Phase 9 observability summaries and bypass counters.
10. Phase 10 common-prefix coordination for LocalTP, GlobalTP/NodeLocalTP, PP, and MoE continuation domains.
11. Phase 11 TP-compatible dense MTP sidecar.
12. Phase 12 ExpertParallel sparse-MoE MTP sidecar plus segmented overlay graph capture.
13. Phase 13 PyTorch parity matrix.
14. Phase 13.5 small-M GEMV-many kernel prerequisite, ROCm first, then CUDA and CPU.
15. Phase 13.7 stochastic MTP verification and GPU sampling.
16. Phase 14 benchmark acceptance matrix.
