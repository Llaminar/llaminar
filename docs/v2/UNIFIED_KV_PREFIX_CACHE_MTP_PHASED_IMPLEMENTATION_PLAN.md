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
| Phase 12: ExpertParallel Sparse MoE MTP Sidecar And Segmented Graph Capture | Complete | Focused Phase 12 slice passed on 2026-06-01: `V2_Unit_MTPGraphConstruction` proves graph-native overlay MoE MTP sidecar execution appends real shifted KV payload and marks sparse manual boundaries correctly after full graph execution; `V2_Unit_MoEOverlayCollectiveWorkspace` proves non-final sparse dispatch/return participants can complete their manual graph boundary after successful publish while final participants still require collective completion; `V2_Unit_MoEExpertWeightService` proves CPU prepared expert slabs are incrementally filled for disjoint masked participants without duplicating slabs; `V2_Unit_ForwardGraphTypes` proves named sparse/collective boundaries split segmented graph-capture plans unless graph-safe collective capture is explicitly enabled; the broader Phase 12 unit slice covering prefill graph cache, ForwardExecutionEngine chunking, MoE runtime table, decode histograms, rebalance controller, prepared-weight store resolution, and local expert prepared weights passed. Focused integration passed: `V2_Integration_MoEOverlaySparseTransport_MPI` and `V2_Integration_SegmentedGraphCaptureExecution`. Real-model graph-native overlay parity passed for Qwen3.5 MoE RocmHot/CpuCold `DecodeParity` and `PrefillParity`. Real Qwen3.6 MoE parity passes for SingleDevice `MTPGreedyMatchesPyTorchDecodeTokens` and `PrefixCacheMTPRestore`. Focused CTest passed on 2026-06-02 for heterogeneous ExpertOverlay ROCm2TP-hot/CPU2LocalTP-cold: `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_Qwen36MoEExpertOverlayPrefixMTPParity_MTPGreedyMatchesBaselineTokens_ROCm2TPHot_CPU2LocalTPCold` and `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_Qwen36MoEExpertOverlayPrefixMTPParity_PrefixCacheMTPRestore_ROCm2TPHot_CPU2LocalTPCold`. Additional focused slice passed on 2026-06-01: `V2_Unit_ForwardExecutionEngine`, `V2_Unit_DeviceGraphOrchestrator`, `V2_Integration_PrefillGraphCacheExecution_CUDA`, and `V2_Integration_PrefillGraphCacheExecution_ROCm` prove chunk index propagation, MoE domain/participant prefill graph identity hooks, snapshot observability for chunk id, bucket length, real-token range, domain id, participant id, placement epoch, topology signature, capture/replay phase, and structured `forward_graph.prefill_graph_lifecycle` counters while preserving padded-bucket reuse across changing real-token lengths. `V2_Unit_ForwardExecutionEngine` also proves placement-changing chunk-boundary maintenance clears the outer bucketed forward graph cache before the next chunk rebuilds under a new epoch. The CUDA/ROCm prefill graph-cache integrations now prove fixed-placement chunk schedules reach captured replay and forced chunk-boundary rebalance schedules clear, rebuild, recapture, and replay under a new placement epoch. `V2_Unit_ForwardExecutionEngine` now also proves live-state restore/truncate can reset captured replay state while preserving cached `ComputeGraph`s; real Qwen3.6 MoE SingleDevice ROCm MTP graph-capture smoke passed after this reset through the previous `-n 3`/`-n 4` rollback crash reproducer. Benchmark speedup evidence is intentionally deferred to Phase 14. | None for Phase 12 correctness and segmented-capture compatibility gates. |
| Phase 13: PyTorch Parity Acceptance Matrix | In progress | Dense and MoE parity targets are registered as normal parity-suite tests, and several Phase 13 cells have already passed as evidence for earlier phases, including single-device dense/MoE, LocalTP dense, NodeLocalTP dense, and heterogeneous ExpertOverlay MoE prefix+MTP parity. Dense Qwen3.6 SingleDevice CPU and CUDA prefix/MTP parity now have explicit CTest entries under `V2_Integration_Parity_Qwen36_CPU_SingleDevice_` and `V2_Integration_Parity_Qwen36_CUDA_SingleDevice_` rather than being inferred from the ROCm SingleDevice harness. Focused CTest passed on 2026-06-02 for all five CPU dense SingleDevice cases: `PrefixRestoreFullHit`, `PrefixRestorePartialHit`, `SplitPrefillMatchesPyTorchDecodeTokens`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Focused CTest also passed on 2026-06-02 for all five CUDA dense SingleDevice cases with the same coverage: `PrefixRestoreFullHit`, `PrefixRestorePartialHit`, `SplitPrefillMatchesPyTorchDecodeTokens`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Focused CTest also passed on 2026-06-02 for all three dense NodeLocalTP CPU cases: `PrefixRestoreFullHit`, `MTPGreedyMatchesPyTorchDecodeTokens`, and `PrefixCacheMTPRestore`. Focused CTest passed on 2026-06-02 for `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_`: the heterogeneous ROCm2TP-hot/CPU2LocalTP-cold MTP and prefix+MTP cases passed, and the 2x ROCm hot-only ExpertOverlay tests passed via an explicit resident-VRAM prerequisite skip on the current 32 GB ROCm machine. The hot-only harness currently requires at least 40 GiB total VRAM per ROCm participant because the no-fallback plan requires all 256 experts to fit on each ROCm participant. | Audit the full Phase 13 matrix requirement-by-requirement, rerun focused missing cells, and mark complete only after every required non-skipped parity cell has direct current evidence. |
| Phase 13.5: Small-M GEMV-Many Kernel Prerequisite | In progress | ROCm now has graph-capturable native-VNNI small-M verifier routes for M=2/3/4 across Q/K/IQ codebooks, fused QKV/GateUp shared-quant dispatch, same-codebook GDN qkv/z batching, heterogeneous-N Qwen3.6 qkv/z batching, fused-SwiGLU/FFN-down routing, declared-workspace FP32 GDN projection batching, and focused ROCm small-M integration/perf coverage. The latest hardening slices added graph-capturable atomic K-partitioning for `kb>1`, preserved dense captured replay state across live prefix restore/truncate while keeping MoE's targeted reset guard, and implemented verifier-row GDN recurrence/short-conv state capture so reject rollback can restore the accepted verifier row instead of replaying accepted tokens. Current same-binary long-lane ROCm evidence before the verifier-row shortcut remains speed-positive: depth-1 MTP reached 32.88 tok/s versus 21.02 tok/s baseline at `The quick brown fox`, `-c 64`, `-n 48`, with 95.83% acceptance; depth-2 reached 29.63 tok/s with 86.29% acceptance and 21 rollbacks. Focused unit/integration regression passed on 2026-06-02, including `V2_Unit_GDNMathematicalCorrectness`, `V2_Unit_GDNKernels`, `V2_Unit_MTPGraphConstruction`, `V2_Unit_PrefillDecodeTransition`, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`. CUDA has a narrower M=2 route with direct SingleDevice dense/MoE speedups. | Keep ROCm SingleDevice dense in Phase 13.5 until profiling shrinks the remaining verifier budget, especially GDN projection, ordinary GEMM, fused Gate/Up, and any remaining rollback overhead. Next repeat longer-prompt ROCm dense evidence with verifier-row restore enabled, then port the M=2/3/4 quantize-once GEMV-many contract and first-class perf rows to CUDA and CPU before returning to Phase 14 acceptance. |
| Phase 14: Benchmark Acceptance And Default-Enablement Readiness | Waiting on Phase 13.5 | `docs/v2/PREFIX_CACHE_MTP_BENCHMARK_NOTES.md` tracks real Qwen3.6 dense/MoE baseline, graph-capture status, and best observed MTP speedups by domain. CUDA SingleDevice dense currently has direct Qwen3.6 evidence for a graph-captured depth-1 MTP speedup after the native M=2 verifier route: 54.02 tok/s versus 40.44 tok/s baseline at `-c 128 -n 64`, with 96.88% acceptance and aligned shifted KV state. CUDA SingleDevice MoE currently has direct Qwen3.6 evidence for the same native M=2 route: 50.89 tok/s versus 31.20 tok/s baseline at `-c 64 -n 16`, with 78.12% acceptance and aligned shifted KV state. ROCm SingleDevice dense now has first speed-positive Qwen3.6 dense evidence after the fused-SwiGLU/FFN down small-M route: 25.51 tok/s versus 18.47 tok/s baseline at `-c 64 -n 16`, but Phase 13.5 remains open until longer prompts, remaining verifier buckets, and CUDA/CPU small-M contracts are covered. `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalPPHardFail` pins the current LocalPP matrix blocker as an early real-model hard fail with zero MTP draft/verifier counters, preventing recurrence of the old late stage-1 shifted-cache crash while the real PP-aware MTP path is still unimplemented. | Do not use Phase 14 to paper over kernel deficits. Resume benchmark acceptance after Phase 13.5 provides graph-capturable M=2/3/4 GEMV-many coverage and perf evidence for the supported backend/codebook matrix; then continue matrix-driven benchmark work until remaining supported domains show concrete speedups or documented blockers with traces. |

Latest ROCm SingleDevice dense Phase 13.5/14 note, 2026-06-02: a valid graph-captured depth-1 run with `The quick brown fox`, `-c 64`, and `-n 48` reached 33.57 tok/s versus a same-context graph-enabled baseline of 21.19 tok/s, or 1.58x, with 95.83% acceptance and `mtp.verifier_forward` still averaging 52.21 ms per two-token captured verifier. The same slice added `V2_Unit_BenchmarkRunnerCPU.FailsBeforePrefillWhenPromptPlusDecodeExceedsContext` after the `-c 64`, `-n 64` ROCm smoke exposed that benchmark prompt plus decode could exceed the context window and fault later on GPU; `BenchmarkRunner` now hard-fails before prefill with a structured context-budget error. Follow-up depth-2 work raised the decode graph cache window to `seq_len <= 4`, so chained verifier `M=3` now uses `main_verifier` decode graph cache/capture instead of the prefill path. `V2_Unit_ForwardExecutionEngineAdvanced` covers `M=3` and `M=4`, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke` proves real Qwen3.6 ROCm GPU-graph miss/hit counters for `seq_len=3`. Graph-captured depth-2 improved to 30.43 tok/s at `-c 64`, `-n 48`, but remains slower than depth-1 because acceptance/rollback costs now dominate. A later hardening slice temporarily forced graph-enabled native-VNNI small-M launchers to direct `kb=1` after default split-reduce still HSA-faulted in full-model graph warmup; the focused ROCm small-M regression filter and real Qwen3.6 `-c 64`, `-n 8` MTP smoke passed with 0 split-reduce launch counters, but verifier time rose to about 86.87 ms/call. The current follow-up replaces that direct clamp with graph-capturable atomic K-partitioning for `kb>1`: focused ROCm small-M graph-capture tests passed, a real Qwen3.6 `-c 64`, `-n 8` graph smoke completed at 17.32 tok/s with 75% acceptance, and every K-partitioned small-M launch reported `path=atomic_reduce` rather than `split_reduce`. A fresh current same-binary long-lane pair confirms the earlier speedup was not lost: 33.25 tok/s MTP versus 21.01 tok/s baseline at `-c 64`, `-n 48`, or 1.58x, with 95.83% acceptance. A rejected deferred post-sidecar checkpoint experiment reached 32.36 tok/s for depth 1 and 22.03 tok/s for depth 2, so the retained path is not another checkpoint-elision fallback. Dense live prefix restore/truncate now preserves captured replay state while MoE keeps a targeted reset guard until sparse collective rollback is graph-capture safe; the post-slice same-binary rerun reached 32.88 tok/s depth 1 versus 21.02 tok/s baseline and 29.63 tok/s depth 2. Verifier-row GDN recurrence/short-conv state capture and restore is now implemented and focused green, including the real Qwen3.6 ROCm chained graph smoke. The next sprint should rerun the longer-prompt ROCm dense ratchet with verifier-row restore enabled, then continue shrinking the captured verifier budget in GDN projection, ordinary GEMM, and fused Gate/Up.

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
- Consider MTP speedup or default-enablement claims only after the relevant backend passes Phase 13.5 small-M GEMV-many gates.
- Enable MTP on TP runners only after Phase 11 parity passes for the relevant TP scope.
- Enable MTP on MoE/ExpertParallel runners only after Phase 12 and Phase 13 MoE parity pass.
- Enable ExpertParallel graph-captured sparse overlay paths only after Phase 12 segmented-capture tests and Phase 14 benchmarks pass.
- Keep stochastic speculative sampling deferred until greedy correctness and acceptance telemetry are stable.
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
12. Done for the mixed-codebook GDN subgroup safety slice: a same-codebook subgroup batching experiment passed synthetic graph-capture tests but caused a real Qwen3.6 ROCm HSA memory access fault. The live path now keeps mixed-codebook GDN projection sets on the known-safe per-projection graph-native small-M route, and `GraphCapturedFusedMixedCodebookGDNProjectionM4BypassesBatchedRoute` requires an explicit `mixed_codebook` batched-route bypass while preserving graph-captured output correctness. A safe real Qwen3.6 ROCm `-c 64 -n 8` rerun completed at 21.65 tok/s with 75% acceptance and no lingering KFD process.
13. Done for the GDN qkv/z heterogeneous-N batching slice: Qwen3.6 GDN qkv/z `N={10240,6144}` pairs now use the generic native-VNNI small-M batched kernel instead of the unsafe Q4/M=2 specialized pair route. Focused ROCm integration captures and replays the exact shape, asserts batched projection counters, and real Qwen3.6 dense ROCm `-c 64 -n 8` completed at 22.35 tok/s with `main_verifier` GDN projection reduced from 95.90 ms to 91.65 ms total versus the earlier safe diagnostic.
14. Done for the FP32 GDN alpha/beta projection slice: ROCm FP32 fused projections now use hipBLAS batched SGEMM for same-`N` alpha/beta-style groups with device pointer arrays supplied by declared `IWorkspaceConsumer` buffers. The batched route hard-fails when workspace is missing instead of allocating ad hoc scratch, and the focused ROCm FP32 graph-capture regression binds a `DeviceWorkspaceManager`, captures/replays the batched call, and compares against CPU reference. A real Qwen3.6 dense ROCm `-c 64 -n 8` diagnostic confirmed the route active at `M=2/4`, `N=48`, and moved the short-prompt same-binary run to 24.70 tok/s MTP versus 21.70 tok/s baseline with 75% acceptance.
15. Done for the chained verifier decode-cache slice: `GraphCacheConfig::decode_seq_len` now defaults to 4, so MTP verifier continuations with `M=3/4` are decode-cache eligible instead of falling into prefill. `V2_Unit_ForwardExecutionEngineAdvanced` covers M=3 and M=4 all-position verifier cache reuse, `V2_Unit_ForwardGraphTypes` pins the default, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke` proves real Qwen3.6 ROCm GPU-graph `main_verifier` miss/hit counters for `seq_len=3`. Depth-2 graph-captured ROCm dense improved to 30.43 tok/s at `-c 64`, `-n 48`, but remains below the 33.57 tok/s depth-1 run because acceptance dropped to 87.10% and rollbacks rose to 20.
16. Done for the stage-timing stream-binding regression: ROCm `LLAMINAR_GPU_STAGE_TIMING=1` depth-1 MTP diagnostics faulted when timeline events were recorded on the worker default stream while graph-captured stages were rebound to a capture stream. `DeviceGraphExecutor` now records start/stop timing events on each stage's currently bound GPU stream, with `V2_Unit_GraphExecutorCollective.TimelineEventsUsePreBoundStageStream` covering the regression. A real Qwen3.6 dense ROCm `-c 64`, `-n 8` diagnostic completed without the old HSA fault and showed `main_verifier` remains the bottleneck: `GDN_PROJECTION` about 20.75 ms/call, ordinary `GEMM` about 14.71 ms/call, and `GEMM_FUSED_GATE_UP` about 11.62 ms/call, while `sidecar_forward` averaged about 3.47 ms/call.
17. Done for the graph-atomic K-partition slice: ROCm graph-enabled native-VNNI small-M no longer falls back to direct `kb=1` for graph safety. When `kb>1` under GPU graphs, the launcher zeroes the declared output on the graph stream, accumulates K partitions with in-kernel atomics, skips the unsafe split-reduce replay kernel, and records `path=atomic_reduce`. Focused ROCm integration covers graph-captured single and batched Qwen3.6-scale routes plus an explicit `KB=2` override, while retaining the hard fail for unsafe overrides above the graph-safe cap. A real Qwen3.6 dense ROCm graph smoke completed at 17.32 tok/s with 75% acceptance; this is a graph-safety/profiling step, not speedup evidence.
18. Done for the fresh long-lane A/B: current safe-path ROCm dense depth-1 MTP still reaches 33.25 tok/s versus 21.01 tok/s baseline at `The quick brown fox`, `-c 64`, `-n 48`, with 95.83% acceptance and fully captured `main_verifier`, `mtp_decode_sidecar`, and `mtp_decode_catchup` replay. The matching short stage diagnostic remains 17.30 tok/s because it has only 8 decode tokens, 75% acceptance, and heavy `stage_gpu,kernel` export.
19. Done for the reject-replay cleanup slice: depth-1 MTP rejects now replay only the accepted prefix token and append the correction token's shifted MTP row from that hidden state; deeper chained drafts now use the same lagged-correction shortcut when the first speculative token is rejected, and only the first post-sidecar checkpoint is captured because later chained sidecar rows are speculative scratch. Focused coverage includes `V2_Unit_PrefillDecodeTransition`, `V2_Unit_MTPGraphConstruction`, `V2_Unit_RankOrchestrator`, and `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke`. Real ROCm depth-2 long-lane evidence confirms the counters move in the intended direction (`post_sidecar_checkpoint_skipped_speculative=51`, `lagged_rejected_correction_replays=6`, measured `replay_tokens` down from 36 to 30), but throughput did not improve because depth-2 acceptance stayed around 76-77% with 32-33 rollbacks.
20. Rejected for default behavior: a lazy-correction rollback shortcut was investigated for first-speculative-token rejects. The idea was to output only the already-restored prefix token, stash the verifier correction as the next step's ready first token, and avoid the current-step shifted-row catch-up. It was mathematically plausible, but real ROCm Qwen3.6 dense depth-2 evidence at `The quick brown fox`, `-c 64`, `-n 48` reached only 27.98 tok/s versus the 20.99 same-binary baseline and worsened exported counters relative to the retained lagged-replay path (`rejected_tokens` 12 to 15, `rollbacks` 15 to 18, `replay_tokens` 30 to 45 in the stats window). Do not reintroduce correction deferral as a default unless it also reduces verifier pressure and preserves the speed ratchet.
21. Next: characterize and shrink the remaining ROCm verifier projection budget, starting with GDN projection, ordinary GEMM, and fused Gate/Up without reintroducing mixed-codebook subgroup batching. Adjacent rollback work should focus on preserving graph replay state across safe logical truncations or avoiding unnecessary recapture, not changing output-token grouping. After the ROCm verifier budget is back below the best speed-positive runs, repeat longer-prompt ROCm dense evidence and port the M=2/3/4 quantize-once GEMV-many contract to CUDA and CPU before returning to Phase 14 acceptance.

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

## Phase 14: Benchmark Acceptance And Default-Enablement Readiness

### Goal

Measure whether prefix cache and MTP provide real speedups on the supported correctness matrix, after the Phase 13.5 small-M GEMV-many prerequisite has landed for the backend under test, and define the evidence needed before any future default enablement.

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

Acceptance targets before considering default enablement:

- Prefix disabled overhead is within noise of baseline, target less than 2% median regression.
- RAM prefix hit shows a real prefill speedup on long shared prompts, with matched-token and timing counters proving prefill was skipped rather than hidden by measurement noise.
- Device-hot tier is faster than RAM hydrate for GPU restore or is documented as not worth enabling for that backend.
- Disk warm path improves repeated process startup or cross-process reuse workloads, or remains opt-in only.
- MTP decode benchmarks must identify the Phase 13.5 small-M GEMV-many route used for the verifier backend/codebook mix.
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
- `src/v2/app/` benchmark CLI plumbing if new flags are needed.
- `tests/v2/performance/`
- `docs/v2/PREFIX_CACHE_MTP_BENCHMARK_NOTES.md`

### Tests

- Benchmark smoke tests with tiny fixtures and no large model requirement.
- JSON schema test for benchmark output.
- Disabled-feature overhead smoke test.
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
15. Phase 14 benchmark acceptance matrix.
