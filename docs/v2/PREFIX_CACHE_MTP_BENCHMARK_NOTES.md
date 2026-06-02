# Prefix Cache And MTP Benchmark Notes

## Phase 14 GPU Graph Capture And MTP Speedup Matrix

This matrix is the durable scoreboard for Phase 14. Phase 14 benchmark
acceptance is currently paused behind Phase 13.5, the small-M GEMV-many kernel
prerequisite added after ROCm SingleDevice evidence showed graph-captured MTP
was correctness-green but still verifier-kernel limited. Update this matrix
whenever a real Qwen 3.6 dense or MoE benchmark changes one of these facts:

- baseline throughput without MTP,
- whether the MTP path is fully graph captured,
- whether collectives are graph captured where the backend technically supports it,
- best observed MTP throughput and speedup.

Use `Pending` only when no measured artifact exists yet. Use `Partial` when
some graph-capture path is active but the whole MTP inference step still has
manual stages, uncaptured collectives, or large host/replay overhead.

For every new MTP graph-capture benchmark, export one combined perf artifact
with:

```bash
LLAMINAR_PERF_STATS_FILTER=mtp,forward_graph
```

The `mtp` records explain draft, verifier, rollback, checkpoint, and replay
costs. The `forward_graph` records explain capture, replay, segmented/manual
stage policy, collective policy, and graph final-sync costs. Keeping both
domains together prevents graph-only artifacts from hiding high-level MTP
decode costs, or MTP-only artifacts from hiding graph-capture policy.

When MTP speedup is blocked by acceptance rather than verifier cost, also run
a focused acceptance trace:

```bash
LLAMINAR_PERF_STATS_FILTER=mtp.acceptance_trace
```

The `mtp.acceptance_trace` counter records one tagged row per measured MTP
step with the condition token, first main token, sidecar draft token, verifier
token, whether the second draft was accepted, and whether verifier state was
committed or replayed.

All current Phase 14 MTP benchmark evidence is for the implemented depth-1
sidecar path (`--mtp-draft-tokens 1`). Requests for deeper MTP drafts now fail
before prefill forward rather than silently running depth-1 behavior under a
depth-N configuration.

When a run is intended to prove a graph-native small-M verifier route, include
the `kernel` perf domain in the combined artifact:

```bash
LLAMINAR_PERF_STATS_FILTER=mtp,forward_graph,kernel
```

| Domain type | Device/backend target | Model class | Baseline decode tok/s | Graph-capture status | Collective capture status | Best MTP decode tok/s | Best MTP speedup | Evidence artifact | Current blocker |
|-------------|-----------------------|-------------|------------------------|----------------------|---------------------------|-----------------------|------------------|-------------------|-----------------|
| SingleDevice current slice | ROCm `rocm:0` | Qwen3.6 dense 27B Q4_K_S workspace-bound MTP smoke | 21.57 same-binary `The quick brown fox`, `-c 64`, `-n 8` | Fully captured for the dense depth-1 path in this smoke. Terminal-hidden row select is now a cached graph stage with a stable declared workspace scalar, GDN recurrence GPU merged-QKV deinterleave now requires the declared `gdn_deinterleave_scratch` graph workspace before backend dispatch, and `WorkspaceAllocator` synchronizes the affected GPU before releasing/reallocating a declared device workspace so in-flight ROCm graph work cannot read freed workspace memory. | N/A | 23.24 same-binary `The quick brown fox`, `-c 64`, `-n 8` | 1.08x current workspace-sync smoke | `/tmp/llaminar-mtp-bench/dense-rocm-rowselect-cache-baseline-c64-n8-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-workspace-sync-mtp-c64-n8-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-workspace-sync-mtp-c64-n8-stats.json`, `/tmp/llaminar-mtp-bench/dense-rocm-workspace-sync-mtp-c64-n8-stats.csv`; focused regressions: `V2_Unit_HiddenStateRowSelectStage`, `V2_Unit_WorkspaceAllocator` selected cases, `V2_Unit_GDNKernels` workspace guard cases | The crash/regression slice is fixed and speed-positive, but only barely. `mtp.verifier_forward` still averages about 60.88 ms per two-token verifier, while sidecar, shifted-row commit, and terminal-hidden refresh are small. Next ROCm work remains verifier graph GPU work and stable workspace naming for other rebuilt graph-native stages, not ad-hoc allocations. |
| SingleDevice | ROCm `rocm:0` | Qwen3.6 dense 27B Q4_K_S | 18.47 current fused-SwiGLU-route fox `-c 64 -n 16`; 19.06 current fused-SwiGLU-route fox `-c 64 -n 8`; 18.07 previous KB-cap same-binary fox `-c 64 -n 16`; 18.96 previous clean all-codebook fox `-c 64 -n 16`; 18.14 sidecar-normal-replay fox `-c 64 -n 8`; historical baselines below | Captured and correctness-green for the dense MTP path: main verifier, sidecar, shifted-prefill, and catch-up replay as capturable segments with zero manual stages; ROCm MTP sidecar shifted-prefill/decode now uses normal replay instead of force-recapture; the M=2/3/4 graph-native verifier route is covered in focused ROCm integration across Q8 and native Q/K/IQ codebooks; fused native small-M QKV/GateUp dispatch shares activation quantization once through a graph-native batched same-codebook route; GDN qkv/z same-codebook subgroups now share activation quantization, and Qwen3.6 heterogeneous-N `N={10240,6144}` qkv/z pairs use the generic graph-native batched route instead of the unsafe Q4/M=2 specialized pair route; fused-SwiGLU/FFN down now uses the graph-native native-VNNI small-M route for eligible M=2/3/4 verifier shapes; per-kernel ROCm scatter-partial workspace slices prevent shared-workspace split-K aliasing; native small-M M=2/3/4 split-K caps default graph-safe KB at 8 and hard-fails unsafe overrides; stage-GPU timing now completes after guarding stale MTP sidecar timeline events | N/A | 25.51 current fused-SwiGLU-route `-c 64 -n 16`; 22.35 heterogeneous-N GDN qkv/z batched route `-c 64 -n 8`; 22.19 fused-SwiGLU-route `-c 64 -n 8`; 22.14 stage-GPU diagnostic `-c 64 -n 8`; 16.21 clean all-codebook route before batched fused projection; 15.80 batched fused-projection shared-partial rerun; 15.27 KB-cap stable rerun; 14.04 sidecar-normal-replay diagnostic | 1.38x current fused-SwiGLU route against 18.47 tok/s baseline; 1.17x current short smoke against 19.06 tok/s baseline; 0.86x previous clean all-codebook route against 18.96 baseline | `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-baseline-n16.json`, `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-mtp-n16.json`, `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-baseline-n16.csv`, `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-mtp-n16.csv`, `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-baseline-n8.json`, `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-mtp-n8.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdn-heteron-batched-mtp-c64-n8-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-gdn-heteron-batched-mtp-c64-n8.csv`, `/tmp/llaminar-mtp-bench/dense-rocm-fusedswiglu-stagegpu-mtp-n8.json`, `/tmp/llaminar-mtp-bench/dense-rocm-fusedswiglu-stagegpu-mtp-n8.csv`; focused Phase 13.5 tests: `V2_Integration_ROCmQuantisedGemmSmallM`, `V2_Unit_GDNKernels`, `V2_Unit_StageTimeline`, `NativeVNNIGEMMPerfTest.MTP_SmallM_VerifierShapes_AllFormats`, `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison` | First ROCm SingleDevice dense speed-positive slice is proven after routing fused-SwiGLU/FFN down through graph-native small-M: verifier replay fell from about 98-99 ms to about 56.3 ms per two-token graph. The GDN qkv/z heterogeneous-N route is now stable and slightly faster on real Qwen3.6, moving `main_verifier` GDN projection from 95.90 ms to 91.65 ms total in the `-c 64 -n 8` diagnostic. This is not Phase 14 complete: repeat longer-prompt evidence, characterize remaining GDN/LM-head/recurrence work, and port the M=2/3/4 quantize-once GEMV-many contract to CUDA and CPU. |
| SingleDevice | CUDA `cuda:0` | Qwen3.6 dense 27B Q4_K_S | 40.44 current same-binary `-c 128 -n 64`; 40.48 earlier `-c 128 -n 64`; 41.39 current `-c 64 -n 16`; 43.76 current no-stats `-c 64 -n 4` | Small-context dense MTP reaches segmented replay with zero manual stages. Main verifier, full decode sidecar, and KV-only shifted-row catch-up decode graphs are capturable/replayed; default 4096-context run still does not fit this 24 GB device | N/A | 54.02 native M=2 verifier `-c 128 -n 64`; 46.93 stage-timing diagnostic `-c 128 -n 16`; 38.16 same-prompt `-n 4` clean run; 37.69 retained KV-only catch-up `-c 128 -n 64`; 36.78 rejected post-sidecar checkpoint-elision experiment | 1.34x current same-binary depth-1 MTP; 0.93x pre-native-M2 retained telemetry; 0.91x rejected checkpoint-elision experiment | `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-m2native-c128-n64-forwardgraph.json`, `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-m2native-c128-n64-forwardgraph-stats.csv`, `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-c128-n64-combined.json`, `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-c128-n64-combined-stats.csv`, `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-stagegpu-c128-n16-stats.csv`; historical artifacts are in Latest CUDA dense evidence below | First concrete CUDA SingleDevice dense MTP speedup is proven for depth-1 MTP after the graph-native CUDA M=2 native-VNNI verifier route: 54.02 tok/s versus 40.44 tok/s baseline with 96.88% acceptance, aligned shifted KV (`current_position=68`, `mtp_cached_tokens=67`), and zero manual segmented-replay stages. A fresh stage-timing diagnostic shows the captured main verifier replay is still about 30.5 ms per two-token graph; non-replay stage events show GEMM, GDN projection, and fused gate/up dominate the remaining GPU work. This is still below the Phase 14 dense target of roughly 2x. Next work should tune the remaining main-verifier cost, investigate deeper draft verification, and port the graph-native small-M path to ROCm, TP, and MoE domains. |
| SingleDevice | ROCm | Qwen3.6 MoE 35B | 21.23 | Partial: MTP GPU graphs now survive rollback/restore through the `-n 4` crash reproducer, but replay state is reset after each live-state rewind | N/A | 10.89 | 0.51x decode | `/tmp/llaminar-mtp-bench/moe-rocm-baseline-n4.json`, `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n3-fixed.json`, `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n4-fixed.json` | Crash fixed by resetting captured forward and MTP sidecar replay state after live prefix restore/truncate. MTP remains slower than baseline with 0% acceptance on this prompt; need sidecar/verifier acceptance and replay-cost work before claiming speedup. |
| SingleDevice | CUDA `cuda:0` | Qwen3.6 MoE 35B | 31.20 current same-binary `-c 64 -n 16`; 42.94 older 10-token `-n 4`; 27.56 older 9-token `-n 4` | Small-context MoE MTP reaches segmented replay with zero manual stages; single-participant rebalance downgrades to observe | N/A | 50.89 native M=2 verifier `-c 64 -n 16`; 22.43 older combined-telemetry short prompt | 1.63x current same-binary depth-1 MTP; 0.52x older combined telemetry | `/tmp/llaminar-mtp-bench/moe-cuda-current-baseline-m2native-c64-n16-forwardgraph.json`, `/tmp/llaminar-mtp-bench/moe-cuda-current-baseline-m2native-c64-n16-forwardgraph-stats.csv`, `/tmp/llaminar-mtp-bench/moe-cuda-current-mtp-m2native-c64-n16-combined.json`, `/tmp/llaminar-mtp-bench/moe-cuda-current-mtp-m2native-c64-n16-combined-stats.csv`; historical artifacts are in Latest CUDA MoE evidence below | CUDA SingleDevice MoE now has a real graph-captured speedup after the native M=2 verifier route: 50.89 tok/s versus 31.20 tok/s baseline, 78.12% acceptance, aligned shifted KV, and zero manual segmented-replay stages. This supports the single-device MoE path, but ExpertParallel MoE remains a separate Phase 14 gate. Next work should confirm across longer prompts and carry the small-M native route into ROCm/EP domains. |
| LocalTP | ROCm `rocm:0,rocm:1` | Qwen3.6 dense 27B Q4_K_S | 19.25 current same-prompt `Paris is -c 64 -n 4`; 24.15 older 9-token prompt | Not fully captured: verifier and sidecar graphs detect collectives and choose `allow_segmented=false`; forcing segmented collective replay for RCCL MTP hard-fails before sidecar launch | RCCL collectives currently force non-captured verifier/sidecar execution; `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1` is blocked for ROCm LocalTP MTP until the path is graph-safe | 11.69 current post-row-fix same-prompt; 20.46 older pre-row-fix same-prompt is no longer trusted for acceptance correctness | 0.61x current post-row-fix same-prompt | `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-baseline-rowfix-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-mtp-rowfix-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-mtp-rowfix-c64-n4-stats.csv`, `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-segmented-hardfail.json` | Correct LocalTP MTP and prefix-cache restore plus MTP are green after fixing all-position GPU sampling to honor the requested verifier row. Current honest MTP speed is slower than baseline with 50% acceptance. The attempted RCCL segmented path produced an HSA memory fault before the guard and now hard-fails with a structured `failure_reason`. Need real graph-safe RCCL/allreduce capture for MTP sidecar/verifier execution, plus acceptance work, before speedup claims. |
| LocalPP | ROCm `stage0=rocm:0, stage1=rocm:1` | Qwen3.6 dense 27B Q4_K_S | 20.47 at `-c 64` | Blocked: MTP is a hard fail before prefill on PP topologies | PP activation transfers are present in the baseline path, but MTP graph capture is not attempted | Blocked | N/A | `/tmp/llaminar-mtp-bench/dense-localpp-rocm-baseline-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-localpp-rocm-mtp-gpugraphs-c64-n4-hardfail.json`; `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalPPHardFail` | PP MTP shifted-prefill and verifier execution are not implemented. The previous late stage-1 shifted-cache failure is now a real-model prefill hard-fail regression with an explicit unsupported-topology message and zero MTP draft/verifier counters. |
| NodeLocalTP | CPU sockets | Qwen3.6 dense 27B Q4_K_S | 9.79 at `-c 64` | N/A for GPU graphs | Host/MPI TP collectives are exercised; no GPU graph capture applies | 7.26 | 0.74x decode | `/tmp/llaminar-mtp-bench/dense-cpu-nodelocaltp-baseline-c64-n4-bench.json`, `/tmp/llaminar-mtp-bench/dense-cpu-nodelocaltp-mtp-c64-n4-bench.json`, `/tmp/llaminar-mtp-bench/dense-cpu-nodelocaltp-mtp-c64-n4-stats.json` | Real Qwen3.6 dense CPU socket TP correctness path is green, with 50% MTP acceptance. MTP is slower than baseline because verifier/replay dominate, not because sidecar launch dominates. |
| Expert overlay EP | 2x ROCm | Qwen3.6 MoE 35B | Blocked before inference | Blocked before graph-capture measurement | Sparse dispatch/return graph capture required where ROCm supports it, but the current run blocks during resident expert preparation first | Blocked | N/A | `/tmp/llaminar-mtp-bench/moe-overlay-rocm2-replicated-streaming-hardfail.txt` | Added `configs/moe_overlay/rocm2_replicated_static.yaml` and harness coverage for a one-rank LocalTP `ReplicatedExperts` ROCm domain. Real Qwen3.6 MoE reaches graph config, but both ROCm devices fail resident expert VRAM preflight by the safety-margin check. `LLAMINAR_WEIGHT_STREAMING=1` is already enabled in the confirming run, but resident MoE expert streaming is not active for this GPU pipeline path. |
| Expert overlay EP | 2x ROCm plus 2x CPU dual-socket | Qwen3.6 MoE 35B | Blocked before inference | Blocked before graph-capture measurement | Heterogeneous sparse collectives must be graph-aware, but the current run stops before sparse dispatch because CPU fallback participant ranks no longer have sidecar endpoint runners | Blocked | N/A | `/tmp/llaminar-mtp-bench/moe-overlay-rocm2-cpu2-endpoint-hardfail.txt` | Added `configs/moe_overlay/rocm2_cpu2_replicated_static.yaml` and harness coverage for a rank-0 ROCm LocalTP hot tier plus rank-1 CPU LocalTP fallback tier. Real Qwen3.6 MoE rank 1 hard-fails as `CpuFallbackParticipant` because sidecar endpoint ranks were removed by graph-native MoE productionization. Next slice needs proper graph-native CPU fallback participant workers and sparse return through `TransferEngine`, not a fallback sidecar. |

Latest workspace-binding validation:

- ROCm FP32 mapped-output GEMM redirects now use the declared
  `rocm_fp32_mapped_redirect` workspace buffer instead of a lazy HBM
  allocation. `LMHeadStage` passes its bound graph workspace to the LM-head
  GEMM and declares the mapped-output redirect requirement during graph
  workspace planning.
- ROCm GDN recurrence merged-QKV GPU deinterleave now hard-fails at the graph
  stage boundary if the declared `gdn_deinterleave_scratch` workspace buffer is
  missing or undersized. This prevents graph execution from falling through to
  the backend's legacy private deinterleave scratch allocation path.
- GDN projection workspace planning now asks each projection kernel for its
  actual output width (`n_qkv`, `n_z`, `n_a`, `n_b`) rather than passing one
  generic graph hint to all four qkv/z/alpha/beta kernels.
- Dense FusedQKV workspace planning now asks the Q, K, and V kernels for their
  actual projection widths (`n_q`, `n_k`, `n_v`) rather than passing one generic
  graph hint across GQA projections.
- Dense FusedGateUp workspace planning now asks the gate and up kernels for
  their actual projection widths (`n_gate`, `n_up`) instead of delegating one
  generic `n` through the fused adapter.
- Focused regressions:
  `Test__ROCmFloatingPointGemmKernel.GraphCapturedBatchedFusedProjectionAlphaBetaM2MatchesReference`,
  `Test__ROCmFloatingPointGemmKernel.BatchedFusedProjectionRequiresWorkspace`,
  `Test__ROCmFloatingPointGemmKernel.MappedOutputRedirectRequiresDeclaredWorkspace`,
  `Test__HiddenStateRowSelectStage.*`, and the selected
  `Test__WorkspaceAllocator` replay/reallocation tests. `V2_Unit_GDNKernels`
  adds `Recurrence_GPUDeinterleaveRequiresBoundWorkspaceBeforeKernelDispatch`
  and `Projection_WorkspaceRequirementsUsePerProjectionN` plus the existing
  ShortConv/GDN recurrence workspace requirement checks. `V2_Unit_FusedQKVGEMMStage`
  and `V2_Unit_FusedGateUpGEMMStage` add `WorkspaceRequirementsUsePerProjectionN`.
- Real Qwen3.6 dense ROCm depth-1 MTP graph-capture smoke completed:
  `/tmp/llaminar-mtp-bench/dense-rocm-workspace-binding-mtp-c64-n8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-workspace-binding-mtp-c64-n8-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-workspace-binding-mtp-c64-n8-stats.csv`.
  This was a validation smoke rather than a new best-speed result: decode was
  17.64 tok/s with 75% acceptance.
- Follow-up release smoke after GDN/FusedQKV/FusedGateUp projection-specific
  workspace sizing completed on Qwen3.6 dense ROCm SingleDevice with GPU stage
  timing enabled: `/tmp/llaminar-mtp-bench/dense-rocm-workspace-projection-sizing-mtp-c64-n8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-workspace-projection-sizing-mtp-c64-n8-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-workspace-projection-sizing-mtp-c64-n8-stats.csv`.
  Decode was 21.43 tok/s with 75% acceptance. This is diagnostic rather than a
  new best-speed result because `LLAMINAR_GPU_STAGE_TIMING=1` was enabled;
  `mtp.verifier_forward` averaged 68.84 ms, with main-verifier GPU stage totals
  still dominated by `GDN_PROJECTION` 270.38 ms, `GEMM` 187.23 ms, and
  `GEMM_FUSED_GATE_UP` 153.09 ms across the 12 verifier calls.

Latest ROCm dense evidence:

- Baseline: `/tmp/llaminar-mtp-bench/dense-rocm-baseline-after.json`.
  - Prefill 2620.36 ms, 227.07 tok/s.
  - Decode 1753.69 ms for 32 tokens, 18.25 tok/s.
- Standard no-graph MTP after the GPU-graph hard-fail guard:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-no-graphs-long-after-guard-bench.json`
  and `/tmp/llaminar-mtp-bench/dense-rocm-mtp-no-graphs-long-after-guard-stats.json`.
  - Uses the same 595-token benchmark prompt family as the ROCm dense
    baseline and earlier no-graph MTP artifacts.
  - Decode 686.10 ms for 8 tokens, 11.66 tok/s.
  - MTP counters: 16 draft steps, 12 accepted tokens, 4 rejected tokens,
    4 rollbacks, 75% acceptance.
  - `verifier_forward`: 12 calls, 1609.19 ms total, about 134.10 ms/call.
  - `sidecar_forward`: 12 calls, 37.01 ms total, about 3.08 ms/call.
  - This is the current supported ROCm dense MTP behavior when GPU graphs are
    disabled: correct and instrumented, but still slower than the 18.25 tok/s
    no-MTP baseline.
- MTP after causal-offset and tiny-row attention work:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-rowdecode-bench.json`.
  - Prefill 4488.75 ms, 132.55 tok/s.
  - Decode 2201.57 ms for 16 tokens, 7.27 tok/s.
  - MTP counters: 32 draft steps, 24 accepted tokens, 8 rejected tokens,
    8 rollbacks, 75% acceptance.
- Perf stats: `/tmp/llaminar-mtp-bench/dense-rocm-mtp-rowdecode-stats.json`.
  - `verifier_forward`: 24 calls, 5208.07 ms total, about 217.0 ms/call.
  - `sidecar_forward`: 24 calls, 71.05 ms total, about 2.96 ms/call.
  - `decode_segmented_phase`: 6 warmup, 3 capture, 15 replay.
- Segment metrics: `/tmp/llaminar-mtp-bench/dense-rocm-mtp-segment-metrics-stats.json`.
  - Segment plan: one capturable segment, 644 stages, zero manual stages.
  - Captured replay launch plus post-launch bookkeeping: about 2.52 ms per replay.
  - Captured replay final sync: about 208.30 ms per replay, so GPU work inside
    the captured verifier graph is now the dominant blocker.
  - The short `-n 8` run reached 7.10 decode tok/s with 75% MTP acceptance.
- Sidecar capture retry: `/tmp/llaminar-mtp-bench/dense-rocm-mtp-sidecar-capture-retry-stats.json`.
  - MTP sidecar graph cache: 3 prefill misses, then 1779 prefill hits and
    12 decode hits.
  - MTP sidecar capture path: 3 `plain_after_build` calls, then 1779 prefill
    segmented calls and 12 decode segmented calls.
  - Sidecar timing remains small: about 3.43 ms/call in decode.
  - Throughput is essentially unchanged at 7.10 decode tok/s because
    `verifier_forward` remains about 222.5 ms/call.
- ROCm small-M verifier path: `/tmp/llaminar-mtp-bench/dense-rocm-mtp-smallm-bench.json`
  and `/tmp/llaminar-mtp-bench/dense-rocm-mtp-smallm-stats.json`.
  - Decode improved to 11.84 tok/s for the short `-n 8` benchmark.
  - `verifier_forward`: 12 calls, 1575.95 ms total, about 131.33 ms/call.
  - `sidecar_forward`: 12 calls, 40.23 ms total, about 3.35 ms/call.
  - MTP counters: 16 draft steps, 12 accepted tokens, 4 rejected tokens,
    4 rollbacks, 75% acceptance.
  - This remains useful historical verifier-kernel evidence, but ROCm MTP
    GPU graphs are now hard-failed after fresh HSA fault/hang reproducers.
- Phase 13.5 ROCm small-M native-VNNI perf:
  `NativeVNNIGEMMPerfTest.MTP_SmallM_VerifierShapes_AllFormats`.
  - Build target: `v2_perf_native_vnni_gemm`.
  - Shapes: `Qwen36_MTP_HiddenProjection` (N=5120, K=5120),
    `Qwen36_FFN_DownProjection` (N=5120, K=17408),
    `Qwen36_GDN_InnerProjection` (N=10240, K=5120),
    `Qwen36_GDN_TimeProjection` (N=1024, K=5120), and
    `Qwen36_GDN_OutputProjection` (N=5120, K=6144), each with
    M in {2, 3, 4}.
  - All native Q/K/IQ codebooks passed with cosine >= 0.999961.
  - Latest speedups versus INT8 ranged from 0.93x to 4.91x across the
    expanded verifier-shaped matrix. The weak row is `IQ3_S` on the small
    GDN time projection at M=2; the large FFN down and GDN output shapes remain
    clearly faster than INT8 for the Q4_K/Q5_K-equivalent formats observed in
    real Qwen3.6 counters.
  - Representative Q4_K-equivalent Q4_1 rows:
    hidden projection 2.29x/2.86x/3.05x for M=2/3/4; FFN down
    2.82x/3.43x/3.28x; GDN inner 2.47x/3.35x/3.91x; GDN time
    1.76x/1.87x/1.77x; GDN output 2.37x/2.90x/3.00x.
  - Route comparison `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison`
    passed for Q4_1/Q5_1 FFN down and GDN output shapes at M=2/4. Direct
    native prefill GEMM was correctness-identical to the small-M route
    (cosine 1.0) but much slower: direct-prefill+quant signals ranged from
    0.13x to 0.29x, where values greater than 1.0 would mean direct prefill
    is faster. This rejects a one-GEMM prefill-route substitution as the next
    ROCm speed lever.
  - `V2_Integration_ROCmQuantisedGemmSmallM` now also requires fused-QKV
    shared-quant calls to route through the graph-native batched same-codebook
    path, including optional bias epilogue support and batched projection
    counters.
  - This proves the lower-level ROCm small-M route is no longer Q4/M=2-only
    and that fused QKV can avoid the per-projection fallback, but it is not
    yet full-inference Phase 14 speedup evidence.
- Phase 13.5 real Qwen3.6 dense all-codebook small-M rerun:
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-allcodebooks-clean-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-allcodebooks-clean-mtp-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-allcodebooks-mtp-preflight-bench.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-allcodebooks-mtp-preflight-stats.json`.
  - Clean same-binary `The quick brown fox`, `-c 64`, `-n 16` baseline:
    843.95 ms for 16 decode tokens, 18.96 tok/s.
  - Clean depth-1 MTP with GPU graphs and the M=2/3/4 native small-M route:
    987.18 ms for 16 decode tokens, 16.21 tok/s, with 32 draft steps,
    28 accepted tokens, 4 rejected tokens, 4 rollbacks, 32 verifier runs,
    64 verifier tokens, and 87.5% acceptance.
  - Diagnostic stats prove the real model exercised
    `kernel.rocm_native_vnni_small_m_calls` for `M=2` and `M=4` across the
    verifier shapes, including Q4_K/Q6_K codebooks and the LM head, plus
    `kernel.rocm_fused_small_m_shared_quant_calls` for QKV and Gate/Up.
  - The lower-level route improves the previous best ROCm dense MTP result
    from 15.39 to 16.21 tok/s, but it is still 0.86x of the clean baseline.
    `main_verifier` segmented replay remains the dominant cost at about
    99.5 ms per two-token graph in the diagnostic run; sidecar and catch-up
    replay remain small at about 3.0 ms and 0.77 ms respectively.
- Phase 13.5 batched fused-projection shared-workspace rerun:
  `/tmp/llaminar-mtp-bench/dense-rocm-batched-smallm-clean-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-batched-smallm-sharedpartials-mtp-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-batched-smallm-sharedpartials-mtp-c64-n16-stats.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-batched-smallm-sharedpartials-mtp-c64-n16-stats.csv`.
  - The first real MTP rerun after enabling the batched fused-projection route
    exposed an HSA memory fault during warmup. The focused reproducer showed
    why the earlier synthetic QKV test missed it: graph stages bind multiple
    ROCm projection kernels to one `DeviceWorkspaceManager`, but split-K
    native small-M GEMV-many needs distinct partial buffers per projection.
  - The fix gives each ROCm GEMM kernel a per-instance
    `ROCM_SCATTER_PARTIAL_*` workspace slice. `V2_Integration_ROCmQuantisedGemmSmallM`
    now includes a Qwen3.6-scale QKV regression and a graph-captured
    Qwen3.6 GDN projection group that binds all projections through one
    shared workspace.
  - The post-fix real MTP run completed with GPU graphs enabled: prefill
    142.89 ms for 4 tokens, decode 1012.81 ms for 16 tokens, 15.80 tok/s,
    32 draft steps, 28 accepted tokens, 4 rejected tokens, 4 rollbacks,
    32 verifier runs, 64 verifier tokens, and 87.5% acceptance.
  - Kernel counters prove the route is active at real Qwen3.6 shapes:
    batched `M=2` gate/up `N=17408`, QKV `N=12288/1024`, GDN/other
    projection groups including `N=10240` and `N=6144`, plus LM-head-shaped
    native small-M calls.
  - This fixed the crash but not the speedup. Combined perf stats still put
    `main_verifier` graph replay at about 98 ms per two-token replay; sidecar
    graph replay and catch-up are much smaller. The next ROCm sprint should
    shrink verifier GPU work rather than add fallback paths.
- Phase 13.5 native small-M split-K cap rerun:
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-batchedgdn-baseline-c64-n16-kbcap8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-batchedgdn-mtp-c64-n16-kbcap8-rerun-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-batchedgdn-mtp-c64-n16-kbcap8-rerun-combined.csv`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-batchedgdn-mtp-c64-n2-kbcap8-bench.json`.
  - The original no-serialization `-n 2` HSA fault was narrowed to high
    native-VNNI small-M split-K fan-out: `KB=1/2/4/8/16` completed the real
    Qwen3.6 MTP smoke, while `KB=32` reproduced the memory fault.
  - The M=2/3/4 native small-M launchers now cap default graph-safe KB at 8.
    Explicit unsafe overrides hard-fail before launching kernels; the focused
    ROCm integration suite covers this with a Qwen3.6-scale GDN projection
    bundle.
  - Current same-binary `The quick brown fox`, `-c 64`, `-n 16` baseline:
    885.51 ms for 16 decode tokens, 18.07 tok/s.
  - Current same-binary depth-1 MTP with GPU graphs and no serialization:
    1047.55 ms for 16 decode tokens, 15.27 tok/s, 32 draft steps, 28 accepted,
    4 rejected, 4 rollbacks, 32 verifier runs, 64 verifier tokens, and 87.5%
    acceptance.
  - Real counters prove active batched `M=2` and `M=4` Gate/Up and QKV routes,
    plus native small-M GDN-shaped `N=10240/6144/5120/1024` projection buckets.
    This is stable correctness evidence, but still only 0.85x of baseline.
- Phase 13.5 Q4_0/Q4_1 M=2 ROCm specialization smoke:
  `/tmp/llaminar-mtp-bench/dense-rocm-phase135-postcommit-mtp-c64-n8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-phase135-postcommit-mtp-c64-n8-stats.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-phase135-q4m2-mtp-c64-n8-bench.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-phase135-q4m2-mtp-c64-n8-stats.json`.
  - The focused ROCm integration regression now replays the Qwen3.6-scale
    graph-captured GDN projection bundle four times and asserts native batched
    small-M route counters, so the shared-workspace split-K fix and batched
    path stay covered.
  - A dedicated Q4_0/Q4_1 `M=2` HIP path for single and batched native-VNNI
    small-M calls completed the real Qwen3.6 dense MTP smoke with GPU graphs:
    prefill 137.06 ms for 4 tokens, decode 570.82 ms for 8 tokens,
    14.02 tok/s, 16 draft steps, 12 accepted, 4 rejected, 4 rollbacks, and
    75% acceptance.
  - The paired post-commit pre-specialization smoke was 13.96 tok/s with
    the same acceptance. `mtp.verifier_forward` moved from 108.06 ms to
    107.86 ms per call and `forward_graph.main_verifier` remained about
    97.94 ms per two-token replay. This confirms the specialization is
    graph-capture stable but not the missing ROCm speedup lever.
- Phase 13.5 ROCm sidecar normal-replay cleanup:
  `/tmp/llaminar-mtp-bench/dense-rocm-stage-timing-baseline-n8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-stage-timing-baseline-n8.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-sidecar-replay-n8-bench.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-sidecar-replay-n8.csv`.
  - Removed the ROCm-specific MTP sidecar `force_recapture` override; the
    focused `V2_Unit_MTPGraphConstruction` regression now requires
    `force_recapture=false` for sidecar decode and shifted-prefill policy.
  - Real Qwen3.6 dense ROCm `The quick brown fox`, `-c 64`, `-n 8` completed
    with sidecar shifted-prefill/decode normal replay: decode 569.97 ms for
    8 tokens, 14.04 tok/s, 16 draft steps, 12 accepted, 4 rejected, 4
    rollbacks, and 75% acceptance. Paired no-MTP baseline was 440.97 ms for
    8 tokens, 18.14 tok/s.
  - Structured counters prove `force_recapture=false` for `mtp_decode_sidecar`,
    `mtp_decode_catchup`, and `mtp_shifted_prefill`. This removes an old ROCm
    graph-capture workaround, but it is not the speedup lever: `main_verifier`
    still replays a 644-stage graph at about 97.95 ms per two-token verifier.
- Phase 13.5 ROCm FP32 GDN alpha/beta batched SGEMM workspace slice:
  `/tmp/llaminar-mtp-bench/dense-rocm-gdn-fp32-hipblas-baseline-c64-n8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-gdn-fp32-hipblas-baseline-c64-n8.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-gdn-fp32-hipblas-mtp-c64-n8-bench.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-gdn-fp32-hipblas-mtp-c64-n8.csv`.
  - `ROCmFloatingPointGemmKernel` now exposes `IWorkspaceConsumer`
    requirements for hipBLAS batched SGEMM pointer arrays. The graph path
    supplies these through named `DeviceWorkspaceManager` buffers; the batched
    route no longer performs ad hoc HIP allocations and hard-fails if workspace
    is missing.
  - `HipBLASGemmKernel` now wraps `hipblasSgemmBatched` for same-shape FP32
    projection groups, allowing GDN alpha/beta-style `M=2/3/4` verifier
    projections to share one hipBLAS call where their output `N` matches.
  - Focused tests passed:
    `Test__ROCmFloatingPointGemmKernel.GraphCapturedBatchedFusedProjectionAlphaBetaM2MatchesReference`,
    `Test__ROCmFloatingPointGemmKernel.BatchedFusedProjectionRequiresWorkspace`,
    `Test__ROCmFloatingPointGemmKernel.TensorInterface_Basic`,
    `Test__ROCmQuantisedGemmSmallM.GraphCapturedFusedQ4KGDNProjectionM2MatchesSeparate`,
    and
    `Test__ROCmQuantisedGemmSmallM.GraphCapturedFusedQ4KQwen36GDNQkvZPairM2UsesHeterogeneousNBatchedRoute`.
  - Real Qwen3.6 dense ROCm `The quick brown fox`, `-c 64`, `-n 8`, GPU
    graphs, and stage timing completed after the workspace-backed hipBLAS
    route landed: same-binary baseline 21.70 tok/s; depth-1 MTP 24.70 tok/s,
    with 16 draft steps, 12 accepted tokens, 4 rejected tokens, 4 rollbacks,
    and 75% acceptance.
  - Structured counters prove the FP32 batched route is active in the real
    model: `kernel.rocm_fp32_batched_projection_calls` for `batch=2`,
    `k=5120`, `n=48`, and `M=2/4`. The MTP run recorded `M=2` count 576
    and `M=4` count 144.
  - The paired diagnostic puts `forward_graph.main_verifier` segmented replay
    at about 48.58 ms per two-token graph, `mtp.verifier_forward` at about
    58.63 ms/call, `mtp.sidecar_forward` at about 2.83 ms/call, and
    `mtp_decode_catchup` replay at about 0.38 ms/call. This is a useful
    verifier win and a 1.14x same-binary speedup for the short prompt, but
    still below the Phase 14 target; remaining work should keep attacking GDN
    projection, generic GEMM, and fused Gate/Up buckets.
- Phase 13.5 fused-SwiGLU/FFN down small-M ROCm route:
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-baseline-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-mtp-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-baseline-n8.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-smallm-fusedswiglu-mtp-n8.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-clean-graphs-n8-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-clean-graphs-n8.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-fusedswiglu-stagegpu-mtp-n8.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-fusedswiglu-stagegpu-mtp-n8.csv`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-fusedswiglu-stagegpu-mtp-n8.log`.
  - `ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu()` now routes
    eligible native-VNNI `M=2/3/4` verifier shapes through the graph-native
    small-M path instead of generic native prefill GEMM. If that selected
    route cannot launch, it hard-fails.
  - Focused regression: `V2_Integration_ROCmQuantisedGemmSmallM` now covers
    fused-SwiGLU/FFN down Q4_K M=2 and Q5_K M=4 with route counters tagged
    `source=fused_swiglu` plus FP32-reference cosine. It also covers
    graph-captured Qwen3.6-scale Q4_K FFN-down shapes for M=2 and M=4
    (`N=5120`, `K=17408`) and graph-captured M=4 GDN projection bundles
    (`N={10240,10240,1024,1024}`, `K=5120`).
  - Same-binary Qwen3.6 dense ROCm `The quick brown fox`, `-c 64`, `-n 16`:
    baseline decode 866.11 ms for 16 tokens, 18.47 tok/s; depth-1 MTP decode
    627.25 ms for 16 tokens, 25.51 tok/s, with 32 draft steps, 28 accepted,
    4 rejected, 4 rollbacks, and 87.5% acceptance in the request summary.
  - Short `-n 8` confirmation: baseline 419.77 ms for 8 tokens, 19.06 tok/s;
    MTP 360.51 ms for 8 tokens, 22.19 tok/s, with 75% acceptance.
  - Structured counters prove real Qwen3.6 fused-SwiGLU/FFN down small-M
    calls at `M=2` and `M=4` for Q4_K/Q5_K-equivalent codebooks and
    `N=5120`, `K=17408`.
  - `main_verifier` graph replay fell from the previous about 98-99 ms per
    two-token graph to about 56.3 ms. Sidecar replay remained about 2.37 ms
    and catch-up replay about 0.35 ms, so the next ROCm work should profile
    the remaining verifier GPU kernels rather than add fallback paths.
  - Clean verifier-focused rerun with stage timing and graph capture:
    `The quick brown fox`, `-c 64`, `-n 8` completed at 22.14 tok/s with
    75% acceptance after guarding MTP sidecar timeline collection against
    stale or unrecorded graph-capture events. `main_verifier` segmented replay
    was 56.15 ms per two-token graph, down from the sidecar-normal-replay
    diagnostic's 97.95 ms; `mtp.verifier_forward` was 66.18 ms/call, down from
    107.92 ms/call. Structured stage-GPU timing for `main_verifier` shows
    about 37.87 ms/pass of GPU work, dominated by GDN projection
    (~15.73 ms/pass), generic GEMM (~8.99 ms/pass), and fused Gate/Up
    (~6.30 ms/pass). Remaining work is to shrink those verifier buckets,
    characterize rollback/acceptance sensitivity, and repeat longer-prompt
    speedup evidence.
  - Rejected mixed-codebook GDN subgroup batching experiment:
    `/tmp/llaminar-mtp-bench/dense-rocm-subgroup-gdn-mtp-c64-n8.log`
    reproduced a real Qwen3.6 ROCm HSA memory access fault after changing the
    fused projection dispatcher to batch same-codebook subgroups inside an
    otherwise mixed-codebook GDN projection set. The focused regression
    `GraphCapturedFusedMixedCodebookGDNProjectionM4BypassesBatchedRoute`
    now proves graph-captured mixed Q4_K/Q5_K GDN projection output correctness
    while requiring an explicit `mixed_codebook` batched-route bypass. The
    follow-up safe real-model rerun,
    `/tmp/llaminar-mtp-bench/dense-rocm-mixedcodebook-safe-mtp-c64-n8-bench.json`
    plus `.csv`, completed at 21.65 tok/s with 75% acceptance and no lingering
    KFD process. Do not reintroduce subgroup batching until the lower-level
    batched native-VNNI launcher has a real-model-safe mixed-group design.
  - GDN qkv/z heterogeneous-N same-codebook batching:
    `/tmp/llaminar-mtp-bench/dense-rocm-gdn-heteron-batched-mtp-c64-n8-bench.json`,
    `/tmp/llaminar-mtp-bench/dense-rocm-gdn-heteron-batched-mtp-c64-n8.csv`,
    and `/tmp/llaminar-mtp-bench/dense-rocm-gdn-heteron-batched-mtp-c64-n8.json`.
    The real Qwen3.6 GDN qkv/z subgroup has `N={10240,6144}`. The first
    same-codebook subgroup attempt sent that pair through the Q4/M=2 batched
    specialization and faulted in the full model. The live path now keeps the
    subgroup batched, but uses the generic native-VNNI small-M batched kernel
    when a Q4/M=2 batch has heterogeneous projection widths.
  - Focused regression:
    `GraphCapturedFusedQ4KQwen36GDNQkvZPairM2UsesHeterogeneousNBatchedRoute`
    captures and replays the exact `M=2`, `K=5120`, `N={10240,6144}` Qwen3.6
    qkv/z shape, asserts batched projection counters, and compares both
    outputs against separate GEMVs with cosine 1.0. The neighboring ROCm
    small-M route tests and `V2_Unit_GDNKernels` also passed.
  - Real-model result: Qwen3.6 dense ROCm `The quick brown fox`, `-c 64`,
    `-n 8`, GPU graphs, depth-1 MTP completed at 22.35 tok/s with 75%
    acceptance. Counters prove the model used batched `M=2` and `M=4`
    `grid_n=256`, `projections=2` launches for the qkv/z pair and no
    `heterogeneous_n_pair` bypass.
  - Verifier deltas versus the earlier safe mixed-codebook run:
    `mtp.verifier_forward` moved from 809.43 ms to 787.89 ms total for
    12 calls, `main_verifier` segmented replay from 168.82 ms to 166.92 ms
    total for 3 replays, and `main_verifier` GDN projection stage-GPU time
    from 95.90 ms to 91.65 ms total. This is a stable verifier win, but the
    remaining `main_verifier` replay is still dominated by GDN projection,
    ordinary GEMM, and fused Gate/Up buckets.
  - Focused verifier replay follow-up:
    `/tmp/llaminar-mtp-bench/dense-rocm-gdn-fp32-alpha-beta-batched-mtp-c64-n8.log`,
    `/tmp/llaminar-mtp-bench/dense-rocm-mtp-c64-n8-nvnni-kb{1,2,4,8}.csv`,
    and direct perf runs of
    `NativeVNNIGEMMPerfTest.MTP_SmallM_DirectPrefillRouteComparison` plus
    `NativeVNNIGEMMPerfTest.MTP_SmallM_VerifierShapes_AllFormats`.
    A custom FP32 small-N alpha/beta projection batch passed its synthetic
    graph-capture unit but faulted in the real Qwen3.6 ROCm smoke before
    producing stats, so it was rejected and not committed. The real-model
    KB sweep showed smaller split-K is worse for verifier replay:
    `KB=1/2/4/8` produced `mtp.verifier_forward` totals of about
    1103.60/892.26/828.80/805.90 ms for 12 calls, with `KB=8` best.
    The focused perf binary agreed for the hot Q4_1/Q5_1 FFN-down and GDN
    output shapes: `KB=8` preserved the best M=4 and Q5-heavy timings while
    `KB=2` regressed sharply. Conclusion: keep the graph-safe native-VNNI
    small-M KB cap at 8; the next ROCm replay lever is not K-partition tuning.
  - Post-reset ROCm graph-capture hardening smoke:
    `/tmp/llaminar-mtp-bench/dense-rocm-mtp-graphs-trace-capture-safe-c64-n1-bench.json`
    and `/tmp/llaminar-mtp-bench/mtp-graphs-trace-capture-safe.log`.
    A real Qwen3.6 dense ROCm `The quick brown fox`, `-c 64`, `-n 1`,
    depth-1 MTP run completed with GPU graphs and `LLAMINAR_TRACE_STAGES=1`.
    Runtime state reported `execution_path=graph`, `prefill_success=true`,
    `decode_success=true`, `mtp_cached_tokens=4`, and repeated
    24-node/12-stage sidecar capture segments. This run is correctness and
    capture-safety evidence only, not speedup evidence, because the one-token
    decode is too short and had 0% accepted speculative tokens.
  - Focused regressions for the same slice:
    `V2_Unit_ComputeStageTraceGraphCapture` proves trace payload sampling does
    not call `fp32_data()` while `GraphCaptureGuard` is active; stage dump now
    hard-fails if requested during graph capture. The ROCm small-M integration
    suite now also hard-fails graph-native fused Qwen3.6 Gate/Up when split-K
    partial workspaces are aliased or undersized. This keeps the verifier path
    on declared `IWorkspaceConsumer`/`DeviceWorkspaceManager` buffers rather
    than ad-hoc allocation or shared scratch.
- Rejected shared-quant fused M=2 experiment:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-sharedquant-bench.json` and
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-sharedquant-stats.json`.
  - Correctness held in the focused small-M integration test, but decode
    regressed to 10.81 tok/s and `verifier_forward` rose to about
    181.20 ms/call.
  - Do not reintroduce this shape without a lower-level kernel change; a true
    batched/two-row GEMV path is likely needed instead.
- Atomic native-VNNI reduction experiment:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-smallm-atomicreduce-bench.json` and
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-smallm-atomicreduce-stats.json`.
  - Enabling `LLAMINAR_ROCM_NVNNI_ATOMIC_REDUCE=1` was effectively neutral:
    decode reached 11.77 tok/s and `verifier_forward` averaged about
    132.05 ms/call.
  - Keep this as an implementation option, not the next optimization axis.
- Native-VNNI M=2 row-overlap experiment:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-bench.json` and
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-stats.json`.
  - Enabling `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=1` completed the real Qwen3.6
    ROCm benchmark without the earlier broad `LLAMINAR_ROCM_CONCURRENT_DECODE`
    GPU fault.
  - Decode reached 11.94 tok/s, with 16 draft steps, 12 accepted tokens,
    4 rejected tokens, 4 rollbacks, and 75% acceptance.
  - `verifier_forward`: 12 calls, 1557.33 ms total, about 129.78 ms/call.
  - `sidecar_forward`: 12 calls, 40.64 ms total, about 3.39 ms/call.
  - Checkpoint capture/restore is no longer the dominant cost in this slice:
    `capture_live_prefix_state` averaged about 0.24 ms and
    `restore_live_prefix_state` averaged about 0.32 ms.
  - Treat this as historical optimization evidence only. The side-stream row
    overlap path is now blocked under GPU graphs until it becomes graph-native
    or is replaced by a true batched/two-row verifier kernel.
- Current GPU-graph smoke for base ROCm MTP:
  `/tmp/llaminar-mtp-bench/dense-rocm-current-baseline-gpugraphs-c64-n4-forwardgraph.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-current-baseline-gpugraphs-c64-n4-forwardgraph-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-gpugraphs-unguarded-c64-n4.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-gpugraphs-unguarded-c64-n4-stats.csv`.
  - The broad ROCm MTP plus GPU-graphs hard fail was stale for the base path.
    With `LLAMINAR_GPU_GRAPHS=1` and
    `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=0`, real Qwen3.6 dense MTP completes a
    short `-c 64`, `-n 4` decode smoke without an HSA fault.
  - Same-prompt baseline decode: 199.93 ms for 4 tokens, 20.01 tok/s.
  - Same-prompt MTP decode: 415.62 ms for 4 tokens, 9.62 tok/s.
  - MTP counters: 8 draft steps, 4 accepted tokens, 4 rejected tokens,
    4 rollbacks, 8 verifier runs, 16 verifier tokens, 50% acceptance.
  - Runtime state remains aligned after the smoke: `current_position=6`,
    `mtp_cached_tokens=5`, and `mtp_kv_cache_count=1`.
  - This is a correctness and graph-capture stability improvement, not a
    speedup. The captured main-verifier graph still dominates decode cost.
- Longer current GPU-graph replay evidence for base ROCm MTP:
  `/tmp/llaminar-mtp-bench/dense-rocm-current-baseline-gpugraphs-c64-n8-forwardgraph.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-current-baseline-gpugraphs-c64-n8-forwardgraph-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-gpugraphs-unguarded-c64-n8-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-gpugraphs-unguarded-c64-n8-combined-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-gpugraphs-unguarded-fox-c64-n8-combined.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-gpugraphs-unguarded-fox-c64-n8-combined-stats.csv`.
  - Same-prompt `Paris is` baseline decode: 420.37 ms for 8 tokens,
    19.03 tok/s.
  - Same-prompt `Paris is` MTP decode: 691.38 ms for 8 tokens, 11.57 tok/s,
    with 16 draft steps, 12 accepted, 4 rejected, 4 rollbacks, and 75%
    acceptance.
  - The `The quick brown fox` MTP run reached 11.68 tok/s with the same 75%
    acceptance but clearer replay counters: `main_verifier` replay reached
    Phase 3 for 3 calls, averaging about 119.66 ms per two-token verifier
    graph; `mtp_decode_sidecar` replay averaged about 2.36 ms and
    `mtp_decode_catchup` replay averaged about 0.44 ms.
  - Baseline `main_decode` replay on the same backend and context averaged
    about 54.70 ms per one-token graph. ROCm MTP is therefore blocked by
    verifier graph GPU work, not host launch overhead, sidecar cost, or
    checkpoint capture/restore.
- Post-recapture ROCm SingleDevice dense graph evidence:
  `/tmp/llaminar-mtp-bench/dense-rocm-post-recap-baseline-c64-n16-forwardgraph.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-post-recap-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-post-recap-mtp-c64-n16-combined.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-post-recap-mtp-c64-n16-bench.json`.
  - Same-binary `The quick brown fox` baseline decode: 870.82 ms for 16
    tokens, 18.37 tok/s.
  - Same-binary depth-1 MTP decode: 1201.17 ms for 16 tokens, 13.32 tok/s,
    with 32 draft steps, 28 accepted tokens, 4 rejected tokens, 4 rollbacks,
    32 verifier runs, 64 verifier tokens, and 87.5% acceptance.
  - MTP correctness and graph replay are stable after the replay-state reset
    and ROCm sidecar stream-binding fixes, but this is still only 0.72x of the
    no-MTP baseline.
  - `verifier_forward` averaged about 126.77 ms/call. The captured
    `main_verifier` segmented replay final sync averaged about 119.89 ms/call,
    while baseline `main_decode` segmented replay final sync averaged about
    54.44 ms/call.
  - `sidecar_forward` averaged about 3.57 ms/call, and checkpoint/restore
    remained sub-millisecond, so the next sprint target is the two-token main
    verifier graph's GPU work rather than sidecar launch or live-state
    bookkeeping.
- Current graph-native Q4_K M=2 ROCm verifier diagnostics:
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-mtp-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-max128-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-streamonly-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-kb3-mtp-c64-n16-bench.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-lane0sum-mtp-c64-n16-bench.json`.
  - Same-binary `The quick brown fox` baseline decode: 902.12 ms for 16
    tokens, 17.74 tok/s.
  - Current default graph-captured MTP decode: 1097.96 ms for 16 tokens,
    14.57 tok/s, with 87.5% acceptance. The Q4_K M=2 kernel route is proven
    by `kernel.rocm_native_vnni_m2_calls` counters for the real model shapes.
  - Best observed graph-captured MTP remains the max-128 segment diagnostic at
    15.39 tok/s. The captured `main_verifier` replay still averaged about
    103.3 ms per two-token graph, so the segment cap did not fix the verifier
    GPU cost.
  - Stream-only replay reached 15.25 tok/s, which confirms HIP graph replay is
    not the only blocker.
  - Forcing `LLAMINAR_ROCM_NVNNI_GEMV_KB=3` reached only 14.73 tok/s. The
    lane-0 activation-sum broadcast experiment regressed to 13.31 tok/s and
    was reverted. Keep both as rejected tuning evidence.
  - Rejection replay is measurable but not safe to elide yet: Qwen3.6 hybrid
    GDN state needs a row-scoped restore contract before a rejected draft can
    output only the first accepted token without replaying the corrected token.
- Rejected post-recapture single-stream native-VNNI M=2 reuse experiment:
  `/tmp/llaminar-mtp-bench/dense-rocm-m2native-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2native-mtp-c64-n16-bench.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-m2native-mtp-c64-n16-combined.csv`.
  - Routing Q4_K `M=2` verifier GEMM through the existing single-stream
    native-VNNI prefill GEMM entrypoint preserved real-model execution but
    regressed MTP decode to 7.23 tok/s versus the same-binary 18.06 tok/s
    no-MTP baseline.
  - The captured `main_verifier` segmented replay final sync rose to about
    231.14 ms/call and `verifier_forward` averaged about 237.92 ms/call.
  - The experiment was reverted. Do not reintroduce this route as-is; a
    graph-capturable two-row verifier path should preserve the GEMV-like memory
    behavior while avoiding side streams.
- Rejected graph-native M=2 native-GEMM reuse experiment:
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-m2gemm-gpugraphs-fox-c64-n8-combined.json`
  and
  `/tmp/llaminar-mtp-bench/dense-rocm-current-mtp-m2gemm-gpugraphs-fox-c64-n8-combined-stats.csv`.
  - Routing Q4_K `M=2` verifier GEMM through the existing single-stream
    native-VNNI prefill GEMM entrypoint passed the focused synthetic
    `V2_Integration_ROCmQuantisedGemmSmallM` parity test, but produced an HSA
    memory access fault under real Qwen3.6 graph capture during warmup and left
    the benchmark process stuck until killed.
  - The experiment was reverted. Do not reintroduce the prefill-GEMM route as
    the MTP verifier fast path without a real captured Qwen3.6 smoke and a
    lower-level fix for the HSA fault.
- GPU-graph hard-fail for native-VNNI M=2 row overlap:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-gpugraph-hardfail.json` and
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-gpugraph-hardfail-stats.json`.
  - A fresh real Qwen3.6 ROCm smoke with `LLAMINAR_GPU_GRAPHS=1` and
    `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=1` produced an HSA memory access fault
    during warmup and then left the self-launched benchmark process stuck.
  - The configuration now fails before the first decode-side MTP verifier
    launch with structured `failure_reason`: `ROCm MTP decode is incompatible
    with LLAMINAR_ROCM_CONCURRENT_M2_ROWS when LLAMINAR_GPU_GRAPHS=1; M=2
    row-overlap launches side streams that are not graph-capture safe`.
  - The stats artifact proves shifted-prefill sidecar graph capture still ran
    safely before the hard fail: `sidecar_decode_capture_policy` records
    `context=mtp_shifted_prefill`, `allow_segmented=true`, `has_collectives=false`,
    and four segmented sidecar reuse attempts before the verifier guard fired.
  - Treat the older M=2 row-overlap numbers as historical optimization evidence,
    not rollout evidence, until the row-overlap verifier becomes graph-native
    or is replaced by a true batched/two-row verifier kernel.
- Stage-GPU diagnostic run:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-stagegpu-bench.json` and
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-stagegpu-stats.json`.
  - Short `-n 4` benchmark with `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=1` reached
    14.81 decode tok/s with 100% MTP acceptance, 8 draft steps, 8 accepted
    tokens, 0 rejected tokens, and 0 rollbacks.
  - `verifier_forward`: 6 measured calls, 786.68 ms total, about
    131.11 ms/call.
  - Main verifier GPU stage timing averaged about 72.02 ms per verifier graph:
    GEMM 31.34 ms, GDN projection 18.76 ms, fused gate/up 11.65 ms, fused QKV
    2.65 ms, and LM head 1.47 ms.
  - The verifier wall/GPU gap remains large: about 129.50 ms wall versus
    72.02 ms GPU per measured verifier iteration, so the next slice should
    separate graph replay synchronization overhead from true kernel time.
- Stage-timing-off comparison:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-notiming-bench.json`.
  - Disabling `LLAMINAR_GPU_STAGE_TIMING` and structured perf export improved
    the comparable `-n 8` MTP run from 11.94 tok/s to 12.32 tok/s.
  - This recovers only a small part of the deficit, so profiling overhead is
    not the main blocker. It did expose that GPU stage timing must remain
    opt-in outside explicit profiling runs.
- Replay-context and verifier-cache diagnostics:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-forward-cache-release-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-forward-cache-release-stats.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-forward-cache-n8-release-bench.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-mtp-forward-cache-n8-release-stats.json`.
  - Short `-n 4` run reached 14.76 decode tok/s, but each measured iteration
    had only one main-verifier miss and one main-verifier hit, so verifier
    graph replay did not reach steady Phase 3.
  - Longer `-n 8` run reached 11.98 decode tok/s with 75% acceptance and did
    reach main-verifier replay: 3 replay calls, one capturable segment, 644
    stages, zero manual stages.
  - Main-verifier replay averaged about 119.56 ms/call. Graph launch plus
    segment bookkeeping averaged about 7.14 ms/call, while final stream sync
    averaged about 112.38 ms/call.
  - Decode sidecar replay is graph captured and remains small: 12 replay calls,
    one capturable 21-stage segment, about 3.24 ms/call total.
  - Shifted-prefill sidecar replay is also separated now:
    `context=mtp_shifted_prefill`, 1773 replay calls, about 2.50 ms/call.
  - The next performance target is the captured main-verifier graph's GPU work
    and completion sync, not host launch overhead or sidecar decode overhead.
- Split final-sync diagnostic:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-syncsplit-n8-release-bench.json`
  and `/tmp/llaminar-mtp-bench/dense-rocm-mtp-syncsplit-n8-release-stats.json`.
  - Decode reached 11.94 tok/s, consistent with the other `-n 8` MTP runs.
  - Main-verifier final sync averaged about 112.40 ms/call.
  - The capture-stream wait accounts for essentially all of that:
    `stream=capture` averaged about 112.40 ms/call, while `stream=default`
    averaged about 1.6 us/call.
  - Decode-sidecar final sync is also capture-stream dominated but small:
    about 3.12 ms/call on `stream=capture` versus about 13 us/call on
    `stream=default`.
  - This rules out a stray default-stream wait as the verifier blocker. The
    captured verifier graph's GPU work and graph-completion semantics are the
    optimization target.
- Graph-native ROCm Q4_K M=2 verifier route and current same-binary
  comparison:
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-baseline-c64-n16-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-baseline-c64-n16-combined.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-mtp-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-mtp-c64-n16-combined.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-default12-mtp-c64-n16-combined.csv`.
  - Focused validation passed before the benchmark:
    `V2_Unit_ForwardGraphTypes`, `V2_Integration_ROCmQuantisedGemmSmallM`,
    `V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_PrefixCacheMTPRestore`,
    and
    `V2_Integration_Parity_Qwen36_SingleDevice_Qwen36SingleDevicePrefixMTPParity_MTPGreedyMatchesPyTorchDecodeTokens`.
  - Same-binary `The quick brown fox`, `-c 64`, `-n 16` baseline decode:
    902.12 ms for 16 tokens, 17.74 tok/s.
  - Same-binary depth-1 MTP decode with GPU graphs and default M2 target
    waves: 1097.96 ms for 16 tokens, 14.57 tok/s, with 32 draft steps,
    28 accepted tokens, 4 rejected tokens, 4 rollbacks, 32 verifier runs,
    64 verifier tokens, and 87.5% acceptance.
  - The Q4_K/Q4_1 M=2 verifier route was exercised in the real model:
    `kernel.rocm_native_vnni_m2_calls` recorded `codebook=5` for the
    verifier shapes `k=5120,n=17408`, `k=5120,n=12288`,
    `k=5120,n=10240`, `k=5120,n=6144`, `k=5120,n=1024`, and
    `k=6144,n=5120`.
  - Main verifier replay improved versus the previous post-recapture base MTP
    path, but not enough for speedup: `main_verifier` segmented replay now
    averages about 103.36 ms per two-token graph, with final sync about
    103.14 ms. Baseline `main_decode` replay on the same binary averages
    about 54.91 ms per one-token graph, with final sync about 54.39 ms.
  - Sidecar and catch-up are not the immediate blocker in this slice:
    `mtp_decode_sidecar` replay averages about 2.60 ms and
    `mtp_decode_catchup` averages about 0.64 ms.
  - Direct ROCm recapture/reinstantiate removed the noisy HIP graph update
    invalid-argument path, but did not materially improve throughput:
    `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-directreinst-mtp-c64-n16-bench.json`
    reached 14.92 tok/s.
  - A tiny target-waves sweep put 12 waves slightly ahead:
    `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-tw12-mtp-c64-n16-bench.json`
    reached 15.19 tok/s, while KB16 split-K and 8 target waves were worse.
    The default M2 route now uses 12 target waves, but this remains below
    baseline.
  - Conclusion: ROCm SingleDevice dense MTP is graph-captured, correctness
    green, and route-instrumented, but not Phase 14 positive yet. The next
    sprint target should be the remaining captured verifier GPU work,
    especially the GDN/GEMM-heavy two-token verifier path or a deeper draft
    path that amortizes verifier cost.
- Shared-quant fused M=2 ROCm verifier route:
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-sharedquant-baseline-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-sharedquant-baseline-c64-n16-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-sharedquant-baseline-c64-n16-combined.csv`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-sharedquant-c64-n16-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-sharedquant-c64-n16-combined.json`,
  and `/tmp/llaminar-mtp-bench/dense-rocm-m2q4-sharedquant-c64-n16-combined.csv`.
  - Focused validation passed before the benchmark:
    `V2_Integration_ROCmQuantisedGemmSmallM` and the two ROCm
    SingleDevice Qwen3.6 prefix/MTP parity cases
    `PrefixCacheMTPRestore` and `MTPGreedyMatchesPyTorchDecodeTokens`.
  - Same-binary `The quick brown fox`, `-c 64`, `-n 16` baseline decode:
    867.12 ms for 16 tokens, 18.45 tok/s.
  - Same-binary depth-1 MTP decode with GPU graphs and shared fused M=2
    activation quantization: 1107.64 ms for 16 tokens, 14.45 tok/s, with
    32 draft steps, 28 accepted tokens, 4 rejected tokens, 4 rollbacks,
    32 verifier runs, 64 verifier tokens, and 87.5% acceptance.
  - The real-model run proves QKV and gate/up no longer re-quantize the same
    two verifier rows per projection: `kernel.rocm_fused_m2_shared_quant_calls`
    recorded `k=5120,projections=3` for QKV and `k=5120,projections=2` for
    gate/up. Shared native projection counters recorded
    `kernel.rocm_native_vnni_m2_calls` with `shared_quant=true`.
  - This is a correctness and routing cleanup, not a speedup. The captured
    `main_verifier` replay still averages about 103.33 ms per two-token graph,
    with final sync about 103.07 ms; decode sidecar replay averages about
    2.65 ms and catch-up replay about 0.64 ms.
  - Rejected ROCm sidecar no-recapture experiment: removing the
    ROCm-specific sidecar `force_recapture` policy passed the synthetic
    graph-construction unit path but failed real Qwen3.6
    `PrefixCacheMTPRestore` parity with an HSA memory access fault after prefix
    restore. The current unit regression records the policy explicitly:
    ROCm sidecar decode/shifted-prefill uses `force_recapture=true`, while CUDA
    remains `force_recapture=false`.
  - Next ROCm SingleDevice sprint target remains the captured verifier graph's
    GPU work and MTP per-step overhead. More host-side fallbacks or flag guards
    are not expected to create the Phase 14 speedup.

Latest CUDA dense evidence:

- Focused correctness prerequisite:
  `ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen36_CUDA_SingleDevice_" --output-on-failure --parallel`.
  - Passed on 2026-06-02 for all five CUDA SingleDevice dense Qwen3.6
    prefix/MTP parity cases: prefix full hit, prefix partial hit, split
    prefill, MTP greedy, and prefix+MTP restore.
  - This closes the CUDA SingleDevice correctness slice for the current
    Phase 13 matrix, but it does not change the Phase 14 speedup blocker:
    current CUDA dense MTP still trails the no-MTP decode baseline.
- CUDA small-M verifier regression slice:
  `ctest --test-dir build_v2_integration -R ^V2_Integration_CUDAGemmParity$ --output-on-failure --parallel`.
  - Added focused CUDA GEMM integration coverage for Q4_K verifier shapes
    `M=2,N=896,K=768`, `M=2,N=512,K=768`, and fused two-projection
    `M=2` verifier execution.
  - Added focused CUDA fused-SwiGLU down coverage for
    `M=2,N=896,K=768`, matching the verifier FFN-down path that was still
    using a prefill-style small-M route after the projection GEMV fix.
  - The tests now cover the row-wise native verifier GEMV path for 512-wide
    projections as well as the larger Qwen3.6 verifier projections.
  - Added reproducer coverage for the earlier ordering crash: a synthetic
    small-M prepared GEMM test followed by `RealModel_Q4_0_AttnQ_TensorAPI`
    now passes for both direct and fused paths. The synthetic prepared-weight
    tests use unique canonical names/model IDs and avoid pre-uploading the
    synthetic quantized weights, preventing host-pinning state pollution before
    the real model TensorAPI path.
- Small-M fused-SwiGLU verifier Phase 14 rerun:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-swiglu-c64-n4.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-swiglu-c64-n4-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-swiglu-c64-n4-stats.csv`.
  - Model/device/context: Qwen3.6 dense 27B Q4_K_S on RTX 3090 `cuda:0`,
    `-c 64`, `-n 4`, deterministic 5-token prompt, GPU graphs enabled.
  - MTP decode: 111.07 ms for 4 tokens, 36.01 tok/s, about 0.82x the
    43.97 tok/s no-MTP baseline.
  - MTP counters: 8 draft steps, 8 accepted, 0 rejected, 0 rollbacks,
    8 verifier runs, 16 verifier tokens, 100% acceptance.
  - High-level MTP timers in the benchmarked section: `decode_step_total`
    averaged about 55.53 ms, `verifier_forward` about 53.05 ms, and
    `sidecar_forward` about 1.95 ms.
  - Main verifier GPU stage timing averaged about 38.85 ms per verifier graph:
    fused gate/up 12.50 ms, fused-SwiGLU/down GEMM 11.99 ms, GDN projection
    6.39 ms, GDN recurrence 1.54 ms, fused QKV 1.47 ms, and LM head 1.33 ms.
  - This improves the same-prompt MTP run from 18.95 to 36.01 tok/s after
    making FFN-down fused-SwiGLU use row-wise native GEMV for small-M verifier
    shapes. The short-run blocker is now the remaining main-verifier graph
    work. The old longer `-n 16` acceptance/replay blocker was traced to
    shifted MTP KV drift and is superseded by the shifted-cache catch-up entry
    below.
- Clean no-stats comparison after fused-SwiGLU verifier work:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-swiglu-nostats-c64-n4.json`
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-swiglu-nostats-c64-n4.json`.
  - Same model/device/context/prompt as the instrumented run, but without
    `LLAMINAR_PERF_STATS_*` export.
  - Baseline decode: 91.41 ms for 4 tokens, 43.76 tok/s.
  - MTP decode: 104.83 ms for 4 tokens, 38.16 tok/s, about 0.87x baseline.
  - MTP counters stayed ideal for this short prompt: 8 draft steps, 8 accepted,
    0 rejected, 0 rollbacks, 100% acceptance.
  - Profiling/export overhead is not the main blocker: the clean run is only
    about 2.15 tok/s faster than the instrumented 36.01 tok/s run.
- MTP acceptance-trace regression and real-model diagnostic:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-acceptance-trace-c64-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-acceptance-trace-c64-n16-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-acceptance-trace-c64-n16-stats.csv`.
  - Added `mtp.acceptance_trace` structured counter coverage in
    `V2_Unit_PrefillDecodeTransition` for both accepted and forced-rejected
    MTP decode branches. This is the regression test for future acceptance
    diagnostics.
  - Real CUDA dense trace command used Qwen3.6 dense 27B Q4_K_S on RTX 3090
    `cuda:0`, `-c 64`, `-n 16`, deterministic 4-token prompt, GPU graphs
    enabled, and `LLAMINAR_PERF_STATS_FILTER=mtp.acceptance_trace`.
  - The run reproduced the longer-decode blocker: 32 draft steps, 12 accepted,
    20 rejected, 20 rollbacks, 32 verifier runs, and 37.5% acceptance.
  - The measured trace records start after warmup at draft step 9 and show a
    stable three-accept, five-reject pattern repeated across the benchmarked
    iterations:
    `37550/33075 -> 888` accepted,
    `888/279 -> 15217` accepted,
    `15217/5388 -> 13` accepted, then verifier disagrees with sidecar drafts
    for the next five steps.
  - Next CUDA dense work should separate true sidecar quality from shifted-KV
    or rollback-state drift after the first rejection. The trace makes this
    debuggable without parsing logs.
- Shifted MTP KV catch-up fix and corrected `-n 16` Phase 14 rerun:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-c64-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-shifted-catchup-c64-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-shifted-catchup-c64-n16-stats.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-shifted-catchup-c64-n16-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-shifted-catchup-c64-n16-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-shifted-catchup-c64-n16-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-shifted-catchup-c64-n16-combined-stats.csv`.
  - Added focused regression coverage in `V2_Unit_PrefillDecodeTransition`
    proving that accepted and forced-rejected MTP decode paths commit shifted
    rows for the accepted/correction token sequence on SingleDevice,
    GlobalTP delegation, and LocalTP fanout.
  - The clean no-MTP baseline reached 386.61 ms for 16 decode tokens,
    41.39 tok/s.
  - The clean corrected MTP run reached 509.22 ms for 16 decode tokens,
    31.42 tok/s, about 0.76x baseline.
  - MTP counters improved to 32 draft steps, 28 accepted tokens,
    4 rejected tokens, 4 rollbacks, and 87.5% acceptance.
  - Runtime state proves the shifted-cache invariant is restored:
    `current_position=20`, `mtp_cached_tokens=19`,
    `mtp_kv_cache_count=1`.
  - The combined `mtp,forward_graph` artifact reached 31.35 tok/s and shows
    the main verifier path is graph captured with one capturable segment,
    644 stages, zero manual stages, and graph-cache hits after the warmup
    misses.
  - `verifier_forward` is the live blocker: 24 measured calls took
    1243.64 ms total, about 51.82 ms per 2-token verifier. Main-verifier
    segmented replay took about 46.05 ms per captured replay call, while
    graph launch plus sync overhead was below 0.4 ms/call.
  - Sidecar and shifted-row catch-up are not the dominant cost:
    `sidecar_forward` averaged about 1.99 ms, `shifted_row_commit` averaged
    about 2.00 ms, and their 21-stage graph replays averaged about
    1.75-1.87 ms.
  - The next CUDA dense speedup slice should reduce graph-captured main
    verifier GPU work below the no-MTP baseline cost for two ordinary decode
    tokens; acceptance is no longer the first blocker for this prompt.
- KV-only shifted-row catch-up optimization:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-kvonly-catchup-c64-n16-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-kvonly-catchup-c64-n16-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-kvonly-catchup-c64-n16-combined-stats.csv`.
  - Added focused regression coverage in `V2_Unit_MTPGraphConstruction`
    proving `input.kv_cache_only=true` builds an MTP graph that stops at
    `layer0_kv_append` and omits `layer0_attention`, the MTP FFN residual,
    final norm, and LM head.
  - Focused validation passed:
    `ctest --test-dir build_v2_integration -R "^V2_Unit_MTPGraphConstruction$|^V2_Unit_PrefillDecodeTransition$" --output-on-failure --parallel`.
  - The combined CUDA Qwen3.6 dense run reached 497.72 ms for 16 decode
    tokens, 32.15 tok/s, about 0.78x the 41.39 tok/s no-MTP baseline and a
    small improvement over the prior 31.35 tok/s combined telemetry run.
  - MTP counters remain healthy for this prompt: 32 draft steps, 28 accepted,
    4 rejected, 4 rollbacks, and 87.5% acceptance.
  - Runtime state still proves the shifted-cache invariant:
    `current_position=20`, `mtp_cached_tokens=19`, `mtp_kv_cache_count=1`.
  - Graph-capture stats now separate the full sidecar and catch-up paths:
    main verifier replay is one 644-stage capturable segment at about
    46.05 ms/call, full decode sidecar replay is one 21-stage capturable
    segment at about 1.87 ms/call, and shifted-row catch-up replay is one
    12-stage capturable segment at about 0.20 ms/call.
  - This removes wasted catch-up work, but it does not change the main Phase 14
    blocker: the verifier graph is still too expensive relative to two
    ordinary decode tokens.
- Longer CUDA dense steady-state comparison:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-c128-n64-forwardgraph.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-c128-n64-forwardgraph-stats.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-c128-n64-forwardgraph-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-kvonly-catchup-c128-n64-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-kvonly-catchup-c128-n64-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-kvonly-catchup-c128-n64-combined-stats.csv`.
  - Same Qwen3.6 dense 27B Q4_K_S model on RTX 3090 `cuda:0`, deterministic
    4-token prompt, GPU graphs enabled, but with `-c 128` and `-n 64` so
    graph warmup/capture is amortized over more decode steps.
  - No-MTP baseline reached 1581.18 ms for 64 decode tokens, 40.48 tok/s.
  - MTP reached 1698.09 ms for 64 decode tokens, 37.69 tok/s, about 0.93x
    baseline.
  - MTP counters improved with the longer run: 128 draft steps, 124 accepted,
    4 rejected, 4 rollbacks, and 96.88% acceptance.
  - Runtime state remained aligned: `current_position=68`, `mtp_cached_tokens=67`,
    and `mtp_kv_cache_count=1`.
  - Baseline steady replay: `main_decode` has one 644-stage capturable segment,
    180 replay calls, about 24.53 ms/call.
  - MTP steady replay: `main_verifier` has one 644-stage capturable segment,
    81 replay calls, about 46.48 ms/call for two verifier tokens; the full
    sidecar remains about 1.88 ms/call and KV-only catch-up about 0.20 ms/call.
  - Conclusion: short-run graph-capture amortization was part of the deficit,
    but not all of it. Depth-1 MTP needs a faster two-token verifier path or
    deeper draft verification to cross the Phase 14 speedup threshold.
- Native CUDA M=2 verifier route speedup slice:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-m2native-c128-n64-forwardgraph.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-m2native-c128-n64-forwardgraph-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-c128-n64-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-c128-n64-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-c128-n64-combined-stats.csv`.
  - Same Qwen3.6 dense 27B Q4_K_S model on RTX 3090 `cuda:0`, deterministic
    4-token prompt, GPU graphs enabled, `-c 128`, `-n 64`, depth-1 MTP.
  - The same-binary no-MTP baseline reached 1582.53 ms for 64 decode tokens,
    40.44 tok/s.
  - The native M=2 verifier route reached 1184.69 ms for 64 decode tokens,
    54.02 tok/s, about 1.34x baseline.
  - MTP counters remained healthy: 128 draft steps, 124 accepted tokens,
    4 rejected tokens, 4 rollbacks, and 96.88% acceptance.
  - Runtime state stayed aligned: `current_position=68`, `mtp_cached_tokens=67`,
    and `mtp_kv_cache_count=1`.
  - Structured kernel counters prove the real model exercised the new route:
    `kernel.cuda_native_vnni_m2_calls` covered Q4_K/Q5_K/IQ-style verifier
    shapes such as `k=5120,n=17408`, `k=6144,n=5120`, and the LM-head-like
    `k=5120,n=248320`.
  - Main verifier replay improved to about 31.04 ms per two-token verifier,
    while sidecar replay stayed about 1.87 ms and shifted-row catch-up stayed
    about 0.20 ms. This is the first real CUDA SingleDevice dense speedup, but
    it is not yet the Phase 14 target speedup.
  - Regression coverage added/passed in `V2_Integration_CUDAGemmParity`:
    `Q4_K_VerifierSmallM_UsesNativeM2Route` proves the M=2 route is exercised,
    `Q4_0_SmallPrefillM4_UsesNativeSmallMRoute` catches the M=4 Q4_0
    small-prefill NaN issue found during the full smoke, and the full ctest
    target passed after routing `M=3/4` through native small-M GEMV.
- CUDA dense M=2 verifier stage-timing diagnostic:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-stagegpu-c128-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-stagegpu-c128-n16-stats.json`,
  and
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2native-stagegpu-c128-n16-stats.csv`.
  - Same Qwen3.6 dense 27B Q4_K_S model on RTX 3090 `cuda:0`,
    deterministic `Paris is` prompt, GPU graphs enabled, `-c 128`, `-n 16`,
    depth-1 MTP, with `LLAMINAR_GPU_STAGE_TIMING=1` and structured stats
    filter `mtp,forward_graph,stage_gpu,kernel`.
  - The diagnostic reached 46.93 decode tok/s with 32 draft steps, 28 accepted,
    4 rejected, 4 rollbacks, and 87.5% acceptance. Stage timing is intrusive,
    so this throughput is diagnostic evidence rather than the best-speed row.
  - Captured main-verifier replay remained the wall/GPU blocker:
    `forward_graph.segmented_replay_total` for `context=main_verifier`
    averaged about 30.54 ms per two-token graph. Sidecar replay averaged about
    1.88 ms and KV-only catch-up replay about 0.20 ms.
  - `stage_gpu` records are useful for stage attribution but do not represent
    captured replay timing. The `main_verifier` stage-event pass averaged about
    21.06 ms GPU time, led by GEMM at 5.82 ms, GDN projection at 4.43 ms,
    fused gate/up at 4.26 ms, short conv at 1.27 ms, GDN recurrence at
    1.05 ms, and fused QKV at 0.97 ms per recorded verifier pass.
  - This points the next CUDA dense optimization at real verifier graph GPU
    work, especially the remaining linear/GDN-heavy stages. Graph launch,
    final sync, sidecar execution, catch-up execution, and checkpoint
    import/export are not the primary 2x blocker in this slice.
- Rejected CUDA M=2 fused-projection overlap experiment:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2concurrent-c128-n64-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2concurrent-c128-n64-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-m2concurrent-c128-n64-combined-stats.csv`.
  - While experimenting, focused CUDA GEMM parity coverage passed for the
    existing Q4_K verifier `M=2` shapes and fused-SwiGLU down path, but the
    code was reverted because real-model improvement was neutral.
  - Same Qwen3.6 dense 27B Q4_K_S CUDA setup as the longer steady-state run:
    `-c 128`, `-n 64`, deterministic prompt, GPU graphs enabled, depth-1 MTP.
  - The experiment reached 1696.70 ms for 64 decode tokens, 37.72 tok/s,
    versus the retained KV-only catch-up result of 1698.09 ms and 37.69 tok/s.
  - MTP counters were unchanged: 128 draft steps, 124 accepted, 4 rejected,
    4 rollbacks, and 96.88% acceptance.
  - Main verifier replay remained about 46.48 ms per two-token verifier.
    This rules out simple fused-projection overlap as the live Phase 14
    speedup blocker; the next slice should target true two-row verifier
    kernel efficiency or deeper draft verification.
- Rejected post-sidecar checkpoint-elision experiment:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-reject-checkpoint-elide-c128-n64-combined.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-reject-checkpoint-elide-c128-n64-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-reject-checkpoint-elide-c128-n64-combined-stats.csv`.
  - The experiment removed the unconditional post-sidecar live-prefix
    checkpoint and, on verifier rejection or token-budget truncation, restored
    the initial checkpoint, reran the sidecar for the first accepted token, and
    then replayed accepted tokens.
  - Focused `V2_Unit_PrefillDecodeTransition` coverage passed while testing
    the idea, but real Qwen3.6 CUDA evidence rejected it.
  - Same `-c 128`, `-n 64`, deterministic Qwen3.6 dense CUDA setup reached
    1740.30 ms for 64 decode tokens, 36.78 tok/s, versus the retained
    1698.09 ms and 37.69 tok/s result.
  - Acceptance changed downward to 121 accepted / 7 rejected in the benchmark
    summary, versus the retained 124 accepted / 4 rejected result. Measured
    counters for the benchmarked section showed 96 initial logical checkpoints
    and 5 sidecar replays after rollback.
  - The likely correctness risk is terminal-hidden ownership: logical restore
    does not currently restore the exact terminal hidden row needed to rerun
    the sidecar after rollback. The retained safe contract keeps the
    post-sidecar checkpoint instead. Unit coverage now asserts that accepted
    and rejected MTP steps take the post-sidecar checkpoint and do not rerun
    the sidecar after rejection.
- Small-M verifier Phase 14 rerun:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-alln-c64-n4.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-alln-c64-n4-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-smallm-alln-c64-n4-stats.csv`.
  - Model/device/context: Qwen3.6 dense 27B Q4_K_S on RTX 3090 `cuda:0`,
    `-c 64`, `-n 4`, deterministic 5-token prompt, GPU graphs enabled.
  - MTP decode: 211.12 ms for 4 tokens, 18.95 tok/s, about 0.43x the
    43.97 tok/s no-MTP baseline.
  - MTP counters: 8 draft steps, 8 accepted, 0 rejected, 0 rollbacks,
    8 verifier runs, 16 verifier tokens, 100% acceptance.
  - High-level MTP timers in the benchmarked section: `decode_step_total`
    averaged about 105.55 ms, `verifier_forward` about 103.07 ms, and
    `sidecar_forward` about 1.96 ms.
  - Main verifier GPU stage timing averaged about 78.31 ms per verifier graph:
    GEMM 42.06 ms, fused gate/up 10.90 ms, GDN projection 7.73 ms,
    GDN recurrence 5.54 ms, KV append 4.75 ms, fused QKV 1.79 ms, and
    LM head 1.34 ms.
  - This improves the prior same-prompt MTP run from 10.92 to 18.95 tok/s,
    but the main verifier graph remains too expensive for Phase 14 acceptance.
- Prior same-prompt Phase 14 rerun before small-M verifier GEMV:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-c64-n4.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-gpugraphs-c64-n4.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-gpugraphs-c64-n4-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-gpugraphs-c64-n4-stats.csv`.
  - Model/device/context: Qwen3.6 dense 27B Q4_K_S on RTX 3090 `cuda:0`,
    `-c 64`, `-n 4`, deterministic 5-token prompt, GPU graphs enabled.
  - Baseline decode: 90.98 ms for 4 tokens, 43.97 tok/s.
  - MTP decode: 366.43 ms for 4 tokens, 10.92 tok/s, about 0.25x baseline.
  - MTP counters: 8 draft steps, 8 accepted, 0 rejected, 0 rollbacks,
    8 verifier runs, 16 verifier tokens, 100% acceptance.
  - High-level MTP timers in the benchmarked section: `decode_step_total`
    averaged about 183.21 ms, `verifier_forward` about 180.73 ms, and
    `sidecar_forward` about 1.95 ms.
  - Forward graph telemetry still reports zero manual segments. The sidecar
    decode graph replay is captured and small: one 21-stage capturable segment,
    about 1.85 ms/call total. The blocker is verifier graph work/replay cost.
- Longer current CUDA stress and tagged stage diagnostics:
  `/tmp/llaminar-mtp-bench/dense-cuda-current-baseline-c64-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-gpugraphs-c64-n16.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-gpugraphs-c64-n16-stats.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-context-tags-smoke.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-context-tags-smoke-stats.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-current-mtp-context-tags-smoke-stats.csv`.
  - The current `-n 16` baseline reached 41.34 decode tok/s; MTP reached
    only 6.85 decode tok/s, about 0.17x baseline.
  - The `-n 16` MTP counters were 32 draft steps, 12 accepted, 20 rejected,
    20 rollbacks, and 37.5% acceptance. Rejected drafts trigger
    `restore_live_prefix_state` and `replay_forward`; replay averaged about
    175 ms while checkpoint restore stayed below 1 ms.
  - The non-MTP baseline main decode graph replay averaged about 24.5 ms in
    the forward-graph stats, so the verifier path is paying roughly 7x the
    steady decode replay cost before acceptance quality is considered.
  - The tagged `stage_gpu` smoke adds `context=main_verifier` to verifier
    records and `context=mtp_decode_sidecar` /
    `context=mtp_shifted_prefill` to sidecar records. The regression unit
    `V2_Unit_StageTimeline` now proves these tags are exported on total,
    per-type, and per-stage records.
  - In the tagged smoke, main verifier GPU stage time averaged about
    118.1 ms per verifier graph. The largest per-type buckets were GEMM
    about 43.6 ms, GDN projection about 27.0 ms, fused gate/up about
    25.7 ms, fused QKV about 10.4 ms, and LM head about 3.0 ms per verifier
    graph. The sidecar decode record remained tiny by comparison.
- Baseline: `/tmp/llaminar-mtp-bench/dense-cuda-baseline-c64-n4.json`.
  - Model: `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf`.
  - Device/context: `cuda:0` RTX 3090, `-c 64`, deterministic 9-token
    prompt, `-n 4`, prefix disabled, MTP disabled.
  - Prefill 185.28 ms, 48.57 tok/s.
  - Decode 91.08 ms for 4 tokens, 43.92 tok/s.
- MTP graph run: `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4.json`.
  - Same model/device/context/prompt with `LLAMINAR_GPU_GRAPHS=1 --mtp
    --mtp-draft-tokens 1`.
  - Prefill 200.68 ms, 44.85 tok/s.
  - Decode 536.33 ms for 4 tokens, 7.46 tok/s.
  - MTP counters: 8 draft steps, 4 accepted tokens, 4 rejected tokens,
    4 rollbacks, 8 verifier runs, 16 verifier tokens, 50% acceptance.
- Graph stats: `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4-stats.json`
  and `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4-stats-bench.json`.
  - Decode segmented phase counters: 6 warmup, 3 capture, 21 replay.
  - Segmented decode plans reported 12 capturable segments and 0 manual
    segments; main verifier capture policy reported no collectives.
  - MTP decode sidecar replay is graph captured and small: one 21-stage
    capturable segment, about 1.86 ms/call total replay time.
  - This proves CUDA small-context graph capture is functioning, but it is not
    close to speedup-ready. Current best MTP decode throughput is about 0.17x
    the same-prompt baseline.
- Combined telemetry rerun:
  `/tmp/llaminar-mtp-bench/dense-cuda-baseline-c64-n4-shortprompt-bench.json`,
  `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4-combined-bench.json`,
  and `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4-combined-stats.json`.
  - Same 10-token prompt, `-c 64`, `-n 4`, and `LLAMINAR_GPU_GRAPHS=1`.
  - Baseline decode: 101.42 ms for 4 tokens, 39.44 tok/s.
  - MTP decode: 433.89 ms for 4 tokens, 9.22 tok/s, about 0.23x baseline.
  - MTP counters: 8 draft steps, 0 accepted, 8 rejected, 8 rollbacks,
    8 verifier runs, 16 verifier tokens, 0% acceptance.
  - High-level MTP timers: `decode_step_total` averaged about 216.93 ms,
    `verifier_forward` about 109.58 ms, `replay_forward` about 104.98 ms,
    and `sidecar_forward` about 1.30 ms.
  - Graph timers: MTP decode sidecar segmented replay remained graph-captured
    and small at about 1.83 ms/call for one 21-stage capturable segment.
  - The optimization target for this cell is not sidecar launch overhead. It is
    draft acceptance plus verifier/replay cost.
- Prompt/context hard-fail regression:
  `/tmp/llaminar-mtp-bench/dense-cuda-baseline-c64-defaultprompt-context-hardfail.json`.
  - Running `-c 64` with benchmark mode's default 595-token prompt now fails
    before prefill with structured `failure_reason`: `benchmark prompt has 595
    tokens but context length is 64; pass a shorter -p/--prompt or increase
    -c/--context-length`.
  - `V2_Unit_BenchmarkRunnerCPU` includes
    `FailsBeforePrefillWhenPromptExceedsContext`, proving the benchmark runner
    rejects oversized prompts before entering `forward()`.

Latest CUDA MoE evidence:

- Baseline: `/tmp/llaminar-mtp-bench/moe-cuda-baseline-c64-n4.json`.
  - Model: `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`.
  - Device/context: `cuda:0` RTX 3090, `-c 64`, deterministic 9-token
    prompt, `-n 4`, prefix disabled, MTP disabled.
  - Prefill 87.89 ms, 102.40 tok/s.
  - Decode 145.13 ms for 4 tokens, 27.56 tok/s.
- MTP graph run: `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4.json`.
  - Same model/device/context/prompt with `LLAMINAR_GPU_GRAPHS=1 --mtp
    --mtp-draft-tokens 1`.
  - Prefill 100.68 ms, 89.39 tok/s.
  - Decode 238.94 ms for 4 tokens, 16.74 tok/s.
  - MTP counters: 8 draft steps, 0 accepted tokens, 8 rejected tokens,
    8 rollbacks, 8 verifier runs, 16 verifier tokens, 0% acceptance.
- Graph stats: `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4-stats.json`
  and `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4-stats-bench.json`.
  - Decode segmented phase counters: 12 warmup, 3 capture, 18 replay.
  - Segmented decode plans included MoE router/expert stages, 0 manual
    segments, and no collectives in the single-device capture policy.
  - MTP decode sidecar replay is graph captured and small: one 24-stage
    capturable segment, about 0.98 ms/call total replay time.
  - This is a stable graph-captured small-context path, but it is not
    speedup-ready. Current best MTP decode throughput is about 0.61x the
    same-prompt baseline, driven by 0% draft acceptance and rollback/replay.
- Combined telemetry rerun:
  `/tmp/llaminar-mtp-bench/moe-cuda-baseline-c64-n4-shortprompt-bench.json`,
  `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4-combined-bench.json`,
  and `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4-combined-stats.json`.
  - Same 10-token prompt, `-c 64`, `-n 4`, and `LLAMINAR_GPU_GRAPHS=1`.
  - Baseline decode: 93.15 ms for 4 tokens, 42.94 tok/s.
  - MTP decode: 178.30 ms for 4 tokens, 22.43 tok/s, about 0.52x baseline.
  - MTP counters: 8 draft steps, 1 accepted, 7 rejected, 7 rollbacks,
    8 verifier runs, 16 verifier tokens, 12.5% acceptance.
  - High-level MTP timers: `decode_step_total` averaged about 89.13 ms,
    `verifier_forward` about 45.65 ms, `replay_forward` about 49.98 ms,
    and `sidecar_forward` about 0.95 ms.
  - Graph timers: MTP decode sidecar segmented replay remained graph-captured
    and small at about 0.78 ms/call for one 24-stage capturable segment.
  - As with dense CUDA, the optimization target is not sidecar graph replay.
    The blocker is acceptance quality plus verifier/replay cost.
- Native CUDA M=2 verifier route rerun:
  `/tmp/llaminar-mtp-bench/moe-cuda-current-baseline-m2native-c64-n16-forwardgraph.json`,
  `/tmp/llaminar-mtp-bench/moe-cuda-current-baseline-m2native-c64-n16-forwardgraph-stats.json`,
  `/tmp/llaminar-mtp-bench/moe-cuda-current-baseline-m2native-c64-n16-forwardgraph-stats.csv`,
  `/tmp/llaminar-mtp-bench/moe-cuda-current-mtp-m2native-c64-n16-combined.json`,
  `/tmp/llaminar-mtp-bench/moe-cuda-current-mtp-m2native-c64-n16-combined-stats.json`,
  and `/tmp/llaminar-mtp-bench/moe-cuda-current-mtp-m2native-c64-n16-combined-stats.csv`.
  - Same Qwen3.6 MoE 35B model on RTX 3090 `cuda:0`, deterministic
    4-token prompt, GPU graphs enabled, `-c 64`, `-n 16`, depth-1 MTP.
  - The same-binary no-MTP baseline reached 512.85 ms for 16 decode tokens,
    31.20 tok/s.
  - The native M=2 verifier route reached 314.42 ms for 16 decode tokens,
    50.89 tok/s, about 1.63x baseline.
  - MTP counters: 32 draft steps, 25 accepted tokens, 7 rejected tokens,
    7 rollbacks, and 78.12% acceptance.
  - Runtime state stayed aligned: `current_position=20`, `mtp_cached_tokens=19`,
    and `mtp_kv_cache_count=1`.
  - Structured kernel counters prove the real model exercised the native M=2
    route across MoE shapes, including Q8_0 and IQ-style codebooks:
    `codebook=19,k=2048,n=8192`, `codebook=8,k=2048,n=8192`,
    `codebook=8,k=4096,n=2048`, and the LM-head-like
    `codebook=8,k=2048,n=248320`.
  - Baseline `main_decode` segmented replay was about 30.09 ms for one
    graph-captured decode token. MTP `main_verifier` replay was about
    18.98 ms for a two-token verifier, with sidecar replay about 0.98 ms and
    shifted-row catch-up replay about 0.12 ms.
  - This proves SingleDevice CUDA MoE can benefit from the same native small-M
    verifier route as dense CUDA. It does not close ExpertParallel MoE or
    heterogeneous EP speedup gates.

Latest LocalTP ROCm dense evidence:

- Baseline: `/tmp/llaminar-mtp-bench/dense-localtp-rocm-baseline-c64-n4.json`.
  - Model: `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf`.
  - Domain: `--tp-devices rocm:0,rocm:1 -tp 2`, `-c 64`,
    deterministic 9-token prompt, `-n 4`, prefix disabled, MTP disabled.
  - Prefill 181.08 ms, 49.70 tok/s.
  - Decode 165.60 ms for 4 tokens, 24.15 tok/s.
- MTP graph run:
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-c64-n4.json`.
  - Same model/domain/context/prompt with `LLAMINAR_GPU_GRAPHS=1 --mtp
    --mtp-draft-tokens 1`.
  - Prefill 200.79 ms, 44.82 tok/s.
  - Decode 195.50 ms for 4 tokens, 20.46 tok/s.
  - MTP counters: 8 draft steps, 8 accepted tokens, 0 rejected tokens,
    0 rollbacks, 8 verifier runs, 16 verifier tokens, 100% acceptance.
- Graph stats:
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-c64-n4-stats.json`
  and `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-c64-n4-stats-bench.json`.
  - Both `ROCm:0` and `ROCm:1` verifier graphs report
    `has_collectives=true`, `allow_segmented=false`,
    `collective_segmented=false`, and `collectives_graph_capturable=false`.
  - This is correct and explicit, but not fully graph captured. The next
    LocalTP speedup gate is graph-safe RCCL/allreduce capture where ROCm
    supports it, or a deliberately optimized manual collective boundary.
  - Current best MTP decode throughput is about 0.85x the same-prompt
    baseline despite perfect acceptance.
- Post-guard graph-enabled/manual-collective run:
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-after-broad-bench.json`
  and `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-after-broad-stats.json`.
  - Same model/domain/context with explicit 7-token prompt, `-n 4`,
    `LLAMINAR_GPU_GRAPHS=1`, and `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0`.
  - Decode 222.91 ms for 4 tokens, 17.94 tok/s.
  - MTP counters: 8 draft steps, 8 accepted tokens, 0 rejected tokens,
    0 rollbacks, 8 verifier runs, 16 verifier tokens, 100% acceptance.
  - Both `ROCm:0` and `ROCm:1` verifier policy records report
    `has_collectives=true`, `allow_segmented=false`,
    `collective_segmented=false`, and `collectives_graph_capturable=false`.
  - Both MTP decode-sidecar policy records report the same collective/manual
    policy with `context=mtp_decode_sidecar`; shifted-prefill sidecar records
    also detect collectives and remain uncaptured.
- Current post-row-fix graph-enabled/manual-collective run:
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-baseline-rowfix-c64-n4.json`,
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-baseline-rowfix-c64-n4-stats.csv`,
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-mtp-rowfix-c64-n4.json`,
  and `/tmp/llaminar-mtp-bench/dense-localtp-rocm-current-mtp-rowfix-c64-n4-stats.csv`.
  - Same model/domain/context with prompt `Paris is`, `-n 4`,
    `LLAMINAR_GPU_GRAPHS=1`, and
    `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0`.
  - Baseline decode: 207.77 ms for 4 tokens, 19.25 tok/s.
  - MTP decode: 342.24 ms for 4 tokens, 11.69 tok/s, about 0.61x baseline.
  - MTP counters: 8 draft steps, 4 accepted tokens, 4 rejected tokens,
    4 rollbacks, 8 verifier runs, 16 verifier tokens, 50% acceptance.
  - The acceptance trace now shows correct verifier-row behavior: after
    accepting `{279, 6511}`, the next ready first token is `321`, and the
    second speculative token `7526` is rejected in favor of verifier token
    `1379`.
  - `main_verifier` and MTP sidecar policy records still report
    `has_collectives=true`, `allow_segmented=false`,
    `collective_segmented=false`, and `collectives_graph_capturable=false`
    on both ROCm participants.
  - Timers show the dominant costs remain verifier and replay:
    `verifier_forward` averaged about 109.03 ms, `replay_forward` about
    111.34 ms, and `sidecar_forward` about 2.60 ms per call.
- Forced segmented-collective reproducer:
  `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-segmented-hardfail.json`
  and `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-segmented-hardfail-stats.json`.
  - Command used `LLAMINAR_GPU_GRAPHS=1`,
    `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1`, `--tp-devices rocm:0,rocm:1`,
    `-tp 2`, `--mtp --mtp-draft-tokens 1`, `-c 64`, deterministic prompt,
    and `-n 4`.
  - Before the hard-fail guard, this command produced an HSA memory access
    fault during warmup and then `RankOrchestrator::forwardMTP` timed out
    after the 30000 ms participant timeout.
  - It now exits before sidecar launch with benchmark JSON
    `failure_reason`: `ROCm LocalTP MTP decode is incompatible with
    LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED; RCCL segmented collective replay
    for MTP sidecar execution is not implemented`.
  - `V2_Unit_PrefillDecodeTransition` covers the early MTP hard fail,
    `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPSmoke` includes
    `Qwen36ROCmLocalTPMTPSegmentedCollectiveHardFailsBeforeDraft` for the
    real-model LocalTP MTP path, and `V2_Integration_LocalTPContext` covers
    the RCCL segmented graph-policy reject reason.
- LocalTP all-position row-sampling regression:
  - A focused matrix rerun found that `LocalTP + prefix cache restore + MTP`
    could accept the first two tokens and then repeat the second token. The
    MTP acceptance trace showed `sampleGreedyFromAllPositionLogitsOnDevice(1)`
    returning the row-0 winner for the verifier terminal row.
  - Root cause: `DeviceSampler::sampleGreedyFromLocalInfos()` honored the
    requested row only for host tensors. GPU local-logit shards always passed
    the base `gpu_ptr` to backend argmax, so row 1 sampled row 0.
  - Fix: GPU sampling now offsets the device pointer by
    `row * vocab_local` after validating the tensor row count. The
    `V2_Unit_RankOrchestrator`
    `MultiChildAllPositionSamplingUsesRequestedColumnParallelShardRow` fixture
    now uses distinct row winners, and
    `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPPrefixSmoke` is the
    real Qwen3.6 ROCm LocalTP regression for prefix restore plus MTP token
    parity.
  - Focused validation passed on 2026-06-02:
    `ctest --test-dir build_v2_integration -R "^V2_Unit_PrefillDecodeTransition$|^V2_Unit_RankOrchestrator$|^V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTP" --output-on-failure --parallel`.

Latest NodeLocalTP CPU dense evidence:

- Baseline: `/tmp/llaminar-mtp-bench/dense-cpu-nodelocaltp-baseline-c64-n4-bench.json`.
  - Model: `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf`.
  - Domain: `-d cpu`, which self-launched 2 MPI ranks on one dual-socket
    node, with 28 OpenMP threads per rank, `-c 64`, the 10-token short prompt,
    and `-n 4`.
  - Prefill 265.74 ms, 37.63 tok/s.
  - Decode 408.61 ms for 4 tokens, 9.79 tok/s.
- MTP run:
  `/tmp/llaminar-mtp-bench/dense-cpu-nodelocaltp-mtp-c64-n4-bench.json`.
  - Same model/domain/context/prompt with `--mtp --mtp-draft-tokens 1`.
  - Prefill 367.04 ms, 27.25 tok/s.
  - Decode 550.70 ms for 4 tokens, 7.26 tok/s, about 0.74x baseline.
  - MTP counters: 8 draft steps, 4 accepted tokens, 4 rejected tokens,
    4 rollbacks, 8 verifier runs, 16 verifier tokens, 50% acceptance.
- Perf stats:
  `/tmp/llaminar-mtp-bench/dense-cpu-nodelocaltp-mtp-c64-n4-stats.json`.
  - Host/MPI TP collectives are exercised: the verifier policy reports
    `has_collectives=true`, `allow_segmented=false`,
    and `collectives_graph_capturable=false`.
  - `verifier_forward`: 6 calls, 947.63 ms total, about 157.94 ms/call.
  - `replay_forward`: 3 calls, 476.76 ms total, about 158.92 ms/call.
  - `sidecar_forward`: 6 calls, 65.75 ms total, about 10.96 ms/call.
  - Live checkpoint export/import is visible but secondary:
    `capture_live_prefix_state` averaged about 17.37 ms/call and
    `restore_live_prefix_state` about 17.47 ms/call.
  - This closes the previous pending CPU matrix cell for correctness and
    speed evidence. Like the GPU cells, the optimization target is verifier
    and replay cost plus draft acceptance, not sidecar launch overhead.

Latest LocalPP ROCm dense evidence:

- Baseline: `/tmp/llaminar-mtp-bench/dense-localpp-rocm-baseline-c64-n4.json`.
  - Model: `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf`.
  - Domain: `--define-domain stage0=rocm:0 --define-domain stage1=rocm:1
    --pp-stage 0=stage0:0-31 --pp-stage 1=stage1:32-63`, `-c 64`,
    deterministic 9-token prompt, `-n 4`, prefix disabled, MTP disabled.
  - Prefill 231.91 ms, 38.81 tok/s.
  - Decode 195.39 ms for 4 tokens, 20.47 tok/s.
- MTP hard-fail before the fix:
  `/tmp/llaminar-mtp-bench/dense-localpp-rocm-mtp-gpugraphs-c64-n4.json`.
  - Same model/domain/context/prompt with `LLAMINAR_GPU_GRAPHS=1 --mtp
    --mtp-draft-tokens 1`.
  - Warmup prefill reached stage 1 and failed with
    `Failed to populate MTP shifted prefill cache`, because PP stage 1 is
    invoked with transferred hidden state and no token-id pointer.
- MTP hard-fail after the regression fix:
  `/tmp/llaminar-mtp-bench/dense-localpp-rocm-mtp-gpugraphs-c64-n4-hardfail.json`.
  - Same command now fails before prefill forward with
    `MTP is not enabled for PP topologies; disable MTP or use a supported
    SingleDevice/TP topology`.
  - `V2_Unit_PrefillDecodeTransition` includes
    `MTPPPTopologyFailsBeforePrefillForward`, proving this remains a hard fail
    and does not enter runner `forward()` or a fallback decode path.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalPPHardFail` is the matching
    real Qwen3.6 ROCm LocalPP regression. It passed on 2026-06-02 alongside
    the LocalTP segmented-collective hard-fail and unit guard, and asserts the
    explicit unsupported-topology error plus zero MTP draft/verifier/rollback
    counters.
  - LocalPP MTP remains unimplemented; the next work is a proper PP-aware MTP
    shifted-prefill and verifier path, then graph-capture/manual-boundary
    analysis for PP activation transfers.

Latest Expert overlay 2x ROCm MoE evidence:

- Parity-suite coverage:
  `V2_Integration_Parity_Qwen36MoE_ExpertOverlay` now registers
  `MTPGreedyMatchesBaselineTokens_ROCm2TPHotOnly` and
  `PrefixCacheMTPRestore_ROCm2TPHotOnly`. On the current 32 GB ROCm cards these
  report an explicit prerequisite skip rather than failing late: the no-fallback
  hot-only plan covers all 256 routed experts and needs the resident expert
  payload plus the runner's 5% VRAM safety margin on each ROCm participant.
  The parity harness requires at least 40 GiB total VRAM per ROCm participant
  for this hot-only fixture because the 32 GiB cards repeatedly fail the real
  runner preflight after graph/model setup with only 8912/8994 MiB free.
  The no-fallback planner coverage hard-fail is pinned by
  `V2_Unit_MoEExpertParallelPlanner`. Focused CTest passed on 2026-06-02 for
  `V2_Integration_Parity_Qwen36MoE_ExpertOverlay_.*ROCm2TPHotOnly$`; both
  hot-only tests completed through the prerequisite path in about 1.3s each.
- Config and harness coverage:
  `configs/moe_overlay/rocm2_replicated_static.yaml` is a one-rank LocalTP
  `ReplicatedExperts` overlay domain over `rocm:0,rocm:1`, with all 256
  routed experts in a single ROCm tier and no fallback.
  `V2_Perf_MoEGraphNativeOverlay` now validates this config alongside the
  existing graph-native overlay benchmark configs.
- Real Qwen3.6 MoE smoke:
  `/tmp/llaminar-mtp-bench/moe-overlay-rocm2-replicated-streaming-hardfail.txt`.
  - Model: `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`.
  - Domain: `configs/moe_overlay/rocm2_replicated_static.yaml`,
    `--mpi-procs 1`, `-c 64`, deterministic prompt, `-n 1`.
  - The run reaches Qwen3.6 MoE graph config and excludes one trailing
    nextn/MTP block from the main graph: 41 layers to 40 layers.
  - Initialization hard-fails before inference during resident expert GPU
    preparation. ROCm:0 reports required 7386 MiB versus 7274 MiB available
    after the safety margin; ROCm:1 reports required 7386 MiB versus
    7356 MiB available after the safety margin.
  - Confirming with `LLAMINAR_WEIGHT_STREAMING=1` produces the corrected
    diagnostic: streaming is already enabled, but resident MoE expert
    streaming is not active for this GPU pipeline path.
  - No benchmark JSON is written because the failure occurs before benchmark
    execution. The next 2x ROCm EP slice must either reduce resident expert
    budget for this config or implement resident MoE expert streaming before
    MTP graph-capture evidence can be collected.

Latest Expert overlay 2x ROCm plus 2x CPU MoE evidence:

- Correctness parity note:
  `V2_Integration_Parity_Qwen36MoE_ExpertOverlay` passed on 2026-06-02 for
  both `MTPGreedyMatchesBaselineTokens_ROCm2TPHot_CPU2LocalTPCold` and
  `PrefixCacheMTPRestore_ROCm2TPHot_CPU2LocalTPCold`. That parity case uses the
  normal in-process ExpertOverlay plan from the Qwen3.6 parity harness. The
  blocked artifacts below are still valid for the separate benchmark CLI/config
  path and remain Phase 14 blockers, not Phase 12 correctness blockers.
- Config and harness coverage:
  `configs/moe_overlay/rocm2_cpu2_replicated_static.yaml` is a two-rank
  heterogeneous overlay config. Rank 0 owns a LocalTP `ReplicatedExperts`
  hot tier over `rocm:0,rocm:1`; rank 1 owns a LocalTP `ReplicatedExperts`
  CPU fallback tier over `cpu:0,cpu:1`.
  `V2_Perf_MoEGraphNativeOverlay` validates the config shape and command
  generation with `--mpi-procs 2`.
- Real Qwen3.6 MoE smoke:
  `/tmp/llaminar-mtp-bench/moe-overlay-rocm2-cpu2-endpoint-hardfail.txt`.
  - Model: `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`.
  - Domain: `configs/moe_overlay/rocm2_cpu2_replicated_static.yaml`,
    `--mpi-procs 2`, `-c 64`, deterministic prompt, `-n 1`.
  - Rank 1 hard-fails before inference:
    `MoE overlay rank 1 has role CpuFallbackParticipant but sidecar endpoint
    ranks were removed by graph-native MoE productionization`.
  - Rank 0 then reports cross-rank initialization failure at
    `buildComputeGraph`.
  - The next heterogeneous EP slice must implement graph-native CPU fallback
    participant workers and host-staged sparse dispatch/return through
    `TransferEngine` before MTP graph-capture or speedup evidence can be
    collected.

Latest ROCm MoE evidence:

- Baseline: `/tmp/llaminar-mtp-bench/moe-rocm-baseline-n4.json`.
  - Model: `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`.
  - Device/context: `rocm:0`, `-c 96`, deterministic 9-token prompt,
    `-n 4`, prefix disabled, MTP disabled.
  - Prefill 138.43 ms, 65.01 tok/s.
  - Decode 188.39 ms for 4 tokens, 21.23 tok/s.
- MTP graph-capture crash reproducer before the fix:
  - Same MoE model on `rocm:0` with `LLAMINAR_GPU_GRAPHS=1 --mtp
    --mtp-draft-tokens 1`.
  - `-n 1` and `-n 2` completed, while `-n 3` hit an HSA memory access fault
    after rollback restored live KV/MTP state and a previously captured HIP
    graph could replay against stale request-local state.
- Fixed MTP graph run:
  `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n3-fixed.json`.
  - `-c 64`, deterministic 9-token prompt, `-n 3`.
  - Prefill 164.98 ms, 54.55 tok/s.
  - Decode 351.66 ms for 3 tokens, 8.53 tok/s.
  - MTP counters: 8 draft steps, 0 accepted tokens, 4 rejected tokens,
    8 rollbacks, 8 verifier runs, 16 verifier tokens.
- Fixed MTP graph run:
  `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n4-fixed.json`.
  - `-c 64`, deterministic 9-token prompt, `-n 4`.
  - Prefill 174.64 ms, 51.53 tok/s.
  - Decode 367.46 ms for 4 tokens, 10.89 tok/s.
  - MTP counters: 8 draft steps, 0 accepted tokens, 8 rejected tokens,
    8 rollbacks, 8 verifier runs, 16 verifier tokens.
  - The crash fix preserves cached `ComputeGraph` objects but resets captured
    forward and MTP sidecar replay handles after live prefix restore/truncate,
    forcing a fresh warm/capture cycle across rollback boundaries.
  - This is a correctness and graph-capture stability improvement, not a
    speedup. Current best MoE MTP throughput is only about 0.51x the baseline
    decode throughput and has 0% acceptance on this prompt.
- Regression coverage:
  - `V2_Unit_MTPGraphConstruction` now includes
    `CPUSidecarGraphCacheRecordsPlainAfterBuildThenPlainReuse`, which proves a
    freshly built MTP sidecar graph records the first execution as
    `plain_after_build` before reuse.
  - The same suite includes
    `GPUSidecarGraphCacheRunsPlainBeforeSegmentedCapture`, which runs the tiny
    real GPU sidecar path when CUDA/ROCm is visible and asserts one
    `plain_after_build` execution before segmented graph-capture reuse. It also
    asserts the structured `sidecar_decode_capture_policy` record for the
    `mtp_decode_sidecar` context. The test skips only when no GPU backend is
    available to the test process.
  - `V2_Unit_PrefillDecodeTransition` includes
    `ROCmMTPAllowsGpuGraphsWithoutM2RowOverlap`, which proves the base ROCm
    MTP GPU-graph configuration reaches `forwardMTP()` when the experimental
    side-stream M=2 row-overlap path is disabled.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmSmoke` now pins the supported
    real Qwen3.6 ROCm dense MTP smoke to `LLAMINAR_GPU_GRAPHS=0`, and asserts
    exact token parity plus active MTP/verifier counters without requiring a
    brittle rollback count.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke` is the matching
    real-model regression for the base ROCm GPU-graph path. It runs with
    `LLAMINAR_GPU_GRAPHS=1`, keeps `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=0`, and
    asserts successful token generation plus active MTP/verifier counters.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPSmoke` keeps
    `LLAMINAR_GPU_GRAPHS=1` and disables only
    `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED`, proving the LocalTP
    graph-enabled/manual-collective path remains green with active MTP/verifier
    counters.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPPrefixSmoke` covers the
    LocalTP prefix-cache restore plus MTP path and catches all-position
    verifier terminal-row sampling regressions with real Qwen3.6 tokens.
  - `V2_Unit_PrefillDecodeTransition` also includes
    `ROCmMTPHardFailsWithM2RowOverlapUnderGpuGraphs`, which reproduces the
    unsafe `LLAMINAR_GPU_GRAPHS=1` plus `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=1`
    configuration and proves it fails before `forwardMTP()` can launch.
  - `V2_Integration_ROCmQuantisedGemmSmallM` covers the ROCm M=2 verifier GEMV
    path and the fused Q/K/V M=2 routing path against CPU/separate-projection
    references for both INT8-VNNI `Q8_0` and native-VNNI `Q4_K`, so future
    verifier-kernel work has a fast hardware regression for the dense Qwen3.6
    benchmark's quantization class.
  - The same focused integration test now graph-captures the supported
    native-VNNI `Q4_K` `M=2` dispatch path inside a HIP graph and compares the
    replayed output against a CPU FP32 reference, proving that future ROCm
    two-row verifier work has a small graph-safety regression before any real
    Qwen3.6 smoke is attempted.
  - The same focused integration test covers the opt-in
    `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=1` native-VNNI row-overlap path against
    the CPU reference before it is used in real-model benchmark experiments.
  - `V2_Unit_ForwardGraphTypes` includes
    `CapturedReplayPerfStatsIncludeContextTag`, which locks in structured
    replay context tags such as `main_verifier`, `mtp_decode_sidecar`, and
    `mtp_shifted_prefill`.
  - `V2_Unit_ForwardExecutionEngine` includes
    `AllPositionShortContinuationPublishesVerifierCacheLookupStats`, which
    locks in cache hit/miss stats for the all-position two-token verifier
    shape used by greedy MTP.
  - `V2_Unit_ForwardExecutionEngine` includes
    `ResetCapturedReplayStatePreservesCachedGraphs`, which locks in the
    rollback crash fix: live-state restore/truncate can drop captured GPU
    replay state without discarding the cached `ComputeGraph`.
  - `V2_Unit_ForwardGraphTypes` also includes
    `ReplayPhasePerfStatsSplitFinalStreamSync`, which locks in per-stream final
    sync timers for graph replay diagnostics.
  - `V2_Unit_ForwardExecutionEngine` now proves bucketed `runPrefillChunk()`
    carries the stable chunk ordinal into graph execution metadata.
  - `V2_Unit_DeviceGraphOrchestrator` now proves prefill graph observations use
    the same MoE domain id and domain-local participant id as the rebalance
    controller path.
  - `V2_Unit_ForwardExecutionEngine` also proves placement-changing
    chunk-boundary maintenance clears the outer bucketed forward graph cache so
    the next chunk rebuilds under the new placement epoch instead of replaying
    stale stage placement.
  - `V2_Integration_PrefillGraphCacheExecution_CUDA` and
    `V2_Integration_PrefillGraphCacheExecution_ROCm` now assert structured
    prefill graph snapshot/perf observability for chunk id, bucket length,
    real-token range, domain id, participant id, placement epoch, topology
    signature, capture/replay phase, and recapture reason while preserving
    padded-bucket graph reuse across changing real-token lengths.
  - The same focused CUDA/ROCm integration suites now include fixed-placement
    chunk schedules that reach captured replay and forced chunk-boundary
    rebalance schedules that clear placement-sensitive cache state, rebuild,
    recapture, and replay under the new placement epoch.

Next graph-capture questions:

- Can the two-token verifier use a true batched/two-row decode GEMV/GDN kernel
  inside the captured graph instead of serial M=1 row launches or prefill-style
  kernels?
- Can captured verifier replay reduce the capture-stream completion time by
  shrinking the captured two-token verifier graph, fusing M=2 kernels, or
  removing unnecessary work before logits sampling?
- For LocalTP/LocalPP/MoE, can collective/manual stages be graph captured or made
  explicit hard boundaries without falling back silently?

## 2026-06-01 Qwen3.6 Dense ROCm Slice

Hardware/topology:

- Model: `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf`
- Device: `rocm:0`
- Command shape: `./build_v2_release/llaminar2 benchmark -m <model> -d rocm:0 -n 16 ...`
- Benchmark mode: 1 warmup plus 3 measured iterations.
- Prompt length: 595 tokens.
- Decode length: 16 tokens.
- JSON artifacts:
  - `/tmp/llaminar_qwen36_dense_rocm_baseline.json`
  - `/tmp/llaminar_qwen36_dense_rocm_prefix_ram.json`
  - `/tmp/llaminar_qwen36_dense_rocm_mtp.json`
  - `/tmp/llaminar_qwen36_dense_rocm_prefix_ram_mtp.json`

| Scenario | Prefill ms | Decode ms | Total ms | Prefill tok/s | Decode tok/s | Overall tok/s | Key counters |
|----------|------------|-----------|----------|---------------|--------------|---------------|--------------|
| Baseline | 2619.80 | 862.65 | 3482.46 | 227.12 | 18.55 | 175.45 | prefix disabled, MTP disabled |
| RAM prefix cache | 84.93 | 865.65 | 950.59 | 7005.54 | 18.48 | 642.76 | 50 hits, 2975 matched tokens, 5 terminal hits, 340.20 MiB RAM |
| MTP greedy | 4275.88 | 8156.03 | 12431.92 | 139.15 | 1.96 | 49.15 | 32 draft steps, 64 accepted, 32 rejected, 32 rollbacks, 66.67% acceptance |
| RAM prefix cache plus MTP | 89.95 | 8198.35 | 8288.30 | 6614.57 | 1.95 | 73.72 | 50 hits, 2975 matched tokens, 5 terminal hits, 2.50 MiB MTP state, 66.67% MTP acceptance |

Observations:

- RAM prefix cache is performance-ready for this dense ROCm prompt class. Prefill improved from 2619.80 ms to 84.93 ms, a 30.8x prefill speedup, and total request time improved from 3482.46 ms to 950.59 ms, a 3.7x total speedup.
- Prefix counters explain the speedup: measured iterations restored 595-token full hits from RAM and restored terminal logits on the hit path.
- MTP greedy is correctness-ready but not performance-ready on this setup. Decode throughput regressed from 18.55 tok/s to about 1.96 tok/s even with 66.67% accepted draft tokens.
- Prefix plus MTP preserves the prefix-cache prefill speedup, but total request time is dominated by the current MTP decode overhead.

Follow-up blockers before MTP default enablement:

- Investigate why MTP sidecar execution is recorded under the prefill stage timeline and costs about 365 ms per small sidecar step on ROCm.
- Reduce MTP sidecar launch/graph overhead and verify the sidecar uses captured or fused graph paths where possible.
- Reduce rollback frequency or verifier replay cost. The trace shows 32 rollbacks for 32 draft steps, so accepted draft tokens are not translating into decode throughput.
- Keep MTP disabled by default until decode tok/s is faster than the baseline for the same model/backend/topology.

Telemetry fix landed with this benchmark slice:

- Benchmark JSON now computes top-level `mtp.acceptance_rate` as `accepted_tokens / (accepted_tokens + rejected_tokens)`, matching the per-request summary and keeping the value bounded between 0 and 1 when each draft step proposes multiple tokens.

## 2026-06-01 MTP Perf Stats Export Slice

Command shape:

```bash
LLAMINAR_PERF_STATS_JSON=/tmp/llaminar_qwen36_dense_rocm_mtp_perf_stats_phasefix.json \
LLAMINAR_PERF_STATS_CSV=/tmp/llaminar_qwen36_dense_rocm_mtp_perf_stats_phasefix.csv \
LLAMINAR_PERF_STATS_FILTER=mtp,forward_graph \
./build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf \
  -d rocm:0 -n 1 --mtp \
  --benchmark-json-output /tmp/llaminar_qwen36_dense_rocm_mtp_phasefix_benchmark.json
```

Key structured records:

| Record | Phase | Count | Total ms | Avg |
|--------|-------|-------|----------|-----|
| `terminal_hidden_row_select` | prefill | 1782 | 4234.15 | 2376.06 us |
| `sidecar_depth0_total` | prefill | 1782 | 1011.52 | 567.63 us |
| `capture_live_prefix_state` | decode | 3 | 1409.12 | 469.71 ms |
| `verifier_forward` | decode | 3 | 722.71 | 240.90 ms |
| `replay_forward` | decode | 3 | 215.66 | 71.89 ms |
| `restore_live_prefix_state` | decode | 3 | 167.19 | 55.73 ms |
| `sidecar_forward` | decode | 3 | 4.36 | 1.45 ms |

Conclusions:

- MTP sidecar decode itself is not the decode bottleneck on this run; checkpoint capture dominates decode wall time.
- MTP shifted-cache prefill is expensive because it performs per-token terminal hidden row selection and per-token depth-0 sidecar execution.
- Next optimization work should target logical checkpoint capture/restore and a batched/captured shifted-cache prefill path before tuning sidecar graph execution.

## 2026-06-02 MTP Matrix Telemetry Guardrail

The matrix now requires combined MTP plus graph perf exports for every new MTP
graph-capture benchmark:

```bash
LLAMINAR_PERF_STATS_FILTER=mtp,forward_graph
```

Regression coverage:

- `V2_Unit_PerfStatsCollector` includes
  `FlushFromEnvWritesMultipleFilteredDomainsForMTPGraphEvidence`, proving a
  single JSON/CSV export keeps both `mtp.*` high-level decode timers and
  `forward_graph.*` capture/replay timers while excluding unrelated domains.
- Future graph-captured CUDA/ROCm dense, MoE, LocalTP, and ExpertOverlay
  artifacts should use this combined filter before updating the speedup
  columns. Older graph-focused artifacts remain useful for capture-policy
  evidence, but reruns should include the high-level MTP timers in the same
  file.
