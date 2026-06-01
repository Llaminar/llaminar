# Prefix Cache And MTP Benchmark Notes

## Phase 14 GPU Graph Capture And MTP Speedup Matrix

This matrix is the durable scoreboard for Phase 14. Update it whenever a real
Qwen 3.6 dense or MoE benchmark changes one of these facts:

- baseline throughput without MTP,
- whether the MTP path is fully graph captured,
- whether collectives are graph captured where the backend technically supports it,
- best observed MTP throughput and speedup.

Use `Pending` only when no measured artifact exists yet. Use `Partial` when
some graph-capture path is active but the whole MTP inference step still has
manual stages, uncaptured collectives, or large host/replay overhead.

| Domain type | Device/backend target | Model class | Baseline decode tok/s | Graph-capture status | Collective capture status | Best MTP decode tok/s | Best MTP speedup | Evidence artifact | Current blocker |
|-------------|-----------------------|-------------|------------------------|----------------------|---------------------------|-----------------------|------------------|-------------------|-----------------|
| SingleDevice | ROCm `rocm:0` | Qwen3.6 dense 27B Q4_K_S | 18.25 | Dense MTP verifier and sidecar reach segmented-graph replay with no manual stages in longer decode runs | N/A | 14.81 | 0.81x decode, diagnostic `-n 4` run | `/tmp/llaminar-mtp-bench/dense-rocm-baseline-after.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-stagegpu-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-forward-cache-n8-release-stats.json` | Opt-in M=2 row-overlap plus perfect-acceptance short run reached 14.81 tok/s, but MTP is still slower than baseline. Main verifier replay is graph captured, yet replay final sync is about 112 ms/call on the `-n 8` run; need true two-row/batched verifier kernels and lower captured verifier GPU/sync time. |
| SingleDevice | CUDA | Qwen3.6 dense 27B Q4_K_S | Pending | Pending | N/A | Pending | Pending | Pending | CUDA:0 is visible but the default 4096-context dense 27B run does not fit: memory planner reports 26.7 GB required vs 23.3 GB available. Need an explicitly marked lower-context CUDA run, a smaller quant, or a larger CUDA device before filling this row. |
| SingleDevice | ROCm | Qwen3.6 MoE 35B | 21.23 | Partial: MTP GPU graphs now survive rollback/restore through the `-n 4` crash reproducer, but replay state is reset after each live-state rewind | N/A | 10.89 | 0.51x decode | `/tmp/llaminar-mtp-bench/moe-rocm-baseline-n4.json`, `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n3-fixed.json`, `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n4-fixed.json` | Crash fixed by resetting captured forward and MTP sidecar replay state after live prefix restore/truncate. MTP remains slower than baseline with 0% acceptance on this prompt; need sidecar/verifier acceptance and replay-cost work before claiming speedup. |
| SingleDevice | CUDA | Qwen3.6 MoE 35B | Pending | Pending | N/A | Pending | Pending | Pending | Need single-device MoE parity/perf run and CUDA availability check. |
| LocalTP | ROCm | Qwen3.6 dense 27B Q4_K_S | Pending | Pending | Target graph-capturable RCCL/allreduce segments where supported | Pending | Pending | Pending | Need TP-compatible dense MTP sidecar and verifier collectives in identical order. |
| LocalPP | ROCm | Qwen3.6 dense 27B Q4_K_S | Pending | Pending | PP activation transfers must be graph-capturable or explicit manual boundaries | Pending | Pending | Pending | Need local PP MTP verifier path and full graph-capture audit. |
| NodeLocalTP | CPU sockets | Qwen3.6 dense 27B Q4_K_S | Pending | N/A for GPU graphs | Host/MPI coordination only | Pending | Pending | Pending | Needed for correctness/speed evidence, but not GPU graph-capture gating. |
| Expert overlay EP | 2x ROCm | Qwen3.6 MoE 35B | Pending | Pending | Sparse dispatch/return graph capture required where ROCm supports it | Pending | Pending | Pending | Need graph-native sparse collectives and MoE MTP sidecar lockstep. |
| Expert overlay EP | 2x ROCm plus 2x CPU dual-socket | Qwen3.6 MoE 35B | Pending | Pending | Heterogeneous sparse collectives must be graph-aware with hard fail for unsupported legs | Pending | Pending | Pending | Need host-staged sparse return path through `TransferEngine`, then graph capture where possible. |

Latest ROCm dense evidence:

- Baseline: `/tmp/llaminar-mtp-bench/dense-rocm-baseline-after.json`.
  - Prefill 2620.36 ms, 227.07 tok/s.
  - Decode 1753.69 ms for 32 tokens, 18.25 tok/s.
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
  - This is the current best ROCm dense MTP evidence, but it is still below
    the 18.25 tok/s no-MTP decode baseline.
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
  - This is the current best ROCm dense MTP evidence, but it is still only
    about 0.65x the no-MTP decode baseline, so the next optimization axis must
    reduce verifier replay work rather than sidecar overhead.
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
    `plain_after_build` execution before segmented graph-capture reuse. The test
    skips only when no GPU backend is available to the test process.
  - `V2_Integration_ROCmQuantisedGemmSmallM` covers the ROCm M=2 verifier GEMV
    path and the fused Q/K/V M=2 routing path against CPU/separate-projection
    references for both INT8-VNNI `Q8_0` and native-VNNI `Q4_K`, so future
    verifier-kernel work has a fast hardware regression for the dense Qwen3.6
    benchmark's quantization class.
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
LLAMINAR_PERF_STATS_FILTER=mtp \
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
