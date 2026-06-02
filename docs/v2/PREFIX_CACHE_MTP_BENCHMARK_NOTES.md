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
| SingleDevice | ROCm `rocm:0` | Qwen3.6 dense 27B Q4_K_S | 18.25 | Blocked: ROCm MTP with `LLAMINAR_GPU_GRAPHS=1` now hard-fails before decode-side MTP launch after real HSA fault/hang reproducers | N/A | 12.32 without GPU graphs; 11.66 standard post-guard | 0.68x decode best observed; 0.64x standard post-guard | `/tmp/llaminar-mtp-bench/dense-rocm-baseline-after.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-no-graphs-long-after-guard-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-notiming-bench.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-gpugraph-hardfail.json`, `/tmp/llaminar-mtp-bench/dense-rocm-mtp-m2rows-gpugraph-hardfail.json` | ROCm MTP GPU graph capture is not rollout-safe. Both the base graph path and the opt-in M=2 row-overlap path produced HSA memory faults/hangs in fresh real Qwen3.6 smokes and now fail with structured `failure_reason`. The supported no-graph path remains green after the guard, but still slower than baseline. Need a graph-safe ROCm MTP verifier/sidecar path, likely graph-native two-row verifier kernels plus safe replay state, before speedup work resumes for this cell. |
| SingleDevice | CUDA `cuda:0` | Qwen3.6 dense 27B Q4_K_S | 43.92 at `-c 64` | Small-context dense MTP reaches segmented replay with zero manual stages; default 4096-context run still does not fit this 24 GB device | N/A | 7.46 | 0.17x decode | `/tmp/llaminar-mtp-bench/dense-cuda-baseline-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-cuda-mtp-gpugraphs-c64-n4-stats.json` | CUDA graph capture is viable at small context, but MTP is much slower than baseline with 50% acceptance. Need verifier/rollback cost reduction and larger-context evidence on a GPU that fits the model. |
| SingleDevice | ROCm | Qwen3.6 MoE 35B | 21.23 | Partial: MTP GPU graphs now survive rollback/restore through the `-n 4` crash reproducer, but replay state is reset after each live-state rewind | N/A | 10.89 | 0.51x decode | `/tmp/llaminar-mtp-bench/moe-rocm-baseline-n4.json`, `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n3-fixed.json`, `/tmp/llaminar-mtp-bench/moe-rocm-mtp-gpugraphs-n4-fixed.json` | Crash fixed by resetting captured forward and MTP sidecar replay state after live prefix restore/truncate. MTP remains slower than baseline with 0% acceptance on this prompt; need sidecar/verifier acceptance and replay-cost work before claiming speedup. |
| SingleDevice | CUDA `cuda:0` | Qwen3.6 MoE 35B | 27.56 at `-c 64` | Small-context MoE MTP reaches segmented replay with zero manual stages; single-participant rebalance downgrades to observe | N/A | 16.74 | 0.61x decode | `/tmp/llaminar-mtp-bench/moe-cuda-baseline-c64-n4.json`, `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4.json`, `/tmp/llaminar-mtp-bench/moe-cuda-mtp-gpugraphs-c64-n4-stats.json` | CUDA MoE graph capture is stable at small context, but MTP is slower with 0% acceptance. Need MoE MTP acceptance quality and verifier/rollback cost reduction before speedup claims. |
| LocalTP | ROCm `rocm:0,rocm:1` | Qwen3.6 dense 27B Q4_K_S | 24.15 at `-c 64` | Not fully captured: verifier graphs detect collectives; forcing segmented collective replay for RCCL MTP now hard-fails before sidecar launch | RCCL collectives currently force non-captured verifier execution by default; `LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1` is blocked for ROCm LocalTP MTP until the sidecar path is made safe | 20.46 | 0.85x decode | `/tmp/llaminar-mtp-bench/dense-localtp-rocm-baseline-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-gpugraphs-c64-n4-stats.json`, `/tmp/llaminar-mtp-bench/dense-localtp-rocm-mtp-segmented-hardfail.json` | Correct LocalTP MTP with 100% acceptance is present, but the attempted RCCL segmented path produced an HSA memory fault before the guard. It now hard-fails with a structured `failure_reason`. Need real graph-safe RCCL/allreduce capture for MTP sidecar execution, or a deliberately optimized manual collective boundary, before speedup claims. |
| LocalPP | ROCm `stage0=rocm:0, stage1=rocm:1` | Qwen3.6 dense 27B Q4_K_S | 20.47 at `-c 64` | Blocked: MTP is a hard fail before prefill on PP topologies | PP activation transfers are present in the baseline path, but MTP graph capture is not attempted | Blocked | N/A | `/tmp/llaminar-mtp-bench/dense-localpp-rocm-baseline-c64-n4.json`, `/tmp/llaminar-mtp-bench/dense-localpp-rocm-mtp-gpugraphs-c64-n4-hardfail.json` | PP MTP shifted-prefill and verifier execution are not implemented. The previous late stage-1 shifted-cache failure is now a prefill hard-fail with an explicit unsupported-topology message. |
| NodeLocalTP | CPU sockets | Qwen3.6 dense 27B Q4_K_S | Pending | N/A for GPU graphs | Host/MPI coordination only | Pending | Pending | Pending | Needed for correctness/speed evidence, but not GPU graph-capture gating. |
| Expert overlay EP | 2x ROCm | Qwen3.6 MoE 35B | Blocked before inference | Blocked before graph-capture measurement | Sparse dispatch/return graph capture required where ROCm supports it, but the current run blocks during resident expert preparation first | Blocked | N/A | `/tmp/llaminar-mtp-bench/moe-overlay-rocm2-replicated-streaming-hardfail.txt` | Added `configs/moe_overlay/rocm2_replicated_static.yaml` and harness coverage for a one-rank LocalTP `ReplicatedExperts` ROCm domain. Real Qwen3.6 MoE reaches graph config, but both ROCm devices fail resident expert VRAM preflight by the safety-margin check. `LLAMINAR_WEIGHT_STREAMING=1` is already enabled in the confirming run, but resident MoE expert streaming is not active for this GPU pipeline path. |
| Expert overlay EP | 2x ROCm plus 2x CPU dual-socket | Qwen3.6 MoE 35B | Blocked before inference | Blocked before graph-capture measurement | Heterogeneous sparse collectives must be graph-aware, but the current run stops before sparse dispatch because CPU fallback participant ranks no longer have sidecar endpoint runners | Blocked | N/A | `/tmp/llaminar-mtp-bench/moe-overlay-rocm2-cpu2-endpoint-hardfail.txt` | Added `configs/moe_overlay/rocm2_cpu2_replicated_static.yaml` and harness coverage for a rank-0 ROCm LocalTP hot tier plus rank-1 CPU LocalTP fallback tier. Real Qwen3.6 MoE rank 1 hard-fails as `CpuFallbackParticipant` because sidecar endpoint ranks were removed by graph-native MoE productionization. Next slice needs proper graph-native CPU fallback participant workers and sparse return through `TransferEngine`, not a fallback sidecar. |

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
- GPU-graph hard-fail for base ROCm MTP:
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-gpugraph-hardfail.json` and
  `/tmp/llaminar-mtp-bench/dense-rocm-mtp-gpugraph-hardfail-stats.json`.
  - A fresh real Qwen3.6 ROCm smoke with `LLAMINAR_GPU_GRAPHS=1` and no
    `LLAMINAR_ROCM_CONCURRENT_M2_ROWS` produced an HSA memory access fault
    during warmup and then left the self-launched benchmark process stuck.
  - The configuration now fails before decode-side MTP launch with structured
    `failure_reason`: `ROCm MTP decode is incompatible with
    LLAMINAR_GPU_GRAPHS=1; ROCm MTP sidecar/verifier graph capture can fault
    during warmup and is disabled until the path is graph-safe`.
  - The stats artifact proves shifted-prefill sidecar graph capture still ran
    safely before the hard fail: `sidecar_decode_capture_policy` records
    `context=mtp_shifted_prefill`, `allow_segmented=true`, `has_collectives=false`,
    and four segmented sidecar reuse attempts before the verifier guard fired.
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

Latest CUDA dense evidence:

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
  - `V2_Unit_PrefillDecodeTransition` covers the early MTP hard fail, and
    `V2_Integration_LocalTPContext` covers the RCCL segmented graph-policy
    reject reason.

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
  - LocalPP MTP remains unimplemented; the next work is a proper PP-aware MTP
    shifted-prefill and verifier path, then graph-capture/manual-boundary
    analysis for PP activation transfers.

Latest Expert overlay 2x ROCm MoE evidence:

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
    `ROCmMTPHardFailsWithGpuGraphsBeforeSidecarLaunch`, which reproduces the
    unsafe base ROCm MTP GPU-graph configuration and proves it fails before
    `forwardMTP()` can launch.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmSmoke` now pins the supported
    real Qwen3.6 ROCm dense MTP smoke to `LLAMINAR_GPU_GRAPHS=0`, and asserts
    exact token parity plus active MTP/verifier counters without requiring a
    brittle rollback count.
  - `V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsHardFail` is the
    matching real-model regression for the HSA fault/hang class. It runs with
    `LLAMINAR_GPU_GRAPHS=1`, expects the structured hard-fail before decode-side
    MTP launch, and asserts no MTP verifier counters were incremented.
  - `V2_Unit_PrefillDecodeTransition` also includes
    `ROCmMTPHardFailsWithM2RowOverlapUnderGpuGraphs`, which reproduces the
    unsafe `LLAMINAR_GPU_GRAPHS=1` plus `LLAMINAR_ROCM_CONCURRENT_M2_ROWS=1`
    configuration and proves it fails before `forwardMTP()` can launch.
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
