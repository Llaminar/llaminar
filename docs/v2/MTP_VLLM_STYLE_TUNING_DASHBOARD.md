# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice,
LocalTP, LocalPP, NodeLocalTP, ExpertOverlay. Keep under 6 KB.

RAG: **G** correct and speed-positive, **A** correct but slow/stale,
**R** failing, speed-negative, or unproven. Fresh speed rows use bounded
`--decode-tokens 16 --perfstats`.

## Current Snapshot

Dense SingleDevice CUDA is green. ROCm dense is correct but still needs a full
speed refresh. SingleDevice MoE correctness is restored after removing the
unaccepted combined routed+shared verifier owner from production graph wiring.
The accepted MoE verifier route is split branch-local math: routed experts use
the grouped verifier pipeline; the shared expert uses standalone
decode-equivalent GEMV-many plus the normal shared-gate combine.

Latest MoE stochastic fixed-d3 regression fix:
`benchmark_results/mtp_vllm_style/20260619T231915Z-moe-stochastic-split-verifier-regression-fix`.
Acceptance recovered to `30/39` (`76.92%`) on both CUDA and ROCm after the
combined verifier owner was hard-disabled in the graph and path guards were
updated to reject its counter.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stoch | MoE greedy | MoE stoch | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused; tests must stay symmetric |
| SingleDevice | CUDA d1 | G | G | R | R | Dense green; MoE correct, speed-negative |
| SingleDevice | ROCm d1 | A | A | A | A/R | MoE acceptance restored, still below baseline |
| LocalTP | CUDA deg2 | R | R | R | R | Dense greedy accepts 0; stochastic unsupported |
| LocalTP | ROCm deg2 | R | R | R | R | Fixed d1 segfaults in LocalTP allreduce |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm deg2 stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Skipped in latest refresh per CPU pause |
| ExpertOverlay | GPU hot + CPU cold | A | R | A | R | Skipped in latest refresh per CPU pause |

## Fresh SingleDevice MoE

| Device | Baseline | Stoch fixed d3 | Acceptance | Main blocker | RAG |
|---|---:|---:|---:|---|:---:|
| CUDA | `138.87 tok/s` | `75.15 tok/s` (`0.54x`) | `30/39` | verifier forward `539.7 ms`; sidecar `43.3 ms`; distribution build `21.7 ms` | R |
| ROCm | `84.81 tok/s` | `73.32 tok/s` (`0.86x`) | `30/39` | verifier forward `307.2 ms`; first-sidecar prelaunch `110.1 ms`; response wait `166.8 ms` | A/R |

Regression note: the bad run
`20260619T230340Z-moe-verifier-stage-refresh` collapsed acceptance to `2/45`
on both GPUs. Root cause was production re-promotion of the combined
routed+shared verifier owner despite existing strict full-model failures. The
guard tests now require routed grouped verifier plus shared GEMV-many and assert
that `mtp.moe_combined_decode_equivalent_verifier_prefill_rows` is absent.

## Fresh Dense SingleDevice

Evidence: `20260619T_dense_grouped_greedy_refresh/single_dense_gpu` and ROCm
bridge split `20260619T_rocm_dense_greedy_bridge_split`.

| Lane | Greedy | Stochastic | RAG |
|---|---:|---:|:---:|
| CUDA dense single | `44.46 -> 74.92 tok/s` (`1.69x`, d3) | `44.47 -> 57.07 tok/s` (`1.28x`, d1) | G |
| ROCm dense single | `31.30 -> 39.79 tok/s` (`1.27x`, dyn) | `31.79 -> 32.16 tok/s` (`1.01x`, dyn) | A |
| ROCm dense focused | fixed d3 `52.06 tok/s`, accepted `30/39` | not rerun | A |

ROCm dense bridge accounting is settled: compact D2H wait was about `0.36 ms`;
the active bottleneck is verifier producer time.

## Correctness Gates

- `V2_Unit_GpuWorkspaceAllocationPolicy` passed after the split-route guard.
- CUDA long-prompt MoE greedy parity passed after the fix:
  `MTPBenchmarkStyleDepth3LongPromptGreedyMatchesReference`.
- CUDA/ROCm path guards passed and now reject the combined verifier counter.
- CUDA/ROCm `MTPStochasticSamplingVerifierRuns` passed after the fix.
- Shared-expert `SharedExpertFFNStage` M=2/3/4 all-codebook gates pass on
  CUDA/ROCm with cosine, relative L2, KLD, and max_abs checks.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Keep SingleDevice priority. CUDA/ROCm MoE MTP is correct enough to optimize
   but not speed-accepted.
2. Attack full MoE verifier producer economics: routed expert FFN, shared FFN,
   router, GDN/attention, and any remaining serial verifier stage.
3. Do not revive the combined routed+shared owner without strict L2, KLD,
   cosine, max_abs, token, and continuation proof.
4. Prefer branch-side concurrent or fused decode-equivalent producers over graph
   edge shuffling.
5. After SingleDevice dense/MoE economics are healthy, return to LocalTP,
   LocalPP, NodeLocalTP, and ExpertOverlay rows.
