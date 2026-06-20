# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice, LocalTP,
LocalPP, NodeLocalTP, ExpertOverlay. Keep under 6 KB.

RAG: **G** correct and speed-positive, **A** correct but slow/stale,
**R** failing, speed-negative, or unproven. Fresh speed rows use bounded
`--decode-tokens 16 --perfstats`.

## Current Snapshot

Dense SingleDevice CUDA is green. ROCm dense is correct but still needs a full
speed refresh. SingleDevice MoE correctness is restored after removing the
unaccepted combined routed+shared verifier owner from production graph wiring.
Accepted MoE verifier route: routed experts use grouped verifier; shared expert
uses decode-equivalent GEMV-many plus normal shared-gate combine.

Latest MoE stochastic fixed-d3 refresh:
`20260620T011447Z-moe-stochastic-rocm-down-kpart`.
Acceptance stayed at `30/39` (`76.92%`) on both GPUs. CUDA shared gate/up
side-stream overlap remains the accepted win. A ROCm split-K down experiment
passed strict microbench gates but did not improve full-model throughput, so it
was not promoted. The new reset guard prevents request reset from clearing
declared ROCm grouped verifier workspace used by captured graphs.

Focused verifier FFN gates remain green. Routed M4 grouped verifier is
`0.1062 ms` vs `9.6544 ms` row replay (`90.9x`); production
`SharedExpertFFNStage` M=2/3/4 all-codebook gates are exact under
cos/L2/KLD/max_abs and speed-positive (`4.5x-8.1x`).

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
| CUDA | `138.26 tok/s` | `88.37 tok/s` (`0.64x`) | `30/39` | verifier `445.9 ms`; graph replay `181.8 ms`; stage body `144.4 ms`; dist build `19.8 ms` | R |
| ROCm | `83.36 tok/s` | `67.04 tok/s` (`0.80x`) | `30/39` | verifier `522.5 ms`; graph replay `198.9 ms`; stage body `168.5 ms`; dist build `58.3 ms` | A/R |

Regression note: `20260619T230340Z-moe-verifier-stage-refresh` collapsed
acceptance to `2/45` on both GPUs. Root cause was re-promotion of the combined
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
- `V2_Unit_MoEForbiddenDependencyScan` now guards ROCm MoE reset/workspace
  ownership so request reset cannot drop captured grouped workspace addresses.
- CUDA long-prompt MoE greedy parity passed after the fix:
  `MTPBenchmarkStyleDepth3LongPromptGreedyMatchesReference`.
- CUDA/ROCm path guards passed and now reject the combined verifier counter.
- CUDA/ROCm `MTPStochasticSamplingVerifierRuns` passed after the fix.
- `V2_Unit_MoERuntimeTable` proves histogram reset preserves placement banks.
- `V2_Unit_PrefillGraphCapturability` rejects shared-gate graph capture
  until the effective gate tensor is device-resident on the stage device.
- `V2_Unit_MoEExpertComputeStage` now proves CUDA M=2..4 shared-expert
  verifier rows declare side-stream GEMV partial workspace structurally.
- Shared-expert `SharedExpertFFNStage` M=2/3/4 all-codebook gates pass on
  CUDA/ROCm with cosine, relative L2, KLD, and max_abs checks.
- The experimental lower-level shared-as-MoE prefill route is not accepted and
  has been pruned from perf gates; the accepted shared route is GEMV-many via
  production `SharedExpertFFNStage`.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Keep SingleDevice priority. CUDA/ROCm MoE MTP is correct enough to optimize
   but not speed-accepted.
2. Attack full MoE verifier producer economics. Latest CUDA stage buckets are
   GEMM `35.2 ms`, routed FFN `27.8 ms`, shared FFN `25.3 ms`, GDN projection
   `17.5 ms`; latest ROCm buckets are routed FFN `48.7 ms`, router `15.1 ms`,
   GDN projection `13.4 ms`, attention/recurrence about `10.8 ms`.
3. Do not revive the combined routed+shared owner without strict L2, KLD,
   cosine, max_abs, token, and continuation proof.
4. Prefer branch-side concurrent or fused decode-equivalent producers over graph
   edge shuffling.
5. After SingleDevice dense/MoE economics are healthy, return to LocalTP,
   LocalPP, NodeLocalTP, and ExpertOverlay rows.
