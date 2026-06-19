# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA, ROCm, CPU across SingleDevice,
LocalTP, LocalPP, NodeLocalTP, and ExpertOverlay. Keep this file under 6 KB.

RAG: **G** = correct and speed-positive. **A** = correct but slow/stale.
**R** = failing, speed-negative, or unproven. Fresh speed rows are bounded
`--decode-tokens 16 --perfstats`.

## Current Snapshot

Dense SingleDevice is the active focus. CUDA dense greedy now uses grouped
device-resident publication and is speed-positive; ROCm dense greedy is correct
and positive but lags CUDA by speedup factor. ROCm stochastic is still blocked
by the compact outcome host bridge/wait path. MoE remains next after dense
SingleDevice is green on both GPU backends.

Strict correctness gate update: dense CUDA/ROCm grouped verifier rows M=1..4
now pass full-distribution checks with cosine, relative L2, symmetric KLD,
max-abs error, and sampled-token equality. Token equality alone is not accepted
as parity for verifier row promotion.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stochastic | MoE greedy | MoE stochastic | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused; keep correctness gates symmetric |
| SingleDevice | CUDA d1 | G | G | R | A | Dense green; MoE speed still pending |
| SingleDevice | ROCm d1 | A | A | R | R | Dense correct but ROCm bridge/wait hurts speed |
| LocalTP | CUDA deg2 | R | R | R | R | Dense greedy accepts 0; stochastic unsupported |
| LocalTP | ROCm deg2 | R | R | R | R | Fixed d1 segfaults in LocalTP allreduce |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm deg2 stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Skipped in latest refresh per CPU pause |
| ExpertOverlay | GPU hot + CPU cold | A | R | A | R | Skipped in latest refresh per CPU pause |

## Fresh Dense SingleDevice Numbers

Evidence:
`benchmark_results/mtp_vllm_style/20260619T_dense_grouped_greedy_refresh/single_dense_gpu`.

Values are `baseline; d1/d2/d3/dyn tok/s`, with best MTP speedup.

| Lane | Greedy | Stochastic | RAG |
|---|---:|---:|:---:|
| CUDA dense single | `44.46; 58.82/70.61/74.92/72.05` (`1.69x`) | `44.47; 57.07/55.44/53.08/53.29` (`1.28x`) | G |
| ROCm dense single | `31.30; 33.78/34.08/39.74/39.79` (`1.27x`) | `31.79; 32.22/25.51/23.50/32.16` (`1.01x`) | A |

Dense greedy acceptance is healthy: CUDA d3 accepts `31/40` with zero rollbacks;
ROCm d3 accepts `30/39` with zero rollbacks. CUDA d3 publish cost is about
`11.0 ms` and compact D2H wait is negligible. ROCm d3 publish cost is about
`16.2 ms`, but the request-summary/outcome wait still drains about `586 ms`
over the short run. ROCm stochastic d1/dyn both spend about `895 ms` in the
host bridge and are only break-even despite 66.7% acceptance.

## Stale Broader Numbers

Old full-dashboard rows remain useful as blockers, not as green evidence:

| Lane | Best recent speed | RAG |
|---|---:|:---:|
| CPU dense single | `0.84x greedy`, `0.81x stochastic d1` | R |
| CUDA MoE single | `0.38x greedy`, `0.93x stochastic` | R |
| ROCm MoE single | `0.47x greedy`, `0.79x stochastic` | R |
| CUDA2 LocalTP dense | `0.42x greedy`, 0% accept | R |
| ROCm2 LocalTP dense | fixed d1 segfault | R |
| ROCm2 LocalPP dense | `0.67x greedy`, `0.60x stochastic` | R |

## Correctness Evidence

- `V2_Unit_PrefillDecodeTransition` passed after grouped greedy resident publication.
- Dense CUDA/ROCm actual MTP parity subset passed: fixed d1/d3, dynamic,
  benchmark prompt, and first-transaction state.
- Dense CUDA/ROCm grouped verifier rows passed:
  `VerifierRowsGroupedDecodeEquivalentM[1-4]` with strict cosine, relative L2,
  symmetric KLD, max-abs, and token gates.
- `V2_Perf_DenseVerifierRows_(CUDA|ROCm)` passed.

## Next Phase 10 Moves

1. Catch ROCm dense up to CUDA: remove or overlap the compact outcome/request
   summary host wait on greedy and stochastic lanes.
2. Re-run the bounded dense GPU matrix after the ROCm bridge work; keep CUDA
   dense as the ratchet.
3. Move to SingleDevice MoE only after dense CUDA/ROCm are both green.
4. Fix LocalTP/LocalPP correctness after SingleDevice dense/MoE economics are
   healthy.
5. Keep routed/shared MoE fusion separate unless strict parity proves L2, KLD,
   cosine, max-abs, and token equivalence.
