# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice, LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay. Keep under 6 KB.

RAG: **G** correct and speed-positive, **A** correct but slow/stale,
**R** failing, speed-negative, or unproven. Fresh rows use bounded
`--decode-tokens 16 --perfstats`.

## Snapshot

Latest MoE run: `20260620T013557Z-moe-ordered-scatter-no-prezero`.
The verifier grouped path is correct on CUDA/ROCm, but SingleDevice MoE MTP
remains speed-negative. This slice removed the ordered-scatter prezero node from
CUDA/ROCm grouped verifier prefill. It is a small/noise-positive win, not a
phase-changing speedup.

Accepted MoE verifier route: routed experts use grouped verifier; shared expert
uses decode-equivalent GEMV-many plus normal shared-gate combine. Do not revive
the old combined routed+shared owner without strict L2, KLD, cosine, max_abs,
token, and continuation proof.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stoch | MoE greedy | MoE stoch | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused; symmetric tests required |
| SingleDevice | CUDA d1 | G | G | R | R | Dense green; MoE correct, speed-negative |
| SingleDevice | ROCm d1 | A | A | A | A/R | MoE correct, still below baseline |
| LocalTP | CUDA deg2 | R | R | R | R | Dense greedy accepts 0; stochastic unsupported |
| LocalTP | ROCm deg2 | R | R | R | R | Fixed d1 segfaults in LocalTP allreduce |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Skipped in latest refresh per CPU pause |
| ExpertOverlay | GPU hot + CPU cold | A | R | A | R | Skipped in latest refresh per CPU pause |

## SingleDevice Speeds

| Lane | Baseline | MTP | Acceptance | RAG |
|---|---:|---:|---:|:---:|
| CUDA dense greedy | `44.46` | `74.92 tok/s` d3 (`1.69x`) | n/a | G |
| CUDA dense stoch | `44.47` | `57.07 tok/s` d1 (`1.28x`) | n/a | G |
| ROCm dense greedy | `31.30` | `39.79 tok/s` dyn (`1.27x`) | n/a | A |
| ROCm dense stoch | `31.79` | `32.16 tok/s` dyn (`1.01x`) | n/a | A |
| CUDA MoE stoch | `138.23` | `88.47 tok/s` d3 (`0.64x`) | `30/39` | R |
| ROCm MoE stoch | `84.58` | `68.18 tok/s` d3 (`0.81x`) | `30/39` | A/R |

Latest MoE stage blockers:

| Device | Main verifier | Stage body | Largest buckets |
|---|---:|---:|---|
| CUDA | `445.1 ms` | `143.9 ms` | GEMM `35.3`, routed FFN `27.2`, shared FFN `25.4` |
| ROCm | `523.7 ms` | `167.8 ms` | routed FFN `47.7`, shared FFN `16.3`, router `15.2` |

## Focused Proofs

- `V2_Unit_MoEForbiddenDependencyScan` guards reset/workspace ownership,
  rejects the old combined verifier counter, and now requires CUDA/ROCm grouped
  prefill to skip output prezero only when ordered scatter owns all rows.
- CUDA/ROCm routed verifier microbench passed strict cos/L2/KLD/max_abs gates.
  ROCm M4: `0.1829 ms` graph vs `4.3160 ms` row replay. CUDA M4:
  `0.1048 ms` graph vs `4.0157 ms` row replay.
- CUDA/ROCm shared direct and `SharedExpertFFNStage` M=2/3/4 gates pass under
  strict metrics; the accepted shared route is GEMV-many, not shared-as-MoE.
- CUDA long-prompt MoE greedy parity and CUDA/ROCm stochastic verifier runs are
  green after pruning the broken combined owner.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Keep SingleDevice priority: dense first, then MoE, then LocalTP/LocalPP.
2. Attack MoE verifier producer economics directly. ROCm routed FFN and CUDA
   dense GEMM buckets are the next largest pieces.
3. Prefer grouped/concurrent decode-equivalent kernels over serial row replay.
4. Eliminate D2H/H2D bridges and graph rebuilds before polishing controllers.
5. After each concrete win: run strict focused parity, refresh tok/s, update
   this dashboard, and make a WiP commit.
