# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice,
LocalTP, LocalPP, NodeLocalTP, ExpertOverlay. Keep under 6 KB.

RAG: **G** correct and speed-positive, **A** correct but slow/stale,
**R** failing, speed-negative, or unproven. Fresh speed rows use bounded
`--decode-tokens 16 --perfstats`, with GPU timing only for diagnosis.

## Current Snapshot

Dense SingleDevice is the only green production-speed lane. CUDA dense greedy
uses grouped device-resident publication and is speed-positive; ROCm dense has
good focused wins but needs a full refresh and stochastic cleanup.

SingleDevice MoE shared-expert M=2..4 verifier kernels are wired through a safe
composite route: routed grouped verifier plus decode-equivalent shared FFN. The
broken single-table routed+shared prefill shortcut has been removed, not merely
disabled. Full MoE MTP remains red/amber because main verifier producer work
still dominates.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stochastic | MoE greedy | MoE stochastic | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused; keep tests symmetric |
| SingleDevice | CUDA d1 | G | G | R | R | Dense green; MoE speed-negative after shared route |
| SingleDevice | ROCm d1 | A | A | A | R | MoE greedy improved but still below baseline |
| LocalTP | CUDA deg2 | R | R | R | R | Dense greedy accepts 0; stochastic unsupported |
| LocalTP | ROCm deg2 | R | R | R | R | Fixed d1 segfaults in LocalTP allreduce |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm deg2 stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Skipped in latest refresh per CPU pause |
| ExpertOverlay | GPU hot + CPU cold | A | R | A | R | Skipped in latest refresh per CPU pause |

## Fresh SingleDevice MoE

Evidence: `benchmark_results/mtp_vllm_style/20260619T195526Z_moe_split_branch_dependency`.
Values are same-run baseline and fixed d3 stochastic after the split-branch
dependency cleanup.

| Device | Stochastic fixed d3 | Main blocker | RAG |
|---|---:|---|:---:|
| CUDA | `139.3 -> 75.1 tok/s` (`0.54x`) | verifier graph `234 ms`; shared `40 ms`, GEMM `34 ms`, GDN proj `29 ms` | R |
| ROCm | `84.0 -> 69.9 tok/s` (`0.83x`) | verifier graph `198 ms`; routed `48 ms`, shared `15 ms`, GDN/attention `~9-13 ms` | A/R |

Acceptance was not the primary failure: CUDA stochastic d3 accepted `30/39`
tokens (`76.9%`), ROCm stochastic d3 also accepted `30/39`. The remaining gap
is grouped verifier producer economics. The next fix needs branch-side
concurrency or a decode-equivalent fused producer; deleting graph edges alone
does not move CUDA enough.

## Fresh Dense SingleDevice

Evidence: `benchmark_results/mtp_vllm_style/20260619T_dense_grouped_greedy_refresh/single_dense_gpu`
and focused ROCm bridge split `20260619T_rocm_dense_greedy_bridge_split`.

| Lane | Greedy | Stochastic | RAG |
|---|---:|---:|:---:|
| CUDA dense single | `44.46 -> 74.92 tok/s` (`1.69x`, d3) | `44.47 -> 57.07 tok/s` (`1.28x`, d1) | G |
| ROCm dense single | `31.30 -> 39.79 tok/s` (`1.27x`, dyn) | `31.79 -> 32.16 tok/s` (`1.01x`, dyn) | A |
| ROCm dense focused | fixed d3 `52.06 tok/s`, accepted `30/39` | not rerun | A |

ROCm dense bridge accounting is settled: the old D2H-sync bucket was mostly
response-ready wait. Actual compact D2H wait was about `0.36 ms`; the active
bottleneck is verifier producer time.

## Correctness Evidence

- CUDA/ROCm MoE stochastic reuse plus grouped verifier M=2/3/4 focused parity is
  green. The CUDA reuse regression was stale stochastic top-k slot metadata
  surviving `clearCache()`; `V2_Unit_GpuWorkspaceAllocationPolicy` now guards
  the reset contract.
- `V2_Unit_GpuWorkspaceAllocationPolicy` passed after promoting the standalone
  grouped shared-expert policy wording.
- `V2_Perf_MoEVerifierPrefill` passed after the shared-expert wiring refresh;
  deleted shortcut perf rows must stay gone. Latest isolated grouped speedups:
  CUDA routed M2/M3/M4 `63.5/75.9/89.7x`, CUDA shared `49.7/72.0/88.8x`;
  ROCm routed `35.5/24.7/33.0x`, ROCm shared `17.5/21.9/19.8x`.
- CUDA full `qwen36` dispatch refresh validated `514,520` rows and `640/640`
  known-shape coverage; not installed because the checked-in broad table has
  lower mean penalty (`0.44%` vs `0.49%`).
- Shared-expert `SharedExpertFFNStage` M=2/3/4 all-codebook gates pass on
  CUDA/ROCm with cosine, relative L2, KLD, and max_abs checks.
- Dense CUDA/ROCm grouped verifier rows M=1..4 pass full-distribution checks:
  cosine, relative L2, symmetric KLD, max_abs, and sampled-token equality.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Keep SingleDevice priority. CUDA/ROCm MoE MTP is correct enough to optimize
   but not speed-accepted.
2. Attack full MoE verifier producer economics: routed expert FFN, shared FFN,
   router, GDN/attention, and any remaining serial verifier stage. Prefer a
   branch-side concurrent or fused decode-equivalent producer over graph-edge
   shuffling.
3. Do not revive the failed single-table routed+shared shortcut without strict
   L2, KLD, cosine, max_abs, token, and continuation proof.
4. Keep NativeVNNI/GEMV policy generated by aspect/work buckets, with exact
   shapes only as overlays.
5. After SingleDevice dense/MoE economics are healthy, return to LocalTP,
   LocalPP, NodeLocalTP, and ExpertOverlay rows.
