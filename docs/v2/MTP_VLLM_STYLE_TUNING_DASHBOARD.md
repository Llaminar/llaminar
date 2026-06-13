# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA, ROCm, and CPU across SingleDevice,
LocalTP, LocalPP, NodeLocalTP, and ExpertOverlay. Keep this file near 6KB.

RAG: **G** = correct and speed-positive near target. **A** = correct but slow,
partial, policy-sensitive, or missing fresh same-run numbers. **R** = failing,
speed-negative, or unproven. Every iteration refreshes no-MTP, fixed d1/d2/d3,
and dynamic for the active lane; broad units plus touched parity pass before a
WiP commit.

## Phase 10 Snapshot

Active blocker: MoE stochastic. Dense lanes are generally accepted, but MoE
stochastic still spends too much time in verifier, condition replay, and device
outcome publication.

Fresh diagnostic run:
`benchmark_results/mtp_vllm_style/20260613T_phase10_moe_stochastic_gpu_stage_deep`

| Lane | Baseline | Best MTP | RAG | Evidence |
|---|---:|---:|:---:|---|
| CUDA MoE stochastic d3 | 138.6 | 137.1 | R | stage-timed; verifier 808 ms, condition 255 ms |
| ROCm MoE stochastic d3 | 77.7 | 84.6 | A | stage-timed only; verifier 1287 ms, condition 357 ms, outcome/D2H 265 ms |

Guarded bonus/row-counter run:
`benchmark_results/mtp_vllm_style/20260613T_phase10_lazy_bonus_moe_stochastic`

| Lane | Baseline | d2 | d3 | Wasted rows |
|---|---:|---:|---:|---:|
| CUDA MoE stochastic | 138.8 | 125.8 (0.91x) | 133.8 (0.96x) | d2 7/132, d3 19/157 |
| ROCm MoE stochastic | 77.5 | 56.5 (0.73x) | 70.9 (0.92x) | d2 50/160, d3 51/180 |

2026-06-13 instrumentation slice: `stochastic_device_physical_verify_rows`,
`stochastic_device_semantic_verify_rows`, and
`stochastic_device_post_reject_rows` now separate real verifier rows from rows
past the first rejection. The standard matrix `summary.tsv` now includes these
as `stochastic_*_verify_rows`, so future dashboard refreshes do not require
manual perfstats extraction. The RB=2 unit regression pins scalar-equivalent
accept/residual/bonus RNG positions and row accounting. The guarded
device-side bonus sampler is graph-captured and skips full-vocab bonus sampling
when a batch rejects before the bonus row, but ROCm is still dominated by
summary/D2H sync rather than bonus-row work.

2026-06-13 resident-outcome slice: `DeviceSpeculativeOutcomeHandle` now exposes
runner-owned compact summary rows on the verifier stream, and the legacy
request-batch host API is a wrapper over resident enqueue plus explicit copy.
Current hot-path scalar decode still copies; the next structural win is
publishing accepted state and feeding the next token from this handle directly.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stochastic | MoE greedy | MoE stochastic | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | G | G | A | A | Correct; host verifier/condition cost is high |
| SingleDevice | CUDA d1 | G | G | A | R | Dense accepted; MoE stoch speed-negative |
| SingleDevice | ROCm d1 | G | G | A | A | Dense accepted; MoE stoch needs non-instrumented confirmation |
| LocalTP | CUDA deg2 | A | R | R | R | Unit path exists; fresh parity/bench matrix needed |
| LocalTP | ROCm deg2 | G | A | R | R | Dense greedy d3 55.4 vs 34.1 tok/s (1.62x) |
| LocalTP | ROCm deg4 | A | R | R | R | Hardware expected; fresh Phase 10 matrix pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm deg2 stages | G | A | R | R | Dense dynamic 62.9 vs 30.3 tok/s (2.08x) |
| NodeLocalTP | CPU sockets | A | A | R | R | Dense parity green; same-run benchmark pending |
| ExpertOverlay | ROCm2TP + CPU2LocalTP | A | R | A | R | MoE parity green; speed remains amber/red |

## SingleDevice Numbers

| Backend | Model | Sampling | Latest same-run decode tok/s | RAG | Blocker |
|---|---|---|---:|:---:|---|
| CUDA | Dense 27B | greedy | base 44.0; d1/d2/d3/dyn 63.9/76.1/91.4/87.8 | G | dynamic trails fixed d3 |
| CUDA | Dense 27B | stochastic | base 44.7; 52.6/52.6/47.7/47.7 | G | d3 short-lane regression |
| ROCm | Dense 27B | greedy | base 30.3; 42.3/48.1/65.0/52.9 | G | absolute CUDA gap |
| ROCm | Dense 27B | stochastic | base 31.3; 37.0/28.7/27.0/36.9 | G | d2/d3 cost |
| CPU | Dense 27B | greedy | base 4.7; 5.9/6.0/9.3/6.1 | G | dynamic shallow |
| CPU | Dense 27B | stochastic | base 4.46; 5.06/5.78/4.85/5.41 | G | verifier/condition |
| CUDA | MoE 35B | greedy | base 139.2; 129.6/136.6/139.1/146.1 | A | weak win, needs repeat |
| CUDA | MoE 35B | stochastic | base 138.8; d2 125.8; d3 133.8 | R | verifier+condition |
| ROCm | MoE 35B | greedy | base 77.0; 79.0/91.5/90.3/86.1 | A | below dense-class win |
| ROCm | MoE 35B | stochastic | base 77.5; d2 56.5; d3 70.9 | R | D2H sync and low d2 acceptance |
| CPU | MoE 35B | greedy | base 17.7; 13.5/13.9/12.4/13.3 | A | host verifier |
| CPU | MoE 35B | stochastic | base 17.6; 14.5/13.3/14.1/14.0 | A | host verifier |

## Current Read

The vLLM-style greedy-draft, one-hot-q sampler path is correct. MoE stochastic
does not need more compact-table tuning right now; it needs cheaper
target/condition execution and cheaper device outcome publication. CUDA remains
near-neutral but not accepted. ROCm d3 improved to 0.92x with guarded bonus
sampling, but still spends about 1.9 s of the short run at the outcome/D2H
sync boundary; the next win must remove or batch that sync.

Target anchors from `ggml-org/llama.cpp@6ddc943`: CUDA dense no-MTP 41.83, d1
54.9, d3 52.5 tok/s; CUDA MoE no-MTP 118.26, d1 142.0, d3 132.8 tok/s.

## Next Phase 10 Moves

1. Consume `DeviceSpeculativeOutcomeHandle` in accepted-state publication and
   next-token staging so stochastic decode can skip the per-step D2H boundary.
2. Reduce condition-forward replay after rejection; it remains visible on both
   CUDA and ROCm MoE stochastic.
3. Refresh LocalTP CUDA2, LocalTP ROCm4, NodeLocalTP CPU, and ExpertOverlay
   benchmark matrices before any rollout claim for those modes.
