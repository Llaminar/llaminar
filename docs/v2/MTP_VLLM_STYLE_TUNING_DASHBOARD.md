# vLLM-Style MTP Tuning Dashboard

Scope: SingleDevice Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU. Keep under 5KB and update every iteration.

Contract: refresh CUDA/ROCm/CPU with no-MTP, fixed d1/d2/d3, and dynamic. Dynamic must be reported beside same-run fixed rows. Before commit, run broad units plus relevant parity.

RAG: Green = correct and speed-positive near target. Amber = correct but slow, partial, or policy-sensitive. Red = failing or unproven.

## Evidence

Latest dense stochastic closeout:
`benchmark_results/mtp_vllm_style/20260610T-phase4-dense-stochastic-closeout-matrix-v2/`
with `--decode-tokens 64 --perfstats`.

Latest full dense/MoE matrix:
`benchmark_results/mtp_vllm_style/20260609T-phase3-row-indexed-accepted-matrix/`
with `--decode-tokens 16 --perfstats`.

Latest correctness gate: Phase 5 focused units
`V2_Unit_MTPStateTransaction`, `V2_Unit_MTPGraphConstruction`,
`V2_Unit_PrefillDecodeTransition`, `V2_Unit_GpuWorkspaceAllocationPolicy`,
plus MTP perfstats/matrix script units.

Latest Phase 5 publication-cost slice:
`benchmark_results/mtp_vllm_style/20260610T-phase5-publication-cost-dense-stochastic-gpu/`
and `...-cpu8/`.

## Matrix

| Backend | Model | Sampling | RAG | Latest decode tok/s | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | base 44.6; d1/d2/d3/dyn 60.9/55.7/60.0/49.3 | dynamic short-run lag |
| CUDA | Dense 27B | stochastic | Green | base 43.88; d1/d2/d3/dyn 37.25/58.09/59.44/53.43 | d1 low acceptance |
| ROCm | Dense 27B | greedy | Green | base 31.3; d1/d2/d3/dyn 45.7/33.4/41.9/40.7 | d2 weak |
| ROCm | Dense 27B | stochastic | Green | base 30.33; d1/d2/d3/dyn 33.48/23.71/24.52/32.57 | d2/d3 acceptance-limited |
| CPU | Dense 27B | greedy | Green | base 4.7; d1/d2/d3/dyn 5.9/6.0/9.3/6.1 | dynamic shallow |
| CPU | Dense 27B | stochastic | Green | base 4.46; d1/d2/d3/dyn 5.06/5.78/4.85/5.41 | verifier/condition cost |
| CUDA | MoE 35B | greedy | Amber | base 112.9; d1/d2/d3/dyn 62.8/66.8/69.6/57.9 | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | base 114.5; d1/d2/d3/dyn 49.8/45.0/42.5/53.6 | low/zero acceptance |
| ROCm | MoE 35B | greedy | Amber | base 64.7; d1/d2/d3/dyn 38.2/35.7/40.3/35.9 | verifier dominates |
| ROCm | MoE 35B | stochastic | Amber | base 64.8; d1/d2/d3/dyn 29.7/25.8/23.1/30.5 | verifier dominates |
| CPU | MoE 35B | greedy | Amber | base 17.7; d1/d2/d3/dyn 13.5/13.9/12.4/13.3 | host verifier cost |
| CPU | MoE 35B | stochastic | Amber | base 17.6; d1/d2/d3/dyn 14.5/13.3/14.1/14.0 | host verifier cost |

## Current Read

- Phase 4 dense SingleDevice is accepted. Its closeout gate covered focused
  units, GPU sampling integration, dense stochastic parity on CPU/CUDA/ROCm,
  and CUDA/ROCm stochastic graph smokes.
- Dense stochastic is speed-positive on all backends in useful bounded lanes:
  CUDA d3 1.35x, ROCm d1 1.10x and dynamic 1.07x, CPU d2 1.30x.
- ROCm d2/d3 stochastic are bad because acceptance collapses to 28.6%/46.2% on
  this prompt, not because of rollbacks or transaction failures.
- CPU stochastic is correct and speed-positive, but verifier and condition
  forward dominate: CPU d2 reports 22.9s verifier and 6.6s condition time for
  the measured decode section.
- MoE is functionally alive on all backends but speed-negative everywhere.
- Phase 5 focused slices are green: live-state mutation reasons are typed,
  publication distinguishes accepted vs rejected correction, and the
  post-condition verifier-base stamp is now a tested logical transaction
  helper instead of a payload checkpoint export. Matrix rows now include
  `publish_count` and `publish_avg_ms` for depth-stability checks.
- Phase 5 publication-cost slice: CUDA publish_avg 0.47-0.56ms, ROCm
  0.29-0.32ms, CPU 3.84-3.86ms. Publication is stable across d1/d2/d3; remaining
  cost is verifier/condition plus debug/prefix checkpoints, not slot publish.
- Forced-reject replay parity is now covered in the runner unit: when no ready
  token exists, the debug oracle derives the next token by forwarding the
  rejected correction, then checks it against full replay.
- Phase 5 is accepted on the focused gate; stale all-position checkpoint tags
  are gone.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next

1. Enter Phase 6: graph-captured draft/verify/sample/publish stress work.
2. Keep the dense stochastic closeout matrix as the per-iteration regression.
3. Move MoE speed work after SingleDevice state publication is cheap.
