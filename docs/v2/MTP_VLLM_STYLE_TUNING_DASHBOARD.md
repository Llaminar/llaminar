# vLLM-Style MTP Tuning Dashboard

Scope: SingleDevice Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU. Keep under 5KB and update every iteration.

Contract: refresh CUDA/ROCm/CPU with no-MTP, fixed d1/d2/d3, and dynamic. Dynamic must be reported beside same-run fixed rows. Before commit, run broad units plus relevant parity.

RAG: Green = correct and speed-positive near target. Amber = correct but slow, partial, or policy-sensitive. Red = failing or unproven.

## Matrix

Latest full matrix:
`benchmark_results/mtp_vllm_style/20260609T-phase3-row-indexed-accepted-matrix/`
with `--decode-tokens 16 --perfstats`.

Latest focused Phase 4 stochastic slice:
`benchmark_results/mtp_vllm_style/20260610T-phase4-device-first-target-summary/`.

| Backend | Model | Sampling | RAG | Latest decode tok/s | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | base 44.6; d1/d2/d3/dyn 60.9/55.7/60.0/49.3 | dynamic short-run lag |
| CUDA | Dense 27B | stochastic | Amber | base 44.7; d1/d2/d3/dyn 45.0/46.6/43.5/45.0 | modest d2 only |
| ROCm | Dense 27B | greedy | Green | base 31.3; d1/d2/d3/dyn 45.7/33.4/41.9/40.7 | d2 weak, dynamic shallow |
| ROCm | Dense 27B | stochastic | Amber | base 31.25; focused d2 26.97 | shifted-prefill/condition/verifier cost |
| CPU | Dense 27B | greedy | Green | base 4.7; d1/d2/d3/dyn 5.9/6.0/9.3/6.1 | dynamic shallow |
| CPU | Dense 27B | stochastic | Amber | base 4.7; d1/d2/d3/dyn 4.7/5.4/4.8/4.6 | verifier/policy cost |
| CUDA | MoE 35B | greedy | Amber | base 112.9; d1/d2/d3/dyn 62.8/66.8/69.6/57.9 | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | base 114.5; d1/d2/d3/dyn 49.8/45.0/42.5/53.6 | zero acceptance |
| ROCm | MoE 35B | greedy | Amber | base 64.7; d1/d2/d3/dyn 38.2/35.7/40.3/35.9 | verifier dominates |
| ROCm | MoE 35B | stochastic | Amber | base 64.8; d1/d2/d3/dyn 29.7/25.8/23.1/30.5 | verifier dominates |
| CPU | MoE 35B | greedy | Amber | base 17.7; d1/d2/d3/dyn 13.5/13.9/12.4/13.3 | host verifier cost |
| CPU | MoE 35B | stochastic | Amber | base 17.6; d1/d2/d3/dyn 14.5/13.3/14.1/14.0 | host verifier cost |

## Current Read

- Phase 3 row-indexed verifier is accepted for dense SingleDevice: focused row-select units/integrations, dense CPU/CUDA/ROCm greedy+stochastic parity, full `^V2_Unit_`, Integration/Release builds, and the full matrix above pass.
- Dense greedy is speed-positive on every backend. Best fixed depth: CUDA d1 1.37x, ROCm d1 1.46x, CPU d3 1.97x.
- Phase 4 sidecar stream handoff, device-token batch verify, sidecar token input, verifier token input, deferred draft-token host reads, and device-first target-token summary are active for penalty-free stochastic GPU rows. Post-rebuild gates pass: `^V2_Unit_` 500/500 plus ROCm stochastic parity/smoke and GPU sampling integrations. CUDA dense short-run has a modest d2 win, 1.04x.
- ROCm stochastic is functionally guarded but speed-negative: latest focused d2 is 26.97 vs 31.25 tok/s. Target-token D2H is reduced to one final/budget read; counters show 15 deferred first-token reads, 15 device-first batch summaries, and target ready events/waits. Next fix: reduce shifted-prefill/condition/verifier host-wall cost.
- MoE is functionally alive on all backends but speed-negative everywhere. Next tuning target is MoE verifier/catch-up cost before any MoE MTP acceptance claim.
- Dynamic depth is safe but conservative on 16-token runs. It should stay a policy-tuning target after the verifier path is cheaper.
- Phase 4 direct first-token sampling was benchmark-rejected and removed. Continue with true device-resident draft/decision plumbing.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next

1. Keep running the full bounded matrix every iteration.
2. Finish Phase 4 by reducing shifted-prefill/condition/verifier cost now that target/draft token D2H is no longer the main ROCm blocker.
3. Revisit dynamic-depth promotion once verifier cost no longer dominates short requests.
