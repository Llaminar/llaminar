# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CPU, CUDA, and ROCm. Keep implementation
detail in project handoffs.

RAG: **G** correct and economical, **A** correct but untuned/stale, **R**
failing or not yet proven. Token equality alone is not verifier parity proof.

## Current State

| Goal | Completion | Remaining proof |
|---|---:|---|
| SingleDevice fully device-resident MTP | 97% | refresh d1 economy and stochastic matrix |
| LocalTP fully device-resident MTP | 99% | remote participants and economy matrix |
| ExpertParallel fully device-resident MTP | 97% | mirrored-head batching across every EP mode |

- Homogeneous CUDA/ROCm execution is full-graph only. Device parameters select
  regimes inside one immutable capture; attention never requests recapture.
- GPU KV append, TurboQuant, verifier publication, and collectives use
  persistent workspace and explicit events. Hot paths have no allocation,
  blocking sync, segmentation, or intermediate host observation.
- Prefix restore preserves graph addresses through producer events. GPU
  FFN/MTP APIs reject null publication streams.

## Production Matrix

| Mode | Device | Dense greedy/stoch | MoE greedy/stoch | Status |
|---|---|:---:|:---:|---|
| SingleDevice | CPU | R/R | A/A | refresh paused |
| SingleDevice | CUDA | A/G | A/R | dense d3 wins; d1/MoE need tuning |
| SingleDevice | ROCm | A/A | A/A | dense d3 wins; d1/MoE need tuning |
| LocalTP | CUDA2 | A/A | A/A | Dynamic/LLEP matrix green; perf active |
| LocalTP | ROCm2 | A/A | A/A | Dynamic/LLEP matrix green; perf pending |
| LocalTP | ROCm4 | A/R | R/R | full refresh pending |
| NodeLocalTP | CPU2 | A/A | R/R | dense E2E green; MoE/perf pending |
| ExpertParallel | GPU+CPU | A/R | G/R | Dynamic/LLEP greedy+prefix green |

## Correctness Proof

- All-format grouped attention, TurboQuant, MoE routing, GDN, short-conv,
  stochastic target preparation, and draft publication sweeps are serial-row
  byte exact on their production backends and M ranges.
- CUDA MoE grouped verifier passed every native format at M=2..16 and M=31
  through the real router/expert path after the small-M grouping fusion.
- Release CUDA2/ROCm2 passed all eight Dynamic/LLEP cells and `166/166` checks
  through 2048 tokens with full capture and clean VRAM release.
- Release CUDA2 LLEP + RAM prefix + stochastic dynamic d4..15 passed `23/23`
  canonical server checks at context 4096 and 1024 output tokens.
  This includes deterministic prefix replay, forced movement, long recall,
  prefill replay, no segmented execution, clean shutdown, and PerfStats path
  assertions.
- The 2026-08-04 canonical grouped-verifier gate passed `89/89` explicit CPU,
  CUDA, CUDA2, ROCm, and ROCm2 cells. CUDA/ROCm KV publication is isolated by
  all 14 cache/source format pairs plus converted-read, logical-restore,
  adversarial, and TurboQuant lifecycle cells; each process uses the production
  graph-capture transaction and exact stream/event ownership.
- TurboQuant codebooks are uploaded once per cache/device on its construction
  stream and covered by the cache constructor's initialization fence. Launch
  wrappers cannot perform codebook upload, and no process-global ready flag can
  incorrectly alias initialization across devices. The complete Integration
  tree rebuilt cleanly and the device-free unit/source gate passed `585/585`.

## CUDA LLEP Economy

- Fixed-d3 control: Qwen3.6-35B-A3B IQ3_S, stochastic sampling, fixed prompt,
  425 prefill + 256 decode tokens, three iterations after one warmup.
- Stable M=4 grouping removed four launches/layer and moved verifier compute
  nodes `1536 -> 1376` per device. Direct FP32 route publication then removed
  one conversion kernel plus one memcpy/layer: compute nodes are now `1336`,
  captured prefill graphs are 80 total nodes/device smaller, and the native hot
  transaction is `1403` kernels including 67 NCCL/control sidecars.
- Runtime expert publication now uses one deterministic two-level warp/wave
  scan for counts, offsets, and active ranks instead of the former quadratic
  per-expert scan and second publication phase. Across M=2..31 the fused plan is
  `1.14x..1.26x` faster on CUDA and `1.16x..1.72x` faster on ROCm; at M=4 it
  moved `12.16 -> 9.76 us` and `30.28 -> 19.43 us`, respectively.
- The M=4 CUDA fused kernel uses 40 registers/thread and 4.23 KiB shared memory,
  has zero spills, and retains 100% theoretical occupancy. The ROCm wave64
  kernel uses 45 VGPR, 89 SGPR, and 4,232 bytes LDS with zero private segment
  and zero spills. The post-change Integration tree rebuilt cleanly, the
  canonical grouped-verifier gate passed `89/89` in 834.31 seconds, and the
  device-free unit/source gate passed `585/585` in 134.99 seconds.
- Isolated M=4 grouping is `5.99 us`: 38 registers/thread, 4.10 KiB shared,
  zero spills, and 100% theoretical per-SM
  occupancy. Achieved whole-GPU occupancy is intentionally low for this single
  dependency block; splitting it would restore launch/dependency overhead.
- Matched decode improved `156.97 -> 164.93 -> 165.92 tok/s`; acceptance stayed
  `61.57%` with zero validation failures. Prefill is `179.51 tok/s`; decode is
  `3.43%` below the historical `171.81 tok/s` promotion target.
- Direct CUDA router publication is 40 registers, zero spills, 100% theoretical
  and 83% achieved occupancy. ROCm is 18 VGPR with no scratch or spills.
- Cooperative CUDA Top-K 40 cut target/draft distribution to `0.238/0.214 ms`;
  it is spill-free and byte exact through M=16.
- Qwen `N=512,K=2048` grouped projection is byte exact across all 21 CUDA
  formats at M=2..31 and gains `1.69x..2.21x`; its eight-CTA geometry remains a target.
- CUDA GDN is byte exact through M=31; at `d_k=d_v=128` it reaches `17.73 us`,
  541 GB/s, and zero spills.
- This RTX 3090 pair has no peer access; mirrored verification avoids its
  measured `941 us/layer` rooted M=5 NCCL collective.

Matched llama.cpp master comparison, tok/s:

| Backend/model | d1 L/LC | d3 L/LC |
|---|---:|---:|
| CUDA dense 27B | `36.24/59.64` | `64.84/60.91` |
| CUDA MoE 35B | `93.01/153.92` | `159.61/171.81` |
| ROCm dense 27B | `20.08/25.15` | `39.28/22.66` |
| ROCm MoE 35B | `46.33/73.93` | `74.94/95.84` |

### Reproducible llama.cpp CUDA Reference

```bash
CUDA_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-cuda/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

On llama.cpp `5788b51`, three runs gave a `155.4 tok/s` decode median and
`1122.5 tok/s` prefill median. Promotion retains the stricter historical
`171.81 tok/s` decode target and requires at least `2245 tok/s` prefill.

## Next Gates

1. Keep native geometry for large kernels and fuse adjacent tiny/control work.
   `V2_Perf_CUDAPersistentVerifierGeometry` found a one-kernel CTA proxy `1.25x`
   faster, but full-lane/ALU8 proxies only `0.964x/0.952x`; ncu attributes
   `63-68%` of issue stalls to cooperative barriers. Zero spills.
2. Exceed `171.81 tok/s` on fixed d3, then tune dynamic depth/hysteresis through
   depth 15 under deterministic prompt scenarios.
3. Repeat the economy pass for ROCm and the remaining SingleDevice/EP lanes.
