# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CPU, CUDA, and ROCm across SingleDevice,
LocalTP, NodeLocalTP, LocalPP, and ExpertParallel. Keep this dashboard under
6 KB; put implementation detail in project handoffs.

RAG: **G** correct and economical, **A** correct but untuned/stale, **R**
failing or not yet proven. Token equality alone is not verifier parity proof.

## Current State

| Goal | Completion | Remaining proof |
|---|---:|---|
| SingleDevice fully device-resident MTP | 96% | refresh d1 economy and full stochastic matrix |
| LocalTP fully device-resident MTP | 97% | remote-participant stochastic/economy matrix |
| ExpertParallel fully device-resident MTP | 95% | mirrored-head request batching across every EP mode |

- Homogeneous CUDA/ROCm execution is full-graph only. Device parameters select
  serial/grouped and short/long sequence-parallel regimes inside one immutable
  captured launch envelope; attention no longer requests host recapture
  variants.
- GPU KV append, TurboQuant conversion, verifier publication, sampling, and
  compact NCCL/RCCL broadcast use planned persistent workspace and explicit
  producer/consumer event ordering. Hot paths contain no allocation, host
  transfer, blocking synchronization, or segmented execution.
- CPU TurboQuant supports production `TQ8-K/TQ4-V` and `TQ8-K/TQ8-V`, D=64,
  128, and 256, with direct small-work execution and physical-core-parallel
  grouped execution.
- Best-effort M1/grouped NativeVNNI policies are installed on all backends.
  Ordinary prefill remains on the legacy production heuristic.
- RAM/disk prefix tiers support model-SHA archives, FIFO demotion/overwrite,
  cold-hit VRAM promotion, restart, and pressure paths.
- CUDA/ROCm MoE prefix restore preserves reusable captures and restores the
  stable-address model-lifetime runtime template through explicit graph-build
  producer streams/events. GPU FFN/MTP graph APIs reject null publication
  streams before device work.
- Integration-build unit gate: `585/585` green on 2026-07-31; GPU graph
  lowering is registered only in the integration tier.

## Production Matrix

| Mode | Device | Dense greedy/stoch | MoE greedy/stoch | Status |
|---|---|:---:|:---:|---|
| SingleDevice | CPU | R/R | A/A | refresh paused |
| SingleDevice | CUDA | A/G | A/R | dense d3 wins; d1/MoE need tuning |
| SingleDevice | ROCm | A/A | A/A | dense d3 wins; d1/MoE need tuning |
| LocalTP | CUDA2 | A/A | R/R | resident request batch green; perf pending |
| LocalTP | ROCm2 | A/A | R/R | resident request batch green; perf pending |
| LocalTP | ROCm4 | A/R | R/R | full refresh pending |
| NodeLocalTP | CPU2 | A/A | R/R | dense E2E green; MoE/perf pending |
| ExpertParallel | GPU+CPU | A/R | G/R | Dynamic/LLEP greedy+prefix green |

## Fresh Correctness Proofs

- CUDA and ROCm all-format grouped attention are serial-row byte exact for
  every `M=2..16`, D=64/128/256, unequal row lengths, and long-context
  sequence-parallel boundaries. Stable-graph long-context suites are green.
- CPU fused TurboQuant quantization and grouped materialization are byte exact
  against serial rows for both TQ modes, D=64/128/256, `M=1..16`, plain and
  RoPE-on-read.
- Direct CUDA/ROCm TurboQuant cache integration now binds the same persistent
  workspace ownership contract as production. The symmetric `8/8` matrix is
  green for TQ4/TQ8, captured request batches, unequal continuation, and
  incremental dequantization; the corresponding CPU cache/TQ gate is `10/10`.
- CUDA/ROCm deterministic MoE route planning covers `M=1..4`,
  `top_k={1,2,4,8,16}`, all 256 expert metadata entries, and 20 repeated exact
  launches per cell.
- CUDA/ROCm GDN and short-conv capture-lifetime matrices cover M=2/3/4 with
  byte-equal continuation and complete live state.
- Fresh Release CUDA2/ROCm2 LLEP+MTP+RAM-prefix full-context cells pass
  `22/22` each with strict full-graph PerfStats, 2048 generated tokens, clean
  shutdown, and complete VRAM release. Dynamic/LLEP long-context and prefix
  matrices are green.
- The original CUDA2 LocalTP long-context stochastic prefix+MTP stale-bank
  reproduction now passes at 590 seconds; matching CUDA2/ROCm2 focused graph
  lifecycle cells are green.
- CUDA/ROCm SingleDevice forced-token publication is device-owned and
  graph-captured: both Release E2E lanes pass `31/31`, report full capture, and
  account for all 96 forced-token transactions without a host control scalar.
- Transfer-state stress covers all 336 slot rotations and 32 cross-stream
  reset epochs on CUDA/ROCm. Source policies forbid eventless publication,
  blocking GPU sync, hidden hot-path allocation, and direct coherence
  transitions.

## Kernel Economy

- Stable LocalTP2 graph replay medians:
  CUDA KV32 `9.196 us`, KV16384 `107.167 us`;
  ROCm KV32 `24.454 us`, KV16384 `380.806 us`.
- CUDA reduction output tiling: `8.54 -> 7.74 us`, 40 registers/thread, zero
  spills, 100% theoretical occupancy, 16 blocks.
- ROCm reduction output tiling: `16.58 -> 14.56 us`, 28 VGPR, 64 SGPR, zero
  scratch/spills, 512 B LDS, 16 waves.
- CUDA long attention phase: 64 registers/thread, zero spills, 50% achieved
  occupancy, 53% compute and 49% memory throughput.
- ROCm long phase: 36 allocated VGPR, 64 SGPR, zero scratch/spills, 99.8% VALU
  utilization, 55.6% memory-unit busy, 75.6% L2 hit.
- Deterministic MoE route planner: CUDA `3.392 us` versus `22.212 us`
  (`6.55x`), 31 registers and zero spills; ROCm `11.2 us`, 16 VGPR, 33 SGPR,
  1.25 KiB LDS, zero scratch/spills.

Matched llama.cpp master comparison, tok/s:

| Backend/model | d1 L/LC | d3 L/LC |
|---|---:|---:|
| CUDA dense 27B | `36.24/59.64` | `64.84/60.91` |
| CUDA MoE 35B | `93.01/153.92` | `159.61/171.81` |
| ROCm dense 27B | `20.08/25.15` | `39.28/22.66` |
| ROCm MoE 35B | `46.33/73.93` | `74.94/95.84` |

## Next Gates

1. Run the full-context greedy and stochastic CPU/CUDA/ROCm matrix with strict
   MTP, prefix, Dynamic/LLEP, collective, and full-graph counters.
2. Close remote-participant lifetime and mirrored-head request batching for
   every LocalTP and ExpertParallel mode.
3. Tune d1 attention/GEMV and MoE grouped FFN until SingleDevice is at least
   llama.cpp economy, preserving byte equality and the device-owned graph.
