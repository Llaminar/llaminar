# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CPU, CUDA, and ROCm. Keep this under 6 KB;
put implementation detail in project handoffs.

RAG: **G** correct and economical, **A** correct but untuned/stale, **R**
failing or not yet proven. Token equality alone is not verifier parity proof.

## Current State

| Goal | Completion | Remaining proof |
|---|---:|---|
| SingleDevice fully device-resident MTP | 97% | refresh d1 economy and full stochastic matrix |
| LocalTP fully device-resident MTP | 99% | remote-participant and economy matrix |
| ExpertParallel fully device-resident MTP | 97% | mirrored-head batching across every EP mode |

- Homogeneous CUDA/ROCm execution is full-graph only. Device parameters select
  regimes inside one immutable capture; attention does not request recapture.
- GPU KV append, TurboQuant conversion, verifier publication, and compact
  NCCL/RCCL broadcast use planned persistent workspace and explicit
  producer/consumer event ordering. Hot paths contain no allocation, blocking
  synchronization, segmented execution, or full-logits host observation.
- MoE prefix restore preserves stable graph addresses through explicit
  producer events. GPU FFN/MTP APIs reject null publication streams.
- Integration unit gate: `585/585` green on 2026-08-02. Hardware-free worker
  contexts keep unit tests CPU-only; physical CUDA/ROCm cases are Integration.

## Production Matrix

| Mode | Device | Dense greedy/stoch | MoE greedy/stoch | Status |
|---|---|:---:|:---:|---|
| SingleDevice | CPU | R/R | A/A | refresh paused |
| SingleDevice | CUDA | A/G | A/R | dense d3 wins; d1/MoE need tuning |
| SingleDevice | ROCm | A/A | A/A | dense d3 wins; d1/MoE need tuning |
| LocalTP | CUDA2 | A/A | A/A | Dynamic/LLEP full matrix green; perf pending |
| LocalTP | ROCm2 | A/A | A/A | Dynamic/LLEP full matrix green; perf pending |
| LocalTP | ROCm4 | A/R | R/R | full refresh pending |
| NodeLocalTP | CPU2 | A/A | R/R | dense E2E green; MoE/perf pending |
| ExpertParallel | GPU+CPU | A/R | G/R | Dynamic/LLEP greedy+prefix green |

## Fresh Correctness Proofs

- All-format grouped attention/TurboQuant sweeps are serial-row byte exact on
  CPU, CUDA, and ROCm through the production M range.
- CUDA/ROCm deterministic MoE route planning covers `M=1..4`, top-k 1..16,
  all 256 experts, and 20 repeated exact launches per cell.
- CUDA/ROCm GDN and short-conv capture-lifetime matrices cover M=2/3/4 with
  byte-equal continuation and complete live state.
- Release CUDA2 and ROCm2 each pass all eight Dynamic/LLEP cells and `166/166`
  checks through 2048 tokens. PerfStats prove full capture, device verification,
  movement, clean shutdown, and VRAM release.
- GPU logits ownership is request-persistent. Fatal host-access guards prevent
  stale mirrors; only the final 4-byte sampled token crosses to the host.
- Canonical Release CUDA2 LLEP stochastic d4..15 plus RAM-prefix E2E is
  `23/23` green at 4096 context/1024 output tokens. PerfStats proves complete
  capture, no segmentation, device verification, real movement, prefix reuse,
  and no host transfer beyond compact response publication and cache tiers.
- Replicated shared-expert residuals no longer enter an invalid allreduce.
  Rooted reduce+broadcast lowering remains graph-captured for genuinely sharded
  contributions and is covered on NCCL and RCCL.

## Kernel Economy

- Stable LocalTP2 graph replay medians:
  CUDA KV32 `9.196 us`, KV16384 `107.167 us`;
  ROCm KV32 `24.454 us`, KV16384 `380.806 us`.
- Deterministic MoE route planner: CUDA `3.392 us` versus `22.212 us`
  (`6.55x`), 31 registers and zero spills; ROCm `11.2 us`, 16 VGPR, 33 SGPR,
  1.25 KiB LDS, zero scratch/spills.
- Removing the invalid shared-expert allreduce improved the production CUDA2
  LLEP stochastic lane from `15.368` to `27.790 tok/s` decode (`+80.8%`), with
  `4030.44 tok/s` prefill. It remains only 16.2% of llama.cpp d3 decode.
- Participant-local overlay preparation removed duplicate cross-GPU repacking:
  graph build fell `61.3 -> 36.5 s`, per-GPU jobs `32129 -> 16385`, and source
  bytes `15.74 -> 9.14 GB`. Missing frozen bindings now fail instead of reading
  another participant's mutable cache.
- On this non-P2P RTX 3090 topology, graph-captured NCCL rooted
  reduce+broadcast at verifier `M=5` costs about `941 us/layer`; collective
  count, payload, and overlap are the immediate Nsight economy targets.

Matched llama.cpp master comparison, tok/s:

| Backend/model | d1 L/LC | d3 L/LC |
|---|---:|---:|
| CUDA dense 27B | `36.24/59.64` | `64.84/60.91` |
| CUDA MoE 35B | `93.01/153.92` | `159.61/171.81` |
| ROCm dense 27B | `20.08/25.15` | `39.28/22.66` |
| ROCm MoE 35B | `46.33/73.93` | `74.94/95.84` |

## Next Gates

1. Profile the full CUDA LLEP lane; reduce/overlap rooted collectives and tune
   every dominant kernel until decode exceeds llama.cpp `171.81 tok/s`.
2. Run the remaining full-context CPU matrix, then close remote-participant
   lifetime and mirrored-head request batching for
   every LocalTP and ExpertParallel mode.
3. Tune d1 attention/GEMV and MoE grouped FFN until SingleDevice is at least
   llama.cpp economy, preserving byte equality and the device-owned graph.
