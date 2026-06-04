# Prefix Cache And MTP Benchmark Notes

Durable Phase 14 scoreboard: latest headline numbers, evidence artifacts, and
live gaps. Keep this file concise; rejected tuning history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `qbf`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 | 53.81 | 1.75x | Correctness green, short of 2x |
| Dense default benchmark, 595p/128d | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 | 46.74 | 1.56x | Depth-sensitive |
| Dense long lane, `qbf`, `-c 64 -n 48` | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.75 | 53.30 | 1.31x | Depth 1 best |
| Dense default lane, 595p/64d | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.82 | 55.58 | 1.36x | d1 wins; dynamic initial-min now 53.83 |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 | 9.50 | 1.64x | Short smoke only |
| MoE default lane, 595p/64d | ROCm `rocm:0` | Qwen3.6 35B A3B | 19.72 | 42.04 | 2.13x | Fixed d1, ratcheted |
| MoE default lane, 595p/64d | CUDA `cuda:0` | Qwen3.6 35B A3B | 103.31 | 131.10 | 1.27x | Clean fused path, near 1.3x |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | After single-device lanes |

llama.cpp CUDA north star, `ggml-org/llama.cpp@6ddc943`:

`llama-bench -p 768 -n 64 -ngl 999 -r 3` is MTP-off only; actual MTP uses
generated `mtp-*` sidecars and `llama-cli --single-turn`.
Artifacts: `benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.

| Lane | Model | Prefill | Decode | Acceptance |
|---|---|---:|---:|---:|
| `llama-bench` no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| `llama-bench` no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| `llama-cli` no-MTP | Dense 27B | 749.0 | 41.5 | n/a |
| `llama-cli` MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| `llama-cli` MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| `llama-cli` no-MTP | MoE 35B A3B | 1120.1 | 112.6 | n/a |
| `llama-cli` MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| `llama-cli` MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## MoE ROCm Evidence

Qwen3.6 35B A3B on `rocm:0`, default benchmark lane:

| Case | Decode | Acceptance | Artifact |
|---|---:|---:|---|
| baseline | 19.72 | n/a | `benchmark_results/rocm_moe_mtp/20260604T034243Z-64e01724-active-expert-prefill-grid-n64-repeat` |
| fixed d1 | 42.04 | 78.12% | same |
| dynamic max d3 | 37.82 | 71.21% | same |

## MoE CUDA Evidence

Qwen3.6 35B A3B on `cuda:0`, default benchmark lane:

| Case | Decode | Acceptance | Artifact |
|---|---:|---:|---|
| latest clean baseline | 103.31 | n/a | `benchmark_results/cuda_moe_mtp/20260604T190711Z-m1-correction-fused-routing-capture/baseline.json` |
| latest clean fixed d1 | 131.10 | 85.94% | `benchmark_results/cuda_moe_mtp/20260604T190711Z-m1-correction-fused-routing-capture/mtp_d1.json` |
| prior fixed d1 ratchet | 126.60 | 89.06% | `benchmark_results/cuda_moe_mtp/20260604T174640Z-shared-expert-prefill/mtp_d1.json` |
| latest fixed d3 | 68.57 | 66.86% | `benchmark_results/cuda_moe_mtp/20260604T144911Z-post-fused-path-regression-refresh/mtp_d3_after_suffix_commit_fix.json` |

## Retained Actions

- ROCm dense/MoE: graph-safe sidecar streams, depth clamping, M=2/3/4 VNNI,
  verifier-row GDN rollback restore, and compact active-expert MoE prefill grids.
- CUDA dense/MoE: verifier-row GDN restore, graph-capturable small-M attention,
  grouped verifier prefill, stream-explicit shared experts, fused split-K MoE
  paths, and graph-capturable cuBLAS batched GDN alpha/beta projections.
- CUDA MoE correction replay forces all-position grouped-prefill routing/expert/
  shared paths for `seq_len=1`; CUDA routing stays device-only during capture.
- CUDA MoE MTP depth-3 parity now uses the stable benchmark-prompt lane, and shared
  expert gate verification allows mathematically valid sigmoid-underflow zero rows.
- Depth>1 MTP rollback now commits suffix shifted-cache rows after an already-committed
  verifier prefix; regressions cover tiny graph construction and CUDA MoE depth-3 parity.
- CUDA and ROCm GPU greedy argmax tie-break to the lowest token id, matching CPU greedy.
- CUDA Qwen3.6 MoE production sampling is guarded by same-step gathered-logits argmax
  parity instead of an unstable independent-run near-tie token oracle.
- Memory planning now charges terminal-row logits and prepared embedding workspace only;
  dense CUDA default context fits the 24 GB card again.
- Explicit non-null GPU stream hard failures remain required.

## Next Work

CUDA MoE MTP now beats the llama.cpp no-MTP `llama-bench` decode anchor but
trails the recalibrated llama.cpp MTP d1 lane (`142.0 tok/s`) by about 8%.
Next target is the remaining verifier budget: router/GDN costs, benchmark-run
capture amortization, and prompt prefill overhead. ROCm MoE keeps the 2.13x ratchet.
