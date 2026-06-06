# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard for current CUDA/ROCm evidence. Raw history stays in
`benchmark_results/` and `/tmp/llaminar-mtp-bench`.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 623.12 | 16.69 | current correctness baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP, replay | 426.11 | 7.30 | correct but regresses |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, replay | 426.44 | 8.25 | correct but regresses |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | beats l.cpp no-MTP |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | pre-hardening MoE ratchet |
| Dense long `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP | n/a | 54.78 | 1.77x over 30.93 baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

Current CUDA dense artifacts:
`benchmark_results/cuda_dense_mtp/20260606T055941Z-cuda-verifier-hardening`.

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`. MTP-off uses `llama-bench`; MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`. Artifact:
`benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.

| Lane | Model | Prefill tok/s | Decode tok/s | Acceptance |
|---|---|---:|---:|---:|
| `llama-bench` no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| `llama-cli` MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| `llama-cli` MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| `llama-bench` no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| `llama-cli` no-MTP | MoE 35B A3B | 1120.1 | 112.6 | n/a |
| `llama-cli` MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| `llama-cli` MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## Current Findings

- CUDA dense verifier-row state shortcuts are unsafe for Qwen3.6 today.
  Forced shortcut d3 fails immediately after accumulated shortcut state:
  `condition=674`, accepted `258,10608,20271,92217`, committed next `48567`,
  sequential replay next `1473`.
- Forced shortcut d1 also fails a deeper continuation check: first next token
  matches, but the committed continuation diverges within 16 greedy tokens.
- The clean local M=4 verifier terminal token can match sequential decode, but
  the captured/restored GDN/KV/hidden payload is not decode-equivalent and later
  contaminates the stream. CUDA therefore keeps verifier-row shortcuts disabled
  and pays replay for correctness.
- CUDA replay path is not economically viable: fixed d1 records 64 rollbacks for
  64 verifier runs, and fixed d3 records 33 rollbacks for 33 verifier runs.
- ROCm remains the proven dense MTP speed lane because its verifier-row restore
  shortcut has parity coverage and avoids accepted-token replay.

## Retained Actions

- CUDA dense: make small-M verifier execution decode-equivalent, or build a
  graph-native sequential verifier that avoids duplicated all-position verifier
  plus replay work. Until then, do not re-enable CUDA verifier-row shortcuts.
- CUDA MoE: rerun after dense hardening before treating pre-hardening MTP wins as
  production evidence.
- ROCm: continue toward the 2x dense target by reducing captured verifier GPU
  work in ordinary GEMM, fused Gate/Up, GDN projection, recurrence, and LM head.
- Shared: keep generated GEMM/GEMV dispatch tables aligned with prefill buckets,
  keep GPU streams explicit, and keep parity tests in the normal suite.
