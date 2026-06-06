# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard for current CUDA/ROCm evidence. Raw history stays in
`benchmark_results/` and `/tmp/llaminar-mtp-bench`.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 704.05 | 41.61 | decode restored after GEMV dispatch fix |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP, sequential verifier | 595.72 | 38.63 | correct, verifier still slower than baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 595.50 | 32.06 | correct, deeper verifier overhead |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | beats l.cpp no-MTP |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | parity re-green after workspace rebind fix |
| Dense long `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP | n/a | 54.78 | 1.77x over 30.93 baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

Current CUDA dense artifacts:
`benchmark_results/cuda_dense_mtp/20260606T070336Z-dense-cuda-stage-attribution`.

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

- A stale generated CUDA NativeVNNI GEMV dispatch table caused the dense decode
  collapse. Exact Qwen3.6 decode-shape rules restored no-MTP decode from about
  16.7 tok/s to 41.6 tok/s, matching the llama.cpp no-MTP decode anchor.
- CUDA dense verifier-row state shortcuts remain unsafe for Qwen3.6. The latest
  forced shortcut strict check diverged after `condition=27775`, accepted
  `383,279`, next `1414`: committed continuation `3294,11,1092,513` versus
  replay `3294,11,1092,369`.
- Direct CUDA recurrence and short-conv verifier-row restore regressions now
  pass multi-step replay checks. The remaining blocker is full-graph all-position
  verifier state decode-equivalence, not the low-level restore primitive alone.
- Focused CUDA stage regressions also prove Q5_1/Q5_K Qwen3.6 gate/up,
  fused-SwiGLU down, and FP16-KV M=4 attention match single-row decode.
- CUDA MoE MTP parity crash was a stale singleton-kernel workspace binding:
  `CUDAMoEKernel` retained old MoE scratch subpointers after a runner/workspace
  lifetime change. MoE now invalidates workspace scratch on every bind, and
  route/grouped gate-up launches validate tensor contracts and live CUDA pointers
  before launch.
- CUDA therefore keeps verifier-row shortcuts disabled and uses a
  decode-equivalent sequential greedy verifier. It is correctness-green but still
  slower than baseline because each accepted verifier row pays one-token main
  replay, shifted-row commit, and sidecar restore overhead.
- ROCm remains the proven dense MTP speed lane because its verifier-row restore
  shortcut has parity coverage and avoids accepted-token replay.

## Retained Actions

- CUDA dense: keep the sequential verifier and attack its remaining overhead,
  especially one-token main graph replay, shifted-row commit, and sidecar restore
  cost. Do not re-enable CUDA verifier-row shortcuts.
- CUDA MoE: refresh d1/d3 benchmarks after the workspace rebind fix before
  treating the current MTP win as production evidence.
- ROCm: continue toward the 2x dense target by reducing captured verifier GPU
  work in ordinary GEMM, fused Gate/Up, GDN projection, recurrence, and LM head.
- Shared: keep generated GEMM/GEMV dispatch tables aligned with prefill buckets,
  keep GPU streams explicit, and keep parity tests in the normal suite.
