# Prefix Cache And MTP Benchmark Notes

Current Phase 14 scoreboard for CUDA/ROCm Qwen3.6 work. Raw run history stays in
`benchmark_results/` and `/tmp/llaminar-mtp-bench`.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 707.92 | 41.73 | current baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP, sequential verifier | 598.97 | 38.33 | safe, speed-negative |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 609.73 | 38.19 | 84.95% acceptance |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 610.88 | 38.52 | holds depth 1; below baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | current baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | speed-positive |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1943.69 | 145.36 | speed-positive |
| Dense long `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP | n/a | 54.78 | 1.77x over 30.93 baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

CUDA dense artifacts:
`/tmp/llaminar-mtp-bench/dense-cuda-phase138-{nomtp,shared-d3}-default.json`.
CUDA MoE artifact:
`benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode`.

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`. MTP-off uses `llama-bench`; MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`.

| Lane | Model | Prefill tok/s | Decode tok/s | Acceptance |
|---|---|---:|---:|---:|
| `llama-bench` no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| `llama-cli` MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| `llama-cli` MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| `llama-bench` no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| `llama-cli` MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| `llama-cli` MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## Current Findings

- Dense CUDA MTP remains speed-negative because accepted speculative tokens still
  pay full decode-equivalent verifier forwards. Stats show
  `decode_equivalent_catchup_forward_one` at 9272.9 ms / 383 forwards, versus
  98.6 ms for shifted commits and 11.8 ms for sampling.
- ROCm and CUDA now share `MTPDecodeCatchup` as the safe oracle. It is correct,
  but it is not the final speed path.
- Phase 13.8 Candidate A, sidecar-chain verifier state, is retired by focused
  ROCm/generic and CUDA Qwen3.6 dense negative parity tests.
- Naive Candidate B, selecting state from an all-position verifier, is retired.
  CUDA M=2 diagnostics drift before the GDN copy boundary, and ROCm/generic M=4
  state parity agrees. New commit-replay negative tests pin this.
- llama.cpp and vLLM do not reveal a safe "commit final batched recurrent state"
  trick. llama.cpp carries target nextn hidden rows into the drafter but still
  truncates/restores target memory after verification. vLLM captures uniform
  `1 + draft_count` decode graphs and passes accepted-token counts into
  GDN/Mamba attention metadata.
- The next clean Phase 13.8 target is an accepted-count-aware stateful verifier
  only if its producer rows are serial-equivalent. Otherwise we should abandon
  the batched-state shortcut framing for Qwen3.6 dense and optimize the shared
  graph-captured serial row loop plus small-M kernels.

## Retained Actions

- CUDA/ROCm dense: first prove whether an accepted-count-aware verifier can
  produce serial-equivalent rows. If not, retire the shortcut premise and make
  Candidate F, graph-captured serial-equivalent catch-up, the fast path. Do not
  re-enable raw all-position verifier-row shortcuts.
- CUDA MoE: keep the 148.5 tok/s ratchet and extend long-prompt/controller
  evidence without weakening parity.
- ROCm: keep the 54.78 tok/s dense ratchet while aligning the state contract
  with CUDA.
- Shared: keep all-codebook M=2/3/4 GEMV equivalence and parity tests in the
  normal suites.
