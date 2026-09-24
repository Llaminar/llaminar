# Qwen3 continuous-generation controls — 2026-09-11

This is diagnostic acquisition for Qwen3-0.6B Q8_0, not approval of a token
corpus or certification of either Docker ISA. All runtime records come from
the unchanged 510-cell canonical inventory exported as
`parity-results/effective-kv-accounting-current-inventory.json`.

The Release server uses FP32 model activations, the declared KV codec and
the model's shared story workload. Every passing cell completes four separate
384-token requests: fresh, full restore, partial extension and full repeat of
the extension. Token equality, authentic prefix outcomes, captured GPU execution
where applicable, memory and clean shutdown remain mandatory.

## Current single-device results

| Backend | KV | Outcome | Whole cell seconds |
|---|---|---|---:|
| CPU | FP16 | Partial response stops at 351; insufficient horizon | 45.657 |
| CPU | Q8_1 | All four requests pass | 61.007 |
| CPU | Q16_1 | Partial response stops at 309; insufficient horizon | 46.121 |
| CPU | TQ | All four requests pass | 69.297 |
| CUDA | FP16 | Partial response stops at 301; insufficient horizon | 13.006 |
| CUDA | Q8_1 | Partial response stops at 372; insufficient horizon | 16.370 |
| CUDA | TQ | All four requests pass | 56.495 |
| ROCm | FP16 | Partial response stops at 260; insufficient horizon | 25.045 |
| ROCm | FP32 | All four requests pass | 41.851 |
| ROCm | Q8_1 | All four requests pass | 38.490 |
| ROCm | TQ | All four requests pass | 59.053 |

Every short partial response listed above was checked against an independently
started cold server receiving exactly the same body, seed and configuration.
All completion token IDs agree, including the stopping point. That excludes
restore-induced drift in those reproductions but does not meet the 384-token
minimum. No EOS handling, prompt bytes, runtime settings or threshold was changed.

CPU/Q16_1's separate cold request also produces exactly 309 tokens, with no
prefix hit, in 12.330 seconds. Two-rank CPU NodeTP subsequently passes all four
requests in **50.881 seconds**. Two-rank CPU NodePP fails its first decode:
the command root (rank zero) tries to sample absent logits owned by the tail
(rank one). This is an actual runtime failure, not short-EOS evidence.

All thirteen Qwen3 controls have now been attempted. After fixing NodePP's
authority, seven pass and six have insufficient horizons with exact
cold/restored witnesses. NodePP now completes fresh/full 384-token responses
but its partial response stops at 213; a cold server produces those same
213 IDs and termination. That recheck takes 42.341 seconds and has clean logs
and shutdown, with actual serving authority rank one. The generation cell
remains red at the unchanged minimum. See the
[ownership fix and regression](production-ci-pipeline-request-authority.md).

No earlier failed control was retried by the unseen selection. Only NodePP
was subsequently rechecked to prove its specific runtime fix.

## Shared prerequisite and artifact identity

All new runs reuse
`effective-kv-accounting-prerequisites-01/prerequisites.json`, proving the
unchanged 647-Unit/135-preflight gate. Their reported prerequisite elapsed
time is zero, and every staged GGUF is a sealed tmpfs cache hit with zero copied
bytes. There is no per-cell Unit/preflight or whole-model hash pass.

Evidence roots under `parity-results/`:

- `generation-qwen3-cuda-tq-01`.
- `generation-qwen3-rocm-compressed-01` (two sequential cells).
- `generation-qwen3-rocm-fp32-01`.
- `generation-qwen3-cuda-q8kv-01` and `generation-qwen3-cuda-q8kv-cold-01`.
- `generation-qwen3-cpu-01` and `generation-qwen3-cpu-fp16-cold-01`.
- `generation-qwen3-cpu-compressed-01` (Q8_1 passes; Q16_1 stops short).
- `generation-qwen3-cpu-q16-cold-01`.
- `generation-qwen3-cpu-tq-01`.
- `generation-qwen3-node-cpu-01` (NodePP first failure).
- `generation-qwen3-node-tp-cpu-01`.
- `generation-qwen3-node-pp-authority-02` and `generation-qwen3-node-pp-cold-02`.

The post-fix NodePP runs use the fresh
`pipeline-request-authority-prerequisites-01/prerequisites.json`: 647 Unit
and 136 preflight registrations, including the new real-MPI regression. Its
20-repeat focused proof also passes. The new 510-cell inventory is byte-for-byte
equal in its cell records; the fix changes execution ownership, not test policy.

The earlier CUDA/ROCm FP16 records and their cold-request witnesses are
described in [the GPU prefix-basis audit](production-ci-gpu-aq8-prefix-basis.md).
Passes here cannot waive the separate HF numerical reds documented there.
