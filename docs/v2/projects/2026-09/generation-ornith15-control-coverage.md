# Ornith 1.5 generation control coverage — 2026-09-11

## Scope and admission

The canonical inventory/history join selected twelve previously unseen
Ornith-1.5-35B Q4_K_M ExpertOverlay MTP-off controls: two-socket CPU NodeTP,
two-CUDA NCCL LocalTP, and two-ROCm RCCL LocalTP, each crossing Static/Dynamic
with Ordinal/Random. No configuration was synthesized by the runner.

All ran sequentially through the AVX-512 Release `llaminar2 serve` binary and
public `/v1/chat/completions`. Each uses FP32 activations, FP16 KV, the model's
canonical prompt and seed, and four requests requiring 384 actual output
tokens. Fresh/full/partial/full prefix restoration, graph-path evidence,
movement policy, memory ownership and clean shutdown remain mandatory.

The unchanged-build receipt
`native-movement-evidence-prerequisites-01/prerequisites.json` covers all
649 Unit and 148 production-preflight entries. It was reused, not rerun for
each cell. The persistent tmpfs Ornith GGUF was a cache hit with zero copied
bytes. No production code, test policy, prompt, precision, seed or threshold
changed during acquisition; every selected overlay cell passed on its first
attempt.

## Completed overlay controls

Local evidence: `parity-results/generation-ornith15-unseen-overlay-01` and
`generation-ornith15-unseen-overlay-driver-01.log`. Total wall time was
1,266.763 seconds (21.1 minutes), including all twelve server lifetimes.

| Topology | Movement | Placement | Cell seconds |
|---|---|---|---:|
| CPU2 NodeTP | Static | Ordinal | 156.044 |
| CPU2 NodeTP | Dynamic | Ordinal | 177.061 |
| CPU2 NodeTP | Static | Random | 159.134 |
| CPU2 NodeTP | Dynamic | Random | 185.142 |
| CUDA2 NCCL | Static | Ordinal | 48.276 |
| CUDA2 NCCL | Dynamic | Ordinal | 51.390 |
| CUDA2 NCCL | Static | Random | 50.084 |
| CUDA2 NCCL | Dynamic | Random | 54.243 |
| ROCm2 RCCL | Static | Ordinal | 91.371 |
| ROCm2 RCCL | Dynamic | Ordinal | 99.005 |
| ROCm2 RCCL | Static | Random | 93.480 |
| ROCm2 RCCL | Dynamic | Random | 100.905 |

All 48 requests completed the full 384-token horizon, totaling 18,432 output
tokens. Every observation reports complete, repeatable execution. Within each
backend topology, all four movement/placement combinations produce identical
token vectors for all four requests. This is an additional observation, not a
claim of cross-backend byte identity or a replacement for numerical proof.

Static reports no movement. CPU Dynamic Ordinal/Random publish 470/510
participant-placement edges respectively. Each rank mirrors the same global
edge records; summing both rank copies would incorrectly double these counts.
GPU Dynamic passes the qualified native copy/apply/byte evidence contract,
including the retained-wave observer repaired in the preceding slice. Dynamic
is slower than Static in these short instrumented probes; no speedup or
benchmark certification is claimed.

## Remaining work

The additional single-ROCm Ornith control passes in 63.051 seconds under the
same receipt in `generation-ornith15-unseen-rocm-single-01` (63.614 seconds
including driver overhead). It completes all four 384-token requests, required
prefix/graph/memory checks and clean shutdown. The single-device typed movement
obligation is not applicable; it does not claim multi-device movement coverage.

Thus all thirteen previously unseen Ornith controls are individually green on
their first attempts: 52 requests and 19,968 output tokens. The fresh canonical
manifest/history join leaves **43 unseen MTP-off controls out of 175**:
36 Qwen3.5-122B ExpertOverlay cases and seven other dense/NodeTP/LocalPP cases.
Continue the remaining unseen controls sequentially, preserving the first red
if one occurs. The shared receipt is still valid because no source/build change
was required during this acquisition.

All observations remain **unapproved diagnostic controls** with
`certification_eligible=false`. The full independent HF provenance, MTP/control
comparisons, per-request movement-ledger certification and baseline approval
are not supplied by these runs. The current Docker pipeline still consumes
the numerical parity report; its cutover to approved generation regression
evidence is unfinished. Both ISA images still need their own full E2E and
benchmark certificates. Do not start or certify an image from this partial
coverage ledger.
