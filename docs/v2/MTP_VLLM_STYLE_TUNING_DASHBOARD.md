# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA, ROCm, CPU across SingleDevice,
LocalTP, LocalPP, NodeLocalTP, and ExpertOverlay. Keep this file near 6KB.

RAG: **G** = correct and speed-positive. **A** = correct but slow,
partial, policy-sensitive, or stale. **R** = failing, speed-negative, or
unproven. Every iteration refreshes no-MTP, fixed d1/d2/d3, and dynamic for the
active lane; broad units plus touched parity pass before a WiP commit.

## Phase 10 Snapshot

Active blocker: MoE stochastic fixed-depth throughput. Dense lanes are
accepted or close; ROCm d3 now beats baseline after the replay-safety split,
but the win is small and CUDA MoE stochastic remains short of target.

Fresh audits: `20260614T012855Z`, `20260614T035626Z`, `20260614T041420Z`.
Correctness checkpoint: Phase 9.5/9.6 are green. CPU/CUDA/ROCm
state-publication parity passes; CUDA/ROCm depth-3 guards prove Qwen3.6 MoE
M=2/3/4 verifier rows stay grouped; ROCm keeps M=3/4 on tuned tile-M=2 K-part.

Phase 9.7 MoE shared decode-equivalent verifier rows are green on CPU/CUDA/ROCm
for M=1..4 with strict raw-logit cosine, relative L2, symmetric KL,
sampled-token, and four-token continuation checks. Dense CPU/CUDA/ROCm grouped
row proofs are green for M=2..4 with strict cosine, relative L2, symmetric KL,
and sampled-token checks; the latest CPU grouped gate took 135.31s real time,
so Phase 9.8 still keeps CPU perf suspect. MoE direct all-position publication stays disabled after the rejected
CUDA M=2 candidate (`row0 cos=0.9826 rel_l2=0.1855`). Phase 9.8 kernel slice:
NativeVNNI static gates, CPU/CUDA/ROCm M=2..4 serial-equivalence, and focused
trainer CSV smoke all pass. Greedy GPU all-position now keeps first+draft
tokens in device slots, removes the resident verifier token-row H2D fallback,
snapshots base-cache counts D2D, keeps scalar GDN/short-conv device-index
publication device-only, lets dense request batches pass the hybrid guard, and
names remaining D2H as response materialization. GDN/short-conv request-batch
restore now has a hard-fail API plus CUDA/ROCm request-owned live-state banks.
GPU request-batch runner smoke is green; compact host bridge still feeds planning.

| Lane | Baseline | Best MTP | RAG | Evidence |
|---|---:|---:|:---:|---|
| CUDA MoE stochastic | 129.0 | d3 124.4 (0.96x) | R | d3 76.9% acc, verifier/producer limited |
| ROCm MoE stochastic | 78.6 | d3 82.0 (1.04x) | A | d3 74.3% acc after MoE sidecar recapture |

Latest signal: compact one-hot draft verification is accepted. Compact D2H
enqueue is sub-ms; large ROCm bridge waits are upstream producer work. Recapturing
MoE sidecars after publication recovers ROCm d3 to 82.0 tok/s vs 78.6 baseline,
but reset is not the floor: `sidecar_replay_reset_ms=10.7`, while
`main_verifier_graph_replay_gpu_ms=1503`; sampled verifier time is led by MoE
expert FFN at 151 ms and router at 21 ms. Phase 10 needs persistent vLLM-style
MoE metadata and fused verifier work, not more D2H/reset plumbing.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stochastic | MoE greedy | MoE stochastic | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | G | G | A | A | Correct; CPU verifier still costly |
| SingleDevice | CUDA d1 | G | G | A | R | MoE M=1..4 verifier proof green; stochastic not speed-positive |
| SingleDevice | ROCm d1 | G | G | A | A | MoE M=1..4 verifier proof green; M3/M4 proof slow |
| LocalTP | CUDA deg2 | A | R | R | R | Fresh parity/bench matrix needed |
| LocalTP | ROCm deg2 | G | A | R | R | Dense greedy d3 55.4 vs 34.1 tok/s |
| LocalTP | ROCm deg4 | A | R | R | R | Hardware expected; fresh matrix pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm deg2 stages | G | A | R | R | Dense dynamic 62.9 vs 30.3 tok/s |
| NodeLocalTP | CPU sockets | A | A | R | R | Dense parity green; benchmark pending |
| ExpertOverlay | ROCm2TP + CPU2LocalTP | A | R | A | R | MoE parity green; speed unaccepted |
| ExpertOverlay | GPU hot + remote CPU cold | R | R | R | R | Missing matched MPI sparse participant graph; fail-closed |

## SingleDevice Numbers

| Backend | Model | Sampling | Latest same-run decode tok/s | RAG | Blocker |
|---|---|---|---:|:---:|---|
| CUDA | Dense 27B | greedy | base 44.0; d1/d2/d3/dyn 63.9/76.1/91.4/87.8 | G | dynamic trails fixed d3 |
| CUDA | Dense 27B | stochastic | base 44.7; 52.6/52.6/47.7/47.7 | G | d3 short-lane regression |
| ROCm | Dense 27B | greedy | base 30.3; 42.3/48.1/65.0/52.9 | G | absolute CUDA gap |
| ROCm | Dense 27B | stochastic | base 31.3; 37.0/28.7/27.0/36.9 | G | d2/d3 cost |
| CPU | Dense 27B | greedy | base 4.7; 5.9/6.0/9.3/6.1 | G | dynamic shallow |
| CPU | Dense 27B | stochastic | base 4.46; 5.06/5.78/4.85/5.41 | G | verifier/condition |
| CUDA | MoE 35B | greedy | base 139.2; 129.6/136.6/139.1/146.1 | A | weak win, repeat needed |
| CUDA | MoE 35B | stochastic | base 128.8; d1/d2/d3 124.5/114.4/128.7 | R | break-even only |
| ROCm | MoE 35B | greedy | base 77.0; 79.0/91.5/90.3/86.1 | A | below dense-class win |
| ROCm | MoE 35B | stochastic | base 78.6; d3 82.0 | A | recapture restores acc; win small |
| CPU | MoE 35B | greedy | base 17.7; 13.5/13.9/12.4/13.3 | A | host verifier |
| CPU | MoE 35B | stochastic | base 17.6; 14.5/13.3/14.1/14.0 | A | host verifier |

Target anchors from `ggml-org/llama.cpp@6ddc943`: CUDA dense no-MTP 41.83, d1
54.9, d3 52.5 tok/s; CUDA MoE no-MTP 118.26, d1 142.0, d3 132.8 tok/s.

## Next Phase 10 Moves

1. Close Phase 9.8 dense economy: remove remaining row-plan/control host
   bridges and prove grouped verifier perf beats serial on CPU/CUDA/ROCm.
2. Reduce full MoE verifier graph replay cost; align routed/shared expert work
   with vLLM fused-MoE runner semantics before polishing controller policy.
3. Move transaction-output adoption behind resident device metadata so response
   D2H is only an output flush.
4. Promote and smoke hybrid request-batched GDN publication end-to-end.
5. Refresh LocalTP CUDA2, LocalTP ROCm4, NodeLocalTP CPU, and ExpertOverlay
   matrices before rollout claims.
