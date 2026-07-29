# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice, LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay. Keep under 6 KB.

RAG: **G** correct/speed-positive, **A** correct but slow/stale, **R** failing
or unproven. Fresh rows use `--decode-tokens 16 --perfstats`.

## Snapshot

Fresh E2E: dense baseline/RAM-prefix/MTP d2 `261/261` on CPU/CUDA/ROCm; CPU
MoE MTP d2 `27/27`; CUDA Qwen3.5 MoE bucket `28/28`.

2026-07-23: best-effort M1/grouped NativeVNNI policies are installed on all
backends; production smoke and the `576/576` unit gate pass. Ordinary prefill
remains heuristic-only. All-format economical retuning remains open.

2026-07-24: the three-tier prefix lifecycle is operational. Model-SHA archives,
RAM/disk demotion, bottom-tier overwrite, and cold-hit VRAM re-promotion are
covered. CUDA/ROCm pressure and restart E2E cells pass with exact tokens and
path counters.

LocalTP compact prefill now uses each child's mirrored full head; rank sampling
fans out to child-resident mailboxes with no logits gather or tiny allreduce.
CUDA2/ROCm2 unequal-length greedy+stochastic production cells are green.
Attention is byte-exact across grouped formats/unequal lengths; captured scalar
replay is byte-exact and has no pinned host mirror or parameter H2D path.

MoE graph metadata uses device-unique leased workspace slots; capture rejects
publication/rebinding and replay never touches the lease mutex. CUDA2/ROCm2
Dynamic PhaseSplit and focused graph/ROCm all-format proofs are green.

2026-07-27: CUDA decode dispatch is phase-invariant: eager graph warmup uses the
captured policy, closing M3/M4 one-ULP drift. All-format, real GDN-weight, and
Dense/MoE M1-M4 operation-equivalence gates pass.

2026-07-28: resident MTP timeline/transaction/response/PerfStats events are
setup-owned; policy forbids event create/destroy in 11 hot paths. CUDA/ROCm
sampling and Qwen3.6 stochastic full-graph smokes pass; unit gate `582/582`.
Greedy verifier reduction and NCCL/RCCL compact broadcast are graph-owned;
strict counters prove both.
CUDA2/ROCm2 Dynamic and LLEP+RAM-prefix cells pass. Architecture estimate:
LocalTP 96%, ExpertParallel 93%; stochastic/economy matrix refresh remains.

2026-07-29: transfer-slot occupancy is now distinct from directory
addressability; immutable runtime and directory baselines reset by
explicit-stream asynchronous D2D. CUDA/ROCm pass all 336 slot rotations,
promoted staging-origin claims, and repeated resets. Fresh Release CUDA2/ROCm2
LLEP+MTP+RAM-prefix full-context cells pass `22/22` each with strict full-graph
PerfStats, 2048-token generation, clean shutdown, and complete VRAM release.
The complete Integration unit gate passes `589/589`. Architecture estimate:
LocalTP 97%, ExpertParallel 95%; remote-participant stochastic coverage and
all-format economy certification remain.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stoch | MoE greedy | MoE stoch | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused |
| SingleDevice | CUDA d1 | G | G | R | R | Dense green; MoE below baseline |
| SingleDevice | ROCm d1 | A | A | A | A | MoE nearly break-even |
| LocalTP | CUDA deg2 | A | A | R | R | Resident request batch green; perf pending |
| LocalTP | ROCm deg2 | A | A | R | R | Resident request batch green; perf pending |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Dense E2E green; perf pending |
| ExpertOverlay | GPU hot + CPU cold | A | R | G | R | Dynamic/LLEP greedy+prefix green; stochastic/perf pending |

## SingleDevice Speeds

| Lane | Baseline | MTP | Acceptance | RAG |
|---|---:|---:|---:|:---:|
| CUDA dense greedy | `44.46` | `74.92 tok/s` d3 (`1.69x`) | n/a | G |
| CUDA dense stoch | `44.47` | `57.07 tok/s` d1 (`1.28x`) | n/a | G |
| ROCm dense greedy | `31.30` | `39.79 tok/s` dyn (`1.27x`) | n/a | A |
| ROCm dense stoch | `31.79` | `32.16 tok/s` dyn (`1.01x`) | n/a | A |
| CUDA MoE stoch | `138.31` | `99.93 tok/s` d3 (`0.72x`) | `30/39` | R |
| ROCm MoE stoch | `84.18` | `80.19 tok/s` d3 (`0.95x`) | `30/39` | A |

MoE depth sweep:

| Device | d1 | d2 | d3 | Dynamic |
|---|---:|---:|---:|---:|
| CUDA | `74.02` (`0.54x`) | `77.96` (`0.56x`) | `99.93` (`0.72x`) | `88.74` (`0.64x`) |
| ROCm | `61.78` (`0.73x`) | `61.98` (`0.74x`) | `80.19` (`0.95x`) | `61.67` (`0.73x`) |

## Focused Proofs

- CUDA/ROCm routed verifier microbench passed strict cos/L2/KLD/max_abs gates.
  ROCm M4: `0.1778 ms` graph vs `4.7366 ms` row replay. CUDA M4:
  `0.1035 ms` graph vs `9.8135 ms` row replay.
- CUDA/ROCm GDN and short-conv capture-lifetime matrices cover M=2/3/4 and
  every accepted row with byte-equal continuation and complete live state.
- CUDA Qwen3.6 MoE bucketed-prefill E2E passed `20/20`; capture launch `691 us`,
  replay `288 us`.
- CUDA2 ExpertOverlay Dynamic long-context prefix+MTP passed with explicit
  sidecar graph invalidation.
- LocalTP GPU grouped path: mirrored children publish resident outcomes;
  request prefill/sampling uses full heads, device-token matrices, resident
  conditions, and per-child mailboxes. No row replay, host plan/materializer,
  logits gather, or tiny allreduce. ROCm attention uses one grouped phase and
  reduction for shared verifier or independent request banks. Attention params
  are device-written on CUDA/ROCm; source policy forbids host mirrors/H2D.
  CUDA/ROCm attention and request-batch gates are green.
- Long-context CUDA2/ROCm2 Dynamic/LLEP is green. Preallocated admission events
  close split-prefill reuse; replicated logits publications are explicitly
  retired. Verifier control writes publish exact-stream arena ownership, and
  captured compact broadcasts cover root/receiver without sync or D2H.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Drive the full-context CPU/CUDA/ROCm MTP E2E matrix with path counters.
2. Close captured collective and remote-participant lifetime for LocalTP/EP.
3. Prove mirrored-head resident request batching across ExpertParallel modes.
4. Attack MoE verifier producer economics directly. ROCm routed FFN and CUDA
   dense GEMM buckets are the next largest pieces.
5. Prefer grouped/concurrent decode-equivalent kernels over serial row replay.
6. After each concrete win: run strict parity, refresh tok/s, update
   this dashboard, and make a WiP commit.
