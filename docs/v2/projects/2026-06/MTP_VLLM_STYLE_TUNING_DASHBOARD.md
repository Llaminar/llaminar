# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice, LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay. Keep under 6 KB.

RAG: **G** correct and speed-positive, **A** correct but slow/stale,
**R** failing, speed-negative, or unproven. Fresh rows use bounded
`--decode-tokens 16 --perfstats`.

## Snapshot

Fresh E2E: dense baseline/RAM-prefix/MTP d2 `261/261` on CPU/CUDA/ROCm; CPU
MoE MTP d2 `27/27`; CUDA Qwen3.5 MoE bucket `28/28`.

2026-07-10: unit `518/518`; grouped gate `47/47` substantive cells (CPU 13,
CUDA 15, ROCm 19; `48/48` with fixture). GPU request-batch stochastic draws
consume resident positions and match scalar bytes. GDN/short-conv capture-once
continuation covers M=2/3/4 and every accepted row byte-for-byte.

LocalTP compact prefill now uses each child's mirrored full head; rank sampling
fans out to child-resident mailboxes with no logits gather or tiny allreduce.
CUDA2/ROCm2 unequal-length greedy+stochastic production cells passed in
`45.77 s`/`63.18 s` with resident MTP, recurrent, and rank fan-out counters.
GPU publication ignores host shadows; CPU grouped publication uses host-owned
terminal rows. GPU-shaped engine units now use a host-injected mock worker.
Architecture estimate: LocalTP 95%, ExpertParallel 80%.

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
| ExpertOverlay | GPU hot + CPU cold | A | R | A | R | CUDA2 Dynamic prefix+MTP green; broader refresh pending |

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

Latest MoE blockers remain verifier producer, sidecar, and distribution build
time on both CUDA and ROCm; blanket main-graph recapture is retired.

## Focused Proofs

- `V2_Unit_MoEForbiddenDependencyScan` guards reset/workspace ownership and
  rejects the old combined verifier counter.
- CUDA/ROCm routed verifier microbench passed strict cos/L2/KLD/max_abs gates.
  ROCm M4: `0.1778 ms` graph vs `4.7366 ms` row replay. CUDA M4:
  `0.1035 ms` graph vs `9.8135 ms` row replay.
- ROCm verifier handoff reruns M4 after workspace rebind/reset.
- CUDA long-prompt MoE greedy parity and CUDA/ROCm stochastic verifier runs are
  green after pruning the broken combined owner.
- Replay preservation gates pass:
  `V2_Unit_{ForwardGraphTypes,PrefillGraphCache,GpuWorkspaceAllocationPolicy}`,
  prefill graph cache execution CUDA/ROCm, multi-turn reset, and Qwen3.6 MoE
  stochastic reset parity.
- CUDA/ROCm GDN and short-conv capture-lifetime matrices cover M=2/3/4 and
  every accepted row with byte-equal continuation and complete live state.
- CUDA Qwen3.6 MoE bucketed-prefill E2E passed `20/20`; capture launch `691 us`,
  replay `288 us`.
- CUDA2 ExpertOverlay Dynamic long-context prefix+MTP passed with explicit
  sidecar graph invalidation.
- LocalTP GPU grouped path: mirrored children publish resident outcomes;
  request prefill/sampling uses full heads, device-token matrices, resident
  conditions, and per-child mailboxes. No row replay, host plan/materializer,
  logits gather, or tiny allreduce. Gates: focused `9/9`, full unit `518/518`,
  CUDA2/ROCm2 request batch `2/2`; prior LocalTP parity `13/13`.
- MPI/server regressions pass: MPI bootstrap, prefill/decode transition,
  CPU MTP thinking `27/27`, dense Qwen3.6 E2E `261/261`.
- Model-load/MTP lifecycle guards pass: `V2_Unit_NodeLeaderPageCache` and
  `MTPMoESidecarReusesPreparedExpertSlabsAfterRawRelease`.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Close captured collective and remote-participant lifetime for LocalTP/EP.
2. Prove mirrored-head resident request batching across ExpertParallel modes.
3. Attack MoE verifier producer economics directly. ROCm routed FFN and CUDA
   dense GEMM buckets are the next largest pieces.
4. Prefer grouped/concurrent decode-equivalent kernels over serial row replay.
5. After each concrete win: run strict parity, refresh tok/s, update
   this dashboard, and make a WiP commit.
