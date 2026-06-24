# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice, LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay. Keep under 6 KB.

RAG: **G** correct and speed-positive, **A** correct but slow/stale,
**R** failing, speed-negative, or unproven. Fresh rows use bounded
`--decode-tokens 16 --perfstats`.

## Snapshot

Fresh E2E: Qwen3.6 dense baseline/RAM-prefix/MTP d2 passed `261/261` on
CPU/CUDA/ROCm; CPU MoE MTP d2 passed `27/27`; CUDA Qwen3.5 MoE bucket E2E
passed `28/28`.

Request-boundary `clear_cache()` separates lazy init from request state.
Replay-safe forward, sidecar, initialized prefill entries, and ready graphs
survive reset; stage hooks clear request mirrors. CUDA Qwen3.6 MoE E2E passed
`20/20`; perfstats show bucket `1536` capture from
`lazy_initialized_after_request_reset` plus ready replay every run.
Graph-on/off benchmark gauge, Qwen3.6 MoE CUDA, `595` prefill tokens:
prefill `192.4 ms / 3092 tok/s` on vs `226.0 ms / 2632 tok/s` off
(`-14.9%` latency, `+17.5%` throughput). Decode also improved in that benchmark
slice, but needs an isolation pass for attribution.

CUDA attention dynamic params no longer use ad hoc `cudaMallocHost`; decode
partials/params use fixed staging plus declared workspace buffers.

CUDA dense greedy d2/d3 acceptance recovered after verifier-row ownership fix.

Accepted MoE verifier route: routed experts use grouped verifier; shared expert
uses decode-equivalent GEMV-many plus normal shared-gate combine. Do not revive
combined routed+shared without strict L2/KLD/cosine/max_abs/token proof.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stoch | MoE greedy | MoE stoch | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused |
| SingleDevice | CUDA d1 | G | G | R | R | Dense green; MoE below baseline |
| SingleDevice | ROCm d1 | A | A | A | A | MoE nearly break-even |
| LocalTP | CUDA deg2 | R | R | R | R | Dense greedy accepts 0; stochastic unsupported |
| LocalTP | ROCm deg2 | R | R | R | R | Fixed d1 segfaults in LocalTP allreduce |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Dense E2E green; perf pending |
| ExpertOverlay | GPU hot + CPU cold | A | R | A | R | Skipped in latest refresh per CPU pause |

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

Latest MoE stage blockers:

| Device | Main verifier | Stage body | Largest buckets |
|---|---:|---:|---|
| CUDA | `386.4 ms` | `23.7 ms` | graph replay `320.5`, sidecar `40.3`, dist build `19.7` |
| ROCm | `422.9 ms` | `26.9 ms` | graph replay `349.0`, sidecar `60.9`, dist build `58.0` |

## Focused Proofs

- `V2_Unit_MoEForbiddenDependencyScan` guards reset/workspace ownership and
  rejects the old combined verifier counter.
- CUDA/ROCm routed verifier microbench passed strict cos/L2/KLD/max_abs gates.
  ROCm M4: `0.1778 ms` graph vs `4.7366 ms` row replay. CUDA M4:
  `0.1035 ms` graph vs `9.8135 ms` row replay.
- ROCm verifier handoff reruns M4 after workspace rebind/reset.
- CUDA/ROCm shared direct and `SharedExpertFFNStage` M=2/3/4 gates pass.
- 2026-06-23 ROCm Qwen3.6 MoE verifier rows M1-M4 grouped/rowwise pass after
  forcing main all-position ROCm rows onto decode-equivalent replay.
- CUDA long-prompt MoE greedy parity and CUDA/ROCm stochastic verifier runs are
  green after pruning the broken combined owner.
- Replay preservation gates pass:
  `V2_Unit_{ForwardGraphTypes,PrefillGraphCache,GpuWorkspaceAllocationPolicy}`,
  prefill graph cache execution CUDA/ROCm, multi-turn reset, and Qwen3.6 MoE
  stochastic reset parity.
- CUDA Qwen3.6 MoE bucketed-prefill E2E passed `20/20`; capture launch `691 us`,
  replay `288 us`.
- CUDA attention guard rejects `cudaMallocHost` / `cudaFreeHost` regression.
- MPI/server regressions pass: MPI bootstrap, prefill/decode transition,
  CPU MTP thinking `27/27`, dense Qwen3.6 E2E `261/261`.
- Model-load/MTP lifecycle guards pass: `V2_Unit_NodeLeaderPageCache` and
  `MTPMoESidecarReusesPreparedExpertSlabsAfterRawRelease`.
- Token equality alone is not an accepted verifier parity proof.

## Next Phase 10 Moves

1. Keep SingleDevice priority: dense first, then MoE, then LocalTP/LocalPP.
2. Attack MoE verifier producer economics directly. ROCm routed FFN and CUDA
   dense GEMM buckets are the next largest pieces.
3. Prefer grouped/concurrent decode-equivalent kernels over serial row replay.
4. Eliminate D2H/H2D bridges and graph rebuilds before polishing controllers.
5. After each concrete win: run strict focused parity, refresh tok/s, update
   this dashboard, and make a WiP commit.
