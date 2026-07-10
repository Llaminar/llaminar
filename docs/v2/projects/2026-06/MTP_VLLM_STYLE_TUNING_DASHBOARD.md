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

Request-boundary `clear_cache()` now separates lazy init from request state;
CUDA Qwen3.6 MoE E2E passed `20/20` with ready graph replay every run.

CUDA attention dynamic params no longer use ad hoc `cudaMallocHost`; decode
partials/params use fixed staging plus declared workspace buffers.

CUDA dense greedy d2/d3 acceptance recovered after verifier-row ownership fix.

2026-07-08 LocalTP GPU MTP uses mirrored full-head child-resident outcomes.
Initial shifted row and suffix catch-up are now prepared from resident compact
outcome metadata/tokens before publication; the host bridge is response-only.
GPU all-position verifier publication now hard-fails without resident compact
outcome publication; host `MTPSpecStepPlanBatch` publication is CPU-only.
Focused MTP/Rank/runner gates pass; full unit `517/517` passes for this slice.

2026-07-10 grouped proof: canonical gate passed `43/43` (CPU 13, CUDA 13,
ROCm 17). GPU request-batch accept/residual/recovery/bonus draws now consume
resident positions in captured CUDA/ROCm kernels; scalar/resident outputs are
byte-identical. Transition `125/125`, ownership policy `96/96`, full build
`935/935`, and explicit CUDA/ROCm sampling lanes pass.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stoch | MoE greedy | MoE stoch | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused |
| SingleDevice | CUDA d1 | G | G | R | R | Dense green; MoE below baseline |
| SingleDevice | ROCm d1 | A | A | A | A | MoE nearly break-even |
| LocalTP | CUDA deg2 | A | A | R | R | Mirrored resident outcome gate green; perf pending |
| LocalTP | ROCm deg2 | A | A | R | R | Mirrored resident outcome gate green; perf pending |
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

Latest MoE blockers remain verifier graph replay, sidecar, and distribution
build time on both CUDA and ROCm.

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
- CUDA Qwen3.6 MoE bucketed-prefill E2E passed `20/20`; capture launch `691 us`,
  replay `288 us`.
- CUDA2 ExpertOverlay Dynamic long-context prefix+MTP passed with explicit
  sidecar graph invalidation.
- CUDA attention guard rejects `cudaMallocHost` / `cudaFreeHost` regression.
- LocalTP GPU grouped path: mirrored children publish resident outcomes;
  greedy/stoch prelaunch uses rank resident mailboxes, request batches use
  device-token matrices/resident conditions, and target-slot commits fan out.
  Stop-first and ROCm segmented-env cases stay resident.
  No row replay, GPU host plans, base-cache vectors, or host materializer. Gates:
  focused units, full unit `517/517`; Qwen3.6 LocalTP parity `13/13` prior.
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
