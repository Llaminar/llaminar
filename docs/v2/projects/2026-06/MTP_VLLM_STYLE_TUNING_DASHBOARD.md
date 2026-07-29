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

The follow-on transfer-state audit added production-shaped cross-stream reset
coverage on CUDA and ROCm: each of 32 request epochs now publishes reset work
on one stream, records an event, and consumes the reset directory from a
different stream. Both canonical rebalance matrices pass `31/31`. The audit
also retired the unused ROCm-only histogram/expert-mask API, its blocking host
observation method, three dead launchers, four self-referential integration
tests, and two unconsumed persistent workspace buffers. Source synchronization
and forbidden-dependency guards remain green. Architecture estimate:
SingleDevice 95%, LocalTP 97%, ExpertParallel 95%.

Homogeneous graph-stable GPU Dynamic/LLEP no longer has a debug-environment
escape hatch to the host controller. Once that topology is recognized, every
invalid device-controller prerequisite is fatal instead of selecting host
publication/apply. The orchestrator's 45-test dependency suite, source policy
guards, and both CUDA/ROCm `31/31` rebalance/reset matrices pass. Accelerator
build caching was also verified directly for CUDA `nvcc` (including its
internal `cicc`/`ptxas` work) and ROCm clang 20; a repeated two-object probe
produced two direct hits, and container-wide cache identity/capacity is now
fixed at a workspace-relative 50 GB.

Fresh SingleDevice Release comparison uses `ggml-org/llama.cpp` master
`afeebe103bd99cda8f5dfaefcabadf890db7fda7`, the same raw prompt, greedy
sampling, 128 forced tokens, fixed d1/d3, and one GPU. llama.cpp runs a warmed
raw `/completion` request (596 prompt tokens including BOS); Llaminar reports
595 prompt tokens and averages three runs after warmup. CUDA dense d3 and ROCm
dense d3 beat llama.cpp. Both d1 lanes and both MoE d3 lanes miss, with the
largest gaps accompanying lower Llaminar MoE acceptance. CUDA dense at the
default 4096 context also fails workspace admission after model load; c1024,
which covers the 723-token workload, is green.

Legacy `LLAMINAR_PROFILING` is now a topology-neutral PerfStats alias. Fresh
CUDA/ROCm MoE d1 smokes each replayed all 30 grouped verifier calls as one
486-stage graph and emitted structured decode-step/maintenance timing; neither
backend selected eager or segmented execution.

## Device And Topology Matrix

| Mode | Device / degree | Dense greedy | Dense stoch | MoE greedy | MoE stoch | Status |
|---|---|:---:|:---:|:---:|:---:|---|
| SingleDevice | CPU d1 | R | R | A | A | CPU refresh paused |
| SingleDevice | CUDA d1 | A | G | A | R | d1 loses to llama.cpp; dense d3 wins |
| SingleDevice | ROCm d1 | A | A | A | A | d1/MoE d3 lose; dense d3 wins |
| LocalTP | CUDA deg2 | A | A | R | R | Resident request batch green; perf pending |
| LocalTP | ROCm deg2 | A | A | R | R | Resident request batch green; perf pending |
| LocalTP | ROCm deg4 | A | R | R | R | Preset/bench refresh pending |
| LocalPP | CUDA stages | A | R | R | R | Correctness/bench refresh pending |
| LocalPP | ROCm stages | R | R | R | R | Prior dense run speed-negative |
| NodeLocalTP | CPU sockets | A | A | R | R | Dense E2E green; perf pending |
| ExpertOverlay | GPU hot + CPU cold | A | R | G | R | Dynamic/LLEP greedy+prefix green; stochastic/perf pending |

## SingleDevice Speeds

Matched llama.cpp comparison:

| Backend/model | Depth | Llaminar | llama.cpp | Ratio | Acceptance L/LC | RAG |
|---|---:|---:|---:|---:|---:|:---:|
| CUDA dense 27B | d1 | `36.24` | `59.64` | `0.61x` | `77.67/88.06%` | R |
| CUDA dense 27B | d3 | `64.84` | `60.91` | `1.06x` | `80.56/69.11%` | G |
| CUDA MoE 35B | d1 | `93.01` | `153.92` | `0.60x` | `80.28/98.44%` | R |
| CUDA MoE 35B | d3 | `159.61` | `171.81` | `0.93x` | `70.27/93.94%` | R |
| ROCm dense 27B | d1 | `20.08` | `25.15` | `0.80x` | `89.55/88.06%` | R |
| ROCm dense 27B | d3 | `39.28` | `22.66` | `1.73x` | `80.56/69.11%` | G |
| ROCm MoE 35B | d1 | `46.33` | `73.93` | `0.63x` | `86.76/98.44%` | R |
| ROCm MoE 35B | d3 | `74.94` | `95.84` | `0.78x` | `78.30/94.95%` | R |

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
