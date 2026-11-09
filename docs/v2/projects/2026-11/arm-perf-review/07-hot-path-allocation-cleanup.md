# Idea 7 — Hot-path allocation / cache cleanup (shared CPU code)

Status: **Open**. These are shared x86+ARM paths, not ARM-specific. They show
up in the ARM hot path once the engine runs and are best as separate small PRs.
Some already have a fatal/typed cleanup pattern to follow (`rejectNumeric`,
idea 10).

## Items

1. **RoPE pair-position tables** —
   `src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp`:
   - `apply_rope_pair_positions()` (~:444) allocates two `std::vector<float>`
     cos/sin tables (rows × head_dim/2) on every call;
   - a second allocation site exists for the seq-len path (~:776).
   - `get_inv_freq_cached()` (~:51, :62) takes a **global mutex** on every call
     (including cache hits) and stores `std::unordered_map` keyed by
     (head_dim<<32 | freq_bits).
   Ideas: bounded stack/arena angle buffers or a persistent per-invocation
   buffer owned by the caller; lock-free/sharded or thread-local inv-freq cache
   (values are determined by (head_dim, freq_base) at model load).

2. **RMSNorm FP32 generic path heap fallback** —
   `src/v2/kernels/cpu/ops/CPURMSNormKernelT.cpp`:
   - `MAX_STACK_ROW_SIZE = 4096` (~:42); rows with `ucols > 4096` fall back to a
     per-call `std::vector` (`heap_scratch.resize`). Any hidden size >= 5120
     allocates per call.
   - Native BF16/FP16/INT32 branches already avoid decoded row banks; the FP32
     residual/remaining branches are the ones still needing work.

3. **Gamma Q12 TLS cache** —
   `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp` (~:3120):
   - `thread_local std::vector<int16_t> gamma_q12_cache` keyed by `gamma`
     pointer + cols. Keying by raw pointer is a stale-identity risk: freed and
     reallocated same-address memory with different values would return stale
     quantized gamma (correctness, not just speed). It is a known risk; the
     dedicated fix is to key by an authentic generation/owner identity or move
     the quantized copy into the retained owner.

## Exactness / risk

- Item 1: angle tables are pure coefficient data; caching must preserve exact
  values (the libm cos/sin merge defect found in the indexer shows separate
  calls matter — keep the same call sequence).
- Item 2: scratch is transient; no arithmetic change, but the fallback must be
  covered by the existing per-caller ownership regressions.
- Item 3: arithmetic is identical when the cache is correct; the risk is
  correctness under reuse (needs the typed identity).

## Validation

- Perf A/B for items 1–2 (allocations removed from per-token path);
  focused regressions for item 3 (same-address different-weights reuse).

## Ideas 25+7 lane plan (2026-11-09) — scope confirmed

- F1: per-bucket PerfStats counters ALREADY EXIST (forward_graph/
  prefill_graph_launch + stage_gpu/prefill_graph.replay + lifecycle + cache
  hit; LLAMINAR_PERF_STATS_CSV + GPU_STAGE_TIMING) — 01#6 'only
  moe_placement/mtp/physical_memory' is STALE (corrected). Per-bucket probe
  possible NOW without nsys (GPU-timing run = profiling; separate wall run).
- F2: MAX_STACK_ROW_SIZE heap fallback is non-production (FP32 path uses no
  scratch; BF16/FP16/Q8_1 only); hidden 5120 does NOT allocate in production.
  Still fixed (7c: in-place residual-add + single-row buffers, stack bound
  8192 floats, allocate-but-correct above).
- F3: gamma_q12 TLS cache = x86-AVX512-only (rmsnorm_q8_1_pure_integer), non-
  production; real stale-identity defect → fix via authentic owner identity
  (TensorBase::weak_from_this(); absent owner → per-call quantise); regression
  on WSL x86 AVX512.
- F4 (production): FP32 + partial RoPE allocate 2× std::vector cos/sin per
  call (per bucket per layer prefill; seq_len=1 decode in partial path; every
  team thread allocates+fills its own copy) → 7a admitted cpu_rope_angles
  workspace (CPUInvocationWorkspace doctrine; missing = fatal; shared fill via
  omp for; byte-identical angle elements; WorkspaceMemoryEstimator admission
  as gate) + 7a2 VerifierAngleRows (thread_local growable) onto the same
  buffer (approved). 7b inv-freq: lock-free thread_local memo (reference-
  stable unordered_map nodes), mutex only on miss.
- 25a: add host-clock-only counters (prefill_chunk_host_wall, inter_chunk_gap,
  family_host_wall, request-intercept split) — no sync/D2H, domain-gated.
- 25b: arch-aware install-cuda.sh (x86_64 vs sbsa) + probe/runbook; recipe
  owner = arm-sparks.
- Commit order: 7d → 7b → 7a → 7c → 25a → 25b (25c tracker-only). Reviewer at
  clean checkpoint → handover to Main. Dead-code flag (apply_with_residual_add
  no production callers) + RMSNorm-no-workspace sent to Main.

## Ideas 7+25 lane progress (2026-11-09, branch rebased onto 302318f1e, local)

- 06b205ccd fix(cpu) 7d: Q8_1 pure-integer gamma keyed by owner identity
  (weak_from_this + address + width; quantisation split into own primitive;
  empty/expired owners requantise per call, never cached). Core-free
  regression + negative control (address-only key fails the 3 cases) in
  ProductionTestPreflight; AVX512 syntax probe on Mac; real AVX512 exec = WSL.
- a588345ac perf(cpu) 7b: inv_freq hits lock-free via 8-entry per-thread memo
  of published node addresses (map never erased; values/storage unchanged);
  regression 12 geometries × 6 threads → single address + bit-exact values.
- c93a2d1a8 build(docker) 25b: install-cuda.sh arch-aware (22.04|24.04 ×
  amd64→x86_64 | arm64→sbsa; else fail closed; x86 unchanged); MODE=profilers
  (CUPTI + nsight-compute + nsight-systems); validate-cuda-profilers.sh
  (identity + cuda,nvtx trace must contain kernels + cudaGraphLaunch);
  mapping test in preflight image-contract suite. uma4 (FROM uma3 +
  MODE=profilers) + runbook → arm-sparks builds/validates.
- Mac: portability NEON_DOTPROD Release 261/261 after 7d; RoPE subset 20/20
  after 7b.
- Decisions folded: 7c does NOT touch apply_with_residual_add; >8192 → FAIL
  FAST (fatal naming width+bound) — declared cpu_rmsnorm_rows workspace =
  tracked follow-up; 7a/7a2 via RoPEStage IWorkspaceConsumer, sizing from
  admitted max rows (incl. grouped verifier rows), bind before capture/replay,
  lifecycle reset/replay test, no thread_local cos/sin left in RoPE TU.
- Process: local --no-verify OK (macOS can't configure full engine); WSL x86
  gate = real prerequisite; branch stays local until gate + handover.

## Follow-up A (from the ideas 7+25 lane, allowlisted '07 follow-up A')

Convert the TEST-ONLY Q16/hybrid RoPE family (apply_rope_q8_1_to_fp32,
q16_1_integer, q16_integer, q8_1→q16 incl. fixed-scale/head variants,
q16→q16 fixed/dynamic, compute_rope_sincos_q15) to the admitted angle
workspace. Proven unreachable in production (QwenGraphBase addRoPE /
addKeyCachePublicationTransforms = FP32 Q/K, no Q_out; key-only factory
rejects Q_out/K_out at ComputeStageFactory.cpp:111; ActivationPrecisionPolicy
rejects non-FP32; GraphResolver sets no outputs — allowlisted by function in
the source policy). Dead persistent_state parameter removed.

## Tip + Main follow-ups (2026-11-09) — pushed, authoritative = 94d7cbe2f

- RoPEStage creates its kernel EXACTLY ONCE; type mismatch FATAL
  (LLAMINAR_UNREACHABLE names both types); recreate/rebind removed.
- Angle workspace sized from EFFECTIVE ROTATED WIDTH (partial-RoPE prefix),
  NOT head_dim — stage/kernel/estimator same contract; estimator via
  ModelMemoryProfile::rope_dimension_count / effectiveRotaryDimension();
  profile wire version 6→7 (stale layouts fail); exact-byte asserts full +
  partial widths.
- Two focused preflight regressions (production defects fixed in e17c08412):
  CPURoPEQ8OneRowTeamHeadOwnership (retained team rotates each shared head
  once) + CPURoPEQ8OneRowNegativePosition (independent of thread's previous
  call) — both reproduce the defect against the BASE TU in standalone control.
- Mac portability: 270/270 NEON_DOTPROD + 268/268 NEON @8a553c922, 0 skips;
  source policy clean. Bundle/patches stale; origin branch authoritative.

- PR #43 = 5922cd1f5: Main RETIRED the unused KernelFactory RoPE cache + two
  dead unregistered test sources flagged by this lane (7d/7a review). The only
  remaining failure in that filter = MoEForbiddenDependencyScan (known base).
