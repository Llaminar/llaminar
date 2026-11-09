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
