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
