# COSMA Prefill Integration Plan

Status: Phase 1 Complete / Phase 1b & Phase 2 Roadmap Active
Scope (Phase 1): Use COSMA only for large prefill (context build) phases; preserve OpenBLAS path for autoregressive decoding and small ops.
Scope (Phase 1b): Stabilization, observability, path coverage, memory accounting groundwork.
Scope (Phase 2): Performance optimizations (fused dequant, in-layout elementwise, overlap streaming, adaptive thresholds, tiling prototype).

## 1. Requirements & Constraints
- Activate COSMA only when (prefill && seq_len >= threshold && !ADAPTIVE_DISABLE_COSMA).
- Avoid full duplication of model weights in COSMA layout (stream blocks on demand).
- Support quantized GGUF / ggml weights: fuse dequantization + layout transform directly into local COSMA blocks.
- Minimize extra copies: at most one transient staging buffer per weight block (or none with fused path).
- Keep test suite functional; fallback seamlessly if COSMA disabled or errors occur.
- Provide instrumentation (timing, bytes processed, relative L2 spot checks in debug).

## 2. High-Level Architecture
1. Decision Gate: Sequence length & env flags decide COSMA vs OpenBLAS.
2. Strategy Cache: Reuse cosma::Strategy objects keyed by bucketed (m,n,k,P).
3. Weight Streaming: Load only local blocks for current matmul, dequant + reorder inline.
4. Activation Conversion: One-time row-major → COSMA conversion per layer input; reuse for multiple projections (Q/K/V).
5. Matmul Execution: Use CosmaMatrix (C++ interface) for correctness; later refine to custom layout if needed.
6. Intermediate Ops: Convert to row-major only for elementwise kernels not yet ported (softmax, RMSNorm, SwiGLU).
7. Memory Reclamation: Release (or pool) weight blocks immediately after final use in the layer.
8. Output Handoff: Convert final layer output back to row-major for downstream/decode path.

## 3. Weight Streaming Strategy
- Determine local blocks via cosma::Mapper / CosmaMatrix metadata.
- For each owned block (rows [r0,r1), cols [c0,c1)):
  - Compute source byte offsets in original (possibly quantized) tensor.
  - Dequantize directly into destination column-major buffer (ld = local_rows) with fused reorder:
    dst[i + j*ld] = dequant(src_row = r0+i, src_col = c0+j)
- No global broadcast; each rank loads its share.
- Provide small WeightBlockCache to allow reuse within layer (e.g., if the same weight appears twice—rare, but future-proof).

## 4. Activation Handling
- Convert input activation once: row-major (m×h) → COSMA (col-major blocks).
- Reuse across Q/K/V projections.
- Keep intermediate Q, K, V in COSMA; only convert to row-major when an op (softmax) lacks COSMA implementation.

## 5. Intermediate Representation & Elementwise Ops
- Phase 1: Perform softmax, RMSNorm, SwiGLU in row-major (temporary conversions).
- Provide adapters:
  - rowMajorToCosma(float* src, CosmaView dst)
  - cosmaToRowMajor(CosmaView src, float* dst)
- Phase 2: Implement block-wise elementwise ops directly on COSMA local blocks to reduce conversions.

## 6. Memory Budget & Tiling
- Environment vars:
  - LLAMINAR_COSMA_PREFILL_THRESHOLD (default: 4096)
  - LLAMINAR_COSMA_MAX_RESIDENT_MB (default: 2048)
  - LLAMINAR_COSMA_VALIDATE_TILE (optional small tile size for spot checks)
- Preflight: Estimate memory for (activation blocks + largest weight local blocks + one output) — if over budget, either tile sequence or fallback.
- Tiling (Phase 2+): Split sequence into tiles processed sequentially; requires attention adaptation (deferred).

## 7. API & Helper Functions
`CosmaPrefillManager` (singleton or context-owned):
- bool enabled_for(int seq_len) const
- CosmaView convert_activation_in(const float* row_major, int m, int k)
- CosmaWeightHandle load_weight(const WeightDescriptor& desc)
- CosmaView matmul(const CosmaView& A, const CosmaWeightHandle& W,
                   int m, int k, int n, bool transposeW=false,
                   float alpha=1.f, float beta=0.f)
- void to_row_major(const CosmaView& src, float* dst)
- void release_weight(CosmaWeightHandle&&)

Support classes:
- StrategyCache (unordered_map key -> Strategy)
- BlockAllocator (pooled aligned float buffers; NUMA-aware)
- WeightBlockCache (optional; key: tensor_id|block_id)
- WeightDescriptor { id, dims (k,n), quant_type, base_ptr, strides }
- CosmaView { CosmaMatrix<float>* mat, dims, helper iterators }

## 8. Prefill Layer Pseudocode (Simplified)
```
if (!cosma_mgr.enabled_for(seq_len)) return fallback_layer();
A = cosma_mgr.convert_activation_in(hidden_in, seq_len, hidden_size)
Wq = cosma_mgr.load_weight(desc(Q))
Wk = cosma_mgr.load_weight(desc(K))
Wv = cosma_mgr.load_weight(desc(V))
Q = cosma_mgr.matmul(A, Wq, seq_len, hidden_size, hidden_size)
K = cosma_mgr.matmul(A, Wk, seq_len, hidden_size, hidden_size)
V = cosma_mgr.matmul(A, Wv, seq_len, hidden_size, hidden_size)
release Wq,Wk,Wv
Scores = cosma_mgr.matmul(Q, K (transpose), seq_len, hidden_size, seq_len)
Scores_row = temp; cosma_mgr.to_row_major(Scores, Scores_row)
softmax_in_place(Scores_row)
Scores_cosma = cosma_mgr.convert_activation_in(Scores_row, seq_len, seq_len)
Context = cosma_mgr.matmul(Scores_cosma, V, seq_len, seq_len, hidden_size)
Wo = cosma_mgr.load_weight(desc(OUT))
AttnOut = cosma_mgr.matmul(Context, Wo, seq_len, hidden_size, hidden_size)
release Wo
// FFN
Wup,Wgate,Wdown = load weights sequentially; compute Up, Gate, convert, SwiGLU, convert back, DownOut
hidden_out_row = final buffer; cosma_mgr.to_row_major(DownOut, hidden_out_row)
```

## 9. Instrumentation & Validation
Metrics per layer:
- matmul_time_ms, dequant_time_ms, convert_time_ms
- bytes_dequantized, bytes_reused
- effective_GFLOPS
- (Debug) relative L2 on validation tile vs OpenBLAS
Logging levels controlled via LLAMINAR_COSMA_LOG_LEVEL.

## 10. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Incorrect block mapping | Wrong results | Start with CosmaMatrix interface; add assertions on norms vs reference tile |
| Excessive conversions | Performance loss | Fuse dequant+reorder; batch conversion; later elementwise in COSMA |
| Memory spikes | OOM | Preflight budgets & serialize W loads |
| Quant variety | Complexity | Pluggable dequant functor registry |
| Softmax bottleneck | Latency | Phase 2: COSMA-native softmax |
| Strategy cache bloat | Memory | Bucket dimensions (round to 64) |

## 11. Roadmap (Updated)

| Phase | Theme | Items |
|-------|-------|-------|
| 1 (DONE) | Correctness & Core Path | Gating, strategy cache, float streaming, activation convert, validation tile, replicated compare, fast path skip, watchdog tests, buffer double-allocation guard. |
| 1b | Stabilization & Observability | Cumulative memory accounting (per-allocation + running total), hash/checksum option, orientation auto-fix tests, structured stats export (JSON/CSV), path forcing tests, env var audit test, MPI wrapper consolidation. |
| 2 | Performance & Layout | Fused quant dequant+layout (Q4_0 first), RMSNorm & SwiGLU in-layout, distributed softmax (naive), streaming overlap (double buffer), adaptive fast/direct threshold auto-tuning, memory tiling prototype, >2 rank strategy evaluation. |
| 3 | Advanced Optimization | FlashAttention-style attention, mixed precision accumulation, persistent weight block cache, strategy eviction (LRU), request concurrency, GPU parity exploration. |

## 12. Phase 1 Implementation Steps
1. Add `CosmaPrefillManager` skeleton & headers.
2. Integrate decision gate in prefill pipeline (leave existing decode path untouched).
3. Implement activation conversion + simple matmul wrapper using CosmaMatrix.
4. Implement weight block streaming (float first; quant stubs).
5. Add instrumentation hooks & env flags.
6. Debug correctness on large prefill test; add tile validation fallback.
7. Optimize: fuse dequant + layout; release provisional layout-based code behind flag.

## 13. Environment Variables (Extended)

Core Gating & Thresholds:
- `LLAMINAR_COSMA_PREFILL_THRESHOLD` (default 4096)
- `ADAPTIVE_DISABLE_COSMA` (disable COSMA entirely)
- `LLAMINAR_COSMA_FAST_PATH_THRESHOLD` (volume m*n*k for fast path replicated cutoff)
- `LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS` (volume threshold for auto direct COSMA selection)
- `LLAMINAR_COSMA_FORCE_DIRECT` (force direct COSMA path)
- `LLAMINAR_COSMA_FORCE_REPLICATED` / `LLAMINAR_COSMA_FORCE_REPLICATED_DIAG`

Validation & Diagnostics:
- `LLAMINAR_COSMA_VALIDATE_TILE` (tile size >0 enables)
- `LLAMINAR_COSMA_COMPARE_REPLICATED` (full replicated reference compare)
- `LLAMINAR_COSMA_FAST_UNVERIFIED` (skip validation for fast path)
- `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE`
- `LLAMINAR_COSMA_DEBUG_RECON`
- `LLAMINAR_COSMA_DUMP_SMALL` (small matrix structural dump)

Dequant / Streaming:
- `LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT` (disable fused path when introduced)
- `LLAMINAR_DEQUANT_STATS`, `LLAMINAR_DEQUANT_ANOMALIES`

Logging / Stats:
- `LLAMINAR_COSMA_LOG_LEVEL` (trace..error)
- `LLAMINAR_COSMA_DUMP_STATS` (aggregate counters at shutdown)

Memory / Safety:
- `LLAMINAR_COSMA_MAX_RESIDENT_MB`

Test / Watchdog:
- `LLAMINAR_COSMA_TEST_PHASE_TIMEOUT_MS`
- `LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS`
- `LLAMINAR_COSMA_TEST_TRACE`
- `LLAMINAR_SKIP_MPI_IN_SINGLE_TEST`

Performance Runtime:
- OpenMP / BLAS: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND`, `KMP_AFFINITY`, `KMP_BLOCKTIME`

## 14. Acceptance Criteria

### Phase 1 (Achieved)
- Large distributed prefill correctness: replicated comparison rel L2 < 1.5e-2 (observed 0.0 on test shapes).
- Fast path correctness parity (small volume) rel L2 < 1e-4.
- Validation tile optional & counters increment.
- No uncontrolled hangs: watchdog + external 60s timeout in tests.
- Double allocation crash eliminated (guard added).

### Phase 1b (Target)
- Path coverage tests: force direct, force replicated diag, replicated compare, fast-unverified suppression all validated.
- Cumulative memory accounting warns when projected peak > `LLAMINAR_COSMA_MAX_RESIDENT_MB`.
- JSON stats dump when `LLAMINAR_COSMA_DUMP_STATS=1` (includes strategy hit/miss, per-path counts, bytes, timings).
- Orientation auto-fix test demonstrates correction without exceeding rel L2 threshold.
- Env var audit test ensures documentation sync (all known vars recognized by manager or test registry).

### Phase 2 (Entry Criteria for Completion)
- Fused dequant implemented for at least one quant format (Q4_0) with rel L2 vs unfused < 1.5e-2.
- RMSNorm + SwiGLU in-layout (no intermediate row-major conversion) for attention & MLP blocks.
- Streaming overlap yields >=10% wall time reduction on benchmark (documented with harness).
- Adaptive threshold tuning adjusts fast path threshold within first N (e.g., 5) large ops (logging decisions).
- Tiling prototype engages gracefully when memory estimate > budget (functional correctness maintained).

## 15. Unit Test Coverage Plan (Additions)

New / Enhanced Tests:
1. `CosmaPrefillManagerPathsTest.DirectCompare` – uses `LLAMINAR_COSMA_FORCE_DIRECT=1` + `LLAMINAR_COSMA_COMPARE_REPLICATED=1`; asserts `cosma_path_calls` increments and rel L2 == 0 within tolerance.
2. `CosmaPrefillManagerPathsTest.ForceReplicatedDiag` – sets `LLAMINAR_COSMA_FORCE_REPLICATED_DIAG=1`; checks delta in `fast_path_calls` and no direct path counters.
3. `CosmaPrefillManagerPathsTest.FastPathUnverified` – sets small shape + `LLAMINAR_COSMA_FAST_UNVERIFIED=1` + tile validation env; ensures validation counter unchanged while fast path count increments.
4. `CosmaPrefillManagerPathsTest.AutoDirectThreshold` – clears forcing vars, selects shape volume > `LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS`; expects cosma path (delta counters) without replicated compare.
5. `CosmaPrefillManagerOrientationAutoFixTest` – (Phase 1b) intentionally supplies transposed operand metadata to trigger `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE` and asserts warning + acceptable rel L2.
6. `CosmaPrefillManagerEnvAuditTest` – enumerates documented env vars, sets each to sentinel, invokes a benign call, and reports any unknown leftover vars (failing if mismatch).

Each test snapshots stats before/after to avoid cross-test interference; skips multi-rank-only scenarios when `world < 2`.

## 16. Metrics & Instrumentation Enhancements
- Add per-op record: `{backend, path, m,n,k, world, ms, gflops, validations}` buffered and flushed on shutdown or when buffer size exceeds threshold.
- Provide helper `dump_stats_json(const std::string &path)` callable from tests.

## 17. Risks (Updated)
| Risk | Phase | Mitigation |
|------|-------|-----------|
| Singleton counters accumulate across tests | 1b | Snapshot deltas; optional reset API (guarded). |
| Overhead of replicated compare on large ops | 1b | Warn if size > threshold; recommend tile validation first. |
| Fused dequant precision drift | 2 | Retain unfused fallback & A/B compare harness. |
| Overlap introduces race conditions | 2 | Use double-buffer state machine with explicit barriers; add TSAN build (optional). |
| Tiling complicates attention semantics | 2 | Start with MLP-only tiling; extend to attention after Flash-style design. |

## 18. Definition of Done Summary
- Phase 1: Achieved and locked (no functional regressions allowed without tests).
- Phase 1b: Focused on safety + observability → must pass new path + audit tests and produce structured stats.
- Phase 2: Performance gains + architectural groundwork for advanced attention & memory efficiency.

---
End of Plan (Living Document – update as phases progress).
