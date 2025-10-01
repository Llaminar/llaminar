# Llaminar
An LLM inferencing engine using COSMA / Open MPI for hyper-scalability 

## Quick Start

### Building
```bash
# Configure and build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

## Debug & Instrumentation Environment Variables

The following environment variables enable additional logging, validation, and instrumentation when developing or debugging Llaminar:

| Variable | Purpose | Values / Default | Notes |
|----------|---------|------------------|-------|
| `LLAMINAR_DEQUANT_STATS` | Emit per-tensor dequantization statistics (min, max, mean, sample values). | `0` (disabled) or `1` (enabled); default: disabled | Affects all quantized tensor formats. Uses `logDequantStats` in `model_loader.cpp`. |
| `LLAMINAR_DEQUANT_ANOMALIES` | Log anomalies during Q6_K (and future) dequant: NaN, Inf, or extreme magnitudes. | Any non-empty value enables | Currently instrumented for `Q6_K`. Falls back to `LLAMINAR_DEQUANT_STATS` if that is enabled. |
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Minimum sequence length to engage COSMA-prefill path. | Integer; default: `4096` | See COSMA Prefill plan docs. |
| `ADAPTIVE_DISABLE_COSMA` | Force-disable COSMA path regardless of length. | Any non-empty value disables | Helpful for A/B comparisons. |
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Soft cap on resident COSMA working set in MiB. | Integer; default: `2048` | Prefill manager may fallback if estimate exceeds budget. |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Enable small tile correctness spot-check (relative L2) vs OpenBLAS. | Integer (tile size) or `0` to disable; default: `0` | Debug / validation only. |
| `LLAMINAR_COSMA_LOG_LEVEL` | Verbosity for COSMA prefill instrumentation. | `trace|debug|info|warn|error`; default: `info` | Independent of global logging level. |
| `OMP_NUM_THREADS` / `OMP_*` | Controls OpenMP threading behavior. | See `run-llaminar.sh` | Script sets optimal defaults (NUMA-aware binding). |
| `KMP_AFFINITY` / `KMP_BLOCKTIME` | Fine-grain thread affinity & spin-wait tuning. | See script defaults | Only relevant for Intel OpenMP runtime. |
| `LLAMINAR_EMBED_TRACE` | Enable detailed per-rank embedding buffer diagnostics (min/max, NaN/Inf counts, token & value preview). | Any non-empty (not `0`) enables; default: disabled | Produces TRACE logs labelled `MPIEmbeddingKernel[local_pre_gather|global_post_gather]`. Combine with `LLAMINAR_EMBED_TRACE_TOKENS` / `LLAMINAR_EMBED_TRACE_DIMS`. |
| `LLAMINAR_EMBED_TRACE_TOKENS` | Limit number of tokens shown in embedding diagnostics. | Positive integer; default: `2` | Ignored unless `LLAMINAR_EMBED_TRACE` enabled. |
| `LLAMINAR_EMBED_TRACE_DIMS` | Limit number of embedding dims per token shown in diagnostics. | Positive integer; default: `8` | Ignored unless tracing enabled. |
| `LLAMINAR_EMBED_FAIL_FAST` | Abort immediately if a non-finite (NaN/Inf) value is detected after copying an embedding row. | Any non-empty (not `0`) enables; default: disabled | Provides detailed token/dim context then `abort()`. Without this set a WARN is logged and execution continues. |

### Quick Usage Examples

```bash
# Enable dequant statistics and anomaly logging
export LLAMINAR_DEQUANT_STATS=1
export LLAMINAR_DEQUANT_ANOMALIES=1

# Force OpenBLAS-only path for comparison
export ADAPTIVE_DISABLE_COSMA=1

# Lower threshold to exercise COSMA prefill with small sequences
export LLAMINAR_COSMA_PREFILL_THRESHOLD=512

# Enable embedding diagnostics
export LLAMINAR_EMBED_TRACE=1
export LLAMINAR_EMBED_TRACE_TOKENS=4
export LLAMINAR_EMBED_TRACE_DIMS=16
# Fail fast on non-finite embedding values
export LLAMINAR_EMBED_FAIL_FAST=1
```

> Tip: Keep instrumentation disabled for performance benchmarking; some options add measurable overhead (e.g., validation tiles, verbose stats).

```

### Running (Canonical Method)
```bash
# Always use the canonical launcher for optimal performance
./run-llaminar.sh [arguments]


# Examples:
./run-llaminar.sh -v --print-topology                    # System info
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v  # Model inference
```

The `run-llaminar.sh` script automatically configures optimal MPI and OpenMP settings:
- **OpenMP**: 28 threads per socket, socket placement, close binding
- **MPI**: 1 process per socket with memory pinning and NUMA awareness
- **Threading**: Adaptive backend selection (single/multi/distributed)

### Manual MPI Execution (Advanced)
```bash
# If canonical script is unavailable
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  ./build/llaminar [arguments]
```

## Architecture

### Architecture Overview (ASCII Diagram)
```
         +-----------------------------------------+
         |            Inference Flow               |
         +-----------------------------------------+
   Model Load & Init (GGUF parse, quant tensors, topology)      
         |
         v
     +----------------------+    Environment / Heuristic    
     | Distribution Mode    |<-----------------------------+
     |  replicated / sharded|   (param count, mem fraction)
     +----------+-----------+
          |
          v
     +----------------------+      +----------------------+
     | Shard / TP Metadata  |      |   Debug Env Snapshot |
     |  ShardSpec (heads/h) |      |  (prefill thresholds) |
     |  TPPartitionSpec (*) |      |                      |
     +----------+-----------+      +----------+-----------+
          |                            |
          +--------------+-------------+
                |
                v
          +-------------------------------+
          |      Layer Forward Pass       |
          +-------------------------------+
          |  Attention  |   MLP   | Norm  |
          +------+------+--------+--------+
           |         |        |
           v         v        v
       (Per MatMul) Backend Selection
           |
       +---------+------------------+
       | PrefillBackend (large seq) |
       | InferenceBackend (decode)  |
       +---------------+------------+
              |
            launch(OpDesc,Ctx)
              |
        +--------------+---------------+
        |  Fallback if Unsupported ->  |
        |      adaptive_matmul()       |
        +--------------+---------------+
              |
          Heuristic Arbitration
              |
       +------------------+------------------+
       |     OpenBLAS (local)      | COSMA (dist) |
       +--------------+------------+--------------+
              |
            Local/Dist Output
              |
        +-----------+------------+
        | Attention Output Modes |
        | local | gather_pre     |
        | gather_post | replicated |
        +-----------+------------+
              |
              Residual
              |
            Next Layer → ...

(* planned future TP executor enabling row/col shard execution.)

Rank 0 One-Time Logs:
  MODEL_DIST ...
  BACKEND_DECISION_SUMMARY component=Attention ...
  BACKEND_DECISION_SUMMARY component=MLP ...
```


All execution paths now go through MPI-aware kernels; legacy non-MPI kernels (LinearKernel, AttentionKernel, RMSNormKernel, MatMulKernel) have been removed. Backend selection (OpenBLAS vs COSMA) is centralized in `adaptive_matmul.h`, ensuring a single decision point and preventing divergence between sequential and distributed code paths.

Design principles:
1. Single backend arbitration layer (adaptive_matmul)
2. MPI kernels handle both single-rank (np=1) and multi-rank modes
3. No duplicate sequential test harnesses; tests target MPI variants directly
4. Future distributed layout optimization (COSTA) will replace copy-in/copy-out in COSMA path (TODO)

Implication: For local benchmarking use `mpirun -np 1` instead of reinstating legacy kernels.

### Attention Output Contract (Post-Gather Removal)
The MPI attention kernel now always emits a per-rank *partial* output (only the contribution from
that rank's owned heads after the output projection). There is no internal gather or reduction.

Caller / pipeline responsibilities:
1. Determine reconstruction semantics: for the current row-sharded `Wo` distribution the per-rank
  outputs are additive and should be summed: `MPI_Allreduce` (SUM) over the full `[seq_len, d_model]` buffer.
2. Perform the reduction before any consumer treats the tensor as a fully replicated activation
  (e.g., before residual add or feeding into RMSNorm/MLP expecting full hidden state).
3. Avoid accidental direct use of the partial result as replicated. An optional safety canary can be
  enabled with `LLAMINAR_ASSERT_REPLICATED_MISUSE=1`, which injects a microscopic per-rank marker in
  the final element to aid detection during debugging.

Removed flags: `LLAMINAR_DISABLE_ATTENTION_GATHER` and `LLAMINAR_LEGACY_ATTENTION_GATHER`—they are no
longer recognized. All documentation or scripts referencing them should be updated accordingly.

Rationale: Eliminating the hidden gather makes communication explicit, prevents double reductions,
and simplifies extending the sharded pathway (e.g., future head-concat vs additive strategies).

### Supported Quantization Formats (Current)

The loader and dequant pipeline currently support the following GGUF tensor types:

- Q4_0
- Q5_0
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q4_K_M (alias layout of Q4_K with fused min variant)
- Q5_K
- Q6_K
- Q8_K
- F16 / F32 (unquantized)

Deprecated / removed (no longer recognized and will trigger an error if encountered in a model):

- Q4_1 (id 3)
- Q5_1 (id 7)
- Q8_1 (id 9)

Reason for removal: These legacy *_1 formats added maintenance complexity (especially Q5_1 with divergent 22 vs 24 byte block variants) without providing clear performance or accuracy wins over the maintained set. Models should be re-converted using any of the supported formats above. The loader now emits a clear diagnostic if a deprecated numeric id is found.

Implications:
1. Existing GGUF files using Q4_1 / Q5_1 / Q8_1 must be re-exported (e.g., via llama.cpp quantization tools) to one of the supported formats.
2. Tests referencing the removed formats have been pruned (e.g., Q5_1 layout micro test).
3. Numeric ids for removed formats are reserved; they will not be repurposed to avoid silent misinterpretation.

## COSMA Prefill (Phase 1)

Phase 1 introduces an optional COSMA-backed prefill path focused on large context construction (long sequence length matrix multiplications). Autoregressive decoding and small matrix products stay on the adaptive OpenBLAS path to avoid communication overheads.

### Engagement Criteria
- Enabled only when: (a) `seq_len >= LLAMINAR_COSMA_PREFILL_THRESHOLD`, (b) world size > 1, and (c) `ADAPTIVE_DISABLE_COSMA` is NOT set.
- Below a conservative volume (`m * n * k < LLAMINAR_COSMA_FAST_PATH_THRESHOLD`), a multi-rank "fast path" executes local OpenBLAS GEMMs plus a broadcast to avoid COSMA overhead.

### Data Flow (Simplified)
1. Row-major activation is converted (or reused if budget denied) into a temporary COSMA layout.
2. Weights are streamed into COSMA-distributed buffers (Phase 1: float32 only; quant fusion planned).
3. `cosma::multiply` executes with MPI barriers before/after to avoid collective hazards.
4. Optional validation tile spot-check (OpenBLAS reference on a small top-left tile) computes relative L2 error.
5. Outputs remain in COSMA layout until needed by elementwise kernels; copy back to row-major as required.

### Environment Variables (Prefill Specific)
| Variable | Purpose | Notes |
|---------|---------|-------|
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Sequence length required to enable COSMA prefill (default 4096). | Set lower to force early testing. |
| `LLAMINAR_COSMA_FAST_PATH_THRESHOLD` | Volume (`m*n*k`) below which a replicated local GEMM path is used. | Avoids COSMA startup & comm for small ops. |
| `ADAPTIVE_DISABLE_COSMA` | Forces all ops onto OpenBLAS adaptive paths. | A/B performance & debugging. |
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Soft upper bound for any single COSMA allocation (Phase 1). | Denies activation/weight conversion if size exceeds budget. |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Tile size (tokens) for relative L2 correctness check. | Rank 0 warns if `rel_l2 > 1e-3`. |
| `LLAMINAR_COSMA_LOG_LEVEL` | Prefill log verbosity (`trace,debug,info,warn,error`). | Independent of global logger. |
| `LLAMINAR_COSMA_DUMP_STATS` | Emit aggregate counters at shutdown. | Use `1` or `true`. |
| `LLAMINAR_COSMA_DUMP_STATS_PATH` | Override destination for JSON stats when dump enabled. | Defaults to `cosma_prefill_stats.json`. |
| `LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT` | Disable fused quantized weight dequant + COSMA population (revert to two-step path). | Safety fallback for new fused path. |

### Instrumentation Counters
Exposed via `CosmaPrefillManager::stats()`:
- `single_rank_calls`, `fast_path_calls`, `cosma_path_calls`
- `bytes_streamed_weights`, `bytes_converted_activations`
- `matmul_invocations`, `validation_tile_checks`
- Accumulated microseconds: `us_stream_weights`, `us_convert_activation`, `us_matmul`

Strategy cache performance: `strategy_hits`, `strategy_misses` from `StrategyCache`.

### Memory Budget (Phase 1b Behavior)
Allocations are now tracked cumulatively. A request is denied when either the single allocation or the projected resident total would exceed `LLAMINAR_COSMA_MAX_RESIDENT_MB`. Releasing COSMA matrices updates the running total, ensuring large prefill sequences respect the soft budget.

### Validation Tile
When enabled, rank 0 recomputes a top-left `(T x T)` GEMM using the original row-major operands with OpenBLAS and compares against the distributed result (after gathering), logging a warning if relative L2 > 1e-3 or NaN. This provides an inexpensive early corruption detector without full matrix duplication.

### Known Limitations (Phase 1)
- No fused dequant + layout yet (weights assumed float32 during streaming).
- No transpose support for weight matrices (requests ignored with warning).
- No cumulative resident memory tracking (single-allocation guard only).
- Elementwise ops (e.g., softmax, RMSNorm) may trigger layout conversions.

### Roadmap (Planned Improvements)
- Fused quantized weight dequant + COSMA block population.
- Block-wise elementwise ops directly in distributed layout.
- Overlapping next weight stream with current GEMM (double buffering).
- Cumulative memory accounting & adaptive tiling when over budget.
- FlashAttention-style attention kernel integration.

### Debug Tips
- Force COSMA path for shorter sequences by lowering `LLAMINAR_COSMA_PREFILL_THRESHOLD`.
- Disable COSMA quickly with `export ADAPTIVE_DISABLE_COSMA=1` for parity/perf baselines.
- Use `LLAMINAR_COSMA_VALIDATE_TILE=64` (or similar) only for debugging—adds conversion overhead.
- Capture shutdown stats: `export LLAMINAR_COSMA_DUMP_STATS=1`.

### Reference
Design details and acceptance criteria are tracked in `.github/instructions/cosma-prefill-plan.instructions.md` (Phase 1).

## Llaminar

High-performance, distributed LLM inference engine.

## Key Documentation

- [Canonical Launch & Runtime Summary](docs/canonical-launch-summary.md)
- [MPI Barrier Guidelines](docs/mpi_barrier_guidelines.md)
- [Partial Attention Output Details](docs/attention-partial-output.md)
- [Tensor Parallel Architecture](docs/tensor_parallel_architecture.md)  <!-- Newly added -->

## Build

See `./run-llaminar.sh -h` for runtime options or consult the development guidelines in `.github/copilot-instructions.md`.
