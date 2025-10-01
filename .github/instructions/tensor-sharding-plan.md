# Hybrid Head + Tensor Parallel Sharding Plan

> Status: Rev 2.1 (adds backend abstraction alignment + replicated vs sharded mode integration)
> Scope: Introduce a two-level parallel layout: inter-socket (MPI) head sharding + intra-socket tensor parallel (TP) for dense projections (CPU today, GPU-ready). Replace legacy implicit gathers with explicit shard metadata & selectable output modes.

---
### Quick Reference (Cheat Sheet)

| Domain | Decision | Primary Driver | Env Override(s) | Outcomes |
|--------|----------|----------------|-----------------|----------|
| Distribution Mode | Replicated vs Sharded | Param count & mem fraction heuristic | `LLAMINAR_DISTRIBUTION_MODE`, `LLAMINAR_FORCE_{REPLICATED,SHARDED}`, `LLAMINAR_SHARDING_PARAM_THRESHOLD`, `LLAMINAR_MODEL_MEM_FRACTION_MAX` | Replica of all weights OR partitioned weights |
| Attention Output Assembly | local / gather_pre / gather_post / replicated(alias) | Sequence length vs threshold | `LLAMINAR_ATTN_OUTPUT_MODE`, `LLAMINAR_ATTN_GATHER_THRESHOLD` | Local heads, Pre-gather single WO, Post-projection reduction |
| Prefill vs Inference Backend | PrefillBackend vs InferenceBackend | `seq_len >= cosma.prefill_threshold` | `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Throughput path vs latency path (matmuls) |
| Tensor Parallel (intra-socket) | Enable TP executors | Implementation phase & disable flag | `LLAMINAR_ATTN_TP_DISABLE` | Identity (today) or row/col split (future) |
| COSMA Usage | Distributed vs local BLAS | Matmul size & prefill flag | `ADAPTIVE_DISABLE_COSMA` | COSMA path or OpenBLAS variants |

Key Environment Variables:
```
LLAMINAR_DISTRIBUTION_MODE=replicated|sharded
LLAMINAR_ATTN_OUTPUT_MODE=local|gather_pre|gather_post|replicated
LLAMINAR_ATTN_GATHER_THRESHOLD=<int seq_len>
LLAMINAR_COSMA_PREFILL_THRESHOLD=<int seq_len>
LLAMINAR_ATTN_TP_DISABLE=1   # Force-disable intra-socket TP features
ADAPTIVE_DISABLE_COSMA=1     # Force OpenBLAS path
```

Backend Decision Flow (simplified):
```
if seq_len >= cosma.prefill_threshold:
    backend = PrefillBackend (favor large / distributed GEMM)
else:
    backend = InferenceBackend (favor low-latency local GEMM)
if backend.launch(...) != Success:
    fallback -> adaptive_matmul (OpenBLAS/COSMA arbitration)
```

Attention Output Mode Selection (high-level):
```
if explicit env mode: use it
else if seq_len >= ATTN_GATHER_THRESHOLD: gather_pre
else: local (decode default)
```

One-Time Rank0 Logs:
```
MODEL_DIST mode=... params=... reason=...
BACKEND_DECISION_SUMMARY component=Attention ...
BACKEND_DECISION_SUMMARY component=MLP ...
```

Parity / Validation Strategy:
1. Run replicated single-rank (baseline).
2. Enable sharded mode; reconstruct tensors via gather in tests.
3. Compare max_abs / rel_L2 within tolerance.

Fast Sanity Checklist Before Benchmarking:
```
grep -R "MPI_Allreduce" src | grep -v stats   # no accidental full activation reductions
./run-llaminar.sh -v --print-topology         # verify shard summaries (future)
```

Refer also to: `llaminar-architecture.instructions.md` (Backend Abstraction Layer section) for deeper rationale.

---
## 0. Deployment Modes (New High-Level Policy)

We standardize Llaminar execution into TWO mutually exclusive model distribution modes, selected primarily by parameter count (and optionally by runtime memory pressure heuristics):

Mode | Target Model Size | Weight Strategy | Parallelism Kind | Collectives in Forward | Primary Goals
-----|-------------------|-----------------|------------------|------------------------|---------------
`ReplicatedDataParallel` | < ~32B params (fits in per-device RAM with comfortable headroom) | Full weight replica per rank | Batch / request (data) parallel; optional head slicing for latency | Minimal (possibly final logits / diagnostics only) | Lowest complexity & latency
`ShardedTensorParallel` | ≥ ~32B params OR memory footprint > X% of device RAM | Partition large weights across ranks (column, row, or hybrid) | Intra-layer weight sharding + (optional) head sharding | Layer-local: AllGather / ReduceScatter / AllReduce as defined | Memory scaling & large prefill throughput

### 0.1 Selection Heuristic (Initial Draft)
At process start (after model metadata parsed):

```
let P_total = total parameter count
let bytes_per_param = (quant_bits / 8) or mixed precision estimate
let est_model_bytes = P_total * bytes_per_param (including optimizer = N/A in inference)
let dev_mem = detected usable memory per rank (or user override)

if (env override forces mode) use that.
else if (P_total < 32B AND est_model_bytes < 0.55 * dev_mem) -> ReplicatedDataParallel
else -> ShardedTensorParallel
```

Rationale:
* 55% cap leaves headroom for KV cache, activation spikes, allocator fragmentation.
* 32B is a pragmatic inflection based on typical 2× HBM GPU or multi-socket DRAM budgets; adjust as empirical data arrives.

### 0.2 Environment Overrides (Proposed)
Add to `debug_env` snapshot (names tentative):
| Variable | Values | Effect |
|----------|--------|--------|
| `LLAMINAR_DISTRIBUTION_MODE` | `replicated` / `sharded` / unset | Force specific mode (bypass heuristic) |
| `LLAMINAR_FORCE_REPLICATED` | any | Shortcut alias forcing replicated (deprecated when mode matures) |
| `LLAMINAR_FORCE_SHARDED` | any | Shortcut alias forcing sharded |
| `LLAMINAR_SHARDING_PARAM_THRESHOLD` | integer (billions) | Override 32B switch point |
| `LLAMINAR_MODEL_MEM_FRACTION_MAX` | float (0.0–0.95) | Replace 0.55 heuristic |

Conflict resolution precedence (highest first):
1. `LLAMINAR_DISTRIBUTION_MODE`
2. FORCE flags (`FORCE_SHARDED` / `FORCE_REPLICATED`)
3. Threshold / fraction overrides
4. Built-in defaults

### 0.3 Execution Surface Differences
Aspect | ReplicatedDataParallel | ShardedTensorParallel
-------|------------------------|----------------------
Weight Load | Single pass, identical mapping per rank | Partition mapping (slice metadata created) |
Memory Footprint | O(model) per rank | O(model / world_size) per rank (plus shard metadata) |
Matmul Kernels | Standard GEMMs | Local shard GEMMs (reduced dimensions) |
Collectives (Attention) | Optional head gather (if configured) | Defined per projection (gather pre/post or reduce-scatter) |
RMSNorm | Local stats only | Scalar AllReduce (sumsq) |
MLP | Fully local | Up/down projections sharded (col/row) |
Failure Modes | Mostly local | Collective mismatch, shard misalignment |
Debug Parity Path | Always available (single rank run) | Requires reconstruction harness |

### 0.4 Transition Path
Phase | Action | Guardrails
------|--------|-----------
1 | Implement selector + metadata flag `distribution_mode` | Silent fallback to replicated on any shard init failure
2 | Weight file loader emits shard slices (contiguous) | Cross-check sum(local_dim) == global_dim
3 | Enable attention output projection sharding | Parity test vs replicated (max_abs, rel_L2)
4 | Enable MLP (W1/W2/W3) sharding | Add perf counters (comm vs compute ms)
5 | Introduce reduce-scatter path for WO | Environment gated
6 | Autotune gather timing (prefill vs decode) | Logging + diff guard

### 0.5 Rollback / Failsafe
If any collective times out or parity validation fails under debug flag:
* Log error with shard descriptors
* Set a runtime sticky flag `sharding_degraded=true`
* Reroute remaining layers through replicated fallback (if memory still sufficient)

### 0.6 Testing Additions (Mode-Aware)
Test | Replicated | Sharded
-----|------------|--------
`ModelLoadModeSelectTest` | ✓ | ✓ (assert mode selection) |
`AttentionWOShardParityTest` | Baseline | Reconstruct vs baseline |
`MLPShardParityTest` | Baseline | Reconstruct vs baseline |
`NormScalarAllReduceTest` | Local sums == global | Validate AllReduce correctness |

### 0.7 Telemetry / Logging
On startup print (rank 0):
```
MODEL_DIST mode=sharded params=47.1B quant=4.0b est_model=~23.5GB dev_mem=79GB mem_frac=0.30 reason="exceeds param threshold"
```
or
```
MODEL_DIST mode=replicated params=7.0B quant=4.0b est_model=~3.5GB dev_mem=63GB mem_frac=0.06 reason="below thresholds"
```

### 0.8 Open Questions
| Question | Current Stance |
|----------|----------------|
| Should we prefer memory fraction or param count first? | Param count gives coarse early exit; fraction refines borderline cases. |
| Multiple quant formats (per-layer variance) impact estimate? | Use worst-case (largest bytes/param) until per-layer accounting added. |
| Dynamic downgrade if fragmentation grows? | Future: periodic allocator watermark check (deferred). |

---

## 0.a Backend Abstraction Alignment (New)

Recent refactors introduced lightweight prefill vs inference backend interfaces (`prefill_backend.*`, `inference_backend.*`) wrapping all projection GEMMs (Q/K/V, WO, Gate/Up/Down). This impacts the sharding plan as follows:

| Concern | Previous State | Current State |
|---------|----------------|---------------|
| Kernel call sites | Direct `adaptive_matmul` invocations | Backend `launch()` with fallback to `adaptive_matmul` |
| Prefill vs Decode heuristic | Scattered size checks | Unified: `seq_len >= debugEnv().cosma.prefill_threshold` |
| Future GPU enablement | Would require editing every kernel | Drop-in GPU backend implementation; sharding logic unchanged |
| Logging | Ad-hoc per-matmul | One-time `BACKEND_DECISION_SUMMARY` per component (rank0) |
| Failure handling | Immediate error | Backend returns `Unsupported/Error` → WARN → fallback |

### Why It Matters for Sharding
1. Shard-aware dimensions (e.g., reduced K or N) are passed uniformly through descriptors, simplifying future TP executor insertion.
2. Collective decisions (gather pre vs post) remain orthogonal; backends only see local (shard) shapes.
3. When weight sharding activates, only the dimensions in the `OpDesc` shrink—no changes to backend selection logic.

### Logging Example
```
BACKEND_DECISION_SUMMARY component=Attention seq_len=8192 threshold=4096 path=prefill prefill_backend=cpu_stub inference_backend=cpu_stub phases=QKV+Out fallback=adaptive_matmul
```

### Relationship to This Plan
The plan’s intra-layer partition steps (WO / MLP sharding, ReduceScatter, etc.) will feed the backends local shard shapes. No additional abstraction layers are expected; the existing backend interfaces become the stable contract between high-level sharding policy and execution.

### Non-Goals
* Backends do not perform collective orchestration.
* Backends do not own sharding metadata (`ShardSpec`/`TPPartitionSpec`).
* Backends do not re-implement heuristic thresholds (single source: `debugEnv()` + adaptive matmul internals).

---

---
## 1. Motivation
The previous design conflated head partitioning and hidden dimension replication, occasionally using `MPI_Allreduce` where concatenation was required. This incurred unnecessary bandwidth and obscured ownership semantics.

We now formalize a Hybrid Parallel Layout:
* Level 1 (MPI ranks / sockets): Head sharding (independent attention computations, minimal comm during decode).
* Level 2 (Intra-socket threads or GPUs): Tensor Parallel (split large dense matmuls for output projection and MLP).

Objectives:
* Eliminate semantically incorrect summations.
* Provide explicit shard contracts (no silent densification).
* Allow adaptive choice of gather timing (pre vs post output projection) based on sequence length & cost model.
* Keep future sequence or vocab sharding pluggable.

---
## 2. High-Level Goals
| Goal | Description | Success Metric |
|------|-------------|----------------|
| Correctness | Eliminate semantic misuse of Allreduce; preserve numerics vs single-rank baseline (within FP tolerance). | Max abs diff < 1e-5; rel L2 < 1e-6 on parity tests. |
| Memory Scaling | Avoid full replica of hidden activations on each rank. | Peak activation memory per rank ≈ (global_hidden / world_size). |
| Communication Efficiency | Hierarchical collectives (intra-socket TP, optional inter-socket head gather) only when required. | Inter-socket bytes reduced to O(seq * hidden) at most once per layer (or deferred). |
| Extensibility | Unified abstraction for future sequence or pipeline parallel expansions. | New kernel integration requires < 200 LOC + no bespoke MPI scatter/gather code. |
| Instrumentation | Introspect shard ownership easily. | `--print-topology` shows per-tensor sharding summary. |

---
## 3. Partitioning Strategy Overview
Two orthogonal axes in this phase:
1. Head Axis (inter-socket) – each MPI rank owns a contiguous block of heads (may be uneven by +1 head for remainders).
2. Hidden Axis (intra-socket TP) – dense matmuls split by row or column depending on kernel phase.

### 3.1 Intra-Socket Tensor Parallel Abstraction (NEW)
We introduce a lightweight, CPU-first tensor parallel specification: `TPPartitionSpec` (see `src/tensors/tp_partition.h`).

Purpose:
* Decouple logical model parallel decisions from kernel implementation details.
* Provide a uniform way to describe row (M) or column (N) partitioning of a GEMM without immediately requiring distributed collectives (initially single-process simulation; MPI not required).
* Allow progressive opt-in: current implementation supplies only a trivial (identity) splitter so existing paths are unaffected.

`TPPartitionSpec` fields:
```
struct TPPartitionSpec {
    enum class Axis { Row, Col } axis; // Partition axis relative to left operand A (Row->split M, Col->split N)
    int  tp_size;                      // Number of intra-socket partitions
    int  tp_rank;                      // Partition index [0, tp_size)
    size_t global_dim;                 // Global extent along axis
    size_t local_offset;               // Offset slice start
    size_t local_dim;                  // Local slice length
};
```

Helper: `compute_tp_partition(global_dim, tp_size, tp_rank, axis)` implements ceil-balanced block distribution (first `global_dim % tp_size` partitions get +1).

Trivial splitter (`TrivialMatmulSplitter`) contract:
```
bool run(A, B, C, M, N, K) // Directly calls provided baseline matmul functor (no slicing yet)
```

Planned evolution:
| Stage | Enhancement | Notes |
|-------|-------------|-------|
| 1 (DONE) | Spec + trivial splitter | Identity path (no perf impact). |
| 2 | Row slice executor | Partition M: each partition computes its row block; concat after. |
| 3 | Column slice executor | Partition N: each partition computes its column block. |
| 4 | Hybrid row/col (2D) | Enables block-cyclic or 2D tiling later if needed. |
| 5 | Fused WO partition | Integrate with attention output projection selection. |
| 6 | MLP integration | Up/Down projection complementary split (e.g., W1/W3 col, W2 row). |

Test strategy additions (see §11): add simulated multi-partition parity tests that reconstruct full C from independently computed slices (currently CPU loops). These prepare for replacing the reconstruction step with in-place writes once real TP executors arrive.

Modes (output assembly choices) (Rev 2.1 semantics):
* `LocalHeads` – Return only local head slice (row-partitioned WO contribution); no collective.
* `GatherHeadsPostProjection` – Local output projection with row-partitioned W_O then `MPI_Allreduce` (SUM) to combine additive hidden contributions (earlier draft said Allgather; implementation uses Allreduce).
* `GatherHeadsPreProjection` – `MPI_Allgatherv` of head contexts (heads-major) followed by single global output projection (avoids redundant per-rank WO matmuls).
* `Replicated` – Planned alias to `gather_pre` guaranteeing fully replicated hidden (currently falls back to LocalHeads).

### Canonical Mapping (Hybrid)
| Tensor / Stage | Inter-Socket (MPI) | Intra-Socket TP | Notes |
|----------------|-------------------|-----------------|-------|
| Q,K,V proj input | Hidden replicated per head shard OR hidden shard if later phase | Column-split (optional) | Phase 1 keeps replicated hidden inside socket. |
| Q,K,V outputs | Head shard | Local (no TP) | Each rank only its heads. |
| Attention scores & softmax | Head shard | Local | No distributed ops. |
| Context (per-head) | Head shard | Local | |
| Output projection (WO) | Input heads local | Column or row TP | Mode decides gather timing. |
| Post-attn hidden | Hidden shard (if TP active) OR replicated per socket | Local TP layout | Sharded only after enabling TP. |
| RMSNorm | Hidden shard (needs scalar Allreduce) | Local partial + scalar Allreduce | Comm is O(seq). |
| MLP up / gate (W1,W3) | Hidden shard | Row/Col TP (policy) | |
| MLP down (W2) | Hidden shard | Complementary split | Possibly ReduceScatter for fusion (later). |

---
## 4. Distributed Tensor Metadata
Define a lightweight struct and attach to `TensorBase` (via optional pointer or subclass):
```cpp
struct ShardSpec {
    enum class Type { Replicated, Sharded };
    enum class Axis { None, Hidden, Heads, Seq /* future */ };
    Type type;          // Replicated or Sharded
    Axis axis;          // Logical model axis
    int world;          // MPI world size
    int rank;           // MPI rank
    size_t global_dim;  // Size along sharded axis (0 if replicated)
    size_t local_offset;// Offset of local slice
    size_t local_dim;   // Local slice length
};
```
Add helper facade:
```cpp
class DistributedTensorView {
public:
    TensorBase* backing;      // Underlying storage for local shard or full
    ShardSpec spec;           // Ownership metadata
    // Accessors
    bool is_sharded() const;
    bool matches(const DistributedTensorView&, ShardSpec::Axis) const; // layout compatibility
};
```

### Construction Helpers
Extend factory:
* `create_head_shard(total_heads, head_dim)`
* `create_hidden_shard(global_hidden, strategy={even})`
* `create_tp_partition(kind=Row|Col, global_shape, tp_size)` (Phase 2).

### Logging / Debug
`tensor->describe()` ⇒ `ATTN_CTX shard axis=Heads rank=1/4 offset=16 size=16 global=64`.

---
## 5. Kernel API Revisions
Each MPI-aware kernel shifts from (implicit gather) to explicit shard contracts:
```cpp
struct AttentionInputs {
    DistributedTensorView x;      // [seq, hidden_shard]
    DistributedTensorView w_qkv;  // Packed or separate shard(s)
    DistributedTensorView w_o;    // Sharded columns (input) or rows (output) depending on fusion path
};
struct AttentionOutputs {
    DistributedTensorView tensor;  // Local heads or concatenated hidden depending on OutputMode
    AttentionAssemblyState state;  // LocalPartial / Concatenated / Replicated
    bool requires_concat_for_dense; // Filled by kernel based on chosen mode
};
```
All kernels refuse to silently densify; any attempt to pass a sharded tensor where a replicated tensor is required triggers a validation error with remediation hint.

---
## 6. Attention Execution Path (Output Modes)
### 6.1 LocalHeads (default for decode)
Compute attention per head shard, project locally, return partial. No inter-socket comm.

### 6.2 GatherHeadsPostProjection (implemented: Allreduce)
Local projection using row-partitioned W_O yields additive hidden contributions; reconstruction performed via `MPI_Allreduce` (SUM). If future W_O column-partition is adopted this may switch to Allgather.

### 6.3 GatherHeadsPreProjection (implemented)
`MPI_Allgatherv` of per-rank head contexts into full heads-major buffer, reorder to standard layout, then single global projection. Sets metadata: concatenated=true, replicated=true.

### 6.4 Replicated (Alias IMPLEMENTED Rev 2.2)
Now implemented as a semantic alias of `GatherHeadsPreProjection`:
* Selecting `LLAMINAR_ATTN_OUTPUT_MODE=replicated` executes the pre-projection gather path (Allgatherv head contexts, single global W_O matmul) but preserves `mode=Replicated` in metadata for downstream logic / diagnostics.
* Metadata: `concatenated=true`, `replicated=true` identical to gather_pre.
* Rationale: Avoid duplicate code while providing a stable legacy/debug mode token.
* Future: If a distinct fully replicated fast path diverges, the alias can split without changing external semantics.

### 6.5 Future (ReduceScatter Path)
Row-sharded WO + fused ReduceScatter to produce hidden shards directly; saves gather if subsequent layers remain sharded.

### 6.6 Removed Operations
- No `MPI_Allreduce` over full activation.
- No staging of concatenated heads unless explicit debug flag: `LLAMINAR_DEBUG_MATERIALIZE_ATTENTION=1` triggers a safeguarded `Allgather`.

---
## 7. Norm & MLP Sharded Semantics
### 7.1 RMSNorm
RMSNorm over hidden dimension requires global mean of squared values:
```
local_sum_sq = sum_i shard(x_i^2)
MPI_Allreduce(sum) -> global_sum_sq
scale = 1 / sqrt(global_sum_sq / hidden_dim + eps)
Apply scale locally.
```
Communication cost: O(seq_len) scalars (very small relative to activations).

### 7.2 Feed-Forward (SwiGLU Style)
Given input shard H_local (size hidden / p):
- W1, W3 column-sharded (shape [hidden/p, ffn_expand]) local matmul valid → gating intermediate is replicated across expansion dimension implicitly formed locally.
- Activation + elementwise ops local.
- W2 row-sharded (shape [ffn_expand, hidden/p]) yields output shard directly.
- If future fusion desired: adopt group reduce patterns if partial outputs accumulate.

### 7.3 Residual Connections
Residual add requires matching shard layout; both operands must share identical `ShardSpec`. Enforce at runtime (abort with log if mismatch).

---
## 8. Collective Patterns Summary (Hybrid)
| Operation | Collective | Frequency | Volume | Notes |
|-----------|-----------|-----------|--------|-------|
| RMSNorm stats | Allreduce (SUM) | Per norm | O(seq) | Cheap scalar path. |
| Attention debug materialize | Allgather (Heads) | Debug only | O(hidden) | Disabled by default. |
| Logits (final layer) | Allgather OR vocabulary-partition softmax later | 1 / token | Possibly delayed | Future vocab sharding TBD. |
| Future gradient (not in scope) | Allreduce / RS | Backprop only | N/A | Placeholder. |

No bulk Allreduce over full hidden activations in forward critical path.

---
## 9. Data Layout Details
| Layout | Shape Example | Memory Order | Comment |
|--------|---------------|--------------|---------|
| Sharded Hidden | [seq, hidden/p] | Row-major | Contiguous per rank. |
| Head Shard Q | [seq, heads/p * head_dim] | Row-major | Heads packed contiguously. |
| Packed QKV (optional) | [seq, 3 * heads/p * head_dim] | Row-major | Allows fused projection. |

Ensure alignment to 64-byte boundaries for vectorization.

---
## 10. API & Code Changes Checklist (Revised Phasing)
| Area | Action |
|------|--------|
| TensorBase | Add optional `ShardSpec*` or embed struct; add `describe()`. |
| TensorFactory | New `create_sharded(axis, global_dim, full_shape)` method. |
| MPIAttentionKernel | Add OutputMode, emit metadata; implement post-projection gather; remove implicit dense write. |
| MPIAttentionKernel (Phase 2) | Add pre-projection gather path + heuristic. |
| MPIAttentionKernel (Phase 3) | Integrate intra-socket TP splits for WO. |
| distributeInputs() | Becomes weight slicing utility returning shard-aligned local views. |
| Output projection | Replace identity assumption; provide sliced W_O. |
| RMSNorm kernel | Accept `ShardSpec`; integrate scalar Allreduce for variance. |
| MLP kernels | Accept/produce sharded activations; adapt matmul wrappers. |
| Test suite | New parity tests: single-rank vs multi-rank concatenated reconstruction. |
| Instrumentation | `--print-topology` lists each active tensor shard summary. |
| Env flags | Debug materialization & validation toggles (see §14). |

---
## 11. Testing Strategy (Augmented)
Add tests for each output mode:
* `AttentionLocalHeadsDecodeTest` – multi-rank decode parity after on-demand gather.
* `AttentionGatherPreProjectionTest` – sequence-length threshold path.
* `AttentionTPParityTest` – ensure TP matmul outputs reconstruct single-rank baseline.
### 11.1 Unit / Micro
- `DistributedShardSpecTest`: verify offset math across uneven splits.
- `AttentionShardParityTest`: construct synthetic small model; gather shards and compare to single-process baseline (QKV + attention).
- `RMSNormShardStatsTest`: feed known vector, verify global variance equals analytic.

### 11.2 Integration
- End-to-end prompt prefill identical logits vs single-rank (within tolerance) using deterministic seed.
- Multi-rank generation test (short sequence) verifying token-by-token parity.

### 11.3 Property Tests
- Random hidden sizes not divisible by world_size: ensure last rank’s `local_dim = ceil` logic correct; reconstruct matches baseline.

### 11.4 Performance Smoke
- Compare wall time before/after on sequence length {128, 1k, 8k}; expect memory drop and no regression for small sizes (<5%).

---
## 12. Performance Model (Forward Pass Attention)
Let:
- H = hidden size
- p = world_size
- S = sequence length
- B = batch (token) count (assume 1 for decode, large for prefill)

Local compute (QKV): O(S * H * H/p_head_factor) unaffected.
Legacy cost: Allreduce ≈ 2 * (p-1)/p * S*H bytes (per layer).
LocalHeads mode: 0 inter-socket until optional gather → at most one `Allgatherv` of S*H.
GatherHeadsPreProjection: Same volume but saves (P-1) redundant WO matmuls.
ReduceScatter future: Potentially lowers final logits or next-layer input volume by factor P when staying sharded.
Benefit: Eliminates dominating term for large H, improves scaling.

---
## 13. Potential Optimizations (Deferred / Layered)
| Optimization | Description | Trigger |
|--------------|-------------|---------|
| Fused QKV projection | Single matmul for packed weights | Large S, prefill. |
| FlashAttention integration | Replace naive softmax loops | S * head_dim large. |
| Pre vs Post gather auto-switch | Model cost heuristic & env override | Distinguish decode vs prefill. |
| ReduceScatter W_O path | Row-sharded W_O with fused reduction | p ≥ 4, large H. |
| TP overlap | Nonblocking collectives hide gather / AllReduce | seq >= 2, multi-layer. |
| Sharded KV cache | Sequence parallel extension | Long context. |

---
## 14. Environment Flags (Expanded)
| Variable | Effect | Default |
|----------|--------|---------|
| `LLAMINAR_ATTN_OUTPUT_MODE` | local | gather_post | gather_pre | replicated | Select attention assembly mode. | default=local decode, gather_pre prefill |
| `LLAMINAR_DEBUG_MATERIALIZE_ATTENTION` | Forces Allgather regardless of mode (post path). | Off |
| `LLAMINAR_ATTN_TP_DISABLE` | Force-disable intra-socket TP (debug). | Off |
| `LLAMINAR_ATTN_GATHER_THRESHOLD` | Sequence length threshold for switching to pre-gather path. | 1024 |
| `LLAMINAR_DUMP_SHARDS` | Logs first N scalars per shard (N=16). | Off |
| `LLAMINAR_SHARD_PARITY_CHECK` | After each major kernel, reconstruct global (Allgather) and diff vs rank0 baseline (small configs). | Off |
| `LLAMINAR_ASSERT_REPLICATED_MISUSE` | Abort if a kernel consumes a replicated tensor where sharded expected or vice versa. | On |

---
## 15. Failure Modes & Mitigations
| Risk | Symptom | Mitigation |
|------|---------|------------|
| Misaligned shard offsets | Parity diff grows with rank index | Add offset unit tests + runtime asserts. |
| Incorrect W_O slicing | Output drift only after projection | Independent output projection parity test. |
| RMSNorm scalar mismatch | Layernorm divergence | Log local/global variance per rank under flag. |
| Hidden not divisible by p | Tail rank size error | Implement floor/ceil split + test. |
| Accidental re-densification | Memory spike | Track global allocation bytes; warn if > expected. |

---
## 16. Implementation Phases (Hybrid Roadmap – Progress Snapshot)
1. (DONE) Scaffolding v2: OutputMode enum, AttentionResult metadata, env parsing.
2. (DONE) GatherHeadsPostProjection path (row-split + Allreduce) integrated.
3. (DONE) GatherHeadsPreProjection path (Allgatherv + single projection) + multi-rank parity test.
4. (DONE) Heuristic auto-switch (seq length threshold) LocalHeads ↔ gather_pre.
5. (DONE) Intra-socket TP abstraction (TPPartitionSpec + trivial splitter + unit tests).
6. (PLANNED) Integrate TP executors into WO + MLP; TP parity tests (row & col).
7. (PLANNED) Overlap (nonblocking collectives / compute) prototypes.
8. (PLANNED) ReduceScatter WO experimental path.
9. (DONE) Replicated mode alias (implemented Rev 2.2) + pending minor legacy flag cleanup.
10. (DONE) Prefill/Inference backend abstraction wiring (attention + MLP) with fallback & rank0 summary log.

---
## 17. Success Criteria & Sign-off Checklist (Updated)
- [ ] All existing attention-related tests updated to shard-aware versions.
- [ ] New shard parity test passes for {H=64, 128, 192} with p=2,3,4.
- [ ] Micro attention test max_abs < 1e-5 multi-rank vs single-rank.
- [ ] Memory footprint (RSS) reduced ~1/p for large H (instrumented sample case).
- [ ] No uses of `MPI_Allreduce` over full `[seq, hidden]` remain (except RMSNorm scalars).
- [ ] OutputMode switching validated: local ↔ gather_post ↔ gather_pre produce identical concatenated tensor.
- [ ] TP matmul parity: max_abs < 1e-5 vs single-rank across {row, col} split.
- [ ] Topology printout lists each tensor with correct offset/size.

---
## 18. Out-of-Scope / Explicit Deferrals
| Item | Reason |
|------|--------|
| Gradient / Backprop sharding | Inference-only refactor phase. |
| Sequence parallel (token dimension) | Additional complexity; revisit after stable hidden sharding. |
| Pipeline parallel | Requires stage partitioning design not yet codified. |
| Mixed precision collectives | Introduce after correctness lock-in. |

---
## 19. Monitoring & Instrumentation Additions
- `PerfCounters::record_shard_bytes(comm_bytes, compute_flops, seq_len)` to correlate diminishing comm cost.
- Diff logger: on parity failure, auto-dump per-rank shard slices + reconstruction snippet.

---
## 20. Example Flow (Decode Step, p=4)
1. Input hidden shard H_i (i∈[0..3]).
2. Local QKV → per-head subset.
3. Local attention → context_i.
4. Local output projection (columns for heads_i) → hidden shard H'_i.
5. RMSNorm: each rank computes local sumsq → Allreduce scalars → finalize.
6. MLP forward (fully local under column/row sharding layout) → H''_i.
7. Repeat layer stack.
8. Final logits (optional): Allgather hidden shards → softmax OR keep sharded softmax if vocab also partitioned later.

---
## 21. Code Pseudocode Snippet (Attention Execute)
```cpp
bool ShardedAttention::execute(const DistTensor& x, DistTensor& out) {
    assert(x.spec.axis == ShardSpec::Axis::Hidden);
    // 1. QKV projection (packed): [seq, hidden/p] x [hidden/p, 3*heads_per_rank*hd] -> local packed
    project_qkv(x, q_local, k_local, v_local); // purely local matmul
    // 2. Apply RoPE (local heads)
    rope_apply(q_local, k_local, n_past);
    // 3. Local attention (primitive)
    fused_attention(q_local, k_local, v_local, ctx_local);
    // 4. Output projection (local columns)
    matmul(ctx_local, w_o_local, out_local); // out_local == out.backing
    // 5. (Optional debug) materialize global
    if (debug_materialize) allgather_concat(out_local, temp_full);
    return true;
}
```

---
## 22. Open Questions
| Question | Tentative Direction |
|----------|---------------------|
| Should W_O be column or row sharded first? | Column (simpler; no ReduceScatter needed). |
| How to handle uneven head counts vs ranks? | Assign floor(heads/p) + distribute remainder first ranks; store head_offset. |
| Integrate with COSMA heuristics? | Provide local shard dims; existing adaptive matmul path should accept reduced K dimension. |

---
## 23. Immediate Next Actions (Execution Queue – Updated)
1. (DONE) Replicated mode alias to gather_pre (metadata distinction) + env selection.
2. Integrate TP executors into WO path (column split baseline) behind `LLAMINAR_ATTN_TP_DISABLE`.
3. (DONE) RMSNorm shard stats test (scalar Allreduce parity) preparing hidden sharding (RMSNormShardStats_Parity passing).
4. Performance instrumentation: gather_pre vs gather_post timing across S={128,2k,8k}; log GFLOPS & comm ms.
5. TP row & col executor implementations + parity tests M,N not divisible by tp_size.
6. ReduceScatter WO experimental path.
7. Overlap prototype (nonblocking gather + compute). 
8. Benchmark harness & reporting integration.

---
## 24. Appendix: Validation Commands (Planned)
```bash
# Run parity tests (multi-rank)
mpirun -np 4 --oversubscribe ./build/test_attention_shard_parity -v

# Print topology after refactor
./run-llaminar.sh -v --print-topology | grep SHARD

# Sanity check: ensure no large Allreduce calls remain
grep -R "MPI_Allreduce" -n src | grep -v stats || true
```

---
## 25. Conclusion
We evolve from a single-axis hidden sharding concept to a hierarchical strategy optimized for modern NUMA and multi-GPU topologies. This layered approach allows us to defer inter-socket communication, exploit intra-socket bandwidth with TP, and cleanly extend toward ReduceScatter and vocab sharding. The staged roadmap ensures incremental correctness while laying the groundwork for aggressive performance optimizations.


---

# Tensor Parallel Architecture in Llaminar

Author: David Sanftenberg

## Purpose
Clarify what "tensor parallel" (TP) means in Llaminar so future design and code paths reflect the *intended* scope: **inter-socket / inter-device tensor partitioning**, not intra-socket micro-partitioning.

## Summary Definition
Tensor Parallel (TP) in Llaminar: A strategy that splits large model weight tensors (e.g. projection matrices) *across distinct compute domains* (CPU sockets or GPUs) so that each domain owns a contiguous shard and participates in collective reconstruction (or avoids it through algorithmic reformulation). TP is **not** about slicing within a single NUMA domain purely to feed more threads; that case is a local threading / BLAS scheduling concern, not architectural TP.

## Why This Clarification Was Needed
Recent experimental code introduced per-socket column partition logic and splitter abstractions inside a single process with one MPI rank per socket. This blurred boundaries between:
- Local kernel micro-optimizations (packing, loop tiling, OpenMP)
- True TP (multi-rank coordinated layout & communication semantics)

Removing the in-process splitter reflects the decision: until we introduce multi-rank head or MLP weight distribution, we stay with a single GEMM per rank.

## Scope of Tensor Parallel in Llaminar
| Aspect | In-Scope (TP) | Out-of-Scope (Not TP) |
|--------|---------------|-----------------------|
| Device boundary | Splitting across MPI ranks mapped to sockets / GPUs | Slicing purely for OpenMP thread balance |
| Communication | Requires gather / reduce / all-reduce / scatter primitives | No inter-rank collectives required |
| Partition object | `ShardSpec`, future TP layout metadata | Temporary per-loop spec inside one rank |
| Performance goal | Reduce per-rank memory & enable scaling larger models | Extract a few % from cache/packing reuse |
| Failure modes | Collective mismatch, shard misalignment | Minor local scheduling inefficiency |

## Architectural Pillars for TP
1. **Deterministic Partitioning**: A canonical mapping from global tensor shape to (rank -> slice range). Implemented via `ShardSpec` today; future expansion: richer TP layout registry.
2. **Collective Semantics**: For each distributed operation, define whether the result is:
   - Fully materialized on each rank (replicated)
   - Partially owned (requiring later gather)
   - Reduced (sum/mean) across ranks
3. **Latency Hiding / Overlap** (future): Stream shard communication while computing local tiles (esp. for attention prefill & MLP projections).
4. **Backend Neutrality**: TP should compose with adaptive matmul (OpenBLAS vs COSMA). Distribution policy decides shape; backend executes local shard tile.
5. **Fallback Grace**: If environment / model size does not justify TP (few ranks, small matrices), remain on single-rank-per-socket replicated path with zero overhead.

## Current State (2025-10-01)
- Execution model: One MPI rank per CPU socket.
- Weights: Replicated per rank (no memory pressure at present target model sizes in tests).
- Output projection (WO): Simplified back to single GEMM (no intra-rank sharding).
- No true cross-rank TP collectives yet for attention or MLP.

## When TP Becomes Necessary
Trigger conditions suggesting real TP implementation:
- Model dimension or parameter count exceeds per-socket memory budget (e.g. > ~40–50 GB resident for targeted hardware).
- Need to reduce per-rank activation footprint to keep cache pressure manageable.
- Prefill latency dominated by large projection GEMMs where each rank could own a fractional slice enabling parallel speedup > comm overhead.

## Planned TP Evolution Path
Stage | Milestone | Key Additions | Risks
------|-----------|---------------|------
1 | Spec Formalization | `TPLayout` struct (tensor -> {rank: slice}) | Consistency across kernels
2 | Read-Only Weight Sharding | Static partition at load (sharded mmap or scatter) | Loader complexity
3 | Attention WO Shard + AllGather | Local matmul + gather heads before next layer | Gather latency
4 | MLP FFN Split (Column/Row Hybrid) | Column split W1/W2 + optional reduce-scatter | Overlapping comm
5 | Overlap & Pipelining | Stream gather while computing next shard | Deadlocks / ordering
6 | Autotuned Policy | Dynamic decision per layer (size, rank count, seq_len) | Policy thrash

## Non-Goals (for TP layer)
- Micro kernel autotiling inside a single shard.
- OpenBLAS thread assignment heuristics (belongs to adaptive backend layer).
- Replacing COSMA topology logic (orthogonal concern: local vs distributed GEMM backend selection).

## Practical Guidance for Contributors
DO:
- Introduce new shard-aware code only behind explicit `ShardSpec` or future `TPLayout` objects.
- Keep environment toggles centralized (extend `debug_env`).
- Write tests that assert reconstruction correctness across >1 rank early.

DON'T:
- Add per-rank intra-socket slicing just for local loops and call it TP.
- Intermix packing / layout transforms with TP policy logic (separate concerns).
- Depend on ad-hoc env flags for novel TP states—register them first.

## FAQ
Q: Why not do intra-socket TP to leverage more cores?
A: We already exploit cores via OpenMP / BLAS. Slicing tensors purely to reissue more smaller GEMMs usually regresses due to packing & launch overheads versus one well-optimized large GEMM.

Q: Will TP hurt small batch / short sequence latency?
A: Proper policy gating should disable TP when comm + synchronization overhead outweighs parallel benefit (e.g., short decode steps).

Q: Can COSMA replace TP for large GEMMs?
A: COSMA distributes a single GEMM but still assumes full operands per rank or well-defined process grids; TP is a *model-graph level* partitioning of weights/activations feeding multiple GEMMs across layers.

## Action Items (Post-Clarification)
- [ ] Add a lightweight `TPLayout` placeholder (no behavior) to anchor future PRs.
- [ ] Audit existing code for any residual intra-rank TP artifacts (now removed from attention kernel projection).
- [ ] Add documentation link reference in `README.md` under Architecture.

---
This document should be updated once weight sharding or collective-backed TP enters the codebase.
