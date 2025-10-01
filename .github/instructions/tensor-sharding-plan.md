# Hybrid Head + Tensor Parallel Sharding Plan

> Status: Rev 2 (supersedes earlier draft focusing only on hidden sharding)
> Scope: Introduce a two-level parallel layout: inter-socket (MPI) head sharding + intra-socket tensor parallel (TP) for dense projections (CPU today, GPU-ready). Replace legacy implicit gathers with explicit shard metadata & selectable output modes.

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

### 6.4 Replicated
For legacy consumers or debug; forces full gather irrespective of efficiency.
1. Local QKV projection (input shard × shard-local weight slice): produces Q_local, K_local, V_local (local heads subset).
2. Local per-head attention → context_local (still head-sharded).
3. Output projection W_O:
   - Layout: columns partitioned by heads*head_dim groups (so each rank owns columns that correspond to its heads). This makes matmul local; result is still **partial contribution** to hidden dimension.
4. If W_O is column-sharded, the result is already *the full hidden dimension subset for this rank* (no need to sum). No inter-rank dependency: **Skip collective**.
5. Optionally perform **RMSNorm pre-MLP locally** (needs global variance? For RMSNorm over hidden: compute local sum of squares, Allreduce a single scalar per sequence position, finalize scaling).

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
9. (PLANNED) Replicated mode alias + cleanup of legacy debug flags.

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
1. Replicated mode alias to gather_pre (metadata distinction) + env selection.
2. Integrate TP executors into WO path (column split baseline) behind `LLAMINAR_ATTN_TP_DISABLE`.
3. RMSNorm shard stats test (scalar Allreduce parity) preparing hidden sharding.
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
