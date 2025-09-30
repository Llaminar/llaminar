# Tensor Sharding & Distributed Execution Refactor Plan

> Status: Draft (Ambitious, green‑field refactor – no transitional compatibility required per request)
> Scope: End-to-end redesign of model execution to eliminate incorrect `MPI_Allreduce` output aggregation and introduce principled tensor sharding primitives across attention, MLP, and normalization stages.

---
## 1. Motivation
Current `MPIAttentionKernel` performs an `MPI_Allreduce (SUM)` over the full hidden dimension to fabricate a dense output tensor on every rank. This is:
- **Semantically incorrect** for head-parallel attention (should concatenate head slices, not sum).
- **Performance-hostile**: O(world_size * bytes) broadcast-like memory traffic.
- **Scalability blocker**: Prevents horizontal scaling of hidden dimension; memory duplicated N×.

We will replace ad‑hoc distribution with a **first-class Distributed Tensor System (DTS)** supporting:
- Explicit partition metadata.
- Shard-preserving kernel APIs.
- Lean collectives (Allgather / ReduceScatter) only when mathematically required.
- Composable fusion opportunities (e.g. Attention + Output Projection + ReduceScatter).

---
## 2. High-Level Goals
| Goal | Description | Success Metric |
|------|-------------|----------------|
| Correctness | Eliminate semantic misuse of Allreduce; preserve numerics vs single-rank baseline (within FP tolerance). | Max abs diff < 1e-5; rel L2 < 1e-6 on parity tests. |
| Memory Scaling | Avoid full replica of hidden activations on each rank. | Peak activation memory per rank ≈ (global_hidden / world_size). |
| Communication Efficiency | Replace full-tensor Allreduce with cheaper collectives. | Comm volume <= (bytes / world_size) for forward pass of attention+MLP (after fusion). |
| Extensibility | Unified abstraction for future sequence or pipeline parallel expansions. | New kernel integration requires < 200 LOC + no bespoke MPI scatter/gather code. |
| Instrumentation | Introspect shard ownership easily. | `--print-topology` shows per-tensor sharding summary. |

---
## 3. Partitioning Strategy Overview
We adopt **model-parallel hidden dimension sharding** (a.k.a. tensor parallelism) initially; sequence & pipeline parallelism are out-of-scope for this phase.

Primary partition kinds:
1. HEAD_SHARD (attention Q/K/V heads & attention output pre-projection)
2. HIDDEN_OUT_SHARD (post-output-projection / feed-forward hidden shards)
3. ROW_SHARD (matmul row partition – optional for later throughput tuning)

### Canonical Mapping
| Tensor | Partition Axis | Rationale |
|--------|----------------|-----------|
| Q, K, V (projected) | Head axis (num_heads) | Independence per head, no cross-head mixes pre-score. |
| Attention scores | Implicit (local heads only) | No need to materialize global scores. |
| Context (softmax * V) | Head axis | Still per-head, aligns with Q/K partition. |
| Output projection weights W_O | Split by input head blocks (columns) | Facilitates fused ReduceScatter across output dim. |
| Post-projection activation | Hidden dim (ReduceScatter result) | Feeds sharded RMSNorm & MLP. |
| MLP FFN weights (W1, W3) | Column shard (input dim) | Matches input shard; matmul locally valid. |
| MLP output weight (W2) | Row shard (output dim) | Enables local partial outputs + Allreduce OR ReduceScatter to logits. |

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
`TensorFactory::create_sharded(axis=Axis::Hidden, global_dim, element_type)` → allocates only local portion sized `global_dim / world (+remainder)`.

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
    DistributedTensorView context_shard; // [seq, hidden_shard] after output projection + (optional) reduce-scatter
};
```
All kernels refuse to silently densify; any attempt to pass a sharded tensor where a replicated tensor is required triggers a validation error with remediation hint.

---
## 6. Attention Execution Path
### 6.1 Chosen Baseline Path (Fused Local + ReduceScatter)
1. Local QKV projection (input shard × shard-local weight slice): produces Q_local, K_local, V_local (local heads subset).
2. Local per-head attention → context_local (still head-sharded).
3. Output projection W_O:
   - Layout: columns partitioned by heads*head_dim groups (so each rank owns columns that correspond to its heads). This makes matmul local; result is still **partial contribution** to hidden dimension.
4. If W_O is column-sharded, the result is already *the full hidden dimension subset for this rank* (no need to sum). No inter-rank dependency: **Skip collective**.
5. Optionally perform **RMSNorm pre-MLP locally** (needs global variance? For RMSNorm over hidden: compute local sum of squares, Allreduce a single scalar per sequence position, finalize scaling).

### 6.2 Alternative (Megatron-like) Path (Optional Future)
- Use row-sharded W_O, producing partial sums; finish with `ReduceScatter` to obtain final hidden shards directly (amortizing what would be Allgather → slice).

### 6.3 Removed Operations
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
## 8. Collective Patterns Summary
| Operation | Collective | Frequency | Volume | Notes |
|-----------|-----------|-----------|--------|-------|
| RMSNorm stats | Allreduce (SUM) | Per norm | O(seq) | Cheap scalar path. |
| Attention debug materialize | Allgather (Heads) | Debug only | O(hidden) | Disabled by default. |
| Logits (final layer) | Allgather or ReduceScatter | 1 / token | Depends on softmax strategy | Might delay until generation step. |
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
## 10. API & Code Changes Checklist
| Area | Action |
|------|--------|
| TensorBase | Add optional `ShardSpec*` or embed struct; add `describe()`. |
| TensorFactory | New `create_sharded(axis, global_dim, full_shape)` method. |
| MPIAttentionKernel | Rewrite `execute()` to output shard only; remove `gatherOutput()`. |
| distributeInputs() | Becomes weight slicing utility returning shard-aligned local views. |
| Output projection | Replace identity assumption; provide sliced W_O. |
| RMSNorm kernel | Accept `ShardSpec`; integrate scalar Allreduce for variance. |
| MLP kernels | Accept/produce sharded activations; adapt matmul wrappers. |
| Test suite | New parity tests: single-rank vs multi-rank concatenated reconstruction. |
| Instrumentation | `--print-topology` lists each active tensor shard summary. |
| Env flags | Debug materialization & validation toggles (see §14). |

---
## 11. Testing Strategy
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
Comm cost baseline (bad): Allreduce ~ 2 * (p-1)/p * S*H bytes.
New cost: zero (main path) + optional RMSNorm scalar Allreduce: O(S) bytes.
Benefit: Eliminates dominating term for large H, improves scaling.

---
## 13. Potential Optimizations (Deferred)
| Optimization | Description | Trigger |
|--------------|-------------|---------|
| Fused QKV projection | Single matmul for packed weights | Large S, prefill. |
| FlashAttention integration | Replace naive softmax loops | S * head_dim large. |
| ReduceScatter W_O path | Row-sharded W_O with fused reduction | p ≥ 4, large H. |
| Sharded KV cache | Sequence parallel extension | Long context. |

---
## 14. Environment Flags (Debug / Validation)
| Variable | Effect | Default |
|----------|--------|---------|
| `LLAMINAR_DEBUG_MATERIALIZE_ATTENTION` | Forces Allgather of attention output (post W_O) for validation. | Off |
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
## 16. Implementation Phases
Even though no transitional deployment required, we execute in controlled phases for safety.

1. **Scaffolding**: Add `ShardSpec`, factory methods, logging; no kernel changes.
2. **Attention Shard Refactor**: Rewrite attention to output shard only; remove Allreduce; add debug materializer.
3. **RMSNorm & MLP Shard Enablement**: Update kernels to accept sharded inputs; minimal collectives for norms.
4. **Parity & Invariants**: Add tests; enforce asserts; run deterministic suite.
5. **Cleanup & Removal**: Delete legacy gather code paths; remove obsolete env vars. 
6. **Optimization Hooks**: Introduce optional fused QKV; prepare stubs for ReduceScatter path.

---
## 17. Success Criteria & Sign-off Checklist
- [ ] All existing attention-related tests updated to shard-aware versions.
- [ ] New shard parity test passes for {H=64, 128, 192} with p=2,3,4.
- [ ] Micro attention test max_abs < 1e-5 multi-rank vs single-rank.
- [ ] Memory footprint (RSS) reduced ~1/p for large H (instrumented sample case).
- [ ] No uses of `MPI_Allreduce` over full `[seq, hidden]` remain (grep gate).
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
## 23. Immediate Next Actions
1. Implement `ShardSpec` & factory.
2. Refactor attention to remove `gatherOutput` and emit shard only.
3. Add parity reconstruction test harness.
4. Update micro test to use concatenated shards instead of reading rank 0 slice.
5. Remove Allreduce path & related environment hooks.

---
## 24. Appendix: Validation Commands (Planned)
```bash
# Run parity tests (multi-rank)
mpirun -np 4 ./build/test_attention_shard_parity -v

# Print topology after refactor
./run-llaminar.sh -v --print-topology | grep SHARD

# Sanity check: ensure no large Allreduce calls remain
grep -R "MPI_Allreduce" -n src | grep -v stats || true
```

---
## 25. Conclusion
This plan replaces an expedient but invalid dense replication model with principled tensor sharding, unlocking scalable multi-rank inference, reducing communication, and clarifying semantics for future optimizations (ReduceScatter fusion, FlashAttention, vocab sharding). The abstractions are intentionally minimal to keep refactor velocity high while establishing a durable foundation.
