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
