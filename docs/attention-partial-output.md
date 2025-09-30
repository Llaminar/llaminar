# Attention Kernel Partial Output Contract

The MPI attention kernel (`MPIAttentionKernel`) now unconditionally produces a *partial* output per
rank: only the local head subset's contribution after the output projection (`Wo`). No implicit
MPI gather or reduction occurs inside the kernel.

## Reconstruction
For the current distribution strategy each rank holds a disjoint set of `Wo` rows (row-sharded by
head). This makes the per-rank projected results **additive**. To reconstruct the full hidden state:

```c++
// out_partial: [seq_len, d_model] local additive slice
std::vector<float> full(out_partial->size());
MPI_Allreduce(out_partial->data(), full.data(), (int)out_partial->size(),
              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
```

If/when a head-concatenation pathway is introduced the reconstruction would instead require an
`Allgather` followed by head reordering. That mode will be explicitly documented if added.

## Removed Environment Flags
The legacy environment toggles:
- `LLAMINAR_DISABLE_ATTENTION_GATHER`
- `LLAMINAR_LEGACY_ATTENTION_GATHER`

have been removed. Any script or doc referencing them should be updated—behavior is now fixed.

## Misuse Guard (Optional)
Set `LLAMINAR_ASSERT_REPLICATED_MISUSE=1` to embed a microscopic per-rank canary in the final element
of the partial output. A debug checker (future helper) can compare terminal elements across ranks
before reduction to detect accidental use-as-replicated.

## Why This Change?
- Eliminates hidden synchronization inside a compute kernel.
- Prevents double-reduction bugs when higher-level code also performs an Allreduce.
- Clarifies ownership semantics for downstream sharded kernels (RMSNorm / MLP) and end-to-end tests.
- Simplifies future introduction of alternative distribution strategies.

## Migration Notes
1. Remove any `setenv("LLAMINAR_DISABLE_ATTENTION_GATHER", ...)` calls from tests/harnesses.
2. Ensure an explicit reduction (currently `MPI_Allreduce`) occurs before using attention output as
   a full hidden activation.
3. Optionally enable the misuse guard during refactors to catch premature consumption.

## Future Enhancements
- Dedicated debug utility to validate canary pattern and assert correct reconstruction order.
- Head-concat mode support with explicit gather helper.
- Mixed precision partial accumulation with loss-scaling diagnostics.

---
Maintainer: Update this document if reconstruction semantics or distribution strategies evolve.
