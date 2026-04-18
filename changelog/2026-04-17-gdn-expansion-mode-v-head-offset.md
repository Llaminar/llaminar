# Fix: GDN expansion-mode missing `global_v_head_offset` under TP

## Summary

Fixed a silent correctness bug in `Qwen35Graph::buildGDNSubgraph` that left
`GDNRecurrenceStage::Params::global_v_head_offset = 0` on rank > 0 in the
**expansion** GQA regime (where the GDN module has more V-heads than K-heads),
causing rank 1+ to pick the wrong K-heads for its V-head slice and produce
broken recurrence outputs for the 27B model under `NodeLocalTP` (2×MPI CPU).

## Impact

- **27B Qwen-3.5** under TP=2 on CPU: prefill parity was `cos ≈ 0.916`
  (KL 0.21) vs single-device `cos ≈ 0.999`. FFN_NORM appeared to be the
  max-drop stage every layer, but the real divergence originated inside
  `GDN_RECURRENCE` on rank 1 — subsequent stages inherited the error and
  `GATED_RMS_NORM` / `gdn_out_proj allreduce` partially masked its
  magnitude.
- Single-device and TP=1 paths were unaffected (rank 0 sets offset=0 which
  is correct).
- Decode path appeared to work because, at the time of original testing,
  the state had been seeded via prefill and the divergence was masked by
  downstream stages and single-token sampling.

## Root Cause

`src/v2/models/qwen35/Qwen35Graph.cpp` set the offset only in two cases:
1. `n_k_heads > n_v_heads` (selection)
2. `n_k_heads == n_v_heads && n_k_heads == n_k_heads_full` (identity)

The **expansion** case (`n_k_heads < n_v_heads`, e.g. 27B TP=2:
`n_k_local=16`, `n_v_local=24`) was missing, so rank 1 kept `offset=0`.
The deinterleave helper then computed `k_idx = (j + 0) % 16 = j % 16` on
both ranks, whereas rank 1 should have used
`k_idx = (j + 24) % 16 = (j + 8) % 16`.

The per-head forensic signature in the dumps:

- Rank 0 output matched single-device exactly (cos=1.0, offset=0 coincidentally correct).
- Rank 1 output had `cos = ±1` per head but magnitudes wildly off, with
  `TP[j] * TP[j+8] ≈ SD[24+j] * SD[24+j+8]` — consistent with pairs of
  V-heads getting swapped K-heads.

## Fix

Replaced the two conditional branches with a single TP-V-sharded check:

```cpp
// V is sharded whenever n_v_heads < n_v_heads_full. Set the offset in
// ALL TP modes where V is sharded (selection / identity / expansion).
if (mpi_ctx_ && mpi_ctx_->world_size() > 1 && n_v_heads < n_v_heads_full)
{
    rec_params.global_v_head_offset = mpi_ctx_->rank() * n_v_heads;
}
```

This uniformly covers selection, identity, and expansion GQA regimes for
TP, and is the only condition that actually matters — "V is sharded".

## Files Changed

- `src/v2/models/qwen35/Qwen35Graph.cpp` — unified offset logic

## Tests

All pass against the Integration build (`build_v2_integration`):

- `V2_Integration_Parity_Qwen35_NodeLocalTP_*` (19/19): 08B, 4B, 27B ×
  {ContextInit, Allreduce, Broadcast, Barrier, PrefillParity, DecodeParity}.
- `V2_Integration_Parity_Qwen35_SingleDevice_*` (26/26).
- GDN unit tests (4/4): repeat-interleave, phase-C regression, math
  correctness, dynamic params, fused-QKV TP weight sharding.

27B PrefillParity specifically moved from `cos ≈ 0.916` → within the
declarative thresholds (`cos ≥ 0.96`), matching the single-device
baseline after the fix.
