# Handover: Prefill Graph Phase 6 Closeout

Date: 2026-05-21
Workspace: `/workspaces/llaminar`
Status: completed/superseded continuation note

This note supersedes the earlier Phase 6 continuation handover that described
padded bucket execution, server same-bucket reuse, GDN real-length semantics,
GPU attention padding parity, and eviction/recapture as formerly open items.
Those Tier 1 Phase 6 gates are now accepted for single-device bucketed prefill
graph capture.

## Completed Scope

- Bucket helpers prepare selected bucket lengths, padded token IDs, absolute
  position IDs, and real/bucket metadata through the shared runtime plan path.
- Bucketed `ForwardGraphSignature` entries use bucket length and device as part
  of the top-level forward-cache key.
- `LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS` now caps reusable bucketed prefill
  forward graphs at the `ForwardExecutionEngine` level across bucket lengths and
  devices. Evictions are logged at `INFO` and included in
  `prefillGraphCacheSnapshot()` telemetry even after an evicted bucket's
  per-entry cache is gone.
- LM-head/row-select, KV append, GDN recurrence, and short-conv stages consume
  replay metadata so padded rows do not drive logits, host KV counts, or
  recurrent state.
- Raw/server-style same-bucket reuse runs through the production forward engine
  path and reuses the same captured graph across different real lengths.
- Evicted eligible buckets explicitly rebuild, warm up, and capture again on the
  next request sequence; they do not silently remain on normal prefill.

## Accepted Coverage

- `V2_Integration_CUDAAttentionPaddingParity`
- `V2_Integration_ROCmAttentionPaddingParity`
- `V2_Integration_CUDAGDNPaddedRealLength`
- `V2_Integration_ROCmGDNPaddedRealLength`
- `V2_Integration_PrefillGraphCacheExecution_CUDA`
- `V2_Integration_PrefillGraphCacheExecution_ROCm`

The shared prefill graph-cache execution suite covers exact warmup/capture/replay,
padded same-bucket reuse across real lengths, raw/server same-bucket reuse, KV
real-token advancement, and cross-bucket eviction/recapture with bucket sizes
64/128 and max buckets 1.

## Remaining Work Outside Phase 6

- Tier 2 distributed/collective bucketed graph capture remains future work.
- ExpertParallel rebalance-domain capture and placement-epoch handling remain
  Tier 1.5/Tier 2 work.
- Snapshot, sparse overlay, CPU-participation, TP/PP collective, and unsupported
  placement-mutation paths should continue to fail preflight loudly until their
  dedicated gates are implemented.