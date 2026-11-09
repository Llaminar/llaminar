## Design notes: single-bucket long prompt vs async multi-bucket submission (idea 25c, 2026-11-09)

Status: notes only, no code. The per-bucket trace decides between these options.

### How prefill runs today (code @ 302318f1e)

- `ForwardExecutionEngine::runPrefillChunkSchedule` validates the whole bucket
  family, then runs its chunks **strictly in sequence**. Each `runPrefillChunk`
  does the following in order:
  1. opens an `IPrefillChunkGraphSubmissionLease` (`host.beginPrefillChunkGraphSubmission`);
  2. builds the fixed-bucket input and runs `updatePrefillReplayParamStages`;
  3. calls `execute` → `launchPrefillGraph` (replay of the bucket's captured graph);
  4. finishes the lease;
  5. evaluates `prefillChunkMaintenanceState` / `onPrefillChunkMaintenance`
     (MoE rebalance boundary).
- The bucket ladder comes from `defaultPrefillGraphBucketSizes()` /
  `LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES` (GB10 trace: 512/448/384/256/128/64).
  For a 2160-token prompt that is a chain of about 5 replays, each with
  host-side boundary work.

### Measurements that now exist (no nsys needed)

| Counter | Measures | Run |
|---|---|---|
| `forward_graph/prefill_graph_launch` | host time of each launch, by bucket | unprofiled |
| `forward_graph/prefill_chunk_host_wall` (new) | runPrefillChunk wall per chunk | unprofiled |
| `forward_graph/prefill_inter_chunk_gap` (new) | host gap between chunks | unprofiled |
| `forward_graph/prefill_family_host_wall` (new) | whole family | unprofiled |
| `stage_gpu/prefill_graph.replay` | GPU-event time per replay | **profiling** (synchronises each replay) |

Reading them:

- Intercept ≈ request prefill time − family wall.
- If chunk wall greatly exceeds GPU replay time, the chunks are host-bound
  (lease, maintenance or launch).
- If the chunk wall ends long after its own launch call returns, the host is
  waiting on the device inside the chunk. That wait is a hidden sync, and the
  lease/finish path is the place to look.

### Option A: one large bucket for long prompts

- Add a bucket equal to the admitted prefill row capacity (for example
  2048/4096), so a long prompt runs as 1–2 replays instead of about 5.
- Cost: workspace/activation BOM scales with the largest bucket. That comes out
  of KV/expert capacity through PhysicalMemoryAuthority, so it has to be
  admitted explicitly, not assumed. It also needs one more captured graph per
  family (capture time plus graph memory).
- Fits if the gap/lease/maintenance terms dominate and are per-chunk constants.
- Does not help if per-token kernel time is the bottleneck. The trace shows
  stage time 0.60 ms/token against 3.0 ms/token wall, so the overhead is the
  target.
- Interaction: the chunk-maintenance (rebalance) cadence of expert placement
  must still fire. With one chunk there is no mid-prefill rebalance boundary,
  so the maintenance policy needs a documented decision.

### Option B: async multi-bucket submission

- Keep the bucket ladder, but submit chunk i+1 before chunk i completes on the
  host side. The replays are already event-ordered on one stream, so only the
  host boundary work has to move off the critical path:
  - lease finish;
  - maintenance evaluation;
  - replay-param staging.
- The protocol cost is serious: the lease and maintenance contracts are
  currently synchronous per chunk. B needs explicit typed staging (maxim 3):
  - double-buffered replay-param publication;
  - a maintenance decision computable from chunk geometry alone, or deferred to
    a family boundary;
  - a lease that covers the whole family.
  No host readback may be added.
- Fits if the inter-chunk gap or lease time dominates while the GPU idles
  between replays. This is visible in nsys as gaps between graph launches.

### Decision rule after the trace

1. If family wall ≈ Σ GPU replay time, the pipeline is not host-bound. Look at
   kernel/occupancy instead (idea 24).
2. If the gap or host chunk overhead is ≥ 50% of family wall and the GPU sits
   idle between launches, choose B (the overhead is host-side and
   serialised).
3. If the overhead per chunk is large and similar for every bucket (launch or
   graph-setup bound), choose A, provided the memory admission for the larger
   bucket fits on GB10 (UMA: the GPU shares 121 GB with the KV/expert pools).
4. Both need the fixed-intercept split first (admission, planning, prefix
   lookup) so that the ~280 ms is not attributed to the pipeline.
