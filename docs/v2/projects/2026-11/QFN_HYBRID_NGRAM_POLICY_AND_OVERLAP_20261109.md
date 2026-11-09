# QFN hybrid ngram policy work and overlap evidence

This audit describes the closed Release r1141 benchmark and r1142 diagnostic
at source `2ae33375f`. They use two MI50 continuation GPUs, two CPU socket
ranks with tensor-sharded experts, RAM ngrams, ConsumerLocal GPU caches and
dynamic MTP. It does not qualify a new cache policy or full-model read-ahead.

## Current observations

Across three measured requests, each GPU reports 198 adaptive lookups and
3,240 unique cache-hit rows, with zero misses, source-native bytes, promotions
or evictions. Those are counters: their zero timing fields do not measure zero
latency. The 36,864 original query IDs use a different unit from unique rows
and must not be used as the denominator for this hit count.

These warm all-hit requests cannot demonstrate RAM fetch overlapping GPU
compute. The selected schedule is `ConsumerScheduled`; no early-current
override was present. This schedule starts lookup at its consumer dependency.
The application currently has no dedicated exposed ngram wait duration or
on-time-prefetch metric in these artifacts.

The profiled prefill contains approximately 58.36 ms and 58.30 ms of summed
pooled-ngram kernel durations on the root and follower GPU. Most of that work
belongs to two coordinator actions:

| Action | Root sum, ms | Follower sum, ms | Calls per GPU |
|---|---:|---:|---:|
| PlanFree | 24.657410 | 25.150846 | 2 |
| CompleteVictims | 30.863804 | 30.306235 | 2 |

Both actions run one policy writer. PlanFree validates participant receipts,
observes the original demand order in four frequency lanes, and constructs
live free-slot plans. CompleteVictims applies original-position recency and
queue promotion/demotion. Neither action scans the allocated cache capacity
or spins waiting for a peer. Skipping them on an all-hit request would change
future eviction decisions.

The 448/64-token buckets supply 7,168/1,024 original ngram IDs. PlanFree takes
21.531/3.127 ms on the root; CompleteVictims takes 26.976/3.888 ms. Approximately
sevenfold timing follows sevenfold live work at fixed cache capacity. This
supports a sequential policy-work diagnosis, not a precise instruction or
memory bottleneck. These are profiled kernel sums, not passive wall-time
savings or transfer wait. The separate approximately 840 ms canonical-ticket
wait is CPU expert-result readiness, not ngram fetch.

## Implementation versus qualification

The dashboard's former blanket statement that adaptive/shared/MTP current
read-ahead was unsupported was stale. `Qwen4ExpGraph` implements the explicit
`EarlyCurrentHostedCandidate` for supported pooled adaptive/shared-reader
roles with retained history and verifier-count identity. Existing evidence is
narrower than full-model performance qualification:

- [CUDA/ROCm RAM/SSD causal tests](QFN_EARLY_CURRENT_CAUSAL_FOCUSED_20261105.md)
  hold the real source worker while independent native work progresses, then
  verify exact delivery and retirement.
- [ROCm TP2 fixed1 and ordinary shared-owner tests](QFN_POOLED_CURRENT_READAHEAD_FOCUSED_20261106.md)
  prove source admission, 1,208 exact comparisons and ordinary reset/reuse.
- The [combined local-cache plan](QFN_LOCAL_HOT_CACHE_HIERARCHY_PLAN.md) records
  separate lifecycle proofs. Those do not establish cold misses or useful
  transformer overlap in this ConsumerLocal benchmark.

Future speculative read-ahead remains disabled at model scheduling level;
lookup/adoption/lifecycle components alone do not establish that integration.

## Next bounded experiments

Four independent frequency-lane writers may reduce PlanFree while preserving
each lane's exact collision, aging and observation order. This is a candidate,
not an installed policy. It needs scalar-state replay, native capture tests,
all shipped architecture builds and isolated profiler/timing evidence.
Recency remains a separate transformation. Even eliminating PlanFree entirely
would save only about 1.3% of this diagnostic prefill span; CPU expert
readiness remains the larger throughput target.

To qualify fetch overlap, use an admitted request with actual cold misses and
nonzero native source bytes. Correlate demand issue, delivery completion,
consumer readiness and concurrent useful GPU compute. Report source-service
time, exposed wait, hidden time and unused bytes separately, retaining exact
generations and live extents. Compare eligible schedules with matched tokens,
placement, cache state and observers; do not infer overlap from warm hits.

Local evidence is retained in `/tmp/qfn-ngram-overlap-audit-r1143/`, including
the original per-bucket launch associations. The underlying passive and
profiled owners are `/tmp/qfn-cpu-auto-panel-blocks-model-r1141/` and
`/tmp/qfn-post-auto-attribution-r1142/`. All native and offline readers for
these observations retired; no cache/model payload was hashed for this audit.
