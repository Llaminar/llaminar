# Idea 25 — Reduce prefill capture/bucket-pipeline overhead on GB10

Status: **Open**. Follow-up to idea 24's dig (2026-11-08); the measurement is
from existing benchmark evidence, no new runs needed to state it.

## Finding

The 3.3× Llaminar-vs-llama.cpp GPU prefill gap (idea 24) is **dominated by
captured bucket-pipeline overhead**, not kernel efficiency:

- Kernels (GEMM 57% + GDN_PROJECTION 39% of stage work) would support
  **~1660 tok/s** at stage speed (0.60 ms/token);
- measured marginal cost is **3.009 ms/token** (≈332 tok/s) plus a fixed
  **~279 ms per-request intercept**;
- so ~80% of prefill wall time is replay/launch/sync overhead of the
  multi-bucket captured pipeline — buckets 512/448/384/256/128/64, graph
  replay + event frontiers per bucket;
- llama.cpp GPU: 0.94 ms/token, single-pass, no bucket boundaries.

## Next probes (need a trace)

1. Per-bucket **replay wall-time** profile (`nsys`, or PerfStats bucket
   counters) to see which boundaries cost most;
2. a **long-prompt single-bucket path** (avoid splitting what can run in one
   capture) and/or **async multi-bucket submission** (overlap replay with
   prior-bucket work instead of serial frontiers) — the two obvious structural
   changes;
3. measure the ~280 ms fixed intercept separately (admission/load/planning
   amortisation) so it is not charged to the pipeline.

## Coordination (2026-11-08)

- Trace owner: `spark-cuda-spil` session; it will run the nsys/PerfStats
  per-bucket trace plus the fixed-intercept separation on `.79` when a window
  exists — needs a GPU-phase run (~30 min), same merged head + model.
- The host is currently with arm-sparks; do **not** schedule unilaterally.
  When a window opens, ping the Spark session to run it, then file the results
  here.

## Validation

- Same model/prompt/length as idea 24; byte-identical output vs existing
  cells; target: close the gap toward llama.cpp's single-pass 0.94 ms/token;
  re-run the full matrix afterward.
