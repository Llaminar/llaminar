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
- **In progress (2026-11-08 14:42, operator-approved, `.79` uncontested):**
  `spark-cuda-spil` is running the trace now — nsys CUDA/NVTX + gpu_trace
  stats on `arm-validation/0cdcecc3c-spills` (Release sm_121) with
  Qwen3.8-Flash-Next-UD-Q4_K_XL, 2160-token prefill; ETA ~45–60 min. Results
  will be filed into this idea (and idea 24) on arrival.

## Trace attempt result (2026-11-08 14:49) — blocked on tooling

- **Blocker 1:** SDK image `llaminar-arm-engine-sdk-cu13:uma3` has **no nsys**
  (only `ncu`; host `nsys 13.0.3` exists at /usr/local/bin/nsys but the
  binary is container-linked).
- **Blocker 2:** host-side run to use host nsys fails (missing
  libmpi*/libopen-pal/libopen-rte/libhwloc/libevent, then OpenMPI self-launch
  breaks under prte semantics outside the container; `--no-mpi-bootstrap`
  hits missing /usr/share/openmpi runtime files). Not a code issue.
- **Reproduced without profiler** (DEBUG run, same setup, 2160 tokens): prefill
  **320.12 tok/s** (6747.48 ms), decode **22.73 tok/s**.
- **Evidence-based split (still valid):** marginal prefill 3.01–3.22 ms/token
  across 2160/7200-token runs, fixed intercept ≈ **280 ms** per request,
  captured stage kernel time **0.60 ms/token** (≈1660 tok/s at stage speed)
  → ~80% of prefill wall time is inter-bucket replay/launch/sync + fixed
  setup, not kernels (GEMM 57% + GDN_PROJECTION 39% of stage work).
- **Action needed (tooling):** add `nvidia-nsight-systems` to the uma3 dev
  image, or a documented host-attach/profile path from inside the container.
  With nsys in the image the trace is a ~15-min run (`nsys profile -t
  cuda,nvtx --stats`); `spark-cuda-spil` will execute it then. `.79` returned
  clean.

## Rerun tree (SHA-only policy, 2026-11-08)

The old `arm-validation` pin branch (and its merged tree `bef53edb6`) is
deleted. Replacement merged tree, built locally per the SHA-only policy:

- merge commit **`a5412af30f475d655a83108980a4f920d753b2d8`**
  (tree `80c81df7c0eb207c978bc2b49671d0067553ef12`),
  `codex/cuda-sm121-spills` merged with upstream
  `Llaminar/llaminar feat/arm64-native-kernels` @ `1a01e6786`;
- spill kernel files verified unchanged vs `be998fdec`.

When nsys lands in `uma3`, rebuild Release sm_121 from this SHA on `.79` and
run the per-bucket trace; results → ideas 24/25.

## Validation

- Same model/prompt/length as idea 24; byte-identical output vs existing
  cells; target: close the gap toward llama.cpp's single-pass 0.94 ms/token;
  re-run the full matrix afterward.
