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
deleted. **Canonical SHA for the trace build — updated:** the CUDA spills are
now a stacked upstream PR (`Llaminar/llaminar` **#44**, draft, base
`feat/arm64-native-kernels`) with linear tip **`09e14d3ae`** (= old merged
tree `a5412af30` content).

- Use **`09e14d3ae`** for the future nsys trace build (Release sm_121, `.79`).
- `a5412af30f475d655a83108980a4f920d753b2d8` (internal merge commit) and
  `bef53edb6` stay as **provenance only**.
- Spill kernel files verified unchanged vs `be998fdec`.

When nsys lands in `uma3`, rebuild Release sm_121 from `09e14d3ae` and run
the per-bucket trace; results → ideas 24/25.

## Host reservation (2026-11-08, brokered via arm-perf)

- `.79` is reserved **GPU-only** for this trace (172G free; executor
  `spark-cuda-spil`) — runs as soon as nsys lands in `uma3`; no order
  dependency with the CPU lane. 
- `.80` is dedicated to the arm-i8mm lane (CPU-only); perf windows are booked
  through arm-perf.

## Validation

- Same model/prompt/length as idea 24; byte-identical output vs existing
  cells; target: close the gap toward llama.cpp's single-pass 0.94 ms/token;
  re-run the full matrix afterward.

- Canonnical trace build SHA updated: **7e709d60e** (PR #43 head; contains
  a62696317 + spec docs + the ARM base). 09e14d3ae stays provenance-only for
  measured numbers. Trace still blocked on nsys in the uma3 image (idea 1
  item 6); .79 GPU slot reserved.

## STRUCTURE-ONLY trace (2026-11-09, cff351901, 35B-MTP UD-IQ3_S, ~2154-token prefill; sealed K14-IDEA25-TRACE-STRUCT-20261109)

- Per-bucket (GPU timing ON): host launch 64→78.2µs / 256→47.7 / 512→38.8;
  GPU replay **64→63.5ms / 256→191.1ms / 512→340.2ms** (~0.66ms/token; sum
  ≈3.83s GPU-event). Wall-only: launch 67.6/51.3/39.8µs (same shape).
- DECISION-RULE READ: prefill wall ≈46s vs GPU replay ≈3.83s → **~91% of the
  prefill wall is HOST-side/gap, not GPU replay**; per-bucket GPU work scales
  sublinearly (64→512: ~5.4× for 8× tokens → fixed per-bucket overhead);
  host launch itself 40–80µs → the gap is CPU-side staging between chunks
  (lease/maintenance/param staging). Strongly supports IDEA-25 DIRECTION B
  (async multi-bucket submission / move host boundary work off the critical
  path); the 25a host timers (not in cff351901 — GPU validation after the
  94d7cbe2f merge) pinpoint which host term dominates.
- LABEL: STRUCTURE-ONLY — kernel timings NOT #43-representative (cff351901
  has demoted kernels; absolute wall includes old-head effects; the split
  ratio is the structural point). Qwen3.8 numbers await Main's spill fold +
  fixed tip. Note: the earlier 2160-token filler tokenized to 9690 (rejected
  at -c8192) → used 480 fillers ≈ 2154 tokens (recorded).

## 2026-11-09 — trace gating

- Source-side possible at 425002104 (Qwen3.8-Flash-Next-UD schema present).
- RUNTIME trace blocked on: operator go-ahead for the CUDA sm_121 fix lane →
  GPU build at 425002104' → per-bucket trace (Dir B). arm-sparks standby.

## Dir-B trace BLOCKER (2026-11-10, arm-sparks @ e8866ec04): planner does not classify Qwen3.8-UD

The Qwen3.8-Flash-Next-UD SCHEMA is present (Qwen4ExpSchema.h) but planning
FAILS: `planning-selection.select: Unclassified request work:
output_hc_down.weight` (also the 'Higher-priority PMA plan must fit before
ngram storage selection' variant with different backend sets) — the
ImmutableRowTablePlacementPlanner does not classify the Qwen3.8-UD expert
tensors at this tip. SAME error at cff351901 → a PERSISTENT planner gap
across tips, not a recent regression. Blocks the idea-25 Direction-B
per-bucket trace (the trace's prerequisite). No known tip where the model
was successfully planned.

ROUTES: (a) Main fix-back for the planner classification of
output_hc_down.weight (or an explanation if the model needs a registration
step); (b) meanwhile the .79 window proceeds with idea-21b (occupancy
candidate, non-attached). Evidence: arm-sparks' probe logs at e8866ec04.

### #71 progress (Main, 2026-11-10, ~20:58Z): baseline regression reproduced; fix in flight

Focused device-free baseline regression reproduced ALL 9 Flash Next
semantic roles as Unclassified (native ARM build;
/tmp/qfn-planning-role-before.log). Root cause: PlanningRequestCostModel's
shared planningOrdinaryWeightKind() predates Flash Next roles (Unclassified
for ALL HC/indexer/PLE projection+norm/convolution roles + the immutable
row table; output_hc_down is just the first encountered tensor). FIX in
canonical work classification/ownership/request cost: typed terminal-vs-
layer/entry use sites + a typed immutable row-gather consumer; a real-
schema sparse Qwen4 source fixture exercising PMA candidate admission +
finite costing across CPU/CUDA/ROCm; existing unknown-role rejection
stays; NO UD registration/requantization workaround; the PMA/ngram error =
a separate admission gate, not weakened. Lands on #43 (always-push) →
**the combined candidate re-rebases onto the new tip** (the #71 fix touches
NO CPU kernel/checker/budget/probe/envelope files → the register-gate
receipts' native inputs are UNCHANGED → binding re-verification against the
new tree; receipts expected to survive; reviewer confirms at re-re-pin).

### #71 window reconciliation (2026-11-10, ~22:46Z) — UNSTARTED → Main executing

arm-sparks confirmed the .79 real-model plan-only window was NEVER STARTED
(neither at 0d6 nor at 35a827a2 — no new build/output/evidence exists; the
'awaiting outcomes' framing replaced by: unstarted → Main executing NOW).
Exclusive .79 handback to Main is explicit (GPU idle, no concurrent owner);
Main inspects the prepared sketch and executes the corrected-35 build +
exact 4096/8192 plans with the canonical driver begin/finish; e886
controls untouched. arm-sparks' stance: zero GPU action until Main hands
back; its .80 A2 rider set = booked next, outputs HELD pending the A1 #180
fix-back re-review.

### #71 Main execution LAUNCHED (2026-11-10, ~22:52Z) — path-verified, no outcomes claimed

.79 tip-gb10 FF verified exact 35a827a2a (bounded source bundle; exclusive
source clean). New host-owned canonical driver begin/finish wraps a fresh
/home/factory/idea25/build-gb10i-e5ee2 configure + unrestricted build +
exact 4096/8192 plan invocations. Artifacts:
/home/factory/idea25/plan71-main-35a827a2a/; owned watcher monitors the SSH
job (no attached tools/inference/OMP overrides); e886 baseline
binary/core hashes + 4-split model stats captured before/after. The
owner draft was REJECTED (invalid driver CLI + hardcoded OMP20); Main's
scripts retained in artifacts. No outcomes claimed until terminal
artifacts exist.

### #71 OUTCOMES (Main, 2026-11-10, ~23:05Z) — 4096 PASS, 8192 BLOCKED, no workaround

- -c4096 PLAN PASS (JSON written; the original "Unclassified" gone);
  AUTO-selected CPU single (1 admitted, 0 capacity-rejected) — explicitly
  NOT GPU runtime/trace proof; evidence reports actual pre-head rows 256 /
  head 1 in prefill, both 1 in decode. -c8192 FAILS with the exact
  independent "Higher-priority PMA before ngram" error — NO workaround
  applied. Canonical driver window COMPLETE/PASS, 0 findings/new records;
  baseline binary/core hashes + 4-split stats unchanged. Main
  investigating the 8192 diagnostic (ownership/qualification) before
  completion/handback. No inference/profiling ran.
