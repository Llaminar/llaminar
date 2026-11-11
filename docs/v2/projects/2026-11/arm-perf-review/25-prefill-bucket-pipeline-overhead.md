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

### #74 — 8192 root-caused (Main, 2026-11-10, ~23:10Z): separate narrow follow-up

Root: the row selector intentionally returns a NONFITTING MANDATORY
diagnostic plan; ordinary candidate admission passed it into optional-ngram
selection BEFORE canonical admission → invalid_argument aborted the search
rather than a typed capacity rejection. #74 fix (narrow): the canonical
PhysicalMemoryPlanAdmissionCertificate on the frozen base BEFORE optional
selection; NO byte estimates/reserve, no row/quota retry, no placement
workaround. Focused CPU/CUDA/ROCm budget-exhaustion regression running
against the old caller before the patch. Same real commands rechecked after
source ACK/green. #71 fix = 4096 REAL proof; 8192 = separate #74. Main
retains exclusive .79 for this bounded follow-up only (no inference/
profiling).

### #74 native GREEN at 5b76a146c; Main executing the .79 proof (2026-11-10, ~23:15Z)

Complete planning Unit + both focused preflights + 20 consecutive
repetitions EACH (5-GTest row/classif + 1-GTest capacity) = native GREEN;
reviewer #181 source ACK exact 5b. The old-caller regression reproduced
the exact wrong invalid_argument on ALL CPU/CUDA/ROCm. Main ACTUALLY
launched the bounded .79 incremental configure/build + the same 4096/8192
plans at 5b in a new canonical driver window (artifacts
/home/factory/idea25/plan74-main-5b76a146c; the old-35 actual app/core
binaries + config/compileDB retained with matching hash mapping before the
incremental; e886 controls untouched). Awaiting the TYPED REJECTION
outcome — NOT claiming 8192 now fits.

### #71/#74 COMPLETE (bounded scope) — .79 handed back (2026-11-10, ~23:17Z)

Exact 5b real proof: **4096 PASS** (CPU single); **8192 = honest canonical
NoFit** (mandatory 126324719040 B > available 123457634304) — no
invalid_argument/table-selection leak, no accounting/placement/row retry
= the typed capacity rejection the fix was designed to produce. Both
source ACKs + native Unit + focused 20x green. Final driver
complete/pass 0 findings; source clean; e886 controls + model stats
unchanged; 35- and 5b-era actual binaries/config/compileDB retained with
hash mapping. Dir-B at 8192 = NOT ADMITTED by real capacity (a hardware
limit, correctly typed — the trace's -c8192 request is CAPACITY-excluded,
not a defect); the 4096 CPU plan is planning evidence, NOT GPU
trace/inference evidence. Source/doc checkpoint follows; #72
(GroupedSealedPlan stub) queued next, #73 (FusedFinalAdd policy triage)
after; addon #175 + A1 #180 independent holds unchanged.

### Dir-B trace — EXPLICIT SCOPE ACK (coordinator, 2026-11-10, ~23:38Z)

Approved: CPU-single per-bucket structure trace at -c 4096 / 5b76a146c
(host-side CPU timings: forward_graph/prefill_graph_launch per-bucket +
per-stage host wall via PerfStats CSV/JSON; NO GPU runtime, NO 8192, NO
NCU/NSYS/CUPTI; production OMP 20T places=cores close; ONE prefill-
dominated pass: benchmark --config <plan> -n 8 --temperature 0 --seed 42
--no-mtp --iterations 1 with the ~2154-token filler; model = on-disk .79
4-split set — no fetch; runner docker uma4; seal
K14-IDEA25-Q38-TRACE-STRUCT-CPU4096-<date> with the CPU label). Prereq
ownership: coordinator + arm-sparks (self-booked window on Main's
handback); Main's plan-only handback + this ACK = inference authorization
for this bounded run only. CONDITION: launch after arm-sparks classifies
the a/b fast-rc1 failure text (config/authoring → its fix; authorization
boundary → STOP + report). This = the operator-approved idea-25 Dir-B
perf-structure evidence (on-demand helper, not a gate).

### Dir-B trace startup STOP — attribution in progress (2026-11-11, ~00:06Z)

The corrected trace run STOPPED at startup (no inference):
RuntimeInitPhase.cpp:457 'prepare_local_runtime': Startup thread affinity
verification failed: multiple OpenMP threads share <truncated> (production
OMP 20T places=cores bind=close; uma4 container). MAIN'S INITIAL SOURCE
ATTRIBUTION: the verifier/duplicate-physical-core diagnostic
(MPIBootstrapPhase) = INHERITED ec7643c474 (March); the RuntimeInitPhase
explicit-CPU invocation/throw = INHERITED e8dc63c523 (Sept); NO
RuntimeInitPhase delta 2948..5b; :457 = the outer catch, NOT a new
verifier. BUT ARM changes worker placement (e72c1c779/417dd3dc5
fastest-first) → placement NOT exonerated. 'Same flags previously passed'
≠ regression proof (the saved CPU-apply plan exercises explicit-CPU
verification; the earlier 35B GPU/automatic intent may not execute the same
gate). GOMP_SPINCOUNT ≠ affinity repair; self-bootstrap sets the default.
STOP HELD; no OMP/cap/validation workaround. Inbound evidence packet
(commands/binary identity/masks/places/constraints/topology).

### Dir-B STOP — STRONG source mechanism (Main, 2026-11-11, ~00:08Z)

Main: configureClusterRankThreads has the BASE fdc72
ScopedOpenMPPartitionAffinity SPANNING the permanent ARM 417dd3dc5
bindOpenMPWorkersFastestFirst. At RAII return the destructor restores the
MASTER's OLD place (e.g. CPU0) while worker10 retains the new CPU0; the
master should remain fast5 → DUPLICATE 0 / VACANT 5. The AUTOMATIC plan
route can AVOID the scope; the APPLIED saved execution selection ENTERS it.
=> LIKELY **REBASE-INTERACTION PRODUCTION PLACEMENT DEFECT**, not a new
verifier and not a GOMP fix (the 'same flags passed before' = automatic
route). Main will do ONLY local model-free repro/source inspection; needs
the complete error rep to confirm. WARNED: the addon's final register
campaign may need to ride a Main NARROW FIX; do NOT change
placement/validation/env on .79 (no workaround).

### OPERATOR DIRECTIVE (2026-11-11 ~01:05Z): "just install it if needed"

The nsys image-change proposal no longer waits on approval — arm-sparks is
authorized to install nsys directly into the GB10 lane image (uma3 or a
successor tag). Sequence: install → `nsys --version` → MINIMAL attach smoke
test on a trivial CUDA process. If the NVRM `refcntRequestReference_IMPL`
attachment issue reproduces, STOP attach attempts and report the exact
error (station driver issue — installing nsys cannot fix it); if the smoke
test attaches cleanly, the profiler-safety HOLD lifts for nsys-specific
use and idea 25's 15-min prefill trace rerun is UNBLOCKED (still scheduled
after the B12 window to avoid host contention).

### NSYS SMOKE TEST CLEAN (arm-sparks, ~04:51Z) — nsys-specific HOLD LIFTS

- nsys 2025.3.2.474 ALREADY PRESENT in the uma4 image (no install needed;
  the operator's install directive is satisfied by the existing tag).
- Smoke: trivial vectorAdd (nvcc sm_121, exit 0) → direct run ok 3069.0
  (exit 0) → `nsys profile -o smoke ./va` = NSYS_PROFILE=0, report
  generated (1/1 100%); refcnt/NVRM/error grep = 0 hits.
- VERDICT: nsys attach works on this host/driver as of now — the
  nsys-specific profiler HOLD LIFTS; idea-25's 15-min prefill trace rerun
  UNBLOCKS (scheduled post-FF per the queue).
- SCOPE NOTE: the earlier NVRM refcntRequestReference_IMPL warnings came
  from the NCU-attachment path (Main's final2 window) — NCU remains a
  SEPARATE question; nothing here clears it.
- Artifacts: .79 /home/factory/idea25/nsys-smoke/ + nsys-smoke-run.log.
