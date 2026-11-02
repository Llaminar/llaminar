# ConsumerLocal TP2 qualification

The focused ordinary-generation panels pass on two RTX 3090s and two MI50s,
each with a separate CPU expert domain, eight GPU-resident experts per layer,
and one shared CPU ngram source/cache/service. Each GPU owns its own local
ngram cache and service channel. This is homogeneous GPU TP with CPU experts;
it does not qualify mixed-vendor compute TP or a shared GPU storage-owner pool.

## Completed scope

Source admission passed all 16 selected cases: CUDA/ROCm, zero/64-row GPU
cache capacity, and MTP off/fixed1/fixed3/dynamic with retained capacity15.
Every source case retired its resources. The driver window contains 52 new
records and no findings. Source admission proves loading and preparation,
not generation correctness or active MTP acceptance.

All four ordinary-generation panels then passed. Each ran the original 25
frozen histories, 178 scored rows, exact reset replay, 159 host-logit
publications, native capture/publication checks and complete process/model
retirement. The paired zero/64-row configurations have identical numerical
metrics within each backend. The complete panel run took 592.13 seconds.

Values below are prefill / decode. Top-5 is mean set overlap divided by five.

| Backend and group | Mean final-logit KL (nats) | Top-1 | Top-5 |
|---|---:|---:|---:|
| CUDA TP2, required short | 0.018523 / 0.034786 | 93.75% / 93.75% | 91.25% / 90.83% |
| CUDA TP2, required held-out short | 0.020159 / 0.006049 | 100% / 100% | 92.50% / 95.42% |
| ROCm TP2, required short | 0.029528 / 0.028153 | 87.50% / 95.83% | 90.00% / 91.67% |
| ROCm TP2, required held-out short | 0.019763 / 0.004590 | 100% / 100% | 90.00% / 92.92% |

The original CPU/FP32 reference uses the same GGUF. Histories, source files,
arithmetic and acceptance policies were unchanged; the previously approved
1.20-nat individual decode-row guard remains. The QSA-threshold group is
diagnostic and has no acceptance badge. These Integration-binary panels are
focused evidence, not Release performance, complete preflight, or an image
certificate.

## Repairs and outstanding work

The observer previously accepted only four-ROCm local TP. It now derives
homogeneous GPU membership and actual CPU expert ownership from the original
prepared plan, validates each physical participant, and checks exact publication
counts for that cohort. Strict all-GPU capture requirements remain intact.
The observer's focused contract and CLI tests pass.

The pre-candidate rebuilt complete Unit suite and focused regressions pass: 974 Unit
plus 41 focused registrations, with no missing or skipped tests. This combined
1,015-case run took 182.95 seconds and includes both production-defect replays
below. The preceding 1,014-case result remains preserved.
The earlier observer source guard now names its actual helper signature.
The two scans that exceeded their unchanged 30-second limit now take 3.46 and
7.22 seconds after removing unnecessary per-file shell/parser work. Their
source inventory, allowlists and negative controls remain unchanged.

The first active CUDA/C0/fixed1 case failed in a production parent-admission
path: after prefix restoration, the first MTP condition could bypass ordinary
TP forward admission and reach a participant's ngram channel without the
original group ticket. It failed cleanly, with no driver findings. The source
fix applies the existing parent admission and failure scope to scalar/batched
conditions and device-token verifiers. The regression runs the actual rank and
source-service lifecycle with native device APIs doubled: the old core fails
at the missing-group guard, while the fixed core passes 20 replays. Its explicit
preflight entry also passes in the rebuilt test suite. The original CUDA TP2,
zero-GPU-cache, fixed-depth-1 native case now passes in 89.91 seconds, including
serial/grouped byte equality, a real accepted draft token and post-proof lookup
reuse. Its driver window and resource retirement pass; 1.42 GB of owned prefix
backing was removed without reading payloads. The completed first matrix passed
all eight fixed1/fixed3 cases and failed all four Dynamic cases at the diagnostic
frontier assertion described below. Every native process retired and the driver
window was clean. The separate post-proof fixture projects and checks every
requester instead of passing a plural placement to a singleton-only helper.

Dynamic depth exposed a second defect after exact token comparison: rank
diagnostics discarded the child's packed resident-condition observation, so
the acceptance-continuation proof could not authenticate the committed frontier.
The repair authenticates every child observation against its own retained handle,
compares logical payloads, revalidates the original aggregate owner after probing,
and projects through that aggregate identity. Device publication generations are
independent; they are not compared across devices. The existing outer request
owner remains responsible for authenticating the pending condition. No new
transfer or host execution-state mirror was added, and no exact-frontier or
byte-equality assertion was relaxed.

The actual-rank regression covers CUDA/ROCm identities at 2/4/8 participants,
missing/foreign observations, differing payloads, reset/republication, capture
opt-out and replacement of an already-probed participant during the final probe.
Its old-core control fails and twenty corrected replays pass. It is explicitly
registered as `V2_Integration_RankOriginalPendingConditionProbeReplay` in
production preflight. All four original Dynamic native cases now pass, including
acceptance and continuation-state checks, with clean driver and retirement
reports. Their 5.70 GB of owned prefix backing was removed metadata-only.
All eight fixed1/fixed3 cases also pass on this same final binary, with clean
driver and retirement checks. Another 11.40 GB of owned prefix backing was
removed metadata-only. The focused active static-expert matrix is now 12/12
green: two backends, two cache capacities and three enabled MTP policies.

Dynamic expert movement, candidate Release timing, complete preflight and the
wider continuation/admission matrix remain open. These ConsumerLocal TP2 results
do not qualify pooled GPU storage owners, PP or mixed-vendor compute.

## First Release component measurements

All four shared-source component cases pass native byte validation, cache
trajectory checks, two request intervals on retained captures, driver checks
and retirement. Each GPU requests eight tokens across 16 heads, or 128 live
rows per measured batch. There are ten measured samples per case. These use
an authored sparse component file, not the full model, and all live source
pages were OS-hot. No SSD-cold latency is claimed.

| Backend / workload | Median complete two-GPU group latency |
|---|---:|
| CUDA, cache-disabled alternating rows | 4.737 ms |
| CUDA, positive-capacity cache churn | 4.853 ms |
| ROCm, cache-disabled alternating rows | 5.078 ms |
| ROCm, positive-capacity cache churn | 6.856 ms |

These latencies need investigation; a functional pass is not an economy win.
The deliberately small shared RAM/window configuration causes nearly one mmap
window eviction per sourced row. Counters identify churn, but do not yet
attribute its share of latency. Compare useful cache/window capacities and
profile the actual provider/service before changing production policy. The
recorded interval includes both captured lookup submissions through native
completion, excluding input upload, CPU oracle and terminal cleanup.

Six additional controls pass the same native/driver/retirement checks with
64 KiB mmap windows and 1,024 admitted windows. All source pages remain OS-hot;
these are component timings, not SSD-cold or full-model measurements.

| Backend | Zero GPU cache, RAM8 | Zero GPU cache, RAM512 | GPU128, every measured row a hit |
|---|---:|---:|---:|
| CUDA TP2 | 0.978 ms | 0.801 ms | 2.680 ms |
| ROCm TP2 | 2.293 ms | 2.196 ms | 10.776 ms |

Each median covers ten complete group requests of 128 live rows per GPU. The
GPU-hit controls contain 2,560 measured GPU hits and zero measured source rows;
256 initial source rows belong to the excluded warming prelude. Larger mmap
windows remove fixture churn, but the GPU-hit path is still uneconomical.

An isolated ROCm trace identifies repeated serial identity validation as the
first target: measured `Pin`, `AcceptFree` and `AcceptVictim` kernels take about
2.95, 1.83 and 1.82 ms per MI50 dispatch, respectively, each with one thread.
The latter two revalidate retained pins even when no new allocations are
accepted. `CompleteVictims` adds about 1.08 ms of ordered queue/cache updates;
that writer remains outside the first optimization. The trace does not execute
the initially suspected `ReadRows` owner phase, and coordinator cancellation is
only about 3.84 microseconds. Profiled durations are attribution evidence, not
substitutes for the unprofiled timing table.

The candidate cooperates within the same kernel on exact identity validation,
preserving the original ordered commit. Its scoped native, numerical and timing
results are recorded below. No cache validation or zero-decision checks are
removed. The profiling environment predeclares
`HSA_TOOLS_ROCPROFILER_V1_TOOLS=0` because the selected profiler otherwise sets it
during runtime initialization; the original immutable environment guard remains
intact. The first rejected profiling receipt is retained.

## Evidence

All paths below are under `/tmp/qfn-read-ahead-root-integration-r1/`:

- `consumer-local-source-r114/source-admission.json`: 16 source cases and
  retirement results, with the adjacent driver report.
- `observer-localtp-build-r115/result.json` and `observer-focused-r115/`:
  coherent rebuild and focused observer checks.
- `consumer-local-panels-r115/summary.json`: four complete green panels;
  each topology directory retains its numerical and native evidence.
- `unit-and-consumer-focused-r115/`: complete red 1,000-registration run,
  including the original failures and timeouts.
- `consumer-source-mtp-full-build-r118/`, `unit-and-consumer-focused-r118/`
  and `consumer-additional-focused-r118/`: coherent full Integration rebuild
  and all 1,014 passing Unit/focused registrations.
- `consumer-local-active-pilot-r115/`: original active-MTP failure, clean driver
  window and metadata-only cleanup of 356 MB of owned prefix backing.
- `consumer-local-active-pilot-r118/`: passing native reproduction after the
  original parent-admission fix; `consumer-local-active-remaining-r119/` holds
  the completed first matrix, including the four original Dynamic failures.
- `rank-pending-probe-build-r120/`, `rank-pending-probe-release-build-r120/`,
  `unit-and-consumer-focused-r120/`: coherent builds and all 1,015 passing checks.
- `consumer-local-dynamic-r120/`: four green original Dynamic cases;
  `consumer-local-fixed-r121/`: fixed-depth rerun on the same final core.
- `consumer-economy-native-r116/`: four passing Release component runs,
  individual native timings, driver reports and `measurements-summary.json`.
- `consumer-window-controls-r119/`: six unprofiled window/RAM/GPU-hit controls.
- `consumer-rocm-hit-profile-r121/rocm2-gpu-hit-control/`: successful isolated
  kernel trace, resources, measured-trajectory summary and clean driver report;
  the rejected cold-environment attempt remains in `consumer-rocm-hit-profile-r120/`.

The benchmark adapter now consumes the original source-bound ALL manifest;
Release intentionally owns performance executables, not an independent parity
matrix. The local binding input preserves the four previous cases and adds the
four reviewed TP2 bindings. All four TP2 benchmark eligibility checks pass using
the refreshed export in `consumer-declared-benchmark-prepared-r119/`.

Before timing, the native benchmark environment was corrected to preserve the
four original OpenMP/OpenBLAS controls. All 27 benchmark-policy tests pass and
the actual old-runner control fails; loader/preload/SDK overrides remain excluded.
The first CUDA/C64 full-model run completed all warmup/measured work with clean
native retirement and driver evidence, but remains unqualified: ordinary MPI
re-execution stripped the explicitly requested `--mpi-procs 1`, so the child's
runtime document reported automatic selection (`0`). The exact config comparator
correctly rejected that difference. The repaired MPI bootstrap preserves that explicit application setting while
still filtering launch-only actions. Its actual-parser and existing-MPI replay
proves both argument spellings, auto versus explicit counts, and no recursive
launch. No Python normalization exception or relaxed config check was added. Prefix-cache and prefill-cap differences
between observer and benchmark are already handled by the original typed
projection; they are not additional faults. The rejected run remains in
`consumer-declared-baselines-r122/`.


## Matched full-model baseline

All four corrected Release runs pass the exact configuration, original-source,
measurement, native-retirement and driver checks. The complete baseline took
552.62 seconds. Before launch, both coherent builds passed and all 17 selected
original MPI/benchmark registrations passed. The earlier complete Unit/focused
run remains separate evidence; complete preflight is still pending.

This is the original four-shard QFN UD-Q4_K_XL with the shared Q8_0 predictor,
TP2 continuation plus CPU experts, eight GPU-resident experts per layer,
ConsumerLocal ngram caches, static ordinal ownership, MTP off (retained capacity
15), FP16 KV, context4096, and the original arithmetic selectors. Workload:
512 tokenizer-confirmed prefill tokens, 256 generated tokens, one warmup and
three measured iterations. Decode rates use the canonical after-prefill
255-token denominator. Original CPU controls are OMP56, places=cores,
proc_bind=false and OpenBLAS1. Prefix cache is enabled by the canonical benchmark
policy; measured prefix reuse is zero. No profiler or overlapping builds ran.

| Backend / GPU and RAM cache row capacity | Median prefill tok/s | Median decode tok/s |
|---|---:|---:|
| CUDA TP2, C0 | 116.711 | 18.233 |
| CUDA TP2, C64 | 116.681 | 18.140 |
| ROCm TP2, C0 | 112.862 | 14.561 |
| ROCm TP2, C64 | 111.218 | 14.855 |

These are accepted diagnostic baselines, not an image certificate or an
improvement claim. They retain the original serial owner validation kernel.
Every sample and exact environment is in `consumer-declared-baselines-r123/`.
The new cooperative candidate must retain native rejection equivalence and
numerical correctness, then be measured against this baseline.

## Cooperative candidate qualification

The candidate passes native rejection/replay checks on both RTX 3090 and MI50,
plus all 14 selected RX 7900 XTX cases (12 adversarial variants and both
513-identity positive paths). Release and Integration compile the complete
CUDA80/86/89/90 and gfx906/gfx1100 target sets. The complete Unit suite passed
974 registrations on the final facade-repair source in 136.11 seconds, with
no missing or skipped cases. Complete production preflight remains pending.

The fresh r128 native batch passes all 107 cases with no skips and clean driver
and retirement checks. It includes the original cross-vendor and adversarial
identity coverage after every fixture/facade repair. The earlier 99/107 result,
eight focused repairs, and original 81-case/22-failure batch remain preserved.
This complete focused batch is not the complete production-preflight suite.

The repairs keep the original production guards and replay assertions:

- Recorded CPU roots now provide the required 64-byte alignment while preserving
  exact logical copy extents. A debugger located the original SIGSEGV in an
  `OwnerFrame` copy from an ordinary byte vector. The scalar replay regression
  and every GPU adversarial case pass after this correction.
- Old native fixtures now use their actual prepared source, one PMA and properly
  declared graph/service instances. Graph owners are created off-worker and
  recorded on their original worker. The 80-case preparation-lifetime test
  predeclares 80 distinct service instances and rejects duplicate consumption.
- ACK topology injection authenticates the original native capture before
  instantiation and rejects late mutation. Failure inputs use the existing
  first-error publisher instead of treating transient healthy0/error0 as abort.
- Cross-vendor all-format tests split their independent C0/positive capacities
  into explicit cases. Every original format, trajectory, byte assertion and
  30-second timeout remains. All four split cases and all nine two/four-owner
  cross-vendor trajectories pass.
- The retained production capture facade omitted native bounded-kind inspection.
  It now delegates to the original backend's class validator. A focused actual-
  owner regression passes 20 repeats and its old-body control fails. Valid,
  forbidden-kind, empty and over-bound shapes are covered; the explicit preflight
  registration and all six original native failures now pass. Fixture diagnostics
  also sequence string-producing checks before borrowing `c_str()`.

## Cooperative candidate timing

All six quiet Release controls pass byte/state assertions, native retirement
and driver checks. Ten matched samples use the same workload and two-request
schedule as the baseline, without profiler or build overlap. These are authored,
OS-hot component measurements, not SSD-cold latency or full-model speedups.

| Backend / control | Baseline median us | Candidate median us | Baseline / candidate |
|---|---:|---:|---:|
| CUDA, C0 / RAM8 | 978.469 | 856.731 | 1.142x |
| CUDA, C0 / RAM512 | 800.709 | 886.698 | 0.903x |
| CUDA, GPU128 / every row a hit | 2680.364 | 2001.718 | 1.339x |
| ROCm, C0 / RAM8 | 2292.639 | 2315.141 | 0.990x |
| ROCm, C0 / RAM512 | 2195.834 | 2264.792 | 0.970x |
| ROCm, GPU128 / every row a hit | 10775.760 | 5766.395 | 1.869x |

Both GPU-hit cases contain exactly 2,560 measured hits and zero sourced rows;
all candidate samples fall below the baseline range. RAM-only behavior does not
show a consistent improvement; the CUDA RAM512 median is slower and must not be
hidden by the hit-path gain. The full-model numerical panel passes, but matched
ordinary-inference timings below do not establish an end-to-end speedup.

A separate ROCm trace attributes the change to the intended kernels: Pin drops
from about 2,950 to 752 us, and AcceptFree/AcceptVictim from about 1,830 to
258 us per dispatch. CompleteVictims remains about 1,125 us and is now the
largest phase in this trace. These profiled intervals are not canonical latency.

Offline linked-resource/ISA inspection finds no scratch-memory spills. CUDA sm86
Pin/Free/Victim use 40/72/79 registers and 672/704/704 bytes of shared memory;
gfx906 uses 31/33/33 VGPRs with the same LDS sizes. ROCm scalar preservation uses
register-lane moves, not scratch traffic. Compiled active-block ceilings do not
measure occupancy or utilization. CUDA profiler attachment remains paused under
the earlier driver finding; no new attachment was attempted.

Current receipts under `/tmp/qfn-read-ahead-root-integration-r1/`:
`pooled-facade-{release,integration}-build-r127/`, `complete-unit-r127/`,
`pooled-native-complete-r125/`, `pooled-native-fixture-repairs-r126/`,
`pooled-facade-native-repairs-r127/`, `cooperative-owner-gfx1100-r126/`,
`cooperative-owner-controls-r126/` and `cooperative-owner-rocm-hit-profile-r125/`.
`cooperative-owner-native-repair-tracking-r127.json` explicitly distinguishes
its historical repaired cases from a complete green batch. The fresh complete
result is `pooled-native-complete-r128/`, including its own driver report.
Compiled resources are in
`/tmp/qfn-cooperative-owner-resources-r124/REVIEW.md`.

## Cooperative candidate full-model accuracy

All four original CUDA/ROCm TP2 panels pass on the final r127 Integration
binary in 474.32 seconds. Each preserves the same 25 histories, 178 scored rows,
source GGUF, CPU/FP32 reference, acceptance policy and zero/64-row cache settings.
Reset byte equality, native capture observations, retirement and driver windows
are green. Direct byte comparison of all 26 saved logit files per panel against
r115 finds no differences (183,756,800 bytes per side per panel). Every scored
row and numerical group summary is identical, so the KL and agreement values
above remain unchanged. This is the focused original panel scope, not approval
of its diagnostic QSA group or a full production certificate.

Receipts: `cooperative-owner-model-panels-r127/summary.json` and
`cooperative-owner-model-panels-r127/prior-output-comparison.json` under the
same evidence root.

## Cooperative candidate matched Release timing

All four original full-model runs pass, including exact configuration/workload,
native retirement and driver checks. Their total duration is 549.68 seconds.
There was no build or profiler overlap. The original 512-token prompt, 256
generated tokens, one warmup and three measurements, CPU worker settings and
model/source identity remain unchanged. Rates below are medians in tok/s;
decode uses the same 255-token after-prefill denominator.

| Backend / GPU cache rows | Baseline P | Candidate P | Baseline D | Candidate D | Decode change |
|---|---:|---:|---:|---:|---:|
| CUDA / 0 | 116.711 | 116.566 | 18.233 | 18.082 | -0.83% |
| CUDA / 64 | 116.681 | 116.941 | 18.140 | 17.988 | -0.84% |
| ROCm / 0 | 112.862 | 112.989 | 14.561 | 14.706 | +1.00% |
| ROCm / 64 | 111.218 | 111.211 | 14.855 | 14.611 | -1.64% |

Prefill changes are all within 0.23%. The component GPU-hit gain does not imply
faster whole-model generation: these measurements show a small mixed throughput
effect, including three lower decode medians. Keep the original samples and
that limitation visible while tuning the remaining serial field validation and
ordered queue work. No statistical equivalence or general speedup is claimed.
The current candidate's active MTP/movement validation remains a separate gate;
the preceding 12-case static-expert result belongs to the earlier r120/r121 core.

Receipts: `cooperative-owner-full-model-perf-r128/{summary,comparison}.json`;
the comparison authenticates equality of the original exact cell, policy,
source export and workload and retains every old/new timing sample.
