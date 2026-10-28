# QFN exact-row dynamic MTP retraining — 2026-10-27

This is a local investigation record, not a production qualification. The active
summary remains in `QFN_MTP_TP4_TUNING.md`; detailed artifacts are ignored. No
qualified retrained policy has been installed. The exact failed-fit table is now
locally installed solely for a labeled sequential diagnostic; its offline status
remains red and it is not eligible for promotion.

## Why retrain

Retained exact-row verifier graphs changed the cost of each speculative depth.
On the frozen 512-token greedy execution-cost control, fixed depth5 reaches
108.43 after-prefill tok/s versus old Dynamic89.16. Castagna's strongest measured
fixed control reaches100.21. A policy trained on the previous execution costs
cannot establish the new optimum; deeper drafts also remain wasteful when they
are rejected.

## Collection and boundaries

Both original seven-prompt cohorts completed: five training prompts and two
held-out prompts, each with Off, fixed1–15 and the installed Dynamic policy.
That is238 cells and714 measured requests, with one warmup and three measured
requests per cell. All MTP outputs exactly equal their own Off token stream.
Each cohort passed native retirement, source/runtime custody, driver and external
overlap checks. All timings are unprofiled Release measurements.

The workload uses the original QFN UD-Q4_K_XL plus shared-Q8_0 MTP predictor,
four MI50s, TP4/PP1, static GPU experts, RAM ngrams, sharded vocabulary,
28 physical NUMA1 workers, retained capacity15, C4096 and256 emitted tokens.
The decode timing denominator is255 after-prefill tokens. Sampling is either
greedy T0/k1/p1/seed42 or stochastic T0.7/k40/p0.9/seed4242. These are short-context
training results; they do not qualify64K/256K behavior or a complete coding session.

Evidence:
- `parity-results/qwen-flash-next/mtp-depth-campaign-{greedy,stochastic}-r1/`
- `/tmp/qfn-mtp-depth-campaign-{greedy,stochastic}-quiet-r1/result.json`
- `/tmp/qfn-mtp-depth-campaign-{greedy,stochastic}-trainer-r1/`

## Findings

The original independent binary interval fitter could choose overlapping or
unsupported intervals and receive accuracy credit from true negatives. The first
greedy fit failed:90% held-out coverage and9.63% maximum selected-depth throughput
loss. The revised fitter chooses contiguous acceptance regions whose target
depth is within the unchanged5% band for every supported training example.
It minimizes region count, then prefers the training-only startup and measured
throughput. Boundaries come only from adjacent training observations, serialized
to the existing PPM ABI. Ordered predicates are checked after serialization.
Empty support, conflicting inseparable observations and missing held-out domain
evidence fail. Exact preferred-label agreement remains a separate diagnostic.

The revised greedy fit passes all75 training and30 held-out observations. Its
startup is5; its30 scoped rules choose4 or5. Maximum held-out loss is1.6337% and
mean loss0.8169%. All29 generic rules and9 generic startups are preserved. This
is an offline result, not yet measured live Dynamic behavior.

The first combined fit remains failed. All five stochastic training prompts
admit depth5 within5%, but the two stochastic holdouts prefer3 and2. The fit
misses their weaker-acceptance support and loses up to10.5827%. This is repeatable:
the relevant triplicates span less than0.5%, well below the9–11% performance gap.
No holdout was moved into training and no gate was relaxed.

Four independently authored, frozen TRAIN-only prompts completed with the same
binary, sampler, seed and budgets: traceback repair, JSON change planning,
telemetry design and lease testing. All four are retained. Their68 cells and204
measured requests pass output equality, native retirement, driver and external
overlap checks. They add60 fixed training rows and no holdouts. The complete
collection now comprises306 cells and918 measured requests.

The combined210-train/60-holdout fit remains failed before holdout scoring:
stochastic builtin, Python and telemetry have identical aggregate depth1
acceptance369/399, but their acceptable target sets are{5,6,7}, {4,5,6} and{3,4}.
No target satisfies all three. No policy was emitted or installed. Receipt:
`/tmp/qfn-mtp-policy-fit-combined-supplement-r1-private/result.json`.

This is not proof of identical live controller windows. Fixed mode collects
physical work totals without adaptive window bookkeeping. Dynamic/Observe
exclude budget-limited outcomes; the physical counters still include those
drafts. The terminal clipping difference is visible in the raw records, but
eligible accepted/attempted totals cannot be reconstructed exactly from them.
Nor may consumed verifier rows silently replace attempted draft tokens.

The next measurement uses the existing Observe mode with a window larger than
the bounded request, retaining every eligible observation without changing
depth or clearing the window. Existing terminal control words supply the
accepted/attempted/full/zero counts; no new GPU ABI, per-token transfer or state
mirror is needed. These runs are untimed diagnostics paired to the frozen
fixed-depth outputs and work. Original unprofiled throughput remains separate.
Only after this feature correction will another fit be evaluated. A sequential
probe/hold objective may also be needed: one shallow observation need not identify
the final best depth. The actual full-request performance gate remains required.

Separate reporting defect: legacy `rejected_tokens` counts rejected transactions,
not every unaccepted draft token. Benchmark/request and HTTP last/cumulative
acceptance must use `accepted_tokens / draft_steps`. For12 accepted of20 attempts
and2 rejected transactions, the correct rate is60%, not85.71%. Training exports
already use the correct denominator. Narrow regressions cover serialization,
weighted accumulation, zero attempts and reset epochs; no sampling or work counter
changes are required.

## Observation implementation

The existing Observe mode now has typed benchmark-campaign support. Terminal
PerfStats export the eligible accepted/rejected/full/zero/rollback words from
the already copied controller row. An exact metric-family filter keeps these
observations opt-in. The trainer accepts explicit controller evidence while
retaining the original physical counters and unprofiled costs separately.
Incomplete or inconsistent evidence and mixed feature semantics fail.

The revised slice passes783 Unit and24 focused checks, including CUDA/ROCm
retained replays proving clipped physical work is excluded from eligible windows.
Reporting regressions cover benchmark, request and HTTP weighted/reset behavior.
Evidence: `/tmp/qfn-mtp-observation-focused-gate-r2/result.json`. The first gate's
three fixture failures remain recorded; their corrections supplied real draft
attempt counts and one terminal publication per request.

All270 observations passed at depths1–15, zero warmups and one measured request,
with window/minimum-samples1024 and256 output tokens. All four TP participants
agree on the unflushed eligible counts. Exact original Off tokens and fixed
physical work pass. Both native processes retired normally, with clean driver,
source/runtime and overlap checks. Observation timings are not performance evidence.
Evidence: `parity-results/qwen-flash-next/mtp-controller-observe-{greedy,stochastic}-r1/`
and `/tmp/qfn-mtp-controller-observe-{greedy,stochastic}-quiet-r1/result.json`.

The original depth1 collision was a physical/eligible mismatch: Python and
builtin have eligible123/133; telemetry has123/132 despite physical123/133.
This removes the inseparable TRAIN partition. It does not establish that small
live windows distinguish these prompts or that the policy generalizes.

The complete corrected210-train/60-holdout fit remains failed. Its later local
diagnostic installation does not change that result.
Greedy passes75/75 training and30/30 holdout observations (maximum held-out
loss1.6337%). Stochastic training passes135/135, but its heldouts have6 unsupported
observations and24 wrong targets:80% coverage,0% economic accuracy, maximum
matched loss9.3894%. Workshop requires depth2/3 but the matched rules choose4;
library requires3 but rules choose4/5. Exact emitted integer predicates agree
with the canonical scorer, with no rule shadowing. Failed receipt:
`/tmp/qfn-mtp-policy-fit-observe-combined-r1-private/result.json`; diagnosis:
`/tmp/qfn-mtp-observe-fit-diagnosis-r1-private/NOTE.md`.

Two TRAIN-only cost-model alternatives also fail the unchanged5% band. A
geometric acceptance forecast overestimates deep-draft yield on repetition,
selecting15 and losing about45%. Exact adjacent-down truncation is valid on the
same inputs but fails as a request-wide predictor: a forecast0.057% gain for4→3
corresponds to a measured7.54% loss. Neither became a production candidate.
Evidence: `/tmp/qfn-mtp-geometric-cost-selector-r1-private/`.

Transaction costs at stochastic depths2–15 vary by about0.5–1.3% across prompts.
Supplemental depth1 costs about53ms versus34ms originally, with no denominator
error. Two fresh, separately admitted unprofiled arms tested whether the first
speculative shape caused this: Off→Fixed2/11→Fixed1→Fixed1. Both passed all output,
physical-work and retirement checks; all12 measured depth1 requests cost
33.80–34.11ms per transaction. First-shape order does not reproduce the anomaly.
Its cause remains unresolved; the old evidence is preserved, not replaced or
pooled silently. Evidence: `/tmp/qfn-mtp-first-shape-order-comparison-r1.json`.

Before broadening controller state, the exact frozen failed-fit predicates are
available as an explicitly red diagnostic artifact for measuring actual sequential
Dynamic behavior. This is not a new fit, offline gate waiver or default promotion.
A whole-request aggregate cannot prove how small live windows evolve. The failed
receipt and immutable diagnostic manifest remain separate:
`/tmp/qfn-mtp-observe-failed-policy-diagnostic-r1-private/`.

## Controller cadence correction

Review found a separate CPU defect: a generated hold could fall through to the
perfect-acceptance early probe, evaluate after four outcomes and clear evidence.
The resident controller correctly waits for the full16-outcome window on holds.
The CPU readiness check now returns false for a matched hold until the full
window. `V2_Integration_MTPLearnedHoldWindowCadence` checks three consecutive
windows, clipped transactions and both spare capacity and a bound at the hold.
This is unrelated to the unresolved ROCm depth1 timing anomaly. The revised correctness slice passes783 Unit and29 focused checks, no skips.
Evidence: `/tmp/qfn-mtp-red-dynamic-diagnostic-focused-r1/unit/` and
`/tmp/qfn-mtp-red-dynamic-diagnostic-fixture-r3/`. The first private selector
selected no native tests and was rejected; its successor requires a nonempty
selection. The initial new native fixture incorrectly used a blocking copy in
capture; both backends rejected it. Capture-safe asynchronous snapshots now pass
all48-outcome/all-depth scripts and retained dirty replays on CUDA and ROCm.
Neither fixture correction changes inference arithmetic or the failed fit.

## Actual sequential trial of the failed-fit policy

Both laws completed21 cells/63 measured requests with exact original and fresh
serial outputs, no bypasses, clean native retirement and clean quiet/driver checks.
The same Release compiled the frozen failed-fit table; no threshold or source
change occurred between cohorts. This is a correctness pass and an economic
rejection, not a promotion. All rates below are after-prefill tok/s.

| Law / prompt | Old Dynamic | Fresh best-fixed control | New Dynamic | Change vs old |
|---|---:|---:|---:|---:|
| greedy / train-repetition | 89.60 | 108.02 (D5) | 107.74 | +20.25% |
| greedy / train-python | 73.46 | 79.50 (D4) | 76.09 | +3.58% |
| greedy / train-prose | 62.89 | 64.46 (D4) | 61.22 | -2.67% |
| greedy / train-builtin | 79.22 | 89.04 (D6) | 87.12 | +9.98% |
| greedy / train-long-python | 65.86 | 70.13 (D4) | 66.83 | +1.47% |
| greedy / holdout-library-app | 79.51 | 84.46 (D7) | 80.40 | +1.11% |
| greedy / holdout-workshop-design | 67.13 | 67.66 (D4) | 67.72 | +0.89% |
| stochastic / train-repetition | 88.79 | 108.39 (D5) | 108.03 | +21.66% |
| stochastic / train-python | 75.68 | 81.13 (D6) | 82.42 | +8.91% |
| stochastic / train-prose | 71.91 | 75.32 (D5) | 75.07 | +4.39% |
| stochastic / train-builtin | 72.80 | 79.67 (D6) | 81.69 | +12.22% |
| stochastic / train-long-python | 60.46 | 62.84 (D3) | 58.90 | -2.58% |
| stochastic / holdout-library-app | 64.59 | 66.24 (D3) | 61.83 | -4.27% |
| stochastic / holdout-workshop-design | 53.39 | 55.54 (D2) | 50.58 | -5.27% |

The repetition gain is20–22%, but greedy prose and stochastic long-Python regress.
The stochastic heldouts lose4.27% and5.27% versus old Dynamic and6.65%/8.93%
versus their fresh fixed controls. The combined policy remains rejected.

A separate TRAIN-only small-window review exposes reachable cycles. Constant
accepted prefix2 moves5→4, then cycles4↔3; prefix3 can cycle5↔4. Requiring
positive full-prefix evidence before learned promotion blocks these examples
without changing any of the54 frozen TRAIN promotion matches. This does not
prove future-context benefit or a live performance win. No new state or host
readback is needed. Evidence: `/tmp/qfn-red-small-window-diagnosis-r1-private/`.
The full-prefix guard and matching trainer are now applied locally. Legacy
percentage-only inputs retain unknown full-prefix evidence and cannot certify a
positive move. All 783 Unit tests and 31 focused checks pass, with no skips,
including native CUDA/ROCm dirty retained replays and two explicit preflight
registrations. Evidence: `/tmp/qfn-mtp-full-prefix-guard-focused-r1/`.

The strict refit on the same 210 TRAIN / 60 HOLDOUT observations still fails.
Its 80 trained predicates and both startup depths are unchanged; all 54 positive
TRAIN matches have observed full prefixes. No candidate was emitted and no gate
was relaxed. Receipt:
`/tmp/qfn-mtp-full-prefix-observe-refit-result-r2-private/result.json`.
A fresh guarded Release completed both laws: 42 cells and 126 measured requests
pass token equality, source/runtime custody, monitoring and clean retirement.
Every Dynamic case has exactly the same physical work and terminal controller
counters as the unguarded trial. The new guard prevents the constructed cycle,
but did not change these real requests. It is not an observed performance fix.

| Guarded law | Seven-prompt geomean vs old Dynamic | Held-out geomean vs old Dynamic | Held-out geomean vs fresh fixed |
|---|---:|---:|---:|
| Greedy | +5.13% | +1.33% | -2.39% |
| Stochastic | +4.27% | -5.09% | -7.45% |

Stochastic library/workshop remain 61.50/50.51 tok/s, versus old Dynamic
64.59/53.39 and fresh fixed 65.70/55.20. The combined candidate remains rejected.
All fourteen comparisons and raw sample references are retained in
`/tmp/qfn-mtp-full-prefix-guard-complete-review-r1.json`; native outputs are
`parity-results/qwen-flash-next/mtp-full-prefix-guard-dynamic-{greedy,stochastic}-diagnostic-r1/`.

TRAIN-only leave-one-prompt-out validation also finds poor cross-prompt transfer.
All 14 fits succeeded; each withheld all 15 depths of its prompt from both
startup selection and rule fitting. Greedy covers 71/75 observations but selects
economical targets for only 41/75; stochastic covers 131/135 and selects economical
targets for 86/135. Worst matched losses are 20.59%/20.56%. No real HOLDOUT data
participated. Evidence: `/tmp/qfn-mtp-train-only-lopo-r1-private/`.

## Historical v1 controller-input replay

The first diagnostic recorded actual transaction inputs, including clipped and
fixed-depth observations, rather than only evaluated windows. Its standalone
device-free tool replayed those inputs through the production CPU or resident
controller and derived the real window chronology. That version retained the
existing diagnostic bank and 100-word control ABI. Its trace occupied 3,488
bytes within that bank; terminal export copies only its header, initial control
and populated 12-byte records. No inference-loop readback or new allocation is
introduced. CPU policy values retain round-trip floating precision.

The current [utility prototype](QFN_MTP_UTILITY_PROBE_PLAN_20261027.md) uses v2:
108 control words and a 3,520-byte maximum trace in the same diagnostic bank.
The original v1 evidence stays bound to its saved executable; it is never padded
or relabeled as a v2 capture.

Strict baseline replay requires the producer's actual compiled source identity,
ordered inputs and independent terminal witnesses. Explicit candidate replay
permits changed controller/table code with unchanged trace/control types, but
stops when a different depth needs unobserved feedback. It never manufactures
counterfactual acceptance or claims throughput from a trace. The standalone
golden, codec, reset and negative fixtures are integrated. The combined gate
passes all 784 Unit and 41 focused native/preflight checks with no skips,
including actual CUDA/ROCm captures. All focused regressions are explicitly
registered in `ProductionTestPreflight`. Evidence:
`/tmp/qfn-mtp-depth-transaction-trace-focused-r2/`. A genuine new Release build
passes at `/tmp/qfn-mtp-depth-transaction-trace-release-r1/`. Both five-request
TRAIN cohorts complete with exact original Off tokens, identical guarded Dynamic
physical work, four-participant agreement, clean retirement and monitoring.
All forty strict baseline replays pass every recorded transition and owned
terminal word. No HOLDOUT participates and diagnostic timings are not economics.

The interim window-only implementation has been replaced. Its attempted gate
passed 779/784 Unit and 31/33 focused entries. Failures exposed the terminal
observer's source-contract placement, a redundant broad statistics gate, raw
benchmark declarations being normalized before validation, and two invalid
fixtures. Preserve that failed receipt at
`/tmp/qfn-depth-window-journal-focused-r1/`; it is not qualification evidence.

The chronology counts each request once, not once per participant. All requests
start at depth5. Repetition holds5 under both laws; greedy prose visits
5→4→3→2→5→4→5→4→5 and stochastic long-Python visits5→4→3→2→5→7→4→5→4.
Those cycles are observed behavior, not proof that any specific alternative
improves throughput. Next attribute each window to the actual canonical learned
or generic decision, then freeze one focused live experiment. Counterfactual
replay must stop at the first depth requiring feedback absent from the trace.

Evidence: `parity-results/qwen-flash-next/mtp-train-controller-trace-{greedy,stochastic}-r1/`,
`/tmp/qfn-mtp-train-controller-trace-baseline-replays-r1/manifest.json` and
`/tmp/qfn-mtp-train-controller-trace-analysis-r1.json`.
See the stable [replay procedure](../../../../tests/v2/unit/mtp/README.md).

## Decision cadence ablation

Canonical controller attribution finds 49 learned matches in 53 evaluated
windows. The four unmatched windows are precisely the 4→3 and 3→2 descents in
greedy prose and stochastic long-Python. They use the generic low-acceptance
threshold, not the zero-acceptance threshold. The initial 5→4 moves after four
outcomes and later 2→5/5→7 jumps are learned decisions. Attribution executes
the real controller after validating all forty baseline replays; it does not
reimplement its decisions in Python. Evidence:
`/tmp/qfn-mtp-train-window-attribution-r1-private/attribution.json`.

The next experiment changes only the existing minimum-sample setting from4
to16. Window16, cooldown8, generated predicates, automatic startup5, capacity15,
model, topology, prompts and samplers remain fixed. This tests early evaluation
without a rebuild or new controller state. Four separate admissions compare
minimum4 and minimum16 under both laws on the original five TRAIN prompts,
using fresh Off controls, one warmup and three measured requests per cell.
Greedy runs minimum4 then16; stochastic reverses that order. Off drift is reported,
not silently normalized away. Timing runs disable trace and profiler collection.

This is a cadence ablation, not a successful refit or a cure for ambiguous global
labels. Preserve the failed offline objective. Score every complete response,
exact own-Off output, all prompt rates and fixed/baseline losses; do not infer
counterfactual acceptance after a replay diverges. No HOLDOUT selects this setting.

Of42 actual moves,38 occur before16 eligible outcomes. Canonical candidate replay
with minimum16 preserves both repetition paths completely; each of the other
eight requests stops at its first changed depth, after4–20 transactions. All
four participants agree. The tool consumes no subsequent unobserved feedback.
Evidence: `/tmp/qfn-mtp-train-window-attribution-r1-private/minimum-samples16-summary.json`.

All four admissions completed normally:40 cells,120 measured responses and40
warmups pass exact serial output, source/runtime custody and resource/driver
checks. The candidate is rejected. Serial Off drift stays below0.4%; the slower
cadence reduces changes but does not improve complete-response economics.

| Law / TRAIN prompt | Minimum4 tok/s | Minimum16 tok/s | Change |
|---|---:|---:|---:|
| greedy / Python |76.613|79.266|+3.46%|
| greedy / repetition |108.345|108.246|−0.09%|
| greedy / builtin |87.762|80.692|−8.06%|
| greedy / long-Python |67.399|65.096|−3.42%|
| greedy / prose |61.777|61.611|−0.27%|
| stochastic / Python |82.231|80.138|−2.55%|
| stochastic / repetition |107.651|107.390|−0.24%|
| stochastic / builtin |81.721|76.499|−6.39%|
| stochastic / long-Python |58.895|58.516|−0.64%|
| stochastic / prose |74.984|71.894|−4.12%|

Equal-prompt geometric-mean changes are−1.75% greedy and−2.82% stochastic.
Evidence: `/tmp/qfn-mtp-cadence-ablation-comparison-r1/comparison.json` and
`parity-results/qwen-flash-next/mtp-cadence-ablation-*-r1/`.

## One-step positive-action experiment

The next independent ablation keeps minimum4/window16/cooldown8/startup5, but
caps positive actions greater than1 to+1 only in the two measured QFN scopes.
Canonical parsing/emission changes18 rules out of109. All predicate boundaries,
rule ordering, negative actions, holds, eleven startup rows and29 unmeasured
domain rows are preserved. No new controller state or ABI is introduced.
This is explicitly experimental; it does not change the failed global-label
fit into a pass. Manifest: `/tmp/qfn-mtp-positive-action-cap1-private/manifest.json`.

The artifact and regressions are locally applied. The new
`V2_Integration_MTPScopedPositiveActionCapOneStep` is explicitly in production
preflight; retained all-depth CUDA/ROCm tests also assert the scoped one-step
invariant. All784 Unit and42 focused native checks pass with zero skips at
`/tmp/qfn-mtp-positive-step-cap-focused-r1/`; the new Release build also passes.
Canonical replay reproduces all40 original participant traces and verifies all40
candidate paths, stopping at the first unsupported selector. The private r1
validator incorrectly required a field absent from the canonical early-stop
schema; r2 fixes that expectation and preserves the original failure. No
production change was needed. Evidence:
`/tmp/qfn-mtp-positive-step-cap-cli-replays-r2/manifest.json`.

The new Release now measures both laws with fresh Off→Dynamic→original-best-fixed controls, keeping
the same preceding Off history as the cadence control. No HOLDOUT selects the
candidate or comparator. Timing preparation:
`/tmp/qfn-mtp-positive-step-cap-timing-r1-private/README.md`.

The complete experiment is rejected. All30 cells,90 measured responses and30
warmups pass exact Off-token equality, source/runtime custody, resource and driver
checks, and normal retirement. Greedy changes+0.12% and stochastic−0.62% on the
equal-prompt geometric mean relative to parent minimum4. The latter has material
regressions; fewer upward steps are not a sufficient economic objective.

| Law / TRAIN prompt | Candidate tok/s | Fresh best-fixed tok/s | Versus parent | Versus fixed |
|---|---:|---:|---:|---:|
| greedy / Python |76.371|80.132|−0.32%|−4.69%|
| greedy / repetition |108.201|108.279|−0.13%|−0.07%|
| greedy / builtin |87.530|89.439|−0.26%|−2.13%|
| greedy / long-Python |67.365|70.359|−0.05%|−4.26%|
| greedy / prose |62.614|64.838|+1.36%|−3.43%|
| stochastic / Python |83.389|80.832|+1.41%|+3.16%|
| stochastic / repetition |108.286|108.049|+0.59%|+0.22%|
| stochastic / builtin |76.981|79.325|−5.80%|−2.95%|
| stochastic / long-Python |57.743|62.481|−1.96%|−7.58%|
| stochastic / prose |77.165|75.082|+2.91%|+2.77%|

Fresh fixed drift is at most0.33%; Off drift is below0.36%. The long-Python
result misses the unchanged5% fixed-loss limit. No HOLDOUT was opened or run.
Evidence: `/tmp/qfn-mtp-positive-step-cap-comparison-r1/comparison.json`.
The artifact and its three test-file edits were restored to their exact parent
snapshots after both native processes retired. Rejected artifacts and test
patches remain in ignored evidence; canonical trace regressions remain installed.
The compiled+1 binaries are now stale and must not run against restored source.
Retirement: `/tmp/qfn-mtp-positive-step-cap-retirement-r1/receipt.json`.

A scoped cost veto on generic downward moves is also not worth a live trial:
the original stochastic long-Python4→3 and3→2 windows still pass its conservative
TRAIN cost-envelope inequality (ratios1.01790 and1.01411). It cannot change that
failing path. Next investigate explicit remembered utility for a probe/return
policy; do not keep varying cadence or thresholds without a new objective.

## Acceptance still required

Keep the failed global-label objective and its receipts unchanged. A new
sequential-action policy needs its own complete-response evidence; matching an
intermediate action to a whole-request fixed winner is not its objective.
Build each candidate and run the complete Unit suite plus focused native
controller/ABI/regression checks. Measure fresh Off,
best-fixed controls and new Dynamic on all seven original prompts for both laws.
Retain the unchanged5% maximum loss target and baseline no-regression comparison.
Separately use untimed production counters to prove device startup, actual learned
matches and exact-row retained replay. Timed runs keep counters/profilers disabled.

Only a proven winner proceeds through fresh complete production prerequisites,
HTTP stochastic/prefix reuse and the full ten-phase OpenCode coding session.
Old qualification receipts and failed fits remain historical evidence.
