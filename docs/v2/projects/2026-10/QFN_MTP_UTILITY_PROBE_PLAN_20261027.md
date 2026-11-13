# QFN MTP reversible proposal experiment

This is an implementation plan for an unqualified candidate. The previous global
target fit, minimum-samples16 trial and positive-step cap remain rejected. Their
evidence is retained in [the retraining investigation](QFN_MTP_DEPTH_RETRAINING_20261027.md).
No held-out prompt selects this design or its costs.

## Objective and boundary

Test whether remembering the outcome of a depth change improves complete-response
economics. Retain the parent proposal rules, startup5, ordinary window/cooldown
configuration and retained graph family. Do not simultaneously replace proposal
generation or train another global fixed-depth classifier. The candidate applies
only to the complete measured QFN ROCm TP4/PP1, exact-row capacity15 domain, under
its greedy and stochastic laws. Other domains keep their existing policy.

Use one shared, pure CPU/device transition function. GPU state remains in the
existing device-owned controller row. No clocks, host polling, new collective,
graph restructuring or parallel allocation ledger is needed. Equal committed
inputs and immutable cost profiles must select the same depth on every TP rank.

## Comparison contract

Estimate utility as actual newly emitted tokens divided by eligible transaction
count times the frozen cost of that depth. Costs come from all five ORIGINAL
TRAIN prompts, fixed depths1–15 and their unprofiled triplicates. Preserve prompt
and sample provenance, ranges and deterministic integer rounding. Supplemental
cost experiments remain a separate cohort. These costs are finite-context model
estimates, not measurements of the current device timeline.

Actual emission is authoritative: the resident append publishes its response
delta before depth observation; the CPU runner already computes its committed
response delta. Accepted drafts plus runs is not an equivalent reward when a
condition token is carried between transactions. Budget-clipped observations do
not enter comparison samples; their real cost still counts in the benchmark.

Compare rational utilities exactly, without floating rates or overflowing triple
products. A uint64 denominator holds the product of uint32 runs and cost; an
exact quotient/remainder comparator handles large fractions. Independent host
uint128 tests cover boundaries and extremes.

## Three phases

1. **Clear:** At an ordinary non-hold parent proposal, save incumbent depth,
   eligible runs and actual emission. Apply the proposed retained depth and enter
   ProbePending. Preserve the parent's ordinary proposal cadence.
2. **ProbePending:** Suppress further proposals. After the existing minimum
   sample count of real eligible target transactions, compare with the incumbent.
   Keep only a strictly better estimated utility; a tie returns. This completion
   precedes parent evaluation, even during cooldown. Consume the probe window
   exactly once. Return resets cooldown; keeping does not invent a selector change.
3. **RejectedComparison:** After returning, retain the failed target's sample.
   Block only that same non-hold target while current incumbent utility is at
   least its cached utility. Reopen when incumbent utility becomes strictly worse
   or a different target is proposed. A hold must not erase the rejection.

The samples come from different token positions. This is a heuristic, not an
identical-history causal comparison. A rejected target can improve while the
incumbent remains stable, leaving stale rejection evidence; a noisy incumbent
drop can reopen a poor target. Report these limitations and measure the full
response, including every probe and return. No new expiry threshold is tuned.

## State and lifecycle

Append five learner words: phase, reference depth, reference runs, reference
emission and current-window emission. Add three response-local counters for
actual probes, keeps and returns. The controller grows100→108 words; canonical
PMA/control-tensor sizing owns the additional32 bytes per row. There are no spare
legacy fields to repurpose. The immutable profile is selected through existing
scope/law and compiled source identity, not a second mutable identifier.

NewRequest clears all state. Authenticated ContinueResponse preserves only the
five learner words and existing window/cooldown state. It resets the three report
counters so terminal aggregation cannot count an earlier response twice. Fixed
and Off stay inactive. Observe retains its actual existing proposal semantics
and never enters a fictitious probe or reports one as executed.

Version trace/replay contracts. Preserve the original100-word captures and saved
baseline tool; never pad them with invented comparison state. A108-word initial
row makes the existing trace3520 bytes, within its3912-byte diagnostic bank.
Keep12-byte records: an explicitly tagged token witness carries resident cumulative
response progress or CPU transaction emission. CPU cumulative progress remains
unavailable. Its independent terminal emission witness comes from the existing
production output counter. Fresh actual traces must validate the new baseline.

## Qualification

Before model work, run the complete Unit suite and focused native/preflight
regressions. Cover strict/tied and unequal-size comparisons, overflow boundaries,
carry/rollback/clipping, incomplete probes, rejection retention/reopening,
Observe/Fixed, unchanged generic behavior, dirty request reset, continuation,
terminal counters, exact ABI/capture identity and malformed replay input. Exercise
the same helper through CPU and captured CUDA/ROCm initialize/budget/commit paths.
Register focused defects explicitly in ProductionTestPreflight.

Then rerun all original TRAIN prompts under both sampling laws with fresh Off,
candidate Dynamic and frozen best-fixed controls. Keep actual output equality,
runtime/resource/driver checks and normal retirement. The unchanged maximum5%
loss against fresh fixed remains a hard gate; report original production and
experimental-parent comparisons with all triplicate ranges. Estimated utility or
replay success cannot substitute for measured throughput.

Only a worthwhile TRAIN result advances to untouched held-out prompts. A final
winner still requires one complete fresh production-preflight gate, nine HTTP
configurations, and the full ten-phase OpenCode app session with the agreed model
context, output budget and prefix-cache tiers. No promotion is implied by this plan.

## Current implementation evidence

The runtime and v2 trace/replay integration pass all785 Unit and51 focused checks
with no skips: `/tmp/qfn-mtp-utility-probe-focused-r3/result.json`. Ten new focused
entries are explicitly registered in ProductionTestPreflight, including six
captured CUDA/ROCm utility cases. The shipped CUDA80/86/89/90 and ROCm gfx906
compilation targets pass the Integration spill guard. The gate covers actual
carry yields, clipping, probe keep/return, request reset, pending continuation,
and a completed continuation with returns1/probes0. Existing counter spans stay
unchanged while the canonical arena accounts for all108 words.

The corrected all-depth assertion distinguishes restoring a saved incumbent from
a new upward proposal: an increase without full-prefix evidence must prove the
pending reference, return event and exact failed-target cache. All other such
increases remain forbidden. The source-matched Release build also passes:
`/tmp/qfn-mtp-reversible-utility-release-r1/result.json`. Ten fresh model captures
and forty strict participant baseline replays pass with exact own-Off output,
clean driver/resource checks and normal retirement:
`/tmp/qfn-mtp-train-controller-trace-v2-utility-baseline-replays-r1/manifest.json`.
One stochastic setup was interrupted by SIGTERM and preserved separately with
zero request results; the fresh replacement completed.

Uninstrumented TRAIN timing completed30 cells,90 measured requests and30 warmups.
Correctness, process/resource custody, driver checks and normal retirement pass.
The unchanged95%-of-fresh-best-fixed economic gate fails greedy builtin
(81.921 vs89.291 tok/s,−8.254%) and stochastic prose
(70.419 vs74.913 tok/s,−6.000%). Stochastic long-Python passes narrowly at−4.992%.
Every triplicate is retained; this trial is rejected, not promoted. Evidence:
`/tmp/qfn-mtp-reversible-utility-comparison-r1/comparison.json`.

## Next experiment: proposal provenance

Recorded execution distinguishes two mechanisms. A cached failed depth blocks
the greedy builtin's later measured recommendation; a short probe reverses the
stochastic prose's measured recommendation. Clearing cache on every return
cannot fix the latter path. That private candidate remains unapplied.

The next candidate guards only actual generic heuristic downward proposals.
Measured-table recommendations, including negative steps and holds, retain the
parent action. Derive a transient typed origin from the actual applied branch;
do not infer it from target direction or table-match existence during cooldown.
Pending still completes its real target sample before another parent decision.
Preserve the existing failed-comparison cache for generic proposals. A direct
parent depth change retires that comparison explicitly; a direct hold preserves
it. Every evaluated window consumes actual emission with runs/accepts/zeros/full
counts exactly once. No new state word, cost profile or threshold is introduced.

This removes an observed protective probe too: stochastic long-Python's first
5→4 change was generated, not generic. Only four of53 original parent windows
were generic, on greedy prose and stochastic long-Python. Their later paths and
whole-response performance require fresh measurement. No replay may manufacture
feedback after its first unsupported depth. Retain the complete TRAIN gate,
fresh controls and all correctness checks before opening the untouched successor
holdout or final qualification. Focused replay/native regressions remain explicit
members of ProductionTestPreflight.

The provenance candidate is applied locally. Current-source evidence is786/786
Unit and60/60 focused Integration, zero skips:
`/tmp/qfn-mtp-provenance-focused-r2/result.json`. Ten captured CUDA/ROCm utility
and provenance cases retain complete control/publication byte checks through
dirty repeats. Two genuine43-transaction greedy/stochastic TP4 captures retain
their original producer identity in a device-free preflight fixture; every
decision and all53 owned terminal words match. Forty historical candidate
replays stop exactly where new depths lack observed feedback, with four-rank
agreement; they are not new baseline or timing evidence.

The first focused run failed one authored continuation fixture under three
registrations: its50-token budget clipped the sixteenth incumbent transaction
before the intended generic evaluation. Budget51 admits that physical work,
then two actual clipped transactions finish the response before continuation.
No production change was needed for this correction. The failed receipt remains
at `/tmp/qfn-mtp-provenance-focused-r1/result.json`.

The current Release completes the unchanged full TRAIN panel:30 cells,90 measured
generations and30 warmups. All outputs, resource/driver checks and retirement pass.
Nine of ten economics cases pass, but stochastic long-Python loses8.625% against
fresh fixed3 (56.989 vs62.369 tok/s). The experiment remains rejected:
`/tmp/qfn-mtp-provenance-comparison-r1/comparison.json`. Its two generic trials per
request both keep; the initial generated demotion has no utility protection.

Diagnose the failed path using retained real traces and collect a focused new
trace only where those histories lack observations. Fresh full baseline capture,
held-out evaluation and final qualification remain for an economic winner. Do not
reinterpret candidate replay after an unsupported selector as observed feedback.

## Current experiment: downward proposals from either origin

The retained traces separate direction from proposal origin. The useful
long-Python trial was a generated 5→4 change. The harmful prose trial was an
upward 5→7 change, and the builtin rejection cache blocked an upward 4→5 move.
Test one general change: every strict downward parent proposal may enter the
existing utility trial; upward proposals execute directly. Preserve the cache
veto for both downward origins. A fresh generated recommendation alone must
not erase a previously measured failure. Keep the same proposal table, costs,
startup, cadence, three phases, control ABI and lifecycle.

This candidate is applied but unqualified. The genuine stochastic long-Python
fixture preserves all 73 recorded transactions and original producer identity.
Its first four depth-five transactions emit 14 tokens; the next four depth-four
transactions emit 6. The old provenance binary fails the return at transaction 8
(actual depth 4, expected 5). The new recorded-input regression checks this
boundary, fresh repeated replay and the first unsupported future depth; it has
an explicit `V2_Integration_MTPDepthTraceReplayRecordedDownward` entry. Additional
shared, CPU and captured CUDA/ROCm cases test generated downward keep/return,
cache veto/reopening and direct upward proposals without weakening byte checks.

The current source passes all 786 Unit and 65 focused registrations with no skips:
`/tmp/qfn-mtp-directional-focused-r1/`. The Release build also passes:
`/tmp/qfn-mtp-directional-release-r1/`. Forty historical candidate replays agree
across all four participants; each stops when a new depth needs unobserved
feedback. These are candidate semantics, not fresh baseline certification. The
new matched TRAIN panel passes all ten unchanged 95%-of-fresh-fixed gates:
`/tmp/qfn-mtp-directional-downward-comparison-r1/comparison.json`. All 30 cells,
90 measured requests and 30 warmups are retained; outputs, resources, driver
checks and retirement pass. Stochastic long-Python is now 60.673 versus62.360
tok/s (−2.706%). Greedy builtin is the narrowest pass at95.0758% of fresh fixed.
Independent review reproduces every row from the raw triplicates. Fresh native
trace certification is running; held-out and final qualification remain pending.
