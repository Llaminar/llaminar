# QFN MTP: verified progress and depth trials

This candidate passes correctness but **fails TRAIN economics and is not promoted**. The
previous downward-only experiment passed all ten original TRAIN comparisons,
but failed three of four frozen validation comparisons, by 5.72%, 8.31% and
12.05% against fresh fixed-depth controls. Its failed receipt remains at
`/tmp/qfn-mtp-directional-downward-final-economic-report-r1/final-evaluation.json`.
The design below uses original TRAIN evidence and source semantics. The failed
validation set must not be presented as untouched evidence for another candidate.

## Findings

The generated target table restricted greedy proposals to depths4/5 and
stochastic proposals to4/5/7. Generic misses can still reach other lower depths,
but a matched hold prevents exploration. A utility gate cannot evaluate an
action which the parent never proposes.

Short trial windows also compare different output frontiers. Let O be actual
compact outputs, L already-emitted leading outputs, P actual published main-state
rows, D=O−P the outgoing publication debt and Y=O−L newly emitted outputs. Then
P=Y+L−D. For non-stopped, unclipped transactions, the actual publication owner
additionally proves P=1+accepted_prefix. Actual Y is correct and must remain the
response ledger; it is not interchangeable with P. In the retained TRAIN traces,
37 of85 evaluated windows have a ±1-token boundary term. That is material for a
four-transaction trial, without establishing a causal performance gain.

Costs remain the immutable original TRAIN estimates: median whole-request decode
time per physical verifier run at each depth, rounded upward to integer
microseconds. They are neither live clocks nor measurements of a particular
four-transaction window. Different token histories and finite-request overhead
remain limitations after the boundary correction.

## Implementation contract

The actual publisher constructs a transient typed witness from O/L/P/stop.
Malformed counts fail before either debt is narrowed to a bool. A stopped
correction can have D=1 even though no next condition is carried; do not erase
that debt using the stop flag. CPU observation follows successful output commit.
Stopped and clipped outcomes remain physical work but do not enter learning.
This eligibility rule applies to all adaptive controllers: stopped outcomes can
be censored by the stop token. Unmeasured domains retain their transition policy,
but now exclude stopped evidence too. Every supplied actual publication must
obey the same eligible progress identity on CPU and GPU.
No host mirror, new allocation, readback, device timer or graph variant is added.

The live trial state remains five words, and the control row remains108 words.
The remembered reference now explicitly stores verified-state progress V. The
current window retains actual emitted Y; transient V=R+A is valid only because
each eligible actual publication was checked. No duplicate mutable V histogram
or accumulator is needed. Both upward and downward scored moves are actual
bounded reversible trials. Keep only a strictly greater V/(R*C); a tie returns.
The existing minimum4/window16/cooldown8, startup5 and request bounds remain.

The shared CPU/device selector considers every admitted depth1..15. Existing
R/A/Z/F counters yield exact integer bounds on same-input prefix truncation. For
depth d, let n=R−Z−F and S=A−dF. At d≥3 and k<d, set
q=(S−n) div(d−2), r=(S−n) mod(d−2). Bounds on summed accepted prefixes are:

```
lower = kF + n + q(k−1) + min(r,k−1)
upper = kF + min(S,kn)
```

Depth1/2 have explicit total special cases. Adjacent truncation is exactly A−F.
At k>d, the bounds are A through A+(k−d)F: unseen acceptance remains unknown.
Add R for verified-state progress. Exact fraction comparison avoids overflow.

First rank strict conservative score improvements by lower/C. Break ties by
distance from current depth, cost, then depth. If none exists, choose the nearest
unresolved candidate in either direction with the same cost/depth tie order.
Otherwise hold. An optimistic upper bound never directly promotes the selector.
Even a conservative same-input gain requires an actual trial: changing depth
changes subsequent verifier frontiers. The original TRAIN fixed4→fixed3 contrast
on greedy long-Python loses7.54% despite a small exact same-input score gain.

Filter the single remembered failed target before ranking; it must not monopolize
selection and hide other candidates. Retain the existing comparison-based retry
rule. This small cache can forget an older failed alternative and sequential
samples can favor a move because the token history changed. No expiry, new
confidence threshold or per-prompt exception is introduced to disguise those
limitations. Complete-response benchmarks remain authoritative.

## Trace and regression contract

Tracev3 uses the three formerly reserved packed bits for actual L/D/stop. Records
remain12 bytes and the maximum trace remains3520 bytes in the existing bank.
Y retains its original units: resident cumulative response count or CPU actual
transaction delta. Replay checks Y/P totals independently against the actual
terminal publishers, as well as carry continuity, stop, controller decisions and
the complete owned terminal projection. Candidate replay stops at the first
unsupported depth; it must not synthesize subsequent feedback.

Old v2 captures keep their producer identity and bytes. They are not upgraded or
restamped. The new reader rejects their incompatible schema. The original ten
fixed15 TRAIN requests completed with exact original serial outputs and physical
work; all40 participant baseline replays passed under their actual v2 executable:
`/tmp/qfn-mtp-fixed15-train-trace-baseline-replays-r1/manifest.json`. Those are
historical input evidence, not new-policy or v3 certification.

Focused device-free tests cover exact integer extrema, all admitted depth/bounds
combinations, large counters, cost ties, missing costs, carry-separated rewards,
both move directions, rejected-target filtering and reset. The first compiled
slice passes31 cases through six Unit/preflight registrations:
`/tmp/qfn-mtp-survival-pure-r2.xml`. An exhaustive case checks245,134 histograms.
One first-run fixture expected the wrong error category for a stale selector;
the corrected test separately checks stale-selector and consumed-window errors.
This is not a complete Unit or native gate.

The complete Unit suite now passes786 registrations; all68 focused registrations
pass without skips or empty GTest filters, including production capture,
continuation, codec, stopped/clipped, serialization and replay on CUDA and ROCm:
`/tmp/qfn-mtp-survival-focused-r2/result.json`. The initial run exposed older
fixtures that hand-wrote a full transaction's publication as d instead of d+1;
they now use the real compact producer and next-condition publication.

Review also found a real prior-window validation defect: a negative accepted
count or an impossible near-maximum full count could be accepted and mutated
before validation. Both safe reproductions failed before the fix and all six
prior-window cases pass afterward. The observer now validates integer population
consistency and next-increment capacity before changing any learner word.
`V2_Integration_MTPDepthWindowCorruptionRejection` explicitly registers the sweep
in ProductionTestPreflight. Evidence: `/tmp/qfn-mtp-prior-window-red-r1.xml` and
`/tmp/qfn-mtp-prior-window-green-r1.xml`. Valid selection costs/cadence are unchanged.

Compilation retains the shipped CUDA80/86/89/90 and ROCm gfx906 spill guard.
Fresh Release passed: `/tmp/qfn-mtp-survival-release-r1/result.json`.
Both original five-prompt TRAIN cohorts now pass the actual v3 capture: all256
output IDs match each original serial control, four participants agree, and
source/runtime/resource/driver/retirement checks pass. All40 strict production
controller baseline replays pass, including complete ordered decisions/windows
and all53 owned terminal words:
`/tmp/qfn-mtp-survival-v3-train-trace-baseline-replays-r1/manifest.json`.
These traced runs make no throughput claim. Small genuine greedy/stochastic
Python fixtures now preserve their producer identities, packed inputs, full
controls, saved ordered decisions/windows and53 owned terminal words. Their
positive/repeated and corruption tests are explicitly registered as
`V2_Integration_MTPDepthTraceReplayRecordedSurvivalV3`. Fresh complete786 Unit and69
focused checks pass without skips: `/tmp/qfn-mtp-survival-focused-r3/result.json`.
Fresh Release also passes at `/tmp/qfn-mtp-survival-release-r2/result.json`.

## Uninstrumented TRAIN result and diagnosis

Both laws completed30 cells,90 measured requests and30 warmups with exact serial
outputs, matched source/runtime, clean resource/driver checks and normal retirement.
The unchanged95%-of-fresh-fixed gate passes only6/10 rows. Greedy repetition/prose
miss by7.794%/6.521%; stochastic repetition/builtin miss by7.865%/9.950%.
Evidence: `/tmp/qfn-mtp-survival-comparison-r1/comparison.json`.
There is no promotion, independent-validation or full-preflight/app advance.

Revalidated replay chronology and exact rational comparisons show no V/Y or
keep/return arithmetic defect. Pricing actual depth residency using the frozen
costs predicts greedy repetition/prose losses7.533%/6.150% and stochastic builtin
9.539%, close to measured losses. These are work-cost estimates, not per-trial
timings or unobserved counterfactuals. Repetition verifies302 rows versus fixed5's
258. Prose takes96 transactions versus fixed4's78. Stochastic builtin takes76
versus fixed6's54, despite every one of eight completed trials being kept and
no returns. A larger rejection cache cannot solve that last case.

The missing evidence is temporal transfer: adjacent four-transaction windows
cover different token histories. Correct same-input bounds and local trial gains
do not prove a future request-wide gain, and repeated acquisition costs real
work in a finite response. Detailed retained notes:
`/tmp/qfn-mtp-survival-greedy-diagnosis-r1-private/NOTE.md` and
`/tmp/qfn-mtp-survival-greedy-diagnosis-r1-private/STOCHASTIC_BUILTIN.md`.
Next review: require consistent target evidence across independent windows using
existing admitted hysteresis. No threshold is fitted, no future feedback is
invented, and no performance improvement is implied. The failed min16 cadence
ablation remains evidence against merely lengthening every sample window.

Final promotion still requires a TRAIN win, genuinely independent validation,
complete fresh production preflight, HTTP/prefix qualification and the full
ten-phase OpenCode session. Old exposed validation prompts cannot qualify a
successor. An immutable prospective validation plan is retained at
`/tmp/qfn-mtp-survival-independent-validation-plan-r1-private/PLAN.md`.
