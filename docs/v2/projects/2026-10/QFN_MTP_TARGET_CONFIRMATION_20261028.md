# QFN MTP: confirm targets across disjoint windows

This successor to the [verified-progress trial](QFN_MTP_VERIFIED_PROGRESS_20261028.md)
is the user's chosen default for the measured QFN ROCm TP4 scope as of
2026-10-28. The user explicitly accepted imperfect speed: the 95%-of-best-fixed
comparison is diagnostic, not a promotion blocker. Correctness, HTTP/prefix and
full OpenCode qualification still apply. The previous candidate's 30 cells and
90 measured requests passed correctness, but only six of ten speed comparisons
reached95%; those failures remain unchanged.

## Motivation and isolated change

The previous controller correctly compared actual published state progress and
charged the measured cost of each depth. Nevertheless, consecutive short windows
cover different token histories. Stochastic builtin kept all eight completed
trials and still lost 9.95%: it repeatedly followed poor windows down to shallow
depths and spent later transactions climbing again. Rejection-cache changes alone
cannot address that case. Frozen costs applied to actual depth residency closely
predict the losses; a cost-table correction is not the indicated next experiment.

This candidate requires the same target on separate completed observation
windows before starting a trial. It uses the existing admitted
`promote_consecutive_windows` value, currently 3, for both directions on the
measured cost path. No threshold is fitted to individual prompts or sampling laws.
The startup depth 5, minimum samples 4, window size 16, cooldown 8, immutable costs,
sampling, capacity 15 and admitted depth bounds remain unchanged. Pending trials
still use their existing comparison and completion cadence.

Disjoint windows are not a claim of statistically independent samples. Repeated
agreement can delay a useful move or consistently endorse a poor one. The third
reference window and following trial still cover different input histories. This
experiment does not establish a confidence bound or repay exploration costs by
construction; the full response benchmark remains authoritative.

## State and transition contract

One typed `MTPProposalConfirmation` owns target depth and completed-window count.
The resident measured path projects it onto existing control words 39 and 26.
The control row remains 108 words. The CPU controller uses a mode-exclusive typed
history owner; its published last decision is not mutable confirmation storage.
The five utility words retain their existing trial and rejection-cache meanings.

Only a completed, cooldown-eligible, non-Pending window can add a confirmation.
Repeated inspection of the same bank at runs 4, 5 and 6 must not count as three
windows. A different target starts count 1. The same target increments the count,
even if its supported/unresolved classification changes. Cache filtering happens
before confirmation; a previously proposed but currently vetoed target receives
no credit.

Below the admitted count, the selector remains unchanged. Consume the actual
R/A/Z/F and emitted-Y bank exactly once and increment the evaluated-window count
once. Preserve the rejection reference, cooldown and probe/keep/return counters.
Publish the actual pending target and its count. At the required count, start the
ordinary charged trial using only that final window's V/R as its reference; do
not invent a pooled reference from previously consumed windows. Normalize the
confirmation to the actual selected depth with count zero.

A completed Hold clears confirmation. Partial, clipped and stopped observations
cannot increment it; incomplete eligible windows retain it. Keep/return completion
clears confirmation to the resulting selector. Fresh configure/request reset
clears it; authenticated ContinueResponse preserves it and the partial window
without counting either twice. Scope/policy changes use the existing reset
boundary. Fixed, Off, Observe and unmeasured-policy behavior remain unchanged;
cost-profile activation currently belongs to measured Dynamic scopes only.

Before any observation mutation, validate bounds, admitted confirmation count,
target and streak. Zero streak requires target=current; a positive streak requires
an admitted different target and must remain below the required count. Pending
trials require zero streak. Reject poisoned or overflowing states, rather than
repairing them. The shared host/device transition returns a proposed next state
without mutating its input so the owner can publish a complete valid transition.

## Qualification

The focused suite must prove threshold 1 as a supported configuration and the
actual default of 3. Cover distinct windows at transactions 4/8/12, changed targets,
Hold, evidence-class changes, cache veto/reopening, partial work, stop, clipping,
reset, ContinueResponse, invalid state and unchanged legacy modes. Native CUDA
and ROCm tests must replay real captured production operations, including dirty
repeats and complete control/publication endpoints. Register the focused
regressions explicitly in ProductionTestPreflight.

The physical control layout and packed trace v3 ABI remain unchanged. Preserve
historical capture bytes, producer identities and saved baseline outputs. A new
executable cannot certify an old-source baseline. Candidate replay must stop at
the first unsupported depth and emit no final-state or future-feedback claim.
New authored confirmation cases complement fresh real captures; they do not
replace them. Replay chronology reports the actual retained streak, not an
inference from recommendations.

After complete Unit and focused native checks, rebuild Release with the shipped
CUDA and ROCm spill guards. Freeze one source-bound candidate descriptor, capture
the original five TRAIN prompts under both sampling laws, and require strict
production baseline replay for all four participants. Install small genuine
recorded regressions and refresh the test gates. Then run the unchanged 30-cell,
90-measurement uninstrumented TRAIN panel with fresh Off/Dynamic/frozen-best-fixed
controls. Retain the 95%-of-fixed comparison as a diagnostic under the user's
2026-10-28 acceptance decision; keep all samples and failures. No old exposed
validation payload may select this candidate.

Default adoption is authorized; complete production preflight, HTTP/prefix
qualification and the full OpenCode coding session remain required. Independent
validation stays distinct from exposed TRAIN observations. The minimum16 cadence
experiment remains rejected: it delayed a single
decision without requiring stable targets across separately consumed windows.

## Current implementation evidence

Shared/CPU/resident confirmation, terminal validation and explicit preflight
replays are implemented without changing the108-word layout or trace v3 ABI.
The complete Unit suite passes786 registrations and focused gates pass75 with
no skips: `/tmp/qfn-mtp-confirmation-focused-r3/result.json`. Both native backends
exercise the actual captured operations. The first run exposed two authored
fixture errors: an obsolete immediate-change assertion and a continuation budget
that clipped a supposedly full row. The corrected tests keep default3, require
three separate windows, and preserve partial evidence across genuine clipped
completion and continuation. Production validation was not relaxed.

Release passes with the CUDA80/86/89/90 and ROCm gfx906 spill guards:
`/tmp/qfn-mtp-confirmation-release-r2/result.json`. Fresh10 captures and40 strict
replays pass with original serial output equality and clean retirement:
`/tmp/qfn-mtp-confirmation-v3-train-trace-baseline-replays-r1/manifest.json`.
Predetermined greedy/stochastic Python captures supply the separate
`MTPDepthTraceConfirmationRecorded.h` fixture and explicit
`V2_Integration_MTPDepthTraceReplayRecordedConfirmationV3` preflight registration.
All53 terminal fields and complete saved chronology, including confirmation
streaks, remain independent expected observations. Original survival fixtures
are unchanged. Across the ten captures, probes72→15 and transactions671→607;
one prompt increases transactions, so these counts do not establish a speed win.
The unchanged30-cell/90-measurement TRAIN panel completed with all output,
source/runtime, resource/driver, quiet-window and retirement checks passing.
Only6/10 speed rows pass. Greedy repetition/builtin/long Python miss by
6.399%/13.984%/6.122%; stochastic repetition misses by6.776%. The candidate is
the chosen default under the user's explicit acceptance of imperfect economics.
Complete preflight and HTTP/app qualification continue toward the original
Castagna/full-session goal. Acceptance does not turn the economic panel green.
`/tmp/qfn-mtp-confirmation-comparison-r1/comparison.json` retains every triplicate.
A stale path in the sealed comparison script was handled by an external explicit
receipt adapter importing the unchanged economic functions; no timing was rerun.

Builtin spends32 transactions at depth2 versus8 in the previous candidate, and
reaches depth5 only after208 emitted tokens. All four observed keep/return verdicts
remain the same when confirmation references are pooled; exact prefix-histogram
alternatives do not change the initial lower-depth winner. The defect is policy
economics, not demonstrated arithmetic corruption. A prospective unresolved
startup-anchor probe may test faster recovery, but is not implemented or credited
with unobserved outcomes. Repetition and context-cost diagnosis remain open.
