# QFN MTP: direct output and penalty history

The user accepted the confirmation controller as the QFN ROCm TP4 default.
Its retrospective speed comparisons remain diagnostic. This investigation
concerns generation correctness required before HTTP and OpenCode qualification.

## Native evidence

The preceding complete Unit/preflight gate passed 2,958 tests with 29 hardware
skips and no failures. A subsequent real HTTP Fixed1 request with presence
penalty 1.5 exposed a draft-publication binding error. Stochastic proposals are
deliberately unpenalized, while the target distribution uses request penalties;
the guard incorrectly required those two policies to match. Greedy publication
also applied its initial penalties outside the retained helper. The repaired
binding preserves the distinct laws and makes the captured greedy helper own
its transform once.

Eight tiny production integration cases then exercised both tail backends,
one- and two-device TP domains, both sampling laws, resets and response budgets
1/65/1/17. Greedy token comparisons passed, but stochastic output first differed
at row 13: MTP returned 7 and the actual Off control returned 4.

Terminal-only observations reduced this to missing generated-token history:

- Both executions first returned token 4 with the same seed and sampling policy.
- Immediately after that one-token response, Off had history[4] = 1; MTP had
  zero counts throughout the history bank.
- At row 13, MTP retained the other generated counts but still lacked token 4.
  Its target logit for 4 was 4.0, versus the correct penalized value 2.5.
  Token 6 was penalized correctly in both executions.
- MTP's ordered target support was [4,5,7,6], with probabilities approximately
  [.3102574,.3102574,.3102574,.0692278]. The resulting sample explains the
  mismatch without an RNG or reduction-order hypothesis.

The direct one-token MTP path bypassed grouped accepted-state publication,
which normally commits history. The token was never recorded; a later request
reset or cache restoration did not erase it. GPU MTP forced output has the same
missing publication. Ordinary forced output already has its own publication.
Forced stop tokens retain their documented terminal-control-only contract;
newly sampled stop tokens are generated output and count once.

Native observations and failures are retained under
`/tmp/qfn-penalty-frontier-observation-r1`,
`/tmp/qfn-penalty-frontier-observation-r2` and
`/tmp/qfn-hosted-draft-law-focused-r1`. Temporary observation code was removed.

## Implementation and regression contract

Use one explicitly admitted, controller-owned captured helper to publish a
direct target token into the existing device history. Reuse the existing backend
history kernel and target/history storage. Preserve exact producer-event
ordering, captured identity and reset reuse. Already-published carried tokens
must not increment history again. Pipeline followers own no helper; terminal
TP participants each own their local publication. The canonical graph-owner
inventory prices that helper through PhysicalMemoryAuthority.

The existing eight cases now compare every terminal participant's actual
history after every response chunk against all returned tokens. All eight fail
on the first direct output before the fix, including greedy cases whose token
comparisons alone passed. Eight further native cases cover repeated forced
output, direct sampling, continuation, sampled stop and unchanged forced-stop
behavior across both vendor orders and one/two-device TP domains. They also
check rejected duplicate publication, reset, and actual retained native graph
ownership. Every case is explicitly registered in ProductionTestPreflight.
Device-free ownership tests cover both backends, 1–8 devices, TP/PP, retained
capacity with MTP initially Off, request widths and graph families.

The retained captured helper is implemented and the eight native history and
token comparisons now pass, along with its physical-owner BOM regression.
The sampled-stop unit fixture initially expected the direct publisher for an
unclipped transaction, but that path correctly defers sampling to the grouped
publisher. Its corrected checks of both owners and exactly-once history pass.

The repeated-force cases exposed a separate ordinary-decoding defect before
reaching MTP: the first forced token succeeds, while the second cannot commit
the prior pending token in a captured-domain pipeline. Native singleton PP has
a sampler-free ForwardOnly parent; captured domains lacked the corresponding
complete operation. The implementation reuses their original
captured metadata, activation and model bodies, with one explicitly admitted
parent per physical participant and mirrored terminal-domain publication.
An ordinary response followed by a force then exposed a separate admission
error: the completed ForwardOnly transaction replaces the device controller,
but the next ordinary window incorrectly requested ContinueResponse. A typed
post-ForwardOnly edge now requests NewRequest without resetting KV or history;
subsequent uninterrupted windows still ContinueResponse. The focused unit
regression failed before this fix and passes afterward on both backend mocks.
Existing native ForwardOnly teacher and failure-retirement controls also pass.

The stochastic grouped-to-forced-to-grouped sequence then demonstrated a stale
`first_token_already_in_history` bit. The first fresh sample after the forced
output was omitted from history (tokens 7/5 in the CUDA-tail repetitions),
changing later penalty-dependent outputs. The original eight paired cases and
the two singleton greedy boundary cases remain green. Four TP boundary cases
also exposed an independent fixture error: expected Rank-level rejection
fatally aborts the collective, so that cohort cannot be reused. Negative
publication guards now run on each actual DGO/context; valid operations still
use the public pipeline fanout.

The retained direct-history launch now also clears the original device policy's
successor-history bit. Every nonterminal NewlyEmitted caller commits that token
through the main model row before another sample; the successor is fresh.
AlreadyPublished does not launch, and grouped carried-token publication keeps
its own bit. The exact policy pointer joins capture identity and arena event
publication. No extra launch, payload storage or host policy mirror is added.
The initial 30-test focused batch passed 26 and failed four. All paired
history checks, all greedy boundary cases and both CUDA/ROCm captured kernel
replays passed. Four stochastic grouped → forced → grouped cases remained red,
covering both vendor orders and one/two-device TP domains. The kernel frontier
fix is therefore necessary but does not complete generation qualification.

Terminal-only KV/control observations confirm a separate lost emitted row.
Before force, both Off and MTP have KV count 275 and have returned token 26
(repetition 0) or 29 (repetition 1) without consuming that final token. Off
commits this prior row before publishing forced token 4; MTP instead drops the
AlreadyEmitted condition and commits 4. Both briefly report KV 276, but those
rows name different input tokens. After four further outputs, Off reaches 280
while MTP reaches 279. Matching history counts cannot repair missing main KV
or recurrent state. See `force-frontier-diagnostic-r1.log` in the evidence
directory above; this diagnostic remains a failed run, not a corrected pass.

The installed correction consumes the original event-published AlreadyEmitted
resident condition through shifted and main state before staging the forced
scalar. Its history uses AlreadyPublished, with no second increment. A
PendingResponse bonus was not emitted and remains replaceable; forced stops
retain their terminal-control-only contract. The two actual model rows require
ordered overlay graph retirement and sparse-query admissions at P and P+1.
The ordered-operation regression fails against the old implementation on both
backend mocks and passes after the correction. Each native force now also
checks actual KV count against prompt plus returned tokens, subtracting one
only when the public force result explicitly marks its token Pending. This
detects the missing row even when greedy outputs happen to match.

All 32 focused integrations pass, including all eight force-boundary cases,
captured history replay, exact history, sparse-scope and overlay-retirement
checks. The complete runner Unit suite passes all 169 cases. Three older
fixture expectations were corrected with stronger assertions: retained greedy
publishers apply the exact request penalty once inside the sampler, and a
published canonical target slot is consumed without another copy. Those cases
are explicitly included in the direct-history preflight registration.
Receipts are `carry-unit-red-r1.xml`, `carry-green-r1.xml` (32 integrations pass;
the complete runner alias initially failed three old expectations), and
`carry-unit-contract-green-r1.xml` (all 169 runner cases, nine focused history
contracts and both Release target-selector registrations pass).

Detailed receipts: `forward-only-green-r1.xml` (17 pass, 8 fail),
`admission-red-r1.xml` (the admission regression before its fix), and
`forward-only-green-r2.xml` (20 pass, 6 fail), and `frontier-green-r1.xml`
(26 pass, 4 fail), all under the focused evidence directory above. Preserve
these failures separately from later passes.

The first refreshed complete gate collected all 3,007 registrations:
2,975 passes, 29 hardware skips and three failed registrations. The entire
414-registration multi-device lane had 387 passes and 27 hardware skips,
including all eight direct-history boundaries. The failures were confined to
two old test fixtures: source assertions omitted the explicit publication-right
argument, and both backend context aliases still expected graph totals before
the one bounded history helper. The corrected checks preserve exact event/right
ordering and independently assert 150/108 total, 83 bounded and unchanged
67/25 general auxiliary owners. All three registrations pass after correction;
production code is unchanged by this fixture batch.

Complete failed evidence remains at
`/tmp/qfn-http-r12-draft-law-prerequisites-r1`; focused corrected evidence is
`/tmp/qfn-r12-preflight-fixture-batch-r1-private/focused.xml`. The failed receipt
is not replaced or represented as a clean final gate.

The corrected R13 complete gate finishes all 3,007 registrations: 2,978 pass,
zero fail and 29 hardware skips. Its independent receipt is
`/tmp/qfn-http-r13-draft-law-prerequisites-r1/prerequisites.json`. The 43-test
H13 preparation, fresh metadata exports and genuine Release R2 also pass for
source patch `9ab446f2df53f9707cb29906775dbb5eb52bc90ae96abf1d135493a1fce29ba6`.
These later results preserve the earlier red receipts and their original scope.

The exact Release direct-history replay passes byte/guard checks and quiet,
driver and process-retirement checks. Native events report median amortized
captured-operation times of 1.136 µs on CUDA and 1.94750004 µs on ROCm, with
64 serial operations per retained graph. Static Release resources have zero
spills. This measures the final operation with native scheduling overhead;
there is no baseline component-speedup claim. Source/build bindings and raw
samples are retained at `/tmp/qfn-direct-history-release-native-events-r1/result.json`.

On the original 512-token prompt/256-token repetition workload, the accepted
dynamic default measures 101.04884388535662 decode tok/s versus fresh Off
43.401003324447416. Both use one warmup and three measured requests; all token
IDs match and quiet/native retirement checks pass. Dynamic is 2.9392498% above
the explicitly dated Castagna control at 98.1635713 tok/s; that comparison is
not a new contemporaneous Castagna run. Receipt:
`/tmp/qfn-mtp-accepted-default-natural-eos-comparison-r1.json`.

All nine local HTTP arms now pass: Off, Fixed1/2/3/5/6/15, diagnostic Dynamic
and Serving-Dynamic. Each completes four fresh/full/partial/full prefix
requests and cache cleanup. Off supplies the serial control; the other eight
arms pass their exact-width and serial-comparison checks. The rollup
is `/tmp/qfn-a1-http-r13-r14-qualification-rollup-r1.json`; this is local HTTP
qualification, not full app or release certification.

Serving-Dynamic proves production automatic startup 0→5, actual depths
1/2/3/5, 746 transactions per participant and 14 probes (8 keeps, 6 returns).
Its exact evidence is
`/tmp/qfn-a1-http-qualification-setup-r14-private/Serving-Dynamic-exact-analysis.json`.
The diagnostic Dynamic arm deliberately starts at 15 and observes depths
1/2/3/4/5/15; its result remains distinct from automatic serving admission.
The H13 Serving native run passed, but a passive observer PermissionError made
that attempt red. The complete fresh R14 Serving retry passes with unchanged
source/binary; preserve the earlier failed receipt at
`/tmp/qfn-a1-http-qualification-setup-r13-private/Serving-Dynamic-execution.json`.

All ten OpenCode phases complete. The stress policy passes 120 requests,
51,086 tokens, 128 tool calls with two errors and one compaction. All 65 app
tests and the independent webapp acceptance check pass. Normal shutdown,
runtime/resource audits and native retirement pass. Original evidence:
[session aggregate](/workspaces/llaminar/parity-results/qwen-flash-next/a1-full-opencode-r14-r1/result.json),
[stress policy](/workspaces/llaminar/parity-results/qwen-flash-next/a1-full-opencode-r14-r1/app/stress.json),
[independent acceptance](/workspaces/llaminar/parity-results/qwen-flash-next/a1-full-opencode-r14-r1/app/session-0000-webapp/webapp-acceptance.json).

The original aggregate remains red for the archive-version audit, UTF-8 native
audit and driver warning. Split UTF-8 token pieces now retain original raw
bytes and strictly parsed metadata/IDs. The metadata-only replay passes 120 responses and
128 native calls with zero changed channels:
[fresh native replay](/tmp/qfn-native-utf8-retained-replay-r1/result.json).
The [original native failure](/workspaces/llaminar/parity-results/qwen-flash-next/a1-full-opencode-r14-r1/native-audit/result.json)
remains unchanged. Replay uses the same metadata/decoder without inference;
other audits retain their outcomes.

The original prefix archive was reclaimed and cannot be re-certified. The
repaired v5 auditor passes ten Python cases and a real native writer fixture
(put, touch, eviction, seven sections):
[native qualification](/tmp/qfn-prefix-format5-native-qualification-r1/result.json).
It reads 4,111 metadata bytes and zero payload bytes. All six focused CTest
entries pass, including explicit preflight writer/auditor coverage:
[focused gate](/tmp/qfn-audit-fixes-ctest-r1.xml).
Both complete gate build targets are current, and the post-repair complete Unit
run passes all 786 tests with no skips:
[Unit gate](/tmp/qfn-audit-fixes-complete-unit-r1.xml).
This is fresh fixture evidence, separate from the original app archive.
The AMDGPU USERPTR workqueue warning remains unresolved, and the existing strict
[driver gate](/workspaces/llaminar/parity-results/qwen-flash-next/a1-full-opencode-r14-r1/driver-report.json)
stays red without a waiver. Nine HTTP arms, the 2,978-pass/29-skip prerequisite
gate, Release replay and matched benchmark remain scoped green evidence;
complete certification is not established.
