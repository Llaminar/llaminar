# QFN TP4 MTP host profile

The accepted `3d46cf4a9` checkpoint reaches 90.164 decode tok/s at fixed depth 3,
against a fresh 91.038 tok/s Castagna reference. Prefill is 789.686 versus
434.574 tok/s. These are medians of three unprofiled requests after one warmup,
using the frozen 512-token prompt and 256 outputs. The repeated-token workload
has 100% draft acceptance; it does not establish coding-session performance.

## Measured remaining host work

An independent four-second CPU `cycles:u` sample at 499 Hz captured 7,323
samples with zero lost samples. The original native benchmark owner, MPI worker
identity, runtime/source bindings, quiet window, driver observations and normal
retirement all pass. Sampling uses DWARF call chains and attaches to the actual
MPI compute process. Profiled rates are explicitly ineligible for the timing
table.

| Disjoint call-stack category | Samples | Share of sampled cycles |
|---|---:|---:|
| Device-controller ticket completion wait | 6,376 | 94.637% |
| Hosted verifier capture-variant walk | 69 | 0.972% |
| Hosted verifier launch preparation | 12 | 0.178% |
| Other hosted verifier work | 9 | 0.133% |
| Other hosted generation | 31 | 0.467% |
| Prefill | 325 | 1.895% |
| Other / unattributed | 501 | 1.717% |

Percentages weight each sample's period. The capture-variant category requires
both `replayHostedMTPGroupedVerifier` and `computeCaptureVariantSignature` in the
same stack. The interval spans request boundaries, so prefill remains separate.
These are CPU-cycle shares, not wall-time or GPU-time fractions. Ticket waiting
does not identify the GPU kernel responsible for that wait.

The original allocation-registry bottleneck is no longer prominent. The small
remaining signature walk includes graph-name map lookup, string comparison,
name hashing and live stage signatures. A narrow candidate can cache ordered
node pointers and immutable order-name hashes while reading each node's current
stage, type and variant on every invocation. Stage replacement and zero-to-
nonzero variants must remain visible; request reset must preserve the view and
topology mutation must invalidate it. This is a candidate, not a measured win.

The candidate's fresh fixed3 median is 90.271 tok/s. A new run of the preserved
accepted binary reaches 90.369 tok/s (original accepted run: 90.164). Token IDs
and 192 accepted drafts / 64 verifier transactions per request agree. Both new
quiet, runtime, driver and retirement receipts pass. The difference is noise;
the ordered-view candidate was rejected and removed before the exact-row build.
The frozen R6 binary/results retain their original identity.

Automatic initialization with capacity3 reaches 90.177 tok/s and holds depth3
over twelve measured policy windows. This removes the earlier explicit-initial1
startup penalty; it does not establish a new execution speedup. Castagna's native
adaptive capacity15 baseline reaches 65.487 tok/s, accepting 695/1,200 drafted
tokens across the three measurements. Its faster depth3 result remains the
performance target. Full-capacity Llaminar comparisons have retired successfully; the current table is in the linked tuning dashboard.

## Worker placement

All four sampled `TPWorkerPool::workerLoop` TIDs retain the admitted OpenMP
partition in all eleven task observations: logical CPUs 28–55 and their 84–111
SMT siblings, representing 28 physical cores. The observed workers run on
different cores. The main thread's `[28,84]` mask therefore does not imply that
GPU submission workers contend on one core. No affinity change is justified.

The first diagnostic collectors rejected valid physical-core/helper masks
before launching `perf`; another attachment attempt raced its predecessor's
completion check. Those failed receipts remain unchanged. The successful run
uses one collector and authenticates physical-core identities, recording all
auxiliary masks without changing them.

## Separate policy defect

The MTP model-class helper searches the architecture name for `moe`. QFN's
`qwen4exp` name therefore selects Dense despite its loaded 512-expert/top-10
geometry. Correcting the context to MoE leaves greedy automatic startup at 3,
but changes stochastic automatic startup from 1 to 3 and selects the MoE learned
thresholds. User-authored bounds and fixed depths remain separate. A metadata-
based fix and explicit preflight regression pass focused validation. The new
source also passes all 783 Unit tests and seven native CUDA/ROCm captured-TP and
learned-depth tests. Full prerequisite refresh and production request checks
remain pending; the accepted checkpoint and old timing receipts are not
retrospectively relabeled.

The existing adaptive timing starts explicitly at 1 with min/max/capacity 1/3/3.
A fresh automatic-initial comparison must change only initial depth to 0 and
retain that envelope. Full production serving instead retains maximum/capacity
15 and needs a separately labeled measurement.

## Evidence and next acceptance

- Unprofiled: `parity-results/qwen-flash-next/mtp-tp4-llaminar-fixed3-qualified-r1/`
  and `mtp-tp4-castagna-fixed3-refresh-r1/`, including their quiet-window receipts.
- Successful profile: `/tmp/qfn-mtp-postqualification-cpu-profile-r5-private/`,
  especially `verification.json`, `profile/sample-summary.json`, `perf.data`,
  task observations and original native/quiet results.
- Placement join: `/tmp/qfn-capture-identity-view-private/affinity-validation.json`.
- Policy source audit: `/tmp/qfn-mtp-auto-initial-model-class-audit-r1-private/findings.md`.
- Candidate checks: `/tmp/qfn-mtp-policy-identity-focused-tests-r2.xml` (6/6),
  `/tmp/qfn-mtp-policy-identity-unit-tests-r1.xml` (783/783), and
  `/tmp/qfn-mtp-policy-identity-device-tests-r1.xml` (7/7).
- Fresh candidate/control: `parity-results/qwen-flash-next/mtp-tp4-llaminar-fixed3-identity-r1/`,
  `mtp-tp4-llaminar-fixed3-accepted-repeat-r7-r1/`,
  `mtp-tp4-llaminar-auto-initial-cap3-identity-r1/`, and
  `mtp-tp4-castagna-adaptive-cap15-r1/`. The accepted repeat's quiet receipt is
  `/tmp/qfn-r7-accepted-fixed3-quiet-r1/result.json`; other quiet receipts are
  adjacent `-quiet` directories. Its implementation source and current checkout
  observation guard are explicitly separate in the R7 packet.

The frozen R6 depth comparison is complete. The ordered-view candidate is removed;
the model-class fix and exact hosted verifier families are integrated. Paired
production-capture GPU traces remain available for attributing any remaining
kernel deficits after the exact-row comparison. Final acceptance still requires the applicable full
gates, dynamic/default-policy checks, a meaningful-text comparison and the
complete ten-phase OpenCode session.

## Exact-row candidate, 2026-10-27

R6 full-capacity controls all pass native, quiet, driver and normal retirement:
Off819.749P/43.433D, Fixed3 789.413P/90.161D, Fixed7 789.076P/86.388D,
Fixed15 788.805P/59.621D, default Dynamic789.761P/62.305D. Fixed3 and Dynamic
perform identical useful work (192 accepted drafts,64 verifier calls,256 logical
rows per measured request), but Dynamic keeps the16-row physical verifier envelope.
Fixed7 accepts210/322 drafts and performs46 verifier calls/368 rows. The
Castagna comparison remains its fastest measured Fixed3 91.038D, not its
slower adaptive-cap15 65.487D.

Evidence roots under parity-results/qwen-flash-next: mtp-tp4-llaminar-
{off,fixed3,fixed7,fixed15,default-dynamic}-cap15-r1, each with adjacent
-quiet observations. The frozen packet is /tmp/qfn-rocm-tp4-mtp-benchmark-r6-private.

User-authorized implementation: exact hosted rows2 through16, using existing
authenticated tickets. Each width retains its actual setup signature, preparation
and publication. Pipeline followers retain complete transactions per width.
Native conditional CUDA continues its admitted envelope. Execution declaration
is projected before capture, shared with admission and checked against native
capabilities; it cannot grow a graph family after the memory decision.

The coherent Integration R4 build passes all783 Unit tests and46 focused
integration registrations;12 eight-GPU pipeline cells skip because this host
has two CUDA cards. The exact native fixture replays all widths2..16 over20
rounds, including dirty large/small transitions, rejection/EOS, clipped response
budgets, carried leading tokens and invalid counts. Repeated-request state
ownership and both CUDA/ROCm pipeline orders pass. This is focused validation,
not a new complete-preflight or real-model certificate.

The first run found a real setup defect: hosted pipeline followers were asked
for a local request admission, although the final stage is its sole authority.
Followers now borrow that owner's immutable depth bounds; no controller shadow
is created. Four direct planner fixtures needed their mixed-domain declaration,
and the new native fixture needed its required verifier-condition output and
correct destination/source ordering. Original failed receipts remain preserved.

Evidence: /tmp/qfn-exact-hosted-build-r4/result.json and
/tmp/qfn-exact-hosted-focused-gate-r1/result.json (source unchanged across both
phases), with separate complete JUnit/logs. The R8 Release packet is
/tmp/qfn-rocm-tp4-mtp-exact-rows-r8-private; its successful build and failed model run are described below. Matched
capture/RSS/VRAM baselines are in
/tmp/qfn-exact-hosted-verifier-cost-baseline-r1.json. Model accuracy and throughput
are not yet measured for the candidate; old qualification remains historical.

A separate pre-existing audit found that overlay epoch/maintenance graph owners
are not fully expressed by the old activation-family allowance. The current
Static A1 sidecar family has an allowance of2 matching its two epoch owners; this
is not an exact-row regression. Keep that independent follow-up scoped rather
than adding a new ledger here. Details:
/tmp/qfn-exact-hosted-verifier-owner-audit-private/pre-existing-overlay-helpers.md.

## First exact-row Release run

R8 Release builds and the packet finalizes with unchanged source/runtime. The
real-model defaultDynamic cap15 run fails warmup before valid timing samples:
initial3 creates draft-publication slots0..2, but composing the full retained
family requires slots3..14. Initial1 also cannot assume an executed chained
sidecar. Cold materialization must prepare these helpers from original admitted
identities without executing unused model drafts. Native regressions will add
initial1/3/7 with capacity15 while retaining initial15 coverage.

Preserve the failed native and quiet receipts in
parity-results/qwen-flash-next/mtp-tp4-llaminar-default-dynamic-exact-r8-r1{,-quiet}.
All processes retire and observed total VRAM returns to0.0405GiB. The completed
main serving family setup took104.642s maximum per worker, versus44.683s in R6.
Its sampled42.207GiB process RSS sum and112.292GiB total VRAM are **incomplete
failed-run observations**: the final controller/helper family did not complete,
so they do not establish the candidate's retained-memory cost or throughput.
The unsealed successor packet is /tmp/qfn-rocm-tp4-mtp-exact-rows-r9-private.

The focused CUDA2→ROCm2 initial3/cap15 regression reproduces the same missing
depth4 publication in9.35s, before the production fix. Evidence:
/tmp/qfn-shallow-initial-red-r1/result.json and junit.xml. Twelve registered
preflight cases cover initial1/3/7, both backend orders and TP widths1/2; they
assert actual initial draft work and stable helper graphs across reset/replay.

## Cold helper regression and continuation diagnosis

The no-launch helper fix passes all12 stochastic shallow-first cases and the
existing46 native checks (12 eight-GPU cases unavailable). The Unit run passed
782/783 suites; two assertions inside the source-policy suite assumed every
capture executes a read. Their corrected cold/live distinction passes its explicit
preflight case. Evidence: /tmp/qfn-exact-hosted-cold-gate-r1/result.json and
/tmp/qfn-cold-helper-policy-greedy-r1/result.json. The latter remains red: allfour
new greedy initial1 cases expose a separate continuation mismatch after budgets
65→1→17. ContinueResponse retains the device learner at depth6, but the old host
bootstrap uses initial1 and an M2 verifier, causing InvalidDepthSelector. The
previous maximum-width graph concealed it. Fix through the existing native
admission ticket0 and retained complete transaction; preserve learner state and
strict transaction1+ authentication. No valid fresh throughput result exists.

Review also found weight publication could replace original sources after graph
binding. The applied guard derives closure from existing graph/cache ownership
and rejects before callbacks, mutation or source destruction; no payload hashes
or duplicate lifecycle state. Its33 device-free ownership/device cases and startup
positive are registered in V2_Integration_WeightPublicationGraphLifetime. The
fresh Unit build/gate is /tmp/qfn-cold-helper-lifecycle-unit-r1.

That first build receipt remains failed: the new device-free fixture assigned an
optional device to a required device field. The corrected fixture passes the
fresh `/tmp/qfn-cold-helper-lifecycle-unit-r2/result.json`: all783 Unit tests and
both explicit cold-capture and weight-lifetime preflight checks. Source identity
is unchanged across that build and validation.

The continuation repair keeps the learned depth on device. ContinueResponse
already initializes a genuine native scheduler ticket at transaction0; a typed
start consumes that ticket without pretending a prior transaction completed.
The first retained branch then produces transaction1 under the existing strict
ordinal checks. The Runner uses the common terminal ledger and subtracts no
host drafts for this entry. Regression evidence must show no external drafting,
one native ticket0 per tail participant, preserved learned depth, exact serial
tokens and unchanged captures across response continuation and request reset.
This combined implementation is not yet qualified or benchmarked.

## Exact-row continuation focused qualification

R1 compilation passes, but all 16 cold cases reject reuse-only family validation.
The accepted-state graph exporter takes a requested physical width yet reads the
last setup-selected cache. Original materialization had hidden this dependency
by selecting each width before export. The const exporter now looks up the
requested program/width directly and authenticates its stored signature and
Params; it does not mutate selection or materialize a child. R1's separate Unit
run passes 782/783; its sole failing source assertion expected authentication
inline rather than in the new typed cursor. Both failed receipts remain intact.

Fresh R2 passes **783 Unit + 16 cold + 51 focused native/source regressions**;
12 registered eight-GPU cases skip for unavailable CUDA cards. Build 194.14s,
cold 75.71s, Unit 120.87s, focused 240.27s. All source-before/after identities match.
The cold matrix covers initial depths 1/3/7, both vendor orders and TP widths 1/2,
with additional greedy initial 1 cases. It proves actual adapted-depth ticket 0
entry, no external continuation drafts, exact serial tokens and stable retained
captures across response boundaries and reset. The cursor tests reject malformed
identity/ordinals and fake completed-admission tickets; sparse sequence tests
prove symmetric start without predecessor retirement or invented progress.

Evidence: `/tmp/qfn-exact-hosted-continuation-gate-r2/result.json` and phase JUnit
files. Prior red receipts are `/tmp/qfn-exact-hosted-continuation-gate-r1/result.json`
and `/tmp/qfn-exact-hosted-continuation-unit-r1/result.json`. This is a focused
qualification, not a new complete-preflight or real-model certificate. The R9
Release build subsequently passed and the one-shot packet was sealed before the
fresh graph-cost and throughput measurements below.

## Valid exact-row Release R9 measurement

The frozen R9 packet is `/tmp/qfn-rocm-tp4-mtp-exact-rows-r9-private`.
It retains the original 512-token synthetic prompt, 256 emitted tokens (255
after-prefill denominator), one warmup, three measurements, cap15 and the same
TP4/runtime/CPU contract. No source changed across build and native windows.

| Arm | Median prefill tok/s | Median decode tok/s | Decode sample range |
|---|---:|---:|---:|
| Default dynamic, exact rows |789.163|89.155|89.153–89.347|
| Fixed3, exact rows control |789.224|90.109|89.836–90.200|

Default dynamic gains43.09% over R6's62.305 D with prefill unchanged. Fixed3 is
unchanged against its90.161 D prior control. Dynamic remains1.06% below the fresh
fixed3 control; both remain below Castagna's91.038 D median. No goal win is claimed.
Every measured request drafts and accepts192 tokens in64 verifiers/256 logical
rows, without rejections. Dynamic stays at depth3 through12 windows across the
three requests. Both arms exactly match all3×256 own-Off token IDs and repeats.
The repeated-token prompt isolates execution cost, not natural coding quality.

Against R6 default dynamic, sampled process-tree RSS sums rise40,923,500,544 to
45,258,047,488 bytes (+4.04GiB; shared pages may be counted more than once).
Total sampled VRAM across four GPUs rises122,310,127,616 to122,709,233,664 bytes
(+0.372GiB, about95MiB/card). Sampling can miss short peaks. Maximum overlapping
worker serving-family setup rises44.683→105.945s; this is not launch-to-ready wall
time and worker timings must not be summed. The following control's pre-load
sample observes43,466,752 VRAM bytes across four cards, confirming return to
baseline after retirement, without establishing immediate-release latency.

Both canonical and independent quiet observers pass: no foreign native overlap,
source/runtime identities unchanged, clean driver window, normal exit and no
surviving owners. Result roots are `mtp-tp4-llaminar-default-dynamic-exact-r9-r1`
and `mtp-tp4-llaminar-fixed3-exact-r9-r1` under
`parity-results/qwen-flash-next`, with corresponding `-quiet` siblings.
Cost receipts: `/tmp/qfn-r9-exact-{dynamic,fixed3}-cost-analysis-r1.json`.
Token receipts: `/tmp/qfn-r9-exact-dynamic-token-resource-proof-r1.json` and
`/tmp/qfn-r9-exact-fixed3-token-proof-r1.json`.

Canonical timing disables/resets PerfStats; it cannot prove actual exact-row
replay tags or both setup outcomes. The analyzer explicitly reports these as
unavailable. Native focused cases prove the family, but separate real-model
diagnostic evidence is still required. Full current-source preflight, stochastic
HTTP/prefix qualification and the full coding session remain outstanding.

## R10/R11 completed fixed-depth sweep and retraining boundary

The source-frozen Release sweep completed fixed4/5/6 on both engines. Llaminar
keeps capacity15 and exact rows5/6/7. Castagna was measured both with n_max15
and with its native depth-sized capacity. The latter is the performance target:
its capacity15 reservation measurably slowed this workload.

| Depth | Castagna cap15 P/D | Castagna native cap P/D | Llaminar cap15 P/D |
|---|---:|---:|---:|
|4|409.210 /85.732|432.679 /92.219|789.051 /96.824|
|5|414.941 /88.076|431.595 /93.760|788.988 /108.426|
|6|408.864 /92.820|424.115 /100.210|788.542 /86.007|

All arms passed native, quiet-window, driver and normal-retirement checks. Every
measured output equals the own-engine Off control,3x256 IDs each. L fixed4/5/6
accepted204/208,213/215 and206/300 drafted tokens respectively. The reported
accepted/(accepted+rejected) probability excludes post-rejection unused drafts;
training instead uses accepted/drafted (98.08%,99.07%,68.67%). The approximate
decode wall cost per verifier transaction is50.65/54.69/59.30ms; this division
is a workload average, not an isolated kernel or graph timing.

L fixed5 exceeds the same-depth native Castagna baseline by15.64% and the best
measured fork depth6 by8.20%. This synthetic repeating-token result is an
execution-cost control. Default Dynamic still holds3,89.155D, and has not been
retrained or promoted. Natural-prompt training/holdout collection and full
OpenCode remain open.

Proofs: `/tmp/qfn-fixed456-castagna-baselines-r1.json` and
`/tmp/qfn-fixed456-llaminar-final-proof-r1.json`; their named native/quiet result
paths retain exact settings, samples and tokens. The first C fixed5-cap15 r1
failed a port bind before native launch and remains failed; a fresh-port r2
passed. No failure was restamped.

Frozen retraining input cohort: `/tmp/qfn-mtp-depth-corpus-20261027-r1/corpus.json`,
five training prompts and two held-out prompts, separate greedy/stochastic
sessions, fixed1-15 plus Off and installed Dynamic. No measurements existed
when these inputs were frozen. Training will use the canonical benchmark with
one admitted runtime per sampling cohort and strict per-request source/policy
evidence. The existing coarse trainer would replace all ROCm/MoE rules; scoped
economics and ordered-rule validation must land before any new default.
