# Ordinary cross-vendor ngram panel result

The four focused Release configurations pass the required short and held-out
frozen-history groups. Each ran all 25 histories (178 scored rows), byte-exact
reset replay, authenticated production captures and publication, clean driver
checks and original-source retirement. These are singleton GPU continuations
with eight GPU experts and the remaining experts on CPU; the opposite vendor
owns storage, not model computation. MTP execution is Off with retained
capacity15, and expert placement is static.

| Compute / storage | Row cache | Short mean KL prefill / decode | Short top1 prefill / decode | Short top5 prefill / decode |
|---|---|---:|---:|---:|
| CUDA / ROCm | C0 | .023410 / .020687 | .8125 / .979167 | .9375 / .91875 |
| CUDA / ROCm | C64 | .023410 / .020687 | .8125 / .979167 | .9375 / .91875 |
| ROCm / CUDA | C0 | .028934 / .040238 | .8125 / .947917 | .9625 / .927083 |
| ROCm / CUDA | C64 | .028934 / .040238 | .8125 / .947917 | .9625 / .927083 |

C0 disables row caching. C64 declares 64 rows and 1 MiB per GPU plus a bounded
64-row RAM tier over the original SSD source. All other workload, arithmetic
and model inputs are identical within each backend pair.

## Numerical policy decision

The original ROCm results failed only the short-decode single-row guard of
1.0 nats: `original-nine:decode:3` was 1.019896 and `:decode:5` was 1.133525.
Mean, per-offset, tail, total-variation and agreement limits passed, as did all
prefill and held-out groups. The user explicitly authorized a 20% increase.
The revised required decode row guard is **1.20 nats** for these four panels;
all other limits remain unchanged. No logits, histories, precision, weights or
native execution changed, and the original red reports remain available.

The canonical `model_kld_corpus_v2.summarize` re-evaluated the retained complete
observations against the revised policies. This is a policy reassessment,
not a native rerun or an accuracy improvement. The qsa-threshold group remains
diagnostic. No full-model/image certificate or active MTP qualification follows
from this focused result.

## Defect fixed before these results

Standalone prefill legitimately starts with generation `Ok=0, Error=None`
before response admission. The pooled source path incorrectly treated `Ok=0`
as cancellation. It now gates on the canonical sticky first-error field;
terminal completion retains its existing stronger condition. Failed read
receipts publish the first storage error and cannot erase an existing error.
Actual production-body replay covers cold prefill, response admission, genuine
cancellation and reset; the old code fails the same replay.

## Evidence and next gate

- Native observations: `/tmp/qfn-read-ahead-root-integration-r1/focused-panels-r73/`.
- Revised policies and canonical reassessment:
  `/tmp/qfn-read-ahead-root-integration-r1/panel-decode-guard-r74/review-and-results.json`.
- Original reviewed corpus and policies:
  `/tmp/qfn-opposite-vendor-panel-parser-r70/`.

Proceed immediately to active fixed1, fixed3 and Dynamic MTP using the retained
production condition/verifier families, with exact serial/grouped equivalence,
live communication extents and repeated reset. Complete preflight, dynamic
expert movement and matched performance are still outstanding.

## Active MTP validation work

The agreed General-graph host allowance is installed in the original PMA;
all 18 selected host/registration checks passed. Four native checks then exposed
two fixture defects: sequential suites attempted to adopt a second authority in
one process, and the cold-prefill trace requested rows absent from its sparse
oracle. The joined fixes retain one original authority for the full declared
population, reject unauthored inputs before publication, and retire through the
existing captured reset/terminal join while preserving the primary exception.
The joined Release/Integration builds and all 31 focused regressions now pass,
including both cross-vendor read directions, cold prefill, authority lifetime,
and observed verifier-row identity. The original failures are in `focused-r77/`;
the successful native and host checks are in `focused-r80/`.

The first Release HTTP Off control failed before readiness because startup
topology evidence used the GPU-primary rank NUMA field for a separately admitted
CPU expert endpoint. The fix projects the frozen CPU endpoint addresses while
retaining the existing physical inventory validation. It does not change
planning or execution. The failed lifecycle and clean driver window remain in
`active-mtp-http-r77/00-off/` under the same evidence root.

The native MTP regression now records the condition token from the existing
committed verifier observation and replays every actual condition/draft row
against serial execution from the same restored state. Integer ngram identities
stay INT64, including values outside floating-point mantissas. This functional
byte proof has an explicit scope separate from frozen-history numerical
acceptance and retains actual acceptance/depth/reset obligations. Its first
runtime attempt stopped before model load because the complete topology name
exceeded the filesystem component limit. The central artifact writer and its
campaign mirror now retain long identities in lossless bounded directory
components. Both explicit regressions, frozen-history replay and the complete
127-case campaign unit suite pass through canonical CTest (`focused-r85/`).
The next native attempt exposed a derived-setup reference-pack read despite
the explicit native-only purpose. The corrected prompt-binding operation keeps
declared native inputs and preserves numerical reference requirements; its
actual owner replay passes, and native validation remains pending. Neither
setup failure loaded the model or produced a byte-equivalence result.

The CUDA+CPU/C0 Off HTTP control is fully green in
`active-mtp-http-r84/00-off/`: four 384-token requests, exact repeated outputs,
full/partial prefix probes, production capture and transfer evidence, clean
exit, zero residual GPU VRAM, and a clean driver window. Its 486,066,770 bytes
of disposable prefix-cache files were removed after the process group exited;
no cache payload was read or hashed for cleanup.

An earlier otherwise successful control exposed an audit defect: HTTP resets
the measurement window after startup, so current hosted-transaction evidence
must not require a cold graph-materialization counter from a previous window.
The auditor now authenticates the same owner's captured heterogeneous replay
and balanced, exact-ABI immutable tickets, branches, fragments and terminal
events. The original failing artifact passes unchanged with the fix; all 157
audit unit cases and the explicit production-preflight regression pass. It
does not permit mutable GPU state downloads or homogeneous CUDA host dispatch.

All four CUDA+CPU/C0 HTTP modes now pass in `active-mtp-http-r84/`:

| MTP | Completed requests / tokens | Accepted drafts per request | Depth updates per request |
|---|---:|---:|---:|
| Off | 4 / 1,536 | 0 | 0 |
| Fixed1 | 4 / 1,536 | 147–157 | 0 |
| Fixed3 | 4 / 1,536 | 196–221 | 0 |
| Dynamic | 4 / 1,536 | 182–200 | 143–151 |

Every active response matches its corresponding Off token IDs exactly; full
and partial prefix restoration retain predictor state. All four lifecycles
pass capture/transfer checks, clean exit, complete VRAM release and driver
checks. The dynamic controller starts from the declared full capacity and
settles at depth3 in these requests. All disposable prefix cache backing was
removed after exit (1,947,118,336 bytes across the four runs).

The reverse ROCm+CPU/C0 family is also green, with the same four 384-token
requests per mode and exact active-to-Off token comparisons:

| MTP | Accepted drafts per request | Depth updates per request |
|---|---:|---:|
| Off | 0 | 0 |
| Fixed1 | 152–163 | 0 |
| Fixed3 | 213–220 | 0 |
| Dynamic | 188–205 | 134–147 |

CUDA+CPU/C64 Off, Fixed1, Fixed3 and Dynamic also pass. Together these twelve
lifecycles complete 18,432 tokens and remove 5,841,354,980 bytes of owned cache
backing after exit. The first CUDA C64 Fixed1 attempt missed its unchanged
60-second readiness bound while a full-parallel test build was active. It never
reached inference and had a clean driver window. Its quiet rerun passes in
190.636 seconds (`active-mtp-http-cached-retry-r90/05-depth_1/`); the original
failure remains recorded. CUDA C64 Fixed3 and Dynamic pass in 202.648 and
215.990 seconds. These elapsed lifecycle times include setup and checks and
are not throughput benchmarks.

Native byte replay remains pending. Its existing same-model metadata and NPY
headers provide snapshot sizing, without importing numerical payloads or
teacher tokens. Raw indexer queries are distinct native-only checkpoints:
their exact payload now comes from the existing source-owned QSA geometry in
the same PMA contribution, rather than requiring a nonexistent reference
header. Five registered capacity regressions pass (`focused-capacity-r90/`),
including multiple request/arena owners and no double charging when a raw
header is present. Further fixture checks require all declared eligible decode
checkpoints so an omitted ngram observation cannot silently shrink byte proof.

The native Fixed1 case now reaches real grouped verification. It exposed an
incomplete accepted-state publication input contract: PLE's retained reads
were absent from the outer stage, so their producer events were not joined to
the exact capture stream. QSA's two retained inputs have the same contract
gap. The fix forwards those original inputs through the existing graph and
TransferEngine machinery. Cold publication capture also moves before acquiring
the reusable logical-state writer, preventing prelaunch failure from stranding
an unfinished writer. Focused old-code controls fail; the coherent r93 build
and native rerun were completed in r93. Original failure:
`native-exact-fixed1-r92/`.

In `native-exact-fixed1-r93/`, publication captures and executes successfully.
All 302 actual input-row comparisons are byte exact, alongside 150 serial
condition-row comparisons and the typed INT64 hash comparison. The real
condition is token13; the predictor proposes561, the main model rejects it,
and both serial and grouped execution emit13,271. The aggregate test remains
red because it assumes checkpoint0 must accept, the source-counter validator
expects8 terminal bytes instead of the actual24 (close plus worker retirement),
and an inspection helper exports then discards a model-context reuse handle,
preventing final source retirement. These are separately tracked fixture fixes;
no accepting checkpoint or complete native qualification is claimed yet.

All four ROCm C64 modes now pass: Off215.254s, Fixed1 202.471s,
Fixed3 212.433s and Dynamic218.724s. The full sixteen-cell HTTP configuration
slice is green:64 requests and24,576 committed tokens, with all owned prefix
cache files removed after process retirement. Aggregate and individual evidence:
`http-static-mtp-completed-r94.json`. This is accumulated diagnostic coverage
across focused fixes, not a single-build or image certificate.

The native fixture fixes are joined for r94: exact24-byte completed worker
terminal accounting, explicit final ownership of observation exports, and
bounded native-prefix acceptance selection preserving checkpoint0's rejection
and complete byte evidence. Native rerun and fresh complete prerequisites remain
open. These singleton-GPU plus CPU-expert tests are correctness/lifecycle evidence,
not a TP4 performance comparison or dynamic-expert-movement qualification.


The complete original-window native sweep (r94 pilot plus r95 remaining 11)
finishes with **39,864 complete grouped-row comparisons and zero mismatches**
across all 12 fixed1/fixed3/Dynamic vendor/cache cells. Every case still fails
its accepting-checkpoint assertion; none is reported as a complete gate pass.
Both driver windows are clean and 4,295,902,185 bytes of owned prefix backing
are retired without reading payloads. Evidence: `native-original-window-r95-summary.json`.

The existing same-GGUF CPU/FP32 predictor reference independently explains
those rejections. Its three depth-zero proposals are 561, 760, 846, exactly the
production proposals; their target successors are 271, 248045, 1787. The
reference contains no accepting edge. This is an invalid test assumption,
not evidence of a predictor arithmetic defect. The 19-stage predictor comparison
has been prepared but not run; matching proposal IDs alone is not a complete
predictor numerical certificate. Bounded evidence and recipe:
`/tmp/qfn-predictor-checkpoint-diagnostic-r95/`.

The r96 fixture therefore declares a separate, fixed 32-prefix native functional
history using the same prompt, greedy sampling and production M1 continuation.
The original checkpoint and every attempted complete row bank remain mandatory;
positive acceptance still must occur, and exhaustion fails without extending the
window. The model's numerical two-step horizon and all frozen-panel thresholds
remain unchanged. Unit replay covers the short rejecting window, later acceptance
inside the fixed functional window, and out-of-window rejection. Static native
qualification is pending this rerun. Separate Dynamic expert families are also
registered: 32 policies (Ordinal/Random, each vendor/cache pair, Off/1/3/Dynamic
MTP), with 24 active native views. Their actual movement, economy and numerical
obligations remain unqualified.


The r96 CUDA/C0 Fixed1 pilot passes the complete native proof and final owner
retirement in 68.61 seconds. Prefixes 0–2 retain the expected rejections; prefix 3
accepts token 198 after condition 1787, and emitted 1787;198 equals the native M1
oracle. The fixed functional bound was 32 before execution; it was not increased
after this result. All 7 affected Unit/preflight entries pass, and inventory
refresh preserves all 5,295 prior entries while adding 91 dynamic-expert/test
registrations. The remaining 11 static native cases are running. Evidence:
`native-exact-fixed1-r96/`, `focused-functional-window-r96/`, and
`ctest-inventory-delta-r96.json`.


The complete r96 native slice finishes with all eight Fixed1/Fixed3 cases green,
all 12 cases observing actual acceptance, and 73,084 complete grouped-row
comparisons byte-exact. The four Dynamic cases stop at one test assertion:
the 17-token acceptance response contains multiple transactions, but the helper
compares its last observed depth 14 with its initial depth 15. The earlier
isolated accepted depth 15 transaction already passed its exact live-row proof.
Drivers are clean; 7,875,733,898 bytes of owned prefix backing are retired.
Evidence: `native-functional-window-r96-summary.json`.

The r97 test correction preserves that 17-token response, positive acceptance,
serial equality and both continuation/state comparisons. It validates the
original isolated full-depth identity separately from the last transaction's
own depth/live extent and admitted capacity, including controllers that grow
from an initial depth below capacity. The original equality fails its focused
replay; the corrected typed validator passes. Only the four affected Dynamic
native cells require rerunning; no production arithmetic or execution policy
changes in this correction. Complete Unit/preflight remains pending.

The r97 coherent build and four affected Unit/preflight entries pass. All four
Dynamic native cases now pass depth attribution and the 17-token response plus
two continuation-token comparisons, then fail the full terminal-state comparison.
CUDA reports candidate position 30 versus serial 31; ROCm reports 33 versus 34.
The matching previous serial hidden row suggests an emitted-but-pending token,
but terminal-logit hashes also differ at that earlier position. Neither selecting
the previous row nor forcing the last emitted token is a valid fix without
authenticating the existing commit and publication contract. The full state
assertion remains fatal. Evidence: `native-exact-dynamic-r97/`; all four cases ran,
the driver window is clean, and 2,625,244,628 bytes of owned prefix backing were
retired without payload reads.

A focused same-committed-position diagnostic is prepared for the original CUDA
C0 Dynamic registration. It compares already-retained full-state observations
and complete raw logit rows, without another forward, token substitution,
numerical relaxation or change to the original response/state assertions.
The r98 CUDA C0 replay confirms stale scalar logits: at position 30 the complete
main banks and terminal hidden are byte exact, while all 248,320 logit elements
differ from the serial row. Candidate argmax is 1787; both the independent native
LM-head row and serial probe predict 760, with maximum absolute difference
13.822502613. This is a publication defect, not floating-point tolerance drift.
The original 17+2 visible tokens remain exact. Evidence and full raw rows are in
`native-exact-dynamic-cuda-c0-r98/`; the driver window is clean and 477,312,197
bytes of owned prefix backing were retired. The diagnostic preserves the original
fatal assertion. Pending-condition authentication and terminal producer repair
are separate scoped changes; penalized stochastic distribution must also retain
raw logit input rather than publishing an already-transformed row.

Two additional original Dynamic Ordinal C64 controls, MTP off, pass all four
384-token HTTP requests on each compute backend. CUDA reports 220 completed
tier-residency movement edges and 22 economy records; ROCm reports 70 edges and
11 economy records. Each uses one compute GPU plus CPU experts, with the other
vendor providing storage only. Prefix repeatability, physical movement checks,
driver windows and normal shutdown pass; VRAM returns to its original level.
Both owned disk-cache directories are removed (972,133,536 bytes total).
Evidence: `dynamic-expert-controls-r98-summary.json` and the original
`dynamic-expert-http-r96/08-off/` and `24-off/` directories. These controls use
the unchanged r94 Release runtime; they do not certify active-MTP movement,
other owner orders, composed GPU compute or a new image.

The r99 pending-condition projection is joined. It reuses the existing packed
diagnostic export and authenticates the original ready-condition handle before
allowing comparison at an earlier committed position. A missing/stale handle,
wrong token, gapped serial history or contradictory disposition fails. The
17+2 visible output checks and full state/logit comparisons remain strict.
Its isolated helper replay passes 20 iterations and all 32 metadata controls;
the changed snapshot ABI requires the forthcoming coherent core/test rebuild.

The r99 producer repair now joins the accepted terminal-hidden helper with a
raw-logit row selector. An absent accepted row preserves both prior values;
physical strides and owned vocabulary columns are explicit. Condition forwards
also copy their terminal row into the existing canonical bank, and the completed
parent publishes that bank through the original event-aware authority. Canonical
logits remain the scalar request-zero bank, matching the schema and CPU publisher;
other batched consumers keep their all-position bank. No new model buffer or
host count readback is introduced. Capture identity includes the new binding.

Both stochastic sampling stages now apply the original penalty law while reading
candidates, preserving raw logits. Their retained target graph has two kernels
(selection and normalization), replacing the destructive penalty pass plus those
two kernels. The new replay tests compare compact IDs/probabilities byte-for-byte
with the original arithmetic and independently require raw input immutability.
Row-copy tests cover the full 248,320-column vocabulary, uneven shards, request-zero
selection, padding, sentinel preservation, producer events and retained resets.
All eight new registrations are explicit preflight entries; inventory grows from
5,387 to 5,395 with no removals. Host syntax checks pass. The coherent Integration
build targets CUDA 80/86/89/90 and HIP gfx906/gfx1100 with the existing spill guard;
native execution is still pending. Package joins and build evidence are in
`join-*-r99*/`, `joined-r99-review.json`, and `integration-publication-r99/`.

The bulky temporary r98 raw-row diagnostic was removed from the ordinary fixture.
The original failing binary-row artifacts remain preserved, while subsequent
runs retain a bounded same-committed-position CSV alongside the strict full-state
comparison and existing first-difference diagnostics.

The r99 coherent build passes in 554.30 seconds with unchanged code inputs.
All eight new focused registrations pass, including CUDA and MI50 raw-logit
immutability and byte-exact sampler output. All five affected Unit suites pass.
The two additional full row-selection suites expose an older synthetic-fixture
lifecycle defect: their cold checkpoint capture has not joined the original
hidden-input upload event, and unwinding an unclosed native capture masks that
exception with a free-during-capture error. The fixtures also unnecessarily
request the generic model-download fixture despite using only synthetic tensors.
The complete focused result remains red (16 of 18 selected entries pass);
driver diagnostics are clean. Evidence: `focused-publication-r99/`.

The r99 CUDA C0 Dynamic full-model retry fails admission before inference.
Its new publisher requires a native canonical LOGITS address while the existing
cold retained-MTP allocation family eagerly materializes only its extension
banks. The original canonical roots are already declared and admitted but remain
lazy at this boundary. The correction is to materialize those same roots through
the existing arena/PMA family, before helper capture; no new reserve or late
recapture is needed. Evidence: `native-exact-dynamic-cuda-c0-r99/`; original GGUF
metadata is unchanged and driver retirement is clean. Full terminal-state
qualification therefore remains pending.

The r99 targeted gfx1100 run also passes both new native regressions (16.72
seconds), with a clean driver window. This complements the native MI50/RTX 3090
checks and the complete shipped-architecture spill build; it is not QFN model
qualification on the 7900 XTX.

The r100 follow-up build passes in 172.63 seconds, without source changes during
compilation. Both repaired full row-selection suites, both cold-checkpoint
20-repeat preflights, the canonical-root allocation-order regression and its
owning Unit suite pass. The new paired native-event benchmark also passes on
CUDA and MI50. All eight selected CTest entries execute with no model fixture,
skips or driver diagnostics. The original canonical-root source is red for both
missing root entries; the corrected source regression passes 20 repeats.

The paired Integration sampling comparison preserves identical compact outputs,
raw logits and history. At M1/M4/M16, CUDA medians are 136.192/236.544/608.768 us
for the legacy three-kernel preparation and 138.240/245.760/652.288 us for the
read-only two-kernel path. MI50 medians are 194.720/241.040/546.640 us versus
198.160/242.720/590.080 us. Component cost rises 0.7–7.95%, with at most 43.52 us
additional time; fewer launches alone did not prove a speedup. Native resource
observations report no local memory. These are Integration component timings,
not Release request-throughput evidence. Complete samples and native resource
records were preserved from the CTest log before the next test run; its normal
JUnit success-output truncation is not usable as the complete timing record.
Evidence: `focused-publication-r100/ctest/sampling-native-events.json` and
`native-events-complete.log`.

The original CUDA C0 Dynamic full-model case is now green in 88.54 seconds.
At authenticated committed position 30, every canonical logit and the complete
main KV/GDN/conv/hidden state are byte exact against serial decode. The original
17-token response and two continuation outputs remain unchanged and exact;
the original pending condition names token 760. The driver window and original
GGUF metadata checks pass; 477,312,201 bytes of owned prefix backing are retired
without reading payloads. Evidence: `native-exact-dynamic-cuda-c0-r100/` and its
bounded `mtp_acceptance_same_frontier.csv`. The other three original Dynamic
cases are now running on the same coherent build. Complete Unit/preflight and
current Release/HTTP qualification remain pending.

All four original Dynamic native cases now pass on r100: CUDA C0/C64 take
88.00/93.29 seconds and MI50 C0/C64 take 108.41/109.62 seconds. Each retains
exact full state at its authenticated committed position, the original response
and continuation checks, real acceptance and captured production-path checks.
Both driver windows and original-input checks are clean. Owned prefix cleanup
totals 2,625,244,630 bytes, with zero payload reads. The combined receipt is
`native-dynamic-r100-summary.json`. These are single compute-GPU plus CPU-expert
cells with the opposite vendor providing ngram storage; they do not certify
multi-GPU compute continuations or fresh active-MTP expert movement.

Static row-copy resource evidence is retained at
`/tmp/qfn-row-select-static-isa-r99/REVIEW.md`: CUDA uses 36 registers on
sm80/sm86/sm89 and 39 on sm90, with no stack/local/shared bytes. HIP uses
13 VGPR/42 SGPR on gfx906 and 16/41 on gfx1100, with no scratch/LDS/spills.
The emitted general 64-bit division branch is unreachable for valid live
operands under the launcher's checked extent; live operands take the guarded
32-bit path. This is static resource evidence, not a throughput measurement.

The fresh complete prerequisite attempt initially stops before building because
strict CTest command discovery rejects an executable that has not yet been built
in this selectively built tree. No Unit/preflight tests ran in that attempt.
The explicit full `v2_unit_gate` plus `v2_production_test_preflight_gate` build is
now collecting all compiler failures with Ninja `-k0`. The current inventory
contains 973 Unit and 2,849 production-preflight entries. The build-before-discovery
bootstrap ordering is being repaired separately, with original failure evidence
preserved in `full-production-preflight-r100/`. Broader qualification is still
pending; the four green model replays are not rerun merely for harness changes.

The complete prerequisite build finishes in 834.10 seconds with four failed
compile edges: three standalone read-ahead tests omit their production-header
C++20 requirement, and the ordinary immutable-host-lookup benchmark still uses
the former service-boundary admission API. The language requirements are now
target-local, and the runner builds before strictly validating runnable CTest
metadata. Installed-image receipts still authenticate before discovery, and
missing commands after a successful build remain fatal. All eight affected
CTest registrations pass in 6.88 seconds, including the complete campaign Unit
suite and the new explicit bootstrap preflight. The benchmark is being migrated
to the existing original-source fixture and retained owners. The independently
built complete 973-case Unit suite is collecting failures while that private
patch is prepared. Evidence: `full-prerequisite-build-r101/`,
`integration-bootstrap-r102/`, `focused-bootstrap-r102/`, and `full-unit-r102/`.

The first complete Unit sweep executes all 973 registrations in 136.20 seconds:
971 pass and two fail, with no skips. The source guard finds a direct environment
read in the retained CPU service; it now uses the existing `DebugEnv::envValue`
authority with unchanged closed keys and unset/empty semantics. The original
scanner reproduces the violation; the actual service has an additional identity
and stale-evidence regression. The other failure is a metadata-only planning
fixture missing its CPU-service projection. The host-lookup benchmark migration
also receives six explicit native preflights: RAM/mmap/adaptive on CUDA and ROCm,
using its existing byte/tail, terminal and original-source retirement checks.

The complete r103 prerequisite rebuild passes in 75.27 seconds with unchanged
source inputs. The canonical unfiltered gate is running 973 Unit and 2,857
production-preflight registrations. It discovers two further fixture corrections:
the exact CMake inventory expectation needs the six new native entries, and the
planning fixture also needs recorded ordering-event metadata for its authored
GPU endpoints. Both reviewed private patches preserve the production checks;
the ordering fixture additionally asserts the mandatory event contribution.
They wait until the source-frozen sweep ends. Native kernels, tolerances and
model inputs are unchanged by these pending corrections. Current evidence:
`full-prerequisite-build-r103/` and `full-production-preflight-r103/`.

The next current Release checkpoint is the existing Ordinal-expert C64 Off to
Dynamic MTP pair in each vendor direction, using freshly recorded Off controls.
The private `/tmp/qfn-next-gates-r100/HANDOFF.md` retains exact registered
identities, canonical driver commands and the remaining movement/native
admission obligations. This remains single-GPU compute plus CPU experts with
opposite-vendor storage. Flexible pooled multi-GPU compute, PP and cross-rank
continuations remain separate implementation and qualification work.

The r103 sweep completes Unit (971/973) and the host preflight lane
(1,692/1,697). Its reproducible host defects are the two fixture expectations
above, a mixed-local registration combining 120 fresh subprocess replays under
one 30-second deadline, and a scanner registration resolving bare `bash` as a
build-directory executable. Repairs preserve all twenty repetitions of each
mixed-local case, give each original case its own unchanged deadline, and use
the discovered absolute shell path. CUDA additionally exposes four original
lookup fixtures that still construct ordinary contexts instead of the required
owned worker session; their typed fixture migration is separate from hardware
qualification.

At 20:41 UTC the hardware portion is interrupted: all four MI50s return all-FF
PCI configuration data and unknown link speed, while native ROCm reports zero
devices. The driver window contains 3,495 new findings, including AMD SMC
messages returning `0xffffffff`. The exact canonical runner receives SIGINT;
both owned CTest groups and their MPI children retire before source work resumes.
No GPU reset or host configuration change is attempted. CUDA/ROCm socket lanes
and the exclusive lane are not qualified by this partial run. The wrapper's
`complete_prerequisite_gate` field describes the requested scope, not successful
completion: `interruption-summary.json` explicitly records incomplete execution.
Evidence is under `full-production-preflight-r103/`, including
`hardware-interruption.json`, `driver-report.json`, and both completed host XMLs.
Hardware-dependent continuation work awaits restoration; device-free repairs
and compilation can proceed.

The user restores the switch fan and reboots. The new boot reports valid PCI
configuration and x16 links for all four MI50s. The repaired prerequisite build
passes in 12.21 seconds. Complete Unit passes 973/973 in 135.26 seconds; all
thirteen repaired/new host-preflight registrations pass in 98.49 seconds. The
earlier r105 Unit attempt exposed another test isolation defect: dry-run
NativeVNNI fixtures still selected automatic real GPU discovery. Ordinary
fixtures now select the CPU fitter explicitly; a focused original-script
regression exercises automatic discovery through fake tools and is registered
in preflight. Production policy selection and deadlines are unchanged. The
local focused runner also uses the canonical affinity-sized parallel budget
and disables hwloc GPU discovery, matching the complete-gate environment.

Eight native lookup registrations then pass: both broad primitive suites and
RAM/mmap/adaptive native Actor replays on CUDA and ROCm. The broad suites run
each compiled case in its own process so original preparation/runtime ownership
cannot be inherited from another case. The original captured protocol, byte
oracles, retirement and worker-affinity checks remain intact. The new driver
window is clean. Evidence: `repairs-build-r106/`, `full-unit-r106/`,
`focused-host-r106/`, and `native-lookup-r106/`.

The remaining canonical CUDA/ROCm socket and exclusive lanes are now running,
using the existing lane planner, exact CTest index validation, disjoint package
binding and native process-retirement authority. This is deliberately recorded
as partial-gate continuation evidence under `hardware-preflight-r106/`; it does
not manufacture an unfiltered prerequisite receipt from earlier runs. The
completed host sweep and its repaired regressions remain separately identified.


The r106 CUDA socket lane completes 277 registrations: 268 pass and nine fail,
without skips. Failures cluster into four areas: opt-in sparse profiler cases
registered in ordinary functional sweeps; opposite-backend context cases plus
retirement endpoint validation; the model-free numerical graph inheriting the
transformer's immutable layer-weight hook; and retained CPU-ticket fixtures
missing their original admitted worker. The same fixture errors also reproduce
in the still-running ROCm lane. Two ROCm eight-device retirement checks are
explicitly skipped for insufficient hardware and remain unqualified. The
exclusive lane has not yet run. These are software results after recovered
PCIe visibility, not a continuation of the prior all-FF hardware failure.

Small repairs remain private while the current source/binaries are frozen.
`/tmp/qfn-pipeline-table-template-r106/` preserves an actual-header original
failure and corrected twenty-preparation CPU replay, both under the canonical
process owner with empty retired session directories. The corrected test graph
prepares its own original table recipe; production Qwen weight checks remain
unchanged. Actual Integration-flag syntax passes for all three consuming test
translation units. `/tmp/qfn-context-retirement-repair-r106/` holds the endpoint
validation/registration repair and four successful syntax checks; actual-body
CPU retirement replay is separate. Neither patch is yet a native pass.

The next ConsumerLocal TP2 slice is independently staged, not installed or
qualified: `/tmp/qfn-consumer-local-tp2-r106/HANDOFF.md` describes the shared CPU
source and exact per-consumer GPU/frame BOM plus sixteen positive source cases
(CUDA/ROCm, C0/C64, Off/Fixed1/Fixed3/Dynamic). Component review found a terminal
race if one channel's successful close treated an unfinished peer as failed.
Normal per-channel completion and complete-group abort now require distinct
ordering, with an additional two-phase Rank reset because its old per-DGO loop
could await before submitting the next peer's close. Delayed-peer, partial
admission and reset proofs are required before coherent native qualification.
Shared GPU-owner pooling, PP and cross-rank continuation remain separate work.


Both r106 socket lanes have completed: CUDA 268 pass / 9 fail / 0 skip;
ROCm 373 pass / 9 fail / 2 skip. The failed registration names match by backend,
and the two skips are the unavailable eight-device retirement cases. The
exclusive lane later stopped at the watchdog interruption described below. All original failure XML/stdout
is retained under `hardware-preflight-r106/`; no complete receipt is claimed.
The endpoint repair additionally has actual patched-body 20/20 green and
original-body 20/20 red CPU-only evidence at
`/tmp/qfn-context-retirement-replay-r106/results.json`. The numerical-table
fixture repair likewise retains its actual original failure. Small repairs
remain unjoined in `preflight-repair-batch-r107/packages.json`.

### Watchdog interruption, 2026-11-01 around 21:59 UTC

The user confirmed a system hang followed by a hardware watchdog reboot. The
old test owner and continuous kernel observer are gone. Before interruption,
the exclusive lane completed 117 of its 499 registrations, all passing; its
last started registration was `V2_Integration_MoEProjectionSharedColumns_ROCm4`.
This identifies the interrupted workload, not its cause. The already completed
Unit 973/973 result and socket-lane XML remain separate evidence; there is no
complete preflight result for this attempt.

The observer retained 725 complete kernel records and a truncated/corrupt tail.
The previous boot's persistent host kernel journal was also recovered through
a read-only mount. The first retained errors are two KIQ fence timeouts on
MI50 `0000:91:00.0`, followed by repeated DMAR invalidation timeouts. The kernel
then explicitly reports a hard lockup on CPU11 in `qi_check_fault`; its stack
includes `qi_submit_sync`, `cache_tag_flush_all`, and `intel_flush_iotlb_all`.
Other CPUs subsequently report stalls. This establishes a kernel lockup in
the invalidation path, but does not distinguish the initial GPU, PCIe or driver
cause. The earlier fan-off incident is a separate event.

Evidence lives in
`/tmp/qfn-read-ahead-root-integration-r1/watchdog-recovery-r107/`, including
`interruption.json`, `previous-kernel-root.log` and a readable extraction of the
original observer records. The canonical observer finalizer fails closed with
`Driver collector retired without complete evidence`. The raw journal remains
unchanged. All seven GPU PCI configuration headers are readable after reboot;
that is not a native runtime qualification. GPU runs remain paused while
CPU-only repairs and coherent compilation continue; no host configuration or
driver changes were made.

The user subsequently authorized GPU resumption. The same interrupted
`MoEProjectionSharedColumns_ROCm4` registration passes in isolation in 64.42s,
with a complete clean driver window in `interrupted-native-r109/`. This is a
fresh scoped result, not an explanation of the earlier crash or a complete
preflight receipt.

The five reviewed small preflight repairs are now joined. Reconfiguration
initially rejected the local paired model metadata because Linux renumbered the
same model filesystem from device259:1 to259:2. Both boot journals name UUID
`6d6cc0c5-1973-4f9c-b760-2497112b8e0a`; all five files preserve their path,
inode, size and nanosecond modification/change timestamps. A separate local
metadata generation at `reboot-model-binding-r108/` updates only that device
field and dependent small-document identities. Its 25 numerical payloads are
hardlinks to the original inodes; histories, reference runtime, arithmetic,
topologies and numerical limits are unchanged. Neither model nor reference
payload bytes were read or hashed, and historical results were not relabelled.
Both the canonical corpus loader and local binding freezer validate the new
generation. The production six-stat replacement guard is unchanged.

`repairs-build-r109/` completes the coherent Unit/preflight prerequisite build
for CUDA80/86/89/90 and HIP906/1100. `full-unit-and-host-r109/` passes all973
Unit registrations plus both explicit context-provenance/retirement endpoint
replays:975/975, no skips or missing cases,134.47s. All20 affected native
registrations pass in317.43s under `native-repairs-r109/`, with a complete clean
driver window. The source remained unchanged throughout those checks. Complete
preflight, grouped-consumer native admission and fresh movement/serving
performance remain outstanding.

### Grouped ConsumerLocal integration

Nine reviewed packages are now joined, with per-package receipts at
`joined-consumer-feature-r110-1/` through `-9/`. The coherent core, complete
Unit/preflight targets and actual QFN matrix are rebuilding under
`consumer-feature-build-r110/`; no old-core ABI execution qualifies these changes.
One CPU provider/cache/worker now serves exact separate GPU consumer frames.
Generic layout/admission metadata covers1/2/4/8 participants, while native
topology selection initially admits homogeneous full-main ConsumerLocal TP2.
The actual16 source cells cover CUDA/ROCm, C0/C64 and Off/fixed1/fixed3/Dynamic.

The same canonical definitions also declare48 execution obligations and36
active native-exact views across static, dynamic-ordinal and dynamic-random
expert policies. These declarations are unqualified. Dynamic policies require
physical movement and economy evidence; active MTP keeps matched Off controls.
New TP2 frozen-history panel bindings are being prepared from the independent
reference, preserving its25 histories,178 rows, arithmetic and numerical limits.
Historical singleton candidate results are not reused as TP2 results.

Review fixed three concrete grouped-service failure paths: a missing close
publishes cancellation before waiting behind another terminal join; joined
failure diagnostics remain readable while peers still hold retirement rights;
and Rank attempts every original participant retirement and final group finish,
preserving the first exception. Model-data reset starts only after successful
native close and service retirement. Focused original-body/component evidence
is retained with each package; coherent linked and native validation is next.

The complete project still includes shared-owner multi-consumer topology,
composed/mixed-vendor compute, dynamic balancing, all selected MTP modes and
matched Release performance. ConsumerLocal replication does not finish pooled
sharding. A separate direct-peer payload candidate remains private and
unpromoted. The local NVIDIA topology query reports chipset-unsupported peer
reads/writes in both3090 directions, so current-hardware support also needs the
explicit NoPeer mapped transport using exact transient live packets.

Grouped-service review adds a second concrete failure interleaving: a participant
missing its own close must cancel before waiting behind another participant's
terminal join mutex. The separately frozen successor
`/tmp/qfn-grouped-source-terminal-failure-r107/` proves the actual earlier object
fails the bounded regression and the corrected body passes twenty repetitions;
the complete component suite passes 28/28. That component evidence did not qualify the new caller ABI. The caller and
generic 1/2/4/8 metadata packages have since joined the coherent r110 build
described above; native and model qualification remain pending. Channel matching
uses original identity when authored TP order reverses physical ordinals,
without changing compute ordering.


### Resumed GPU admission and original declaration propagation

The coherent ConsumerLocal build r110 passed. Complete Unit plus the focused
source/lifecycle selection executed all 994 registrations: 989 initially passed.
The five failed registrations reduced to a renamed terminal-result helper in
source checks and incomplete synthetic TP assignments. The joined repairs use
the canonical TP assignment producer; the production planner checks remain
intact. The original five failures and synchronization self-test pass in the
six-case r111 rerun. Build evidence confirms only those three test files changed.
This is a repaired scoped result, not complete production preflight.

Full source discovery then found six existing opposite-vendor active case IDs
of 258 or 264 bytes. The source writer and reader still assumed one filename,
although numerical parity already supported bounded identity components. The
repair shares that existing lossless encoding between both workflows, retains
short paths and complete logical identities, and cleans only owned nested cache
leaves. Explicit long-identity and cleanup regressions are registered in preflight.
The recorded 714-case discovery now selects 564 hardware-eligible cases and
leaves 150 explicitly unqualified in device-free replay.

The focused ConsumerLocal native run r112 executed all 16 declared cases in
613.81 seconds. CUDA and ROCm, both cache capacities, and all four MTP modes
failed at the same initialization boundary: the Rank configuration had dropped
the original retained CPU-service declaration before constructing child runners.
All runner/native resources retired; the complete driver window contains 55
records and zero findings. These failures occurred before inference or active
MTP execution and provide no numerical or performance qualification.

The joined fix carries the same declaration through Runtime to Rank and its TP
and PP child configurations. It introduces no second service owner or capacity
calculation. The actual Rank projection and ModelContext/PMA token regression
passes twenty repetitions, and the original omission fails its control. This
fix and the artifact-path repair rebuilt together under
`consumer-native-repair-build-r113/` in 273.89 seconds, with the full prerequisite
target closure and QFN matrix rebuilt coherently. Complete Unit973 plus the
24 focused regressions pass 997/997 without skips or missing cases in 168.46
seconds (`unit-and-consumer-focused-r113/`). Source stayed unchanged through
both operations. This is not a complete ProductionTestPreflight result.

All 16 native source cells pass in `consumer-local-source-r114/` (1229.05 seconds),
using full 714-case canonical discovery and hardware selection before projecting
the declared TP2 slice. Both CUDA and ROCm, C0/C64, and Off/fixed1/fixed3/Dynamic
admit successfully. All runner/native resources retire; the complete driver
window has 52 new records and zero findings. This qualifies the declared admission
slice, not the complete source matrix or inference.

The first of four fresh numerical panels loaded and prepared CUDA TP2, then
stopped before inference at an old observer restriction: only four-card ROCm
LocalTP was accepted. `consumer-local-panels-r114/` retains this failure with
clean native retirement and a clean driver window. No TP2 KL result exists yet.
The observer must derive participant counts and graph proof from the actual
admitted GPU and expert owners, retaining full-GPU capture requirements and
explicit CPU expert boundaries. Neither reference outputs, histories nor limits
are being changed. The prepared panels retain 25 histories and 178 rows each.

The Release application and existing pooled/host lookup performance targets
also rebuilt coherently in `consumer-release-build-r114/` (518.56 seconds).
This build is not a performance measurement. The separate NoPeer transport and
new benchmark/tooling packages remain private pending focused integration.
