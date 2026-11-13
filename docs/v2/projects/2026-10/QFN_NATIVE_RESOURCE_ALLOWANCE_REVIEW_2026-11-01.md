# Focused native resource allowance review

This review follows the user's 2026-11-01 scope correction: exact owned storage
stays with PhysicalMemoryAuthority; opaque runtime storage uses conservative,
typed, tested allowances. It does not require a complete private allocator
census. The immediate purpose is to unblock real QFN continuation tests while
preserving actual ownership, lifecycle and capacity checks.

## Admission policy correction

The user explicitly rejected application-source hashes as admission keys.
Conservative native resource limits must remain usable across ordinary graph,
planner and kernel edits. The follow-up correction also rejects development
machine fingerprints and observed test populations as production eligibility
limits. Select opaque allowances by supported backend and typed resource family;
the configured loader owns native ABI compatibility. Exact buffers, stacks and
current compiled objects remain individually accounted by
PhysicalMemoryAuthority.

Remove application-source fingerprint generation and matching from the GPU
context, auxiliary resource and CPU service admission paths. A source revision
may identify test evidence; it must not decide whether an unchanged resource
family can be admitted. Actual owners, devices, generations, pointers and events
continue to establish live ownership. Changes to resource consumption receive
focused regression and limit testing; routine graph edits do not require another
native memory qualification campaign or a manual compatibility-version bump.

Do not compare live opaque allocation samples against a measured byte peak or
reject a request because it exceeds that peak by a few bytes. Observations inform
rounded planning allowances with substantial headroom. PMA still owns every
declared instance and rejects actual capacity shortages or undeclared creation.
There is no runtime repricing or parallel live ledger.

The coordinated follow-up uses unchanged GPU context/event units, a remote-graph
host unit of 1 GiB per started 1,024 declared native nodes, and a CPU service unit
of `max(2 MiB, round_up(declared_TLS_bytes + 1 MiB, 2 MiB))`. CPU wire pricing
carries the actual declaration's stack, guard, page and object geometry; it does
not recover historical `sizeof` values from allowance rows. Empirical worker,
module, graph-instance and graph-shape maxima are coverage evidence, not hard
support limits. Existing native graph structure and ownership checks remain.

The source-hash refreshes and historical population limits described below are
evidence of the problem, not a procedure to repeat. The resource-family policy
and current-geometry wire are now implemented. Their focused validation is
green; real-model continuation admission has separate failures described below.

## Current validation boundary

The r56 Release and Integration builds pass, retaining CUDA 80/86/89/90 and
ROCm gfx906/gfx1100 compilation. All 50 selected admission/lifecycle/reporting
registrations are green after one fixture-only correction in r57. Actual
installed CPU consumers pass in Release; original CUDA/ROCm source lifecycles
pass with clean driver checks. This is focused validation, not complete Unit,
preflight, numerical parity or model-performance qualification.

The first four opposite-vendor ordinary retained15 source cases all execute
and fail during graph construction: the full-model planner omitted the existing
directed-transport BOM. The first exact CPU staging allocation is 4 KiB for C0
and 8 KiB for C64; admission declared zero. The common retained-participant
budget now includes the existing channel BOM for mandatory, candidate-growth,
resolved and restored plans. Exact claims and wire equivalence pass focused
regressions, and all four r57 model cases pass this former failure point.

Failure cleanup exposed a second defect: admitted prepared GPU births survived
initialization failure because runner retirement is conditional on inference
being ready. Cleanup now uses actual original births, including storage-only
GPUs, after ordinary model/graph owners release their storage. Focused replays
exercise actual runner shutdown and preserve borrowed-source ownership.
The early runner retirement CSV does not prove final process retirement;
the r55 aggregate's `all_resources_retired:true` is invalidated by the subsequent
terminal abort. Schema 2 now requires a separate invocation-bound receipt after
the actual matrix entrypoint's GPU lifetime retires. All four r57 failure paths
provide both receipts, without a terminal abort; driver checks stay clean.

The flat remote graph preparation defect exposed by r57 is corrected in r58.
Preparation now delegates through the existing backend contract after original
PMA/worker/stream checks. CUDA still rejects stack growth on source-prepared
contexts before any legacy probe or mutation; HIP has no instance-time stack
expansion. Release and Integration builds and six focused registrations pass,
including repeated exact-owner and native-failure regressions. No additional
allowance or estimator was needed.

All four r58 model cases pass that boundary and expose a cold calling-context
defect: the executor creates its capture stream through the original worker,
then performs preparation and recording-owner checks on the control thread.
Both checks now dispatch to the original worker, preserving source/control-block
and exact stream identity, with no GPU completion fence or hot replay dispatch.
The r59 Release and Integration builds and all nine focused registrations pass,
including both new cases repeated twenty times. A separate observer fix includes
storage-only GPUs in the numerical harness's passive device inventory.

All four r59 source cases pass preparation binding and remain red during capture
planning. The generic collective flag treats participant-local native ngram
transfer endpoints as manual CPU work when ExpertOverlay has CPU services. The
extra manual segments violate its existing authenticated 48-boundary lifecycle
(139 total segments for C0, 151 for C64). Correct the transport classification;
do not invent CPU cutpoints, disguise transfers as compute, or relax that
lifecycle. This is a graph-composition defect, not an opaque allowance failure.
All four failures retire both original native contexts, produce final process
receipts and leave a clean driver window. Model admission is still unqualified.

The r60 capture fix gives participant-local native transfers an explicit capture
authority while preserving their collective identity and readiness checks. The
shared planner predicate no longer mistakes those transfers for CPU services;
the existing 48 logical CPU boundaries remain unchanged. Request admission,
publisher construction and reset publication now use the original source worker.
Release and Integration builds pass. All 20 focused registrations are green
after correcting one test's borrowed-worker destruction order in r61; both
original failing receipts remain available. No opaque allowance was increased.

All four r61 opposite-vendor runners now initialize successfully. Their tests
remain red afterward: the materialization oracle incorrectly treats MTP Off as
absence of retained predictor capacity, although these four controls explicitly
retain depth15. Subsequent shutdown exposes a separate ownership defect when
the destructor repeats teardown after the context has already been sealed.
Final process retirement is therefore **not proved** in r61. The complete
172.95-second batch gathers all four failures and has a clean driver window.
Those two defects are repaired in r62–r64. The oracle now uses retained
capacity for predictor materialization while keeping execution checks separate.
Successful runner shutdown releases its borrowed preparation alias after the
existing seal, preventing the retired runner from sealing a successor's source.
The focused lifecycle tests include actual exported-context shutdown, destructor
reentry and successor reuse. The foreign-authority negative control exercises
the fatal `noexcept` boundary in a child process; it must not expect a catchable
exception. All thirteen selected r63 registrations pass using their latest
r63/r64 results, including the full final-retirement Unit executable.

The r64 native batch still fails all four cases after successful initialization
(174.93 seconds; clean driver window). The next assertion exposes a production
inventory error: `allocators()` advertises physical owners but returns GPU
consumers, omitting a declared storage-only owner. Final process cleanup also
fails: exported model-context retirement handles positive model-weight roots
but omits original storage-only native births. These are separate fixes; neither
permits lowering the owner-count assertion or relaxing the native reset guard.
The four cells and full current Unit/preflight remain unqualified. The canonical
diagnostic export also needs to retain declared storage owners and Off-mode
predictor capacity before these cells can run their numerical panels.

r65 closes those failures: the allocator view borrows actual immutable layout
owners, including zero-capacity storage-only GPUs, while consumers remain a
separate projection. Exported-context retirement retains the original native
source until model owners and positive model tickets retire, then resets its
remaining storage-only births through the existing source path. The strict
physical-reset requirement and source/PMA identity checks remain unchanged.
The declared diagnostic exporter now retains the original storage placement and
Off-mode predictor capacity; automatic production benchmark projection is
unchanged.

The coherent Release and Integration builds pass; all **25 focused registrations
pass**. All **four real-model source controls pass** in **168.75 seconds**, with
final resource retirement, zero unexecuted selected cells and a clean driver
window. This qualifies loading and cleanup for the original static-expert,
MTP-Off retained15 slice. It does not qualify model numerics, active MTP,
dynamic expert movement, throughput or the complete source matrix. The next
acceptance work is actual inference and frozen-history accuracy, followed by
matched Release economy and the active execution-policy combinations.

Evidence is under `/tmp/qfn-read-ahead-root-integration-r1/`:
`release-build-r56`, `integration-build-r56`, `integration-build-r57`,
`focused-admission-r57-summary.json`, `native-source-r57-window`,
`original-cpu-consumers-release-r56b`, `opposite-vendor-source-r55` and
`opposite-vendor-source-r57`, `release-build-r58`, `integration-build-r58`,
`focused-r58`, `opposite-vendor-source-r58`, `release-build-r59`,
`integration-build-r59`, `focused-r59` and `opposite-vendor-source-r59`.
The latest slice adds `release-build-r60`, `integration-build-r60`,
`integration-build-r61`, `focused-r60`, `focused-r61`,
`focused-admission-r61-summary.json` and `opposite-vendor-source-r61`.
Follow-up evidence is `integration-build-r62-fixture`, `focused-r62-fixture`,
`release-build-r63`, `integration-build-r63`, `focused-r63`,
`integration-build-r64`, `focused-r64` and `opposite-vendor-source-r64`.
The successful follow-up is `release-build-r65`, `integration-build-r65`,
`focused-r65` and `opposite-vendor-source-r65`.
The first r56 consumer wrapper failed before
launch because its session directory was absent; r56b corrects the wrapper and
executes both checks. The following sections retain historical evidence.

## GPU selected-worker creation

The first current-source authored cross-vendor lookup rejected admission before
creating its selected CUDA worker. Installed rows still named the earlier
`e52b5725...` creator source. The new uninstrumented production keys retained the
same runtime, driver, BLAS provider, pthread defaults, stream/handle/event counts
and compiled object geometries; their source identity is now
`8750ba1dbedbc1990852bcf085bf8d68880dec456610dc79e06cc1a9e69fd9e1`.

The existing native birth diagnostic exercised one cold lifetime and two warm
lifetimes per backend, with two uses and actual runtime reset after each lifetime.
It used the original creator, configured SDKs and an independent device-memory
observer. Both backends passed the existing receipt checks and driver window.
Selected-device VRAM returned exactly to the pre-process value after every
process; ROCm GTT did too.

| Backend | Maximum observed VRAM increase | Maximum RSS increase | Maximum observed GTT increase | Retained host allowance | Retained device allowance |
|---|---:|---:|---:|---:|---:|
| CUDA, RTX 3090 | 286,588,928 B | 109,580,288 B | Not exposed by this observer | 256 MiB | 384 MiB |
| ROCm, MI50 | 180,465,664 B | 35,401,728 B | 84,017,152 B | 256 MiB | 256 MiB |

The selected allowances are unchanged from the earlier reviewed classes. They
are conservative limits, not assertions that RSS or allocator samples enumerate
every private byte. Exact mapped worker stacks and objects remain separately
declared. Opaque commitments retain their `ProcessRuntime` lifetime. Old rows
for unreachable creator versions are replaced by the two measured current-source
classes; no UUID allowlist or anonymous reserve is added.

Local evidence:

- `/tmp/qfn-read-ahead-root-integration-r1/context-allowance-review-r44.json`
  (SHA-256 `c4df05b80ec2c571dd7793e09158957c08cb1d748341c46456151b91025ff118`).
- `context-key-r43-{cuda,rocm}` under the same directory: uninstrumented original keys.
- `context-birth-r43-{cuda,rocm}`: original cold/warm creation, use, reset and process-exit records.

## CPU IO and executor-ticket workers

The closed mapped-pthread family retains four selected provider identities and
cheap local loader mutation checks. Passive TLS libraries do not each create a
new portable class. Eligibility permits up to 64 modules and 1 MiB of declared
TLS memory. The candidate opaque allowance is 2 MiB for each admitted service
lifetime, with at most sixteen live services combined across both roles.

The existing process owner enforces that combined limit across independent
bindings. Exact 8 MiB stacks, object mappings and guards retain their own typed
ownership; the allowance includes retained libc bookkeeping and is not released
when a service joins. Repeated service lifetimes contribute their original
ProcessRuntime commitments rather than reusing already-spent allowance.

The r40 mixed populations of two and sixteen workers each passed 21 lifetimes.
The sixteen-worker run's maximum observed allocator-live increase was 113,744 B
over its metadata baseline; the loaded TLS population was 26 modules and
312,255 B per worker. This evidence informed conservative headroom, not a claim
of complete byte enumeration. Current-family r41 Integration and r42 Release
identity/constructor/use/join diagnostics both pass six fresh processes each.
The r41 focused slice also passes the existing actual IO and executor consumer
bodies with recorded declarations.

The four candidate rows bind the measured Release and Integration creator
identities. The explicit original installed-declaration IO and executor tests
now pass in both builds. The Integration focused slice passes all 25 selected
registrations, including combined-population limits and original unknown-class
rejection. The fixed ordering-event allowance was refreshed after unchanged
native creator review and fresh CUDA/ROCm checks. The resulting r48 native
lookup matrix passes all eight cells: 80 byte-checked transactions plus two
injected creation failures, across both vendor directions, repeated lifetimes,
request reset, M512 empty and partially populated GPU caches. Actual transfer
extents, original-context cleanup, driver windows and process files pass.

One fixture defect was corrected: its between-case cleanup assertion demanded
zero while the original two GPU worker stack/object roots were still live.
It now requires exactly the canonical preparation roots between cases and zero
exact allocations after original retirement. The focused 20-repeat regression
also rejects an extra allocation under that same owner; no allowance was widened.

The joined serving implementation subsequently passed the r50 Release build and
the same eight-cell native matrix, including all 80 byte-checked transactions
and both injected creation failures. The r51 Integration build also passes.
Focused joined regressions are running; full-model multi-request qualification
remains open. The instrumented native matrix is correctness evidence, not a
latency benchmark or complete preflight.

Local evidence is under `/tmp/qfn-read-ahead-root-integration-r1/`:
`host-service-growth-r40`, `host-service-native-r41`, `host-service-native-r42`,
`focused-r45`, `original-cpu-consumers-release-r45` and `authored-matrix-r48`.
The original-consumer checks reuse the same IO and
executor test bodies; their unknown-class controls still reject pricing and
creation before any native worker exists.
