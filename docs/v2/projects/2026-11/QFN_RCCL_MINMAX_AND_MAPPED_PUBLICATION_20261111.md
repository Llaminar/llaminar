# Native reduction ABI and mapped publication repairs

This note records focused qualification on 2026-11-11. Complete Unit and
hardware-eligible production preflight, real QFN parity, and Release performance
remain separate requirements. The active target is PP(EP(4×MI50), EP(4×MI50))
with native ngram shards distributed across all eight GPUs.

## RCCL minimum-budget reduction

The isolated RCCL loader declared MIN=2 and MAX=3. The selected RCCL header
defines MAX=2 and MIN=3. The pipeline maintenance admission operation requested
MIN but therefore executed MAX. An inactive participant contributes INT_MAX to
the intended minimum, so the incorrect operation could remove the maintenance
bound. Unequal active periods could likewise select a larger allowance.

The failure preserved an actual controller with one commit remaining and a
verifier result containing two new commits. This was a transport ABI defect,
not a numerical tolerance or MTP sampling discrepancy. The loader now matches
the native header. Internal backend wrapper opcodes retain their different
encoding: their existing adapters translate those codes to native enums.

All eight previously failing original request-lifecycle cells now pass:
EP2→EP1 and EP1→EP2→EP4, each with ordinal/random placement and fixed-depth-3 /
dynamic MTP. Each original fixture retains its fresh, full-prefix and partial-
prefix requests and reset checks. The complete driver interval is clean, with
zero findings. The focused command took 138.1 seconds including observation.

Evidence: `/tmp/qfn-rccl-and-focused-followups-20261111/mtp1.xml` and
`mtp1-qualification.json`; command/driver artifacts use
`/tmp/qfn-full-preflight-after-cpu-next/q7-rccl-fixed-unequal-mtp1*`.
The small extracted original protocol words remain under
`/tmp/qfn-mtp-cadence-core-readonly/`. The two large GPU dumps were removed after
the repaired cases passed; the matching binary closure remains retained.

The added prevention checks compile the isolated NCCL/RCCL enums against their
selected dependency headers, exercise the actual dynamic-loader ABI, and run
captured INT32 MIN/MAX over original CUDA2 and ROCm2/4 cohorts. Twenty retained
replays vary signed values and inactive participants. These added checks are
explicit preflight entries. Their CUDA2 and ROCm2/4 runs now pass all twenty
replays, with complete clean driver observations. The native API is rooted
`Reduce`: the corrected test requires exact results on the declared root and
untouched receive poison on nonroots, rather than expecting allreduce results.
The host ABI and dynamic-loader checks also pass.

## Mapped dispatch belongs to the acquired generation

A follower can enter its retained admission and dispatch waits while the prior
request's reusable signals are still published. Those waits can complete before
the new grant is acquired. The validator formerly loaded the dispatch signal
only once after acquiring the new grant, and could reject the newly reset
descriptor before the new source published it.

The preserved native failure names generation 2, with the correct grant identity,
but ConsumeDispatch observed timeline 0. The eventual source descriptor and all
compact row/route operands were exact. The original tests now join failure
outcomes across the original MPI ranks before teardown, avoiding a secondary
MPI_Finalize/next-test-collective disagreement.

The common CUDA/HIP scalar and cooperative validators now wait for dispatch
after acquiring the current grant. They retain the original system-acquire
loads, exact descriptor validation and absorbing cancellation protocol. There
are no new graph nodes, host readbacks, allocations or recapture. New device-free
and native regressions delay dispatch after follower acquisition, cover abort,
and retain partial/empty work and compact-byte checks. All 22 original mapped
aliases pass, as do both backend directions of the delayed-dispatch and
cancellation regressions. The latter retain the complete 4/3/1 captured row
roster; their instrumentation oracle checks every retained family. The full
epoch-protocol Unit owner and its focused publication regression also pass.

The combined 29-case native run, including the three MIN/MAX cases, completed
in 146.8 seconds including observation, with zero driver findings and no skips.
Evidence is `/tmp/qfn-focused-fixture-corrections-20261111/native1.xml` and
`/tmp/qfn-full-preflight-after-cpu-next/q8-fixture-native1/`.
Original failure evidence remains `mapped1.xml` beside the MTP receipt; two of
three diagnostic aliases failed in 23.4 seconds with a clean driver interval.

## Admission and fixture boundaries

Ten CPU/GPU joint captured-execution regressions pass after correcting an
invalid fixture domain that specified both rank-local ownership and explicit
rank membership. CUDA and ROCm each cover prefill, decode, verifier, original
fanout and zero-weight compaction. Evidence: `cpujoint1.xml` in the same focused
artifact directory, with complete clean driver observations.

The fixed-head ngram regression also exposed a production distinction:
`ResolvedImmutableRowTablePlacement::allocators()` returned PLE consumers for
fixed shards. Later pipeline stages can own shards without executing PLE.
The correction projects actual layout owners while preserving the separate
consumer permission check. Its existing explicit fixed-head preflight test now
asserts both roles and passes. The full MemoryPlanner owner now passes all 478
bodies. Its remaining fixture failure was an absent directed peer matrix in
the three-tier automatic-selection test; complete negative observations now
distinguish unavailable peer access from missing discovery. The full route
policy/configuration owners and their explicit preflight regressions pass.

The original mixed lowering test exposed a missing prepared-store binding in
its singleton fixture. It now binds the same admitted model store. Production
Batch construction additionally checks that binding and reuses the existing
model/PMA identity validation, producing an explicit diagnostic instead of a
null dereference. The full Qwen MoE graph Unit owner and the focused authority
contract pass. The original CUDA→ROCm lowering case passes along with twelve
previously failing preparation/lifecycle cases: EP4→EP4 and EP1→EP2→EP4
preparation with fixed/dynamic MTP, and EP1→EP2 ordinary request lifecycles.
That thirteen-case native run took 291.7 seconds including complete clean driver
observation. Receipts are under
`/tmp/qfn-lowering-prepared-store-install-20261111/`, including
`focused-qualification.json`; the earlier 5-host/29-native summary is under
`/tmp/qfn-focused-fixture-corrections-20261111/focused-qualification.json`.

Legacy packet components now retain their original source and PMA through
resource creation, captured replay and retirement. Their 23 exact aliases
preserve the original byte/metadata oracles. All 23 passed on the expanded host,
including the two same-owner CUDA/ROCm cases made available by the new MI50 on
NUMA0. The eight-GPU QFN target remains the original eight gfx906 GPUs on NUMA1;
selection uses observed ownership and ISA rather than fixed device ordinals.

Production background stream pools retain their original prepared permit and
declare completion/timing events through the same PMA. Public reset rejects
live admitted streams; repeated transfers reuse retained events. Serial
host/device calibration reuses its existing worker stream without an auxiliary
pool. Component fixtures use the explicit test-only infrastructure boundary;
production retains mandatory admitted original contexts. The affected 522-step
build completed successfully in 392.9 seconds. Its earlier interrupted build
is not a certificate.

The subsequent 41-case native batch passed 37 tests in 162.1 seconds. That
includes all 23 packet aliases, both CPU/GPU floating fabrics, calibration and
joined prepared services, and four ROCm peer-lane formats. Four CUDA direct-peer
cases correctly rejected absent directed access on that boot. The complete
driver window reported zero findings. Evidence:
`/tmp/qfn-transfer-caller-composition-20261111/native1.xml` and
`/tmp/qfn-full-preflight-after-cpu-next/q10-caller-fabric-native1/`.

The eligibility repair consumes the complete frozen directed-access matrix.
It requires one usable same-owner directed edge, not a symmetric clique, and
selects the actual owner before creating native resources. Missing observations
remain errors. C++ owner-selection and Python qualification regressions cover
asymmetric access, an edge beyond the selected prefix, a later eligible owner,
missing facts, cross-NUMA rejection and repeated selection. Both have explicit
preflight registrations. The host eligibility/registration batch passes all
five selected CTest entries, including the complete Python campaign tests.

Five capacity-fixture omissions and the event fixture's process-root lifecycle
are repaired. Both backend event workers now share one original preparation
and admission, with their own process executable. All five focused CTest
entries pass; no production budget or failure check was relaxed. Receipts:
`/tmp/qfn-host-fixture-repair-20261111/focused.xml` and
`/tmp/qfn-directed-peer-install-20261111/host1.xml`.

The user subsequently enabled 3090 P2P and rebooted. The current boot is
`84ef232e-7b3e-4e85-a9b8-91a0c09172ad`; the previous native receipt belongs to
`fdb2359a-0a84-42ad-a807-cc311f040d04`. Fresh discovery now observes CUDA peer access in both directions.
All eleven current-boot native cases pass: four formats on each backend, two
local mixed-vendor packets and the ROCm projection-exchange fixture. The last
fixture now resolves two actually owned same-NUMA devices through the original
singleton MPI bootstrap; ordinal zero is no longer assumed to belong to that
cohort. The 53.3-second CTest run has complete clean driver observation.
Evidence: `/tmp/qfn-directed-peer-install-20261111/native-current.xml` and
`/tmp/qfn-full-preflight-after-cpu-next/q14-native-current1/`.

Complete gate dependencies now compile. Besides the two observed-fixture
name-lookup corrections, this refreshed the header-relative owner helper and
one component-only raw route-storage caller. All original component byte
oracles remain. The final full-target build is recorded as
`q14-all-gate-build1.json`; it is build evidence, not a passed prerequisite run.

The next exact model slice has validated task-local paired inputs at
`/tmp/qfn-ep4ep4-final-inputs-20261111/`. Its sixteen histories, teachers, reset
controls, original source and CPU/FP32 reference report are unchanged. The
reviewed HC-up two-term pilot limits are projected to the new topology without
inheriting old TP4 outcomes, approval or calibration. A cold query through the
actual original MPI binder produced the expected physical configuration; it
loaded no model and emits no qualification receipt. The complete observed
paired route now accepts catalog-bound external topology metadata while
requiring every other original corpus field unchanged; focused and complete
receipts remain distinct. Its Unit owner and focused preflight test pass.
Complete Unit/preflight and real-model parity/performance remain open.

The refreshed paired matrix builds and discovers all 2,134 QFN numerical
cases (`q16-all-gate-build1.json`). Release configuration now removes one
retired executable name; its mapped packet tests already belong to the retained
captured-transfer owner. The paired document remains an Integration-only input.
The Release comparison also now checks MPI placement independently from the
session helper's build identity. Integration and Release retain distinct,
stat-pinned executable helpers in their own build directories; rankfile, MPI
executable, population, binding and overlap policy must still match exactly.
The complete Python Unit owner and the explicit
`V2_Integration_ObservedPipelineMatchedReleaseLaunch` regression pass in
`/tmp/qfn-release-launch-install-20261111/focused.xml`. This repairs harness
admission only, without claiming new model accuracy or throughput.

The recorded hardware supports 4,174 of 4,212 preflight registrations before
that additional host regression. All 36 EP4+EP4 cases and eight directed-peer
cases are eligible. Thirty-eight geometries remain explicitly unqualified:
larger CUDA populations or larger distinct-NUMA ROCm/mixed populations. The
new NUMA0 MI50 admits seventeen additional existing geometries. The next complete
run will collect all completed test failures across the canonical lanes.

The composed patch receipt is
`/tmp/qfn-admission-abi-composition-20261111/install.json`. Full shipping CUDA
sm80/86/89/90 and ROCm gfx906/gfx1100 kernel compilation passed spill checks; the
configuration serializer's aggregate-field audit required adding the new typed
route policy and diagnostic to its lossless version-13 document. Round-trip,
malformed-document and restored-runtime-policy regressions now pass. The first
combined build's serializer issue is resolved. None of these component results
is a real-model throughput measurement.
