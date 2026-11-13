# MPI runtime retirement during QFN admission

Updated 2026-10-30. This note records the cleanup repair and its qualification
boundary. It does not qualify additional model/topology admission cells.

## Observed failure

The complete r16 prerequisite run collected 3,123 passes, one failure and 29
hardware skips across 3,153 tests. Unit passed 809/809 and the host lane passed
1,262/1,262. `V2_Integration_GPURamPrefixArena_CUDA` failed inside MPI startup,
before its test body: PMIx encountered root-owned files from an older job whose
PID had been reused. Disk capacity and the prefix arena were not the cause.

Setting only OpenMPI's ORTE temporary-directory option did not contain PMIx's
shared-memory backing files. Native tracing showed that PMIx also consumed
`TMPDIR`. Deleting global files by name or PID would not establish ownership.

Original evidence remains in
`/tmp/qfn-admission-integration-r16/final-failure-inventory.json`,
`mpi-startup-failure.txt` and `mpi-startup-collision.json`. The r16 receipt remains
failed; focused repairs do not rewrite it as a passing gate.

## Production ownership

`MPISessionOwner` supplies one process-entry boundary for Llaminar, its MPI
frontends and canonical process-owned tests. A small CPU-only sibling helper
starts a retained ancestor before MPI/runtime setup. That ancestor owns fresh
temporary and shared-memory directories, authenticates original process and
directory identities, and retains the actual descendant tree through exit.

Normal exit, initialization failure, signals, abandoned descendants and abrupt
proxy death share the same retirement path. Files are removed only after native
child retirement. Finalization alone does not release files, and successful
parent exit does not leave its daemon children indefinitely. A child that cannot
retire leaves its backing intact with an error. There is no global cleanup scan.
Persistent user-selected cache and model paths remain outside these private
runtime roots; this change adds no cache-payload checksum or hash pass.

Startup loans bind the original guardian, immutable protocol and original
directory device/inode/UID. Partial or replaced loans fail before initialization.
Original arguments, interactive I/O and the application's actual exit status
are retained. Native MPI receives its own descriptor lifecycle, without a
partial inherited startup loan.

## Native daemon environment

A real native remote-daemon-path test found a second fault: OpenMPI 4.1.6
merged launcher-valued application environment over daemon environment. Correct
MPI output and clean teardown concealed vader mappings in the launcher's shared
root instead of the daemon's root.

The required `mca_schizo_llaminar` native component uses OpenMPI's existing
`setup_fork` callback after its ordinary environment merge. It projects the
daemon's original locations from the shared declaration in
`MPISessionEnvironment.h`. It creates no additional owner or rank communication channel.
Rank readiness binds actual root identities and is checked before `MPI_Init`.
Default native component selection is preserved; an explicit selector that
omits the required component fails precisely.

The helper and its sibling component are built and installed together. The
component is bound to the selected OpenMPI 4.1.6 ABI. Missing components, wrong
layout/version and incompatible selection fail before native rank entry.
Installed-test receipts include the native module as a required runtime asset.

## Regression evidence

- The final private source passes 49 local lifecycle cases, including actual
  two-rank collectives, singleton MPI, failure, repeated interruption, orphan
  cleanup, CTest timeout, concurrent owners and replaced ownership metadata.
- The native daemon suite passes all seven cases. Actual PMIx and vader mappings
  and all eight location bindings agree with the live daemon's original roots.
  Success and interruption retire all observed process handles and directories.
- CTest exposed an observer defect: a rank briefly inherited its parent's
  `orted` command line before `exec`. The observer now resolves the live rank's
  actual parent and executable instead of selecting a historical command-line
  match. The original failure was reproduced; the corrected interrupted case
  passes 20 consecutive CTest runs, followed by the complete seven-case suite.
  Failure receipts are emitted into CTest output before temporary cleanup.
- Direct-entry replays exercise original integration/E2E/shared mains, original
  argument identity and terminal status. Metadata, campaign and installation
  regressions cover owner registration, timeout grace and native module assets.

Frozen component evidence:
`/tmp/qfn-mpi-production-lifecycle-evidence-r1/native-module-r2/HANDOFF.md`.
Local lifecycle evidence: `final-native-lifecycle-r3.json` and its log in the
same evidence parent. Native daemon evidence:
`/tmp/qfn-mpi-native-daemon-preflight-r1/native-r3/`.
Observer regression evidence:
`/tmp/qfn-mpi-native-daemon-observer-r1/HANDOFF.md`.

## Current qualification

The r18 build and 65 focused native regressions passed. The complete failure
inventory across its original gate and diagnostic continuations is **3,167
passes, nine failures and 29 hardware-unqualified skips**. This is diagnostic
coverage, not a passing canonical receipt. The original gate stopped after two
native test guardians retained their temporary owner files; their numerical
test bodies passed. Six other fixtures lacked the now-required owned-backing
startup acknowledgment. One GDN capture fixture completed all six cases but
exceeded its unchanged 120-second deadline during teardown.

The reboot continuation ran the remaining 299 tests on the original MI50 UUIDs
with native ISA identity: 271 passed, that GDN deadline failed, and 27 were
hardware-unqualified. Its actual process and file roots all retired. Original
sources and collective libraries remained unchanged throughout collection.
Evidence: `/tmp/qfn-admission-integration-r18/final-diagnostic-inventory.json`,
`/tmp/qfn-admission-r18-prerequisites/` and
`/tmp/qfn-admission-r18-reboot-continuation-r2/`.

r19 integrates independent directory enumeration instead of a shared `dup`
cursor, typed retirement states and preservation of the first cleanup error.
An original-code replay failed 80/80 times; the repair passed 80/80. That
reproduced defect has not yet been proven to cause the two intermittent native
guardian failures. The fixture repairs use the real owned-backing test link.
The GDN fixture checks every inactive byte with a bounded block comparison,
removing 1.47 billion scalar assertions while retaining poison coverage and
the same deadline. Full native requalification follows the integrated rebuild.

The integrated batch also addresses shared-file creation failure.
Whole-translation-unit native NCCL/RCCL replays reproduce leaked names after
failed allocation/mapping, clobbered existing names and a failed-open heap
leak. Exclusive creation and partial cleanup fix those cases on both backends.
Builder proofs are bound to actual source/compiler/library identities; installed
verification needs no compiler, GPU or vendor source. RCCL patch-chain tests
also prove upgrades from the published predecessor, not just a local intermediate.
The new complete application build and r19 gate remain pending.

## Remaining ownership boundaries

An interrupted creator needs containment as well as normal error cleanup.
Matched native tests still find global NCCL/RCCL shared files after forced exit,
even with the partial-creation repair. The preserved control is
`/tmp/qfn-sdk-abrupt-file-retirement-r1/`; it must not be represented as fixed by
the partial-file patch. A private scoped native-body candidate passes repeated
normal and interrupted lifetimes, but communicator initialization, transport,
proxy and NVLS propagation and native qualification remain outstanding.

Project-owned ShmemSpin and MoE shared backing are being converted to borrows
of the existing session's original directory and file identities. Focused
tests include interrupted setup, replaced directories, ordinary descriptor-number
reuse, peer identity/extent rejection and startup failure. These are private
component results until the complete application build and gate pass. Their
original failures and candidate evidence live in
`/tmp/qfn-owned-shmem-project-evidence-r1/`.

The native daemon proof uses actual OpenMPI over a declared same-host transport
adapter. Cross-host filesystem namespaces and scheduler personalities remain
separate qualification obligations. External unowned MPI launchers retain their
own job ownership. Host reboot or termination of the guardian itself can retain
private roots for explicit recovery; future runs use fresh roots.
