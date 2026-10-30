# Immutable lookup worker cancellation and retirement

Size limit: 15,000 UTF-8 bytes. Focused defect and qualification record;
this does not certify the complete QFN source-admission matrix.

## Failure and correction

The retained immutable-row I/O worker could stop on cancellation before
acquiring a published GPU request. In that case no completion established
worker retirement. An acquired subordinate source grant could also remain
held after the worker stopped. A consumer must distinguish a completed response
from an aborted interval; cancellation cannot manufacture payload consumption.

The service now authenticates both the original admitted generation and the
actual persistent-worker invocation. Cancellation rejects a foreign or retired
ticket. Every worker exit releases its own acquired grant, then publishes that
exact retirement pair. Captured bindings and independent reader pins retain
their original backing until their own lifetimes end. Worker retirement alone
does not authorize graph or payload-reader retirement.

Review of the first candidate found a second race: cancellation could publish
after worker retirement but before `await`, leaving those metadata bytes absent
from the terminal counters. Cancellation and worker retirement now use the same
existing CPU lifecycle mutex. Late cancellation rejects without writing; repeated
cancellation of a still-live invocation is idempotent. Invocation publication
is acquired before reading its generation body in separate statements.

The mapped acknowledgment remains 64 bytes: the new fields use its reserved
space. A normal interval reports 24 terminal metadata bytes: one GPU close word
and the CPU worker's two-word retirement witness. Cancellation without a GPU
close reports 32 bytes for the cancellation and retirement pairs. No GPU model
state is read or mirrored by these CPU lifecycle operations, and no new backing
allocation is introduced.

## Regressions and evidence

Six actual-worker tests in
`tests/v2/unit/execution/qwen4exp/Test__ImmutableLookupSubordinate.cpp` cover:

- Cancellation before request acquisition, with captured backing retained.
- Cancellation of an acquired grant without fabricated consumption.
- Large-to-small and empty request reset, including stale tickets.
- Publication/cancellation races with either truthful completion or no acquired
  completion, while requiring exact retirement.
- An actual SSD worker's partial-payload failure without successful bytes or
  promotion acknowledgments; the allocation-failure control is process-local.
- Late cancellation before `await`, exact metadata accounting, and subsequent
  admission rejecting the old invocation.

The explicit `V2_Integration_ImmutableLookupWorkerCancellationReplay` registration
belongs to `ProductionTestPreflight`; broad Unit coverage remains in place.
Existing native assertions now count the retirement metadata on both vendors,
including empty and reset intervals. Direct host-processor tests still count
only the messages their processor actually handles.

Main-tree Integration/AVX512 qualification:

| Gate | Result |
|---|---|
| Complete Unit | 792/792 pass; zero skips; 117.1 s |
| Worker preflight replay | 20/20 invocations pass; 10.6 s |
| Affected native capture/reset gates | 24/24 pass; 12 CUDA and 12 ROCm; zero skips |
| Native retirement and driver window | Complete process retirement; clean driver window |
| Compilation | Canonical CUDA SM80/86/89/90 and HIP gfx906 spill guards enabled |

Main evidence is `/tmp/qfn-worker-cancellation-main-r2/`: build commands/results,
Unit and focused JUnit, exact native inventory, process retirement and complete
driver observation. The focused native transaction took 34.3 s. This is not a
fresh execution of the complete production-preflight inventory.

Private negative controls and repeated qualification are retained in
`/tmp/qfn-immutable-worker-cancellation-r2/evidence/`. The original main worker
body fails the missing-retirement regression; the preserved first candidate
fails the late-cancellation regression. The restored corrected implementation
passes. The reviewed aggregate patch is SHA256
`b17fea4483977aab117a13c44d5d2e0a9ad665a4584d7fcd0b0ba3cd161664ea`;
the main checkpoint additionally includes five native metadata-count assertions.

Pooled RAM/SSD source integration, multi-owner transport, and complete native
admission remain separate work in
[the admission matrix](QFN_SOURCE_ADMISSION_MATRIX.md) and
[the preparation plan](QFN_NATIVE_ADMISSION_PREPARATION.md).
