# QFN: MI50 IH2 overflow and mapped preparation lifetime

This is an investigation record, not a tuning dashboard. The warning is not
yet qualified as resolved. The changes and observations below do not establish
that it caused the separate long-lifetime host watchdog resets.

## What the warning means

The running Vega20 driver allocates a 4,096-byte IH2 interrupt ring containing
128 32-byte interrupt vectors. Its overflow warning reports overwritten
interrupt records. This is distinct from an inference command ring, a tensor
buffer overflow, or an out-of-memory admission decision. The warning alone
does not identify a GPU virtual-memory fault.

The inspected host runs Linux 6.14.0-37-generic and amdgpu
7.1.9.31600000, module build ID
`f4cd17aab5cb859073d4cf559890d3f3124cedac`. The saved module was matched to the
loaded module note before interpreting disassembly or kernel offsets. AMD's
packaged source supplies context; a package-version match alone is not proof
of the loaded binary layout.

The [Vega20 driver source](https://raw.githubusercontent.com/ROCm/amdgpu/40b645b/drivers/gpu/drm/amd/amdgpu/vega20_ih.c)
shows deferred IH2 processing and overflow recovery. The native driver clears
the raw overflow flag before formatting its warning and masks the returned
write pointer. Consequently neither the printed hexadecimal value nor a
masked returned pointer is a substitute for observing the recovery branch.

## Confirmed traffic source

The first complete kernel census observed only ATS client 21/source 1 on IH2:
1,780 records per MI50 in that window. No GMC fault callback was observed in
the corresponding instrumented interval. This does not exclude a fault in a
different interval or a record already lost to overflow.

A small existing source-setup test using **only ROCm:0** established the
cross-device mechanism. Intel IOMMU device-TLB invalidations were emitted to
all four MI50 PCI functions, sharing `dmar1`, domain 2, PASID 0. Their queued
invalidation addresses exactly matched the following ATS record addresses
on all four IH2 rings. This is shared-domain invalidation fanout; it is not
evidence that Llaminar allocated four copies for that one-device test.

The trace establishes the operational connection to invalidation requests.
The consulted public headers do not define enough of source 1's payload to
classify it as a fault or a harmless completion solely from that source ID.
Do not assign either meaning without further hardware documentation.

The exact packaged driver has no registered amdgpu consumer for ATS client
`0x15`. Normal dispatch advances the ring, emits only a debug diagnostic for
an unregistered client and forwards the vector to KFD. KFD v9 rejects the
observed VMID 0 and excludes ATS from its accepted client list, so it does not
queue a KFD event for these records. That establishes the normal software
disposition, not that overflow is safe: the overwritten interval may contain
other clients, and source 1 hardware status remains undocumented here.

Native mapped/GTT ownership remains valuable for allocation, alias and release
lifetimes. It does **not** eliminate IOMMU or device-TLB maintenance. Earlier
comments claiming that native mapped ownership guarantees freedom from IH2
storms have been corrected. The previously diagnosed anonymous USERPTR
compaction/THP invalidations are a separate established ownership issue.

## Service delay and direct ring observations

A kernel-only model run found invalidation bursts and queue-to-execution
delays up to 219 ms, while the IH2 callback itself took at most 73 microseconds.
The largest delays occurred during weight preparation, before retained graph
family creation. Thus graph ticket allocation alone cannot explain every
observed warning phase.

A subsequent probe observed entry/return pointers from the driver's original
`vega20_ih_get_wptr` call, without issuing another MMIO read. Non-recovery
samples measure `(wptr-rptr)&mask` at that boundary. A read-pointer change
inside this function identifies the native overflow-recovery path; its
post-recovery extent must not be interpreted as pre-overflow headroom.

That full model load and recreation run passed. Its four sampled highwaters
were 10, 10, 10 and 60 vectors, with no recovery. Of 153 observed IH2 queue
delays exceeding 1 ms, 143 directly followed `delayed_fput` on the same worker
within 1 ms of that callback's completion. Activation itself took only
804–4,857 ns for those delayed items. This demonstrates a specific shared
worker-queue service delay, rather than assuming application CPU saturation.

The longest delays overlapped `delayed_fput` executing on a different worker.
Those overlaps do not prove exclusive blocking or identify the complete
cause of the historical overflow. A callback's wall duration also includes
sleeping and preemption; it is not CPU time. No instrumented run described
here has yet captured the overflow itself.

[Linux's workqueue documentation](https://docs.kernel.org/core-api/workqueue.html)
explains the shared bound worker pools and the separate high-priority pools.
This makes IH2 worker-pool isolation a plausible driver candidate, not a
qualified fix.

## Concrete Llaminar preparation defect

`materializeForwardWorkspaceFamily` constructs temporary graphs with the
existing typed `WorkspaceDeclaration` intent. MoE lowering nevertheless
physically bound dispatch and canonical-return tickets into the builder's
persistent mapped arena. Destroying a temporary declaration graph dropped its
slice references but did not reclaim the arena's bump positions or backing.
Actual retained graph construction then allocated additional slots.

The implemented fix uses the same immutable geometry and original construction
intent to distinguish declarations from physical bindings. Declaration-only
tickets must allocate no native payload, publication word or metadata slot;
their graph still declares the original stages and workspace requirements.
Executable materialization creates a fresh physical ticket. It does not reset
the arena, recycle captured pointers, invent a second accounting ledger, or
guess a large reservation from the measured workload.

Focused regressions prove both zero allocation for declarations and rejection
of declaration-only execution/publication, including CPU no-op methods. They
compare actual stage workspace requirements, exercise the original model
factory, and retain physical ticket replay/reset behavior. Native model
load/recreation is covered separately below. Removing this unnecessary work is worthwhile independently;
it must not be described as a proven complete IH2 remedy.

## Declaration-fix qualification

The coherent Integration build and two refreshed native capture test targets
passed with the existing spill guards and configured CUDA SM80/86/89/90 and
ROCm gfx906/gfx1100 target set. Seven selected host CTest entries passed,
including complete mapped-transfer and MoE graph owners; six native CUDA/ROCm
entries passed. All 13 entries had no skips and clean driver windows. The five
new declaration-specific aliases are registered explicitly in production
preflight. The complete Unit/preflight gate was not rerun for this slice.

The unchanged four-MI50 RAM source/admission/recreation case then passed with
MTP off (316.00 s CTest), followed by the original active fixed-depth-1,
retained-capacity-3 case (315.82 s). Both driver windows were clean, including
cleanup. These are real source, capture and lifecycle checks, not numerical
generation or Release performance certificates. The failure-only probe
observed no overflow in either window.

Off admission owner, ngram setup, planned-memory and retirement artifacts are
identical to the original r884 run; only observed available CPU memory changes.
Those exports do not isolate actual ticket arena backing/slice totals. No full
model memory-saving estimate is inferred from reserved capacity. The focused
factory regressions directly prove zero ticket slices/backings for declarations;
the original physical-binding control fails those assertions.

The feature checkpoint contains only the reviewed declaration/MTP changes and
ownership-comment corrections. The Integration checkout also contains pending
read-ahead work. The new enum extraction in `IGraphBuilder.h` is transplanted
without that unrelated work; matching changed implementations/hunks are
qualified here, not the entire feature tree as a release build.

Local validation receipts: `/tmp/qfn-workspace-ticket-integration-r885/`,
`/tmp/qfn-ih2-declaration-model-r886/` and
`/tmp/qfn-ticket-artifact-export-audit-r886/`.

## Focused native observations

These are diagnostic timings from the coherent Integration binary, not Release
throughput claims. The ordinary model selector retains its original fixture,
MPI guardian, socket binding, four explicit MI50 identities and timeout. Driver
observation spans cleanup. There were no test skips.

| Observation | Result | Interpretation |
|---|---|---|
| Full model with kernel invalidation/workqueue trace, r877 | Passed; 319.85 s CTest; driver clean | Bursts plus long queue delays; no overflow reproduced |
| Full model with direct ring/workqueue trace, r880 | Passed; 341.14 s CTest; driver clean | Sampled maximum 60/128 vectors; no recovery |
| Original model with overflow/workqueue-only probe, r884 | Passed; 316.62 s CTest; driver clean | No overflow branch; queue delay reached 1.004 s, demonstrating that delay alone is insufficient |
| Original full model without BPF/API tracing, r883 | Application assertions passed; 342.58 s CTest; **driver failed** | Two IH2 overflows, on PCI `9a:00.0` and `91:00.0`; clean traced runs are not a fix |
| Real ticket/retained RCCL lifecycle, r878 | Passed; 6.72 s | Actual compact publication, unchanged unused tails and retirement |
| Recorded mapped backing extents, sequential outer setup, r878 | Passed; 50.46 s | Allocation volume alone did not reproduce overflow |
| Same backing extents with original TP worker fan-out, r881 | Passed; driver clean | Outer setup/first-touch concurrent; native allocator still owns its serialization |
| Concurrent backing case repeated five times, r881 | Five passes; driver clean | Ten allocation lifetimes on five original retained cliques; no overflow |

The large diagnostic replays 18 observed backing extents per participant,
72 regions and 49,576,673,280 declared bytes per family, twice on one retained
communicator. These are summed allocation extents, **not peak physical pages
or RSS**. It does not reconstruct the unobserved sizes of all 29,424 logical
arena slice requests or replace full-model qualification. The approximately
50 GB diagnostic remains a private manual investigation, not a Unit or routine
preflight workload.

No host driver, IRQ affinity, ring capacity, IOMMU mode, timeout policy or
warning classification was changed. Native ownership and capture remain the
production implementations.

## Driver change boundary

Submitting the same Vega20 IH2 work item to `system_highpri_wq` is the narrowest
reviewed experiment for isolating normal-pool cleanup delays. It preserves CPU
locality and per-work serialization, but is not a real-time scheduling
guarantee. It has not been installed or called production-safe.

The packaged driver also needs a lifecycle audit before such a change is
qualified: its current disable helper resets hardware/software ring pointers
while masking the ring, and no IH2 work drain was found before final backing
release. A safe boundary needs producer quiescence without pointer reset,
IRQ synchronization, draining the original work, then pointer reset/release.
Suspend, reset, shared-IRQ and VF behavior must be preserved. This is a
pre-existing source-level concern, not an observed use-after-free diagnosis.
Blindly adding cancellation after the current reset, changing affinity,
increasing the ring, or suppressing warnings would not close those contracts.

## Evidence and remaining work

Local detailed receipts are retained outside the source repository:

- `/tmp/qfn-ih2-driver-semantics-r872/R875-CAUSAL-ATTRIBUTION.md`: exact QI/ATS
  address attribution and observer limits.
- `/tmp/qfn-ih2-driver-semantics-r872/r877/NOTE.md` and `r880/NOTE.md`:
  burst/service and direct occupancy analyses.
- `/tmp/qfn-native-mapped-lifecycle-trace-r869/analysis/FINAL.md`: original
  API lifetimes and backing geometry; every observed lifetime retired.
- `/tmp/qfn-native-mapped-observation-r878/` and
  `/tmp/qfn-native-mapped-concurrent-observation-r881/`: native lifecycle runs.
- `/tmp/qfn-ih2-workqueue-option-r881/LIFECYCLE.md`: driver reset/drain boundary.
- `/tmp/qfn-ih2-driver-semantics-r872/ATS-DISPATCH.md`, `r883/PHASES.md` and
  `r884/NOTE.md`: consumer disposition, warning phase attribution and the
  reduced-tracing observation.
- `/tmp/qfn-workspace-ticket-declaration-r877/`: allocation-fix development
  and focused checks; coherent qualification is recorded above.

Earlier observers retained explicit limitations: first-use aggregate-map races,
incomplete raw-print reconciliation, and optional activation-map cleanup
warnings. Revision 2 fixes the cleanup warning and distinguishes ordinary
highwater from recovery extent. Queue lifecycle mismatches remain reported;
do not turn partial pairs into an exact universal latency distribution.

The uninstrumented original model reproduced two overflows at boot-monotonic
364251.679549 and 364502.477733 seconds. The first occurred during weight
preparation, after follower upload and before coordinator GPU preparation.
The second falls within the recreated 512-row prefill family. A narrower
observer probes only the authenticated overflow branches plus workqueue lifecycle, avoiding
normal interrupt and invalidation probes. Its read-only raw-vector snapshots
are concurrent and may already contain overwritten records. Missing queue
boundaries explicitly clear timing fields rather than reusing a prior delay.

The main declaration fix has host regression and original-behavior negative
control evidence. The same construction intent now propagates through
MTP manifests and predictor builders. Its focused host tests, independent
review and the coherent native/model checks above pass. Preserve the distinction
between a clean run, reduced avoidable mapping work, and a reproduced/qualified
overflow fix.

## Separate Intel IOMMU timeout-recovery defect

The workspace's November 3 Intel IOMMU investigation identifies upstream
[`fe89277c9ceb`](https://github.com/torvalds/linux/commit/fe89277c9ceb0d6af0aa665bcf24a41d8b1b79cd).
Batched submissions can put completion entries at even queue indices; the old
recovery code aborts only odd entries and can leave polling stuck indefinitely
after an invalidation timeout. The correction visits every affected entry.
The exact official Ubuntu source matching this host still contains the defect,
and the unmodified upstream patch applies cleanly. This is a concrete candidate
for the previously recorded IOMMU hard lockup, not a demonstrated remedy for
initial endpoint timeouts or AMD IH2 overflow. Host kernel upgrade/backport
qualification is a separate work item; no kernel or boot change belongs to this
Llaminar checkpoint.
