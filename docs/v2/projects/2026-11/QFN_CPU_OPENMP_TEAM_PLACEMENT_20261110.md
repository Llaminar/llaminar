# QFN CPU expert-team placement

The 2×MI50 + 2×CPU profile found a real startup-policy mismatch: the CPU-only
socket used its core-bound startup team, while expert computation on the
GPU-owning socket created another OpenMP team with socket-wide places. GNU
libgomp was correctly loaded on both ranks. The performance target remains
500 prefill / 40 decode tok/s; the profile itself is not a throughput result.

## Evidence and cause

The existing Release benchmark owner retained the original MPI/socket launch,
28 physical workers per socket, Qwen3.8-Flash-Next UD-Q4_K_XL and its Q8_0 MTP
sidecar. Context 262,144; 512 prompt tokens; 256 generated tokens; seed 42; one warmup
and three measured requests were unchanged. Dynamic MTP, expert balancing,
RAM ngrams with local GPU caches, and 16/32 GiB RAM/disk prefix tiers remained
enabled. The preceding native-Q8 table candidate remained uninstalled.

A separate warmed `perf` attachment sampled user cycles every 256M cycles with
32 KiB DWARF stacks and actual CPU IDs. Exact emitted benchmark entry/return
probes identify four complete windows: prefill/decode for measured requests 2/3.
Both native MPI identities and the configured GNU DSO were authenticated. No
sample losses, parser omissions or sampled-CPU/mask mismatches were observed.
All 768 generated tokens and complete MTP records match the unprofiled control.

Execution rank 0's actual expert team comprises its retained ticket-service
thread plus 27 OpenMP workers. Every member has sampled native expert-kernel
instructions; their allowed mask spans the whole GPU-owning socket. Actual
samples use both SMT siblings. The separate 27 startup workers pinned to cores
29–55 consume zero CPU ticks in the retained 18.65-second observation bracket.
Execution rank 1's 28 expert workers instead run on singleton cores 0–27.

`MPILaunchConfig` defaulted to `OMP_PLACES=sockets`. Local discovery exported
that before exec. The child later wrote `cores`, after libgomp could already
have parsed its environment, and explicitly pinned only the bootstrap caller's
team. CPU experts serviced through `RetainedParentTicketServiceWorker` can
create another team from a separate pthread. Hostfile launches also lacked
portable pre-exec placement publication. In addition, a new pthread does not
inherit a different thread's `omp_set_num_threads`/dynamic-team task settings.

GNU accounts for 60.86% prefill / 76.21% decode of sampled user cycles on rank 0 and
66.81% / 81.77% on rank 1. These are resource shares, not removable elapsed time:
spin overlaps useful work and sleeping/futex time is absent. Many GNU internal
symbols are stripped;1,847included chains lack a recognized thread/process
entry frame. Direct kernel samples and complete service-worker caller examples
establish the roles above without pretending every chain unwound completely.

## Scoped change

The existing launcher now installs core places before exec and explicitly
exports placement/binding names to hostfile ranks. MPI socket mapping stays
unchanged, and remote worker counts still come from each rank's own geometry.
Explicit launch-policy strings remain explicit. A core place may contain its
allowed SMT siblings; the requirement is one worker per physical core.

Rank sizing uses the existing `ScopedOpenMPPartitionAffinity` to observe its
whole admitted partition even when libgomp has already narrowed the initial
thread to its first place. This prevents accidentally admitting one worker.

The existing immutable retained-service program captures the declaring task's
worker-count and OpenMP dynamic-team policy. Its actual service pthread applies
them once before entering its work loop. A later materializer or caller cannot
replace them. Program revalidation retains these frozen task settings while
still reobserving pthread defaults and loader/provider state. There is no new
placement registry, communicator, memory ledger or per-token affinity operation.

## Qualification

Seven focused CTest registrations are green across the final runs:

- Full `V2_Unit_MPIBootstrap` and explicit `V2_Integration_MPIBootstrapOpenMPPlacement`.
- Existing native `V2_Integration_ClusterInventoryBootstrap`, extended to run
  the actual retained pthread and its first/repeated OpenMP teams.
- Existing `V2_Integration_CPUStartupAffinity` and `V2_Integration_OpenMPRuntimeBinding`.
- Full `V2_Integration_RetainedHostServiceOwnerReplay` and explicit
  `V2_Integration_RetainedHostServiceOpenMPBudget`.

The budget regression changes the materializer/caller's task settings and
checks actual teams at budgets 1/3 with OpenMP dynamic teams on/off. Each case
uses its own original process authority. Native hostfile coverage proves two
real socket ranks retain their full physical-core budgets and unique places.
Command-policy tests also cover rank counts 1/2/4/8; those are not claims of
native four/eight-socket qualification. The focused regressions belong to
ProductionTestPreflight; the full preflight run remains explicitly deferred.

The new tests caught two candidate/fixture problems before model execution:
revalidation initially substituted the materializer's ICVs, and the initial
fixture attempted several independent process authorities in one child.
Both are corrected. OpenMP directives also live outside the death-test macro
argument so GCC binds them to their intended lexical blocks.

The matched unprofiled original-owner comparison and retirement checks pass:

| Policy | Prefill tok/s | After-prefill decode tok/s |
|---|---:|---:|
| Preceding automatic control | 297.572 | 37.389 |
| Core places and retained task budget | 305.221 | 38.528 |

This is +2.57% prefill / +3.05% decode in one completed comparison. Candidate
requests measure 315.754/295.289/305.307 prefill and 38.375/38.379/38.834 decode.
All 768 token IDs, complete MTP records and the resolved initial expert plan
match. Runtime-configuration differences are only the two output paths.
Dynamic balancing stays enabled: observed wave timing, same-priority moves and
transfer bytes differ, so this does not isolate a pure core-binding effect or
establish a general speedup. The 500/40 target remains open.

Fresh admission passes at context 262,144. Peak observed compute-process RSS
is approximately 36.01/77.07 GiB, versus 36.01/77.11 GiB in the control; these
per-process observations are not summed into physical-memory use or a leak
certificate. Native, driver and process-tree retirement all pass.

## Separate follow-ups

The profile identifies the CPU28 singleton control thread as the actual
`MoEOverlayResidencyMaintenanceService` placement planner, chiefly
`buildMinimumCostPlacement` and `scoreObservedMigrationSet`. Its CPU demand
could contend with a core-bound expert worker. An existing maintenance-only
partition-inheritance patch and original-worker regression remain separate
from this experiment; no benefit is inferred from its sampled CPU time.

Another source audit found an already-empty CPU expert invocation repeats
clearing six pointer tables plus the mask: 24,576 bytes plus mask at 512 experts.
The live-reader release clear is required. Avoiding the subsequent redundant
clear is a separate lifecycle optimization, currently unimplemented/unmeasured.

## Local evidence

- Profile and offline stacks: `/tmp/qfn-openmp-team-binding-next/observed` and
  `analysis/{NOTE.md,attribution.json,principal-roles.json,bracketing-thread-cpu.json}`.
- Focused build/tests: `/tmp/qfn-cpu-eight-row-rocm101-root/openmp-team-binding-*`.
- Original unprofiled control: `/tmp/qfn-current-gpu-capacity-next/automatic-arm`.
- Candidate benchmark: `/tmp/qfn-current-gpu-capacity-next/openmp-core-binding-arm`.
- Redundant-clear audit: `/tmp/qfn-cpu-joint-rebind-audit-next/NOTE.md`.

The profiled lifetime retired successfully and reclaimed 727,912,512 bytes of
owned prefix blobs; the unprofiled candidate reclaimed another 727,912,505 bytes.
Both cleanups use metadata only, without hashing or reading cache payloads.
Large local records remain outside the source repository.
