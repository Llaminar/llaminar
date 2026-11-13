# ROCm 10.1 upgrade and retained runtime contracts

Status: provisional NUMA-correct QFN run and focused SDK/NUMA/native RCCL
gates are closed green. Bounded original-model OpenMP attribution identifies
runtime pause loops; the original recorder receipt remains red for requested
SIGINT. Closed reduced-spin controls regress; GNU wait policy stays unchanged.
Full preflight remains deferred; this
scoped qualification is not a complete SDK/preflight certificate.

The upgrade accompanies the QFN NUMA ownership correction. A process must
execute GPUs on its own observed NUMA node. Updating the runtime is not proof
that the observed caller-affinity widening is fixed; the application boundary
still preserves its calling thread's exact affinity across hardware discovery.

## Release identity

The signed AMD Core SDK repository supplies `10.1.0-3`, SDK version `10.1.0`,
HIP `7.16.26385`. The immutable `therock-10.1` tag points to
`b324869fb3116353ac7eec5e6926aa3d452ac987`; its submodule pins are:

| Component | Source revision |
|---|---|
| rocm-systems | `7da43026e4f2ef7600433c38c820732f7a205d22` |
| rocm-libraries | `39d8d60365d2272a5366f76ccb3bfba24b219eaf` |
| LLVM | `d6f6cb691863e9258034ab48c1dfce21689d5606` |

The staged compiler reports that exact LLVM revision. The canonical source
authority remains `scripts/docker/rocm-release.env`; the kernel driver is a
separate dependency and is not changed by this upgrade.

## Custom 10.0 repair audit

| Repair | Exact 10.1 source | Action |
|---|---|---|
| Native-default host thread stacks, upstream #9643 | Already present; old patch reverse-applies | Retire patch, retain native TLS/graph regression |
| SDMA sharing after exclusive engine assignments fill, upstream #9652 | Already present; old patch reverse-applies | Retire patch, retain oversubscribed-stream byte regression |
| Empty graph segment completion publication | Still missing | Retain patch and native regression |
| Ordering-only SDMA event publication | Still missing | Retain patch and held-frontier regressions |
| Progress-safe graph queue collapse | Still missing | Retain patch and flat/child wait regressions |
| Event producer generation/lifetime | Still missing | Retain patch; refresh one upstream context line only |
| Final graph kernel argument metadata | Still missing | Retain patch and argument-geometry regression |
| Unshared preferred-queue selection | Still missing | Retain patch and actual-body original/candidate regression |
| Parallel-LTO per-partition allocation remarks | Still missing | Retain compiler repair; qualify the changed device driver |

All six retained HIP patches apply in the canonical order to the exact release
source. The staged runtime built successfully and passed the preferred-queue
original-code negative control and repaired implementation check. The original
nineteen-entry HIP/native primitive lane now passes on both shipping targets;
final repaired RCCL qualification is tracked separately below.

RCCL's project-specific collective ABI and its patch chain are pinned separately
from the SDK. They must be rebuilt against the selected SDK and retain their
capture, live-byte, lifecycle, and final resource evidence. rocBLAS must likewise
use the matching source and generate both shipping architectures, `gfx906` and
`gfx1100`; an older SDK's device library is not a substitute.

## NUMA findings

The earlier CLR application-affinity repair
[`d6bd7cb`](https://github.com/ROCm/rocm-systems/commit/d6bd7cb7a0711de173cf7cca6481df4c607013ef)
is already in the custom 10.0 source. The 10.1 host-NUMA virtual-memory APIs
address memory placement, not caller thread affinity.

The newer ROCr helper-thread change
[`840728c`](https://github.com/ROCm/rocm-systems/commit/840728c09779c3723c6dda6c44242e700e7a48b3)
postdates the checked 10.1 release. It keeps helper threads within the creator's
NUMA domains, with explicit exact-mask behavior when the affinity override is
disabled. Both native thread-creation implementations target the new helper,
not its creator. This is relevant to locality, but it is not an established
explanation or repair for the separately observed caller mask widening.

The user authorized backporting other confirmed related repairs. The narrowly
scoped ROCr component now builds against the selected SDK. Its native-source
regression exercises both thread-creation implementations: six of fourteen
original cases fail restricted-NUMA expectations, while all fourteen patched
cases pass. All twenty-eight observations preserve the caller's mask. The
fixture covers exact-mask override, restricted and multiple NUMA domains,
unbound callers, noncontiguous node IDs, and unreadable/error cases. This is a
source-level helper-affinity proof; native inference qualification is separate.

The ROCr image-shader build requires `xxd`, so the compiler-image dependency
closure now declares it alongside `libelf-dev`, `libnuma-dev`, and `pkg-config`.
The runtime image receives only the built HSA library, local aliases, license,
and proof. Both HIP and ROCr component exports pass their existing-installation
validation.

## Compiler integration finding

10.1 defaults HIP relocatable device linking to `clang-linker-wrapper`. The
existing proof rejected its driver output because the former global `-B`
selection no longer establishes that the repaired linker executes. This is a
useful rejection, not evidence that the compiler repair is unnecessary.
Qualification must authenticate the actual device linker and all final
per-partition allocation records without weakening the spill guard.

The integration now uses the new driver's explicit offload-compiler linker
selection. The repaired executable has the canonical `ld.lld` name; a generic
`lld` name makes the inner Clang driver choose the wrong linker protocol.
Qualification follows both wrapper and inner driver commands and resolves
relative remark paths against the original link working directory.

The original/repaired pressure proof passes on `gfx906` and `gfx1100`, with
two and eight partitions per architecture. All sixteen pressure kernels remain
present, and final device-code bytes and kernel resource metadata are unchanged.
The new driver also emits two weak C++ runtime helper functions. Allocation
coverage therefore follows every defined function in the actual ELF symbol
table, including helpers, rather than assuming the kernel roster is the entire
code object. Missing, extra, duplicated, or mismatched records fail validation.
The focused parser/contract suite passes nineteen tests. The actual RCCL link
also needed an explicit eight-partition device-link option. Its option/operand
pair stays grouped through CMake so RCCL's existing offload-linker option cannot
cause deduplication to drop the second forwarding flag. Both native driver
actions authenticate with the corrected option.

The stable `/opt/rocm-sdk` alias exposed another driver-check defect: the
configured wrapper used the alias while the actual driver printed its physical
installation path. Identity now follows the same existing executable file at
each wrapper, inner compiler and native linker boundary, retaining invocation
basenames and original argv. Missing paths, foreign files, duplicate linker
options and a generic `lld` spelling remain errors. Twenty-one focused tests
pass, and the actual RCCL two-architecture driver preview passes with the alias.

The matching rocBLAS source build also completes for both shipping targets.
The prior RCCL build completes for both targets; its final allocation evidence
covers every emitted function. This is complete external-library resource
evidence, not a claim that all upstream RCCL kernels are spill-free. The later
explicit-P2P selector repair changes its receipt; that rebuild, final resource
checks and the nine-entry native RCCL replay now complete successfully.

### Unrequested GVN remark analysis

The first full RCCL link exposed a separate compiler cost. Both architecture
linkers spent roughly ten minutes in GVN's `reportMayClobberedLoad`, reached
through `analyzeLoadAvailability`. Passive stacks from both processes agree;
this was host compiler work, not a GPU or collective stall. The evidence does
not establish an infinite optimizer loop.

GVN's `allowExtraAnalysis("gvn")` accepts the presence of any serialized remark
stream. The build requests only `stack-frame-layout`, but GVN still performs
expensive clobber searches for diagnostics that the stream later discards.
The scoped repair must honor that stream's existing pass filter before doing
diagnostic work, while preserving explicitly requested terminal GVN remarks.
It must not alter GVN optimization decisions or suppress allocation evidence.
Qualification requires an actual GVN-pass witness for selected, unfiltered,
excluded, terminal-only and disabled diagnostics, plus unchanged optimized IR
and the existing device-code/resource proof. The original RCCL inputs and
passive stacks are retained under `rccl-link-diagnostic/` for the relink.

The actual original/repaired GVN witness passes all six diagnostic modes.
For the stack-only stream, original GVN constructs one clobber diagnostic and
the repair constructs none. Requested diagnostics and optimized IR remain
identical. The rebuilt linker also passes the complete `gfx906`/`gfx1100`,
two/eight-partition device-byte and resource proof; this component build and
qualification took 37.8 seconds. The subsequent full RCCL build completed;
configuration through the later test-registration failure took 1,222 seconds.

The repaired full link advances beyond GVN into
`AMDGPULowerModuleLDS::refineUsesAlignmentAndAA`. Later passive samples show
time in alias-metadata domain collection and `MDNode::concatenate`. The
refinement walk is depth-bounded; repeated metadata processing remains expensive.
An exact `(old-noalias, new-noalias)` result cache could reuse the existing
union/intersection result, but actual repeated-pair counts have not been
measured. It must preserve cross-domain unions, same-domain intersections,
alignment updates and traversal. The upstream rerun-no-op change addresses a
later lowering phase, not this sampled initial optimization pass. No LDS patch
is installed: this target pass also needs private/generated AMDGPU headers,
which would expand the compiler qualification beyond the current repair.

### Driver-preview diagnostic serialization

The wrapper's parallel dry-run workers can interleave printed device commands
on one diagnostic stream. Torn argv must not be reconstructed or accepted as
proof. Authentication now adds `--wrapper-jobs=1` only to the existing wrapper
preview, after its original options and before any end-of-options marker. It
retains the original extracted inputs and inner `clang -###` checks. The actual
link argv and eight parallel LTO partitions remain unchanged; this is neither
serialized production linking nor a different compiler owner.

The fresh `lto-build8` original/repaired GVN and two-architecture device-byte /
resource proof passes in **37.282 seconds**. `release-configure14` completes
with exit0 in **1,216.682 seconds**, including the changed-receipt RCCL rebuild;
focused build5 completes with exit0 in **179.489 seconds**. Final host/resource
checks and repaired native RCCL results are green within the scope below.

### Ordinary HIP compilation record routing

The first complete engine build rejected 56 HIP objects because final allocation
records were absent from compiler stdout. The new device-link driver transforms
`-foptimization-record-file=-` into the filename `-.opt.ld.yaml`; parallel
translation units and architectures consequently compete for one sidecar. The
guard correctly rejects this incomplete evidence rather than accepting that file.

A two-architecture compile of the actual `HIPGraphTimelineKernels.hip` reproduces
the failure. Forwarding `-plugin-opt=opt-remarks-filename=-` directly through
`-Xoffload-linker` restores stdout delivery and produces no sidecar. The strict
audit then sees both kernel functions and both emitted runtime helpers on both
architectures. Direct compile times were 3.315 seconds before and 3.097 seconds
after; this is a routing witness, not a compiler performance claim. The actual
ccache miss and direct hit both pass the guard, emit no sidecar and retain
identical object bytes (3.855 and 0.140 seconds). Thirty-seven guard tests pass.
The complete engine/native compiler controls remain required qualification.
Evidence: `compile-record-routing/result.json` under the local evidence directory.

## Legacy SMI dependency isolation

Core 10.1 packages provide AMD-SMI, while the separately pinned RCCL source
still uses the legacy `rsmi_*` ABI. Its additive CMake package search selected
`/opt/rocm/lib/librocm_smi64.so.1.0` from a different SDK. The exact 10.1 systems
revision still contains `projects/rocm-smi-lib`, so that unchanged component is
now built into the selected SDK instead of borrowing an older binary or
rewriting RCCL's topology implementation. Its library exports SONAME
`librocm_smi64.so.1`, depends only on system libraries, and its CMake exports
resolve relative to the installed SDK.

RCCL discovery now requires that SDK's package and rejects both a foreign
warmed package cache and imported library locations outside its public library
directory. Runtime loader tests also require HIP and ROCr to resolve from the
same physical SDK root; independently valid receipts from two installations
cannot certify a mixed process.

The backend's five direct HSA interop opens now use CMake's exact SDK-owned
ABI-1 library path. A bare `libhsa-runtime64.so` name could otherwise resolve
through an older system loader entry. Five focused, device-free CTests pass:
actual configured and original call-site controls, plus SDK discovery coverage
for missing libraries, conflicting search paths, warmed caches and escaping
symlinks. Native loader and inference qualification remain separate.

## Focused qualification and repaired test boundaries

The canonical Release test-owner selection originally hid requested custom
proof targets behind `NOT V2_PERF_TESTS_ONLY`, and a later functional-only
wrapper hid original HIP/ABI declarations. The repair retains the original
owners and per-owner alias filtering: explicitly marked proof utilities are
admitted, shared RCCL build authority is published before consumers, and only
the affected declarations are moved outside the wrapper. Script-only checks
use their explicit original registrations. Missing requested owners are reported
together. This does not enable the full preflight or replace native owners.
The original focused inventory reached all 96 expected aliases before the new
P2P-selector alias was added.

The protocol-default proof also assumed that hipify prepended its runtime
include at byte 0. Current 10.1 output retains the source header and places the
include before the other includes. The exact original-code patch reversal now
preserves either observed layout, rejects missing/malformed includes and still
compiles the real native/original consumers. The repaired original proof owner
and focused build2 complete in 171.005 s; the original forward/reverse/concurrent
controls fail the geometry oracle as required, not by crashing.

The real HIP loader positives authenticate both the application and original
argument-geometry owner under inherited and clean/default environments. The
shadow negative control initially rejected a valid DT_RPATH outcome: the loader
prevented replacement, so the authenticated original remained loaded. It now
first proves the supplied fixture's real foreign SONAME and accepts only the
exact foreign fixture rejected by the validator, or injection prevention with
the identical authenticated physical clean-baseline runtime. Missing/ambiguous
bindings, another foreign file, another authenticated copy and stale receipts
remain failures. All 16 Unit controls pass; the two actual Unit/integration
registrations pass in 5.879 s.

Closed focused results under `/tmp/qfn-rocm101-upgrade/gates/`:

| Original gate | Closed result and scope |
|---|---|
| Final host contracts | **53/53 pass, 5.105 s**, including loader, resource evidence, selector, receipt and Release-registration contracts. No missing, extra or nonpassing entries. |
| Focused NUMA host owners | **13/13 green** in the completed scoped qualification. |
| HIP/primitive lane, actual gfx1100 | 19/19 pass, 60.453 s; logical device 0 remapped to physical ROCm 0. |
| HIP/primitive lane, actual gfx906 | 19/19 pass, 62.050 s; logical device 0 remapped to physical ROCm 1. |
| Full-visibility inventory/bootstrap | 4/4 pass, 6.525 s: single-rank and MPI owners on both vendors. |
| Repaired native RCCL gfx906 cohort | **9/9 pass, 36.549 s**, including all four original HostTransportStorage parameter cases and normal-P2P captured/live-byte replays. |

Standalone device remaps establish the specified native semantics on each ISA;
they do not replace full-visibility production NUMA admission or prove model
performance. The repaired RCCL gate uses the four physical gfx906 devices;
these results do not imply a multi-gfx1100 cohort or full production preflight.

### Explicit RCCL P2P selection

The original HostTransportStorage regression requests `NCCL_P2P_DISABLE=1` to
exercise native SHM storage. Actual RCCL `paths.cc` reads that user selection,
then Intel/ARM/Zhaoxin CPU defaults overwrite it with `PATH_PXB`. The real native
diagnostic shows disable=1 but P2P/direct-pointer connectors, so the zero SHM
connection receipts correctly expose a selector defect. The fixture assertions
and storage implementation are unchanged.

The narrow vendor repair applies the existing explicit user selection after
unchanged CPU defaults. Unset/default behavior and hardware peer checks remain
intact; both disable and explicit level retain their actual parser semantics.
Its patch joins the canonical chain and both Docker/local receipt producers.
The original selector proof owner now includes actual complete `paths.cc` and
`topo.cc` consumers plus the exact original-code reversal; each explicit-policy
cell runs in a fresh process to preserve native static caching. The original
collective-selector checks remain. The actual native P2P proof completes all
**11 policies / 2,640 checks with zero failures**; the preserved original body
has exactly **96 failures**. The fresh final host/resource gates and unchanged
native storage replay pass. The four TP2/TP4 × CreatorFirst/PeerFirst storage
cases and all nine native cohort registrations execute successfully. The prior
real red diagnostic remains evidence of the original wrong connector choice.

Two proof-integration follow-ups retain the original consumer boundaries. The
completed-Broadcast receipt parser formerly required a literal leading
`targets-` body; the canonical producer now prefixes the exact current
`p2p-<patch-hash>-` identity. That consumer authenticates and removes only the
exact current prefix, then applies its unchanged targets/completed/capture
checks and full receipt/DSO equality. Missing, foreign, duplicated or reordered
prefixes remain errors; no generic prefix or rewritten receipt is accepted.

The new host-only P2P probe initially omitted `ncclDebugNoWarn`, the exact
thread-local logging variable declared by native `debug.h`. It now supplies
that state beside its existing probe-local logger. The real selector and level
parser remain compiled from the complete native translation units. The final
proof reaches every original and repaired policy case; no selector behavior is
substituted to resolve the missing logging symbol.

## Closed provisional NUMA-correct model

The original admitted 512-prompt/256-output workload closes with **300.092 prefill
and 38.357 after-prefill decode tok/s**: one warmup, three measured iterations,
context 262144, dynamic MTP/balancing and unchanged original cache/expert policies.
All **768 generated IDs**, complete per-iteration MTP work and initial expert
placement match the retained r1151 control exactly. Native/driver exit0, no new
driver findings, complete observations and full original-process retirement are
recorded. These are provisional rates, not a speedup comparison: the historical
reference used invalid remote GPU ownership and cannot certify placement gains.

The closed comparison's only unhandled configuration leaf was the explicit
`config_file_path`. Its exact path and UTF-8 document are already bound by the
original admission receipt; authorizing only that formerly blank leaf leaves
zero unexpected runtime-configuration differences. Initial placement matches
147 records on execution-rank 0 and 49 on rank1.

Passive thread observations show PID 1132815 restricted to NUMA 0 and PID 1132816
to NUMA 1. However, this benchmark does not publish native rank→PID identity.
Nonzero DRM bytes on all five cards include HIP discovery allocations and
shared host mappings; they must not be treated as GPU execution ownership.
No PID ordering or byte threshold fills the missing join. Native inventory /
strict ownership gates are independent evidence, not a substitute passive join.
A future benchmark observation can reuse the existing native process-identity
and published-inventory APIs, preserving startup records across the benchmark
reset; no new ownership ledger or runtime decision authority is required.

Per-process peaks, sampled measured-loop deltas and all ten unsummed DRM clients
are retained in `/tmp/qfn-numa-rocm101-closed-evidence-next/`. After full original
retirement, the exact run-owned prefix archive was inventoried and removed:
727,912,503 logical bytes, 16 payload entries plus metadata, 28,679 metadata bytes
read and **zero payload bytes read**. No SDK/model/shared cache was reclaimed.

### Bounded OpenMP wait attribution

The original complete routed FFN already executes gate/up, SwiGLU/Q8 and down
inside one team. The closed `observed2` profile authenticates both actual native
ranks loading the exact same GNU libgomp. Two complete measured Prefill/Decode
pairs use 16M user-cycle sample periods and period 1 phase events on one monotonic
clock. Exclusive process user-cycle share is about **59–68% Prefill /76–82%
Decode** in libgomp; 99.54–99.77% of that runtime share falls in two saved
`pause`/poll loops. These are resource shares, not wall-time savings or “80%
OpenMP overhead”; concurrent necessary CPU/GPU/peer progress remains possible.

The original recorder helper reports red because requested SIGINT exits with
-2; its receipt is preserved. A separate actual-recorder control confirms that
this requested shutdown retires the group and flushes the recording. Offline
analysis retains only the two complete phase pairs and reports no lost/unparsed
records; it does not turn that original receipt green.

[Detailed results, exact cohorts and limits](QFN_CPU_OPENMP_WAIT_ATTRIBUTION_20261109.md)
retain runtime/process identities, phase durations, cycle periods, hot-loop
addresses and control receipts. Closed matched controls retain GNU's unchanged
wait policy: fresh unset **299.559883 P /38.993654 after-prefill D** versus
GOMP_SPINCOUNT=0 **199.393089 /29.381258** and 1000 **203.444590 /32.726287**.
All 768 IDs/full MTP/config/initial placement match; native closure checks pass,
and actual waiting inputs are verified in both ranks. Dynamic movement can differ, so no
identical-route or isolated wakeup-cost claim is made. All three roughly 0.728 GB
owned archives were reclaimed by metadata-only inventory after retirement.
No cycle percentage is converted into removable elapsed milliseconds.

## Qualification boundary and local evidence

Local work is under `/tmp/qfn-rocm101-upgrade/`. The workspace SDK alias now
selects the staged 10.1 closure for qualification; the original 10.0 installation
is retained. The global loader configuration contains only the SDK's public
library directory, excluding compiler-private LLVM libraries that could replace
the selected OpenMP runtime. Fresh `rocminfo` sees the 7900 XTX and all four
MI50s; both RTX 3090s remain visible with driver 595.99.02. These discovery
observations do not replace native graph or inference tests.

The evidence directory contains signed package
metadata, exact selected package records, compiler inputs, build commands and
results. `/tmp/qfn-rocm101-upgrade-review/` contains the patch audit and upstream
source comparisons. These are local evidence, not published corpus payloads.

Required focused checks include packaging/loader binding, both shipping
architectures' compile evidence, native HIP graph/event/SDMA regressions, RCCL
capture and lifetime contracts, NUMA ownership and affinity controls, and the
matched QFN production workload. The full production preflight remains deferred
for this iteration at the user's direction. No performance improvement or
complete production qualification is claimed by the source audit alone.

Primary release reference:
[AMD ROCm 10.1 release notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html).
