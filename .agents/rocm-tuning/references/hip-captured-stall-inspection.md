# Passive inspection of captured HIP stalls

Use this procedure when a retained production graph stops progressing while
one or more ranks wait for a collective, graph terminal event or maintenance
ticket. Keep the same model, saved placement, request sequence, prefix-cache
policy and native dependencies. Do not disable capture or serialize builders.

## Preserve the uninstrumented failure

1. Retain the exact cell, argv, request/response artifacts, driver checkpoint,
   server log and owning MPI process group. Record which HTTP operation last
   progressed; a subsequent event wait may only be the observer of the fault.
2. Identify the compute PID, not the `mpirun` wrapper. Validate its executable
   and ownership before attaching. Run only one accelerator job on the host.
3. Read `/proc/<pid>/maps` to record the actual HIP and RCCL paths. Authenticate
   dependency revisions and patches from their installed receipts. An SDK
   version string alone cannot distinguish repaired and distribution DSOs.
4. Take ordinary host stacks first. Attach ROCgdb only after the stall; attach,
   profiling and synchronous logging can perturb or conceal the race. Treat
   these runs as diagnosis, never throughput or correctness certificates.

Example passive snapshots, after assigning a validated decimal compute PID:

For a distributed ticket timeout, inspect the continuation rank as well as the
waiting follower. Idle device workers plus a request stack in disk prefix-cache
eviction or filesystem writes is a host-work boundary, not evidence of a HIP
queue cycle. Correlate several bounded snapshots and exact write sizes before
changing native graph ordering. Prefix archive compaction must use its typed
background worker; it may not rewrite the active archive under foreground
index locks. Its focused `V2_Integration_PrefixArchiveBackgroundPublication`
preflight entry checks retained-inode restores and concurrent committed tails.
Keep the standard collective deadline; a larger timeout is not a storage fix.

For a storage producer, distinguish bounded scratch memory from bounded dirty
filesystem pages. A successful `sync_file_range` does not prove physical
writeback: on Linux OverlayFS its logical `file->f_mapping` can contain no
pages while writes go to a backing inode. Inspect the serving process's actual
mount and descriptor, not only the cache pathname or syscall return values.
The archive maintenance writer validates actual `O_DSYNC` descriptor flags and
performs bounded writes on its existing background worker; final file and
directory durability remain separate publication edges.

If passive BPF inspection is warranted, import the running kernel's complete
`struct file` from a typed `filp_close` kfunc, resolve the active syscall's fd
through that task's real fdtable, and record only inode size, filesystem magic
and `f_mapping->nrpages`. Correlate the descriptor with `/proc/PID/fd` and the
active compaction/append method. Do not guess private layout offsets or read
cached payload bytes. Use `bpftrace -B line` for small redirected observations:
default stdout buffering can hide a valid observation until the probe exits.
Compare pointers explicitly with zero; some bpftrace versions reject a pointer
used directly as an `if` condition. Failed compilation/attachment is missing
evidence, never evidence that the native operation did not execute.

```bash
sudo timeout --signal=INT --kill-after=5s 25s gdb -q --batch \
  -p "$LLAMINAR_STALLED_PID" \
  -ex 'set pagination off' -ex 'thread apply all bt 12' -ex detach

sudo timeout --signal=INT --kill-after=5s 20s /opt/rocm/bin/rocgdb -q --batch \
  -p "$LLAMINAR_STALLED_PID" \
  -ex 'set pagination off' -ex 'info agents' -ex 'info queues' \
  -ex 'info dispatches -full' -ex 'info threads' -ex detach
```

Save output under the failed cell's ignored artifact directory. For instruction
attribution, select a small representative set of the reported GPU waves and
inspect their program counters/disassembly. Never single-step, set breakpoints,
call inferior methods, read or modify model tensors, or resume an alternative
execution path. Diagnostic attachments are not a replacement scheduler.

Collect host stacks before loading private type helpers when a bounded callback
must distinguish host prefix work from a GPU wait. GDB `-ex` settings execute
after its initial attachment work: use `-iex` to suppress thread announcements,
SDK debug-script autoloading and network debug discovery beforehand. Keep
ordinary shared-library discovery enabled for the native unwind inventory.
Restricted symbol loading and `-readnever` can produce truncated or misleading
backtraces with some debugger/dependency builds; a native progress frame followed
by guessed stack addresses does not identify the blocking operation or prove
application stack corruption. Validate the intended settings against a bounded,
device-free process stopped in a known call such as `MPI_Bcast`: the backtrace
must reach that API and its known application caller.

For the HTTP hammer's ten-second callback budget, collect both authenticated
ranks concurrently with a smaller per-debugger deadline. Print the main-thread
stack before all-thread output so optional worker frames cannot consume the
budget needed for command/collective attribution. For example, after PID and
birth/ancestry authentication:

```bash
sudo -n timeout --signal=INT --kill-after=1s 6s gdb -nx -nh -q --batch \
  -iex 'set pagination off' -iex 'set print thread-events off' \
  -iex 'set auto-load off' -iex 'set debuginfod enabled off' \
  -iex 'set auto-solib-add on' -p "$LLAMINAR_STALLED_PID" \
  -ex 'thread 1' -ex 'bt 24' -ex 'thread apply all bt 20' -ex detach
```

Collect private queue/signal layouts separately if these host stacks warrant
it. Any manually loaded application/type object still requires its concrete
ELF identity and load bias from that exact process's zero-offset maps entry;
never reuse an address from another run. Preserve both discovery and
execution-rank identities: automatic planning can reorder membership, so a
discovery rank number does not identify the HTTP continuation or follower role.

If automatic library-symbol discovery consumes the callback budget before the
first stack, retain native ELF/unwind discovery but load relevant modules
individually. An unescaped `|` alternation in `sharedlibrary` matched no modules
on the validated GDB and produced bogus MPI frames; individually matched
modules reached `PMPI_Bcast` and its known caller in a device-free control with
126 idle workers per rank and the actual Release DSO preloaded. Do not accept
a fast trace whose application/native call chain cannot be authenticated.

```text
set auto-solib-add off
sharedlibrary libllaminar2_core
sharedlibrary libmpi
sharedlibrary libopen-pal
sharedlibrary mca_
sharedlibrary libevent
sharedlibrary libpmix
sharedlibrary libstdc
sharedlibrary libc[.]so
sharedlibrary libgomp
```

Add the actually mapped native libraries needed by that stack, such as
`libamdhip64`, `libhsa-runtime64`, `libcuda`, `nccl`, `libhipblas` and
`librocblas`, using separate commands before printing frames. Do not load
compiler/JIT debug inventories merely because they are mapped. Keep the
complete callback within its ten-second budget, including authentication and
detach; validate this procedure against known calls on the installed toolchain.
Save `ps -L -p PID -o pid,tid,stat,wchan:48,comm` before ptrace, so even a
timed-out attachment retains host wait-channel evidence. Missing rank stacks,
a missing detach receipt or truncated output are incomplete diagnostic
observations, not successful callbacks or evidence of application corruption.

## Separate queue metadata from model data

The native HSA queue ring, dispatch descriptors, immutable kernel arguments,
signal objects and host transport-control structures are CPU-addressable
protocol metadata. Inspect them through matching DWARF layouts and CPU threads.
Do not treat a GPU dispatch's default address space as host memory. Loading a
small unlinked debug object can expose optimized-out private layouts; it must
use the exact dependency source, headers, defines and compiler ABI. Keep it
outside production binaries. `sizeof` alone may not emit a usable DWARF type.
Resolve optional component headers through `cmake/ROCmSDKHeaders.cmake` too:
a cached CK include directory from another SDK is not a valid tuned baseline.
The SDK compiler-binding and SDK-header-discovery preflight entries prove the
actual compile database and warmed-build component lookup, respectively.
For GCC, use retained typed pointer declarations and `-g -O0
-fno-eliminate-unused-debug-types -femit-class-debug-always`: pointer declarations
without the last flag may still leave polymorphic owners as incomplete
declarations. Verify `ptype` and `sizeof(*pointer)` in a separate debugger before
attaching to a live failure.

For LLVM/Clang, derive the helper's compile command from the native runtime's
actual compilation database (`ninja -C <runtime-build> -t compdb`). Preserve
the dependency's compiler, defines, include roots and ABI; replace only the
input/output and dependency-file destinations, select `-g -O0`, and add
`-fstandalone-debug -fno-eliminate-unused-debug-types`. A header-only imitation
compiled with the application's flags can produce incompatible layouts. Native
types also change across SDK releases: Core 10 separates `GraphExecBase`,
`GraphExecClassic` and `GraphExecSegmented`, whereas older helpers may name
`GraphExec`. Rebuild the unlinked helper when the dependency identity changes.
Quote qualified type names in debugger expressions, for example
`ptype /o 'amd::roc::VirtualGPU'`; otherwise GDB's namespace parser may report
a missing type even when complete DWARF is present. Never link or execute this
helper in the service.

For each native queue, retain:

- Device/agent and physical queue identity, not just the HIP logical stream.
- Read/write indices, validated power-of-two ring capacity and packet indices.
- Packet type/header, barrier dependencies and completion-signal identity.
- Kernel entry and immutable argument identity/geometry for the blocked front.
- CPU signal kind/value from the actual HSA layout, not a guessed offset.

Read only the validated live ring extent plus a small preceding window; never
scan an unbounded address range or infer a queue's capacity from an unrelated
allocation. Ring wrap requires the native capacity mask. A packet invalidated
by the command processor may already be fetched but not completed: neither a
header alone nor a read index proves device execution or a dependency cycle.
Native producers can reserve before waiting for space, so write-minus-read may
exceed physical capacity without those extra reserved bodies being published.
Bound inspection to one physical ring; never label aliased future reservations
as live packets. Check the whole-batch admission bound against the native
single-packet/barrier contract, especially equality at the vacant-slot boundary.

Match collective pairs by communicator generation, sequence, actual counts,
data type, operation, channel geometry and consumer ownership. A GPU polling
LL flags while its peer has no executing dispatch can be an upstream graph or
packet-publication problem. It is not sufficient grounds to alter transport,
add stream synchronization or remove the collective.

For an asymmetric LL gather, retain the preceding dispatch arguments as well
as the currently executing dispatch. One participant reaching the next gather
does not prove that its preceding operation used the right live count. Match
those arguments to the actual retained kernel-node parameters and dependency
edges, recursively inspecting child graphs when necessary. Decode register
roles from the exact active entry's ISA, not another protocol specialization.
Inspect all wave lanes and the EXEC mask: a default debugger array truncation
can hide the upper half of an active wave and falsely suggest that all polled
flags match. Transport flags and step counters are protocol metadata; model
activation/weight payloads are outside this passive inspection.

Audit graph boundaries separately from internal edges. Every independent root
must consume the launch stream's incoming frontier, and completion must join
every executed branch before a dependent graph can communicate on the same
communicator. A host-submission order or an apparently correct node count is
not proof of either contract. Reduce a suspected boundary defect to a native
model-free retained-graph test with sticky device receipts before changing the
production ordering policy.

## Packet publication and signal lifetime

Each AQL packet's valid header is its publication point. The command processor
can prefetch interior packets, so holding only the first header invalid is not
a sufficient body-publication contract. Native batching must copy every body
with an invalid header, then release-publish complete interior packets and the
first packet last, followed by the end-of-batch doorbell. Preserve kernel,
barrier, completion-signal and system-scope semantics.

Keep publication faults distinct from signal reuse, queue aliasing and a real
producer/consumer cycle. A pending event or a zero signal value does not alone
prove any of those. Inspect the actual generation and owning graph before
adding a lifecycle transition. Prefer a narrow repair in the sole native
authority to another host mirror, polling controller or replay layer.

If kernel logs name a USERPTR eviction/restore worker, collect a separate,
bounded `perf` trace of `amdgpu_amdkfd_evict_userptr` and
`amdgpu_amdkfd_restore_userptr_worker`. Resolve stripped module symbols with
`perf probe --module` and the matching host `amdgpu.ko`; an unqualified probe
may mistake a relocated module symbol for an out-of-range kernel address.
Record monotonic timestamps and kernel call chains, then correlate with the
same server's startup, prefix operations and teardown. Run the server as its
original user, with its original interpreter and explicit environment, even
when the observer needs sudo. Remove only the probes created by this diagnostic
before uninstrumented certification. A zero-event trace means the observed
failure did not reproduce; it does not establish that all future registrations
are immune. Do not allowlist CPU-hog, IH overflow or other driver warnings.

### Attribute USERPTR work to its actual memory owner

Eviction workers can run in `kcompactd` or a kernel workqueue rather than the
inference process. The sampled task PID is therefore not the allocation owner.
For a bounded diagnostic, derive the probe arguments from the **running host
module's** ABI and record the interval/notifier identity, registered range and
owning process alongside the invalidation reason. Do not copy structure offsets
from another kernel or infer ownership from the nearest HTTP request timestamp.

Match the loaded module's build identity before inspecting its ABI. Running
kernel BTF (`/sys/kernel/btf/vmlinux`) supplies common notifier/range layouts;
the exact module's disassembly supplies module-private owner links. If the
distribution `bpftool` wrapper cannot select the running kernel, an installed
versioned tool may still read that BTF format; verify the decoded fields rather
than substituting headers from a different kernel. Derive all probe offsets
again for that host. Map a recorded owner PID through `NSpid` when the service
runs in a container; the perf worker's task PID and the container's application
PID are not interchangeable allocation identities.

Include `amdgpu_hmm_invalidate_hsa` when it exists in that exact module, together
with the eviction and restore functions. Retain kernel call chains and task
events. A chain through `try_to_migrate`, `rmap_walk_file`/`rmap_walk_anon`,
`migrate_pages`, `compact_zone` and `kcompactd` distinguishes memory compaction
from application unregister/free; `MMU_NOTIFY_CLEAR` alone does not do so.
Interpret reason constants against the matching kernel headers.

Before the owner process exits, save its `/proc/<pid>/maps` and identify which
mapping overlaps each registered interval. Normalize inclusive registration
ends to the maps' half-open range convention. Deleted `/dev/shm/nccl-*` mappings
are still live mapped files, not proof of freed memory. Correlate registration
and unregister sizes, flags and caller stacks with the configured RCCL/HIP
paths; do not attribute a transport mapping to a prefix-cache buffer merely
because both are host memory. Startup and teardown must be inside the trace.

For an already recorded trace, these read-only commands retain the event
definitions and task attribution without another inference run:

```bash
sudo perf evlist -v -i "$LLAMINAR_USERPTR_TRACE"
sudo perf script --show-task-events -i "$LLAMINAR_USERPTR_TRACE"
```

Check lost-sample reports and observation boundaries before counting events.
Remove only this diagnostic's named probes, not the whole host probe namespace.
Keep instrumented results separate from uninstrumented full-HTTP certification.

If the repair changes RCCL storage, prove the concrete connection kind through
`V2_Integration_RCCLHostTransportStorage`, not correct output alone. Same-process
native pinned backing and cross-process file/cuMem IPC have different ownership
contracts; a raw pointer must never be used as cross-process IPC. The passive
receipt counts endpoint mappings, not unique physical allocations or admission
capacity. Exercise captured partial live extents and both retirement orders,
then resume the exact HTTP cell's fresh-process stability loop.

The canonical installer is `scripts/docker/install-hip-graph-runtime.sh`.
Runtime repair reuse authenticates every patch and the installed DSO; both
shipping ISA images consume the builder's same native library. For an isolated
runtime A/B, place the matching standard-SONAME DSO first in `LD_LIBRARY_PATH`
and verify `/proc/<pid>/maps`. Do not substitute a function interposer for an
unmodified-runtime control. The matching SDK/HSA dependencies remain required.

### Isolate native scheduling collapse

For a graph with an independent waiter and producer, inspect native queue
assignment as well as the declared graph edges. A finite-work cost heuristic
must not place the waiter ahead of its producer on one in-order queue. In
segmented HIP graphs, inspect `GraphExecSegmented::ShouldCollapseToSingleStream`
and distinguish finite work from typed memory/event/semaphore waits, callbacks
and opaque children. A topologically valid parent alone does not prove progress.

Use `tests/v2/repro/rocm/hip_graph_wait_progress.hip` for the model-free
child-graph control. Build with both roots bound to the selected SDK:

```bash
LLAMINAR_REPRO_DIR=$(mktemp -d)
"$ROCM_PATH/bin/hipcc" -O3 -std=c++20 \
  --hip-path="$ROCM_PATH" --rocm-path="$ROCM_PATH" --offload-arch=gfx906 \
  tests/v2/repro/rocm/hip_graph_wait_progress.hip \
  -o "$LLAMINAR_REPRO_DIR/hip-graph-progress"
```

Compare that same executable against an unmodified matching runtime and the
candidate DSO, verifying the loader identity. Its nonempty children separate
collapse/progress from empty-segment completion. The bounded cancellation is
negative-control cleanup only: the failing result is recorded before releasing
the wait. A scheduling override such as `DEBUG_HIP_GRAPH_MIN_OVERLAP=0` may
isolate the heuristic, but cannot certify a repair or remain in production.
`V2_Integration_HIPGraphWaitConcurrency` adds flat/child and 32/64-bit coverage.
Preserve finite-work optimizations; measure the repaired runtime separately
before claiming that correctness also preserves throughput.

## Prove the repair

Run the model-free `V2_Integration_HIPGraphBatchPublication` preflight test.
Its all-kernel graph exercises captured packet batching, long independent
branches, queue wrap and ordered launch-stream changes. Per-node sticky
receipts detect earlier omitted, duplicated, reordered or corrupted work even
if later replays appear correct. Empty/event-node fixtures can select another
native scheduling path and do not substitute for this test.

Run `V2_Integration_HIPGraphLaunchOrdering` for both native stream frontiers.
It delays a GPU-owned producer and checks sticky receipts on every independent
root, through same-stream work, an external event wait and a parent-to-child
dependency. Widths beyond the physical queue count also exercise queue reuse.
Unlike a single-root fork/join, this exposes auxiliary roots that bypass an
earlier producer even when all graph-internal edges are correct. Its delay is
bounded diagnostic pressure, never a production kernel or a throughput sample.

The exit sweep delays every root in turn at widths 4, 8, 14, 16, 30 and 32,
both directly and inside a retained child. A same-stream successor writes its
sticky missing-root result on device before terminal readback. Small widths
alone do not prove completion when native command maps rehash and independent
segments reuse a FIFO. Trace each queue's actual last submitted command into
the final join; dependency level and unordered-map iteration are not submission
order. The returned nested-graph frontier must name that exact joined command,
not an earlier segment on the launch queue. Teardown of a negative runtime
control may explicitly retire the device only after its immutable failing
result exists; such synchronization must never become an inference repair.

An intermittent defect need not fail on every small test invocation. Separate
functional coverage from an authenticated before/after reproducer; do not
claim a passing old-runtime run establishes release immunity. For release
attribution, compare the exact affected routine and installed dependency
revisions, not only the current application diff.

After a candidate repair, rerun the exact full HTTP cell with `--repeat 20
--fail-fast` through `scripts/ci/run_model_parity_e2e.py`. Every iteration must
start and retire a fresh server and validate its long-context, prefix-cache,
movement, tool, memory and driver evidence. Retained immutable staged model
files may be reused. Warm-server request loops are diagnostic evidence only.
Run Unit once per changed slice and production preflight serially against the
same fixed dependency build; never pay those gate costs for each cell. Resume
the full HTTP inventory only after the focused defect is stable. Keep dated
failures and measurements in project documents, not this workflow.

## Independent DMA completed, but its event is still pending

Inspect the exact copy stream's tracker signal and the exact recorded event's
signal separately. A zero native SDMA signal certifies the submitted copy, but
an event-record compute barrier can still be pending behind unrelated held
work when logical streams share a physical AQL queue. Authenticate both stream
handles, physical queue identity and the signal values from the same paused
snapshot. Correct bytes observed after releasing a hold do not prove progress.

`tests/v2/repro/rocm/hip_sdma_event_progress.hip` is a standalone, model-free
reproducer. It uses ordinary NoCU copies, timing-disabled events and a retained
native wait/write graph. Build it with the selected SDK's `hipcc`, C++20 and
the actual architecture; do not change queue limits, priorities, runtime
heuristics or copy engines to obtain a pass. The first argument selects 0–64
held graphs. `--wait-frontier` and `--compute-frontier` are negative controls:
those streams have genuine post-copy dependencies and must remain pending.
All scenarios verify live byte extents, guards and repeated event generations.
The explicitly registered `HIPSDMAEventPublication_*` preflight entries drive
the same source, so the upstream reproducer and production gate cannot drift.

Ordering events should retain a native DMA receipt when that receipt covers
the complete logical stream frontier; pending cross-queue dependencies, cache
work and compute invalidate that substitution. Native signal references must
outlive every observer and prevent reuse while an event still names them.
Keep real timing and capture semantics separate. Llaminar's worker-event API
defaults to `GPUEventPurpose::Ordering`; profiling explicitly selects `Timing`.
When auditing callers, enumerate every `eventElapsedTime` consumer before
changing creation semantics. Prove both policies against actual CUDA/HIP
handles, not just a source scan or the mocked policy.

Stripped shipping DSOs need not expose `hip::g_devices`. A source-bound DWARF
type helper describes layouts, not symbol addresses. Supply only native stream
handles authenticated from the stopped process's own objects or retained
diagnostic output. Never transplant a global address from another ELF build,
guess opaque pointer offsets, or invoke native APIs in the stopped inferior.

## Trigger snapshots without token tracing

Token tracing can introduce device readbacks or enough host delay to conceal a
race. A CPU-only uprobe on the exact published-terminal boundary is a narrower
observer. Resolve the symbol from the serving build, not a different Integration
library. This example works with bpftrace 0.20 and avoids newer map-iteration
syntax:

```bash
LLAMINAR_STALL_CORE="$(readlink -f build_v2_release/libllaminar2_core.so)"
LLAMINAR_STALL_SYMBOL="$(nm -D --defined-only "$LLAMINAR_STALL_CORE" |
  awk '$3 ~ /GraphSegmentCache.*waitForPublishedCaptureStreamTerminal/ {print $3}')"
sudo bpftrace -e "
uprobe:${LLAMINAR_STALL_CORE}:${LLAMINAR_STALL_SYMBOL}
{ @entered[pid, tid] = nsecs; }
uretprobe:${LLAMINAR_STALL_CORE}:${LLAMINAR_STALL_SYMBOL}
{ delete(@entered[pid, tid]); }
interval:s:1
{ printf(\"OBSERVER_NOW %llu\\n\", nsecs); print(@entered); }
"
```

Require exactly one resolved symbol. Parse the latest `OBSERVER_NOW` frame,
not an earlier pending call that has since completed; three seconds is a useful
diagnostic trigger before the ordinary 30-second fatal boundary. Suppress the
automatic attachment while validating the inspector manually. BPF PID/TID
values can be in the host namespace even though `/proc` is container-scoped.
Authenticate local executable, process ancestry and MPI rank independently;
never pass an unmapped BPF PID to a container debugger.

### Do not mistake a storage wait for a native graph stall

Save `ps -L` states and wait channels before ptrace. A `D`-state disk worker
can prevent GDB's all-stop attachment completing even when every other thread
can be stopped. Preserve `/proc/<pid>/task/<tid>/syscall`, kernel stacks, maps
and descriptor pathnames with the same authenticated process/birth identity.
An intentionally skipped or empty debugger stack is incomplete evidence, not
a successful inspection. Do not lengthen the production protocol timeout.

Observe the continuation authority's CPU boundaries alongside the follower's
ticket wait: public prefill, prefix lookup, required RAM-capacity receipts,
archive verification/hydration, append and compaction. Paired entry/return
uprobes retain only active calls and their CPU object identity. Derive private
mutex offsets from the serving build's actual compiler flags, and correlate
the exact futex address with that same object's pointer. Validate the observer
first with a device-free archive control holding a separately opened advisory
lock; it must detect the predicted writer/reader dependency and verify exact
restored bytes after releasing the owned lock.

Include retained archive descriptor retirement when lookup outlives hydration.
Closing the last descriptor of an unlinked compacted inode can perform native
filesystem reclamation before libc returns. Observe the ticket's CPU release
boundary, its exact close and inode-eviction identity/extent. Kernel task-work
can run after the syscall-exit tracepoint but before the user-space return:
that tracepoint alone does not bound retirement latency. Validate inode
observation with a tiny actual archive replacement, not a guessed address or
a large synthetic deletion. Resolve kernel structures through the running
kernel's BTF; older bpftrace versions may need a typed `kfunc` argument to load
the complete definition rather than an fdtable's forward declaration.

These probes and debugger attachments are diagnostics. Their elapsed times
are not uninstrumented latency or stability certificates. Do not attribute a
model timeout from a separate lock control, a transient sampled frame, a
historical active-call entry, or the follower's waiting stack alone.

If boundary probes themselves conceal the failure, use them only during startup
to collect `GraphSegmentCache` identities, then detach and reap bpftrace before
the suspect request. A separate diagnostic process can poll the existing
CPU-owned `terminal_fence_published_generation` and
`terminal_fence_observed_generation` fields with `process_vm_readv`. This must
not read model/device state, call the inferior, or change an inference decision.
Compile `offsetof` and the cache size from the serving build's actual header,
compiler and feature flags; copied numeric offsets are not portable evidence.
Authenticate the executable, `/proc/<pid>/stat` start identity and harness
ancestry before reading an address, and recheck identity during polling. Cache
retirement can make an address unreadable; that is not a terminal stall.

Read both counters, then reread publication and reject a changing/torn sample.
Only a stable pending generation sustained through the diagnostic threshold
triggers an attachment; a normal observation resets that threshold. Do not
interpret an earlier pending generation as the current transaction. Log probe
retirement explicitly and keep the observer outside the server process. These
diagnostic runs still do not replace the uninstrumented twenty-lifetime gate.

For bpftrace versions that expose only host PIDs, the running kernel's BTF can
resolve the current namespace ID through
`task_struct::group_leader->thread_pid`, `pid::level` and its `upid` array. Use
BTF-derived `offsetof`/`sizeof` and pointer arithmetic if that version rejects
dynamic array indexing. Validate the result against a harmless known process
before attaching to serving ranks. Do not infer namespace equivalence from a
single-field `NSpid`, or hard-code another kernel's structure layout.

Use ordinary host `gdb` for thread stacks and CPU-addressable native
queue/signal metadata. It does not reserve AMD GPU debug trap IDs. Validate
the inspector with this host-only route first. If wave/dispatch information is
needed, detach the host inspector before using `rocgdb`, and avoid overlapping
ROCm debugger sessions. Debugger-induced trap-reservation errors are not a
clean production driver certificate and must not be allowlisted.

For a stripped application, `GPUDeviceContextPool` retains the exact HIP
stream seeds. Resolve its named singleton from the stopped process's own ELF,
then its typed AMD context owner and the native device's complete stream
inventory. Build the unlinked application type helper from the actual host
compiler/feature flags; build the native helper from the installed runtime's
Ninja compile command, preserving its source, generated-header and SDK roots.
Force complete DWARF for `amd_signal_t`, `amd_queue_v2_t` and concrete
libstdc++ hash-node instantiations. A pointer declaration alone can leave a
template or polymorphic class incomplete; explicit template instantiation and
GCC `-femit-class-debug-always` resolve the respective cases. Keep helpers
unlinked and verify `ptype`/type lookup before relying on an attached snapshot.

SDK-relative libstdc++ paths may not auto-load the distribution's visualizers.
Explicitly register the installed `libstdcxx.v6.printers` when needed and set
the debugger language to C++. Use the authenticated types and visualizers;
do not compensate for missing debug metadata with guessed private offsets.
Validate context-map keys against device ordinals, bound every container walk,
and label queue sampling when the live frontier exceeds the inspection bound.
An attachment can perturb scheduling even without GPU readbacks: preserve its
results as diagnostics and rerun race/performance certification uninstrumented.
