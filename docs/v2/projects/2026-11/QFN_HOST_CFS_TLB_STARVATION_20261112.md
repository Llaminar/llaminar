# QFN startup: host scheduler starvation of AMD TLB work

## Scope and current conclusion

The target remains the original QFN `PP(EP(4×MI50), EP(4×MI50))` layout with
fixed-head ngrams sharded across all eight devices. This investigation preserves
the original MPI rankfile, physical GPU identities, GGUF, 16-history observer
corpus, arithmetic and capture policy. It neither accepts failed numerical
results nor establishes a throughput baseline. The passing application repair
slice is authorized for checkpointing; the host kernel remains unchanged.

The live evidence strongly identifies **host CPU scheduling starvation** as
the immediate startup failure. An AMD TLB-flush work item is queued with an
already-signaled dependency, but the runnable kernel worker receives no CPU
time while normal-priority native initialization threads occupy the same CPU.
The running kernel contains a documented CFS reweight defect with this symptom.
The upstream repair has been built and packaged, but the paired KVM experiment
did not reproduce starvation on either kernel and is explicitly inconclusive.
Host-specific driver qualification and activation have not occurred. A separate
application affinity defect is confirmed and repaired below. Both unchanged
eight-GPU retries complete and retire cleanly on the original kernel. The
second also qualifies fresh numerical observations after a separate logits
publication fix. This does not establish the precise scheduler defect's
causality or qualify the uninstalled kernel candidate.

## Identity and preserved experiments

- Source checkpoint: `b308d23384f671aaf890913c6490a682406cb421`, branch
  `feat/qfn-ngram-continuation`, authoritative checkout
  `/tmp/qfn-qualified-checkpoint-r127/source`.
- Original discovery rank 1, NUMA1; stage 0 uses ROCm 7/6/8/9 and layers 0–23;
  stage 1 uses ROCm 1/3/4/2 and layers 24–47. Unused GPUs remain visible.
- User-rebooted boot: `4d78c044-3705-412b-9d28-150c0128c855`.
- Kernel: `6.17.0-1000-generic`, Ubuntu 6.17.13 derivative
  `6.17.0-1000.42+llaminar2`; GNU build ID
  `bc3a31fc96512be21f0f3ed43ee8727b0de6f1e1`.
- Loaded AMD module: `7.1.9.31600000`, source version
  `BFEAE8D9287FE23B9907691`. Host module pathname:
  `/lib/modules/6.17.0-1000-generic/updates/dkms/amdgpu.ko`.
- Selected HIP/HSA SDK: `/opt/rocm-sdk`, Core 10.1 with the already-qualified
  local native repairs. RCCL is the exact configured source-built dependency.

`q99` reaches M512 capture and reports `layer24_gdn_wo_allreduce`, arrivals
1/4, after the standard 30-second deadline. Other workers wait in native
allocation/mapping. The run eventually aborts with native status 134. Its
driver window completes without new findings; that does not prove progress.

`q100` captures the actual initiating mapping stack, then is stopped through
its authenticated MPI launcher after the stalled observation is preserved.
`q101` pre-arms bounded passive fence tracing and retains host worker state.
It is also stopped through the original launcher. Both report failed work,
complete native retirement and clean complete driver windows. No process is
reinterpreted as a successful observer, and no original result is overwritten.

These runs include host debugger/BPF observations and are **diagnostic only**.
Their durations are not performance samples. The canonical corpus has no
completed result from them. The earlier short-history detailed fixture's
Decode mean KL remains 0.1982729 under its unchanged failing contract.

## Evidence chain

1. **Original device allocation.** The blocked thread follows
   `ncclP2pAllocateShareableBuffer` → `ncclCudaCallocDebug` →
   `hipExtMallocWithFlags` → `deviceLocalAlloc` → `deviceAllowAccess` →
   `hsa_amd_agents_allow_access` → KFD map ioctl. Exact HIP ELF call-site
   disassembly identifies the otherwise stripped middle frames. The RCCL
   build selects uncached device backing in its legacy allocation branch.

2. **Broad peer access is real but is not a demonstrated cause.** The ioctl
   carries nine destination GPU IDs, including devices outside the stage.
   The pinned HIP implementation grants access to every capable peer once
   access is enabled. A simple replacement with the enabled-peer list would
   break incremental peer enablement and race allocation publication; explicit
   command consumers, pools and IPC also use access grants. No such shortcut
   was installed. The ioctl contains no allocation size, and its userspace
   success count is not updated until return.

3. **Software fence, not an outstanding ring job.** Live BTF-based tracing
   records a wait on `amdgpu_tlb_fence`. The retained instance has context
   2388, sequence 124, flags 9 and five references. On this exact kernel,
   flags 9 mean 64-bit sequence and signaling enabled; signaled is bit 1.
   Its dependency is the already-signaled `dma_fence_stub` (flags 14).
   The owner is PCI `0000:97:00.0`, PASID 110. All ten GPUs' sampled debugfs
   ring counters have equal emitted and signaled values.

4. **Queued worker never receives CPU.** The work belongs to `events`,
   pool 114, CPU28, NUMA1. The queue is not plugged, has 74 active items
   against limit 1024, and retains the flush work as pending/active. The pool
   reports zero running workers and two idle workers. Worker 21737 is runnable,
   off-CPU, and has no current function. Its execution runtime, context-switch
   count and last execution time remain unchanged across repeated observations,
   hundreds of seconds after its last dispatch. All 754 inspected kernel
   workers were readable; none was executing an AMD/DRM/fence wait.

5. **Native waiters monopolize the same CPU.** Four selected observer threads
   run on CPU28, each consuming about one quarter of it. Every relevant thread
   uses normal scheduling policy and priority 120. Their stacks show repeated
   nonblocking socket receive through `ncclPollProxyResponse` inside
   `ncclProxyCallBlocking`, during native P2P setup. No runnable RT/DL offender
   was observed. A 0.3-second host-only debugger attachment pauses the waiters;
   immediately afterward the first communicator completes, which it had not
   done during the preceding ten minutes. This is a diagnostic perturbation,
   not a proposed production scheduler or workaround.

The exact module's disassembly authenticates private TLB-work field offsets;
live kernel BTF supplies generic fence, workqueue and task layouts. Observations
read CPU protocol metadata only. No GPU/model tensors, native GPU APIs, device
resets, visibility masks or alternative inference paths were used.

## Confirmed application affinity defect

`MPIBootstrapPhase::configureClusterRankThreads` explicitly binds every OpenMP
worker, including the calling master, to a physical core. Its existing
`ScopedOpenMPPartitionAffinity` restores a mask only when that scope first
expanded it. With `OMP_PROC_BIND=false` supplied before process startup, the
partition initially needs no change. The later explicit master pin therefore
survived the scope. Native child threads subsequently inherit the caller's
singleton mask.

The public observer enters this actual bootstrap path. Its native RCCL
initialization starts ordinary threads; the native topology code intersects
the inherited mask with GPU locality rather than manufacturing a wider rank
mask. This explains how otherwise parallel initialization can concentrate on
one core. q101 measured the threads executing on CPU28, but did not retain their
allowed-mask files: the exact inheritance mechanism is source-derived and
reproduced separately, not a claimed live-mask measurement.

The scoped repair reuses `ScopedCallerThreadAffinity` around rank-worker
admission. It restores the caller's exact incoming mask on success or exception,
independently of partition expansion. Existing OpenMP pool workers keep their
individual bindings. It changes neither rank ownership nor native collective,
GPU visibility, synchronization or scheduling policy.

An isolated native check compiled the exact production method and its new
fixture body. Removing only the restoration guard reproduces the leaked master
mask; the repair passes 20 separate process runs. Evidence:
`/tmp/qfn-bootstrap-caller-affinity-lifecycle-np86w5hg/result.json`. This is focused
failure reproduction, not a full-model certificate. Four affected aliases,
including `V2_Integration_MPIBootstrapCallerAffinity`, now also pass against the
actual rebuilt Integration and Release owners (`q104` and `q107`). The fixture
covers automatic/full/partial/repeated teams, native child inheritance,
retained worker placement and nested-call rejection.

The [wider worker audit](QFN_HOST_THREAD_AFFINITY_AUDIT_20261112.md) found more
independent control and loading workers inheriting a single OpenMP place. Its
first combined Integration checks pass 30/30, and the disk-worker lifecycle
aliases pass another 40 repeated invocations. The unchanged q117 model retry
uses that frozen Integration core; live native masks now show rank-local CPU
eligibility for controls and separate per-core masks for existing compute
workers. The retry completed all 16 histories/112 observations in 82.621 seconds
without intervention, with matching reset bytes, a clean driver window and
complete retirement. Its decode observations exposed a separate stale-logit
publication issue and are not an accuracy pass. Newer NUMA and test edits remain
outside that frozen runtime identity until their owners are rebuilt.

## Matching upstream defect and proposed repair

The [Ubuntu scheduler report](https://bugs.launchpad.net/bugs/2171398) describes
normal-priority load in a child cgroup starving root-cgroup kernel workers.
The implicated change is `6d71a9c61604` in `reweight_entity`. The actual running
kernel build ID matches the retained r905 build. Its cached compiled source's
`kernel/sched/fair.c` matches the retained candidate and contains that code.

The reviewed upstream series is ordered; taking only the revert is insufficient
because the kernel also contains the related zero-vruntime tracking change:

1. `b3d99f43c72b`: fix zero-vruntime tracking.
2. `1319ea57529e`: correct that tracking fix.
3. `101f3498b4bd`: revert the problematic reweight placement change.

Primary patches and the Ubuntu 6.17 adaptations are retained in
`/tmp/qfn-cfs-starvation-repair-20261112/`. They apply to an isolated copy of
`fair.c`; no application or host runtime was changed for this review. Known
AMD TLB gating and reset-domain-unlock fixes were also checked and are already
present. No verified upstream TLB-workqueue repair was found that better
matches the observation.

The isolated kernel candidate is `6.17.0-1001-generic`, package version
`6.17.0-1001.42+llaminar3`, build ID
`a76150710f93165144a83bdfb8b8749901c99717`. The current ABI1000 rollback remains
available. All 12 packages were built, with all 6,796 module payload paths
preserved after ABI normalization. Image/config identity, header/Rust closure,
private-key exclusion and inherited IOMMU/DRM PRIME fixes were checked. The
configuration differs only in its version signature. Complete source/object
and package caches are retained in `/tmp/qfn-cfs-starvation-kernel-20261112/`.

This package audit does not qualify the host's custom AMD, NVIDIA or pex-window
DKMS products against the new ABI. No host package, loaded module, boot default
or kernel has changed. Installation and reboot remain a separate decision after
those dependencies are reviewable.

## Paired KVM experiment: inconclusive

Twelve predeclared trials ran the actual old and candidate images under KVM:
three alternating pairs for each of constant and changing child-cgroup weights,
20 seconds per trial, one guest CPU, 64 normal-priority spinners and a root-group
1 ms observer. The starvation threshold was one second. **Neither kernel
failed**; old maximum lateness was 7.307 ms and candidate maximum was 5.979 ms.
This neither reproduces the native failure nor qualifies its repair.

Every guest authenticated its image build ID, normal priority and zero loaded
modules. All 786 children retired and all exact owned VM containers were
removed. Guest steal time and host-vCPU scheduling delay are retained. No
parameter search or automatic retry was used to manufacture a result. The
experiment did not observe whether the implicated reweight precondition
actually occurred. See `/tmp/qfn-cfs-starvation-vm-regression-20261112/RESULTS.md`.

## Required validation

- Apply and compile the complete ordered repair against the authenticated
  patched source; compare the final scheduler diff and retained unrelated fixes.
- Audit image/modules/header/Rust install products, feature parity, signing-key
  exclusion and build identity. Preserve current rollback artifacts.
- Run paired isolated KVM guests using the actual old and candidate kernel
  images, with normal-priority child-cgroup load and a root-group latency
  observer. Include constant and changing weights. An old image that does not
  reproduce the failure makes that experiment inconclusive, not a repair pass.
- Rebuild and run the application affinity regression against the actual
  Integration and Release owners, preserving the selected SDK and OpenMP ABI.
- After activation, authenticate the running kernel/module identities and all
  original devices. Repeat the unchanged public 16-history observer without
  debugger/BPF attachments. Retain timing, driver, resource and retirement proof.
- Confirm cold communicator setup and capture progress, then run the affected
  real inference and focused native transport/capture regressions. An improved
  startup time alone does not certify logits, MTP or complete preflight.
- Only then collect the matched unprofiled Release baseline and separate
  inference-stage profile for the eight-GPU tuning objective.

## Local evidence

Full q99/q100/q101 results remain under
`/tmp/qfn-postcheckpoint-unscored-single-q8-corpus-q{99,100,101}/`.
The q101 directory includes `passive-kernel-fences.log`, `live-tlb-fence.log`,
`live-tlb-work-pool-r3.log`, `host-native-spinner-stacks.log`,
`host-fence-inspection-r2/report.json` and `host-scheduler-inspection/report.json`.
Exact owner/stop receipts and bounded observers are under
`/tmp/qfn-postcheckpoint-focused-launch/`. The initial q100 ELF analysis is
`/tmp/qfn-q100-hip-frame-elf-evidence.txt`; its missing original maps/inode
caveat remains explicit. q101 preserves the original process mappings.

These are detailed investigation records, not a size-bounded tuning dashboard.
