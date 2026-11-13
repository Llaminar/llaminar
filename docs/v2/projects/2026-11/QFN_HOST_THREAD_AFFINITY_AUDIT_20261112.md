# Independent host workers and OpenMP caller affinity

## Contract and scope

The recurring failure is an independent worker inheriting one OpenMP place
instead of the MPI rank's admitted CPU partition. Explicit compute placement
and explicit NUMA first-touch placement are different contracts: broadening
every thread's mask would break them. This audit follows each actual creation
boundary and its callers before selecting a repair.

Use the existing `ScopedOpenMPPartitionAffinity` only around independent child
construction. It derives the current admitted OpenMP partition, preserves the
native caller mask when binding is disabled, and rejects creation from a nested
OpenMP compute region. Restore the creator before waiting, bookkeeping or
compute. `ScopedCallerThreadAffinity` separately owns exact restoration around
an operation that itself changes its caller's mask, regardless of whether a
partition scope expanded that mask initially.

No new rank mapping, NUMA authority, scheduling override, GPU visibility mask
or per-token affinity operation is introduced. Explicit prepared-thread
attributes remain authoritative. The user authorized checkpointing this passing
slice after its original eight-GPU diagnostic completed.

## Findings and qualification

| Boundary | Finding and intended repair | Evidence status |
|---|---|---|
| `MPIBootstrapPhase::configureClusterRankThreads` | Worker binding also pins the master. With initially disabled binding, the old partition scope performed no restoration. Retain the caller's exact mask independently. | Fixed. Original body fails; repaired body passes 20 separate processes and GNU/LLVM runtime checks. Four actual bootstrap/placement aliases pass in both Integration and Release. |
| NCCL/RCCL coordinator initialization and abort | Raw control-thread creation inherits the creator's single place under bound OpenMP startup. Native initializer/proxy children inherit it again. Scope only construction; active compute-region aborts retain their assigned mask. | Fixed. Twelve bound/unbound control and abort aliases plus eight earlier readiness schedules pass in the actual Integration owner. Original-source controls fail as expected; isolated GNU and LLVM checks pass. |
| Heterogeneous bridge progress | Independent persistent progress worker starts without the partition scope. | Fixed and included in those control aliases. Synchronous legacy pair-spawn APIs have no identified production caller and remain unqualified. |
| Dynamic device expert controller | Its outer-host activation starts the independent submitter after finite-placement capture. | Fixed and included in those control aliases. Capture and activation bookkeeping retain creator placement. |
| HTTP process-statistics collector | Collector starts before the listener's existing scope. | Fixed. Actual constructor/callback masks and retirement pass; the saved original core fails the new worker-mask assertion. |
| Disk-prefix maintenance and persistence | Both independent I/O workers inherit the inference caller's place. | Fixed. Actual constructors, both OS task masks, two archive lifetimes and retirement pass. Each bound/unbound alias also passes 20 consecutive invocations; the saved original core fails both worker-mask assertions. |
| GPU staging lanes, progress polling, GPU preparation futures | Independent loading control inherits the creator's place. Polling leaves its active flag set if construction fails; partial staging construction destroys already-created joinable threads. | Fixed. All six bound/unbound aliases pass in the canonical Integration owner and in isolated GNU/LLVM checks. Polling rolls back its flag; partial pool construction stops and joins every created lane before unwinding. CPU packing placement is preserved. |
| Mmap reclamation | Every WeightManager starts an independent reclaim worker, later used after first prefill. Page-table reclamation must not inherit the inference master's single place. | Fixed. Both new bound/unbound aliases and the existing memfd lifecycle alias pass in the canonical owner. The original constructor fails worker/descendant-mask assertions. |
| NUMA first-touch scope | Target-mask installation precedes a throwing vector allocation. Some restore failures are logged but not fatal; backing storage can escape ownership before record insertion. | Fixed. Storage is prepared before borrowing placement, each successful write is owned, restoration is authenticated and backing is retained until record publication. The full allocator Unit alias, new fault/lifecycle alias and original real-NUMA policy-isolation alias pass. |

The observed RCCL startup stall is also associated with a runnable AMD TLB
worker receiving no CPU. The [host scheduler investigation](QFN_HOST_CFS_TLB_STARVATION_20261112.md)
keeps those native observations and the candidate Linux repair separate.
An affinity regression passing does not prove that kernel repair, model quality
or full inference progress.

## Boundaries deliberately preserved

- CUDA/ROCm device-context workers, independent GPU TP workers, immutable lookup
  I/O, retained host services and ordinary residency maintenance already have
  explicit independent-control placement. Their existing native regressions
  remain relevant to this audit.
- `SerializedInferenceExecutor` and ChatUI generation own compute work and
  intentionally retain their creator's placement. The HTTP tests verify that
  expanding service-worker eligibility does not alter an existing inference
  worker.
- The CPU copy lane's destination NUMA placement is explicit. It is not an
  accidental OpenMP singleton and must not be replaced with a generic mask.
- TransferEngine delegates worker creation to the backend-owned context pool;
  it does not need another parallel affinity authority.
- Common CPU/GPU source preparation and CPU producer/consumer packing retain
  compute placement. The older `GPUDeviceWorker` pool and WeightManager
  finalization entrypoint have no identified production callers; diagnostic
  asynchronous stage dumps are separate from ordinary inference. These paths
  are classified, not certified or silently broadened.

These classifications are findings from the reviewed paths, not a claim that
the wider audit or every native topology is already qualified.

The final whole-`src/v2` scan traced native affinity writes and asynchronous
construction rather than matching only thread class names. CPU mask writes are
owned by the MPI bootstrap pin loop, the two existing scoped-affinity helpers,
NUMA first touch and `NUMAAllocator::bindThreadToNode`. The latter's live caller
is the intentionally destination-bound CPU copy lane. No production
`pthread_setaffinity_np`, pthread affinity-attribute or default-attribute writer
was found. OpenMP team-size changes belong to startup admission and retained
worker budgets; the unused `ThreadCountGuard` does not narrow native masks.
MPI bootstrap/session process creation already has its construction scope.
The shutdown notifier inherits an already-admitted HTTP service worker.
This final pass identified mmap reclamation as the remaining live independent
worker gap; it found no additional unowned production mask writer.

## Regression method

Exercise bound and disabled OpenMP policies in separate processes, supplied
before runtime initialization. `KMP_AFFINITY` must not silently override the
requested policy when a test uses LLVM OpenMP. Native masks must be observed
on the actual production-created worker and its relevant descendants, with
the creator's original mask checked after startup and retirement. A direct
test of the helper alone cannot establish correct use by a service.

The bootstrap fixture covers auto/full/partial/repeated teams and rejection
of nested calls. The broader service cases cover already-created compute
workers, independent progress, repeated lifetimes and shutdown. Native
placement tests use available CPUs rather than assuming this host's socket or
core numbering. Focused production-defect aliases belong to
`ProductionTestPreflight`; these are lifecycle/contract tests, not recorded
controller or kernel replay claims.

Original-code controls remain explicitly failing evidence. The prior real
Integration core is retained locally for negative controls of the newly added
stats/archive tests; it is never installed for normal inference or called a
passing runtime. No full preflight rerun is planned for this iteration.

## Current evidence

- Original bootstrap failure and repaired GNU 20/20 plus LLVM check:
  `/tmp/qfn-bootstrap-caller-affinity-lifecycle-np86w5hg/result.json`.
- Actual Integration build/test: `/tmp/qfn-full-preflight-after-cpu-next/`
  `q103-affinity-integration-build.json`, `q104-affinity-integration-checks.xml`.
- Actual Release build/test: `q105-affinity-release-build.json`,
  `q107-affinity-release-checks.xml` in the same directory. `q106` failed test
  discovery because the Release tree requires explicit `-C Release`; it ran
  no test and remains retained.
- Both coordinator original/candidate native-mask controls:
  `/tmp/qfn-native-coordinator-affinity-audit-2c684qe4/result.json`.
- Original real-core background-service control:
  `/tmp/qfn-background-affinity-original-control-20261112/run-r4.json` and
  `identity.json`. Earlier fixture-development attempts are retained separately
  and are not valid negative controls.
- Combined Integration checks: `q115-control-affinity-focused-checks.xml`
  in the same build/test directory: **30/30 pass**. Archive repetition:
  `q116-prefix-affinity-lifecycle-loop.json`, **40/40 invocations pass**.
- Coordinator lifecycle controls and GNU/LLVM evidence:
  `/tmp/qfn-control-affinity-lifecycle-build-xa1vte2a/`.
- Loader isolated cases and exact-original controls:
  `/tmp/qfn-loader-control-affinity-r3_bkmyj/`.
- NUMA actual translation-unit controls:
  `/tmp/qfn-numa-placement-lifecycle-20261112/`. The repaired source passes all
  four focused cases, including 20 placement lifetimes and seven fatal-restore
  checks. The same fixture fails three cases against the original source; its
  ordinary rejected-policy case passes both. Startup/free-symbol recursion and
  syntax checks also pass.
- Mmap reclaim actual-class controls:
  `/tmp/qfn-mmap-reclaim-affinity-xigpu1p4/results.json`. Coverage includes 20
  submitted lifetimes, explicit native defaults, idle/failed retirement,
  constructor failure and nested-call rejection.
- Final combined Integration build `q118` passes; focused CTest selection
  `q119-broad-affinity-focused-checks.xml` passes **47/47 with zero skips** in
  19.78 seconds. It includes the original ownership tests, every new alias,
  the complete NUMA allocator owner and actual OpenMP runtime binding.
  This is the scoped audit gate, not a
  complete Unit/preflight, Release or hardware-matrix certificate.
- Final loading failure-path review reproduces `SIGABRT` when the second staging
  thread cannot start. Constructor rollback and normal destruction now share
  stop/join retirement, and each worker captures its stable heap-owned lane
  instead of reading a growing vector. Faults at creations 1/2/3/8 prove exact
  create/join balance and original error propagation. Controls are in
  `/tmp/qfn-staging-partial-construction-2xphblwc/`. The final `q123` build and
  `q124` focused selection pass: six strengthened loading aliases plus the
  separate publication lifecycle and complete LogitsGatherer Unit alias,
  **8/8 with zero skips**. The NUMA warning-only cleanup passes its alias in
  `q122`; that run's two publication aliases failed a test-position assumption,
  corrected before `q124`, without changing production position ownership.

The unchanged eight-MI50 corpus retry completed in 82.621 seconds on the current
host kernel with the frozen q113 Integration owner. All 16 histories and 112
observations completed; reset bytes matched, the driver window was clean and
all original GPU owners retired with VRAM at its exact baseline. Its startup
observations show rank-1
control threads eligible on CPUs `28-55,84-111`, with the 27 existing non-master
OpenMP workers individually pinned. The current source
also contains newer NUMA/test edits: those are explicitly not part of that
frozen runtime. Actual masks, PSRs, resources and the source/build boundary are
retained in `/tmp/qfn-postcheckpoint-focused-launch/q117-prep/`.

The numerical outcome remains `UNCALIBRATED`, with no pass verdict. Inspection
found decode observations reusing the prefill logits; this is a separate
publication defect under investigation, not evidence of quantization error.
The additional loading/NUMA owners now pass their canonical focused aliases.
The observer-publication repair passes focused tests against the actual rank
completion and real gatherer; the saved original core fails all three new cases.
The unchanged `q125` native retry also completes all 16 histories/112 rows with
fresh decode logits, matching reset bytes, a clean driver window and complete
retirement. All GPU memory returns to baseline. Its P/D mean KL is
0.0260335/0.0225803; unchanged corpus limits pass in the separate comparison.
The canonical unscored report remains `UNCALIBRATED`, not a release certificate.
See `/tmp/qfn-postcheckpoint-focused-launch/q125-prep/OBSERVATION.md` and the
[inference note](QFN_EP4_EP4_POST_CHECKPOINT_INFERENCE_20261112.md).
