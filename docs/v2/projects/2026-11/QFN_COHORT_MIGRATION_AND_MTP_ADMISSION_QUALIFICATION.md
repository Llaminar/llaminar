# QFN cohort movement and active-MTP qualification

This is a scoped qualification record, not a complete preflight, model, Release
performance or image certificate. The preceding bounded-loader/Static numerical
checkpoint is `3312168c7`. The current work remains on the same feature branch;
compact transfer changes are now composed and have focused host/native coverage.

## Numerical boundary remains unchanged

The approved TP2 short-decode policy permits at most one row above the ordinary
1.2 KL ceiling, capped at 1.5. Mean KL and worst per-offset mean remain 0.1.
The outlier stays in every aggregate. The retained panel mean is 0.0379831;
the single exceptional row is 1.3826411. The 25 scorer tests pass, including
two-outlier, exceeded-cap, aggregate and agreement negative cases. No new model
accuracy result is implied by the placement or source-admission tests below.

## Whole/cohort movement metadata

`MoEExpertPlacementTarget` derives one checked target from the canonical owner
row: either its whole participant or its complete immutable original cohort.
It owns no allocation, quota or live placement ledger. The existing capacity,
cycle and shadow-slot builders now consume this target. Physical slot flow
includes every member; logical expert/tier flow is counted once.

The actual builders pass tests for two through eight metadata members,
whole/cohort and cohort/cohort cycles, direct tier-skipping cycles, repeated
large/empty/small waves, missing fragments, changed cuts, foreign full source
geometry/precision, duplicate experts and moved-from cohorts. Reciprocal
redistribution between different admitted cuts of the same original matrix
passes; incompatible original matrices do not. The existing whole-owner and
residency-authority unit suites also pass. These tests create no extra MPI
ranks or artificial physical sockets.

Source-authenticated joint placement snapshots and CPU fragment epoch banks
pass their component contracts. Physical-fragment endpoint wiring now stages
and executes exact destinations on two genuine CPU sockets. Successful staging,
failed-piece drain and setup rejection each pass 20 repetitions per rank, with
real retired-reader and reclamation checks. Live GPU packet membership also
passes native CUDA/ROCm capture tests. Global publication and retirement through
complete CPU/GPU inference remain unresolved.
Their runtime guards are unchanged. Component correctness alone does not admit
a dynamic execution mode. The broader design remains in
[the dynamic residency plan](QFN_SHARDED_EXPERT_DYNAMIC_RESIDENCY_PLAN.md).

## CPU fragment epoch-bank foundation

The existing participant residency registry now owns typed whole or CPU-fragment
payloads. A fragment retains its original model/preparation, cohort member and
three actual prepared engines. Live placement is separate from immutable source
ownership; an incoming candidate cannot substitute a matching-looking foreign
source or change the admitted cuts and quotas. Existing dispatch readers borrow
the actual epoch bank. Cold graph metadata no longer permanently pins retired
engine roots.

Projection completion now checks disjoint, exact destination coverage and waits
for every original operation before publication. An installed-bank arrival
authenticates the actual installed fragment and starts unready; installation
alone does not prove completed copies or globally current placement. The whole
path uses the same completion mechanism for its singleton operation.

Integration exposed a genuine multi-layer startup defect: the graph builder
borrowed a CPU bank after registering only one layer, before the complete bank
could be installed. Both production callers now use one complete initial-bank
prepass, with atomic installation before any graph layer borrows it. The old
per-layer registration API is removed. Repeated graph-family construction
authenticates the same installed bank without rebuilding it or resurrecting
retired initial state. Tests cover multiple layers, foreign source owners,
post-retirement inspection, incoming expert execution and delayed old readers.
Two fixture defects were also repaired: independent temporary model files now
have independent lifetimes, and NUMA allocations declare their exact physical
root sizes instead of using a heap-only test budget.

Five focused contracts are explicitly registered in `ProductionTestPreflight`:
`PreparedProjectionIntersectionContract`, `CPUFragmentPayloadContract`,
`CPUJointPlacementCandidateContract`, `CPUFragmentEpochBankContract` and
`CPUJointInitialBankPrepassContract` (all with the `V2_Integration_` prefix).
`common-cpu-host147` passes all 13 selected registrations in 9.43 seconds,
including the owning Unit suites and 20 repeats of these contracts.
`common-cpu-native148` passes four native registrations in 11.05 seconds:
MPI residency consensus, the six-GPU CUDA/ROCm controller-fabric control, and
both backend CPU-joint padded-prefill contracts. No cases skip; driver windows
are clean. Build144 builds the coherent production and test targets; build146
refreshes the corrected registry fixture.

The final real-model control, `common-cpu-exact149`, passes in 287.94 seconds
on one MI50 plus two physical CPU sockets, with one MPI rank and 28 physical
workers per socket, 64 GPU/448 CPU experts, RAM ngrams and dynamic MTP capacity
three. All 3,624 verifier comparisons, 900 row-zero comparisons and six integer
records are byte-exact across six checkpoints. Acceptance and three dynamic
controller windows produce exact serial tokens; four exact-M3 follower
transactions complete. Both runners retire, the GPU allocation ledger falls
from 18,720,329,732 bytes to zero, and driver diagnostics are clean. After
retirement, 716,017,664 allocated bytes of owned prefix artifacts are removed.
CSV, rank and cleanup evidence remain under
`/tmp/qfn-cpu-joint-mtp-exact-r371/dynamic/`; focused driver/build receipts are
under `/tmp/qfn-cpu-joint-qualified-r224/`.

This is native serial/grouped correctness with **Static expert placement** and
dynamic MTP. It is not a fresh PyTorch KL panel, dynamic expert-movement proof,
Release performance result or full-preflight certificate. Complete preflight67
remains unqualified; the user requested scoped validation for this slice.

The original prepared-source binding is now installed at the existing model
producer boundaries. It authenticates the actual `ModelContext`, concrete
`WeightManager`, engine registry and preparation control block; a matching
model ID alone is insufficient. Weak retained source owners reject new use
after expiry while genuine prepared banks retain their own execution lifetime.
Public migration-factory construction requires that binding. Component tests
retain an explicit private test-access boundary rather than a runtime bypass.

`common-cpu-host115` passes all six selected registrations in 3.02 seconds:
the three owning registry/residency/fabric unit suites, original-source
lifecycle and nonempty-bank contracts repeated 20 times, and CPU maintenance
coverage. Driver diagnostics are clean and no cases skip. Build114 includes
the production core, metadata and model-test binaries. These checks establish
source ownership, not dynamic cohort transfer or full-model correctness.

## Whole/cohort physical transfer and exact lane demand

The physical fabric now enumerates each nonempty original-coordinate
source/destination intersection for Gate, Up and Down. Each destination engine
receives all of its pieces, and becomes ready only after their real completion.
Native CPU movement preserves every live encoded field without transferring
allocation padding; floating roots retain their independent source/destination
strides. Shared Q6 storage-word updates use the same local scatter lock across
CPU, GPU-to-CPU and remote destinations. Source-bank leases survive pending
copies and abort/drain; no whole CPU expert or fresh GGUF read is introduced.

Admission and runtime demand use the same piece enumerator and fixed targets.
The retained lane bound obeys both global and per-layer cycle limits, including
unequal layer costs. Remote piece IDs are assigned from the complete wave before
rank filtering, preserving unique identities and deliberate gaps for skewed
members. Metadata covers two through eight members; native qualification uses
one actual MPI rank on each of the two physical sockets.

The joint transaction author validates its exact current source/preparation,
every changed logical expert, complete capacity-preserving cycles and every
member's shadow slots. It neither publishes a candidate nor enables a guarded
runtime mode. The geometry validator now accepts complete original cohorts
without converting one member into a scalar owner; scalar table creation still
rejects cohorts explicitly. Restoration compares the actual live target rows
to the original placement instead of comparing snapshot pointers. Its bounded
cycle composer uses the same sealed joint producer; the progress limit comes
from checked original layer/expert geometry. The new two-layer, nonzero-origin
source contract proves construction without publication. Genuine changed-epoch
joint restoration remains pending the complete commit integration.

`common-cpu-host158` passes all 25 selected registrations in 13.27 seconds,
including the owning suites and repeated focused piece, lane-demand, ordinal,
transaction, geometry and live-native-byte contracts. `common-cpu-gpu160`
passes the six-GPU controller-fabric control and both CUDA/ROCm CPU-joint
padded-prefill contracts (three registrations, 9.06 seconds). No cases skip and
driver windows are clean. `common-cpu-host163` passes the three owning
protocol/transport/fabric suites after the terminal-readiness repair.

The native component fixture uses real source contexts, prepared roots, PMA
owners, MPI lanes and two physical sockets. It delays a real final Down piece,
checks incomplete readiness, then compares every destination FP32 bit and
executes one-hot GEMM through the destination engines. Each rank sends and
receives exactly 147,456 live payload bytes for the swap. Separate cases cover
failed final-piece drain and setup failure before any copy or MPI request.

This native work exposed a real transport defect: a completed source operation
released its lane, then a repeated readiness query incorrectly rejected that
released lane as unadmitted. Terminal Ready now remains stable after the actual
transfer and final ACK; genuinely unadmitted work still fails. Repeated Ready
and abort-drained queries must leave all transfer/protocol counters unchanged.
The successful-copy and setup-failure cases passed in `common-cpu-native162`;
its remaining failure was an incorrect test expectation that a retired bank
remains published. Publication is removed immediately while existing readers
retain its storage. The corrected regression observes the actual retired node
under its existing maintenance lock, then proves one pending reader, zero after
real abort/drain, and no retained node after the normal reclamation sweep. It
adds no ownership state, accounting ledger or inference-path diagnostic call.

Build164 coherently rebuilds the core and ten owning test targets.
`common-cpu-host165` passes all 12 affected host registrations in 6.92 seconds.
`common-cpu-native166` passes existing MPI consensus and the new three-case
physical lifecycle registration in 6.20 seconds. The latter repeats 20 times
per actual rank. `common-cpu-native167` repeats that registration with complete
untruncated output (3.41 seconds): 40 successful rank-case records for each of
the three cases, 120 total, no skips and a clean driver window. Source banks
remain alive only for their actual pending readers; transfer counters stay
stable across repeated terminal queries, and final owned weight/shadow bytes
return to zero. Exact command, full log, repetition summary and driver receipts
remain under `/tmp/qfn-cpu-joint-qualified-r224/common-cpu-native167/`.

These are component transfer results, not global dynamic placement, restored
inference, fresh HF KL or Release performance certificates. Full preflight67
remains unqualified; no broad gate was rerun for this focused slice.

## Live cohort targets in captured GPU packets

The acquired placement bank now supplies each expert's complete target through
its existing signed route word: a whole participant, absence, or a fixed CPU
cohort tier. Captured launch metadata retains only original cohort/member
geometry. It no longer freezes an expert-membership mask or initial epoch.
Initial graph construction and later bank publication use the same encoding;
shared-row selection reads that same acquired bank. No extra owner table,
bitmap bank, mutable host mirror or capacity reserve is introduced.

The channel authenticates its fixed cohort tag as semantic identity. Activation
ABI 8 and mapping ABI 7 reuse reserved words without growing channel storage.
Joint routes omit only exact positive/negative zero before entry/row compaction;
bitwise classification preserves nonzero subnormals under CUDA FTZ. Ordinary
whole-expert selection and numerical kernels retain their arithmetic. Router
hotness is still counted before fanout, not once per packet member.

Build168 coherently builds core, both backends and the owning packet/metadata
suites. CUDA SM80/86/89/90 and HIP gfx906/gfx1100 compile with mandatory spill
guards intact. Build169 refreshes CPU serving metadata. Host170 passes 16 of
19 registrations; host173 passes the three repaired cases. Repairs update the
stale ABI-7 assertion and remove an invalid fixture that put multiple original
CPU-cohort members in one MPI rank. All valid declared 2/4/8-member GPU packet
cases remain; these declare metadata, not native four/eight-socket execution.
Ordinary two-consumer shared-row controls remain. A native mixed GPU/cohort
shared group still needs separate qualification.

The former device-free sparse-phase fixture fabricated successful empty
collective completion without an original prepared bank. Its replacement,
`ActualStageRejectsSparsePhaseWithoutOriginalPreparedBank`, asserts rejection
before any collective/counter publication or output overwrite across the phase
inputs. The obsolete name and scripted-success arm are removed. This is a
source-boundary negative regression; successful sparse-phase publication still
belongs to genuine prepared rooted service integration.

Native171 correctly rejected the new fixture's prefilled acquired ticket.
Native174 then rejected a second materialization of an already captured graph.
The fixture now starts its source ticket empty, obtains the real native reader,
and retains the actual original `MaterializedUnlaunched` executable. A changed
placement writes only its reserved bank before native ready/publish/acquire.
The same captured executable dispatches after that publication, and terminal
checks prove exact slots, coefficient bits, payload bytes, untouched tails and
reader release. No production guard was weakened to admit either fixture.

Native174 passes all 47 existing cases in the 51-case owning packet suite,
plus both explicit shared-union registrations. Its four new cohort cases pass
after repair in `common-cpu-native176` (four registrations, 8.81 seconds), on
CUDA and ROCm with no skips and clean driver evidence. The earlier exact-row
controls also pass on both backends. Build172 and build175 rebuild the repaired
fixtures. These receipts are under `/tmp/qfn-cpu-joint-qualified-r224/`.

These GPU packet traces use fresh channels and terminal cancellation after
inspection; they do not invent CPU completion or reset a cancelled channel.
They prove capture-before-bank-change dispatch, not repeated successful
GPU-to-CPU compute/return transactions or global cohort movement. The first genuine
two-socket rooted round trip is qualified below; repeated successful reuse and
CUDA remain separate obligations. Dynamic cohort inference, controller cost
qualification, fresh model KL, Release performance and complete preflight
remain unqualified.

## First prepared ROCm/CPU rooted round trip

`common-cpu-native188` passes the genuine two-socket case in 3.38 seconds with
no skips and clean driver evidence. One actual MPI rank occupies each physical
socket; rank zero owns the ROCm continuation and CPU root, and rank one computes
the other fragment. Both use original nonzero FP32 GGUF preparation and admitted
CPU storage. This is a reduced common-MoE integration case, not a QFN model KL
or throughput result.

One retained graph publishes the actual source-local ticket and remote packet,
then consumes the contributor completion and the root's canonical result. Both
CPU shards execute their actual prepared engines and original rooted collective.
The test checks a live nine-row prefix inside capacity 64, exact native gather
and dispatch bytes, zero contributor return bytes, original slot placement,
untouched output tails, final epoch-reader release and native-domain retirement.
It observes successful endpoint completion and resets the original protocol;
terminal cancellation does not stand in for success. Both temporary GGUFs retire
normally. The output oracle checks forwarding against the actual root result;
independent fragment arithmetic remains in its existing owning suite.

Fixture repairs retain the production contracts: only the original root owns
canonical storage, grouped roots and named scratch have separate exact BOM
terms, the local return uses the production certificate-only layout, and the CPU
follower activates before waiting for publication and consuming once. Unexpected
exceptions now report before the admitted native owner unwinds and abort the
actual MPI job promptly. No production guard was relaxed for these repairs.

This case covers one nonempty prefill invocation followed by successful reset.
Repeated nonempty/empty/small reuse, CUDA, other successful phases and changed
placement epochs remain pending. Builds181/183/185/187 and the reduced failure
receipts retain the path to this first green native result; they are not model
accuracy or Release performance measurements.

## Shared histogram candidate construction

The existing histogram proposal now uses one candidate-construction boundary
for complete tier targets, unchanged prerequisites and bounded cycle subsets.
Fixed CPU cohorts use the sealed current-source producer and exact member
capacity checks; ordinary whole ownership retains its existing path. This
avoids assigning the cohort sentinel to a scalar owner or duplicating migration,
cycle and shadow derivation. The original economy lock remains held through
production histogram cadence, including when the shared derivation returns.

`common-cpu-host180` passes six scoped registrations in 4.14 seconds, including
all 74 cases in the owning residency-authority suite, pipeline-stage contracts,
existing dependent-cycle and nonblocking-demand cases, and 20 repetitions of the
new source-bound candidate contract. The new case exercises unchanged placement
and a bounded whole/cohort exchange from genuine preparation and recorded
logical demand. It checks original activation counts, all member shadows and
absence of publication. There are no skips; build179 is coherent and passes.

This proves candidate construction, not an enabled cohort controller. The
public Static/Dynamic guards remain. Enabled scalar participant rearrangement
and joint economy without complete member/intersection prices are rejected;
no enabled policy axis is silently removed. The existing promotion/demotion
criteria remain the target for both whole and sharded experts. Actual cohort
service/movement measurements, distributed adoption, ownership publication and
complete native inference remain separate obligations.

## Dense distributed cohort targets

Dense authoritative entries now carry the same checked whole-owner/cohort target
used by the live GPU placement bank. The proposal header remains 96 bytes and
each live entry remains 28 bytes; semantic ABI 3 rejects the former scalar-only
ABI 2. Every changed and unchanged coordinate is reconstructed against the
receiver's original fixed source layout. Export requires the authority's actual
current source snapshot. Existing semantic execution identity includes original
cohort members, cuts and native format metadata once per declaration; no model
weights or source files are hashed.

`common-cpu-focused190` passes all five selected registrations in 6.14 seconds:
75 authority cases, 30 protocol cases, 20 real MPI consensus/publication cases
per rank, the new complete-target source contract, and the ROCm rooted round
trip. It has no skips and clean driver evidence; build189 coherently rebuilds
all affected owners. The metadata contract covers declared cohorts of two
through eight members, old-ABI rejection, absent/foreign/scalar aliases, partial
plans, foreign export authority and mismatched original cuts/members. This does
not claim additional physical sockets.

Reconstruction remains behind the public Dynamic/admission guards. These
results qualify unpublished proposals and existing whole-owner MPI behavior;
actual changed-epoch cohort staging, publication, service/transfer economy and
model inference still require their own complete proof.

## CUDA native-program stack

The original source worker now provisions a typed 2 KiB native-program stack
before source publication, inside the existing 384 MiB opaque commitment.
Captured-program inspection checks that capacity and cannot create a late
stack commitment. The change adds no byte reserve, live ledger, source hash,
per-graph estimate or warmup kernel.

The device-free contract proves capacity limits and original PMA/control-block
ownership. The actual two-3090 captured NCCL regression passes, with two input
patterns and three replays, zero warmups, clean native retirement and clean
driver evidence. All six existing fixed CUDA forward/reverse transfer cases
(64 B, 1,440 B and 737,280 B) also pass in the Integration build. This is not a
Release throughput comparison or complete ngram lookup qualification.

An isolated coherent control removes only the constructor's provisioning body.
The same refreshed test then fails before NCCL creation with
`required=2048 prepared=2048 actual=1024`. Its native contexts still retire and
the driver window stays clean. The control uses the same ABI, all four CUDA
architecture objects and original GNU OpenMP runtime; it changes no shared
build artifacts. The positive and expected-negative results remain separate.

## Active MTP

The original predictor-source check now authenticates complete local
Gate/Up/Down requests from the same whole/cohort preparation authority. It no
longer requires a scalar whole owner for a genuine shard cohort. The existing
exact-MTP fixture uses the established coordinated session so expert-only MPI
ranks serve the authority's commands instead of starting independent oracles.

The canonical source matrix now includes Off, fixed 1, fixed 3 and dynamic
depth with retained capacity 3 for the real two-socket cell. Expanding an
Off-only seed previously left its public maximum depth at zero; the shared
expander now assigns the selected positive envelope when activating its
predictor, while preserving already-declared positive model maxima and every
Off seed. Public parser and per-rank source projections are tested together.

The first real fixed-1 source run on one UUID-selected MI50 and two physical
CPU sockets reached graph construction, then failed after about 165 seconds:
the colocated CPU workspace code looked for the predictor's original tensor
in the main GGUF directory. The predictor resides in a separate artifact.
The shared source-aware loader fix passed all five focused metadata/registry
registrations. The next real run (`r337`, about 168 seconds) passed the repaired
continuation setup but exposed the same main-only lookup in the remote CPU
follower. Both call sites now use `originalCPUExpertWorkspaceSourceFormats`;
the duplicate follower geometry parser is removed. The common resolver retains
all exact source shape/format checks and is tested with distinct main/predictor
formats and missing/changed source metadata. Those source runs remain red;
they do not qualify active MTP admission or native exact inference. Both
ranks retired and the ROCm context reset in each recorded run.

The next source run (`r339`, 311.47 seconds) initialized both ranks, then the
follower's evidence check incorrectly demanded a full continuation-program
completion from its expert-only CPU control endpoint. The repaired evidence
projector authenticates the original source plan and distinguishes continuation
and expert-fragment obligations. Predictor records, original fragment geometry,
prepared-store and PMA checks remain mandatory. Foreign continuation evidence
on an expert-only rank is rejected. Off/1/3/dynamic plan regressions and the
owning model-definition suite pass. The next real fixed-1 source run (`r342`)
passes in 320.33 seconds with a clean driver window and all resources retired.
Rank zero retains 1,728 prepared predictor records and the completed
continuation family; rank one retains 1,344 records and its authenticated
expert-only role. Both report retained depth 3, complete retirement and removed
prefix leaves. This is setup proof, explicitly not controller execution or
numerical inference. Fixed-3 and dynamic source admission also pass (`r343`,
322.25 and 319.82 seconds), with clean driver windows, all ranks/owners retired
and no remaining prefix files. The dynamic observation retains automatic
initial-depth intent and capacity 3; it does not claim a controller decision.
Active source admission is green for this exact physical topology. Native
serial/grouped decoding is the next independent gate.

The exact-MTP fixture review also found two lifecycle defects before native
execution: an exact-only proof dereferenced an absent frozen-history record,
and its lookup validation occurred after coordinated shutdown destroyed model
ownership. The repair reports actual original rank/socket resources
without inventing a history, and validates lookup after forward evidence is
written but before shutdown. The owning model-definition suite and both focused
role/resource registrations pass (`common-cpu-host105`, 3.22 seconds, no skips,
driver clean). Exact grouped/serial inference remains a separate unqualified
obligation.

The first native exact invocation (`common-cpu-exact107`) stopped before model
setup because its invocation omitted the existing reference-cache relocation.
The retained same-model `main-ram-mmap` reference pack supplies authenticated
snapshot headers and the original nine prompt IDs; no reference values enter
the native serial/grouped comparison. With
`LLAMINAR_PARITY_REFERENCE_CACHE_ROOT=/workspaces/llaminar`, the next invocation
(`common-cpu-exact108`, 115.44 seconds) reached graph setup, which rejected the
original NUMA-qualified MI50 identifier as unavailable. Both results remain
red, with clean driver windows and no skips. The difference between ordinary
exact setup and the successful source-admission path is under investigation;
neither run establishes native decode correctness.

The fixture now calls the existing frontend discovery initializer with its
original configuration before acquiring MPI inventory. Explicit accelerator
intent keeps the GPU view host-wide while CPU allocation and rank affinity
remain socket-local. It does not rediscover or relax an incompatible existing
view. The focused startup contract checks Off/1/3/dynamic and rejects wrong
GPU/CPU NUMA identities; the CPU-only control remains local. That regression
and the complete owning model-definition unit suite pass in
`common-cpu-host111` (two registrations, 3.26 seconds, no skips, driver clean).
The initial new test missed its enclosing C++ namespace in build109; build110
repairs it and successfully builds the metadata, production-matrix and both
transfer-performance targets.

The next native run (`common-cpu-exact112`, artifacts under
`/tmp/qfn-cpu-joint-mtp-exact-r350/`) passed setup and recorded six checkpoint
positions. Its six all-row verifier CSVs contain 1,812 byte-exact rows, with
exact integer checks and no recorded numerical failure. It then deadlocked
in passive test evidence: rank zero entered the owner-order allreduce while
rank one waited in the original worker-command broadcast. The subsequent
movement audit also contained an uncommanded allreduce. Both authenticated
host stacks and the completed CSVs are retained under `passive-stall/`.

The owning CTest was deliberately interrupted after diagnosis (516.89 seconds,
driver clean); it published no completed test case and cannot be a certificate.
No model execution, GPU queue or CPU rooted reduction was the observed wait.
All owned processes exited; the interrupted run's owned prefix directory was
removed after recording file metadata. The repair must join both passive
audits on all ranks after worker-command shutdown and before model teardown,
preserving every original bank/movement check. Native qualification remains
open until an uninstrumented run completes its evidence and retirement.

The coordinated-session repair now invokes both passive audits on all ranks
after the original SHUTDOWN command and before runner teardown. A typed,
noncopyable obligation rejects missing or duplicate audit ownership. Local
audit prerequisites and callback failures are agreed before normal retirement;
the original owner-order and movement reductions remain intact. The new native
two-rank regression exercises both authority ranks, exact collective ordering,
failure propagation, unchanged ordinary sessions and missing/duplicate use.
`common-cpu-focused119` passes six registrations in 10.28 seconds, including
that regression, the owning MPI suites, model-definition coverage and the
required fixture. No cases skip and driver diagnostics are clean.

The fresh fixed-1 native rerun (`common-cpu-exact120`, artifacts
`/tmp/qfn-cpu-joint-mtp-exact-r354/depth1/`) passes in 301.76 seconds. All 1,812
all-row records, 900 row-zero records and six integer records are exact across
checkpoint positions zero through five. The actual accepted-token witness
matches its serial oracle, and the production path certifies the ROCm scheduler
ticket boundary with the explicit heterogeneous CPU execution boundary. Both
ranks complete evidence and teardown; canonical native allocation bytes retire
to zero, the exclusive ROCm context resets and driver diagnostics stay clean.
The run-owned prefix directory is removed after preserving its file metadata.
This is native serial/grouped functional proof, not a new PyTorch KL panel or
Release throughput measurement. Fixed-3 and dynamic native proof remain next.

The same session also now agrees local shutdown exceptions before its final
retirement fence, consumes the single retirement attempt and never retries it
from destruction. A failed cleanup stays a failure; this does not recover a
shutdown implementation blocked inside an internal collective. Before the
variable-size coverage sums, the existing MPI context agrees the actual layer
count and integer IDs. No hash or second manifest authority is introduced.
The seven-case passive-audit regression includes one-rank shutdown exceptions,
different layer counts, same-count/different IDs, duplicates and progress after
rejection. `common-cpu-focused123` passes all four selected registrations in
7.40 seconds, including the owning MPI and metadata suites, with no skips and
clean driver evidence. Build122 passes; build121's misspelled target stopped
before compilation and is retained as an invocation error.

Fixed-3 then passes in `common-cpu-exact124` (296.67 seconds, artifacts
`/tmp/qfn-cpu-joint-mtp-exact-r355/depth3/`). Its six checkpoint positions
contain 3,624 exact all-row records, 900 exact row-zero records and six exact
integer records. The actual three-draft transaction and accepted-token witness
match the serial oracle. Both socket-bound ranks retire, the native allocation
ledger returns to zero, the driver window is clean and owned disk prefixes are
removed. Dynamic-depth native proof is still pending.

The first dynamic native attempt (`common-cpu-exact125`, 290.19 seconds) is
red because `ModelParityCase::requestedMTPDraftDepth()` returned the standard
15 for every dynamic case, while this case explicitly declares a maximum and
retained capacity of three. Runtime correctly executed depth three. The first
604 all-row comparisons, 150 row-zero records and integer check are exact;
the fixture then expected fifteen attempted drafts and stopped its remaining
checkpoint proof. Both ranks still retired and the driver window stayed clean.
The owned prefix leaf was removed after preserving metadata. This result is
not a completed dynamic proof. The correction projects the declared dynamic
ceiling through the existing helper, with a focused fixture/runtime/public-CLI
regression covering ceilings 1/3/6/15 and independent larger retained capacity.
Fixed-depth and Off semantics remain unchanged; no arithmetic or KL limit moves.

Build126 installs that declared-ceiling correction. `common-cpu-focused127`
passes all four selected registrations in 3.19 seconds, including the owning
model-definition suite and the explicit preflight regression for ceilings
1/3/6/15. There are no skips and the driver window is clean.

The fresh dynamic invocation (`common-cpu-exact128`, 297.01 seconds, artifacts
`/tmp/qfn-cpu-joint-mtp-exact-r356/dynamic/`) passes all six checkpoint positions:
all 3,624 all-row comparisons are exact. Subsequent adaptive execution selects
a three-row verifier transaction, which the CPU follower rejects before graph
admission. Its execution-shape manifest was incorrectly derived from the
power-of-two storage families, omitting live M3 even though the admitted M4
storage supports it. The result remains red: it proves neither completed
adaptive execution nor normal native retirement. All owned processes exit and
the driver window is clean; the owned prefix leaf is removed after preserving
metadata. The repair must retain exact live execution and communication sizes
while sharing the existing admitted storage, without allocating a physical
root for every possible row count.

Build129 separates those two manifests: exact decode identities cover every
row through the declared verifier bound and retain the original prefill ladder;
each selects an already-admitted supporting storage family. M3 keeps live
compute and packet extents at three over M4 backing. CPU endpoints retain one
maximum-sized host program, and GPU specializations are prepared before serving.
The original failure diagnostic is now emitted before cancellation can abort
the MPI domain. No physical root, reservation or runtime capture is added.
`common-cpu-focused130` passes all four selected registrations in 1.85 seconds,
including the owning transaction suite and the explicit live-row preflight
regression. Coverage includes bounds 1/4/16/64, the original odd prefill bucket
600, unchanged CPU storage quotes and twenty protocol generations of
4→3(partial)→1 with reset. There are no skips and driver diagnostics are clean.
The fresh dynamic model invocation (`common-cpu-exact131`, 300.16 seconds,
artifacts `/tmp/qfn-cpu-joint-mtp-exact-r358/dynamic/`) passes. All 3,624 all-row,
900 row-zero and six integer comparisons are exact across six checkpoint
positions. The adaptive witness completes three controller windows, seven
attempted drafts and four verifier transactions; its five emitted tokens match
the serial oracle. The separate acceptance witness commits three accepted
drafts and also matches serial tokens. Both actual socket ranks use 28 workers
and retire cleanly. The canonical native allocation ledger again returns
18,720,329,732 bytes to zero, the driver window is clean and 716,017,664 allocated
bytes of run-owned prefix files are removed after preserving metadata. Fixed
1, fixed 3 and dynamic native proof now pass for this topology. No new PyTorch
KL panel or Release throughput claim is made. The follower log records four
actual M3 admissions and completed transactions.

Build132 adds complementary native capacity regressions. `common-cpu-native133`
passes all four explicit CUDA/ROCm registrations in 10.84 seconds, with no skips
and clean driver evidence. The actual production expert stage captures M4 and
M3 over the same M4 tensors/workspace for every quantized format and
FP16/BF16/FP32. Thirty-five large-to-small, partial, empty and reset replays
match the independent serial oracle while preserving the unused fourth row.
The separate packet fixtures use actual captured consume/return stages at
M4/M3/M1, verify exact completed traffic and preserve poisoned backing suffixes.
These packet fixtures are independent retained transactions; only the expert
stage test claims the same physical backing across M4→M3. No production kernel
arithmetic, numerical limits or inference policy is changed by these tests.

The original CUDA and ROCm live-row registrations also pass in
`common-cpu-native134`. That invocation exposes only one MI50, so the full
packet suite correctly fails on four skipped two-ROCm cases despite its other
45 cases passing. The corrected packet-only invocation
(`common-cpu-native135`, 18.64 seconds) exposes two UUID-selected MI50s alongside
the two CUDA devices: all 49 packet/return-validation cases pass without skips
and the driver window stays clean. No production change or relaxed assertion
was used to correct this invocation. The new focused native registrations use
the ordinary 30-second model-free watchdog; their measured durations are
2.22–2.76 seconds. Full production preflight remains deliberately unrun for
this scoped slice, as requested.

## Compact original-coordinate expert transfers

The existing lane and remote projection protocol now carry checked original
source/destination rectangles with independent roots and strides. CPU-to-CPU
and CPU/GPU wire chunks contain only live encoded columns and their native
metadata. Resident N64 padding remains storage, never a communication extent.
The same copier preserves all 21 quantized formats, including Q6 shared bit
planes, source scale/minimum tuples and IQ grid factors. Floating transfers
preserve original row strides, including non-aligned K17 rectangles.

CPU cursors retain the original send staging until the exact MPI payload
request completes; destination scatter validates the complete prefix before
writing. No full CPU expert assembly, GGUF reread, requantization or new
transport/ledger is introduced. Small 73-byte staging is covered independently
of resident block size, with exact short-tail bytes and delayed ACK negatives.

The first composed native run (`common-cpu-transfer99`) passed 9/13
registrations. Three CUDA cases exposed an IQ3_S compact promotion bug: one
quartet-address helper silently used a 64-column stride for a 47-column tail,
causing a native trap and three NVIDIA Xid 43 reports. The fourth failure was
a stale floating-transfer counter assertion after adding rectangle traffic;
the ROCm quantized cases passed. These red reports and driver faults remain
preserved. The CUDA fix passes live width explicitly and removes its unsafe
default argument. Floating assertions now count both whole and rectangular
operations.

All four repaired native registrations pass in `common-cpu-native103`, 8.11
seconds, no skips and a clean driver window. Direct and captured local/remote
proofs cover original unequal origins, compact tails, dirty repeated commands
and exact bytes. The ROCm captured cases passed in the earlier run. CUDA
80/86/89/90 and HIP gfx906/gfx1100 compiled with unchanged spill guards; runtime
claims here apply only to the actual 3090/MI50 tests. Profiling, Release economy
and the physical fabric's whole/cohort fanout, coverage, publication and old-reader
retirement remain pending. These primitives do not enable dynamic cohort movement.

The owning native MPI suite initially retained resident-padding expectations
(`common-cpu-mpi116`). Its four affected cases now require the compact live
extent explicitly: 1,280 wire bytes rather than 2,560 resident bytes, and
16 exact 80-byte chunks through a 97-byte lane. The direct destination oracle
compares the entire image, including untouched dirty padding. Uninitialized
fabric slots are checked on every live plane without inventing initial padding
values. Async ownership, ACK, reversed-context and RCU checks remain intact.
The owning suite and explicit 20-repeat compact-wire regression both pass in
`common-cpu-focused119`; the red original report remains preserved.

Static resource evidence in `/tmp/qfn-tier-weight-static-resources-r342/`
binds the configured source/objects to all four CUDA and both HIP targets.
Standalone IQ3_S promotion uses 30/36/36/30 registers on sm80/86/89/90;
gfx906 uses 23 VGPRs and 48 SGPRs, and gfx1100 uses 24/45. HIP spill, private
and LDS metadata are zero. CUDA standalone IQ3_S/Q4/Q6 code has no local
loads/stores. The mapped service retains five local-memory instruction sites
per architecture from its small source-array lowering, so this evidence does
not claim that the entire service is register-only. It is static allocation
evidence, not achieved occupancy or throughput.

The existing transfer performance fixture now accepts an explicit original-
coordinate rectangle and emits exact wire bytes, both endpoint geometries and
raw event batches. It retains aligned controls, both directions, independent
byte oracles and the existing economy thresholds. Both performance targets
build in `common-cpu-build106`. Integration timings are diagnostic only;
Release and full migration economy remain separate obligations.

The first timing invocation (`r346`) stopped before child launch because its
scratch guardian directory had not been created. The next (`r347`) exposed
stale benchmark source metadata: IQ3_S physical storage needs a coefficient
plane even though its mathematical descriptor is symmetric. The fixture now
uses the same canonical device descriptor as the already-passing all-format
native fixture, including IQ4_XS storage/decoder identity. Original source
bytes and representation-preserving packing remain unchanged. Both red runs
have clean driver evidence and contain no timing rows.

The repaired diagnostic panel (`/tmp/qfn-expert-transfer-timing-r349/`)
records 64 byte-valid rows across CUDA/ROCm, IQ3_S/Q4_0/Q6_K/Q3_K, both
directions, kernel/streaming and whole/rectangle shapes. Every process has a
clean driver window. All 32 whole-projection rows pass the existing economy
floors; useful-byte streaming throughput ranges from 6.94 to 9.64 GB/s.
All 32 tiny-rectangle rows remain economy-red with unchanged thresholds.
For those rows, isolated kernel events measure one two-unit chunk (1,880 to
3,572 bytes), while complete streaming events include four such chunks.
The two scopes must not be compared as equal work. IQ3_S promotion is a
specific remaining cost: its tiny chunk takes 9.68 microseconds on CUDA and
26.10 on MI50 in event batch means, close to the corresponding whole-control
kernel times (11.83/29.01). These include native submission/scheduling effects;
they are not standalone hardware service-time or complete migration results.

## Evidence and validation scope

Local evidence root: `/tmp/qfn-cpu-joint-qualified-r224/`.

| Evidence | Result |
|---|---|
| Builds 77/78 | Preserved compiler failures in new caller/test API names; repaired |
| Builds 79/80/82/85/87/90/92 | Scoped coherent builds passed; no complete gate build claimed |
| `common-cpu-focused81` | 11/13 registrations passed; two MTP policy registrations exposed new test/setup issues |
| `common-cpu-focused93` | Five metadata/registry/predictor registrations passed, 3.95 s, no skips, driver clean |
| `common-cpu-focused91` | Both repaired registrations passed, including the complete owning model-definition unit suite |
| `common-cpu-native86` | Actual CUDA2 source-stack/capture regression passed, 6.83 s, no skips, driver clean |
| `common-cpu-native89` | Six fixed CUDA transfer cases passed, 27.37 s, no skips, driver clean |
| `/tmp/qfn-cuda-source-stack-negative-r334/native-original-negative/summary.json` | Expected original-body failure, 6.71 s; native reset and driver clean |
| `/tmp/qfn-cpu-joint-active-mtp-r335/source-MTPDepth1/` | Red diagnostic-only real-model source run; main/predictor directory confusion |
| `/tmp/qfn-cpu-joint-active-mtp-r337/source-MTPDepth1/` | Red diagnostic-only real-model source run; follower copy of the same source lookup defect |
| `/tmp/qfn-cpu-joint-active-mtp-r339/source-MTPDepth1/` | Red diagnostic-only source run; expert-only follower incorrectly required continuation setup evidence |
| `common-cpu-transfer99` | 9/13 passed; IQ3_S native CUDA trap and stale floating counters preserved, driver window red |
| `common-cpu-build101` | Coherent scoped build passed after caller/test compile repairs; complete shipped kernel targets had compiled in build98/100 |
| `common-cpu-host102` | Five owning model/transport/fabric and MTP-role registrations passed, 4.36 s, no skips, driver clean |
| `common-cpu-native103` | All four repaired rectangle registrations passed, 8.11 s, no skips, driver clean |
| `/tmp/qfn-cpu-joint-active-mtp-r342/source-MTPDepth1/` | Fixed-1 real-model source admission passed, 320.33 s, both ranks/owners retired, prefixes removed, driver clean; diagnostic-only |
| `/tmp/qfn-cpu-joint-active-mtp-r343/source-MTPDepth3/` | Fixed-3 source admission passed, 322.25 s, all resources retired and prefixes removed, driver clean; diagnostic-only |
| `/tmp/qfn-cpu-joint-active-mtp-r343/source-MTPDynamicDepth/` | Dynamic source admission passed, 319.82 s, automatic initial-depth intent retained, all resources retired and prefixes removed, driver clean; diagnostic-only |
| `common-cpu-host105` | Owning model-definition plus two focused MTP role/resource registrations passed, 3.22 s, no skips, driver clean |
| `common-cpu-exact107` | Native exact setup rejected missing reference headers before inference; original canonical reference-cache relocation omitted |
| `common-cpu-exact108` | Native exact graph setup rejected NUMA-qualified MI50 identity, 115.44 s, no skips, driver clean; decoding not reached |
| `common-cpu-build110` | Startup/benchmark repairs and owning tests compiled; build109's namespace error retained |
| `common-cpu-host111` | Startup contract and complete owning model-definition unit suite passed, 3.26 s, two registrations, no skips, driver clean |
| `/tmp/qfn-expert-transfer-timing-r349/` | 64 byte-valid diagnostic timing rows, all driver windows clean; 32 whole rows economy-green, 32 tiny rectangle rows economy-red; Integration only |
| `common-cpu-exact112` | Six native checkpoint positions and 1,812 exact all-row records; then diagnostic collective mismatch, deliberately interrupted after saved stacks, driver clean; no completed test/certification |
| `common-cpu-host115` | Six prepared-source/lifetime registrations passed, 3.02 s, no skips, driver clean; includes original/nonempty-bank contracts repeated 20 times |
| `common-cpu-mpi116` | Owning native MPI suite exposed four stale padded-wire assertions; preserved red result, driver clean |
| `common-cpu-build117` / `common-cpu-build118` | Coherent passive-evidence and compact-wire fixture builds passed |
| `common-cpu-focused119` | Six registrations passed, 10.28 s, no skips, driver clean; owning MPI suites, new passive-evidence regression and compact-wire cases repeated 20 times |
| `common-cpu-exact120` | Fixed-1 native functional proof passed, 301.76 s; 1,812 all-row, 900 row-zero and six integer records exact; accepted-token witness exact, original owners retired, prefix removed, driver clean |
| `common-cpu-focused123` | Four registrations passed, 7.40 s; owning metadata/MPI and hardened seven-case passive audit, no skips, driver clean |
| `common-cpu-exact124` | Fixed-3 native functional proof passed, 296.67 s; 3,624 all-row, 900 row-zero and six integer records exact; accepted-token witness exact, original owners retired, prefix removed, driver clean |
| `common-cpu-exact125` | Dynamic attempt red: fixture expected the standard depth15 despite the declared/runtime ceiling3; first 604 all-row records exact, retirement and driver clean; no completed proof |

The original red reports remain intact. Focused repairs do not convert the
previous complete-preflight receipt into a green one. No complete preflight
was run for this slice, and no model/corpus reference or arithmetic was changed.
