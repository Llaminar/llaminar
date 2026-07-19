---
name: nativevnni-gemm-tuning
description: Tune, benchmark, profile, train, certify, and install Llaminar NativeVNNI GEMV/GEMM dispatch policies across CPU, CUDA, and ROCm. Use for NativeVNNI kernel candidate work, all-format dispatch sweeps, M=1 decode, grouped MTP verifier rows, ordinary prefill, dense or MoE projection geometry, profiler-informed fitting, Git LFS corpus publication or reuse, exact overlays, generic rules, p95 regret failures, scorer acceleration, and generated dispatch .inc files.
---

# NativeVNNI GEMM Tuning

Use the common evidence and policy pipeline. Do not create a backend-local
heuristic, model-shape override, CSV parser, or installation path.

## Non-Negotiable Invariants

1. Require grouped MTP rows to be byte-identical to independent serial M=1
   decode rows for every format and backend. Relaxed FP metrics are not an MTP
   acceptance threshold.
2. Measure the production kernel path with production graph/stream modes. Do
   not disable optimizations to manufacture determinism.
3. Keep canonical timing separate from profiler launches. `ncu`, rocprofiler,
   and Linux `perf` replay must never become a latency label.
4. Retain every measured candidate, raw timing sample, and per-physical-launch
   profiler record. Do not retain winners only.
5. Emit exact overlays and generic rules. Exact overlays have runtime
   precedence, but cannot replace generic coverage or participate in sealed
   generic certification.
6. Evaluate measured p95 regret and conservative p95 regret UCB independently
   in every required generic domain. A domain passes only when both are
   strictly below 5%; a corpus is promotable only when at least 99% of required
   domains pass. Maximum regret and global cell p95 are diagnostics, not
   installation gates. The less-than-1% over-budget set must remain explicit
   and must still own real learned generic trees.
7. Keep CPU, CUDA, and ROCm feature and format coverage symmetric. CPU policy
   identity includes AVX2 build/runtime, AVX512 build with AVX2 runtime, and
   AVX512 build/runtime regimes.
8. Never install from partial, pointer-only, unprofiled, stale-build, failed-CV,
   or development-only evidence. Never silently fall back to a serial policy,
   row replay, CPU fitting, or another scorer after explicit acceleration.
9. Unit tests launch no GPU work. Put CUDA/ROCm execution in integration or
   performance suites.
10. Prove dispatch totality independently along all three runtime axes. Every
    positive M must map to exactly one certified bucket, every legal positive
    `(N,K)` geometry must reach exactly one complementary generic-tree leaf,
    and every production runtime codebook must own every required domain.
    Keep focused negative regressions that remove one M domain, one codebook,
    or one tree branch while exact overlays remain present; installation must
    reject all three cases.

## Use The Turnkey Transaction

Run one backend end to end:

```bash
scripts/train_native_vnni_dispatch.sh --backend cuda --install
scripts/train_native_vnni_dispatch.sh --backend rocm --install
scripts/train_native_vnni_dispatch.sh --backend cpu --install -- \
  --cpu-format-shards
scripts/train_native_vnni_dispatch.sh --backend cpu-prefill --install -- \
  --cpu-format-shards
```

Pass hardware/tool/lane controls after `--`. Do not pass `--shapes` or
`--shape-partition`; add production overlays to the shared inventory. The
driver owns backend, profile, output path, fit-only flags, and installation.
CPU collection automatically uses every detected physical socket. Each MPMD
rank owns a distinct resumable timing cell and retains the production
per-socket thread count; set `--cpu-measurement-lanes N` only for deliberate
isolation.

CPU production refresh is dependency ordered. Certify and install the complete
`Fast M=1` decode policy first, rebuild both AVX2 and AVX512 trainers against
that include, and only then collect ordinary-prefill or grouped-verifier
evidence through the production `Auto` route. Use
`--stop-after-cpu-decode --install` for this explicit checkpoint. Do not make a
prefill profiler process bypass an uncertified M=1 selector with a diagnostic
serial-oracle override; that would profile a non-production path.

Every CPU timing stage publishes shape/format/ISA partials atomically. Use
`--cpu-batch-limit N` to bound one invocation and rerun against the same output
directory to collect only missing partials. M=1 fixture synthesis prepares
independent source formats in parallel before timing; packed construction and
candidate measurements remain format-serial so benchmark kernels never
contend. A timing-harness setup optimization requires a new build identity but
must preserve each format's deterministic fixture bytes.

Post-freeze CPU-prefill collection coalesces source formats only when geometry,
ISA regime, and ordered M inventory are identical. One process still packs,
validates, and times every format separately, while byte-identical legacy Q8_1
activation fixtures are cached by M and format-seed class. Pair MPMD jobs only
inside equal `(format phase count, M inventory)` groups and retain the runtime
phase-inventory handshake; grouping by M alone can deadlock at the next format
phase. The current 1,859-record witness plan must serialize to 1,337 process
jobs without losing or duplicating a format record.

Fit-only CPU-prefill replay keeps the immutable development build provenance
through generic fit and freeze. A newly launched post-freeze seal is a separate
measurement generation: its resume contract and certification context must use
the current AVX2/AVX512 trainer digest and current host/compiler provenance.
Reject collection when the current production serial-policy hash differs from
the frozen development generation; a harness-only rebuild is not permission to
certify changed arithmetic.

The command:

1. computes the shared resolved shape-inventory digest;
2. rebuilds both CUDA and ROCm leaf-primary scorer DSOs plus the selected
   backend trainer;
3. runs both scorer integration equivalence suites;
4. selects a corpus by backend, architecture, shape digest, and collection
   argument digest;
5. collects timing and isolated profiler evidence only when no matching corpus
   exists;
6. fits the development corpus, requires complete generic-tree totality plus
   the 99%-of-domains performance quota, freezes the generic digest, opens
   sealed evidence, applies the same domain quota, emits, validates, and
   optionally installs the generated include;
7. seals successful new evidence and atomically publishes it for Git LFS; or
8. verifies and refits an existing corpus without launching candidate kernels
   or profilers.

Complete-tree GPU scoring owns one grow-only graph cache and exact frontier
workspace per worker process. On the current two-GA102/four-gfx906 fitting host,
the profiled canonical scheduler uses four worker lanes per physical CUDA or
ROCm device. Each lane retains about 0.79 GiB at the production compact-frontier
high-water mark, so four lanes consume about 3.2 GiB per device and outperform
one or two lanes without exhausting the 24/32 GiB cards. Treat four as a
measured configuration for this hardware, not a universal constant: profile
lane count again on a different architecture or memory size, keep the complete
per-device high-water allocation below its reviewed VRAM budget, and never add
lanes merely to create more host workers.

Use `scripts/refresh_native_vnni_dispatch_tables.sh` directly only for focused
diagnostic phases, partition collection, or lineage/refinement work that the
turnkey driver delegates internally.

## Corpus Rules

Published generations live at:

```text
benchmark_results/native_vnni_dispatch/corpora/
  <backend>/<architecture>/<shape-digest>-<configuration-digest>/
```

`corpus.manifest.json` is reviewable ordinary Git. Large CSV, JSON, profiler,
trace, and archive payloads are Git LFS objects according to `.gitattributes`.
The corpus sealer/verifier is:

```bash
PYTHONPATH=tests/v2/performance/kernels \
  python3 -m native_vnni_dispatch.corpus_bundle verify \
    --manifest <corpus>/corpus.manifest.json \
    --repository-root .
```

Reject unresolved LFS pointers, symlinks, `.inprogress` files, missing payloads,
digest changes, and stale shape inventories. Collect into the ignored `work/`
staging root and publish only after certification. A fit failure must reuse the
same immutable timing/profiler corpus; generate additional measurements only
from an authenticated refinement or shape-lineage plan.

Fit-only replay uses `--skip-sweep --reuse-profiler-evidence`. It may regenerate
common observations and authenticated profiler feature tables. Every profiler
transaction also retains an exact compact observation-witness CSV. If an older
transaction omitted that file, recover it with `profiler_evidence
compact-witnesses --feature-table ...`; the command reconstructs only strict
common-observation columns, verifies their embedded digests, and authenticates
them against request/evidence manifests. It must not run trainer timing tests or
profiler collectors.

When changing only profiler-model features, assess the cross-fitted teacher
before paying for the complete tree frontier. Add
`--cpu-prefill-diagnostic-max-leaves 1` to a fit-only
`--stop-after-cpu-prefill-freeze` replay. Despite that authorization flag's
historical name, diagnostic mode stops immediately after the development-fit
report and does not freeze, seal, or install anything. Run the profiler-informed
half normally, then repeat the identical command with
`--cpu-prefill-ablate-profiler-features`; the reports are retained separately as
`.profiler-informed` and `.profiler-ablated`. The ablation still authenticates
the complete request/evidence/witness catalog and withholds only learner
features. These options are deliberately rejected for collection, sealing,
installation, and every non-CPU-prefill workflow. The teacher's held choices do
not depend on leaf budget; a promising diagnostic can then be rerun at the
canonical leaf budget while reusing every durable profiler surface.

Before either A/B fit, inspect the exact model-visible evidence rather than raw
counter JSON:

```bash
PYTHONPATH=tests/v2/performance/kernels \
  python3 -m native_vnni_dispatch.profiler_diagnostics \
    --observation <compact-observation-witness.csv> \
    --requests <exact-v4-requests.json> \
    --evidence <exact-v4-evidence.json> \
    --output <profiler-diagnostics.json>
```

Require complete per-candidate contests for the stable instruction/work and L1
targets, meaningful within-contest variation, and explicit reliability rates
for duration-derived targets. Runtime codebook and prepared-family counts must
match the physical execution identities requested by the corpus; source-format
aliases may deduplicate only when their complete prepared launch identity is
identical. Never treat a binary reliability flag as a continuous learner
metric.

Every profiler record belongs to one exact physical production invocation:
backend, architecture/ISA, operation, bundle, prepared family, packing ABI,
runtime codebook, effective candidate, execution mode, `M`, ordered projection
`N` vector, aggregate `N`, and `K`. Generate one request for every measured
launchable point. Deduplicate only true aliases with that complete identity.
Never broadcast counters from a representative anchor to another work size or
candidate, and never average unrelated launches into one descriptor.

Request schema `native-vnni-profiler-request-v4-exact-point` is the current fit
contract. Historical v3 anchor catalogs remain immutable and readable for
provenance, but are not admissible current fit evidence. A replacement
transaction is derived from existing canonical timing witnesses, profiles each
exact point separately, and does not repeat canonical timing. Evidence
composition rejects incompatible request or collector generations, missing
records, conflicting counters, or a profiled launch whose complete identity
differs from its request. Compatible immutable raw-profiler feature generations
are normalized to the current exporter schema during composition; a feature
engineering change is never authority to relaunch hardware counters.

When a final timing corpus extends a previously complete profiler transaction,
do not reprofile the covered launch surface. Run `profiler_evidence
emit-missing-requests` with the final corpus and every covered request/evidence
manifest, collect the emitted exact delta with `--resume --retry-failures`, and
compose the transactions only after exact physical-key coverage succeeds. A
zero-row delta is the proof that the profiler catalog covers the fit corpus.
The turnkey driver discovers completed content-addressed deltas that were not
yet atomically published, authenticates them, and includes them in coverage
before emitting another request. It also resumes when only the durable
`.inprogress.jsonl` collector journal exists. CPU decode retains one stable
development run ID per output directory so a restart reopens the same partial
transaction instead of creating a timestamp sibling. Physical resume identity
ignores timing-run provenance but includes arithmetic fingerprint, candidate
policy hash, schedule, workspace, prepared resources, threading/stream mode,
execution mode, and exact `M/N/K` geometry.

Migrate the recipe through its authenticated command, not by editing JSON:

```bash
PYTHONPATH=tests/v2/performance/kernels \
  python3 -m native_vnni_dispatch.cpu_prefill_replay_recipe \
  set-primary-profiler \
  --recipe <corpus>/cpu_prefill_replay_recipe.v1.json \
  --requests <corpus>/cpu_prefill_profiler_requests.aligned.json \
  --evidence <corpus>/cpu_prefill_profiler_evidence.aligned.json \
  --observations <corpus>/cpu_prefill_profiler_observation_witnesses.aligned.csv \
  --clear-additive-profiler-transactions
```

The clear flag is required when the replacement transaction completely
supersedes incompatible additive catalogs; the command refuses to discard
those lineage records implicitly.

The immutable corpus is the reusable evidence product. A failed fit does not
authorize another broad measurement sweep. Refit from the same timing and
per-candidate profiler records, adjust learner features/search, or append only
an authenticated refinement plan for named domains. Resume collection from
content-addressed checkpoints; never discard completed cells because one later
shape is flaky or one fitter configuration changes.

Refinement fits are content-addressed and incremental. Candidate costs
invalidate only the changed generic domain. Profiler-informed CV invalidates
every codebook and prefill M in the changed cross-M transfer pool because those
domains share one surrogate. M is a numeric model feature, not a pool boundary;
held-out N/K geometry must be removed across all M and formats before fitting.
Unrelated ISA, arithmetic-contract, operation, bundle, and execution-mode pools
must remain cache hits. Cache CV on normalized model-visible descriptor content
plus the affected pool's cost identities; keep the full request/evidence/corpus
provenance digest in frozen policy metadata. Never put a global additive
timing-corpus digest in every domain's CV key. Cache the stable final
publication leaves with each production CV entry as well; an identical replay
must not deserialize candidate costs or rerun final GPU tree fitting for
unchanged domains.

Profiler-model parallelism is surface-level, not pool-level. Materialize each
pool's normalized candidate feature index once in the parent, fork it
copy-on-write, and schedule independent held-out forests across the complete
host worker budget. Default CPU-bound process pools to affinity-visible physical
cores, not SMT threads; explicit worker environment variables remain diagnostic
overrides. A cross-M pool may contain most of the useful evidence; it must not
reduce a two-socket fit to one active process.

Post-accelerator CV reduction is also domain-parallel. The parent groups fold
results by generic domain, forks immutable cost/fold matrices copy-on-write,
and sends only inherited domain indices to workers. Cap the reduction pool at
the affinity-visible physical-core count even when
`LLAMINAR_NATIVE_VNNI_POLICY_WORKERS` requests more; merge results in canonical
domain order and publish fit-cache records only from the parent. With
`LLAMINAR_NATIVE_VNNI_POLICY_TIMING=1`, require a nonzero
`cv_reduction_workers` field for an uncached multi-domain fit and compare
`cv_reduction` separately from device `fold_fit`. Serial and parallel runs over
the same evidence must produce identical policy objects and generated bytes.
The first 108-domain CPU decode run after compact-frontier reduction measured
`cv_reduction=13.578s` with 56 workers; preserve this as the current regression
baseline and investigate separately from accelerator `fold_fit`.

Every profiler descriptor retains the exact isolated launch identity, including
`M`, ordered `N`, aggregate `N`, and `K`. Dynamic counters such as throughput,
IPC, cache behavior, achieved occupancy, traffic, and duration describe only
that exact point; they are never shape-independent candidate constants and are
never model inputs for a held/unseen point. Model inputs are runtime geometry,
codebook, candidate configuration, analytical schedule geometry, and genuinely
static resource descriptors. Exact dynamic metrics are centered against the
other candidates at the same work point and added as standardized auxiliary
ExtraTrees targets beside log-regret. Their combined target variance weight is
bounded below the timing target, so they can teach which execution properties
track candidate economy without becoming a runtime dependency or overwhelming
canonical latency. Admit one metric for a candidate contest only when every
competing candidate published it; optional-event absence is neutral, never the
numeric value zero. Apply reviewed reliability weights inside the bounded
auxiliary budget: stable instructions-per-work outrank L1 behavior, and sparse
LLC or duration-derived evidence remains secondary. During coherent N/K CV,
remove held geometry across every M and format before constructing both regret
and auxiliary targets. A fold with no varying candidate-relative dynamic target
is measured-only; configuration features cannot masquerade as profiler
influence.

CPU decode retains two normalized views of the same exact counters. Per
expected MiB captures the bandwidth and weight-reuse question; per million
logical MACs captures scheduler and instruction overhead as grouped M grows.
Expose decode arithmetic intensity from runtime-visible geometry, and expose
both effective GB/s and GOP/s for GPU decode. Do not classify grouped verifier
rows as prefill merely because M is larger than the original MTP range.

Treat the profiler surrogate as two additive CV candidates. The bounded-prior
variants may reshape only training costs before an ordinary generic tree is
fitted. The cross-fitted teacher may instead choose a held candidate directly
from its complete predicted regret surface. In both cases the choice is judged
only by held canonical timing. A winning teacher is never a runtime dependency:
its out-of-fold candidate decisions are distilled by the existing publication
tournament into a bounded, total generic C++ tree.

Profiler records must also contain reviewed candidate-aware schedule geometry.
Raw `(M,N,K)` plus a categorical family name is insufficient when two kernels
publish different task grids. Derive exact runtime-visible task counts, worker
waves, final-wave utilization, rows/output values/MACs per task, and tile-tail
utilization from the same launch formulas used in production. These analytical
features contain no winner or latency label; held canonical timing decides
whether they improve selection. Add formulas symmetrically as CUDA and ROCm
candidate families enter the common fitter.

Persist each paid profiler prediction surface in the policy fit cache. Its
identity binds the complete cross-M candidate-cost pool, model-visible profiler
descriptor digest, held N/K geometry set, and exact prediction-point inventory.
Store only the domain-local training points consumed by bounded-prior fitting
and held points consumed by teacher scoring; do not persist unrelated transfer-
pool rows. Encode predictions losslessly and reject partial or foreign maps on
reload. An identical replay must report zero profiler-model workers even when
its in-memory cache starts empty; changing timing labels, descriptors, holdouts,
or requested points must produce a new immutable surface. Publish worker results
in completion order; ordered executor draining can strand completed surfaces
behind one slow forest and violates the interruption-resume contract.

Large CPU common-observation checkpoints are parsed in deterministic byte-range
process partitions and reduced in source order. Keep the measured 8-worker
default unless a new corpus benchmark proves another winner. Use
`LLAMINAR_NATIVE_VNNI_ADAPTATION_WORKERS` for controlled A/B runs and
`LLAMINAR_NATIVE_VNNI_DISABLE_PARALLEL_ADAPTATION=1` only for serial diagnosis;
parallel and serial replay must produce the identical corpus digest.

Large common-observation checkpoints are also emitted in deterministic process
partitions. Workers format private row-range shards, and the parent assembles
them in source order before one atomic publication. Control this only for A/B
diagnosis with `LLAMINAR_NATIVE_VNNI_IO_WORKERS`; parallel and serial CSV bytes
must remain identical.

Canonical observation hashing and fit-cache identity preparation are also
host-parallel. `ObservationCorpus.digest()` serializes/sorts row shards in fork
workers, merges them in canonical order, streams the historical byte sequence
into SHA-256, and memoizes the result on the immutable corpus. Domain-local
semantic, paired-edge, legacy-migration, and serial-oracle cache identities are
computed independently across policy workers. Use
`LLAMINAR_NATIVE_VNNI_CORPUS_DIGEST_WORKERS` and
`LLAMINAR_NATIVE_VNNI_POLICY_WORKERS` only for controlled A/B diagnosis; a
serial/parallel equality regression is mandatory for either implementation.
With `LLAMINAR_NATIVE_VNNI_POLICY_TIMING=1`, inspect `cache_identity`,
`cost_keys`, `profiler_identity`, `cache_lookup`, and `candidate_costs`
separately instead of treating their sum as tree-search time.

CPU prefill split migrations use
`--cpu-prefill-fit-development-lineage-plan` during fit-only replay. Supply the
immutable source aggregate/timing, all source route manifests, every historical
refinement plan in order, and every distinct split manifest named by those
plans. The analyzer routes each plan by its embedded split digest; never flatten
a mixed v8/v9 lineage onto the newest source split or omit older rounds.
For collection, pass the intended plan path even when it does not exist. The
refresh transaction first probes the exact target binaries, authors the plan
from those route manifests, and thereafter treats it as immutable resumable
evidence. Do not precompute a plan against another build.

Once a CPU-prefill corpus has candidate expansion, split lineage, or generic
refinement history, production replay MUST use its typed
`cpu_prefill_replay_recipe.v1.json`; the individual flags above are diagnostic
and collection primitives, not an operator interface. The direct supported
replay is:

```bash
scripts/refresh_native_vnni_dispatch_tables.sh \
  --backend cpu-prefill --profile all \
  --output-dir <corpus> --skip-sweep --reuse-profiler-evidence \
  --collect-cpu-prefill-sealed-after-freeze \
  --cpu-prefill-fit-replay-recipe \
    <corpus>/cpu_prefill_replay_recipe.v1.json \
  --install
```

The public `train_native_vnni_dispatch.sh --backend cpu-prefill --install`
command discovers that recipe automatically in a materialized corpus. Recipe
preflight runs before route probes and authenticates candidate-expansion
source pairing, target/source routes, distinct split generations, historical
versus current plan roles, additive timing order, profiler triplets, and
checkpoint state. A copied path-bound complete checkpoint is automatically
rebased after its complete plan partition is validated; only transaction
provenance changes, and historical timing is not reinterpreted under newer
adapter code. This adaptation launches no trainer kernel or profiler.
For a complete checkpoint, preflight forwards its authenticated raw corpus
identity to development fit, freeze, and certification; those processes must
not stream-hash the same raw sidecars again. Prefix and sealed contexts keep
their own independently derived identities.
Only an explicitly declared `prefix` checkpoint may be extended, and its input
count is mandatory. Never infer checkpoint state from file existence.

CPU-prefill sealed evidence is planned after generic freeze. Every frozen leaf
must receive one physical route witness from authenticated target route
manifests or an untimed diagnostic completion probe. Generated
`CPUPrefillAutoRefine_*` geometry must be absent from development, and the
self-contained witness plan embeds its exact C++ route. Validate that every
frozen predicate is exercised before launching timing and again during
certification; a fixed hand-authored sealed shape list is not sufficient.

Do not defer sealed-capacity planning until installation. Before final
development refinement, authenticated preflight must report the provisional
leaf count, untouched-reserve reachability, projected witness geometry,
source-format, format/M-cell, process-job, and candidate-measurement counts,
plus calibrated wall time by ISA regime and available measurement lane. Fix
the reserve commitment and coverage obligations before fitting, but keep all
holdout timings opaque until the final generic digest is frozen. Development
CV may guide fitting; sealed evidence must never do so. If a policy changes
after sealed results are inspected, burn that seal and certify against a new
untouched partition.

A burned seal is reusable development evidence only through a typed
`cpu-prefill-burned-seal-development-v1` transaction. The transaction binds
the aggregate, timing sidecar, old witness plan, and the git/build/compiler/
host/runtime/serial-policy provenance that actually measured it. Add it to the
CPU-prefill replay recipe with `cpu_prefill_replay_recipe add-burned-seal`.
Replay materializes a mixed common-observation checkpoint before fitting and
validates each original corpus-ID partition independently. Never append failed
seal CSVs under development provenance, and never reuse that opened partition
to certify the changed policy.

Track every newly discovered orchestration failure in the project document's
**Turnkey replay defect journal** before fixing it, and add a focused negative
preflight regression. A dry command rendering is not authenticated preflight.

## Shared Shape And Candidate Ownership

Use `native_vnni_dispatch.shape_manifest.load_shape_manifest()` as the one
resolved CPU/CUDA/ROCm shape inventory. `tests/v2/CMakeLists.txt` centralizes
the JSON sources used by all C++ trainers. Add Qwen release architecture data
once to the shared catalog; never add corresponding backend-specific lists.
Runtime dispatch keys remain model-agnostic: codebook, execution mode, M, N,
and K, with generic predicates based on codebook, geometry/aspect, and work.
Synthetic boundary-refinement geometries are generic-learning evidence only.
Exact overlays may be emitted only for shape/M cells declared as production
overlays in the shared manifest/prefill matrix, and emitted names must be the
canonical production names rather than synthetic evidence aliases.

Register launchable template instantiations in the common candidate registry.
The sweep must execute the real registered candidate ID and confirm the
observed route/counters. Unsupported combinations need explicit typed records,
not missing rows. Do not serialize a combinatorially projected formula corpus;
retain measured cells and evaluate formulas during fitting.

## Measurement Surfaces

- **Fast M=1 decode/GEMV:** bandwidth-sensitive serial decode. Optimize the
  fraction of sustainable memory throughput while accounting for launch and
  reduction overhead.
- **Grouped verifier M=2..16 plus sentinels:** require byte equality with each
  serial row. M=31 is evidence that runtime support is not artificially capped;
  production M is not limited to the measured buckets.
- **Ordinary prefill/GEMM:** sweep M by geometry under the bounded backend
  budget. CPU includes deep M only on smaller geometries; GPU may cover larger
  M on larger geometries. Dispatch clamps/interpolates unseen work sizes.
- **Dense and MoE:** include attention, GDN, short-conv consumers, dense FFN,
  expert gate/up/down, MTP hidden/embedding projection, and LM head geometries
  that actually route through NativeVNNI.

Sweep every supported source format and validate source-to-execution-codebook
aliases. Train on execution codebooks but retain source formats for evidence
completeness.

## Profiling And Feature Use

Derive immutable profiler requests only after timing is complete. Profile each
physical candidate launch in a fresh discrete process. Keep pipeline dispatches
(quantize/producer/reducer/epilogue) ordered and separate.

- For CUDA metrics and Nsight procedure, read
  [cuda-tuning](../cuda-tuning/SKILL.md).
- For ROCm metrics, ISA, and rocprofiler limits, read
  [rocm-tuning](../rocm-tuning/SKILL.md).
- For CPU, use trainer-owned `perf_event_open` groups attached to the exact
  persistent OpenMP worker TIDs. Arm all per-thread groups concurrently after
  warmup, execute exactly one production candidate launch, disarm concurrently,
  and reject multiplexed events. Do not use system-wide `perf stat` FIFO
  control: polling delay admits unrelated work after the target launch.

For CPU short launches, retain raw evidence but admit duration-derived
features only when the exact wall interval is at least 500 microseconds and
`task_clock / wall_clock` does not exceed the physical worker count by more
than 5%. Instructions per logical work and L1 behavior are stable primary
signals in the focused probe. Cycles, reference cycles, task clock, and IPC are
secondary only after that reliability gate. One-shot wall/task clock values are
diagnostics beside the repeated canonical timing corpus, not auxiliary timing
labels. Normalize decode counters by both expected payload MiB and logical
million MACs; normalize prefill by logical million MACs. Assign sparse LLC misses lower
reliability weight than instructions and L1 evidence. Raw counters remain in
the immutable corpus so later feature studies never require recollection.
The 2026-07-18 calibration measured roughly 0.2% repeat CV for instructions and
L1 traffic on grouped decode, versus up to 20% for LLC misses on sub-500-us
launches. On a 152064x5120 Q4_0 launch, Pairwise M=16 executed about 7.9 times
the M=2 instructions in only 2.0 times the wall duration while LLC misses stayed
approximately flat, directly demonstrating grouped weight reuse. At equal M,
WideRows used about 6% more instructions and 47% more L1 loads than Pairwise on
that geometry despite fewer misses and slower canonical timing. Never interpret
a lower miss count without its total traffic and logical-work denominators.

CPU collection amortizes process startup with exact-request TSV batches.
Requests may share one process only when build/ISA/threading, operation,
contract/bundle, execution mode, and N/K geometry match. M, format, and
candidate may vary within that process. The trainer reuses packed fixtures and
the persistent OpenMP team, but it rebinds request/output identity, resets and
arms every per-TID event group, executes one candidate launch, disarms, reads,
and atomically publishes before advancing. Each request directory contains only
its perf CSV, binary provenance, and exact batch-member record; shared process
logs and canonical-style diagnostics live outside request evidence digests.
Batch completion is all-ID checked and appended once to the authenticated
`<evidence>.inprogress.jsonl` journal. The journal fsyncs each completed batch,
recovers every newline-complete member after interruption, and materializes the
full canonical evidence manifest only when the invocation exits. Never rewrite
the complete growing JSON after each request or batch. Use
`--disable-cpu-process-batching` only to diagnose the amortized protocol, and
adjust `--cpu-process-batch-size` only after a focused throughput/reliability A/B.

Normalize metrics into architecture-relative features that explain candidate
economy. Decode emphasizes memory efficiency, transactions, launch structure,
and occupancy needed to hide latency. Prefill emphasizes compute utilization,
tensor/VNNI issue, occupancy, register/scratch pressure, and avoidable memory
traffic. Profiler features may influence the fit only after authenticated join
to the exact timing corpus or its exact compact request witness. A broad
observation superset is not itself a witness. Run feature ablations when their
effect is unclear.

## Fit And Certification

Exact seen-shape overlays lock the best observed launch. Generic rules are fit
by codebook, execution mode, geometry/aspect, and work size and must dispatch
every unseen smaller or larger shape. Use geometry-atomic grouped CV, not
random row CV or shape-name grouping: all aliases at one exact `(N,K)` must
leave training together. For sufficiently large domains, withhold coherent
balanced N/K regions so both local neighborhoods and low/high support
boundaries are tested; spreading adjacent points round-robin across folds
proves interpolation only and is not an unseen-shape certificate. Small
domains use leave-one-geometry-out validation. The fold schema participates
in fit-cache identity.
Freeze generic IR before opening sealed evidence. Evaluate sealed shapes with
exact overlays disabled and do not refit after seeing them.

Promotion is domain based:

1. every required codebook/ISA/execution-mode/M domain must have complete CV
   coverage and a nonempty generic tree;
2. each domain independently computes nearest-rank p95 observed regret and p95
   simultaneous-UCB regret, with both required to be strictly below 5%;
3. at least 99% of required domains must pass those two performance gates;
4. the remaining less-than-1% performance exceptions keep their best measured
   generic trees and typed diagnostics so runtime dispatch remains total; and
5. missing domains, uncovered points, empty trees, ambiguous predicates,
   unexercised sealed leaves, byte mismatches, or M/N*K/codebook holes are
   structural failures and cannot consume the performance-exception allowance.

Global p95 and maximum regret remain useful refinement diagnostics, but neither
may replace the per-domain decision. Exact overlays never count toward generic
coverage, CV, sealed rule exercise, or the 99% quota.

Production fitting auto-discovers every CUDA and ROCm device. Rebuild and test:

```bash
cmake --build build_v2_release --parallel \
  --target v2_native_vnni_leaf_primary_scorer_cuda \
           v2_native_vnni_leaf_primary_scorer_rocm
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_NativeVNNILeafPrimaryScorer_(CUDA|ROCm)$' \
  --output-on-failure --parallel
```

Keep uploaded regret matrices, streams, and scratch buffers persistent in GPU
scorers. Batch leaves, refill lanes dynamically, and restore deterministic task
order before policy reduction. Do not synchronize after each leaf.

Grouped CV uses scorer ABI v12's fused
`llaminarNativeVNNITreeSearchAndEvaluate` path. One graph replay must perform
the complete beam search, held-out threshold precomputation, exact-candidate
selection, tree routing, and compact fold-result publication on an explicit
non-default stream. Persistent device scratch owns all matrices, descriptors,
trees, and reductions. The only synchronization is one terminal stream sync
after the compact final D2H; full-device syncs, per-leaf syncs, temporary host
trees, and Python held-out replay are contract violations. Keep the standalone
tree-publication entry point only for final policy publication, where the host
actually needs the stable tree IR.

Keep expanded candidates compact and virtual. A candidate stores its parent
frontier index, split descriptor, scalar objective, and exact signature hash;
it does not copy a complete leaf array. Compute the positioned-token signature
hash with one 32-lane subgroup, then compare every colliding canonical token
sequence in full. Hash equality is only a lookup accelerator, never a
correctness predicate. Preserve the dense one-thread, left-to-right FP64 mean,
but compute order-only expanded-tree p95 in the existing subgroup finishing
kernel. Candidate-column p95 uses reversible uint64 IEEE-754 order keys and an
exact bounded upper-tail scan while retaining the selected source bits.

All scratch, graph executables, streams, and matrices are allocated or grown
before capture/replay and then reused. Stable replay must not allocate, free,
grow scratch, recapture, or synchronize except for the single terminal stream
wait after compact D2H publication. Keep runtime-counter regressions for these
invariants. Timeline allocation records across test cases are acceptable only
when they belong to session setup, reviewed high-water growth, or teardown,
never to a captured graph node or replay.

ABI v12 carries eight 64-bit mask words and therefore supports up to 512
training points and 512 held-out points in one captured transaction. Keep the
CUDA/ROCm integration regression at the production high-water mark of 444
collapsed-aspect CPU decode points. A smaller test would miss the ABI v7
pre-capture failure that originally surfaced this requirement.

Build both scorer DSOs with explicit device `-O3`, including Integration
builds. CMake custom build types do not automatically imply CUDA/HIP
optimization. Profile scorer-kernel changes with isolated integration launches:

- CUDA: ncu `SpeedOfLight`, `Occupancy`, `LaunchStats`,
  `MemoryWorkloadAnalysis`, and `WarpStateStats`;
- ROCm: inspect code-object private segment, stack, SGPR/VGPR counts, and spills,
  then collect `VALUUtilization`, `MemUnitBusy`, and `L2CacheHit` in separate
  rocprofiler passes on gfx906; and
- reject any production kernel with local/private scratch or register spills.

The current profiled scorer has zero CUDA local-memory spills and zero ROCm
scratch/private allocation in child scoring, compact objective finishing,
deduplication, and top-k selection. The retained subgroup signature hash reduced
representative deduplication from 2.162 ms to 0.011 ms on GA102 and from
13.654 ms to 0.026 ms on gfx906. Splitting exact ordered mean from order-only
p95 reduced objective-plus-finishing time from about 2.115 ms to 1.134 ms on
GA102 and from about 4.492 ms to 2.232 ms on gfx906. Treat these as regression
baselines, not timeless thresholds; remeasure after toolchain or ABI changes.

The held-out scorer kernels are integer predicates/reductions plus exact FP64
comparisons. Tensor cores do not accelerate this workload and would weaken the
exact-key contract; use them only if a future scorer stage becomes a real
matrix operation and exact equivalence is separately proved.

## Kernel Tuning Loop

1. Reproduce one dispatch/economy gap in a focused `Perf__` test.
2. Add the candidate to the real templated registry and route it through the
   production dispatch surface.
3. Prove all-format byte equality for grouped rows before considering timing.
4. Gather stable canonical timing and separate profiler evidence.
5. Inspect occupancy, vector/tensor/VNNI issue, register/VGPR pressure, spills,
   memory transactions, barriers, and grid utilization.
6. Keep real winners even when they increase learner complexity; improve the
   generic fit rather than deleting useful kernel families.
7. Refit from the retained corpus, certify, install, run all-format integration
   sweeps, then run model-level parity and performance gates.

For a backend handoff, finish the complete transaction before starting another
backend: freeze, sealed-certify, validate the staged artifact/include, install,
run backend smoke coverage through production dispatch, and preserve the
result. CUDA and ROCm collection/fitting should otherwise remain symmetric and
may run concurrently on disjoint devices; always rebuild and use both scorer
families so policy search can use every available GPU.

Do not tune one model format in isolation. Turn every discovered format/shape
bug into an all-format regression when the behavior is format-generic.

## Required Validation

At minimum run:

```bash
ctest --test-dir build_v2_integration \
  -L Unit -L NativeVNNI \
  --output-on-failure --parallel
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_.*NativeVNNI|^V2_Integration_.*VerifierRows' \
  --output-on-failure --parallel
```

The unit command is deliberately label-based. Every pure-Python corpus,
lineage, measurement-plan, profiler-evidence, fitter, and generated-policy
regression must be registered with both labels so adding a module cannot make
it invisible to the canonical gate. Python unit tests must never launch GPU
work; scorer execution belongs in the integration suite.

Also run backend all-format grouped equivalence, graph-captured/eager, dense/MoE,
and model parity suites touched by the change. Build performance targets from
release builds and correctness/integration targets from the integration build.

## Internal Reference

Read [references/pipeline.md](references/pipeline.md) when changing corpus
schema, refresh phases, fitter/scorer internals, generated artifacts, or resume
semantics. The full design and current status remain in
`docs/v2/projects/2026-07/NATIVE_VNNI_BATCH_INVARIANT_LEARNED_DISPATCH_POLICY.md`.
