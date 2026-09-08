# NativeVNNI Dispatch Pipeline Reference

## Contents

- [Authority Map](#authority-map)
- [Evidence Flow](#evidence-flow)
- [Corpus Publication](#corpus-publication)
- [Backend Artifacts](#backend-artifacts)
- [Fit Internals](#fit-internals)
- [Failure And Resume Semantics](#failure-and-resume-semantics)
- [Change Checklist](#change-checklist)

## Authority Map

| Concern | Authority |
|---|---|
| Turnkey transaction | `scripts/train_native_vnni_dispatch.sh` |
| Phase collection/fit executor | `scripts/refresh_native_vnni_dispatch_tables.sh` |
| Immutable bundle | `native_vnni_dispatch/corpus_bundle.py` |
| Shared shape inventory | `native_vnni_dispatch/shape_manifest.py` and its centralized JSON sources |
| Qwen release geometry derivation | `native_vnni_dispatch/qwen_release_geometry.py` |
| Format/codebook aliases | `tests/v2/performance/kernels/native_vnni_codebooks.py` |
| Candidate registry | `native_vnni_dispatch/candidate_registry.py` |
| Common observation schema | `native_vnni_dispatch/schema.py` |
| Corpus validation | `native_vnni_dispatch/corpus.py`, `validation.py`, backend adapters |
| Generic learner | `native_vnni_dispatch/segmented_policy.py` |
| Policy IR/compiler | `native_vnni_dispatch/policy_ir.py`, `compiler.py` |
| Profiler requests/evidence | `profiler_evidence.py`, `profiler_collectors.py`, `profiler_model.py` |
| GPU tree search and grouped-CV scoring | `policy_accelerator.py`, `NativeVNNITreeSearchGpu.cuh` |
| CPU prefill route/split lineage | `cpu_prefill_*` modules beneath `native_vnni_dispatch/` |

The backend analyzer scripts adapt evidence and invoke common policy machinery;
they must not implement independent learning contracts.

## Evidence Flow

```text
shared shape inventory + format aliases + candidate registry
                         |
                         v
production trainer launches -> raw aggregate + raw timing sidecar
                         |
                         v
backend adapter -> validated common observations
                         |
          +--------------+---------------+
          |                              |
          v                              v
canonical median labels       isolated profiler requests
                                         |
                                         v
                              ncu / rocprof / perf launches
                                         |
                                         v
                              authenticated normalized features
          |                              |
          +--------------+---------------+
                         v
grouped shape CV -> 99%-domain gate -> frozen generic IR -> sealed domain gate
                         |
                         v
exact overlays + generic rules -> generated .inc -> runtime resolver tests
```

The timing sidecar owns raw samples and stationarity. The aggregate owns
candidate correctness, route identity, and summary timing. Profiler evidence
owns hardware counters from a separate launch. Digests bind every join.

## Corpus Publication

The turnkey driver computes:

1. backend identity;
2. architecture identity;
3. resolved shape-inventory SHA-256;
4. canonical digest of forwarded collection arguments.

It collects into
`benchmark_results/native_vnni_dispatch/work/collect-*`. CPU partials are
published by atomic rename and resumed only when their collection contract
matches. A successful complete refresh is sealed by `corpus_bundle.py`, then
the whole directory is renamed into `corpora/native_vnni_dispatch/` in the
optional `Llaminar/corpora` submodule. The data repository owns LFS pulls and
payload commits; the source repository still owns inventory-source provenance.
Push the data commit and its LFS objects before advancing the source gitlink.

The corpus manifest binds:

- schema version;
- backend/profile/architecture;
- collection argument digest and ordered arguments;
- resolved shape-inventory digest and source-file digests;
- every payload's relative path, byte size, and SHA-256;
- a corpus ID derived from all authenticated fields.

Do not include fit caches in the corpus identity. They are disposable
performance artifacts. Do retain refinement plans, route manifests, request and
evidence JSON, exact compact profiler observation witnesses, raw profiler
reports, timing sidecars, and all candidate rows.
For a production `all` profile, the bundle validator also requires the complete
backend development, sealed, profiler, certified-policy, and generated-include
artifact set. This prevents an intentional collection checkpoint from becoming
a publishable corpus.

For refit, verify and copy/reflink the corpus into an ignored `fit-*` workspace.
Remove the copied corpus manifest, rebuild current binaries/scorers, then invoke
the refresh executor with `--skip-sweep --reuse-profiler-evidence`. Generated
artifacts may change in the workspace; corpus bytes may not.

An additive CPU prefill lineage additionally passes
`--cpu-prefill-fit-development-lineage-plan`. The refresh executor forwards
separate source and target route manifests plus every source refinement plan and
its authored split through adaptation, development CV, freeze, and sealed
certification. The stable fit input names are
`cpu_prefill_sweep.development-lineage.combined.csv` and its timing sidecar.

## Backend Artifacts

### Additive dense GPU prefill

After a filtered `production_dense_prefill_sweep` plan has completed and passed
`combine`, generate its exact delta while retaining the previous include:

```bash
PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_vnni_dispatch.dense_production_overlay \
  --backend <cuda|rocm> --work-dir <complete-work-dir> \
  --retain-base-include <immutable-installed-base.inc> \
  --retention-receipt <result-dir>/retention.json \
  --output <candidate.inc> --summary-csv <result-dir>/fresh-winners.csv
```

The runtime selector and unrelated base rows are preserved byte-for-byte;
changed ABI, duplicate/unsorted keys and noncanonical row syntax fail before
publication. CUDA rows with the explicit struct-default staging field omitted
remain unchanged when outside the delta. The receipt binds base, measured delta
and final include bytes and their key cardinalities. It is published after the
include and must match its output digest; interruption is not certification.
Keep the immutable source alongside the new plan and timing evidence. Retained
rows are not relabeled as fresh measurements, and generic rules never enter this
operation. Validate installed route selection, all-format byte equivalence,
isolated resources/spills and matched Release throughput before accepting the
candidate; a successful source merge alone is not promotion.

### CUDA decode and grouped verifier

An explicitly requested exact-only refresh is independent of generic training.
Use the CUDA analyzer's paired `--retain-auto-policy-json` and
`--retain-auto-include` options with the production exact-only/completeness
flags. The emitter rejects a changed selector ABI, absent generic/grouped
sections, duplicate keys, or an unauthenticated base. It scopes old certificate
provenance to the retained policy and emits new exact-corpus provenance
separately; it must never relabel the changed full include as the old sealed IR.
Only shared production geometries receive exact precedence. M1 changes still
require grouped arithmetic and model-level regression gates; retaining grouped
launch code does not prove its behavior under a newly inherited partition.

The registry distinguishes global `kpar` from CTA-local `fused_kpar` while
sharing their exact-KB arithmetic fingerprint. The analyzer emits a distinct
physical family, and the runtime's canonical-M1 query exports the same ordered
partition count to grouped DP4A and tensor-core consumers. Fused candidates
carry compiler/occupancy admission fields in raw aggregate evidence; a missing
resource proof or spilled specialization is not eligible for timing/install.
The profiler surrogate derives task grids and partial-storage volume from the
current runtime geometry, never from a differently shaped profiling anchor.

Both publication families use nominal shape-resolved recipes for generic
fitting. Literal counts remain physical exact-overlay evidence only. The
projection preserves the exact timing/output witness and resolves the recipe
to a legal concrete count before dispatch. A generated leaf may not add a
hidden K-admission predicate: that would invalidate the common tree's totality
proof. Profiler requests name the physical witness, never the virtual recipe.

- development M1 aggregate/timing;
- paired development requests/evidence/certificates;
- M1 profiler requests/evidence/raw reports;
- frozen M1 policy IR and sealed M1 aggregate/timing;
- grouped verifier aggregate/timing against staged M1;
- final profiler evidence and generated include;
- install target:
  `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc`.

### ROCm decode and grouped verifier

- Fast development aggregate/timing and profiler evidence;
- frozen Fast policy and sealed aggregate/timing;
- verifier aggregate/timing against staged Fast policy;
- final profiler evidence and generated include;
- install target:
  `src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc`.

### CPU grouped verifier

- development/sealed aggregate and timing for all three ISA regimes;
- perf request/evidence/features per physical candidate;
- frozen/certified policy and generated include;
- install target:
  `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc`.

### CPU ordinary prefill research

- serial route manifests for each build/runtime ISA regime;
- split manifest and authenticated development lineage;
- all historical source splits when refinement rounds span split revisions;
- candidate-expansion and generic-refinement plans plus additive timing;
- immutable profiler source observations and profiler evidence;
- frozen policy, sealed witness plan, sealed aggregate/timing, and certificate.

This surface is diagnostic only. It has no source-tree install target and must
not be consumed by production; ordinary prefill remains on the total runtime
heuristic for every backend.

## Fit Internals

Runtime points are grouped by semantic contract, backend architecture, codebook,
execution mode, shape geometry, and M/work bucket. Generic model selection uses
shape-group CV to prevent rows from one geometry leaking across folds. Exact
overlays are selected from measured seen-shape winners after generic policy
certification.

Prediction-cache point inventories retain one canonical order and a length-framed
JSON digest. Reuse the existing flattened runtime sort key and standard-encoder
value fragments rather than serializing every runtime field for every candidate.
An independent historical-encoder regression owns byte compatibility, including
all runtime discriminators and JSON escaping. Runtime/domain structural hash
caches accelerate local lookups only; pickle carries value fields without the
process-salted cache so equal keys remain interchangeable across interpreters.

Compact CV label views belong to the scorer lane, not a process-global cache.
Keep at most one active source-cost tuple, held-out geometry set and read-only
prediction surface; retain only demanded profiler-influence variants within
that scope. Identity includes the prediction inventory and backing array, so
rebinding a diagnostic surface invalidates retained views. Mutable mappings
must be materialized afresh; writable production arrays are a hard error.
Feature policy and boundary placement are label-invariant but still require
independent tree evaluation. Measured regret fields and final CV results must
equal the uncached oracle. Join the lane's executor tasks before releasing its
views and scorer, including partially initialized worker failure paths.

GPU scorers accelerate exact candidate-primary key calculation, complete beam
search, and grouped held-out evaluation. Each worker owns one vendor runtime,
ordinal, non-default stream, uploaded matrix, and persistent growable scratch.
Host orchestration schedules independent fit tasks dynamically across all
workers. Accelerator output must be byte/exact-key equivalent to the Python
serial oracle; task scheduling cannot alter final IR.

Scorer ABI v8 fuses grouped CV into one captured device pipeline: beam search,
threshold-predicate precomputation, exact held-out winner selection, fitted-tree
routing, and compact fold-result publication. All temporary trees and
held-out reductions remain device resident. One compact D2H and one terminal
stream synchronization are allowed after graph replay; intermediate stream
synchronizations, full-device synchronizations, per-leaf host publication, and
Python held-out replay are forbidden. Final policy publication uses the
separate tree-returning API because stable IR is its actual output.

Eight-word point masks provide a reviewed 512-point graph capacity. CUDA and
ROCm integration tests exercise the current 444-point collapsed-aspect CPU
decode domain so corpus growth beyond the old 256-point ABI remains covered.

CUDA/HIP scorer targets require explicit `-O3` even in the Integration build.
Vendor profiling must show zero local/private scratch and zero SGPR/VGPR or
CUDA local-memory spills. On gfx906, collect rocprofiler counters in compatible
separate passes instead of requesting an oversized combined PMC set.

The generic dispatch must remain total for unseen shapes and work sizes. Clamp
or extrapolate using geometry/aspect/work predicates; never fail dispatch merely
because an exact M/N/K tuple was absent from the corpus.

Development and sealed promotion use the same domain contract. Every required
domain must have complete evidence coverage and a nonempty generic tree. Both
nearest-rank p95 observed regret and nearest-rank p95 simultaneous-UCB regret
are computed per domain and must be strictly below 5% for that domain to pass.
At least 99% of required domains must pass. The less-than-1% performance
exception set retains its best measured generic trees and typed diagnostics;
it never becomes a dispatch hole. Structural failures such as missing domains,
empty trees, uncovered points, byte mismatches, unexercised sealed leaves, or
M/N*K/codebook totality failures are outside the quota and always block.

Global cell p95 and maximum regret are retained only to guide refinement. Exact
overlays are fitted after generic policy work and are disabled during generic
CV and sealed evaluation, so they cannot improve either coverage or quota.

## Failure And Resume Semantics

| Failure | Action |
|---|---|
| Interrupted CPU collection | Rerun identical turnkey command; resume only validated atomic partials. |
| Unresolved LFS pointer | Run `git lfs pull` for the corpus; do not parse pointer text. |
| Shape inventory changed | Select a new digest generation and collect missing required evidence. |
| Build/route/candidate registry changed | Reject incompatible timing or create an authenticated migration/expansion plan. |
| Profiler feature experiment changed | Re-export/remine the same profiler evidence from its compact timing witness; recover a missing older witness from the digest-bearing feature table; do not retime kernels. |
| Nsight SI byte units were decoded as IEC | Use the offline `profiler_reparse` transaction on completed CUDA batch evidence. Authenticate original raw reports and stream membership, change only byte-valued metric interpretation, retain the original manifest and lineage, and never repeat hardware measurements. |
| Fewer than 99% of domains pass strict p95/UCB | Refit from the same corpus or generate a focused authenticated refinement plan for the typed over-budget domains. |
| Up to 1% of domains exceed strict p95/UCB | Retain explicit diagnostics and real total generic trees; continue only when every structural gate passes. |
| New real candidate wins | Keep it, expand generic evidence/model capacity, and recertify. |
| Sealed gate failed | Do not refit on sealed rows. Return to development evidence/model design. |
| Scorer DSO/device failed after explicit selection | Hard fail; do not continue on CPU. |
| Generated include validation failed | Do not install; fix compiler/IR and regenerate from the same corpus. |

## Change Checklist

When changing the pipeline:

1. update schema/version and digest inputs deliberately;
2. update all backend adapters symmetrically;
3. update shared inventory rather than backend shape arrays;
4. preserve canonical timing versus profiler isolation;
5. preserve fit-only zero-kernel behavior;
6. add focused unit tests for manifest/corpus/compiler logic;
7. put GPU scorer or kernel execution tests in integration/performance;
8. verify CPU/CUDA/ROCm generated resolver IDs;
9. run all-format grouped byte-equivalence suites;
10. update `SKILL.md`, this pipeline reference, and the July project document
    in the same change.
