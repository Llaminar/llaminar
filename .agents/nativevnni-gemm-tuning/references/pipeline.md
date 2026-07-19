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
the whole directory is renamed into `corpora/`.

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

### CUDA decode and grouped verifier

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

### CPU ordinary prefill

- serial route manifests for each build/runtime ISA regime;
- split manifest and authenticated development lineage;
- all historical source splits when refinement rounds span split revisions;
- candidate-expansion and generic-refinement plans plus additive timing;
- immutable profiler source observations and profiler evidence;
- frozen policy, sealed witness plan, sealed aggregate/timing, certificate;
- install target:
  `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIPrefillPolicyGenerated.inc`.

## Fit Internals

Runtime points are grouped by semantic contract, backend architecture, codebook,
execution mode, shape geometry, and M/work bucket. Generic model selection uses
shape-group CV to prevent rows from one geometry leaking across folds. Exact
overlays are selected from measured seen-shape winners after generic policy
certification.

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
