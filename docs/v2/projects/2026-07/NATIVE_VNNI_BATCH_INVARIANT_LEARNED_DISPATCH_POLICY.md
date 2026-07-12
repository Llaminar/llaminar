# Cross-Backend Batch-Invariant NativeVNNI Learned Dispatch Policy

- **Date**: 2026-07-11
- **Status**: Implementation in progress; common compiler and CPU verifier migration operational
- **Scope**: CPU, CUDA, and ROCm NativeVNNI GEMV/GEMM dispatch for `Fast M=1..4`, a frozen serial-M1 oracle, and bitwise MTP verifier rows `M=2..4`
- **Parent project**: [vLLM-Style MTP Project Plan](../2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md)
- **Evidence dashboard**: [vLLM-Style MTP Tuning Dashboard](../2026-06/MTP_VLLM_STYLE_TUNING_DASHBOARD.md)
- **Normative language**: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY` are requirements levels.

---

## 1. Executive Decision

Llaminar will use one shared, numerics-constrained dispatch compiler for CPU,
CUDA, and ROCm NativeVNNI small-M GEMV/GEMM policies. Each backend will retain
its own candidate kernels and generated C++ include, but all three backends will
use the same:

- semantic dispatch contracts;
- observation schema;
- candidate eligibility model;
- alias-robust runtime exact-winner construction;
- generic aspect/work learner;
- shape-grouped development/CV and sealed-certification protocol;
- maximum 3% performance-regret gate;
- generated-policy intermediate representation;
- provenance and stale-policy checks; and
- staged validation and installation transaction.

The compiler pipeline is:

```text
backend candidate measurements
              |
              v
contract-specific correctness eligibility
              |
              v
complete candidate-by-shape cost matrix
              |
              +------> alias-robust runtime exact winner per key
              |
              v
common bounded aspect/work learner minimizing measured latency regret
              |
              v
grouped development CV, then fit/freeze final generic policy on development
              |
              v
sealed shape certification, exact overlays disabled, no refit
              |
              v
100% sealed coverage + max worst-surface regret <= 3.0%
              |
              v
exact overlays + certified generic rules + policy manifest
              |
              v
CUDA / ROCm / CPU emitters
              |
              v
staged compile + route tests + byte parity + model parity
              |
              v
atomic bundle publish
```

There are two independent dispatch contracts:

1. `Fast`: ordinary performance-oriented execution using the backend's normal
   numerical correctness requirements.
2. `VerifierSerialM1Bitwise`: grouped MTP verifier execution where every
   `M=2..4` FP32 output row MUST be byte-for-byte identical to the same row
   produced by the backend's production serial `M=1` decode path.

The coverage matrix is:

```text
Fast                         M=1,2,3,4
FrozenSerialM1Oracle         M=1 dependency selected and staged first
VerifierSerialM1Bitwise      M=2,3,4 certified against that frozen dependency
```

Both contracts receive first-class exact overlays and generic aspect/work
rules. They are trained and scored separately. A verifier generic rule is
compared with the byte-certified alias-robust verifier exact winner, never
with an unsafe `Fast` winner.

### 1.1 Implementation status on 2026-07-11

The shared schema, candidate registry, exact oracle, segmented-regret learner,
common corpus, and CPU/ROCm verifier adapters are implemented. The CPU verifier
path is the first fully trained production consumer of the common compiler.
Its dispatch key includes all runtime-visible performance dimensions:

- compile-time ISA (`AVX2` or `AVX512`);
- effective runtime ISA (`AVX2` or `AVX512`);
- OpenMP thread count;
- normalized runtime codebook;
- verifier depth `M`; and
- exact `N` and `K`, followed by certified generic aspect/work rules.

CPU training MUST collect three independent regimes. An AVX512-capable host
does not make the first two interchangeable:

1. AVX2 build with AVX2 runtime dispatch;
2. AVX512 build with runtime dispatch forced to AVX2; and
3. AVX512 build with native AVX512 runtime dispatch.

The AVX2 and AVX512 builds use independent oneDNN build/install directories
(`external/onednn/build-avx2` and `external/onednn/build-avx512`) so configuring
one build cannot silently replace the linked ISA regime of the other.

The first production CPU corpus used 28 threads on an Intel Xeon Gold 6238R
and retained 4,536 strong aggregate observations plus 84,924 raw timing rows.
It covered every registered source-format alias, 18 runtime codebooks,
verifier `M=2..4`, twelve Qwen 3.6 dense/MoE production shapes, and all three
ISA regimes. Every accepted candidate had zero output-byte mismatches and zero
repeat-byte mismatches. The compiler emitted 1,944 exact policies plus the
cross-validated generic policy into:

```text
src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc
```

The reproducible production refresh command is:

```bash
scripts/refresh_native_vnni_dispatch_tables.sh \
  --backend cpu \
  --profile qwen36 \
  --cpu-threads 28 \
  --cpu-avx2-sweep-bin build_v2_release/tests/v2/v2_perf_cpu_native_vnni_gemv \
  --cpu-avx512-sweep-bin build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_gemv \
  --output-dir benchmark_results/native_vnni_dispatch/<run-id> \
  --install
```

The wrapper rejects CPU installation from smoke or partial profiles, rejects
an incomplete ISA matrix, and validates generated codebook references before
publishing. Production resolution has no serial-row or untrained-policy
fallback: an uncertified build/runtime/thread domain fails closed.

CUDA common-compiler migration and the remaining ROCm batched-projection
trainer remain open. Therefore this document is not yet a claim that the
cross-backend transaction or sealed-corpus release gate is complete.

## 2. Authority, Applicability, and Non-Goals

### 2.1 Authority

This policy is normative for NativeVNNI small-M dispatch training, generation,
runtime selection, and table installation after it is accepted. It refines the
grouped-verifier requirements in Phase 9.8 of the parent MTP plan.

Where older NativeVNNI documents accept cosine similarity, relative L2, KL
divergence, or nonzero maximum absolute error for a grouped verifier kernel,
this policy takes precedence. Those metrics remain useful diagnostics and
model-level checks, but they cannot waive byte equality at the grouped-kernel
publication boundary.

Historical background, not normative for verifier eligibility:

- [NativeVNNI Project Plan](../2026-02/NATIVE_VNNI_PROJECT_PLAN.md)
- [NativeVNNI GEMV Expansion](../2026-03/NATIVE_VNNI_GEMV_EXPANSION.md)
- [NativeVNNI GEMM Design](../2026-03/NATIVE_VNNI_GEMM_DESIGN.md)
- [CUDA Dual-Format Quantised GEMM Plan](../2026-03/CUDA_DUAL_FORMAT_QUANTISED_GEMM_PROJECT_PLAN.md)
- [ROCm Dual-Format Prefill GEMM Plan](../2026-02/ROCM_DUAL_FORMAT_PREFILL_GEMM_PROJECT_PLAN.md)

### 2.2 Initial applicability

The first implementation covers:

- CPU, CUDA, and ROCm;
- NativeVNNI prepared quantized weight families and every source-format alias
  advertised by the backend;
- `Fast M=1,2,3,4`, a frozen production serial-M1 oracle, and verifier
  `M=2,3,4` projections;
- single projection, fused projection groups, fused gate/up, fused
  SwiGLU/down, dense, and routed/shared MoE paths that use the same prepared
  kernels;
- exact production shapes and generic nearby shapes; and
- eager and captured execution where graph capture is a production route.

The framework SHOULD be reusable for large-M prefill policy generation, but
large-M prefill migration is not required for the first batch-invariance
release.

### 2.3 Non-goals

This policy does not require:

- the same kernel candidate IDs on different backends;
- identical performance choices across different hardware architectures;
- one universal policy trained across CUDA SM, ROCm GFX, and CPU
  microarchitectures;
- replacing backend-specific kernel implementations;
- accepting serial row replay as an economical grouped verifier
  implementation; or
- making a generic performance claim for a bucket with insufficient independent
  sealed evidence.

The commonality is the contract, data, learner, evaluation, and promotion
mechanism. Candidate enumeration and C++ encoding remain backend adapters.

## 3. Baseline Audit Before Migration

The table below records the implementation at project start. Section 1.1 is the
authoritative current progress snapshot; this baseline is retained to explain
why each migration is required.

| Backend | Existing behavior | Policy gap |
|---|---|---|
| CUDA | A 3%-set-cover plus `log2(M,N,K)` decision tree is followed by a separate exact-overlay/aspect analyzer. | The tree evaluates its training corpus, not held-out shapes. The overlay analyzer scores newly inferred aspect/work rules while `--base-include` preserves a different tree fallback, so the scored generic policy can differ from the emitted policy. Trainer correctness is not a verifier byte certificate. |
| ROCm | Exact winners are emitted above aspect/work rules. | The trainer/generator eligibility collapses the candidate matrix to exact winners, then optimizes modal winner labels. Its reported metric is in-sample exact-label hit rate, not latency regret, and its trainer eligibility permits approximate output despite stronger byte-exact integration tests. |
| CPU | A generated exact `(codebook,M,N,K)` table selects Pairwise or WideRows. | There is no generic learner. `Auto` measurements can be labeled as a policy they did not execute, fused/MoE paths do not all consult the same selector, and trainer/generator eligibility is tolerance-based despite stronger byte-exact integration tests. |

The refresh wrapper currently validates generated codebook IDs and then copies
each `.inc` independently. It does not prove byte eligibility, candidate
reachability, held-out generic regret, M1-oracle freshness, or transactional
all-backend installation.

Primary audited surfaces are listed in Appendix D.

## 4. Terminology

**Semantic contract**

The numerical and execution contract under which a candidate is selected.
This policy defines `Fast` and `VerifierSerialM1Bitwise`.

**Source format**

The externally loaded tensor format, such as a GGUF Q/K/IQ format.

**Prepared family**

The concrete runtime weight representation consumed by a kernel. Multiple
source formats may map to one prepared family or dispatch codebook.

**Runtime alias group**

All source formats that become indistinguishable at the runtime dispatch key.
A policy cannot choose different candidates for members of the same alias
group.

**Candidate ID**

A stable backend-specific name for a forceable kernel configuration.
`Auto` is not a candidate ID.

**Effective candidate ID**

The configuration actually launched after contract normalization. Nominal
candidate IDs that normalize to the same launch share one effective ID for
training purposes.

**Arithmetic fingerprint**

A stable description of activation quantization, weight decode, accumulator
type, K partition, partial-sum layout, reduction order, epilogue order, and
atomic/non-atomic behavior for an effective launch.

**Serial M1 oracle**

The actual production `M=1` decode route for the backend, prepared family,
operation kind, and shape. It is not a separate reference implementation.

**Alias-robust runtime exact winner**

The eligible runtime-representable candidate selected for a measured exact key
by minimizing worst-alias and required-execution-mode normalized latency under
one semantic contract. It need not be the fastest candidate for every alias.

**Exact overlay**

A generated entry keyed by the full exact runtime shape and placed above the
generic policy.

**Generic rule**

A generated candidate selection keyed by prepared family/codebook, semantic
contract, `M`, aspect bucket, and a bounded work-size segment.

**Certified floor**

A genuinely grouped, contract-eligible candidate proven safe for the declared
domain. It is not an unrestricted fast fallback and is not hidden row replay.

**Performance regret**

The relative latency loss of a selected candidate against the alias-robust
runtime exact winner for the same semantic contract and runtime key.

**Policy ABI**

The versioned meaning of dispatch keys, candidate IDs, arithmetic
fingerprints, generated fields, and lookup order.

## 5. Non-Negotiable Invariants

### 5.1 Verifier byte equality

For every supported backend, prepared family, operation kind, source-format
alias, shape, deterministic payload seed, and `M in {2,3,4}`:

```text
grouped_output[M,N]
    == bytewise_concat(
           production_serial_M1(row_0),
           ...,
           production_serial_M1(row_M-1))
```

The comparison MUST cover every logical `M x N` FP32 result element row by row
using `memcmp` or an equivalent exact byte comparison. Leading-dimension
padding is compared only when it is initialized and explicitly part of the
operation contract. A one-ULP or signed-zero difference is failure.
Correctness fixtures MUST reject unexpected NaNs rather than treating matching
NaN payload bytes as useful proof.

Cosine, relative L2, symmetric KL, maximum absolute error, and sampled-token
equality MUST NOT be used as candidate eligibility substitutes.

The proof has two layers:

1. a primitive kernel proof starting from identical prepared weights and
   identical prequantized activation bytes; and
2. a public-operation proof starting from the same input tensor and covering
   activation quantization, alpha/beta behavior, bias, and every fused
   epilogue inside the declared `operation_kind`.

Both layers are required where the production operation owns those steps.

### 5.2 Stable oracle

The production serial M1 oracle MUST be byte-stable across repeated hot runs
and the supported first-use/cold preparation route. An unstable oracle makes
the affected verifier domain uncertifiable.

### 5.3 Contract isolation

`Fast` and `VerifierSerialM1Bitwise` MUST resolve independently. A fast M=2..4
entry, environment override, atomic reduction option, or ambient tuning flag
MUST NOT influence verifier resolution unless that exact effective candidate
is separately verifier-certified.

### 5.4 Exact and generic parity of treatment

Every backend MUST emit both exact overlays and generic rules through the same
common policy IR. CPU MUST NOT remain an exact-only special case.

### 5.5 Exact precedence

Runtime lookup MUST test an exact overlay before a generic rule. Sealed
certification MUST bypass exact overlays so exact entries cannot mask a broken
generic policy.

### 5.6 Held-out 3% maximum

Every promoted generic bucket MUST have 100% required sealed coverage and a
maximum sealed worst-surface regret no greater than 3.0%. Mean or p95 regret
alone is insufficient.

### 5.7 Runtime-surface aliasing

Candidates MUST be evaluated on the dispatch surface the runtime can express.
If several source formats share one codebook/prepared-family key, a candidate
must pass correctness for every alias and performance selection must use the
worst alias-relative result.

### 5.8 No quiet fallback

If no grouped candidate satisfies the verifier contract, the capability is
unsupported and MUST fail closed before publication. The runtime MUST NOT fall
through to `Fast`, silently replay rows, or claim generic performance coverage.

### 5.9 Measured route equals emitted route

Every candidate selected by a generated table MUST be forceable in the trainer
and reachable in the production launcher. Generator acceptance MUST fail if a
candidate cannot round-trip through the backend emitter and resolver.

### 5.10 Reproducibility

A generated artifact MUST identify its corpus, candidate registry, learner,
feature schema, serial-M1 oracle policies, development folds, and sealed split
by stable digest.
A stale or incomplete certificate MUST fail generation or runtime validation.

## 6. Shared Architecture

### 6.1 Components

The target implementation has six layers:

1. **Backend candidate registry**: declares forceable candidates, static
   reachability, normalization, arithmetic fingerprinting, and C++ encoding.
2. **Backend trainer adapter**: runs the production prepared-weight path and
   emits common observations for every candidate.
3. **Common corpus validator**: verifies schema, completeness, aliases,
   numerical evidence, timing quality, and oracle freshness.
4. **Common policy compiler**: constructs alias-robust runtime exact winners, grouped development
   CV, frozen generic aspect/work rules, and sealed certification.
5. **Backend emitter**: maps common IR candidate IDs to backend-specific `.inc`
   declarations without changing policy decisions.
6. **Acceptance driver**: compiles staged artifacts, compares Python/C++
   resolution, runs parity/performance gates, and installs atomically.

The recommended common Python package is:

```text
tests/v2/performance/kernels/native_vnni_dispatch/
    schema.py
    candidate_observation.py
    corpus.py
    exact_oracle.py
    segmented_policy.py
    splits.py
    certification.py
    policy_ir.py
    validation.py
    adapters/
        cuda.py
        rocm.py
        cpu.py
```

Existing trainers MAY remain in their backend directories, but their CSVs
must conform to the common schema and their analyzers must delegate policy
fitting and evaluation to this package.

### 6.2 Dispatch key

The logical dispatch key is:

```text
backend
architecture_class
semantic_contract
operation_kind
bundle_signature and ordered projection-N vector
prepared-family/runtime-codebook vector
M, K
```

`source_format` is retained in the corpus for alias correctness and
performance aggregation, but it is not a runtime discriminator unless the
backend actually exposes it in its production dispatch API.

`architecture_class` MUST prevent measurements from incompatible devices from
being pooled. Initial classes SHOULD include:

- CUDA compute capability plus any kernel ABI-relevant device class;
- ROCm GFX target plus wave/kernel ABI-relevant class; and
- CPU ISA tier plus a declared microarchitecture tuning class.

For CPU, thread count, affinity/binding policy, and NUMA placement are part of
the architecture/topology key unless evidence explicitly certifies that one
policy is portable across those values. A table's 3% claim applies only to the
architecture/topology key recorded in its manifest.

Unknown architectures use only an explicitly certified portable floor or are
unsupported; they do not consume a table trained for another class by default.

### 6.3 Explicit runtime contract

The target C++ API carries semantics in the dispatch request, not in ambient
backend state:

```cpp
enum class GemmSemanticContract : uint8_t
{
    Fast,
    VerifierSerialM1Bitwise,
};

struct GemmDispatchRequest
{
    GemmSemanticContract contract;
    GemmOperationKind operation;
    int m;
    int k;
    std::span<const GemmProjectionDescriptor> projections;
};

struct GemmProjectionDescriptor
{
    GemmProjectionRole role;
    int n;
    PreparedFamilyId prepared_family;
    NativeVNNICodebookId codebook;
    PackingABI packing;
    GemmEpilogueSignature epilogue;
};

struct ResolvedGemmDispatch
{
    CandidateId candidate;
    EffectiveCandidateId effective_candidate;
    DispatchMatchKind match; // Exact, Generic, CertifiedFloor
    PolicyVersion policy;
    ArithmeticFingerprint arithmetic;
};
```

`bundle_signature` is derived from the ordered projection roles, prepared
families, codebooks, packing ABIs, and epilogues. The exact key also carries
the ordered projection-N vector.

Version 1 generic bundle learning is restricted to homogeneous bundles: every
projection shares K, source format, source codebook, prepared family, runtime
codebook, packing ABI, and a candidate support domain. Its generic features
are:

```text
aggregate_n = sum(N_i)
aspect_ratio = aggregate_n / K
work_items = aggregate_n * K
```

The candidate must be eligible for every projection and for the ordered fused
epilogue as a whole. Alias coverage substitutes the same tested alias across
the complete homogeneous bundle. A bundle mixing source aliases or any other
listed identity is heterogeneous in v1 and must be decomposed into supported
homogeneous sub-bundles or assigned a separately versioned operation policy;
it MUST NOT consume a single-projection or homogeneous generic certificate.

`ITensorGemm::beginVerifierDecodeEquivalentScope()` is the current common
bridge and MAY remain while callers migrate. The final implementation SHOULD
pass `GemmSemanticContract` through the call/plan object so concurrent host
threads and streams cannot observe process-global mode changes.

### 6.4 Canonical format and prepared-family registry

One versioned registry MUST define:

```text
source format and source codebook
backend execution codebook
prepared family and packing ABI
source-format aliases
block size and divisibility constraints
supported semantic contracts and operation kinds
```

The trainer inventory, all-format verifier inventory, backend traits, and
generated-policy validator MUST be derived from or checked against this
registry. A source alias may share a runtime entry only after every alias
passes byte eligibility for the same prepared-family packing ABI.

The audit found a concrete inventory divergence that this registry must close:
`tests/v2/utils/QuantizedVerifierFormats.h` advertises source `Q8_K`/codebook
21 and maps GPU execution to codebook 19, while
`tests/v2/performance/kernels/native_vnni_codebooks.py` does not currently list
`Q8_K`. Inventory-consistency tests MUST fail on this class of omission.

## 7. Common Measurement Corpus

### 7.1 Observation granularity

The corpus MUST retain every measured candidate, not just winners. One
aggregate observation represents one:

```text
(run, backend, architecture, contract, operation, bundle signature,
 ordered projection-N vector, execution mode, source format,
 prepared family, M, K, candidate, payload trial set)
```

Raw timing samples MUST be retained as an auditable sidecar artifact for the
lifetime of every accepted bundle. The aggregate row contains robust timing
statistics and a digest of the raw samples. Whether those samples live in the
repository or durable CI/artifact storage remains an implementation choice.

### 7.2 Required schema

| Group | Required fields |
|---|---|
| Schema/provenance | `schema_version`, `run_id`, `corpus_id`, `git_revision`, `build_id`, `compiler_id`, `policy_abi`, `learner_version` |
| Hardware | `backend`, `architecture_class`, `device_name`, `driver_runtime`, `threading_or_stream_mode` |
| Runtime key | `semantic_contract`, `operation_kind`, `bundle_signature`, `projection_n_vector`, `source_format`, `source_codebook_id`, `prepared_family_id`, `packing_abi`, `runtime_codebook_id`, `shape_group_id`, `shape_name`, `execution_mode`, `m`, `aggregate_n`, `k` |
| Features | `aspect_ratio`, `aspect_bucket`, `work_items`, `n_tail_class`, `k_tail_class`, `alignment_class` |
| Candidate | `candidate_id`, `effective_candidate_id`, `candidate_family`, `config_json`, `supported`, `graph_capture_ok`, `generic_eligible` |
| Arithmetic | `arithmetic_fingerprint`, `serial_m1_policy_id`, `serial_m1_policy_hash`, `candidate_policy_hash`, `ordered_reduction`, `uses_atomic_reduction` |
| Correctness | `trial_set_hash`, `bitwise_equal`, `repeat_equal`, `mismatch_count`, `first_mismatch_index`, `grouped_output_digest`, `serial_output_digest`, `max_abs`, `relative_l2`, `cosine`, `symmetric_kld` |
| Timing | `warmup_count`, `sample_count`, `min_us`, `median_us`, `p95_us`, `mad_us`, `cv`, `timing_sample_hash`, `effective_bandwidth_gbs` |
| Route proof | `forced_route_ok`, `observed_candidate_id`, `route_counter_ok`, `workspace_ok`, `explicit_stream_ok` |

Missing required fields are errors. Empty correctness values are not interpreted
as passing values.

### 7.3 Candidate enumeration rules

- Every measured candidate MUST have a stable explicit ID.
- `Auto`, a previous generated policy, or a heuristic fallback MUST NOT be
  recorded as though it were a concrete candidate.
- Each registry entry MUST declare an arithmetic signature, schedule
  signature, support predicate, required prepared resources, workspace
  requirements, graph-capture status, and runtime launch mapping.
- The trainer MUST force the candidate and confirm the observed route ID.
- Nominal candidates that normalize to one effective launch MUST be collapsed
  before learning.
- A candidate unsupported for a shape remains an explicit unsupported matrix
  cell; it is not silently omitted.
- A candidate missing for a required alias is unavailable for that runtime key.

### 7.4 Shape inventory

The production corpus MUST include:

- every current exact production shape;
- every advertised prepared family and source-format alias;
- `Fast M=1,2,3,4`, plus verifier `M=2,3,4` against the frozen serial-M1
  policy, together for each logical shape;
- nearby unseen dimensions on both sides of aspect boundaries `0.75`, `2.0`,
  and `16.0`;
- log-spaced work sizes within each aspect bucket;
- N and K tile-tail/alignment boundaries;
- fused and unfused operation kinds that have different launcher
  reachability; and
- graph-captured cells for every captured production route.

The inventory SHOULD live in one backend-neutral versioned manifest. Backend
adapters MAY mark a shape/candidate unsupported but MUST NOT silently remove a
required family or M value.

### 7.5 Eager and captured execution modes

`execution_mode` is a required corpus axis but is not a dispatch discriminator
in policy ABI v1. A candidate selected for an operation that supports both
eager and graph-captured production execution MUST pass correctness,
reachability, workspace, and repeatability in both modes. Exact and generic
selection use the worst normalized regret across every required execution
mode. If an operation has only one production mode, only that mode is required.

A future policy MAY select different candidates by execution mode, but that
requires a coordinated runtime-key, feature-schema, IR, and policy-ABI change
on all three backends.

### 7.6 Measurement protocol

Production acceptance measurements MUST:

- use Release builds and the production prepared-weight path;
- prepare/upload/repack weights outside the timed candidate loop;
- use an explicit non-default CUDA/HIP stream and record timing events on that
  stream;
- use the production CPU affinity/OpenMP configuration;
- perform at least 5 warmups and 30 timed samples per candidate by default;
- run a broad first-pass candidate sweep in deterministic randomized order;
- use median latency for fitting;
- retain variance and raw samples; and
- run correctness on multiple deterministic payload seeds, including
  structural edge payloads and the production packed representation.

Quick and family-smoke profiles MAY use fewer timings for workflow validation,
but they cannot produce installable artifacts.

After the broad sweep identifies the provisional alias-robust exact candidate
and the fixed generic candidate for each sealed certification cell, a second
paired confirmation pass interleaves only those candidates. Confirmation
sampling continues until the one-sided simultaneous 95% confidence bound for
the maximum regret across all required cells has no more than 0.5
percentage-point uncertainty. The implementation may use a bootstrap maximum
statistic, familywise correction, or another recorded method with equivalent
simultaneous coverage. Independent per-cell 95% intervals are insufficient.

A cell that cannot reach the required precision is inconclusive and blocks
promotion; it is not rounded down to a pass.

## 8. Candidate Eligibility and Oracle Versioning

### 8.1 Fast eligibility

A `Fast` candidate is eligible only if it:

- is statically supported and production-reachable;
- passes the backend's normal numerical correctness gate;
- observes the forced route;
- satisfies stream, workspace, and graph-capture contracts; and
- is deterministic to the degree required by the ordinary operation.

Fast correctness is separate from verifier correctness.

### 8.2 Verifier eligibility

A `VerifierSerialM1Bitwise` candidate is eligible only if all of the following
are true:

```text
supported
&& forced_route_ok
&& route_counter_ok
&& bitwise_equal
&& repeat_equal
&& serial_m1_policy_hash == current_serial_m1_policy_hash
&& no_uncertified_atomic_reduction
&& stream/workspace/capture contracts satisfied
```

Eligibility is evaluated for every source-format alias. One failing alias
rejects the candidate for the shared runtime key.

### 8.3 Exact-only versus generic-eligible

Empirical equality at one exact shape is sufficient only for an exact entry.
A candidate may enter a generic rule only when:

1. its arithmetic fingerprint is structurally compatible with the serial M1
   policy for the declared shape domain; and
2. it passes byte equality on all development and sealed tail/alignment classes
   in that domain.

This prevents isolated lucky equality from being extrapolated across unseen K
partitions or reduction trees.

### 8.4 Serial M1 policy identity

The serial M1 policy ID MUST resolve from the real production launcher. Its
hash includes at least:

- policy ABI;
- backend and architecture class;
- operation and prepared family;
- effective candidate ID;
- arithmetic fingerprint;
- kernel-math ABI; and
- shape-dependent K partition/reduction parameters.

Changing the M1 resolver, kernel math, quantization order, reduction order,
epilogue, or relevant compiler specialization invalidates dependent grouped
certificates. The generator MUST reject a base artifact with a different M1
hash rather than retaining its verifier entries.

M1 selection and verifier certification form an ordered two-stage
transaction:

1. train the `Fast M=1` policy;
2. generate, build, and stage that production M1 artifact;
3. resolve and freeze its policy/arithmetic hashes;
4. collect every `M=2..4` verifier observation against that exact staged M1
   artifact; and
5. compile verifier exact and generic policies without changing M1.

Any subsequent change to the staged M1 decision, code, packing ABI, or hash
restarts verifier collection and certification from step 3. A transaction MUST
NOT retune M1 after collecting dependent verifier rows.

## 9. Alias-Robust Runtime Exact Winner Construction

Exact winners are constructed after correctness filtering and before generic
learning.

For shape `s`, alias `f`, required execution mode `x`, candidate `c`, and
contract `q`, let:

```text
L(c,s,f,x,q) = median measured latency
Lbest(s,f,x,q) = min L(e,s,f,x,q) over eligible candidates e
surface_regret(c,s,q) = max_(f,x) (L(c,s,f,x,q) / Lbest(s,f,x,q) - 1)
```

Because policy ABI v1 cannot distinguish aliases or required execution modes,
the alias-robust runtime exact winner is selected lexicographically by:

1. minimum `surface_regret`;
2. minimum p95 alias/mode regret;
3. minimum mean normalized latency;
4. lower measured variance;
5. stable candidate ID as deterministic tie-breaker.

Candidates whose measured difference is smaller than the declared timing
uncertainty form a co-winner set. Tie-breaking within that set uses stability,
lower policy complexity, and stable candidate ID rather than claiming a noisy
ordering as fact.

For a non-aliased, single-mode key this reduces to the fastest stable eligible
candidate.

Exact overlays MUST be emitted for required production keys. Training-only
synthetic shapes MAY remain generic-only unless they expose a demonstrated
generic failure that the runtime domain must support.

No exact entry may reference a candidate missing from the compiled backend
registry.

## 10. Generic Aspect/Work Learner

### 10.1 Common v1 feature policy

Version 1 uses the existing common aspect definition:

```text
aspect_ratio = aggregate_N / K

very_wide: aspect_ratio >= 16.0
wide:       aspect_ratio >= 2.0
balanced:   aspect_ratio >= 0.75
tall:       otherwise

work_items = aggregate_N * K
```

Rules are learned independently for:

```text
(backend, architecture_class, semantic_contract, operation_kind,
 bundle_signature, prepared_family/runtime_codebook, M, aspect_bucket)
```

Each bucket may contain at most three contiguous work-size segments in v1.
Candidate support constraints must cover every tail/alignment class assigned
to the segment. Learning aspect boundaries themselves MAY be introduced in a
future feature-schema version, but all backends must move together.

A generic leaf MUST contain at least two distinct development-training shape
groups and must be exercised by at least one sealed certification shape group;
the production corpus SHOULD target four or more distinct development shapes
per leaf. A rule that merely encodes one measured shape is an exact entry
disguised as a generic policy and is not promotable.

### 10.2 Required cost matrix

The learner consumes the complete eligible candidate latency matrix. It MUST
NOT consume only exact-winner labels. A missing candidate measurement makes
that candidate invalid for any segment containing the missing shape.

### 10.3 Segment objective

For each proposed segment and candidate, compute regret against the
alias-robust runtime exact winner for each development-training shape, alias,
and required execution mode. Choose cuts and candidates using this
deterministic lexicographic objective. Sealed certification rows MUST NOT
participate in any choice:

1. lowest maximum development-training regret;
2. fewest development-training points above 3.0%;
3. lowest p95 regret;
4. lowest mean regret;
5. fewest segments;
6. stable candidate and cut ordering.

Modal exact-winner count, candidate-label accuracy, and family hit rate MAY be
reported as diagnostics but MUST NOT drive promotion.

An exhaustive search or dynamic program over at most three contiguous
segments is preferred to an unrestricted decision tree. The small bounded
policy is deterministic, reviewable, and maps identically to all backends.

### 10.4 Failure to meet 3%

If a bucket cannot meet the generic gate:

- required production-domain coverage blocks promotion;
- optional coverage is marked `performance_unpromoted` in the manifest;
- known production shapes may still use certified exact overlays; and
- unknown shapes may use only a certified grouped floor, otherwise the lane is
  unsupported.

The learner MUST NOT relax numerical eligibility to recover performance.

## 11. Shape-Grouped Development and Sealed Holdout Protocol

### 11.1 Split unit

The split unit is `shape_group_id`, representing one logical model role and
`(N,K)` neighborhood. All rows for that group remain together:

- every candidate;
- every source-format alias;
- every M value governed by that certification surface;
- eager and captured variants; and
- repeated timing/correctness trials.

Random candidate-row splitting is forbidden because it leaks the held-out
shape into training.

### 11.2 Development and sealed certification partitions

Before feature, cut, candidate, or complexity choices are made, logical shape
groups MUST be assigned to:

1. a development partition used for fitting and grouped cross-validation; and
2. a sealed certification partition used exactly once to certify the final
   emitted generic policy.

The assignment is deterministic from a versioned split manifest and seed.
Every candidate, alias, M value, execution mode, and trial for a shape follows
that shape into the same partition. The sealed measurements MUST remain
unavailable to learner selection code until the final generic policy IR and
digest are frozen. Sealed feature rows likewise remain hidden from selection
code; only the predeclared coverage obligations are visible before the policy
is frozen.

The split manifest defines separate certification surfaces for frozen
`Fast M=1`, remaining `Fast M=2..4`, and `VerifierSerialM1Bitwise M=2..4`.
Their sealed shape groups SHOULD be disjoint; the verifier sealed groups MUST
not have had verifier candidate performance exposed while M1 was selected.

The sealed partition MUST exercise every promoted generic leaf, aspect/work
boundary neighborhood, required tail/divisibility class, and relevant model
family. A domain with insufficient independent development and sealed shape
groups cannot claim generic certification and remains exact-only.

Once a sealed shape has been inspected for a certification decision, it cannot
later be added to development under the same certificate. Doing so requires a
new untouched sealed partition and a new manifest digest.

### 11.3 Development cross-validation and final fit

Development folds MUST be deterministic from the corpus and split-manifest
version.

- Use grouped 5-fold cross-validation when a domain has enough development
  shape groups.
- Use leave-one-shape-group-out validation for sparse development domains.
- Stratification SHOULD preserve prepared family, M coverage, aspect bucket,
  work scale, bundle signature, and model-family diversity without breaking
  group integrity.
- Where several shapes come from one model family, at least one development
  evaluation SHOULD leave the whole model family out.

Development cross-validation selects the feature-schema version, permitted
rule complexity, work cuts, and candidates. After those choices are frozen,
the learner fits the final generic policy exactly once on all development
groups. The resulting generic IR and digest are frozen before sealed data is
opened.

### 11.4 Sealed generic-only evaluation

The exact generic IR that will be emitted is evaluated on the sealed
certification partition:

1. disable every exact overlay;
2. resolve the frozen generic candidate for each sealed runtime key;
3. find that candidate's sealed measured latency in every required execution
   mode;
4. construct the sealed alias-robust runtime exact winner under the same
   semantic contract;
5. run the paired confirmation timing protocol; and
6. aggregate the worst source-format alias and required execution mode.

A candidate not measured on a sealed point is uncovered, not zero-regret. For
generic candidate `g` and runtime-representable sealed exact winner `cexact`:

```text
generic_regret(g,s,q) =
    max_(f,x) (L(g,s,f,x,q) / L(cexact,s,f,x,q) - 1)
```

The denominator is the exact candidate the runtime can actually dispatch for
the shared alias/mode key, not a different unattainable per-format candidate.

No learner parameter, segment cut, or candidate may change after sealed
evaluation. A failure requires a new policy attempt and a new untouched sealed
partition; the failed sealed shapes become development evidence only in that
later attempt.

The sealed certificate MUST record the pre-open
`frozen_generic_policy_digest` and the validator MUST prove it matches the
generic section of the emitted IR byte-for-byte.

Exact production overlays are constructed separately and MAY use all accepted
exact-shape measurements, including sealed shapes, because exact lookup is
bypassed during generic certification. They MUST NOT alter or retrain the
frozen generic IR.

### 11.5 Acceptance statistic

For every promoted generic bucket:

```text
sealed_coverage == 100%
sealed_max_observed_worst_surface_regret <= 0.03
sealed_simultaneous_95pct_upper_regret <= 0.03
bitwise_failures == 0 for VerifierSerialM1Bitwise
```

The summary MUST report development-CV metrics separately from sealed
certification metrics, including median, p95, maximum, simultaneous confidence
bound, shape count, alias count, execution-mode count, fold count, and every
near-threshold cell.

After sealed certification, the common IR is emitted without refitting. The
C++ resolver round-trip proves that the compiled selector emits the exact
frozen decisions; the sealed timing certificate proves their performance.

## 12. Generated Artifacts and Provenance

### 12.1 Common policy IR

The learner emits backend-neutral policy IR before C++ generation:

```cpp
struct ExactDispatchEntry
{
    SemanticContract contract;
    OperationKind operation;
    BundleSignature bundle;
    ProjectionNVector projection_n;
    PreparedFamilyId prepared_family;
    uint8_t codebook;
    uint8_t m;
    int aggregate_n;
    int k;
    CandidateId candidate;
    ArithmeticFingerprint arithmetic;
};

struct GenericDispatchRule
{
    SemanticContract contract;
    OperationKind operation;
    BundleSignature bundle;
    PreparedFamilyId prepared_family;
    uint8_t codebook;
    uint8_t m;
    AspectBucket aspect;
    int64_t max_work_items;
    CandidateId candidate;
    ArithmeticFingerprint arithmetic;
};
```

Backends MAY encode these as constexpr arrays, switches, or generated helper
functions, but emitter round-trip tests must resolve the same candidate as the
IR for every test key.

The common IR metadata MUST carry a per-contract
`frozen_generic_policy_digest`. The sealed certificate binds to that digest,
and the digest covers only generic decisions and their feature schema. It is
distinct from the final policy/bundle digest, which also covers sealed-derived
exact overlays and other artifact metadata.

### 12.2 Installable artifact set

Each backend policy installation consists of:

1. the generated `.inc` consumed by production;
2. a machine-readable `.policy.json` containing exact and generic IR;
3. an acceptance `.manifest.json`; and
4. a human-readable summary.

These files live inside a versioned immutable bundle. A bundle index records
all backend/architecture artifacts it contains and is the target of the single
atomic active-pointer switch defined in Section 15.

The manifest MUST contain:

```text
schema_version
policy_abi
learner_version
feature_schema_version
backend and architecture_class
git/build/compiler identifiers
candidate_registry_hash
format_registry_hash and packing_abi_hash
corpus_hash and trial-set hash
serial_m1_policy_hashes
per-contract frozen_generic_policy_digest
exact-entry count
generic-rule count
required-key coverage
development-CV and sealed-split manifest hashes
development-CV metrics
sealed median/p95/max/simultaneous-UCB regret
rejected, uncovered, and performance-unpromoted buckets
bitwise certificate counts
emitter round-trip result
staged test results
generation timestamp
```

The `.inc`, policy IR, and manifest MUST share one policy digest. Every
generated `.inc` MUST embed constexpr metadata for the policy ABI,
architecture class, policy digest, candidate-registry hash, format-registry
hash, packing-ABI hash, per-contract frozen generic policy digests, and
relevant serial-M1 oracle hashes. The compiled resolver MUST compare those
values with the compiled/runtime identities before using the table. A JSON
manifest alone is not runtime validation.

Production builds MUST expose the active digest in version output or debug
diagnostics.

### 12.3 Base artifacts

Partial refreshes merge measurement corpora and retrain the policy. They MUST
NOT layer new exact entries over an old opaque generic `.inc` and then report
metrics for a different newly inferred fallback.

If an unchanged base generic policy is intentionally retained, evaluation must
load and score the actual retained IR. A base include without matching IR,
manifest, and compatible oracle hashes is not mergeable.

## 13. Runtime Resolution, Failure Policy, and Telemetry

### 13.1 Lookup order

Runtime resolution is:

```text
1. validate policy ABI/backend/architecture/contract
2. exact overlay lookup
3. certified generic aspect/work lookup
4. certified floor for the declared domain
5. fail closed
```

Verifier resolution MUST never call the unrestricted fast resolver as step 4.

### 13.2 Launch validation

Before launching, the backend MUST validate:

- candidate support for the concrete operation, bundle signature, ordered
  projection-N vector, aggregate N, M, K, and execution mode;
- workspace availability;
- stream ownership;
- graph-capture compatibility;
- arithmetic fingerprint compatibility for verifier calls; and
- absence of an unapproved runtime override.

An unsupported emitted candidate is a generator/runtime contract failure, not
an invitation to choose another heuristic silently.

### 13.3 Concurrency

Semantic mode MUST be per request/call/plan. Process-global verifier flags are
not permitted in the target architecture. Nested and concurrent verifier/fast
calls on different host threads and streams MUST resolve independently.

### 13.4 Route telemetry

Every grouped verifier dispatch SHOULD record, through the route-record design
in [Perf Stats Collector](../2026-06/PERF_STATS_COLLECTOR.md):

```text
backend
semantic_contract
operation_kind
bundle_signature
projection_n_vector
aggregate_n
prepared_family/codebook
M,K
candidate_id and effective_candidate_id
match_kind = exact | generic | certified_floor
policy_digest
frozen_generic_policy_digest
serial_m1_policy_hash
execution_mode = graph | eager
```

Tests MUST assert route telemetry so byte-equal output from an accidental row
loop or different kernel does not masquerade as a grouped-policy proof.

## 14. Backend Migration Requirements

### 14.1 CUDA

CUDA migration MUST:

1. Replace the independent decision-tree and aspect/overlay fitting paths with
   the common learner.
2. Retain every candidate's timing row rather than relying only on `is_best`.
3. Emit explicit byte and serial-M1-policy evidence for verifier candidates.
4. Normalize candidate settings under the verifier contract before assigning
   `effective_candidate_id`; nominal target-wave, KB, or two-phase settings
   that launch identically must not become different labels through timing
   noise.
5. Prove that every emitted tile/CPT/family tuple is supported by the grouped
   production dispatcher. Current support includes the repaired `256x4` KPAR
   route; regression coverage must prevent future generated/runtime drift.
6. Encode optional ROWPAR preparation as an explicit candidate resource
   predicate. The current verifier path suppresses lazy ROWPAR promotion by
   passing a null row-major slot; that behavior must remain covered unless a
   future ROWPAR verifier candidate uses the same immutable prepared
   representation as M1 and passes the complete byte gate.
7. Preserve explicit non-default stream timing, prepared-weight reuse, and
   workspace sizing through `IWorkspaceConsumer`.

Primary migration surfaces:

- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp`
- `tests/v2/performance/kernels/cuda/gemm/infer_gemv_dispatch_heuristic.py`
- `tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc`

### 14.2 ROCm

ROCm migration MUST:

1. Preserve the full candidate-by-shape latency matrix instead of collapsing
   immediately to the fastest exact row.
2. Replace modal KB/target-wave selection and label-hit metrics with the common
   regret learner and sealed-certification gate.
3. Replace cosine/tolerance eligibility with byte equality against production
   serial M1 for verifier candidates.
4. Make the real serial M1 launcher and grouped resolver consume one canonical
   policy source; duplicated fallback heuristics are prohibited.
5. Replace the process-global verifier mode with the explicit semantic
   contract.
6. Reject atomic reduction and arithmetic-changing environment overrides for
   verifier calls unless the exact effective route is separately certified.
7. Include Q8 and IQ special routes in the same candidate registry and policy
   surface rather than retaining handwritten dispatch exceptions.

Primary migration surfaces:

- `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Throughput.cpp`
- `tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

### 14.3 CPU

CPU migration MUST:

1. Replace exact-only generation with the same exact-plus-generic common IR.
2. Benchmark explicit forceable candidates instead of inferring the executed
   policy from an `Auto` phase name.
3. Give Pairwise, Wide3, Wide4, K-parallel, fused, and any other selectable
   implementation stable candidate IDs and route counters.
4. Compare every verifier candidate with production serial M1 using bytes and
   emit the serial-M1 policy hash.
5. Route single projection, fused projection, fused gate/up/down, and routed or
   shared MoE paths through the same semantic resolver. A fused path may not
   bypass the learned policy.
6. Add generic aspect/work rules and grouped development/sealed data, including CPU
   topology and tail-sensitive shapes.
7. Preserve Pairwise only as a certified floor where it is genuinely grouped
   and performance/correctness evidence covers the domain.

Primary migration surfaces:

- `tests/v2/performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_GEMV.cpp`
- `tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc`
- CPU fused and MoE callers in `CPUNativeVNNIGemmKernel.h`

## 15. Transactional Retune and Installation

### 15.1 Profile classes

Profiles are classified as:

- `quick`: script/schema smoke only;
- `family-smoke`: bounded candidate/family workflow proof;
- `partial-production`: complete evidence for a declared subset, merged into a
  complete compatible corpus; or
- `production`: complete required inventory and installable evidence.

`quick` and `family-smoke` MUST reject `--install`.

### 15.2 Required transaction

An installable refresh performs these steps in order:

1. **Preflight**
   - identify backend/architecture targets;
   - freeze schema, learner, feature, policy ABI, and candidate registry;
   - validate required shape/format/M inventory;
   - freeze the development/sealed split manifest without exposing sealed
     measurements to the learner;
   - create an isolated staging directory.
2. **Collect and stage Fast M1**
   - run explicit `Fast M=1` candidates with generated policy disabled;
   - validate and certify its generic policy using its own development/sealed
     split;
   - compile, emit, and build the staged production M1 policy without a
     post-certification refit;
   - freeze its policy, packing, and arithmetic hashes.
3. **Collect dependent policies**
   - run explicit `Fast M=2..4` candidates;
   - run verifier `M=2..4` candidates against the exact frozen staged M1
     artifact;
   - collect common-schema observations and raw timing artifacts;
   - verify route, stream, workspace, and correctness fields.
   - allow architecture corpora to be collected on separate hosts while
     preserving one schema, manifest, and artifact transaction.
   - restart dependent collection if any M1 input or hash changes.
4. **Validate corpus**
   - reject missing required keys, aliases, candidates, M values, or trials;
   - reject stale M1 hashes and unstable measurements.
5. **Compile and freeze generic policy**
   - filter eligibility;
   - construct alias/mode-robust runtime exact winners;
   - use grouped development CV to select learner complexity;
   - fit the final generic policy on development shapes only;
   - freeze its IR/digest before opening sealed measurements;
   - evaluate that exact policy on the sealed partition with exact overlays
     disabled;
   - require the 3% maximum sealed-certification gate;
   - write final IR/manifest without refitting the generic policy.
6. **Emit immutable bundle**
   - generate every selected backend include from the accepted IR;
   - embed the required runtime metadata;
   - validate IDs, sorting, reachability, and deterministic output;
   - assemble a complete immutable bundle identified by policy digest.
7. **Stage build**
   - build Llaminar against the staged includes without modifying checked-in
     artifacts;
   - run Python-to-C++ resolver round-trip tests.
8. **Correctness acceptance**
   - enumerate exact entries and generic leaves/boundaries;
   - run all-format grouped byte parity and model operation equivalence.
9. **Performance acceptance**
   - validate the emitted generic policy against the sealed certificate;
   - run relevant full-model benchmarks outside profiling mode.
10. **Atomic publish**
    - under one install lock, rename the completed staged directory into a
      versioned immutable bundle location;
    - atomically rename one `active` symlink or pointer file from the old
      complete bundle to the new complete bundle;
    - record old/new digests in the transaction journal.

A failure before the single active-pointer rename leaves the active production
bundle unchanged. A crash after it exposes either the complete old bundle or
the complete new bundle, never a mixture. Rollback atomically repoints
`active` to the prior immutable bundle. The build include path must resolve
through this active bundle; sequential copying of several destination files is
not an atomic install and is forbidden.

`--skip-sweep` is acceptable only when the supplied corpus passes schema,
build, hardware, trial-set, and provenance digest checks. It is not a way to
reuse an unverifiable CSV.

### 15.3 Cross-backend atomicity

A change to the common schema, learner, feature policy, policy ABI, or IR
requires CPU, CUDA, and ROCm regeneration and acceptance in one transaction.

A hardware-specific corpus refresh MAY update one backend if the common
versions are unchanged and the other installed manifests remain compatible.
It still creates a complete immutable bundle containing the accepted unchanged
artifacts for the other backends and publishes that bundle through the one
active-pointer update.

### 15.4 Base merge policy

Staged profiles merge compatible corpora and retrain. Direct `.inc` copying or
`--base-include` overlay installation is forbidden unless the base has
machine-readable IR and the validator proves that the scored and emitted
generic policies are identical.

## 16. Required Tests

### 16.1 Common schema and learner units

The common learner requires synthetic regression tests proving that:

- a one-ULP verifier candidate is rejected even with perfect-looking cosine;
- missing `bitwise_equal`, oracle hash, or route evidence is rejection;
- an unsafe lowest-latency candidate is filtered before exact selection;
- a policy with 97% winner-label accuracy and one 40% performance miss fails;
- a policy with low winner-label accuracy but every regret below 3% passes;
- exact overlays cannot mask a broken generic policy during sealed evaluation;
- candidate/format/M/mode rows from one shape cannot leak across development
  folds or into the sealed partition;
- sealed observations are unavailable until the final generic digest is frozen;
- the emitted generic IR is not refit after sealed certification;
- opposing alias/mode winners choose the deterministic worst-surface robust
  candidate;
- missing alias coverage rejects a candidate;
- a stale M1 oracle hash invalidates verifier entries and requires dependent
  evidence to be recollected;
- a candidate unsupported by the production launcher fails generation;
- segment fitting and artifact digests are deterministic;
- base-policy metrics describe the exact emitted generic rules; and
- a domain with one shape cannot claim generic certification;
- `Auto` cannot be relabeled as an explicit CPU candidate;
- canonical-format inventory divergence, including a missing `Q8_K` entry, is
  rejected;
- quick/family-smoke installation is rejected; and
- a failed bundle transaction leaves the active pointer unchanged.

### 16.2 Backend emitter and resolver units

Each backend MUST test:

- exact precedence over generic rules;
- every aspect boundary and work-segment boundary;
- Python IR to generated C++ round-trip candidate equality;
- every emitted candidate's launcher reachability;
- homogeneous bundle-signature and aggregate-feature resolution;
- rejection of an uncertified heterogeneous bundle;
- worst-mode aggregation across eager and captured observations;
- separation of `Fast` and verifier policies with a deliberately poisoned fast
  table;
- nested and concurrent contract resolution;
- embedded policy/architecture/registry/oracle metadata mismatch failure;
- telemetry match kind and candidate identity; and
- deterministic fail-closed behavior for missing coverage.

Existing generated-dispatch units remain necessary but are not sufficient:

```bash
ctest --test-dir build_v2_integration \
  -R 'V2_Unit_(NativeVNNIDispatchRefreshScript|CUDAGemvDispatchGeneratorAliases|CUDAGemvDispatchBaseMerge|ROCmNativeVNNIDecodeTrainerGenerator|ROCmNativeVNNITrainerCsvValidator|NativeVNNIGeneratedDispatchCodebooks)' \
  --output-on-failure --parallel
```

These tests remain during migration. Once opaque `--base-include` merging is
removed, `V2_Unit_CUDAGemvDispatchBaseMerge` SHOULD be retired and replaced by
common corpus-merge and emitted-policy identity tests.

### 16.3 Kernel byte-equivalence matrix

Generated-table acceptance MUST enumerate:

- every exact verifier entry;
- at least one interior point in every generic leaf;
- both sides of every aspect and work boundary;
- every prepared family/codebook and source-format alias;
- `M=2,3,4`;
- eager and graph-captured production routes;
- cold first use, warmed execution, and repeated execution;
- fused and unfused paths;
- alpha/beta, existing-output, and bias/epilogue variants where supported;
- dense and routed/shared MoE operation kinds; and
- every emit-able reduction/K-partition family.

For each cell, force the resolved candidate, run grouped rows once, run the
production M1 oracle for each row, compare every logical FP32 result element,
and assert the grouped route counter. The public-operation layer must
additionally start from the same input tensor so activation quantization and
preparation are inside the proof boundary.

The canonical cross-backend prefix remains mandatory:

```bash
ctest --test-dir build_v2_integration -N \
  -R '^V2_Integration_GroupedVerifierRows_'
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_GroupedVerifierRows_' \
  --output-on-failure --parallel
```

The prefix is authoritative; a hand-maintained test list is not.

### 16.4 Model-level gates

Kernel equality is necessary but not sufficient. A promoted table MUST also
pass the affected Qwen dense and MoE verifier-operation equivalence,
continuation, graph-capture, prefix-cache/MTP, and stochastic tests. These
model-level gates keep cosine/L2/KLD and token checks as layered diagnostics,
but they do not replace kernel `memcmp`.

The current six backend/model operation-equivalence lanes are mandatory:

```text
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_Dense_CPU
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_Dense_CUDA
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_Dense_ROCm
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_MoE_CPU
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_MoE_CUDA
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_MoE_ROCm
```

### 16.5 Performance gate

The install test MUST parse the staged manifest and fail unless every required
generic bucket reports:

- complete sealed shape/alias/execution-mode coverage;
- zero missing candidate measurements;
- maximum worst-surface regret at or below 3.0%;
- simultaneous 95% upper bound on maximum regret at or below 3.0%;
- a generic policy digest frozen before sealed evaluation and unchanged in the
  emitted IR; and
- zero verifier byte failures.

Full-model throughput must then remain within its declared regression budget;
an isolated generic-policy pass does not authorize a model-level regression.

## 17. Promotion Criteria

A backend/architecture/contract domain is promotable only when all rows below
are green.

| Gate | Required evidence |
|---|---|
| Corpus completeness | Every required operation, bundle, family, alias, M, exact shape, development/sealed group, execution mode, and explicit candidate is represented. |
| Oracle stability | Production M1 route is repeat-byte-stable and its current policy hash is recorded. |
| Candidate correctness | Every selected verifier exact/generic/floor candidate is byte-equal for all required trials. |
| Exact policy | All required production keys resolve to eligible reachable candidates. |
| Generic policy | The exact frozen generic IR has 100% sealed coverage and max worst-surface regret/simultaneous UCB `<=3.0%`, with no post-certification refit. |
| Emitter/runtime | Common IR and C++ resolver agree; exact precedence, boundaries, digests, and telemetry pass. |
| Grouped implementation | Route counters prove one genuine grouped implementation, not hidden row replay. |
| Integration | Canonical grouped prefix and affected Qwen operation/continuation gates pass. |
| Performance | Relevant kernel and model benchmarks are non-regressing beyond the accepted budget. |
| Installation | One immutable complete bundle is published by an atomic active-pointer rename and can roll back by digest. |

No aggregate green percentage can override a red required cell.

## 18. Phased Implementation Plan

### Phase 0: Freeze unsafe promotion paths

- Reject `--install` for quick and family-smoke profiles.
- Prevent partial copy-on-success before all selected backends validate.
- Require staged outputs and a transaction directory.
- Add source-policy tests preventing new hand-coded exact-shape exceptions.

Exit: current tables remain usable, but no retune can bypass the future
certificate boundary.

### Phase 1: Common schema, inventory, and candidate registries

- Add the shared observation schema and validator.
- Add the backend-neutral shape, development-fold, and sealed-split manifests.
- Create explicit CUDA, ROCm, and CPU candidate registries.
- Add effective-candidate normalization and launcher reachability checks.
- Add serial-M1 policy and arithmetic fingerprints.

Exit: all three trainers can emit complete, validated common rows without
changing production dispatch.

### Phase 2: Common exact and generic compiler

- Implement eligibility filtering and alias aggregation.
- Implement exact robust-winner construction.
- Implement bounded aspect/work segmentation over the full cost matrix.
- Implement deterministic grouped development CV, a frozen final development
  fit, and sealed generic-only certification with no later refit.
- Implement policy IR, manifests, and synthetic learner tests.

Exit: one compiler can consume fixture corpora for every backend and prove or
reject the 3% sealed-certification claim.

### Phase 3: Backend trainer and emitter migration

- Migrate CUDA away from the two mismatched generic learners.
- Migrate ROCm away from modal winner rows and approximate verifier gates.
- Add explicit CPU candidates and first-class generic output.
- Add emitter/resolver round-trip tests for all three.

Exit: all backend `.inc` files are decisions from the same common compiler.

### Phase 4: Explicit runtime semantic contract

- Add `GemmSemanticContract` to the common dispatch request/plan.
- Route every single, fused, dense, and MoE verifier caller through it.
- Remove process-global backend mode state and arithmetic-changing verifier
  overrides.
- Add policy digest validation and route telemetry.

Exit: ordinary fast tables cannot affect grouped verifier execution, including
under concurrent calls.

### Phase 5: Staged acceptance and atomic publish

- Extend the refresh wrapper to produce versioned immutable
  include/IR/manifest bundles and one atomic active pointer.
- Build against staged artifacts.
- Enumerate exact/generic runtime decisions and run the canonical grouped
  matrix.
- Require model parity, sealed performance certification, and atomic bundle
  publication.

Exit: an installed table is reproducible, byte-certified, sealed-certified,
and rollbackable.

### Phase 6: Production retune and default policy

- Collect complete blessed hardware corpora for CPU, CUDA, and ROCm.
- Train, stage, and freeze `Fast M=1` first; then recollect and certify
  verifier `M=2..4` against that exact staged artifact while training the
  remaining `Fast M=2..4` policies.
- Run the full MTP benchmark matrix.
- Update the parent plan and evidence dashboard with artifact digests,
  correctness counts, development-CV and sealed regret, and model throughput.

Exit: all three backends have first-class exact and generic learned tables
under this policy.

## 19. Failure Handling and Rollback

- Corpus or schema failure: do not fit or emit.
- No eligible verifier candidate: mark the domain unsupported; do not fall
  back to fast or row replay.
- Sealed certification above 3%: do not promote that required domain. A later
  policy attempt may use those revealed shapes as development evidence only
  after reserving a new untouched sealed partition.
- Stale M1 hash: invalidate and recollect, then retrain, every dependent
  verifier entry.
- Emitter/runtime mismatch: reject the artifact as corrupt.
- Integration or model parity failure: retain the prior installed transaction.
- Model performance regression: retain the prior table even if microbench and
  sealed gates pass.
- Post-install regression: atomically repoint `active` to the previous
  immutable bundle digest; never restore only one file from the set.

## 20. Open Implementation Decisions

These choices must be resolved before Phase 1 exits and recorded in the policy
ABI or manifest as appropriate:

1. The initial blessed `architecture_class` inventory for CPU model families,
   CUDA compute capabilities, and ROCm GFX targets.
2. The repository versus CI-artifact retention policy for raw timing samples
   and full corpora.
3. The exact simultaneous/familywise bootstrap or correction method used for
   the 95% maximum-regret upper bound.
4. The initial kernel-math ABI fields and which compiler changes invalidate an
   arithmetic certificate.
5. Whether operation kinds with identical candidate reachability may share a
   policy bucket, with proof, or remain separate in v1.
6. The required model-level throughput regression budget after the kernel
   sealed certification gate passes.

None of these decisions may weaken verifier byte equality or the maximum 3%
sealed-certification regret requirement.

## Appendix A: Example Observation

```json
{
  "schema_version": 1,
  "run_id": "2026-07-11-rocm-gfx906-production-01",
  "corpus_id": "sha256:corpus...",
  "git_revision": "0123456789abcdef",
  "build_id": "rocm-release-01234567",
  "compiler_id": "clang-19.0.0-rocm",
  "policy_abi": 1,
  "learner_version": "native-vnni-segmented-regret-v1",
  "backend": "rocm",
  "architecture_class": "gfx906-native-vnni-v1",
  "device_name": "AMD Instinct gfx906",
  "driver_runtime": "rocm-example",
  "threading_or_stream_mode": "explicit_non_default_stream",
  "semantic_contract": "VerifierSerialM1Bitwise",
  "operation_kind": "FusedGateUp",
  "bundle_signature": "gate_up:homo:cb5:fp32-epilogue:v1",
  "projection_n_vector": [17408, 17408],
  "source_format": "Q4_K",
  "source_codebook_id": 5,
  "prepared_family_id": "NativeVNNI_CB5",
  "packing_abi": "native-vnni-cb5-v1",
  "runtime_codebook_id": 5,
  "shape_group_id": "qwen36-ffn-gate-up-neighborhood",
  "shape_name": "Qwen36_FFN_GateUp",
  "execution_mode": "graph_captured",
  "m": 3,
  "aggregate_n": 34816,
  "k": 5120,
  "aspect_ratio": 6.8,
  "aspect_bucket": "wide",
  "work_items": 178257920,
  "n_tail_class": "aggregate_n_mod_256=0",
  "k_tail_class": "k_mod_256=0",
  "alignment_class": "prepared_16b_aligned",
  "candidate_id": "rocm.kb4.tw8.ordered",
  "effective_candidate_id": "rocm.kb4.tw8.ordered",
  "candidate_family": "shared_rows",
  "config_json": {"kb": 4, "target_waves": 8, "reduction": "ordered"},
  "supported": true,
  "graph_capture_ok": true,
  "generic_eligible": true,
  "arithmetic_fingerprint": "sha256:...",
  "serial_m1_policy_id": "rocm.cb5.m1.kb4.tw8.ordered",
  "serial_m1_policy_hash": "sha256:...",
  "candidate_policy_hash": "sha256:...",
  "ordered_reduction": true,
  "uses_atomic_reduction": false,
  "trial_set_hash": "sha256:trials...",
  "bitwise_equal": true,
  "repeat_equal": true,
  "mismatch_count": 0,
  "first_mismatch_index": null,
  "grouped_output_digest": "sha256:...",
  "serial_output_digest": "sha256:...",
  "max_abs": 0.0,
  "relative_l2": 0.0,
  "cosine": 1.0,
  "symmetric_kld": 0.0,
  "warmup_count": 5,
  "sample_count": 30,
  "min_us": 40.98,
  "median_us": 41.25,
  "p95_us": 41.91,
  "mad_us": 0.19,
  "cv": 0.006,
  "timing_sample_hash": "sha256:samples...",
  "effective_bandwidth_gbs": 216.1,
  "forced_route_ok": true,
  "observed_candidate_id": "rocm.kb4.tw8.ordered",
  "route_counter_ok": true,
  "workspace_ok": true,
  "explicit_stream_ok": true
}
```

## Appendix B: Learner Pseudocode

```python
def compile_policy(development_observations, sealed_partition_handle,
                   sealed_commitment, required_inventory,
                   frozen_m1_manifest, split_manifest):
    development = validate_common_schema(development_observations)
    development = normalize_effective_candidates(development)
    assert_development_surface(
        development, required_inventory, frozen_m1_manifest, split_manifest
    )
    assert_opaque_sealed_commitment(sealed_commitment, split_manifest)

    development = filter_by_semantic_contract(development)
    development_exact = build_alias_mode_robust_exact_winners(development)

    # CV chooses complexity using development shapes only.
    cv_results = grouped_development_cv(development, split_manifest)
    learner_spec = select_learner_spec(cv_results)

    # This is the final generic policy. Freeze it before opening sealed data.
    final_generic = fit_bounded_aspect_work_rules(
        development, development_exact, learner_spec
    )
    final_generic_digest = freeze_policy_digest(final_generic)

    # No sealed row or feature has been loaded before this point.
    sealed_observations = open_sealed_partition(
        sealed_partition_handle,
        sealed_commitment,
        final_generic_digest,
    )
    sealed = validate_common_schema(sealed_observations)
    sealed = normalize_effective_candidates(sealed)
    assert_disjoint_shape_groups(development, sealed, split_manifest)
    assert_sealed_surface(
        sealed, required_inventory, frozen_m1_manifest, sealed_commitment
    )
    sealed_exact = build_alias_mode_robust_exact_winners(sealed)
    sealed_results = evaluate_generic_only(
        final_generic, sealed, sealed_exact
    )
    sealed_results = paired_confirmation_timing(
        final_generic, sealed_exact, sealed_results
    )

    require_complete_coverage(sealed_results)
    require_max_worst_surface_regret(sealed_results, limit=0.03)
    require_simultaneous_max_regret_ucb(sealed_results, limit=0.03)

    # Exact overlays are separate and do not retrain final_generic.
    all_eligible = development.concat(sealed)
    all_exact = build_alias_mode_robust_exact_winners(all_eligible)
    policy_ir = emit_common_ir(
        exact_overlays=all_exact.required_production_entries(),
        generic_rules=final_generic,
        development_cv=cv_results,
        sealed_certificate=sealed_results,
        frozen_generic_policy_digest=final_generic_digest,
        frozen_m1_manifest=frozen_m1_manifest,
    )
    return policy_ir
```

## Appendix C: Example Acceptance Manifest Fragment

```json
{
  "policy_abi": 1,
  "learner_version": "native-vnni-segmented-regret-v1",
  "feature_schema_version": "aspect-work-v1",
  "backend": "cuda",
  "architecture_class": "sm90-native-vnni-v1",
  "policy_digest": "sha256:...",
  "candidate_registry_hash": "sha256:...",
  "format_registry_hash": "sha256:...",
  "packing_abi_hash": "sha256:...",
  "corpus_hash": "sha256:...",
  "development_fold_manifest_hash": "sha256:...",
  "sealed_split_manifest_hash": "sha256:...",
  "frozen_serial_m1_policy_digest": "sha256:...",
  "serial_m1_policy_hashes": ["sha256:..."],
  "contracts": {
    "Fast": {
      "frozen_generic_policy_digest": "sha256:fast-generic...",
      "exact_entries": 612,
      "generic_rules": 104,
      "development_cv_p95_regret_pct": 1.6,
      "sealed_coverage_pct": 100.0,
      "sealed_p95_regret_pct": 1.7,
      "sealed_max_regret_pct": 2.6,
      "sealed_simultaneous_max_ucb_regret_pct": 2.9
    },
    "VerifierSerialM1Bitwise": {
      "frozen_generic_policy_digest": "sha256:verifier-generic...",
      "exact_entries": 612,
      "generic_rules": 96,
      "bitwise_cells_passed": 7344,
      "bitwise_cells_failed": 0,
      "development_cv_p95_regret_pct": 1.8,
      "sealed_coverage_pct": 100.0,
      "sealed_p95_regret_pct": 1.9,
      "sealed_max_regret_pct": 2.5,
      "sealed_simultaneous_max_ucb_regret_pct": 2.8
    }
  },
  "emitter_round_trip": "pass",
  "staged_grouped_verifier_gate": "pass"
}
```

## Appendix D: Current Implementation Surface Inventory

Common runtime and refresh:

- `src/v2/tensors/TensorKernels.h`
- `scripts/refresh_native_vnni_dispatch_tables.sh`
- `tests/v2/performance/kernels/native_vnni_codebooks.py`
- `tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py`

CUDA:

- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp`
- `tests/v2/performance/kernels/cuda/gemm/infer_gemv_dispatch_heuristic.py`
- `tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu`
- `src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc`

ROCm:

- `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Throughput.cpp`
- `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_GEMM.cpp`
- `tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc`

CPU:

- `tests/v2/performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_GEMV.cpp`
- `tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc`

Canonical integration coverage:

- `tests/v2/utils/QuantizedVerifierFormats.h`
- `tests/v2/integration/kernels/cuda/Test__CUDAGemmParity.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmSmallM.cpp`
- `tests/v2/integration/kernels/cpu/native_vnni/Test__CPUNativeVNNI_GEMV.cpp`
- `tests/v2/integration/parity/qwen36/Test__Qwen36_MTPForwardVerifierOperationEquivalence.cpp`
