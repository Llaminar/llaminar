# CPU GDN: shared leaf readout, backward queries, and their combination

Date: 2026-10-24. Worktree: `llaminar-gdn-leaf-harness`, branch
`feat/gdn-speculative-leaf-harness`, starting at `3dd32e833`.
This is a CPU-only experiment authorized to change serial arithmetic and redo
model qualification. CUDA and ROCm arithmetic were not changed.

**Arithmetic verdict:** backward queries can preserve a common serial/grouped
byte contract after redefining ordinary decode too. The fixed-epoch experiment
below proves this; differences from the older serial arithmetic are intentional.
Changing only the grouped evaluator still fails, as the earlier comparison in
this report shows. This report records the model-qualified one-hop shared readout and the
fixed-epoch research comparison. A later
[production experiment](gated-deltanet-epoch-production-results.md) integrates the
P=8 combination into the CPU core and measures whole models. The branching tree
evaluators remain research implementations; no whole-model tree speedup or
release certification is claimed here.

**Performance verdict:** retaining shared flush prefixes and contracting queries
backward is promising for branching trees. With P=8 and every checkpoint phase
included, the combination is 1.26x faster than the qualified leaf-only component
on both measured branching trees. Backward propagation alone is slower there;
the combination also loses on stars and has no reliable chain gain. Keep this
as a branching-tree research candidate, not a general CPU replacement.

## Follow-up: a common fixed-epoch contract

The user explicitly permitted further arithmetic changes while requiring one
contract for ordinary and grouped execution. `EpochReadoutCPU.cpp` implements
that contract with a fixed checkpoint period P. Each state owns its dense
checkpoint, a live count, and at most P−1 raw key/value/decay/beta records.
Every output contracts its query through the pending records and current token,
projects the canonical checkpoint, then applies value FMAs oldest to newest.
The checkpoint advances only after exactly P accepted positions, using the
original rounded update sequence. P belongs to arithmetic/cache identity.

Acceptance, grouping, snapshots and prefix restoration must **not** flush a
partial epoch. They copy the complete checkpoint/history state. A negative
control proves that prematurely converting it to a dense-only snapshot changes
the following output from `2^-25` to zero. Speculation depth and tree shape do
not determine rounding boundaries.

An independent scalar M=1 interpreter, grouped backward evaluation, and a
combined implementation all produce identical bytes. The combined implementation
reuses rounded intermediate prefixes only when constructing required canonical
checkpoints. It never substitutes those auxiliary states into a query readout.
Thus the two optimized algorithms share exactly the same arithmetic for a given P.

The final native suite passes 837 tree/phase/geometry cases over periods
1/2/4/8/16, every phase, nine topologies and three geometries. A 65-token stream
also passes repartitioning into 1/2/3/7/16/31 and mixed groups with snapshot
round trips. Additional focused tests cover every accepted endpoint and total
rejection, all admitted periods 1 through 32 and their phases, malformed state
counts, overflow-sized geometry, and the premature-flush negative control.
Inactive serialized history slots remain zero and the input state is immutable.
Each of the three modes passes 16,270,908 scalar comparisons in the tree sweep;
repartitioning adds 480,160 comparisons. All six standalone research CTests pass.

### Matched timing under the new contract

The final comparison uses the qualified one-hop leaf algorithm and both epoch
algorithms on identical prepared operands and identical materialized initial
state. Each is checked against its own ordinary M=1 definition: leaf-only uses
`adjusted-parent-fma-v1`; backward and combined share the fixed-epoch definition.
The two epoch algorithms have identical outputs and complete state for the same
P. Cross-contract output equality is deliberately not required. The independent
scalar epoch oracle is not a timing baseline.

Release AVX2 research kernels run on one physical core of the Xeon Gold 6238R,
pinned to CPU 0. Three fresh processes each use 24 samples, 32 complete lifecycles
per sample, 100 ms warmup per mode, and all six orders of the three candidates.
A lifecycle includes two verify/accept/continue rounds and terminal serialization
of the complete state. Required checkpoint flushes and accepted-history copying
are timed. Admission, allocation, topology planning, operand preparation and
oracle checks are outside timing for all candidates.

For each process, sum each mode's median latency across **every initial phase**
of the selected period, then divide the leaf sum by the candidate sum. The table
shows the median of these three ratios and their repeat range. This equally
weights phases; it is not a measured model acceptance/phase distribution. Higher
means faster; 1.00x is the qualified leaf-only component.

| Tree | Nodes / depth | P=8 backward | P=8 combined |
|---|---:|---:|---:|
| Star | 16 / 1 | 0.594x (0.553–0.603) | 0.594x (0.554–0.602) |
| Ternary | 39 / 3 | 0.833x (0.808–0.845) | **1.265x (1.247–1.266)** |
| Binary | 30 / 4 | 0.796x (0.775–0.863) | **1.262x (1.230–1.327)** |
| Chain | 16 / 16 | 1.009x (0.934–1.144) | 0.882x (0.828–1.011) |

The period sweep includes P=1/2/4/8/16. At P=4, combined gains are 1.176x and
1.206x on the ternary and binary trees; at P=16 they are 1.015x and 1.180x.
No tested period makes the combination a universal improvement. P=1 is a
separate implementation of the same one-hop arithmetic, so its timing difference
from the qualified leaf implementation is not an arithmetic benefit.

Full backward evaluation repeats shared-prefix work when several branches cross
a canonical checkpoint boundary. The combination reuses rounded prefixes for
those required flushes, which explains its advantage. Long pending history adds
work even to a star, where the leaf-only method already contracts a single
update. The earlier roughly 2x drop-in timings below did not pay for this
grouping-independent state contract and are not admissible speedup claims.

For `(dk,dv)=(128,128)`, the P=8 serialized research state is 73,796 bytes per
head versus 65,536 dense bytes, a 12.6% increase. It reserves eight raw records,
at most seven of which remain live between calls. The research workspace still
reserves candidate matrix slots; no reduction in admitted physical memory is
claimed. The four final projection bodies have no packed SIMD stack operands.
Hardware performance counters remain unavailable on this host.

These checks qualify the **new research arithmetic**, not the production model
graph: its full state representation is not installed in CPU prefix-cache,
memory-admission, request-bank or verifier-snapshot interfaces. The model results
later in this report qualify only the one-hop shared readout. They must not be
reused as model-quality evidence for a larger checkpoint period.

## Earlier experiment: the model-qualified one-hop contract

The [earlier investigation](gated-deltanet-leaf-harness-results.md) proved that
adjusted queries cannot recover the old rounded serial outputs in general.
This follow-up deliberately defines a new CPU readout, shared by ordinary
decode, grouped prefill/verifier rows, and request-batched decode:

```text
dot = fold_in_increasing_key_order(fma(k[j], q[j], dot))
eta = beta * dot
qhat[j] = decay * fma(-eta, k[j], q[j])
out[v] = fma(eta, value[v],
             fold_in_increasing_key_order(fma(parent[j,v], qhat[j], sum)))
```

Queries/keys and gates retain canonical preparation. State advancement retains
the old rounded decay, key projection, correction and FMA update. SIMD lanes
parallelize independent value columns without changing a dot's reduction order.
The CPU prefix fingerprint includes `adjusted-parent-fma-v1`, since later layers'
KV and recurrent states depend on these readouts. Old cache entries cannot be
reused under the changed arithmetic.

A speculative leaf needs only its adjusted-query action. It owns the immutable
parent and raw key/value/gates; only an accepted leaf pays its deferred key
projection and dense update. Internal nodes still materialize rounded states.
There is no triangular reassociation, production row replay, or runtime switch
to the older readout on this experimental branch.

## Drop-in tests against the one-hop contract

| Variant | Verification state policy | Backward work |
|---|---|---|
| Leaf-only | Retain every rounded internal state; omit leaf state/correction | One update per leaf |
| Full backward | Retain only the initial checkpoint | Entire root-to-candidate path, including the candidate |
| Combined | Retain the upper rounded prefix; omit remaining states/corrections | Suffixes of at most two updates |

The combined experiment uses `cutoff=max(1, maximum_depth-2)`. This deliberately
gives it actual rounded prefix states, a stronger byte-preservation control
than reconstructing a prefix from reassociated correction sums. It is one
concrete hybrid, not an exhaustive search over prefix policies.

The literal proposed accumulation computes `y=fma(c,v,y)` while walking backward,
then adds `S0*x`. It already differs from the new one-hop readout on
5,735/74,880 checked one-hop output scalars. The main comparison gives backward
contraction its more favorable FP32 ordering: retain the path coefficients,
project the final query, and apply coefficient/value FMAs oldest to newest.
This matches the one-hop readout exactly. It still fails at two or more updates.

Exact-rational tests cover 35 topology/gate combinations and every prefix cut.
All three identities pass. Native checks use actual production M=1, independently
authenticated by an FP32 interpreter: 81 cases across `(dk,dv)=(7,16),(64,64),
(128,128)`, nine topologies and three gate regimes, plus every acceptance endpoint
and rejection on an irregular tree. Both rounds use matching accepted histories.

| Variant | Cases with differing outputs / 81 | Different output scalars / 217,152 | Maximum absolute difference |
|---|---:|---:|---:|
| Leaf-only | 0 | 0 | 0 |
| Full backward | 44 | 58,810 | 3.8743e-7 |
| Combined | 29 | 21,750 | 3.5763e-7 |

Every variant passes all 1,114,208 accepted-state scalar comparisons, including
continuation after rejection and acceptance. Missing accepted updates are
resolved in the original rounded order; their cost is included in timing.
Small output differences are recorded as byte failures, not tolerance passes.

A minimal witness isolates the problem. Start with scalar state 1 and apply an
identity prefix. Next use `k=decay=beta=1`, `v=2^-25`. Rounded state advancement
produces zero because `fl(2^-25-1)=-1`. A following query with `q=decay=1`, `k=0`
therefore returns zero under the new serial definition. Backward contraction
returns `2^-25`. Keeping the identity prefix's exact rounded state does not fix
the combined variant. The native harness reproduces both results.

## Earlier drop-in component timing

Release core and Release AVX2 research kernels, GCC 14.2, Xeon Gold 6238R,
one physical core pinned to CPU 0. Each of three fresh processes records 24
samples per mode, 32 repeated lifecycles per sample, 100 ms warmup per mode,
and all six candidate orders. One lifecycle verifies two trees, accepts a leaf
in each, resolves missing accepted updates, and publishes terminal dense state.
Query contraction and prefix-state construction are timed. Allocation,
topology planning, normalization/gates, oracle checks and warmup are not.
All candidates group up to four queries sharing the same checkpoint.

Ratios below are the median of the three within-process ratios against the
leaf-only candidate in this same tree harness. Higher means faster. These
ratios do not multiply with the separate leaf-versus-legacy measurements.

| Tree | Nodes / depth | Full backward | Combined | Additional byte-safe gain? |
|---|---:|---:|---:|---|
| Star | 39 / 1 | 1.00x | 1.01x | No; same one-hop arithmetic, within noise |
| Four-by-four | 20 / 2 | 1.38x | 1.02x | Full backward fails; hybrid reduces to one-hop |
| Ternary | 39 / 3 | 2.12x | 1.90x | Both fail bytes |
| Binary | 30 / 4 | 2.09x | 1.63x | Both fail bytes |
| Chain | 16 / 16 | 0.91x | 1.02x | Both fail bytes; full-backward range 0.87–1.17x |
| Irregular | 9 / 4 | 1.65x | 1.07x | Both fail bytes |

Ternary ratios range 2.04–2.16x and 1.85–1.93x across processes. Its two-round
query/key action counts are leaf-only `78+26`, backward `78+6`, combined `78+10`.
Backward suffix-update counts are 54, 204 and 126 respectively, excluding the
ordinary one-hop prefix computation. On the chain all three pay `32+32` matrix
actions once accepted-state resolution is included; full backward adds much
more path work. Fewer candidate actions alone are not an economy proof.

The separate matched star lifecycle compares the shared leaf readout with the
previous old-byte-exact compact prototype. It includes both deferred resolutions
and terminal publication, uses 64 lifecycles/sample, and passes each candidate's
own serial byte oracle. Median paired gains are 1.13x at M=2, 1.71x at M=4,
1.93x at M=8, 2.16x at M=16, and 2.18x at M=39. M=1 is 0.85x, about 17% slower.
These are cache-hot core measurements, not served model throughput.

Assembly inspection found packed accumulator stack traffic in the first backward
kernel. Explicit fixed-row unrolling and separate action bodies removed it; the
reported timing uses that final build. Its four action bodies and the shared
leaf kernel have no packed SIMD stack operands. The production readout saves
scalar `eta` across ISA queries; this is not a claim of zero stack traffic.
Hardware `perf` counters were denied (`perf_event_paranoid=4`), so IPC/cache and
achieved bandwidth attribution remain unmeasured. Host settings were not changed.

## Model qualification and production-path evidence

The shared-readout production core passed **737 Unit + 612 ProductionTestPreflight
entries**, with zero failures. Distinct AVX2 compilation also passed focused
readout/GDN/BF16-router checks. Both the Release AVX512 core and distinct AVX2
core pass the larger 84-case native leaf suite, each with 147,099,264 exact scalar
comparisons across outputs, state, continuation, terminal publication and live
extents. Existing grouped CPU operation tests also pass.

The official production-campaign driver passed the complete seven-case Qwen3.5
CPU campaign: 0.8B Q4_0 with FP16/Q8_1/Q16_1 KV; 4B Q8_0 with FP16 KV, including
20-step decode; and 27B Q8_0 with FP16/Q8_1 KV. All 50 decoded tokens match their
HF references. Minimum cosine is 0.999652; maximum KL is 0.00288029.

Six focused official Qwen3.6 35B-A3B IQ3S CPU cells also pass: MTP off, fixed
depths 1/2/3/15, and dynamic depth. Each retains the canonical three-row HF
check and prefix lifecycle, with 2/3 HF tokens matching, cosine at least 0.99869
and KL at most 0.00760347. No thresholds, oracle outputs or reference metadata
were changed. These are six selected cells, not the full twelve-cell MoE
campaign: the separate Ornith Q8 reference pack was unavailable.

Qualification exposed a preexisting CPU BF16 MTP router rejection, reproduced
by the frozen old Release binary. The repair consumes native BF16 weights with
the existing fixed-tree SIMD matmul and unchanged FP32 activations. It preserves
Q8 expert-input publication and adds focused AVX2/AVX512 preflight regressions.
CPU-only gate registration/factory defects and one filesystem-timestamp test
fixture race were also corrected; no unavailable native GPU test is fabricated.

An actual Release stochastic MoE run completes 384 tokens with default dynamic
MTP, 141 verifier transactions, 363 stochastic acceptance tests, 243 accepts,
120 residual samples, and depth 1–15 with 11 policy updates. PerfStats records
1,490,880 adjusted-readout rows and 817 native BF16-router calls. This
instrumented run proves path execution, not timing or distributional equivalence.

## Whole-model timing and remaining qualification limits

Six fresh Release processes per dense model, order old/new/new/old/old/new,
one warmup and one measured request each, CPU 0–27, 28 physical workers, normal
MPI bootstrap, 595 prompt tokens and 384 generated tokens. Prompt SHA256 is
`f61f5a8f5078abf10fbe5f6805fa79cc06633764b26a9419d264ba1aaf63861f`.
Greedy sampling uses temperature 0, without the diagnostic deterministic-kernel
mode; prefix caching remains enabled. Models without a sidecar select MTP off.

0.8B matches all 384 generated tokens in all six runs. Median decode is
38.09 → 38.74 tokens/s (+1.7%), but the ranges overlap: old 37.84–38.21,
new 38.11–40.02. Prefill is 1,679 → 1,692 tokens/s. This is no demonstrated
whole-model speed win. The 4B greedy continuations first differ at token index
93; median decode is 14.50 → 14.29 tokens/s, but it is not an exact-token
comparison. The retained stochastic 0.8B cohort also differs in tokens and is
not used for a matched speed claim.

The production grouped-GDN scaling diagnostic covers AVX2/AVX512 runtime routes,
1/2/4/8/16/28 physical workers, M=1/4/16, 32 value heads, 16 key heads and width
128. Both old and new cores preserve bytes. The existing economy assertion
fails on several low-worker cases in **both** binaries; it was not lowered.
The diagnostic is best-of-32, not repeated median evidence. At 28 workers the
new/old grouped latency ratios span gains and regressions, so it does not certify
universal CPU economy. Complete serving-tree integration, broader model/HTTP
qualification, and economy across supported worker counts remain open.

Changing only the readout is sufficient for the one-hop leaf experiment. The
fixed-epoch research definition now also establishes byte invariance for longer
contractions. Installing it would require its complete state/history layout in
the production memory and cache authorities, a new arithmetic fingerprint, and
fresh model and economy qualification. The one-hop model results do not qualify
that replacement. The current production verifier is linear; the measured
branching-tree gain does not establish an improvement for that workload.

## Comparison with the proposed baselines

[TreeWY §3](https://arxiv.org/html/2608.20961v1#S3) computes pseudo-values through
a strict-ancestor triangular system, then reconstructs the accepted state. Its
FP32/BF16 correctness criteria permit numerical differences; §4 explicitly says
token streams are not bit identical. It therefore does not establish Llaminar's
serial-byte contract. Backward queries replace its candidate correction solve
with per-path contractions; the local tests confirm the algebra, not equivalent
production semantics.

[ReplaySSM's verifier](https://github.com/Johnny-Liou/ReplaySSM/blob/main/vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py)
loads checkpoint tiles once for both query and key projections, forms the
intra-window triangular correction system, and stores circular correction/key/
gate history. Accepted cursors and flushes govern checkpoint materialization.
Thus removing a projection does not necessarily remove a checkpoint load.
Its linked implementation is a GPU chain verifier. Neither external kernel was
run on this CPU host, so the timings above are not head-to-head TreeWY or
ReplaySSM results.

## Reproduction and local evidence

Commands and contracts are in the
[harness README](../../../../tests/v2/research/gdn_leaf_elimination/README.md).
All raw evidence stays ignored under `parity-results/gdn-shared-readout/`:

- `prerequisites-r6/prerequisites.json`, `experiment-identity.json`.
- `core-release-avx512-qualified-build.json`, `core-distinct-avx2-qualified-build.json`.
- `model-panel-summary.json`, `model-*-r2.json`, `reference-metadata-validation.json`.
- `shared-lifecycle-r{1,2,3}.json`, `shared-lifecycle-summary.json`.
- `backward-correctness.json`, `backward-timing-r{1,2,3}.json`, `backward-timing-summary.json`.
- `epoch-correctness-final.json`, `epoch-matched-r{1,2,3}.json`, `epoch-matched-summary.json`.
- `epoch-assembly.json`, `epoch-actions.asm`, `epoch-final-evidence.json`, `epoch-final-ctest.log`.
- `release-greedy-summary.json`, `release-greedy-{08b,4b}/`, `release-08b/`.
- `release-moe-path.json`, `release-moe-path-perfstats.json`, `release-moe-path-summary.json`.
- `production-gdn-scaling/`, `final-assembly-identities.json`, `perf-access.txt`.

Earlier failed qualification and initial code-generation evidence are retained
separately. No result payload or local model/reference pack belongs in a commit.
