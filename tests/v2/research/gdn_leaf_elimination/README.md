# Gated DeltaNet leaf-elimination research harness

This is a reproducible core-recurrence experiment, outside production inference.
It distinguishes exact algebra, floating-point byte equivalence, and measured
component cost. The dated findings live in the
[project report](../../../../docs/v2/projects/2026-10/gated-deltanet-leaf-harness-results.md).
The CPU arithmetic change and backward-query comparison have a separate
[follow-up report](../../../../docs/v2/projects/2026-10/gated-deltanet-shared-readout-and-backward-query-results.md).

The fixed-period CPU implementation now lives in production; see the
[whole-model experiment](../../../../docs/v2/projects/2026-10/gated-deltanet-epoch-production-results.md).
The native bridges below describe historical dense-state experiments and require
their frozen core **and matching source headers**. Configuration rejects the new
CPU history ABI rather than passing a dense buffer to it. Current production
validation uses `v2_test_cpu_fixed_epoch_gdn` and the registered GDN gates.

`harness.py` implements three independent evaluators: ordinary dense recurrence,
the full triangular correction system, and internal-only corrections with
adjusted leaf queries. It supports exact rational, FP64 and explicit FP32
arithmetic, bounded accepted histories, one pending update, rejection, internal
acceptance, every endpoint's continuation, and flushes. Its fixed LCG and seeds
make the inputs reproducible without a model or accelerator. The two query
variants adjust only leaves or every output. Reductions have a fixed order and
do not call BLAS. The 980-case sweep uses available physical cores.

`native.py` calls the real `CPUGatedDeltaNet::recurrent_step` through
`NativeCPU.cpp`, authenticating both state and output against an independent
FMA control before comparing adjusted-query readouts. It tries fused FP32 and
FP64 readout arithmetic. The supplied keys are already normalized; the native
kernel owns query scaling and gate conversion with `use_qk_l2norm=false`.

`exact.py` exercises a different, memory-saving variant in `ExactLeafCPU.cpp`.
It preserves native FP32 rounding and both projections, groups sibling rows,
retains dense internal states, and suppresses dense leaf states. The accepted
leaf retains its dense parent plus `(decay, key, exact correction)`. The next
verification fuses its materialization into the first key pass. It tests every
accepted endpoint, continuation, terminal materialization, input ownership,
guarded extents and large/small/empty replay. The experimental tile has four
sibling rows and eight values, using AVX2 instructions; the production oracle
can independently select AVX2 or AVX-512. This is not two separately optimized
backend implementations.

`benchmark.py` times two complete native star verifications with a fixed
accepted path and a final dense-state publication. Both candidates use the
same operands and four-row arithmetic. The dense control completes row groups
before advancing to keep partially written snapshots cache-local. The compact
candidate pays for old-pending materialization and the terminal flush. Setup,
allocations, preprocessing, oracle checks and warmup are outside native timing.
Raw interleaved samples are retained. This is a synthetic component comparison,
not a production or full-model speedup claim.

## Run the retained checks

From the repository root, with the repository Python requirements installed:

```bash
python3 tests/v2/research/gdn_leaf_elimination/test_harness.py
python3 tests/v2/research/gdn_leaf_elimination/harness.py \
  --output parity-results/gdn-leaf-harness/reference.json \
  --require-byte-equivalence
```

The unit checks pass when the experiment's algebra, lifecycle and diagnostic
contracts hold. The original proposal's strict sweep intentionally exits **1**
on observed byte differences; exit zero without `--require-byte-equivalence`
means only that the diagnostic completed. It never certifies production.
The fast reference tests are also registered as
`V2_Unit_GDNLeafEliminationReference` in the canonical test inventory.

For native checks, explicitly select an existing shared-core build and its
matching source tree. The native bridge requires a Release build of its own.
Correctness checks can observe an Integration or Release core; timing requires
both bridge and core to be Release. The bridge does not rebuild that core, so
its evidence is an existing-binary diagnostic rather than a fresh source gate.
For a workspace that is changing concurrently, capture the core DSO and its
matching source/header dependency closure first. Select those immutable copies
with `-DLLAMINAR_CORE_LIBRARY=/absolute/path/to/libllaminar2_core.so` and
`-DLLAMINAR_CORE_SOURCE=/absolute/path/to/snapshot/src/v2`; retain
`LLAMINAR_CORE_BUILD` for the original build metadata. Record the snapshot
hashes alongside the reports and check the bridge's resolved library with
`ldd` before running. These options select an explicit oracle, never a substitute
for a failed native control.

```bash
GDN_NINJA_BIN="$(command -v ninja)"
cmake -S tests/v2/research/gdn_leaf_elimination -B build_gdn_leaf_research \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${GDN_NINJA_BIN}" \
  -DLLAMINAR_CORE_BUILD=/absolute/path/to/build_v2_release
cmake --build build_gdn_leaf_research --parallel
ctest --test-dir build_gdn_leaf_research \
  --output-on-failure --parallel --no-tests=error

OMP_NUM_THREADS=1 LLAMINAR_ISA_LEVEL=avx2 \
  python3 tests/v2/research/gdn_leaf_elimination/exact.py \
  --library build_gdn_leaf_research/libv2_research_gdn_leaf_native.so \
  --output parity-results/gdn-leaf-harness/exact-avx2.json
```

Repeat the native command with `LLAMINAR_ISA_LEVEL=avx512` when supported by the
selected core and hardware. No failed native control is skipped or replaced.
Use `native.py --library ... --output ...` to retain the adjusted-query
counterexamples. The standalone CTest inventory is explicitly research-only.

For a single-thread timing experiment, select an available physical CPU after
inspecting the actual topology; the command below uses CPU 0 as an example:

```bash
taskset -c 0 env OMP_NUM_THREADS=1 LLAMINAR_ISA_LEVEL=avx512 \
  python3 tests/v2/research/gdn_leaf_elimination/benchmark.py \
  --library build_gdn_leaf_research/libv2_research_gdn_leaf_native.so \
  --output parity-results/gdn-leaf-harness/lifecycle.json
```

Reports contain seeds, comparison counts, first mismatches, source/binary
identities and raw timing samples. Keep generated reports and assembly under
ignored `parity-results/`; do not publish them as certified corpora. The native
bridge uses same-machine build paths and is not an installed test-runner asset.

Promotion requires an explicit production design for state ownership and tree
topology, backend-specific arithmetic proofs, complete capture/publication and
prefix-cache coverage, production preflight, model accuracy panels and matched
Release serving benchmarks. None of those are implied by a research pass.

## Experimental shared CPU readout

This experimental branch changes the production CPU serial, prefill and grouped
readout to one FP32 program. `GDN_RESEARCH_SHARED_READOUT` selects the matching
standalone research bridge; it is not a production runtime switch:

```text
dot = increasing-key FMA(k[j], q[j], dot)
eta = beta * dot
qhat[j] = decay * FMA(-eta, k[j], q[j])
output[v] = FMA(eta, value[v], increasing-key FMA(parent[j,v], qhat[j], sum))
```

Here `q` and `k` have already undergone the ordinary normalization and query
scaling. Dense state advancement keeps its existing rounded operations. This
readout intentionally changes old serial bytes and requires new model
qualification. Its arithmetic revision also separates persisted prefix-cache
entries from the old definition. GPU readouts are outside this experiment.
The default `native.py` and `exact.py` controls above require a core with the
old definition; use the following bridge with the changed core.

```bash
cmake -S tests/v2/research/gdn_leaf_elimination -B build_gdn_shared_harness \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${GDN_NINJA_BIN}" \
  -DLLAMINAR_CORE_BUILD=/absolute/path/to/changed_cpu_core_build \
  -DGDN_RESEARCH_SHARED_READOUT=ON
cmake --build build_gdn_shared_harness --parallel
OMP_NUM_THREADS=1 LLAMINAR_ISA_LEVEL=avx512 \
  python3 tests/v2/research/gdn_leaf_elimination/shared.py \
  --library build_gdn_shared_harness/libv2_research_gdn_leaf_native.so \
  --output parity-results/gdn-shared-readout/core-avx512.json
```

`shared.py` compares every readout, accepted state, endpoint continuation and
terminal flush with the actual changed production M=1 kernel, whose bridge
also checks an independent interpreter. It covers four successive rounds,
empty/chain/star/irregular trees, raw-operand ownership, poisoned retained
banks, M=1..16/31/63 and large/small/empty replay. Internal nodes keep dense
states. Leaves perform only the adjusted-query projection and retain their
immutable parent plus raw key/value/gates; resolving an accepted leaf pays its
previously omitted key projection. Rejected leaves never compute it.

After model qualification, configure the bridge against a changed **Release**
core and run `shared_benchmark.py --library ... --output ...` with the same
physical-CPU pinning used above. The three interleaved controls are shared
leaves, shared dense snapshots, and the previous compact algorithm with old
arithmetic. Each pays two verification rounds and terminal state publication;
the shared candidates also pay adjusted-query preparation inside the timed
interval. Each control must match its own serial output definition and the
common state-update definition before its timings are admitted.

This branching evaluator remains research-only. The production linear MTP
verifier still needs successive states, so eliminating rejected sibling
projections does not by itself predict a served-model speedup. Compare ordinary
Release serving separately with a frozen binary from before the arithmetic
change. Keep model qualification, core byte equivalence, synthetic timing and
whole-model timing as separate results.

## Backward queries and bounded-suffix combination

The shared-readout build also produces `libv2_research_gdn_backward_native.so`.
`backward.py` compares three explicit modes on identical inputs: leaf-only,
full backward query propagation, and a combined mode retaining the upper
rounded prefix and contracting at most two remaining updates. It also checks
the literal backward `y` accumulator separately from a final FMA chain that
preserves the shared one-hop arithmetic. `test_backward.py` proves exact-rational
equivalence at every prefix cut against independently materialized states.

```bash
python3 tests/v2/research/gdn_leaf_elimination/test_backward.py
OMP_NUM_THREADS=1 LLAMINAR_ISA_LEVEL=avx512 \
  python3 tests/v2/research/gdn_leaf_elimination/backward.py \
  --library build_gdn_shared_harness/libv2_research_gdn_backward_native.so \
  --output parity-results/gdn-shared-readout/backward-correctness.json
OMP_NUM_THREADS=1 LLAMINAR_ISA_LEVEL=avx512 taskset -c 0 \
  python3 tests/v2/research/gdn_leaf_elimination/backward.py \
  --library build_gdn_shared_harness/libv2_research_gdn_backward_native.so \
  --measure --samples 24 --iterations 32 \
  --output parity-results/gdn-shared-readout/backward-timing.json
```

Rebuild the standalone bridge after enabling the shared-readout option. Select
an actual physical CPU from local topology rather than assuming CPU 0 is free.
Timing requires both core and bridge to be Release. Repeat in fresh processes;
retain each output rather than overwriting evidence. Normalization and gates,
allocation and topology preparation are outside native timing; query contraction,
two verify/accept rounds, missing accepted-state updates and final publication
are inside it. Checkpoint loads are shared across up to four live queries.

This driver is a **diagnostic**: exit zero means its controls, state bytes,
guards and known rounding witness behaved as expected. Consult each mode's
`byte_equivalent` and `serial_outputs.mismatches`; a timing record with differing
output bytes is explicitly inadmissible for production. Passing exact-rational
tests or accepted-state reconstruction does not override an output-byte failure.

## Common serial/grouped arithmetic with fixed epochs

`EpochReadoutCPU.cpp` and `epoch.py` define an additional research contract in
which ordinary M=1 and grouped execution both contract queries backward. A fixed
period P, independent of speculation depth and acceptance boundaries, determines
when the rounded dense checkpoint advances. State owns the checkpoint, live
phase, and raw pending key/value/decay/beta records. Acceptance and serialization
preserve that complete state; a partial-epoch flush changes the arithmetic.
The period and contract revision must belong to any future cache identity.

The independent scalar serial interpreter supplies output and complete-state
oracles. Backward mode groups queries by their canonical checkpoint; combined
mode additionally reuses rounded prefixes to construct required checkpoints.
Auxiliary prefix states never become query readout bases. Both modes must match
the same serial contract, including inactive serialized slots, at every phase.
The driver fails on any byte difference. `test_epoch.py` covers every acceptance
endpoint, rejection, all periods 1..32, invalid geometry/counts, and a negative
control demonstrating why dense-only snapshots are invalid.

The shared-readout build above also produces `libv2_research_gdn_epoch_native.so`.
Run the full standalone suite or its individual epoch controls:

```bash
cmake --build build_gdn_shared_harness --parallel
OMP_NUM_THREADS=1 ctest --test-dir build_gdn_shared_harness \
  --output-on-failure --parallel --no-tests=error
OMP_NUM_THREADS=1 python3 tests/v2/research/gdn_leaf_elimination/epoch.py \
  --library build_gdn_shared_harness/libv2_research_gdn_epoch_native.so \
  --output parity-results/gdn-shared-readout/epoch-correctness.json
OMP_NUM_THREADS=1 python3 tests/v2/research/gdn_leaf_elimination/test_epoch.py \
  --library build_gdn_shared_harness/libv2_research_gdn_epoch_native.so
OMP_NUM_THREADS=1 taskset -c 0 \
  python3 tests/v2/research/gdn_leaf_elimination/epoch.py \
  --library build_gdn_shared_harness/libv2_research_gdn_epoch_native.so \
  --measure --samples 24 --iterations 32 \
  --output parity-results/gdn-shared-readout/epoch-matched-timing.json
```

Correctness uses available physical cores and covers nine topologies, three
geometries and every phase of P=1/2/4/8/16. A 65-token stream is repartitioned into
1/2/3/7/16/31 and mixed groups, with complete-state serialization round trips.
Timing is pinned to one selected physical core and compares the two epoch modes
with the qualified one-hop leaf algorithm on identical operands and materialized
initial state. Each algorithm must match its own serial definition; the two epoch
modes additionally share identical output/state bytes. The scalar epoch oracle
is not timed as a performance baseline. Both verification rounds, acceptance,
required checkpoint flushes and complete terminal serialization are timed.

Repeat timing in fresh processes. For a period/shape, sum each candidate's median
latency over every initial phase within a process, then divide the leaf baseline
sum by each epoch candidate sum. Report the median and range of those paired
ratios; selecting only a favorable initial phase overstates the benefit. Equal
phase weighting does not establish the phase distribution of a real model.

This contract is not installed in production memory admission, prefix-cache,
request banks or verifier snapshots. The one-hop model qualification cannot
qualify P>1. The dated report records the branching gains, star/chain regressions,
state-storage cost and remaining production qualification.
