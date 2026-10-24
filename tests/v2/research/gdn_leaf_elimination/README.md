# Gated DeltaNet leaf-elimination research harness

This is a reproducible core-recurrence experiment, outside production inference.
It distinguishes exact algebra, floating-point byte equivalence, and measured
component cost. The dated findings live in the
[project report](../../../../docs/v2/projects/2026-10/gated-deltanet-leaf-harness-results.md).

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
