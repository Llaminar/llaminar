# Model-free native CPU kernel checks

This CMake project shares production kernel sources and canonical Unit/preflight
registrations with the full V2 build. It does not build an alternative inference
engine, admit a model, initialize GPU backends, or certify production
factory/arena/cache/request lifetimes. Passing it does not replace the complete
engine Unit and `ProductionTestPreflight` gates.

## Configure, build, discover

Use genuine C/C++ toolchains, installed GTest/OpenMP and the dependencies required
by the shared registrations. Select the actual architecture's compiler envelope;
foreign or unavailable ISA requests fail instead of choosing another path.

```sh
NINJA_BIN="$(command -v ninja)"
cmake -S tests/v2/portability -B build_arm_portability -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMINAR_CPU_ISA=NEON \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="$NINJA_BIN"
cmake --build build_arm_portability --parallel
ctest --test-dir build_arm_portability -N
ctest --test-dir build_arm_portability --output-on-failure --parallel
ctest --test-dir build_arm_portability -L '^ProductionTestPreflight$' \
  --repeat until-fail:20 --output-on-failure --parallel
```

`NEON_DOTPROD` and `NEON_I8MM` are separate admitted compiler envelopes. On x86-64, configure
separate `AVX2` and `AVX512` build directories; do not relabel one binary's
observations as another ISA. The supported values and exact flags belong to
`cmake/CPUInstructionSet.cmake`. Keep Debug, sanitizer, alternate-compiler and
Release evidence separate. Test discovery, not a copied count or name list,
is the inventory authority.

Integration defaults initialize both compiler languages before `project()` opens
its per-configuration caches. Authored nonempty final flags and preexisting
normal/cache `_INIT` values remain caller-owned; defaults do not force another
envelope. An inherited empty Integration cache is rejected: use a fresh build
directory or explicitly supply the intended optimized flags. A configuration
name alone is not optimization evidence. The installed native C/C++ witness
checks optimization-present/NDEBUG; exact flags, symbols, ISA and every relevant
TU still require actual compile-command/codegen evidence. Cache-state tests use
genuine CMake without a runtime compiler SDK or executable temporary mount.

AppleClang may require explicit paths to a genuine installed OpenMP runtime.
Configure the standard `OpenMP_CXX_FLAGS`, `OpenMP_CXX_INCLUDE_DIR`,
`OpenMP_CXX_LIB_NAMES` and `OpenMP_omp_LIBRARY` cache entries rather than creating
headers/libraries or weakening runtime identity checks. This focused build does
not implement the full engine's platform memory/placement backend.

## Original INT32 RMSNorm

The shared RMS targets compile the actual production TU and the legacy INT32
and precision caller TUs. Focused INT32 preflights exercise full-width FP64
summary programs, native original-row code/scale publication, explicit scalar
controls, complete-bank exclusions and immediate retained completion. The
precision target's focused case replaces an obsolete decoded-row helper; its
other precision cases retain their canonical full-engine Unit registration.

```sh
cmake --build build_arm_portability --parallel --target \
  v2_test_cpu_rmsnorm_workshare v2_test_cpu_rmsnorm_precision
ctest --test-dir build_arm_portability -R '^V2_Integration_CPURMSNormInt32' \
  --output-on-failure --parallel
```

Scalar diagnostics are not an empty native receipt. Original integers widen
straight to FP64 for summaries; publication separately rounds
`(float(x)*inverse)*gamma`, with no decoded/TLS row or scratch resizing. Native
vector RNE and scalar-tail ties-away remain distinct, as do ISA summary trees.
This row-INT8 scale has no block-Q8/FP32-quantizer floor. Do not infer factory,
physical placement, whole-engine inference or economy from these checks.

## Isolated AQ8 kernel economics

The canonical model-free `Perf__CPUAttentionKeyQ8.cpp` target measures exactly
one head dimension, native ISA and operation per process:

```sh
cmake --build build_arm_portability --parallel --target v2_perf_cpu_attention_key_q8
build_arm_portability/v2_perf_cpu_attention_key_q8 --help
build_arm_portability/v2_perf_cpu_attention_key_q8 \
  --operation dot --query-rows 4 --head-dim 128 \
  --heads 128 --iterations 2000 --workers 1 --isa neon
```

`encode` and `decode` retain their own programs. `dot` consumes original
compressed keys and shares register fragments across `--query-rows 1..4`;
other row counts reject, and encode/decode do not accept multi-query rows.

Dot fixtures, immutable query/anchor/ISA binding, independent decode/dot
reference authentication, complete NaN poisoning and warmup precede timing.
The timed loop does not allocate, rebind inputs or construct decoded heads.
`dot_rowsN` CSV latency is per compressed key and its N-query tile; divide by N
only for an explicitly reported per-query comparison. Dot CSV also retains all
seven sample latencies in original sample order. A poisoned no-op or partially
written bank cannot receive `byte_exact=true`.

Record the exact binary/arguments, compiler/runtime ISA, inherited placement and
hardware topology. Derive worker budgets from actual accessible physical cores;
one worker above selects one isolated candidate, not a general build/test cap.
Use separate process launches for each width/shape/worker candidate and keep
profiler overhead outside canonical timing. NUMA first-write setup is not proof
of physical CPU placement. Follow `.agents/cpu-tuning/SKILL.md` for actual PMU,
ISA, NUMA and scaling qualification. A benchmarked tile or leaf does not prove
that a production caller used it; the real entry, ownership and path evidence
remain separate requirements.
