# Isolated ROCm Kernel Profiling

Use this workflow after PerfStats has identified an expensive captured graph or
fragment. Profile a standalone harness that invokes the exact production kernel
entrypoint with the same geometry, tensor format, stream contract, and
persistent workspace. Do not disable production graph capture to make rocprof
attach to whole-model inference.

## Timing trace

Keep the canonical repeated timing run unprofiled. Use a separate one-iteration
profiler run to identify dispatch indices and kernel symbols:

```bash
RESULT=/tmp/llaminar-rocm-kernel-profile
mkdir -p "$RESULT"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
rocprof --stats --timestamp on --basenames off \
  -d "$RESULT" -o "$RESULT/dispatch.csv" \
  ./build_v2_release/tests/v2/<isolated-perf-binary> \
  --gtest_filter='Perf__ExactSuite.ExactProductionShape'

column -t -s, < "$RESULT/dispatch.stats.csv"
```

Repeat the unprofiled harness enough times to establish a stable median. Treat
rocprof timing as attribution evidence, not as the canonical benchmark sample.

## Candidate-only counters

Counter evidence belongs to one exact kernel/ISA/shape/M candidate. First read
the timing CSV and identify the candidate's dispatch range. On the rocprof v1
shipped with ROCm 7.1.1, `range: 1 : 3` selects dispatches 1 and 2. Confirm the
semantics against the emitted CSV whenever the profiler version changes.

Create a counter request such as `/tmp/rocm-counters.txt`:

```text
pmc : Wavefronts VALUUtilization VALUBusy SALUBusy LDSInsts LDSBankConflict ALUStalledByLDS
range: 1 : 3
pmc : FetchSize MemUnitBusy MemUnitStalled
range: 1 : 3
pmc : WriteSize
range: 1 : 3
pmc : L2CacheHit
range: 1 : 3
```

Then run only the deterministic isolated harness under that request:

```bash
COUNTERS=/tmp/rocm-counters.txt
RESULT=/tmp/llaminar-rocm-kernel-counters
mkdir -p "$RESULT"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
rocprof -i "$COUNTERS" --timestamp on --basenames off \
  -d "$RESULT" -o "$RESULT/counters.csv" \
  ./build_v2_release/tests/v2/<isolated-perf-binary> \
  --gtest_filter='Perf__ExactSuite.ExactProductionShape'
```

MI50 cannot collect every derived memory metric in one hardware pass. If
rocprof reports `Input metrics out of HW limit`, retain its valid grouping or
split the metrics into separate `pmc` records. rocprof reruns the executable for
each record, so inputs and dispatch ordering must remain deterministic.

Interpret related metrics together:

- `DurationNs` is dispatch latency; use it to attribute an unprofiled timing
  regression or win.
- `scr` and `.private_segment_fixed_size` expose private scratch. Both should be
  zero unless a measured exception is justified.
- `arch_vgpr`, `sgpr`, `lds`, and `Wavefronts` constrain occupancy and reveal
  whether the grid supplies enough waves for all CUs.
- `VALUUtilization` estimates active lanes in issued vector instructions. Very
  low utilization often indicates divergence or lane-zero serial work.
- `VALUBusy` and `SALUBusy` distinguish vector and scalar pipeline pressure.
- Interpret `LDSBankConflict` with `ALUStalledByLDS`. Bank conflicts are not the
  limiting factor when LDS stall is near zero and a conflict-reducing A/B
  variant is slower.
- `FetchSize`, `WriteSize`, `MemUnitBusy`, `MemUnitStalled`, and `L2CacheHit`
  distinguish bandwidth/cache pressure from comparison, reduction, and launch
  latency. Normalize traffic against logical bytes for the invocation.

## Wave64 vectorization and reductions

Do not diagnose scalar execution solely from one source value per thread or an
ISA `global_load_dword`. VMEM executes across active wave lanes: 64 contiguous
lane addresses make that instruction a coalesced 256-byte wave access. Verify
lane address progression, active-lane utilization, transaction bytes, and
throughput before changing it to per-lane `dwordx4` loads.

For cooperative reductions, confirm HIP shuffles lower to wave operations such
as `ds_bpermute_b32`. Compare against lane-zero loops. An exact Top-K kernel can
give every lane one sorted list, reduce current heads with the canonical total
comparator, broadcast the winning lane, and advance only its cursor. This
changes comparison topology without changing floating-point arithmetic, but it
must still pass the all-M grouped-versus-serial byte sweep.

## ROCm 7 code-object extraction

ROCm 7 HIP objects commonly store a bundle in `.hip_fatbin`. Extract the exact
target from the compiled object with the installed ROCm LLVM tools:

```bash
OBJ=build_v2_release/CMakeFiles/<target>.dir/path/Kernel.hip.o

/opt/rocm/llvm/bin/llvm-objcopy \
  --dump-section .hip_fatbin=/tmp/kernel.hip_fatbin "$OBJ" /dev/null
/opt/rocm/llvm/bin/clang-offload-bundler \
  --list --type=o --input=/tmp/kernel.hip_fatbin
/opt/rocm/llvm/bin/clang-offload-bundler \
  --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx906 \
  --input=/tmp/kernel.hip_fatbin \
  --output=/tmp/kernel.gfx906.co

/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/kernel.gfx906.co
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx906 \
  --disassemble-symbols='<mangled-kernel-symbol>' \
  /tmp/kernel.gfx906.co
```

Use the exact target printed by `clang-offload-bundler --list`; do not assume
all builds use `gfx906`. For each retained specialization, record
`.private_segment_fixed_size`, `vgpr_count`, `sgpr_count`, both spill counts,
and wavefront size. Census VMEM, LDS, shuffle/permute, and store instructions in
that exact symbol.
