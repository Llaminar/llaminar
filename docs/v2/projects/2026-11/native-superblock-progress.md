# Native CPU superblock progress repair

## Walk progress

A superblock walk over `[first, last)` advances whenever
`0 <= first <= last <= blocks <= INT_MAX` and the phase is in `[0,7]`. Any
iteration it enters has `block < last <= blocks`. The untruncated
source-superblock end is strictly greater than `block`, so clipping it to
`last` still advances. The size_t intermediate is bounded by INT_MAX + 15, and
the clipped end fits in an int.

## Ordered partition arithmetic

The hot programs keep their original int expressions unchanged:

    first = begin + p * width
    last  = min(first + width, end)

NEON uses `begin` = 0 and `end` = blocks_per_row. x86 single-scale uses the
admitted tile range. For p < partitions, the widest intermediate is
`begin + p * width + width <= begin + partitions * width`. Every expression is
therefore representable in int exactly when
`int64(begin) + int64(partitions) * width <= INT_MAX`.

- **NEON bound.** With `begin` = 0, the bound is `partitions * width <= INT_MAX`.
- **x86 bound.** The bound is `begin + partitions * width <= INT_MAX`, together
  with `0 <= begin <= end <= blocks_per_row`.

This is a typed API-representability constraint. It is not a capacity cap or a
topology whitelist:

- One partition of INT_MAX blocks is admitted.
- INT_MAX blocks in two partitions are rejected: the width is 2^30, so the
  product is 2^31. That plan genuinely exceeds the existing hot int arithmetic.

The NEON view derives `blocks_per_row = ceil(K / 32) <= 2^26` and uses at most
sixteen partitions. An admitted NEON publication therefore never reaches the
rejection. The check guards the invariant; it does not limit real geometry.

An excess partition has `first >= last`, an empty (possibly reversed) interval.
Every walk, including the IQ4_XS hoisted-header path, loads only inside
`block < last`. Such a partition therefore reads nothing and terminates at once.

## One cold admission authority

`CPUNativeOrderedPartitionPlan.h` owns the noexcept predicates
`admitsSuperblockWalk`, `admitsOrderedPartitionArithmetic` and
`admitsOrderedPartitionPlan`. It also owns the fatal admissions
`requireSuperblockWalk` and `requireOrderedPartitionWidth`. The rejection
reporters `failSuperblockWalk` and `failOrderedPartitionPlan` are declared
`[[noreturn, gnu::cold, gnu::noinline]] noexcept` and are defined in
`CPUNativeOrderedPartitionPlan.cpp`, which owns the logging.

The header has no logging or stream dependency. The NEON tile register/stack
proof binds the probe's complete dependency closure, and that closure rejects
the iostream initialization assembly `utils/Assertions.h` brings in. A
`.globl _ZSt21ios_base_library_initv` structural failure on the retained GNU 14
NEON build demonstrated this. The reporters' always-active unreachable
assertion throws under noexcept, so a rejection is fatal in Release as well as
Integration.

### Producers and entries

The checks below run once per call or construction. None runs in a hot program.

- **Canonical width producer.** `requireOrderedPartitionWidth` computes the
  ceiling in int64, checks that it fits, and returns it for the existing int
  field. Two sites use it:
  - the NEON tile constructor (`blocks_per_partition_`), which runs once per
    tile, outside the K programs;
  - the NEON packed-prefill chunk producer (`per_partition`).

  A NEON plan rejection is fatal because it is a broken invariant of a sealed
  publication, not a recoverable request. `requireExecutable` is unchanged.
- **NEON tile source.** Its diff from `60694357f` is limited to the constructor's
  width initializer and one doc line, both edited in place. The source receives
  the plan header through `CPUNativeQuantisedWeightView.h`, the owner of the
  admitted `blocks_per_row` extent. Every role-mapped loop line is therefore
  unchanged, and the tile-stack roles entry differs only in its source hash.
  Neither the key map's bound headers (`CPUNativeQuantisedEncoding.h`,
  `CPUNativeQuantisedPreparedFootprint.h`, `NativeQuantisedFormatInfo.h`) nor the
  bound compact helper source (`CPUNativeQuantisedMultiScaleKernels.h`) changes.
- **x86 single-scale.** `requireSingleScaleTile` keeps its recoverable rejection
  and now uses the shared exact predicate `admitsOrderedPartitionArithmetic`.
- **Entry walk admission.** `requireSuperblockWalk` runs at the entry of:
  - the two x86 single-scale kernels;
  - the NEON packed-prefill `pack`;
  - the x86 packed-prefill walk.

  The original stuck range first=8, last=9, blocks=8 can reach these entries
  through their `begin`/`end` arguments.

There is no check inside `forPartitionBlocks` or in any per-partition or
per-column code.

## Tests

`V2_Unit_NativeSuperblockWalkContract` and the preflight
`V2_Integration_NativeSuperblockWalkContract` are registered in the explicit
preflight inventory. Both call the production predicates and producer directly.

- **Admitted plans.** These cover singleton, odd, sixteen-, seventeen- and
  37-partition plans, overpartitioning, empty rows, nonzero begin and all
  phases.
- **Exact boundary.** At the boundary, MAX passes and MAX + 1 fails, at begin 0
  and at nonzero begin.
- **Fatal controls.** These cover:
  - the original stuck range first=8, last=9, blocks=8;
  - the producer rejecting the two-partition INT_MAX plan, with its actual
    product in the diagnostic;
  - zero partitions and an invalid phase;
  - on x86, the actual single-scale kernel entry called with the stuck range.

Core-free NEON test and perf targets that include the tile headers compile the
reporter translation unit directly.

These device-free contract tests claim no kernel, stack, numerical or backend
qualification.

## NEON tile register pressure (GCC 14)

The stack gate compares every native NEON tile program with its develop
`63b053401` analogue (`CPUNativeTileStackTable.json`). The repairs keep the
decode, the INT32 group order, the FP32 block program, the block order and the
partition tree. Several are per-program choices that were measured with the
GCC 14 witness in the pinned toolchain:

- `opaqueStripColumn()`: whether an empty register barrier re-derives the strip
  column in every K block;
- `q5GroupFifth()`: which fifth-bit addressing a Q5 program uses;
- the ordered `-0` partition seed (`orderedPartitionSeedNEON()`) and the
  programs that keep the partition-0 select instead;
- the backend-native row-pair accumulator carry, and `no-thread-jumps` on
  `runCoded()`;
- the single-row SDOT barriers around the Q5_K/IQ3_S group decodes and the
  coded factor phase;
- `kUnscheduledPrograms` (compiled with `no-schedule-insns`) and
  `kUnthreadedPrograms` (exact programs compiled with `no-thread-jumps`), keyed
  by the witness's program key. Static assertions require well-formed, unique
  keys of instantiable programs. The register-evidence unit test requires every
  key to name a program the witness audits, so a stale key fails. The
  `CPUCodegenAttributeSourcePolicy` gate admits only these arithmetic-preserving
  `optimize` options, always together with `fp-contract=off` and
  `no-associative-math`.

None of these choices changes arithmetic, but each one depends on the
compiler's register allocation. On a compiler or flag change, measure them
again: run the tile proof driver for NEON, NEON_DOTPROD and NEON_I8MM
(Integration and Release). Regenerate the committed fingerprints only with
`tests/v2/cmake/CPUNativeTileCountRegeneration.py`, which reads every count
from the attempts' own records, and review its printed changes.
