# Native CPU superblock progress repair

The host walk boundary requires `0 <= first <= last <= blocks <= INT_MAX`
and phase in `[0,7]`. For any entered iteration, `block < last <= blocks`.
The untruncated source-superblock end is strictly greater than block;
clipping it to blocks and last therefore still advances. Size_t intermediate
arithmetic is bounded by INT_MAX + 15. The resulting end fits int.

Ordered partitions have one interval authority, `orderedPartitionInterval(begin,
end, partition, width)` in `CPUNativeOrderedPartitionPlan.h`. It forms the start
and end in int64 and saturates both at end, so excess partitions become explicit
empty intervals instead of reversed ranges and nothing overflows. The three NEON
prepared-projection sites (begin 0, end the weight row extent) and the two x86
single-scale sites (the admitted tile's begin and end) all use it. Neither the
partition count, the ordered fold nor any nonempty interval changes. The
packed-prefill producers already clip starts: NEON `p * per_partition` at its
row extent and x86 `kpart * width` in int64.

`admitsOrderedPartitionPlan` accepts a whole plan only if the range is an
admitted walk, there is at least one partition, the width is nonnegative and the
partitions cover the range. Every interval the helper then forms lies inside the
range, so it is an admitted walk by construction. NEON admits its canonical plan
once per call in `requireExecutable`, the batch binding's admission before any
team, using the same `blocksPerPartitionFor` width every tile executes; a
rejected plan is a `std::invalid_argument` there, like the other admission
failures of that function. The always-inlined per-partition x column hot
programs (`forPartitionBlocks`) carry no check. x86 admits its plan in
`requireSingleScaleTile` and its walk range at kernel entry, once per call. The
packed-prefill walk boundaries remain authoritative for their caller-provided
ranges.

The plan header has no logging or stream dependency, because the NEON tile
register/stack proof binds the probe's complete dependency closure and rejects
the iostream initialization assembly that `utils/Assertions.h` brings in. The
fatal wrappers `requireSuperblockWalk` and `requireOrderedPartitionPlan` live in
`CPUNativeSuperblockWalkContract.h`, used by the x86 kernels and both
packed-prefill walks. Their always-active unreachable assertion is noexcept at
these boundaries, making invalid geometry fatal in Release as well as
Integration; debug-only assertions would leave the original production hang
possible. No check occurs per block or per partition.

Focused tests call the production helpers directly over all phases, empty and
ragged rows, INT_MAX, singleton, odd and 37-partition plans, overpartitioning and
nonzero begin. They prove the intervals are ordered, contiguous, cover the range
and equal the unsaturated form whenever nonempty. Fatal controls include the
original stuck range first=8,last=9,blocks=8, plus negative, reversed, oversized
and invalid phase inputs, and zero-partition, negative-width and non-covering
plans. Full kernel, stack, numerical and backend qualification is not claimed by
these device-free contract tests.
