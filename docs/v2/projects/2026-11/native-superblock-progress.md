# Native CPU superblock progress repair

The host walk boundary requires `0 <= first <= last <= blocks <= INT_MAX`
and phase in `[0,7]`. For any entered iteration, `block < last <= blocks`.
The untruncated source-superblock end is strictly greater than block;
clipping it to blocks and last therefore still advances. Size_t intermediate
arithmetic is bounded by INT_MAX + 15. The resulting end fits int.

NEON prepared-projection ordered partitions derive bounds from partition index,
width and the weight row extent. Their starts now saturate at the row extent,
so excess partitions become explicit empty intervals instead of reversed ranges.
Ends and products use int64 before clipping. x86 single-scale ordered partitions
similarly saturate at the admitted tile end; requireSingleScaleTile already
bounds the partition product. Neither changes the partition count, ordered fold,
or any nonempty interval. The packed-prefill producers already clip starts:
NEON `p * per_partition` at its row extent and x86 `kpart * width` in int64.
The boundary remains authoritative for invalid caller-provided ranges.

The always-active unreachable assertion is noexcept at this boundary, making
invalid geometry fatal in Release as well as Integration; debug-only assertions
would leave the original production hang possible. No check occurs per block.

Focused tests exercise all phases, empty and ragged rows, INT_MAX, singleton and
odd partition counts, and overpartitioning. Fatal controls include the original
stuck range first=8,last=9,blocks=8, plus negative, reversed, oversized and invalid
phase inputs. Full kernel, stack, numerical and backend qualification is not
claimed by these device-free contract tests.
