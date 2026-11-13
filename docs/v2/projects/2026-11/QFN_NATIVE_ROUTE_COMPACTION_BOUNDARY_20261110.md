# QFN native route compaction: arithmetic and capture boundary

The current two-MI50/two-CPU performance configuration measures 335.999
prefill and 40.798 after-prefill decode tokens/s. The target remains 500/40.
The [nine-slot qualification](QFN_HYBRID_NINE_SLOT_MIGRATION_20261110.md)
owns those numbers. This investigation installs no production transport or
arithmetic change.

## Why inspect the existing reduction

The continuation graph reduces a canonical FP32 route array containing both
owned results and zero placeholders for unowned routes. CPU-owned positions
are subsequently populated from their actual CPU owners. Native delivery of
only GPU-owned route rows could reduce traffic. The existing addressed,
device-counted NCCL/RCCL transfer API is the intended transport authority.

The source declaration amounts to 2.34375 GiB per participant for 48 layers,
512 tokens, top-10 routing and hidden width 2560. This is a declared extent,
not a measured wire-byte count. The retained root RCCL interval is about
150 ms, including communication, arithmetic and waiting. The peer's much
larger collective interval overlaps root-side CPU-result waits. Neither
interval establishes an equivalent removable latency.

An apportioned route has one useful GPU contribution. Replacing its reduction
with a byte transfer still has to preserve the arithmetic effects of adding
positive zero: signed zeros, subnormals and NaNs make a plain copy insufficient
as a universal equivalence argument. Tensor-sharded routes with multiple
partials are a different numerical problem and are outside this experiment.

## Actual compiled arithmetic witness

The original local-TP capture test executable was extended temporarily. It
compared the production rooted FP32 sum against the existing native addressed
byte transfer followed by the existing backend `vectorAddInplace` with a
positive-zero operand. Root-local ownership omitted the peer transfer.
Prepared-source admission, PhysicalMemoryAuthority, TransferEngine, retained
graph ownership, exact streams and terminal retirement remained the original
implementations. No test-specific GPU arithmetic kernel was added.

Both real devices served independently as root and owner. Widths 31, 32, 33,
1023, 1024 and 1025 each received 20 dirty replays with alternating endpoint
submission order. A 24-word raw-bit corpus included signed zeros, signed
minimum and maximum subnormals, normal values, infinities and distinct signed
quiet/signaling NaNs. Host code copied integer representations without
performing floating-point arithmetic. Checks covered the unchanged wire bank,
the normalized root result and inactive root guards. They did not certify
unobserved nonroot guards or arbitrary participant counts.

| Actual domain | Result | CTest elapsed |
|---|---|---:|
| Two MI50s, RCCL, coherent ROCm 10.1 SDK | Byte-exact pass | 5.238 s |
| Two RTX 3090s, NCCL | Proposed normalization rejected | 8.538 s |

All 42,236 CUDA mismatches were the four signed minimum/maximum subnormal
patterns. NCCL retained their bits; the existing CUDA vector-add path returned
positive zero. Other tested patterns, raw transfers and observed guards
passed. This is an incompatibility of the proposed replacement with the
original native arithmetic, not a failure of the existing production
reduction. A future CUDA compact consumer needs explicitly
subnormal-preserving normalization. Globally changing vector-add semantics or
weakening the witness would not qualify that replacement.

The temporary witness and its speculative preflight registrations were
removed after the experiment. Its patch, complete native receipts and
`native-results.json` remain under
`/tmp/qfn-native-compact-route-arithmetic-witness-next/`. A production
replacement must bring back the appropriate focused regression and qualify
the complete live-byte path, including empty/partial/reset execution.

## Remaining implementation boundary

The existing grouping publication could assign each original route slot a
compact wire ordinal. The down-projection leaf could write its unchanged
weighted value directly to that prefix, avoiding a separate bulk packing
pass. The root's canonical fold could consume indexed compact banks while
preserving its original top-k arithmetic order. The proven MI50 normalization
addresses only the one-owner arithmetic leaf of this design.

The unresolved boundary is captured dispatch sized to device-produced live
work. The current HIP graph interface has no conditional or indirect-grid
operation, and grouped-expert admission currently retains physical directory
grids with device guards. Copying that pattern, introducing a fixed CTA pool,
reading live counts back to the host or recapturing would violate the current
execution contract. The CPU ticket service runs alongside an already submitted
parent; it is not a general selector for arbitrary downstream live grids.

No small compliant production replacement has been established here. Keep
the original reduction until the dispatch boundary has a concrete design and
the complete path passes capture, byte-equivalence, live-payload and model
economy tests. The private source map is
`/tmp/qfn-native-compact-route-protocol-next/follow-up/NOTE.md`.
