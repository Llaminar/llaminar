# Grouped verifier attention: lane-exact gfx1100 WMMA

This is an implementation and proof record, not a tuning dashboard. The current
candidate is explicit-only: Automatic still selects the ordered decode producer.
The ordinary M512 prefill attention result does not qualify grouped MTP decode.

## Arithmetic and ownership

The FP16/HD256 candidate groups four physical queries in a 1024-thread CTA. It
retains each row's original device-selected split range, ring mapping and one to
four logical wavefronts. Sparse matrix products reproduce both half-dot updates
for each of the original 64 lanes. All lane partials enter the unchanged XOR
score reduction, online softmax and FP32 probability/value accumulation. Lane0
publishes each wave's maxima/sum; output lanes retain their own arithmetic.
Ascending wave merge and the original split reducer preserve serial decode.

Keeping all lane scores matters: prior shared-root experiments preserved finite
scores but changed NaN payloads that survived into final output. The new producer
stores every lane score, then reuses the flat LDS bank for wave output only after
a CTA barrier retires every score reader. Tail rows and surplus splits own no
output; empty participating splits publish neutral bytes. Production inactive
physical rows receive the canonical one-key sentinel and are actually computed.

Equal split starts/ring mappings with four original waves share a 16-key tile.
Otherwise a union of each row/wave's next key retains its original traversal.
This preserves unequal row lengths and split boundaries without host metadata,
additional persistent workspace, allocations or recapture. Explicit unsupported
native requests fail admission; Automatic does not enter the candidate yet.

## Captured production proof

`V2_Integration_ROCmVerifierAttentionWmmaReplay` is explicitly registered in
ProductionTestPreflight and is run as a focused feature test. It covers:

- Every M2–16 at physical KV capacities64 and8193, changing query data, ring
  wrap, full/partial/zero-live publications, regrowth and physical tile tails.
- Ordinary and replicated GQA, plus a distinct heads16/KV2 case that actually
  addresses both cache heads.
- Native exceptional half inputs, signed zero, infinities, subnormals and NaN
  payloads. Every partial-output/max/sum bank and final output matches ordered
  grouped attention; every physical row also matches independent serial M=1.
- The real KVCacheAppendStage and AttentionComputeStage with RoPE-on-read,
  M2/3/4/8/9/16 and histories30/513. One captured transaction replays three times,
  restoring a device sequence-state checkpoint before repeated append. Poisoned
  outputs match between replays and match independent serial append/decode.
- Native graph symbols, launch geometry, zero private scratch, actual producer
  input/parameter/partial pointers, and original reducer input/output bindings.

Initial finite/special proof:
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/mtp-attention-wmma-focused-1794763017905034631/receipt.json`.
Append-stage proof:
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/mtp-attention-wmma-focused-1794763296895051092/receipt.json`.
The four-test proof after the control and NaN-order repairs below passes:
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/mtp-attention-wmma-focused-1794764899657573496/receipt.json`.
The expanded five-test proof then passes in
`mtp-attention-wmma-focused-1794765521990305025/receipt.json`.
These are local source/binary-bound evidence records, not published corpus data.

## Independent thread override regression

Both existing ordinary and grouped launch bridges incorrectly required the
split-count environment variable to be present before honoring a TPB override.
The fix makes TPB presence independent in both bridges. Default selection and
all device partition formulas remain unchanged.

The regression truly unsets the split variable; an empty value would hide the
bug. At KV8 it uses Q=K=0 and V channel0 values
`[65504, 2^-10, -65504, 0, -65504, 0, 65504, 0]`. Original one/three-wave
arithmetic produces+0, while two/four-wave arithmetic produces2^-13. Captured
TPB arguments and the exact result therefore detect an ignored override even
when all compared implementations share it. Tests include default selection,
TPB-only64/128/192/256, the legacy empty-split-variable control, serial M=1,
and exceptional-value grouped replay under reduced wave counts.

The original-code negative run fails exactly these captured arguments and
arithmetic expectations:
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/mtp-attention-wmma-focused-1794763701893480720/receipt.json`.
Its `negative-source/` directory retains source bytes authenticated against that
receipt; binary digests and the full failing log remain in the receipt.

## NaN operand-order regression

The one-wave special-value replay exposed an online PV mismatch that the
four-wave merge had masked. LLVM commuted the two multiplicands in the C++ FMA:
both the previous output and rescaling factor can contain different NaN
payloads, so that exchange changes the selected payload. Explicit
`v_fma_f32(scale_old, old_output, rounded_probability_times_value)` preserves
the original scalar instruction's operand roles. Rebuilt ISA also confirms
that the preceding mixed-precision product retains half-V first and probability
second. Finite arithmetic, wave merging and the original split reducer remain
unchanged.

The failing run is `mtp-attention-wmma-focused-1794764531296588412`; the repaired
four-test run is `mtp-attention-wmma-focused-1794764899657573496`, both under the
same evidence root. The expanded `AllWaveSpecialValuesDirtyReplay` covers
TPB64/128/192/256 at M2/4/9/16. It and the independent TPB-control witness now
have a separate explicit preflight registration,
`V2_Integration_ROCmVerifierAttentionControlAndPayloadReplay`.

## Complete-operation economy

`Perf__ROCmVerifierAttentionWmma.cpp` captures parameter publication, the grouped
producer and the original reducer as one component. Internal ordered-timeline
events bracket eight operations per sample. Each AB/BA pair has independent
poisoned output banks and an ordinary serial oracle, observed before another
pair can repair the result. Profiling records one operation separately.

The initial panel fixes M2/4/8/9/16 and heads8/KV1 or heads16/KV2, retaining a64K
capacity while live KV grows from64 through65536. Each key has31 paired samples.
`V2_Integration_ROCmVerifierAttentionTimingReplay` separately tests repeated
timing and dirty reset/regrowth and is registered in ProductionTestPreflight.
The unpadded generation wins zero of90 cells, with geometric-mean Dot2/WMMA
ratio0.389956. At M16/heads16/KV2/64K the complete operation takes25.93ms versus
9.28ms for ordered Dot2. The exact producer's LDS-bank-conflict metric is54.37%.
Its timing, loaded binary, source and profiles are retained in
`mtp-attention-wmma-unpadded-snapshot/snapshot.json`.

Padding each64-float score plane to65 rotates the transposed lane reads across
LDS banks. Combined with the NaN-order repair, this generation completes all90
cells and2,790 paired samples but still wins none. Its geometric-mean ratio is
0.511611, range0.176974–0.838610. M16/heads16/KV2/64K improves to16.32ms versus
9.28ms for Dot2. The measured bank-conflict metric falls to16.19%. Resources are
61 logical VGPRs,19,008 LDS bytes and zero private scratch. This is a useful
candidate improvement, not an economical production selection.

The source-bound timing is
`mtp-attention-wmma-timing-1794764937179295282/receipt.json`.
Across all ten fixed query/head geometries, doubling live KV from32K to64K
increases ordered latency by1.9545–1.9904× and WMMA by1.9549–2.0034×. Both scale
linearly in this range; the unresolved issue is absolute WMMA performance.

The score-padded generation has eight exact producer counter passes, including
45.12% occupancy,64.64% memory-unit busy and82.16% L2 hit rate. Its complete
binary/source/timing/profile snapshot is
`mtp-attention-wmma-score-padded-snapshot/snapshot.json`.

Padding the four query rows from256 to258 halves removes the remaining measured
LDS conflicts (0.0%), with45.14% occupancy and unchanged61 VGPRs. This gives only
a0.8% median native-latency improvement across the90 cells; none beats Dot2.
The complete-operation geometric-mean ratio is0.515856. Timing is
`mtp-attention-wmma-timing-1794765790534916090/receipt.json`; the exact generation
and both profiler passes are preserved in
`mtp-attention-wmma-query-padded-snapshot/snapshot.json`.

All five affected feature registrations pass27 GTests for the query-padded
generation: focused WMMA replay, control/NaN-payload replay, timing replay,
captured append/RoPE, and native decode dimensions/ring cursors across formats.
See `mtp-attention-wmma-feature-1794765634125575482/validation.json`. Its original
collector incorrectly expected17 append cases; canonical executable discovery
shows20 and all20 passed. The separate validation retains the original failed
collector assertion, source/binary hashes, full passing native log and discovery
output. No native test failed, and no result was replaced by a rerun.

Release and Integration builds cover both shipped gfx906/gfx1100 targets with
spill guards enabled. The gfx906 native body is unavailable and admission rejects it.

## Cooperative key loading and the remaining layout limit

The current candidate stages each16-key tile with contiguous half2 global loads,
one wave per key, before the sparse matrix consumers read padded LDS. It retains
key bits and arithmetic but adds8,256 shared bytes and one barrier per tile.
All five focused replay tests pass in
`mtp-attention-wmma-focused-1794766108801094108/receipt.json`.
The final five-registration feature run also passes all27 GTests in
`mtp-attention-wmma-feature-1794766401171684939/receipt.json`, with unchanged
source/binary hashes and complete process retirement. These are the same
focused and affected feature gates listed above; full preflight was not run.

The complete90-cell/2,790-sample panel improves native latency by a geometric
mean1.13979× over query padding alone. It still wins zero cells against Dot2:
Dot2/WMMA range0.221431–0.850804, geometric mean0.587804. At
M16/heads16/KV2/64K, native latency is14.85ms versus9.29ms for Dot2. Doubling
KV32K→64K costs1.9545–1.9881× for Dot2 and1.9496–1.9963× for WMMA.
Timing is `mtp-attention-wmma-timing-1794766145362564478/receipt.json`.

Eight exact producer profiles pass. The native kernel uses58 logical VGPRs,
27,280 LDS bytes and zero private scratch/spills. Measured occupancy rises to
66.92%; LDS conflicts remain0.0%, memory-unit busy is62.95%, and L2 hit rate is
81.74%. The complete generation is preserved in
`mtp-attention-wmma-key-staged-snapshot/snapshot.json`.

The four-query sparse layout retains only4/16 matrix output rows and uses4/16
reduction coordinates:1/16 of the dense matrix MACs contribute in the common
key case. The union mapping consumes fewer cross-query outputs. These are
arithmetic-count ratios, not measured hardware utilization or an impossibility
proof. They explain why further padding is not the next experiment.

A reviewed structural candidate would group two query rows with eight heads
sharing one KV head and consume all four WMMA result registers. It is not yet
implemented or qualified. Its64KiB score bank must be swizzled rather than
padded; metadata and Q operands must remain in registers. Four independent
logical wave states per physical wave must preserve every NaN payload and
original key order. The score→max/sum→scale→output scratch lifecycle needs a
barrier after all owner waves snapshot their scales, before any output writes.
Head grouping must authenticate local and replicated GQA ownership, and full
fragment mapping must be proven before integration. Register pressure and
64KiB occupancy are explicit risks. Automatic remains Dot2 until a complete
operation wins with its byte and resource contracts intact.
