# Idea 8 — ARM prepared-weight layout (dead bytes + expanded formats)

Status: **Open**. ABI-bearing change; last in dependency order, and only with
measurements behind it. Highest memory value on GB10 because CPU and GPU share
one UMA pool (see idea 2).

## Context

The NEON path reuses the x86 VNNI 64×32 block layout unchanged
(`CPUNativeVNNIPreparedFootprint.h`, `CPUNativeVNNIEncoding.h`,
`CPUNativeVNNIWeightView.h`):

1. **Dead compensation bytes.** The 128-byte `comp_int16` section exists for
   x86 `u8xs8` compensation (`sum((qa+128)*qw) - 128*sum(qw)`). ARM signed×signed
   SDOT computes `sum(qa*qw)` directly and the review found no NEON reader of
   the compensation section (Q6 dual-scale and CompactMultiScale carry their own
   metadata; `WeightView` arms it only for encodings that need it).
   Stride cost at the reviewed head: NibbleLUT 1280 B (10.0% dead),
   ExpandedInt8 2304 B (5.56% dead).
2. **Expanded formats.** Canonical mapping `CPUNativeVNNIEncoding.h:36-73`:
   - nibble codebooks 0/4/5 (4-bit, LUT),
   - IQ4_XS source-aware override,
   - IQ3_S source-scale expanded (~9.5 bpw vs 3.44 bpw source),
   - Q6_K native dual-scale,
   - CompactMultiScale families,
   - remaining supported tuples → ExpandedInt8 (9 bpw).
   Explicit rotation always yields ExpandedInt8.
   ARM `vqtbl` makes in-register nibble decode cheap, so bandwidth-bound decode
   would benefit from keeping IQ3_S/IQ5/expanded formats compact.

## Proposed shape (outline, not implementation)

A typed physical-encoding/layout identity on ARM, threaded through:
packer → view → footprint/BOM → prepared owners → cache keys → wire/slot/
movement → policy/evidence. Not a new parallel ledger; not merely a new archive
hash. Either:
- drop only the compensation section for encodings where no ARM reader exists
  (validated by mutation tests per encoding), or
- keep IQ3_S and other expanded formats in a compact ARM-native block.

## Exactness / risk

- Medium–high: layout is part of the prepared identity. Every consumer of the
  stride (packer encode/decode, readers, checksums, migrations, MPI wire
  codec, BOM pricing, replay fixtures) must agree; a partial update would mix
  layouts across ranks/models.
- Arithmetic per weight is unchanged; only physical bytes and strides change.

## Validation

- Device-free BOM/pricing tests for the new footprint; prepare→load→infer
  byte-auth for every format incl. MXFP4; migration/MPI wire compatibility runs;
  GB10 UMA admission with idea 2 to show real capacity gain. Unmeasured
  prepared-vs-GGUF resident ratios today: no complete ARM engine has loaded a
  real model yet.
