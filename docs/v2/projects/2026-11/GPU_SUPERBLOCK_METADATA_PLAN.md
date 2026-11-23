# GPU superblock-coded NativeQuantised metadata

Status: CUDA implemented (2026-11-23); ROCm and CPU tier pending.

## Problem

Every GPU (CUDA/ROCm) NativeQuantised matrix stores, per 32-value block, a
payload plus up to three metadata planes: a primary FP16 word, a secondary
FP16 word and (Q2_K) a 32-bit packed effective-minimum word. GGUF superblock
formats store one FP16 `d` (and `dmin`) per 256 values plus a few bits of
scale code per 32 values. The per-32 FP16 planes therefore replicate or
pre-multiply information the file stores once, and the GPU footprint exceeds
the file:

| Format | File B/32 | GPU B/32 today | Ratio | Primary today | Secondary today |
|---|---|---|---|---|---|
| IQ2_XXS (15) | 8.25 | 10 | 1.21 | half(d(0.5+s)/4) | - |
| IQ2_XS (14) | 9.25 | 13 | 1.41 | half(d(0.5+lo)/4) | half(d(0.5+hi)/4) |
| IQ2_S (13) | 10.25 | 13 | 1.27 | d | coefficient byte (16-bit word) |
| IQ3_XXS (12) | 12.25 | 14 | 1.14 | half(d(0.5+s)/2) | - |
| IQ3_S (11) | 13.75 | 17 | 1.24 | d | coefficient nibble (16-bit word) |
| IQ1_S (16) | 6.25 | 10 | 1.60 | half(d(2s+1)) | half(dl*delta) |
| IQ1_M (17) | 7.00 | 10 | 1.43 | half(d(2lo+1)) | half(d(2hi+1)) |
| IQ4_XS (24) | 17.00 | 20 | 1.18 | d | coefficient (16-bit word) |
| Q2_K (10) | 10.50 | 16 | 1.52 | half(d*sc_lo) | half(d*sc_hi) + emins 4 B |
| Q3_K (9) | 13.75 | 16 | 1.16 | half(d(s_lo-32)) | half(d(s_hi-32)) |
| Q4_K (5) | 18.00 | 20 | 1.11 | half(d*sc) | half(-dmin*m) |
| Q5_K (7) | 22.00 | 24 | 1.09 | half(d*sc) | half(-dmin*m) |
| Q6_K (8) | 26.25 | 28 | 1.07 | d | two int8 coefficients |

For a ~100 GB low-bit MoE GGUF this costs well over 10 GB on the GPU, enough
to stop a model that fits a 128 GB integrated-GPU host from loading.
llama.cpp uploads GGUF blocks verbatim (1.00x).

## Design: superblock-coded planes (Tier 1)

A superblock physical encoding replaces the per-32 FP16 planes with:

- **Code plane** (the existing `scales` pointer slot): per 32-value block,
  exactly the source's scale-code bits, bit-packed (12 for Q3_K/Q4_K/Q5_K,
  16 for Q6_K/Q2_K, 8 for IQ2_S, 6 for IQ1_M/IQ4_XS, 4 for IQ3_S), plus four
  bytes of read padding. IQ3_XXS/IQ2_XS/IQ2_XXS/IQ1_S keep their code inside
  the payload (Tier 2) and have no code plane.
- **Superblock plane** (the existing `mins` pointer slot): per source
  superblock, FP16 `d` (and `dmin` for formats with minima).
- No emins plane.

Payload layouts are unchanged, so every decoder body is unchanged.

A single typed loader reconstructs, from (code, d, dmin), the **identical
staged words** the current planes hold: primary FP16 bits, secondary FP16 bits
and the Q2_K emin word. The reconstruction uses the repack's exact float
expression and `canonicalPreparedHalfBits`, so every downstream instruction,
reduction order and output byte is unchanged. The equality is a device-free
exhaustive proof (every code value x every `d`/`dmin` bit pattern) plus GPU
replay of real tensors (old planes vs reconstructed words).

### Addressing

Kernels address block metadata with `linear = blk * N + n` (block-major) or
`linear = n * KB + blk` (row-major `_rm` variants). The superblock index needs
the row coordinate and the source superblock phase of the prepared K origin
(`NativeQuantisedRepackColumns::firstBlock() % 8`; bounded expert column
slices can start mid-superblock):

- block-major: `sb = ((blk + phase) >> 3) * N + n`
- row-major: `sb = n * ceil((KB + phase) / 8) + ((blk + phase) >> 3)`

A typed `NativeQuantisedBlockAddress {linear, superblock}` built by one helper
per layout replaces every raw `linear`; the phase travels with the matrix
(descriptor field for grouped kernels, launch argument for dense kernels).

### Density

Payload + code + superblock bits per 256 values equal the GGUF bits for all
thirteen codable formats (Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_XS, IQ3_S,
IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M); the codec unit test asserts
this table. The only remaining excess is 256-byte region alignment and the
four padding bytes per code plane (well under 1 MB for a ~100 GB model).

Producers write codes with disjoint-bit `atomicAnd`/`atomicOr`, so concurrent
blocks need no zeroing and no ordering. A matrix bound at a sub-byte offset of
a coalesced run carries the bit shift in metadata-layout bits 3-5 (bit 7
coded, bits 0-2 superblock phase). Q4_K/Q5_K share physical codebooks 5/7 with
Q4_1/Q5_1 without conflict: the representation is per matrix and only
superblock sources are ever coded.

### Tier 2 (payload redundancy)

The remaining excess is duplicated code bits (IQ1_S/IQ1_M scale bits also live
in the payload `qh`) and the parity bit of each expanded sign mask
(IQ2_XXS/IQ2_XS/IQ3_XXS store 8-bit masks for 7-bit sign indices). Tier 2
reads IQ1 codes from the payload and stores the scale nibble in the freed
parity bits, reaching 1.000 for every format. It changes decoder bodies and is
landed for CUDA together with Tier 1.

### CPU tier and migration

CPU compact units (`CompactMultiScale`) also store per-32 FP16 products
(e.g. both IQ2_XS half scales), not `d` and codes. CPU-to-GPU expert promotion
cannot recover `d` and the codes from those products, so a superblock-coded
GPU format requires the CPU unit to carry the same code + superblock data. The
CPU compact representation therefore adopts the same codec (reconstructing the
FP16 words in the NEON/AVX loaders), which also removes the measured CPU
prepared-weight inflation (1.33-1.66x of file for the UD variants). Migration
then copies codes and superblock words in both directions; no derived value is
ever inverted.

## Work inventory (Tier 1)

Producers:
- CPU compact multi-scale packer/writer and CPU kernels' scale loads (NEON,
  AVX2, AVX-512), so CPU units carry codes + superblock words.
- CUDA/ROCm VNNI repack kernels (13 formats), dense and grouped destinations,
  column slices (phase).
- CPU-to-GPU expert-tier conversion (`CUDAExpertTierWeightKernels.cu`,
  `ROCmExpertTierWeightKernels.hip`) and GPU-to-CPU inverse; stream manifests
  (`ExpertTierWeightStream`, `ExpertTierWeightDeviceLayout`), remote projection
  manifests, reusable slot capacity union
  (`NativeQuantisedReusableDeviceAllocationFormat`).
- Unpack kernels (`CUDAVnniUnpackKernels.cu`, `VnniUnpackKernels.hip`).
- Persisted GPU packed-weight cache format (`GpuPackedWeightsFormat.h`).

Consumers (staged-word reads; 40 primary loader sites plus secondary/emin
reads): CUDA GEMV shards, prefill device/shards, MoE grouped prefill, MoE
kernels, Q8 residual (grouped and native), retained HC-FFN readers (ROCm),
ROCm GEMV/GEMM/WMMA/verifier/population projections.

Accounting: `nativeQuantisedPackedRegionSizes`, `NativeQuantisedMetadataPlaneBytes`,
`WeightVRAMPool::planWeight`, `WeightMemoryEstimator::exactGpuPackedMatrixBytes`,
expert slot geometry, PMA BOM.

## Gates

1. Device-free: exhaustive code x d (x dmin) reconstruction equality per format
   against the repack expression; address helpers for every layout, phase and
   ragged K; region-size accounting.
2. Native replay (CUDA and ROCm suites, `ProductionTestPreflight`): repack real
   tensors both ways and compare reconstructed staged words with the legacy
   planes; byte-exact GEMV/prefill/MoE/verifier outputs across all formats,
   M sweep, column slices with nonzero phase, tier migration round trips.
3. Spill guard on every shipping architecture; ncu/rocprof of each changed
   kernel family (loader adds a few ALU ops per block; no throughput loss
   accepted without evidence).
4. Real low-bit MoE GGUFs: dry-run admission at file density on an
   integrated GPU, then load and generate.

## Status

- CUDA: all thirteen formats coded at file density (readers, repack, async
  prefill staging, row-major diagnostic copies); migration slots, expert
  transfer and rebalance reject coded descriptors.
- ROCm: readers and repack reject coded layouts, so ROCm keeps per-block
  planes. Conversion needs a HIP compile host.
- CPU compact tier: still per-block FP16 words (see below). The
  NEON/AVX2/AVX-512 loaders need the codec before coded CPU-to-GPU promotion
  can exist.

## CPU tier and expert movement (next phase)

Measured per 32 values (bytes), from the production stride/allocation helpers:

| Format | GGUF | CPU unit | x | GPU migration slot | x |
|---|---|---|---|---|---|
| IQ1_S | 6.25 | 12 | 1.92 | 10 | 1.60 |
| IQ1_M | 7.00 | 12 | 1.71 | 10 | 1.43 |
| IQ2_XXS | 8.25 | 12 | 1.45 | 10 | 1.21 |
| IQ2_XS | 9.25 | 16 | 1.73 | 13 | 1.41 |
| IQ2_S | 10.25 | 16 | 1.56 | 13 | 1.27 |
| IQ3_XXS | 12.25 | 16 | 1.31 | 14 | 1.14 |
| IQ3_S | 13.75 | 38 | 2.76 | 17 | 1.24 |
| Q2_K | 10.50 | 16 | 1.52 | 16 | 1.52 |
| Q3_K | 13.75 | 16 | 1.16 | 16 | 1.16 |
| Q4_K | 18.00 | 22 | 1.22 | 20 | 1.11 |
| Q5_K | 22.00 | 38 | 1.73 | 36 | 1.64 |
| Q6_K | 26.25 | 28 | 1.07 | 28 | 1.07 |
| IQ4_XS | 17.00 | 22 | 1.29 | 20 | 1.18 |
| MXFP4 | 17.00 | 20 | 1.18 | 17 | 1.00 |
| Q8_0 | 34.00 | 36 | 1.06 | 34 | 1.00 |

CPU-to-GPU and GPU-to-CPU expert streams carry CPU units, so transfer
volume has the CPU overhead too. Sources of the excess: per-block FP16 scale
products instead of codes + superblock d; payloads padded to whole 32-bit
words per column (IQ2_XS/IQ2_S 9 -> 12 bytes); expanded signed-INT8 for
IQ3_S/Q5_K/Q5_0/Q5_1; per-block planes in reusable GPU slots.

Target: CPU units carry the source payload bytes (byte-interleaved) and the
same codes + superblock words as CUDA; NEON/AVX2/AVX-512 loaders reconstruct
the identical FP16 words with the shared codec; GPU migration slots become
coded; streams copy codes and superblock words in both directions. Gates:
byte-identical CPU outputs per ISA, captured CPU<->GPU round trips, wire
extents equal the file bytes.

## Sequencing

1. Codec header + exhaustive device-free proofs (all formats).
2. Address type + helpers; descriptor phase field.
3. Q4_K/Q5_K/IQ4_XS after the IQ formats (done: the per-matrix
   representation lets them share codebooks 5/7 with Q4_1/Q5_1).
4. Format by format in descending waste order: IQ2_S, IQ3_XXS, IQ2_XS,
   IQ1_S/IQ1_M, IQ3_S, Q6_K, IQ2_XXS, then K-quants and IQ4_XS. Each format
   switches every producer and consumer on both backends at once; the legacy
   planes for that format are retired in the same change.
5. Tier 2.
