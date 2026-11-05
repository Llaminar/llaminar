# MXFP4 native E8M0 scale plane (issue #26)

Status: implemented in PR #27, stacked on PR #25 (`feat/mxfp4-weights`).
Stages 1–4 are complete; stage 5 evidence is summarised in the PR body.

## Goal

Prepared MXFP4 (physical codebook 25) currently stores each 32-value block as
16 code bytes plus an FP16 scale `2^(e-128)`: 18 B/32, against 17 B/32 in the
GGUF file. It also accepts only the 40 exponents with an exact FP16 scale,
`e ∈ [104, 143]`. This work stores the original E8M0 byte in a 1-byte scale
plane (17 B/32). Kernels build the FP32 scale from bits, which widens the
accepted domain to `e ∈ [2, 252]`.

## Exponent contract

| Exponent byte | FP16 plane (today) | Native E8M0 plane |
|---|---|---|
| 104–143 | exact | exact, same FP32 scale bits |
| 2–103, 144–252 | rejected | exact normal FP32 `2^(e-128)` |
| 0–1 | rejected | rejected: FP32 subnormal scale, flushed by CUDA `--ftz=true` |
| 253–254 | rejected | rejected: the largest code (12) times the scale overflows FP32 |
| 255 | rejected | rejected: OCP MX NaN (ggml decodes it as finite 2^127) |
| any, all codes ±0 | scale +0 | scale byte 0 → FP32 +0 (ggml writes e = 0) |

`mxfp4::normalScaleFP32Bits(e) = (e - 1) << 23` is the single construction.
For `e ∈ [104, 143]` it is bit-identical to widening the current FP16 scale.
So the FP32 contribution tree, and with it the IQ4_NL arithmetic policy
(25 → 4) and its certified schedules, stays valid for every checkpoint that
loads today.

All-zero blocks need a scale that decodes to 0. E8M0 has no zero encoding, so
the prepared plane reserves byte 0. `e = 0` is rejected for nonzero blocks
anyway, so byte 0 means "scale 0" without ambiguity.

## Stages

Each stage is behaviour-preserving until the final switch. Every commit keeps
the CPU/CUDA/ROCm gates green.

1. **Contract** (done): `exponentFitsNativeE8M0`, `normalScaleFP32Bits` and
   `nativeVnniScalePlaneBytes(codebook)` (still 2 for every codebook). The
   packed-region sizes derive from the latter.
2. **Size authority:** route every `scales_bytes` producer and consumer through
   `nativeVnniScalePlaneBytes` / `nativeVnniPackedRegionSizes`. These are the
   pool, expert-transfer descriptors and slot directories, tier layout,
   rebalance geometry and BOM estimators; 88 references, mostly
   `GPUExpertTransfer*`, `CUDAMoEKernels.cu`, `ROCmMoEKernels.hip` and
   `WeightVRAMPool`. No literal `sizeof(uint16_t)` for the primary plane may
   remain.
3. **Scale loads:** one templated loader per backend,
   `weightScale<Codebook>(plane, linear)`, replaces direct `scales[linear]`
   FP16 widening in:
   - CPU AVX2/AVX-512 GEMV/GEMM and the window packer;
   - the CUDA decode body, GEMV, prefill and grouped MoE;
   - ROCm GEMV, the GEMM shard, grouped prefill and
     `ROCmNativeVNNIDecodeEquivalentMath.h`;
   - the Q8 residual visitor, the embedding decode and the expert-tier units.
   For every codebook the loader still widens FP16.
4. **Switch:** codebook 25 gets a 1-byte plane in one commit:
   - preparation writes the E8M0 byte (CPU packer, `packVnniBlock`, CUDA and
     ROCm repack);
   - loaders build FP32 bits;
   - the validation rule moves from `blockFitsFP16` to the native contract.
5. **Evidence:**
   - BOM and capacity tests at 17 B/32;
   - FP16-plane vs E8M0-plane equivalence for every `e ∈ [104, 143]`;
   - focused regressions for 2/103/104/143/144/252 and the rejected 0/1/253/255 (nonzero), plus the zero marker emitted only for all-zero payloads;
   - the all-format sweeps on all three backends;
   - paired Release GEMV, grouped MoE and one end-to-end model on CUDA and on
     ROCm gfx906;
   - spill-guard compiles on every shipped architecture.

## Constraints

- Admission owns validation. `e8m0Scale` / `decodePrimary` assume an admitted
  byte. Every producer (CPU packer, `packVnniBlock`, CUDA/ROCm repack, tier
  CPU→GPU inverse) rejects nonzero blocks outside `[2, 252]` and emits the zero
  marker only for all-zero payloads.
- The scale encoding is part of sealed preparation and wire identity. Every
  format that could carry an old FP16 codebook-25 record bumps its version,
  so such a record is never accepted as E8M0 or CPU-BF16 just because its
  codebook and strides match:
  - `ExpertTierWeightStreamManifest` v4;
  - remote projection ABI 5;
  - GPU packed weights v5;
  - CPU packed archive (`PACKED_WEIGHTS_VERSION`) 8.
- A normal finite scale bounds individual weights, not contributions. Dot
  products keep the ordinary IEEE FP32 contract.

- 25 remains a distinct physical decoder. It never joins 4/24 launch folding.
- A 1-byte plane changes vectorized scale loads and tile staging, so occupancy,
  VGPRs and spills must be re-measured per target. If the measured optimum
  moves, MXFP4 gets its own dispatch-policy entries.
- Staged scales (LDS and register slots) carry a normalized 16-bit word:
  the original FP16 bits, or for native 25 `powerOfTwoBF16Word(e)`, which
  `decodeStagedPrimary` places with `<< 16`. cp.async copies elements
  verbatim and cannot normalize a byte plane. In CUDA BK64 staging
  (`CUDANativeVNNIPrefillDevice.cuh`) the native-25 primary metadata is
  therefore loaded synchronously as eight scalar bytes and normalized directly
  into the shared staged-word slots. Inactive columns get 0. Payloads and the
  secondary plane keep their cp.async schedule, and compute reads the staged
  scales only after the existing CTA barrier. This is a schedule difference
  from the FP16 plane's 16-byte cp.async; the paired prefill measurements in
  the PR identify it. The plane-indexing loader is never applied to staged or
  CPU-unit slots.
- The expert-tier CPU unit keeps its 2-byte scale slot, offsets and stride.
  Only the separated GPU plane shrinks. Native 25 stores a BF16 power-of-two
  word in the CPU slot: `w = 0` for the zero marker, otherwise
  `w = (e - 1) << 7`. It widens as `float_bits(uint32(w) << 16)`, which equals
  `(e - 1) << 23` over the whole admitted range, with no BF16 arithmetic. GPU→CPU
  conversion encodes it. CPU→GPU validates a positive normal power of two with
  biased exponent 1..251 and stores `e = (w >> 7) + 1`. These words never go
  through F16C.
