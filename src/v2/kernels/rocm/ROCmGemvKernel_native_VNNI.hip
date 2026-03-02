/**
 * @file ROCmGemvKernel_native_VNNI.hip
 * @brief Native-VNNI GEMV kernels for lossless sub-8-bit decode on AMD MI50/MI60 (gfx906)
 *
 * This file contains native-VNNI kernels that decode quantized weights
 * losslessly and compute GEMV using v_dot4_i32_i8 with per-block FP16 scales.
 *
 * Supported formats:
 *   Q4_0:   w_int8 = q - 8          (exact, symmetric)
 *   IQ4_NL: w_int8 = lut[q]         (exact, LUT-based symmetric)
 *   IQ4_XS: w_int8 = lut[q]         (reuses IQ4_NL kernel, sub-block scales precomputed)
 *   Q4_1:   w_int8 = q, min corr.   (asymmetric, scale + min)
 *   Q4_K:   w_int8 = q, min corr.   (reuses Q4_1 kernel, sub-block scales precomputed)
 *   Q5_0:   w_int8 = 5bit - 16      (exact, symmetric with high-bit array)
 *   Q5_1:   w_int8 = 5bit, min corr.(asymmetric, scale + min + high-bit array)
 *   Q5_K:   w_int8 = 5bit, min corr.(reuses Q5_1 kernel, sub-block scales precomputed)
 *   Q6_K:   w_int8 = 6bit - 32      (symmetric, dual-scale per 32-element block)
 *   Q3_K:   w_int8 = 3bit - 4       (symmetric, dual-scale per 32-element block)
 *   Q2_K:   w_int8 = 2bit (unsigned)(asymmetric, dual-scale + embedded mins)
 *   IQ3_S:  grid LUT → signed INT8  (9-bit grid index + direct sign bytes)
 *   IQ3_XXS:grid LUT → signed INT8  (8-bit grid index + pre-resolved signs)
 *   IQ2_S:  grid LUT → signed INT8  (10-bit grid index + direct signs, dual-scale)
 *   IQ2_XS: grid LUT → signed INT8  (9-bit grid index + pre-resolved signs, dual-scale)
 *   IQ2_XXS:grid LUT → signed INT8  (8-bit grid index + pre-resolved signs)
 *
 * Unlike ratio-VNNI (which bakes scales into INT8 weights with truncation),
 * native-VNNI keeps per-block FP16 scales separate and decodes nibbles
 * directly to INT8.
 *
 * Split from ROCmGemvKernel.hip.
 *
 * @see ROCmGemvKernel_INT8_VNNI.hip for INT8 VNNI kernels
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "../../backends/rocm/HipDeviceGuard.h"

constexpr uint8_t CODEBOOK_IQ4 = 4;

__device__ __constant__ int8_t k_iq4_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

// ============================================================================
// Device-constant IQ grid lookup tables (initialized at runtime via
// rocmInitIQGridTables before first IQ kernel launch).
// ============================================================================
__device__ __constant__ uint32_t d_iq3s_grid[512];    // IQ3_S:   512 × 4B = 2KB
__device__ __constant__ uint32_t d_iq3xxs_grid[256];  // IQ3_XXS: 256 × 4B = 1KB
__device__ __constant__ uint64_t d_iq2s_grid[1024];   // IQ2_S:   1024 × 8B = 8KB
__device__ __constant__ uint64_t d_iq2xs_grid[512];   // IQ2_XS:  512 × 8B = 4KB
__device__ __constant__ uint64_t d_iq2xxs_grid[256];  // IQ2_XXS: 256 × 8B = 2KB
__device__ __constant__ uint64_t d_iq1s_grid[2048];   // IQ1_S/M: 2048 × 8B = 16KB


// ============================================================================
// Format tags for native-VNNI kernel dispatch
//
// Each value is the codebook_id stored in ROCmPackedWeights::native_vnni_codebook_id
// and passed to rocmGemv_native_vnni_fp32() for template selection.
// ============================================================================
enum NativeVNNIFormat : uint8_t {
    NVNNI_Q4_0   = 0,   // 4-bit symmetric, linear decode (q - 8)
    NVNNI_IQ4_NL = 4,   // 4-bit symmetric, LUT decode
    NVNNI_Q4_1   = 5,   // 4-bit asymmetric (q as-is, has min)
    NVNNI_Q5_0   = 6,   // 5-bit symmetric, linear decode (q - 16)
    NVNNI_Q5_1   = 7,   // 5-bit asymmetric (q as-is, has min)
    NVNNI_Q6_K   = 8,   // 6-bit symmetric, dual-scale per 32-element block
    NVNNI_Q3_K   = 9,   // 3-bit symmetric, dual-scale per 32-element block
    NVNNI_Q2_K   = 10,  // 2-bit asymmetric, dual-scale + embedded mins
    NVNNI_IQ3_S  = 11,  // IQ3 grid (4 elem/lookup), 9-bit index, direct signs
    NVNNI_IQ3_XXS= 12,  // IQ3 grid (4 elem/lookup), 8-bit index, pre-resolved signs
    NVNNI_IQ2_S  = 13,  // IQ2 grid (8 elem/lookup), 10-bit index, direct signs, dual-scale
    NVNNI_IQ2_XS = 14,  // IQ2 grid (8 elem/lookup), 9-bit index, pre-resolved signs, dual-scale
    NVNNI_IQ2_XXS= 15,  // IQ2 grid (8 elem/lookup), 8-bit index, pre-resolved signs
    NVNNI_IQ1_S  = 16,  // IQ1 grid (8 elem/lookup), 11-bit index, signed values {-1,0,+1}, delta correction
    NVNNI_IQ1_M  = 17,  // IQ1 grid (8 elem/lookup), 11-bit index, signed values {-1,0,+1}, dual-scale + delta
};

// Compile-time format properties
template <NativeVNNIFormat FMT>
struct NVNNITraits {
    // Asymmetric formats have a min correction term: f_acc += sum_a * block_m
    static constexpr bool is_asymmetric = (FMT == NVNNI_Q4_1 || FMT == NVNNI_Q5_1 || FMT == NVNNI_IQ1_S);
    // --- IQ grid format traits ---
    static constexpr bool is_iq3_grid = (FMT == NVNNI_IQ3_S || FMT == NVNNI_IQ3_XXS);
    static constexpr bool is_iq2_grid = (FMT == NVNNI_IQ2_S || FMT == NVNNI_IQ2_XS || FMT == NVNNI_IQ2_XXS);
    static constexpr bool is_iq1_grid = (FMT == NVNNI_IQ1_S || FMT == NVNNI_IQ1_M);
    static constexpr bool is_iq_grid  = (is_iq3_grid || is_iq2_grid);
    // High bits per grid lookup stored in qh byte (0, 1, or 2)
    static constexpr int qh_bits_per_lookup =
        (FMT == NVNNI_IQ2_S) ? 2 : (FMT == NVNNI_IQ3_S || FMT == NVNNI_IQ2_XS) ? 1 : 0;
    static constexpr bool has_qh = (qh_bits_per_lookup > 0);
    // Dual-scale: two FP16 scales per 32-element block (stored in scales + mins arrays)
    // Elements 0-15 use scale_lo (scales[]), elements 16-31 use scale_hi (mins[])
    static constexpr bool is_dual_scale = (FMT == NVNNI_Q6_K || FMT == NVNNI_Q3_K || FMT == NVNNI_Q2_K
                                        || FMT == NVNNI_IQ2_S || FMT == NVNNI_IQ2_XS
                                        || FMT == NVNNI_IQ1_M);
    // Dual-scale + asymmetric: embedded FP16 mins in payload for split sum_a correction
    static constexpr bool is_dual_scale_asym = (FMT == NVNNI_Q2_K || FMT == NVNNI_IQ1_M);
    // Offset of embedded FP16 min_lo/min_hi within payload (only for dual_scale_asym)
    static constexpr int embedded_min_offset = (FMT == NVNNI_IQ1_M) ? 6 : 8;
    static constexpr bool is_5bit       = (FMT == NVNNI_Q5_0 || FMT == NVNNI_Q5_1);
    static constexpr bool is_6bit       = (FMT == NVNNI_Q6_K);
    static constexpr bool is_3bit       = (FMT == NVNNI_Q3_K);
    static constexpr bool is_2bit       = (FMT == NVNNI_Q2_K);
    static constexpr bool uses_lut      = (FMT == NVNNI_IQ4_NL);
    static constexpr int  payload_bytes =
        is_iq1_grid ? ((FMT == NVNNI_IQ1_M) ? 10 : 6) :
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : (is_3bit || is_2bit) ? 12 : 16;
    // Centering offset for symmetric formats; asymmetric formats don't center
    static constexpr int center_4bit = (FMT == NVNNI_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NVNNI_Q5_0) ? 16 : 0;
};

// ============================================================================
// IQ grid decode helpers
// ============================================================================

// Conditionally negate 4 unsigned bytes packed in uint32_t using 4 sign bits.
// sign_lo4 bit j → element j: 0=positive, 1=negative.
// Safe: all IQ grid values > 0, so XOR+add never carries across bytes.
__device__ inline uint32_t iq_apply_signs_4(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t mask =
        (static_cast<uint32_t>(-(sign_lo4 & 1)) & 0xFFu) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 1) & 1)) & 0xFFu) << 8) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 2) & 1)) & 0xFFu) << 16) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 3) & 1)) & 0xFFu) << 24);
    uint32_t carry = mask & 0x01010101u;
    return (grid4 ^ mask) + carry;
}

// Format-specific IQ3 grid lookup (resolved at compile time)
template <NativeVNNIFormat FMT>
__device__ inline uint32_t iq3_grid_lookup(int idx)
{
    if constexpr (FMT == NVNNI_IQ3_S) return d_iq3s_grid[idx];
    else                               return d_iq3xxs_grid[idx];
}

// Format-specific IQ2 grid lookup (resolved at compile time)
template <NativeVNNIFormat FMT>
__device__ inline uint64_t iq2_grid_lookup(int idx)
{
    if constexpr (FMT == NVNNI_IQ2_S)       return d_iq2s_grid[idx];
    else if constexpr (FMT == NVNNI_IQ2_XS) return d_iq2xs_grid[idx];
    else                                    return d_iq2xxs_grid[idx];
}

// IQ1 grid lookup: shared table for both IQ1_S and IQ1_M
// Returns uint64_t = 8 signed int8 values in {-1, 0, +1}
__device__ inline uint64_t iq1_grid_lookup(int idx)
{
    return d_iq1s_grid[idx];
}


// ============================================================================
// Native-VNNI GEMV kernel: lossless decode with per-block FP16 scales
//
// Supports Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL via compile-time NativeVNNIFormat tag.
//
// Symmetric formats (Q4_0, Q5_0, IQ4_NL):
//   int32_block = Σ sdot4(a_packed, w_packed)   // 8 sdot4 calls
//   f_acc += (float)int32_block * block_d
//
// Asymmetric formats (Q4_1, Q5_1) — min correction via activation sum:
//   f_acc += (float)int32_block * block_d + (float)sum_a * block_m
//   where sum_a = Σ a_i computed via sdot4 with ones vector
//
// Output is FP32 with scale_A applied inline — no epilogue kernel needed.
// ============================================================================
template <int TILE_N, int CPT, NativeVNNIFormat FMT>
__global__ void gemv_native_vnni_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations (INT8)
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × payload_bytes] payload
    const uint16_t*  __restrict__ d_block_scales,  // [blocks_per_row × N] FP16 scale (d)
    const uint16_t*  __restrict__ d_block_mins,    // [blocks_per_row × N] FP16 min (m), NULL for symmetric
    float*           __restrict__ d_C_fp32,        // [N] output (FP32), used when kblocks=1
    float*           __restrict__ d_partial_fp32,  // [KB × N] partials, used when kblocks>1
    const float*     __restrict__ d_scale_A,       // [1] activation scale
    int N, int K, int kblocks)
{
    using Traits = NVNNITraits<FMT>;
    static_assert(CPT == 1 || CPT == 2, "native-VNNI supports CPT in {1,2}");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
        f_acc[c] = 0.0f;

    if (active)
    {
        for (int b = b_start; b < b_end; ++b)
        {
            // A pointer for this block's 32 INT8 activation values
            const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);

            // Compute sum of 32 activation values once per block.
            // Needed for: asymmetric formats (min correction) and
            //             bias-fold formats (deferred centering correction).
            int32_t sum_a = 0;
            if constexpr (Traits::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;  // {1, 1, 1, 1} as packed INT8
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);
            }

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                    const uint8_t* payload = d_payload + linear * Traits::payload_bytes;
                    const uint16_t scale_bits = d_block_scales[linear];

                    // Decode 32 quantized values to 8 packed INT32 words (4 INT8 each)
                    // Nibble layout: qs[j] = lo_nibble(elem_j) | hi_nibble(elem_{j+16})
                    // Groups 0-3: elements [0..15] from low nibbles
                    // Groups 4-7: elements [16..31] from high nibbles
                    int32_t packed_groups[8];

                    if constexpr (Traits::uses_lut)
                    {
                        // IQ4_NL / IQ4_XS: nibble → codebook INT8 via register-based v_perm_b32
                        //
                        // The 16-entry codebook {-127..113} is packed into 4 constexpr registers.
                        // v_perm_b32(src0, src1, sel) selects 4 bytes from {src1[3:0], src0[3:0]}
                        // using sel[i] & 0x7 for each output byte i.
                        //
                        // Two v_perm calls cover indices 0-7 and 8-15. A branchless byte-wise
                        // mux selects per nibble based on bit 3. This eliminates 32 data-dependent
                        // constant-memory lookups per block, replacing them with pure ALU.
                        //
                        // Codebook (uint8 representation):
                        //   0-3:  {0x81, 0x98, 0xAD, 0xBF}  = {-127, -104, -83, -65}
                        //   4-7:  {0xCF, 0xDD, 0xEA, 0xF6}  = {-49, -35, -22, -10}
                        //   8-11: {0x01, 0x0D, 0x19, 0x26}  = {1, 13, 25, 38}
                        //   12-15:{0x35, 0x45, 0x59, 0x71}  = {53, 69, 89, 113}
                        constexpr uint32_t CB_LO_A = 0xBFAD9881u;  // cb[0..3] (src0 for nibbles 0-7)
                        constexpr uint32_t CB_LO_B = 0xF6EADDCFu;  // cb[4..7] (src1 for nibbles 0-7)
                        constexpr uint32_t CB_HI_A = 0x26190D01u;  // cb[8..11] (src0 for nibbles 8-15)
                        constexpr uint32_t CB_HI_B = 0x71594535u;  // cb[12..15] (src1 for nibbles 8-15)

                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            // Load 4 payload bytes as a packed uint32
                            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

                            // --- Low nibbles (elements [4g..4g+3]) ---
                            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                            {
                                const uint32_t sel = lo_nibs & 0x07070707u;  // low 3 bits per byte
                                const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_A, CB_LO_B, sel);
                                const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_A, CB_HI_B, sel);
                                // Byte-wise mux: select r_hi where nibble >= 8 (bit 3 set)
                                uint32_t m = (lo_nibs >> 3) & 0x01010101u;  // 0x01 or 0x00 per byte
                                m |= (m << 1);   // 0x03
                                m |= (m << 2);   // 0x0F
                                m |= (m << 4);   // 0xFF
                                packed_groups[g] = static_cast<int32_t>((r_hi & m) | (r_lo & ~m));
                            }

                            // --- High nibbles (elements [4g+16..4g+19]) ---
                            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                            {
                                const uint32_t sel = hi_nibs & 0x07070707u;
                                const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_A, CB_LO_B, sel);
                                const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_A, CB_HI_B, sel);
                                uint32_t m = (hi_nibs >> 3) & 0x01010101u;
                                m |= (m << 1);
                                m |= (m << 2);
                                m |= (m << 4);
                                packed_groups[g + 4] = static_cast<int32_t>((r_hi & m) | (r_lo & ~m));
                            }
                        }
                    }
                    else if constexpr (Traits::is_5bit)
                    {
                        // Q5_0 / Q5_1: 5-bit decode with high-bit array
                        // Payload layout: [qs[0..15], qh[0..3]] = 20 bytes
                        // 5th bit for element j: (qh_bits >> j) & 1
                        // Q5_0: val = (nib | (hbit<<4)) - 16  → [-16, +15]
                        // Q5_1: val = (nib | (hbit<<4))        → [0, 31]
                        const uint32_t qh_bits = *reinterpret_cast<const uint32_t*>(payload + 16);

                        if constexpr (Traits::center_5bit == 0)
                        {
                            // Q5_1: val = nib | (hbit<<4) (unsigned, no centering)
                            // SWAR: extract nibbles, deposit hbits at bit 4 of each byte
                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                            {
                                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

                                // Low nibbles + hbits
                                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                                     | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                                packed_groups[g] = static_cast<int32_t>(lo_nibs | hb_lo);

                                // High nibbles + hbits
                                const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                                const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
                                const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                                     | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
                                packed_groups[g + 4] = static_cast<int32_t>(hi_nibs | hb_hi);
                            }
                        }
                        else
                        {
                            // Q5_0: val = (nib | (hbit<<4)) - 16 (centered)
                            // Same SWAR as Q5_1 for nibble+hbit assembly, then
                            // byte-parallel subtract-16 via XOR-SUB-XOR bias trick:
                            //   val ^= 0x80 (bias to [128,159] → safe from borrow)
                            //   val -= 0x10 (subtract 16 without inter-byte borrow)
                            //   val ^= 0x80 (remove bias → correct int8 encoding)
                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                            {
                                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

                                // Low nibbles + hbits
                                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                                     | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                                uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
                                val_lo -= 0x10101010u;
                                val_lo ^= 0x80808080u;
                                packed_groups[g] = static_cast<int32_t>(val_lo);

                                // High nibbles + hbits
                                const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                                const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
                                const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                                     | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
                                uint32_t val_hi = (hi_nibs | hb_hi) ^ 0x80808080u;
                                val_hi -= 0x10101010u;
                                val_hi ^= 0x80808080u;
                                packed_groups[g + 4] = static_cast<int32_t>(val_hi);
                            }
                        }
                    }
                    else if constexpr (Traits::is_3bit)
                    {
                        // Q3_K: 3-bit decode via v_perm_b32 (8-entry codebook, single call)
                        // Payload layout:
                        //   [0..7]:  q2 — packed 2-bit:  byte[g] = elem[4g]&3 | (elem[4g+1]&3)<<2 | ...
                        //   [8..11]: hbits — packed 1-bit high bits: 32 bits in 4 bytes
                        // Decoded: int8 = (low2 | (hbit << 2)) - 4  → range [-4, +3]
                        //
                        // 3-bit codebook: {-4,-3,-2,-1,0,1,2,3} = {0xFC,0xFD,0xFE,0xFF,0x00,0x01,0x02,0x03}
                        // Only 8 entries → single v_perm call (selector bits [2:0] = value)
                        constexpr uint32_t Q3_CB_A = 0xFFFEFDFCu;  // indices 0-3: {-4,-3,-2,-1}
                        constexpr uint32_t Q3_CB_B = 0x03020100u;  // indices 4-7: {0,1,2,3}

                        const uint32_t hbits = *reinterpret_cast<const uint32_t*>(payload + 8);

                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            const uint8_t q2_byte = payload[g];
                            const int base_elem = g * 4;

                            // Unpack 4 × 2-bit values to byte positions [0..3]
                            uint32_t q2 = (static_cast<uint32_t>(q2_byte & 3u))
                                        | (static_cast<uint32_t>((q2_byte >> 2) & 3u) << 8)
                                        | (static_cast<uint32_t>((q2_byte >> 4) & 3u) << 16)
                                        | (static_cast<uint32_t>(q2_byte >> 6) << 24);

                            // Extract 4 consecutive high bits → deposit as bit 2 of each byte
                            const uint32_t hb4 = (hbits >> base_elem) & 0xFu;
                            const uint32_t hb_deposit = ((hb4 & 1u) << 2)
                                                      | ((hb4 & 2u) << 9)
                                                      | ((hb4 & 4u) << 16)
                                                      | ((hb4 & 8u) << 23);

                            // 3-bit value per byte [0..7], single v_perm → centered output
                            packed_groups[g] = static_cast<int32_t>(
                                __builtin_amdgcn_perm(Q3_CB_A, Q3_CB_B, q2 | hb_deposit));
                        }
                    }
                    else if constexpr (Traits::is_2bit)
                    {
                        // Q2_K: 2-bit decode from 12-byte payload
                        // Payload layout:
                        //   [0..7]:   q2 — packed 2-bit:  byte[g] = elem[4g]&3 | (elem[4g+1]&3)<<2 | ...
                        //   [8..11]:  embedded FP16 mins: min_lo (2B) + min_hi (2B)
                        // Decoded: int8 = q2 (unsigned, range [0, 3])
                        // Asymmetric min correction applied via split sum_a in accumulation
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            const uint8_t q2_byte = payload[g];
                            const int8_t v0 = static_cast<int8_t>((q2_byte >> 0) & 3);
                            const int8_t v1 = static_cast<int8_t>((q2_byte >> 2) & 3);
                            const int8_t v2 = static_cast<int8_t>((q2_byte >> 4) & 3);
                            const int8_t v3 = static_cast<int8_t>((q2_byte >> 6) & 3);
                            packed_groups[g] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(v0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(v1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24));
                        }
                    }
                    else if constexpr (Traits::is_6bit)
                    {
                        // Q6_K: 6-bit SWAR decode from 24-byte payload
                        // Payload layout:
                        //   [0..15]:  ql — paired nibbles: byte[i] = (raw6[i] & 0xF) | ((raw6[i+16] & 0xF) << 4)
                        //   [16..23]: qh — packed 2-bit:   byte[i] = (raw6[4i]>>4)&3 | ... (4 elements per byte)
                        // Decoded: int8 = ((ql_nib) | ((qh_2bit) << 4)) - 32  → range [-32, +31]
                        //
                        // SWAR: extract nibbles, scatter 2-bit high fields to byte positions
                        // at bits [5:4], then byte-parallel subtract-32 via XOR-SUB-XOR bias trick
                        const uint8_t* qh_p = payload + 16;

                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                            const uint32_t h_lo = qh_p[g];      // 2-bit fields for elements 4g..4g+3
                            const uint32_t h_hi = qh_p[g + 4];  // 2-bit fields for elements 4g+16..4g+19

                            // Low nibbles + scattered 2-bit highs + subtract-32
                            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                            const uint32_t hb_lo = ((h_lo & 0x03u) << 4)  | ((h_lo & 0x0Cu) << 10)
                                                 | ((h_lo & 0x30u) << 16) | ((h_lo & 0xC0u) << 22);
                            uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
                            val_lo -= 0x20202020u;
                            val_lo ^= 0x80808080u;
                            packed_groups[g] = static_cast<int32_t>(val_lo);

                            // High nibbles + scattered 2-bit highs + subtract-32
                            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                            const uint32_t hb_hi = ((h_hi & 0x03u) << 4)  | ((h_hi & 0x0Cu) << 10)
                                                 | ((h_hi & 0x30u) << 16) | ((h_hi & 0xC0u) << 22);
                            uint32_t val_hi = (hi_nibs | hb_hi) ^ 0x80808080u;
                            val_hi -= 0x20202020u;
                            val_hi ^= 0x80808080u;
                            packed_groups[g + 4] = static_cast<int32_t>(val_hi);
                        }
                    }
                    else if constexpr (Traits::is_iq_grid)
                    {
                        // IQ grid decode: constant-memory LUT lookup + sign application
                        // Payload: [qs × N_QS][qh × 0or1][signs × 4]
                        //   IQ3_S:   [qs×8][qh×1][signs×4] = 13B   IQ3_XXS: [qs×8][signs×4] = 12B
                        //   IQ2_S:   [qs×4][qh×1][signs×4] = 9B    IQ2_XS:  [qs×4][qh×1][signs×4] = 9B
                        //   IQ2_XXS: [qs×4][signs×4] = 8B
                        constexpr int n_qs = Traits::is_iq3_grid ? 8 : 4;
                        constexpr int sign_off = n_qs + (Traits::has_qh ? 1 : 0);
                        const uint8_t qh_byte = Traits::has_qh ? payload[n_qs] : 0;

                        if constexpr (Traits::is_iq3_grid)
                        {
                            // IQ3: 8 grid lookups × 4 elements = 32 elements
                            // Each uint32_t grid entry → 4 unsigned bytes
                            #pragma unroll
                            for (int g = 0; g < 8; ++g)
                            {
                                int idx = payload[g];
                                if constexpr (Traits::has_qh)
                                    idx |= (static_cast<int>((qh_byte >> g) & 1) << 8);
                                const uint32_t grid4 = iq3_grid_lookup<FMT>(idx);
                                // 4 sign bits: lower nibble for even g, upper for odd
                                const uint8_t sb = payload[sign_off + g / 2];
                                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sn));
                            }
                        }
                        else
                        {
                            // IQ2: 4 grid lookups × 8 elements = 32 elements
                            // Each uint64_t grid entry → 8 unsigned bytes → 2 sdot4 groups
                            #pragma unroll
                            for (int l = 0; l < 4; ++l)
                            {
                                int idx = payload[l];
                                if constexpr (Traits::qh_bits_per_lookup == 1)
                                    idx |= (static_cast<int>((qh_byte >> l) & 1) << 8);
                                else if constexpr (Traits::qh_bits_per_lookup == 2)
                                    idx |= (static_cast<int>((qh_byte >> (2 * l)) & 3) << 8);
                                const uint64_t grid8 = iq2_grid_lookup<FMT>(idx);
                                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                                const uint8_t signs = payload[sign_off + l];
                                packed_groups[l * 2]     = static_cast<int32_t>(
                                    iq_apply_signs_4(g_lo, signs & 0xF));
                                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                                    iq_apply_signs_4(g_hi, signs >> 4));
                            }
                        }
                    }
                    else if constexpr (Traits::is_iq1_grid)
                    {
                        // IQ1: 4 grid lookups × 8 signed elements {-1,0,+1} = 32 elements
                        // Grid values already signed — no sign application needed.
                        // Payload: [qs×4][qh×2] = 6 bytes (IQ1_S) or [qs×4][qh×2][delta_lo×2][delta_hi×2] = 10 bytes (IQ1_M)
                        // Grid index: 11 bits (8 from qs + 3 from qh)
                        if constexpr (FMT == NVNNI_IQ1_S)
                        {
                            // IQ1_S: qh is a uint16_t with format:
                            //   bits 0-2:   high 3 bits for group 0's grid index
                            //   bits 3-5:   high 3 bits for group 1's grid index
                            //   bits 6-8:   high 3 bits for group 2's grid index
                            //   bits 9-11:  high 3 bits for group 3's grid index
                            //   bits 12-14: sub-scale selector (consumed during packing, ignored here)
                            //   bit 15:     delta sign (consumed during packing as min correction)
                            const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                                                   | (static_cast<uint16_t>(payload[5]) << 8);
                            #pragma unroll
                            for (int l = 0; l < 4; ++l)
                            {
                                const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
                                const uint64_t grid8 = iq1_grid_lookup(idx);
                                packed_groups[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(grid8));
                                packed_groups[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
                            }
                        }
                        else // NVNNI_IQ1_M
                        {
                            // IQ1_M: qh bytes encode high index bits + delta signs
                            //   qh[0]: bits 0-2 = group 0 high idx, bit 3 = delta sign 0,
                            //          bits 4-6 = group 1 high idx, bit 7 = delta sign 1
                            //   qh[1]: bits 0-2 = group 2 high idx, bit 3 = delta sign 2,
                            //          bits 4-6 = group 3 high idx, bit 7 = delta sign 3
                            const uint8_t qh0 = payload[4], qh1 = payload[5];
                            const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
                            const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
                            const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
                            const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);
                            uint64_t g;
                            g = iq1_grid_lookup(idx0);
                            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(g));
                            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                            g = iq1_grid_lookup(idx1);
                            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(g));
                            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                            g = iq1_grid_lookup(idx2);
                            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(g));
                            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                            g = iq1_grid_lookup(idx3);
                            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(g));
                            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                        }
                    }
                    else
                    {
                        // Q4_0 / Q4_1: 4-bit nibble decode via SWAR / v_perm_b32
                        if constexpr (Traits::center_4bit == 8)
                        {
                            // Q4_0: w = nibble - 8 → range [-8,+7]
                            // v_perm_b32 register LUT: same pattern as IQ4_NL
                            //   nibble 0→0xF8(-8), 1→0xF9(-7), ..., 7→0xFF(-1)
                            //   nibble 8→0x00(0),  9→0x01(1),  ..., 15→0x07(7)
                            constexpr uint32_t CB_LO_A = 0xFBFAF9F8u;  // cb[0..3]
                            constexpr uint32_t CB_LO_B = 0xFFFEFDFCu;  // cb[4..7]
                            constexpr uint32_t CB_HI_A = 0x03020100u;  // cb[8..11]
                            constexpr uint32_t CB_HI_B = 0x07060504u;  // cb[12..15]

                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                            {
                                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

                                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                                {
                                    const uint32_t sel = lo_nibs & 0x07070707u;
                                    const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_A, CB_LO_B, sel);
                                    const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_A, CB_HI_B, sel);
                                    uint32_t m = (lo_nibs >> 3) & 0x01010101u;
                                    m |= (m << 1); m |= (m << 2); m |= (m << 4);
                                    packed_groups[g] = static_cast<int32_t>((r_hi & m) | (r_lo & ~m));
                                }

                                const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                                {
                                    const uint32_t sel = hi_nibs & 0x07070707u;
                                    const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_A, CB_LO_B, sel);
                                    const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_A, CB_HI_B, sel);
                                    uint32_t m = (hi_nibs >> 3) & 0x01010101u;
                                    m |= (m << 1); m |= (m << 2); m |= (m << 4);
                                    packed_groups[g + 4] = static_cast<int32_t>((r_hi & m) | (r_lo & ~m));
                                }
                            }
                        }
                        else
                        {
                            // Q4_1: w = nibble (unsigned, range [0,15], center_4bit = 0)
                            // Pure SWAR nibble extraction — no arithmetic needed
                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                            {
                                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                                packed_groups[g] = static_cast<int32_t>(raw & 0x0F0F0F0Fu);
                                packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & 0x0F0F0F0Fu);
                            }
                        }
                    }

                    // sdot4 accumulation and scale application
                    if constexpr (Traits::is_dual_scale)
                    {
                        // Dual-scale (Q6_K): elements 0-15 use scale_lo, 16-31 use scale_hi
                        // scale_lo is in d_block_scales[], scale_hi is in d_block_mins[]
                        int32_t acc_lo = 0, acc_hi = 0;
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                            acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
                        #pragma unroll
                        for (int g = 4; g < 8; ++g)
                            acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

                        const float scale_lo = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                        const uint16_t hi_bits = d_block_mins[linear];
                        const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&hi_bits));
                        f_acc[c] += static_cast<float>(acc_lo) * scale_lo
                                  + static_cast<float>(acc_hi) * scale_hi;

                        // Q2_K/IQ1_M: dual-scale + embedded mins correction
                        // Payload offset varies: Q2_K at byte 8, IQ1_M at byte 6
                        if constexpr (Traits::is_dual_scale_asym)
                        {
                            constexpr int32_t ones = 0x01010101;
                            int32_t sum_a_lo = 0, sum_a_hi = 0;
                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                                sum_a_lo = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_lo, false);
                            #pragma unroll
                            for (int g = 4; g < 8; ++g)
                                sum_a_hi = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_hi, false);

                            const float min_lo = __half2float(*reinterpret_cast<const __half*>(payload + Traits::embedded_min_offset));
                            const float min_hi = __half2float(*reinterpret_cast<const __half*>(payload + Traits::embedded_min_offset + 2));
                            f_acc[c] += static_cast<float>(sum_a_lo) * min_lo
                                      + static_cast<float>(sum_a_hi) * min_hi;
                        }
                    }
                    else
                    {
                        // Single-scale: 8× sdot4 with one scale factor
                        int32_t block_acc = 0;
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                            block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

                        const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                        f_acc[c] += static_cast<float>(block_acc) * block_d;

                        // Asymmetric min correction: f_acc += sum_a * block_m
                        // Derives from: value = d*q + m  →  Σ a_i*value_i = d*Σ(a_i*q_i) + m*Σ(a_i)
                        if constexpr (Traits::is_asymmetric)
                        {
                            const uint16_t min_bits = d_block_mins[linear];
                            const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                            f_acc[c] += static_cast<float>(sum_a) * block_m;
                        }
                    }
                }
            }
        }

        // Apply activation scale and write output
        const float scale_a = d_scale_A[0];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                const float result = f_acc[c] * scale_a;
                if (kblocks == 1)
                {
                    // Direct: write final output
                    d_C_fp32[n] = result;
                }
                else
                {
                    // Scatter: plain store to partial buffer (no atomicAdd)
                    d_partial_fp32[static_cast<size_t>(kb) * N + n] = result;
                }
            }
        }
    }
}


// ============================================================================
// Reduce kernel: sum K-partitions from scatter buffer to final output.
// Simple summation — no per-column scales or bias (handled externally).
//
// Grid: dim3((N + TILE_N - 1) / TILE_N), Block: dim3(TILE_N / CPT)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_native_vnni_reduce_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB × N] partial buffer
    float* __restrict__ d_C_fp32,                // [N] output
    int N, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N) return;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < kblocks; ++kb)
            sum += d_partial_fp32[static_cast<size_t>(kb) * N + n];

        d_C_fp32[n] = sum;
    }
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

extern "C" {

// ============================================================================
// Native-VNNI GEMV dispatch: lossless decode with FP16 per-block scales
//
// 2-tier dispatch (matching INT8 scatter pattern):
//   KB=1  → Direct:         single kernel writes final output, no partials
//   KB>1  → Scatter+Reduce: scatter writes partials, reduce sums to output
//
// Eliminates atomicAdd and output memset from the old implementation.
// ============================================================================
bool rocmGemv_native_vnni_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,  // __half* (FP16 scale d)
    const void* d_block_mins,    // __half* (FP16 min m), NULL for symmetric formats
    float* d_C_fp32,
    const float* d_scale_A,
    float* d_partial_fp32,       // [KB_MAX × N] partial buffer (nullable when KB=1)
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = 64;             // Output columns per block (reduced for 100% coalescing)
    constexpr int CPT = 1;             // Columns per thread (1 = perfect 16B stride alignment)
    constexpr int THREADS = TN / CPT;  // 64 threads = 1 wavefront (wavefront64)
    constexpr int NUM_CUS = 60;        // MI50/MI60

    const int blocks_per_row = K / 32;
    const int grid_n = (N + TN - 1) / TN;

    // =========================================================================
    // CU-aware KB heuristic (adapted from INT8 scatter)
    //
    // Goal: maximize CU occupancy by splitting K across grid.y blocks.
    // Each K-partition writes to a partial buffer (plain store, no atomicAdd).
    // A separate reduce kernel sums partials.
    //
    // When KB=1, the kernel writes directly to output (no partials, no reduce).
    //
    // With TN=64/CPT=1, coalescing is 100% (16B stride). Higher occupancy
    // target ensures enough waves per SIMD to hide HBM2 latency (~400 cycles).
    // KB_CAP must match SCATTER_KB_MAX in ROCmQuantisedGemmKernel::getWorkspaceRequirements().
    // =========================================================================
    constexpr int TARGET_WAVES_PER_CU = 24;   // 6 waves/SIMD (4 SIMDs/CU) for latency hiding
    constexpr int MIN_BLOCKS_PER_WAVE = 1;     // Allow single K-block per wave
    constexpr int KB_CAP = 64;                 // Must match SCATTER_KB_MAX in workspace alloc

    int target_total_waves = TARGET_WAVES_PER_CU * NUM_CUS;  // 1440
    int kb_raw = max(1, (target_total_waves + grid_n - 1) / grid_n);
    int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
    kb_max = min(kb_max, KB_CAP);  // Never exceed workspace partial buffer size
    kb_raw = min(kb_raw, kb_max);

    // Round to nearest factor of blocks_per_row for even partitioning
    int kb = kb_raw;
    if (blocks_per_row % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }
    if (kb < 1) kb = 1;

    // Validate partial buffer for scatter path
    if (kb > 1 && !d_partial_fp32)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
        return false;
    }

    dim3 block(THREADS, 1, 1);
    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16   = static_cast<const uint16_t*>(d_block_mins);

    // Dispatch macro: handles both direct (kb=1) and scatter+reduce (kb>1) paths
    // for each format template instantiation, avoiding 30 copy-pasted launches.
    #define LAUNCH_NATIVE_VNNI(FMT_TAG)                                                  \
        if (kb <= 1) {                                                                    \
            hipLaunchKernelGGL(                                                           \
                HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN, CPT, FMT_TAG>),            \
                dim3(grid_n, 1), block, 0, static_cast<hipStream_t>(stream),             \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16,                           \
                d_C_fp32, d_partial_fp32, d_scale_A, N, K, 1);                           \
        } else {                                                                          \
            hipLaunchKernelGGL(                                                           \
                HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN, CPT, FMT_TAG>),            \
                dim3(grid_n, kb), block, 0, static_cast<hipStream_t>(stream),            \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16,                           \
                d_C_fp32, d_partial_fp32, d_scale_A, N, K, kb);                          \
            hipLaunchKernelGGL(                                                           \
                HIP_KERNEL_NAME(gemv_native_vnni_reduce_kernel_t<TN, CPT>),              \
                dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),                \
                d_partial_fp32, d_C_fp32, N, kb);                                        \
        }

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_NATIVE_VNNI(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_NATIVE_VNNI(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_NATIVE_VNNI(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_NATIVE_VNNI(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_NATIVE_VNNI(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_NATIVE_VNNI(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_NATIVE_VNNI(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_NATIVE_VNNI(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_NATIVE_VNNI(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_NATIVE_VNNI(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_NATIVE_VNNI(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_NATIVE_VNNI(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_NATIVE_VNNI(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_NATIVE_VNNI(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_NATIVE_VNNI(NVNNI_IQ1_M);    break;
    default:
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] Unknown codebook_id=%u (N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    #undef LAUNCH_NATIVE_VNNI

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] Kernel launch failed: %s (N=%d K=%d kb=%d codebook=%u)\n",
                hipGetErrorString(err), N, K, kb, static_cast<unsigned>(codebook_id));
        return false;
    }

    return true;
}

// ============================================================================
// IQ grid table initialization: copies host-side LUT data to device constant
// memory.  Must be called once per device before launching any IQ grid kernel.
// ============================================================================
bool rocmInitIQGridTables(
    int device_id,
    const void* h_iq3s_grid,    // [512]  uint32_t
    const void* h_iq3xxs_grid,  // [256]  uint32_t
    const void* h_iq2s_grid,    // [1024] uint64_t
    const void* h_iq2xs_grid,   // [512]  uint64_t
    const void* h_iq2xxs_grid,  // [256]  uint64_t
    const void* h_iq1s_grid)    // [2048] uint64_t (IQ1_S/IQ1_M shared)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipError_t e;
    e = hipMemcpyToSymbol(d_iq3s_grid, h_iq3s_grid, sizeof(d_iq3s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq3s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq3xxs_grid, h_iq3xxs_grid, sizeof(d_iq3xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq3xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2s_grid, h_iq2s_grid, sizeof(d_iq2s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq2s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2xs_grid, h_iq2xs_grid, sizeof(d_iq2xs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq2xs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2xxs_grid, h_iq2xxs_grid, sizeof(d_iq2xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq2xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq1s_grid, h_iq1s_grid, sizeof(d_iq1s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq1s failed: %s\n", hipGetErrorString(e)); return false; }
    return true;
}

} // extern "C"
