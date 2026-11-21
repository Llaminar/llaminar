/**
 * @file ROCmNativeQuantisedGemmDecode.hip.inc
 * @brief Shared compact weight decoders for ordered dot4 and native WMMA prefill.
 *
 * Every physical codebook has one decoder and metadata contract. Each compiling
 * translation unit owns its constant tables, initialized explicitly before any
 * graph capture. Sharing source does not couple compiled wave widths: ordered
 * shards retain wave64 and the independently compiled gfx1100 WMMA uses wave64.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include "../../../tensors/NativeQuantisedFormatInfo.h"
#include "../../../backends/rocm/HipDeviceGuard.h"

// ============================================================================
// Format tags — full set matching ROCmGemvKernel_native_quantised.hip
//
// Each value is the codebook_id from ROCmPackedWeights::native_quantised_codebook_id.
// ============================================================================

/** @brief Physical payload and scale identities shared by both compute families. */
enum NativeQuantisedGemmFormat : uint8_t {
    NATIVE_QUANTISED_GEMM_Q4_0    = 0,   // 4-bit symmetric, linear decode (q - 8)
    NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook,
    NATIVE_QUANTISED_GEMM_IQ4_NL  = 4,   // 4-bit symmetric, LUT decode via v_perm_b32
    NATIVE_QUANTISED_GEMM_Q4_1    = 5,   // 4-bit asymmetric (q as-is, has min)
    NATIVE_QUANTISED_GEMM_Q5_0    = 6,   // 5-bit symmetric, linear decode (q - 16)
    NATIVE_QUANTISED_GEMM_Q5_1    = 7,   // 5-bit asymmetric (q as-is, has min)
    NATIVE_QUANTISED_GEMM_Q6_K    = 8,   // 6-bit symmetric, dual-scale per 32-element block
    NATIVE_QUANTISED_GEMM_Q3_K    = 9,   // 3-bit symmetric, dual-scale per 32-element block
    NATIVE_QUANTISED_GEMM_Q2_K    = 10,  // 2-bit asymmetric, dual-scale + embedded mins
    NATIVE_QUANTISED_GEMM_IQ3_S   = 11,  // IQ3 grid, 9-bit index, direct signs
    NATIVE_QUANTISED_GEMM_IQ3_XXS = 12,  // IQ3 grid, 8-bit index, pre-resolved signs
    NATIVE_QUANTISED_GEMM_IQ2_S   = 13,  // IQ2 grid, 10-bit index, dual-scale
    NATIVE_QUANTISED_GEMM_IQ2_XS  = 14,  // IQ2 grid, 9-bit index, dual-scale
    NATIVE_QUANTISED_GEMM_IQ2_XXS = 15,  // IQ2 grid, 8-bit index
    NATIVE_QUANTISED_GEMM_IQ1_S   = 16,  // IQ1 grid, 11-bit index, asymmetric, embedded scales
    NATIVE_QUANTISED_GEMM_IQ1_M   = 17,  // IQ1 grid, 11-bit index, dual-scale + embedded mins
    NATIVE_QUANTISED_GEMM_Q8_0    = 19,  // 8-bit symmetric, direct INT8 payload (32 bytes)
    NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN =
        llaminar2::kNativeQuantisedExpandedInt8MinCodebook,
};

// ============================================================================
// Compile-time format traits
// ============================================================================

/** @brief Compile-time compact payload and exact source correction requirements. */
template <NativeQuantisedGemmFormat FMT>
struct NATIVE_QUANTISEDGemmTraits {
    static constexpr bool is_asymmetric =
        FMT == NATIVE_QUANTISED_GEMM_Q4_1 || FMT == NATIVE_QUANTISED_GEMM_Q5_1 ||
        FMT == NATIVE_QUANTISED_GEMM_IQ1_S || FMT == NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN;
    static constexpr bool is_iq3_grid = (FMT == NATIVE_QUANTISED_GEMM_IQ3_S || FMT == NATIVE_QUANTISED_GEMM_IQ3_XXS);
    static constexpr bool is_iq2_grid = (FMT == NATIVE_QUANTISED_GEMM_IQ2_S || FMT == NATIVE_QUANTISED_GEMM_IQ2_XS || FMT == NATIVE_QUANTISED_GEMM_IQ2_XXS);
    static constexpr bool is_iq1_grid = (FMT == NATIVE_QUANTISED_GEMM_IQ1_S || FMT == NATIVE_QUANTISED_GEMM_IQ1_M);
    static constexpr bool is_iq_grid  = (is_iq3_grid || is_iq2_grid || is_iq1_grid);
    static constexpr int qh_bits_per_lookup =
        (FMT == NATIVE_QUANTISED_GEMM_IQ2_S) ? 2 : (FMT == NATIVE_QUANTISED_GEMM_IQ3_S || FMT == NATIVE_QUANTISED_GEMM_IQ2_XS) ? 1 : 0;
    static constexpr bool has_qh = (qh_bits_per_lookup > 0);
    static constexpr bool is_dual_scale = (FMT == NATIVE_QUANTISED_GEMM_Q6_K || FMT == NATIVE_QUANTISED_GEMM_Q3_K || FMT == NATIVE_QUANTISED_GEMM_Q2_K
                                        || FMT == NATIVE_QUANTISED_GEMM_IQ2_S || FMT == NATIVE_QUANTISED_GEMM_IQ2_XS
                                        || FMT == NATIVE_QUANTISED_GEMM_IQ1_M);
    static constexpr bool is_dual_scale_asym = (FMT == NATIVE_QUANTISED_GEMM_Q2_K);
    // All scales/mins are in global arrays (not embedded in payload).
    static constexpr bool has_embedded_scales = false;
    static constexpr int  embedded_scale_offset = -1;
    static constexpr int  embedded_min_offset = -1;
    // Q2_K needs a separate emin array (per-sub-block FP16 min pairs).
    // IQ1_M uses Pattern D plus the canonical signed one-eighth correction.
    static constexpr bool needs_emin_array = (FMT == NATIVE_QUANTISED_GEMM_Q2_K);
    static constexpr bool is_5bit   = (FMT == NATIVE_QUANTISED_GEMM_Q5_0 || FMT == NATIVE_QUANTISED_GEMM_Q5_1);
    static constexpr bool is_6bit   = (FMT == NATIVE_QUANTISED_GEMM_Q6_K);
    static constexpr bool is_3bit   = (FMT == NATIVE_QUANTISED_GEMM_Q3_K);
    static constexpr bool is_2bit   = (FMT == NATIVE_QUANTISED_GEMM_Q2_K);
    static constexpr bool is_8bit =
        FMT == NATIVE_QUANTISED_GEMM_Q8_0 || FMT == NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN;
    static constexpr bool uses_lut  = (FMT == NATIVE_QUANTISED_GEMM_IQ4_NL || FMT == NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE);
    static constexpr int  payload_bytes =
        is_8bit ? 32 :                              // Q8_0: 32 raw INT8 values
        is_iq1_grid ? 6 :                          // IQ1_S/IQ1_M: qs[4]+qh[2], native size
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 12 : is_2bit ? 8 : 16;
    static constexpr int center_4bit = (FMT == NATIVE_QUANTISED_GEMM_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NATIVE_QUANTISED_GEMM_Q5_0) ? 16 : 0;
    // Does the compute phase need an auxiliary scale/min?
    static constexpr bool needs_aux_compute = is_dual_scale || is_asymmetric ||
        FMT == NATIVE_QUANTISED_GEMM_IQ3_S || FMT == NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE;
    // Aux value always loaded from d_block_mins (no embedded scales anymore).
    static constexpr bool loads_aux_from_global = needs_aux_compute;
    // Cooperative sum_a precompute: Pattern DA needs per-half sums for emin
    // correction, Pattern A needs full-block sums for min correction. Both
    // benefit from cooperative A-load phase computation instead of inline
    // per-thread accumulation (eliminates M_PER_THREAD×8 sdot4/block + 4 VGPRs).
    static constexpr bool needs_cooperative_sum_a =
        is_dual_scale_asym || is_asymmetric;
    // Number of uint32 dwords to stage payload (ceil(payload_bytes / 4))
    static constexpr int payload_dwords = (payload_bytes + 3) / 4;

    // ---- Per-pattern streaming dispatch for LM_Head shapes ----
    //
    // On N-heavy shapes (LM_Head N≥16384, K≤4096), cooperative vs streaming
    // performance depends critically on decode complexity (3B_LM_Head M=128):
    //
    //   Pattern S (simple decode):    streaming wins (Q4_0 0.82x vs coop 0.74x)
    //   Pattern D (dual-scale):       streaming can win for lookup-heavy formats
    //   Pattern A (asymmetric):       coop wins     (Q4_1 0.61x vs streaming 0.49x)
    //   Pattern DA (dual-scale+asym): coop wins 2×  (Q2_K 0.34x vs streaming 0.15x)
    //
    // Root cause: Pattern A/DA decode is heavy enough that per-thread streaming
    // causes register spilling.  Cooperative shares A-data through LDS,
    // reducing per-thread register pressure by ~20-30%.
    //
    // Q6_K is deliberately redirected to the cooperative family for wide
    // production-prefill geometries below: current gfx906 measurements across
    // M=32..1024 show 11-17% lower latency than this older streaming policy.
    // IQ1_M's four activation-group sums are computed cooperatively once per
    // row/block and consumed from LDS by all output-column owners. This makes
    // the streaming family both exact and economical for IQ1_M.
    static constexpr bool prefer_streaming =
        !is_asymmetric &&
        !is_dual_scale_asym;
};

// ============================================================================
// IQ grid constant memory tables (separate from GEMV file — each .hip TU
// has its own __constant__ symbols)
// ============================================================================

static __device__ __constant__ uint32_t d_iq3s_grid_gemm[512];
static __device__ __constant__ uint32_t d_iq3xxs_grid_gemm[256];
static __device__ __constant__ uint64_t d_iq2s_grid_gemm[1024];
static __device__ __constant__ uint64_t d_iq2xs_grid_gemm[512];
static __device__ __constant__ uint64_t d_iq2xxs_grid_gemm[256];
static __device__ __constant__ uint64_t d_iq1s_grid_gemm[2048];

/**
 * @brief Read a four-byte IQ3 grid vector from this TU's immutable table.
 * @tparam FMT IQ3 physical codebook choosing the table size.
 * @param idx Admitted encoded grid index.
 * @return Four unsigned grid values packed in one word.
 */
template <NativeQuantisedGemmFormat FMT>
__device__ __forceinline__ uint32_t iq3_grid_lookup_gemm(int idx)
{
    if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ3_S) return d_iq3s_grid_gemm[idx];
    else return d_iq3xxs_grid_gemm[idx];
}

/**
 * @brief Read an eight-byte IQ2 grid vector from this TU's immutable table.
 * @tparam FMT IQ2 physical codebook choosing the table size.
 * @param idx Admitted encoded grid index.
 * @return Eight unsigned grid values packed in one double word.
 */
template <NativeQuantisedGemmFormat FMT>
__device__ __forceinline__ uint64_t iq2_grid_lookup_gemm(int idx)
{
    if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ2_S)       return d_iq2s_grid_gemm[idx];
    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ2_XS) return d_iq2xs_grid_gemm[idx];
    else                                          return d_iq2xxs_grid_gemm[idx];
}

/**
 * @brief Read the shared IQ1 vector from this TU's immutable table.
 * @param idx Admitted eleven-bit encoded grid index.
 * @return Eight grid values packed in one double word.
 */
__device__ __forceinline__ uint64_t iq1_grid_lookup_gemm(int idx)
{
    return d_iq1s_grid_gemm[idx];
}

/**
 * @brief Apply four independent source sign bits with packed integer arithmetic.
 * @param grid4 Four unsigned grid values in packed bytes.
 * @param sign_lo4 Source sign bits in the low nibble.
 * @return Four signed INT8 values with the same lane order as serial GEMV.
 */
__device__ __forceinline__ uint32_t iq_apply_signs_4_gemm(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t spread = (static_cast<uint32_t>(sign_lo4 & 0xFu) * 0x00204081u) & 0x01010101u;
    uint32_t mask;
    asm("v_lshlrev_b32 %0, 8, %1\n\t"
        "v_sub_u32 %0, %0, %1"
        : "=v"(mask) : "v"(spread));
    return (grid4 ^ mask) + spread;
}

// ============================================================================
// Payload load helper: loads payload_bytes into dword array
//
// Uses vectorized loads where alignment permits. For non-power-of-2 sizes,
// individual dword/byte loads are used (misaligned dword loads are supported
// on gfx906 with negligible penalty).
// ============================================================================

/**
 * @brief Load one compact payload without reading beyond its declared extent.
 * @tparam PB Physical payload size in bytes.
 * @param src Start of the live compact source block.
 * @param out Destination words, including a zero-padded final partial word.
 */
template <int PB>
__device__ __forceinline__ void load_payload(const uint8_t* __restrict__ src,
                                              uint32_t* __restrict__ out)
{
    constexpr int FULL_DW = PB / 4;
    constexpr int TAIL = PB % 4;

    if constexpr (PB >= 16) {
        // First 16 bytes as uint4
        uint4 p = *reinterpret_cast<const uint4*>(src);
        out[0] = p.x; out[1] = p.y; out[2] = p.z; out[3] = p.w;
        if constexpr (FULL_DW >= 5) out[4] = *reinterpret_cast<const uint32_t*>(src + 16);
        if constexpr (FULL_DW >= 6) out[5] = *reinterpret_cast<const uint32_t*>(src + 20);
        if constexpr (FULL_DW >= 7) out[6] = *reinterpret_cast<const uint32_t*>(src + 24);
        if constexpr (FULL_DW >= 8) out[7] = *reinterpret_cast<const uint32_t*>(src + 28);
    } else if constexpr (PB >= 8) {
        uint2 p = *reinterpret_cast<const uint2*>(src);
        out[0] = p.x; out[1] = p.y;
        if constexpr (FULL_DW >= 3) out[2] = *reinterpret_cast<const uint32_t*>(src + 8);
        if constexpr (FULL_DW >= 4) out[3] = *reinterpret_cast<const uint32_t*>(src + 12);
    } else {
        #pragma unroll
        for (int i = 0; i < FULL_DW; ++i)
            out[i] = *reinterpret_cast<const uint32_t*>(src + i * 4);
    }

    if constexpr (TAIL > 0) {
        uint32_t tail = 0;
        #pragma unroll
        for (int b = 0; b < TAIL; ++b)
            tail |= static_cast<uint32_t>(src[FULL_DW * 4 + b]) << (b * 8);
        out[FULL_DW] = tail;
    }
}

// ============================================================================
// Device decode functions: raw payload dwords → 8 packed INT32 groups
//
// Each 32-element quantized block produces 8 int32_t "packed groups",
// each containing 4 signed INT8:
//   groups[0..3]: elements [0..15]
//   groups[4..7]: elements [16..31]
// ============================================================================

/**
 * @brief Decode one compact block into the canonical 32 signed INT8 values.
 * @tparam FMT Physical decoder, independent of the selected compute family.
 * @param raw Complete loaded payload words; unused final bytes are zero.
 * @param packed_groups Eight packed INT32 groups in source K order.
 */
template <NativeQuantisedGemmFormat FMT>
__device__ __forceinline__ void decode_block(
    const uint32_t* __restrict__ raw, int32_t packed_groups[8])
{
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;

    if constexpr (FMT == NATIVE_QUANTISED_GEMM_Q4_0)
    {
        // Q4_0: SWAR centering — (nibble + 0x78) ^ 0x80 → [-8,+7]
        constexpr uint32_t SWAR_ADD = 0x78787878u;
        constexpr uint32_t SWAR_XOR = 0x80808080u;
        constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            packed_groups[g]     = static_cast<int32_t>(((raw[g] & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
            packed_groups[g + 4] = static_cast<int32_t>((((raw[g] >> 4) & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
        }
    }
    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ4_NL || FMT == NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE)
    {
        // IQ4_NL: v_perm_b32 codebook LUT
        constexpr uint32_t CB_LO_A = 0xBFAD9881u, CB_LO_B = 0xF6EADDCFu;
        constexpr uint32_t CB_HI_A = 0x26190D01u, CB_HI_B = 0x71594535u;
        constexpr uint32_t MUX_ID = 0x03020100u, MUX_B3 = 0x04040404u;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t lo = raw[g] & 0x0F0F0F0Fu;
            const uint32_t hi = (raw[g] >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t s = lo & 0x07070707u;
                packed_groups[g] = static_cast<int32_t>(__builtin_amdgcn_perm(
                    __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, s),
                    __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, s),
                    MUX_ID | ((lo >> 1) & MUX_B3)));
            }
            {
                const uint32_t s = hi & 0x07070707u;
                packed_groups[g + 4] = static_cast<int32_t>(__builtin_amdgcn_perm(
                    __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, s),
                    __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, s),
                    MUX_ID | ((hi >> 1) & MUX_B3)));
            }
        }
    }
    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_Q4_1)
    {
        // Q4_1: pure nibble extraction (unsigned, range [0,15])
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            packed_groups[g]     = static_cast<int32_t>(raw[g] & 0x0F0F0F0Fu);
            packed_groups[g + 4] = static_cast<int32_t>((raw[g] >> 4) & 0x0F0F0F0Fu);
        }
    }
    else if constexpr (Traits::is_8bit)
    {
        // Q8_0: 32 raw INT8 values — direct load, no decode needed
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            packed_groups[g] = static_cast<int32_t>(raw[g]);
    }
    else if constexpr (Traits::is_5bit)
    {
        // Q5_0 / Q5_1: nibble + high-bit from qh dword at payload[16..19]
        //
        // Each qh nibble holds 4 individual high-bits for 4 elements.
        // We need to scatter bit i to bit 4 of byte i.  The multiply-mask
        // trick: (nibble * 0x00204081) & 0x01010101 isolates each bit into
        // its target byte's bit 0, then << 4 moves it to bit 4.
        //
        // This replaces the 11-op per-instance (4 AND + 4 SHIFT + 3 OR)
        // scatter with 3 ops (MUL + AND + SHIFT), saving 64 VALU per block.
        constexpr uint32_t BIT_SPREAD = 0x00204081u;
        constexpr uint32_t BYTE_MASK  = 0x01010101u;
        const uint32_t qh_bits = raw[4]; // 5th dword
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t lo_nibs = raw[g] & 0x0F0F0F0Fu;
            const uint32_t hi_nibs = (raw[g] >> 4) & 0x0F0F0F0Fu;
            const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
            const uint32_t hb_lo = ((hb4_lo * BIT_SPREAD) & BYTE_MASK) << 4;
            const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
            const uint32_t hb_hi = ((hb4_hi * BIT_SPREAD) & BYTE_MASK) << 4;
            if constexpr (Traits::center_5bit == 16)
            {
                // Q5_0: centered — XOR-SUB-XOR subtract 16
                uint32_t vlo = (lo_nibs | hb_lo) ^ 0x80808080u;
                vlo -= 0x10101010u; vlo ^= 0x80808080u;
                packed_groups[g] = static_cast<int32_t>(vlo);
                uint32_t vhi = (hi_nibs | hb_hi) ^ 0x80808080u;
                vhi -= 0x10101010u; vhi ^= 0x80808080u;
                packed_groups[g + 4] = static_cast<int32_t>(vhi);
            }
            else
            {
                // Q5_1: unsigned (no centering)
                packed_groups[g]     = static_cast<int32_t>(lo_nibs | hb_lo);
                packed_groups[g + 4] = static_cast<int32_t>(hi_nibs | hb_hi);
            }
        }
    }
    else if constexpr (Traits::is_6bit)
    {
        // Q6_K: 24B → nibble + 2-bit qh − 32
        //
        // 2-bit scatter via two-pass multiply-mask (5 ops instead of 11):
        //   qh byte [ab][cd][ef][gh] → byte3[5:4]=ab, byte2[5:4]=cd,
        //                               byte1[5:4]=ef, byte0[5:4]=gh
        //
        //   Pass 1 (even pairs 0,2): (qh * 0x00010010) & 0x00300030
        //   Pass 2 (odd pairs 1,3):  (qh * 0x00400400) & 0x30003000
        //   Result = pass1 | pass2
        //
        // No carry contamination because pairs are spaced 4+ bits apart
        // within each pass.  Verified for all 256 possible qh values.
        constexpr uint32_t QH_MUL_EVEN = 0x00010010u;
        constexpr uint32_t QH_MASK_EVEN = 0x00300030u;
        constexpr uint32_t QH_MUL_ODD  = 0x00400400u;
        constexpr uint32_t QH_MASK_ODD  = 0x30003000u;
        const uint8_t* qh_p = reinterpret_cast<const uint8_t*>(raw) + 16;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t lo_nibs = raw[g] & 0x0F0F0F0Fu;
            const uint32_t hi_nibs = (raw[g] >> 4) & 0x0F0F0F0Fu;
            const uint32_t h_lo = qh_p[g], h_hi = qh_p[g + 4];
            const uint32_t hb_lo = ((h_lo * QH_MUL_EVEN) & QH_MASK_EVEN)
                                 | ((h_lo * QH_MUL_ODD) & QH_MASK_ODD);
            const uint32_t hb_hi = ((h_hi * QH_MUL_EVEN) & QH_MASK_EVEN)
                                 | ((h_hi * QH_MUL_ODD) & QH_MASK_ODD);
            uint32_t vlo = (lo_nibs | hb_lo) ^ 0x80808080u;
            vlo -= 0x20202020u; vlo ^= 0x80808080u;
            packed_groups[g] = static_cast<int32_t>(vlo);
            uint32_t vhi = (hi_nibs | hb_hi) ^ 0x80808080u;
            vhi -= 0x20202020u; vhi ^= 0x80808080u;
            packed_groups[g + 4] = static_cast<int32_t>(vhi);
        }
    }
    else if constexpr (Traits::is_3bit)
    {
        // Q3_K: native 12B layout — [8B pre-transposed 2-bit][4B grouped hbits]
        //
        // Payload layout (native-size, packed during CPU packing):
        //   raw[0]: half 0 — 2-bit pre-transposed (Q2_K-style), groups 0-3
        //   raw[1]: half 1 — 2-bit pre-transposed (Q2_K-style), groups 4-7
        //   raw[2]: hbits word — group g's 4 hbits at bits [g*4..g*4+3]
        //
        // Decode: Q2_K-style shift+mask for low2, BIT_SPREAD multiply-scatter
        // for hbits, XOR-SUB-XOR centering [0..7] → [-4..+3].
        //
        // BIT_SPREAD = 0x00204081: scatters 4 bits from a nibble into 4 separate
        // bytes' bit 0.  (hb4 * BIT_SPREAD) & 0x01010101 puts bit k into byte k.
        // Verified for all 16 possible hb4 values (0x0..0xF).
        constexpr uint32_t BIT_SPREAD = 0x00204081u;
        const uint32_t hbit_word = raw[2];
        #pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            const uint32_t r = raw[half];
            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                // Extract 2-bit values (Q2_K-style pre-transposed)
                const uint32_t low2_4 = (r >> (g * 2)) & 0x03030303u;
                // Extract and scatter hbit for this group
                const uint32_t hb4 = (hbit_word >> ((half * 4 + g) * 4)) & 0xFu;
                const uint32_t hb = ((hb4 * BIT_SPREAD) & 0x01010101u) << 2;
                // Merge: 3-bit value = low2 | (hbit << 2), range [0..7]
                // Center: [0..7] → [-4..+3] via XOR-SUB-XOR bias trick
                uint32_t val = (low2_4 | hb) ^ 0x80808080u;
                val -= 0x04040404u;
                val ^= 0x80808080u;
                packed_groups[half * 4 + g] = static_cast<int32_t>(val);
            }
        }
    }
    else if constexpr (Traits::is_2bit)
    {
        // Q2_K: pre-transposed 2-bit extraction from 8B payload
        // [0..3] half0, [4..7] half1  (emins in separate d_block_emins array)
        #pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            const uint32_t r = raw[half];
            const int bg = half * 4;
            packed_groups[bg]     = static_cast<int32_t>(r & 0x03030303u);
            packed_groups[bg + 1] = static_cast<int32_t>((r >> 2) & 0x03030303u);
            packed_groups[bg + 2] = static_cast<int32_t>((r >> 4) & 0x03030303u);
            packed_groups[bg + 3] = static_cast<int32_t>((r >> 6) & 0x03030303u);
        }
    }
    else if constexpr (Traits::is_iq3_grid)
    {
        // IQ3_S (13B) / IQ3_XXS (12B): grid lookup + sign application
        if constexpr (!Traits::has_qh)
        {
            // IQ3_XXS: 3 dwords [qs0-7][signs0-3]
            const uint32_t w0 = raw[0], w1 = raw[1], w2 = raw[2];
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                const uint32_t word = (g < 4) ? w0 : w1;
                const int idx = (word >> ((g & 3) * 8)) & 0xFF;
                const uint32_t grid4 = iq3_grid_lookup_gemm<FMT>(idx);
                const uint8_t sb = static_cast<uint8_t>(w2 >> ((g / 2) * 8));
                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4_gemm(grid4, sn));
            }
        }
        else
        {
            // IQ3_S: 3 dwords + 1 ubyte [qs0-7][qh][signs0-3]
            const uint32_t w0 = raw[0], w1 = raw[1], w2 = raw[2];
            const uint8_t signs3 = reinterpret_cast<const uint8_t*>(raw)[12];
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                const uint32_t word = (g < 4) ? w0 : w1;
                int idx = (word >> ((g & 3) * 8)) & 0xFF;
                idx |= (static_cast<int>((w2 >> g) & 1) << 8);
                const uint32_t grid4 = iq3_grid_lookup_gemm<FMT>(idx);
                const uint8_t sb = (g < 6)
                    ? static_cast<uint8_t>(w2 >> ((g / 2 + 1) * 8))
                    : signs3;
                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4_gemm(grid4, sn));
            }
        }
    }
    else if constexpr (Traits::is_iq2_grid)
    {
        if constexpr (!Traits::has_qh)
        {
            // IQ2_XXS (8B): 2 dwords [qs0-3][signs0-3]
            const uint32_t w_qs = raw[0], w_signs = raw[1];
            #pragma unroll
            for (int l = 0; l < 4; ++l)
            {
                const int idx = (w_qs >> (l * 8)) & 0xFF;
                const uint64_t grid8 = iq2_grid_lookup_gemm<FMT>(idx);
                const uint8_t signs = static_cast<uint8_t>(w_signs >> (l * 8));
                packed_groups[l * 2]     = static_cast<int32_t>(
                    iq_apply_signs_4_gemm(static_cast<uint32_t>(grid8), signs & 0xF));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                    iq_apply_signs_4_gemm(static_cast<uint32_t>(grid8 >> 32), signs >> 4));
            }
        }
        else
        {
            // IQ2_S (9B) / IQ2_XS (9B): [qs0-3][qh+signs0-2] + signs3
            const uint32_t w_qs = raw[0], w_qh_signs = raw[1];
            const uint8_t signs3 = reinterpret_cast<const uint8_t*>(raw)[8];
            #pragma unroll
            for (int l = 0; l < 4; ++l)
            {
                int idx = (w_qs >> (l * 8)) & 0xFF;
                if constexpr (Traits::qh_bits_per_lookup == 1)
                    idx |= (static_cast<int>((w_qh_signs >> l) & 1) << 8);
                else if constexpr (Traits::qh_bits_per_lookup == 2)
                    idx |= (static_cast<int>((w_qh_signs >> (2 * l)) & 3) << 8);
                const uint64_t grid8 = iq2_grid_lookup_gemm<FMT>(idx);
                const uint8_t sb = (l < 3)
                    ? static_cast<uint8_t>(w_qh_signs >> ((l + 1) * 8))
                    : signs3;
                packed_groups[l * 2]     = static_cast<int32_t>(
                    iq_apply_signs_4_gemm(static_cast<uint32_t>(grid8), sb & 0xF));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                    iq_apply_signs_4_gemm(static_cast<uint32_t>(grid8 >> 32), sb >> 4));
            }
        }
    }
    else if constexpr (Traits::is_iq1_grid)
    {
        if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_S)
        {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw);
            const uint16_t qh_word = static_cast<uint16_t>(p[4]) | (static_cast<uint16_t>(p[5]) << 8);
            #pragma unroll
            for (int l = 0; l < 4; ++l)
            {
                const int idx = p[l] | (((qh_word >> (3 * l)) & 7) << 8);
                const uint64_t g = iq1_grid_lookup_gemm(idx);
                packed_groups[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(g));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            }
        }
        else // IQ1_M: base grid; signed one-eighth delta is committed separately
        {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw);
            const uint8_t qh0 = p[4], qh1 = p[5];
            const int idx0 = p[0] | ((qh0 & 0x07) << 8);
            const int idx1 = p[1] | (((qh0 >> 4) & 0x07) << 8);
            const int idx2 = p[2] | ((qh1 & 0x07) << 8);
            const int idx3 = p[3] | (((qh1 >> 4) & 0x07) << 8);

            const uint64_t g0 = iq1_grid_lookup_gemm(idx0);
            const uint64_t g1 = iq1_grid_lookup_gemm(idx1);
            const uint64_t g2 = iq1_grid_lookup_gemm(idx2);
            const uint64_t g3 = iq1_grid_lookup_gemm(idx3);
            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(g0));
            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(g0 >> 32));
            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(g1));
            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(g1 >> 32));
            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(g2));
            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(g2 >> 32));
            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(g3));
            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(g3 >> 32));
        }
    }
}


/**
 * @brief Initialize this translation unit's immutable native codebook tables.
 * @param device_id Explicit device owning these constant symbols.
 * @param h_iq3s_grid Host IQ3_S table with 512 uint32 entries.
 * @param h_iq3xxs_grid Host IQ3_XXS table with 256 uint32 entries.
 * @param h_iq2s_grid Host IQ2_S table with 1024 uint64 entries.
 * @param h_iq2xs_grid Host IQ2_XS table with 512 uint64 entries.
 * @param h_iq2xxs_grid Host IQ2_XXS table with 256 uint64 entries.
 * @param h_iq1s_grid Host shared IQ1 table with 2048 uint64 entries.
 * @return True after every table has been published during backend setup.
 * @details Internal linkage preserves ownership of the including TU's symbols.
 */
static bool initialize_native_quantised_gemm_grids(
    int device_id, const void *h_iq3s_grid, const void *h_iq3xxs_grid,
    const void *h_iq2s_grid, const void *h_iq2xs_grid, const void *h_iq2xxs_grid,
    const void *h_iq1s_grid)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipError_t e;
    e = hipMemcpyToSymbol(d_iq3s_grid_gemm, h_iq3s_grid, sizeof(d_iq3s_grid_gemm));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_gemm] iq3s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq3xxs_grid_gemm, h_iq3xxs_grid, sizeof(d_iq3xxs_grid_gemm));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_gemm] iq3xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2s_grid_gemm, h_iq2s_grid, sizeof(d_iq2s_grid_gemm));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_gemm] iq2s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2xs_grid_gemm, h_iq2xs_grid, sizeof(d_iq2xs_grid_gemm));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_gemm] iq2xs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2xxs_grid_gemm, h_iq2xxs_grid, sizeof(d_iq2xxs_grid_gemm));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_gemm] iq2xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq1s_grid_gemm, h_iq1s_grid, sizeof(d_iq1s_grid_gemm));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_gemm] iq1s failed: %s\n", hipGetErrorString(e)); return false; }

    return true;
}
