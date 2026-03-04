/**
 * @file ROCmGemmKernel_native_VNNI.hip
 * @brief Native-VNNI GEMM kernels for lossless sub-8-bit prefill on AMD MI50/MI60 (gfx906)
 *
 * @target AMD Instinct MI50 / MI60 (gfx906), 60 CUs, 4 SIMDs/CU
 * @isa    v_dot4_i32_i8 (VNNI: 4-way signed INT8 dot product, 1 cycle/op)
 * @op     C[M×N] (FP32) = scale_A[M] × Σ_b { block_d[b,N] × sdot4(A_int8[M,K], W_decoded[K,N]) }
 *
 * This file contains the production native-VNNI GEMM kernels for the **prefill**
 * (prompt-processing) phase on AMD gfx906 GPUs.  Unlike the INT8 GEMM kernels
 * (ROCmQuantisedGemmKernel_INT8_VNNI.hip), these kernels:
 *
 *   1. Load COMPACT quantized payloads from HBM (4.5 bpw for Q4_0/IQ4_NL
 *      vs 8 bpw for INT8) — halving HBM bandwidth requirements
 *   2. Decode quantized weights to INT8 in-register during the load phase
 *      (Decode-in-Load architecture)
 *   3. Use FP32 accumulators with per-block FP16 scales (no separate epilogue)
 *   4. Apply activation scale inline → FP32 output directly
 *
 * The decode-in-load approach stores decoded INT8 weights in LDS (same format
 * as INT8 GEMM), making the compute phase identical to V3's inner loop.
 * The key benefit is that global memory reads are halved — the decode overhead
 * is hidden in the load-to-LDS pipeline.
 *
 * Tile dispatch mirrors INT8 V3/V7 strategy:
 *   K >= N (K-heavy) → N_TILE=64  (V3-derived, deeper K iteration)
 *   K <  N (N-heavy) → N_TILE=128 (V7-derived, wider N coverage, all threads load B)
 *
 * Currently supports:
 *   Q4_0   (codebook_id=0): 4-bit symmetric, linear decode (q - 8)
 *   IQ4_NL (codebook_id=4): 4-bit symmetric, LUT decode via v_perm_b32
 *
 * @see ROCmQuantisedGemmKernel_INT8_VNNI.hip for INT8 VNNI GEMM (V3/V7)
 * @see ROCmGemvKernel_native_VNNI.hip for native-VNNI GEMV (M=1 decode)
 * @see docs/v2/NATIVE_VNNI_GEMM_DESIGN.md for architecture design document
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "../../backends/rocm/HipDeviceGuard.h"

// ============================================================================
// Format tags — full set matching ROCmGemvKernel_native_VNNI.hip
//
// Each value is the codebook_id from ROCmPackedWeights::native_vnni_codebook_id.
// ============================================================================

enum NativeVNNIGemmFormat : uint8_t {
    NVNNI_GEMM_Q4_0    = 0,   // 4-bit symmetric, linear decode (q - 8)
    NVNNI_GEMM_IQ4_NL  = 4,   // 4-bit symmetric, LUT decode via v_perm_b32
    NVNNI_GEMM_Q4_1    = 5,   // 4-bit asymmetric (q as-is, has min)
    NVNNI_GEMM_Q5_0    = 6,   // 5-bit symmetric, linear decode (q - 16)
    NVNNI_GEMM_Q5_1    = 7,   // 5-bit asymmetric (q as-is, has min)
    NVNNI_GEMM_Q6_K    = 8,   // 6-bit symmetric, dual-scale per 32-element block
    NVNNI_GEMM_Q3_K    = 9,   // 3-bit symmetric, dual-scale per 32-element block
    NVNNI_GEMM_Q2_K    = 10,  // 2-bit asymmetric, dual-scale + embedded mins
    NVNNI_GEMM_IQ3_S   = 11,  // IQ3 grid, 9-bit index, direct signs
    NVNNI_GEMM_IQ3_XXS = 12,  // IQ3 grid, 8-bit index, pre-resolved signs
    NVNNI_GEMM_IQ2_S   = 13,  // IQ2 grid, 10-bit index, dual-scale
    NVNNI_GEMM_IQ2_XS  = 14,  // IQ2 grid, 9-bit index, dual-scale
    NVNNI_GEMM_IQ2_XXS = 15,  // IQ2 grid, 8-bit index
    NVNNI_GEMM_IQ1_S   = 16,  // IQ1 grid, 11-bit index, asymmetric, embedded scales
    NVNNI_GEMM_IQ1_M   = 17,  // IQ1 grid, 11-bit index, dual-scale + embedded mins
};

// ============================================================================
// Compile-time format traits
// ============================================================================

template <NativeVNNIGemmFormat FMT>
struct NVNNIGemmTraits {
    static constexpr bool is_asymmetric = (FMT == NVNNI_GEMM_Q4_1 || FMT == NVNNI_GEMM_Q5_1 || FMT == NVNNI_GEMM_IQ1_S);
    static constexpr bool is_iq3_grid = (FMT == NVNNI_GEMM_IQ3_S || FMT == NVNNI_GEMM_IQ3_XXS);
    static constexpr bool is_iq2_grid = (FMT == NVNNI_GEMM_IQ2_S || FMT == NVNNI_GEMM_IQ2_XS || FMT == NVNNI_GEMM_IQ2_XXS);
    static constexpr bool is_iq1_grid = (FMT == NVNNI_GEMM_IQ1_S || FMT == NVNNI_GEMM_IQ1_M);
    static constexpr bool is_iq_grid  = (is_iq3_grid || is_iq2_grid || is_iq1_grid);
    static constexpr int qh_bits_per_lookup =
        (FMT == NVNNI_GEMM_IQ2_S) ? 2 : (FMT == NVNNI_GEMM_IQ3_S || FMT == NVNNI_GEMM_IQ2_XS) ? 1 : 0;
    static constexpr bool has_qh = (qh_bits_per_lookup > 0);
    static constexpr bool is_dual_scale = (FMT == NVNNI_GEMM_Q6_K || FMT == NVNNI_GEMM_Q3_K || FMT == NVNNI_GEMM_Q2_K
                                        || FMT == NVNNI_GEMM_IQ2_S || FMT == NVNNI_GEMM_IQ2_XS
                                        || FMT == NVNNI_GEMM_IQ1_M);
    static constexpr bool is_dual_scale_asym = (FMT == NVNNI_GEMM_Q2_K || FMT == NVNNI_GEMM_IQ1_M);
    static constexpr int embedded_min_offset = (FMT == NVNNI_GEMM_IQ1_M) ? 6 : 8;
    static constexpr bool has_embedded_scales = (FMT == NVNNI_GEMM_IQ1_S || FMT == NVNNI_GEMM_IQ1_M);
    static constexpr int  embedded_scale_offset =
        (FMT == NVNNI_GEMM_IQ1_S) ? 6 : (FMT == NVNNI_GEMM_IQ1_M) ? 10 : -1;
    static constexpr bool is_5bit   = (FMT == NVNNI_GEMM_Q5_0 || FMT == NVNNI_GEMM_Q5_1);
    static constexpr bool is_6bit   = (FMT == NVNNI_GEMM_Q6_K);
    static constexpr bool is_3bit   = (FMT == NVNNI_GEMM_Q3_K);
    static constexpr bool is_2bit   = (FMT == NVNNI_GEMM_Q2_K);
    static constexpr bool uses_lut  = (FMT == NVNNI_GEMM_IQ4_NL);
    static constexpr int  payload_bytes =
        is_iq1_grid ? ((FMT == NVNNI_GEMM_IQ1_M) ? 14 : 10) :
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 16 : is_2bit ? 12 : 16;
    static constexpr int center_4bit = (FMT == NVNNI_GEMM_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NVNNI_GEMM_Q5_0) ? 16 : 0;
    // Does the compute phase need an auxiliary scale/min?
    static constexpr bool needs_aux_compute = is_dual_scale || is_asymmetric;
    // Is the auxiliary value loaded from d_block_mins (vs embedded in payload)?
    static constexpr bool loads_aux_from_global = needs_aux_compute && !has_embedded_scales;
    // Number of uint32 dwords to stage payload (ceil(payload_bytes / 4))
    static constexpr int payload_dwords = (payload_bytes + 3) / 4;
};

// ============================================================================
// IQ grid constant memory tables (separate from GEMV file — each .hip TU
// has its own __constant__ symbols)
// ============================================================================

__device__ __constant__ uint32_t d_iq3s_grid_gemm[512];
__device__ __constant__ uint32_t d_iq3xxs_grid_gemm[256];
__device__ __constant__ uint64_t d_iq2s_grid_gemm[1024];
__device__ __constant__ uint64_t d_iq2xs_grid_gemm[512];
__device__ __constant__ uint64_t d_iq2xxs_grid_gemm[256];
__device__ __constant__ uint64_t d_iq1s_grid_gemm[2048];

// IQ grid lookup helpers
template <NativeVNNIGemmFormat FMT>
__device__ __forceinline__ uint32_t iq3_grid_lookup_gemm(int idx)
{
    if constexpr (FMT == NVNNI_GEMM_IQ3_S) return d_iq3s_grid_gemm[idx];
    else return d_iq3xxs_grid_gemm[idx];
}

template <NativeVNNIGemmFormat FMT>
__device__ __forceinline__ uint64_t iq2_grid_lookup_gemm(int idx)
{
    if constexpr (FMT == NVNNI_GEMM_IQ2_S)       return d_iq2s_grid_gemm[idx];
    else if constexpr (FMT == NVNNI_GEMM_IQ2_XS) return d_iq2xs_grid_gemm[idx];
    else                                          return d_iq2xxs_grid_gemm[idx];
}

__device__ __forceinline__ uint64_t iq1_grid_lookup_gemm(int idx)
{
    return d_iq1s_grid_gemm[idx];
}

// SWAR conditional negate: 4 unsigned bytes via 4 sign bits (same as GEMV)
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

template <NativeVNNIGemmFormat FMT>
__device__ __forceinline__ void decode_block(
    const uint32_t* __restrict__ raw, int32_t packed_groups[8])
{
    using Traits = NVNNIGemmTraits<FMT>;

    if constexpr (FMT == NVNNI_GEMM_Q4_0)
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
    else if constexpr (FMT == NVNNI_GEMM_IQ4_NL)
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
    else if constexpr (FMT == NVNNI_GEMM_Q4_1)
    {
        // Q4_1: pure nibble extraction (unsigned, range [0,15])
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            packed_groups[g]     = static_cast<int32_t>(raw[g] & 0x0F0F0F0Fu);
            packed_groups[g + 4] = static_cast<int32_t>((raw[g] >> 4) & 0x0F0F0F0Fu);
        }
    }
    else if constexpr (Traits::is_5bit)
    {
        // Q5_0 / Q5_1: nibble + high-bit from qh dword at payload[16..19]
        const uint32_t qh_bits = raw[4]; // 5th dword
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t lo_nibs = raw[g] & 0x0F0F0F0Fu;
            const uint32_t hi_nibs = (raw[g] >> 4) & 0x0F0F0F0Fu;
            const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
            const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                 | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
            const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
            const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                 | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
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
        const uint8_t* qh_p = reinterpret_cast<const uint8_t*>(raw) + 16;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t lo_nibs = raw[g] & 0x0F0F0F0Fu;
            const uint32_t hi_nibs = (raw[g] >> 4) & 0x0F0F0F0Fu;
            const uint32_t h_lo = qh_p[g], h_hi = qh_p[g + 4];
            const uint32_t hb_lo = ((h_lo & 0x03u) << 4)  | ((h_lo & 0x0Cu) << 10)
                                 | ((h_lo & 0x30u) << 16) | ((h_lo & 0xC0u) << 22);
            const uint32_t hb_hi = ((h_hi & 0x03u) << 4)  | ((h_hi & 0x0Cu) << 10)
                                 | ((h_hi & 0x30u) << 16) | ((h_hi & 0xC0u) << 22);
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
        // Q3_K: nibble-repacked → v_perm 8-entry codebook
        constexpr uint32_t Q3_CB_A = 0xFFFEFDFCu; // {-4,-3,-2,-1}
        constexpr uint32_t Q3_CB_B = 0x03020100u;  // {0,1,2,3}
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            packed_groups[g]     = static_cast<int32_t>(
                __builtin_amdgcn_perm(Q3_CB_B, Q3_CB_A, raw[g] & 0x0F0F0F0Fu));
            packed_groups[g + 4] = static_cast<int32_t>(
                __builtin_amdgcn_perm(Q3_CB_B, Q3_CB_A, (raw[g] >> 4) & 0x0F0F0F0Fu));
        }
    }
    else if constexpr (Traits::is_2bit)
    {
        // Q2_K: pre-transposed 2-bit extraction from 12B payload
        // [0..3] half0, [4..7] half1, [8..11] embedded mins
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
        if constexpr (FMT == NVNNI_GEMM_IQ1_S)
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
        else // IQ1_M
        {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw);
            const uint8_t qh0 = p[4], qh1 = p[5];
            const int idx0 = p[0] | ((qh0 & 0x07) << 8);
            const int idx1 = p[1] | (((qh0 >> 4) & 0x07) << 8);
            const int idx2 = p[2] | ((qh1 & 0x07) << 8);
            const int idx3 = p[3] | (((qh1 >> 4) & 0x07) << 8);
            uint64_t g;
            g = iq1_grid_lookup_gemm(idx0);
            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = iq1_grid_lookup_gemm(idx1);
            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = iq1_grid_lookup_gemm(idx2);
            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = iq1_grid_lookup_gemm(idx3);
            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
        }
    }
}


// ============================================================================
// Native-VNNI GEMM kernel: Decode-in-Load with INT32 block accumulators
//
// Architecture: LDS double-buffered pipeline (V3-derived)
//   - A tile: INT8 activations → LDS (identical to INT8 V3)
//   - B tile: raw Q4 payload loaded from HBM, decoded to INT8, stored in LDS
//   - Scales: FP16 per-block scales loaded into small LDS array
//   - Compute: INT32 accumulation within each block → FP32 scale at block exit
//
// Tile geometry (default S64, BPT=2):
//   N_TILE=64, BPT=2 (2 blocks/tile = 16 k-groups = 64 K-elements)
//   M_TILE=64, M_PER_THREAD=4, THREADS_M=16, THREADS_N=16, N_PER_THREAD=4
//   Grid: (ceil(N/64), ceil(M/64))
//
// LDS budget:
//   a_lds[2][16×64]  =  8192 bytes (A: same as V3)
//   b_lds[2][16×64]  =  8192 bytes (B: decoded INT8)
//   scale_lds[2][2×64] = 512 bytes (FP16 per-block scales)
//   Total: 16896 bytes ≈ 16.5 KB → 2 WGs/CU
//
// Key optimisation: INT32 block accumulation.
// Within each quantized block (8 k-groups), dot products accumulate
// directly into int32 via sdot4(a, b, block_acc). Scale conversion
// happens once per block (not per k-group), reducing inner-loop VALU
// from 3 instructions/dot to 1 instruction/dot — a 2.4× reduction.
// The K-step loop uses `#pragma clang loop unroll(disable)` to
// contain register pressure (matching V7's approach).
//
// Output: FP32 with scale_A applied inline. No separate epilogue needed.
// ============================================================================

// ============================================================================
// Per-block accumulate helper — handles four accumulation patterns:
//
//   Pattern S (simple symmetric):     acc += block_acc * scale
//   Pattern A (asymmetric):           acc += block_acc * scale + sum_a * aux
//   Pattern D (dual-scale symmetric): acc += acc_lo * scale_lo + acc_hi * aux_hi
//   Pattern DA (dual-scale + asym):   acc += acc_lo * scale + acc_hi * aux
//                                          + sum_a_lo * emin_lo + sum_a_hi * emin_hi
//
// Called once per block (bp) per tile to avoid triplication of compute code
// in safe/boundary/final loops.
// ============================================================================

template <int N_TILE, int BPT, int M_TILE, NativeVNNIGemmFormat FMT,
          int BLOCK_SIZE_ = 256, int M_PER_THREAD_ = 4>
__device__ __forceinline__ void accumulate_block_gemm(
    int bp,
    const int32_t* __restrict__ a_buf,
    const int32_t* __restrict__ b_buf,
    const uint16_t* __restrict__ s_buf,
    const uint16_t* __restrict__ aux_buf,
    const uint16_t* __restrict__ emin_lo_buf,
    const uint16_t* __restrict__ emin_hi_buf,
    float (&acc)[M_PER_THREAD_][N_TILE / (BLOCK_SIZE_ / (M_TILE / M_PER_THREAD_))],
    int m_base, int t_n)
{
    using Traits = NVNNIGemmTraits<FMT>;

    constexpr int M_PER_THREAD = M_PER_THREAD_;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE_ / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;

    if constexpr (Traits::is_dual_scale_asym)
    {
        // Pattern DA: dual-scale + asymmetric (Q2_K, IQ1_M)
        //
        // Two-phase accumulation to reduce VGPR pressure:
        // Instead of maintaining acc_lo/acc_hi/sum_a_lo/sum_a_hi simultaneously
        // (24 VGPRs at N128/M16), we process each half and apply its scale
        // immediately, reusing the same temp arrays (12 VGPRs).  Saves 12 VGPRs
        // which eliminates scratch spilling on gfx906.
        int32_t temp_acc[M_PER_THREAD][N_PER_THREAD];
        int32_t temp_sum[M_PER_THREAD];

        // ---- Lo half (groups 0-3): accumulate then apply scale_lo + emin_lo ----
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            temp_sum[mm] = 0;
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                temp_acc[mm][nn] = 0;
        }
        #pragma clang loop unroll_count(2)
        for (int g = 0; g < 4; ++g)
        {
            const int kk = bp * 8 + g;
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                temp_sum[mm] = __builtin_amdgcn_sdot4(
                    a_reg[mm], static_cast<int32_t>(0x01010101), temp_sum[mm], false);
            }
            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    temp_acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], temp_acc[mm][nn], false);
        }
        // Apply lo scale immediately → temp_acc/temp_sum are dead after this
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
            const float scale_lo = __half2float(*reinterpret_cast<const __half*>(&s_buf[sn]));
            const float emin_lo  = __half2float(*reinterpret_cast<const __half*>(&emin_lo_buf[sn]));
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                acc[mm][nn] = __fmaf_rn(static_cast<float>(temp_acc[mm][nn]), scale_lo, acc[mm][nn]);
                acc[mm][nn] = __fmaf_rn(static_cast<float>(temp_sum[mm]), emin_lo, acc[mm][nn]);
            }
        }

        // ---- Hi half (groups 4-7): reuse temp arrays, apply scale_hi + emin_hi ----
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            temp_sum[mm] = 0;
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                temp_acc[mm][nn] = 0;
        }
        #pragma clang loop unroll_count(2)
        for (int g = 4; g < 8; ++g)
        {
            const int kk = bp * 8 + g;
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                temp_sum[mm] = __builtin_amdgcn_sdot4(
                    a_reg[mm], static_cast<int32_t>(0x01010101), temp_sum[mm], false);
            }
            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    temp_acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], temp_acc[mm][nn], false);
        }
        // Apply hi scale
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
            const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&aux_buf[sn]));
            const float emin_hi  = __half2float(*reinterpret_cast<const __half*>(&emin_hi_buf[sn]));
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                acc[mm][nn] = __fmaf_rn(static_cast<float>(temp_acc[mm][nn]), scale_hi, acc[mm][nn]);
                acc[mm][nn] = __fmaf_rn(static_cast<float>(temp_sum[mm]), emin_hi, acc[mm][nn]);
            }
        }
    }
    else if constexpr (Traits::is_dual_scale)
    {
        // Pattern D: dual-scale symmetric (Q6_K, Q3_K, IQ2_S, IQ2_XS)
        //
        // Two accumulation strategies based on N_PER_THREAD:
        //
        // N_PER_THREAD >= 4 (N128/M32 tile): FUSED inline-scaled accumulation.
        //   Eliminates temp_acc entirely by scaling each dot4 result into FP32
        //   acc immediately.  Saves 16 VGPRs (128+scratch → 92+0 on gfx906).
        //   Extra VALU: +6% from per-dot4 cvt+fma.  Net win due to scratch
        //   elimination (+15-25% for heavy-scratch formats like Q6_K).
        //
        // N_PER_THREAD < 4 (M16, N64/M32 tiles): TWO-PHASE accumulation.
        //   temp_acc[M_PER_THREAD][N_PER_THREAD] fits within 128 VGPRs without
        //   scratch (8 int32 at M16/N128, 8 at N64/M32).  Avoids the fused
        //   approach's extra VALU overhead where it's not needed.
        if constexpr (N_PER_THREAD >= 4)
        {
            // ---- FUSED path (N128/M32): no temp_acc, scale per-dot4 ----
            //
            // Register pressure (N128/M32, M_PER_THREAD=4, N_PER_THREAD=4):
            //   acc[4][4]=16 + scale[4]=4 + a_reg[4]+b_reg[4]=8 = 28 → 92 VGPRs
            //
            // Precision: dot4 result (±65024) exact in FP32.  Accumulating 4
            // scaled FP32 vs scaling int32 sum once: ~3 ULP per phase, negligible.

            // Lo half (groups 0-3): fused scale_lo into acc
            float scale_lo[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
                scale_lo[nn] = __half2float(
                    *reinterpret_cast<const __half*>(&s_buf[sn]));
            }
            #pragma clang loop unroll(disable)
            for (int g = 0; g < 4; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        acc[mm][nn] = __fmaf_rn(
                            static_cast<float>(__builtin_amdgcn_sdot4(
                                a_reg[mm], b_reg[nn], 0, false)),
                            scale_lo[nn], acc[mm][nn]);
            }

            // Hi half (groups 4-7): fused scale_hi into acc
            float scale_hi[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
                scale_hi[nn] = __half2float(
                    *reinterpret_cast<const __half*>(&aux_buf[sn]));
            }
            #pragma clang loop unroll(disable)
            for (int g = 4; g < 8; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        acc[mm][nn] = __fmaf_rn(
                            static_cast<float>(__builtin_amdgcn_sdot4(
                                a_reg[mm], b_reg[nn], 0, false)),
                            scale_hi[nn], acc[mm][nn]);
            }
        }
        else
        {
            // ---- TWO-PHASE path (M16/N64): temp_acc fits in 128 VGPRs ----
            int32_t temp_acc[M_PER_THREAD][N_PER_THREAD];

            // Lo half (groups 0-3)
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    temp_acc[mm][nn] = 0;
            #pragma clang loop unroll(disable)
            for (int g = 0; g < 4; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        temp_acc[mm][nn] = __builtin_amdgcn_sdot4(
                            a_reg[mm], b_reg[nn], temp_acc[mm][nn], false);
            }
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
                const float scale_lo = __half2float(
                    *reinterpret_cast<const __half*>(&s_buf[sn]));
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    acc[mm][nn] = __fmaf_rn(
                        static_cast<float>(temp_acc[mm][nn]),
                        scale_lo, acc[mm][nn]);
            }

            // Hi half (groups 4-7)
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    temp_acc[mm][nn] = 0;
            #pragma clang loop unroll(disable)
            for (int g = 4; g < 8; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        temp_acc[mm][nn] = __builtin_amdgcn_sdot4(
                            a_reg[mm], b_reg[nn], temp_acc[mm][nn], false);
            }
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
                const float scale_hi = __half2float(
                    *reinterpret_cast<const __half*>(&aux_buf[sn]));
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    acc[mm][nn] = __fmaf_rn(
                        static_cast<float>(temp_acc[mm][nn]),
                        scale_hi, acc[mm][nn]);
            }
        }
    }
    else if constexpr (Traits::is_asymmetric)
    {
        // Pattern A: asymmetric single-scale (Q4_1, Q5_1, IQ1_S)
        int32_t block_acc[M_PER_THREAD][N_PER_THREAD];
        int32_t sum_a[M_PER_THREAD];
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            sum_a[mm] = 0;
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                block_acc[mm][nn] = 0;
        }
        #pragma clang loop unroll_count(4)
        for (int g = 0; g < 8; ++g)
        {
            const int kk = bp * 8 + g;
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                sum_a[mm] = __builtin_amdgcn_sdot4(
                    a_reg[mm], static_cast<int32_t>(0x01010101), sum_a[mm], false);
            }
            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    block_acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], block_acc[mm][nn], false);
        }
        // FMA: acc += block_acc * scale + sum_a * aux(min)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
            const float block_scale = __half2float(*reinterpret_cast<const __half*>(&s_buf[sn]));
            const float block_min   = __half2float(*reinterpret_cast<const __half*>(&aux_buf[sn]));
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                acc[mm][nn] = __fmaf_rn(static_cast<float>(block_acc[mm][nn]), block_scale, acc[mm][nn]);
                acc[mm][nn] = __fmaf_rn(static_cast<float>(sum_a[mm]), block_min, acc[mm][nn]);
            }
        }
    }
    else
    {
        // Pattern S: simple symmetric (Q4_0, IQ4_NL, Q5_0, IQ3_S, IQ3_XXS, IQ2_XXS)
        int32_t block_acc[M_PER_THREAD][N_PER_THREAD];
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                block_acc[mm][nn] = 0;

        #pragma clang loop unroll_count(4)
        for (int g = 0; g < 8; ++g)
        {
            const int kk = bp * 8 + g;
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    block_acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], block_acc[mm][nn], false);
        }
        // FMA: acc += block_acc * scale
        float block_scale[N_PER_THREAD];
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const uint16_t s_bits = s_buf[bp * N_TILE + t_n * N_PER_THREAD + nn];
            block_scale[nn] = __half2float(*reinterpret_cast<const __half*>(&s_bits));
        }
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                acc[mm][nn] = __fmaf_rn(
                    static_cast<float>(block_acc[mm][nn]), block_scale[nn], acc[mm][nn]);
    }
}

template <int N_TILE, int BPT, int M_TILE, NativeVNNIGemmFormat FMT, int MIN_BLOCKS = 2>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void native_vnni_gemm_kernel(
    const int8_t*    __restrict__ d_A_int8,       // [M × K] INT8 activations, row-major
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × PB] native-VNNI payload
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N] FP16 per-block scales
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N] FP16 per-block min/aux
    float*           __restrict__ d_output,       // [M × N] FP32 output, row-major
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    int M, int N, int K)
{
    // -------------------------------------------------------------------------
    // Tile and thread geometry
    // -------------------------------------------------------------------------
    constexpr int KT = BPT * 8;   // k-groups per tile (each k-group = 4 INT8 elements)
    constexpr int BLOCK_SIZE = 256;
    constexpr int M_PER_THREAD = 4;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;

    using Traits = NVNNIGemmTraits<FMT>;
    constexpr int PB = Traits::payload_bytes;
    constexpr int PD = Traits::payload_dwords;

    static_assert(M_TILE == 16 || M_TILE == 32 || M_TILE == 64 || M_TILE == 128,
                  "M_TILE must be 16, 32, 64, or 128");
    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT >= 8, "Need at least 1 block per tile");

    // Workgroup tile position
    const int n_block = blockIdx.x * N_TILE;
    const int m_block = blockIdx.y * M_TILE;

    // Thread position within tile
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];       // A: same layout as INT8 V3
    __shared__ int32_t b_lds[2][KT * N_TILE];       // B: decoded INT8 (same layout as V3)
    __shared__ uint16_t scale_lds[2][BPT * N_TILE];  // Per-block FP16 scales
    __shared__ uint16_t aux_lds[2][Traits::needs_aux_compute ? BPT * N_TILE : 1];  // Aux scale/min
    __shared__ uint16_t emin_lo_lds[2][Traits::is_dual_scale_asym ? BPT * N_TILE : 1]; // Embedded min lo
    __shared__ uint16_t emin_hi_lds[2][Traits::is_dual_scale_asym ? BPT * N_TILE : 1]; // Embedded min hi

    // FP32 accumulators — accumulate across all K-blocks
    float acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0.0f;

    // Pre-load per-row activation scales (one per M-row this thread owns)
    float scale_a[M_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int gm = m_block + m_base + mm;
        scale_a[mm] = (gm < M) ? d_scales_A[gm] : 0.0f;
    }

    // K dimensions
    const int blocks_per_row = K / 32;
    const int total_k_groups = K / 4;
    const int num_block_tiles = (blocks_per_row + BPT - 1) / BPT;

    if (num_block_tiles == 0) return;

    // =========================================================================
    // Safe-tile split (mirrors INT8 V7's branchless interior optimization)
    //
    // Pre-compute how many tiles have ALL loads guaranteed in-bounds:
    //   - a_m_safe: all M-rows in this workgroup tile are within [0, M)
    //   - b_n_safe: all N-cols in this workgroup tile are within [0, N)
    //   - safe_k:   number of tiles with full K coverage (no partial blocks)
    //
    // For tiles 0..num_safe-1, ALL global loads (A and B) are unconditional
    // → no branch overhead in the hot loop → better IPC and icache utilization.
    // For aligned shapes (common in transformer layers), num_safe == num_block_tiles
    // → the boundary loop never executes.
    // =========================================================================
    const bool a_m_safe = (m_block + M_TILE <= M);
    const bool b_n_safe = (n_block + N_TILE <= N);
    const int  safe_k   = blocks_per_row / BPT;  // tiles with full K coverage
    const int  num_safe = (a_m_safe && b_n_safe)
                        ? (safe_k < num_block_tiles ? safe_k : num_block_tiles)
                        : 0;

    // Reinterpret A as int32_t* for vectorized loads (4 packed INT8 per int32)
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int a_row_stride = K / 4;  // k-groups per A row

    // =========================================================================
    // A cooperative load geometry (identical to INT8 V3)
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry — Decode-in-Load
    //
    // First BPT×N_TILE threads load raw payload (PB bytes each) and scales.
    // Thread mapping: (block_local, n_local) via div/mod N_TILE.
    //
    // Each thread:
    //   1. Loads PB bytes of raw payload via load_payload<PB>()
    //   2. Loads 1 FP16 scale (from d_block_scales or embedded in payload)
    //   3. Optionally loads 1 FP16 aux (from d_block_mins or payload)
    //   4. Decodes payload → 8 packed INT32 groups
    //   5. Writes 8 int32 to b_lds + scales/aux/emin to LDS
    // =========================================================================
    constexpr int B_LOAD_THREADS = BPT * N_TILE;  // 128 for BPT=2, N_TILE=64

    const int b_load_block = threadIdx.x / N_TILE;  // which block in tile (0..BPT-1)
    const int b_load_n     = threadIdx.x % N_TILE;  // which N-column (tile-local)
    const int b_load_n_global = n_block + b_load_n;  // global N-column

    // Staging registers for pipeline
    uint4 a_staging[A_NUM_PASSES];
    uint32_t b_payload_dw[PD];  // variable-size payload dwords
    uint16_t b_scale_staging;   // 1 FP16 scale
    uint16_t b_aux_staging;     // 1 FP16 aux scale/min (for needs_aux_compute)
    uint16_t b_emin_lo_staging; // embedded min lo (for dual_scale_asym)
    uint16_t b_emin_hi_staging; // embedded min hi (for dual_scale_asym)

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    {
        const bool tile0_safe = (num_safe > 0);

        // A loads
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;
            if (tile0_safe)
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(gm) * a_row_stride + a_kg);
            }
            else if (__builtin_expect(gm < M && a_kg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(gm) * a_row_stride + a_kg);
            }
            else
            {
                a_staging[p].x = (gm < M && a_kg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 0] : 0;
                a_staging[p].y = (gm < M && a_kg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 1] : 0;
                a_staging[p].z = (gm < M && a_kg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 2] : 0;
                a_staging[p].w = (gm < M && a_kg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 3] : 0;
            }
        }

        // B loads
        #pragma unroll
        for (int d = 0; d < PD; ++d) b_payload_dw[d] = 0;
        b_scale_staging = 0;
        if constexpr (Traits::needs_aux_compute) b_aux_staging = 0;
        if constexpr (Traits::is_dual_scale_asym) { b_emin_lo_staging = 0; b_emin_hi_staging = 0; }
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = b_load_block;
            if (tile0_safe || __builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                load_payload<PB>(d_payload + linear * PB, b_payload_dw);
                const uint8_t* pbytes = reinterpret_cast<const uint8_t*>(b_payload_dw);
                if constexpr (Traits::has_embedded_scales)
                    b_scale_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_scale_offset);
                else
                    b_scale_staging = d_block_scales[linear];
                if constexpr (Traits::loads_aux_from_global)
                    b_aux_staging = d_block_mins[linear];
                else if constexpr (Traits::needs_aux_compute && Traits::has_embedded_scales) {
                    if constexpr (FMT == NVNNI_GEMM_IQ1_S)
                        b_aux_staging = *reinterpret_cast<const uint16_t*>(pbytes + 8);
                    else if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                        b_aux_staging = *reinterpret_cast<const uint16_t*>(pbytes + 12);
                }
                if constexpr (Traits::is_dual_scale_asym) {
                    b_emin_lo_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset);
                    b_emin_hi_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset + 2);
                }
            }
        }
    }

    // Write preload to a_lds[0] + b_lds[0]
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
    {
        const int mi = a_mi[p];
        if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
        a_lds[0][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
        a_lds[0][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
        a_lds[0][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
        a_lds[0][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
    }
    if (threadIdx.x < B_LOAD_THREADS)
    {
        int32_t packed_groups[8];
        decode_block<FMT>(b_payload_dw, packed_groups);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            b_lds[0][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

        scale_lds[0][b_load_block * N_TILE + b_load_n] = b_scale_staging;
        if constexpr (Traits::needs_aux_compute)
            aux_lds[0][b_load_block * N_TILE + b_load_n] = b_aux_staging;
        if constexpr (Traits::is_dual_scale_asym) {
            emin_lo_lds[0][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
            emin_hi_lds[0][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
        }
    }
    __syncthreads();  // buf[0] ready

    // =========================================================================
    // SAFE LOOP: tiles 1..num_safe-1  (unconditional loads — no bounds checks)
    //
    // Split into a separate loop from the boundary path so the compiler can
    // fully optimize the load phase: no dead branches, no speculative zero-
    // init, better instruction scheduling.  For aligned shapes (common in
    // transformer layers), num_safe == num_block_tiles → boundary loop is
    // never entered.
    //
    // Note: despite the code duplication of the compute + LDS-write phases,
    // benchmarks show this is ~10-20% faster than a single loop with a
    // runtime branch (the branch prevents the compiler from optimizing the
    // safe load path aggressively).
    // =========================================================================
    for (int tile = 1; tile < num_safe; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int tile_block_base = tile * BPT;
        const int kt_base = tile_block_base * 8;

        // --- 1. SAFE LOAD: all loads guaranteed in-bounds ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;
            const int gkg = kt_base + a_kg;
            a_staging[p] = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(gm) * a_row_stride + gkg);
        }

        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = tile_block_base + b_load_block;
            const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
            load_payload<PB>(d_payload + linear * PB, b_payload_dw);
            const uint8_t* pbytes = reinterpret_cast<const uint8_t*>(b_payload_dw);
            if constexpr (Traits::has_embedded_scales)
                b_scale_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_scale_offset);
            else
                b_scale_staging = d_block_scales[linear];
            if constexpr (Traits::loads_aux_from_global)
                b_aux_staging = d_block_mins[linear];
            else if constexpr (Traits::needs_aux_compute && Traits::has_embedded_scales) {
                if constexpr (FMT == NVNNI_GEMM_IQ1_S)
                    b_aux_staging = *reinterpret_cast<const uint16_t*>(pbytes + 8);
                else if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                    b_aux_staging = *reinterpret_cast<const uint16_t*>(pbytes + 12);
            }
            if constexpr (Traits::is_dual_scale_asym) {
                b_emin_lo_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset);
                b_emin_hi_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset + 2);
            }
        }

        // --- 2. Compute on CURRENT buffer ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT>(
                bp, a_lds[cur_buf], b_lds[cur_buf],
                scale_lds[cur_buf], aux_lds[cur_buf],
                emin_lo_lds[cur_buf], emin_hi_lds[cur_buf],
                acc, m_base, t_n);

        // --- 3. Barrier: all reads from cur_buf done ---
        __syncthreads();

        // --- 4. Decode + write staging → next buffer ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }

        if (threadIdx.x < B_LOAD_THREADS)
        {
            int32_t packed_groups[8];
            decode_block<FMT>(b_payload_dw, packed_groups);

            #pragma unroll
            for (int g = 0; g < 8; ++g)
                b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

            scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
            if constexpr (Traits::needs_aux_compute)
                aux_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_aux_staging;
            if constexpr (Traits::is_dual_scale_asym) {
                emin_lo_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
                emin_hi_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
            }
        }

        // --- 5. Barrier: LDS writes visible ---
        __syncthreads();
    }

    // =========================================================================
    // BOUNDARY LOOP: tiles max(1,num_safe)..num_block_tiles-1  (checked loads)
    //
    // Handles edge tiles where M, N, or K dimensions may be out of bounds.
    // Identical compute and LDS-write phases; only the load phase adds bounds
    // checks and zero-initialization for partial tiles.
    // =========================================================================
    for (int tile = (num_safe > 1 ? num_safe : 1); tile < num_block_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int tile_block_base = tile * BPT;
        const int kt_base = tile_block_base * 8;

        // --- 1. BOUNDARY LOAD: with bounds checks ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(gm < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(gm) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (gm < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (gm < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (gm < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (gm < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 3] : 0;
            }
        }

        #pragma unroll
        for (int d = 0; d < PD; ++d) b_payload_dw[d] = 0;
        b_scale_staging = 0;
        if constexpr (Traits::needs_aux_compute) b_aux_staging = 0;
        if constexpr (Traits::is_dual_scale_asym) { b_emin_lo_staging = 0; b_emin_hi_staging = 0; }
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = tile_block_base + b_load_block;
            if (__builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                load_payload<PB>(d_payload + linear * PB, b_payload_dw);
                const uint8_t* pbytes = reinterpret_cast<const uint8_t*>(b_payload_dw);
                if constexpr (Traits::has_embedded_scales)
                    b_scale_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_scale_offset);
                else
                    b_scale_staging = d_block_scales[linear];
                if constexpr (Traits::loads_aux_from_global)
                    b_aux_staging = d_block_mins[linear];
                else if constexpr (Traits::needs_aux_compute && Traits::has_embedded_scales) {
                    if constexpr (FMT == NVNNI_GEMM_IQ1_S)
                        b_aux_staging = *reinterpret_cast<const uint16_t*>(pbytes + 8);
                    else if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                        b_aux_staging = *reinterpret_cast<const uint16_t*>(pbytes + 12);
                }
                if constexpr (Traits::is_dual_scale_asym) {
                    b_emin_lo_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset);
                    b_emin_hi_staging = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset + 2);
                }
            }
        }

        // --- 2. Compute on CURRENT buffer ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT>(
                bp, a_lds[cur_buf], b_lds[cur_buf],
                scale_lds[cur_buf], aux_lds[cur_buf],
                emin_lo_lds[cur_buf], emin_hi_lds[cur_buf],
                acc, m_base, t_n);

        // --- 3. Barrier: all reads from cur_buf done ---
        __syncthreads();

        // --- 4. Decode + write staging → next buffer ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }

        if (threadIdx.x < B_LOAD_THREADS)
        {
            int32_t packed_groups[8];
            decode_block<FMT>(b_payload_dw, packed_groups);

            #pragma unroll
            for (int g = 0; g < 8; ++g)
                b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

            scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
            if constexpr (Traits::needs_aux_compute)
                aux_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_aux_staging;
            if constexpr (Traits::is_dual_scale_asym) {
                emin_lo_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
                emin_hi_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
            }
        }

        // --- 5. Barrier: LDS writes visible ---
        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE: Compute the last tile (already in LDS)
    // =========================================================================
    {
        const int last_buf = (num_block_tiles - 1) & 1;

        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT>(
                bp, a_lds[last_buf], b_lds[last_buf],
                scale_lds[last_buf], aux_lds[last_buf],
                emin_lo_lds[last_buf], emin_hi_lds[last_buf],
                acc, m_base, t_n);
    }

    // =========================================================================
    // STORE RESULTS: Apply scale_A and write FP32 to global memory
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;
        if (m_idx >= M) continue;

        const float sa = scale_a[mm];

        if constexpr (N_PER_THREAD >= 4)
        {
            // Vectorized store path
            if (n_base + N_PER_THREAD - 1 < N)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    float4 out;
                    out.x = acc[mm][chunk + 0] * sa;
                    out.y = acc[mm][chunk + 1] * sa;
                    out.z = acc[mm][chunk + 2] * sa;
                    out.w = acc[mm][chunk + 3] * sa;
                    *reinterpret_cast<float4*>(d_output + row_off + chunk) = out;
                }
            }
            else
            {
                // Scalar fallback for boundary columns
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                        d_output[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn] * sa;
                }
            }
        }
        else
        {
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const int n_idx = n_base + nn;
                if (n_idx < N)
                    d_output[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn] * sa;
            }
        }
    }
}


// ============================================================================
// Streaming LM_Head kernel: barrier-free, LDS-free register streaming
//
// Architecture: Per-thread independent streaming through K dimension.
// Each thread computes M_TILE M-rows × 1 N-column with zero synchronization.
//
// Why this is faster for LM_Head (N=151936, K=896-2048):
//   1. LM_Head is 100% COMPUTE-BOUND (BW util 5-25% at 1000 GB/s)
//   2. The cooperative LDS kernel spends ~50% of time in barriers
//      (28-64 barriers per WG at ~150 cyc each vs ~5000-10000 VALU)
//   3. Decode VALU cannot overlap with compute VALU (same pipe)
//   4. BW savings from 4.5 bpw are stranded (not BW-limited)
//
// This kernel eliminates ALL barriers and ALL LDS usage:
//   - B: loaded per-thread from global memory → decode in registers
//   - A: loaded per-thread, all 256 threads read same addresses → L2 broadcast
//   - Zero __syncthreads → VALU pipeline runs uninterrupted
//   - ~50 VGPRs → 3-wave occupancy (vs 2-wave for LDS kernel)
//
// Thread geometry (M_TILE=16):
//   THREADS_M = 1 (each thread owns all M_TILE M-rows)
//   THREADS_N = 256 (each thread owns 1 N-column)
//   N_PER_WG = 256
//   Zero B decode redundancy (each B block decoded exactly once)
//
// B data redundancy (across m-tiles): absorbed by L2 cache (4MB).
// A data broadcast: 256 threads read same address → L0/L1 cache broadcast.
// ============================================================================

template <int M_TILE, NativeVNNIGemmFormat FMT, int MIN_BLOCKS = 3>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void streaming_lm_head_kernel(
    const int8_t*    __restrict__ d_A_int8,       // [M × K] INT8 activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × PB] payload
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N] FP16 scales
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N] FP16 min/aux
    float*           __restrict__ d_output,       // [M × N] FP32 output
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    int M, int N, int K)
{
    static_assert(M_TILE == 16 || M_TILE == 32,
                  "Streaming LM_Head kernel supports M_TILE 16 or 32");

    using Traits = NVNNIGemmTraits<FMT>;
    constexpr int PB = Traits::payload_bytes;

    // Thread position: 1 thread = M_TILE M-rows × 1 N-column
    const int n_col   = blockIdx.x * 256 + threadIdx.x;
    const int m_block = blockIdx.y * M_TILE;

    if (n_col >= N) return;

    // Reinterpret A as int32 (4 packed INT8 per int32 k-group)
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int a_row_stride  = K / 4;   // k-groups per row
    const int blocks_per_row = K / 32;  // quantized blocks per row

    // FP32 accumulators — accumulated across all K-blocks
    float acc[M_TILE];
    #pragma unroll
    for (int mm = 0; mm < M_TILE; ++mm)
        acc[mm] = 0.0f;

    // Bounds: if entire M_TILE is within [0, M), skip per-element checks
    const bool m_safe = (m_block + M_TILE <= M);

    // =========================================================================
    // Main loop: stream through K-blocks with zero barriers
    // =========================================================================
    for (int block = 0; block < blocks_per_row; ++block)
    {
        // --- 1. Load B payload + scale/aux for this thread's N-column ---
        const size_t linear = static_cast<size_t>(block) * N + n_col;
        uint32_t b_payload_dw[Traits::payload_dwords];
        load_payload<PB>(d_payload + linear * PB, b_payload_dw);
        const uint8_t* pbytes = reinterpret_cast<const uint8_t*>(b_payload_dw);

        uint16_t scale_bits;
        if constexpr (Traits::has_embedded_scales)
            scale_bits = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_scale_offset);
        else
            scale_bits = d_block_scales[linear];

        uint16_t aux_bits = 0;
        if constexpr (Traits::loads_aux_from_global)
            aux_bits = d_block_mins[linear];
        else if constexpr (Traits::needs_aux_compute && Traits::has_embedded_scales) {
            if constexpr (FMT == NVNNI_GEMM_IQ1_S)
                aux_bits = *reinterpret_cast<const uint16_t*>(pbytes + 8);
            else if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                aux_bits = *reinterpret_cast<const uint16_t*>(pbytes + 12);
        }

        uint16_t emin_lo_bits = 0, emin_hi_bits = 0;
        if constexpr (Traits::is_dual_scale_asym) {
            emin_lo_bits = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset);
            emin_hi_bits = *reinterpret_cast<const uint16_t*>(pbytes + Traits::embedded_min_offset + 2);
        }

        // --- 2. Decode B → 8 INT32 packed groups ---
        int32_t packed[8];
        decode_block<FMT>(b_payload_dw, packed);

        // --- 3. Accumulate based on format pattern ---
        if constexpr (Traits::is_dual_scale_asym)
        {
            // Pattern DA: two-phase accumulation (Q2_K, IQ1_M)
            const int kg_base = block * 8;

            // Lo half
            int32_t temp_acc[M_TILE], temp_sum[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                temp_acc[mm] = 0; temp_sum[mm] = 0;
            }
            #pragma unroll
            for (int kg = 0; kg < 4; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = m_block + mm;
                    const int32_t a_val = (m_safe || gm < M)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    temp_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, temp_acc[mm], false);
                    temp_sum[mm] = __builtin_amdgcn_sdot4(
                        a_val, static_cast<int32_t>(0x01010101), temp_sum[mm], false);
                }
            }
            const float slo = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float elo = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]), slo, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(temp_sum[mm]), elo, acc[mm]);
            }

            // Hi half (reuse temp_acc, temp_sum)
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                temp_acc[mm] = 0; temp_sum[mm] = 0;
            }
            #pragma unroll
            for (int kg = 4; kg < 8; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = m_block + mm;
                    const int32_t a_val = (m_safe || gm < M)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    temp_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, temp_acc[mm], false);
                    temp_sum[mm] = __builtin_amdgcn_sdot4(
                        a_val, static_cast<int32_t>(0x01010101), temp_sum[mm], false);
                }
            }
            const float shi = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            const float ehi = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]), shi, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(temp_sum[mm]), ehi, acc[mm]);
            }
        }
        else if constexpr (Traits::is_dual_scale)
        {
            // Pattern D: two-phase accumulation (Q6_K, Q3_K, IQ2_S, IQ2_XS)
            const int kg_base = block * 8;

            // Lo half
            int32_t temp_acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                temp_acc[mm] = 0;
            #pragma unroll
            for (int kg = 0; kg < 4; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = m_block + mm;
                    const int32_t a_val = (m_safe || gm < M)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    temp_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, temp_acc[mm], false);
                }
            }
            const float slo = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]), slo, acc[mm]);

            // Hi half (reuse temp_acc)
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                temp_acc[mm] = 0;
            #pragma unroll
            for (int kg = 4; kg < 8; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = m_block + mm;
                    const int32_t a_val = (m_safe || gm < M)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    temp_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, temp_acc[mm], false);
                }
            }
            const float shi = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]), shi, acc[mm]);
        }
        else if constexpr (Traits::is_asymmetric)
        {
            // Pattern A: asymmetric single-scale (Q4_1, Q5_1, IQ1_S)
            int32_t block_acc[M_TILE], sum_a[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                block_acc[mm] = 0; sum_a[mm] = 0;
            }
            const int kg_base = block * 8;
            #pragma unroll
            for (int kg = 0; kg < 8; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = m_block + mm;
                    const int32_t a_val = (m_safe || gm < M)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    block_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, block_acc[mm], false);
                    sum_a[mm] = __builtin_amdgcn_sdot4(
                        a_val, static_cast<int32_t>(0x01010101), sum_a[mm], false);
                }
            }
            const float bs = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float bm = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = __fmaf_rn(static_cast<float>(block_acc[mm]), bs, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(sum_a[mm]), bm, acc[mm]);
            }
        }
        else
        {
            // Pattern S: simple symmetric (Q4_0, IQ4_NL, Q5_0, ...)
            int32_t block_acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                block_acc[mm] = 0;
            const int kg_base = block * 8;
            if (m_safe)
            {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int32_t a_val = a_global[
                            static_cast<int64_t>(m_block + mm) * a_row_stride + a_kg_idx];
                        block_acc[mm] = __builtin_amdgcn_sdot4(
                            a_val, b_val, block_acc[mm], false);
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int gm = m_block + mm;
                        const int32_t a_val = (gm < M)
                            ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                        block_acc[mm] = __builtin_amdgcn_sdot4(
                            a_val, b_val, block_acc[mm], false);
                    }
                }
            }
            const float block_scale = __half2float(
                *reinterpret_cast<const __half*>(&scale_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                acc[mm] = __fmaf_rn(
                    static_cast<float>(block_acc[mm]), block_scale, acc[mm]);
        }
    }

    // =========================================================================
    // STORE: Apply scale_A and write FP32 output
    // =========================================================================
    if (m_safe)
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
        {
            const int gm = m_block + mm;
            d_output[static_cast<int64_t>(gm) * N + n_col] =
                acc[mm] * d_scales_A[gm];
        }
    }
    else
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
        {
            const int gm = m_block + mm;
            if (gm < M)
                d_output[static_cast<int64_t>(gm) * N + n_col] =
                    acc[mm] * d_scales_A[gm];
        }
    }
}


// ============================================================================
// Tile dispatch: select N_TILE (64 or 128) and M_TILE for minimal waste
//
// N_TILE selection mirrors INT8 V3/V7 strategy:
//   K >= N (K-heavy: AttnOut, FFN_Dn) → N_TILE=64  (deeper K iteration)
//   K <  N (N-heavy: FFN_Up, LM_Head) → N_TILE=128 (wider N coverage)
//
// N_TILE=128 benefits for N-heavy shapes:
//   - ALL 256 threads participate in B cooperative loading (vs 128 at N64)
//   - Half the grid in N dimension → fewer kernel launches + barriers
//   - N_PER_THREAD doubles (4→8 at M64) → better compute density
//   - LDS budget: 25.6KB at M64 → 2 WGs/CU, occupancy 2 maintained
//
// M_TILE selection uses V7's minimum-waste strategy: pick the M_TILE
// that minimizes padding waste for the given M value.
// ============================================================================

template <NativeVNNIGemmFormat FMT>
static bool launch_native_vnni_gemm(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    const uint16_t* d_block_mins,
    float* d_output,
    const float* d_scales_A,
    int M, int N, int K,
    int device_id, void* stream)
{
    constexpr int BPT = 2;
    constexpr int BLOCK_SIZE = 256;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // Templated launch helper (default 2-wave)
    auto launch = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);

        native_vnni_gemm_kernel<NT, BPT, MT, FMT>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins,
                d_output, d_scales_A, M, N, K);

        return hipGetLastError() == hipSuccess;
    };

    // 3-wave variant: MIN_BLOCKS=3 tells compiler to target ≤84 VGPRs
    auto launch3 = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);

        native_vnni_gemm_kernel<NT, BPT, MT, FMT, 3>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins,
                d_output, d_scales_A, M, N, K);

        return hipGetLastError() == hipSuccess;
    };

    // Streaming LM_Head variant: zero barriers, zero LDS, 3-wave occupancy
    auto launch_streaming = [&](auto m_tile_tag) -> bool
    {
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr int N_PER_WG = 256;  // each thread handles 1 N-column
        dim3 grid((N + N_PER_WG - 1) / N_PER_WG, (M + MT - 1) / MT, 1);
        dim3 block(256, 1, 1);

        streaming_lm_head_kernel<MT, FMT>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins,
                d_output, d_scales_A, M, N, K);

        return hipGetLastError() == hipSuccess;
    };

    // M_TILE selection: minimize row waste (same as INT8 V7)
    auto select_m_tile = [](int rows) -> int {
        int best_mt = 64;
        int best_waste = ((rows + 63) / 64) * 64 - rows;
        for (int mt : {32, 16}) {
            int waste = ((rows + mt - 1) / mt) * mt - rows;
            if (waste < best_waste) {
                best_waste = waste;
                best_mt = mt;
            }
        }
        return best_mt;
    };

    const int m_tile = (M <= 64) ? select_m_tile(M) : 64;

    // N_TILE selection: N-heavy → N128 (wider coverage), K-heavy → N64 (deeper K)
    //
    // N128 dispatch uses a HYBRID tile strategy:
    //   - M16 tiles with 3-wave occupancy (84 VGPRs) when GPU is undersaturated
    //   - M32 tiles with 2-wave occupancy (128 VGPRs) when GPU is saturated
    //
    // M16/3-wave halves accumulators (16→8/thread) for 50% more waves/SIMD.
    // This wins when total WG count is small (GPU can't fill all SIMDs at
    // 2-wave). For large grids, M32/2-wave has 2× better A-tile reuse per
    // WG and dominates once all SIMDs are already occupied.
    //
    // Threshold: total M16 WGs ≤ 256 (~1 WG/SIMD on MI50/60: 60 CUs × 4).
    // Below this, 3-wave occupancy provides the biggest speedup over INT8.
    // Above this, M32 tile efficiency wins despite lower occupancy.
    const bool use_n128 = (N > K);

    // Dispatch tags for (N_TILE, M_TILE) combinations
    using N64  = std::integral_constant<int, 64>;
    using N128 = std::integral_constant<int, 128>;
    using M16  = std::integral_constant<int, 16>;
    using M32  = std::integral_constant<int, 32>;
    using M64  = std::integral_constant<int, 64>;

    // =========================================================================
    // STREAMING DISPATCH: LM_Head-like shapes (massive N, moderate K)
    //
    // For shapes where N >> K and N is very large, the cooperative-LDS kernel
    // is compute-bound with ~50% time in barriers.  The streaming kernel
    // eliminates ALL barriers and LDS, achieving 3-wave occupancy.
    //
    // Condition: N >= 16384 captures LM_Head (N=151936) and large FFN_Up
    // (e.g. 7B N=18944).  K <= 4096 is a safety check.
    // =========================================================================
    if (N >= 16384 && K <= 4096) {
        return launch_streaming(M16{});
    }

    if (use_n128) {
        // Dual-scale-asymmetric formats (Pattern DA: Q2_K, IQ1_M) suffer from
        // catastrophic register pressure at the N128/M32 tile.  Even with
        // two-phase accumulation, Pattern DA needs temp_acc + temp_sum +
        // embedded min values + pipeline staging → 128 VGPRs + 680 bytes scratch.
        //
        // Force M16 for Pattern DA: achieves 2-wave with zero scratch.
        //
        // NOTE: Pattern D formats (Q6_K, Q3_K, IQ2_S, IQ2_XS) have scratch
        // spilling at N128/M32 (72-132 bytes) that can't be eliminated without
        // sacrificing tile efficiency.  Tried alternatives (all regressed):
        //   - N128/M16 for Pattern D: M16 tile loses 2× A-data reuse → -15%
        //   - N64/M32 for Pattern D: doubles A-data bandwidth → -10% at M≥128
        //   - Fully unroll(disable): partial help (+8-19%) but scratch persists
        // The M32/2-wave+scratch path remains the best compromise.
        if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale_asym) {
            return launch3(N128{}, M16{});
        }

        // Hybrid N128 dispatch: M16/3-wave for small grids, M32/2-wave for large
        //
        // NOTE: M_TILE > 32 is NOT viable on N128 due to A-LDS bank conflicts.
        // gfx906 has 32 LDS banks at 4-byte granularity.  With M_PER_THREAD=4,
        // each thread reads 4 consecutive int32 from banks (t_m*4)%32 through
        // (t_m*4+3)%32.  Bank groups wrap every 8 threads:
        //   THREADS_M=8  (M32):  8 unique bank groups → zero conflicts
        //   THREADS_M=16 (M64):  2-way conflicts → ~1.7× penalty
        //   THREADS_M=32 (M128): 4-way conflicts → ~2.5× penalty
        // The A-data reuse gain from N_PER_THREAD 4→8 is completely eaten by
        // the bank conflict stalls.  Empirically: M64/N128 and M128/N64 both
        // regressed LM_Head from 0.88× to 0.30-0.37× (2.5× slower).
        const int n128_blocks = (N + 127) / 128;
        const int m16_blocks  = (M + 15) / 16;
        const int total_wgs_m16 = n128_blocks * m16_blocks;

        if (total_wgs_m16 <= 256) {
            // GPU undersaturated: M16/3-wave → more waves hide latency
            return launch3(N128{}, M16{});
        } else {
            // GPU saturated: M32/2-wave → better tile efficiency
            const int m_tile_n128 = (m_tile <= 16) ? 16 : 32;
            if (m_tile_n128 <= 16)
                return launch3(N128{}, M16{});
            else
                return launch(N128{}, M32{});
        }
    } else {
        // Dual-scale-asymmetric formats (Pattern DA): force M16 on N64 path.
        // N64/M64 causes 652 bytes scratch spill for Pattern DA.
        if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale_asym) {
            return launch3(N64{}, M16{});
        }

        // N64 (K-heavy) dispatch with hybrid M16/3-wave for small-M shapes.
        //
        // Mirrors the N128 hybrid strategy: when the total M16 workgroup count
        // is ≤256 (~1 WG/SIMD on MI50/60), the M16/3-wave variant provides
        // better GPU coverage (more CUs active) despite lower per-tile compute
        // density.  This is critical for K-heavy shapes (FFN_Dn, AttnOut) at
        // small M where few N64 workgroups are launched.
        //
        // With M32/2-wave, e.g. 7B_AttnOut M=32 has only 56 WGs → 28 CUs (47%).
        // M16/3-wave doubles WGs to 112 → 37-56 CUs (62-93%) active.
        //
        // For M≥128, the waste-minimizing M64/M32 tiles already achieve good
        // GPU coverage (many WGs) and their superior A-data reuse dominates.
        //
        // Threshold is 128 WGs (vs 256 for N128) because N64 tiles have half
        // the column-output of N128 tiles, so per-WG compute density is lower.
        // Above 128 WGs the GPU has enough parallelism for M32/M64 tiles to
        // win via better A-data reuse.  Verified: keeps 0.5B/3B M=64 wins
        // (+0.02..+0.42) while avoiding 7B M=64 regressions (224 WGs > 128).
        if (M <= 64) {
            const int n64_blocks = (N + 63) / 64;
            const int m16_blocks = (M + 15) / 16;
            const int total_wgs_m16 = n64_blocks * m16_blocks;

            if (total_wgs_m16 <= 128) {
                return launch3(N64{}, M16{});
            }
        }

        // Dual-scale formats (Pattern D): cap tile at M32 on N64 path.
        // N64/M64 causes 124 bytes scratch spill for Pattern D (M_PER_THREAD=4
        // with two-phase still needs too many registers at THREADS_M=16).
        int effective_m_tile = m_tile;
        if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale) {
            effective_m_tile = (effective_m_tile > 32) ? 32 : effective_m_tile;
        }

        switch (effective_m_tile) {
            case 16:  return launch(N64{}, M16{});
            case 32:  return launch(N64{}, M32{});
            default:  return launch(N64{}, M64{});
        }
    }
}


// ============================================================================
// C API: Format-dispatched native-VNNI GEMM
//
// Mirrors rocmGemv_native_vnni_fp32() but for M>1 prefill.
// Output is FP32 with scale_A applied inline — no separate epilogue.
// ============================================================================

extern "C"
bool rocmGemm_native_vnni_fp32(
    const int8_t*  d_A_int8,       // [M × K] INT8 activations
    const uint8_t* d_payload,      // [blocks_per_row × N × PB] native-VNNI payload
    const void*    d_block_scales, // [blocks_per_row × N] __half per-block scales
    const void*    d_block_mins,   // [blocks_per_row × N] __half per-block mins/aux (nullable)
    float*         d_output,       // [M × N] FP32 output
    const float*   d_scales_A,     // [M] per-row activation scales
    int M, int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || K % 32 != 0)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const auto* scales = reinterpret_cast<const uint16_t*>(d_block_scales);
    const auto* mins   = reinterpret_cast<const uint16_t*>(d_block_mins);

    #define LAUNCH_GEMM(FMT) \
        return launch_native_vnni_gemm<FMT>( \
            d_A_int8, d_payload, scales, mins, d_output, d_scales_A, \
            M, N, K, device_id, stream)

    switch (codebook_id)
    {
    case NVNNI_GEMM_Q4_0:     LAUNCH_GEMM(NVNNI_GEMM_Q4_0);
    case NVNNI_GEMM_IQ4_NL:   LAUNCH_GEMM(NVNNI_GEMM_IQ4_NL);
    case NVNNI_GEMM_Q4_1:     LAUNCH_GEMM(NVNNI_GEMM_Q4_1);
    case NVNNI_GEMM_Q5_0:     LAUNCH_GEMM(NVNNI_GEMM_Q5_0);
    case NVNNI_GEMM_Q5_1:     LAUNCH_GEMM(NVNNI_GEMM_Q5_1);
    case NVNNI_GEMM_Q6_K:     LAUNCH_GEMM(NVNNI_GEMM_Q6_K);
    case NVNNI_GEMM_Q3_K:     LAUNCH_GEMM(NVNNI_GEMM_Q3_K);
    case NVNNI_GEMM_Q2_K:     LAUNCH_GEMM(NVNNI_GEMM_Q2_K);
    case NVNNI_GEMM_IQ3_S:    LAUNCH_GEMM(NVNNI_GEMM_IQ3_S);
    case NVNNI_GEMM_IQ3_XXS:  LAUNCH_GEMM(NVNNI_GEMM_IQ3_XXS);
    case NVNNI_GEMM_IQ2_S:    LAUNCH_GEMM(NVNNI_GEMM_IQ2_S);
    case NVNNI_GEMM_IQ2_XS:   LAUNCH_GEMM(NVNNI_GEMM_IQ2_XS);
    case NVNNI_GEMM_IQ2_XXS:  LAUNCH_GEMM(NVNNI_GEMM_IQ2_XXS);
    case NVNNI_GEMM_IQ1_S:    LAUNCH_GEMM(NVNNI_GEMM_IQ1_S);
    case NVNNI_GEMM_IQ1_M:    LAUNCH_GEMM(NVNNI_GEMM_IQ1_M);
    default:
        fprintf(stderr, "[rocmGemm_native_vnni_fp32] Unsupported codebook_id=%d\n",
                codebook_id);
        return false;
    }

    #undef LAUNCH_GEMM
}


// ============================================================================
// IQ grid table initialization for GEMM translation unit.
// Each .hip file has its own __constant__ symbols; must be initialized
// separately from the GEMV init (rocmInitIQGridTables).
// ============================================================================

extern "C"
bool rocmInitIQGridTables_gemm(
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
