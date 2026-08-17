/**
 * @file ROCmNativeVNNIGemmShardImpl.hip.inc
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
 * @see docs/v2/projects/2026-03/NATIVE_VNNI_GEMM_DESIGN.md for architecture design document
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/NativeVnniFormatInfo.h"
#include "../../../utils/DebugEnv.h"
#include "ROCmNativeVNNIDecodeEquivalentMath.h"
#include "ROCmNativeVNNIGemmShard.h"

#ifndef LLAMINAR_ROCM_NATIVE_VNNI_SHARD
#error "ROCmNativeVNNIGemmShardImpl.hip.inc requires a physical shard index"
#endif

#if LLAMINAR_ROCM_NATIVE_VNNI_SHARD < 0 || LLAMINAR_ROCM_NATIVE_VNNI_SHARD > 7
#error "ROCm NativeVNNI GEMM shard index must be in [0, 7]"
#endif

#define LLAMINAR_ROCM_NVNNI_CONCAT_IMPL(A, B) A##B
#define LLAMINAR_ROCM_NVNNI_CONCAT(A, B) \
    LLAMINAR_ROCM_NVNNI_CONCAT_IMPL(A, B)
#define LLAMINAR_ROCM_NVNNI_SHARD_SYMBOL(PREFIX) \
    LLAMINAR_ROCM_NVNNI_CONCAT(PREFIX, LLAMINAR_ROCM_NATIVE_VNNI_SHARD)

extern "C" bool rocmGemv_native_vnni_query_serial_m1_config_with_policy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu);

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
    NVNNI_GEMM_Q8_0    = 19,  // 8-bit symmetric, direct INT8 payload (32 bytes)
    NVNNI_GEMM_EXPANDED_INT8_MIN =
        llaminar2::kNativeVnniExpandedInt8MinCodebook,
};

// ============================================================================
// Compile-time format traits
// ============================================================================

template <NativeVNNIGemmFormat FMT>
struct NVNNIGemmTraits {
    static constexpr bool is_asymmetric =
        FMT == NVNNI_GEMM_Q4_1 || FMT == NVNNI_GEMM_Q5_1 ||
        FMT == NVNNI_GEMM_IQ1_S || FMT == NVNNI_GEMM_EXPANDED_INT8_MIN;
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
    static constexpr bool is_dual_scale_asym = (FMT == NVNNI_GEMM_Q2_K);
    // All scales/mins are in global arrays (not embedded in payload).
    static constexpr bool has_embedded_scales = false;
    static constexpr int  embedded_scale_offset = -1;
    static constexpr int  embedded_min_offset = -1;
    // Q2_K needs a separate emin array (per-sub-block FP16 min pairs).
    // IQ1_M uses Pattern D plus the canonical signed one-eighth correction.
    static constexpr bool needs_emin_array = (FMT == NVNNI_GEMM_Q2_K);
    static constexpr bool is_5bit   = (FMT == NVNNI_GEMM_Q5_0 || FMT == NVNNI_GEMM_Q5_1);
    static constexpr bool is_6bit   = (FMT == NVNNI_GEMM_Q6_K);
    static constexpr bool is_3bit   = (FMT == NVNNI_GEMM_Q3_K);
    static constexpr bool is_2bit   = (FMT == NVNNI_GEMM_Q2_K);
    static constexpr bool is_8bit =
        FMT == NVNNI_GEMM_Q8_0 || FMT == NVNNI_GEMM_EXPANDED_INT8_MIN;
    static constexpr bool uses_lut  = (FMT == NVNNI_GEMM_IQ4_NL);
    static constexpr int  payload_bytes =
        is_8bit ? 32 :                              // Q8_0: 32 raw INT8 values
        is_iq1_grid ? 6 :                          // IQ1_S/IQ1_M: qs[4]+qh[2], native size
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 12 : is_2bit ? 8 : 16;
    static constexpr int center_4bit = (FMT == NVNNI_GEMM_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NVNNI_GEMM_Q5_0) ? 16 : 0;
    // Does the compute phase need an auxiliary scale/min?
    static constexpr bool needs_aux_compute = is_dual_scale || is_asymmetric;
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

template <int N_TILE, int BPT, int M_TILE, NativeVNNIGemmFormat FMT, int UNROLL_G = 4,
          int BLOCK_SIZE_ = 256, int M_PER_THREAD_ = 4>
__device__ __forceinline__ void accumulate_block_gemm(
    int bp,
    const int32_t* __restrict__ a_buf,
    const int32_t* __restrict__ b_buf,
    const uint16_t* __restrict__ s_buf,
    const uint16_t* __restrict__ aux_buf,
    const uint16_t* __restrict__ iq1_m_qh_buf,
    const uint16_t* __restrict__ emin_lo_buf,
    const uint16_t* __restrict__ emin_hi_buf,
    const int32_t* __restrict__ sum_a_buf,
    float (&acc)[M_PER_THREAD_][N_TILE / (BLOCK_SIZE_ / (M_TILE / M_PER_THREAD_))],
    int m_base, int t_n,
    const float (&sa_blk)[M_PER_THREAD_])
{
    using Traits = NVNNIGemmTraits<FMT>;

    constexpr int M_PER_THREAD = M_PER_THREAD_;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE_ / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;

    if constexpr (Traits::is_dual_scale_asym)
    {
        // Pattern DA: dual-scale + asymmetric (Q2_K only)
        //
        // Keep both exact INT32 half-dots live until the serial-order base
        // update, then commit the two precomputed half-sums as one correction.
        // Applying each half immediately changes the expression tree when the
        // kernel tile changes.
        int32_t low_dot[M_PER_THREAD][N_PER_THREAD];
        int32_t high_dot[M_PER_THREAD][N_PER_THREAD];
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                low_dot[mm][nn] = 0;
                high_dot[mm][nn] = 0;
            }

        #pragma clang loop unroll_count(2)
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
                    low_dot[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], low_dot[mm][nn], false);
        }

        #pragma clang loop unroll_count(2)
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
                    high_dot[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], high_dot[mm][nn], false);
        }

        int32_t low_sum[M_PER_THREAD];
        int32_t high_sum[M_PER_THREAD];
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            low_sum[mm] =
                sum_a_buf[(bp * 2) * M_TILE + m_base + mm];
            high_sum[mm] =
                sum_a_buf[(bp * 2 + 1) * M_TILE + m_base + mm];
        }

        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
            const float scale_lo = __half2float(
                *reinterpret_cast<const __half*>(&s_buf[sn]));
            const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&aux_buf[sn]));
            const float emin_lo  = __half2float(*reinterpret_cast<const __half*>(&emin_lo_buf[sn]));
            const float emin_hi  = __half2float(*reinterpret_cast<const __half*>(&emin_hi_buf[sn]));
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                acc[mm][nn] = rocm_native_vnni_commit_dual_scale_block(
                    acc[mm][nn],
                    low_dot[mm][nn],
                    high_dot[mm][nn],
                    scale_lo,
                    scale_hi,
                    sa_blk[mm]);
                acc[mm][nn] = rocm_native_vnni_commit_dual_scale_correction(
                    acc[mm][nn],
                    low_sum[mm],
                    high_sum[mm],
                    emin_lo,
                    emin_hi,
                    sa_blk[mm]);
            }
        }
    }
    else if constexpr (Traits::is_dual_scale)
    {
        // Pattern D: dual-scale symmetric (Q6_K, Q3_K, IQ2_S, IQ2_XS)
        //
        // The exact serial M=1 contract first creates one INT32 sum for each
        // 16-element half-block and only then applies both weight scales and
        // the activation scale.  Retain both half-sums until the shared commit
        // helper is called.  Scaling each dot4 independently is algebraically
        // valid but changes FP32 parenthesization as M crosses tile boundaries.
        int32_t low_dot[M_PER_THREAD][N_PER_THREAD];
        int32_t high_dot[M_PER_THREAD][N_PER_THREAD];

        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                low_dot[mm][nn] = 0;
                high_dot[mm][nn] = 0;
            }

        #pragma clang loop unroll_count(2)
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
                    low_dot[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], low_dot[mm][nn], false);
        }

        #pragma clang loop unroll_count(2)
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
                    high_dot[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], high_dot[mm][nn], false);
        }

        int32_t iq1_m_group_sum[M_PER_THREAD][4];
        if constexpr (FMT == NVNNI_GEMM_IQ1_M)
        {
            constexpr int32_t ones = 0x01010101;
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                #pragma unroll
                for (int group = 0; group < 4; ++group)
                {
                    const int kg0 = bp * 8 + group * 2;
                    const int kg1 = kg0 + 1;
                    iq1_m_group_sum[mm][group] = __builtin_amdgcn_sdot4(
                        a_buf[kg1 * M_TILE + m_base + mm],
                        ones,
                        __builtin_amdgcn_sdot4(
                            a_buf[kg0 * M_TILE + m_base + mm],
                            ones,
                            0,
                            false),
                        false);
                }
            }
        }

        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
            const float scale_lo = __half2float(
                *reinterpret_cast<const __half*>(&s_buf[sn]));
            const float scale_hi = __half2float(
                *reinterpret_cast<const __half*>(&aux_buf[sn]));
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                acc[mm][nn] = rocm_native_vnni_commit_dual_scale_block(
                    acc[mm][nn],
                    low_dot[mm][nn],
                    high_dot[mm][nn],
                    scale_lo,
                    scale_hi,
                    sa_blk[mm]);
                if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                {
                    constexpr float delta = 0.125f;
                    const uint16_t qh =
                        iq1_m_qh_buf[
                            bp * N_TILE +
                            t_n * N_PER_THREAD +
                            nn];
                    const uint8_t qh0 = static_cast<uint8_t>(qh);
                    const uint8_t qh1 = static_cast<uint8_t>(qh >> 8);
                    acc[mm][nn] =
                        rocm_native_vnni_commit_iq1_m_delta_correction(
                            acc[mm][nn],
                            iq1_m_group_sum[mm][0],
                            iq1_m_group_sum[mm][1],
                            iq1_m_group_sum[mm][2],
                            iq1_m_group_sum[mm][3],
                            (qh0 & 0x08) ? -delta : delta,
                            (qh0 & 0x80) ? -delta : delta,
                            (qh1 & 0x08) ? -delta : delta,
                            (qh1 & 0x80) ? -delta : delta,
                            scale_lo,
                            scale_hi,
                            sa_blk[mm]);
                }
            }
        }
    }
    else if constexpr (Traits::is_asymmetric)
    {
        // Pattern A: asymmetric single-scale (Q4_1, Q5_1, IQ1_S)
        //
        // sum_a is pre-computed cooperatively during A-load and stored in
        // sum_a_buf (same layout as Pattern DA: per-half sums at bhi*M_TILE).
        // This eliminates M_PER_THREAD × 8 sdot4/block from the inner loop
        // and removes 4 VGPRs (sum_a[] array no longer live across groups).
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
        // FMA: acc += block_acc * scale + sum_a * aux(min)
        // Read cooperative sum_a from LDS (lo + hi halves = full block sum)
        int32_t sum_a[M_PER_THREAD];
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            sum_a[mm] = sum_a_buf[(bp * 2) * M_TILE + m_base + mm]
                      + sum_a_buf[(bp * 2 + 1) * M_TILE + m_base + mm];
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const int sn = bp * N_TILE + t_n * N_PER_THREAD + nn;
            const float block_scale = __half2float(*reinterpret_cast<const __half*>(&s_buf[sn]));
            const float block_min   = __half2float(*reinterpret_cast<const __half*>(&aux_buf[sn]));
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                acc[mm][nn] = rocm_native_vnni_commit_single_scale_block(
                    acc[mm][nn],
                    block_acc[mm][nn],
                    block_scale,
                    sa_blk[mm]);
                acc[mm][nn] = rocm_native_vnni_commit_single_scale_correction(
                    acc[mm][nn],
                    sum_a[mm],
                    block_min,
                    sa_blk[mm]);
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

        // UNROLL_G controls the inner group loop unroll for tuning.
        // Default (4) matches original behavior: #pragma clang loop unroll_count(4).
#define NVNNI_PATTERN_S_G_BODY \
        { \
            const int kk = bp * 8 + g; \
            int32_t a_reg[M_PER_THREAD]; \
            _Pragma("unroll") \
            for (int mm = 0; mm < M_PER_THREAD; ++mm) \
                a_reg[mm] = a_buf[kk * M_TILE + m_base + mm]; \
            int32_t b_reg[N_PER_THREAD]; \
            _Pragma("unroll") \
            for (int nn = 0; nn < N_PER_THREAD; ++nn) \
                b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn]; \
            _Pragma("unroll") \
            for (int mm = 0; mm < M_PER_THREAD; ++mm) \
                _Pragma("unroll") \
                for (int nn = 0; nn < N_PER_THREAD; ++nn) \
                    block_acc[mm][nn] = __builtin_amdgcn_sdot4( \
                        a_reg[mm], b_reg[nn], block_acc[mm][nn], false); \
        }
        if constexpr (UNROLL_G == 0) {
            for (int g = 0; g < 8; ++g) NVNNI_PATTERN_S_G_BODY
        } else if constexpr (UNROLL_G == 2) {
            #pragma clang loop unroll_count(2)
            for (int g = 0; g < 8; ++g) NVNNI_PATTERN_S_G_BODY
        } else if constexpr (UNROLL_G == 1) {
            #pragma unroll
            for (int g = 0; g < 8; ++g) NVNNI_PATTERN_S_G_BODY
        } else { // UNROLL_G == 4 (default)
            #pragma clang loop unroll_count(4)
            for (int g = 0; g < 8; ++g) NVNNI_PATTERN_S_G_BODY
        }
#undef NVNNI_PATTERN_S_G_BODY
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
                acc[mm][nn] = rocm_native_vnni_commit_single_scale_block(
                    acc[mm][nn],
                    block_acc[mm][nn],
                    block_scale[nn],
                    sa_blk[mm]);
    }
}

/**
 * @brief Commit one completed serial-M1 K partition into workgroup-local state.
 *
 * Serial decode gives each split-K worker a zero-initialized accumulator, then
 * the reducer adds those partials in ascending partition order. A prefill tile
 * already has enough M-by-N parallelism and does not benefit from launching
 * separate K workers, but it must retain the same FP32 expression tree. Each
 * output owner therefore stores its running partition in `acc`, commits it to
 * a private register accumulator at the exact serial boundary, and starts the
 * next partition from positive zero. Keeping both values in registers avoids
 * adding global traffic or repeated LDS round trips to the K loop.
 *
 * @tparam N_TILE Number of output columns owned by the workgroup.
 * @tparam M_TILE Number of output rows owned by the workgroup.
 * @tparam M_PER_THREAD Number of output rows owned by one thread.
 * @tparam N_PER_THREAD Number of output columns owned by one thread.
 * @param acc Current partition accumulators.
 * @param ordered_sum Per-thread ordered reduction accumulators.
 * @param global_block Zero-based 32-value weight block just accumulated.
 * @param blocks_per_row Total number of real weight blocks.
 * @param blocks_per_partition Serial-M1 partition span.
 * @param m_base Thread-local row origin within the workgroup tile.
 * @param t_n Thread column-group index.
 */
template <int N_TILE, int M_TILE, int M_PER_THREAD, int N_PER_THREAD>
__device__ __forceinline__ void commit_serial_m1_partition_if_complete(
    float (&acc)[M_PER_THREAD][N_PER_THREAD],
    float (&ordered_sum)[M_PER_THREAD][N_PER_THREAD],
    int global_block,
    int blocks_per_row,
    int blocks_per_partition,
    int,
    int)
{
    if (global_block >= blocks_per_row)
        return;

    const int completed_blocks = global_block + 1;
    const bool partition_complete =
        completed_blocks == blocks_per_row ||
        (completed_blocks % blocks_per_partition) == 0;
    if (!partition_complete)
        return;

    const bool first_partition =
        completed_blocks <= blocks_per_partition;
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            ordered_sum[mm][nn] =
                first_partition
                    ? acc[mm][nn]
                    : __fadd_rn(
                          ordered_sum[mm][nn],
                          acc[mm][nn]);
            acc[mm][nn] = 0.0f;
        }
    }
}

/**
 * @brief Reload the completed serial-M1 reduction tree for final publication.
 *
 * @tparam N_TILE Number of output columns owned by the workgroup.
 * @tparam M_PER_THREAD Number of output rows owned by one thread.
 * @tparam N_PER_THREAD Number of output columns owned by one thread.
 * @param acc Destination register tile used by the existing vectorized store.
 * @param ordered_sum Per-thread ordered reduction accumulators.
 * @param m_base Thread-local row origin within the workgroup tile.
 * @param t_n Thread column-group index.
 */
template <int N_TILE, int M_PER_THREAD, int N_PER_THREAD>
__device__ __forceinline__ void load_serial_m1_partition_sum(
    float (&acc)[M_PER_THREAD][N_PER_THREAD],
    const float (&ordered_sum)[M_PER_THREAD][N_PER_THREAD],
    int,
    int)
{
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            acc[mm][nn] = ordered_sum[mm][nn];
        }
    }
}

template <int N_TILE,
          int BPT,
          int M_TILE,
          NativeVNNIGemmFormat FMT,
          int MIN_BLOCKS = 2,
          int UNROLL_G = 4,
          bool FULL_TILES = false>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void native_vnni_gemm_kernel(
    const int8_t*    __restrict__ d_A_int8,       // [M × K] INT8 activations, row-major
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × PB] native-VNNI payload
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N] FP16 per-block scales
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N] FP16 per-block min/aux
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row × N] packed FP16 emin pair (Q2_K only, else NULL)
    float*           __restrict__ d_output,       // [M × N] FP32 output, row-major
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    int M, int N, int K,
    const float*     __restrict__ d_scales_A_blockwise, // [M × blocks_per_row] per-block activation scales (nullable)
    int serial_m1_kblocks)
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
    __shared__ uint16_t iq1_m_qh_lds[
        2][FMT == NVNNI_GEMM_IQ1_M ? BPT * N_TILE : 1];
    __shared__ uint16_t emin_lo_lds[2][Traits::is_dual_scale_asym ? BPT * N_TILE : 1]; // Embedded min lo
    __shared__ uint16_t emin_hi_lds[2][Traits::is_dual_scale_asym ? BPT * N_TILE : 1]; // Embedded min hi
    // Precomputed activation sums for cooperative sum_a (Pattern DA + Pattern A).
    // Computed cooperatively during A-load: each A-load thread covers exactly
    // one block-half for one M-row (4 sdot4 per thread vs 32 sdot4/block in
    // the old per-thread approach → 64× fewer sdot4 total + 4 fewer VGPRs).
    // Layout: [block_half_idx * M_TILE + mi], block_half_idx = bp*2 + {0=lo,1=hi}
    // Pattern DA reads halves separately (lo/hi emin correction).
    // Pattern A reads and sums both halves (full-block min correction).
    __shared__ int32_t sum_a_lds[2][Traits::needs_cooperative_sum_a ? BPT * 2 * M_TILE : 1];
    // FP32 accumulators — accumulate across all K-blocks
    float acc[M_PER_THREAD][N_PER_THREAD];
    float ordered_partition_sum[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            acc[mm][nn] = 0.0f;
            ordered_partition_sum[mm][nn] = 0.0f;
        }

    // K dimensions
    const int blocks_per_row = K / 32;
    const int total_k_groups = K / 4;
    const int num_block_tiles = (blocks_per_row + BPT - 1) / BPT;
    const int blocks_per_partition =
        (blocks_per_row + serial_m1_kblocks - 1) / serial_m1_kblocks;

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
    // FULL_TILES is selected only by a launch-site divisibility proof. Keeping
    // that fact in the kernel type lets LLVM erase the checked-load loop and
    // scalar output tail completely. Those paths otherwise retain thread and
    // pointer geometry across the whole K loop, which creates scratch traffic
    // even though production transformer dimensions never take either edge.
    const int num_safe = FULL_TILES
        ? num_block_tiles
        : ((a_m_safe && b_n_safe)
               ? (safe_k < num_block_tiles ? safe_k : num_block_tiles)
               : 0);

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
    uint16_t b_iq1_m_qh_staging; // IQ1_M high-index and delta-sign bytes
    uint16_t b_emin_lo_staging; // embedded min lo (for dual_scale_asym)
    uint16_t b_emin_hi_staging; // embedded min hi (for dual_scale_asym)

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    {
        const bool tile0_safe = FULL_TILES || (num_safe > 0);

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
        if constexpr (FMT == NVNNI_GEMM_IQ1_M) b_iq1_m_qh_staging = 0;
        if constexpr (Traits::is_dual_scale_asym) { b_emin_lo_staging = 0; b_emin_hi_staging = 0; }
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = b_load_block;
            if (tile0_safe || __builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                load_payload<PB>(d_payload + linear * PB, b_payload_dw);
                b_scale_staging = d_block_scales[linear];
                if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                    b_iq1_m_qh_staging =
                        static_cast<uint16_t>(b_payload_dw[1]);
                if constexpr (Traits::needs_aux_compute)
                    b_aux_staging = d_block_mins[linear];
                if constexpr (Traits::is_dual_scale_asym) {
                    if constexpr (Traits::needs_emin_array) {
                        const uint32_t emin_packed = d_block_emins[linear];
                        b_emin_lo_staging = static_cast<uint16_t>(emin_packed);
                        b_emin_hi_staging = static_cast<uint16_t>(emin_packed >> 16);
                    } else {
                        b_emin_lo_staging = 0;
                        b_emin_hi_staging = 0;
                    }
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
        if constexpr (Traits::needs_cooperative_sum_a) {
            // Cooperative sum_a: this thread's 4 k-groups = 1 block-half
            const int bhi = a_kg / 4;
            int32_t sa = __builtin_amdgcn_sdot4(
                static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0, false);
            sa = __builtin_amdgcn_sdot4(
                static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa, false);
            sa = __builtin_amdgcn_sdot4(
                static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa, false);
            sa = __builtin_amdgcn_sdot4(
                static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa, false);
            sum_a_lds[0][bhi * M_TILE + mi] = sa;
        }
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
        if constexpr (FMT == NVNNI_GEMM_IQ1_M)
            iq1_m_qh_lds[0][b_load_block * N_TILE + b_load_n] =
                b_iq1_m_qh_staging;
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
            b_scale_staging = d_block_scales[linear];
            if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                b_iq1_m_qh_staging =
                    static_cast<uint16_t>(b_payload_dw[1]);
            if constexpr (Traits::needs_aux_compute)
                b_aux_staging = d_block_mins[linear];
            if constexpr (Traits::is_dual_scale_asym) {
                if constexpr (Traits::needs_emin_array) {
                    const uint32_t emin_packed = d_block_emins[linear];
                    b_emin_lo_staging = static_cast<uint16_t>(emin_packed);
                    b_emin_hi_staging = static_cast<uint16_t>(emin_packed >> 16);
                } else {
                    b_emin_lo_staging = 0;
                    b_emin_hi_staging = 0;
                }
            }
        }

        // --- 2. Compute on CURRENT buffer (contains tile-1's data) ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            float sa_blk[M_PER_THREAD];
            if (d_scales_A_blockwise) {
                const int gblk = (tile - 1) * BPT + bp;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm) {
                    const int gm = m_block + m_base + mm;
                    sa_blk[mm] = (FULL_TILES || gm < M)
                        ? d_scales_A_blockwise[
                              static_cast<int64_t>(gm) * blocks_per_row + gblk]
                        : 0.0f;
                }
            } else {
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    sa_blk[mm] = 1.0f;
            }
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT, UNROLL_G>(
                bp, a_lds[cur_buf], b_lds[cur_buf],
                scale_lds[cur_buf], aux_lds[cur_buf],
                iq1_m_qh_lds[cur_buf],
                emin_lo_lds[cur_buf], emin_hi_lds[cur_buf],
                sum_a_lds[cur_buf],
                acc, m_base, t_n, sa_blk);
            if (serial_m1_kblocks > 1)
            {
                const int global_block = (tile - 1) * BPT + bp;
                commit_serial_m1_partition_if_complete<
                    N_TILE, M_TILE, M_PER_THREAD, N_PER_THREAD>(
                        acc,
                        ordered_partition_sum,
                        global_block,
                        blocks_per_row,
                        blocks_per_partition,
                        m_base,
                        t_n);
            }
        }

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
            if constexpr (Traits::needs_cooperative_sum_a) {
                const int bhi = a_kg / 4;
                int32_t sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0, false);
                sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa, false);
                sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa, false);
                sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa, false);
                sum_a_lds[nxt_buf][bhi * M_TILE + mi] = sa;
            }
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
            if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                iq1_m_qh_lds[nxt_buf][b_load_block * N_TILE + b_load_n] =
                    b_iq1_m_qh_staging;
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
        if constexpr (FMT == NVNNI_GEMM_IQ1_M) b_iq1_m_qh_staging = 0;
        if constexpr (Traits::is_dual_scale_asym) { b_emin_lo_staging = 0; b_emin_hi_staging = 0; }
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = tile_block_base + b_load_block;
            if (__builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                load_payload<PB>(d_payload + linear * PB, b_payload_dw);
                b_scale_staging = d_block_scales[linear];
                if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                    b_iq1_m_qh_staging =
                        static_cast<uint16_t>(b_payload_dw[1]);
                if constexpr (Traits::needs_aux_compute)
                    b_aux_staging = d_block_mins[linear];
                if constexpr (Traits::is_dual_scale_asym) {
                    if constexpr (Traits::needs_emin_array) {
                        const uint32_t emin_packed = d_block_emins[linear];
                        b_emin_lo_staging = static_cast<uint16_t>(emin_packed);
                        b_emin_hi_staging = static_cast<uint16_t>(emin_packed >> 16);
                    } else {
                        b_emin_lo_staging = 0;
                        b_emin_hi_staging = 0;
                    }
                }
            }
        }

        // --- 2. Compute on CURRENT buffer (contains tile-1's data) ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            float sa_blk[M_PER_THREAD];
            if (d_scales_A_blockwise) {
                const int gblk = (tile - 1) * BPT + bp;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm) {
                    const int gm = m_block + m_base + mm;
                    sa_blk[mm] = (FULL_TILES || gm < M)
                        ? d_scales_A_blockwise[
                              static_cast<int64_t>(gm) * blocks_per_row + gblk]
                        : 0.0f;
                }
            } else {
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    sa_blk[mm] = 1.0f;
            }
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT, UNROLL_G>(
                bp, a_lds[cur_buf], b_lds[cur_buf],
                scale_lds[cur_buf], aux_lds[cur_buf],
                iq1_m_qh_lds[cur_buf],
                emin_lo_lds[cur_buf], emin_hi_lds[cur_buf],
                sum_a_lds[cur_buf],
                acc, m_base, t_n, sa_blk);
            if (serial_m1_kblocks > 1)
            {
                const int global_block = (tile - 1) * BPT + bp;
                commit_serial_m1_partition_if_complete<
                    N_TILE, M_TILE, M_PER_THREAD, N_PER_THREAD>(
                        acc,
                        ordered_partition_sum,
                        global_block,
                        blocks_per_row,
                        blocks_per_partition,
                        m_base,
                        t_n);
            }
        }

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
            if constexpr (Traits::needs_cooperative_sum_a) {
                const int bhi = a_kg / 4;
                int32_t sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0, false);
                sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa, false);
                sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa, false);
                sa = __builtin_amdgcn_sdot4(
                    static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa, false);
                sum_a_lds[nxt_buf][bhi * M_TILE + mi] = sa;
            }
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
            if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                iq1_m_qh_lds[nxt_buf][b_load_block * N_TILE + b_load_n] =
                    b_iq1_m_qh_staging;
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
        {
            float sa_blk[M_PER_THREAD];
            if (d_scales_A_blockwise) {
                const int gblk = (num_block_tiles - 1) * BPT + bp;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm) {
                    const int gm = m_block + m_base + mm;
                    sa_blk[mm] = (FULL_TILES || gm < M)
                        ? d_scales_A_blockwise[
                              static_cast<int64_t>(gm) * blocks_per_row + gblk]
                        : 0.0f;
                }
            } else {
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    sa_blk[mm] = 1.0f;
            }
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT, UNROLL_G>(
                bp, a_lds[last_buf], b_lds[last_buf],
                scale_lds[last_buf], aux_lds[last_buf],
                iq1_m_qh_lds[last_buf],
                emin_lo_lds[last_buf], emin_hi_lds[last_buf],
                sum_a_lds[last_buf],
                acc, m_base, t_n, sa_blk);
            if (serial_m1_kblocks > 1)
            {
                const int global_block =
                    (num_block_tiles - 1) * BPT + bp;
                commit_serial_m1_partition_if_complete<
                    N_TILE, M_TILE, M_PER_THREAD, N_PER_THREAD>(
                        acc,
                        ordered_partition_sum,
                        global_block,
                        blocks_per_row,
                        blocks_per_partition,
                        m_base,
                        t_n);
            }
        }
    }

    if (serial_m1_kblocks > 1)
    {
        load_serial_m1_partition_sum<
            N_TILE, M_PER_THREAD, N_PER_THREAD>(
                acc,
                ordered_partition_sum,
                m_base,
                t_n);
    }

    // =========================================================================
    // STORE RESULTS: Apply scale_A and write FP32 to global memory
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;
        if constexpr (!FULL_TILES)
        {
            if (m_idx >= M)
                continue;
        }

        // Keep row scales out of the K-loop live range. The former prologue
        // array survived until this epilogue and forced three values into
        // scratch in the production Q6_K N128/M32 binary. Blockwise-scaled
        // rows already consumed their scale inside each ordered K block.
        const float sa = d_scales_A_blockwise ? 1.0f : d_scales_A[m_idx];

        if constexpr (N_PER_THREAD >= 4)
        {
            // Vectorized store path
            if constexpr (FULL_TILES)
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
                if (n_base + N_PER_THREAD - 1 < N)
                {
                    const int64_t row_off =
                        static_cast<int64_t>(m_idx) * N + n_base;
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
                    // Scalar publication is reserved for a genuine N edge.
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    {
                        const int n_idx = n_base + nn;
                        if (n_idx < N)
                            d_output[
                                static_cast<int64_t>(m_idx) * N + n_idx] =
                                acc[mm][nn] * sa;
                    }
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
// Streaming LM_Head kernel: register streaming with optional setup-only LDS
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
// The symmetric fast path eliminates all barriers and LDS usage:
//   - B: loaded per-thread from global memory → decode in registers
//   - A: loaded per-thread, all 256 threads read same addresses → L2 broadcast
//   - Zero __syncthreads → VALU pipeline runs uninterrupted
//   - ~50 VGPRs → 3-wave occupancy (vs 2-wave for LDS kernel)
//
// Formats with row-only correction terms use one cooperative setup phase and
// one barrier before the K loop. The K loop itself remains synchronization
// free, and no correction sum is redundantly recomputed per N column.
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
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, else nullptr)
    float*           __restrict__ d_output,       // [M × N] FP32 output
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    const float*     __restrict__ d_scales_A_blockwise, // [M × blocks_per_row] per-block activation scales
    int M, int N, int K,
    int serial_m1_kblocks)
{
    static_assert(M_TILE == 8 || M_TILE == 16 || M_TILE == 32,
                  "Streaming LM_Head kernel supports M_TILE 8, 16, or 32");

    using Traits = NVNNIGemmTraits<FMT>;
    constexpr int PB = Traits::payload_bytes;

    // Thread position: 1 thread = M_TILE M-rows × 1 N-column
    const int n_col   = blockIdx.x * 256 + threadIdx.x;
    const int m_block = blockIdx.y * M_TILE;

    // Reinterpret A as int32 (4 packed INT8 per int32 k-group)
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int a_row_stride  = K / 4;   // k-groups per row
    const int blocks_per_row = K / 32;  // quantized blocks per row

    // Bounds: if entire M_TILE is within [0, M), skip per-element checks
    const bool m_safe = (m_block + M_TILE <= M);

    // =========================================================================
    // Cooperative activation-correction sums pre-computed into LDS
    //
    // Instead of each thread computing sum_a independently (128 sdot4 per block
    // per thread — 99.6% redundant across the 256 threads in a WG), pre-compute
    // sum_a cooperatively: 256 threads share M_TILE × blocks_per_row work items.
    // This eliminates sum_a from the hot inner loop, reducing per-block VALU by
    // 45% and dropping VGPRs from ~70 (3-wave) to ~54 (4-wave).
    //
    // Pattern A LDS: M_TILE × blocks_per_row × 4 bytes.
    // IQ1_M LDS:    M_TILE × blocks_per_row × 4 groups × 4 bytes.
    // =========================================================================
    extern __shared__ int32_t sum_a_shared[];

    if constexpr (Traits::is_asymmetric)
    {
        const int total_items = M_TILE * blocks_per_row;
        for (int item = threadIdx.x; item < total_items; item += 256)
        {
            const int mm    = item / blocks_per_row;
            const int block = item % blocks_per_row;
            const int gm    = m_block + mm;
            int32_t s = 0;
            if (m_safe || gm < M) {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                    s = __builtin_amdgcn_sdot4(
                        a_global[static_cast<int64_t>(gm) * a_row_stride + block * 8 + kg],
                        static_cast<int32_t>(0x01010101), s, false);
            }
            sum_a_shared[item] = s;
        }
        __syncthreads();
    }
    else if constexpr (FMT == NVNNI_GEMM_IQ1_M)
    {
        /*
         * IQ1_M's signed one-eighth correction depends on four sums of eight
         * activation values for every row/block. Those sums are independent
         * of N, so computing them in every output-column thread would repeat
         * identical work 256 times. The workgroup computes each sum once and
         * publishes it through LDS before any thread enters the K loop.
         *
         * Layout: [row][block][group], where group selects one of the four
         * consecutive eight-value regions in the 32-value quantization block.
         */
        constexpr int kGroupsPerBlock = 4;
        constexpr int32_t kPackedOnes = 0x01010101;
        const int items_per_row = blocks_per_row * kGroupsPerBlock;
        const int total_items = M_TILE * items_per_row;
        for (int item = threadIdx.x; item < total_items; item += 256)
        {
            const int mm = item / items_per_row;
            const int row_item = item - mm * items_per_row;
            const int block = row_item / kGroupsPerBlock;
            const int group = row_item - block * kGroupsPerBlock;
            const int gm = m_block + mm;
            int32_t sum = 0;
            if (m_safe || gm < M)
            {
                const int kg0 = block * 8 + group * 2;
                const int kg1 = kg0 + 1;
                const int64_t row_base =
                    static_cast<int64_t>(gm) * a_row_stride;
                sum = __builtin_amdgcn_sdot4(
                    a_global[row_base + kg1],
                    kPackedOnes,
                    __builtin_amdgcn_sdot4(
                        a_global[row_base + kg0],
                        kPackedOnes,
                        0,
                        false),
                    false);
            }
            sum_a_shared[item] = sum;
        }
        __syncthreads();
    }

    // Now safe to exit early for out-of-bounds N-columns
    // (correction pre-compute must complete for ALL threads before barrier)
    if (n_col >= N) return;

    // FP32 accumulators — accumulated across all K-blocks
    float acc[M_TILE];
    float ordered_partition_sum[M_TILE];
    #pragma unroll
    for (int mm = 0; mm < M_TILE; ++mm)
    {
        acc[mm] = 0.0f;
        ordered_partition_sum[mm] = 0.0f;
    }

    // =========================================================================
    // Main loop: stream through K-blocks with zero barriers
    // =========================================================================
    for (int block = 0; block < blocks_per_row; ++block)
    {
        // --- 0. Compute per-block activation scales ---
        float sa_blk[M_TILE];
        if (d_scales_A_blockwise) {
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                const int gm = m_block + mm;
                sa_blk[mm] = (m_safe || gm < M) ? d_scales_A_blockwise[static_cast<int64_t>(gm) * blocks_per_row + block] : 0.0f;
            }
        } else {
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                sa_blk[mm] = 1.0f;
        }

        // --- 1. Load B payload + scale/aux for this thread's N-column ---
        const size_t linear = static_cast<size_t>(block) * N + n_col;
        uint32_t b_payload_dw[Traits::payload_dwords];
        load_payload<PB>(d_payload + linear * PB, b_payload_dw);

        uint16_t scale_bits = d_block_scales[linear];

        uint16_t aux_bits = 0;
        if constexpr (Traits::needs_aux_compute)
            aux_bits = d_block_mins[linear];

        uint16_t emin_lo_bits = 0, emin_hi_bits = 0;
        if constexpr (Traits::is_dual_scale_asym) {
            if constexpr (Traits::needs_emin_array) {
                const uint32_t emin_packed = d_block_emins[linear];
                emin_lo_bits = static_cast<uint16_t>(emin_packed);
                emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
            }
        }

        // --- 2. Decode B → 8 INT32 packed groups ---
        int32_t packed[8];
        decode_block<FMT>(b_payload_dw, packed);

        // --- 3. Accumulate based on format pattern ---
        if constexpr (Traits::is_dual_scale_asym)
        {
            // Pattern DA: two-phase accumulation (Q2_K only)
            // Single-path with ternary bounds check — avoids m_safe code
            // duplication that would push VGPR past 64 (losing 4-wave occupancy).
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
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]) * sa_blk[mm], slo, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(temp_sum[mm]) * sa_blk[mm], elo, acc[mm]);
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
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]) * sa_blk[mm], shi, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(temp_sum[mm]) * sa_blk[mm], ehi, acc[mm]);
            }
        }
        else if constexpr (Traits::is_dual_scale)
        {
            // Pattern D: retain both half-dots until one serial-order update.
            const int kg_base = block * 8;

            int32_t low_dot[M_TILE];
            int32_t high_dot[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                low_dot[mm] = 0;
                high_dot[mm] = 0;
            }
            if (m_safe)
            {
                #pragma unroll
                for (int kg = 0; kg < 4; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int32_t a_val = a_global[
                            static_cast<int64_t>(m_block + mm) * a_row_stride + a_kg_idx];
                        low_dot[mm] = __builtin_amdgcn_sdot4(
                            a_val, b_val, low_dot[mm], false);
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int kg = 0; kg < 4; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int gm = m_block + mm;
                        const int32_t a_val = (gm < M)
                            ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                        low_dot[mm] = __builtin_amdgcn_sdot4(
                            a_val, b_val, low_dot[mm], false);
                    }
                }
            }

            if (m_safe)
            {
                #pragma unroll
                for (int kg = 4; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int32_t a_val = a_global[
                            static_cast<int64_t>(m_block + mm) * a_row_stride + a_kg_idx];
                        high_dot[mm] = __builtin_amdgcn_sdot4(
                            a_val, b_val, high_dot[mm], false);
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int kg = 4; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int gm = m_block + mm;
                        const int32_t a_val = (gm < M)
                            ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                        high_dot[mm] = __builtin_amdgcn_sdot4(
                            a_val, b_val, high_dot[mm], false);
                    }
                }
            }

            const float slo =
                __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float shi = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            uint16_t iq1_m_qh = 0;
            if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                iq1_m_qh = static_cast<uint16_t>(b_payload_dw[1]);
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = rocm_native_vnni_commit_dual_scale_block(
                    acc[mm],
                    low_dot[mm],
                    high_dot[mm],
                    slo,
                    shi,
                    sa_blk[mm]);
                if constexpr (FMT == NVNNI_GEMM_IQ1_M)
                {
                    constexpr int kGroupsPerBlock = 4;
                    constexpr float kDelta = 0.125f;
                    const int sum_base =
                        (mm * blocks_per_row + block) * kGroupsPerBlock;
                    const uint8_t qh0 = static_cast<uint8_t>(iq1_m_qh);
                    const uint8_t qh1 =
                        static_cast<uint8_t>(iq1_m_qh >> 8);
                    acc[mm] =
                        rocm_native_vnni_commit_iq1_m_delta_correction(
                            acc[mm],
                            sum_a_shared[sum_base],
                            sum_a_shared[sum_base + 1],
                            sum_a_shared[sum_base + 2],
                            sum_a_shared[sum_base + 3],
                            (qh0 & 0x08) ? -kDelta : kDelta,
                            (qh0 & 0x80) ? -kDelta : kDelta,
                            (qh1 & 0x08) ? -kDelta : kDelta,
                            (qh1 & 0x80) ? -kDelta : kDelta,
                            slo,
                            shi,
                            sa_blk[mm]);
                }
            }
        }
        else if constexpr (Traits::is_asymmetric)
        {
            // Pattern A: asymmetric single-scale (Q4_1, Q5_1, IQ1_S)
            //
            // sum_a is pre-computed in LDS (cooperative phase above).
            // This eliminates M_TILE × 8 redundant sdot4 per block per thread
            // and drops the VGPR count from ~70 (3-wave) to ~54 (4-wave).
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
                        block_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, block_acc[mm], false);
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
                        block_acc[mm] = __builtin_amdgcn_sdot4(a_val, b_val, block_acc[mm], false);
                    }
                }
            }
            const float bs = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float bm = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = rocm_native_vnni_commit_single_scale_block(
                    acc[mm],
                    block_acc[mm],
                    bs,
                    sa_blk[mm]);
                acc[mm] = rocm_native_vnni_commit_single_scale_correction(
                    acc[mm],
                    sum_a_shared[mm * blocks_per_row + block],
                    bm,
                    sa_blk[mm]);
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
                acc[mm] = rocm_native_vnni_commit_single_scale_block(
                    acc[mm],
                    block_acc[mm],
                    block_scale,
                    sa_blk[mm]);
        }

        /*
         * Reproduce serial M=1's split-K reducer inside each output owner.
         * A streaming thread owns one N column and all M_TILE rows, so both
         * the current partition and ordered sum remain private registers.
         */
        if (serial_m1_kblocks > 1)
        {
            const int blocks_per_partition =
                (blocks_per_row + serial_m1_kblocks - 1) /
                serial_m1_kblocks;
            const int completed_blocks = block + 1;
            const bool partition_complete =
                completed_blocks == blocks_per_row ||
                (completed_blocks % blocks_per_partition) == 0;
            if (partition_complete)
            {
                const bool first_partition =
                    completed_blocks <= blocks_per_partition;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    ordered_partition_sum[mm] =
                        first_partition
                            ? acc[mm]
                            : __fadd_rn(
                                  ordered_partition_sum[mm],
                                  acc[mm]);
                    acc[mm] = 0.0f;
                }
            }
        }
    }

    if (serial_m1_kblocks > 1)
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
            acc[mm] = ordered_partition_sum[mm];
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
            const float sa = d_scales_A_blockwise ? 1.0f : d_scales_A[gm];
            d_output[static_cast<int64_t>(gm) * N + n_col] =
                acc[mm] * sa;
        }
    }
    else
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
        {
            const int gm = m_block + mm;
            if (gm < M) {
                const float sa = d_scales_A_blockwise ? 1.0f : d_scales_A[gm];
                d_output[static_cast<int64_t>(gm) * N + n_col] =
                    acc[mm] * sa;
            }
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
    const uint32_t* d_block_emins,
    float* d_output,
    const float* d_scales_A,
    const float* d_scales_A_blockwise,
    int M, int N, int K,
    int device_id, void* stream,
    uint8_t arithmetic_policy_codebook_id)
{
    constexpr int BPT = 2;
    constexpr int BLOCK_SIZE = 256;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const bool stream_capture_active = [&]() {
        if (llaminar2::isGraphCaptureActive())
            return true;
        if (!hip_stream)
            return false;
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        const hipError_t err = hipStreamIsCapturing(hip_stream, &status);
        return err == hipSuccess && status == hipStreamCaptureStatusActive;
    }();

    /*
     * Batch invariance is an arithmetic launch property, not a debug mode.
     * Resolve the exact split count used by serial M=1 before choosing a tile
     * family. Every cooperative family below replays those K boundaries
     * locally, so changing M or N_TILE cannot change a published row.
     */
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    int serial_m1_kblocks = rocm_env.nvnni_gemv_kb;
    if (serial_m1_kblocks <= 0)
    {
        int ignored_target_waves = 0;
        if (!rocmGemv_native_vnni_query_serial_m1_config_with_policy(
                arithmetic_policy_codebook_id,
                N,
                K,
                &serial_m1_kblocks,
                &ignored_target_waves))
        {
            fprintf(
                stderr,
                "[rocmGemm_native_vnni_fp32] no serial M=1 split-K policy "
                "for batch-invariant prefill (M=%d N=%d K=%d "
                "codebook=%u arithmetic_policy_codebook=%u)\n",
                M,
                N,
                K,
                static_cast<unsigned>(FMT),
                static_cast<unsigned>(arithmetic_policy_codebook_id));
            return false;
        }
    }
    serial_m1_kblocks =
        min(max(serial_m1_kblocks, 1), K / 32);

    auto report_launch_status = [&](const char* route,
                                    int n_tile,
                                    int m_tile,
                                    int min_blocks,
                                    int unroll,
                                    hipError_t err) -> bool
    {
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemm_native_vnni_fp32] launch failed: %s "
                    "route=%s M=%d N=%d K=%d codebook=%u N_TILE=%d M_TILE=%d "
                    "min_blocks=%d unroll=%d device=%d stream=%p\n",
                    hipGetErrorString(err),
                    route ? route : "unknown",
                    M, N, K,
                    static_cast<unsigned>(FMT),
                    n_tile,
                    m_tile,
                    min_blocks,
                    unroll,
                    device_id,
                    stream);
        }
        return err == hipSuccess;
    };

    auto clear_launch_error = [&]() {
        if (!stream_capture_active)
            (void)hipGetLastError();
    };

    auto complete_launch = [&](const char* route,
                               int n_tile,
                               int m_tile,
                               int min_blocks,
                               int unroll,
                               bool full_tiles,
                               const void* function) -> bool
    {
        rocmNativeVNNIPrefill_recordLastLaunchSelection(
            static_cast<uint8_t>(FMT),
            n_tile,
            m_tile,
            min_blocks,
            unroll,
            full_tiles,
            function,
            BLOCK_SIZE);
        if (stream_capture_active)
            return true;
        return report_launch_status(
            route,
            n_tile,
            m_tile,
            min_blocks,
            unroll,
            hipGetLastError());
    };

    // Templated launch helper (default 2-wave)
    auto launch = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr bool Q6_CHECKED_MB2_SPILLS =
            FMT == NVNNI_GEMM_Q6_K &&
            ((NT == 64 && MT == 64) || (NT == 128 && MT == 32));
        constexpr int MB = Q6_CHECKED_MB2_SPILLS ? 1 : 2;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
            native_vnni_gemm_kernel<NT, BPT, MT, FMT, MB, 4, false>));

        clear_launch_error();
        native_vnni_gemm_kernel<NT, BPT, MT, FMT, MB, 4, false>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                d_output, d_scales_A, M, N, K, d_scales_A_blockwise,
                serial_m1_kblocks);

        return complete_launch(
            Q6_CHECKED_MB2_SPILLS ? "q6_checked_mb1" : "default",
            NT, MT, MB, 4, /*full_tiles=*/false, function);
    };

    // 3-wave variant: MIN_BLOCKS=3 tells compiler to target ≤84 VGPRs
    auto launch3 = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
            native_vnni_gemm_kernel<NT, BPT, MT, FMT, 3>));

        clear_launch_error();
        native_vnni_gemm_kernel<NT, BPT, MT, FMT, 3>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                d_output, d_scales_A, M, N, K, d_scales_A_blockwise,
                serial_m1_kblocks);

        return complete_launch(
            "three_wave", NT, MT, 3, 4, /*full_tiles=*/false, function);
    };

    // Streaming LM_Head variant: zero barriers, zero LDS, 3-wave occupancy
    // (Pattern A uses small LDS for sum_a precompute → 4-wave achievable)
    auto launch_streaming = [&](auto m_tile_tag) -> bool
    {
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr int N_PER_WG = 256;  // each thread handles 1 N-column
        dim3 grid((N + N_PER_WG - 1) / N_PER_WG, (M + MT - 1) / MT, 1);
        dim3 block(256, 1, 1);

        // Row-only correction sums are computed once per workgroup. Pattern A
        // stores one full-block sum; IQ1_M stores four eight-value group sums.
        const int bpr = K / 32;
        const size_t sum_a_bytes =
            NVNNIGemmTraits<FMT>::is_asymmetric
                ? (MT * bpr * sizeof(int32_t))
                : FMT == NVNNI_GEMM_IQ1_M
                    ? (MT * bpr * 4 * sizeof(int32_t))
                : 0;
        const size_t shared_bytes = sum_a_bytes;
        const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
            streaming_lm_head_kernel<MT, FMT>));

        clear_launch_error();
        streaming_lm_head_kernel<MT, FMT>
            <<<grid, block, shared_bytes, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                d_output, d_scales_A, d_scales_A_blockwise, M, N, K,
                serial_m1_kblocks);

        return complete_launch(
            "streaming_lm_head",
            N_PER_WG,
            MT,
            3,
            0,
            /*full_tiles=*/false,
            function);
    };

    auto launch_exact_config = [&](auto n_tile_tag,
                                   auto m_tile_tag,
                                   auto min_blocks_tag,
                                   auto unroll_tag,
                                   auto full_tiles_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr int MB = decltype(min_blocks_tag)::value;
        constexpr int UG = decltype(unroll_tag)::value;
        constexpr bool FULL_TILES = decltype(full_tiles_tag)::value;
        constexpr bool Q6_CHECKED_SPILLS =
            FMT == NVNNI_GEMM_Q6_K && !FULL_TILES && MB == 2 &&
            ((NT == 64 && MT == 64) || (NT == 128 && MT == 32));
        if constexpr (Q6_CHECKED_SPILLS)
        {
            // This source-level rejection prevents the known spilling Q6_K
            // specialization from becoming a launchable override or even a
            // compiled dispatch target. The trainer inventory mirrors it.
            return false;
        }
        else
        {
            const dim3 grid(
                FULL_TILES ? N / NT : (N + NT - 1) / NT,
                FULL_TILES ? M / MT : (M + MT - 1) / MT,
                1);
            const dim3 block(BLOCK_SIZE, 1, 1);
            const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                native_vnni_gemm_kernel<
                    NT, BPT, MT, FMT, MB, UG, FULL_TILES>));
            clear_launch_error();
            native_vnni_gemm_kernel<NT, BPT, MT, FMT, MB, UG, FULL_TILES>
                <<<grid, block, 0, hip_stream>>>(
                    d_A_int8,
                    d_payload,
                    d_block_scales,
                    d_block_mins,
                    d_block_emins,
                    d_output,
                    d_scales_A,
                    M,
                    N,
                    K,
                    d_scales_A_blockwise,
                    serial_m1_kblocks);
            return complete_launch(
                FULL_TILES ? "configured_full_tiles" : "configured_checked",
                NT,
                MT,
                MB,
                UG,
                FULL_TILES,
                function);
        }
    };

    auto launch_config = [&](int n_tile,
                             int m_tile_cfg,
                             int min_blocks,
                             int unroll,
                             bool full_tiles) -> bool
    {
        if ((min_blocks != 1 && min_blocks != 2) ||
            (unroll != 0 && unroll != 1 && unroll != 2 && unroll != 4))
        {
            return false;
        }
        if (full_tiles &&
            ((M % m_tile_cfg) != 0 || (N % n_tile) != 0 ||
             (K % (BPT * 32)) != 0))
        {
            return false;
        }

        auto dispatch_unroll = [&](auto nt, auto mt, auto mb, auto full)
        {
            switch (unroll)
            {
            case 0:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 0>{}, full);
            case 1:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 1>{}, full);
            case 2:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 2>{}, full);
            default:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 4>{}, full);
            }
        };

        auto dispatch_min_blocks = [&](auto nt, auto mt, auto full)
        {
            if (min_blocks == 1)
                return dispatch_unroll(
                    nt, mt, std::integral_constant<int, 1>{}, full);
            return dispatch_unroll(
                nt, mt, std::integral_constant<int, 2>{}, full);
        };

        auto dispatch_m_tile = [&](auto nt, auto full)
        {
            if (m_tile_cfg == 16)
                return dispatch_min_blocks(
                    nt, std::integral_constant<int, 16>{}, full);
            if (m_tile_cfg == 32)
                return dispatch_min_blocks(
                    nt, std::integral_constant<int, 32>{}, full);
            if (n_tile == 64 && m_tile_cfg == 64 && !full_tiles)
                return dispatch_min_blocks(
                    nt, std::integral_constant<int, 64>{}, full);
            return false;
        };

        auto dispatch_n_tile = [&](auto full)
        {
            if (n_tile == 64)
                return dispatch_m_tile(
                    std::integral_constant<int, 64>{}, full);
            if (n_tile == 128 && m_tile_cfg != 64)
                return dispatch_m_tile(
                    std::integral_constant<int, 128>{}, full);
            return false;
        };

        if (full_tiles)
        {
            // The extra proof-carrying template surface is currently compiled
            // only for Q6_K, the production hotspot that justified it.
            if constexpr (FMT == NVNNI_GEMM_Q6_K)
                return dispatch_n_tile(std::true_type{});
            return false;
        }
        return dispatch_n_tile(std::false_type{});
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

    // =========================================================================
    // ENV VAR OVERRIDE DISPATCH (for tuning sweeps)
    //
    // When any LLAMINAR_ROCM_NVNNI_* override is set, bypass the default
    // heuristic and launch the exact kernel variant specified. Full-tile
    // specializations are compiled only for Q6_K; checked candidates retain
    // the existing all-format template surface.
    // =========================================================================
    const auto& env = llaminar2::debugEnv().rocm;
    const bool has_overrides = (env.nvnni_mt > 0 || env.nvnni_unroll >= 0 ||
                                env.nvnni_min_blocks > 0 ||
                                env.nvnni_force_n64 || env.nvnni_force_n128 ||
                                env.nvnni_full_tiles >= 0);
    if (has_overrides) {
        int eff_mt = (env.nvnni_mt > 0) ? env.nvnni_mt : m_tile;
        const int eff_unroll = (env.nvnni_unroll >= 0) ? env.nvnni_unroll : 4;
        const int eff_min_blocks = (env.nvnni_min_blocks > 0) ? env.nvnni_min_blocks : 2;
        const bool eff_n128 = env.nvnni_force_n128 ? true :
                              env.nvnni_force_n64 ? false : use_n128;
        if (eff_n128 && eff_mt > 32)
            eff_mt = 32;
        return launch_config(
            eff_n128 ? 128 : 64,
            eff_mt,
            eff_min_blocks,
            eff_unroll,
            env.nvnni_full_tiles == 1);
    }

    // =========================================================================
    // PATTERN S OPTIMIZED DISPATCH (Q4_0, IQ4_NL)
    //
    // Sweep results (gfx906, MI50/60) show N128/MT32/MB1/UG2 wins 41/45
    // shape×M combinations across Qwen2.5-0.5B/3B/7B.  MIN_BLOCKS=1
    // eliminates VGPR spills (same insight as INT8-VNNI), and UNROLL_G=2
    // reduces register pressure vs the default UG=4.
    //
    // Exceptions at small N (≤1024) / low M:
    //   - 0.5B_AttnOut (N=896):  N64/MT16/MB2 wins at M=128 (+19%)
    //   - 0.5B_FFN_Dn (N=896):  N64/MT32/MB1 wins at M=128 (+24%)
    // These shapes have too few N128-tiles to saturate the GPU.
    //
    // Cooperative N128/MT32/MB1/UG2 also beats streaming for LM_Head shapes
    // (verified: 7B_LM_Head +13% vs streaming Auto).
    // =========================================================================
    if constexpr (!NVNNIGemmTraits<FMT>::is_dual_scale &&
                  !NVNNIGemmTraits<FMT>::is_dual_scale_asym &&
                  !NVNNIGemmTraits<FMT>::is_asymmetric) {
        // Pattern S: simple symmetric (Q4_0, IQ4_NL, Q5_0, IQ3_S, IQ3_XXS, IQ2_XXS)
        // Use MB1/UG2 always for these formats.
        auto launch_mb1_ug2 = [&](auto n_tile_tag, auto m_tile_tag) -> bool
        {
            constexpr int NT = decltype(n_tile_tag)::value;
            constexpr int MT = decltype(m_tile_tag)::value;
            dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
            dim3 block(BLOCK_SIZE, 1, 1);
            const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                native_vnni_gemm_kernel<NT, BPT, MT, FMT, 1, 2>));

            clear_launch_error();
            native_vnni_gemm_kernel<NT, BPT, MT, FMT, 1, 2>
                <<<grid, block, 0, hip_stream>>>(
                    d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                    d_output, d_scales_A, M, N, K, d_scales_A_blockwise,
                    serial_m1_kblocks);

            return complete_launch(
                "simple_symmetric_mb1_ug2",
                NT,
                MT,
                1,
                2,
                /*full_tiles=*/false,
                function);
        };

        using N64_  = std::integral_constant<int, 64>;
        using N128_ = std::integral_constant<int, 128>;
        using M16_  = std::integral_constant<int, 16>;
        using M32_  = std::integral_constant<int, 32>;

        // Small-N exception: N≤1024 with few rows → N64 tiles better
        if (N <= 1024 && M <= 128) {
            // At small N, N128 tiles produce too few WGs.
            // N64/MT16 or MT32 depending on waste
            const int n64_mt = (m_tile <= 16) ? 16 : 32;
            if (n64_mt <= 16) {
                return launch_mb1_ug2(N64_{}, M16_{});
            } else {
                return launch_mb1_ug2(N64_{}, M32_{});
            }
        }

        // Default: N128/MT32/MB1/UG2
        return launch_mb1_ug2(N128_{}, M32_{});
    }

    // Dispatch tags for (N_TILE, M_TILE) combinations
    using N64  = std::integral_constant<int, 64>;
    using N128 = std::integral_constant<int, 128>;
    using M16  = std::integral_constant<int, 16>;
    using M32  = std::integral_constant<int, 32>;
    using M64  = std::integral_constant<int, 64>;
    using M8   = std::integral_constant<int, 8>;

    // Q6_K is the dominant Qwen 3.6 dense-prefill codebook. A complete sweep of
    // every physical graph bucket and all active 35B MoE geometries proves that
    // N128/M32 full tiles are byte-identical and spill-free. M=600 is the sole
    // non-M32-aligned capture; its common winner is checked N128/M16/MB1. Both
    // routes retain serial-M1 K partition boundaries and arithmetic order.
    if constexpr (FMT == NVNNI_GEMM_Q6_K)
    {
        if (M >= 32)
        {
            const bool full_tiles =
                (M % 32) == 0 &&
                (N % 128) == 0 &&
                (K % (BPT * 32)) == 0;
            if (full_tiles)
                return launch_config(128, 32, 2, 4, true);
            return launch_config(128, 16, 1, 4, false);
        }
    }

    // =========================================================================
    // STREAMING DISPATCH: LM_Head-like shapes (massive N, moderate K)
    //
    // For shapes where N >> K and N is very large, the streaming kernel
    // eliminates ALL barriers and LDS, achieving 3-4 wave occupancy.
    //
    // However, this only benefits Pattern S and D formats (light decode).
    // Pattern A and DA formats have heavy decode that causes register
    // spilling in the streaming kernel — cooperative is 25-130% faster.
    //
    // N threshold is format-dependent:
    //   Pattern D (is_dual_scale): N >= 8192.  The cooperative kernel suffers
    //     72-132 bytes scratch at N128/M32 due to temp_acc[4][4] + scale arrays
    //     + LDS pipeline staging.  The streaming kernel uses temp_acc[M_TILE]
    //     (1D array) → 4-wave occupancy, zero scratch.  Tested: streaming wins
    //     by 15-30% at N=11008 (3B_FFN_Up) for Q3_K, Q6_K, IQ2_S, IQ2_XS.
    //   Pattern S: N >= 16384.  Cooperative is scratch-free at N128/M32 so only
    //     massive-N shapes (LM_Head) benefit from streaming's barrier-free design.
    //
    // K <= 4096 is a safety check (streaming loads A per-thread, not via LDS).
    // =========================================================================
    if constexpr (NVNNIGemmTraits<FMT>::prefer_streaming) {
        constexpr int streaming_n_threshold =
            NVNNIGemmTraits<FMT>::is_dual_scale ? 8192 : 16384;
        if (N >= streaming_n_threshold && K <= 4096) {
            // IQ1_M uses the same M8/three-wave tile. An M16/two-wave variant
            // was measured across M=64..256 and regressed by 25-38%; the lost
            // latency hiding outweighed reduced decode and activation loads.
            if constexpr (FMT == NVNNI_GEMM_IQ1_M) {
                return launch_streaming(M8{});
            // Other Pattern D formats use M8: temp_acc[8] = 8 VGPRs.
            // A-data per thread = 8×K/4×4 = 2K bytes ≤ 16KB.
            } else if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale) {
                return launch_streaming(M8{});
            } else {
                return launch_streaming(M16{});
            }
        }
    }

    // NOTE: Pattern DA (Q2_K) must NOT use streaming — tested at N=11008
    // and it was 2-3× slower (Q2_K: 1390μs vs 565μs cooperative).  The Pattern DA
    // decode + accumulation (temp_acc + temp_sum + emin per half) is
    // too register-heavy for streaming's per-thread A-loading model.
    // IQ1_M was moved from DA to D (emins always zero) and is now streaming-eligible.

    // Pattern A streaming with cooperative sum_a precompute:
    // Only for formats where streaming is proven faster than cooperative.
    // - IQ1_S: +18% (heavy grid decode benefits hugely from 4-wave occupancy)
    // - Q4_1:  +2.5% (light decode, small but consistent improvement)
    // NOT Q5_1: -16% regression (heavy 5-bit qh decode + 20B payload → VGPRs
    //           push past 4-wave threshold, and extra HBM for larger payload)
    if constexpr (FMT == NVNNI_GEMM_Q4_1 || FMT == NVNNI_GEMM_IQ1_S) {
        if (N >= 8192 && K <= 4096) {
            return launch_streaming(M16{});
        }
    }

    if (use_n128) {
        // Dual-scale-asymmetric formats (Pattern DA: Q2_K only):
        // N128/M16/2-wave is the best tile for Pattern DA.
        // IQ1_M moved to Pattern D (emins always zero) — uses hybrid dispatch below.
        //
        // Tested alternatives for Q2_K (all regressed):
        //   - N128/M32/2w: +200 bytes scratch from temp_acc+temp_sum+emin staging
        //   - N64/M16/3w:  +55% slower! N_PER_THREAD 2→1 kills ILP in the heavy
        //                  DA accumulation (sdot4 interleaving across N-columns)
        //   - Streaming:   +95% slower! Per-thread A-loading + DA register pressure
        //   - Split-emin:  Went to 1-wave! 4 separate loops = too much code for RA
        //   - launch3 (3w hint): compiler still achieves 2w only, no perf change
        if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale_asym) {
            return launch(N128{}, M16{});
        }

        // Pattern A (asymmetric: Q4_1, Q5_1, IQ1_S): force M32/2-wave on N128.
        // N128/M16/3w spills 40-72 bytes scratch to achieve 3-wave occupancy.
        // N128/M32/2w is scratch-free (126-128 VGPRs) with better A-data reuse.
        // Tested hybrid M16/M32: no improvement at M=32 (142→143μs for Q4_1),
        // minor regression at M=32 for Q5_1 (147→151μs).  Scratch cost ≈ 3-wave gain.
        if constexpr (NVNNIGemmTraits<FMT>::is_asymmetric) {
            return launch(N128{}, M32{});
        }

        // Hybrid N128 dispatch: M16/3-wave for small grids, M32/2-wave for large
        //
        // Guard with if constexpr: only dual_scale (non-asym) formats reach here
        // at runtime (Q6_K, Q3_K, IQ2_S, IQ2_XS, IQ1_M), but without the guard
        // the compiler still instantiates launch3 for all formats — causing
        // occupancy failures for Q2_K (drops to 1-wave at N128/M16/3w).
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
        if constexpr (!NVNNIGemmTraits<FMT>::is_dual_scale_asym &&
                      !NVNNIGemmTraits<FMT>::is_asymmetric) {
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
        }
    } else {
        // Dual-scale-asymmetric formats (Pattern DA: Q2_K only): force M16 on N64.
        // N64/M16/3w: VGPR=84, 84B scratch (compiler spills to hit 3-wave).
        // N64/M16/2w: VGPR=106, 0 scratch.  Use 2-wave to avoid scratch.
        // IQ1_M now uses Pattern D and falls through to the hybrid dispatch below.
        if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale_asym) {
            return launch(N64{}, M16{});
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

        // Cap M_TILE at 32 for Pattern A and Pattern D on N64 path.
        //
        // Pattern A (asymmetric: Q4_1, Q5_1, IQ1_S):
        //   N64/M64 causes scratch spill (8-16B) and saturates VGPRs at 127-128.
        //   N64/M32 is clean: 108-111 VGPRs, zero scratch, 2-wave.
        //
        // Pattern D (dual-scale: Q6_K, Q3_K, IQ2_S, IQ2_XS):
        //   N64/M64 causes 124 bytes scratch spill for Pattern D (M_PER_THREAD=4
        //   with two-phase still needs too many registers at THREADS_M=16).
        int effective_m_tile = m_tile;
        if constexpr (NVNNIGemmTraits<FMT>::is_dual_scale ||
                       NVNNIGemmTraits<FMT>::is_asymmetric) {
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
bool LLAMINAR_ROCM_NVNNI_SHARD_SYMBOL(
    rocmGemm_native_vnni_fp32_shard_)(
    const int8_t*  d_A_int8,       // [M × K] INT8 activations
    const uint8_t* d_payload,      // [blocks_per_row × N × PB] native-VNNI payload
    const void*    d_block_scales, // [blocks_per_row × N] __half per-block scales
    const void*    d_block_mins,   // [blocks_per_row × N] __half per-block mins/aux (nullable)
    const void*    d_block_emins,  // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, nullable)
    float*         d_output,       // [M × N] FP32 output
    const float*   d_scales_A,     // [M] per-row activation scales
    const float*   d_scales_A_blockwise, // [M × blocks_per_row] per-block activation scales
    int M, int N, int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || K % 32 != 0)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const auto* scales = reinterpret_cast<const uint16_t*>(d_block_scales);
    const auto* mins   = reinterpret_cast<const uint16_t*>(d_block_mins);
    const auto* emins  = reinterpret_cast<const uint32_t*>(d_block_emins);

    #define LAUNCH_GEMM(FMT) \
        return launch_native_vnni_gemm<FMT>( \
            d_A_int8, d_payload, scales, mins, emins, d_output, d_scales_A, \
            d_scales_A_blockwise, M, N, K, device_id, stream, \
            arithmetic_policy_codebook_id)

    switch (codebook_id)
    {
#if LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 0
    case NVNNI_GEMM_Q4_0:     LAUNCH_GEMM(NVNNI_GEMM_Q4_0);
    case NVNNI_GEMM_IQ4_NL:   LAUNCH_GEMM(NVNNI_GEMM_IQ4_NL);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 1
    case NVNNI_GEMM_Q4_1:     LAUNCH_GEMM(NVNNI_GEMM_Q4_1);
    case NVNNI_GEMM_Q5_0:     LAUNCH_GEMM(NVNNI_GEMM_Q5_0);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 2
    case NVNNI_GEMM_Q5_1:     LAUNCH_GEMM(NVNNI_GEMM_Q5_1);
    case NVNNI_GEMM_Q6_K:     LAUNCH_GEMM(NVNNI_GEMM_Q6_K);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 3
    case NVNNI_GEMM_Q3_K:     LAUNCH_GEMM(NVNNI_GEMM_Q3_K);
    case NVNNI_GEMM_Q2_K:     LAUNCH_GEMM(NVNNI_GEMM_Q2_K);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 4
    case NVNNI_GEMM_IQ3_S:    LAUNCH_GEMM(NVNNI_GEMM_IQ3_S);
    case NVNNI_GEMM_IQ3_XXS:  LAUNCH_GEMM(NVNNI_GEMM_IQ3_XXS);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 5
    case NVNNI_GEMM_IQ2_S:    LAUNCH_GEMM(NVNNI_GEMM_IQ2_S);
    case NVNNI_GEMM_IQ2_XS:   LAUNCH_GEMM(NVNNI_GEMM_IQ2_XS);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 6
    case NVNNI_GEMM_IQ2_XXS:  LAUNCH_GEMM(NVNNI_GEMM_IQ2_XXS);
    case NVNNI_GEMM_IQ1_S:    LAUNCH_GEMM(NVNNI_GEMM_IQ1_S);
#elif LLAMINAR_ROCM_NATIVE_VNNI_SHARD == 7
    case NVNNI_GEMM_IQ1_M:    LAUNCH_GEMM(NVNNI_GEMM_IQ1_M);
    case NVNNI_GEMM_Q8_0:     LAUNCH_GEMM(NVNNI_GEMM_Q8_0);
    case NVNNI_GEMM_EXPANDED_INT8_MIN:
        LAUNCH_GEMM(NVNNI_GEMM_EXPANDED_INT8_MIN);
#endif
    default:
        fprintf(
            stderr,
            "[rocmGemm_native_vnni_fp32_shard_%d] codebook_id=%d does not "
            "belong to this physical shard\n",
            LLAMINAR_ROCM_NATIVE_VNNI_SHARD,
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
bool LLAMINAR_ROCM_NVNNI_SHARD_SYMBOL(
    rocmInitIQGridTables_gemm_shard_)(
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

#undef LLAMINAR_ROCM_NVNNI_SHARD_SYMBOL
#undef LLAMINAR_ROCM_NVNNI_CONCAT
#undef LLAMINAR_ROCM_NVNNI_CONCAT_IMPL
