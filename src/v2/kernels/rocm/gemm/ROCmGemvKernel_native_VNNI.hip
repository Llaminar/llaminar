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

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../utils/DebugEnv.h"

#include <atomic>

// ============================================================================
// Runtime tuning overrides for native-VNNI GEMV dispatch.
// Values < 0 mean "use default heuristic".
// ============================================================================
static std::atomic<int> g_native_vnni_kb_override{-1};
static std::atomic<int> g_native_vnni_target_waves_override{-1};

extern "C" void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu)
{
    g_native_vnni_kb_override.store(kb, std::memory_order_relaxed);
    g_native_vnni_target_waves_override.store(target_waves_per_cu, std::memory_order_relaxed);
}

extern "C" void rocmGemv_native_vnni_reset_tuning_overrides()
{
    rocmGemv_native_vnni_set_tuning_overrides(-1, -1);
}

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
    NVNNI_Q8_0   = 19,  // 8-bit symmetric, direct INT8 payload (32 bytes)
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
    // Dual-scale + asymmetric: split sum_a correction with per-sub-block emins
    // IQ1_M uses Pattern D (not DA): has ±IQ1S_DELTA per-group correction (handled separately)
    static constexpr bool is_dual_scale_asym = (FMT == NVNNI_Q2_K);
    // Q2_K: emins in separate global array
    static constexpr bool needs_emin_array = (FMT == NVNNI_Q2_K);
    // All scales/mins now in global arrays — no embedded data in payload
    static constexpr bool has_embedded_scales = false;
    static constexpr int  embedded_scale_offset = -1;
    static constexpr int  embedded_min_offset = -1;
    static constexpr bool is_5bit       = (FMT == NVNNI_Q5_0 || FMT == NVNNI_Q5_1);
    static constexpr bool is_6bit       = (FMT == NVNNI_Q6_K);
    static constexpr bool is_3bit       = (FMT == NVNNI_Q3_K);
    static constexpr bool is_2bit       = (FMT == NVNNI_Q2_K);
    static constexpr bool is_8bit       = (FMT == NVNNI_Q8_0);
    static constexpr bool uses_lut      = (FMT == NVNNI_IQ4_NL);
    static constexpr int  payload_bytes =
        is_8bit ? 32 :                                // Q8_0: 32 raw INT8 values
        is_iq1_grid ? 6 :                            // IQ1_S(6), IQ1_M(6): qs[4]+qh[2] only
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 12 : is_2bit ? 8 : 16;
    // Centering offset for symmetric formats; asymmetric formats don't center
    static constexpr int center_4bit = (FMT == NVNNI_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NVNNI_Q5_0) ? 16 : 0;
    // IQ LDS grid cache properties (only meaningful for IQ grid/IQ1 grid formats)
    static constexpr int grid_entries =
        (FMT == NVNNI_IQ3_S) ? 512 : (FMT == NVNNI_IQ3_XXS) ? 256 :
        (FMT == NVNNI_IQ2_S) ? 1024 : (FMT == NVNNI_IQ2_XS) ? 512 : (FMT == NVNNI_IQ2_XXS) ? 256 :
        (FMT == NVNNI_IQ1_S || FMT == NVNNI_IQ1_M) ? 2048 : 0;
    static constexpr int grid_entry_bytes = is_iq3_grid ? 4 : (is_iq2_grid || is_iq1_grid) ? 8 : 0;
    static constexpr int grid_lds_bytes = grid_entries * grid_entry_bytes;
};

template <NativeVNNIFormat FMT>
struct NVNNIGroupedMoESupported {
    static constexpr bool value =
        FMT == NVNNI_Q4_0 || FMT == NVNNI_IQ4_NL || FMT == NVNNI_Q4_1 ||
        FMT == NVNNI_Q5_0 || FMT == NVNNI_Q5_1 || FMT == NVNNI_Q6_K ||
        FMT == NVNNI_Q3_K || FMT == NVNNI_Q2_K || FMT == NVNNI_IQ3_S ||
        FMT == NVNNI_IQ3_XXS || FMT == NVNNI_IQ2_S || FMT == NVNNI_IQ2_XS ||
        FMT == NVNNI_IQ2_XXS || FMT == NVNNI_IQ1_S || FMT == NVNNI_IQ1_M ||
        FMT == NVNNI_Q8_0;
};

// ============================================================================
// IQ grid decode helpers
// ============================================================================

// Conditionally negate 4 unsigned bytes packed in uint32_t using 4 sign bits.
// sign_lo4 bit j → element j: 0=positive, 1=negative.
// Safe: all IQ grid values > 0, so XOR+add never carries across bytes.
//
// O4: SWAR multiply-based bit-to-byte expansion replaces per-bit BFE chain.
//
// Previous ISA: 4× v_bfe_i32 + 4× v_and + 3× v_lshl_or ≈ 15 VALU per call.
// New ISA:      1× v_mul_u24 + 1× v_and + 1× v_lshl + 1× v_sub + 1× v_xad ≈ 5 VALU (all 1-cyc).
//
// Math:
//   spread = (sign_lo4 * 0x00204081) & 0x01010101
//     Scatters bit[i] into byte[i] position 0.  The constant 0x00204081 has
//     bits 0, 7, 14, 21 set so that bit k of sign_lo4 lands in byte k's bit 0
//     after masking.  No cross-byte carry because
//     max(sign_lo4) = 15 and 15 * 0x00204081 = 0x01E3C78F (each byte < 256).
//
//   mask = (spread << 8) - spread
//     Expands 0x00→0x00, 0x01→0xFF per byte.  Algebraically equivalent to
//     spread * 255, but compiles to v_lshlrev_b32 + v_sub_u32 (2× 1-cycle)
//     instead of v_mul_lo_u32 (1× 4-cycle).  Cross-byte borrows are benign
//     because each byte in spread is 0 or 1, so (shifted_byte - spread_byte)
//     always produces 0x00 or 0xFF with exactly the carries needed by
//     adjacent byte lanes.
//
//   Result: (grid4 ^ mask) + spread  is two's complement conditional negate.
//     (carry = mask & 0x01010101 = spread, since mask bytes are 0x00 or 0xFF)
__device__ inline uint32_t iq_apply_signs_4(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t spread = (static_cast<uint32_t>(sign_lo4 & 0xFu) * 0x00204081u) & 0x01010101u;
    // mask = (spread << 8) - spread  ≡  spread * 255
    // Inline asm prevents the compiler from strength-reducing this back to
    // v_mul_lo_u32 (4-cycle).  The shift+sub form uses v_lshlrev_b32 +
    // v_sub_u32 (2× 1-cycle), eliminating 11 stall cycles per block.
    uint32_t mask;
    asm("v_lshlrev_b32 %0, 8, %1\n\t"
        "v_sub_u32 %0, %0, %1"
        : "=v"(mask) : "v"(spread));
    return (grid4 ^ mask) + spread;
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
// O2: Grid read helpers — abstract constant-memory vs LDS grid access
//
// Template parameter USE_LDS selects the data source at compile time:
//   false → reads from __constant__ device memory (non-LDS kernel path)
//   true  → reads from LDS-cached copy (LDS kernel path)
// This allows the iq_decode_accumulate_block helper to serve both kernels.
// ============================================================================
template <NativeVNNIFormat FMT, bool USE_LDS>
__device__ __forceinline__ uint32_t iq3_grid_read(int idx,
    [[maybe_unused]] const uint32_t* lds32)
{
    if constexpr (USE_LDS) return lds32[idx];
    else return iq3_grid_lookup<FMT>(idx);
}

template <NativeVNNIFormat FMT, bool USE_LDS>
__device__ __forceinline__ uint64_t iq2_grid_read(int idx,
    [[maybe_unused]] const uint64_t* lds64)
{
    if constexpr (USE_LDS) return lds64[idx];
    else return iq2_grid_lookup<FMT>(idx);
}

template <bool USE_LDS>
__device__ __forceinline__ uint64_t iq1_grid_read(int idx,
    [[maybe_unused]] const uint64_t* lds64)
{
    if constexpr (USE_LDS) return lds64[idx];
    else return iq1_grid_lookup(idx);
}

// ============================================================================
// O2: Per-block IQ decode + accumulate helper (2-block unroll building block)
//
// Processes one 32-element quantized block for IQ grid formats (IQ3, IQ2, IQ1).
// Extracted from the main kernel's inner loop so the 2-block unrolled loop can
// call it twice per iteration, giving the compiler visibility into both blocks'
// grid lookups simultaneously.  The instruction scheduler then overlaps grid
// load latency from block N+1 with compute from block N.
//
// Template params:
//   CPT     — columns per thread (1 or 2)
//   FMT     — NativeVNNIFormat tag (selects decode + accumulate path)
//   USE_LDS — false: read grid from constant memory; true: read from LDS
// ============================================================================
template <int CPT, NativeVNNIFormat FMT, bool USE_LDS = false>
__device__ __forceinline__ void iq_decode_accumulate_block(
    const int32_t* __restrict__ a4,
    [[maybe_unused]] int32_t sum_a,
    const uint8_t*  __restrict__ d_payload,
    const uint16_t* __restrict__ d_block_scales,
    const uint16_t* __restrict__ d_block_mins,
    int n_base, int N, size_t b_offset_N,
    float* f_acc,
    [[maybe_unused]] const uint32_t* lds32 = nullptr,
    [[maybe_unused]] const uint64_t* lds64 = nullptr,
    float scale_ab = 1.0f)
{
    using Traits = NVNNITraits<FMT>;
    static_assert(Traits::is_iq_grid || Traits::is_iq1_grid,
                  "iq_decode_accumulate_block only for IQ grid formats");

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            const size_t linear = b_offset_N + static_cast<size_t>(n);
            const uint8_t* payload = d_payload + linear * Traits::payload_bytes;

            // Scale always from global array
            uint16_t scale_bits = d_block_scales[linear];

            int32_t packed_groups[8];

            // ---- IQ grid decode ----
            if constexpr (Traits::is_iq_grid)
            {
                constexpr int n_qs = Traits::is_iq3_grid ? 8 : 4;
                constexpr int sign_off = n_qs + (Traits::has_qh ? 1 : 0);
                const uint8_t qh_byte = Traits::has_qh ? payload[n_qs] : 0;

                if constexpr (Traits::is_iq3_grid && !Traits::has_qh)
                {
                    // O5: Wide payload load for IQ3_XXS (12B = 3 dwords).
                    //
                    // The compiler generates 12× global_load_ubyte for 12 bytes
                    // of payload when each byte feeds a separate dependency chain.
                    // Manually loading as 3 dwords + extracting via shift+mask
                    // replaces 12 VMEM with 3 VMEM + ~16 cheap 1-cycle VALU.
                    //
                    // ISA impact (IQ3_XXS LDS inner loop):
                    //   Before: 12× global_load_ubyte + 1× global_load_ushort = 13 VMEM
                    //   After:  3× global_load_dword  + 1× global_load_ushort =  4 VMEM
                    //                                                           (−69%)
                    //
                    // Payload layout: [qs0..qs7][signs0..signs3] = 12 bytes
                    // payload_bytes=12 is 4-byte aligned → dword loads are safe.
                    const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                    const uint32_t w0 = pw[0];  // qs[0..3]
                    const uint32_t w1 = pw[1];  // qs[4..7]
                    const uint32_t w2 = pw[2];  // signs[0..3]

                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        const uint32_t word = (g < 4) ? w0 : w1;
                        const int idx = (word >> ((g & 3) * 8)) & 0xFF;
                        const uint32_t grid4 = iq3_grid_read<FMT, USE_LDS>(idx, lds32);
                        // sign byte g/2: byte (g/2) of w2.  Max g/2=3 → shift 24 OK.
                        const uint8_t sb = static_cast<uint8_t>(w2 >> ((g / 2) * 8));
                        const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                        packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sn));
                    }
                }
                else if constexpr (Traits::is_iq3_grid && Traits::has_qh)
                {
                    // O6: Wide payload loads for IQ3_S (13B = 3 dwords + 1 ubyte).
                    //
                    // The compiler generates dwordx2(0-7) + ushort(8-9) + 3×ubyte(10,11,12)
                    // = 5 payload VMEM.  Loading bytes 0-11 as 3 consecutive dwords
                    // (which the compiler coalesces to dwordx3) reduces payload to:
                    //   1× dwordx3 + 1× ubyte = 2 payload VMEM  (−60%)
                    //
                    // Payload layout: [qs0..qs7][qh][signs0..signs3] = 13 bytes
                    const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                    const uint32_t w0 = pw[0];  // qs[0..3]
                    const uint32_t w1 = pw[1];  // qs[4..7]
                    const uint32_t w2 = pw[2];  // {qh, signs[0], signs[1], signs[2]}
                    const uint8_t signs3 = payload[12];  // signs[3]

                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        const uint32_t word = (g < 4) ? w0 : w1;
                        int idx = (word >> ((g & 3) * 8)) & 0xFF;
                        idx |= (static_cast<int>((w2 >> g) & 1) << 8);  // qh bit g from w2 low byte
                        const uint32_t grid4 = iq3_grid_read<FMT, USE_LDS>(idx, lds32);
                        // signs[0..2] in w2 bytes 1,2,3; signs[3] separate
                        const uint8_t sb = (g < 6)
                            ? static_cast<uint8_t>(w2 >> ((g / 2 + 1) * 8))
                            : signs3;
                        const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                        packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sn));
                    }
                }
                else if constexpr (Traits::is_iq2_grid && !Traits::has_qh)
                {
                    // O5: Wide payload load for IQ2_XXS (8B = 2 dwords).
                    //
                    // ISA impact (IQ2_XXS LDS inner loop):
                    //   Before: 8× global_load_ubyte + 1× global_load_ushort = 9 VMEM
                    //   After:  2× global_load_dword + 1× global_load_ushort = 3 VMEM
                    //                                                         (−67%)
                    //
                    // Payload layout: [qs0..qs3][signs0..signs3] = 8 bytes
                    // payload_bytes=8 is 4-byte aligned → dword loads are safe.
                    const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                    const uint32_t w_qs    = pw[0];  // qs[0..3]
                    const uint32_t w_signs = pw[1];  // signs[0..3]

                    #pragma unroll
                    for (int l = 0; l < 4; ++l)
                    {
                        const int idx = (w_qs >> (l * 8)) & 0xFF;
                        const uint64_t grid8 = iq2_grid_read<FMT, USE_LDS>(idx, lds64);
                        const uint32_t g_lo = static_cast<uint32_t>(grid8);
                        const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                        const uint8_t signs = static_cast<uint8_t>(w_signs >> (l * 8));
                        packed_groups[l * 2]     = static_cast<int32_t>(
                            iq_apply_signs_4(g_lo, signs & 0xF));
                        packed_groups[l * 2 + 1] = static_cast<int32_t>(
                            iq_apply_signs_4(g_hi, signs >> 4));
                    }
                }
                else // is_iq2_grid && has_qh (IQ2_XS, IQ2_S)
                {
                    // O6: Wide payload loads for IQ2_XS/IQ2_S (9B = dwordx2 + ubyte).
                    //
                    // The compiler generates dword(0-3) + ushort(4-5) + 3×ubyte(6,7,8)
                    // = 5 payload VMEM.  Loading bytes 0-7 as dwordx2 reduces payload to:
                    //   1× dwordx2 + 1× ubyte = 2 payload VMEM  (−60%)
                    //
                    // Payload layout: [qs0..qs3][qh][signs0..signs3] = 9 bytes
                    const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                    const uint32_t w_qs       = pw[0];  // qs[0..3]
                    const uint32_t w_qh_signs = pw[1];  // {qh, signs[0], signs[1], signs[2]}
                    const uint8_t signs3 = payload[8];   // signs[3]

                    #pragma unroll
                    for (int l = 0; l < 4; ++l)
                    {
                        int idx = (w_qs >> (l * 8)) & 0xFF;
                        if constexpr (Traits::qh_bits_per_lookup == 1)
                            idx |= (static_cast<int>((w_qh_signs >> l) & 1) << 8);
                        else if constexpr (Traits::qh_bits_per_lookup == 2)
                            idx |= (static_cast<int>((w_qh_signs >> (2 * l)) & 3) << 8);
                        const uint64_t grid8 = iq2_grid_read<FMT, USE_LDS>(idx, lds64);
                        const uint32_t g_lo = static_cast<uint32_t>(grid8);
                        const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                        // signs[0..2] in w_qh_signs bytes 1,2,3; signs[3] separate
                        const uint8_t signs_byte = (l < 3)
                            ? static_cast<uint8_t>(w_qh_signs >> ((l + 1) * 8))
                            : signs3;
                        packed_groups[l * 2]     = static_cast<int32_t>(
                            iq_apply_signs_4(g_lo, signs_byte & 0xF));
                        packed_groups[l * 2 + 1] = static_cast<int32_t>(
                            iq_apply_signs_4(g_hi, signs_byte >> 4));
                    }
                }
            }
            else // is_iq1_grid
            {
                if constexpr (FMT == NVNNI_IQ1_S)
                {
                    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                                           | (static_cast<uint16_t>(payload[5]) << 8);
                    #pragma unroll
                    for (int l = 0; l < 4; ++l)
                    {
                        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
                        const uint64_t grid8 = iq1_grid_read<USE_LDS>(idx, lds64);
                        packed_groups[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(grid8));
                        packed_groups[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
                    }
                }
                else // IQ1_M
                {
                    const uint8_t qh0 = payload[4], qh1 = payload[5];
                    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
                    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
                    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
                    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);
                    uint64_t g;
                    g = iq1_grid_read<USE_LDS>(idx0, lds64);
                    packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(g));
                    packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                    g = iq1_grid_read<USE_LDS>(idx1, lds64);
                    packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(g));
                    packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                    g = iq1_grid_read<USE_LDS>(idx2, lds64);
                    packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(g));
                    packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                    g = iq1_grid_read<USE_LDS>(idx3, lds64);
                    packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(g));
                    packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
                }
            }

            // ---- sdot4 accumulation + scale ----
            if constexpr (Traits::is_dual_scale)
            {
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
                f_acc[c] += (static_cast<float>(acc_lo) * scale_lo
                          + static_cast<float>(acc_hi) * scale_hi) * scale_ab;

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

                    // Q2_K: per-sub-block emin correction
                    constexpr float min_lo = 0.0f;
                    constexpr float min_hi = 0.0f;
                    f_acc[c] += (static_cast<float>(sum_a_lo) * min_lo
                              + static_cast<float>(sum_a_hi) * min_hi) * scale_ab;
                }

                // IQ1_M: ±IQ1S_DELTA per 8-element group correction.
                // True weight[j] = dl * (grid[j] + delta), but the sdot4 above
                // only accumulates dl * dot(a, grid).  We add the missing
                // dl * delta * sum_a_per_group for each of the 4 groups.
                // Delta signs: qh0 bit 3 (group 0), qh0 bit 7 (group 1),
                //              qh1 bit 3 (group 2), qh1 bit 7 (group 3).
                if constexpr (FMT == NVNNI_IQ1_M)
                {
                    constexpr float IQ1S_DELTA = 0.125f;
                    constexpr int32_t ones = 0x01010101;
                    const uint8_t qh0 = payload[4], qh1 = payload[5];

                    // Per-group sum of activation bytes (8 int8 elements per group)
                    int32_t sa_g0 = __builtin_amdgcn_sdot4(a4[1], ones,
                                    __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
                    int32_t sa_g1 = __builtin_amdgcn_sdot4(a4[3], ones,
                                    __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
                    int32_t sa_g2 = __builtin_amdgcn_sdot4(a4[5], ones,
                                    __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
                    int32_t sa_g3 = __builtin_amdgcn_sdot4(a4[7], ones,
                                    __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

                    f_acc[c] += ((d0 * static_cast<float>(sa_g0) + d1 * static_cast<float>(sa_g1)) * scale_lo
                              + (d2 * static_cast<float>(sa_g2) + d3 * static_cast<float>(sa_g3)) * scale_hi) * scale_ab;
                }
            }
            else
            {
                int32_t block_acc = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

                const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                f_acc[c] += static_cast<float>(block_acc) * block_d * scale_ab;

                if constexpr (Traits::is_asymmetric)
                {
                    const uint16_t min_bits = d_block_mins[linear];
                    const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                    f_acc[c] += static_cast<float>(sum_a) * block_m * scale_ab;
                }
            }
        }
    }
}


// ============================================================================
// Native-VNNI GEMV kernel: lossless decode with per-block FP16 scales
//
// Supports all ≤6-bit quantized formats via compile-time NativeVNNIFormat tag:
//   Q-quants:  Q4_0, Q4_1, Q5_0, Q5_1
//   K-quants:  Q2_K, Q3_K, Q4_K (Q4_1-like), Q5_K (Q5_1-like), Q6_K
//   IQ-quants: IQ4_NL, IQ4_XS (IQ4_NL-like), IQ3_S, IQ3_XXS,
//              IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M
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
    const uint32_t*  __restrict__ d_block_emins,   // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, else nullptr)
    float*           __restrict__ d_C_fp32,        // [N] output (FP32), used when kblocks=1
    float*           __restrict__ d_partial_fp32,  // [KB × N] partials, used when kblocks>1
    const float*     __restrict__ d_scale_A,       // [1] per-row activation scale (CK fallback, used when d_scale_A_blockwise==nullptr)
    const float*     __restrict__ d_scale_A_blockwise, // [blocks_per_row] per-block activation scales
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
        if constexpr (Traits::is_iq2_grid)
        {
            // ================================================================
            // O2: 2-block unrolled loop for IQ2 grid formats (non-LDS path)
            //
            // Processes two K-blocks per iteration so the compiler can overlap
            // constant-memory grid lookup latencies between blocks.  Block
            // N+1's grid loads are issued while block N's dot4 chain executes,
            // hiding ~100 cycles of dependent-load latency per block.
            //
            // Restricted to IQ2 (not IQ3/IQ1) based on benchmarking:
            //   IQ2_S/XS: +5-12% on non-LDS shapes (4-8KB grid, fits in L1)
            //   IQ3: neutral (8 lookups/block already saturate L1 bandwidth)
            //   IQ1: regression (16KB grid thrashes L1, extra loads worsen it)
            //
            // Not applied to LDS kernel: LDS reads have ~20 cycle latency
            // with 4 independent reads already overlapping within one block,
            // so 2-block unroll adds overhead without latency benefit.
            // ================================================================
            int b = b_start;
            for (; b + 1 < b_end; b += 2)
            {
                const int32_t* a4_0 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);
                const int32_t* a4_1 = reinterpret_cast<const int32_t*>(d_A_int8 + (b + 1) * 32);

                int32_t sum_a_0 = 0, sum_a_1 = 0;
                if constexpr (Traits::is_asymmetric)
                {
                    constexpr int32_t ones = 0x01010101;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a_0 = __builtin_amdgcn_sdot4(a4_0[g], ones, sum_a_0, false);
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a_1 = __builtin_amdgcn_sdot4(a4_1[g], ones, sum_a_1, false);
                }

                // Decode + accumulate both blocks (compiler sees both sets of
                // grid loads and can schedule them for maximum overlap)
                const float scale_ab_0 = d_scale_A_blockwise ? d_scale_A_blockwise[b] : 1.0f;
                const float scale_ab_1 = d_scale_A_blockwise ? d_scale_A_blockwise[b + 1] : 1.0f;
                iq_decode_accumulate_block<CPT, FMT, false>(
                    a4_0, sum_a_0, d_payload, d_block_scales, d_block_mins,
                    n_base, N, static_cast<size_t>(b) * static_cast<size_t>(N), f_acc,
                    nullptr, nullptr, scale_ab_0);
                iq_decode_accumulate_block<CPT, FMT, false>(
                    a4_1, sum_a_1, d_payload, d_block_scales, d_block_mins,
                    n_base, N, static_cast<size_t>(b + 1) * static_cast<size_t>(N), f_acc,
                    nullptr, nullptr, scale_ab_1);
            }
            // Handle odd remainder block
            if (b < b_end)
            {
                const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);
                int32_t sum_a = 0;
                if constexpr (Traits::is_asymmetric)
                {
                    constexpr int32_t ones = 0x01010101;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);
                }
                iq_decode_accumulate_block<CPT, FMT, false>(
                    a4, sum_a, d_payload, d_block_scales, d_block_mins,
                    n_base, N, static_cast<size_t>(b) * static_cast<size_t>(N), f_acc,
                    nullptr, nullptr, d_scale_A_blockwise ? d_scale_A_blockwise[b] : 1.0f);
            }
        }
        else
        {
        // ---- Original single-block loop for non-IQ formats ----
        for (int b = b_start; b < b_end; ++b)
        {
            // Per-block activation scale (blockwise) or 1.0 (fallback)
            const float scale_ab = d_scale_A_blockwise ? d_scale_A_blockwise[b] : 1.0f;

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
                        // v_perm_b32(src0, src1, sel) selects 4 bytes from {src0[3:0], src1[3:0]}
                        // using sel[i] & 0x7 for each output byte i (0-3 → src0, 4-7 → src1).
                        //
                        // Two v_perm calls look up the low (0-7) and high (8-15) codebook halves
                        // using nibble & 7 as selector.  A third v_perm byte-muxes the results:
                        //   mux_sel = 0x03020100 | ((nibble >> 1) & 0x04040404)
                        // Byte i of mux_sel = i (pick r_lo) when nibble < 8, or i+4 (pick r_hi)
                        // when nibble >= 8.  This replaces the old shift+and+mul(4-cyc)+bfi chain
                        // with shift+and+or+perm — all 1-cycle ops.
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
                        constexpr uint32_t MUX_IDENTITY = 0x03020100u;  // byte-identity permutation
                        constexpr uint32_t MUX_BIT3_MASK = 0x04040404u; // bit3 → +4 offset per byte

                        // NOTE: __builtin_amdgcn_perm(a, b, sel) on HIP/ROCm maps sel 0-3 → b,
                        // sel 4-7 → a. We pass (B, A) so sel 0-3 picks from A, sel 4-7 from B.
                        // Mux perm uses (r_hi, r_lo) so bit3=0 → r_lo, bit3=1 → r_hi.
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

                            // --- Low nibbles (elements [4g..4g+3]) ---
                            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                            {
                                const uint32_t sel = lo_nibs & 0x07070707u;
                                const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, sel);
                                const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, sel);
                                const uint32_t mux = MUX_IDENTITY | ((lo_nibs >> 1) & MUX_BIT3_MASK);
                                packed_groups[g] = static_cast<int32_t>(__builtin_amdgcn_perm(r_hi, r_lo, mux));
                            }

                            // --- High nibbles (elements [4g+16..4g+19]) ---
                            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                            {
                                const uint32_t sel = hi_nibs & 0x07070707u;
                                const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, sel);
                                const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, sel);
                                const uint32_t mux = MUX_IDENTITY | ((hi_nibs >> 1) & MUX_BIT3_MASK);
                                packed_groups[g + 4] = static_cast<int32_t>(__builtin_amdgcn_perm(r_hi, r_lo, mux));
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
                        // Q3_K: native 12B layout — [8B pre-transposed 2-bit][4B grouped hbits]
                        //
                        // Decode: Q2_K-style shift+mask for low2, BIT_SPREAD scatter for hbits,
                        //         XOR-SUB-XOR centering to [-4..+3]
                        //
                        // Payload layout (pre-transposed during CPU packing):
                        //   [0..3]:  half 0 — byte[j] = e[j]&3 | (e[j+4]&3)<<2 | (e[j+8]&3)<<4 | (e[j+12]&3)<<6
                        //   [4..7]:  half 1 — same pattern for elements 16-31
                        //   [8..11]: grouped hbits — group g's 4 hbits at bits [g*4..g*4+3] of uint32
                        constexpr uint32_t BIT_SPREAD = 0x00204081u;
                        const uint32_t hbit_word = *reinterpret_cast<const uint32_t*>(payload + 8);

                        #pragma unroll
                        for (int half = 0; half < 2; ++half)
                        {
                            const uint32_t r = *reinterpret_cast<const uint32_t*>(payload + half * 4);
                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                            {
                                // Extract 2-bit values (Q2_K-style pre-transposed)
                                const uint32_t low2_4 = (r >> (g * 2)) & 0x03030303u;
                                // Extract and scatter hbit for this group
                                const uint32_t hb4 = (hbit_word >> ((half * 4 + g) * 4)) & 0xFu;
                                const uint32_t hb = ((hb4 * BIT_SPREAD) & 0x01010101u) << 2;
                                // Merge: 3-bit value = low2 | (hbit << 2), range [0..7]
                                uint32_t val = low2_4 | hb;
                                // Center: [0..7] → [-4..+3] via XOR-SUB-XOR bias trick
                                val = (val ^ 0x80808080u) - 0x04040404u;
                                val ^= 0x80808080u;
                                packed_groups[half * 4 + g] = static_cast<int32_t>(val);
                            }
                        }
                    }
                    else if constexpr (Traits::is_2bit)
                    {
                        // Q2_K: Pre-transposed 2-bit decode — zero-transpose field extraction
                        //
                        // Payload layout (pre-transposed during CPU packing, 8B native):
                        //   [0..3]:  half 0 — byte[j] = e[j] | e[j+4]<<2 | e[j+8]<<4 | e[j+12]<<6
                        //   [4..7]:  half 1 — same pattern for elements 16-31
                        //   (emins now in separate d_block_emins array)
                        //
                        // Field extraction directly produces group-order packed INT8:
                        //   groups[bg]     = raw & 0x03030303  → {e[0], e[1], e[2], e[3]}
                        //   groups[bg + 1] = (raw>>2) & 0x03   → {e[4], e[5], e[6], e[7]}
                        //   etc.
                        //
                        // Previous ISA cost:  66 VALU (14 field extract + 16 v_perm transpose)
                        // New ISA cost:       ~50 VALU (14 field extract only, zero v_perm)
                        // Payload size:       unchanged (12 bytes)
                        #pragma unroll
                        for (int half = 0; half < 2; ++half)
                        {
                            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + half * 4);
                            const int bg = half * 4;
                            packed_groups[bg]     = static_cast<int32_t>(raw & 0x03030303u);
                            packed_groups[bg + 1] = static_cast<int32_t>((raw >> 2) & 0x03030303u);
                            packed_groups[bg + 2] = static_cast<int32_t>((raw >> 4) & 0x03030303u);
                            packed_groups[bg + 3] = static_cast<int32_t>((raw >> 6) & 0x03030303u);
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
                        // IQ grid decode: LUT lookup + sign application
                        // O5: Wide payload loads for XXS variants.
                        constexpr int n_qs = Traits::is_iq3_grid ? 8 : 4;
                        constexpr int sign_off = n_qs + (Traits::has_qh ? 1 : 0);
                        const uint8_t qh_byte = Traits::has_qh ? payload[n_qs] : 0;

                        if constexpr (Traits::is_iq3_grid && !Traits::has_qh)
                        {
                            // IQ3_XXS: wide dword loads (see O5 in iq_decode_accumulate_block)
                            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                            const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
                            #pragma unroll
                            for (int g = 0; g < 8; ++g)
                            {
                                const uint32_t word = (g < 4) ? w0 : w1;
                                const int idx = (word >> ((g & 3) * 8)) & 0xFF;
                                const uint32_t grid4 = iq3_grid_lookup<FMT>(idx);
                                const uint8_t sb = static_cast<uint8_t>(w2 >> ((g / 2) * 8));
                                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sn));
                            }
                        }
                        else if constexpr (Traits::is_iq3_grid)
                        {
                            // O6: IQ3_S wide dword loads (see iq_decode_accumulate_block)
                            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                            const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
                            const uint8_t signs3 = payload[12];
                            #pragma unroll
                            for (int g = 0; g < 8; ++g)
                            {
                                const uint32_t word = (g < 4) ? w0 : w1;
                                int idx = (word >> ((g & 3) * 8)) & 0xFF;
                                idx |= (static_cast<int>((w2 >> g) & 1) << 8);
                                const uint32_t grid4 = iq3_grid_lookup<FMT>(idx);
                                const uint8_t sb = (g < 6)
                                    ? static_cast<uint8_t>(w2 >> ((g / 2 + 1) * 8))
                                    : signs3;
                                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sn));
                            }
                        }
                        else if constexpr (Traits::is_iq2_grid && !Traits::has_qh)
                        {
                            // IQ2_XXS: wide dword loads (see O5 in iq_decode_accumulate_block)
                            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                            const uint32_t w_qs = pw[0], w_signs = pw[1];
                            #pragma unroll
                            for (int l = 0; l < 4; ++l)
                            {
                                const int idx = (w_qs >> (l * 8)) & 0xFF;
                                const uint64_t grid8 = iq2_grid_lookup<FMT>(idx);
                                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                                const uint8_t signs = static_cast<uint8_t>(w_signs >> (l * 8));
                                packed_groups[l * 2]     = static_cast<int32_t>(
                                    iq_apply_signs_4(g_lo, signs & 0xF));
                                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                                    iq_apply_signs_4(g_hi, signs >> 4));
                            }
                        }
                        else
                        {
                            // O6: IQ2_XS/IQ2_S wide dword loads (see iq_decode_accumulate_block)
                            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
                            const uint32_t w_qs = pw[0], w_qh_signs = pw[1];
                            const uint8_t signs3 = payload[8];
                            #pragma unroll
                            for (int l = 0; l < 4; ++l)
                            {
                                int idx = (w_qs >> (l * 8)) & 0xFF;
                                if constexpr (Traits::qh_bits_per_lookup == 1)
                                    idx |= (static_cast<int>((w_qh_signs >> l) & 1) << 8);
                                else if constexpr (Traits::qh_bits_per_lookup == 2)
                                    idx |= (static_cast<int>((w_qh_signs >> (2 * l)) & 3) << 8);
                                const uint64_t grid8 = iq2_grid_lookup<FMT>(idx);
                                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                                const uint8_t signs_byte = (l < 3)
                                    ? static_cast<uint8_t>(w_qh_signs >> ((l + 1) * 8))
                                    : signs3;
                                packed_groups[l * 2]     = static_cast<int32_t>(
                                    iq_apply_signs_4(g_lo, signs_byte & 0xF));
                                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                                    iq_apply_signs_4(g_hi, signs_byte >> 4));
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
                    else if constexpr (Traits::is_8bit)
                    {
                        // Q8_0: 32 raw INT8 values — direct load, no decode needed
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                            packed_groups[g] = *reinterpret_cast<const int32_t*>(payload + g * 4);
                    }
                    else
                    {
                        // Q4_0 / Q4_1: 4-bit nibble decode via SWAR / v_perm_b32
                        if constexpr (Traits::center_4bit == 8)
                        {
                            // Q4_0: w = nibble - 8 → range [-8,+7]
                            // SWAR byte-wise subtraction: (nibble + 0x78) ^ 0x80
                            //   nibble [0,15] + 0x78 = [0x78, 0x87] — no inter-byte carry
                            //   XOR 0x80 → signed INT8: 0x78→0xF8(-8) ... 0x87→0x07(+7)
                            constexpr uint32_t SWAR_ADD = 0x78787878u;
                            constexpr uint32_t SWAR_XOR = 0x80808080u;
                            constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;

                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                            {
                                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                                packed_groups[g]     = static_cast<int32_t>(((raw & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
                                packed_groups[g + 4] = static_cast<int32_t>((((raw >> 4) & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
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
                        f_acc[c] += (static_cast<float>(acc_lo) * scale_lo
                                  + static_cast<float>(acc_hi) * scale_hi) * scale_ab;

                        // Q2_K/IQ1_M: dual-scale + emin/delta correction
                        // Q2_K: emins from separate d_block_emins array
                        // IQ1_M: has ±IQ1S_DELTA per-group correction (handled below)
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

                            float min_lo, min_hi;
                            if constexpr (Traits::needs_emin_array) {
                                const uint32_t emin_packed = d_block_emins[linear];
                                const uint16_t emin_lo_bits = static_cast<uint16_t>(emin_packed);
                                const uint16_t emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
                                min_lo = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
                                min_hi = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
                            } else {
                                min_lo = 0.0f;
                                min_hi = 0.0f;
                            }
                            f_acc[c] += (static_cast<float>(sum_a_lo) * min_lo
                                      + static_cast<float>(sum_a_hi) * min_hi) * scale_ab;
                        }

                        // IQ1_M: ±IQ1S_DELTA per 8-element group correction.
                        // True weight[j] = dl * (grid[j] + delta), but the sdot4 above
                        // only accumulates dl * dot(a, grid).  We add the missing
                        // dl * delta * sum_a_per_group for each of the 4 groups.
                        if constexpr (FMT == NVNNI_IQ1_M)
                        {
                            constexpr float IQ1S_DELTA = 0.125f;
                            constexpr int32_t ones = 0x01010101;
                            const uint8_t qh0 = payload[4], qh1 = payload[5];

                            int32_t sa_g0 = __builtin_amdgcn_sdot4(a4[1], ones,
                                            __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
                            int32_t sa_g1 = __builtin_amdgcn_sdot4(a4[3], ones,
                                            __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
                            int32_t sa_g2 = __builtin_amdgcn_sdot4(a4[5], ones,
                                            __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
                            int32_t sa_g3 = __builtin_amdgcn_sdot4(a4[7], ones,
                                            __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

                            const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

                            f_acc[c] += ((d0 * static_cast<float>(sa_g0) + d1 * static_cast<float>(sa_g1)) * scale_lo
                                      + (d2 * static_cast<float>(sa_g2) + d3 * static_cast<float>(sa_g3)) * scale_hi) * scale_ab;
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
                        f_acc[c] += static_cast<float>(block_acc) * block_d * scale_ab;

                        // Asymmetric min correction: f_acc += sum_a * block_m
                        // Derives from: value = d*q + m  →  Σ a_i*value_i = d*Σ(a_i*q_i) + m*Σ(a_i)
                        if constexpr (Traits::is_asymmetric)
                        {
                            const uint16_t min_bits = d_block_mins[linear];
                            const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                            f_acc[c] += static_cast<float>(sum_a) * block_m * scale_ab;
                        }
                    }
                }
            }
        }
        } // close else (non-IQ formats)

        // Apply activation scale and write output
        // When blockwise scales are active, per-block scales were already fused
        // into the accumulation loop above, so scale_a = 1.0.
        const float scale_a = d_scale_A_blockwise ? 1.0f : d_scale_A[0];
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
                else if (d_partial_fp32 == nullptr)
                {
                    // Atomic reduce: accumulate directly to output (pre-zeroed)
                    atomicAdd(&d_C_fp32[n], result);
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

struct DeviceNativeVNNIMatrixDesc
{
    const uint8_t* payload;
    const void* scales;
    const void* mins;
    const void* emins;
    int n;
    int k;
    uint32_t blocks_per_row;
    uint8_t codebook_id;
    uint8_t reserved[3];
};

__global__ void moe_grouped_swiglu_quantize_blockwise_kernel(
    const float* const* __restrict__ gate_ptrs,
    const float* const* __restrict__ up_ptrs,
    int8_t* __restrict__ A_int8,
    float* __restrict__ scales_A_blockwise,
    int num_active,
    int K)
{
    const int slot = blockIdx.x;
    if (slot >= num_active) return;

    constexpr int BLOCK_SIZE = 32;
    constexpr int NUM_WARPS = 8;
    const int blocks_per_row = K / BLOCK_SIZE;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const float* gate = gate_ptrs[slot];
    const float* up = up_ptrs[slot];
    int8_t* row_int8 = A_int8 + static_cast<size_t>(slot) * K;
    float* row_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

    for (int b = warp_id; b < blocks_per_row; b += NUM_WARPS)
    {
        const int k_start = b * BLOCK_SIZE;
        const float g = gate[k_start + lane];
        const float u = up[k_start + lane];
        const float val = (g / (1.0f + expf(-g))) * u;

        float abs_val = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

        const float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        const float inv_scale = 1.0f / scale;

        if (lane == 0)
            row_scales[b] = scale;

        const float qval = val * inv_scale;
        row_int8[k_start + lane] = static_cast<int8_t>(
            rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
    }
}

template <NativeVNNIFormat FMT>
__device__ __forceinline__ void moe_decode_native_vnni_groups(
    const uint8_t* __restrict__ payload,
    int32_t* packed_groups)
{
    using Traits = NVNNITraits<FMT>;
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE decode requires a supported native-VNNI format");

    if constexpr (Traits::uses_lut)
    {
        constexpr uint32_t CB_LO_A = 0xBFAD9881u;
        constexpr uint32_t CB_LO_B = 0xF6EADDCFu;
        constexpr uint32_t CB_HI_A = 0x26190D01u;
        constexpr uint32_t CB_HI_B = 0x71594535u;
        constexpr uint32_t MUX_IDENTITY = 0x03020100u;
        constexpr uint32_t MUX_BIT3_MASK = 0x04040404u;

        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t lo_sel = lo_nibs & 0x07070707u;
            const uint32_t lo_r0 = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, lo_sel);
            const uint32_t lo_r1 = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, lo_sel);
            const uint32_t lo_mux = MUX_IDENTITY | ((lo_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g] = static_cast<int32_t>(__builtin_amdgcn_perm(lo_r1, lo_r0, lo_mux));

            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
            const uint32_t hi_sel = hi_nibs & 0x07070707u;
            const uint32_t hi_r0 = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, hi_sel);
            const uint32_t hi_r1 = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, hi_sel);
            const uint32_t hi_mux = MUX_IDENTITY | ((hi_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g + 4] = static_cast<int32_t>(__builtin_amdgcn_perm(hi_r1, hi_r0, hi_mux));
        }
    }
    else if constexpr (Traits::is_5bit)
    {
        const uint32_t qh_bits = *reinterpret_cast<const uint32_t*>(payload + 16);

        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
            const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                 | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
            uint32_t val_lo = lo_nibs | hb_lo;

            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
            const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
            const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                 | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
            uint32_t val_hi = hi_nibs | hb_hi;

            if constexpr (Traits::center_5bit == 16)
            {
                val_lo ^= 0x80808080u;
                val_lo -= 0x10101010u;
                val_lo ^= 0x80808080u;
                val_hi ^= 0x80808080u;
                val_hi -= 0x10101010u;
                val_hi ^= 0x80808080u;
            }

            packed_groups[g] = static_cast<int32_t>(val_lo);
            packed_groups[g + 4] = static_cast<int32_t>(val_hi);
        }
    }
    else if constexpr (Traits::is_3bit)
    {
        constexpr uint32_t BIT_SPREAD = 0x00204081u;
        const uint32_t hbit_word = *reinterpret_cast<const uint32_t*>(payload + 8);

        #pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + half * 4);
            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t low2_4 = (raw >> (g * 2)) & 0x03030303u;
                const uint32_t hb4 = (hbit_word >> ((half * 4 + g) * 4)) & 0xFu;
                const uint32_t hb = ((hb4 * BIT_SPREAD) & 0x01010101u) << 2;
                uint32_t val = (low2_4 | hb) ^ 0x80808080u;
                val -= 0x04040404u;
                val ^= 0x80808080u;
                packed_groups[half * 4 + g] = static_cast<int32_t>(val);
            }
        }
    }
    else if constexpr (Traits::is_2bit)
    {
        #pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + half * 4);
            const int bg = half * 4;
            packed_groups[bg] = static_cast<int32_t>(raw & 0x03030303u);
            packed_groups[bg + 1] = static_cast<int32_t>((raw >> 2) & 0x03030303u);
            packed_groups[bg + 2] = static_cast<int32_t>((raw >> 4) & 0x03030303u);
            packed_groups[bg + 3] = static_cast<int32_t>((raw >> 6) & 0x03030303u);
        }
    }
    else if constexpr (Traits::is_6bit)
    {
        const uint8_t* qh_p = payload + 16;

        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            const uint32_t h_lo = qh_p[g];
            const uint32_t h_hi = qh_p[g + 4];

            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t hb_lo = ((h_lo & 0x03u) << 4)  | ((h_lo & 0x0Cu) << 10)
                                 | ((h_lo & 0x30u) << 16) | ((h_lo & 0xC0u) << 22);
            uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
            val_lo -= 0x20202020u;
            val_lo ^= 0x80808080u;
            packed_groups[g] = static_cast<int32_t>(val_lo);

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
        if constexpr (Traits::is_iq3_grid && !Traits::has_qh)
        {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t qs0 = words[0];
            const uint32_t qs1 = words[1];
            const uint32_t signs = words[2];

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                const uint32_t word = (g < 4) ? qs0 : qs1;
                const int idx = (word >> ((g & 3) * 8)) & 0xFF;
                const uint32_t grid4 = iq3_grid_lookup<FMT>(idx);
                const uint8_t sign_byte = static_cast<uint8_t>(signs >> ((g / 2) * 8));
                const uint8_t sign_nibble = (g & 1) ? (sign_byte >> 4) : (sign_byte & 0xF);
                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sign_nibble));
            }
        }
        else if constexpr (Traits::is_iq3_grid)
        {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t qs0 = words[0];
            const uint32_t qs1 = words[1];
            const uint32_t qh_signs = words[2];
            const uint8_t signs3 = payload[12];

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                const uint32_t word = (g < 4) ? qs0 : qs1;
                int idx = (word >> ((g & 3) * 8)) & 0xFF;
                idx |= (static_cast<int>((qh_signs >> g) & 1) << 8);
                const uint32_t grid4 = iq3_grid_lookup<FMT>(idx);
                const uint8_t sign_byte = (g < 6)
                    ? static_cast<uint8_t>(qh_signs >> ((g / 2 + 1) * 8))
                    : signs3;
                const uint8_t sign_nibble = (g & 1) ? (sign_byte >> 4) : (sign_byte & 0xF);
                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, sign_nibble));
            }
        }
        else if constexpr (Traits::is_iq2_grid && !Traits::has_qh)
        {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t qs = words[0];
            const uint32_t signs = words[1];

            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                const int idx = (qs >> (g * 8)) & 0xFF;
                const uint64_t grid8 = iq2_grid_lookup<FMT>(idx);
                const uint32_t grid_lo = static_cast<uint32_t>(grid8);
                const uint32_t grid_hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t sign_byte = static_cast<uint8_t>(signs >> (g * 8));
                packed_groups[g * 2] = static_cast<int32_t>(iq_apply_signs_4(grid_lo, sign_byte & 0xF));
                packed_groups[g * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4(grid_hi, sign_byte >> 4));
            }
        }
        else
        {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t qs = words[0];
            const uint32_t qh_signs = words[1];
            const uint8_t signs3 = payload[8];

            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                int idx = (qs >> (g * 8)) & 0xFF;
                if constexpr (Traits::qh_bits_per_lookup == 1)
                    idx |= (static_cast<int>((qh_signs >> g) & 1) << 8);
                else if constexpr (Traits::qh_bits_per_lookup == 2)
                    idx |= (static_cast<int>((qh_signs >> (2 * g)) & 3) << 8);
                const uint64_t grid8 = iq2_grid_lookup<FMT>(idx);
                const uint32_t grid_lo = static_cast<uint32_t>(grid8);
                const uint32_t grid_hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t sign_byte = (g < 3)
                    ? static_cast<uint8_t>(qh_signs >> ((g + 1) * 8))
                    : signs3;
                packed_groups[g * 2] = static_cast<int32_t>(iq_apply_signs_4(grid_lo, sign_byte & 0xF));
                packed_groups[g * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4(grid_hi, sign_byte >> 4));
            }
        }
    }
    else if constexpr (Traits::is_iq1_grid)
    {
        if constexpr (FMT == NVNNI_IQ1_S)
        {
            const uint16_t qh_word = static_cast<uint16_t>(payload[4]) |
                                     (static_cast<uint16_t>(payload[5]) << 8);
            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                const int idx = payload[g] | (((qh_word >> (3 * g)) & 7) << 8);
                const uint64_t grid8 = iq1_grid_lookup(idx);
                packed_groups[g * 2] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
                packed_groups[g * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            }
        }
        else
        {
            const uint8_t qh0 = payload[4];
            const uint8_t qh1 = payload[5];
            const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
            const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
            const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
            const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

            uint64_t grid8 = iq1_grid_lookup(idx0);
            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            grid8 = iq1_grid_lookup(idx1);
            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            grid8 = iq1_grid_lookup(idx2);
            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            grid8 = iq1_grid_lookup(idx3);
            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
        }
    }
    else if constexpr (Traits::is_8bit)
    {
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            packed_groups[g] = *reinterpret_cast<const int32_t*>(payload + g * 4);
    }
    else
    {
        constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;

        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            if constexpr (Traits::center_4bit == 8)
            {
                constexpr uint32_t SWAR_ADD = 0x78787878u;
                constexpr uint32_t SWAR_XOR = 0x80808080u;
                packed_groups[g] = static_cast<int32_t>(((raw & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
                packed_groups[g + 4] = static_cast<int32_t>((((raw >> 4) & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
            }
            else
            {
                packed_groups[g] = static_cast<int32_t>(raw & NIBBLE_MASK);
                packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & NIBBLE_MASK);
            }
        }
    }
}

template <NativeVNNIFormat FMT>
__device__ __forceinline__ float moe_native_vnni_dual_scale_correction(
    const int32_t* __restrict__ a4,
    const uint8_t* __restrict__ payload,
    const uint32_t* __restrict__ emin_base,
    size_t linear,
    float scale_lo,
    float scale_hi,
    float scale_ab)
{
    using Traits = NVNNITraits<FMT>;
    float correction = 0.0f;

    if constexpr (Traits::is_dual_scale_asym)
    {
        constexpr int32_t ones = 0x01010101;
        int32_t sum_a_lo = 0;
        int32_t sum_a_hi = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            sum_a_lo = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_lo, false);
        #pragma unroll
        for (int g = 4; g < 8; ++g)
            sum_a_hi = __builtin_amdgcn_sdot4(a4[g], ones, sum_a_hi, false);

        const uint32_t emin_packed = emin_base[linear];
        const uint16_t emin_lo_bits = static_cast<uint16_t>(emin_packed);
        const uint16_t emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
        const float min_lo = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
        const float min_hi = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
        correction += (static_cast<float>(sum_a_lo) * min_lo +
                       static_cast<float>(sum_a_hi) * min_hi) * scale_ab;
    }

    if constexpr (FMT == NVNNI_IQ1_M)
    {
        constexpr float IQ1S_DELTA = 0.125f;
        constexpr int32_t ones = 0x01010101;
        const uint8_t qh0 = payload[4];
        const uint8_t qh1 = payload[5];

        const int32_t sum_a_g0 = __builtin_amdgcn_sdot4(a4[1], ones,
                                __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
        const int32_t sum_a_g1 = __builtin_amdgcn_sdot4(a4[3], ones,
                                __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
        const int32_t sum_a_g2 = __builtin_amdgcn_sdot4(a4[5], ones,
                                __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
        const int32_t sum_a_g3 = __builtin_amdgcn_sdot4(a4[7], ones,
                                __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

        const float delta0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
        const float delta1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
        const float delta2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
        const float delta3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

        correction += ((delta0 * static_cast<float>(sum_a_g0) + delta1 * static_cast<float>(sum_a_g1)) * scale_lo +
                       (delta2 * static_cast<float>(sum_a_g2) + delta3 * static_cast<float>(sum_a_g3)) * scale_hi) * scale_ab;
    }

    return correction;
}

template <NativeVNNIFormat FMT, bool INDEX_BY_EXPERT_ID>
__global__ void moe_grouped_native_vnni_down_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    int num_active,
    int N,
    int K)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE down requires a supported native-VNNI descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    if (n >= N) return;

    const int blocks_per_row = K / 32;
    float total = 0.0f;

    #pragma unroll 1
    for (int slot = 0; slot < num_active; ++slot)
    {
        int desc_idx = slot;
        if constexpr (INDEX_BY_EXPERT_ID)
            desc_idx = expert_ids[slot];
        const DeviceNativeVNNIMatrixDesc desc = descs[desc_idx];
        const uint8_t* payload_base = desc.payload;
        const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
        const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
        const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);
        const int8_t* a_slot = A_int8 + static_cast<size_t>(slot) * K;
        const float* scale_slot = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

        float expert_acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(a_slot + b * 32);
            int32_t packed_groups[8];
            const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
            const uint8_t* payload = payload_base + linear * NVNNITraits<FMT>::payload_bytes;
            moe_decode_native_vnni_groups<FMT>(payload, packed_groups);

            const float scale_ab = scale_slot[b];
            if constexpr (NVNNITraits<FMT>::is_dual_scale)
            {
                int32_t acc_lo = 0;
                int32_t acc_hi = 0;
                #pragma unroll
                for (int g = 0; g < 4; ++g)
                    acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
                #pragma unroll
                for (int g = 4; g < 8; ++g)
                    acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

                const uint16_t scale_lo_bits = scale_base[linear];
                const uint16_t scale_hi_bits = min_base[linear];
                const float scale_lo = __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
                const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
                expert_acc += (static_cast<float>(acc_lo) * scale_lo +
                               static_cast<float>(acc_hi) * scale_hi) * scale_ab;
                expert_acc += moe_native_vnni_dual_scale_correction<FMT>(
                    a4, payload, emin_base, linear, scale_lo, scale_hi, scale_ab);
            }
            else
            {
                int32_t block_acc = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

                const uint16_t scale_bits = scale_base[linear];
                const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                expert_acc += static_cast<float>(block_acc) * block_d * scale_ab;

                if constexpr (NVNNITraits<FMT>::is_asymmetric)
                {
                    constexpr int32_t ones = 0x01010101;
                    int32_t sum_a = 0;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                    const uint16_t min_bits = min_base[linear];
                    const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                    expert_acc += static_cast<float>(sum_a) * block_m * scale_ab;
                }
            }
        }

        total += route_weights[slot] * expert_acc;
    }

    output[n] = total;
}

template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_down_parallel_experts_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    int num_active,
    int N,
    int K)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE parallel down requires a supported native-VNNI descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    const int expert_id = expert_ids[slot];
    if (expert_id < 0) return;

    const int blocks_per_row = K / 32;
    const DeviceNativeVNNIMatrixDesc desc = descs[expert_id];
    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);
    const int8_t* a_slot = A_int8 + static_cast<size_t>(slot) * K;
    const float* scale_slot = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

    float expert_acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(a_slot + b * 32);
        int32_t packed_groups[8];
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
        const uint8_t* payload = payload_base + linear * NVNNITraits<FMT>::payload_bytes;
        moe_decode_native_vnni_groups<FMT>(payload, packed_groups);

        const float scale_ab = scale_slot[b];
        if constexpr (NVNNITraits<FMT>::is_dual_scale)
        {
            int32_t acc_lo = 0;
            int32_t acc_hi = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g)
                acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
            #pragma unroll
            for (int g = 4; g < 8; ++g)
                acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

            const uint16_t scale_lo_bits = scale_base[linear];
            const uint16_t scale_hi_bits = min_base[linear];
            const float scale_lo = __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
            const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
            expert_acc += (static_cast<float>(acc_lo) * scale_lo +
                           static_cast<float>(acc_hi) * scale_hi) * scale_ab;
            expert_acc += moe_native_vnni_dual_scale_correction<FMT>(
                a4, payload, emin_base, linear, scale_lo, scale_hi, scale_ab);
        }
        else
        {
            int32_t block_acc = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

            const uint16_t scale_bits = scale_base[linear];
            const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            expert_acc += static_cast<float>(block_acc) * block_d * scale_ab;

            if constexpr (NVNNITraits<FMT>::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;
                int32_t sum_a = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                const uint16_t min_bits = min_base[linear];
                const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                expert_acc += static_cast<float>(sum_a) * block_m * scale_ab;
            }
        }
    }

    atomicAdd(output + n, route_weights[slot] * expert_acc);
}

__global__ void moe_grouped_hidden_quantize_blockwise_kernel(
    const float* __restrict__ hidden,
    int8_t* __restrict__ A_int8,
    float* __restrict__ scales_A_blockwise,
    int K)
{
    constexpr int BLOCK_SIZE = 32;
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = block_idx * BLOCK_SIZE + lane;
    if (lane >= BLOCK_SIZE || k >= K) return;

    const float val = hidden[k];
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

    const float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (lane == 0)
        scales_A_blockwise[block_idx] = scale;

    const float qval = val * inv_scale;
    A_int8[k] = static_cast<int8_t>(
        rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
}

template <NativeVNNIFormat FMT>
__device__ __forceinline__ float moe_native_vnni_dot_desc_range(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K,
    int b_start,
    int b_end)
{
    const int blocks_per_row = K / 32;
    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);
    float acc = 0.0f;

    if (b_start < 0) b_start = 0;
    if (b_end > blocks_per_row) b_end = blocks_per_row;

    for (int b = b_start; b < b_end; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(A_int8 + b * 32);
        int32_t packed_groups[8];
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
        const uint8_t* payload = payload_base + linear * NVNNITraits<FMT>::payload_bytes;
        moe_decode_native_vnni_groups<FMT>(payload, packed_groups);

        const float scale_ab = scales_A_blockwise[b];
        if constexpr (NVNNITraits<FMT>::is_dual_scale)
        {
            int32_t acc_lo = 0;
            int32_t acc_hi = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g)
                acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
            #pragma unroll
            for (int g = 4; g < 8; ++g)
                acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

            const uint16_t scale_lo_bits = scale_base[linear];
            const uint16_t scale_hi_bits = min_base[linear];
            const float scale_lo = __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
            const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
            acc += (static_cast<float>(acc_lo) * scale_lo +
                    static_cast<float>(acc_hi) * scale_hi) * scale_ab;
            acc += moe_native_vnni_dual_scale_correction<FMT>(
                a4, payload, emin_base, linear, scale_lo, scale_hi, scale_ab);
        }
        else
        {
            int32_t block_acc = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

            const uint16_t scale_bits = scale_base[linear];
            const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            acc += static_cast<float>(block_acc) * block_d * scale_ab;

            if constexpr (NVNNITraits<FMT>::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;
                int32_t sum_a = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

                const uint16_t min_bits = min_base[linear];
                const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                acc += static_cast<float>(sum_a) * block_m * scale_ab;
            }
        }
    }

    return acc;
}

template <NativeVNNIFormat FMT>
__device__ __forceinline__ float moe_native_vnni_dot_desc(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K)
{
    return moe_native_vnni_dot_desc_range<FMT>(
        desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
}

template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_gate_up_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int K)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE gate/up requires a supported native-VNNI descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    const int expert_id = expert_ids[slot];
    const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];

    gate_outputs[slot][n] = moe_native_vnni_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_vnni_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_gate_up_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeVNNIMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int k_partitions)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE K-partition gate/up requires a supported native-VNNI descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int k_part = blockIdx.y;
    const int slot = blockIdx.z;
    if (slot >= num_active || k_part >= k_partitions || n >= N) return;

    const int expert_id = expert_ids[slot];
    const size_t partial_index =
        (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) + static_cast<size_t>(k_part)) *
            static_cast<size_t>(N) +
        static_cast<size_t>(n);

    if (expert_id < 0)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row) b_end = blocks_per_row;

    if (b_start >= b_end)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];

    gate_partials[partial_index] = moe_native_vnni_dot_desc_range<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
    up_partials[partial_index] = moe_native_vnni_dot_desc_range<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
}

__global__ void moe_grouped_native_vnni_gate_up_kpart_reduce_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int k_partitions)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const size_t slot_base = static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
    for (int k_part = 0; k_part < k_partitions; ++k_part)
    {
        const size_t idx = slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
        gate_sum += gate_partials[idx];
        up_sum += up_partials[idx];
    }

    gate_outputs[slot][n] = gate_sum;
    up_outputs[slot][n] = up_sum;
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

template <int TILE_N, int CPT>
__global__ void gemv_native_vnni_reduce_m2_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB × 2 × N] partial buffer
    float* __restrict__ d_C_fp32,                // [2 × N] output
    int N, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N) return;

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < kblocks; ++kb)
        {
            sum0 += d_partial_fp32[(static_cast<size_t>(kb) * 2u) * N + n];
            sum1 += d_partial_fp32[(static_cast<size_t>(kb) * 2u + 1u) * N + n];
        }

        d_C_fp32[n] = sum0;
        d_C_fp32[static_cast<size_t>(N) + n] = sum1;
    }
}

template <int TILE_N, int CPT, int SMALL_M>
__global__ void gemv_native_vnni_reduce_small_m_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB × SMALL_M × N] partial buffer
    float* __restrict__ d_C_fp32,                // [SMALL_M × N] output
    int N, int kblocks)
{
    static_assert(SMALL_M >= 2 && SMALL_M <= 4, "small-M native-VNNI supports M in {2,3,4}");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        #pragma unroll
        for (int row = 0; row < SMALL_M; ++row)
        {
            float sum = 0.0f;
            #pragma unroll 16
            for (int kb = 0; kb < kblocks; ++kb)
            {
                const size_t idx =
                    (static_cast<size_t>(kb) * SMALL_M + static_cast<size_t>(row)) *
                    static_cast<size_t>(N) + static_cast<size_t>(n);
                sum += d_partial_fp32[idx];
            }
            d_C_fp32[static_cast<size_t>(row) * N + n] = sum;
        }
    }
}

constexpr int NVNNI_SMALL_M_MAX_BATCH = 8;

struct NativeVNNISmallMBatchedArgs
{
    int num_projections = 0;
    int total_grid_n = 0;
    int N[NVNNI_SMALL_M_MAX_BATCH] = {};
    int grid_offsets[NVNNI_SMALL_M_MAX_BATCH + 1] = {};
    const uint8_t* payload[NVNNI_SMALL_M_MAX_BATCH] = {};
    const uint16_t* scales[NVNNI_SMALL_M_MAX_BATCH] = {};
    const uint16_t* mins[NVNNI_SMALL_M_MAX_BATCH] = {};
    const uint32_t* emins[NVNNI_SMALL_M_MAX_BATCH] = {};
    const float* bias[NVNNI_SMALL_M_MAX_BATCH] = {};
    float* output[NVNNI_SMALL_M_MAX_BATCH] = {};
    float* partial[NVNNI_SMALL_M_MAX_BATCH] = {};
};

__device__ __forceinline__ int native_vnni_batched_projection_for_tile(
    const NativeVNNISmallMBatchedArgs& args,
    int global_tile)
{
    int proj = 0;
    #pragma unroll
    for (int i = 1; i < NVNNI_SMALL_M_MAX_BATCH; ++i)
    {
        if (i < args.num_projections && global_tile >= args.grid_offsets[i])
            proj = i;
    }
    return proj;
}

template <NativeVNNIFormat FMT>
__device__ __forceinline__ float native_vnni_small_m_dot_decoded_block(
    const int32_t* __restrict__ a4,
    const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,
    const uint32_t* __restrict__ emin_base,
    size_t linear,
    float scale_ab)
{
    using Traits = NVNNITraits<FMT>;

    if constexpr (Traits::is_dual_scale)
    {
        int32_t acc_lo = 0;
        int32_t acc_hi = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            acc_lo = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_lo, false);
        #pragma unroll
        for (int g = 4; g < 8; ++g)
            acc_hi = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], acc_hi, false);

        const uint16_t scale_lo_bits = scale_base[linear];
        const uint16_t scale_hi_bits = min_base[linear];
        const float scale_lo = __half2float(*reinterpret_cast<const __half*>(&scale_lo_bits));
        const float scale_hi = __half2float(*reinterpret_cast<const __half*>(&scale_hi_bits));
        float acc = (static_cast<float>(acc_lo) * scale_lo +
                     static_cast<float>(acc_hi) * scale_hi) * scale_ab;
        acc += moe_native_vnni_dual_scale_correction<FMT>(
            a4, payload, emin_base, linear, scale_lo, scale_hi, scale_ab);
        return acc;
    }
    else
    {
        int32_t block_acc = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

        const uint16_t scale_bits = scale_base[linear];
        const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
        float acc = static_cast<float>(block_acc) * block_d * scale_ab;

        if constexpr (Traits::is_asymmetric)
        {
            constexpr int32_t ones = 0x01010101;
            int32_t sum_a = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);

            const uint16_t min_bits = min_base[linear];
            const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
            acc += static_cast<float>(sum_a) * block_m * scale_ab;
        }
        return acc;
    }
}

template <int TILE_N, int CPT, int SMALL_M, NativeVNNIFormat FMT>
__global__ void gemv_native_vnni_small_m_batched_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [SMALL_M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [SMALL_M x blocks_per_row]
    NativeVNNISmallMBatchedArgs args,
    int K,
    int kblocks)
{
    static_assert(SMALL_M >= 2 && SMALL_M <= 4, "small-M native-VNNI supports M in {2,3,4}");
    static_assert(CPT == 1 || CPT == 2, "native-VNNI small-M supports CPT in {1,2}");
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "small-M native-VNNI requires a supported native-VNNI format");

    const int global_tile = blockIdx.x;
    const int proj = native_vnni_batched_projection_for_tile(args, global_tile);
    const int local_tile = global_tile - args.grid_offsets[proj];
    const int N = args.N[proj];
    const int n_base = local_tile * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    const uint8_t* payload_base = args.payload[proj];
    const uint16_t* scale_base = args.scales[proj];
    const uint16_t* min_base = args.mins[proj];
    const uint32_t* emin_base = args.emins[proj];
    const float* bias_base = args.bias[proj];
    float* out_base = args.output[proj];
    float* partial_base = args.partial[proj];

    float acc[SMALL_M][CPT];
    #pragma unroll
    for (int row = 0; row < SMALL_M; ++row)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[row][c] = 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        for (int b = b_start; b < b_end; ++b)
        {
            int32_t packed_groups[8];
            const size_t linear =
                static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
            const uint8_t* payload = payload_base + linear * NVNNITraits<FMT>::payload_bytes;
            moe_decode_native_vnni_groups<FMT>(payload, packed_groups);

            #pragma unroll
            for (int row = 0; row < SMALL_M; ++row)
            {
                const int32_t* a4 = reinterpret_cast<const int32_t*>(
                    d_A_int8 + static_cast<size_t>(row) * K + static_cast<size_t>(b) * 32);
                const float scale_ab =
                    d_scale_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b];
                acc[row][c] += native_vnni_small_m_dot_decoded_block<FMT>(
                    a4, packed_groups, payload, scale_base, min_base,
                    emin_base, linear, scale_ab);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        #pragma unroll
        for (int row = 0; row < SMALL_M; ++row)
        {
            if (kblocks == 1)
            {
                out_base[static_cast<size_t>(row) * N + n] =
                    acc[row][c] + (bias_base ? bias_base[n] : 0.0f);
            }
            else
            {
                partial_base[(static_cast<size_t>(kb) * SMALL_M + static_cast<size_t>(row)) *
                             static_cast<size_t>(N) + static_cast<size_t>(n)] = acc[row][c];
            }
        }
    }
}

template <int TILE_N, int CPT, int SMALL_M>
__global__ void gemv_native_vnni_reduce_small_m_batched_kernel_t(
    NativeVNNISmallMBatchedArgs args,
    int kblocks)
{
    static_assert(SMALL_M >= 2 && SMALL_M <= 4, "small-M native-VNNI supports M in {2,3,4}");

    const int global_tile = blockIdx.x;
    const int proj = native_vnni_batched_projection_for_tile(args, global_tile);
    const int local_tile = global_tile - args.grid_offsets[proj];
    const int N = args.N[proj];
    const int n_base = local_tile * TILE_N + threadIdx.x * CPT;

    const float* partial_base = args.partial[proj];
    const float* bias_base = args.bias[proj];
    float* out_base = args.output[proj];

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        #pragma unroll
        for (int row = 0; row < SMALL_M; ++row)
        {
            float sum = 0.0f;
            #pragma unroll 16
            for (int kb = 0; kb < kblocks; ++kb)
            {
                sum += partial_base[(static_cast<size_t>(kb) * SMALL_M + static_cast<size_t>(row)) *
                                    static_cast<size_t>(N) + static_cast<size_t>(n)];
            }
            out_base[static_cast<size_t>(row) * N + n] =
                sum + (bias_base ? bias_base[n] : 0.0f);
        }
    }
}

template <int TILE_N, int CPT, int SMALL_M, NativeVNNIFormat FMT>
__global__ void gemv_native_vnni_small_m_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [SMALL_M × K] quantized activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × payload_bytes]
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N]
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N], nullable when unused
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row × N], Q2_K only
    float*           __restrict__ d_C_fp32,       // [SMALL_M × N] output
    float*           __restrict__ d_partial_fp32, // [KB × SMALL_M × N] partials
    const float*     __restrict__ d_scale_A_blockwise, // [SMALL_M × blocks_per_row]
    int N, int K, int kblocks)
{
    static_assert(SMALL_M >= 2 && SMALL_M <= 4, "small-M native-VNNI supports M in {2,3,4}");
    static_assert(CPT == 1 || CPT == 2, "native-VNNI small-M supports CPT in {1,2}");
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "small-M native-VNNI requires a supported native-VNNI format");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float acc[SMALL_M][CPT];
    #pragma unroll
    for (int row = 0; row < SMALL_M; ++row)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[row][c] = 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        for (int b = b_start; b < b_end; ++b)
        {
            int32_t packed_groups[8];
            const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
            const uint8_t* payload = d_payload + linear * NVNNITraits<FMT>::payload_bytes;
            moe_decode_native_vnni_groups<FMT>(payload, packed_groups);

            #pragma unroll
            for (int row = 0; row < SMALL_M; ++row)
            {
                const int32_t* a4 = reinterpret_cast<const int32_t*>(
                    d_A_int8 + static_cast<size_t>(row) * K + static_cast<size_t>(b) * 32);
                const float scale_ab =
                    d_scale_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b];
                acc[row][c] += native_vnni_small_m_dot_decoded_block<FMT>(
                    a4, packed_groups, payload, d_block_scales, d_block_mins,
                    d_block_emins, linear, scale_ab);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        #pragma unroll
        for (int row = 0; row < SMALL_M; ++row)
        {
            if (kblocks == 1)
            {
                d_C_fp32[static_cast<size_t>(row) * N + n] = acc[row][c];
            }
            else
            {
                const size_t idx =
                    (static_cast<size_t>(kb) * SMALL_M + static_cast<size_t>(row)) *
                    static_cast<size_t>(N) + static_cast<size_t>(n);
                d_partial_fp32[idx] = acc[row][c];
            }
        }
    }
}

// Q4_0/Q4_1 two-row verifier kernel. Qwen3.6 Q4_K uses the Q4_1 native
// codebook after CPU packing, so this provides the graph-safe ROCm MTP path
// without touching the broader IQ/native-VNNI matrix yet.
template <int TILE_N, int CPT, NativeVNNIFormat FMT>
__global__ void gemv_native_vnni_q4_m2_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [2 × K] quantized activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × 16]
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N]
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N], Q4_1 only
    float*           __restrict__ d_C_fp32,       // [2 × N] output
    float*           __restrict__ d_partial_fp32, // [KB × 2 × N] partials
    const float*     __restrict__ d_scale_A_blockwise, // [2 × blocks_per_row]
    int N, int K, int kblocks)
{
    using Traits = NVNNITraits<FMT>;
    static_assert(FMT == NVNNI_Q4_0 || FMT == NVNNI_Q4_1, "M2 native-VNNI verifier only supports Q4_0/Q4_1");
    static_assert(CPT == 1 || CPT == 2, "native-VNNI M2 supports CPT in {1,2}");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float acc0[CPT];
    float acc1[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc0[c] = 0.0f;
        acc1[c] = 0.0f;
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        for (int b = b_start; b < b_end; ++b)
        {
            const int32_t* a0 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);
            const int32_t* a1 = reinterpret_cast<const int32_t*>(d_A_int8 + K + b * 32);

            int32_t packed_groups[8];
            const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
            const uint8_t* payload = d_payload + linear * Traits::payload_bytes;

            if constexpr (Traits::center_4bit == 8)
            {
                constexpr uint32_t SWAR_ADD = 0x78787878u;
                constexpr uint32_t SWAR_XOR = 0x80808080u;
                constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;

                #pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                    packed_groups[g] = static_cast<int32_t>(((raw & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
                    packed_groups[g + 4] = static_cast<int32_t>((((raw >> 4) & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
                }
            }
            else
            {
                #pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                    packed_groups[g] = static_cast<int32_t>(raw & 0x0F0F0F0Fu);
                    packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & 0x0F0F0F0Fu);
                }
            }

            int32_t dot0 = 0;
            int32_t dot1 = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                dot0 = __builtin_amdgcn_sdot4(a0[g], packed_groups[g], dot0, false);
                dot1 = __builtin_amdgcn_sdot4(a1[g], packed_groups[g], dot1, false);
            }

            const uint16_t scale_bits = d_block_scales[linear];
            const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float scale_a0 = d_scale_A_blockwise[b];
            const float scale_a1 = d_scale_A_blockwise[blocks_per_row + b];
            acc0[c] += static_cast<float>(dot0) * block_d * scale_a0;
            acc1[c] += static_cast<float>(dot1) * block_d * scale_a1;

            if constexpr (Traits::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;
                int32_t sum0 = 0;
                int32_t sum1 = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    sum0 = __builtin_amdgcn_sdot4(a0[g], ones, sum0, false);
                    sum1 = __builtin_amdgcn_sdot4(a1[g], ones, sum1, false);
                }

                const uint16_t min_bits = d_block_mins[linear];
                const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                acc0[c] += static_cast<float>(sum0) * block_m * scale_a0;
                acc1[c] += static_cast<float>(sum1) * block_m * scale_a1;
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        if (kblocks == 1)
        {
            d_C_fp32[n] = acc0[c];
            d_C_fp32[static_cast<size_t>(N) + n] = acc1[c];
        }
        else
        {
            d_partial_fp32[(static_cast<size_t>(kb) * 2u) * N + n] = acc0[c];
            d_partial_fp32[(static_cast<size_t>(kb) * 2u + 1u) * N + n] = acc1[c];
        }
    }
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

// ============================================================================
// IQ LDS kernel: LDS-cached grid lookups for IQ grid formats
//
// Uses 256-thread workgroups (4 wavefronts) with cooperative LDS preload of
// the grid lookup table.  Eliminates constant cache thrashing from divergent
// per-thread grid accesses — the dominant bottleneck for IQ formats.
//
// Microbenchmark results (decode-only, 7B FFN_Up shape):
//   IQ1_S/M: +190% decode speedup  (16KB grid → 4 WGs/CU, 16 waves)
//   IQ3_S:   +115% decode speedup  (2KB grid  → 10 WGs/CU, 40 waves)
//   IQ2_S:   +82%  decode speedup  (8KB grid  → 8 WGs/CU, 32 waves)
//   IQ2_XS:  +76%  decode speedup  (4KB grid  → 10 WGs/CU, 40 waves)
//
// Otherwise identical to the main kernel: sdot4 accumulation, FP16 scales,
// dual-scale, asymmetric min correction, scatter+reduce K-partitioning.
// ============================================================================
template <int IQ_TN, NativeVNNIFormat FMT>
__global__ void gemv_iq_lds_kernel_t(
    const int8_t*    __restrict__ d_A_int8,
    const uint8_t*   __restrict__ d_payload,
    const uint16_t*  __restrict__ d_block_scales,
    const uint16_t*  __restrict__ d_block_mins,
    const uint32_t*  __restrict__ d_block_emins,
    float*           __restrict__ d_C_fp32,
    float*           __restrict__ d_partial_fp32,
    const float*     __restrict__ d_scale_A,
    const float*     __restrict__ d_scale_A_blockwise,
    int N, int K, int kblocks)
{
    using Traits = NVNNITraits<FMT>;
    static_assert(Traits::is_iq_grid || Traits::is_iq1_grid,
                  "LDS kernel only for IQ grid formats");

    // ---- LDS grid table ----
    __shared__ uint8_t lds_raw[Traits::grid_lds_bytes];
    [[maybe_unused]] uint32_t* lds32 = reinterpret_cast<uint32_t*>(lds_raw);
    [[maybe_unused]] uint64_t* lds64 = reinterpret_cast<uint64_t*>(lds_raw);

    // Cooperative preload: 256 threads load grid from constant → LDS
    if constexpr (Traits::is_iq3_grid)
    {
        const uint32_t* src = (FMT == NVNNI_IQ3_S) ? d_iq3s_grid : d_iq3xxs_grid;
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds32[i] = src[i];
    }
    else if constexpr (Traits::is_iq2_grid)
    {
        const uint64_t* src =
            (FMT == NVNNI_IQ2_S)   ? d_iq2s_grid :
            (FMT == NVNNI_IQ2_XS)  ? d_iq2xs_grid : d_iq2xxs_grid;
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds64[i] = src[i];
    }
    else // IQ1
    {
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds64[i] = d_iq1s_grid[i];
    }
    __syncthreads();

    // ---- Main GEMV logic ----
    const int n = blockIdx.x * IQ_TN + threadIdx.x;
    const bool active = (n < N);
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float f_acc = 0.0f;

    if (active)
    {
        for (int b = b_start; b < b_end; ++b)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);
            const float scale_ab = d_scale_A_blockwise ? d_scale_A_blockwise[b] : 1.0f;

            int32_t sum_a = 0;
            if constexpr (Traits::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);
            }

            iq_decode_accumulate_block<1, FMT, true>(
                a4, sum_a, d_payload, d_block_scales, d_block_mins,
                n, N, static_cast<size_t>(b) * static_cast<size_t>(N), &f_acc,
                lds32, lds64, scale_ab);
        }

        // ---- Apply activation scale and write output ----
        const float result = f_acc * (d_scale_A_blockwise ? 1.0f : d_scale_A[0]);
        if (kblocks == 1)
            d_C_fp32[n] = result;
        else if (d_partial_fp32 == nullptr)
            atomicAdd(&d_C_fp32[n], result);
        else
            d_partial_fp32[static_cast<size_t>(kb) * N + n] = result;
    }
}


extern "C" {

bool rocmMoE_grouped_gate_up_native_vnni_decode_table(
    const float* d_hidden,
    const DeviceNativeVNNIMatrixDesc* d_gate_desc_table,
    const DeviceNativeVNNIMatrixDesc* d_up_desc_table,
    const int* d_expert_ids,
    float* const* d_gate_outputs,
    float* const* d_up_outputs,
    int8_t* d_hidden_int8,
    float* d_hidden_scales,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
        !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table] invalid arguments\n");
        return false;
    }

    (void)hipGetLastError();
    const int blocks_per_row = K / 32;
    hipLaunchKernelGGL(moe_grouped_hidden_quantize_blockwise_kernel,
        dim3(blocks_per_row), dim3(32), 0, static_cast<hipStream_t>(stream),
        d_hidden, d_hidden_int8, d_hidden_scales, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table] hidden quantize launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    constexpr int TN = 64;
    dim3 grid((N + TN - 1) / TN, num_active);
    dim3 block(TN);

    #define LAUNCH_GROUPED_GATE_UP(FMT_TAG)                                                    \
        hipLaunchKernelGGL(                                                                    \
            HIP_KERNEL_NAME(moe_grouped_native_vnni_gate_up_kernel_t<FMT_TAG>),                \
            grid, block, 0, static_cast<hipStream_t>(stream),                                  \
            d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table,                \
            d_expert_ids, d_gate_outputs, d_up_outputs, num_active, N, K)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_GATE_UP(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_GATE_UP(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_GATE_UP(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_GATE_UP(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_GATE_UP(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_GATE_UP(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_GATE_UP(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_GATE_UP(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_GATE_UP(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_GATE_UP

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table] gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart(
    const float* d_hidden,
    const DeviceNativeVNNIMatrixDesc* d_gate_desc_table,
    const DeviceNativeVNNIMatrixDesc* d_up_desc_table,
    const int* d_expert_ids,
    float* const* d_gate_outputs,
    float* const* d_up_outputs,
    int8_t* d_hidden_int8,
    float* d_hidden_scales,
    float* d_gate_partials,
    float* d_up_partials,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int k_partitions,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    const bool valid_k_partitions = (k_partitions == 2 || k_partitions == 4 || k_partitions == 8);
    if (!d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
        !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
        !d_gate_partials || !d_up_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || !valid_k_partitions)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart] invalid arguments\n");
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();
    const int blocks_per_row = K / 32;
    hipLaunchKernelGGL(moe_grouped_hidden_quantize_blockwise_kernel,
        dim3(blocks_per_row), dim3(32), 0, hip_stream,
        d_hidden, d_hidden_int8, d_hidden_scales, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart] hidden quantize launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    constexpr int TN = 64;
    dim3 partial_grid((N + TN - 1) / TN, k_partitions, num_active);
    dim3 block(TN);

    #define LAUNCH_GROUPED_GATE_UP_KPART(FMT_TAG)                                             \
        hipLaunchKernelGGL(                                                                    \
            HIP_KERNEL_NAME(moe_grouped_native_vnni_gate_up_kpart_kernel_t<FMT_TAG>),          \
            partial_grid, block, 0, hip_stream,                                                \
            d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table,                \
            d_expert_ids, d_gate_partials, d_up_partials, num_active, N, K, k_partitions)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_GATE_UP_KPART(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_GATE_UP_KPART

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart] partial gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    dim3 reduce_grid((N + TN - 1) / TN, num_active);
    hipLaunchKernelGGL(moe_grouped_native_vnni_gate_up_kpart_reduce_kernel,
        reduce_grid, block, 0, hip_stream,
        d_gate_partials, d_up_partials, d_gate_outputs, d_up_outputs,
        num_active, N, k_partitions);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart] reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_vnni_decode(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeVNNIMatrixDesc* d_descs,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    float* d_output,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!d_gate_ptrs || !d_up_ptrs || !d_descs || !d_weights ||
        !d_swiglu_int8 || !d_swiglu_scales || !d_output ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode] invalid arguments\n");
        return false;
    }

    (void)hipGetLastError();
    constexpr int QUANT_THREADS = 256;
    hipLaunchKernelGGL(moe_grouped_swiglu_quantize_blockwise_kernel,
        dim3(num_active), dim3(QUANT_THREADS), 0, static_cast<hipStream_t>(stream),
        d_gate_ptrs, d_up_ptrs, d_swiglu_int8, d_swiglu_scales,
        num_active, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode] quantize launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    constexpr int TN = 64;
    dim3 grid((N + TN - 1) / TN);
    dim3 block(TN);

    #define LAUNCH_GROUPED_DOWN_DIRECT(FMT_TAG)                                               \
        hipLaunchKernelGGL(                                                                    \
            HIP_KERNEL_NAME(moe_grouped_native_vnni_down_kernel_t<FMT_TAG, false>),           \
            grid, block, 0, static_cast<hipStream_t>(stream),                                  \
            d_swiglu_int8, d_swiglu_scales, d_descs, nullptr, d_weights,                      \
            d_output, num_active, N, K)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_DOWN_DIRECT(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_DOWN_DIRECT

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode] down launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeVNNIMatrixDesc* d_desc_table,
    const int* d_expert_ids,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    float* d_output,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!d_gate_ptrs || !d_up_ptrs || !d_desc_table || !d_expert_ids || !d_weights ||
        !d_swiglu_int8 || !d_swiglu_scales || !d_output ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] invalid arguments\n");
        return false;
    }

    (void)hipGetLastError();
    constexpr int QUANT_THREADS = 256;
    hipLaunchKernelGGL(moe_grouped_swiglu_quantize_blockwise_kernel,
        dim3(num_active), dim3(QUANT_THREADS), 0, static_cast<hipStream_t>(stream),
        d_gate_ptrs, d_up_ptrs, d_swiglu_int8, d_swiglu_scales,
        num_active, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] quantize launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    constexpr int TN = 64;
    dim3 grid((N + TN - 1) / TN);
    dim3 block(TN);

    #define LAUNCH_GROUPED_DOWN_TABLE(FMT_TAG)                                                \
        hipLaunchKernelGGL(                                                                    \
            HIP_KERNEL_NAME(moe_grouped_native_vnni_down_kernel_t<FMT_TAG, true>),            \
            grid, block, 0, static_cast<hipStream_t>(stream),                                  \
            d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,            \
            d_output, num_active, N, K)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_DOWN_TABLE(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_DOWN_TABLE(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_DOWN_TABLE

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] down launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeVNNIMatrixDesc* d_desc_table,
    const int* d_expert_ids,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    float* d_output,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!d_gate_ptrs || !d_up_ptrs || !d_desc_table || !d_expert_ids || !d_weights ||
        !d_swiglu_int8 || !d_swiglu_scales || !d_output ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel] invalid arguments\n");
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();
    constexpr int QUANT_THREADS = 256;
    hipLaunchKernelGGL(moe_grouped_swiglu_quantize_blockwise_kernel,
        dim3(num_active), dim3(QUANT_THREADS), 0, hip_stream,
        d_gate_ptrs, d_up_ptrs, d_swiglu_int8, d_swiglu_scales,
        num_active, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel] quantize launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    err = hipMemsetAsync(d_output, 0, static_cast<size_t>(N) * sizeof(float), hip_stream);
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel] output memset failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    constexpr int TN = 64;
    dim3 grid((N + TN - 1) / TN, num_active);
    dim3 block(TN);

    #define LAUNCH_GROUPED_DOWN_PARALLEL(FMT_TAG)                                             \
        hipLaunchKernelGGL(                                                                    \
            HIP_KERNEL_NAME(moe_grouped_native_vnni_down_parallel_experts_kernel_t<FMT_TAG>),  \
            grid, block, 0, hip_stream,                                                        \
            d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,            \
            d_output, num_active, N, K)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_DOWN_PARALLEL(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_DOWN_PARALLEL

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel] down launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"

// ============================================================================
// Native-VNNI GEMV dispatch: lossless decode with FP16 per-block scales
//
// 2-tier dispatch (matching INT8 scatter pattern):
//   KB=1  → Direct:         single kernel writes final output, no partials
//   KB>1  → Scatter+Reduce: scatter writes partials, reduce sums to output
//
// Eliminates atomicAdd and output memset from the old implementation.
// ============================================================================
template <int SMALL_M, NativeVNNIFormat FMT>
bool rocmLaunch_native_vnni_small_m_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    const uint16_t* d_block_mins,
    const uint32_t* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    float* d_partial_fp32,
    int N, int K,
    int device_id, void* stream)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = 64;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    const int blocks_per_row = K / 32;
    const int grid_n = (N + TN - 1) / TN;
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = g_native_vnni_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
    const int kb_override = (api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb;
    const int waves_override = (api_waves_override > 0) ? api_waves_override : rocm_env.nvnni_gemv_target_waves;

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    int kb_raw = max(1, (target_total_waves + grid_n - 1) / grid_n);
    int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
    kb_max = min(kb_max, KB_CAP);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (blocks_per_row % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }
    if (kb_override > 0)
    {
        kb = min(kb_override, KB_CAP);
        if (kb > blocks_per_row) kb = blocks_per_row;
    }
    if (kb < 1) kb = 1;

    if (kb > 1 && !d_partial_fp32)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] kb=%d>1 but no small-M partial buffer (M=%d N=%d K=%d)\n",
                kb, SMALL_M, N, K);
        return false;
    }

    dim3 block(THREADS, 1, 1);
    dim3 grid(grid_n, kb, 1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_native_vnni_small_m_kernel_t<TN, CPT, SMALL_M, FMT>),
        grid, block, 0, static_cast<hipStream_t>(stream),
        d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
        d_C_fp32, d_partial_fp32, d_scale_A_blockwise, N, K, kb);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] kernel launch failed: %s (M=%d N=%d K=%d codebook=%u)\n",
                hipGetErrorString(err), SMALL_M, N, K, static_cast<unsigned>(FMT));
        return false;
    }

    if (kb > 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_native_vnni_reduce_small_m_kernel_t<TN, CPT, SMALL_M>),
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_partial_fp32, d_C_fp32, N, kb);
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] reduce launch failed: %s (M=%d N=%d K=%d codebook=%u)\n",
                hipGetErrorString(err), SMALL_M, N, K, static_cast<unsigned>(FMT));
        return false;
    }

    return true;
}

template <int SMALL_M, NativeVNNIFormat FMT>
bool rocmLaunch_native_vnni_small_m_batched_fp32(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int K,
    int device_id,
    void* stream)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = 64;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    (void)device_id;

    const int blocks_per_row = K / 32;
    NativeVNNISmallMBatchedArgs args{};
    args.num_projections = num_projections;
    for (int p = 0; p < num_projections; ++p)
    {
        const int grid_n = (Ns[p] + TN - 1) / TN;
        args.N[p] = Ns[p];
        args.grid_offsets[p] = args.total_grid_n;
        args.total_grid_n += grid_n;
        args.payload[p] = d_payloads[p];
        args.scales[p] = d_block_scales[p];
        args.mins[p] = d_block_mins ? d_block_mins[p] : nullptr;
        args.emins[p] = d_block_emins ? d_block_emins[p] : nullptr;
        args.bias[p] = d_biases ? d_biases[p] : nullptr;
        args.output[p] = d_outputs[p];
        args.partial[p] = d_partials ? d_partials[p] : nullptr;
    }
    args.grid_offsets[num_projections] = args.total_grid_n;

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = g_native_vnni_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
    const int kb_override = (api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb;
    const int waves_override = (api_waves_override > 0) ? api_waves_override : rocm_env.nvnni_gemv_target_waves;

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    int kb_raw = max(1, (target_total_waves + args.total_grid_n - 1) / args.total_grid_n);
    int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
    kb_max = min(kb_max, KB_CAP);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (blocks_per_row % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }
    if (kb_override > 0)
    {
        kb = min(kb_override, KB_CAP);
        if (kb > blocks_per_row) kb = blocks_per_row;
    }
    if (kb < 1) kb = 1;

    if (kb > 1)
    {
        for (int p = 0; p < num_projections; ++p)
        {
            if (!args.partial[p])
            {
                fprintf(stderr,
                        "[rocmGemv_native_vnni_small_m_batched_fp32] kb=%d>1 but projection %d has no partial buffer (M=%d N=%d K=%d)\n",
                        kb, p, SMALL_M, args.N[p], K);
                return false;
            }
        }
    }

    dim3 block(THREADS, 1, 1);
    dim3 grid(args.total_grid_n, kb, 1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_native_vnni_small_m_batched_kernel_t<TN, CPT, SMALL_M, FMT>),
        grid, block, 0, static_cast<hipStream_t>(stream),
        d_A_int8, d_scale_A_blockwise, args, K, kb);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_fp32] kernel launch failed: %s (M=%d K=%d projections=%d codebook=%u)\n",
                hipGetErrorString(err), SMALL_M, K, num_projections, static_cast<unsigned>(FMT));
        return false;
    }

    if (kb > 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_native_vnni_reduce_small_m_batched_kernel_t<TN, CPT, SMALL_M>),
            dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream),
            args, kb);
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_fp32] reduce launch failed: %s (M=%d K=%d projections=%d codebook=%u)\n",
                hipGetErrorString(err), SMALL_M, K, num_projections, static_cast<unsigned>(FMT));
        return false;
    }

    return true;
}

extern "C" {

bool rocmGemv_native_vnni_small_m_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (M < 2 || M > 4)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] M must be 2, 3, or 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] K must be multiple of 32 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    if (!d_A_int8 || !d_payload || !d_block_scales || !d_C_fp32 || !d_scale_A_blockwise)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] missing required pointer (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16 = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32 = static_cast<const uint32_t*>(d_block_emins);

    #define LAUNCH_SMALL_M(FMT_TAG)                                                              \
        do {                                                                                      \
            using Traits = NVNNITraits<FMT_TAG>;                                                  \
            if constexpr (Traits::is_dual_scale || Traits::is_asymmetric)                         \
            {                                                                                     \
                if (!d_mins_u16)                                                                  \
                {                                                                                 \
                    fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n", \
                            static_cast<unsigned>(codebook_id), M, N, K);                         \
                    return false;                                                                 \
                }                                                                                 \
            }                                                                                     \
            if constexpr (Traits::needs_emin_array)                                               \
            {                                                                                     \
                if (!d_emins_u32)                                                                 \
                {                                                                                 \
                    fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n", \
                            static_cast<unsigned>(codebook_id), M, N, K);                         \
                    return false;                                                                 \
                }                                                                                 \
            }                                                                                     \
            switch (M)                                                                            \
            {                                                                                     \
            case 2: return rocmLaunch_native_vnni_small_m_fp32<2, FMT_TAG>(                       \
                        d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, d_C_fp32,     \
                        d_scale_A_blockwise, d_partial_fp32, N, K, device_id, stream);            \
            case 3: return rocmLaunch_native_vnni_small_m_fp32<3, FMT_TAG>(                       \
                        d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, d_C_fp32,     \
                        d_scale_A_blockwise, d_partial_fp32, N, K, device_id, stream);            \
            case 4: return rocmLaunch_native_vnni_small_m_fp32<4, FMT_TAG>(                       \
                        d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, d_C_fp32,     \
                        d_scale_A_blockwise, d_partial_fp32, N, K, device_id, stream);            \
            default: return false;                                                                \
            }                                                                                     \
        } while (0)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_SMALL_M(NVNNI_Q4_0);
    case NVNNI_IQ4_NL:   LAUNCH_SMALL_M(NVNNI_IQ4_NL);
    case NVNNI_Q4_1:     LAUNCH_SMALL_M(NVNNI_Q4_1);
    case NVNNI_Q5_0:     LAUNCH_SMALL_M(NVNNI_Q5_0);
    case NVNNI_Q5_1:     LAUNCH_SMALL_M(NVNNI_Q5_1);
    case NVNNI_Q6_K:     LAUNCH_SMALL_M(NVNNI_Q6_K);
    case NVNNI_Q3_K:     LAUNCH_SMALL_M(NVNNI_Q3_K);
    case NVNNI_Q2_K:     LAUNCH_SMALL_M(NVNNI_Q2_K);
    case NVNNI_IQ3_S:    LAUNCH_SMALL_M(NVNNI_IQ3_S);
    case NVNNI_IQ3_XXS:  LAUNCH_SMALL_M(NVNNI_IQ3_XXS);
    case NVNNI_IQ2_S:    LAUNCH_SMALL_M(NVNNI_IQ2_S);
    case NVNNI_IQ2_XS:   LAUNCH_SMALL_M(NVNNI_IQ2_XS);
    case NVNNI_IQ2_XXS:  LAUNCH_SMALL_M(NVNNI_IQ2_XXS);
    case NVNNI_IQ1_S:    LAUNCH_SMALL_M(NVNNI_IQ1_S);
    case NVNNI_IQ1_M:    LAUNCH_SMALL_M(NVNNI_IQ1_M);
    case NVNNI_Q8_0:     LAUNCH_SMALL_M(NVNNI_Q8_0);
    default:
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] unsupported codebook_id=%u (M=%d N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), M, N, K);
        return false;
    }

    #undef LAUNCH_SMALL_M
    return false;
}

bool rocmGemv_native_vnni_small_m_batched_fp32(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    if (M < 2 || M > 4)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] M must be 2, 3, or 4 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] K must be positive and multiple of 32 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (num_projections <= 0 || num_projections > NVNNI_SMALL_M_MAX_BATCH)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] invalid projection count=%d (max=%d)\n",
                num_projections, NVNNI_SMALL_M_MAX_BATCH);
        return false;
    }
    if (!d_A_int8 || !d_payloads || !d_block_scales || !d_outputs ||
        !d_scale_A_blockwise || !d_partials || !Ns)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] missing required pointer (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    for (int p = 0; p < num_projections; ++p)
    {
        if (Ns[p] <= 0 || !d_payloads[p] || !d_block_scales[p] || !d_outputs[p] || !d_partials[p])
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] invalid projection %d (N=%d payload=%p scales=%p output=%p partial=%p)\n",
                    p, Ns[p], static_cast<const void*>(d_payloads[p]),
                    static_cast<const void*>(d_block_scales[p]),
                    static_cast<const void*>(d_outputs[p]),
                    static_cast<const void*>(d_partials[p]));
            return false;
        }
    }

#define DISPATCH_BATCHED_FMT(FMT_TAG)                                                                   \
    do {                                                                                                \
        using Traits = NVNNITraits<FMT_TAG>;                                                            \
        if constexpr (Traits::is_dual_scale || Traits::is_asymmetric) {                                 \
            if (!d_block_mins) {                                                                        \
                fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] codebook_id=%u requires mins/aux array (M=%d K=%d projections=%d)\n", \
                        codebook_id, M, K, num_projections);                                            \
                return false;                                                                           \
            }                                                                                           \
            for (int p = 0; p < num_projections; ++p) {                                                  \
                if (!d_block_mins[p]) {                                                                 \
                    fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] projection %d codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n", \
                            p, codebook_id, M, Ns[p], K);                                               \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        if constexpr (Traits::needs_emin_array) {                                                       \
            if (!d_block_emins) {                                                                       \
                fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] codebook_id=%u requires emins array (M=%d K=%d projections=%d)\n", \
                        codebook_id, M, K, num_projections);                                            \
                return false;                                                                           \
            }                                                                                           \
            for (int p = 0; p < num_projections; ++p) {                                                  \
                if (!d_block_emins[p]) {                                                                \
                    fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] projection %d codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n", \
                            p, codebook_id, M, Ns[p], K);                                               \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        switch (M) {                                                                                    \
            case 2: return rocmLaunch_native_vnni_small_m_batched_fp32<2, FMT_TAG>(                     \
                d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins, d_biases, d_outputs, \
                d_scale_A_blockwise, d_partials, Ns, num_projections, K, device_id, stream);             \
            case 3: return rocmLaunch_native_vnni_small_m_batched_fp32<3, FMT_TAG>(                     \
                d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins, d_biases, d_outputs, \
                d_scale_A_blockwise, d_partials, Ns, num_projections, K, device_id, stream);             \
            case 4: return rocmLaunch_native_vnni_small_m_batched_fp32<4, FMT_TAG>(                     \
                d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins, d_biases, d_outputs, \
                d_scale_A_blockwise, d_partials, Ns, num_projections, K, device_id, stream);             \
            default: return false;                                                                      \
        }                                                                                               \
    } while (0)

    switch (codebook_id)
    {
    case NVNNI_Q4_0:     DISPATCH_BATCHED_FMT(NVNNI_Q4_0);
    case NVNNI_IQ4_NL:   DISPATCH_BATCHED_FMT(NVNNI_IQ4_NL);
    case NVNNI_Q4_1:     DISPATCH_BATCHED_FMT(NVNNI_Q4_1);
    case NVNNI_Q5_0:     DISPATCH_BATCHED_FMT(NVNNI_Q5_0);
    case NVNNI_Q5_1:     DISPATCH_BATCHED_FMT(NVNNI_Q5_1);
    case NVNNI_Q6_K:     DISPATCH_BATCHED_FMT(NVNNI_Q6_K);
    case NVNNI_Q3_K:     DISPATCH_BATCHED_FMT(NVNNI_Q3_K);
    case NVNNI_Q2_K:     DISPATCH_BATCHED_FMT(NVNNI_Q2_K);
    case NVNNI_IQ3_S:    DISPATCH_BATCHED_FMT(NVNNI_IQ3_S);
    case NVNNI_IQ3_XXS:  DISPATCH_BATCHED_FMT(NVNNI_IQ3_XXS);
    case NVNNI_IQ2_S:    DISPATCH_BATCHED_FMT(NVNNI_IQ2_S);
    case NVNNI_IQ2_XS:   DISPATCH_BATCHED_FMT(NVNNI_IQ2_XS);
    case NVNNI_IQ2_XXS:  DISPATCH_BATCHED_FMT(NVNNI_IQ2_XXS);
    case NVNNI_IQ1_S:    DISPATCH_BATCHED_FMT(NVNNI_IQ1_S);
    case NVNNI_IQ1_M:    DISPATCH_BATCHED_FMT(NVNNI_IQ1_M);
    case NVNNI_Q8_0:     DISPATCH_BATCHED_FMT(NVNNI_Q8_0);
    default:
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] unsupported codebook_id=%u (M=%d K=%d projections=%d)\n",
                codebook_id, M, K, num_projections);
        return false;
    }

#undef DISPATCH_BATCHED_FMT
}

bool rocmGemv_native_vnni_q4_m2_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    float* d_partial_fp32,
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    if (codebook_id != static_cast<uint8_t>(NVNNI_Q4_0) &&
        codebook_id != static_cast<uint8_t>(NVNNI_Q4_1))
    {
        return false;
    }
    return rocmGemv_native_vnni_small_m_fp32(
        d_A_int8, d_payload, d_block_scales, d_block_mins, nullptr,
        d_C_fp32, d_scale_A_blockwise, d_partial_fp32,
        2, N, K, codebook_id, device_id, stream);
}

bool rocmGemv_native_vnni_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,  // __half* (FP16 scale d)
    const void* d_block_mins,    // __half* (FP16 min m), NULL for symmetric formats
    const void* d_block_emins,   // uint32_t* packed {lo,hi} emins (Q2_K only, else nullptr)
    float* d_C_fp32,
    const float* d_scale_A,
    float* d_partial_fp32,       // [KB_MAX × N] partial buffer (nullable when KB=1)
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream,
    const float* d_scale_A_blockwise)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int NUM_CUS = 60;        // MI50/MI60
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = 64;         // Must match SCATTER_KB_MAX in workspace alloc

    const int blocks_per_row = K / 32;
    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16   = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32  = static_cast<const uint32_t*>(d_block_emins);

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const bool use_atomic_reduce = rocm_env.nvnni_atomic_reduce;

    // =========================================================================
    // IQ grid formats: LDS-cached grid kernel with 256-thread workgroups
    //
    // Constant memory grid lookups are the dominant bottleneck for IQ formats.
    // Preloading the grid table into LDS gives 76-190% decode speedup.
    // Uses 4 wavefronts/WG to amortize the one-time LDS preload cost.
    //
    // Occupancy (MI60, 64KB LDS, 256 threads = 4 waves/WG):
    //   IQ3_XXS (1KB):  10 WGs/CU → 40 waves — HW max
    //   IQ3_S   (2KB):  10 WGs/CU → 40 waves — HW max
    //   IQ2_XXS (2KB):  10 WGs/CU → 40 waves — HW max
    //   IQ2_XS  (4KB):  10 WGs/CU → 40 waves — HW max
    //   IQ2_S   (8KB):   8 WGs/CU → 32 waves
    //   IQ1_S/M (16KB):  4 WGs/CU → 16 waves (offset by +190% decode speed)
    //
    // Shape-aware dispatch (two criteria, either triggers LDS kernel):
    //
    //   1. CU saturation: grid_n = ceil(N/256) >= NUM_CUS/2
    //      Ensures enough N-blocks to distribute across CUs.
    //      Below this, TN=64 kernel creates 4× more blocks.
    //
    //   2. K-heavy aspect ratio: K/N >= 4 && grid_n >= MIN_GRID_N
    //      K-dominated shapes (FFN_Dn) have long K-loops with many grid lookups.
    //      LDS caching eliminates constant memory contention even at low N.
    //      K-partitioning (scatter+reduce) provides the needed parallelism.
    //      The grid_n floor ensures the LDS kernel has workable parallelism.
    //
    // All thresholds are ratio-based or hardware-relative — no hardcoded
    // row/column counts — so they scale to arbitrary model dimensions.
    // =========================================================================
    const bool is_iq_format = (codebook_id >= 11 && codebook_id <= 17);

    // Strategy 1: CU-saturation check — need enough N-blocks for the wider tile
    constexpr int IQ_LDS_TN = 256;
    constexpr int IQ_LDS_MIN_N_BLOCKS = NUM_CUS / 2;  // Hardware-relative: half the CU count
    const int grid_n_lds = (N + IQ_LDS_TN - 1) / IQ_LDS_TN;
    const bool n_saturated = (grid_n_lds >= IQ_LDS_MIN_N_BLOCKS);

    // Strategy 2: K-heavy aspect ratio — long K-loops benefit most from LDS grid
    // cache because they do many grid lookups per thread.  Uses two ratio-based
    // checks instead of absolute block counts:
    //   - K/N ratio >= 4: shape is K-dominated (FFN_Dn pattern)
    //   - grid_n_lds >= 6 (~10% of CUs): minimum parallelism for LDS kernel's
    //     256-thread workgroups.  Shapes with fewer N-blocks (e.g., 0.5B FFN_Dn
    //     with grid_n=4) don't have enough work for the wider tile.
    constexpr float IQ_K_HEAVY_RATIO = 4.0f;          // Aspect ratio threshold
    constexpr int IQ_K_HEAVY_MIN_GRID_N = NUM_CUS / 10;  // ~10% of CUs
    const bool is_k_heavy = (K >= IQ_K_HEAVY_RATIO * N) &&
                            (grid_n_lds >= IQ_K_HEAVY_MIN_GRID_N);

    const bool use_iq_lds = is_iq_format && (n_saturated || is_k_heavy);

    if (use_iq_lds)
    {
        constexpr int IQ_TN = 256;         // 256 columns per block
        constexpr int IQ_THREADS = 256;    // 4 wavefronts per workgroup

        const int grid_n = grid_n_lds;     // Already computed above

        // LDS-aware KB heuristic: max waves/CU depends on grid table LDS size
        // and decode VALU cost.
        //
        // O8: Reduced occupancy for IQ2 formats.
        //
        // On gfx906, the wavefront scheduler round-robins ALL waves on a SIMD
        // before returning to the first.  When the inner loop has many VALU
        // instructions per block, high occupancy creates a long VALU round
        // that starves the memory pipeline.
        //
        // IQ2 decode has 127 VALU/block.  At 8 waves/SIMD the round takes
        // 1016 cycles, leaving memory idle ~80%.  Reducing to 4/SIMD (16
        // waves/CU) halves the round to 508 cycles.
        //
        // IQ3 has 94 VALU/block (ISA-verified), giving a 940-cycle round at
        // 10 waves/SIMD.  Despite this, 40 waves remains optimal because IQ3
        // has fewer memory bytes per block (15B vs ~25B for IQ2), and the 8
        // ds_read_b32 per block benefit from occupancy-based LDS latency
        // hiding.  Tested at 20 waves (O9 experiment) — no improvement.
        //
        // IQ1 is already at 16 waves from LDS size (16KB/WG).
        int max_waves_per_cu;
        if (codebook_id >= 16)       // IQ1_S, IQ1_M (16KB LDS)
            max_waves_per_cu = 16;
        else if (codebook_id >= 13 && codebook_id <= 15)  // IQ2_S, IQ2_XS, IQ2_XXS
            max_waves_per_cu = 16;
        else                         // IQ3_S, IQ3_XXS (≤2KB LDS, 94 VALU/block)
            max_waves_per_cu = 40;

        // Each 256-thread block contains 4 wavefronts (wavefront64).
        // Convert wave target to block target before computing KB.
        constexpr int WAVES_PER_BLOCK = IQ_THREADS / 64;  // 4
        int target_total_blocks = (max_waves_per_cu * NUM_CUS) / WAVES_PER_BLOCK;
        int kb_raw = max(1, (target_total_blocks + grid_n - 1) / grid_n);

        // Each partition must have enough K-blocks to amortize the LDS preload
        // + __syncthreads().  Without this, small-N shapes over-partition K
        // (e.g., 3B_AttnOut: kb=64 → 1 K-block/partition → 77μs → 18% eff).
        constexpr int IQ_MIN_K_BLOCKS_PER_PART = 8;
        int kb_max = max(1, blocks_per_row / IQ_MIN_K_BLOCKS_PER_PART);
        kb_max = min(kb_max, KB_CAP);
        kb_raw = min(kb_raw, kb_max);

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

        if (kb > 1 && !d_partial_fp32 && !use_atomic_reduce)
        {
            fprintf(stderr, "[rocmGemv_native_vnni_fp32] IQ: kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        dim3 block(IQ_THREADS, 1, 1);

        #define LAUNCH_IQ_LDS(FMT_TAG)                                                       \
            if (kb <= 1) {                                                                    \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_iq_lds_kernel_t<IQ_TN, FMT_TAG>),                   \
                    dim3(grid_n, 1), block, 0, static_cast<hipStream_t>(stream),              \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,               \
                    d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, 1);       \
            } else if (use_atomic_reduce) {                                                   \
                hipMemsetAsync(d_C_fp32, 0, N * sizeof(float), static_cast<hipStream_t>(stream)); \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_iq_lds_kernel_t<IQ_TN, FMT_TAG>),                   \
                    dim3(grid_n, kb), block, 0, static_cast<hipStream_t>(stream),             \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,               \
                    d_C_fp32, nullptr, d_scale_A, d_scale_A_blockwise, N, K, kb);             \
            } else {                                                                          \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_iq_lds_kernel_t<IQ_TN, FMT_TAG>),                   \
                    dim3(grid_n, kb), block, 0, static_cast<hipStream_t>(stream),             \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,               \
                    d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, kb);      \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_native_vnni_reduce_kernel_t<IQ_TN, 1>),              \
                    dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),                 \
                    d_partial_fp32, d_C_fp32, N, kb);                                         \
            }

        switch (static_cast<NativeVNNIFormat>(codebook_id))
        {
        case NVNNI_IQ3_S:    LAUNCH_IQ_LDS(NVNNI_IQ3_S);    break;
        case NVNNI_IQ3_XXS:  LAUNCH_IQ_LDS(NVNNI_IQ3_XXS);  break;
        case NVNNI_IQ2_S:    LAUNCH_IQ_LDS(NVNNI_IQ2_S);    break;
        case NVNNI_IQ2_XS:   LAUNCH_IQ_LDS(NVNNI_IQ2_XS);   break;
        case NVNNI_IQ2_XXS:  LAUNCH_IQ_LDS(NVNNI_IQ2_XXS);  break;
        case NVNNI_IQ1_S:    LAUNCH_IQ_LDS(NVNNI_IQ1_S);    break;
        case NVNNI_IQ1_M:    LAUNCH_IQ_LDS(NVNNI_IQ1_M);    break;
        default:
            fprintf(stderr, "[rocmGemv_native_vnni_fp32] Unexpected IQ codebook_id=%u\n",
                    static_cast<unsigned>(codebook_id));
            return false;
        }

        #undef LAUNCH_IQ_LDS
    }
    else
    {
        // =====================================================================
        // Non-IQ formats + IQ small-N fallback: TN=64, 1 wavefront/WG
        // =====================================================================
        constexpr int TN = 64;
        constexpr int CPT = 1;
        constexpr int THREADS = TN / CPT;  // 64 threads = 1 wavefront

        const int grid_n = (N + TN - 1) / TN;

        // Check runtime tuning overrides. The explicit C API wins over env so
        // microbench sweeps can override process-level defaults deterministically.
        const int api_kb_override = g_native_vnni_kb_override.load(std::memory_order_relaxed);
        const int api_waves_override = g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
        const int kb_override = (api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb;
        const int waves_override = (api_waves_override > 0) ? api_waves_override : rocm_env.nvnni_gemv_target_waves;

        constexpr int DEFAULT_TARGET_WAVES_PER_CU = 24;

        // Q8_0 (codebook_id=19) has a trivial decode path (direct INT8 load),
        // so it benefits from lower occupancy pressure. Tuning sweep shows
        // tw=4 matches or beats INT8-VNNI on MI50/MI60 across all 7B shapes.
        constexpr int Q8_0_TARGET_WAVES_PER_CU = 4;
        const int default_waves = (codebook_id == static_cast<uint8_t>(NVNNI_Q8_0))
                                      ? Q8_0_TARGET_WAVES_PER_CU
                                      : DEFAULT_TARGET_WAVES_PER_CU;

        const int TARGET_WAVES_PER_CU = (waves_override > 0) ? waves_override : default_waves;
        int target_total_waves = TARGET_WAVES_PER_CU * NUM_CUS;
        int kb_raw = max(1, (target_total_waves + grid_n - 1) / grid_n);
        int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
        kb_max = min(kb_max, KB_CAP);
        kb_raw = min(kb_raw, kb_max);

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

        // Experimental Q8_0 direct mode removes the second reduce launch for
        // decode projections. It is opt-in while shape-level throughput is
        // validated against the production split-K heuristic.
        if (rocm_env.nvnni_q8_direct && codebook_id == static_cast<uint8_t>(NVNNI_Q8_0))
        {
            kb = 1;
        }
        else if (kb_override > 0)
        {
            kb = min(kb_override, KB_CAP);
            if (kb > blocks_per_row) kb = blocks_per_row;
        }

        if (kb > 1 && !d_partial_fp32 && !use_atomic_reduce)
        {
            fprintf(stderr, "[rocmGemv_native_vnni_fp32] kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        dim3 block(THREADS, 1, 1);

        #define LAUNCH_NATIVE_VNNI(FMT_TAG)                                                  \
            if (kb <= 1) {                                                                    \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN, CPT, FMT_TAG>),            \
                    dim3(grid_n, 1), block, 0, static_cast<hipStream_t>(stream),             \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,              \
                    d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, 1);      \
            } else if (use_atomic_reduce) {                                                   \
                hipMemsetAsync(d_C_fp32, 0, N * sizeof(float), static_cast<hipStream_t>(stream)); \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN, CPT, FMT_TAG>),            \
                    dim3(grid_n, kb), block, 0, static_cast<hipStream_t>(stream),            \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,              \
                    d_C_fp32, nullptr, d_scale_A, d_scale_A_blockwise, N, K, kb);            \
            } else {                                                                          \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN, CPT, FMT_TAG>),            \
                    dim3(grid_n, kb), block, 0, static_cast<hipStream_t>(stream),            \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,              \
                    d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, kb);     \
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
        // IQ formats fall through here when N ≤ IQ_LDS_N_THRESHOLD
        case NVNNI_IQ3_S:    LAUNCH_NATIVE_VNNI(NVNNI_IQ3_S);    break;
        case NVNNI_IQ3_XXS:  LAUNCH_NATIVE_VNNI(NVNNI_IQ3_XXS);  break;
        case NVNNI_IQ2_S:    LAUNCH_NATIVE_VNNI(NVNNI_IQ2_S);    break;
        case NVNNI_IQ2_XS:   LAUNCH_NATIVE_VNNI(NVNNI_IQ2_XS);   break;
        case NVNNI_IQ2_XXS:  LAUNCH_NATIVE_VNNI(NVNNI_IQ2_XXS);  break;
        case NVNNI_IQ1_S:    LAUNCH_NATIVE_VNNI(NVNNI_IQ1_S);    break;
        case NVNNI_IQ1_M:    LAUNCH_NATIVE_VNNI(NVNNI_IQ1_M);    break;
        case NVNNI_Q8_0:     LAUNCH_NATIVE_VNNI(NVNNI_Q8_0);     break;
        default:
            fprintf(stderr, "[rocmGemv_native_vnni_fp32] Unknown codebook_id=%u (N=%d K=%d)\n",
                    static_cast<unsigned>(codebook_id), N, K);
            return false;
        }

        #undef LAUNCH_NATIVE_VNNI
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] Kernel launch failed: %s (N=%d K=%d codebook=%u)\n",
                hipGetErrorString(err), N, K, static_cast<unsigned>(codebook_id));
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
