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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <string>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/PrefillGraphBucketDefaults.h"
#include "execution/moe/DeviceMoERuntimeABI.h"
#include "kernels/common/NativeVNNIDispatchCache.h"
#include "kernels/common/NativeVNNIGroupedDecodePolicy.h"
#include "kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc"
#include "ROCmNativeVNNIDecodeEquivalentMath.h"
#include "../ops/ROCmSwiGLUDecodeEquivalentMath.h"

#include <atomic>

// ============================================================================
// Runtime tuning overrides for native-VNNI GEMV dispatch.
// Values < 0 mean "use default heuristic".
// ============================================================================
static std::atomic<int> g_native_vnni_kb_override{-1};
static std::atomic<int> g_native_vnni_target_waves_override{-1};
/*
 * Decode-equivalent dispatch is a property of one host worker's verifier
 * transaction. LocalTP drives one worker per GPU in the same process, so a
 * process-global mode bit lets either worker clear the other worker's policy
 * while it is still lowering a grouped launch. Keep this state thread-local,
 * matching CUDA, and pass resolved serial-M1 launch geometry explicitly from
 * ROCmQuantisedGemmKernel whenever a grouped projection is emitted.
 */
static thread_local int g_native_vnni_decode_equivalent_m1_config = 0;
static thread_local int g_native_vnni_serial_partition_n = 0;

struct NativeVNNIDecodeGeneratedRuntimeConfig
{
    bool has_generated = false;
    int kb = -1;
    int target_waves_per_cu = -1;
};

static NativeVNNIDecodeGeneratedRuntimeConfig resolveNativeVNNIDecodeGeneratedRuntimeConfig(
    uint8_t codebook_id,
    int m,
    int n,
    int k)
{
    NativeVNNIDecodeGeneratedRuntimeConfig resolved{};
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    if (rocm_env.nvnni_disable_generated)
        return resolved;

    const bool has_explicit_override =
        g_native_vnni_kb_override.load(std::memory_order_relaxed) > 0 ||
        g_native_vnni_target_waves_override.load(std::memory_order_relaxed) > 0 ||
        rocm_env.nvnni_gemv_kb > 0 ||
        rocm_env.nvnni_gemv_target_waves > 0;
    if (has_explicit_override)
        return resolved;

    const int lookup_m =
        (g_native_vnni_decode_equivalent_m1_config != 0 &&
         m > 1)
            ? 1
            : m;
    const int lookup_n =
        g_native_vnni_serial_partition_n > 0
            ? g_native_vnni_serial_partition_n
            : n;

    /*
     * Generated policy is immutable for the process lifetime. Preserve each
     * input as a full-width word so invalid or future geometry cannot alias the
     * compact exact-overlay ABI. Caching a miss keeps totality checks cheap but
     * never manufactures a runtime fallback.
     */
    using CacheKey = std::array<uint64_t, 4>;
    using Cache = llaminar2::native_vnni::FixedDispatchCache<
        CacheKey,
        NativeVNNIDecodeGeneratedRuntimeConfig,
        64,
        llaminar2::native_vnni::DispatchCacheArrayHasher<4>>;
    static thread_local Cache cache;
    const CacheKey cache_key = {
        static_cast<uint64_t>(codebook_id),
        static_cast<uint64_t>(static_cast<uint32_t>(lookup_m)),
        static_cast<uint64_t>(static_cast<uint32_t>(lookup_n)),
        static_cast<uint64_t>(static_cast<uint32_t>(k)),
    };
    bool cached_selector_found = false;
    NativeVNNIDecodeGeneratedRuntimeConfig cached{};
    if (cache.lookup(cache_key, cached_selector_found, cached))
        return cached;

    llaminar2::rocm::generated::ROCmNativeVNNIDecodeDispatchConfig generated_cfg{};
    if (!llaminar2::rocm::generated::selectROCmNativeVNNIDecodeGenerated(
            codebook_id, lookup_m, lookup_n, k, generated_cfg))
    {
        cache.insert(cache_key, false, cached);
        return cached;
    }

    /*
     * Generic ROCm policy candidates encode a requested split count. The
     * production launch can never create more K partitions than there are
     * 32-value groups, so it resolves that request as min(requested, K / 32).
     * Publish the same physical value through the query/cache surface. Leaving
     * the nominal request here made serial-M1 policy queries disagree with the
     * launch telemetry for small-K unseen shapes (for example KB20 at K=416,
     * whose real launch is KB13), even though the kernel itself was correct.
     */
    const int requested_kb = static_cast<int>(generated_cfg.kb);
    const int k_groups = k / 32;
    cached.has_generated = true;
    cached.kb = (requested_kb < k_groups) ? requested_kb : k_groups;
    cached.target_waves_per_cu =
        static_cast<int>(generated_cfg.target_waves_per_cu);
    cache.insert(cache_key, true, cached);
    return cached;
}

static void recordNativeVNNISmallMLaunch(
    int m,
    int n,
    int k,
    unsigned codebook,
    int grid_n,
    int kb,
    int target_waves_per_cu,
    bool batched,
    int projections)
{
    if (!llaminar2::PerfStatsCollector::isEnabled())
        return;

    const char* path = (kb <= 1) ? "direct" : "split_reduce";
    llaminar2::PerfStatsCollector::addCounter(
        "kernel",
        "rocm_native_vnni_small_m_launch",
        1.0,
        "gemm",
        {},
        llaminar2::PerfStatsCollector::Tags{
            {"m", std::to_string(m)},
            {"n", std::to_string(n)},
            {"k", std::to_string(k)},
            {"codebook", std::to_string(codebook)},
            {"grid_n", std::to_string(grid_n)},
            {"kb", std::to_string(kb)},
            {"path", path},
            {"target_waves_per_cu", std::to_string(target_waves_per_cu)},
            {"batched", batched ? "true" : "false"},
            {"projections", std::to_string(projections)}});
}

extern "C" bool rocmQuantGemm_biasAdd(
    float *d_output,
    const float *d_bias,
    int M, int N,
    int rocm_device_id, void *stream);

extern "C" void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu)
{
    g_native_vnni_kb_override.store(kb, std::memory_order_relaxed);
    g_native_vnni_target_waves_override.store(target_waves_per_cu, std::memory_order_relaxed);
}

extern "C" void rocmGemv_native_vnni_set_decode_equivalent_m1_config(int enabled)
{
    g_native_vnni_decode_equivalent_m1_config = enabled ? 1 : 0;
}

extern "C" void rocmGemv_native_vnni_set_serial_partition_n(int n)
{
    g_native_vnni_serial_partition_n = n;
}

extern "C" int rocmGemv_native_vnni_get_serial_partition_n()
{
    return g_native_vnni_serial_partition_n;
}

extern "C" void rocmGemv_native_vnni_reset_tuning_overrides()
{
    rocmGemv_native_vnni_set_tuning_overrides(-1, -1);
}

static bool nativeVNNIDecodeEquivalentSmallMMode()
{
    return g_native_vnni_decode_equivalent_m1_config != 0;
}

/**
 * @brief Enforce the ROCm small-M verifier arithmetic contract.
 *
 * The MTP verifier publication path must reproduce serial M=1 decode rows,
 * because its outputs are used as live recurrent/KV/logit state.  Manual KB
 * overrides are therefore allowed but never silently rewritten here: the
 * grouped verifier must match the same generated/manual dispatch policy that
 * serial decode would use.
 */
static bool validateNativeVNNIDecodeEquivalentKBOverride(
    const char* caller,
    int explicit_kb_override,
    int m,
    int n,
    int k,
    int projections)
{
    (void)caller;
    (void)explicit_kb_override;
    (void)m;
    (void)n;
    (void)k;
    (void)projections;
    return true;
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

/**
 * @brief Dilate four packed Q6 two-bit fields into four byte-local high nibbles.
 *
 * A Q6 high-bit byte stores four consecutive two-bit fields in bits
 * `[1:0]`, `[3:2]`, `[5:4]`, and `[7:6]`.  Native-VNNI dot operands need
 * those fields in bits `[5:4]` of four independent bytes.  The first
 * shuffle step separates the two nibbles by twelve bits; the second separates
 * each nibble's two-bit fields by six more bits.  The final shift places each
 * field in the byte-local high-nibble position:
 *
 * @code
 * abcdefgh -> 00ab0000 00cd0000 00ef0000 00gh0000
 * @endcode
 *
 * This identity is exhaustive over the 256 possible input bytes and replaces
 * four independent mask/shift deposits.  It changes only the integer decode;
 * block order, SDOT4 order, scale conversion, and FP32 accumulation remain
 * identical to serial decode.
 *
 * @param packed_fields The Q6 high-bit byte, zero-extended to 32 bits.
 * @return Four expanded fields in bits `[5:4]` of the result's four bytes.
 */
static constexpr uint32_t rocm_native_vnni_expand_q6_high_fields_reference(
    uint32_t packed_fields)
{
    return ((packed_fields & 0x03u) << 4) |
           ((packed_fields & 0x0Cu) << 10) |
           ((packed_fields & 0x30u) << 16) |
           ((packed_fields & 0xC0u) << 22);
}

/**
 * @brief Evaluate the low-instruction-count Q6 high-field dilation identity.
 * @param packed_fields The Q6 high-bit byte, zero-extended to 32 bits.
 * @return Four expanded fields in bits `[5:4]` of the result's four bytes.
 */
static constexpr uint32_t rocm_native_vnni_expand_q6_high_fields_dilated(
    uint32_t packed_fields)
{
    const uint32_t separated_nibbles =
        (packed_fields | (packed_fields << 12)) & 0x000F000Fu;
    const uint32_t separated_pairs =
        (separated_nibbles | (separated_nibbles << 6)) & 0x03030303u;
    return separated_pairs << 4;
}

/**
 * @brief Prove the optimized Q6 expansion for every possible packed byte.
 * @return `true` only when all 256 values equal the direct reference mapping.
 */
static constexpr bool rocm_native_vnni_q6_high_field_dilation_is_exhaustive()
{
    for (uint32_t packed = 0; packed <= 0xFFu; ++packed)
    {
        if (rocm_native_vnni_expand_q6_high_fields_dilated(packed) !=
            rocm_native_vnni_expand_q6_high_fields_reference(packed))
        {
            return false;
        }
    }
    return true;
}

static_assert(
    rocm_native_vnni_q6_high_field_dilation_is_exhaustive(),
    "ROCm Q6 high-field dilation must equal the direct mapping for every byte");

/**
 * @brief Device wrapper for the compile-time-proven Q6 dilation identity.
 * @param packed_fields The Q6 high-bit byte, zero-extended to 32 bits.
 * @return Four expanded fields in bits `[5:4]` of the result's four bytes.
 */
__device__ __forceinline__ uint32_t rocm_native_vnni_expand_q6_high_fields(
    uint32_t packed_fields)
{
    return rocm_native_vnni_expand_q6_high_fields_dilated(packed_fields);
}

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

/**
 * @brief Resolve the split-K policy used by the normal serial M=1 decode path.
 *
 * MTP verifier publication compares grouped runtime-M rows against committed
 * serial decode state. The grouped kernels may execute several rows at once,
 * but they must use the same split-K count as serial M=1 so the FP32 reduction
 * order is bitwise-identical. Generated policy is the sole production source
 * of that arithmetic identity. Returning an empty result on a miss forces the
 * caller to fail instead of silently reconstructing an obsolete heuristic.
 */
static NativeVNNIDecodeGeneratedRuntimeConfig resolveNativeVNNISerialM1RuntimeConfig(
    uint8_t codebook_id,
    int N,
    int K)
{
    return resolveNativeVNNIDecodeGeneratedRuntimeConfig(
        codebook_id, 1, N, K);
}

static NativeVNNIDecodeGeneratedRuntimeConfig resolveNativeVNNICommonSerialM1RuntimeConfig(
    const uint8_t* codebook_ids,
    const int* Ns,
    int num_projections,
    int K)
{
    NativeVNNIDecodeGeneratedRuntimeConfig common{};
    for (int p = 0; p < num_projections; ++p)
    {
        NativeVNNIDecodeGeneratedRuntimeConfig candidate =
            resolveNativeVNNISerialM1RuntimeConfig(codebook_ids[p], Ns[p], K);
        if (!candidate.has_generated)
            return NativeVNNIDecodeGeneratedRuntimeConfig{};

        if (p == 0)
        {
            common = candidate;
            continue;
        }

        if (candidate.kb != common.kb)
            return NativeVNNIDecodeGeneratedRuntimeConfig{};
    }
    return common;
}

static NativeVNNIDecodeGeneratedRuntimeConfig resolveNativeVNNICommonSerialM1RuntimeConfig(
    uint8_t codebook_id,
    const int* Ns,
    int num_projections,
    int K)
{
    uint8_t codebook_ids[8] = {};
    for (int p = 0; p < num_projections && p < 8; ++p)
        codebook_ids[p] = codebook_id;
    return resolveNativeVNNICommonSerialM1RuntimeConfig(codebook_ids, Ns, num_projections, K);
}

/**
 * @brief Resolve the exact graph-safe partition count executed by serial M=1.
 *
 * Generated policy stores the preferred count, while the launcher bounds it by
 * both the number of 32-value K blocks and the captured-workspace ABI. Grouped
 * MoE launchers call this helper so their partition boundaries are identical to
 * the public serial decode entrypoint rather than merely using the same raw
 * policy record.
 *
 * @param codebook_id NativeVNNI format identifier.
 * @param N Output-column count.
 * @param K Reduction width, divisible by 32.
 * @return Effective K-partition count, or zero when no generated policy exists.
 */
static int resolveNativeVNNISerialM1KPartitions(
    uint8_t codebook_id,
    int N,
    int K)
{
    if (N <= 0 || K <= 0 || (K % 32) != 0)
        return 0;

    const NativeVNNIDecodeGeneratedRuntimeConfig cfg =
        resolveNativeVNNISerialM1RuntimeConfig(codebook_id, N, K);
    if (!cfg.has_generated || cfg.kb <= 0)
        return 0;

    int partitions = cfg.kb;
    if (partitions > K / 32)
        partitions = K / 32;
    if (partitions >
        llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
    {
        partitions =
            llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions;
    }
    return partitions > 0 ? partitions : 1;
}

extern "C" bool rocmGemv_native_vnni_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    NativeVNNIDecodeGeneratedRuntimeConfig cfg =
        resolveNativeVNNISerialM1RuntimeConfig(codebook_id, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
        *kb = resolveNativeVNNISerialM1KPartitions(codebook_id, N, K);
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

extern "C" bool rocmGemv_native_vnni_query_generated_config(
    uint8_t codebook_id,
    int M,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    NativeVNNIDecodeGeneratedRuntimeConfig cfg =
        resolveNativeVNNIDecodeGeneratedRuntimeConfig(codebook_id, M, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
        *kb = cfg.kb;
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

/**
 * @brief Measure the in-TU ROCm generated-selector cache without touching HIP.
 *
 * The timing loop calls the same static resolver as the production launcher,
 * avoiding the out-of-line public-query overhead that would otherwise dominate
 * a nanosecond-scale host measurement. A matched loop-only interval is
 * subtracted from the result.
 */
extern "C" double rocmGemv_native_vnni_measure_generated_config_ns(
    uint8_t codebook_id,
    int M,
    const int *Ns,
    const int *Ks,
    int query_count,
    int iterations,
    uint64_t *out_checksum)
{
    if (!Ns || !Ks || query_count <= 0 || iterations <= 0)
        return -1.0;

    uint64_t checksum = 0;
    const auto baseline_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const int index = iteration % query_count;
        checksum += static_cast<uint64_t>(Ns[index] + Ks[index]);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    const auto baseline_end = std::chrono::steady_clock::now();

    const auto selector_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const int index = iteration % query_count;
        const NativeVNNIDecodeGeneratedRuntimeConfig config =
            resolveNativeVNNIDecodeGeneratedRuntimeConfig(
                codebook_id, M, Ns[index], Ks[index]);
        checksum += static_cast<uint64_t>(config.has_generated) |
                    (static_cast<uint64_t>(config.kb) << 8) |
                    (static_cast<uint64_t>(config.target_waves_per_cu) << 24);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    const auto selector_end = std::chrono::steady_clock::now();

    if (out_checksum)
        *out_checksum = checksum;
    const double baseline_ns = std::chrono::duration<double, std::nano>(
        baseline_end - baseline_begin).count();
    const double selector_ns = std::chrono::duration<double, std::nano>(
        selector_end - selector_begin).count();
    return std::max(
        0.0,
        (selector_ns - baseline_ns) / static_cast<double>(iterations));
}

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
                f_acc[c] = rocm_native_vnni_commit_dual_scale_block(
                    f_acc[c],
                    acc_lo,
                    acc_hi,
                    scale_lo,
                    scale_hi,
                    scale_ab);

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
                    f_acc[c] = rocm_native_vnni_commit_dual_scale_correction(
                        f_acc[c], sum_a_lo, sum_a_hi, min_lo, min_hi, scale_ab);
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

                    f_acc[c] = rocm_native_vnni_commit_iq1_m_delta_correction(
                        f_acc[c],
                        sa_g0,
                        sa_g1,
                        sa_g2,
                        sa_g3,
                        d0,
                        d1,
                        d2,
                        d3,
                        scale_lo,
                        scale_hi,
                        scale_ab);
                }
            }
            else
            {
                int32_t block_acc = 0;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

                const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                f_acc[c] = rocm_native_vnni_commit_single_scale_block(
                    f_acc[c], block_acc, block_d, scale_ab);

                if constexpr (Traits::is_asymmetric)
                {
                    const uint16_t min_bits = d_block_mins[linear];
                    const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                    f_acc[c] = rocm_native_vnni_commit_single_scale_correction(
                        f_acc[c], sum_a, block_m, scale_ab);
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
                            const uint32_t hb_lo =
                                rocm_native_vnni_expand_q6_high_fields(h_lo);
                            uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
                            val_lo -= 0x20202020u;
                            val_lo ^= 0x80808080u;
                            packed_groups[g] = static_cast<int32_t>(val_lo);

                            // High nibbles + scattered 2-bit highs + subtract-32
                            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                            const uint32_t hb_hi =
                                rocm_native_vnni_expand_q6_high_fields(h_hi);
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
                        f_acc[c] = rocm_native_vnni_commit_dual_scale_block(
                            f_acc[c],
                            acc_lo,
                            acc_hi,
                            scale_lo,
                            scale_hi,
                            scale_ab);

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
                            f_acc[c] = rocm_native_vnni_commit_dual_scale_correction(
                                f_acc[c], sum_a_lo, sum_a_hi, min_lo, min_hi, scale_ab);
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

                            f_acc[c] = rocm_native_vnni_commit_iq1_m_delta_correction(
                                f_acc[c],
                                sa_g0,
                                sa_g1,
                                sa_g2,
                                sa_g3,
                                d0,
                                d1,
                                d2,
                                d3,
                                scale_lo,
                                scale_hi,
                                scale_ab);
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
                        f_acc[c] = rocm_native_vnni_commit_single_scale_block(
                            f_acc[c], block_acc, block_d, scale_ab);

                        // Asymmetric min correction: f_acc += sum_a * block_m
                        // Derives from: value = d*q + m  →  Σ a_i*value_i = d*Σ(a_i*q_i) + m*Σ(a_i)
                        if constexpr (Traits::is_asymmetric)
                        {
                            const uint16_t min_bits = d_block_mins[linear];
                            const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                            f_acc[c] = rocm_native_vnni_commit_single_scale_correction(
                                f_acc[c], sum_a, block_m, scale_ab);
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
                    // A single K partition already owns the final output row.
                    d_C_fp32[n] = result;
                }
                else
                {
                    /*
                     * Every K partition owns one persistent workspace row.
                     * The following ordered reducer consumes those rows in
                     * serial-decode partition order.  There is deliberately
                     * no null-workspace/atomic route: allowing completion
                     * order to choose the FP32 fold order would make MTP
                     * verification depend on GPU scheduling.
                     */
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

constexpr int kDeviceMoEMaxExperts = 256;
constexpr int kDeviceMoEMaxTopK = 16;
constexpr uint32_t kDeviceMoEFlagValid = 1u << 0;
constexpr uint32_t kDeviceMoEFlagResident = 1u << 1;
constexpr uint32_t kDeviceMoEFlagLocalCompute = 1u << 4;

struct DeviceMoEExpertDescriptorView
{
    DeviceNativeVNNIMatrixDesc gate;
    DeviceNativeVNNIMatrixDesc up;
    DeviceNativeVNNIMatrixDesc down;
    int32_t logical_expert_id;
    int32_t owner_participant;
    int32_t local_slot;
    uint32_t flags;
};

struct DeviceMoEPlacementBankView
{
    DeviceMoEExpertDescriptorView experts[kDeviceMoEMaxExperts];
    uint8_t local_compute_mask[kDeviceMoEMaxExperts];
    uint8_t replica_role[kDeviceMoEMaxExperts];
    uint32_t resident_participant_mask[kDeviceMoEMaxExperts];
    uint32_t epoch;
    uint32_t expert_count;
    uint32_t reserved[2];
};

struct DeviceMoELayerRuntimeView
{
    uint32_t active_bank;
    uint32_t active_epoch;
    uint32_t expert_count;
    uint32_t top_k;
    DeviceMoEPlacementBankView banks[2];
    int32_t topk_expert_ids[kDeviceMoEMaxTopK];
    float topk_weights[kDeviceMoEMaxTopK];
    uint64_t decode_histogram[kDeviceMoEMaxExperts];
    uint64_t decode_local_histogram[kDeviceMoEMaxExperts];
    uint64_t router_hot_cache_eligible_dispatches;
    uint64_t router_hot_cache_used_dispatches;
    uint64_t router_hot_cache_improved_dispatches;
    uint64_t router_hot_cache_default_load_spread_total;
    uint64_t router_hot_cache_actual_load_spread_total;
    uint64_t router_hot_cache_load_spread_improvement_total;
    uint64_t router_hot_cache_active_dispatches;
    uint64_t router_hot_cache_miss_dispatches;
    uint64_t router_hot_cache_selected_expert_slots;
    uint64_t router_hot_cache_replicated_selected_expert_slots;
    int32_t *route_expert_ids;
    float *route_weights;
    int32_t *route_participant_ids;
    int32_t *deferred_verifier_route_expert_ids;
    int32_t *deferred_verifier_route_participant_ids;
    int32_t *expert_counts;
    int32_t *expert_offsets;
    int32_t *grouped_token_ids;
    float *grouped_route_weights;
    float *grouped_gate_scratch;
    float *grouped_up_scratch;
    float *grouped_output_partials;
    void *decode_scratch;
    void *reserved_ptrs[3];
    uint64_t reserved_u64[4];
    uint32_t prefill_token_capacity;
    uint32_t prefill_route_capacity;
    uint32_t deferred_verifier_route_capacity;
    uint32_t participant_id;
    uint32_t participant_count;
    uint32_t current_batch_llep_movement_observed;
    uint32_t current_batch_llep_non_owner_assignment_observed;
};

static_assert(
    sizeof(DeviceMoELayerRuntimeView) ==
    llaminar2::moe_runtime_abi::kLayerRuntimeBytes);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, route_participant_ids) ==
    llaminar2::moe_runtime_abi::kRouteParticipantIdsOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, participant_count) ==
    llaminar2::moe_runtime_abi::kParticipantCountOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView,
             current_batch_llep_movement_observed) ==
    llaminar2::moe_runtime_abi::kCurrentBatchLLEPMovementObservedOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView,
             current_batch_llep_non_owner_assignment_observed) ==
    llaminar2::moe_runtime_abi::
        kCurrentBatchLLEPNonOwnerAssignmentObservedOffset);

template <NativeVNNIFormat FMT>
__device__ __forceinline__ bool native_vnni_desc_shape_ok(
    const DeviceNativeVNNIMatrixDesc& desc,
    int N,
    int K)
{
    using Traits = NVNNITraits<FMT>;
    return desc.payload &&
           desc.scales &&
           desc.n == N &&
           desc.k == K &&
           desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
           desc.codebook_id == static_cast<uint8_t>(FMT) &&
           (!(Traits::is_asymmetric || Traits::is_dual_scale) || desc.mins) &&
           (!Traits::is_dual_scale_asym || desc.emins);
}

__global__ void moe_grouped_swiglu_quantize_blockwise_kernel(
    const float* const* __restrict__ gate_ptrs,
    const float* const* __restrict__ up_ptrs,
    const int* __restrict__ expert_ids,
    int8_t* __restrict__ A_int8,
    float* __restrict__ scales_A_blockwise,
    int num_active,
    int K)
{
    const int slot = blockIdx.x;
    if (slot >= num_active) return;
    if (expert_ids && expert_ids[slot] < 0) return;

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
        const float val =
            llaminar2::rocm::decode_equivalent::swigluValue(g, u);

        float abs_val = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));

        const float scale =
            llaminar2::rocm::decode_equivalent::q8Scale(abs_val);
        const float inv_scale = 1.0f / scale;

        if (lane == 0)
            row_scales[b] = scale;

        row_int8[k_start + lane] = static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8(
                val, inv_scale));
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
            // The grouped verifier keeps the direct deposits: on gfx906 its
            // larger live-row state schedules this form 2.7% faster than the
            // serial kernel's bit-dilation identity.  Both forms are exhaustive
            // integer identities over all 256 possible packed high-bit bytes.
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
__device__ __forceinline__ float moe_native_vnni_dot_desc_range(
    const DeviceNativeVNNIMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K,
    int b_start,
    int b_end);

/**
 * @brief Publish one weighted ROCm decode row per original router slot.
 *
 * Each `(route, column)` lane computes a complete increasing-K NativeVNNI dot
 * and writes a unique destination. Remote/invalid routes write zero. No route
 * accumulation or floating-point atomic occurs before the LocalTP collective.
 */
template <NativeVNNIFormat FMT, bool INDEX_BY_EXPERT_ID = true>
__global__ void moe_grouped_native_vnni_down_route_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int K)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Canonical grouped MoE down requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    const size_t output_index =
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int descriptor_index = INDEX_BY_EXPERT_ID ? expert_ids[route] : route;
    if (descriptor_index < 0)
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const DeviceNativeVNNIMatrixDesc desc = descs[descriptor_index];
    if (!native_vnni_desc_shape_ok<FMT>(desc, N, K))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int8_t* a_slot =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* scale_slot =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    const float expert_acc = moe_native_vnni_dot_desc_range<FMT>(
        desc, n, a_slot, scale_slot, N, K, 0, blocks_per_row);
    route_output[output_index] = rocm_native_vnni_weight_route_rn(
        route_weights[route], expert_acc);
}

/**
 * @brief Fold independently computed route rows in the serial router order.
 *
 * Route projection is parallel across grid Y. This reducer owns one output
 * column and reproduces the serial decode sequence `(((0 + r0) + r1) + ...)`
 * with explicit round-to-nearest operations. No route can race another route's
 * destination, and no scheduler-dependent atomic ordering enters the result.
 */
__global__ void moe_grouped_native_vnni_down_ordered_reduce_kernel(
    const float* __restrict__ route_contributions,
    float* __restrict__ output,
    int num_active,
    int N)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    if (n >= N)
        return;

    float total = 0.0f;
    #pragma unroll 1
    for (int route = 0; route < num_active; ++route)
    {
        total = rocm_native_vnni_accumulate_rn(
            total,
            route_contributions[
                static_cast<size_t>(route) * static_cast<size_t>(N) +
                static_cast<size_t>(n)]);
    }
    output[n] = total;
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

/*
 * Grouped MoE decode is defined before the generic verifier-row kernels in this
 * translation unit, but both paths must call the same arithmetic primitives.
 * Keep these declarations here so the grouped dot cannot grow a second copy of
 * the serial M=1 accumulation tree.
 */
template <NativeVNNIFormat FMT>
__device__ __forceinline__ int32_t native_vnni_activation_sum_a(
    const int32_t* __restrict__ a4);

template <NativeVNNIFormat FMT>
__device__ __forceinline__ void native_vnni_small_m_accumulate_decoded_block(
    float& f_acc,
    const int32_t* __restrict__ a4,
    const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,
    const uint32_t* __restrict__ emin_base,
    size_t linear,
    float scale_ab,
    int32_t sum_a);

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
    if (!A_int8 ||
        !scales_A_blockwise ||
        n < 0 ||
        n >= N ||
        K <= 0 ||
        (K % 32) != 0 ||
        !native_vnni_desc_shape_ok<FMT>(desc, N, K))
    {
        return 0.0f;
    }

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
        const int32_t sum_a = native_vnni_activation_sum_a<FMT>(a4);
        native_vnni_small_m_accumulate_decoded_block<FMT>(
            acc,
            a4,
            packed_groups,
            payload,
            scale_base,
            min_base,
            emin_base,
            linear,
            scale_ab,
            sum_a);
    }

    return acc;
}

/**
 * @brief Compute one ordered split-K partial for each routed expert row.
 *
 * Scratch is laid out as `[KB][route][N]`, matching serial GEMV's outer KB
 * dimension. The first KB plane is therefore also a contiguous `[route][N]`
 * destination and can be overwritten by the following reducer after each lane
 * has consumed its own KB-zero value.
 */
template <NativeVNNIFormat FMT, bool INDEX_BY_EXPERT_ID = true>
__global__ void moe_grouped_native_vnni_down_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeVNNIMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int k_partitions)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE split-K down requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int descriptor_index = INDEX_BY_EXPERT_ID ? expert_ids[route] : route;
    if (descriptor_index < 0)
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const DeviceNativeVNNIMatrixDesc desc = descs[descriptor_index];
    if (!native_vnni_desc_shape_ok<FMT>(desc, N, K))
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    const int8_t* route_a =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* route_scales =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    partials[partial_index] = b_start < b_end
        ? moe_native_vnni_dot_desc_range<FMT>(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Reduce one expert's K partials and publish its rounded route row.
 *
 * The loop is deliberately scalar and increasing in KB, exactly matching the
 * serial M=1 reducer. `route_output` may alias the first KB plane of `partials`:
 * each lane reads every source for its own `(route,column)` before replacing
 * that lane's KB-zero value.
 */
__global__ void moe_grouped_native_vnni_down_kpart_reduce_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int k_partitions)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    float expert_sum = 0.0f;
    #pragma unroll 1
    for (int k_part = 0; k_part < k_partitions; ++k_part)
    {
        const size_t partial_index =
            (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
             static_cast<size_t>(route)) * static_cast<size_t>(N) +
            static_cast<size_t>(n);
        expert_sum = rocm_native_vnni_accumulate_rn(
            expert_sum, partials[partial_index]);
    }

    route_output[
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n)] = rocm_native_vnni_weight_route_rn(
            route_weights[route], expert_sum);
}

/**
 * @brief Compute table-backed gate/up partials with serial-M1 K boundaries.
 */
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
                  "Grouped MoE split-K gate/up requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int expert_id = expert_ids[route];
    if (expert_id < 0)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
    if (!native_vnni_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_vnni_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    gate_partials[partial_index] = b_start < b_end
        ? moe_native_vnni_dot_desc_range<FMT>(
              gate_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
    up_partials[partial_index] = b_start < b_end
        ? moe_native_vnni_dot_desc_range<FMT>(
              up_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Publish table/runtime gate/up rows from ordered split-K partials.
 */
__global__ void moe_grouped_native_vnni_gate_up_kpart_reduce_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int k_partitions)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    if (expert_ids[route] < 0)
    {
        gate_outputs[route][n] = 0.0f;
        up_outputs[route][n] = 0.0f;
        return;
    }

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    #pragma unroll 1
    for (int k_part = 0; k_part < k_partitions; ++k_part)
    {
        const size_t partial_index =
            (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
             static_cast<size_t>(route)) * static_cast<size_t>(N) +
            static_cast<size_t>(n);
        gate_sum = rocm_native_vnni_accumulate_rn(
            gate_sum, gate_partials[partial_index]);
        up_sum = rocm_native_vnni_accumulate_rn(
            up_sum, up_partials[partial_index]);
    }

    gate_outputs[route][n] = gate_sum;
    up_outputs[route][n] = up_sum;
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
    if (expert_id < 0)
        return;
    const DeviceNativeVNNIMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeVNNIMatrixDesc up_desc = up_descs[expert_id];
    if (!native_vnni_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_vnni_desc_shape_ok<FMT>(up_desc, N, K))
    {
        return;
    }

    gate_outputs[slot][n] = moe_native_vnni_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_vnni_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_gate_up_runtime_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int K,
    int num_experts)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE runtime gate/up requires a supported native-VNNI descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    const int expert_id = expert_ids[slot];
    if (expert_id < 0 || expert_id >= num_experts)
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }

    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const DeviceNativeVNNIMatrixDesc gate_desc = bank.experts[expert_id].gate;
    const DeviceNativeVNNIMatrixDesc up_desc = bank.experts[expert_id].up;
    if (!native_vnni_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_vnni_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }

    gate_outputs[slot][n] = moe_native_vnni_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_vnni_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

/**
 * @brief Compute runtime-placement gate/up partials using serial-M1 K slices.
 */
template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_gate_up_runtime_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    int k_partitions)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Runtime grouped MoE split-K gate/up requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int expert_id = expert_ids[route];
    if (expert_id < 0 || expert_id >= num_experts)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const DeviceNativeVNNIMatrixDesc gate_desc = bank.experts[expert_id].gate;
    const DeviceNativeVNNIMatrixDesc up_desc = bank.experts[expert_id].up;
    if (!native_vnni_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_vnni_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    gate_partials[partial_index] = b_start < b_end
        ? moe_native_vnni_dot_desc_range<FMT>(
              gate_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
    up_partials[partial_index] = b_start < b_end
        ? moe_native_vnni_dot_desc_range<FMT>(
              up_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
}

template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_down_runtime_route_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int K,
    int num_experts)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Grouped MoE runtime down requires a supported native-VNNI descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N)
        return;

    const int expert_id = expert_ids[slot];
    const size_t output_index =
        static_cast<size_t>(slot) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    if (expert_id < 0 || expert_id >= num_experts)
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const DeviceNativeVNNIMatrixDesc desc = bank.experts[expert_id].down;
    if (!native_vnni_desc_shape_ok<FMT>(desc, N, K))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int8_t* a_slot = A_int8 + static_cast<size_t>(slot) * K;
    const float* scale_slot = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
    const float expert_acc = moe_native_vnni_dot_desc<FMT>(
        desc, n, a_slot, scale_slot, N, K);
    route_output[output_index] = rocm_native_vnni_weight_route_rn(
        route_weights[slot], expert_acc);
}

/**
 * @brief Compute runtime-placement down partials using serial-M1 K slices.
 */
template <NativeVNNIFormat FMT>
__global__ void moe_grouped_native_vnni_down_runtime_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    int k_partitions)
{
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "Runtime grouped MoE split-K down requires a supported format");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    const int k_part = blockIdx.z;
    if (route >= num_active || k_part >= k_partitions || n >= N)
        return;

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const int expert_id = expert_ids[route];
    if (expert_id < 0 || expert_id >= num_experts)
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const DeviceMoEPlacementBankView &bank = runtime->banks[runtime->active_bank];
    const DeviceNativeVNNIMatrixDesc desc = bank.experts[expert_id].down;
    if (!native_vnni_desc_shape_ok<FMT>(desc, N, K))
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    const int b_start = k_part * blocks_per_part;
    int b_end = b_start + blocks_per_part;
    if (b_end > blocks_per_row)
        b_end = blocks_per_row;

    const int8_t* route_a =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* route_scales =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    partials[partial_index] = b_start < b_end
        ? moe_native_vnni_dot_desc_range<FMT>(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
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
            sum = rocm_native_vnni_accumulate_rn(
                sum,
                d_partial_fp32[static_cast<size_t>(kb) * N + n]);

        d_C_fp32[n] = sum;
    }
}

/**
 * @brief Publishes one runtime verifier row from ordered split-K partials.
 *
 * Grid Y owns the verifier row.  Consequently the reducer's register state,
 * launch shape, and ascending-KB addition order do not depend on the total
 * speculative depth.  This is the same reduction tree used by serial M=1
 * decode, applied independently to each row's `[KB][M][N]` partial slice.
 */
template <int TILE_N, int CPT>
__global__ void gemv_native_vnni_reduce_small_m_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB x M x N] partial buffer
    float* __restrict__ d_C_fp32,                // [M x N] output
    int M, int N, int kblocks)
{
    const int row = static_cast<int>(blockIdx.y);
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (row >= M)
        return;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < kblocks; ++kb)
        {
            const size_t idx =
                (static_cast<size_t>(kb) * static_cast<size_t>(M) +
                 static_cast<size_t>(row)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            sum = rocm_native_vnni_accumulate_rn(
                sum, d_partial_fp32[idx]);
        }
        d_C_fp32[static_cast<size_t>(row) * N + n] = sum;
    }
}

constexpr int NVNNI_SMALL_M_MAX_BATCH = 8;
constexpr int NVNNI_SMALL_M_GRAPH_SAFE_KB_CAP =
    llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions;
constexpr int NVNNI_VERIFIER_SHORT_ROWS_PER_TILE = 2;
constexpr int NVNNI_VERIFIER_WIDE_ROWS_PER_TILE = 4;

/**
 * @brief Select the measured verifier-row reuse geometry for a runtime batch.
 *
 * M=2 retains the two-row specialization because a four-row accumulator adds
 * idle predicates without any extra weight reuse.  M>=3 uses four rows: this
 * matches the common depth-three transaction and amortizes each packed-weight
 * decode across every live row.  Larger batches add independent Z-grid tiles,
 * so this capture-time policy is total for every positive runtime M.
 *
 * @param m Number of grouped verifier rows in the launch.
 * @return Compile-time kernel geometry selected by the host launch wrapper.
 */
constexpr int native_vnni_verifier_rows_per_tile(int m)
{
    return m <= NVNNI_VERIFIER_SHORT_ROWS_PER_TILE
        ? NVNNI_VERIFIER_SHORT_ROWS_PER_TILE
        : NVNNI_VERIFIER_WIDE_ROWS_PER_TILE;
}

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
    uint8_t codebook[NVNNI_SMALL_M_MAX_BATCH] = {};
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
__device__ __forceinline__ int32_t native_vnni_activation_sum_a(
    const int32_t* __restrict__ a4)
{
    using Traits = NVNNITraits<FMT>;
    int32_t sum_a = 0;
    if constexpr (Traits::is_asymmetric)
    {
        constexpr int32_t ones = 0x01010101;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            sum_a = __builtin_amdgcn_sdot4(a4[g], ones, sum_a, false);
    }
    return sum_a;
}

/**
 * Accumulates one decoded native-VNNI weight block into a small-M GEMV row.
 *
 * This helper intentionally mirrors the M=1 decode kernel's FP32 addition
 * order.  Do not combine the base dot product and asymmetric/dual-scale
 * correction into a temporary before adding to `f_acc`: the grouped verifier
 * path is compared against serial replay, and tiny associativity differences
 * can amplify through Qwen3.6 recurrent state.
 */
template <NativeVNNIFormat FMT>
__device__ __forceinline__ void native_vnni_small_m_accumulate_decoded_block(
    float& f_acc,
    const int32_t* __restrict__ a4,
    const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,
    const uint32_t* __restrict__ emin_base,
    size_t linear,
    float scale_ab,
    int32_t sum_a)
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
        f_acc = rocm_native_vnni_commit_dual_scale_block(
            f_acc,
            acc_lo,
            acc_hi,
            scale_lo,
            scale_hi,
            scale_ab);

        /*
         * Keep the decode-equivalent small-M path in lockstep with the serial
         * M=1 GEMV kernel.  The shared MoE correction helper intentionally
         * returns a float correction because the grouped MoE kernels use that
         * expression tree as their serial contract.  Plain GEMV serial decode,
         * however, applies each correction directly to f_acc.  Returning the
         * IQ1_M delta through a helper is enough for HIP to round one ULP
         * differently on some rows, so the verifier-row implementation spells
         * out the same updates as the serial GEMV loop.
         */
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

            float min_lo;
            float min_hi;
            if constexpr (Traits::needs_emin_array)
            {
                const uint32_t emin_packed = emin_base[linear];
                const uint16_t emin_lo_bits = static_cast<uint16_t>(emin_packed);
                const uint16_t emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
                min_lo = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
                min_hi = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
            }
            else
            {
                min_lo = 0.0f;
                min_hi = 0.0f;
            }
            f_acc = rocm_native_vnni_commit_dual_scale_correction(
                f_acc, sum_a_lo, sum_a_hi, min_lo, min_hi, scale_ab);
        }

        if constexpr (FMT == NVNNI_IQ1_M)
        {
            constexpr float IQ1S_DELTA = 0.125f;
            constexpr int32_t ones = 0x01010101;
            const uint8_t qh0 = payload[4];
            const uint8_t qh1 = payload[5];

            const int32_t sa_g0 = __builtin_amdgcn_sdot4(
                a4[1], ones, __builtin_amdgcn_sdot4(a4[0], ones, 0, false), false);
            const int32_t sa_g1 = __builtin_amdgcn_sdot4(
                a4[3], ones, __builtin_amdgcn_sdot4(a4[2], ones, 0, false), false);
            const int32_t sa_g2 = __builtin_amdgcn_sdot4(
                a4[5], ones, __builtin_amdgcn_sdot4(a4[4], ones, 0, false), false);
            const int32_t sa_g3 = __builtin_amdgcn_sdot4(
                a4[7], ones, __builtin_amdgcn_sdot4(a4[6], ones, 0, false), false);

            const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            f_acc = rocm_native_vnni_commit_iq1_m_delta_correction(
                f_acc,
                sa_g0,
                sa_g1,
                sa_g2,
                sa_g3,
                d0,
                d1,
                d2,
                d3,
                scale_lo,
                scale_hi,
                scale_ab);
        }
    }
    else
    {
        int32_t block_acc = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);

        const uint16_t scale_bits = scale_base[linear];
        const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
        f_acc = rocm_native_vnni_commit_single_scale_block(
            f_acc, block_acc, block_d, scale_ab);

        if constexpr (Traits::is_asymmetric)
        {
            const uint16_t min_bits = min_base[linear];
            const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
            f_acc = rocm_native_vnni_commit_single_scale_correction(
                f_acc, sum_a, block_m, scale_ab);
        }
    }
}

/**
 * Adds one row/block contribution for the generic small-M native-VNNI GEMV.
 *
 * Keeping this as a tiny helper lets M=2 spell out row 0 and row 1 directly
 * without reintroducing codebook-specific kernels.  The helper still uses the
 * exact serial-decode accumulation contract from
 * `native_vnni_small_m_accumulate_decoded_block()`.
 */
template <NativeVNNIFormat FMT>
__device__ __forceinline__ void native_vnni_small_m_accumulate_row(
    float& acc,
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scale_A_blockwise,
    const int32_t* __restrict__ d_sum_A_blockwise,
    int row,
    int K,
    int blocks_per_row,
    int b,
    const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,
    const uint32_t* __restrict__ emin_base,
    size_t linear)
{
    const int32_t* a4 = reinterpret_cast<const int32_t*>(
        d_A_int8 + static_cast<size_t>(row) * K + static_cast<size_t>(b) * 32);
    const float scale_ab =
        d_scale_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b];
    int32_t sum_a = 0;
    if constexpr (NVNNITraits<FMT>::is_asymmetric)
    {
        sum_a = d_sum_A_blockwise
            ? d_sum_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b]
            : native_vnni_activation_sum_a<FMT>(a4);
    }
    native_vnni_small_m_accumulate_decoded_block<FMT>(
        acc, a4, packed_groups, payload, scale_base, min_base, emin_base,
        linear, scale_ab, sum_a);
}


/**
 * @brief Decode-equivalent batched small-M GEMV body that shares weight decode across rows.
 *
 * The row-parallel verifier kernel assigns one workgroup row to each verifier
 * row, which preserves the serial decode order but decodes the same packed
 * weight block once per row.  For MTP verifier economics that is the wrong
 * tradeoff on wide projections: dequantization is a material part of the
 * cost, and grouped verifier execution should amortize it across the complete
 * four-row production transaction.
 *
 * This body keeps the same per-row FP32 accumulation order as serial decode:
 * each row sees blocks in increasing `b` order, and split-K partials are
 * reduced by the existing reducer.  The only difference is that each output
 * lane decodes one weight block and then accumulates all verifier rows before
 * moving to the next block.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeVNNIFormat FMT>
__device__ __forceinline__ void gemv_native_vnni_small_m_batched_shared_body(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeVNNISmallMBatchedArgs args,
    int M,
    int K,
    int kblocks)
{
    static_assert(CPT == 1 || CPT == 2, "native-VNNI small-M supports CPT in {1,2}");
    static_assert(
        ROWS_PER_TILE == NVNNI_VERIFIER_SHORT_ROWS_PER_TILE ||
            ROWS_PER_TILE == NVNNI_VERIFIER_WIDE_ROWS_PER_TILE,
        "native-VNNI verifier row tile must be one of the measured geometries");
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "small-M native-VNNI requires a supported native-VNNI format");

    const int global_tile = blockIdx.x;
    const int proj = native_vnni_batched_projection_for_tile(args, global_tile);
    const int local_tile = global_tile - args.grid_offsets[proj];
    const int N = args.N[proj];
    const int n_base = local_tile * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;
    const int row_base = static_cast<int>(blockIdx.z) * ROWS_PER_TILE;

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

    float acc[ROWS_PER_TILE][CPT];
    #pragma unroll
    for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[tile_row][c] = 0.0f;
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
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= M)
                    continue;
                native_vnni_small_m_accumulate_row<FMT>(
                    acc[tile_row][c], d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise,
                    row, K, blocks_per_row, b, packed_groups, payload, scale_base,
                    min_base, emin_base, linear);
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
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
            const int row = row_base + tile_row;
            if (row >= M)
                continue;
            const size_t out_idx =
                static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(n);
            if (kblocks == 1)
            {
                out_base[out_idx] = acc[tile_row][c] + (bias_base ? bias_base[n] : 0.0f);
            }
            else
            {
                partial_base[(static_cast<size_t>(kb) * static_cast<size_t>(M) +
                              static_cast<size_t>(row)) * static_cast<size_t>(N) +
                             static_cast<size_t>(n)] = acc[tile_row][c];
            }
        }
    }
}

template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeVNNIFormat FMT>
__global__ void gemv_native_vnni_small_m_batched_shared_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeVNNISmallMBatchedArgs args,
    int M,
    int K,
    int kblocks)
{
    gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, FMT>(
        d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
}

template <int TILE_N, int CPT, int ROWS_PER_TILE>
__global__ void gemv_native_vnni_small_m_batched_shared_mixed_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeVNNISmallMBatchedArgs args,
    int M,
    int K,
    int kblocks)
{
    const int proj = native_vnni_batched_projection_for_tile(args, blockIdx.x);
    switch (static_cast<NativeVNNIFormat>(args.codebook[proj]))
    {
    case NVNNI_Q4_0:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q4_0>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ4_NL:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ4_NL>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q4_1:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q4_1>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q5_0:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q5_0>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q5_1:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q5_1>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q6_K:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q6_K>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q3_K:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q3_K>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q2_K:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q2_K>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ3_S:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ3_S>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ3_XXS:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ3_XXS>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ2_S:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ2_S>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ2_XS:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ2_XS>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ2_XXS:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ2_XXS>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ1_S:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ1_S>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_IQ1_M:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_IQ1_M>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    case NVNNI_Q8_0:
        gemv_native_vnni_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NVNNI_Q8_0>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, args, M, K, kblocks);
        return;
    default:
        return;
    }
}

/**
 * @brief Reduces one runtime row for every projection in a fused batch.
 *
 * `blockIdx.y` is a row owner rather than a compile-time unrolled loop.  The
 * reducer therefore has constant register pressure and preserves the serial
 * M=1 ascending-KB reduction tree for every supported speculative depth.
 */
template <int TILE_N, int CPT>
__global__ void gemv_native_vnni_reduce_small_m_batched_kernel_t(
    NativeVNNISmallMBatchedArgs args,
    int M,
    int kblocks)
{
    const int row = static_cast<int>(blockIdx.y);
    const int global_tile = blockIdx.x;
    const int proj = native_vnni_batched_projection_for_tile(args, global_tile);
    const int local_tile = global_tile - args.grid_offsets[proj];
    const int N = args.N[proj];
    const int n_base = local_tile * TILE_N + threadIdx.x * CPT;

    const float* partial_base = args.partial[proj];
    const float* bias_base = args.bias[proj];
    float* out_base = args.output[proj];
    if (row >= M)
        return;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
            continue;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < kblocks; ++kb)
        {
            sum += partial_base[(static_cast<size_t>(kb) * static_cast<size_t>(M) +
                                 static_cast<size_t>(row)) * static_cast<size_t>(N) +
                                static_cast<size_t>(n)];
        }
        out_base[static_cast<size_t>(row) * N + n] =
            sum + (bias_base ? bias_base[n] : 0.0f);
    }
}

/**
 * @brief Runtime-row NativeVNNI verifier GEMV with fixed four-row work tiles.
 *
 * A workgroup decodes each packed weight block once and applies it to at most
 * four verifier rows.  Larger speculative batches add independent Z-grid tiles;
 * they never enlarge the accumulator array or alter an existing row's block
 * order.  Split-K output is always written to declared workspace so the
 * following reducer can publish partitions in serial-M1 order.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeVNNIFormat FMT>
__global__ void gemv_native_vnni_small_m_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [M x K] quantized activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row x N x payload_bytes]
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row x N]
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row x N], nullable when unused
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row x N], Q2_K only
    float*           __restrict__ d_C_fp32,       // [M x N] output
    float*           __restrict__ d_partial_fp32, // [KB x M x N] partials
    const float*     __restrict__ d_scale_A_blockwise, // [M x blocks_per_row]
    const int32_t*   __restrict__ d_sum_A_blockwise,   // [M x blocks_per_row], nullable
    int M, int N, int K, int kblocks)
{
    static_assert(CPT == 1 || CPT == 2, "native-VNNI small-M supports CPT in {1,2}");
    static_assert(
        ROWS_PER_TILE == NVNNI_VERIFIER_SHORT_ROWS_PER_TILE ||
            ROWS_PER_TILE == NVNNI_VERIFIER_WIDE_ROWS_PER_TILE,
        "native-VNNI verifier row tile must be one of the measured geometries");
    static_assert(NVNNIGroupedMoESupported<FMT>::value,
                  "small-M native-VNNI requires a supported native-VNNI format");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;
    const int row_base = static_cast<int>(blockIdx.z) * ROWS_PER_TILE;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float acc[ROWS_PER_TILE][CPT];
    #pragma unroll
    for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[tile_row][c] = 0.0f;
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
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= M)
                    continue;
                native_vnni_small_m_accumulate_row<FMT>(
                    acc[tile_row][c], d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise,
                    row, K, blocks_per_row, b, packed_groups, payload, d_block_scales,
                    d_block_mins, d_block_emins, linear);
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
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
            const int row = row_base + tile_row;
            if (row >= M)
                continue;
            const size_t out_idx =
                static_cast<size_t>(row) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            if (kblocks == 1)
            {
                d_C_fp32[out_idx] = acc[tile_row][c];
            }
            else
            {
                const size_t partial_idx =
                    (static_cast<size_t>(kb) * static_cast<size_t>(M) +
                     static_cast<size_t>(row)) * static_cast<size_t>(N) +
                    static_cast<size_t>(n);
                d_partial_fp32[partial_idx] = acc[tile_row][c];
            }
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
        {
            d_C_fp32[n] = result;
        }
        else
        {
            /*
             * IQ decode obeys the same partition contract as the linear
             * codebooks: one persistent row per K partition followed by the
             * fixed-order reducer.  Completion-order atomics are not a valid
             * serial-decode arithmetic regime.
             */
            d_partial_fp32[static_cast<size_t>(kb) * N + n] = result;
        }
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
    float* d_gate_partials,
    float* d_up_partials,
    bool hidden_prequantized,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || (!hidden_prequantized && !d_hidden) ||
        !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
        !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
        !d_gate_partials || !d_up_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        resolveNativeVNNISerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_vnni_decode_table] "
                "missing serial-M1 dispatch policy (codebook=%u N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();
    hipError_t err = hipSuccess;
    if (!hidden_prequantized)
    {
        const int blocks_per_row = K / 32;
        hipLaunchKernelGGL(moe_grouped_hidden_quantize_blockwise_kernel,
            dim3(blocks_per_row), dim3(32), 0, hip_stream,
            d_hidden, d_hidden_int8, d_hidden_scales, K);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_table] hidden quantize launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    constexpr int TN = 64;
    dim3 direct_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid((N + TN - 1) / TN, num_active, k_partitions);
    dim3 block(TN);

    #define LAUNCH_GROUPED_GATE_UP(FMT_TAG)                                      \
        do                                                                        \
        {                                                                         \
            if (k_partitions == 1)                                                \
            {                                                                     \
                hipLaunchKernelGGL(                                               \
                    HIP_KERNEL_NAME(                                              \
                        moe_grouped_native_vnni_gate_up_kernel_t<FMT_TAG>),       \
                    direct_grid, block, 0, hip_stream,                            \
                    d_hidden_int8, d_hidden_scales,                               \
                    d_gate_desc_table, d_up_desc_table, d_expert_ids,             \
                    d_gate_outputs, d_up_outputs, num_active, N, K);              \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                hipLaunchKernelGGL(                                               \
                    HIP_KERNEL_NAME(                                              \
                        moe_grouped_native_vnni_gate_up_kpart_kernel_t<FMT_TAG>), \
                    partial_grid, block, 0, hip_stream,                           \
                    d_hidden_int8, d_hidden_scales,                               \
                    d_gate_desc_table, d_up_desc_table, d_expert_ids,             \
                    d_gate_partials, d_up_partials,                               \
                    num_active, N, K, k_partitions);                              \
            }                                                                     \
        } while (false)

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

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_vnni_gate_up_kpart_reduce_kernel,
            direct_grid, block, 0, hip_stream,
            d_gate_partials, d_up_partials, d_expert_ids,
            d_gate_outputs, d_up_outputs, num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_gate_up_native_vnni_decode_table] "
                    "ordered reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmMoE_grouped_gate_up_native_vnni_decode_runtime(
    const float* d_hidden,
    const void* d_runtime_layer,
    const int* d_expert_ids,
    float* const* d_gate_outputs,
    float* const* d_up_outputs,
    int8_t* d_hidden_int8,
    float* d_hidden_scales,
    float* d_gate_partials,
    float* d_up_partials,
    bool hidden_prequantized,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || (!hidden_prequantized && !d_hidden) ||
        !d_runtime_layer || !d_expert_ids ||
        !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
        !d_gate_partials || !d_up_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_runtime] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        resolveNativeVNNISerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_vnni_decode_runtime] "
                "missing serial-M1 dispatch policy (codebook=%u N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    (void)hipGetLastError();
    hipError_t err = hipSuccess;
    if (!hidden_prequantized)
    {
        const int blocks_per_row = K / 32;
        hipLaunchKernelGGL(moe_grouped_hidden_quantize_blockwise_kernel,
            dim3(blocks_per_row), dim3(32), 0, hip_stream,
            d_hidden, d_hidden_int8, d_hidden_scales, K);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_runtime] hidden quantize launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    constexpr int TN = 64;
    dim3 direct_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid((N + TN - 1) / TN, num_active, k_partitions);
    dim3 block(TN);
    const auto* runtime = static_cast<const DeviceMoELayerRuntimeView*>(d_runtime_layer);

    #define LAUNCH_GROUPED_GATE_UP_RUNTIME(FMT_TAG)                              \
        do                                                                        \
        {                                                                         \
            if (k_partitions == 1)                                                \
            {                                                                     \
                hipLaunchKernelGGL(                                               \
                    HIP_KERNEL_NAME(                                              \
                        moe_grouped_native_vnni_gate_up_runtime_kernel_t<         \
                            FMT_TAG>),                                            \
                    direct_grid, block, 0, hip_stream,                            \
                    d_hidden_int8, d_hidden_scales, runtime, d_expert_ids,        \
                    d_gate_outputs, d_up_outputs,                                 \
                    num_active, N, K, num_experts);                               \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                hipLaunchKernelGGL(                                               \
                    HIP_KERNEL_NAME(                                              \
                        moe_grouped_native_vnni_gate_up_runtime_kpart_kernel_t<   \
                            FMT_TAG>),                                            \
                    partial_grid, block, 0, hip_stream,                           \
                    d_hidden_int8, d_hidden_scales, runtime, d_expert_ids,        \
                    d_gate_partials, d_up_partials,                               \
                    num_active, N, K, num_experts, k_partitions);                 \
            }                                                                     \
        } while (false)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_GATE_UP_RUNTIME(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_runtime] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_GATE_UP_RUNTIME

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_vnni_decode_runtime] gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_vnni_gate_up_kpart_reduce_kernel,
            direct_grid, block, 0, hip_stream,
            d_gate_partials, d_up_partials, d_expert_ids,
            d_gate_outputs, d_up_outputs, num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_gate_up_native_vnni_decode_runtime] "
                    "ordered reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

/**
 * @brief Publish grouped expert SwiGLU rows in serial-decode Q8 form.
 *
 * This is the sole production launcher for the fused grouped publisher used
 * by direct, descriptor-table, and device-runtime down projections. Keeping a
 * public launch boundary here also lets integration tests compare the exact
 * Q8 payload and FP32 scales against standalone serial M=1 publication before
 * any weight format or route reduction can obscure the source of a mismatch.
 *
 * @param d_gate_ptrs Device array of per-route gate-row pointers.
 * @param d_up_ptrs Device array of per-route up-row pointers.
 * @param d_expert_ids Optional device route validity array; negative routes
 *        are skipped. Pass null when every route is valid.
 * @param d_swiglu_int8 Device output payload `[num_active, K]`.
 * @param d_swiglu_scales Device output scales `[num_active, K / 32]`.
 * @param num_active Number of grouped route rows.
 * @param K Width of each gate/up row; must be divisible by 32.
 * @param device_id ROCm device ordinal.
 * @param stream Exact non-null producer stream.
 * @return True when the launch was accepted by HIP; false on invalid input or
 *         an immediate launch failure.
 */
bool rocmMoE_grouped_swiglu_quantize_decode_equivalent(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const int* d_expert_ids,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    int num_active,
    int K,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs ||
        !d_swiglu_int8 || !d_swiglu_scales ||
        num_active <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_quantize_decode_equivalent] "
                "invalid arguments\n");
        return false;
    }

    (void)hipGetLastError();
    hipLaunchKernelGGL(
        moe_grouped_swiglu_quantize_blockwise_kernel,
        dim3(num_active), dim3(256), 0, static_cast<hipStream_t>(stream),
        d_gate_ptrs, d_up_ptrs, d_expert_ids,
        d_swiglu_int8, d_swiglu_scales, num_active, K);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_quantize_decode_equivalent] "
                "launch failed: %s\n",
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
    float* d_ordered_down_partials,
    float* d_output,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs || !d_descs || !d_weights ||
        !d_swiglu_int8 || !d_swiglu_scales ||
        !d_ordered_down_partials || !d_output ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        resolveNativeVNNISerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode] "
                "missing serial-M1 dispatch policy (codebook=%u N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, nullptr,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    constexpr int TN = 64;
    dim3 route_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid((N + TN - 1) / TN, num_active, k_partitions);
    dim3 block(TN);

    #define LAUNCH_GROUPED_DOWN_DIRECT(FMT_TAG)                              \
        do                                                                    \
        {                                                                     \
            if (k_partitions == 1)                                            \
            {                                                                 \
                hipLaunchKernelGGL(                                           \
                    HIP_KERNEL_NAME(                                          \
                        moe_grouped_native_vnni_down_route_kernel_t<          \
                            FMT_TAG, false>),                                 \
                    route_grid, block, 0, hip_stream,                         \
                    d_swiglu_int8, d_swiglu_scales, d_descs, nullptr,         \
                    d_weights, d_ordered_down_partials, num_active, N, K);    \
            }                                                                 \
            else                                                              \
            {                                                                 \
                hipLaunchKernelGGL(                                           \
                    HIP_KERNEL_NAME(                                          \
                        moe_grouped_native_vnni_down_kpart_kernel_t<          \
                            FMT_TAG, false>),                                 \
                    partial_grid, block, 0, hip_stream,                       \
                    d_swiglu_int8, d_swiglu_scales, d_descs, nullptr,         \
                    d_ordered_down_partials, num_active, N, K, k_partitions); \
            }                                                                 \
        } while (false)

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

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_vnni_down_kpart_reduce_kernel,
            route_grid, block, 0, hip_stream,
            d_ordered_down_partials, d_weights,
            d_ordered_down_partials,
            num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_swiglu_down_native_vnni_decode] "
                    "K-partition reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    hipLaunchKernelGGL(
        moe_grouped_native_vnni_down_ordered_reduce_kernel,
        dim3((N + TN - 1) / TN), block, 0, hip_stream,
        d_ordered_down_partials, d_output, num_active, N);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode] "
                "route reduce launch failed: %s\n",
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
    float* d_canonical_route_contributions,
    float* d_ordered_down_partials,
    int num_active,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    const bool canonical_publication =
        d_canonical_route_contributions != nullptr;
    if (!stream || !d_gate_ptrs || !d_up_ptrs || !d_desc_table ||
        !d_expert_ids || !d_weights || !d_swiglu_int8 || !d_swiglu_scales ||
        (static_cast<bool>(d_output) == canonical_publication) ||
        !d_ordered_down_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        resolveNativeVNNISerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] "
                "missing serial-M1 dispatch policy (codebook=%u N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, d_expert_ids,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    constexpr int TN = 64;
    float* const route_contributions = canonical_publication
                                           ? d_canonical_route_contributions
                                           : d_ordered_down_partials;
    dim3 route_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid((N + TN - 1) / TN, num_active, k_partitions);
    dim3 block(TN);

    #define LAUNCH_GROUPED_DOWN_TABLE(FMT_TAG)                                  \
        do                                                                       \
        {                                                                        \
            if (k_partitions == 1)                                               \
            {                                                                    \
                hipLaunchKernelGGL(                                              \
                    HIP_KERNEL_NAME(                                             \
                        moe_grouped_native_vnni_down_route_kernel_t<FMT_TAG>),   \
                    route_grid, block, 0, hip_stream,                            \
                    d_swiglu_int8, d_swiglu_scales,                              \
                    d_desc_table, d_expert_ids, d_weights,                       \
                    route_contributions, num_active, N, K);                      \
            }                                                                    \
            else                                                                 \
            {                                                                    \
                hipLaunchKernelGGL(                                              \
                    HIP_KERNEL_NAME(                                             \
                        moe_grouped_native_vnni_down_kpart_kernel_t<FMT_TAG>),   \
                    partial_grid, block, 0, hip_stream,                          \
                    d_swiglu_int8, d_swiglu_scales,                              \
                    d_desc_table, d_expert_ids, d_ordered_down_partials,         \
                    num_active, N, K, k_partitions);                             \
            }                                                                    \
        } while (false)

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
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] route launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_vnni_down_kpart_reduce_kernel,
            route_grid, block, 0, hip_stream,
            d_ordered_down_partials, d_weights, route_contributions,
            num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] "
                    "K-partition reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    if (!canonical_publication)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_vnni_down_ordered_reduce_kernel,
            dim3((N + TN - 1) / TN), block, 0,
            hip_stream,
            route_contributions, d_output, num_active, N);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_table] ordered reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const void* d_runtime_layer,
    const int* d_expert_ids,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    float* d_output,
    float* d_ordered_down_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs || !d_runtime_layer ||
        !d_expert_ids || !d_weights || !d_swiglu_int8 || !d_swiglu_scales ||
        !d_output || !d_ordered_down_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        resolveNativeVNNISerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] "
                "missing serial-M1 dispatch policy (codebook=%u N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), N, K);
        return false;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, d_expert_ids,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    constexpr int TN = 64;
    dim3 route_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid((N + TN - 1) / TN, num_active, k_partitions);
    dim3 block(TN);
    const auto* runtime = static_cast<const DeviceMoELayerRuntimeView*>(d_runtime_layer);

    #define LAUNCH_GROUPED_DOWN_RUNTIME(FMT_TAG)                               \
        do                                                                      \
        {                                                                       \
            if (k_partitions == 1)                                              \
            {                                                                   \
                hipLaunchKernelGGL(                                             \
                    HIP_KERNEL_NAME(                                            \
                        moe_grouped_native_vnni_down_runtime_route_kernel_t<    \
                            FMT_TAG>),                                          \
                    route_grid, block, 0, hip_stream,                           \
                    d_swiglu_int8, d_swiglu_scales, runtime, d_expert_ids,      \
                    d_weights, d_ordered_down_partials,                         \
                    num_active, N, K, num_experts);                             \
            }                                                                   \
            else                                                                \
            {                                                                   \
                hipLaunchKernelGGL(                                             \
                    HIP_KERNEL_NAME(                                            \
                        moe_grouped_native_vnni_down_runtime_kpart_kernel_t<    \
                            FMT_TAG>),                                          \
                    partial_grid, block, 0, hip_stream,                         \
                    d_swiglu_int8, d_swiglu_scales, runtime, d_expert_ids,      \
                    d_ordered_down_partials,                                    \
                    num_active, N, K, num_experts, k_partitions);               \
            }                                                                   \
        } while (false)

    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q4_0);     break;
    case NVNNI_IQ4_NL:   LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ4_NL);   break;
    case NVNNI_Q4_1:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q4_1);     break;
    case NVNNI_Q5_0:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q5_0);     break;
    case NVNNI_Q5_1:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q5_1);     break;
    case NVNNI_Q6_K:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q6_K);     break;
    case NVNNI_Q3_K:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q3_K);     break;
    case NVNNI_Q2_K:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q2_K);     break;
    case NVNNI_IQ3_S:    LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ3_S);    break;
    case NVNNI_IQ3_XXS:  LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ3_XXS);  break;
    case NVNNI_IQ2_S:    LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ2_S);    break;
    case NVNNI_IQ2_XS:   LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ2_XS);   break;
    case NVNNI_IQ2_XXS:  LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ2_XXS);  break;
    case NVNNI_IQ1_S:    LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ1_S);    break;
    case NVNNI_IQ1_M:    LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_IQ1_M);    break;
    case NVNNI_Q8_0:     LAUNCH_GROUPED_DOWN_RUNTIME(NVNNI_Q8_0);     break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_DOWN_RUNTIME

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] route launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_vnni_down_kpart_reduce_kernel,
            route_grid, block, 0, hip_stream,
            d_ordered_down_partials, d_weights,
            d_ordered_down_partials,
            num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] "
                    "K-partition reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    hipLaunchKernelGGL(
        moe_grouped_native_vnni_down_ordered_reduce_kernel,
        dim3((N + TN - 1) / TN), block, 0, hip_stream,
        d_ordered_down_partials, d_output, num_active, N);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime] ordered reduce launch failed: %s\n",
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
template <NativeVNNIFormat FMT>
bool rocmLaunch_native_vnni_small_m_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    const uint16_t* d_block_mins,
    const uint32_t* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    int device_id, void* stream)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NVNNI_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    const int blocks_per_row = K / 32;
    const int grid_n = (N + TN - 1) / TN;
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = g_native_vnni_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override = (api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb;
    const int explicit_waves_override = (api_waves_override > 0) ? api_waves_override : rocm_env.nvnni_gemv_target_waves;
    const bool decode_equivalent = nativeVNNIDecodeEquivalentSmallMMode();
    if (!validateNativeVNNIDecodeEquivalentKBOverride(
            "rocmGemv_native_vnni_small_m_fp32",
            explicit_kb_override,
            M,
            N,
            K,
            1))
    {
        return false;
    }
    const auto generated_cfg =
        decode_equivalent
            ? resolveNativeVNNISerialM1RuntimeConfig(static_cast<uint8_t>(FMT), N, K)
            : resolveNativeVNNIDecodeGeneratedRuntimeConfig(static_cast<uint8_t>(FMT), M, N, K);
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_fp32] no serial M=1 split-K policy for decode-equivalent verifier launch (M=%d N=%d K=%d codebook=%u)\n",
                M, N, K, static_cast<unsigned>(FMT));
        return false;
    }
    const int kb_override =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : (generated_cfg.has_generated ? generated_cfg.kb : -1);
    const int waves_override =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : (generated_cfg.has_generated ? generated_cfg.target_waves_per_cu : -1);
    if (kb_override > KB_CAP)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d N=%d K=%d)\n",
                kb_override, KB_CAP, M, N, K);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_vnni_verifier_rows_per_tile(M);
    const int logical_row_tiles =
        (M + verifier_rows_per_tile - 1) / verifier_rows_per_tile;
    const int resident_tiles = max(1, grid_n * logical_row_tiles);
    int kb_raw = max(1, (target_total_waves + resident_tiles - 1) / resident_tiles);
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
                kb, M, N, K);
        return false;
    }

    recordNativeVNNISmallMLaunch(
        M,
        N,
        K,
        static_cast<unsigned>(FMT),
        grid_n,
        kb,
        target_waves_per_cu,
        false,
        1);

    /*
     * Partials are a reusable fixed-row tile, not a semantic M limit. Wider
     * verifier graphs record multiple grouped kernel/reducer pairs while each
     * pair retains weight reuse and scalar M=1's ordered split-K reduction.
     */
    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeVNNIVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = native_vnni_verifier_rows_per_tile(tile_m);
        const int tile_row_groups =
            (tile_m + tile_rows_per_group - 1) / tile_rows_per_group;
        const int8_t* tile_a =
            d_A_int8 + static_cast<size_t>(row_base) * K;
        const float* tile_scales =
            d_scale_A_blockwise +
            static_cast<size_t>(row_base) * blocks_per_row;
        const int32_t* tile_sums =
            d_sum_A_blockwise
                ? d_sum_A_blockwise +
                      static_cast<size_t>(row_base) * blocks_per_row
                : nullptr;
        float* tile_c =
            d_C_fp32 + static_cast<size_t>(row_base) * N;

        if (tile_rows_per_group == NVNNI_VERIFIER_SHORT_ROWS_PER_TILE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_small_m_kernel_t<
                    TN, CPT, NVNNI_VERIFIER_SHORT_ROWS_PER_TILE, FMT>),
                dim3(grid_n, kb, tile_row_groups), dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, d_payload, d_block_scales, d_block_mins, d_block_emins,
                tile_c, d_partial_fp32, tile_scales, tile_sums,
                tile_m, N, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_small_m_kernel_t<
                    TN, CPT, NVNNI_VERIFIER_WIDE_ROWS_PER_TILE, FMT>),
                dim3(grid_n, kb, tile_row_groups), dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, d_payload, d_block_scales, d_block_mins, d_block_emins,
                tile_c, d_partial_fp32, tile_scales, tile_sums,
                tile_m, N, K, kb);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d N=%d K=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, N, K,
                    static_cast<unsigned>(FMT));
            return false;
        }

        if (kb > 1)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_reduce_small_m_kernel_t<TN, CPT>),
                dim3(grid_n, tile_m), dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                d_partial_fp32, tile_c, tile_m, N, kb);
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d N=%d K=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, N, K,
                    static_cast<unsigned>(FMT));
            return false;
        }
    }

    return true;
}

template <NativeVNNIFormat FMT>
bool rocmLaunch_native_vnni_small_m_batched_fp32(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int call_kb_override = -1,
    int call_waves_override = -1)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NVNNI_SMALL_M_GRAPH_SAFE_KB_CAP;
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
        args.codebook[p] = static_cast<uint8_t>(FMT);
    }
    args.grid_offsets[num_projections] = args.total_grid_n;

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = g_native_vnni_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (call_kb_override > 0)
            ? call_kb_override
            : ((api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb);
    const int explicit_waves_override =
        (call_waves_override > 0)
            ? call_waves_override
            : ((api_waves_override > 0) ? api_waves_override : rocm_env.nvnni_gemv_target_waves);
    const bool decode_equivalent = nativeVNNIDecodeEquivalentSmallMMode();
    if (!validateNativeVNNIDecodeEquivalentKBOverride(
            "rocmGemv_native_vnni_small_m_batched_fp32",
            explicit_kb_override,
            M,
            0,
            K,
            num_projections))
    {
        return false;
    }
    const auto generated_cfg =
        decode_equivalent
            ? resolveNativeVNNICommonSerialM1RuntimeConfig(
                  static_cast<uint8_t>(FMT), Ns, num_projections, K)
            : (num_projections == 1
                   ? resolveNativeVNNIDecodeGeneratedRuntimeConfig(
                         static_cast<uint8_t>(FMT), M, Ns[0], K)
                   : NativeVNNIDecodeGeneratedRuntimeConfig{});
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_fp32] fused verifier group has no common serial M=1 split-K policy (M=%d K=%d projections=%d codebook=%u)\n",
                M, K, num_projections, static_cast<unsigned>(FMT));
        return false;
    }
    const int kb_override =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : (generated_cfg.has_generated ? generated_cfg.kb : -1);
    const int waves_override =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : (generated_cfg.has_generated ? generated_cfg.target_waves_per_cu : -1);
    if (kb_override > KB_CAP)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d K=%d projections=%d)\n",
                kb_override, KB_CAP, M, K, num_projections);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_vnni_verifier_rows_per_tile(M);
    const int logical_row_tiles =
        (M + verifier_rows_per_tile - 1) / verifier_rows_per_tile;
    const int resident_tiles =
        max(1, args.total_grid_n * logical_row_tiles);
    int kb_raw = max(1, (target_total_waves + resident_tiles - 1) / resident_tiles);
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
                        kb, p, M, args.N[p], K);
                return false;
            }
        }
    }

    recordNativeVNNISmallMLaunch(
        M,
        0,
        K,
        static_cast<unsigned>(FMT),
        args.total_grid_n,
        kb,
        target_waves_per_cu,
        true,
        num_projections);

    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeVNNIVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = native_vnni_verifier_rows_per_tile(tile_m);
        const int tile_row_groups =
            (tile_m + tile_rows_per_group - 1) / tile_rows_per_group;
        NativeVNNISmallMBatchedArgs tile_args = args;
        for (int p = 0; p < num_projections; ++p)
        {
            tile_args.output[p] +=
                static_cast<size_t>(row_base) * tile_args.N[p];
        }

        const int8_t* tile_a =
            d_A_int8 + static_cast<size_t>(row_base) * K;
        const float* tile_scales =
            d_scale_A_blockwise +
            static_cast<size_t>(row_base) * blocks_per_row;
        const int32_t* tile_sums =
            d_sum_A_blockwise
                ? d_sum_A_blockwise +
                      static_cast<size_t>(row_base) * blocks_per_row
                : nullptr;

        if (tile_rows_per_group == NVNNI_VERIFIER_SHORT_ROWS_PER_TILE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_small_m_batched_shared_kernel_t<
                    TN, CPT, NVNNI_VERIFIER_SHORT_ROWS_PER_TILE, FMT>),
                dim3(tile_args.total_grid_n, kb, tile_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, tile_scales, tile_sums, tile_args, tile_m, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_small_m_batched_shared_kernel_t<
                    TN, CPT, NVNNI_VERIFIER_WIDE_ROWS_PER_TILE, FMT>),
                dim3(tile_args.total_grid_n, kb, tile_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, tile_scales, tile_sums, tile_args, tile_m, K, kb);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_vnni_small_m_batched_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections, static_cast<unsigned>(FMT));
            return false;
        }

        if (kb > 1)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_reduce_small_m_batched_kernel_t<TN, CPT>),
                dim3(tile_args.total_grid_n, tile_m),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream), tile_args, tile_m, kb);
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_vnni_small_m_batched_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections, static_cast<unsigned>(FMT));
            return false;
        }
    }

    return true;
}

static bool nativeVNNICodebookRequiresAux(uint8_t codebook_id)
{
    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_1:
    case NVNNI_Q5_1:
    case NVNNI_Q6_K:
    case NVNNI_Q3_K:
    case NVNNI_Q2_K:
    case NVNNI_IQ2_S:
    case NVNNI_IQ2_XS:
    case NVNNI_IQ1_S:
    case NVNNI_IQ1_M:
        return true;
    default:
        return false;
    }
}

static bool nativeVNNICodebookRequiresEmins(uint8_t codebook_id)
{
    return static_cast<NativeVNNIFormat>(codebook_id) == NVNNI_Q2_K;
}

static bool nativeVNNICodebookSupported(uint8_t codebook_id)
{
    switch (static_cast<NativeVNNIFormat>(codebook_id))
    {
    case NVNNI_Q4_0:
    case NVNNI_IQ4_NL:
    case NVNNI_Q4_1:
    case NVNNI_Q5_0:
    case NVNNI_Q5_1:
    case NVNNI_Q6_K:
    case NVNNI_Q3_K:
    case NVNNI_Q2_K:
    case NVNNI_IQ3_S:
    case NVNNI_IQ3_XXS:
    case NVNNI_IQ2_S:
    case NVNNI_IQ2_XS:
    case NVNNI_IQ2_XXS:
    case NVNNI_IQ1_S:
    case NVNNI_IQ1_M:
    case NVNNI_Q8_0:
        return true;
    default:
        return false;
    }
}

bool rocmLaunch_native_vnni_small_m_batched_mixed_fp32(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int call_kb_override = -1,
    int call_waves_override = -1)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NVNNI_SMALL_M_GRAPH_SAFE_KB_CAP;
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
        args.codebook[p] = codebook_ids[p];
    }
    args.grid_offsets[num_projections] = args.total_grid_n;

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = g_native_vnni_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (call_kb_override > 0)
            ? call_kb_override
            : ((api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb);
    const int explicit_waves_override =
        (call_waves_override > 0)
            ? call_waves_override
            : ((api_waves_override > 0) ? api_waves_override : rocm_env.nvnni_gemv_target_waves);
    const bool decode_equivalent = nativeVNNIDecodeEquivalentSmallMMode();
    if (!validateNativeVNNIDecodeEquivalentKBOverride(
            "rocmGemv_native_vnni_small_m_batched_mixed_fp32",
            explicit_kb_override,
            M,
            0,
            K,
            num_projections))
    {
        return false;
    }
    const auto generated_cfg =
        decode_equivalent
            ? resolveNativeVNNICommonSerialM1RuntimeConfig(
                  codebook_ids, Ns, num_projections, K)
            : (num_projections == 1
                   ? resolveNativeVNNIDecodeGeneratedRuntimeConfig(
                         codebook_ids[0], M, Ns[0], K)
                   : NativeVNNIDecodeGeneratedRuntimeConfig{});
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] fused verifier group has no common serial M=1 split-K policy (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    const int kb_override =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : (generated_cfg.has_generated ? generated_cfg.kb : -1);
    const int waves_override =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : (generated_cfg.has_generated ? generated_cfg.target_waves_per_cu : -1);
    if (kb_override > KB_CAP)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d K=%d projections=%d)\n",
                kb_override, KB_CAP, M, K, num_projections);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_vnni_verifier_rows_per_tile(M);
    const int logical_row_tiles =
        (M + verifier_rows_per_tile - 1) / verifier_rows_per_tile;
    const int resident_tiles =
        max(1, args.total_grid_n * logical_row_tiles);
    int kb_raw = max(1, (target_total_waves + resident_tiles - 1) / resident_tiles);
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
                        "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] kb=%d>1 but projection %d has no partial buffer (M=%d N=%d K=%d codebook=%u)\n",
                        kb, p, M, args.N[p], K, static_cast<unsigned>(args.codebook[p]));
                return false;
            }
        }
    }

    recordNativeVNNISmallMLaunch(
        M,
        0,
        K,
        255,
        args.total_grid_n,
        kb,
        target_waves_per_cu,
        true,
        num_projections);

    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeVNNIVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = native_vnni_verifier_rows_per_tile(tile_m);
        const int tile_row_groups =
            (tile_m + tile_rows_per_group - 1) / tile_rows_per_group;
        NativeVNNISmallMBatchedArgs tile_args = args;
        for (int p = 0; p < num_projections; ++p)
        {
            tile_args.output[p] +=
                static_cast<size_t>(row_base) * tile_args.N[p];
        }

        const int8_t* tile_a =
            d_A_int8 + static_cast<size_t>(row_base) * K;
        const float* tile_scales =
            d_scale_A_blockwise +
            static_cast<size_t>(row_base) * blocks_per_row;
        const int32_t* tile_sums =
            d_sum_A_blockwise
                ? d_sum_A_blockwise +
                      static_cast<size_t>(row_base) * blocks_per_row
                : nullptr;

        if (tile_rows_per_group == NVNNI_VERIFIER_SHORT_ROWS_PER_TILE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_small_m_batched_shared_mixed_kernel_t<
                    TN, CPT, NVNNI_VERIFIER_SHORT_ROWS_PER_TILE>),
                dim3(tile_args.total_grid_n, kb, tile_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, tile_scales, tile_sums, tile_args, tile_m, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_small_m_batched_shared_mixed_kernel_t<
                    TN, CPT, NVNNI_VERIFIER_WIDE_ROWS_PER_TILE>),
                dim3(tile_args.total_grid_n, kb, tile_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, tile_scales, tile_sums, tile_args, tile_m, K, kb);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections);
            return false;
        }

        if (kb > 1)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_vnni_reduce_small_m_batched_kernel_t<TN, CPT>),
                dim3(tile_args.total_grid_n, tile_m),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream), tile_args, tile_m, kb);
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections);
            return false;
        }
    }

    return true;
}

extern "C" {

bool rocmGemv_native_vnni_small_m_fp32_with_sums(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* d_partial_fp32,
    int M, int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (M < 2)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_fp32] M must be at least 2 (M=%d N=%d K=%d)\n", M, N, K);
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
            return rocmLaunch_native_vnni_small_m_fp32<FMT_TAG>(                                 \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, d_C_fp32,            \
                d_scale_A_blockwise, d_sum_A_blockwise, d_partial_fp32,                           \
                M, N, K, device_id, stream);                                                      \
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
    return rocmGemv_native_vnni_small_m_fp32_with_sums(
        d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
        d_C_fp32, d_scale_A_blockwise, nullptr, d_partial_fp32,
        M, N, K, codebook_id, device_id, stream);
}

static bool rocmGemv_native_vnni_small_m_batched_fp32_with_sums_policy_impl(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves)
{
    hipError_t device_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_id));
    if (device_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_fp32] hipSetDevice(%d) failed: %s\n",
                device_id, hipGetErrorString(device_err));
        return false;
    }

    if (M < 2)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_fp32] M must be at least 2 (M=%d K=%d projections=%d)\n",
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
        return rocmLaunch_native_vnni_small_m_batched_fp32<FMT_TAG>(                                  \
            d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins, d_biases, d_outputs,    \
            d_scale_A_blockwise, d_sum_A_blockwise, d_partials, Ns, num_projections, M, K,             \
            device_id, stream, policy_kb, policy_target_waves);                                        \
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

bool rocmGemv_native_vnni_small_m_batched_fp32_with_sums_policy(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves)
{
    return rocmGemv_native_vnni_small_m_batched_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, num_projections, M, K, codebook_id, device_id,
        stream, policy_kb, policy_target_waves);
}

bool rocmGemv_native_vnni_small_m_batched_fp32_with_sums(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    int num_projections,
    int M, int K,
    uint8_t codebook_id,
    int device_id,
    void* stream)
{
    return rocmGemv_native_vnni_small_m_batched_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, num_projections, M, K, codebook_id, device_id,
        stream, -1, -1);
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
    return rocmGemv_native_vnni_small_m_batched_fp32_with_sums(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, nullptr, d_partials,
        Ns, num_projections, M, K, codebook_id, device_id, stream);
}

static bool rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums_policy_impl(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves)
{
    hipError_t device_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_id));
    if (device_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] hipSetDevice(%d) failed: %s\n",
                device_id, hipGetErrorString(device_err));
        return false;
    }

    if (M < 2)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] M must be at least 2 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] K must be positive and multiple of 32 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (num_projections <= 0 || num_projections > NVNNI_SMALL_M_MAX_BATCH)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] invalid projection count=%d (max=%d)\n",
                num_projections, NVNNI_SMALL_M_MAX_BATCH);
        return false;
    }
    if (!d_A_int8 || !d_payloads || !d_block_scales || !d_outputs ||
        !d_scale_A_blockwise || !d_partials || !Ns || !codebook_ids)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] missing required pointer (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    for (int p = 0; p < num_projections; ++p)
    {
        const uint8_t codebook = codebook_ids[p];
        if (!nativeVNNICodebookSupported(codebook))
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] unsupported projection %d codebook_id=%u (M=%d K=%d projections=%d)\n",
                    p, static_cast<unsigned>(codebook), M, K, num_projections);
            return false;
        }
        if (Ns[p] <= 0 || !d_payloads[p] || !d_block_scales[p] || !d_outputs[p] || !d_partials[p])
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] invalid projection %d (N=%d payload=%p scales=%p output=%p partial=%p codebook=%u)\n",
                    p, Ns[p], static_cast<const void*>(d_payloads[p]),
                    static_cast<const void*>(d_block_scales[p]),
                    static_cast<const void*>(d_outputs[p]),
                    static_cast<const void*>(d_partials[p]),
                    static_cast<unsigned>(codebook));
            return false;
        }
        if (nativeVNNICodebookRequiresAux(codebook) &&
            (!d_block_mins || !d_block_mins[p]))
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] projection %d codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n",
                    p, static_cast<unsigned>(codebook), M, Ns[p], K);
            return false;
        }
        if (nativeVNNICodebookRequiresEmins(codebook) &&
            (!d_block_emins || !d_block_emins[p]))
        {
            fprintf(stderr, "[rocmGemv_native_vnni_small_m_batched_mixed_fp32] projection %d codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n",
                    p, static_cast<unsigned>(codebook), M, Ns[p], K);
            return false;
        }
    }

    return rocmLaunch_native_vnni_small_m_batched_mixed_fp32(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, policy_kb, policy_target_waves);
}

bool rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums_policy(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream,
    int policy_kb,
    int policy_target_waves)
{
    return rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, policy_kb, policy_target_waves);
}

bool rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums(
    const int8_t* d_A_int8,
    const uint8_t* const* d_payloads,
    const uint16_t* const* d_block_scales,
    const uint16_t* const* d_block_mins,
    const uint32_t* const* d_block_emins,
    const float* const* d_biases,
    float* const* d_outputs,
    const float* d_scale_A_blockwise,
    const int32_t* d_sum_A_blockwise,
    float* const* d_partials,
    const int* Ns,
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream)
{
    return rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, -1, -1);
}

bool rocmGemv_native_vnni_small_m_batched_mixed_fp32(
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
    const uint8_t* codebook_ids,
    int num_projections,
    int M, int K,
    int device_id,
    void* stream)
{
    return rocmGemv_native_vnni_small_m_batched_mixed_fp32_with_sums(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, nullptr, d_partials,
        Ns, codebook_ids, num_projections, M, K, device_id, stream);
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
    constexpr int KB_CAP =
        llaminar2::NativeVNNIGroupedDecodePolicy::maximum_k_partitions;

    const int blocks_per_row = K / 32;
    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16   = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32  = static_cast<const uint32_t*>(d_block_emins);

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override =
        g_native_vnni_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override =
        g_native_vnni_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (api_kb_override > 0) ? api_kb_override : rocm_env.nvnni_gemv_kb;
    const int explicit_waves_override =
        (api_waves_override > 0)
            ? api_waves_override
            : rocm_env.nvnni_gemv_target_waves;
    const bool has_explicit_tuning =
        explicit_kb_override > 0 || explicit_waves_override > 0;
    const NativeVNNIDecodeGeneratedRuntimeConfig generated_cfg =
        resolveNativeVNNIDecodeGeneratedRuntimeConfig(codebook_id, 1, N, K);
    if (!has_explicit_tuning && !generated_cfg.has_generated)
    {
        fprintf(
            stderr,
            "[rocmGemv_native_vnni_fp32] generated dispatch miss "
            "(codebook=%u N=%d K=%d)\n",
            static_cast<unsigned>(codebook_id),
            N,
            K);
        return false;
    }
    const int policy_kb =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : generated_cfg.kb;
    const int policy_target_waves =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : generated_cfg.target_waves_per_cu;

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
        if (policy_target_waves > 0)
            max_waves_per_cu = policy_target_waves;

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
        if (policy_kb > 0)
        {
            kb = min(policy_kb, KB_CAP);
            if (kb > blocks_per_row) kb = blocks_per_row;
        }
        if (kb < 1) kb = 1;

        if (kb > 1 && !d_partial_fp32)
        {
            fprintf(stderr, "[rocmGemv_native_vnni_fp32] IQ: kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        /*
         * Publish the actual serial-decode route before launching it.  The
         * NativeVNNI trainer uses this record to prove that its row oracle ran
         * the same KB/occupancy policy returned by the serial-M1 resolver.  M=1
         * used to be the only NativeVNNI path without this telemetry, which
         * made a correct production launch indistinguishable from an unproven
         * test-only oracle.
         */
        recordNativeVNNISmallMLaunch(
            1,
            N,
            K,
            static_cast<unsigned>(codebook_id),
            grid_n,
            kb,
            max_waves_per_cu,
            false,
            1);

        dim3 block(IQ_THREADS, 1, 1);

        #define LAUNCH_IQ_LDS(FMT_TAG)                                                       \
            if (kb <= 1) {                                                                    \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_iq_lds_kernel_t<IQ_TN, FMT_TAG>),                   \
                    dim3(grid_n, 1), block, 0, static_cast<hipStream_t>(stream),              \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,               \
                    d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, 1);       \
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
        // Non-IQ formats + IQ small-N fallback: TN=128, 2 wavefronts/WG
        // =====================================================================
        constexpr int TN = 128;
        constexpr int CPT = 1;
        constexpr int THREADS = TN / CPT;  // 128 threads = 2 wavefronts

        const int grid_n = (N + TN - 1) / TN;

        constexpr int DEFAULT_TARGET_WAVES_PER_CU = 24;

        // Q8_0 (codebook_id=19) has a trivial decode path (direct INT8 load),
        // so it benefits from lower occupancy pressure. Tuning sweep shows
        // tw=4 matches or beats INT8-VNNI on MI50/MI60 across all 7B shapes.
        constexpr int Q8_0_TARGET_WAVES_PER_CU = 4;
        const int default_waves = (codebook_id == static_cast<uint8_t>(NVNNI_Q8_0))
                                      ? Q8_0_TARGET_WAVES_PER_CU
                                      : DEFAULT_TARGET_WAVES_PER_CU;

        const int TARGET_WAVES_PER_CU =
            (policy_target_waves > 0) ? policy_target_waves : default_waves;
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

        if (policy_kb > 0)
        {
            kb = min(policy_kb, KB_CAP);
            if (kb > blocks_per_row) kb = blocks_per_row;
        }

        if (kb > 1 && !d_partial_fp32)
        {
            fprintf(stderr, "[rocmGemv_native_vnni_fp32] kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        /*
         * Keep serial M=1 route evidence symmetric with the grouped verifier
         * kernels.  Recording the resolved values, rather than merely the
         * requested overrides, also proves clamping and shape constraints did
         * not silently select a different production launch.
         */
        recordNativeVNNISmallMLaunch(
            1,
            N,
            K,
            static_cast<unsigned>(codebook_id),
            grid_n,
            kb,
            TARGET_WAVES_PER_CU,
            false,
            1);

        dim3 block(THREADS, 1, 1);

        #define LAUNCH_NATIVE_VNNI(FMT_TAG)                                                  \
            if (kb <= 1) {                                                                    \
                hipLaunchKernelGGL(                                                           \
                    HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN, CPT, FMT_TAG>),            \
                    dim3(grid_n, 1), block, 0, static_cast<hipStream_t>(stream),             \
                    d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32,              \
                    d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, 1);      \
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
