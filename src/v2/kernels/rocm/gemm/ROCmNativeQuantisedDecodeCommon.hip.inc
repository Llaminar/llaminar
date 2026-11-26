/**
 * @file ROCmNativeQuantisedDecodeCommon.hip.inc
 * @brief Shared device decoders, format traits, and internal cross-TU declarations for sharded ROCm NativeQuantised GEMV/MoE.
 */
#pragma once
/**
 * @file ROCmGemvKernel_native_quantised.hip
 * @brief NativeQuantised GEMV kernels for lossless sub-8-bit decode on AMD MI50/MI60 (gfx906)
 *
 * This file contains native-quantised kernels that decode quantized weights
 * losslessly and compute GEMV using v_dot4_i32_i8. Q6 keeps original half d
 * plus signed subscales; other formats retain their prepared FP16 metadata.
 *
 * Supported formats:
 *   Q4_0:   w_int8 = q - 8          (exact, symmetric)
 *   IQ4_NL: w_int8 = lut[q]         (exact, LUT-based symmetric)
 *   IQ4_XS: w_int8 = lut[q]         (physical24, original d/raw coefficient)
 *   Q4_1:   w_int8 = q, min corr.   (asymmetric, scale + min)
 *   Q4_K:   w_int8 = q, min corr.   (reuses Q4_1 kernel, sub-block scales precomputed)
 *   Q5_0:   w_int8 = 5bit - 16      (exact, symmetric with high-bit array)
 *   Q5_1:   w_int8 = 5bit, min corr.(asymmetric, scale + min + high-bit array)
 *   Q5_K:   w_int8 = 5bit, min corr.(reuses Q5_1 kernel, sub-block scales precomputed)
 *   Q6_K:   w_int8 = 6bit - 32      (symmetric, dual-scale per 32-element block)
 *   Q3_K:   w_int8 = 3bit - 4       (symmetric, dual-scale per 32-element block)
 *   Q2_K:   w_int8 = 2bit (unsigned)(asymmetric, dual-scale + embedded mins)
 *   IQ3_S:  grid LUT → signed INT8; original half base/raw nibble resolve
 *           one FP32 factor before the unchanged single-dot contribution
 *   IQ3_XXS:grid LUT → signed INT8  (8-bit grid index + pre-resolved signs)
 *   IQ2_S:  grid LUT → signed INT8  (10-bit grid index + direct signs, dual-scale)
 *   IQ2_XS: grid LUT → signed INT8  (9-bit grid index + pre-resolved signs, dual-scale)
 *   IQ2_XXS:grid LUT → signed INT8  (8-bit grid index + pre-resolved signs)
 *
 * Unlike ratio-VNNI (which bakes scales into INT8 weights with truncation),
 * native-quantised keeps per-block metadata separate and decodes nibbles directly
 * to INT8. Q6 reconstructs exact FP32 d*subscale before the unchanged dot tree.
 *
 * Captured verifier launches separate physical scratch capacity from the
 * device controller's live row count. Fused bundles use bounded row workers,
 * keep the serial-M1 reduction tree, and specialize their publication target
 * before capture. Neither inactive capacity nor an optional bias may extend
 * decoder live ranges into register spills.
 *
 * Native-Q8 serial diagnostic geometry scopes alter only independent column
 * ownership, require exact pre-capture symbol resources and preserve every
 * source-policy K partition. Ordinary inference retains the existing
 * TN128/CPT1 launch. Thread-local receipts observe construction topology;
 * they never mirror live device state or count captured graph replays.
 *
 * Participant-local routed decode keeps original split-K partitions in CTA-owned
 * LDS and fuses their ordered publication into one native launch per projection.
 * Source-policy partition counts and public route fragments remain unchanged.
 *
 * An opt-in gfx906 HC-down contributor reuses each original physical Q8 block
 * across up to four device-admitted verifier rows and preserves the staged reducer.
 *
 * Split from ROCmGemvKernel.hip.
 *
 * @see ROCmGemvKernel_INT8_VNNI.hip for INT8 VNNI kernels
 *
 * Projection work admission is copied independently of arithmetic dispatch.
 * Borrowed original request intervals guard native operand/partial/output
 * accesses; physical matrix strides and each row's reduction order stay fixed.
 * Native signed INT8 dot products share the exact unsaturated arithmetic
 * contract across gfx906 and gfx1100; lane ownership and reductions are unchanged.
 */

#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/common/ProjectionWorkLaunch.h"
#include "kernels/common/NativeQuantisedPackedBits.h"
#include "kernels/common/NativeTileCompletion.cuh"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/common/GPUQ8ResidualNative.h"
#include <hip/hip_cooperative_groups.h>
#include <optional>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include "kernels/common/NativeEmbeddingIQTables.h"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <string>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/rocm/gemm/ROCmGroupedVerifierLaunch.h"
#include "ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeM1Reduction.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/PrefillGraphBucketDefaults.h"
#include "execution/moe/DeviceMoEOverlayEpochABI.h"
#include "execution/moe/DeviceMoERuntimeABI.h"
#include "execution/moe/DeviceMoEIntermediateShardBinding.h"
#include "kernels/common/MoEIntermediateShardRowExecution.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"
#include "kernels/common/NativeQuantisedDispatchCache.h"
#include "kernels/common/DeviceRowWorkGrid.h"
#include "kernels/common/NativeQuantisedGroupedDecodePolicy.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "ROCmNativeQuantisedSerialGeometry.h"
#include "ROCmResidentMoEFFN.h"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "ROCmGateUpLastProducer.h"
#include "ROCmDownLastProducer.h"
#include "kernels/common/DeviceOrderedRouteFold.h"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/rocm/moe/ROCmSharedExpertGateAddBody.cuh"
#include <hip/hip_cooperative_groups.h>
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <exception>
#include <utility>
#include "../ops/ROCmSwiGLUDecodeEquivalentMath.h"

#include <atomic>
#include <stdexcept>


constexpr uint8_t CODEBOOK_IQ4 = 4;

static __device__ __constant__ int8_t k_iq4_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

// ============================================================================
// Device-constant IQ grid lookup tables (initialized at runtime via
// rocmInitIQGridTables before first IQ kernel launch).
// ============================================================================
static __device__ __constant__ uint32_t d_iq3s_grid[512];    // IQ3_S:   512 × 4B = 2KB
static __device__ __constant__ uint32_t d_iq3xxs_grid[256];  // IQ3_XXS: 256 × 4B = 1KB
static __device__ __constant__ uint64_t d_iq2s_grid[1024];   // IQ2_S:   1024 × 8B = 8KB
static __device__ __constant__ uint64_t d_iq2xs_grid[512];   // IQ2_XS:  512 × 8B = 4KB
static __device__ __constant__ uint64_t d_iq2xxs_grid[256];  // IQ2_XXS: 256 × 8B = 2KB
static __device__ __constant__ uint64_t d_iq1s_grid[2048];   // IQ1_S/M: 2048 × 8B = 16KB


// ============================================================================
// Format tags for native-quantised kernel dispatch
//
// Each value is the codebook_id stored in ROCmPackedWeights::native_quantised_codebook_id
// and passed to rocmGemv_native_quantised_fp32() for template selection.
// ============================================================================
enum NativeQuantisedFormat : uint8_t {
    NATIVE_QUANTISED_Q4_0   = 0,   // 4-bit symmetric, linear decode (q - 8)
    NATIVE_QUANTISED_IQ4_NL = 4,   // IQ4_NL original half scale
    NATIVE_QUANTISED_IQ4_XS_SOURCE = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook, // Original IQ4_XS tuple
    NATIVE_QUANTISED_Q4_1   = 5,   // 4-bit asymmetric (q as-is, has min)
    NATIVE_QUANTISED_Q5_0   = 6,   // 5-bit symmetric, linear decode (q - 16)
    NATIVE_QUANTISED_Q5_1   = 7,   // 5-bit asymmetric (q as-is, has min)
    NATIVE_QUANTISED_Q6_K   = 8,   // 6-bit symmetric, dual-scale per 32-element block
    NATIVE_QUANTISED_Q3_K   = 9,   // 3-bit symmetric, dual-scale per 32-element block
    NATIVE_QUANTISED_Q2_K   = 10,  // 2-bit asymmetric, dual-scale + embedded mins
    NATIVE_QUANTISED_IQ3_S  = 11,  // IQ3 grid (4 elem/lookup), 9-bit index, direct signs
    NATIVE_QUANTISED_IQ3_XXS= 12,  // IQ3 grid (4 elem/lookup), 8-bit index, pre-resolved signs
    NATIVE_QUANTISED_IQ2_S  = 13,  // IQ2 grid (8 elem/lookup), 10-bit index, direct signs, dual-scale
    NATIVE_QUANTISED_IQ2_XS = 14,  // IQ2 grid (8 elem/lookup), 9-bit index, pre-resolved signs, dual-scale
    NATIVE_QUANTISED_IQ2_XXS= 15,  // IQ2 grid (8 elem/lookup), 8-bit index, pre-resolved signs
    NATIVE_QUANTISED_IQ1_S  = 16,  // IQ1 grid (8 elem/lookup), 11-bit index, signed values {-1,0,+1}, delta correction
    NATIVE_QUANTISED_IQ1_M  = 17,  // IQ1 grid (8 elem/lookup), 11-bit index, signed values {-1,0,+1}, dual-scale + delta
    NATIVE_QUANTISED_Q8_0   = 19,  // 8-bit symmetric, direct INT8 payload (32 bytes)
    NATIVE_QUANTISED_EXPANDED_INT8_MIN =
        llaminar2::kNativeQuantisedExpandedInt8MinCodebook,
};

// Compile-time format properties
template <NativeQuantisedFormat FMT>
struct NATIVE_QUANTISEDTraits {
    // Asymmetric formats have a min correction term: f_acc += sum_a * block_m
    static constexpr bool is_asymmetric =
        FMT == NATIVE_QUANTISED_Q4_1 || FMT == NATIVE_QUANTISED_Q5_1 ||
        FMT == NATIVE_QUANTISED_IQ1_S || FMT == NATIVE_QUANTISED_EXPANDED_INT8_MIN;
    // --- IQ grid format traits ---
    static constexpr bool is_iq3_grid = (FMT == NATIVE_QUANTISED_IQ3_S || FMT == NATIVE_QUANTISED_IQ3_XXS);
    static constexpr bool is_iq2_grid = (FMT == NATIVE_QUANTISED_IQ2_S || FMT == NATIVE_QUANTISED_IQ2_XS || FMT == NATIVE_QUANTISED_IQ2_XXS);
    static constexpr bool is_iq1_grid = (FMT == NATIVE_QUANTISED_IQ1_S || FMT == NATIVE_QUANTISED_IQ1_M);
    static constexpr bool is_iq_grid  = (is_iq3_grid || is_iq2_grid);
    // High bits per grid lookup stored in qh byte (0, 1, or 2)
    static constexpr int qh_bits_per_lookup =
        (FMT == NATIVE_QUANTISED_IQ2_S) ? 2 : (FMT == NATIVE_QUANTISED_IQ3_S || FMT == NATIVE_QUANTISED_IQ2_XS) ? 1 : 0;
    static constexpr bool has_qh = (qh_bits_per_lookup > 0);
    // Dual-scale: two FP16 scales per 32-element block (stored in scales + mins arrays)
    // Dual halves use resolved FP32 scales; Q6 stores d and packed coefficients.
    static constexpr bool is_dual_scale = (FMT == NATIVE_QUANTISED_Q6_K || FMT == NATIVE_QUANTISED_Q3_K || FMT == NATIVE_QUANTISED_Q2_K
                                        || FMT == NATIVE_QUANTISED_IQ2_S || FMT == NATIVE_QUANTISED_IQ2_XS
                                        || FMT == NATIVE_QUANTISED_IQ1_M);
    // Dual-scale + asymmetric: split sum_a correction with per-sub-block emins
    // IQ1_M uses Pattern D (not DA): has ±IQ1S_DELTA per-group correction (handled separately)
    static constexpr bool is_dual_scale_asym = (FMT == NATIVE_QUANTISED_Q2_K);
    // Q2_K: emins in separate global array
    static constexpr bool needs_emin_array = (FMT == NATIVE_QUANTISED_Q2_K);
    // All scales/mins now in global arrays — no embedded data in payload
    static constexpr bool has_embedded_scales = false;
    static constexpr int  embedded_scale_offset = -1;
    static constexpr int  embedded_min_offset = -1;
    static constexpr bool is_5bit       = (FMT == NATIVE_QUANTISED_Q5_0 || FMT == NATIVE_QUANTISED_Q5_1);
    static constexpr bool is_6bit       = (FMT == NATIVE_QUANTISED_Q6_K);
    static constexpr bool is_3bit       = (FMT == NATIVE_QUANTISED_Q3_K);
    static constexpr bool is_2bit       = (FMT == NATIVE_QUANTISED_Q2_K);
    static constexpr bool is_8bit =
        FMT == NATIVE_QUANTISED_Q8_0 || FMT == NATIVE_QUANTISED_EXPANDED_INT8_MIN;
    static constexpr bool requires_secondary_metadata =
        is_asymmetric || is_dual_scale || FMT == NATIVE_QUANTISED_IQ3_S || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE;
    static constexpr bool uses_lut      = (FMT == NATIVE_QUANTISED_IQ4_NL || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE);
    static constexpr int  payload_bytes =
        is_8bit ? 32 :                                // Q8_0: 32 raw INT8 values
        is_iq1_grid ? 6 :                            // IQ1_S(6), IQ1_M(6): qs[4]+qh[2] only
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 12 : is_2bit ? 8 : 16;
    // Centering offset for symmetric formats; asymmetric formats don't center
    static constexpr int center_4bit = (FMT == NATIVE_QUANTISED_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NATIVE_QUANTISED_Q5_0) ? 16 : 0;
    // IQ LDS grid cache properties (only meaningful for IQ grid/IQ1 grid formats)
    static constexpr int grid_entries =
        (FMT == NATIVE_QUANTISED_IQ3_S) ? 512 : (FMT == NATIVE_QUANTISED_IQ3_XXS) ? 256 :
        (FMT == NATIVE_QUANTISED_IQ2_S) ? 1024 : (FMT == NATIVE_QUANTISED_IQ2_XS) ? 512 : (FMT == NATIVE_QUANTISED_IQ2_XXS) ? 256 :
        (FMT == NATIVE_QUANTISED_IQ1_S || FMT == NATIVE_QUANTISED_IQ1_M) ? 2048 : 0;
    static constexpr int grid_entry_bytes = is_iq3_grid ? 4 : (is_iq2_grid || is_iq1_grid) ? 8 : 0;
    static constexpr int grid_lds_bytes = grid_entries * grid_entry_bytes;
};

/**
 * @brief Dilate four packed Q6 two-bit fields into four byte-local high nibbles.
 *
 * A Q6 high-bit byte stores four consecutive two-bit fields in bits
 * `[1:0]`, `[3:2]`, `[5:4]`, and `[7:6]`.  NativeQuantised dot operands need
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
static constexpr uint32_t rocm_native_quantised_expand_q6_high_fields_reference(
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
static constexpr uint32_t rocm_native_quantised_expand_q6_high_fields_dilated(
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
static constexpr bool rocm_native_quantised_q6_high_field_dilation_is_exhaustive()
{
    for (uint32_t packed = 0; packed <= 0xFFu; ++packed)
    {
        if (rocm_native_quantised_expand_q6_high_fields_dilated(packed) !=
            rocm_native_quantised_expand_q6_high_fields_reference(packed))
        {
            return false;
        }
    }
    return true;
}

static_assert(
    rocm_native_quantised_q6_high_field_dilation_is_exhaustive(),
    "ROCm Q6 high-field dilation must equal the direct mapping for every byte");

/**
 * @brief Device wrapper for the compile-time-proven Q6 dilation identity.
 * @param packed_fields The Q6 high-bit byte, zero-extended to 32 bits.
 * @return Four expanded fields in bits `[5:4]` of the result's four bytes.
 */
__device__ __forceinline__ uint32_t rocm_native_quantised_expand_q6_high_fields(
    uint32_t packed_fields)
{
    return rocm_native_quantised_expand_q6_high_fields_dilated(packed_fields);
}

template <NativeQuantisedFormat FMT>
struct NATIVE_QUANTISEDGroupedMoESupported {
    static constexpr bool value =
        FMT == NATIVE_QUANTISED_Q4_0 || FMT == NATIVE_QUANTISED_IQ4_NL || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE || FMT == NATIVE_QUANTISED_Q4_1 ||
        FMT == NATIVE_QUANTISED_Q5_0 || FMT == NATIVE_QUANTISED_Q5_1 || FMT == NATIVE_QUANTISED_Q6_K ||
        FMT == NATIVE_QUANTISED_Q3_K || FMT == NATIVE_QUANTISED_Q2_K || FMT == NATIVE_QUANTISED_IQ3_S ||
        FMT == NATIVE_QUANTISED_IQ3_XXS || FMT == NATIVE_QUANTISED_IQ2_S || FMT == NATIVE_QUANTISED_IQ2_XS ||
        FMT == NATIVE_QUANTISED_IQ2_XXS || FMT == NATIVE_QUANTISED_IQ1_S || FMT == NATIVE_QUANTISED_IQ1_M ||
        FMT == NATIVE_QUANTISED_Q8_0 || FMT == NATIVE_QUANTISED_EXPANDED_INT8_MIN;
};

/**
 * @brief Resolve serial-M1 policy from an exact source arithmetic codebook.
 *
 * Generated dispatch remains the economy authority for target wave count and
 * for codebooks without a heterogeneous arithmetic contract.  A certified
 * movable policy instead replaces only the K-partition count: launch tuning
 * must not change the FP32 expression tree when an expert moves between CUDA
 * and ROCm.
 */

struct ROCmMoESerialM1PolicyTable
{
    static constexpr int kCodebookCapacity = 32;

    /*
     * Eight byte-sized partition counts live in each scalar word.  Kernels
     * receive these as four independent kernarg scalars: an addressable
     * `uint8_t[32]` parameter made clang materialize a per-work-item private
     * copy, producing 16--24 bytes of scratch in every descriptor-local wave.
     */
    uint64_t k_partitions_0_7 = 0;
    uint64_t k_partitions_8_15 = 0;
    uint64_t k_partitions_16_23 = 0;
    uint64_t k_partitions_24_31 = 0;
    uint8_t maximum_k_partitions = 0;
    uint8_t reserved[3]{};
};

/** @brief Store one host-resolved partition count in the packed policy table. */


struct NativeQuantisedDecodeGeneratedRuntimeConfig
{
    bool has_generated = false;
    int kb = -1;
    int target_waves_per_cu = -1;
};

namespace llaminar2::rocm::native_quantised_internal
{
    int kbOverride() noexcept;
    int targetWavesOverride() noexcept;
    bool decodeEquivalentSmallMMode() noexcept;
    int serialPartitionN() noexcept;
    bool nativeQ8SerialScopeActive() noexcept;
    bool validateDecodeEquivalentKBOverride(
        const char* caller, int explicit_kb_override, int m, int n, int k, int projections);
    void recordSmallMLaunch(
        int m, int n, int k, uint8_t codebook, int grid_n, int kb,
        int target_waves_per_cu, bool batched, int projections,
        int arithmetic_policy_codebook = -1);
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveDecodeGeneratedRuntimeConfigForPolicy(
        uint8_t arithmetic_policy_codebook_id, int m, int n, int k);
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveDecodeGeneratedRuntimeConfig(
        uint8_t codebook_id, int m, int n, int k);
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveSerialM1RuntimeConfigForPolicy(
        uint8_t arithmetic_policy_codebook_id, int N, int K);
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveSerialM1RuntimeConfig(
        uint8_t codebook_id, int N, int K);
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveCommonSerialM1RuntimeConfig(
        uint8_t codebook_id, const int *Ns, int num_projections, int K);
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveCommonSerialM1RuntimeConfig(
        const uint8_t *codebook_ids, const int *Ns, int num_projections, int K);
    int resolveSerialM1KPartitions(uint8_t codebook_id, int N, int K);
    int resolveSerialM1KPartitionsForPolicy(uint8_t policy_codebook_id, int N, int K);
    void recordMoEDescriptorLocalDecode(
        const char* projection,
        const char* descriptor_source,
        uint8_t table_codebook,
        uint32_t policy_codebook_mask,
        const ROCmMoESerialM1PolicyTable& policy_table);
    llaminar2::rocm::NativeQ8SerialKernelResources inspectNativeQ8SerialSymbol(
        const void *fn, const char *name, int threads);
} // namespace llaminar2::rocm::native_quantised_internal

static constexpr uint8_t defaultNativeQuantisedArithmeticPolicyCodebook(uint8_t execution_codebook_id)
{
    return execution_codebook_id == llaminar2::kNativeQuantisedExpandedInt8MinCodebook
        ? static_cast<uint8_t>(llaminar2::native_quantised_formats::Q4_K.codebook_id)
        : execution_codebook_id;
}

extern "C" bool rocmQuantGemm_biasAdd(
    float *C, const float *bias, int M, int N, int rocm_device_id, void *stream);
extern "C" bool rocmNativeEmbeddingIQTableView(
    int device_id, llaminar2::NativeEmbeddingIQTables* out);


namespace llaminar2::rocm::native_quantised_internal
{
    bool buildROCmMoESerialM1PolicyTable(
        int N, int K, uint32_t required_policy_mask, ROCmMoESerialM1PolicyTable &out);
}


#include "ROCmPersistentHCRead.h"
#include "ROCmNativeLayoutRouterFFN.h"
#include "ROCmFullNativeHCFFN.h"

namespace llaminar2::rocm
{
    void resolveResidentMoEFFNPolicies(ResidentMoEFFNArgs& args, ResidentMoEFFNPolicyMasks masks);
    void validatePersistentHCRead(const PersistentHCReadArgs& a, bool ctaOwned = false,
        FullNativeHCFFNOutputDemand output_demand = FullNativeHCFFNOutputDemand::all());

    struct NativeLayoutRouterFFNLaunchGeometry {
        ResidentMoEFFNDecoderProgram decoder;
        std::size_t dynamic_shared_bytes;
        std::uint64_t useful_ctas;
    };
    NativeLayoutRouterFFNLaunchGeometry resolveNativeLayoutRouterFFNGeometry(
        DeviceId device, NativeLayoutRouterFFNArgs& args,
        ResidentMoEFFNPolicyMasks masks, std::uint64_t generation);
}


#define LLAMINAR_DEFINE_ROCM_IQ_GRID_INIT(SUFFIX)                                         \
namespace llaminar2::rocm::native_quantised_internal {                                    \
bool initIQGridTables_##SUFFIX(                                                           \
    int device_id, const void* h_iq3s_grid, const void* h_iq3xxs_grid,                    \
    const void* h_iq2s_grid, const void* h_iq2xs_grid,                                    \
    const void* h_iq2xxs_grid, const void* h_iq1s_grid)                                   \
{                                                                                         \
    llaminar2::HipDeviceGuard::setDevice(device_id);                                      \
    hipError_t e;                                                                         \
    e = hipMemcpyToSymbol(d_iq3s_grid, h_iq3s_grid, sizeof(d_iq3s_grid));                 \
    if (e != hipSuccess) return false;                                                    \
    e = hipMemcpyToSymbol(d_iq3xxs_grid, h_iq3xxs_grid, sizeof(d_iq3xxs_grid));           \
    if (e != hipSuccess) return false;                                                    \
    e = hipMemcpyToSymbol(d_iq2s_grid, h_iq2s_grid, sizeof(d_iq2s_grid));                 \
    if (e != hipSuccess) return false;                                                    \
    e = hipMemcpyToSymbol(d_iq2xs_grid, h_iq2xs_grid, sizeof(d_iq2xs_grid));              \
    if (e != hipSuccess) return false;                                                    \
    e = hipMemcpyToSymbol(d_iq2xxs_grid, h_iq2xxs_grid, sizeof(d_iq2xxs_grid));           \
    if (e != hipSuccess) return false;                                                    \
    e = hipMemcpyToSymbol(d_iq1s_grid, h_iq1s_grid, sizeof(d_iq1s_grid));                 \
    if (e != hipSuccess) return false;                                                    \
    return true;                                                                          \
}                                                                                         \
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
template <NativeQuantisedFormat FMT>
__device__ inline uint32_t iq3_grid_lookup(int idx)
{
    if constexpr (FMT == NATIVE_QUANTISED_IQ3_S) return d_iq3s_grid[idx];
    else                               return d_iq3xxs_grid[idx];
}

// Format-specific IQ2 grid lookup (resolved at compile time)
template <NativeQuantisedFormat FMT>
__device__ inline uint64_t iq2_grid_lookup(int idx)
{
    if constexpr (FMT == NATIVE_QUANTISED_IQ2_S)       return d_iq2s_grid[idx];
    else if constexpr (FMT == NATIVE_QUANTISED_IQ2_XS) return d_iq2xs_grid[idx];
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
template <NativeQuantisedFormat FMT, bool USE_LDS>
__device__ __forceinline__ uint32_t iq3_grid_read(int idx,
    [[maybe_unused]] const uint32_t* lds32)
{
    if constexpr (USE_LDS) return lds32[idx];
    else return iq3_grid_lookup<FMT>(idx);
}

template <NativeQuantisedFormat FMT, bool USE_LDS>
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
//   FMT     — NativeQuantisedFormat tag (selects decode + accumulate path)
//   USE_LDS — false: read grid from constant memory; true: read from LDS
// ============================================================================
template <int CPT, NativeQuantisedFormat FMT, bool USE_LDS = false>
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
    using Traits = NATIVE_QUANTISEDTraits<FMT>;
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
                if constexpr (FMT == NATIVE_QUANTISED_IQ1_S)
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
                    acc_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_lo);
                #pragma unroll
                for (int g = 4; g < 8; ++g)
                    acc_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_hi);

                const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
                    scale_bits, d_block_mins[linear]);
                const float scale_lo = weight_scales.low;
                const float scale_hi = weight_scales.high;
                if constexpr (Traits::is_dual_scale_asym)
                {
                    constexpr int32_t ones = 0x01010101;
                    int32_t sum_a_lo = 0, sum_a_hi = 0;
                    #pragma unroll
                    for (int g = 0; g < 4; ++g)
                        sum_a_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a_lo);
                    #pragma unroll
                    for (int g = 4; g < 8; ++g)
                        sum_a_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a_hi);

                    // Q2_K: per-sub-block emin correction
                    constexpr float min_lo = 0.0f;
                    constexpr float min_hi = 0.0f;
                    f_acc[c] =
                        rocm_native_quantised_commit_corrected_dual_scale_block(
                            f_acc[c],
                            acc_lo,
                            acc_hi,
                            scale_lo,
                            scale_hi,
                            sum_a_lo,
                            sum_a_hi,
                            min_lo,
                            min_hi,
                            scale_ab);
                }

                // IQ1_M: ±IQ1S_DELTA per 8-element group correction.
                // True weight[j] = dl * (grid[j] + delta), but the sdot4 above
                // only accumulates dl * dot(a, grid).  We add the missing
                // dl * delta * sum_a_per_group for each of the 4 groups.
                // Delta signs: qh0 bit 3 (group 0), qh0 bit 7 (group 1),
                //              qh1 bit 3 (group 2), qh1 bit 7 (group 3).
                else if constexpr (FMT == NATIVE_QUANTISED_IQ1_M)
                {
                    constexpr float IQ1S_DELTA = 0.125f;
                    constexpr int32_t ones = 0x01010101;
                    const uint8_t qh0 = payload[4], qh1 = payload[5];

                    // Per-group sum of activation bytes (8 int8 elements per group)
                    int32_t sa_g0 = ::llaminar2::rocm_integer_contract::signedDot4(a4[1], ones,
                                    ::llaminar2::rocm_integer_contract::signedDot4(a4[0], ones, 0));
                    int32_t sa_g1 = ::llaminar2::rocm_integer_contract::signedDot4(a4[3], ones,
                                    ::llaminar2::rocm_integer_contract::signedDot4(a4[2], ones, 0));
                    int32_t sa_g2 = ::llaminar2::rocm_integer_contract::signedDot4(a4[5], ones,
                                    ::llaminar2::rocm_integer_contract::signedDot4(a4[4], ones, 0));
                    int32_t sa_g3 = ::llaminar2::rocm_integer_contract::signedDot4(a4[7], ones,
                                    ::llaminar2::rocm_integer_contract::signedDot4(a4[6], ones, 0));

                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

                    f_acc[c] = rocm_native_quantised_commit_iq1_m_block(
                        f_acc[c],
                        acc_lo,
                        acc_hi,
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
                else
                {
                    f_acc[c] = rocm_native_quantised_commit_dual_scale_block(
                        f_acc[c],
                        acc_lo,
                        acc_hi,
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
                    block_acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], block_acc);

                const float block_d = rocm_native_quantised_load_single_weight_scale<FMT>(
                    scale_bits, (FMT == NATIVE_QUANTISED_IQ3_S || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE) ? d_block_mins[linear] : uint16_t{0});
                if constexpr (Traits::is_asymmetric)
                {
                    const uint16_t min_bits = d_block_mins[linear];
                    const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                    f_acc[c] =
                        rocm_native_quantised_commit_corrected_single_scale_block(
                            f_acc[c],
                            block_acc,
                            block_d,
                            sum_a,
                            block_m,
                            scale_ab);
                }
                else
                {
                    f_acc[c] = rocm_native_quantised_commit_single_scale_block(
                        f_acc[c], block_acc, block_d, scale_ab);
                }
            }
        }
    }
}

// ============================================================================
// NativeQuantised GEMV kernel: lossless decode with per-block FP16 scales
//
// Supports all ≤6-bit quantized formats via compile-time NativeQuantisedFormat tag:
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
/**
 * @brief Original scalar NativeQuantised producer with explicit logical tile/partition/lane ownership.
 * @details This is the former kernel body unchanged in arithmetic. Standalone
 * and resident entries share it, so physical grid virtualization cannot invent
 * another decoder, dot loop, K partition boundary or partial publication order.
 */
template <int TILE_N, int CPT, NativeQuantisedFormat FMT, bool HasWork = false>
__device__ __forceinline__ void nativeM1ColumnsBody(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations (INT8)
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × payload_bytes] payload
    const uint16_t*  __restrict__ d_block_scales,  // [blocks_per_row × N] FP16 scale (d)
    const uint16_t*  __restrict__ d_block_mins,    // [blocks_per_row × N] FP16 min (m), NULL for symmetric
    const uint32_t*  __restrict__ d_block_emins,   // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, else nullptr)
    float*           __restrict__ d_C_fp32,        // [N] output (FP32), used when kblocks=1
    float*           __restrict__ d_partial_fp32,  // [KB × N] partials, used when kblocks>1
    const float*     __restrict__ d_scale_A,       // [1] per-row activation scale (CK fallback, used when d_scale_A_blockwise==nullptr)
    const float*     __restrict__ d_scale_A_blockwise, // [blocks_per_row] per-block activation scales
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work,
    unsigned logical_tile,unsigned logical_partition,unsigned logical_lane)
{
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return; // Same ordinary M1 route; no unauthored reads/writes.

    using Traits = NATIVE_QUANTISEDTraits<FMT>;
    static_assert(CPT == 1 || CPT == 2, "native-quantised supports CPT in {1,2}");

    const int n_base = logical_tile * TILE_N + logical_lane * CPT;
    const bool active = (n_base < N);
    const int kb = logical_partition;

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
                        sum_a_0 = ::llaminar2::rocm_integer_contract::signedDot4(a4_0[g], ones, sum_a_0);
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                        sum_a_1 = ::llaminar2::rocm_integer_contract::signedDot4(a4_1[g], ones, sum_a_1);
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
                        sum_a = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a);
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
                    sum_a = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a);
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
                                rocm_native_quantised_expand_q6_high_fields(h_lo);
                            uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
                            val_lo -= 0x20202020u;
                            val_lo ^= 0x80808080u;
                            packed_groups[g] = static_cast<int32_t>(val_lo);

                            // High nibbles + scattered 2-bit highs + subtract-32
                            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                            const uint32_t hb_hi =
                                rocm_native_quantised_expand_q6_high_fields(h_hi);
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
                        if constexpr (FMT == NATIVE_QUANTISED_IQ1_S)
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
                        else // NATIVE_QUANTISED_IQ1_M
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
                        // Q6 stores original d and packed coefficients in these two words.
                        int32_t acc_lo = 0, acc_hi = 0;
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                            acc_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_lo);
                        #pragma unroll
                        for (int g = 4; g < 8; ++g)
                            acc_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_hi);

                        const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
                            scale_bits, d_block_mins[linear]);
                        const float scale_lo = weight_scales.low;
                        const float scale_hi = weight_scales.high;
                        // Q2_K/IQ1_M: dual-scale + emin/delta correction
                        // Q2_K: emins from separate d_block_emins array
                        // IQ1_M: has ±IQ1S_DELTA per-group correction (handled below)
                        if constexpr (Traits::is_dual_scale_asym)
                        {
                            constexpr int32_t ones = 0x01010101;
                            int32_t sum_a_lo = 0, sum_a_hi = 0;
                            #pragma unroll
                            for (int g = 0; g < 4; ++g)
                                sum_a_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a_lo);
                            #pragma unroll
                            for (int g = 4; g < 8; ++g)
                                sum_a_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a_hi);

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
                            f_acc[c] =
                                rocm_native_quantised_commit_corrected_dual_scale_block(
                                    f_acc[c],
                                    acc_lo,
                                    acc_hi,
                                    scale_lo,
                                    scale_hi,
                                    sum_a_lo,
                                    sum_a_hi,
                                    min_lo,
                                    min_hi,
                                    scale_ab);
                        }

                        // IQ1_M: ±IQ1S_DELTA per 8-element group correction.
                        // True weight[j] = dl * (grid[j] + delta), but the sdot4 above
                        // only accumulates dl * dot(a, grid).  We add the missing
                        // dl * delta * sum_a_per_group for each of the 4 groups.
                        else if constexpr (FMT == NATIVE_QUANTISED_IQ1_M)
                        {
                            constexpr float IQ1S_DELTA = 0.125f;
                            constexpr int32_t ones = 0x01010101;
                            const uint8_t qh0 = payload[4], qh1 = payload[5];

                            int32_t sa_g0 = ::llaminar2::rocm_integer_contract::signedDot4(a4[1], ones,
                                            ::llaminar2::rocm_integer_contract::signedDot4(a4[0], ones, 0));
                            int32_t sa_g1 = ::llaminar2::rocm_integer_contract::signedDot4(a4[3], ones,
                                            ::llaminar2::rocm_integer_contract::signedDot4(a4[2], ones, 0));
                            int32_t sa_g2 = ::llaminar2::rocm_integer_contract::signedDot4(a4[5], ones,
                                            ::llaminar2::rocm_integer_contract::signedDot4(a4[4], ones, 0));
                            int32_t sa_g3 = ::llaminar2::rocm_integer_contract::signedDot4(a4[7], ones,
                                            ::llaminar2::rocm_integer_contract::signedDot4(a4[6], ones, 0));

                            const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

                            f_acc[c] = rocm_native_quantised_commit_iq1_m_block(
                                f_acc[c],
                                acc_lo,
                                acc_hi,
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
                        else
                        {
                            f_acc[c] = rocm_native_quantised_commit_dual_scale_block(
                                f_acc[c],
                                acc_lo,
                                acc_hi,
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
                            block_acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], block_acc);

                        const float block_d = rocm_native_quantised_load_single_weight_scale<FMT>(
                            scale_bits, (FMT == NATIVE_QUANTISED_IQ3_S || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE) ?
                                d_block_mins[linear] : uint16_t{0});
                        // Asymmetric min correction: f_acc += sum_a * block_m
                        // Derives from: value = d*q + m  →  Σ a_i*value_i = d*Σ(a_i*q_i) + m*Σ(a_i)
                        if constexpr (Traits::is_asymmetric)
                        {
                            const uint16_t min_bits = d_block_mins[linear];
                            const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
                            f_acc[c] =
                                rocm_native_quantised_commit_corrected_single_scale_block(
                                    f_acc[c],
                                    block_acc,
                                    block_d,
                                    sum_a,
                                    block_m,
                                    scale_ab);
                        }
                        else
                        {
                            f_acc[c] = rocm_native_quantised_commit_single_scale_block(
                                f_acc[c], block_acc, block_d, scale_ab);
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


__device__ __forceinline__ float nativeM1FoldColumnValue(const float* partials,int n,int N,int kblocks) {
        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < kblocks; ++kb)
            sum = rocm_native_quantised_accumulate_rn(
                sum,
                partials[static_cast<size_t>(kb) * N + n]);

 return sum;
}
template <int TILE_N, int CPT>
__device__ __forceinline__ void nativeM1FoldColumns(const float *d_partial_fp32,
    float *d_C_fp32,int N,int kblocks,unsigned tile,unsigned lane)
{
    const int n_base = tile * TILE_N + lane * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N) return;

        const float sum=nativeM1FoldColumnValue(d_partial_fp32,n,N,kblocks);
        d_C_fp32[n] = sum;
    }
}




/**
 * @brief Use the canonical descriptor type for ordinary and resident helpers.
 *
 * The original TU maintained a byte-identical global mirror. A canonical alias
 * makes pointers from typed prepared resident arguments and runtime descriptor
 * views the SAME C++ type; no aliasing cast or per-thread descriptor copy is
 * required. Pin every original byte offset as well as the existing expert/bank
 * ABI assertions below so this change cannot alter native launch/runtime layout.
 */
using DeviceNativeQuantisedMatrixDesc=llaminar2::DeviceNativeQuantisedMatrixDesc;
static_assert(sizeof(DeviceNativeQuantisedMatrixDesc)==56 && alignof(DeviceNativeQuantisedMatrixDesc)==8);
static_assert(offsetof(DeviceNativeQuantisedMatrixDesc,payload)==0 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,scales)==8 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,mins)==16 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,emins)==24 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,n)==32 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,k)==36 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,blocks_per_row)==40 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,codebook_id)==44 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,allocation_payload_bytes_per_block)==45 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,allocation_has_mins)==46 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,allocation_has_emins)==47 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,source_codebook_id)==48 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,source_is_superblock)==49 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,source_identity_present)==50 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,reserved)==51 &&
    offsetof(DeviceNativeQuantisedMatrixDesc,reserved_tail)==52,
    "Canonical NativeQuantised descriptor must retain every original ROCm ABI field");

constexpr int kDeviceMoEMaxExperts = llaminar2::moe_runtime_abi::kMaxExperts;
constexpr int kDeviceMoEMaxTopK = 16;
constexpr uint32_t kDeviceMoEFlagValid = 1u << 0;
constexpr uint32_t kDeviceMoEFlagResident = 1u << 1;
constexpr uint32_t kDeviceMoEFlagLocalCompute = 1u << 4;

struct DeviceMoEExpertDescriptorView
{
    DeviceNativeQuantisedMatrixDesc gate;
    DeviceNativeQuantisedMatrixDesc up;
    DeviceNativeQuantisedMatrixDesc down;
    llaminar2::DeviceMoEFloatingMatrixDesc floating_gate;
    llaminar2::DeviceMoEFloatingMatrixDesc floating_up;
    llaminar2::DeviceMoEFloatingMatrixDesc floating_down;
    int32_t logical_expert_id;
    int32_t owner_participant;
    int32_t local_slot;
    uint32_t flags;
    llaminar2::DeviceMoEWeightFormat weight_format;
    llaminar2::DeviceMoEWeightFormat floating_allocation_format;
    llaminar2::DeviceMoEProjectionSet projection_set;
};

static_assert(
    sizeof(DeviceMoEExpertDescriptorView) == 240,
    "ROCm GEMV expert descriptor view must match the shared runtime ABI");

static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_gate) == 168);
static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_up) == 184);
static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_down) == 200);
static_assert(offsetof(DeviceMoEExpertDescriptorView, logical_expert_id) == 216);
static_assert(offsetof(DeviceMoEExpertDescriptorView, weight_format) == 232);
static_assert(offsetof(DeviceMoEExpertDescriptorView, floating_allocation_format) == 234);
static_assert(offsetof(DeviceMoEExpertDescriptorView, projection_set) == 236);

struct DeviceMoEPlacementBankView
{
    DeviceMoEExpertDescriptorView experts[kDeviceMoEMaxExperts];
    uint8_t local_compute_mask[kDeviceMoEMaxExperts];
    uint8_t replica_role[kDeviceMoEMaxExperts];
    uint32_t resident_participant_mask[kDeviceMoEMaxExperts];
    int32_t overlay_route_participant[kDeviceMoEMaxExperts];
    uint32_t epoch;
    uint32_t expert_count;
    uint32_t reserved[2];
};

static_assert(
    sizeof(DeviceMoEPlacementBankView) ==
    llaminar2::moe_runtime_abi::kPlacementBankBytes);
static_assert(
    offsetof(DeviceMoEPlacementBankView, overlay_route_participant) ==
    llaminar2::moe_runtime_abi::kOverlayRouteParticipantOffset);

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
    uint64_t prefill_histogram[kDeviceMoEMaxExperts];
    uint64_t prefill_local_histogram[kDeviceMoEMaxExperts];
    uint64_t grouped_verifier_histogram[kDeviceMoEMaxExperts];
    uint64_t grouped_verifier_local_histogram[kDeviceMoEMaxExperts];
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
    uint32_t current_batch_llep_transient_bank_active;
    llaminar2::moe_runtime_abi::DeviceMoERuntimeHistogramBank
        *runtime_histogram_banks;
    const uint32_t *runtime_histogram_active_bank;
    const llaminar2::DeviceMoEOverlayEpochTicket *overlay_epoch_ticket;
    const DeviceMoEPlacementBankView *overlay_placement_banks;
    const llaminar2::DeviceMoEOverlayEpochStatus *overlay_epoch_status;
    llaminar2::DeviceMoEIntermediateShardBinding intermediate_shard;
};

static_assert(offsetof(DeviceMoELayerRuntimeView, intermediate_shard) ==
    llaminar2::moe_runtime_abi::kIntermediateShardBindingOffset);
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
static_assert(
    offsetof(DeviceMoELayerRuntimeView,
             current_batch_llep_transient_bank_active) ==
    llaminar2::moe_runtime_abi::
        kCurrentBatchLLEPTransientBankActiveOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, runtime_histogram_banks) ==
    llaminar2::moe_runtime_abi::kRuntimeHistogramBanksOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, runtime_histogram_active_bank) ==
    llaminar2::moe_runtime_abi::kRuntimeHistogramActiveBankOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, overlay_epoch_ticket) ==
    llaminar2::moe_runtime_abi::kOverlayEpochTicketOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, overlay_placement_banks) ==
    llaminar2::moe_runtime_abi::kOverlayPlacementBanksOffset);
static_assert(
    offsetof(DeviceMoELayerRuntimeView, overlay_epoch_status) ==
    llaminar2::moe_runtime_abi::kOverlayEpochStatusOffset);

/** Sentinel returned when a runtime inference bank is not provable. */
constexpr uint32_t kInvalidMoERuntimeExecutionBank = 0xffffffffu;

/** @return Canonical durable placement banks for this runtime role. */
__device__ __forceinline__ const DeviceMoEPlacementBankView *
moe_runtime_durable_placement_banks(const DeviceMoELayerRuntimeView *runtime)
{
    return runtime && runtime->overlay_placement_banks
               ? runtime->overlay_placement_banks
               : (runtime ? runtime->banks : nullptr);
}

/** @return Placement banks visible to this exact captured child graph. */
__device__ __forceinline__ const DeviceMoEPlacementBankView *
moe_runtime_placement_banks(const DeviceMoELayerRuntimeView *runtime)
{
    if (!runtime)
        return nullptr;
    return runtime->current_batch_llep_transient_bank_active != 0u
               ? runtime->banks
               : moe_runtime_durable_placement_banks(runtime);
}

/**
 * @brief Select the request-pinned runtime bank without following maintenance.
 *
 * A null ticket is the explicit non-overlay path.  ExpertOverlay readers use
 * the ticket's bank only when its positive generation and exact epoch agree
 * with that bank's descriptor publication.
 */
__device__ __forceinline__ uint32_t moe_runtime_execution_bank(
    const DeviceMoELayerRuntimeView *runtime)
{
    if (!runtime)
        return kInvalidMoERuntimeExecutionBank;
    if (!runtime->overlay_epoch_ticket)
    {
        return runtime->active_bank <= 1u
                   ? runtime->active_bank
                   : kInvalidMoERuntimeExecutionBank;
    }

    const uint64_t epoch = runtime->overlay_epoch_ticket->epoch;
    const uint64_t selector = runtime->overlay_epoch_ticket->selector;
    const uint32_t bank = static_cast<uint32_t>(selector & 1u);
    const uint64_t generation = selector >> 1u;
    const DeviceMoEPlacementBankView *placement_banks =
        moe_runtime_durable_placement_banks(runtime);
    if (epoch == 0u || generation == 0u || bank > 1u ||
        !placement_banks ||
        static_cast<uint64_t>(placement_banks[bank].epoch) != epoch)
    {
        return kInvalidMoERuntimeExecutionBank;
    }
    if (runtime->current_batch_llep_transient_bank_active != 0u)
    {
        const uint32_t transient_bank = runtime->active_bank;
        if (!runtime->overlay_placement_banks || transient_bank > 1u ||
            static_cast<uint64_t>(runtime->banks[transient_bank].epoch) !=
                epoch)
        {
            return kInvalidMoERuntimeExecutionBank;
        }
        return transient_bank;
    }
    return bank;
}

/** @brief Authenticate original fragment execution before any direct runtime GEMV descriptor load. */
__device__ __forceinline__ bool moe_runtime_native_descriptor_selected(
    const DeviceMoELayerRuntimeView *runtime, const DeviceMoEPlacementBankView &bank, int expert)
{
    if (!runtime->intermediate_shard.enabled())
    {
        if (bank.experts[expert].projection_set != llaminar2::DeviceMoEProjectionSet::IntermediateShard)
            return true;
    }
    else
    {
        const auto selection = llaminar2::selectMoEIntermediateShardRow(runtime->intermediate_shard,
            bank, expert, static_cast<int>(runtime->participant_id));
        if (selection.disposition != llaminar2::MoEIntermediateShardRowDisposition::Invalid)
            return selection.disposition == llaminar2::MoEIntermediateShardRowDisposition::Local;
    }
    printf("native_quantised_runtime_fragment_invalid participant=%u expert=%d\n", runtime->participant_id, expert);
    __builtin_trap();
    return false;
}

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ bool native_quantised_desc_shape_ok(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int N,
    int K)
{
    using Traits = NATIVE_QUANTISEDTraits<FMT>;
    return desc.payload &&
           desc.scales &&
           desc.n == N &&
           desc.k == K &&
           desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
           (desc.codebook_id == static_cast<uint8_t>(FMT) ||
            (FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE && desc.codebook_id == NATIVE_QUANTISED_IQ4_NL)) &&
           (!(Traits::requires_secondary_metadata && desc.codebook_id != NATIVE_QUANTISED_IQ4_NL) || desc.mins) &&
           (!Traits::is_dual_scale_asym || desc.emins);
}



template <NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t native_quantised_activation_sum_a(
    const int32_t* __restrict__ a4)
{
    using Traits = NATIVE_QUANTISEDTraits<FMT>;
    int32_t sum_a = 0;
    if constexpr (Traits::is_asymmetric)
    {
        constexpr int32_t ones = 0x01010101;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            sum_a = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a);
    }
    return sum_a;
}

/**
 * @brief Commit one completed single-scale integer dot using the original FP32 order.
 * @tparam FMT Original single-scale decoder.
 * @param f_acc Original running partition accumulator.
 * @param block_acc Exact complete32-code signed integer dot.
 * @param scale_bits Original primary metadata bits. @param secondary_bits Original secondary metadata bits.
 * @param scale_ab Original activation scale. @param sum_a Exact activation code sum.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_commit_single_dot_bits(float& f_acc,int32_t block_acc,
    uint16_t scale_bits,uint16_t secondary_bits,float scale_ab,int32_t sum_a)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    static_assert(!Traits::is_dual_scale);
    const float block_d = rocm_native_quantised_load_single_weight_scale<FMT>(
        scale_bits, (FMT == NATIVE_QUANTISED_IQ3_S || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE) ? secondary_bits : uint16_t{0});
    if constexpr (Traits::is_asymmetric)
    {
        const uint16_t min_bits = secondary_bits;
        const float block_m = __half2float(*reinterpret_cast<const __half*>(&min_bits));
        f_acc = rocm_native_quantised_commit_corrected_single_scale_block(
            f_acc,
            block_acc,
            block_d,
            sum_a,
            block_m,
            scale_ab);
    }
    else
    {
        f_acc = rocm_native_quantised_commit_single_scale_block(
            f_acc, block_acc, block_d, scale_ab);
    }
}

/**
 * @brief Preserve the original pointer API while sharing one scalar-bit commit body.
 * @tparam FMT Original single-scale decoder.
 * @param f_acc Original running partition accumulator. @param block_acc Exact integer dot.
 * @param scale_base Original primary plane. @param min_base Original secondary plane.
 * @param linear Original metadata index. @param scale_ab Original activation scale.
 * @param sum_a Exact signed activation sum.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_commit_single_dot(float& f_acc,int32_t block_acc,
    const uint16_t* scale_base,const uint16_t* min_base,size_t linear,float scale_ab,int32_t sum_a)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    const uint16_t primary=scale_base[linear];
    const uint16_t secondary=Traits::requires_secondary_metadata?min_base[linear]:uint16_t{0};
    native_quantised_small_m_commit_single_dot_bits<FMT>(f_acc,block_acc,primary,secondary,scale_ab,sum_a);
}

/**
 * @brief Construct the original IQ1_M signed one-eighth from its source sign bit.
 * @tparam SignBit Original sign position, three or seven, in the packed qh byte.
 * @param qh Original packed high/sign byte; all unrelated bits are ignored.
 * @return Bit-identical positive or negative0.125f, without FP selection/arithmetic.
 * @details Keep the exact IEEE binary32 exponent/mantissa as integer bits and
 * insert only the original sign. This avoids retaining both FP constants
 * across unrelated format phases; the canonical commit order is unchanged.
 */
template<unsigned SignBit>
__device__ __forceinline__ float native_quantised_iq1_m_delta_bits(uint8_t qh)
{
    static_assert(SignBit==3 || SignBit==7);
    const uint32_t bits=0x3e000000u|((uint32_t(qh)&(1u<<SignBit))<<(31-SignBit));
    return __builtin_bit_cast(float,bits);
}

/**
 * @brief Commit original integer dots with the single canonical FP32 arithmetic body.
 * @tparam FMT Original independent physical decoder.
 * @param f_acc Original increasing-K32 partition sum.
 * @param a4 Original activation words, used only for the unchanged correction terms.
 * @param acc_lo Original groups0..3 dot, or groups0..7 dot for single-scale formats.
 * @param acc_hi Original groups4..7 dot for dual-scale formats, otherwise zero.
 * @param payload Original packed block, retaining IQ1_M correction bits.
 * @param primary_bits Original primary half bits. @param secondary_bits Original secondary half bits.
 * @param extended_bits Original packed extended minima, zero when the format has none.
 * @param scale_ab Original activation scale. @param sum_a Exact signed activation sum.
 *
 * This helper intentionally mirrors the M=1 decode kernel's FP32 addition
 * order.  Do not combine the base dot product and asymmetric/dual-scale
 * correction into a temporary before adding to `f_acc`: the grouped verifier
 * path is compared against serial replay, and tiny associativity differences
 * can amplify through Qwen3.6 recurrent state.
 */
template <NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_commit_integer_dots_bits(
    float& f_acc,
    const int32_t* __restrict__ a4,
    int32_t acc_lo,
    int32_t acc_hi,
    const uint8_t* __restrict__ payload,
    uint16_t primary_bits,
    uint16_t secondary_bits,
    uint32_t extended_bits,
    float scale_ab,
    int32_t sum_a)
{
    using Traits = NATIVE_QUANTISEDTraits<FMT>;

    if constexpr (Traits::is_dual_scale)
    {
        const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
            primary_bits, secondary_bits);
        const float scale_lo = weight_scales.low;
        const float scale_hi = weight_scales.high;
        // Corrected formats form one complete block contribution before the
        // row update. This is the same publication boundary used by CUDA and
        // prevents a backend-specific extra rounding against the running row.
        if constexpr (Traits::is_dual_scale_asym)
        {
            constexpr int32_t ones = 0x01010101;
            int32_t sum_a_lo = 0;
            int32_t sum_a_hi = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g)
                sum_a_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a_lo);
            #pragma unroll
            for (int g = 4; g < 8; ++g)
                sum_a_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a_hi);

            float min_lo;
            float min_hi;
            if constexpr (Traits::needs_emin_array)
            {
                const uint32_t emin_packed = extended_bits;
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
            f_acc = rocm_native_quantised_commit_corrected_dual_scale_block(
                f_acc,
                acc_lo,
                acc_hi,
                scale_lo,
                scale_hi,
                sum_a_lo,
                sum_a_hi,
                min_lo,
                min_hi,
                scale_ab);
        }

        else if constexpr (FMT == NATIVE_QUANTISED_IQ1_M)
        {
            constexpr int32_t ones = 0x01010101;
            const uint8_t qh0 = payload[4];
            const uint8_t qh1 = payload[5];

            const int32_t sa_g0 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[1], ones, ::llaminar2::rocm_integer_contract::signedDot4(a4[0], ones, 0));
            const int32_t sa_g1 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[3], ones, ::llaminar2::rocm_integer_contract::signedDot4(a4[2], ones, 0));
            const int32_t sa_g2 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[5], ones, ::llaminar2::rocm_integer_contract::signedDot4(a4[4], ones, 0));
            const int32_t sa_g3 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[7], ones, ::llaminar2::rocm_integer_contract::signedDot4(a4[6], ones, 0));

            const float d0 = native_quantised_iq1_m_delta_bits<3>(qh0);
            const float d1 = native_quantised_iq1_m_delta_bits<7>(qh0);
            const float d2 = native_quantised_iq1_m_delta_bits<3>(qh1);
            const float d3 = native_quantised_iq1_m_delta_bits<7>(qh1);

            f_acc = rocm_native_quantised_commit_iq1_m_block(
                f_acc,
                acc_lo,
                acc_hi,
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
        else
        {
            f_acc = rocm_native_quantised_commit_dual_scale_block(
                f_acc,
                acc_lo,
                acc_hi,
                scale_lo,
                scale_hi,
                scale_ab);
        }
    }
    else
    {
        native_quantised_small_m_commit_single_dot_bits<FMT>(
            f_acc,acc_lo,primary_bits,secondary_bits,scale_ab,sum_a);
    }
}

/**
 * @brief Accumulate the original decoded array then delegate to the shared FP32 commit.
 * @tparam FMT Original physical decoder.
 * @param f_acc Increasing-K32 partition sum. @param a4 Original activation words.
 * @param packed_groups Original decoded groups0..7. @param payload Original IQ correction bits.
 * @param primary_bits Original primary metadata. @param secondary_bits Original secondary metadata.
 * @param extended_bits Original extended minima. @param scale_ab Original activation scale.
 * @param sum_a Original signed activation sum.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_accumulate_decoded_block_bits(
    float& f_acc,const int32_t* __restrict__ a4,const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,uint16_t primary_bits,uint16_t secondary_bits,
    uint32_t extended_bits,float scale_ab,int32_t sum_a)
{
    int32_t acc_lo=0,acc_hi=0;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::is_dual_scale)
    {
        #pragma unroll
        for(unsigned group=0;group<4;++group)
            acc_lo=::llaminar2::rocm_integer_contract::signedDot4(a4[group],packed_groups[group],acc_lo);
        #pragma unroll
        for(unsigned group=4;group<8;++group)
            acc_hi=::llaminar2::rocm_integer_contract::signedDot4(a4[group],packed_groups[group],acc_hi);
    }
    else
    {
        #pragma unroll
        for(unsigned group=0;group<8;++group)
            acc_lo=::llaminar2::rocm_integer_contract::signedDot4(a4[group],packed_groups[group],acc_lo);
    }
    native_quantised_small_m_commit_integer_dots_bits<FMT>(f_acc,a4,acc_lo,acc_hi,payload,
        primary_bits,secondary_bits,extended_bits,scale_ab,sum_a);
}

/**
 * @brief Preserve every original decoded-block caller with one shared scalar-bit arithmetic body.
 * @tparam FMT Original independent physical format.
 * @param f_acc Running original partition sum. @param a4 Original eight activation words.
 * @param packed_groups Original decoded weight words. @param payload Original packed block.
 * @param scale_base Primary metadata plane. @param min_base Secondary metadata plane.
 * @param emin_base Extended metadata plane. @param linear Original K32-major index.
 * @param scale_ab Original activation scale. @param sum_a Exact original activation sum.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_accumulate_decoded_block(
    float& f_acc,const int32_t* __restrict__ a4,const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,const uint32_t* __restrict__ emin_base,
    size_t linear,float scale_ab,int32_t sum_a)
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    const uint16_t primary=scale_base[linear];
    const uint16_t secondary=Traits::requires_secondary_metadata?min_base[linear]:uint16_t{0};
    const uint32_t extended=Traits::needs_emin_array?emin_base[linear]:uint32_t{0};
    native_quantised_small_m_accumulate_decoded_block_bits<FMT>(f_acc,a4,packed_groups,payload,
        primary,secondary,extended,scale_ab,sum_a);
}

/**
 * Adds one row/block contribution for the generic small-M native-quantised GEMV.
 *
 * Keeping this as a tiny helper lets M=2 spell out row 0 and row 1 directly
 * without reintroducing codebook-specific kernels.  The helper still uses the
 * exact serial-decode accumulation contract from
 * `native_quantised_small_m_accumulate_decoded_block()`.
 */

