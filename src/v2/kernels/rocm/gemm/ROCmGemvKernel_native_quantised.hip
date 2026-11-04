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
#include "kernels/rocm/gemm/ROCmNativeQuantisedDecodeDispatchGenerated.inc"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "ROCmNativeQuantisedNibbleLUT.h"
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

namespace llaminar2::rocm
{
namespace { thread_local GateUpProducerLoads gate_up_producer_loads=GateUpProducerLoads::Original; }
/** @copydoc currentGateUpProducerLoads */
GateUpProducerLoads currentGateUpProducerLoads() noexcept {return gate_up_producer_loads;}
/** @copydoc ScopedGateUpProducerLoads::ScopedGateUpProducerLoads */
ScopedGateUpProducerLoads::ScopedGateUpProducerLoads(GateUpProducerLoads loads):previous_(gate_up_producer_loads)
{
    if(llaminar2::isGraphCaptureActive())throw std::logic_error("Gate/up operand loads must be selected before capture");
    if(loads!=GateUpProducerLoads::Original&&loads!=GateUpProducerLoads::PairedK32)
        throw std::invalid_argument("Unknown gate/up operand load policy");
    gate_up_producer_loads=loads;
}
/** @copydoc ScopedGateUpProducerLoads::~ScopedGateUpProducerLoads */
ScopedGateUpProducerLoads::~ScopedGateUpProducerLoads(){gate_up_producer_loads=previous_;}


}

// ============================================================================
// Runtime tuning overrides for native-quantised GEMV dispatch.
// Values < 0 mean "use default heuristic".
// ============================================================================
static std::atomic<int> g_native_quantised_kb_override{-1};
static std::atomic<int> g_native_quantised_target_waves_override{-1};
/*
 * Decode-equivalent dispatch is a property of one host worker's verifier
 * transaction. LocalTP drives one worker per GPU in the same process, so a
 * process-global mode bit lets either worker clear the other worker's policy
 * while it is still lowering a grouped launch. Keep this state thread-local,
 * matching CUDA, and pass resolved serial-M1 launch geometry explicitly from
 * ROCmQuantisedGemmKernel whenever a grouped projection is emitted.
 */
static thread_local int g_native_quantised_decode_equivalent_m1_config = 0;
static thread_local int g_native_quantised_serial_partition_n = 0;

namespace
{
/** One lowering thread owns its diagnostic scope; no process-global geometry switch. */
thread_local const llaminar2::rocm::ScopedNativeQ8SerialGeometry *g_native_q8_serial_scope = nullptr;
/** Immutable topology observation, not device execution state or a replay counter. */
thread_local std::optional<llaminar2::rocm::NativeQ8SerialLaunch> g_native_q8_serial_launch;
}

void llaminar2::rocm::validateNativeQ8SerialScopeBinding(int device_ordinal, void *stream)
{
    if (!g_native_q8_serial_scope) return;
    if (device_ordinal != g_native_q8_serial_scope->admission().deviceOrdinal() ||
        !stream || stream == static_cast<void *>(hipStreamLegacy) ||
        stream == static_cast<void *>(hipStreamPerThread))
    {
        g_native_q8_serial_launch.reset();
        throw std::logic_error("native-Q8 serial geometry requires its admitted device and exact nondefault stream");
    }
}

void llaminar2::rocm::clearNativeQ8SerialLaunch() noexcept
{
    g_native_q8_serial_launch.reset();
}

std::optional<llaminar2::rocm::NativeQ8SerialLaunch>
llaminar2::rocm::lastNativeQ8SerialLaunch() noexcept
{
    return g_native_q8_serial_launch;
}

std::optional<llaminar2::rocm::NativeQ8SerialGeometry>
llaminar2::rocm::currentNativeQ8SerialGeometry() noexcept
{
    return g_native_q8_serial_scope
        ? std::optional{g_native_q8_serial_scope->admission().geometry()} : std::nullopt;
}

llaminar2::rocm::ScopedNativeQ8SerialGeometry::ScopedNativeQ8SerialGeometry(
    const NativeQ8SerialGeometryAdmission &admission)
    : admission_(admission), previous_(g_native_q8_serial_scope)
{
    if (llaminar2::isGraphCaptureActive())
        throw std::logic_error("native-Q8 serial geometry must be scoped before capture");
    const auto &env = llaminar2::debugEnv().rocm;
    if (g_native_quantised_kb_override.load(std::memory_order_relaxed) > 0 ||
        g_native_quantised_target_waves_override.load(std::memory_order_relaxed) > 0 ||
        env.native_quantised_gemv_kb > 0 || env.native_quantised_gemv_target_waves > 0 ||
        env.native_quantised_disable_generated)
        throw std::logic_error("native-Q8 serial geometry requires unchanged generated arithmetic, without KB/TW overrides");
    (void)nativeQ8SerialGeometrySpec(admission_.geometry());
    g_native_q8_serial_scope = this;
}

llaminar2::rocm::ScopedNativeQ8SerialGeometry::~ScopedNativeQ8SerialGeometry() noexcept
{
    // A scope belongs to one construction thread and must retire in LIFO order.
    if (g_native_q8_serial_scope != this)
        std::terminate();
    g_native_q8_serial_scope = previous_;
}


struct NativeQuantisedDecodeGeneratedRuntimeConfig
{
    bool has_generated = false;
    int kb = -1;
    int target_waves_per_cu = -1;
};

/**
 * @brief Select the historical policy for a self-describing execution format.
 *
 * Codebook 23 is a migration-only physical representation and therefore has
 * no independent tuning corpus. Legacy one-ID callers retain the prior
 * codebook-19 policy. Production callers carrying source provenance use the
 * explicit two-ID entrypoints below and never consult this default.
 */
static constexpr uint8_t defaultNativeQuantisedArithmeticPolicyCodebook(
    uint8_t execution_codebook_id)
{
    return execution_codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook ||
                   execution_codebook_id == llaminar2::kNativeQuantisedMXFP4Codebook
               ? static_cast<uint8_t>(4)
               : execution_codebook_id ==
                   llaminar2::kNativeQuantisedExpandedInt8MinCodebook
               ? static_cast<uint8_t>(19)
               : execution_codebook_id;
}

/**
 * @brief Resolve generated launch policy from an exact arithmetic codebook.
 *
 * The caller has already separated physical decoding from arithmetic policy.
 * This function must not canonicalize or infer another codebook: doing so
 * would make a CPU-cold promotion change its FP32 reduction tree.
 */
static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
    uint8_t arithmetic_policy_codebook_id,
    int m,
    int n,
    int k)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig resolved{};
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    if (rocm_env.native_quantised_disable_generated)
        return resolved;

    const bool has_explicit_override =
        g_native_quantised_kb_override.load(std::memory_order_relaxed) > 0 ||
        g_native_quantised_target_waves_override.load(std::memory_order_relaxed) > 0 ||
        rocm_env.native_quantised_gemv_kb > 0 ||
        rocm_env.native_quantised_gemv_target_waves > 0;
    if (has_explicit_override)
        return resolved;

    const int lookup_m =
        (g_native_quantised_decode_equivalent_m1_config != 0 &&
         m > 1)
            ? 1
            : m;
    const int lookup_n =
        g_native_quantised_serial_partition_n > 0
            ? g_native_quantised_serial_partition_n
            : n;

    /*
     * Generated policy is immutable for the process lifetime. Preserve each
     * input as a full-width word so invalid or future geometry cannot alias the
     * compact exact-overlay ABI. Caching a miss keeps totality checks cheap but
     * never manufactures a runtime fallback.
     */
    using CacheKey = std::array<uint64_t, 4>;
    using Cache = llaminar2::native_quantised::FixedDispatchCache<
        CacheKey,
        NativeQuantisedDecodeGeneratedRuntimeConfig,
        64,
        llaminar2::native_quantised::DispatchCacheArrayHasher<4>>;
    static thread_local Cache cache;
    const CacheKey cache_key = {
        static_cast<uint64_t>(arithmetic_policy_codebook_id),
        static_cast<uint64_t>(static_cast<uint32_t>(lookup_m)),
        static_cast<uint64_t>(static_cast<uint32_t>(lookup_n)),
        static_cast<uint64_t>(static_cast<uint32_t>(k)),
    };
    bool cached_selector_found = false;
    NativeQuantisedDecodeGeneratedRuntimeConfig cached{};
    if (cache.lookup(cache_key, cached_selector_found, cached))
        return cached;

    llaminar2::rocm::generated::ROCmNativeQuantisedDecodeDispatchConfig generated_cfg{};
    if (!llaminar2::rocm::generated::selectROCmNativeQuantisedDecodeGenerated(
            arithmetic_policy_codebook_id,
            lookup_m,
            lookup_n,
            k,
            generated_cfg))
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

/** @brief Resolve policy for a legacy self-describing execution codebook. */
static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedDecodeGeneratedRuntimeConfig(
    uint8_t execution_codebook_id,
    int m,
    int n,
    int k)
{
    return resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
        defaultNativeQuantisedArithmeticPolicyCodebook(execution_codebook_id),
        m,
        n,
        k);
}

static void recordNativeQuantisedSmallMLaunch(
    int m,
    int n,
    int k,
    unsigned codebook,
    int grid_n,
    int kb,
    int target_waves_per_cu,
    bool batched,
    int projections,
    int arithmetic_policy_codebook = -1)
{
    if (!llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
        return;

    const char* path = (kb <= 1) ? "direct" : "split_reduce";
    llaminar2::PerfStatsCollector::Tags tags{
        {"m", std::to_string(m)}, {"n", std::to_string(n)},
        {"k", std::to_string(k)}, {"codebook", std::to_string(codebook)},
        {"arithmetic_policy_codebook", std::to_string(arithmetic_policy_codebook >= 0
            ? arithmetic_policy_codebook : static_cast<int>(codebook))},
        {"grid_n", std::to_string(grid_n)}, {"kb", std::to_string(kb)},
        {"path", path}, {"target_waves_per_cu", std::to_string(target_waves_per_cu)},
        {"batched", batched ? "true" : "false"},
        {"projections", std::to_string(projections)}};
    if (m == 1 && codebook == 19 && !batched && g_native_q8_serial_launch)
    {
        const auto &launch = *g_native_q8_serial_launch;
        const auto spec = llaminar2::rocm::nativeQ8SerialGeometrySpec(launch.geometry);
        tags.emplace("serial_tile_n", std::to_string(spec.tile_columns));
        tags.emplace("serial_cpt", std::to_string(spec.columns_per_thread));
        tags.emplace("serial_block_threads", std::to_string(launch.producer.block_x));
        tags.emplace("serial_arithmetic_policy_n", std::to_string(launch.arithmetic_policy_n));
        tags.emplace("serial_blocks_per_partition", std::to_string(launch.blocks_per_partition));
        tags.emplace("serial_nonempty_partitions", std::to_string(launch.nonempty_partitions));
        tags.emplace("serial_producer", launch.producer.specialization);
        tags.emplace("serial_reducer", launch.reducer.function ? launch.reducer.specialization : "absent");
    }
    llaminar2::PerfStatsCollector::addCounter(
        "kernel", "rocm_native_quantised_small_m_launch", 1.0, "gemm", {}, std::move(tags));
}

extern "C" bool rocmQuantGemm_biasAdd(
    float *d_output,
    const float *d_bias,
    int M, int N,
    int rocm_device_id, void *stream);

extern "C" void rocmGemv_native_quantised_set_tuning_overrides(int kb, int target_waves_per_cu)
{
    g_native_quantised_kb_override.store(kb, std::memory_order_relaxed);
    g_native_quantised_target_waves_override.store(target_waves_per_cu, std::memory_order_relaxed);
}

extern "C" void rocmGemv_native_quantised_set_decode_equivalent_m1_config(int enabled)
{
    g_native_quantised_decode_equivalent_m1_config = enabled ? 1 : 0;
}

extern "C" void rocmGemv_native_quantised_set_serial_partition_n(int n)
{
    g_native_quantised_serial_partition_n = n;
}

extern "C" int rocmGemv_native_quantised_get_serial_partition_n()
{
    return g_native_quantised_serial_partition_n;
}

extern "C" void rocmGemv_native_quantised_reset_tuning_overrides()
{
    rocmGemv_native_quantised_set_tuning_overrides(-1, -1);
}

static bool nativeQuantisedDecodeEquivalentSmallMMode()
{
    return g_native_quantised_decode_equivalent_m1_config != 0;
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
static bool validateNativeQuantisedDecodeEquivalentKBOverride(
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
// Format tags for native-quantised kernel dispatch
//
// Each value is the codebook_id stored in ROCmPackedWeights::native_quantised_codebook_id
// and passed to rocmGemv_native_quantised_fp32() for template selection.
// ============================================================================
enum NativeQuantisedFormat : uint8_t {
    NATIVE_QUANTISED_Q4_0   = 0,   // 4-bit symmetric, linear decode (q - 8)
    NATIVE_QUANTISED_IQ4_NL = 4,   // IQ4_NL original half scale
    NATIVE_QUANTISED_IQ4_XS_SOURCE = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook, // Original IQ4_XS tuple
    NATIVE_QUANTISED_MXFP4  = llaminar2::kNativeQuantisedMXFP4Codebook, // OCP MXFP4: IQ4_NL geometry, doubled-E2M1 lookup
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
    static constexpr bool uses_lut      = (FMT == NATIVE_QUANTISED_IQ4_NL || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE || FMT == NATIVE_QUANTISED_MXFP4);
    /// Packed signed lookup words of a uses_lut format (kvalues_iq4nl or doubled E2M1).
    static constexpr llaminar2::rocm::nibble_lut::Words lut =
        uses_lut ? llaminar2::rocm::nibble_lut::words<uses_lut ? static_cast<int>(FMT) : 4>()
                 : llaminar2::rocm::nibble_lut::Words{};
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
        FMT == NATIVE_QUANTISED_Q4_0 || FMT == NATIVE_QUANTISED_IQ4_NL || FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE || FMT == NATIVE_QUANTISED_MXFP4 || FMT == NATIVE_QUANTISED_Q4_1 ||
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
static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig resolved =
        resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, 1, N, K);
    if (!resolved.has_generated)
        return resolved;

    if (llaminar2::MoEProjectionNumericalContract::
            ownsOrderedKPartitionTree(arithmetic_policy_codebook_id))
    {
        resolved.kb = llaminar2::MoEProjectionNumericalContract::
            orderedKPartitionsForWidth(K);
    }
    return resolved;
}

/**
 * @brief Resolve the split-K policy used by normal serial M=1 decode.
 *
 * MTP verifier publication compares grouped runtime-M rows against committed
 * serial decode state. The grouped kernels may execute several rows at once,
 * but they must consume this same typed policy so their FP32 reduction order is
 * bitwise-identical. Returning an empty result on a generated-policy miss
 * forces the caller to fail rather than reconstruct an obsolete heuristic.
 */
static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedSerialM1RuntimeConfig(
    uint8_t codebook_id,
    int N,
    int K)
{
    return resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
        defaultNativeQuantisedArithmeticPolicyCodebook(codebook_id), N, K);
}

static NativeQuantisedDecodeGeneratedRuntimeConfig resolveNativeQuantisedCommonSerialM1RuntimeConfig(
    const uint8_t* codebook_ids,
    const int* Ns,
    int num_projections,
    int K)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig common{};
    for (int p = 0; p < num_projections; ++p)
    {
        NativeQuantisedDecodeGeneratedRuntimeConfig candidate =
            resolveNativeQuantisedSerialM1RuntimeConfig(codebook_ids[p], Ns[p], K);
        if (!candidate.has_generated)
            return NativeQuantisedDecodeGeneratedRuntimeConfig{};

        if (p == 0)
        {
            common = candidate;
            continue;
        }

        if (candidate.kb != common.kb)
            return NativeQuantisedDecodeGeneratedRuntimeConfig{};
    }
    return common;
}

static NativeQuantisedDecodeGeneratedRuntimeConfig resolveNativeQuantisedCommonSerialM1RuntimeConfig(
    uint8_t codebook_id,
    const int* Ns,
    int num_projections,
    int K)
{
    uint8_t codebook_ids[8] = {};
    for (int p = 0; p < num_projections && p < 8; ++p)
        codebook_ids[p] = codebook_id;
    return resolveNativeQuantisedCommonSerialM1RuntimeConfig(codebook_ids, Ns, num_projections, K);
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
 * @param codebook_id NativeQuantised format identifier.
 * @param N Output-column count.
 * @param K Reduction width, divisible by 32.
 * @return Effective K-partition count, or zero when no generated policy exists.
 */
static int resolveNativeQuantisedSerialM1KPartitions(
    uint8_t codebook_id,
    int N,
    int K)
{
    if (N <= 0 || K <= 0 || (K % 32) != 0)
        return 0;

    const NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfig(codebook_id, N, K);
    if (!cfg.has_generated || cfg.kb <= 0)
        return 0;

    int partitions = cfg.kb;
    if (partitions > K / 32)
        partitions = K / 32;
    if (partitions >
        llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        partitions =
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
    }
    return partitions > 0 ? partitions : 1;
}

/**
 * @brief Resolve graph-safe serial partitions from an exact source policy.
 */
static int resolveNativeQuantisedSerialM1KPartitionsForPolicy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K)
{
    if (N <= 0 || K <= 0 || (K % 32) != 0)
        return 0;

    const NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, N, K);
    if (!cfg.has_generated || cfg.kb <= 0)
        return 0;

    int partitions = cfg.kb;
    if (partitions > K / 32)
        partitions = K / 32;
    if (partitions >
        llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        partitions =
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
    }
    return partitions > 0 ? partitions : 1;
}

extern "C" bool rocmGemv_native_quantised_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfig(codebook_id, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
        *kb = resolveNativeQuantisedSerialM1KPartitions(codebook_id, N, K);
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

/**
 * @brief Query serial-M1 geometry for an explicit arithmetic policy.
 *
 * Unlike the legacy query, this entrypoint never maps the migration-only
 * execution codebook to a guessed policy. Callers must provide the canonical
 * source policy codebook retained in prepared-weight provenance.
 */
extern "C" bool rocmGemv_native_quantised_query_serial_m1_config_with_policy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    const NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
    {
        *kb = resolveNativeQuantisedSerialM1KPartitionsForPolicy(
            arithmetic_policy_codebook_id, N, K);
    }
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

/**
 * @brief Immutable serial-M1 split-K geometry available to one captured MoE graph.
 *
 * A dynamic expert-overlay table can contain several physical NativeQuantised
 * codebooks, and a migrated expert can change physical representation without
 * changing its source arithmetic policy.  The captured grouped kernel therefore
 * cannot bake one table-wide codebook or one table-wide partition count into its
 * launch.  This compact value table is copied into the graph's kernel arguments;
 * each routed descriptor selects its own entry on device.
 *
 * Indexes are canonical source-policy codebooks.  Unsupported or untrained
 * entries remain zero and are rejected before capture when the descriptor table
 * advertises them.  `maximum_k_partitions` sizes the stable grid while every
 * route still uses exactly its serial-M1 partition boundaries.
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
static void setROCmMoESerialM1PolicyPartitions(
    ROCmMoESerialM1PolicyTable& table,
    uint8_t policy,
    uint8_t partitions)
{
    if (policy >= ROCmMoESerialM1PolicyTable::kCodebookCapacity)
        return;

    const unsigned shift = static_cast<unsigned>(policy & 7u) * 8u;
    const uint64_t encoded = static_cast<uint64_t>(partitions) << shift;
    switch (policy >> 3u)
    {
    case 0:
        table.k_partitions_0_7 |= encoded;
        break;
    case 1:
        table.k_partitions_8_15 |= encoded;
        break;
    case 2:
        table.k_partitions_16_23 |= encoded;
        break;
    case 3:
        table.k_partitions_24_31 |= encoded;
        break;
    default:
        break;
    }
}

/**
 * @brief Build graph-stable descriptor-local split-K policy for one projection.
 *
 * The generated selector is immutable, as is the descriptor table's source
 * arithmetic-policy mask.  Build entries only for that declared mask: table
 * upload/update rejects a policy-mask change, while compact-to-expanded expert
 * movement retains the source arithmetic policy in descriptor provenance.
 * Consequently a captured graph remains valid across overlay publication
 * without paying grid depth for unrelated codebooks.
 *
 * @param N Output width of the projection.
 * @param K Reduction width of the projection.
 * @param required_policy_mask Policies present in the current host descriptor
 *                             table; every set bit must have generated evidence.
 * @param table Receives the by-policy partition counts and maximum grid depth.
 * @return True when all required policies are represented exactly.
 */
static bool buildROCmMoESerialM1PolicyTable(
    int N,
    int K,
    uint32_t required_policy_mask,
    ROCmMoESerialM1PolicyTable& table)
{
    static constexpr std::array<uint8_t, 16> kCanonicalPolicies = {
        0, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 19};

    table = {};
    uint32_t represented_mask = 0;
    for (const uint8_t policy : kCanonicalPolicies)
    {
        const uint32_t policy_bit =
            uint32_t{1} << static_cast<uint32_t>(policy);
        if ((required_policy_mask & policy_bit) == 0)
        {
            /*
             * Grid Z is the maximum partition count represented here.  Do not
             * launch empty partition planes for formats the immutable table can
             * never select; each extra plane creates one block per route and
             * output tile in every captured decode replay.
             */
            continue;
        }

        const int partitions =
            resolveNativeQuantisedSerialM1KPartitionsForPolicy(policy, N, K);
        if (partitions <= 0)
            continue;

        setROCmMoESerialM1PolicyPartitions(
            table,
            policy,
            static_cast<uint8_t>(partitions));
        represented_mask |= policy_bit;
        if (partitions > static_cast<int>(table.maximum_k_partitions))
        {
            table.maximum_k_partitions = static_cast<uint8_t>(partitions);
        }
    }

    return table.maximum_k_partitions > 0 &&
           (required_policy_mask & ~represented_mask) == 0;
}

/**
 * @brief Record proof that descriptor-local decode policy was selected.
 *
 * This counter is emitted by the host launcher, including while a graph is
 * being recorded.  It does not inspect device data or synchronize the stream;
 * production campaigns can therefore prove mixed/runtime routing without
 * perturbing the captured execution timeline.
 */
static void recordROCmMoEDescriptorLocalDecode(
    const char* projection,
    const char* descriptor_source,
    uint8_t table_codebook,
    uint32_t policy_codebook_mask,
    const ROCmMoESerialM1PolicyTable& policy_table)
{
    if (!llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
        return;
    llaminar2::PerfStatsCollector::addCounter(
        "kernel",
        "rocm_moe_descriptor_local_decode_calls",
        1.0,
        "moe",
        {},
        {{"projection", projection},
         {"descriptor_source", descriptor_source},
         {"table_codebook", std::to_string(table_codebook)},
         {"policy_codebook_mask", std::to_string(policy_codebook_mask)},
         {"maximum_k_partitions",
          std::to_string(policy_table.maximum_k_partitions)}});
}

extern "C" bool rocmGemv_native_quantised_query_generated_config(
    uint8_t codebook_id,
    int M,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedDecodeGeneratedRuntimeConfig(codebook_id, M, N, K);
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
extern "C" double rocmGemv_native_quantised_measure_generated_config_ns(
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
        const NativeQuantisedDecodeGeneratedRuntimeConfig config =
            resolveNativeQuantisedDecodeGeneratedRuntimeConfig(
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
            uint16_t scale_bits = ::llaminar2::native_quantised_scale::loadStagedPrimary<static_cast<uint8_t>(FMT)>(d_block_scales, linear);

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
                    const uint16_t scale_bits = ::llaminar2::native_quantised_scale::loadStagedPrimary<static_cast<uint8_t>(FMT)>(d_block_scales, linear);

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
                        // The packed words come from Traits::lut (IQ4_NL shown; MXFP4 uses
                        // the doubled E2M1 table). IQ4_NL codebook (uint8 representation):
                        //   0-3:  {0x81, 0x98, 0xAD, 0xBF}  = {-127, -104, -83, -65}
                        //   4-7:  {0xCF, 0xDD, 0xEA, 0xF6}  = {-49, -35, -22, -10}
                        //   8-11: {0x01, 0x0D, 0x19, 0x26}  = {1, 13, 25, 38}
                        //   12-15:{0x35, 0x45, 0x59, 0x71}  = {53, 69, 89, 113}
                        constexpr uint32_t CB_LO_A = Traits::lut.lo_a;  // cb[0..3] (src0 for nibbles 0-7)
                        constexpr uint32_t CB_LO_B = Traits::lut.lo_b;  // cb[4..7] (src1 for nibbles 0-7)
                        constexpr uint32_t CB_HI_A = Traits::lut.hi_a;  // cb[8..11] (src0 for nibbles 8-15)
                        constexpr uint32_t CB_HI_B = Traits::lut.hi_b;  // cb[12..15] (src1 for nibbles 8-15)
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

template <int TILE_N, int CPT, NativeQuantisedFormat FMT, bool HasWork = false>
__global__ void gemv_native_quantised_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations (INT8)
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × payload_bytes] payload
    const uint16_t*  __restrict__ d_block_scales,  // [blocks_per_row × N] FP16 scale (d)
    const uint16_t*  __restrict__ d_block_mins,    // [blocks_per_row × N] FP16 min (m), NULL for symmetric
    const uint32_t*  __restrict__ d_block_emins,   // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, else nullptr)
    float*           __restrict__ d_C_fp32,        // [N] output (FP32), used when kblocks=1
    float*           __restrict__ d_partial_fp32,  // [KB × N] partials, used when kblocks>1
    const float*     __restrict__ d_scale_A,       // [1] per-row activation scale (CK fallback, used when d_scale_A_blockwise==nullptr)
    const float*     __restrict__ d_scale_A_blockwise, // [blocks_per_row] per-block activation scales
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work)
{
    nativeM1ColumnsBody<TILE_N,CPT,FMT,HasWork>(d_A_int8,d_payload,d_block_scales,d_block_mins,d_block_emins,
        d_C_fp32,d_partial_fp32,d_scale_A,d_scale_A_blockwise,N,K,kblocks,work,blockIdx.x,blockIdx.y,threadIdx.x);
}

/**
 * @brief Retain the original eight increasing-word signed dot4 operations.
 * @param a_low Original activation words0..3. @param a_high Original words4..7.
 * @param w_low Original prepared Q8 words0..3. @param w_high Original words4..7.
 * @return Exact original complete K32 integer dot, with no saturation/reassociation.
 */
__device__ __forceinline__ int32_t nativeQ8SerialOrderedDot(
    int4 a_low,int4 a_high,int4 w_low,int4 w_high)
{
    int32_t dot=0;
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.x,w_low.x,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.y,w_low.y,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.z,w_low.z,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.w,w_low.w,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_high.x,w_high.x,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_high.y,w_high.y,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_high.z,w_high.z,dot);
    return ::llaminar2::rocm_integer_contract::signedDot4(a_high.w,w_high.w,dot);
}

/**
 * @brief Prefetch two independent Q8 K32 blocks before their original ascending commits.
 * @tparam TILE_N Existing64/128 independent-column geometry; one column per lane.
 * @tparam HasWork Exact original immutable row-work mode.
 * @details This diagnostic entry changes no decoder, partition, accumulation,
 * destination or work extent. Prefetch values are register-only scalars/int4s;
 * no local aggregate address escapes, LDS/counter is introduced, or native
 * operation is added. The original producer remains a separate unchanged symbol.
 * @param A Original signed-Q8 activation bytes. @param payload Original prepared Q8 bytes.
 * @param scales Original FP16 weight scales. @param output Original direct output whenP1.
 * @param partials Original partition-major FP32 bank whenP>1. @param row_scale Original optional row activation scale.
 * @param block_scales Original optional K32 activation scales. @param N Exact columns. @param K Exact reduction width.
 * @param partitions Unchanged original source-policy partition count. @param work Original borrowed row authority.
 */
template<int TILE_N,bool HasWork>
__global__ __launch_bounds__(TILE_N) void gemv_native_q8_two_k32_lookahead_kernel_t(
    const int8_t* __restrict__ A,const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scales,float* __restrict__ output,float* __restrict__ partials,
    const float* __restrict__ row_scale,const float* __restrict__ block_scales,
    int N,int K,int partitions,llaminar2::ProjectionRowWork work)
{
    static_assert(TILE_N==64||TILE_N==128);
    if constexpr(HasWork)if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
    const unsigned n=blockIdx.x*TILE_N+threadIdx.x;
    if(n>=static_cast<unsigned>(N))return;
    const int partition=blockIdx.y,blocks=K/32;
    const int per=(blocks+partitions-1)/partitions;
    int b=partition*per;
    const int end=min(b+per,blocks);
    float acc=0.0f;
    for(;b+1<end;b+=2)
    {
        const size_t linear0=static_cast<size_t>(b)*N+static_cast<size_t>(n);
        const size_t linear1=static_cast<size_t>(b+1)*N+static_cast<size_t>(n);
        const auto* w0=reinterpret_cast<const int32_t*>(payload+linear0*32);
        const auto* w1=reinterpret_cast<const int32_t*>(payload+linear1*32);
        const int4 w0_low=make_int4(w0[0],w0[1],w0[2],w0[3]);
        const int4 w0_high=make_int4(w0[4],w0[5],w0[6],w0[7]);
        const int4 w1_low=make_int4(w1[0],w1[1],w1[2],w1[3]);
        const int4 w1_high=make_int4(w1[4],w1[5],w1[6],w1[7]);
        const uint16_t scale0=scales[linear0],scale1=scales[linear1];
        const auto* a0=reinterpret_cast<const int32_t*>(A+b*32);
        const auto* a1=reinterpret_cast<const int32_t*>(A+(b+1)*32);
        const int4 a0_low=make_int4(a0[0],a0[1],a0[2],a0[3]);
        const int4 a0_high=make_int4(a0[4],a0[5],a0[6],a0[7]);
        const int4 a1_low=make_int4(a1[0],a1[1],a1[2],a1[3]);
        const int4 a1_high=make_int4(a1[4],a1[5],a1[6],a1[7]);
        const float activation0=block_scales?block_scales[b]:1.0f;
        const float activation1=block_scales?block_scales[b+1]:1.0f;
        const int32_t dot0=nativeQ8SerialOrderedDot(a0_low,a0_high,w0_low,w0_high);
        acc=rocm_native_quantised_commit_single_scale_block(acc,dot0,
            rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scale0,0),activation0);
        // The preceding RN add must remain before this second block's RN add.
        // No pair sum, fused accumulator or partition-dependent reassociation.
        const int32_t dot1=nativeQ8SerialOrderedDot(a1_low,a1_high,w1_low,w1_high);
        acc=rocm_native_quantised_commit_single_scale_block(acc,dot1,
            rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scale1,0),activation1);
    }
    if(b<end)
    {
        const size_t linear=static_cast<size_t>(b)*N+static_cast<size_t>(n);
        const auto* w=reinterpret_cast<const int32_t*>(payload+linear*32);
        const auto* a=reinterpret_cast<const int32_t*>(A+b*32);
        const int32_t dot=nativeQ8SerialOrderedDot(
            make_int4(a[0],a[1],a[2],a[3]),make_int4(a[4],a[5],a[6],a[7]),
            make_int4(w[0],w[1],w[2],w[3]),make_int4(w[4],w[5],w[6],w[7]));
        acc=rocm_native_quantised_commit_single_scale_block(acc,dot,
            rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scales[linear],0),
            block_scales?block_scales[b]:1.0f);
    }
    const float result=acc*(block_scales?1.0f:row_scale[0]);
    if(partitions==1)output[n]=result;
    else partials[static_cast<size_t>(partition)*N+static_cast<size_t>(n)]=result;
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

        if (lane == 0)
            row_scales[b] = scale;

        row_int8[k_start + lane] = static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
                val, scale));
    }
}

/**
 * @brief Decode one canonical Q4 four-code group from one packed nibble word.
 * @tparam FMT Q4_0 or Q4_1 physical decoder.
 * @param raw Original four nibble-pairs, without scale/minimum metadata.
 * @param high Select original elements16..31 rather than0..15.
 * @return Four original signed/unsigned int8 codes packed in one int32.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t moe_decode_q4_group(uint32_t raw,bool high)
{
    static_assert(FMT==NATIVE_QUANTISED_Q4_0 || FMT==NATIVE_QUANTISED_Q4_1);
    const uint32_t nibbles=(high?raw>>4:raw)&0x0F0F0F0Fu;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::center_4bit==8)
        return static_cast<int32_t>((nibbles+0x78787878u)^0x80808080u);
    else return static_cast<int32_t>(nibbles);
}
/**
 * @brief Decode one canonical Q5 four-code group, retaining original centering.
 * @tparam FMT Q5_0 or Q5_1 physical decoder.
 * @param raw Original four nibble-pairs.
 * @param high_bits Original32-bit fifth-bit plane.
 * @param group Original group index0..7; indices4..7 select high nibbles.
 * @return Four original signed/unsigned int8 codes packed in one int32.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t moe_decode_q5_group(uint32_t raw,uint32_t high_bits,unsigned group)
{
    static_assert(FMT==NATIVE_QUANTISED_Q5_0 || FMT==NATIVE_QUANTISED_Q5_1);
    const uint32_t nibbles=(group>=4?raw>>4:raw)&0x0F0F0F0Fu;
    const uint32_t hb4=(high_bits>>(group*4))&0xFu;
    // Share the exact four-bit deposit; keep group selection and Q5 centering
    // unchanged while avoiding four independently live shift/mask terms.
    const uint32_t hb=::llaminar2::native_quantised::q5HighBitsToPackedBytes(hb4);
    uint32_t value=nibbles|hb;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::center_5bit==16)
    {value^=0x80808080u;value-=0x10101010u;value^=0x80808080u;}
    return static_cast<int32_t>(value);
}

/**
 * @brief Load one packed input word at the current ordered integer-dot boundary.
 * @tparam Ordered NativeLayout's scheduling-only raw-input dependency; false for legacy callers.
 * @param payload Original canonical payload. @param offset Original byte offset.
 * @param previous_dot Prior integer dot; never changes loaded bits.
 * @return Original word, with no allocation, fence or numerical operation.
 */
template<bool Ordered>
__device__ __forceinline__ uint32_t moe_native_group_word(const uint8_t* payload,unsigned offset,int32_t previous_dot)
{
    uint32_t raw=*reinterpret_cast<const uint32_t*>(payload+offset);
    if constexpr(Ordered)asm volatile("" : "+v"(raw) : "v"(previous_dot));
    return raw;
}
/**
 * @brief Load one bounded packed byte at the current integer-dot boundary.
 * @tparam Ordered Whether to retain the scheduling-only dependency.
 * @param payload Original payload. @param offset In-bounds original byte offset.
 * @param previous_dot Prior integer dot. @return Original byte widened without sign extension.
 */
template<bool Ordered>
__device__ __forceinline__ uint32_t moe_native_group_byte(const uint8_t* payload,unsigned offset,int32_t previous_dot)
{
    uint32_t raw=payload[offset];
    if constexpr(Ordered)asm volatile("" : "+v"(raw) : "v"(previous_dot));
    return raw;
}
/**
 * @brief Default immutable device-table lookup authority for canonical IQ decode.
 * @details Explicit full-entry views may borrow byte-identical phase-local LDS;
 * all existing owners retain these original module-table reads by default.
 */
struct NativeQuantisedGridLookup {
    /** @brief Read the original four-code IQ3 entry. @tparam FMT Original format. @param index Original decoded index. @return Unchanged packed bytes. */
    template<NativeQuantisedFormat FMT> __device__ __forceinline__ uint32_t iq3(unsigned index) const {return iq3_grid_lookup<FMT>(index);}
    /** @brief Read the original eight-code IQ2 entry. @tparam FMT Original format. @param index Original decoded index. @return Unchanged packed bytes. */
    template<NativeQuantisedFormat FMT> __device__ __forceinline__ uint64_t iq2(unsigned index) const {return iq2_grid_lookup<FMT>(index);}
    /** @brief Read the original IQ1 entry. @param index Original decoded index. @return Unchanged packed bytes. */
    __device__ __forceinline__ uint64_t iq1(unsigned index) const {return iq1_grid_lookup(index);}
};
/**
 * @brief Decode exactly one original four-code group, without expanding future groups.
 * @tparam FMT Canonical physical codebook from the existing supported inventory.
 * @tparam Ordered Apply NativeLayout's register-only dependency to every raw packed input.
 * @tparam Grid Explicit immutable lookup view; existing owners use original device tables.
 * @param payload Original canonical block, including original IQ sign/tail bytes.
 * @param group Original output group0..7. @param previous_dot Prior integer result, unchanged by decode.
 * @param table_view Lookup authority with byte-identical entries, valid for this complete consumer.
 * @return The identical four signed/unsigned int8 codes packed in one integer word.
 * @details The legacy eight-group API delegates here with Ordered=false. Ordered
 * mode delays raw inputs before mapping, rather than serializing already-expanded
 * values. Grid lookups and bit mappings are the original canonical operations.
 */
template<NativeQuantisedFormat FMT,bool Ordered=false,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ int32_t moe_decode_native_quantised_group(
    const uint8_t* __restrict__ payload,unsigned group,int32_t previous_dot=0,Grid table_view={})
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
        "Grouped MoE decode requires a supported native-quantised format");
    if constexpr(Traits::uses_lut)
    {
        constexpr uint32_t CB_LO_A=Traits::lut.lo_a,CB_LO_B=Traits::lut.lo_b;
        constexpr uint32_t CB_HI_A=Traits::lut.hi_a,CB_HI_B=Traits::lut.hi_b;
        constexpr uint32_t MUX_IDENTITY=0x03020100u,MUX_BIT3_MASK=0x04040404u;
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot);
        const uint32_t nibbles=(group>=4?raw>>4:raw)&0x0F0F0F0Fu;
        const uint32_t sel=nibbles&0x07070707u;
        const uint32_t r0=__builtin_amdgcn_perm(CB_LO_B,CB_LO_A,sel);
        const uint32_t r1=__builtin_amdgcn_perm(CB_HI_B,CB_HI_A,sel);
        const uint32_t mux=MUX_IDENTITY|((nibbles>>1)&MUX_BIT3_MASK);
        return static_cast<int32_t>(__builtin_amdgcn_perm(r1,r0,mux));
    }
    else if constexpr(Traits::is_5bit)
    {
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot);
        const uint32_t high=moe_native_group_word<Ordered>(payload,16,previous_dot);
        return moe_decode_q5_group<FMT>(raw,high,group);
    }
    else if constexpr(Traits::is_3bit)
    {
        constexpr uint32_t BIT_SPREAD=0x00204081u;
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group/4)*4,previous_dot);
        const uint32_t high=moe_native_group_word<Ordered>(payload,8,previous_dot);
        const uint32_t low2=(raw>>((group%4)*2))&0x03030303u;
        const uint32_t hb4=(high>>(group*4))&0xFu;
        const uint32_t hb=((hb4*BIT_SPREAD)&0x01010101u)<<2;
        uint32_t value=(low2|hb)^0x80808080u;
        value-=0x04040404u;value^=0x80808080u;
        return static_cast<int32_t>(value);
    }
    else if constexpr(Traits::is_2bit)
    {
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group/4)*4,previous_dot);
        return static_cast<int32_t>((raw>>((group%4)*2))&0x03030303u);
    }
    else if constexpr(Traits::is_6bit)
    {
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot);
        // Q6's24-byte payload has two complete aligned high-bit words.
        // Keep that word, rather than its extracted byte, behind the current
        // dot boundary; byte extraction and dilation then remain group-local.
        const uint32_t high_word=moe_native_group_word<Ordered>(payload,16+(group/4)*4,previous_dot);
        const uint32_t high=(high_word>>((group%4)*8))&0xFFu;
        const uint32_t nibbles=(group>=4?raw>>4:raw)&0x0F0F0F0Fu;
        const uint32_t hb=rocm_native_quantised_expand_q6_high_fields(high);
        uint32_t value=(nibbles|hb)^0x80808080u;
        value-=0x20202020u;value^=0x80808080u;
        return static_cast<int32_t>(value);
    }
    else if constexpr(Traits::is_iq3_grid)
    {
        const uint32_t qs=moe_native_group_word<Ordered>(payload,(group/4)*4,previous_dot);
        const uint32_t signs=moe_native_group_word<Ordered>(payload,8,previous_dot);
        int idx=(qs>>((group%4)*8))&0xFF;
        uint8_t sign_byte;
        if constexpr(Traits::has_qh)
        {
            idx|=static_cast<int>((signs>>group)&1)<<8;
            sign_byte=group<6?static_cast<uint8_t>(signs>>((group/2+1)*8)):
                static_cast<uint8_t>(moe_native_group_byte<Ordered>(payload,12,previous_dot));
        }
        else sign_byte=static_cast<uint8_t>(signs>>((group/2)*8));
        const uint8_t sign_nibble=group&1?sign_byte>>4:sign_byte&0xF;
        return static_cast<int32_t>(iq_apply_signs_4(table_view.template iq3<FMT>(idx),sign_nibble));
    }
    else if constexpr(Traits::is_iq2_grid)
    {
        const unsigned lookup=group/2;
        const uint32_t qs=moe_native_group_word<Ordered>(payload,0,previous_dot);
        const uint32_t signs=moe_native_group_word<Ordered>(payload,4,previous_dot);
        int idx=(qs>>(lookup*8))&0xFF;
        uint8_t sign_byte;
        if constexpr(Traits::has_qh)
        {
            if constexpr(Traits::qh_bits_per_lookup==1)idx|=static_cast<int>((signs>>lookup)&1)<<8;
            else idx|=static_cast<int>((signs>>(2*lookup))&3)<<8;
            sign_byte=lookup<3?static_cast<uint8_t>(signs>>((lookup+1)*8)):
                static_cast<uint8_t>(moe_native_group_byte<Ordered>(payload,8,previous_dot));
        }
        else sign_byte=static_cast<uint8_t>(signs>>(lookup*8));
        const uint64_t grid=table_view.template iq2<FMT>(idx);
        const uint32_t word=group&1?static_cast<uint32_t>(grid>>32):static_cast<uint32_t>(grid);
        const uint8_t sign_nibble=group&1?sign_byte>>4:sign_byte&0xF;
        return static_cast<int32_t>(iq_apply_signs_4(word,sign_nibble));
    }
    else if constexpr(Traits::is_iq1_grid)
    {
        const unsigned lookup=group/2;
        const uint32_t qs=moe_native_group_byte<Ordered>(payload,lookup,previous_dot);
        unsigned high;
        if constexpr(FMT==NATIVE_QUANTISED_IQ1_S)
        {
            const unsigned h0=moe_native_group_byte<Ordered>(payload,4,previous_dot);
            const unsigned h1=moe_native_group_byte<Ordered>(payload,5,previous_dot);
            high=((h0|(h1<<8))>>(3*lookup))&7;
        }
        else high=(moe_native_group_byte<Ordered>(payload,4+lookup/2,previous_dot)>>((lookup%2)*4))&7;
        const uint64_t grid=table_view.iq1(qs|(high<<8));
        return group&1?static_cast<int32_t>(static_cast<uint32_t>(grid>>32)):
            static_cast<int32_t>(static_cast<uint32_t>(grid));
    }
    else if constexpr(Traits::is_8bit)
        return static_cast<int32_t>(moe_native_group_word<Ordered>(payload,group*4,previous_dot));
    else return moe_decode_q4_group<FMT>(moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot),group>=4);
}

/**
 * @brief Decode the original eight groups through the canonical single-group mapping.
 * @tparam FMT Existing supported physical decoder.
 * @param payload Original canonical block. @param packed_groups Receives groups0..7 unchanged.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void moe_decode_native_quantised_groups(
    const uint8_t* __restrict__ payload,int32_t* packed_groups)
{
    #pragma unroll
    for(unsigned group=0;group<8;++group)
        packed_groups[group]=moe_decode_native_quantised_group<FMT>(payload,group);
}

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ float moe_native_quantised_dot_desc_range(
    const DeviceNativeQuantisedMatrixDesc& desc,
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
 * Each `(route, column)` lane computes a complete increasing-K NativeQuantised dot
 * and writes a unique destination. Remote/invalid routes write zero. No route
 * accumulation or floating-point atomic occurs before the LocalTP collective.
 */
template <NativeQuantisedFormat FMT, bool INDEX_BY_EXPERT_ID = true>
__global__ void moe_grouped_native_quantised_down_route_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int K)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
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

    const DeviceNativeQuantisedMatrixDesc desc = descs[descriptor_index];
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
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
    const float expert_acc = moe_native_quantised_dot_desc_range<FMT>(
        desc, n, a_slot, scale_slot, N, K, 0, blocks_per_row);
    route_output[output_index] = rocm_native_quantised_weight_route_rn(
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
__global__ void moe_grouped_native_quantised_down_ordered_reduce_kernel(
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
        total = rocm_native_quantised_accumulate_rn(
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

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(abs_val);
    if (lane == 0)
        scales_A_blockwise[block_idx] = scale;

    A_int8[k] = static_cast<int8_t>(
        llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
            val, scale));
}

/*
 * Grouped MoE decode is defined before the generic verifier-row kernels in this
 * translation unit, but both paths must call the same arithmetic primitives.
 * Keep these declarations here so the grouped dot cannot grow a second copy of
 * the serial M=1 accumulation tree.
 */
template <NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t native_quantised_activation_sum_a(
    const int32_t* __restrict__ a4);

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_accumulate_decoded_block(
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

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ float moe_native_quantised_dot_desc_range(
    const DeviceNativeQuantisedMatrixDesc& desc,
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
        !native_quantised_desc_shape_ok<FMT>(desc, N, K))
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
        const uint8_t* payload = payload_base + linear * NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
        moe_decode_native_quantised_groups<FMT>(payload, packed_groups);

        const float scale_ab = scales_A_blockwise[b];
        const int32_t sum_a = native_quantised_activation_sum_a<FMT>(a4);
        // Mixed IQ4_NL/IQ4_XS pairs share only integer decode. Each actual
        // descriptor retains its own scale program and optional raw word.
        if constexpr (FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE)
        {
            if (desc.codebook_id == NATIVE_QUANTISED_IQ4_NL)
                native_quantised_small_m_accumulate_decoded_block<NATIVE_QUANTISED_IQ4_NL>(
                    acc, a4, packed_groups, payload, scale_base, min_base,
                    emin_base, linear, scale_ab, sum_a);
            else
                native_quantised_small_m_accumulate_decoded_block<FMT>(
                    acc, a4, packed_groups, payload, scale_base, min_base,
                    emin_base, linear, scale_ab, sum_a);
        }
        else
        {
            native_quantised_small_m_accumulate_decoded_block<FMT>(
                acc, a4, packed_groups, payload, scale_base, min_base,
                emin_base, linear, scale_ab, sum_a);
        }
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
template <NativeQuantisedFormat FMT, bool INDEX_BY_EXPERT_ID = true>
__global__ void moe_grouped_native_quantised_down_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ descs,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int k_partitions)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
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

    const DeviceNativeQuantisedMatrixDesc desc = descs[descriptor_index];
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
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
        ? moe_native_quantised_dot_desc_range<FMT>(
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
__global__ void moe_grouped_native_quantised_down_kpart_reduce_kernel(
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
        expert_sum = rocm_native_quantised_accumulate_rn(
            expert_sum, partials[partial_index]);
    }

    route_output[
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n)] = rocm_native_quantised_weight_route_rn(
            route_weights[route], expert_sum);
}

/**
 * @brief Compute table-backed gate/up partials with serial-M1 K boundaries.
 */
template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_kpart_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int k_partitions)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
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

    const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
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
        ? moe_native_quantised_dot_desc_range<FMT>(
              gate_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
    up_partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              up_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Publish table/runtime gate/up rows from ordered split-K partials.
 */
__global__ void moe_grouped_native_quantised_gate_up_kpart_reduce_kernel(
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
        gate_sum = rocm_native_quantised_accumulate_rn(
            gate_sum, gate_partials[partial_index]);
        up_sum = rocm_native_quantised_accumulate_rn(
            up_sum, up_partials[partial_index]);
    }

    gate_outputs[route][n] = gate_sum;
    up_outputs[route][n] = up_sum;
}

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ float moe_native_quantised_dot_desc(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K)
{
    return moe_native_quantised_dot_desc_range<FMT>(
        desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
}

template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_kernel_t(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int K)
{
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE gate/up requires a supported native-quantised descriptor");

    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= num_active || n >= N) return;

    const int expert_id = expert_ids[slot];
    if (expert_id < 0)
        return;
    const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
    const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
    {
        return;
    }

    gate_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_runtime_kernel_t(
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
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE runtime gate/up requires a supported native-quantised descriptor");

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

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc gate_desc = bank.experts[expert_id].gate;
    const DeviceNativeQuantisedMatrixDesc up_desc = bank.experts[expert_id].up;
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
    {
        gate_outputs[slot][n] = 0.0f;
        up_outputs[slot][n] = 0.0f;
        return;
    }

    gate_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        gate_desc, n, A_int8, scales_A_blockwise, N, K);
    up_outputs[slot][n] = moe_native_quantised_dot_desc<FMT>(
        up_desc, n, A_int8, scales_A_blockwise, N, K);
}

/**
 * @brief Compute runtime-placement gate/up partials using serial-M1 K slices.
 */
template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_gate_up_runtime_kpart_kernel_t(
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
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
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

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc gate_desc = bank.experts[expert_id].gate;
    const DeviceNativeQuantisedMatrixDesc up_desc = bank.experts[expert_id].up;
    if (!native_quantised_desc_shape_ok<FMT>(gate_desc, N, K) ||
        !native_quantised_desc_shape_ok<FMT>(up_desc, N, K))
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
        ? moe_native_quantised_dot_desc_range<FMT>(
              gate_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
    up_partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range<FMT>(
              up_desc, n, A_int8, scales_A_blockwise,
              N, K, b_start, b_end)
        : 0.0f;
}

template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_down_runtime_route_kernel_t(
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
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "Grouped MoE runtime down requires a supported native-quantised descriptor");

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

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        route_output[output_index] = 0.0f;
        return;
    }
    const int blocks_per_row = K / 32;
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        route_output[output_index] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc desc = bank.experts[expert_id].down;
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int8_t* a_slot = A_int8 + static_cast<size_t>(slot) * K;
    const float* scale_slot = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;
    const float expert_acc = moe_native_quantised_dot_desc<FMT>(
        desc, n, a_slot, scale_slot, N, K);
    route_output[output_index] = rocm_native_quantised_weight_route_rn(
        route_weights[slot], expert_acc);
}

/**
 * @brief Compute runtime-placement down partials using serial-M1 K slices.
 */
template <NativeQuantisedFormat FMT>
__global__ void moe_grouped_native_quantised_down_runtime_kpart_kernel_t(
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
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
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

    const uint32_t execution_bank = moe_runtime_execution_bank(runtime);
    if (execution_bank > 1u)
    {
        partials[partial_index] = 0.0f;
        return;
    }
    const DeviceMoEPlacementBankView &bank =
        moe_runtime_placement_banks(runtime)[execution_bank];
    if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id))
    {
        partials[partial_index] = 0.0f;
        return;
    }
    const DeviceNativeQuantisedMatrixDesc desc = bank.experts[expert_id].down;
    if (!native_quantised_desc_shape_ok<FMT>(desc, N, K))
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
        ? moe_native_quantised_dot_desc_range<FMT>(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Return the canonical source policy carried by a device descriptor.
 *
 * Q8_1 and Q8_K keep distinct source identities but intentionally share the
 * Q8_0 accelerator arithmetic policy, and MXFP4 shares IQ4_NL's.  Expanded codebook 23 never supplies a
 * policy of its own: its authoritative source identity selects the reduction
 * tree that existed before migration.
 */
__device__ __forceinline__ uint8_t
moe_native_quantised_descriptor_policy_codebook(
    const DeviceNativeQuantisedMatrixDesc* desc)
{
    if (!desc)
        return 0xffu;
    uint8_t policy = desc->source_identity_present
                         ? desc->source_codebook_id
                         : desc->codebook_id;
    if (policy == 20 || policy == 21)
        policy = 19;
    // MXFP4 executes its own decoder but shares IQ4_NL's certified policy,
    // exactly as the host canonicalDeviceVnniCodebookId() masks describe it.
    if (policy == llaminar2::kNativeQuantisedMXFP4Codebook)
        policy = 4;
    return policy;
}

/**
 * @brief Resolve one descriptor's exact serial-M1 K-partition count on device.
 */
__device__ __forceinline__ int moe_native_quantised_descriptor_k_partitions(
    const DeviceNativeQuantisedMatrixDesc* desc,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    const uint8_t policy =
        moe_native_quantised_descriptor_policy_codebook(desc);
    if (policy >= ROCmMoESerialM1PolicyTable::kCodebookCapacity)
        return 0;

    uint64_t packed = 0;
    switch (policy >> 3u)
    {
    case 0:
        packed = policy_partitions_0_7;
        break;
    case 1:
        packed = policy_partitions_8_15;
        break;
    case 2:
        packed = policy_partitions_16_23;
        break;
    case 3:
        packed = policy_partitions_24_31;
        break;
    default:
        return 0;
    }
    const unsigned shift = static_cast<unsigned>(policy & 7u) * 8u;
    return static_cast<int>((packed >> shift) & uint64_t{0xff});
}

/**
 * @brief Decode one descriptor using its physical execution codebook.
 * @tparam DecoderMask Cold immutable compiled decoder family; original generic callers retain the complete mask.
 *
 * The switch is uniform for every lane in a `(route, K-partition)` block.  It
 * therefore preserves the specialized decoder bodies while allowing a single
 * captured graph to consume compact and streamed-expanded expert slots.
 */
template<std::uint32_t DecoderMask=llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders>
__device__ __forceinline__ float moe_native_quantised_dot_desc_range_dynamic(
    const DeviceNativeQuantisedMatrixDesc* desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K,
    int b_start,
    int b_end)
{
    static_assert(DecoderMask!=0&&(DecoderMask&~llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders)==0);
    if (!desc)
        return 0.0f;
    switch (static_cast<NativeQuantisedFormat>(desc->codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q4_0))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q4_0>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ4_NL:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ4_NL))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ4_NL>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ4_XS_SOURCE))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ4_XS_SOURCE>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_MXFP4:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_MXFP4))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_MXFP4>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q4_1:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q4_1))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q4_1>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q5_0:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q5_0))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q5_0>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q5_1:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q5_1))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q5_1>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q6_K:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q6_K))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q6_K>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q3_K:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q3_K))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q3_K>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q2_K:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q2_K))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q2_K>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ3_S:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ3_S))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ3_S>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ3_XXS:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ3_XXS))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ3_XXS>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ2_S:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ2_S))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ2_S>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ2_XS:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ2_XS))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ2_XS>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ2_XXS:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ2_XXS))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ2_XXS>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ1_S:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ1_S))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ1_S>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ1_M:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ1_M))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ1_M>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q8_0:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q8_0))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q8_0>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_EXPANDED_INT8_MIN))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_EXPANDED_INT8_MIN>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    default:
        break;
    }
    // Both unknown physical formats and formats excluded by this admitted
    // compiled family are fatal; no wider decoder recovery is available.
    __builtin_trap();
    return 0.0f;
}

/**
 * @brief List exactly the emitted paired programs, not a second format policy.
 * @details The proof below exhausts the existing execution-decoder inventory
 * and canonical gate/up compatibility predicate. Extending either authority
 * without extending this closed emitted registry fails compilation.
 */
#define LLAMINAR_MOE_PAIRED_K32_PROGRAMS(Apply) \
    Apply(NATIVE_QUANTISED_Q4_0,NATIVE_QUANTISED_Q4_0) \
    Apply(NATIVE_QUANTISED_IQ4_NL,NATIVE_QUANTISED_IQ4_NL) \
    Apply(NATIVE_QUANTISED_IQ4_XS_SOURCE,NATIVE_QUANTISED_IQ4_XS_SOURCE) \
    Apply(NATIVE_QUANTISED_Q4_1,NATIVE_QUANTISED_Q4_1) \
    Apply(NATIVE_QUANTISED_Q5_0,NATIVE_QUANTISED_Q5_0) \
    Apply(NATIVE_QUANTISED_Q5_1,NATIVE_QUANTISED_Q5_1) \
    Apply(NATIVE_QUANTISED_Q6_K,NATIVE_QUANTISED_Q6_K) \
    Apply(NATIVE_QUANTISED_Q3_K,NATIVE_QUANTISED_Q3_K) \
    Apply(NATIVE_QUANTISED_Q2_K,NATIVE_QUANTISED_Q2_K) \
    Apply(NATIVE_QUANTISED_IQ3_S,NATIVE_QUANTISED_IQ3_S) \
    Apply(NATIVE_QUANTISED_IQ3_XXS,NATIVE_QUANTISED_IQ3_XXS) \
    Apply(NATIVE_QUANTISED_IQ2_S,NATIVE_QUANTISED_IQ2_S) \
    Apply(NATIVE_QUANTISED_IQ2_XS,NATIVE_QUANTISED_IQ2_XS) \
    Apply(NATIVE_QUANTISED_IQ2_XXS,NATIVE_QUANTISED_IQ2_XXS) \
    Apply(NATIVE_QUANTISED_IQ1_S,NATIVE_QUANTISED_IQ1_S) \
    Apply(NATIVE_QUANTISED_IQ1_M,NATIVE_QUANTISED_IQ1_M) \
    Apply(NATIVE_QUANTISED_Q8_0,NATIVE_QUANTISED_Q8_0) \
    Apply(NATIVE_QUANTISED_EXPANDED_INT8_MIN,NATIVE_QUANTISED_EXPANDED_INT8_MIN) \
    Apply(NATIVE_QUANTISED_IQ4_NL,NATIVE_QUANTISED_IQ4_XS_SOURCE) \
    Apply(NATIVE_QUANTISED_IQ4_XS_SOURCE,NATIVE_QUANTISED_IQ4_NL)

/** @return Whether the emitted programs exactly cover the canonical supported compatibility matrix. */
constexpr bool moePairedK32RegistryMatchesCanonical() noexcept
{
#define LLAMINAR_MOE_PAIRED_K32_RECORD(Gate,Up) std::array<unsigned,2>{unsigned(Gate),unsigned(Up)},
    constexpr std::array programs{LLAMINAR_MOE_PAIRED_K32_PROGRAMS(LLAMINAR_MOE_PAIRED_K32_RECORD)};
#undef LLAMINAR_MOE_PAIRED_K32_RECORD
    constexpr auto supported=llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders;
    unsigned gate_inventory=0,up_inventory=0;
    for(const auto& program:programs)
    {
        if(program[0]>=32||program[1]>=32)return false;
        gate_inventory|=1u<<program[0];up_inventory|=1u<<program[1];
    }
    if(gate_inventory!=supported||up_inventory!=supported)return false;
    for(unsigned gate=0;gate<32;++gate)
    for(unsigned up=0;up<32;++up)
    {
        unsigned matches=0;
        for(const auto& program:programs)matches+=(program[0]==gate&&program[1]==up);
        const bool canonical=(supported&(1u<<gate))&&(supported&(1u<<up))&&
            llaminar2::areNativeQuantisedGateUpCodebooksCompatible(static_cast<uint8_t>(gate),static_cast<uint8_t>(up));
        if(matches!=(canonical?1u:0u))return false;
    }
    return true;
}
static_assert(moePairedK32RegistryMatchesCanonical(),
    "Paired K32 programs must exactly cover the canonical decoder/compatibility inventory");

/**
 * @brief Consume one role's unchanged decoder and rounded K32 contribution.
 * @tparam FMT Original physical decoder, independent of the other role.
 * @param desc Original descriptor; original source-policy arithmetic stays outside execution dispatch.
 * @param linear Exact original [K32,N] coordinate.
 * @param a4 Once-loaded original eight activation words for this K32.
 * @param scale Original published FP32 activation scale.
 * @param sum Exact original eight-dot4 activation sum, shared only as an integer value.
 * @param accumulator This role's original increasing-block FP32 chain.
 * @note Only one decoded8word weight operand is live at a time. No local aggregate
 * address escapes this force-inlined call, and no new scratch allocation exists.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void moePairedK32Commit(const DeviceNativeQuantisedMatrixDesc& desc,size_t linear,
    const int32_t* a4,float scale,int32_t sum,float& accumulator)
{
    const auto* payload=desc.payload+linear*NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    int32_t groups[8];
    moe_decode_native_quantised_groups<FMT>(payload,groups);
    native_quantised_small_m_accumulate_decoded_block<FMT>(accumulator,a4,groups,payload,
        static_cast<const uint16_t*>(desc.scales),static_cast<const uint16_t*>(desc.mins),
        static_cast<const uint32_t*>(desc.emins),linear,scale,sum);
}
/**
 * @brief Commit the unchanged canonical block from immutable planes captured before the loop.
 * @tparam FMT Original independently admitted execution decoder.
 * @param payload_base Original prepared payload plane; no descriptor aggregate copy.
 * @param scales Original primary metadata plane. @param mins Original secondary metadata plane.
 * @param emins Original dual-minimum plane. @param linear Original [K32,N] coordinate.
 * @param a4 Shared once-loaded activation words. @param scale Original FP32 sidecar value.
 * @param sum Shared exact integer sum. @param accumulator Original role-owned ascending chain.
 * @note The caller reads these pointers once before any rounded commit. Keeping
 * scalar plane values separate avoids descriptor rereads across numerical
 * compiler barriers without introducing another live allocation or byte authority.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void moePairedK32CommitPlanes(const uint8_t* payload_base,
    const uint16_t* scales,const uint16_t* mins,const uint32_t* emins,size_t linear,
    const int32_t* a4,float scale,int32_t sum,float& accumulator)
{
    const auto* payload=payload_base+linear*NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    int32_t groups[8];
    moe_decode_native_quantised_groups<FMT>(payload,groups);
    native_quantised_small_m_accumulate_decoded_block<FMT>(accumulator,a4,groups,payload,
        scales,mins,emins,linear,scale,sum);
}
/**
 * @brief Walk only the union of two original partition ranges, sharing identical activations.
 * @tparam Gate Original gate physical format selected once before the K32 loop.
 * @tparam Up Original up physical format selected by the same canonical compatibility registry.
 * @param gate Original gate descriptor. @param up Original up descriptor.
 * @param n Original column lane. @param input Original Q8 hidden row.
 * @param scales Original FP32 block scales. @param N Original column count.
 * @param K Original reduction width. @param gate_begin Gate's original first K32.
 * @param gate_end Gate's original exclusive last K32. @param up_begin Up's original first K32.
 * @param up_end Up's original exclusive last K32. @param gate_acc Original gate partial result.
 * @param up_acc Original up partial result.
 * @details Each role sees exactly its original ascending block subsequence and
 * canonical per-format contribution helper. Gaps are skipped before any input
 * load. Invalid original shapes contribute+0 exactly as the original range dot.
 * Integer sum sharing does not change IQ dual-half/subgroup correction helpers.
 */
template<NativeQuantisedFormat Gate,NativeQuantisedFormat Up>
__device__ __forceinline__ void moePairedK32RangeTyped(const DeviceNativeQuantisedMatrixDesc& gate,
    const DeviceNativeQuantisedMatrixDesc& up,int n,const int8_t* input,const float* scales,int N,int K,
    int gate_begin,int gate_end,int up_begin,int up_end,float& gate_acc,float& up_acc)
{
    static_assert(llaminar2::areNativeQuantisedGateUpCodebooksCompatible(uint8_t(Gate),uint8_t(Up)));
    gate_acc=0.0f;up_acc=0.0f;
    if(!input||!scales||n<0||n>=N||K<=0||K%32)return;
    const bool valid_gate=native_quantised_desc_shape_ok<Gate>(gate,N,K);
    const bool valid_up=native_quantised_desc_shape_ok<Up>(up,N,K);
    if(valid_gate&&valid_up&&gate_begin==up_begin&&gate_end==up_end)
    {
        // Authenticate the equality once. Each role still owns its original
        // range; equal ranges permit unconditional commits to independent chains.
        // Read immutable plane addresses before the loop, rather than requesting
        // descriptor fields again after a canonical rounded-operation barrier.
        const uint8_t* gate_payload=gate.payload;
        const uint16_t* gate_scales=static_cast<const uint16_t*>(gate.scales);
        const uint16_t* gate_mins=static_cast<const uint16_t*>(gate.mins);
        const uint32_t* gate_emins=static_cast<const uint32_t*>(gate.emins);
        const uint8_t* up_payload=up.payload;
        const uint16_t* up_scales=static_cast<const uint16_t*>(up.scales);
        const uint16_t* up_mins=static_cast<const uint16_t*>(up.mins);
        const uint32_t* up_emins=static_cast<const uint32_t*>(up.emins);
        #pragma unroll 1
        for(int b=gate_begin;b<gate_end;++b)
        {
            const auto* source=reinterpret_cast<const int32_t*>(input+size_t(b)*32u);
            int32_t activation[8];
            #pragma unroll
            for(int word=0;word<8;++word)activation[word]=source[word];
            const float scale=scales[b];
            int32_t sum=0;
            if constexpr(NATIVE_QUANTISEDTraits<Gate>::is_asymmetric||NATIVE_QUANTISEDTraits<Up>::is_asymmetric)
            {
                #pragma unroll
                for(int word=0;word<8;++word)
                    sum=::llaminar2::rocm_integer_contract::signedDot4(activation[word],0x01010101,sum);
            }
            const size_t linear=size_t(b)*size_t(N)+size_t(n);
            moePairedK32CommitPlanes<Gate>(gate_payload,gate_scales,gate_mins,gate_emins,
                linear,activation,scale,sum,gate_acc);
            moePairedK32CommitPlanes<Up>(up_payload,up_scales,up_mins,up_emins,
                linear,activation,scale,sum,up_acc);
        }
        return;
    }
    // Unequal ranges or one invalid original role retain the existing exact
    // union walk, including skipped gaps, role-specific+0 and source metadata.
    const int first=min(gate_begin,up_begin),last=max(gate_end,up_end);
    #pragma unroll 1
    for(int b=first;b<last;++b)
    {
        const bool active_gate=valid_gate&&b>=gate_begin&&b<gate_end;
        const bool active_up=valid_up&&b>=up_begin&&b<up_end;
        if(!active_gate&&!active_up)continue; // No activation read or padded contribution in a gap.
        const auto* source=reinterpret_cast<const int32_t*>(input+size_t(b)*32u);
        int32_t activation[8];
        #pragma unroll
        for(int word=0;word<8;++word)activation[word]=source[word];
        const float scale=scales[b];
        const bool needs_sum=(active_gate&&NATIVE_QUANTISEDTraits<Gate>::is_asymmetric)||
            (active_up&&NATIVE_QUANTISEDTraits<Up>::is_asymmetric);
        int32_t sum=0;
        if(needs_sum)
        {
            #pragma unroll
            for(int word=0;word<8;++word)
                sum=::llaminar2::rocm_integer_contract::signedDot4(activation[word],0x01010101,sum);
        }
        const size_t linear=size_t(b)*size_t(N)+size_t(n);
        if(active_gate)moePairedK32Commit<Gate>(gate,linear,activation,scale,sum,gate_acc);
        if(active_up)moePairedK32Commit<Up>(up,linear,activation,scale,sum,up_acc);
    }
}
/**
 * @brief Select exactly one admitted format pair before entering the original K32 partition walk.
 * @param gate Original gate descriptor. @param up Original up descriptor.
 * @param n Original column lane. @param input Original Q8 hidden row.
 * @param scales Original FP32 block scales. @param N Original column count.
 * @param K Original reduction width. @param gate_begin Gate's original first K32.
 * @param gate_end Gate's original exclusive last K32. @param up_begin Up's original first K32.
 * @param up_end Up's original exclusive last K32. @param gate_acc Original gate partial result.
 * @param up_acc Original up partial result.
 * @note The registry is compile-time checked against the existing canonical
 * predicate. An unknown or incompatible pair traps; no per-block dispatcher,
 * wider Cartesian family or automatic generic recovery exists.
 */
__device__ __forceinline__ void moePairedK32Range(const DeviceNativeQuantisedMatrixDesc& gate,
    const DeviceNativeQuantisedMatrixDesc& up,int n,const int8_t* input,const float* scales,int N,int K,
    int gate_begin,int gate_end,int up_begin,int up_end,float& gate_acc,float& up_acc)
{
    const unsigned key=(unsigned(gate.codebook_id)<<8)|unsigned(up.codebook_id);
    switch(key)
    {
#define LLAMINAR_MOE_PAIRED_K32_CASE(Gate,Up) \
    case (unsigned(Gate)<<8)|unsigned(Up): \
        moePairedK32RangeTyped<Gate,Up>(gate,up,n,input,scales,N,K,gate_begin,gate_end,up_begin,up_end,gate_acc,up_acc);return;
        LLAMINAR_MOE_PAIRED_K32_PROGRAMS(LLAMINAR_MOE_PAIRED_K32_CASE)
#undef LLAMINAR_MOE_PAIRED_K32_CASE
    default:__builtin_trap();return;
    }
}
#undef LLAMINAR_MOE_PAIRED_K32_PROGRAMS

/**
 * @brief Load gate/up descriptors from a static table or the active runtime bank.
 */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ bool moe_native_quantised_load_gate_up_descriptors(
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    int expert_id,
    int num_experts,
    const DeviceNativeQuantisedMatrixDesc*& gate_desc,
    const DeviceNativeQuantisedMatrixDesc*& up_desc)
{
    if (expert_id < 0 || expert_id >= num_experts)
        return false;
    if constexpr (RUNTIME_TABLE)
    {
        const uint32_t execution_bank =
            moe_runtime_execution_bank(runtime);
        if (execution_bank > 1u)
            return false;
        const DeviceMoEPlacementBankView& bank =
            moe_runtime_placement_banks(runtime)[execution_bank];
        if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id)) return false;
        gate_desc = &bank.experts[expert_id].gate;
        up_desc = &bank.experts[expert_id].up;
    }
    else
    {
        gate_desc = &gate_descs[expert_id];
        up_desc = &up_descs[expert_id];
    }
    return true;
}

/** @brief Load a down descriptor from a static table or active runtime bank. */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ bool moe_native_quantised_load_down_descriptor(
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    int expert_id,
    int num_experts,
    const DeviceNativeQuantisedMatrixDesc*& down_desc)
{
    if (expert_id < 0 || expert_id >= num_experts)
        return false;
    if constexpr (RUNTIME_TABLE)
    {
        const uint32_t execution_bank =
            moe_runtime_execution_bank(runtime);
        if (execution_bank > 1u)
            return false;
        const DeviceMoEPlacementBankView& bank =
            moe_runtime_placement_banks(runtime)[execution_bank];
        if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id)) return false;
        down_desc = &bank.experts[expert_id].down;
    }
    else
    {
        down_desc = &down_descs[expert_id];
    }
    return true;
}

/**
 * @brief Produce descriptor-local gate/up split-K partials for a stable graph.
 */
/**
 * @brief Execute one original indexed64-column split-K producer task.
 * @details Original wrapper and last-producer entry share the exact body;
 * physical completion metadata cannot change source-policy arithmetic.
 */
template <bool RUNTIME_TABLE,bool PairedK32=false>
__device__ __forceinline__ void moeGateUpOriginalPartialTask(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    unsigned logical_tile,unsigned logical_route,unsigned logical_part,unsigned logical_lane)
{
    constexpr int TILE_N = 64;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    const int k_part = logical_part;
    if (route >= num_active || n >= N ||
        k_part >= maximum_k_partitions)
    {
        return;
    }

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const DeviceNativeQuantisedMatrixDesc* gate_desc = nullptr;
    const DeviceNativeQuantisedMatrixDesc* up_desc = nullptr;
    if (!moe_native_quantised_load_gate_up_descriptors<RUNTIME_TABLE>(
            gate_descs, up_descs, runtime, expert_ids[route], num_experts,
            gate_desc, up_desc))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int gate_partitions =
        moe_native_quantised_descriptor_k_partitions(
            gate_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    const int up_partitions =
        moe_native_quantised_descriptor_k_partitions(
            up_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (gate_partitions <= 0 || up_partitions <= 0)
    {
        // A published descriptor without generated arithmetic evidence is fatal.
        __builtin_trap();
        return;
    }

    const int blocks_per_row = K / 32;
    if constexpr(PairedK32)
    {
        const int gate_count=(blocks_per_row+gate_partitions-1)/gate_partitions;
        const int up_count=(blocks_per_row+up_partitions-1)/up_partitions;
        const int gate_begin=k_part<gate_partitions?min(blocks_per_row,k_part*gate_count):blocks_per_row;
        const int up_begin=k_part<up_partitions?min(blocks_per_row,k_part*up_count):blocks_per_row;
        float gate_acc=0.f,up_acc=0.f;
        moePairedK32Range(*gate_desc,*up_desc,n,A_int8,scales_A_blockwise,N,K,
            gate_begin,min(blocks_per_row,gate_begin+gate_count),
            up_begin,min(blocks_per_row,up_begin+up_count),gate_acc,up_acc);
        gate_partials[partial_index]=gate_acc;up_partials[partial_index]=up_acc;
        return;
    }
    if (k_part < gate_partitions)
    {
        const int blocks_per_part =
            (blocks_per_row + gate_partitions - 1) / gate_partitions;
        const int b_start = k_part * blocks_per_part;
        const int b_end = min(blocks_per_row, b_start + blocks_per_part);
        gate_partials[partial_index] = b_start < b_end
            ? moe_native_quantised_dot_desc_range_dynamic(
                  gate_desc, n, A_int8, scales_A_blockwise,
                  N, K, b_start, b_end)
            : 0.0f;
    }
    else
    {
        gate_partials[partial_index] = 0.0f;
    }

    if (k_part < up_partitions)
    {
        const int blocks_per_part =
            (blocks_per_row + up_partitions - 1) / up_partitions;
        const int b_start = k_part * blocks_per_part;
        const int b_end = min(blocks_per_row, b_start + blocks_per_part);
        up_partials[partial_index] = b_start < b_end
            ? moe_native_quantised_dot_desc_range_dynamic(
                  up_desc, n, A_int8, scales_A_blockwise,
                  N, K, b_start, b_end)
            : 0.0f;
    }
    else
    {
        up_partials[partial_index] = 0.0f;
    }
}

template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions)
{
    moeGateUpOriginalPartialTask<RUNTIME_TABLE>(
        A_int8, scales_A_blockwise, gate_descs, up_descs, runtime, expert_ids, gate_partials, up_partials, num_active, N, K, num_experts, policy_partitions_0_7, policy_partitions_8_15, policy_partitions_16_23, policy_partitions_24_31, maximum_k_partitions,
        blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x);
}

/**
 * @brief Reduce descriptor-local gate/up partials in each source policy's order.
 */
template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_gate_up_descriptor_reduce_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* const* __restrict__ gate_outputs,
    float* const* __restrict__ up_outputs,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    const int route = blockIdx.y;
    if (route >= num_active || n >= N)
        return;

    const DeviceNativeQuantisedMatrixDesc* gate_desc = nullptr;
    const DeviceNativeQuantisedMatrixDesc* up_desc = nullptr;
    if (!moe_native_quantised_load_gate_up_descriptors<RUNTIME_TABLE>(
            gate_descs, up_descs, runtime, expert_ids[route], num_experts,
            gate_desc, up_desc))
    {
        gate_outputs[route][n] = 0.0f;
        up_outputs[route][n] = 0.0f;
        return;
    }

    const int gate_partitions =
        moe_native_quantised_descriptor_k_partitions(
            gate_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    const int up_partitions =
        moe_native_quantised_descriptor_k_partitions(
            up_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (gate_partitions <= 0 || up_partitions <= 0)
    {
        __builtin_trap();
        return;
    }

    // A one-partition serial launch publishes the dot directly.  Copying its
    // sole partial avoids changing a possible signed zero through `0 + value`.
    const size_t first_partial_index =
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    float gate_sum = gate_partials[first_partial_index];
    if (gate_partitions > 1)
    {
        gate_sum = 0.0f;
        #pragma unroll 1
        for (int part = 0; part < gate_partitions; ++part)
        {
            const size_t partial_index =
                (static_cast<size_t>(part) *
                     static_cast<size_t>(num_active) +
                 static_cast<size_t>(route)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            gate_sum = rocm_native_quantised_accumulate_rn(
                gate_sum, gate_partials[partial_index]);
        }
    }

    float up_sum = up_partials[first_partial_index];
    if (up_partitions > 1)
    {
        up_sum = 0.0f;
        #pragma unroll 1
        for (int part = 0; part < up_partitions; ++part)
        {
            const size_t partial_index =
                (static_cast<size_t>(part) *
                     static_cast<size_t>(num_active) +
                 static_cast<size_t>(route)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            up_sum = rocm_native_quantised_accumulate_rn(
                up_sum, up_partials[partial_index]);
        }
    }

    gate_outputs[route][n] = gate_sum;
    up_outputs[route][n] = up_sum;
}

/** @brief Closed fold load scheduling: staged arithmetic remains the original one-part loop. */
enum class MoEGateUpFoldLoads {Original,TwoPartialLookahead};
/**
 * @brief Load two independent partials before executing their original ordered additions.
 * @param partials Original immutable source-policy partial bank after publication.
 * @param partitions Authenticated positive original descriptor-local partition count.
 * @param num_active Exact original route count used by every producer.
 * @param route Original route index, unchanged by load scheduling.
 * @param columns Original output width/producer bank stride.
 * @param column Original active column index.
 * @return Bit-identical direct partial for P1, otherwise the original ascending-partition RN sum.
 * @details At most two partial values are pending. The odd final partition is
 * loaded only when in bounds; no padding or another source policy is consumed.
 * Each addition still depends on the immediately preceding accumulator. These
 * independent memory loads do not authorize reassociation or a wider reduction.
 */
__device__ __forceinline__ float moeGateUpTwoPartialLookaheadSum(
    const float* __restrict__ partials,int partitions,int num_active,int route,int columns,int column)
{
    const size_t first=static_cast<size_t>(route)*static_cast<size_t>(columns)+static_cast<size_t>(column);
    if(partitions==1)return partials[first]; // Preserve the original signed-zero direct publication.
    float sum=0.0f;
    int part=0;
    #pragma unroll 1
    for(;part+1<partitions;part+=2)
    {
        const size_t first_index=(static_cast<size_t>(part)*static_cast<size_t>(num_active)+
            static_cast<size_t>(route))*static_cast<size_t>(columns)+static_cast<size_t>(column);
        const size_t second_index=(static_cast<size_t>(part+1)*static_cast<size_t>(num_active)+
            static_cast<size_t>(route))*static_cast<size_t>(columns)+static_cast<size_t>(column);
        const float first_value=partials[first_index];
        const float second_value=partials[second_index];
        sum=rocm_native_quantised_accumulate_rn(sum,first_value);
        sum=rocm_native_quantised_accumulate_rn(sum,second_value);
    }
    if(part<partitions)
    {
        const size_t tail_index=(static_cast<size_t>(part)*static_cast<size_t>(num_active)+
            static_cast<size_t>(route))*static_cast<size_t>(columns)+static_cast<size_t>(column);
        sum=rocm_native_quantised_accumulate_rn(sum,partials[tail_index]);
    }
    return sum;
}

/**
 * @brief Reduce gate/up partials and publish the exact Q8 SwiGLU row in-place.
 *
 * Production grouped M=1 decode consumes gate/up only through the following
 * down projection, for both explicit routing and runtime-table routing.
 * Materializing two FP32 rows and launching a second kernel to read them back
 * is therefore avoidable.  This reducer retains the exact descriptor-local
 * ascending-partition additions used above, then performs the same 32-value
 * SwiGLU maximum reduction and Q8 rounding as
 * `moe_grouped_swiglu_quantize_blockwise_kernel`.
 *
 * One 64-lane workgroup owns two independent Q8 blocks.  XOR masks never set
 * wavefront bit 5, so the lower and upper 32 lanes cannot exchange values on a
 * wave64 device.  Invalid routes intentionally leave their Q8 scratch
 * untouched: the down kernel authenticates the same negative expert id before
 * reading that scratch and publishes a zero canonical route row.
 */
template <bool RUNTIME_TABLE,MoEGateUpFoldLoads Loads=MoEGateUpFoldLoads::Original>
__device__ __forceinline__ void moeGateUpOriginalFoldSwiGLUQ8Task(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    int8_t* __restrict__ swiglu_int8,
    float* __restrict__ swiglu_scales,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    unsigned logical_tile,unsigned logical_route,unsigned logical_lane)
{
    static_assert(Loads==MoEGateUpFoldLoads::Original||Loads==MoEGateUpFoldLoads::TwoPartialLookahead);
    constexpr int TILE_N = 64;
    constexpr int Q8_BLOCK = 32;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    if (route >= num_active)
        return;

    const DeviceNativeQuantisedMatrixDesc* gate_desc = nullptr;
    const DeviceNativeQuantisedMatrixDesc* up_desc = nullptr;
    if (!moe_native_quantised_load_gate_up_descriptors<RUNTIME_TABLE>(
            gate_descs, up_descs, runtime, expert_ids[route], num_experts,
            gate_desc, up_desc))
    {
        return;
    }

    const int gate_partitions =
        moe_native_quantised_descriptor_k_partitions(
            gate_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    const int up_partitions =
        moe_native_quantised_descriptor_k_partitions(
            up_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (gate_partitions <= 0 || up_partitions <= 0)
    {
        __builtin_trap();
        return;
    }

    const bool active_lane = n < N;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (active_lane)
    {
        /*
         * Preserve the one-partition signed-zero rule and the exact
         * increasing-partition FP32 addition order of the standalone reducer.
         */
        if constexpr(Loads==MoEGateUpFoldLoads::TwoPartialLookahead)
        {
            gate_sum=moeGateUpTwoPartialLookaheadSum(gate_partials,gate_partitions,num_active,route,N,n);
            up_sum=moeGateUpTwoPartialLookaheadSum(up_partials,up_partitions,num_active,route,N,n);
        }
        else
        {
            const size_t first_partial_index =
                static_cast<size_t>(route) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            gate_sum = gate_partials[first_partial_index];
            if (gate_partitions > 1)
            {
                gate_sum = 0.0f;
                #pragma unroll 1
                for (int part = 0; part < gate_partitions; ++part)
                {
                    const size_t partial_index =
                        (static_cast<size_t>(part) *
                             static_cast<size_t>(num_active) +
                         static_cast<size_t>(route)) * static_cast<size_t>(N) +
                        static_cast<size_t>(n);
                    gate_sum = rocm_native_quantised_accumulate_rn(
                        gate_sum, gate_partials[partial_index]);
                }
            }

            up_sum = up_partials[first_partial_index];
            if (up_partitions > 1)
            {
                up_sum = 0.0f;
                #pragma unroll 1
                for (int part = 0; part < up_partitions; ++part)
                {
                    const size_t partial_index =
                        (static_cast<size_t>(part) *
                             static_cast<size_t>(num_active) +
                         static_cast<size_t>(route)) * static_cast<size_t>(N) +
                        static_cast<size_t>(n);
                    up_sum = rocm_native_quantised_accumulate_rn(
                        up_sum, up_partials[partial_index]);
                }
            }
        }
    }

    const float swiglu = active_lane
                             ? llaminar2::rocm::decode_equivalent::swigluValue(
                                   gate_sum, up_sum)
                             : 0.0f;
    float block_abs_max = fabsf(swiglu);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        block_abs_max = fmaxf(
            block_abs_max,
            __shfl_xor(block_abs_max, mask));

    if (!active_lane)
        return;

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(block_abs_max);
    const int q8_block = n / Q8_BLOCK;
    const int lane = logical_lane & (Q8_BLOCK - 1);
    if (lane == 0)
    {
        swiglu_scales[
            static_cast<size_t>(route) * static_cast<size_t>(N / Q8_BLOCK) +
            static_cast<size_t>(q8_block)] = scale;
    }
    swiglu_int8[
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n)] = static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
                swiglu, scale));
}

template <bool RUNTIME_TABLE>
__global__ void
moe_grouped_native_quantised_gate_up_descriptor_reduce_swiglu_q8_kernel(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    int8_t* __restrict__ swiglu_int8,
    float* __restrict__ swiglu_scales,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    moeGateUpOriginalFoldSwiGLUQ8Task<RUNTIME_TABLE>(
        gate_partials, up_partials, gate_descs, up_descs, runtime, expert_ids, swiglu_int8, swiglu_scales, num_active, N, num_experts, policy_partitions_0_7, policy_partitions_8_15, policy_partitions_16_23, policy_partitions_24_31,
        blockIdx.x,blockIdx.y,threadIdx.x);
}

/**
 * @brief Keep the original64-thread split-K grid and publish each tile from its last completed producer.
 * @details Integer acquire/release tickets order the existing global partial
 * stores. Every nonfinal CTA retires immediately; no spinning, cooperative
 * launch, global residency bound, floating atomic or altered fold is involved.
 */
template<bool PairedK32>
__device__ __forceinline__ void moeGateUpLastProducerBody(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int8_t* swiglu_int8,float* swiglu_scales,llaminar2::NativeTileCompletionView completion)
{
    using llaminar2::NativeTileCompletionDecision;
    const unsigned tiles=(static_cast<unsigned>(N)+63u)/64u;
    if(!completion.counters||completion.tiles!=tiles||completion.routes!=static_cast<unsigned>(num_active)||
        completion.contributors!=static_cast<unsigned>(maximum_k_partitions)||blockDim.x!=64u||
        blockIdx.x>=tiles||blockIdx.y>=completion.routes||blockIdx.z>=completion.contributors)__builtin_trap();
    moeGateUpOriginalPartialTask<false,PairedK32>(
        A_int8, scales_A_blockwise, gate_descs, up_descs, runtime, expert_ids, gate_partials, up_partials, num_active, N, K, num_experts, policy_partitions_0_7, policy_partitions_8_15, policy_partitions_16_23, policy_partitions_24_31, maximum_k_partitions,
        blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x);
    // A full CTA joins all original partial writers before its leader releases
    // them into the modification-order chain. Tail lanes never return here.
    __syncthreads();
    __shared__ unsigned decision;
    auto* counter=completion.counters+size_t(blockIdx.y)*completion.tiles+blockIdx.x;
    if(threadIdx.x==0)
    {
        const auto result=llaminar2::native_tile_completion::arriveForPublication(
            counter,completion.contributors);
        if(result==NativeTileCompletionDecision::Invalid)__builtin_trap();
        decision=static_cast<unsigned>(result);
    }
    // This distributes the final leader's acquired global payload to every
    // original consumer lane before it reads the unchanged ordered bank.
    __syncthreads();
    if(decision!=static_cast<unsigned>(NativeTileCompletionDecision::Publish))return;
    moeGateUpOriginalFoldSwiGLUQ8Task<false,MoEGateUpFoldLoads::TwoPartialLookahead>(
        gate_partials,up_partials,gate_descs,up_descs,nullptr,expert_ids,
        swiglu_int8,swiglu_scales,num_active,N,num_experts,
        policy_partitions_0_7,policy_partitions_8_15,policy_partitions_16_23,policy_partitions_24_31,
        blockIdx.x,blockIdx.y,threadIdx.x);
    // Invalid routes still discharge and reset their protocol without reading
    // stale Q8 bytes. All valid publication writes finish before reset/reuse.
    __syncthreads();
    if(threadIdx.x==0)llaminar2::native_tile_completion::reset(counter);
}

/** @brief Original last-producer arithmetic and load policy; oracle remains independently retained. */
__global__ __launch_bounds__(64) void moe_grouped_native_quantised_gate_up_last_producer_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int8_t* swiglu_int8,float* swiglu_scales,llaminar2::NativeTileCompletionView completion)
{
    moeGateUpLastProducerBody<false>(A_int8,scales_A_blockwise,gate_descs,up_descs,runtime,expert_ids,gate_partials,up_partials,
        num_active,N,K,num_experts,policy_partitions_0_7,policy_partitions_8_15,
        policy_partitions_16_23,policy_partitions_24_31,maximum_k_partitions,swiglu_int8,swiglu_scales,completion);
}

/** @brief Same exact original64-thread completion with paired K32 activation/sum consumption. */
__global__ __launch_bounds__(64) void moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int8_t* swiglu_int8,float* swiglu_scales,llaminar2::NativeTileCompletionView completion)
{
    moeGateUpLastProducerBody<true>(A_int8,scales_A_blockwise,gate_descs,up_descs,runtime,expert_ids,gate_partials,up_partials,
        num_active,N,K,num_experts,policy_partitions_0_7,policy_partitions_8_15,
        policy_partitions_16_23,policy_partitions_24_31,maximum_k_partitions,swiglu_int8,swiglu_scales,completion);
}

/**
 * @brief Produce descriptor-local down partials for table or runtime routing.
 * @param A_int8 Original route-major Q8 down input.
 * @param scales_A_blockwise Original route-major Q8 scales.
 * @param down_descs Original compact immutable/runtime-published descriptor table.
 * @param runtime Original runtime bank for the original runtime-table specialization.
 * @param expert_ids Original selected slot IDs, including invalid/empty slots.
 * @param partials Original partition-major [part,route,H] publication.
 * @param num_active Original selected slot extent.
 * @param N Original H output width.
 * @param K Original participant-local F reduction width.
 * @param num_experts Original descriptor table extent.
 * @param policy_partitions_0_7 Original source-policy partition entries0..7.
 * @param policy_partitions_8_15 Original source-policy partition entries8..15.
 * @param policy_partitions_16_23 Original source-policy partition entries16..23.
 * @param policy_partitions_24_31 Original source-policy partition entries24..31.
 * @param maximum_k_partitions Original physical grid-Z extent, including zero partial obligations.
 * @param logical_tile Original64-column grid-X coordinate.
 * @param logical_route Original selected-slot grid-Y coordinate.
 * @param logical_part Original split-K grid-Z coordinate.
 * @param logical_lane Original64-thread column lane.
 */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ void moeDownOriginalPartialTask(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int logical_tile,int logical_route,int logical_part,int logical_lane)
{
    constexpr int TILE_N = 64;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    const int k_part = logical_part;
    if (route >= num_active || n >= N ||
        k_part >= maximum_k_partitions)
    {
        return;
    }

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const DeviceNativeQuantisedMatrixDesc* desc = nullptr;
    if (!moe_native_quantised_load_down_descriptor<RUNTIME_TABLE>(
            down_descs, runtime, expert_ids[route], num_experts, desc))
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int route_partitions =
        moe_native_quantised_descriptor_k_partitions(
            desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (route_partitions <= 0)
    {
        __builtin_trap();
        return;
    }
    if (k_part >= route_partitions)
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + route_partitions - 1) / route_partitions;
    const int b_start = k_part * blocks_per_part;
    const int b_end = min(blocks_per_row, b_start + blocks_per_part);
    const int8_t* route_a =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* route_scales =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range_dynamic(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
}

template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_down_descriptor_kpart_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions)
{
    moeDownOriginalPartialTask<RUNTIME_TABLE>(A_int8,scales_A_blockwise,down_descs,runtime,expert_ids,partials,num_active,N,K,num_experts,policy_partitions_0_7,policy_partitions_8_15,policy_partitions_16_23,policy_partitions_24_31,maximum_k_partitions,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x);
}

/**
 * @brief Evaluate one original partition without changing its dot expression.
 * @param descriptor Original admitted source/execution descriptor.
 * @param n Participant-local output column.
 * @param activation Exact published Q8 row.
 * @param scales Original per-block Q8 scales.
 * @param N Original output width.
 * @param K Original reduction width.
 * @param part Original serial-M1 partition index.
 * @param partitions Original descriptor-local partition count.
 * @return Byte-identical original split-K partial, including empty partitions.
 */
template<std::uint32_t DecoderMask=llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders>
__device__ __forceinline__ float moe_native_quantised_block_partial(
    const DeviceNativeQuantisedMatrixDesc* descriptor, int n,
    const int8_t* activation, const float* scales, int N, int K,
    int part, int partitions)
{
    const int blocks = K / 32;
    const int width = (blocks + partitions - 1) / partitions;
    const int first = part * width;
    const int end = min(blocks, first + width);
    return first < end
        ? moe_native_quantised_dot_desc_range_dynamic<DecoderMask>(
              descriptor, n, activation, scales, N, K, first, end)
        : 0.0f;
}

/**
 * @brief Fold CTA-owned original partitions with the serial signed-zero rule.
 * @param partials Shared partition-major plane written before the CTA barrier.
 * @param column Original column within the 32-column publication tile.
 * @param partitions Exact descriptor-local arithmetic partition count.
 * @return Original increasing-partition explicit-RN result.
 */
__device__ __forceinline__ float moe_native_quantised_block_fold(
    const float* partials, int column, int partitions)
{
    if (partitions == 1)
        return partials[column];
    float value = 0.0f;
    #pragma unroll 1
    for (int part = 0; part < partitions; ++part)
        value = rocm_native_quantised_accumulate_rn(
            value, partials[part * 32 + column]);
    return value;
}

/**
 * @brief Reduce one route's descriptor-local K tree, then apply router weight.
 * @param partials Original partition-major partial plane.
 * @param down_descs Original compact descriptor table.
 * @param runtime Original runtime bank for the runtime-table specialization.
 * @param expert_ids Original selected route IDs.
 * @param route_weights Original selected route weights; invalid descriptors never read them.
 * @param route_output Original public route-major weighted contribution plane.
 * @param num_active Original selected slot extent.
 * @param N Original H width.
 * @param num_experts Original descriptor table extent.
 * @param policy_partitions_0_7 Original source-policy partition entries0..7.
 * @param policy_partitions_8_15 Original source-policy partition entries8..15.
 * @param policy_partitions_16_23 Original source-policy partition entries16..23.
 * @param policy_partitions_24_31 Original source-policy partition entries24..31.
 * @param logical_tile Original64-column output tile.
 * @param logical_route Original selected slot, without compaction or reordering.
 * @param logical_lane Original64-thread column lane.
 */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ void moeDownOriginalWeightedFoldTask(
    const float* __restrict__ partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int logical_tile,int logical_route,int logical_lane)
{
    constexpr int TILE_N = 64;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    if (route >= num_active || n >= N)
        return;

    const size_t output_index =
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const DeviceNativeQuantisedMatrixDesc* desc = nullptr;
    if (!moe_native_quantised_load_down_descriptor<RUNTIME_TABLE>(
            down_descs, runtime, expert_ids[route], num_experts, desc))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int route_partitions =
        moe_native_quantised_descriptor_k_partitions(
            desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (route_partitions <= 0)
    {
        __builtin_trap();
        return;
    }

    float expert_sum = partials[output_index];
    if (route_partitions > 1)
    {
        expert_sum = 0.0f;
        #pragma unroll 1
        for (int part = 0; part < route_partitions; ++part)
        {
            const size_t partial_index =
                (static_cast<size_t>(part) *
                     static_cast<size_t>(num_active) +
                 static_cast<size_t>(route)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            expert_sum = rocm_native_quantised_accumulate_rn(
                expert_sum, partials[partial_index]);
        }
    }
    route_output[output_index] = rocm_native_quantised_weight_route_rn(
        route_weights[route], expert_sum);
}

template <bool RUNTIME_TABLE>
__global__ void moe_grouped_native_quantised_down_descriptor_reduce_kernel(
    const float* __restrict__ partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    moeDownOriginalWeightedFoldTask<RUNTIME_TABLE>(partials,down_descs,runtime,expert_ids,route_weights,route_output,num_active,N,num_experts,policy_partitions_0_7,policy_partitions_8_15,policy_partitions_16_23,policy_partitions_24_31,blockIdx.x,blockIdx.y,threadIdx.x);
}

/**
 * @brief Complete original down partitions, then weighted canonical routes, without waiting for an unscheduled CTA.
 * @details Physical128-thread CTAs own two adjacent ORIGINAL64-column tasks.
 * Each lane passes the exact original virtual tile/lane to unchanged helpers;
 * the source-policy grid-Z and every per-column partial/fold remain unchanged.
 * One whole-CTA barrier joins both original tasks before each release arrival.
 * The first unique publisher owns only one(tile,route); the second owns only one output tile. No CTA serializes H or
 * recomputes another producer. Invalid descriptors still publish original +0
 * and discharge both counter obligations. Both public planes survive unchanged.
 * All local lanes join before each device-scope release arrival and after each
 * unique acquire; nonfinal CTAs retire immediately. Stream order separates
 * epochs, and only the unique owner resets each counter before kernel retirement.
 */
__global__ __launch_bounds__(llaminar2::NativeDownCompletionPlan::columns_per_tile) void moe_grouped_native_quantised_down_last_producer_kernel(
    const int8_t* A,const float* scales,const DeviceNativeQuantisedMatrixDesc* descriptors,
    const int* expert_ids,const float* weights,float* partials,float* contributions,float* output,
    int routes,int N,int K,int experts,std::uint64_t p0,std::uint64_t p1,std::uint64_t p2,std::uint64_t p3,
    int maximum_partitions,llaminar2::NativeDownCompletionView completion)
{
    constexpr unsigned PhysicalColumns=llaminar2::NativeDownCompletionPlan::columns_per_tile;
    static_assert(PhysicalColumns==128);
    const unsigned tiles=static_cast<unsigned>(N)/PhysicalColumns+(static_cast<unsigned>(N)%PhysicalColumns!=0);
    const int original_tile=static_cast<int>(blockIdx.x)*2+static_cast<int>(threadIdx.x)/64;
    const int original_lane=static_cast<int>(threadIdx.x)%64;
    if(!completion.partitions.counters||!completion.routes||completion.partitions.tiles!=tiles||
        completion.partitions.routes!=static_cast<unsigned>(routes)||
        completion.partitions.contributors!=static_cast<unsigned>(maximum_partitions)||blockDim.x!=PhysicalColumns||
        blockIdx.x>=tiles||blockIdx.y>=completion.partitions.routes||blockIdx.z>=completion.partitions.contributors)
        __builtin_trap();
    moeDownOriginalPartialTask<false>(A,scales,descriptors,nullptr,expert_ids,partials,
        routes,N,K,experts,p0,p1,p2,p3,maximum_partitions,original_tile,blockIdx.y,blockIdx.z,original_lane);
    __syncthreads();
    __shared__ llaminar2::NativeTileCompletionDecision decision;
    auto* const partition_counter=completion.partitions.counters+
        static_cast<std::size_t>(blockIdx.y)*completion.partitions.tiles+blockIdx.x;
    if(threadIdx.x==0)
    {
        decision=llaminar2::native_tile_completion::arriveForPublication(partition_counter,completion.partitions.contributors);
        if(decision==llaminar2::NativeTileCompletionDecision::Invalid)__builtin_trap();
    }
    __syncthreads();
    if(decision!=llaminar2::NativeTileCompletionDecision::Publish)return;
    moeDownOriginalWeightedFoldTask<false>(partials,descriptors,nullptr,expert_ids,weights,contributions,
        routes,N,experts,p0,p1,p2,p3,original_tile,blockIdx.y,original_lane);
    __syncthreads();
    if(threadIdx.x==0)
    {
        llaminar2::native_tile_completion::reset(partition_counter);
        decision=llaminar2::native_tile_completion::arriveForPublication(
            completion.routes+blockIdx.x,completion.partitions.routes);
        if(decision==llaminar2::NativeTileCompletionDecision::Invalid)__builtin_trap();
    }
    __syncthreads();
    if(decision!=llaminar2::NativeTileCompletionDecision::Publish)return;
    const int n=original_tile*64+original_lane;
    if(n<N)
        output[n]=llaminar2::device_ordered_route_fold::sum(contributions+n,static_cast<std::size_t>(N),routes);
    __syncthreads();
    if(threadIdx.x==0)llaminar2::native_tile_completion::reset(completion.routes+blockIdx.x);
}

/**
 * @brief Original descriptor partial fold indexed independently of a native grid.
 * @param partials Original partition-major [part,route,column] plane.
 * @param route Original selected slot, never a compacted ordinal.
 * @param column Original output column.
 * @param routes Exact original selected slot count.
 * @param columns Original physical column stride.
 * @param partitions Original descriptor-local serial-M1 arithmetic count.
 * @return Existing increasing-partition RN result, including one-part signed zero.
 */
__device__ __forceinline__ float residentMoEOrderedPartial(const float* partials,int route,int column,
    int routes,int columns,int partitions)
{
    const auto first=static_cast<size_t>(route)*columns+column;
    if(partitions==1)return partials[first];
    float sum=0.f;
    #pragma unroll 1
    for(int part=0;part<partitions;++part)
        sum=rocm_native_quantised_accumulate_rn(sum,partials[(static_cast<size_t>(part)*routes+route)*columns+column]);
    return sum;
}

/** @brief Authenticate the exact physical role envelope and original source-policy entry; absence is fatal. */
__device__ __forceinline__ int residentMoEPartitions(const DeviceNativeQuantisedMatrixDesc* desc,
    llaminar2::rocm::ResidentMoEPartitionPolicy policy,std::uint32_t execution_mask)
{
    if(!desc||!llaminar2::rocm::residentMoEFFNDecoderAllows(execution_mask,desc->codebook_id))__builtin_trap();
    const int count=moe_native_quantised_descriptor_k_partitions(desc,policy.p0,policy.p1,policy.p2,policy.p3);
    if(count<=0||count>int(policy.maximum))__builtin_trap();
    return count;
}

/**
 * @brief Complete post-router local routed/shared FFN under native grid ordering.
 * @tparam TaskIndex Cold-selected uint32 or uint64 task scheduling, never tensor address arithmetic.
 * @tparam DecoderMask Complete immutable execution-envelope group union, independent of source arithmetic policy.
 * @param a Original source-authenticated descriptor/runtime and persistent arena views.
 * @details All CTAs reach every grid.sync. Each numerical task keeps its original
 * K partition and scalar contribution helper; changing physical resident workers
 * changes scheduling only. Four independent 64-column teams retain the original
 * producer wave layout inside each physical 256-thread CTA. Routed scratch dies
 * before the shared branch reuses it, matching the original nonconcurrent arena plan. Public intermediate
 * tensors and canonical route contributions retain their original values.
 */
// Shared descriptor addresses name original immutable arena tables. No helper
// takes an address into this by-value argument, avoiding a private kernarg copy.
template<class TaskIndex,std::uint32_t DecoderMask>
__global__ __launch_bounds__(256,1) void rocmResidentMoEFFNKernel(const llaminar2::rocm::ResidentMoEFFNArgs a)
{
    static_assert(DecoderMask!=0&&(DecoderMask&~llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders)==0);
    const auto grid=cooperative_groups::this_grid();
    if(!grid.is_valid())__builtin_trap();
    const unsigned lane=threadIdx.x;
    const unsigned team_lane=lane%64;
    // Shipping kernels compile with the original wave64 contract. The team is
    // uniform within that wave, allowing descriptor/task division in scalar
    // registers while four waves retain independent projection tasks.
    const auto team=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(lane/64)));
    const TaskIndex first_task=TaskIndex(blockIdx.x)*4+team;
    const TaskIndex task_stride=TaskIndex(gridDim.x)*4;
    using llaminar2::rocm::residentMoEFFNNextTask;
    using llaminar2::rocm::residentMoEFFNTaskCoordinates;
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    const int H=a.hidden_columns;
    const bool live=a.rows.activeRows()!=0;
    if(!live)
    {
        for(size_t i=size_t(blockIdx.x)*256+lane;i<size_t(a.top_k)*H;i+=size_t(gridDim.x)*256)
            a.route_contributions[i]=0.f;
        for(size_t n=size_t(blockIdx.x)*256+lane;n<size_t(H);n+=size_t(gridDim.x)*256)
        {a.routed_output[n]=0.f;}
    }
    if(live&&(!runtime||runtime->expert_count!=unsigned(a.experts)||runtime->top_k!=unsigned(a.top_k)||
        !runtime->intermediate_shard.sameIdentity(a.original)||
        moe_runtime_execution_bank(runtime)==kInvalidMoERuntimeExecutionBank))__builtin_trap();

    // The same original blockwise publisher serves both nonconcurrent branches.
    // Every Q8 group remains a complete original32-lane XOR maximum reduction.
    for(size_t block=size_t(blockIdx.x)*8+lane/32;block<size_t(H/32);block+=size_t(gridDim.x)*8)
    {
        const size_t n=block*32+lane%32;
        const float value=a.hidden[n];float maximum=fabsf(value);
        #pragma unroll
        for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor(maximum,mask));
        const float scale=llaminar2::rocm::decode_equivalent::q8Scale(maximum);
        a.hidden_q8[n]=static_cast<int8_t>(llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(value,scale));
        if((lane%32)==0)a.hidden_scales[block]=scale;
    }
    grid.sync();
    // Two source-declared branches execute sequentially because their original
    // arena planes are shared, never concurrent. This is not serial row replay.
    for(int shared=0;shared<2;++shared)
    {
        // Ordinary shared-table M1 computes its public FP32 gate/up planes
        // even when the final gate epilogue masks an inactive row. Retain that
        // publication while never dereferencing inactive routed slot state.
        if(!shared&&!live)continue;
        const int F=shared?a.shared_columns:a.routed_columns;
        const int routes=shared?1:a.top_k;
        const auto gate_policy=shared?a.shared_gate_policy:a.routed_gate_policy;
        const auto down_policy=shared?a.shared_down_policy:a.routed_down_policy;
        const auto gate_execution=shared?a.execution.shared_gate:a.execution.routed_gate;
        const auto down_execution=shared?a.execution.shared_down:a.execution.routed_down;
        const TaskIndex gate_tiles=(static_cast<unsigned>(F)+63u)/64u;
        const TaskIndex route_count=static_cast<unsigned>(routes);
        const TaskIndex gate_tasks=gate_tiles*route_count*gate_policy.maximum;
        for(TaskIndex task=first_task;task<gate_tasks;task=residentMoEFFNNextTask(task,gate_tasks,task_stride))
        {
            const auto coordinate=residentMoEFFNTaskCoordinates(task,gate_tiles,route_count);
            const int route=int(coordinate.route),part=int(coordinate.partition);
            const size_t n=size_t(coordinate.tile)*64+team_lane;
            const DeviceNativeQuantisedMatrixDesc *gate=nullptr,*up=nullptr;
            const bool ready=shared?(gate=a.shared_gate,up=a.shared_up,true):
                moe_native_quantised_load_gate_up_descriptors<true>(nullptr,nullptr,runtime,
                    runtime->topk_expert_ids[route],a.experts,gate,up);
            if(n<F)
            {
                const auto index=(size_t(part)*routes+route)*F+n;
                // Keep one all-format decoder callsite and retire each scalar
                // partial before the next projection. This phase alone writes
                // both original arena banks; consumers wait at the unchanged
                // following grid.sync. Only the accumulator live range changes.
                #pragma unroll 1
                for(int projection=0;projection<2;++projection)
                {
                    const auto* descriptor=projection?up:gate;
                    float value=0.f;
                    if(ready)
                    {
                        const int partitions=residentMoEPartitions(descriptor,gate_policy,gate_execution);
                        if(part<partitions)value=moe_native_quantised_block_partial<DecoderMask>(
                            descriptor,n,a.hidden_q8,a.hidden_scales,F,H,part,partitions);
                    }
                    (projection?a.up_partials:a.gate_partials)[index]=value;
                }
            }
        }
        grid.sync();
        const TaskIndex fold_tasks=gate_tiles*route_count;
        for(TaskIndex task=first_task;task<fold_tasks;task=residentMoEFFNNextTask(task,fold_tasks,task_stride))
        {
            const auto coordinate=residentMoEFFNTaskCoordinates(task,gate_tiles,route_count);
            const int route=int(coordinate.route);
            const size_t n=size_t(coordinate.tile)*64+team_lane;
            const DeviceNativeQuantisedMatrixDesc *gate=nullptr,*up=nullptr;
            const bool ready=shared?(gate=a.shared_gate,up=a.shared_up,true):
                moe_native_quantised_load_gate_up_descriptors<true>(nullptr,nullptr,runtime,
                    runtime->topk_expert_ids[route],a.experts,gate,up);
            float value=0.f;
            if(ready&&n<F)
            {
                const float g=residentMoEOrderedPartial(a.gate_partials,route,n,routes,F,residentMoEPartitions(gate,gate_policy,gate_execution));
                const float u=residentMoEOrderedPartial(a.up_partials,route,n,routes,F,residentMoEPartitions(up,gate_policy,gate_execution));
                if(shared){a.shared_gate_output[n]=g;a.shared_up_output[n]=u;}
                value=llaminar2::rocm::decode_equivalent::swigluValue(g,u);
            }
            float maximum=fabsf(value);
            #pragma unroll
            for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor(maximum,mask));
            if(ready&&n<F)
            {
                const float scale=llaminar2::rocm::decode_equivalent::q8Scale(maximum);
                const auto index=size_t(route)*F+n;
                a.swiglu_q8[index]=static_cast<int8_t>(llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(value,scale));
                if((lane%32)==0)a.swiglu_scales[size_t(route)*(F/32)+n/32]=scale;
            }
        }
        grid.sync();
        const TaskIndex down_tiles=(static_cast<unsigned>(H)+63u)/64u;
        const TaskIndex down_tasks=down_tiles*route_count*down_policy.maximum;
        for(TaskIndex task=first_task;task<down_tasks;task=residentMoEFFNNextTask(task,down_tasks,task_stride))
        {
            const auto coordinate=residentMoEFFNTaskCoordinates(task,down_tiles,route_count);
            const int route=int(coordinate.route),part=int(coordinate.partition);
            const size_t n=size_t(coordinate.tile)*64+team_lane;
            const DeviceNativeQuantisedMatrixDesc* down=nullptr;
            const bool ready=shared?(down=a.shared_down,true):moe_native_quantised_load_down_descriptor<true>(
                nullptr,runtime,runtime->topk_expert_ids[route],a.experts,down);
            if(n<H)
            {
                float value=0.f;
                if(ready)
                {
                    const int parts=residentMoEPartitions(down,down_policy,down_execution);
                    if(part<parts)value=moe_native_quantised_block_partial<DecoderMask>(down,n,a.swiglu_q8+size_t(route)*F,
                        a.swiglu_scales+size_t(route)*(F/32),H,F,part,parts);
                }
                a.down_partials[(size_t(part)*routes+route)*H+n]=value;
            }
        }
        grid.sync();
        for(TaskIndex task=first_task;task<down_tiles;task=residentMoEFFNNextTask(task,down_tiles,task_stride))
        {
            const size_t n=size_t(task)*64+team_lane;
            if(n<H)
            {
                float total=0.f;
                for(int route=0;route<routes;++route)
                {
                    const DeviceNativeQuantisedMatrixDesc* down=nullptr;
                    const bool ready=shared?(down=a.shared_down,true):moe_native_quantised_load_down_descriptor<true>(
                        nullptr,runtime,runtime->topk_expert_ids[route],a.experts,down);
                    float value=0.f;
                    if(ready)value=rocm_native_quantised_weight_route_rn(shared?1.f:runtime->topk_weights[route],
                        residentMoEOrderedPartial(a.down_partials,route,n,routes,H,residentMoEPartitions(down,down_policy,down_execution)));
                    if(!shared)a.route_contributions[size_t(route)*H+n]=value;
                    total=rocm_native_quantised_accumulate_rn(total,value);
                }
                if(shared)a.shared_output[n]=total;else a.routed_output[n]=total;
            }
        }
        grid.sync(); // Close the original scratch lifetime before reusing it.
    }
    if(blockIdx.x==0)llaminar2::rocm::moe_shared_gate::gateAdd<false>(a.hidden,a.shared_input_gate,a.shared_output,
        a.routed_output,a.combined_output,1,H,0,live?1:0);
}

#include "ROCmPersistentMoEFFNBody.hip.inc"
#include "ROCmOrdinaryMoEFFNBody.hip.inc"

namespace llaminar2::rocm
{
namespace
{
thread_local ResidentMoEFFNProgram g_resident_moe_ffn_program=ResidentMoEFFNProgram::Original;
/** @brief Name both exact scheduler entries for one explicit decoder-structure union. */
template<unsigned Groups>
const void* residentMoEFFNFunctionForWidth(ResidentMoEFFNSchedulerWidth width)
{
    constexpr auto mask=ResidentMoEFFNDecoderProgram::familyMask<Groups>();
    switch(width)
    {
    case ResidentMoEFFNSchedulerWidth::UInt32:
        return reinterpret_cast<const void*>(&rocmResidentMoEFFNKernel<std::uint32_t,mask>);
    case ResidentMoEFFNSchedulerWidth::UInt64:
        return reinterpret_cast<const void*>(&rocmResidentMoEFFNKernel<std::uint64_t,mask>);
    }
    throw std::invalid_argument("Resident FFN decoder program has an invalid scheduler width");
}
}
/** @copydoc currentResidentMoEFFNProgram */
ResidentMoEFFNProgram currentResidentMoEFFNProgram() noexcept {return g_resident_moe_ffn_program;}
/** @copydoc ScopedResidentMoEFFNProgram::ScopedResidentMoEFFNProgram */
ScopedResidentMoEFFNProgram::ScopedResidentMoEFFNProgram(ResidentMoEFFNProgram program)
    :previous_(g_resident_moe_ffn_program)
{
    if(::llaminar2::isGraphCaptureActive())throw std::logic_error("Resident FFN program must be selected before capture");
    if(program!=ResidentMoEFFNProgram::Original&&program!=ResidentMoEFFNProgram::PartitionLocal&&
       program!=ResidentMoEFFNProgram::OrdinaryPhases&&program!=ResidentMoEFFNProgram::NativeLayoutRouter)
        throw std::invalid_argument("Resident FFN program selection is unknown");
    g_resident_moe_ffn_program=program;
}
ScopedResidentMoEFFNProgram::~ScopedResidentMoEFFNProgram(){g_resident_moe_ffn_program=previous_;}

/** @copydoc residentMoEFFNKernelFunction */
const void* residentMoEFFNKernelFunction(ResidentMoEFFNDecoderProgram program,ResidentMoEFFNSchedulerWidth width)
{
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::PartitionLocal)
    {
        if(program.groups()!=1)
            throw std::invalid_argument("Partition-local FFN diagnostic requires the complete affine/byte execution envelope {5,7,19,23}");
        if(width!=ResidentMoEFFNSchedulerWidth::UInt32&&width!=ResidentMoEFFNSchedulerWidth::UInt64)
            throw std::invalid_argument("Partition-local FFN diagnostic scheduler width is unknown");
        return width==ResidentMoEFFNSchedulerWidth::UInt32
            ?reinterpret_cast<const void*>(&rocmPersistentMoEFFNKernel<std::uint32_t>)
            :reinterpret_cast<const void*>(&rocmPersistentMoEFFNKernel<std::uint64_t>);
    }
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::OrdinaryPhases)
        throw std::invalid_argument("Ordinary FFN has two native entries; use its exact ordinary program identity");
    // All fifteen source-envelope unions are real compiled entries. No admission
    // failure, current route or current physical residency changes this choice.
    switch(program.groups())
    {
    case 1:return residentMoEFFNFunctionForWidth<1>(width);
    case 2:return residentMoEFFNFunctionForWidth<2>(width);
    case 3:return residentMoEFFNFunctionForWidth<3>(width);
    case 4:return residentMoEFFNFunctionForWidth<4>(width);
    case 5:return residentMoEFFNFunctionForWidth<5>(width);
    case 6:return residentMoEFFNFunctionForWidth<6>(width);
    case 7:return residentMoEFFNFunctionForWidth<7>(width);
    case 8:return residentMoEFFNFunctionForWidth<8>(width);
    case 9:return residentMoEFFNFunctionForWidth<9>(width);
    case 10:return residentMoEFFNFunctionForWidth<10>(width);
    case 11:return residentMoEFFNFunctionForWidth<11>(width);
    case 12:return residentMoEFFNFunctionForWidth<12>(width);
    case 13:return residentMoEFFNFunctionForWidth<13>(width);
    case 14:return residentMoEFFNFunctionForWidth<14>(width);
    case 15:return residentMoEFFNFunctionForWidth<15>(width);
    }
    throw std::invalid_argument("Resident FFN decoder program is not in its complete registry");
}
namespace
{
/** @brief Resolve exactly the existing serial-M1 source-policy authority for either submission mechanism. */
void resolveResidentMoEFFNPolicies(ResidentMoEFFNArgs& args,ResidentMoEFFNPolicyMasks masks)
{
    const auto policy=[](int N,int K,std::uint32_t mask) {
        ROCmMoESerialM1PolicyTable original{};
        if(!buildROCmMoESerialM1PolicyTable(N,K,mask,original))
            throw std::invalid_argument("Resident FFN has no complete original serial-M1 partition policy");
        return ResidentMoEPartitionPolicy{original.k_partitions_0_7,original.k_partitions_8_15,
            original.k_partitions_16_23,original.k_partitions_24_31,original.maximum_k_partitions};
    };
    if(!args.runtime||!args.hidden||!args.shared_gate||!args.shared_up||!args.shared_down||
        !args.original.enabled()||args.rows.physicalRows()!=1||
        args.hidden_columns<=0||args.routed_columns<=0||args.shared_columns<=0||args.experts<=0||args.top_k<=0||
        args.top_k>kDeviceMoEMaxTopK||args.hidden_columns%32||args.routed_columns%32||args.shared_columns%32)
        throw std::invalid_argument("Resident FFN requires an original native M1 fragment and complete aligned geometry");
    if(!masks.execution.routed_gate||!masks.execution.routed_down||
        !masks.execution.shared_gate||!masks.execution.shared_down)
        throw std::invalid_argument("Resident FFN requires every original physical role envelope");
    args.execution=masks.execution;
    args.routed_gate_policy=policy(args.routed_columns,args.hidden_columns,masks.routed_gate);
    args.routed_down_policy=policy(args.hidden_columns,args.routed_columns,masks.routed_down);
    args.shared_gate_policy=policy(args.shared_columns,args.hidden_columns,masks.shared_gate);
    args.shared_down_policy=policy(args.hidden_columns,args.shared_columns,masks.shared_down);
}
/** @brief Cold exact ordinary function/geometry observation; residency is evidence, never a grid clamp. */
ResidentMoEFFNEntryEvidence prepareOrdinaryMoEFFNEntry(DeviceId device,const void* function,
    std::uint64_t grid,std::size_t dynamic_shared)
{
    const auto& hardware=rocmHardwareMetadata().at(device.ordinal);
    constexpr unsigned threads=256;
    // HSA dispatch packet work-item dimensions are uint32. The useful grid is
    // never padded or clamped to occupancy; an unrepresentable launch rejects.
    if(!function||!grid||grid>std::numeric_limits<unsigned>::max()/threads||
        hardware.maximum_threads_per_block<int(threads)||dynamic_shared>hardware.lds_per_block)
        throw std::invalid_argument("Ordinary FFN exceeds its exact native grid/CTA/LDS extent");
    int current=-1;
    if(hipGetDevice(&current)!=hipSuccess||current!=device.ordinal)
        throw std::logic_error("Ordinary FFN preparation has a foreign current worker");
    hipFuncAttributes attributes{};int resident_blocks=0;
    if(hipFuncGetAttributes(&attributes,function)!=hipSuccess||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,function,threads,dynamic_shared)!=hipSuccess||
        attributes.numRegs<=0||attributes.maxThreadsPerBlock<int(threads)||resident_blocks<=0||
        attributes.localSizeBytes!=0||attributes.sharedSizeBytes>hardware.lds_per_block-dynamic_shared)
        throw std::invalid_argument("Ordinary FFN function has invalid compiler resources or exact dynamic LDS residency");
    return {.function=function,.grid=unsigned(grid),.threads=threads,.dynamic_shared_bytes=dynamic_shared,
        .resources={unsigned(attributes.numRegs),unsigned(attributes.maxThreadsPerBlock),unsigned(resident_blocks),
            attributes.localSizeBytes,attributes.sharedSizeBytes}};
}
/** @brief Bind two useful ordinary work grids and exact local activation/fold storage from original policies. */
ResidentMoEFFNProgramEvidence prepareOrdinaryMoEFFNEvidence(DeviceId device,const ResidentMoEFFNArgs& args,
    ResidentMoEFFNDecoderProgram program,std::uint64_t generation)
{
    const auto tiles=[](int columns,unsigned parts) {
        if(!parts||parts>unsigned(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions))
            throw std::invalid_argument("Ordinary FFN source policy exceeds its original supported partition inventory");
        const auto width=256/parts;
        return (std::uint64_t(columns)+width-1)/width;
    };
    const auto grid_a=tiles(args.routed_columns,args.routed_gate_policy.maximum)*unsigned(args.top_k)+
        tiles(args.shared_columns,args.shared_gate_policy.maximum);
    const auto grid_b=(std::uint64_t(args.hidden_columns)+31)/32;
    const auto hidden=std::uint64_t(args.hidden_columns),values=std::uint64_t(args.top_k)*args.routed_columns+args.shared_columns;
    const auto dynamic_a=hidden+8*(hidden/32);
    const unsigned workers=std::min(8u,args.routed_down_policy.maximum);
    if(!workers||!args.shared_down_policy.maximum||
        args.routed_down_policy.maximum>unsigned(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)||
        args.shared_down_policy.maximum>unsigned(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions))
        throw std::invalid_argument("Ordinary FFN down source policy is incomplete");
    const auto partial_rows=std::max((8/workers)*args.routed_down_policy.maximum,args.shared_down_policy.maximum);
    const auto dynamic_b=values+8*(values/32)+std::uint64_t(32)*partial_rows*sizeof(float)+
        std::uint64_t(args.top_k)*32*sizeof(float);
    if(dynamic_a>std::numeric_limits<std::size_t>::max()||dynamic_b>std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Ordinary FFN exact dynamic LDS exceeds the native address type");
    const auto functions=ordinaryMoEFFNKernelFunctions(program);
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::OrdinaryPhases,.device=device,.generation=generation};
    evidence.entries[0]=prepareOrdinaryMoEFFNEntry(device,functions[0],grid_a,std::size_t(dynamic_a));
    evidence.entries[1]=prepareOrdinaryMoEFFNEntry(device,functions[1],grid_b,std::size_t(dynamic_b));
    return evidence;
}
}

namespace
{
/** @brief Reuse the existing finite source-envelope registry for both ordinary phases. */
template<unsigned Groups>
std::array<const void*,2> ordinaryMoEFFNFunctionsForFamily()
{
    constexpr auto mask=ResidentMoEFFNDecoderProgram::familyMask<Groups>();
    return {reinterpret_cast<const void*>(&rocmOrdinaryMoEFFNGateUpKernel<mask>),
        reinterpret_cast<const void*>(&rocmOrdinaryMoEFFNDownKernel<mask>)};
}
}
/** @copydoc ordinaryMoEFFNKernelFunctions */
std::array<const void*,2> ordinaryMoEFFNKernelFunctions(ResidentMoEFFNDecoderProgram program)
{
    if(!program.requiredMask()||(program.requiredMask()&~ResidentMoEFFNDecoderProgram::all_decoders))
        throw std::invalid_argument("Ordinary FFN requires a complete supported immutable execution envelope");
    switch(program.groups())
    {
    case 1:return ordinaryMoEFFNFunctionsForFamily<1>();
    case 2:return ordinaryMoEFFNFunctionsForFamily<2>();
    case 3:return ordinaryMoEFFNFunctionsForFamily<3>();
    case 4:return ordinaryMoEFFNFunctionsForFamily<4>();
    case 5:return ordinaryMoEFFNFunctionsForFamily<5>();
    case 6:return ordinaryMoEFFNFunctionsForFamily<6>();
    case 7:return ordinaryMoEFFNFunctionsForFamily<7>();
    case 8:return ordinaryMoEFFNFunctionsForFamily<8>();
    case 9:return ordinaryMoEFFNFunctionsForFamily<9>();
    case 10:return ordinaryMoEFFNFunctionsForFamily<10>();
    case 11:return ordinaryMoEFFNFunctionsForFamily<11>();
    case 12:return ordinaryMoEFFNFunctionsForFamily<12>();
    case 13:return ordinaryMoEFFNFunctionsForFamily<13>();
    case 14:return ordinaryMoEFFNFunctionsForFamily<14>();
    case 15:return ordinaryMoEFFNFunctionsForFamily<15>();
    }
    throw std::invalid_argument("Ordinary FFN source-envelope family is invalid");
}

/** @copydoc PreparedMoEFFNProgram::PreparedMoEFFNProgram */
PreparedMoEFFNProgram::PreparedMoEFFNProgram(ResidentMoEFFNProgramEvidence evidence,
    std::optional<NativeCooperativeKernelAdmission> cooperative)
    :evidence_(evidence),cooperative_(std::move(cooperative))
{
    if((evidence_.kind!=ResidentMoEFFNLaunchKind::Cooperative&&evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryPhases&&evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryResident)||
        !evidence_.device.is_rocm()||evidence_.device.ordinal<0||!evidence_.generation)
        throw std::invalid_argument("Prepared FFN has an incomplete typed native program identity");
    for(const auto& entry:evidence_.nativeEntries())
        if(!entry.function||!entry.grid||(entry.threads!=256&&entry.threads!=1024)||!entry.resources.registers_per_thread||
            entry.resources.maximum_threads_per_block<entry.threads||!entry.resources.blocks_per_processor||entry.resources.private_bytes)
            throw std::invalid_argument("Prepared FFN has incomplete native entry/geometry/resource evidence");
    if(evidence_.kind!=ResidentMoEFFNLaunchKind::OrdinaryPhases)
    {
        const auto& entry=evidence_.entries[0];
        if(!cooperative_||cooperative_->device()!=evidence_.device||cooperative_->generation()!=evidence_.generation||
            cooperative_->function()!=entry.function||cooperative_->threads()!=entry.threads||
            cooperative_->dynamicSharedBytes()!=entry.dynamic_shared_bytes||entry.grid>cooperative_->maximumGrid()||
            evidence_.entries[1].function)
            throw std::invalid_argument("Prepared FFN cooperative observations do not match its actual admission");
    }
    else if(cooperative_||evidence_.entries[0].function==evidence_.entries[1].function)
        throw std::invalid_argument("Prepared ordinary FFN must retain two real entries and no cooperative admission");
}

/** @copydoc prepareResidentMoEFFN */
PreparedMoEFFNProgram prepareResidentMoEFFN(DeviceId device,ResidentMoEFFNArgs& args,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation)
{
    if(!device.is_rocm()||!generation||isGraphCaptureActive())
        throw std::logic_error("FFN preparation requires its native owner generation outside capture");
    resolveResidentMoEFFNPolicies(args,masks);
    const auto program=ResidentMoEFFNDecoderProgram::fromExecutionMask(masks.execution.combined());
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::OrdinaryPhases)
    {
        args.resident_grid=0; // Ordinary useful grids belong to the typed entry evidence.
        return PreparedMoEFFNProgram(prepareOrdinaryMoEFFNEvidence(device,args,program,generation),std::nullopt);
    }
    // Producer maxima bound folds/SwiGLU and output folds as every original
    // policy has at least one partition. Hidden quantization needs eight Q8
    // groups/CTA, no more than ceil(H/64)/4 CTAs. No live count is read here.
    const std::array phases{
        ResidentMoEFFNTaskGeometry::projection(args.routed_columns,args.top_k,args.routed_gate_policy.maximum),
        ResidentMoEFFNTaskGeometry::projection(args.hidden_columns,args.top_k,args.routed_down_policy.maximum),
        ResidentMoEFFNTaskGeometry::projection(args.shared_columns,1,args.shared_gate_policy.maximum),
        ResidentMoEFFNTaskGeometry::projection(args.hidden_columns,1,args.shared_down_policy.maximum)};
    const auto& geometry=*std::max_element(phases.begin(),phases.end(),
        [](const auto& left,const auto& right){return left.tasks()<right.tasks();});
    std::uint64_t structural_tasks=0;
    if(currentResidentMoEFFNProgram()==ResidentMoEFFNProgram::PartitionLocal)
    {
        const auto tiles=[](int columns,unsigned partitions)->std::uint64_t {
            if(!partitions||partitions>NativeQuantisedGroupedDecodePolicy::maximum_k_partitions||partitions>256)
                throw std::invalid_argument("Partition-local FFN has an invalid original partition policy");
            const auto width=256/partitions;
            return (std::uint64_t(columns)+width-1)/width;
        };
        const auto gate=tiles(args.routed_columns,args.routed_gate_policy.maximum)*args.top_k+
            tiles(args.shared_columns,args.shared_gate_policy.maximum);
        const auto down=tiles(args.hidden_columns,args.routed_down_policy.maximum)*args.top_k+
            tiles(args.hidden_columns,args.shared_down_policy.maximum);
        const auto values=std::uint64_t(args.top_k)*args.routed_columns+args.shared_columns;
        const auto quant=(values+255)/256;
        const auto hidden=(std::uint64_t(args.hidden_columns)+255)/256;
        structural_tasks=std::max({gate,down,quant,hidden});
    }
    // Select the exact useful-CTA scheduler width BEFORE admitting its native
    // function. H/F/topK are positive signed32 source axes: their products and
    // summed routed/shared task counts fit uint64; no tensor address narrows.
    const auto width=structural_tasks?residentMoEFFNSchedulerWidthForExtent(structural_tasks):
        geometry.uses32BitScheduler()?ResidentMoEFFNSchedulerWidth::UInt32:ResidentMoEFFNSchedulerWidth::UInt64;
    const void* function=residentMoEFFNKernelFunction(program,width);
    const auto admission=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(device,
        function,256,0,generation);
    args.resident_grid=structural_tasks
        ?static_cast<unsigned>(std::min(std::uint64_t(admission.maximumGrid()),structural_tasks))
        :geometry.grid(admission.maximumGrid());
    ResidentMoEFFNProgramEvidence evidence{.kind=ResidentMoEFFNLaunchKind::Cooperative,.device=device,.generation=generation};
    evidence.entries[0]={admission.function(),args.resident_grid,admission.threads(),
        admission.dynamicSharedBytes(),admission.resources()};
    return PreparedMoEFFNProgram(evidence,admission);
}
/** @copydoc launchResidentMoEFFN */
void launchResidentMoEFFN(const PreparedMoEFFNProgram& program,const ResidentMoEFFNArgs& args,
    void* stream,std::uint64_t generation)
{
    const auto native=static_cast<hipStream_t>(stream);
    if(!native||native==hipStreamLegacy||native==hipStreamPerThread)
        throw std::invalid_argument("FFN program requires its exact non-default native stream");
    if(generation!=program.evidence_.generation)
        throw std::logic_error("FFN program owner generation changed before native submission");
    ResidentMoEFFNArgs copied=args;void* parameters[]{&copied};
    if(program.evidence_.kind==ResidentMoEFFNLaunchKind::Cooperative)
    {
        if(args.resident_grid!=program.evidence_.entries[0].grid)
            throw std::logic_error("Cooperative FFN changed its sealed native grid");
        NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::launch(*program.cooperative_,parameters,
            args.resident_grid,stream,generation);
        return;
    }
    for(const auto& entry:program.evidence_.nativeEntries())
    {
        const auto status=hipLaunchKernel(entry.function,dim3(entry.grid),dim3(entry.threads),parameters,
            entry.dynamic_shared_bytes,native);
        if(status!=hipSuccess)throw std::runtime_error(std::string("ordinary FFN native phase launch failed: ")+hipGetErrorString(status));
    }
}
}

// ============================================================================
// Reduce kernel: sum K-partitions from scatter buffer to final output.
// Simple summation — no per-column scales or bias (handled externally).
//
// Grid: dim3((N + TILE_N - 1) / TILE_N), Block: dim3(TILE_N / CPT)
// ============================================================================
/**
 * @brief Fold one original tile in the unchanged ascending-KB explicit-RN order.
 * @tparam TILE_N Original reducer tile width.
 * @tparam CPT Original independent columns per lane.
 * @param d_partial_fp32 Original producer's distinct [KB][N] partial bank.
 * @param d_C_fp32 Original output owner.
 * @param N Original physical width.
 * @param kblocks Original resolved arithmetic partition count.
 * @param tile Projection-local output tile.
 * @param lane Original workgroup lane.
 */
/** @brief One original increasing-partition RN reducer result. @param partials Original plane.
 * @param n Original output column. @param N Original column stride. @param kblocks Exact original partitions. @return Original bits. */
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

template <int TILE_N, int CPT, bool HasWork = false>
__global__ void gemv_native_quantised_reduce_kernel_t(const float *__restrict__ partials,
    float *__restrict__ output,int N,int partitions, llaminar2::ProjectionRowWork work)
{
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return; // Same ordinary M1 route; no unauthored reads/writes.

    nativeM1FoldColumns<TILE_N,CPT>(partials,output,N,partitions,blockIdx.x,threadIdx.x);
}

/**
 * @brief Keep the original128-column reducer grid and publish its original HC output locally.
 * @tparam HasWork Exact original projection work mode.
 * @param partials Original partition bank. @param output Public raw-down output.
 * @param N Original output width. @param partitions Unchanged source partitions.
 * @param work Original projection work, independent of publisher count0.
 * @param publication Existing activated and four Q8 bank destinations.
 */
template<bool HasWork>
__global__ __launch_bounds__(128) void nativeM1HCDownReducePublish(const float* partials,float* output,
 int N,int partitions,llaminar2::ProjectionRowWork work,llaminar2::GPUHCDownOutputView publication) {
 if constexpr(HasWork)if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
 const unsigned n=blockIdx.x*128+threadIdx.x;
 // N is a multiple of32: each tail subgroup is wholly absent or wholly active.
 if(n>=static_cast<unsigned>(N))return;
 const float raw=nativeM1FoldColumnValue(partials,n,N,partitions);
 output[n]=raw;
 if(publication.rows.activeRows()!=0)llaminar2::hc_down_publication::publish(raw,publication,n);
}

/** @brief Concatenate original live reducer tiles without extending any projection. */
template <int TILE_N,int CPT, bool HasWork = false>
__global__ void nativeM1GroupedReduce(llaminar2::rocm::NativeM1ReductionGrid group)
{
    int p=0;
    #pragma unroll
    for(int i=1;i<4;++i)if(i<group.count && int(blockIdx.x)>=group.offsets[i])p=i;
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(group.work[p],0))return;
    nativeM1FoldColumns<TILE_N,CPT>(group.partials[p],group.outputs[p],group.columns[p],
        group.partitions[p],blockIdx.x-group.offsets[p],threadIdx.x);
}

namespace llaminar2::rocm
{
/** @brief Only native producer submission can issue a pending reduction authority. */
struct NativeM1ReductionIssuer
{
    /** @brief Reject reuse before any GPU submission. */
    static void requireEmpty(const NativeM1ReductionTicket &ticket)
    {
        if(ticket.state_!=NativeM1ReductionTicket::State::Empty)
            throw std::invalid_argument("M1 producer ticket was already issued");
    }
    /** @brief Publish a ticket only after the exact original producer launch succeeded. */
    static void issue(NativeM1ReductionTicket &ticket,const float *partials,float *output,
        int n,int kb,int tile,int cpt,int device,void *stream,llaminar2::ProjectionRowWork work,bool has_work)
    {
        requireEmpty(ticket);
        ticket.shape_={n,kb,tile,cpt};ticket.partials_=partials;ticket.output_=output;
        ticket.device_=device;ticket.stream_=stream;ticket.work_=work;
        ticket.has_work_=has_work;
        ticket.state_=kb>1?NativeM1ReductionTicket::State::Pending:NativeM1ReductionTicket::State::Direct;
    }
};
}

/** @brief Retain ordinary submission or issue its exact successful producer ticket. */
template <int TN,int CPT,bool HasWork>
void finishNativeM1Producer(float *partials,float *output,int n,int kb,int device,void *stream,
    llaminar2::rocm::NativeM1ReductionTicket *deferred,llaminar2::ProjectionRowWork work)
{
    if(deferred)
    {
        if(hipPeekAtLastError()!=hipSuccess)throw std::runtime_error("M1 deferred producer submission failed");
        llaminar2::rocm::NativeM1ReductionIssuer::issue(*deferred,partials,output,n,kb,TN,CPT,device,stream,work,HasWork);
    }
    else if(kb>1)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_quantised_reduce_kernel_t<TN,CPT,HasWork>),
            dim3(1+(n-1)/TN),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),partials,output,n,kb,work);
}



/**
 * @brief Compute independent original Q8 K32 contributions in bounded CTA-owned windows.
 * @tparam HasWork Borrowed original device row authority; false retains the plain M1 ABI.
 * @param A Original quantized input. @param payload Canonical block-major native Q8 bytes.
 * @param scales Original FP16 weight scales. @param output Original public output for P1.
 * @param partials Original [P][N] partial bank. @param scale_a Original optional row scale.
 * @param block_scales Original optional FP32 K32 scales. @param N Physical output width.
 * @param K Original reduction width. @param partitions Exact source-resolved partition count.
 * @param work Original immutable row-work binding, copied into the retained graph.
 *
 * Four contributor groups own32 adjacent columns each. A lane computes complete
 * original signed-dot8 chains and the SAME rounded K32 contribution helper as
 * nativeM1ColumnsBody. Owner group0 folds each real contribution in ascending K
 * order, retaining its accumulator across windows. Never sum contributor-local
 * FP32 accumulators: that would change the source's expression tree. The second
 * local barrier retires every LDS reader before the next window overwrites it.
 *
 * The bounded32x32 FP32 array is explicit4KiB LDS, never per-thread/private
 * storage. Arbitrarily long admitted K partitions use repeated windows. Tail
 * columns and empty partition planes reach the identical barrier structure;
 * empty planes retain the original positive-zero initialization and final scale
 * multiply. No synthetic zero contribution enters an active owner's fold.
 */
template<bool HasWork>
__global__ __launch_bounds__(128) void gemv_native_q8_k32_contribution_tile_kernel_t(
    const int8_t* __restrict__ A,const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scales,float* __restrict__ output,float* __restrict__ partials,
    const float* __restrict__ scale_a,const float* __restrict__ block_scales,
    int N,int K,int partitions,llaminar2::ProjectionRowWork work)
{
    if constexpr(HasWork)
        if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
    constexpr int Columns=32,Contributors=4,WindowBlocks=32;
    __shared__ float contributions[WindowBlocks][Columns];
    const unsigned group=threadIdx.x/Columns,column=threadIdx.x%Columns;
    const int n=static_cast<int>(blockIdx.x)*Columns+static_cast<int>(column);
    const int blocks=K/32,per=(blocks+partitions-1)/partitions;
    const int first=static_cast<int>(blockIdx.y)*per,last=min(first+per,blocks);
    float accumulator=0.0f;
    for(int begin=first;begin<last;begin+=WindowBlocks)
    {
        const int count=min(WindowBlocks,last-begin);
        for(int offset=static_cast<int>(group);offset<count;offset+=Contributors)
        {
            if(n<N)
            {
                const int b=begin+offset;
                const auto linear=static_cast<size_t>(b)*static_cast<size_t>(N)+static_cast<size_t>(n);
                const auto* weights=reinterpret_cast<const int32_t*>(payload+linear*32);
                const auto* activation=reinterpret_cast<const int32_t*>(A+b*32);
                int32_t dot=0;
#pragma unroll
                for(int word=0;word<8;++word)
                    dot=::llaminar2::rocm_integer_contract::signedDot4(activation[word],weights[word],dot);
                const float scale=rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scales[linear],uint16_t{0});
                contributions[offset][column]=::llaminar2::device_native_quantised_contract::singleScaleBlock(
                    dot,scale,block_scales?block_scales[b]:1.0f);
            }
        }
        __syncthreads();
        if(group==0&&n<N)
            for(int offset=0;offset<count;++offset)
                accumulator=rocm_native_quantised_accumulate_rn(accumulator,contributions[offset][column]);
        __syncthreads();
    }
    if(group==0&&n<N)
    {
        const float result=accumulator*(block_scales?1.0f:scale_a[0]);
        if(partitions==1)output[n]=result;
        else partials[static_cast<size_t>(blockIdx.y)*N+n]=result;
    }
}

namespace
{
/** @brief Exact loaded producer/reducer pair for one existing template geometry. */
struct NativeQ8SerialSymbols
{
    const void *producer;
    const void *reducer;
    const char *producer_name;
    const char *reducer_name;
};

/**
 * @brief Resolve the same symbols the launch builder will enqueue.
 * @tparam TN Output columns per workgroup.
 * @tparam CPT Independent output columns per lane.
 * @return Loaded HIP symbols and stable readable specialization spellings.
 */
template <int TN, int CPT, bool HasWork = false>
NativeQ8SerialSymbols nativeQ8SerialSymbols()
{
#define NQ8_SYMBOL_NAMES(N, C) \
    HasWork ? "gemv_native_quantised_kernel_t<" #N "," #C ",NATIVE_QUANTISED_Q8_0,true>" : "gemv_native_quantised_kernel_t<" #N "," #C ",NATIVE_QUANTISED_Q8_0,false>", \
    HasWork ? "gemv_native_quantised_reduce_kernel_t<" #N "," #C ",true>" : "gemv_native_quantised_reduce_kernel_t<" #N "," #C ",false>"
    const char *producer_name = nullptr;
    const char *reducer_name = nullptr;
#define NQ8_NAME_CASE(N, C) \
    if constexpr (TN == N && CPT == C) { \
        const char *names[] = {NQ8_SYMBOL_NAMES(N, C)}; \
        producer_name = names[0]; reducer_name = names[1]; }
    NQ8_NAME_CASE(64, 1)
    NQ8_NAME_CASE(128, 1)
    NQ8_NAME_CASE(256, 1)
    NQ8_NAME_CASE(128, 2)
    NQ8_NAME_CASE(256, 2)
#undef NQ8_NAME_CASE
#undef NQ8_SYMBOL_NAMES
    static_assert((TN == 64 && CPT == 1) || (TN == 128 && (CPT == 1 || CPT == 2)) ||
                  (TN == 256 && (CPT == 1 || CPT == 2)));
    return {reinterpret_cast<const void *>(gemv_native_quantised_kernel_t<TN, CPT, NATIVE_QUANTISED_Q8_0, HasWork>),
            reinterpret_cast<const void *>(gemv_native_quantised_reduce_kernel_t<TN, CPT, HasWork>),
            producer_name, reducer_name};
}

/**
 * @brief Resolve a diagnostic load schedule while retaining the existing reducer symbol.
 * @tparam TN Original independent-column tile. @tparam CPT Original columns per lane.
 * @tparam HasWork Exact original row-work mode. @param loads Explicit cold memory schedule.
 * @return Original pair or the admitted lookahead producer with the SAME reducer.
 * @throws std::invalid_argument For an unknown/unimplemented diagnostic tuple.
 */
template<int TN,int CPT,bool HasWork>
NativeQ8SerialSymbols nativeQ8SerialSymbols(llaminar2::rocm::NativeQ8SerialProducerLoads loads)
{
    using Loads=llaminar2::rocm::NativeQ8SerialProducerLoads;
    auto result=nativeQ8SerialSymbols<TN,CPT,HasWork>();
    if(loads==Loads::Original)return result;
    if(loads==Loads::K32ContributionTile4x32)
    {
        if constexpr(TN==128&&CPT==1)
        {
            result.producer=reinterpret_cast<const void*>(gemv_native_q8_k32_contribution_tile_kernel_t<HasWork>);
            result.producer_name=HasWork?"gemv_native_q8_k32_contribution_tile_kernel_t<true>":"gemv_native_q8_k32_contribution_tile_kernel_t<false>";
            return result; // Original128-column reducer remains the same symbol.
        }
        throw std::invalid_argument("native-Q8 K32 contribution tile requires original128 CPT1 reducer geometry");
    }
    if(loads!=Loads::TwoK32Lookahead)
        throw std::invalid_argument("unknown native-Q8 producer load schedule");
    if constexpr(CPT==1&&(TN==64||TN==128))
    {
        result.producer=reinterpret_cast<const void*>(gemv_native_q8_two_k32_lookahead_kernel_t<TN,HasWork>);
        if constexpr(TN==64)result.producer_name=HasWork?"gemv_native_q8_two_k32_lookahead_kernel_t<64,true>":"gemv_native_q8_two_k32_lookahead_kernel_t<64,false>";
        else result.producer_name=HasWork?"gemv_native_q8_two_k32_lookahead_kernel_t<128,true>":"gemv_native_q8_two_k32_lookahead_kernel_t<128,false>";
        return result;
    }
    throw std::invalid_argument("native-Q8 lookahead requires admitted64/128 CPT1 geometry");
}

/**
 * @brief Select one exact compiled pair; unknown enum values fail before HIP calls.
 * @param geometry Typed requested geometry.
 * @param loads Exact cold-admitted producer schedule.
 * @return The function pair shared with the actual launch dispatch.
 */
NativeQ8SerialSymbols nativeQ8SerialSymbols(llaminar2::rocm::NativeQ8SerialGeometry geometry,
    llaminar2::rocm::NativeQ8SerialProducerLoads loads)
{
    return llaminar2::dispatchProjectionWorkMode(1,[&]<bool HasWork>() {
    using G = llaminar2::rocm::NativeQ8SerialGeometry;
    switch (geometry)
    {
    case G::TN64CPT1: return nativeQ8SerialSymbols<64, 1, HasWork>(loads);
    case G::TN128CPT1: return nativeQ8SerialSymbols<128, 1, HasWork>(loads);
    case G::TN256CPT1: return nativeQ8SerialSymbols<256, 1, HasWork>(loads);
    case G::TN128CPT2: return nativeQ8SerialSymbols<128, 2, HasWork>(loads);
    case G::TN256CPT2: return nativeQ8SerialSymbols<256, 2, HasWork>(loads);
    }
    throw std::invalid_argument("unknown ROCm native-Q8 serial geometry");
    });
}

/**
 * @brief Authenticate a loaded specialization at its exact block width.
 * @param function Exact symbol selected by nativeQ8SerialSymbols().
 * @param name Static readable template spelling; function remains authority.
 * @param threads Physical workgroup width, not a target-wave heuristic.
 * @return Complete compiler and occupancy evidence with zero private memory.
 * @throws std::runtime_error on incomplete, nonresident or spilling resources.
 */
llaminar2::rocm::NativeQ8SerialKernelResources inspectNativeQ8SerialSymbol(
    const void *function, const char *name, int threads)
{
    hipFuncAttributes attributes{};
    int active_blocks = 0;
    if (!function || !name || threads <= 0 ||
        hipFuncGetAttributes(&attributes, function) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, function, threads, 0) != hipSuccess ||
        attributes.numRegs <= 0 || attributes.maxThreadsPerBlock < threads ||
        active_blocks <= 0 || attributes.localSizeBytes != 0)
        throw std::runtime_error(std::string("native-Q8 serial resource admission failed: ") + (name ? name : "missing symbol"));
    return {.function = function, .specialization = name, .block_threads = threads,
            .registers_per_thread = attributes.numRegs,
            .local_memory_bytes_per_thread = attributes.localSizeBytes,
            .static_shared_memory_bytes = attributes.sharedSizeBytes,
            .max_threads_per_block = attributes.maxThreadsPerBlock,
            .max_active_blocks_per_sm = active_blocks};
}

/**
 * @brief Enqueue the existing serial producer and ordered reducer at one geometry.
 * @tparam TN Independent output columns owned by one workgroup.
 * @tparam CPT Independent output chains per thread, one or two.
 * @tparam FMT Existing compiled decoder; new geometries instantiate Q8 only.
 * @param A Existing quantized activation bytes.
 * @param payload Existing prepared native weight payload.
 * @param scales Existing prepared primary scale metadata.
 * @param mins Existing prepared optional asymmetric metadata.
 * @param emins Existing prepared optional extended metadata.
 * @param output Live FP32 output vector.
 * @param partials Existing persistent [KB][N] partial workspace.
 * @param scale_a Existing per-row activation scale, when used.
 * @param block_scales Existing per-block activation scales, when used.
 * @param N Physical descriptor/output width.
 * @param K Complete unchanged reduction width.
 * @param kb Resolved production arithmetic partition count.
 * @param arithmetic_policy Source codebook owning the arithmetic tree.
 * @param device Exact HIP device.
 * @param stream Exact non-null producer stream.
 * @param geometry Actual selected geometry for the launch receipt.
 * @note No resource queries, allocations, transfers or waits occur here.
 */
template <int TN, int CPT, NativeQuantisedFormat FMT>
void launchNativeQuantisedSerialGeometry(
    const int8_t *A, const uint8_t *payload, const uint16_t *scales,
    const uint16_t *mins, const uint32_t *emins, float *output, float *partials,
    const float *scale_a, const float *block_scales, int N, int K, int kb,
    uint8_t arithmetic_policy, int device, void *stream,
    llaminar2::rocm::NativeQ8SerialGeometry geometry,
    llaminar2::rocm::NativeM1ReductionTicket *deferred=nullptr,
    const llaminar2::rocm::NativeQ8SerialGeometryAdmission *prepared=nullptr)
{
    static_assert(CPT == 1 || CPT == 2);
    static_assert(TN % CPT == 0);
    using Loads=llaminar2::rocm::NativeQ8SerialProducerLoads;
    const auto loads=prepared?prepared->producerLoads():
        g_native_q8_serial_scope?g_native_q8_serial_scope->admission().producerLoads():Loads::Original;
    const auto producer_spec=llaminar2::rocm::nativeQ8SerialProducerSpec(geometry,loads);
    const int grid_n = 1 + (N - 1) / (FMT==NATIVE_QUANTISED_Q8_0?producer_spec.tile_columns:TN);
    const dim3 block(TN / CPT, 1, 1);
    const auto work=llaminar2::projectionWorkForLaunch(1);
    return llaminar2::dispatchProjectionWorkMode(1,[&]<bool HasWork>() {
        if(prepared) {
            // Cold owner admission, not a capture-time scope or hardware query.
            // Exact function comparison also authenticates the original row-work
            // mode: a false-mode certificate cannot launch a counted program.
            const auto symbols=nativeQ8SerialSymbols<TN,CPT,HasWork>(loads);
            if(g_native_q8_serial_scope||FMT!=NATIVE_QUANTISED_Q8_0||TN!=128||CPT!=1||
                prepared->deviceOrdinal()!=device||prepared->geometry()!=geometry||loads!=Loads::TwoK32Lookahead||
                prepared->producerResources().function!=symbols.producer||
                prepared->reducerResources().function!=symbols.reducer||
                prepared->producerResources().block_threads!=TN/CPT)
                throw std::logic_error("HC lookahead producer changed its exact cold program/device/work geometry");
        }

        if constexpr(FMT==NATIVE_QUANTISED_Q8_0&&CPT==1&&(TN==64||TN==128))
        {
            if(loads==Loads::K32ContributionTile4x32)
            {
                if constexpr(TN==128)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_q8_k32_contribution_tile_kernel_t<HasWork>),
                        dim3(grid_n,kb,1),dim3(producer_spec.block_threads),0,static_cast<hipStream_t>(stream),
                        A,payload,scales,output,partials,scale_a,block_scales,N,K,kb,work);
                else throw std::invalid_argument("native-Q8 K32 contribution tile requires original128 CPT1 reducer geometry");
            }
            else if(loads==Loads::TwoK32Lookahead)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_q8_two_k32_lookahead_kernel_t<TN,HasWork>),
                    dim3(grid_n,kb,1),block,0,static_cast<hipStream_t>(stream),
                    A,payload,scales,output,partials,scale_a,block_scales,N,K,kb,work);
            else if(loads==Loads::Original)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_quantised_kernel_t<TN,CPT,FMT,HasWork>),
                    dim3(grid_n,kb,1),block,0,static_cast<hipStream_t>(stream),
                    A,payload,scales,mins,emins,output,partials,scale_a,block_scales,N,K,kb,work);
            else throw std::invalid_argument("unknown native-Q8 producer load schedule");
        }
        else
        {
            if(loads!=Loads::Original)throw std::invalid_argument("native-Q8 lookahead requires admitted64/128 CPT1 source");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_quantised_kernel_t<TN,CPT,FMT,HasWork>),
                dim3(grid_n,kb,1),block,0,static_cast<hipStream_t>(stream),
                A,payload,scales,mins,emins,output,partials,scale_a,block_scales,N,K,kb,work);
        }
    if (g_native_q8_serial_scope && hipPeekAtLastError() != hipSuccess)
    {
        g_native_q8_serial_launch.reset();
        throw std::runtime_error("native-quantised serial producer submission failed");
    }
    // TN/CPT retain the original source/reducer ticket geometry. K-cooperative
    // producer ownership is independently authenticated by its physical receipt.
    finishNativeM1Producer<TN,CPT,HasWork>(partials,output,N,kb,device,stream,deferred,work);
    if (g_native_q8_serial_scope && hipPeekAtLastError() != hipSuccess)
    {
        g_native_q8_serial_launch.reset();
        throw std::runtime_error("native-quantised serial ordered-reducer submission failed");
    }
    if constexpr (FMT == NATIVE_QUANTISED_Q8_0)
    {
        const auto symbols = nativeQ8SerialSymbols<TN,CPT,HasWork>(loads);
        const int b_per = 1 + (K / 32 - 1) / kb;
        const int nonempty = 1 + (K / 32 - 1) / b_per;
        const auto launch = [&](const void *function, const char *name, int grid_x,int grid_y,int threads)
        {
            return llaminar2::rocm::NativeQ8SerialKernelLaunch{
                function, name, grid_x, grid_y, 1, threads, 1, 1};
        };
        // Publish topology only. Replays never update this host observation.
        g_native_q8_serial_launch = llaminar2::rocm::NativeQ8SerialLaunch{
            .geometry = geometry, .producer_loads=loads, .physical_n = N,
            .arithmetic_policy_n = g_native_quantised_serial_partition_n > 0
                ? g_native_quantised_serial_partition_n : N,
            .k = K, .device_ordinal = device, .execution_codebook = 19,
            .arithmetic_policy_codebook = arithmetic_policy, .k_partitions = kb,
            .blocks_per_partition = b_per, .nonempty_partitions = nonempty,
            .partial_elements = kb > 1 ? std::size_t(kb) * N : 0u,
            .stream = stream,
            .producer = launch(symbols.producer, symbols.producer_name,grid_n,kb,producer_spec.block_threads),
            .reducer = kb > 1 && !deferred ? launch(symbols.reducer, symbols.reducer_name,1+(N-1)/TN,1,TN/CPT)
                : llaminar2::rocm::NativeQ8SerialKernelLaunch{}};
    }
    });
}
} // namespace

llaminar2::rocm::NativeQ8SerialGeometryAdmission
llaminar2::rocm::admitNativeQ8SerialGeometry(NativeQ8SerialGeometry geometry,int device_ordinal,
    NativeQ8SerialProducerLoads producer_loads)
{
    const auto spec = nativeQ8SerialGeometrySpec(geometry);
    const auto producer_spec=nativeQ8SerialProducerSpec(geometry,producer_loads);
    const auto symbols = nativeQ8SerialSymbols(geometry,producer_loads);
    if (device_ordinal < 0)
        throw std::invalid_argument("native-Q8 serial resource admission needs an exact device");
    if (llaminar2::isGraphCaptureActive())
        throw std::logic_error("native-Q8 serial resources must be inspected before capture");
    if (llaminar2::HipDeviceGuard::setDevice(device_ordinal) != 0)
        throw std::runtime_error("native-Q8 serial resource admission could not select device");
    return {geometry, device_ordinal,
        inspectNativeQ8SerialSymbol(symbols.producer, symbols.producer_name, producer_spec.block_threads),
        inspectNativeQ8SerialSymbol(symbols.reducer, symbols.reducer_name, spec.block_threads),producer_loads};
}

/**
 * @brief Publishes one runtime verifier row from ordered split-K partials.
 *
 * Grid Y owns a bounded row worker, not an allocation-capacity row. A counted
 * replay visits only live rows by grid stride. The reducer's register state
 * and ascending-KB addition order do not depend on speculative capacity; this
 * is serial M=1's reduction applied to each row's `[KB][M][N]` partial slice.
 */
template <int TILE_N, int CPT, bool HasWork = false>
__global__ void gemv_native_quantised_reduce_small_m_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB x M x N] partial buffer
    float* __restrict__ d_C_fp32,                // [M x N] output
    int M, int N, int kblocks, llaminar2::DeviceRowRange rows, llaminar2::ProjectionRowWork work)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    // The count is borrowed once. Physical M remains the split-K plane stride
    // after a large-to-small replay; no inactive partial is read or published.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(work,rows,int(blockIdx.y),int(gridDim.y),1);
    int row,active_m;
    while(rowWork.next(row,active_m))
    {
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
                sum = rocm_native_quantised_accumulate_rn(
                    sum, d_partial_fp32[idx]);
            }
            d_C_fp32[static_cast<size_t>(row) * N + n] = sum;
        }
    }
}

namespace
{
constexpr int kHCDownColumns = 320;
constexpr int kHCDownReduction = 10240;
constexpr int kHCDownPartitions = 16;
constexpr int kHCDownPackedBlocks = kHCDownReduction / 32;
constexpr int kHCDownWindow = kHCDownPackedBlocks / kHCDownPartitions;
constexpr int kHCDownTileRows = 4;
constexpr int kHCDownTileColumns = 32;

/** @brief Existing counted row owner and persistent prepared GEMV storage. */
struct HCDownLongKArgs
{
    const int8_t *activation;
    const uint8_t *payload;
    const uint16_t *weight_scales;
    const float *activation_scales;
    float *partials;
    llaminar2::DeviceRowRange rows;
};

/**
 * @brief Produce the original FP32 K32 × 20 terms for one [partition,row,column] tile.
 *
 * Every lane decodes one physical Q8 block, reuses it across the live row
 * tile and publishes one typed single-scale term per live row. Four lanes
 * fold those terms in increasing original block order. The unchanged public
 * reducer consumes ascending partitions with the existing physical row stride.
 */
__global__ __launch_bounds__(kHCDownTileColumns * kHCDownWindow)
void qfn_hc_down_long_k_contributions(HCDownLongKArgs args)
{
    const int row_base = int(blockIdx.z) * kHCDownTileRows;
    const int width = min(kHCDownTileRows, args.rows.activeRows() - row_base);
    if (width <= 0)
        return;

    const int thread = int(threadIdx.x);
    const int local_group = thread / kHCDownTileColumns;
    const int local_col = thread % kHCDownTileColumns;
    const int group = int(blockIdx.y) * kHCDownWindow + local_group;
    const int col = int(blockIdx.x) * kHCDownTileColumns + local_col;
    const int weight_index = group * kHCDownColumns + col;
    const auto *packed = reinterpret_cast<const int4 *>(args.payload + weight_index * 32);
    const int4 lo = packed[0], hi = packed[1];
    const int weight[] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const float scale = __half2float(__ushort_as_half(args.weight_scales[weight_index]));
    __shared__ float terms[kHCDownTileRows][kHCDownWindow][kHCDownTileColumns];
#pragma unroll
    for (int row = 0; row < kHCDownTileRows; ++row)
    {
        if (row >= width)
            break;
        const int input_index = (row_base + row) * kHCDownPackedBlocks + group;
        const auto *input = reinterpret_cast<const int *>(args.activation + input_index * 32);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            dot = llaminar2::rocm_integer_contract::signedDot4(input[i], weight[i], dot);
        terms[row][local_group][local_col] =
            llaminar2::device_native_quantised_contract::singleScaleBlock(
                dot, scale, args.activation_scales[input_index]);
    }
    __syncthreads();

    // Each live scratch word has one writer. Inactive words remain dirty.
    if (local_group < width)
    {
        float sum = 0.0f;
#pragma unroll
        for (int group_offset = 0; group_offset < kHCDownWindow; ++group_offset)
            sum = llaminar2::device_native_quantised_contract::accumulate(
                sum, terms[local_group][group_offset][local_col]);
        const auto plane = int(blockIdx.y) * args.rows.physicalRows() + row_base + local_group;
        args.partials[plane * kHCDownColumns + col] = sum;
    }
}

/** @brief Preserve all serial-rounded terms and publish original SiLU/Q8 planes locally. */
__global__ void groupedHCDownReducePublish(
    const float* partials, float* raw, llaminar2::GPUHCDownOutputView publication)
{
    const int column = int(blockIdx.x) * 64 + int(threadIdx.x);
    const int row = int(blockIdx.y);
    if (row >= publication.rows.activeRows())
        return;
    const int offset = row * kHCDownColumns + column;
    float sum = 0.0f;
#pragma unroll
    for (int partition = 0; partition < kHCDownPartitions; ++partition)
        sum = llaminar2::device_native_quantised_contract::accumulate(
            sum, partials[partition * publication.rows.physicalRows() * kHCDownColumns + offset]);
    raw[offset] = sum;
    llaminar2::hc_down_publication::publish(sum, publication, offset);
}

} // namespace

constexpr int NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP =
    llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
constexpr int NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE = 2;
constexpr int NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE = 4;

/**
 * @brief Select the measured verifier-row reuse geometry for a runtime batch.
 *
 * M=2 retains the two-row specialization because a four-row accumulator adds
 * idle predicates without any extra weight reuse.  M>=3 uses four rows: this
 * matches the common four-row transaction and amortizes each packed-weight
 * decode across its live rows. The launch family owns how these fixed tiles
 * are scheduled; counted fused bundles use a bounded grid with live iteration.
 * This arithmetic reuse geometry is total for every positive runtime M.
 *
 * @param m Number of grouped verifier rows in the launch.
 * @return Compile-time kernel geometry selected by the host launch wrapper.
 */
constexpr int native_quantised_verifier_rows_per_tile(int m)
{
    return m <= NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE
        ? NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE
        : NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE;
}

/**
 * @brief Resolve the projection owning one captured column tile.
 * @param args Immutable ordered column intervals and borrowed operand planes.
 * @param global_tile Workgroup X coordinate in the concatenated bundle grid.
 * @return Uniform projection index; launch validation guarantees membership.
 */
__device__ __forceinline__ int native_quantised_batched_projection_for_tile(
    const NativeQuantisedSmallMBatchedArgs& args,
    int global_tile)
{
    int proj = 0;
    #pragma unroll
    for (int i = 1; i < NATIVE_QUANTISED_SMALL_M_MAX_BATCH; ++i)
    {
        if (i < args.num_projections && global_tile >= args.grid_offsets[i])
            proj = i;
    }
    return proj;
}


/**
 * @brief One resolved projection, borrowed by every physical decoder arm.
 *
 * Resolve the bundle before switching codebooks so the eight-projection
 * pointer tables need not stay live through each inlined decoder. The partial
 * pointer already names this CTA's physical K plane, never a live-row stride.
 */
struct NativeQuantisedSmallMProjection
{
    unsigned n; ///< Validated positive physical output width.
    unsigned column_base; ///< This lane's nonnegative output column, including tail lanes.
    const uint8_t* payload; ///< Borrowed packed weight blocks.
    const uint16_t* scales; ///< Primary scale plane.
    const uint16_t* mins; ///< Optional secondary/minimum plane.
    const uint32_t* emins; ///< Optional extended minima.
    const float* bias; ///< Optional output bias.
    float* output; ///< This physical row tile's output.
    float* partial; ///< This CTA's split-K plane, or null for direct output.
};

/** @brief Capture-time publication contract; producers never need both destinations. */
enum class NativeQuantisedVerifierPublication : uint8_t { DirectOutput, OrderedPartial };

/**
 * @brief Resolve one lane's projection before entering a format decoder.
 * @param args Captured bundle inventory; no device allocation is performed.
 * @param projection Bundle index selected uniformly by this workgroup.
 * @param physical_m Physical scratch stride, independent of live rows.
 * @return Borrowed operands with the physical split-K plane already selected.
 */
template <int TILE_N, int CPT, NativeQuantisedVerifierPublication Publication>
__device__ __forceinline__ NativeQuantisedSmallMProjection native_quantised_resolve_small_m_projection(
    const NativeQuantisedSmallMBatchedArgs& args, int projection,
    int physical_m)
{
    const auto n = static_cast<unsigned>(args.N[projection]);
    const auto local_tile = blockIdx.x - static_cast<unsigned>(args.grid_offsets[projection]);
    float* partial = nullptr;
    float* output = nullptr;
    const float* bias = nullptr;
    if constexpr (Publication == NativeQuantisedVerifierPublication::OrderedPartial)
        partial = args.partial[projection] + static_cast<size_t>(blockIdx.y) * physical_m * n;
    else
    {
        output = args.output[projection];
        bias = args.bias[projection];
    }
    return {n, local_tile * TILE_N + threadIdx.x * CPT,
            args.payload[projection], args.scales[projection],
            args.mins[projection], args.emins[projection],
            bias, output, partial};
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
    const uint16_t primary=::llaminar2::native_quantised_scale::loadStagedPrimary<static_cast<uint8_t>(FMT)>(scale_base,linear);
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
    const uint16_t primary=::llaminar2::native_quantised_scale::loadStagedPrimary<static_cast<uint8_t>(FMT)>(scale_base,linear);
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
template <NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_accumulate_row(
    float& acc,
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scale_A_blockwise,
    const int32_t* __restrict__ d_sum_A_blockwise,
    unsigned row,
    unsigned K,
    unsigned blocks_per_row,
    unsigned b,
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
    if constexpr (NATIVE_QUANTISEDTraits<FMT>::is_asymmetric)
    {
        sum_a = d_sum_A_blockwise
            ? d_sum_A_blockwise[static_cast<size_t>(row) * blocks_per_row + b]
            : native_quantised_activation_sum_a<FMT>(a4);
    }
    native_quantised_small_m_accumulate_decoded_block<FMT>(
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
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT,
          NativeQuantisedVerifierPublication Publication>
__device__ __forceinline__ void gemv_native_quantised_small_m_batched_shared_tile(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // Required [M x blocks_per_row]
    NativeQuantisedSmallMProjection projection,
    unsigned K,
    unsigned partition_blocks,
    unsigned active_m,
    unsigned row_base,
    const float* resolved_bias)
{
    static_assert(CPT == 1 || CPT == 2, "native-quantised small-M supports CPT in {1,2}");
    static_assert(
        ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE ||
            ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE,
        "native-quantised verifier row tile must be one of the measured geometries");
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "small-M native-quantised requires a supported native-quantised format");

    const unsigned N = projection.n;
    const unsigned n_base = projection.column_base;
    const unsigned kb = blockIdx.y;
    // Public launch validation admits only positive N/K and partition widths.
    // Preserve that contract through inlining so address products do not keep
    // unnecessary signed high halves live in the scalar register file.
    __builtin_assume(N > 0);
    __builtin_assume(K > 0);
    __builtin_assume(partition_blocks > 0);
    // Fused callers publish activation sums with the shared quantized input.
    // The public boundary rejects a missing plane; no decoder may reconstruct
    // another sum authority inside its row loop.
    __builtin_assume(d_sum_A_blockwise != nullptr);
    const unsigned blocks_per_row = K / 32;
    const unsigned b_per = partition_blocks;
    const unsigned b_start = kb * b_per;
    const unsigned b_end = min(b_start + b_per, blocks_per_row);

    const uint8_t* payload_base = projection.payload;
    const uint16_t* scale_base = projection.scales;
    const uint16_t* min_base = projection.mins;
    const uint32_t* emin_base = projection.emins;
    float* out_base = projection.output;
    float* partial_base = projection.partial;

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
        // Tail lanes borrow the final valid column for read-only arithmetic.
        // Mask only publication below: keeping an EXEC mask through the whole
        // row/decoder loop extends scalar lifetimes and causes register spills.
        // Valid columns retain their exact addresses and operation order.
        const unsigned n = min(n_base + c, N - 1);
        for (unsigned b = b_start; b < b_end; ++b)
        {
            int32_t packed_groups[8];
            const size_t linear =
                static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
            const uint8_t* payload = payload_base + linear * NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
            moe_decode_native_quantised_groups<FMT>(payload, packed_groups);

            #pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const unsigned row = row_base + tile_row;
                if (row >= active_m)
                    continue;
                native_quantised_small_m_accumulate_row<FMT>(
                    acc[tile_row][c], d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise,
                    row, K, blocks_per_row, b, packed_groups, payload, scale_base,
                    min_base, emin_base, linear);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const unsigned n = n_base + c;
        if (n >= N)
            continue;

        #pragma unroll
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
            const unsigned row = row_base + tile_row;
            if (row >= active_m)
                continue;
            const size_t out_idx =
                static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(n);
            if constexpr (Publication == NativeQuantisedVerifierPublication::DirectOutput)
            {
                out_base[out_idx] = acc[tile_row][c] + resolved_bias[c];
            }
            else
            {
                partial_base[out_idx] = acc[tile_row][c];
            }
        }
    }
}

/**
 * @brief Visit live row tiles with a bounded grid and unchanged tile arithmetic.
 *
 * Column/split-K workers already expose parallelism for wide projections.
 * Grid stride admits additional tiles only when the device count includes
 * them, avoiding thousands of workgroups whose sole job would be to exit.
 * Each tile resets its fixed register accumulator and publishes to the same
 * physical scratch plane as the rectangular launch; no row shares a reduction.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT,
          NativeQuantisedVerifierPublication Publication>
__device__ __forceinline__ void gemv_native_quantised_small_m_batched_shared_body(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scale_A_blockwise,
    const int32_t* __restrict__ d_sum_A_blockwise,
    NativeQuantisedSmallMProjection projection, unsigned K, unsigned partition_blocks, unsigned active_m,unsigned row_first)
{
    float resolved_bias[CPT] = {};
    if constexpr (Publication == NativeQuantisedVerifierPublication::DirectOutput)
    {
        // Bias is immutable across rows. Materialize each lane's value once,
        // then retire its pointer/presence predicate before decoding weights.
        // Tail lanes read a valid column but never publish their output.
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            if (projection.bias)
                resolved_bias[c] = projection.bias[min(projection.column_base + c, projection.n - 1)];
    }
    projection.bias = nullptr;
    // A runtime traversal must not speculate extra register-resident row tiles.
    const unsigned row_base = row_first;
    {
        gemv_native_quantised_small_m_batched_shared_tile<TILE_N, CPT, ROWS_PER_TILE, FMT, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise,
            projection, K, partition_blocks, active_m, row_base, resolved_bias);
    }
}

template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT,
          NativeQuantisedVerifierPublication Publication, bool HasWork = false>
__global__ __launch_bounds__(TILE_N / CPT) void gemv_native_quantised_small_m_batched_shared_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeQuantisedSmallMBatchedArgs args,
    int K,
    int partition_blocks)
{
    // The row descriptor is also the sole physical scratch-stride authority.
    const int M = args.rows.physicalRows();
    // Resolve admission before format dispatch. The count's pointer/bounds
    // are dead before the decoder's long scalar-register live ranges begin.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(args.work,args.rows,int(blockIdx.z),int(gridDim.z),ROWS_PER_TILE);
    int row_first,active_m;
    while(rowWork.next(row_first,active_m))
    {
    const int proj = native_quantised_batched_projection_for_tile(args, blockIdx.x);
    const auto projection = native_quantised_resolve_small_m_projection<TILE_N, CPT, Publication>(
        args, proj, M);
    gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, FMT, Publication>(
        d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
    } // Only authored request tiles enter format dispatch.
}

template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedVerifierPublication Publication, bool HasWork = false>
__global__ __launch_bounds__(TILE_N / CPT) void gemv_native_quantised_small_m_batched_shared_mixed_kernel_t(
    const int8_t* __restrict__ d_A_int8,                // [M x K]
    const float* __restrict__ d_scale_A_blockwise,      // [M x blocks_per_row]
    const int32_t* __restrict__ d_sum_A_blockwise,      // [M x blocks_per_row], nullable
    NativeQuantisedSmallMBatchedArgs args,
    int K,
    int partition_blocks)
{
    // The row descriptor is also the sole physical scratch-stride authority.
    const int M = args.rows.physicalRows();
    // Resolve admission before format dispatch. The count's pointer/bounds
    // are dead before the decoder's long scalar-register live ranges begin.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(args.work,args.rows,int(blockIdx.z),int(gridDim.z),ROWS_PER_TILE);
    int row_first,active_m;
    while(rowWork.next(row_first,active_m))
    {
    const int proj = native_quantised_batched_projection_for_tile(args, blockIdx.x);
    const auto projection = native_quantised_resolve_small_m_projection<TILE_N, CPT, Publication>(
        args, proj, M);
    const auto codebook = static_cast<NativeQuantisedFormat>(
        __builtin_amdgcn_readfirstlane(static_cast<int>(args.codebook[proj])));
    switch (codebook)
    {
    case NATIVE_QUANTISED_Q4_0:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q4_0, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ4_NL:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ4_NL, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ4_XS_SOURCE, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_MXFP4:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_MXFP4, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q4_1:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q4_1, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q5_0:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q5_0, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q5_1:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q5_1, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q6_K:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q6_K, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q3_K:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q3_K, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q2_K:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q2_K, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ3_S:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ3_S, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ3_XXS:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ3_XXS, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ2_S:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ2_S, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ2_XS:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ2_XS, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ2_XXS:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ2_XXS, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ1_S:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ1_S, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_IQ1_M:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_IQ1_M, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_Q8_0:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_Q8_0, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        gemv_native_quantised_small_m_batched_shared_body<TILE_N, CPT, ROWS_PER_TILE, NATIVE_QUANTISED_EXPANDED_INT8_MIN, Publication>(
            d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, projection, K, partition_blocks, active_m,row_first);
        break;
    default:
        break;
    }
    } // Only authored request tiles enter format dispatch.
}

/**
 * @brief Reduces one runtime row for every projection in a fused batch.
 *
 * `blockIdx.y` owns a row residue class, advancing by gridDim.y only while
 * rows are live. The fixed-width register state and ascending-KB fold preserve
 * serial-M1 arithmetic for every depth without launching idle physical rows.
 */
template <int TILE_N, int CPT, bool HasWork = false>
__global__ void gemv_native_quantised_reduce_small_m_batched_kernel_t(
    NativeQuantisedSmallMBatchedArgs args,
    int kblocks)
{
    // The row descriptor is also the sole physical scratch-stride authority.
    const int M = args.rows.physicalRows();
    const int global_tile = blockIdx.x;
    const int proj = native_quantised_batched_projection_for_tile(args, global_tile);
    const int local_tile = global_tile - args.grid_offsets[proj];
    const int N = args.N[proj];
    const int n_base = local_tile * TILE_N + threadIdx.x * CPT;

    const float* partial_base = args.partial[proj];
    const float* bias_base = args.bias[proj];
    float* out_base = args.output[proj];
    // Physical M remains the scratch stride even when fewer rows are live.
    auto rowWork=llaminar2::projectionCursorFor<HasWork>(args.work,args.rows,int(blockIdx.y),int(gridDim.y),1);
    int row,active_m;
    while(rowWork.next(row,active_m))
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n >= N)
                continue;

            float sum = 0.0f;
            // Runtime rows must not multiply sixteen simultaneous address
            // live ranges; the ascending serial-M1 fold remains unchanged.
            #pragma unroll 1
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
}

/**
 * @brief Runtime-row NativeQuantised verifier GEMV with fixed four-row work tiles.
 *
 * A workgroup decodes each packed weight block once and applies it to at most
 * four verifier rows. Counted captures use bounded Z-grid workers that visit
 * further live tiles by grid stride, rather than scheduling idle capacity.
 * Workers never enlarge the accumulator array or alter an existing row's
 * block order. Split-K uses the fixed physical workspace stride so the reducer
 * can publish partitions in serial-M1 order on every replay.
 */
template <int TILE_N, int CPT, int ROWS_PER_TILE, NativeQuantisedFormat FMT, bool HasWork = false>
__global__ void gemv_native_quantised_small_m_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [M x K] quantized activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row x N x payload_bytes]
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row x N]
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row x N], nullable when unused
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row x N], Q2_K only
    float*           __restrict__ d_C_fp32,       // [M x N] output
    float*           __restrict__ d_partial_fp32, // [KB x M x N] partials
    const float*     __restrict__ d_scale_A_blockwise, // [M x blocks_per_row]
    const int32_t*   __restrict__ d_sum_A_blockwise,   // [M x blocks_per_row], nullable
    int M, int N, int K, int kblocks, llaminar2::DeviceRowRange rows, llaminar2::ProjectionRowWork work)
{
    static_assert(CPT == 1 || CPT == 2, "native-quantised small-M supports CPT in {1,2}");
    static_assert(
        ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE ||
            ROWS_PER_TILE == NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE,
        "native-quantised verifier row tile must be one of the measured geometries");
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
                  "small-M native-quantised requires a supported native-quantised format");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int kb = blockIdx.y;
    // Geometry supplies enough column/K parallelism before admitting extra
    // row workers. The whole CTA borrows one count; padding owns no dot product.
    int active_m;
    std::uint32_t authored_rows=0;
    if constexpr(HasWork)
    {
        // The existing launcher slices the persistent workspace at this exact
        // bound; this is not a semantic M limit. Observe original publications
        // once before decoder lifetimes, then retain only derived row bits.
        static_assert(llaminar2::kDefaultNativeQuantisedVerifierRowCapacity<=32);
        __shared__ std::uint32_t published_rows;
        if(threadIdx.x==0)
        {
            if(M<=0 || M>llaminar2::kDefaultNativeQuantisedVerifierRowCapacity)
                __builtin_trap();
            std::uint32_t mask=0;
            for(int row=0;row<M;++row)
                if(work.rowIsActive(row))mask|=std::uint32_t(1)<<row;
            published_rows=mask;
        }
        // All threads consume the same derived mask before any packed weight,
        // activation or partial access. It is republished on every graph replay.
        __syncthreads();
        authored_rows=__builtin_amdgcn_readfirstlane(published_rows);
        if(!authored_rows)return;
        active_m=M;
    }
    else active_m=rows.activeRows();

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    for(int row_base=int(blockIdx.z)*ROWS_PER_TILE;row_base<active_m;
        row_base+=int(gridDim.z)*ROWS_PER_TILE)
    {
        if constexpr(HasWork)
            if(!((authored_rows>>row_base)&((std::uint32_t(1)<<ROWS_PER_TILE)-1)))continue;
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
                const uint8_t* payload = d_payload + linear * NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
                moe_decode_native_quantised_groups<FMT>(payload, packed_groups);

#pragma unroll
                for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                {
                    const int row = row_base + tile_row;
                    if (row >= active_m)
                        continue;
                    if constexpr(HasWork)
                        if(!(authored_rows&(std::uint32_t(1)<<row)))continue;
                    native_quantised_small_m_accumulate_row<FMT>(
                        acc[tile_row][c], d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise, row, K,
                        blocks_per_row, b, packed_groups, payload, d_block_scales, d_block_mins,
                        d_block_emins, linear);
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
                if (row >= active_m)
                    continue;
                if constexpr(HasWork)
                    if(!(authored_rows&(std::uint32_t(1)<<row)))continue;
                const size_t out_idx =
                    static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(n);
                if (kblocks == 1)
                {
                    d_C_fp32[out_idx] = acc[tile_row][c];
                }
                else
                {
                    const size_t partial_idx = (static_cast<size_t>(kb) * static_cast<size_t>(M) +
                                                static_cast<size_t>(row)) *
                                                   static_cast<size_t>(N) +
                                               static_cast<size_t>(n);
                    d_partial_fp32[partial_idx] = acc[tile_row][c];
                }
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
/**
 * @brief Original IQ-LDS producer with explicit logical tile and original K partition.
 * @details The original256-thread preload, decoder and arithmetic are shared
 * with resident execution. Every physical lane participates in its CTA barriers;
 * callers fence shared reuse between successive logical producer tasks.
 */
template <int IQ_TN, NativeQuantisedFormat FMT, bool HasWork = false>
__device__ __forceinline__ void nativeM1IQLDSBody(
    const int8_t*    __restrict__ d_A_int8,
    const uint8_t*   __restrict__ d_payload,
    const uint16_t*  __restrict__ d_block_scales,
    const uint16_t*  __restrict__ d_block_mins,
    const uint32_t*  __restrict__ d_block_emins,
    float*           __restrict__ d_C_fp32,
    float*           __restrict__ d_partial_fp32,
    const float*     __restrict__ d_scale_A,
    const float*     __restrict__ d_scale_A_blockwise,
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work,
    unsigned logical_tile,unsigned logical_partition)
{
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return; // Same ordinary M1 route; no unauthored reads/writes.

    using Traits = NATIVE_QUANTISEDTraits<FMT>;
    static_assert(Traits::is_iq_grid || Traits::is_iq1_grid,
                  "LDS kernel only for IQ grid formats");

    // ---- LDS grid table ----
    __shared__ uint8_t lds_raw[Traits::grid_lds_bytes];
    [[maybe_unused]] uint32_t* lds32 = reinterpret_cast<uint32_t*>(lds_raw);
    [[maybe_unused]] uint64_t* lds64 = reinterpret_cast<uint64_t*>(lds_raw);

    // Cooperative preload: 256 threads load grid from constant → LDS
    if constexpr (Traits::is_iq3_grid)
    {
        const uint32_t* src = (FMT == NATIVE_QUANTISED_IQ3_S) ? d_iq3s_grid : d_iq3xxs_grid;
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds32[i] = src[i];
    }
    else if constexpr (Traits::is_iq2_grid)
    {
        const uint64_t* src =
            (FMT == NATIVE_QUANTISED_IQ2_S)   ? d_iq2s_grid :
            (FMT == NATIVE_QUANTISED_IQ2_XS)  ? d_iq2xs_grid : d_iq2xxs_grid;
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
    const int n = logical_tile * IQ_TN + threadIdx.x;
    const bool active = (n < N);
    const int kb = logical_partition;

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
                    sum_a = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a);
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

template <int IQ_TN, NativeQuantisedFormat FMT, bool HasWork = false>
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
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work)
{
    nativeM1IQLDSBody<IQ_TN,FMT,HasWork>(d_A_int8,d_payload,d_block_scales,d_block_mins,d_block_emins,
        d_C_fp32,d_partial_fp32,d_scale_A,d_scale_A_blockwise,N,K,kblocks,work,blockIdx.x,blockIdx.y);
}


extern "C" {

/**
 * @brief Execute table-routed gate/up and optionally publish fused Q8 SwiGLU.
 *
 * Supplying both fused publication pointers selects the production decode
 * contract: the ordered gate/up reducer writes the exact Q8 rows consumed by
 * down projection and does not materialize the FP32 output rows.  Supplying
 * neither retains the independently executable two-step oracle.  A half-bound
 * publication is rejected so callers cannot silently read stale workspace.
 */
bool rocmMoE_grouped_gate_up_native_quantised_decode_table(
    const float* d_hidden,
    const DeviceNativeQuantisedMatrixDesc* d_gate_desc_table,
    const DeviceNativeQuantisedMatrixDesc* d_up_desc_table,
    const int* d_expert_ids,
    float* const* d_gate_outputs,
    float* const* d_up_outputs,
    int8_t* d_fused_swiglu_int8,
    float* d_fused_swiglu_scales,
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
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream,
    const llaminar2::NativeTileCompletionView* completion)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    const bool fused_swiglu_publication =
        d_fused_swiglu_int8 != nullptr || d_fused_swiglu_scales != nullptr;
    if (!stream || (!hidden_prequantized && !d_hidden) ||
        !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
        ((!d_gate_outputs || !d_up_outputs) && !fused_swiglu_publication) ||
        (static_cast<bool>(d_fused_swiglu_int8) !=
         static_cast<bool>(d_fused_swiglu_scales)) ||
        !d_hidden_int8 || !d_hidden_scales ||
        !d_gate_partials || !d_up_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
        (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_table] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_table] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    recordROCmMoEDescriptorLocalDecode(
        "gate_up", "static_table", codebook_id,
        policy_codebook_mask, policy_table);

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
            fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_table] hidden quantize launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    if(completion)
    {
        if(!fused_swiglu_publication||!completion->counters||
            completion->tiles!=(static_cast<unsigned>(N)+63u)/64u||completion->routes!=unsigned(num_active)||
            completion->contributors!=policy_table.maximum_k_partitions)
            throw std::logic_error("Last-producer gate/up changed its exact prepared counter/partition geometry");
        if(llaminar2::rocm::currentGateUpProducerLoads()==llaminar2::rocm::GateUpProducerLoads::PairedK32)
        {
        hipLaunchKernelGGL(moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel,
            dim3(completion->tiles,num_active,policy_table.maximum_k_partitions),dim3(64),0,hip_stream,
            d_hidden_int8,d_hidden_scales,d_gate_desc_table,d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,d_gate_partials,d_up_partials,
            num_active,N,K,num_experts,policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,int(policy_table.maximum_k_partitions),
            d_fused_swiglu_int8,d_fused_swiglu_scales,*completion);
        }
        else
        {
        hipLaunchKernelGGL(moe_grouped_native_quantised_gate_up_last_producer_kernel,
            dim3(completion->tiles,num_active,policy_table.maximum_k_partitions),dim3(64),0,hip_stream,
            d_hidden_int8,d_hidden_scales,d_gate_desc_table,d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,d_gate_partials,d_up_partials,
            num_active,N,K,num_experts,policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,int(policy_table.maximum_k_partitions),
            d_fused_swiglu_int8,d_fused_swiglu_scales,*completion);
        }
        return hipGetLastError()==hipSuccess;
    }

    constexpr int TN = 64;
    dim3 output_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid(
        (N + TN - 1) / TN,
        num_active,
        policy_table.maximum_k_partitions);
    dim3 block(TN);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel<false>),
        partial_grid, block, 0, hip_stream,
        d_hidden_int8, d_hidden_scales,
        d_gate_desc_table, d_up_desc_table,
        static_cast<const DeviceMoELayerRuntimeView*>(nullptr),
        d_expert_ids, d_gate_partials, d_up_partials,
        num_active, N, K, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_table] gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (fused_swiglu_publication)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(moe_grouped_native_quantised_gate_up_descriptor_reduce_swiglu_q8_kernel<false>),
            output_grid,block,0,hip_stream,d_gate_partials,d_up_partials,
            d_gate_desc_table,d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,
            d_fused_swiglu_int8,d_fused_swiglu_scales,num_active,N,num_experts,
            policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                moe_grouped_native_quantised_gate_up_descriptor_reduce_kernel<false>),
            output_grid, block, 0, hip_stream,
            d_gate_partials, d_up_partials,
            d_gate_desc_table, d_up_desc_table,
            static_cast<const DeviceMoELayerRuntimeView*>(nullptr),
            d_expert_ids, d_gate_outputs, d_up_outputs,
            num_active, N, num_experts,
            policy_table.k_partitions_0_7,
            policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,
            policy_table.k_partitions_24_31);
    }
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_table] "
                "descriptor-local ordered reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmMoE_grouped_gate_up_native_quantised_decode_runtime(
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
    uint32_t policy_codebook_mask,
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
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    recordROCmMoEDescriptorLocalDecode(
        "gate_up", "runtime_bank", codebook_id,
        policy_codebook_mask, policy_table);

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
            fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] hidden quantize launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    constexpr int TN = 64;
    dim3 output_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid(
        (N + TN - 1) / TN,
        num_active,
        policy_table.maximum_k_partitions);
    dim3 block(TN);
    const auto* runtime = static_cast<const DeviceMoELayerRuntimeView*>(d_runtime_layer);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel<true>),
        partial_grid, block, 0, hip_stream,
        d_hidden_int8, d_hidden_scales,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_gate_partials, d_up_partials,
        num_active, N, K, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] gate/up launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_gate_up_descriptor_reduce_kernel<true>),
        output_grid, block, 0, hip_stream,
        d_gate_partials, d_up_partials,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_gate_outputs, d_up_outputs,
        num_active, N, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_gate_up_native_quantised_decode_runtime] "
                "descriptor-local ordered reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
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

bool rocmMoE_grouped_swiglu_down_native_quantised_decode(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeQuantisedMatrixDesc* d_descs,
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
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode] invalid arguments\n");
        return false;
    }

    const int k_partitions =
        resolveNativeQuantisedSerialM1KPartitions(codebook_id, N, K);
    if (k_partitions <= 0)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
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
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
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
                        moe_grouped_native_quantised_down_route_kernel_t<          \
                            FMT_TAG, false>),                                 \
                    route_grid, block, 0, hip_stream,                         \
                    d_swiglu_int8, d_swiglu_scales, d_descs, nullptr,         \
                    d_weights, d_ordered_down_partials, num_active, N, K);    \
            }                                                                 \
            else                                                              \
            {                                                                 \
                hipLaunchKernelGGL(                                           \
                    HIP_KERNEL_NAME(                                          \
                        moe_grouped_native_quantised_down_kpart_kernel_t<          \
                            FMT_TAG, false>),                                 \
                    partial_grid, block, 0, hip_stream,                       \
                    d_swiglu_int8, d_swiglu_scales, d_descs, nullptr,         \
                    d_ordered_down_partials, num_active, N, K, k_partitions); \
            }                                                                 \
        } while (false)

    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q4_0);     break;
    case NATIVE_QUANTISED_IQ4_NL:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ4_NL);   break;
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ4_XS_SOURCE);   break;
    case NATIVE_QUANTISED_MXFP4:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_MXFP4);   break;
    case NATIVE_QUANTISED_Q4_1:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q4_1);     break;
    case NATIVE_QUANTISED_Q5_0:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q5_0);     break;
    case NATIVE_QUANTISED_Q5_1:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q5_1);     break;
    case NATIVE_QUANTISED_Q6_K:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q6_K);     break;
    case NATIVE_QUANTISED_Q3_K:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q3_K);     break;
    case NATIVE_QUANTISED_Q2_K:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q2_K);     break;
    case NATIVE_QUANTISED_IQ3_S:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ3_S);    break;
    case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ3_XXS);  break;
    case NATIVE_QUANTISED_IQ2_S:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ2_S);    break;
    case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ2_XS);   break;
    case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ2_XXS);  break;
    case NATIVE_QUANTISED_IQ1_S:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ1_S);    break;
    case NATIVE_QUANTISED_IQ1_M:    LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_IQ1_M);    break;
    case NATIVE_QUANTISED_Q8_0:     LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_Q8_0);     break;
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        LAUNCH_GROUPED_DOWN_DIRECT(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
        break;
    default:
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode] unsupported codebook_id=%u\n",
                static_cast<unsigned>(codebook_id));
        return false;
    }

    #undef LAUNCH_GROUPED_DOWN_DIRECT

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode] down launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (k_partitions > 1)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_quantised_down_kpart_reduce_kernel,
            route_grid, block, 0, hip_stream,
            d_ordered_down_partials, d_weights,
            d_ordered_down_partials,
            num_active, N, k_partitions);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
                    "K-partition reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }

    hipLaunchKernelGGL(
        moe_grouped_native_quantised_down_ordered_reduce_kernel,
        dim3((N + TN - 1) / TN), block, 0, hip_stream,
        d_ordered_down_partials, d_output, num_active, N);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode] "
                "route reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Execute table-routed down projection from exact Q8 SwiGLU rows.
 *
 * When @p swiglu_prequantized is true, @p d_swiglu_int8 and
 * @p d_swiglu_scales are the publication produced by the fused gate/up
 * reducer, and the FP32 gate/up pointer arrays are not read.  The false form
 * remains the split-path oracle and performs the ordinary SwiGLU/Q8 kernel
 * before down projection.
 */
bool rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
    const float* const* d_gate_ptrs,
    const float* const* d_up_ptrs,
    const DeviceNativeQuantisedMatrixDesc* d_desc_table,
    const int* d_expert_ids,
    const float* d_weights,
    int8_t* d_swiglu_int8,
    float* d_swiglu_scales,
    bool swiglu_prequantized,
    float* d_output,
    float* d_canonical_route_contributions,
    float* d_ordered_down_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint8_t codebook_id,
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream,
    const llaminar2::NativeDownCompletionView* completion)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    const bool canonical_publication =
        d_canonical_route_contributions != nullptr;
    if (!stream ||
        ((!d_gate_ptrs || !d_up_ptrs) && !swiglu_prequantized) ||
        !d_desc_table ||
        !d_expert_ids || !d_weights || !d_swiglu_int8 || !d_swiglu_scales ||
        (completion ? (!d_output || !canonical_publication) : (static_cast<bool>(d_output) == canonical_publication)) ||
        !d_ordered_down_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
        (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    recordROCmMoEDescriptorLocalDecode(
        "down", "static_table", codebook_id,
        policy_codebook_mask, policy_table);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!swiglu_prequantized &&
        !rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, d_expert_ids,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    if(completion)
    {
        const auto expected=llaminar2::NativeDownCompletionPlan::make(N,num_active,policy_table.maximum_k_partitions);
        if(!completion->partitions.counters||!completion->routes||
            completion->partitions.tiles!=expected.partitions().tiles()||
            completion->partitions.routes!=expected.partitions().routes()||
            completion->partitions.contributors!=expected.partitions().contributors()||
            completion->routes!=completion->partitions.counters+expected.partitions().counters())
            return false;
        hipLaunchKernelGGL(moe_grouped_native_quantised_down_last_producer_kernel,
            dim3(expected.partitions().tiles(),num_active,policy_table.maximum_k_partitions),
            dim3(llaminar2::NativeDownCompletionPlan::columns_per_tile),0,hip_stream,
            d_swiglu_int8,d_swiglu_scales,d_desc_table,d_expert_ids,d_weights,d_ordered_down_partials,
            d_canonical_route_contributions,d_output,num_active,N,K,num_experts,
            policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
            policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,
            static_cast<int>(policy_table.maximum_k_partitions),*completion);
        return hipGetLastError()==hipSuccess;
    }

    constexpr int TN = 64;
    float* const route_contributions = canonical_publication
                                           ? d_canonical_route_contributions
                                           : d_ordered_down_partials;
    dim3 block(TN);
    const dim3 partial_grid((N+TN-1)/TN,num_active,policy_table.maximum_k_partitions);
    const dim3 route_grid((N+TN-1)/TN,num_active);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(moe_grouped_native_quantised_down_descriptor_kpart_kernel<false>),
        partial_grid,block,0,hip_stream,d_swiglu_int8,d_swiglu_scales,d_desc_table,
        static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,
        d_ordered_down_partials,num_active,N,K,num_experts,
        policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));
    if (hipGetLastError() != hipSuccess)
        return false;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(moe_grouped_native_quantised_down_descriptor_reduce_kernel<false>),
        route_grid,block,0,hip_stream,d_ordered_down_partials,d_desc_table,
        static_cast<const DeviceMoELayerRuntimeView*>(nullptr),d_expert_ids,d_weights,
        route_contributions,num_active,N,num_experts,
        policy_table.k_partitions_0_7,policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,policy_table.k_partitions_24_31);
    if (hipGetLastError() != hipSuccess)
        return false;

    if (!canonical_publication)
    {
        hipLaunchKernelGGL(
            moe_grouped_native_quantised_down_ordered_reduce_kernel,
            dim3((N + TN - 1) / TN), block, 0,
            hip_stream,
            route_contributions, d_output, num_active, N);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_table] ordered reduce launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime(
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
    uint32_t policy_codebook_mask,
    int device_id,
    void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (!stream || !d_gate_ptrs || !d_up_ptrs || !d_runtime_layer ||
        !d_expert_ids || !d_weights || !d_swiglu_int8 || !d_swiglu_scales ||
        !d_output || !d_ordered_down_partials ||
        num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] invalid arguments\n");
        return false;
    }

    ROCmMoESerialM1PolicyTable policy_table{};
    if (!buildROCmMoESerialM1PolicyTable(
            N, K, policy_codebook_mask, policy_table))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] "
                "missing descriptor-local serial-M1 policy "
                "(codebook=%u policy_mask=0x%08x N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id),
                static_cast<unsigned>(policy_codebook_mask), N, K);
        return false;
    }
    recordROCmMoEDescriptorLocalDecode(
        "down", "runtime_bank", codebook_id,
        policy_codebook_mask, policy_table);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!rocmMoE_grouped_swiglu_quantize_decode_equivalent(
            d_gate_ptrs, d_up_ptrs, d_expert_ids,
            d_swiglu_int8, d_swiglu_scales,
            num_active, K, device_id, stream))
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] "
                "grouped Q8 publication failed\n");
        return false;
    }
    hipError_t err = hipSuccess;

    constexpr int TN = 64;
    dim3 route_grid((N + TN - 1) / TN, num_active);
    dim3 partial_grid(
        (N + TN - 1) / TN,
        num_active,
        policy_table.maximum_k_partitions);
    dim3 block(TN);
    const auto* runtime = static_cast<const DeviceMoELayerRuntimeView*>(d_runtime_layer);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_down_descriptor_kpart_kernel<true>),
        partial_grid, block, 0, hip_stream,
        d_swiglu_int8, d_swiglu_scales,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_ordered_down_partials,
        num_active, N, K, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31,
        static_cast<int>(policy_table.maximum_k_partitions));

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] route launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            moe_grouped_native_quantised_down_descriptor_reduce_kernel<true>),
        route_grid, block, 0, hip_stream,
        d_ordered_down_partials,
        static_cast<const DeviceNativeQuantisedMatrixDesc*>(nullptr),
        runtime, d_expert_ids, d_weights, d_ordered_down_partials,
        num_active, N, num_experts,
        policy_table.k_partitions_0_7,
        policy_table.k_partitions_8_15,
        policy_table.k_partitions_16_23,
        policy_table.k_partitions_24_31);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] "
                "descriptor-local K-partition reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }

    hipLaunchKernelGGL(
        moe_grouped_native_quantised_down_ordered_reduce_kernel,
        dim3((N + TN - 1) / TN), block, 0, hip_stream,
        d_ordered_down_partials, d_output, num_active, N);
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime] ordered reduce launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"

// ============================================================================
// NativeQuantised GEMV dispatch: lossless decode with FP16 per-block scales
//
// 2-tier dispatch (matching INT8 scatter pattern):
//   KB=1  → Direct:         single kernel writes final output, no partials
//   KB>1  → Scatter+Reduce: scatter writes partials, reduce sums to output
//
// Eliminates atomicAdd and output memset from the old implementation.
// ============================================================================
/**
 * @brief Record physical grouped tiles with device-owned arithmetic admission.
 *
 * The published count never changes the generated serial-M1 K partitions,
 * scratch size, or captured grid. Each fixed scratch tile receives a slice of
 * the same count owner; its producer and reducer agree on both logical row
 * admission and the unchanged physical partial-buffer stride.
 */
template <NativeQuantisedFormat FMT>
bool rocmLaunch_native_quantised_small_m_fp32(
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
    int device_id, void* stream,
    uint8_t arithmetic_policy_codebook_id,
    const llaminar2::DeviceRowRange *row_range,
    const llaminar2::GPUHCDownOutputView* publication = nullptr)
{
    const auto rows = row_range
        ? *row_range : llaminar2::DeviceRowRange::fullyActive(M);
    if (!stream || rows.physicalRows() != M)
        return false;

    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    const int blocks_per_row = K / 32;
    const int grid_n = (N + TN - 1) / TN;
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override = g_native_quantised_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_quantised_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override = (api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb;
    const int explicit_waves_override = (api_waves_override > 0) ? api_waves_override : rocm_env.native_quantised_gemv_target_waves;
    const bool decode_equivalent = nativeQuantisedDecodeEquivalentSmallMMode();
    if (!validateNativeQuantisedDecodeEquivalentKBOverride(
            "rocmGemv_native_quantised_small_m_fp32",
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
            ? resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
                  arithmetic_policy_codebook_id, N, K)
            : resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
                  arithmetic_policy_codebook_id, M, N, K);
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_fp32] no serial M=1 split-K policy for decode-equivalent verifier launch (M=%d N=%d K=%d codebook=%u)\n",
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
                "[rocmGemv_native_quantised_small_m_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d N=%d K=%d)\n",
                kb_override, KB_CAP, M, N, K);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_quantised_verifier_rows_per_tile(M);
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

    // A standalone Z or materialized head still belongs to its original stage
    // operation. Preserve the actual output, source and Q8 publication before
    // tiling so a scalar optimization retains the same full operand matching.
    NativeQuantisedSmallMBatchedArgs original_packet{.rows=rows,
        .work=llaminar2::projectionWorkForLaunch(M,row_range)};
    original_packet.num_projections=1; original_packet.total_grid_n=grid_n;
    original_packet.N[0]=N; original_packet.grid_offsets[1]=grid_n;
    original_packet.payload[0]=d_payload; original_packet.scales[0]=d_block_scales;
    original_packet.mins[0]=d_block_mins; original_packet.emins[0]=d_block_emins;
    original_packet.output[0]=d_C_fp32; original_packet.partial[0]=d_partial_fp32;
    original_packet.codebook[0]=static_cast<uint8_t>(FMT);
    // Only the complete materialized head may authorize a new singleton call;
    // existing QKV/GU subgroups retain their separately qualified topology.
    const auto verifier_candidate = llaminar2::rocm::nativeQuantisedVerifierProducer(
        device_id, decode_equivalent, {rows, d_payload, N, K, 1, static_cast<uint8_t>(FMT),
        generated_cfg.has_generated && explicit_kb_override <= 0 && explicit_waves_override <= 0 &&
        kb == generated_cfg.kb && target_waves_per_cu == generated_cfg.target_waves_per_cu &&
        (g_native_quantised_serial_partition_n <= 0 || g_native_quantised_serial_partition_n == N)
            ? llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::CanonicalSerial
            : llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::Unmeasured,
        (llaminar2::rocm::hasNativeQuantisedGDNVerifier() ||
            llaminar2::rocm::hasNativeQuantisedMaterializedHead())?&original_packet:nullptr,
        d_A_int8,d_scale_A_blockwise,d_sum_A_blockwise});
    const bool wmma = verifier_candidate != llaminar2::rocm::NativeQuantisedVerifierCandidate::Dot4;

    if (kb > 1 && !d_partial_fp32)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] kb=%d>1 but no small-M partial buffer (M=%d N=%d K=%d)\n",
                kb, M, N, K);
        return false;
    }

    bool hc_down_long_k = false;
    if constexpr (FMT == NATIVE_QUANTISED_Q8_0)
    {
        if (decode_equivalent && generated_cfg.has_generated &&
            generated_cfg.kb == kHCDownPartitions && kb == kHCDownPartitions &&
            arithmetic_policy_codebook_id == static_cast<uint8_t>(NATIVE_QUANTISED_Q8_0) &&
            N == kHCDownColumns && K == kHCDownReduction && M >= 2 && M <= 8 &&
            row_range && row_range->countOwner() && !d_block_mins && !d_block_emins &&
            !llaminar2::currentProjectionRowWork(M) &&
            (publication || llaminar2::qfnGroupedHCDownLongKExperiment()))
        {
            const auto &hardware = llaminar2::rocmHardwareMetadata().at(device_id);
            hc_down_long_k = (hardware.isa_target == "gfx906" || hardware.isa_target.starts_with("gfx906:")) &&
                hardware.compute_units == 60 && hardware.native_wave_width == 64 &&
                hardware.maximum_threads_per_block >= kHCDownTileColumns * kHCDownWindow;
            if (hc_down_long_k &&
                (reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
                 (reinterpret_cast<std::uintptr_t>(d_payload) & 15u) ||
                 (reinterpret_cast<std::uintptr_t>(d_A_int8) & 3u)))
            {
                fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] invalid exact HC-down alignment or stream\n");
                return false;
            }
        }
    }

    if (publication && (!hc_down_long_k || M != kHCDownTileRows ||
        publication->columns != kHCDownColumns ||
        publication->rows.captureIdentity() != rows.captureIdentity() ||
        !publication->activated || !publication->primary || !publication->residual ||
        !publication->primaryScales || !publication->residualScales))
        return false;

    recordNativeQuantisedSmallMLaunch(
        M,
        N,
        K,
        static_cast<unsigned>(FMT),
        hc_down_long_k ? N / kHCDownTileColumns : grid_n,
        kb,
        target_waves_per_cu,
        false,
        1,
        static_cast<int>(arithmetic_policy_codebook_id));

    /*
     * Partials are a reusable fixed-row tile, not a semantic M limit. Wider
     * verifier graphs record multiple grouped kernel/reducer pairs while each
     * pair retains weight reuse and scalar M=1's ordered split-K reduction.
     */
    const int workspace_tile_rows = min(
        M,
        llaminar2::kDefaultNativeQuantisedVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const auto tile_rows = rows.slice(row_base, tile_m);
        const int tile_rows_per_group = wmma ? 16 : native_quantised_verifier_rows_per_tile(tile_m);
        const int tile_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_rows, tile_rows_per_group,
            static_cast<int64_t>(grid_n) * kb, target_total_waves);
        const int reducer_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_rows, 1, grid_n, target_total_waves);
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

        if (hc_down_long_k)
        {
            static_assert(llaminar2::kDefaultNativeQuantisedVerifierRowCapacity >= 8);
            const dim3 grid(N / kHCDownTileColumns, kb, (tile_m + kHCDownTileRows - 1) / kHCDownTileRows);
            constexpr int threads = kHCDownTileColumns * kHCDownWindow;
            hipLaunchKernelGGL(qfn_hc_down_long_k_contributions, grid, dim3(threads), 0,
                static_cast<hipStream_t>(stream),
                (HCDownLongKArgs{tile_a, d_payload, d_block_scales, tile_scales, d_partial_fp32, tile_rows}));
            llaminar2::PerfStatsCollector::addCounter("gemm", "qfn_grouped_hc_down_long_k_nodes", 1.0,
                "gemm", "rocm:" + std::to_string(device_id));
        }
        else if (wmma)
        {
            NativeQuantisedSmallMBatchedArgs bundle{.rows=tile_rows,
                .work=llaminar2::projectionWorkForLaunch(M,row_range).slice(row_base,tile_m)};
            bundle.num_projections=1; bundle.total_grid_n=grid_n;
            bundle.N[0]=N; bundle.grid_offsets[1]=grid_n;
            bundle.payload[0]=d_payload; bundle.scales[0]=d_block_scales;
            bundle.mins[0]=d_block_mins; bundle.emins[0]=d_block_emins;
            bundle.output[0]=tile_c; bundle.partial[0]=d_partial_fp32;
            bundle.codebook[0]=static_cast<uint8_t>(FMT);
            llaminar2::rocm::submitNativeQuantisedVerifierWmma({
                .activations=tile_a,.activation_scales=tile_scales,.activation_sums=tile_sums,.projections=bundle,
                .device=device_id,.k=K,.partitions=kb,.row_workers=tile_row_groups,
                .has_request_work=llaminar2::currentProjectionRowWork(M)!=nullptr,
                .publication=kb==1 ? llaminar2::rocm::VerifierWmmaPublication::Standalone :
                    llaminar2::rocm::VerifierWmmaPublication::OrderedPartial},stream);
        }
        else if (tile_rows_per_group == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE)
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_small_m_kernel_t<
                    TN, CPT, NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE, FMT, HasWork>),
                dim3(grid_n, kb, tile_row_groups), dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, d_payload, d_block_scales, d_block_mins, d_block_emins,
                tile_c, d_partial_fp32, tile_scales, tile_sums,
                tile_m, N, K, kb, tile_rows,llaminar2::projectionWorkForLaunch(M,row_range).slice(row_base,tile_m));
        });
        }
        else
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_small_m_kernel_t<
                    TN, CPT, NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE, FMT, HasWork>),
                dim3(grid_n, kb, tile_row_groups), dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                tile_a, d_payload, d_block_scales, d_block_mins, d_block_emins,
                tile_c, d_partial_fp32, tile_scales, tile_sums,
                tile_m, N, K, kb, tile_rows,llaminar2::projectionWorkForLaunch(M,row_range).slice(row_base,tile_m));
        });
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d N=%d K=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, N, K,
                    static_cast<unsigned>(FMT));
            return false;
        }

        if (publication)
        {
            hipLaunchKernelGGL(groupedHCDownReducePublish, dim3(grid_n, tile_m), dim3(THREADS), 0,
                static_cast<hipStream_t>(stream), d_partial_fp32, tile_c, *publication);
        }
        else if (kb > 1)
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_reduce_small_m_kernel_t<TN, CPT, HasWork>),
                dim3(grid_n, reducer_row_groups), dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream),
                d_partial_fp32, tile_c, tile_m, N, kb, tile_rows,llaminar2::projectionWorkForLaunch(M,row_range).slice(row_base,tile_m));
        });
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d N=%d K=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, N, K,
                    static_cast<unsigned>(FMT));
            return false;
        }
    }

    return true;
}

template <NativeQuantisedFormat FMT>
bool rocmLaunch_native_quantised_small_m_batched_fp32(
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
    int call_waves_override = -1,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    (void)device_id;

    const int blocks_per_row = K / 32;
    const auto rows = row_range ? *row_range :
        llaminar2::DeviceRowRange::fullyActive(M);
    if (!stream || rows.physicalRows() != M)
        return false;
    NativeQuantisedSmallMBatchedArgs args{.rows = rows,.work=llaminar2::projectionWorkForLaunch(M,row_range)};
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
    const int api_kb_override = g_native_quantised_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_quantised_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (call_kb_override > 0)
            ? call_kb_override
            : ((api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb);
    const int explicit_waves_override =
        (call_waves_override > 0)
            ? call_waves_override
            : ((api_waves_override > 0) ? api_waves_override : rocm_env.native_quantised_gemv_target_waves);
    const bool decode_equivalent = nativeQuantisedDecodeEquivalentSmallMMode();
    if (!validateNativeQuantisedDecodeEquivalentKBOverride(
            "rocmGemv_native_quantised_small_m_batched_fp32",
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
            ? resolveNativeQuantisedCommonSerialM1RuntimeConfig(
                  static_cast<uint8_t>(FMT), Ns, num_projections, K)
            : (num_projections == 1
                   ? resolveNativeQuantisedDecodeGeneratedRuntimeConfig(
                         static_cast<uint8_t>(FMT), M, Ns[0], K)
                   : NativeQuantisedDecodeGeneratedRuntimeConfig{});
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_fp32] fused verifier group has no common serial M=1 split-K policy (M=%d K=%d projections=%d codebook=%u)\n",
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
                "[rocmGemv_native_quantised_small_m_batched_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d K=%d projections=%d)\n",
                kb_override, KB_CAP, M, K, num_projections);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_quantised_verifier_rows_per_tile(M);
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

    bool canonical_members = generated_cfg.has_generated && api_kb_override <= 0 && api_waves_override <= 0 &&
        rocm_env.native_quantised_gemv_kb <= 0 && rocm_env.native_quantised_gemv_target_waves <= 0;
    canonical_members = canonical_members && llaminar2::rocm::matchesNativeQuantisedVerifierSerialPolicies(
        num_projections, {kb,target_waves_per_cu}, [&](int member)
            -> std::optional<llaminar2::rocm::NativeQuantisedVerifierSerialPolicy> {
            const auto policy = resolveNativeQuantisedSerialM1RuntimeConfig(static_cast<uint8_t>(FMT), Ns[member], K);
            if(!policy.has_generated || (g_native_quantised_serial_partition_n > 0 &&
                g_native_quantised_serial_partition_n != Ns[member]))return std::nullopt;
            return llaminar2::rocm::NativeQuantisedVerifierSerialPolicy{policy.kb,policy.target_waves_per_cu};
        });
    const auto verifier_candidate = llaminar2::rocm::nativeQuantisedVerifierProducer(
        device_id, decode_equivalent, {rows, args.payload[0], Ns[0], K, num_projections, static_cast<uint8_t>(FMT),
        canonical_members ? llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::CanonicalSerial
            : llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::Unmeasured,
        &args, d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise});
    const bool wmma = verifier_candidate != llaminar2::rocm::NativeQuantisedVerifierCandidate::Dot4;

    if (kb > 1)
    {
        for (int p = 0; p < num_projections; ++p)
        {
            if (!args.partial[p])
            {
                fprintf(stderr,
                        "[rocmGemv_native_quantised_small_m_batched_fp32] kb=%d>1 but projection %d has no partial buffer (M=%d N=%d K=%d)\n",
                        kb, p, M, args.N[p], K);
                return false;
            }
        }
    }

    recordNativeQuantisedSmallMLaunch(
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
        llaminar2::kDefaultNativeQuantisedVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = wmma ? 16 : native_quantised_verifier_rows_per_tile(tile_m);
        NativeQuantisedSmallMBatchedArgs tile_args = args;
        tile_args.rows = rows.slice(row_base, tile_m);
        tile_args.work = args.work.slice(row_base,tile_m);
        const int tile_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, tile_rows_per_group,
            static_cast<int64_t>(args.total_grid_n) * kb, target_total_waves);
        const int reducer_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, 1, args.total_grid_n, target_total_waves);
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

        // Publication is fixed by the serial-M1 K-partition policy. Compile
        // only its live destination/epilogue into this captured producer.
        const auto launch = [&]<NativeQuantisedVerifierPublication Publication>()
        {
            if (wmma)
            {
                llaminar2::rocm::submitNativeQuantisedVerifierWmma({
                    .activations=tile_a,.activation_scales=tile_scales,.activation_sums=tile_sums,.projections=tile_args,
                    .device=device_id,.k=K,.partitions=kb,.row_workers=tile_row_groups,
                    .has_request_work=llaminar2::currentProjectionRowWork(M)!=nullptr,
                    .publication=Publication==NativeQuantisedVerifierPublication::DirectOutput
                        ? llaminar2::rocm::VerifierWmmaPublication::Batched
                        : llaminar2::rocm::VerifierWmmaPublication::OrderedPartial},stream);
            }
            else if (tile_rows_per_group == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE)
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE, FMT, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
            else
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE, FMT, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
        };
        if (kb == 1)
            launch.template operator()<NativeQuantisedVerifierPublication::DirectOutput>();
        else
            launch.template operator()<NativeQuantisedVerifierPublication::OrderedPartial>();

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections, static_cast<unsigned>(FMT));
            return false;
        }

        if (kb > 1)
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_reduce_small_m_batched_kernel_t<TN, CPT, HasWork>),
                dim3(tile_args.total_grid_n, reducer_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream), tile_args, kb);
        });
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d codebook=%u)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections, static_cast<unsigned>(FMT));
            return false;
        }
    }

    return true;
}

static bool nativeQuantisedCodebookRequiresAux(uint8_t codebook_id)
{
    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_IQ3_S:
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
    case NATIVE_QUANTISED_Q4_1:
    case NATIVE_QUANTISED_Q5_1:
    case NATIVE_QUANTISED_Q6_K:
    case NATIVE_QUANTISED_Q3_K:
    case NATIVE_QUANTISED_Q2_K:
    case NATIVE_QUANTISED_IQ2_S:
    case NATIVE_QUANTISED_IQ2_XS:
    case NATIVE_QUANTISED_IQ1_S:
    case NATIVE_QUANTISED_IQ1_M:
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        return true;
    default:
        return false;
    }
}

static bool nativeQuantisedCodebookRequiresEmins(uint8_t codebook_id)
{
    return static_cast<NativeQuantisedFormat>(codebook_id) == NATIVE_QUANTISED_Q2_K;
}

static bool nativeQuantisedCodebookSupported(uint8_t codebook_id)
{
    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:
    case NATIVE_QUANTISED_IQ4_NL:
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
    case NATIVE_QUANTISED_MXFP4:
    case NATIVE_QUANTISED_Q4_1:
    case NATIVE_QUANTISED_Q5_0:
    case NATIVE_QUANTISED_Q5_1:
    case NATIVE_QUANTISED_Q6_K:
    case NATIVE_QUANTISED_Q3_K:
    case NATIVE_QUANTISED_Q2_K:
    case NATIVE_QUANTISED_IQ3_S:
    case NATIVE_QUANTISED_IQ3_XXS:
    case NATIVE_QUANTISED_IQ2_S:
    case NATIVE_QUANTISED_IQ2_XS:
    case NATIVE_QUANTISED_IQ2_XXS:
    case NATIVE_QUANTISED_IQ1_S:
    case NATIVE_QUANTISED_IQ1_M:
    case NATIVE_QUANTISED_Q8_0:
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        return true;
    default:
        return false;
    }
}

bool rocmLaunch_native_quantised_small_m_batched_mixed_fp32(
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
    int call_waves_override = -1,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP = NATIVE_QUANTISED_SMALL_M_GRAPH_SAFE_KB_CAP;
    constexpr int TN = 64;
    constexpr int CPT = 1;
    constexpr int THREADS = TN / CPT;

    (void)device_id;

    const int blocks_per_row = K / 32;
    const auto rows = row_range ? *row_range :
        llaminar2::DeviceRowRange::fullyActive(M);
    if (!stream || rows.physicalRows() != M)
        return false;
    NativeQuantisedSmallMBatchedArgs args{.rows = rows,.work=llaminar2::projectionWorkForLaunch(M,row_range)};
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
    const int api_kb_override = g_native_quantised_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override = g_native_quantised_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (call_kb_override > 0)
            ? call_kb_override
            : ((api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb);
    const int explicit_waves_override =
        (call_waves_override > 0)
            ? call_waves_override
            : ((api_waves_override > 0) ? api_waves_override : rocm_env.native_quantised_gemv_target_waves);
    const bool decode_equivalent = nativeQuantisedDecodeEquivalentSmallMMode();
    if (!validateNativeQuantisedDecodeEquivalentKBOverride(
            "rocmGemv_native_quantised_small_m_batched_mixed_fp32",
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
            ? resolveNativeQuantisedCommonSerialM1RuntimeConfig(
                  codebook_ids, Ns, num_projections, K)
            : (num_projections == 1
                   ? resolveNativeQuantisedDecodeGeneratedRuntimeConfig(
                         codebook_ids[0], M, Ns[0], K)
                   : NativeQuantisedDecodeGeneratedRuntimeConfig{});
    if (decode_equivalent &&
        explicit_kb_override <= 0 &&
        !generated_cfg.has_generated)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] fused verifier group has no common serial M=1 split-K policy (M=%d K=%d projections=%d)\n",
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
                "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] requested kb=%d exceeds graph-safe small-M cap=%d (M=%d K=%d projections=%d)\n",
                kb_override, KB_CAP, M, K, num_projections);
        return false;
    }

    constexpr int DEFAULT_TARGET_WAVES_PER_CU = 12;
    const int target_waves_per_cu = (waves_override > 0) ? waves_override : DEFAULT_TARGET_WAVES_PER_CU;
    const int target_total_waves = target_waves_per_cu * NUM_CUS;
    const int verifier_rows_per_tile = native_quantised_verifier_rows_per_tile(M);
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

    // Equal K partitions alone do not authenticate a complete subgroup's
    // performance policy: every member must also own the actual wave target.
    bool canonical_members = (llaminar2::rocm::ScopedNativeQuantisedVerifierBundle::current() ||
        llaminar2::rocm::hasNativeQuantisedGDNVerifier()) &&
        generated_cfg.has_generated && api_kb_override <= 0 && api_waves_override <= 0 &&
        rocm_env.native_quantised_gemv_kb <= 0 && rocm_env.native_quantised_gemv_target_waves <= 0;
    canonical_members = canonical_members && llaminar2::rocm::matchesNativeQuantisedVerifierSerialPolicies(
        num_projections, {kb,target_waves_per_cu}, [&](int member)
            -> std::optional<llaminar2::rocm::NativeQuantisedVerifierSerialPolicy> {
            const auto policy = resolveNativeQuantisedSerialM1RuntimeConfig(codebook_ids[member], Ns[member], K);
            if(!policy.has_generated || (g_native_quantised_serial_partition_n > 0 &&
                g_native_quantised_serial_partition_n != Ns[member]))return std::nullopt;
            return llaminar2::rocm::NativeQuantisedVerifierSerialPolicy{policy.kb,policy.target_waves_per_cu};
        });
    const auto verifier_candidate = llaminar2::rocm::nativeQuantisedVerifierProducer(
        device_id, decode_equivalent, {rows, args.payload[0], Ns[0], K, num_projections, args.codebook[0],
        canonical_members ? llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::CanonicalSerial
            : llaminar2::rocm::NativeQuantisedVerifierPolicyMatch::Unmeasured,
        &args, d_A_int8, d_scale_A_blockwise, d_sum_A_blockwise});
    const bool wmma = verifier_candidate != llaminar2::rocm::NativeQuantisedVerifierCandidate::Dot4;

    if (kb > 1)
    {
        for (int p = 0; p < num_projections; ++p)
        {
            if (!args.partial[p])
            {
                fprintf(stderr,
                        "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] kb=%d>1 but projection %d has no partial buffer (M=%d N=%d K=%d codebook=%u)\n",
                        kb, p, M, args.N[p], K, static_cast<unsigned>(args.codebook[p]));
                return false;
            }
        }
    }

    recordNativeQuantisedSmallMLaunch(
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
        llaminar2::kDefaultNativeQuantisedVerifierRowCapacity);
    for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
    {
        const int tile_m = min(workspace_tile_rows, M - row_base);
        const int tile_rows_per_group = wmma ? 16 : native_quantised_verifier_rows_per_tile(tile_m);
        NativeQuantisedSmallMBatchedArgs tile_args = args;
        tile_args.rows = rows.slice(row_base, tile_m);
        tile_args.work = args.work.slice(row_base,tile_m);
        const int tile_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, tile_rows_per_group,
            static_cast<int64_t>(args.total_grid_n) * kb, target_total_waves);
        const int reducer_row_groups = llaminar2::deviceRowWorkerGroups(
            tile_args.rows, 1, args.total_grid_n, target_total_waves);
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

        // Publication is fixed by the serial-M1 K-partition policy. Compile
        // only its live destination/epilogue into this captured producer.
        const auto launch = [&]<NativeQuantisedVerifierPublication Publication>()
        {
            if (wmma)
            {
                llaminar2::rocm::submitNativeQuantisedVerifierWmma({
                    .activations=tile_a,.activation_scales=tile_scales,.activation_sums=tile_sums,.projections=tile_args,
                    .device=device_id,.k=K,.partitions=kb,.row_workers=tile_row_groups,
                    .has_request_work=llaminar2::currentProjectionRowWork(M)!=nullptr,
                    .publication=Publication==NativeQuantisedVerifierPublication::DirectOutput
                        ? llaminar2::rocm::VerifierWmmaPublication::Batched
                        : llaminar2::rocm::VerifierWmmaPublication::OrderedPartial},stream);
            }
            else if (tile_rows_per_group == NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE)
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_mixed_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_SHORT_ROWS_PER_TILE, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
            else
            {
                llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_native_quantised_small_m_batched_shared_mixed_kernel_t<
                        TN, CPT, NATIVE_QUANTISED_VERIFIER_WIDE_ROWS_PER_TILE, Publication, HasWork>),
                    dim3(tile_args.total_grid_n, kb, tile_row_groups),
                    dim3(THREADS, 1, 1), 0,
                    static_cast<hipStream_t>(stream),
                    tile_a, tile_scales, tile_sums, tile_args, K,
                    (blocks_per_row + kb - 1) / kb);
        });
            }
        };
        if (kb == 1)
            launch.template operator()<NativeQuantisedVerifierPublication::DirectOutput>();
        else
            launch.template operator()<NativeQuantisedVerifierPublication::OrderedPartial>();

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] kernel launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections);
            return false;
        }

        if (kb > 1)
        {
            llaminar2::dispatchProjectionWorkMode(M,[&]<bool HasWork>() {
        hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_native_quantised_reduce_small_m_batched_kernel_t<TN, CPT, HasWork>),
                dim3(tile_args.total_grid_n, reducer_row_groups),
                dim3(THREADS, 1, 1), 0,
                static_cast<hipStream_t>(stream), tile_args, kb);
        });
        }

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] reduce launch failed: %s (M=%d tile_base=%d tile_m=%d K=%d projections=%d)\n",
                    hipGetErrorString(err), M, row_base, tile_m, K,
                    num_projections);
            return false;
        }
    }

    return true;
}

namespace llaminar2::rocm {
bool enqueueGroupedHCDownReducePublish(
    const int8_t* activation, const uint8_t* weights, const uint16_t* weight_scales,
    float* raw, const float* activation_scales, const int32_t* activation_sums,
    float* partials, int device, void* stream, const GPUHCDownOutputView& publication)
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !activation || !weights ||
        !weight_scales || !raw || !activation_scales || !activation_sums || !partials)
        return false;
    HipDeviceGuard::setDevice(device);
    return rocmLaunch_native_quantised_small_m_fp32<NATIVE_QUANTISED_Q8_0>(
        activation, weights, weight_scales, nullptr, nullptr, raw, activation_scales, activation_sums,
        partials, kHCDownTileRows, kHCDownColumns, kHCDownReduction, device, stream,
        static_cast<uint8_t>(NATIVE_QUANTISED_Q8_0), &publication.rows, &publication);
}
} // namespace llaminar2::rocm

extern "C" {

/**
 * @brief Execute verifier-depth GEMV with distinct decoder and math policy.
 *
 * `codebook_id` selects the physical weight decoder. The arithmetic policy ID
 * selects the generated split-K geometry whose ordered reduction every row
 * must reproduce. Expert promotion may change only the former.
 */
bool rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
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
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream,
    const llaminar2::DeviceRowRange *row_range)
{
    if (!stream || N <= 0 || K <= 0 ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] invalid stream or physical row geometry (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (M <= 0 || (M == 1 && (!row_range || !row_range->countOwner())) ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] M requires positive rows and an explicit count owner at M1 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] K must be multiple of 32 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }
    if (!d_A_int8 || !d_payload || !d_block_scales || !d_C_fp32 || !d_scale_A_blockwise)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] missing required pointer (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16 = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32 = static_cast<const uint32_t*>(d_block_emins);

    #define LAUNCH_SMALL_M(FMT_TAG)                                                              \
        do {                                                                                      \
            using Traits = NATIVE_QUANTISEDTraits<FMT_TAG>;                                                  \
            if constexpr (Traits::requires_secondary_metadata)                         \
            {                                                                                     \
                if (!d_mins_u16)                                                                  \
                {                                                                                 \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n", \
                            static_cast<unsigned>(codebook_id), M, N, K);                         \
                    return false;                                                                 \
                }                                                                                 \
            }                                                                                     \
            if constexpr (Traits::needs_emin_array)                                               \
            {                                                                                     \
                if (!d_emins_u32)                                                                 \
                {                                                                                 \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n", \
                            static_cast<unsigned>(codebook_id), M, N, K);                         \
                    return false;                                                                 \
                }                                                                                 \
            }                                                                                     \
            return rocmLaunch_native_quantised_small_m_fp32<FMT_TAG>(                                 \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, d_C_fp32,            \
                d_scale_A_blockwise, d_sum_A_blockwise, d_partial_fp32,                           \
                M, N, K, device_id, stream, arithmetic_policy_codebook_id, row_range);                       \
        } while (0)

    switch (static_cast<NativeQuantisedFormat>(codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q4_0);
    case NATIVE_QUANTISED_IQ4_NL:   LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ4_NL);
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:   LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ4_XS_SOURCE);
    case NATIVE_QUANTISED_MXFP4:   LAUNCH_SMALL_M(NATIVE_QUANTISED_MXFP4);
    case NATIVE_QUANTISED_Q4_1:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q4_1);
    case NATIVE_QUANTISED_Q5_0:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q5_0);
    case NATIVE_QUANTISED_Q5_1:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q5_1);
    case NATIVE_QUANTISED_Q6_K:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q6_K);
    case NATIVE_QUANTISED_Q3_K:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q3_K);
    case NATIVE_QUANTISED_Q2_K:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q2_K);
    case NATIVE_QUANTISED_IQ3_S:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ3_S);
    case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ3_XXS);
    case NATIVE_QUANTISED_IQ2_S:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ2_S);
    case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ2_XS);
    case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ2_XXS);
    case NATIVE_QUANTISED_IQ1_S:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ1_S);
    case NATIVE_QUANTISED_IQ1_M:    LAUNCH_SMALL_M(NATIVE_QUANTISED_IQ1_M);
    case NATIVE_QUANTISED_Q8_0:     LAUNCH_SMALL_M(NATIVE_QUANTISED_Q8_0);
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        LAUNCH_SMALL_M(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
    default:
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_fp32] unsupported codebook_id=%u (M=%d N=%d K=%d)\n",
                static_cast<unsigned>(codebook_id), M, N, K);
        return false;
    }

    #undef LAUNCH_SMALL_M
    return false;
}

/** @brief Execute verifier-depth GEMV for self-describing weight bytes. */
bool rocmGemv_native_quantised_small_m_fp32_with_sums(
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
    return rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_C_fp32,
        d_scale_A_blockwise,
        d_sum_A_blockwise,
        d_partial_fp32,
        M,
        N,
        K,
        codebook_id,
        defaultNativeQuantisedArithmeticPolicyCodebook(codebook_id),
        device_id,
        stream);
}

/** @brief Execute verifier-depth GEMV with explicit source math policy. */
bool rocmGemv_native_quantised_small_m_fp32_with_policy(
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
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream)
{
    return rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_C_fp32,
        d_scale_A_blockwise,
        nullptr,
        d_partial_fp32,
        M,
        N,
        K,
        codebook_id,
        arithmetic_policy_codebook_id,
        device_id,
        stream);
}

bool rocmGemv_native_quantised_small_m_fp32(
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
    return rocmGemv_native_quantised_small_m_fp32_with_sums(
        d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
        d_C_fp32, d_scale_A_blockwise, nullptr, d_partial_fp32,
        M, N, K, codebook_id, device_id, stream);
}

static bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy_impl(
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
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    hipError_t device_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_id));
    if (device_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_fp32] hipSetDevice(%d) failed: %s\n",
                device_id, hipGetErrorString(device_err));
        return false;
    }

    if (M <= 0 || (M == 1 && (!row_range || !row_range->countOwner())) ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] M requires positive rows and an explicit count owner at M1 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] K must be positive and multiple of 32 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (num_projections <= 0 || num_projections > NATIVE_QUANTISED_SMALL_M_MAX_BATCH)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] invalid projection count=%d (max=%d)\n",
                num_projections, NATIVE_QUANTISED_SMALL_M_MAX_BATCH);
        return false;
    }
    if (!d_A_int8 || !d_payloads || !d_block_scales || !d_outputs ||
        !d_scale_A_blockwise || !d_sum_A_blockwise || !d_partials || !Ns)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] missing required pointer (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    for (int p = 0; p < num_projections; ++p)
    {
        if (Ns[p] <= 0 || !d_payloads[p] || !d_block_scales[p] || !d_outputs[p] || !d_partials[p])
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] invalid projection %d (N=%d payload=%p scales=%p output=%p partial=%p)\n",
                    p, Ns[p], static_cast<const void*>(d_payloads[p]),
                    static_cast<const void*>(d_block_scales[p]),
                    static_cast<const void*>(d_outputs[p]),
                    static_cast<const void*>(d_partials[p]));
            return false;
        }
    }

#define DISPATCH_BATCHED_FMT(FMT_TAG)                                                                   \
    do {                                                                                                \
        using Traits = NATIVE_QUANTISEDTraits<FMT_TAG>;                                                            \
        if constexpr (Traits::requires_secondary_metadata) {                                 \
            if (!d_block_mins) {                                                                        \
                fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] codebook_id=%u requires mins/aux array (M=%d K=%d projections=%d)\n", \
                        codebook_id, M, K, num_projections);                                            \
                return false;                                                                           \
            }                                                                                           \
            for (int p = 0; p < num_projections; ++p) {                                                  \
                if (!d_block_mins[p]) {                                                                 \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] projection %d codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n", \
                            p, codebook_id, M, Ns[p], K);                                               \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        if constexpr (Traits::needs_emin_array) {                                                       \
            if (!d_block_emins) {                                                                       \
                fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] codebook_id=%u requires emins array (M=%d K=%d projections=%d)\n", \
                        codebook_id, M, K, num_projections);                                            \
                return false;                                                                           \
            }                                                                                           \
            for (int p = 0; p < num_projections; ++p) {                                                  \
                if (!d_block_emins[p]) {                                                                \
                    fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] projection %d codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n", \
                            p, codebook_id, M, Ns[p], K);                                               \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        return rocmLaunch_native_quantised_small_m_batched_fp32<FMT_TAG>(                                  \
            d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins, d_biases, d_outputs,    \
            d_scale_A_blockwise, d_sum_A_blockwise, d_partials, Ns, num_projections, M, K,             \
            device_id, stream, policy_kb, policy_target_waves, row_range);                                        \
    } while (0)

    switch (codebook_id)
    {
    case NATIVE_QUANTISED_Q4_0:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q4_0);
    case NATIVE_QUANTISED_IQ4_NL:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ4_NL);
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ4_XS_SOURCE);
    case NATIVE_QUANTISED_MXFP4:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_MXFP4);
    case NATIVE_QUANTISED_Q4_1:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q4_1);
    case NATIVE_QUANTISED_Q5_0:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q5_0);
    case NATIVE_QUANTISED_Q5_1:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q5_1);
    case NATIVE_QUANTISED_Q6_K:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q6_K);
    case NATIVE_QUANTISED_Q3_K:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q3_K);
    case NATIVE_QUANTISED_Q2_K:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q2_K);
    case NATIVE_QUANTISED_IQ3_S:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ3_S);
    case NATIVE_QUANTISED_IQ3_XXS:  DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ3_XXS);
    case NATIVE_QUANTISED_IQ2_S:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ2_S);
    case NATIVE_QUANTISED_IQ2_XS:   DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ2_XS);
    case NATIVE_QUANTISED_IQ2_XXS:  DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ2_XXS);
    case NATIVE_QUANTISED_IQ1_S:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ1_S);
    case NATIVE_QUANTISED_IQ1_M:    DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_IQ1_M);
    case NATIVE_QUANTISED_Q8_0:     DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_Q8_0);
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        DISPATCH_BATCHED_FMT(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
    default:
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_fp32] unsupported codebook_id=%u (M=%d K=%d projections=%d)\n",
                codebook_id, M, K, num_projections);
        return false;
    }

#undef DISPATCH_BATCHED_FMT
}

bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy(
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
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range)
{
    return rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, num_projections, M, K, codebook_id, device_id,
        stream, policy_kb, policy_target_waves, row_range);
}

bool rocmGemv_native_quantised_small_m_batched_fp32_with_sums(
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
    return rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, num_projections, M, K, codebook_id, device_id,
        stream, -1, -1);
}


static bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy_impl(
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
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range = nullptr)
{
    hipError_t device_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_id));
    if (device_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] hipSetDevice(%d) failed: %s\n",
                device_id, hipGetErrorString(device_err));
        return false;
    }

    if (M <= 0 || (M == 1 && (!row_range || !row_range->countOwner())) ||
        (row_range && row_range->physicalRows() != M))
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] M requires positive rows and an explicit count owner at M1 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (K <= 0 || (K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] K must be positive and multiple of 32 (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    if (num_projections <= 0 || num_projections > NATIVE_QUANTISED_SMALL_M_MAX_BATCH)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] invalid projection count=%d (max=%d)\n",
                num_projections, NATIVE_QUANTISED_SMALL_M_MAX_BATCH);
        return false;
    }
    if (!d_A_int8 || !d_payloads || !d_block_scales || !d_outputs ||
        !d_scale_A_blockwise || !d_sum_A_blockwise || !d_partials || !Ns || !codebook_ids)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] missing required pointer (M=%d K=%d projections=%d)\n",
                M, K, num_projections);
        return false;
    }
    for (int p = 0; p < num_projections; ++p)
    {
        const uint8_t codebook = codebook_ids[p];
        if (!nativeQuantisedCodebookSupported(codebook))
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] unsupported projection %d codebook_id=%u (M=%d K=%d projections=%d)\n",
                    p, static_cast<unsigned>(codebook), M, K, num_projections);
            return false;
        }
        if (Ns[p] <= 0 || !d_payloads[p] || !d_block_scales[p] || !d_outputs[p] || !d_partials[p])
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] invalid projection %d (N=%d payload=%p scales=%p output=%p partial=%p codebook=%u)\n",
                    p, Ns[p], static_cast<const void*>(d_payloads[p]),
                    static_cast<const void*>(d_block_scales[p]),
                    static_cast<const void*>(d_outputs[p]),
                    static_cast<const void*>(d_partials[p]),
                    static_cast<unsigned>(codebook));
            return false;
        }
        if (nativeQuantisedCodebookRequiresAux(codebook) &&
            (!d_block_mins || !d_block_mins[p]))
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] projection %d codebook_id=%u requires mins/aux buffer (M=%d N=%d K=%d)\n",
                    p, static_cast<unsigned>(codebook), M, Ns[p], K);
            return false;
        }
        if (nativeQuantisedCodebookRequiresEmins(codebook) &&
            (!d_block_emins || !d_block_emins[p]))
        {
            fprintf(stderr, "[rocmGemv_native_quantised_small_m_batched_mixed_fp32] projection %d codebook_id=%u requires emins buffer (M=%d N=%d K=%d)\n",
                    p, static_cast<unsigned>(codebook), M, Ns[p], K);
            return false;
        }
    }

    return rocmLaunch_native_quantised_small_m_batched_mixed_fp32(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, policy_kb, policy_target_waves, row_range);
}

bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy(
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
    int policy_target_waves,
    const llaminar2::DeviceRowRange *row_range)
{
    return rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, policy_kb, policy_target_waves, row_range);
}

bool rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums(
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
    return rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy_impl(
        d_A_int8, d_payloads, d_block_scales, d_block_mins, d_block_emins,
        d_biases, d_outputs, d_scale_A_blockwise, d_sum_A_blockwise,
        d_partials, Ns, codebook_ids, num_projections, M, K, device_id,
        stream, -1, -1);
}


/**
 * @brief Retain the original ordinary-M1 IQ-LDS selection exactly for every caller.
 * @param execution Actual physical decoder codebook. @param n Output columns. @param k Reduction width.
 * @return The existing native source route; this does not select a resident grid or change K partitions.
 */
static bool nativeM1UsesIQLDS(std::uint8_t execution,int n,int k)
{
    constexpr int original_cus=60,iq_tile=256;
    const int tiles=(n+iq_tile-1)/iq_tile;
    return execution>=11 && execution<=17 &&
        (tiles>=original_cus/2 || (k>=4.0f*n && tiles>=original_cus/10));
}
/**
 * @brief Execute serial NativeQuantised GEMV with an explicit arithmetic policy.
 *
 * The execution codebook owns physical decoding; the arithmetic policy
 * codebook owns generated occupancy and split-K reduction order. These IDs
 * differ after compact CPU weights are promoted into expanded codebook 23.
 */
static bool nativeM1WithReduction(
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
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream,
    const float* d_scale_A_blockwise,
    llaminar2::rocm::NativeM1ReductionTicket *deferred,
    const llaminar2::rocm::NativeQ8SerialGeometryAdmission *prepared=nullptr)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2 || device_id<0 || N<=0 || K<=0 || !d_A_int8 || !d_payload || !d_block_scales || !d_C_fp32)
        throw std::invalid_argument("Native M1 submission requires exact non-null operands, device and stream");
    if(prepared&&(g_native_q8_serial_scope||codebook_id!=NATIVE_QUANTISED_Q8_0||
        arithmetic_policy_codebook_id!=19||!d_scale_A_blockwise||g_native_quantised_serial_partition_n!=0||
        prepared->deviceOrdinal()!=device_id||prepared->geometry()!=llaminar2::rocm::NativeQ8SerialGeometry::TN128CPT1||
        prepared->producerLoads()!=llaminar2::rocm::NativeQ8SerialProducerLoads::TwoK32Lookahead))
        throw std::logic_error("Prepared HC lookahead requires its original native19/blockwise/source arithmetic purpose");
    if(deferred)llaminar2::rocm::NativeM1ReductionIssuer::requireEmpty(*deferred);
    llaminar2::rocm::clearNativeQ8SerialLaunch();
    if (g_native_q8_serial_scope)
    {
        llaminar2::rocm::validateNativeQ8SerialScopeBinding(device_id, stream);
        const auto &env = llaminar2::debugEnv().rocm;
        if (codebook_id != NATIVE_QUANTISED_Q8_0 ||
            !llaminar2::MoEProjectionNumericalContract::ownsOrderedKPartitionTree(
                arithmetic_policy_codebook_id) || N <= 0 || K <= 0 ||
            g_native_quantised_kb_override.load(std::memory_order_relaxed) > 0 ||
            g_native_quantised_target_waves_override.load(std::memory_order_relaxed) > 0 ||
            env.native_quantised_gemv_kb > 0 || env.native_quantised_gemv_target_waves > 0)
            throw std::logic_error("native-Q8 serial geometry scope requires exact device/native19/stream and unchanged production arithmetic");
    }
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_fp32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int NUM_CUS = 60;        // MI50/MI60
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP =
        llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;

    const int blocks_per_row = K / 32;
    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16   = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32  = static_cast<const uint32_t*>(d_block_emins);

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override =
        g_native_quantised_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override =
        g_native_quantised_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb;
    const int explicit_waves_override =
        (api_waves_override > 0)
            ? api_waves_override
            : rocm_env.native_quantised_gemv_target_waves;
    const bool has_explicit_tuning =
        explicit_kb_override > 0 || explicit_waves_override > 0;
    const NativeQuantisedDecodeGeneratedRuntimeConfig generated_cfg =
        resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, N, K);
    if (!has_explicit_tuning && !generated_cfg.has_generated)
    {
        fprintf(
            stderr,
            "[rocmGemv_native_quantised_fp32] generated dispatch miss "
            "(codebook=%u arithmetic_policy_codebook=%u N=%d K=%d)\n",
            static_cast<unsigned>(codebook_id),
            static_cast<unsigned>(arithmetic_policy_codebook_id),
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

    // Strategy 1: CU-saturation check — need enough N-blocks for the wider tile
    constexpr int IQ_LDS_TN = 256;
    const int grid_n_lds = (N + IQ_LDS_TN - 1) / IQ_LDS_TN;

    // Strategy 2: K-heavy aspect ratio — long K-loops benefit most from LDS grid
    // cache because they do many grid lookups per thread.  Uses two ratio-based
    // checks instead of absolute block counts:
    //   - K/N ratio >= 4: shape is K-dominated (FFN_Dn pattern)
    //   - grid_n_lds >= 6 (~10% of CUs): minimum parallelism for LDS kernel's
    //     256-thread workgroups.  Shapes with fewer N-blocks (e.g., 0.5B FFN_Dn
    //     with grid_n=4) don't have enough work for the wider tile.

    const bool use_iq_lds = nativeM1UsesIQLDS(codebook_id,N,K);

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
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] IQ: kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        /*
         * Publish the actual serial-decode route before launching it.  The
         * NativeQuantised trainer uses this record to prove that its row oracle ran
         * the same KB/occupancy policy returned by the serial-M1 resolver.  M=1
         * used to be the only NativeQuantised path without this telemetry, which
         * made a correct production launch indistinguishable from an unproven
         * test-only oracle.
         */
        recordNativeQuantisedSmallMLaunch(
            1,
            N,
            K,
            static_cast<unsigned>(codebook_id),
            grid_n,
            kb,
            max_waves_per_cu,
            false,
            1,
            static_cast<int>(arithmetic_policy_codebook_id));

        dim3 block(IQ_THREADS, 1, 1);

        #define LAUNCH_IQ_LDS(FMT_TAG) \
            do { llaminar2::dispatchProjectionWorkMode(1,[&]<bool HasWork>() { \
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_iq_lds_kernel_t<IQ_TN,FMT_TAG,HasWork>), \
                    dim3(grid_n,kb),block,0,static_cast<hipStream_t>(stream), \
                    d_A_int8,d_payload,d_scales_u16,d_mins_u16,d_emins_u32, \
                    d_C_fp32,d_partial_fp32,d_scale_A,d_scale_A_blockwise,N,K,kb,llaminar2::projectionWorkForLaunch(1)); \
                finishNativeM1Producer<IQ_TN,1,HasWork>(d_partial_fp32,d_C_fp32,N,kb,device_id,stream,deferred,llaminar2::projectionWorkForLaunch(1)); \
            }); } while(false)

        switch (static_cast<NativeQuantisedFormat>(codebook_id))
        {
        case NATIVE_QUANTISED_IQ3_S:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ3_S);    break;
        case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ3_XXS);  break;
        case NATIVE_QUANTISED_IQ2_S:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ2_S);    break;
        case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ2_XS);   break;
        case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ2_XXS);  break;
        case NATIVE_QUANTISED_IQ1_S:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ1_S);    break;
        case NATIVE_QUANTISED_IQ1_M:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ1_M);    break;
        default:
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] Unexpected IQ codebook_id=%u\n",
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

        const auto serial_geometry = prepared?prepared->geometry():g_native_q8_serial_scope
            ? g_native_q8_serial_scope->admission().geometry()
            : llaminar2::rocm::NativeQ8SerialGeometry::TN128CPT1;
        const auto serial_spec = llaminar2::rocm::nativeQ8SerialGeometrySpec(serial_geometry);
        const int physical_tile_n = codebook_id == NATIVE_QUANTISED_Q8_0 ? serial_spec.tile_columns : TN;
        const int grid_n = 1 + (N - 1) / physical_tile_n;

        constexpr int DEFAULT_TARGET_WAVES_PER_CU = 24;

        // Both expanded INT8 representations have the same payload traffic
        // and direct decode.  Codebook 23 adds only the already-vectorized
        // asymmetric correction, so it intentionally inherits codebook 19's
        // trained launch policy until its own certification corpus is added.
        constexpr int Q8_0_TARGET_WAVES_PER_CU = 4;
        const bool uses_expanded_int8 =
            codebook_id == static_cast<uint8_t>(NATIVE_QUANTISED_Q8_0) ||
            codebook_id == static_cast<uint8_t>(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
        const int default_waves = uses_expanded_int8
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

        if ((g_native_q8_serial_scope||prepared) && kb !=
            llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(K))
            throw std::logic_error("native-Q8 serial geometry cannot change the production K-partition tree");

        if (kb > 1 && !d_partial_fp32)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        /*
         * Keep serial M=1 route evidence symmetric with the grouped verifier
         * kernels.  Recording the resolved values, rather than merely the
         * requested overrides, also proves clamping and shape constraints did
         * not silently select a different production launch.
         */
        #define LAUNCH_NATIVE_QUANTISED(FMT_TAG) \
            launchNativeQuantisedSerialGeometry<TN, CPT, FMT_TAG>( \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, \
                d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, kb, \
                arithmetic_policy_codebook_id, device_id, stream, serial_geometry,deferred,prepared)

        switch (static_cast<NativeQuantisedFormat>(codebook_id))
        {
        case NATIVE_QUANTISED_Q4_0:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q4_0);     break;
        case NATIVE_QUANTISED_IQ4_NL:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ4_NL);   break;
        case NATIVE_QUANTISED_IQ4_XS_SOURCE:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ4_XS_SOURCE);   break;
        case NATIVE_QUANTISED_MXFP4:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_MXFP4);   break;
        case NATIVE_QUANTISED_Q4_1:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q4_1);     break;
        case NATIVE_QUANTISED_Q5_0:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q5_0);     break;
        case NATIVE_QUANTISED_Q5_1:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q5_1);     break;
        case NATIVE_QUANTISED_Q6_K:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q6_K);     break;
        case NATIVE_QUANTISED_Q3_K:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q3_K);     break;
        case NATIVE_QUANTISED_Q2_K:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q2_K);     break;
        // IQ formats fall through here when N ≤ IQ_LDS_N_THRESHOLD
        case NATIVE_QUANTISED_IQ3_S:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ3_S);    break;
        case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ3_XXS);  break;
        case NATIVE_QUANTISED_IQ2_S:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ2_S);    break;
        case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ2_XS);   break;
        case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ2_XXS);  break;
        case NATIVE_QUANTISED_IQ1_S:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ1_S);    break;
        case NATIVE_QUANTISED_IQ1_M:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ1_M);    break;
        case NATIVE_QUANTISED_Q8_0:
        {
            using G = llaminar2::rocm::NativeQ8SerialGeometry;
#define LAUNCH_NATIVE_Q8_GEOMETRY(TILE, COLUMNS) \
            launchNativeQuantisedSerialGeometry<TILE, COLUMNS, NATIVE_QUANTISED_Q8_0>( \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, \
                d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, kb, \
                arithmetic_policy_codebook_id, device_id, stream, serial_geometry,deferred,prepared)
            switch (serial_geometry)
            {
            case G::TN64CPT1: LAUNCH_NATIVE_Q8_GEOMETRY(64, 1); break;
            case G::TN128CPT1: LAUNCH_NATIVE_Q8_GEOMETRY(128, 1); break;
            case G::TN256CPT1: LAUNCH_NATIVE_Q8_GEOMETRY(256, 1); break;
            case G::TN128CPT2: LAUNCH_NATIVE_Q8_GEOMETRY(128, 2); break;
            case G::TN256CPT2: LAUNCH_NATIVE_Q8_GEOMETRY(256, 2); break;
            default: throw std::invalid_argument("unknown ROCm native-Q8 serial geometry");
            }
#undef LAUNCH_NATIVE_Q8_GEOMETRY
            break;
        }
        case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
            LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
            break;
        default:
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] Unknown codebook_id=%u (N=%d K=%d)\n",
                    static_cast<unsigned>(codebook_id), N, K);
            return false;
        }

        recordNativeQuantisedSmallMLaunch(
            1,
            N,
            K,
            static_cast<unsigned>(codebook_id),
            grid_n,
            kb,
            TARGET_WAVES_PER_CU,
            false,
            1,
            static_cast<int>(arithmetic_policy_codebook_id));

        #undef LAUNCH_NATIVE_QUANTISED
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        llaminar2::rocm::clearNativeQ8SerialLaunch();
        fprintf(stderr, "[rocmGemv_native_quantised_fp32] Kernel launch failed: %s (N=%d K=%d codebook=%u)\n",
                hipGetErrorString(err), N, K, static_cast<unsigned>(codebook_id));
        return false;
    }

    return true;
}

/** @brief Preserve the public immediate-reduction ABI through the shared original dispatcher. */
bool rocmGemv_native_quantised_fp32_with_policy(const int8_t *A,const uint8_t *payload,
    const void *scales,const void *mins,const void *emins,float *output,const float *scale_a,
    float *partials,int n,int k,uint8_t execution,uint8_t policy,int device,void *stream,
    const float *block_scales)
{
    return nativeM1WithReduction(A,payload,scales,mins,emins,output,scale_a,partials,
        n,k,execution,policy,device,stream,block_scales,nullptr);
}

/** @brief Execute serial GEMV for self-describing weight bytes. */
bool rocmGemv_native_quantised_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A,
    float* d_partial_fp32,
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream,
    const float* d_scale_A_blockwise)
{
    return rocmGemv_native_quantised_fp32_with_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_C_fp32,
        d_scale_A,
        d_partial_fp32,
        N,
        K,
        codebook_id,
        defaultNativeQuantisedArithmeticPolicyCodebook(codebook_id),
        device_id,
        stream,
        d_scale_A_blockwise);
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

/**
 * @brief Borrow existing backend grid addresses after their generation owner published them.
 * @param ordinal Exact device selected during source-weight preparation.
 * @param tables Destination non-owning immutable view; no cache or allocation is created.
 * @return True when all six existing source codebooks have valid device addresses.
 */
extern "C" bool rocmNativeEmbeddingIQTableView(
    int ordinal, llaminar2::NativeEmbeddingIQTables *tables)
{
    if (!tables || ordinal < 0) return false;
    *tables = {};
    if (hipSetDevice(ordinal) != hipSuccess) return false;
    void *address = nullptr;
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq3s_grid)) != hipSuccess) return false;
    tables->iq3s = static_cast<const uint32_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq3xxs_grid)) != hipSuccess) return false;
    tables->iq3xxs = static_cast<const uint32_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq2s_grid)) != hipSuccess) return false;
    tables->iq2s = static_cast<const uint64_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq2xs_grid)) != hipSuccess) return false;
    tables->iq2xs = static_cast<const uint64_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq2xxs_grid)) != hipSuccess) return false;
    tables->iq2xxs = static_cast<const uint64_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq1s_grid)) != hipSuccess) return false;
    tables->iq1s = static_cast<const uint64_t *>(address);
    return true;
}

namespace llaminar2::rocm
{
bool enqueueNativeM1Producer(NativeM1ReductionTicket &ticket,const std::int8_t *A,
    const std::uint8_t *payload,const void *scales,const void *mins,const void *emins,
    float *output,const float *scale_a,float *partials,int n,int k,std::uint8_t execution,
    std::uint8_t policy,int device,void *stream,const float *block_scales)
{
    return nativeM1WithReduction(A,payload,scales,mins,emins,output,scale_a,partials,
        n,k,execution,policy,device,stream,block_scales,&ticket);
}

/** @copydoc enqueueNativeM1HCDownProjection */
bool enqueueNativeM1HCDownProjection(const std::int8_t *A,const std::uint8_t *payload,
    const void *scales,const void *mins,const void *emins,float *output,const float *scale_a,
    float *partials,int n,int k,std::uint8_t execution,std::uint8_t policy,int device,void *stream,
    const float *block_scales,const GPUHCDownOutputView &publication,
    const NativeQ8SerialGeometryAdmission *prepared)
{
 if(execution!=19||policy!=19||n<64||n>512||n%32||k<=32||k%32||
    publication.columns!=n||publication.rows.physicalRows()!=1||!publication.activated||
    !publication.primary||!publication.residual||!publication.primaryScales||!publication.residualScales||
    !partials||!block_scales||reinterpret_cast<std::uintptr_t>(stream)<=2||device<0||
    g_native_q8_serial_scope||g_native_quantised_kb_override.load(std::memory_order_relaxed)>0||
    g_native_quantised_target_waves_override.load(std::memory_order_relaxed)>0||
    debugEnv().rocm.native_quantised_gemv_kb>0||debugEnv().rocm.native_quantised_gemv_target_waves>0)
    throw std::invalid_argument("HC down epilogue requires original native19 M1 reducer policy and sealed complete banks");
 NativeM1ReductionTicket ticket;
 if(!nativeM1WithReduction(A,payload,scales,mins,emins,output,scale_a,partials,
    n,k,execution,policy,device,stream,block_scales,&ticket,prepared))
    throw std::runtime_error("Original HC down producer enqueue failed");
 const auto& shape=ticket.producerShape();
 if(ticket.state_!=NativeM1ReductionTicket::State::Pending||shape.tile_columns!=128||shape.columns_per_thread!=1)
    throw std::logic_error("HC down epilogue has no original pending128-column reducer");
 if(ticket.has_work_)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1HCDownReducePublish<true>),dim3((n+127)/128),dim3(128),0,
        static_cast<hipStream_t>(stream),ticket.partials_,ticket.output_,n,shape.partitions,ticket.work_,publication);
 else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1HCDownReducePublish<false>),dim3((n+127)/128),dim3(128),0,
        static_cast<hipStream_t>(stream),ticket.partials_,ticket.output_,n,shape.partitions,ticket.work_,publication);
 if(hipGetLastError()!=hipSuccess)throw std::runtime_error("Original HC reducer epilogue enqueue failed");
 ticket.state_=NativeM1ReductionTicket::State::Consumed;
 return true;
}

bool submitNativeM1Reductions(std::span<NativeM1ReductionTicket> tickets,int device,void *stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2 || device<0 || tickets.empty() || tickets.size()>4)
        throw std::invalid_argument("M1 reduction wave requires exact stream/device and one to four tickets");
    // Validate all authorities before submitting any reducer or consuming any ticket.
    for(const auto &t:tickets)
        if((t.state_!=NativeM1ReductionTicket::State::Pending && t.state_!=NativeM1ReductionTicket::State::Direct) ||
            t.device_!=device || t.stream_!=stream)
            throw std::invalid_argument("M1 reduction ticket is absent, consumed or foreign");
    const auto interval=[](const void *p,std::size_t bytes)
    {
        const auto address=reinterpret_cast<std::uintptr_t>(p);
        if(!address || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
            throw std::invalid_argument("M1 reduction operand extent is absent or overflows");
        return std::pair{address,address+bytes};
    };
    const auto overlap=[](auto a,auto b){return a.first<b.second && b.first<a.second;};
    for(std::size_t i=0;i<tickets.size();++i)
    {
        const auto &a=tickets[i];const auto out=interval(a.output_,std::size_t(a.shape_.columns)*sizeof(float));
        for(std::size_t j=0;j<tickets.size();++j)
        {
            const auto &b=tickets[j];
            if(i!=j && overlap(out,interval(b.output_,std::size_t(b.shape_.columns)*sizeof(float))))
                throw std::invalid_argument("M1 reduction outputs alias");
            if(b.state_==NativeM1ReductionTicket::State::Pending &&
                overlap(out,interval(b.partials_,std::size_t(b.shape_.columns)*b.shape_.partitions*sizeof(float))))
                throw std::invalid_argument("M1 output aliases pending partial storage");
            if(i!=j && a.state_==NativeM1ReductionTicket::State::Pending && b.state_==NativeM1ReductionTicket::State::Pending &&
                overlap(interval(a.partials_,std::size_t(a.shape_.columns)*a.shape_.partitions*sizeof(float)),
                    interval(b.partials_,std::size_t(b.shape_.columns)*b.shape_.partitions*sizeof(float))))
                throw std::invalid_argument("M1 pending producers share partial storage");
        }
    }
    if(static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device))!=hipSuccess)return false;
    std::array<bool,4> selected{};
    for(std::size_t first=0;first<tickets.size();++first)
    {
        if(selected[first])continue;
        auto &leader=tickets[first];
        if(leader.state_==NativeM1ReductionTicket::State::Direct){selected[first]=true;continue;}
        std::array<NativeM1ReductionShape,4> shapes{};std::array<std::size_t,4> indices{};std::size_t count=0;
        for(std::size_t i=first;i<tickets.size();++i)
            if(!selected[i] && tickets[i].state_==NativeM1ReductionTicket::State::Pending &&
                tickets[i].shape_.tile_columns==leader.shape_.tile_columns &&
                tickets[i].shape_.columns_per_thread==leader.shape_.columns_per_thread)
            {selected[i]=true;indices[count]=i;shapes[count++]=tickets[i].shape_;}
        auto grid=nativeM1ReductionGrid(std::span(shapes.data(),count));
        for(std::size_t i=0;i<count;++i){grid.partials[i]=tickets[indices[i]].partials_;grid.outputs[i]=tickets[indices[i]].output_;grid.work[i]=tickets[indices[i]].work_;grid.has_work=grid.has_work || tickets[indices[i]].has_work_;}
#define LAUNCH_M1_REDUCTION(TN,CPT) \
        do { if(grid.has_work) { \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1GroupedReduce<TN,CPT,true>),dim3(grid.tiles),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),grid); \
        } else { \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1GroupedReduce<TN,CPT,false>),dim3(grid.tiles),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),grid); \
        } } while(false)
        switch(leader.shape_.tile_columns*10+leader.shape_.columns_per_thread)
        {
        case 641: LAUNCH_M1_REDUCTION(64,1);break;
        case 1281: LAUNCH_M1_REDUCTION(128,1);break;
        case 1282: LAUNCH_M1_REDUCTION(128,2);break;
        case 2561: LAUNCH_M1_REDUCTION(256,1);break;
        case 2562: LAUNCH_M1_REDUCTION(256,2);break;
        default:throw std::invalid_argument("M1 reduction ticket has no compiled geometry");
        }
#undef LAUNCH_M1_REDUCTION
        if(hipPeekAtLastError()!=hipSuccess)
        {
            for(auto &t:tickets)t.state_=NativeM1ReductionTicket::State::Consumed;
            return false;
        }
    }
    for(auto &t:tickets)t.state_=NativeM1ReductionTicket::State::Consumed;
    return true;
}
}

namespace llaminar2::rocm
{
/** @copydoc downLastProducerPlan */
NativeDownCompletionPlan downLastProducerPlan(int columns,int reduction,int routes,std::uint32_t policy_mask)
{
    ROCmMoESerialM1PolicyTable policy{};
    if(columns<=0||reduction<=0||routes<=0||columns%32||reduction%32||
        !buildROCmMoESerialM1PolicyTable(columns,reduction,policy_mask,policy))
        throw std::invalid_argument("Last-producer down has no original serial-M1 partition policy");
    return NativeDownCompletionPlan::make(columns,routes,policy.maximum_k_partitions);
}
/** @copydoc downLastProducerKernelFunction */
const void* downLastProducerKernelFunction() noexcept
{return reinterpret_cast<const void*>(moe_grouped_native_quantised_down_last_producer_kernel);}

/** @copydoc gateUpLastProducerPlan */
NativeTileCompletionPlan gateUpLastProducerPlan(int columns,int reduction,int routes,std::uint32_t policy_mask)
{
    ROCmMoESerialM1PolicyTable original{};
    if(columns<=0||reduction<=0||routes<=0||columns%32||reduction%32||
        !buildROCmMoESerialM1PolicyTable(columns,reduction,policy_mask,original))
        throw std::invalid_argument("Last-producer gate/up requires its complete original M1 source policy");
    return NativeTileCompletionPlan::make(columns,routes,original.maximum_k_partitions);
}
/** @return Exact original-grid last-producer program identity, without device resource queries. */
const void* gateUpLastProducerKernelFunction() noexcept
{
    return currentGateUpProducerLoads()==GateUpProducerLoads::PairedK32
        ?reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel)
        :reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_last_producer_kernel);
}
/** @copydoc gateUpLastProducerKernelFunction(GateUpProducerLoads) */
const void* gateUpLastProducerKernelFunction(GateUpProducerLoads loads)
{
    if(loads==GateUpProducerLoads::Original)
        return reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_last_producer_kernel);
    if(loads==GateUpProducerLoads::PairedK32)
        return reinterpret_cast<const void*>(&moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel);
    throw std::invalid_argument("Unknown gate/up producer loads for native function identity");
}
}

#include "ROCmPersistentHCReadBody.hip.inc"

namespace llaminar2::rocm {
/** @brief Validate the borrowed original physical phase bindings without reading live GPU state. */
static void validatePersistentHCRead(const PersistentHCReadArgs& a,bool ctaOwned=false,
 FullNativeHCFFNOutputDemand output_demand=FullNativeHCFFNOutputDemand::all()) {
 const int H=a.norm.hidden,R=a.down_weights.n;
 if(a.norm.extent.rows!=1||H<=0||(ctaOwned?H%8:H%32)||static_cast<std::int64_t>(H)*4!=a.down_weights.k||
    R<64||R>512||R%32||!std::isfinite(a.norm.epsilon)||a.norm.epsilon<=0.0f||
    !a.norm.input||!a.norm.gain||!a.norm.output||!a.input_codes||!a.input_scales||
    !a.down_partials||!a.raw_down||!a.mixed||
    (output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)&&!a.publication.activated)||
    a.down_weights.codebook_id!=19||a.up.weights.codebook_id!=19||
    !a.down_weights.source_identity_present||!a.up.weights.source_identity_present||
    a.publication.columns!=R||a.up.weights.k!=R||a.up.weights.n!=4*H||
    a.up.rows.physicalRows()!=1||a.publication.rows.physicalRows()!=1||
    a.up.rows.captureIdentity()!=a.publication.rows.captureIdentity()||
    a.up.primary!=a.publication.primary||a.up.residual!=a.publication.residual||
    a.up.primaryScales!=a.publication.primaryScales||a.up.residualScales!=a.publication.residualScales||
    a.up.quantizer!=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal||
    (output_demand.contains(FullNativeHCFFNOutputRole::HCUp)&&!a.up.output)||
    a.publication.primary!=a.input_codes||a.publication.primaryScales!=a.input_scales||
    (a.injection.weights==nullptr)!=(a.injection.output==nullptr)||
    (a.injection.weights&&(a.injection.rows.physicalRows()!=1||
      (a.injection.load_pairs!=1&&a.injection.load_pairs!=4&&a.injection.load_pairs!=8))))
  throw std::invalid_argument("Persistent HC requires exact original M1 branch/native19/publication bindings");
 (void)requireQ8ResidualNativeMatrix(a.down_weights,1,R,4*H);
 (void)requireQ8ResidualNativeMatrix(a.up.weights,1,4*H,R);
 const std::size_t K=static_cast<std::size_t>(4)*H;
 // Each phase is checked separately: the original full input code/scale bank
 // deliberately becomes the shorter two-term primary prefix after down retires.
 const std::array normWrites{Q8ResidualStorageSpan{a.norm.output,K*sizeof(float),alignof(float)}};
 const std::array normReads{Q8ResidualStorageSpan{a.norm.gain,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:K*sizeof(float),alignof(float)}};
 validateQ8ResidualStorage(normWrites,normReads);
 const std::array downWrites{Q8ResidualStorageSpan{a.input_codes,K,16},
  Q8ResidualStorageSpan{a.input_scales,K/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_partials,static_cast<std::size_t>(a.partitions)*R*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.injection.output,a.injection.output?4*sizeof(float):0,alignof(float)}};
 const std::array downReads{Q8ResidualStorageSpan{a.norm.output,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.gain,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_weights.payload,K/32*R*32,16},
  Q8ResidualStorageSpan{a.down_weights.scales,K/32*R*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.injection.weights,a.injection.weights?4*K*sizeof(float):0,alignof(float)}};
 validateQ8ResidualStorage(downWrites,downReads);
 const std::array endWrites{Q8ResidualStorageSpan{a.raw_down,static_cast<std::size_t>(R)*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.publication.activated,
   output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)&&a.publication.activated!=a.raw_down?
    static_cast<std::size_t>(R)*sizeof(float):0,alignof(float)},
  Q8ResidualStorageSpan{a.publication.primary,static_cast<std::size_t>(R),16},
  Q8ResidualStorageSpan{a.publication.residual,static_cast<std::size_t>(R),16},
  Q8ResidualStorageSpan{a.publication.primaryScales,static_cast<std::size_t>(R)/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.publication.residualScales,static_cast<std::size_t>(R)/32*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.up.output,output_demand.contains(FullNativeHCFFNOutputRole::HCUp)?K*sizeof(float):0,alignof(float)},
  Q8ResidualStorageSpan{a.mixed,static_cast<std::size_t>(H)*sizeof(float),alignof(float)}};
 const std::array endReads{Q8ResidualStorageSpan{a.norm.output,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.input,a.norm.input==a.norm.output?0:K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.norm.gain,K*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_partials,static_cast<std::size_t>(a.partitions)*R*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{a.down_weights.payload,K/32*R*32,16},
  Q8ResidualStorageSpan{a.down_weights.scales,K/32*R*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.up.weights.payload,K/32*R*32,16},
  Q8ResidualStorageSpan{a.up.weights.scales,K/32*R*sizeof(std::uint16_t),alignof(std::uint16_t)},
  Q8ResidualStorageSpan{a.injection.output,a.injection.output?4*sizeof(float):0,alignof(float)}};
 validateQ8ResidualStorage(endWrites,endReads);
}
/** @copydoc resolvePersistentHCReadPolicy */
void resolvePersistentHCReadPolicy(PersistentHCReadArgs& args,FullNativeHCFFNOutputDemand output_demand) {
 if(isGraphCaptureActive())
  throw std::invalid_argument("Persistent HC source-policy resolution requires cold preparation");
 const auto& env=debugEnv().rocm;
 if(g_native_q8_serial_scope||g_native_quantised_kb_override.load(std::memory_order_relaxed)>0||
    g_native_quantised_target_waves_override.load(std::memory_order_relaxed)>0||
    env.native_quantised_gemv_kb>0||env.native_quantised_gemv_target_waves>0)
  throw std::invalid_argument("Persistent HC cannot replace an explicitly overridden original arithmetic policy");
 if(args.down_weights.codebook_id!=19||!args.down_weights.source_identity_present)
  throw std::invalid_argument("Persistent HC lost original signed-Q8 source policy");
 args.partitions=resolveNativeQuantisedSerialM1KPartitionsForPolicy(
  canonicalDeviceVnniCodebookId(args.down_weights.source_codebook_id),args.down_weights.n,args.down_weights.k);
 if(args.partitions<=0)throw std::invalid_argument("Persistent HC original split-K source policy is absent");
 validatePersistentHCRead(args,false,output_demand);
}
/** @copydoc preparePersistentHCRead */
NativeCooperativeKernelAdmission preparePersistentHCRead(DeviceId device,PersistentHCReadArgs& args,
 std::uint64_t generation) {
 if(!device.is_rocm()||!generation||isGraphCaptureActive())
  throw std::invalid_argument("Persistent HC admission requires its original native owner cold");
 resolvePersistentHCReadPolicy(args);
 const auto admission=NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(device,
  reinterpret_cast<const void*>(persistentHCReadKernel),256,0,generation);
 const unsigned downTasks=static_cast<unsigned>((args.down_weights.n+63)/64)*args.partitions+
  (args.injection.weights?4u:0u);
 const unsigned liveTasks=std::max({4u,static_cast<unsigned>((args.down_weights.k/32+7)/8),downTasks,
  static_cast<unsigned>((args.down_weights.n+255)/256),static_cast<unsigned>((args.norm.hidden+7)/8)});
 unsigned bound=admission.maximumGrid();
 if(args.grid_policy==PersistentHCGridPolicy::OnePerProcessor)
  bound=std::min(bound,static_cast<unsigned>(rocmHardwareMetadata().at(device.ordinal).compute_units));
 else if(args.grid_policy!=PersistentHCGridPolicy::MaximumResident)
  throw std::invalid_argument("Persistent HC grid policy is not declared");
 args.resident_grid=std::min(bound,liveTasks);
 return admission;
}
/** @copydoc launchPersistentHCRead */
void launchPersistentHCRead(const NativeCooperativeKernelAdmission& admission,const PersistentHCReadArgs& args,
 void* stream,std::uint64_t generation) {
 if(admission.function()!=reinterpret_cast<const void*>(persistentHCReadKernel)||admission.threads()!=256||
    !args.resident_grid||args.resident_grid>admission.maximumGrid())
  throw std::invalid_argument("Persistent HC exact prepared program/grid changed");
 void* parameters[]{const_cast<PersistentHCReadArgs*>(&args)};
 NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::launch(admission,parameters,
  args.resident_grid,stream,generation);
}
} // namespace llaminar2::rocm

#include "ROCmCTAOwnedHCReadBody.hip.inc"
namespace llaminar2::rocm {
/** @copydoc prepareCTAOwnedHCRead */
CTAOwnedHCReadProgram prepareCTAOwnedHCRead(DeviceId device,PersistentHCReadArgs& args,std::uint64_t generation) {
 if(!device.is_rocm()||!generation||isGraphCaptureActive())
  throw std::invalid_argument("CTA-owned HC requires its exact original native owner cold");
 int currentDevice=-1;
 if(hipGetDevice(&currentDevice)!=hipSuccess||currentDevice!=device.ordinal)
  throw std::invalid_argument("CTA-owned HC cold resource admission has a foreign current native device");
 const auto& env=debugEnv().rocm;
 if(g_native_q8_serial_scope||g_native_quantised_kb_override.load(std::memory_order_relaxed)>0||
    g_native_quantised_target_waves_override.load(std::memory_order_relaxed)>0||
    env.native_quantised_gemv_kb>0||env.native_quantised_gemv_target_waves>0)
  throw std::invalid_argument("CTA-owned HC cannot replace an overridden original arithmetic policy");
 args.partitions=resolveNativeQuantisedSerialM1KPartitionsForPolicy(
  canonicalDeviceVnniCodebookId(args.down_weights.source_codebook_id),args.down_weights.n,args.down_weights.k);
 requireCTAOwnedHCTailGeometry(args.norm.hidden,args.partitions,args.raw_down==args.publication.activated);
 // The complete storage validator is shared with the original implementation;
 // its H%32 restriction belongs to that cooperative program, not this tail.
 validatePersistentHCRead(args,true);
 CTAOwnedHCReadProgram result;result.device_=device;result.generation_=generation;
 result.hidden_=args.norm.hidden;result.rank_=args.down_weights.n;result.partitions_=args.partitions;
 const auto select=[&]<unsigned G>() {return inspectNativeQ8SerialSymbol(
  reinterpret_cast<const void*>(ctaOwnedHCTailKernel<G>),"ctaOwnedHCTailKernel",1024);};
 switch(args.down_weights.n/32) {
 case 2:result.resources_=select.template operator()<2>();break;
 case 3:result.resources_=select.template operator()<3>();break;
 case 4:result.resources_=select.template operator()<4>();break;
 case 5:result.resources_=select.template operator()<5>();break;
 case 6:result.resources_=select.template operator()<6>();break;
 case 7:result.resources_=select.template operator()<7>();break;
 case 8:result.resources_=select.template operator()<8>();break;
 case 9:result.resources_=select.template operator()<9>();break;
 case 10:result.resources_=select.template operator()<10>();break;
 case 11:result.resources_=select.template operator()<11>();break;
 case 12:result.resources_=select.template operator()<12>();break;
 case 13:result.resources_=select.template operator()<13>();break;
 case 14:result.resources_=select.template operator()<14>();break;
 case 15:result.resources_=select.template operator()<15>();break;
 case 16:result.resources_=select.template operator()<16>();break;

 default:throw std::invalid_argument("CTA-owned HC has no complete original short-K group specialization");
 }
 return result;
}
/** @copydoc launchCTAOwnedHCRead */
void launchCTAOwnedHCRead(const CTAOwnedHCReadProgram& program,const PersistentHCReadArgs& args,
 void* stream,std::uint64_t generation) {
 if(reinterpret_cast<std::uintptr_t>(stream)<=2||program.generation_!=generation||!generation||
    !program.device_.is_rocm()||program.hidden_!=args.norm.hidden||program.rank_!=args.down_weights.n||
    program.partitions_!=args.partitions||!program.resources_.function)
  throw std::invalid_argument("CTA-owned HC exact prepared program/workspace geometry changed");
 if(!rocmQuantGemm_quantizeActivationsBlockwise(args.norm.output,args.input_codes,args.input_scales,
    1,args.down_weights.k,program.device_.ordinal,stream,32))
  throw std::runtime_error("CTA-owned HC original SingleQ8 producer failed");
 NativeM1ReductionTicket ticket;
 if(!enqueueNativeM1Producer(ticket,args.input_codes,args.down_weights.payload,args.down_weights.scales,
    args.down_weights.mins,args.down_weights.emins,args.raw_down,nullptr,args.down_partials,
    args.down_weights.n,args.down_weights.k,args.down_weights.codebook_id,
    canonicalDeviceVnniCodebookId(args.down_weights.source_codebook_id),program.device_.ordinal,stream,args.input_scales))
  throw std::runtime_error("CTA-owned HC original down producer failed");
 CTAOwnedHCTicketConsumer::submit(ticket,program,args,program.device_.ordinal,stream);
}
}

#include "ROCmNativeLayoutRouterFFNBody.hip.inc"
#include "ROCmNativeLayoutRouterFFNLaunch.hip.inc"
#include "ROCmFullNativeHCFFNBody.hip.inc"
#include "ROCmFullNativeHCFFNLaunch.hip.inc"

#if !defined(__HIP_DEVICE_COMPILE__)
#include "ROCmRoutedGridDonation.h"

namespace llaminar2::rocm {
/** @copydoc inspectRoutedGridDonation */
RoutedGridDonation inspectRoutedGridDonation(std::uint32_t routed_formats, unsigned prefix_bytes,
    std::size_t hidden_bytes, std::size_t panel_and_records_bytes) noexcept {
    const auto after_hidden = prefix_bytes > hidden_bytes ? unsigned(prefix_bytes - hidden_bytes) : 0u;
    const auto available = after_hidden > panel_and_records_bytes ? after_hidden - panel_and_records_bytes : 0;
    const auto layout = routedGridLayout(routed_formats, available);
    return {layout.iq2_entries, layout.iq3_small_entries, layout.iq3_extra_small_entries};
}
}
#endif
