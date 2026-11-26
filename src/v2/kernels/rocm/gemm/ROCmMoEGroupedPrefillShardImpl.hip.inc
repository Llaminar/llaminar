#ifndef LLAMINAR_ROCM_MOE_PREFILL_SHARD
#define LLAMINAR_ROCM_MOE_PREFILL_SHARD -1
#endif
/**
 * @file ROCmMoEGroupedPrefillKernels.hip
 * @brief Captured, byte-exact grouped MoE prefill for gfx906 and gfx1100.
 *
 * The pipeline is entirely device-owned and graph-capturable. It consumes the
 * route groups and descriptor tables already resident on the GPU; no launch
 * reads route counts on the host and no stream synchronization is required.
 *
 * Small verifier buckets and exact trained sparse-overlay shapes use direct
 * route-owned projections because their latency is dominated by launch and
 * directory overhead. Longer prefill uses a compact device-built tile
 * directory and a generated policy trained across every execution codebook,
 * reciprocal aspect ratio, and M anchor:
 *
 *   1. Reuse router-owned Q8 rows, or quantize each original hidden row once.
 *   2. Build a packed expert/tile directory entirely on the device.
 *   3. Decode each gate/up weight block once, resolving exact Q6 source factors
 *      (other formats retain prepared metadata), for a trained 4/8/12/16-row tile
 *      while retaining serial M=1 partition boundaries and reduction order.
 *   4. Quantize exact SwiGLU rows for the down projection.
 *   5. Decode each down weight block once per independently trained row tile
 *      into route partials.
 *   6. Publish those partials with one writer per token/output column, adding
 *      routes in original top-k order.
 * Directory builders borrow the same device admission as tiled consumers;
 * an inactive family performs no prefix scan or directory write. Reactivating
 * that family rebuilds its directory from the newly published group plan.
 *
 * The capture-time projection transaction can stop after step 4 or begin at
 * step 5. Both forms reuse these exact kernels. A down column slice carries
 * the original full-width arithmetic key, so changing physical ownership
 * cannot silently change serial-M1 partition boundaries.
 *
 * Native gfx1100 WMMA recording consumes the same directory16 publication and
 * source partition policy, with separate physical IQ4 decoders and unchanged
 * SwiGLU/ordered publication. Its dot-engine policy is independent of live route
 * admission; sparse replay retains its existing route-owned family.
 * Native row-floor candidates select Dot4 or WMMA inside one consumer using the
 * whole-expert count. Microtiles share directory16; the wide candidate binds its
 * role-specific row plan to both planner and consumer. The last partial tile
 * cannot select a different engine for the same expert.
 *
 * The long-prefill Dot4 projection kernels expose 64/128/256 output-column lanes and
 * 4/8/12/16-row tiles. Production dispatch comes exclusively from the checked-in
 * all-format trainer output; a missing policy is a hard failure. No production
 * path uses floating-point atomics, row replay, replay-time host-authored work
 * metadata, or scheduler-dependent reduction order, so grouped verifier
 * publication remains byte-identical to serial row decode.
 *
 * @see ROCmGemvKernel_native_quantised.hip for decode grouped kernels (M=1)
 * @see ROCmNativeQuantisedGemmShardImpl.hip.inc for non-grouped prefill GEMM
 * Native signed INT8 dot products share the exact unsaturated arithmetic
 * contract across gfx906 and gfx1100; lane ownership and reductions are unchanged.
 */

#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/rocm/common/ROCmQ4PairedDot.h"
#include "kernels/rocm/common/ROCmWaveAddress.h"
#include <bit>
#include <hip/hip_runtime.h>
#include "kernels/common/MoEGroupedDirectory.h"
#include <hip/hip_fp16.h>
#include "kernels/rocm/common/ROCmIQ3XXSLookupBatch.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <cstdio>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../tensors/NativeQuantisedFormatInfo.h"
#include "../../../utils/DebugEnv.h"
#include "../../common/NativeQuantisedGroupedDecodePolicy.h"
#include "../../common/DeviceNativeQuantisedMatrixDesc.h"
#include "../../common/DeviceOrderedRouteFold.h"
#include "ROCmMoEGroupedPrefillKernels.h"
#include "ROCmMoEPrefillArithmetic.h"
#include "ROCmMoEGroupedWmma.h"
#include "ROCmMoEGroupedPopulationProjection.h"
#include "ROCmMoEGroupedCompactPopulation.h"
#include "execution/moe/MoEGroupedWorkspaceGeometry.h"
#include "ROCmMoERouteOwnedTuning.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include <exception>
#include <iterator>
#include <string>
#include "../ops/ROCmSwiGLUDecodeEquivalentMath.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "ROCmMoEGroupedPrefillDispatchGenerated.inc"
#include "ROCmMoEProductionPrefillPolicy.h"
#include "ROCmMoEGroupedPrefillRoutePolicy.h"

using llaminar2::rocm::ROCmMoEGroupedRouteAdmission;
using llaminar2::rocm::ROCmMoEGroupedPrefillRouteStrategy;
using llaminar2::rocm::MoEPrefillProjectionRole;
using llaminar2::rocm::MoEGroupedDotAdmission;
using llaminar2::rocm::MoEGroupedDotProducer;
using llaminar2::rocm::MoEGroupedRowPlan;
using llaminar2::rocm::MoEGroupedPopulationBinding;
using llaminar2::rocm::MoEGroupedPopulationControl;

/**
 * @brief Bound publication scheduling independently of captured row capacity.
 *
 * Both arithmetic families stay in the graph because live routing chooses the
 * winner on device. A full slot-by-column grid made even the inactive family
 * dispatch hundreds of thousands of workgroups. Thirty-two slot lanes retain
 * column parallelism for sparse routing; each lane walks disjoint original
 * slots, preserving one writer per element without changing any reduction.
 */
static constexpr int kCanonicalPublicationSlotLanes = 32;

namespace
{
/** One lexical host lowering scope; no mutable device value is mirrored here. */
thread_local const llaminar2::rocm::ScopedMoERouteOwnedGateUpGeometry *g_moe_route_scope = nullptr;
/** Accepted host-enqueue observation only; graph replay never writes this value. */
thread_local std::optional<llaminar2::rocm::MoERouteOwnedGateUpLaunchSet> g_moe_route_launch;
/**
 * @brief Resolve one explicit construction authority; model bindings never inherit a diagnostic override.
 * @param explicit_admission Immutable graph/table-owned capability, or null for ordinary/diagnostic calls.
 * @return Exactly the supplied model capability or the original lexical diagnostic capability.
 * @throws std::logic_error If both authorities are present.
 */
const llaminar2::rocm::MoERouteOwnedGateUpAdmission *resolveMoERouteOwnedAdmission(
    const llaminar2::rocm::MoERouteOwnedGateUpAdmission *explicit_admission)
{
    if (explicit_admission && g_moe_route_scope)
        throw std::logic_error("model route-owned capability cannot overlap a diagnostic scope");
    return explicit_admission ? explicit_admission :
        (g_moe_route_scope ? &g_moe_route_scope->admission() : nullptr);
}

/**
 * @brief Stack-owned complete host-submission observation shared by explicit model and diagnostic lowering.
 *
 * No pending pointer or mutable model policy is thread-local. The raw call owns
 * this bounded record, each original leaf appends to it, and only complete native
 * submission success can publish it to that caller. Replay never executes it.
 */
class MoERouteOwnedSubmissionObservation final
{
public:
    /** @brief Start one original raw transaction under its immutable capability. */
    explicit MoERouteOwnedSubmissionObservation(const llaminar2::rocm::MoERouteOwnedGateUpAdmission &admission)
        : admission_(admission)
    { pending_.execution_envelope = admission.executionEnvelope(); }
    MoERouteOwnedSubmissionObservation(const MoERouteOwnedSubmissionObservation &) = delete;
    MoERouteOwnedSubmissionObservation &operator=(const MoERouteOwnedSubmissionObservation &) = delete;
    /**
     * @brief Append one accepted leaf locally, rejecting duplicate/foreign/overflow records.
     * @param leaf Actual successful original host submission with exact native resources.
     * @throws std::logic_error For incomplete ownership or a repeated/foreign decoder.
     */
    void record(const llaminar2::rocm::MoERouteOwnedGateUpLaunch &leaf)
    {
        const auto bit = llaminar2::nativeQuantisedCodebookMaskBit(leaf.execution_codebook);
        if (!(admission_.launchCodebookMask() & bit) || pending_.leaf_count >= pending_.leaves.size())
            throw std::logic_error("route-owned observation has a foreign leaf or exceeds its bounded ABI");
        for (std::size_t index = 0; index < pending_.leaf_count; ++index)
            if (pending_.leaves[index].execution_codebook == leaf.execution_codebook)
                throw std::logic_error("route-owned decoder family was submitted twice");
        pending_.leaves[pending_.leaf_count++] = leaf;
    }
    /**
     * @brief Return only a complete original normalized family set.
     * @return Immutable accepted host construction evidence, not hardware/replay work.
     * @throws std::logic_error For any omitted or repeated admitted family.
     */
    const llaminar2::rocm::MoERouteOwnedGateUpLaunchSet &complete() const
    {
        std::uint32_t actual = 0;
        for (std::size_t index = 0; index < pending_.leaf_count; ++index)
        {
            const auto bit = llaminar2::nativeQuantisedCodebookMaskBit(pending_.leaves[index].execution_codebook);
            if (!bit || (actual & bit)) throw std::logic_error("route-owned observation repeated a decoder family");
            actual |= bit;
        }
        if (actual != admission_.launchCodebookMask() || pending_.leaf_count != admission_.leafCount())
            throw std::logic_error("route-owned observation omitted an admitted decoder family");
        return pending_;
    }
private:
    const llaminar2::rocm::MoERouteOwnedGateUpAdmission &admission_;
    llaminar2::rocm::MoERouteOwnedGateUpLaunchSet pending_{};
};

/** @brief Reject arithmetic/expert-tile mixing without querying HIP or changing policy. */
void requireMoERouteOwnedUnmodifiedPolicy()
{
    const auto &env = llaminar2::debugEnv().rocm;
    if (env.native_quantised_disable_generated || env.native_quantised_gemv_kb > 0 ||
        env.native_quantised_gemv_target_waves > 0 || env.moe_prefill_gateup_tile_m > 0 ||
        env.moe_prefill_gateup_tile_n > 0 || env.moe_prefill_down_tile_m > 0 ||
        env.moe_prefill_down_tile_n > 0)
        throw std::logic_error("route-owned geometry requires unchanged serial arithmetic and no expert-tile overrides");
}
} // namespace

#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
void llaminar2::rocm::validateMoERouteOwnedGateUpScopeBinding(
    int device, void *stream, int rows, const MoEPrefillProjectionExecution &execution,
    MoERouteOwnedProjectionShape shape, const MoERouteOwnedGateUpAdmission *explicit_admission)
{
    if (!explicit_admission) g_moe_route_launch.reset();
    const auto *admission = resolveMoERouteOwnedAdmission(explicit_admission);
    if (!admission) return;
    if (device != admission->deviceOrdinal() || rows != 1 ||
        !execution.executesGateUp() || !stream ||
        stream == static_cast<void *>(hipStreamLegacy) ||
        stream == static_cast<void *>(hipStreamPerThread))
        throw std::logic_error("route-owned gate/up geometry requires physical M1 gate/up on its admitted device and exact nondefault stream");
    const auto &proof = *admission;
    if (proof.reductionWidth() > 0 &&
        (shape.reduction_columns != proof.reductionWidth() || shape.output_columns != proof.outputWidth()))
        throw std::logic_error("route-owned partition execution reached a foreign admitted N/K shape");
    requireMoERouteOwnedUnmodifiedPolicy();
}

void llaminar2::rocm::validateMoERouteOwnedGateUpCodebook(std::uint32_t mask, std::uint32_t policy_mask,
    const MoERouteOwnedGateUpAdmission *explicit_admission)
{
    const auto *admission = resolveMoERouteOwnedAdmission(explicit_admission);
    if (!admission) return;
    if (!explicit_admission) g_moe_route_launch.reset();
    const auto &envelope = admission->executionEnvelope();
    if (mask != envelope.execution_codebook_mask || policy_mask != envelope.policy_codebook_mask)
        throw std::logic_error("route-owned gate/up geometry requires its exact admitted physical/source-policy envelope");
}

std::optional<llaminar2::rocm::MoERouteOwnedGateUpGeometry>
llaminar2::rocm::currentMoERouteOwnedGateUpGeometry() noexcept
{
    return g_moe_route_scope ? std::optional{g_moe_route_scope->admission().geometry()} : std::nullopt;
}

std::optional<llaminar2::rocm::MoERouteOwnedGateUpLaunch>
llaminar2::rocm::lastMoERouteOwnedGateUpLaunch() noexcept
{
    if (!g_moe_route_launch || g_moe_route_launch->leaf_count != 1) return std::nullopt;
    return g_moe_route_launch->leaves[0];
}

std::optional<llaminar2::rocm::MoERouteOwnedGateUpLaunchSet>
llaminar2::rocm::lastMoERouteOwnedGateUpLaunchSet() noexcept { return g_moe_route_launch; }

llaminar2::rocm::ScopedMoERouteOwnedGateUpGeometry::ScopedMoERouteOwnedGateUpGeometry(
    const MoERouteOwnedGateUpAdmission &admission) : admission_(admission)
{
    g_moe_route_launch.reset();
    if (g_moe_route_scope || llaminar2::isGraphCaptureActive())
        throw std::logic_error("route-owned geometry must have one nonoverlapping scope established before capture");
    requireMoERouteOwnedUnmodifiedPolicy();
    (void)moeRouteOwnedGateUpColumns(admission.geometry());
    g_moe_route_scope = this;
}

llaminar2::rocm::ScopedMoERouteOwnedGateUpGeometry::~ScopedMoERouteOwnedGateUpGeometry() noexcept
{
    if (g_moe_route_scope != this) std::terminate();
    g_moe_route_scope = nullptr;
}
#endif

// The small shared ABI has no host tensor or x86 dependency. Do not maintain
// another descriptor definition beside the weight-preparation authority.
using llaminar2::DeviceNativeQuantisedMatrixDesc;

__device__ __forceinline__ void moe_prefill_fail_fast_invalid_grouped_index()
{
    __builtin_trap();
}

// NativeQuantisedFormat enum (must match ROCmGemvKernel_native_quantised.hip)
enum NativeQuantisedFormatPrefill : uint8_t {
    NATIVE_QUANTISED_PF_Q4_0    = 0,
    NATIVE_QUANTISED_PF_IQ4_XS_SOURCE = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook,
    NATIVE_QUANTISED_PF_IQ4_NL  = 4,
    NATIVE_QUANTISED_PF_Q4_1    = 5,
    NATIVE_QUANTISED_PF_Q5_0    = 6,
    NATIVE_QUANTISED_PF_Q5_1    = 7,
    NATIVE_QUANTISED_PF_Q6_K    = 8,
    NATIVE_QUANTISED_PF_Q3_K    = 9,
    NATIVE_QUANTISED_PF_Q2_K    = 10,
    NATIVE_QUANTISED_PF_IQ3_S   = 11,
    NATIVE_QUANTISED_PF_IQ3_XXS = 12,
    NATIVE_QUANTISED_PF_IQ2_S   = 13,
    NATIVE_QUANTISED_PF_IQ2_XS  = 14,
    NATIVE_QUANTISED_PF_IQ2_XXS = 15,
    NATIVE_QUANTISED_PF_IQ1_S   = 16,
    NATIVE_QUANTISED_PF_IQ1_M   = 17,
    NATIVE_QUANTISED_PF_Q8_0    = 19,
    NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN =
        llaminar2::kNativeQuantisedExpandedInt8MinCodebook,
};

extern "C" bool rocmGemv_native_quantised_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu);

// Compile-time format properties for prefill (mirrors NATIVE_QUANTISEDTraits in GEMV)
template <NativeQuantisedFormatPrefill FMT>
struct PrefillNATIVE_QUANTISEDTraits {
    static constexpr bool is_asymmetric =
        FMT == NATIVE_QUANTISED_PF_Q4_1 || FMT == NATIVE_QUANTISED_PF_Q5_1 ||
        FMT == NATIVE_QUANTISED_PF_IQ1_S || FMT == NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN;
    static constexpr bool is_iq3_grid = (FMT == NATIVE_QUANTISED_PF_IQ3_S || FMT == NATIVE_QUANTISED_PF_IQ3_XXS);
    static constexpr bool is_iq2_grid = (FMT == NATIVE_QUANTISED_PF_IQ2_S || FMT == NATIVE_QUANTISED_PF_IQ2_XS || FMT == NATIVE_QUANTISED_PF_IQ2_XXS);
    static constexpr bool is_iq1_grid = (FMT == NATIVE_QUANTISED_PF_IQ1_S || FMT == NATIVE_QUANTISED_PF_IQ1_M);
    static constexpr bool is_iq_grid  = (is_iq3_grid || is_iq2_grid);
    static constexpr int qh_bits_per_lookup =
        (FMT == NATIVE_QUANTISED_PF_IQ2_S) ? 2 : (FMT == NATIVE_QUANTISED_PF_IQ3_S || FMT == NATIVE_QUANTISED_PF_IQ2_XS) ? 1 : 0;
    static constexpr bool has_qh = (qh_bits_per_lookup > 0);
    static constexpr bool is_dual_scale = (FMT == NATIVE_QUANTISED_PF_Q6_K || FMT == NATIVE_QUANTISED_PF_Q3_K || FMT == NATIVE_QUANTISED_PF_Q2_K
                                        || FMT == NATIVE_QUANTISED_PF_IQ2_S || FMT == NATIVE_QUANTISED_PF_IQ2_XS
                                        || FMT == NATIVE_QUANTISED_PF_IQ1_M);
    static constexpr bool is_dual_scale_asym = (FMT == NATIVE_QUANTISED_PF_Q2_K);
    static constexpr bool needs_emin_array = (FMT == NATIVE_QUANTISED_PF_Q2_K);
    static constexpr bool is_5bit = (FMT == NATIVE_QUANTISED_PF_Q5_0 || FMT == NATIVE_QUANTISED_PF_Q5_1);
    static constexpr bool is_6bit = (FMT == NATIVE_QUANTISED_PF_Q6_K);
    static constexpr bool is_3bit = (FMT == NATIVE_QUANTISED_PF_Q3_K);
    static constexpr bool is_2bit = (FMT == NATIVE_QUANTISED_PF_Q2_K);
    static constexpr bool is_8bit =
        FMT == NATIVE_QUANTISED_PF_Q8_0 || FMT == NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN;
    static constexpr bool requires_secondary_metadata =
        is_asymmetric || is_dual_scale || FMT == NATIVE_QUANTISED_PF_IQ3_S || FMT == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE;
    static constexpr bool uses_lut = (FMT == NATIVE_QUANTISED_PF_IQ4_NL || FMT == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE);
    static constexpr int payload_bytes =
        is_8bit ? 32 :
        is_iq1_grid ? 6 :
        is_iq3_grid ? (has_qh ? 13 : 12) :
        is_iq2_grid ? (has_qh ? 9 : 8) :
        is_6bit ? 24 : is_5bit ? 20 : is_3bit ? 12 : is_2bit ? 8 : 16;
    static constexpr int center_4bit = (FMT == NATIVE_QUANTISED_PF_Q4_0) ? 8 : 0;
    static constexpr int center_5bit = (FMT == NATIVE_QUANTISED_PF_Q5_0) ? 16 : 0;
};

template <NativeQuantisedFormatPrefill FMT>
__device__ __forceinline__ bool prefill_native_quantised_desc_shape_ok(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int N,
    int K)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    return desc.payload &&
           desc.scales &&
           desc.n == N &&
           desc.k == K &&
           desc.blocks_per_row == static_cast<uint32_t>(K / 32) &&
           (desc.codebook_id == static_cast<uint8_t>(FMT) ||
            (FMT == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE && desc.codebook_id == NATIVE_QUANTISED_PF_IQ4_NL)) &&
           (!(Traits::requires_secondary_metadata && desc.codebook_id != NATIVE_QUANTISED_PF_IQ4_NL) || desc.mins) &&
           (!Traits::is_dual_scale_asym || desc.emins);
}

// IQ grid lookup tables (separate copies for this TU — initialized via
// rocmInitIQGridTables_moe_prefill, same data as GEMV/GEMM TUs)
static __device__ __constant__ uint32_t d_pf_iq3s_grid[512];
static __device__ __constant__ uint32_t d_pf_iq3xxs_grid[256];
static __device__ __constant__ uint64_t d_pf_iq2s_grid[1024];
static __device__ __constant__ uint64_t d_pf_iq2xs_grid[512];
static __device__ __constant__ uint64_t d_pf_iq2xxs_grid[256];
static __device__ __constant__ uint64_t d_pf_iq1s_grid[2048];

// IQ3 grid lookup (compile-time format dispatch)
template <NativeQuantisedFormatPrefill FMT>
__device__ inline uint32_t pf_iq3_grid_lookup(int idx)
{
    if constexpr (FMT == NATIVE_QUANTISED_PF_IQ3_S) return d_pf_iq3s_grid[idx];
    else                                  return d_pf_iq3xxs_grid[idx];
}

// IQ2 grid lookup (compile-time format dispatch)
template <NativeQuantisedFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ inline uint64_t pf_iq2_grid_lookup(
    int idx,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    if constexpr (FMT == NATIVE_QUANTISED_PF_IQ2_S)
    {
        if constexpr (USE_STAGED_IQ2_S)
            return staged_iq2s_grid[idx];
        else
            return d_pf_iq2s_grid[idx];
    }
    else if constexpr (FMT == NATIVE_QUANTISED_PF_IQ2_XS) return d_pf_iq2xs_grid[idx];
    else                                        return d_pf_iq2xxs_grid[idx];
}

// IQ1 grid lookup (shared for IQ1_S and IQ1_M)
__device__ inline uint64_t pf_iq1_grid_lookup(int idx)
{
    return d_pf_iq1s_grid[idx];
}

// Conditionally negate 4 unsigned bytes packed in uint32_t using 4 sign bits.
// sign_lo4 bit j → element j: 0=positive, 1=negative.
__device__ inline uint32_t pf_iq_apply_signs_4(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t spread = (static_cast<uint32_t>(sign_lo4 & 0xFu) * 0x00204081u) & 0x01010101u;
    uint32_t mask;
    asm("v_lshlrev_b32 %0, 8, %1\n\t"
        "v_sub_u32 %0, %0, %1"
        : "=v"(mask) : "v"(spread));
    return (grid4 ^ mask) + spread;
}


// ============================================================================
// Kernel 1: Original-Row Quantization (all tokens, single launch)
//
// Each routed expert consumes the same hidden row. Quantizing `[seq_len, K]`
// once avoids writing `top_k` byte-identical copies before expert grouping.
//
// Large batches pack eight independent Q8 blocks into one workgroup. Tiny
// batches retain one block per workgroup for enough independently schedulable
// work. Neither layout changes the 32-value arithmetic or its byte contract.
// ============================================================================

/**
 * @brief Publish original hidden rows through independent 32-lane Q8 trees.
 *
 * This is the heterogeneous expert quantizer: its fixed reciprocal tree must
 * not be replaced with an algebraically equivalent native division. Workgroup
 * packing only amortizes dispatch and fills both halves of each gfx9 wave.
 * Inactive subgroups return uniformly and never participate in another block's
 * max tree. No CTA barrier or additional storage is needed.
 *
 * @param hidden Contiguous FP32 original rows, not repeated per routed expert.
 * @param A_int8_out Same row geometry in signed Q8 bytes.
 * @param scales_out One canonical widened-half scale per 32 source values.
 * @param seq_len Physical captured row count, including padded rows.
 * @param K Positive row width, validated as a multiple of 32 by the launcher.
 */
#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
__global__ __launch_bounds__(256)
void moe_prefill_hidden_quantize_rows_kernel(
    const float* __restrict__ hidden,            // [seq_len, d_model]
    int8_t*      __restrict__ A_int8_out,        // [seq_len, K] quantized original rows
    float*       __restrict__ scales_out,        // [seq_len, blocks_per_row] per-block scales
    int seq_len,
    int K)                                       // d_model (columns)
{
    const int blocks_per_row = K / 32;
    const unsigned block_in_row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (block_in_row >= static_cast<unsigned>(blocks_per_row) || blockIdx.y >= static_cast<unsigned>(seq_len))
        return;
    const size_t block_index = static_cast<size_t>(blockIdx.y) * blocks_per_row + block_in_row;
    const int lane = threadIdx.x % 32;
    const size_t index = block_index * 32 + lane;

    // Load one element from the hidden state
    const float val = hidden[index];

    // Warp-cooperative max reduction for quantization scale
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask, 32));

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(abs_val);

    // Write scale (one per block, lane 0 writes)
    if (lane == 0)
        scales_out[block_index] = scale;

    // Quantize and write
    A_int8_out[index] =
        static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
                val, scale));
}
#endif


// ============================================================================
// Kernel 2: Grouped Gate+Up GEMM (all experts, single launch)
//
// Uses the GEMV-style approach (1 thread per N-column, iterate over K blocks)
// extended to M>1 via blockIdx.y for token indexing.
//
// Grid: dim3((N + TILE_N - 1) / TILE_N, max_tokens_per_expert, num_experts)
//   blockIdx.x = N-tile (64 threads per tile)
//   blockIdx.y = token within expert
//   blockIdx.z = expert index
// Block: 64 threads
//
// Each thread computes one output element: output[slot, n] = dot(A_row, B_col_n)
// ============================================================================

template <NativeQuantisedFormatPrefill FMT, bool USE_STAGED_IQ2_S = false, bool BATCH_IQ3_XXS = false>
__device__ __forceinline__ void moe_prefill_decode_groups(
    const uint8_t* __restrict__ payload,
    int32_t* packed_groups,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;

    if constexpr (Traits::is_8bit) {
        // Q8_0: 32 raw INT8 values — direct load, no decode needed
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            packed_groups[g] = *reinterpret_cast<const int32_t*>(payload + g * 4);
    }
    else if constexpr (Traits::is_iq1_grid) {
        // IQ1_S / IQ1_M: 4 grid lookups × 8 signed elements {-1,0,+1} = 32 elements
        if constexpr (FMT == NATIVE_QUANTISED_PF_IQ1_S) {
            const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                                   | (static_cast<uint16_t>(payload[5]) << 8);
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
                const uint64_t grid8 = pf_iq1_grid_lookup(idx);
                packed_groups[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(grid8));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            }
        } else { // NATIVE_QUANTISED_PF_IQ1_M
            const uint8_t qh0 = payload[4], qh1 = payload[5];
            const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
            const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
            const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
            const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);
            uint64_t g;
            g = pf_iq1_grid_lookup(idx0);
            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = pf_iq1_grid_lookup(idx1);
            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = pf_iq1_grid_lookup(idx2);
            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
            g = pf_iq1_grid_lookup(idx3);
            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(g));
            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
        }
    }
    else if constexpr (Traits::is_iq_grid) {
        // IQ3 and IQ2 grid formats — LUT lookup + sign application
        if constexpr (Traits::is_iq3_grid && !Traits::has_qh) {
            // IQ3_XXS: 8-byte qs + 4-byte signs = 12 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
            uint32_t words[8];
            if constexpr(BATCH_IQ3_XXS)
                llaminar2::rocm::loadIQ3XXSGridBatch(w0,w1,d_pf_iq3xxs_grid,words);
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const uint32_t grid4=[&] {
                    if constexpr(BATCH_IQ3_XXS)return words[g];
                    const uint32_t word = (g < 4) ? w0 : w1;
                    return pf_iq3_grid_lookup<FMT>((word >> ((g & 3) * 8)) & 0xFF);
                }();
                const uint8_t sb = static_cast<uint8_t>(w2 >> ((g / 2) * 8));
                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                packed_groups[g] = static_cast<int32_t>(pf_iq_apply_signs_4(grid4, sn));
            }
        }
        else if constexpr (Traits::is_iq3_grid) {
            // IQ3_S: 8-byte qs + 1-byte qh + 4-byte signs = 13 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w0 = pw[0], w1 = pw[1], w2 = pw[2];
            const uint8_t signs3 = payload[12];
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const uint32_t word = (g < 4) ? w0 : w1;
                int idx = (word >> ((g & 3) * 8)) & 0xFF;
                idx |= (static_cast<int>((w2 >> g) & 1) << 8);
                const uint32_t grid4 = pf_iq3_grid_lookup<FMT>(idx);
                const uint8_t sb = (g < 6)
                    ? static_cast<uint8_t>(w2 >> ((g / 2 + 1) * 8))
                    : signs3;
                const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
                packed_groups[g] = static_cast<int32_t>(pf_iq_apply_signs_4(grid4, sn));
            }
        }
        else if constexpr (Traits::is_iq2_grid && !Traits::has_qh) {
            // IQ2_XXS: 4-byte qs + 4-byte signs = 8 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w_qs = pw[0], w_signs = pw[1];
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = (w_qs >> (l * 8)) & 0xFF;
                const uint64_t grid8 =
                    pf_iq2_grid_lookup<FMT, USE_STAGED_IQ2_S>(
                        idx, staged_iq2s_grid);
                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t signs = static_cast<uint8_t>(w_signs >> (l * 8));
                packed_groups[l * 2]     = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_lo, signs & 0xF));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_hi, signs >> 4));
            }
        }
        else if constexpr (Traits::is_iq2_grid) {
            // IQ2_S / IQ2_XS: 4-byte qs + 4-byte qh_signs + 1-byte signs3 = 9 bytes
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(payload);
            const uint32_t w_qs = pw[0], w_qh_signs = pw[1];
            const uint8_t signs3 = payload[8];
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int idx = (w_qs >> (l * 8)) & 0xFF;
                if constexpr (Traits::qh_bits_per_lookup == 1)
                    idx |= (static_cast<int>((w_qh_signs >> l) & 1) << 8);
                else if constexpr (Traits::qh_bits_per_lookup == 2)
                    idx |= (static_cast<int>((w_qh_signs >> (2 * l)) & 3) << 8);
                const uint64_t grid8 =
                    pf_iq2_grid_lookup<FMT, USE_STAGED_IQ2_S>(
                        idx, staged_iq2s_grid);
                const uint32_t g_lo = static_cast<uint32_t>(grid8);
                const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t signs_byte = (l < 3)
                    ? static_cast<uint8_t>(w_qh_signs >> ((l + 1) * 8))
                    : signs3;
                packed_groups[l * 2]     = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_lo, signs_byte & 0xF));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(
                    pf_iq_apply_signs_4(g_hi, signs_byte >> 4));
            }
        }
    }
    else if constexpr (Traits::is_3bit) {
        // Q3_K: 12-byte payload [8B pre-transposed 2-bit][4B grouped hbits]
        constexpr uint32_t BIT_SPREAD = 0x00204081u;
        const uint32_t hbit_word = *reinterpret_cast<const uint32_t*>(payload + 8);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint32_t r = *reinterpret_cast<const uint32_t*>(payload + half * 4);
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t low2_4 = (r >> (g * 2)) & 0x03030303u;
                const uint32_t hb4 = (hbit_word >> ((half * 4 + g) * 4)) & 0xFu;
                const uint32_t hb = ((hb4 * BIT_SPREAD) & 0x01010101u) << 2;
                uint32_t val = low2_4 | hb;
                // Center: [0..7] → [-4..+3]
                val = (val ^ 0x80808080u) - 0x04040404u;
                val ^= 0x80808080u;
                packed_groups[half * 4 + g] = static_cast<int32_t>(val);
            }
        }
    }
    else if constexpr (Traits::is_2bit) {
        // Q2_K: 8-byte pre-transposed payload
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + half * 4);
            const int bg = half * 4;
            packed_groups[bg]     = static_cast<int32_t>(raw & 0x03030303u);
            packed_groups[bg + 1] = static_cast<int32_t>((raw >> 2) & 0x03030303u);
            packed_groups[bg + 2] = static_cast<int32_t>((raw >> 4) & 0x03030303u);
            packed_groups[bg + 3] = static_cast<int32_t>((raw >> 6) & 0x03030303u);
        }
    }
    else if constexpr (Traits::is_5bit) {
        // Q5_0 / Q5_1: 20-byte payload [16B nibbles + 4B high bits]
        const uint32_t qh_bits = *reinterpret_cast<const uint32_t*>(payload + 16);
        if constexpr (Traits::center_5bit == 0) {
            // Q5_1: unsigned [0,31]
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                     | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                packed_groups[g] = static_cast<int32_t>(lo_nibs | hb_lo);
                const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
                const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
                const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11)
                                     | ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
                packed_groups[g + 4] = static_cast<int32_t>(hi_nibs | hb_hi);
            }
        } else {
            // Q5_0: centered [-16, +15]
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
                const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11)
                                     | ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                uint32_t val_lo = (lo_nibs | hb_lo) ^ 0x80808080u;
                val_lo -= 0x10101010u;
                val_lo ^= 0x80808080u;
                packed_groups[g] = static_cast<int32_t>(val_lo);
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
    else if constexpr (Traits::is_6bit) {
        // Q6_K: 24-byte payload [16B lo nibbles + 8B hi 2-bit fields]
        const uint8_t* qh_p = payload + 16;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
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
    else if constexpr (FMT == NATIVE_QUANTISED_PF_IQ4_NL || FMT == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE) {
        /*
         * IQ4_NL / IQ4_XS verifier prefill must follow the same v_perm decode
         * expression as the M=1 MoE row-decode helper.  The removed pair-LUT
         * produced the intended int8 values, but it gave this translation unit
         * a different expression tree and shifted FP32 verifier rows by ULPs.
         */
        constexpr uint32_t CB_LO_A = 0xBFAD9881u;
        constexpr uint32_t CB_LO_B = 0xF6EADDCFu;
        constexpr uint32_t CB_HI_A = 0x26190D01u;
        constexpr uint32_t CB_HI_B = 0x71594535u;
        constexpr uint32_t MUX_IDENTITY = 0x03020100u;
        constexpr uint32_t MUX_BIT3_MASK = 0x04040404u;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);

            const uint32_t lo_nibs = raw & 0x0F0F0F0Fu;
            const uint32_t lo_sel = lo_nibs & 0x07070707u;
            const uint32_t lo_r0 = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, lo_sel);
            const uint32_t lo_r1 = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, lo_sel);
            const uint32_t lo_mux = MUX_IDENTITY | ((lo_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g] =
                static_cast<int32_t>(__builtin_amdgcn_perm(lo_r1, lo_r0, lo_mux));

            const uint32_t hi_nibs = (raw >> 4) & 0x0F0F0F0Fu;
            const uint32_t hi_sel = hi_nibs & 0x07070707u;
            const uint32_t hi_r0 = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, hi_sel);
            const uint32_t hi_r1 = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, hi_sel);
            const uint32_t hi_mux = MUX_IDENTITY | ((hi_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g + 4] =
                static_cast<int32_t>(__builtin_amdgcn_perm(hi_r1, hi_r0, hi_mux));
        }
    }
    else if constexpr (Traits::center_4bit == 8) {
        // Q4_0: nibble - 8 → [-8, +7]
        constexpr uint32_t SWAR_ADD = 0x78787878u;
        constexpr uint32_t SWAR_XOR = 0x80808080u;
        constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            packed_groups[g]     = static_cast<int32_t>(((raw & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
            packed_groups[g + 4] = static_cast<int32_t>((((raw >> 4) & NIBBLE_MASK) + SWAR_ADD) ^ SWAR_XOR);
        }
    }
    else {
        // Q4_1: unsigned nibbles [0,15]
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(payload + g * 4);
            packed_groups[g] = static_cast<int32_t>(raw & 0x0F0F0F0Fu);
            packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & 0x0F0F0F0Fu);
        }
    }
}

/**
 * @brief Accumulate one decoded NativeQuantised K block into an FP32 row result.
 *
 * Serial decode and grouped prefill must execute the same floating-point
 * expression for every row.  Keeping the expression in one helper is more than
 * code reuse: it prevents a row-tiled optimization from accidentally changing
 * the order of a dual-scale base term, an asymmetric minimum correction, or an
 * IQ1_M delta correction.  The caller may share the decoded weight payload
 * across several independent rows. Each serial partition starts from zero and
 * invokes this helper once per K block before its independently rounded result
 * enters the ordered partition reducer.
 *
 * @tparam FMT NativeQuantised codebook represented by @p desc.
 * @param desc Prepared device descriptor for one expert matrix.
 * @param linear Block-major descriptor index, `k_block * N + output_column`.
 * @param payload Pointer to the encoded weight payload for @p linear.
 * @param packed_groups Eight packed INT8 groups decoded from @p payload.
 * @param a4 Eight packed INT8 activation groups for the current row/K block.
 * @param scale_ab Per-row activation scale for the current K block.
 * @param acc Running FP32 accumulator for this row and output column.
 * @return The accumulator after exactly one serial-shaped K-block update.
 */
template <NativeQuantisedFormatPrefill FMT, bool UseExactSum=false>
__device__ __forceinline__ float moe_prefill_accumulate_decoded_block(
    const DeviceNativeQuantisedMatrixDesc& desc,
    size_t linear,
    const uint8_t* __restrict__ payload,
    const int32_t* __restrict__ packed_groups,
    const int32_t* __restrict__ a4,
    float scale_ab,
    float acc, int exactSum=0)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);

    if constexpr (Traits::is_dual_scale)
    {
        int32_t acc_lo = 0;
        int32_t acc_hi = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g)
            acc_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_lo);
#pragma unroll
        for (int g = 4; g < 8; ++g)
            acc_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_hi);

        const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
            scale_base[linear], min_base[linear]);
        const float scale_lo = weight_scales.low;
        const float scale_hi = weight_scales.high;
        if constexpr (Traits::is_dual_scale_asym)
        {
            constexpr int32_t ones = 0x01010101;
            int32_t sum_a_lo = 0;
            int32_t sum_a_hi = 0;
#pragma unroll
            for (int g = 0; g < 4; ++g)
                sum_a_lo = ::llaminar2::rocm_integer_contract::signedDot4(
                    a4[g], ones, sum_a_lo);
#pragma unroll
            for (int g = 4; g < 8; ++g)
                sum_a_hi = ::llaminar2::rocm_integer_contract::signedDot4(
                    a4[g], ones, sum_a_hi);

            const uint32_t emin_packed = emin_base[linear];
            const uint16_t emin_lo_bits =
                static_cast<uint16_t>(emin_packed);
            const uint16_t emin_hi_bits =
                static_cast<uint16_t>(emin_packed >> 16);
            const float min_lo = __half2float(
                *reinterpret_cast<const __half*>(&emin_lo_bits));
            const float min_hi = __half2float(
                *reinterpret_cast<const __half*>(&emin_hi_bits));
            acc = rocm_native_quantised_commit_corrected_dual_scale_block(
                acc,
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

        else if constexpr (FMT == NATIVE_QUANTISED_PF_IQ1_M)
        {
            constexpr float IQ1S_DELTA = 0.125f;
            constexpr int32_t ones = 0x01010101;
            const uint8_t qh0 = payload[4];
            const uint8_t qh1 = payload[5];

            const int32_t sum_a_g0 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[1], ones,
                ::llaminar2::rocm_integer_contract::signedDot4(a4[0], ones, 0));
            const int32_t sum_a_g1 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[3], ones,
                ::llaminar2::rocm_integer_contract::signedDot4(a4[2], ones, 0));
            const int32_t sum_a_g2 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[5], ones,
                ::llaminar2::rocm_integer_contract::signedDot4(a4[4], ones, 0));
            const int32_t sum_a_g3 = ::llaminar2::rocm_integer_contract::signedDot4(
                a4[7], ones,
                ::llaminar2::rocm_integer_contract::signedDot4(a4[6], ones, 0));

            const float delta0 =
                (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta1 =
                (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta2 =
                (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta3 =
                (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            acc = rocm_native_quantised_commit_iq1_m_block(
                acc,
                acc_lo,
                acc_hi,
                sum_a_g0,
                sum_a_g1,
                sum_a_g2,
                sum_a_g3,
                delta0,
                delta1,
                delta2,
                delta3,
                scale_lo,
                scale_hi,
                scale_ab);
        }
        else
        {
            acc = rocm_native_quantised_commit_dual_scale_block(
                acc,
                acc_lo,
                acc_hi,
                scale_lo,
                scale_hi,
                scale_ab);
        }
    }
    else
    {
        int32_t dot = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g)
            dot = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], dot);

        const uint16_t scale_bits = scale_base[linear];
        // Physical24 also admits an IQ4_NL gate/up partner. Resolve the
        // actual descriptor word meaning independently for each projection.
        float block_d;
        if constexpr (FMT == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE)
            block_d = desc.codebook_id == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE
                ? rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_PF_IQ4_XS_SOURCE>(
                      scale_bits, min_base[linear])
                : rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_PF_IQ4_NL>(
                      scale_bits, uint16_t{0});
        else
            block_d = rocm_native_quantised_load_single_weight_scale<FMT>(scale_bits,
                FMT == NATIVE_QUANTISED_PF_IQ3_S ? min_base[linear] : uint16_t{0});
        if constexpr (Traits::is_asymmetric)
        {
            int32_t sum_a = exactSum;
            if constexpr(!UseExactSum) {
                constexpr int32_t ones = 0x01010101;
                sum_a = 0;
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a);
            }

            const uint16_t min_bits = min_base[linear];
            const float block_m =
                __half2float(*reinterpret_cast<const __half*>(&min_bits));
            acc = rocm_native_quantised_commit_corrected_single_scale_block(
                acc,
                dot,
                block_d,
                sum_a,
                block_m,
                scale_ab);
        }
        else
        {
            acc = rocm_native_quantised_commit_single_scale_block(
                acc, dot, block_d, scale_ab);
        }
    }

    return acc;
}

/**
 * @brief Weight metadata pre-decoded once per K32 block and reused across TILE_M rows.
 *
 * Hoisting scale/min/delta loads and half-to-float conversions out of the per-row
 * accumulator loop removes up to 16x redundant global loads per K block without
 * changing a single floating-point operand passed to the serial commit helpers.
 */
template <NativeQuantisedFormatPrefill FMT>
struct MoEPrefillDecodedBlockWeights
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    float s0; // block_d (single-scale) or scale_lo (dual-scale)
    float s1; // block_m (asym) or scale_hi (dual-scale)
    float m0; // min_lo (dual-scale asym)
    float m1; // min_hi (dual-scale asym)
    uint32_t qh_packed; // qh0 | (qh1 << 8) for IQ1_M
};

template <NativeQuantisedFormatPrefill FMT>
__device__ __forceinline__ MoEPrefillDecodedBlockWeights<FMT>
moe_prefill_decode_block_weights(
    const DeviceNativeQuantisedMatrixDesc& desc,
    size_t linear,
    const uint8_t* __restrict__ payload)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    MoEPrefillDecodedBlockWeights<FMT> w{};
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);

    if constexpr (Traits::is_dual_scale)
    {
        const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
            scale_base[linear], min_base[linear]);
        w.s0 = weight_scales.low;
        w.s1 = weight_scales.high;
        if constexpr (Traits::is_dual_scale_asym)
        {
            const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);
            const uint32_t emin_packed = emin_base[linear];
            const uint16_t emin_lo_bits = static_cast<uint16_t>(emin_packed);
            const uint16_t emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
            w.m0 = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
            w.m1 = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
        }
        else if constexpr (FMT == NATIVE_QUANTISED_PF_IQ1_M)
        {
            w.qh_packed = static_cast<uint32_t>(payload[4]) |
                          (static_cast<uint32_t>(payload[5]) << 8);
        }
    }
    else
    {
        const uint16_t scale_bits = scale_base[linear];
        if constexpr (FMT == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE)
        {
            w.s0 = desc.codebook_id == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE
                ? rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_PF_IQ4_XS_SOURCE>(
                      scale_bits, min_base[linear])
                : rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_PF_IQ4_NL>(
                      scale_bits, uint16_t{0});
        }
        else
        {
            w.s0 = rocm_native_quantised_load_single_weight_scale<FMT>(
                scale_bits,
                FMT == NATIVE_QUANTISED_PF_IQ3_S ? min_base[linear] : uint16_t{0});
        }
        if constexpr (Traits::is_asymmetric)
        {
            const uint16_t min_bits = min_base[linear];
            w.s1 = __half2float(*reinterpret_cast<const __half*>(&min_bits));
        }
    }
    return w;
}

/**
 * @brief Accumulate one decoded K32 block using pre-decoded weight metadata and LDS-staged activation sums.
 *
 * Executes the exact same signedDot4 and rocm_native_quantised_commit_* arithmetic
 * as @ref moe_prefill_accumulate_decoded_block, preserving serial-M1 bit identity
 * while removing redundant per-row metadata loads and per-lane activation sums.
 */
template <NativeQuantisedFormatPrefill FMT>
__device__ __forceinline__ float moe_prefill_accumulate_preloaded_block(
    const MoEPrefillDecodedBlockWeights<FMT>& w,
    const int32_t* __restrict__ packed_groups,
    const int32_t* __restrict__ a4,
    const int32_t* __restrict__ staged_sum_a_row,
    float scale_ab,
    float acc)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    if constexpr (Traits::is_dual_scale)
    {
        int32_t acc_lo = 0;
        int32_t acc_hi = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g)
            acc_lo = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_lo);
#pragma unroll
        for (int g = 4; g < 8; ++g)
            acc_hi = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], acc_hi);

        if constexpr (Traits::is_dual_scale_asym)
        {
            acc = rocm_native_quantised_commit_corrected_dual_scale_block(
                acc,
                acc_lo,
                acc_hi,
                w.s0,
                w.s1,
                staged_sum_a_row[0],
                staged_sum_a_row[1],
                w.m0,
                w.m1,
                scale_ab);
        }
        else if constexpr (FMT == NATIVE_QUANTISED_PF_IQ1_M)
        {
            constexpr float IQ1S_DELTA = 0.125f;
            const uint8_t qh0 = static_cast<uint8_t>(w.qh_packed & 0xFFu);
            const uint8_t qh1 = static_cast<uint8_t>((w.qh_packed >> 8) & 0xFFu);
            const float delta0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            const float delta3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            acc = rocm_native_quantised_commit_iq1_m_block(
                acc,
                acc_lo,
                acc_hi,
                staged_sum_a_row[0],
                staged_sum_a_row[1],
                staged_sum_a_row[2],
                staged_sum_a_row[3],
                delta0,
                delta1,
                delta2,
                delta3,
                w.s0,
                w.s1,
                scale_ab);
        }
        else
        {
            acc = rocm_native_quantised_commit_dual_scale_block(
                acc,
                acc_lo,
                acc_hi,
                w.s0,
                w.s1,
                scale_ab);
        }
    }
    else
    {
        int32_t dot = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g)
            dot = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], packed_groups[g], dot);

        if constexpr (Traits::is_asymmetric)
        {
            acc = rocm_native_quantised_commit_corrected_single_scale_block(
                acc,
                dot,
                w.s0,
                staged_sum_a_row[0],
                w.s1,
                scale_ab);
        }
        else
        {
            acc = rocm_native_quantised_commit_single_scale_block(
                acc, dot, w.s0, scale_ab);
        }
    }
    return acc;
}

/**
 * @brief Compute one NativeQuantised dot product over a bounded K-block range.
 *
 * This is the canonical single-row expression used by grouped prefill and its
 * serial-row oracle.  Long-prefill kernels below share decoded weights across
 * rows but deliberately call the same per-block accumulator so optimization
 * cannot relax byte equality.
 */
template <NativeQuantisedFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ float moe_prefill_dot_desc_range(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int b_start,
    int b_end,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    if (!slot_A ||
        !slot_scales ||
        n < 0 ||
        n >= N ||
        desc.n != N ||
        desc.blocks_per_row == 0 ||
        b_start < 0 ||
        b_end > static_cast<int>(desc.blocks_per_row) ||
        !prefill_native_quantised_desc_shape_ok<FMT>(
            desc, N, static_cast<int>(desc.blocks_per_row) * 32))
    {
        return 0.0f;
    }

    const uint8_t* payload_base = desc.payload;
    float acc = 0.0f;

    for (int b = b_start; b < b_end; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) +
                              static_cast<size_t>(n);
        int32_t packed_groups[8];
        constexpr int PB = Traits::payload_bytes;
        const uint8_t* payload = payload_base + linear * PB;
        moe_prefill_decode_groups<FMT, USE_STAGED_IQ2_S>(
            payload, packed_groups, staged_iq2s_grid);

        const int32_t* a4 = reinterpret_cast<const int32_t*>(slot_A + b * 32);
        acc = moe_prefill_accumulate_decoded_block<FMT>(
            desc, linear, payload, packed_groups, a4, slot_scales[b], acc);
    }

    return acc;
}

/**
 * @brief Reproduce serial M=1 split-K partials and their ordered reduction.
 *
 * This function keeps all partials local to the owning output lane. It does not
 * replay rows or materialize a global split-K workspace: each partition runs
 * the canonical block accumulator from zero, and the lane adds completed
 * partials in increasing partition order with the same explicit RN operation
 * used by serial decode's reducer kernel.
 *
 * @tparam FMT NativeQuantised codebook represented by @p desc.
 * @tparam USE_STAGED_IQ2_S Whether IQ2_S codebook reads use an LDS copy.
 * @param desc Prepared matrix descriptor.
 * @param n Output column owned by this lane.
 * @param slot_A Quantized activation row.
 * @param slot_scales Per-block activation scales.
 * @param N Matrix output width.
 * @param k_partitions Exact effective serial-M1 partition count.
 * @param staged_iq2s_grid Optional LDS-resident IQ2_S codebook.
 * @return Byte-equivalent serial-M1 result for this output element.
 */
template <NativeQuantisedFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ float moe_prefill_dot_desc_serial_m1(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int k_partitions,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    const int blocks_per_row = static_cast<int>(desc.blocks_per_row);
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return 0.0f;
    }

    if (k_partitions == 1)
    {
        return moe_prefill_dot_desc_range<FMT, USE_STAGED_IQ2_S>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            0,
            blocks_per_row,
            staged_iq2s_grid);
    }

    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    float total = 0.0f;
#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        const float partial = begin < end
                                  ? moe_prefill_dot_desc_range<
                                        FMT, USE_STAGED_IQ2_S>(
                                        desc,
                                        n,
                                        slot_A,
                                        slot_scales,
                                        N,
                                        begin,
                                        end,
                                        staged_iq2s_grid)
                                  : 0.0f;
        total = rocm_native_quantised_accumulate_rn(total, partial);
    }
    return total;
}

/**
 * @brief Integer and scale terms for one exact dual-scale K-block contribution.
 *
 * These are the operands consumed by
 * @ref rocm_native_quantised_commit_dual_scale_block. Keeping the FP32 accumulator
 * out of this structure lets several lanes prepare independent K blocks while
 * one owner lane still commits them in serial-decode order.
 */
struct MoEPrefillDualScaleBlockTerms
{
    int32_t low_dot = 0;       ///< INT32 dot for K elements 0..15.
    int32_t high_dot = 0;      ///< INT32 dot for K elements 16..31.
    float low_scale = 0.0f;    ///< Source-resolved low-half FP32 weight scale.
    float high_scale = 0.0f;   ///< Source-resolved high-half FP32 weight scale.
    float activation_scale = 0.0f; ///< Blockwise activation scale.
};

/**
 * @brief Prepare one correction-free dual-scale block without changing FP32 state.
 *
 * The payload decode and both `sdot4` chains are identical to
 * @ref moe_prefill_accumulate_decoded_block. Only the independent integer work
 * is separated from the serial FP32 commit so neighboring lanes can expose
 * enough wavefronts to occupy gfx906 for M=1..4 verifier buckets.
 *
 * @tparam FMT Correction-free dual-scale NativeQuantised format.
 * @tparam USE_STAGED_IQ2_S Whether IQ2_S codebook reads use the LDS copy.
 * @param desc Prepared matrix descriptor.
 * @param n Output column owned by the cooperating lane subgroup.
 * @param slot_A Quantized activation row, optionally staged in LDS.
 * @param slot_scales Activation block scales, optionally staged in LDS.
 * @param N Matrix output width.
 * @param block_index Increasing-K block index to prepare.
 * @return Exact integer dots and scale operands for the block.
 */
template <NativeQuantisedFormatPrefill FMT, bool USE_STAGED_IQ2_S = false>
__device__ __forceinline__ MoEPrefillDualScaleBlockTerms
moe_prefill_prepare_dual_scale_block_terms(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int block_index,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    static_assert(
        Traits::is_dual_scale &&
            !Traits::is_dual_scale_asym &&
            FMT != NATIVE_QUANTISED_PF_IQ1_M,
        "subgroup preparation requires a correction-free dual-scale format");
    static_assert(!USE_STAGED_IQ2_S || FMT == NATIVE_QUANTISED_PF_IQ2_S,
                  "only IQ2_S may consume the staged IQ2_S codebook");

    const size_t linear =
        static_cast<size_t>(block_index) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const uint8_t* payload =
        desc.payload + linear * static_cast<size_t>(Traits::payload_bytes);
    int32_t packed_groups[8];
    moe_prefill_decode_groups<FMT, USE_STAGED_IQ2_S>(
        payload, packed_groups, staged_iq2s_grid);

    const int32_t* a4 = reinterpret_cast<const int32_t*>(
        slot_A + static_cast<size_t>(block_index) * 32u);
    MoEPrefillDualScaleBlockTerms terms{};
#pragma unroll
    for (int group = 0; group < 4; ++group)
    {
        terms.low_dot = ::llaminar2::rocm_integer_contract::signedDot4(
            a4[group], packed_groups[group], terms.low_dot);
    }
#pragma unroll
    for (int group = 4; group < 8; ++group)
    {
        terms.high_dot = ::llaminar2::rocm_integer_contract::signedDot4(
            a4[group], packed_groups[group], terms.high_dot);
    }

    const uint16_t* low_scales =
        static_cast<const uint16_t*>(desc.scales);
    const uint16_t* high_scales =
        static_cast<const uint16_t*>(desc.mins);
    const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
        low_scales[linear], high_scales[linear]);
    terms.low_scale = weight_scales.low;
    terms.high_scale = weight_scales.high;
    terms.activation_scale = slot_scales[block_index];
    return terms;
}

/**
 * @brief Compute one dual-scale output with parallel preparation and serial commits.
 *
 * `K_LANES` adjacent lanes own one output column. During each iteration they
 * prepare consecutive K blocks independently. Every lane then participates in
 * bit-preserving shuffles, while subgroup lane zero calls the canonical RN
 * commit helper for source lanes 0..K_LANES-1. Consequently the observable
 * FP32 expression remains block 0, block 1, ... exactly as in serial decode;
 * this is parallel work preparation, not a floating-point reduction.
 *
 * @tparam FMT Correction-free dual-scale NativeQuantised format.
 * @tparam USE_STAGED_IQ2_S Whether IQ2_S codebook reads use the LDS copy.
 * @tparam K_LANES Number of adjacent lanes cooperating on one output.
 * @param desc Prepared matrix descriptor.
 * @param n Output column shared by this subgroup.
 * @param slot_A Quantized activation row, normally staged in LDS.
 * @param slot_scales Activation block scales, normally staged in LDS.
 * @param N Matrix output width.
 * @param blocks_per_row Number of 32-element K blocks.
 * @param k_partitions Exact effective serial-M1 partition count.
 * @return Exact output on subgroup lane zero; other lanes return an unused zero.
 */
template <
    NativeQuantisedFormatPrefill FMT,
    bool USE_STAGED_IQ2_S,
    int K_LANES>
__device__ __forceinline__ float moe_prefill_dual_scale_subgroup_dot(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int blocks_per_row,
    int k_partitions,
    const uint64_t* __restrict__ staged_iq2s_grid = nullptr)
{
    static_assert(K_LANES == 2 || K_LANES == 4,
                  "dual-scale verifier subgroup width must be two or four lanes");
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    static_assert(
        Traits::is_dual_scale &&
            !Traits::is_dual_scale_asym &&
            FMT != NATIVE_QUANTISED_PF_IQ1_M,
        "subgroup dot requires a correction-free dual-scale format");
    const int subgroup_lane = static_cast<int>(threadIdx.x) & (K_LANES - 1);
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return 0.0f;
    }

    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;
    float total = 0.0f;
#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        float partial = 0.0f;

        for (int first_block = begin;
             first_block < end;
             first_block += K_LANES)
        {
            const int block_index = first_block + subgroup_lane;
            MoEPrefillDualScaleBlockTerms local{};
            if (block_index < end)
            {
                local = moe_prefill_prepare_dual_scale_block_terms<
                    FMT, USE_STAGED_IQ2_S>(
                    desc,
                    n,
                    slot_A,
                    slot_scales,
                    N,
                    block_index,
                    staged_iq2s_grid);
            }

#pragma unroll
            for (int source_lane = 0; source_lane < K_LANES; ++source_lane)
            {
                const int32_t low_dot =
                    __shfl(local.low_dot, source_lane, K_LANES);
                const int32_t high_dot =
                    __shfl(local.high_dot, source_lane, K_LANES);
                const float low_scale =
                    __shfl(local.low_scale, source_lane, K_LANES);
                const float high_scale =
                    __shfl(local.high_scale, source_lane, K_LANES);
                const float activation_scale =
                    __shfl(local.activation_scale, source_lane, K_LANES);

                if (subgroup_lane == 0 &&
                    first_block + source_lane < end)
                {
                    partial = rocm_native_quantised_commit_dual_scale_block(
                        partial,
                        low_dot,
                        high_dot,
                        low_scale,
                        high_scale,
                        activation_scale);
                }
            }
        }

        if (subgroup_lane == 0)
        {
            total = k_partitions == 1
                        ? partial
                        : rocm_native_quantised_accumulate_rn(total, partial);
        }
    }

    return subgroup_lane == 0 ? total : 0.0f;
}

namespace
{
    /** Number of low bits reserved for an expert id in a packed work item. */
    constexpr uint32_t kMoEPrefillExpertBits =
        llaminar2::moe_grouped_directory::kExpertBits;
    /**
     * @brief Maximum descriptor-table width represented by the work directory.
     *
     * Flash Next has 512 routed experts and one separate shared expert.  A
     * verifier may economically publish those eleven selected routes through one
     * descriptor table, so the directory must admit expert id 512 rather than
     * treating 512 as a hard table-size limit. Ten id bits represent ids
     * 0..1023, hence 1024 table entries without enlarging each packed work item.
     */
    constexpr int kMoEPrefillMaxExperts =
        static_cast<int>(uint32_t{1} << kMoEPrefillExpertBits);
    /** Maximum expert id representable by the compact device work directory. */
    constexpr uint32_t kMoEPrefillExpertMask =
        (uint32_t{1} << kMoEPrefillExpertBits) - 1u;
    /** Trainer/runtime tile-row marker for device-adaptive TM12/TM16 work. */
    constexpr int kMoEPrefillAdaptiveTileRows = 20;
    /** Number of count words preceding adaptive TM12 and TM16 spans. */
    constexpr int kMoEPrefillAdaptiveDirectoryHeaderWords = 2;
    /** Header word publishing the live compact TM12 span length. */
    constexpr int kMoEPrefillAdaptiveTile12CountWord = 0;
    /** Header word publishing the live compact TM16 span length. */
    constexpr int kMoEPrefillAdaptiveTile16CountWord = 1;
    /** Sentinel written to unused graph-captured work-directory entries. */
    constexpr uint32_t kMoEPrefillUnusedTile = UINT32_MAX;
}

/**
 * @brief Build a compact device-owned directory of populated expert row tiles.
 *
 * A graph-captured prefill launch cannot read the current maximum expert count
 * on the host.  Launching a rectangular
 * `num_experts * ceil(seq_len / TILE_M)` grid would preserve device ownership,
 * but most workgroups would immediately exit and long-prefill throughput would
 * collapse. This one-block planner performs a parallel inclusive scan over the
 * already device-resident grouping counts and writes only populated
 * `(expert, first_row)` tiles into @p directory.
 * The following projection kernels launch against a conservative fixed capacity
 * and skip the `UINT32_MAX` tail.
 *
 * Each directory entry uses nine low bits for the expert id and the remaining
 * bits for the first row local to that expert. Nine bits support descriptor
 * tables with up to 512 entries, including a 256-routed-plus-one-shared fused
 * verifier table, while still supporting millions of grouped rows. Planning is
 * deterministic, contains no host synchronization, and is safe to capture and
 * replay as part of the full GPU graph.
 *
 * @tparam TILE_M Fixed rows per workgroup, or zero for the sealed wide row plan.
 * @param admission The exact live-family publication also borrowed by consumers.
 * @param row_plan Complete immutable row policy; the wide consumer borrows the same value.
 */
template <int TILE_M>
__global__ __launch_bounds__(kMoEPrefillMaxExperts, 1)
void moe_grouped_build_prefill_tile_directory_kernel(
    const int* __restrict__ group_counts,
    uint32_t* __restrict__ directory,
    int num_experts,
    int max_directory_tiles,
    ROCmMoEGroupedRouteAdmission admission,
    MoEGroupedRowPlan row_plan)
{
    static_assert(TILE_M == 0 || TILE_M == 4 || TILE_M == 8 ||
                  TILE_M == 12 || TILE_M == 16,
                  "grouped-prefill sweep supports row tiles 4/8/12/16");
    // A stale directory is deliberately unobservable to the inactive tiled
    // family: every consumer checks this same uniform predicate first. Avoid
    // both the capacity-wide tail stores and the scan's eighteen barriers.
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
        return;
    if constexpr (TILE_M == 0)
        if (!row_plan.isWide()) __builtin_trap();
    const int tid = threadIdx.x;

    if (num_experts <= 0 || num_experts > kMoEPrefillMaxExperts ||
        max_directory_tiles <= 0)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    for (int tile = tid; tile < max_directory_tiles; tile += blockDim.x)
        directory[tile] = kMoEPrefillUnusedTile;

    __shared__ int inclusive_tile_counts[kMoEPrefillMaxExperts];
    int count = tid < num_experts ? group_counts[tid] : 0;
    if (count < 0 || static_cast<uint32_t>(count) > llaminar2::moe_grouped_directory::kMaximumRows)
    {
        if (tid < num_experts)
            moe_prefill_fail_fast_invalid_grouped_index();
        count = 0;
    }
    const int row_step = TILE_M ? TILE_M : row_plan.rows(count);
    const int expert_tiles = count > 0 ? (count + row_step - 1) / row_step : 0;
    inclusive_tile_counts[tid] = expert_tiles;
    __syncthreads();

    // Hillis-Steele is deliberately used for this tiny 256-element directory:
    // its regular barriers are graph-capture friendly and much cheaper than a
    // lane-zero serial scan on gfx906.
    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1)
    {
        const int addend = tid >= stride
                               ? inclusive_tile_counts[tid - stride]
                               : 0;
        __syncthreads();
        inclusive_tile_counts[tid] += addend;
        __syncthreads();
    }

    const int total_tiles = inclusive_tile_counts[blockDim.x - 1];
    if (total_tiles > max_directory_tiles)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    if (tid < num_experts)
    {
        const int first_tile = tid == 0 ? 0 : inclusive_tile_counts[tid - 1];
        for (int tile = 0; tile < expert_tiles; ++tile)
        {
            const uint32_t first_row = static_cast<uint32_t>(tile * row_step);
            directory[first_tile + tile] =
                (first_row << kMoEPrefillExpertBits) |
                static_cast<uint32_t>(tid);
        }
    }
}


/**
 * @brief Choose TM12 or TM16 from one expert's live grouped-row count.
 *
 * Gate/up makes one projection-wide choice. TM16 is used only when every active
 * expert contains exactly 16 rows, where it removes one third of the workgroups
 * without mixing a lower-occupancy specialization into the replay. Every other
 * live distribution uses TM12. This distinction matters on gfx906: the TM12
 * gate/up kernel occupies four blocks per CU at 64 VGPRs, whereas TM16 occupies
 * three at 72 VGPRs. Launching both for one skewed distribution costs more than
 * the handful of TM16 tiles save.
 *
 * Down projection is less register-bound, so it chooses the tile with fewer
 * padded row evaluations; ties favor TM16's greater weight reuse. These choices
 * change only grouping of independent rows. Each row still traverses K and
 * reduces partials in the serial-M1 order implemented by the fixed
 * specializations.
 *
 * @param count Number of live rows owned by one expert.
 * @param gateup_all_active_counts_are_16 Device-reduced projection-wide gate/up
 *        predicate. It is ignored for the down projection.
 * @return The exact fixed specialization that owns this expert's rows.
 */
template <MoEPrefillProjectionRole ROLE>
__device__ __forceinline__ int moe_prefill_adaptive_tile_rows(
    int count,
    bool gateup_all_active_counts_are_16)
{
    if constexpr (ROLE == MoEPrefillProjectionRole::GateUp)
        return gateup_all_active_counts_are_16 ? 16 : 12;

    const int padded_12 = ((count + 11) / 12) * 12;
    const int padded_16 = ((count + 15) / 16) * 16;
    return padded_12 < padded_16 ? 12 : 16;
}

/**
 * @brief Build compact TM12 and TM16 spans entirely from device row counts.
 *
 * One planner emits a two-word device header followed by independently compact
 * TM12 and TM16 spans. Each projection specialization receives its own span and
 * live count, so it never launches the other specialization's conservative
 * capacity merely to reject tagged entries. Both launches remain permanently
 * present in the captured graph; replay therefore needs neither a conditional
 * graph node nor host-authored route state.
 *
 * @tparam ROLE Gate/up or down projection selection policy.
 * @param admission Shared device-owned family selection, never a host row count.
 */
template <MoEPrefillProjectionRole ROLE>
__global__ __launch_bounds__(kMoEPrefillMaxExperts, 1)
void moe_grouped_build_adaptive_prefill_tile_directory_kernel(
    const int* __restrict__ group_counts,
    uint32_t* __restrict__ directory,
    int num_experts,
    int max_tile12_tiles,
    int max_tile16_tiles,
    ROCmMoEGroupedRouteAdmission admission)
{
    // Both adaptive spans belong solely to expert-tiled work. Do not rewrite
    // their headers or visit group counts when route-owned work is selected.
    // On the next tiled replay this producer reconstructs both live spans
    // before the unchanged stream-ordered projection consumers can read them.
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
        return;
    const int tid = threadIdx.x;
    if (num_experts <= 0 || num_experts > kMoEPrefillMaxExperts ||
        max_tile12_tiles <= 0 || max_tile16_tiles <= 0)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    __shared__ int inclusive_tile12_counts[kMoEPrefillMaxExperts];
    __shared__ int inclusive_tile16_counts[kMoEPrefillMaxExperts];
    int count = tid < num_experts ? group_counts[tid] : 0;
    if (count < 0 || static_cast<uint32_t>(count) > llaminar2::moe_grouped_directory::kMaximumRows)
    {
        if (tid < num_experts)
            moe_prefill_fail_fast_invalid_grouped_index();
        count = 0;
    }

    bool gateup_all_active_counts_are_16 = false;
    if constexpr (ROLE == MoEPrefillProjectionRole::GateUp)
    {
        /*
         * Reuse the TM12 prefix array for an exact projection-wide predicate
         * before populating the real directory counts. The scan is integer-only
         * and has a fixed synchronization tree, so no route-dependent atomic or
         * host observation enters the captured transaction.
         */
        inclusive_tile12_counts[tid] = count > 0 && count != 16 ? 1 : 0;
        __syncthreads();
        for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1)
        {
            const int addend = tid >= stride
                                   ? inclusive_tile12_counts[tid - stride]
                                   : 0;
            __syncthreads();
            inclusive_tile12_counts[tid] += addend;
            __syncthreads();
        }
        gateup_all_active_counts_are_16 =
            inclusive_tile12_counts[blockDim.x - 1] == 0;
    }

    const int tile_rows = count > 0
                              ? moe_prefill_adaptive_tile_rows<ROLE>(
                                    count,
                                    gateup_all_active_counts_are_16)
                              : 12;
    const int expert_tiles =
        count > 0 ? (count + tile_rows - 1) / tile_rows : 0;
    inclusive_tile12_counts[tid] = tile_rows == 12 ? expert_tiles : 0;
    inclusive_tile16_counts[tid] = tile_rows == 16 ? expert_tiles : 0;
    __syncthreads();

    for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1)
    {
        const int tile12_addend = tid >= stride
                                      ? inclusive_tile12_counts[tid - stride]
                                      : 0;
        const int tile16_addend = tid >= stride
                                      ? inclusive_tile16_counts[tid - stride]
                                      : 0;
        __syncthreads();
        inclusive_tile12_counts[tid] += tile12_addend;
        inclusive_tile16_counts[tid] += tile16_addend;
        __syncthreads();
    }

    const int total_tile12 =
        inclusive_tile12_counts[blockDim.x - 1];
    const int total_tile16 =
        inclusive_tile16_counts[blockDim.x - 1];
    if (total_tile12 > max_tile12_tiles || total_tile16 > max_tile16_tiles)
    {
        if (tid == 0)
            moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    if (tid == 0)
    {
        directory[kMoEPrefillAdaptiveTile12CountWord] =
            static_cast<uint32_t>(total_tile12);
        directory[kMoEPrefillAdaptiveTile16CountWord] =
            static_cast<uint32_t>(total_tile16);
    }

    if (tid < num_experts && count > 0)
    {
        const bool owns_tile16 = tile_rows == 16;
        const int first_tile =
            tid == 0
                ? 0
                : (owns_tile16
                       ? inclusive_tile16_counts[tid - 1]
                       : inclusive_tile12_counts[tid - 1]);
        uint32_t* const owned_directory =
            directory + kMoEPrefillAdaptiveDirectoryHeaderWords +
            (owns_tile16 ? max_tile12_tiles : 0);
        for (int tile = 0; tile < expert_tiles; ++tile)
        {
            const uint32_t first_row =
                static_cast<uint32_t>(tile * tile_rows);
            owned_directory[first_tile + tile] =
                (first_row << kMoEPrefillExpertBits) |
                static_cast<uint32_t>(tid);
        }
    }
}

/**
 * @brief Visit a live row prefix with constant accumulator indices.
 *
 * Nest the guards so the first absent row ends the visit, while template
 * expansion makes every array index a constant. A runtime loop break defeats
 * compiler unrolling; independent per-row guards evaluate all absent rows.
 * This recursion is fully inlined: it adds no device stack or runtime loop.
 *
 * @tparam Row Compile-time first row to visit.
 * @tparam Count Compile-time accumulator-array extent.
 * @tparam Function Device callable accepting an integral constant row index.
 * @param live_rows Number of populated rows in this tile.
 * @param visit Operation on exactly one populated row; called in row order.
 */
template <int Row, int Count, class Function>
__device__ __forceinline__ void moe_prefill_visit_live_rows(
    int live_rows, const Function& visit)
{
    static_assert(Row >= 0 && Row < Count);
    if (Row < live_rows)
    {
        visit(std::integral_constant<int, Row>{});
        if constexpr (Row + 1 < Count)
            moe_prefill_visit_live_rows<Row + 1, Count>(live_rows, visit);
    }
}

/**
 * @brief Batch-invariant expert-tiled gate/up projection for long prefill.
 *
 * One workgroup owns one populated expert tile, one 64-column output tile, and
 * one projection role (`gate` or `up`).  The expert weight payload is decoded
 * once per K block and reused across `TILE_M` independent routed rows.  Each row
 * still visits K blocks in increasing order and updates its own FP32 accumulator
 * through @ref moe_prefill_accumulate_decoded_block, so the optimization changes
 * data reuse without changing a row's serial-decode arithmetic.
 *
 * The activation block is staged in a very small LDS tile.  This prevents all 64
 * output-column lanes from redundantly loading the same routed activation while
 * retaining four-wave occupancy on gfx906.  Gate and up use separate workgroups
 * in one launch (`blockIdx.z`) to keep their accumulator lifetimes disjoint and
 * avoid register pressure from halving MI50 occupancy.
 *
 * @tparam FMT NativeQuantised codebook for experts handled by this launch.
 * @tparam TILE_M Trained expert-row tile (4, 8, 12, or 16 rows).
 * @tparam TILE_N Trained output-column tile (64, 128, or 256 lanes).
 * @param A_int8 Quantized original token or route-slot activation rows.
 * @param scales_A_blockwise FP32 scale for each activation K32 block.
 * @param group_token_indices Device-owned original row identities.
 * @param group_counts Complete live row count for each expert.
 * @param group_offsets Exclusive offsets into grouped row banks.
 * @param work_directory Current fixed or adaptive packed expert tiles.
 * @param gate_descs Prepared gate descriptor table.
 * @param up_descs Prepared up descriptor table.
 * @param gate_out Unweighted grouped gate projection bank.
 * @param up_out Unweighted grouped up projection bank.
 * @param max_directory_tiles Captured directory capacity.
 * @param seq_len Original captured token capacity.
 * @param top_k Route slots per token.
 * @param grouped_indices_are_route_slots Whether indices address route slots.
 * @param input_is_original_rows Whether activation rows retain token indexing.
 * @param N Prepared projection output columns.
 * @param K Projection contraction width.
 * @param k_partition_policy Original source-owned arithmetic partition policy.
 * @param num_experts Descriptor and group-plan cardinality.
 * @param adaptive_tile_count Optional device-owned current directory extent.
 * @param admission Shared device selection of the ExpertTiled route family.
 */
/**
 * @brief Lightweight intra-workgroup barrier for ping-pong LDS staging.
 *
 * On Wave64 AMD GPUs (`gfx906`/`gfx1100` under `-mwavefrontsize64`), a
 * 64-thread workgroup executes as one lockstep wavefront. Waiting only on
 * pending LDS writes avoids hardware `s_barrier` overhead while preserving
 * exact cross-lane LDS visibility. Multi-wave workgroups use `__syncthreads()`.
 */
template <int TILE_N>
__device__ __forceinline__ void moe_prefill_block_barrier()
{
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
    if constexpr (TILE_N == 64)
    {
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
    else
    {
        __syncthreads();
    }
#else
    __syncthreads();
#endif
}

template <NativeQuantisedFormatPrefill FMT, int TILE_M, int TILE_N>
__global__ __launch_bounds__(TILE_N, 2)
void moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const int* __restrict__ group_token_indices,
    const int* __restrict__ group_counts,
    const int* __restrict__ group_offsets,
    const uint32_t* __restrict__ work_directory,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    int max_directory_tiles,
    int seq_len,
    int top_k,
    int grouped_indices_are_route_slots,
    int input_is_original_rows,
    int N,
    int K,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    int num_experts,
    const uint32_t* __restrict__ adaptive_tile_count,
    ROCmMoEGroupedRouteAdmission admission)
{
    // Uniform device-owned admission precedes all barriers and weight reads.
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
        return;
    static_assert(TILE_M == 4 || TILE_M == 8 ||
                  TILE_M == 12 || TILE_M == 16,
                  "grouped-prefill gate/up supports row tiles 4/8/12/16");
    static_assert(TILE_N == 64 || TILE_N == 128 || TILE_N == 256,
                  "grouped-prefill gate/up supports column tiles 64/128/256");
    const int n = static_cast<int>(blockIdx.x) * TILE_N + threadIdx.x;
    const int directory_index = static_cast<int>(blockIdx.y);
    const int projection_role = static_cast<int>(blockIdx.z);
    if (directory_index >= max_directory_tiles || projection_role > 1)
        return;
    if (adaptive_tile_count &&
        directory_index >= static_cast<int>(*adaptive_tile_count))
    {
        return;
    }

    uint32_t packed_tile = work_directory[directory_index];
    if (packed_tile == kMoEPrefillUnusedTile)
        return;

    const int expert_id =
        static_cast<int>(packed_tile & kMoEPrefillExpertMask);
    const int first_row =
        static_cast<int>(packed_tile >> kMoEPrefillExpertBits);
    if (expert_id < 0 || expert_id >= num_experts)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    const int count = group_counts[expert_id];
    const int grouped_offset = group_offsets[expert_id];
    if (first_row < 0 || first_row >= count || grouped_offset < 0)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const int rows_in_tile = min(TILE_M, count - first_row);
    const int first_grouped_slot = grouped_offset + first_row;

    const DeviceNativeQuantisedMatrixDesc desc =
        projection_role == 0 ? gate_descs[expert_id] : up_descs[expert_id];
    if (!prefill_native_quantised_desc_shape_ok<FMT>(desc, N, K))
        return;

    const int k_partitions = k_partition_policy.forDescriptor(desc);

    float* projection_out = projection_role == 0 ? gate_out : up_out;
    const int blocks_per_row = K / 32;
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const uint8_t* payload_base = desc.payload;
    /*
     * A trained TILE_N may be wider than N or leave a partial final column
     * tile. Those lanes must still cross every LDS barrier because activation
     * staging is distributed across the complete workgroup. They simply do no
     * descriptor or output memory work. Returning them above would make the
     * workgroup barrier protocol divergent and corrupt later row tiles.
     */
    const bool owns_column = n < N;

    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    constexpr int kSumAStride =
        (FMT == NATIVE_QUANTISED_PF_IQ1_M) ? 4 :
        Traits::is_dual_scale_asym ? 2 :
        Traits::is_asymmetric ? 1 : 0;

    __shared__ __align__(16) int32_t staged_A[2][TILE_M * 8];
    __shared__ float staged_scales[2][TILE_M];
    __shared__ int32_t staged_sum_a[2][kSumAStride > 0 ? TILE_M * kSumAStride : 1];
    __shared__ int source_rows[TILE_M];

    if (threadIdx.x < static_cast<unsigned int>(rows_in_tile))
    {
        const int grouped_slot = first_grouped_slot + static_cast<int>(threadIdx.x);
        const int source_index = input_is_original_rows
                                     ? group_token_indices[grouped_slot]
                                     : grouped_slot;
        const int input_row = input_is_original_rows && grouped_indices_are_route_slots
                                  ? source_index / top_k
                                  : source_index;
        if (input_row < 0 ||
            input_row >= (input_is_original_rows ? seq_len : INT32_MAX))
        {
            moe_prefill_fail_fast_invalid_grouped_index();
            return;
        }
        source_rows[threadIdx.x] = input_row;
    }
    moe_prefill_block_barrier<TILE_N>();
    if (threadIdx.x >= static_cast<unsigned int>(rows_in_tile) &&
        threadIdx.x < static_cast<unsigned int>(TILE_M))
    {
        source_rows[threadIdx.x] = source_rows[0];
    }

    const int u_rows = __builtin_amdgcn_readfirstlane(rows_in_tile);

    // Zero-init inactive tail slots once so 2-row groups run branchlessly.
    if (static_cast<int>(threadIdx.x) >= u_rows * 8 &&
        static_cast<int>(threadIdx.x) < TILE_M * 8)
    {
        staged_A[0][threadIdx.x] = 0;
        staged_A[1][threadIdx.x] = 0;
    }
    if (static_cast<int>(threadIdx.x) >= u_rows &&
        static_cast<int>(threadIdx.x) < TILE_M)
    {
        staged_scales[0][threadIdx.x] = 0.0f;
        staged_scales[1][threadIdx.x] = 0.0f;
        if constexpr (kSumAStride > 0)
        {
#pragma unroll
            for (int s = 0; s < kSumAStride; ++s)
            {
                staged_sum_a[0][static_cast<int>(threadIdx.x) * kSumAStride + s] = 0;
                staged_sum_a[1][static_cast<int>(threadIdx.x) * kSumAStride + s] = 0;
            }
        }
    }

    const auto stage_k_block = [&](int block_idx, int buf) __attribute__((always_inline))
    {
        if (static_cast<int>(threadIdx.x) < u_rows)
        {
            const int r = static_cast<int>(threadIdx.x);
            const int input_row = source_rows[r];
            const int4* in16 = reinterpret_cast<const int4*>(
                A_int8 + static_cast<size_t>(input_row) * static_cast<size_t>(K) +
                static_cast<size_t>(block_idx) * 32u);
            const int4 lo = in16[0];
            const int4 hi = in16[1];
            int4* dst16 = reinterpret_cast<int4*>(staged_A[buf] + r * 8);
            dst16[0] = lo;
            dst16[1] = hi;
            staged_scales[buf][r] = scales_A_blockwise[
                static_cast<size_t>(input_row) * static_cast<size_t>(blocks_per_row) +
                static_cast<size_t>(block_idx)];
            if constexpr (kSumAStride > 0)
            {
                constexpr int32_t ones = 0x01010101;
                const int32_t s0 = ::llaminar2::rocm_integer_contract::signedDot4(
                    lo.y, ones, ::llaminar2::rocm_integer_contract::signedDot4(lo.x, ones, 0));
                const int32_t s1 = ::llaminar2::rocm_integer_contract::signedDot4(
                    lo.w, ones, ::llaminar2::rocm_integer_contract::signedDot4(lo.z, ones, 0));
                const int32_t s2 = ::llaminar2::rocm_integer_contract::signedDot4(
                    hi.y, ones, ::llaminar2::rocm_integer_contract::signedDot4(hi.x, ones, 0));
                const int32_t s3 = ::llaminar2::rocm_integer_contract::signedDot4(
                    hi.w, ones, ::llaminar2::rocm_integer_contract::signedDot4(hi.z, ones, 0));
                const int sb = r * kSumAStride;
                if constexpr (FMT == NATIVE_QUANTISED_PF_IQ1_M)
                {
                    staged_sum_a[buf][sb + 0] = s0;
                    staged_sum_a[buf][sb + 1] = s1;
                    staged_sum_a[buf][sb + 2] = s2;
                    staged_sum_a[buf][sb + 3] = s3;
                }
                else if constexpr (Traits::is_dual_scale_asym)
                {
                    staged_sum_a[buf][sb + 0] = s0 + s1;
                    staged_sum_a[buf][sb + 1] = s2 + s3;
                }
                else
                {
                    staged_sum_a[buf][sb] = (s0 + s1) + (s2 + s3);
                }
            }
        }
    };

    float totals[TILE_M] = {};
    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;

    if (blocks_per_row > 0)
    {
        stage_k_block(0, 0);
        moe_prefill_block_barrier<TILE_N>();
    }
    int cur_buf = 0;

#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        float partials[TILE_M] = {};
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        for (int b = begin; b < end; ++b)
        {
            const int next_buf = cur_buf ^ 1;
            int32_t packed_groups[8] = {};
            MoEPrefillDecodedBlockWeights<FMT> block_weights{};
            if (owns_column)
            {
                constexpr int payload_bytes = Traits::payload_bytes;
                const size_t linear =
                    static_cast<size_t>(b) * static_cast<size_t>(N) +
                    static_cast<size_t>(n);
                const uint8_t* payload = payload_base + linear * payload_bytes;
                moe_prefill_decode_groups<FMT>(payload, packed_groups);
                block_weights = moe_prefill_decode_block_weights<FMT>(desc, linear, payload);
            }

            if (b + 1 < blocks_per_row)
            {
                stage_k_block(b + 1, next_buf);
            }

            if (owns_column)
            {
#pragma unroll
                for (int rg = 0; rg < TILE_M; rg += 2)
                {
                    if (rg < u_rows)
                    {
#pragma unroll
                        for (int ri = 0; ri < 2; ++ri)
                        {
                            const int local_row = rg + ri;
                            const int32_t* a4 = staged_A[cur_buf] + local_row * 8;
                            const int32_t* sum_row =
                                kSumAStride > 0 ? (staged_sum_a[cur_buf] + local_row * kSumAStride) : nullptr;
                            partials[local_row] = moe_prefill_accumulate_preloaded_block<FMT>(
                                block_weights, packed_groups, a4,
                                sum_row,
                                staged_scales[cur_buf][local_row],
                                partials[local_row]);
                        }
                    }
                }
            }
            if (b + 1 < blocks_per_row)
            {
                moe_prefill_block_barrier<TILE_N>();
                cur_buf = next_buf;
            }
        }

        if (owns_column)
        {
#pragma unroll
            for (int local_row = 0; local_row < TILE_M; ++local_row)
            {
                if (local_row < u_rows)
                {
                    totals[local_row] = k_partitions == 1
                                            ? partials[local_row]
                                            : rocm_native_quantised_accumulate_rn(
                                                  totals[local_row], partials[local_row]);
                }
            }
        }
    }

    if (owns_column)
    {
#pragma unroll
        for (int local_row = 0; local_row < TILE_M; ++local_row)
        {
            if (local_row < u_rows)
            {
                const int grouped_slot = first_grouped_slot + local_row;
                projection_out[
                    static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) +
                    static_cast<size_t>(n)] = totals[local_row];
            }
        }
    }
}


/**
 * @brief Decode-equivalent grouped gate/up projection for routed prefill rows.
 *
 * Grouped publication must match rowwise serial decode byte-for-byte for every
 * prefill bucket, not only MTP verifier rows. One block owns one compact route
 * slot, one trained output-column tile, and one projection role.
 * `blockIdx.z == 0` computes gate while `blockIdx.z == 1` computes up. The two
 * projections have no arithmetic dependency, so exposing them as independent
 * workgroups improves occupancy without changing either projection's serial-M1
 * partition tree or adding another graph launch. The canonical group plan
 * resolves the expert and original token for each compact row. The
 * grid is bounded by the largest route count that can select this family,
 * not the capacity of the retained prefill graph. Non-local holes and inactive
 * large-prefill launches therefore consume no projection workgroups. FP32
 * gate/up scratch is written exactly
 * like the M=1 decode-table path before one grouped SwiGLU quantization pass
 * consumes it.
 *
 * Explicit partition-worker diagnostics retain the same source-owned K ranges,
 * original range helper and ascending RN fold. Four full64-column planes own
 * disjoint complete partitions and publish one exact partial per bank cell;
 * inactive N lanes remain through the publication barrier. This is independent
 * of the existing adjacent K_LANES block-preparation optimization.
 * @tparam FMT Actual prepared physical decoder, with original source metadata.
 * @tparam TILE_N Independent output columns owned by this workgroup.
 * @tparam STAGE_IQ2_S_GRID Existing immutable IQ2_S codebook staging.
 * @tparam K_LANES Existing adjacent lanes preparing blocks within a partition.
 * @tparam PARALLEL_SOURCE_PARTITIONS Four full TN64 planes compute original partitions; ordinary false remains unchanged.
 * @param A_int8 Original quantized activation rows.
 * @param scales_A_blockwise Original activation-block scales.
 * @param grouped_token_indices Original compact token/route indices.
 * @param grouped_indices_are_route_slots Whether an original index includes the top-k offset.
 * @param gate_descs Original prepared gate descriptors and source identities.
 * @param up_descs Original prepared up descriptors and source identities.
 * @param gate_out Owned gate output, one writer per compact row/column.
 * @param up_out Owned up output, one writer per compact row/column.
 * @param total_slots Original captured route-slot bound, not a host live count.
 * @param seq_len Original token-row bound.
 * @param top_k Original routes per token.
 * @param input_is_original_rows Whether input is token-owned rather than compact-slot-owned.
 * @param N Physical gate/up output columns.
 * @param K Original reduction width in K32 blocks.
 * @param k_partition_policy Original immutable source-policy resolver output.
 * @param num_experts Original descriptor-table extent.
 * @param admission Device-owned live route/owner authority for this retained family.
 */
template <
    NativeQuantisedFormatPrefill FMT,
    int TILE_N = 64,
    bool STAGE_IQ2_S_GRID = false,
    int K_LANES = 1,
    bool PARALLEL_SOURCE_PARTITIONS = false>
__global__ __launch_bounds__(TILE_N * K_LANES *
    (PARALLEL_SOURCE_PARTITIONS ? llaminar2::rocm::kMoERouteOwnedGateUpPartitionWorkers : 1))
void moe_grouped_gate_up_decode_equivalent_prefill_kernel(
    const int8_t*  __restrict__ A_int8,              // [total_slots, K]
    const float*   __restrict__ scales_A_blockwise,  // [total_slots, K / 32]
    const int*     __restrict__ grouped_token_indices, // compact row -> token or route
    int grouped_indices_are_route_slots,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs, // [num_experts]
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,   // [num_experts]
    float*         __restrict__ gate_out,            // [total_slots, N]
    float*         __restrict__ up_out,              // [total_slots, N]
    int total_slots,
    int seq_len,
    int top_k,
    int input_is_original_rows,
    int N,
    int K,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    int num_experts,
    ROCmMoEGroupedRouteAdmission admission)
{
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned))
        return;
    const int grouped_slot = static_cast<int>(blockIdx.y);
    if (grouped_slot >= admission.activeSlots())
        return;
    static_assert(TILE_N == 64 || TILE_N == 128 || TILE_N == 256,
                  "route-owned grouped gate/up supports measured gfx906 column tiles");
    static_assert(!STAGE_IQ2_S_GRID || FMT == NATIVE_QUANTISED_PF_IQ2_S,
                  "only IQ2_S may consume the staged IQ2_S codebook");
    static_assert(K_LANES == 1 || K_LANES == 2 || K_LANES == 4,
                  "route-owned grouped gate/up supports one, two, or four K lanes");
    static_assert(
        K_LANES == 1 ||
            FMT == NATIVE_QUANTISED_PF_Q6_K ||
            FMT == NATIVE_QUANTISED_PF_IQ2_S,
        "only tuned correction-free dual-scale formats may use subgroup preparation");
    static_assert(!PARALLEL_SOURCE_PARTITIONS ||
        (TILE_N == llaminar2::rocm::kMoERouteOwnedGateUpPartitionColumns && K_LANES == 1),
        "four partition-worker planes require the original TN64/K1 range program");
    static_assert(TILE_N * K_LANES <= 256,
                  "route-owned grouped gate/up exceeds the tuned gfx906 block size");

    /*
     * IQ2_S performs four data-dependent 64-bit codebook lookups for every
     * 32-element K block. With one-wave workgroups those vector-memory reads
     * were repeated by every route/projection wave even though the complete
     * codebook is only 8 KiB. The IQ2_S specialization uses four waves per
     * workgroup and cooperatively copies the immutable table into LDS once.
     * Every output lane then reads the identical uint64_t entries from LDS;
     * its dot products, dual-scale commits, corrections, partition boundaries,
     * and ordered partition reduction are unchanged.
     *
     * Ordinary inline execution uses zero dynamic shared bytes for every other
     * codebook. Explicit four-plane diagnostics instead own an exact source-
     * partition partial bank; that mode never changes ordinary geometry.
     */
    constexpr size_t kIQ2SGridBytes =
        llaminar2::rocm::moeRouteOwnedGateUpCodebookSharedBytes(static_cast<std::uint8_t>(FMT));
    constexpr size_t kStagedCodebookBytes =
        STAGE_IQ2_S_GRID ? kIQ2SGridBytes : 0u;
    extern __shared__ uint8_t staged_bytes[];
    uint64_t* staged_iq2s_grid =
        reinterpret_cast<uint64_t*>(staged_bytes);
    if constexpr (STAGE_IQ2_S_GRID)
    {
        constexpr int kIQ2SGridEntries = 1024;
        for (int i = threadIdx.x; i < kIQ2SGridEntries; i += blockDim.x)
            staged_iq2s_grid[i] = d_pf_iq2s_grid[i];
    }

    const int output_lane = PARALLEL_SOURCE_PARTITIONS
        ? static_cast<int>(threadIdx.x) % llaminar2::rocm::kMoERouteOwnedGateUpPartitionColumns
        : static_cast<int>(threadIdx.x) / K_LANES;
    const int n = blockIdx.x * TILE_N + output_lane;
    const int projection_role = static_cast<int>(blockIdx.z);
    if (grouped_slot >= total_slots || projection_role > 1)
        return;

    const int expert_id = admission.expertForSlot(grouped_slot);

    const DeviceNativeQuantisedMatrixDesc desc =
        projection_role == 0 ? gate_descs[expert_id] : up_descs[expert_id];
    if (!prefill_native_quantised_desc_shape_ok<FMT>(desc, N, K))
        return;

    const int k_partitions = k_partition_policy.forDescriptor(desc);

    const int blocks_per_row = K / 32;
    int input_row = grouped_slot;
    if (input_is_original_rows)
    {
        const int index = grouped_token_indices[grouped_slot];
        if (index < 0)
            __builtin_trap();
        input_row = grouped_indices_are_route_slots ? index / top_k : index;
        if (input_row < 0 || input_row >= seq_len)
            __builtin_trap();
    }
    const int8_t* slot_A = A_int8 + static_cast<size_t>(input_row) * K;
    const float* slot_scales =
        scales_A_blockwise + static_cast<size_t>(input_row) * blocks_per_row;

    /*
     * The subgroup geometry makes activation addresses lane-varying, so relying
     * on scalar-cache broadcast would turn four unique block reads into
     * redundant vector traffic. One 256-thread workgroup instead copies the
     * activation row and its scale row into launch LDS before any lane exits for
     * a partial N tile. IQ2_S places these rows immediately after its immutable
     * codebook. Both publications share one barrier, avoiding an extra
     * workgroup-wide rendezvous while retaining an explicit producer/consumer
     * edge for every staged byte.
     */
    if constexpr (K_LANES > 1)
    {
        int8_t* staged_A = reinterpret_cast<int8_t*>(
            staged_bytes + kStagedCodebookBytes);
        float* staged_scales = reinterpret_cast<float*>(
            staged_bytes + kStagedCodebookBytes + static_cast<size_t>(K));
        for (int index = static_cast<int>(threadIdx.x);
             index < K;
             index += static_cast<int>(blockDim.x))
        {
            staged_A[index] = slot_A[index];
        }
        for (int index = static_cast<int>(threadIdx.x);
             index < blocks_per_row;
             index += static_cast<int>(blockDim.x))
        {
            staged_scales[index] = slot_scales[index];
        }
        slot_A = staged_A;
        slot_scales = staged_scales;
    }

    if constexpr (STAGE_IQ2_S_GRID || K_LANES > 1)
        __syncthreads();

    if constexpr (PARALLEL_SOURCE_PARTITIONS)
    {
        // The original descriptor/source policy chooses partitions. Four full
        // column planes merely compute disjoint complete partitions concurrently.
        // A uniform failure precedes all bank accesses and barriers.
        if (k_partitions <= 0 || k_partitions > blocks_per_row ||
            k_partitions > llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
        {
            moe_prefill_fail_fast_invalid_grouped_index();
            return;
        }
        constexpr int columns = llaminar2::rocm::kMoERouteOwnedGateUpPartitionColumns;
        constexpr int workers = llaminar2::rocm::kMoERouteOwnedGateUpPartitionWorkers;
        // Every aligned wave64, and both wave32s within one column plane,
        // has one worker. Make that invariant visible before the tail-column
        // branch so activation addresses can use scalar loads without changing
        // any lane's partition, partial, or final reduction order.
        const int worker = __builtin_amdgcn_readfirstlane(
            static_cast<int>(threadIdx.x) / columns);
        // Immutable IQ2_S grid readers remain live while other workers publish partials.
        // The shared layout therefore gives partials a disjoint grid-prefix offset.
        float* partials = reinterpret_cast<float*>(staged_bytes + kStagedCodebookBytes);
        const int blocks_per_partition = (blocks_per_row + k_partitions - 1) / k_partitions;
        if (n < N)
        {
#pragma unroll 1
            for (int partition = worker; partition < k_partitions; partition += workers)
            {
                const int begin = partition * blocks_per_partition;
                const int end = min(begin + blocks_per_partition, blocks_per_row);
                // Original range helper starts at +0 and commits every original
                // K32 contribution in order. Empty original partitions are +0.
                partials[static_cast<size_t>(partition) * columns + output_lane] = begin < end
                    ? moe_prefill_dot_desc_range<FMT, STAGE_IQ2_S_GRID>(
                          desc, n, slot_A, slot_scales, N, begin, end, staged_iq2s_grid)
                    : 0.0f;
            }
        }
        // Tail columns cannot return before this publication edge. Each live
        // (partition,column) has one writer, and only worker0 consumes it.
        __syncthreads();
        if (worker == 0 && n < N)
        {
            float value;
            if (k_partitions == 1)
                value = partials[output_lane]; // Original KB1 has no extra +0 add.
            else
            {
                float total = 0.0f;
#pragma unroll 1
                for (int partition = 0; partition < k_partitions; ++partition)
                    total = rocm_native_quantised_accumulate_rn(total,
                        partials[static_cast<size_t>(partition) * columns + output_lane]);
                value = total;
            }
            float* projection_out = projection_role == 0 ? gate_out : up_out;
            projection_out[static_cast<size_t>(grouped_slot) * N + n] = value;
        }
        return;
    }

    if (n >= N)
        return;

    const size_t out_index =
        static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) +
        static_cast<size_t>(n);

    float* projection_out = projection_role == 0 ? gate_out : up_out;
    if constexpr (K_LANES > 1)
    {
        const float value = moe_prefill_dual_scale_subgroup_dot<
            FMT, STAGE_IQ2_S_GRID, K_LANES>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            blocks_per_row,
            k_partitions,
            staged_iq2s_grid);
        if ((static_cast<int>(threadIdx.x) & (K_LANES - 1)) == 0)
            projection_out[out_index] = value;
    }
    else
    {
        projection_out[out_index] =
            moe_prefill_dot_desc_serial_m1<FMT, STAGE_IQ2_S_GRID>(
                desc,
                n,
                slot_A,
                slot_scales,
                N,
                k_partitions,
                staged_iq2s_grid);
    }
}

/**
 * @brief Launch one compile-time route-owned gate/up geometry.
 *
 * This helper is the single host boundary for verifier gate/up geometry.  It
 * computes persistent LDS capacity before graph capture and launches on the
 * caller's exact stream; graph replay performs no host policy lookup.  Every
 * specialization invokes the same byte-exact kernel body above.
 *
 * @tparam FMT NativeQuantised execution codebook.
 * @tparam TILE_N Output columns owned by one workgroup before K subdivision.
 * @tparam K_LANES Adjacent lanes preparing blocks within one serial partition.
 * @tparam PARALLEL_SOURCE_PARTITIONS Explicit four-plane computation of unchanged source-owned partitions; false preserves ordinary launches.
 * @param stream Non-null graph-capture stream that owns this launch.
 * @param A_int8 Quantized input rows.
 * @param scales_A_blockwise Per-row activation scales.
 * @param grouped_token_indices Compact route's original token or route index.
 * @param grouped_indices_are_route_slots Whether indices include top-k offset.
 * @param gate_descs Prepared gate descriptors.
 * @param up_descs Prepared up descriptors.
 * @param gate_out Compact gate output rows.
 * @param up_out Compact up output rows.
 * @param total_slots Number of original route slots.
 * @param seq_len Number of original token rows.
 * @param top_k Routed experts per original token.
 * @param input_is_original_rows Whether activation rows are token-owned.
 * @param N Projection output width.
 * @param K Projection reduction width.
 * @param k_partition_policy Source-codebook-indexed serial-M1 partition policy.
 * @param num_experts Number of descriptors in each table.
 * @param admission Original device-owned route/extent authority; no host count is read.
 * @param launch_proof Optional immutable table-owned or diagnostic native capability; never resolved from model TLS.
 */
template <
    NativeQuantisedFormatPrefill FMT,
    int TILE_N,
    int K_LANES,
    bool PARALLEL_SOURCE_PARTITIONS = false>
void launch_moe_grouped_gate_up_route_owned_geometry(
    hipStream_t stream,
    const int8_t* A_int8,
    const float* scales_A_blockwise,
    const int* grouped_token_indices,
    int grouped_indices_are_route_slots,
    const DeviceNativeQuantisedMatrixDesc* gate_descs,
    const DeviceNativeQuantisedMatrixDesc* up_descs,
    float* gate_out,
    float* up_out,
    int total_slots,
    int seq_len,
    int top_k,
    int input_is_original_rows,
    int N,
    int K,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    int num_experts,
    ROCmMoEGroupedRouteAdmission admission,
    const llaminar2::rocm::MoERouteOwnedGateUpAdmission *launch_proof = nullptr)
{
    constexpr bool kStageIQ2SGrid = FMT == NATIVE_QUANTISED_PF_IQ2_S;
    constexpr bool kStageActivation = K_LANES > 1;
    constexpr int partition_workers = PARALLEL_SOURCE_PARTITIONS
        ? llaminar2::rocm::kMoERouteOwnedGateUpPartitionWorkers : 1;
    if constexpr (PARALLEL_SOURCE_PARTITIONS)
        if (!launch_proof)
            throw std::logic_error("parallel source partitions require an exact native capability");
    const auto layout = llaminar2::rocm::moeRouteOwnedGateUpK1SharedLayout(
        static_cast<std::uint8_t>(FMT), PARALLEL_SOURCE_PARTITIONS
            ? llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes
            : llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::Serial,
        PARALLEL_SOURCE_PARTITIONS ? launch_proof->maximumKPartitions() : 0);
    const size_t lds_bytes = layout.total_bytes +
        (kStageActivation
             ? static_cast<size_t>(K) +
                   static_cast<size_t>(K / 32) * sizeof(float)
             : 0u);
    if (launch_proof)
    {
        const auto &proof = *launch_proof;
        const auto &resources = proof.leafResources(static_cast<uint8_t>(FMT));
        const void *actual_function = reinterpret_cast<const void *>(HIP_KERNEL_NAME(
            moe_grouped_gate_up_decode_equivalent_prefill_kernel<FMT, TILE_N, kStageIQ2SGrid, K_LANES, PARALLEL_SOURCE_PARTITIONS>));
        if (K_LANES != 1 ||
            (proof.partitionExecution() == llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes) != PARALLEL_SOURCE_PARTITIONS ||
            (proof.reductionWidth() > 0 && (proof.reductionWidth() != K || proof.outputWidth() != N ||
                !std::equal(std::begin(k_partition_policy.by_codebook), std::end(k_partition_policy.by_codebook),
                    proof.arithmeticPartitions().begin()))) || resources.function != actual_function ||
            resources.block_threads != TILE_N * partition_workers || resources.dynamic_shared_memory_bytes != lds_bytes)
            throw std::logic_error("route-owned launch changed its inspected exact leaf or staging geometry");
    }
    const int compact_capacity = admission.maximumSlotsFor(
        ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
    if (compact_capacity <= 0)
        throw std::invalid_argument("MoE route-owned launch has no admitted compact rows");
    const dim3 grid((N + TILE_N - 1) / TILE_N, compact_capacity, 2);
    const dim3 block(TILE_N * K_LANES * partition_workers);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(moe_grouped_gate_up_decode_equivalent_prefill_kernel<
            FMT, TILE_N, kStageIQ2SGrid, K_LANES, PARALLEL_SOURCE_PARTITIONS>),
        grid,
        block,
        lds_bytes,
        stream,
        A_int8,
        scales_A_blockwise,
        grouped_token_indices,
        grouped_indices_are_route_slots,
        gate_descs,
        up_descs,
        gate_out,
        up_out,
        total_slots,
        seq_len,
        top_k,
        input_is_original_rows,
        N,
        K,
        k_partition_policy,
        num_experts, admission);
}

/**
 * @brief Lower explicit immutable native geometry or retain the original automatic policy.
 * @tparam FMT Original compiled physical decoder; source arithmetic remains descriptor-owned.
 * @param stream Exact original producer stream.
 * @param A_int8 Original quantized activation rows.
 * @param scales_A_blockwise Original K32 scales.
 * @param grouped_token_indices Original device group-to-token map.
 * @param grouped_indices_are_route_slots Whether grouped entries name route slots.
 * @param gate_descs Original device gate table, not a host descriptor mirror.
 * @param up_descs Original device up table.
 * @param gate_out Existing persistent gate bank.
 * @param up_out Existing persistent up bank.
 * @param total_slots Captured route capacity, not a communication extent.
 * @param seq_len Actual physical token rows.
 * @param top_k Original routes per token.
 * @param input_is_original_rows Original activation row interpretation.
 * @param N Actual gate/up output columns.
 * @param K Actual gate/up reduction columns.
 * @param k_partition_policy Same original source-policy resolver result.
 * @param num_experts Original descriptor count.
 * @param admission Original device-owned route admission.
 * @param launch_proof Explicit table-owned or scoped native capability; null retains original automatic selection.
 * @param observation Stack-owned accepted-submission evidence, required only with launch_proof.
 *
 * IQ2_S is codebook-stage limited at tiny route counts and K-loop limited once
 * the route grid broadens.  Measurements on gfx906 select four K lanes for at
 * most sixteen route slots and two K lanes above that boundary. Q6_K has no
 * 8-KiB codebook publication and retains its four-lane winner. Other formats
 * preserve one-lane serial-K execution until an exact profiled specialization
 * proves a different geometry.
 */
template <NativeQuantisedFormatPrefill FMT>
void launch_moe_grouped_gate_up_route_owned(
    hipStream_t stream,
    const int8_t* A_int8,
    const float* scales_A_blockwise,
    const int* grouped_token_indices,
    int grouped_indices_are_route_slots,
    const DeviceNativeQuantisedMatrixDesc* gate_descs,
    const DeviceNativeQuantisedMatrixDesc* up_descs,
    float* gate_out,
    float* up_out,
    int total_slots,
    int seq_len,
    int top_k,
    int input_is_original_rows,
    int N,
    int K,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    int num_experts,
    ROCmMoEGroupedRouteAdmission admission,
    const llaminar2::rocm::MoERouteOwnedGateUpAdmission *launch_proof = nullptr,
    MoERouteOwnedSubmissionObservation *observation = nullptr)
{
    if (launch_proof)
    {
        const auto &admission_proof = *launch_proof;
        if (!(admission_proof.launchCodebookMask() & llaminar2::nativeQuantisedCodebookMaskBit(static_cast<uint8_t>(FMT))) || seq_len != 1)
            throw std::logic_error("route-owned geometry reached a foreign decoder or physical row count");
        const auto geometry = admission_proof.geometry();
        const int columns = llaminar2::rocm::moeRouteOwnedGateUpColumns(geometry);
        #define LAUNCH_ADMITTED_ROUTE_TN(TN, PARALLEL) \
            launch_moe_grouped_gate_up_route_owned_geometry<FMT, TN, 1, PARALLEL>( \
                stream, A_int8, scales_A_blockwise, grouped_token_indices, \
                grouped_indices_are_route_slots, gate_descs, up_descs, gate_out, up_out, \
                total_slots, seq_len, top_k, input_is_original_rows, N, K, \
                k_partition_policy, num_experts, admission, launch_proof)
        if (admission_proof.partitionExecution() == llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes)
        {
            if (geometry != llaminar2::rocm::MoERouteOwnedGateUpGeometry::TN64)
                throw std::logic_error("parallel source partitions require their admitted TN64 leaf");
            LAUNCH_ADMITTED_ROUTE_TN(64, true);
        }
        else
        {
            switch (geometry)
            {
            case llaminar2::rocm::MoERouteOwnedGateUpGeometry::TN64: LAUNCH_ADMITTED_ROUTE_TN(64, false); break;
            case llaminar2::rocm::MoERouteOwnedGateUpGeometry::TN128: LAUNCH_ADMITTED_ROUTE_TN(128, false); break;
            case llaminar2::rocm::MoERouteOwnedGateUpGeometry::TN256: LAUNCH_ADMITTED_ROUTE_TN(256, false); break;
            }
        }
        #undef LAUNCH_ADMITTED_ROUTE_TN
        if (hipGetLastError() != hipSuccess)
        {
            throw std::runtime_error("route-owned diagnostic producer submission failed");
        }
        const int slots = admission.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
        llaminar2::rocm::MoERouteOwnedGateUpLaunch receipt{
            .geometry = geometry, .partition_execution = admission_proof.partitionExecution(),
            .partition_worker_planes = admission_proof.partitionExecution() ==
                llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes ?
                    llaminar2::rocm::kMoERouteOwnedGateUpPartitionWorkers : 1,
            .execution_codebook = static_cast<uint8_t>(FMT),
            .device_ordinal = admission_proof.deviceOrdinal(), .stream = stream,
            .n = N, .k = K, .route_capacity = slots,
            .grid_x = 1 + (N - 1) / columns, .grid_y = slots, .grid_z = 2,
            .block_x = admission_proof.leafResources(static_cast<uint8_t>(FMT)).block_threads,
            .block_y = 1, .block_z = 1,
            .resources = admission_proof.leafResources(static_cast<uint8_t>(FMT))};
        std::copy(std::begin(k_partition_policy.by_codebook), std::end(k_partition_policy.by_codebook),
            receipt.arithmetic_partitions.begin());
        if (!observation) throw std::logic_error("route-owned explicit leaf has no stack observation owner");
        observation->record(receipt);
        return;
    }
    if constexpr (FMT == NATIVE_QUANTISED_PF_IQ2_S)
    {
        if (total_slots <= 16)
        {
            launch_moe_grouped_gate_up_route_owned_geometry<FMT, 64, 4>(
                stream, A_int8, scales_A_blockwise, grouped_token_indices,
                grouped_indices_are_route_slots, gate_descs, up_descs, gate_out, up_out,
                total_slots, seq_len, top_k, input_is_original_rows,
                N, K, k_partition_policy, num_experts, admission);
        }
        else
        {
            launch_moe_grouped_gate_up_route_owned_geometry<FMT, 128, 2>(
                stream, A_int8, scales_A_blockwise, grouped_token_indices,
                grouped_indices_are_route_slots, gate_descs, up_descs, gate_out, up_out,
                total_slots, seq_len, top_k, input_is_original_rows,
                N, K, k_partition_policy, num_experts, admission);
        }
    }
    else if constexpr (FMT == NATIVE_QUANTISED_PF_Q6_K)
    {
        launch_moe_grouped_gate_up_route_owned_geometry<FMT, 64, 4>(
            stream, A_int8, scales_A_blockwise, grouped_token_indices,
            grouped_indices_are_route_slots, gate_descs, up_descs, gate_out, up_out,
            total_slots, seq_len, top_k, input_is_original_rows,
            N, K, k_partition_policy, num_experts, admission);
    }
    else
    {
        launch_moe_grouped_gate_up_route_owned_geometry<FMT, 64, 1>(
            stream, A_int8, scales_A_blockwise, grouped_token_indices,
            grouped_indices_are_route_slots, gate_descs, up_descs, gate_out, up_out,
            total_slots, seq_len, top_k, input_is_original_rows,
            N, K, k_partition_policy, num_experts, admission);
    }
}


/** Eight independent Q8 blocks share a workgroup, not an arithmetic reduction. */
static constexpr int kMoESwiGLUQuantizeThreads = 256;

/**
 * @brief Quantize the live grouped SwiGLU rows with serial-decode arithmetic.
 *
 * Each 32-lane subgroup owns one Q8 block. Packing eight such subgroups into a
 * workgroup fills both halves of a gfx9 wave and amortizes workgroup dispatch.
 * The max tree stays confined to those original 32 lanes, so neither the scale
 * nor quantized bytes depend on neighbouring rows. The captured grid covers
 * capacity, but only the device-published compact prefix may read gate/up.
 *
 * @param gate_data Grouped gate projection scratch.
 * @param up_data Grouped up projection scratch in the same compact order.
 * @param A_int8_out Quantized SwiGLU rows, one writer per element.
 * @param scales_out One FP32 scale per 32-element Q8 block.
 * @param intermediate Expert width, validated as a positive multiple of 32.
 * @param admission Borrowed immutable geometry and live grouping publication.
 */
#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
__global__ __launch_bounds__(kMoESwiGLUQuantizeThreads)
void moe_grouped_swiglu_quantize_prefill_kernel(
    const float* __restrict__ gate_data,         // [total_slots, intermediate]
    const float* __restrict__ up_data,           // [total_slots, intermediate]
    int8_t*      __restrict__ A_int8_out,        // [total_slots, intermediate] quantized SwiGLU output
    float*       __restrict__ scales_out,        // [total_slots, blocks_per_intermediate]
    int intermediate,
    ROCmMoEGroupedRouteAdmission admission)
{
    constexpr int blocks_per_workgroup = kMoESwiGLUQuantizeThreads / 32;
    const int blocks_per_row = intermediate / 32;
    const size_t grouped_block = static_cast<size_t>(blockIdx.x) *
        blocks_per_workgroup + threadIdx.x / 32;
    const size_t live_blocks = static_cast<size_t>(admission.activeSlots()) *
        blocks_per_row;
    // The predicate is uniform within each independent 32-lane max tree.
    // No CTA barrier follows it and empty participants read no projection data.
    if (grouped_block >= live_blocks)
        return;

    const int lane = threadIdx.x % 32;
    const size_t idx = grouped_block * 32 + lane;

    // Use the exact serial-decode exponential and reciprocal instruction tree.
    const float g = gate_data[idx];
    const float u = up_data[idx];
    const float val =
        llaminar2::rocm::decode_equivalent::swigluValue(g, u);

    // Warp-cooperative max reduction for quantization
    float abs_val = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask, 32));

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(abs_val);

    if (lane == 0)
        scales_out[grouped_block] = scale;

    A_int8_out[idx] =
        static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
                val, scale));
}
#endif

/**
 * @brief Batch-invariant expert-tiled down projection for long prefill.
 *
 * The kernel mirrors the gate/up row tile: one decoded expert weight block is
 * reused across several grouped SwiGLU rows, while every row owns an independent
 * accumulator whose K-block order matches serial decode.  Results are written to
 * per-route FP32 rows in @p down_partials.  A following ordered publisher reads
 * those rows in original router order; separating projection from publication is
 * what permits weight reuse without reintroducing nondeterministic FP32 atomics.
 *
 * @tparam FMT NativeQuantised codebook for experts handled by this launch.
 * @tparam TILE_M Trained expert-row tile (4, 8, 12, or 16 rows).
 * @tparam TILE_N Trained output-column tile (64, 128, or 256 lanes).
 * @param A_int8 Quantized grouped SwiGLU rows.
 * @param scales_A_blockwise FP32 scales for the grouped K32 activation blocks.
 * @param down_descs Prepared full or column-sliced down descriptors.
 * @param group_counts Complete live row count for each expert.
 * @param group_offsets Exclusive offsets into grouped activation rows.
 * @param work_directory Current fixed or adaptive packed expert tiles.
 * @param down_partials Unweighted FP32 grouped projection bank.
 * @param max_directory_tiles Captured directory capacity.
 * @param N Prepared physical output columns.
 * @param K Projection contraction width.
 * @param k_partition_policy Original unsliced source arithmetic partitions.
 * @param num_experts Descriptor and group-plan cardinality.
 * @param adaptive_tile_count Optional device-owned current directory extent.
 * @param admission Shared device selection of the ExpertTiled route family.
 */
template <NativeQuantisedFormatPrefill FMT, int TILE_M, int TILE_N>
__global__ __launch_bounds__(TILE_N, 2)
void moe_grouped_down_batch_invariant_tiled_prefill_kernel(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const int* __restrict__ group_counts,
    const int* __restrict__ group_offsets,
    const uint32_t* __restrict__ work_directory,
    float* __restrict__ down_partials,
    int max_directory_tiles,
    int N,
    int K,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    int num_experts,
    const uint32_t* __restrict__ adaptive_tile_count,
    ROCmMoEGroupedRouteAdmission admission)
{
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
        return;
    static_assert(TILE_M == 4 || TILE_M == 8 ||
                  TILE_M == 12 || TILE_M == 16,
                  "grouped-prefill down supports row tiles 4/8/12/16");
    static_assert(TILE_N == 64 || TILE_N == 128 || TILE_N == 256,
                  "grouped-prefill down supports column tiles 64/128/256");
    const int n = static_cast<int>(blockIdx.x) * TILE_N + threadIdx.x;
    const int directory_index = static_cast<int>(blockIdx.y);
    if (directory_index >= max_directory_tiles)
        return;
    if (adaptive_tile_count &&
        directory_index >= static_cast<int>(*adaptive_tile_count))
    {
        return;
    }

    uint32_t packed_tile = work_directory[directory_index];
    if (packed_tile == kMoEPrefillUnusedTile)
        return;

    const int expert_id =
        static_cast<int>(packed_tile & kMoEPrefillExpertMask);
    const int first_row =
        static_cast<int>(packed_tile >> kMoEPrefillExpertBits);
    if (expert_id < 0 || expert_id >= num_experts)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }

    const int count = group_counts[expert_id];
    const int grouped_offset = group_offsets[expert_id];
    if (first_row < 0 || first_row >= count || grouped_offset < 0)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const int rows_in_tile = min(TILE_M, count - first_row);
    const int first_grouped_slot = grouped_offset + first_row;

    const DeviceNativeQuantisedMatrixDesc desc = down_descs[expert_id];
    if (!prefill_native_quantised_desc_shape_ok<FMT>(desc, N, K))
        return;

    const int k_partitions = k_partition_policy.forDescriptor(desc);

    const int blocks_per_row = K / 32;
    if (k_partitions <= 0 ||
        k_partitions > blocks_per_row ||
        k_partitions >
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        moe_prefill_fail_fast_invalid_grouped_index();
        return;
    }
    const uint8_t* payload_base = desc.payload;
    /*
     * Inactive lanes in a partial column tile are still participants in the
     * LDS producer/consumer protocol. Keep them alive through every barrier,
     * while guarding descriptor reads and final stores with this ownership bit.
     */
    const bool owns_column = n < N;
    using Traits = PrefillNATIVE_QUANTISEDTraits<FMT>;
    constexpr int kSumAStride =
        (FMT == NATIVE_QUANTISED_PF_IQ1_M) ? 4 :
        Traits::is_dual_scale_asym ? 2 :
        Traits::is_asymmetric ? 1 : 0;

    __shared__ __align__(16) int32_t staged_A[2][TILE_M * 8];
    __shared__ float staged_scales[2][TILE_M];
    __shared__ int32_t staged_sum_a[2][kSumAStride > 0 ? TILE_M * kSumAStride : 1];

    const int u_rows = __builtin_amdgcn_readfirstlane(rows_in_tile);
    const int u_first_slot = __builtin_amdgcn_readfirstlane(first_grouped_slot);

    if (static_cast<int>(threadIdx.x) >= u_rows * 8 &&
        static_cast<int>(threadIdx.x) < TILE_M * 8)
    {
        staged_A[0][threadIdx.x] = 0;
        staged_A[1][threadIdx.x] = 0;
    }
    if (static_cast<int>(threadIdx.x) >= u_rows &&
        static_cast<int>(threadIdx.x) < TILE_M)
    {
        staged_scales[0][threadIdx.x] = 0.0f;
        staged_scales[1][threadIdx.x] = 0.0f;
        if constexpr (kSumAStride > 0)
        {
#pragma unroll
            for (int s = 0; s < kSumAStride; ++s)
            {
                staged_sum_a[0][static_cast<int>(threadIdx.x) * kSumAStride + s] = 0;
                staged_sum_a[1][static_cast<int>(threadIdx.x) * kSumAStride + s] = 0;
            }
        }
    }

    const auto stage_k_block = [&](int block_idx, int buf) __attribute__((always_inline))
    {
        if (static_cast<int>(threadIdx.x) < u_rows)
        {
            const int r = static_cast<int>(threadIdx.x);
            const int grouped_slot = u_first_slot + r;
            const int4* in16 = reinterpret_cast<const int4*>(
                A_int8 + static_cast<size_t>(grouped_slot) * static_cast<size_t>(K) +
                static_cast<size_t>(block_idx) * 32u);
            const int4 lo = in16[0];
            const int4 hi = in16[1];
            int4* dst16 = reinterpret_cast<int4*>(staged_A[buf] + r * 8);
            dst16[0] = lo;
            dst16[1] = hi;
            staged_scales[buf][r] = scales_A_blockwise[
                static_cast<size_t>(grouped_slot) * static_cast<size_t>(blocks_per_row) +
                static_cast<size_t>(block_idx)];
            if constexpr (kSumAStride > 0)
            {
                constexpr int32_t ones = 0x01010101;
                const int32_t s0 = ::llaminar2::rocm_integer_contract::signedDot4(
                    lo.y, ones, ::llaminar2::rocm_integer_contract::signedDot4(lo.x, ones, 0));
                const int32_t s1 = ::llaminar2::rocm_integer_contract::signedDot4(
                    lo.w, ones, ::llaminar2::rocm_integer_contract::signedDot4(lo.z, ones, 0));
                const int32_t s2 = ::llaminar2::rocm_integer_contract::signedDot4(
                    hi.y, ones, ::llaminar2::rocm_integer_contract::signedDot4(hi.x, ones, 0));
                const int32_t s3 = ::llaminar2::rocm_integer_contract::signedDot4(
                    hi.w, ones, ::llaminar2::rocm_integer_contract::signedDot4(hi.z, ones, 0));
                const int sb = r * kSumAStride;
                if constexpr (FMT == NATIVE_QUANTISED_PF_IQ1_M)
                {
                    staged_sum_a[buf][sb + 0] = s0;
                    staged_sum_a[buf][sb + 1] = s1;
                    staged_sum_a[buf][sb + 2] = s2;
                    staged_sum_a[buf][sb + 3] = s3;
                }
                else if constexpr (Traits::is_dual_scale_asym)
                {
                    staged_sum_a[buf][sb + 0] = s0 + s1;
                    staged_sum_a[buf][sb + 1] = s2 + s3;
                }
                else
                {
                    staged_sum_a[buf][sb] = (s0 + s1) + (s2 + s3);
                }
            }
        }
    };

    float totals[TILE_M] = {};
    const int blocks_per_partition =
        (blocks_per_row + k_partitions - 1) / k_partitions;

    if (blocks_per_row > 0)
    {
        stage_k_block(0, 0);
        moe_prefill_block_barrier<TILE_N>();
    }
    int cur_buf = 0;

#pragma unroll 1
    for (int partition = 0; partition < k_partitions; ++partition)
    {
        float partials[TILE_M] = {};
        const int begin = partition * blocks_per_partition;
        const int end = min(begin + blocks_per_partition, blocks_per_row);
        for (int b = begin; b < end; ++b)
        {
            const int next_buf = cur_buf ^ 1;
            int32_t packed_groups[8] = {};
            MoEPrefillDecodedBlockWeights<FMT> block_weights{};
            if (owns_column)
            {
                constexpr int payload_bytes = Traits::payload_bytes;
                const size_t linear =
                    static_cast<size_t>(b) * static_cast<size_t>(N) +
                    static_cast<size_t>(n);
                const uint8_t* payload = payload_base + linear * payload_bytes;
                moe_prefill_decode_groups<FMT>(payload, packed_groups);
                block_weights = moe_prefill_decode_block_weights<FMT>(desc, linear, payload);
            }

            if (b + 1 < blocks_per_row)
            {
                stage_k_block(b + 1, next_buf);
            }

            if (owns_column)
            {
#pragma unroll
                for (int rg = 0; rg < TILE_M; rg += 2)
                {
                    if (rg < u_rows)
                    {
#pragma unroll
                        for (int ri = 0; ri < 2; ++ri)
                        {
                            const int local_row = rg + ri;
                            const int32_t* a4 = staged_A[cur_buf] + local_row * 8;
                            const int32_t* sum_row =
                                kSumAStride > 0 ? (staged_sum_a[cur_buf] + local_row * kSumAStride) : nullptr;
                            partials[local_row] = moe_prefill_accumulate_preloaded_block<FMT>(
                                block_weights, packed_groups, a4,
                                sum_row,
                                staged_scales[cur_buf][local_row],
                                partials[local_row]);
                        }
                    }
                }
            }
            if (b + 1 < blocks_per_row)
            {
                moe_prefill_block_barrier<TILE_N>();
                cur_buf = next_buf;
            }
        }

        if (owns_column)
        {
#pragma unroll
            for (int local_row = 0; local_row < TILE_M; ++local_row)
            {
                if (local_row < u_rows)
                {
                    totals[local_row] = k_partitions == 1
                                            ? partials[local_row]
                                            : rocm_native_quantised_accumulate_rn(
                                                  totals[local_row], partials[local_row]);
                }
            }
        }
    }

    if (owns_column)
    {
#pragma unroll
        for (int local_row = 0; local_row < TILE_M; ++local_row)
        {
            if (local_row < u_rows)
            {
                const int grouped_slot = u_first_slot + local_row;
                down_partials[
                    static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) +
                    static_cast<size_t>(n)] = totals[local_row];
            }
        }
    }
}

/**
 * @brief Publish precomputed down rows in original top-k order.
 *
 * Each lane owns exactly one token/output-column result.  It walks the router's
 * original route slots, resolves each slot to its grouped partial row, and
 * executes `total += weight * partial` in the same order as serial decode.  The
 * kernel has one writer per output value, so publication is deterministic and
 * contains no floating-point atomics.
 */
#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
__global__ __launch_bounds__(64, 4)
void moe_grouped_down_partials_ordered_publish_prefill_kernel(
    const float* __restrict__ down_partials,
    const int* __restrict__ original_to_grouped,
    const float* __restrict__ grouped_weights,
    float* __restrict__ output,
    int seq_len,
    int top_k,
    int d_model,
    ROCmMoEGroupedRouteAdmission admission)
{
    // The unselected publisher must not overwrite the selected family's bytes.
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
        return;
    constexpr int TILE_N = 64;
    const int n = static_cast<int>(blockIdx.x) * TILE_N + threadIdx.x;
    const int token = static_cast<int>(blockIdx.y);
    if (token >= seq_len || n >= d_model)
        return;

    float total = 0.0f;
#pragma unroll 1
    for (int route = 0; route < top_k; ++route)
    {
        const int original_slot = token * top_k + route;
        const int grouped_slot = original_to_grouped[original_slot];
        if (grouped_slot < 0)
            continue;
        const float weighted_route = rocm_native_quantised_weight_route_rn(
            grouped_weights[grouped_slot],
            down_partials[
                static_cast<size_t>(grouped_slot) *
                    static_cast<size_t>(d_model) +
                static_cast<size_t>(n)]);
        total = rocm_native_quantised_accumulate_rn(total, weighted_route);
    }

    output[
        static_cast<size_t>(token) * static_cast<size_t>(d_model) +
        static_cast<size_t>(n)] = total;
}
#endif


/**
 * @brief Dispatch one NativeQuantised down dot by the descriptor's codebook id.
 *
 * The verifier publication kernel below must walk original route slots in a
 * single thread so it can preserve serial decode's `route 0, route 1, ...`
 * accumulation order.  Production descriptor tables are allowed to contain
 * mixed down codebooks across experts, so launching one templated kernel per
 * codebook would either overwrite earlier partial sums or reintroduce atomic
 * ordering.  This small device-side switch keeps the ordered publication kernel
 * generic while still reusing the exact same per-codebook dot helper as the
 * grouped gate/up kernels and serial decode.
 */
__device__ __forceinline__ float moe_prefill_dot_desc_serial_m1_dispatch(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int k_partitions)
{
    switch (static_cast<NativeQuantisedFormatPrefill>(desc.codebook_id))
    {
    case NATIVE_QUANTISED_PF_Q4_0:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q4_0>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ4_NL:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ4_NL>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ4_XS_SOURCE:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ4_XS_SOURCE>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q4_1:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q4_1>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q5_0:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q5_0>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q5_1:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q5_1>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q6_K:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q6_K>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q3_K:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q3_K>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q2_K:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q2_K>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ3_S:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ3_S>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ3_XXS:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ3_XXS>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ2_S:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ2_S>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ2_XS:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ2_XS>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ2_XXS:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ2_XXS>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ1_S:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ1_S>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_IQ1_M:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_IQ1_M>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_Q8_0:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_Q8_0>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    case NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN:
        return moe_prefill_dot_desc_serial_m1<NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN>(
            desc, n, slot_A, slot_scales, N, k_partitions);
    default:
        moe_prefill_fail_fast_invalid_grouped_index();
        return 0.0f;
    }
}

/** Sentinel selecting the first-class heterogeneous descriptor-table kernel. */
constexpr int kMoEPrefillMixedCodebookTag = -1;

/**
 * @brief Resolve one down-projection dot under a capture-time codebook policy.
 *
 * Homogeneous prepared tables instantiate the exact NativeQuantised format, letting
 * HIP inline only that decoder and discard the per-output format switch. A
 * genuinely heterogeneous table instantiates @ref kMoEPrefillMixedCodebookTag
 * and performs the descriptor-owned dispatch intentionally.
 *
 * @tparam FORMAT_TAG NativeQuantised format value, or the mixed-table sentinel.
 */
template <int FORMAT_TAG>
__device__ __forceinline__ float moe_prefill_down_dot(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ slot_A,
    const float* __restrict__ slot_scales,
    int N,
    int blocks_per_row,
    const MoEPrefillSerialM1KPartitionPolicy& k_partition_policy)
{
    if constexpr (FORMAT_TAG == kMoEPrefillMixedCodebookTag)
    {
        return moe_prefill_dot_desc_serial_m1_dispatch(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            k_partition_policy.forDescriptor(desc));
    }
    else
    {
        constexpr auto kFormat =
            static_cast<NativeQuantisedFormatPrefill>(FORMAT_TAG);
        if (!prefill_native_quantised_desc_shape_ok<kFormat>(
                desc, N, blocks_per_row * 32))
        {
            /*
             * A homogeneous mask and a mismatched descriptor cannot describe a
             * valid captured graph. Trap on device instead of publishing a
             * plausible zero row and allowing inference to limp onward.
             */
            moe_prefill_fail_fast_invalid_grouped_index();
            return 0.0f;
        }
        return moe_prefill_dot_desc_serial_m1<kFormat>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            k_partition_policy.forDescriptor(desc));
    }
}

/**
 * @brief Batch-invariant ordered down projection and publication.
 *
 * This is the canonical ROCm grouped-prefill down contract for every M bucket.
 * For each token/output column, one thread walks the original route slots in
 * top-k order, computes the down dot for that route's expert, applies the route
 * weight, and accumulates into a single FP32 total. No per-route down rows are
 * materialized and no atomic add participates in the result, so full prefill,
 * split prefill, and row-by-row decode all retain identical arithmetic order.
 *
 * The kernel is still a true grouped implementation: all request tokens, all
 * route slots, and all active experts are published by one graph-capturable
 * device dispatch. The only serial loop is the tiny top-k route loop inside
 * each token/column lane, exactly where serial decode defines the mathematical
 * accumulation order.
 *
 * @tparam FORMAT_TAG NativeQuantised format value, or the mixed-table sentinel.
 */
template <int FORMAT_TAG>
__global__ __launch_bounds__(64)
void moe_grouped_down_publish_original_token_ordered_prefill_kernel(
    const int8_t* __restrict__ swiglu_int8,          // [total_slots, intermediate]
    const float* __restrict__ swiglu_scales,         // [total_slots, intermediate / 32]
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs, // [num_experts]
    const int* __restrict__ original_to_grouped,     // [seq_len * top_k]
    const int* __restrict__ original_expert_ids,     // [seq_len * top_k], null when num_experts == 1
    const float* __restrict__ grouped_weights,       // [total_slots]
    float* __restrict__ output,                      // [seq_len, d_model], overwritten
    int seq_len,
    int top_k,
    int N,
    int K,
    int num_experts,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    ROCmMoEGroupedRouteAdmission admission)
{
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned))
        return;
    constexpr int TILE_N = 64;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    if (n >= N)
        return;

    const int blocks_per_row = K / 32;
    for (int token = static_cast<int>(blockIdx.y); token < seq_len;
         token += static_cast<int>(gridDim.y))
    {
        float total = 0.0f;
#pragma unroll 1
        for (int route = 0; route < top_k; ++route)
        {
            const int original_slot = token * top_k + route;
            const int grouped_slot = original_to_grouped[original_slot];
            if (grouped_slot < 0)
                continue;

            const int expert_id = original_expert_ids
                                      ? original_expert_ids[original_slot]
                                      : ((num_experts == 1) ? 0 : -1);
            if (expert_id < 0 || expert_id >= num_experts)
                continue;

            const DeviceNativeQuantisedMatrixDesc desc = down_descs[expert_id];
            const int8_t* slot_A = swiglu_int8 + static_cast<size_t>(grouped_slot) * K;
            const float* slot_scales =
                swiglu_scales + static_cast<size_t>(grouped_slot) * blocks_per_row;
            const float expert_acc = moe_prefill_down_dot<FORMAT_TAG>(
                desc,
                n,
                slot_A,
                slot_scales,
                N,
                blocks_per_row,
                k_partition_policy);
            const float weighted_route = rocm_native_quantised_weight_route_rn(
                grouped_weights[grouped_slot], expert_acc);
            total = rocm_native_quantised_accumulate_rn(total, weighted_route);
        }

        output[static_cast<size_t>(token) * static_cast<size_t>(N) + static_cast<size_t>(n)] = total;
    }
}

/**
 * @brief Compile-time geometry shared by canonical route publication and inspection.
 *
 * Output columns stay wave-sized for every format. Only the physical Q5_1
 * specialization assigns four workers to the original ordered K partitions.
 */
template <int FORMAT_TAG, int WORKERS>
struct MoECanonicalRouteDownGeometry
{
    static constexpr int columns = 64;
    static_assert(WORKERS == 1 || (WORKERS == 4 && FORMAT_TAG == static_cast<int>(NATIVE_QUANTISED_PF_Q5_1)));
    static constexpr int workers = WORKERS;
    static constexpr int threads = columns * workers;
};

/**
 * @brief Publish one weighted grouped down row per original router slot.
 *
 * Invalid or participant-remote slots explicitly write zero. The route axis is
 * preserved through the LocalTP collective and is reduced only afterward by
 * the canonical router-order epilogue. A bounded slot grid avoids scheduling
 * the entire physical matrix when device admission chooses the other family.
 *
 * @details The Q5_1 physical specialization uses four wave-uniform workers
 * over the original serial-M1 K partitions. A full-capacity partition-major
 * shared bank and two slot-loop barriers preserve ascending RN publication
 * for every M. Other formats and mixed-table execution retain the original
 * single-wave implementation.
 * @tparam FORMAT_TAG NativeQuantised format value, or the mixed-table sentinel.
 * @tparam WORKERS One serial wave or four original-partition worker waves.
 */
template <int FORMAT_TAG, int WORKERS = (FORMAT_TAG == static_cast<int>(NATIVE_QUANTISED_PF_Q5_1) ? 4 : 1)>
__global__ __launch_bounds__((MoECanonicalRouteDownGeometry<FORMAT_TAG, WORKERS>::threads))
void moe_grouped_down_publish_canonical_routes_prefill_kernel(
    const int8_t* __restrict__ swiglu_int8,
    const float* __restrict__ swiglu_scales,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const int* __restrict__ original_to_grouped,
    const int* __restrict__ original_expert_ids,
    const float* __restrict__ grouped_weights,
    float* __restrict__ route_output,
    int total_slots,
    int N,
    int K,
    int num_experts,
    MoEPrefillSerialM1KPartitionPolicy k_partition_policy,
    ROCmMoEGroupedRouteAdmission admission)
{
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned))
        return;
    if constexpr (WORKERS == 4)
    {
        constexpr int columns = MoECanonicalRouteDownGeometry<FORMAT_TAG, WORKERS>::columns;
        constexpr int workers = MoECanonicalRouteDownGeometry<FORMAT_TAG, WORKERS>::workers;
        constexpr int maximum_partitions = llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
        __shared__ float partials[maximum_partitions][columns];
        const int output_lane = static_cast<int>(threadIdx.x) % columns;
        const int worker = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x) / columns);
        const int n = static_cast<int>(blockIdx.x) * columns + output_lane;
        const int blocks_per_row = K / 32;

        for (int original_slot = static_cast<int>(blockIdx.y); original_slot < total_slots;
             original_slot += static_cast<int>(gridDim.y))
        {
            const size_t output_index = static_cast<size_t>(original_slot) * N + n;
            const int grouped_slot = __builtin_amdgcn_readfirstlane(original_to_grouped[original_slot]);
            const int expert_id = original_expert_ids
                ? __builtin_amdgcn_readfirstlane(original_expert_ids[original_slot])
                : (num_experts == 1 ? 0 : -1);
            if (grouped_slot < 0 || expert_id < 0 || expert_id >= num_experts)
            {
                // The original publisher authors positive zero for every invalid
                // route column. Exactly one wave writes it; no duplicate writers.
                if (worker == 0 && n < N)
                    route_output[output_index] = 0.0f;
                continue;
            }
            if (grouped_slot >= total_slots)
                moe_prefill_fail_fast_invalid_grouped_index();
            const DeviceNativeQuantisedMatrixDesc desc = down_descs[expert_id];
            if (!prefill_native_quantised_desc_shape_ok<NATIVE_QUANTISED_PF_Q5_1>(desc, N, K))
                moe_prefill_fail_fast_invalid_grouped_index();
            const int partitions = k_partition_policy.forDescriptor(desc);
            if (partitions <= 0 || partitions > blocks_per_row || partitions > maximum_partitions)
                moe_prefill_fail_fast_invalid_grouped_index();
            const int blocks_per_partition = (blocks_per_row + partitions - 1) / partitions;
            const int8_t* slot_A = swiglu_int8 + static_cast<size_t>(grouped_slot) * K;
            const float* slot_scales = swiglu_scales + static_cast<size_t>(grouped_slot) * blocks_per_row;
            if (n < N)
            {
#pragma unroll 1
                for (int partition = worker; partition < partitions; partition += workers)
                {
                    const int begin = partition * blocks_per_partition;
                    const int end = min(begin + blocks_per_partition, blocks_per_row);
                    partials[partition][output_lane] = begin < end
                        ? moe_prefill_dot_desc_range<NATIVE_QUANTISED_PF_Q5_1>(
                              desc, n, slot_A, slot_scales, N, begin, end)
                        : 0.0f;
                }
            }
            // Tail lanes remain participants: every live partial is published
            // before wave 0 reads it. No thread exits early on a column tail.
            __syncthreads();
            if (worker == 0 && n < N)
            {
                float value = partials[0][output_lane];
                if (partitions != 1)
                {
                    float total = 0.0f;
#pragma unroll 1
                    for (int partition = 0; partition < partitions; ++partition)
                        total = rocm_native_quantised_accumulate_rn(total, partials[partition][output_lane]);
                    value = total;
                }
                route_output[output_index] = grouped_weights
                    ? rocm_native_quantised_weight_route_rn(grouped_weights[grouped_slot], value) : value;
            }
            // Every slot iteration retires wave 0's bank reads before any other
            // wave can overwrite them for the next slot. Keep this second barrier
            // explicit for all-M publication, including tail-column participants.
            __syncthreads();
        }
        return;
    }
    constexpr int TILE_N = MoECanonicalRouteDownGeometry<FORMAT_TAG, WORKERS>::columns;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    if (n >= N)
        return;

    const int blocks_per_row = K / 32;
    for (int original_slot = blockIdx.y; original_slot < total_slots;
         original_slot += gridDim.y)
    {
        const size_t output_index =
            static_cast<size_t>(original_slot) * static_cast<size_t>(N) +
            static_cast<size_t>(n);
        const int grouped_slot = original_to_grouped[original_slot];
        const int expert_id = original_expert_ids
                                  ? original_expert_ids[original_slot]
                                  : ((num_experts == 1) ? 0 : -1);
        if (grouped_slot < 0 || expert_id < 0 || expert_id >= num_experts)
        {
            route_output[output_index] = 0.0f;
            continue;
        }

        const DeviceNativeQuantisedMatrixDesc desc = down_descs[expert_id];
        const int8_t* slot_A =
            swiglu_int8 + static_cast<size_t>(grouped_slot) * K;
        const float* slot_scales =
            swiglu_scales + static_cast<size_t>(grouped_slot) * blocks_per_row;
        const float expert_acc = moe_prefill_down_dot<FORMAT_TAG>(
            desc,
            n,
            slot_A,
            slot_scales,
            N,
            blocks_per_row,
            k_partition_policy);
        route_output[output_index] = grouped_weights
            ? rocm_native_quantised_weight_route_rn(grouped_weights[grouped_slot], expert_acc) : expert_acc;
    }
}

/**
 * @brief Scatter expert-tiled partials with one exact multiply per route element.
 *
 * This bandwidth-only epilogue uses four wavefronts per workgroup, unlike the
 * register-heavy direct projection. Slot-strided ownership keeps its grid
 * bounded while covering every active and remote/zero slot exactly once.
 */
#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
__global__ __launch_bounds__(256, 4)
void moe_grouped_down_partials_canonical_publish_prefill_kernel(
    const float* __restrict__ down_partials,
    const int* __restrict__ original_to_grouped,
    const float* __restrict__ grouped_weights,
    float* __restrict__ route_output,
    int total_slots,
    int d_model,
    ROCmMoEGroupedRouteAdmission admission)
{
    if (!admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled))
        return;
    constexpr int TILE_N = 256;
    const int n = blockIdx.x * TILE_N + threadIdx.x;
    if (n >= d_model)
        return;

    for (int original_slot = blockIdx.y; original_slot < total_slots;
         original_slot += gridDim.y)
    {
        const int grouped_slot = original_to_grouped[original_slot];
        const size_t output_index =
            static_cast<size_t>(original_slot) *
                static_cast<size_t>(d_model) +
            static_cast<size_t>(n);
        if (grouped_slot < 0)
        {
            route_output[output_index] = 0.0f;
            continue;
        }
        const float value = down_partials[static_cast<size_t>(grouped_slot) * d_model + n];
        route_output[output_index] = grouped_weights
            ? rocm_native_quantised_weight_route_rn(grouped_weights[grouped_slot], value) : value;
    }
}
#endif

/**
 * @brief Sum route slots in serial router order while overlapping their loads.
 *
 * Each lane remains the unique writer of one output. The shared helper stages
 * a bounded register window, but preserves every rounded addition and reads
 * only the declared route extent. No graph node, workspace or stream changes.
 */
#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
__global__ __launch_bounds__(256)
void moe_reduce_canonical_route_contributions_kernel(
    const float* __restrict__ route_contributions,
    float* __restrict__ output,
    int seq_len,
    int top_k,
    int d_model)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= seq_len || n >= d_model)
        return;

    const size_t first = static_cast<size_t>(token) * top_k * d_model + n;
    const float sum = llaminar2::device_ordered_route_fold::sum(
        route_contributions + first, static_cast<size_t>(d_model), top_k);
    output[
        static_cast<size_t>(token) * static_cast<size_t>(d_model) +
        static_cast<size_t>(n)] = sum;
}
#endif

// ============================================================================
// C API: Full grouped MoE prefill pipeline (graph-capturable)
//
// Launches one fixed, fully grouped pipeline on the caller's stream. Tiny MTP
// buckets use direct route-owned projection. Long prefill builds a compact
// device work directory, reuses decoded expert weights across row tiles, and
// publishes materialized down rows in original top-k order. No path uses FP32
// atomics, host synchronization, or replay-time host-authored work metadata.
// ============================================================================

namespace
{

    /** Compile-time candidate selected for one graph-captured projection. */
    struct MoEPrefillTileConfig
    {
        int rows = 0;
        int columns = 0;
    };

    /** Joint gate/up and down policy chosen once for one captured MoE graph. */
    struct MoEPrefillPairConfig
    {
        MoEPrefillTileConfig gateup{};
        MoEPrefillTileConfig down{};
        bool exact_overlay = false;
    };

    bool moe_prefill_valid_tile_rows(int rows)
    {
        return rows == 4 || rows == 8 || rows == 12 || rows == 16 ||
               rows == kMoEPrefillAdaptiveTileRows;
    }

    bool moe_prefill_valid_tile_columns(int columns)
    {
        return columns == 64 || columns == 128 || columns == 256;
    }

    /**
     * @brief Select one LUT-family producer while preserving descriptor identities.
     * @param mask Actual physical formats present in a descriptor table.
     * @return The same mask except that24 owns both IQ4 LUT members when present.
     * @details Each24 specialization resolves scale words using the actual
     * descriptor, so mixed4/24 and24/4 use the same integer projection family.
     */
    uint32_t moe_prefill_launch_mask(uint32_t mask)
    {
        return llaminar2::rocm::moeRouteOwnedGateUpLaunchCodebookMask(mask);
    }

    /** Return the sole execution codebook in a descriptor mask, if any. */
    int moe_prefill_single_codebook(uint32_t codebook_mask)
    {
        codebook_mask = moe_prefill_launch_mask(codebook_mask);
        if (codebook_mask == 0 ||
            (codebook_mask & (codebook_mask - 1u)) != 0u)
        {
            return -1;
        }

        int codebook = 0;
        while ((codebook_mask >>= 1u) != 0u)
            ++codebook;
        return codebook;
    }

    /**
     * @brief Resolve every codebook's exact serial-M1 split-K policy for capture.
     *
     * The generated decode selector is queried only while the graph is being
     * constructed. The resulting fixed-size value becomes an ordinary captured
     * kernel argument, so graph replay performs no host lookup or transfer. A
     * missing or out-of-range policy is fatal to graph construction rather than
     * silently changing the arithmetic tree.
     */
    bool moe_prefill_resolve_serial_m1_k_partitions(
        uint32_t codebook_mask,
        int n,
        int k,
        MoEPrefillSerialM1KPartitionPolicy& policy)
    {
        if (codebook_mask == 0 || n <= 0 || k <= 0 || (k % 32) != 0)
            return false;

        bool resolved_any = false;
        for (uint8_t codebook_id = 0; codebook_id < 32; ++codebook_id)
        {
            if ((codebook_mask &
                 (uint32_t{1} << static_cast<uint32_t>(codebook_id))) == 0)
            {
                continue;
            }

            int k_partitions = 0;
            if (!rocmGemv_native_quantised_query_serial_m1_config(
                    codebook_id,
                    n,
                    k,
                    &k_partitions,
                    nullptr) ||
                k_partitions <= 0 ||
                k_partitions > k / 32 ||
                k_partitions >
                    llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
            {
                return false;
            }

            policy.by_codebook[codebook_id] =
                static_cast<uint8_t>(k_partitions);
            resolved_any = true;
        }
        return resolved_any;
    }

    /** Apply explicit trainer controls after generated production selection. */
    void moe_prefill_apply_tile_override(
        MoEPrefillProjectionRole role,
        MoEPrefillTileConfig &config)
    {
        const auto &env = llaminar2::debugEnv().rocm;
        if (role == MoEPrefillProjectionRole::GateUp)
        {
            if (env.moe_prefill_gateup_tile_m > 0)
                config.rows = env.moe_prefill_gateup_tile_m;
            if (env.moe_prefill_gateup_tile_n > 0)
                config.columns = env.moe_prefill_gateup_tile_n;
            return;
        }

        if (env.moe_prefill_down_tile_m > 0)
            config.rows = env.moe_prefill_down_tile_m;
        if (env.moe_prefill_down_tile_n > 0)
            config.columns = env.moe_prefill_down_tile_n;
    }

    /**
     * @brief Select a grouped projection candidate for capture or trainer use.
     *
     * Production is selected exclusively by the complete generated
     * codebook/M/aspect-ratio table. Role-specific environment overrides are
     * intentionally explicit trainer controls: they let the Perf__ sweep hold
     * down projection fixed while training gate/up, then reverse the experiment,
     * without introducing a second benchmark-only implementation. A generated
     * policy miss returns an invalid zero tile and is rejected by the caller;
     * there is no silent generic production fallback.
     */
    MoEPrefillTileConfig moe_prefill_select_tile_config(
        MoEPrefillProjectionRole role,
        NativeQuantisedFormatPrefill format,
        int m,
        int n,
        int k)
    {
        MoEPrefillTileConfig config{};
        llaminar2::rocm::generated::ROCmMoEGroupedPrefillConfig generated{};
        const uint8_t role_id =
            role == MoEPrefillProjectionRole::GateUp ? 0u : 1u;
        // Expanded asymmetric INT8 has codebook-19 payload traffic and the
        // same launch geometry. It deliberately inherits the sealed codebook-
        // 19 policy until a dedicated synthetic-format corpus is certified;
        // execution and telemetry retain codebook 23.
        const uint8_t policy_codebook =
            format == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE ? static_cast<uint8_t>(NATIVE_QUANTISED_PF_IQ4_NL) :
            format == NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN
                ? static_cast<uint8_t>(NATIVE_QUANTISED_PF_Q8_0)
                : static_cast<uint8_t>(format);
        if (!llaminar2::rocm::generated::selectROCmMoEGroupedPrefillGenerated(
                policy_codebook, role_id, m, n, k, generated))
        {
            return config;
        }
        config.rows = generated.tile_m;
        config.columns = generated.tile_n;
        // Measured gfx906 Wave64 zero-s_barrier (TILE_N=64) + int4 inline staged_sum_a
        // tile policy across all codebooks: avoids CTA starvation at small M (<=128)
        // and lane waste on narrow sharded expert widths (N<=256 / N<=640).
        if (m <= 128)
        {
            config.rows = 4;
            config.columns =
                (role == MoEPrefillProjectionRole::GateUp && n <= 640) ? 64 : 128;
        }
        else if (role == MoEPrefillProjectionRole::GateUp && n <= 256)
        {
            config.rows = 12;
            config.columns = 64;
        }
        else if (role == MoEPrefillProjectionRole::Down && k <= 256)
        {
            config.rows = 12;
            config.columns = 64;
        }
        else if ((role == MoEPrefillProjectionRole::GateUp && n <= 640) ||
                 (role == MoEPrefillProjectionRole::Down && k <= 640))
        {
            config.rows = 12;
            config.columns = 128;
        }
        moe_prefill_apply_tile_override(role, config);
        return config;
    }

    /**
     * @brief Select one joint production policy for a uniform-codebook pair.
     *
     * Exact production overlays have precedence over the complementary generic
     * role tables. A key miss is expected for an unseen geometry or positive M
     * and therefore retains the total generic policy. By contrast, a malformed
     * exact entry remains selected and invalid so the launch fails hard instead
     * of silently hiding a corrupt generated artifact.
     */
    MoEPrefillPairConfig moe_prefill_select_pair_config(
        llaminar2::MoEPrefillMeasuredOperation operation,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int m,
        int hidden_size,
        int expert_width,
        int expert_count,
        int top_k)
    {
        MoEPrefillPairConfig pair{};
        const int gateup_codebook =
            moe_prefill_single_codebook(gateup_codebook_mask);
        const int down_codebook =
            moe_prefill_single_codebook(down_codebook_mask);
        if (gateup_codebook < 0 || down_codebook < 0)
            return pair;

        pair.gateup = moe_prefill_select_tile_config(
            MoEPrefillProjectionRole::GateUp,
            static_cast<NativeQuantisedFormatPrefill>(gateup_codebook),
            m,
            expert_width,
            hidden_size);
        pair.down = moe_prefill_select_tile_config(
            MoEPrefillProjectionRole::Down,
            static_cast<NativeQuantisedFormatPrefill>(down_codebook),
            m,
            hidden_size,
            expert_width);

        llaminar2::rocm::generated::ROCmMoEProductionPrefillOverlayConfig
            exact{};
        const auto policy_codebook = [](int codebook)
        {
            return codebook == NATIVE_QUANTISED_PF_IQ4_XS_SOURCE ? static_cast<uint8_t>(NATIVE_QUANTISED_PF_IQ4_NL) :
                   codebook == NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN
                       ? static_cast<uint8_t>(NATIVE_QUANTISED_PF_Q8_0)
                       : static_cast<uint8_t>(codebook);
        };
        if (!llaminar2::rocm::selectROCmMoEProductionPrefillOverlayForOperation(
                operation,
                policy_codebook(gateup_codebook),
                policy_codebook(down_codebook),
                hidden_size,
                expert_width,
                expert_count,
                top_k,
                m,
                exact))
        {
            return pair;
        }

        pair.gateup = {
            .rows = exact.gateup_tile_m,
            .columns = exact.gateup_tile_n,
        };
        pair.down = {
            .rows = exact.down_tile_m,
            .columns = exact.down_tile_n,
        };
        pair.exact_overlay = true;
        moe_prefill_apply_tile_override(
            MoEPrefillProjectionRole::GateUp, pair.gateup);
        moe_prefill_apply_tile_override(
            MoEPrefillProjectionRole::Down, pair.down);
        return pair;
    }

    int moe_prefill_directory_capacity(
        int total_slots,
        int num_experts,
        int tile_rows)
    {
        const int active_expert_bound =
            total_slots < num_experts ? total_slots : num_experts;
        const int conservative_bound =
            active_expert_bound + (total_slots + tile_rows - 1) / tile_rows;
        return conservative_bound < total_slots
                   ? conservative_bound
                   : total_slots;
    }

    /**
     * @brief Read immutable compiler resources for one HIP kernel function.
     *
     * Candidate training calls this before the first launch.  In particular,
     * `localSizeBytes` is the runtime's authoritative static indication that
     * the compiler assigned per-thread scratch/local memory.  Such a candidate
     * is not eligible for timing because register spilling is a launch-policy
     * defect, not a performance feature for the learner to rediscover.
     */
    bool moe_prefill_query_function_resources(
        const void* function,
        int block_threads,
        int* registers_per_thread,
        size_t* local_memory_bytes_per_thread,
        size_t* static_shared_memory_bytes,
        int* max_threads_per_block,
        int* max_active_blocks_per_sm)
    {
        if (!function || block_threads <= 0 || !registers_per_thread ||
            !local_memory_bytes_per_thread || !static_shared_memory_bytes ||
            !max_threads_per_block || !max_active_blocks_per_sm)
        {
            return false;
        }

        hipFuncAttributes attributes{};
        if (hipFuncGetAttributes(&attributes, function) != hipSuccess)
            return false;

        int active_blocks = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                function,
                block_threads,
                /*dynamicSMemSize=*/0) != hipSuccess ||
            active_blocks <= 0)
        {
            return false;
        }

        *registers_per_thread = attributes.numRegs;
        *local_memory_bytes_per_thread = attributes.localSizeBytes;
        *static_shared_memory_bytes = attributes.sharedSizeBytes;
        *max_threads_per_block = attributes.maxThreadsPerBlock;
        *max_active_blocks_per_sm = active_blocks;
        return attributes.numRegs > 0 &&
               attributes.maxThreadsPerBlock >= block_threads;
    }

    /**
     * @brief Resolve one role-specific template specialization for inspection.
     */
    template <NativeQuantisedFormatPrefill FMT, int TILE_M, int TILE_N>
    bool moe_prefill_query_candidate_resources(
        MoEPrefillProjectionRole role,
        int* registers_per_thread,
        size_t* local_memory_bytes_per_thread,
        size_t* static_shared_memory_bytes,
        int* max_threads_per_block,
        int* max_active_blocks_per_sm)
    {
        const void* function = nullptr;
        if (role == MoEPrefillProjectionRole::GateUp)
        {
            function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<
                    FMT, TILE_M, TILE_N>));
        }
        else
        {
            function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                moe_grouped_down_batch_invariant_tiled_prefill_kernel<
                    FMT, TILE_M, TILE_N>));
        }
        return moe_prefill_query_function_resources(
            function,
            TILE_N,
            registers_per_thread,
            local_memory_bytes_per_thread,
            static_shared_memory_bytes,
            max_threads_per_block,
            max_active_blocks_per_sm);
    }

    /**
     * @brief Inspect every kernel embedded by one adaptive projection choice.
     *
     * Adaptive dispatch is a single capture-time policy choice, but its replay
     * consists of the device directory planner followed by the TM12 and TM16
     * projection specializations. The trainer must reject the choice when any
     * member spills or cannot sustain useful occupancy, so this query reports
     * the maximum pressure and minimum execution ceiling across all three
     * kernels. Reporting only one projection specialization would let an
     * uninspected kernel enter the production graph through a misleadingly
     * healthy candidate record.
     *
     * @tparam FMT NativeQuantised codebook used by both projection kernels.
     * @tparam TILE_N Output-column width and projection block size.
     * @param role Gate/up or down projection role.
     * @param registers_per_thread Receives the largest VGPR allocation.
     * @param local_memory_bytes_per_thread Receives the largest spill footprint.
     * @param static_shared_memory_bytes Receives the largest LDS allocation.
     * @param max_threads_per_block Receives the smallest launch ceiling.
     * @param max_active_blocks_per_sm Receives the smallest occupancy ceiling.
     * @return true only when the planner and both projections can be inspected.
     */
    template <NativeQuantisedFormatPrefill FMT, int TILE_N>
    bool moe_prefill_query_adaptive_candidate_resources(
        MoEPrefillProjectionRole role,
        int* registers_per_thread,
        size_t* local_memory_bytes_per_thread,
        size_t* static_shared_memory_bytes,
        int* max_threads_per_block,
        int* max_active_blocks_per_sm)
    {
        int registers[3] = {};
        size_t local_memory[3] = {};
        size_t shared_memory[3] = {};
        int max_threads[3] = {};
        int active_blocks[3] = {};

        if (!moe_prefill_query_candidate_resources<FMT, 12, TILE_N>(
                role,
                &registers[0],
                &local_memory[0],
                &shared_memory[0],
                &max_threads[0],
                &active_blocks[0]) ||
            !moe_prefill_query_candidate_resources<FMT, 16, TILE_N>(
                role,
                &registers[1],
                &local_memory[1],
                &shared_memory[1],
                &max_threads[1],
                &active_blocks[1]))
        {
            return false;
        }

        const void* planner =
            role == MoEPrefillProjectionRole::GateUp
                ? reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                      moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                          MoEPrefillProjectionRole::GateUp>))
                : reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                      moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                          MoEPrefillProjectionRole::Down>));
        if (!moe_prefill_query_function_resources(
                planner,
                kMoEPrefillMaxExperts,
                &registers[2],
                &local_memory[2],
                &shared_memory[2],
                &max_threads[2],
                &active_blocks[2]))
        {
            return false;
        }

        *registers_per_thread = registers[0];
        *local_memory_bytes_per_thread = local_memory[0];
        *static_shared_memory_bytes = shared_memory[0];
        *max_threads_per_block = max_threads[0];
        *max_active_blocks_per_sm = active_blocks[0];
        for (int member = 1; member < 3; ++member)
        {
            if (registers[member] > *registers_per_thread)
                *registers_per_thread = registers[member];
            if (local_memory[member] > *local_memory_bytes_per_thread)
                *local_memory_bytes_per_thread = local_memory[member];
            if (shared_memory[member] > *static_shared_memory_bytes)
                *static_shared_memory_bytes = shared_memory[member];
            if (max_threads[member] < *max_threads_per_block)
                *max_threads_per_block = max_threads[member];
            if (active_blocks[member] < *max_active_blocks_per_sm)
                *max_active_blocks_per_sm = active_blocks[member];
        }
        return true;
    }
}


namespace llaminar2::rocm::moe_prefill_shards {
struct MoERouteLeafSymbol { const void *function; const char *name; int threads; size_t dynamic_lds; };

struct AdaptiveDirectoryView {
    uint32_t* header = nullptr;
    uint32_t* tile12 = nullptr;
    uint32_t* tile16 = nullptr;
    int tile12_capacity = 0;
    int tile16_capacity = 0;
    [[nodiscard]] bool valid() const noexcept {
        return header && tile12 && tile16 && tile12_capacity > 0 && tile16_capacity > 0;
    }
};
} // namespace llaminar2::rocm::moe_prefill_shards
using llaminar2::rocm::moe_prefill_shards::MoERouteLeafSymbol;
using llaminar2::rocm::moe_prefill_shards::AdaptiveDirectoryView;

namespace
{
/** @brief Exact loaded leaf identity and its native launch-owned staging extent. */
// MoERouteLeafSymbol defined in llaminar2::rocm::moe_prefill_shards

/** @brief Resolve only compiled physical decoder tags, without canonicalizing source aliases. */
template <class Select>
auto selectMoERouteOwnedFormat(uint8_t codebook, Select select)
{
    switch (codebook)
    {
    case NATIVE_QUANTISED_PF_Q4_0: return select.template operator()<NATIVE_QUANTISED_PF_Q4_0>();
    case NATIVE_QUANTISED_PF_IQ4_NL: return select.template operator()<NATIVE_QUANTISED_PF_IQ4_NL>();
    case NATIVE_QUANTISED_PF_IQ4_XS_SOURCE: return select.template operator()<NATIVE_QUANTISED_PF_IQ4_XS_SOURCE>();
    case NATIVE_QUANTISED_PF_Q4_1: return select.template operator()<NATIVE_QUANTISED_PF_Q4_1>();
    case NATIVE_QUANTISED_PF_Q5_0: return select.template operator()<NATIVE_QUANTISED_PF_Q5_0>();
    case NATIVE_QUANTISED_PF_Q5_1: return select.template operator()<NATIVE_QUANTISED_PF_Q5_1>();
    case NATIVE_QUANTISED_PF_Q6_K: return select.template operator()<NATIVE_QUANTISED_PF_Q6_K>();
    case NATIVE_QUANTISED_PF_Q3_K: return select.template operator()<NATIVE_QUANTISED_PF_Q3_K>();
    case NATIVE_QUANTISED_PF_Q2_K: return select.template operator()<NATIVE_QUANTISED_PF_Q2_K>();
    case NATIVE_QUANTISED_PF_IQ3_S: return select.template operator()<NATIVE_QUANTISED_PF_IQ3_S>();
    case NATIVE_QUANTISED_PF_IQ3_XXS: return select.template operator()<NATIVE_QUANTISED_PF_IQ3_XXS>();
    case NATIVE_QUANTISED_PF_IQ2_S: return select.template operator()<NATIVE_QUANTISED_PF_IQ2_S>();
    case NATIVE_QUANTISED_PF_IQ2_XS: return select.template operator()<NATIVE_QUANTISED_PF_IQ2_XS>();
    case NATIVE_QUANTISED_PF_IQ2_XXS: return select.template operator()<NATIVE_QUANTISED_PF_IQ2_XXS>();
    case NATIVE_QUANTISED_PF_IQ1_S: return select.template operator()<NATIVE_QUANTISED_PF_IQ1_S>();
    case NATIVE_QUANTISED_PF_IQ1_M: return select.template operator()<NATIVE_QUANTISED_PF_IQ1_M>();
    case NATIVE_QUANTISED_PF_Q8_0: return select.template operator()<NATIVE_QUANTISED_PF_Q8_0>();
    case NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN: return select.template operator()<NATIVE_QUANTISED_PF_EXPANDED_INT8_MIN>();
    default: throw std::invalid_argument("unsupported physical ROCm route-owned native decoder");
    }
}

/**
 * @brief Resolve the exact shared producer symbol and launch-owned partial-bank size before capture.
 * @tparam FMT Actual prepared decoder.
 * @tparam TN Independent-column width; explicit four-plane execution requires64.
 * @tparam PARALLEL_SOURCE_PARTITIONS Whether four workers compute unchanged original partitions.
 * @param maximum_partitions Authenticated original source-policy maximum; unused by serial resources.
 * @return Actual native function, block width and exact dynamic LDS requirement.
 */
template <NativeQuantisedFormatPrefill FMT, int TN, bool PARALLEL_SOURCE_PARTITIONS = false>
MoERouteLeafSymbol routeGateUpSymbol(int maximum_partitions)
{
    static_assert(!PARALLEL_SOURCE_PARTITIONS ||
        (TN == llaminar2::rocm::kMoERouteOwnedGateUpPartitionColumns));
    return {reinterpret_cast<const void *>(HIP_KERNEL_NAME(
        moe_grouped_gate_up_decode_equivalent_prefill_kernel<FMT, TN, FMT == NATIVE_QUANTISED_PF_IQ2_S, 1, PARALLEL_SOURCE_PARTITIONS>)),
        __PRETTY_FUNCTION__, TN * (PARALLEL_SOURCE_PARTITIONS ? llaminar2::rocm::kMoERouteOwnedGateUpPartitionWorkers : 1),
        llaminar2::rocm::moeRouteOwnedGateUpK1SharedLayout(static_cast<std::uint8_t>(FMT),
            PARALLEL_SOURCE_PARTITIONS
                ? llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes
                : llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution::Serial,
            maximum_partitions).total_bytes};
}

/**
 * @brief Resolve one exact scoped mode without substituting any decoder or geometry.
 * @tparam FMT Actual physical decoder validated against the bounded candidate domain.
 * @param geometry Independent output-column geometry.
 * @param partition_execution Serial source program or explicit four full worker planes.
 * @param maximum_partitions Existing source-policy maximum, required for exact parallel resources.
 * @return Exact compiled leaf identity and immutable launch-owned LDS.
 * @throws std::invalid_argument For an unsupported explicit mode/decoder/geometry.
 */
template <NativeQuantisedFormatPrefill FMT>
MoERouteLeafSymbol routeGateUpSymbolForGeometry(llaminar2::rocm::MoERouteOwnedGateUpGeometry geometry,
    llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution partition_execution, int maximum_partitions)
{
    using G = llaminar2::rocm::MoERouteOwnedGateUpGeometry;
    using P = llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution;
    if (partition_execution == P::FourWorkerPlanes)
    {
        if (geometry != G::TN64) throw std::invalid_argument("parallel source partitions support only TN64");
        return routeGateUpSymbol<FMT, 64, true>(maximum_partitions);
    }
    if (partition_execution != P::Serial) throw std::invalid_argument("unknown route-owned partition execution");
    switch (geometry)
    {
    case G::TN64: return routeGateUpSymbol<FMT, 64>(maximum_partitions);
    case G::TN128: return routeGateUpSymbol<FMT, 128>(maximum_partitions);
    case G::TN256: return routeGateUpSymbol<FMT, 256>(maximum_partitions);
    }
    throw std::invalid_argument("unknown route-owned producer geometry");
}

/** @brief Inspect the exact format-owned 64-column publication leaf, never an expert-tiled proxy. */
template <NativeQuantisedFormatPrefill FMT>
MoERouteLeafSymbol routeDownSymbol(llaminar2::MoEPrefillOutputLayout layout,
    llaminar2::rocm::MoECanonicalDownExecution execution)
{
    switch (layout)
    {
    case llaminar2::MoEPrefillOutputLayout::OriginalRouteRows:
        if constexpr (FMT == NATIVE_QUANTISED_PF_Q5_1)
        {
            if (execution == llaminar2::rocm::MoECanonicalDownExecution::FourWorkers)
                return {reinterpret_cast<const void *>(HIP_KERNEL_NAME(
                    moe_grouped_down_publish_canonical_routes_prefill_kernel<static_cast<int>(FMT), 4>)),
                    __PRETTY_FUNCTION__, 256, 0};
        }
        return {reinterpret_cast<const void *>(HIP_KERNEL_NAME(
            moe_grouped_down_publish_canonical_routes_prefill_kernel<static_cast<int>(FMT), 1>)),
            __PRETTY_FUNCTION__, 64, 0};
    case llaminar2::MoEPrefillOutputLayout::TokenRows:
        return {reinterpret_cast<const void *>(HIP_KERNEL_NAME(
            moe_grouped_down_publish_original_token_ordered_prefill_kernel<static_cast<int>(FMT)>)),
            __PRETTY_FUNCTION__, 64, 0};
    }
    throw std::invalid_argument("unknown route-owned down publication layout");
}

/** @brief Require cold resource ownership before selecting a device or querying native metadata. */
void requireMoERouteOwnedColdInspection(int device)
{
    if (device < 0) throw std::invalid_argument("route-owned native inspection requires an exact device");
    if (llaminar2::isGraphCaptureActive())
        throw std::logic_error("route-owned resources must be inspected before graph capture");
    if (llaminar2::HipDeviceGuard::setDevice(device) != 0)
        throw std::runtime_error("route-owned resource inspection could not select device");
}

/** @brief Admit one exact leaf with its real dynamic codebook LDS and no private storage. */
llaminar2::rocm::MoERouteOwnedKernelResources inspectMoERouteLeaf(MoERouteLeafSymbol symbol)
{
    hipFuncAttributes attributes{};
    int active = 0;
    if (!symbol.function || !symbol.name || symbol.threads <= 0 ||
        hipFuncGetAttributes(&attributes, symbol.function) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&active, symbol.function,
            symbol.threads, symbol.dynamic_lds) != hipSuccess ||
        attributes.numRegs <= 0 || attributes.localSizeBytes != 0 ||
        attributes.maxThreadsPerBlock < symbol.threads || active <= 0)
        throw std::runtime_error("route-owned exact-leaf resource admission failed");
    return {.function = symbol.function, .specialization = symbol.name,
        .block_threads = symbol.threads, .registers_per_thread = attributes.numRegs,
        .local_memory_bytes_per_thread = attributes.localSizeBytes,
        .static_shared_memory_bytes = attributes.sharedSizeBytes,
        .dynamic_shared_memory_bytes = symbol.dynamic_lds,
        .max_threads_per_block = attributes.maxThreadsPerBlock,
        .max_active_blocks_per_sm = active};
}
} // namespace


namespace llaminar2::rocm::moe_prefill_shards {

struct GateUpLaunchContext {
    hipStream_t hip_stream;
    const int8_t* act_int8;
    const float* act_scales;
    const int* d_group_token_indices;
    const int* d_group_counts;
    const int* d_group_offsets;
    void* d_group_work_directory;
    const DeviceNativeQuantisedMatrixDesc* gate_descs;
    const DeviceNativeQuantisedMatrixDesc* up_descs;
    float* d_scratch_gate;
    float* d_scratch_up;
    int total_slots;
    int seq_len;
    int top_k;
    bool grouped_indices_are_route_slots;
    int intermediate;
    int d_model;
    MoEPrefillSerialM1KPartitionPolicy gateup_k_partition_policy;
    int num_experts;
    ROCmMoEGroupedRouteAdmission admission;
    const llaminar2::rocm::MoERouteOwnedGateUpAdmission* launch_proof;
    void* submission_observation_ptr;
    uint32_t gateup_codebook_mask;
    bool use_route_owned;
    bool use_expert_row_tiles;
    bool native_wmma;
    bool has_uniform_codebook_pair;
    MoEPrefillTileConfig pair_gateup;
    llaminar2::rocm::MoEGroupedPrefillSubmission* pending;
    void* cb_ctx;
    int (*ensure_work_directory)(void*, int);
    AdaptiveDirectoryView (*build_adaptive_work_directory)(void*, MoEPrefillProjectionRole);
};

struct RouteOwnedDownLaunchContext {
    hipStream_t hip_stream;
    const int8_t* d_scratch_swiglu_int8;
    const float* d_scratch_swiglu_scales;
    const DeviceNativeQuantisedMatrixDesc* down_descs;
    const int* d_original_to_grouped;
    const int* d_original_expert_ids;
    const float* projection_weights;
    const float* d_group_weights;
    float* d_canonical_route_contributions;
    float* d_output;
    int total_slots;
    int seq_len;
    int top_k;
    int down_columns;
    int intermediate;
    int num_experts;
    MoEPrefillSerialM1KPartitionPolicy down_k_partition_policy;
    ROCmMoEGroupedRouteAdmission admission;
    bool canonical_publication;
    uint32_t down_codebook_mask;
};

struct ExpertTiledDownLaunchContext {
    hipStream_t hip_stream;
    const int8_t* d_scratch_swiglu_int8;
    const float* d_scratch_swiglu_scales;
    const DeviceNativeQuantisedMatrixDesc* down_descs;
    const int* d_group_counts;
    const int* d_group_offsets;
    void* d_group_work_directory;
    float* d_scratch_gate;
    int seq_len;
    int down_columns;
    int intermediate;
    int num_experts;
    MoEPrefillSerialM1KPartitionPolicy down_k_partition_policy;
    ROCmMoEGroupedRouteAdmission admission;
    uint32_t down_codebook_mask;
    bool has_uniform_codebook_pair;
    MoEPrefillTileConfig pair_down;
    llaminar2::rocm::MoEGroupedPrefillSubmission* pending;
    void* cb_ctx;
    int (*ensure_work_directory)(void*, int);
    AdaptiveDirectoryView (*build_adaptive_work_directory)(void*, MoEPrefillProjectionRole);
};

#define LLAMINAR_DECL_MOE_PF_SHARD(S) \
    bool routeGateUpSymbol_s##S(uint8_t codebook, llaminar2::rocm::MoERouteOwnedGateUpGeometry geometry, \
        llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution partition_execution, int maximum_partitions, MoERouteLeafSymbol& out); \
    bool routeDownSymbol_s##S(uint8_t codebook, llaminar2::MoEPrefillOutputLayout layout, \
        llaminar2::rocm::MoECanonicalDownExecution execution, MoERouteLeafSymbol& out); \
    bool queryResources_s##S(uint8_t codebook_id, MoEPrefillProjectionRole role, int tile_m, int tile_n, \
        int* registers_per_thread, size_t* local_memory_bytes_per_thread, size_t* static_shared_memory_bytes, \
        int* max_threads_per_block, int* max_active_blocks_per_sm); \
    bool launchGateUp_s##S(const GateUpLaunchContext& ctx, bool& launched); \
    bool launchRouteOwnedDown_s##S(const RouteOwnedDownLaunchContext& ctx, bool& matched); \
    bool launchDown_s##S(const ExpertTiledDownLaunchContext& ctx, bool& launched); \
    bool initIQ_s##S(int device_id, const void* h_iq3s, const void* h_iq3xxs, const void* h_iq2s, \
        const void* h_iq2xs, const void* h_iq2xxs, const void* h_iq1s);

LLAMINAR_DECL_MOE_PF_SHARD(0)
LLAMINAR_DECL_MOE_PF_SHARD(1)
LLAMINAR_DECL_MOE_PF_SHARD(2)
LLAMINAR_DECL_MOE_PF_SHARD(3)
LLAMINAR_DECL_MOE_PF_SHARD(4)
LLAMINAR_DECL_MOE_PF_SHARD(5)
#undef LLAMINAR_DECL_MOE_PF_SHARD

} // namespace llaminar2::rocm::moe_prefill_shards

#if LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0

llaminar2::rocm::MoERouteOwnedGateUpPlan
llaminar2::rocm::planMoERouteOwnedGateUp(
    MoERouteOwnedGateUpGeometry geometry, const llaminar2::NativeQuantisedExecutionFormatEnvelope &envelope, int device,
    MoERouteOwnedGateUpPartitionExecution partition_execution, MoERouteOwnedProjectionShape shape)
{
    (void)moeRouteOwnedGateUpColumns(geometry);
    if (!moeRouteOwnedGateUpPartitionExecutionValid(partition_execution, envelope, shape) ||
        (partition_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes && geometry != MoERouteOwnedGateUpGeometry::TN64))
        throw std::invalid_argument("route-owned partition execution requires its exact supported envelope, TN and N/K");
    if (device < 0) throw std::invalid_argument("route-owned logical plan requires an exact device");
    if (llaminar2::isGraphCaptureActive())
        throw std::logic_error("route-owned partition policy must be admitted before graph capture");
    requireMoERouteOwnedUnmodifiedPolicy();
    std::array<std::uint8_t, 32> partitions{};
    int maximum_partitions = 0;
    if (shape.reduction_columns > 0)
    {
        MoEPrefillSerialM1KPartitionPolicy resolved{};
        // Invoke the original resolver with original source-policy bits and the
        // same physical gate/up N/K as raw production lowering, never physical23.
        if (!moe_prefill_resolve_serial_m1_k_partitions(envelope.policy_codebook_mask,
                shape.output_columns, shape.reduction_columns, resolved))
            throw std::invalid_argument("route-owned admission could not resolve the original source partitions");
        std::copy(std::begin(resolved.by_codebook), std::end(resolved.by_codebook), partitions.begin());
        maximum_partitions = moeRouteOwnedGateUpMaximumKPartitions(envelope, partitions, shape.reduction_columns);
        if (maximum_partitions <= 0)
            throw std::invalid_argument("route-owned admission received an incomplete original partition policy");
    }
    (void)moeRouteOwnedGateUpPlanIdentity(geometry, envelope, device, partition_execution, shape, partitions);
    return {geometry, envelope, device, partition_execution, shape, partitions, maximum_partitions};
}

llaminar2::rocm::MoERouteOwnedGateUpAdmission
llaminar2::rocm::admitMoERouteOwnedGateUpGeometry(
    MoERouteOwnedGateUpGeometry geometry, const llaminar2::NativeQuantisedExecutionFormatEnvelope &envelope, int device,
    MoERouteOwnedGateUpPartitionExecution partition_execution, MoERouteOwnedProjectionShape shape)
{
    return admitMoERouteOwnedGateUpGeometry(planMoERouteOwnedGateUp(geometry, envelope, device, partition_execution, shape));
}

llaminar2::rocm::MoERouteOwnedGateUpAdmission
llaminar2::rocm::admitMoERouteOwnedGateUpGeometry(const MoERouteOwnedGateUpPlan &plan)
{
    const auto geometry = plan.geometry();
    const auto &envelope = plan.executionEnvelope();
    const auto device = plan.deviceOrdinal();
    const auto partition_execution = plan.partitionExecution();
    const auto shape = plan.shape();
    const auto &partitions = plan.arithmeticPartitions();
    const auto maximum_partitions = plan.maximumKPartitions();
    (void)plan.identity();
    requireMoERouteOwnedUnmodifiedPolicy();
    const auto launch_mask = moeRouteOwnedGateUpLaunchCodebookMask(envelope.execution_codebook_mask);
    std::array<MoERouteLeafSymbol, 32> symbols{};
    // Resolve every family before the first HIP query: unsupported members
    // cannot leave a partially admitted capability or skip an empty leaf.
    for (std::uint8_t codebook = 0; codebook < 32; ++codebook)
        if (launch_mask & llaminar2::nativeQuantisedCodebookMaskBit(codebook))
            {
                using namespace llaminar2::rocm::moe_prefill_shards;
                MoERouteLeafSymbol sym{};
                const bool ok =
                    routeGateUpSymbol_s0(codebook, geometry, partition_execution, maximum_partitions, sym) ||
                    routeGateUpSymbol_s1(codebook, geometry, partition_execution, maximum_partitions, sym) ||
                    routeGateUpSymbol_s2(codebook, geometry, partition_execution, maximum_partitions, sym) ||
                    routeGateUpSymbol_s3(codebook, geometry, partition_execution, maximum_partitions, sym) ||
                    routeGateUpSymbol_s4(codebook, geometry, partition_execution, maximum_partitions, sym) ||
                    routeGateUpSymbol_s5(codebook, geometry, partition_execution, maximum_partitions, sym);
                if (!ok) throw std::invalid_argument("route-owned gate/up codebook is unsupported");
                symbols[codebook] = sym;
            }
    requireMoERouteOwnedColdInspection(device);
    std::array<MoERouteOwnedKernelResources, 32> resources{};
    for (std::uint8_t codebook = 0; codebook < 32; ++codebook)
        if (launch_mask & llaminar2::nativeQuantisedCodebookMaskBit(codebook))
            resources[codebook] = inspectMoERouteLeaf(symbols[codebook]);
    return {geometry, envelope, device, resources, partition_execution, shape, partitions, maximum_partitions};
}

llaminar2::rocm::MoERouteOwnedGateUpAdmission
llaminar2::rocm::admitMoERouteOwnedGateUpGeometry(
    MoERouteOwnedGateUpGeometry geometry, uint8_t codebook, int device)
{
    const auto bit = llaminar2::nativeQuantisedCodebookMaskBit(codebook);
    return admitMoERouteOwnedGateUpGeometry(geometry,
        llaminar2::NativeQuantisedExecutionFormatEnvelope{bit, bit}, device);
}

llaminar2::rocm::MoERouteOwnedKernelResources
llaminar2::rocm::inspectMoERouteOwnedDown(uint8_t codebook,
    MoEPrefillOutputLayout layout, MoECanonicalDownShape shape, int device)
{
    const auto execution = selectMoECanonicalDownExecution(shape);
    MoERouteLeafSymbol symbol{};
    {
        using namespace llaminar2::rocm::moe_prefill_shards;
        const bool ok =
            routeDownSymbol_s0(codebook, layout, execution, symbol) ||
            routeDownSymbol_s1(codebook, layout, execution, symbol) ||
            routeDownSymbol_s2(codebook, layout, execution, symbol) ||
            routeDownSymbol_s3(codebook, layout, execution, symbol) ||
            routeDownSymbol_s4(codebook, layout, execution, symbol) ||
            routeDownSymbol_s5(codebook, layout, execution, symbol);
        if (!ok) throw std::invalid_argument("route-owned down codebook is unsupported");
    }
    requireMoERouteOwnedColdInspection(device);
    return inspectMoERouteLeaf(symbol);
}

extern "C" {

/**
 * @brief Query the generated tile selected for one long-prefill projection.
 *
 * This host-only introspection entry point shares the production selector,
 * including explicit trainer overrides. It performs no GPU work and is used by
 * PerfStats and integration tests to prove that the policy advertised for a
 * graph is the policy its capture launcher selected. M<=8 direct execution is
 * intentionally handled by the caller because it bypasses this generated
 * long-prefill surface.
 *
 * @param codebook_id NativeQuantised execution codebook stored in the descriptor.
 * @param projection_role Zero for gate/up, one for down.
 * @param m Number of original token rows.
 * @param n Projection output width.
 * @param k Projection reduction width.
 * @param tile_m Receives the trained expert-row tile.
 * @param tile_n Receives the trained output-column tile.
 * @return true when the generated table contains a valid compiled candidate;
 *         false for invalid arguments, unsupported codebooks, or a table miss.
 */
bool rocmMoE_grouped_prefill_query_tile_config(
    uint8_t codebook_id,
    int projection_role,
    int m,
    int n,
    int k,
    int* tile_m,
    int* tile_n)
{
    if (!tile_m || !tile_n || projection_role < 0 || projection_role > 1)
        return false;
    const auto role = projection_role == 0
                          ? MoEPrefillProjectionRole::GateUp
                          : MoEPrefillProjectionRole::Down;
    const MoEPrefillTileConfig config = moe_prefill_select_tile_config(
        role,
        static_cast<NativeQuantisedFormatPrefill>(codebook_id),
        m,
        n,
        k);
    if (!moe_prefill_valid_tile_rows(config.rows) ||
        !moe_prefill_valid_tile_columns(config.columns))
    {
        return false;
    }
    *tile_m = config.rows;
    *tile_n = config.columns;
    return true;
}

/**
 * @brief Query the qualified capture-time policy for one production MoE pair.
 *
 * Unlike the role-only trainer query, this API has every runtime key used by
 * the exact overlay. It therefore reports the same joint gate/up and down
 * geometry that the production launcher will embed in a captured graph. An
 * unseen exact key remains valid through the complementary generic tables and
 * reports @p exact_overlay as zero.
 *
 * @param operation Actual phase/publication/grouping qualification.
 * @param gateup_codebook_id Uniform gate/up execution codebook.
 * @param down_codebook_id Uniform down execution codebook.
 * @param m Number of original token rows.
 * @param hidden_size Model hidden width.
 * @param expert_width Routed expert intermediate width.
 * @param expert_count Number of routed experts.
 * @param top_k Routes selected per token.
 * @param gateup_tile_m Receives the gate/up expert-row tile.
 * @param gateup_tile_n Receives the gate/up output-column tile.
 * @param down_tile_m Receives the down expert-row tile.
 * @param down_tile_n Receives the down output-column tile.
 * @param exact_overlay Receives one for an exact overlay and zero for generic.
 * @return true when both selected configurations name compiled candidates.
 */
bool rocmMoE_grouped_prefill_query_production_pair_config_for_operation(
    llaminar2::MoEPrefillMeasuredOperation operation,
    uint8_t gateup_codebook_id,
    uint8_t down_codebook_id,
    int m,
    int hidden_size,
    int expert_width,
    int expert_count,
    int top_k,
    int *gateup_tile_m,
    int *gateup_tile_n,
    int *down_tile_m,
    int *down_tile_n,
    int *exact_overlay)
{
    if (gateup_codebook_id >= 32 || down_codebook_id >= 32 ||
        m <= 0 || hidden_size <= 0 || expert_width <= 0 ||
        expert_count <= 0 || top_k <= 0 || top_k > expert_count ||
        !gateup_tile_m || !gateup_tile_n || !down_tile_m || !down_tile_n ||
        !exact_overlay)
    {
        return false;
    }

    const MoEPrefillPairConfig pair = moe_prefill_select_pair_config(
        operation,
        uint32_t{1} << gateup_codebook_id,
        uint32_t{1} << down_codebook_id,
        m,
        hidden_size,
        expert_width,
        expert_count,
        top_k);
    if (!moe_prefill_valid_tile_rows(pair.gateup.rows) ||
        !moe_prefill_valid_tile_columns(pair.gateup.columns) ||
        !moe_prefill_valid_tile_rows(pair.down.rows) ||
        !moe_prefill_valid_tile_columns(pair.down.columns))
    {
        return false;
    }

    *gateup_tile_m = pair.gateup.rows;
    *gateup_tile_n = pair.gateup.columns;
    *down_tile_m = pair.down.rows;
    *down_tile_n = pair.down.columns;
    *exact_overlay = pair.exact_overlay ? 1 : 0;
    return true;
}

/**
 * @brief Preserve the existing complete-operation trainer query ABI.
 * @details Old trainer cells always execute complete(full H), token rows and
 *          ordinary unmasked grouping. Phase callers must use the typed query.
 * @param gateup_codebook_id Existing uniform gate/up execution codebook.
 * @param down_codebook_id Existing uniform down execution codebook.
 * @param m Captured token rows.
 * @param hidden_size Original hidden width.
 * @param expert_width Complete intermediate width.
 * @param expert_count Logical expert descriptor count.
 * @param top_k Routes per token.
 * @param gateup_tile_m Receives selected gate/up row tile.
 * @param gateup_tile_n Receives selected gate/up column tile.
 * @param down_tile_m Receives selected down row tile.
 * @param down_tile_n Receives selected down column tile.
 * @param exact_overlay Receives one only for a measured complete exact row.
 * @return Whether the unchanged selected generic/exact geometry is compiled.
 */
extern "C" bool rocmMoE_grouped_prefill_query_production_pair_config(
    uint8_t gateup_codebook_id, uint8_t down_codebook_id,
    int m, int hidden_size, int expert_width, int expert_count, int top_k,
    int *gateup_tile_m, int *gateup_tile_n, int *down_tile_m, int *down_tile_n,
    int *exact_overlay)
{
    return rocmMoE_grouped_prefill_query_production_pair_config_for_operation(
        llaminar2::MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
        gateup_codebook_id, down_codebook_id, m, hidden_size, expert_width,
        expert_count, top_k, gateup_tile_m, gateup_tile_n, down_tile_m,
        down_tile_n, exact_overlay);
}

/**
 * @brief Query static resources for one compiled grouped-prefill candidate.
 *
 * This inspection performs no allocation, launch, transfer, or synchronization.
 * It is the pre-timing eligibility gate used by the exhaustive trainer.  A
 * successful query with non-zero `local_memory_bytes_per_thread` identifies a
 * register-spilling specialization and therefore excludes that specialization
 * from the launch tournament.
 *
 * @param codebook_id NativeQuantised execution codebook compiled into the decoder.
 * @param projection_role Zero for gate/up, one for down.
 * @param tile_m Candidate expert-row tile.
 * @param tile_n Candidate output-column tile and physical block width.
 * @param registers_per_thread Receives compiler-assigned VGPR usage.
 * @param local_memory_bytes_per_thread Receives compiler-owned scratch bytes.
 * @param static_shared_memory_bytes Receives compiler-owned LDS bytes.
 * @param max_threads_per_block Receives the kernel's launch ceiling.
 * @param max_active_blocks_per_sm Receives occupancy at this block geometry.
 * @return true only when the exact compiled specialization can be inspected.
 */
bool rocmMoE_grouped_prefill_query_kernel_resources(
    uint8_t codebook_id,
    int projection_role,
    int tile_m,
    int tile_n,
    int* registers_per_thread,
    size_t* local_memory_bytes_per_thread,
    size_t* static_shared_memory_bytes,
    int* max_threads_per_block,
    int* max_active_blocks_per_sm)
{
    if (projection_role < 0 || projection_role > 1 ||
        !moe_prefill_valid_tile_rows(tile_m) ||
        !moe_prefill_valid_tile_columns(tile_n))
    {
        return false;
    }

    const auto role = projection_role == 0
                          ? MoEPrefillProjectionRole::GateUp
                          : MoEPrefillProjectionRole::Down;

    using namespace llaminar2::rocm::moe_prefill_shards;
    if (queryResources_s0(codebook_id, role, tile_m, tile_n, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm)) return true;
    if (queryResources_s1(codebook_id, role, tile_m, tile_n, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm)) return true;
    if (queryResources_s2(codebook_id, role, tile_m, tile_n, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm)) return true;
    if (queryResources_s3(codebook_id, role, tile_m, tile_n, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm)) return true;
    if (queryResources_s4(codebook_id, role, tile_m, tile_n, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm)) return true;
    if (queryResources_s5(codebook_id, role, tile_m, tile_n, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm)) return true;
    return false;
}

/**
 * @brief Record the declared exact projection transaction on one HIP stream.
 *
 * Every pointer is persistent caller-owned storage. Gate/up publishes existing
 * grouped INT8 rows and FP32 scales; down consumes those same bits and matching
 * grouping, and writes compact columns in original top-k order. A split caller
 * owns the explicit event or collective edge between these transactions. No
 * intermediate CPU observation, temporary allocation or synchronization occurs.
 * @param execution Immutable phase, output slice and full-width arithmetic key.
 * @return False on invalid bindings, unsupported geometry or HIP launch error.
 */
bool rocmMoE_grouped_prefill_pipeline(
    // Input
    const float*  d_hidden,              // [seq_len, d_model] FP32 hidden states
    const int8_t* d_prequantized_hidden, // optional [seq_len, d_model] router-owned Q8 rows
    const float*  d_prequantized_hidden_scales, // optional [seq_len, d_model / 32]
    // Weight descriptors (already on device)
    const void*   d_gate_desc_table,     // DeviceNativeQuantisedMatrixDesc[num_experts]
    const void*   d_up_desc_table,       // DeviceNativeQuantisedMatrixDesc[num_experts]
    const void*   d_down_desc_table,     // DeviceNativeQuantisedMatrixDesc[num_experts]
    // Grouping data (from prepareExpertGroupsAsync)
    const int*    d_group_counts,        // [num_experts] per-expert token count
    const int*    d_group_offsets,       // [num_experts] prefix-sum offsets
    const int*    d_group_token_indices, // [total_slots] original token indices
    const int*    d_original_to_grouped,  // [seq_len * top_k] required route slot -> grouped slot
    const int*    d_original_expert_ids,  // [seq_len * top_k] required unless num_experts == 1
    const float*  d_group_weights,       // [total_slots] routing weights
    int*          d_group_work_directory,// dedicated header + TM12/TM16 device spans
    // Scratch buffers (caller provides, sized for total_slots)
    int8_t*       d_scratch_A_int8,      // [total_slots, max(d_model, intermediate)]
    float*        d_scratch_scales,      // [total_slots, max_blocks_per_row]
    float*        d_scratch_gate,        // [total_slots, max(d_model, intermediate)]
    float*        d_scratch_up,          // [total_slots, intermediate]
    int8_t*       d_scratch_swiglu_int8, // [total_slots, intermediate]
    float*        d_scratch_swiglu_scales,// [total_slots, blocks_per_intermediate]
    // Output
    float*        d_output,              // [seq_len, d_model] overwritten in original top-k order
    float*        d_canonical_route_contributions, // optional [seq_len, top_k, d_model]
    // Dimensions
    int num_experts,
    int d_model,
    int intermediate,
    int total_slots,                     // seq_len * top_k
    int top_k,                           // routes per token, needed by ordered publication
    int grouped_indices_are_route_slots, // d_group_token_indices stores route slots, not token rows
    // Format masks launch only the compiled NativeQuantised specializations present
    // in each descriptor table. Mixed-codebook expert tables remain supported.
    uint32_t gateup_codebook_mask,
    uint32_t down_codebook_mask,
    // Policy masks retain the original source codebooks after representation
    // normalization (for example compact Q5_1 promoted to expanded INT8+min).
    uint32_t gateup_policy_codebook_mask,
    uint32_t down_policy_codebook_mask,
    int device_id,
    void* stream,
    const llaminar2::MoEPrefillProjectionExecution& execution,
    llaminar2::MoEPrefillRouteScope route_scope,
    const llaminar2::rocm::MoERouteOwnedGateUpAdmission *explicit_admission,
    llaminar2::rocm::MoERouteOwnedGateUpLaunchSet *completed_observation,
    uint32_t *d_population_selection,
    llaminar2::rocm::MoEGroupedPrefillSubmission *submitted,
    llaminar2::rocm::MoEGroupedSourceTuple singleton_sources,
    const llaminar2::MoEPrefillGroupingProvenance::DirectoryBorrow *grouping_directory,
    llaminar2::rocm::MoEGroupedSourceTuple routed_sources)
{
    if (submitted) *submitted = {};
    if (completed_observation) *completed_observation = {};
    if ((explicit_admission != nullptr) != (completed_observation != nullptr))
        throw std::logic_error("model route-owned lowering requires one capability and one caller-owned observation");
    const auto *launch_proof = resolveMoERouteOwnedAdmission(explicit_admission);
    llaminar2::rocm::validateMoERouteOwnedGateUpScopeBinding(device_id, stream,
        top_k > 0 && total_slots > 0 && total_slots % top_k == 0 ? total_slots / top_k : -1,
        execution, {.output_columns = intermediate, .reduction_columns = d_model}, explicit_admission);
    llaminar2::rocm::validateMoERouteOwnedGateUpCodebook(gateup_codebook_mask, gateup_policy_codebook_mask, explicit_admission);
    std::optional<MoERouteOwnedSubmissionObservation> submission_observation;
    if (launch_proof) submission_observation.emplace(*launch_proof);
    const bool executes_gateup = execution.executesGateUp();
    const bool executes_down = execution.executesDown();
    const int down_columns = execution.columnCount();
    if (num_experts <= 0 || num_experts > kMoEPrefillMaxExperts ||
        static_cast<uint32_t>(total_slots) > llaminar2::moe_grouped_directory::kMaximumRows ||
        d_model <= 0 || intermediate <= 0 ||
        execution.modelColumns() != d_model ||
        total_slots <= 0 || top_k <= 0 || (total_slots % top_k) != 0 ||
        (d_model % 32) != 0 || (intermediate % 32) != 0 ||
        gateup_codebook_mask == 0 || down_codebook_mask == 0 ||
        gateup_policy_codebook_mask == 0 ||
        down_policy_codebook_mask == 0 || !stream)
        return false;
    if (!d_group_counts || !d_group_offsets || !d_group_work_directory ||
        !d_scratch_swiglu_int8 || !d_scratch_swiglu_scales ||
        (executes_gateup && (!d_hidden || !d_gate_desc_table || !d_up_desc_table ||
            !d_group_token_indices || !d_scratch_A_int8 || !d_scratch_scales ||
            !d_scratch_gate || !d_scratch_up)) ||
        (executes_down && (!d_down_desc_table || !d_original_to_grouped ||
            !d_group_weights || !d_scratch_gate ||
            (!d_output && !d_canonical_route_contributions) ||
            (!d_original_expert_ids && num_experts != 1))))
        return false;
    const int seq_len = total_slots / top_k;
    const bool canonical_publication =
        d_canonical_route_contributions != nullptr;
    const bool raw_cohort =
        execution.routeWeighting() == llaminar2::MoEExpertRouteWeighting::CompletedCohort;
    if (raw_cohort && (!canonical_publication || d_output))
        return false;
    const float *const projection_weights = raw_cohort ? nullptr : d_group_weights;
    const auto measured_operation = execution.measuredOperation(
        canonical_publication ? llaminar2::MoEPrefillOutputLayout::OriginalRouteRows
                              : llaminar2::MoEPrefillOutputLayout::TokenRows,
        route_scope);
    const bool has_uniform_codebook_pair =
        moe_prefill_single_codebook(gateup_codebook_mask) >= 0 &&
        moe_prefill_single_codebook(down_codebook_mask) >= 0;
    const int gateup_codebook =
        moe_prefill_single_codebook(gateup_codebook_mask);
    const int down_codebook =
        moe_prefill_single_codebook(down_codebook_mask);
    const llaminar2::rocm::ROCmMoEGroupedPrefillRouteKey route_key{
            .gateup_codebook = static_cast<uint8_t>(gateup_codebook),
            .down_codebook = static_cast<uint8_t>(down_codebook),
            .hidden_size = d_model,
            .expert_width = intermediate,
            .expert_count = num_experts,
            .top_k = top_k,
            .rows = seq_len,
            .operation = measured_operation,
        };
    const auto route_decision =
        llaminar2::rocm::selectROCmMoEGroupedPrefillRouteStrategy(route_key);
    if (!route_decision.valid())
        return false;
    // Retain every family reachable by the device's group publication. A
    // padded M=16 verifier with four live rows must not inherit the M=16
    // prefill schedule. Both families borrow one predicate and one scratch
    // arena; only the selected family writes on each replay.
    const ROCmMoEGroupedRouteAdmission admission(
        route_key, d_group_counts, d_group_offsets);
    const bool use_route_owned =
        admission.maySelect(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
    const bool use_expert_row_tiles =
        admission.maySelect(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    if (launch_proof && (!use_route_owned || use_expert_row_tiles))
        throw std::logic_error("route-owned diagnostic requires only its actual M1 route-owned family");
    const MoEPrefillPairConfig pair_config =
        use_expert_row_tiles && has_uniform_codebook_pair
            ? moe_prefill_select_pair_config(
                  measured_operation,
                  gateup_codebook_mask,
                  down_codebook_mask,
                  seq_len,
                  d_model,
                  intermediate,
                  num_experts,
                  top_k)
            : MoEPrefillPairConfig{};
    if (use_expert_row_tiles && has_uniform_codebook_pair &&
        (!moe_prefill_valid_tile_rows(pair_config.gateup.rows) ||
         !moe_prefill_valid_tile_columns(pair_config.gateup.columns) ||
         !moe_prefill_valid_tile_rows(pair_config.down.rows) ||
         !moe_prefill_valid_tile_columns(pair_config.down.columns)))
    {
        return false;
    }

    MoEPrefillSerialM1KPartitionPolicy gateup_k_partition_policy{};
    MoEPrefillSerialM1KPartitionPolicy down_k_partition_policy{};
    if (!moe_prefill_resolve_serial_m1_k_partitions(
            gateup_policy_codebook_mask,
            intermediate,
            d_model,
            gateup_k_partition_policy) ||
        !moe_prefill_resolve_serial_m1_k_partitions(
            down_policy_codebook_mask,
            d_model,
            intermediate,
            down_k_partition_policy))
    {
        return false;
    }

    if (launch_proof && launch_proof->reductionWidth() > 0)
    {
        const auto &proof = *launch_proof;
        if (proof.outputWidth() != intermediate || proof.reductionWidth() != d_model ||
            !std::equal(std::begin(gateup_k_partition_policy.by_codebook), std::end(gateup_k_partition_policy.by_codebook),
                proof.arithmeticPartitions().begin()))
            throw std::logic_error("route-owned raw preparation changed its sealed original N/K partition policy");
    }

    if (llaminar2::HipDeviceGuard::setDevice(device_id) != hipSuccess) return false;
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (!hip_stream || hip_stream == hipStreamLegacy || hip_stream == hipStreamPerThread) return false;
    const llaminar2::rocm::MoEGroupedPopulationKey population_key{
        .operation = measured_operation,
        .input = grouped_indices_are_route_slots ? llaminar2::rocm::MoEGroupedPopulationInput::OriginalRouteSlots
                                                : llaminar2::rocm::MoEGroupedPopulationInput::OriginalTokens,
        .preparation = d_prequantized_hidden && d_prequantized_hidden_scales
            ? llaminar2::rocm::MoEGroupedPopulationPreparation::PublishedQ8
            : llaminar2::rocm::MoEGroupedPopulationPreparation::Internal,
        .rows = seq_len, .hidden = d_model, .intermediate = intermediate, .experts = num_experts, .top_k = top_k,
        .gateup_physical = gateup_codebook_mask, .down_physical = down_codebook_mask,
        .gateup_source = gateup_policy_codebook_mask, .down_source = down_policy_codebook_mask,
        .singleton_sources = singleton_sources, .routed_sources = routed_sources};
    const auto dot_policy = llaminar2::rocm::selectedMoEGroupedPrefillPolicy(device_id,population_key);
    llaminar2::rocm::MoEGroupedPrefillSubmission pending;
    pending.policy = dot_policy;
    pending.exact_ordered_overlay = pair_config.exact_overlay;
    const auto dot_engine = dot_policy.engine;
    const bool column_slots=dot_policy.directory==llaminar2::rocm::MoEGroupedDirectorySource::GroupingColumnSlots;
    const bool grouping_offset_slots = dot_policy.directory != llaminar2::rocm::MoEGroupedDirectorySource::RolePlanner;
    const auto packed_layout=column_slots ? llaminar2::MoEGroupingPackedLayout::ColumnFloor8 : llaminar2::MoEGroupingPackedLayout::WideFloor8;
    if (grouping_offset_slots)
    {
        if (!grouping_directory || dot_engine != llaminar2::rocm::MoEGroupedPrefillEngine::WideDot4Wmma ||
            dot_policy.minimum_expert_rows != llaminar2::rocm::MoEGroupedWmmaRowFloor::Eight ||
            measured_operation != llaminar2::MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes ||
            grouped_indices_are_route_slots || d_model != 2048 || intermediate != 512 ||
            (seq_len != 9 && seq_len != 16) || num_experts != 256 || top_k != 8 ||
            (column_slots && (seq_len!=16 || gateup_codebook_mask!=(uint32_t{1}<<13) ||
                down_codebook_mask!=(uint32_t{1}<<24) || gateup_policy_codebook_mask!=(uint32_t{1}<<13) ||
                down_policy_codebook_mask!=(uint32_t{1}<<4))))
            throw std::logic_error("grouping offset slots require the complete original static routed candidate");
        (void)grouping_directory->requireProjection(d_group_counts, d_group_offsets,
            reinterpret_cast<uint32_t *>(d_group_work_directory), seq_len, top_k, num_experts,
            llaminar2::DeviceId::rocm(device_id), stream,packed_layout);
    }
    else if (grouping_directory && grouping_directory->present())
        throw std::logic_error("a role planner cannot overwrite a live grouping directory publication");
    const bool population_adaptive = dot_engine == llaminar2::rocm::MoEGroupedPrefillEngine::PopulationAdaptive;
    const bool native_wmma = dot_engine != llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4;
    if (population_adaptive && !d_population_selection) return false;
    std::optional<MoEGroupedPopulationControl> population_control;
    if (population_adaptive) population_control.emplace(MoEGroupedPopulationControl{MoEGroupedPopulationBinding(d_population_selection)});
    const bool compact_population = population_adaptive &&
        dot_policy.population == llaminar2::rocm::MoEGroupedPopulationDomain::Compact;
    using CompactPopulation = llaminar2::rocm::MoEGroupedCompactPopulationTransaction;
    std::optional<CompactPopulation> compact_transaction;
    std::optional<CompactPopulation::GateUpSubmission> compact_produced;
    if (compact_population)
        compact_transaction.emplace(population_key,llaminar2::rocm::MoEGroupedPopulationWorkspace{
            d_population_selection,reinterpret_cast<uint32_t *>(d_group_work_directory),
            llaminar2::MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(total_slots,num_experts)},
            d_group_counts,d_group_offsets,device_id,stream);
    const auto make_dot_admission = [&](llaminar2::rocm::MoEGroupedPrefillEngine engine,
        llaminar2::rocm::MoEGroupedWmmaRowFloor floor, MoEPrefillProjectionRole role,
        int source_n, int physical_n, int k)
    {
        if (compact_population)
            return MoEGroupedDotAdmission::nativeCandidate(role,source_n,physical_n,k,
                llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert);
        if(column_slots)return MoEGroupedDotAdmission::columnCandidate(role,source_n,physical_n,k);
        return engine == llaminar2::rocm::MoEGroupedPrefillEngine::WideDot4Wmma
            ? MoEGroupedDotAdmission::wideCandidate(role,source_n,physical_n,k,floor)
            : MoEGroupedDotAdmission::nativeCandidate(role,source_n,physical_n,k,floor);
    };
    (void)hipGetLastError(); // clear prior errors
    const bool has_prequantized_hidden =
        d_prequantized_hidden && d_prequantized_hidden_scales;
    if ((d_prequantized_hidden == nullptr) !=
        (d_prequantized_hidden_scales == nullptr))
        return false;
    const bool reuse_prequantized_hidden = has_prequantized_hidden;

    std::optional<MoEGroupedRowPlan> directory_plan;
    int current_directory_capacity = 0;
    const auto ensure_work_directory = [&](int tile_rows) -> int
    {
        if (!moe_prefill_valid_tile_rows(tile_rows))
            return 0;
        const auto plan = MoEGroupedRowPlan::fixed(tile_rows);

        const int capacity = moe_prefill_directory_capacity(
            total_slots, num_experts, tile_rows);
        if (directory_plan && *directory_plan == plan)
            return current_directory_capacity;

        /*
         * One thread owns one descriptor-table entry in the inclusive scan.
         * Launch the full representable width so inactive tail entries are
         * explicitly zero and the final scan element always contains the
         * complete tile count, including expert id 512 and above.
         */
        dim3 directory_block(llaminar2::moe_grouped_directory::directoryThreads(num_experts));
        switch (tile_rows)
        {
        case 4:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<4>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity, admission, plan);
            break;
        case 8:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<8>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity, admission, plan);
            break;
        case 12:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<12>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity, admission, plan);
            break;
        case 16:
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<16>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                reinterpret_cast<uint32_t*>(d_group_work_directory),
                num_experts,
                capacity, admission, plan);
            break;
        default:
            return 0;
        }

        if (hipGetLastError() != hipSuccess) return 0;
        directory_plan = plan;
        current_directory_capacity = capacity;
        return capacity;
    };

    /** Borrow the consumer's complete count-only row plan for directory publication. */
    const auto ensure_native_directory = [&](const MoEGroupedDotAdmission &dot) -> int
    {
        const auto plan = dot.rowPlan();
        if (grouping_offset_slots)
        {
            if (plan != (column_slots ? MoEGroupedRowPlan::column(plan.role()) :
                MoEGroupedRowPlan::wide(plan.role(), llaminar2::rocm::MoEGroupedWmmaRowFloor::Eight)))
                throw std::logic_error("offset-slot publication cannot serve an incompatible native row plan");
            return grouping_directory->requireProjection(d_group_counts, d_group_offsets,
                reinterpret_cast<uint32_t *>(d_group_work_directory), seq_len, top_k, num_experts,
                llaminar2::DeviceId::rocm(device_id), stream,packed_layout).totalSlots();
        }
        if (!plan.isWide()) return ensure_work_directory(16);
        if (directory_plan && *directory_plan == plan) return current_directory_capacity;
        const int capacity = moe_prefill_directory_capacity(total_slots,num_experts,plan.minimumRows());
        if (capacity <= 0) return 0;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_build_prefill_tile_directory_kernel<0>),
            dim3(1),dim3(llaminar2::moe_grouped_directory::directoryThreads(num_experts)),0,hip_stream,
            d_group_counts,reinterpret_cast<uint32_t*>(d_group_work_directory),num_experts,
            capacity,admission,plan);
        if (hipGetLastError() != hipSuccess) return 0;
        directory_plan = plan;
        current_directory_capacity = capacity;
        return capacity;
    };

    /** Reserve the producer's conservative span; the population shard fills it on the same stream. */
    const auto population_directory_capacity = [&](MoEPrefillProjectionRole role) -> int
    {
        directory_plan.reset();
        current_directory_capacity = 0;
        return compact_population ? CompactPopulation::kDirectoryCapacity :
            moe_prefill_directory_capacity(total_slots,num_experts,
                role == MoEPrefillProjectionRole::GateUp ? 12 : 16);
    };

    // AdaptiveDirectoryView defined in llaminar2::rocm::moe_prefill_shards

    const auto build_adaptive_work_directory = [&] (
        MoEPrefillProjectionRole role) -> AdaptiveDirectoryView
    {
        const int tile12_capacity = moe_prefill_directory_capacity(
            total_slots, num_experts, /*minimum adaptive tile=*/12);
        const int tile16_capacity =
            role == MoEPrefillProjectionRole::GateUp
                ? (total_slots < num_experts ? total_slots : num_experts)
                : moe_prefill_directory_capacity(
                      total_slots,
                      num_experts,
                      /*maximum adaptive tile=*/16);
        auto* const header =
            reinterpret_cast<uint32_t*>(d_group_work_directory);
        AdaptiveDirectoryView view{
            .header = header,
            .tile12 = header + kMoEPrefillAdaptiveDirectoryHeaderWords,
            .tile16 = header + kMoEPrefillAdaptiveDirectoryHeaderWords +
                      tile12_capacity,
            .tile12_capacity = tile12_capacity,
            .tile16_capacity = tile16_capacity,
        };
        if (!view.valid())
            return {};

        dim3 directory_block(llaminar2::moe_grouped_directory::directoryThreads(num_experts));
        if (role == MoEPrefillProjectionRole::GateUp)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                        MoEPrefillProjectionRole::GateUp>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                header,
                num_experts,
                tile12_capacity,
                tile16_capacity, admission);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    moe_grouped_build_adaptive_prefill_tile_directory_kernel<
                        MoEPrefillProjectionRole::Down>),
                dim3(1), directory_block, 0, hip_stream,
                d_group_counts,
                header,
                num_experts,
                tile12_capacity,
                tile16_capacity, admission);
        }
        if (hipGetLastError() != hipSuccess) return {};
        directory_plan.reset();
        current_directory_capacity = 0;
        return view;
    };

    const int blocks_per_d_model = d_model / 32;

    // =========================================================================
    // Kernel 1: Quantize each original hidden row once.
    //
    // A verifier router already quantizes every original row for its Q8 logits
    // GEMV.  When those device rows are published, gate/up addresses them
    // through d_group_token_indices and this launch is unnecessary.  Ordinary
    // prefill and non-Q8 routing use the same row-owned Q8 contract here. The
    // grouped projection resolves every expert slot back to this original row.
    // =========================================================================
    if (executes_gateup && !reuse_prequantized_hidden)
    {
        // Keep tiny decode/verifier grids distributed across CUs. Larger
        // prefill already supplies ample waves; packing eight Q8 blocks cuts
        // workgroup scheduling by 8x without changing a single quantized byte.
        const unsigned threads = seq_len <= 16 ? 32u : 256u;
        // Keep rows on a separate grid axis: large valid row/width products
        // must not silently truncate a flattened 32-bit grid dimension.
        dim3 grid((blocks_per_d_model + threads / 32 - 1) / (threads / 32), seq_len);
        dim3 block(threads);
        hipLaunchKernelGGL(moe_prefill_hidden_quantize_rows_kernel,
            grid, block, 0, hip_stream,
            d_hidden, d_scratch_A_int8, d_scratch_scales, seq_len, d_model);
    }

    // =========================================================================
    // Kernel 2: Batch-invariant gate/up projection.
    //
    // Verifier buckets retain the low-latency route-owned geometry. Long prefill
    // consumes the compact device directory and amortizes weight decode across
    // rows. Trained expert-row tiles preserve every row's serial-M1 partition
    // boundaries and ordered partial reduction.
    // =========================================================================
    if (executes_gateup)
    {
        const auto* gate_descs = static_cast<const DeviceNativeQuantisedMatrixDesc*>(d_gate_desc_table);
        const auto* up_descs = static_cast<const DeviceNativeQuantisedMatrixDesc*>(d_up_desc_table);

        struct DirCallbacks {
            decltype(ensure_work_directory)* ensure_fn;
            decltype(build_adaptive_work_directory)* adaptive_fn;
        } dir_cb{&ensure_work_directory, &build_adaptive_work_directory};
        const llaminar2::rocm::moe_prefill_shards::GateUpLaunchContext gu_ctx{
            .hip_stream = hip_stream,
            .act_int8 = reuse_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8,
            .act_scales = reuse_prequantized_hidden ? d_prequantized_hidden_scales : d_scratch_scales,
            .d_group_token_indices = d_group_token_indices,
            .d_group_counts = d_group_counts,
            .d_group_offsets = d_group_offsets,
            .d_group_work_directory = d_group_work_directory,
            .gate_descs = gate_descs,
            .up_descs = up_descs,
            .d_scratch_gate = d_scratch_gate,
            .d_scratch_up = d_scratch_up,
            .total_slots = total_slots,
            .seq_len = seq_len,
            .top_k = top_k,
            .grouped_indices_are_route_slots = (grouped_indices_are_route_slots != 0),
            .intermediate = intermediate,
            .d_model = d_model,
            .gateup_k_partition_policy = gateup_k_partition_policy,
            .num_experts = num_experts,
            .admission = admission,
            .launch_proof = launch_proof,
            .submission_observation_ptr = submission_observation ? static_cast<void*>(&*submission_observation) : nullptr,
            .gateup_codebook_mask = gateup_codebook_mask,
            .use_route_owned = use_route_owned,
            .use_expert_row_tiles = use_expert_row_tiles,
            .native_wmma = native_wmma,
            .has_uniform_codebook_pair = has_uniform_codebook_pair,
            .pair_gateup = pair_config.gateup,
            .pending = &pending,
            .cb_ctx = &dir_cb,
            .ensure_work_directory = [](void* p, int tm) -> int { return (*static_cast<DirCallbacks*>(p)->ensure_fn)(tm); },
            .build_adaptive_work_directory = [](void* p, MoEPrefillProjectionRole r) -> AdaptiveDirectoryView { return (*static_cast<DirCallbacks*>(p)->adaptive_fn)(r); },
        };
        bool launched_gateup = false;
        {
            using namespace llaminar2::rocm::moe_prefill_shards;
            if (!launchGateUp_s0(gu_ctx, launched_gateup) ||
                !launchGateUp_s1(gu_ctx, launched_gateup) ||
                !launchGateUp_s2(gu_ctx, launched_gateup) ||
                !launchGateUp_s3(gu_ctx, launched_gateup) ||
                !launchGateUp_s4(gu_ctx, launched_gateup) ||
                !launchGateUp_s5(gu_ctx, launched_gateup))
                return false;
        }
        if (!launched_gateup)
        {
            fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported gate/up codebook mask=0x%x\n",
                    gateup_codebook_mask);
            return false;
        }

        if (use_expert_row_tiles && native_wmma)
        {
            const auto engine = population_adaptive ? llaminar2::rocm::MoEGroupedPrefillEngine::WideDot4Wmma : dot_engine;
            const auto gateup_dot_admission = make_dot_admission(engine,dot_policy.minimum_expert_rows,
                MoEPrefillProjectionRole::GateUp,intermediate,intermediate,d_model);
            llaminar2::rocm::MoEGroupedPrefillRoleSubmission native_submitted;
            const int capacity = population_adaptive ? population_directory_capacity(MoEPrefillProjectionRole::GateUp)
                : ensure_native_directory(gateup_dot_admission);
            if (capacity <= 0) return false;
            const llaminar2::rocm::MoEGroupedWmmaArguments args = {
                .activations = reuse_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8,
                .activation_scales = reuse_prequantized_hidden ? d_prequantized_hidden_scales : d_scratch_scales,
                .group_token_indices = d_group_token_indices,
                .group_counts = d_group_counts, .group_offsets = d_group_offsets,
                .directory = reinterpret_cast<const uint32_t *>(d_group_work_directory),
                .descriptors = gate_descs, .second_descriptors = up_descs,
                .output = d_scratch_gate, .second_output = d_scratch_up,
                .directory_capacity = capacity, .rows = seq_len, .top_k = top_k,
                .n = intermediate, .k = d_model, .experts = num_experts, .device = device_id,
                .input = grouped_indices_are_route_slots
                    ? llaminar2::rocm::MoEGroupedWmmaInput::OriginalRouteSlots
                    : llaminar2::rocm::MoEGroupedWmmaInput::OriginalTokens,
                .partitions = gateup_k_partition_policy, .admission = admission,
                .dot_admission = gateup_dot_admission, .directory_layout=packed_layout};
            if (compact_population)
            {
                auto produced = compact_transaction->gateUp(gateup_codebook_mask,args);
                if (!produced) return false;
                native_submitted = produced->receipt();
                compact_produced.emplace(std::move(*produced));
            }
            else
            {
                const bool accepted = population_adaptive
                    ? llaminar2::rocm::submitMoEGroupedPopulation(MoEPrefillProjectionRole::GateUp,
                        {d_population_selection,reinterpret_cast<uint32_t *>(d_group_work_directory),
                         llaminar2::MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(total_slots,num_experts)},
                        gateup_codebook_mask,args,*population_control,stream,&native_submitted)
                    : llaminar2::rocm::submitMoEGroupedWmma(engine,gateup_codebook_mask,args,stream,&native_submitted,grouping_directory);
                if (!accepted) return false;
            }
            for (size_t i = 0; i < native_submitted.size; ++i)
                pending.gateup.record(native_submitted.tiles[i]);
        }

        constexpr int blocks_per_workgroup = kMoESwiGLUQuantizeThreads / 32;
        const size_t quant_blocks = static_cast<size_t>(total_slots) *
            (intermediate / 32);
        dim3 quant_grid((quant_blocks + blocks_per_workgroup - 1) /
                        blocks_per_workgroup);
        dim3 quant_block(kMoESwiGLUQuantizeThreads);
        hipLaunchKernelGGL(
            moe_grouped_swiglu_quantize_prefill_kernel,
            quant_grid, quant_block, 0, hip_stream,
            d_scratch_gate, d_scratch_up,
            d_scratch_swiglu_int8, d_scratch_swiglu_scales,
            intermediate, admission);
    }

    // =========================================================================
    // Kernels 4-5: down projection and fixed-order publication.
    //
    // Tiny verifier buckets compute and publish directly. Long prefill writes
    // row-tiled down partials into the now-dead gate scratch, then one owner per
    // token/column reduces those rows in original top-k order.
    // =========================================================================
    if (executes_down)
    {
        constexpr int TILE_N = 64;
        dim3 block(TILE_N);
        const auto* down_descs =
            static_cast<const DeviceNativeQuantisedMatrixDesc*>(d_down_desc_table);

        if (use_route_owned)
        {
            /*
             * The codebook mask is immutable graph identity. A one-bit mask
             * therefore selects one fully inlined down decoder during capture;
             * only a table that deliberately contains multiple formats enters
             * the mixed-table specialization. Both routes execute the same
             * ordered publication kernels and differ solely in decoder dispatch.
             */
            #define LAUNCH_CANONICAL_DOWN_WORKERS(FORMAT_TAG, WORKERS) \
                do { \
                    using CanonicalGeometry = MoECanonicalRouteDownGeometry<FORMAT_TAG, WORKERS>; \
                    dim3 grid((down_columns + CanonicalGeometry::columns - 1) / CanonicalGeometry::columns, \
                              std::min(total_slots, kCanonicalPublicationSlotLanes)); \
                    hipLaunchKernelGGL( \
                        HIP_KERNEL_NAME(moe_grouped_down_publish_canonical_routes_prefill_kernel<FORMAT_TAG, WORKERS>), \
                        grid, dim3(CanonicalGeometry::threads), 0, hip_stream, \
                        d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                        d_original_to_grouped, d_original_expert_ids, projection_weights, \
                        d_canonical_route_contributions, total_slots, down_columns, \
                        intermediate, num_experts, down_k_partition_policy, admission); \
                } while (0)

            #define LAUNCH_ROUTE_OWNED_DOWN(FORMAT_TAG) \
                do { \
                    if (canonical_publication) { \
                        const auto execution = llaminar2::rocm::selectMoECanonicalDownExecution( \
                            {down_columns, intermediate, total_slots}); \
                        [&]<int SelectedFormat>() { \
                            if constexpr (SelectedFormat == static_cast<int>(NATIVE_QUANTISED_PF_Q5_1)) { \
                                if (execution == llaminar2::rocm::MoECanonicalDownExecution::FourWorkers) { \
                                    LAUNCH_CANONICAL_DOWN_WORKERS(SelectedFormat, 4); \
                                } else { \
                                    LAUNCH_CANONICAL_DOWN_WORKERS(SelectedFormat, 1); \
                                } \
                            } else { \
                                LAUNCH_CANONICAL_DOWN_WORKERS(SelectedFormat, 1); \
                            } \
                        }.template operator()<FORMAT_TAG>(); \
                    } else { \
                        dim3 grid((down_columns + TILE_N - 1) / TILE_N, \
                                  std::min(seq_len, kCanonicalPublicationSlotLanes)); \
                        hipLaunchKernelGGL( \
                            HIP_KERNEL_NAME(moe_grouped_down_publish_original_token_ordered_prefill_kernel<FORMAT_TAG>), \
                            grid, block, 0, hip_stream, \
                            d_scratch_swiglu_int8, d_scratch_swiglu_scales, down_descs, \
                            d_original_to_grouped, d_original_expert_ids, d_group_weights, \
                            d_output, seq_len, top_k, down_columns, intermediate, num_experts, \
                            down_k_partition_policy, admission); \
                    } \
                } while (0)

            const llaminar2::rocm::moe_prefill_shards::RouteOwnedDownLaunchContext rd_ctx{
                .hip_stream = hip_stream,
                .d_scratch_swiglu_int8 = d_scratch_swiglu_int8,
                .d_scratch_swiglu_scales = d_scratch_swiglu_scales,
                .down_descs = down_descs,
                .d_original_to_grouped = d_original_to_grouped,
                .d_original_expert_ids = d_original_expert_ids,
                .projection_weights = projection_weights,
                .d_group_weights = d_group_weights,
                .d_canonical_route_contributions = d_canonical_route_contributions,
                .d_output = d_output,
                .total_slots = total_slots,
                .seq_len = seq_len,
                .top_k = top_k,
                .down_columns = down_columns,
                .intermediate = intermediate,
                .num_experts = num_experts,
                .down_k_partition_policy = down_k_partition_policy,
                .admission = admission,
                .canonical_publication = canonical_publication,
                .down_codebook_mask = down_codebook_mask,
            };
            bool matched_route_down = false;
            {
                using namespace llaminar2::rocm::moe_prefill_shards;
                if (!launchRouteOwnedDown_s0(rd_ctx, matched_route_down) ||
                    !launchRouteOwnedDown_s1(rd_ctx, matched_route_down) ||
                    !launchRouteOwnedDown_s2(rd_ctx, matched_route_down) ||
                    !launchRouteOwnedDown_s3(rd_ctx, matched_route_down) ||
                    !launchRouteOwnedDown_s4(rd_ctx, matched_route_down) ||
                    !launchRouteOwnedDown_s5(rd_ctx, matched_route_down))
                    return false;
            }
            if (!matched_route_down)
            {
                if (down_codebook_mask == 0 ||
                    (down_codebook_mask & (down_codebook_mask - 1u)) == 0)
                {
                    fprintf(stderr,
                            "[rocmMoE_grouped_prefill_pipeline] invalid route-owned down codebook mask=0x%x\n",
                            down_codebook_mask);
                    return false;
                }
                LAUNCH_ROUTE_OWNED_DOWN(kMoEPrefillMixedCodebookTag);
            }
            #undef LAUNCH_ROUTE_OWNED_DOWN
            #undef LAUNCH_CANONICAL_DOWN_WORKERS
            pending.down_route_owned = true;
        }
        if (use_expert_row_tiles)
        {
            bool launched_down = false;
            if (!native_wmma)
            {
                struct DownDirCallbacks {
                    decltype(ensure_work_directory)* ensure_fn;
                    decltype(build_adaptive_work_directory)* adaptive_fn;
                } down_cb{&ensure_work_directory, &build_adaptive_work_directory};
                const llaminar2::rocm::moe_prefill_shards::ExpertTiledDownLaunchContext ed_ctx{
                    .hip_stream = hip_stream,
                    .d_scratch_swiglu_int8 = d_scratch_swiglu_int8,
                    .d_scratch_swiglu_scales = d_scratch_swiglu_scales,
                    .down_descs = down_descs,
                    .d_group_counts = d_group_counts,
                    .d_group_offsets = d_group_offsets,
                    .d_group_work_directory = d_group_work_directory,
                    .d_scratch_gate = d_scratch_gate,
                    .seq_len = seq_len,
                    .down_columns = down_columns,
                    .intermediate = intermediate,
                    .num_experts = num_experts,
                    .down_k_partition_policy = down_k_partition_policy,
                    .admission = admission,
                    .down_codebook_mask = down_codebook_mask,
                    .has_uniform_codebook_pair = has_uniform_codebook_pair,
                    .pair_down = pair_config.down,
                    .pending = &pending,
                    .cb_ctx = &down_cb,
                    .ensure_work_directory = [](void* p, int tm) -> int { return (*static_cast<DownDirCallbacks*>(p)->ensure_fn)(tm); },
                    .build_adaptive_work_directory = [](void* p, MoEPrefillProjectionRole r) -> AdaptiveDirectoryView { return (*static_cast<DownDirCallbacks*>(p)->adaptive_fn)(r); },
                };
                using namespace llaminar2::rocm::moe_prefill_shards;
                if (!launchDown_s0(ed_ctx, launched_down) ||
                    !launchDown_s1(ed_ctx, launched_down) ||
                    !launchDown_s2(ed_ctx, launched_down) ||
                    !launchDown_s3(ed_ctx, launched_down) ||
                    !launchDown_s4(ed_ctx, launched_down) ||
                    !launchDown_s5(ed_ctx, launched_down))
                    return false;
            }
            if (native_wmma)
            {
                    const auto engine = population_adaptive ? llaminar2::rocm::MoEGroupedPrefillEngine::WideDot4Wmma : dot_engine;
                const auto down_dot_admission = make_dot_admission(engine,dot_policy.minimum_expert_rows,
                    MoEPrefillProjectionRole::Down,d_model,down_columns,intermediate);
                llaminar2::rocm::MoEGroupedPrefillRoleSubmission native_submitted;
                const int capacity = population_adaptive ? population_directory_capacity(MoEPrefillProjectionRole::Down)
                    : ensure_native_directory(down_dot_admission);
                if (capacity <= 0) return false;
                const llaminar2::rocm::MoEGroupedWmmaArguments args = {
                    .activations = d_scratch_swiglu_int8,
                    .activation_scales = d_scratch_swiglu_scales,
                    .group_counts = d_group_counts, .group_offsets = d_group_offsets,
                    .directory = reinterpret_cast<const uint32_t *>(d_group_work_directory),
                    .descriptors = down_descs, .output = d_scratch_gate,
                    .directory_capacity = capacity, .rows = seq_len, .top_k = top_k,
                    .n = down_columns, .k = intermediate, .experts = num_experts, .device = device_id,
                    .input = llaminar2::rocm::MoEGroupedWmmaInput::GroupedRows,
                    .partitions = down_k_partition_policy, .admission = admission,
                    .dot_admission = down_dot_admission, .directory_layout=packed_layout};
                if (compact_population)
                {
                    if (!compact_produced) throw std::logic_error("Compact down has no current GU producer");
                    auto submitted = compact_transaction->down(std::move(*compact_produced),down_codebook_mask,args);
                    if (!submitted) return false;
                    native_submitted = *submitted;
                }
                else
                {
                    const bool accepted = population_adaptive
                        ? llaminar2::rocm::submitMoEGroupedPopulation(MoEPrefillProjectionRole::Down,
                        {d_population_selection,reinterpret_cast<uint32_t *>(d_group_work_directory),
                         llaminar2::MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(total_slots,num_experts)},
                        down_codebook_mask,args,*population_control,stream,&native_submitted)
                        : llaminar2::rocm::submitMoEGroupedWmma(engine,down_codebook_mask,args,stream,&native_submitted,grouping_directory);
                    if (!accepted) return false;
                }
                for (size_t i = 0; i < native_submitted.size; ++i)
                    pending.down.record(native_submitted.tiles[i]);
                launched_down = true;
            }

            if (!launched_down)
            {
                fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] unsupported down codebook mask=0x%x\n",
                        down_codebook_mask);
                return false;
            }

            if (canonical_publication)
            {
                constexpr int publish_columns = 256;
                dim3 publish_grid(
                    (down_columns + publish_columns - 1) / publish_columns,
                    std::min(total_slots, kCanonicalPublicationSlotLanes));
                hipLaunchKernelGGL(
                    moe_grouped_down_partials_canonical_publish_prefill_kernel,
                    publish_grid, dim3(publish_columns), 0, hip_stream,
                    d_scratch_gate,
                    d_original_to_grouped,
                    projection_weights,
                    d_canonical_route_contributions,
                    total_slots,
                    down_columns, admission);
            }
            else
            {
                dim3 publish_grid(
                    (down_columns + TILE_N - 1) / TILE_N,
                    seq_len);
                hipLaunchKernelGGL(
                    moe_grouped_down_partials_ordered_publish_prefill_kernel,
                    publish_grid, block, 0, hip_stream,
                    d_scratch_gate, d_original_to_grouped, d_group_weights,
                    d_output, seq_len, top_k, down_columns, admission);
            }
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmMoE_grouped_prefill_pipeline] kernel launch error: %s\n",
                hipGetErrorString(err));
        return false;
    }

    if (submission_observation)
    {
        const auto &completed = submission_observation->complete();
        if (completed_observation) *completed_observation = completed;
        else g_moe_route_launch = completed; // Explicit diagnostic compatibility only.
    }
    pending.state = llaminar2::rocm::MoEGroupedPrefillSubmission::State::Complete;
    if (submitted) *submitted = pending;
    return true;
}

bool rocmMoE_reduce_canonical_route_contributions(
    const float* d_route_contributions,
    float* d_output,
    int seq_len,
    int top_k,
    int d_model,
    int device_id,
    void* stream)
{
    if (!d_route_contributions || !d_output || !stream ||
        seq_len <= 0 || top_k <= 0 || d_model <= 0)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);
    constexpr int kThreads = 256;
    dim3 grid((d_model + kThreads - 1) / kThreads, seq_len);
    hipLaunchKernelGGL(
        moe_reduce_canonical_route_contributions_kernel,
        grid,
        dim3(kThreads),
        0,
        static_cast<hipStream_t>(stream),
        d_route_contributions,
        d_output,
        seq_len,
        top_k,
        d_model);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmMoE_reduce_canonical_route_contributions] launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// IQ grid table initialization for this TU.
// Must be called once per device before launching any IQ grid format kernel.
// ============================================================================
bool rocmInitIQGridTables_moe_prefill(
    int device_id,
    const void* h_iq3s_grid,
    const void* h_iq3xxs_grid,
    const void* h_iq2s_grid,
    const void* h_iq2xs_grid,
    const void* h_iq2xxs_grid,
    const void* h_iq1s_grid)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipError_t e;
    e = hipMemcpyToSymbol(d_pf_iq3s_grid, h_iq3s_grid, sizeof(d_pf_iq3s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq3s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq3xxs_grid, h_iq3xxs_grid, sizeof(d_pf_iq3xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq3xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq2s_grid, h_iq2s_grid, sizeof(d_pf_iq2s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq2s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq2xs_grid, h_iq2xs_grid, sizeof(d_pf_iq2xs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq2xs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq2xxs_grid, h_iq2xxs_grid, sizeof(d_pf_iq2xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq2xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_pf_iq1s_grid, h_iq1s_grid, sizeof(d_pf_iq1s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables_moe_prefill] iq1s failed: %s\n", hipGetErrorString(e)); return false; }
    using namespace llaminar2::rocm::moe_prefill_shards;
    if (!initIQ_s0(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid) ||
        !initIQ_s1(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid) ||
        !initIQ_s2(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid) ||
        !initIQ_s3(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid) ||
        !initIQ_s4(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid) ||
        !initIQ_s5(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid))
        return false;
    return llaminar2::rocm::initializeMoEGroupedWmmaGrids(device_id,
        h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid) &&
        llaminar2::rocm::initializeMoEGroupedPopulationGrids(device_id,h_iq2s_grid);
}

} // extern "C"

#include "ROCmGroupedFFNFrontier.hip.inc"


#else // LLAMINAR_ROCM_MOE_PREFILL_SHARD >= 0

namespace llaminar2::rocm::moe_prefill_shards {

#define LLAMINAR_PF_SYM_IMPL(BASE, S) BASE##_s##S
#define LLAMINAR_PF_SYM(BASE, S) LLAMINAR_PF_SYM_IMPL(BASE, S)

bool LLAMINAR_PF_SYM(routeGateUpSymbol, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    uint8_t codebook, llaminar2::rocm::MoERouteOwnedGateUpGeometry geometry,
    llaminar2::rocm::MoERouteOwnedGateUpPartitionExecution partition_execution,
    int maximum_partitions, MoERouteLeafSymbol& out)
{
#define LLAMINAR_PF_CASE(FMT) \
    if (codebook == static_cast<uint8_t>(FMT)) { \
        out = routeGateUpSymbolForGeometry<FMT>(geometry, partition_execution, maximum_partitions); \
        return true; \
    }
    LLAMINAR_ROCM_MOE_PREFILL_FOR_EACH_CODEBOOK(LLAMINAR_PF_CASE)
#undef LLAMINAR_PF_CASE
    return false;
}

bool LLAMINAR_PF_SYM(routeDownSymbol, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    uint8_t codebook, llaminar2::MoEPrefillOutputLayout layout,
    llaminar2::rocm::MoECanonicalDownExecution execution, MoERouteLeafSymbol& out)
{
#define LLAMINAR_PF_CASE(FMT) \
    if (codebook == static_cast<uint8_t>(FMT)) { \
        out = routeDownSymbol<FMT>(layout, execution); \
        return true; \
    }
    LLAMINAR_ROCM_MOE_PREFILL_FOR_EACH_CODEBOOK(LLAMINAR_PF_CASE)
#undef LLAMINAR_PF_CASE
    return false;
}

bool LLAMINAR_PF_SYM(queryResources, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    uint8_t codebook_id, MoEPrefillProjectionRole role, int tile_m, int tile_n,
    int* registers_per_thread, size_t* local_memory_bytes_per_thread,
    size_t* static_shared_memory_bytes, int* max_threads_per_block,
    int* max_active_blocks_per_sm)
{
#define QUERY_TILE_N(FMT, TM) \
    switch (tile_n) { \
    case 64:  return moe_prefill_query_candidate_resources<FMT, TM, 64>(role, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm); \
    case 128: return moe_prefill_query_candidate_resources<FMT, TM, 128>(role, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm); \
    case 256: return moe_prefill_query_candidate_resources<FMT, TM, 256>(role, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm); \
    default:  return false; \
    }
#define QUERY_ADAPTIVE_TILE_N(FMT) \
    switch (tile_n) { \
    case 64:  return moe_prefill_query_adaptive_candidate_resources<FMT, 64>(role, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm); \
    case 128: return moe_prefill_query_adaptive_candidate_resources<FMT, 128>(role, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm); \
    case 256: return moe_prefill_query_adaptive_candidate_resources<FMT, 256>(role, registers_per_thread, local_memory_bytes_per_thread, static_shared_memory_bytes, max_threads_per_block, max_active_blocks_per_sm); \
    default:  return false; \
    }
#define LLAMINAR_PF_CASE(FMT) \
    if (static_cast<NativeQuantisedFormatPrefill>(codebook_id) == FMT) { \
        switch (tile_m) { \
        case 4:  QUERY_TILE_N(FMT, 4); \
        case 8:  QUERY_TILE_N(FMT, 8); \
        case 12: QUERY_TILE_N(FMT, 12); \
        case 16: QUERY_TILE_N(FMT, 16); \
        case kMoEPrefillAdaptiveTileRows: QUERY_ADAPTIVE_TILE_N(FMT); \
        default: return false; \
        } \
    }
    LLAMINAR_ROCM_MOE_PREFILL_FOR_EACH_CODEBOOK(LLAMINAR_PF_CASE)
#undef LLAMINAR_PF_CASE
#undef QUERY_ADAPTIVE_TILE_N
#undef QUERY_TILE_N
    return false;
}

bool LLAMINAR_PF_SYM(launchGateUp, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    const GateUpLaunchContext& c, bool& launched)
{
    auto* obs = static_cast<MoERouteOwnedSubmissionObservation*>(c.submission_observation_ptr);
#define LAUNCH_GU_TILED(FMT, TM, TN) \
    do { \
        const int cap = c.ensure_work_directory(c.cb_ctx, TM); \
        if (cap <= 0) return false; \
        dim3 grid((c.intermediate + TN - 1) / TN, cap, 2); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<FMT, TM, TN>), \
            grid, dim3(TN), 0, c.hip_stream, c.act_int8, c.act_scales, \
            c.d_group_token_indices, c.d_group_counts, c.d_group_offsets, \
            reinterpret_cast<const uint32_t*>(c.d_group_work_directory), \
            c.gate_descs, c.up_descs, c.d_scratch_gate, c.d_scratch_up, \
            cap, c.seq_len, c.top_k, c.grouped_indices_are_route_slots, 1, c.intermediate, c.d_model, \
            c.gateup_k_partition_policy, c.num_experts, nullptr, c.admission); \
        if (hipGetLastError() != hipSuccess) return false; \
        c.pending->gateup.record({llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4, \
            llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert, TM, TN, TN, 0, 1}); \
    } while (0)
#define LAUNCH_GU_ADAPTIVE(FMT, TN) \
    do { \
        const AdaptiveDirectoryView ad = c.build_adaptive_work_directory(c.cb_ctx, MoEPrefillProjectionRole::GateUp); \
        if (!ad.valid()) return false; \
        dim3 g12((c.intermediate + TN - 1) / TN, ad.tile12_capacity, 2); \
        dim3 g16((c.intermediate + TN - 1) / TN, ad.tile16_capacity, 2); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<FMT, 12, TN>), \
            g12, dim3(TN), 0, c.hip_stream, c.act_int8, c.act_scales, \
            c.d_group_token_indices, c.d_group_counts, c.d_group_offsets, ad.tile12, \
            c.gate_descs, c.up_descs, c.d_scratch_gate, c.d_scratch_up, \
            ad.tile12_capacity, c.seq_len, c.top_k, c.grouped_indices_are_route_slots, 1, c.intermediate, c.d_model, \
            c.gateup_k_partition_policy, c.num_experts, ad.header + kMoEPrefillAdaptiveTile12CountWord, c.admission); \
        if (hipGetLastError() != hipSuccess) return false; \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel<FMT, 16, TN>), \
            g16, dim3(TN), 0, c.hip_stream, c.act_int8, c.act_scales, \
            c.d_group_token_indices, c.d_group_counts, c.d_group_offsets, ad.tile16, \
            c.gate_descs, c.up_descs, c.d_scratch_gate, c.d_scratch_up, \
            ad.tile16_capacity, c.seq_len, c.top_k, c.grouped_indices_are_route_slots, 1, c.intermediate, c.d_model, \
            c.gateup_k_partition_policy, c.num_experts, ad.header + kMoEPrefillAdaptiveTile16CountWord, c.admission); \
        if (hipGetLastError() != hipSuccess) return false; \
        c.pending->gateup.record({llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4, llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert, 12, TN, TN, 0, 1}); \
        c.pending->gateup.record({llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4, llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert, 16, TN, TN, 0, 1}); \
    } while (0)
#define DISPATCH_GU_TN(FMT, TM, CFG) \
    switch ((CFG).columns) { case 64: LAUNCH_GU_TILED(FMT, TM, 64); break; case 128: LAUNCH_GU_TILED(FMT, TM, 128); break; case 256: LAUNCH_GU_TILED(FMT, TM, 256); break; default: return false; }
#define LLAMINAR_PF_CASE(FMT) \
    if (moe_prefill_launch_mask(c.gateup_codebook_mask) & (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
        if (c.use_route_owned) { \
            launch_moe_grouped_gate_up_route_owned<FMT>(c.hip_stream, c.act_int8, c.act_scales, \
                c.d_group_token_indices, c.grouped_indices_are_route_slots, c.gate_descs, c.up_descs, \
                c.d_scratch_gate, c.d_scratch_up, c.total_slots, c.seq_len, c.top_k, 1, c.intermediate, c.d_model, \
                c.gateup_k_partition_policy, c.num_experts, c.admission, c.launch_proof, obs); \
            c.pending->gateup_route_owned = true; \
        } \
        if (c.use_expert_row_tiles && !c.native_wmma) { \
            const MoEPrefillTileConfig cand = c.has_uniform_codebook_pair ? c.pair_gateup \
                : moe_prefill_select_tile_config(MoEPrefillProjectionRole::GateUp, FMT, c.seq_len, c.intermediate, c.d_model); \
            if (!moe_prefill_valid_tile_rows(cand.rows) || !moe_prefill_valid_tile_columns(cand.columns)) return false; \
            switch (cand.rows) { \
            case 4:  DISPATCH_GU_TN(FMT, 4, cand); break; \
            case 8:  DISPATCH_GU_TN(FMT, 8, cand); break; \
            case 12: DISPATCH_GU_TN(FMT, 12, cand); break; \
            case 16: DISPATCH_GU_TN(FMT, 16, cand); break; \
            case kMoEPrefillAdaptiveTileRows: \
                switch (cand.columns) { case 64: LAUNCH_GU_ADAPTIVE(FMT, 64); break; case 128: LAUNCH_GU_ADAPTIVE(FMT, 128); break; case 256: LAUNCH_GU_ADAPTIVE(FMT, 256); break; default: return false; } \
                break; \
            default: return false; \
            } \
        } \
        launched = true; \
    }
    LLAMINAR_ROCM_MOE_PREFILL_FOR_EACH_CODEBOOK(LLAMINAR_PF_CASE)
#undef LLAMINAR_PF_CASE
#undef DISPATCH_GU_TN
#undef LAUNCH_GU_ADAPTIVE
#undef LAUNCH_GU_TILED
    return true;
}

bool LLAMINAR_PF_SYM(launchRouteOwnedDown, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    const RouteOwnedDownLaunchContext& c, bool& matched)
{
    constexpr int TILE_N = 64;
    const uint32_t mask = moe_prefill_launch_mask(c.down_codebook_mask);
#define LAUNCH_CD_W(FTAG, W) \
    do { \
        using CG = MoECanonicalRouteDownGeometry<FTAG, W>; \
        dim3 grid((c.down_columns + CG::columns - 1) / CG::columns, std::min(c.total_slots, kCanonicalPublicationSlotLanes)); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_down_publish_canonical_routes_prefill_kernel<FTAG, W>), \
            grid, dim3(CG::threads), 0, c.hip_stream, c.d_scratch_swiglu_int8, c.d_scratch_swiglu_scales, \
            c.down_descs, c.d_original_to_grouped, c.d_original_expert_ids, c.projection_weights, \
            c.d_canonical_route_contributions, c.total_slots, c.down_columns, c.intermediate, \
            c.num_experts, c.down_k_partition_policy, c.admission); \
    } while (0)
#define LLAMINAR_PF_CASE(FMT) \
    if (mask == (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
        if (c.canonical_publication) { \
            const auto exec = llaminar2::rocm::selectMoECanonicalDownExecution({c.down_columns, c.intermediate, c.total_slots}); \
            [&]<int SelectedFormat>() { \
                if constexpr (SelectedFormat == static_cast<int>(NATIVE_QUANTISED_PF_Q5_1)) { \
                    if (exec == llaminar2::rocm::MoECanonicalDownExecution::FourWorkers) LAUNCH_CD_W(SelectedFormat, 4); \
                    else LAUNCH_CD_W(SelectedFormat, 1); \
                } else { \
                    LAUNCH_CD_W(SelectedFormat, 1); \
                } \
            }.template operator()<static_cast<int>(FMT)>(); \
        } else { \
            dim3 grid((c.down_columns + TILE_N - 1) / TILE_N, \
                      std::min(c.seq_len, kCanonicalPublicationSlotLanes)); \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_down_publish_original_token_ordered_prefill_kernel<static_cast<int>(FMT)>), \
                grid, dim3(TILE_N), 0, c.hip_stream, c.d_scratch_swiglu_int8, c.d_scratch_swiglu_scales, \
                c.down_descs, c.d_original_to_grouped, c.d_original_expert_ids, c.d_group_weights, \
                c.d_output, c.seq_len, c.top_k, c.down_columns, c.intermediate, c.num_experts, \
                c.down_k_partition_policy, c.admission); \
        } \
        matched = true; \
        return true; \
    }
    LLAMINAR_ROCM_MOE_PREFILL_FOR_EACH_CODEBOOK(LLAMINAR_PF_CASE)
#undef LLAMINAR_PF_CASE
#undef LAUNCH_CD_W
    return true;
}

bool LLAMINAR_PF_SYM(launchDown, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    const ExpertTiledDownLaunchContext& c, bool& launched)
{
#define LAUNCH_DN_TILED(FMT, TM, TN) \
    do { \
        const int cap = c.ensure_work_directory(c.cb_ctx, TM); \
        if (cap <= 0) return false; \
        dim3 grid((c.down_columns + TN - 1) / TN, cap); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_down_batch_invariant_tiled_prefill_kernel<FMT, TM, TN>), \
            grid, dim3(TN), 0, c.hip_stream, c.d_scratch_swiglu_int8, c.d_scratch_swiglu_scales, c.down_descs, \
            c.d_group_counts, c.d_group_offsets, reinterpret_cast<const uint32_t*>(c.d_group_work_directory), \
            c.d_scratch_gate, cap, c.down_columns, c.intermediate, c.down_k_partition_policy, c.num_experts, nullptr, c.admission); \
        if (hipGetLastError() != hipSuccess) return false; \
        c.pending->down.record({llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4, llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert, TM, TN, TN, 0, 1}); \
    } while (0)
#define LAUNCH_DN_ADAPTIVE(FMT, TN) \
    do { \
        const AdaptiveDirectoryView ad = c.build_adaptive_work_directory(c.cb_ctx, MoEPrefillProjectionRole::Down); \
        if (!ad.valid()) return false; \
        dim3 g12((c.down_columns + TN - 1) / TN, ad.tile12_capacity); \
        dim3 g16((c.down_columns + TN - 1) / TN, ad.tile16_capacity); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_down_batch_invariant_tiled_prefill_kernel<FMT, 12, TN>), \
            g12, dim3(TN), 0, c.hip_stream, c.d_scratch_swiglu_int8, c.d_scratch_swiglu_scales, c.down_descs, \
            c.d_group_counts, c.d_group_offsets, ad.tile12, c.d_scratch_gate, ad.tile12_capacity, \
            c.down_columns, c.intermediate, c.down_k_partition_policy, c.num_experts, ad.header + kMoEPrefillAdaptiveTile12CountWord, c.admission); \
        if (hipGetLastError() != hipSuccess) return false; \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_grouped_down_batch_invariant_tiled_prefill_kernel<FMT, 16, TN>), \
            g16, dim3(TN), 0, c.hip_stream, c.d_scratch_swiglu_int8, c.d_scratch_swiglu_scales, c.down_descs, \
            c.d_group_counts, c.d_group_offsets, ad.tile16, c.d_scratch_gate, ad.tile16_capacity, \
            c.down_columns, c.intermediate, c.down_k_partition_policy, c.num_experts, ad.header + kMoEPrefillAdaptiveTile16CountWord, c.admission); \
        if (hipGetLastError() != hipSuccess) return false; \
        c.pending->down.record({llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4, llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert, 12, TN, TN, 0, 1}); \
        c.pending->down.record({llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4, llaminar2::rocm::MoEGroupedWmmaRowFloor::EveryExpert, 16, TN, TN, 0, 1}); \
    } while (0)
#define DISPATCH_DN_TN(FMT, TM, CFG) \
    switch ((CFG).columns) { case 64: LAUNCH_DN_TILED(FMT, TM, 64); break; case 128: LAUNCH_DN_TILED(FMT, TM, 128); break; case 256: LAUNCH_DN_TILED(FMT, TM, 256); break; default: return false; }
#define LLAMINAR_PF_CASE(FMT) \
    if (moe_prefill_launch_mask(c.down_codebook_mask) & (uint32_t{1} << static_cast<uint32_t>(FMT))) { \
        const MoEPrefillTileConfig cand = c.has_uniform_codebook_pair ? c.pair_down \
            : moe_prefill_select_tile_config(MoEPrefillProjectionRole::Down, FMT, c.seq_len, c.down_columns, c.intermediate); \
        if (!moe_prefill_valid_tile_rows(cand.rows) || !moe_prefill_valid_tile_columns(cand.columns)) return false; \
        switch (cand.rows) { \
        case 4:  DISPATCH_DN_TN(FMT, 4, cand); break; \
        case 8:  DISPATCH_DN_TN(FMT, 8, cand); break; \
        case 12: DISPATCH_DN_TN(FMT, 12, cand); break; \
        case 16: DISPATCH_DN_TN(FMT, 16, cand); break; \
        case kMoEPrefillAdaptiveTileRows: \
            switch (cand.columns) { case 64: LAUNCH_DN_ADAPTIVE(FMT, 64); break; case 128: LAUNCH_DN_ADAPTIVE(FMT, 128); break; case 256: LAUNCH_DN_ADAPTIVE(FMT, 256); break; default: return false; } \
            break; \
        default: return false; \
        } \
        launched = true; \
    }
    LLAMINAR_ROCM_MOE_PREFILL_FOR_EACH_CODEBOOK(LLAMINAR_PF_CASE)
#undef LLAMINAR_PF_CASE
#undef DISPATCH_DN_TN
#undef LAUNCH_DN_ADAPTIVE
#undef LAUNCH_DN_TILED
    return true;
}

bool LLAMINAR_PF_SYM(initIQ, LLAMINAR_ROCM_MOE_PREFILL_SHARD)(
    int device_id, const void* h_iq3s, const void* h_iq3xxs, const void* h_iq2s,
    const void* h_iq2xs, const void* h_iq2xxs, const void* h_iq1s)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);
    if (hipMemcpyToSymbol(d_pf_iq3s_grid, h_iq3s, sizeof(d_pf_iq3s_grid)) != hipSuccess) return false;
    if (hipMemcpyToSymbol(d_pf_iq3xxs_grid, h_iq3xxs, sizeof(d_pf_iq3xxs_grid)) != hipSuccess) return false;
    if (hipMemcpyToSymbol(d_pf_iq2s_grid, h_iq2s, sizeof(d_pf_iq2s_grid)) != hipSuccess) return false;
    if (hipMemcpyToSymbol(d_pf_iq2xs_grid, h_iq2xs, sizeof(d_pf_iq2xs_grid)) != hipSuccess) return false;
    if (hipMemcpyToSymbol(d_pf_iq2xxs_grid, h_iq2xxs, sizeof(d_pf_iq2xxs_grid)) != hipSuccess) return false;
    if (hipMemcpyToSymbol(d_pf_iq1s_grid, h_iq1s, sizeof(d_pf_iq1s_grid)) != hipSuccess) return false;
    return true;
}

#undef LLAMINAR_PF_SYM
#undef LLAMINAR_PF_SYM_IMPL

} // namespace llaminar2::rocm::moe_prefill_shards

#endif // LLAMINAR_ROCM_MOE_PREFILL_SHARD < 0
