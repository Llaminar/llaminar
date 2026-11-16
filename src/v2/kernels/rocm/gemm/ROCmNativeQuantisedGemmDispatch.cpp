/**
 * @file ROCmNativeQuantisedGemmDispatch.cpp
 * @brief Public dispatch and constant-table publication for ROCm NativeQuantised.
 *
 * This translation unit intentionally contains no GPU template code. It maps
 * each runtime codebook to the one physical HIP shard that owns its kernel
 * instantiations. Exact ownership makes unsupported or duplicate mappings
 * structurally visible while allowing all eight expensive shards to compile in
 * parallel.
 */

#include "ROCmNativeQuantisedGemmShard.h"
#include "ROCmVNNIPrefillLaunch.h"
#include "ROCmNativeQuantisedWmmaLaunch.h"
#include "ROCmNativeQuantisedVerifierLaunch.h"
#include "ROCmNativeQuantisedGDNVerifier.h"
#include "kernels/common/VerifierGDNBundle.h"
#include "kernels/common/ProjectionWorkLaunch.h"
#include "kernels/common/VerifierProjectionBundle.h"
#include "ROCmNativeQuantisedArchitecture.h"
#include "../../../tensors/NativeQuantisedFormatInfo.h"
#include "../../../utils/PerfStatsCollector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>

#include <hip/hip_runtime_api.h>

namespace
{
/*
 * Production capture may compile several device graphs concurrently on
 * independent host workers. A thread-local observation avoids a shared lock
 * and guarantees that a trainer reads only the launch it issued itself.
 */
thread_local std::optional<llaminar2::rocm::VNNIPrefillLaunch>
    g_last_prefill_launch_selection;
thread_local std::optional<llaminar2::rocm::NativeQuantisedWmmaTile> g_wmma_override;

constexpr ROCmNativeQuantisedGemmShardFn shardForCodebook(uint8_t codebook_id)
{
    switch (codebook_id)
    {
    case 0:
    case 4:
    case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        return rocmGemm_native_quantised_fp32_shard_0;
    case 5:
    case 6:
        return rocmGemm_native_quantised_fp32_shard_1;
    case 7:
    case 8:
        return rocmGemm_native_quantised_fp32_shard_2;
    case 9:
    case 10:
        return rocmGemm_native_quantised_fp32_shard_3;
    case 11:
    case 12:
        return rocmGemm_native_quantised_fp32_shard_4;
    case 13:
    case 14:
        return rocmGemm_native_quantised_fp32_shard_5;
    case 15:
    case 16:
        return rocmGemm_native_quantised_fp32_shard_6;
    case 17:
    case 19:
    case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
        return rocmGemm_native_quantised_fp32_shard_7;
    default:
        return nullptr;
    }
}

/** @brief Preserve legacy policy semantics for a self-describing decoder. */
constexpr uint8_t defaultArithmeticPolicyCodebook(uint8_t codebook_id)
{
    return codebook_id == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook ? uint8_t{4} :
           codebook_id == llaminar2::kNativeQuantisedExpandedInt8MinCodebook
               ? static_cast<uint8_t>(19)
               : codebook_id;
}

constexpr std::array<ROCmNativeQuantisedGridInitShardFn, 10> kGridInitializers = {
    rocmInitIQGridTables_gemm_shard_0,
    rocmInitIQGridTables_gemm_shard_1,
    rocmInitIQGridTables_gemm_shard_2,
    rocmInitIQGridTables_gemm_shard_3,
    rocmInitIQGridTables_gemm_shard_4,
    rocmInitIQGridTables_gemm_shard_5,
    rocmInitIQGridTables_gemm_shard_6,
    rocmInitIQGridTables_gemm_shard_7,
    llaminar2::rocm::initializeNativeQuantisedWmmaGrids,
    llaminar2::rocm::initializeNativeQuantisedVerifierWmmaGrids,
};
} // namespace

std::optional<llaminar2::rocm::NativeQuantisedWmmaTile>
llaminar2::rocm::nativeQuantisedWmmaOverride() noexcept { return g_wmma_override; }

void llaminar2::rocm::setNativeQuantisedWmmaOverride(
    std::optional<NativeQuantisedWmmaTile> tile) noexcept { g_wmma_override = tile; }

void llaminar2::rocm::publishVNNIPrefillLaunch(
    const VNNIPrefillLaunch &launch)
{
    if (!launch.function || launch.block_threads <= 0 ||
        launch.n_tile <= 0 || launch.m_tile <= 0 || launch.min_blocks <= 0)
        throw std::invalid_argument("incomplete ROCm VNNI prefill launch identity");
    g_last_prefill_launch_selection = launch;
}

void llaminar2::rocm::clearVNNIPrefillLaunch() noexcept
{
    g_last_prefill_launch_selection.reset();
}

std::optional<llaminar2::rocm::VNNIPrefillLaunch>
llaminar2::rocm::lastVNNIPrefillLaunch() noexcept
{
    return g_last_prefill_launch_selection;
}

bool llaminar2::rocm::inspectVNNIPrefillLaunch(
    const VNNIPrefillLaunch &selection, VNNIPrefillLaunchResources &resources)
{
    resources = {};
    if (!selection.function || selection.block_threads <= 0)
    {
        return false;
    }

    hipFuncAttributes attributes{};
    if (hipFuncGetAttributes(&attributes, selection.function) != hipSuccess)
        return false;

    int active_blocks = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            selection.function,
            selection.block_threads,
            selection.dynamic_shared_memory_bytes) != hipSuccess ||
        active_blocks <= 0)
    {
        return false;
    }

    resources.registers_per_thread = attributes.numRegs;
    resources.local_memory_bytes_per_thread = attributes.localSizeBytes;
    resources.static_shared_memory_bytes = attributes.sharedSizeBytes;
    resources.max_threads_per_block = attributes.maxThreadsPerBlock;
    resources.max_active_blocks_per_sm = active_blocks;
    return attributes.numRegs > 0 &&
           attributes.maxThreadsPerBlock >= selection.block_threads;
}

/**
 * @brief Dispatch prefill with separate physical and arithmetic identities.
 *
 * The physical codebook selects a shard and decoder. The arithmetic policy
 * codebook is forwarded unchanged so that every cooperative row reproduces
 * the source format's serial-M1 split tree after cross-tier normalization.
 */
extern "C" bool rocmGemm_native_quantised_fp32_with_policy(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const void *d_block_scales,
    const void *d_block_mins,
    const void *d_block_emins,
    float *d_output,
    const float *d_scales_A,
    const float *d_scales_A_blockwise,
    int M,
    int N,
    int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id,
    void *stream)
{
    const ROCmNativeQuantisedGemmShardFn shard = shardForCodebook(codebook_id);
    if (!shard)
    {
        std::fprintf(
            stderr,
            "[rocmGemm_native_quantised_fp32] unsupported codebook_id=%u\n",
            static_cast<unsigned>(codebook_id));
        return false;
    }

    // The successful producer must publish this invocation's physical launch.
    // Retaining an older same-codebook observation would forge capture evidence.
    llaminar2::rocm::clearVNNIPrefillLaunch();
    const bool launched = shard(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_output,
        d_scales_A,
        d_scales_A_blockwise,
        M,
        N,
        K,
        codebook_id,
        arithmetic_policy_codebook_id,
        device_id,
        stream);

    if (launched &&
        llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
    {
        const auto observed = llaminar2::rocm::lastVNNIPrefillLaunch();
        if (!observed || observed->execution_codebook != codebook_id)
            throw std::logic_error("ROCm native prefill omitted its physical launch observation");
        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "rocm_native_quantised_prefill_launch",
            1.0,
            "gemm",
            "rocm:" + std::to_string(device_id),
            llaminar2::PerfStatsCollector::Tags{
                {"codebook", std::to_string(static_cast<int>(codebook_id))},
                {"arithmetic_policy_codebook",
                 std::to_string(
                     static_cast<int>(arithmetic_policy_codebook_id))},
                {"m", std::to_string(M)},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"producer", llaminar2::rocm::vnniPrefillProducerName(observed->producer)},
                {"n_tile", std::to_string(observed->n_tile)},
                {"m_tile", std::to_string(observed->m_tile)},
                {"min_blocks", std::to_string(observed->min_blocks)},
                {"unroll", std::to_string(observed->unroll)},
                {"full_tiles", observed->full_tiles ? "1" : "0"},
                {"block_threads", std::to_string(observed->block_threads)},
                {"dynamic_shared_memory_bytes", std::to_string(observed->dynamic_shared_memory_bytes)}});
    }
    return launched;
}

/** @brief Dispatch self-describing prefill through the explicit policy ABI. */
extern "C" bool rocmGemm_native_quantised_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const void *d_block_scales,
    const void *d_block_mins,
    const void *d_block_emins,
    float *d_output,
    const float *d_scales_A,
    const float *d_scales_A_blockwise,
    int M,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void *stream)
{
    return rocmGemm_native_quantised_fp32_with_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_output,
        d_scales_A,
        d_scales_A_blockwise,
        M,
        N,
        K,
        codebook_id,
        defaultArithmeticPolicyCodebook(codebook_id),
        device_id,
        stream);
}

extern "C" bool rocmInitIQGridTables_gemm(
    int device_id,
    const void *h_iq3s_grid,
    const void *h_iq3xxs_grid,
    const void *h_iq2s_grid,
    const void *h_iq2xs_grid,
    const void *h_iq2xxs_grid,
    const void *h_iq1s_grid)
{
    for (const ROCmNativeQuantisedGridInitShardFn initialize : kGridInitializers)
    {
        if (!initialize(
                device_id,
                h_iq3s_grid,
                h_iq3xxs_grid,
                h_iq2s_grid,
                h_iq2xs_grid,
                h_iq2xxs_grid,
                h_iq1s_grid))
        {
            return false;
        }
    }
    return true;
}

namespace llaminar2::rocm {
namespace {
thread_local std::optional<NativeQuantisedVerifierSelection> verifier_selection;
thread_local const ScopedNativeQuantisedVerifierCandidate *verifier_scope=nullptr;
}
std::optional<NativeQuantisedVerifierSelection> nativeQuantisedVerifierSelection() noexcept
{ return verifier_selection; }
ScopedNativeQuantisedVerifierCandidate::ScopedNativeQuantisedVerifierCandidate(
    NativeQuantisedVerifierCandidate candidate, int device) : previous_(verifier_selection), previous_scope_(verifier_scope)
{
    const auto hardware = nativeQuantisedHardwareForDevice(device);
    switch (candidate) {
    case NativeQuantisedVerifierCandidate::Automatic:
    case NativeQuantisedVerifierCandidate::Dot4: break;
    case NativeQuantisedVerifierCandidate::WmmaM16N64:
        if (hardware.architecture != native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
            throw std::invalid_argument("Native verifier WMMA requires actual gfx1100");
        break;
    default: throw std::invalid_argument("Unknown native verifier candidate");
    }
    verifier_selection = NativeQuantisedVerifierSelection{candidate,device};
    verifier_scope=this;
}
ScopedNativeQuantisedVerifierCandidate::~ScopedNativeQuantisedVerifierCandidate()
{
    if(verifier_scope!=this)std::terminate();
    verifier_selection=previous_;verifier_scope=previous_scope_;
}
thread_local const ScopedNativeQuantisedVerifierOperation *ScopedNativeQuantisedVerifierOperation::active_ = nullptr;
ScopedNativeQuantisedVerifierOperation::ScopedNativeQuantisedVerifierOperation(
    std::optional<NativeQuantisedVerifierOperation> operation) noexcept
    : operation_(operation), previous_(active_)
{ active_ = this; }
ScopedNativeQuantisedVerifierOperation::~ScopedNativeQuantisedVerifierOperation()
{
    if (active_ != this) std::terminate();
    active_ = previous_;
}
std::optional<NativeQuantisedVerifierOperation> ScopedNativeQuantisedVerifierOperation::current() noexcept
{ return active_ ? active_->operation_ : std::nullopt; }

thread_local const ScopedNativeQuantisedVerifierBundle *ScopedNativeQuantisedVerifierBundle::active_ = nullptr;
ScopedNativeQuantisedVerifierBundle::ScopedNativeQuantisedVerifierBundle(
    std::optional<NativeQuantisedVerifierBundleOperation> operation) noexcept
    : operation_(operation), previous_(active_)
{ active_ = this; }
ScopedNativeQuantisedVerifierBundle::~ScopedNativeQuantisedVerifierBundle()
{
    if (active_ != this) std::terminate();
    active_ = previous_;
}
const NativeQuantisedVerifierBundleOperation *ScopedNativeQuantisedVerifierBundle::current() noexcept
{ return active_ && active_->operation_ ? &*active_->operation_ : nullptr; }

namespace
{
/** @brief Shared packet authentication; complete operations own their own role eligibility. */
bool matchesVerifierMembers(std::span<const NativeQuantisedVerifierBundleMember> members,
    int original_device,int k,const DeviceRowRange &rows,const int8_t *activations,
    const float *activation_scales,const int32_t *activation_sums,
    const NativeQuantisedVerifierWork &work,int device) noexcept
{
    if (members.empty() || members.size() > NATIVE_QUANTISED_SMALL_M_MAX_BATCH || original_device != device || k != work.k ||
        work.rows.physicalRows() < 2 || work.rows.physicalRows() > 16 ||
        rows.captureIdentity() != work.rows.captureIdentity() ||
        work.policy != NativeQuantisedVerifierPolicyMatch::CanonicalSerial || !work.bundle ||
        !activations || !activation_scales || !activation_sums ||
        activations != work.activations || activation_scales != work.activation_scales ||
        activation_sums != work.activation_sums || work.projections < 1 || work.projections > static_cast<int>(members.size()))
        return false;
    const auto &packet = *work.bundle;
    if (packet.num_projections != work.projections || packet.rows.captureIdentity() != work.rows.captureIdentity() ||
        packet.work.captureIdentity() != ProjectionRowWork::fromPrefix(work.rows).captureIdentity() ||
        packet.payload[0] != work.weights || packet.N[0] != work.n || packet.codebook[0] != work.physical_codebook)
        return false;
    for (int member = 0; member < static_cast<int>(members.size()); ++member)
    {
        const auto &original = members[member];
        if (!original.weights || !original.scales || !original.output || original.n <= 0 ||
            !original.source.present || canonicalDeviceVnniCodebookId(original.source) == 255)
            return false;
        for (int earlier = 0; earlier < member; ++earlier)
            if (members[earlier].output == original.output) return false;
    }
    unsigned seen = 0;
    int previous_member = -1;
    for (int projection = 0; projection < work.projections; ++projection)
    {
        if (packet.bias[projection]) return false;
        int match = -1;
        for (int member = 0; member < static_cast<int>(members.size()); ++member)
        {
            const auto &original = members[member];
            if (packet.output[projection] == original.output && packet.payload[projection] == original.weights &&
                packet.N[projection] == original.n && packet.scales[projection] == original.scales &&
                packet.mins[projection] == original.mins && packet.emins[projection] == original.emins &&
                packet.codebook[projection] == canonicalDeviceVnniCodebookId(original.source))
                match = member;
        }
        if (match <= previous_member || (seen & (1u << match))) return false;
        seen |= 1u << match;
        previous_member = match;
    }
    return true;
}

/** @brief Validate both quantized roles and distinct outputs before considering a subgroup. */
bool validGDNOperation(const NativeQuantisedGDNVerifierOperation &operation) noexcept
{
    if(operation.device<0 || !operation.input || operation.k<1) return false;
    if(operation.precision==QuantizedActivationOperandPrecision::TwoTermQ8) {
        if(!operation.correction_rows || operation.correction_rows->captureIdentity()!=operation.rows.captureIdentity())return false;
    } else if(operation.precision!=QuantizedActivationOperandPrecision::SingleQ8 || operation.correction_rows)return false;
    for(const auto *member:{&operation.qkv,&operation.z})
        if(!member->weights || !member->scales || !member->output || member->n<1 ||
            !member->source.present || canonicalDeviceVnniCodebookId(member->source)==255)return false;
    for(const auto *control:{&operation.alpha,&operation.beta})
        if(!control->engine || !control->weights.valid() || control->weights.k!=operation.k || !control->output ||
            control->weights.type!=TensorType::FP16 || control->precision!=FloatingActivationOperandPrecision::FP16RNE)return false;
    const std::array outputs{operation.qkv.output,operation.z.output,operation.alpha.output,operation.beta.output};
    for(size_t i=0;i<outputs.size();++i)for(size_t j=0;j<i;++j)if(outputs[i]==outputs[j])return false;
    return true;
}
}

bool matchesNativeQuantisedVerifierBundle(const NativeQuantisedVerifierBundleOperation &operation,
    const NativeQuantisedVerifierWork &work,int device) noexcept
{
    const int expected=operation.kind==VerifierProjectionBundleKind::AttentionQKV?3:
        operation.kind==VerifierProjectionBundleKind::AttentionKV?2:0;
    return expected && operation.count==expected && matchesVerifierMembers(
        std::span(operation.members).first(expected),operation.device,operation.k,operation.rows,
        operation.activations,operation.activation_scales,operation.activation_sums,work,device);
}

thread_local const ScopedNativeQuantisedGDNVerifier *ScopedNativeQuantisedGDNVerifier::active_=nullptr;
ScopedNativeQuantisedGDNVerifier::ScopedNativeQuantisedGDNVerifier(
    std::optional<NativeQuantisedGDNVerifierOperation> operation) noexcept
    :operation_(std::move(operation)),previous_(active_) {active_=this;}
ScopedNativeQuantisedGDNVerifier::~ScopedNativeQuantisedGDNVerifier()
{if(active_!=this)std::terminate();active_=previous_;}
const NativeQuantisedGDNVerifierOperation *ScopedNativeQuantisedGDNVerifier::current() noexcept
{return active_ && active_->operation_?&*active_->operation_:nullptr;}

bool hasNativeQuantisedGDNVerifier() noexcept
{return ScopedNativeQuantisedGDNVerifier::current()!=nullptr;}

bool matchesNativeQuantisedGDNVerifier(const NativeQuantisedGDNVerifierOperation &operation,
    const NativeQuantisedVerifierWork &work,int device) noexcept
{
    if(!validGDNOperation(operation))return false;
    const std::array members{operation.qkv,operation.z};
    const auto eligible=std::span(members).subspan(
        operation.precision==QuantizedActivationOperandPrecision::TwoTermQ8?1:0);
    return matchesVerifierMembers(eligible,operation.device,operation.k,operation.rows,
        operation.activations,operation.activation_scales,operation.activation_sums,work,device);
}

namespace
{
/** @brief Source-specific full-extent masks for complete original QKV/KV operations. */
struct VerifierBundleMeasuredWidths
{
    NativeQuantisedSourceIdentity source;
    std::array<uint16_t,7> rows;
};
#include "ROCmNativeQuantisedVerifierBundlePolicy.inc"
static_assert([] {
    for (size_t i = 0; i < kVerifierBundleMeasuredWidths.size(); ++i)
    {
        const auto &entry = kVerifierBundleMeasuredWidths[i];
        if (!entry.source.present || canonicalDeviceVnniCodebookId(entry.source) == 255) return false;
        for (auto mask : entry.rows) if (mask & 0x8000u) return false;
        for (size_t j = 0; j < i; ++j)
            if (entry.source == kVerifierBundleMeasuredWidths[j].source) return false;
    }
    return true;
}(), "Complete verifier bundle masks require unique source identities and widths M2..16");

/** @brief Measured widths retain complete source identity and separate public operations.
 * Bit M-2 admits one exact physical width, only within M2..16. Shape columns
 * are N/K17408/5120,5120/17408,10240/5120. Zero and nonmonotone masks are valid;
 * unmeasured sources, geometries and widths remain the ordered implementation.
 * Each admitted cell won its complete-operation median and at least22 of31
 * paired comparisons. Source aliases do not borrow another source's evidence. */
struct VerifierMeasuredWidths
{
    NativeQuantisedSourceIdentity source;
    std::array<uint16_t,3> projection_rows;
    std::array<uint16_t,3> swiglu_rows;
};
constexpr std::array<VerifierMeasuredWidths,21> kVerifierMeasuredWidths{{
    {{0,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q4_0
    {{4,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // IQ4_NL
    {{4,true,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // IQ4_XS
    {{5,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q4_1
    {{5,true,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q4_K
    {{6,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q5_0
    {{7,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q5_1
    {{7,true,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q5_K
    {{8,true,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q6_K
    {{9,true,true},{0x7f80,0x7f80,0x7f80},{0x7f80,0x7f80,0x7f80}}, // Q3_K
    {{10,true,true},{0x7f80,0x7f80,0x7f80},{0x7f80,0x7f80,0x7f80}}, // Q2_K
    {{11,true,true},{0x7f80,0x7fe0,0x7f80},{0x7f80,0x7fe0,0x7f80}}, // IQ3_S
    {{12,true,true},{0x7f80,0x7fc0,0x7f80},{0x7f80,0x7fc0,0x7f80}}, // IQ3_XXS
    {{13,true,true},{0x7f80,0x7f80,0x7f80},{0x7f80,0x7f80,0x7f80}}, // IQ2_S
    {{14,true,true},{0x7f80,0x7f80,0x7f80},{0x7f80,0x7f80,0x7f80}}, // IQ2_XS
    {{15,true,true},{0x7f00,0x7f80,0x7f80},{0x7f00,0x7f80,0x7f80}}, // IQ2_XXS
    {{16,true,true},{0x7800,0x7f80,0x7800},{0x7800,0x7f80,0x7800}}, // IQ1_S
    {{17,true,true},{0x7800,0x7e00,0x7800},{0x7000,0x7f00,0x7800}}, // IQ1_M
    {{19,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q8_0
    {{20,false,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q8_1
    {{21,true,true},{0x7ff8,0x7ff8,0x7ff8},{0x7ff8,0x7ff8,0x7ff8}}, // Q8_K
}};
static_assert([] {
    for(size_t i=0;i<kVerifierMeasuredWidths.size();++i) {
        const auto &row=kVerifierMeasuredWidths[i];
        if(!row.source.present || canonicalDeviceVnniCodebookId(row.source)==255)return false;
        for(auto mask:row.projection_rows)if(mask&0x8000u)return false;
        for(auto mask:row.swiglu_rows)if(mask&0x8000u)return false;
        for(size_t j=0;j<i;++j)if(row.source==kVerifierMeasuredWidths[j].source)return false;
    }
    return true;
}(),"Measured verifier source tuples must be unique/catalogued and widths bounded to M2..16");

/** @brief Resolve a measured complete operation without observing device-owned live rows.
 * @param operation Authenticated original public operation. @param m Exact physical width.
 * @return Whether this source/operation/geometry/width has qualified native evidence. */
bool measuredVerifierWmmaKey(const NativeQuantisedVerifierOperation &operation,int m) noexcept
{
    if(m<2 || m>16)return false;
    const int shape=operation.n==17408 && operation.k==5120 ? 0 :
        operation.n==5120 && operation.k==17408 ? 1 :
        operation.n==10240 && operation.k==5120 ? 2 : -1;
    if(shape<0)return false;
    for(const auto &row:kVerifierMeasuredWidths)if(row.source==operation.source) {
        const auto *masks=operation.kind==NativeQuantisedVerifierOperationKind::Projection ? &row.projection_rows :
            operation.kind==NativeQuantisedVerifierOperationKind::SwiGLUDown ? &row.swiglu_rows : nullptr;
        return masks && (((*masks)[shape] & (uint16_t{1}<<(m-2)))!=0);
    }
    return false;
}
}

bool measuredNativeQuantisedVerifierBundle(const NativeQuantisedVerifierBundleOperation &operation) noexcept
{
    const int m = operation.rows.physicalRows();
    const bool qkv = operation.kind == VerifierProjectionBundleKind::AttentionQKV;
    if (m < 2 || m > 16 ||
        (qkv ? operation.count != 3 :
            operation.kind != VerifierProjectionBundleKind::AttentionKV || operation.count != 2))
        return false;
    const auto &source = operation.members[0].source;
    for (int member = 0; member < operation.count; ++member)
        if (operation.members[member].source != source) return false;
    const auto n = [&](int member) { return operation.members[member].n; };
    int shape = -1;
    if (qkv)
    {
        if (operation.k == 5120 && n(0) == 12288 && n(1) == 1024 && n(2) == 1024) shape = 0;
        if (operation.k == 5120 && n(0) == 6144 && n(1) == 512 && n(2) == 512) shape = 2;
        if (operation.k == 2048 && n(0) == 8192 && n(1) == 512 && n(2) == 512) shape = 4;
        if (operation.k == 2048 && n(0) == 2048 && n(1) == 512 && n(2) == 512) shape = 6;
    }
    else
    {
        if (operation.k == 5120 && n(0) == 1024 && n(1) == 1024) shape = 1;
        if (operation.k == 5120 && n(0) == 512 && n(1) == 512) shape = 3;
        if (operation.k == 2048 && n(0) == 512 && n(1) == 512) shape = 5;
    }
    if (shape < 0) return false;
    for (const auto &entry : kVerifierBundleMeasuredWidths)
        if (entry.source == source) return (entry.rows[shape] & (uint16_t{1} << (m-2))) != 0;
    return false;
}

namespace
{
/** @brief Source-specific complete GDN masks; columns preserve QKV arithmetic. */
struct GDNVerifierMeasuredWidths
{
    NativeQuantisedSourceIdentity source;
    std::array<uint16_t,2> rows;
};
#include "ROCmNativeQuantisedGDNVerifierPolicy.inc"
static_assert([] {
    for(size_t i=0;i<kGDNVerifierMeasuredWidths.size();++i) {
        const auto &entry=kGDNVerifierMeasuredWidths[i];
        if(!entry.source.present || canonicalDeviceVnniCodebookId(entry.source)==255)return false;
        for(auto mask:entry.rows)if(mask & 0x8000u)return false;
        for(size_t j=0;j<i;++j)if(entry.source==kGDNVerifierMeasuredWidths[j].source)return false;
    }
    return true;
}(),"Complete GDN masks require unique source identities and widths M2..16");
}

bool measuredNativeQuantisedGDNVerifier(const NativeQuantisedGDNVerifierOperation &operation) noexcept
{
    const int m=operation.rows.physicalRows();
    if(m<2 || m>16 || !validGDNOperation(operation) || operation.k!=2560 ||
        operation.qkv.n!=10240 || operation.z.n!=6144 || operation.alpha.weights.n!=48 ||
        operation.beta.weights.n!=48 || operation.qkv.source!=operation.z.source)return false;
    const int mode=operation.precision==QuantizedActivationOperandPrecision::TwoTermQ8?1:0;
    for(const auto &entry:kGDNVerifierMeasuredWidths)
        if(entry.source==operation.qkv.source)return (entry.rows[mode] & (uint16_t{1}<<(m-2)))!=0;
    return false;
}

NativeQuantisedVerifierCandidate nativeQuantisedVerifierProducer(
    int device, bool decode_equivalent, const NativeQuantisedVerifierWork &work)
{
    if (verifier_selection) {
        if (verifier_selection->device != device)
            throw std::invalid_argument("Native verifier candidate belongs to another device");
        if (verifier_selection->candidate == NativeQuantisedVerifierCandidate::Dot4)
            return NativeQuantisedVerifierCandidate::Dot4;
        if (verifier_selection->candidate != NativeQuantisedVerifierCandidate::Automatic) {
            if (!decode_equivalent)
                throw std::invalid_argument("Native verifier WMMA requires serial-M1 arithmetic scope");
            return verifier_selection->candidate;
        }
    }
    if (const auto *bundle = ScopedNativeQuantisedVerifierBundle::current())
    {
        const bool qualified = decode_equivalent && !currentProjectionRowWork(work.rows.physicalRows()) &&
            ScopedCapturedVerifierExtent::current().matches(DeviceId::rocm(device), work.rows) &&
            matchesNativeQuantisedVerifierBundle(*bundle,work,device) && measuredNativeQuantisedVerifierBundle(*bundle) &&
            nativeQuantisedHardwareForDevice(device).architecture == native_quantised::NativeQuantisedArchitecture::rocm(11,0,0);
        return qualified ? NativeQuantisedVerifierCandidate::WmmaM16N64 : NativeQuantisedVerifierCandidate::Dot4;
    }
    if (const auto *gdn = ScopedNativeQuantisedGDNVerifier::current())
    {
        const bool qualified = decode_equivalent && !currentProjectionRowWork(work.rows.physicalRows()) &&
            ScopedCapturedVerifierExtent::current().matches(DeviceId::rocm(device),work.rows) &&
            matchesNativeQuantisedGDNVerifier(*gdn,work,device) && measuredNativeQuantisedGDNVerifier(*gdn) &&
            nativeQuantisedHardwareForDevice(device).architecture == native_quantised::NativeQuantisedArchitecture::rocm(11,0,0);
        return qualified ? NativeQuantisedVerifierCandidate::WmmaM16N64 : NativeQuantisedVerifierCandidate::Dot4;
    }
    // An unmeasured original bundle cannot gain scalar eligibility by splitting
    // into a same-shaped singleton subgroup or by failing provenance admission.
    if (ScopedVerifierProjectionBundle::current() || ScopedVerifierGDNBundle::current()) return NativeQuantisedVerifierCandidate::Dot4;
    const auto operation = ScopedNativeQuantisedVerifierOperation::current();
    const int m = work.rows.physicalRows();
    if (!decode_equivalent || !operation || m < 2 || m > 16 || work.projections != 1 ||
        currentProjectionRowWork(m) || work.policy != NativeQuantisedVerifierPolicyMatch::CanonicalSerial ||
        !operation->source.present ||
        work.physical_codebook != canonicalDeviceVnniCodebookId(operation->source) ||
        operation->device != device || operation->m != m || operation->n != work.n || operation->k != work.k ||
        !operation->weights || operation->weights != work.weights ||
        !ScopedCapturedVerifierExtent::current().matches(DeviceId::rocm(device), work.rows))
        return NativeQuantisedVerifierCandidate::Dot4;
    if (operation->kind != NativeQuantisedVerifierOperationKind::Projection &&
        operation->kind != NativeQuantisedVerifierOperationKind::SwiGLUDown)
        throw std::invalid_argument("Unknown original verifier operation");
    // A larger fused operation cannot acquire eligibility by splitting into
    // singleton leaves. Every source, operation and physical width owns its key.
    if (!measuredVerifierWmmaKey(*operation,m) || nativeQuantisedHardwareForDevice(device).architecture !=
        native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        return NativeQuantisedVerifierCandidate::Dot4;
    return NativeQuantisedVerifierCandidate::WmmaM16N64;
}
}
