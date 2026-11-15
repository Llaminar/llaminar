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
#include "kernels/common/ProjectionWorkLaunch.h"
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
    const auto operation = ScopedNativeQuantisedVerifierOperation::current();
    const int m = work.rows.physicalRows();
    if (!decode_equivalent || !operation || m < 5 || m > 16 || work.projections != 1 ||
        currentProjectionRowWork(m) || work.policy != NativeQuantisedVerifierPolicyMatch::CanonicalSerial ||
        work.physical_codebook != kNativeQuantisedIQ4XSSourceScaleCodebook ||
        !operation->source.present || operation->source.codebook_id != 4 || !operation->source.is_superblock ||
        operation->device != device || operation->m != m || operation->n != work.n || operation->k != work.k ||
        !operation->weights || operation->weights != work.weights ||
        !ScopedCapturedVerifierExtent::current().matches(DeviceId::rocm(device), work.rows))
        return NativeQuantisedVerifierCandidate::Dot4;
    if (operation->kind != NativeQuantisedVerifierOperationKind::Projection &&
        operation->kind != NativeQuantisedVerifierOperationKind::SwiGLUDown)
        throw std::invalid_argument("Unknown original verifier operation");
    // Complete public operations, exact physical M2..16, 31 AB/BA pairs:
    // Dot4 wins every M2..4 key; native WMMA wins every M5..16 key. A larger
    // fused operation cannot acquire eligibility by splitting into singleton leaves.
    const bool measured_shape = (work.n == 17408 && work.k == 5120) ||
        (work.n == 5120 && work.k == 17408) || (work.n == 10240 && work.k == 5120);
    if (!measured_shape || nativeQuantisedHardwareForDevice(device).architecture !=
        native_quantised::NativeQuantisedArchitecture::rocm(11,0,0))
        return NativeQuantisedVerifierCandidate::Dot4;
    return NativeQuantisedVerifierCandidate::WmmaM16N64;
}
}
