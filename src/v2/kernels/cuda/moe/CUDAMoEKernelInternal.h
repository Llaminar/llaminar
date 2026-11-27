/**
 * @file CUDAMoEKernelInternal.h
 * @brief Shared CUDA MoE extern "C" declarations and descriptor/geometry validation helpers for CUDAMoEKernel shards.
 */
#pragma once
/**
 * @file CUDAMoEKernel.cpp
 * @brief CUDA MoE kernel bridge implementation.
 *
 * The C++ bridge owns tensor coherence, persistent scratch allocation, and
 * error reporting while delegating CUDA launches to typed backend wrappers.
 * This keeps MPI-heavy project headers out of nvcc
 * compilation and preserves the established CUDA backend split used by other
 * kernels. Compact projection export borrows the runtime grouping's exact
 * device count; private regrouping scratch is not an alternate count authority.
 * Import restores the existing activation bits, without host inspection or an
 * extra quantization boundary, before the unchanged down projection.
 * Grouped down publications admit dirty output storage through a checked live
 * overwrite extent. Only a successfully enqueued complete writer can publish
 * that output's exact-stream event; inactive routes still own explicit zeros.
 * IQ4_XS physical format 24 preserves the original half base and raw six-bit
 * coefficient in separate 16-bit planes. Its LUT payload and source4 reduction
 * schedule are unchanged; the coefficient is storage metadata, never a minimum.
 * Decode refreshes only its selected expert descriptors from the acquired
 * bank; prefill continues to publish the complete grouped descriptor plan.
 * Native M1 gate reduction publishes SwiGLU codes directly to the Down launch.
 */

#include "kernels/common/MoERuntimeDescriptorLaunch.h"
#include "kernels/common/RouterQ8Launch.h"
#include "kernels/common/OriginalM1HiddenQ8Publication.h"
#include "CUDAMoEKernel.h"
#include "CUDARouterQ8Prefill.h"
#include "kernels/common/MoEGroupedFloatingPrefillKernels.h"
#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "kernels/common/MoEGroupingInitialization.h"
#include "CUDAMoEBatchInvariantPolicy.h"
#include "CUDAMoEOverlayActivationPacketKernels.h"
#include "CUDAMoEOverlayDeviceControllerKernels.h"
#include "CUDAMoEOverlayEpochKernels.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/cuda/CUDAHardwareMetadata.h"
#include "../gemm/CUDADeviceWorkspace.h"
#include "../gemm/CUDAMoEGroupedPrefillKernels.h"
#include "../gemm/CUDAMoEProductionPrefillPolicy.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/NativeQuantisedFormatInfo.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

extern "C" bool cudaNativeQuantisedInitIQGridTables_tuned();


namespace llaminar2::cudamoe_detail
{
    bool ensureCudaMoEIQGridTablesInitialized(int device_ordinal);
} // namespace llaminar2::cudamoe_detail
using llaminar2::cudamoe_detail::ensureCudaMoEIQGridTablesInitialized;

namespace
{
    constexpr const char *kCudaGroupedDownDescriptorLeaseDomain =
        "cuda_moe_grouped_down_descriptors";
    constexpr const char *kCudaGroupedGateUpDescriptorLeaseDomain =
        "cuda_moe_grouped_gateup_descriptors";
    constexpr const char *kCudaGroupedExpertMaskLeaseDomain =
        "cuda_moe_grouped_expert_masks";
    constexpr const char *kCudaRouterQ8GatePublicationDomain =
        "cuda_moe_router_q8_gate_cache";

    /**
     * @brief Preserve every router identity component for workspace sharing.
     */
    llaminar2::PersistentWorkspacePublicationKey routerPublicationKey(
        const llaminar2::DeviceResidentRouterGateCacheKey &key) noexcept
    {
        return {
            .word0 = key.workspace_id,
            .word1 = static_cast<std::uint64_t>(key.source_device_ptr),
            .word2 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(key.d_model)),
            .word3 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(key.num_experts)) |
                (static_cast<std::uint64_t>(key.basis) << 32u),
        };
    }

    /**
     * @brief Append every semantic descriptor field to an exact identity key.
     *
     * Device pointer values are the model-lifetime weight identity consumed by
     * captured grouped kernels. Recording fields individually avoids comparing
     * compiler padding and avoids reducing correctness to a collision-prone
     * digest.
     */
    void appendGroupedDescriptorIdentity(
        std::vector<std::uint64_t> &words,
        const llaminar2::DeviceNativeQuantisedMatrixDesc &descriptor)
    {
        words.push_back(static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(descriptor.payload)));
        words.push_back(static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(descriptor.scales)));
        words.push_back(static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(descriptor.mins)));
        words.push_back(static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(descriptor.emins)));
        words.push_back(static_cast<std::uint64_t>(
            static_cast<std::uint32_t>(descriptor.n)));
        words.push_back(static_cast<std::uint64_t>(
            static_cast<std::uint32_t>(descriptor.k)));
        words.push_back(
            static_cast<std::uint64_t>(descriptor.blocks_per_row));
        words.push_back(
            static_cast<std::uint64_t>(descriptor.codebook_id) |
            (static_cast<std::uint64_t>(
                 descriptor.allocation_payload_bytes_per_block)
             << 8U) |
            (static_cast<std::uint64_t>(
                 descriptor.allocation_has_mins)
             << 16U) |
            (static_cast<std::uint64_t>(
                 descriptor.allocation_has_emins)
             << 24U));
        words.push_back(
            static_cast<std::uint64_t>(descriptor.source_codebook_id) |
            (static_cast<std::uint64_t>(descriptor.source_is_superblock) << 8U) |
            (static_cast<std::uint64_t>(descriptor.source_identity_present) << 16U));
    }

    /**
     * @brief Build a collision-free identity for one grouped descriptor table.
     *
     * @param primary Down descriptors, or gate descriptors for a paired table.
     * @param secondary Up descriptors for a paired table; null for down tables.
     * @param num_experts Number of records in each present descriptor array.
     * @param d_model Model hidden width.
     * @param intermediate Routed expert intermediate width.
     */
    llaminar2::PersistentWorkspacePublicationKey
    groupedDescriptorPublicationKey(
        const llaminar2::DeviceNativeQuantisedMatrixDesc *primary,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *secondary,
        int num_experts,
        int d_model,
        int intermediate,
        llaminar2::MoEDecodeDescriptorSource descriptor_source)
    {
        llaminar2::PersistentWorkspacePublicationKey key{
            .word0 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(num_experts)),
            .word1 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(d_model)),
            .word2 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(intermediate)),
            .word3 = (secondary ? 2U : 1U) | (static_cast<std::uint64_t>(descriptor_source) << 16U),
        };
        key.identity_words.reserve(
            static_cast<std::size_t>(num_experts) *
            (secondary ? 18U : 9U));
        for (int expert = 0; expert < num_experts; ++expert)
        {
            appendGroupedDescriptorIdentity(
                key.identity_words,
                primary[expert]);
        }
        if (secondary)
        {
            for (int expert = 0; expert < num_experts; ++expert)
            {
                appendGroupedDescriptorIdentity(
                    key.identity_words,
                    secondary[expert]);
            }
        }
        return key;
    }

    /**
     * @brief Build an exact identity for a contiguous-floating descriptor table.
     *
     * Pointer identity is part of captured graph identity: two prepared engines
     * with equal values but different allocations cannot share a publication
     * whose replay embeds the former address.
     */
    llaminar2::PersistentWorkspacePublicationKey
    groupedFloatingDescriptorPublicationKey(
        const llaminar2::DeviceMoEFloatingMatrixDesc *primary,
        const llaminar2::DeviceMoEFloatingMatrixDesc *secondary,
        llaminar2::DeviceMoEWeightFormat format,
        int num_experts,
        int d_model,
        int intermediate,
        llaminar2::MoEDecodeDescriptorSource descriptor_source)
    {
        llaminar2::PersistentWorkspacePublicationKey key{
            .word0 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(num_experts)),
            .word1 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(d_model)),
            .word2 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(intermediate)),
            .word3 = 0x100U | (secondary ? 2U : 1U) |
                     (static_cast<std::uint64_t>(descriptor_source) << 16U) |
                     (static_cast<std::uint64_t>(format) << 32U),
        };
        key.identity_words.reserve(
            static_cast<std::size_t>(num_experts) *
            (secondary ? 6U : 3U));
        auto append = [&](const llaminar2::DeviceMoEFloatingMatrixDesc &desc)
        {
            key.identity_words.push_back(static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(desc.data)));
            key.identity_words.push_back(static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(desc.n)));
            key.identity_words.push_back(static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(desc.k)));
        };
        for (int expert = 0; expert < num_experts; ++expert)
            append(primary[expert]);
        if (secondary)
        {
            for (int expert = 0; expert < num_experts; ++expert)
                append(secondary[expert]);
        }
        return key;
    }

    /**
     * @brief Validate the stage-owned stream for a device-resident MoE launch.
     *
     * Rebalance and LLEP calls must not consult CUDAKernelBase::gpu_stream_:
     * that member belongs to the legacy device-wide binding and can be
     * retargeted by another graph capture thread. The immutable context is the
     * sole stream authority for these migrated primitives.
     */
    void *explicitMoELaunchStream(
        const llaminar2::MoEKernelLaunchContext &launch,
        const char *operation)
    {
        if (!launch.hasExplicitStream())
        {
            LOG_ERROR("[CUDAMoEKernel] " << operation
                                         << " requires an explicit stage-owned CUDA stream");
            return nullptr;
        }
        return launch.stream;
    }

    bool setMoEDevice(int device_ordinal, const char *context)
    {
        cudaError_t err = cudaSetDevice(device_ordinal);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] cudaSetDevice(" << device_ordinal
                                                        << ") failed in " << context << ": " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    llaminar2::ITensor *requireTensor(llaminar2::ITensor *tensor, const char *context)
    {
        if (!tensor)
            LOG_ERROR("[CUDAMoEKernel] " << context << " requires a tensor");
        return tensor;
    }

    bool requireTensorOnDevice(llaminar2::ITensor *tensor, llaminar2::DeviceId device, void *stream, const char *name)
    {
        (void)name;
        llaminar2::TransferEngine::requireDeviceInput(tensor, device, stream);
        return true;
    }

    bool requireOutputOnDevice(llaminar2::ITensor *tensor, llaminar2::DeviceId device, void *stream, const char *name)
    {
        (void)name;
        llaminar2::TransferEngine::requireDeviceOutput(tensor, device, stream);
        return true;
    }

    /**
     * @brief Publish an asynchronous CUDA tensor write through TransferEngine.
     *
     * The kernel wrapper supplies the exact producer device and stream. The
     * transfer service owns concrete tensor validation and the event-backed
     * coherence transition consumed by later streams.
     */
    void markDeviceWritten(llaminar2::ITensor *tensor, llaminar2::DeviceId device, void *stream)
    {
        llaminar2::TransferEngine::publishDeviceWrite(
            tensor,
            device,
            stream);
    }

    void markSynced(llaminar2::ITensor *tensor)
    {
        llaminar2::TransferEngine::publishSynchronized(tensor);
    }

    llaminar2::PerfStatsCollector::Tags groupedDecodeTags(
        const char *source,
        int active_slots,
        int d_model,
        int intermediate,
        const char *route)
    {
        return {
            {"source", source},
            {"active_slots", std::to_string(active_slots)},
            {"d_model", std::to_string(d_model)},
            {"intermediate", std::to_string(intermediate)},
            {"route", route}};
    }

    void recordGroupedDecodeCounter(
        const char *name,
        const char *source,
        int active_slots,
        int d_model,
        int intermediate,
        const char *route)
    {
        llaminar2::PerfStatsCollector::addCounter(
            "kernel", name, 1.0, {}, {},
            groupedDecodeTags(source, active_slots, d_model, intermediate, route));
    }

    bool isAligned(const void *ptr, std::uintptr_t alignment)
    {
        return ptr && ((reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0);
    }

    bool requireAlignedPointer(const void *ptr, std::uintptr_t alignment, const char *name, const char *context)
    {
        if (isAligned(ptr, alignment))
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << alignment
                                      << "-byte aligned " << name << " pointer, got " << ptr);
        return false;
    }

    bool requireTensorType(const llaminar2::ITensor *tensor, llaminar2::TensorType expected,
                           const char *name, const char *context)
    {
        if (tensor && tensor->native_type() == expected)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name << " type "
                                      << llaminar2::tensorTypeName(expected)
                                      << ", got "
                                      << (tensor ? llaminar2::tensorTypeName(tensor->native_type()) : "null"));
        return false;
    }

    bool requireTensorTypeOneOf(const llaminar2::ITensor *tensor,
                                llaminar2::TensorType a,
                                llaminar2::TensorType b,
                                const char *name,
                                const char *context)
    {
        if (tensor && (tensor->native_type() == a || tensor->native_type() == b))
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name << " type "
                                      << llaminar2::tensorTypeName(a) << " or "
                                      << llaminar2::tensorTypeName(b)
                                      << ", got "
                                      << (tensor ? llaminar2::tensorTypeName(tensor->native_type()) : "null"));
        return false;
    }

    /**
     * @brief True when MoE prefill grouping wrapper tracing is requested.
     *
     * This intentionally uses the same environment flag as the stage-level
     * trace.  The wrapper log is a lower observation point for captured prefill
     * graphs where the ordinary stage body may not be visible during replay.
     */
    bool traceMoEPrefillGroupingWrapperEnabled()
    {
        const bool assignment_trace =
            !llaminar2::DebugEnv::isFalseyEnv("LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE");
        return assignment_trace || llaminar2::debugEnv().execution.prefill_graph_trace;
    }

    bool requireTensorElements(const llaminar2::ITensor *tensor, size_t required,
                               const char *name, const char *context)
    {
        if (tensor && tensor->numel() >= required)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " to have at least " << required
                                      << " elements, got " << (tensor ? tensor->numel() : 0));
        return false;
    }

    bool requireMatrixCapacity(const llaminar2::ITensor *tensor,
                               int required_rows,
                               int required_cols,
                               const char *name,
                               const char *context)
    {
        if (!tensor || required_rows < 0 || required_cols < 0)
            return false;
        const auto &shape = tensor->shape();
        const bool ok = shape.size() >= 2 &&
                        shape[0] >= static_cast<size_t>(required_rows) &&
                        shape[1] >= static_cast<size_t>(required_cols);
        if (ok)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " matrix capacity at least "
                                      << required_rows << "x" << required_cols
                                      << ", got shape "
                                      << (shape.empty() ? 0 : shape[0]) << "x"
                                      << (shape.size() > 1 ? shape[1] : 1));
        return false;
    }

    bool requireCudaDevicePointer(const void *ptr, int expected_device,
                                  const char *name, const char *context, void *stream)
    {
        if (!ptr)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context << " requires non-null " << name << " pointer");
            return false;
        }
        if (llaminar2::isGraphCaptureActive())
            return true;
        if (stream)
        {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            const cudaError_t capture_err =
                cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &capture_status);
            if (capture_err == cudaSuccess && capture_status != cudaStreamCaptureStatusNone)
                return true;
            if (capture_err != cudaSuccess)
                cudaGetLastError();
        }

        cudaPointerAttributes attr{};
        const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            LOG_ERROR("[CUDAMoEKernel] " << context << " cudaPointerGetAttributes failed for "
                                          << name << " pointer " << ptr << ": "
                                          << cudaGetErrorString(err));
            return false;
        }
        // Snapshot-enabled activation tensors may be zero-copy mapped host memory.
        // They are valid CUDA kernel inputs only when the pointer is the
        // device-visible alias returned by cudaHostGetDevicePointer(), not an
        // arbitrary host pointer.
        const bool mapped_device_alias =
            attr.type == cudaMemoryTypeHost &&
            attr.device == expected_device &&
            attr.devicePointer == ptr;
        const bool type_ok = attr.type == cudaMemoryTypeDevice ||
                             attr.type == cudaMemoryTypeManaged ||
                             mapped_device_alias;
        if (type_ok && attr.device == expected_device)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " pointer on CUDA device " << expected_device
                                      << ", got attr.device=" << attr.device
                                      << " attr.type=" << static_cast<int>(attr.type)
                                      << " device_ptr=" << attr.devicePointer
                                      << " host_ptr=" << attr.hostPointer
                                      << " ptr=" << ptr);
        return false;
    }

    int selectGroupedPrefillTileM(int requested_tile_m, int max_tokens_per_expert)
    {
        switch (requested_tile_m)
        {
        case 2:
        case 4:
        case 8:
        case 16:
            return requested_tile_m;
        default:
            break;
        }

        // Runtime-M verifier prefill uses fixed-size grouped tiles. The compact
        // TM=2 template wins for very small active groups; larger groups widen
        // the tile without changing the row-wise arithmetic contract.
        if (max_tokens_per_expert <= 4)
            return 2;
        if (max_tokens_per_expert <= 8)
            return 8;
        return 16;
    }

    /** Stable generic width used when an exact production key was not swept. */
    constexpr int kCUDAMoEGenericGateUpOrderedTileN = 128;

    /** Capture-time launch geometry and its reviewable dispatch provenance. */
    struct CUDAMoEProductionPrefillPolicy
    {
        int gateup_tile_n = 0;
        llaminar2::cuda::moe::GroupedImmaColumns imma_gateup_columns =
            llaminar2::cuda::moe::GroupedImmaColumns::Columns32;
        llaminar2::cuda::moe::GroupedImmaColumns imma_down_columns =
            llaminar2::cuda::moe::GroupedImmaColumns::Columns32;
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule
            imma_gateup_schedule = llaminar2::cuda::moe::
                GroupedImmaGateUpSchedule::ParallelProjections;
        const char *source = "missing";
    };

    /** Return whether a width names one compiled arithmetic-neutral candidate. */
    bool validCUDAMoEGateUpOrderedTileN(int tile_n) noexcept
    {
        return tile_n >= 64 && tile_n <= 256 && (tile_n % 32) == 0;
    }

    /**
     * @brief Select the sole execution family without changing descriptor masks.
     * @param codebook_mask Every actual physical format in the retained table.
     * @return The single family, or -1 for unrelated mixed formats.
     */
    int singleCUDAMoECodebook(uint32_t codebook_mask) noexcept
    {
        // Family24 executes both IQ4 descriptors without changing their mask.
        if (codebook_mask & (uint32_t{1} << llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook))
            codebook_mask &= ~(uint32_t{1} << 4);
        if (codebook_mask == 0 ||
            (codebook_mask & (codebook_mask - 1U)) != 0U)
        {
            return -1;
        }

        int codebook = 0;
        while ((codebook_mask >>= 1U) != 0U)
            ++codebook;
        return codebook;
    }

    /**
     * @brief Format one descriptor-table codebook mask for stable PerfStats.
     *
     * CUDA and ROCm route diagnostics intentionally use the same fixed-width
     * hexadecimal representation. This preserves every descriptor bit, makes
     * mixed tables visible, and lets benchmark tooling compare backend records
     * without interpreting signed decimal values or locale-specific output.
     * Launch policy continues to consume the original integer mask.
     *
     * @param codebook_mask Exact OR-reduction of every descriptor codebook id.
     * @return Lowercase fixed-width hexadecimal in @c 0x00000000 form.
     */
    std::string cudaCodebookMaskTag(uint32_t codebook_mask)
    {
        char text[11]{};
        std::snprintf(text, sizeof(text), "0x%08x", codebook_mask);
        return text;
    }

    /**
     * @brief Select the complete CUDA gate/up geometry before graph capture.
     *
     * Exact Qwen production keys take precedence over the total generic policy.
     * Heterogeneous descriptor tables cannot name one exact codebook pair and
     * therefore retain the generic width explicitly. The Perf__ tournament is
     * the sole higher-precedence control: its active bit distinguishes a forced
     * 128-thread candidate from the ordinary generic value 128. Any malformed
     * selected width remains invalid and causes the production caller to fail;
     * it is never replaced after selection.
     * @param device_ordinal The admitted capture owner. Device-free queries
     *        omit it and use the architecture-independent policy.
     */
    CUDAMoEProductionPrefillPolicy selectCUDAMoEProductionPrefillPolicy(
        llaminar2::MoEPrefillMeasuredOperation operation,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int seq_len,
        int hidden_size,
        int expert_width,
        int expert_count,
        int top_k,
        bool use_gateup_kpart,
        int device_ordinal = -1)
    {
        if (gateup_codebook_mask == 0 || down_codebook_mask == 0 ||
            seq_len <= 0 || hidden_size <= 0 || expert_width <= 0 ||
            expert_count <= 0 || top_k <= 0 || top_k > expert_count)
        {
            return {};
        }

        if (!use_gateup_kpart)
        {
            const auto grouped_imma =
                llaminar2::cuda::moe::selectGroupedImmaLaunchPolicy(
                    operation,
                    singleCUDAMoECodebook(gateup_codebook_mask),
                    singleCUDAMoECodebook(down_codebook_mask),
                    seq_len,
                    hidden_size,
                    expert_width,
                    expert_count,
                    top_k);
            CUDAMoEProductionPrefillPolicy policy{
                .gateup_tile_n = llaminar2::cuda::moe::groupedImmaColumns(
                    grouped_imma.gate_up_columns),
                .imma_gateup_columns = grouped_imma.gate_up_columns,
                .imma_down_columns = grouped_imma.down_columns,
                .imma_gateup_schedule = grouped_imma.gate_up_schedule,
                .source = grouped_imma.exact_overlay
                              ? "tensor_core_imma_exact_overlay"
                              : "tensor_core_imma_generic",
            };
            const auto &gemm = llaminar2::debugEnv().gemm;
            if (gemm.cuda_moe_imma_geometry_override_active)
            {
                policy.imma_gateup_columns =
                    static_cast<llaminar2::cuda::moe::GroupedImmaColumns>(
                        gemm.cuda_moe_imma_gateup_columns);
                policy.imma_down_columns =
                    static_cast<llaminar2::cuda::moe::GroupedImmaColumns>(
                        gemm.cuda_moe_imma_down_columns);
                policy.imma_gateup_schedule = static_cast<
                    llaminar2::cuda::moe::GroupedImmaGateUpSchedule>(
                    gemm.cuda_moe_imma_gateup_schedule);
                policy.gateup_tile_n =
                    gemm.cuda_moe_imma_gateup_columns;
                policy.source = "tensor_core_imma_explicit_override";
            }
            if (!llaminar2::cuda::moe::validGroupedImmaGateUpColumns(
                    policy.imma_gateup_columns) ||
                !llaminar2::cuda::moe::validGroupedImmaDownColumns(
                    policy.imma_down_columns) ||
                !llaminar2::cuda::moe::validGroupedImmaGateUpSchedule(
                    policy.imma_gateup_schedule))
            {
                return {};
            }
            return policy;
        }

        CUDAMoEProductionPrefillPolicy policy{
            .gateup_tile_n = kCUDAMoEGenericGateUpOrderedTileN,
            .source = "generic_mixed_codebooks",
        };
        const int gateup_codebook =
            singleCUDAMoECodebook(gateup_codebook_mask);
        const int down_codebook = singleCUDAMoECodebook(down_codebook_mask);
        if (gateup_codebook >= 0 && down_codebook >= 0)
        {
            policy.source = "generic";
            int exact_tile_n = 0;
            if (llaminar2::cuda::
                    selectCUDAMoEProductionPrefillOverlayForOperation(
                        operation,
                        static_cast<uint8_t>(gateup_codebook),
                        static_cast<uint8_t>(down_codebook),
                        hidden_size,
                        expert_width,
                        expert_count,
                        top_k,
                        seq_len,
                        exact_tile_n))
            {
                policy.gateup_tile_n = exact_tile_n;
                policy.source = "exact_overlay";
            }
            else if (device_ordinal >= 0 && seq_len == 4 && top_k == 10 &&
                     hidden_size == llaminar2::CUDAMoEBatchInvariantPolicy::qfn_gate_up_k &&
                     expert_width == llaminar2::CUDAMoEBatchInvariantPolicy::qfn_gate_up_n &&
                     llaminar2::CUDAMoEBatchInvariantPolicy::qfnPackedGateUp(
                         static_cast<uint8_t>(gateup_codebook)) &&
                     gateup_codebook != llaminar2::native_quantised_formats::Q2_K.codebook_id &&
                     gateup_codebook != llaminar2::native_quantised_formats::Q6_K.codebook_id)
            {
                const auto &hardware = llaminar2::cudaHardwareMetadata().at(device_ordinal);
                if (hardware.sm_major == 8 && hardware.sm_minor == 6)
                {
                    policy.gateup_tile_n = 64;
                    policy.source = "sm86_qfn_verifier";
                }
            }
        }

        const auto &gemm = llaminar2::debugEnv().gemm;
        if (gemm.cuda_moe_gateup_ordered_kpart_tile_n_override_active)
        {
            policy.gateup_tile_n =
                gemm.cuda_moe_gateup_ordered_kpart_tile_n;
            policy.source = "explicit_override";
        }
        if (!validCUDAMoEGateUpOrderedTileN(policy.gateup_tile_n))
            return {};
        return policy;
    }

    llaminar2::PerfStatsCollector::Tags groupedPrefillTags(
        int seq_len,
        int top_k,
        int num_experts,
        int active_expert_slots,
        int tile_m,
        int tile_n)
    {
        const int total_slots = seq_len * top_k;
        return {
            {"seq_len", std::to_string(seq_len)},
            {"top_k", std::to_string(top_k)},
            {"total_slots", std::to_string(total_slots)},
            {"activation_quant_rows", std::to_string(seq_len)},
            {"active_expert_slots", std::to_string(active_expert_slots)},
            {"num_experts", std::to_string(num_experts)},
            {"tile_m", std::to_string(tile_m)},
            {"tile_n", std::to_string(tile_n)}};
    }

    void recordGroupedPrefillCounters(
        int seq_len,
        int top_k,
        int num_experts,
        int active_expert_slots,
        int tile_m,
        int tile_n,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        uint32_t gateup_policy_codebook_mask,
        uint32_t down_policy_codebook_mask,
        bool use_grouped_imma,
        bool use_gateup_kpart,
        bool fuse_swiglu_requested,
        bool use_ordered_down_kpart,
        bool ordered_scatter,
        bool canonical_route_publication,
        int splitk_tile_rows,
        llaminar2::cuda::moe::GroupedImmaColumns imma_gateup_columns,
        llaminar2::cuda::moe::GroupedImmaColumns imma_down_columns,
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule
            imma_gateup_schedule,
        const char *policy_source)
    {
        auto tags = groupedPrefillTags(seq_len, top_k, num_experts, active_expert_slots, tile_m, tile_n);
        /*
         * `tile_m` belongs only to the full-K grouped launcher.  Ordered KPART
         * instead traverses a bounded row workspace named by
         * `splitk_tile_rows`; reporting only the legacy numeric fields made a
         * zero `tile_m` look like missing geometry to downstream gates.  Keep
         * both legacy dimensions for corpus compatibility, but publish the
         * authoritative geometry owner so tests and profilers cannot confuse
         * two launch families with different arithmetic contracts.
         */
        tags["gateup_geometry_contract"] = use_grouped_imma
                                                   ? "tensor_core_imma"
                                                   : (use_gateup_kpart
                                                          ? "ordered_split_k"
                                                          : "full_k");
        tags["row_tile_source"] = use_grouped_imma
                                          ? "device_work_directory"
                                          : (use_gateup_kpart
                                                 ? "splitk_tile_rows"
                                                 : "tile_m");
        if (use_grouped_imma)
        {
            tags["work_scheduler"] = "compact_directory_grid";
            tags["empty_directory_tail"] = "sentinel_cta_exit";
            tags["imma_gateup_columns"] = std::to_string(
                llaminar2::cuda::moe::groupedImmaColumns(
                    imma_gateup_columns));
            tags["imma_down_columns"] = std::to_string(
                llaminar2::cuda::moe::groupedImmaColumns(
                    imma_down_columns));
            tags["imma_gateup_schedule"] =
                imma_gateup_schedule == llaminar2::cuda::moe::
                                               GroupedImmaGateUpSchedule::
                                                   PairedProjections
                    ? "paired_projections"
                    : "parallel_projections";
        }
        tags["gateup_codebook_mask"] =
            cudaCodebookMaskTag(gateup_codebook_mask);
        tags["down_codebook_mask"] =
            cudaCodebookMaskTag(down_codebook_mask);
        tags["gateup_policy_codebook_mask"] =
            cudaCodebookMaskTag(gateup_policy_codebook_mask);
        tags["down_policy_codebook_mask"] =
            cudaCodebookMaskTag(down_policy_codebook_mask);
        tags["policy_source"] = policy_source ? policy_source : "missing";
        if (use_gateup_kpart || use_ordered_down_kpart)
        {
            tags["splitk_tile_rows"] = std::to_string(splitk_tile_rows);
            tags["splitk_tile_count"] =
                std::to_string((seq_len + splitk_tile_rows - 1) / splitk_tile_rows);
        }
        if (use_gateup_kpart)
        {
            tags["gateup_k_partitions"] = std::to_string(
                llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions);
            tags["gateup_ordered_tile_n"] = std::to_string(tile_n);
        }
        if (use_ordered_down_kpart)
        {
            tags["down_k_partitions"] = std::to_string(
                llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions);
            tags["down_ordered_tile_n"] = std::to_string(
                llaminar2::debugEnv().gemm.cuda_moe_down_ordered_kpart_tile_n);
        }
        if (active_expert_slots > 0)
        {
            llaminar2::PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_grouped_prefill_active_expert_grid_calls",
                1.0, {}, {}, tags);
        }

        tags["swiglu_path"] = use_grouped_imma
                                      ? "split_exact_quantizer"
                                      : ((use_gateup_kpart ||
                                          fuse_swiglu_requested)
                                             ? "fused"
                                             : "split");
        tags["requested_fused_swiglu"] = fuse_swiglu_requested ? "true" : "false";
        if (active_expert_slots > 0)
        {
            if (use_grouped_imma)
                tags["gateup_route"] = "grouped_imma_prefill";
            else if (use_gateup_kpart)
                tags["gateup_route"] = "kpart_prefill";
            else
                tags["gateup_route"] = fuse_swiglu_requested
                                           ? "fullk_fused_swiglu_prefill"
                                           : "fullk_prefill";

            tags["down_route"] = use_grouped_imma
                                         ? "grouped_imma_prefill"
                                         : (use_ordered_down_kpart
                                                ? "ordered_kpart_prefill"
                                                : "grouped_prefill");
            tags["down_accumulation"] = use_grouped_imma
                                                ? "serial_m1_kpart_in_cta"
                                                : (use_ordered_down_kpart
                                                       ? "row_ordered_kpart"
                                                       : (ordered_scatter
                                                              ? "row_ordered"
                                                              : "slot_scatter"));
            if (use_grouped_imma)
            {
                tags["down_publication"] = canonical_route_publication
                                                ? "route_major_collective"
                                                : "ordered_direct";
            }
            if (use_ordered_down_kpart)
            {
                tags["down_publication"] = canonical_route_publication
                                                ? "route_major_collective"
                                                : "fused_direct";
                if (!canonical_route_publication)
                {
                    tags["down_direct_warps"] = std::to_string(
                        llaminar2::CUDAMoEBatchInvariantPolicy::directDownWarps(top_k));
                }
            }
        }
        else if (ordered_scatter)
        {
            tags["gateup_route"] = "serial";
            tags["down_route"] = "serial";
            tags["down_accumulation"] = "row_ordered";
        }
        else
        {
            tags["gateup_route"] = "serial";
            tags["down_route"] = "serial";
            tags["down_accumulation"] = "slot_scatter";
        }
        llaminar2::PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_grouped_prefill_swiglu_path_calls",
            1.0, {}, {}, std::move(tags));
    }

    bool isCudaStreamCapturing(void *stream)
    {
        if (!stream)
            return false;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t err = cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status);
        return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
    }

    /**
     * Return true whenever CUDA work is being recorded for replay.
     *
     * Llaminar has an internal graph-capture guard, but several low-level tests
     * and capture utilities use CUDA stream capture directly. Runtime MoE
     * pointer tables must treat both modes identically: all host-staged pointer
     * arrays have to be uploaded before capture begins so replay never depends
     * on stack-owned host arrays or implicit H2D copies.
     */
    bool isCudaMoEDecodeCaptureActive(void *stream)
    {
        return llaminar2::isGraphCaptureActive() || isCudaStreamCapturing(stream);
    }

    /**
     * @brief Reject host publication or ownership changes while CUDA captures.
     *
     * Persistent MoE metadata must be completely owned and populated before
     * capture begins. Keeping this check in one helper prevents descriptor,
     * mask, and workspace-rebind paths from gradually acquiring inconsistent
     * capture rules.
     *
     * @param stream Explicit stream whose capture state participates in the
     *        decision.
     * @param context Operation name included in the hard-failure diagnostic.
     * @return true when the caller must stop without mutating metadata.
     */
    bool rejectCudaPersistentMetadataMutationDuringCapture(
        void *stream,
        const char *context)
    {
        if (!isCudaMoEDecodeCaptureActive(stream))
            return false;

        LOG_ERROR("[CUDAMoEKernel] Persistent metadata mutation '" << context
                                                                  << "' is forbidden during graph capture");
        return true;
    }

    int *runtimeTopKExpertIdsDevice(llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        auto *base = reinterpret_cast<char *>(runtime_layer);
        return reinterpret_cast<int *>(base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_expert_ids));
    }

    float *runtimeTopKWeightsDevice(llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        auto *base = reinterpret_cast<char *>(runtime_layer);
        return reinterpret_cast<float *>(base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_weights));
    }

    bool cudaGroupedPrefillSupportsCodebook(uint8_t cb)
    {
        switch (cb)
        {
        case 0:
        case 4:
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 19:
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    bool cudaGroupedPrefillRequiresMins(uint8_t cb)
    {
        switch (cb)
        {
        case 11: // Original IQ3_S source nibble; no additive correction.
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        case 5:
        case 7:
        case 8:
        case 9:
        case 10:
        case 13:
        case 14:
        case 16:
        case 17:
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    bool cudaGroupedPrefillRequiresEmins(uint8_t cb)
    {
        return cb == 10;
    }

    constexpr uint8_t kCudaMoEMixedCodebookSentinel = 0xffu;

    uint32_t cudaGroupedPrefillCodebookBit(uint8_t cb)
    {
        return cb < 32 ? (uint32_t{1} << cb) : 0u;
    }

    /**
     * @brief Validate source provenance against the physical execution format.
     *
     * Expanded INT8+minimum bytes are shared by several compact asymmetric
     * codebooks. A descriptor using execution codebook 23 therefore cannot
     * select an exact arithmetic contract unless the producer also publishes
     * its source identity. Legacy immutable descriptors remain accepted for
     * self-describing execution formats, while any supplied provenance is
     * checked against the central NativeQuantised catalog.
     */
    bool validateCudaGroupedSourceIdentity(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc)
    {
        if (!desc.source_identity_present)
        {
            return desc.codebook_id != llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook &&
                   desc.codebook_id != llaminar2::kNativeQuantisedExpandedInt8MinCodebook;
        }
        const auto *source =
            llaminar2::native_quantised_formats::forSourceIdentity(
                desc.source_codebook_id,
                desc.source_is_superblock != 0);
        return source != nullptr &&
               llaminar2::deviceVnniExecutionCompatibleWithSource(
                   *source, desc.codebook_id);
    }

    /** @return Bit selecting the source arithmetic-policy codebook. */
    uint32_t cudaGroupedPrefillPolicyCodebookBit(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc)
    {
        if (!validateCudaGroupedSourceIdentity(desc))
            return 0u;
        const uint8_t source_codebook = desc.source_identity_present
                                            ? desc.source_codebook_id
                                            : desc.codebook_id;
        return cudaGroupedPrefillCodebookBit(
            llaminar2::canonicalDeviceVnniCodebookId(source_codebook));
    }

    bool cudaGroupedPrefillMaskNeedsIQTables(uint32_t mask)
    {
        for (uint8_t cb = 11; cb <= 17; ++cb)
        {
            if (mask & cudaGroupedPrefillCodebookBit(cb))
                return true;
        }
        return false;
    }



    bool validateCudaGroupedDesc(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        int k,
        uint8_t codebook_id)
    {
        return desc.valid() &&
               desc.n == n &&
               desc.k == k &&
               desc.blocks_per_row == static_cast<uint32_t>(k / 32) &&
               desc.codebook_id == codebook_id &&
               (!cudaGroupedPrefillRequiresMins(codebook_id) || desc.mins != nullptr) &&
               (!cudaGroupedPrefillRequiresEmins(codebook_id) || desc.emins != nullptr);
    }

    bool validateCudaGroupedDescShape(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        int k)
    {
        return desc.valid() &&
               desc.n == n &&
               desc.k == k &&
               desc.blocks_per_row == static_cast<uint32_t>(k / 32) &&
               validateCudaGroupedSourceIdentity(desc) &&
               cudaGroupedPrefillSupportsCodebook(desc.codebook_id) &&
               (!cudaGroupedPrefillRequiresMins(desc.codebook_id) || desc.mins != nullptr) &&
               (!cudaGroupedPrefillRequiresEmins(desc.codebook_id) || desc.emins != nullptr);
    }

    bool isBlankGroupedDesc(const llaminar2::DeviceNativeQuantisedMatrixDesc &desc)
    {
        return desc.payload == nullptr &&
               desc.scales == nullptr &&
               desc.mins == nullptr &&
               desc.emins == nullptr &&
               desc.n == 0 &&
               desc.k == 0 &&
               desc.blocks_per_row == 0;
    }

    /** @return Whether a sparse floating descriptor deliberately owns no slot. */
    bool isBlankGroupedFloatingDesc(
        const llaminar2::DeviceMoEFloatingMatrixDesc &desc)
    {
        return desc.data == nullptr && desc.n == 0 && desc.k == 0;
    }

    /** @return Whether one floating descriptor exactly matches its projection. */
    bool validateGroupedFloatingDesc(
        const llaminar2::DeviceMoEFloatingMatrixDesc &desc,
        int n,
        int k)
    {
        return desc.valid() && desc.n == n && desc.k == k;
    }
}

extern "C"
{

    bool cudaMoE_route_logits(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_route_logits_fp16(
        const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts, int device_idx, void *stream,
        const int *live_rows = nullptr);

    bool cudaMoE_route_logits_bf16(
        const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);







    bool cudaMoE_route_logits_decode_equivalent_rows(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_route_logits_bf16_decode_equivalent_rows(
        const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_softmax_topk(
        float *logits, float *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k, bool normalize_weights,
        int device_idx, void *stream,
        const int *device_effective_seq_len,
        void *deferred_selected_route_ledger = nullptr);

    bool cudaMoE_softmax_topk_decode_runtime(
        float *logits, void *runtime_layer,
        float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights,
        bool write_legacy_outputs, bool update_runtime_histogram,
        llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        void *runtime_layers,
        const void *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        void *rebalance_command_header,
        const void *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        const void *rebalance_config,
        void *rebalance_apply_status,
        void *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count,
        const int32_t *absolute_position_ids,
        int device_idx, void *stream,
        const int32_t *active_row_count_device);

    bool cudaMoE_softmax_topk_decode_unique_owner_runtime(
        float *logits,
        void *runtime_layer,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts,
        int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        int device_idx,
        void *stream,
        const int32_t *active_row_count_device);

    bool cudaMoE_decode_route_select_runtime(
        const int *expert_indices, const float *expert_weights,
        void *runtime_layer,
        float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy, int device_idx, void *stream);

    bool cudaMoE_device_rebalance_controller(
        void *runtime_layers,
        const unsigned long long *gathered_histograms,
        void *status,
        const void *config,
        void *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        void *command_header,
        void *wave_state,
        void *controller_state,
        void *llep_layer_plans,
        void *placement_plan_scratch,
        uint32_t command_buffer_count,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_histograms(
        void *runtime_layers,
        unsigned long long *local_histograms,
        const void *config,
        const void *wave_state,
        const void *controller_state,
        uint32_t command_buffer_count,
        uint32_t histogram_source_mask,
        unsigned long long *previous_activation_counts,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_directory(
        void *runtime_layers,
        void *local_directory,
        const void *config,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_source_descriptors(
        void *runtime_layers,
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_source_descriptors,
        const void *config,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_project_rebalance_domain_commands(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_plan_entries,
        void *local_command_headers,
        const void *config,
        void *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        const void *gathered_wave_states,
        void *local_wave_states,
        void *runtime_layers,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        void *transfer_slot_claim_index,
        int device_idx,
        void *stream);

    bool cudaMoE_project_prefill_llep_domain_commands(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_plan_entries,
        uint32_t *local_plan_count,
        void *local_command_header,
        const void *config,
        void *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        void *runtime_layers,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        void *transfer_slot_claim_index,
        int device_idx,
        void *stream);

    bool cudaMoE_materialize_prefill_llep_transfer_commands(
        const void *runtime_layer,
        void *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        void *command_header,
        void *status,
        const void *config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_materialize_prefill_llep_mirrored_domain_commands(
        const void *runtime_layer,
        void *mirrored_plan_entries,
        void *mirrored_command_headers,
        uint32_t plan_capacity,
        void *status,
        const void *config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_compact_payloads(
        const void *plan_entries,
        const void *command_headers,
        uint32_t plan_capacity,
        const void *local_source_descriptors,
        void *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_collective_payloads(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        const void *local_directory,
        void *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_unpack_rebalance_collective_payloads(
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const void *command_header,
        const void *gathered_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_init_rebalance_graph_controller_state(
        void *controller_state,
        const void *config,
        int device_idx,
        void *stream);

    bool cudaMoE_reset_rebalance_graph_transaction_for_request(
        void *controller_state,
        void *command_headers,
        void *wave_states,
        uint32_t *plan_counts,
        uint32_t command_buffer_count,
        const void *config,
        int device_idx,
        void *stream);

    bool cudaMoE_publish_rebalance_transfer_complete(
        void *controller_state,
        const void *command_header,
        const void *wave_state,
        const void *copy_status,
        const void *plan_entries,
        uint32_t plan_capacity,
        const void *gathered_copy_status,
        const void *config,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_apply_ready_rebalance_wave(
        void *runtime_layers,
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        void *command_header,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        int target_layer,
        uint32_t command_buffer_count,
        const void *overlay_reservation_status,
        int device_idx,
        void *stream);

    bool cudaMoE_finalize_overlay_rebalance_publication(
        void *runtime_layers,
        uint32_t layer_count,
        uint32_t expert_count,
        void *control,
        uint64_t *candidate_epoch,
        void *status,
        const void *apply_status,
        void *controller_state,
        void *command_headers,
        uint32_t command_buffer_count,
        const void *plan_entries,
        uint32_t plan_capacity,
        uint64_t estimated_expert_bytes,
        llaminar2::DeviceMoERebalanceMovementJournalView journal,
        int device_idx,
        void *stream);

    bool cudaMoE_apply_rebalance_arrivals(
        void *runtime_layers,
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const void *command_header,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        int target_layer,
        int device_idx,
        void *stream);

    bool cudaMoE_float_to_int(const float *input, int *output, int count, int device_idx, void *stream);
    bool cudaMoE_float_to_masked_int(
        const float *input,
        int *output,
        const uint8_t *expert_mask,
        int count,
        int num_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_gather_tokens(
        const float *hidden, float *batch_buffer, const int *token_indices,
        int num_tokens, int d_model, int device_idx, void *stream);

    bool cudaMoE_copy_token_row(
        const float *source, float *row_buffer,
        int row_index, int row_width, int device_idx, void *stream);

    bool cudaMoE_scatter_add(
        float *output, const float *expert_output, const int *token_indices,
        const float *weights, int num_tokens, int d_model, int device_idx, void *stream);

    bool cudaMoE_write_token_row(
        float *destination, const float *row_buffer,
        int row_index, int row_width, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate_add(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate_add_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream);

    bool cudaMoE_swiglu(float *gate, const float *up, int count, int device_idx, void *stream);
    bool cudaMoE_weighted_add(float *output, const float *input, float weight, int count, int device_idx, void *stream);

    bool cudaMoE_count_per_expert(
        const int *routing_indices, int *expert_counts, int total_slots,
        int num_experts, int device_idx, void *stream);

    bool cudaMoE_exclusive_scan(
        const int *expert_counts, int *expert_offsets,
        int num_experts, int device_idx, void *stream);

    bool cudaMoE_build_active_expert_list(
        const int *expert_counts, int *active_expert_ids,
        int num_experts, int max_active_experts,
        int device_idx, void *stream);

    bool cudaMoE_scatter_tokens(
        const int *routing_indices, const float *routing_weights,
        int *write_heads, const int *expert_offsets,
        int *grouped_token_indices, float *grouped_weights,
        int total_slots, int top_k, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_scatter_tokens_deterministic(
        const int *routing_indices,
        const float *routing_weights,
        const int *expert_offsets,
        const int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int total_slots,
        int top_k,
        int num_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_group_tokens_small_float(
        const float *routing_indices,
        const float *routing_weights,
        int *expert_counts,
        int *expert_offsets,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_prepare_shared_expert_group(
        int *expert_offsets,
        int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        llaminar2::DeviceRowRange rows,
        int device_idx,
        void *stream);

    bool cudaMoE_group_prefill_routes_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int *original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections);

    bool cudaMoE_group_prefill_routes_and_materialize_plan_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int *original_to_grouped,
        llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        llaminar2::DeviceMoEWeightFormat expected_format,
        int device_idx,
        void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        llaminar2::CUDAMoERuntimeGateContract gate_contract);

    bool cudaMoE_regroup_prefill_routes_runtime_assignments(
        void *runtime,
        int *original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream);

    bool cudaMoE_regroup_prefill_routes_and_materialize_plan_runtime(
        void *runtime,
        int *original_to_grouped,
        llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int retain_routes_for_deferred_commit,
        llaminar2::DeviceMoEWeightFormat expected_format,
        int device_idx,
        void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections,
        llaminar2::CUDAMoERuntimeGateContract gate_contract);

    bool cudaMoE_commit_grouped_verifier_histograms(
        void *runtime,
        const int32_t *accepted_state_counts,
        const int32_t *publication_ok_flags,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_commit_grouped_verifier_histograms_batch(
        void *runtime_layers,
        int layer_count,
        size_t layer_stride_bytes,
        const int32_t *accepted_state_counts,
        const int32_t *publication_ok_flags,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_assign_prefill_routes_least_loaded_resident(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        const int32_t *absolute_position_ids,
        const int32_t *active_row_count,
        int device_idx,
        void *stream);

    bool cudaMoE_plan_prefill_routes_least_loaded_current_batch(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        uint32_t min_chunk_tokens,
        uint32_t alpha_numerator,
        uint32_t alpha_denominator,
        uint32_t lambda_numerator,
        uint32_t lambda_denominator,
        uint64_t min_spread_improvement,
        uint32_t min_spread_improvement_divisor,
        uint64_t min_spread_improvement_per_critical_path_slot,
        uint64_t min_foreign_rows_per_critical_path_slot,
        uint32_t max_weight_transfers,
        uint32_t max_non_owner_experts_per_participant,
        int enable_balanced_skip,
        int device_idx,
        void *stream);

    bool cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        const void *transfer_status,
        const void *apply_status,
        int device_idx,
        void *stream);

    bool cudaMoE_prefill_gather_expert_runtime(
        const void *runtime,
        const float *hidden,
        float *batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream);

    bool cudaMoE_prefill_scatter_expert_runtime(
        float *output,
        const float *expert_output,
        const void *runtime,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream);

    bool cudaMoE_gather_expert_fixed(
        const float *hidden, float *batch_buffer,
        const int *expert_offsets, const int *expert_counts,
        const int *grouped_token_indices,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool cudaMoE_scatter_expert_fixed(
        float *output, const float *expert_output,
        const int *expert_offsets, const int *expert_counts,
        const int *grouped_token_indices,
        const float *grouped_weights,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart(
        const float *d_hidden,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream,
        llaminar2::CUDAMoESwiGLUPublication swiglu = {});

    bool cudaMoE_grouped_gate_up_floating_decode_table(
        const float *d_hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        float *d_canonical_route_contributions,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream,
        llaminar2::CUDAMoEDecodeDownInput input = llaminar2::CUDAMoEDecodeDownInput::FloatGateUp);

    bool cudaMoE_grouped_swiglu_down_floating_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        float *d_output,
        float *d_canonical_route_contributions,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const int8_t *d_prequantized_hidden,
        const float *d_prequantized_hidden_scales,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_up_desc_table,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        uint32_t *d_group_work_directory,
        int group_work_directory_entries,
        const int *d_group_token_indices,
        const int *d_original_to_grouped,
        const int *d_original_expert_ids,
        const int *d_active_expert_ids,
        const float *d_group_weights,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        float *d_gate_partials,
        float *d_up_partials,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_down_partials,
        float *d_scratch_down_out,
        float *d_output,
        float *d_canonical_route_contributions,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int top_k,
        int active_expert_slots,
        int grouped_indices_are_route_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int gateup_k_partitions,
        int gateup_ordered_tile_n,
        int down_k_partitions,
        int splitk_tile_rows,
        llaminar2::CUDAMoEBatchInvariantPolicy::GroupedProjectionEngine
            projection_engine,
        llaminar2::cuda::moe::GroupedImmaColumns imma_gateup_columns,
        llaminar2::cuda::moe::GroupedImmaColumns imma_down_columns,
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule
            imma_gateup_schedule,
        int device_idx,
        void *stream,
        const llaminar2::MoEPrefillProjectionExecution &execution,
        llaminar2::CUDAMoERuntimeGateContract gate_contract = {});


    bool cudaMoE_reduce_canonical_route_contributions(
        const float *d_route_contributions,
        float *d_output,
        int seq_len,
        int top_k,
        int d_model,
        int device_idx,
        void *stream);

    bool cudaMoE_publish_shared_expert_rank_bank(
        const float *d_shared_output,
        float *d_canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *d_effective_seq_len,
        int device_idx,
        void *stream);

    bool cudaMoE_finalize_canonical_publication(
        const float *d_input,
        const float *d_gate_inp,
        const float *d_canonical_publication,
        float *d_routed_output,
        float *d_shared_output,
        float *d_combined_output,
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        const int *d_effective_seq_len,
        int device_idx,
        void *stream);

}
