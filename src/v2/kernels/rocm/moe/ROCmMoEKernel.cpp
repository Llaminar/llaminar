/**
 * @file ROCmMoEKernel.cpp
 * @details Model-prepared FP16 router gates select half-RNE hidden operands.
 * Native FP32/BF16 and Q8 paths retain their own policies; no half-only gate
 * cache or workspace-owned floating weight derivative duplicates model storage.
 * @brief ROCm MoE kernel implementation — calls extern "C" HIP bridges
 *
 * Separating .hip and .cpp allows hipcc to compile only the HIP code
 * without encountering issues with MPI or other complex C++ headers.
 * This bridge validates persistent tensor/workspace bindings and their exact
 * stream before delegating to HIP. Compact projection export uses the runtime
 * grouping's device-owned counts, never possibly stale private grouping counts.
 * Import reconstructs existing Q8/scales or FP32 bits before down projection;
 * it neither changes arithmetic nor observes live GPU extents on the host.
 * Grouped down output starts with unspecified bytes. Checked live extents and
 * successful complete writers precede every output publication, including
 * zero contributions for inactive routes; gate/up alone owns no down output.
 * Decode refreshes only its selected expert descriptors from the acquired
 * bank; prefill continues to publish the complete grouped descriptor plan.
 * Complete HC/router/FFN preparation reuses this same source/table owner and
 * seals the original stage-owned HC phase binding with one full-function
 * admission. Named temporary reuse is permitted only after its proved grid
 * retirement; public sources, controller tables and counts remain protected.
 */

#include "kernels/common/MoERuntimeDescriptorLaunch.h"
#include "kernels/common/RouterQ8Launch.h"
#include "kernels/common/OriginalM1HiddenQ8Publication.h"
#include "ROCmMoEKernel.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include "../gemm/ROCmResidentMoEFFN.h"
#include "../gemm/ROCmNativeLayoutRouterFFN.h"
#include "../gemm/ROCmFullNativeHCFFN.h"
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNDevice.h"
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNBinding.h"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include "kernels/common/NativeQuantisedGroupedDecodePolicy.h"
#include "utils/FNV1a.h"
#include "kernels/common/MoEGroupedFloatingPrefillKernels.h"
#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "kernels/common/MoEGroupingInitialization.h"
#include "ROCmMoEOverlayActivationPacketKernels.h"
#include "ROCmMoEOverlayDeviceControllerKernels.h"
#include "ROCmMoEOverlayEpochKernels.h"
#include "../gemm/HipBLASGemmKernel.h"
#include "../gemm/ROCmWeightPacker.h"
#include "../gemm/ROCmMoEGroupedPrefillKernels.h"
#include "../gemm/ROCmMoERouteOwnedTuning.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/rocm/ROCmHardwareMetadata.h"
#include "../../../backends/DeviceId.h"
#include "../../../tensors/ITensor.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/NativeQuantisedFormatInfo.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/ROCmKernelProfiler.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <span>
#include <string_view>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2::rocm
{
/** @copydoc compiledMoERouteOwnedModelPartitionPolicy */
MoERouteOwnedModelPartitionPolicy compiledMoERouteOwnedModelPartitionPolicy() noexcept
{
#if LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES == 1
    return MoERouteOwnedModelPartitionPolicy::FourWorkerPlanes;
#elif LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES == 0
    return MoERouteOwnedModelPartitionPolicy::Serial;
#elif LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES == 2
    return MoERouteOwnedModelPartitionPolicy::Automatic;
#else
#error "invalid source-bound ROCm owner partition policy"
#endif
}
}

namespace
{
    constexpr size_t kGroupedDecodeGateLogitsSharedCapBytes = 48 * 1024;
    constexpr const char *kROCmGroupedDownDescriptorLeaseDomain =
        "rocm_moe_grouped_down_descriptors";
    constexpr const char *kROCmGroupedGateUpDescriptorLeaseDomain =
        "rocm_moe_grouped_gateup_descriptors";
    constexpr const char *kROCmGroupedExpertMaskLeaseDomain =
        "rocm_moe_grouped_expert_masks";
    constexpr const char *kROCmRouterQ8GatePublicationDomain =
        "rocm_moe_router_q8_gate_cache";

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
     * Pointer and format fields are normalized individually so immutable
     * publication equality neither observes C++ padding nor relies on a
     * collision-prone digest.
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
     * @brief Build the exact identity for a down or paired gate/up table.
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

    /** @brief Build an exact captured identity for floating descriptor tables. */
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
     * Rebalance and LLEP primitives use this immutable context rather than the
     * mutable stream inherited from ROCmKernelBase, which may be rebound by a
     * concurrent main, MTP, or maintenance graph capture.
     */
    void *explicitMoELaunchStream(
        const llaminar2::MoEKernelLaunchContext &launch,
        const char *operation)
    {
        if (!launch.hasExplicitStream())
        {
            LOG_ERROR("[ROCmMoEKernel] " << operation
                                         << " requires an explicit stage-owned HIP stream");
            return nullptr;
        }
        return launch.stream;
    }

    bool setMoEDevice(int device_ordinal, const char *context)
    {
        hipError_t err = hipSetDevice(device_ordinal);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] hipSetDevice(" << device_ordinal
                                                      << ") failed in " << context << ": " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool validateDevicePointerOrLog(
        const void *ptr,
        int expected_device,
        const char *pointer_name,
        const char *scope)
    {
#if LLAMINAR_ASSERTIONS_ACTIVE
        if (!ptr)
        {
            LOG_ERROR("[" << scope << "] " << pointer_name << " is null");
            return false;
        }
        hipPointerAttribute_t attr{};
        hipError_t err = hipPointerGetAttributes(&attr, ptr);
        if (err == hipSuccess && attr.device != expected_device)
        {
            LOG_ERROR("[" << scope << "] " << pointer_name
                          << " is on ROCm device " << attr.device
                          << ", expected " << expected_device);
            return false;
        }
        return true;
#else
        (void)ptr;
        (void)expected_device;
        (void)pointer_name;
        (void)scope;
        return true;
#endif
    }

    /// @brief Returns whether grouped MoE kernels instantiate this native-quantised codebook.
    bool groupedDecodeSupportsCodebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:  // Q4_0
        case 4:  // IQ4_NL
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook: // IQ4_XS original tuple
        case 5:  // Q4_1 / Q4_K
        case 6:  // Q5_0
        case 7:  // Q5_1 / Q5_K
        case 8:  // Q6_K
        case 9:  // Q3_K
        case 10: // Q2_K
        case 11: // IQ3_S
        case 12: // IQ3_XXS
        case 13: // IQ2_S
        case 14: // IQ2_XS
        case 15: // IQ2_XXS
        case 16: // IQ1_S
        case 17: // IQ1_M
        case 19: // Q8_0
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    /// @brief Returns whether the descriptor's mins pointer is required for this format.
    bool groupedDecodeRequiresMins(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook: // Raw source coefficient, not a min
        case 11: // Mandatory original IQ3_S source nibble; not a minimum
        case 5:  // Q4_1 / Q4_K min correction
        case 7:  // Q5_1 / Q5_K min correction
        case 8:  // Q6_K packed signed low/high subscales
        case 9:  // Q3_K high half scale
        case 10: // Q2_K high half scale
        case 13: // IQ2_S high half scale
        case 14: // IQ2_XS high half scale
        case 16: // IQ1_S min correction
        case 17: // IQ1_M high half scale
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook: // Expanded asymmetric min correction
            return true;
        default:
            return false;
        }
    }

    /// @brief Returns whether the descriptor's emins pointer is required for this format.
    bool groupedDecodeRequiresEmins(uint8_t codebook_id)
    {
        return codebook_id == 10; // Q2_K stores embedded min correction separately.
    }

    constexpr uint8_t kROCmMoEMixedCodebookSentinel = 0xff;

    uint32_t groupedPrefillCodebookBit(uint8_t codebook_id)
    {
        return groupedDecodeSupportsCodebook(codebook_id)
                   ? (uint32_t{1} << static_cast<uint32_t>(codebook_id))
                   : 0u;
    }

    /**
     * @brief Return whether an execution envelope requires the IQ grid LUTs.
     *
     * The compact IQ3/IQ2/IQ1 formats use device-constant decode tables. The
     * descriptor publication boundary owns making those immutable tables live
     * before any retained graph can embed the descriptors; model loaders are
     * not required to duplicate that execution-kernel invariant.
     */
    bool groupedPrefillMaskNeedsIQTables(uint32_t codebook_mask)
    {
        for (uint8_t codebook = 11u; codebook <= 17u; ++codebook)
        {
            if ((codebook_mask & groupedPrefillCodebookBit(codebook)) != 0u)
                return true;
        }
        return false;
    }

    /**
     * @brief Validate source provenance and its normalized execution format.
     *
     * Expanded INT8+minimum bytes are intentionally shared by several compact
     * asymmetric sources, so codebook 23 is meaningless for exact arithmetic
     * without provenance. Other descriptors retain compatibility with legacy
     * immutable test fixtures, but authoritative provenance is validated when
     * present rather than trusted as an unchecked policy-table index.
     */
    bool validateGroupedSourceIdentity(
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

    /** @return Bit selecting the serial-M1 source arithmetic policy. */
    uint32_t groupedPrefillPolicyCodebookBit(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc)
    {
        if (!validateGroupedSourceIdentity(desc))
            return 0u;
        const uint8_t source_codebook = desc.source_identity_present
                                            ? desc.source_codebook_id
                                            : desc.codebook_id;
        return groupedPrefillCodebookBit(
            llaminar2::canonicalDeviceVnniCodebookId(source_codebook));
    }

    bool validateGroupedDownDesc(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc,
        int d_model,
        int intermediate)
    {
        return desc.valid() && desc.n == d_model && desc.k == intermediate &&
               desc.blocks_per_row == static_cast<uint32_t>(intermediate / 32) &&
               validateGroupedSourceIdentity(desc) &&
               groupedDecodeSupportsCodebook(desc.codebook_id) &&
               (!groupedDecodeRequiresMins(desc.codebook_id) || desc.mins) &&
               (!groupedDecodeRequiresEmins(desc.codebook_id) || desc.emins);
    }

    bool validateGroupedGateUpDesc(
        const llaminar2::DeviceNativeQuantisedMatrixDesc &desc,
        int d_model,
        int intermediate)
    {
        return desc.valid() && desc.n == intermediate && desc.k == d_model &&
               desc.blocks_per_row == static_cast<uint32_t>(d_model / 32) &&
               validateGroupedSourceIdentity(desc) &&
               groupedDecodeSupportsCodebook(desc.codebook_id) &&
               (!groupedDecodeRequiresMins(desc.codebook_id) || desc.mins) &&
               (!groupedDecodeRequiresEmins(desc.codebook_id) || desc.emins);
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

    const int *runtimeTopKExpertIdsDevice(const llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        const auto *base = reinterpret_cast<const char *>(runtime_layer);
        return reinterpret_cast<const int *>(
            base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_expert_ids));
    }

    const float *runtimeTopKWeightsDevice(const llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        const auto *base = reinterpret_cast<const char *>(runtime_layer);
        return reinterpret_cast<const float *>(
            base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_weights));
    }

    bool isHipStreamCapturing(void *stream)
    {
        if (!stream)
            return false;
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        const hipError_t err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &status);
        return err == hipSuccess && status == hipStreamCaptureStatusActive;
    }

    /**
     * @brief Publish an asynchronous HIP tensor write through TransferEngine.
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

    bool requireTensorOnDevice(llaminar2::ITensor *tensor,
                              llaminar2::DeviceId device,
                              void *stream,
                              const char *name,
                              const char *context)
    {
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                         << " requires an explicit HIP stream for " << name);
            return false;
        }

        (void)context;
        llaminar2::TransferEngine::requireDeviceInput(
            tensor,
            device,
            stream);
        return true;
    }

    bool requireOutputOnDevice(llaminar2::ITensor *tensor,
                              llaminar2::DeviceId device,
                              void *stream,
                              const char *name,
                              const char *context)
    {
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                         << " requires an explicit HIP stream for " << name);
            return false;
        }

        (void)context;
        llaminar2::TransferEngine::requireDeviceOutput(
            tensor,
            device,
            stream);
        return true;
    }
}

// Forward-declare extern "C" bridge functions (defined in ROCmMoEKernels.hip)
extern "C"
{
    bool hipMoE_gate_logits_single_token(
        const float *hidden, const float *gate_weights, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_kpart(
        const float *hidden, const float *gate_weights, float *logits,
        float *partials,
        int d_model, int num_experts, int k_partitions,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_kpart_partials(
        const float *hidden, const float *gate_weights, float *partials,
        int d_model, int num_experts, int k_partitions,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_grouped(
        const float *hidden, const float *gate_weights, float *logits,
        int d_model, int num_experts, size_t shared_mem_bytes,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_grouped_decode_router_rows(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts, size_t shared_mem_bytes,
        int device_idx, void *stream,
        const int *device_effective_seq_len);

    bool hipMoE_gate_logits_single_token_fp16_weights(
        const float *hidden, const void *gate_weights_fp16, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_bf16_weights(
        const float *hidden, const void *gate_weights_bf16, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_fp32_decode_equivalent_rows(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream,
        const int *device_effective_seq_len = nullptr);

    bool hipMoE_gate_logits_fp16_decode_equivalent_rows(
        const float *hidden, const void *gate_weights_fp16, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream,
        const int *device_effective_seq_len = nullptr);

    bool hipMoE_gate_logits_bf16_decode_equivalent_rows(
        const float *hidden, const void *gate_weights_bf16, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream,
        const int *device_effective_seq_len = nullptr);







    bool hipMoE_softmax_topk_decode_runtime(
        float *logits,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
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

    bool hipMoE_softmax_topk_decode_runtime_wave64(
        float *logits,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
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

    bool hipMoE_softmax_topk_decode_unique_owner_wave64(
        float *logits,
        void *runtime,
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

    bool hipMoE_softmax_topk_decode_equivalent_rows(
        float *logits,
        float *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        int device_idx, void *stream,
        const int *device_effective_seq_len = nullptr,
        void *deferred_selected_route_ledger = nullptr);

    bool hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
        const float *partials,
        float *router_probabilities,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int k_partitions, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
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

    bool hipMoE_decode_route_select_runtime(
        const int *expert_indices,
        const float *expert_weights,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        int device_idx, void *stream);

    bool hipMoE_device_rebalance_controller(
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

    bool hipMoE_pack_rebalance_histograms(
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

    bool hipMoE_pack_rebalance_directory(
        void *runtime_layers,
        void *local_directory,
        const void *config,
        int device_idx,
        void *stream);

    bool hipMoE_pack_rebalance_source_descriptors(
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

    bool hipMoE_project_rebalance_domain_commands(
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

    bool hipMoE_project_prefill_llep_domain_commands(
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

    bool hipMoE_materialize_prefill_llep_transfer_commands(
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

    bool hipMoE_materialize_prefill_llep_mirrored_domain_commands(
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

    bool hipMoE_pack_rebalance_compact_payloads(
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

    bool hipMoE_pack_rebalance_collective_payloads(
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

    bool hipMoE_unpack_rebalance_collective_payloads(
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

    bool hipMoE_init_rebalance_graph_controller_state(
        void *controller_state,
        const void *config,
        int device_idx,
        void *stream);

    bool hipMoE_reset_rebalance_graph_transaction_for_request(
        void *controller_state,
        void *command_headers,
        void *wave_states,
        uint32_t *plan_counts,
        uint32_t command_buffer_count,
        const void *config,
        int device_idx,
        void *stream);

    bool hipMoE_publish_rebalance_transfer_complete(
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

    bool hipMoE_apply_ready_rebalance_wave(
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

    bool hipMoE_finalize_overlay_rebalance_publication(
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

    bool hipMoE_apply_rebalance_arrivals(
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

    bool hipMoE_gather_tokens(
        const float *hidden, float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_copy_token_row(
        const float *source, float *row_buffer,
        int row_index, int row_width,
        int device_idx, void *stream);

    bool hipMoE_scatter_add(
        float *output, const float *expert_output,
        const int *token_indices, const float *weights,
        int num_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_write_token_row(
        float *destination, const float *row_buffer,
        int row_index, int row_width,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate(
        const float *input, const float *gate_inp,
        float *shared_output, float *gate_scratch,
        int seq_len, int d_model,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate_effective_seq_len(
        const float *input, const float *gate_inp,
        float *shared_output, float *gate_scratch,
        int seq_len, int d_model,
        const int *device_effective_seq_len,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate_decode_fused(
        const float *input, const float *gate_inp,
        float *shared_output, int d_model,
        int device_idx, void *stream, const int *device_effective_seq_len = nullptr);

    bool hipMoE_shared_expert_gate_add(
        const float *input, const float *gate_inp,
        float *shared_output, const float *routed_residual,
        float *combined_output, int seq_len, int d_model,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate_add_effective_seq_len(
        const float *input, const float *gate_inp,
        float *shared_output, const float *routed_residual,
        float *combined_output, int seq_len, int d_model,
        const int *device_effective_seq_len,
        int device_idx, void *stream);

    bool hipMoE_swiglu(
        float *gate, const float *up,
        int count,
        int device_idx, void *stream);

    bool hipMoE_weighted_add(
        float *output, const float *input,
        float weight, int count,
        int device_idx, void *stream);

    // Phase 3: Token grouping bridges
    bool hipMoE_count_per_expert(
        const int *routing_indices, int *expert_counts,
        int total_slots, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_exclusive_scan(
        int *expert_counts, int *expert_offsets,
        int num_experts,
        int device_idx, void *stream);

    bool hipMoE_build_active_expert_list(
        const int *expert_counts, int *active_expert_ids,
        int num_experts, int max_active_experts,
        int device_idx, void *stream);

    bool hipMoE_max_expert_count(
        const int *expert_counts, int *d_max_out,
        int num_experts,
        int device_idx, void *stream);

    bool hipMoE_scatter_tokens(
        const int *routing_indices, const float *routing_weights,
        const int *expert_offsets, int *write_heads,
        int *grouped_token_indices, int *original_to_grouped,
        float *grouped_weights,
        int total_slots, int num_experts, int top_k,
        int device_idx, void *stream);

    bool hipMoE_group_tokens_small_float(
        const float *routing_indices, const float *routing_weights,
        int *expert_counts, int *expert_offsets,
        int *grouped_token_indices, int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int total_slots, int num_experts, int top_k,
        int max_active_experts,
        int device_idx, void *stream);

    bool hipMoE_prepare_shared_expert_group(
        int *expert_offsets,
        int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        float *grouped_weights,
        int *active_expert_ids,
        llaminar2::DeviceRowRange rows,
        int device_idx,
        void *stream);

    // Phase 4: Tensor-aware utility bridges
    bool hipMoE_float_to_int(
        const float *d_input, int *d_output, int count,
        int device_idx, void *stream);

    bool hipMoE_float_to_masked_int(
        const float *d_input,
        int *d_output,
        const uint8_t *d_expert_mask,
        int count,
        int num_experts,
        int device_idx,
        void *stream);

    bool hipMoE_stage_mutable_pointer_arrays(
        float **d_a,
        float **d_b,
        float *const *h_a,
        float *const *h_b,
        int count,
        int device_idx,
        void *stream);

    bool hipMoE_stage_const_pointer_arrays(
        const float **d_a,
        const float **d_b,
        const float *const *h_a,
        const float *const *h_b,
        int count,
        int device_idx,
        void *stream);

    bool hipMoE_group_prefill_routes_runtime(
        const float *routing_indices, const float *routing_weights,
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        int device_idx, void *stream,
        llaminar2::DeviceMoEProjectionSet expected_projections);

    bool hipMoE_regroup_prefill_routes_runtime_assignments(
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
        int retain_routes_for_deferred_commit,
        int device_idx, void *stream);

    bool hipMoE_group_prefill_routes_and_materialize_plan_runtime(
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
        llaminar2::DeviceMoEProjectionSet expected_projections);

    bool hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
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
        llaminar2::DeviceMoEProjectionSet expected_projections);

    bool hipMoE_commit_grouped_verifier_histograms(
        void *runtime,
        const int32_t *accepted_state_counts,
        const int32_t *publication_ok_flags,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k,
        int device_idx, void *stream);

    bool hipMoE_assign_prefill_routes_least_loaded_resident(
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
        const int32_t *absolute_position_ids,
        const int32_t *active_row_count,
        int device_idx, void *stream);

    bool hipMoE_plan_prefill_routes_least_loaded_current_batch(
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
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
        int device_idx, void *stream);

    bool hipMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
        int device_idx, void *stream);

    bool hipMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
        const void *transfer_status,
        const void *apply_status,
        int device_idx, void *stream);

    bool hipMoE_prefill_gather_expert_runtime(
        const void *runtime,
        const float *hidden,
        float *batch_buffer,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_prefill_scatter_expert_runtime(
        float *output,
        const float *expert_output,
        const void *runtime,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool rocmMoE_grouped_swiglu_down_native_quantised_decode(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_descs,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_ordered_down_partials,
        float *d_output,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        bool swiglu_prequantized,
        float *d_output,
        float *d_canonical_route_contributions,
        float *d_ordered_route_scratch,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        uint32_t policy_codebook_mask,
        int device_idx,
        void *stream,
        const llaminar2::NativeDownCompletionView* completion=nullptr);

    bool rocmMoE_grouped_swiglu_down_floating_decode_table(
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

    bool rocmMoE_grouped_gate_up_native_quantised_decode_table(
        const float *d_hidden,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeQuantisedMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_fused_swiglu_int8,
        float *d_fused_swiglu_scales,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        float *d_gate_partials,
        float *d_up_partials,
        bool hidden_prequantized,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        uint32_t policy_codebook_mask,
        int device_idx,
        void *stream,
        const llaminar2::NativeTileCompletionView* completion=nullptr);

    bool rocmMoE_grouped_gate_up_floating_decode_table(
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

    bool rocmMoE_grouped_gate_up_native_quantised_decode_runtime(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        float *d_gate_partials,
        float *d_up_partials,
        bool hidden_prequantized,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        uint32_t policy_codebook_mask,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        float *d_ordered_route_scratch,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        uint32_t policy_codebook_mask,
        int device_idx,
        void *stream);



    bool rocmMoE_publish_shared_expert_rank_bank(
        const float *d_shared_output,
        float *d_canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *d_effective_seq_len,
        int device_id,
        void *stream);

    bool rocmMoE_finalize_canonical_publication(
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
        int device_id,
        void *stream);

    bool rocmMoE_grouped_prefill_query_tile_config(
        uint8_t codebook_id,
        int projection_role,
        int m,
        int n,
        int k,
        int *tile_m,
        int *tile_n);

    bool rocmMoE_grouped_prefill_query_production_pair_config(
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
        int *exact_overlay);

}

namespace
{
    /**
     * @brief Human-readable generated-policy values attached to PerfStats.
     *
     * Descriptor tables may contain one execution codebook or a heterogeneous
     * mixture. The telemetry deliberately uses strings so mixed tables and an
     * impossible policy miss remain visible instead of being misreported as a
     * numeric legacy default.
     */
    struct MoEPrefillPolicyTags
    {
        std::string tile_m;
        std::string tile_n;
    };

    /** Joint production policy and provenance recorded for one captured pair. */
    struct MoEPrefillPairPolicyTags
    {
        MoEPrefillPolicyTags gateup;
        MoEPrefillPolicyTags down;
        std::string source;
    };

    /**
     * @brief Recover one codebook id from a descriptor-table bit mask.
     *
     * @param codebook_mask Bit @c n is set when codebook @c n occurs in the
     *                      uploaded descriptor table.
     * @return The unique decoder family, or -1 for unrelated mixed formats.
     * @details Physical4/24 share LUT integer decode. Family24 resolves each
     * descriptor scale independently; original physical masks remain retained.
     */
    int singleCodebookFromMask(uint32_t codebook_mask)
    {
        const uint32_t iq4_tuple = llaminar2::nativeQuantisedCodebookMaskBit(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook);
        if (codebook_mask & iq4_tuple)
            codebook_mask &= ~llaminar2::nativeQuantisedCodebookMaskBit(uint8_t{4});
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
     * @brief Format a descriptor-table codebook mask for stable PerfStats tags.
     *
     * Fixed-width hexadecimal preserves every bit and makes uniform and mixed
     * tables visually distinguishable without depending on locale or signed
     * integer formatting. The resulting string is diagnostic metadata only;
     * launch policy continues to consume the original integer mask.
     *
     * @param codebook_mask Exact OR-reduction of all descriptor codebook ids.
     * @return Lowercase fixed-width hexadecimal in @c 0x00000000 form.
     */
    std::string codebookMaskTag(uint32_t codebook_mask)
    {
        char text[11]{};
        std::snprintf(text, sizeof(text), "0x%08x", codebook_mask);
        return text;
    }

    /**
     * @brief Describe the joint policy embedded in one grouped-prefill graph.
     *
     * This query has the complete exact-overlay key and therefore cannot report
     * a generic role geometry while production executes a model-specific pair.
     * Mixed descriptor tables continue to launch one generated generic policy
     * per execution codebook and are explicitly reported as mixed. A missing
     * uniform policy is an internal invariant violation: the launch must already
     * have failed rather than publishing misleading telemetry.
     */
    MoEPrefillPairPolicyTags queryMoEPrefillPairPolicyTags(
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int seq_len,
        int hidden_size,
        int expert_width,
        int expert_count,
        int top_k,
        llaminar2::MoEPrefillMeasuredOperation operation)
    {
        if (seq_len <= 8)
            return {{"1", "64"}, {"1", "64"}, "direct"};

        const int gateup_codebook =
            singleCodebookFromMask(gateup_codebook_mask);
        const int down_codebook = singleCodebookFromMask(down_codebook_mask);
        if (gateup_codebook < 0 || down_codebook < 0)
        {
            return {
                {"mixed", "mixed"},
                {"mixed", "mixed"},
                "generic_mixed_codebooks"};
        }

        int gateup_tile_m = 0;
        int gateup_tile_n = 0;
        int down_tile_m = 0;
        int down_tile_n = 0;
        int exact_overlay = 0;
        if (!rocmMoE_grouped_prefill_query_production_pair_config_for_operation(
                operation,
                static_cast<uint8_t>(gateup_codebook),
                static_cast<uint8_t>(down_codebook),
                seq_len,
                hidden_size,
                expert_width,
                expert_count,
                top_k,
                &gateup_tile_m,
                &gateup_tile_n,
                &down_tile_m,
                &down_tile_n,
                &exact_overlay))
        {
            return {
                {"missing", "missing"},
                {"missing", "missing"},
                "missing"};
        }
        return {
            {std::to_string(gateup_tile_m), std::to_string(gateup_tile_n)},
            {std::to_string(down_tile_m), std::to_string(down_tile_n)},
            exact_overlay != 0 ? "exact_overlay" : "generic"};
    }

    bool tryGroupedDecodeGateLogits(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream)
    {
        if (!llaminar2::debugEnv().rocm.moe_grouped_decode_router)
            return false;

        const size_t shared_mem_bytes = static_cast<size_t>(d_model) * sizeof(float);
        if (num_experts <= 1 || shared_mem_bytes > kGroupedDecodeGateLogitsSharedCapBytes)
            return false;

        return hipMoE_gate_logits_single_token_grouped(
            hidden, gate_weights, logits,
            d_model, num_experts, shared_mem_bytes,
            device_ordinal, stream);
    }

    /**
     * @brief Launch FP32 grouped rows with the exact router geometry selected by decode.
     *
     * Decode can use either the shared-hidden two-experts-per-block kernel or
     * the one-expert-per-block kernel. This policy function is the single owner
     * of that choice for every M, preventing routeWithTensors(), verifier
     * routing, and runtime decode from silently adopting different reduction
     * trees. A selected launch failure is returned directly; it never retries
     * with the other arithmetic path.
     */
    bool launchDecodeEquivalentFP32Rows(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int seq_len,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream,
        const int *device_effective_seq_len)
    {
        const size_t shared_mem_bytes =
            static_cast<size_t>(d_model) * sizeof(float);
        const bool decode_uses_shared_hidden_router =
            llaminar2::debugEnv().rocm.moe_grouped_decode_router &&
            num_experts > 1 &&
            shared_mem_bytes <= kGroupedDecodeGateLogitsSharedCapBytes;
        if (decode_uses_shared_hidden_router)
        {
            return hipMoE_gate_logits_grouped_decode_router_rows(
                hidden,
                gate_weights,
                logits,
                seq_len,
                d_model,
                num_experts,
                shared_mem_bytes,
                device_ordinal,
                stream,
                device_effective_seq_len);
        }

        return hipMoE_gate_logits_fp32_decode_equivalent_rows(
            hidden,
            gate_weights,
            logits,
            seq_len,
            d_model,
            num_experts,
            device_ordinal,
            stream,
            device_effective_seq_len);
    }

    bool launchDecodeGateLogits(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream,
        const char *context)
    {
        if (tryGroupedDecodeGateLogits(hidden, gate_weights, logits,
                                       d_model, num_experts,
                                       device_ordinal, stream))
        {
            return true;
        }

        if (!hipMoE_gate_logits_single_token(hidden, gate_weights, logits,
                                             d_model, num_experts,
                                             device_ordinal, stream))
        {
            LOG_ERROR("[" << context << "] single-token gate logits kernel failed");
            return false;
        }
        return true;
    }

    bool launchDecodeGateLogitsForGateType(
        const float *hidden,
        const void *gate_weights,
        llaminar2::TensorType gate_type,
        float *logits,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream,
        const char *context)
    {
        switch (gate_type)
        {
        case llaminar2::TensorType::FP32:
            return launchDecodeGateLogits(
                hidden, static_cast<const float *>(gate_weights), logits,
                d_model, num_experts, device_ordinal, stream, context);
        case llaminar2::TensorType::FP16:
            if (!hipMoE_gate_logits_single_token_fp16_weights(
                    hidden, gate_weights, logits,
                    d_model, num_experts, device_ordinal, stream))
            {
                LOG_ERROR("[" << context << "] FP16 gate logits kernel failed");
                return false;
            }
            return true;
        case llaminar2::TensorType::BF16:
            if (!hipMoE_gate_logits_single_token_bf16_weights(
                    hidden, gate_weights, logits,
                    d_model, num_experts, device_ordinal, stream))
            {
                LOG_ERROR("[" << context << "] BF16 gate logits kernel failed");
                return false;
            }
            return true;
        default:
            LOG_ERROR("[" << context << "] unsupported router gate dtype "
                          << llaminar2::tensorTypeName(gate_type));
            return false;
        }
    }

}

namespace llaminar2
{

    bool ROCmMoEKernel::beginOriginalMoEOverlayInvocation(
        const MoEKernelLaunchContext &context,
        const MoEOverlayOriginalNativeBeginLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "beginOriginalMoEOverlayInvocation");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeBegin(&packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireOriginalMoEOverlayInvocation(
        const MoEKernelLaunchContext &context,
        const MoEOverlayOriginalNativeRetireLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocation");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeRetire(&packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireOriginalMoEOverlayInvocationBatch(
        const MoEKernelLaunchContext &context,
        const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocationBatch");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeRetireSingleRowBatch(&packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireOriginalMoEOverlayInvocationBatch(
        const MoEKernelLaunchContext &context,
        const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocationBatch");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeRetireMultiRowBatch(&packet, device_ordinal_, stream);
    }


    bool ROCmMoEKernel::packMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationDispatchPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packMoEOverlayActivationDispatch] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationDispatchConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMoEOverlayActivationDispatch] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationReturnPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packMoEOverlayActivationReturn] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationReturnConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMoEOverlayActivationReturn] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMoEOverlayCanonicalRouteTicket(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayCanonicalRouteTicketConsumeLaunch &ticket)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayCanonicalRouteTicket");
        if (!stream || !ticket.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMoEOverlayCanonicalRouteTicket] complete mapped ticket binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayConsumeCanonicalRouteTicket(
            &ticket, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatch] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackSingleRowDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowMoEOverlayActivationDispatch] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeSingleRowDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packSingleRowMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packSingleRowMoEOverlayActivationReturn] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackSingleRowReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturn] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeSingleRowReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationDispatchBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch] persistent topology batch and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackSingleRowDispatchBatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationReturnBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch] persistent topology batch, gather scratch, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeSingleRowReturnBatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMultiRowMoEOverlayActivationReturnBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch] persistent topology batch, row-index scratch, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeMultiRowReturnBatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::publishNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRoutePublishLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishNodeLocalCanonicalRoutes");
        if (!stream || !publication.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::publishNodeLocalCanonicalRoutes] complete mapped lane, route assignment, canonical contribution, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayPublishNodeLocalCanonicalRoutes(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::acquireNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "acquireNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::acquireNodeLocalCanonicalRoutes] persistent peer bindings, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayAcquireNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::stageNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "stageNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::stageNodeLocalCanonicalRoutes] mapped sources, root scratch, route assignment, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayStageNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::foldNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "foldNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::foldNodeLocalCanonicalRoutes] persistent peer bindings, staged payload, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayFoldNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::beginNodeLocalDensePublication(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginNodeLocalDensePublication");
        if (!stream || !publication.valid() ||
            !publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::beginNodeLocalDensePublication] root binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayBeginNodeLocalDensePublication(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::finishNodeLocalDensePublication(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishNodeLocalDensePublication");
        if (!stream || !publication.valid() ||
            !publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::finishNodeLocalDensePublication] root binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayFinishNodeLocalDensePublication(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::beginNodeLocalDensePublicationConsume(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginNodeLocalDensePublicationConsume");
        if (!stream || !publication.valid() || publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::beginNodeLocalDensePublicationConsume] peer binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayBeginNodeLocalDensePublicationConsume(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::finishNodeLocalDensePublicationConsume(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishNodeLocalDensePublicationConsume");
        if (!stream || !publication.valid() || publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::finishNodeLocalDensePublicationConsume] peer binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayFinishNodeLocalDensePublicationConsume(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::runMoEOverlayDeviceControllerAction(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayDeviceControllerActionLaunch &action)
    {
        void *stream = explicitMoELaunchStream(
            launch, "runMoEOverlayDeviceControllerAction");
        if (!stream || !action.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::runMoEOverlayDeviceControllerAction] a valid mapped controller action and explicit HIP stream are required");
            return false;
        }
        return hipMoEOverlayRunDeviceControllerAction(
            &action, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::beginMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayServiceTelemetrySample *sample)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginMoEOverlayServiceTelemetry");
        if (!stream || !sample)
            return false;
        return hipMoEOverlayBeginServiceTelemetry(
            sample, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::finishMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
        DeviceMoEOverlayServiceTelemetrySample *sample,
        std::uint32_t num_experts,
        MoEOverlayServicePhaseHint hint,
        const MoEOverlayInferenceGraphRole *runtime_graph_role)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishMoEOverlayServiceTelemetry");
        if (!stream || !runtime_layer || !layer_telemetry || !sample ||
            num_experts == 0u)
        {
            return false;
        }
        return hipMoEOverlayFinishServiceTelemetry(
            runtime_layer,
            layer_telemetry,
            sample,
            num_experts,
            static_cast<std::uint32_t>(hint),
            runtime_graph_role,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::publishMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        const DeviceMoEOverlayServiceTelemetryCell *telemetry,
        const DeviceMoEOverlayServiceTelemetrySample *samples,
        std::uint32_t layer_count,
        std::int32_t participant_id,
        MoEOverlayDeviceServiceTelemetryPublicationHeader *publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishMoEOverlayServiceTelemetry");
        if (!stream || !telemetry || !samples || !publication ||
            layer_count == 0u ||
            participant_id < 0)
        {
            return false;
        }
        return hipMoEOverlayPublishServiceTelemetry(
            telemetry,
            samples,
            layer_count,
            participant_id,
            publication,
            device_ordinal_,
            stream);
    }

    ROCmMoEKernel::ROCmMoEKernel(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_ordinal);
        ROCmKernelBase::setDeviceContext(&ctx);

        // Create hipBLAS GEMM kernel using device context (shares hipBLAS handle)
        blas_gemm_ = std::make_unique<rocm::HipBLASGemmKernel>(&ctx);

        /*
         * Construction establishes device resources only. The graph executor
         * has not assigned a producer stream yet, so binding hipBLAS here would
         * either consult an implicit context default or make construction
         * order-dependent. bindGPUStream() performs the parent/child stream
         * transition atomically once the executor supplies its exact stream.
         */
        LOG_TRACE("[ROCmMoEKernel] Created unbound for ROCm device "
                  << device_ordinal);
    }

    void ROCmMoEKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        const uint64_t new_workspace_id = workspace ? workspace->id() : 0;
        if (workspace_ == workspace && bound_workspace_id_ == new_workspace_id)
        {
            const bool needs_descriptor_rebind =
                workspace_ &&
                (std::any_of(grouped_down_desc_tables_.begin(), grouped_down_desc_tables_.end(),
                             [](const GroupedDownDescriptorTable &table)
                             {
                                 return table.valid && !table.host_descs.empty() && !table.device_descs;
                             }) ||
                 std::any_of(grouped_gateup_desc_tables_.begin(), grouped_gateup_desc_tables_.end(),
                             [](const GroupedGateUpDescriptorTable &table)
                             {
                                 return table.valid &&
                                        !table.host_gate_descs.empty() &&
                                        !table.host_up_descs.empty() &&
                                        (!table.device_gate_descs || !table.device_up_descs);
                             }));
            if (needs_descriptor_rebind)
                (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
            return;
        }
        ROCmKernelBase::bindWorkspace(workspace);
        bound_workspace_id_ = new_workspace_id;
        clearWorkspaceScratchBindings();
        if (workspace_)
            (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
    }

    bool ROCmMoEKernel::bindWorkspaceBuffer(
        void **ptr,
        const char *name,
        size_t bytes,
        const char *context)
    {
        if (!ptr || !name || bytes == 0)
            return false;
        if (!validateROCmWorkspaceBinding(workspace_, device_ordinal_, "ROCmMoEKernel"))
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                         << " requires graph-owned MoE workspace");
            return false;
        }
        void *buffer = workspace_->getBuffer(name);
        const size_t available = workspace_->getBufferSize(name);
        if (!buffer || available < bytes)
        {
            LOG_ERROR("[ROCmMoEKernel] " << context << " missing required workspace buffer '"
                                         << name << "' (need " << bytes << " bytes, have "
                                         << available << ")");
            return false;
        }
        *ptr = buffer;
        scratch_workspace_bound_ = true;
        return true;
    }

    bool ROCmMoEKernel::bindGroupedDescriptorTableSlot(
        const char *buffer_name,
        std::size_t slot,
        int num_experts,
        DeviceNativeQuantisedMatrixDesc **device_descs,
        const char *context)
    {
        if (!buffer_name || !device_descs || num_experts <= 0 ||
            slot >= static_cast<std::size_t>(
                        MoEWorkspaceBuffers::kGroupedDescriptorTableSlots))
        {
            return false;
        }

        void *base = nullptr;
        if (!bindWorkspaceBuffer(
                &base,
                buffer_name,
                sizeof(DeviceNativeQuantisedMatrixDesc),
                context))
        {
            return false;
        }

        const std::size_t available = workspace_->getBufferSize(buffer_name);
        const std::size_t slot_bytes =
            available /
            static_cast<std::size_t>(
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
        const std::size_t descriptor_stride =
            slot_bytes / sizeof(DeviceNativeQuantisedMatrixDesc);
        if (descriptor_stride < static_cast<std::size_t>(num_experts))
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                           << " descriptor slot is too narrow in '"
                                           << buffer_name << "': slot=" << slot
                                           << " stride=" << descriptor_stride
                                           << " requested_experts=" << num_experts
                                           << " workspace_bytes=" << available);
            return false;
        }

        const std::size_t descriptor_offset = slot * descriptor_stride;
        const std::size_t descriptor_capacity =
            available / sizeof(DeviceNativeQuantisedMatrixDesc);
        if (descriptor_offset + static_cast<std::size_t>(num_experts) >
            descriptor_capacity)
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                           << " descriptor slot exceeds workspace capacity in '"
                                           << buffer_name << "': offset=" << descriptor_offset
                                           << " requested_experts=" << num_experts
                                           << " capacity=" << descriptor_capacity);
            return false;
        }

        *device_descs =
            static_cast<DeviceNativeQuantisedMatrixDesc *>(base) + descriptor_offset;
        return true;
    }

    void ROCmMoEKernel::clearWorkspaceScratchBindings() noexcept
    {
        if(original_m1_hidden_q8_read_)
            original_m1_hidden_q8_read_lifecycle_=OriginalM1HiddenQ8ReadLifecycle::Invalidated;
        original_m1_hidden_q8_read_.reset();
        gate_up_last_producer_.clear();
        down_last_producer_.clear();
        paired_router_descriptors_.reset();
        grouped_prefill_grouping_.invalidate();
        d_write_heads_ = nullptr;
        d_staging_indices_ = nullptr;
        d_staging_weights_ = nullptr;
        d_grouped_gate_ptrs_ = nullptr;
        d_grouped_up_ptrs_ = nullptr;
        d_grouped_expert_ids_ = nullptr;
        d_grouped_decode_weights_ = nullptr;
        d_grouped_down_descs_ = nullptr;
        d_grouped_swiglu_int8_ = nullptr;
        d_grouped_swiglu_scales_ = nullptr;
        d_grouped_down_partials_ = nullptr;
        d_grouped_gate_output_ptrs_ = nullptr;
        d_grouped_up_output_ptrs_ = nullptr;
        d_grouped_gateup_expert_ids_ = nullptr;
        d_grouped_hidden_int8_ = nullptr;
        d_grouped_hidden_scales_ = nullptr;
        d_grouped_gateup_gate_partials_ = nullptr;
        d_grouped_gateup_up_partials_ = nullptr;
        d_shared_gate_scratch_ = nullptr;
        d_route_logits_ = nullptr;
        d_route_logits_partials_ = nullptr;
        d_router_q8_hidden_ = nullptr;
        d_router_q8_hidden_scales_ = nullptr;
        invalidateRouterQ8HiddenPublication();
        d_group_int_indices_ = nullptr;
        d_group_offsets_ = nullptr;
        d_group_counts_ = nullptr;
        d_group_max_tokens_ = nullptr;
        d_group_expert_mask_ = nullptr;
        group_expert_mask_workspace_lease_.reset();
        d_group_token_indices_ = nullptr;
        d_group_original_to_grouped_ = nullptr;
        d_group_weights_ = nullptr;
        d_group_active_expert_ids_ = nullptr;
        d_prefill_A_int8_ = nullptr;
        d_prefill_A_scales_ = nullptr;
        d_prefill_swiglu_int8_ = nullptr;
        d_prefill_swiglu_scales_ = nullptr;
        d_prefill_gate_ = nullptr;
        d_prefill_up_ = nullptr;
        d_prefill_work_directory_ = nullptr;

        max_write_heads_experts_ = 0;
        staging_capacity_ = 0;
        grouped_decode_active_cap_ = 0;
        grouped_decode_intermediate_cap_ = 0;
        grouped_decode_d_model_cap_ = 0;
        grouped_gateup_active_cap_ = 0;
        grouped_gateup_d_model_cap_ = 0;
        grouped_gateup_intermediate_cap_ = 0;
        shared_gate_scratch_capacity_ = 0;
        route_logits_capacity_ = 0;
        route_logits_partials_capacity_ = 0;
        router_q8_hidden_rows_cap_ = 0;
        router_q8_hidden_d_model_cap_ = 0;
        router_q8_hidden_blocks_cap_ = 0;
        invalidateRouterQ8HiddenPublication();
        group_active_expert_slots_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_expert_mask_cap_ = 0;
        group_expert_mask_hash_ = 0;
        group_expert_mask_num_experts_ = 0;
        group_expert_mask_active_experts_ = 0;
        group_expert_mask_published_ = false;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        router_q8_gate_cache_.clear();
        for (auto &table : grouped_down_desc_tables_)
        {
            table.device_descs = nullptr;
            table.device_floating_descs = nullptr;
            table.workspace_publication.reset();
            table.workspace_slot = 0;
        }
        for (auto &table : grouped_gateup_desc_tables_)
        {
            table.device_gate_descs = nullptr;
            table.device_up_descs = nullptr;
            table.device_floating_gate_descs = nullptr;
            table.device_floating_up_descs = nullptr;
            table.workspace_publication.reset();
            table.workspace_slot = 0;
        }
        fixed_gateup_metadata_states_.clear();
        fixed_down_metadata_states_.clear();
        runtime_pointer_workspace_owners_.reset();
        gateup_pointer_slot_ready_.fill(false);
        down_pointer_slot_ready_.fill(false);
        scratch_workspace_bound_ = false;
    }

    bool ROCmMoEKernel::publishGroupedDownDescriptorTable(
        GroupedDownDescriptorTable &table,
        const char *context)
    {
        const bool floating = deviceMoEWeightFormatIsFloating(
            table.weight_format);
        if (!workspace_ || !table.valid ||
            (floating ? table.host_floating_descs.empty()
                      : table.host_descs.empty()) ||
            table.num_experts <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "down descriptor publication")
                      << " requires a bound workspace and complete host table");
            return false;
        }
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "down descriptor publication")
                      << " requires an explicit HIP stream");
            return false;
        }

        // Source2 consumes live descriptors from its runtime bank. An empty
        // original template needs local initialization only: allocation E is
        // not an H2D communication extent. Keep the same slot and ready event.
        const bool empty_original_image = table.original_family.valid() &&
            (floating ? (std::all_of(table.host_floating_descs.begin(),table.host_floating_descs.end(),isBlankGroupedFloatingDesc))
                      : (std::all_of(table.host_descs.begin(),table.host_descs.end(),isBlankGroupedDesc)));

        if (floating)
        {
            const size_t desc_bytes =
                static_cast<size_t>(table.num_experts) *
                sizeof(DeviceMoEFloatingMatrixDesc);
            const auto publication_result =
                workspace_->getOrCreatePersistentPublication(
                    kROCmGroupedDownDescriptorLeaseDomain,
                    groupedFloatingDescriptorPublicationKey(
                        table.host_floating_descs.data(),
                        nullptr,
                        table.weight_format,
                        table.num_experts,
                        table.d_model,
                        table.intermediate, table.descriptor_source),
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                    [&](std::size_t slot) -> std::shared_ptr<void>
                    {
                        DeviceNativeQuantisedMatrixDesc *slot_base = nullptr;
                        if (!bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::
                                    ROCM_GROUPED_DOWN_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &slot_base,
                                "ROCm grouped floating down descriptor publication"))
                        {
                            return {};
                        }
                        auto *device_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                slot_base);

                        hipEvent_t ready_event = nullptr;
                        hipError_t err = hipEventCreateWithFlags(
                            &ready_event,
                            hipEventDisableTiming);
                        if (err != hipSuccess || !ready_event)
                            return {};
                        auto publication =
                            std::shared_ptr<
                                GroupedDescriptorWorkspacePublication>(
                                new GroupedDescriptorWorkspacePublication{
                                    .ready_event =
                                        static_cast<void *>(ready_event),
                                    .primary_descs = device_descs,
                                    .secondary_descs = nullptr,
                                    .workspace_slot = slot,
                                },
                                [](GroupedDescriptorWorkspacePublication *value)
                                {
                                    if (value && value->ready_event)
                                    {
                                        (void)hipEventDestroy(
                                            static_cast<hipEvent_t>(
                                                value->ready_event));
                                    }
                                    delete value;
                                });
                        err = empty_original_image
                            ? hipMemsetAsync(device_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_descs,table.host_floating_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                        if (err != hipSuccess)
                            return {};
                        err = hipEventRecord(ready_event, stream);
                        if (err != hipSuccess)
                            std::terminate();
                        return publication;
                    });
            if (!publication_result)
                return false;

            auto publication = std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
            const hipError_t wait_err = hipStreamWaitEvent(
                stream,
                static_cast<hipEvent_t>(publication->ready_event),
                0);
            if (wait_err != hipSuccess)
                return false;

            table.device_descs = nullptr;
            table.device_floating_descs =
                static_cast<DeviceMoEFloatingMatrixDesc *>(
                    publication->primary_descs);
            table.workspace_publication = std::move(publication);
            table.workspace_slot = publication_result.slot;
            return true;
        }

        const size_t desc_bytes =
            static_cast<size_t>(table.num_experts) *
            sizeof(DeviceNativeQuantisedMatrixDesc);
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    table.host_descs.data(),
                    nullptr,
                    table.num_experts,
                    table.d_model,
                    table.intermediate, table.descriptor_source),
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    DeviceNativeQuantisedMatrixDesc *device_descs = nullptr;
                    if (!bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                ROCM_GROUPED_DOWN_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_descs,
                            "ROCm grouped down descriptor publication"))
                    {
                        return {};
                    }

                    hipEvent_t ready_event = nullptr;
                    hipError_t err = hipEventCreateWithFlags(
                        &ready_event,
                        hipEventDisableTiming);
                    if (err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Failed to create grouped "
                                  "down descriptor readiness event: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    auto publication =
                        std::shared_ptr<
                            GroupedDescriptorWorkspacePublication>(
                            new GroupedDescriptorWorkspacePublication{
                                .ready_event =
                                    static_cast<void *>(ready_event),
                                .primary_descs = device_descs,
                                .secondary_descs = nullptr,
                                .workspace_slot = slot,
                            },
                            [](GroupedDescriptorWorkspacePublication *value)
                            {
                                if (value && value->ready_event)
                                {
                                    (void)hipEventDestroy(
                                        static_cast<hipEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    err = empty_original_image
                            ? hipMemsetAsync(device_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_descs,table.host_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped down descriptor "
                                  "H2D publication failed: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    err = hipEventRecord(ready_event, stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped down descriptor "
                                  "publication cannot record readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to publish grouped down "
                      "descriptor table");
            return false;
        }

        auto publication =
            std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
        const hipError_t wait_err = hipStreamWaitEvent(
            stream,
            static_cast<hipEvent_t>(publication->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to adopt grouped down "
                      "descriptor publication: "
                      << hipGetErrorString(wait_err));
            return false;
        }

        table.device_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->primary_descs);
        table.device_floating_descs = nullptr;
        table.workspace_publication = std::move(publication);
        table.workspace_slot = publication_result.slot;
        return true;
    }

    bool ROCmMoEKernel::publishGroupedGateUpDescriptorTable(
        GroupedGateUpDescriptorTable &table,
        const char *context)
    {
        const bool floating = deviceMoEWeightFormatIsFloating(
            table.weight_format);
        if (!workspace_ || !table.valid ||
            (floating ? table.host_floating_gate_descs.empty()
                      : table.host_gate_descs.empty()) ||
            (floating ? table.host_floating_up_descs.empty()
                      : table.host_up_descs.empty()) ||
            table.num_experts <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "gate/up descriptor publication")
                      << " requires a bound workspace and complete host tables");
            return false;
        }
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "gate/up descriptor publication")
                      << " requires an explicit HIP stream");
            return false;
        }

        // Source2 consumes live descriptors from its runtime bank. An empty
        // original template needs local initialization only: allocation E is
        // not an H2D communication extent. Keep the same slot and ready event.
        const bool empty_original_image = table.original_family.valid() &&
            (floating ? (std::all_of(table.host_floating_gate_descs.begin(),table.host_floating_gate_descs.end(),isBlankGroupedFloatingDesc) &&
               std::all_of(table.host_floating_up_descs.begin(),table.host_floating_up_descs.end(),isBlankGroupedFloatingDesc))
                      : (std::all_of(table.host_gate_descs.begin(),table.host_gate_descs.end(),isBlankGroupedDesc) &&
               std::all_of(table.host_up_descs.begin(),table.host_up_descs.end(),isBlankGroupedDesc)));

        if (floating)
        {
            const size_t desc_bytes =
                static_cast<size_t>(table.num_experts) *
                sizeof(DeviceMoEFloatingMatrixDesc);
            const auto publication_result =
                workspace_->getOrCreatePersistentPublication(
                    kROCmGroupedGateUpDescriptorLeaseDomain,
                    groupedFloatingDescriptorPublicationKey(
                        table.host_floating_gate_descs.data(),
                        table.host_floating_up_descs.data(),
                        table.weight_format,
                        table.num_experts,
                        table.d_model,
                        table.intermediate, table.descriptor_source),
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                    [&](std::size_t slot) -> std::shared_ptr<void>
                    {
                        DeviceNativeQuantisedMatrixDesc *gate_slot_base = nullptr;
                        DeviceNativeQuantisedMatrixDesc *up_slot_base = nullptr;
                        if (!bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::
                                    ROCM_GROUPED_GATE_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &gate_slot_base,
                                "ROCm grouped floating gate descriptor publication") ||
                            !bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::
                                    ROCM_GROUPED_UP_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &up_slot_base,
                                "ROCm grouped floating up descriptor publication"))
                        {
                            return {};
                        }
                        auto *device_gate_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                gate_slot_base);
                        auto *device_up_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                up_slot_base);

                        hipEvent_t ready_event = nullptr;
                        hipError_t err = hipEventCreateWithFlags(
                            &ready_event,
                            hipEventDisableTiming);
                        if (err != hipSuccess || !ready_event)
                            return {};
                        auto publication =
                            std::shared_ptr<
                                GroupedDescriptorWorkspacePublication>(
                                new GroupedDescriptorWorkspacePublication{
                                    .ready_event =
                                        static_cast<void *>(ready_event),
                                    .primary_descs = device_gate_descs,
                                    .secondary_descs = device_up_descs,
                                    .workspace_slot = slot,
                                },
                                [](GroupedDescriptorWorkspacePublication *value)
                                {
                                    if (value && value->ready_event)
                                    {
                                        (void)hipEventDestroy(
                                            static_cast<hipEvent_t>(
                                                value->ready_event));
                                    }
                                    delete value;
                                });
                        err = empty_original_image
                            ? hipMemsetAsync(device_gate_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_gate_descs,table.host_floating_gate_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                        if (err != hipSuccess)
                            return {};
                        err = empty_original_image
                            ? hipMemsetAsync(device_up_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_up_descs,table.host_floating_up_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                        if (err != hipSuccess)
                            std::terminate();
                        err = hipEventRecord(ready_event, stream);
                        if (err != hipSuccess)
                            std::terminate();
                        return publication;
                    });
            if (!publication_result)
                return false;

            auto publication = std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
            const hipError_t wait_err = hipStreamWaitEvent(
                stream,
                static_cast<hipEvent_t>(publication->ready_event),
                0);
            if (wait_err != hipSuccess)
                return false;

            table.device_gate_descs = nullptr;
            table.device_up_descs = nullptr;
            table.device_floating_gate_descs =
                static_cast<DeviceMoEFloatingMatrixDesc *>(
                    publication->primary_descs);
            table.device_floating_up_descs =
                static_cast<DeviceMoEFloatingMatrixDesc *>(
                    publication->secondary_descs);
            table.workspace_publication = std::move(publication);
            table.workspace_slot = publication_result.slot;
            return true;
        }

        const size_t desc_bytes =
            static_cast<size_t>(table.num_experts) *
            sizeof(DeviceNativeQuantisedMatrixDesc);
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    table.host_gate_descs.data(),
                    table.host_up_descs.data(),
                    table.num_experts,
                    table.d_model,
                    table.intermediate, table.descriptor_source),
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    DeviceNativeQuantisedMatrixDesc *device_gate_descs =
                        nullptr;
                    DeviceNativeQuantisedMatrixDesc *device_up_descs =
                        nullptr;
                    if (!bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                ROCM_GROUPED_GATE_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_gate_descs,
                            "ROCm grouped gate descriptor publication") ||
                        !bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                ROCM_GROUPED_UP_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_up_descs,
                            "ROCm grouped up descriptor publication"))
                    {
                        return {};
                    }

                    hipEvent_t ready_event = nullptr;
                    hipError_t err = hipEventCreateWithFlags(
                        &ready_event,
                        hipEventDisableTiming);
                    if (err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Failed to create grouped "
                                  "gate/up descriptor readiness event: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    auto publication =
                        std::shared_ptr<
                            GroupedDescriptorWorkspacePublication>(
                            new GroupedDescriptorWorkspacePublication{
                                .ready_event =
                                    static_cast<void *>(ready_event),
                                .primary_descs = device_gate_descs,
                                .secondary_descs = device_up_descs,
                                .workspace_slot = slot,
                            },
                            [](GroupedDescriptorWorkspacePublication *value)
                            {
                                if (value && value->ready_event)
                                {
                                    (void)hipEventDestroy(
                                        static_cast<hipEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    err = empty_original_image
                            ? hipMemsetAsync(device_gate_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_gate_descs,table.host_gate_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped gate descriptor "
                                  "H2D publication failed: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    err = empty_original_image
                            ? hipMemsetAsync(device_up_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_up_descs,table.host_up_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped up descriptor H2D "
                                  "publication failed after gate submission: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    err = hipEventRecord(ready_event, stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped gate/up descriptor "
                                  "publication cannot record readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to publish grouped gate/up "
                      "descriptor table");
            return false;
        }

        auto publication =
            std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
        const hipError_t wait_err = hipStreamWaitEvent(
            stream,
            static_cast<hipEvent_t>(publication->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to adopt grouped gate/up "
                      "descriptor publication: "
                      << hipGetErrorString(wait_err));
            return false;
        }

        table.device_gate_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->primary_descs);
        table.device_up_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->secondary_descs);
        table.device_floating_gate_descs = nullptr;
        table.device_floating_up_descs = nullptr;
        table.workspace_publication = std::move(publication);
        table.workspace_slot = publication_result.slot;
        return true;
    }

    bool ROCmMoEKernel::rebindGroupedDescriptorTablesToWorkspace(const char *context)
    {
        if (grouped_down_desc_tables_.empty() && grouped_gateup_desc_tables_.empty())
            return true;
        if (rejectDecodeStagingDuringCapture("rebind grouped descriptor tables"))
            return false;
        if (!setMoEDevice(device_ordinal_, context ? context : "rebindGroupedDescriptorTablesToWorkspace"))
            return false;
        for (auto &table : grouped_down_desc_tables_)
        {
            const bool floating = deviceMoEWeightFormatIsFloating(
                table.weight_format);
            if (!table.valid ||
                (floating ? table.host_floating_descs.empty()
                          : table.host_descs.empty()) ||
                table.num_experts <= 0)
                continue;
            if (!publishGroupedDownDescriptorTable(table, context))
                return false;
        }

        for (auto &table : grouped_gateup_desc_tables_)
        {
            const bool floating = deviceMoEWeightFormatIsFloating(
                table.weight_format);
            if (!table.valid ||
                (floating ? table.host_floating_gate_descs.empty()
                          : table.host_gate_descs.empty()) ||
                (floating ? table.host_floating_up_descs.empty()
                          : table.host_up_descs.empty()) ||
                table.num_experts <= 0)
            {
                continue;
            }
            if (!publishGroupedGateUpDescriptorTable(table, context))
                return false;
        }
        return true;
    }

    ROCmMoEKernel::~ROCmMoEKernel()
    {
        (void)setMoEDevice(device_ordinal_, "destructor");
        clearWorkspaceScratchBindings();
    }

    void ROCmMoEKernel::resetDynamicState()
    {
        grouped_prefill_grouping_.invalidate();
        if (!setMoEDevice(device_ordinal_, "resetDynamicState"))
            return;

        /*
         * This hook is a hard kernel-dynamic reset, not the replay-preserving
         * request boundary. Captured HIP graph replay keeps kernel-owned
         * workspace tables alive by avoiding KernelFactory::resetAllDynamicState().
         * When this hook is invoked, grouped scratch, mask upload hashes,
         * histogram buffers, pointer-table caches, and descriptor registries
         * must be rebuilt before the next eager warmup or capture.
         */
        clearWorkspaceScratchBindings();
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
    }

    bool ROCmMoEKernel::runMoEOverlayEpochParentAction(
        const MoEKernelLaunchContext &launch,DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochStatus *status,DeviceMoEOverlayEpochParentAction action)
    {
        void *stream=explicitMoELaunchStream(launch,"runMoEOverlayEpochParentAction");
        if(!stream || !control || !status)return false;
        return hipMoEOverlayEpochParentAction(control,status,action,device_ordinal_,stream);
    }

    bool ROCmMoEKernel::acquireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status,
        const std::uint64_t *external_admission_epoch,
        DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier,
        MoEOverlayPeerPlacementEpochBinding peer_placement_epoch)
    {
        void *stream = explicitMoELaunchStream(
            launch, "acquireMoEOverlayEpoch");
        if (!stream || !control || !ticket || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::acquireMoEOverlayEpoch] control, ticket, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochAcquire(
            control,
            ticket,
            status,
            external_admission_epoch,
            admission_barrier,
            peer_placement_epoch,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::releaseMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "releaseMoEOverlayEpoch");
        if (!stream || !control || !ticket || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::releaseMoEOverlayEpoch] control, ticket, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochRelease(
            control, ticket, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::reserveMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "reserveMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::reserveMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochReserveCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::markMoEOverlayEpochCandidateReady(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "markMoEOverlayEpochCandidateReady");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::markMoEOverlayEpochCandidateReady] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochMarkCandidateReady(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::publishMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::publishMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochPublishCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::abortMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "abortMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::abortMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochAbortCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "retireMoEOverlayEpoch");
        if (!stream || !control || !retiring_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::retireMoEOverlayEpoch] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochRetire(
            control, retiring_epoch, status, device_ordinal_, stream);
    }

    void ROCmMoEKernel::syncBlasStream()
    {
        if (blas_gemm_)
            blas_gemm_->bindStream(ExplicitGPUStream{ROCmKernelBase::getStream()});
    }

    void ROCmMoEKernel::clearGPUStreamBinding()
    {
        ROCmKernelBase::clearGPUStreamBinding();
        if (blas_gemm_)
            blas_gemm_->clearStreamBinding();
    }

    bool ROCmMoEKernel::ensureSharedGateScratchCapacity(int seq_len)
    {
        if (seq_len <= shared_gate_scratch_capacity_)
            return true;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_shared_gate_scratch_),
                                 MoEWorkspaceBuffers::ROCM_SHARED_GATE,
                                 static_cast<size_t>(seq_len) * sizeof(float),
                                 "ensureSharedGateScratchCapacity"))
        {
            shared_gate_scratch_capacity_ = 0;
            return false;
        }
        shared_gate_scratch_capacity_ = seq_len;
        return true;
    }

    bool ROCmMoEKernel::ensureRouteBufferCapacity(size_t logits_count)
    {
        if (logits_count > route_logits_capacity_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_logits_),
                                     MoEWorkspaceBuffers::ROUTE_LOGITS,
                                     logits_count * sizeof(float),
                                     "ensureRouteBufferCapacity(logits)"))
            {
                route_logits_capacity_ = 0;
                return false;
            }
            route_logits_capacity_ = logits_count;
        }

        return d_route_logits_ != nullptr;
    }

    bool ROCmMoEKernel::ensureRouteLogitsPartialsCapacity(size_t partial_count)
    {
        if (partial_count == 0)
            return false;

        if (partial_count <= route_logits_partials_capacity_)
            return d_route_logits_partials_ != nullptr;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_logits_partials_),
                                 MoEWorkspaceBuffers::ROCM_ROUTE_LOGITS_PARTIALS,
                                 partial_count * sizeof(float),
                                 "ensureRouteLogitsPartialsCapacity"))
        {
            d_route_logits_partials_ = nullptr;
            route_logits_partials_capacity_ = 0;
            return false;
        }

        route_logits_partials_capacity_ = partial_count;
        return true;
    }

    bool ROCmMoEKernel::ensureRouterQ8HiddenScratchCapacity(int rows, int d_model)
    {
        if (rows <= 0 || d_model <= 0 || (d_model % 32) != 0)
            return false;

        const int blocks_per_row = d_model / 32;
        if (rows <= router_q8_hidden_rows_cap_ &&
            d_model <= router_q8_hidden_d_model_cap_ &&
            blocks_per_row <= router_q8_hidden_blocks_cap_ &&
            d_router_q8_hidden_ && d_router_q8_hidden_scales_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_router_q8_hidden_),
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN,
                                 static_cast<size_t>(rows) *
                                     static_cast<size_t>(d_model) * sizeof(int8_t),
                                 "ensureRouterQ8HiddenScratchCapacity(hidden)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_router_q8_hidden_scales_),
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES,
                                 static_cast<size_t>(rows) *
                                     static_cast<size_t>(blocks_per_row) * sizeof(float),
                                 "ensureRouterQ8HiddenScratchCapacity(scales)"))
        {
            d_router_q8_hidden_ = nullptr;
            d_router_q8_hidden_scales_ = nullptr;
            router_q8_hidden_rows_cap_ = 0;
            router_q8_hidden_d_model_cap_ = 0;
            router_q8_hidden_blocks_cap_ = 0;
            return false;
        }

        router_q8_hidden_rows_cap_ = rows;
        router_q8_hidden_d_model_cap_ = d_model;
        router_q8_hidden_blocks_cap_ = blocks_per_row;
        return true;
    }

    bool ROCmMoEKernel::prepareOriginalM1HiddenQ8Publication(const TensorBase* hidden,DeviceRowRange rows)
    {
        const auto identity=rows.captureIdentity();auto& publication=router_q8_hidden_publication_;
        if(isGraphCaptureActive()||!publication||!publication->requiredOriginalM1Source()||
           hidden!=publication->requiredOriginalM1Source()||identity!=DeviceRowRange::fullyActive(1).captureIdentity()||!workspace_)
            throw std::invalid_argument("Original M1 hidden Q8 requires its cold declared fully-active source");
        const auto stream=getStream();const int width=publication->requiredOriginalM1Width();
        if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumer) {
            if(!publication->original_m1_binding)
                publication->original_m1_binding=std::make_shared<OriginalM1HiddenQ8Binding>(
                    DeviceId::rocm(device_ordinal_),*hidden,width,stream,*workspace_,OriginalM1HiddenQ8Bank::RouterROCm);
            publication->original_m1_binding->validate(*hidden,stream,workspace_);
        } else {
            if(!publication->original_m1_binding)throw std::logic_error("Original M1 hidden Q8 producer must prepare before its required reader");
            if(original_m1_hidden_q8_read_lifecycle_==OriginalM1HiddenQ8ReadLifecycle::Invalidated)
                throw std::logic_error("Original M1 hidden Q8 reader was retired by hard scratch reset");
            if(!original_m1_hidden_q8_read_)
                original_m1_hidden_q8_read_=std::make_shared<OriginalM1HiddenQ8ReadView>(
                    publication->original_m1_binding,*hidden,stream,*workspace_);
            original_m1_hidden_q8_read_->validate(*hidden,stream,workspace_);
            original_m1_hidden_q8_read_lifecycle_=OriginalM1HiddenQ8ReadLifecycle::Prepared;
        }
        return true;
    }

    std::uint64_t ROCmMoEKernel::originalM1HiddenQ8BindingSignature() const
    {
        if(!router_q8_hidden_publication_||!router_q8_hidden_publication_->requiredOriginalM1Source())return 0;
        const auto& publication=*router_q8_hidden_publication_;
        if(publication.original_m1_binding) {
            const auto stream=ExplicitGPUStream(gpu_stream_).get();
            if(original_m1_hidden_q8_read_lifecycle_==OriginalM1HiddenQ8ReadLifecycle::Invalidated)
                throw std::logic_error("Original M1 hidden Q8 reader lost its prepared scratch lifetime");
            if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumer)
                publication.original_m1_binding->validate(*publication.requiredOriginalM1Source(),stream,workspace_);
            else if(original_m1_hidden_q8_read_)
                original_m1_hidden_q8_read_->validate(*publication.requiredOriginalM1Source(),stream,workspace_);
        }
        // Declaration identity stays stable before/after first cold preparation;
        // leases independently authenticate every embedded pointer on replay.
        return 0x4d31513800000001ULL;
    }

    bool ROCmMoEKernel::prepareRequiredRouterQ8Hidden(int rows, int width, int top_k)
    {
        (void)top_k;
        if (!router_q8_hidden_publication_ ||
            !router_q8_hidden_publication_->hasRequiredConsumer())
            return true;
        if (router_q8_publication_access_ != MoERouterQ8PublicationAccess::ProducerAndConsumer ||
            !router_q8_hidden_publication_->acceptsProducerRows(rows, width))
        {
            LOG_ERROR("[ROCmMoEKernel] required Q8 producer geometry/access mismatch");
            return false;
        }
        return ensureRouterQ8HiddenScratchCapacity(rows, width);
    }

    bool ROCmMoEKernel::publishRequiredRouterQ8Hidden(
        const float *source, int rows, int width, int top_k, const int *live_rows)
    {
        if (!router_q8_hidden_publication_ ||
            !router_q8_hidden_publication_->hasRequiredConsumer())
            return true;
        if (!prepareRequiredRouterQ8Hidden(rows, width, top_k) ||
            !hipMoE_publish_original_hidden_q8_rows(source, d_router_q8_hidden_,
                d_router_q8_hidden_scales_, rows, width, device_ordinal_, getStream(), live_rows))
            return false;
        publishRouterQ8Hidden(source, rows,
            isGraphCaptureActive() ||
                (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
                isHipStreamCapturing(getStream()),
            RouterQ8Basis::Original);
        return true;
    }

    void ROCmMoEKernel::invalidateRouterQ8HiddenPublication() noexcept
    {
        if (router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::ProducerAndConsumer &&
            router_q8_hidden_publication_)
        {
            router_q8_hidden_publication_->clearPayload();
        }
    }

    bool ROCmMoEKernel::bindRouterQ8HiddenPublication(
        std::shared_ptr<MoERouterQ8HiddenPublication> publication,
        MoERouterQ8PublicationAccess access)
    {
        if (!publication)
        {
            LOG_ERROR("[ROCmMoEKernel::bindRouterQ8HiddenPublication] null publication");
            return false;
        }
        if (publication->device_ordinal >= 0 &&
            (publication->backend != DeviceType::ROCm ||
             publication->device_ordinal != device_ordinal_))
        {
            LOG_ERROR("[ROCmMoEKernel::bindRouterQ8HiddenPublication] device mismatch"
                      << " publication_backend="
                      << static_cast<int>(publication->backend)
                      << " publication_ordinal=" << publication->device_ordinal
                      << " kernel_ordinal=" << device_ordinal_);
            return false;
        }

        publication->backend = DeviceType::ROCm;
        publication->device_ordinal = device_ordinal_;
        if(router_q8_hidden_publication_&&router_q8_hidden_publication_->original_m1_binding&&
           (publication!=router_q8_hidden_publication_||access!=router_q8_publication_access_))
            throw std::logic_error("Captured original M1 hidden Q8 publication/access cannot be rebound");
        router_q8_hidden_publication_ = std::move(publication);
        router_q8_publication_access_ = access;
        return true;
    }

    void ROCmMoEKernel::publishRouterQ8Hidden(
        const float *source,
        int rows,
        bool recorded_during_capture,
        RouterQ8Basis router_basis) noexcept
    {
        if (router_q8_publication_access_ !=
                MoERouterQ8PublicationAccess::ProducerAndConsumer ||
            !router_q8_hidden_publication_)
        {
            LOG_ERROR("[ROCmMoEKernel::publishRouterQ8Hidden] consumer-only kernel attempted publication");
            return;
        }
        if (!source || rows <= 0 ||
            rows > router_q8_hidden_rows_cap_ ||
            !d_router_q8_hidden_ || !d_router_q8_hidden_scales_)
        {
            invalidateRouterQ8HiddenPublication();
            return;
        }

        auto &publication = *router_q8_hidden_publication_;
        publication.backend = DeviceType::ROCm;
        publication.device_ordinal = device_ordinal_;
        publication.source_rows = source;
        publication.quantized_rows = d_router_q8_hidden_;
        publication.row_scales = d_router_q8_hidden_scales_;
        publication.published_rows = rows;
        publication.d_model_capacity = router_q8_hidden_d_model_cap_;
        publication.blocks_per_row_capacity =
            router_q8_hidden_blocks_cap_;
        publication.capture_recorded = recorded_during_capture;
        if (router_basis == RouterQ8Basis::SignedHadamard32)
            PerfStatsCollector::addCounter("kernel", "rocm_moe_router_hadamard32_calls", 1.0,
                "router", "rocm:" + std::to_string(device_ordinal_),
                {{"rows", std::to_string(rows)}, {"expert_hidden_basis", "original"}, {"block", "32"}});

    }

    bool ROCmMoEKernel::canReuseRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model) const
    {
        if(router_q8_hidden_publication_&&router_q8_hidden_publication_->requiredOriginalM1Source()) {
            const auto& p=*router_q8_hidden_publication_;const auto& binding=p.original_m1_binding;
            if(!binding||rows!=1||d_model!=p.requiredOriginalM1Width()||source!=binding->source())
                throw std::logic_error("Required original M1 hidden Q8 source/row geometry is absent");
            (void)originalM1HiddenQ8BindingSignature();
            if(p.backend!=DeviceType::ROCm||p.device_ordinal!=device_ordinal_||p.d_model_capacity<d_model||p.blocks_per_row_capacity<d_model/32||
               p.source_rows!=source||p.quantized_rows!=binding->codes()||p.row_scales!=binding->scales()||p.published_rows!=1||
               (p.capture_recorded&&!isGraphCaptureActive()))
                throw std::logic_error("Required original M1 hidden Q8 payload/provenance is absent");
            return true;
        }
        if (!debugEnv().rocm.moe_reuse_router_q8_hidden ||
            !router_q8_hidden_publication_ ||
            rows <= 0 ||
            d_model <= 0 ||
            router_q8_hidden_publication_->backend != DeviceType::ROCm ||
            router_q8_hidden_publication_->device_ordinal != device_ordinal_ ||
            router_q8_hidden_publication_->source_rows != source ||
            router_q8_hidden_publication_->published_rows < rows ||
            router_q8_hidden_publication_->d_model_capacity < d_model ||
            router_q8_hidden_publication_->blocks_per_row_capacity <
                ((d_model + 31) / 32) ||
            !router_q8_hidden_publication_->quantized_rows ||
            !router_q8_hidden_publication_->row_scales)
        {
            return false;
        }

        /*
         * A capture producer and consumer are ordered correctly on the same
         * HIP stream even though neither has run yet.  Outside capture, those
         * recorded kernels have not materialized bytes, so only an eager
         * publication may be reused by eager expert decode.
         */
        return !router_q8_hidden_publication_->capture_recorded ||
               isDecodeGraphCaptureActive();
    }

    const ROCmMoEKernel::RouterQ8GateCacheEntry *ROCmMoEKernel::getOrCreateQ8RouterGateCache(
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().rocm.moe_router_q8)
            return nullptr;
        if (!gate_device_ptr || d_model <= 0 || num_experts <= 0 ||
            (d_model % 32) != 0)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] invalid Q8 router gate request "
                      "(gate_device_ptr=" << static_cast<const void *>(gate_device_ptr)
                      << " d_model=" << d_model
                      << " num_experts=" << num_experts << ")");
            return nullptr;
        }
        if (!workspace_)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                      "graph-owned workspace is required before cache publication");
            return nullptr;
        }

        const DeviceResidentRouterGateCacheKey cache_key =
            DeviceResidentRouterGateCacheKey::make(
                workspace_->id(), gate_device_ptr, d_model, num_experts,
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        for (const auto &entry : router_q8_gate_cache_)
        {
            if (entry.key == cache_key &&
                entry.d_gate_weights_q8 && entry.d_gate_scales)
            {
                return &entry;
            }
        }

        const bool capture_active = isGraphCaptureActive() ||
                                    (deviceContext() && deviceContext()->isDeviceGraphCaptureActive());
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        void *stream = getStream();
        if (stream)
        {
            const hipError_t capture_err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture_status);
            if (capture_err != hipSuccess)
                capture_status = hipStreamCaptureStatusNone;
        }
        if (capture_active || capture_status == hipStreamCaptureStatusActive)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] Q8 router cache miss during graph capture");
            return nullptr;
        }
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] explicit HIP stream is required");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] router gate size overflow");
            return nullptr;
        }
        const size_t element_count = d_model_sz * experts_sz;
        const int blocks_per_row = d_model / 32;
        const size_t scale_count = static_cast<size_t>(num_experts) * static_cast<size_t>(blocks_per_row);

        if (!setMoEDevice(device_ordinal_, "getOrCreateQ8RouterGateCache"))
            return nullptr;

        void *gate_weights_base = nullptr;
        void *gate_scales_base = nullptr;
        const size_t weights_payload_bytes = element_count * sizeof(int8_t);
        const size_t scales_payload_bytes = scale_count * sizeof(float);
        if (!bindWorkspaceBuffer(&gate_weights_base,
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_WEIGHTS,
                                 weights_payload_bytes,
                                 "ROCm Q8 router gate weights") ||
            !bindWorkspaceBuffer(&gate_scales_base,
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_SCALES,
                                 scales_payload_bytes,
                                 "ROCm Q8 router gate scales"))
        {
            return nullptr;
        }

        /*
         * Model router weights are immutable, while the paired routed pipeline
         * kernels are graph-local by design. Publish one conversion through the
         * workspace manager and let every graph adopt it. The setup factory
         * records an event before exposing the publication, so cross-stream
         * adoption is an explicit HIP dependency rather than a host sync.
         */
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmRouterQ8GatePublicationDomain,
                routerPublicationKey(cache_key),
                MoEWorkspaceBuffers::kRouterGateCacheSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    void *weights = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_WEIGHTS,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        weights_payload_bytes);
                    void *scales = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_SCALES,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        scales_payload_bytes);
                    if (!weights || !scales)
                        return {};

                    hipEvent_t ready_event = nullptr;
                    hipError_t event_err = hipEventCreateWithFlags(
                        &ready_event, hipEventDisableTiming);
                    if (event_err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "failed to create immutable publication event: "
                                  << hipGetErrorString(event_err));
                        return {};
                    }

                    auto publication =
                        std::shared_ptr<RouterGateWorkspacePublication>(
                            new RouterGateWorkspacePublication{
                                .ready_event = static_cast<void *>(ready_event),
                                .primary_weights = weights,
                                .scales = scales,
                                .workspace_slot = slot,
                            },
                            [](RouterGateWorkspacePublication *value)
                            {
                                if (value && value->ready_event)
                                {
                                    (void)hipEventDestroy(
                                        static_cast<hipEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    if (!hipMoE_quantize_router_gate_q8(
                            gate_device_ptr,
                            static_cast<int8_t *>(weights),
                            static_cast<float *>(scales),
                            d_model,
                            num_experts,
                            device_ordinal_,
                            stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic)))
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "FP32->Q8 router gate conversion launch failed");
                        return {};
                    }

                    event_err = hipEventRecord(
                        ready_event, static_cast<hipStream_t>(stream));
                    if (event_err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "failed to record immutable publication event: "
                                  << hipGetErrorString(event_err));
                        /*
                         * Conversion work is already in flight. There is no
                         * coherent unwind without its readiness event, and a
                         * stream/device sync would be an implicit recovery
                         * path. Stop the process before stale bytes can escape.
                         */
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                      "failed to publish immutable router gate");
            return nullptr;
        }

        auto publication =
            std::static_pointer_cast<RouterGateWorkspacePublication>(
                publication_result.publication);
        const hipError_t wait_err = hipStreamWaitEvent(
            static_cast<hipStream_t>(stream),
            static_cast<hipEvent_t>(publication->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                      "failed to adopt immutable router publication: "
                      << hipGetErrorString(wait_err));
            return nullptr;
        }

        RouterQ8GateCacheEntry entry{};
        entry.key = cache_key;
        entry.blocks_per_row = blocks_per_row;
        entry.element_count = element_count;
        entry.scale_count = scale_count;
        entry.d_gate_weights_q8 =
            static_cast<int8_t *>(publication->primary_weights);
        entry.d_gate_scales =
            static_cast<float *>(publication->scales);
        entry.workspace_publication = std::move(publication);
        entry.workspace_slot = publication_result.slot;
        router_q8_gate_cache_.push_back(std::move(entry));

        LOG_TRACE("[ROCmMoEKernel] Adopted Q8 router gate source_device="
                  << static_cast<const void *>(gate_device_ptr)
                  << " workspace_id=" << cache_key.workspace_id
                  << " slot=" << publication_result.slot
                  << " created=" << publication_result.created
                  << " shape=[" << num_experts << "," << d_model << "] payload_bytes="
                  << element_count << " scale_bytes=" << (scale_count * sizeof(float)));
        return &router_q8_gate_cache_.back();
    }


    // =========================================================================
    // routeCore() — Shared GPU routing logic: gate GEMM + direct FP32 publication.
    // =========================================================================

    bool ROCmMoEKernel::routeCore(
        const float *hidden, const void *gate_weights, TensorType gate_type,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        float *output_indices, float *output_weights,
        const int *device_effective_seq_len,
        const MoERouterOwnedRowOutputs *owned)
    {
        if (!owned && (!output_indices || !output_weights))
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] final FP32 route outputs are required");
            return false;
        }
        const size_t logits_count = static_cast<size_t>(seq_len) * num_experts;
        if (!ensureRouteBufferCapacity(logits_count))
            return false;
        if (device_effective_seq_len &&
            !validateDevicePointerOrLog(device_effective_seq_len,
                                        device_ordinal_,
                                        "effective prefill sequence length",
                                        "ROCmMoEKernel::routeCore"))
        {
            return false;
        }

        // The two destinations deliberately share gate preparation and hidden-Q8
        // ownership. The packet specialization also performs top-k publication,
        // so complete-route top-k below must not run a second time.
        const auto route_owned = [&](MoERouterPreparedFormat format, const void *gate,
                                     const float *scales = nullptr) {
            const MoERouterOwnedRowsLaunch launch{
                .partition = owned->partition,
                .capacity = seq_len, .width = d_model, .experts = num_experts,
                .top_k = top_k, .normalize = normalize_weights,
                .live_rows = device_effective_seq_len, .hidden = hidden,
                .format = format,
                .q8_basis = format == MoERouterPreparedFormat::BlockQ8 ?
                    routerQ8Basis(debugEnv().gemm.moe_router_arithmetic) : RouterQ8Basis::Original,
                .gate = gate, .gate_scales = scales,
                .hidden_q8 = d_router_q8_hidden_, .hidden_scales = d_router_q8_hidden_scales_,
                .logits = d_route_logits_, .selected = owned->selected,
                .selected_bytes = owned->selected_bytes};
            return rocm::routeOwnedRows(launch, getStream());
        };
        const bool decode_single_token = (seq_len == 1);
        if (owned && decode_single_token) return false;
        bool used_q8_grouped_router = false;
        bool used_fp16_grouped_router = false;
        invalidateRouterQ8HiddenPublication();
        if (!prepareRequiredRouterQ8Hidden(seq_len, d_model, top_k))
            return false;
        const bool router_publishes_q8 = !decode_single_token &&
            gate_type == TensorType::FP32 && debugEnv().rocm.moe_router_q8;
        if (!router_publishes_q8 && !publishRequiredRouterQ8Hidden(
                hidden, seq_len, d_model, top_k, device_effective_seq_len))
            return false;
        if (decode_single_token)
        {
            if (!launchDecodeGateLogitsForGateType(
                    hidden, gate_weights, gate_type, d_route_logits_,
                    d_model, num_experts,
                    device_ordinal_, getStream(),
                    "ROCmMoEKernel::routeCore"))
            {
                return false;
            }
        }
        else
        {
            /*
             * Prefill routing is part of autoregressive state construction, so
             * changing the prefill bucket must not change a row's logits or
             * top-k weights.  Every native gate format therefore uses the same
             * expert-owned, four-row-tiled reduction shape as serial decode.
             * The tile shares weight loads across rows without introducing a
             * batch-shaped GEMM reduction or floating-point atomic publication.
             */
            bool logits_ready = false;
            if (gate_type == TensorType::FP32)
            {
                const auto &rocm_env = debugEnv().rocm;
                if (rocm_env.moe_router_q8)
                {
                    if ((d_model % 32) != 0 ||
                        !ensureRouterQ8HiddenScratchCapacity(seq_len, d_model))
                    {
                        LOG_ERROR("[ROCmMoEKernel::routeCore] batch-invariant Q8 prefill router "
                                  "requires 32-column blocks and declared row scratch");
                        return false;
                    }
                    const auto *q8_gate = getOrCreateQ8RouterGateCache(
                        static_cast<const float *>(gate_weights),
                        d_model,
                        num_experts);
                    if (!q8_gate)
                    {
                        LOG_ERROR("[ROCmMoEKernel::routeCore] Q8 prefill router gate cache unavailable");
                        return false;
                    }
                    logits_ready = owned
                        ? route_owned(MoERouterPreparedFormat::BlockQ8,
                                      q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales)
                        : hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
                        hidden,
                        d_router_q8_hidden_,
                        d_router_q8_hidden_scales_,
                        q8_gate->d_gate_weights_q8,
                        q8_gate->d_gate_scales,
                        d_route_logits_,
                        seq_len,
                        d_model,
                        num_experts,
                        device_ordinal_,
                        getStream(),
                        device_effective_seq_len, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
                    if (logits_ready)
                    {
                        used_q8_grouped_router = true;
                        publishRouterQ8Hidden(
                            hidden,
                            seq_len,
                            isGraphCaptureActive() ||
                                (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
                                isHipStreamCapturing(getStream()),
                            routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
                    }
                }
                else
                {
                    logits_ready = owned ? route_owned(MoERouterPreparedFormat::FP32, gate_weights)
                        : launchDecodeEquivalentFP32Rows(
                        hidden,
                        static_cast<const float *>(gate_weights),
                        d_route_logits_,
                        seq_len,
                        d_model,
                        num_experts,
                        device_ordinal_,
                        getStream(),
                        device_effective_seq_len);
                }
            }
            else if (gate_type == TensorType::FP16)
            {
                logits_ready = owned ? route_owned(MoERouterPreparedFormat::FP16, gate_weights)
                    : hipMoE_gate_logits_fp16_decode_equivalent_rows(
                    hidden,
                    gate_weights,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len);
                used_fp16_grouped_router = logits_ready;
            }
            else if (gate_type == TensorType::BF16)
            {
                logits_ready = owned ? route_owned(MoERouterPreparedFormat::BF16, gate_weights)
                    : hipMoE_gate_logits_bf16_decode_equivalent_rows(
                    hidden,
                    gate_weights,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len);
            }
            else
            {
                LOG_ERROR("[ROCmMoEKernel::routeCore] unsupported batch-invariant prefill router gate type "
                          << tensorTypeName(gate_type));
                return false;
            }

            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::routeCore] batch-invariant grouped prefill logits failed");
                return false;
            }
        }

        // The row-owned top-k kernel mirrors serial decode and independently
        // handles every padded graph row from the device effective-length scalar.
        if (!owned && !hipMoE_softmax_topk_decode_equivalent_rows(
                d_route_logits_,
                output_indices,
                output_weights,
                seq_len,
                num_experts,
                top_k,
                normalize_weights,
                device_ordinal_,
                getStream(),
                device_effective_seq_len))
        {
            return false;
        }

        if (!decode_single_token)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_batch_invariant_prefill_router_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"row_tile", "16"},
                 {"route", used_q8_grouped_router
                               ? "grouped_q8"
                               : (used_fp16_grouped_router ? "grouped_fp16"
                                                          : "grouped_native")},
                 {"device_effective_len", device_effective_seq_len ? "1" : "0"}});
        }

        return true;
    }

    // =========================================================================
    // gatherTokenBatch() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::gatherTokenBatch(
        const float *hidden,
        float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatch"))
            return;

        if (!hipMoE_gather_tokens(hidden, batch_buffer, token_indices,
                                  num_tokens, d_model,
                                  device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatch] kernel launch failed");
        }
    }

    // =========================================================================
    // scatterAddWeighted() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::scatterAddWeighted(
        float *output,
        const float *expert_output,
        const int *token_indices,
        const float *weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterAddWeighted"))
            return;

        if (!hipMoE_scatter_add(output, expert_output, token_indices, weights,
                                num_tokens, d_model,
                                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeighted] kernel launch failed");
        }
    }

    // =========================================================================
    // sharedExpertGate() — All pointers are device pointers
    //
    // Needs a small scratch buffer for gate values [seq_len floats].
    // Allocates on demand (small — at most a few KB).
    // =========================================================================

    void ROCmMoEKernel::sharedExpertGate(
        const float *input,
        const float *gate_inp,
        float *shared_output,
        int seq_len, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SHARED_GATE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || d_model <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "sharedExpertGate"))
            return;

        if (seq_len == 1)
        {
            if (!hipMoE_shared_expert_gate_decode_fused(input, gate_inp, shared_output,
                                                        d_model,
                                                        device_ordinal_, getStream()))
            {
                LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] fused decode kernel launch failed");
            }
            return;
        }

        if (!ensureSharedGateScratchCapacity(seq_len))
            return;

        if (!hipMoE_shared_expert_gate(input, gate_inp, shared_output, d_shared_gate_scratch_,
                                       seq_len, d_model,
                                       device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] kernel launch failed");
        }
    }

    // =========================================================================
    // swiGLU() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "swiGLU"))
            return;

        if (!hipMoE_swiglu(gate, up, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLU] kernel launch failed");
        }
    }

    void ROCmMoEKernel::weightedAdd(float *output, const float *input,
                                    float weight, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "weightedAdd"))
            return;

        if (!hipMoE_weighted_add(output, input, weight, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAdd] kernel launch failed");
        }
    }

    // =========================================================================
    // Device-side token grouping
    // =========================================================================

    /**
     * @brief Reset all grouping publications before the retained HIP pipeline.
     * @param d_routing_indices Original expert IDs, including invalid slots.
     * @param d_routing_weights Immutable per-route probability bits.
     * @param seq_len Admitted token extent, including inactive bucket capacity.
     * @param num_experts Number of expert counters and scatter cursors.
     * @param top_k Route slots per token.
     * @param d_expert_offsets Output exclusive expert-count prefix.
     * @param d_expert_counts Output integer expert counts.
     * @param d_grouped_token_indices Output compact route IDs; unused slots are zero.
     * @param d_grouped_weights Output compact probabilities; unused slots are zero.
     * @return False on binding, initialization, counting, scanning or scatter failure.
     */
    bool ROCmMoEKernel::groupTokensByExpertDevice(
        const int *d_routing_indices,
        const float *d_routing_weights,
        int seq_len, int num_experts, int top_k,
        int *d_expert_offsets,
        int *d_expert_counts,
        int *d_grouped_token_indices,
        float *d_grouped_weights)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "groupTokensByExpertDevice"))
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        // Bind the existing arena cursor before the common reset. Counting and
        // scanning do not touch cursors, so they can be reset alongside maps.
        if (!d_write_heads_ || max_write_heads_experts_ < num_experts)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_write_heads_),
                                     MoEWorkspaceBuffers::GROUP_WRITE_HEADS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "groupTokensByExpertDevice(write_heads)"))
            {
                d_write_heads_ = nullptr;
                max_write_heads_experts_ = 0;
                return false;
            }
            max_write_heads_experts_ = num_experts;
        }
        if (!hipMoE_initialize_grouping({
                .expert_counts = d_expert_counts,
                .write_heads = d_write_heads_,
                .original_to_grouped = d_group_original_to_grouped_,
                .grouped_token_indices = d_grouped_token_indices,
                .grouped_weights = d_grouped_weights,
                .num_experts = num_experts,
                .total_slots = total_slots}, device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] grouping initialization failed");
            return false;
        }

        // Step 2: Count per expert
        if (!hipMoE_count_per_expert(d_routing_indices, d_expert_counts,
                                     total_slots, num_experts,
                                     device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] count_per_expert failed");
            return false;
        }

        // Step 3: Exclusive scan (expert_counts → expert_offsets)
        if (!hipMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                   num_experts,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] exclusive_scan failed");
            return false;
        }

        if (d_group_active_expert_ids_ &&
            !hipMoE_build_active_expert_list(
                d_expert_counts, d_group_active_expert_ids_,
                num_experts, std::min(total_slots, num_experts),
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] active expert list failed");
            return false;
        }

        // Scatter consumes the cursors reset before counting on this stream.
        if (!hipMoE_scatter_tokens(d_routing_indices, d_routing_weights,
                                   d_expert_offsets, d_write_heads_,
                                   d_grouped_token_indices,
                                   d_group_original_to_grouped_,
                                   d_grouped_weights,
                                   total_slots, num_experts, top_k,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] scatter_tokens failed");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Tensor-aware API overrides — GPU implementations
    //
    // These use gpu_data_ptr() for device-resident data and keep
    // all computation on device.  Small host-side arrays (token indices,
    // routing weights) are uploaded via cached staging buffers.
    // =========================================================================

    bool ROCmMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return d_staging_indices_ && d_staging_weights_;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_staging_indices_),
                                 MoEWorkspaceBuffers::STAGING_INDICES,
                                 static_cast<size_t>(count) * sizeof(int),
                                 "ensureStagingCapacity(indices)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_staging_weights_),
                                 MoEWorkspaceBuffers::STAGING_WEIGHTS,
                                 static_cast<size_t>(count) * sizeof(float),
                                 "ensureStagingCapacity(weights)"))
        {
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return false;
        }
        staging_capacity_ = count;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedDecodeCapacity(
        int num_active,
        int intermediate,
        int d_model)
    {
        if (num_active <= 0 || intermediate <= 0 || d_model <= 0 ||
            (intermediate % 32) != 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedDecodeCapacity"))
            return false;

        const int blocks_per_row = intermediate / 32;
        const bool pointer_capacity_ok = grouped_decode_active_cap_ >= num_active;
        const bool activation_capacity_ok = grouped_decode_intermediate_cap_ >= intermediate;
        const bool output_capacity_ok = grouped_decode_d_model_cap_ >= d_model;
        if (pointer_capacity_ok && activation_capacity_ok && output_capacity_ok &&
            d_grouped_gate_ptrs_ && d_grouped_up_ptrs_ && d_grouped_decode_weights_ &&
            d_grouped_expert_ids_ && d_grouped_down_descs_ &&
            d_grouped_swiglu_int8_ && d_grouped_swiglu_scales_ &&
            d_grouped_down_partials_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_GATE_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedDecodeCapacity(gate_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_UP_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedDecodeCapacity(up_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_expert_ids_),
                                 MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 "ensureGroupedDecodeCapacity(expert_ids)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_decode_weights_),
                                 MoEWorkspaceBuffers::DECODE_WEIGHTS,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 "ensureGroupedDecodeCapacity(weights)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_down_descs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_DOWN_DESCS,
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeQuantisedMatrixDesc),
                                 "ensureGroupedDecodeCapacity(down_descs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_swiglu_int8_),
                                 MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,
                                 static_cast<size_t>(num_active) * intermediate * sizeof(int8_t),
                                 "ensureGroupedDecodeCapacity(swiglu_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_swiglu_scales_),
                                 MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
                                 static_cast<size_t>(num_active) * blocks_per_row * sizeof(float),
                                 "ensureGroupedDecodeCapacity(swiglu_scales)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(
                    &d_grouped_down_partials_),
                MoEWorkspaceBuffers::DOWN_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(d_model) * sizeof(float),
                "ensureGroupedDecodeCapacity(ordered_down_partials)"))
        {
            d_grouped_gate_ptrs_ = nullptr;
            d_grouped_up_ptrs_ = nullptr;
            d_grouped_expert_ids_ = nullptr;
            d_grouped_decode_weights_ = nullptr;
            d_grouped_down_descs_ = nullptr;
            d_grouped_swiglu_int8_ = nullptr;
            d_grouped_swiglu_scales_ = nullptr;
            d_grouped_down_partials_ = nullptr;
            grouped_decode_active_cap_ = 0;
            grouped_decode_intermediate_cap_ = 0;
            grouped_decode_d_model_cap_ = 0;
            return false;
        }

        grouped_decode_active_cap_ = num_active;
        grouped_decode_intermediate_cap_ = intermediate;
        grouped_decode_d_model_cap_ = d_model;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedGateUpCapacity(
        int num_active,
        int d_model,
        int intermediate)
    {
        if (num_active <= 0 || d_model <= 0 || intermediate <= 0 ||
            (d_model % 32) != 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedGateUpCapacity"))
            return false;

        const int blocks_per_row = d_model / 32;
        if (grouped_gateup_active_cap_ >= num_active &&
            grouped_gateup_d_model_cap_ >= d_model &&
            grouped_gateup_intermediate_cap_ >= intermediate &&
            d_grouped_gate_output_ptrs_ && d_grouped_up_output_ptrs_ &&
            d_grouped_gateup_expert_ids_ &&
            d_grouped_hidden_int8_ && d_grouped_hidden_scales_ &&
            d_grouped_gateup_gate_partials_ &&
            d_grouped_gateup_up_partials_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_output_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_GATE_OUTPUT_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedGateUpCapacity(gate_output_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_output_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_UP_OUTPUT_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedGateUpCapacity(up_output_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gateup_expert_ids_),
                                 MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 "ensureGroupedGateUpCapacity(expert_ids)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_hidden_int8_),
                                 MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,
                                 static_cast<size_t>(d_model) * sizeof(int8_t),
                                 "ensureGroupedGateUpCapacity(hidden_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_hidden_scales_),
                                 MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
                                 static_cast<size_t>(blocks_per_row) * sizeof(float),
                                 "ensureGroupedGateUpCapacity(hidden_scales)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_grouped_gateup_gate_partials_),
                MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(intermediate) * sizeof(float),
                "ensureGroupedGateUpCapacity(gate_partials)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_grouped_gateup_up_partials_),
                MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(intermediate) * sizeof(float),
                "ensureGroupedGateUpCapacity(up_partials)"))
        {
            d_grouped_gate_output_ptrs_ = nullptr;
            d_grouped_up_output_ptrs_ = nullptr;
            d_grouped_gateup_expert_ids_ = nullptr;
            d_grouped_hidden_int8_ = nullptr;
            d_grouped_hidden_scales_ = nullptr;
            d_grouped_gateup_gate_partials_ = nullptr;
            d_grouped_gateup_up_partials_ = nullptr;
            grouped_gateup_active_cap_ = 0;
            grouped_gateup_d_model_cap_ = 0;
            grouped_gateup_intermediate_cap_ = 0;
            return false;
        }

        grouped_gateup_active_cap_ = num_active;
        grouped_gateup_d_model_cap_ = d_model;
        grouped_gateup_intermediate_cap_ = intermediate;
        return true;
    }

    bool ROCmMoEKernel::isDecodeGraphCaptureActive() const
    {
        const bool capture_active = isGraphCaptureActive() ||
                                    (deviceContext() && deviceContext()->isDeviceGraphCaptureActive());
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        void *stream = getStream();
        if (stream)
        {
            const hipError_t capture_err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture_status);
            if (capture_err != hipSuccess)
                capture_status = hipStreamCaptureStatusNone;
        }
        return capture_active || capture_status == hipStreamCaptureStatusActive;
    }

    bool ROCmMoEKernel::rejectDecodeStagingDuringCapture(const char *context) const
    {
        if (!isDecodeGraphCaptureActive())
            return false;

        LOG_ERROR("[ROCmMoEKernel] " << context
                                    << " requires pointer/metadata staging during graph capture. "
                                       "Run a warmup execution with the same scratch tensors and workspace first.");
        return true;
    }

    bool ROCmMoEKernel::resolveFixedTableGateUpMetadata(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        const int *expert_ids,
        int num_active,
        const int **device_expert_ids)
    {
        if (!expert_ids || !device_expert_ids || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::GateUp,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "fixed-table gate/up metadata"))
        {
            return false;
        }

        auto *slot_ids = static_cast<int *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_GATEUP_EXPERT_IDS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(int)));
        if (!slot_ids)
            return false;

        const auto state = fixed_gateup_metadata_states_.find(workspace_slot);
        if (state != fixed_gateup_metadata_states_.end())
        {
            const bool matches = state->second.num_active == num_active &&
                                 std::equal(
                                     state->second.expert_ids.begin(),
                                     state->second.expert_ids.begin() + num_active,
                                     expert_ids);
            if (!matches)
            {
                LOG_ERROR("[ROCmMoEKernel] Fixed-table gate/up metadata is immutable "
                          "for one captured owner slot; use the device-routed API "
                          "for changing expert ids");
                return false;
            }
            *device_expert_ids = slot_ids;
            return true;
        }
        if (capture_active)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table gate/up metadata was not "
                      "published before graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "resolveFixedTableGateUpMetadata"))
            return false;

        const hipError_t err = hipMemcpyAsync(
            slot_ids,
            expert_ids,
            static_cast<std::size_t>(num_active) * sizeof(int),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table gate/up metadata publication failed: "
                      << hipGetErrorString(err));
            return false;
        }

        FixedGateUpMetadataState published;
        published.num_active = num_active;
        std::copy(expert_ids, expert_ids + num_active, published.expert_ids.begin());
        fixed_gateup_metadata_states_.emplace(workspace_slot, std::move(published));
        *device_expert_ids = slot_ids;
        return true;
    }

    bool ROCmMoEKernel::resolveFixedTableDownMetadata(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        const int *expert_ids,
        const float *expert_weights,
        int num_active,
        const int **device_expert_ids,
        const float **device_expert_weights)
    {
        if (!expert_ids || !expert_weights || !device_expert_ids ||
            !device_expert_weights || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::Down,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "fixed-table down metadata"))
        {
            return false;
        }

        auto *slot_ids = static_cast<int *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_DOWN_EXPERT_IDS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(int)));
        auto *slot_weights = static_cast<float *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_DOWN_WEIGHTS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(float)));
        if (!slot_ids || !slot_weights)
            return false;

        const auto state = fixed_down_metadata_states_.find(workspace_slot);
        if (state != fixed_down_metadata_states_.end())
        {
            const bool matches =
                state->second.num_active == num_active &&
                std::equal(
                    state->second.expert_ids.begin(),
                    state->second.expert_ids.begin() + num_active,
                    expert_ids) &&
                std::equal(
                    state->second.expert_weights.begin(),
                    state->second.expert_weights.begin() + num_active,
                    expert_weights);
            if (!matches)
            {
                LOG_ERROR("[ROCmMoEKernel] Fixed-table down metadata is immutable "
                          "for one captured owner slot; use the device-routed API "
                          "for changing expert ids or weights");
                return false;
            }
            *device_expert_ids = slot_ids;
            *device_expert_weights = slot_weights;
            return true;
        }
        if (capture_active)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table down metadata was not "
                      "published before graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "resolveFixedTableDownMetadata"))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            slot_ids,
            expert_ids,
            static_cast<std::size_t>(num_active) * sizeof(int),
            hipMemcpyHostToDevice,
            stream);
        if (err == hipSuccess)
        {
            err = hipMemcpyAsync(
                slot_weights,
                expert_weights,
                static_cast<std::size_t>(num_active) * sizeof(float),
                hipMemcpyHostToDevice,
                stream);
        }
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table down metadata publication failed: "
                      << hipGetErrorString(err));
            return false;
        }

        FixedDownMetadataState published;
        published.num_active = num_active;
        std::copy(expert_ids, expert_ids + num_active, published.expert_ids.begin());
        std::copy(
            expert_weights,
            expert_weights + num_active,
            published.expert_weights.begin());
        fixed_down_metadata_states_.emplace(workspace_slot, std::move(published));
        *device_expert_ids = slot_ids;
        *device_expert_weights = slot_weights;
        return true;
    }

    bool ROCmMoEKernel::runtimePointerWorkspaceSlot(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        MoERuntimePointerArrayRole role,
        MoERuntimePointerWorkspaceAccess access,
        std::size_t *workspace_slot,
        const char *context)
    {
        if (!workspace_slot)
            return false;

        const auto slot = runtime_pointer_workspace_owners_.resolve(
            workspace_,
            role,
            persistent_descriptor_slot,
            static_cast<std::size_t>(scope),
            access,
            context);
        if (!slot || *slot >= kRuntimePointerArrayWorkspaceEntries)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "runtime pointer table")
                      << " could not resolve an exclusive pointer workspace slot");
            return false;
        }

        *workspace_slot = *slot;
        return true;
    }

    /**
     * @brief Bind and publish the table-owned gate/up pointer arrays.
     * @param persistent_descriptor_slot Descriptor lifetime owning this slot.
     * @param scope Distinct execution use of the descriptor table.
     * @param top_k Number of active pointers to publish.
     * @param gate_ptrs Prepared gate output addresses on this device.
     * @param up_ptrs Prepared up output addresses on this device.
     * @param d_gate_ptrs Receives the immutable device gate pointer-array address.
     * @param d_up_ptrs Receives the immutable device up pointer-array address.
     * @return True when warmup publication or capture-time reuse is complete.
     *
     * Pointer publication owns its arena binding, independently of quantized
     * projection scratch. Capture may only consume an already prepared slot.
     */
    bool ROCmMoEKernel::stageRuntimeGateUpPointerArrays(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        float ***d_gate_ptrs,
        float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::GateUp,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "grouped gate/up"))
            return false;
        if (!setMoEDevice(device_ordinal_, "stageRuntimeGateUpPointerArrays"))
            return false;

        if (!d_grouped_gate_output_ptrs_ || !d_grouped_up_output_ptrs_)
        {
            const size_t bytes = kRuntimePointerArrayWorkspaceEntries *
                                 kRuntimePointerArrayMaxTopK * sizeof(float *);
            if (capture_active ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_output_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_GATE_OUTPUT_PTRS, bytes,
                    "stageRuntimeGateUpPointerArrays(gate)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_output_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_UP_OUTPUT_PTRS, bytes,
                    "stageRuntimeGateUpPointerArrays(up)"))
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Grouped gate/up workspace is not prepared");
                return false;
            }
        }

        float **slot_gate_ptrs =
            d_grouped_gate_output_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;
        float **slot_up_ptrs =
            d_grouped_up_output_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;

        if (capture_active)
        {
            if (!gateup_pointer_slot_ready_[workspace_slot])
                return rejectDecodeStagingDuringCapture("grouped gate/up pointer workspace slot");
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Null gate/up output pointer for slot "
                          << slot << " persistent_descriptor_slot="
                          << persistent_descriptor_slot);
                return false;
            }
        }

        if (!hipMoE_stage_mutable_pointer_arrays(
                slot_gate_ptrs,
                slot_up_ptrs,
                gate_ptrs.data(),
                up_ptrs.data(),
                top_k,
                device_ordinal_,
                getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Pointer staging failed");
            return false;
        }

        gateup_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    /**
     * @brief Bind and publish table-owned inputs without requiring Q8 scratch.
     * @param persistent_descriptor_slot Descriptor lifetime owning this slot.
     * @param scope Distinct execution use of the descriptor table.
     * @param top_k Number of active pointers to publish.
     * @param gate_ptrs Prepared gate inputs on this device.
     * @param up_ptrs Prepared up inputs on this device.
     * @param d_gate_ptrs Receives the immutable device gate pointer-array address.
     * @param d_up_ptrs Receives the immutable device up pointer-array address.
     * @return True when warmup publication or capture-time reuse is complete.
     */
    bool ROCmMoEKernel::stageRuntimeDownPointerArrays(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<const float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<const float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        const float ***d_gate_ptrs,
        const float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::Down,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "grouped down"))
            return false;
        if (!setMoEDevice(device_ordinal_, "stageRuntimeDownPointerArrays"))
            return false;

        if (!d_grouped_gate_ptrs_ || !d_grouped_up_ptrs_)
        {
            const size_t bytes = kRuntimePointerArrayWorkspaceEntries *
                                 kRuntimePointerArrayMaxTopK * sizeof(float *);
            if (capture_active ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_GATE_PTRS, bytes,
                    "stageRuntimeDownPointerArrays(gate)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_UP_PTRS, bytes,
                    "stageRuntimeDownPointerArrays(up)"))
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Grouped down workspace is not prepared");
                return false;
            }
        }

        const float **slot_gate_ptrs =
            d_grouped_gate_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;
        const float **slot_up_ptrs =
            d_grouped_up_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;

        if (capture_active)
        {
            if (!down_pointer_slot_ready_[workspace_slot])
                return rejectDecodeStagingDuringCapture("grouped down pointer workspace slot");
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Null gate/up pointer for slot "
                          << slot << " persistent_descriptor_slot="
                          << persistent_descriptor_slot);
                return false;
            }
        }

        if (!hipMoE_stage_const_pointer_arrays(
                slot_gate_ptrs,
                slot_up_ptrs,
                gate_ptrs.data(),
                up_ptrs.data(),
                top_k,
                device_ordinal_,
                getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Pointer staging failed");
            return false;
        }

        down_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    bool ROCmMoEKernel::routeWithTensorsImpl(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len,
        const char *context)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[" << context << "] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[" << context << "] explicit HIP stream is required");
            return false;
        }
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_idx || !d_wt)
        {
            LOG_ERROR("[" << context << "] output tensors have no device allocation");
            return false;
        }

        if (!routeCore(h, g, gate_type, seq_len, d_model, num_experts, top_k,
                       normalize_weights, d_idx, d_wt,
                       device_effective_seq_len))
            return false;

        host_result.expert_indices.clear();
        host_result.expert_weights.clear();
        host_result.router_logits.clear();

        /*
         * Routing tensors remain device-authoritative in eager execution and
         * graph replay alike.  Snapshot infrastructure performs any requested
         * D2H materialization after publication, at an explicit observation
         * boundary, rather than maintaining a live host mirror here.
         */
        const auto device = DeviceId::rocm(device_ordinal_);
        markDeviceWritten(output_indices, device, getStream());
        markDeviceWritten(output_weights, device, getStream());

        return true;
    }

    bool ROCmMoEKernel::routeWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result)
    {
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    nullptr,
                                    "ROCmMoEKernel::routeWithTensors");
    }

    bool ROCmMoEKernel::routeWithTensorsEffectiveSeqLen(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len)
    {
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    device_effective_seq_len,
                                    "ROCmMoEKernel::routeWithTensorsEffectiveSeqLen");
    }

    bool ROCmMoEKernel::publishOwnedRouteRows(const MoERouterRowPublication &publication)
    {
        return rocm::publishRouterOwnedRows(publication, getStream());
    }

    bool ROCmMoEKernel::routeOwnedRowsWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        const MoERouterRowPacketLayout &layout, int d_model, int num_experts,
        bool normalize_weights, ITensor *packet, std::uint64_t *selected_bytes,
        const std::int32_t *live_rows)
    {
        constexpr const char *context = "ROCmMoEKernel::routeOwnedRowsWithTensors";
        void *stream = getStream();
        const auto device = DeviceId::rocm(device_ordinal_);
        if (!stream || !layout.valid() || d_model <= 0 || num_experts < layout.top_k ||
            !live_rows || !selected_bytes || !hidden || !gate_weights || !packet ||
            hidden->native_type() != TensorType::FP32 ||
            hidden->numel() < std::size_t(layout.capacity) * d_model ||
            gate_weights->numel() < std::size_t(num_experts) * d_model ||
            packet->size_bytes() < layout.packetBytes() ||
            !requireTensorOnDevice(hidden, device, stream, "hidden", context) ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights", context) ||
            !requireOutputOnDevice(packet, device, stream, "router packet", context) ||
            !validateDevicePointerOrLog(selected_bytes, device_ordinal_, "packet extent", context))
            return false;
        const MoERouterOwnedRowOutputs owned{layout.partition,
            static_cast<MoERouterSelectedRoute *>(packet->gpu_data_ptr()), selected_bytes};
        if (!routeCore(static_cast<const float *>(hidden->gpu_data_ptr()), gate_weights->gpu_data_ptr(),
                       gate_weights->native_type(), layout.capacity, d_model, num_experts, layout.top_k,
                       normalize_weights, nullptr, nullptr, live_rows, &owned))
            return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool ROCmMoEKernel::prepareRouteLaunch(
        ITensor *gate_weights,
        const MoERouteLaunchPlan &plan)
    {
        constexpr const char *kContext = "ROCmMoEKernel::prepareRouteLaunch";
        if (!setMoEDevice(device_ordinal_, kContext))
            return false;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[" << kContext << "] explicit HIP stream is required");
            return false;
        }
        if (!plan.valid() ||
            (plan.kind == MoERouteLaunchKind::RuntimeDecode &&
             plan.physical_rows != 1))
        {
            LOG_ERROR("[" << kContext << "] invalid route launch plan"
                      << " kind=" << static_cast<int>(plan.kind)
                      << " rows=" << plan.physical_rows
                      << " d_model=" << plan.d_model
                      << " num_experts=" << plan.num_experts
                      << " top_k=" << plan.top_k);
            return false;
        }
        if (plan.kind == MoERouteLaunchKind::RuntimeDecode)
            prepared_decode_resident_set_policy_ = plan.resident_set_policy;
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(
                gate_weights,
                device,
                stream,
                "gate_weights",
                kContext))
        {
            return false;
        }
        if (gate_weights->native_type() != TensorType::FP32 &&
            gate_weights->native_type() != TensorType::FP16 &&
            gate_weights->native_type() != TensorType::BF16)
        {
            LOG_ERROR("[" << kContext << "] router gate must be FP32, FP16 or BF16, got "
                      << tensorTypeName(gate_weights->native_type()));
            return false;
        }

        const size_t required_gate =
            static_cast<size_t>(plan.num_experts) *
            static_cast<size_t>(plan.d_model);
        if (gate_weights->numel() < required_gate ||
            !gate_weights->gpu_data_ptr())
        {
            LOG_ERROR("[" << kContext << "] router gate capacity or device pointer is invalid"
                      << " elements=" << gate_weights->numel()
                      << "/" << required_gate
                      << " device_ptr=" << gate_weights->gpu_data_ptr());
            return false;
        }

        if (!prepareRequiredRouterQ8Hidden(plan.physical_rows, plan.d_model, plan.top_k))
            return false;

        const size_t logits_count =
            static_cast<size_t>(plan.physical_rows) *
            static_cast<size_t>(plan.num_experts);
        if (!ensureRouteBufferCapacity(logits_count))
        {
            LOG_ERROR("[" << kContext << "] failed to bind persistent route scratch");
            return false;
        }

        if (gate_weights->native_type() != TensorType::FP32)
            return true;

        const auto &rocm_env = debugEnv().rocm;
        if (plan.kind == MoERouteLaunchKind::DecodeEquivalentVerifier &&
            rocm_env.moe_router_kpart_decode)
        {
            LOG_ERROR("[" << kContext << "] decode-equivalent verifier routing "
                      "cannot be prepared while k-part decode routing is selected");
            return false;
        }

        const auto *gate_device =
            static_cast<const float *>(gate_weights->gpu_data_ptr());
        if (rocm_env.moe_router_q8)
        {
            if ((plan.d_model % 32) != 0)
            {
                LOG_ERROR("[" << kContext << "] Q8 routing requires d_model "
                          "to be a multiple of 32, got " << plan.d_model);
                return false;
            }
            if (!ensureRouterQ8HiddenScratchCapacity(
                    plan.physical_rows,
                    plan.d_model))
            {
                LOG_ERROR("[" << kContext << "] failed to bind persistent Q8 hidden-row scratch");
                return false;
            }
            if (!getOrCreateQ8RouterGateCache(
                    gate_device,
                    plan.d_model,
                    plan.num_experts))
            {
                LOG_ERROR("[" << kContext << "] failed to publish immutable Q8 router gate");
                return false;
            }
            return true;
        }

        if (plan.kind == MoERouteLaunchKind::RuntimeDecode &&
            rocm_env.moe_router_kpart_decode)
        {
            if (rocm_env.moe_router_kparts <= 0 ||
                !ensureRouteLogitsPartialsCapacity(
                    static_cast<size_t>(plan.num_experts) *
                    static_cast<size_t>(rocm_env.moe_router_kparts)))
            {
                LOG_ERROR("[" << kContext << "] failed to bind persistent k-part route scratch");
                return false;
            }
            return true;
        }

        return true;
    }

    bool ROCmMoEKernel::routeVerifierRowsDecodeEquivalent(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        const int *device_effective_seq_len,
        DeviceMoELayerRuntime *deferred_selected_route_ledger)
    {
        constexpr const char *kContext = "ROCmMoEKernel::routeVerifierRowsDecodeEquivalent";
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!setMoEDevice(device_ordinal_, kContext))
            return false;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[" << kContext << "] explicit HIP stream is required");
            return false;
        }

        if (seq_len < 1 || d_model <= 0 ||
            num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[" << kContext << "] invalid verifier routing shape seq_len="
                          << seq_len << " d_model=" << d_model
                          << " num_experts=" << num_experts
                          << " top_k=" << top_k);
            return false;
        }

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(hidden, device, stream, "hidden", kContext) ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights", kContext) ||
            !requireOutputOnDevice(output_indices, device, stream, "output_indices", kContext) ||
            !requireOutputOnDevice(output_weights, device, stream, "output_weights", kContext))
        {
            return false;
        }

        if (hidden->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[" << kContext << "] hidden tensor must be FP32, got "
                          << tensorTypeName(hidden->native_type()));
            return false;
        }
        if (output_indices->native_type() != TensorType::FP32 ||
            output_weights->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[" << kContext << "] routing outputs must be FP32 tensors");
            return false;
        }

        const size_t required_hidden =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_gate =
            static_cast<size_t>(num_experts) * static_cast<size_t>(d_model);
        const size_t required_topk =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (hidden->numel() < required_hidden ||
            gate_weights->numel() < required_gate ||
            output_indices->numel() < required_topk ||
            output_weights->numel() < required_topk)
        {
            LOG_ERROR("[" << kContext << "] tensor capacity is too small"
                          << " hidden=" << hidden->numel() << "/" << required_hidden
                          << " gate=" << gate_weights->numel() << "/" << required_gate
                          << " indices=" << output_indices->numel() << "/" << required_topk
                          << " weights=" << output_weights->numel() << "/" << required_topk);
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
        {
            LOG_ERROR("[" << kContext << "] null device pointer hidden="
                          << static_cast<const void *>(d_hidden)
                          << " gate=" << d_gate
                          << " indices=" << static_cast<void *>(d_idx)
                          << " weights=" << static_cast<void *>(d_wt));
            return false;
        }

        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 =
            gate_weights->native_type() == TensorType::FP32;
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[" << kContext << "] Q8 verifier router requires d_model to be a multiple of 32, got "
                          << d_model);
            return false;
        }
        if (rocm_env.moe_router_kpart_decode)
        {
            LOG_ERROR("[" << kContext << "] ROCm k-part decode router is enabled, but verifier-row "
                          "publication does not yet have a k-part row contract");
            return false;
        }

        const size_t logits_count =
            static_cast<size_t>(seq_len) * static_cast<size_t>(num_experts);
        if (!ensureRouteBufferCapacity(logits_count))
        {
            LOG_ERROR("[" << kContext << "] route scratch allocation failed");
            return false;
        }

        bool used_q8_grouped_router = false;
        bool used_fp16_grouped_router = false;
        invalidateRouterQ8HiddenPublication();
        if (!prepareRequiredRouterQ8Hidden(seq_len, d_model, top_k))
            return false;
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                d_hidden, seq_len, d_model, top_k, device_effective_seq_len))
            return false;
        if (q8_router_requested)
        {
            if (!ensureRouterQ8HiddenScratchCapacity(seq_len, d_model))
            {
                LOG_ERROR("[" << kContext << "] Q8 router hidden scratch unavailable for d_model="
                              << d_model);
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(d_gate), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[" << kContext << "] Q8 router gate cache unavailable");
                return false;
            }
            if (!hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
                    d_hidden,
                    d_router_q8_hidden_,
                    d_router_q8_hidden_scales_,
                    q8_gate->d_gate_weights_q8,
                    q8_gate->d_gate_scales,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic)))
            {
                LOG_ERROR("[" << kContext << "] grouped Q8 verifier logits failed");
                return false;
            }
            used_q8_grouped_router = true;
            publishRouterQ8Hidden(
                d_hidden,
                seq_len,
                isDecodeGraphCaptureActive(),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }
        else if (gate_is_fp32)
        {
            if (!launchDecodeEquivalentFP32Rows(d_hidden, static_cast<const float *>(d_gate),
                    d_route_logits_, seq_len, d_model, num_experts, device_ordinal_,
                    getStream(), device_effective_seq_len))
            {
                LOG_ERROR("[" << kContext << "] grouped FP32 verifier logits failed");
                return false;
            }
        }
        else if (gate_weights->native_type() == TensorType::FP16)
        {
            if (!hipMoE_gate_logits_fp16_decode_equivalent_rows(
                    d_hidden,
                    d_gate,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len))
            {
                LOG_ERROR("[" << kContext << "] grouped FP16 verifier logits failed");
                return false;
            }
            used_fp16_grouped_router = true;
        }
        else if (gate_weights->native_type() == TensorType::BF16)
        {
            if (!hipMoE_gate_logits_bf16_decode_equivalent_rows(
                    d_hidden,
                    d_gate,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len))
            {
                LOG_ERROR("[" << kContext << "] grouped BF16 verifier logits failed");
                return false;
            }
        }
        else
        {
            LOG_ERROR("[" << kContext << "] unsupported router gate dtype "
                          << tensorTypeName(gate_weights->native_type()));
            return false;
        }

        if (!hipMoE_softmax_topk_decode_equivalent_rows(
                d_route_logits_,
                d_idx,
                d_wt,
                seq_len,
                num_experts,
                top_k,
                normalize_weights,
                device_ordinal_,
                getStream(),
                device_effective_seq_len,
                static_cast<void *>(deferred_selected_route_ledger)))
        {
            LOG_ERROR("[" << kContext << "] grouped decode-equivalent softmax/top-k failed");
            return false;
        }

        markDeviceWritten(output_indices, device, getStream());
        markDeviceWritten(output_weights, device, getStream());
        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_moe_decode_equivalent_runtime_m_router_calls",
            1.0,
            {},
            {},
            {{"seq_len", std::to_string(seq_len)},
             {"d_model", std::to_string(d_model)},
             {"num_experts", std::to_string(num_experts)},
             {"top_k", std::to_string(top_k)},
             {"row_tile", "16"},
             {"tile_count", std::to_string((seq_len + 15) / 16)},
             {"route", used_q8_grouped_router ? "grouped_decode_equivalent_q8"
                                               : (used_fp16_grouped_router
                                                      ? "grouped_decode_equivalent_fp16"
                                                      : "grouped_decode_equivalent")}});
        return true;
    }

    bool ROCmMoEKernel::decodeRouteSelect(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const int32_t *absolute_position_ids_device,
        RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *active_row_count_device)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !hidden || !gate_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] null runtime/input/gate tensor");
            return false;
        }
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] invalid dimensions d_model=" << d_model
                                                                                       << " num_experts=" << num_experts
                                                                                       << " top_k=" << top_k);
            return false;
        }

        if (paired_router_descriptors_)
        {
            requirePairedRouterDescriptors(runtime_layer, num_experts, top_k);
            if (row_execution_policy != RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
                throw std::logic_error("Paired descriptor router changed its original local-fragment policy");
        }

        // The runtime table is a graph-build invariant validated by the MoE
        // stages before decode capture.  Do not copy it back here: this path is
        // part of vLLM-style speculative decode replay and must stay entirely
        // device-resident once the graph is hot.

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts)))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] route logits scratch allocation failed");
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] legacy outputs requested without tensors");
                return false;
            }

            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
            if (!legacy_indices || !legacy_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] legacy output tensors have no device allocation");
                return false;
            }
        }

        bool logits_ready = false;
        bool runtime_ready = false;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        invalidateRouterQ8HiddenPublication();
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
        // M1 restored-prefix verification uses runtime routing but a grouped
        // shared consumer. Floating router math must still publish that
        // consumer's original-basis Q8 rows on this exact captured stream.
        // The cold demand contract makes ordinary no-consumer decode a no-op.
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                h, /*rows=*/1, d_model, top_k, active_row_count_device))
            return false;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!ensureRouterQ8HiddenScratchCapacity(/*rows=*/1, d_model))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router hidden scratch unavailable for d_model="
                          << d_model);
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(g), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router gate cache unavailable");
                return false;
            }
            void *stream = requireStream("ROCmMoEKernel::decodeRouteSelect Q8 router");
            if (!stream)
                return false;
            logits_ready = hipMoE_gate_logits_single_token_q8_weights(
                h, d_router_q8_hidden_, d_router_q8_hidden_scales_,
                q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales,
                d_route_logits_, d_model, num_experts,
                device_ordinal_, stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router logits kernel failed");
                return false;
            }
            publishRouterQ8Hidden(
                h,
                /*rows=*/1,
                isDecodeGraphCaptureActive(),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }

        if (!logits_ready && gate_is_fp32 && rocm_env.moe_router_kpart_decode)
        {
            const int k_partitions = rocm_env.moe_router_kparts;
            const size_t partial_count = static_cast<size_t>(num_experts) * static_cast<size_t>(k_partitions);
            if (!ensureRouteLogitsPartialsCapacity(partial_count))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] K-part router was requested but scratch allocation failed");
                return false;
            }
            const bool partials_ready = hipMoE_gate_logits_single_token_kpart_partials(
                h, static_cast<const float *>(g), d_route_logits_partials_,
                d_model, num_experts, k_partitions,
                device_ordinal_, getStream());
            if (!partials_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] K-part router logits kernel failed");
                return false;
            }
            runtime_ready = hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
                d_route_logits_partials_,
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                k_partitions,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                nullptr,
                -1,
                1u,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] fused K-part router runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !logits_ready &&
            !launchDecodeGateLogitsForGateType(
                h, g, gate_type, d_route_logits_,
                d_model, num_experts,
                device_ordinal_, getStream(),
                "ROCmMoEKernel::decodeRouteSelect"))
        {
            return false;
        }

        if (paired_router_descriptors_)
        {
            if (runtime_ready)
                throw std::logic_error("Paired descriptor router selected a different prepared producer");
            runtime_ready = hipMoE_softmax_topk_decode_selected_descriptors(
                d_route_logits_, runtime_layer, legacy_indices, legacy_weights,
                num_experts, top_k, normalize_weights, write_legacy_outputs,
                update_runtime_histogram, row_execution_policy,
                paired_router_descriptors_->outputs, device_ordinal_, getStream(), active_row_count_device);
            if (!runtime_ready)
                throw std::runtime_error("Paired ROCm router/descriptor publication failed");
            PerfStatsCollector::addCounter("moe", "router_selected_descriptors_enqueued", 1.0,
                "graph_capture", DeviceId::rocm(device_ordinal_).toString(),
                {{"descriptor_source", "original_intermediate_shard"}});
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk &&
            num_experts <= static_cast<int>(kDeviceMoEMaxExperts) &&
            row_execution_policy == RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            prepared_decode_resident_set_policy_ ==
                RoutedExpertResidentSetPolicy::UniqueOwner)
        {
            runtime_ready =
                hipMoE_softmax_topk_decode_unique_owner_wave64(
                    d_route_logits_,
                    static_cast<void *>(runtime_layer),
                    legacy_indices,
                    legacy_weights,
                    num_experts,
                    top_k,
                    normalize_weights,
                    write_legacy_outputs,
                    update_runtime_histogram,
                    device_ordinal_,
                    getStream(),
                    active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] unique-owner "
                          "wave64 decode router failed");
                return false;
            }
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= static_cast<int>(kDeviceMoEMaxExperts))
        {
            runtime_ready = hipMoE_softmax_topk_decode_runtime_wave64(
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                nullptr,
                -1,
                1u,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] wave64 decode softmax/top-k runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !hipMoE_softmax_topk_decode_runtime(
                                  d_route_logits_,
                                  static_cast<void *>(runtime_layer),
                                  legacy_indices,
                                  legacy_weights,
                                  num_experts,
                                  top_k,
                                  normalize_weights,
                                  write_legacy_outputs,
                                  update_runtime_histogram,
                                  row_execution_policy,
                                  nullptr,
                                  nullptr,
                                  0u,
                                  nullptr,
                                  nullptr,
                                  0u,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  -1,
                                  1u,
                                  absolute_position_ids_device,
                                  device_ordinal_,
                                  getStream(),
                                  active_row_count_device))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] decode softmax/top-k runtime kernel failed");
            return false;
        }

        if (write_legacy_outputs)
        {
            const DeviceId device = DeviceId::rocm(device_ordinal_);
            markDeviceWritten(output_indices, device, getStream());
            markDeviceWritten(output_weights, device, getStream());
        }

        return true;
    }

    bool ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply(
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const DeviceMoERebalancePlanEntry *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *rebalance_command_header,
        const DeviceMoEExpertDirectoryEntry *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        const DeviceMoERebalanceConfig &rebalance_config,
        DeviceMoERebalanceApplyStatus *rebalance_apply_status,
        DeviceMoERebalanceGraphControllerState *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count,
        const int32_t *absolute_position_ids_device,
        RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *active_row_count_device)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layers || !runtime_layer || !hidden || !gate_weights ||
            !rebalance_plan_entries || rebalance_plan_capacity == 0 ||
            !rebalance_command_header || !rebalance_apply_status ||
            !rebalance_controller_state ||
            !validateDeviceMoERebalanceConfig(rebalance_config))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] invalid runtime or rebalance binding");
            return false;
        }
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] invalid dimensions d_model=" << d_model
                                                                                                               << " num_experts=" << num_experts
                                                                                                               << " top_k=" << top_k);
            return false;
        }

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(hidden, device, getStream(), "hidden", "decodeRouteSelectWithReadyRebalanceApply") ||
            !requireTensorOnDevice(gate_weights, device, getStream(), "gate_weights", "decodeRouteSelectWithReadyRebalanceApply"))
        {
            return false;
        }

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts)))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] route logits scratch allocation failed");
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights ||
                !requireOutputOnDevice(output_indices, device, getStream(), "output_indices", "decodeRouteSelectWithReadyRebalanceApply") ||
                !requireOutputOnDevice(output_weights, device, getStream(), "output_weights", "decodeRouteSelectWithReadyRebalanceApply"))
            {
                return false;
            }

            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
            if (!legacy_indices || !legacy_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] legacy output tensors have no device allocation");
                return false;
            }
        }

        bool logits_ready = false;
        bool runtime_ready = false;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        invalidateRouterQ8HiddenPublication();
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
        // M1 restored-prefix verification uses runtime routing but a grouped
        // shared consumer. Floating router math must still publish that
        // consumer's original-basis Q8 rows on this exact captured stream.
        // The cold demand contract makes ordinary no-consumer decode a no-op.
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                h, /*rows=*/1, d_model, top_k, active_row_count_device))
            return false;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!ensureRouterQ8HiddenScratchCapacity(/*rows=*/1, d_model))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router hidden scratch unavailable for d_model="
                          << d_model);
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(g), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router gate cache unavailable");
                return false;
            }
            void *stream = requireStream("ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply Q8 router");
            if (!stream)
                return false;
            logits_ready = hipMoE_gate_logits_single_token_q8_weights(
                h, d_router_q8_hidden_, d_router_q8_hidden_scales_,
                q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales,
                d_route_logits_, d_model, num_experts,
                device_ordinal_, stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router logits kernel failed");
                return false;
            }
            publishRouterQ8Hidden(
                h,
                /*rows=*/1,
                isDecodeGraphCaptureActive(),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }

        if (!logits_ready && gate_is_fp32 && rocm_env.moe_router_kpart_decode)
        {
            const int k_partitions = rocm_env.moe_router_kparts;
            const size_t partial_count = static_cast<size_t>(num_experts) * static_cast<size_t>(k_partitions);
            if (!ensureRouteLogitsPartialsCapacity(partial_count))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] K-part router was requested but scratch allocation failed");
                return false;
            }
            const bool partials_ready = hipMoE_gate_logits_single_token_kpart_partials(
                h, static_cast<const float *>(g), d_route_logits_partials_,
                d_model, num_experts, k_partitions,
                device_ordinal_, getStream());
            if (!partials_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] K-part router logits kernel failed");
                return false;
            }
            runtime_ready = hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
                d_route_logits_partials_,
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                k_partitions,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                runtime_layers,
                rebalance_plan_entries,
                rebalance_plan_capacity,
                rebalance_command_header,
                rebalance_local_transfer_slots,
                rebalance_local_transfer_slot_count,
                &rebalance_config,
                rebalance_apply_status,
                rebalance_controller_state,
                rebalance_target_layer,
                rebalance_command_buffer_count,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] fused K-part router runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !logits_ready &&
            !launchDecodeGateLogitsForGateType(
                h, g, gate_type, d_route_logits_,
                d_model, num_experts,
                device_ordinal_, getStream(),
                "ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply"))
        {
            return false;
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= static_cast<int>(kDeviceMoEMaxExperts))
        {
            runtime_ready = hipMoE_softmax_topk_decode_runtime_wave64(
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                runtime_layers,
                rebalance_plan_entries,
                rebalance_plan_capacity,
                rebalance_command_header,
                rebalance_local_transfer_slots,
                rebalance_local_transfer_slot_count,
                &rebalance_config,
                rebalance_apply_status,
                rebalance_controller_state,
                rebalance_target_layer,
                rebalance_command_buffer_count,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] wave64 decode softmax/top-k runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !hipMoE_softmax_topk_decode_runtime(
                                  d_route_logits_,
                                  static_cast<void *>(runtime_layer),
                                  legacy_indices,
                                  legacy_weights,
                                  num_experts,
                                  top_k,
                                  normalize_weights,
                                  write_legacy_outputs,
                                  update_runtime_histogram,
                                  row_execution_policy,
                                  runtime_layers,
                                  rebalance_plan_entries,
                                  rebalance_plan_capacity,
                                  rebalance_command_header,
                                  rebalance_local_transfer_slots,
                                  rebalance_local_transfer_slot_count,
                                  &rebalance_config,
                                  rebalance_apply_status,
                                  rebalance_controller_state,
                                  rebalance_target_layer,
                                  rebalance_command_buffer_count,
                                  absolute_position_ids_device,
                                  device_ordinal_,
                                  getStream(),
                                  active_row_count_device))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] decode softmax/top-k runtime kernel failed");
            return false;
        }

        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, getStream());
            markDeviceWritten(output_weights, device, getStream());
        }

        return true;
    }

    bool ROCmMoEKernel::runDeviceRebalanceController(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const uint64_t *gathered_histograms,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceWaveState *wave_state,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoELLEPLayerPlanScratch *llep_layer_plans,
        DeviceMoEPlacementBank *placement_plan_scratch)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] invalid device rebalance config");
            return false;
        }
        const bool deferred_planning = hasDeviceMoERebalanceFlag(
            config.flags,
            DeviceMoERebalanceFlags::DeferRuntimeApply);
        if (!runtime_layers || !gathered_histograms || !status ||
            !controller_state ||
            (deferred_planning && !placement_plan_scratch))
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] runtime layers, gathered histograms, status, controller state, and placement planner scratch must be non-null");
            return false;
        }
        if (config.routed_assignment_policy ==
                kDeviceMoERebalanceAssignmentLeastLoadedResident &&
            !llep_layer_plans)
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] LLEP requires graph-lifetime parallel planner scratch");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "runDeviceRebalanceController");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "runDeviceRebalanceController"))
            return false;

        return hipMoE_device_rebalance_controller(
            runtime_layers,
            reinterpret_cast<const unsigned long long *>(gathered_histograms),
            status,
            &config,
            plan_entries,
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            command_header,
            wave_state,
            controller_state,
            llep_layer_plans,
            placement_plan_scratch,
            command_buffer_count,
            local_transfer_slots,
            local_transfer_slot_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceHistograms(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        uint64_t *local_histograms,
        const DeviceMoERebalanceConfig &config,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count,
        uint32_t histogram_source_mask,
        uint64_t *previous_activation_counts)
    {
        if (!validateDeviceMoERebalanceConfig(config) ||
            !moe_runtime_abi::validHistogramSourceMask(
                histogram_source_mask))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceHistograms] invalid device rebalance config or histogram source mask");
            return false;
        }
        if (!runtime_layers || !local_histograms)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceHistograms] runtime layers and local histograms must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceHistograms");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceHistograms"))
            return false;

        return hipMoE_pack_rebalance_histograms(
            runtime_layers,
            reinterpret_cast<unsigned long long *>(local_histograms),
            &config,
            wave_state,
            controller_state,
            command_buffer_count,
            histogram_source_mask,
            reinterpret_cast<unsigned long long *>(
                previous_activation_counts),
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceDirectory(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoEExpertDirectoryEntry *local_directory,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceDirectory] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !local_directory)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceDirectory] runtime layers and local directory must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceDirectory");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceDirectory"))
            return false;

        return hipMoE_pack_rebalance_directory(
            runtime_layers,
            local_directory,
            &config,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceSourceDescriptors(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceSourceDescriptors] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || !command_headers ||
            !local_source_descriptors)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceSourceDescriptors] runtime, projected plans, headers, and source descriptor output must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceSourceDescriptors");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceSourceDescriptors"))
            return false;

        return hipMoE_pack_rebalance_source_descriptors(
            runtime_layers,
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            &config,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::projectDeviceRebalanceDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *local_command_headers,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceWaveState *gathered_wave_states,
        DeviceMoERebalanceWaveState *local_wave_states,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::projectDeviceRebalanceDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_command_headers ||
            !runtime_layers || !local_transfer_slots ||
            !transfer_slot_claim_index ||
            plan_capacity == 0 || local_transfer_slot_count == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::projectDeviceRebalanceDomainCommands] gathered/local commands, runtime layers, and transfer directory must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "projectDeviceRebalanceDomainCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "projectDeviceRebalanceDomainCommands"))
            return false;

        return hipMoE_project_rebalance_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_command_headers,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            gathered_wave_states,
            local_wave_states,
            runtime_layers,
            local_transfer_slots,
            local_transfer_slot_count,
            transfer_slot_claim_index,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::projectPrefillLeastLoadedDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        uint32_t *local_plan_count,
        DeviceMoERebalanceCommandBufferHeader *local_command_header,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::projectPrefillLeastLoadedDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_plan_count || !local_command_header ||
            !status || !runtime_layers || !local_transfer_slots ||
            !transfer_slot_claim_index ||
            plan_capacity == 0 || payload_slot_capacity == 0 ||
            local_transfer_slot_count == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::projectPrefillLeastLoadedDomainCommands] gathered commands, local command output, runtime layers, transfer directory, status, and payload capacity are required");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "projectPrefillLeastLoadedDomainCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "projectPrefillLeastLoadedDomainCommands"))
            return false;

        return hipMoE_project_prefill_llep_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_plan_count,
            local_command_header,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            runtime_layers,
            local_transfer_slots,
            local_transfer_slot_count,
            transfer_slot_claim_index,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *mirrored_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *mirrored_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !mirrored_plan_entries ||
            !mirrored_command_headers || !status || plan_capacity == 0 ||
            payload_slot_capacity == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] runtime, mirrored command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] layer index out of range"
                      << " layer=" << layer_idx
                      << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = explicitMoELaunchStream(
            launch,
            "materializePrefillLeastLoadedMirroredDomainCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(
                device_ordinal_,
                "materializePrefillLeastLoadedMirroredDomainCommands"))
        {
            return false;
        }

        return hipMoE_materialize_prefill_llep_mirrored_domain_commands(
            runtime_layer,
            mirrored_plan_entries,
            mirrored_command_headers,
            plan_capacity,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !plan_entries || !plan_count || !command_header || !status ||
            plan_capacity == 0 || payload_slot_capacity == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands] runtime, command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands] layer index out of range"
                      << " layer=" << layer_idx << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "materializePrefillLeastLoadedTransferCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "materializePrefillLeastLoadedTransferCommands"))
            return false;

        return hipMoE_materialize_prefill_llep_transfer_commands(
            runtime_layer,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceCompactPayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCompactPayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCompactPayloads] plans, headers, source descriptors, payload, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceCompactPayloads");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCompactPayloads"))
            return false;

        return hipMoE_pack_rebalance_compact_payloads(
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::applyDeviceRebalanceArrivals(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::applyDeviceRebalanceArrivals] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !local_transfer_slots || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::applyDeviceRebalanceArrivals] runtime, plan, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "applyDeviceRebalanceArrivals");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "applyDeviceRebalanceArrivals"))
            return false;

        return hipMoE_apply_rebalance_arrivals(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            target_layer,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceCollectivePayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_directory,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCollectivePayloads] gathered plans, headers, local directory, payload, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceCollectivePayloads");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCollectivePayloads"))
            return false;

        return hipMoE_pack_rebalance_collective_payloads(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_directory,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::unpackDeviceRebalanceCollectivePayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const uint8_t *gathered_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::unpackDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[ROCmMoEKernel::unpackDeviceRebalanceCollectivePayloads] plan, gathered payload, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "unpackDeviceRebalanceCollectivePayloads");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "unpackDeviceRebalanceCollectivePayloads"))
            return false;

        return hipMoE_unpack_rebalance_collective_payloads(
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            gathered_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::initializeDeviceRebalanceGraphController(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::initializeDeviceRebalanceGraphController] invalid device rebalance config");
            return false;
        }
        if (!controller_state)
        {
            LOG_ERROR("[ROCmMoEKernel::initializeDeviceRebalanceGraphController] controller state must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "initializeDeviceRebalanceGraphController");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "initializeDeviceRebalanceGraphController"))
            return false;

        return hipMoE_init_rebalance_graph_controller_state(
            controller_state,
            &config,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::resetDeviceRebalanceGraphTransactionForRequest(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_headers,
        DeviceMoERebalanceWaveState *wave_states,
        uint32_t *plan_counts,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::resetDeviceRebalanceGraphTransactionForRequest] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_headers || !wave_states ||
            !plan_counts || command_buffer_count == 0u ||
            command_buffer_count > 2u)
        {
            LOG_ERROR("[ROCmMoEKernel::resetDeviceRebalanceGraphTransactionForRequest] transaction buffers must be non-null and command-buffer count must be one or two");
            return false;
        }
        void *stream = explicitMoELaunchStream(
            launch,
            "resetDeviceRebalanceGraphTransactionForRequest");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(
                device_ordinal_,
                "resetDeviceRebalanceGraphTransactionForRequest"))
        {
            return false;
        }

        return hipMoE_reset_rebalance_graph_transaction_for_request(
            controller_state,
            command_headers,
            wave_states,
            plan_counts,
            command_buffer_count,
            &config,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::publishDeviceRebalanceTransferComplete(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceApplyStatus *copy_status,
        const DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t plan_capacity,
        const DeviceMoERebalanceApplyStatus *gathered_copy_status,
        const DeviceMoERebalanceConfig &config,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::publishDeviceRebalanceTransferComplete] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_header || !wave_state || !copy_status ||
            !plan_entries || plan_capacity == 0u || !gathered_copy_status)
        {
            LOG_ERROR("[ROCmMoEKernel::publishDeviceRebalanceTransferComplete] controller state, command header, wave state, copy status, plan entries, and gathered copy status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "publishDeviceRebalanceTransferComplete");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "publishDeviceRebalanceTransferComplete"))
            return false;

        return hipMoE_publish_rebalance_transfer_complete(
            controller_state,
            command_header,
            wave_state,
            copy_status,
            plan_entries,
            plan_capacity,
            gathered_copy_status,
            &config,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::applyReadyDeviceRebalanceWave(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer,
        uint32_t command_buffer_count,
        const DeviceMoEOverlayEpochStatus *overlay_reservation_status)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::applyReadyDeviceRebalanceWave] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !status || !controller_state)
        {
            LOG_ERROR("[ROCmMoEKernel::applyReadyDeviceRebalanceWave] runtime, plan, status, controller state, and command ABI must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "applyReadyDeviceRebalanceWave");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "applyReadyDeviceRebalanceWave"))
            return false;

        return hipMoE_apply_ready_rebalance_wave(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            target_layer,
            command_buffer_count,
            overlay_reservation_status,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::finalizeMoEOverlayRebalancePublication(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer_count,
        std::uint32_t expert_count,
        DeviceMoEOverlayEpochControl *control,
        std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *reservation_and_publication_status,
        const DeviceMoERebalanceApplyStatus *apply_status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_headers,
        std::uint32_t command_buffer_count,
        const DeviceMoERebalancePlanEntry *plan_entries,
        std::uint32_t plan_capacity,
        std::uint64_t estimated_expert_bytes,
        DeviceMoERebalanceMovementJournalView journal)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finalizeMoEOverlayRebalancePublication");
        if (!runtime_layers || layer_count == 0u || expert_count == 0u ||
            !control || !candidate_epoch ||
            !reservation_and_publication_status || !apply_status || !stream ||
            !controller_state || !command_headers || command_buffer_count == 0u ||
            command_buffer_count > 2u || !plan_entries || plan_capacity == 0u ||
            !journal.state || !journal.waves || !journal.edges ||
            journal.wave_capacity == 0u || journal.edge_capacity == 0u)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::finalizeMoEOverlayRebalancePublication] "
                "runtime family, epoch state, apply status, and explicit stream are required");
            return false;
        }
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(
                device_ordinal_,
                "finalizeMoEOverlayRebalancePublication"))
        {
            return false;
        }
        return hipMoE_finalize_overlay_rebalance_publication(
            runtime_layers,
            layer_count,
            expert_count,
            control,
            candidate_epoch,
            reservation_and_publication_status,
            apply_status,
            controller_state,
            command_headers,
            command_buffer_count,
            plan_entries,
            plan_capacity,
            estimated_expert_bytes,
            journal,
            device_ordinal_,
            stream);
    }

    void ROCmMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        if (!setMoEDevice(device_ordinal_, "zeroBuffer"))
            return;

        void *ptr = tensor->gpu_data_ptr();
        if (!ptr)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] tensor has no device allocation");
            return;
        }
        hipError_t err = hipMemsetAsync(ptr, 0, bytes, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] hipMemsetAsync failed: " << hipGetErrorString(err));
            return;
        }
        markDeviceWritten(
            tensor,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    void ROCmMoEKernel::gatherTokenBatchFromTensors(
        ITensor *hidden, ITensor *batch_buffer,
        const int *host_token_indices, int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatchFromTensors"))
            return;

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        float *b = static_cast<float *>(batch_buffer->gpu_data_ptr());

        if (!h || !b)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] null device pointer");
            return;
        }
        // Upload host token indices to device staging buffer
        if (!ensureStagingCapacity(num_tokens))
            return;

        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }

        gatherTokenBatch(h, b, d_staging_indices_, num_tokens, d_model);
        markDeviceWritten(
            batch_buffer,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    bool ROCmMoEKernel::copyTokenRowFromTensor(
        ITensor *source, ITensor *row_buffer,
        int row_index, int row_width)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (row_index < 0 || row_width <= 0)
            return false;
        if (!setMoEDevice(device_ordinal_, "copyTokenRowFromTensor"))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(source, device, stream, "source", "copyTokenRowFromTensor") ||
            !requireOutputOnDevice(row_buffer, device, stream, "row_buffer", "copyTokenRowFromTensor"))
        {
            return false;
        }

        const auto *src = static_cast<const float *>(source->gpu_data_ptr());
        auto *dst = static_cast<float *>(row_buffer->gpu_data_ptr());
        if (!hipMoE_copy_token_row(src, dst, row_index, row_width, device_ordinal_, stream))
            return false;

        markDeviceWritten(row_buffer, device, stream);
        return true;
    }

    void ROCmMoEKernel::scatterAddWeightedFromTensors(
        ITensor *output, ITensor *expert_output,
        const int *host_token_indices, const float *host_weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterAddWeightedFromTensors"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *e = static_cast<const float *>(expert_output->gpu_data_ptr());

        if (!o || !e)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] null device pointer");
            return;
        }
        // Upload host indices + weights to device staging
        if (!ensureStagingCapacity(num_tokens))
            return;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }
        err = hipMemcpyAsync(
            d_staging_weights_, host_weights,
            num_tokens * sizeof(float), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D weights failed: "
                      << hipGetErrorString(err));
            return;
        }

        scatterAddWeighted(o, e, d_staging_indices_, d_staging_weights_,
                           num_tokens, d_model);
        markDeviceWritten(
            output,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    bool ROCmMoEKernel::writeTokenRowToTensor(
        ITensor *destination, ITensor *row_buffer,
        int row_index, int row_width)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (row_index < 0 || row_width <= 0)
            return false;
        if (!setMoEDevice(device_ordinal_, "writeTokenRowToTensor"))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireOutputOnDevice(destination, device, stream, "destination", "writeTokenRowToTensor") ||
            !requireTensorOnDevice(row_buffer, device, stream, "row_buffer", "writeTokenRowToTensor"))
        {
            return false;
        }

        auto *dst = static_cast<float *>(destination->gpu_data_ptr());
        const auto *src = static_cast<const float *>(row_buffer->gpu_data_ptr());
        if (!hipMoE_write_token_row(dst, src, row_index, row_width, device_ordinal_, stream))
            return false;

        markDeviceWritten(destination, device, stream);
        return true;
    }

    void ROCmMoEKernel::sharedExpertGateFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateFromTensors"))
            return;

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateFromTensors") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateFromTensors") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateFromTensors"))
            return;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());

        if (!in || !gi || !so)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensors] null device pointer");
            return;
        }

        sharedExpertGate(in, gi, so, seq_len, d_model);
        markDeviceWritten(shared_output, device, stream);
    }

    bool ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateFromTensorsEffectiveSeqLen"))
            return false;
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateFromTensorsEffectiveSeqLen"))
            return false;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());
        if (!in || !gi || !so)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (seq_len == 1)
        {
            // Keep the existing fused M1 reduction tree; its optional count
            // suppresses dot work and overwrites the shared row when empty.
            if (!hipMoE_shared_expert_gate_decode_fused(in, gi, so, d_model,
                    device_ordinal_, stream, device_effective_seq_len))
                return false;
            markDeviceWritten(shared_output, device, stream);
            return true;
        }

        if (!ensureSharedGateScratchCapacity(seq_len))
            return false;
        if (!hipMoE_shared_expert_gate_effective_seq_len(
                in, gi, so, d_shared_gate_scratch_,
                seq_len, d_model, device_effective_seq_len,
                device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        return true;
    }

    void ROCmMoEKernel::sharedExpertGateAddFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateAddFromTensors"))
            return;

        if (seq_len <= 0)
            return;

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateAddFromTensors") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateAddFromTensors") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateAddFromTensors") ||
            !requireTensorOnDevice(routed_residual, device, stream, "routed_residual", "sharedExpertGateAddFromTensors") ||
            !requireOutputOnDevice(combined_output, device, stream, "combined_output", "sharedExpertGateAddFromTensors"))
            return;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const float *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        float *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensors] null device pointer");
            return;
        }

        if (!hipMoE_shared_expert_gate_add(in, gi, so, rr, co,
                                           seq_len, d_model,
                                           device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensors] fused gate-add kernel launch failed");
            return;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
    }

    bool ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateAddFromTensorsEffectiveSeqLen"))
            return false;
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = getStream();
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(input, device, stream, "input", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(gate_inp, device, stream, "gate_inp", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(shared_output, device, stream, "shared_output", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireTensorOnDevice(routed_residual, device, stream, "routed_residual", "sharedExpertGateAddFromTensorsEffectiveSeqLen") ||
            !requireOutputOnDevice(combined_output, device, stream, "combined_output", "sharedExpertGateAddFromTensorsEffectiveSeqLen"))
            return false;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const float *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        float *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (!hipMoE_shared_expert_gate_add_effective_seq_len(
                in, gi, so, rr, co, seq_len, d_model,
                device_effective_seq_len, device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] fused gate-add kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
        return true;
    }

    void ROCmMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        if (!setMoEDevice(device_ordinal_, "swiGLUFromTensors"))
            return;

        float *g = static_cast<float *>(gate->gpu_data_ptr());
        float *u = static_cast<float *>(up->gpu_data_ptr());

        if (!g || !u)
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLUFromTensors] null device pointer");
            return;
        }

        swiGLU(g, u, count);
        markDeviceWritten(
            gate,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    void ROCmMoEKernel::weightedAddFromTensors(
        ITensor *output, ITensor *input, float weight, int count)
    {
        if (!setMoEDevice(device_ordinal_, "weightedAddFromTensors"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *in = static_cast<const float *>(input->gpu_data_ptr());

        if (!o || !in)
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAddFromTensors] null device pointer");
            return;
        }

        weightedAdd(o, in, weight, count);
        markDeviceWritten(
            output,
            DeviceId::rocm(device_ordinal_),
            getStream());
    }

    int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable(
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != DeviceMoEWeightFormat::NativeQuantised)) return -1;

        if (!down_descs || num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (intermediate % 32) != 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped down descriptor table"))
        {
            return -1;
        }

        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertDownDescriptorTable"))
            return -1;

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeQuantisedExecutionFormatEnvelope format_envelope = original_family
            ? original_family->nativeDownEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (original_family && !original_family->matchesNativeDown(desc)) return -1;

            if (!desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }

            if (!validateGroupedDownDesc(desc, d_model, intermediate))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }
            if (!original_family && !addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable))
            {
                return -1;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return -1;
        uint8_t codebook_id = kROCmMoEMixedCodebookSentinel;
        for (uint8_t codebook = 0u; codebook < 32u; ++codebook)
        {
            const uint32_t bit = nativeQuantisedCodebookMaskBit(codebook);
            if ((codebook_mask & bit) == 0u)
                continue;
            if (groupedPrefillCodebookBit(codebook) == 0u)
                return -1;
            if ((codebook_mask & (codebook_mask - 1u)) == 0u)
                codebook_id = codebook;
        }

        // The IQ4 LUT family may contain source-tuple24 and ordinary4.
        // One template24 launch owns the family; each descriptor retains its
        // independent factor words and the original physical mask is unchanged.
        const int iq4_family_codebook = singleCodebookFromMask(codebook_mask);
        if (iq4_family_codebook >= 0)
            codebook_id = static_cast<uint8_t>(iq4_family_codebook);

        // Publish immutable IQ tables before accepting or reusing a retained
        // descriptor table. A cache hit must never bypass device prerequisites.
        if (groupedPrefillMaskNeedsIQTables(codebook_mask) &&
            !rocm::ensureIQGridTablesInitialized(device_ordinal_))
        {
            LOG_ERROR(
                "[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] "
                "IQ grid table initialization failed on ROCm device "
                << device_ordinal_);
            return -1;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) ||
                existing.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
                !existing.device_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.policy_codebook_mask != policy_codebook_mask ||
                existing.descriptor_source != descriptor_source ||
                existing.host_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_descs.data(), down_descs, desc_bytes) == 0)
                return static_cast<int>(index);
        }

        GroupedDownDescriptorTable table;
        table.host_descs.assign(down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;
        table.policy_codebook_mask = policy_codebook_mask;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.weight_format = DeviceMoEWeightFormat::NativeQuantised;
        table.valid = true;
        if (!publishGroupedDownDescriptorTable(
                table,
                "upload grouped expert down descriptor table"))
        {
            return -1;
        }
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    /** @copydoc IMoEKernel::prepareOwnerGateUpCaptureBinding */
    std::uint64_t ROCmMoEKernel::prepareOwnerGateUpCaptureBinding(int table_id, int rows,
                                                                    std::uint64_t expected_intent)
    {
        if (isGraphCaptureActive()) throw std::logic_error("owner gate/up capability must be bound before native capture");
        if (table_id < 0 || table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) || rows <= 0)
            throw std::invalid_argument("owner gate/up preparation requires its original table and physical rows");
        auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.deviceReady()) throw std::logic_error("owner gate/up table is not prepared");
        const bool selected = rows == 1 && table.weight_format == DeviceMoEWeightFormat::NativeQuantised &&
            (table.descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable);
        if (!selected)
        {
            if (expected_intent) throw std::logic_error("owner gate/up intent names an unsupported physical operation");
            return 0;
        }
        if (!expected_intent)
        {
            if (table.owner_m1_admission || table.owner_m1_intent)
                throw std::logic_error("retained owner capability cannot be replaced by an unbound component policy");
            return 0; // Original explicitly unbound component policy; model declarations validate their own nonzero intent.
        }
        const auto plan = rocm::planMoERouteOwnedGateUp(rocm::MoERouteOwnedGateUpGeometry::TN64,
            {table.codebook_mask, table.policy_codebook_mask}, device_ordinal_,
            rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes,
            {.output_columns = table.intermediate, .reduction_columns = table.d_model});
        if (plan.identity() != expected_intent)
            throw std::logic_error("owner gate/up table does not match its predeclared immutable intent");
        if (table.owner_m1_admission)
        {
            if (table.owner_m1_intent != expected_intent)
                throw std::logic_error("owner gate/up capability cannot change behind a retained graph");
            return table.owner_m1_intent;
        }
        auto admission = rocm::admitMoERouteOwnedGateUpGeometry(plan);
        // Publish only the complete native capability. The original table and
        // graph retirement own this host metadata; no memory reserve is added.
        table.owner_m1_admission = std::move(admission);
        table.owner_m1_intent = expected_intent;
        return expected_intent;
    }

    /** @copydoc IMoEKernel::ownerGateUpCaptureBindingSignature */
    std::uint64_t ROCmMoEKernel::ownerGateUpCaptureBindingSignature(int table_id, int rows) const
    {
        if (table_id < 0 || table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) || rows <= 0)
            throw std::logic_error("owner gate/up capture binding lost its descriptor table");
        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.deviceReady()) throw std::logic_error("owner gate/up capture binding lost its prepared storage");
        const bool selected = rows == 1 && table.weight_format == DeviceMoEWeightFormat::NativeQuantised &&
            (table.descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable);
        if (!selected) return 0;
        if (table.owner_m1_admission.has_value() != (table.owner_m1_intent != 0))
            throw std::logic_error("owner gate/up table lost its complete immutable capability");
        return table.owner_m1_intent;
    }

    /** @copydoc ROCmMoEKernel::ownerGateUpAdmissionForCall */
    const rocm::MoERouteOwnedGateUpAdmission *ROCmMoEKernel::ownerGateUpAdmissionForCall(
        int table_id, int rows, const MoEPrefillProjectionExecution &execution,
        rocm::MoERouteOwnedProjectionShape shape) const
    {
        if (!execution.executesGateUp() || execution.executesDown() || rows != 1 ||
            table_id < 0 || table_id >= static_cast<int>(grouped_gateup_desc_tables_.size())) return nullptr;
        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            table.descriptor_source != MoEDecodeDescriptorSource::RuntimePlacementTable) return nullptr;
        if (!table.owner_m1_admission && !table.owner_m1_intent)
            return nullptr; // Existing unbound component path, not a declared graph policy.
        if (!table.owner_m1_admission || !table.owner_m1_intent)
            throw std::logic_error("owner gate/up table lost its complete immutable capability");
        if (rocm::currentMoERouteOwnedGateUpGeometry())
            throw std::logic_error("prepared model owner capability cannot be overridden by a diagnostic scope");
        // A bound operation cannot silently execute under a different shape.
        if (shape.output_columns != table.intermediate || shape.reduction_columns != table.d_model)
            throw std::logic_error("owner M1 gate/up requires its exact cold model capability");
        return &*table.owner_m1_admission;
    }

    /** @copydoc ROCmMoEKernel::recordOwnerGateUpCaptureSubmissions */
    void ROCmMoEKernel::recordOwnerGateUpCaptureSubmissions(
        const rocm::MoERouteOwnedGateUpLaunchSet &completed, std::uint64_t intent) const
    {
        if (!PerfStatsCollector::isDomainEnabled("kernel")) return;
        for (std::size_t index = 0; index < completed.leaf_count; ++index)
        {
            const auto &leaf = completed.leaves[index];
            PerfStatsCollector::addCounter("kernel", "rocm_moe_owner_m1_partition_submissions", 1.0,
                "submission", DeviceId::rocm(device_ordinal_).toString(),
                {{"observation", "host_execute_or_capture"}, {"partition_execution", "four_worker_planes"},
                 {"intent", std::to_string(intent)}, {"execution_codebook", std::to_string(leaf.execution_codebook)},
                 {"n", std::to_string(leaf.n)}, {"k", std::to_string(leaf.k)},
                 {"worker_planes", std::to_string(leaf.partition_worker_planes)},
                 {"grid_x", std::to_string(leaf.grid_x)}, {"grid_y", std::to_string(leaf.grid_y)}, {"grid_z", std::to_string(leaf.grid_z)},
                 {"block_threads", std::to_string(leaf.block_x)},
                 {"dynamic_shared_bytes", std::to_string(leaf.resources.dynamic_shared_memory_bytes)},
                 {"native_function", std::to_string(reinterpret_cast<std::uintptr_t>(leaf.resources.function))}});
        }
    }

    int ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables(
        const DeviceNativeQuantisedMatrixDesc *gate_descs,
        const DeviceNativeQuantisedMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != DeviceMoEWeightFormat::NativeQuantised)) return -1;

        if (!gate_descs || !up_descs || num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (d_model % 32) != 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped gate/up descriptor tables"))
        {
            return -1;
        }

        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertGateUpDescriptorTables"))
            return -1;

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeQuantisedExecutionFormatEnvelope format_envelope = original_family
            ? original_family->nativeGateUpEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete sparse descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (original_family && !original_family->matchesNativeGateUp(gate_desc,up_desc)) return -1;
            if (!gate_desc.valid() || !up_desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (!areNativeQuantisedGateUpCodebooksCompatible(gate_desc.codebook_id, up_desc.codebook_id))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Gate/up codebook mismatch for expert "
                          << expert_id << ": gate=" << static_cast<int>(gate_desc.codebook_id)
                          << " up=" << static_cast<int>(up_desc.codebook_id));
                return -1;
            }

            if (!validateGroupedGateUpDesc(gate_desc, d_model, intermediate) ||
                !validateGroupedGateUpDesc(up_desc, d_model, intermediate))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Invalid descriptor pair for expert "
                          << expert_id);
                return -1;
            }
            const auto add_descriptor = [&](
                                            const DeviceNativeQuantisedMatrixDesc &desc)
            {
                return addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!original_family && (!add_descriptor(gate_desc) || !add_descriptor(up_desc)))
            {
                return -1;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return -1;
        uint8_t codebook_id = kROCmMoEMixedCodebookSentinel;
        for (uint8_t codebook = 0u; codebook < 32u; ++codebook)
        {
            const uint32_t bit = nativeQuantisedCodebookMaskBit(codebook);
            if ((codebook_mask & bit) == 0u)
                continue;
            if (groupedPrefillCodebookBit(codebook) == 0u)
                return -1;
            if ((codebook_mask & (codebook_mask - 1u)) == 0u)
                codebook_id = codebook;
        }
        const int iq4_family_codebook = singleCodebookFromMask(codebook_mask);
        if (iq4_family_codebook >= 0)
            codebook_id = static_cast<uint8_t>(iq4_family_codebook);

        // Gate/up graphs use the same device-constant tables as down graphs.
        // Establish the prerequisite before a cached table can be returned.
        if (groupedPrefillMaskNeedsIQTables(codebook_mask) &&
            !rocm::ensureIQGridTablesInitialized(device_ordinal_))
        {
            LOG_ERROR(
                "[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] "
                "IQ grid table initialization failed on ROCm device "
                << device_ordinal_);
            return -1;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) ||
                existing.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
                !existing.device_gate_descs ||
                !existing.device_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.policy_codebook_mask != policy_codebook_mask ||
                existing.descriptor_source != descriptor_source ||
                existing.host_gate_descs.size() != static_cast<size_t>(num_experts) ||
                existing.host_up_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_gate_descs.data(), gate_descs, desc_bytes) == 0 &&
                std::memcmp(existing.host_up_descs.data(), up_descs, desc_bytes) == 0)
                return static_cast<int>(index);
        }

        GroupedGateUpDescriptorTable table;
        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;
        table.policy_codebook_mask = policy_codebook_mask;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.weight_format = DeviceMoEWeightFormat::NativeQuantised;
        table.valid = true;
        if (!publishGroupedGateUpDescriptorTable(
                table,
                "upload grouped expert gate/up descriptor tables"))
        {
            return -1;
        }
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    int ROCmMoEKernel::uploadGroupedExpertFloatingDownDescriptorTable(
        const DeviceMoEFloatingMatrixDesc *down_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != weight_format)) return -1;

        if (!down_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped floating down descriptor table") ||
            !setMoEDevice(
                device_ordinal_,
                "uploadGroupedExpertFloatingDownDescriptorTable"))
        {
            return -1;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &desc = down_descs[expert];
            if (isBlankGroupedFloatingDesc(desc))
                continue;
            if (original_family && !original_family->matchesFloatingDown(desc)) return -1;
            if (!validateGroupedFloatingDesc(desc, d_model, intermediate))
                return -1;
            any_live = true;
        }
        if (!any_live && !original_family)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) || existing.weight_format != weight_format ||
                !existing.device_floating_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.descriptor_source != descriptor_source ||
                existing.host_floating_descs.size() !=
                    static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(
                    existing.host_floating_descs.data(),
                    down_descs,
                    desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedDownDescriptorTable table;
        table.host_floating_descs.assign(
            down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.weight_format = weight_format;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.valid = true;
        if (!publishGroupedDownDescriptorTable(
                table,
                "upload grouped floating expert down descriptor table"))
        {
            return -1;
        }
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    int ROCmMoEKernel::uploadGroupedExpertFloatingGateUpDescriptorTables(
        const DeviceMoEFloatingMatrixDesc *gate_descs,
        const DeviceMoEFloatingMatrixDesc *up_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        const MoEIntermediateShardDescriptorDeclaration *original_family)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return -1;
        if (original_family &&
            (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
             !original_family->validFor(DeviceId::rocm(device_ordinal_),num_experts,d_model,intermediate) ||
             original_family->weightFormat() != weight_format)) return -1;

        if (!gate_descs || !up_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return -1;
        }
        if (rejectDecodeStagingDuringCapture(
                "upload grouped floating gate/up descriptor tables") ||
            !setMoEDevice(
                device_ordinal_,
                "uploadGroupedExpertFloatingGateUpDescriptorTables"))
        {
            return -1;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool gate_blank = isBlankGroupedFloatingDesc(gate_descs[expert]);
            const bool up_blank = isBlankGroupedFloatingDesc(up_descs[expert]);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return -1;
            }
            if (original_family && !original_family->matchesFloatingGateUp(gate_descs[expert],up_descs[expert])) return -1;
            if (!validateGroupedFloatingDesc(
                    gate_descs[expert], intermediate, d_model) ||
                !validateGroupedFloatingDesc(
                    up_descs[expert], intermediate, d_model))
            {
                return -1;
            }
            any_live = true;
        }
        if (!any_live && !original_family)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.original_family.sameIdentity(original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{}) || existing.weight_format != weight_format ||
                !existing.device_floating_gate_descs ||
                !existing.device_floating_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.descriptor_source != descriptor_source ||
                existing.host_floating_gate_descs.size() !=
                    static_cast<size_t>(num_experts) ||
                existing.host_floating_up_descs.size() !=
                    static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(
                    existing.host_floating_gate_descs.data(),
                    gate_descs,
                    desc_bytes) == 0 &&
                std::memcmp(
                    existing.host_floating_up_descs.data(),
                    up_descs,
                    desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_floating_gate_descs.assign(
            gate_descs, gate_descs + num_experts);
        table.host_floating_up_descs.assign(
            up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.weight_format = weight_format;
        table.descriptor_source = descriptor_source;
        table.original_family = original_family ? *original_family : MoEIntermediateShardDescriptorDeclaration{};
        table.valid = true;
        if (!publishGroupedGateUpDescriptorTable(
                table,
                "upload grouped floating expert gate/up descriptor tables"))
        {
            return -1;
        }
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    bool ROCmMoEKernel::updateGroupedExpertDownDescriptorTable(
        int descriptor_table_id,
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (intermediate % 32) != 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped down descriptor table"))
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertDownDescriptorTable"))
            return false;

        auto &table = grouped_down_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid ||
            table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !table.device_descs ||
            !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(table.descriptor_source);
        NativeQuantisedExecutionFormatEnvelope format_envelope = table.original_family.valid()
            ? table.original_family.nativeDownEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (table.original_family.valid() && !table.original_family.matchesNativeDown(desc)) return false;
            if (!desc.valid() || !validateGroupedDownDesc(desc, d_model, intermediate))
                return false;
            if (!table.original_family.valid() && !addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable))
            {
                return false;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return false;
        const int sole_codebook = singleCodebookFromMask(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kROCmMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask ||
            table.policy_codebook_mask != policy_codebook_mask)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    down_descs,
                    nullptr,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_descs,
                        down_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] "
                                  "descriptor refresh failed: "
                                  << hipGetErrorString(err));
                        return false;
                    }
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] "
                                  "cannot publish refreshed descriptor readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return true;
                }))
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertDownDescriptorTable] "
                      "cannot rewrite refreshed descriptor publication");
            return false;
        }

        table.host_descs.assign(down_descs, down_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables(
        int descriptor_table_id,
        const DeviceNativeQuantisedMatrixDesc *gate_descs,
        const DeviceNativeQuantisedMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (d_model % 32) != 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped gate/up descriptor tables"))
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertGateUpDescriptorTables"))
            return false;

        auto &table = grouped_gateup_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid ||
            table.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !table.device_gate_descs ||
            !table.device_up_descs || !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            moEDecodeUsesRuntimeDescriptors(table.descriptor_source);
        NativeQuantisedExecutionFormatEnvelope format_envelope = table.original_family.valid()
            ? table.original_family.nativeGateUpEnvelope() : NativeQuantisedExecutionFormatEnvelope{};
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return false;
            }
            if (table.original_family.valid() && !table.original_family.matchesNativeGateUp(gate_desc,up_desc)) return false;
            if (!gate_desc.valid() || !up_desc.valid() ||
                !areNativeQuantisedGateUpCodebooksCompatible(gate_desc.codebook_id, up_desc.codebook_id) ||
                !validateGroupedGateUpDesc(gate_desc, d_model, intermediate) ||
                !validateGroupedGateUpDesc(up_desc, d_model, intermediate))
            {
                return false;
            }
            const auto add_descriptor = [&](
                                            const DeviceNativeQuantisedMatrixDesc &desc)
            {
                return addNativeQuantisedExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeQuantisedSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!table.original_family.valid() && (!add_descriptor(gate_desc) || !add_descriptor(up_desc)))
            {
                return false;
            }
        }
        const uint32_t codebook_mask =
            format_envelope.execution_codebook_mask;
        const uint32_t policy_codebook_mask =
            format_envelope.policy_codebook_mask;
        if (codebook_mask == 0 || policy_codebook_mask == 0)
            return false;
        const int sole_codebook = singleCodebookFromMask(codebook_mask);
        const uint8_t codebook_id =
            sole_codebook >= 0
                ? static_cast<uint8_t>(sole_codebook)
                : kROCmMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask ||
            table.policy_codebook_mask != policy_codebook_mask)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeQuantisedMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    gate_descs,
                    up_descs,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_gate_descs,
                        gate_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "gate descriptor refresh failed: "
                                  << hipGetErrorString(err));
                        return false;
                    }
                    err = hipMemcpyAsync(
                        table.device_up_descs,
                        up_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "up refresh failed after gate submission: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                                  "cannot publish refreshed descriptor readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return true;
                }))
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables] "
                      "cannot rewrite refreshed descriptor publication");
            return false;
        }

        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::updateGroupedExpertFloatingDownDescriptorTable(
        int descriptor_table_id,
        const DeviceMoEFloatingMatrixDesc *down_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            descriptor_table_id < 0 ||
            descriptor_table_id >=
                static_cast<int>(grouped_down_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped floating down descriptor table") ||
            !setMoEDevice(
                device_ordinal_,
                "updateGroupedExpertFloatingDownDescriptorTable"))
        {
            return false;
        }

        auto &table = grouped_down_desc_tables_[
            static_cast<size_t>(descriptor_table_id)];
        if (!table.valid || table.weight_format != weight_format ||
            !table.device_floating_descs || !table.workspace_publication ||
            table.num_experts != num_experts || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const auto &desc = down_descs[expert];
            if (isBlankGroupedFloatingDesc(desc))
                continue;
            if (!validateGroupedFloatingDesc(desc, d_model, intermediate))
                return false;
            any_live = true;
        }
        if (!any_live)
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
            return false;
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedFloatingDescriptorPublicationKey(
                    down_descs,
                    nullptr,
                    weight_format,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_floating_descs,
                        down_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                        return false;
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                        std::terminate();
                    return true;
                }))
        {
            return false;
        }
        table.host_floating_descs.assign(
            down_descs, down_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::updateGroupedExpertFloatingGateUpDescriptorTables(
        int descriptor_table_id,
        const DeviceMoEFloatingMatrixDesc *gate_descs,
        const DeviceMoEFloatingMatrixDesc *up_descs,
        DeviceMoEWeightFormat weight_format,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs ||
            !deviceMoEWeightFormatIsFloating(weight_format) ||
            descriptor_table_id < 0 ||
            descriptor_table_id >=
                static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (rejectDecodeStagingDuringCapture(
                "update grouped floating gate/up descriptor tables") ||
            !setMoEDevice(
                device_ordinal_,
                "updateGroupedExpertFloatingGateUpDescriptorTables"))
        {
            return false;
        }

        auto &table = grouped_gateup_desc_tables_[
            static_cast<size_t>(descriptor_table_id)];
        if (!table.valid || table.weight_format != weight_format ||
            !table.device_floating_gate_descs ||
            !table.device_floating_up_descs ||
            !table.workspace_publication ||
            table.num_experts != num_experts || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        bool any_live = false;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            const bool gate_blank = isBlankGroupedFloatingDesc(gate_descs[expert]);
            const bool up_blank = isBlankGroupedFloatingDesc(up_descs[expert]);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return false;
            }
            if (!validateGroupedFloatingDesc(
                    gate_descs[expert], intermediate, d_model) ||
                !validateGroupedFloatingDesc(
                    up_descs[expert], intermediate, d_model))
            {
                return false;
            }
            any_live = true;
        }
        if (!any_live)
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
            return false;
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedFloatingDescriptorPublicationKey(
                    gate_descs,
                    up_descs,
                    weight_format,
                    num_experts,
                    d_model,
                    intermediate, table.descriptor_source),
                table.workspace_publication,
                table.workspace_slot,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&]()
                {
                    hipError_t err = hipMemcpyAsync(
                        table.device_floating_gate_descs,
                        gate_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                        return false;
                    err = hipMemcpyAsync(
                        table.device_floating_up_descs,
                        up_descs,
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                        std::terminate();
                    err = hipEventRecord(
                        static_cast<hipEvent_t>(
                            table.workspace_publication->ready_event),
                        stream);
                    if (err != hipSuccess)
                        std::terminate();
                    return true;
                }))
        {
            return false;
        }
        table.host_floating_gate_descs.assign(
            gate_descs, gate_descs + num_experts);
        table.host_floating_up_descs.assign(
            up_descs, up_descs + num_experts);
        return true;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromTable(
        const TensorBase *input,
        const int *expert_ids,
        int descriptor_table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || !expert_ids || descriptor_table_id < 0 || num_active <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating =
            deviceMoEWeightFormatIsFloating(table.weight_format);
        if (!table.valid || !table.deviceReady() || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] "
                      "descriptor table mismatch"
                      << " table_id=" << descriptor_table_id
                      << " valid=" << table.valid
                      << " device_ready=" << table.deviceReady()
                      << " format="
                      << static_cast<std::uint32_t>(table.weight_format)
                      << " table_d_model=" << table.d_model
                      << " requested_d_model=" << d_model
                      << " table_intermediate=" << table.intermediate
                      << " requested_intermediate=" << intermediate);
            return false;
        }
        if (!floating && (d_model % 32) != 0)
            return false;

        for (int i = 0; i < num_active; ++i)
        {
            const int expert_id = expert_ids[i];
            if (expert_id < 0 || expert_id >= table.num_experts)
            {
                return false;
            }
        }

        // Floating weights have no Q8/K-part intermediate dependency, including
        // their geometry constraints. Metadata/pointer publication is shared.
        if (!floating && !ensureGroupedGateUpCapacity(
                num_active, d_model, intermediate))
            return false;

        const int *fixed_device_expert_ids = nullptr;
        if (!resolveFixedTableGateUpMetadata(
                table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                num_active,
                &fixed_device_expert_ids))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && rejectDecodeStagingDuringCapture("grouped gate/up input tensor"))
            return false;
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    DeviceId::rocm(device_ordinal_),
                    getStream(),
                    "input",
                    "groupedExpertGateUpDecodeFromTable"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        if (!d_hidden)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] Input tensor has no device pointer");
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_output_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_output_ptrs = {};
        for (int i = 0; i < num_active; ++i)
        {
            if (!gate_outputs[i] || !up_outputs[i])
                return false;
            gate_output_ptrs[i] = static_cast<float *>(gate_outputs[i]->gpu_data_ptr());
            up_output_ptrs[i] = static_cast<float *>(up_outputs[i]->gpu_data_ptr());
            if (!gate_output_ptrs[i] || !up_output_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] Null gate/up output pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float **d_gate_output_ptrs = nullptr;
        float **d_up_output_ptrs = nullptr;
        if (!stageRuntimeGateUpPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                num_active, gate_output_ptrs, up_output_ptrs,
                &d_gate_output_ptrs, &d_up_output_ptrs))
        {
            return false;
        }

        /* Table decode shares stable metadata/pointer publications between
         * packed and contiguous formats; dispatch only selects the arithmetic
         * kernel appropriate for the descriptor family. */
        const bool reuse_required_hidden=router_q8_hidden_publication_&&router_q8_hidden_publication_->requiredOriginalM1Source();
        int8_t* original_hidden_codes=d_grouped_hidden_int8_;
        float* original_hidden_scales=d_grouped_hidden_scales_;
        if(reuse_required_hidden) {
            const auto& publication=*router_q8_hidden_publication_;
            if(floating)throw std::logic_error("Required M1 hidden Q8 reader has incompatible floating projection arithmetic");
            if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumer) {
                if(!canReuseRouterQ8Hidden(d_hidden,1,d_model))throw std::logic_error("Required M1 routed hidden Q8 reader is absent");
                original_hidden_codes=publication.original_m1_binding->codes();
                original_hidden_scales=publication.original_m1_binding->scales();
            } else {
            if(!original_m1_hidden_q8_read_||publication.backend!=DeviceType::ROCm||publication.device_ordinal!=device_ordinal_||
               publication.d_model_capacity<d_model||publication.blocks_per_row_capacity<d_model/32)
                throw std::logic_error("Required M1 shared hidden Q8 reader was not cold prepared on its exact participant/width");
            original_m1_hidden_q8_read_->validate(*input,getStream(),workspace_);
            original_m1_hidden_q8_read_->requirePayload(publication.source_rows,publication.quantized_rows,publication.row_scales,
                publication.published_rows,
                publication.capture_recorded?OriginalM1HiddenQ8Use::CaptureRecording:OriginalM1HiddenQ8Use::OrderedEager,
                isGraphCaptureActive()?OriginalM1HiddenQ8Use::CaptureRecording:OriginalM1HiddenQ8Use::OrderedEager);
            // The original low-level ABI combines writable independent scratch
            // with prequantized input. Only this sealed read view selects true;
            // that arm never writes codes/scales. No caller supplies raw reuse.
            original_hidden_codes=const_cast<int8_t*>(original_m1_hidden_q8_read_->codes());
            original_hidden_scales=const_cast<float*>(original_m1_hidden_q8_read_->scales());
            }
        }

        const bool ok = floating
            ? rocmMoE_grouped_gate_up_floating_decode_table(
                  d_hidden,
                  table.device_floating_gate_descs,
                  table.device_floating_up_descs,
                  fixed_device_expert_ids,
                  d_gate_output_ptrs,
                  d_up_output_ptrs,
                  num_active,
                  intermediate,
                  d_model,
                  table.num_experts,
                  table.weight_format,
                  device_ordinal_,
                  getStream())
            : rocmMoE_grouped_gate_up_native_quantised_decode_table(
                  d_hidden,
                  table.device_gate_descs,
                  table.device_up_descs,
                  fixed_device_expert_ids,
                  d_gate_output_ptrs,
                  d_up_output_ptrs,
                  nullptr,
                  nullptr,
                  original_hidden_codes,
                  original_hidden_scales,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  reuse_required_hidden,
                  num_active,
                  intermediate,
                  d_model,
                  table.num_experts,
                  table.codebook_id,
                  table.policy_codebook_mask,
                  device_ordinal_,
                  getStream());

        if (ok)
        {
            for (int i = 0; i < num_active; ++i)
            {
                markDeviceWritten(
                    gate_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
                markDeviceWritten(
                    up_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        int descriptor_table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || !routing_indices || descriptor_table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > 16 || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedGateUpCapacity(top_k, d_model, intermediate))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRouting"))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden &&
            rejectDecodeStagingDuringCapture("grouped gate/up input tensor"))
            return false;
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    DeviceId::rocm(device_ordinal_),
                    getStream(),
                    "input",
                    "groupedExpertGateUpDecodeFromRouting"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const float *d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_routing_indices &&
            !requireTensorOnDevice(routing_indices, device, getStream(),
                                  "routing_indices", "groupedExpertGateUpDecodeFromRouting"))
        {
            return false;
        }
        d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_hidden || !d_routing_indices)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] Missing input/routing device pointer");
            return false;
        }

        host_grouped_gate_output_ptrs_.assign(static_cast<size_t>(top_k), nullptr);
        host_grouped_up_output_ptrs_.assign(static_cast<size_t>(top_k), nullptr);
        for (int i = 0; i < top_k; ++i)
        {
            if (!requireOutputOnDevice(gate_outputs[i], device, getStream(),
                                      "gate_output", "groupedExpertGateUpDecodeFromRouting") ||
                !requireOutputOnDevice(up_outputs[i], device, getStream(),
                                      "up_output", "groupedExpertGateUpDecodeFromRouting"))
            {
                return false;
            }
            host_grouped_gate_output_ptrs_[i] = static_cast<float *>(gate_outputs[i]->gpu_data_ptr());
            host_grouped_up_output_ptrs_[i] = static_cast<float *>(up_outputs[i]->gpu_data_ptr());
            if (!host_grouped_gate_output_ptrs_[i] || !host_grouped_up_output_ptrs_[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] Null gate/up output pointer for slot "
                          << i);
                return false;
            }
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_output_ptrs_, host_grouped_gate_output_ptrs_.data(),
                                        static_cast<size_t>(top_k) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_output_ptrs_, host_grouped_up_output_ptrs_.data(),
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] H2D pointer staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        /*
         * The ROCm MTP verifier deliberately replays decode-equivalent rows from
         * device routing tensors. In LocalTP routed-expert overlays the route
         * tensor names global experts, while this participant should compute only
         * its mask-active shard before the MoE output allreduce.  The masked
         * conversion writes -1 for nonlocal top-k slots into backend scratch; the
         * grouped table kernels already skip those inactive slots.
         */
        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!hipMoE_float_to_masked_int(
                    d_routing_indices,
                    d_grouped_gateup_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    getStream()))
            {
                return false;
            }
        }
        else if (!hipMoE_float_to_int(
                     d_routing_indices, d_grouped_gateup_expert_ids_, top_k,
                     device_ordinal_, getStream()))
        {
            return false;
        }

        /*
         * Gate/up owns one increasing-K accumulator per route and output column.
         * That arithmetic is the decode contract for both explicit routing and
         * runtime placement. Split-K variants are deliberately absent because
         * reducing partition totals changes the FP32 addition tree.
         */
        const bool ok = rocmMoE_grouped_gate_up_native_quantised_decode_table(
            d_hidden,
            table.device_gate_descs,
            table.device_up_descs,
            d_grouped_gateup_expert_ids_,
            d_grouped_gate_output_ptrs_,
            d_grouped_up_output_ptrs_,
            nullptr,
            nullptr,
            d_grouped_hidden_int8_,
            d_grouped_hidden_scales_,
            d_grouped_gateup_gate_partials_,
            d_grouped_gateup_up_partials_,
            false,
            top_k,
            intermediate,
            d_model,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
        {
            for (int i = 0; i < top_k; ++i)
            {
                markDeviceWritten(gate_outputs[i], device, getStream());
                markDeviceWritten(up_outputs[i], device, getStream());
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int descriptor_table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !input || descriptor_table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedGateUpCapacity(top_k, d_model, intermediate))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRuntime"))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden &&
            rejectDecodeStagingDuringCapture("grouped gate/up input tensor"))
            return false;
        if (!capture_active)
        {
            if (!requireTensorOnDevice(
                    const_cast<TensorBase *>(input),
                    DeviceId::rocm(device_ordinal_),
                    getStream(),
                    "input",
                    "groupedExpertGateUpDecodeFromRuntime"))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        if (!d_hidden || !d_expert_ids)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] Missing input/runtime device pointer");
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_output_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_output_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_output_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_output_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_output_ptrs[slot] || !up_output_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] Null gate/up output pointer for slot "
                          << slot);
                return false;
            }
        }

        float **d_gate_output_ptrs = nullptr;
        float **d_up_output_ptrs = nullptr;
        if (!stageRuntimeGateUpPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::RuntimeTwoStep,
                top_k, gate_output_ptrs, up_output_ptrs,
                &d_gate_output_ptrs, &d_up_output_ptrs))
        {
            return false;
        }

        const bool reuse_router_q8_hidden =
            canReuseRouterQ8Hidden(d_hidden, /*rows=*/1, d_model);
        int8_t *gateup_hidden_int8 = reuse_router_q8_hidden ? d_router_q8_hidden_ : d_grouped_hidden_int8_;
        float *gateup_hidden_scales = reuse_router_q8_hidden ? d_router_q8_hidden_scales_ : d_grouped_hidden_scales_;

        const bool ok = rocmMoE_grouped_gate_up_native_quantised_decode_runtime(
            d_hidden,
            runtime_layer,
            d_expert_ids,
            d_gate_output_ptrs,
            d_up_output_ptrs,
            gateup_hidden_int8,
            gateup_hidden_scales,
            d_grouped_gateup_gate_partials_,
            d_grouped_gateup_up_partials_,
            reuse_router_q8_hidden,
            top_k,
            intermediate,
            d_model,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
        {
            for (int i = 0; i < top_k; ++i)
            {
                markDeviceWritten(
                    gate_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
                markDeviceWritten(
                    up_outputs[i],
                    DeviceId::rocm(device_ordinal_),
                    getStream());
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int gateup_descriptor_table_id,
        int down_descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask,
        ITensor *canonical_route_contributions,
        DeviceMoELayerRuntime *runtime_layer,
        MoEDecodeDescriptorSource descriptor_source,
        MoEExplicitRouteDemand demand)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source) || !validMoEExplicitRouteDemand(demand)) return false;

        if (!input || !routing_indices || !routing_weights ||
            gateup_descriptor_table_id < 0 || down_descriptor_table_id < 0 ||
            top_k <= 0 || !output || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            gateup_descriptor_table_id >=
                static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table =
            grouped_gateup_desc_tables_[gateup_descriptor_table_id];
        if (static_cast<size_t>(down_descriptor_table_id) >= grouped_down_desc_tables_.size()) return false;
        const auto &down_table = grouped_down_desc_tables_[down_descriptor_table_id];
        if (!down_table.valid ||
            (gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        if (!gateup_table.valid || gateup_table.num_experts <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "gate/up descriptor table is unavailable");
            return false;
        }
        const bool use_runtime_descriptors =
            moEDecodeUsesRuntimeDescriptors(descriptor_source);
        if ((use_runtime_descriptors && (!runtime_layer || expert_mask)) ||
            (!use_runtime_descriptors && runtime_layer))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "runtime placement requires exactly one runtime layer and no host mask");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_, "groupedExpertDecodeFromRouting") ||
            !ensureGroupedGateUpCapacity(top_k, d_model, intermediate))
        {
            return false;
        }

        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "an explicit stream is required");
            return false;
        }

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const bool capture_active = isDecodeGraphCaptureActive();
        const float *device_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *device_routing_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if ((!device_routing_indices || !device_routing_weights) &&
            capture_active)
        {
            rejectDecodeStagingDuringCapture(
                "fused explicit-routing tensors");
            return false;
        }
        if (!device_routing_indices &&
            !requireTensorOnDevice(
                routing_indices,
                device,
                stream,
                "routing_indices",
                "groupedExpertDecodeFromRouting"))
        {
            return false;
        }
        if (!device_routing_weights &&
            !requireTensorOnDevice(
                routing_weights,
                device,
                stream,
                "routing_weights",
                "groupedExpertDecodeFromRouting"))
        {
            return false;
        }

        device_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        device_routing_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!device_routing_indices || !device_routing_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "routing tensors have no device publication");
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(
                    expert_mask, gateup_table.num_experts) ||
                !hipMoE_float_to_masked_int(
                    device_routing_indices,
                    d_grouped_gateup_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    gateup_table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!hipMoE_float_to_int(
                     device_routing_indices,
                     d_grouped_gateup_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        const int *resolved_expert_ids = d_grouped_gateup_expert_ids_;
        const float *resolved_weights = device_routing_weights;
        if (use_runtime_descriptors)
        {
            /* Keep route publication, placement filtering, and histogram
             * accounting in one exact captured follower stream. */
            if (!hipMoE_decode_route_select_runtime(
                    d_grouped_gateup_expert_ids_,
                    device_routing_weights,
                    runtime_layer,
                    /*legacy_indices=*/nullptr,
                    /*legacy_weights=*/nullptr,
                    gateup_table.num_experts,
                    top_k,
                    /*write_legacy_outputs=*/false,
                    /*update_runtime_histogram=*/demand == MoEExplicitRouteDemand::PublishDecode,
                    descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable
                        ? RoutedExpertRowExecutionPolicy::IntermediateShardLocal : RoutedExpertRowExecutionPolicy::ParticipantAssigned,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
            resolved_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
            resolved_weights = runtimeTopKWeightsDevice(runtime_layer);
        }

        return groupedExpertDecodeResolved(
            runtime_layer,
            input,
            gateup_descriptor_table_id,
            down_descriptor_table_id,
            top_k,
            output,
            d_model,
            intermediate,
            resolved_expert_ids,
            resolved_weights,
            descriptor_source,
            /*allow_router_q8_reuse=*/false,
            descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable
                ? "runtime_intermediate_shard" : use_runtime_descriptors ? "routing_runtime" : "routing",
            canonical_route_contributions);
    }

    const ROCmMoEKernel::PairedRouterDescriptors &ROCmMoEKernel::requirePairedRouterDescriptors(
        DeviceMoELayerRuntime *runtime, int num_experts, int top_k) const
    {
        if (!paired_router_descriptors_)
            throw std::logic_error("Missing original paired router descriptor preparation");
        const auto &binding = *paired_router_descriptors_;
        if (runtime != binding.runtime || workspace_ != binding.workspace || !workspace_ ||
            workspace_->id() != binding.workspace_id || getStream() != binding.stream ||
            num_experts != binding.outputs.expert_count || top_k != binding.top_k)
            throw std::logic_error("Paired router descriptors changed runtime, workspace, stream or geometry");
        return binding;
    }

    MoEDecodeDescriptorPublisher ROCmMoEKernel::preparePairedRouterDecodeDescriptors(
        DeviceMoELayerRuntime *runtime, int gateup_id, int down_id, int top_k)
    {
        if (isGraphCaptureActive() || !workspace_ || !runtime ||
            reinterpret_cast<std::uintptr_t>(getStream()) <= 2 ||
            gateup_id < 0 || down_id < 0 || top_k <= 0 || top_k > int(kDeviceMoEMaxTopK) ||
            gateup_id >= int(grouped_gateup_desc_tables_.size()) || down_id >= int(grouped_down_desc_tables_.size()))
            throw std::logic_error("Paired router descriptors require complete cold graph preparation");
        const auto &gate = grouped_gateup_desc_tables_[gateup_id];
        const auto &down = grouped_down_desc_tables_[down_id];
        if (!gate.valid || !gate.deviceReady() || !down.valid || !down.deviceReady() ||
            gate.descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
            down.descriptor_source != gate.descriptor_source || gate.weight_format != down.weight_format ||
            gate.num_experts != down.num_experts || gate.d_model != down.d_model ||
            gate.intermediate != down.intermediate || top_k > gate.num_experts)
            throw std::logic_error("Paired router descriptors require the same original fragment family");
        const bool ordinary=rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::OrdinaryPhases||
            rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::NativeLayoutRouter;
        const auto &policy = debugEnv().rocm;
        if(ordinary&&(!policy.moe_router_wave_topk||policy.moe_router_kpart_decode||
            gate.num_experts>int(kDeviceMoEMaxExperts)))
            throw std::invalid_argument("Ordinary FFN requires the original paired wave-topK descriptor publisher");
        if (paired_router_descriptors_)
        {
            const auto &original = requirePairedRouterDescriptors(runtime, gate.num_experts, top_k);
            if (original.gateup_table != gateup_id || original.down_table != down_id)
                throw std::logic_error("Paired router descriptors changed original table handles");
            return MoEDecodeDescriptorPublisher::Router;
        }
        // This is a cold performance-policy choice, not recovery after a failed
        // operation. The ordinary FFN candidate explicitly consumes this
        // publication on every HIP target; defaults retain the measured gfx906
        // selection and other architectures' independent two-node operation.
        if (!ordinary&&(!rocmHardwareMetadata().at(device_ordinal_).isa_target.starts_with("gfx906") ||
            !policy.moe_router_wave_topk || policy.moe_router_kpart_decode ||
            gate.num_experts > int(kDeviceMoEMaxExperts)))
            return MoEDecodeDescriptorPublisher::ExpertConsumer;
        MoEDecodeDescriptorPublication outputs{
            .expert_count = gate.num_experts,
            .format = gate.weight_format,
            .projections = DeviceMoEProjectionSet::IntermediateShard};
        if (!bindGroupedDescriptorTableSlot(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gate.workspace_slot, gate.num_experts, &outputs.gate, "paired router gate descriptors") ||
            !bindGroupedDescriptorTableSlot(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gate.workspace_slot, gate.num_experts, &outputs.up, "paired router up descriptors") ||
            !bindGroupedDescriptorTableSlot(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down.workspace_slot, down.num_experts, &outputs.down, "paired router down descriptors") || !outputs.valid())
            throw std::logic_error("Paired router descriptors lack the original admitted arena destinations");
        paired_router_descriptors_ = PairedRouterDescriptors{
            runtime, workspace_, workspace_->id(), getStream(), gateup_id, down_id, top_k, outputs};
        return MoEDecodeDescriptorPublisher::Router;
    }

    bool ROCmMoEKernel::prepareGroupedRuntimeDecodeLaunchState(
        int gateup_table_id,
        int down_table_id,
        int top_k,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;

        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] Runtime grouped-decode preparation "
                      "requires the exact non-null producer stream");
            return false;
        }
        if (isDecodeGraphCaptureActive())
        {
            LOG_ERROR("[ROCmMoEKernel] Runtime grouped-decode launch state must "
                      "be prepared before graph capture begins");
            return false;
        }
        if (gateup_table_id < 0 || down_table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            d_model <= 0 || intermediate <= 0 ||
            gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model ||
            down_table.intermediate != intermediate ||
            !setMoEDevice(device_ordinal_,
                          "prepareGroupedRuntimeDecodeLaunchState") ||
            !ensureGroupedGateUpCapacity(top_k, d_model, intermediate) ||
            !ensureGroupedDecodeCapacity(top_k, intermediate, d_model) ||
            !ensureGroupedPrefillScratchCapacity(
                top_k, d_model, intermediate))
        {
            return false;
        }
        if (!floating &&
            ((d_model % 32) != 0 || (intermediate % 32) != 0))
        {
            return false;
        }

        if (moEDecodeUsesRuntimeDescriptors(descriptor_source))
        {
            DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
            if (!bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_gate_descs,
                    "ROCm runtime decode gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_up_descs,
                    "ROCm runtime decode up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    down_table.num_experts,
                    &runtime_down_descs,
                    "ROCm runtime decode down descriptors"))
            {
                return false;
            }
        }

        const int max_dim = std::max(d_model, intermediate);
        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] =
                d_prefill_gate_ + static_cast<size_t>(slot) * max_dim;
            up_ptrs[slot] =
                d_prefill_up_ + static_cast<size_t>(slot) * intermediate;
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }

        float **device_gate_ptrs = nullptr;
        float **device_up_ptrs = nullptr;
        const float **device_down_gate_ptrs = nullptr;
        const float **device_down_up_ptrs = nullptr;
        return stageRuntimeGateUpPointerArrays(
                   gateup_table.workspace_slot,
                   RuntimePointerArrayScope::RuntimeFused,
                   top_k,
                   gate_ptrs,
                   up_ptrs,
                   &device_gate_ptrs,
                   &device_up_ptrs) &&
               stageRuntimeDownPointerArrays(
                   down_table.workspace_slot,
                   RuntimePointerArrayScope::RuntimeFused,
                   top_k,
                   const_gate_ptrs,
                   const_up_ptrs,
                   &device_down_gate_ptrs,
                   &device_down_up_ptrs);
    }

    bool ROCmMoEKernel::prepareGroupedTableDecodeLaunchState(
        const int *expert_ids,
        const float *expert_weights,
        int gateup_table_id,
        int down_table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped-decode preparation "
                      "requires the exact non-null producer stream");
            return false;
        }
        if (isDecodeGraphCaptureActive())
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped-decode launch state "
                      "must be prepared before graph capture begins");
            return false;
        }
        if (!expert_ids || !expert_weights || !gate_outputs || !up_outputs ||
            !output || gateup_table_id < 0 || down_table_id < 0 ||
            num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            d_model <= 0 || intermediate <= 0 ||
            gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_table_id];
        if (gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const auto &down_table = grouped_down_desc_tables_[down_table_id];
        if (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model ||
            down_table.intermediate != intermediate ||
            !setMoEDevice(device_ordinal_,
                          "prepareGroupedTableDecodeLaunchState"))
        {
            return false;
        }
        if (!floating &&
            ((d_model % 32) != 0 || (intermediate % 32) != 0))
        {
            return false;
        }
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 ||
                expert_ids[slot] >= gateup_table.num_experts ||
                expert_ids[slot] >= down_table.num_experts)
            {
                return false;
            }
        }
        const int *fixed_gateup_expert_ids = nullptr;
        const int *fixed_down_expert_ids = nullptr;
        const float *fixed_down_expert_weights = nullptr;
        if ((!floating &&
             (!ensureGroupedGateUpCapacity(num_active, d_model, intermediate) ||
              !ensureGroupedDecodeCapacity(num_active, intermediate, d_model))) ||
            !resolveFixedTableGateUpMetadata(
                gateup_table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                num_active,
                &fixed_gateup_expert_ids) ||
            !resolveFixedTableDownMetadata(
                down_table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                expert_weights,
                num_active,
                &fixed_down_expert_ids,
                &fixed_down_expert_weights))
        {
            return false;
        }
        // Both families publish immutable metadata and pointer arrays. Only
        // quantized arithmetic above binds Q8/K-part intermediate workspaces.
        (void)fixed_gateup_expert_ids;
        (void)fixed_down_expert_ids;
        (void)fixed_down_expert_weights;

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK>
            const_up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            gate_ptrs[slot] = gate_outputs[slot]
                                  ? static_cast<float *>(
                                        gate_outputs[slot]->gpu_data_ptr())
                                  : nullptr;
            up_ptrs[slot] = up_outputs[slot]
                                ? static_cast<float *>(
                                      up_outputs[slot]->gpu_data_ptr())
                                : nullptr;
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped decode has no "
                          "persistent gate/up scratch pointer for slot "
                          << slot);
                return false;
            }
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }
        if (!output->gpu_data_ptr())
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table grouped decode output is not "
                      "bound to persistent device storage before capture");
            return false;
        }

        float **device_gate_ptrs = nullptr;
        float **device_up_ptrs = nullptr;
        const float **device_down_gate_ptrs = nullptr;
        const float **device_down_up_ptrs = nullptr;
        return stageRuntimeGateUpPointerArrays(
                   gateup_table.workspace_slot,
                   RuntimePointerArrayScope::TableDecode,
                   num_active,
                   gate_ptrs,
                   up_ptrs,
                   &device_gate_ptrs,
                   &device_up_ptrs) &&
               stageRuntimeDownPointerArrays(
                   down_table.workspace_slot,
                   RuntimePointerArrayScope::TableDecode,
                   num_active,
                   const_gate_ptrs,
                   const_up_ptrs,
                   &device_down_gate_ptrs,
                   &device_down_up_ptrs);
    }

    bool ROCmMoEKernel::groupedExpertDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_descriptor_table_id,
        int down_descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source,
        ITensor *canonical_route_contributions)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;

        if (!runtime_layer)
            return false;

        return groupedExpertDecodeResolved(
            runtime_layer,
            input,
            gateup_descriptor_table_id,
            down_descriptor_table_id,
            top_k,
            output,
            d_model,
            intermediate,
            runtimeTopKExpertIdsDevice(runtime_layer),
            runtimeTopKWeightsDevice(runtime_layer),
            descriptor_source,
            /*allow_router_q8_reuse=*/true,
            descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable
                ? "runtime_intermediate_shard" : moEDecodeUsesRuntimeDescriptors(descriptor_source)
                ? "runtime_compact_table"
                : "runtime_static_table",
            canonical_route_contributions);
    }

    bool ROCmMoEKernel::groupedExpertDecodeResolved(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_descriptor_table_id,
        int down_descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const int *d_expert_ids,
        const float *d_weights,
        MoEDecodeDescriptorSource descriptor_source,
        bool allow_router_q8_reuse,
        const char *counter_source,
        ITensor *canonical_route_contributions)
    {
        if (!validMoEDecodeDescriptorSource(descriptor_source)) return false;

        const bool use_runtime_descriptors = moEDecodeUsesRuntimeDescriptors(descriptor_source);

        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || gateup_descriptor_table_id < 0 ||
            down_descriptor_table_id < 0 || top_k <= 0 || !output ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (!d_expert_ids || !d_weights ||
            (use_runtime_descriptors && !runtime_layer))
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
        {
            return false;
        }
        if (gateup_descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_descriptor_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_descriptor_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
            return false;

        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.d_model != d_model || gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model || down_table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        const hipStream_t stream = static_cast<hipStream_t>(getStream());
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!setMoEDevice(device_ordinal_, "groupedExpertDecodeFromRuntime"))
            return false;

        /*
         * This fused wrapper is intentionally workspace-native. The stage no
         * longer has to materialize one TensorBase per top-k slot, and captured
         * graphs replay stable pointer-table slots instead of host-owned scratch
         * tensors. If a graph would need to upload/allocate here, fail loudly.
         */
        if (!ensureGroupedGateUpCapacity(
                top_k, d_model, intermediate) ||
            !ensureGroupedDecodeCapacity(top_k, intermediate, d_model) ||
            !ensureGroupedPrefillScratchCapacity(top_k, d_model, intermediate))
        {
            return false;
        }

        /*
         * Runtime placement changes descriptor values while the graph-owned
         * workspace identity remains stable.  Materialize the active device
         * bank into compact projection tables before launching decode GEMVs.
         *
         * The former runtime kernels indexed the full DeviceMoELayerRuntime
         * object from every output lane.  That structure contains two complete
         * 256-expert banks plus routing and histogram state, so a single
         * transfer-backed expert forced every decode projection through a
         * cache-hostile descriptor walk.  LLEP consequently made otherwise
         * ordinary Qwen 3.6 layers tens of milliseconds slower.
         *
         * This publication is a device-to-device kernel on the exact producer
         * stream.  It is graph capturable, performs no allocation or host
         * transfer, and feeds the same compact table kernels used by immutable
         * placement.  Therefore mutable and immutable placement differ only in
         * where the table values are published, not in GEMV arithmetic.
         */
        const DeviceNativeQuantisedMatrixDesc *decode_gate_descs =
            gateup_table.device_gate_descs;
        const DeviceNativeQuantisedMatrixDesc *decode_up_descs =
            gateup_table.device_up_descs;
        const DeviceNativeQuantisedMatrixDesc *decode_down_descs =
            down_table.device_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_gate_descs =
            gateup_table.device_floating_gate_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_up_descs =
            gateup_table.device_floating_up_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_down_descs =
            down_table.device_floating_descs;
        if (use_runtime_descriptors)
        {
            DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
            if (!bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_gate_descs,
                    "ROCm runtime decode gate descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                    gateup_table.workspace_slot,
                    gateup_table.num_experts,
                    &runtime_up_descs,
                    "ROCm runtime decode up descriptors") ||
                !bindGroupedDescriptorTableSlot(
                    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                    down_table.workspace_slot,
                    down_table.num_experts,
                    &runtime_down_descs,
                    "ROCm runtime decode down descriptors"))
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] "
                          "failed to bind graph-owned runtime descriptor slots");
                return false;
            }
            // External route packets deliberately own their selected IDs and
            // descriptor publication, even when this kernel instance also
            // prepared an original-router graph. Only the original runtime
            // route entry can consume that router's paired publication.
            if (paired_router_descriptors_ && allow_router_q8_reuse)
            {
                const auto &binding = requirePairedRouterDescriptors(runtime_layer, gateup_table.num_experts, top_k);
                if (descriptor_source != MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable ||
                    binding.gateup_table != gateup_descriptor_table_id || binding.down_table != down_descriptor_table_id ||
                    binding.outputs.gate != runtime_gate_descs || binding.outputs.up != runtime_up_descs ||
                    binding.outputs.down != runtime_down_descs || binding.outputs.format != gateup_table.weight_format ||
                    d_expert_ids != runtimeTopKExpertIdsDevice(runtime_layer) || d_weights != runtimeTopKWeightsDevice(runtime_layer))
                    throw std::logic_error("Paired descriptor consumer changed its original producer/table contract");
            }
            else
            {
                const bool materialized = hipMoE_materialize_runtime_selected_decode_descriptors(
                    runtime_layer, d_expert_ids, runtime_gate_descs, runtime_up_descs,
                    runtime_down_descs, gateup_table.num_experts, top_k,
                    gateup_table.weight_format, device_ordinal_, stream,
                    moEDecodeProjectionSet(descriptor_source));
                if (!materialized)
                {
                    LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] "
                              "failed to materialize compact runtime descriptor tables");
                    return false;
                }
            }
            decode_gate_descs = runtime_gate_descs;
            decode_up_descs = runtime_up_descs;
            decode_down_descs = runtime_down_descs;
            decode_floating_gate_descs =
                reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                    runtime_gate_descs);
            decode_floating_up_descs =
                reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                    runtime_up_descs);
            decode_floating_down_descs =
                reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                    runtime_down_descs);
        }

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && rejectDecodeStagingDuringCapture("fused grouped decode input tensor"))
            return false;
        if (!d_hidden)
        {
            TransferEngine::requireDeviceInput(
                const_cast<TensorBase *>(input),
                device,
                stream);
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }

        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical route contribution tensor"
                                                  : "fused grouped decode output tensor";
        float *d_publication_output = static_cast<float *>(
            publication_output->gpu_data_ptr());
        if (!d_publication_output &&
            rejectDecodeStagingDuringCapture(publication_output_name))
            return false;
        if (!d_publication_output)
        {
            TransferEngine::requireDeviceOutput(
                publication_output,
                device,
                stream);
            d_publication_output = static_cast<float *>(
                publication_output->gpu_data_ptr());
        }
        const auto* down_completion=downCompletionForCall(down_descriptor_table_id,top_k);
        if(down_completion&&(!canonical_route_contributions||floating))
            throw std::logic_error("Down completion requires both original canonical route and routed output planes");
        float *d_output = canonical_route_contributions
                              ? nullptr
                              : d_publication_output;
        float *d_canonical_route_contributions = canonical_route_contributions
                                                     ? d_publication_output
                                                     : nullptr;

        if (!d_hidden || !d_expert_ids || !d_weights ||
            !d_prefill_gate_ || !d_prefill_up_ ||
            (!d_output && !d_canonical_route_contributions))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] missing runtime/output device pointer");
            return false;
        }

        if(down_completion)
        {
            d_output=static_cast<float*>(output->gpu_data_ptr());
            if(!d_output||d_output==d_canonical_route_contributions)
                throw std::logic_error("Down completion lost its distinct materialized original routed output");
        }

        const int max_dim = std::max(d_model, intermediate);
        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> const_up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] = d_prefill_gate_ + static_cast<size_t>(slot) * max_dim;
            up_ptrs[slot] = d_prefill_up_ + static_cast<size_t>(slot) * intermediate;
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!stageRuntimeGateUpPointerArrays(
                gateup_table.workspace_slot, RuntimePointerArrayScope::RuntimeFused,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        /*
         * Gate/up follows the same complete increasing-K dot-product contract
         * as serial decode. Partition totals are not an interchangeable result:
         * adding those totals changes FP32 grouping and therefore token bytes.
         */
        const bool reuse_router_q8_hidden =
            !floating && allow_router_q8_reuse &&
            canReuseRouterQ8Hidden(d_hidden, /*rows=*/1, d_model);
        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel", "rocm_moe_gateup_reused_router_q8_hidden_calls", 1.0, {}, {},
                {{"top_k", std::to_string(top_k)},
                 {"d_model", std::to_string(d_model)}});
        }
        int8_t *gateup_hidden_int8 = reuse_router_q8_hidden ? d_router_q8_hidden_ : d_grouped_hidden_int8_;
        float *gateup_hidden_scales = reuse_router_q8_hidden ? d_router_q8_hidden_scales_ : d_grouped_hidden_scales_;
        /*
         * NativeQuantised decode has no consumer for the intermediate FP32 gate/up
         * rows.  Publish the exact decode-equivalent Q8 SwiGLU rows directly
         * from the ordered partial reducer.  Floating-point weights retain
         * their own fixed-tree FP32 contract and therefore do not enter this
         * NativeQuantised publication path.
         */
        const bool gateup_ok = floating
            ? rocmMoE_grouped_gate_up_floating_decode_table(
                  d_hidden,
                  decode_floating_gate_descs,
                  decode_floating_up_descs,
                  d_expert_ids,
                  d_gate_ptrs,
                  d_up_ptrs,
                  top_k,
                  intermediate,
                  d_model,
                  gateup_table.num_experts,
                  gateup_table.weight_format,
                  device_ordinal_,
                  stream)
            : rocmMoE_grouped_gate_up_native_quantised_decode_table(
                  d_hidden,
                  decode_gate_descs,
                  decode_up_descs,
                  d_expert_ids,
                  d_gate_ptrs,
                  d_up_ptrs,
                  d_grouped_swiglu_int8_,
                  d_grouped_swiglu_scales_,
                  gateup_hidden_int8,
                  gateup_hidden_scales,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  reuse_router_q8_hidden,
                  top_k,
                  intermediate,
                  d_model,
                  gateup_table.num_experts,
                  gateup_table.codebook_id,
                  gateup_table.policy_codebook_mask,
                  device_ordinal_,
                  stream,gateUpCompletionForCall(gateup_descriptor_table_id,top_k));
        if (!gateup_ok)
            return false;

        const float **d_down_gate_ptrs = nullptr;
        const float **d_down_up_ptrs = nullptr;
        if (!stageRuntimeDownPointerArrays(
                down_table.workspace_slot, RuntimePointerArrayScope::RuntimeFused,
                top_k, const_gate_ptrs, const_up_ptrs,
                &d_down_gate_ptrs, &d_down_up_ptrs))
        {
            return false;
        }

        /*
         * The NativeQuantised branch consumes the Q8 rows published immediately
         * above on this exact stream.  No event or synchronization is needed
         * inside one stream; the boolean is a strict data-contract selector,
         * not permission to infer readiness from pointer non-nullness.
         */
        const bool down_ok = floating
            ? rocmMoE_grouped_swiglu_down_floating_decode_table(
                  d_down_gate_ptrs,
                  d_down_up_ptrs,
                  decode_floating_down_descs,
                  d_expert_ids,
                  d_weights,
                  d_output,
                  d_canonical_route_contributions,
                  top_k,
                  d_model,
                  intermediate,
                  down_table.num_experts,
                  down_table.weight_format,
                  device_ordinal_,
                  stream)
            : rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
                  d_down_gate_ptrs,
                  d_down_up_ptrs,
                  decode_down_descs,
                  d_expert_ids,
                  d_weights,
                  d_grouped_swiglu_int8_,
                  d_grouped_swiglu_scales_,
                  true,
                  d_output,
                  d_canonical_route_contributions,
                  d_grouped_down_partials_,
                  top_k,
                  d_model,
                  intermediate,
                  down_table.num_experts,
                  down_table.codebook_id,
                  down_table.policy_codebook_mask,
                  device_ordinal_,
                  stream,down_completion);
        if (!down_ok)
            return false;

        markDeviceWritten(
            canonical_route_contributions
                ? canonical_route_contributions
                : output,
            device,
            stream);
        recordGroupedDecodeCounter(
            "rocm_moe_grouped_decode_fused_calls",
            counter_source,
            top_k, d_model, intermediate,
            floating ? "floating_fixed_tree"
                     : "ordered_route_parallel_down");
        return true;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromTable(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        int descriptor_table_id,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            !output || descriptor_table_id < 0 || num_active <= 0 ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        const bool floating =
            deviceMoEWeightFormatIsFloating(table.weight_format);
        if (!table.valid || !table.deviceReady() || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] "
                      "descriptor table mismatch"
                      << " table_id=" << descriptor_table_id
                      << " valid=" << table.valid
                      << " device_ready=" << table.deviceReady()
                      << " format="
                      << static_cast<std::uint32_t>(table.weight_format)
                      << " table_d_model=" << table.d_model
                      << " requested_d_model=" << d_model
                      << " table_intermediate=" << table.intermediate
                      << " requested_intermediate=" << intermediate);
            return false;
        }
        if (!floating && (intermediate % 32) != 0)
            return false;

        for (int i = 0; i < num_active; ++i)
        {
            const int expert_id = expert_ids[i];
            if (expert_id < 0 || expert_id >= table.num_experts)
            {
                return false;
            }
        }

        // Table-owned floating inputs bypass quantized activation workspaces.
        if (!floating && !ensureGroupedDecodeCapacity(num_active, intermediate, d_model))
            return false;

        const int *fixed_device_expert_ids = nullptr;
        const float *fixed_device_expert_weights = nullptr;
        if (!resolveFixedTableDownMetadata(
                table.workspace_slot,
                RuntimePointerArrayScope::TableDecode,
                expert_ids,
                expert_weights,
                num_active,
                &fixed_device_expert_ids,
                &fixed_device_expert_weights))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int i = 0; i < num_active; ++i)
        {
            if (!gate_tensors[i] || !up_tensors[i])
                return false;
            gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!gate_ptrs[i] || !up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] Null gate/up device pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output && rejectDecodeStagingDuringCapture("grouped down output tensor"))
            return false;
        if (!d_output)
        {
            TransferEngine::requireDeviceOutput(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
            d_output = static_cast<float *>(output->gpu_data_ptr());
        }
        if (!d_output)
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!stageRuntimeDownPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::TableDecode,
                num_active, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
            return false;

        const bool ok = floating
            ? rocmMoE_grouped_swiglu_down_floating_decode_table(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_floating_descs,
                  fixed_device_expert_ids,
                  fixed_device_expert_weights,
                  d_output,
                  nullptr,
                  num_active,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.weight_format,
                  device_ordinal_,
                  getStream())
            : rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  fixed_device_expert_ids,
                  fixed_device_expert_weights,
                  d_grouped_swiglu_int8_,
                  d_grouped_swiglu_scales_,
                  false,
                  d_output,
                  nullptr,
                  d_grouped_down_partials_,
                  num_active,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.codebook_id,
                  table.policy_codebook_mask,
                  device_ordinal_,
                  getStream());

        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromRouting(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !routing_indices || !routing_weights ||
            !output || descriptor_table_id < 0 || top_k <= 0 ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > 16 || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedDecodeCapacity(top_k, intermediate, d_model))
            return false;

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < top_k; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] Null gate/up device pointer for slot "
                          << i);
                return false;
            }
        }

        const float *d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_routing_indices &&
            !requireTensorOnDevice(routing_indices, device, getStream(),
                                  "routing_indices", "groupedExpertDownDecodeFromRouting"))
        {
            return false;
        }
        d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_weights &&
            !requireTensorOnDevice(routing_weights, device, getStream(),
                                  "routing_weights", "groupedExpertDownDecodeFromRouting"))
        {
            return false;
        }
        d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output &&
            !requireOutputOnDevice(output, device, getStream(),
                                  "moe_output", "groupedExpertDownDecodeFromRouting"))
        {
            return false;
        }
        d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_routing_indices || !d_weights || !d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] Missing routing/output device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(top_k) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] H2D pointer staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!hipMoE_float_to_masked_int(
                    d_routing_indices,
                    d_grouped_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    getStream()))
            {
                return false;
            }
        }
        else if (!hipMoE_float_to_int(
                     d_routing_indices, d_grouped_expert_ids_, top_k,
                     device_ordinal_, getStream()))
        {
            return false;
        }

        /*
         * Routes execute concurrently into unique rows. The canonical helper
         * then folds those rows in original top-k order, including zero rows for
         * non-local experts, so masked and unmasked decode share one reduction
         * tree without floating-point atomics.
         */
        const bool ok = rocmMoE_grouped_swiglu_down_native_quantised_decode_table(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            table.device_descs,
            d_grouped_expert_ids_,
            d_weights,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            false,
            d_output,
            nullptr,
            d_grouped_down_partials_,
            top_k,
            d_model,
            intermediate,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
            markDeviceWritten(output, device, getStream());
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromRuntime(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        DeviceMoELayerRuntime *runtime_layer,
        int descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !runtime_layer || !output ||
            descriptor_table_id < 0 || top_k <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable)
            return false; // This legacy split/static API has no authenticated fragment-row contract.

        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedDecodeCapacity(top_k, intermediate, d_model))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRuntime] Null gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        const float *d_weights = runtimeTopKWeightsDevice(runtime_layer);
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_expert_ids || !d_weights || !d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRuntime] Missing runtime/output device pointer");
            return false;
        }

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!stageRuntimeDownPointerArrays(
                table.workspace_slot, RuntimePointerArrayScope::RuntimeTwoStep,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_quantised_decode_runtime(
            d_gate_ptrs,
            d_up_ptrs,
            runtime_layer,
            d_expert_ids,
            d_weights,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_output,
            d_grouped_down_partials_,
            top_k,
            d_model,
            intermediate,
            table.num_experts,
            table.codebook_id,
            table.policy_codebook_mask,
            device_ordinal_,
            getStream());

        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecode(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        const DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            !down_descs || !output || num_active <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16 || (intermediate % 32) != 0)
        {
            return false;
        }

        uint8_t codebook_id = down_descs[0].codebook_id;
        for (int i = 0; i < num_active; ++i)
        {
            if (expert_ids[i] < 0 || !down_descs[i].valid() ||
                down_descs[i].n != d_model || down_descs[i].k != intermediate ||
                down_descs[i].blocks_per_row != static_cast<uint32_t>(intermediate / 32) ||
                !areNativeQuantisedGateUpCodebooksCompatible(
                    down_descs[i].codebook_id, codebook_id) ||
                (groupedDecodeRequiresMins(down_descs[i].codebook_id) && !down_descs[i].mins) ||
                (groupedDecodeRequiresEmins(down_descs[i].codebook_id) && !down_descs[i].emins))
            {
                return false;
            }
        }

        // Preserve the old IQ4_NL/IQ4_XS mixed-down domain: the LUT decode
        // is shared, but physical24 resolves only its own raw source factors.
        for (int i = 0; i < num_active; ++i)
            if (down_descs[i].codebook_id == kNativeQuantisedIQ4XSSourceScaleCodebook)
                codebook_id = kNativeQuantisedIQ4XSSourceScaleCodebook;

        if (!groupedDecodeSupportsCodebook(codebook_id))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Unsupported native-quantised codebook "
                      << static_cast<int>(codebook_id));
            return false;
        }

        if (!ensureGroupedDecodeCapacity(num_active, intermediate, d_model))
            return false;

        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < num_active; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Null gate/up device pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Output tensor has no device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(num_active) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_decode_weights_, expert_weights,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_down_descs_, down_descs,
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeQuantisedMatrixDesc),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] H2D staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_quantised_decode(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            d_grouped_down_descs_,
            d_grouped_decode_weights_,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_grouped_down_partials_,
            d_output,
            num_active,
            d_model,
            intermediate,
            codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    // =========================================================================
    // Phase 4: GPU-side expert dispatch for prefill
    // =========================================================================

    bool ROCmMoEKernel::groupPrefillRoutes(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices, ITensor *routing_weights,
        int current_tokens, int max_tokens,
        int num_experts, int top_k,
        bool filter_to_local_runtime_experts,
        bool retain_routes_for_deferred_commit)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !routing_indices || !routing_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] null runtime or routing tensor");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] invalid dimensions current_tokens=" << current_tokens
                                                                                               << " max_tokens=" << max_tokens
                                                                                               << " num_experts=" << num_experts
                                                                                               << " top_k=" << top_k);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "groupPrefillRoutes"))
            return false;

        const float *d_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_indices || !d_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] routing tensors have no device pointers");
            return false;
        }

        return hipMoE_group_prefill_routes_runtime(
            d_indices,
            d_weights,
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            filter_to_local_runtime_experts ? 1 : 0,
            retain_routes_for_deferred_commit ? 1 : 0,
            device_ordinal_,
            getStream(),
            DeviceMoEProjectionSet::CompleteExpert);
    }

    bool ROCmMoEKernel::regroupPrefillRoutesForDiagnostics(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        bool retain_routes_for_deferred_commit)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(getStream()));
        if (!runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 ||
            current_tokens > max_tokens || num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::regroupPrefillRoutesForDiagnostics] "
                "invalid route-only diagnostic contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "regroupPrefillRoutesForDiagnostics"))
        {
            return false;
        }

        return hipMoE_regroup_prefill_routes_runtime_assignments(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            retain_routes_for_deferred_commit ? 1 : 0,
            device_ordinal_,
            getStream());
    }

    bool ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRouter(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        bool filter_to_local_runtime_experts,
        MoEGroupedPlanDemand demand,
        DeviceMoEProjectionSet expected_projections)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(getStream()));
        if (!validMoEGroupedPlanDemand(demand) || !runtime_layer || !routing_indices || !routing_weights ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
            max_tokens > std::numeric_limits<int>::max() / top_k ||
            !deviceMoEProjectionSetValid(expected_projections) ||
            gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRouter] "
                      "invalid complete-plan contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "publishCompleteGroupedPrefillPlanFromRouter"))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard))
            return false;

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts)
        {
            return false;
        }

        const int max_slots = max_tokens * top_k;
        if (max_slots > group_slots_cap_ ||
            !d_group_active_expert_ids_ ||
            !d_group_original_to_grouped_)
        {
            if (!bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_active_expert_ids_),
                    MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                    static_cast<size_t>(num_experts) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRouter(active_ids)") ||
                !bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_original_to_grouped_),
                    MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                    static_cast<size_t>(max_slots) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRouter(inverse_map)"))
            {
                return false;
            }
            group_slots_cap_ = max_slots;
        }

        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "ROCm complete runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "ROCm complete runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "ROCm complete runtime prefill down descriptors"))
        {
            return false;
        }

        const auto *device_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const auto *device_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        const int active_expert_slots = std::min(max_slots, num_experts);
        return device_indices && device_weights &&
               hipMoE_group_prefill_routes_and_materialize_plan_runtime(
                   device_indices,
                   device_weights,
                   static_cast<void *>(runtime_layer),
                   d_group_original_to_grouped_,
                   runtime_gate_descs,
                   runtime_up_descs,
                   runtime_down_descs,
                   d_group_active_expert_ids_,
                   current_tokens * top_k,
                   max_slots,
                   num_experts,
                   top_k,
                   active_expert_slots,
                   filter_to_local_runtime_experts ? 1 : 0,
                   static_cast<int>(demand),
                   gateup_table.weight_format,
                   device_ordinal_,
                   getStream(), expected_projections);
    }

    bool ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        MoEGroupedPlanDemand demand,
        DeviceMoEProjectionSet expected_projections)
    {
        grouped_prefill_grouping_.invalidate();
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!validMoEGroupedPlanDemand(demand) || !runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
            max_tokens > std::numeric_limits<int>::max() / top_k ||
            !deviceMoEProjectionSetValid(expected_projections) ||
            gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments] "
                      "invalid complete-plan contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "publishCompleteGroupedPrefillPlanFromRuntimeAssignments"))
        {
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (expected_projections == DeviceMoEProjectionSet::IntermediateShard))
            return false;

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts)
        {
            return false;
        }

        const int max_slots = max_tokens * top_k;
        if (max_slots > group_slots_cap_ ||
            !d_group_active_expert_ids_ ||
            !d_group_original_to_grouped_)
        {
            if (!bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_active_expert_ids_),
                    MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                    static_cast<size_t>(num_experts) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRuntimeAssignments(active_ids)") ||
                !bindWorkspaceBuffer(
                    reinterpret_cast<void **>(&d_group_original_to_grouped_),
                    MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                    static_cast<size_t>(max_slots) * sizeof(int),
                    "publishCompleteGroupedPrefillPlanFromRuntimeAssignments(inverse_map)"))
            {
                return false;
            }
            group_slots_cap_ = max_slots;
        }

        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "ROCm assigned runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "ROCm assigned runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "ROCm assigned runtime prefill down descriptors"))
        {
            return false;
        }

        const int active_expert_slots = std::min(max_slots, num_experts);

        return hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
                   static_cast<void *>(runtime_layer),
                   d_group_original_to_grouped_,
                   runtime_gate_descs,
                   runtime_up_descs,
                   runtime_down_descs,
                   d_group_active_expert_ids_,
                   current_tokens * top_k,
                   max_slots,
                   num_experts,
                   top_k,
                   active_expert_slots,
                   static_cast<int>(demand),
                   gateup_table.weight_format,
                   device_ordinal_,
                   getStream(), expected_projections);
    }

    bool ROCmMoEKernel::commitGroupedVerifierHistograms(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        int total_rows,
        int num_experts,
        int top_k)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "commitGroupedVerifierHistograms");
        if (!stream)
            return false;
        if (!runtime_layer ||
            !accepted_state_counts_device ||
            !publication_ok_flags_device ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            total_rows != request_count * rows_per_request ||
            num_experts <= 0 ||
            top_k <= 0)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::commitGroupedVerifierHistograms] invalid "
                "device publication contract");
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "commitGroupedVerifierHistograms"))
        {
            return false;
        }

        return hipMoE_commit_grouped_verifier_histograms(
            static_cast<void *>(runtime_layer),
            accepted_state_counts_device,
            publication_ok_flags_device,
            request_count,
            rows_per_request,
            total_rows,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::assignPrefillRoutesLeastLoadedResident(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens, int num_experts, int top_k,
        const int32_t *absolute_position_ids_device,
        const int32_t *active_row_count_device)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesLeastLoadedResident");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));

        if (!runtime_layer || !absolute_position_ids_device ||
            !active_row_count_device)
        {
            LOG_ERROR(
                "[ROCmMoEKernel::assignPrefillRoutesLeastLoadedResident] "
                "runtime, device position row, and device active-row count "
                "must be non-null");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesLeastLoadedResident] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "assignPrefillRoutesLeastLoadedResident"))
            return false;

        return hipMoE_assign_prefill_routes_least_loaded_resident(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            absolute_position_ids_device,
            active_row_count_device,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens, int num_experts, int top_k,
        const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "planPrefillRoutesLeastLoadedCurrentBatch");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));

        if (!runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (config.expert_count != static_cast<uint32_t>(num_experts) ||
            config.participant_count == 0u ||
            config.alpha_numerator == 0u ||
            config.alpha_denominator == 0u ||
            config.lambda_numerator == 0u ||
            config.lambda_denominator == 0u)
        {
            LOG_ERROR("[ROCmMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid LLEP config");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "planPrefillRoutesLeastLoadedCurrentBatch"))
            return false;

        return hipMoE_plan_prefill_routes_least_loaded_current_batch(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            config.min_chunk_tokens,
            config.alpha_numerator,
            config.alpha_denominator,
            config.lambda_numerator,
            config.lambda_denominator,
            config.min_spread_improvement,
            config.min_spread_improvement_divisor,
            config.min_spread_improvement_per_critical_path_slot,
            config.min_foreign_rows_per_critical_path_slot,
            config.max_weight_transfers,
            config.max_non_owner_experts_per_participant,
            config.enable_balanced_skip ? 1 : 0,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers");
        if (!stream)
            return false;
        if (!runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers"))
            return false;

        return hipMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const DeviceMoERebalanceStatus *transfer_status,
        const DeviceMoERebalanceApplyStatus *apply_status)
    {
        void *stream = explicitMoELaunchStream(
            launch,
            "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers");
        if (!stream)
            return false;
        if (!runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] null runtime");
            return false;
        }
        if (!transfer_status || !apply_status)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] transfer and apply status are required");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers"))
            return false;
        if (!validateDevicePointerOrLog(
                runtime_layer,
                device_ordinal_,
                "runtime_layer",
                "ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers") ||
            !validateDevicePointerOrLog(
                transfer_status,
                device_ordinal_,
                "transfer_status",
                "ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers") ||
            !validateDevicePointerOrLog(
                apply_status,
                device_ordinal_,
                "apply_status",
                "ROCmMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers"))
        {
            return false;
        }

        return hipMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            transfer_status,
            apply_status,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *batch_buffer,
        int expert_id, int max_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !hidden || !batch_buffer)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] invalid arguments expert_id=" << expert_id
                                                                                                          << " max_tokens=" << max_tokens
                                                                                                          << " d_model=" << d_model);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "gatherPrefillExpertBatchFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_batch = static_cast<float *>(batch_buffer->gpu_data_ptr());
        if (!d_hidden || !d_batch)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = hipMoE_prefill_gather_expert_runtime(
            static_cast<const void *>(runtime_layer),
            d_hidden,
            d_batch,
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            getStream());
        if (ok)
        {
            markDeviceWritten(
                batch_buffer,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    bool ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime(
        ITensor *output, ITensor *expert_results,
        DeviceMoELayerRuntime *runtime_layer,
        int expert_id, int max_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (!output || !expert_results || !runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] invalid arguments expert_id=" << expert_id
                                                                                                             << " max_tokens=" << max_tokens
                                                                                                             << " d_model=" << d_model);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "scatterPrefillExpertResultsFromRuntime"))
            return false;

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const float *d_expert_output = static_cast<const float *>(expert_results->gpu_data_ptr());
        if (!d_output || !d_expert_output)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = hipMoE_prefill_scatter_expert_runtime(
            d_output,
            d_expert_output,
            static_cast<const void *>(runtime_layer),
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            getStream());
        if (ok)
        {
            markDeviceWritten(
                output,
                DeviceId::rocm(device_ordinal_),
                getStream());
        }
        return ok;
    }

    // =========================================================================
    // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
    // =========================================================================

    bool ROCmMoEKernel::prepareExpertGroupsAsync(
        ITensor *routing_indices, ITensor *routing_weights,
        int seq_len, int num_experts, int top_k)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginAllRoutes();
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "prepareExpertGroupsAsync"))
            return false;

        const int total_slots = seq_len * top_k;

        // 1. Join both routing producers onto this planner's explicit stream.
        //
        // A transfer that omits the consumer stream may return before this
        // planner reads the route table. The stream-aware helper either uploads
        // on this stream or joins the tensor's existing producer event, making
        // that race structurally impossible.
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();
        if (!requireTensorOnDevice(
                routing_indices,
                device,
                stream,
                "routing_indices",
                "prepareExpertGroupsAsync") ||
            !requireTensorOnDevice(
                routing_weights,
                device,
                stream,
                "routing_weights",
                "prepareExpertGroupsAsync"))
        {
            return false;
        }

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] null device pointers");
            return false;
        }

        // 2. Lazy-allocate grouping buffers
        if (total_slots > group_slots_cap_ ||
            !d_group_int_indices_ ||
            !d_group_token_indices_ ||
            !d_group_original_to_grouped_ ||
            !d_group_weights_ ||
            !d_group_active_expert_ids_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_original_to_grouped)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(total_slots) * sizeof(float),
                                     "prepareExpertGroupsAsync(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_original_to_grouped_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_active_expert_slots_ = 0;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareExpertGroupsAsync(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = num_experts;
        }

        const bool use_small_grouping =
            total_slots <= 64 &&
            num_experts <= static_cast<int>(kDeviceMoEMaxExperts) &&
            top_k <= 16;
        if (use_small_grouping)
        {
            const int active_expert_slots = std::min(total_slots, num_experts);
            /*
             * Small verifier batches use a single graph-capturable planner
             * kernel that publishes every piece of route metadata consumed by
             * grouped execution.  d_group_int_indices_ is deliberately passed
             * as original route-slot expert ids; executeGroupedPrefillPipeline
             * later gives that same buffer to the parallel down publication
             * kernel when ROCm serial decode is using parallel down.
             */
            if (!hipMoE_group_tokens_small_float(
                    d_float_indices,
                    d_float_weights,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_original_to_grouped_,
                    d_group_int_indices_,
                    d_group_weights_,
                    d_group_active_expert_ids_,
                    total_slots,
                    num_experts,
                    top_k,
                    active_expert_slots,
                    device_ordinal_,
                    getStream()))
            {
                LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] small-M grouping failed");
                group_active_expert_slots_ = 0;
                return false;
            }
            group_active_expert_slots_ = active_expert_slots;

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "rocm_moe_small_prefill_grouping_calls",
                    1.0,
                    "moe",
                    DeviceId::rocm(device_ordinal_).to_string(),
                    PerfStatsCollector::Tags{
                        {"total_slots", std::to_string(total_slots)},
                        {"num_experts", std::to_string(num_experts)},
                        {"top_k", std::to_string(top_k)}});
            }

            std::move(grouping_publication).publish();
            return true;
        }

        // 3. Convert float indices → int on device
        group_active_expert_slots_ = 0;
        if (!hipMoE_float_to_int(d_float_indices, d_group_int_indices_,
                                 total_slots, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] float_to_int failed");
            return false;
        }

        // 4. Group tokens by expert (counts, offsets, scatter) — all on device
        if (!groupTokensByExpertDevice(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] groupTokensByExpertDevice failed");
            return false;
        }
        group_active_expert_slots_ = std::min(total_slots, num_experts);

        // Launch device-side max-reduction over expert counts (async, no sync).
        // Result in d_group_max_tokens_ — consumed by GEMM grid early-exit.
        if (!hipMoE_max_expert_count(d_group_counts_, d_group_max_tokens_,
                                     num_experts, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] max_expert_count failed");
            return false;
        }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
        {
            /*
             * Wider runtime-M verifier groups intentionally use the scalable
             * count/scan/scatter planner. Publishing this route separately
             * lets correctness gates prove that a missing compact-planner
             * counter is a policy transition, not an unobserved fallback.
             */
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_general_prefill_grouping_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"total_slots", std::to_string(total_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"top_k", std::to_string(top_k)}});
        }

        // NO D2H copy, NO hipStreamSynchronize — data stays on device
        // for consumption by executeGroupedPrefillPipeline()

        std::move(grouping_publication).publish();
        return true;
    }

    bool ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask(
        ITensor *routing_indices,
        ITensor *routing_weights,
        int seq_len,
        int num_experts,
        int top_k)
    {
        auto grouping_publication = grouped_prefill_grouping_.beginSelectedRoutes();
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(
                device_ordinal_,
                "prepareExpertGroupsAsyncUsingPublishedMask"))
            return false;

        const int total_slots = seq_len * top_k;
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();
        if (!requireTensorOnDevice(
                routing_indices,
                device,
                stream,
                "routing_indices",
                "prepareExpertGroupsAsyncUsingPublishedMask") ||
            !requireTensorOnDevice(
                routing_weights,
                device,
                stream,
                "routing_weights",
                "prepareExpertGroupsAsyncUsingPublishedMask"))
        {
            return false;
        }

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "null device pointers");
            return false;
        }

        if (total_slots > group_slots_cap_ ||
            !d_group_int_indices_ ||
            !d_group_token_indices_ ||
            !d_group_original_to_grouped_ ||
            !d_group_weights_ ||
            !d_group_active_expert_ids_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_original_to_grouped)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(total_slots) * sizeof(float),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_original_to_grouped_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_active_expert_slots_ = 0;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareExpertGroupsAsyncUsingPublishedMask(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = num_experts;
        }

        /*
         * The graph path consumes only the backend-owned mask. Missing
         * publication is a lifecycle error, not an invitation to stage a host
         * pointer while HIP is recording graph nodes.
         */
        if (!group_expert_mask_published_ ||
            !d_group_expert_mask_ ||
            group_expert_mask_cap_ < num_experts ||
            group_expert_mask_num_experts_ != num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "fixed-topology expert mask was not published for the current "
                      "workspace and expert count before graph execution"
                      << " requested_experts=" << num_experts
                      << " published_experts=" << group_expert_mask_num_experts_
                      << " capacity=" << group_expert_mask_cap_);
            return false;
        }

        group_active_expert_slots_ = 0;
        if (!hipMoE_float_to_masked_int(
                d_float_indices,
                d_group_int_indices_,
                d_group_expert_mask_,
                total_slots,
                num_experts,
                device_ordinal_,
                getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "float_to_masked_int failed");
            return false;
        }

        if (!groupTokensByExpertDevice(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "groupTokensByExpertDevice failed");
            return false;
        }
        /*
         * Masked LocalTP grouping emits a compact active-id prefix followed by
         * -1 padding.  Bound the stable grouped grid by the number of experts
         * this participant can own instead of launching planes for every global
         * expert.  This mirrors CUDA and keeps masked publication economical.
         */
        group_active_expert_slots_ =
            std::min(total_slots, group_expert_mask_active_experts_);

        if (!hipMoE_max_expert_count(d_group_counts_, d_group_max_tokens_,
                                     num_experts, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask] "
                      "max_expert_count failed");
            return false;
        }

        PerfStatsCollector::addCounter(
            "kernel", "rocm_moe_masked_prefill_grouping_calls", 1.0, {}, {},
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"num_experts", std::to_string(num_experts)},
             {"mask_active_experts", std::to_string(group_expert_mask_active_experts_)},
             {"expert_grid_slots", std::to_string(group_active_expert_slots_)}});
        std::move(grouping_publication).publish();
        return true;
    }

    bool ROCmMoEKernel::updateGroupedPrefillExpertMask(
        const uint8_t *expert_mask,
        int num_experts)
    {
        if (num_experts <= 0 || !expert_mask)
            return false;

        if (!setMoEDevice(device_ordinal_, "updateGroupedPrefillExpertMask"))
            return false;
        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                      "an explicit stream is required");
            return false;
        }

        uint64_t mask_hash = 1469598103934665603ull;
        int active_experts = 0;
        for (int i = 0; i < num_experts; ++i)
        {
            mask_hash ^= static_cast<uint64_t>(expert_mask[i]);
            mask_hash *= 1099511628211ull;
            active_experts += expert_mask[i] != 0u ? 1 : 0;
        }
        mask_hash ^= static_cast<uint64_t>(num_experts);
        mask_hash *= 1099511628211ull;

        /*
         * Warmup publishes immutable participant ownership once. Graph
         * construction may reuse that exact device table, but any attempt to
         * change or allocate it while capture is active remains fatal.
         */
        if (group_expert_mask_published_ &&
            group_expert_mask_hash_ == mask_hash &&
            group_expert_mask_num_experts_ == num_experts)
            return true;

        if (isGraphCaptureActive() || isHipStreamCapturing(stream))
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                      "host expert-mask publication is forbidden during graph capture");
            return false;
        }

        if (!d_group_expert_mask_ || group_expert_mask_cap_ < num_experts)
        {
            if (!workspace_)
            {
                LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                          "graph-owned workspace is required before mask publication");
                return false;
            }

            auto lease = workspace_->acquirePersistentSlot(
                kROCmGroupedExpertMaskLeaseDomain,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
            void *mask_table = nullptr;
            const size_t required_table_bytes =
                static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) *
                static_cast<size_t>(num_experts) * sizeof(uint8_t);
            if (!lease ||
                !bindWorkspaceBuffer(
                    &mask_table,
                    MoEWorkspaceBuffers::GROUP_EXPERT_MASK,
                    required_table_bytes,
                    "updateGroupedPrefillExpertMask(group_expert_mask_table)"))
            {
                d_group_expert_mask_ = nullptr;
                group_expert_mask_workspace_lease_.reset();
                group_expert_mask_cap_ = 0;
                group_expert_mask_hash_ = 0;
                group_expert_mask_num_experts_ = 0;
                group_expert_mask_active_experts_ = 0;
                group_expert_mask_published_ = false;
                return false;
            }

            const size_t table_bytes =
                workspace_->getBufferSize(MoEWorkspaceBuffers::GROUP_EXPERT_MASK);
            const size_t slot_stride =
                table_bytes /
                static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
            if (slot_stride < static_cast<size_t>(num_experts))
            {
                LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] "
                          "leased expert-mask slot is too narrow: stride="
                          << slot_stride << " requested_experts=" << num_experts);
                d_group_expert_mask_ = nullptr;
                group_expert_mask_workspace_lease_.reset();
                group_expert_mask_cap_ = 0;
                group_expert_mask_published_ = false;
                return false;
            }

            d_group_expert_mask_ =
                static_cast<uint8_t *>(mask_table) + lease->slot() * slot_stride;
            group_expert_mask_workspace_lease_ = std::move(lease);
            group_expert_mask_cap_ = static_cast<int>(slot_stride);
            group_expert_mask_hash_ = 0;
            group_expert_mask_num_experts_ = 0;
            group_expert_mask_active_experts_ = 0;
            group_expert_mask_published_ = false;
        }

        hipError_t err = hipMemcpyAsync(
            d_group_expert_mask_,
            expert_mask,
            static_cast<size_t>(num_experts) * sizeof(uint8_t),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::updateGroupedPrefillExpertMask] H2D mask copy failed: "
                      << hipGetErrorString(err));
            return false;
        }

        group_expert_mask_hash_ = mask_hash;
        group_expert_mask_num_experts_ = num_experts;
        group_expert_mask_active_experts_ = active_experts;
        group_expert_mask_published_ = true;
        return true;
    }

    bool ROCmMoEKernel::prepareSharedExpertPrefillGroup(DeviceRowRange rows)
    {
        grouped_prefill_grouping_.invalidate();
        const int seq_len = rows.physicalRows();
        if (seq_len <= 0)
            return false;
        if (!setMoEDevice(device_ordinal_, "prepareSharedExpertPrefillGroup"))
            return false;

        if (seq_len > group_slots_cap_ ||
            !d_group_int_indices_ ||
            !d_group_token_indices_ ||
            !d_group_original_to_grouped_ ||
            !d_group_weights_ ||
            !d_group_active_expert_ids_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(seq_len) * sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(seq_len) * sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(seq_len) * sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_original_to_grouped)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(seq_len) * sizeof(float),
                                     "prepareSharedExpertPrefillGroup(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_original_to_grouped_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = seq_len;
        }
        if (group_experts_cap_ < 1)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareSharedExpertPrefillGroup(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = 1;
        }

        if (!hipMoE_prepare_shared_expert_group(
                d_group_offsets_,
                d_group_counts_,
                d_group_token_indices_,
                d_group_original_to_grouped_,
                d_group_weights_,
                d_group_active_expert_ids_,
                rows,
                device_ordinal_,
                getStream()))
        {
            return false;
        }

        group_active_expert_slots_ = 1;
        if (PerfStatsCollector::isDomainEnabled("kernel"))
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_shared_expert_prefill_group_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"seq_len", std::to_string(seq_len)},
                    {"top_k", "1"},
                    {"active_expert_slots", "1"}});
        }
        return true;
    }

    /**
     * @brief Resolve preallocated grouped scratch without changing arena size.
     * @param total_slots Physical routed-row capacity of the captured graph.
     * @param d_model Hidden projection width, including floating-point tails.
     * @param intermediate Expert intermediate width.
     * @return True when every required named arena region has sufficient space.
     */
    bool ROCmMoEKernel::ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate)
    {
        if (!setMoEDevice(device_ordinal_, "ensureGroupedPrefillScratchCapacity"))
            return false;

        const bool need_realloc = (total_slots > prefill_slots_cap_ ||
                                   d_model > prefill_d_model_cap_ ||
                                   intermediate > prefill_intermediate_cap_);
        if (!need_realloc &&
            d_prefill_A_int8_ &&
            d_prefill_A_scales_ &&
            d_prefill_swiglu_int8_ &&
            d_prefill_swiglu_scales_ &&
            d_prefill_gate_ &&
            d_prefill_up_)
        {
            return true;
        }

        const int max_dim = (d_model > intermediate) ? d_model : intermediate;
        // Match the workspace BOM and CUDA binding. Floating matrices need not
        // be multiples of a quantized block; flooring here can request a zero
        // length binding even though the arena already owns the padded region.
        const int max_blocks = (max_dim + 31) / 32;
        const int intermediate_blocks = (intermediate + 31) / 32;

        // Hidden-row Q8 data remains live through gate/up, while the exact
        // SwiGLU Q8 rows remain live through down projection. Those lifetimes
        // overlap across CTAs, so they require distinct device buffers. Once
        // SwiGLU has consumed the FP32 gate rows, long-prefill down projection
        // reuses the larger gate allocation for deterministic route partials;
        // the ordered publisher then folds those rows in original top-k order.
        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_A_int8_),
                                 MoEWorkspaceBuffers::PREFILL_A_INT8,
                                 static_cast<size_t>(total_slots) * max_dim * sizeof(int8_t),
                                 "ensureGroupedPrefillScratchCapacity(a_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_A_scales_),
                                 MoEWorkspaceBuffers::PREFILL_A_SCALES,
                                 static_cast<size_t>(total_slots) * max_blocks * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(a_scales)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_swiglu_int8_),
                                 MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,
                                 static_cast<size_t>(total_slots) * intermediate * sizeof(int8_t),
                                 "ensureGroupedPrefillScratchCapacity(swiglu_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_swiglu_scales_),
                                 MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                                 static_cast<size_t>(total_slots) * intermediate_blocks * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(swiglu_scales)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_gate_),
                                 MoEWorkspaceBuffers::PREFILL_GATE,
                                 static_cast<size_t>(total_slots) * max_dim * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(gate)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_up_),
                                 MoEWorkspaceBuffers::PREFILL_UP,
                                 static_cast<size_t>(total_slots) * intermediate * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(up)"))
        {
            d_prefill_A_int8_ = nullptr;
            d_prefill_A_scales_ = nullptr;
            d_prefill_swiglu_int8_ = nullptr;
            d_prefill_swiglu_scales_ = nullptr;
            d_prefill_gate_ = nullptr;
            d_prefill_up_ = nullptr;
            prefill_slots_cap_ = 0;
            prefill_d_model_cap_ = 0;
            prefill_intermediate_cap_ = 0;
            return false;
        }

        prefill_slots_cap_ = total_slots;
        prefill_d_model_cap_ = d_model;
        prefill_intermediate_cap_ = intermediate;
        return true;
    }

    /**
     * @brief Resolve the dedicated graph-persistent adaptive work directory.
     *
     * The directory is not interchangeable with the compact active-expert-id
     * list: the adaptive planner publishes two live counts and two independent
     * tile spans into it on every replay. Binding the named region here keeps
     * that mutable lifetime explicit and prevents future grouping code from
     * aliasing a buffer embedded in a captured projection graph.
     */
    bool ROCmMoEKernel::ensureGroupedPrefillWorkDirectoryCapacity(
        int total_slots,
        int num_experts)
    {
        if (total_slots <= 0 || num_experts <= 0)
            return false;
        const std::size_t words =
            MoEWorkspaceBuffers::rocmAdaptivePrefillDirectoryWords(
                static_cast<std::size_t>(total_slots),
                static_cast<std::size_t>(num_experts));
        if (!bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_prefill_work_directory_),
                MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY,
                words * sizeof(uint32_t),
                "ensureGroupedPrefillWorkDirectoryCapacity"))
        {
            d_prefill_work_directory_ = nullptr;
            return false;
        }
        return true;
    }

    bool ROCmMoEKernel::exportGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const std::int32_t *route_participants, ITensor *packet, int participant)
    {
        // These pointers are owned by the already-admitted grouped workspace.
        // Do not expose names or pointer arithmetic to model graph callers.
        if (!layout.valid() || !route_participants || !packet ||
            packet->size_bytes() < layout.packetBytes() ||
            participant < 0 || static_cast<std::uint32_t>(participant) >= layout.participants ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !d_group_original_to_grouped_ || !setMoEDevice(device_ordinal_, "exportGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireOutputOnDevice(packet, device, stream, "packet", "exportGroupedPrefillIntermediates"))
            return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        const MoEGroupedIntermediatePackLaunch launch{
            .layout = layout,
            .route_owners = route_participants,
            .original_to_grouped = d_group_original_to_grouped_,
            .grouped_values = floating ? static_cast<const void *>(d_prefill_gate_)
                                      : static_cast<const void *>(d_prefill_swiglu_int8_),
            .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_,
            .packet = static_cast<std::uint32_t *>(packet->gpu_data_ptr()),
            .participant = participant};
        if (!rocm::packGroupedIntermediate(launch, stream)) return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool ROCmMoEKernel::importGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const std::int32_t *route_participants, ITensor *packets)
    {
        if (!layout.valid() || !route_participants || !packets ||
            packets->size_bytes() < layout.packetBytes() * layout.participants ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !d_group_original_to_grouped_ || !setMoEDevice(device_ordinal_, "importGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireTensorOnDevice(packets, device, stream, "packets", "importGroupedPrefillIntermediates"))
            return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        const MoEGroupedIntermediateConsumeLaunch launch{
            .layout = layout,
            .route_owners = route_participants,
            .original_to_grouped = d_group_original_to_grouped_,
            .participant_packets = static_cast<const std::uint32_t *>(packets->gpu_data_ptr()),
            .grouped_values = floating ? static_cast<void *>(d_prefill_gate_)
                                      : static_cast<void *>(d_prefill_swiglu_int8_),
            .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_};
        return rocm::consumeGroupedIntermediate(launch, stream);
    }

    bool ROCmMoEKernel::exportCompactGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const DeviceMoELayerRuntime &runtime, ITensor *packet,
        std::uint64_t *packet_bytes)
    {
        if (!layout.compactValid() || !packet || !packet_bytes ||
            reinterpret_cast<std::uintptr_t>(packet_bytes) % alignof(std::uint64_t) ||
            packet->size_bytes() < layout.compactCapacityBytes() ||
            runtime.participant_count != layout.participants || runtime.participant_id >= layout.participants ||
            !runtime.expert_count || runtime.expert_count > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            runtime.prefill_route_capacity < layout.route_capacity ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !runtime.expert_offsets || !runtime.expert_counts || !runtime.route_participant_ids ||
            !d_group_original_to_grouped_ ||
            !setMoEDevice(device_ordinal_, "exportCompactGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireOutputOnDevice(packet, device, stream, "packet", "exportCompactGroupedPrefillIntermediates")) return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        // Runtime-plan execution reads runtime-owned counts, not the kernel's
        // private grouping scratch. That scratch may still describe a prior
        // layer. Bind the same authoritative addresses used by gate/up, with
        // the actual expert geometry rather than the allocation's capacity.
        const MoECompactIntermediatePackLaunch launch{
            .payload = {
                .layout = layout,
                .route_owners = runtime.route_participant_ids,
                .original_to_grouped = d_group_original_to_grouped_,
                .grouped_values = floating ? static_cast<const void *>(d_prefill_gate_)
                                          : static_cast<const void *>(d_prefill_swiglu_int8_),
                .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_,
                .packet = static_cast<std::uint32_t *>(packet->gpu_data_ptr()),
                .participant = static_cast<std::int32_t>(runtime.participant_id)},
            .last_group_offset = runtime.expert_offsets + runtime.expert_count - 1,
            .last_group_count = runtime.expert_counts + runtime.expert_count - 1,
            .packet_bytes = packet_bytes};
        if (!rocm::packCompactIntermediate(launch, stream)) return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool ROCmMoEKernel::importCompactGroupedPrefillIntermediates(
        const MoEGroupedIntermediateLayout &layout,
        const std::int32_t *route_participants, ITensor *packet,
        std::size_t packet_byte_offset, const std::uint64_t *packet_bytes, int participant)
    {
        if (!layout.compactValid() || !route_participants || !packet || !packet_bytes ||
            reinterpret_cast<std::uintptr_t>(packet_bytes) % alignof(std::uint64_t) ||
            packet_byte_offset % alignof(std::uint32_t) || packet_byte_offset > packet->size_bytes() ||
            layout.compactCapacityBytes() > packet->size_bytes() - packet_byte_offset ||
            participant < 0 || static_cast<std::uint32_t>(participant) >= layout.participants ||
            layout.route_capacity > static_cast<std::uint32_t>(group_slots_cap_) ||
            layout.route_capacity > static_cast<std::uint32_t>(prefill_slots_cap_) ||
            layout.columns > static_cast<std::uint32_t>(prefill_intermediate_cap_) ||
            !d_group_original_to_grouped_ ||
            !setMoEDevice(device_ordinal_, "importCompactGroupedPrefillIntermediates"))
            return false;
        const auto device = DeviceId::rocm(device_ordinal_);
        auto *stream = getStream();
        if (!stream || !requireTensorOnDevice(packet, device, stream, "packet", "importCompactGroupedPrefillIntermediates")) return false;
        const bool floating = layout.encoding == MoEGroupedIntermediateEncoding::FP32;
        // Regrouping has already rebound the original router slots. Import
        // changes only this producer's live rows, so peers and empty packets
        // cannot overwrite each other's contributions or any unused capacity.
        // The offset selects bytes, not a new tensor/coherence authority. The
        // capture ledger above authenticates the published arena owner itself.
        const MoECompactIntermediateConsumeLaunch launch{
            .layout = layout,
            .route_owners = route_participants,
            .original_to_grouped = d_group_original_to_grouped_,
            .packet = static_cast<const std::uint32_t *>(packet->gpu_data_ptr()) + packet_byte_offset / sizeof(std::uint32_t),
            .packet_bytes = packet_bytes,
            .participant = participant,
            .grouped_values = floating ? static_cast<void *>(d_prefill_gate_)
                                      : static_cast<void *>(d_prefill_swiglu_int8_),
            .grouped_scales = floating ? nullptr : d_prefill_swiglu_scales_};
        return rocm::consumeCompactIntermediate(launch, stream);
    }

    bool ROCmMoEKernel::executeGroupedPrefillProjection(
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        const MoEPrefillProjectionExecution &execution,
        ITensor *canonical_route_contributions)
    {
        if (execution.isIntermediateShard() ||
            (gateup_desc_table_id >= 0 && static_cast<size_t>(gateup_desc_table_id) < grouped_gateup_desc_tables_.size() &&
             grouped_gateup_desc_tables_[gateup_desc_table_id].descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) ||
            (down_desc_table_id >= 0 && static_cast<size_t>(down_desc_table_id) < grouped_down_desc_tables_.size() &&
             grouped_down_desc_tables_[down_desc_table_id].descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable))
        {
            LOG_ERROR("Intermediate-shard partial execution requires the acquired runtime plan and original cohort binding");
            return false;
        }

        const auto *owner_admission = ownerGateUpAdmissionForCall(gateup_desc_table_id, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model});
        rocm::MoERouteOwnedGateUpLaunchSet owner_completed{};
        rocm::validateMoERouteOwnedGateUpScopeBinding(device_ordinal_, gpu_stream_, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model}, owner_admission);
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0 ||
            seq_len > std::numeric_limits<int>::max() / top_k ||
            execution.modelColumns() != d_model)
            return false;

        if (!setMoEDevice(device_ordinal_, "executeGroupedPrefillProjection"))
            return false;

        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] invalid gateup descriptor table id "
                      << gateup_desc_table_id);
            return false;
        }
        if (down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] invalid down descriptor table id "
                      << down_desc_table_id);
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        rocm::validateMoERouteOwnedGateUpCodebook(gateup_table.codebook_mask, gateup_table.policy_codebook_mask, owner_admission);
        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            (execution.executesDown() && down_table.d_model != execution.columnCount()) ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] descriptor tables not valid");
            return false;
        }

        const int total_slots = seq_len * top_k;
        const int active_expert_slots = group_active_expert_slots_;

        // Ensure scratch buffers
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate))
            return false;
        if (!ensureGroupedPrefillWorkDirectoryCapacity(
                total_slots, num_experts))
            return false;

        // Join the hidden-state producer and the exact publication target to
        // this pipeline. Canonical LocalTP publication does not own or inspect
        // the later reducer destination.
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();
        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if ((execution.executesGateUp() && !requireTensorOnDevice(
                hidden,
                device,
                stream,
                "hidden",
                "executeGroupedPrefillProjection")) ||
            (execution.executesDown() && !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name,
                "executeGroupedPrefillProjection")))
        {
            return false;
        }

        const float *d_hidden = execution.executesGateUp()
            ? static_cast<const float *>(hidden->gpu_data_ptr()) : nullptr;
        float *d_output = !execution.executesDown() || canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            execution.executesDown() && canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;

        if (isGraphCaptureActive() &&
            ((execution.executesGateUp() && !d_hidden) ||
             (execution.executesDown() && !d_output && !d_canonical_route_contributions)))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] null device pointers during graph capture");
            return false;
        }

        const bool shared_singleton_expert = (num_experts == 1 && top_k == 1);
        const int *d_original_expert_ids_for_pipeline =
            shared_singleton_expert ? nullptr : d_group_int_indices_;
        if (floating)
        {
            if (!d_group_original_to_grouped_ || !d_group_weights_ ||
                (!shared_singleton_expert &&
                 !d_original_expert_ids_for_pipeline))
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] "
                          "floating grouped-prefill route publication is incomplete");
                return false;
            }
            const bool ok = execution.overwriteOutputThenPublish(
                execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
                execution.executesDown() ? publication_output->size_bytes() : 0u,
                seq_len, top_k,
                canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                              : MoEPrefillOutputLayout::TokenRows,
                stream,
                [&](std::span<float> live_output, void *producer_stream)
                {
                    return rocmMoE_grouped_floating_prefill_pipeline(
                        d_hidden,
                        gateup_table.device_floating_gate_descs,
                        gateup_table.device_floating_up_descs,
                        down_table.device_floating_descs,
                        d_group_original_to_grouped_,
                        d_original_expert_ids_for_pipeline,
                        d_group_weights_,
                        d_prefill_gate_,
                        canonical_route_contributions ? nullptr : live_output.data(),
                        canonical_route_contributions ? live_output.data() : nullptr,
                        seq_len,
                        total_slots,
                        top_k,
                        d_model,
                        intermediate,
                        num_experts,
                        gateup_table.weight_format,
                        device_ordinal_,
                        producer_stream,
                        execution);
                },
                [&](const EnqueuedWorkspaceRead<float> &read)
                {
                    markDeviceWritten(publication_output, device, read.stream());
                    return true;
                });
            if (!ok)
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] "
                          "floating grouped ROCm pipeline failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_floating_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"active_expert_slots", std::to_string(active_expert_slots)},
                 {"descriptor_source", "static_table"},
                 {"weight_format", std::to_string(static_cast<uint32_t>(
                                       gateup_table.weight_format))}});
            return true;
        }

        const bool reuse_router_q8_hidden = execution.executesGateUp() &&
            canReuseRouterQ8Hidden(d_hidden, seq_len, d_model);
        if (execution.executesGateUp() && router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::RequiredConsumer &&
            !reuse_router_q8_hidden)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] required "
                      "router Q8 publication is unavailable"
                      << " rows=" << seq_len
                      << " d_model=" << d_model);
            return false;
        }
        /*
         * Ordered publication is the sole grouped-prefill contract. One lane
         * owns each token/column and accumulates routes in top-k order. Small
         * verifier buckets compute each route directly; long prefill first
         * materializes deterministic route partials so decoded weights can be
         * reused across expert-row tiles. Neither variant relies on a
         * scheduler-dependent floating-point atomic reduction.
         */
        const bool ordered_scatter_overwrites_output =
            active_expert_slots > 0 && d_group_original_to_grouped_ != nullptr;
        if (!ordered_scatter_overwrites_output ||
            (!shared_singleton_expert && !d_group_int_indices_))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] "
                      "batch-invariant grouped prefill requires the original route map and expert ids");
            return false;
        }
        // Launch the fixed, fully grouped pipeline without a host synchronization.
        const bool ok = execution.overwriteOutputThenPublish(
            execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
            execution.executesDown() ? publication_output->size_bytes() : 0u,
            seq_len, top_k,
            canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                          : MoEPrefillOutputLayout::TokenRows,
            stream,
            [&](std::span<float> live_output, void *producer_stream)
            {
                return rocmMoE_grouped_prefill_pipeline(
                    d_hidden,
                    reuse_router_q8_hidden
                        ? router_q8_hidden_publication_->quantized_rows
                        : nullptr,
                    reuse_router_q8_hidden
                        ? router_q8_hidden_publication_->row_scales
                        : nullptr,
                    gateup_table.device_gate_descs,
                    gateup_table.device_up_descs,
                    down_table.device_descs,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_original_to_grouped_,
                    d_original_expert_ids_for_pipeline,
                    d_group_weights_,
                    reinterpret_cast<int *>(d_prefill_work_directory_),
                    d_prefill_A_int8_,
                    d_prefill_A_scales_,
                    d_prefill_gate_,
                    d_prefill_up_,
                    d_prefill_swiglu_int8_,
                    d_prefill_swiglu_scales_,
                    canonical_route_contributions ? nullptr : live_output.data(),
                    canonical_route_contributions ? live_output.data() : nullptr,
                    num_experts,
                    d_model,
                    intermediate,
                    total_slots,
                    top_k,
                    0,
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    gateup_table.policy_codebook_mask,
                    down_table.policy_codebook_mask,
                    device_ordinal_,
                    producer_stream,
                    execution,
                    grouped_prefill_grouping_.scope(), owner_admission,
                    owner_admission ? &owner_completed : nullptr);
            },
            [&](const EnqueuedWorkspaceRead<float> &read)
            {
                markDeviceWritten(publication_output, device, read.stream());
                return true;
            });

        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjection] pipeline failed for layer");
            return false;
        }

        if (owner_admission)
            recordOwnerGateUpCaptureSubmissions(owner_completed, gateup_table.owner_m1_intent);

        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_router_q8_reuse_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", "static_table"}});
        }

            if (PerfStatsCollector::isDomainEnabled("kernel") &&
                active_expert_slots > 0)
        {
            const MoEPrefillPairPolicyTags policy =
                queryMoEPrefillPairPolicyTags(
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    seq_len,
                    d_model,
                    intermediate,
                    num_experts,
                    top_k,
                    execution.measuredOperation(
                        canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                                      : MoEPrefillOutputLayout::TokenRows,
                        grouped_prefill_grouping_.scope()));
            const std::string common_row_tile =
                policy.gateup.tile_m == policy.down.tile_m
                    ? policy.gateup.tile_m
                    : "mixed";
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_batch_invariant_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"seq_len", std::to_string(seq_len)},
                    {"top_k", std::to_string(top_k)},
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"gateup_codebook_mask",
                     codebookMaskTag(gateup_table.codebook_mask)},
                    {"down_codebook_mask",
                     codebookMaskTag(down_table.codebook_mask)},
                    {"gateup_policy_codebook_mask",
                     codebookMaskTag(gateup_table.policy_codebook_mask)},
                    {"down_policy_codebook_mask",
                     codebookMaskTag(down_table.policy_codebook_mask)},
                    {"gateup_route",
                     seq_len > 8
                         ? (reuse_router_q8_hidden
                                ? "expert_tiled_router_q8"
                                : "expert_tiled_original_row_quant")
                         : (reuse_router_q8_hidden
                                ? "route_owned_router_q8"
                                : "route_owned_original_row_quant")},
                    {"down_route",
                     seq_len > 8
                         ? "expert_tiled_partials_ordered_publish"
                         : "direct_ordered_publish"},
                    {"gateup_tile_m", policy.gateup.tile_m},
                    {"gateup_tile_n", policy.gateup.tile_n},
                    {"down_tile_m", policy.down.tile_m},
                    {"down_tile_n", policy.down.tile_n},
                    {"policy_source", policy.source},
                    {"row_tile", common_row_tile},
                    // These tags describe the captured envelope, not a host
                    // observation of mutable device-owned live row counts.
                    {"geometry_role", "physical_capacity"},
                    {"route_admission", "device_group_plan"},
                    {"grouping", "static"}});
        }
        return true;
    }

    bool ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan(
        DeviceMoELayerRuntime *device_runtime_layer,
        const DeviceMoELayerRuntime &runtime_host_layer,
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        const MoEPrefillProjectionExecution &execution,
        ITensor *canonical_route_contributions)
    {
        if (execution.isIntermediateShard() != runtime_host_layer.intermediate_shard.enabled() ||
            (execution.isIntermediateShard() &&
             (runtime_host_layer.intermediate_shard.modelColumns() != static_cast<uint32_t>(d_model) ||
              runtime_host_layer.intermediate_shard.localIntermediateColumns() != static_cast<uint32_t>(intermediate) ||
              !canonical_route_contributions)))
            return false;

        const auto *owner_admission = ownerGateUpAdmissionForCall(gateup_desc_table_id, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model});
        rocm::MoERouteOwnedGateUpLaunchSet owner_completed{};
        rocm::validateMoERouteOwnedGateUpScopeBinding(device_ordinal_, gpu_stream_, seq_len, execution,
            {.output_columns = intermediate, .reduction_columns = d_model}, owner_admission);
        grouped_prefill_grouping_.invalidate();
        if (!device_runtime_layer)
            return false;
        // Mutable placement never owns immutable column slices. Binding a
        // local-N hint to its full-N packed matrix would change indexing.
        if (execution.executesDown() && execution.columnCount() != d_model)
        {
            LOG_ERROR("Sliced MoE down execution requires the immutable fixed-down descriptor table");
            return false;
        }
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0 ||
            seq_len > std::numeric_limits<int>::max() / top_k ||
            execution.modelColumns() != d_model)
        {
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "executeGroupedPrefillProjectionFromPublishedRuntimePlan"))
            return false;
        if (runtime_host_layer.expert_count != static_cast<uint32_t>(num_experts) ||
            runtime_host_layer.top_k != static_cast<uint32_t>(top_k) ||
            runtime_host_layer.prefill_token_capacity < static_cast<uint32_t>(seq_len) ||
            runtime_host_layer.prefill_route_capacity < static_cast<uint32_t>(seq_len * top_k) ||
            !runtime_host_layer.expert_counts ||
            !runtime_host_layer.expert_offsets ||
            !runtime_host_layer.grouped_token_ids ||
            !runtime_host_layer.grouped_route_weights ||
            !runtime_host_layer.route_expert_ids)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] invalid runtime scratch contract"
                      << " expert_count=" << runtime_host_layer.expert_count
                      << " expected_experts=" << num_experts
                      << " top_k=" << runtime_host_layer.top_k
                      << " expected_top_k=" << top_k
                      << " token_capacity=" << runtime_host_layer.prefill_token_capacity
                      << " seq_len=" << seq_len
                      << " route_capacity=" << runtime_host_layer.prefill_route_capacity
                      << " total_slots=" << (seq_len * top_k));
            return false;
        }
        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] invalid descriptor table id");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if ((gateup_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (execution.isIntermediateShard()) ||
            (down_table.descriptor_source == MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable) !=
                (execution.isIntermediateShard()))
            return false;

        rocm::validateMoERouteOwnedGateUpCodebook(gateup_table.codebook_mask, gateup_table.policy_codebook_mask, owner_admission);
        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            (execution.executesDown() && down_table.d_model != execution.columnCount()) ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] descriptor table shape mismatch");
            return false;
        }

        const int total_slots = seq_len * top_k;
        if (!ensureGroupedPrefillScratchCapacity(
                total_slots, d_model, intermediate) ||
            !ensureGroupedPrefillWorkDirectoryCapacity(
                total_slots, num_experts))
            return false;
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        void *stream = getStream();

        /*
         * Resolve both fused-publication outputs before launch.  These are
         * persistent workspace addresses selected by graph identity; binding
         * performs validation and pointer arithmetic only, never allocation.
         */
        if (total_slots > group_slots_cap_ ||
            !d_group_original_to_grouped_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_original_to_grouped_),
                                     MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "executeGroupedPrefillProjectionFromPublishedRuntimePlan(group_original_to_grouped)"))
            {
                d_group_original_to_grouped_ = nullptr;
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] runtime grouping workspace is required");
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        const int active_expert_slots = std::min(total_slots, num_experts);

        /*
         * Complete plan publication has already written these graph-owned
         * descriptor slots and the ordered inverse map. The compute consumer
         * resolves the same persistent addresses without hidden publication
         * kernels of its own.
         */
        DeviceNativeQuantisedMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeQuantisedMatrixDesc *runtime_down_descs = nullptr;
        if (!bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_gate_descs,
                "ROCm runtime prefill gate descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
                gateup_table.workspace_slot,
                num_experts,
                &runtime_up_descs,
                "ROCm runtime prefill up descriptors") ||
            !bindGroupedDescriptorTableSlot(
                MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                down_table.workspace_slot,
                num_experts,
                &runtime_down_descs,
                "ROCm runtime prefill down descriptors"))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                      "failed to bind graph-owned runtime descriptor slots");
            return false;
        }

        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if ((execution.executesGateUp() && !requireTensorOnDevice(
                hidden,
                device,
                stream,
                "hidden",
                "executeGroupedPrefillProjectionFromPublishedRuntimePlan")) ||
            (execution.executesDown() && !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name,
                "executeGroupedPrefillProjectionFromPublishedRuntimePlan")))
        {
            return false;
        }

        const float *d_hidden = execution.executesGateUp()
            ? static_cast<const float *>(hidden->gpu_data_ptr()) : nullptr;
        float *d_output = !execution.executesDown() || canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            execution.executesDown() && canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;
        if ((execution.executesGateUp() && !d_hidden) ||
            (execution.executesDown() && !d_output && !d_canonical_route_contributions))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] null device pointers");
            return false;
        }

        group_active_expert_slots_ = active_expert_slots;

        if (floating)
        {
            const bool ok = execution.overwriteOutputThenPublish(
                execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
                execution.executesDown() ? publication_output->size_bytes() : 0u,
                seq_len, top_k,
                canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                              : MoEPrefillOutputLayout::TokenRows,
                stream,
                [&](std::span<float> live_output, void *producer_stream)
                {
                    return rocmMoE_grouped_floating_prefill_pipeline(
                        d_hidden,
                        reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                            runtime_gate_descs),
                        reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                            runtime_up_descs),
                        reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                            runtime_down_descs),
                        d_group_original_to_grouped_,
                        runtime_host_layer.route_expert_ids,
                        runtime_host_layer.grouped_route_weights,
                        d_prefill_gate_,
                        canonical_route_contributions ? nullptr : live_output.data(),
                        canonical_route_contributions ? live_output.data() : nullptr,
                        seq_len,
                        total_slots,
                        top_k,
                        d_model,
                        intermediate,
                        num_experts,
                        gateup_table.weight_format,
                        device_ordinal_,
                        producer_stream,
                        execution);
                },
                [&](const EnqueuedWorkspaceRead<float> &read)
                {
                    markDeviceWritten(publication_output, device, read.stream());
                    return true;
                });
            if (!ok)
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] "
                          "floating grouped ROCm pipeline failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_floating_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"active_expert_slots", std::to_string(active_expert_slots)},
                 {"descriptor_source", execution.isIntermediateShard() ? "runtime_intermediate_shard" : "runtime_table"},
                 {"weight_format", std::to_string(static_cast<uint32_t>(
                                       gateup_table.weight_format))}});
            return true;
        }

        const bool reuse_router_q8_hidden = execution.executesGateUp() &&
            canReuseRouterQ8Hidden(d_hidden, seq_len, d_model);
        const bool ok = execution.overwriteOutputThenPublish(
            execution.executesDown() ? static_cast<float *>(publication_output->gpu_data_ptr()) : nullptr,
            execution.executesDown() ? publication_output->size_bytes() : 0u,
            seq_len, top_k,
            canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                          : MoEPrefillOutputLayout::TokenRows,
            stream,
            [&](std::span<float> live_output, void *producer_stream)
            {
                return rocmMoE_grouped_prefill_pipeline(
                    d_hidden,
                    reuse_router_q8_hidden ? d_router_q8_hidden_ : nullptr,
                    reuse_router_q8_hidden ? d_router_q8_hidden_scales_ : nullptr,
                    runtime_gate_descs,
                    runtime_up_descs,
                    runtime_down_descs,
                    runtime_host_layer.expert_counts,
                    runtime_host_layer.expert_offsets,
                    runtime_host_layer.grouped_token_ids,
                    d_group_original_to_grouped_,
                    runtime_host_layer.route_expert_ids,
                    runtime_host_layer.grouped_route_weights,
                    reinterpret_cast<int *>(d_prefill_work_directory_),
                    d_prefill_A_int8_,
                    d_prefill_A_scales_,
                    d_prefill_gate_,
                    d_prefill_up_,
                    d_prefill_swiglu_int8_,
                    d_prefill_swiglu_scales_,
                    canonical_route_contributions ? nullptr : live_output.data(),
                    canonical_route_contributions ? live_output.data() : nullptr,
                    num_experts,
                    d_model,
                    intermediate,
                    total_slots,
                    top_k,
                    1,
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    gateup_table.policy_codebook_mask,
                    down_table.policy_codebook_mask,
                    device_ordinal_,
                    producer_stream,
                    execution,
                    MoEPrefillRouteScope::SelectedRoutes, owner_admission,
                    owner_admission ? &owner_completed : nullptr);
            },
            [&](const EnqueuedWorkspaceRead<float> &read)
            {
                markDeviceWritten(publication_output, device, read.stream());
                return true;
            });
        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillProjectionFromPublishedRuntimePlan] grouped ROCm pipeline failed");
            return false;
        }

        if (owner_admission)
            recordOwnerGateUpCaptureSubmissions(owner_completed, gateup_table.owner_m1_intent);

        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_router_q8_reuse_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"descriptor_source", execution.isIntermediateShard() ? "runtime_intermediate_shard" : "runtime_table"}});
        }

            if (PerfStatsCollector::isDomainEnabled("kernel") &&
                active_expert_slots > 0)
        {
            const MoEPrefillPairPolicyTags policy =
                queryMoEPrefillPairPolicyTags(
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    seq_len,
                    d_model,
                    intermediate,
                    num_experts,
                    top_k,
                    execution.measuredOperation(
                        canonical_route_contributions ? MoEPrefillOutputLayout::OriginalRouteRows
                                                      : MoEPrefillOutputLayout::TokenRows,
                        MoEPrefillRouteScope::SelectedRoutes));
            const std::string common_row_tile =
                policy.gateup.tile_m == policy.down.tile_m
                    ? policy.gateup.tile_m
                    : "mixed";
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_batch_invariant_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"seq_len", std::to_string(seq_len)},
                    {"top_k", std::to_string(top_k)},
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"gateup_codebook_mask",
                     codebookMaskTag(gateup_table.codebook_mask)},
                    {"down_codebook_mask",
                     codebookMaskTag(down_table.codebook_mask)},
                    {"gateup_policy_codebook_mask",
                     codebookMaskTag(gateup_table.policy_codebook_mask)},
                    {"down_policy_codebook_mask",
                     codebookMaskTag(down_table.policy_codebook_mask)},
                    {"gateup_route",
                     seq_len > 8
                         ? (reuse_router_q8_hidden
                                ? "expert_tiled_router_q8"
                                : "expert_tiled_original_row_quant")
                         : (reuse_router_q8_hidden
                                ? "route_owned_router_q8"
                                : "route_owned_original_row_quant")},
                    {"down_route",
                     seq_len > 8
                         ? "expert_tiled_partials_ordered_publish"
                         : "direct_ordered_publish"},
                    {"gateup_tile_m", policy.gateup.tile_m},
                    {"gateup_tile_n", policy.gateup.tile_n},
                    {"down_tile_m", policy.down.tile_m},
                    {"down_tile_n", policy.down.tile_n},
                    {"policy_source", policy.source},
                    {"row_tile", common_row_tile},
                    {"geometry_role", "physical_capacity"},
                    {"route_admission", "device_group_plan"},
                    {"grouping", "runtime"}});
        }
        return true;
    }

    bool ROCmMoEKernel::reduceCanonicalRouteContributions(
        ITensor *canonical_route_contributions,
        ITensor *output,
        int seq_len,
        int top_k,
        int d_model)
    {
        if (!canonical_route_contributions || !output ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0)
        {
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "reduceCanonicalRouteContributions"))
        {
            return false;
        }

        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::reduceCanonicalRouteContributions] "
                      "an explicit stream is required");
            return false;
        }
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(
                canonical_route_contributions,
                device,
                stream,
                "canonical_route_contributions",
                "reduceCanonicalRouteContributions") ||
            !requireOutputOnDevice(
                output,
                device,
                stream,
                "output",
                "reduceCanonicalRouteContributions"))
        {
            return false;
        }

        const float *d_contributions = static_cast<const float *>(
            canonical_route_contributions->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!rocmMoE_reduce_canonical_route_contributions(
                d_contributions,
                d_output,
                seq_len,
                top_k,
                d_model,
                device_ordinal_,
                stream))
        {
            return false;
        }

        markDeviceWritten(output, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_moe_canonical_route_reduce_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"d_model", std::to_string(d_model)}});
        return true;
    }

    bool ROCmMoEKernel::publishSharedExpertRankBank(
        ITensor *shared_output,
        ITensor *canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *device_effective_seq_len)
    {
        constexpr const char *kContext =
            "publishSharedExpertRankBank";
        if (!shared_output || !canonical_publication ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0 || participant_index < 0 ||
            participant_index >= participant_count)
        {
            return false;
        }

        void *stream = requireStream(
            "ROCmMoEKernel::publishSharedExpertRankBank");
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_elements =
            row_elements *
            (static_cast<size_t>(top_k) +
             static_cast<size_t>(participant_count));
        if (shared_output->numel() < row_elements ||
            canonical_publication->numel() < required_elements ||
            !setMoEDevice(device_ordinal_, kContext) ||
            !requireTensorOnDevice(
                shared_output,
                device,
                stream,
                "shared_output",
                kContext) ||
            !requireTensorOnDevice(
                canonical_publication,
                device,
                stream,
                "canonical_publication",
                kContext))
        {
            return false;
        }

        if (!rocmMoE_publish_shared_expert_rank_bank(
                static_cast<const float *>(shared_output->gpu_data_ptr()),
                static_cast<float *>(
                    canonical_publication->gpu_data_ptr()),
                seq_len,
                top_k,
                d_model,
                participant_index,
                participant_count,
                device_effective_seq_len,
                device_ordinal_,
                stream))
        {
            return false;
        }

        markDeviceWritten(canonical_publication, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_moe_shared_rank_bank_publish_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"participants", std::to_string(participant_count)}});
        return true;
    }

    bool ROCmMoEKernel::finalizeCanonicalMoEPublication(
        ITensor *input,
        ITensor *gate_inp,
        ITensor *canonical_publication,
        ITensor *routed_output,
        ITensor *shared_output,
        ITensor *combined_output,
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        const int *device_effective_seq_len)
    {
        constexpr const char *kContext =
            "finalizeCanonicalMoEPublication";
        if (!input || !gate_inp || !canonical_publication ||
            !routed_output || !shared_output || !combined_output ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0)
        {
            return false;
        }

        void *stream = requireStream(
            "ROCmMoEKernel::finalizeCanonicalMoEPublication");
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_elements =
            row_elements *
            (static_cast<size_t>(top_k) +
             static_cast<size_t>(participant_count));
        if (input->numel() < row_elements ||
            gate_inp->numel() < static_cast<size_t>(d_model) ||
            canonical_publication->numel() < required_elements ||
            routed_output->numel() < row_elements ||
            shared_output->numel() < row_elements ||
            combined_output->numel() < row_elements ||
            !setMoEDevice(device_ordinal_, kContext) ||
            !requireTensorOnDevice(
                input, device, stream, "input", kContext) ||
            !requireTensorOnDevice(
                gate_inp, device, stream, "gate_inp", kContext) ||
            !requireTensorOnDevice(
                canonical_publication,
                device,
                stream,
                "canonical_publication",
                kContext) ||
            !requireOutputOnDevice(
                routed_output,
                device,
                stream,
                "routed_output",
                kContext) ||
            !requireOutputOnDevice(
                shared_output,
                device,
                stream,
                "shared_output",
                kContext) ||
            !requireOutputOnDevice(
                combined_output,
                device,
                stream,
                "combined_output",
                kContext))
        {
            return false;
        }

        if (!rocmMoE_finalize_canonical_publication(
                static_cast<const float *>(input->gpu_data_ptr()),
                static_cast<const float *>(gate_inp->gpu_data_ptr()),
                static_cast<const float *>(
                    canonical_publication->gpu_data_ptr()),
                static_cast<float *>(routed_output->gpu_data_ptr()),
                static_cast<float *>(shared_output->gpu_data_ptr()),
                static_cast<float *>(combined_output->gpu_data_ptr()),
                seq_len,
                top_k,
                d_model,
                participant_count,
                device_effective_seq_len,
                device_ordinal_,
                stream))
        {
            return false;
        }

        markDeviceWritten(routed_output, device, stream);
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_moe_canonical_publication_finalize_calls",
            1.0,
            "moe",
            device.to_string(),
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)},
             {"participants", std::to_string(participant_count)}});
        return true;
    }

namespace
{
/** @brief One retained existing named arena interval, never an allocation authority. */
struct ResidentFFNScratchBinding
{
    const char* name=nullptr;
    void* pointer=nullptr;
    std::shared_ptr<const WorkspaceBufferLease> lease;
    std::size_t bytes=0;
};
/** @brief Existing immutable singleton table handles and exact captured descriptor leases. */
struct ResidentFFNSharedTables
{
    int gate_up=-1,down=-1;
    std::array<std::shared_ptr<const WorkspaceBufferLease>,3> leases;
};
}
/** @brief Immutable borrowed counter/table/scratch seal; the arena remains allocation authority. */
struct ROCmMoEKernel::GateUpLastProducerBinding
{
    rocm::GateUpProducerLoads loads;
    DeviceWorkspaceManager* workspace;
    std::uint64_t generation;
    void* stream;
    int table,top_k,hidden,intermediate,experts;
    std::uint32_t execution_mask,policy_mask;
    NativeTileCompletionView view;
    std::size_t counter_bytes;
    std::shared_ptr<const WorkspaceBufferLease> lease;
    std::shared_ptr<GroupedDescriptorWorkspacePublication> publication;
    const void* original_gate;
    const void* original_up;
    std::size_t descriptor_slot;
    MoEIntermediateShardDescriptorDeclaration original;
    struct ScratchSeal
    {
        const char* name;
        const void* pointer;
        std::size_t bytes;
        std::shared_ptr<const WorkspaceBufferLease> lease;
    };
    std::vector<ScratchSeal> scratch;
    std::uint64_t signature;
};

/** @copydoc IMoEKernel::prepareGateUpPublication */
std::uint64_t ROCmMoEKernel::prepareGateUpPublication(int table,int top_k,MoEGateUpPublicationPolicy policy)
{
    if(::llaminar2::isGraphCaptureActive()||!workspace_||reinterpret_cast<std::uintptr_t>(getStream())<=2||
        table<0||table>=int(grouped_gateup_desc_tables_.size())||top_k<=0||top_k>int(kDeviceMoEMaxTopK))
        throw std::logic_error("Gate/up publication requires cold original table/stream/workspace admission");
    if(policy==MoEGateUpPublicationPolicy::Staged)
    {
        if(gate_up_last_producer_.contains(table))
            throw std::logic_error("A prepared last-producer gate/up owner cannot become Staged behind its captured graph");
        return 0;
    }
    if(policy!=MoEGateUpPublicationPolicy::LastProducer)
        throw std::invalid_argument("Unknown gate/up publication policy");
    // Binding a borrowed manager does not select a HIP device on this host
    // thread. Reuse the canonical backend/device guard before setup traffic.
    if(!validateROCmWorkspaceBinding(workspace_,device_ordinal_,"prepareGateUpPublication") ||
        !setMoEDevice(device_ordinal_,"prepareGateUpPublication"))
        throw std::logic_error("Last-producer gate/up lost its original ROCm workspace/device");
    if(gate_up_last_producer_.contains(table))
    {
        if(gate_up_last_producer_.at(table)->loads!=rocm::currentGateUpProducerLoads())
            throw std::logic_error("Prepared gate/up producer loads cannot be reselected");
        return gateUpPublicationSignature(table,top_k);
    }
    const auto& source=grouped_gateup_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        top_k>source.num_experts||!source.workspace_publication)
        throw std::invalid_argument("Last-producer gate/up requires an original native M1 descriptor table");
    const auto plan=rocm::gateUpLastProducerPlan(source.intermediate,source.d_model,top_k,source.policy_codebook_mask);
    auto* counters=static_cast<std::uint32_t*>(workspace_->getBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION));
    if(!counters||workspace_->getBufferSize(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION)<plan.bytes())
        throw std::invalid_argument("Last-producer gate/up has no sufficient named PMA counter bank");
    auto binding=std::make_shared<GateUpLastProducerBinding>(GateUpLastProducerBinding{
        rocm::currentGateUpProducerLoads(),workspace_,workspace_->id(),getStream(),table,top_k,source.d_model,source.intermediate,source.num_experts,
        source.codebook_mask,source.policy_codebook_mask,plan.bind(counters),plan.bytes(),
        workspace_->retainBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION,0,plan.bytes()),source.workspace_publication,
        source.device_gate_descs,source.device_up_descs,source.workspace_slot,source.original_family,{},{}});
    for(const auto* name:std::array{MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
        MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
        MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
        MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,
        MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES})
    {
        const auto* pointer=workspace_->getBuffer(name);
        if(!pointer)throw std::logic_error("Last-producer gate/up original scratch/table preparation is incomplete");
        const auto bytes=workspace_->getBufferSize(name);
        binding->scratch.push_back({name,pointer,bytes,workspace_->retainBuffer(name,0,bytes)});
    }
    // Source tensors, partial banks and counters are distinct admitted names.
    // The public cold metadata boundary proves completion of this exact setup
    // copy before host staging retires. A retained leaf may later be imported
    // into a parent launched on another stream, so capture-stream order alone
    // is insufficient. Only this one-time publication waits on its own event;
    // every subsequent reset remains device-owned in the last producer CTA.
    const std::vector<std::byte> initial_counters(plan.bytes(),std::byte{0});
    TransferEngine::instance().initializeWorkspaceMetadata(*binding->lease,initial_counters,getStream());
    std::uint64_t signature=kFNV1a64OffsetBasis;
    const auto mix=[&](std::uint64_t value){signature=(signature^value)*kFNV1a64Prime;};
    for(const auto* pointer:std::array<const void*,7>{workspace_,getStream(),counters,source.device_gate_descs,
        source.device_up_descs,source.workspace_publication.get(),rocm::gateUpLastProducerKernelFunction()})
        mix(reinterpret_cast<std::uintptr_t>(pointer));
    mix(workspace_->id());mix(static_cast<std::uint64_t>(binding->loads));mix(table);mix(top_k);mix(source.d_model);mix(source.intermediate);mix(source.num_experts);
    mix(source.codebook_mask);mix(source.policy_codebook_mask);mix(source.workspace_slot);
    mix(plan.tiles());mix(plan.contributors());mix(plan.bytes());
    for(const auto& seal:binding->scratch){mix(reinterpret_cast<std::uintptr_t>(seal.pointer));mix(seal.bytes);}
    binding->signature=signature?signature:1;
    gate_up_last_producer_.emplace(table,std::move(binding));
    return gateUpPublicationSignature(table,top_k);
}

/** @copydoc IMoEKernel::gateUpPublicationSignature */
std::uint64_t ROCmMoEKernel::gateUpPublicationSignature(int table,int top_k) const
{
    const auto found=gate_up_last_producer_.find(table);
    if(found==gate_up_last_producer_.end())return 0;
    const auto& binding=*found->second;
    if(!workspace_||workspace_!=binding.workspace||workspace_->id()!=binding.generation||getStream()!=binding.stream||
        table!=binding.table||top_k!=binding.top_k||table<0||table>=int(grouped_gateup_desc_tables_.size()))
        throw std::logic_error("Last-producer gate/up changed its original owner/generation/stream/table");
    const auto& source=grouped_gateup_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        source.d_model!=binding.hidden||source.intermediate!=binding.intermediate||source.num_experts!=binding.experts||
        source.codebook_mask!=binding.execution_mask||source.policy_codebook_mask!=binding.policy_mask||
        source.workspace_slot!=binding.descriptor_slot||source.workspace_publication!=binding.publication||
        source.device_gate_descs!=binding.original_gate||source.device_up_descs!=binding.original_up||
        !source.original_family.sameIdentity(binding.original))
        throw std::logic_error("Last-producer gate/up changed its original descriptor/source publication");
    if(workspace_->getBuffer(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION)!=binding.view.counters||
        workspace_->getBufferSize(MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION)<binding.counter_bytes)
        throw std::logic_error("Last-producer gate/up counter bank was rebound or released");
    for(const auto& seal:binding.scratch)
        if(workspace_->getBuffer(seal.name)!=seal.pointer||workspace_->getBufferSize(seal.name)<seal.bytes)
            throw std::logic_error("Last-producer gate/up original scratch/table storage changed");
    return binding.signature;
}

/** @copydoc gateUpCompletionForCall */
const NativeTileCompletionView* ROCmMoEKernel::gateUpCompletionForCall(int table,int top_k) const
{
    const auto signature=gateUpPublicationSignature(table,top_k);
    if(signature&&gate_up_last_producer_.at(table)->loads!=rocm::currentGateUpProducerLoads())
        throw std::logic_error("Prepared gate/up producer loads cannot change behind retained capture identity");
    return signature?&gate_up_last_producer_.at(table)->view:nullptr;
}

/** @brief Immutable borrowed counter/table/scratch seal; the arena remains allocation authority. */
struct ROCmMoEKernel::DownLastProducerBinding
{
    DeviceWorkspaceManager* workspace;
    std::uint64_t generation;
    void* stream;
    int table,top_k,hidden,intermediate,experts;
    std::uint32_t execution_mask,policy_mask;
    NativeDownCompletionView view;
    std::size_t counter_bytes;
    std::shared_ptr<const WorkspaceBufferLease> lease;
    std::shared_ptr<GroupedDescriptorWorkspacePublication> publication;
    const void* original_down;
    std::size_t descriptor_slot;
    MoEIntermediateShardDescriptorDeclaration original;
    struct ScratchSeal
    {
        const char* name;
        const void* pointer;
        std::size_t bytes;
        std::shared_ptr<const WorkspaceBufferLease> lease;
    };
    std::vector<ScratchSeal> scratch;
    std::uint64_t signature;
};

/** @copydoc IMoEKernel::prepareDownPublication */
std::uint64_t ROCmMoEKernel::prepareDownPublication(int table,int top_k,MoEDownPublicationPolicy policy)
{
    if(::llaminar2::isGraphCaptureActive()||!workspace_||reinterpret_cast<std::uintptr_t>(getStream())<=2||
        table<0||table>=int(grouped_down_desc_tables_.size())||top_k<=0||top_k>int(kDeviceMoEMaxTopK))
        throw std::logic_error("Down/route publication requires cold original table/stream/workspace admission");
    if(policy==MoEDownPublicationPolicy::Staged)
    {
        if(down_last_producer_.contains(table))
            throw std::logic_error("A prepared last-producer down/route owner cannot become Staged behind its captured graph");
        return 0;
    }
    if(policy!=MoEDownPublicationPolicy::LastProducerRouteFold)
        throw std::invalid_argument("Unknown down/route publication policy");
    // Binding a borrowed manager does not select a HIP device on this host
    // thread. Reuse the canonical backend/device guard before setup traffic.
    if(!validateROCmWorkspaceBinding(workspace_,device_ordinal_,"prepareDownPublication") ||
        !setMoEDevice(device_ordinal_,"prepareDownPublication"))
        throw std::logic_error("Last-producer down/route lost its original ROCm workspace/device");
    if(down_last_producer_.contains(table))return downPublicationSignature(table,top_k);
    const auto& source=grouped_down_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        top_k>source.num_experts||!source.workspace_publication)
        throw std::invalid_argument("Last-producer down/route requires an original native M1 descriptor table");
    const auto plan=rocm::downLastProducerPlan(source.d_model,source.intermediate,top_k,source.policy_codebook_mask);
    auto* counters=static_cast<std::uint32_t*>(workspace_->getBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION));
    if(!counters||workspace_->getBufferSize(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION)<plan.bytes())
        throw std::invalid_argument("Last-producer down/route has no sufficient named PMA counter bank");
    auto binding=std::make_shared<DownLastProducerBinding>(DownLastProducerBinding{
        workspace_,workspace_->id(),getStream(),table,top_k,source.d_model,source.intermediate,source.num_experts,
        source.codebook_mask,source.policy_codebook_mask,plan.bind(counters),plan.bytes(),
        workspace_->retainBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION,0,plan.bytes()),source.workspace_publication,
        source.device_descs,source.workspace_slot,source.original_family,{},{}});
    for(const auto* name:std::array{MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
        MoEWorkspaceBuffers::DOWN_PARTIALS,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,
        MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES})
    {
        const auto* pointer=workspace_->getBuffer(name);
        if(!pointer)throw std::logic_error("Last-producer down/route original scratch/table preparation is incomplete");
        const auto bytes=workspace_->getBufferSize(name);
        binding->scratch.push_back({name,pointer,bytes,workspace_->retainBuffer(name,0,bytes)});
    }
    // Source tensors, partial banks and counters are distinct admitted names.
    // The public cold metadata boundary proves completion of this exact setup
    // copy before host staging retires. A retained leaf may later be imported
    // into a parent launched on another stream, so capture-stream order alone
    // is insufficient. Only this one-time publication waits on its own event;
    // every subsequent reset remains device-owned in the last producer CTA.
    const std::vector<std::byte> initial_counters(plan.bytes(),std::byte{0});
    TransferEngine::instance().initializeWorkspaceMetadata(*binding->lease,initial_counters,getStream());
    std::uint64_t signature=kFNV1a64OffsetBasis;
    const auto mix=[&](std::uint64_t value){signature=(signature^value)*kFNV1a64Prime;};
    for(const auto* pointer:std::array<const void*,6>{workspace_,getStream(),counters,source.device_descs,
        source.workspace_publication.get(),rocm::downLastProducerKernelFunction()})
        mix(reinterpret_cast<std::uintptr_t>(pointer));
    mix(workspace_->id());mix(table);mix(top_k);mix(source.d_model);mix(source.intermediate);mix(source.num_experts);
    mix(source.codebook_mask);mix(source.policy_codebook_mask);mix(source.workspace_slot);
    mix(plan.partitions().tiles());mix(plan.partitions().contributors());mix(plan.bytes());
    for(const auto& seal:binding->scratch){mix(reinterpret_cast<std::uintptr_t>(seal.pointer));mix(seal.bytes);}
    binding->signature=signature?signature:1;
    down_last_producer_.emplace(table,std::move(binding));
    return downPublicationSignature(table,top_k);
}

/** @copydoc IMoEKernel::downPublicationSignature */
std::uint64_t ROCmMoEKernel::downPublicationSignature(int table,int top_k) const
{
    const auto found=down_last_producer_.find(table);
    if(found==down_last_producer_.end())return 0;
    const auto& binding=*found->second;
    if(!workspace_||workspace_!=binding.workspace||workspace_->id()!=binding.generation||getStream()!=binding.stream||
        table!=binding.table||top_k!=binding.top_k||table<0||table>=int(grouped_down_desc_tables_.size()))
        throw std::logic_error("Last-producer down/route changed its original owner/generation/stream/table");
    const auto& source=grouped_down_desc_tables_[table];
    if(!source.valid||!source.deviceReady()||source.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
        source.d_model!=binding.hidden||source.intermediate!=binding.intermediate||source.num_experts!=binding.experts||
        source.codebook_mask!=binding.execution_mask||source.policy_codebook_mask!=binding.policy_mask||
        source.workspace_slot!=binding.descriptor_slot||source.workspace_publication!=binding.publication||
        source.device_descs!=binding.original_down||
        !source.original_family.sameIdentity(binding.original))
        throw std::logic_error("Last-producer down/route changed its original descriptor/source publication");
    if(workspace_->getBuffer(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION)!=binding.view.partitions.counters||
        workspace_->getBufferSize(MoEWorkspaceBuffers::DOWN_TILE_COMPLETION)<binding.counter_bytes)
        throw std::logic_error("Last-producer down/route counter bank was rebound or released");
    for(const auto& seal:binding.scratch)
        if(workspace_->getBuffer(seal.name)!=seal.pointer||workspace_->getBufferSize(seal.name)<seal.bytes)
            throw std::logic_error("Last-producer down/route original scratch/table storage changed");
    return binding.signature;
}

/** @copydoc downCompletionForCall */
const NativeDownCompletionView* ROCmMoEKernel::downCompletionForCall(int table,int top_k) const
{
    const auto signature=downPublicationSignature(table,top_k);
    return signature?&down_last_producer_.at(table)->view:nullptr;
}

#include "ROCmFullNativeHCFFNOwner.inc"

/** @brief Sealed native program plus original tensor/workspace lifetime observations. */
class ROCmMoEKernel::PreparedResidentFFN final : public PreparedResidentMoEFFN
{
public:
    /**
     * @brief Retain original FFN authority and one exact already-admitted native program.
     * @param owner Original backend source/table/workspace authority.
     * @param request Original tensor/source declaration retained by the stage.
     * @param args Complete admitted FFN argument snapshot.
     * @param admission Actual standalone or full function's immutable resource/admission evidence.
     * @param stream Exact original non-default submission stream.
     * @param scratch Existing named FFN leases and consumed byte extents.
     * @param shared_tables Original immutable shared descriptor publication handles.
     * @param router Optional original router/suffix snapshot.
     * @param router_tensors Original router/ID/weight tensor identity observations.
     * @param barrier_slot Existing exclusive instance slot, retained until graph retirement.
     * @param prefix Optional admitted HC phase; its original stage retains and validates physical leases.
     */
    PreparedResidentFFN(ROCmMoEKernel& owner,ResidentMoEFFNRequest request,rocm::ResidentMoEFFNArgs args,
        rocm::PreparedMoEFFNProgram admission,void* stream,std::vector<ResidentFFNScratchBinding> scratch,
        ResidentFFNSharedTables shared_tables,std::optional<rocm::NativeLayoutRouterFFNArgs> router={},
        std::array<TensorBase*,3> router_tensors={},
        std::shared_ptr<PersistentWorkspaceSlotLease> barrier_slot={},
        std::optional<rocm::FullNativeHCFFNPrefixArgs> prefix={})
        :owner_(&owner),request_(std::move(request)),args_(args),admission_(admission),stream_(stream),
         workspace_(owner.getWorkspace()),workspace_id_(workspace_->id()),scratch_(std::move(scratch)),
         shared_tables_(std::move(shared_tables)),gate_publication_(owner.grouped_gateup_desc_tables_.at(request_.gate_up_table).workspace_publication),
         down_publication_(owner.grouped_down_desc_tables_.at(request_.down_table).workspace_publication),
         shared_gate_publication_(owner.grouped_gateup_desc_tables_.at(shared_tables_.gate_up).workspace_publication),
         shared_down_publication_(owner.grouped_down_desc_tables_.at(shared_tables_.down).workspace_publication),
         barrier_slot_(std::move(barrier_slot)),native_router_(std::move(router)),full_prefix_(std::move(prefix)),router_tensors_(router_tensors),router_publication_((admission.evidence().kind==ResidentMoEFFNLaunchKind::OrdinaryPhases||native_router_)
            ?owner.paired_router_descriptors_:std::nullopt)
    {
        const auto mix=[&](std::uint64_t value){signature_=(signature_^value)*kFNV1a64Prime;};
        for(const void* pointer:std::array<const void*,23>{args_.runtime,args_.hidden,args_.gates,args_.ups,args_.downs,
            args_.shared_gate,args_.shared_up,args_.shared_down,
            args_.shared_input_gate,args_.hidden_q8,args_.hidden_scales,args_.gate_partials,args_.up_partials,args_.down_partials,
            args_.swiglu_q8,args_.swiglu_scales,args_.route_contributions,args_.routed_output,args_.shared_output,args_.combined_output,
            args_.shared_gate_output,args_.shared_up_output,stream_})mix(reinterpret_cast<std::uintptr_t>(pointer));
        for(const auto& matrix:std::array{request_.shared.gate,request_.shared.up,request_.shared.down})
        {
            for(const void* pointer:std::array<const void*,4>{matrix.payload,matrix.scales,matrix.mins,matrix.emins})
                mix(reinterpret_cast<std::uintptr_t>(pointer));
            mix(matrix.n);mix(matrix.k);mix(matrix.blocks_per_row);mix(matrix.codebook_id);
            mix(matrix.source_identity_present);mix(matrix.source_codebook_id);mix(matrix.source_is_superblock);
            mix(matrix.allocation_payload_bytes_per_block);mix(matrix.allocation_has_mins);mix(matrix.allocation_has_emins);
        }
        const auto rows=args_.rows.captureIdentity();
        mix(reinterpret_cast<std::uintptr_t>(rows.count));mix(rows.count_capacity);mix(rows.first_physical_row);
        mix(rows.physical_rows);mix(rows.repeats_per_row);mix(rows.schema_version);
        mix(args_.hidden_columns);mix(args_.routed_columns);mix(args_.shared_columns);mix(args_.experts);mix(args_.top_k);mix(args_.resident_grid);
        mix(args_.execution.routed_gate);mix(args_.execution.routed_down);
        mix(args_.execution.shared_gate);mix(args_.execution.shared_down);
        mix(args_.original.firstIntermediateColumn());mix(args_.original.intermediateColumns());
        mix(args_.original.target().participantId());mix(args_.original.target().cohortTarget());
        const auto& evidence=admission_.evidence();
        sealRetainedModelReceipt();
        mix(static_cast<std::uint64_t>(evidence.kind));mix(evidence.generation);mix(evidence.nativeEntries().size());
        mix(static_cast<std::uint64_t>(evidence.device.type));mix(evidence.device.ordinal);
        for(const auto& entry:evidence.nativeEntries())
        {
            mix(reinterpret_cast<std::uintptr_t>(entry.function));mix(entry.grid);mix(entry.threads);mix(entry.dynamic_shared_bytes);
            mix(entry.resources.registers_per_thread);mix(entry.resources.maximum_threads_per_block);
            mix(entry.resources.blocks_per_processor);mix(entry.resources.private_bytes);mix(entry.resources.static_shared_bytes);
        }
        if(router_publication_)
        {
            mix(router_publication_->gateup_table);mix(router_publication_->down_table);
            mix(static_cast<unsigned>(router_publication_->outputs.format));
            mix(static_cast<unsigned>(router_publication_->outputs.projections));
        }
        for(const auto& binding:scratch_)
        {mix(reinterpret_cast<std::uintptr_t>(binding.pointer));mix(binding.bytes);}
        mix(workspace_id_);
        mix(shared_tables_.gate_up);mix(shared_tables_.down);
        for(const auto& p:std::array{args_.routed_gate_policy,args_.routed_down_policy,args_.shared_gate_policy,args_.shared_down_policy})
        {mix(p.p0);mix(p.p1);mix(p.p2);mix(p.p3);mix(p.maximum);}
        if(native_router_)
        {
            mix(reinterpret_cast<std::uintptr_t>(native_router_->barrier_state));
            for(std::size_t i=0;i<router_tensors_.size();++i)
            {
                if(!router_tensors_[i])throw std::logic_error("Native-layout router tensor owner is absent");
                router_shapes_[i]=router_tensors_[i]->shape();
                mix(reinterpret_cast<std::uintptr_t>(router_tensors_[i]));
                for(const auto extent:router_shapes_[i])mix(extent);
            }
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_weights));
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_logits));
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_indices));
            mix(reinterpret_cast<std::uintptr_t>(native_router_->router_weights_output));
            mix(native_router_->normalize_router_weights);mix(native_router_->write_router_outputs);
            // Full-entry windows are immutable captured policy, independent of
            // current row count or routes. Standalone entries retain zero fields.
            mix(native_router_->full_entry_panel.rows);mix(native_router_->full_entry_panel.blocks);
        }
        if(full_prefix_)
        {
            if(!native_router_)throw std::logic_error("Full native HC requires its existing router/FFN owner");
            validateFullNativeHCFFNPrefixBinding(*full_prefix_,*native_router_);
            mixFullNativeHCFFNPrefix(*full_prefix_,mix);
        }
        if(!signature_)signature_=1;
    }
    /** @copydoc PreparedResidentMoEFFN::programEvidence */
    const ResidentMoEFFNProgramEvidence& programEvidence() const noexcept override {return admission_.evidence();}
    /** @copydoc PreparedResidentMoEFFN::captureSignature */
    std::uint64_t captureSignature() const noexcept override {return signature_;}
    /** @copydoc PreparedResidentMoEFFN::validateBinding */
    void validateBinding() const override
    {
        if(owner_->getStream()!=stream_||owner_->getWorkspace()!=workspace_||workspace_->id()!=workspace_id_)
            throw std::logic_error("Resident FFN changed its original stream/workspace owner");
        for(const auto& binding:scratch_)
            if(workspace_->getBuffer(binding.name)!=binding.pointer||workspace_->getBufferSize(binding.name)<binding.bytes)
                throw std::logic_error("Resident FFN original scratch storage was rebound");
        if(request_.gate_up_table>=int(owner_->grouped_gateup_desc_tables_.size())||
            request_.down_table>=int(owner_->grouped_down_desc_tables_.size())||
            shared_tables_.gate_up>=int(owner_->grouped_gateup_desc_tables_.size())||
            shared_tables_.down>=int(owner_->grouped_down_desc_tables_.size()))
            throw std::logic_error("Resident FFN original descriptor table retired");
        const auto& gate=owner_->grouped_gateup_desc_tables_[request_.gate_up_table];
        const auto& down=owner_->grouped_down_desc_tables_[request_.down_table];
        if(!gate.valid||!down.valid||gate.workspace_publication!=gate_publication_||
            down.workspace_publication!=down_publication_||((!router_publication_||native_router_)&&gate.device_gate_descs!=args_.gates)||
            ((!router_publication_||native_router_)&&(gate.device_up_descs!=args_.ups||down.device_descs!=args_.downs))||
            !gate.original_family.sameIdentity(request_.original)||!down.original_family.sameIdentity(request_.original)||
            gate.codebook_mask!=args_.execution.routed_gate||down.codebook_mask!=args_.execution.routed_down||
            gate.codebook_mask!=request_.original.nativeGateUpEnvelope().execution_codebook_mask||
            down.codebook_mask!=request_.original.nativeDownEnvelope().execution_codebook_mask||
            gate.policy_codebook_mask!=request_.original.nativeGateUpEnvelope().policy_codebook_mask||
            down.policy_codebook_mask!=request_.original.nativeDownEnvelope().policy_codebook_mask||
            gate.num_experts!=args_.experts||down.num_experts!=args_.experts||gate.d_model!=args_.hidden_columns||
            down.d_model!=args_.hidden_columns||gate.intermediate!=args_.routed_columns||down.intermediate!=args_.routed_columns)
            throw std::logic_error("Resident FFN original descriptor storage or source family changed");
        if(router_publication_&&!native_router_)
        {
            const auto& original=*router_publication_;
            const auto& current=owner_->requirePairedRouterDescriptors(request_.runtime,args_.experts,args_.top_k);
            if(current.gateup_table!=original.gateup_table||current.down_table!=original.down_table||
                current.gateup_table!=request_.gate_up_table||current.down_table!=request_.down_table||
                current.outputs.gate!=args_.gates||current.outputs.up!=args_.ups||current.outputs.down!=args_.downs||
                current.outputs.format!=DeviceMoEWeightFormat::NativeQuantised||
                current.outputs.projections!=DeviceMoEProjectionSet::IntermediateShard)
                throw std::logic_error("Ordinary FFN original router descriptor publication changed");
        }
        if(native_router_)
        {
            const auto barrier_bytes=full_prefix_?rocm::fullNativeHCFFNSlotBytes(full_prefix_->implementation):NativeLayoutBarrierLayout::bytes;
            auto* barrier_base=barrier_slot_?static_cast<std::uint32_t*>(workspace_->getPersistentSlotBuffer(
                MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                barrier_slot_->slot(),barrier_bytes)):nullptr;
            if(!barrier_base||native_router_->barrier_state!=barrier_base+
                (full_prefix_?FullNativeHCFFNBarrierLayout::suffix_word_offset:0))
                throw std::logic_error("Native-layout barrier slot binding changed");
            if(full_prefix_)
            {
                if(full_prefix_->barrier_state!=barrier_base)
                    throw std::logic_error("Full native HC prefix barrier slot binding changed");
                validateFullNativeHCFFNPrefixBinding(*full_prefix_,*native_router_);
            }
            const std::array<const void*,3> expected{native_router_->router_weights,native_router_->router_indices,native_router_->router_weights_output};
            for(std::size_t i=0;i<router_tensors_.size();++i)
                if(router_tensors_[i]->gpu_data_ptr()!=expected[i]||router_tensors_[i]->shape()!=router_shapes_[i])
                    throw std::logic_error("Native-layout router tensor storage or shape changed");
            const auto& publisher=owner_->requirePairedRouterDescriptors(request_.runtime,args_.experts,args_.top_k);
            if(!router_publication_||publisher.runtime!=router_publication_->runtime||
                publisher.workspace!=router_publication_->workspace||publisher.workspace_id!=router_publication_->workspace_id||
                publisher.stream!=router_publication_->stream||publisher.outputs.gate!=router_publication_->outputs.gate||
                publisher.outputs.up!=router_publication_->outputs.up||publisher.outputs.down!=router_publication_->outputs.down)
                throw std::logic_error("Native-layout router publication owner changed");
            if(publisher.gateup_table!=request_.gate_up_table||publisher.down_table!=request_.down_table||
                publisher.outputs.expert_count!=native_router_->router_publication.expert_count||
                publisher.outputs.format!=native_router_->router_publication.format||
                publisher.outputs.projections!=native_router_->router_publication.projections||
                publisher.outputs.gate!=native_router_->router_publication.gate||publisher.outputs.up!=native_router_->router_publication.up||
                publisher.outputs.down!=native_router_->router_publication.down||owner_->d_route_logits_!=native_router_->router_logits)
                throw std::logic_error("Native-layout FFN original router publication changed");
        }
        const auto& shared_gate=owner_->grouped_gateup_desc_tables_[shared_tables_.gate_up];
        const auto& shared_down=owner_->grouped_down_desc_tables_[shared_tables_.down];
        // Immutable source metadata is checked field by field, never by a live
        // source-data digest or a device readback. Table rewrites must reject
        // even when their publication object and device addresses survive.
        const auto same=[](const DeviceNativeQuantisedMatrixDesc& a,const DeviceNativeQuantisedMatrixDesc& b) {
            return a.payload==b.payload && a.scales==b.scales && a.mins==b.mins && a.emins==b.emins &&
                a.n==b.n && a.k==b.k && a.blocks_per_row==b.blocks_per_row && a.codebook_id==b.codebook_id &&
                a.allocation_payload_bytes_per_block==b.allocation_payload_bytes_per_block &&
                a.allocation_has_mins==b.allocation_has_mins && a.allocation_has_emins==b.allocation_has_emins &&
                a.source_codebook_id==b.source_codebook_id && a.source_is_superblock==b.source_is_superblock &&
                a.source_identity_present==b.source_identity_present && a.reserved==b.reserved && a.reserved_tail==b.reserved_tail;
        };
        if(!shared_gate.valid||!shared_down.valid||shared_gate.num_experts!=1||shared_down.num_experts!=1||
            shared_gate.weight_format!=DeviceMoEWeightFormat::NativeQuantised||shared_down.weight_format!=DeviceMoEWeightFormat::NativeQuantised||
            shared_gate.descriptor_source!=MoEDecodeDescriptorSource::StaticDescriptorTable||
            shared_down.descriptor_source!=MoEDecodeDescriptorSource::StaticDescriptorTable||
            shared_gate.workspace_publication!=shared_gate_publication_||shared_down.workspace_publication!=shared_down_publication_||
            shared_gate.device_gate_descs!=args_.shared_gate||shared_gate.device_up_descs!=args_.shared_up||
            shared_gate.codebook_mask!=args_.execution.shared_gate||shared_down.codebook_mask!=args_.execution.shared_down||
            shared_down.device_descs!=args_.shared_down||shared_gate.d_model!=args_.hidden_columns||
            shared_down.d_model!=args_.hidden_columns||shared_gate.intermediate!=args_.shared_columns||shared_down.intermediate!=args_.shared_columns||
            shared_gate.host_gate_descs.size()!=1||shared_gate.host_up_descs.size()!=1||shared_down.host_descs.size()!=1||
            !same(shared_gate.host_gate_descs[0],request_.shared.gate)||!same(shared_gate.host_up_descs[0],request_.shared.up)||
            !same(shared_down.host_descs[0],request_.shared.down))
            throw std::logic_error("Resident FFN original immutable shared descriptor publication changed");
        // Re-evaluate the arena's exact named table intervals. Retained leases
        // preserve old bytes after growth, but never authorize replaying them.
        const auto tableMatches=[&](const char* name,std::size_t slot,const void* pointer,std::size_t experts) {
            const auto available=workspace_->getBufferSize(name);
            const auto stride=(available/MoEWorkspaceBuffers::kGroupedDescriptorTableSlots/
                sizeof(DeviceNativeQuantisedMatrixDesc))*sizeof(DeviceNativeQuantisedMatrixDesc);
            const auto base=reinterpret_cast<std::uintptr_t>(workspace_->getBuffer(name));
            return base && slot<MoEWorkspaceBuffers::kGroupedDescriptorTableSlots &&
                stride>=experts*sizeof(DeviceNativeQuantisedMatrixDesc) &&
                (slot+1)*stride<=available &&
                reinterpret_cast<std::uintptr_t>(pointer)==base+slot*stride;
        };
        if(router_publication_&&!native_router_&&(
            !tableMatches(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,gate.workspace_slot,args_.gates,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,gate.workspace_slot,args_.ups,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,down.workspace_slot,args_.downs,args_.experts)))
            throw std::logic_error("Ordinary FFN original router descriptor arena changed");
        if(!tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,gate.workspace_slot,gate.device_gate_descs,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES,gate.workspace_slot,gate.device_up_descs,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES,down.workspace_slot,down.device_descs,args_.experts)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,shared_gate.workspace_slot,args_.shared_gate,1)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES,shared_gate.workspace_slot,args_.shared_up,1)||
            !tableMatches(MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES,shared_down.workspace_slot,args_.shared_down,1))
            throw std::logic_error("Resident FFN original named descriptor arena changed");
        const auto tensors=std::array<ITensor*,7>{request_.hidden,request_.shared_input_gate,request_.route_contributions,
            request_.routed_output,request_.shared_output,request_.combined_output,request_.shared_gate_scratch};
        const auto addresses=std::array<const void*,7>{args_.hidden,args_.shared_input_gate,args_.route_contributions,
            args_.routed_output,args_.shared_output,args_.combined_output,args_.shared_gate_output};
        for(std::size_t i=0;i<tensors.size();++i)
            if(addresses[i]&&(!tensors[i]||tensors[i]->gpu_data_ptr()!=addresses[i]))
                throw std::logic_error("Resident FFN public tensor storage changed");
        if(args_.shared_up_output&&(!request_.shared_up_scratch||request_.shared_up_scratch->gpu_data_ptr()!=args_.shared_up_output))
            throw std::logic_error("Resident FFN shared up storage changed");
    }
    /** @copydoc PreparedResidentMoEFFN::enqueue */
    void enqueue(void* stream) const override
    {
        if(stream!=stream_)throw std::logic_error("Resident FFN submission changed its original stream");
        validateBinding();
        if(full_prefix_)
        {
            const rocm::FullNativeHCFFNArgs full{*native_router_,*full_prefix_};
            if(full_prefix_->implementation==rocm::FullNativeHCFFNImplementation::RetainedGFX906M1) {
                const auto& evidence=admission_.evidence();
                const auto histogram=rocm::retainedM1NativeHistogramPolicy(request_.model_main_inference?
                    request_.model_main_inference->histogram_producer:MoERuntimeHistogramProducerRole::ProductionDecode);
                if(evidence.generation!=workspace_id_||evidence.nativeEntries().size()!=1||
                   evidence.nativeEntries().front().function!=rocm::retainedM1HCFFNKernelFunction(full.prefix.output_demand,
                       full.prefix.selected_descriptor_demand,full.suffix.ffn.execution,histogram))
                    throw std::logic_error("Retained M1 HC/FFN lost its exact prepared function");
                rocm::launchRetainedM1HCFFNDevice(full,request_.shared.gate,request_.shared.up,request_.shared.down,
                    rocm::retainedM1HCFFNScratch(full_prefix_->barrier_state),workspace_id_,stream,histogram);
                if(!retained_model_receipt_.empty()&&::llaminar2::isGraphCaptureActive())
                    PerfStatsCollector::addCounter("gpu_graph_inventory","retained_m1_layer_binding",1.0,
                        "graph_setup",evidence.device.toString(),retained_model_receipt_);
            } else rocm::launchFullNativeHCFFN(admission_,full,stream,workspace_id_);
            PerfStatsCollector::addCounter("kernel","rocm_full_native_hc_ffn_calls",1.0,"moe",
                admission_.evidence().device.toString(),{{"resident_grid",std::to_string(args_.resident_grid)}});
        }
        else if(native_router_)rocm::launchNativeLayoutRouterFFN(admission_,*native_router_,stream,workspace_id_);
        else rocm::launchResidentMoEFFN(admission_,args_,stream,workspace_id_);
        PerfStatsCollector::addCounter("kernel","rocm_resident_moe_ffn_calls",1.0,"moe",admission_.evidence().device.toString(),
            {{"top_k",std::to_string(args_.top_k)},{"routed_f",std::to_string(args_.routed_columns)},
             {"shared_f",std::to_string(args_.shared_columns)},{"resident_grid",std::to_string(args_.resident_grid)},{"native_launches",std::to_string(admission_.evidence().nativeEntries().size())}});
    }
private:
    /**
     * @brief Seal an opt-in diagnostic census from original model, detector and prepared launch owners.
     * @pre The exact registered HIP entry and its stream were admitted cold by the original backend.
     * @details Captured replay never resolves function names or constructs tags.
     */
    void sealRetainedModelReceipt()
    {
        if(!request_.model_main_inference ||
           !PerfStatsCollector::isRecordFamilyEnabled("gpu_graph_inventory","retained_m1_layer_binding"))return;
        if(!full_prefix_ || full_prefix_->implementation!=rocm::FullNativeHCFFNImplementation::RetainedGFX906M1)
            throw std::logic_error("Retained model receipt requires its original retained preparation");

        const auto sourceName=[](const ExpertWeightFormat& role) -> const char* {
            const auto source=role.native_quantised;
            if(role.isNativeQuantised() && source.present) {
                if(source.codebook_id==5 && source.is_superblock)return "Q4_K";
                if(source.codebook_id==7)return source.is_superblock?"Q5_K":"Q5_1";
                if(source.codebook_id==19 && !source.is_superblock)return "Q8_0";
            }
            throw std::logic_error("Retained model receipt lost its original GGUF source identity");
        };
        const auto& mode=*request_.model_main_inference;
        const auto& sources=request_.original.sourceFormats();
        const auto& evidence=admission_.evidence();
        const auto& entry=evidence.nativeEntries().front();
        const auto* symbol=hipKernelNameRefByPtr(entry.function,static_cast<hipStream_t>(stream_));
        if(!symbol||!*symbol||mode.source_layer<0||!barrier_slot_)
            throw std::logic_error("Retained model receipt requires its resolved HIP kernel and source layer");
        const auto& hardware=rocmHardwareMetadata().at(evidence.device.ordinal);
        retained_model_receipt_={
            {"layer",std::to_string(mode.source_layer)},
            {"h",std::to_string(args_.hidden_columns)},{"r",std::to_string(full_prefix_->hc.publication.columns)},
            {"f",std::to_string(args_.routed_columns)},{"s",std::to_string(args_.shared_columns)},
            {"experts",std::to_string(args_.experts)},{"top_k",std::to_string(args_.top_k)},
            {"f_start",std::to_string(args_.original.firstIntermediateColumn())},
            {"s_start",std::to_string(mode.shared_source_first_column)},
            {"gate",sourceName(sources[0])},{"up",sourceName(sources[1])},{"down",sourceName(sources[2])},
            {"pci",hardware.pci_address},{"phase","decode"},{"role","main_inference"},
            {"rows",std::to_string(args_.rows.physicalRows())},{"requests","1"},
            {"scratch_slot",std::to_string(barrier_slot_->slot())},
            {"kernel",symbol},{"grid",std::to_string(entry.grid)+"x1x1"},{"block",std::to_string(entry.threads)+"x1x1"},
        };
    }

    ROCmMoEKernel* owner_;
    const ResidentMoEFFNRequest request_;
    PerfStatsCollector::Tags retained_model_receipt_; ///< Cold optional model census; no device state is cached here.
    const rocm::ResidentMoEFFNArgs args_;
    const rocm::PreparedMoEFFNProgram admission_;
    const std::shared_ptr<PersistentWorkspaceSlotLease> barrier_slot_; ///< Exclusive barrier bytes until graph retirement.
    const std::optional<rocm::NativeLayoutRouterFFNArgs> native_router_;
    const std::optional<rocm::FullNativeHCFFNPrefixArgs> full_prefix_; ///< Original stage-owned phase snapshot, not another source/table owner.
    const std::array<TensorBase*,3> router_tensors_;
    std::array<std::vector<std::size_t>,3> router_shapes_;
    void* const stream_;
    DeviceWorkspaceManager* const workspace_;
    const std::uint64_t workspace_id_;
    const std::vector<ResidentFFNScratchBinding> scratch_;
    const ResidentFFNSharedTables shared_tables_;
    const std::optional<PairedRouterDescriptors> router_publication_; ///< Same original router's existing borrowed destinations.
    const std::shared_ptr<GroupedDescriptorWorkspacePublication> gate_publication_,down_publication_,
        shared_gate_publication_,shared_down_publication_;
    std::uint64_t signature_=kFNV1a64OffsetBasis;
};

/** @copydoc IResidentMoEFFNPreparation::prepareResidentMoEFFN */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareResidentMoEFFN(
    const ResidentMoEFFNRequest& request,void* stream)
{
    return prepareResidentMoEFFNImpl(request,stream,nullptr,nullptr,nullptr,false);
}

/** @copydoc ROCmMoEKernel::prepareNativeLayoutRouterFFN */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareNativeLayoutRouterFFN(
    const ResidentMoEFFNRequest& request,TensorBase* router,TensorBase* indices,TensorBase* weights,bool normalize,void* stream)
{
    if(!router||router->native_type()!=TensorType::FP16||!indices||!weights||
        indices->native_type()!=TensorType::FP32||weights->native_type()!=TensorType::FP32)
        throw std::invalid_argument("Native-layout router FFN requires the original prepared FP16 router and public outputs");
    const rocm::ScopedResidentMoEFFNProgram selection(rocm::ResidentMoEFFNProgram::NativeLayoutRouter);
    preparePairedRouterDecodeDescriptors(request.runtime,request.gate_up_table,request.down_table,request.top_k);
    return prepareResidentMoEFFNImpl(request,stream,router,indices,weights,normalize);
}

/** @copydoc ROCmMoEKernel::prepareFullNativeHCFFN */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareFullNativeHCFFN(
    const ResidentMoEFFNRequest& request,TensorBase* router,TensorBase* indices,TensorBase* weights,
    bool normalize,const rocm::FullNativeHCFFNPrefixArgs& prefix,void* stream)
{
    if(!router||router->native_type()!=TensorType::FP16||!indices||!weights||
        indices->native_type()!=TensorType::FP32||weights->native_type()!=TensorType::FP32||prefix.barrier_state)
        throw std::invalid_argument("Full native HC requires original FP16 router outputs and an unbound barrier");
    const rocm::ScopedResidentMoEFFNProgram selection(rocm::ResidentMoEFFNProgram::NativeLayoutRouter);
    preparePairedRouterDecodeDescriptors(request.runtime,request.gate_up_table,request.down_table,request.top_k);
    return prepareResidentMoEFFNImpl(request,stream,router,indices,weights,normalize,&prefix);
}

/** @brief Share the original source/table/arena resolver without preparing another native program. */
std::unique_ptr<PreparedResidentMoEFFN> ROCmMoEKernel::prepareResidentMoEFFNImpl(
    const ResidentMoEFFNRequest& request,void* stream,TensorBase* router,TensorBase* indices,TensorBase* weights,bool normalize,
    const rocm::FullNativeHCFFNPrefixArgs* prefix)
{
    const auto device=DeviceId::rocm(device_ordinal_);
    if(!workspace_||!stream||stream!=getStream()||::llaminar2::isGraphCaptureActive())
        throw std::logic_error("Resident FFN preparation requires its original workspace/stream outside capture");
    const auto& original=request.original.originalBinding();
    const int H=int(original.modelColumns()),F=int(original.localIntermediateColumns());
    if(!request.original.valid()||request.original.weightFormat()!=DeviceMoEWeightFormat::NativeQuantised||
        request.rows.physicalRows()!=1||request.rows.capacity()!=1||!request.runtime||!request.hidden||request.top_k<=0||
        request.top_k>int(kDeviceMoEMaxTopK)||request.gate_up_table<0||request.down_table<0||
        request.gate_up_table>=int(grouped_gateup_desc_tables_.size())||request.down_table>=int(grouped_down_desc_tables_.size()))
        throw std::invalid_argument("Resident FFN requires an original native-quantised ordinary M1 source declaration");
    const auto& gate=grouped_gateup_desc_tables_[request.gate_up_table];
    const auto& down=grouped_down_desc_tables_[request.down_table];
    if(!gate.valid||!down.valid||!gate.deviceReady()||!down.deviceReady()||gate.num_experts!=down.num_experts||
        gate.d_model!=H||down.d_model!=H||gate.intermediate!=F||down.intermediate!=F||
        gate.descriptor_source!=MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable||
        down.descriptor_source!=MoEDecodeDescriptorSource::IntermediateShardRuntimePlacementTable||
        !gate.original_family.sameIdentity(request.original)||!down.original_family.sameIdentity(request.original)||
        !request.original.validFor(device,gate.num_experts,H,F))
        throw std::invalid_argument("Resident FFN changed original private table/family identities");
    const auto routed_gate_envelope=request.original.nativeGateUpEnvelope();
    const auto routed_down_envelope=request.original.nativeDownEnvelope();
    if(!routed_gate_envelope.valid()||!routed_down_envelope.valid()||
        gate.codebook_mask!=routed_gate_envelope.execution_codebook_mask||
        down.codebook_mask!=routed_down_envelope.execution_codebook_mask||
        gate.policy_codebook_mask!=routed_gate_envelope.policy_codebook_mask||
        down.policy_codebook_mask!=routed_down_envelope.policy_codebook_mask)
        throw std::invalid_argument("Resident FFN table masks do not match the complete original source envelopes");
    const int shared_f=request.shared.gate.n;
    if(request.shared.weight_format!=DeviceMoEWeightFormat::NativeQuantised||shared_f<=0||shared_f%32||
        !validateGroupedGateUpDesc(request.shared.gate,H,shared_f)||!validateGroupedGateUpDesc(request.shared.up,H,shared_f)||
        !validateGroupedDownDesc(request.shared.down,H,shared_f))
        throw std::invalid_argument("Resident FFN shared source has invalid original native projection geometry");
    // The original prepared owner seals physical validity before these SAME
    // immutable shared tables become available to a full-entry consumer. This
    // does not certify allocation spans independently of the prepared owner.
    for(const auto& descriptor:std::array{request.shared.gate,request.shared.up,request.shared.down})
        if(!NativeQuantisedDescriptorValidation::publishable(descriptor))
            throw std::invalid_argument("Resident FFN shared publication has incomplete native physical planes");
    NativeQuantisedExecutionFormatEnvelope shared_gate,shared_down;
    const auto add=[](NativeQuantisedExecutionFormatEnvelope& envelope,const DeviceNativeQuantisedMatrixDesc& desc){
        return addNativeQuantisedExecutionFormat(envelope,desc.codebook_id,
            {.codebook_id=desc.source_codebook_id,.is_superblock=bool(desc.source_is_superblock),.present=bool(desc.source_identity_present)},false);
    };
    if(!add(shared_gate,request.shared.gate)||!add(shared_gate,request.shared.up)||!add(shared_down,request.shared.down))
        throw std::invalid_argument("Resident FFN shared source has no original arithmetic-policy identity");
    const auto pointer=[](ITensor* tensor,std::size_t elements)->float* {
        if(!tensor||tensor->native_type()!=TensorType::FP32||tensor->size_bytes()/sizeof(float)<elements||!tensor->gpu_data_ptr())
            throw std::invalid_argument("Resident FFN requires original materialized FP32 tensor storage");
        return static_cast<float*>(tensor->gpu_data_ptr());
    };
    rocm::ResidentMoEFFNArgs args;
    args.runtime=request.runtime;args.original=original;args.rows=request.rows;
    args.hidden=pointer(request.hidden,H);args.shared_input_gate=pointer(request.shared_input_gate,H);
    args.gates=gate.device_gate_descs;args.ups=gate.device_up_descs;args.downs=down.device_descs;
    const bool ordinary=rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::OrdinaryPhases;
    if(ordinary)
    {
        const auto& publisher=requirePairedRouterDescriptors(request.runtime,gate.num_experts,request.top_k);
        if(publisher.gateup_table!=request.gate_up_table||publisher.down_table!=request.down_table||
            publisher.outputs.format!=DeviceMoEWeightFormat::NativeQuantised||
            publisher.outputs.projections!=DeviceMoEProjectionSet::IntermediateShard)
            throw std::logic_error("Ordinary FFN requires its exact original router descriptor producer");
        args.gates=publisher.outputs.gate;args.ups=publisher.outputs.up;args.downs=publisher.outputs.down;
    }
    args.hidden_columns=H;args.routed_columns=F;args.shared_columns=shared_f;args.experts=gate.num_experts;args.top_k=request.top_k;
    const auto output_demand=prefix?prefix->output_demand:FullNativeHCFFNOutputDemand::all();
    const auto output=[&](FullNativeHCFFNOutputRole role,ITensor* tensor,std::size_t elements) {
        return output_demand.contains(role)?pointer(tensor,elements):nullptr;
    };
    args.route_contributions=output(FullNativeHCFFNOutputRole::RouteContributions,request.route_contributions,size_t(request.top_k)*H);
    args.routed_output=output(FullNativeHCFFNOutputRole::Routed,request.routed_output,H);
    args.shared_output=output(FullNativeHCFFNOutputRole::Shared,request.shared_output,H);
    args.combined_output=pointer(request.combined_output,H);
    args.shared_gate_output=output(FullNativeHCFFNOutputRole::SharedGate,request.shared_gate_scratch,shared_f);
    args.shared_up_output=output(FullNativeHCFFNOutputRole::SharedUp,request.shared_up_scratch,shared_f);
    struct Span{const void* pointer;std::size_t bytes;};
    const std::array public_reads{Span{args.hidden,size_t(H)*sizeof(float)},Span{args.shared_input_gate,size_t(H)*sizeof(float)}};
    const auto publication=[](const void* pointer,std::size_t bytes){return Span{pointer,pointer?bytes:0};};
    const std::array public_writes{publication(args.route_contributions,size_t(request.top_k)*H*sizeof(float)),
        publication(args.routed_output,size_t(H)*sizeof(float)),publication(args.shared_output,size_t(H)*sizeof(float)),
        publication(args.combined_output,size_t(H)*sizeof(float)),publication(args.shared_gate_output,size_t(shared_f)*sizeof(float)),
        publication(args.shared_up_output,size_t(shared_f)*sizeof(float))};
    const auto disjoint=[](Span a,Span b){const auto x=reinterpret_cast<std::uintptr_t>(a.pointer),y=reinterpret_cast<std::uintptr_t>(b.pointer);
        return !a.bytes||!b.bytes||(a.bytes<=std::numeric_limits<std::uintptr_t>::max()-x&&b.bytes<=std::numeric_limits<std::uintptr_t>::max()-y&&
            (x+a.bytes<=y||y+b.bytes<=x));};
    for(size_t i=0;i<public_writes.size();++i)
    {
        for(const auto& read:public_reads)if(!disjoint(public_writes[i],read))throw std::invalid_argument("Resident FFN output aliases an original input");
        for(size_t j=0;j<i;++j)if(!disjoint(public_writes[i],public_writes[j]))throw std::invalid_argument("Resident FFN public outputs alias");
    }
    std::vector<ResidentFFNScratchBinding> scratch;
    const auto bind=[&](const char* name,size_t bytes)->void* {
        auto* value=workspace_->getBuffer(name);
        if(!value||bytes>workspace_->getBufferSize(name))throw std::invalid_argument(std::string("Resident FFN original arena is insufficient: ")+name);
        const Span span{value,bytes};
        for(const auto& read:public_reads)if(!disjoint(span,read))throw std::invalid_argument("Resident FFN scratch aliases an original input");
        for(const auto& output:public_writes)if(!disjoint(span,output))throw std::invalid_argument("Resident FFN scratch aliases a public output");
        scratch.push_back({name,value,workspace_->retainBuffer(name,0,bytes),bytes});return value;
    };
    if(ordinary)
    {
        // These are the existing expert-indexed producer destinations, not a
        // second table or a new allocation. Retain their original named bytes.
        bind(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE));
        bind(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE));
        bind(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE,workspace_->getBufferSize(MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE));
    }
    const size_t intermediate_values=std::max(size_t(request.top_k)*F,size_t(shared_f));
    args.hidden_q8=static_cast<int8_t*>(bind(MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,H));
    args.hidden_scales=static_cast<float*>(bind(MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,size_t(H/32)*sizeof(float)));
    args.gate_partials=static_cast<float*>(bind(MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
        intermediate_values*NativeQuantisedGroupedDecodePolicy::maximum_k_partitions*sizeof(float)));
    args.up_partials=static_cast<float*>(bind(MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
        intermediate_values*NativeQuantisedGroupedDecodePolicy::maximum_k_partitions*sizeof(float)));
    args.down_partials=static_cast<float*>(bind(MoEWorkspaceBuffers::DOWN_PARTIALS,
        size_t(request.top_k)*H*NativeQuantisedGroupedDecodePolicy::maximum_k_partitions*sizeof(float)));
    args.swiglu_q8=static_cast<int8_t*>(bind(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,intermediate_values));
    args.swiglu_scales=static_cast<float*>(bind(MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,intermediate_values/32*sizeof(float)));
    const rocm::ResidentMoEFFNPolicyMasks masks{
        .routed_gate=gate.policy_codebook_mask,.routed_down=down.policy_codebook_mask,
        .shared_gate=shared_gate.policy_codebook_mask,.shared_down=shared_down.policy_codebook_mask,
        .execution={routed_gate_envelope.execution_codebook_mask,routed_down_envelope.execution_codebook_mask,
            shared_gate.execution_codebook_mask,shared_down.execution_codebook_mask}};
    // Reuse the original immutable descriptor publication infrastructure and
    // admitted slots. These are the same singleton shared tables the staged
    // SharedExpertFFNStage prepares; neither weights nor allocation authority
    // are duplicated. Appending these host vectors can invalidate gate/down
    // references above, so all original scalars/pointers are already copied.
    ResidentFFNSharedTables shared_tables;
    shared_tables.gate_up=uploadGroupedExpertGateUpDescriptorTables(&request.shared.gate,&request.shared.up,1,H,shared_f);
    shared_tables.down=uploadGroupedExpertDownDescriptorTable(&request.shared.down,1,H,shared_f);
    if(shared_tables.gate_up<0||shared_tables.down<0)
        throw std::runtime_error("Resident FFN could not publish its original immutable shared descriptors");
    const auto& shared_gate_table=grouped_gateup_desc_tables_[shared_tables.gate_up];
    const auto& shared_down_table=grouped_down_desc_tables_[shared_tables.down];
    args.shared_gate=shared_gate_table.device_gate_descs;args.shared_up=shared_gate_table.device_up_descs;
    args.shared_down=shared_down_table.device_descs;
    const auto retain=[&](const char* name,std::size_t slot) {
        const auto bytes=workspace_->getBufferSize(name);
        const auto stride=(bytes/MoEWorkspaceBuffers::kGroupedDescriptorTableSlots/sizeof(DeviceNativeQuantisedMatrixDesc))*
            sizeof(DeviceNativeQuantisedMatrixDesc);
        if(stride<sizeof(DeviceNativeQuantisedMatrixDesc)||slot>=MoEWorkspaceBuffers::kGroupedDescriptorTableSlots)
            throw std::logic_error("Resident FFN original immutable descriptor slot is incomplete");
        return workspace_->retainBuffer(name,slot*stride,sizeof(DeviceNativeQuantisedMatrixDesc));
    };
    shared_tables.leases={retain(MoEWorkspaceBuffers::ROCM_GROUPED_GATE_DESC_TABLES,shared_gate_table.workspace_slot),
        retain(MoEWorkspaceBuffers::ROCM_GROUPED_UP_DESC_TABLES,shared_gate_table.workspace_slot),
        retain(MoEWorkspaceBuffers::ROCM_GROUPED_DOWN_DESC_TABLES,shared_down_table.workspace_slot)};
    if(router)
    {
        // Shared descriptor registration above may grow the owning vectors.
        // Use the already copied geometry, never the earlier gate-table reference.
        const auto& publication=requirePairedRouterDescriptors(request.runtime,args.experts,request.top_k);
        if(!ensureRouteBufferCapacity(args.experts))throw std::logic_error("Native-layout router logits arena is unavailable");
        bind(MoEWorkspaceBuffers::ROUTE_LOGITS,std::size_t(args.experts)*sizeof(float));
        for(const auto* name:{MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
            MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE})
            bind(name,workspace_->getBufferSize(name));
        rocm::NativeLayoutRouterFFNArgs combined;
        auto barrier_slot=workspace_->acquirePersistentSlot(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
        if(!barrier_slot)throw std::logic_error("Native-layout barrier already has a retained owner");
        const auto barrier_bytes=prefix?rocm::fullNativeHCFFNSlotBytes(prefix->implementation):NativeLayoutBarrierLayout::bytes;
        const auto barrier_bank_bytes=workspace_->getBufferSize(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER);
        if(barrier_bank_bytes<barrier_bytes*MoEWorkspaceBuffers::kGroupedDescriptorTableSlots)
            throw std::invalid_argument("Native-layout barrier bank is smaller than its admitted slot payloads");
        // Slot addresses use the actual merged bank stride, not this caller's
        // payload width. A standalone 16-byte owner can occupy a late slot in a
        // bank enlarged for 56-byte full owners, so retain the actual named extent.
        bind(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,barrier_bank_bytes);
        auto* barrier_base=static_cast<std::uint32_t*>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
            barrier_slot->slot(),barrier_bytes));
        if(!barrier_base)throw std::logic_error("Native-layout barrier slot is unavailable");
        combined.barrier_state=barrier_base+(prefix?FullNativeHCFFNBarrierLayout::suffix_word_offset:0);
        const auto barrier_offset=reinterpret_cast<const std::byte*>(barrier_base)-
            static_cast<const std::byte*>(workspace_->getBuffer(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER));
        const auto barrier_lease=workspace_->retainBuffer(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,barrier_offset,barrier_bytes);
        if(!prefix)TransferEngine::instance().initializeWorkspaceMetadata(*barrier_lease,
            std::vector<std::byte>(barrier_bytes,std::byte{0}),stream);
        combined.ffn=args;combined.router_weights=router->gpu_data_ptr();combined.router_logits=d_route_logits_;
        combined.router_indices=static_cast<float*>(indices->gpu_data_ptr());
        combined.router_weights_output=static_cast<float*>(weights->gpu_data_ptr());
        combined.router_publication=publication.outputs;combined.normalize_router_weights=normalize;combined.write_router_outputs=true;
        if(router->size_bytes()!=std::size_t(args.experts)*H*sizeof(uint16_t)||
            indices->size_bytes()<std::size_t(request.top_k)*sizeof(float)||weights->size_bytes()<std::size_t(request.top_k)*sizeof(float)||
            !combined.router_weights||!combined.router_indices||!combined.router_weights_output)
            throw std::invalid_argument("Native-layout router original tensor extents are invalid");
        const std::array router_writes{Span{combined.router_indices,std::size_t(request.top_k)*sizeof(float)},
            Span{combined.router_weights_output,std::size_t(request.top_k)*sizeof(float)},
            Span{combined.router_logits,std::size_t(args.experts)*sizeof(float)}};
        for(std::size_t i=0;i<router_writes.size();++i)
        {
            if(!disjoint(router_writes[i],Span{combined.router_weights,router->size_bytes()}))
                throw std::invalid_argument("Native-layout router output aliases its original router weights");
            for(const auto& read:public_reads)if(!disjoint(router_writes[i],read))
                throw std::invalid_argument("Native-layout router output aliases original input");
            for(const auto& output:public_writes)if(!disjoint(router_writes[i],output))
                throw std::invalid_argument("Native-layout router output aliases FFN output");
            for(std::size_t j=0;j<i;++j)if(!disjoint(router_writes[i],router_writes[j]))
                throw std::invalid_argument("Native-layout router public outputs alias");
        }
        for(const auto& output:public_writes)
            if(!disjoint(output,Span{combined.router_weights,router->size_bytes()}))
                throw std::invalid_argument("Native-layout FFN output aliases original router weights");
        // Original source-policy tables are resolved by the same native authority.
        // The full branch admits only the actual complete function, never a
        // standalone HC or suffix function before replacing that admission.
        if(prefix)
        {
            rocm::FullNativeHCFFNArgs full{combined,*prefix};
            full.prefix.barrier_state=barrier_base;
            if(full.prefix.implementation==rocm::FullNativeHCFFNImplementation::RetainedGFX906M1)
                (void)rocm::ROCmRetainedM1HCFFNBinding::resolve(full,request.shared,masks);
            const auto admission=full.prefix.implementation==rocm::FullNativeHCFFNImplementation::RetainedGFX906M1
                ?rocm::prepareRetainedM1HCFFN(device,full,masks,workspace_->id(),
                    rocm::retainedM1NativeHistogramPolicy(request.model_main_inference?
                        request.model_main_inference->histogram_producer:MoERuntimeHistogramProducerRole::ProductionDecode))
                :rocm::prepareFullNativeHCFFN(device,full,masks,workspace_->id());
            validateFullNativeHCFFNPrefixBinding(full.prefix,full.suffix);
            validateFullNativeHCFFNStorage(full.prefix,full.suffix,scratch,*workspace_);
            TransferEngine::instance().initializeWorkspaceMetadata(*barrier_lease,
                std::vector<std::byte>(barrier_bytes,std::byte{0}),stream);
            return std::make_unique<PreparedResidentFFN>(*this,request,full.suffix.ffn,admission,stream,std::move(scratch),
                std::move(shared_tables),full.suffix,std::array<TensorBase*,3>{router,indices,weights},
                std::move(barrier_slot),full.prefix);
        }
        const auto admission=rocm::prepareNativeLayoutRouterFFN(device,combined,masks,workspace_->id());
        return std::make_unique<PreparedResidentFFN>(*this,request,combined.ffn,admission,stream,std::move(scratch),
            std::move(shared_tables),combined,std::array<TensorBase*,3>{router,indices,weights},std::move(barrier_slot));
    }
    const auto admission=rocm::prepareResidentMoEFFN(device,args,masks,workspace_->id());
    return std::make_unique<PreparedResidentFFN>(*this,request,args,admission,stream,std::move(scratch),std::move(shared_tables));
}

} // namespace llaminar2
