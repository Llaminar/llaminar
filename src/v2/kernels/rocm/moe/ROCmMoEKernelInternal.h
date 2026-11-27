/**
 * @file ROCmMoEKernelInternal.h
 * @brief Shared HIP extern "C" bridge declarations and internal descriptor/policy helpers for ROCmMoEKernel shards.
 */
#pragma once
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
#include "kernels/rocm/gemm/ROCmGroupedFFNFrontier.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmGroupedFFNFormatSelection.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"
#include "../gemm/ROCmResidentMoEFFN.h"
#include "../gemm/ROCmNativeLayoutRouterFFN.h"
#include "../gemm/ROCmFullNativeHCFFN.h"
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
#include "../../../execution/moe/MoEIntermediateShardDescriptorDeclaration.h"
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

    bool hipMoE_gate_logits_fp16_decode_equivalent_rows_with_policy(
        const float *hidden, const void *gate_weights_fp16, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream,
        const int *device_effective_seq_len, bool qualified_default);

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

    bool hipMoE_softmax_topk_decode_equivalent_rows_with_policy(
        float *logits,
        float *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        int device_idx, void *stream,
        const int *device_effective_seq_len,
        void *deferred_selected_route_ledger, bool qualified_default);

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

    bool hipMoE_build_active_expert_list_with_offset_slots(const llaminar2::MoEGroupingPackedDirectory &binding);

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

    bool hipMoE_commit_grouped_verifier_histograms_batch(
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

    /** @param engine Actual submitted policy. @return Stable capture-policy name. */
    const char *groupedEngineTag(llaminar2::rocm::MoEGroupedPrefillEngine engine)
    {
        using enum llaminar2::rocm::MoEGroupedPrefillEngine;
        switch (engine)
        {
        case OrderedDot4: return "dot4";
        case WmmaM16N32: return "wmma_n32";
        case WmmaM16N64: return "wmma_n64";
        case WideDot4Wmma: return "wide_dot4_wmma";
        case PopulationAdaptive: return "device_population_v1";
        case Automatic: break;
        }
        throw std::logic_error("unresolved grouped engine in completed submission");
    }

    /** @param role Exact successfully submitted tiled geometries.
     * @return Diagnostic roster of columns, rows, threads, ordered rows, floor and leaf count. */
    std::string groupedTileRoster(const llaminar2::rocm::MoEGroupedPrefillRoleSubmission &role)
    {
        std::string result;
        for (size_t i = 0; i < role.size; ++i)
        {
            const auto &tile = role.tiles[i];
            if (!result.empty()) result += ';';
            result += std::string(groupedEngineTag(tile.engine)) + ":" + std::to_string(tile.rows) + "x" +
                std::to_string(tile.columns) + "/threads=" + std::to_string(tile.threads) +
                "/ordered_rows=" + std::to_string(tile.ordered_rows) +
                "/ordered_columns=" + std::to_string(tile.ordered_columns) +
                "/floor=" + std::to_string(static_cast<int>(tile.floor)) + "/leaves=" + std::to_string(tile.count);
        }
        return result.empty() ? "none" : result;
    }

    /** @param role Accepted tiled leaves. @param route_owned Whether the direct family was submitted.
     * @return Retained route families; never a claim about the live replay choice. */
    std::string groupedRouteRoster(const llaminar2::rocm::MoEGroupedPrefillRoleSubmission &role, bool route_owned)
    {
        if (role.size) return route_owned ? "device_route_owned_or_expert_tiled" : "expert_tiled";
        return route_owned ? "route_owned" : "not_submitted";
    }

    /** @param submitted Completed low-level construction receipt.
     * @return Geometry tags from actual submissions, with mixed families represented explicitly. */
    MoEPrefillPairPolicyTags submittedMoEPrefillPairPolicyTags(
        const llaminar2::rocm::MoEGroupedPrefillSubmission &submitted)
    {
        using Receipt = llaminar2::rocm::MoEGroupedPrefillSubmission;
        if (submitted.state != Receipt::State::Complete)
            throw std::logic_error("grouped telemetry requires a successful submission receipt");
        const auto role_tags = [](const auto &role, bool route_owned) -> MoEPrefillPolicyTags {
            if (role.size == 0) return {route_owned ? "route_owned" : "none",route_owned ? "route_owned" : "none"};
            if (role.size != 1) return {"mixed","mixed"};
            const auto &tile = role.tiles[0];
            return {tile.ordered_rows ? std::to_string(tile.ordered_rows) + "/" + std::to_string(tile.rows)
                                      : std::to_string(tile.rows),
                    tile.ordered_columns ? std::to_string(tile.ordered_columns) + "/" + std::to_string(tile.columns)
                                         : std::to_string(tile.columns)};
        };
        return {role_tags(submitted.gateup,submitted.gateup_route_owned),
                role_tags(submitted.down,submitted.down_route_owned),
                submitted.policy.engine == llaminar2::rocm::MoEGroupedPrefillEngine::OrderedDot4
                    ? (submitted.exact_ordered_overlay ? "exact_overlay" : "generic")
                    : groupedEngineTag(submitted.policy.engine)};
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

