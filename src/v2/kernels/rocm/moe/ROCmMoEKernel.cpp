/**
 * @file ROCmMoEKernel.cpp
 * @brief ROCm MoE kernel implementation — calls extern "C" HIP bridges
 *
 * Separating .hip and .cpp allows hipcc to compile only the HIP code
 * without encountering issues with MPI or other complex C++ headers.
 */

#include "ROCmMoEKernel.h"
#include "ROCmMoEOverlayActivationPacketKernels.h"
#include "ROCmMoEOverlayDeviceControllerKernels.h"
#include "ROCmMoEOverlayEpochKernels.h"
#include "../gemm/HipBLASGemmKernel.h"
#include "../gemm/ROCmWeightPacker.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/DeviceId.h"
#include "../../../tensors/ITensor.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/NativeVnniFormatInfo.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/ROCmKernelProfiler.h"

#include <hip/hip_runtime.h>
#include <algorithm>
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
    constexpr const char *kROCmRouterFP16GatePublicationDomain =
        "rocm_moe_router_fp16_gate_cache";

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
                static_cast<std::uint32_t>(key.num_experts)),
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
        const llaminar2::DeviceNativeVNNIMatrixDesc &descriptor)
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
        const llaminar2::DeviceNativeVNNIMatrixDesc *primary,
        const llaminar2::DeviceNativeVNNIMatrixDesc *secondary,
        int num_experts,
        int d_model,
        int intermediate)
    {
        llaminar2::PersistentWorkspacePublicationKey key{
            .word0 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(num_experts)),
            .word1 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(d_model)),
            .word2 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(intermediate)),
            .word3 = secondary ? 2U : 1U,
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
        int intermediate)
    {
        llaminar2::PersistentWorkspacePublicationKey key{
            .word0 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(num_experts)),
            .word1 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(d_model)),
            .word2 = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(intermediate)),
            .word3 = 0x100U | (secondary ? 2U : 1U) |
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

    /// @brief Returns whether grouped MoE kernels instantiate this native-VNNI codebook.
    bool groupedDecodeSupportsCodebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:  // Q4_0
        case 4:  // IQ4_NL / IQ4_XS
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
        case llaminar2::kNativeVnniExpandedInt8MinCodebook:
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
        case 5:  // Q4_1 / Q4_K min correction
        case 7:  // Q5_1 / Q5_K min correction
        case 8:  // Q6_K high half scale
        case 9:  // Q3_K high half scale
        case 10: // Q2_K high half scale
        case 13: // IQ2_S high half scale
        case 14: // IQ2_XS high half scale
        case 16: // IQ1_S min correction
        case 17: // IQ1_M high half scale
        case llaminar2::kNativeVnniExpandedInt8MinCodebook: // Expanded asymmetric min correction
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
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc)
    {
        if (!desc.source_identity_present)
        {
            return desc.codebook_id !=
                   llaminar2::kNativeVnniExpandedInt8MinCodebook;
        }
        const auto *source =
            llaminar2::native_vnni_formats::forSourceIdentity(
                desc.source_codebook_id,
                desc.source_is_superblock != 0);
        return source != nullptr &&
               llaminar2::deviceVnniExecutionCompatibleWithSource(
                   *source, desc.codebook_id);
    }

    /** @return Bit selecting the serial-M1 source arithmetic policy. */
    uint32_t groupedPrefillPolicyCodebookBit(
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc)
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
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
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
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
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

    bool isBlankGroupedDesc(const llaminar2::DeviceNativeVNNIMatrixDesc &desc)
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

    bool hipMoE_fp32_to_fp16(
        const float *input, void *output_fp16, int count,
        int device_idx, void *stream);

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

    bool hipMoE_quantize_router_gate_q8(
        const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_q8_weights(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
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
        bool fully_replicated_local_rows,
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
        int device_idx, void *stream);

    bool hipMoE_softmax_topk_decode_runtime_wave64(
        float *logits,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        bool fully_replicated_local_rows,
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
        int device_idx, void *stream);

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
        bool fully_replicated_local_rows,
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
        int device_idx, void *stream);

    bool hipMoE_decode_route_select_runtime(
        const int *expert_indices,
        const float *expert_weights,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
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
        int device_idx, void *stream);

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
        int seq_len,
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
        int device_idx, void *stream);

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
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
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
        void *stream);

    bool hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
        void *runtime,
        int *original_to_grouped,
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int retain_routes_for_deferred_commit,
        llaminar2::DeviceMoEWeightFormat expected_format,
        int device_idx,
        void *stream);

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

    bool hipMoE_materialize_runtime_prefill_descriptor_tables(
        const void *runtime,
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int device_idx,
        void *stream);

    bool hipMoE_materialize_runtime_floating_descriptor_tables(
        const void *runtime,
        llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *down_descs,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream);

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

    bool rocmMoE_grouped_swiglu_down_native_vnni_decode(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_descs,
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

    bool rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_desc_table,
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
        void *stream);

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

    bool rocmMoE_grouped_gate_up_native_vnni_decode_table(
        const float *d_hidden,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
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
        void *stream);

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

    bool rocmMoE_grouped_gate_up_native_vnni_decode_runtime(
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

    bool rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime(
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

    bool rocmMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const int8_t *d_prequantized_hidden,
        const float *d_prequantized_hidden_scales,
        const void *d_gate_desc_table,
        const void *d_up_desc_table,
        const void *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const int *d_group_token_indices,
        const int *d_original_to_grouped,
        const int *d_original_expert_ids,
        const float *d_group_weights,
        int *d_group_work_directory,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_output,
        float *d_canonical_route_contributions,
        int num_experts,
        int d_model,
        int intermediate,
        int total_slots,
        int top_k,
        int grouped_indices_are_route_slots,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        uint32_t gateup_policy_codebook_mask,
        uint32_t down_policy_codebook_mask,
        int device_id,
        void *stream);

    bool rocmMoE_grouped_floating_prefill_pipeline(
        const float *d_hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_up_desc_table,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_down_desc_table,
        const int *d_original_to_grouped,
        const int *d_original_expert_ids,
        const float *d_grouped_weights,
        float *d_grouped_gate,
        float *d_grouped_up,
        float *d_output,
        float *d_canonical_route_contributions,
        int seq_len,
        int total_slots,
        int top_k,
        int d_model,
        int intermediate,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_id,
        void *stream);

    bool rocmMoE_reduce_canonical_route_contributions(
        const float *d_route_contributions,
        float *d_output,
        int seq_len,
        int top_k,
        int d_model,
        int device_id,
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
     * @return The unique codebook id, or -1 when the table is empty or mixed.
     */
    int singleCodebookFromMask(uint32_t codebook_mask)
    {
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
        int top_k)
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
        if (!rocmMoE_grouped_prefill_query_production_pair_config(
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
        DeviceNativeVNNIMatrixDesc **device_descs,
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
                sizeof(DeviceNativeVNNIMatrixDesc),
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
            slot_bytes / sizeof(DeviceNativeVNNIMatrixDesc);
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
            available / sizeof(DeviceNativeVNNIMatrixDesc);
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
            static_cast<DeviceNativeVNNIMatrixDesc *>(base) + descriptor_offset;
        return true;
    }

    void ROCmMoEKernel::clearWorkspaceScratchBindings() noexcept
    {
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
        router_fp16_gate_cache_.clear();
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
                        table.intermediate),
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                    [&](std::size_t slot) -> std::shared_ptr<void>
                    {
                        DeviceNativeVNNIMatrixDesc *slot_base = nullptr;
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
                        err = hipMemcpyAsync(
                            device_descs,
                            table.host_floating_descs.data(),
                            desc_bytes,
                            hipMemcpyHostToDevice,
                            stream);
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
            sizeof(DeviceNativeVNNIMatrixDesc);
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    table.host_descs.data(),
                    nullptr,
                    table.num_experts,
                    table.d_model,
                    table.intermediate),
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    DeviceNativeVNNIMatrixDesc *device_descs = nullptr;
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

                    err = hipMemcpyAsync(
                        device_descs,
                        table.host_descs.data(),
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
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

        table.device_descs = static_cast<DeviceNativeVNNIMatrixDesc *>(
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
                        table.intermediate),
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                    [&](std::size_t slot) -> std::shared_ptr<void>
                    {
                        DeviceNativeVNNIMatrixDesc *gate_slot_base = nullptr;
                        DeviceNativeVNNIMatrixDesc *up_slot_base = nullptr;
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
                        err = hipMemcpyAsync(
                            device_gate_descs,
                            table.host_floating_gate_descs.data(),
                            desc_bytes,
                            hipMemcpyHostToDevice,
                            stream);
                        if (err != hipSuccess)
                            return {};
                        err = hipMemcpyAsync(
                            device_up_descs,
                            table.host_floating_up_descs.data(),
                            desc_bytes,
                            hipMemcpyHostToDevice,
                            stream);
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
            sizeof(DeviceNativeVNNIMatrixDesc);
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    table.host_gate_descs.data(),
                    table.host_up_descs.data(),
                    table.num_experts,
                    table.d_model,
                    table.intermediate),
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    DeviceNativeVNNIMatrixDesc *device_gate_descs =
                        nullptr;
                    DeviceNativeVNNIMatrixDesc *device_up_descs =
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

                    err = hipMemcpyAsync(
                        device_gate_descs,
                        table.host_gate_descs.data(),
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped gate descriptor "
                                  "H2D publication failed: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    err = hipMemcpyAsync(
                        device_up_descs,
                        table.host_up_descs.data(),
                        desc_bytes,
                        hipMemcpyHostToDevice,
                        stream);
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

        table.device_gate_descs = static_cast<DeviceNativeVNNIMatrixDesc *>(
            publication->primary_descs);
        table.device_up_descs = static_cast<DeviceNativeVNNIMatrixDesc *>(
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
        router_q8_hidden_publication_ = std::move(publication);
        router_q8_publication_access_ = access;
        return true;
    }

    void ROCmMoEKernel::publishRouterQ8Hidden(
        const float *source,
        int rows,
        bool recorded_during_capture) noexcept
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
    }

    bool ROCmMoEKernel::canReuseRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model) const noexcept
    {
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
                workspace_->id(), gate_device_ptr, d_model, num_experts);
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
                            stream))
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

    const void *ROCmMoEKernel::getOrCreateFP16RouterGateCache(
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().rocm.moe_router_fp16)
            return nullptr;
        if (!gate_device_ptr || d_model <= 0 || num_experts <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] invalid FP16 router gate request");
            return nullptr;
        }
        if (!workspace_)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] "
                      "graph-owned workspace is required before cache publication");
            return nullptr;
        }

        const DeviceResidentRouterGateCacheKey cache_key =
            DeviceResidentRouterGateCacheKey::make(
                workspace_->id(), gate_device_ptr, d_model, num_experts);
        for (const auto &entry : router_fp16_gate_cache_)
        {
            if (entry.key == cache_key &&
                entry.d_gate_weights_fp16)
            {
                return entry.d_gate_weights_fp16;
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
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] FP16 router cache miss during graph capture");
            return nullptr;
        }
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] explicit HIP stream is required");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] router gate size overflow");
            return nullptr;
        }
        const size_t element_count = d_model_sz * experts_sz;
        if (element_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] router gate too large for conversion kernel: elements="
                      << element_count);
            return nullptr;
        }

        if (!setMoEDevice(device_ordinal_, "getOrCreateFP16RouterGateCache"))
            return nullptr;

        void *gate_weights_base = nullptr;
        const size_t weights_payload_bytes =
            element_count * sizeof(uint16_t);
        if (!bindWorkspaceBuffer(&gate_weights_base,
                                 MoEWorkspaceBuffers::ROCM_ROUTER_FP16_GATE_WEIGHTS,
                                 weights_payload_bytes,
                                 "ROCm FP16 router gate weights"))
        {
            return nullptr;
        }

        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmRouterFP16GatePublicationDomain,
                routerPublicationKey(cache_key),
                MoEWorkspaceBuffers::kRouterGateCacheSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    void *weights = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::ROCM_ROUTER_FP16_GATE_WEIGHTS,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        weights_payload_bytes);
                    if (!weights)
                        return {};

                    hipEvent_t ready_event = nullptr;
                    hipError_t event_err = hipEventCreateWithFlags(
                        &ready_event, hipEventDisableTiming);
                    if (event_err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] "
                                  "failed to create immutable publication event: "
                                  << hipGetErrorString(event_err));
                        return {};
                    }

                    auto publication =
                        std::shared_ptr<RouterGateWorkspacePublication>(
                            new RouterGateWorkspacePublication{
                                .ready_event = static_cast<void *>(ready_event),
                                .primary_weights = weights,
                                .scales = nullptr,
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

                    if (!hipMoE_fp32_to_fp16(
                            gate_device_ptr,
                            weights,
                            static_cast<int>(element_count),
                            device_ordinal_,
                            stream))
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] "
                                  "FP32->FP16 router gate conversion launch failed");
                        return {};
                    }

                    event_err = hipEventRecord(
                        ready_event, static_cast<hipStream_t>(stream));
                    if (event_err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] "
                                  "failed to record immutable publication event: "
                                  << hipGetErrorString(event_err));
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] "
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
            LOG_ERROR("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] "
                      "failed to adopt immutable router publication: "
                      << hipGetErrorString(wait_err));
            return nullptr;
        }

        RouterFP16GateCacheEntry entry{};
        entry.key = cache_key;
        entry.element_count = element_count;
        entry.d_gate_weights_fp16 = publication->primary_weights;
        entry.workspace_publication = std::move(publication);
        entry.workspace_slot = publication_result.slot;
        router_fp16_gate_cache_.push_back(std::move(entry));

        LOG_DEBUG("[ROCmMoEKernel] Adopted FP16 router gate source_device="
                  << static_cast<const void *>(gate_device_ptr)
                  << " workspace_id=" << cache_key.workspace_id
                  << " slot=" << publication_result.slot
                  << " created=" << publication_result.created
                  << " shape=[" << num_experts << "," << d_model << "] bytes="
                  << (element_count * sizeof(uint16_t)));
        return router_fp16_gate_cache_.back().d_gate_weights_fp16;
    }

    // =========================================================================
    // routeCore() — Shared GPU routing logic: gate GEMM + direct FP32 publication.
    // =========================================================================

    bool ROCmMoEKernel::routeCore(
        const float *hidden, const void *gate_weights, TensorType gate_type,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        float *output_indices, float *output_weights,
        const int *device_effective_seq_len)
    {
        if (!output_indices || !output_weights)
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

        const bool decode_single_token = (seq_len == 1);
        bool used_q8_grouped_router = false;
        bool used_fp16_grouped_router = false;
        invalidateRouterQ8HiddenPublication();
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
                    logits_ready = hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
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
                        device_effective_seq_len);
                    if (logits_ready)
                    {
                        used_q8_grouped_router = true;
                        publishRouterQ8Hidden(
                            hidden,
                            seq_len,
                            isGraphCaptureActive() ||
                                (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
                                isHipStreamCapturing(getStream()));
                    }
                }
                else if (rocm_env.moe_router_fp16)
                {
                    const void *gate_fp16 = getOrCreateFP16RouterGateCache(
                        static_cast<const float *>(gate_weights),
                        d_model,
                        num_experts);
                    if (!gate_fp16)
                    {
                        LOG_ERROR("[ROCmMoEKernel::routeCore] FP16 prefill router gate cache unavailable");
                        return false;
                    }
                    logits_ready = hipMoE_gate_logits_fp16_decode_equivalent_rows(
                        hidden,
                        gate_fp16,
                        d_route_logits_,
                        seq_len,
                        d_model,
                        num_experts,
                        device_ordinal_,
                        getStream(),
                        device_effective_seq_len);
                    used_fp16_grouped_router = logits_ready;
                }
                else
                {
                    logits_ready = launchDecodeEquivalentFP32Rows(
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
                logits_ready = hipMoE_gate_logits_fp16_decode_equivalent_rows(
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
                logits_ready = hipMoE_gate_logits_bf16_decode_equivalent_rows(
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
        if (!hipMoE_softmax_topk_decode_equivalent_rows(
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

    bool ROCmMoEKernel::groupTokensByExpertDevice(
        const int *d_routing_indices,
        const float *d_routing_weights,
        int seq_len, int num_experts, int top_k,
        int *d_expert_offsets,
        int *d_expert_counts,
        int *d_grouped_token_indices,
        float *d_grouped_weights)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "groupTokensByExpertDevice"))
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        // Step 1: Zero expert_counts
        err = hipMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync expert_counts failed: "
                      << hipGetErrorString(err));
            return false;
        }

        if (d_group_original_to_grouped_)
        {
            err = hipMemsetAsync(d_group_original_to_grouped_, 0xff,
                                 static_cast<size_t>(total_slots) * sizeof(int), stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync original_to_grouped failed: "
                          << hipGetErrorString(err));
                return false;
            }
        }
        /*
         * Masked/padded routing can leave the compact grouped tail unused.  The
         * grouped prefill gather is sized by total_slots, so clear stale rows
         * before scatter publishes the valid compact prefix.
         */
        err = hipMemsetAsync(d_grouped_token_indices, 0,
                             static_cast<size_t>(total_slots) * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync grouped_token_indices failed: "
                      << hipGetErrorString(err));
            return false;
        }
        err = hipMemsetAsync(d_grouped_weights, 0,
                             static_cast<size_t>(total_slots) * sizeof(float), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync grouped_weights failed: "
                      << hipGetErrorString(err));
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

        // Step 4: Lazy-allocate write_heads scratch buffer
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

        // Zero write_heads
        err = hipMemsetAsync(d_write_heads_, 0, num_experts * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync write_heads failed: "
                      << hipGetErrorString(err));
            return false;
        }

        // Step 5: Scatter tokens into grouped arrays
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
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeVNNIMatrixDesc),
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
                        NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
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
                        NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(intermediate) * sizeof(float),
                "ensureGroupedGateUpCapacity(gate_partials)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_grouped_gateup_up_partials_),
                MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
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
            LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Grouped gate/up workspace is not bound");
            return false;
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
            LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Grouped down workspace is not bound");
            return false;
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
            gate_weights->native_type() != TensorType::BF16)
        {
            LOG_ERROR("[" << kContext << "] router gate must be FP32 or BF16, got "
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

        const bool fp16_cache_is_consumed =
            rocm_env.moe_router_fp16 &&
            (plan.kind != MoERouteLaunchKind::GroupedPrefill ||
             plan.physical_rows > 1);
        if (fp16_cache_is_consumed &&
            !getOrCreateFP16RouterGateCache(
                gate_device,
                plan.d_model,
                plan.num_experts))
        {
            LOG_ERROR("[" << kContext << "] failed to publish immutable FP16 router gate");
            return false;
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
                    device_effective_seq_len))
            {
                LOG_ERROR("[" << kContext << "] grouped Q8 verifier logits failed");
                return false;
            }
            used_q8_grouped_router = true;
            publishRouterQ8Hidden(
                d_hidden,
                seq_len,
                isDecodeGraphCaptureActive());
        }
        else if (gate_is_fp32)
        {
            const void *g_fp16 = getOrCreateFP16RouterGateCache(
                static_cast<const float *>(d_gate), d_model, num_experts);
            if (rocm_env.moe_router_fp16 && !g_fp16)
            {
                LOG_ERROR("[" << kContext << "] FP16 router was requested but gate cache is unavailable");
                return false;
            }
            if (g_fp16)
            {
                if (!hipMoE_gate_logits_fp16_decode_equivalent_rows(
                        d_hidden,
                        g_fp16,
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
            else if (!launchDecodeEquivalentFP32Rows(
                         d_hidden,
                         static_cast<const float *>(d_gate),
                         d_route_logits_,
                         seq_len,
                         d_model,
                         num_experts,
                         device_ordinal_,
                         getStream(),
                         device_effective_seq_len))
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
        RoutedExpertRowExecutionPolicy row_execution_policy)
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
        const bool fully_replicated_local_rows =
            row_execution_policy ==
            RoutedExpertRowExecutionPolicy::FullyReplicatedLocal;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        invalidateRouterQ8HiddenPublication();
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
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
                device_ordinal_, stream);
            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router logits kernel failed");
                return false;
            }
            publishRouterQ8Hidden(
                h,
                /*rows=*/1,
                isDecodeGraphCaptureActive());
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
                fully_replicated_local_rows,
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
                getStream());
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] fused K-part router runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !logits_ready && gate_is_fp32)
        {
            const void *g_fp16 = getOrCreateFP16RouterGateCache(
                static_cast<const float *>(g), d_model, num_experts);
            if (rocm_env.moe_router_fp16 && !g_fp16)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] FP16 router was requested but gate cache is unavailable");
                return false;
            }
            if (g_fp16)
            {
                logits_ready = hipMoE_gate_logits_single_token_fp16_weights(
                    h, g_fp16, d_route_logits_,
                    d_model, num_experts,
                    device_ordinal_, getStream());
                if (!logits_ready)
                {
                    LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] FP16 router logits kernel failed");
                    return false;
                }
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

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= 256)
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
                fully_replicated_local_rows,
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
                getStream());
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
                                  fully_replicated_local_rows,
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
                                  getStream()))
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
        RoutedExpertRowExecutionPolicy row_execution_policy)
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
        const bool fully_replicated_local_rows =
            row_execution_policy ==
            RoutedExpertRowExecutionPolicy::FullyReplicatedLocal;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        invalidateRouterQ8HiddenPublication();
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
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
                device_ordinal_, stream);
            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router logits kernel failed");
                return false;
            }
            publishRouterQ8Hidden(
                h,
                /*rows=*/1,
                isDecodeGraphCaptureActive());
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
                fully_replicated_local_rows,
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
                getStream());
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] fused K-part router runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !logits_ready && gate_is_fp32)
        {
            const void *g_fp16 = getOrCreateFP16RouterGateCache(
                static_cast<const float *>(g), d_model, num_experts);
            if (rocm_env.moe_router_fp16 && !g_fp16)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] FP16 router was requested but gate cache is unavailable");
                return false;
            }
            if (g_fp16)
            {
                logits_ready = hipMoE_gate_logits_single_token_fp16_weights(
                    h, g_fp16, d_route_logits_,
                    d_model, num_experts,
                    device_ordinal_, getStream());
                if (!logits_ready)
                {
                    LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] FP16 router logits kernel failed");
                    return false;
                }
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

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= 256)
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
                fully_replicated_local_rows,
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
                getStream());
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
                                  fully_replicated_local_rows,
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
                                  getStream()))
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
        DeviceMoELLEPLayerPlanScratch *llep_layer_plans)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !gathered_histograms || !status || !controller_state)
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] runtime layers, gathered histograms, status, and persistent controller state must be non-null");
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
        const DeviceMoERebalanceApplyStatus *apply_status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finalizeMoEOverlayRebalancePublication");
        if (!runtime_layers || layer_count == 0u || expert_count == 0u ||
            !control || !candidate_epoch ||
            !reservation_and_publication_status || !apply_status || !stream)
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
            sharedExpertGateFromTensors(input, gate_inp, shared_output, seq_len, d_model);
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
        const DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
    {
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
            descriptor_source ==
            MoEDecodeDescriptorSource::RuntimePlacementTable;
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeVnniExecutionFormatEnvelope format_envelope;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;

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
            if (!addNativeVnniExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeVnniSourceIdentity{
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
            const uint32_t bit = nativeVnniCodebookMaskBit(codebook);
            if ((codebook_mask & bit) == 0u)
                continue;
            if (groupedPrefillCodebookBit(codebook) == 0u)
                return -1;
            if ((codebook_mask & (codebook_mask - 1u)) == 0u)
                codebook_id = codebook;
        }

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
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                existing.weight_format != DeviceMoEWeightFormat::NativeVNNI ||
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
        table.weight_format = DeviceMoEWeightFormat::NativeVNNI;
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

    int ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables(
        const DeviceNativeVNNIMatrixDesc *gate_descs,
        const DeviceNativeVNNIMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
    {
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
            descriptor_source ==
            MoEDecodeDescriptorSource::RuntimePlacementTable;
        if (!runtime_mutable &&
            descriptor_source !=
                MoEDecodeDescriptorSource::StaticDescriptorTable)
        {
            return -1;
        }
        NativeVnniExecutionFormatEnvelope format_envelope;
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

            if (!gate_desc.valid() || !up_desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (gate_desc.codebook_id != up_desc.codebook_id)
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
                                            const DeviceNativeVNNIMatrixDesc &desc)
            {
                return addNativeVnniExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeVnniSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!add_descriptor(gate_desc) || !add_descriptor(up_desc))
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
            const uint32_t bit = nativeVnniCodebookMaskBit(codebook);
            if ((codebook_mask & bit) == 0u)
                continue;
            if (groupedPrefillCodebookBit(codebook) == 0u)
                return -1;
            if ((codebook_mask & (codebook_mask - 1u)) == 0u)
                codebook_id = codebook;
        }

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
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                existing.weight_format != DeviceMoEWeightFormat::NativeVNNI ||
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
        table.weight_format = DeviceMoEWeightFormat::NativeVNNI;
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
        int intermediate)
    {
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
            if (!validateGroupedFloatingDesc(desc, d_model, intermediate))
                return -1;
            any_live = true;
        }
        if (!any_live)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid || existing.weight_format != weight_format ||
                !existing.device_floating_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
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
        int intermediate)
    {
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
            if (!validateGroupedFloatingDesc(
                    gate_descs[expert], intermediate, d_model) ||
                !validateGroupedFloatingDesc(
                    up_descs[expert], intermediate, d_model))
            {
                return -1;
            }
            any_live = true;
        }
        if (!any_live)
            return -1;

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) *
            sizeof(DeviceMoEFloatingMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid || existing.weight_format != weight_format ||
                !existing.device_floating_gate_descs ||
                !existing.device_floating_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
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
        const DeviceNativeVNNIMatrixDesc *down_descs,
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
            table.weight_format != DeviceMoEWeightFormat::NativeVNNI ||
            !table.device_descs ||
            !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            table.descriptor_source ==
            MoEDecodeDescriptorSource::RuntimePlacementTable;
        NativeVnniExecutionFormatEnvelope format_envelope;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (!desc.valid() || !validateGroupedDownDesc(desc, d_model, intermediate))
                return false;
            if (!addNativeVnniExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeVnniSourceIdentity{
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
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    down_descs,
                    nullptr,
                    num_experts,
                    d_model,
                    intermediate),
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
        const DeviceNativeVNNIMatrixDesc *gate_descs,
        const DeviceNativeVNNIMatrixDesc *up_descs,
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
            table.weight_format != DeviceMoEWeightFormat::NativeVNNI ||
            !table.device_gate_descs ||
            !table.device_up_descs || !table.workspace_publication ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        const bool runtime_mutable =
            table.descriptor_source ==
            MoEDecodeDescriptorSource::RuntimePlacementTable;
        NativeVnniExecutionFormatEnvelope format_envelope;
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
            if (!gate_desc.valid() || !up_desc.valid() ||
                gate_desc.codebook_id != up_desc.codebook_id ||
                !validateGroupedGateUpDesc(gate_desc, d_model, intermediate) ||
                !validateGroupedGateUpDesc(up_desc, d_model, intermediate))
            {
                return false;
            }
            const auto add_descriptor = [&](
                                            const DeviceNativeVNNIMatrixDesc &desc)
            {
                return addNativeVnniExecutionFormat(
                    format_envelope,
                    desc.codebook_id,
                    NativeVnniSourceIdentity{
                        .codebook_id = desc.source_codebook_id,
                        .is_superblock =
                            desc.source_is_superblock != 0u,
                        .present =
                            desc.source_identity_present != 0u,
                    },
                    runtime_mutable);
            };
            if (!add_descriptor(gate_desc) || !add_descriptor(up_desc))
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
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        if (!workspace_->rewritePersistentPublication(
                kROCmGroupedGateUpDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    gate_descs,
                    up_descs,
                    num_experts,
                    d_model,
                    intermediate),
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
                    intermediate),
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
                    intermediate),
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

        if (!ensureGroupedGateUpCapacity(
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
            : rocmMoE_grouped_gate_up_native_vnni_decode_table(
                  d_hidden,
                  table.device_gate_descs,
                  table.device_up_descs,
                  fixed_device_expert_ids,
                  d_gate_output_ptrs,
                  d_up_output_ptrs,
                  nullptr,
                  nullptr,
                  d_grouped_hidden_int8_,
                  d_grouped_hidden_scales_,
                  d_grouped_gateup_gate_partials_,
                  d_grouped_gateup_up_partials_,
                  false,
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
        const bool ok = rocmMoE_grouped_gate_up_native_vnni_decode_table(
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

        const bool ok = rocmMoE_grouped_gate_up_native_vnni_decode_runtime(
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
        MoEDecodeDescriptorSource descriptor_source)
    {
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
        if (!gateup_table.valid || gateup_table.num_experts <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRouting] "
                      "gate/up descriptor table is unavailable");
            return false;
        }
        const bool use_runtime_descriptors =
            descriptor_source ==
            MoEDecodeDescriptorSource::RuntimePlacementTable;
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
                    /*update_runtime_histogram=*/true,
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
            use_runtime_descriptors,
            /*allow_router_q8_reuse=*/false,
            use_runtime_descriptors ? "routing_runtime" : "routing",
            canonical_route_contributions);
    }

    bool ROCmMoEKernel::prepareGroupedRuntimeDecodeLaunchState(
        int gateup_table_id,
        int down_table_id,
        int top_k,
        int d_model,
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
    {
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

        if (descriptor_source ==
            MoEDecodeDescriptorSource::RuntimePlacementTable)
        {
            DeviceNativeVNNIMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *runtime_down_descs = nullptr;
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
        const auto &down_table = grouped_down_desc_tables_[down_table_id];
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
        if (!ensureGroupedGateUpCapacity(
                num_active, d_model, intermediate) ||
            !ensureGroupedDecodeCapacity(num_active, intermediate, d_model) ||
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
        // Preparation publishes metadata addresses; execution consumes them.
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
            descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable,
            /*allow_router_q8_reuse=*/true,
            descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable
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
        bool use_runtime_descriptors,
        bool allow_router_q8_reuse,
        const char *counter_source,
        ITensor *canonical_route_contributions)
    {
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
        const DeviceNativeVNNIMatrixDesc *decode_gate_descs =
            gateup_table.device_gate_descs;
        const DeviceNativeVNNIMatrixDesc *decode_up_descs =
            gateup_table.device_up_descs;
        const DeviceNativeVNNIMatrixDesc *decode_down_descs =
            down_table.device_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_gate_descs =
            gateup_table.device_floating_gate_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_up_descs =
            gateup_table.device_floating_up_descs;
        const DeviceMoEFloatingMatrixDesc *decode_floating_down_descs =
            down_table.device_floating_descs;
        if (use_runtime_descriptors)
        {
            DeviceNativeVNNIMatrixDesc *runtime_gate_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *runtime_up_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *runtime_down_descs = nullptr;
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
            const bool materialized = floating
                ? hipMoE_materialize_runtime_floating_descriptor_tables(
                      runtime_layer,
                      reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                          runtime_gate_descs),
                      reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                          runtime_up_descs),
                      reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                          runtime_down_descs),
                      gateup_table.num_experts,
                      gateup_table.weight_format,
                      device_ordinal_,
                      stream)
                : hipMoE_materialize_runtime_prefill_descriptor_tables(
                      runtime_layer,
                      runtime_gate_descs,
                      runtime_up_descs,
                      runtime_down_descs,
                      gateup_table.num_experts,
                      device_ordinal_,
                      stream);
            if (!materialized)
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDecodeFromRuntime] "
                          "failed to materialize compact runtime descriptor tables");
                return false;
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
         * NativeVNNI decode has no consumer for the intermediate FP32 gate/up
         * rows.  Publish the exact decode-equivalent Q8 SwiGLU rows directly
         * from the ordered partial reducer.  Floating-point weights retain
         * their own fixed-tree FP32 contract and therefore do not enter this
         * NativeVNNI publication path.
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
            : rocmMoE_grouped_gate_up_native_vnni_decode_table(
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
                  stream);
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
         * The NativeVNNI branch consumes the Q8 rows published immediately
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
            : rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
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
                  stream);
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

        if (!ensureGroupedDecodeCapacity(num_active, intermediate, d_model))
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
            : rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
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
        const bool ok = rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
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

        const bool ok = rocmMoE_grouped_swiglu_down_native_vnni_decode_runtime(
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
        const DeviceNativeVNNIMatrixDesc *down_descs,
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

        const uint8_t codebook_id = down_descs[0].codebook_id;
        for (int i = 0; i < num_active; ++i)
        {
            if (expert_ids[i] < 0 || !down_descs[i].valid() ||
                down_descs[i].n != d_model || down_descs[i].k != intermediate ||
                down_descs[i].blocks_per_row != static_cast<uint32_t>(intermediate / 32) ||
                down_descs[i].codebook_id != codebook_id ||
                (groupedDecodeRequiresMins(down_descs[i].codebook_id) && !down_descs[i].mins) ||
                (groupedDecodeRequiresEmins(down_descs[i].codebook_id) && !down_descs[i].emins))
            {
                return false;
            }
        }

        if (!groupedDecodeSupportsCodebook(codebook_id))
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Unsupported native-VNNI codebook "
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
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeVNNIMatrixDesc),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] H2D staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_vnni_decode(
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
            getStream());
    }

    bool ROCmMoEKernel::regroupPrefillRoutesForDiagnostics(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        bool retain_routes_for_deferred_commit)
    {
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
        bool retain_routes_for_deferred_commit)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(getStream()));
        if (!runtime_layer || !routing_indices || !routing_weights ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
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

        DeviceNativeVNNIMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *runtime_down_descs = nullptr;
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
                   retain_routes_for_deferred_commit ? 1 : 0,
                   gateup_table.weight_format,
                   device_ordinal_,
                   getStream());
    }

    bool ROCmMoEKernel::publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        int gateup_desc_table_id,
        int down_desc_table_id,
        bool retain_routes_for_deferred_commit)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer ||
            current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0 ||
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

        DeviceNativeVNNIMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *runtime_down_descs = nullptr;
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
                   retain_routes_for_deferred_commit ? 1 : 0,
                   gateup_table.weight_format,
                   device_ordinal_,
                   getStream());
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
            num_experts <= 256 &&
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

            if (PerfStatsCollector::isEnabled())
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

        if (PerfStatsCollector::isEnabled())
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

        return true;
    }

    bool ROCmMoEKernel::prepareExpertGroupsAsyncUsingPublishedMask(
        ITensor *routing_indices,
        ITensor *routing_weights,
        int seq_len,
        int num_experts,
        int top_k)
    {
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

    bool ROCmMoEKernel::prepareSharedExpertPrefillGroup(int seq_len)
    {
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
                seq_len,
                device_ordinal_,
                getStream()))
        {
            return false;
        }

        group_active_expert_slots_ = 1;
        if (PerfStatsCollector::isEnabled())
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
        const int max_blocks = max_dim / 32;

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
                                 static_cast<size_t>(total_slots) * (intermediate / 32) * sizeof(float),
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

    bool ROCmMoEKernel::executeGroupedPrefillPipeline(
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        ITensor *canonical_route_contributions)
    {
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "executeGroupedPrefillPipeline"))
            return false;

        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] invalid gateup descriptor table id "
                      << gateup_desc_table_id);
            return false;
        }
        if (down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] invalid down descriptor table id "
                      << down_desc_table_id);
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);

        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            down_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] descriptor tables not valid");
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
        if (!requireTensorOnDevice(
                hidden,
                device,
                stream,
                "hidden",
                "executeGroupedPrefillPipeline") ||
            !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name,
                "executeGroupedPrefillPipeline"))
        {
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_output = canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;

        if (isGraphCaptureActive() &&
            (!d_hidden ||
             (!d_output && !d_canonical_route_contributions)))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] null device pointers during graph capture");
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
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] "
                          "floating grouped-prefill route publication is incomplete");
                return false;
            }
            const bool ok = rocmMoE_grouped_floating_prefill_pipeline(
                d_hidden,
                gateup_table.device_floating_gate_descs,
                gateup_table.device_floating_up_descs,
                down_table.device_floating_descs,
                d_group_original_to_grouped_,
                d_original_expert_ids_for_pipeline,
                d_group_weights_,
                d_prefill_gate_,
                d_prefill_up_,
                d_output,
                d_canonical_route_contributions,
                seq_len,
                total_slots,
                top_k,
                d_model,
                intermediate,
                num_experts,
                gateup_table.weight_format,
                device_ordinal_,
                stream);
            if (!ok)
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] "
                          "floating grouped ROCm pipeline failed");
                return false;
            }
            markDeviceWritten(publication_output, device, stream);
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

        const bool reuse_router_q8_hidden =
            canReuseRouterQ8Hidden(d_hidden, seq_len, d_model);
        if (router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::RequiredConsumer &&
            !reuse_router_q8_hidden)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] required "
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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] "
                      "batch-invariant grouped prefill requires the original route map and expert ids");
            return false;
        }
        // Launch the fixed, fully grouped pipeline without a host synchronization.
        const bool ok = rocmMoE_grouped_prefill_pipeline(
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
            d_output,
            d_canonical_route_contributions,
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
            getStream());

        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] pipeline failed for layer");
            return false;
        }

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

        markDeviceWritten(
            canonical_route_contributions
                ? canonical_route_contributions
                : output,
            DeviceId::rocm(device_ordinal_),
            getStream());
        if (PerfStatsCollector::isEnabled() && active_expert_slots > 0)
        {
            const MoEPrefillPairPolicyTags policy =
                queryMoEPrefillPairPolicyTags(
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    seq_len,
                    d_model,
                    intermediate,
                    num_experts,
                    top_k);
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
                    {"grouping", "static"}});
        }
        return true;
    }

    bool ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan(
        DeviceMoELayerRuntime *device_runtime_layer,
        const DeviceMoELayerRuntime &runtime_host_layer,
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k,
        ITensor *canonical_route_contributions)
    {
        if (!device_runtime_layer)
            return false;
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0)
        {
            return false;
        }
        if (!setMoEDevice(
                device_ordinal_,
                "executeGroupedPrefillPipelineFromPublishedRuntimePlan"))
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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] invalid runtime scratch contract"
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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] invalid descriptor table id");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        const bool floating = deviceMoEWeightFormatIsFloating(
            gateup_table.weight_format);
        if (!gateup_table.valid || !gateup_table.deviceReady() ||
            !down_table.valid || !down_table.deviceReady() ||
            gateup_table.weight_format != down_table.weight_format ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            down_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] descriptor table shape mismatch");
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
                                     "executeGroupedPrefillPipelineFromPublishedRuntimePlan(group_original_to_grouped)"))
            {
                d_group_original_to_grouped_ = nullptr;
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] runtime grouping workspace is required");
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
        DeviceNativeVNNIMatrixDesc *runtime_gate_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *runtime_up_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *runtime_down_descs = nullptr;
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
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] "
                      "failed to bind graph-owned runtime descriptor slots");
            return false;
        }

        ITensor *publication_output = canonical_route_contributions
                                          ? canonical_route_contributions
                                          : output;
        const char *publication_output_name = canonical_route_contributions
                                                  ? "canonical_route_contributions"
                                                  : "output";
        if (!requireTensorOnDevice(
                hidden,
                device,
                stream,
                "hidden",
                "executeGroupedPrefillPipelineFromPublishedRuntimePlan") ||
            !requireOutputOnDevice(
                publication_output,
                device,
                stream,
                publication_output_name,
                "executeGroupedPrefillPipelineFromPublishedRuntimePlan"))
        {
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_output = canonical_route_contributions
                              ? nullptr
                              : static_cast<float *>(output->gpu_data_ptr());
        float *d_canonical_route_contributions =
            canonical_route_contributions
                ? static_cast<float *>(
                      canonical_route_contributions->gpu_data_ptr())
                : nullptr;
        if (!d_hidden ||
            (!d_output && !d_canonical_route_contributions))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] null device pointers");
            return false;
        }

        group_active_expert_slots_ = active_expert_slots;

        if (floating)
        {
            const bool ok = rocmMoE_grouped_floating_prefill_pipeline(
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
                d_prefill_up_,
                d_output,
                d_canonical_route_contributions,
                seq_len,
                total_slots,
                top_k,
                d_model,
                intermediate,
                num_experts,
                gateup_table.weight_format,
                device_ordinal_,
                stream);
            if (!ok)
            {
                LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] "
                          "floating grouped ROCm pipeline failed");
                return false;
            }
            markDeviceWritten(publication_output, device, stream);
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_floating_calls",
                1.0,
                "moe",
                device.to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"active_expert_slots", std::to_string(active_expert_slots)},
                 {"descriptor_source", "runtime_table"},
                 {"weight_format", std::to_string(static_cast<uint32_t>(
                                       gateup_table.weight_format))}});
            return true;
        }

        const bool reuse_router_q8_hidden =
            canReuseRouterQ8Hidden(d_hidden, seq_len, d_model);
        const bool ok = rocmMoE_grouped_prefill_pipeline(
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
            d_output,
            d_canonical_route_contributions,
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
            getStream());
        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipelineFromPublishedRuntimePlan] grouped ROCm pipeline failed");
            return false;
        }

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
                 {"descriptor_source", "runtime_table"}});
        }

        markDeviceWritten(
            canonical_route_contributions
                ? canonical_route_contributions
                : output,
            DeviceId::rocm(device_ordinal_),
            getStream());
        if (PerfStatsCollector::isEnabled() && active_expert_slots > 0)
        {
            const MoEPrefillPairPolicyTags policy =
                queryMoEPrefillPairPolicyTags(
                    gateup_table.codebook_mask,
                    down_table.codebook_mask,
                    seq_len,
                    d_model,
                    intermediate,
                    num_experts,
                    top_k);
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

} // namespace llaminar2
