/**
 * @file MoERuntimeDescriptorLaunch.h
 * @brief Compiler-checked CUDA/HIP descriptor-publication launch contracts.
 *
 * Definitions, production bridges and native replays include this same ABI.
 * Projection ownership is mandatory: a stale C-linkage declaration must fail
 * compilation rather than pass an unspecified policy scalar into a GPU graph.
 * Output storage remains arena-owned and each launch uses its exact stream.
 */
#pragma once
#include "DeviceNativeQuantisedMatrixDesc.h"
#include "DeviceMoEFloatingMatrixDesc.h"
#include "MoEDecodeDescriptorPublication.h"
#include "NativeTileCompletion.h"
#include "execution/config/RoutedExpertPolicy.h"

extern "C"
{
    /**
     * @brief Original128-lane FP16 router projection, retained as the exact control.
     * @param hidden Original FP32 hidden values, rounded to half at each load.
     * @param gate_weights Native FP16 source matrix with contiguous expert rows.
     * @param logits Original FP32 output. @param d_model Reduction width.
     * @param num_experts Exact expert extent. @param device_idx Rank-local ordinal.
     * @param stream Exact prepared stream. @return Native submission success.
     */
    bool hipMoE_gate_logits_single_token_fp16_weights(
        const float *hidden, const void *gate_weights, float *logits,
        int d_model, int num_experts, int device_idx, void *stream);

    /**
     * @brief Existing runtime router retaining its explicit ready-rebalance boundary.
     * @param logits Original FP32 logits. @param runtime Original acquired device runtime.
     * @param legacy_indices Complete selected IDs. @param legacy_weights Complete selected weights.
     * @param num_experts Source extent. @param top_k Selected packet width.
     * @param normalize_weights Original serial normalization policy.
     * @param write_legacy_outputs Whether both selected outputs are required.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param row_execution_policy Original owner/replica/fragment execution contract.
     * @param runtime_layers Optional existing ready-rebalance runtime domain.
     * @param rebalance_plan_entries Existing device command entries.
     * @param rebalance_plan_capacity Admitted command capacity.
     * @param rebalance_command_header Existing command publication.
     * @param rebalance_local_transfer_slots Original native transfer slots.
     * @param rebalance_local_transfer_slot_count Admitted transfer-slot count.
     * @param rebalance_config Original immutable movement configuration.
     * @param rebalance_apply_status Device-owned publication status.
     * @param rebalance_controller_state Device-owned controller state.
     * @param rebalance_target_layer Original target layer or -1 without an apply transaction.
     * @param rebalance_command_buffer_count Admitted original command-bank count.
     * @param absolute_position_ids Original device position coordinates when required.
     * @param device_idx Original device ordinal. @param stream Exact non-default stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
    bool cudaMoE_softmax_topk_decode_runtime(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        void *runtime_layers, const void *rebalance_plan_entries, uint32_t rebalance_plan_capacity,
        void *rebalance_command_header, const void *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count, const void *rebalance_config,
        void *rebalance_apply_status, void *rebalance_controller_state, int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count, const int32_t *absolute_position_ids,
        int device_idx, void *stream, const int32_t *active_row_count_device);

    /**
     * @brief Existing runtime router retaining its explicit ready-rebalance boundary.
     * @param logits Original FP32 logits. @param runtime Original acquired device runtime.
     * @param legacy_indices Complete selected IDs. @param legacy_weights Complete selected weights.
     * @param num_experts Source extent. @param top_k Selected packet width.
     * @param normalize_weights Original serial normalization policy.
     * @param write_legacy_outputs Whether both selected outputs are required.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param row_execution_policy Original owner/replica/fragment execution contract.
     * @param runtime_layers Optional existing ready-rebalance runtime domain.
     * @param rebalance_plan_entries Existing device command entries.
     * @param rebalance_plan_capacity Admitted command capacity.
     * @param rebalance_command_header Existing command publication.
     * @param rebalance_local_transfer_slots Original native transfer slots.
     * @param rebalance_local_transfer_slot_count Admitted transfer-slot count.
     * @param rebalance_config Original immutable movement configuration.
     * @param rebalance_apply_status Device-owned publication status.
     * @param rebalance_controller_state Device-owned controller state.
     * @param rebalance_target_layer Original target layer or -1 without an apply transaction.
     * @param rebalance_command_buffer_count Admitted original command-bank count.
     * @param absolute_position_ids Original device position coordinates when required.
     * @param device_idx Original device ordinal. @param stream Exact non-default stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
    bool hipMoE_softmax_topk_decode_runtime_wave64(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_execution_policy,
        void *runtime_layers, const void *rebalance_plan_entries, uint32_t rebalance_plan_capacity,
        void *rebalance_command_header, const void *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count, const void *rebalance_config,
        void *rebalance_apply_status, void *rebalance_controller_state, int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count, const int32_t *absolute_position_ids,
        int device_idx, void *stream, const int32_t *active_row_count_device);

    /**
     * @brief Existing independent whole-owner router without descriptor publication.
     * @param logits Original FP32 logits, overwritten with probabilities.
     * @param runtime Original device runtime and acquired bank.
     * @param legacy_indices Optional complete selected ID output.
     * @param legacy_weights Optional complete selected weight output.
     * @param num_experts Original source extent. @param top_k Selected packet width.
     * @param normalize_weights Preserve original serial normalization.
     * @param write_legacy_outputs Whether complete selected outputs are required.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param device_idx Original device ordinal. @param stream Exact non-default stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
    bool cudaMoE_softmax_topk_decode_unique_owner_runtime(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, int device_idx, void *stream,
        const int32_t *active_row_count_device);

    /**
     * @brief Existing independent whole-owner router without descriptor publication.
     * @param logits Original FP32 logits, overwritten with probabilities.
     * @param runtime Original device runtime and acquired bank.
     * @param legacy_indices Optional complete selected ID output.
     * @param legacy_weights Optional complete selected weight output.
     * @param num_experts Original source extent. @param top_k Selected packet width.
     * @param normalize_weights Preserve original serial normalization.
     * @param write_legacy_outputs Whether complete selected outputs are required.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param device_idx Original device ordinal. @param stream Exact non-default stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
    bool hipMoE_softmax_topk_decode_unique_owner_wave64(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, int device_idx, void *stream,
        const int32_t *active_row_count_device);

    /**
     * @brief Publish the exact decode route and its descriptors in one retained kernel.
     * @param logits Original FP32 router logits, overwritten with probabilities.
     * @param runtime Original device runtime retaining the acquired placement bank.
     * @param legacy_indices Optional complete selected ID output.
     * @param legacy_weights Optional complete selected weight output.
     * @param num_experts Exact original source extent.
     * @param top_k Exact selected packet width.
     * @param normalize_weights Preserve the backend's original serial normalization.
     * @param write_legacy_outputs Whether both complete selected outputs are bound.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param row_policy Original independent whole-owner or local-fragment contract.
     * @param publication Existing arena descriptor outputs with complete identity.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-default producer and consumer stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
    bool cudaMoE_softmax_topk_decode_selected_descriptors(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_policy,
        llaminar2::MoEDecodeDescriptorPublication publication,
        int device_idx, void *stream, const int32_t *active_row_count_device);

    /**
     * @brief Publish the exact decode route and its descriptors in one retained kernel.
     * @param logits Original FP32 router logits, overwritten with probabilities.
     * @param runtime Original device runtime retaining the acquired placement bank.
     * @param legacy_indices Optional complete selected ID output.
     * @param legacy_weights Optional complete selected weight output.
     * @param num_experts Exact original source extent.
     * @param top_k Exact selected packet width.
     * @param normalize_weights Preserve the backend's original serial normalization.
     * @param write_legacy_outputs Whether both complete selected outputs are bound.
     * @param update_runtime_histogram Whether this graph owns demand evidence.
     * @param row_policy Original independent whole-owner or local-fragment contract.
     * @param publication Existing arena descriptor outputs with complete identity.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-default producer and consumer stream.
     * @param active_row_count_device Optional device-owned live row count.
     * @return Whether validation and native submission succeeded.
     */
    bool hipMoE_softmax_topk_decode_selected_descriptors(
        float *logits, void *runtime, float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_policy,
        llaminar2::MoEDecodeDescriptorPublication publication,
        int device_idx, void *stream, const int32_t *active_row_count_device);

    /**
     * @brief Complete FP16 decode routing from original paired projection teams.
     * @param hidden Original FP32 row, rounded to half by the unchanged dot expression.
     * @param gate_weights Original contiguous FP16 expert matrix; never requantized.
     * @param logits Original public probabilities, including inactive-row writes.
     * @param runtime Original acquired placement/route authority.
     * @param legacy_indices Optional complete selected expert IDs.
     * @param legacy_weights Optional complete selected weight bits.
     * @param d_model Positive original reduction width, including odd widths.
     * @param num_experts Exact source extent, with an optional odd final team.
     * @param top_k Exact selected width bounded by the source expert extent.
     * @param normalize_weights Preserve serial slot-order FP32 normalization.
     * @param write_legacy_outputs Publish both complete selected outputs.
     * @param update_runtime_histogram Publish original integer demand evidence.
     * @param row_policy Original independent whole-owner or fragment contract.
     * @param publication Original admitted descriptor destinations and format.
     * @param completion One initialized PMA counter: tiles=routes=1 and
     *        contributors=ceil(num_experts/2). The last producer resets it.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-default stream; no allocation or synchronization.
     * @param active_row_count_device Original device-owned live count.
     * @return Whether the complete binding and native submission succeeded.
     */
    bool hipMoE_fp16_projection_route_selected_descriptors(
        const float *hidden, const void *gate_weights, float *logits, void *runtime,
        float *legacy_indices, float *legacy_weights, int d_model, int num_experts,
        int top_k, bool normalize_weights, bool write_legacy_outputs,
        bool update_runtime_histogram, llaminar2::RoutedExpertRowExecutionPolicy row_policy,
        llaminar2::MoEDecodeDescriptorPublication publication,
        llaminar2::NativeTileCompletionView completion,
        int device_idx, void *stream, const int32_t *active_row_count_device);

    /**
     * @brief Refresh the exact decode ID packet from its acquired bank.
     * @param runtime Device runtime with stable bank/epoch bindings.
     * @param expert_ids Device-selected IDs; -1 names an inactive slot.
     * @param gate_descs Stable expert-indexed gate table.
     * @param up_descs Stable expert-indexed up table.
     * @param down_descs Stable expert-indexed down table.
     * @param num_experts Admitted logical table extent.
     * @param top_k Exact packet width; never source/workspace capacity.
     * @param format Original native or floating descriptor family.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param projections Mandatory movable projection ownership contract.
     * @return Whether validation and native submission succeeded.
     */
    bool cudaMoE_materialize_runtime_selected_decode_descriptors(
        const void *runtime, const int *expert_ids,
        llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts, int top_k, llaminar2::DeviceMoEWeightFormat format,
        int device_idx, void *stream, llaminar2::DeviceMoEProjectionSet projections);

    /**
     * @brief Publish the complete native descriptor domain for grouped preparation.
     * @param runtime Device runtime retaining its active/acquired bank.
     * @param gate_descs Full admitted gate table.
     * @param up_descs Full admitted up table.
     * @param down_descs Full admitted down table.
     * @param num_experts Complete logical table extent.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-null publication stream.
     * @param projections Mandatory movable projection ownership contract.
     * @return Whether validation and native submission succeeded.
     */
    bool cudaMoE_materialize_runtime_prefill_descriptor_tables(
        const void *runtime, llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts, int device_idx, void *stream,
        llaminar2::DeviceMoEProjectionSet projections);

    /**
     * @brief Publish one complete floating descriptor domain without conversion.
     * @param runtime Device runtime retaining its active/acquired bank.
     * @param gate_descs Full admitted gate table.
     * @param up_descs Full admitted up table.
     * @param down_descs Full admitted down table.
     * @param num_experts Complete logical table extent.
     * @param format Original floating scalar family.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-null publication stream.
     * @param projections Mandatory movable projection ownership contract.
     * @return Whether validation and native submission succeeded.
     */
    bool cudaMoE_materialize_runtime_floating_descriptor_tables(
        const void *runtime, llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *down_descs, int num_experts,
        llaminar2::DeviceMoEWeightFormat format, int device_idx, void *stream,
        llaminar2::DeviceMoEProjectionSet projections);

    /**
     * @brief Refresh the exact decode ID packet from its acquired bank.
     * @param runtime Device runtime with stable bank/epoch bindings.
     * @param expert_ids Device-selected IDs; -1 names an inactive slot.
     * @param gate_descs Stable expert-indexed gate table.
     * @param up_descs Stable expert-indexed up table.
     * @param down_descs Stable expert-indexed down table.
     * @param num_experts Admitted logical table extent.
     * @param top_k Exact packet width; never source/workspace capacity.
     * @param format Original native or floating descriptor family.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-null producer/consumer stream.
     * @param projections Mandatory movable projection ownership contract.
     * @return Whether validation and native submission succeeded.
     */
    bool hipMoE_materialize_runtime_selected_decode_descriptors(
        const void *runtime, const int *expert_ids,
        llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts, int top_k, llaminar2::DeviceMoEWeightFormat format,
        int device_idx, void *stream, llaminar2::DeviceMoEProjectionSet projections);

    /**
     * @brief Publish the complete native descriptor domain for grouped preparation.
     * @param runtime Device runtime retaining its active/acquired bank.
     * @param gate_descs Full admitted gate table.
     * @param up_descs Full admitted up table.
     * @param down_descs Full admitted down table.
     * @param num_experts Complete logical table extent.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-null publication stream.
     * @param projections Mandatory movable projection ownership contract.
     * @return Whether validation and native submission succeeded.
     */
    bool hipMoE_materialize_runtime_prefill_descriptor_tables(
        const void *runtime, llaminar2::DeviceNativeQuantisedMatrixDesc *gate_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *up_descs,
        llaminar2::DeviceNativeQuantisedMatrixDesc *down_descs,
        int num_experts, int device_idx, void *stream,
        llaminar2::DeviceMoEProjectionSet projections);

    /**
     * @brief Publish one complete floating descriptor domain without conversion.
     * @param runtime Device runtime retaining its active/acquired bank.
     * @param gate_descs Full admitted gate table.
     * @param up_descs Full admitted up table.
     * @param down_descs Full admitted down table.
     * @param num_experts Complete logical table extent.
     * @param format Original floating scalar family.
     * @param device_idx Original rank-local device ordinal.
     * @param stream Exact non-null publication stream.
     * @param projections Mandatory movable projection ownership contract.
     * @return Whether validation and native submission succeeded.
     */
    bool hipMoE_materialize_runtime_floating_descriptor_tables(
        const void *runtime, llaminar2::DeviceMoEFloatingMatrixDesc *gate_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *up_descs,
        llaminar2::DeviceMoEFloatingMatrixDesc *down_descs, int num_experts,
        llaminar2::DeviceMoEWeightFormat format, int device_idx, void *stream,
        llaminar2::DeviceMoEProjectionSet projections);

} // extern "C"
