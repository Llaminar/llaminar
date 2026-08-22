#pragma once

/**
 * @file CUDAMoEKernel.h
 * @brief CUDA implementation of device-resident MoE routing and dispatch primitives.
 *
 * Provides the CUDA backend for `IMoEKernel`, keeping MoE routing results,
 * expert gather/scatter buffers, shared expert gates, and fallback SwiGLU
 * activations on-device. GEMM remains handled by the existing tensor-aware
 * CUDA GEMM engines; this class owns only the non-GEMM MoE glue kernels and
 * persistent scratch needed by those kernels.
 *
 * Lifecycle: Instances are created and cached by `KernelFactory` per CUDA
 * device. Scratch allocations are retained across calls and released when the
 * cached kernel is destroyed.
 */

#include "../../IMoEKernel.h"
#include "../../common/DeviceResidentRouterGateCache.h"
#include "../CUDAKernelBase.h"
#include "../gemm/CUDADeviceWorkspace.h"
#include "../../../tensors/TensorType.h"
#include "../../../execution/moe/MoERuntimePointerWorkspaceOwners.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class PersistentWorkspaceSlotLease;

    /**
     * @brief CUDA backend for MoE router, grouping, gather/scatter, and elementwise glue.
     *
     * The compute stages remain backend-neutral and call this through
     * `IMoEKernel`. All CUDA runtime interaction is isolated here and in the
     * companion `.cu` bridge file.
     */
    class CUDAMoEKernel final : public IMoEKernel, public CUDAKernelBase
    {
    public:
        /// @brief Construct a CUDA MoE kernel bound to one CUDA ordinal.
        explicit CUDAMoEKernel(int device_ordinal);

        /// @brief Release persistent CUDA scratch buffers.
        ~CUDAMoEKernel() override;

        /// @brief Clear request-shaped host grouping metadata while retaining device scratch.
        void resetDynamicState() override;

        /** @copydoc IMoEKernel::packMoEOverlayActivationDispatch */
        bool packMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationDispatchPackLaunch &packet) override;

        /** @copydoc IMoEKernel::consumeMoEOverlayActivationDispatch */
        bool consumeMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationDispatchConsumeLaunch &packet) override;

        /** @copydoc IMoEKernel::packMoEOverlayActivationReturn */
        bool packMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationReturnPackLaunch &packet) override;

        /** @copydoc IMoEKernel::consumeMoEOverlayActivationReturn */
        bool consumeMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationReturnConsumeLaunch &packet) override;

        /** @copydoc IMoEKernel::packSingleRowMoEOverlayActivationDispatch */
        bool packSingleRowMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowDispatchPackLaunch &packet) override;

        /** @copydoc IMoEKernel::consumeSingleRowMoEOverlayActivationDispatch */
        bool consumeSingleRowMoEOverlayActivationDispatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowDispatchConsumeLaunch &packet) override;

        /** @copydoc IMoEKernel::packSingleRowMoEOverlayActivationReturn */
        bool packSingleRowMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowReturnPackLaunch &packet) override;

        /** @copydoc IMoEKernel::consumeSingleRowMoEOverlayActivationReturn */
        bool consumeSingleRowMoEOverlayActivationReturn(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowReturnConsumeLaunch &packet) override;

        /** @copydoc IMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch */
        bool packSingleRowMoEOverlayActivationDispatchBatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowDispatchBatchLaunch &packet) override;

        /** @copydoc IMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch */
        bool consumeSingleRowMoEOverlayActivationReturnBatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationSingleRowReturnBatchLaunch &packet) override;
        /** @copydoc IMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch */
        bool consumeMultiRowMoEOverlayActivationReturnBatch(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayActivationMultiRowReturnBatchLaunch &packet) override;

        /** @copydoc IMoEKernel::publishNodeLocalCanonicalRoutes */
        bool publishNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRoutePublishLaunch &publication) override;

        /** @copydoc IMoEKernel::acquireNodeLocalCanonicalRoutes */
        bool acquireNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRouteConsumeLaunch &consumption) override;

        /** @copydoc IMoEKernel::stageNodeLocalCanonicalRoutes */
        bool stageNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRouteConsumeLaunch &consumption) override;

        /** @copydoc IMoEKernel::foldNodeLocalCanonicalRoutes */
        bool foldNodeLocalCanonicalRoutes(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalRouteConsumeLaunch &consumption) override;

        /** @copydoc IMoEKernel::beginNodeLocalDensePublication */
        bool beginNodeLocalDensePublication(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication) override;

        /** @copydoc IMoEKernel::finishNodeLocalDensePublication */
        bool finishNodeLocalDensePublication(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication) override;

        /** @copydoc IMoEKernel::beginNodeLocalDensePublicationConsume */
        bool beginNodeLocalDensePublicationConsume(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication) override;

        /** @copydoc IMoEKernel::finishNodeLocalDensePublicationConsume */
        bool finishNodeLocalDensePublicationConsume(
            const MoEKernelLaunchContext &launch,
            const MoENodeLocalDensePublicationLaunch &publication) override;

        /** @copydoc IMoEKernel::runMoEOverlayDeviceControllerAction */
        bool runMoEOverlayDeviceControllerAction(
            const MoEKernelLaunchContext &launch,
            const MoEOverlayDeviceControllerActionLaunch &action) override;

        /** @copydoc IMoEKernel::beginMoEOverlayServiceTelemetry */
        bool beginMoEOverlayServiceTelemetry(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayServiceTelemetrySample *sample) override;

        /** @copydoc IMoEKernel::finishMoEOverlayServiceTelemetry */
        bool finishMoEOverlayServiceTelemetry(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
            DeviceMoEOverlayServiceTelemetrySample *sample,
            std::uint32_t num_experts,
            MoEOverlayServicePhaseHint hint,
            const MoEOverlayInferenceGraphRole *runtime_graph_role = nullptr)
            override;

        /** @copydoc IMoEKernel::publishMoEOverlayServiceTelemetry */
        bool publishMoEOverlayServiceTelemetry(
            const MoEKernelLaunchContext &launch,
            const DeviceMoEOverlayServiceTelemetryCell *telemetry,
            const DeviceMoEOverlayServiceTelemetrySample *samples,
            std::uint32_t layer_count,
            std::int32_t participant_id,
            MoEOverlayDeviceServiceTelemetryPublicationHeader *publication)
            override;

        /** @copydoc IMoEKernel::acquireMoEOverlayEpoch */
        bool acquireMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            DeviceMoEOverlayEpochTicket *ticket,
            DeviceMoEOverlayEpochStatus *status,
            const std::uint64_t *external_admission_epoch = nullptr,
            DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier = {},
            MoEOverlayPeerPlacementEpochBinding peer_placement_epoch = {}) override;

        /** @copydoc IMoEKernel::releaseMoEOverlayEpoch */
        bool releaseMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            DeviceMoEOverlayEpochTicket *ticket,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @copydoc IMoEKernel::reserveMoEOverlayEpochCandidate */
        bool reserveMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @copydoc IMoEKernel::finalizeMoEOverlayRebalancePublication */
        bool finalizeMoEOverlayRebalancePublication(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            std::uint32_t layer_count,
            std::uint32_t expert_count,
            DeviceMoEOverlayEpochControl *control,
            std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *reservation_and_publication_status,
            const DeviceMoERebalanceApplyStatus *apply_status) override;

        /** @copydoc IMoEKernel::markMoEOverlayEpochCandidateReady */
        bool markMoEOverlayEpochCandidateReady(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @copydoc IMoEKernel::publishMoEOverlayEpochCandidate */
        bool publishMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @copydoc IMoEKernel::abortMoEOverlayEpochCandidate */
        bool abortMoEOverlayEpochCandidate(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *candidate_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        /** @copydoc IMoEKernel::retireMoEOverlayEpoch */
        bool retireMoEOverlayEpoch(
            const MoEKernelLaunchContext &launch,
            DeviceMoEOverlayEpochControl *control,
            const std::uint64_t *retiring_epoch,
            DeviceMoEOverlayEpochStatus *status) override;

        void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) override;

        void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) override;

        void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) override;

        void swiGLU(float *gate, const float *up, int count) override;

        void weightedAdd(float *output, const float *input,
                         float weight, int count) override;

        bool routeWithTensors(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result) override;

        bool routeWithTensorsEffectiveSeqLen(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result,
            const int *device_effective_seq_len) override;

        bool routeVerifierRowsDecodeEquivalent(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            const int *device_effective_seq_len = nullptr,
            DeviceMoELayerRuntime *deferred_selected_route_ledger = nullptr) override;

        /**
         * @brief Bind CUDA router scratch and immutable gate caches before capture.
         *
         * This setup-only operation may quantize an FP32 router matrix into its
         * persistent Q8 workspace slot, but it never computes or publishes a
         * route result.
         */
        bool prepareRouteLaunch(
            ITensor *gate_weights,
            const MoERouteLaunchPlan &plan) override;

        bool decodeRouteSelect(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *gate_weights,
            int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            bool write_legacy_outputs,
            bool update_runtime_histogram,
            const int32_t *absolute_position_ids_device,
            RoutedExpertRowExecutionPolicy row_execution_policy) override;

        bool decodeRouteSelectWithReadyRebalanceApply(
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
            RoutedExpertRowExecutionPolicy row_execution_policy) override;

        void zeroBuffer(ITensor *tensor, size_t bytes) override;

        void gatherTokenBatchFromTensors(
            ITensor *hidden, ITensor *batch_buffer,
            const int *host_token_indices, int num_tokens, int d_model) override;

        bool copyTokenRowFromTensor(
            ITensor *source, ITensor *row_buffer,
            int row_index, int row_width) override;

        void scatterAddWeightedFromTensors(
            ITensor *output, ITensor *expert_output,
            const int *host_token_indices, const float *host_weights,
            int num_tokens, int d_model) override;

        bool writeTokenRowToTensor(
            ITensor *destination, ITensor *row_buffer,
            int row_index, int row_width) override;

        void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model) override;

        bool sharedExpertGateFromTensorsEffectiveSeqLen(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model,
            const int *device_effective_seq_len) override;

        void sharedExpertGateAddFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            ITensor *routed_residual, ITensor *combined_output,
            int seq_len, int d_model) override;

        bool sharedExpertGateAddFromTensorsEffectiveSeqLen(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            ITensor *routed_residual, ITensor *combined_output,
            int seq_len, int d_model,
            const int *device_effective_seq_len) override;

        void swiGLUFromTensors(ITensor *gate, ITensor *up, int count) override;

        void weightedAddFromTensors(
            ITensor *output, ITensor *input, float weight, int count) override;

        bool groupTokensByExpertDevice(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            float *d_grouped_weights) override;

        bool groupPrefillRoutes(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *routing_indices, ITensor *routing_weights,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            bool filter_to_local_runtime_experts = false,
            bool retain_routes_for_deferred_commit = false) override;

        /**
         * @brief Rebuild grouped rows for focused assignment diagnostics.
         *
         * Production execution publishes descriptors, active ids, and inverse
         * mapping through publishCompleteGroupedPrefillPlanFromRuntimeAssignments().
         * This concrete-only entrypoint exists so integration tests can isolate
         * the stable grouping algorithm without manufacturing descriptor tables;
         * it is intentionally absent from IMoEKernel and execution-stage code.
         */
        bool regroupPrefillRoutesForDiagnostics(
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens,
            int max_tokens,
            int num_experts,
            int top_k,
            bool retain_routes_for_deferred_commit = false);

        bool publishCompleteGroupedPrefillPlanFromRouter(
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
            bool retain_routes_for_deferred_commit = false) override;

        bool publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens,
            int max_tokens,
            int num_experts,
            int top_k,
            int gateup_desc_table_id,
            int down_desc_table_id,
            bool retain_routes_for_deferred_commit = false) override;

        bool commitGroupedVerifierHistograms(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            const int32_t *accepted_state_counts_device,
            const int32_t *publication_ok_flags_device,
            int request_count,
            int rows_per_request,
            int total_rows,
            int num_experts,
            int top_k) override;

        bool assignPrefillRoutesLeastLoadedResident(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            const int32_t *absolute_position_ids_device,
            const int32_t *active_row_count_device) override;

        bool planPrefillRoutesLeastLoadedCurrentBatch(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config) override;

        bool assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k) override;

        bool assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
            const DeviceMoERebalanceStatus *transfer_status,
            const DeviceMoERebalanceApplyStatus *apply_status) override;

        bool gatherPrefillExpertBatchFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            ITensor *hidden, ITensor *batch_buffer,
            int expert_id, int max_tokens, int d_model) override;

        bool scatterPrefillExpertResultsFromRuntime(
            ITensor *output, ITensor *expert_results,
            DeviceMoELayerRuntime *runtime_layer,
            int expert_id, int max_tokens, int d_model) override;

        /// @brief Upload persistent down-projection descriptors for grouped CUDA prefill.
        int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable) override;

        /// @brief Upload persistent gate/up descriptor tables for grouped CUDA prefill.
        int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable) override;

        /** @copydoc IMoEKernel::uploadGroupedExpertFloatingDownDescriptorTable */
        int uploadGroupedExpertFloatingDownDescriptorTable(
            const DeviceMoEFloatingMatrixDesc *down_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate) override;

        /** @copydoc IMoEKernel::uploadGroupedExpertFloatingGateUpDescriptorTables */
        int uploadGroupedExpertFloatingGateUpDescriptorTables(
            const DeviceMoEFloatingMatrixDesc *gate_descs,
            const DeviceMoEFloatingMatrixDesc *up_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate) override;

        bool updateGroupedExpertDownDescriptorTable(
            int descriptor_table_id,
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        bool updateGroupedExpertGateUpDescriptorTables(
            int descriptor_table_id,
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        /** @copydoc IMoEKernel::updateGroupedExpertFloatingDownDescriptorTable */
        bool updateGroupedExpertFloatingDownDescriptorTable(
            int descriptor_table_id,
            const DeviceMoEFloatingMatrixDesc *down_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate) override;

        /** @copydoc IMoEKernel::updateGroupedExpertFloatingGateUpDescriptorTables */
        bool updateGroupedExpertFloatingGateUpDescriptorTables(
            int descriptor_table_id,
            const DeviceMoEFloatingMatrixDesc *gate_descs,
            const DeviceMoEFloatingMatrixDesc *up_descs,
            DeviceMoEWeightFormat weight_format,
            int num_experts,
            int d_model,
            int intermediate) override;

        /// @brief Prepare device-only expert grouping metadata for graph-captured prefill.
        bool prepareExpertGroupsAsync(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        bool prepareExpertGroupsAsyncUsingPublishedMask(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        bool updateGroupedPrefillExpertMask(
            const uint8_t *expert_mask,
            int num_experts) override;

        /// @brief Prepare grouped prefill metadata for the always-active shared expert.
        bool prepareSharedExpertPrefillGroup(int seq_len) override;

        /** @copydoc IMoEKernel::bindRouterQ8HiddenPublication */
        bool bindRouterQ8HiddenPublication(
            std::shared_ptr<MoERouterQ8HiddenPublication> publication,
            MoERouterQ8PublicationAccess access) override;

        /// @brief Execute fixed-topology grouped MoE prefill without host synchronization.
        bool executeGroupedPrefillPipeline(
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k,
            ITensor *canonical_route_contributions = nullptr) override;

        bool executeGroupedPrefillPipelineFromPublishedRuntimePlan(
            DeviceMoELayerRuntime *device_runtime_layer,
            const DeviceMoELayerRuntime &runtime_host_layer,
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k,
            ITensor *canonical_route_contributions = nullptr) override;

        /// @brief Execute grouped gate/up decode from a persistent descriptor table and static host ids.
        bool groupedExpertGateUpDecodeFromTable(
            const TensorBase *input,
            const int *expert_ids,
            int descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        /// @brief Execute grouped SwiGLU/down decode from a persistent descriptor table and static host ids/weights.
        bool groupedExpertDownDecodeFromTable(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            const int *expert_ids,
            const float *expert_weights,
            int descriptor_table_id,
            int num_active,
            ITensor *output,
            int d_model,
            int intermediate) override;

        /// @brief Execute grouped gate/up decode from FP32 routing indices already resident on CUDA.
        bool groupedExpertGateUpDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            int table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr) override;

        /// @brief Execute graph-capturable grouped gate/up decode from runtime-table top-k ids.
        bool groupedExpertGateUpDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        /// @brief Execute grouped SwiGLU/down decode from FP32 routing indices and weights on CUDA.
        bool groupedExpertDownDecodeFromRouting(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr) override;

        /// @brief Execute fused workspace-native decode from device routing tensors.
        bool groupedExpertDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int gateup_table_id,
            int down_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr,
            ITensor *canonical_route_contributions = nullptr,
            DeviceMoELayerRuntime *runtime_layer = nullptr,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable) override;

        /// @brief Execute graph-capturable grouped SwiGLU/down decode from runtime-table ids and weights.
        bool groupedExpertDownDecodeFromRuntime(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            DeviceMoELayerRuntime *runtime_layer,
            int table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate) override;

        /** @copydoc IMoEKernel::prepareGroupedRuntimeDecodeLaunchState */
        bool prepareGroupedRuntimeDecodeLaunchState(
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source) override;

        /** @copydoc IMoEKernel::prepareGroupedTableDecodeLaunchState */
        bool prepareGroupedTableDecodeLaunchState(
            const int *expert_ids,
            const float *expert_weights,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            ITensor *output,
            int d_model,
            int intermediate) override;

        /// @brief Execute fused graph-capturable runtime-routed gate/up, SwiGLU, and down decode.
        bool groupedExpertDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_table_id,
            int down_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::RuntimePlacementTable,
            ITensor *canonical_route_contributions = nullptr) override;

        bool reduceCanonicalRouteContributions(
            ITensor *canonical_route_contributions,
            ITensor *output,
            int seq_len,
            int top_k,
            int d_model) override;

        bool publishSharedExpertRankBank(
            ITensor *shared_output,
            ITensor *canonical_publication,
            int seq_len,
            int top_k,
            int d_model,
            int participant_index,
            int participant_count,
            const int *device_effective_seq_len = nullptr) override;

        bool finalizeCanonicalMoEPublication(
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
            const int *device_effective_seq_len = nullptr) override;

        bool runDeviceRebalanceController(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            const uint64_t *gathered_histograms,
            DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalancePlanEntry *plan_entries = nullptr,
            uint32_t *plan_count = nullptr,
            uint32_t plan_capacity = 0,
            uint32_t payload_slot_capacity = 0,
            DeviceMoERebalanceCommandBufferHeader *command_header = nullptr,
            DeviceMoERebalanceWaveState *wave_state = nullptr,
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1,
            const DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr,
            uint32_t local_transfer_slot_count = 0,
            DeviceMoELLEPLayerPlanScratch *llep_layer_plans = nullptr) override;

        bool packDeviceRebalanceHistograms(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            uint64_t *local_histograms,
            const DeviceMoERebalanceConfig &config,
            const DeviceMoERebalanceWaveState *wave_state = nullptr,
            const DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1,
            uint32_t histogram_source_mask =
                moe_runtime_abi::kAllHistogramSourcesMask,
            uint64_t *previous_activation_counts = nullptr) override;

        bool packDeviceRebalanceDirectory(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            DeviceMoEExpertDirectoryEntry *local_directory,
            const DeviceMoERebalanceConfig &config) override;

        bool packDeviceRebalanceSourceDescriptors(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            const DeviceMoERebalancePlanEntry *plan_entries,
            const DeviceMoERebalanceCommandBufferHeader *command_headers,
            uint32_t plan_capacity,
            DeviceMoEExpertDirectoryEntry *local_source_descriptors,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1) override;

        bool projectDeviceRebalanceDomainCommands(
            const MoEKernelLaunchContext &launch,
            const DeviceMoERebalancePlanEntry *gathered_plan_entries,
            const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
            uint32_t plan_capacity,
            DeviceMoERebalancePlanEntry *local_plan_entries,
            DeviceMoERebalanceCommandBufferHeader *local_command_headers,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalanceStatus *status = nullptr,
            uint32_t payload_slot_capacity = 0,
            uint32_t command_buffer_count = 1,
            const DeviceMoERebalanceWaveState *gathered_wave_states = nullptr,
            DeviceMoERebalanceWaveState *local_wave_states = nullptr,
            DeviceMoELayerRuntime *runtime_layers = nullptr,
            const DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr,
            uint32_t local_transfer_slot_count = 0,
            DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index = nullptr) override;

        bool projectPrefillLeastLoadedDomainCommands(
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
            uint32_t command_buffer_count = 1) override;

        bool materializePrefillLeastLoadedTransferCommands(
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
            uint32_t command_buffer_count = 1) override;

        bool materializePrefillLeastLoadedMirroredDomainCommands(
            const MoEKernelLaunchContext &launch,
            const DeviceMoELayerRuntime *runtime_layer,
            DeviceMoERebalancePlanEntry *mirrored_plan_entries,
            DeviceMoERebalanceCommandBufferHeader *mirrored_command_headers,
            uint32_t plan_capacity,
            DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceConfig &config,
            uint32_t payload_slot_capacity,
            uint32_t layer_idx) override;

        bool packDeviceRebalanceCompactPayloads(
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
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1) override;

        bool packDeviceRebalanceCollectivePayloads(
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
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1) override;

        bool unpackDeviceRebalanceCollectivePayloads(
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
            DeviceMoERebalanceGraphControllerState *controller_state = nullptr,
            uint32_t command_buffer_count = 1) override;

        bool initializeDeviceRebalanceGraphController(
            const MoEKernelLaunchContext &launch,
            DeviceMoERebalanceGraphControllerState *controller_state,
            const DeviceMoERebalanceConfig &config) override;

        bool resetDeviceRebalanceGraphTransactionForRequest(
            const MoEKernelLaunchContext &launch,
            DeviceMoERebalanceGraphControllerState *controller_state,
            DeviceMoERebalanceCommandBufferHeader *command_headers,
            DeviceMoERebalanceWaveState *wave_states,
            uint32_t *plan_counts,
            uint32_t command_buffer_count,
            const DeviceMoERebalanceConfig &config) override;

        bool publishDeviceRebalanceTransferComplete(
            const MoEKernelLaunchContext &launch,
            DeviceMoERebalanceGraphControllerState *controller_state,
            const DeviceMoERebalanceCommandBufferHeader *command_header,
            const DeviceMoERebalanceWaveState *wave_state,
            const DeviceMoERebalanceApplyStatus *copy_status,
            const DeviceMoERebalancePlanEntry *plan_entries,
            uint32_t plan_capacity,
            const DeviceMoERebalanceApplyStatus *gathered_copy_status,
            const DeviceMoERebalanceConfig &config,
            uint32_t command_buffer_count = 1) override;

        bool applyReadyDeviceRebalanceWave(
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
            DeviceMoERebalanceCommandBufferHeader *command_header = nullptr,
            int target_layer = -1,
            uint32_t command_buffer_count = 1,
            const DeviceMoEOverlayEpochStatus *overlay_reservation_status = nullptr) override;

        bool applyDeviceRebalanceArrivals(
            const MoEKernelLaunchContext &launch,
            DeviceMoELayerRuntime *runtime_layers,
            const DeviceMoERebalancePlanEntry *plan_entries,
            const uint32_t *plan_count,
            uint32_t plan_capacity,
            const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
            uint32_t local_transfer_slot_count,
            const DeviceMoERebalanceConfig &config,
            DeviceMoERebalanceApplyStatus *status,
            const DeviceMoERebalanceCommandBufferHeader *command_header = nullptr,
            int target_layer = -1) override;

        bool supports_device(int device_idx) const override
        {
            return device_idx == device_ordinal_ || device_idx >= 0;
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }

        /// @brief Forward the validated stage stream into the CUDA base class.
        void bindGPUStream(ExplicitGPUStream stream) override
        {
            CUDAKernelBase::bindGPUStream(stream);
        }

        /// @brief Explicitly end this kernel's borrowed stream lifetime.
        void clearGPUStreamBinding() override
        {
            CUDAKernelBase::clearGPUStreamBinding();
        }

        /// @brief Bind graph-owned workspace scratch used by routing/grouped MoE kernels.
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        void unbindWorkspace() override { bindWorkspace(nullptr); }

    private:
        /**
         * @brief Shared fused decode implementation after route metadata is resolved.
         *
         * Both runtime-table routing and explicit routing tensors terminate
         * here.  Keeping one implementation makes their arithmetic plan,
         * persistent pointer tables, and output publication structurally
         * identical.
         */
        bool groupedExpertDecodeResolved(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_table_id,
            int down_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            const int *device_expert_ids,
            const float *device_weights,
            bool use_runtime_descriptors,
            bool allow_router_q8_reuse,
            const char *counter_source,
            ITensor *canonical_route_contributions);

        static constexpr std::size_t kRuntimePointerArrayMaxTopK = 16;
        static constexpr std::size_t kRuntimePointerArrayTableSlots = 1024;
        static constexpr std::size_t kRuntimePointerArrayWorkspaceScopes = 3;
        static constexpr std::size_t kRuntimePointerArrayWorkspaceEntries =
            kRuntimePointerArrayTableSlots * kRuntimePointerArrayWorkspaceScopes;

        /**
         * @brief Stable graph-capture slot bands for grouped decode pointer arrays.
         *
         * Table-driven decode, two-step runtime decode, and fused runtime decode
         * can share immutable descriptors while writing different scratch and
         * output tensors. CUDA graphs capture the pointer-array device address,
         * so scope participates in each graph-local owner's exclusive lease key.
         */
        enum class RuntimePointerArrayScope : std::size_t
        {
            TableDecode = 0,
            RuntimeTwoStep = 1,
            RuntimeFused = 2,
        };

        DeviceId deviceId() const { return DeviceId::cuda(device_ordinal_); }

        /**
         * @brief Produce router logits and publish final FP32 route tensors.
         *
         * Logits use graph-stable workspace scratch because they are private to
         * the two-stage router. Expert IDs and weights are final graph outputs,
         * so softmax/top-k writes the caller-owned tensors directly. Keeping
         * those outputs in the contract prevents a second conversion kernel or
         * device-to-device publication copy from becoming part of replay.
         *
         * @param hidden Device FP32 hidden rows.
         * @param gate_weights Device gate matrix in @p gate_type.
         * @param gate_type Native gate tensor type.
         * @param seq_len Number of physical rows in the captured launch.
         * @param d_model Hidden width.
         * @param num_experts Router output width.
         * @param top_k Number of routes published per row.
         * @param normalize_weights Whether selected weights are renormalized.
         * @param output_indices Caller-owned FP32 expert IDs.
         * @param output_weights Caller-owned FP32 selected weights.
         * @param device_effective_seq_len Optional device scalar masking padded rows.
         * @return True when both router kernels were enqueued successfully.
         */
        bool routeCore(const float *hidden, const void *gate_weights, TensorType gate_type,
                       int seq_len, int d_model, int num_experts, int top_k,
                       bool normalize_weights,
                       float *output_indices, float *output_weights,
                       const int *device_effective_seq_len = nullptr);
        bool routeWithTensorsImpl(
            ITensor *hidden, ITensor *gate_weights,
            int seq_len, int d_model, int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices, ITensor *output_weights,
            MoERoutingResult &host_result,
            const int *device_effective_seq_len,
            const char *context);
        bool ensureStagingCapacity(int count);
        bool ensureRouteBufferCapacity(size_t logits_count);
        bool ensureGroupingBufferCapacity(int total_slots, int num_experts);
        bool ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate);
        /**
         * @brief Bind the graph-owned compact IMMA work directory.
         *
         * The directory is populated from device route counts on every replay.
         * Binding only resolves its persistent arena address and validates that
         * the captured bucket's worst-case tile distribution fits.
         */
        bool ensureGroupedImmaDirectoryCapacity(
            int total_slots,
            int num_experts);
        /**
         * @brief Bind decode/Q8-publication scratch for the declared row count.
         *
         * `hidden_rows` is independent of `top_k`: grouped verifier routing
         * publishes one quantized hidden row per verifier row, while ordinary
         * expert decode still needs at most `top_k` expert slots.
         */
        bool ensureGroupedGateUpDecodeCapacity(
            int top_k,
            int d_model,
            int hidden_rows = 1);
        /**
         * @brief Bind reusable split-K gate/up partial storage for active routes.
         *
         * Grouped verifier prefill passes at most one fixed row tile's route
         * slots here. Decode passes its ordinary top-k width. The distinction
         * keeps workspace bounded independently of runtime verifier depth.
         */
        bool ensureGroupedGateUpKPartScratchCapacity(
            int active_slots,
            int k_partitions,
            int intermediate);
        /**
         * @brief Bind reusable split-K down partial storage for output rows.
         *
         * `slots` is the number of rows in the current reusable verifier tile,
         * or one for ordinary serial decode.
         */
        bool ensureGroupedDownKPartScratchCapacity(int k_partitions, int d_model, int slots = 1);
        bool ensureGroupedDownDecodeCapacity(int top_k, int intermediate);
        /// @brief Ensure a routing-only expert-id buffer that cannot race host-table decode metadata.
        bool ensureRoutingDecodeMetadataCapacity(int num_active);
        /**
         * @brief Publish or resolve immutable gate/up expert ids for one graph owner.
         *
         * Warmup acquires an exclusive persistent slot associated with the
         * pointer-table owner and publishes the host ids on the exact CUDA
         * stream. Capture performs lookup only and therefore cannot allocate or
         * stage host data in the hot path.
         */
        bool resolveFixedTableGateUpMetadata(
            std::size_t persistent_descriptor_slot,
            RuntimePointerArrayScope scope,
            const int *expert_ids,
            int num_active,
            const int **device_expert_ids);
        /**
         * @brief Publish or resolve immutable down expert ids and weights.
         *
         * The returned buffers share the down pointer-table owner's exclusive
         * physical slot. A different identity for an existing owner is a hard
         * error; changing routes must use the device-routed decode interface.
         */
        bool resolveFixedTableDownMetadata(
            std::size_t persistent_descriptor_slot,
            RuntimePointerArrayScope scope,
            const int *expert_ids,
            const float *expert_weights,
            int num_active,
            const int **device_expert_ids,
            const float **device_expert_weights);
        bool groupTokensByExpertDeviceMapped(
            const int *d_routing_indices,
            const float *d_routing_weights,
            int seq_len, int num_experts, int top_k,
            int *d_expert_offsets,
            int *d_expert_counts,
            int *d_grouped_token_indices,
            int *d_original_to_grouped,
            int *d_original_expert_ids,
            float *d_grouped_weights);
        bool ensureRuntimeGateUpPointerArrays(
            std::size_t persistent_descriptor_slot,
            RuntimePointerArrayScope scope,
            int top_k,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            float ***d_gate_ptrs,
            float ***d_up_ptrs);
        bool ensureRuntimeDownPointerArrays(
            std::size_t persistent_descriptor_slot,
            RuntimePointerArrayScope scope,
            int top_k,
            const std::array<const float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<const float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            const float ***d_gate_ptrs,
            const float ***d_up_ptrs);
        bool runtimePointerWorkspaceSlot(
            std::size_t persistent_descriptor_slot,
            RuntimePointerArrayScope scope,
            MoERuntimePointerArrayRole role,
            MoERuntimePointerWorkspaceAccess access,
            std::size_t *workspace_slot,
            const char *context);
        bool bindWorkspaceBuffer(void **ptr, const char *name, size_t bytes, const char *context);

        /**
         * @brief Resolve one fixed-stride grouped descriptor-table slot.
         *
         * Routed MoE tables and singleton shared-expert tables coexist in the
         * same graph-owned descriptor arena.  Every slot must consequently use
         * the workspace's maximum-expert stride; multiplying a slot by the
         * current table width lets narrow shared tables overwrite routed
         * descriptors.  The helper validates the table width and returns the
         * isolated device range owned by the requested slot.
         *
         * @param buffer_name Workspace buffer containing one descriptor role.
         * @param slot Stable descriptor-table slot selected by the registry.
         * @param num_experts Number of live descriptors in this table.
         * @param device_descs Receives the beginning of the isolated slot.
         * @param context Diagnostic operation name used on validation failure.
         * @return True when the complete table fits in an isolated slot.
         */
        bool bindGroupedDescriptorTableSlot(
            const char *buffer_name,
            std::size_t slot,
            int num_experts,
            DeviceNativeVNNIMatrixDesc **device_descs,
            const char *context);

        bool rebindGroupedDescriptorTablesToWorkspace(const char *context);
        void clearWorkspaceScratchBindings() noexcept;
        void releaseDeviceBuffers() noexcept;
        struct RouterQ8GateCacheEntry;
        const RouterQ8GateCacheEntry *getOrCreateQ8RouterGateCache(
            const float *gate_device_ptr,
            int d_model,
            int num_experts);
        bool tryRouteDecodeLogitsQ8(
            const float *d_hidden,
            const float *d_gate,
            int d_model,
            int num_experts,
            int top_k,
            const char *context);
        /**
         * @brief Revoke the device router-to-expert Q8 hidden publication.
         *
         * CUDA graph arenas reuse the same FP32 addresses across layers and
         * tokens, so pointer identity alone cannot prove that decode scratch
         * belongs to the current router invocation.
         */
        void invalidateRouterQ8HiddenPublication() noexcept;

        /**
         * @brief Publish contiguous Q8 rows produced by the current router.
         *
         * @param source FP32 device row base quantized by the router.
         * @param rows Number of valid rows in decode hidden scratch.
         * @param recorded_during_capture True when the producer was recorded,
         *        but not yet executed, by the active CUDA graph capture.
         */
        void publishRouterQ8Hidden(
            const float *source,
            int rows,
            bool recorded_during_capture) noexcept;

        /**
         * @brief Return whether the following expert stage may consume the publication.
         *
         * Capture-recorded rows are consumable only by a consumer recorded in
         * that same active capture.  Eager code must wait for a fresh eager
         * router producer instead of observing scratch from an older launch.
         */
        bool canReuseRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model) const noexcept;

        /**
         * @brief Explain why a router Q8 hidden publication cannot be consumed.
         *
         * @return A stable perf-diagnostic reason string, or nullptr when the
         *         publication is eligible for the requested grouped expert pass.
         */
        const char *routerQ8HiddenReuseBlockReason(
            const float *source,
            int rows,
            int d_model) const noexcept;

        /**
         * @brief Workspace-owned grouped descriptor bytes and readiness edge.
         *
         * Graph-local routed-pipeline kernels retain this shared publication.
         * Captured arguments therefore keep a stable address while separately
         * materialized MTP depths and prefill buckets adopt one exact prepared
         * weight table rather than consuming private descriptor slots.
         */
        struct GroupedDescriptorWorkspacePublication
        {
            void *ready_event = nullptr;
            void *primary_descs = nullptr;
            void *secondary_descs = nullptr;
            std::size_t workspace_slot = 0;
        };

        struct GroupedDownDescriptorTable
        {
            DeviceNativeVNNIMatrixDesc *device_descs = nullptr;
            DeviceMoEFloatingMatrixDesc *device_floating_descs = nullptr;
            std::shared_ptr<GroupedDescriptorWorkspacePublication>
                workspace_publication;
            std::vector<DeviceNativeVNNIMatrixDesc> host_descs;
            std::vector<DeviceMoEFloatingMatrixDesc> host_floating_descs;
            std::size_t workspace_slot = 0;
            int num_experts = 0;
            int d_model = 0;
            int intermediate = 0;
            uint8_t codebook_id = 0;
            uint32_t codebook_mask = 0;
            uint32_t policy_codebook_mask = 0;
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable;
            DeviceMoEWeightFormat weight_format = DeviceMoEWeightFormat::NativeVNNI;
            bool valid = false;

            /** @return Whether the selected descriptor family has device storage. */
            [[nodiscard]] bool deviceReady() const noexcept
            {
                return weight_format == DeviceMoEWeightFormat::NativeVNNI
                           ? device_descs != nullptr
                           : deviceMoEWeightFormatIsFloating(weight_format) &&
                                 device_floating_descs != nullptr;
            }
        };

        struct GroupedGateUpDescriptorTable
        {
            DeviceNativeVNNIMatrixDesc *device_gate_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *device_up_descs = nullptr;
            DeviceMoEFloatingMatrixDesc *device_floating_gate_descs = nullptr;
            DeviceMoEFloatingMatrixDesc *device_floating_up_descs = nullptr;
            std::shared_ptr<GroupedDescriptorWorkspacePublication>
                workspace_publication;
            std::vector<DeviceNativeVNNIMatrixDesc> host_gate_descs;
            std::vector<DeviceNativeVNNIMatrixDesc> host_up_descs;
            std::vector<DeviceMoEFloatingMatrixDesc> host_floating_gate_descs;
            std::vector<DeviceMoEFloatingMatrixDesc> host_floating_up_descs;
            std::size_t workspace_slot = 0;
            int num_experts = 0;
            int d_model = 0;
            int intermediate = 0;
            uint8_t codebook_id = 0;
            uint32_t codebook_mask = 0;
            uint32_t policy_codebook_mask = 0;
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable;
            DeviceMoEWeightFormat weight_format = DeviceMoEWeightFormat::NativeVNNI;
            bool valid = false;

            /** @return Whether both selected descriptor tables are resident. */
            [[nodiscard]] bool deviceReady() const noexcept
            {
                return weight_format == DeviceMoEWeightFormat::NativeVNNI
                           ? device_gate_descs != nullptr && device_up_descs != nullptr
                           : deviceMoEWeightFormatIsFloating(weight_format) &&
                                 device_floating_gate_descs != nullptr &&
                                 device_floating_up_descs != nullptr;
            }
        };

        /**
         * @brief Publish or adopt one exact down-descriptor table.
         *
         * The one-time H2D producer records a readiness event. Every graph-local
         * adopter waits on that event through its explicit stream, with no host
         * synchronization and no inference-hot-path allocation.
         */
        bool publishGroupedDownDescriptorTable(
            GroupedDownDescriptorTable &table,
            const char *context);

        /**
         * @brief Publish or adopt one exact paired gate/up descriptor table.
         */
        bool publishGroupedGateUpDescriptorTable(
            GroupedGateUpDescriptorTable &table,
            const char *context);

        /**
         * @brief Workspace-owned immutable CUDA router-weight publication.
         *
         * One instance is shared by every graph-local MoE kernel that consumes
         * the same source weight in a workspace. The readiness event orders the
         * one-time quantization producer before a kernel first adopts the
         * publication on its explicit stream. It is never consulted during
         * captured replay.
         */
        struct RouterGateWorkspacePublication
        {
            void *ready_event = nullptr;
            void *primary_weights = nullptr;
            void *scales = nullptr;
            std::size_t workspace_slot = 0;
        };

        struct RouterQ8GateCacheEntry
        {
            DeviceResidentRouterGateCacheKey key{};
            int blocks_per_row = 0;
            size_t element_count = 0;
            size_t scale_count = 0;
            int8_t *d_gate_weights_q8 = nullptr;
            float *d_gate_scales = nullptr;
            std::shared_ptr<RouterGateWorkspacePublication> workspace_publication;
            std::size_t workspace_slot = 0;
        };

        int device_ordinal_ = 0;
        int *d_staging_indices_ = nullptr;
        float *d_staging_weights_ = nullptr;
        int staging_capacity_ = 0;

        float *d_route_logits_ = nullptr;
        size_t route_logits_capacity_ = 0;
        bool route_logits_workspace_bound_ = false;
        uint64_t bound_workspace_id_ = 0;

        int *d_group_int_indices_ = nullptr;
        int *d_group_offsets_ = nullptr;
        int *d_group_counts_ = nullptr;
        int *d_group_token_indices_ = nullptr;
        int *d_group_original_to_grouped_ = nullptr;
        int *d_group_original_expert_ids_ = nullptr;
        int *d_group_write_heads_ = nullptr;
        float *d_group_weights_ = nullptr;
        int *d_group_active_expert_ids_ = nullptr;
        uint8_t *d_group_expert_mask_ = nullptr;
        std::shared_ptr<PersistentWorkspaceSlotLease> group_expert_mask_workspace_lease_;
        int group_active_expert_slots_ = 0;
        int group_slots_cap_ = 0;
        int group_experts_cap_ = 0;
        bool group_buffers_workspace_bound_ = false;
        int group_expert_mask_cap_ = 0;
        uint64_t group_expert_mask_hash_ = 0;
        int group_expert_mask_num_experts_ = 0;
        int group_expert_mask_active_experts_ = 0;
        bool group_expert_mask_published_ = false;

        std::vector<GroupedDownDescriptorTable> grouped_down_desc_tables_;
        std::vector<GroupedGateUpDescriptorTable> grouped_gateup_desc_tables_;
        /**
         * @brief Warmup readiness for graph-owned grouped-decode pointer slots.
         *
         * CUDA graph replay captures the device address of the pointer-array
         * slot, not the host pointer values used to stage it. Each graph-local
         * kernel therefore owns an exclusive role-and-scope lease even when its
         * immutable weight descriptors are shared with another graph. Warmup
         * stages the leased slot and capture consumes only proven leases.
         */
        std::array<bool, kRuntimePointerArrayWorkspaceEntries> gateup_pointer_slot_ready_{};
        std::array<bool, kRuntimePointerArrayWorkspaceEntries> down_pointer_slot_ready_{};
        /** Exclusive mutable pointer slots, independent of shared weight descriptors. */
        MoERuntimePointerWorkspaceOwners runtime_pointer_workspace_owners_;

        /** Host identity retained for one immutable fixed-table gate/up slot. */
        struct FixedGateUpMetadataState
        {
            std::array<int, kRuntimePointerArrayMaxTopK> expert_ids{};
            int num_active = 0;
        };
        /** Host identity retained for one immutable fixed-table down slot. */
        struct FixedDownMetadataState
        {
            std::array<int, kRuntimePointerArrayMaxTopK> expert_ids{};
            std::array<float, kRuntimePointerArrayMaxTopK> expert_weights{};
            int num_active = 0;
        };
        std::unordered_map<std::size_t, FixedGateUpMetadataState>
            fixed_gateup_metadata_states_;
        std::unordered_map<std::size_t, FixedDownMetadataState>
            fixed_down_metadata_states_;

        int8_t *d_prefill_A_int8_ = nullptr;
        float *d_prefill_A_scales_ = nullptr;
        int8_t *d_prefill_swiglu_int8_ = nullptr;
        float *d_prefill_swiglu_scales_ = nullptr;
        float *d_prefill_gate_ = nullptr;
        float *d_prefill_up_ = nullptr;
        uint32_t *d_prefill_imma_directory_ = nullptr;
        int prefill_imma_directory_entries_cap_ = 0;
        int prefill_slots_cap_ = 0;
        int prefill_d_model_cap_ = 0;
        int prefill_intermediate_cap_ = 0;

        int8_t *d_decode_hidden_int8_ = nullptr;
        float *d_decode_hidden_scales_ = nullptr;
        /** Capture-time publication metadata; payload remains entirely on device. */
        std::shared_ptr<MoERouterQ8HiddenPublication>
            router_q8_hidden_publication_ =
                std::make_shared<MoERouterQ8HiddenPublication>();
        /** Typed authority prevents a sibling consumer from clearing producer state. */
        MoERouterQ8PublicationAccess router_q8_publication_access_ =
            MoERouterQ8PublicationAccess::ProducerAndConsumer;
        int decode_gateup_topk_cap_ = 0;
        int decode_gateup_d_model_cap_ = 0;
        int decode_hidden_rows_cap_ = 0;
        std::vector<RouterQ8GateCacheEntry> router_q8_gate_cache_;

        // Split-K partials scratch for the grouped gate/up decode projection.
        // Layout: [top_k][k_partitions][intermediate] per buffer.
        float *d_grouped_gateup_gate_partials_ = nullptr;
        float *d_grouped_gateup_up_partials_ = nullptr;
        int grouped_gateup_kpart_active_cap_ = 0;
        int grouped_gateup_kpart_partitions_cap_ = 0;
        int grouped_gateup_kpart_intermediate_cap_ = 0;

        int8_t *d_decode_swiglu_int8_ = nullptr;
        float *d_decode_swiglu_scales_ = nullptr;
        int decode_down_topk_cap_ = 0;
        int decode_down_intermediate_cap_ = 0;

        int *d_routing_decode_expert_ids_ = nullptr;
        int routing_decode_metadata_cap_ = 0;

        // Split-K partials scratch for the grouped SwiGLU down projection.
        // Decode layout: [1][k_partitions][d_model].
        // Verifier prefill layout: [total_slots][k_partitions][d_model].
        float *d_grouped_down_partials_ = nullptr;
        int grouped_down_kpart_partitions_cap_ = 0;
        int grouped_down_kpart_d_model_cap_ = 0;
        int grouped_down_kpart_slots_cap_ = 0;

        bool scratch_workspace_bound_ = false;
    };

} // namespace llaminar2
