/**
 * @file ROCmMoEKernel.h
 * @brief ROCm/HIP implementation of MoE kernel operations
 *
 * Implements IMoEKernel for ROCm GPU execution using HIP kernels.
 * Follows the three-file pattern: .h (class), .cpp (bridge), .hip (kernels).
 *
 * Operations:
 * - Router: gate logits (GEMV) → softmax → top-k selection
 * - Token gather: parallel row copy to expert batch buffer
 * - Scatter-add: weighted accumulation of expert outputs
 * - Shared expert gate: sigmoid dot + elementwise scale
 * - SwiGLU: silu(gate) * up activation
 *
 * All operations are dispatched on the kernel's bound HIP stream.
 */

#pragma once

#include "../../IMoEKernel.h"
#include "../../common/DeviceResidentRouterGateCache.h"
#include "../ROCmKernelBase.h"
#include "../../../tensors/TensorType.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2
{
    class PersistentWorkspaceSlotLease;

    namespace rocm
    {
        class HipBLASGemmKernel;
    }

    /**
     * @brief ROCm GPU implementation of MoE kernel operations
     *
     * Uses HIP kernels for device-native MoE operations. All methods
     * expect device pointers and dispatch work on the bound GPU stream.
     *
     * Constructor requires a device ordinal. The kernel obtains its
     * HIP stream from GPUDeviceContextPool.
     */
    class ROCmMoEKernel : public IMoEKernel, public ROCmKernelBase
    {
    public:
        /**
         * @brief Construct a ROCm MoE kernel for the given device
         * @param device_ordinal ROCm GPU ordinal (0-based)
         */
        explicit ROCmMoEKernel(int device_ordinal);
        ~ROCmMoEKernel() override;

        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override { bindWorkspace(nullptr); }

        /**
         * @brief Reset request-shaped host metadata while preserving device buffers.
         *
         * Session resets clear histogram counts and CPU-side grouping metadata.
         * Device scratch, runtime pointer arrays, descriptor tables, and router
         * weight conversion caches stay resident because captured prefill graphs
         * can reference those stable device addresses across requests.
         */
        void resetDynamicState() override;

        // =================================================================
        // IMoEKernel interface
        // =================================================================

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

        // =================================================================
        // Tensor-aware API overrides (GPU implementations)
        // =================================================================

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
            const int *device_effective_seq_len = nullptr) override;

        /**
         * @brief Bind HIP router scratch and immutable gate caches before capture.
         *
         * The selected route kind determines whether Q8 hidden rows, k-part
         * partials, or a persistent Q8/FP16 gate publication is required.  No
         * routing output is produced by this setup operation.
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

        bool groupedExpertDownDecode(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            const int *expert_ids,
            const float *expert_weights,
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_active,
            ITensor *output,
            int d_model,
            int intermediate) override;

        int uploadGroupedExpertDownDescriptorTable(
            const DeviceNativeVNNIMatrixDesc *down_descs,
            int num_experts,
            int d_model,
            int intermediate) override;

        int uploadGroupedExpertGateUpDescriptorTables(
            const DeviceNativeVNNIMatrixDesc *gate_descs,
            const DeviceNativeVNNIMatrixDesc *up_descs,
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

        bool groupedExpertGateUpDecodeFromTable(
            const TensorBase *input,
            const int *expert_ids,
            int descriptor_table_id,
            int num_active,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate) override;

        bool groupedExpertGateUpDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr) override;

        bool groupedExpertGateUpDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int descriptor_table_id,
            int top_k,
            ITensor *const *gate_outputs,
            ITensor *const *up_outputs,
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

        /**
         * @brief Fused runtime-table single-token expert decode.
         *
         * ROCm mirrors CUDA's graph-capturable runtime path: route ids and
         * weights are read from the device-resident runtime table, temporary
         * gate/up activations live in declared MoE workspace buffers, and no
         * host routing or tensor scratch ownership is involved on replay.
         */
        bool groupedExpertDecodeFromRuntime(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            MoEDecodeDescriptorSource descriptor_source =
                MoEDecodeDescriptorSource::RuntimePlacementTable,
            ITensor *canonical_route_contributions = nullptr) override;

        /**
         * @brief Execute fused workspace-native decode from device routing tensors.
         *
         * Explicit verifier routing and runtime-table routing share the same
         * ordered gate/up and down implementation after route normalization.
         */
        bool groupedExpertDecodeFromRouting(
            const TensorBase *input,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr,
            ITensor *canonical_route_contributions = nullptr) override;

        bool reduceCanonicalRouteContributions(
            ITensor *canonical_route_contributions,
            ITensor *output,
            int seq_len,
            int top_k,
            int d_model) override;

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

        bool groupedExpertDownDecodeFromRouting(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            ITensor *routing_indices,
            ITensor *routing_weights,
            int descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate,
            const uint8_t *expert_mask = nullptr) override;

        bool groupedExpertDownDecodeFromRuntime(
            ITensor *const *gate_tensors,
            ITensor *const *up_tensors,
            DeviceMoELayerRuntime *runtime_layer,
            int descriptor_table_id,
            int top_k,
            ITensor *output,
            int d_model,
            int intermediate) override;

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
            uint32_t command_buffer_count = 1) override;

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
            uint32_t local_transfer_slot_count = 0) override;

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
            uint32_t command_buffer_count = 1) override;

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

        // =================================================================
        // Phase 3: Device-side token grouping (prefill optimization)
        // =================================================================

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

        bool regroupPrefillRoutesFromRuntimeAssignments(
            DeviceMoELayerRuntime *runtime_layer,
            int current_tokens, int max_tokens,
            int num_experts, int top_k,
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

        // =================================================================
        // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
        // =================================================================

        bool prepareExpertGroupsAsync(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        bool prepareExpertGroupsAsyncUsingPublishedMask(
            ITensor *routing_indices, ITensor *routing_weights,
            int seq_len, int num_experts, int top_k) override;

        bool updateGroupedPrefillExpertMask(
            const uint8_t *expert_mask,
            int num_experts) override;

        bool prepareSharedExpertPrefillGroup(int seq_len) override;

        bool executeGroupedPrefillPipeline(
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k,
            ITensor *canonical_route_contributions = nullptr) override;

        bool executeGroupedPrefillPipelineFromRuntime(
            DeviceMoELayerRuntime *device_runtime_layer,
            const DeviceMoELayerRuntime &runtime_host_layer,
            ITensor *hidden, ITensor *output,
            int gateup_desc_table_id,
            int down_desc_table_id,
            int seq_len, int d_model, int intermediate,
            int num_experts, int top_k,
            ITensor *canonical_route_contributions = nullptr) override;

        bool hasGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate) const
        {
            return total_slots <= prefill_slots_cap_ &&
                   d_model <= prefill_d_model_cap_ &&
                   intermediate <= prefill_intermediate_cap_;
        }

        bool hasGroupingBufferCapacity(int total_slots, int num_experts) const
        {
            return total_slots <= group_slots_cap_ &&
                   num_experts <= group_experts_cap_ &&
                   d_write_heads_ != nullptr &&
                   num_experts <= max_write_heads_experts_;
        }

        // =================================================================
        // ITensorKernel interface
        // =================================================================

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // GPU only
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }

        /// Propagate a validated stream binding to the child hipBLAS kernel.
        void bindGPUStream(ExplicitGPUStream stream) override
        {
            ROCmKernelBase::bindGPUStream(stream);
            syncBlasStream();
        }

        /// Explicitly end both parent and child borrowed-stream lifetimes.
        void clearGPUStreamBinding() override;

    private:
        /**
         * @brief Shared fused decode implementation after route metadata is resolved.
         *
         * @param runtime_layer Device runtime placement table when runtime
         *        descriptors are selected; null for static descriptor tables.
         * @param device_expert_ids Device-owned normalized expert ids.
         * @param device_weights Device-owned route weights.
         * @param use_runtime_descriptors Select mutable placement descriptors
         *        instead of immutable descriptor tables.
         * @param allow_router_q8_reuse True only when this route source owns
         *        the matching router quantization publication.
         * @param counter_source Stable perfstats source label.
         */
        bool groupedExpertDecodeResolved(
            DeviceMoELayerRuntime *runtime_layer,
            const TensorBase *input,
            int gateup_descriptor_table_id,
            int down_descriptor_table_id,
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
         * The persistent descriptor lease identifies graph ownership across all
         * MoE kernel objects sharing one device workspace. Table decode,
         * two-step runtime decode, and fused runtime decode can target different
         * scratch/output buffers for the same descriptor. HIP graphs capture
         * the pointer-array device address, so ROCm uses the same globally
         * leased, scoped-slot contract as CUDA.
         */
        enum class RuntimePointerArrayScope : std::size_t
        {
            TableDecode = 0,
            RuntimeTwoStep = 1,
            RuntimeFused = 2,
        };

        void syncBlasStream();
        bool ensureStagingCapacity(int count);
        bool ensureGroupedDecodeCapacity(int num_active, int intermediate);
        bool ensureGroupedGateUpCapacity(int num_active, int d_model);
        bool ensureGroupedGateUpKPartScratchCapacity(int num_active, int k_partitions, int intermediate);
        bool ensureGroupedGateUpDecodeMetadata(const int *expert_ids, int num_active);
        bool ensureGroupedDownDecodeMetadata(const int *expert_ids, const float *expert_weights, int num_active);
        bool isDecodeGraphCaptureActive() const;
        bool rejectDecodeStagingDuringCapture(const char *context) const;
        bool stageRuntimeGateUpPointerArrays(
            std::size_t persistent_descriptor_slot,
            RuntimePointerArrayScope scope,
            int top_k,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &gate_ptrs,
            const std::array<float *, kRuntimePointerArrayMaxTopK> &up_ptrs,
            float ***d_gate_ptrs,
            float ***d_up_ptrs);
        bool stageRuntimeDownPointerArrays(
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
            std::size_t *workspace_slot,
            const char *context) const;
        bool ensureSharedGateScratchCapacity(int seq_len);
        bool ensureRouteBufferCapacity(size_t logits_count);
        bool ensureRouteLogitsPartialsCapacity(size_t partial_count);
        /**
         * @brief Bind row-major Q8 router publication scratch for a prefill bucket.
         *
         * The same graph-stable workspace is shared by serial decode, grouped
         * MTP verification, and ordinary prefill.  Sizing by the declared row
         * bucket lets production prefill use the batch-invariant grouped router
         * without a verifier-only four-row ceiling or a hot-path allocation.
         */
        bool ensureRouterQ8HiddenScratchCapacity(int rows, int d_model);
        /**
         * @brief Invalidate the router-to-expert Q8 hidden publication.
         *
         * The ROCm MoE kernel is shared by router and expert stages.  Any route
         * that does not produce Q8 hidden rows must revoke the previous
         * publication before an expert stage can inspect it; matching only the
         * arena pointer is insufficient because graph buffers are deliberately
         * reused across layers and decode steps.
         */
        void invalidateRouterQ8HiddenPublication() noexcept;

        /**
         * @brief Publish Q8 hidden rows produced by the immediately preceding router.
         *
         * @param source FP32 device row base that was quantized by the router.
         * @param rows Number of contiguous source rows in the Q8 scratch.
         * @param recorded_during_capture True when the producer launch was
         *        recorded into the currently active HIP graph rather than run
         *        eagerly.
         */
        void publishRouterQ8Hidden(
            const float *source,
            int rows,
            bool recorded_during_capture) noexcept;

        /**
         * @brief Test whether grouped gate/up may consume the router Q8 rows.
         *
         * A capture-recorded publication is valid only while that same capture
         * is active.  Once capture ends, its kernels have not executed yet and
         * host-side eager code must not mistake old scratch bytes for the
         * recorded producer's output.
         */
        bool canReuseRouterQ8Hidden(
            const float *source,
            int rows,
            int d_model) const noexcept;
        bool bindWorkspaceBuffer(void **ptr, const char *name, size_t bytes, const char *context);

        /**
         * @brief Resolve one fixed-stride grouped descriptor-table slot.
         *
         * The graph stores routed tables with one descriptor per model expert
         * beside singleton shared-expert tables.  Slot addresses therefore
         * cannot use the current table's width: doing so places a one-expert
         * table inside a previously published routed table.  This helper
         * derives the maximum-expert stride from the graph-owned workspace and
         * validates the requested table width before returning its device
         * address.
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

        void clearWorkspaceScratchBindings() noexcept;
        bool rebindGroupedDescriptorTablesToWorkspace(const char *context);
        struct RouterQ8GateCacheEntry;
        const RouterQ8GateCacheEntry *getOrCreateQ8RouterGateCache(
            const float *gate_device_ptr,
            int d_model,
            int num_experts);
        const void *getOrCreateFP16RouterGateCache(
            const float *gate_device_ptr,
            int d_model,
            int num_experts);

        /**
         * @brief Produce router logits and publish final FP32 route tensors.
         *
         * The intermediate logits remain in graph-stable workspace. The final
         * expert IDs and weights are written directly into caller-owned tensors
         * by the production decode-equivalent softmax/top-k kernel. This makes
         * the producer/output relationship explicit and excludes post-router
         * conversion kernels and device-to-device copies from captured replay.
         *
         * @param hidden Device FP32 hidden rows.
         * @param gate_weights Device gate matrix in @p gate_type.
         * @param gate_type Native gate tensor type.
         * @param seq_len Number of physical rows in the launch.
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

        /**
         * @brief Workspace-owned grouped descriptor bytes and readiness edge.
         *
         * Every graph-local HIP kernel adopts this exact publication on its
         * explicit stream. Descriptor capacity therefore scales with unique
         * prepared weights rather than prefill-bucket or MTP-depth graph count.
         */
        struct GroupedDescriptorWorkspacePublication
        {
            void *ready_event = nullptr;
            DeviceNativeVNNIMatrixDesc *primary_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *secondary_descs = nullptr;
            std::size_t workspace_slot = 0;
        };

        struct GroupedDownDescriptorTable
        {
            DeviceNativeVNNIMatrixDesc *device_descs = nullptr;
            std::shared_ptr<GroupedDescriptorWorkspacePublication>
                workspace_publication;
            std::vector<DeviceNativeVNNIMatrixDesc> host_descs;
            int num_experts = 0;
            int d_model = 0;
            int intermediate = 0;
            uint8_t codebook_id = 0;
            uint32_t codebook_mask = 0;
            std::size_t workspace_slot = 0;
            bool valid = false;
        };

        struct GroupedGateUpDescriptorTable
        {
            DeviceNativeVNNIMatrixDesc *device_gate_descs = nullptr;
            DeviceNativeVNNIMatrixDesc *device_up_descs = nullptr;
            std::shared_ptr<GroupedDescriptorWorkspacePublication>
                workspace_publication;
            std::vector<DeviceNativeVNNIMatrixDesc> host_gate_descs;
            std::vector<DeviceNativeVNNIMatrixDesc> host_up_descs;
            int num_experts = 0;
            int d_model = 0;
            int intermediate = 0;
            uint8_t codebook_id = 0;
            uint32_t codebook_mask = 0;
            std::size_t workspace_slot = 0;
            bool valid = false;
        };

        /**
         * @brief Publish or adopt one exact down-descriptor table.
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
         * @brief Workspace-owned immutable HIP router-weight publication.
         *
         * A model gate conversion belongs to the workspace allocation rather
         * than any graph-local kernel object. The readiness event provides the
         * only setup-time ordering edge needed when another graph first adopts
         * the immutable bytes on a different explicit stream.
         */
        struct RouterGateWorkspacePublication
        {
            void *ready_event = nullptr;
            void *primary_weights = nullptr;
            void *scales = nullptr;
            std::size_t workspace_slot = 0;
        };

        struct RouterFP16GateCacheEntry
        {
            DeviceResidentRouterGateCacheKey key{};
            size_t element_count = 0;
            void *d_gate_weights_fp16 = nullptr;
            std::shared_ptr<RouterGateWorkspacePublication> workspace_publication;
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

        /**
         * @brief Warmup readiness for graph-owned grouped-decode pointer slots.
         *
         * HIP graph replay captures the device address of the pointer-array slot,
         * not the host pointer values.  ROCm therefore uses deterministic scoped
         * slots and refuses capture until warmup has staged the requested slot
         * through the declared MoE workspace.  The booleans are intentionally
         * just readiness markers; pointer values are not cached on the kernel
         * object.
         */
        std::array<bool, kRuntimePointerArrayWorkspaceEntries> gateup_pointer_slot_ready_{};
        std::array<bool, kRuntimePointerArrayWorkspaceEntries> down_pointer_slot_ready_{};

        int device_ordinal_;
        std::unique_ptr<rocm::HipBLASGemmKernel> blas_gemm_;

        // Device-resident expert mask used by asynchronous grouped routing.
        uint8_t *d_group_expert_mask_ = nullptr; ///< [max_experts_] for masked async grouping
        std::shared_ptr<PersistentWorkspaceSlotLease> group_expert_mask_workspace_lease_;
        int group_expert_mask_cap_ = 0;
        uint64_t group_expert_mask_hash_ = 0;
        int group_expert_mask_num_experts_ = 0;
        int group_expert_mask_active_experts_ = 0;
        bool group_expert_mask_published_ = false;

        // Phase 3: write_heads scratch buffer for token grouping
        int *d_write_heads_ = nullptr; ///< [max_write_heads_experts_] on device
        int max_write_heads_experts_ = 0;

        // Staging buffers for tensor-aware gather/scatter (H2D of small host arrays)
        int *d_staging_indices_ = nullptr;   ///< [staging_capacity_] ints on device
        float *d_staging_weights_ = nullptr; ///< [staging_capacity_] floats on device
        int staging_capacity_ = 0;

        // Grouped decode staging for ROCm native-VNNI MoE down path.
        const float **d_grouped_gate_ptrs_ = nullptr;
        const float **d_grouped_up_ptrs_ = nullptr;
        int *d_grouped_expert_ids_ = nullptr;
        float *d_grouped_decode_weights_ = nullptr;
        DeviceNativeVNNIMatrixDesc *d_grouped_down_descs_ = nullptr;
        int8_t *d_grouped_swiglu_int8_ = nullptr;
        float *d_grouped_swiglu_scales_ = nullptr;
        int grouped_decode_active_cap_ = 0;
        int grouped_decode_intermediate_cap_ = 0;
        std::vector<GroupedDownDescriptorTable> grouped_down_desc_tables_;
        std::vector<int> grouped_down_cached_expert_ids_;
        std::vector<float> grouped_down_cached_weights_;

        // Grouped decode staging for ROCm native-VNNI MoE gate/up path.
        float **d_grouped_gate_output_ptrs_ = nullptr;
        float **d_grouped_up_output_ptrs_ = nullptr;
        int *d_grouped_gateup_expert_ids_ = nullptr;
        int8_t *d_grouped_hidden_int8_ = nullptr;
        float *d_grouped_hidden_scales_ = nullptr;
        float *d_grouped_gateup_gate_partials_ = nullptr;
        float *d_grouped_gateup_up_partials_ = nullptr;
        int grouped_gateup_active_cap_ = 0;
        int grouped_gateup_d_model_cap_ = 0;
        int grouped_gateup_kpart_active_cap_ = 0;
        int grouped_gateup_kpart_partitions_cap_ = 0;
        int grouped_gateup_kpart_intermediate_cap_ = 0;
        std::vector<GroupedGateUpDescriptorTable> grouped_gateup_desc_tables_;
        std::vector<float *> host_grouped_gate_output_ptrs_;
        std::vector<float *> host_grouped_up_output_ptrs_;
        std::vector<int> host_grouped_gateup_expert_ids_;
        std::vector<int> grouped_gateup_cached_expert_ids_;

        // Reusable scratch for sharedExpertGate() gate values.
        float *d_shared_gate_scratch_ = nullptr; ///< [shared_gate_scratch_capacity_] floats on device
        int shared_gate_scratch_capacity_ = 0;

        // Reusable private logits scratch for routeCore().
        float *d_route_logits_ = nullptr;            ///< [route_logits_capacity_] floats on device
        float *d_route_logits_partials_ = nullptr;   ///< [route_logits_partials_capacity_] floats on device
        int8_t *d_router_q8_hidden_ = nullptr;       ///< [router_q8_hidden_rows_cap_, router_q8_hidden_d_model_cap_] device rows
        float *d_router_q8_hidden_scales_ = nullptr; ///< [router_q8_hidden_rows_cap_, router_q8_hidden_blocks_cap_] device scales
        const float *router_q8_hidden_source_ = nullptr; ///< FP32 row base that produced the published Q8 rows
        int router_q8_hidden_rows_ = 0; ///< Number of contiguous valid rows in the publication
        bool router_q8_hidden_valid_ = false; ///< True only after a Q8 router producer has been issued
        bool router_q8_hidden_capture_recorded_ = false; ///< Producer exists only inside the current graph capture
        size_t route_logits_capacity_ = 0;
        size_t route_logits_partials_capacity_ = 0;
        int router_q8_hidden_rows_cap_ = 0;
        int router_q8_hidden_d_model_cap_ = 0;
        int router_q8_hidden_blocks_cap_ = 0;
        std::vector<RouterFP16GateCacheEntry> router_fp16_gate_cache_;
        std::vector<RouterQ8GateCacheEntry> router_q8_gate_cache_;

        // Phase 4: GPU-side expert grouping state (for prepareExpertGroups)
        int *d_group_int_indices_ = nullptr;   ///< float→int converted routing indices
        int *d_group_offsets_ = nullptr;       ///< [num_experts] exclusive prefix sums
        int *d_group_counts_ = nullptr;        ///< [num_experts] per-expert token counts
        int *d_group_max_tokens_ = nullptr;    ///< [1] device-side max(d_group_counts_) (async reduction)
        int *d_group_token_indices_ = nullptr; ///< [total_slots] grouped token indices
        int *d_group_original_to_grouped_ = nullptr; ///< [total_slots] original route slot to grouped slot
        float *d_group_weights_ = nullptr;     ///< [total_slots] grouped routing weights
        int *d_group_active_expert_ids_ = nullptr; ///< [min(total_slots,num_experts)] compact active expert ids
        int group_active_expert_slots_ = 0;    ///< Fixed active-expert launch slots from the last small grouping pass
        int group_slots_cap_ = 0;              ///< capacity for total_slots buffers
        int group_experts_cap_ = 0;            ///< capacity for num_experts buffers

        // Phase 5: Grouped prefill pipeline scratch buffers.
        //
        // The fused ROCm verifier path must not write SwiGLU activations into
        // the same buffers that hold gathered hidden activations: independent
        // output-column blocks can finish the fused gate/up epilogue while
        // other blocks are still reading A/scales.  Keep the handoff explicit
        // and workspace-owned so graph replay cannot observe an in-place race.
        int8_t *d_prefill_A_int8_ = nullptr;       ///< [prefill_slots_cap_, max(d_model,intermediate)]
        float *d_prefill_A_scales_ = nullptr;      ///< [prefill_slots_cap_, max_blocks_per_row]
        int8_t *d_prefill_swiglu_int8_ = nullptr;  ///< [prefill_slots_cap_, intermediate]
        float *d_prefill_swiglu_scales_ = nullptr; ///< [prefill_slots_cap_, intermediate / 32]
        float *d_prefill_gate_ = nullptr;          ///< [prefill_slots_cap_, max(d_model,intermediate)]
        float *d_prefill_up_ = nullptr;            ///< [prefill_slots_cap_, intermediate]
        int prefill_slots_cap_ = 0;           ///< Current capacity (total_slots)
        int prefill_d_model_cap_ = 0;         ///< Current d_model capacity
        int prefill_intermediate_cap_ = 0;    ///< Current intermediate capacity

        bool scratch_workspace_bound_ = false;
        uint64_t bound_workspace_id_ = 0;

        bool ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate);
    };

} // namespace llaminar2
