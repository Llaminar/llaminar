/**
 * @file MoERoutingStage.h
 * @brief MoE routing stage: softmax top-k expert selection
 *
 * Extracted from MoEExpertComputeStage to enable independent routing computation.
 * Outputs raw routing results (expert indices as float, normalized weights)
 * without any expert-ID ownership masking; MoEExpertComputeStage handles that.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../moe/DecodeExpertHistogram.h"
#include "../../moe/MoERuntimeTable.h"

#include <memory>
#include <string>
#include <vector>
namespace llaminar2
{

    /**
     * @brief MoE routing stage: compute expert selection via softmax top-k
     *
     * Calls IMoEKernel::route() to compute routing, then writes results
     * to output tensors as FP32. It does not apply expert-ID ownership masking;
     * the responsibility of MoEExpertComputeStage.
     *
     * Outputs:
     * - output_indices: FP32 [seq_len * top_k] expert IDs cast to float
     * - output_weights: FP32 [seq_len * top_k] normalized routing weights
     */
    class MoERoutingStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input
            TensorBase *input = nullptr; ///< Normalized hidden [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            // Router config
            TensorBase *gate_weights = nullptr; ///< Router gate [num_experts, d_model]
            int num_experts = 0;
            int top_k = 0;
            bool norm_topk_prob = true;

            // Layer info for histogram
            int layer_idx = -1;
            DecodeExpertHistogram *decode_histogram = nullptr;
            IMoERuntimeTable *moe_runtime_table = nullptr;
            /**
             * @brief Graph-lowered ownership of each selected routed row.
             *
             * This must match the paired MoEExpertComputeStage. A physically
             * replicated placement can either distribute rows for LLEP economy
             * or execute every row independently for a mirrored MTP sidecar;
             * routing owns that distinction because it publishes the runtime
             * expert IDs consumed by the expert kernel.
             */
            RoutedExpertRowExecutionPolicy routed_row_execution_policy =
                RoutedExpertRowExecutionPolicy::ParticipantAssigned;
            bool force_grouped_verifier_prefill_for_decode = false;
            /**
             * @brief Device-owned absolute position row shared with RoPE.
             *
             * Runtime decode uses this stable semantic coordinate when
             * partitioning replicated experts. The pointer is graph-local and
             * remains stable across capture replay while its contents advance
             * on the graph's publication stream.
             */
            const int32_t *absolute_position_ids_device = nullptr;
            /**
             * @brief Explicit graph-local owner shared with the routed expert stage.
             *
             * The paired stages are sequential and exchange router-produced Q8
             * hidden rows through this backend object. A distinct owner must be
             * created for every separately captured main or MTP graph.
             */
            std::shared_ptr<MoERoutedPipelineKernelOwner> routed_pipeline_kernel_owner;

            // Optional graph-capturable decode rebalance apply piggyback.
            // When enabled, the route kernel applies any ready device-side
            // rebalance wave before publishing this token's top-k/local masks.
            bool device_rebalance_route_apply = false;
            std::string device_rebalance_workspace_name;
            DeviceMoERebalanceConfig device_rebalance_config;
            DeviceMoEExpertDirectoryEntry *device_rebalance_local_transfer_slots = nullptr;
            uint32_t device_rebalance_local_transfer_slot_count = 0;
            uint32_t device_rebalance_plan_capacity = 0;
            uint32_t device_rebalance_command_buffer_count = 1;
            int device_rebalance_apply_layer_idx = -2;

            /**
             * @brief Select grouped serial-decode-equivalent verifier routing.
             *
             * MTP all-position verifier batches produce several candidate rows
             * at once.  GPU backends execute those rows as one economical
             * grouped launch whose per-row K traversal and reduction order are
             * byte-equivalent to M=1 decode.  This flag never permits production
             * row replay; it selects the dedicated grouped verifier contract.
             */
            bool force_decode_equivalent_verifier_prefill = false;

            /**
             * @brief Device-owned logical row count for every variable-row GPU graph.
             *
             * Bucketed prefill and grouped verification both launch a stable
             * physical M while only a leading prefix belongs to the current
             * request. The graph captures this model-lifetime address, whose
             * value is published by request admission or verifier preparation
             * on the graph's producer stream. Backends invalidate every suffix
             * route directly from this scalar, so no host replay parameter or
             * scalar upload can influence captured topology.
             */
            const int32_t *active_row_count_device = nullptr;

            // Outputs (written by this stage)
            TensorBase *output_indices = nullptr; ///< FP32 [seq_len * top_k] expert IDs as float
            TensorBase *output_weights = nullptr; ///< FP32 [seq_len * top_k] normalized weights
            // Buffer IDs for coherence
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
            BufferId output_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
        };

        explicit MoERoutingStage(Params params);
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        std::string name() const override { return "moe_router"; }
        size_t estimatedFlops() const override;

        int layerIndex() const { return params_.layer_idx; }

        bool allowsZeroOutput() const override { return false; }
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /**
         * @brief Describe every routing graph-capture admission predicate.
         *
         * A router is often the first stage to expose an incomplete
         * device-resident MoE lifecycle. Include both cold-preflight and warm
         * readiness facts so a mandatory full-graph failure identifies the
         * missing contract directly instead of merely reporting MOE_ROUTER.
         */
        std::string graphCaptureReadinessDebugString() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return params_.device_id.is_gpu() &&
                           supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        /**
         * @brief Clear per-request routing metadata while preserving kernel handles.
         */
        void resetSessionState() override;
        /**
         * @brief Clear routing diagnostics while preserving captured MoE state.
         *
         * Variable row counts are owned by the request-geometry allocation and
         * are not stage-local replay metadata. This hook therefore preserves
         * only immutable launch state and backend descriptor identity.
         */
        void resetSessionStatePreservingCapturedReplay() override;
        /**
         * @brief Preserve warmed routing workspace for capture-from-Initialized.
         */
        void resetSessionStatePreservingLazyInitialization() override;
        /**
         * @brief Reset backend launch metadata owned exclusively by this stage.
         *
         * Routing stages retain independent MoE kernel instances so a sibling
         * expert or maintenance graph cannot retarget their stream, workspace,
         * or descriptor storage. A hard graph invalidation resets that private
         * backend state; captured-replay request boundaries deliberately do not
         * call this method.
         */
        void invalidateKernelDynamicState() override;

        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            params_.routed_pipeline_kernel_owner.reset();
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }

        /**
         * @brief Expose the graph-lowered row ownership policy to graph tests.
         *
         * Routing publishes the runtime expert IDs consumed by the paired
         * expert stage.  A graph-construction regression must therefore prove
         * both stages received the same typed policy rather than inspecting
         * only the downstream stage and assuming the router agreed.
         */
        RoutedExpertRowExecutionPolicy
        routedExpertRowExecutionPolicyForTesting() const noexcept
        {
            return params_.routed_row_execution_policy;
        }

        /** @brief Expose the captured device row-count owner to graph tests. */
        const int32_t *activeRowCountDeviceForTesting() const noexcept
        {
            return params_.active_row_count_device;
        }

    private:
        Params params_;

        /// Stashed routing results for snapshot capture
        mutable std::vector<float> routing_indices_f32_;
        mutable std::vector<float> routing_weights_;
        mutable std::vector<float> router_logits_;

        /**
         * @brief Graph-local MoE kernel and optional non-owning test override.
         *
         * A router may share this owner only with its paired routed-expert
         * consumer so the captured producer/consumer chain can reuse the same
         * Q8 hidden publication. Separately captured main, verifier, and
         * maintenance graphs always own distinct backend objects.
         */
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

        /// Pre-allocated routing result (avoids heap allocs per decode token)
        mutable MoERoutingResult cached_routing_;
        DeviceMoELayerRuntime *moe_runtime_layer_ = nullptr;

        IMoEKernel *ensureMoEKernel() const;
        /** @brief Classify the production arithmetic route prepared for capture. */
        MoERouteLaunchKind routeLaunchKind() const noexcept;
        bool isDeviceRoutedDecodeGraphCapturable() const;
        bool isDeviceRoutedPrefillExecutionSupported() const;
        bool isDeviceRoutedPrefillGraphCaptureSupported() const;
        bool isDeviceRoutedPrefillGraphCapturable() const;
        bool isDecodeEquivalentVerifierPrefillExecutionSupported() const;
        bool isDecodeEquivalentVerifierPrefillGraphCaptureSupported() const;
        bool isDecodeEquivalentVerifierPrefillGraphCapturable() const;
        bool hasInitializedRuntimeTableIfProvided() const;
        bool executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);
        void recordRuntimeHistogramTokenBoundary() const;
        void stashRoutingResults(
            const std::vector<int> &expert_indices,
            const std::vector<float> &expert_weights,
            int seq_len, int top_k) const;

    };

} // namespace llaminar2
