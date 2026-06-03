/**
 * @file MoERoutingStage.h
 * @brief MoE routing stage: softmax top-k expert selection
 *
 * Extracted from MoEExpertComputeStage to enable independent routing computation.
 * Outputs raw routing results (expert indices as float, normalized weights)
 * without any EP masking — downstream MoEExpertComputeStage handles that.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../moe/DecodeExpertHistogram.h"
#include "../../moe/MoERuntimeTable.h"

#include <vector>
namespace llaminar2
{

    /**
     * @brief MoE routing stage: compute expert selection via softmax top-k
     *
     * Calls IMoEKernel::route() to compute routing, then writes results
     * to output tensors as FP32. Does NOT apply EP masking — that is
     * the responsibility of MoEExpertComputeStage.
     *
     * Outputs:
     * - output_indices: FP32 [seq_len * top_k] expert IDs cast to float
     * - output_weights: FP32 [seq_len * top_k] normalized routing weights
     */
    class MoERoutingStage : public IComputeStage
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
        bool requiresPostWarmupGraphSegmentRebuild() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        void onGraphReplayed() override;
        bool needsOnGraphReplayed() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Clear per-request routing metadata while preserving kernel handles.
         */
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            routing_indices_f32_.clear();
            routing_weights_.clear();
            router_logits_.clear();
            cached_routing_ = MoERoutingResult{};
        }

        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel) { moe_kernel_ = kernel; }

    private:
        Params params_;

        /// Stashed routing results for snapshot capture
        mutable std::vector<float> routing_indices_f32_;
        mutable std::vector<float> routing_weights_;
        mutable std::vector<float> router_logits_;

        /// Cached MoE kernel
        mutable IMoEKernel *moe_kernel_ = nullptr;

        /// Pre-allocated routing result (avoids heap allocs per decode token)
        mutable MoERoutingResult cached_routing_;
        DeviceMoELayerRuntime *moe_runtime_layer_ = nullptr;

        IMoEKernel *ensureMoEKernel() const;
        bool isDeviceRoutedDecodeGraphCapturable() const;
        bool isDeviceRoutedPrefillGraphCaptureSupported() const;
        bool isDeviceRoutedPrefillGraphCapturable() const;
        bool hasInitializedRuntimeTableIfProvided() const;
        void recordRuntimeHistogramTokenBoundary() const;
        void stashRoutingResults(
            const std::vector<int> &expert_indices,
            const std::vector<float> &expert_weights,
            int seq_len, int top_k) const;
    };

} // namespace llaminar2
