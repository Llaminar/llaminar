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
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../moe/DecodeExpertHistogram.h"
#include "../../moe/MoERuntimeTable.h"

#include <memory>
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
            bool force_grouped_verifier_prefill_for_decode = false;

            /**
             * @brief Permit eager GPU routeWithTensors() for partial expert owners.
             *
             * Single-device GPU decode must use the device-routed runtime table
             * so graph capture, route metadata, and expert execution share one
             * device-owned source of truth.  LocalTP ExpertParallel runners are
             * different: each participant owns only a subset of experts and the
             * current runtime table is a full-owner contract.  Until the sharded
             * runtime-table reducer exists, those TP participants use the explicit
             * mask/range + allreduce path in eager mode and must not advertise
             * decode graph capture.
             */
            bool allow_eager_gpu_single_row_route_for_partial_expert_owner = false;

            /**
             * @brief Replay verifier rows through the normal one-token route path.
             *
             * MTP all-position verifier batches produce several candidate rows at
             * once.  The row we later publish must behave exactly as if decode
             * had processed that token by itself.  Some backend router kernels
             * choose different math for tiny prefill batches than for M=1 decode,
             * so this flag asks the stage to split the batch internally and route
             * each row with seq_len=1 before scattering the rows back.
             */
            bool force_decode_equivalent_verifier_prefill = false;

            // Outputs (written by this stage)
            TensorBase *output_indices = nullptr; ///< FP32 [seq_len * top_k] expert IDs as float
            TensorBase *output_weights = nullptr; ///< FP32 [seq_len * top_k] normalized weights
            // Buffer IDs for coherence
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
            BufferId output_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
        };

        explicit MoERoutingStage(Params params);
        ~MoERoutingStage() override;

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        std::string name() const override { return "moe_router"; }
        size_t estimatedFlops() const override;

        int layerIndex() const { return params_.layer_idx; }

        bool allowsZeroOutput() const override { return false; }
        bool isGraphCapturable() const override;
        bool supportsWarmupDependentGraphCapture() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool hasPrefillReplayParams() const override { return params_.device_id.is_gpu() && params_.seq_len > 1; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        bool needsGraphLaunchPreparation() const override { return hasPrefillReplayParams(); }
        void onGraphReplayed() override;
        bool needsOnGraphReplayed() const override;
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
         * @brief Clear routing mirrors while preserving captured MoE replay slots.
         *
         * Padded prefill replay refreshes the effective-length scalar before
         * graph launch and records histogram boundaries after replay. Keeping
         * the workspace-backed scalar and runtime table identity alive is
         * required for Ready prefill graphs preserved across clear_cache().
         */
        void resetSessionStatePreservingCapturedReplay() override;
        /**
         * @brief Preserve warmed routing workspace for capture-from-Initialized.
         */
        void resetSessionStatePreservingLazyInitialization() override;

        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel) { moe_kernel_ = kernel; }

    private:
        struct GpuEffectiveSeqLenState;

        Params params_;

        /// Stashed routing results for snapshot capture
        mutable std::vector<float> routing_indices_f32_;
        mutable std::vector<float> routing_weights_;
        mutable std::vector<float> router_logits_;

        /// Cached MoE kernel
        mutable IMoEKernel *moe_kernel_ = nullptr;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

        /// Pre-allocated routing result (avoids heap allocs per decode token)
        mutable MoERoutingResult cached_routing_;
        DeviceMoELayerRuntime *moe_runtime_layer_ = nullptr;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        std::unique_ptr<GpuEffectiveSeqLenState> gpu_effective_seq_len_state_;

        IMoEKernel *ensureMoEKernel() const;
        bool isDeviceRoutedDecodeGraphCapturable() const;
        bool isDeviceRoutedPrefillExecutionSupported() const;
        bool isDeviceRoutedPrefillGraphCaptureSupported() const;
        bool isDeviceRoutedPrefillGraphCapturable() const;
        bool isDecodeEquivalentVerifierPrefillExecutionSupported() const;
        bool isDecodeEquivalentVerifierPrefillGraphCaptureSupported() const;
        bool isDecodeEquivalentVerifierPrefillGraphCapturable() const;
        bool hasInitializedRuntimeTableIfProvided() const;
        bool executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);
        int effectivePrefillSeqLen() const;
        void refreshPinnedEffectiveSeqLen();
        bool ensureGpuEffectiveSeqLenStateInitialized();
        bool uploadGpuEffectiveSeqLen();
        void releaseGpuEffectiveSeqLenState();
        void recordRuntimeHistogramTokenBoundary() const;
        void stashRoutingResults(
            const std::vector<int> &expert_indices,
            const std::vector<float> &expert_weights,
            int seq_len, int top_k) const;

    };

} // namespace llaminar2
