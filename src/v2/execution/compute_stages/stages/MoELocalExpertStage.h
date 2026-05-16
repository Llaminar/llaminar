/**
 * @file MoELocalExpertStage.h
 * @brief Participant-local graph-native sparse MoE expert compute stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../moe/MoEOverlaySparseCollective.h"
#include "../../../loaders/ExpertSlabTypes.h"

#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{
    class FP32Tensor;
    class TensorBase;
    class ITensorGemm;
    class PreparedWeightStore;
    class ExpertGemmRegistry;
    class DeviceWorkspaceManager;

    class MoELocalExpertStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const MoEOverlaySparseRows *input_rows = nullptr;
            std::shared_ptr<const MoEOverlaySparseRows> input_rows_lifetime;
            MoEOverlayReturnRows *output_rows = nullptr;
            std::shared_ptr<MoEOverlayReturnRows> output_rows_lifetime;
            std::shared_ptr<MoEOverlayCollectiveWorkspace> workspace_lifetime;

            TensorBase *gate_exps = nullptr;
            TensorBase *up_exps = nullptr;
            TensorBase *down_exps = nullptr;
            int num_experts = 0;
            int top_k = 0;
            int d_model = 0;
            int expert_intermediate = 0;
            int layer_idx = -1;
            std::vector<bool> expert_mask;

            // ================================================================
            // Prepared expert state — analogous to MoEExpertComputeStage::Params.
            // No peer participant / runtime / runner fields.
            // When prepared_gate_gemm is non-empty (size == num_experts), the
            // execute() path skips inline extractExpertViews /
            // prepareExpertGemmEngines and uses these engines directly.
            // ================================================================
            std::vector<ITensorGemm *> prepared_gate_gemm;
            std::vector<ITensorGemm *> prepared_up_gemm;
            std::vector<ITensorGemm *> prepared_down_gemm;
            /// Keeps MoE batch-constructed kernel objects alive alongside the stage.
            std::vector<std::shared_ptr<ITensorGemm>> moe_owned_kernels;
            /// Pointer to model-context-owned PreparedWeightStore.  Not owned here.
            PreparedWeightStore *prepared_store = nullptr;
            /// Pointer to model-context-owned ExpertGemmRegistry.  Not owned here.
            ExpertGemmRegistry *expert_registry = nullptr;
            /// Cached slab references for PreparedWeightStore-based resolution.
            std::optional<ExpertSlabRef> gate_slab_ref;
            std::optional<ExpertSlabRef> up_slab_ref;
            std::optional<ExpertSlabRef> down_slab_ref;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoELocalExpertStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::MOE_LOCAL_EXPERT; }
        std::string name() const override { return "moe_local_expert"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        const Params &params() const { return params_; }

    private:
        bool ensureCompactCapacity(size_t rows) const;

        Params params_;
        mutable size_t compact_capacity_ = 0;
        mutable std::shared_ptr<FP32Tensor> compact_hidden_;
        mutable std::shared_ptr<FP32Tensor> compact_routing_indices_;
        mutable std::shared_ptr<FP32Tensor> compact_routing_weights_;
        mutable std::shared_ptr<FP32Tensor> compact_output_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
    };

} // namespace llaminar2