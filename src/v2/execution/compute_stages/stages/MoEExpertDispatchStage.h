/**
 * @file MoEExpertDispatchStage.h
 * @brief Host-side MoE expert-parallel dispatch descriptor stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../moe/MoEExpertParallelPlan.h"
#include "../../moe/MoEExpertTokenRowTransfer.h"

#include <cstddef>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    struct MoEExpertDispatchEntry
    {
        int token_row = -1;
        int route_slot = -1;
        int expert_id = -1;
        float route_weight = 0.0f;
    };

    struct MoEExpertTierDispatch
    {
        int tier_index = -1;
        std::string tier_name;
        std::string domain;
        bool fallback = false;
        bool transfer_required = false;
        MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::None;
        MoEExpertTransferVolume transfer_volume;
        std::vector<MoEExpertDispatchEntry> entries;

        /// Original dense token-row indices selected for this tier. Sparse
        /// transfer helpers use these indices to scatter-add returned rows.
        std::vector<int> token_rows;
    };

    struct MoEExpertDispatchOutput
    {
        int seq_len = 0;
        int top_k = 0;
        int d_model = 0;
        std::string continuation_domain;
        std::vector<MoEExpertTierDispatch> tiers;

        size_t estimatedTransferBytes() const
        {
            size_t total = 0;
            for (const auto &tier : tiers)
                total += tier.transfer_volume.totalBytes();
            return total;
        }

        size_t denseTransferBytes() const
        {
            size_t total = 0;
            for (const auto &tier : tiers)
            {
                if (tier.transfer_required)
                    total += tier.transfer_volume.denseTotalBytes();
            }
            return total;
        }
    };

    /**
     * @brief Converts router top-k tensors into per-tier host work descriptors.
     *
     * Phase 4 intentionally stops at descriptors: it does not copy hidden rows,
     * launch expert kernels, or perform cross-domain transfer.
     */
    class MoEExpertDispatchStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *routing_indices = nullptr; ///< FP32 expert ids [seq_len, top_k] or flat
            const ITensor *routing_weights = nullptr; ///< FP32 route weights [seq_len, top_k] or flat
            const ITensor *hidden = nullptr;          ///< Optional hidden state for future sparse row transfer
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            std::optional<BufferId> hidden_buffer_id;

            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            std::string continuation_domain; ///< Empty means transfer metadata is informational only
            MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::Auto;

            std::optional<ExpertLayerPlacement> placement;
            std::vector<ExpertRoutedTier> routed_tiers;
            MoEExpertDispatchOutput *output = nullptr;
            std::shared_ptr<MoEExpertDispatchOutput> output_lifetime;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEExpertDispatchStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_DISPATCH; }
        std::string name() const override { return "moe_expert_dispatch"; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
