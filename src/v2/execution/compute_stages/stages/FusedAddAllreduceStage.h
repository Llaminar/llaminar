/**
 * @file FusedAddAllreduceStage.h
 * @brief Fused residual-add + TP allreduce stage
 *
 * Combines: output = input_a + input_b  followed by  MPI_Allreduce(output)
 * into a single stage dispatch, eliminating one stage boundary per MoE layer.
 *
 * Used in MoE TP path where routed expert output + shared expert output
 * are summed and then allreduced across ranks.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../../collective/ITPContext.h"

#include <optional>
#include <string>

namespace llaminar2
{

    class FusedAddAllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Residual-add inputs
            const ITensor *input_a = nullptr;  ///< First addend (e.g., MOE_COMBINED_OUTPUT)
            const ITensor *input_b = nullptr;  ///< Second addend (e.g., MOE_SHARED_EXPERT_OUTPUT)
            ITensor *output = nullptr;         ///< Sum destination (e.g., ATTN_PROJ), also allreduce target

            /// Number of elements to process (0 = use input_a->numel())
            size_t num_elements = 0;

            // TP allreduce params
            ITPContext *tp_ctx = nullptr;       ///< TP context for allreduce
            size_t allreduce_count = 0;        ///< Elements to allreduce (0 = use num_elements)
            std::string stage_name;            ///< Stage name for TP context lookup

            // Buffer IDs for contract-based coherence
            std::optional<BufferId> input_a_buffer_id;
            std::optional<BufferId> input_b_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        explicit FusedAddAllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::FUSED_ADD_ALLREDUCE; }
        std::string name() const override { return "FusedAddAllreduce"; }
        bool requiresAllreduce() const override { return true; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
