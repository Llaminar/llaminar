/**
 * @file MTPConcatStage.h
 * @brief Concatenate draft embedding and terminal hidden rows for MTP projection.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    class MTPConcatStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *hidden = nullptr;
            ITensor *embedding = nullptr;
            ITensor *output = nullptr;

            int num_tokens = 1;
            int hidden_dim = 0;

            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> embedding_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MTPConcatStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MTP_CONCAT; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
