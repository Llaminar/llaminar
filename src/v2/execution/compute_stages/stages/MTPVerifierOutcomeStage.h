/**
 * @file MTPVerifierOutcomeStage.h
 * @brief Graph-owned terminal reduction and LocalTP publication for MTP.
 *
 * The stage is the explicit end of a grouped verifier graph.  A root
 * participant converts full-vocabulary verifier logits into row-wise greedy
 * tokens, compares those tokens with the device-owned draft row, and writes the
 * compact speculative outcome.  A mirrored LocalTP graph then broadcasts the
 * compact token and metadata rows on the same captured stream.
 *
 * No allocation, host/device transfer, host scalar outcome, or stream
 * synchronization is permitted in execute().  Every address is a persistent
 * arena binding and every ordering relationship is represented by graph
 * dependencies or the stage's explicit stream.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../../memory/BufferId.h"

#include <array>
#include <string>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Capture the complete greedy verifier outcome transaction.
     *
     * In SingleDevice mode the sole participant performs the reduction.  In a
     * homogeneous mirrored LocalTP domain only participant zero performs the
     * reduction and every participant enters matching NCCL/RCCL broadcasts.
     * Non-root participants therefore avoid redundant full-vocabulary argmax
     * work while still receiving device-local compact buffers for later state
     * publication.
     */
    class MTPVerifierOutcomeStage final : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *logits = nullptr;
            MTPVerifierOutcomeGraphMode mode =
                MTPVerifierOutcomeGraphMode::Disabled;
            MTPVerifierOutcomeGraphBinding binding;

            int verifier_row_count = 0;
            int vocab_size = 0;

            ILocalTPContext *local_tp_ctx = nullptr;
            int local_tp_device_index = 0;
            int local_tp_root_device_index = 0;
            bool publish_mirrored_local_tp = false;

            std::string stage_name = "mtp_verifier_outcome";
        };

        static_assert(StageParamsRequired<Params>);

        explicit MTPVerifierOutcomeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_VERIFIER_OUTCOME;
        }
        std::string name() const override { return params_.stage_name; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return true; }
        bool isCollectiveStage() const override
        {
            return params_.publish_mirrored_local_tp;
        }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const noexcept { return params_; }

    private:
        [[nodiscard]] bool validate() const;
        [[nodiscard]] bool isRootParticipant() const noexcept;

        Params params_;
        std::array<LocalTPCollectiveSidebandBuffer, 2>
            mirrored_outcome_sidebands_;
    };

} // namespace llaminar2
