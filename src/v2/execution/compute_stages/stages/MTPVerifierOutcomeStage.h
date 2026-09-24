/**
 * @file MTPVerifierOutcomeStage.h
 * @brief Graph-owned participant-local terminal reduction for MTP.
 *
 * The stage is the explicit end of a grouped verifier graph.  Each participant
 * converts its full-vocabulary verifier logits into row-wise greedy tokens,
 * compares those tokens with its device-owned draft row, and writes its compact
 * speculative outcome. A mirrored TP graph therefore remains completely
 * participant-local at every scope and contains no compact control collective.
 *
 * No allocation, host/device transfer, host scalar outcome, or stream
 * synchronization is permitted in execute().  Every address is a persistent
 * arena binding and every ordering relationship is represented by graph
 * dependencies or the stage's explicit stream.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../../memory/BufferId.h"

#include <string>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Capture the complete greedy verifier outcome transaction.
     *
     * In SingleDevice mode the sole participant performs the reduction. In a
     * homogeneous mirrored TP domain every participant performs identical
     * reduction math against its local full-vocabulary logits. This modest
     * duplicate arithmetic removes latency-dominated tiny collectives and makes
     * each child outcome ready on the stream that actually produced it.
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

            MTPVerifierOutcomeOwnershipPolicy ownership_policy =
                MTPVerifierOutcomeOwnershipPolicy::Unspecified;
            /** True when this participant's logits contain the full vocabulary. */
            bool participant_full_vocabulary = false;

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
        bool isCollectiveStage() const override { return false; }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const noexcept { return params_; }

    private:
        [[nodiscard]] bool validate() const;

        Params params_;
    };

} // namespace llaminar2
