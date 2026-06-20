/**
 * @file MoEGraphRoleRunner.h
 * @brief Scalar-only graph-native participant runner for MoE overlay role execution.
 */

#pragma once

#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "backends/DeviceId.h"
#include "interfaces/IMPIContext.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    enum class MoEGraphParticipantRole
    {
        Continuation,
        AcceleratorExpert,
        CpuExpert,
        Relay,
    };

    struct MoEGraphStepHeader
    {
        uint64_t generation_id = 1;
        uint64_t step_id = 0;
        int seq_len = 0;
        int position_offset = 0;
        bool decode = false;
    };

    struct MoEGraphParticipantSpec
    {
        int world_rank = -1;
        int participant_id = -1;
        DeviceId device = DeviceId::invalid();
        MoEGraphParticipantRole role = MoEGraphParticipantRole::Relay;
        bool continuation_root = false;
    };

    struct MoEGraphParticipantStatus
    {
        MoEGraphParticipantSpec participant;
        bool ok = false;
        uint64_t generation_id = 0;
        uint64_t step_id = 0;
        int seq_len = 0;
        int position_offset = 0;
    };

    class MoEGraphRoleRunner final : public IInferenceRunner
    {
    public:
        struct Config
        {
            std::vector<MoEGraphParticipantSpec> participant_specs;
            std::vector<std::unique_ptr<IInferenceRunner>> local_participant_runners;
            std::shared_ptr<IMPIContext> mpi_ctx;
            bool continuation_root = false;
            std::string architecture;
            uint64_t generation_id = 1;
        };

        explicit MoEGraphRoleRunner(Config config);

        bool forward(const int *tokens, int seq_len) override;
        const float *logits() const override;
        int vocab_size() const override;
        void clear_cache() override;
        int get_position() const override;
        ExecutionPath executionPath() const override;
        const char *architecture() const override;

        const MoEGraphStepHeader &lastStepHeader() const { return last_header_; }
        const std::vector<MoEGraphParticipantStatus> &lastStatuses() const { return last_statuses_; }

        static std::string participantIdentityKey(const MoEGraphParticipantSpec &spec);

    private:
        struct RankSummary
        {
            int world_rank = 0;
            int ok = 0;
            uint64_t generation_id = 0;
            uint64_t step_id = 0;
            int seq_len = 0;
            int position_offset = 0;
        };

        int continuationRootRank() const;
        bool publishAndReceiveStepHeader(MoEGraphStepHeader &header);
        bool gatherStatuses(const MoEGraphStepHeader &header, bool local_ok);

        Config config_;
        std::vector<MoEGraphParticipantSpec> local_specs_;
        int position_ = 0;
        uint64_t next_step_id_ = 0;
        std::string architecture_;
        int continuation_local_runner_index_ = -1;
        MoEGraphStepHeader last_header_{};
        std::vector<MoEGraphParticipantStatus> last_statuses_;
    };

} // namespace llaminar2
