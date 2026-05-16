#include "MoEGraphRoleRunner.h"

#include "utils/Logger.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace llaminar2
{
    namespace
    {
        struct WireHeader
        {
            uint64_t generation_id = 0;
            uint64_t step_id = 0;
            int seq_len = 0;
            int position_offset = 0;
            int decode = 0;
        };
    }

    MoEGraphRoleRunner::MoEGraphRoleRunner(Config config)
        : config_(std::move(config)), architecture_(config_.architecture)
    {
        const int local_rank = config_.mpi_ctx ? config_.mpi_ctx->rank() : 0;
        for (const auto &spec : config_.participant_specs)
        {
            if (spec.world_rank == local_rank)
                local_specs_.push_back(spec);
        }

        const size_t paired = std::min(local_specs_.size(), config_.local_participant_runners.size());
        for (size_t index = 0; index < paired; ++index)
        {
            if (local_specs_[index].continuation_root)
            {
                continuation_local_runner_index_ = static_cast<int>(index);
                break;
            }
        }

        if (architecture_.empty())
        {
            if (continuation_local_runner_index_ >= 0)
            {
                architecture_ = config_.local_participant_runners[static_cast<size_t>(continuation_local_runner_index_)]->architecture();
            }
            else if (!config_.local_participant_runners.empty())
            {
                architecture_ = config_.local_participant_runners.front()->architecture();
            }
        }

        next_step_id_ = 0;
        last_header_.generation_id = config_.generation_id;
    }

    bool MoEGraphRoleRunner::forward(const int *tokens, int seq_len)
    {
        MoEGraphStepHeader header;
        header.generation_id = config_.generation_id;
        header.step_id = next_step_id_;
        header.seq_len = seq_len;
        header.position_offset = position_;
        header.decode = (seq_len == 1);

        if (!publishAndReceiveStepHeader(header))
            return false;

        bool local_ok = true;
        for (auto &participant_runner : config_.local_participant_runners)
        {
            if (!participant_runner)
                continue;
            if (!participant_runner->forward(tokens, header.seq_len))
                local_ok = false;
        }

        if (local_ok)
            position_ = header.position_offset + header.seq_len;

        last_header_ = header;
        next_step_id_ = header.step_id + 1;
        return gatherStatuses(header, local_ok);
    }

    const float *MoEGraphRoleRunner::logits() const
    {
        if (continuation_local_runner_index_ < 0)
            return nullptr;
        const size_t index = static_cast<size_t>(continuation_local_runner_index_);
        if (index >= config_.local_participant_runners.size() || !config_.local_participant_runners[index])
            return nullptr;
        return config_.local_participant_runners[index]->logits();
    }

    int MoEGraphRoleRunner::vocab_size() const
    {
        if (continuation_local_runner_index_ < 0)
            return 0;
        const size_t index = static_cast<size_t>(continuation_local_runner_index_);
        if (index >= config_.local_participant_runners.size() || !config_.local_participant_runners[index])
            return 0;
        return config_.local_participant_runners[index]->vocab_size();
    }

    void MoEGraphRoleRunner::clear_cache()
    {
        position_ = 0;
        next_step_id_ = 0;
        for (auto &participant_runner : config_.local_participant_runners)
        {
            if (participant_runner)
                participant_runner->clear_cache();
        }
    }

    int MoEGraphRoleRunner::get_position() const
    {
        return position_;
    }

    ExecutionPath MoEGraphRoleRunner::executionPath() const
    {
        return ExecutionPath::GRAPH;
    }

    const char *MoEGraphRoleRunner::architecture() const
    {
        return architecture_.c_str();
    }

    std::string MoEGraphRoleRunner::participantIdentityKey(const MoEGraphParticipantSpec &spec)
    {
        std::ostringstream out;
        out << "world=" << spec.world_rank << ";participant=" << spec.participant_id;
        return out.str();
    }

    int MoEGraphRoleRunner::continuationRootRank() const
    {
        for (const auto &spec : config_.participant_specs)
        {
            if (spec.continuation_root)
                return spec.world_rank;
        }
        if (config_.continuation_root && config_.mpi_ctx)
            return config_.mpi_ctx->rank();
        return 0;
    }

    bool MoEGraphRoleRunner::publishAndReceiveStepHeader(MoEGraphStepHeader &header)
    {
        if (!config_.mpi_ctx || config_.mpi_ctx->world_size() <= 1 || config_.mpi_ctx->communicator() == MPI_COMM_NULL)
            return true;

        WireHeader wire;
        const int root_rank = continuationRootRank();
        if (config_.mpi_ctx->rank() == root_rank)
        {
            wire.generation_id = header.generation_id;
            wire.step_id = header.step_id;
            wire.seq_len = header.seq_len;
            wire.position_offset = header.position_offset;
            wire.decode = header.decode ? 1 : 0;
        }

        const int rc = MPI_Bcast(&wire, static_cast<int>(sizeof(WireHeader)), MPI_BYTE, root_rank, config_.mpi_ctx->communicator());
        if (rc != MPI_SUCCESS)
        {
            LOG_ERROR("[MoEGraphRoleRunner] Failed to broadcast MoE graph step header");
            return false;
        }

        header.generation_id = wire.generation_id;
        header.step_id = wire.step_id;
        header.seq_len = wire.seq_len;
        header.position_offset = wire.position_offset;
        header.decode = wire.decode != 0;
        return true;
    }

    bool MoEGraphRoleRunner::gatherStatuses(const MoEGraphStepHeader &header, bool local_ok)
    {
        std::vector<RankSummary> summaries;
        const int world_size = config_.mpi_ctx ? config_.mpi_ctx->world_size() : 1;

        RankSummary local;
        local.world_rank = config_.mpi_ctx ? config_.mpi_ctx->rank() : 0;
        local.ok = local_ok ? 1 : 0;
        local.generation_id = header.generation_id;
        local.step_id = header.step_id;
        local.seq_len = header.seq_len;
        local.position_offset = header.position_offset;

        if (!config_.mpi_ctx || world_size <= 1 || config_.mpi_ctx->communicator() == MPI_COMM_NULL)
        {
            summaries.push_back(local);
        }
        else
        {
            summaries.resize(static_cast<size_t>(world_size));
            const int rc = MPI_Allgather(
                &local,
                static_cast<int>(sizeof(RankSummary)),
                MPI_BYTE,
                summaries.data(),
                static_cast<int>(sizeof(RankSummary)),
                MPI_BYTE,
                config_.mpi_ctx->communicator());
            if (rc != MPI_SUCCESS)
            {
                LOG_ERROR("[MoEGraphRoleRunner] Failed to collect MoE graph participant status");
                return false;
            }
        }

        std::vector<MoEGraphParticipantStatus> statuses;
        statuses.reserve(config_.participant_specs.size());

        bool global_ok = true;
        for (const auto &summary : summaries)
        {
            const bool match =
                summary.generation_id == header.generation_id &&
                summary.step_id == header.step_id &&
                summary.seq_len == header.seq_len;
            if (!match)
                global_ok = false;
            if (summary.ok == 0)
                global_ok = false;
        }

        for (const auto &spec : config_.participant_specs)
        {
            const auto it = std::find_if(summaries.begin(), summaries.end(),
                                         [&](const RankSummary &summary)
                                         {
                                             return summary.world_rank == spec.world_rank;
                                         });
            MoEGraphParticipantStatus status;
            status.participant = spec;
            status.generation_id = header.generation_id;
            status.step_id = header.step_id;
            status.seq_len = header.seq_len;
            status.position_offset = header.position_offset;
            status.ok = false;

            if (it != summaries.end())
            {
                const bool match =
                    it->generation_id == header.generation_id &&
                    it->step_id == header.step_id &&
                    it->seq_len == header.seq_len;
                status.ok = (it->ok != 0) && match;
            }

            statuses.push_back(status);
        }

        last_statuses_ = std::move(statuses);
        return global_ok;
    }

} // namespace llaminar2
