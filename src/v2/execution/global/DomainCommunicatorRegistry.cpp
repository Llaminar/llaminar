/**
 * @file DomainCommunicatorRegistry.cpp
 * @brief Implementation of DomainCommunicatorRegistry
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#include "DomainCommunicatorRegistry.h"
#include "../../collective/GlobalTPContext.h"
#include "../../utils/Logger.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace llaminar2
{

    void DomainCommunicatorRegistry::initialize(
        const GlobalPPTopology &topology,
        MPI_Comm world_comm,
        int world_rank)
    {
        contexts_.clear();

        // Collect global TP stages sorted by ascending stage_id so every rank
        // calls MPI_Comm_split in the same order (required for correctness).
        std::vector<const GlobalPPStageSpec *> global_tp_stages;
        global_tp_stages.reserve(topology.stages.size());
        for (const auto &stage : topology.stages)
        {
            if (stage.is_global_tp)
                global_tp_stages.push_back(&stage);
        }
        std::sort(global_tp_stages.begin(), global_tp_stages.end(),
                  [](const auto *a, const auto *b)
                  { return a->stage_id < b->stage_id; });

        for (const auto *stage : global_tp_stages)
        {
            const auto &ranks = stage->participating_ranks;

            // Determine color and key for MPI_Comm_split.
            //   - Participants: color=0 (all in the same new communicator),
            //     key = index in participating_ranks (deterministic ordering).
            //   - Non-participants: color=MPI_UNDEFINED (excluded from new comm).
            int color = MPI_UNDEFINED;
            int key = world_rank; // key is irrelevant for MPI_UNDEFINED but must be set

            for (int i = 0; i < static_cast<int>(ranks.size()); ++i)
            {
                if (ranks[i] == world_rank)
                {
                    color = 0;
                    key = i;
                    break;
                }
            }

            // GlobalTPContext::createWithSplit calls MPI_Comm_split internally.
            // For non-participants (color==MPI_UNDEFINED) it returns nullptr after
            // the split completes, so the collective is still satisfied.
            auto ctx = GlobalTPContext::createWithSplit(
                world_comm,
                stage->stage_id, // domain_id = stage_id for unambiguous keying
                color,
                key);

            if (color != MPI_UNDEFINED)
            {
                // We are a participant.
                if (ctx && ctx->isValid())
                {
                    contexts_[stage->stage_id] = std::move(ctx);
                    LOG_INFO("[DomainCommunicatorRegistry] Rank " << world_rank
                                                                  << " joined domain for stage " << stage->stage_id
                                                                  << " (domain_size=" << contexts_[stage->stage_id]->degree() << ")");
                }
                else
                {
                    LOG_ERROR("[DomainCommunicatorRegistry] Failed to create context for stage "
                              << stage->stage_id << " on rank " << world_rank);
                    throw std::runtime_error("DomainCommunicatorRegistry: failed to create GlobalTPContext");
                }
            }
            else
            {
                // Non-participant: ctx must be nullptr (handled by createWithSplit).
                LOG_DEBUG("[DomainCommunicatorRegistry] Rank " << world_rank
                                                               << " is not a participant of stage " << stage->stage_id);
                assert(ctx == nullptr && "createWithSplit should return nullptr for MPI_UNDEFINED");
            }
        }

        LOG_DEBUG("[DomainCommunicatorRegistry] Rank " << world_rank
                                                       << " initialized with " << contexts_.size()
                                                       << " domain context(s)");
    }

    bool DomainCommunicatorRegistry::hasContextForStage(int stage_id) const
    {
        return contexts_.count(stage_id) > 0;
    }

    std::shared_ptr<IGlobalTPContext> DomainCommunicatorRegistry::globalTPContextForStage(int stage_id) const
    {
        auto it = contexts_.find(stage_id);
        if (it == contexts_.end())
            return nullptr;
        return it->second;
    }

    std::vector<int> DomainCommunicatorRegistry::stageIds() const
    {
        std::vector<int> ids;
        ids.reserve(contexts_.size());
        for (const auto &kv : contexts_)
            ids.push_back(kv.first);
        return ids;
    }

    void DomainCommunicatorRegistry::addContextForTest(int stage_id, std::shared_ptr<IGlobalTPContext> ctx)
    {
        contexts_[stage_id] = std::move(ctx);
    }

} // namespace llaminar2
