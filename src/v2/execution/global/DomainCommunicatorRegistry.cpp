/**
 * @file DomainCommunicatorRegistry.cpp
 * @brief Exact admitted PP-stage communicator and endpoint construction.
 *
 * Stage membership determines split ordering and the participant's physical
 * address. No world-rank or CPU-zero convention can replace that topology.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#include "DomainCommunicatorRegistry.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include <optional>
#include <variant>
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
        int world_rank,std::span<const OriginalCPUPipelineProgram> cpu_expert_programs,
        std::shared_ptr<PhysicalMemoryAuthority> original_memory)
    {
        if(cpu_owners_) throw std::logic_error("Original CPU pipeline domain registry is setup-only");
        contexts_.clear();
        if(!cpu_expert_programs.empty()) cpu_owners_=std::make_shared<OriginalCPUPipelineDomainOwners>();

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

            std::shared_ptr<GlobalTPContext> ctx;
            const auto cpu=std::find_if(cpu_expert_programs.begin(),cpu_expert_programs.end(),
                [&](const auto& program){return program.stageId()==stage->stage_id;});
            if(cpu!=cpu_expert_programs.end()) {
                const auto& program=cpu->nativeProgram();
                if(program.participants().size()!=stage->participating_ranks.size() ||
                    cpu->scope().first_layer!=stage->first_layer || cpu->scope().last_layer!=stage->last_layer+1)
                    throw std::invalid_argument("CPU pipeline context changed its original stage program geometry");
                for(std::size_t member=0;member<program.participants().size();++member)
                    if(program.participants()[member].world_rank!=stage->participating_ranks[member] ||
                        stage->per_rank_devices.size()!=program.participants().size() ||
                        program.participants()[member].address!=stage->per_rank_devices[member])
                        throw std::invalid_argument("CPU pipeline context changed its original rank/NUMA program");
                std::optional<std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse>> use;
                const auto ready=RankInitializationLifecycle::execute({0,"cpu_pipeline_original_domain_commitment"},[&]{
                    if(!original_memory || original_memory->worldRank()!=world_rank)
                        throw std::invalid_argument("CPU pipeline split requires its original rank PMA");
                    if(color==MPI_UNDEFINED) use.emplace(std::in_place_type<NativeCPUCollectiveProgram::ProcessUse>,
                        program.admitExcludedParent(original_memory,world_rank));
                    else use.emplace(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,
                        program.admit(original_memory,world_rank));
                    return true;
                },[&](auto identity,auto outcome){return MPIRankInitializationConsensus::reach(world_comm,identity,outcome);});
                if(!ready.succeeded()) throw std::invalid_argument(ready.diagnostic("CPU pipeline original domain"));
                auto original=GlobalTPContext::createAdmittedCPUExpertDomain(world_comm,program,std::move(*use));
                if(original) {cpu_owners_->retain(original);ctx=std::move(original);}
            }
            else {
                // The ordinary global stage path remains unchanged. Every
                // excluded original process still enters this SAME parent split.
                ctx = GlobalTPContext::createWithSplit(
                    world_comm,stage->stage_id,color,key,
                    color == MPI_UNDEFINED ? std::nullopt : std::optional<GlobalDeviceAddress>(
                        stage->per_rank_devices.empty() ? stage->per_rank_device : stage->per_rank_devices.at(key)),
                    "",stage->backend);
            }

            if (color != MPI_UNDEFINED)
            {
                // We are a participant.
                if (ctx && ctx->isValid())
                {
                    contexts_[stage->stage_id] = std::move(ctx);
                    LOG_DEBUG("[DomainCommunicatorRegistry] Rank " << world_rank
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
