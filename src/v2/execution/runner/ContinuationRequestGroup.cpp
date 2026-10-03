/**
 * @file ContinuationRequestGroup.cpp
 * @brief Minimal prompt transport and prefix consensus for continuation owners.
 *
 * World command metadata and sparse execution tickets remain unchanged. The
 * retained continuation group is the only recipient authority for prompt
 * bytes, so expert-only followers neither allocate nor receive unused tokens.
 * No per-request communicator creation, barriers or payload padding is needed.
 */
#include "ContinuationRequestGroup.h"

#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    ContinuationRequestGroup::ContinuationRequestGroup(
        std::shared_ptr<IMPIContext> world, std::vector<int> consumer_ranks)
        : world_(std::move(world)), consumer_ranks_(std::move(consumer_ranks))
    {
        if (!world_ || world_->world_size() <= 0 || world_->rank() < 0 ||
            world_->rank() >= world_->world_size() || consumer_ranks_.empty() ||
            !std::is_sorted(consumer_ranks_.begin(), consumer_ranks_.end()) ||
            std::adjacent_find(consumer_ranks_.begin(), consumer_ranks_.end()) !=
                consumer_ranks_.end() ||
            consumer_ranks_.front() < 0 || consumer_ranks_.back() >= world_->world_size())
            throw std::invalid_argument("Continuation request group has invalid execution membership");

        role_ = std::binary_search(consumer_ranks_.begin(), consumer_ranks_.end(), world_->rank())
                    ? Role::TokenConsumer : Role::ExpertOnly;
        if (consumer_ranks_.size() == static_cast<std::size_t>(world_->world_size()))
        {
            scope_ = Scope::OrchestrationWorld;
            return;
        }
        if (consumer_ranks_.size() == 1u)
        {
            scope_ = Scope::ProcessLocalContinuation;
            return;
        }

        // Subgroup order is execution-rank order, not device order or root zero.
        // Expert-only peers join this startup split, never its request payloads.
        scope_ = Scope::ContinuationRankGroup;
        if (world_->communicator() == MPI_COMM_NULL)
            throw std::invalid_argument("Continuation request subgroup has no live MPI context");
        MPI_Comm subgroup = MPI_COMM_NULL;
        if (MPI_Comm_split(world_->communicator(),
                           role_ == Role::TokenConsumer ? 1 : MPI_UNDEFINED,
                           world_->rank(), &subgroup) != MPI_SUCCESS)
            throw std::runtime_error("Failed to create continuation request subgroup");

        int size = 0;
        if ((role_ == Role::TokenConsumer &&
             (subgroup == MPI_COMM_NULL || MPI_Comm_size(subgroup, &size) != MPI_SUCCESS ||
              size != static_cast<int>(consumer_ranks_.size()))) ||
            (role_ == Role::ExpertOnly && subgroup != MPI_COMM_NULL))
        {
            if (subgroup != MPI_COMM_NULL)
                MPI_Comm_free(&subgroup);
            throw std::runtime_error("Continuation request subgroup has incorrect membership");
        }
        group_comm_ = subgroup;
    }

    ContinuationRequestGroup::~ContinuationRequestGroup()
    {
        if (group_comm_ == MPI_COMM_NULL)
            return;
        int initialized = 0, finalized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
            MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
            LOG_ERROR("[ContinuationRequestGroup] Owned communicator outlived MPI");
        else if (MPI_Comm_free(&group_comm_) != MPI_SUCCESS)
            LOG_ERROR("[ContinuationRequestGroup] Failed to retire owned communicator");
    }

    const char *ContinuationRequestGroup::scopeName() const noexcept
    {
        switch (scope_)
        {
        case Scope::OrchestrationWorld: return "orchestration_world";
        case Scope::ProcessLocalContinuation: return "process_local_continuation";
        case Scope::ContinuationRankGroup: return "continuation_rank_group";
        }
        std::terminate();
    }

    MPI_Comm ContinuationRequestGroup::coordinationCommunicator() const
    {
        if (role_ != Role::TokenConsumer)
            throw std::logic_error("Expert-only rank attempted continuation prefix/KV consensus");
        switch (scope_)
        {
        case Scope::OrchestrationWorld:
            return world_->world_size() > 1 ? world_->communicator() : MPI_COMM_NULL;
        case Scope::ProcessLocalContinuation:
            return MPI_COMM_NULL;
        case Scope::ContinuationRankGroup:
            if (group_comm_ == MPI_COMM_NULL)
                throw std::logic_error("Continuation consumer lost its retained request subgroup");
            return group_comm_;
        }
        std::terminate();
    }

    int ContinuationRequestGroup::validatePrompt(std::size_t token_count, int root_rank) const
    {
        if (token_count == 0u ||
            token_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Prompt payload count must be positive and fit the MPI wire extent");
        const auto root = std::lower_bound(consumer_ranks_.begin(), consumer_ranks_.end(), root_rank);
        if (root == consumer_ranks_.end() || *root != root_rank)
            throw std::invalid_argument("Prompt command root is not a continuation consumer");
        return static_cast<int>(root - consumer_ranks_.begin());
    }

    void ContinuationRequestGroup::broadcastPrompt(
        std::int32_t *tokens, std::size_t count, int root_rank, int group_root) const
    {
        if (scope_ == Scope::OrchestrationWorld && world_->world_size() > 1)
            world_->broadcast_int32(tokens, count, root_rank);
        else if (scope_ == Scope::ContinuationRankGroup)
        {
            if (MPI_Bcast(tokens, static_cast<int>(count), MPI_INT32_T,
                          group_root, coordinationCommunicator()) != MPI_SUCCESS)
                throw std::runtime_error("Continuation prompt payload broadcast failed");
        }
        // A sole consumer already owns its tokens. Excluded ranks never reach
        // this method, even when they share a process-local continuation scope.
    }

    void ContinuationRequestGroup::publishPrompt(
        std::span<const std::int32_t> tokens, int root_rank) const
    {
        const int group_root = validatePrompt(tokens.size(), root_rank);
        if (role_ != Role::TokenConsumer || world_->rank() != root_rank)
            throw std::logic_error("Only the continuation command root may publish a prompt");
        // MPI leaves the root buffer untouched. This avoids an unnecessary
        // prompt-sized copy merely to accommodate MPI's mutable-buffer API.
        broadcastPrompt(const_cast<std::int32_t *>(tokens.data()), tokens.size(), root_rank, group_root);
        recordPromptExtent(tokens.size(), root_rank);
    }

    std::vector<std::int32_t> ContinuationRequestGroup::receivePrompt(
        std::size_t token_count, int root_rank) const
    {
        const int group_root = validatePrompt(token_count, root_rank);
        if (world_->rank() == root_rank)
            throw std::logic_error("Continuation command root cannot receive its own prompt");
        if (role_ == Role::ExpertOnly)
        {
            recordPromptExtent(token_count, root_rank);
            return {}; // No allocation and no payload exchange on expert followers.
        }
        std::vector<std::int32_t> tokens(token_count);
        broadcastPrompt(tokens.data(), token_count, root_rank, group_root);
        recordPromptExtent(token_count, root_rank);
        return tokens;
    }

    void ContinuationRequestGroup::recordPromptExtent(std::size_t count, int root_rank) const
    {
        if (!PerfStatsCollector::isDomainEnabled("orchestration_command"))
            return;
        const std::size_t copies = role_ == Role::ExpertOnly ? 0u
            : (world_->rank() == root_rank ? consumerCount() - 1u : 1u);
        PerfStatsCollector::addCounter("orchestration_command", "prompt_payload_bytes",
            static_cast<double>(count) * sizeof(std::int32_t) * copies, "prefill", {},
            {{"scope", scopeName()}, {"consumer_ranks", std::to_string(consumerCount())},
             {"world_root_rank", std::to_string(root_rank)},
             {"payload_role", role_ == Role::ExpertOnly ? "expert_only" : "continuation"},
             {"direction", world_->rank() == root_rank ? "logical_send" : "receive"}});
    }
}
