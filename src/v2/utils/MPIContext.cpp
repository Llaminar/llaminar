/**
 * @file MPIContext.cpp
 * @brief Owned execution-communicator admission using existing phase consensus.
 *
 * This is a startup-only topology transaction, not an inference controller.
 * Local preparation is authenticated before MPI_Comm_split; excluded ranks
 * leave with a typed inactive result. Active contexts retain projected hardware
 * and the parent owner, so later planning cannot rediscover or substitute WORLD.
 */
#include "utils/MPIContext.h"
#include "planning/ExecutionRankMembership.h"
#include "execution/runner/RankInitializationLifecycle.h"

namespace llaminar2
{
    MPIContext::MPIContext(std::shared_ptr<IMPIContext> discovery,
        std::shared_ptr<const ExecutionRankMembership> membership, int rank)
        : discovery_owner_(std::move(discovery)), membership_(std::move(membership)),
          rank_(rank), world_size_(membership_->inventory().world_size), comm_(MPI_COMM_NULL),
          local_rank_(membership_->inventory().ranks.at(rank).local_rank)
    {
    }

    MPIContext::~MPIContext()
    {
        // Topology owns derived communicators. Their destruction must precede
        // the parent split, including when the last owner is not the runner.
        topology_.reset();
        if (communicator_ownership_ == CommunicatorOwnership::Owned && comm_ != MPI_COMM_NULL)
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Comm_free(&comm_);
        }
    }

    std::variant<InactiveMPIRank, std::shared_ptr<MPIContext>> MPIContextFactory::selectRanks(
        const std::shared_ptr<IMPIContext> &discovery, const std::vector<int> &ranks)
    {
        if (!discovery || discovery->communicator() == MPI_COMM_NULL)
            throw std::invalid_argument("Execution admission requires a live discovery communicator");
        const auto comm = discovery->communicator();
        int actual_rank = -1, actual_size = 0;
        if (MPI_Comm_rank(comm, &actual_rank) != MPI_SUCCESS || MPI_Comm_size(comm, &actual_size) != MPI_SUCCESS)
            throw std::runtime_error("Cannot read execution admission communicator membership");
        std::uint32_t ordinal = 0;
        const auto phase = [&](std::string_view name, const auto &work) {
            const auto result = RankInitializationLifecycle::execute({ordinal++, name}, work,
                [&](auto identity, auto outcome) { return MPIRankInitializationConsensus::reach(comm, identity, outcome); });
            if (!result.succeeded()) throw std::runtime_error(result.diagnostic("Execution rank admission"));
        };
        phase("validate_discovery_context", [&] {
            return discovery->rank() == actual_rank && discovery->world_size() == actual_size;
        });
        // The original context owns the only hardware exchange. In an admitted
        // context this is already a projection, so nested selection is pure too.
        const auto inventory = discovery->clusterInventory();
        std::shared_ptr<const ExecutionRankMembership> membership;
        std::shared_ptr<MPIContext> active;
        std::vector<int> published;
        phase("prepare_execution_membership", [&] {
            membership = std::make_shared<const ExecutionRankMembership>(*inventory, ranks);
            // Fixed discovery-size payload avoids an extra extent protocol and
            // authenticates exact order/absence without lossy hashes.
            published.assign(static_cast<size_t>(actual_size), -1);
            if (actual_rank == 0) std::copy(ranks.begin(), ranks.end(), published.begin());
            if (const auto selected_rank = membership->executionRank(actual_rank))
                active = std::shared_ptr<MPIContext>(new MPIContext(discovery, membership, *selected_rank));
            return true;
        });
        if (MPI_Bcast(published.data(), actual_size, MPI_INT, 0, comm) != MPI_SUCCESS)
            throw std::runtime_error("Cannot publish execution rank membership");
        phase("authenticate_execution_membership", [&] {
            for (int index = 0; index < actual_size; ++index)
                if (published[index] != (index < static_cast<int>(ranks.size()) ? ranks[index] : -1)) return false;
            return true;
        });
        MPI_Comm selected = MPI_COMM_NULL;
        const int result = MPI_Comm_split(comm, active ? 0 : MPI_UNDEFINED,
                                         active ? active->rank() : actual_rank, &selected);
        // Bind ownership before the final vote: all successful local handles
        // retire even if a peer reports a split failure.
        if (active && selected != MPI_COMM_NULL)
        {
            active->comm_ = selected;
            active->communicator_ownership_ = MPIContext::CommunicatorOwnership::Owned;
        }
        phase("bind_execution_communicator", [&] {
            return result == MPI_SUCCESS && (active ? selected != MPI_COMM_NULL : selected == MPI_COMM_NULL);
        });
        if (active) return active;
        return InactiveMPIRank{actual_rank};
    }
}
