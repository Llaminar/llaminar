/**
 * @file NodeDetection.cpp
 * @brief Physical MPI node identity from shared-memory communicator membership.
 *
 * Hostnames are diagnostic labels, not physical identities: localhost, short
 * names, FQDNs and container aliases must not create ghost nodes. MPI supplies
 * shared-memory groups; their lowest communicator rank gives a stable label
 * which the pure validator converts to compact node IDs on every participant.
 */
#include "NodeDetection.h"
#include <algorithm>
#include <map>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Fail a collective initialization phase on an MPI API error. */
        void requireMPI(int result, const char *operation)
        {
            if (result != MPI_SUCCESS)
                throw std::runtime_error(std::string("Physical node discovery failed: ") + operation);
        }
    }

    NodeDetectionResult NodeDetection::detect(MPI_Comm comm, const std::string &hostfile_path)
    {
        // Kept as a source-compatible launch input, never interpreted as a
        // physical topology. The MPI launcher has already consumed its slots.
        (void)hostfile_path;
        if (comm == MPI_COMM_NULL) return {};
        int size = 0, rank = 0;
        requireMPI(MPI_Comm_size(comm, &size), "communicator size");
        requireMPI(MPI_Comm_rank(comm, &rank), "communicator rank");
        MPI_Comm local = MPI_COMM_NULL;
        requireMPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local), "shared-memory split");
        int leader = rank;
        const int reduced = MPI_Allreduce(MPI_IN_PLACE, &leader, 1, MPI_INT, MPI_MIN, local);
        requireMPI(MPI_Comm_free(&local), "shared-memory communicator release");
        requireMPI(reduced, "shared-memory leader");
        std::vector<int> leaders(static_cast<std::size_t>(size));
        requireMPI(MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, comm), "shared-memory membership");

        char hostname[MPI_MAX_PROCESSOR_NAME] = {};
        int length = 0;
        requireMPI(MPI_Get_processor_name(hostname, &length), "processor label");
        std::vector<char> labels(static_cast<std::size_t>(size) * MPI_MAX_PROCESSOR_NAME);
        requireMPI(MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                                labels.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm), "processor labels");
        auto result = fromSharedNodeLeaders(leaders);
        for (int peer = 0; peer < size; ++peer)
        {
            const char *begin = labels.data() + static_cast<std::size_t>(peer) * MPI_MAX_PROCESSOR_NAME;
            result.hostnames.emplace_back(begin, std::find(begin, begin + MPI_MAX_PROCESSOR_NAME, '\0'));
        }
        return result;
    }

    NodeDetectionResult NodeDetection::fromSharedNodeLeaders(const std::vector<int> &leaders)
    {
        NodeDetectionResult result;
        std::map<int, int> nodes;
        for (std::size_t rank = 0; rank < leaders.size(); ++rank)
        {
            const int leader = leaders[rank];
            if (leader < 0 || static_cast<std::size_t>(leader) > rank ||
                leaders[static_cast<std::size_t>(leader)] != leader)
                throw std::invalid_argument("Invalid shared-memory node leader for communicator rank " + std::to_string(rank));
            const auto [node, inserted] = nodes.try_emplace(leader, static_cast<int>(nodes.size()));
            (void)inserted;
            result.node_ids.push_back(node->second);
        }
        result.node_count = static_cast<int>(nodes.size());
        return result;
    }

    NodeDetectionResult NodeDetection::fromHostnames(const std::vector<std::string> &hostnames)
    {
        // This pure helper is for synthetic topology fixtures. Real MPI callers
        // must use detect(): equal labels do not prove shared address space.
        NodeDetectionResult result;
        result.hostnames = hostnames;
        std::map<std::string, int> nodes;
        for (const auto &hostname : hostnames)
        {
            const auto [node, inserted] = nodes.try_emplace(hostname, static_cast<int>(nodes.size()));
            (void)inserted;
            result.node_ids.push_back(node->second);
        }
        result.node_count = static_cast<int>(nodes.size());
        return result;
    }
} // namespace llaminar2
