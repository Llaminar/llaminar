/**
 * @file HierarchicalCommBuilder.cpp
 * @brief Implementation of MPI communicator hierarchy construction
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "HierarchicalCommBuilder.h"
#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // CommHierarchy
    // =========================================================================

    CommHierarchy::~CommHierarchy()
    {
        freeAll();
    }

    CommHierarchy::CommHierarchy(CommHierarchy &&other) noexcept
        : comms_(std::move(other.comms_)),
          root_path_(std::move(other.root_path_)),
          world_comm_(other.world_comm_)
    {
        // Clear other's map so it doesn't free anything
        other.comms_.clear();
        other.world_comm_ = MPI_COMM_NULL;
    }

    CommHierarchy &CommHierarchy::operator=(CommHierarchy &&other) noexcept
    {
        if (this != &other)
        {
            // Free existing communicators
            freeAll();

            // Move from other
            comms_ = std::move(other.comms_);
            root_path_ = std::move(other.root_path_);
            world_comm_ = other.world_comm_;

            // Clear other
            other.comms_.clear();
            other.world_comm_ = MPI_COMM_NULL;
        }
        return *this;
    }

    void CommHierarchy::addCommunicator(const std::string &path, MPI_Comm comm, int depth)
    {
        if (comm == MPI_COMM_NULL)
        {
            return; // Don't add null communicators
        }

        CommEntry entry;
        entry.comm = comm;
        entry.depth = depth;
        comms_[path] = entry;

        // Track the shallowest path as root
        if (root_path_.empty() || depth < comms_[root_path_].depth)
        {
            root_path_ = path;
        }
    }

    MPI_Comm CommHierarchy::getCommunicator(const std::string &path) const
    {
        auto it = comms_.find(path);
        if (it != comms_.end())
        {
            return it->second.comm;
        }
        return MPI_COMM_NULL;
    }

    MPI_Comm CommHierarchy::rootCommunicator() const
    {
        // If we have a root path with a communicator, use it
        if (!root_path_.empty())
        {
            auto it = comms_.find(root_path_);
            if (it != comms_.end() && it->second.comm != MPI_COMM_NULL)
            {
                return it->second.comm;
            }
        }
        // Otherwise return world
        return world_comm_;
    }

    std::vector<std::string> CommHierarchy::allPaths() const
    {
        std::vector<std::string> paths;
        paths.reserve(comms_.size());
        for (const auto &kv : comms_)
        {
            paths.push_back(kv.first);
        }
        return paths;
    }

    std::string CommHierarchy::toString() const
    {
        std::ostringstream ss;
        ss << "CommHierarchy (" << comms_.size() << " communicators):\n";

        // Sort by depth for readable output
        std::vector<std::pair<std::string, CommEntry>> entries(comms_.begin(), comms_.end());
        std::sort(entries.begin(), entries.end(),
                  [](const auto &a, const auto &b)
                  { return a.second.depth < b.second.depth; });

        for (const auto &kv : entries)
        {
            std::string indent(kv.second.depth * 2, ' ');
            ss << indent << "- " << kv.first << " (depth=" << kv.second.depth
               << ", comm=" << kv.second.comm << ")\n";
        }

        if (!root_path_.empty())
        {
            ss << "Root path: " << root_path_ << "\n";
        }

        return ss.str();
    }

    void CommHierarchy::freeAll()
    {
#ifdef HAVE_MPI
        if (comms_.empty())
        {
            return;
        }

        // Guard against calls after MPI_Finalize (static destruction ordering)
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (mpi_finalized)
        {
            comms_.clear();
            root_path_.clear();
            return;
        }

        // Sort by depth DESCENDING so we free children before parents
        std::vector<std::pair<std::string, CommEntry>> entries(comms_.begin(), comms_.end());
        std::sort(entries.begin(), entries.end(),
                  [](const auto &a, const auto &b)
                  { return a.second.depth > b.second.depth; });

        for (auto &kv : entries)
        {
            MPI_Comm comm = kv.second.comm;
            // Don't free world or null communicators
            if (comm != MPI_COMM_WORLD && comm != MPI_COMM_NULL)
            {
                MPI_Comm_free(&comm);
            }
        }
#endif
        comms_.clear();
        root_path_.clear();
    }

    // =========================================================================
    // HierarchicalCommBuilder
    // =========================================================================

    CommHierarchy HierarchicalCommBuilder::build(const ParallelismTree &tree,
                                                 MPI_Comm world_comm,
                                                 int my_rank)
    {
        CommHierarchy hierarchy;
        hierarchy.world_comm_ = world_comm;

        // Start recursion from root with empty base path
        const std::string root_path = tree.root.name.empty() ? "root" : tree.root.name;
        buildRecursive(tree.root, root_path, world_comm, my_rank, 0, hierarchy);

        return hierarchy;
    }

    void HierarchicalCommBuilder::buildRecursive(const ParallelismNode &node,
                                                 const std::string &path,
                                                 MPI_Comm parent_comm,
                                                 int my_rank,
                                                 int depth,
                                                 CommHierarchy &hierarchy)
    {
        // Step 1: DEVICE leaves don't need communicators
        if (node.isLeaf())
        {
            return;
        }

        // Step 2: Determine if this node needs a communicator
        MPI_Comm node_comm = parent_comm;

        if (node.type == ParallelismNodeType::TENSOR_PARALLEL && node.isCrossRank())
        {
            // TP node spanning multiple ranks needs a split communicator
            int color = calculateTPColor(node, path, my_rank);

#ifdef HAVE_MPI
            // Only actually split when MPI is available
            MPI_Comm new_comm = MPI_COMM_NULL;
            int split_result = MPI_Comm_split(parent_comm, color, my_rank, &new_comm);

            if (split_result == MPI_SUCCESS && new_comm != MPI_COMM_NULL)
            {
                hierarchy.addCommunicator(path, new_comm, depth);
                node_comm = new_comm;
            }
#else
            // Without MPI, we track what would be created for testing
            // Use color as a mock "communicator"
            if (color >= 0)
            {
                // Use path hash as mock comm value
                MPI_Comm mock_comm = pathToColor(path);
                hierarchy.addCommunicator(path, mock_comm, depth);
                node_comm = mock_comm;
            }
#endif
        }
        // PP nodes don't need their own communicator - they use P2P

        // Step 3: Recurse into children
        for (const auto &child : node.children)
        {
            std::string child_path = path + "/" + child.name;
            buildRecursive(child, child_path, node_comm, my_rank, depth + 1, hierarchy);
        }
    }

    int HierarchicalCommBuilder::calculateTPColor(const ParallelismNode &node,
                                                  const std::string &path,
                                                  int rank)
    {
        // Get ranks in this TP domain
        std::set<int> domain_ranks = node.leafRanks();

        // If this rank is not in the TP domain, return MPI_UNDEFINED
        if (domain_ranks.find(rank) == domain_ranks.end())
        {
            return MPI_UNDEFINED;
        }

        // Rank is in the domain - return a consistent color for this TP group
        // The color must be the same for all ranks in the domain
        return pathToColor(path);
    }

    bool HierarchicalCommBuilder::needsCommunicator(const ParallelismNode &node)
    {
        // Only TP nodes that cross ranks need communicators
        if (node.type != ParallelismNodeType::TENSOR_PARALLEL)
        {
            return false;
        }

        // TP node must span multiple ranks
        return node.isCrossRank();
    }

    int HierarchicalCommBuilder::pathToColor(const std::string &path)
    {
        // Use std::hash and reduce to positive integer
        std::hash<std::string> hasher;
        size_t hash = hasher(path);

        // Reduce to positive int (avoid overflow issues)
        // Use a large prime to spread values
        constexpr int LARGE_PRIME = 2147483647; // 2^31 - 1
        int color = static_cast<int>(hash % static_cast<size_t>(LARGE_PRIME));
        if (color < 0)
        {
            color = -color;
        }

        return color;
    }

} // namespace llaminar2
