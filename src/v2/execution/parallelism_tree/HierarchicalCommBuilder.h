/**
 * @file HierarchicalCommBuilder.h
 * @brief Build MPI communicator hierarchies from the parallelism tree
 *
 * HierarchicalCommBuilder walks the ParallelismTree top-down, creating MPI
 * communicators at each TP node that spans multiple MPI ranks. The resulting
 * CommHierarchy owns all created communicators and frees them in reverse
 * order (children before parents) on destruction.
 *
 * Key concepts:
 * - TP nodes that cross ranks need dedicated communicators for allreduce
 * - TP nodes within a single rank use local collectives (no MPI comm needed)
 * - PP nodes use P2P (send/recv), not collective communicators
 * - Each communicator is tagged with a tree path for lookup
 *
 * Example: 4-rank tree with cross-rank TP
 * ```
 * PP(global)
 * ├── TP(global_tp, [rank0+rank1])  ← needs MPI comm split
 * └── TP(global_tp2, [rank2+rank3]) ← needs MPI comm split
 * ```
 *
 * Builds hierarchy with two split communicators:
 * - "global/global_tp" → ranks 0,1
 * - "global/global_tp2" → ranks 2,3
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "ParallelismTree.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

namespace llaminar2
{

    // =========================================================================
    // CommHierarchy
    // =========================================================================

    /**
     * @brief RAII wrapper for hierarchical MPI communicators
     *
     * Manages the lifetime of split communicators, freeing them in reverse order
     * (children before parents) to prevent MPI errors.
     *
     * Each communicator is tagged with the path in the tree (e.g., "global/host0/socket0_tp").
     */
    class CommHierarchy
    {
    public:
        CommHierarchy() = default;
        ~CommHierarchy();

        // Move-only
        CommHierarchy(CommHierarchy &&other) noexcept;
        CommHierarchy &operator=(CommHierarchy &&other) noexcept;
        CommHierarchy(const CommHierarchy &) = delete;
        CommHierarchy &operator=(const CommHierarchy &) = delete;

        /**
         * @brief Add a communicator to the hierarchy
         *
         * @param path Tree path (e.g., "global/host0/socket0_tp")
         * @param comm MPI communicator
         * @param depth Nesting depth (for ordering destruction)
         */
        void addCommunicator(const std::string &path, MPI_Comm comm, int depth);

        /**
         * @brief Get communicator for a tree path
         *
         * @param path Tree path
         * @return MPI_Comm or MPI_COMM_NULL if not found
         */
        MPI_Comm getCommunicator(const std::string &path) const;

        /**
         * @brief Get the root communicator (typically world)
         */
        MPI_Comm rootCommunicator() const;

        /**
         * @brief Get all communicator paths
         */
        std::vector<std::string> allPaths() const;

        /**
         * @brief Number of communicators in hierarchy
         */
        size_t size() const { return comms_.size(); }

        /**
         * @brief Check if hierarchy is empty
         */
        bool empty() const { return comms_.empty(); }

        /**
         * @brief Human-readable dump
         */
        std::string toString() const;

    private:
        friend class HierarchicalCommBuilder;

        struct CommEntry
        {
            MPI_Comm comm = MPI_COMM_NULL;
            int depth = 0;
        };

        std::map<std::string, CommEntry> comms_;
        std::string root_path_;
        MPI_Comm world_comm_ = MPI_COMM_WORLD; ///< The world communicator (not freed)

        /**
         * @brief Free all communicators in depth-descending order
         *
         * Called by destructor and move-assignment operator.
         */
        void freeAll();
    };

    // =========================================================================
    // HierarchicalCommBuilder
    // =========================================================================

    /**
     * @brief Build MPI communicator hierarchy from parallelism tree
     *
     * Walks the tree top-down, splitting communicators at each TP node
     * that spans multiple MPI ranks.
     *
     * - TP nodes: Split by domain_id (ranks in same TP group get same color)
     * - PP nodes: Children inherit parent comm (use P2P, not collectives)
     */
    class HierarchicalCommBuilder
    {
    public:
        /**
         * @brief Build communicator hierarchy from tree
         *
         * @param tree The parallelism tree (with layers assigned)
         * @param world_comm The root communicator (typically MPI_COMM_WORLD)
         * @param my_rank This rank's MPI rank
         * @return CommHierarchy with all necessary split communicators
         */
        static CommHierarchy build(const ParallelismTree &tree,
                                   MPI_Comm world_comm,
                                   int my_rank);

        /**
         * @brief Calculate the color for a rank in a TP domain
         *
         * Ranks within the same TP domain get the same color.
         * Ranks outside the domain get MPI_UNDEFINED (-1).
         *
         * The color is computed as the hash of the TP node's path within
         * the tree, modulo a large prime to avoid collision.
         *
         * @param node TP node to analyze
         * @param path Path to this node in the tree
         * @param rank MPI rank
         * @return Color for MPI_Comm_split (-1 if undefined)
         */
        static int calculateTPColor(const ParallelismNode &node,
                                    const std::string &path,
                                    int rank);

        /**
         * @brief Determine if a node needs its own communicator
         *
         * A node needs a comm if:
         * - It's a TP node that crosses ranks (isCrossRank())
         * - It's used for collectives
         *
         * @param node Node to check
         * @return true if node needs a dedicated communicator
         */
        static bool needsCommunicator(const ParallelismNode &node);

    private:
        /**
         * @brief Recursive helper to build communicators
         *
         * @param node Current node in the tree
         * @param path Path to this node (e.g., "global/host0/socket0_tp")
         * @param parent_comm Parent communicator for splitting
         * @param my_rank This rank's MPI rank
         * @param depth Current nesting depth
         * @param hierarchy Output hierarchy being built
         */
        static void buildRecursive(const ParallelismNode &node,
                                   const std::string &path,
                                   MPI_Comm parent_comm,
                                   int my_rank,
                                   int depth,
                                   CommHierarchy &hierarchy);

        /**
         * @brief Compute a hash-based color for a path
         *
         * Uses std::hash<std::string> and reduces to a positive integer.
         *
         * @param path Tree path
         * @return Positive color value
         */
        static int pathToColor(const std::string &path);
    };

} // namespace llaminar2
