/**
 * @file GlobalTPContext.h
 * @brief Implementation of GLOBAL tensor parallelism context
 *
 * Provides concrete implementation of IGlobalTPContext for managing
 * tensor parallelism across multiple MPI ranks using UPI/MPI.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "IGlobalTPContext.h"
#include "ICollectiveBackend.h"
#include "backends/UPIBackend.h"
#include "config/OrchestrationConfig.h" // For CollectiveBackendType
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <mpi.h>

namespace llaminar2
{

    // Forward declarations
    struct TPDomain;
    struct RankExecutionPlan;

    /**
     * @brief Concrete implementation of GLOBAL tensor parallelism context
     *
     * Manages cross-MPI-rank tensor parallelism for CPU-only TP domains.
     * Uses UPICollectiveBackend for efficient MPI collective operations
     * over UPI interconnect.
     *
     * Creation patterns:
     * - From TPDomain (has pre-created MPI communicator)
     * - From explicit parameters (creates communicator via MPI_Comm_split)
     *
     * Thread safety: Single thread should use this context. Multiple contexts
     * (different domains) can be used concurrently from different threads.
     */
    class GlobalTPContext : public IGlobalTPContext
    {
    public:
        /**
         * @brief Create from TPDomain
         *
         * The TPDomain already contains the domain-specific MPI communicator
         * created by TPDomainBuilder::createCPUCrossRankDomain().
         *
         * @param domain Pre-configured TP domain with valid communicator
         * @return Unique pointer to GlobalTPContext, or nullptr on error
         */
        static std::unique_ptr<GlobalTPContext> create(const TPDomain &domain);

        /**
         * @brief Create from explicit parameters
         *
         * Creates a new domain communicator via MPI_Comm_split.
         *
         * @param base_comm Base communicator (typically MPI_COMM_WORLD)
         * @param domain_id Identifier for this global TP domain
         * @param color MPI_Comm_split color (ranks with same color form domain)
         * @param key MPI_Comm_split key (determines ordering within domain)
         * @param hostfile_path Optional MPI hostfile for node detection ordering
         * @return Unique pointer to GlobalTPContext, or nullptr on error
         */
        static std::unique_ptr<GlobalTPContext> createWithSplit(
            MPI_Comm base_comm,
            int domain_id,
            int color,
            int key,
            const std::string &hostfile_path = "",
            CollectiveBackendType backend_type = CollectiveBackendType::UPI);

        /**
         * @brief Create for testing (uses existing communicator directly)
         *
         * @param comm Pre-created domain communicator
         * @param domain_id Domain identifier
         * @param world_ranks Vector of world ranks in this domain
         * @param node_ids Optional per-rank node IDs. If empty, auto-detected via
         *                 hostname gathering over the domain communicator. Ranks with
         *                 the same node_id are on the same physical node.
         * @return Unique pointer to GlobalTPContext
         */
        static std::unique_ptr<GlobalTPContext> createForTest(
            MPI_Comm comm,
            int domain_id,
            std::vector<int> world_ranks,
            std::vector<int> node_ids = {},
            CollectiveBackendType backend_type = CollectiveBackendType::UPI);

        ~GlobalTPContext() override;

        // Disable copy (owns MPI communicator potentially)
        GlobalTPContext(const GlobalTPContext &) = delete;
        GlobalTPContext &operator=(const GlobalTPContext &) = delete;

        // Enable move
        GlobalTPContext(GlobalTPContext &&other) noexcept;
        GlobalTPContext &operator=(GlobalTPContext &&other) noexcept;

        // =========================================================================
        // ITPContext Implementation
        // =========================================================================

        int degree() const override;
        int myIndex() const override;

        /**
         * @brief Dynamic scope based on node placement
         *
         * Returns NODE_LOCAL if all participating ranks are on the same physical
         * node, GLOBAL otherwise. This replaces the hardcoded GLOBAL from
         * IGlobalTPContext, enabling the same GlobalTPContext class to serve
         * both NodeLocalTP (cross-socket, same machine) and true GlobalTP
         * (cross-machine) use cases.
         */
        TPScope scope() const override;

        CollectiveBackendType backend() const override { return backend_type_; }
        bool allreduce(TensorBase *tensor) override;
        bool allreduce(TensorBase *tensor, const std::string &stage_name, size_t count = 0) override;
        bool broadcast(TensorBase *tensor, int source_index) override;
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override;
        void requestAbort() override;
        bool isAbortRequested() const override { return abort_requested_.load(std::memory_order_acquire); }

        // =========================================================================
        // IGlobalTPContext Implementation
        // =========================================================================

        MPI_Comm communicator() const override;
        int domainId() const override;
        const std::vector<int> &worldRanks() const override;
        GlobalDeviceAddress localDevice() const override;
        void barrier() const override;
        bool allgatherBytes(const void *send_data, void *recv_data, size_t byte_count) const override;
        bool send(const TensorBase *tensor, int dest_index) override;
        bool recv(TensorBase *tensor, int source_index) override;

        // =========================================================================
        // Additional Methods
        // =========================================================================

        /**
         * @brief Check if context is valid (has valid communicator)
         * @return true if communicator is not MPI_COMM_NULL
         */
        bool isValid() const;

        // =========================================================================
        // Node Awareness
        // =========================================================================

        /**
         * @brief Check if all domain ranks are on the same physical node
         *
         * When true, this GlobalTPContext is conceptually a NodeLocalTP context:
         * all participants share the same machine and communicate via UPI or
         * shared memory rather than cross-node networking.
         *
         * @return true if all ranks have the same node_id
         */
        bool isAllRanksOnSameNode() const;

        /**
         * @brief Get the node ID for a specific domain rank
         *
         * @param domain_index Index within this TP domain (0 to degree()-1)
         * @return Node ID, or -1 if domain_index is out of range
         */
        int nodeId(int domain_index) const;

        /**
         * @brief Get node IDs for all ranks in this domain
         *
         * Each element corresponds to the physical node of the rank at that
         * domain index. Ranks with the same node_id are co-located.
         *
         * @return Vector of node IDs (size == degree())
         */
        const std::vector<int> &nodeIds() const;

        /**
         * @brief Get the number of distinct physical nodes in this domain
         * @return Number of unique node IDs (1 if all same node)
         */
        int nodeCount() const;

        std::string collectiveBackendNameForDiagnostics() const;

    private:
        // Private constructor - use factory methods
        GlobalTPContext(
            MPI_Comm domain_comm,
            int domain_id,
            int my_rank_in_domain,
            int domain_size,
            std::vector<int> world_ranks,
            bool owns_communicator,
            CollectiveBackendType backend_type = CollectiveBackendType::UPI,
            std::vector<int> node_ids = {});

        /// Auto-detect node IDs by gathering hostnames over domain_comm_
        void detectNodeIds();

        MPI_Comm domain_comm_;                        ///< Domain-specific MPI communicator
        int domain_id_;                               ///< Domain identifier
        int my_rank_in_domain_;                       ///< Our rank within domain (0 to size-1)
        int domain_size_;                             ///< Number of participants
        std::vector<int> world_ranks_;                ///< World ranks of all domain members
        std::vector<int> node_ids_;                   ///< Per-rank node ID (same ID = same physical node)
        bool all_same_node_;                          ///< Cached: all ranks on same node?
        int node_count_;                              ///< Cached: number of distinct nodes
        bool owns_communicator_;                      ///< Whether we created the communicator
        CollectiveBackendType backend_type_;          ///< Backend type for this context
        std::unique_ptr<ICollectiveBackend> backend_; ///< Backend for collective operations (ShmemSpin or UPI)
        std::atomic<bool> abort_requested_{false};    ///< One-sided failure/cancel flag
    };

} // namespace llaminar2
