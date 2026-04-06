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
#include "backends/UPIBackend.h"
#include "config/OrchestrationConfig.h" // For CollectiveBackendType
#include <memory>
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
         * @return Unique pointer to GlobalTPContext, or nullptr on error
         */
        static std::unique_ptr<GlobalTPContext> createWithSplit(
            MPI_Comm base_comm,
            int domain_id,
            int color,
            int key);

        /**
         * @brief Create for testing (uses existing communicator directly)
         *
         * @param comm Pre-created domain communicator
         * @param domain_id Domain identifier
         * @param world_ranks Vector of world ranks in this domain
         * @return Unique pointer to GlobalTPContext
         */
        static std::unique_ptr<GlobalTPContext> createForTest(
            MPI_Comm comm,
            int domain_id,
            std::vector<int> world_ranks);

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
        // scope() is provided by IGlobalTPContext (returns TPScope::GLOBAL)
        CollectiveBackendType backend() const override { return backend_type_; }
        bool allreduce(TensorBase *tensor) override;
        bool allreduce(TensorBase *tensor, const std::string &stage_name, size_t count = 0) override;
        bool broadcast(TensorBase *tensor, int source_index) override;
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override;

        // =========================================================================
        // IGlobalTPContext Implementation
        // =========================================================================

        MPI_Comm communicator() const override;
        int domainId() const override;
        const std::vector<int> &worldRanks() const override;
        GlobalDeviceAddress localDevice() const override;
        void barrier() const override;
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

    private:
        // Private constructor - use factory methods
        GlobalTPContext(
            MPI_Comm domain_comm,
            int domain_id,
            int my_rank_in_domain,
            int domain_size,
            std::vector<int> world_ranks,
            bool owns_communicator,
            CollectiveBackendType backend_type = CollectiveBackendType::UPI);

        MPI_Comm domain_comm_;                          ///< Domain-specific MPI communicator
        int domain_id_;                                 ///< Domain identifier
        int my_rank_in_domain_;                         ///< Our rank within domain (0 to size-1)
        int domain_size_;                               ///< Number of participants
        std::vector<int> world_ranks_;                  ///< World ranks of all domain members
        bool owns_communicator_;                        ///< Whether we created the communicator
        CollectiveBackendType backend_type_;            ///< Backend type for this context
        std::unique_ptr<UPICollectiveBackend> backend_; ///< Backend for collective operations
    };

} // namespace llaminar2
