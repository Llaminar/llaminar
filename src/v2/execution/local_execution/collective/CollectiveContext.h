/**
 * @file CollectiveContext.h
 * @brief Runtime context for collective operations
 *
 * This is the bridge between abstract collective stages (AllreduceStage,
 * AllGatherStage) and concrete backend implementations (MPI, NCCL, RCCL, Host).
 *
 * Architecture:
 * - Model graphs (Qwen2Graph) create abstract collective stages
 * - DeviceGraphExecutor holds a CollectiveContext
 * - When executing collective stages, DeviceGraphExecutor delegates to CollectiveContext
 * - CollectiveContext selects and uses the appropriate backend
 *
 * This separation ensures:
 * - Model graphs stay pure (no backend knowledge)
 * - Backend selection is centralized
 * - Easy to add new backends without changing model code
 *
 * TESTABILITY:
 * - IBackendRouter interface allows injecting mock routers
 * - Config struct allows testing various configurations
 * - No global state - all dependencies injected
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../../interfaces/ICollectiveContext.h"
#include "../../../collective/ICollectiveBackend.h"
#include "../../../collective/BackendRouter.h"
#include "../../../config/TPDomain.h"
#include "../../../backends/DeviceId.h"
#include "../../../utils/MPIContext.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ITensor;
    class ClusterInventory;

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /**
     * @brief Convert tensor native type to collective data type for MPI/NCCL/RCCL
     *
     * @param tensor The tensor to query (must not be nullptr)
     * @return CollectiveDataType matching the tensor's storage format
     * @throws std::invalid_argument if tensor is nullptr
     * @throws std::invalid_argument if tensor type is not supported (e.g., quantized types)
     */
    CollectiveDataType tensorToCollectiveDataType(const ITensor *tensor);

    /**
     * @brief Runtime context for collective operations
     *
     * Implements ICollectiveContext interface.
     * Injected into DeviceGraphExecutor at construction time.
     * Encapsulates ALL collective backend knowledge - model graphs never see this.
     *
     * Usage in DeviceGraphExecutor:
     *
     *   // At construction
     *   DeviceGraphExecutor executor(graph, collective_ctx);
     *
     *   // At execution
     *   if (stage->type() == ComputeStageType::ALLREDUCE) {
     *       collective_ctx_->executeAllreduce(stage->buffer(), stage->count(), device);
     *   }
     */
    class CollectiveContext : public ICollectiveContext
    {
    public:
        /**
         * @brief Configuration for CollectiveContext
         */
        struct Config
        {
            /// MPI context (nullptr for single-rank, no MPI)
            std::shared_ptr<IMPIContext> mpi_ctx = nullptr;

            /// Cluster inventory for device discovery
            const ClusterInventory *cluster_inventory = nullptr;

            /// Force a specific backend (AUTO = let router decide)
            CollectiveBackendType forced_backend = CollectiveBackendType::AUTO;

            /// Enable verbose logging of collective operations
            bool verbose = false;
        };

        /**
         * @brief Construct collective context from config
         *
         * Creates a real BackendRouter internally.
         */
        explicit CollectiveContext(const Config &config);

        /**
         * @brief Construct with injected router (for testing)
         *
         * Allows injection of mock router for unit testing.
         *
         * @param router Pre-configured router (takes ownership)
         * @param mpi_ctx MPI context (may be nullptr)
         * @param local_devices Devices on this rank
         */
        CollectiveContext(
            std::unique_ptr<IBackendRouter> router,
            std::shared_ptr<IMPIContext> mpi_ctx,
            std::vector<DeviceId> local_devices);

        ~CollectiveContext() override;

        // =====================================================================
        // High-Level Collective Operations (ICollectiveContext)
        // These are called by DeviceGraphExecutor when executing collective stages
        // =====================================================================

        /**
         * @brief Execute an AllReduce operation
         *
         * Automatically selects the best backend based on tensor location.
         *
         * @param buffer In-place buffer to reduce
         * @param count Number of elements (0 = use buffer->numel())
         * @param tensor_device Device where tensor resides
         * @param op Reduction operation (default: SUM)
         * @return true on success
         */
        bool executeAllreduce(
            ITensor *buffer,
            size_t count,
            DeviceId tensor_device,
            CollectiveOp op = CollectiveOp::ALLREDUCE_SUM) override;

        /**
         * @brief Execute an AllGather operation
         *
         * @param local_input Local slice [seq_len, local_dim]
         * @param full_output Full output [seq_len, full_dim]
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @return true on success
         */
        bool executeAllgather(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device) override;

        /**
         * @brief Execute a strided AllGather for column-parallel operations
         *
         * Optimized for column-parallel LM head where output has interleaved
         * columns from all ranks. Uses NCCL + GPU deinterleave kernel to avoid
         * host memory staging.
         *
         * Input:  [seq_len, local_dim] on each rank
         * Output: [seq_len, local_dim * world_size] with strided layout
         *
         * @param local_input Local slice [seq_len, local_dim]
         * @param full_output Full output [seq_len, full_dim]
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside (must be CUDA)
         * @return true on success, false if not CUDA or backend not available
         */
        bool executeStridedAllgather(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device);

        /**
         * @brief Execute a variable-sized AllGather operation (allgatherv)
         *
         * Unlike executeAllgather which assumes equal send counts per rank,
         * this method supports variable send counts needed for heterogeneous
         * tensor parallelism (e.g., different head counts per device).
         *
         * @param local_input Local slice to send [seq_len, local_dim]
         * @param full_output Full output [seq_len, sum(recv_counts)]
         * @param recv_counts Elements per rank (size = world_size)
         * @param displacements Offset in output per rank (size = world_size)
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @return true on success
         */
        bool executeAllgatherv(
            ITensor *local_input,
            ITensor *full_output,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            size_t actual_seq_len,
            DeviceId tensor_device) override;

        /**
         * @brief Execute a Broadcast operation
         *
         * @param buffer Buffer to broadcast (in-place)
         * @param count Number of elements
         * @param root_device Device that holds the source data
         * @param tensor_device Device where buffer resides
         * @return true on success
         */
        bool executeBroadcast(
            ITensor *buffer,
            size_t count,
            int root_rank,
            DeviceId tensor_device) override;

        // =====================================================================
        // Query Methods (ICollectiveContext)
        // =====================================================================

        /// Does this configuration require collectives?
        bool requiresCollectives() const override;

        /// Get the world size (number of ranks)
        int worldSize() const override;

        /// Get the local rank
        int rank() const override;

        /// Get devices on this rank
        const std::vector<DeviceId> &localDevices() const override { return local_devices_; }

        /// Check if a specific backend is available
        bool isBackendAvailable(CollectiveBackendType type) const override;

        // =====================================================================
        // Advanced: Direct Access (ICollectiveContext)
        // =====================================================================

        /// Get the backend router interface (internal use)
        IBackendRouter *router() override { return router_.get(); }

        /// Get the MPI context (may be nullptr)
        IMPIContext *mpiContext() override { return mpi_ctx_.get(); }

        // =====================================================================
        // Domain-Aware Collective Operations
        // =====================================================================

        /**
         * @brief Execute an AllReduce operation in a specific TP domain
         *
         * Uses BackendRouter::selectBackendForDomain() for domain-aware
         * backend selection. Falls back to non-domain executeAllreduce()
         * when domain is nullptr.
         *
         * @param buffer In-place buffer to reduce
         * @param count Number of elements (0 = use buffer->numel())
         * @param tensor_device Device where tensor resides
         * @param op Reduction operation
         * @param domain TP domain to use (nullptr = fallback)
         * @return true on success
         */
        bool executeAllreduceInDomain(
            ITensor *buffer,
            size_t count,
            DeviceId tensor_device,
            CollectiveOp op,
            const TPDomain *domain) override;

        /**
         * @brief Execute an AllGather operation in a specific TP domain
         *
         * @param local_input Local slice [seq_len, local_dim]
         * @param full_output Full output [seq_len, full_dim]
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @param domain TP domain to use (nullptr = fallback)
         * @return true on success
         */
        bool executeAllgatherInDomain(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device,
            const TPDomain *domain) override;

        /**
         * @brief Execute a variable-sized AllGather operation in a specific TP domain
         *
         * @param local_input Local slice to send [seq_len, local_dim]
         * @param full_output Full output [seq_len, sum(recv_counts)]
         * @param recv_counts Elements per rank (size = world_size)
         * @param displacements Offset in output per rank (size = world_size)
         * @param actual_seq_len Actual sequence length (0 = use tensor rows)
         * @param tensor_device Device where tensors reside
         * @param domain TP domain to use (nullptr = fallback)
         * @return true on success
         */
        bool executeAllgathervInDomain(
            ITensor *local_input,
            ITensor *full_output,
            const std::vector<int> &recv_counts,
            const std::vector<int> &displacements,
            size_t actual_seq_len,
            DeviceId tensor_device,
            const TPDomain *domain) override;

        // =====================================================================
        // Buffer Registration Support (Phase 3)
        // =====================================================================

        /**
         * @brief Check if buffer registration is required for any backend
         *
         * Returns false — no backends require buffer registration.
         *
         * @return false always
         */
        bool requiresBufferRegistration() const;

    private:
        Config config_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        std::unique_ptr<IBackendRouter> router_; // Interface, not concrete!

        /// Devices on this rank
        std::vector<DeviceId> local_devices_;

        /// Cached world size
        int world_size_ = 1;

        /// Select backend for a tensor on a specific device
        ICollectiveBackend *selectBackend(DeviceId tensor_device);

        /// Build device group for collective operation
        DeviceGroup buildDeviceGroup(DeviceId tensor_device) const;
    };

    /**
     * @brief Factory for creating CollectiveContext
     *
     * Provides convenient construction patterns.
     */
    class CollectiveContextFactory
    {
    public:
        /**
         * @brief Create context for single-rank, single-device (no collectives)
         */
        static std::unique_ptr<CollectiveContext> createSingleDevice();

        /**
         * @brief Create context for MPI-based tensor parallelism (current behavior)
         */
        static std::unique_ptr<CollectiveContext> createMPI(
            std::shared_ptr<IMPIContext> mpi_ctx);

        /**
         * @brief Create context for intra-node multi-GPU (NCCL/RCCL)
         */
        static std::unique_ptr<CollectiveContext> createIntraNode(
            const ClusterInventory &inventory,
            std::shared_ptr<IMPIContext> mpi_ctx = nullptr);

        /**
         * @brief Create context with injected router (for testing)
         *
         * @param router Pre-configured router (or mock)
         * @param mpi_ctx MPI context
         * @param local_devices Devices on this rank
         */
        static std::unique_ptr<CollectiveContext> createWithRouter(
            std::unique_ptr<IBackendRouter> router,
            std::shared_ptr<IMPIContext> mpi_ctx,
            std::vector<DeviceId> local_devices);

        /**
         * @brief Create context with full configuration
         */
        static std::unique_ptr<CollectiveContext> create(
            const CollectiveContext::Config &config);
    };

} // namespace llaminar2
