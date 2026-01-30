/**
 * @file BackendRouter.h
 * @brief Routes collective operations to appropriate backends
 *
 * INTERNAL IMPLEMENTATION - Not exposed to model graphs!
 *
 * The BackendRouter is used internally by CollectiveContext to select
 * the optimal backend for collective operations. Model graphs never
 * interact with this class directly.
 *
 * The BackendRouter is the central coordinator for collective communication.
 * It selects the optimal backend based on:
 * - Device group composition (CUDA-only → NCCL, ROCm-only → RCCL, etc.)
 * - Operation scope (local vs global)
 * - Backend availability (NCCL requires HAVE_NCCL, etc.)
 *
 * Selection Logic:
 * 1. All CUDA GPUs, local scope → NCCL
 * 2. All ROCm GPUs, local scope → RCCL
 * 3. Global scope (cross-rank) → MPI
 * 4. Heterogeneous (mixed types) → Host (or multi-phase)
 * 5. Fallback → Host
 *
 * TESTABILITY:
 * - IBackendRouter interface allows mocking in tests
 * - IBackendFactory interface allows injecting mock backends
 * - All dependencies are injected via constructor
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "ICollectiveBackend.h"
#include "DeviceGroup.h"
#include "../execution/mpi_orchestration/DeviceInventory.h"
#include "../utils/MPIContext.h"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <functional>

namespace llaminar2
{

    // Forward declarations for domain-aware routing
    struct TPDomain;
    class UPICollectiveBackend;

    /**
     * @brief Backend selection result
     */
    struct BackendSelection
    {
        CollectiveBackendType type = CollectiveBackendType::HOST;
        std::string reason; // Why this backend was selected

        /// For heterogeneous groups: multi-phase execution plan
        bool requires_multi_phase = false;
        std::vector<CollectiveBackendType> phase_backends;
    };

    // =========================================================================
    // IBackendFactory - For dependency injection and testing
    // =========================================================================

    /**
     * @brief Factory interface for creating collective backends
     *
     * Allows injection of mock backends for testing.
     */
    class IBackendFactory
    {
    public:
        virtual ~IBackendFactory() = default;

        /// Create a backend of the specified type
        virtual std::unique_ptr<ICollectiveBackend> createBackend(
            CollectiveBackendType type,
            std::shared_ptr<MPIContext> mpi_ctx) = 0;

        /// Check if a backend type is available (library compiled in)
        virtual bool isAvailable(CollectiveBackendType type) const = 0;
    };

    // =========================================================================
    // IBackendRouter - Interface for mocking
    // =========================================================================

    /**
     * @brief Abstract interface for backend routing
     *
     * Allows CollectiveContext to use a mock router in tests.
     */
    class IBackendRouter
    {
    public:
        virtual ~IBackendRouter() = default;

        /// Get optimal backend for a device group
        virtual ICollectiveBackend *getBackend(const DeviceGroup &group) = 0;

        /// Get backend by explicit type
        virtual ICollectiveBackend *getBackend(CollectiveBackendType type) = 0;

        /// Get backend selection info without acquiring backend
        virtual BackendSelection selectBackend(const DeviceGroup &group) const = 0;

        /// Check if a backend type is available
        virtual bool isAvailable(CollectiveBackendType type) const = 0;

        /// Execute heterogeneous AllReduce
        virtual bool executeHeterogeneousAllReduce(
            const DeviceGroup &group,
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) = 0;

        /// Execute heterogeneous AllGather
        virtual bool executeHeterogeneousAllGather(
            const DeviceGroup &group,
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype) = 0;

        /// Select backend for a specific TP domain
        virtual ICollectiveBackend *selectBackendForDomain(const TPDomain *domain) = 0;

        /// Check if domain-aware routing is available
        virtual bool hasDomainSupport() const = 0;
    };

    // =========================================================================
    // BackendRouter - Concrete implementation
    // =========================================================================

    /**
     * @brief Routes collective operations to appropriate backends
     *
     * Owns backend instances and manages their lifecycle.
     * Provides automatic backend selection based on device group.
     *
     * Usage:
     *   BackendRouter router(mpi_ctx, cluster_inventory);
     *   auto* backend = router.getBackend(device_group);
     *   backend->allreduce(...);
     *
     * Testing:
     *   auto mock_factory = std::make_unique<MockBackendFactory>();
     *   BackendRouter router(mpi_ctx, inventory, std::move(mock_factory));
     */
    class BackendRouter : public IBackendRouter
    {
    public:
        /**
         * @brief Construct router with MPI context and cluster topology
         *
         * @param mpi_ctx MPI context (may be nullptr for single-rank)
         * @param cluster_inventory Full cluster device inventory
         * @param factory Backend factory (nullptr = use default production factory)
         */
        BackendRouter(
            std::shared_ptr<MPIContext> mpi_ctx,
            const ClusterInventory &cluster_inventory,
            std::unique_ptr<IBackendFactory> factory = nullptr);

        ~BackendRouter();

        // =====================================================================
        // Backend Access
        // =====================================================================

        /**
         * @brief Get optimal backend for a device group
         *
         * Automatically selects and initializes the best backend.
         * Returns cached backend if group was seen before.
         *
         * @param group Device group for collective
         * @return Backend pointer (owned by router), or nullptr on error
         */
        ICollectiveBackend *getBackend(const DeviceGroup &group);

        /**
         * @brief Get backend by explicit type
         *
         * Returns nullptr if type is not available or not initialized.
         *
         * @param type Specific backend type
         * @return Backend pointer, or nullptr if unavailable
         */
        ICollectiveBackend *getBackend(CollectiveBackendType type);

        /**
         * @brief Get backend selection info without acquiring backend
         */
        BackendSelection selectBackend(const DeviceGroup &group) const;

        // =====================================================================
        // Backend Management
        // =====================================================================

        /**
         * @brief Initialize a backend for a device group
         *
         * Called automatically by getBackend(), but can be called
         * explicitly for pre-initialization.
         *
         * @param type Backend type
         * @param group Device group
         * @return true on success
         */
        bool initializeBackend(CollectiveBackendType type, const DeviceGroup &group);

        /**
         * @brief Check if a backend type is available
         */
        bool isAvailable(CollectiveBackendType type) const;

        /**
         * @brief Get list of available backend types
         */
        std::vector<CollectiveBackendType> availableBackends() const;

        /**
         * @brief Shutdown all backends
         */
        void shutdown();

        // =====================================================================
        // Heterogeneous Support
        // =====================================================================

        /**
         * @brief Execute multi-phase AllReduce for heterogeneous groups
         *
         * When a group has mixed device types, we need multiple phases:
         * 1. AllReduce within same-type subgroups (NCCL for CUDAs, etc.)
         * 2. AllReduce across subgroup representatives via Host
         * 3. Broadcast results back to subgroup members
         *
         * @param group Heterogeneous device group
         * @param buffer In-place buffer (must be accessible from all devices)
         * @param count Element count
         * @param dtype Data type
         * @param op Reduction operation
         * @return true on success
         */
        bool executeHeterogeneousAllReduce(
            const DeviceGroup &group,
            void *buffer,
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op);

        /**
         * @brief Execute multi-phase AllGather for heterogeneous groups
         */
        bool executeHeterogeneousAllGather(
            const DeviceGroup &group,
            const void *send_buf,
            void *recv_buf,
            size_t send_count,
            CollectiveDataType dtype);

        // =====================================================================
        // Diagnostics
        // =====================================================================

        /**
         * @brief Get diagnostic string for current backend state
         */
        std::string diagnostics() const;

        // =====================================================================
        // Domain-Aware Backend Selection
        // =====================================================================

        /**
         * @brief Select backend for a specific TP domain
         *
         * Domain-aware routing logic:
         * - GPU_INTRA_RANK domains:
         *   - Heterogeneous GPUs (CUDA+ROCm) → PCIeBAR backend (~25μs latency)
         *   - All CUDA → NCCL backend
         *   - All ROCm → RCCL backend
         * - CPU_CROSS_RANK domains → UPI backend (MPI over UPI ~50 GB/s)
         *
         * Falls back to MPI backend if preferred backend unavailable.
         *
         * @param domain The tensor parallel domain (may be nullptr)
         * @return Backend appropriate for the domain, or nullptr if unavailable
         */
        ICollectiveBackend *selectBackendForDomain(const TPDomain *domain);

        /**
         * @brief Register UPI backend for CPU cross-rank domains
         *
         * The UPI backend uses a domain-specific MPI communicator for
         * cross-socket CPU tensor parallelism over UPI/QPI/Infinity Fabric.
         *
         * @param backend UPI backend instance (ownership transferred)
         */
        void registerUPIBackend(std::unique_ptr<UPICollectiveBackend> backend);

        /**
         * @brief Check if domain-aware routing is available
         *
         * Domain support requires a UPI backend to be registered.
         *
         * @return true if UPI backend is registered
         */
        bool hasDomainSupport() const;

    private:
        std::shared_ptr<MPIContext> mpi_ctx_;
        ClusterInventory cluster_inventory_;
        std::unique_ptr<IBackendFactory> factory_;

        // Backend instances (lazily created via factory)
        std::unique_ptr<ICollectiveBackend> mpi_backend_;
        std::unique_ptr<ICollectiveBackend> nccl_backend_;
        std::unique_ptr<ICollectiveBackend> rccl_backend_;
        std::unique_ptr<ICollectiveBackend> pcie_bar_backend_;
        std::unique_ptr<ICollectiveBackend> host_backend_;

        // Group name → initialized backend mapping
        std::unordered_map<std::string, ICollectiveBackend *> group_backend_cache_;

        // UPI backend for CPU cross-rank domains (optional)
        std::unique_ptr<UPICollectiveBackend> upi_backend_;

        // Internal helpers
        ICollectiveBackend *getOrCreateBackend(CollectiveBackendType type);
        CollectiveBackendType selectBackendType(const DeviceGroup &group) const;
        std::string makeGroupKey(const DeviceGroup &group) const;

        // Domain-aware selection helpers
        bool hasHeterogeneousGPUs(const std::vector<DeviceId> &devices) const;
        bool allCUDA(const std::vector<DeviceId> &devices) const;
        bool allROCm(const std::vector<DeviceId> &devices) const;
    };

    // =========================================================================
    // DefaultBackendFactory - Production implementation
    // =========================================================================

    /**
     * @brief Default factory that creates real backend implementations
     *
     * Used in production. Tests can inject MockBackendFactory instead.
     */
    class DefaultBackendFactory : public IBackendFactory
    {
    public:
        std::unique_ptr<ICollectiveBackend> createBackend(
            CollectiveBackendType type,
            std::shared_ptr<MPIContext> mpi_ctx) override;

        bool isAvailable(CollectiveBackendType type) const override;
    };

    // =========================================================================
    // GlobalBackendRouter - Optional singleton (avoid in new code)
    // =========================================================================

    /**
     * @brief Global singleton router (optional convenience)
     *
     * For applications that want a single shared router.
     * Must be initialized before use via initGlobalRouter().
     *
     * NOTE: Prefer dependency injection over this singleton for testability.
     */
    class GlobalBackendRouter
    {
    public:
        static void init(
            std::shared_ptr<MPIContext> mpi_ctx,
            const ClusterInventory &cluster_inventory);

        static BackendRouter *get();

        static void shutdown();

    private:
        static std::unique_ptr<BackendRouter> instance_;
    };

} // namespace llaminar2
