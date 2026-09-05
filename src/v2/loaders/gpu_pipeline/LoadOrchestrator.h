#pragma once

/**
 * @file LoadOrchestrator.h
 * @brief Coordinates GPU weight planning, staging, pipeline loading, and finalization.
 *
 * LoadOrchestrator owns one WeightVRAMPool per target device plus pinned host rings
 * for upload overlap. Prepared GEMM kernels retain the orchestrator as a lifetime
 * owner so persistent pool allocations outlive model execution; finalize() releases
 * only temporary staging resources after all queued weight jobs have completed.
 */

#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    class IBackend;

    /**
     * @brief Explicit opt-out used only by isolated allocator/kernel tests.
     *
     * Production code must provide @ref PhysicalMemoryAuthority. Naming the
     * test-only path in the type system keeps a missing admission dependency
     * from silently becoming a late free-memory preflight.
     */
    struct TestOnlyUnadmittedGPUAllocation final
    {
        explicit constexpr TestOnlyUnadmittedGPUAllocation() = default;
    };

    /** Public spelling required at deliberate test call sites. */
    inline constexpr TestOnlyUnadmittedGPUAllocation
        kTestOnlyUnadmittedGPUAllocation{};

    /**
     * @brief Coordinates admitted GPU weight planning and asynchronous upload.
     *
     * One instance owns a persistent prepared-weight allocation and, while
     * loading, paired device/pinned-host staging rings. Production allocation
     * is legal only through the rank-bound physical-memory authority supplied
     * at construction. The corresponding live claims remain beside the owned
     * allocations for exactly their physical lifetimes.
     */
    class LoadOrchestrator
    {
    public:
        /** @brief Construct a planning-only instance that cannot allocate GPU memory. */
        LoadOrchestrator() = default;

        /**
         * @brief Construct a production allocator bound to one persistent owner.
         * @param backend Exact CUDA or ROCm backend; must outlive this object.
         * @param memory_authority Sole admitted rank-local allocation authority.
         * @param persistent_owner Owner charged for the prepared weight region.
         * @throws std::invalid_argument for null or invalid configuration.
         */
        LoadOrchestrator(
            IBackend *backend,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            PhysicalMemoryOwner persistent_owner);

        /**
         * @brief Construct an explicitly unadmitted allocator for tests only.
         * @param backend Injected test/real backend, or null for device-free
         *        logical slot-pool tests that never dereference device bytes.
         * @param test_only Required opt-out token.
         */
        LoadOrchestrator(
            IBackend *backend,
            TestOnlyUnadmittedGPUAllocation test_only);

        struct DeviceContext
        {
            int device_id = -1;
            std::unique_ptr<WeightVRAMPool> pool;
            std::unique_ptr<PinnedRingBuffer> pinned_ring;
            std::vector<WeightJob> pending_jobs;
            /** Exact per-owner byte plan for the one persistent GPU allocation. */
            std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
                persistent_owner_bytes{};
            /** Unrounded terminal offset, stable before and after materialization. */
            std::size_t persistent_planned_bytes = 0u;
            /** Owner receiving final allocation-alignment padding. */
            PhysicalMemoryOwner last_persistent_owner =
                PhysicalMemoryOwner::Count;
            /** Live claims paired with the persistent prepared-weight allocation. */
            std::vector<PhysicalMemoryAllocationLease> weight_leases;
            /** Live claim paired with temporary device upload staging. */
            std::optional<PhysicalMemoryAllocationLease> device_staging_lease;
            /** Live claim paired with temporary pinned-host upload staging. */
            std::optional<PhysicalMemoryAllocationLease> host_staging_lease;
        };

        ~LoadOrchestrator();

        /// Add a device to manage.
        void addDevice(int device_id);

        /// Plan a weight for a specific device.
        void planWeight(int device_id, const std::string &name,
                        int N, int K, int payload_bytes_per_block,
                        bool is_asymmetric, bool has_emins,
                        size_t raw_gguf_bytes);

        /**
         * @brief Plan one weight against an explicit physical owner line.
         *
         * This form is used by pools that co-locate live and transactional
         * expert regions in one allocation. Alignment bytes introduced before
         * the weight are charged to this same owner, so the owner totals sum to
         * the allocator's exact physical byte count.
         */
        void planWeightForOwner(
            int device_id,
            const std::string &name,
            int N,
            int K,
            int payload_bytes_per_block,
            bool is_asymmetric,
            bool has_emins,
            size_t raw_gguf_bytes,
            PhysicalMemoryOwner owner);

        /// Plan a raw (floating-point) weight for a specific device. No repack needed.
        void planRawWeight(int device_id, const std::string &name, int N, int K, size_t raw_bytes);

        /** @brief Plan one floating-point weight against an explicit owner. */
        void planRawWeightForOwner(
            int device_id,
            const std::string &name,
            int N,
            int K,
            size_t raw_bytes,
            PhysicalMemoryOwner owner);

        /** @return Exact pre-allocation bytes charged to one persistent owner. */
        [[nodiscard]] size_t plannedPersistentBytes(
            int device_id,
            PhysicalMemoryOwner owner) const;

        /// Allocate all device pools. Throws on failure.
        /// @param pinned_slot_size  Max raw GGUF weight size (bytes per pinned slot).
        /// @param num_h2d_streams   Number of ring buffer slots.
        void allocate(size_t pinned_slot_size, int num_h2d_streams = 3);

        /// Get pool for a device. Returns nullptr if not found.
        WeightVRAMPool *getPool(int device_id);
        const WeightVRAMPool *getPool(int device_id) const;

        /// Number of managed devices.
        size_t numDevices() const;

        /// Add a weight job to be loaded on a specific device.
        /// Jobs larger than the allocated staging slot are split at complete
        /// source-row boundaries. GPU repack kernels publish each row chunk into
        /// the correct full-N packed coordinates.
        void addWeightJob(int device_id, const WeightJob &job);

        /// Total raw bytes of pending jobs for a specific device.
        size_t totalPendingBytes(int device_id) const;

        /// Number of physical (possibly row-chunked) jobs pending for a device.
        size_t pendingJobCount(int device_id) const;

        /// Execute all pending weight jobs with pipelined H2D + GPU repack.
        /// Each device is processed via a DeviceLoadPipeline. Throws on failure.
        /// @param progress_cb  Optional per-job progress callback (bytes_loaded, total_bytes)
        void load(DeviceLoadPipeline::ProgressCallback progress_cb = nullptr);

        /// Release temporary staging regions after loading completes.
        void finalize();

        /// Release all resources.
        void release();

    private:
        /** @brief Construction mode controlling whether allocation is admitted. */
        enum class AllocationAuthorityKind
        {
            PlanningOnly,
            Production,
            ExplicitTest,
        };

        DeviceContext *findDevice(int device_id);
        const DeviceContext *findDevice(int device_id) const;

        /// Create backend-appropriate RepackKernels function pointer struct.
        RepackKernels createRepackKernels() const;

        /** @brief Record the exact pool-offset growth introduced by one weight. */
        void recordPersistentOwnerGrowth(
            DeviceContext &ctx,
            PhysicalMemoryOwner owner,
            size_t bytes_before,
            size_t bytes_after);

        /**
         * @brief Resolve the complete owner split for the eventual allocation.
         *
         * WeightVRAMPool aligns each internal region while weights are planned,
         * then rounds the terminal pool extent once more at allocation time.
         * This method is the sole attribution point for that final tail: both
         * admission-facing queries and materialization consume its result.
         *
         * @param ctx Device whose immutable weight plan is being resolved.
         * @return Per-owner bytes summing to the exact persistent allocation.
         * @throws std::logic_error if production bytes have no valid owner.
         * @throws std::overflow_error if final attribution would wrap.
         */
        [[nodiscard]] std::array<
            std::size_t,
            PhysicalMemoryBOM::ownerCount()>
        resolvedPersistentOwnerBytes(const DeviceContext &ctx) const;

        /**
         * @brief Retire every allocation and claim materialized by allocate().
         *
         * The helper deliberately preserves device and weight plans so a
         * caller can inspect or retry setup after an allocation transaction
         * fails. Physical storage is retired before its matching ledger lease.
         */
        void rollbackMaterializedAllocations() noexcept;

        IBackend *backend_ = nullptr;
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority_;
        PhysicalMemoryOwner persistent_owner_ = PhysicalMemoryOwner::Count;
        AllocationAuthorityKind allocation_authority_kind_ =
            AllocationAuthorityKind::PlanningOnly;
        std::vector<DeviceContext> devices_;
    };

} // namespace llaminar2
