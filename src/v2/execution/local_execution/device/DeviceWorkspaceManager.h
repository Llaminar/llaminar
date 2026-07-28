/**
 * @file DeviceWorkspaceManager.h
 * @brief Per-device workspace buffer management
 *
 * DeviceWorkspaceManager allocates workspace buffers within a memory budget
 * and provides named buffer access for kernels.
 *
 * Works with any device type: CPU, CUDA, ROCm.
 * (Formerly GpuWorkspaceManager.h)
 *
 * Design:
 * - One manager per device (DeviceId)
 * - Allocates contiguous block up front
 * - Suballocates named buffers from the block
 * - No reallocation during inference (hot path is zero-alloc)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "WorkspaceDescriptor.h"
#include "../../../backends/DeviceId.h"
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Exact identity for one immutable workspace-backed publication.
     *
     * The workspace manager deliberately does not interpret these words. A
     * producer supplies a collision-free tuple appropriate to its domain, such
     * as workspace generation, immutable device allocation, and matrix
     * geometry for a prepared router gate. Keeping the tuple structured avoids
     * lossy hash-only identities and makes accidental aliasing impossible.
     */
    struct PersistentWorkspacePublicationKey
    {
        std::uint64_t word0 = 0;
        std::uint64_t word1 = 0;
        std::uint64_t word2 = 0;
        std::uint64_t word3 = 0;

        friend bool operator==(
            const PersistentWorkspacePublicationKey &left,
            const PersistentWorkspacePublicationKey &right) noexcept = default;
    };

    struct PersistentWorkspacePublicationKeyHash
    {
        size_t operator()(
            const PersistentWorkspacePublicationKey &key) const noexcept;
    };

    /**
     * @brief Result of adopting or creating an immutable publication.
     */
    struct PersistentWorkspacePublicationResult
    {
        std::shared_ptr<void> publication;
        size_t slot = std::numeric_limits<size_t>::max();
        bool created = false;

        explicit operator bool() const noexcept
        {
            return publication != nullptr &&
                   slot != std::numeric_limits<size_t>::max();
        }
    };

    using PersistentWorkspacePublicationFactory =
        std::function<std::shared_ptr<void>(size_t slot)>;

    namespace detail
    {
        struct PersistentWorkspacePublicationRecord
        {
            size_t slot = std::numeric_limits<size_t>::max();
            std::shared_ptr<void> publication;
        };

        /**
         * @brief Shared ownership state for graph-stable workspace slot leases.
         *
         * The workspace manager owns this registry while its device allocation
         * is alive. Individual kernel leases keep only a weak reference, so a
         * workspace teardown never depends on kernels being destroyed in a
         * particular order.
         */
        struct PersistentWorkspaceSlotRegistry
        {
            std::mutex mutex;
            std::unordered_map<std::string, std::vector<bool>> occupied_slots;
            std::unordered_map<
                std::string,
                std::unordered_map<
                    PersistentWorkspacePublicationKey,
                    PersistentWorkspacePublicationRecord,
                    PersistentWorkspacePublicationKeyHash>>
                immutable_publications;
        };
    } // namespace detail

    /**
     * @brief RAII ownership token for one persistent workspace metadata slot.
     *
     * Request scratch buffers may be reused by sequential graph stages because
     * each producer runs immediately before its consumer. Graph metadata such
     * as weight descriptor tables and expert masks is different: every captured
     * graph records its address and expects the bytes at that address to remain
     * unchanged for the graph lifetime. This token gives one kernel exclusive
     * ownership of one such slot and returns it automatically when the kernel
     * releases or rebinds its workspace.
     */
    class PersistentWorkspaceSlotLease
    {
    public:
        ~PersistentWorkspaceSlotLease();

        PersistentWorkspaceSlotLease(const PersistentWorkspaceSlotLease &) = delete;
        PersistentWorkspaceSlotLease &operator=(const PersistentWorkspaceSlotLease &) = delete;
        PersistentWorkspaceSlotLease(PersistentWorkspaceSlotLease &&) = delete;
        PersistentWorkspaceSlotLease &operator=(PersistentWorkspaceSlotLease &&) = delete;

        /**
         * @brief Return the zero-based slot exclusively owned by this lease.
         */
        size_t slot() const noexcept { return slot_; }

    private:
        friend class DeviceWorkspaceManager;

        PersistentWorkspaceSlotLease(
            std::weak_ptr<detail::PersistentWorkspaceSlotRegistry> registry,
            std::string domain,
            size_t slot)
            : registry_(std::move(registry)),
              domain_(std::move(domain)),
              slot_(slot)
        {
        }

        std::weak_ptr<detail::PersistentWorkspaceSlotRegistry> registry_;
        std::string domain_;
        size_t slot_ = std::numeric_limits<size_t>::max();
    };

    /**
     * @brief Manages pre-allocated workspace buffers for a single device
     *
     * DeviceWorkspaceManager provides centralized workspace buffer management for
     * kernels. Instead of each kernel allocating its own workspace, the
     * manager pre-allocates a single contiguous block and suballocates named
     * buffers within it.
     *
     * Works with any device type: CPU, CUDA, ROCm.
     * (Formerly GpuWorkspaceManager)
     *
     * **Usage**:
     * ```cpp
     * DeviceWorkspaceManager mgr(DeviceId::cuda(0), 256 * 1024 * 1024);  // 256MB budget
     *
     * WorkspaceRequirements reqs;
     * reqs.buffers.push_back({"gemm_workspace", 64 * 1024 * 1024});
     * reqs.buffers.push_back({"attention_scores", 32 * 1024 * 1024});
     *
     * if (mgr.allocate(reqs)) {
     *     void* gemm_ws = mgr.getBuffer("gemm_workspace");
     *     void* attn_scores = mgr.getBuffer("attention_scores");
     *     // ... use buffers ...
     * }
     *
     * mgr.release();  // Free all memory
     * ```
     *
     * **Thread Safety**: Not thread-safe. Caller must synchronize access.
     */
    class DeviceWorkspaceManager
    {
    public:
        /**
         * @brief Construct a workspace manager for a device
         * @param device Target device (CPU, CUDA, or ROCm)
         * @param budget_bytes Maximum bytes available for workspace
         */
        DeviceWorkspaceManager(DeviceId device, size_t budget_bytes);

        ~DeviceWorkspaceManager();

        // Non-copyable, non-movable (owns device memory)
        DeviceWorkspaceManager(const DeviceWorkspaceManager &) = delete;
        DeviceWorkspaceManager &operator=(const DeviceWorkspaceManager &) = delete;
        DeviceWorkspaceManager(DeviceWorkspaceManager &&) = delete;
        DeviceWorkspaceManager &operator=(DeviceWorkspaceManager &&) = delete;

        // =========================================================================
        // Allocation
        // =========================================================================

        /**
         * @brief Allocate workspace buffers from requirements
         * @param requirements Collection of buffer descriptors
         * @return true if all required buffers allocated, false on failure
         *
         * Allocates a single contiguous block from the backend and suballocates
         * named buffers at aligned offsets within the block.
         *
         * Note: Non-required buffers that don't fit are silently skipped.
         */
        bool allocate(const WorkspaceRequirements &requirements);

        /**
         * @brief Zero the complete allocated workspace on an explicit stream.
         *
         * Persistent cache-owned arenas use this after one-time planning so
         * every suballocation begins from a deterministic state without
         * exposing or iterating implementation-specific raw GPU allocations.
         * GPU callers must provide the producer stream; CPU callers may pass
         * nullptr because host memset has no stream-ordering contract.
         *
         * @param stream CUDA/HIP producer stream, or nullptr for CPU storage.
         * @return true when the complete allocation was initialized.
         */
        bool zeroAll(void *stream);

        /**
         * @brief Check if buffers have been allocated
         */
        bool isAllocated() const { return allocated_; }

        /**
         * @brief Release all allocated buffers
         *
         * Frees the underlying memory block and clears all buffer mappings.
         * After release(), isAllocated() returns false.
         */
        void release();

        // =========================================================================
        // Buffer Access
        // =========================================================================

        /**
         * @brief Get pointer to a named buffer
         * @param name Buffer name
         * @return Pointer to buffer (nullptr if not found)
         */
        void *getBuffer(const std::string &name) const;

        /**
         * @brief Get size of a named buffer
         * @param name Buffer name
         * @return Size in bytes (0 if not found)
         */
        size_t getBufferSize(const std::string &name) const;

        /**
         * @brief Check if a named buffer exists
         * @param name Buffer name
         */
        bool hasBuffer(const std::string &name) const;

        /**
         * @brief Get all buffer names
         */
        std::vector<std::string> bufferNames() const;

        /**
         * @brief Resolve one fixed-stride address in a persistent buffer.
         *
         * Persistent owners may request different payload sizes while sharing
         * a workspace table sized for the largest supported payload. Computing
         * `slot * payload_bytes` would make those owners use different strides
         * and can overlap live graph-captured data. This method always derives
         * the stride from the complete declared buffer and common slot count.
         *
         * The caller must already hold the matching
         * `PersistentWorkspaceSlotLease`; this method resolves only the byte
         * address and validates that the requested payload fits.
         *
         * @param name Named workspace buffer split into equal persistent slots.
         * @param slot_capacity Number of slots used when the buffer was sized.
         * @param slot Zero-based slot owned by the caller's lease.
         * @param payload_bytes Number of bytes the caller will write.
         * @return Beginning of the isolated slot, or nullptr on any mismatch.
         */
        void *getPersistentSlotBuffer(
            const std::string &name,
            size_t slot_capacity,
            size_t slot,
            size_t payload_bytes) const;

        /**
         * @brief Acquire one exclusive graph-lifetime metadata slot.
         *
         * Slot domains are logical ownership namespaces. CUDA gate/up
         * descriptors, CUDA down descriptors, ROCm descriptors, and expert
         * masks use separate domains even when their slot numbers happen to be
         * equal. Acquisition occurs during graph setup or warmup only; inference
         * hot paths retain and reuse the returned token without host work.
         *
         * @param domain Stable logical namespace for the metadata table.
         * @param slot_capacity Number of equal-sized slots in that table.
         * @return An RAII lease, or nullptr when the domain is invalid or full.
         */
        std::shared_ptr<PersistentWorkspaceSlotLease> acquirePersistentSlot(
            const std::string &domain,
            size_t slot_capacity);

        /**
         * @brief Publish immutable graph metadata once per exact identity.
         *
         * Unlike @ref acquirePersistentSlot, this API represents data that is
         * byte-stable for the workspace lifetime and may be consumed by many
         * separately captured graph-local kernels. The first caller reserves a
         * slot and invokes @p factory while holding the setup-only registry
         * lock. Later callers receive the same object and slot; they cannot
         * consume additional capacity or republish the bytes.
         *
         * The factory must submit any device initialization and record its
         * readiness event before returning. It must not synchronize the device
         * or recursively acquire another persistent slot. Returning nullptr
         * rolls the reservation back, allowing a later setup attempt to retry.
         * No part of this API is called from graph replay or the inference hot
         * path: graph-local kernels retain the returned shared object.
         *
         * @param domain Stable logical namespace for one physical slot table.
         * @param key Exact immutable publication identity within @p domain.
         * @param slot_capacity Number of fixed-stride slots in the table.
         * @param factory First-publication callback receiving the reserved slot.
         * @return Shared publication, slot, and whether this call created it.
         */
        PersistentWorkspacePublicationResult
        getOrCreatePersistentPublication(
            const std::string &domain,
            const PersistentWorkspacePublicationKey &key,
            size_t slot_capacity,
            const PersistentWorkspacePublicationFactory &factory);

        // =========================================================================
        // Metrics
        // =========================================================================

        /**
         * @brief Get the device this manager is bound to
         */
        DeviceId device() const { return device_; }

        /**
         * @brief Monotonic host identity for this manager instance.
         *
         * Kernels that cache raw workspace sub-buffer pointers use this to
         * distinguish a genuinely unchanged manager from a new manager that
         * happened to be allocated at the same host address.
         */
        uint64_t id() const { return id_; }

        /**
         * @brief Get the total budget in bytes
         */
        size_t budget() const { return budget_bytes_; }

        /**
         * @brief Get the number of bytes used (including alignment padding)
         */
        size_t used() const { return used_bytes_; }

        /**
         * @brief Get the remaining budget in bytes
         */
        size_t remaining() const { return budget_bytes_ - used_bytes_; }

        /**
         * @brief Get the number of allocated buffers
         */
        size_t bufferCount() const { return buffers_.size(); }

    private:
        DeviceId device_;
        uint64_t id_;
        size_t budget_bytes_;
        size_t used_bytes_ = 0;
        bool allocated_ = false;

        // Main allocation block
        void *block_ = nullptr;
        size_t block_size_ = 0;

        // Named buffer offsets within block
        struct BufferInfo
        {
            size_t offset;
            size_t size;
        };
        std::unordered_map<std::string, BufferInfo> buffers_;

        /**
         * @brief Host-only ownership registry for immutable graph metadata.
         *
         * The registry is deliberately separate from the device allocation.
         * release() replaces it, invalidating every old weak lease without
         * requiring a callback into kernels whose destruction order may differ
         * from workspace destruction order.
         */
        std::shared_ptr<detail::PersistentWorkspaceSlotRegistry>
            persistent_slot_registry_ =
                std::make_shared<detail::PersistentWorkspaceSlotRegistry>();

        /**
         * @brief Align offset to alignment boundary
         * @param offset Current offset
         * @param alignment Required alignment (must be power of 2)
         * @return Aligned offset >= input offset
         */
        static size_t alignUp(size_t offset, size_t alignment);

        /**
         * @brief Internal helper to allocate buffers
         * @param buffers Pointers to buffer descriptors
         * @param total_size Total size to allocate
         * @return true on success
         */
        bool allocateBuffers(
            const std::vector<const WorkspaceDescriptor *> &buffers,
            size_t total_size);
    };

} // namespace llaminar2
