/**
 * @file BufferArena.h
 * @brief Single source of truth for device-local activation buffer management
 *
 * BufferArena owns all activation/scratch/workspace buffers and provides:
 *   - Typed registration with BufferId keys
 *   - External buffer registration (for weights)
 *   - Coherence tracking (host vs. device authoritative)
 *   - Borrow tracking with aliasing validation (debug builds)
 *   - Typed BufferView handles for safe, access-controlled access
 *
 * The GraphExecutor is the sole runtime consumer — stages interact
 * with buffers only through StageBoundBuffers, never directly.
 */

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "BufferId.h"
#include "BufferAccess.h"
#include "CoherenceTracker.h"
#include "StageBoundBuffers.h"
#include "backends/DeviceId.h"
#include "execution/debug/BufferRole.h"

namespace llaminar2
{

    // Forward declarations
    class ITensor;
    class TensorBase;
    class TensorFactory;

    /**
     * @brief Configuration for BufferArena allocation behavior.
     *
     * Activation tensors are always ordinary host/device tensors. GPU-visible
     * host mappings belong to explicit TransferEngine control/data-plane
     * regions; allowing a diagnostic flag to change activation placement would
     * make graph capture identity and replay performance depend on the caller.
     */
    struct ArenaConfig
    {
        /// TensorFactory for NUMA-aware allocation (required for allocate())
        TensorFactory *factory = nullptr;

    };

    /**
     * @brief Allocation statistics tracked by BufferArena.
     */
    struct ArenaAllocationStats
    {
        size_t total_buffers = 0;
        size_t total_bytes = 0;
        void reset()
        {
            total_buffers = total_bytes = 0;
        }
    };

    /**
     * @brief Central manager for all activation / scratch / workspace buffers.
     *
     * Lifecycle:
     *   1. Construction
     *   2. registerBuffer() / registerExternalBuffer() for every buffer
     *   3. registerAlias() for scratch buffers that share storage
     *   4. allocate() — allocates all host memory
     *   5. Runtime: prepareForRead / prepareForWrite / markWritten per-stage
     *   6. Destruction: all owned tensors freed
     */
    class BufferArena
    {
    public:
        BufferArena() = default;
        explicit BufferArena(const ArenaConfig &config) : config_(config) {}
        ~BufferArena() = default;

        // Non-copyable, movable
        BufferArena(const BufferArena &) = delete;
        BufferArena &operator=(const BufferArena &) = delete;
        BufferArena(BufferArena &&) = default;
        BufferArena &operator=(BufferArena &&) = default;

        /**
         * @brief Set or update arena configuration.
         *
         * Must be called before allocate() if not passed via constructor.
         */
        void setConfig(const ArenaConfig &config) { config_ = config; }

        /**
         * @brief Get allocation statistics.
         */
        const ArenaAllocationStats &stats() const { return stats_; }

        // =====================================================================
        // Registration (called during graph setup, before allocate())
        // =====================================================================

        /**
         * @brief Register a buffer to be arena-owned.
         *
         * The arena will create a TensorBase of the requested shape/dtype
         * at allocate() time.
         *
         * @param id        Unique buffer identifier
         * @param rows      Number of rows
         * @param cols      Number of columns
         * @param dtype     Data type string ("FP32", "BF16", "Q8_1", etc.)
         * @param device    Home device for the buffer
         * @return true on success, false if id already registered
         */
        bool registerBuffer(BufferId id, size_t rows, size_t cols,
                            const char *dtype, DeviceId device);

        /**
         * @brief Register a graph-declared buffer without discarding tensor axes.
         *
         * BufferArena stores owned tensors as a two-dimensional matrix because
         * the kernel-facing tensor classes expose rows and columns. Graph
         * schemas, however, may declare tensors with any rank. This overload is
         * the required bridge between those representations: axis zero remains
         * the row dimension and every trailing axis is flattened into columns.
         * Consequently, a schema shape of `[R, A, B]` is allocated as
         * `[R, A * B]` while preserving exactly the same element capacity.
         *
         * Keeping this conversion inside BufferArena prevents graph callers
         * from accidentally copying only `shape[0]` and `shape[1]`, which would
         * silently under-allocate rank-three publication tensors. Empty shapes,
         * zero dimensions, and products that cannot fit in `size_t` are invalid
         * graph contracts and fail immediately before any allocation occurs.
         *
         * @param id Unique buffer identifier.
         * @param descriptor Complete graph buffer descriptor, including shape,
         *        tensor type, and target device.
         * @return true on success, false when @p id was already registered.
         * @throws std::invalid_argument if the descriptor has no dimensions or
         *         contains a zero-sized dimension.
         * @throws std::overflow_error if its flattened element count overflows
         *         `size_t`.
         */
        bool registerBuffer(BufferId id, const BufferDescriptor &descriptor);

        /**
         * @brief Register an externally-owned buffer (weights).
         *
         * The arena does NOT own this tensor — it just tracks its coherence
         * and provides borrow tracking.
         *
         * @param id      Unique buffer identifier
         * @param tensor  Externally-owned tensor (lifetime must outlive arena)
         * @return true on success
         */
        bool registerExternalBuffer(BufferId id, ITensor *tensor);

        /**
         * @brief Bind or replace an externally-owned scratch/output buffer.
         *
         * Unlike registerExternalBuffer(), this may be called at runtime for
         * dynamic-shape activations such as all-position verifier logits. The
         * buffer slot must not be arena-owned. Rebinding resets coherence to
         * UNINITIALIZED so the next stage write establishes authority through
         * the normal prepareForWrite/markWritten path.
         *
         * @param id      Unique buffer identifier
         * @param tensor  Externally-owned tensor (lifetime must outlive use)
         * @return true on success
         */
        bool bindExternalBuffer(BufferId id, ITensor *tensor);

        /**
         * @brief Declare that two buffers may share storage (aliasing).
         *
         * The arena will validate at runtime (debug builds) that aliased
         * buffers are never write-borrowed simultaneously.
         *
         * @param a  First buffer
         * @param b  Second buffer (shares storage with a)
         */
        void registerAlias(BufferId a, BufferId b);

        // =====================================================================
        // Static helpers for mapping names/types to BufferId
        // =====================================================================

        /**
         * @brief Map a buffer name string to its BufferId.
         *
         * Used by orchestrators to map resolved buffer descriptors (which use
         * string names from the schema) to BufferId enum values.
         *
         * @param name Buffer name (e.g., "Q", "attn_proj", "current_hidden")
         * @return Corresponding BufferId, or BufferId::_COUNT if unknown
         */
        static BufferId bufferNameToId(const std::string &name);

        /**
         * @brief Convert BufferTensorType enum to dtype string.
         *
         * @param type Tensor type enum
         * @return Dtype string (e.g., "FP32", "BF16", "Q8_1")
         */
        static const char *bufferTensorTypeToStr(BufferTensorType type);

        /**
         * @brief Allocate all registered arena-owned buffers.
         *
         * Creates TensorBase objects for all registered buffers.
         * Must be called after all registerBuffer() calls and before
         * any prepareForRead/Write calls.
         *
         * @return true on success
         */
        bool allocate();

        /**
         * @brief Log the allocation summary and address-sorted memory map.
         *
         * The compact size summary remains TRACE-only. DEBUG logging adds an
         * address-sorted map with half-open ranges, gaps or overlaps, device,
         * ownership, shape, and alias group. The address map makes physical
         * neighbors visible when diagnosing an out-of-bounds device writer.
         */
        void logAllocationSummary() const;

        /**
         * @brief Format the current arena allocation map in address order.
         *
         * GPU pointers are preferred whenever a tensor owns device storage;
         * otherwise the host allocation is reported. Address spaces are
         * grouped by device before sorting because equal virtual addresses on
         * different GPUs are unrelated. Unbound registrations are retained at
         * the end of their device group so a missing allocation is visible.
         *
         * @return Multi-line diagnostic map, or an empty string before the
         *         arena has been allocated.
         */
        std::string allocationAddressMap() const;

        /**
         * @brief Establish storage for an arena buffer during graph construction.
         *
         * This allocation-only operation intentionally has no stream parameter:
         * it cannot publish a write or establish device authority. Runtime
         * producers must instead call prepareForWrite() with their exact stream.
         *
         * @param id Buffer whose stable GPU address is required.
         * @param target GPU that will own the allocation.
         * @return true after storage exists on @p target.
         */
        bool allocateDeviceStorage(BufferId id, DeviceId target);

        // =====================================================================
        // Runtime coherence (called per-stage by GraphExecutor)
        // =====================================================================

        /**
         * @brief Ensure buffer data is available on target device for reading.
         *
         * Transfers data if needed (H2D, D2H, D2D).
         *
         * @return true on success
         */
        bool prepareForRead(BufferId id, DeviceId target, void *stream = nullptr);

        /**
         * @brief Ensure buffer has allocated storage on target for writing.
         *
         * Does NOT transfer data — the kernel will overwrite it.
         *
         * @return true on success
         */
        bool prepareForWrite(BufferId id, DeviceId target, void *stream);

        /**
         * @brief Mark buffer as written on the given device.
         *
         * Updates coherence state: the specified device now holds
         * the authoritative copy. Records a GPU completion event
         * on the tensor for fine-grained sync.
         *
         * @param id      Buffer that was written
         * @param device  Device that now holds authoritative data
         * @param stream  Exact non-null GPU stream where the write was
         *                enqueued. May be omitted only for CPU writes.
         */
        void markWritten(BufferId id, DeviceId device, void *stream = nullptr);

        /**
         * @brief Lightweight mark for graph replay (Phase 3).
         *
         * Updates coherence state and tensor flags without recording
         * a per-tensor GPU completion event. Use only when the graph executor
         * publishes the replay's graph-level completion event before any
         * external consumer can observe the tensor.
         *
         * @param id      Buffer that was written
         * @param device  Device that now holds authoritative data
         */
        void markWrittenFlagsOnly(BufferId id, DeviceId device);

        // =====================================================================
        // Borrow tracking (checked in debug builds)
        // =====================================================================

        /**
         * @brief Acquire a read borrow on a buffer.
         *
         * Multiple simultaneous read borrows are allowed.
         * Debug builds assert no write borrow is active on aliased buffers.
         */
        void acquireReadBorrow(BufferId id);

        /**
         * @brief Acquire a write borrow on a buffer.
         *
         * Debug builds assert no read or write borrows are active on this
         * buffer or any aliased buffer.
         */
        void acquireWriteBorrow(BufferId id);

        /**
         * @brief Release a read borrow.
         */
        void releaseReadBorrow(BufferId id);

        /**
         * @brief Release a write borrow.
         */
        void releaseWriteBorrow(BufferId id);

        /**
         * @brief Validate that no borrows are active (call between graph executions).
         */
        bool validateNoBorrowsActive() const;

        // =====================================================================
        // Buffer access (for building StageBoundBuffers)
        // =====================================================================

        /**
         * @brief Get the underlying tensor for a buffer.
         *
         * Used by GraphExecutor to build StageBoundBuffers entries.
         *
         * @return ITensor pointer, or nullptr if not registered
         */
        ITensor *getTensor(BufferId id) const;

        /**
         * @brief Get shared ownership of an arena-owned tensor.
         *
         * Returns the shared_ptr for arena-owned buffers, enabling callers
         * (e.g., InferenceState) to hold a reference beyond the arena's
         * lifetime. Returns nullptr for external (non-owned) buffers.
         *
         * @param id Buffer to query
         * @return shared_ptr<TensorBase>, or nullptr if not arena-owned
         */
        std::shared_ptr<TensorBase> getSharedTensor(BufferId id) const;

        /**
         * @brief Get the device pointer for a buffer (GPU or host).
         *
         * For GPU buffers, returns the GPU data pointer.
         * For CPU buffers, returns the host data pointer.
         *
         * @param id     Buffer to query
         * @param target Target device (determines which pointer to return)
         * @return Raw pointer on the target device
         */
        void *getDevicePtr(BufferId id, DeviceId target) const;

        /**
         * @brief Get rows for a registered buffer.
         */
        size_t getRows(BufferId id) const;

        /**
         * @brief Get cols for a registered buffer.
         */
        size_t getCols(BufferId id) const;

        // =====================================================================
        // Introspection
        // =====================================================================

        /**
         * @brief Get the current coherence state of a buffer.
         */
        CoherenceState getCoherenceState(BufferId id) const;

        /**
         * @brief Check if a buffer is registered.
         */
        bool isRegistered(BufferId id) const;

        /**
         * @brief Number of registered buffers.
         */
        size_t registeredCount() const;

        /**
         * @brief Check if allocate() has been called.
         */
        bool isAllocated() const { return allocated_; }

        /**
         * @brief Iterate over all registered BufferIds.
         *
         * Calls fn(BufferId) for every buffer that has been registered
         * (arena-owned or external). Enables auto-discovery of model-specific
         * buffers without hardcoding in orchestrator infrastructure.
         *
         * @param fn  Callable accepting a single BufferId argument
         */
        template <typename Fn>
        void forEachRegistered(Fn &&fn) const
        {
            for (size_t i = 0; i < kBufferCount; ++i)
            {
                if (buffers_[i].registered)
                {
                    fn(static_cast<BufferId>(i));
                }
            }
        }

    private:
        /// Internal state for each managed buffer slot
        struct ManagedBuffer
        {
            bool registered = false;

            // Ownership
            std::shared_ptr<TensorBase> owned_tensor; ///< Arena-owned
            ITensor *external_tensor = nullptr;       ///< Externally owned

            // Registration info (for deferred allocation)
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = nullptr;
            DeviceId home_device;

            // Coherence
            CoherenceState coherence;

            // Borrow tracking (debug builds enforce, release builds just count)
            int active_read_borrows = 0;
            bool active_write_borrow = false;

            // Aliasing group (-1 = not aliased)
            int alias_group = -1;

            /// Get the ITensor regardless of ownership
            ITensor *tensor() const;

            /// Get the TensorBase (for coherence ops); may be null for external
            TensorBase *tensorBase() const;
        };

        static constexpr size_t kBufferCount = static_cast<size_t>(BufferId::_COUNT);

        std::array<ManagedBuffer, kBufferCount> buffers_{};
        bool allocated_ = false;

        // Aliasing groups: each group is a set of BufferIds that share storage
        int next_alias_group_ = 0;

        // Configuration and stats
        ArenaConfig config_;
        ArenaAllocationStats stats_;

        /// Get mutable buffer slot (asserts registered)
        ManagedBuffer &buf(BufferId id);
        const ManagedBuffer &buf(BufferId id) const;

        /// Validate borrow safety for the given access mode
        void validateBorrowSafe(BufferId id, BufferAccess access) const;

        /// Check if any buffer in the same alias group has an active write borrow
        bool aliasGroupHasWriteBorrow(int group, BufferId exclude) const;

        /// Check if any buffer in the same alias group has an active borrow
        bool aliasGroupHasAnyBorrow(int group, BufferId exclude) const;

        /// Create a tensor for a registered buffer using TensorFactory
        std::shared_ptr<TensorBase> createTensorForBuffer(const ManagedBuffer &b, BufferId id) const;
    };

} // namespace llaminar2
