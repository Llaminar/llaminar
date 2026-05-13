/**
 * @file ShmemSpinBackend.h
 * @brief Shared-memory spin-wait collective backend for N-rank intra-node CPU TP
 *
 * Replaces MPI_Allreduce with a purpose-built spin-wait protocol for the
 * common case: N MPI ranks on the same node doing FLOAT32/FP16/BF16 SUM
 * allreduce of small vectors (≤8192 elements / 32KB per rank).
 *
 * Protocol:
 *   1. Each rank copies its data to a shared-memory staging buffer
 *   2. Signals "ready" via atomic epoch counter (store-release)
 *   3. Spins on all peers' epoch counters (load-acquire + _mm_pause)
 *   4. AVX-512 reduces all N buffers into caller's output
 *
 * For operations outside the fast path (non-SUM, count > MAX,
 * allgather, broadcast, etc.), delegates to a wrapped UPICollectiveBackend.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "UPIBackend.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Shared-memory layout for N-rank spin-wait allreduce
     *
     * Dynamically sized based on rank count. Mapped via POSIX shared memory.
     * Layout (all cache-line aligned):
     *   [0, 64)                          → Header (num_ranks)
     *   [64, 64 + N*64)                  → N × EpochSlot (one per rank)
     *   [64 + N*64, 64 + N*64 + N*32KB)  → N × float[MAX_COUNT] buffers
     *
     * Access epoch slots and buffers via epoch_at(rank) / buffer_at(rank).
     */
    struct alignas(64) ShmemSpinArena
    {
        static constexpr size_t CACHE_LINE = 64;
        static constexpr size_t MAX_COUNT = 8192; // Max elements per allreduce (32KB FP32, 16KB FP16/BF16)

        /// Per-rank epoch counters (each on its own cache line)
        struct alignas(CACHE_LINE) EpochSlot
        {
            std::atomic<uint64_t> epoch;
            char pad_[CACHE_LINE - sizeof(std::atomic<uint64_t>)];
        };

        // Header (occupies first cache line)
        int32_t num_ranks;

        // Variable-length data follows — use epoch_at() / buffer_at()

        EpochSlot *epoch_at(int rank)
        {
            auto *base = reinterpret_cast<char *>(this) + sizeof(ShmemSpinArena);
            return reinterpret_cast<EpochSlot *>(base) + rank;
        }
        const EpochSlot *epoch_at(int rank) const
        {
            auto *base = reinterpret_cast<const char *>(this) + sizeof(ShmemSpinArena);
            return reinterpret_cast<const EpochSlot *>(base) + rank;
        }

        float *buffer_at(int rank)
        {
            auto *base = reinterpret_cast<char *>(this) + sizeof(ShmemSpinArena)
                         + static_cast<size_t>(num_ranks) * sizeof(EpochSlot);
            return reinterpret_cast<float *>(base) + static_cast<size_t>(rank) * MAX_COUNT;
        }
        const float *buffer_at(int rank) const
        {
            auto *base = reinterpret_cast<const char *>(this) + sizeof(ShmemSpinArena)
                         + static_cast<size_t>(num_ranks) * sizeof(EpochSlot);
            return reinterpret_cast<const float *>(base) + static_cast<size_t>(rank) * MAX_COUNT;
        }

        /// Total arena size in bytes for a given rank count
        static size_t compute_size(int num_ranks)
        {
            return sizeof(ShmemSpinArena) // header (one cache line)
                   + static_cast<size_t>(num_ranks) * sizeof(EpochSlot)
                   + static_cast<size_t>(num_ranks) * MAX_COUNT * sizeof(float);
        }
    };

    static_assert(sizeof(ShmemSpinArena) == 64, "ShmemSpinArena header must be one cache line");

    /**
     * @brief Shared-memory spin-wait collective backend for N-rank intra-node allreduce
     *
     * Fast path: FLOAT32/FP16/BF16 SUM allreduce with count ≤ MAX_COUNT
     * Fallback:  Delegates to UPICollectiveBackend (MPI) for everything else
     *
     * Thread Safety:
     * - Single backend instance should be used from one thread per rank
     * - N instances (one per rank) coordinate via shared memory
     */
    class ShmemSpinBackend : public ICollectiveBackend
    {
    public:
        /**
         * @brief Create shared-memory spin-wait backend
         *
         * @param domain_id  Unique domain identifier (included in generated shm names)
         * @param my_rank    This rank's index (0 to N-1)
         * @param fallback   UPI backend for non-fast-path operations (takes ownership)
         */
        ShmemSpinBackend(int domain_id, int my_rank,
                         std::unique_ptr<UPICollectiveBackend> fallback);

        ~ShmemSpinBackend() override;

        // Non-copyable, non-movable (owns shared memory mapping)
        ShmemSpinBackend(const ShmemSpinBackend &) = delete;
        ShmemSpinBackend &operator=(const ShmemSpinBackend &) = delete;
        ShmemSpinBackend(ShmemSpinBackend &&) = delete;
        ShmemSpinBackend &operator=(ShmemSpinBackend &&) = delete;

        // =====================================================================
        // Identity
        // =====================================================================

        CollectiveBackendType type() const override { return CollectiveBackendType::UPI; }
        std::string name() const override { return "ShmemSpin"; }

        // =====================================================================
        // Capability Queries
        // =====================================================================

        bool supportsDevice(DeviceType type) const override;
        bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override;
        bool isAvailable() const override;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        bool initialize(const DeviceGroup &group) override;
        bool isInitialized() const override;
        void shutdown() override;
        void abort() override;

        // =====================================================================
        // Collective Operations
        // =====================================================================

        bool allreduce(void *buffer, size_t count,
                       CollectiveDataType dtype, CollectiveOp op) override;

        bool allgather(const void *send_buf, void *recv_buf,
                       size_t send_count, CollectiveDataType dtype) override;

        bool allgatherv(const void *send_buf, size_t send_count,
                        void *recv_buf,
                        const std::vector<int> &recv_counts,
                        const std::vector<int> &displacements,
                        CollectiveDataType dtype) override;

        bool reduceScatter(const void *send_buf, void *recv_buf,
                           size_t recv_count, CollectiveDataType dtype,
                           CollectiveOp op) override;

        bool broadcast(void *buffer, size_t count,
                       CollectiveDataType dtype, int root_rank) override;

        bool synchronize() override;

        // =====================================================================
        // Diagnostics
        // =====================================================================

        std::string lastError() const override { return last_error_; }

        // =====================================================================
        // Accessors (for testing)
        // =====================================================================

        /// Get the shared-memory arena pointer (nullptr if not initialized)
        ShmemSpinArena *arena() const { return arena_; }

        /// Get this rank's epoch counter value
        uint64_t currentEpoch() const { return my_epoch_; }

        /// Get the POSIX shared-memory name used by this initialized backend.
        const std::string &shmName() const { return shm_name_; }

        /// Check if a given allreduce would use the fast path
        bool isFastPath(size_t count, CollectiveDataType dtype, CollectiveOp op) const;

        // =================================================================
        // Vectorized reduction — public static (pure functions, no state)
        // NOTE: out may alias a (for in-place N-way accumulation).
        // =================================================================

        /// Runtime ISA-dispatched sum: out[i] = a[i] + b[i]
        static void reduce(float *out, const float *a, const float *b,
                           size_t count);

        /// ISA-specific implementations
        static void reduce_scalar(float *out, const float *a, const float *b,
                                  size_t count);
        static void reduce_avx2(float *out, const float *a, const float *b,
                                size_t count);
        static void reduce_avx512(float *out, const float *a, const float *b,
                                  size_t count);

        /// FP16 reduce: convert to FP32, add, convert back
        static void reduce_fp16(uint16_t *out, const uint16_t *a,
                                const uint16_t *b, size_t count);
        static void reduce_fp16_scalar(uint16_t *out, const uint16_t *a,
                                       const uint16_t *b, size_t count);
        static void reduce_fp16_avx2(uint16_t *out, const uint16_t *a,
                                     const uint16_t *b, size_t count);
        static void reduce_fp16_avx512(uint16_t *out, const uint16_t *a,
                                       const uint16_t *b, size_t count);

        /// BF16 reduce: bit-shift to FP32, add, truncate back
        static void reduce_bf16(uint16_t *out, const uint16_t *a,
                                const uint16_t *b, size_t count);
        static void reduce_bf16_scalar(uint16_t *out, const uint16_t *a,
                                       const uint16_t *b, size_t count);
        static void reduce_bf16_avx2(uint16_t *out, const uint16_t *a,
                                     const uint16_t *b, size_t count);
        static void reduce_bf16_avx512(uint16_t *out, const uint16_t *a,
                                       const uint16_t *b, size_t count);

    private:
        /// Create or open the POSIX shared memory segment
        bool setupSharedMemory();

        /// Unmap and optionally unlink the shared memory segment
        void teardownSharedMemory();

        /// Wait for a peer epoch in the shared-memory protocol, with abort/timeout handling.
        bool waitForPeerEpoch(int peer_rank, uint64_t target_epoch, const char *phase, size_t count = 0);

        int domain_id_;
        int my_rank_;
        int num_ranks_ = 0;                  ///< Total ranks (set during initialize)
        uint64_t my_epoch_ = 0;

        std::string shm_name_;               ///< POSIX shm name generated per initialize() call
        int shm_fd_ = -1;                    ///< File descriptor for shared memory
        size_t arena_size_ = 0;              ///< Mapped size in bytes (for munmap)
        ShmemSpinArena *arena_ = nullptr;     ///< Mapped shared-memory arena
        bool shm_unlinked_ = false;           ///< Whether rank 0 has unlinked shm_name_

        std::unique_ptr<UPICollectiveBackend> fallback_; ///< MPI fallback for non-fast-path ops
        std::atomic<bool> abort_requested_{false};       ///< Local abort requested for this backend
        bool initialized_ = false;
        mutable std::string last_error_;
    };

} // namespace llaminar2
