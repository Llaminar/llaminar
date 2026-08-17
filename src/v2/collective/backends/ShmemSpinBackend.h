/**
 * @file ShmemSpinBackend.h
 * @brief Shared-memory spin-wait collective backend for N-rank intra-node CPU TP
 *
 * Replaces MPI_Allreduce with a purpose-built spin-wait protocol for N MPI
 * ranks on the same node doing FLOAT32/FP16/BF16 SUM allreduce. Logical
 * payloads of every size use a bounded, chunked shared-memory arena.
 *
 * Protocol:
 *   1. Each rank copies its data to its shared-memory staging buffer
 *   2. Signals "ready" via atomic epoch counter (store-release)
 *   3. Spins on all peers' epoch counters (load-acquire + _mm_pause)
 *   4. AVX-512 reduces all N buffers into caller's output
 *   5. Signals "consumed" before any rank may reuse its staging buffer
 *
 * Native protocols also cover variable packed-record publication to one root
 * and broadcast from one root. Operations with different semantics, such as
 * non-SUM reductions and allgather, delegate to the wrapped UPI backend.
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
#include <vector>

namespace llaminar2
{

    /**
     * @brief Shared-memory layout for N-rank spin-wait allreduce
     *
     * Dynamically sized based on rank count. Mapped via POSIX shared memory.
     * Layout (all cache-line aligned):
     *   [0, 64)                          → Header (num_ranks)
     *   [64, 64 + N*64)                  → N × EpochSlot (one per rank)
     *   [64 + N*64, end)                   → N rank-local chunk buffers
     *
     * Access epoch slots and buffers via epoch_at(rank) / buffer_at(rank).
     */
    struct alignas(64) ShmemSpinArena
    {
        static constexpr size_t CACHE_LINE = 64;

        /**
         * @brief Maximum elements reduced during one protocol epoch.
         *
         * Larger logical payloads execute as consecutive chunks without
         * changing any output element's arithmetic order. One million FP32
         * elements occupies 4 MiB per rank and covers the Qwen3.6-35B
         * 434-by-2048 prefill payload in one epoch.
         */
        static constexpr size_t CHUNK_CAPACITY = 1u << 20;

        /// Per-rank epoch counters (each on its own cache line)
        struct alignas(CACHE_LINE) EpochSlot
        {
            std::atomic<uint64_t> epoch;
            /**
             * Number of FP32 payload elements offered by this participant for
             * the current rooted packed-record transaction. The owning rank
             * writes the value before publishing `epoch`; readers acquire the
             * epoch before consuming it.
             */
            std::atomic<uint64_t> payload_elements;
            char pad_[CACHE_LINE - 2u * sizeof(std::atomic<uint64_t>)];
        };

        // Header (occupies first cache line)
        int32_t num_ranks;

        // Variable-length data follows — use epoch_at() / buffer_at(rank)

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
            return reinterpret_cast<float *>(base) +
                   static_cast<size_t>(rank) * CHUNK_CAPACITY;
        }
        const float *buffer_at(int rank) const
        {
            auto *base = reinterpret_cast<const char *>(this) + sizeof(ShmemSpinArena)
                         + static_cast<size_t>(num_ranks) * sizeof(EpochSlot);
            return reinterpret_cast<const float *>(base) +
                   static_cast<size_t>(rank) * CHUNK_CAPACITY;
        }

        /// Total arena size in bytes for a given rank count
        static size_t compute_size(int num_ranks)
        {
            return sizeof(ShmemSpinArena) // header (one cache line)
                   + static_cast<size_t>(num_ranks) * sizeof(EpochSlot)
                   + static_cast<size_t>(num_ranks) * CHUNK_CAPACITY * sizeof(float);
        }
    };

    static_assert(sizeof(ShmemSpinArena) == 64, "ShmemSpinArena header must be one cache line");

    /**
     * @brief Shared-memory spin-wait collective backend for N-rank intra-node allreduce
     *
     * The native path handles every FLOAT32/FP16/BF16 SUM payload size by
     * partitioning it into bounded chunks. Other operations delegate to the
     * wrapped UPI backend because they have different collective semantics.
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
         * @param general_backend UPI backend for collective operations outside
         *                        the native shared-memory SUM contract
         */
        ShmemSpinBackend(int domain_id, int my_rank,
                         std::unique_ptr<UPICollectiveBackend> general_backend);

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

        /**
         * @brief Gather variable-width FP32 record blocks through shared memory.
         *
         * Each participant publishes its total element count and streams its
         * payload through its NUMA-local persistent arena buffer. The root
         * appends complete participant blocks to `root_records`, beginning with
         * its own block. The method is total over payload size: records larger
         * than one arena slot are transferred in bounded chunks without MPI
         * payload traffic or hot-path allocation.
         *
         * @param local_records Participant-local packed records.
         * @param local_record_count Number of local records.
         * @param root_records Root receive storage; ignored on non-root ranks.
         * @param root_record_capacity Root storage capacity in records.
         * @param record_width_elements Width of one record in FP32 elements.
         * @param root_rank Root participant index.
         * @param gathered_record_count Total records on root; zero elsewhere.
         * @return true after every participant has completed the transaction.
         */
        bool gatherVariableFloatRecordsToRoot(
            const float *local_records,
            size_t local_record_count,
            float *root_records,
            size_t root_record_capacity,
            size_t record_width_elements,
            int root_rank,
            size_t &gathered_record_count);

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

        /**
         * @brief Return whether an allreduce uses the native shared-memory path.
         *
         * @param count Logical element count. Size never removes a supported
         *              SUM reduction from the shared-memory path.
         * @param dtype Element representation.
         * @param op Reduction operation.
         */
        bool usesSharedMemoryAllreduce(
            size_t count,
            CollectiveDataType dtype,
            CollectiveOp op) const;

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

        std::unique_ptr<UPICollectiveBackend> general_backend_; ///< UPI implementation for other collective semantics
        std::atomic<bool> abort_requested_{false};       ///< Local abort requested for this backend
        /** Persistent per-rank element counts for rooted publication. */
        std::vector<size_t> rooted_publication_element_counts_;
        /** Persistent root destination offsets in FP32 elements. */
        std::vector<size_t> rooted_publication_element_offsets_;
        bool initialized_ = false;
        mutable std::string last_error_;
    };

} // namespace llaminar2
