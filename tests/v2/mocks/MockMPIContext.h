/**
 * @file MockMPIContext.h
 * @brief Mock MPI context for unit testing distributed logic without real MPI
 *
 * This mock enables:
 * - Testing distributed algorithms without MPI runtime
 * - Simulating multi-rank scenarios in a single process
 * - Failure injection for robustness testing
 * - Call tracking for behavior verification
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "interfaces/IMPIContext.h"
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <cstring>
#include <stdexcept>

namespace llaminar2::test
{

    /**
     * @brief Mock MPI context for unit testing
     *
     * Provides configurable behavior for testing distributed logic:
     * - Configurable rank and world_size
     * - No-op or simulated collective operations
     * - Call tracking for verification
     * - Optional failure injection
     *
     * Usage:
     * @code
     * auto mock = std::make_shared<MockMPIContext>(
     *     MockMPIContext::Config{.rank = 1, .world_size = 4});
     *
     * // Use in place of real MPIContext
     * stage->execute(mock.get());
     *
     * // Verify behavior
     * EXPECT_EQ(mock->barrier_call_count(), 2);
     * @endcode
     */
    class MockMPIContext : public IMPIContext
    {
    public:
        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Configuration options for the mock
         */
        struct Config
        {
            int rank = 0;              ///< Simulated MPI rank
            int world_size = 1;        ///< Simulated world size
            bool track_calls = true;   ///< Record call counts for verification
            bool simulate_noop = true; ///< If true, collectives are no-ops

            // Failure injection
            bool barrier_should_fail = false;   ///< If true, barrier() throws
            bool allreduce_should_fail = false; ///< If true, allreduce throws
            float allreduce_corruption = 0.0f;  ///< Multiply allreduce results by this factor (1.0 = no corruption)
        };

        /**
         * @brief Default constructor (single-rank mock)
         */
        MockMPIContext()
            : config_{0, 1, true, true, false, false, 0.0f} {}

        /**
         * @brief Construct mock with configuration
         * @param config Configuration options
         */
        explicit MockMPIContext(const Config &config)
            : config_(config)
        {
            // Validate configuration
            if (config_.rank < 0 || config_.rank >= config_.world_size)
            {
                throw std::invalid_argument("MockMPIContext: rank must be in [0, world_size)");
            }
            if (config_.world_size < 1)
            {
                throw std::invalid_argument("MockMPIContext: world_size must be >= 1");
            }
        }

        /**
         * @brief Convenience constructor for simple cases
         * @param rank Simulated rank
         * @param world_size Simulated world size
         */
        MockMPIContext(int rank, int world_size)
            : config_{rank, world_size, true, true, false, false, 0.0f}
        {
            // Validate configuration
            if (config_.rank < 0 || config_.rank >= config_.world_size)
            {
                throw std::invalid_argument("MockMPIContext: rank must be in [0, world_size)");
            }
            if (config_.world_size < 1)
            {
                throw std::invalid_argument("MockMPIContext: world_size must be >= 1");
            }
        }

        // =========================================================================
        // IMPIContext Implementation - Identity
        // =========================================================================

        int rank() const override { return config_.rank; }
        int world_size() const override { return config_.world_size; }
        bool is_root() const override { return config_.rank == 0; }

        // =========================================================================
        // IMPIContext Implementation - Synchronization
        // =========================================================================

        void barrier() const override
        {
            if (config_.track_calls)
            {
                barrier_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (config_.barrier_should_fail)
            {
                throw std::runtime_error("MockMPIContext: injected barrier failure");
            }
            // No-op in single-process mock
        }

        // =========================================================================
        // IMPIContext Implementation - All-Reduce (FP32)
        // =========================================================================

        void allreduce_sum(const float *send_data, float *recv_data, size_t count) const override
        {
            if (config_.track_calls)
            {
                allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (config_.allreduce_should_fail)
            {
                throw std::runtime_error("MockMPIContext: injected allreduce failure");
            }

            // Simulate: in single-rank mock, output = input (optionally corrupted)
            if (config_.simulate_noop)
            {
                std::memcpy(recv_data, send_data, count * sizeof(float));

                // Apply corruption factor if set
                if (config_.allreduce_corruption != 0.0f && config_.allreduce_corruption != 1.0f)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        recv_data[i] *= config_.allreduce_corruption;
                    }
                }
            }
        }

        void allreduce_sum_inplace(float *data, size_t count) const override
        {
            if (config_.track_calls)
            {
                allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (config_.allreduce_should_fail)
            {
                throw std::runtime_error("MockMPIContext: injected allreduce failure");
            }

            // Simulate: in single-rank mock, data is unchanged (optionally corrupted)
            if (config_.simulate_noop && config_.allreduce_corruption != 0.0f &&
                config_.allreduce_corruption != 1.0f)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    data[i] *= config_.allreduce_corruption;
                }
            }
        }

        // =========================================================================
        // IMPIContext Implementation - All-Reduce (Quantized)
        // =========================================================================

        void allreduce_q8_1_inplace(Q8_1Block * /*data*/, size_t /*n_blocks*/) const override
        {
            if (config_.track_calls)
            {
                allreduce_q8_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock - quantized blocks left unchanged
        }

        void allreduce_q16_1_inplace(Q16_1Block * /*data*/, size_t /*n_blocks*/) const override
        {
            if (config_.track_calls)
            {
                allreduce_q16_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock - quantized blocks left unchanged
        }

        // =========================================================================
        // IMPIContext Implementation - All-Reduce (FP16/BF16)
        // =========================================================================

        void allreduce_fp16_inplace(uint16_t * /*data*/, size_t /*count*/) const override
        {
            if (config_.track_calls)
            {
                allreduce_fp16_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock
        }

        void allreduce_bf16_inplace(uint16_t * /*data*/, size_t /*count*/) const override
        {
            if (config_.track_calls)
            {
                allreduce_bf16_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock
        }

        // =========================================================================
        // IMPIContext Implementation - Broadcast
        // =========================================================================

        void broadcast(float * /*data*/, size_t /*count*/, int /*root*/) const override
        {
            if (config_.track_calls)
            {
                broadcast_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock - data unchanged
        }

        // =========================================================================
        // IMPIContext Implementation - All-Gather
        // =========================================================================

        void allgather(const float *send_data, float *recv_data, size_t count) const override
        {
            if (config_.track_calls)
            {
                allgather_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            // Simulate: fill recv_data with send_data repeated world_size times
            // In real MPI, each rank's data would be different
            if (config_.simulate_noop)
            {
                for (int r = 0; r < config_.world_size; ++r)
                {
                    std::memcpy(recv_data + r * count, send_data, count * sizeof(float));
                }
            }
        }

        void allgather_bytes(const void *send_data, void *recv_data, size_t byte_count) const override
        {
            if (config_.track_calls)
            {
                allgather_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            // Simulate: fill recv_data with send_data repeated world_size times
            if (config_.simulate_noop)
            {
                auto *recv_bytes = static_cast<char *>(recv_data);
                for (int r = 0; r < config_.world_size; ++r)
                {
                    std::memcpy(recv_bytes + r * byte_count, send_data, byte_count);
                }
            }
        }

        void allgatherv_bytes(const void *send_data, int send_count,
                              void *recv_data, const int *recv_counts,
                              const int *displs) const override
        {
            if (config_.track_calls)
            {
                allgatherv_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            // Simulate: copy this rank's data to appropriate position
            if (config_.simulate_noop)
            {
                auto *recv_bytes = static_cast<char *>(recv_data);
                // Copy our data to all positions (simplified simulation)
                for (int r = 0; r < config_.world_size; ++r)
                {
                    int copy_count = std::min(send_count, recv_counts[r]);
                    std::memcpy(recv_bytes + displs[r], send_data, copy_count);
                }
            }
        }

        // =========================================================================
        // IMPIContext Implementation - Point-to-Point (Mock: No-Op)
        // =========================================================================

        void send(const void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                  int /*dest*/, int /*tag*/) const override
        {
            if (config_.track_calls)
            {
                send_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock - data would be sent in real MPI
        }

        void recv(void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                  int /*src*/, int /*tag*/, MPI_Status * /*status*/) const override
        {
            if (config_.track_calls)
            {
                recv_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock - would receive data in real MPI
        }

        MPI_Request isend(const void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                          int /*dest*/, int /*tag*/) const override
        {
            if (config_.track_calls)
            {
                isend_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Return a null request - mock doesn't track pending requests
            return MPI_REQUEST_NULL;
        }

        MPI_Request irecv(void * /*data*/, size_t /*count*/, MPI_Datatype /*type*/,
                          int /*src*/, int /*tag*/) const override
        {
            if (config_.track_calls)
            {
                irecv_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Return a null request - mock doesn't track pending requests
            return MPI_REQUEST_NULL;
        }

        void wait(MPI_Request *request, MPI_Status * /*status*/) const override
        {
            if (config_.track_calls)
            {
                wait_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Set request to NULL as MPI_Wait would
            if (request)
            {
                *request = MPI_REQUEST_NULL;
            }
        }

        void waitAll(std::vector<MPI_Request> &requests) const override
        {
            if (config_.track_calls)
            {
                waitall_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Set all requests to NULL as MPI_Waitall would
            for (auto &req : requests)
            {
                req = MPI_REQUEST_NULL;
            }
        }

        void probe(int /*src*/, int /*tag*/, MPI_Status *status) const override
        {
            if (config_.track_calls)
            {
                probe_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Fill status with mock values
            if (status)
            {
                status->MPI_SOURCE = config_.rank == 0 ? 1 : 0;
                status->MPI_TAG = 0;
                status->MPI_ERROR = MPI_SUCCESS;
            }
        }

        bool iprobe(int /*src*/, int /*tag*/, MPI_Status *status) const override
        {
            if (config_.track_calls)
            {
                iprobe_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Mock: always return false (no message pending)
            if (status)
            {
                status->MPI_SOURCE = MPI_ANY_SOURCE;
                status->MPI_TAG = MPI_ANY_TAG;
                status->MPI_ERROR = MPI_SUCCESS;
            }
            return false;
        }

        void sendFloat(const float *data, size_t count, int dest, int tag) const override
        {
            send(data, count, MPI_FLOAT, dest, tag);
        }

        void recvFloat(float *data, size_t count, int src, int tag,
                       MPI_Status *status) const override
        {
            recv(data, count, MPI_FLOAT, src, tag, status);
        }

        void sendBytes(const void *data, size_t byte_count, int dest, int tag) const override
        {
            send(data, byte_count, MPI_BYTE, dest, tag);
        }

        void recvBytes(void *data, size_t byte_count, int src, int tag,
                       MPI_Status *status) const override
        {
            recv(data, byte_count, MPI_BYTE, src, tag, status);
        }

        int getCount(const MPI_Status & /*status*/, MPI_Datatype /*type*/) const override
        {
            // Mock: return 0 (no actual message)
            return 0;
        }

        // =========================================================================
        // IMPIContext Implementation - Work Distribution
        // =========================================================================

        std::pair<size_t, size_t> get_local_slice(size_t total_elements) const override
        {
            size_t base_count = total_elements / config_.world_size;
            size_t remainder = total_elements % config_.world_size;

            size_t start = config_.rank * base_count +
                           std::min(static_cast<size_t>(config_.rank), remainder);
            size_t count = base_count + (config_.rank < static_cast<int>(remainder) ? 1 : 0);

            return {start, count};
        }

        std::pair<size_t, size_t> distribute_rows(size_t total_rows) const override
        {
            return get_local_slice(total_rows);
        }

        // =========================================================================
        // Test Utilities - Call Counting
        // =========================================================================

        /**
         * @brief Get the number of barrier() calls
         */
        size_t barrier_call_count() const
        {
            return barrier_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of FP32 allreduce calls
         */
        size_t allreduce_call_count() const
        {
            return allreduce_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of Q8_1 allreduce calls
         */
        size_t allreduce_q8_call_count() const
        {
            return allreduce_q8_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of Q16_1 allreduce calls
         */
        size_t allreduce_q16_call_count() const
        {
            return allreduce_q16_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of FP16 allreduce calls
         */
        size_t allreduce_fp16_call_count() const
        {
            return allreduce_fp16_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of BF16 allreduce calls
         */
        size_t allreduce_bf16_call_count() const
        {
            return allreduce_bf16_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of broadcast() calls
         */
        size_t broadcast_call_count() const
        {
            return broadcast_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of allgather() calls
         */
        size_t allgather_call_count() const
        {
            return allgather_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of allgatherv() calls
         */
        size_t allgatherv_call_count() const
        {
            return allgatherv_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get total collective operation call count
         */
        size_t total_collective_calls() const
        {
            return barrier_call_count() + allreduce_call_count() +
                   allreduce_q8_call_count() + allreduce_q16_call_count() +
                   allreduce_fp16_call_count() + allreduce_bf16_call_count() +
                   broadcast_call_count() + allgather_call_count() +
                   allgatherv_call_count();
        }

        /**
         * @brief Reset all call counters
         */
        void reset_call_counts()
        {
            barrier_calls_.store(0, std::memory_order_relaxed);
            allreduce_calls_.store(0, std::memory_order_relaxed);
            allreduce_q8_calls_.store(0, std::memory_order_relaxed);
            allreduce_q16_calls_.store(0, std::memory_order_relaxed);
            allreduce_fp16_calls_.store(0, std::memory_order_relaxed);
            allreduce_bf16_calls_.store(0, std::memory_order_relaxed);
            broadcast_calls_.store(0, std::memory_order_relaxed);
            allgather_calls_.store(0, std::memory_order_relaxed);
            allgatherv_calls_.store(0, std::memory_order_relaxed);
            send_calls_.store(0, std::memory_order_relaxed);
            recv_calls_.store(0, std::memory_order_relaxed);
            isend_calls_.store(0, std::memory_order_relaxed);
            irecv_calls_.store(0, std::memory_order_relaxed);
            wait_calls_.store(0, std::memory_order_relaxed);
            waitall_calls_.store(0, std::memory_order_relaxed);
            probe_calls_.store(0, std::memory_order_relaxed);
            iprobe_calls_.store(0, std::memory_order_relaxed);
        }

        // =========================================================================
        // Test Utilities - P2P Call Counting
        // =========================================================================

        /**
         * @brief Get the number of send() calls
         */
        size_t send_call_count() const
        {
            return send_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of recv() calls
         */
        size_t recv_call_count() const
        {
            return recv_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of isend() calls
         */
        size_t isend_call_count() const
        {
            return isend_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of irecv() calls
         */
        size_t irecv_call_count() const
        {
            return irecv_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of wait() calls
         */
        size_t wait_call_count() const
        {
            return wait_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of waitAll() calls
         */
        size_t waitall_call_count() const
        {
            return waitall_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of probe() calls
         */
        size_t probe_call_count() const
        {
            return probe_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of iprobe() calls
         */
        size_t iprobe_call_count() const
        {
            return iprobe_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get total P2P operation call count
         */
        size_t total_p2p_calls() const
        {
            return send_call_count() + recv_call_count() +
                   isend_call_count() + irecv_call_count() +
                   wait_call_count() + waitall_call_count() +
                   probe_call_count() + iprobe_call_count();
        }

        // =========================================================================
        // Test Utilities - Configuration Modification
        // =========================================================================

        /**
         * @brief Enable/disable barrier failure injection
         */
        void set_barrier_fails(bool fails) { config_.barrier_should_fail = fails; }

        /**
         * @brief Enable/disable allreduce failure injection
         */
        void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }

        /**
         * @brief Set allreduce corruption factor
         * @param factor Multiply allreduce results by this (1.0 = no corruption)
         */
        void set_allreduce_corruption(float factor) { config_.allreduce_corruption = factor; }

        /**
         * @brief Get current configuration (read-only)
         */
        const Config &config() const { return config_; }

    private:
        Config config_;

        // Atomic call counters (thread-safe) - Collectives
        mutable std::atomic<size_t> barrier_calls_{0};
        mutable std::atomic<size_t> allreduce_calls_{0};
        mutable std::atomic<size_t> allreduce_q8_calls_{0};
        mutable std::atomic<size_t> allreduce_q16_calls_{0};
        mutable std::atomic<size_t> allreduce_fp16_calls_{0};
        mutable std::atomic<size_t> allreduce_bf16_calls_{0};
        mutable std::atomic<size_t> broadcast_calls_{0};
        mutable std::atomic<size_t> allgather_calls_{0};
        mutable std::atomic<size_t> allgatherv_calls_{0};

        // Atomic call counters (thread-safe) - Point-to-Point
        mutable std::atomic<size_t> send_calls_{0};
        mutable std::atomic<size_t> recv_calls_{0};
        mutable std::atomic<size_t> isend_calls_{0};
        mutable std::atomic<size_t> irecv_calls_{0};
        mutable std::atomic<size_t> wait_calls_{0};
        mutable std::atomic<size_t> waitall_calls_{0};
        mutable std::atomic<size_t> probe_calls_{0};
        mutable std::atomic<size_t> iprobe_calls_{0};
    };

} // namespace llaminar2::test
