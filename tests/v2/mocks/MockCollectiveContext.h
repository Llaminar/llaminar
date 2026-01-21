/**
 * @file MockCollectiveContext.h
 * @brief Mock collective context for unit testing without MPI/NCCL runtime
 *
 * This mock enables:
 * - Testing pipeline stages without collective runtime
 * - Simulating multi-rank scenarios in a single process
 * - Failure injection for robustness testing
 * - Call tracking for behavior verification
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "interfaces/ICollectiveContext.h"
#include "config/TPDomain.h"
#include "backends/DeviceId.h"
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sstream>

namespace llaminar2::test
{

    /**
     * @brief Mock collective context for unit testing
     *
     * Provides configurable behavior for testing distributed logic:
     * - Configurable rank and world_size
     * - No-op or simulated collective operations
     * - Call tracking for verification
     * - Optional failure injection
     *
     * Usage:
     * @code
     * // Simple construction
     * auto mock = MockCollectiveContext::Builder()
     *     .withRank(1)
     *     .withWorldSize(4)
     *     .withDevice(DeviceId::cpu())
     *     .build();
     *
     * // Use in place of real CollectiveContext
     * executor.setCollectiveContext(mock.get());
     *
     * // Verify behavior
     * EXPECT_EQ(mock->allreduce_call_count(), 2);
     * @endcode
     */
    class MockCollectiveContext : public ICollectiveContext
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
            int rank = 0;                        ///< Simulated MPI rank
            int world_size = 1;                  ///< Simulated world size
            std::vector<DeviceId> local_devices; ///< Devices on this "rank"
            bool track_calls = true;             ///< Record call counts for verification
            bool simulate_noop = true;           ///< If true, collectives are no-ops

            // Backend availability
            bool mpi_available = true;   ///< MPI backend available
            bool nccl_available = false; ///< NCCL backend available
            bool rccl_available = false; ///< RCCL backend available
            bool host_available = true;  ///< Host backend available

            // Failure injection
            bool allreduce_should_fail = false; ///< If true, allreduce returns false
            bool allgather_should_fail = false; ///< If true, allgather returns false
            bool broadcast_should_fail = false; ///< If true, broadcast returns false

            // Custom behavior hooks (optional)
            std::function<bool(ITensor *, size_t, DeviceId, CollectiveOp)> allreduce_hook;
            std::function<bool(ITensor *, ITensor *, size_t, DeviceId)> allgather_hook;
            std::function<bool(ITensor *, size_t, int, DeviceId)> broadcast_hook;
        };

        /**
         * @brief Builder pattern for MockCollectiveContext configuration
         */
        class Builder
        {
        public:
            Builder() = default;

            /// Set the simulated rank
            Builder &withRank(int rank)
            {
                config_.rank = rank;
                return *this;
            }

            /// Set the simulated world size
            Builder &withWorldSize(int world_size)
            {
                config_.world_size = world_size;
                return *this;
            }

            /// Add a device to local devices
            Builder &withDevice(DeviceId device)
            {
                config_.local_devices.push_back(device);
                return *this;
            }

            /// Set all local devices
            Builder &withDevices(std::vector<DeviceId> devices)
            {
                config_.local_devices = std::move(devices);
                return *this;
            }

            /// Enable/disable call tracking
            Builder &withCallTracking(bool enabled)
            {
                config_.track_calls = enabled;
                return *this;
            }

            /// Enable/disable no-op simulation
            Builder &withNoopSimulation(bool enabled)
            {
                config_.simulate_noop = enabled;
                return *this;
            }

            /// Set MPI backend availability
            Builder &withMPIAvailable(bool available)
            {
                config_.mpi_available = available;
                return *this;
            }

            /// Set NCCL backend availability
            Builder &withNCCLAvailable(bool available)
            {
                config_.nccl_available = available;
                return *this;
            }

            /// Set RCCL backend availability
            Builder &withRCCLAvailable(bool available)
            {
                config_.rccl_available = available;
                return *this;
            }

            /// Set Host backend availability
            Builder &withHostAvailable(bool available)
            {
                config_.host_available = available;
                return *this;
            }

            /// Enable allreduce failure injection
            Builder &withAllreduceFails(bool fails)
            {
                config_.allreduce_should_fail = fails;
                return *this;
            }

            /// Enable allgather failure injection
            Builder &withAllgatherFails(bool fails)
            {
                config_.allgather_should_fail = fails;
                return *this;
            }

            /// Enable broadcast failure injection
            Builder &withBroadcastFails(bool fails)
            {
                config_.broadcast_should_fail = fails;
                return *this;
            }

            /// Set custom allreduce hook
            Builder &withAllreduceHook(std::function<bool(ITensor *, size_t, DeviceId, CollectiveOp)> hook)
            {
                config_.allreduce_hook = std::move(hook);
                return *this;
            }

            /// Set custom allgather hook
            Builder &withAllgatherHook(std::function<bool(ITensor *, ITensor *, size_t, DeviceId)> hook)
            {
                config_.allgather_hook = std::move(hook);
                return *this;
            }

            /// Set custom broadcast hook
            Builder &withBroadcastHook(std::function<bool(ITensor *, size_t, int, DeviceId)> hook)
            {
                config_.broadcast_hook = std::move(hook);
                return *this;
            }

            /// Build the mock context
            std::unique_ptr<MockCollectiveContext> build()
            {
                return std::make_unique<MockCollectiveContext>(config_);
            }

            /// Build as shared_ptr
            std::shared_ptr<MockCollectiveContext> buildShared()
            {
                return std::make_shared<MockCollectiveContext>(config_);
            }

        private:
            Config config_;
        };

        // =========================================================================
        // Presets
        // =========================================================================

        /**
         * @brief Create a single-rank CPU-only context (no collectives)
         */
        static std::unique_ptr<MockCollectiveContext> singleRankCPU()
        {
            return Builder()
                .withRank(0)
                .withWorldSize(1)
                .withDevice(DeviceId::cpu())
                .build();
        }

        /**
         * @brief Create a multi-rank CPU context for testing tensor parallelism
         * @param rank This rank's index
         * @param world_size Total number of ranks
         */
        static std::unique_ptr<MockCollectiveContext> multiRankCPU(int rank, int world_size)
        {
            return Builder()
                .withRank(rank)
                .withWorldSize(world_size)
                .withDevice(DeviceId::cpu())
                .build();
        }

        /**
         * @brief Create a context that simulates failure
         */
        static std::unique_ptr<MockCollectiveContext> failingContext()
        {
            return Builder()
                .withRank(0)
                .withWorldSize(2)
                .withDevice(DeviceId::cpu())
                .withAllreduceFails(true)
                .withAllgatherFails(true)
                .withBroadcastFails(true)
                .build();
        }

        /**
         * @brief Create a context with no backends available
         */
        static std::unique_ptr<MockCollectiveContext> noBackendsAvailable()
        {
            return Builder()
                .withRank(0)
                .withWorldSize(2)
                .withDevice(DeviceId::cpu())
                .withMPIAvailable(false)
                .withNCCLAvailable(false)
                .withRCCLAvailable(false)
                .withHostAvailable(false)
                .build();
        }

        // =========================================================================
        // Constructors
        // =========================================================================

        /**
         * @brief Default constructor (single-rank, no collectives)
         */
        MockCollectiveContext()
            : config_{}
        {
            config_.local_devices.push_back(DeviceId::cpu());
        }

        /**
         * @brief Construct with configuration
         * @param config Configuration options
         */
        explicit MockCollectiveContext(const Config &config)
            : config_(config)
        {
            // Validate configuration
            if (config_.rank < 0 || config_.rank >= config_.world_size)
            {
                throw std::invalid_argument(
                    "MockCollectiveContext: rank must be in [0, world_size). Got rank=" +
                    std::to_string(config_.rank) + ", world_size=" + std::to_string(config_.world_size));
            }
            if (config_.world_size < 1)
            {
                throw std::invalid_argument(
                    "MockCollectiveContext: world_size must be >= 1. Got " +
                    std::to_string(config_.world_size));
            }
            // Default device if none specified
            if (config_.local_devices.empty())
            {
                config_.local_devices.push_back(DeviceId::cpu());
            }
        }

        /**
         * @brief Convenience constructor for simple cases
         * @param rank Simulated rank
         * @param world_size Simulated world size
         */
        MockCollectiveContext(int rank, int world_size)
            : MockCollectiveContext(Config{rank, world_size, {DeviceId::cpu()}}) {}

        // =========================================================================
        // ICollectiveContext Implementation - Operations
        // =========================================================================

        bool executeAllreduce(
            ITensor *buffer,
            size_t count,
            DeviceId tensor_device,
            CollectiveOp op = CollectiveOp::ALLREDUCE_SUM) override
        {

            if (config_.track_calls)
            {
                allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
                last_allreduce_op_ = op;
                last_allreduce_count_ = count;
                last_allreduce_device_ = tensor_device;
            }

            // Failure injection
            if (config_.allreduce_should_fail)
            {
                return false;
            }

            // Custom hook
            if (config_.allreduce_hook)
            {
                return config_.allreduce_hook(buffer, count, tensor_device, op);
            }

            // No-op simulation: success without modification
            return true;
        }

        bool executeAllgather(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device) override
        {

            if (config_.track_calls)
            {
                allgather_calls_.fetch_add(1, std::memory_order_relaxed);
                last_allgather_seq_len_ = actual_seq_len;
                last_allgather_device_ = tensor_device;
            }

            // Failure injection
            if (config_.allgather_should_fail)
            {
                return false;
            }

            // Custom hook
            if (config_.allgather_hook)
            {
                return config_.allgather_hook(local_input, full_output, actual_seq_len, tensor_device);
            }

            // No-op simulation: success without modification
            return true;
        }

        bool executeAllgatherv(
            ITensor * /*local_input*/,
            ITensor * /*full_output*/,
            const std::vector<int> & /*recv_counts*/,
            const std::vector<int> & /*displacements*/,
            size_t actual_seq_len,
            DeviceId tensor_device) override
        {

            if (config_.track_calls)
            {
                allgatherv_calls_.fetch_add(1, std::memory_order_relaxed);
                last_allgatherv_seq_len_ = actual_seq_len;
                last_allgatherv_device_ = tensor_device;
            }

            // Failure injection (use same flag as allgather)
            if (config_.allgather_should_fail)
            {
                return false;
            }

            // No-op simulation: success without modification
            return true;
        }

        bool executeBroadcast(
            ITensor *buffer,
            size_t count,
            int root_rank,
            DeviceId tensor_device) override
        {

            if (config_.track_calls)
            {
                broadcast_calls_.fetch_add(1, std::memory_order_relaxed);
                last_broadcast_count_ = count;
                last_broadcast_root_ = root_rank;
                last_broadcast_device_ = tensor_device;
            }

            // Failure injection
            if (config_.broadcast_should_fail)
            {
                return false;
            }

            // Custom hook
            if (config_.broadcast_hook)
            {
                return config_.broadcast_hook(buffer, count, root_rank, tensor_device);
            }

            // No-op simulation: success without modification
            return true;
        }

        // =========================================================================
        // ICollectiveContext Implementation - Domain-Aware Operations
        // =========================================================================

        bool executeAllreduceInDomain(
            ITensor *buffer,
            size_t count,
            DeviceId tensor_device,
            CollectiveOp op,
            const TPDomain *domain) override
        {

            if (config_.track_calls)
            {
                allreduce_in_domain_calls_.fetch_add(1, std::memory_order_relaxed);
                last_allreduce_domain_ = domain;
            }

            // Fallback to non-domain method (same behavior as real implementation)
            if (!domain)
            {
                return executeAllreduce(buffer, count, tensor_device, op);
            }

            // Skip trivial domains
            if (domain->isTrivial())
            {
                return true;
            }

            // Failure injection (use same flag as allreduce)
            if (config_.allreduce_should_fail)
            {
                return false;
            }

            // No-op simulation: success without modification
            return true;
        }

        bool executeAllgatherInDomain(
            ITensor *local_input,
            ITensor *full_output,
            size_t actual_seq_len,
            DeviceId tensor_device,
            const TPDomain *domain) override
        {

            if (config_.track_calls)
            {
                allgather_in_domain_calls_.fetch_add(1, std::memory_order_relaxed);
                last_allgather_domain_ = domain;
            }

            // Fallback to non-domain method
            if (!domain)
            {
                return executeAllgather(local_input, full_output, actual_seq_len, tensor_device);
            }

            // Skip trivial domains
            if (domain->isTrivial())
            {
                return true;
            }

            // Failure injection
            if (config_.allgather_should_fail)
            {
                return false;
            }

            // No-op simulation: success without modification
            return true;
        }

        bool executeAllgathervInDomain(
            ITensor *local_input,
            ITensor *full_output,
            const std::vector<int> & /*recv_counts*/,
            const std::vector<int> & /*displacements*/,
            size_t actual_seq_len,
            DeviceId tensor_device,
            const TPDomain *domain) override
        {

            if (config_.track_calls)
            {
                allgatherv_in_domain_calls_.fetch_add(1, std::memory_order_relaxed);
                last_allgatherv_domain_ = domain;
            }

            // Fallback to non-domain method
            if (!domain)
            {
                return executeAllgatherv(local_input, full_output, {}, {},
                                         actual_seq_len, tensor_device);
            }

            // Skip trivial domains
            if (domain->isTrivial())
            {
                return true;
            }

            // Failure injection
            if (config_.allgather_should_fail)
            {
                return false;
            }

            // No-op simulation: success without modification
            return true;
        }

        // =========================================================================
        // ICollectiveContext Implementation - Query Methods
        // =========================================================================

        bool requiresCollectives() const override
        {
            return config_.world_size > 1;
        }

        int worldSize() const override
        {
            return config_.world_size;
        }

        int rank() const override
        {
            return config_.rank;
        }

        const std::vector<DeviceId> &localDevices() const override
        {
            return config_.local_devices;
        }

        bool isBackendAvailable(CollectiveBackendType type) const override
        {
            switch (type)
            {
            case CollectiveBackendType::MPI:
                return config_.mpi_available;
            case CollectiveBackendType::NCCL:
                return config_.nccl_available;
            case CollectiveBackendType::RCCL:
                return config_.rccl_available;
            case CollectiveBackendType::HOST:
                return config_.host_available;
            case CollectiveBackendType::AUTO:
                return config_.mpi_available || config_.nccl_available ||
                       config_.rccl_available || config_.host_available;
            }
            return false;
        }

        // =========================================================================
        // ICollectiveContext Implementation - Advanced Access
        // =========================================================================

        IBackendRouter *router() override
        {
            return nullptr; // Mock doesn't have a real router
        }

        MPIContext *mpiContext() override
        {
            return nullptr; // Mock doesn't have a real MPI context
        }

        // =========================================================================
        // Test Utilities - Call Counting
        // =========================================================================

        /**
         * @brief Get the number of executeAllreduce() calls
         */
        size_t allreduce_call_count() const
        {
            return allreduce_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of executeAllgather() calls
         */
        size_t allgather_call_count() const
        {
            return allgather_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of executeBroadcast() calls
         */
        size_t broadcast_call_count() const
        {
            return broadcast_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get total collective operation call count
         */
        size_t total_collective_calls() const
        {
            return allreduce_call_count() + allgather_call_count() + broadcast_call_count();
        }

        /**
         * @brief Reset all call counters
         */
        void reset_call_counts()
        {
            allreduce_calls_.store(0, std::memory_order_relaxed);
            allgather_calls_.store(0, std::memory_order_relaxed);
            broadcast_calls_.store(0, std::memory_order_relaxed);
            allreduce_in_domain_calls_.store(0, std::memory_order_relaxed);
            allgather_in_domain_calls_.store(0, std::memory_order_relaxed);
            allgatherv_in_domain_calls_.store(0, std::memory_order_relaxed);
        }

        // =========================================================================
        // Test Utilities - Domain-Aware Call Counting
        // =========================================================================

        /**
         * @brief Get the number of executeAllreduceInDomain() calls
         */
        size_t allreduce_in_domain_call_count() const
        {
            return allreduce_in_domain_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of executeAllgatherInDomain() calls
         */
        size_t allgather_in_domain_call_count() const
        {
            return allgather_in_domain_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of executeAllgathervInDomain() calls
         */
        size_t allgatherv_in_domain_call_count() const
        {
            return allgatherv_in_domain_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the last domain passed to executeAllreduceInDomain
         */
        const TPDomain *last_allreduce_domain() const { return last_allreduce_domain_; }

        /**
         * @brief Get the last domain passed to executeAllgatherInDomain
         */
        const TPDomain *last_allgather_domain() const { return last_allgather_domain_; }

        /**
         * @brief Get the last domain passed to executeAllgathervInDomain
         */
        const TPDomain *last_allgatherv_domain() const { return last_allgatherv_domain_; }

        // =========================================================================
        // Test Utilities - Last Call Information
        // =========================================================================

        /// Get the last allreduce operation type
        CollectiveOp last_allreduce_op() const { return last_allreduce_op_; }

        /// Get the last allreduce count
        size_t last_allreduce_count() const { return last_allreduce_count_; }

        /// Get the last allreduce device
        DeviceId last_allreduce_device() const { return last_allreduce_device_; }

        /// Get the last allgather sequence length
        size_t last_allgather_seq_len() const { return last_allgather_seq_len_; }

        /// Get the last allgather device
        DeviceId last_allgather_device() const { return last_allgather_device_; }

        /// Get the last broadcast count
        size_t last_broadcast_count() const { return last_broadcast_count_; }

        /// Get the last broadcast root rank
        int last_broadcast_root() const { return last_broadcast_root_; }

        /// Get the last broadcast device
        DeviceId last_broadcast_device() const { return last_broadcast_device_; }

        // =========================================================================
        // Test Utilities - Configuration Modification
        // =========================================================================

        /**
         * @brief Enable/disable allreduce failure injection
         */
        void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }

        /**
         * @brief Enable/disable allgather failure injection
         */
        void set_allgather_fails(bool fails) { config_.allgather_should_fail = fails; }

        /**
         * @brief Enable/disable broadcast failure injection
         */
        void set_broadcast_fails(bool fails) { config_.broadcast_should_fail = fails; }

        /**
         * @brief Set MPI backend availability
         */
        void set_mpi_available(bool available) { config_.mpi_available = available; }

        /**
         * @brief Set NCCL backend availability
         */
        void set_nccl_available(bool available) { config_.nccl_available = available; }

        /**
         * @brief Set RCCL backend availability
         */
        void set_rccl_available(bool available) { config_.rccl_available = available; }

        /**
         * @brief Set Host backend availability
         */
        void set_host_available(bool available) { config_.host_available = available; }

        /**
         * @brief Get current configuration (read-only)
         */
        const Config &config() const { return config_; }

        /**
         * @brief Get description string for debugging
         */
        std::string description() const
        {
            std::ostringstream oss;
            oss << "MockCollectiveContext{rank=" << config_.rank
                << ", world_size=" << config_.world_size
                << ", devices=" << config_.local_devices.size()
                << ", calls=" << total_collective_calls() << "}";
            return oss.str();
        }

    private:
        Config config_;

        // Atomic call counters (thread-safe)
        mutable std::atomic<size_t> allreduce_calls_{0};
        mutable std::atomic<size_t> allgather_calls_{0};
        mutable std::atomic<size_t> allgatherv_calls_{0};
        mutable std::atomic<size_t> broadcast_calls_{0};

        // Domain-aware call counters
        mutable std::atomic<size_t> allreduce_in_domain_calls_{0};
        mutable std::atomic<size_t> allgather_in_domain_calls_{0};
        mutable std::atomic<size_t> allgatherv_in_domain_calls_{0};

        // Last call information
        CollectiveOp last_allreduce_op_ = CollectiveOp::ALLREDUCE_SUM;
        size_t last_allreduce_count_ = 0;
        DeviceId last_allreduce_device_ = DeviceId::cpu();

        size_t last_allgather_seq_len_ = 0;
        DeviceId last_allgather_device_ = DeviceId::cpu();

        size_t last_allgatherv_seq_len_ = 0;
        DeviceId last_allgatherv_device_ = DeviceId::cpu();

        size_t last_broadcast_count_ = 0;
        int last_broadcast_root_ = 0;
        DeviceId last_broadcast_device_ = DeviceId::cpu();

        // Domain-aware last call information
        const TPDomain *last_allreduce_domain_ = nullptr;
        const TPDomain *last_allgather_domain_ = nullptr;
        const TPDomain *last_allgatherv_domain_ = nullptr;
    };

} // namespace llaminar2::test
