/**
 * @file MockWorkDistributor.h
 * @brief Mock work distributor for unit testing without real MPI/device topology
 *
 * This mock enables:
 * - Testing work distribution algorithms without MPI runtime
 * - Simulating different world sizes and device configurations
 * - Predefined work slices for deterministic testing
 * - Call tracking for behavior verification
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "execution/mpi_orchestration/IWorkDistributor.h"
#include <vector>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace llaminar2::test
{

    /**
     * @brief Mock work distributor for unit testing
     *
     * Provides configurable behavior for testing work distribution:
     * - Configurable rank, world_size, and devices
     * - Default even distribution or custom slice overrides
     * - Call tracking for verification
     * - Presets for common scenarios (single rank, tensor parallel, heterogeneous)
     *
     * Usage:
     * @code
     * auto mock = MockWorkDistributor::Builder()
     *     .withRank(0)
     *     .withWorldSize(4)
     *     .withDevices({DeviceId::cpu(), DeviceId::cuda(0)})
     *     .build();
     *
     * auto slice = mock->getRankSlice(1000);
     * EXPECT_EQ(slice.count, 250);  // 1000 / 4 ranks
     * @endcode
     */
    class MockWorkDistributor : public IWorkDistributor
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
            int rank = 0;                                      ///< Simulated MPI rank
            int world_size = 1;                                ///< Simulated world size
            std::vector<DeviceId> devices = {DeviceId::cpu()}; ///< Devices for this rank
            std::vector<float> device_weights;                 ///< Relative compute power per device
            bool track_calls = true;                           ///< Record call counts for verification

            // Override behavior
            bool use_custom_rank_slice = false;          ///< Use custom slice instead of computed
            WorkSlice custom_rank_slice;                 ///< Custom rank slice to return
            bool use_custom_device_slices = false;       ///< Use custom device slices
            std::vector<WorkSlice> custom_device_slices; ///< Custom device slices to return
        };

        // =========================================================================
        // Builder Pattern
        // =========================================================================

        /**
         * @brief Builder for constructing MockWorkDistributor with fluent API
         */
        class Builder
        {
        public:
            Builder() = default;

            /**
             * @brief Set the MPI rank
             */
            Builder &withRank(int rank)
            {
                config_.rank = rank;
                return *this;
            }

            /**
             * @brief Set the world size
             */
            Builder &withWorldSize(int world_size)
            {
                config_.world_size = world_size;
                return *this;
            }

            /**
             * @brief Set the devices for this rank
             */
            Builder &withDevices(std::vector<DeviceId> devices)
            {
                config_.devices = std::move(devices);
                return *this;
            }

            /**
             * @brief Add a CPU device
             */
            Builder &withCPU()
            {
                // Clear default devices on first explicit device addition
                if (config_.devices.size() == 1 && config_.devices[0].is_cpu())
                {
                    config_.devices.clear();
                }
                config_.devices.push_back(DeviceId::cpu());
                return *this;
            }

            /**
             * @brief Add a CUDA device
             */
            Builder &withCUDA(int ordinal = 0)
            {
                // Clear default devices on first explicit device addition
                if (config_.devices.size() == 1 && config_.devices[0].is_cpu())
                {
                    config_.devices.clear();
                }
                config_.devices.push_back(DeviceId::cuda(ordinal));
                return *this;
            }

            /**
             * @brief Add a ROCm device
             */
            Builder &withROCm(int ordinal = 0)
            {
                // Clear default devices on first explicit device addition
                if (config_.devices.size() == 1 && config_.devices[0].is_cpu())
                {
                    config_.devices.clear();
                }
                config_.devices.push_back(DeviceId::rocm(ordinal));
                return *this;
            }

            /**
             * @brief Set device weights for uneven distribution
             */
            Builder &withDeviceWeights(std::vector<float> weights)
            {
                config_.device_weights = std::move(weights);
                return *this;
            }

            /**
             * @brief Enable/disable call tracking
             */
            Builder &withCallTracking(bool enabled)
            {
                config_.track_calls = enabled;
                return *this;
            }

            /**
             * @brief Override the rank slice with a custom value
             */
            Builder &withCustomRankSlice(WorkSlice slice)
            {
                config_.use_custom_rank_slice = true;
                config_.custom_rank_slice = slice;
                return *this;
            }

            /**
             * @brief Override device slices with custom values
             */
            Builder &withCustomDeviceSlices(std::vector<WorkSlice> slices)
            {
                config_.use_custom_device_slices = true;
                config_.custom_device_slices = std::move(slices);
                return *this;
            }

            /**
             * @brief Build the mock with configured options
             */
            std::shared_ptr<MockWorkDistributor> build()
            {
                return std::make_shared<MockWorkDistributor>(config_);
            }

        private:
            Config config_;
        };

        // =========================================================================
        // Preset Factory Methods
        // =========================================================================

        /**
         * @brief Create a single-rank, single-device mock (simplest case)
         */
        static std::shared_ptr<MockWorkDistributor> singleRank()
        {
            return Builder().withRank(0).withWorldSize(1).withDevices({DeviceId::cpu()}).build();
        }

        /**
         * @brief Create a tensor-parallel mock (multiple ranks, single device each)
         * @param rank This rank's index
         * @param world_size Total ranks
         */
        static std::shared_ptr<MockWorkDistributor> tensorParallel(int rank, int world_size)
        {
            return Builder().withRank(rank).withWorldSize(world_size).withDevices({DeviceId::cpu()}).build();
        }

        /**
         * @brief Create a heterogeneous mock (single rank, multiple devices)
         * @param devices Devices for this rank
         * @param weights Optional device weights
         */
        static std::shared_ptr<MockWorkDistributor> heterogeneous(
            std::vector<DeviceId> devices,
            std::vector<float> weights = {})
        {
            return Builder()
                .withRank(0)
                .withWorldSize(1)
                .withDevices(std::move(devices))
                .withDeviceWeights(std::move(weights))
                .build();
        }

        /**
         * @brief Create a multi-GPU mock (single rank, multiple CUDA devices)
         * @param num_gpus Number of CUDA GPUs
         */
        static std::shared_ptr<MockWorkDistributor> multiGPU(int num_gpus)
        {
            std::vector<DeviceId> devices;
            for (int i = 0; i < num_gpus; ++i)
            {
                devices.push_back(DeviceId::cuda(i));
            }
            return Builder().withRank(0).withWorldSize(1).withDevices(devices).build();
        }

        /**
         * @brief Create a full distributed mock (multiple ranks, multiple devices)
         * @param rank This rank's index
         * @param world_size Total ranks
         * @param devices Devices for this rank
         */
        static std::shared_ptr<MockWorkDistributor> distributed(
            int rank, int world_size, std::vector<DeviceId> devices)
        {
            return Builder()
                .withRank(rank)
                .withWorldSize(world_size)
                .withDevices(std::move(devices))
                .build();
        }

        // =========================================================================
        // Constructors
        // =========================================================================

        /**
         * @brief Default constructor (single-rank, single-device)
         */
        MockWorkDistributor()
            : config_{0, 1, {DeviceId::cpu()}, {}, true, false, {}, false, {}} {}

        /**
         * @brief Construct mock with configuration
         * @param config Configuration options
         */
        explicit MockWorkDistributor(const Config &config)
            : config_(config)
        {
            validateConfig();
        }

        /**
         * @brief Simple constructor for common cases
         * @param rank Simulated rank
         * @param world_size Simulated world size
         * @param devices Devices for this rank
         */
        MockWorkDistributor(int rank, int world_size, std::vector<DeviceId> devices = {DeviceId::cpu()})
            : config_{rank, world_size, std::move(devices), {}, true, false, {}, false, {}}
        {
            validateConfig();
        }

        // =========================================================================
        // IWorkDistributor Implementation - Rank Distribution
        // =========================================================================

        WorkSlice getRankSlice(size_t total_elements) const override
        {
            if (config_.track_calls)
            {
                get_rank_slice_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            if (config_.use_custom_rank_slice)
            {
                return config_.custom_rank_slice;
            }

            // Standard even distribution
            return computeRankSlice(total_elements, config_.rank, config_.world_size);
        }

        std::vector<WorkSlice> getAllRankSlices(size_t total_elements) const override
        {
            if (config_.track_calls)
            {
                get_all_rank_slices_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<WorkSlice> slices;
            slices.reserve(config_.world_size);
            for (int r = 0; r < config_.world_size; ++r)
            {
                slices.push_back(computeRankSlice(total_elements, r, config_.world_size));
            }
            return slices;
        }

        bool rankHasWork(size_t total_elements) const override
        {
            if (config_.track_calls)
            {
                rank_has_work_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            return getRankSlice(total_elements).count > 0;
        }

        // =========================================================================
        // IWorkDistributor Implementation - Device Distribution
        // =========================================================================

        WorkSlice getDeviceSlice(size_t rank_elements, DeviceId device) const override
        {
            if (config_.track_calls)
            {
                get_device_slice_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            auto slices = getAllDeviceSlices(rank_elements);

            // Find the device
            for (size_t i = 0; i < config_.devices.size(); ++i)
            {
                if (config_.devices[i].type == device.type &&
                    config_.devices[i].ordinal == device.ordinal)
                {
                    return slices[i];
                }
            }

            // Device not found - return empty slice
            return WorkSlice{0, 0, 0, -1};
        }

        std::vector<WorkSlice> getAllDeviceSlices(size_t rank_elements) const override
        {
            if (config_.track_calls)
            {
                get_all_device_slices_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            if (config_.use_custom_device_slices)
            {
                return config_.custom_device_slices;
            }

            // Compute slices based on weights or even distribution
            auto weights = getNormalizedWeights();
            std::vector<WorkSlice> slices;
            slices.reserve(config_.devices.size());

            size_t start = 0;
            for (size_t i = 0; i < config_.devices.size(); ++i)
            {
                size_t count = static_cast<size_t>(weights[i] * rank_elements);

                // Handle remainder for last device
                if (i == config_.devices.size() - 1)
                {
                    count = rank_elements - start;
                }

                slices.push_back(WorkSlice{
                    start,
                    start + count,
                    count,
                    static_cast<int>(i)});
                start += count;
            }

            return slices;
        }

        int getDeviceForElement(size_t element_idx, size_t rank_elements) const override
        {
            if (config_.track_calls)
            {
                get_device_for_element_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            auto slices = getAllDeviceSlices(rank_elements);
            for (size_t i = 0; i < slices.size(); ++i)
            {
                if (slices[i].contains(element_idx))
                {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        // =========================================================================
        // IWorkDistributor Implementation - Hierarchical Distribution
        // =========================================================================

        std::vector<HierarchicalSlice> distribute(size_t total_elements) const override
        {
            if (config_.track_calls)
            {
                distribute_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            auto rank_slice = getRankSlice(total_elements);
            auto device_slices = getAllDeviceSlices(rank_slice.count);

            std::vector<HierarchicalSlice> result;
            result.reserve(device_slices.size());

            for (size_t i = 0; i < device_slices.size(); ++i)
            {
                result.push_back(HierarchicalSlice{
                    config_.rank,
                    config_.devices[i],
                    rank_slice.start + device_slices[i].start,
                    rank_slice.start + device_slices[i].end,
                    device_slices[i].start,
                    device_slices[i].count});
            }

            return result;
        }

        HierarchicalSlice getPrimaryDeviceSlice(size_t total_elements) const override
        {
            if (config_.track_calls)
            {
                get_primary_device_slice_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            auto all = distribute(total_elements);
            if (all.empty())
            {
                return HierarchicalSlice{config_.rank, DeviceId::cpu(), 0, 0, 0, 0};
            }
            return all[0];
        }

        // =========================================================================
        // IWorkDistributor Implementation - MoE Expert Distribution
        // =========================================================================

        std::vector<ExpertAssignment> distributeExperts(int num_experts) const override
        {
            if (config_.track_calls)
            {
                distribute_experts_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<ExpertAssignment> assignments;
            assignments.reserve(num_experts);

            // Round-robin distribution across all devices
            size_t total_devices = config_.devices.size() * config_.world_size;

            for (int e = 0; e < num_experts; ++e)
            {
                size_t global_device = e % total_devices;
                int target_rank = static_cast<int>(global_device / config_.devices.size());
                size_t device_idx = global_device % config_.devices.size();

                // Only include local devices (this rank's)
                if (target_rank == config_.rank)
                {
                    assignments.push_back(ExpertAssignment{
                        e,
                        config_.devices[device_idx],
                        target_rank});
                }
            }

            return assignments;
        }

        std::vector<TokenRouting> routeTokensToExperts(
            const float *router_output,
            const std::vector<ExpertAssignment> &expert_assignments,
            int top_k,
            int seq_len,
            int num_experts) const override
        {

            if (config_.track_calls)
            {
                route_tokens_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            // Simple mock implementation: each token goes to its top-k experts
            std::vector<TokenRouting> routing;

            // Build expert_id -> assignment map
            std::vector<const ExpertAssignment *> local_experts(num_experts, nullptr);
            for (const auto &ea : expert_assignments)
            {
                if (ea.expert_id < num_experts)
                {
                    local_experts[ea.expert_id] = &ea;
                }
            }

            // Route tokens
            for (int tok = 0; tok < seq_len; ++tok)
            {
                // Find top-k experts for this token
                std::vector<std::pair<float, int>> scores;
                for (int e = 0; e < num_experts; ++e)
                {
                    scores.emplace_back(router_output[tok * num_experts + e], e);
                }
                std::sort(scores.begin(), scores.end(), std::greater<>());

                // Add to routing for local experts only
                for (int k = 0; k < top_k && k < static_cast<int>(scores.size()); ++k)
                {
                    int expert_id = scores[k].second;
                    if (local_experts[expert_id])
                    {
                        // Find or create routing entry
                        TokenRouting *entry = nullptr;
                        for (auto &r : routing)
                        {
                            if (r.expert_id == expert_id)
                            {
                                entry = &r;
                                break;
                            }
                        }
                        if (!entry)
                        {
                            routing.push_back(TokenRouting{
                                expert_id,
                                local_experts[expert_id]->device,
                                {},
                                {}});
                            entry = &routing.back();
                        }
                        entry->token_indices.push_back(tok);
                        entry->weights.push_back(scores[k].first);
                    }
                }
            }

            return routing;
        }

        // =========================================================================
        // IWorkDistributor Implementation - Utility Methods
        // =========================================================================

        size_t estimateMemoryPerDevice(size_t total_bytes) const override
        {
            if (config_.track_calls)
            {
                estimate_memory_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            if (config_.devices.empty())
                return total_bytes;

            // Account for rank distribution first
            size_t per_rank = total_bytes / config_.world_size;

            // Then device distribution
            return per_rank / config_.devices.size();
        }

        std::vector<size_t> getElementCountsPerDevice(size_t total_elements) const override
        {
            if (config_.track_calls)
            {
                get_element_counts_calls_.fetch_add(1, std::memory_order_relaxed);
            }

            auto rank_slice = getRankSlice(total_elements);
            auto device_slices = getAllDeviceSlices(rank_slice.count);

            std::vector<size_t> counts;
            counts.reserve(device_slices.size());
            for (const auto &slice : device_slices)
            {
                counts.push_back(slice.count);
            }
            return counts;
        }

        // =========================================================================
        // IWorkDistributor Implementation - Accessors
        // =========================================================================

        int worldSize() const override { return config_.world_size; }
        int rank() const override { return config_.rank; }
        const std::vector<DeviceId> &devices() const override { return config_.devices; }
        size_t deviceCount() const override { return config_.devices.size(); }
        bool hasMultipleDevices() const override { return config_.devices.size() > 1; }

        // =========================================================================
        // Test Utilities - Call Counting
        // =========================================================================

        size_t get_rank_slice_call_count() const
        {
            return get_rank_slice_calls_.load(std::memory_order_relaxed);
        }

        size_t get_all_rank_slices_call_count() const
        {
            return get_all_rank_slices_calls_.load(std::memory_order_relaxed);
        }

        size_t rank_has_work_call_count() const
        {
            return rank_has_work_calls_.load(std::memory_order_relaxed);
        }

        size_t get_device_slice_call_count() const
        {
            return get_device_slice_calls_.load(std::memory_order_relaxed);
        }

        size_t get_all_device_slices_call_count() const
        {
            return get_all_device_slices_calls_.load(std::memory_order_relaxed);
        }

        size_t get_device_for_element_call_count() const
        {
            return get_device_for_element_calls_.load(std::memory_order_relaxed);
        }

        size_t distribute_call_count() const
        {
            return distribute_calls_.load(std::memory_order_relaxed);
        }

        size_t get_primary_device_slice_call_count() const
        {
            return get_primary_device_slice_calls_.load(std::memory_order_relaxed);
        }

        size_t distribute_experts_call_count() const
        {
            return distribute_experts_calls_.load(std::memory_order_relaxed);
        }

        size_t route_tokens_call_count() const
        {
            return route_tokens_calls_.load(std::memory_order_relaxed);
        }

        size_t estimate_memory_call_count() const
        {
            return estimate_memory_calls_.load(std::memory_order_relaxed);
        }

        size_t get_element_counts_call_count() const
        {
            return get_element_counts_calls_.load(std::memory_order_relaxed);
        }

        size_t total_call_count() const
        {
            return get_rank_slice_call_count() +
                   get_all_rank_slices_call_count() +
                   rank_has_work_call_count() +
                   get_device_slice_call_count() +
                   get_all_device_slices_call_count() +
                   get_device_for_element_call_count() +
                   distribute_call_count() +
                   get_primary_device_slice_call_count() +
                   distribute_experts_call_count() +
                   route_tokens_call_count() +
                   estimate_memory_call_count() +
                   get_element_counts_call_count();
        }

        void reset_call_counts()
        {
            get_rank_slice_calls_.store(0, std::memory_order_relaxed);
            get_all_rank_slices_calls_.store(0, std::memory_order_relaxed);
            rank_has_work_calls_.store(0, std::memory_order_relaxed);
            get_device_slice_calls_.store(0, std::memory_order_relaxed);
            get_all_device_slices_calls_.store(0, std::memory_order_relaxed);
            get_device_for_element_calls_.store(0, std::memory_order_relaxed);
            distribute_calls_.store(0, std::memory_order_relaxed);
            get_primary_device_slice_calls_.store(0, std::memory_order_relaxed);
            distribute_experts_calls_.store(0, std::memory_order_relaxed);
            route_tokens_calls_.store(0, std::memory_order_relaxed);
            estimate_memory_calls_.store(0, std::memory_order_relaxed);
            get_element_counts_calls_.store(0, std::memory_order_relaxed);
        }

        // =========================================================================
        // Test Utilities - Configuration Access
        // =========================================================================

        const Config &config() const { return config_; }

        /**
         * @brief Update custom rank slice at runtime
         */
        void setCustomRankSlice(WorkSlice slice)
        {
            config_.use_custom_rank_slice = true;
            config_.custom_rank_slice = slice;
        }

        /**
         * @brief Clear custom rank slice override
         */
        void clearCustomRankSlice()
        {
            config_.use_custom_rank_slice = false;
        }

        /**
         * @brief Update custom device slices at runtime
         */
        void setCustomDeviceSlices(std::vector<WorkSlice> slices)
        {
            config_.use_custom_device_slices = true;
            config_.custom_device_slices = std::move(slices);
        }

        /**
         * @brief Clear custom device slices override
         */
        void clearCustomDeviceSlices()
        {
            config_.use_custom_device_slices = false;
        }

    private:
        Config config_;

        // Call counters (mutable for const methods)
        mutable std::atomic<size_t> get_rank_slice_calls_{0};
        mutable std::atomic<size_t> get_all_rank_slices_calls_{0};
        mutable std::atomic<size_t> rank_has_work_calls_{0};
        mutable std::atomic<size_t> get_device_slice_calls_{0};
        mutable std::atomic<size_t> get_all_device_slices_calls_{0};
        mutable std::atomic<size_t> get_device_for_element_calls_{0};
        mutable std::atomic<size_t> distribute_calls_{0};
        mutable std::atomic<size_t> get_primary_device_slice_calls_{0};
        mutable std::atomic<size_t> distribute_experts_calls_{0};
        mutable std::atomic<size_t> route_tokens_calls_{0};
        mutable std::atomic<size_t> estimate_memory_calls_{0};
        mutable std::atomic<size_t> get_element_counts_calls_{0};

        void validateConfig()
        {
            if (config_.rank < 0 || config_.rank >= config_.world_size)
            {
                throw std::invalid_argument("MockWorkDistributor: rank must be in [0, world_size)");
            }
            if (config_.world_size < 1)
            {
                throw std::invalid_argument("MockWorkDistributor: world_size must be >= 1");
            }
            if (config_.devices.empty())
            {
                // Provide default CPU device if none specified
                config_.devices.push_back(DeviceId::cpu());
            }
            if (!config_.device_weights.empty() &&
                config_.device_weights.size() != config_.devices.size())
            {
                throw std::invalid_argument("MockWorkDistributor: device_weights size must match devices size");
            }
        }

        /**
         * @brief Compute rank slice using standard even distribution
         */
        static WorkSlice computeRankSlice(size_t total_elements, int rank, int world_size)
        {
            size_t base_count = total_elements / world_size;
            size_t remainder = total_elements % world_size;

            size_t start = rank * base_count + std::min(static_cast<size_t>(rank), remainder);
            size_t count = base_count + (rank < static_cast<int>(remainder) ? 1 : 0);

            return WorkSlice{start, start + count, count, rank};
        }

        /**
         * @brief Get normalized device weights (sum to 1.0)
         */
        std::vector<float> getNormalizedWeights() const
        {
            if (config_.device_weights.empty())
            {
                // Equal weights
                float w = 1.0f / config_.devices.size();
                return std::vector<float>(config_.devices.size(), w);
            }

            // Normalize provided weights
            float sum = 0.0f;
            for (float w : config_.device_weights)
            {
                sum += w;
            }

            std::vector<float> normalized;
            normalized.reserve(config_.device_weights.size());
            for (float w : config_.device_weights)
            {
                normalized.push_back(w / sum);
            }
            return normalized;
        }
    };

} // namespace llaminar2::test
