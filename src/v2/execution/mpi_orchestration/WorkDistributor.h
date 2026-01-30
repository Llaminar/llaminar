/**
 * @file WorkDistributor.h
 * @brief Hierarchical work distribution: World → Rank → Device → Thread
 * @author David Sanftenberg
 * @date December 2025
 *
 * WorkDistributor computes how work should be split across the MPI/device/thread
 * hierarchy WITHOUT performing any actual work. It's a pure computation of indices.
 *
 * Example usage:
 * @code
 *   WorkDistributor dist({
 *       .world_size = 2,
 *       .rank = 0,
 *       .devices = {0, 1},  // CPU + GPU
 *   });
 *
 *   // Split 4096 output features across ranks
 *   auto rank_slice = dist.getRankSlice(4096);
 *   // rank_slice.start = 0, .end = 2048, .count = 2048
 *
 *   // Further split this rank's work across devices
 *   auto dev_slices = dist.getAllDeviceSlices(rank_slice.count);
 *   // dev_slices[0] = CPU slice, dev_slices[1] = GPU slice
 * @endcode
 */

#pragma once

#include "IWorkDistributor.h"
#include "../../backends/DeviceId.h"
#include <cstddef>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Hierarchical work distribution across MPI ranks and devices
     *
     * Computes work slices at each level:
     * 1. World level: Split across MPI ranks (tensor parallelism)
     * 2. Rank level: Split across devices within a rank (CPU + GPUs)
     * 3. Device level: Split across threads/streams (OpenMP / CUDA)
     *
     * All splits use contiguous ranges for cache efficiency.
     */
    class WorkDistributor : public IWorkDistributor
    {
    public:
        /**
         * @brief Configuration for work distribution
         */
        struct Config
        {
            int world_size = 1;                ///< Total MPI ranks
            int rank = 0;                      ///< This rank's index (0-indexed)
            std::vector<DeviceId> devices;     ///< Devices for this rank (e.g., {cpu(), cuda(0)})
            std::vector<float> device_weights; ///< Relative compute power per device (default: equal)
        };

        // Use types from IWorkDistributor interface
        using WorkSlice = IWorkDistributor::WorkSlice;
        using HierarchicalSlice = IWorkDistributor::HierarchicalSlice;
        using ExpertAssignment = IWorkDistributor::ExpertAssignment;
        using TokenRouting = IWorkDistributor::TokenRouting;

        // =========================================================================
        // Construction
        // =========================================================================

        /**
         * @brief Construct with configuration
         * @param config Distribution configuration
         */
        explicit WorkDistributor(Config config);

        /**
         * @brief Convenience constructor for single-device per rank
         * @param world_size Total MPI ranks
         * @param rank This rank's index
         * @param device Device (DeviceId::cpu() = no device, just MPI distribution)
         */
        WorkDistributor(int world_size, int rank, DeviceId device = DeviceId::cpu());

        // =========================================================================
        // Rank-Level Distribution (Tensor Parallelism)
        // =========================================================================

        /**
         * @brief Get work slice for this rank
         *
         * Splits total_elements evenly across world_size ranks.
         * Last rank gets any remainder.
         *
         * @param total_elements Total elements to distribute
         * @return WorkSlice for this rank
         */
        WorkSlice getRankSlice(size_t total_elements) const override;

        /**
         * @brief Get work slices for all ranks
         * @param total_elements Total elements to distribute
         * @return Vector of WorkSlice, one per rank
         */
        std::vector<WorkSlice> getAllRankSlices(size_t total_elements) const override;

        /**
         * @brief Check if this rank has any work
         * @param total_elements Total elements to distribute
         * @return true if getRankSlice().count > 0
         */
        bool rankHasWork(size_t total_elements) const override;

        // =========================================================================
        // Device-Level Distribution (Heterogeneous Execution)
        // =========================================================================

        /**
         * @brief Get work slice for a specific device within this rank
         *
         * Splits rank_elements across devices according to device_weights.
         * If no weights specified, splits evenly.
         *
         * @param rank_elements Elements assigned to this rank
         * @param device Device identifier
         * @return WorkSlice for the specified device
         */
        WorkSlice getDeviceSlice(size_t rank_elements, DeviceId device) const override;

        /**
         * @brief Get work slices for all devices in this rank
         * @param rank_elements Elements assigned to this rank
         * @return Vector of WorkSlice, one per device
         */
        std::vector<WorkSlice> getAllDeviceSlices(size_t rank_elements) const override;

        /**
         * @brief Get device index that should handle a specific element
         * @param element_idx Element index within rank's work
         * @param rank_elements Total elements for this rank
         * @return Device index in this rank
         */
        int getDeviceForElement(size_t element_idx, size_t rank_elements) const override;

        // =========================================================================
        // Full Hierarchy Distribution
        // =========================================================================

        /**
         * @brief Distribute work across entire hierarchy (rank + device)
         *
         * Returns the slice assigned to each (rank, device) pair in the world.
         * Only includes this rank's devices in detail.
         *
         * @param total_elements Total elements to distribute
         * @return Vector of HierarchicalSlice for this rank's devices
         */
        std::vector<HierarchicalSlice> distribute(size_t total_elements) const override;

        /**
         * @brief Get the hierarchical slice for this rank's primary device
         * @param total_elements Total elements to distribute
         * @return HierarchicalSlice for device 0 of this rank
         */
        HierarchicalSlice getPrimaryDeviceSlice(size_t total_elements) const override;

        // =========================================================================
        // MoE Expert Distribution
        // =========================================================================

        /**
         * @brief Distribute experts across devices (Expert Parallelism)
         *
         * Maps each expert to a device based on device weights/capacity.
         * Used at model load time to determine expert placement.
         *
         * @param num_experts Total number of experts in the MoE layer
         * @return Vector of ExpertAssignment mapping experts to devices
         */
        std::vector<ExpertAssignment> distributeExperts(int num_experts) const override;

        /**
         * @brief Route tokens to experts based on router output
         *
         * Given router probabilities, determines which tokens go to which
         * expert/device. Used at inference time for dynamic dispatch.
         *
         * @param router_output Router scores [seq_len, num_experts]
         * @param expert_assignments Pre-computed expert-to-device mapping
         * @param top_k Number of experts per token
         * @param seq_len Sequence length
         * @param num_experts Total experts
         * @return Vector of TokenRouting, one per active expert
         */
        std::vector<TokenRouting> routeTokensToExperts(
            const float *router_output,
            const std::vector<ExpertAssignment> &expert_assignments,
            int top_k,
            int seq_len,
            int num_experts) const override;

        /**
         * @brief Get experts assigned to a specific device
         * @param expert_assignments Full expert mapping
         * @param device Device to query
         * @return Expert IDs hosted on this device
         */
        static std::vector<int> getExpertsForDevice(
            const std::vector<ExpertAssignment> &expert_assignments,
            DeviceId device);

        // =========================================================================
        // Utility Methods
        // =========================================================================

        /**
         * @brief Estimate memory per device for a given total
         *
         * Useful for deciding if data fits in GPU memory.
         *
         * @param total_bytes Total bytes to distribute
         * @return Bytes per device (average if weighted)
         */
        size_t estimateMemoryPerDevice(size_t total_bytes) const override;

        /**
         * @brief Get element counts per device for allocations
         * @param total_elements Total elements
         * @return Vector of element counts, one per device
         */
        std::vector<size_t> getElementCountsPerDevice(size_t total_elements) const override;

        // =========================================================================
        // Accessors
        // =========================================================================

        int worldSize() const override { return config_.world_size; }
        int rank() const override { return config_.rank; }
        const std::vector<DeviceId> &devices() const override { return config_.devices; }
        size_t deviceCount() const override { return config_.devices.size(); }
        bool hasMultipleDevices() const override { return config_.devices.size() > 1; }

    private:
        Config config_;

        // Compute normalized device weights (sum to 1.0)
        std::vector<float> getNormalizedWeights() const;
    };

} // namespace llaminar2
