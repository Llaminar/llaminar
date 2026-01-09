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

#include "../backends/DeviceId.h"
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
    class WorkDistributor
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

        /**
         * @brief A contiguous slice of work
         */
        struct WorkSlice
        {
            size_t start = 0; ///< First element (inclusive)
            size_t end = 0;   ///< Last element (exclusive)
            size_t count = 0; ///< Number of elements (end - start)
            int owner = -1;   ///< Owner index (rank or device)

            bool empty() const { return count == 0; }
            bool contains(size_t idx) const { return idx >= start && idx < end; }
        };

        /**
         * @brief Full hierarchical slice with both rank and device info
         */
        struct HierarchicalSlice
        {
            int rank;            ///< MPI rank that owns this slice
            DeviceId device;     ///< Device within the rank
            size_t global_start; ///< Start offset in global work
            size_t global_end;   ///< End offset in global work (exclusive)
            size_t local_start;  ///< Start offset within this device's portion
            size_t local_count;  ///< Elements assigned to this device
        };

        /**
         * @brief Expert assignment for MoE architectures
         *
         * Maps experts to devices for expert-parallel execution.
         */
        struct ExpertAssignment
        {
            int expert_id;   ///< Expert index (0 to num_experts-1)
            DeviceId device; ///< Device that owns this expert
            int rank;        ///< MPI rank (for distributed experts)
        };

        /**
         * @brief Token routing for MoE dispatch
         *
         * After router selects top-k experts per token, this struct tracks
         * which tokens go to which expert/device.
         */
        struct TokenRouting
        {
            int expert_id;                  ///< Target expert
            DeviceId device;                ///< Device hosting this expert
            std::vector<int> token_indices; ///< Tokens routed to this expert
            std::vector<float> weights;     ///< Router weights for combining
        };

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
        WorkSlice getRankSlice(size_t total_elements) const;

        /**
         * @brief Get work slices for all ranks
         * @param total_elements Total elements to distribute
         * @return Vector of WorkSlice, one per rank
         */
        std::vector<WorkSlice> getAllRankSlices(size_t total_elements) const;

        /**
         * @brief Check if this rank has any work
         * @param total_elements Total elements to distribute
         * @return true if getRankSlice().count > 0
         */
        bool rankHasWork(size_t total_elements) const;

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
        WorkSlice getDeviceSlice(size_t rank_elements, DeviceId device) const;

        /**
         * @brief Get work slices for all devices in this rank
         * @param rank_elements Elements assigned to this rank
         * @return Vector of WorkSlice, one per device
         */
        std::vector<WorkSlice> getAllDeviceSlices(size_t rank_elements) const;

        /**
         * @brief Get device index that should handle a specific element
         * @param element_idx Element index within rank's work
         * @param rank_elements Total elements for this rank
         * @return Device index in this rank
         */
        int getDeviceForElement(size_t element_idx, size_t rank_elements) const;

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
        std::vector<HierarchicalSlice> distribute(size_t total_elements) const;

        /**
         * @brief Get the hierarchical slice for this rank's primary device
         * @param total_elements Total elements to distribute
         * @return HierarchicalSlice for device 0 of this rank
         */
        HierarchicalSlice getPrimaryDeviceSlice(size_t total_elements) const;

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
        std::vector<ExpertAssignment> distributeExperts(int num_experts) const;

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
            int num_experts) const;

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
        size_t estimateMemoryPerDevice(size_t total_bytes) const;

        /**
         * @brief Get element counts per device for allocations
         * @param total_elements Total elements
         * @return Vector of element counts, one per device
         */
        std::vector<size_t> getElementCountsPerDevice(size_t total_elements) const;

        // =========================================================================
        // Accessors
        // =========================================================================

        int worldSize() const { return config_.world_size; }
        int rank() const { return config_.rank; }
        const std::vector<DeviceId> &devices() const { return config_.devices; }
        size_t deviceCount() const { return config_.devices.size(); }
        bool hasMultipleDevices() const { return config_.devices.size() > 1; }

    private:
        Config config_;

        // Compute normalized device weights (sum to 1.0)
        std::vector<float> getNormalizedWeights() const;
    };

} // namespace llaminar2
