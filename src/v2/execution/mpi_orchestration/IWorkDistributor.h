/**
 * @file IWorkDistributor.h
 * @brief Interface for hierarchical work distribution across MPI ranks and devices
 *
 * Abstracts work distribution to enable:
 * 1. Unit testing without actual MPI/device topology
 * 2. Custom work distribution strategies
 * 3. Deterministic testing of distributed algorithms
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Abstract interface for hierarchical work distribution
     *
     * This interface abstracts work distribution across:
     * - MPI ranks (tensor parallelism)
     * - Devices within a rank (heterogeneous execution)
     * - MoE expert placement (expert parallelism)
     *
     * Implementations:
     * - WorkDistributor: Real implementation with configurable topology
     * - MockWorkDistributor: Test implementation with predefined slices
     */
    class IWorkDistributor
    {
    public:
        virtual ~IWorkDistributor() = default;

        // =========================================================================
        // Work Slice Types (shared with implementations)
        // =========================================================================

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

            bool operator==(const WorkSlice &other) const
            {
                return start == other.start && end == other.end &&
                       count == other.count && owner == other.owner;
            }
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

            bool operator==(const HierarchicalSlice &other) const
            {
                return rank == other.rank &&
                       device.type == other.device.type &&
                       device.ordinal == other.device.ordinal &&
                       global_start == other.global_start &&
                       global_end == other.global_end &&
                       local_start == other.local_start &&
                       local_count == other.local_count;
            }
        };

        /**
         * @brief Expert assignment for MoE architectures
         */
        struct ExpertAssignment
        {
            int expert_id;   ///< Expert index (0 to num_experts-1)
            DeviceId device; ///< Device that owns this expert
            int rank;        ///< MPI rank (for distributed experts)

            bool operator==(const ExpertAssignment &other) const
            {
                return expert_id == other.expert_id &&
                       device.type == other.device.type &&
                       device.ordinal == other.device.ordinal &&
                       rank == other.rank;
            }
        };

        /**
         * @brief Token routing for MoE dispatch
         */
        struct TokenRouting
        {
            int expert_id;                  ///< Target expert
            DeviceId device;                ///< Device hosting this expert
            std::vector<int> token_indices; ///< Tokens routed to this expert
            std::vector<float> weights;     ///< Router weights for combining
        };

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
        virtual WorkSlice getRankSlice(size_t total_elements) const = 0;

        /**
         * @brief Get work slices for all ranks
         * @param total_elements Total elements to distribute
         * @return Vector of WorkSlice, one per rank
         */
        virtual std::vector<WorkSlice> getAllRankSlices(size_t total_elements) const = 0;

        /**
         * @brief Check if this rank has any work
         * @param total_elements Total elements to distribute
         * @return true if getRankSlice().count > 0
         */
        virtual bool rankHasWork(size_t total_elements) const = 0;

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
        virtual WorkSlice getDeviceSlice(size_t rank_elements, DeviceId device) const = 0;

        /**
         * @brief Get work slices for all devices in this rank
         * @param rank_elements Elements assigned to this rank
         * @return Vector of WorkSlice, one per device
         */
        virtual std::vector<WorkSlice> getAllDeviceSlices(size_t rank_elements) const = 0;

        /**
         * @brief Get device index that should handle a specific element
         * @param element_idx Element index within rank's work
         * @param rank_elements Total elements for this rank
         * @return Device index in this rank
         */
        virtual int getDeviceForElement(size_t element_idx, size_t rank_elements) const = 0;

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
        virtual std::vector<HierarchicalSlice> distribute(size_t total_elements) const = 0;

        /**
         * @brief Get the hierarchical slice for this rank's primary device
         * @param total_elements Total elements to distribute
         * @return HierarchicalSlice for device 0 of this rank
         */
        virtual HierarchicalSlice getPrimaryDeviceSlice(size_t total_elements) const = 0;

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
        virtual std::vector<ExpertAssignment> distributeExperts(int num_experts) const = 0;

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
        virtual std::vector<TokenRouting> routeTokensToExperts(
            const float *router_output,
            const std::vector<ExpertAssignment> &expert_assignments,
            int top_k,
            int seq_len,
            int num_experts) const = 0;

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
        virtual size_t estimateMemoryPerDevice(size_t total_bytes) const = 0;

        /**
         * @brief Get element counts per device for allocations
         * @param total_elements Total elements
         * @return Vector of element counts, one per device
         */
        virtual std::vector<size_t> getElementCountsPerDevice(size_t total_elements) const = 0;

        // =========================================================================
        // Accessors
        // =========================================================================

        /**
         * @brief Get the world size (total MPI ranks)
         */
        virtual int worldSize() const = 0;

        /**
         * @brief Get this rank's index (0-indexed)
         */
        virtual int rank() const = 0;

        /**
         * @brief Get devices for this rank
         */
        virtual const std::vector<DeviceId> &devices() const = 0;

        /**
         * @brief Get number of devices in this rank
         */
        virtual size_t deviceCount() const = 0;

        /**
         * @brief Check if this rank has multiple devices
         */
        virtual bool hasMultipleDevices() const = 0;

        // =========================================================================
        // Static Utility Methods
        // =========================================================================

        /**
         * @brief Get experts assigned to a specific device
         * @param expert_assignments Full expert mapping
         * @param device Device to query
         * @return Expert IDs hosted on this device
         */
        static std::vector<int> getExpertsForDevice(
            const std::vector<ExpertAssignment> &expert_assignments,
            DeviceId device)
        {
            std::vector<int> experts;
            for (const auto &ea : expert_assignments)
            {
                if (ea.device.type == device.type && ea.device.ordinal == device.ordinal)
                {
                    experts.push_back(ea.expert_id);
                }
            }
            return experts;
        }

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create a mock work distributor for testing
         * @param rank Simulated rank
         * @param world_size Simulated world size
         * @param devices Devices for this rank (default: CPU only)
         * @return Mock distributor that implements IWorkDistributor
         */
        static std::shared_ptr<IWorkDistributor> createMock(
            int rank, int world_size,
            std::vector<DeviceId> devices = {DeviceId::cpu()});
    };

} // namespace llaminar2
