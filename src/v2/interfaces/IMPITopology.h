/**
 * @file IMPITopology.h
 * @brief Interface for MPI topology discovery and work distribution
 *
 * Abstracts MPI topology detection to enable:
 * 1. Unit testing with deterministic topologies
 * 2. Simulating heterogeneous clusters (CPU-only, GPU, mixed)
 * 3. Testing work distribution without real MPI
 * 4. Mocking multi-node configurations
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

namespace llaminar2
{

    // Forward declarations - actual types in MPITopology.h and DeviceInventory.h
    struct RankPlacement;
    struct DeviceCapability;
    struct WorkRange;
    struct ClusterInventory;
    struct SliceMetadata;
    struct PlacementInput;
    struct PlacementPlan;

    /**
     * @brief Abstract interface for MPI topology discovery and work distribution
     *
     * This interface abstracts topology detection and work distribution to enable:
     * - Unit testing of distributed logic without real MPI runtime
     * - Simulating heterogeneous cluster configurations
     * - Testing tensor parallelism work distribution algorithms
     * - Mocking multi-node topologies with different device configurations
     *
     * Implementations:
     * - MPITopology: Real MPI-backed implementation with auto-detection
     * - MockMPITopology: Test implementation with configurable topologies
     *
     * Usage:
     * @code
     * // In production code
     * auto topology = IMPITopology::discover(mpi_context);
     * WorkRange heads = topology->get_head_range(14);  // Get this rank's head range
     *
     * // In tests
     * auto mock = IMPITopology::createMock(0, 2, {placement_rank0, placement_rank1});
     * WorkRange heads = mock->get_head_range(14);  // Deterministic for testing
     * @endcode
     */
    class IMPITopology
    {
    public:
        virtual ~IMPITopology() = default;

        // =========================================================================
        // Identity
        // =========================================================================

        /**
         * @brief Get the MPI rank of this process
         * @return Rank (0-indexed)
         */
        virtual int rank() const = 0;

        /**
         * @brief Get the total number of MPI processes
         * @return World size
         */
        virtual int world_size() const = 0;

        // =========================================================================
        // Topology Information
        // =========================================================================

        /**
         * @brief Get the number of physical nodes (machines) in the cluster
         * @return Node count
         */
        virtual int node_count() const = 0;

        /**
         * @brief Get the number of ranks per physical node
         * @return Ranks per node (assumes uniform distribution)
         */
        virtual int ranks_per_node() const = 0;

        /**
         * @brief Get this rank's placement information
         * @return RankPlacement with node_id, local_rank, devices, etc.
         */
        virtual const RankPlacement &placement() const = 0;

        /**
         * @brief Get placement information for any rank
         * @param rank Rank to query (0..world_size-1)
         * @return RankPlacement for the specified rank
         */
        virtual const RankPlacement &get_placement(int rank) const = 0;

        /**
         * @brief Get all known rank placements
         * @return Vector of RankPlacement for all ranks
         */
        virtual const std::vector<RankPlacement> &all_placements() const = 0;

        // =========================================================================
        // Role Queries
        // =========================================================================

        /**
         * @brief Check if this rank is the coordinator (rank 0)
         * @return true if rank == 0
         *
         * Note: The coordinator ALSO participates in compute by default.
         */
        virtual bool is_coordinator() const = 0;

        /**
         * @brief Check if this rank participates in compute
         * @return true if rank should perform computation (default: all ranks)
         */
        virtual bool is_compute_participant() const = 0;

        /**
         * @brief Check if this rank is the leader of its node (local_rank == 0)
         * @return true if this rank is node leader
         */
        virtual bool is_node_leader() const = 0;

        // =========================================================================
        // Work Distribution (Tensor Parallelism)
        // =========================================================================

        /**
         * @brief Get attention head range for this rank
         * @param total_heads Total attention heads in model
         * @return WorkRange specifying which heads this rank computes
         *
         * Example: 14 heads, 2 ranks:
         * - Rank 0: [0, 7)
         * - Rank 1: [7, 14)
         */
        virtual WorkRange get_head_range(int total_heads) const = 0;

        /**
         * @brief Get KV head range for this rank (GQA-aware)
         * @param total_kv_heads Total KV heads (may differ from attention heads)
         * @return WorkRange for this rank's KV heads
         */
        virtual WorkRange get_kv_head_range(int total_kv_heads) const = 0;

        /**
         * @brief Get column range for column-parallel GEMM
         * @param total_cols Total columns in weight matrix
         * @return WorkRange specifying column range for this rank
         *
         * Used for: Q/K/V projections, Gate/Up projections, LM Head
         */
        virtual WorkRange get_column_range(size_t total_cols) const = 0;

        /**
         * @brief Get row range for row-parallel GEMM
         * @param total_rows Total rows in weight matrix
         * @return WorkRange specifying row range for this rank
         *
         * Used for: Wo projection, Down projection
         */
        virtual WorkRange get_row_range(size_t total_rows) const = 0;

        /**
         * @brief Get vocabulary range for parallel LM head
         * @param vocab_size Total vocabulary size
         * @return WorkRange specifying vocab range for this rank
         */
        virtual WorkRange get_vocab_range(size_t vocab_size) const = 0;

        /**
         * @brief Get FFN intermediate dimension range
         * @param ffn_dim Total FFN intermediate dimension
         * @return WorkRange for this rank's FFN portion
         */
        virtual WorkRange get_ffn_range(size_t ffn_dim) const = 0;

        // =========================================================================
        // Capability Information
        // =========================================================================

        /**
         * @brief Get the cluster-wide device inventory
         * @return ClusterInventory with all device information across all ranks
         *
         * Note: Only valid after capability exchange (done in MPITopology constructor)
         */
        virtual const ClusterInventory &clusterInventory() const = 0;

        /**
         * @brief Get compute weights for all ranks
         * @return Vector of compute weights (for weighted work distribution)
         */
        virtual std::vector<float> get_compute_weights() const = 0;

        /**
         * @brief Check if this rank has GPU/accelerator access
         * @return true if rank has at least one GPU
         */
        virtual bool has_accelerator() const = 0;

        /**
         * @brief Get primary compute device for this rank
         * @return Device index (0 for CPU, >0 for GPU)
         */
        virtual int get_device() const = 0;

        /**
         * @brief Get all devices available to this rank
         * @return Vector of DeviceCapability
         */
        virtual const std::vector<DeviceCapability> &get_devices() const = 0;

        // =========================================================================
        // Placement Strategy (for orchestration wiring)
        // =========================================================================

        /**
         * @brief Compute a PlacementPlan for the given model input
         *
         * This method integrates with PlacementStrategyFactory to determine
         * optimal device placement based on model characteristics and cluster
         * capabilities.
         *
         * @param input PlacementInput with model info and constraints
         * @return PlacementPlan specifying device placement for all layers
         */
        virtual PlacementPlan computePlacement(const PlacementInput &input) const = 0;

        // =========================================================================
        // SliceMetadata Creation
        // =========================================================================

        /**
         * @brief Create SliceMetadata for row-parallel sharding
         * @param original_rows Total rows in original weight
         * @param original_cols Total cols in original weight
         * @param inner_is_presliced Whether inner tensor already has only slice data
         * @return SliceMetadata configured for this rank's row slice
         */
        virtual SliceMetadata createRowParallelMeta(
            size_t original_rows,
            size_t original_cols,
            bool inner_is_presliced = false) const = 0;

        /**
         * @brief Create SliceMetadata for column-parallel sharding
         * @param original_rows Total rows in original weight
         * @param original_cols Total cols in original weight
         * @param inner_is_presliced Whether inner tensor already has only slice data
         * @return SliceMetadata configured for this rank's column slice
         */
        virtual SliceMetadata createColumnParallelMeta(
            size_t original_rows,
            size_t original_cols,
            bool inner_is_presliced = false) const = 0;

        // =========================================================================
        // Debugging
        // =========================================================================

        /**
         * @brief Get human-readable topology summary
         * @return String describing the topology
         */
        virtual std::string to_string() const = 0;

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create a mock topology for testing
         * @param rank Simulated rank
         * @param world_size Simulated world size
         * @param placements Vector of RankPlacement for all ranks
         * @return Mock topology implementing IMPITopology
         */
        static std::shared_ptr<IMPITopology> createMock(
            int rank,
            int world_size,
            const std::vector<RankPlacement> &placements);
    };

} // namespace llaminar2
