/**
 * @file MPITopology.h
 * @brief MPI topology abstraction for distributed tensor-parallel inference
 *
 * Provides mapping between:
 * - MPI ranks (logical)
 * - Physical nodes (machines)
 * - CPU sockets/NUMA nodes
 * - GPUs/accelerators
 *
 * Also provides work distribution calculations for tensor parallelism.
 *
 * Design principles:
 * - ALL ranks participate in compute by default (including rank 0)
 * - Equal work division by default, with hooks for future heterogeneous distribution
 * - Uses existing TensorSlice/SliceMetadata from tensors/TensorSlice.h
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#pragma once

#include <mpi.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace llaminar2
{

    // Forward declarations - actual types in tensors/TensorSlice.h
    struct SliceMetadata;
    enum class SliceMode;

    // Forward declarations - placement types in execution/PlacementPlan.h
    struct PlacementPlan;
    struct PlacementInput;

    /**
     * @brief Compute capability descriptor for a single device
     */
    struct DeviceCapability
    {
        enum class Type
        {
            CPU,
            CUDA,
            ROCm,
            Unknown
        };

        Type type = Type::CPU;
        int device_id = 0;             ///< Device index (0 for CPU)
        size_t memory_bytes = 0;       ///< Total device memory
        size_t compute_units = 0;      ///< SM count (GPU) or core count (CPU)
        float relative_compute = 1.0f; ///< Relative compute power (for work distribution)
        std::string name;              ///< Device name string
    };

    /**
     * @brief Information about a single MPI rank's placement and capabilities
     */
    struct RankPlacement
    {
        int rank;             ///< MPI rank (0..world_size-1)
        int node_id;          ///< Physical node/machine (0..node_count-1)
        int local_rank;       ///< Rank within node (0..ranks_per_node-1)
        int socket_id;        ///< CPU socket within node
        int numa_node;        ///< NUMA node for memory affinity
        std::string hostname; ///< Node hostname

        /// Devices available to this rank
        std::vector<DeviceCapability> devices;

        /// Total relative compute power (sum of device capabilities)
        float total_compute_power() const
        {
            float total = 0.0f;
            for (const auto &d : devices)
            {
                total += d.relative_compute;
            }
            return total > 0.0f ? total : 1.0f; // At least 1.0 for CPU-only
        }
    };

    /**
     * @brief Work distribution range for a rank
     *
     * Represents a contiguous range [start, end) of work items.
     * Used for dividing heads, rows, columns, vocab, etc. across ranks.
     */
    struct WorkRange
    {
        size_t start = 0; ///< Start index (inclusive)
        size_t end = 0;   ///< End index (exclusive)

        /// Number of items in range
        size_t size() const { return end > start ? end - start : 0; }

        /// Check if range is empty
        bool empty() const { return start >= end; }

        /// Check if index is within range
        bool contains(size_t idx) const { return idx >= start && idx < end; }

        /// Create equal-division range for rank
        static WorkRange for_rank_equal(size_t total, int rank, int world_size)
        {
            if (world_size <= 0 || rank < 0 || rank >= world_size)
            {
                return {0, 0};
            }

            size_t per_rank = total / static_cast<size_t>(world_size);
            size_t remainder = total % static_cast<size_t>(world_size);

            // Distribute remainder across first 'remainder' ranks
            size_t r = static_cast<size_t>(rank);
            size_t start = r * per_rank + std::min(r, remainder);
            size_t extra = (r < remainder) ? 1 : 0;
            size_t end = start + per_rank + extra;

            return {start, end};
        }

        /// Create weighted range based on compute power
        static WorkRange for_rank_weighted(
            size_t total,
            int rank,
            const std::vector<float> &rank_weights)
        {
            if (rank_weights.empty() || rank < 0 ||
                static_cast<size_t>(rank) >= rank_weights.size())
            {
                return {0, 0};
            }

            // Calculate total weight
            float total_weight = 0.0f;
            for (float w : rank_weights)
            {
                total_weight += w;
            }
            if (total_weight <= 0.0f)
            {
                return for_rank_equal(total, rank, static_cast<int>(rank_weights.size()));
            }

            // Calculate cumulative start positions
            size_t start = 0;
            for (int r = 0; r < rank; ++r)
            {
                size_t items = static_cast<size_t>(
                    (rank_weights[r] / total_weight) * static_cast<float>(total) + 0.5f);
                start += items;
            }

            // Calculate this rank's end
            size_t end;
            if (rank == static_cast<int>(rank_weights.size()) - 1)
            {
                end = total; // Last rank gets remainder
            }
            else
            {
                size_t items = static_cast<size_t>(
                    (rank_weights[rank] / total_weight) * static_cast<float>(total) + 0.5f);
                end = std::min(start + items, total);
            }

            return {start, end};
        }
    };

    /**
     * @brief MPI topology abstraction for distributed inference
     *
     * Handles:
     * 1. Topology detection (nodes, sockets, devices)
     * 2. Communication patterns (intra-node, inter-node)
     * 3. Work distribution for tensor parallelism
     * 4. Device capability discovery and aggregation
     *
     * Design principles:
     * - ALL ranks (including rank 0) participate in compute by default
     * - Equal work division by default; future support for weighted distribution
     * - Integrates with existing SliceMetadata from tensors/TensorSlice.h
     *
     * Startup sequence:
     * 1. MPITopology is constructed on each rank
     * 2. Each rank detects its local devices/capabilities
     * 3. AllGather exchanges capabilities across all ranks
     * 4. Work distribution can then be computed (equal or weighted)
     *
     * Usage:
     * ```cpp
     * MPITopology topo(MPI_COMM_WORLD);
     *
     * // Get this rank's head range for attention
     * WorkRange heads = topo.get_head_range(14);  // 14 total heads
     * // Rank 0: [0, 7), Rank 1: [7, 14)
     *
     * // Create SliceMetadata for tensor parallelism
     * SliceMetadata meta = topo.createRowParallelMeta(4096, 896);
     *
     * // All ranks participate in compute
     * topo.is_compute_participant(); // true for all ranks by default
     * ```
     */
    class MPITopology
    {
    public:
        // =========================================================================
        // Construction
        // =========================================================================

        /**
         * @brief Construct topology from MPI communicator
         * @param comm MPI communicator (usually MPI_COMM_WORLD)
         *
         * Auto-detects:
         * - Node boundaries via MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)
         * - NUMA topology via NUMATopology class
         * - GPU/accelerator capabilities
         *
         * After construction, calls exchangeCapabilities() to gather
         * device info from all ranks.
         */
        explicit MPITopology(MPI_Comm comm = MPI_COMM_WORLD);

        /**
         * @brief Construct topology with explicit configuration (for testing)
         * @param rank This rank's ID
         * @param world_size Total ranks
         * @param ranks_per_node Ranks per physical node
         * @param comm Communicator (default MPI_COMM_WORLD)
         */
        MPITopology(int rank, int world_size, int ranks_per_node,
                    MPI_Comm comm = MPI_COMM_WORLD);

        /// Destructor (frees derived communicators)
        ~MPITopology();

        // Non-copyable, movable
        MPITopology(const MPITopology &) = delete;
        MPITopology &operator=(const MPITopology &) = delete;
        MPITopology(MPITopology &&) noexcept;
        MPITopology &operator=(MPITopology &&) noexcept;

        // =========================================================================
        // Basic Topology Queries
        // =========================================================================

        /// This rank's ID (0..world_size-1)
        int rank() const { return rank_; }

        /// Total number of MPI ranks
        int world_size() const { return world_size_; }

        /// Number of physical nodes (machines)
        int node_count() const { return node_count_; }

        /// Ranks per physical node (assumes uniform distribution)
        int ranks_per_node() const { return ranks_per_node_; }

        /// Get this rank's full placement information
        const RankPlacement &placement() const { return placement_; }

        /// Get placement for any rank (after capability exchange)
        const RankPlacement &get_placement(int rank) const;

        /// Check if this rank is the coordinator (rank 0)
        /// Note: Coordinator ALSO participates in compute by default
        bool is_coordinator() const { return rank_ == 0; }

        /// Check if this rank participates in compute (default: ALL ranks)
        bool is_compute_participant() const { return compute_participant_; }

        /// Set whether this rank participates in compute
        /// Use sparingly - typically all ranks should compute
        void set_compute_participant(bool participate) { compute_participant_ = participate; }

        /// Number of ranks that participate in compute
        int compute_world_size() const;

        // =========================================================================
        // Communication Patterns
        // =========================================================================

        /// Check if two ranks are on the same physical node
        bool same_node(int rank_a, int rank_b) const;

        /// Get communicator for ranks on same node (MPI_COMM_TYPE_SHARED)
        /// Returns MPI_COMM_NULL if not available
        MPI_Comm intra_node_comm() const { return intra_node_comm_; }

        /// Get communicator for leader ranks (one per node)
        /// Only valid on rank 0 of each node
        MPI_Comm inter_node_comm() const { return inter_node_comm_; }

        /// Get the global communicator
        MPI_Comm world_comm() const { return world_comm_; }

        /// Check if this rank is the leader of its node (local_rank == 0)
        bool is_node_leader() const { return placement_.local_rank == 0; }

        // =========================================================================
        // Tensor Parallelism: Work Distribution
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
        WorkRange get_head_range(int total_heads) const;

        /**
         * @brief Get KV head range for this rank (GQA-aware)
         * @param total_kv_heads Total KV heads (may differ from attention heads)
         * @return WorkRange for this rank's KV heads
         *
         * Note: If total_kv_heads < world_size, some ranks will have empty ranges.
         * In GQA, Q heads are sharded but KV heads may be replicated.
         */
        WorkRange get_kv_head_range(int total_kv_heads) const;

        /**
         * @brief Get column range for column-parallel GEMM
         * @param total_cols Total columns in weight matrix
         * @return WorkRange specifying column range for this rank
         *
         * Used for: Q/K/V projections, Gate/Up projections, LM Head
         * Each rank computes output[:, range.start:range.end]
         */
        WorkRange get_column_range(size_t total_cols) const;

        /**
         * @brief Get row range for row-parallel GEMM
         * @param total_rows Total rows in weight matrix
         * @return WorkRange specifying row range for this rank
         *
         * Used for: Wo projection, Down projection
         * Each rank has weight[range.start:range.end, :] and computes partial output
         */
        WorkRange get_row_range(size_t total_rows) const;

        /**
         * @brief Get vocabulary range for parallel LM head
         * @param vocab_size Total vocabulary size
         * @return WorkRange specifying vocab range for this rank
         */
        WorkRange get_vocab_range(size_t vocab_size) const;

        /**
         * @brief Get FFN intermediate dimension range
         * @param ffn_dim Total FFN intermediate dimension
         * @return WorkRange for this rank's FFN portion
         */
        WorkRange get_ffn_range(size_t ffn_dim) const;

        // =========================================================================
        // SliceMetadata Creation (integrates with TensorSlice.h)
        // =========================================================================

        /**
         * @brief Create SliceMetadata for row-parallel sharding
         * @param original_rows Total rows in original weight
         * @param original_cols Total cols in original weight
         * @param inner_is_presliced Whether inner tensor already has only slice data
         * @return SliceMetadata configured for this rank's row slice
         */
        SliceMetadata createRowParallelMeta(
            size_t original_rows,
            size_t original_cols,
            bool inner_is_presliced = false) const;

        /**
         * @brief Create SliceMetadata for column-parallel sharding
         * @param original_rows Total rows in original weight
         * @param original_cols Total cols in original weight
         * @param inner_is_presliced Whether inner tensor already has only slice data
         * @return SliceMetadata configured for this rank's column slice
         */
        SliceMetadata createColumnParallelMeta(
            size_t original_rows,
            size_t original_cols,
            bool inner_is_presliced = false) const;

        // =========================================================================
        // Device Capability Management
        // =========================================================================

        /**
         * @brief Exchange device capabilities with all ranks
         *
         * Called automatically during construction. Can be called again
         * if device configuration changes.
         *
         * After this call, get_placement(rank) returns complete info for any rank.
         */
        void exchangeCapabilities();

        /// Get all known rank placements (after capability exchange)
        const std::vector<RankPlacement> &all_placements() const { return all_placements_; }

        /// Get compute weights for all ranks (for weighted work distribution)
        std::vector<float> get_compute_weights() const;

        // =========================================================================
        // Placement Strategy
        // =========================================================================

        /**
         * @brief Compute a PlacementPlan for weight/compute distribution
         *
         * Uses the gathered device capabilities (from exchangeCapabilities())
         * along with model metadata to compute optimal placement.
         *
         * IMPORTANT: This is deterministic - all ranks compute the same plan.
         * No broadcast needed; each rank runs the same algorithm.
         *
         * @param architecture Model architecture name (e.g., "qwen2")
         * @param n_layers Number of transformer layers
         * @param d_model Hidden dimension
         * @param d_ff FFN intermediate dimension
         * @param vocab_size Vocabulary size
         * @param n_heads Attention heads
         * @param n_kv_heads KV heads (for GQA)
         * @param quant_type Quantization type (e.g., "Q4_0")
         * @param estimated_memory Total model memory estimate
         * @param strategy_name Strategy to use (empty = auto-select)
         * @return Computed PlacementPlan
         */
        PlacementPlan computePlacement(
            const std::string &architecture,
            int n_layers,
            size_t d_model,
            size_t d_ff,
            size_t vocab_size,
            size_t n_heads,
            size_t n_kv_heads,
            const std::string &quant_type,
            size_t estimated_memory,
            const std::string &strategy_name = "") const;

        /**
         * @brief Compute a PlacementPlan from a PlacementInput struct
         *
         * Alternative interface that takes a pre-populated PlacementInput.
         * The topology fields will be filled in automatically.
         *
         * @param input Placement input with model info (topology fields will be overwritten)
         * @return Computed PlacementPlan
         */
        PlacementPlan computePlacement(PlacementInput input) const;

        // =========================================================================
        // Device Mapping
        // =========================================================================

        /// Get primary compute device for this rank
        int get_device() const;

        /// Get all devices available to this rank
        const std::vector<DeviceCapability> &get_devices() const { return placement_.devices; }

        /// Check if this rank has GPU/accelerator access
        bool has_accelerator() const;

        // =========================================================================
        // Debugging / Logging
        // =========================================================================

        /// Get human-readable topology summary
        std::string to_string() const;

        /// Print topology to LOG_INFO (all ranks print in order)
        void print_topology() const;

    private:
        int rank_;
        int world_size_;
        int node_count_;
        int ranks_per_node_;
        bool compute_participant_ = true; ///< All ranks compute by default

        RankPlacement placement_;
        std::vector<RankPlacement> all_placements_; ///< Placements from all ranks

        MPI_Comm world_comm_;
        MPI_Comm intra_node_comm_;
        MPI_Comm inter_node_comm_;
        bool owns_comms_; ///< Whether we created the derived communicators

        /// Detect topology from MPI communicator
        void detect_topology();

        /// Setup derived communicators (intra-node, inter-node)
        void setup_communicators();

        /// Detect NUMA node and socket for this rank
        void detect_numa_placement();

        /// Detect GPU/accelerator capabilities for this rank
        void detect_device_capabilities();
    };

} // namespace llaminar2
