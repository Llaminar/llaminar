/**
 * @file ParallelismTree.h
 * @brief Recursive parallelism tree for hierarchical PP/TP composition
 *
 * A ParallelismTree describes an arbitrarily nested hierarchy of pipeline-parallel
 * and tensor-parallel groups across a distributed cluster. The tree has three
 * node types:
 *
 * - **PIPELINE_PARALLEL**: Children execute sequentially, each handling a
 *   contiguous range of transformer layers. Activations are transferred between
 *   children via MPI send/recv or local buffer handoff.
 *
 * - **TENSOR_PARALLEL**: Children execute the same layers in parallel with
 *   sharded weights. Requires allreduce collectives at appropriate synchronization
 *   points.
 *
 * - **DEVICE**: Leaf node representing a single compute device on a specific
 *   MPI rank. This is the actual hardware unit that runs kernels.
 *
 * The tree is compiled bottom-up into nested IInferenceRunner instances by
 * TreeToRunnerCompiler (Phase 5). Each PP node becomes a PipelineRunner, each
 * TP node becomes a MultiDeviceOrchestrator(TP mode), and each DEVICE leaf
 * becomes a DeviceGraphOrchestrator.
 *
 * Example: 2 machines × 2 sockets × 2 GPUs per socket
 * ```
 * PP(global)
 * ├── PP(host0)
 * │   ├── TP(socket0, rank=0, [cuda:0, cuda:1])
 * │   └── TP(socket1, rank=1, [cuda:0, cuda:1])
 * └── PP(host1)
 *     ├── TP(socket2, rank=2, [cuda:0, cuda:1])
 *     └── TP(socket3, rank=3, [cuda:0, cuda:1])
 * ```
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../../backends/GlobalDeviceAddress.h"
#include "../../config/CollectiveBackendType.h"
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // ParallelismNodeType
    // =========================================================================

    /**
     * @brief Type of node in the parallelism tree
     */
    enum class ParallelismNodeType
    {
        PIPELINE_PARALLEL, ///< Children execute sequentially (layer partitioning)
        TENSOR_PARALLEL,   ///< Children execute same layers in parallel (weight sharding)
        DEVICE             ///< Leaf: one compute device on one MPI rank
    };

    /**
     * @brief Convert node type to string
     */
    inline const char *parallelismNodeTypeName(ParallelismNodeType type)
    {
        switch (type)
        {
        case ParallelismNodeType::PIPELINE_PARALLEL:
            return "PP";
        case ParallelismNodeType::TENSOR_PARALLEL:
            return "TP";
        case ParallelismNodeType::DEVICE:
            return "DEVICE";
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // ParallelismNode
    // =========================================================================

    /**
     * @brief A node in the parallelism tree
     *
     * Can be a PP/TP interior node with children, or a DEVICE leaf.
     * Nodes are value types — move-only due to unique_ptr children.
     *
     * After tree construction, call assignLayers() on the root to assign
     * layer ranges, then validate() to check invariants.
     */
    struct ParallelismNode
    {
        // =====================================================================
        // Identity
        // =====================================================================

        ParallelismNodeType type = ParallelismNodeType::DEVICE;
        std::string name; ///< Human-readable name (e.g., "host0", "socket0_tp")

        // =====================================================================
        // Children (PP and TP nodes only)
        // =====================================================================

        std::vector<ParallelismNode> children;

        // =====================================================================
        // TP-specific configuration
        // =====================================================================

        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        std::vector<float> tp_weights; ///< Proportional weights (empty = equal split)

        // =====================================================================
        // Device leaf configuration
        // =====================================================================

        GlobalDeviceAddress device; ///< Valid only for DEVICE nodes
        int owning_rank = -1;       ///< MPI rank that owns this device

        // =====================================================================
        // Layer assignment (set by assignLayers())
        // =====================================================================

        int first_layer = -1; ///< First layer index (inclusive, 0-based)
        int last_layer = -1;  ///< Last layer index (inclusive, 0-based)

        // =====================================================================
        // Embedding / LM head flags (set by assignLayers())
        // =====================================================================

        bool has_embedding = false; ///< This subtree includes the embedding
        bool has_lm_head = false;   ///< This subtree includes the LM head

        // =====================================================================
        // Queries
        // =====================================================================

        /** @brief Number of layers assigned to this node */
        int layerCount() const
        {
            return (first_layer >= 0 && last_layer >= first_layer)
                       ? (last_layer - first_layer + 1)
                       : 0;
        }

        /** @brief Is this a leaf node? */
        bool isLeaf() const { return type == ParallelismNodeType::DEVICE; }

        /** @brief Is this an interior node? */
        bool isInterior() const { return !isLeaf(); }

        /**
         * @brief Collect all MPI ranks that own DEVICE leaves in this subtree
         * @return Set of unique rank IDs
         */
        std::set<int> leafRanks() const;

        /**
         * @brief Collect all DEVICE leaf nodes in this subtree
         * @return Vector of pointers to leaf nodes (non-owning)
         */
        std::vector<const ParallelismNode *> leafDevices() const;

        /**
         * @brief Does this subtree span multiple MPI ranks?
         * @return true if leafRanks().size() > 1
         */
        bool isCrossRank() const;

        /**
         * @brief Total number of DEVICE leaves in this subtree
         */
        int leafCount() const;

        /**
         * @brief Collect the set of DeviceTypes across all DEVICE leaves
         * @return Set of unique DeviceTypes (e.g., {CUDA}, {CUDA, ROCm})
         */
        std::set<DeviceType> leafDeviceTypes() const;

        /**
         * @brief Does this subtree contain devices from multiple GPU vendors?
         *
         * Returns true if the leaf devices include both CUDA and ROCm
         * (or any other combination of different GPU types). This is
         * the key indicator that HOST or HETEROGENEOUS backend is needed.
         *
         * @return true if leaves contain more than one GPU vendor type
         */
        bool isMixedVendor() const;

        /**
         * @brief Get the total TP degree by counting children (for TP nodes)
         * @return Number of children (or 1 for non-TP nodes)
         */
        int tpDegree() const
        {
            return (type == ParallelismNodeType::TENSOR_PARALLEL)
                       ? static_cast<int>(children.size())
                       : 1;
        }

        // =====================================================================
        // Equality (for testing)
        // =====================================================================

        bool operator==(const ParallelismNode &other) const;
        bool operator!=(const ParallelismNode &other) const { return !(*this == other); }
    };

    // =========================================================================
    // ParallelismTree
    // =========================================================================

    /**
     * @brief Root container for the parallelism tree
     *
     * Holds the root node plus model-level metadata (total layers, world size).
     * Provides tree-level operations: assignLayers, validate, toString.
     */
    struct ParallelismTree
    {
        ParallelismNode root;

        int total_layers = 0; ///< Total transformer layers in the model
        int world_size = 0;   ///< Total MPI ranks

        // =====================================================================
        // Layer Assignment
        // =====================================================================

        /**
         * @brief Assign layer ranges to all nodes in the tree
         *
         * Distributes layers top-down:
         * - PP nodes: split layers across children (equal or proportional)
         * - TP nodes: all children get the same layer range as the parent
         * - DEVICE leaves: inherit layer range from parent
         *
         * Also sets has_embedding (first_layer == 0) and has_lm_head
         * (last_layer == total_layers - 1) on leaf nodes.
         *
         * @param total_layers Total number of transformer layers
         */
        void assignLayers(int total_layers);

        // =====================================================================
        // Validation
        // =====================================================================

        /**
         * @brief Validate the tree for consistency
         *
         * Checks:
         * - PP children cover all parent layers (no gaps/overlaps)
         * - TP children all have the same layer range
         * - DEVICE leaves have valid owning_rank within [0, world_size)
         * - TP nodes have ≥ 2 children
         * - At least one DEVICE leaf exists
         * - All layer ranges are within [0, total_layers)
         *
         * @return List of error messages (empty = valid)
         */
        std::vector<std::string> validate() const;

        // =====================================================================
        // Display
        // =====================================================================

        /**
         * @brief Human-readable indented tree visualization
         * @return Multi-line string showing the tree structure
         */
        std::string toString() const;
    };

    // =========================================================================
    // Fluent Builder Functions
    // =========================================================================

    /**
     * @brief Create a pipeline-parallel node with children
     *
     * Example:
     * ```cpp
     * auto tree_root = PP("global", {
     *     PP("host0", { TP("s0", {cuda(0), cuda(1)}, 0), TP("s1", {cuda(0), cuda(1)}, 1) }),
     *     PP("host1", { TP("s0", {cuda(0), cuda(1)}, 2), TP("s1", {cuda(0), cuda(1)}, 3) })
     * });
     * ```
     */
    ParallelismNode PP(std::string name, std::vector<ParallelismNode> children);

    /**
     * @brief Create a tensor-parallel node from device addresses on a single rank
     *
     * Convenience for the common case where all TP devices belong to one MPI rank.
     *
     * @param name Human-readable name
     * @param devices List of device addresses (all same rank)
     * @param owning_rank MPI rank that owns all devices
     * @param backend Collective backend for allreduce (default AUTO)
     */
    ParallelismNode TP(std::string name,
                       std::vector<GlobalDeviceAddress> devices,
                       int owning_rank,
                       CollectiveBackendType backend = CollectiveBackendType::AUTO);

    /**
     * @brief Create a tensor-parallel node with arbitrary child nodes
     *
     * For cross-rank TP or nested TP (each child can be a subtree).
     *
     * @param name Human-readable name
     * @param children Child nodes (one per TP participant)
     * @param backend Collective backend for allreduce (default AUTO)
     */
    ParallelismNode TP(std::string name,
                       std::vector<ParallelismNode> children,
                       CollectiveBackendType backend = CollectiveBackendType::AUTO);

    /**
     * @brief Create a single-device leaf node
     *
     * @param device Device address
     * @param owning_rank MPI rank that owns the device
     */
    ParallelismNode Device(GlobalDeviceAddress device, int owning_rank);

} // namespace llaminar2
