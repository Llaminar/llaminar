/**
 * @file HeterogeneousMultiDomainStrategy.h
 * @brief Strategy for heterogeneous multi-domain tensor and pipeline parallelism
 *
 * This strategy generates a parallelization plan for clusters with heterogeneous
 * hardware (mixed NVIDIA/AMD GPUs, multi-socket CPUs) using a hierarchical approach:
 *
 * 1. **Pipeline Parallel (PP)** across physical nodes (InfiniBand)
 * 2. **Tensor Parallel (TP)** within nodes:
 *    - GPU domains: NVIDIA↔AMD via HOST (intra-rank)
 *    - CPU domains: CPUs across ranks via MPI over UPI (cross-rank)
 *
 * Example 4-rank, 2-node cluster:
 *   Node 0: Ranks 0-1
 *     - GPU_TP_0 (Rank 0's GPUs, HOST) → layers 0-4
 *     - GPU_TP_1 (Rank 1's GPUs, HOST) → layers 5-9
 *     - CPU_TP (Ranks 0+1 CPUs, MPI/UPI) → layers 10-13
 *   Node 1: Ranks 2-3 (similar structure)
 *
 * The strategy is deterministic: all ranks compute identical plans.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../PlacementStrategy.h"
#include "../DeviceInventory.h"
#include "../../../config/TPDomain.h"
#include <vector>
#include <string>
#include <cstdint>

namespace llaminar2
{

    /**
     * @brief Assignment of a TP domain for layer execution
     *
     * Describes which devices and ranks participate in a domain,
     * and which layers that domain is responsible for.
     */
    struct DomainAssignment
    {
        int domain_id = -1;                               ///< Unique domain ID within the plan
        TPDomainType type = TPDomainType::GPU_INTRA_RANK; ///< Domain type (GPU or CPU)
        std::vector<int> ranks;                           ///< MPI ranks participating in this domain
        std::vector<DeviceId> devices;                    ///< Devices in this domain
        int node_id = -1;                                 ///< Physical node ID (-1 for cross-node)
        int layer_start = 0;                              ///< First layer (inclusive)
        int layer_end = 0;                                ///< Last layer (exclusive)
        float compute_weight = 0.0f;                      ///< Relative compute power of this domain

        /// Get number of layers assigned to this domain
        int layerCount() const { return layer_end - layer_start; }

        /// Check if domain is valid (has layers and devices)
        bool isValid() const
        {
            return domain_id >= 0 &&
                   layer_end > layer_start &&
                   !devices.empty() &&
                   !ranks.empty();
        }

        /// String representation for logging
        std::string toString() const;
    };

    /**
     * @brief Configuration options for heterogeneous multi-domain strategy
     */
    struct HeterogeneousConfig
    {
        bool enable_gpu_tp = true;            ///< Use GPU tensor parallelism
        bool enable_cpu_tp = true;            ///< Use CPU tensor parallelism
        float cpu_compute_fraction = 0.2f;    ///< Fraction of layers for CPU domains (0.0-1.0)
        int min_layers_per_domain = 2;        ///< Minimum layers per domain
        bool prefer_gpu_early_layers = true;  ///< Put GPU domains on earlier layers (better for prefill)
        bool enable_pipeline_parallel = true; ///< Enable cross-node pipeline parallelism
    };

    /**
     * @brief Pipeline stage representing layers on a physical node
     */
    struct PipelineStage
    {
        int node_id = -1;                      ///< Physical node ID
        int stage_id = -1;                     ///< Stage index (0 = first)
        std::vector<int> ranks;                ///< Ranks on this node
        std::vector<DomainAssignment> domains; ///< TP domains within this stage
        int layer_start = 0;                   ///< First layer (inclusive)
        int layer_end = 0;                     ///< Last layer (exclusive)
    };

    /**
     * @brief Complete heterogeneous parallelism plan
     *
     * Contains all information needed to set up:
     * - Pipeline parallel stages (cross-node)
     * - Tensor parallel domains (within-node)
     * - Layer-to-domain mappings
     */
    struct HeterogeneousPlan
    {
        std::vector<PipelineStage> stages;     ///< PP stages (one per node)
        std::vector<DomainAssignment> domains; ///< All TP domains (flattened)
        int total_layers = 0;                  ///< Total model layers
        int world_size = 0;                    ///< Total MPI ranks
        int node_count = 0;                    ///< Physical nodes

        /// Get domain assignment for a specific layer
        const DomainAssignment *getDomainForLayer(int layer_idx) const;

        /// Get pipeline stage for a specific layer
        const PipelineStage *getStageForLayer(int layer_idx) const;

        /// Serialize plan for MPI broadcast
        std::vector<uint8_t> serialize() const;

        /// Deserialize plan from MPI broadcast
        static HeterogeneousPlan deserialize(const std::vector<uint8_t> &data);

        /// String representation for logging
        std::string toString() const;
    };

    /**
     * @brief Heterogeneous multi-domain placement strategy
     *
     * Generates placement plans for clusters with mixed GPU types (NVIDIA/AMD)
     * and multi-socket CPUs. Uses hierarchical parallelism:
     *
     * - **Level 1**: Pipeline Parallel across physical nodes
     * - **Level 2**: Tensor Parallel within nodes
     *   - GPU domains: Per-rank GPUs (HOST)
     *   - CPU domains: CPUs across ranks (MPI/UPI)
     *
     * Key features:
     * - Deterministic: All ranks compute identical plans
     * - Heterogeneous-aware: Balances work by compute capability
     * - Phase-aware: GPU domains prioritized for compute-bound prefill
     */
    class HeterogeneousMultiDomainStrategy : public LayerPlacementStrategy
    {
    public:
        /**
         * @brief Construct with default configuration
         */
        HeterogeneousMultiDomainStrategy() = default;

        /**
         * @brief Construct with custom configuration
         * @param config Strategy configuration
         */
        explicit HeterogeneousMultiDomainStrategy(HeterogeneousConfig config);

        /**
         * @brief Compute placement plan from inputs
         * @param input Model and topology information
         * @return Computed placement plan
         *
         * Generates a hierarchical plan with:
         * - Layer assignments to TP domains
         * - Device mappings for weights/compute
         * - Domain metadata for collective configuration
         */
        PlacementPlan compute(const PlacementInput &input) const override;

        /**
         * @brief Get strategy name
         */
        std::string name() const override { return "HeterogeneousMultiDomain"; }

        /**
         * @brief Check if strategy is applicable
         * @param input Model and topology information
         * @return true if cluster has heterogeneous hardware worth exploiting
         */
        bool isApplicable(const PlacementInput &input) const override;

        /**
         * @brief Compute domain assignments from cluster inventory
         * @param inventory Complete cluster device information
         * @param n_layers Number of model layers
         * @return Vector of domain assignments
         *
         * This is the core algorithm that:
         * 1. Detects topology (nodes, ranks per node, GPU types)
         * 2. Creates GPU domains (per-rank, intra-rank TP)
         * 3. Creates CPU domains (per-node, cross-rank TP)
         * 4. Assigns layers proportional to compute weight
         */
        std::vector<DomainAssignment> computeDomainAssignments(
            const ClusterInventory &inventory,
            int n_layers) const;

        /**
         * @brief Generate full heterogeneous plan
         * @param inventory Complete cluster device information
         * @param n_layers Number of model layers
         * @return Complete plan with PP stages and TP domains
         */
        HeterogeneousPlan generatePlan(
            const ClusterInventory &inventory,
            int n_layers) const;

        /**
         * @brief Get current configuration
         */
        const HeterogeneousConfig &config() const { return config_; }

    private:
        HeterogeneousConfig config_;

        /**
         * @brief Detect GPU types present in a rank
         * @param rank_inv Rank inventory
         * @return Pair of (has_nvidia, has_amd)
         */
        std::pair<bool, bool> detectGPUTypes(const RankInventory &rank_inv) const;

        /**
         * @brief Calculate total compute weight for a rank's GPUs
         * @param rank_inv Rank inventory
         * @return Compute weight (sum of GPU tflops)
         */
        float calculateGPUComputeWeight(const RankInventory &rank_inv) const;

        /**
         * @brief Calculate compute weight for a node's CPUs
         * @param node_inv Node inventory
         * @param cluster_inv Cluster inventory (for rank details)
         * @return Compute weight based on CPU cores and bandwidth
         */
        float calculateCPUComputeWeight(
            const NodeInventory &node_inv,
            const ClusterInventory &cluster_inv) const;

        /**
         * @brief Distribute layers across domains by compute weight
         * @param domains Vector of domains to assign layers to
         * @param n_layers Total layers to distribute
         * @param gpu_domains_first Whether to assign GPU domains first (early layers)
         */
        void distributeLayers(
            std::vector<DomainAssignment> &domains,
            int n_layers,
            bool gpu_domains_first) const;

        /**
         * @brief Create GPU TP domain for a rank
         * @param rank_inv Rank inventory
         * @param domain_id Unique domain ID
         * @return DomainAssignment for GPU TP
         */
        DomainAssignment createGPUDomain(
            const RankInventory &rank_inv,
            int domain_id) const;

        /**
         * @brief Create CPU TP domain for a node
         * @param node_inv Node inventory
         * @param cluster_inv Cluster inventory
         * @param domain_id Unique domain ID
         * @return DomainAssignment for CPU TP
         */
        DomainAssignment createCPUDomain(
            const NodeInventory &node_inv,
            const ClusterInventory &cluster_inv,
            int domain_id) const;
    };

} // namespace llaminar2
