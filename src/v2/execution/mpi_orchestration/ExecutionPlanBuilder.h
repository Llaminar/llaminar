/**
 * @file ExecutionPlanBuilder.h
 * @brief Concrete implementation of IExecutionPlanBuilder
 *
 * Translates OrchestrationConfig into RankExecutionPlans:
 * 1. Named domains mode: Parse domain definitions and PP stage mappings
 * 2. Simple TP/PP mode: Auto-configure based on --tp, --pp, --device
 *
 * The builder is deterministic: all ranks compute identical plans given
 * the same inputs.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IExecutionPlanBuilder.h"
#include "RankExecutionPlan.h"
#include "DeviceInventory.h"
#include "../../config/OrchestrationConfig.h"
#include "../../execution/global_pp/GlobalPPTopology.h"
#include <memory>
#include <vector>
#include <map>

namespace llaminar2
{

    /**
     * @brief Internal representation of a resolved TP domain
     *
     * After parsing, domains are resolved to this form which
     * contains all computed information needed for plan building.
     */
    struct ResolvedDomain
    {
        int id = -1;                              ///< Unique domain ID
        std::string name;                         ///< Domain name
        std::vector<GlobalDeviceAddress> devices; ///< Devices in domain
        std::vector<float> weights;               ///< Work distribution
        CollectiveBackendType backend = CollectiveBackendType::AUTO;

        // Rank mappings (computed during resolution)
        std::vector<int> ranks;           ///< MPI ranks in this domain
        std::map<int, int> rank_to_index; ///< Rank -> index within domain
    };

    /**
     * @brief Internal representation of a resolved PP stage
     */
    struct ResolvedPPStage
    {
        int stage_id = -1;
        int domain_id = -1; ///< Associated domain
        std::string domain_name;
        int first_layer = 0;
        int last_layer = -1;
        std::vector<int> ranks; ///< Ranks executing this stage
    };

    /**
     * @brief Concrete implementation of IExecutionPlanBuilder
     *
     * Builds execution plans by:
     * 1. Resolving domain definitions to device/rank mappings
     * 2. Assigning layers to PP stages (or single domain)
     * 3. Computing weight shards for TP
     * 4. Determining PP communication neighbors
     */
    class ExecutionPlanBuilder : public IExecutionPlanBuilder
    {
    public:
        ExecutionPlanBuilder() = default;
        ~ExecutionPlanBuilder() override = default;

        /**
         * @brief Build execution plans for all MPI ranks
         */
        std::vector<RankExecutionPlan> buildAllPlans(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory) override;

        /**
         * @brief Build execution plan for a specific rank
         */
        RankExecutionPlan buildPlanForRank(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory,
            int rank) override;

        /**
         * @brief Validate configuration
         */
        std::vector<std::string> validateConfig(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory) override;

        /**
         * @brief Build a GlobalPPTopology from named-domain PP configuration.
         *
         * Converts each PPStageDefinition into a GlobalPPStageSpec, deriving
         * local vs global scope from DomainDefinition::scope, owner_rank, and
         * explicit_ranks.  For AUTO-scope domains, single-rank domains become
         * local stages and multi-rank domains become global TP stages.
         *
         * Requires at least one PPStageDefinition in the config; returns a
         * minimal single-stage topology otherwise.
         */
        GlobalPPTopology buildGlobalPPTopology(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory);

    private:
        // =====================================================================
        // Domain Resolution
        // =====================================================================

        /**
         * @brief Resolve domain definitions to internal representation
         */
        std::vector<ResolvedDomain> resolveDomains(
            const OrchestrationConfig &config,
            const ClusterInventory &cluster_inventory);

        /**
         * @brief Create implicit domain for simple TP mode
         */
        ResolvedDomain createImplicitDomain(
            const OrchestrationConfig &config,
            const ClusterInventory &cluster_inventory);

        /**
         * @brief Find which rank owns a device
         */
        int findRankForDevice(
            const GlobalDeviceAddress &device,
            const ClusterInventory &cluster_inventory);

        // =====================================================================
        // PP Stage Resolution
        // =====================================================================

        /**
         * @brief Resolve PP stage definitions
         */
        std::vector<ResolvedPPStage> resolvePPStages(
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const std::vector<ResolvedDomain> &domains,
            const ClusterInventory &cluster_inventory);

        /**
         * @brief Create PP stages for simple PP mode (equal layer split)
         */
        std::vector<ResolvedPPStage> createSimplePPStages(
            int pp_degree,
            int n_layers,
            const ClusterInventory &cluster_inventory);

        // =====================================================================
        // Weight Sharding
        // =====================================================================

        /**
         * @brief Calculate weight shard info for a rank
         */
        WeightShardInfo calculateWeightShard(
            int rank,
            const ResolvedDomain *primary_domain,
            const OrchestrationConfig &config);

        // =====================================================================
        // Plan Building
        // =====================================================================

        /**
         * @brief Build plan for one rank using named domains
         */
        RankExecutionPlan buildPlanWithDomains(
            int rank,
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory,
            const std::vector<ResolvedDomain> &domains,
            const std::vector<ResolvedPPStage> &pp_stages);

        /**
         * @brief Build plan for simple TP/PP mode
         */
        RankExecutionPlan buildSimplePlan(
            int rank,
            const OrchestrationConfig &config,
            const ModelConfig &model_config,
            const ClusterInventory &cluster_inventory);

        // =====================================================================
        // Backend Selection
        // =====================================================================

        /**
         * @brief Select collective backend for a set of devices
         */
        CollectiveBackendType selectBackend(
            const std::vector<GlobalDeviceAddress> &devices);

        /**
         * @brief Select backend for cross-rank communication
         */
        CollectiveBackendType selectCrossRankBackend(
            const ClusterInventory &cluster_inventory);
    };

    // =========================================================================
    // Free Functions
    // =========================================================================

    /**
     * @brief Render multi-domain topology info for --show-topology output.
     *
     * Returns GlobalPPTopology::toTable() followed by a per-rank stage runner
     * count and domain name summary.
     */
    std::string renderMultiDomainTopologyInfo(const GlobalPPTopology &topo, int world_size);

} // namespace llaminar2
