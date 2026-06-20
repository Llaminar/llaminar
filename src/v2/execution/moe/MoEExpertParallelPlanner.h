/**
 * @file MoEExpertParallelPlanner.h
 * @brief Static residency planner for same-layer MoE expert-parallel overlays.
 */

#pragma once

#include "MoEExpertParallelPlan.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{

    class DecodeExpertHistogram;

    struct MoEExpertModelMetadata
    {
        int num_layers = 0;
        int num_experts = 0;
        int d_model = 0;
        int routed_intermediate_size = 0;
        int shared_intermediate_size = 0;
        bool has_shared_expert = false;
        std::string routed_quant_type = "F32";
        std::string shared_quant_type = "F32";
    };

    struct MoEExpertLayerTierMask
    {
        int layer = -1;
        int tier_index = -1;
        std::vector<int> expert_ids;
    };

    /**
     * @brief Per-layer diagnostics produced by the RoutedTierRebalanced policy.
     */
    struct MoERoutedTierRebalanceLayerDiagnostics
    {
        int layer = -1;
        /// Number of experts assigned to each tier (index mirrors plan.routed_tiers).
        std::vector<int> tier_expert_counts;
        /// Expected GPU token hits per token (sum of GPU-tier expert probabilities).
        /// Computed from histogram activation counts if available; else 0.0.
        float expected_gpu_hit_rate = 0.0f;
        /// Expected CPU fallback rows per token (1.0 - expected_gpu_hit_rate).
        float expected_cpu_fallback_rows = 0.0f;
        /// Fraction of experts assigned to non-fallback tiers (GPU coverage).
        float gpu_coverage_ratio = 0.0f;
        /// Total routed expert memory bytes assigned to non-fallback tiers for this layer.
        size_t gpu_tier_memory_bytes = 0;
    };

    struct MoERoutedTierRebalanceDiagnostics
    {
        std::vector<MoERoutedTierRebalanceLayerDiagnostics> layers;
        /// Average GPU hit rate across all layers.
        float avg_gpu_hit_rate = 0.0f;
        /// Average CPU fallback rows across all layers.
        float avg_cpu_fallback_rows = 0.0f;
        /// Average GPU coverage ratio across all layers.
        float avg_gpu_coverage_ratio = 0.0f;
        /// Whether a histogram was used for ordering.
        bool histogram_used = false;
    };

    /**
     * @brief Options for the RoutedTierRebalanced residency policy.
     *
     * When enabled, the planner runs a deterministic histogram-driven
     * hot-cache placement across routed tiers: highest-activation experts
     * land in highest-priority non-fallback tiers first; uncovered experts
     * fall to the fallback tier (if configured).
     *
     * Defaults are backward-compatible (disabled).
     */
    struct MoERoutedTierRebalancerOptions
    {
        bool enabled = false;
        /// Activation count above which an expert is considered "hot" for
        /// promotion diagnostics.  Informational only — does not affect
        /// assignment order; use decode_histogram + tier capacities for that.
        uint64_t promotion_threshold = 0;
        /// Activation count below which an expert is considered "cold" for
        /// demotion diagnostics.  Informational only.
        uint64_t demotion_threshold = 0;
        /// Optional previous placements for hysteresis.  When provided,
        /// already-assigned experts are kept in their tier unless capacity
        /// forces eviction.  Currently reserved for future use.
        std::vector<ExpertLayerPlacement> previous_placements;
    };

    struct MoEExpertParallelPlannerOptions
    {
        const DecodeExpertHistogram *decode_histogram = nullptr;
        std::vector<ExpertLayerPlacement> explicit_placements;
        std::vector<MoEExpertLayerTierMask> explicit_masks;
        MoERoutedTierRebalancerOptions rebalancer;
    };

    struct MoEExpertParallelPlannerInput
    {
        MoEExpertParallelPlan plan;
        MoEExpertModelMetadata metadata;
        MoEExpertParallelPlannerOptions options;
    };

    struct MoEExpertTierMemoryEstimate
    {
        int tier_index = -1;
        std::string tier_name;
        std::string domain;
        size_t routed_expert_count = 0;
        size_t routed_expert_bytes = 0;
    };

    struct MoEExpertDomainMemoryEstimate
    {
        std::string domain;
        size_t shared_expert_bytes = 0;
        size_t routed_expert_bytes = 0;

        size_t totalBytes() const
        {
            return shared_expert_bytes + routed_expert_bytes;
        }
    };

    struct MoEExpertParallelMemoryEstimate
    {
        std::string shared_expert_domain;
        size_t routed_expert_bytes_per_expert = 0;
        size_t shared_expert_bytes_per_layer = 0;
        size_t total_shared_expert_bytes = 0;
        size_t total_routed_expert_bytes = 0;
        std::vector<MoEExpertTierMemoryEstimate> tiers;
        std::vector<MoEExpertDomainMemoryEstimate> domains;
    };

    struct MoEExpertParallelPlannerResult
    {
        MoEExpertParallelPlan planned_plan;
        MoEExpertParallelMemoryEstimate memory;
        /// Populated when residency_policy == RoutedTierRebalanced.
        MoERoutedTierRebalanceDiagnostics rebalance_diagnostics;
    };

    class MoEExpertParallelPlanner
    {
    public:
        static MoEExpertParallelPlannerResult plan(const MoEExpertParallelPlannerInput &input);

        static MoEExpertParallelPlannerResult plan(
            const MoEExpertParallelPlan &plan,
            const MoEExpertModelMetadata &metadata,
            const MoEExpertParallelPlannerOptions &options = {});

        static size_t estimateRoutedExpertBytesPerExpert(const MoEExpertModelMetadata &metadata);
        static size_t estimateSharedExpertBytesPerLayer(const MoEExpertModelMetadata &metadata);
        static size_t estimateTotalSharedExpertBytes(const MoEExpertModelMetadata &metadata);
    };

} // namespace llaminar2
