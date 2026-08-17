/**
 * @file MoERoutedExpertPlacementPlanner.h
 * @brief Capacity-exact and phase-cost-aware routed-expert placement planning.
 *
 * The planner consumes immutable model geometry, setup-resolved per-layer tier
 * quotas, optional routing evidence, and optional certified service costs. It
 * never inspects tier labels or probes devices. Histogram-driven placement is
 * deterministic across ranks: exact integer phase costs are minimized first,
 * incumbent movement second, and expert identity last.
 */

#pragma once

#include "DecodeExpertHistogram.h"
#include "MoERoutedExpertPlacementPlan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    struct MoERoutedExpertModelMetadata
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

    struct MoERoutedExpertLayerTierMask
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
        /// Whether exact phase-specific service costs drove assignment.
        bool phase_service_profile_used = false;
        /// Authenticated setup profile identity, empty when raw counts drove planning.
        std::string phase_service_profile_identity;
    };

    /**
     * @brief Measured service time for one tier/layer and inference phase.
     *
     * `nanoseconds_per_activation` follows the retained production histogram
     * order: decode, real prefill, then accepted grouped-verifier rows. Values
     * are positive setup evidence for the exact model projection geometry and
     * prepared codebook. They are not inferred from a tier label or backend,
     * and phase-specific backend crossovers are valid measured outcomes.
     */
    struct MoERoutedTierLayerPhaseServiceCost
    {
        int tier_index = -1;
        int layer = -1;
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            nanoseconds_per_activation{};
    };

    /**
     * @brief Complete immutable service profile consumed by tier assignment.
     *
     * A valid profile contains exactly one row for every `(tier, layer)` and a
     * stable non-empty identity derived by its setup-time certification owner.
     * Integer priority controls capacity fill and the final deterministic tie;
     * measured service time controls the primary runtime economy objective.
     */
    struct MoERoutedTierServiceProfile
    {
        std::string identity;
        /** Phases reachable under the immutable runtime MTP policy. */
        ExpertHistogramProductionSourceMask active_sources =
            kAllExpertHistogramProductionSources;
        std::vector<MoERoutedTierLayerPhaseServiceCost> costs;
    };

    /**
     * @brief Options for the RoutedTierRebalanced residency policy.
     *
     * When enabled, the planner runs deterministic histogram-driven,
     * demand-ranked placement across routed tiers: highest-activation experts
     * land in lowest-numeric-priority non-coverage tiers first; uncovered
     * experts land in the coverage tier (if configured). Tier labels and
     * declaration indices never participate in ordering.
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
        /// the exact optimizer minimizes movement after minimizing measured
        /// service time, then applies its deterministic expert-id tie break.
        std::vector<RoutedExpertLayerPlacement> previous_placements;
    };

    struct MoERoutedExpertPlacementPlannerOptions
    {
        /// Live histogram view used by ordinary synchronous planning callers.
        const DecodeExpertHistogram *decode_histogram = nullptr;
        /// Immutable generation used by asynchronous residency migration.
        /// Mutually exclusive with `decode_histogram`.
        const DecodeExpertHistogramWindow *decode_histogram_window = nullptr;
        /**
         * Optional complete setup-certified phase service profile.
         *
         * When present, histogram-driven policies solve the exact fixed-quota
         * minimum-cost assignment. When absent they retain the historical raw
         * aggregate-count ordering; production dynamic composition is expected
         * to install a certified profile before claiming cost-optimal movement.
         */
        const MoERoutedTierServiceProfile *phase_service_profile = nullptr;
        std::vector<RoutedExpertLayerPlacement> explicit_placements;
        std::vector<MoERoutedExpertLayerTierMask> explicit_masks;
        MoERoutedTierRebalancerOptions rebalancer;
    };

    struct MoERoutedExpertPlacementPlannerInput
    {
        MoERoutedExpertPlacementPlan plan;
        MoERoutedExpertModelMetadata metadata;
        MoERoutedExpertPlacementPlannerOptions options;
    };

    struct MoERoutedExpertTierMemoryEstimate
    {
        int tier_index = -1;
        std::string tier_name;
        std::string domain;
        size_t routed_expert_count = 0;
        size_t routed_expert_bytes = 0;
    };

    struct MoERoutedExpertDomainMemoryEstimate
    {
        std::string domain;
        size_t shared_expert_bytes = 0;
        size_t routed_expert_bytes = 0;

        size_t totalBytes() const
        {
            return shared_expert_bytes + routed_expert_bytes;
        }
    };

    struct MoERoutedExpertPlacementMemoryEstimate
    {
        std::string shared_expert_domain;
        size_t routed_expert_bytes_per_expert = 0;
        size_t shared_expert_bytes_per_layer = 0;
        size_t total_shared_expert_bytes = 0;
        size_t total_routed_expert_bytes = 0;
        std::vector<MoERoutedExpertTierMemoryEstimate> tiers;
        std::vector<MoERoutedExpertDomainMemoryEstimate> domains;
    };

    struct MoERoutedExpertPlacementPlannerResult
    {
        MoERoutedExpertPlacementPlan planned_plan;
        MoERoutedExpertPlacementMemoryEstimate memory;
        /// Populated when residency_policy == RoutedTierRebalanced.
        MoERoutedTierRebalanceDiagnostics rebalance_diagnostics;
    };

    class MoERoutedExpertPlacementPlanner
    {
    public:
        static MoERoutedExpertPlacementPlannerResult plan(const MoERoutedExpertPlacementPlannerInput &input);

        static MoERoutedExpertPlacementPlannerResult plan(
            const MoERoutedExpertPlacementPlan &plan,
            const MoERoutedExpertModelMetadata &metadata,
            const MoERoutedExpertPlacementPlannerOptions &options = {});

        static size_t estimateRoutedExpertBytesPerExpert(const MoERoutedExpertModelMetadata &metadata);
        static size_t estimateSharedExpertBytesPerLayer(const MoERoutedExpertModelMetadata &metadata);
        static size_t estimateTotalSharedExpertBytes(const MoERoutedExpertModelMetadata &metadata);
    };

} // namespace llaminar2
