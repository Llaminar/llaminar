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
        /**
         * Total routed-layer storage required by the retained graph family.
         *
         * This may exceed the ordinary decoder layer count when a model keeps
         * routed NextN/MTP sidecar banks resident for graph reuse.  Consumers
         * must not infer a main-inference boundary from this capacity.
         */
        int num_layers = 0;
        int num_experts = 0;
        int d_model = 0;
        int routed_intermediate_size = 0;
        int shared_intermediate_size = 0;
        bool has_shared_expert = false;
        std::string routed_quant_type = "F32";
        std::string shared_quant_type = "F32";
        /**
         * Exclusive upper bound of ordinary main-model layer identities.
         *
         * The model-aware resolver always publishes this independently of
         * retained sidecar capacity.  A zero value means that the metadata was
         * not resolved from a live model and is invalid for inference-window
         * boundary selection.
         */
        int main_inference_layer_count = 0;
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
     * are positive setup evidence for economy-priced recurring phases and exact
     * zero for reachable-but-exceptional or unreachable phases. The values are
     * not inferred from a tier label or backend, and phase-specific backend
     * crossovers are valid measured outcomes.
     */
    struct MoERoutedTierLayerPhaseServiceCost
    {
        int tier_index = -1;
        int layer = -1;
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            nanoseconds_per_activation{};
    };

    /**
     * @brief Measured routed service cost for one exact overlay participant.
     *
     * Tier aggregates are sufficient for cold-start capacity placement, but a
     * live residency transaction must price the parallel critical path across
     * the actual owners.  Keeping this row in the certified profile prevents a
     * slower participant from being hidden by a tier label and lets promotion,
     * demotion, and same-tier skew correction share one measured objective.
     */
    struct MoERoutedParticipantLayerPhaseServiceCost
    {
        int participant_id = -1;
        int layer = -1;
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            nanoseconds_per_activation{};
    };

    /**
     * @brief Complete immutable service profile consumed by tier assignment.
     *
     * A placement-valid profile contains exactly one tier row for every
     * `(tier, layer)`.  A profile used by the live ExpertOverlay economy also
     * contains one participant row for every `(participant, layer)`.  The tier
     * rows are conservative aggregates used for cold-start capacity placement;
     * transaction admission uses the participant rows to compare the exact
     * before/after parallel critical path.  The stable identity covers both.
     */
    struct MoERoutedTierServiceProfile
    {
        std::string identity;
        /** Graph reachability and economy-priced subset for every layer. */
        ExpertHistogramProductionTopology production_topology;
        std::vector<MoERoutedTierLayerPhaseServiceCost> costs;
        std::vector<MoERoutedParticipantLayerPhaseServiceCost>
            participant_costs;
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
