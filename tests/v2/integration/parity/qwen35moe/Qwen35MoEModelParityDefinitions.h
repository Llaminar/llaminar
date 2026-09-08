/**
 * @file Qwen35MoEModelParityDefinitions.h
 * @brief Authenticated Qwen3.5 MoE identities and ExpertOverlay topologies.
 *
 * Every Qwen3.5 MoE production parity binary consumes these declarations so
 * model identity, Hugging Face reference identity, numerical thresholds, and
 * routed-expert topology cannot drift between fixtures.  Tier direction is
 * expressed only by integer priority; names are diagnostic labels and carry
 * no placement semantics.
 */

#pragma once

#include "../ModelParityDefinition.h"

#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test::parity::qwen35moe
{
    /** Exact prompt shared by the authenticated 35B reference packs. */
    inline constexpr const char *kQwen35MoEParityPrompt =
        "The quick brown fox jumps over the lazy dog";

    /** Authenticated tokenization of @ref kQwen35MoEParityPrompt. */
    inline constexpr std::array<int, 9> kQwen35MoEParityTokenIds = {
        760,
        3841,
        13477,
        37550,
        33075,
        888,
        279,
        15217,
        5388,
    };

    /**
     * Complete authenticated demand window used by movement-only cells.
     *
     * A shorter prefix allowed the first demand bank to make a statistically
     * valid but one-axis-only decision. The complete parity prompt gives every
     * routed layer the same causal workload that numerical parity subsequently
     * exercises, so a published promotion is re-exercised by the reference
     * path instead of being certified only by unrelated synthetic traffic.
     */
    inline constexpr int kQwen35MoEMovementProofInitialWindowRows =
        static_cast<int>(kQwen35MoEParityTokenIds.size());

    /** @return An owning copy of the authenticated prompt tokenization. */
    inline std::vector<int> qwen35MoEParityTokenIds()
    {
        return {
            kQwen35MoEParityTokenIds.begin(),
            kQwen35MoEParityTokenIds.end(),
        };
    }

    /**
     * Requests discarded symmetrically while convergence graphs warm.
     *
     * The speed witness runs only after graph materialization, transport
     * profiling, and live economy certification have already exercised the
     * exact production graph family. A further discarded 122B request would
     * warm no new state and would consume part of the ten-minute parity-cell
     * budget without contributing evidence.
     */
    inline constexpr int kQwen35MoEConvergenceTimingWarmupRequests = 0;
    /** Full-prefill observations retained in each placement cohort. */
    inline constexpr int kQwen35MoEConvergenceTimingMeasuredRequests = 3;
    /** Complete stationary request corpus replayed on both sides of the A/B. */
    inline constexpr int kQwen35MoEConvergenceTimingCorpusRequests =
        kQwen35MoEConvergenceTimingWarmupRequests +
        kQwen35MoEConvergenceTimingMeasuredRequests;
    /** Routed decode forwards issued after every convergence prefill. */
    inline constexpr int kQwen35MoEConvergenceTimingDecodeForwards = 2;
    /** Production-segmented routed rows in every convergence prefill. */
    inline constexpr std::size_t kQwen35MoEConvergenceTimingPromptRows = 17u;
    /** Largest canonical parity decode horizon among the shared 35B/122B cells. */
    inline constexpr int kQwen35MoEMaximumParityDecodeForwards = 5;

    /** @return Routed rows executed by one complete timing request. */
    inline constexpr std::uint64_t
    qwen35MoEConvergenceTimingRequestRoutedRows() noexcept
    {
        return static_cast<std::uint64_t>(
                   kQwen35MoEConvergenceTimingPromptRows) +
               static_cast<std::uint64_t>(
                   kQwen35MoEConvergenceTimingDecodeForwards);
    }

    /**
     * @return Exact routed-row corpus that one immutable timing epoch retains.
     *
     * This geometry is shared by matrix construction and the real-weight
     * fixture. Keeping it beside the typed model definition prevents a
     * fixture-local timing change from silently under-sizing the production
     * histogram policy selected by the expander.
     */
    inline constexpr std::uint64_t
    qwen35MoEConvergenceTimingCohortRoutedRows() noexcept
    {
        return static_cast<std::uint64_t>(
                   kQwen35MoEConvergenceTimingCorpusRequests) *
               qwen35MoEConvergenceTimingRequestRoutedRows();
    }

    /**
     * @return Conservative routed rows in the numerical proof after timing.
     *
     * The shared speed-witness geometry serves both the 35B five-step parity
     * model and the 122B four-step model. Reserving the larger authenticated
     * request keeps Dynamic maintenance live without allowing a new placement
     * publication to invalidate the prefix between its seed and restore.
     */
    inline constexpr std::uint64_t
    qwen35MoEMaximumNumericalParityRoutedRows() noexcept
    {
        return static_cast<std::uint64_t>(
                   kQwen35MoEParityTokenIds.size()) +
               static_cast<std::uint64_t>(
                   kQwen35MoEMaximumParityDecodeForwards);
    }

    /** @return Complete post-movement evidence protected in one epoch. */
    inline constexpr std::uint64_t
    qwen35MoEConvergenceProtectedRoutedRows() noexcept
    {
        return qwen35MoEConvergenceTimingCohortRoutedRows() +
               qwen35MoEMaximumNumericalParityRoutedRows();
    }

    /**
     * Fixed demand-bank width for the matched before/after speed witness.
     *
     * The 57-row timing corpus and at most 14 numerical-parity rows must fit
     * strictly inside one immutable epoch.
     * Three request-matched prefills give the latency gate an odd median and
     * six decode observations per side. Every 17-row prefill is necessarily
     * segmented across the production 16-row capture capacity, so shortening
     * the witness does not weaken its captured 16+1 execution proof. Request
     * admission is sequential, but the fourth asynchronous publication can
     * complete while one already-admitted 19-row request is executing. The
     * remaining rows therefore retain that exact overlap, the complete
     * post-movement cohort, and the canonical prefix/decode proof strictly
     * below the next publication threshold.
     * Movement-only cells use their separate shorter policy.
     */
    inline constexpr int kQwen35MoEConvergenceHistogramWindowRows = 96;

    static_assert(
        static_cast<std::uint64_t>(
            kQwen35MoEConvergenceHistogramWindowRows) >
        qwen35MoEConvergenceProtectedRoutedRows() +
            qwen35MoEConvergenceTimingRequestRoutedRows(),
        "The convergence window must retain one in-flight request and the "
        "complete post-movement timing and numerical-parity proof");
    static_assert(kQwen35MoEConvergenceTimingWarmupRequests >= 0);
    static_assert(
        kQwen35MoEConvergenceTimingMeasuredRequests % 2 == 1,
        "The convergence median must retain an odd request count");

    /** @return Canonical Qwen3.5-35B-A3B Q4_K_XL identity. */
    inline ModelParityModelDefinition qwen35MoE35BQ4KXLParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen35MoE_35B_Q4KXL",
            .model_path =
                "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            .reference_directory = "pytorch_qwen35_moe_snapshots",
            .prompt = kQwen35MoEParityPrompt,
            .token_ids = qwen35MoEParityTokenIds(),
            .decode_steps = kQwen35MoEMaximumParityDecodeForwards,
            .max_seq_len = 4096,
            .transformer_layers = 40,
            .attention_heads = 16,
            .kv_heads = 2,
        };
    }

    /** @return Canonical smaller Q3_K_S GPU proof identity. */
    inline ModelParityModelDefinition qwen35MoE35BQ3KSParityModel()
    {
        auto model = qwen35MoE35BQ4KXLParityModel();
        model.test_id = "Qwen35MoE_35B_Q3KS";
        model.model_path = "models/Qwen3.5-35B-A3B-Q3_K_S.gguf";
        model.reference_directory = "pytorch_qwen35_moe_q3ks_snapshots";
        return model;
    }

    /** @return Strict single-device numerical contract shared by all backends. */
    inline BackendThresholds qwen35MoESingleDeviceThresholds()
    {
        return {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.03f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 3,
        };
    }

    /** @return Shard-aware numerical contract for routed multi-device graphs. */
    inline BackendThresholds qwen35MoEMultiDeviceThresholds()
    {
        return {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .excluded_stages = {
                "Q_PROJECTION",
                "K_PROJECTION",
                "V_PROJECTION",
                "Q_NORM",
                "K_NORM",
                "Q_ROPE",
                "K_ROPE",
                "ATTENTION_CONTEXT",
                "FA_GATE",
                "ATTENTION_CONTEXT_GATED",
                "FFN_GATE",
                "FFN_UP",
                "FFN_SWIGLU",
                "QKV_PROJECTION",
                "GDN_CONV1D_OUTPUT",
                "GDN_Z_PROJECTION",
                "GDN_DELTA_RULE_OUTPUT",
                "GDN_NORM_GATE_OUTPUT",
                /* Branch-local partials are not canonical post-collective values. */
                "MOE_EXPERT_OUTPUT",
                "MOE_SHARED_EXPERT_OUTPUT",
                "MOE_SHARED_GATE_OUTPUT",
            },
            .allreduce_stages = {},
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 4,
        };
    }

    /**
     * @return Numerical contract for ExpertOverlay's completed sparse return.
     *
     * Generic tensor-parallel graphs expose `MOE_EXPERT_OUTPUT` before their
     * cross-participant reduction, so the ordinary multi-device contract must
     * exclude that branch-local value.  ExpertOverlay deliberately republishes
     * the same semantic key from the final sparse-return consume stage.  That
     * value is the canonical routed sum and therefore remains a required
     * checkpoint in both the numerical comparison and its CSV evidence.
     */
    inline BackendThresholds qwen35MoEExpertOverlayThresholds()
    {
        auto thresholds = qwen35MoEMultiDeviceThresholds();
        thresholds.excluded_stages.erase(
            std::remove(
                thresholds.excluded_stages.begin(),
                thresholds.excluded_stages.end(),
                "MOE_EXPERT_OUTPUT"),
            thresholds.excluded_stages.end());
        return thresholds;
    }

    /** Accelerator family owning dense continuation in a multi-tier proof. */
    enum class Qwen35MoEOverlayContinuationBackend : std::uint8_t
    {
        CUDA,
        ROCm,
    };

    /**
     * @brief Compact source of truth for one 35B graph-native topology.
     *
     * Counts identify physical participants, never expert quotas. The live
     * capacity planner fills each integer-priority tier to its exact remaining
     * capacity after every named model-lifetime allocation is charged.
     * `segmented_prefill` adds the complementary fixed-bucket graph profile to
     * the same centrally expanded placement/movement matrix.
     */
    struct Qwen35MoE35BOverlayTopologySpec
    {
        const char *test_id;
        int cuda_participants;
        int rocm_participants;
        int cpu_participants;
        Qwen35MoEOverlayContinuationBackend continuation;
        bool segmented_prefill;
        ModelParityDynamicSpeedupWitness dynamic_speedup_witness;
    };

    /** @return Every 35B graph-native topology formerly registered by hand. */
    inline const std::array<Qwen35MoE35BOverlayTopologySpec, 4> &
    qwen35MoE35BOverlayTopologySpecs()
    {
        static const std::array<Qwen35MoE35BOverlayTopologySpec, 4> specs{{
            {
                "CUDA1_ROCm1_CPU2_2xMPI_NodeExpertOverlay",
                1,
                1,
                2,
                Qwen35MoEOverlayContinuationBackend::CUDA,
                true,
                ModelParityDynamicSpeedupWitness::Random,
            },
            {
                "CUDA1_CPU2_2xMPI_NodeExpertOverlay",
                1,
                0,
                2,
                Qwen35MoEOverlayContinuationBackend::CUDA,
                false,
                ModelParityDynamicSpeedupWitness::Disabled,
            },
            {
                "ROCm1_CPU2_2xMPI_NodeExpertOverlay",
                0,
                1,
                2,
                Qwen35MoEOverlayContinuationBackend::ROCm,
                false,
                ModelParityDynamicSpeedupWitness::Disabled,
            },
            {
                "CUDA1_ROCm1_2xMPI_NodeExpertOverlay",
                1,
                1,
                0,
                Qwen35MoEOverlayContinuationBackend::CUDA,
                false,
                ModelParityDynamicSpeedupWitness::Disabled,
            },
        }};
        return specs;
    }

    /** @brief Build one inventory-bound single-accelerator overlay domain. */
    inline RoutedExpertDomain qwen35MoE35BAcceleratorOverlayDomain(
        Qwen35MoEOverlayContinuationBackend backend,
        bool continuation)
    {
        RoutedExpertDomain domain;
        if (backend == Qwen35MoEOverlayContinuationBackend::CUDA)
        {
            domain.name = "cuda_hot";
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
        }
        else
        {
            domain.name = continuation ? "rocm_hot" : "rocm_warm";
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0)};
        }
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        return domain;
    }

    /** @brief Build the two-socket CPU overlay domain bound by live inventory. */
    inline RoutedExpertDomain qwen35MoE35BCpuOverlayDomain(
        int participant_count)
    {
        if (participant_count <= 0)
            throw std::invalid_argument(
                "35B CPU overlay domain requires positive participant count");
        RoutedExpertDomain domain;
        domain.name = "cpu_cold";
        domain.scope = ExecutionDomainScope::NODE_LOCAL;
        domain.backend = CollectiveBackendType::UPI;
        for (int numa = 0; numa < participant_count; ++numa)
            domain.participants.push_back(GlobalDeviceAddress::cpu(numa));
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        return domain;
    }

    /**
     * @brief Build one immutable automatic-capacity multi-tier blueprint.
     *
     * Priorities are deliberately non-semantic and non-consecutive. All tier
     * quotas remain zero so the production memory BOM, rather than the test,
     * determines exact residency. The generated case later installs
     * Static/Dynamic and Ordinal/Random without changing this topology.
     */
    inline std::shared_ptr<const MoERoutedExpertPlacementPlan>
    qwen35MoE35BOverlayPlan(const Qwen35MoE35BOverlayTopologySpec &spec)
    {
        const bool cuda_continuation =
            spec.continuation ==
            Qwen35MoEOverlayContinuationBackend::CUDA;
        if ((cuda_continuation && spec.cuda_participants != 1) ||
            (!cuda_continuation && spec.rocm_participants != 1))
        {
            throw std::invalid_argument(
                "35B overlay continuation requires exactly one accelerator");
        }

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = RoutedExpertOwnerOrder::Ordinal;

        auto continuation = qwen35MoE35BAcceleratorOverlayDomain(
            spec.continuation, true);
        plan->continuation_domain = continuation.name;
        plan->base_model_domain = continuation.name;
        plan->shared_expert_domain = continuation.name;
        plan->continuation_domain_spec.domain = continuation.name;
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::Replicated);
        plan->domains.push_back(std::move(continuation));

        if (cuda_continuation && spec.rocm_participants > 0)
        {
            plan->domains.push_back(
                qwen35MoE35BAcceleratorOverlayDomain(
                    Qwen35MoEOverlayContinuationBackend::ROCm, false));
        }
        else if (!cuda_continuation && spec.cuda_participants > 0)
        {
            plan->domains.push_back(
                qwen35MoE35BAcceleratorOverlayDomain(
                    Qwen35MoEOverlayContinuationBackend::CUDA, false));
        }
        if (spec.cpu_participants > 0)
            plan->domains.push_back(
                qwen35MoE35BCpuOverlayDomain(spec.cpu_participants));

        constexpr std::array<int, 3> priorities{-20, 7, 41};
        for (std::size_t index = 0; index < plan->domains.size(); ++index)
        {
            const int priority =
                plan->domains.size() == 2u && index == 1u
                    ? 17
                    : priorities.at(index);
            plan->routed_tiers.push_back({
                .name = "priority_" + std::to_string(priority),
                .domain = plan->domains[index].name,
                .priority = priority,
                .max_experts_per_layer = 0,
                .memory_budget_bytes = 0,
                .fallback = index + 1u == plan->domains.size(),
            });
        }

        const auto validation = validateMoERoutedExpertPlacementPlan(*plan);
        if (!validation.ok())
        {
            std::string message =
                "invalid Qwen3.5 MoE 35B graph-native overlay plan";
            for (const auto &error : validation.errors)
                message += "\n - " + error;
            throw std::logic_error(message);
        }
        return plan;
    }

    /** @return Typed participants and rank ownership for one 35B topology. */
    inline ModelParityTopologyDefinition qwen35MoE35BOverlayTopology(
        const Qwen35MoE35BOverlayTopologySpec &spec)
    {
        ModelParityTopologyDefinition topology{
            .test_id = spec.test_id,
            .kind = ModelParityTopologyKind::NodeMultiDomain,
            .collective = Collective::None,
            .mpi_ranks = 2,
            .expert_overlay_plan = qwen35MoE35BOverlayPlan(spec),
        };
        for (int ordinal = 0; ordinal < spec.cuda_participants; ++ordinal)
        {
            topology.participants.push_back({
                GlobalDeviceAddress::cuda(ordinal), std::nullopt});
        }
        for (int ordinal = 0; ordinal < spec.rocm_participants; ++ordinal)
        {
            topology.participants.push_back({
                GlobalDeviceAddress::rocm(ordinal), std::nullopt});
        }
        for (int numa = 0; numa < spec.cpu_participants; ++numa)
        {
            topology.participants.push_back({
                GlobalDeviceAddress::cpu(numa), numa});
        }
        return topology;
    }

    /**
     * @brief Build one homogeneous single-domain ExpertOverlay authority.
     *
     * Dense/shared work remains tensor parallel while routed experts are
     * apportioned across the same participants. A zero quota and budget ask
     * production capacity planning to fill the only fallback tier to its exact
     * admitted capacity rather than embedding a model-specific cap.
     *
     * @param domain_name Stable diagnostic domain identity.
     * @param scope Rank-local or node-local execution scope.
     * @param backend Exact collective for the homogeneous participants.
     * @param participants Physical devices in logical participant order.
     * @return Valid immutable blueprint for typed matrix expansion.
     */
    inline std::shared_ptr<const MoERoutedExpertPlacementPlan>
    qwen35MoESingleDomainOverlayPlan(
        std::string domain_name,
        ExecutionDomainScope scope,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        RoutedExpertDomain domain;
        domain.name = domain_name;
        domain.scope = scope;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::SingleDomain;
        plan->continuation_domain = domain_name;
        plan->base_model_domain = domain_name;
        plan->shared_expert_domain = domain_name;
        plan->continuation_domain_spec.domain = domain_name;
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::TensorParallel);
        plan->continuation_domain_spec.hidden_layout =
            MoEContinuationActivationLayout::ReplicatedHidden;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
        plan->dense_domains = {domain.toExecutionDomainDefinition()};
        plan->domains = {std::move(domain)};
        plan->routed_tiers = {{
            .name = "priority_0",
            .domain = domain_name,
            .priority = 0,
            .max_experts_per_layer = 0,
            .memory_budget_bytes = 0,
            .fallback = true,
        }};

        const auto validation = validateMoERoutedExpertPlacementPlan(*plan);
        if (!validation.ok())
        {
            std::string message =
                "invalid Qwen3.5 MoE single-domain parity plan";
            for (const auto &error : validation.errors)
                message += "\n - " + error;
            throw std::logic_error(message);
        }
        return plan;
    }

    /** @return Two-ROCm rank-local ExpertOverlay topology. */
    inline ModelParityTopologyDefinition qwen35MoERocm2LocalTPTopology()
    {
        return {
            .test_id = "LocalTP_RCCL_2xROCm_ExpertOverlay",
            .kind = ModelParityTopologyKind::RankLocalTensorParallel,
            .participants = {
                {GlobalDeviceAddress::rocm(0), 0},
                {GlobalDeviceAddress::rocm(1), 0},
            },
            .collective = Collective::RCCL,
            .mpi_ranks = 1,
            .expert_overlay_plan = qwen35MoESingleDomainOverlayPlan(
                "qwen35_moe_rocm_local_tp",
                ExecutionDomainScope::RANK_LOCAL,
                CollectiveBackendType::RCCL,
                {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
        };
    }

    /** @return Two-rank/two-NUMA CPU ExpertOverlay topology. */
    inline ModelParityTopologyDefinition qwen35MoECPU2NodeTPTopology()
    {
        return {
            .test_id = "NodeTP_2xMPI_CPU_ExpertOverlay",
            .kind = ModelParityTopologyKind::NodeTensorParallel,
            .participants = {
                {GlobalDeviceAddress::cpu(0), 0},
                {GlobalDeviceAddress::cpu(1), 1},
            },
            .collective = Collective::MPI,
            .mpi_ranks = 2,
            .expert_overlay_plan = qwen35MoESingleDomainOverlayPlan(
                "qwen35_moe_cpu_node_tp",
                ExecutionDomainScope::NODE_LOCAL,
                CollectiveBackendType::MPI,
                {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}),
        };
    }

    /** @return Short-test economics that still require measured positive payoff. */
    inline MoERebalanceRuntimeConfig qwen35MoEDynamicParityEconomics()
    {
        MoERebalanceRuntimeConfig config;
        config.mode = MoERebalanceRuntimeMode::Dynamic;
        config.window_size = 1;
        config.max_window_size = 1;
        config.window_growth_factor = 1.0f;
        config.dynamic_imbalance_threshold_per_mille = 1000;
        config.dynamic_min_improvement_per_mille = 0;
        config.dynamic_max_swaps_per_layer = 20;
        config.dynamic_max_plan_entries_per_wave = 20;
        config.dynamic_min_window_activations = 0;
        config.device_min_load_spread_improvement = 0;
        config.device_min_load_spread_improvement_divisor = 0;
        config.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.device_min_foreign_rows_per_critical_path_payload_slot = 0;
        config.device_min_router_spread_improvement_per_payload_slot = 0;
        config.device_max_post_wave_load_spread_per_mille = 1000;
        config.device_maintenance_slack_tokens = 0;
        config.device_min_maintenance_period_tokens = 1;
        config.device_initial_maintenance_period_tokens = 1;
        config.migration_payoff_horizon_tokens = 65'536;
        config.release_raw_expert_weights = false;
        return config;
    }

    /**
     * @brief Join one model and topology to the standard precision/policy axes.
     */
    inline ModelParityDefinition qwen35MoEParityDefinition(
        ModelParityModelDefinition model,
        ModelParityTopologyDefinition topology,
        BackendThresholds thresholds)
    {
        ModelParityDefinition definition;
        definition.model = std::move(model);
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        const auto dynamic_policy = qwen35MoEDynamicParityEconomics();
        definition.dynamic_rebalance = {
            .economic_movement = dynamic_policy,
            .economic_movement_and_observed_speedup = dynamic_policy,
        };
        definition.collective_evidence_source =
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
        return definition;
    }

    /**
     * @brief Compose one 35B graph-native topology with every standard axis.
     *
     * The authenticated layer count determines the concurrent migration width:
     * one tier cycle per layer plus one same-priority skew cycle. This replaces
     * the former fixture-owned mutable value with declarative campaign policy.
     * The CUDA/ROCm/CPU topology also crosses the ordinary and four-row
     * segmented captured-prefill profiles.
     */
    inline ModelParityDefinition qwen35MoE35BGraphNativeParityDefinition(
        const Qwen35MoE35BOverlayTopologySpec &spec)
    {
        auto model = qwen35MoE35BQ4KXLParityModel();
        auto definition = qwen35MoEParityDefinition(
            model,
            qwen35MoE35BOverlayTopology(spec),
            qwen35MoEExpertOverlayThresholds());
        auto movement_policy =
            definition.dynamic_rebalance.economic_movement;
        movement_policy.window_size = 256;
        movement_policy.max_window_size = 256;
        movement_policy.migration_transfer_slots =
            static_cast<std::uint32_t>(model.transformer_layers + 1);
        movement_policy.dynamic_max_plan_entries_per_wave =
            std::max(
                movement_policy.dynamic_max_plan_entries_per_wave,
                static_cast<std::uint32_t>(
                    model.transformer_layers + 1));

        auto observed_speedup_policy = movement_policy;
        observed_speedup_policy.window_size =
            kQwen35MoEConvergenceHistogramWindowRows;
        observed_speedup_policy.max_window_size =
            kQwen35MoEConvergenceHistogramWindowRows;
        observed_speedup_policy.window_growth_factor = 1.0F;
        definition.dynamic_rebalance = {
            .economic_movement = std::move(movement_policy),
            .economic_movement_and_observed_speedup =
                std::move(observed_speedup_policy),
        };
        if (spec.segmented_prefill)
        {
            definition.features.prefill_graph = {
                ModelParityPrefillGraphPolicy{},
                ModelParityPrefillGraphPolicy{
                    .mode =
                        ModelParityPrefillGraphMode::SegmentedCaptured,
                    .captured_rows = 4,
                },
            };
        }
        definition.features.dynamic_speedup_witness =
            spec.dynamic_speedup_witness;
        return definition;
    }
} // namespace llaminar2::test::parity::qwen35moe
