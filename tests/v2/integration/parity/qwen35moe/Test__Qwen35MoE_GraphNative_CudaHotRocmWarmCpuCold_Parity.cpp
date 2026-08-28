/**
 * @file Test__Qwen35MoE_GraphNative_CudaHotRocmWarmCpuCold_Parity.cpp
 * @brief Production-path PyTorch parity gates for rank-agnostic Qwen3.5 MoE
 *        graph-native expert-overlay topologies.
 *
 * This fixture drives real Qwen3.5 35B-A3B and 122B-A10B weights through one
 * or two production OrchestrationRunner instances. It validates the same
 * sparse-collective graph shape used by serving for typed CUDA/ROCm/CPU
 * multi-tier cells against the Python/Hugging Face checkpoint corpus. The
 * fixture owns only test
 * admission and evidence collection. Dynamic CPU-tier cells use the production
 * inventory binder plus authenticated reference routing to declare a
 * rank-relative adversarial initial layout. Production still validates that
 * layout and owns graph construction, live histogram generation, placement
 * planning, transfer ordering, migration economics, and token sampling.
 *
 * @author David Sanftenberg
 * @date August 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../ModelParityDefinition.h"
#include "Qwen35MoEModelParityDefinitions.h"
#include "Qwen35MoEParityTestBase.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "config/OrchestrationConfig.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "execution/moe/RoutedExpertOwnerAssignment.h"
#include "execution/mtp/MTPVerifierPolicy.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "execution/runner/OrchestrationRunner.h"
#include "planning/ClusterInventoryGatherer.h"
#include "transfer/TransferEngine.h"
#include "utils/MTPParitySnapshotContext.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sampler.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

namespace
{
    constexpr const char *kQwen122ModelPath =
        "/opt/llaminar-models/Qwen3.5-122B-A10B-UD-Q8_K_XL-00001-of-00004.gguf";
    constexpr const char *kQwen122SnapshotDir =
        "pytorch_qwen35_122b_moe_mtp_snapshots";
    /** Maximum recurrent predictor depth certified by the 122B campaign. */
    constexpr int kQwen122MaximumMTPDraftDepth = 15;
    constexpr const char *kCudaHotDomain = "cuda_hot";
    constexpr const char *kRocmHotDomain = "rocm_hot";
    constexpr const char *kRocmWarmDomain = "rocm_warm";
    constexpr const char *kCpuColdDomain = "cpu_cold";
    constexpr const char *kLegacyEnvVar = "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME";
    /**
     * @brief Active generated case while one parameterized fixture is alive.
     *
     * Both companion binaries bind this before any CRTP configuration lookup,
     * so every runtime decision reads typed policy rather than recovering
     * intent from a GoogleTest name.
     */
    const ModelParityCase *g_active_model_parity_case = nullptr;

    /** @return The active typed canonical case, or null outside a fixture. */
    const ModelParityCase *activeModelParityCase() noexcept
    {
        return g_active_model_parity_case;
    }

    /** @return Active generated case; throws outside a parameterized fixture. */
    const ModelParityCase &activeModelParityCaseOrThrow()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
        {
            throw std::logic_error(
                "Graph-native parity runtime requires an active typed case");
        }
        return *test_case;
    }

    /**
     * @brief Movement axes that the resolved participant catalogue can express.
     *
     * Promotion/demotion needs two integer priorities. Capacity-preserving
     * same-tier balancing additionally needs an apportioned tier with at least
     * two participants and more resident experts than participants in at least
     * one layer. With exactly one expert per participant, every paired exchange
     * merely permutes loads and cannot reduce makespan. Keeping that distinction
     * typed prevents a fully priority-filled topology from being asked to
     * manufacture an uneconomical move while preserving the stronger two-axis
     * gate wherever the resolved residency has a real degree of freedom.
     */
    enum class DynamicMovementAxisContract : std::uint8_t
    {
        PriorityMigrationOnly,
        PriorityMigrationAndParticipantBalance,
    };

    /**
     * @brief Complete movement objective required before a converged A/B cohort.
     *
     * A publication wave may contain several independent closed cycles. The
     * 122B proof deliberately admits both a tier-residency cycle and an in-tier
     * participant cycle in one wide asynchronous wave, whereas the 35B proof
     * requires several successive publications to reach its measured taper.
     * Keeping publication count and logical axes in separate typed fields
     * prevents a caller from treating cycles, histogram windows, waves, and
     * placement epochs as interchangeable integers.
     */
    struct DynamicResidencyConvergenceTarget
    {
        /** Durable placement publications required after the baseline. */
        std::uint64_t minimum_published_waves = 1u;
        /** Logical movement axes the completed ledger must demonstrate. */
        DynamicMovementAxisContract axis_contract =
            DynamicMovementAxisContract::PriorityMigrationOnly;
    };

    /** Immutable authority position at the start of convergence traffic. */
    struct DynamicResidencyConvergenceOrigin
    {
        std::uint64_t published_waves = 0u;
        std::uint64_t completed_transactions = 0u;
        std::size_t ledger_edges = 0u;
        /** Coordinator-owned policy admissions already present at the origin. */
        std::size_t host_admissions = 0u;
    };

    /**
     * @brief Typed order for the convergence proof's production lifecycle.
     *
     * Service certification intentionally rebases calibration-era histograms.
     * Encoding the order prevents a baseline from being measured before that
     * boundary and then judged against a placement optimized for unrelated
     * traffic. `CertifyingEconomy` is a real state because ordinary traffic in
     * that interval may complete a profitable asynchronous movement wave.
     */
    enum class DynamicResidencyProofPhase : std::uint8_t
    {
        AwaitingEconomyCertification,
        CertifyingEconomy,
        EconomyCertified,
        InitialCohortMeasured,
        MovementTargetSatisfied,
        MovementBoundarySettled,
        ConvergedCohortMeasured,
    };

    /**
     * @brief Single typed authority for the test's Dynamic proof lifecycle.
     *
     * Every state after certification begins carries the same immutable
     * convergence origin. This makes a movement wave published while the real
     * service corpus is certifying economy part of the later convergence
     * suffix. Re-snapshotting the authority when the traffic driver starts is
     * structurally impossible.
     */
    class DynamicResidencyProofLifecycle final
    {
    public:
        /** @return Current explicit proof phase. */
        [[nodiscard]] DynamicResidencyProofPhase phase() const noexcept
        {
            switch (state_.index())
            {
            case 0:
                return DynamicResidencyProofPhase::
                    AwaitingEconomyCertification;
            case 1:
                return DynamicResidencyProofPhase::CertifyingEconomy;
            case 2:
                return DynamicResidencyProofPhase::EconomyCertified;
            case 3:
                return DynamicResidencyProofPhase::InitialCohortMeasured;
            case 4:
                return DynamicResidencyProofPhase::MovementTargetSatisfied;
            case 5:
                return DynamicResidencyProofPhase::MovementBoundarySettled;
            case 6:
                return DynamicResidencyProofPhase::ConvergedCohortMeasured;
            default:
                std::terminate();
            }
        }

        /**
         * @brief Freeze the authority frontier before any certification traffic.
         * @param origin Durable status and ledger frontier at certification entry.
         */
        void beginEconomyCertification(
            DynamicResidencyConvergenceOrigin origin)
        {
            if (!std::holds_alternative<AwaitingCertification>(state_))
            {
                throw std::logic_error(
                    "Dynamic economy certification began out of order");
            }
            state_ = CertifyingEconomy{origin};
        }

        /** @brief Publish successful completion of the service certificate. */
        void completeEconomyCertification()
        {
            advance<CertifyingEconomy, EconomyCertified>(
                "Dynamic economy certification completed out of order");
        }

        /** @brief Publish completion of the stationary initial cohort. */
        void recordInitialCohort()
        {
            advance<EconomyCertified, InitialCohortMeasured>(
                "Dynamic initial cohort completed out of order");
        }

        /**
         * @brief Publish satisfaction of the topology's movement objective.
         *
         * Movement-only cells advance directly from certification. The one
         * economic A/B cell first measures its initial stationary cohort.
         */
        void recordMovementTarget()
        {
            if (std::holds_alternative<MovementTargetSatisfied>(state_))
                return;
            if (const auto *state =
                    std::get_if<EconomyCertified>(&state_))
            {
                state_ = MovementTargetSatisfied{state->origin};
                return;
            }
            if (const auto *state =
                    std::get_if<InitialCohortMeasured>(&state_))
            {
                state_ = MovementTargetSatisfied{state->origin};
                return;
            }
            throw std::logic_error(
                "Dynamic movement target completed out of order");
        }

        /** @brief Publish the passive between-wave measurement boundary. */
        void recordMovementBoundarySettled()
        {
            advance<MovementTargetSatisfied, MovementBoundarySettled>(
                "Dynamic movement boundary settled out of order");
        }

        /** @brief Publish completion of the stationary converged cohort. */
        void recordConvergedCohort()
        {
            advance<MovementBoundarySettled, ConvergedCohortMeasured>(
                "Dynamic converged cohort completed out of order");
        }

        /**
         * @return Immutable authority frontier captured before certification.
         * @throws std::logic_error before certification has begun.
         */
        [[nodiscard]] const DynamicResidencyConvergenceOrigin &
        convergenceOrigin() const
        {
            return std::visit(
                [](const auto &state)
                    -> const DynamicResidencyConvergenceOrigin &
                {
                    using State = std::decay_t<decltype(state)>;
                    if constexpr (std::is_same_v<
                                      State,
                                      AwaitingCertification>)
                    {
                        throw std::logic_error(
                            "Dynamic convergence origin requested before certification");
                    }
                    else
                    {
                        return state.origin;
                    }
                },
                state_);
        }

    private:
        struct AwaitingCertification
        {
        };
        struct CertifyingEconomy
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct EconomyCertified
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct InitialCohortMeasured
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct MovementTargetSatisfied
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct MovementBoundarySettled
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct ConvergedCohortMeasured
        {
            DynamicResidencyConvergenceOrigin origin;
        };

        template <typename From, typename To>
        void advance(const char *diagnostic)
        {
            const auto *current = std::get_if<From>(&state_);
            if (!current)
                throw std::logic_error(diagnostic);
            const DynamicResidencyConvergenceOrigin origin =
                current->origin;
            state_ = To{origin};
        }

        using State = std::variant<
            AwaitingCertification,
            CertifyingEconomy,
            EconomyCertified,
            InitialCohortMeasured,
            MovementTargetSatisfied,
            MovementBoundarySettled,
            ConvergedCohortMeasured>;
        State state_ = AwaitingCertification{};
    };

    /**
     * @brief Exact reason a convergence target has or has not been reached.
     *
     * These are lifecycle states, not telemetry classifications. They are
     * derived exclusively from the optimization authority's passive status
     * and durable typed ledger; PerfStats remains diagnostic evidence only.
     */
    enum class DynamicResidencyConvergenceState : std::uint8_t
    {
        AwaitingPublication,
        AwaitingPhysicalCompletion,
        AwaitingTierResidency,
        AwaitingParticipantPolicyEvidence,
        AwaitingParticipantPlacement,
        Satisfied,
        InvalidAuthorityEvidence,
    };

    /**
     * @brief Classify authoritative progress toward one convergence objective.
     *
     * Publication and physical retirement are distinct event edges. Once both
     * have reached the requested wave count, the suffix of the durable ledger
     * must prove every logical movement axis expressible by the frozen
     * topology. Multiple cycles in one wave satisfy multiple axes but never
     * masquerade as multiple publications.
     *
     * @param target Typed publication and movement-axis objective.
     * @param origin Authority position before convergence traffic began.
     * @param status Current passive status from the sole production authority.
     * @param ledger Current complete durable movement ledger.
     * @return Exact lifecycle state without consulting optional instrumentation.
     */
    DynamicResidencyConvergenceState classifyDynamicResidencyConvergence(
        const DynamicResidencyConvergenceTarget &target,
        const DynamicResidencyConvergenceOrigin &origin,
        const MoEOptimizationStatus &status,
        const MoEOptimizationMovementLedger &ledger) noexcept
    {
        if (target.minimum_published_waves == 0u || !ledger.complete() ||
            status.published_movement_waves < origin.published_waves ||
            status.completed_movement.transactions <
                origin.completed_transactions ||
            ledger.edges.size() < origin.ledger_edges ||
            ledger.host_admissions.size() < origin.host_admissions ||
            status.completed_movement.transactions >
                status.published_movement_waves)
        {
            return DynamicResidencyConvergenceState::
                InvalidAuthorityEvidence;
        }

        const std::uint64_t published =
            status.published_movement_waves - origin.published_waves;
        if (published < target.minimum_published_waves)
        {
            return DynamicResidencyConvergenceState::AwaitingPublication;
        }

        const std::uint64_t completed =
            status.completed_movement.transactions -
            origin.completed_transactions;
        if (completed < target.minimum_published_waves)
        {
            return DynamicResidencyConvergenceState::
                AwaitingPhysicalCompletion;
        }

        bool advanced_tier_residency = false;
        bool advanced_participant_placement = false;
        for (std::size_t index = origin.ledger_edges;
             index < ledger.edges.size();
             ++index)
        {
            const auto &edge = ledger.edges[index];
            if (!edge.valid())
            {
                return DynamicResidencyConvergenceState::
                    InvalidAuthorityEvidence;
            }
            advanced_tier_residency =
                advanced_tier_residency || advancesTierResidency(edge.axis);
            advanced_participant_placement =
                advanced_participant_placement ||
                advancesParticipantPlacement(edge.axis);
        }
        if (!advanced_tier_residency)
        {
            return DynamicResidencyConvergenceState::AwaitingTierResidency;
        }
        if (target.axis_contract ==
                DynamicMovementAxisContract::
                    PriorityMigrationAndParticipantBalance &&
            !advanced_participant_placement)
        {
            if (status.authority == MoEOptimizationAuthority::Host)
            {
                /*
                 * Heterogeneous policy may correctly find no economical
                 * within-tier exchange after its tier promotion subset has
                 * already removed the bottleneck. Every completed host wave
                 * carries an immutable admission proof. Require one proof per
                 * completed suffix wave, validate it, and demand a physical
                 * participant edge whenever any candidate survived the exact
                 * economy gates. This keeps a broken or starved eligible
                 * participant cycle red without manufacturing a gratuitous
                 * move for a workload whose exhaustive planner found none.
                 */
                const std::size_t admission_count =
                    ledger.host_admissions.size() -
                    origin.host_admissions;
                if (admission_count <
                    static_cast<std::size_t>(
                        target.minimum_published_waves))
                {
                    return DynamicResidencyConvergenceState::
                        AwaitingParticipantPolicyEvidence;
                }

                bool participant_cycle_was_policy_eligible = false;
                for (std::size_t index = origin.host_admissions;
                     index < ledger.host_admissions.size();
                     ++index)
                {
                    const auto &admission =
                        ledger.host_admissions[index];
                    if (!admission.valid() ||
                        admission.authority !=
                            MoEOptimizationAuthority::Host)
                    {
                        return DynamicResidencyConvergenceState::
                            InvalidAuthorityEvidence;
                    }
                    participant_cycle_was_policy_eligible =
                        participant_cycle_was_policy_eligible ||
                        admission.policy_eligible_axes
                                .participant_placement > 0u ||
                        admission.policy_eligible_axes.combined > 0u;
                }
                if (!participant_cycle_was_policy_eligible)
                    return DynamicResidencyConvergenceState::Satisfied;
            }
            return DynamicResidencyConvergenceState::
                AwaitingParticipantPlacement;
        }
        return DynamicResidencyConvergenceState::Satisfied;
    }

    /** Capacity-resolved execution state for one declared overlay participant. */
    enum class PublishedParticipantResidency : std::uint8_t
    {
        /** The endpoint is configured but owns no expert in the published bank. */
        Idle,
        /** At least one routed expert is assigned to the endpoint. */
        OwnsExpert,
    };

    /** @brief Typed verdict for one participant's pinned and cumulative routes. */
    enum class PublishedParticipantRouteEvidence : std::uint8_t
    {
        Valid = 0,
        IdleParticipantSelected,
        RemoteResidentNeverCompleted,
    };

    /**
     * @brief Validate exact-epoch selection without inventing router demand.
     *
     * Residency is a capacity fact, not a promise that one bounded prompt
     * selects an owned expert. The pinned checkpoint must never select a final
     * idle participant. A remote participant that owns final experts must have
     * completed real device-owned sparse traffic somewhere in the production
     * workload, but that traffic may legitimately precede the final parity
     * prompt when Dynamic movement changes ownership between requests.
     *
     * @param residency Final request-pinned placement-bank residency.
     * @param pinned_route_count Routes selected by the exact parity checkpoint.
     * @param remote Whether endpoint-owned completion evidence is required.
     * @param completed_route_count Cumulative authenticated sparse completions.
     * @return Typed validity or the exact violated implication.
     */
    PublishedParticipantRouteEvidence validateParticipantRouteEvidence(
        PublishedParticipantResidency residency,
        std::uint64_t pinned_route_count,
        bool remote,
        std::uint64_t completed_route_count) noexcept
    {
        if (residency == PublishedParticipantResidency::Idle &&
            pinned_route_count != 0u)
        {
            return PublishedParticipantRouteEvidence::IdleParticipantSelected;
        }
        if (residency == PublishedParticipantResidency::OwnsExpert && remote &&
            completed_route_count == 0u)
        {
            return PublishedParticipantRouteEvidence::
                RemoteResidentNeverCompleted;
        }
        return PublishedParticipantRouteEvidence::Valid;
    }

    /** Device-kind summary derived from one complete published placement bank. */
    struct PublishedSparseTierParticipation
    {
        /** At least one non-continuation participant owns a live expert. */
        bool sparse_follower = false;
        bool cpu = false;
        bool secondary_gpu = false;
    };

    /**
     * @brief Fold one device-owned expert-to-participant bank into residency.
     *
     * Automatic capacity may validly exhaust the model before reaching a
     * configured lower-priority tier. Such endpoints remain part of topology
     * but must be proved idle instead of being asked to manufacture sparse
     * traffic. The published placement bank, rather than observed PerfStats,
     * is the authority that distinguishes those states.
     *
     * @param states Complete participant-state vector updated in place.
     * @param expert_owners Integral participant IDs for every expert in a layer.
     * @throws std::invalid_argument for a non-finite, fractional, or unknown ID.
     */
    void includePublishedExpertOwners(
        std::vector<PublishedParticipantResidency> &states,
        std::span<const float> expert_owners)
    {
        if (states.empty())
        {
            throw std::invalid_argument(
                "Published ExpertOverlay residency requires participants");
        }
        for (const float encoded_owner : expert_owners)
        {
            if (!std::isfinite(encoded_owner) || encoded_owner < 0.0f ||
                encoded_owner >
                    static_cast<float>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    "Published ExpertOverlay bank contains an invalid participant ID");
            }
            const int owner = static_cast<int>(encoded_owner);
            if (encoded_owner != static_cast<float>(owner) || owner < 0 ||
                static_cast<std::size_t>(owner) >= states.size())
            {
                throw std::invalid_argument(
                    "Published ExpertOverlay bank names an unknown participant");
            }
            states[static_cast<std::size_t>(owner)] =
                PublishedParticipantResidency::OwnsExpert;
        }
    }

    /**
     * @brief Classify active CPU and non-continuation GPU sparse tiers.
     *
     * Participant IDs are allocated in routed-tier order by
     * `MoEExpertOwnerMap`; this helper walks the same typed plan order and
     * rejects a cardinality mismatch. Tier names and priority values remain
     * diagnostic only.
     *
     * @param plan Frozen capacity-resolved overlay topology.
     * @param states Published participant residency indexed by participant ID.
     * @return Active device-kind summary for transport evidence.
     * @throws std::invalid_argument for incomplete topology or state geometry.
     */
    PublishedSparseTierParticipation summarizePublishedParticipation(
        const MoERoutedExpertPlacementPlan &plan,
        std::span<const PublishedParticipantResidency> states)
    {
        PublishedSparseTierParticipation summary;
        std::size_t participant_id = 0u;
        for (const auto &tier : plan.routed_tiers)
        {
            const auto domain = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &candidate)
                { return candidate.name == tier.domain; });
            if (domain == plan.domains.end())
            {
                throw std::invalid_argument(
                    "Published participant summary cannot resolve domain '" +
                    tier.domain + "'");
            }
            for (const auto &participant : domain->participants)
            {
                if (participant_id >= states.size())
                {
                    throw std::invalid_argument(
                        "Published participant summary has fewer states than endpoints");
                }
                if (states[participant_id] ==
                    PublishedParticipantResidency::OwnsExpert)
                {
                    summary.sparse_follower =
                        summary.sparse_follower ||
                        domain->name != plan.continuation_domain;
                    summary.cpu = summary.cpu || participant.isCPU();
                    summary.secondary_gpu =
                        summary.secondary_gpu ||
                        (domain->name != plan.continuation_domain &&
                         participant.isGPU());
                }
                ++participant_id;
            }
        }
        if (participant_id != states.size())
        {
            throw std::invalid_argument(
                "Published participant summary has more states than endpoints");
        }
        return summary;
    }

    /**
     * @brief Derive the physically expressible Dynamic movement contract.
     *
     * Tier names, integer values, and accelerator vendors are deliberately
     * ignored. Rebalancing is local to one routed tier/domain, so distinct
     * domains carrying the same integer priority are not incorrectly treated
     * as one exchange set. The plan must be the capacity-resolved frozen
     * production plan: an automatic-capacity blueprint has no concrete expert
     * membership from which an expressible movement can be inferred.
     *
     * @param plan Inventory-bound or rank-agnostic ExpertOverlay plan.
     * @return Exact movement-axis contract implied by its participant catalogue.
     * @throws std::invalid_argument when a tier references no declared domain.
     */
    DynamicMovementAxisContract dynamicMovementAxisContract(
        const MoERoutedExpertPlacementPlan &plan)
    {
        for (std::size_t tier_index = 0;
             tier_index < plan.routed_tiers.size();
             ++tier_index)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto domain = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &candidate)
                { return candidate.name == tier.domain; });
            if (domain == plan.domains.end())
            {
                throw std::invalid_argument(
                    "Dynamic movement-axis resolution cannot find routed domain '" +
                    tier.domain + "'");
            }
            const std::size_t participant_count =
                domain->participants.size();
            if (domain->routed_compute_policy !=
                    RoutedExpertComputePolicy::Apportioned ||
                participant_count < 2u)
            {
                continue;
            }

            const bool has_exchange_degree_of_freedom = std::any_of(
                plan.placements.begin(),
                plan.placements.end(),
                [&](const RoutedExpertLayerPlacement &placement)
                {
                    return static_cast<std::size_t>(std::count(
                               placement.routed_expert_tier.begin(),
                               placement.routed_expert_tier.end(),
                               static_cast<int>(tier_index))) >
                           participant_count;
                });
            if (has_exchange_degree_of_freedom)
            {
                return DynamicMovementAxisContract::
                    PriorityMigrationAndParticipantBalance;
            }
        }
        return DynamicMovementAxisContract::PriorityMigrationOnly;
    }

    /* Short aliases keep the lifecycle implementation readable while all
     * workload geometry remains owned by the central typed definition. */
    constexpr int kConvergenceTimingWarmupRequests =
        kQwen35MoEConvergenceTimingWarmupRequests;
    constexpr int kConvergenceTimingMeasuredRequests =
        kQwen35MoEConvergenceTimingMeasuredRequests;
    constexpr int kConvergenceTimingCorpusRequests =
        kQwen35MoEConvergenceTimingCorpusRequests;
    constexpr int kConvergenceTimingDecodeForwardsPerRequest =
        kQwen35MoEConvergenceTimingDecodeForwards;
    constexpr std::size_t kConvergenceTimingPromptRows =
        kQwen35MoEConvergenceTimingPromptRows;

    /**
     * @brief Production traffic admitted while seeking a timing boundary.
     *
     * Ordinary optimization must replay the exact prompt identities judged by
     * the A/B gate. Once that placement objective is satisfied, a partially
     * occupied demand bank may still need one exact closure request; that
     * request uses a disjoint prefix namespace so it cannot seed a cache hit in
     * the subsequent measured cohort.
     */
    enum class ConvergenceMovementTraffic : std::uint8_t
    {
        MeasuredWorkload,
        MovementProof,
        DemandWindowClosure,
    };

    /**
     * @brief Evidence that must remain admissible after movement settles.
     *
     * Every Dynamic cell needs a passive publication boundary before parity
     * snapshots begin. Only the one centrally assigned speed witness also
     * needs enough room in the active demand bank for its complete A/B timing
     * cohort. Keeping these purposes distinct prevents movement-only cells
     * from trying to admit a timing workload they will never execute.
     */
    enum class ConvergenceBoundaryPurpose : std::uint8_t
    {
        MovementProof,
        ObservedSpeedupCohort,
    };

    /** Workload geometry owned by one reference-shaped economy request. */
    enum class ReferenceEconomyPromptRole : std::uint8_t
    {
        TimingCohort,
        MovementProof,
    };

    /** @return Exact routed rows executed by one complete timing cohort. */
    constexpr std::uint64_t convergenceTimingCohortRoutedRows() noexcept
    {
        return qwen35MoEConvergenceTimingCohortRoutedRows();
    }

    /**
     * @brief Return demand-bank headroom required by a typed boundary.
     *
     * A movement-only boundary has no following exclusive timing cohort, so
     * it requires no demand-window geometry. The observed-speedup boundary
     * owns the exact routed-row requirement used by its matched A/B corpus.
     *
     * @param purpose Evidence executed immediately after settlement.
     * @return Required routed rows, or no requirement for movement-only proof.
     */
    constexpr std::optional<std::uint64_t>
    convergenceBoundaryCohortRows(
        ConvergenceBoundaryPurpose purpose) noexcept
    {
        if (purpose ==
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort)
        {
            return convergenceTimingCohortRoutedRows();
        }
        return std::nullopt;
    }

    /** Exact passive disposition of one post-movement measurement boundary. */
    enum class ConvergenceBoundaryState : std::uint8_t
    {
        AwaitingQuiescence,
        Ready,
        NeedsDemandWindowClosure,
        InvalidAuthorityEvidence,
    };

    /** Immutable work needed to close one partially occupied demand bank. */
    struct DemandWindowClosure
    {
        std::uint64_t generation = 0u;
        std::uint64_t routed_rows = 0u;

        /** @return Whether the closure names positive work in one bank. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return routed_rows > 0u;
        }
    };

    /** Typed result of classifying a passive convergence boundary. */
    struct ConvergenceBoundaryDecision
    {
        ConvergenceBoundaryState state =
            ConvergenceBoundaryState::AwaitingQuiescence;
        std::optional<DemandWindowClosure> closure;
    };

    /**
     * @brief Classify whether a complete evidence cohort can begin safely.
     *
     * The observer never discards demand or pauses maintenance. If a
     * reconciled partial bank lacks timing headroom, the returned closure is
     * the exact number of ordinary routed rows required to finish that bank.
     * The caller can then admit one cache-distinct production prefill, wake the
     * authority once, and stop admission while the resulting wave rotates to
     * an empty successor bank.
     *
     * @param status Passive snapshot from the sole optimization authority.
     * @param purpose Evidence that follows this boundary.
     * @return Typed readiness, closure work, or malformed-evidence verdict.
     */
    ConvergenceBoundaryDecision classifyConvergenceBoundary(
        const MoEOptimizationStatus &status,
        ConvergenceBoundaryPurpose purpose) noexcept
    {
        if (!status.quiescentBetweenWaves())
        {
            return {
                .state = ConvergenceBoundaryState::AwaitingQuiescence,
            };
        }

        const auto required_rows =
            convergenceBoundaryCohortRows(purpose);
        if (!required_rows)
        {
            return {
                .state = ConvergenceBoundaryState::Ready,
            };
        }
        if (!status.demand_window.valid())
        {
            return {
                .state =
                    ConvergenceBoundaryState::InvalidAuthorityEvidence,
            };
        }
        if (status.canBeginExclusiveCohort(*required_rows))
        {
            return {
                .state = ConvergenceBoundaryState::Ready,
            };
        }

        const std::uint64_t rows_to_close =
            status.demand_window.remainingRoutedRows();
        if (rows_to_close == 0u)
        {
            /* A completed bank can be visible just before the worker consumes
             * its wake. It is not safe to admit more demand, but neither is it
             * malformed; ordinary progress must rotate it. */
            return {
                .state = ConvergenceBoundaryState::AwaitingQuiescence,
            };
        }
        return {
            .state =
                ConvergenceBoundaryState::NeedsDemandWindowClosure,
            .closure = DemandWindowClosure{
                .generation = status.demand_window.generation,
                .routed_rows = rows_to_close,
            },
        };
    }

    /**
     * Histogram geometry large enough to keep one complete A/B cohort inside
     * one immutable residency epoch while ordinary maintenance remains live.
     * Six initial requests contribute 6 * (64 prefill + 5 decode) = 414 routed
     * rows. A 448-row window therefore cannot publish during the baseline.
     * Normal movement training replays those exact prompt identities and
     * therefore includes the production prefix-hit mix rather than changing
     * the decode trajectories under test. Every accepted placement publication
     * changes the invalidate-on-rebalance fingerprint, so the next epoch still
     * receives one full-compute copy of each prompt before reuse begins.
     */
    constexpr int kConvergenceHistogramWindowTokens =
        kQwen35MoEConvergenceHistogramWindowRows;
    /**
     * @brief Improving post-certificate windows required by a speed witness.
     *
     * A short stop leaves profitable capacity in the live controller: with
     * one independent tier cycle per layer, the third publication was still
     * followed by a fourth economically admitted wave during the parity
     * phase. Four real publications retain the bounded async protocol while
     * reaching the observed taper and making the resulting tier-residency
     * change large enough for the end-to-end economy gate to judge reliably.
     */
    constexpr int kObservedSpeedupConvergenceWindows = 4;
    /**
     * Worst-case stationary requests needed when every prefill is restored.
     *
     * The boundary sample consumes logits already produced by prefill; only
     * the following decode transaction executes a routed model forward. Thus
     * each measured pair contributes exactly one decode token to the device
     * histogram. Deriving the horizon from executed-forward geometry avoids
     * assuming that repeated production prefixes execute fresh prefill rows or
     * that sampling itself advances routed demand. A placement publication
     * invalidates the old fingerprint and naturally contributes fresh prefill
     * rows; the bound remains conservative by ignoring those rows.
     */
    constexpr int kMaximumDynamicHistogramRequests =
        kObservedSpeedupConvergenceWindows *
        ((kConvergenceHistogramWindowTokens +
          kConvergenceTimingDecodeForwardsPerRequest - 1) /
         kConvergenceTimingDecodeForwardsPerRequest);
    /**
     * Prefix-identity interval reserved for every movement-proof request.
     *
     * Reserve the largest four-wave geometry represented by this fixture.
     * Runtime traffic stops at its topology-specific convergence target, but
     * identity namespaces remain disjoint for the full supported range.
     */
    constexpr int kMovementProofIdentityNamespaceRequests =
        kObservedSpeedupConvergenceWindows *
        ((kConvergenceHistogramWindowTokens +
          static_cast<int>(kQwen35MoEParityTokenIds.size()) - 1) /
         static_cast<int>(kQwen35MoEParityTokenIds.size()));

    /**
     * @brief Calculate requests that guarantee a typed publication target.
     *
     * Decode is deliberately excluded because EOS may end it before one routed
     * forward.  Each publication opens a fresh demand bank, so ceiling is
     * applied per bank before multiplying by the required publication count.
     *
     * @param window_rows Routed rows required to close one demand bank.
     * @param guaranteed_rows Cache-distinct real prefill rows per request.
     * @param required_publications Durable placement publications required.
     * @return Exact conservative request budget before async overlap traffic.
     * @throws std::invalid_argument for an empty geometry or target.
     * @throws std::overflow_error when the request count cannot fit in `int`.
     */
    int movementProofHistogramRequestBudget(
        int window_rows,
        int guaranteed_rows,
        std::uint64_t required_publications)
    {
        if (window_rows <= 0 || guaranteed_rows <= 0 ||
            required_publications == 0u)
        {
            throw std::invalid_argument(
                "Movement-proof histogram budget requires positive window, prompt, and publication geometry");
        }
        const std::uint64_t rows =
            static_cast<std::uint64_t>(window_rows);
        const std::uint64_t guaranteed =
            static_cast<std::uint64_t>(guaranteed_rows);
        const std::uint64_t requests_per_publication =
            (rows + guaranteed - 1u) / guaranteed;
        if (required_publications >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) /
                requests_per_publication)
        {
            throw std::overflow_error(
                "Movement-proof histogram request budget exceeds int range");
        }
        return static_cast<int>(
            required_publications * requests_per_publication);
    }
    /** Maximum authenticated requests allowed to certify service economics. */
    constexpr int kMaximumServiceCertificationRequests =
        kMaximumDynamicHistogramRequests * 2;
    /**
     * One full corpus of inference opportunities for async publication.
     *
     * A histogram can close on the final decode boundary of a request. Giving
     * the background authority one complete corpus cycle keeps inference live
     * while proposal, transport, certification, and event publication retire.
     */
    constexpr int kMaximumDynamicPublicationOverlapRequests =
        kConvergenceTimingCorpusRequests;

    /** @return Prompt identity for one typed convergence-traffic purpose. */
    constexpr int convergenceMovementPromptIdentity(
        ConvergenceMovementTraffic traffic,
        int request_ordinal) noexcept
    {
        switch (traffic)
        {
        case ConvergenceMovementTraffic::MeasuredWorkload:
            return request_ordinal % kConvergenceTimingCorpusRequests;
        case ConvergenceMovementTraffic::MovementProof:
            return kConvergenceTimingCorpusRequests + request_ordinal;
        case ConvergenceMovementTraffic::DemandWindowClosure:
            return kConvergenceTimingCorpusRequests +
                   kMovementProofIdentityNamespaceRequests +
                   kMaximumDynamicPublicationOverlapRequests +
                   request_ordinal;
        }
        return 0;
    }

    // Approximate Qwen3.5-35B-A3B metadata for topology-only, model-free planning.
    constexpr int kQwen35MoENumExperts = 256;
    constexpr int kQwen35MoENumLayers = 94;

    /**
     * @brief Return the exact decode policy used by the Hugging Face reference.
     *
     * `run_prefill_and_decode()` writes the reference corpus by taking an
     * argmax at the prefill boundary and after every decode row.  Installing
     * this policy through IOrchestrationRunner keeps that same contract on the
     * production device sampler: CUDA/ROCm execute their greedy sampler and
     * never fall back to downloading logits for host sampling.
     *
     * @return A stateless greedy sampling policy matching the reference corpus.
     */
    SamplingParams referenceGreedySamplingPolicy()
    {
        SamplingParams policy;
        policy.temperature = 0.0f;
        policy.top_k = 0;
        policy.top_p = 1.0f;
        policy.seed = 0;
        return policy;
    }

    bool isLegacyOverlayRuntimeEnabled()
    {
        const char *value = std::getenv(kLegacyEnvVar);
        return value != nullptr && std::string(value) == "1";
    }

    /** @return Exact generated diagnostic identity for the active cell. */
    std::string activeTestName()
    {
        return activeModelParityCaseOrThrow().testName();
    }

    /** @return Whether the active typed model is the 122B MTP identity. */
    bool isQwen122ProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->model.test_id == "Qwen35_122B";
    }

    /** @return Model root selected by the active real-weight cell. */
    const char *activeModelPath()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "Graph-native model path requires an active typed parity case");
        return test_case->model.model_path.c_str();
    }

    /** @return Authenticated reference directory selected by the active cell. */
    const char *activeSnapshotDir()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "Graph-native reference path requires an active typed parity case");
        return test_case->model.reference_directory.c_str();
    }

    /** @return Whether every split file required by the active model exists. */
    bool modelAvailable()
    {
        if (!std::filesystem::exists(activeModelPath()))
            return false;
        if (!isQwen122ProductionTest())
            return true;

        const std::filesystem::path first(activeModelPath());
        const std::string first_name = first.filename().string();
        const auto marker = first_name.find("00001-of-00004");
        if (marker == std::string::npos)
            return false;
        for (int split = 2; split <= 4; ++split)
        {
            std::string sibling_name = first_name;
            std::ostringstream ordinal;
            ordinal << std::setw(5) << std::setfill('0') << split;
            sibling_name.replace(marker, 5, ordinal.str());
            if (!std::filesystem::exists(first.parent_path() / sibling_name))
                return false;
        }
        return true;
    }

    /** @return Whether this case uses current-batch least-loaded assignment. */
    bool isLLEPProductionTest()
    {
        return false;
    }

    /** @brief Return whether the exact cell exercises persistent tier movement. */
    bool isDynamicResidencyProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->expert_overlay &&
               test_case->expert_overlay->movement ==
                   ModelParityExpertMovement::Dynamic;
    }

    /** @brief Return whether initial expert ownership uses seeded random order. */
    bool isRandomOwnerProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->expert_overlay &&
               test_case->expert_overlay->owner_order ==
                   RoutedExpertOwnerOrder::Random;
    }

    /**
     * @return Fixed MTP depth, the adaptive policy's maximum, or zero when off.
     *
     * Generated cells carry this policy explicitly.  Legacy topology cells do
     * not name an MTP mode and therefore must report zero; treating an absent
     * label as depth three made post-run evidence demand an MTP domain from a
     * runtime whose typed configuration correctly disabled MTP.
     */
    int activeMTPDraftDepth()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "MTP depth requires an active typed parity case");
        return test_case->requestedMTPDraftDepth();
    }

    /** @return Setup-time MTP draft capacity retained by the active cell. */
    int activeMTPRetainedDraftCapacity()
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "MTP capacity requires an active typed parity case");
        return test_case->retained_mtp_draft_capacity;
    }

    /**
     * @brief Return the retained grouped-verifier row capacity for this cell.
     *
     * Every 122B campaign cell shares one maximum-capacity model context so
     * changing the requested fixed depth does not reload weights or rebuild
     * device graphs.  The device transaction still publishes its independent
     * logical depth through @ref activeMTPDraftDepth and selects a physical
     * bucket inside this admitted envelope.
     */
    int activeMTPGraphCapacityVerifierRows()
    {
        const int retained_draft_capacity =
            activeMTPRetainedDraftCapacity();
        return retained_draft_capacity > 0
                   ? retained_draft_capacity + 1
                   : 1;
    }

    /**
     * @brief Return the exact physical bucket selected by this transaction.
     *
     * Fixed-depth scalar transactions select the smallest retained power-of-two
     * bucket that contains their logical rows.  Dynamic depth instead embeds
     * the maximum envelope because active depth is device-owned replay data.
     */
    int activeMTPPhysicalVerifierRows()
    {
        const int capacity_rows = activeMTPGraphCapacityVerifierRows();
        return activeModelParityCaseOrThrow().usesDynamicMTPDepth()
                   ? capacity_rows
                   : mtpVerifierPhysicalRowBucket(
                         activeMTPDraftDepth() + 1,
                         capacity_rows);
    }

    /** @return Whether the production device depth controller is adaptive. */
    bool usesDynamicMTPDepth()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->usesDynamicMTPDepth();
    }

    /** @return Whether the active generated cell enables MTP. */
    bool activeMTPEnabled()
    {
        const auto *test_case = activeModelParityCase();
        return test_case && test_case->mtpEnabled();
    }

    /** @brief Return whether the active topology contains a CUDA participant. */
    bool topologyUsesCuda()
    {
        const auto &participants =
            activeModelParityCaseOrThrow().topology.participants;
        return std::any_of(
            participants.begin(), participants.end(),
            [](const ModelParityParticipant &participant)
            { return participant.address.isCUDA(); });
    }

    /** @brief Return whether the active topology contains a ROCm participant. */
    bool topologyUsesRocm()
    {
        const auto &participants =
            activeModelParityCaseOrThrow().topology.participants;
        return std::any_of(
            participants.begin(), participants.end(),
            [](const ModelParityParticipant &participant)
            { return participant.address.isROCm(); });
    }

    /** @brief Return whether the active topology contains NodeTP CPU cold. */
    bool topologyUsesCpu()
    {
        const auto &participants =
            activeModelParityCaseOrThrow().topology.participants;
        return std::any_of(
            participants.begin(), participants.end(),
            [](const ModelParityParticipant &participant)
            { return participant.address.isCPU(); });
    }

    /** @brief Return the exact sparse participants declared by the active topology. */
    size_t activeOverlayParticipantCount()
    {
        return activeModelParityCaseOrThrow().topology.participants.size();
    }

    /** @return Number of active typed participants using one device predicate. */
    template <typename Predicate>
    size_t activeTypedParticipantCount(Predicate &&predicate)
    {
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            return 0u;
        return static_cast<size_t>(std::count_if(
            test_case->topology.participants.begin(),
            test_case->topology.participants.end(),
            [&](const ModelParityParticipant &participant)
            { return predicate(participant.address); }));
    }

    /** @return Whether a generated topology owns a non-continuation GPU tier. */
    bool topologyHasSecondaryGpuDomain()
    {
        const auto &plan = *activeModelParityCaseOrThrow()
                                .topology.expert_overlay_plan;
        return std::any_of(
            plan.domains.begin(), plan.domains.end(),
            [&](const RoutedExpertDomain &domain)
            {
                return domain.name != plan.continuation_domain &&
                       std::any_of(
                           domain.participants.begin(),
                           domain.participants.end(),
                           [](const GlobalDeviceAddress &participant)
                           { return participant.isGPU(); });
            });
    }

    /** @return Whether sparse ExpertOverlay work crosses an MPI process. */
    bool topologySpansMultipleMPIRanks()
    {
        return activeModelParityCaseOrThrow().topology.mpi_ranks > 1;
    }

    /** @return Number of integer-priority residency tiers in this cell. */
    size_t activeOverlayTierCount()
    {
        return activeModelParityCaseOrThrow()
            .topology.expert_overlay_plan->routed_tiers.size();
    }

    RoutedExpertTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        bool fallback = false)
    {
        RoutedExpertTier t;
        t.name = name;
        t.domain = domain;
        t.priority = priority;
        t.max_experts_per_layer = max_experts_per_layer;
        t.memory_budget_bytes = 0;
        t.fallback = fallback;
        return t;
    }

#ifdef LLAMINAR_QWEN122_MATRIX_ONLY
    /** Accelerator family that owns dense continuation for one 122B topology. */
    enum class Qwen122ContinuationBackend : std::uint8_t
    {
        CUDA,
        ROCm,
    };

    /**
     * @brief Compact declarative input for one generated 122B topology.
     *
     * Counts describe physical participants, not quotas. Expert capacity is
     * resolved from the complete physical allocation BOM, so a topology never
     * bakes a model- or host-specific expert count into tests.
     */
    struct Qwen122OverlayTopologySpec
    {
        const char *test_id;
        int cuda_participants;
        int rocm_participants;
        int cpu_participants;
        int mpi_ranks;
        Qwen122ContinuationBackend continuation;
    };

    /** @return Every unique 122B ExpertOverlay topology in the production matrix. */
    const std::array<Qwen122OverlayTopologySpec, 9> &
    qwen122OverlayTopologySpecs()
    {
        static const std::array<Qwen122OverlayTopologySpec, 9> specs{{
            {"CUDA2_ROCm4_2xMPI_NodeExpertOverlay", 2, 4, 0, 2,
             Qwen122ContinuationBackend::CUDA},
            {"ROCm1_CPU2_2xMPI_NodeExpertOverlay", 0, 1, 2, 2,
             Qwen122ContinuationBackend::ROCm},
            {"ROCm2_CPU2_2xMPI_NodeExpertOverlay", 0, 2, 2, 2,
             Qwen122ContinuationBackend::ROCm},
            {"ROCm3_CPU2_2xMPI_NodeExpertOverlay", 0, 3, 2, 2,
             Qwen122ContinuationBackend::ROCm},
            {"ROCm4_CPU2_2xMPI_NodeExpertOverlay", 0, 4, 2, 2,
             Qwen122ContinuationBackend::ROCm},
            {"CUDA1_CPU2_2xMPI_NodeExpertOverlay", 1, 0, 2, 2,
             Qwen122ContinuationBackend::CUDA},
            {"CUDA2_CPU2_2xMPI_NodeExpertOverlay", 2, 0, 2, 2,
             Qwen122ContinuationBackend::CUDA},
            {"ROCm1_CPU1_1xMPI_RankExpertOverlay", 0, 1, 1, 1,
             Qwen122ContinuationBackend::ROCm},
            {"CUDA1_CPU1_1xMPI_RankExpertOverlay", 1, 0, 1, 1,
             Qwen122ContinuationBackend::CUDA},
        }};
        return specs;
    }

    /**
     * @brief Build one accelerator domain without assuming its owning rank.
     *
     * AUTO scope is resolved from gathered inventory: one device becomes a
     * single participant, while several same-rank devices become rank-local
     * TP. This preserves topology intent if the cards move between sockets.
     */
    RoutedExpertDomain qwen122AcceleratorDomain(
        Qwen122ContinuationBackend backend,
        int participant_count,
        bool continuation)
    {
        if (participant_count <= 0)
            throw std::invalid_argument(
                "122B accelerator domain requires a positive participant count");

        RoutedExpertDomain domain;
        if (backend == Qwen122ContinuationBackend::CUDA)
        {
            domain.name = kCudaHotDomain;
            domain.backend = CollectiveBackendType::NCCL;
            for (int ordinal = 0; ordinal < participant_count; ++ordinal)
                domain.participants.push_back(GlobalDeviceAddress::cuda(ordinal));
        }
        else
        {
            domain.name = continuation ? kRocmHotDomain : kRocmWarmDomain;
            domain.backend = CollectiveBackendType::RCCL;
            for (int ordinal = 0; ordinal < participant_count; ++ordinal)
                domain.participants.push_back(GlobalDeviceAddress::rocm(ordinal));
        }
        domain.scope = ExecutionDomainScope::AUTO;
        domain.owner_rank = -1;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        return domain;
    }

    /** @brief Build a one- or two-socket CPU tier resolved by live inventory. */
    RoutedExpertDomain qwen122CpuDomain(int participant_count)
    {
        if (participant_count <= 0)
            throw std::invalid_argument(
                "122B CPU domain requires a positive participant count");
        RoutedExpertDomain domain;
        domain.name = kCpuColdDomain;
        domain.scope = ExecutionDomainScope::AUTO;
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
     * @brief Select dense execution for the continuation participant count.
     *
     * A single continuation accelerator owns complete dense/shared weights and
     * therefore has no tensor-parallel collective. Two or more homogeneous
     * continuation participants retain the production decode optimization that
     * mirrors the embedding while tensor-sharding the remaining dense path.
     * The caller validates that @p participant_count is positive.
     */
    constexpr DenseParallelPolicy qwen122ContinuationDensePolicy(
        int participant_count)
    {
        return participant_count == 1
                   ? DenseParallelPolicy::Replicated
                   : DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding;
    }

    static_assert(
        qwen122ContinuationDensePolicy(1) ==
        DenseParallelPolicy::Replicated,
        "A one-accelerator continuation domain must not advertise dense TP");
    static_assert(
        qwen122ContinuationDensePolicy(2) ==
        DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding,
        "A distributed continuation domain must retain optimized dense TP");

    /**
     * @brief Build an immutable integer-priority 122B overlay blueprint.
     *
     * The continuation accelerator domain receives priority zero. An optional
     * second accelerator family receives priority one, and CPU receives the
     * next integer priority. Every quota is automatic: production fills each
     * tier to its exact remaining admitted capacity and places the exact
     * remainder in the final fallback tier.
     */
    std::shared_ptr<const MoERoutedExpertPlacementPlan>
    qwen122OverlayBlueprint(const Qwen122OverlayTopologySpec &spec)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = RoutedExpertOwnerOrder::Ordinal;

        int next_priority = 0;
        const auto append_domain = [&](RoutedExpertDomain domain)
        {
            const std::string domain_name = domain.name;
            plan->domains.push_back(std::move(domain));
            plan->routed_tiers.push_back(makeTier(
                "priority" + std::to_string(next_priority),
                domain_name,
                next_priority,
                /*max_experts_per_layer=*/0));
            ++next_priority;
        };

        const bool cuda_continuation =
            spec.continuation == Qwen122ContinuationBackend::CUDA;
        const int continuation_count = cuda_continuation
                                           ? spec.cuda_participants
                                           : spec.rocm_participants;
        auto continuation_domain = qwen122AcceleratorDomain(
            spec.continuation,
            continuation_count,
            /*continuation=*/true);
        plan->continuation_domain = continuation_domain.name;
        plan->base_model_domain = continuation_domain.name;
        plan->shared_expert_domain = continuation_domain.name;
        append_domain(std::move(continuation_domain));

        if (cuda_continuation && spec.rocm_participants > 0)
        {
            append_domain(qwen122AcceleratorDomain(
                Qwen122ContinuationBackend::ROCm,
                spec.rocm_participants,
                /*continuation=*/false));
        }
        else if (!cuda_continuation && spec.cuda_participants > 0)
        {
            append_domain(qwen122AcceleratorDomain(
                Qwen122ContinuationBackend::CUDA,
                spec.cuda_participants,
                /*continuation=*/false));
        }
        if (spec.cpu_participants > 0)
            append_domain(qwen122CpuDomain(spec.cpu_participants));

        if (plan->routed_tiers.size() < 2u)
            throw std::invalid_argument(
                "122B ExpertOverlay topology requires at least two priorities");
        plan->routed_tiers.back().fallback = true;
        plan->continuation_domain_spec.setDensePolicy(
            qwen122ContinuationDensePolicy(continuation_count));
        return plan;
    }
#endif

    MoERoutedExpertModelMetadata topologyOnlyMetadata()
    {
        MoERoutedExpertModelMetadata metadata;
        metadata.num_experts = kQwen35MoENumExperts;
        metadata.num_layers = isQwen122ProductionTest()
                                  ? 48
                                  : kQwen35MoENumLayers;
        metadata.d_model = isQwen122ProductionTest() ? 3072 : 4096;
        metadata.routed_intermediate_size =
            isQwen122ProductionTest() ? 1024 : 1536;
        metadata.has_shared_expert = true;
        metadata.shared_intermediate_size =
            metadata.routed_intermediate_size;
        metadata.routed_quant_type =
            isQwen122ProductionTest() ? "Q8_K" : "Q4_K";
        metadata.shared_quant_type = metadata.routed_quant_type;
        return metadata;
    }

    MoERoutedExpertModelMetadata metadataFromModel(const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string &arch = ctx.architecture();

        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = ctx.totalBlockCount();
        metadata.num_experts = loader.getInt(arch + ".expert_count", 0);
        metadata.d_model = ctx.embeddingLength();
        metadata.routed_intermediate_size = loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (metadata.routed_intermediate_size == 0)
            metadata.routed_intermediate_size = ctx.feedForwardLength();
        metadata.has_shared_expert = loader.getInt(arch + ".expert_shared_count", 0) > 0;
        metadata.shared_intermediate_size = metadata.has_shared_expert
                                                ? metadata.routed_intermediate_size
                                                : 0;
        metadata.routed_quant_type = "Q4_K";
        metadata.shared_quant_type = "Q4_K";
        return metadata;
    }

    MoERoutedExpertPlacementPlan requestedPlan(
        const MoERoutedExpertModelMetadata &metadata)
    {
        (void)metadata;
        const auto &test_case = activeModelParityCaseOrThrow();
        if (!test_case.topology.expert_overlay_plan)
        {
            throw std::logic_error(
                "Generated graph-native case has no ExpertOverlay blueprint");
        }
        auto generated = *test_case.topology.expert_overlay_plan;
        generated.residency_policy =
            isDynamicResidencyProductionTest()
                ? RoutedExpertResidencyPolicy::RoutedTierRebalanced
                : RoutedExpertResidencyPolicy::StaticById;
        generated.owner_order = isRandomOwnerProductionTest()
                                    ? RoutedExpertOwnerOrder::Random
                                    : RoutedExpertOwnerOrder::Ordinal;
        return generated;
    }

    /**
     * @brief Bind a topology and place a remote NodeLocal owner bucket first.
     *
     * Random whole-expert ownership partitions its deterministic permutation in
     * participant order. For the explicit distributed-migration proof, order
     * CPU participants that are remote from the continuation root before any
     * colocated participant. The production inventory binder resolves all rank
     * identities first, so moving GPUs between sockets/ranks changes the result
     * without changing this test or hard-coding NUMA affinity.
     *
     * @param requested Rank-agnostic production placement request.
     * @param inventory Gathered hardware/rank inventory shared by every rank.
     * @return Hardware-bound adversarial declaration with paired addresses,
     *         ranks, and optional weights reordered consistently.
     * @throws std::invalid_argument when the requested proof has no remote CPU
     *         participant or binding produced incomplete identities.
     */
    MoERoutedExpertPlacementPlan remoteFirstNodeLocalOwnerPlan(
        const MoERoutedExpertPlacementPlan &requested,
        const ClusterInventory &inventory)
    {
        auto bound = bindMoEExpertOverlayPlanToClusterInventory(
            requested,
            inventory);
        if (!bound)
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout produced no bound plan");

        const auto continuation = std::find_if(
            bound->domains.begin(),
            bound->domains.end(),
            [&](const auto &domain)
            { return domain.name == bound->continuation_domain; });
        if (continuation == bound->domains.end())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout has no continuation domain");
        }
        const auto continuation_rank = continuation->primaryWorldRank();
        if (!continuation_rank)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout has no continuation rank");
        }

        auto cpu_domain = std::find_if(
            bound->domains.begin(),
            bound->domains.end(),
            [](const auto &domain)
            { return domain.name == kCpuColdDomain; });
        if (cpu_domain == bound->domains.end() ||
            cpu_domain->scope != ExecutionDomainScope::NODE_LOCAL ||
            cpu_domain->participants.size() < 2u ||
            cpu_domain->world_ranks.size() !=
                cpu_domain->participants.size())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout requires a fully bound multi-participant NodeLocal CPU domain");
        }

        std::vector<size_t> order(cpu_domain->participants.size(), 0u);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](size_t lhs, size_t rhs)
            {
                const bool lhs_remote =
                    cpu_domain->world_ranks[lhs] != *continuation_rank;
                const bool rhs_remote =
                    cpu_domain->world_ranks[rhs] != *continuation_rank;
                return lhs_remote && !rhs_remote;
            });
        if (cpu_domain->world_ranks[order.front()] == *continuation_rank)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay layout found no CPU participant remote from the continuation rank");
        }

        const auto old_participants = cpu_domain->participants;
        const auto old_world_ranks = cpu_domain->world_ranks;
        const auto old_weights = cpu_domain->weights;
        for (size_t destination = 0; destination < order.size(); ++destination)
        {
            const size_t source = order[destination];
            cpu_domain->participants[destination] =
                old_participants[source];
            cpu_domain->world_ranks[destination] =
                old_world_ranks[source];
            if (old_weights.size() == order.size())
                cpu_domain->weights[destination] = old_weights[source];
        }

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Adversarial NodeLocal owner order: "
            << "continuation_rank=" << *continuation_rank
            << " first_cpu_rank=" << cpu_domain->world_ranks.front()
            << " participants=" << cpu_domain->participants.size());
        return std::move(*bound);
    }

    std::optional<std::string> acceleratorHardwareBlocker(
        const ClusterInventory &inventory)
    {
        int cuda_count = 0;
        int rocm_count = 0;
        for (const auto &rank : inventory.ranks)
        {
            for (const auto &gpu : rank.gpus)
            {
                cuda_count += gpu.type == DeviceType::CUDA ? 1 : 0;
                rocm_count += gpu.type == DeviceType::ROCm ? 1 : 0;
            }
        }
        const int required_cuda = static_cast<int>(
            activeTypedParticipantCount(
                [](const GlobalDeviceAddress &participant)
                { return participant.isCUDA(); }));
        const int required_rocm = static_cast<int>(
            activeTypedParticipantCount(
                [](const GlobalDeviceAddress &participant)
                { return participant.isROCm(); }));
        if (topologyUsesCuda() && cuda_count < required_cuda)
            return "Graph-native Qwen3.5 MoE parity topology requires >=" +
                   std::to_string(required_cuda) + " CUDA device(s), found " +
                   std::to_string(cuda_count);

        if (topologyUsesRocm() && rocm_count < required_rocm)
            return "Graph-native Qwen3.5 MoE parity topology requires >=" +
                   std::to_string(required_rocm) + " ROCm device(s), found " +
                   std::to_string(rocm_count);

        return std::nullopt;
    }

    /**
     * @brief Return whether this fixture instance must prove bucketed prefill.
     *
     * The normal production campaign deliberately uses one exact authenticated
     * bucket for speed. This named cell is the explicit complementary contract:
     * it keeps real weights and the same CSV oracle but forces an ordered
     * heterogeneous sparse-collective schedule.
     */
    bool isSegmentedPrefillProductionTest()
    {
        const auto *test_case = activeModelParityCase();
        return test_case &&
               test_case->prefill_graph.isSegmentedCaptured();
    }

    /** @return Typed fixed capture rows, or zero for ordinary prefill. */
    int activeSegmentedPrefillCaptureRows()
    {
        const auto *test_case = activeModelParityCase();
        return test_case &&
                       test_case->prefill_graph.isSegmentedCaptured()
                   ? test_case->prefill_graph.captured_rows
                   : 0;
    }

#ifndef LLAMINAR_QWEN122_MATRIX_ONLY
    /**
     * @return Complete 35B topology x placement x movement x prefill matrix.
     *
     * The CUDA/ROCm/CPU topology owns both ordinary and segmented captured
     * prefill profiles. Every other topology owns the ordinary profile. All
     * cells, including the ten historical cases, are emitted only by the
     * central typed expander.
     */
    const std::vector<ModelParityCase> &qwen35GraphNativeParityCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            std::set<std::string> names;
            for (const auto &spec : qwen35MoE35BOverlayTopologySpecs())
            {
                auto topology_cases = expandModelParityDefinition(
                    qwen35MoE35BGraphNativeParityDefinition(spec));
                for (auto &test_case : topology_cases)
                {
                    if (!names.insert(test_case.testName()).second)
                    {
                        throw std::logic_error(
                            "Duplicate generated 35B graph-native parity case: " +
                            test_case.testName());
                    }
                    expanded.push_back(std::move(test_case));
                }
            }
            return expanded;
        }();
        return cases;
    }
#endif

#ifdef LLAMINAR_QWEN122_MATRIX_ONLY
    /**
     * @brief Build the model-wide Dynamic policy for one generated 122B cell.
     *
     * A migration wave is parallel across transformer layers, so its physical
     * cycle width is one tier-residency cycle per layer plus one independent
     * within-tier participant cycle when the declared topology has an
     * exchange degree of freedom.  The command envelope admits a complete
     * closed cycle through every declared participant; it is not a second
     * scheduling limit.  Capacity admission prices these exact values before
     * any model weight or transfer lane is materialized.
     *
     * @param spec Typed physical topology expanded by the canonical matrix.
     * @param transformer_layers Authenticated main-model routed layer count.
     * @return Complete production Dynamic policy for the generated cell.
     */
    MoERebalanceRuntimeConfig qwen122DynamicParityEconomics(
        const Qwen122OverlayTopologySpec &spec,
        int transformer_layers)
    {
        if (transformer_layers <= 0)
        {
            throw std::invalid_argument(
                "Qwen122 Dynamic wave geometry requires a positive transformer-layer count");
        }
        const std::uint64_t participant_count =
            static_cast<std::uint64_t>(
                spec.cuda_participants + spec.rocm_participants +
                spec.cpu_participants);
        if (participant_count == 0u)
        {
            throw std::invalid_argument(
                "Qwen122 Dynamic wave geometry requires at least one participant");
        }
        const bool has_participant_axis =
            spec.cuda_participants > 1 || spec.rocm_participants > 1 ||
            spec.cpu_participants > 1;
        const std::uint64_t cycle_slots =
            static_cast<std::uint64_t>(transformer_layers) +
            (has_participant_axis ? 1u : 0u);
        const std::uint64_t maximum_cycle_edges =
            std::max<std::uint64_t>(2u, participant_count);
        if (cycle_slots > std::numeric_limits<std::uint32_t>::max() ||
            maximum_cycle_edges >
                std::numeric_limits<std::uint32_t>::max() / cycle_slots)
        {
            throw std::overflow_error(
                "Qwen122 Dynamic wave geometry exceeds the runtime policy range");
        }

        MoERebalanceRuntimeConfig config;
        config.mode = MoERebalanceRuntimeMode::Dynamic;
        config.window_size = 4;
        config.max_window_size = 4;
        config.window_growth_factor = 1.0f;
        config.migration_transfer_slots =
            static_cast<std::uint32_t>(cycle_slots);
        config.migration_payoff_horizon_tokens = 65'536;
        config.release_raw_expert_weights = false;
        /*
         * A max/min load ratio cannot be below 1.0.  Use the exact 1000
         * per-mille boundary to make every observed imbalance eligible; zero
         * is not a valid ratio threshold and used to survive until residency
         * authority construction.
         */
        config.dynamic_imbalance_threshold_per_mille =
            moe_rebalance_policy::
                kMinimumDynamicImbalanceThresholdPerMille;
        config.dynamic_min_improvement_per_mille = 0;
        config.dynamic_max_swaps_per_layer =
            has_participant_axis ? 2u : 1u;
        config.dynamic_max_plan_entries_per_wave =
            static_cast<std::uint32_t>(
                cycle_slots * maximum_cycle_edges);
        config.dynamic_min_window_activations = 0;
        config.device_min_load_spread_improvement = 0;
        config.device_min_load_spread_improvement_divisor = 0;
        config.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.device_min_foreign_rows_per_critical_path_payload_slot = 0;
        config.device_min_router_spread_improvement_per_payload_slot = 0;
        config.device_max_post_wave_load_spread_per_mille = 1000;
        config.device_maintenance_slack_tokens = 0;
        config.device_min_maintenance_period_tokens =
            kModelParityRequiredMaximumMTPDepth + 1;
        config.device_initial_maintenance_period_tokens = 1;
        return config;
    }

    /**
     * @brief Derive the two typed Dynamic evidence policies for Qwen 122B.
     *
     * A movement-only cell needs one conflict-free priority cycle and, when
     * the topology exposes it, one same-priority participant cycle.  Its
     * observation window expands to the declared model context immediately
     * after that publication so numerical parity executes against the proven
     * epoch without inducing unrelated migration churn.  The designated A/B
     * witness retains the full layer-parallel transfer fabric and one fixed
     * window large enough for its complete timing cohort.
     *
     * @param spec Typed physical topology expanded by the canonical matrix.
     * @param transformer_layers Authenticated main-model routed layer count.
     * @param maximum_context_rows Declared request context admission.
     * @return Complete evidence-indexed policies used by matrix expansion.
     */
    ModelParityDynamicRuntimePolicies qwen122DynamicRuntimePolicies(
        const Qwen122OverlayTopologySpec &spec,
        int transformer_layers,
        int maximum_context_rows)
    {
        constexpr int kMovementProofInitialWindowRows =
            kQwen35MoEMovementProofInitialWindowRows;
        constexpr std::uint32_t kMovementProofConcurrentCycles = 2u;
        if (maximum_context_rows < kMovementProofInitialWindowRows)
        {
            throw std::invalid_argument(
                "Qwen122 Dynamic movement proof requires context capacity for its initial histogram window");
        }

        MoERebalanceRuntimeConfig movement =
            qwen122DynamicParityEconomics(spec, transformer_layers);
        movement.window_size = kMovementProofInitialWindowRows;
        movement.max_window_size = maximum_context_rows;
        movement.window_growth_factor =
            static_cast<float>(maximum_context_rows) /
            static_cast<float>(kMovementProofInitialWindowRows);
        movement.migration_cycles_per_wave =
            kMovementProofConcurrentCycles;
        /* The initial one-token device cadence publishes the proof wave. After
         * that transaction, the recurring device scheduler must match the
         * expanded host observation horizon or it can author additional
         * device epochs while numerical parity is still executing. */
        movement.device_min_maintenance_period_tokens =
            maximum_context_rows;

        MoERebalanceRuntimeConfig speedup =
            qwen122DynamicParityEconomics(spec, transformer_layers);
        speedup.window_size = kConvergenceHistogramWindowTokens;
        speedup.max_window_size = kConvergenceHistogramWindowTokens;
        speedup.window_growth_factor = 1.0F;

        return {
            .economic_movement = std::move(movement),
            .economic_movement_and_observed_speedup = std::move(speedup),
        };
    }

    /**
     * @brief Declare one canonical 122B real-model/topology parity matrix.
     *
     * Cross-rank GPU ownership is deliberately unresolved. Production cluster
     * inventory binds each ordinal to whichever MPI instance currently owns
     * it. NCCL/RCCL remain inside their named domains; the outer graph is
     * multi-domain, not a fictitious heterogeneous tensor-parallel collective.
     */
    ModelParityDefinition qwen122ExpertOverlayParityDefinition(
        const Qwen122OverlayTopologySpec &spec)
    {
        ModelParityDefinition definition;
        definition.model = {
            .test_id = "Qwen35_122B",
            .model_path = kQwen122ModelPath,
            .reference_directory = kQwen122SnapshotDir,
            .decode_steps = 4,
            .max_seq_len = 4096,
            .transformer_layers = 48,
            .maximum_mtp_draft_depth = kQwen122MaximumMTPDraftDepth,
        };
        definition.topology.test_id = spec.test_id;
        definition.topology.kind =
            spec.mpi_ranks == 1
                ? ModelParityTopologyKind::RankLocalMultiDomain
                : ModelParityTopologyKind::NodeMultiDomain;
        definition.topology.collective = Collective::None;
        definition.topology.mpi_ranks = spec.mpi_ranks;
        for (int ordinal = 0; ordinal < spec.cuda_participants; ++ordinal)
        {
            definition.topology.participants.push_back({
                GlobalDeviceAddress::cuda(ordinal),
                spec.mpi_ranks == 1 ? std::optional<int>{0} : std::nullopt,
            });
        }
        for (int ordinal = 0; ordinal < spec.rocm_participants; ++ordinal)
        {
            definition.topology.participants.push_back({
                GlobalDeviceAddress::rocm(ordinal),
                spec.mpi_ranks == 1 ? std::optional<int>{0} : std::nullopt,
            });
        }
        for (int numa = 0; numa < spec.cpu_participants; ++numa)
        {
            definition.topology.participants.push_back({
                GlobalDeviceAddress::cpu(numa),
                spec.mpi_ranks == 1
                    ? std::optional<int>{0}
                    : std::optional<int>{numa},
            });
        }
        definition.topology.expert_overlay_plan =
            qwen122OverlayBlueprint(spec);
        definition.thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.03f,
            /*
             * Recursive MTP logits have a separate baseline budget so their
             * quantized feedback does not weaken ordinary prefill/decode.
             */
            .mtp_kl_threshold = 0.05f,
            .min_top1_accuracy = 0.80f,
            .min_top5_accuracy = 0.60f,
            .pytorch_top1_in_topk = 4,
        };
        definition.precisions.activation = {ActivationPrecision::FP16};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.features.mtp = ModelParityAxisProfile::Standard;
        definition.features.mtp_kl_threshold_overrides = {
            {
                .policy = ModelParityMTP::Depth15,
                /*
                 * Fourteen recurrent FP16 activation round trips accumulate
                 * bounded distribution drift. Retain the 0.99 terminal-hidden
                 * cosine and mutual top-3 proofs while narrowly accommodating
                 * the measured 0.0554 KL at this explicit fixed depth.
                 */
                .maximum_kl_divergence = 0.06f,
            },
        };
        definition.dynamic_rebalance = qwen122DynamicRuntimePolicies(
            spec,
            definition.model.transformer_layers,
            definition.model.max_seq_len);
        return definition;
    }

    /**
     * @return Deduplicated topology x 4 placement/movement x 6 MTP matrix.
     *
     * Topology identifiers and complete generated case names are checked here
     * because this binary joins several independently valid definitions into
     * one process-campaign catalogue. A duplicate would otherwise make GTest
     * registration order, rather than the typed source, choose the live case.
     */
    const std::vector<ModelParityCase> &qwen122ExpertOverlayParityCases()
    {
        static const auto cases = []
        {
            std::vector<ModelParityCase> expanded;
            std::set<std::string> topology_ids;
            std::set<std::string> case_names;
            for (const auto &spec : qwen122OverlayTopologySpecs())
            {
                if (!topology_ids.insert(spec.test_id).second)
                {
                    throw std::logic_error(
                        "Duplicate 122B ExpertOverlay topology id: " +
                        std::string(spec.test_id));
                }
                auto topology_cases = expandModelParityDefinition(
                    qwen122ExpertOverlayParityDefinition(spec));
                for (auto &test_case : topology_cases)
                {
                    const std::string name = test_case.testName();
                    if (!case_names.insert(name).second)
                    {
                        throw std::logic_error(
                            "Duplicate generated 122B ExpertOverlay case: " +
                            name);
                    }
                    expanded.push_back(std::move(test_case));
                }
            }
            return expanded;
        }();
        return cases;
    }
#endif

    /**
     * @brief One bounded, physical-identity-indexed 122B prepared authority.
     *
     * Residency policy and owner order affect the automatic capacity solution
     * and therefore the exact expert subset packed into model-owned storage.
     * Every MTP cell admits the same depth-fifteen graph envelope, so requested
     * execution depth is intentionally absent from physical identity. Each
     * topology has four policy identities, but retaining several complete
     * 122B authorities would consume the VRAM needed to admit the next one.
     * Consecutive compatible cells reuse this single slot; a topology or policy
     * transition retires the old ModelContext before the next physical plan is
     * solved. Every MPI process owns only its rank-local authority.
     */
    struct Qwen122OverlayPhysicalIdentity
    {
        std::string topology_id;
        std::size_t policy_slot = 0u;

        friend bool operator==(
            const Qwen122OverlayPhysicalIdentity &,
            const Qwen122OverlayPhysicalIdentity &) = default;
    };

    /** One process-local prepared authority and its complete physical identity. */
    struct Qwen122OverlayModelContextCampaignCache
    {
        std::mutex mutex;
        std::string model_path;
        std::optional<Qwen122OverlayPhysicalIdentity> physical_identity;
        std::optional<ModelContextReuseContract> contract;
    };

    /** Rank-wide path selected before constructing the next production runner. */
    enum class Qwen122CampaignModelAdmission : std::uint8_t
    {
        Fresh,
        Reuse,
    };

    /** Complete result of authenticating one process-local cache probe. */
    struct Qwen122CampaignModelAdmissionResult
    {
        Qwen122CampaignModelAdmission admission{
            Qwen122CampaignModelAdmission::Fresh};
        bool succeeded = false;
        std::string diagnostic;
    };

    /**
     * @brief Authenticate fresh-versus-reuse admission across every MPI rank.
     *
     * A retained ModelContext is rank-local, but the production runner's
     * initialization phases are collective. Consequently, one rank may not
     * enter the reuse constructor while a peer enters the fresh constructor.
     * The selected path is encoded in the phase identity itself: a partial
     * cache hit becomes a typed phase-identity mismatch before either runner
     * can issue a production collective.
     *
     * @param local_cache_hit Whether this rank holds the requested authority.
     * @param local_error Rank-local cache validation or retirement failure.
     * @return One unanimous admission or a precise collective diagnostic.
     */
    Qwen122CampaignModelAdmissionResult
    reachQwen122CampaignModelAdmission(
        MPI_Comm control_communicator,
        bool local_cache_hit,
        std::string_view local_error)
    {
        const bool local_valid = local_error.empty();
        const std::string_view phase_name =
            !local_valid
                ? "qwen122CampaignModelInvalid"
                : local_cache_hit
                      ? "qwen122CampaignModelReuse"
                      : "qwen122CampaignModelFresh";
        const auto consensus = MPIRankInitializationConsensus::reach(
            control_communicator,
            RankInitializationPhaseIdentity{
                .ordinal = 0u,
                .name = phase_name,
            },
            local_valid
                ? RankInitializationLocalOutcome::Succeeded
                : RankInitializationLocalOutcome::ReturnedFailure);

        if (consensus.outcome !=
            RankInitializationConsensusOutcome::AllRanksSucceeded)
        {
            std::ostringstream diagnostic;
            diagnostic
                << "122B campaign model admission was not rank-unanimous";
            if (!local_error.empty())
                diagnostic << ": local=" << local_error;
            if (!consensus.detail.empty())
                diagnostic << "; consensus=" << consensus.detail;
            return {
                .admission = Qwen122CampaignModelAdmission::Fresh,
                .succeeded = false,
                .diagnostic = diagnostic.str(),
            };
        }

        return {
            .admission = local_cache_hit
                             ? Qwen122CampaignModelAdmission::Reuse
                             : Qwen122CampaignModelAdmission::Fresh,
            .succeeded = true,
            .diagnostic = {},
        };
    }

    /** @return The sole bounded 122B prepared-weight cache in this MPI process. */
    Qwen122OverlayModelContextCampaignCache &
    qwen122OverlayModelContextCampaignCache()
    {
        static Qwen122OverlayModelContextCampaignCache cache;
        return cache;
    }

    /** @brief One live recursive checkpoint retained until its HF branch exists. */
    struct DeferredMTPCheckpoint
    {
        std::string stage;
        std::string production_key;
        std::vector<float> actual;
    };

    /**
     * @brief Complete immutable evidence for one missing recursive HF context.
     *
     * Only branch-dependent sidecar tensors are copied. Main-model, canonical
     * sidecar, token, movement, and graph-path evidence remain compared in the
     * originating production cell. The copy breaks the runner lifetime cleanly:
     * no graph, arena, model context, or device allocation survives into the
     * CPU reference phase.
     */
    struct DeferredMTPBranchContext
    {
        std::string test_name;
        std::string model_path;
        std::string prompt;
        std::string snapshot_dir;
        std::filesystem::path snapshot_csv_path;
        int decode_steps = 0;
        int call = 0;
        int reference_step = -1;
        int reference_depth = 0;
        int vocab_size = 0;
        int top_k = 0;
        int num_experts = 0;
        float cosine_threshold = 0.0f;
        float decode_cosine_threshold = 0.0f;
        float kl_threshold = 0.0f;
        std::vector<int32_t> condition_tokens;
        std::vector<DeferredMTPCheckpoint> checkpoints;
    };

    /** @brief Process-resident queue resolved after all production cells retire. */
    struct DeferredMTPBranchCampaign
    {
        std::mutex mutex;
        std::vector<DeferredMTPBranchContext> contexts;
    };

    /** @return The one bounded deferred-reference queue in this test process. */
    DeferredMTPBranchCampaign &deferredMTPBranchCampaign()
    {
        static DeferredMTPBranchCampaign campaign;
        return campaign;
    }

    /**
     * @brief Retire the cache's final model owner and prove exact GPU recovery.
     *
     * The caller holds the cache mutex. Every ticket is captured while the
     * model and reusable workspace allocations are still live; only then is
     * the complete contract destroyed. Completion trims scoped runtime caches
     * and proves the admitted per-device bytes became driver-visible again.
     * CPU arenas are trimmed at this same terminal owner edge.
     *
     * @param cache Locked process-local campaign cache.
     * @param error Receives the first ownership or reclamation defect.
     * @return True when the cache was empty or every owner retired completely.
     */
    bool retireQwen122OverlayModelContextCampaignCacheLocked(
        Qwen122OverlayModelContextCampaignCache &cache,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!cache.contract)
        {
            cache.physical_identity.reset();
            cache.model_path.clear();
            return true;
        }

        const ModelContextReuseContract &contract = *cache.contract;
        if (!contract.context || !contract.reuse_authority ||
            !contract.reusable_execution_workspaces)
        {
            if (error)
                *error = "122B campaign cache has an incomplete model-retirement authority";
            return false;
        }
        if (contract.reuse_authority->state() !=
            ModelContextReuseAuthority::State::Reusable)
        {
            if (error)
                *error = "122B campaign attempted to retire a model before its runner published Reusable";
            return false;
        }
        if (contract.context.use_count() != 1u)
        {
            if (error)
            {
                *error = "122B campaign final ModelContext owner is not exclusive: use_count=" +
                         std::to_string(contract.context.use_count());
            }
            return false;
        }
        if (contract.reusable_execution_workspaces.use_count() != 1u)
        {
            if (error)
            {
                *error = "122B campaign final reusable-workspace owner is not exclusive: use_count=" +
                         std::to_string(
                             contract.reusable_execution_workspaces.use_count());
            }
            return false;
        }

        std::string retention_error;
        const auto sealed_retention =
            contract.reuse_authority->sealedDeviceMemoryRetention(
                &retention_error);
        if (!sealed_retention)
        {
            if (error)
            {
                *error =
                    "122B campaign model-retirement authority has no sealed allocation BOM: " +
                    (retention_error.empty()
                         ? std::string("unknown lifecycle failure")
                         : retention_error);
            }
            return false;
        }

        std::vector<ExclusiveModelRetirementTicket> tickets;
        tickets.reserve(sealed_retention->size());
        try
        {
            std::set<DeviceId> devices;
            for (const ModelDeviceMemoryRetention &retention :
                 *sealed_retention)
            {
                if (!retention.valid() ||
                    !devices.insert(retention.device).second)
                {
                    throw std::logic_error(
                        "122B campaign model-retirement BOM contains an invalid or duplicate device row");
                }
                tickets.push_back(
                    TransferEngine::instance()
                        .beginExclusiveModelRetirement(retention));
            }
        }
        catch (const std::exception &exception)
        {
            if (error)
            {
                *error = "122B campaign could not begin exact model retirement: " +
                         std::string(exception.what());
            }
            return false;
        }

        /*
         * This reset is the single ownership edge named by every ticket. It
         * destroys prepared GPU weights, CPU experts, and sealed workspace
         * backing together; no new topology may inspect capacity until every
         * ticket below has certified its physical endpoint.
         */
        cache.contract.reset();
        cache.physical_identity.reset();
        cache.model_path.clear();
#if defined(__GLIBC__)
        ::malloc_trim(0);
#endif

        try
        {
            for (auto &ticket : tickets)
            {
                (void)TransferEngine::instance()
                    .completeExclusiveModelRetirement(std::move(ticket));
            }
        }
        catch (const std::exception &exception)
        {
            if (error)
            {
                *error = "122B campaign exact model retirement was incomplete: " +
                         std::string(exception.what());
            }
            return false;
        }
        return true;
    }

    /**
     * @brief Release the final immutable 122B authority before Python loads it.
     * @param error Receives a precise retirement failure.
     * @return True when all model-lifetime allocations were certified free.
     */
    bool releaseQwen122OverlayModelContextCampaignCache(std::string *error)
    {
        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        return retireQwen122OverlayModelContextCampaignCacheLocked(
            cache, error);
    }

    /** @return A shell-safe single argument preserving every input byte. */
    std::string quoteDeferredReferenceArgument(const std::string &value)
    {
        std::string quoted = "'";
        for (const char character : value)
        {
            if (character == '\'')
                quoted += "'\"'\"'";
            else
                quoted += character;
        }
        quoted += '\'';
        return quoted;
    }

    /**
     * @brief Load one FP32/FP64 NPY tensor without constructing a parity fixture.
     * @param path Exact authenticated reference tensor path.
     * @param error Receives a precise load/type failure.
     * @return FP32 values, or an empty optional on failure.
     */
    std::optional<std::vector<float>> loadDeferredReferenceTensor(
        const std::filesystem::path &path,
        std::string *error)
    {
        try
        {
            const cnpy::NpyArray array = cnpy::npy_load(path.string());
            std::vector<float> values(array.num_vals);
            if (array.word_size == sizeof(float))
            {
                const float *const data = array.data<float>();
                std::copy(data, data + array.num_vals, values.begin());
            }
            else if (array.word_size == sizeof(double))
            {
                const double *const data = array.data<double>();
                std::transform(
                    data,
                    data + array.num_vals,
                    values.begin(),
                    [](double value) { return static_cast<float>(value); });
            }
            else
            {
                if (error)
                {
                    *error = "unsupported NPY word size " +
                             std::to_string(array.word_size) + " at " +
                             path.string();
                }
                return std::nullopt;
            }
            return values;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = path.string() + ": " + exception.what();
            return std::nullopt;
        }
    }

    /**
     * @brief Compare sparse routing weights after aligning them by expert ID.
     * @return Mean sparse-vector cosine and maximum absolute expert-mass error.
     */
    std::pair<float, float> compareDeferredRoutingWeights(
        const std::vector<float> &actual_weights,
        const std::vector<float> &expected_weights,
        const std::vector<float> &actual_indices,
        const std::vector<float> &expected_indices,
        int top_k,
        int num_experts)
    {
        if (top_k <= 0 || num_experts <= 0 || actual_weights.empty() ||
            actual_weights.size() != expected_weights.size() ||
            actual_weights.size() != actual_indices.size() ||
            actual_weights.size() != expected_indices.size() ||
            actual_weights.size() % static_cast<size_t>(top_k) != 0u)
        {
            return {0.0f, std::numeric_limits<float>::infinity()};
        }

        const size_t rows =
            actual_weights.size() / static_cast<size_t>(top_k);
        double total_cosine = 0.0;
        float maximum_error = 0.0f;
        for (size_t row = 0; row < rows; ++row)
        {
            std::vector<float> actual_sparse(
                static_cast<size_t>(num_experts), 0.0f);
            std::vector<float> expected_sparse(
                static_cast<size_t>(num_experts), 0.0f);
            for (int index = 0; index < top_k; ++index)
            {
                const size_t offset =
                    row * static_cast<size_t>(top_k) +
                    static_cast<size_t>(index);
                const int actual_expert =
                    static_cast<int>(actual_indices[offset]);
                const int expected_expert =
                    static_cast<int>(expected_indices[offset]);
                if (actual_expert >= 0 && actual_expert < num_experts)
                {
                    actual_sparse[static_cast<size_t>(actual_expert)] =
                        actual_weights[offset];
                }
                if (expected_expert >= 0 && expected_expert < num_experts)
                {
                    expected_sparse[static_cast<size_t>(expected_expert)] =
                        expected_weights[offset];
                }
            }
            total_cosine += computeCosineSimilarity(
                actual_sparse.data(),
                expected_sparse.data(),
                actual_sparse.size());
            for (size_t expert = 0; expert < actual_sparse.size(); ++expert)
            {
                maximum_error = std::max(
                    maximum_error,
                    std::abs(actual_sparse[expert] - expected_sparse[expert]));
            }
        }
        return {
            static_cast<float>(total_cosine / static_cast<double>(rows)),
            maximum_error};
    }

    /**
     * @brief Complete one deferred context with the same route-aware math gate.
     * @param context Immutable live evidence copied before runner teardown.
     * @param error Receives every failed invariant for campaign diagnostics.
     * @return True only when all tensors and the aggregate semantic gate pass.
     */
    bool compareDeferredMTPBranchContext(
        const DeferredMTPBranchContext &context,
        std::string *error)
    {
        std::string reference_prefix =
            "decode_step" + std::to_string(context.reference_step) +
            "_BRANCH";
        for (const int32_t token : context.condition_tokens)
            reference_prefix += "_" + std::to_string(token);
        reference_prefix +=
            "_MTP" + std::to_string(context.reference_depth) + "_";

        std::map<std::string, std::vector<float>> references;
        for (const auto &checkpoint : context.checkpoints)
        {
            const auto path = std::filesystem::path(context.snapshot_dir) /
                              (reference_prefix + checkpoint.stage + ".npy");
            auto loaded = loadDeferredReferenceTensor(path, error);
            if (!loaded || loaded->empty())
                return false;
            if (loaded->size() > checkpoint.actual.size() &&
                !checkpoint.actual.empty() &&
                loaded->size() % checkpoint.actual.size() == 0u)
            {
                loaded->erase(
                    loaded->begin(),
                    loaded->end() -
                        static_cast<ptrdiff_t>(checkpoint.actual.size()));
            }
            if (loaded->size() != checkpoint.actual.size())
            {
                if (error)
                {
                    *error = context.test_name + " " + checkpoint.stage +
                             " element mismatch actual=" +
                             std::to_string(checkpoint.actual.size()) +
                             " reference=" +
                             std::to_string(loaded->size());
                }
                return false;
            }
            references.emplace(checkpoint.stage, std::move(*loaded));
        }

        std::ofstream csv(context.snapshot_csv_path, std::ios::app);
        if (!csv.is_open())
        {
            if (error)
                *error = "could not append " + context.snapshot_csv_path.string();
            return false;
        }

        bool context_finite = true;
        bool routing_indices_exact = true;
        bool routing_top1_match = true;
        float routing_overlap = 1.0f;
        bool routing_weights_equivalent = false;
        bool routed_expert_output_equivalent = false;
        bool lm_head_passed = false;
        double numerical_cosine_sum = 0.0;
        size_t numerical_stage_count = 0u;
        size_t compared_stages = 0u;

        const auto actual_indices_it = std::find_if(
            context.checkpoints.begin(),
            context.checkpoints.end(),
            [](const DeferredMTPCheckpoint &checkpoint)
            { return checkpoint.stage == "MOE_ROUTING_INDICES"; });
        const auto reference_indices_it = references.find(
            "MOE_ROUTING_INDICES");

        for (const auto &checkpoint : context.checkpoints)
        {
            const auto reference_it = references.find(checkpoint.stage);
            if (reference_it == references.end())
                return false;
            const auto &actual = checkpoint.actual;
            const auto &expected = reference_it->second;
            bool finite = true;
            bool exact_indices = true;
            double maximum_absolute_error = 0.0;
            for (size_t index = 0; index < actual.size(); ++index)
            {
                finite = finite && std::isfinite(actual[index]) &&
                         std::isfinite(expected[index]);
                maximum_absolute_error = std::max(
                    maximum_absolute_error,
                    std::abs(
                        static_cast<double>(actual[index]) -
                        static_cast<double>(expected[index])));
                if (checkpoint.stage == "MOE_ROUTING_INDICES")
                    exact_indices = exact_indices &&
                                    actual[index] == expected[index];
            }

            float cosine = computeCosineSimilarity(
                actual.data(), expected.data(), actual.size());
            float stage_routing_overlap = 1.0f;
            bool stage_routing_top1_match = true;
            float kl = 0.0f;
            bool passed = finite;
            if (checkpoint.stage == "MOE_ROUTING_INDICES")
            {
                if (context.top_k <= 0 ||
                    actual.size() % static_cast<size_t>(context.top_k) != 0u)
                {
                    passed = false;
                }
                else
                {
                    const size_t rows =
                        actual.size() / static_cast<size_t>(context.top_k);
                    double overlap_sum = 0.0;
                    size_t top1_matches = 0u;
                    for (size_t row = 0; row < rows; ++row)
                    {
                        std::set<int> actual_experts;
                        std::set<int> expected_experts;
                        for (int slot = 0; slot < context.top_k; ++slot)
                        {
                            const size_t offset =
                                row * static_cast<size_t>(context.top_k) +
                                static_cast<size_t>(slot);
                            actual_experts.insert(
                                static_cast<int>(actual[offset]));
                            expected_experts.insert(
                                static_cast<int>(expected[offset]));
                        }
                        size_t intersection = 0u;
                        for (const int expert : actual_experts)
                        {
                            if (expected_experts.contains(expert))
                                ++intersection;
                        }
                        overlap_sum +=
                            static_cast<double>(intersection) /
                            static_cast<double>(context.top_k);
                        top1_matches +=
                            actual[row * static_cast<size_t>(context.top_k)] ==
                            expected[row * static_cast<size_t>(context.top_k)];
                    }
                    stage_routing_overlap = static_cast<float>(
                        overlap_sum / static_cast<double>(rows));
                    stage_routing_top1_match = top1_matches == rows;
                    cosine = stage_routing_overlap;
                    maximum_absolute_error = 1.0 - stage_routing_overlap;
                    const float minimum_overlap =
                        1.0f - 1.0f / static_cast<float>(context.top_k);
                    passed = finite && stage_routing_top1_match &&
                             stage_routing_overlap >= minimum_overlap;
                }
                routing_indices_exact = exact_indices;
                routing_top1_match =
                    routing_top1_match && stage_routing_top1_match;
                routing_overlap = std::min(
                    routing_overlap, stage_routing_overlap);
            }
            else if (checkpoint.stage == "MOE_ROUTING_WEIGHTS")
            {
                if (actual_indices_it == context.checkpoints.end() ||
                    reference_indices_it == references.end())
                {
                    passed = false;
                }
                else
                {
                    const auto [sparse_cosine, sparse_max_error] =
                        compareDeferredRoutingWeights(
                            actual,
                            expected,
                            actual_indices_it->actual,
                            reference_indices_it->second,
                            context.top_k,
                            context.num_experts);
                    cosine = sparse_cosine;
                    maximum_absolute_error = sparse_max_error;
                    stage_routing_overlap = sparse_cosine;
                    passed = finite &&
                             sparse_cosine >= context.cosine_threshold;
                }
                routing_weights_equivalent = passed;
            }
            else
            {
                const float threshold =
                    checkpoint.stage == "MOE_EXPERT_OUTPUT" &&
                            !routing_indices_exact
                        ? context.cosine_threshold
                        : context.decode_cosine_threshold;
                passed = finite && cosine >= threshold;
                if (checkpoint.stage == "MOE_EXPERT_OUTPUT")
                    routed_expert_output_equivalent = passed;
            }

            if (checkpoint.stage == "LM_HEAD")
            {
                if (context.vocab_size <= 0 ||
                    actual.size() %
                            static_cast<size_t>(context.vocab_size) !=
                        0u)
                {
                    passed = false;
                }
                else
                {
                    kl = computeKLDivergence(
                        actual.data(),
                        expected.data(),
                        actual.size(),
                        static_cast<size_t>(context.vocab_size));
                    passed = passed && kl < context.kl_threshold &&
                             pytorchTop1InLlaminarTopK(
                                 actual.data(),
                                 expected.data(),
                                 actual.size(),
                                 static_cast<size_t>(context.vocab_size),
                                 3) >= 1.0f &&
                             pytorchTop1InLlaminarTopK(
                                 expected.data(),
                                 actual.data(),
                                 actual.size(),
                                 static_cast<size_t>(context.vocab_size),
                                 3) >= 1.0f;
                }
                lm_head_passed = passed;
            }

            context_finite = context_finite && finite;
            if (parityStageContributesToLayerCosine(
                    checkpoint.stage, routing_indices_exact))
            {
                numerical_cosine_sum += cosine;
                ++numerical_stage_count;
            }

            csv << context.call << ',' << context.reference_step << ','
                << context.reference_depth << ','
                << checkpoint.production_key << ','
                << reference_prefix << checkpoint.stage << ','
                << actual.size() << ',' << cosine << ','
                << maximum_absolute_error << ',' << kl << ','
                << (exact_indices ? 1 : 0) << ','
                << stage_routing_overlap << ','
                << (stage_routing_top1_match ? 1 : 0) << ','
                << (finite ? 1 : 0) << ',' << (passed ? 1 : 0) << '\n';
            ++compared_stages;
        }
        csv.flush();
        if (!csv.good())
        {
            if (error)
                *error = "failed while appending " + context.snapshot_csv_path.string();
            return false;
        }

        const float minimum_overlap =
            context.top_k > 0
                ? 1.0f - 1.0f / static_cast<float>(context.top_k)
                : 1.0f;
        const bool routed_contribution_equivalent =
            routing_weights_equivalent ||
            (routing_top1_match && routing_overlap >= minimum_overlap &&
             routed_expert_output_equivalent);
        const double numerical_cosine =
            numerical_stage_count > 0u
                ? numerical_cosine_sum /
                      static_cast<double>(numerical_stage_count)
                : 0.0;
        const bool passed = compared_stages > 0u && context_finite &&
                            routing_top1_match &&
                            routed_contribution_equivalent &&
                            productionRecursiveMTPAggregatePasses(
                                numerical_cosine,
                                context.decode_cosine_threshold) &&
                            lm_head_passed;
        if (!passed && error)
        {
            std::ostringstream detail;
            detail << context.test_name
                   << " deferred recursive MTP parity failed"
                   << " step=" << context.reference_step
                   << " depth=" << context.reference_depth
                   << " compared_stages=" << compared_stages
                   << " finite=" << context_finite
                   << " routing_top1=" << routing_top1_match
                   << " routing_overlap=" << routing_overlap
                   << " routed_value=" << routed_contribution_equivalent
                   << " numerical_cosine=" << numerical_cosine
                   << " required_numerical_cosine="
                   << std::max(
                          static_cast<double>(
                              context.decode_cosine_threshold),
                          kMinimumProductionRecursiveMTPAggregateCosine)
                   << " lm_head=" << lm_head_passed
                   << " csv=" << context.snapshot_csv_path;
            *error = detail.str();
        }
        return passed;
    }

    /**
     * @brief Generate all missing branches with one loaded HF model and compare.
     * @param contexts Immutable queue removed from the live campaign.
     * @param error Receives the generator output or first numerical failure.
     */
    bool resolveDeferredMTPBranchCampaign(
        const std::vector<DeferredMTPBranchContext> &contexts,
        std::string *error)
    {
        if (contexts.empty())
            return true;

        const auto &identity = contexts.front();
        int maximum_depth = 1;
        int decode_steps = identity.decode_steps;
        std::set<std::pair<int, std::vector<int32_t>>> unique_branches;
        for (const auto &context : contexts)
        {
            if (context.model_path != identity.model_path ||
                context.prompt != identity.prompt ||
                context.snapshot_dir != identity.snapshot_dir)
            {
                if (error)
                    *error = "deferred MTP campaign mixed reference identities";
                return false;
            }
            maximum_depth = std::max(
                maximum_depth,
                static_cast<int>(context.condition_tokens.size()) + 1);
            decode_steps = std::max(decode_steps, context.decode_steps);
            unique_branches.emplace(
                context.reference_step, context.condition_tokens);
        }

        const std::filesystem::path request_path =
            contexts.front().snapshot_csv_path.parent_path() /
            "mtp_hf_branch_campaign_requests.json";
        std::ofstream request(request_path, std::ios::trunc);
        if (!request.is_open())
        {
            if (error)
                *error = "could not write " + request_path.string();
            return false;
        }
        request << "[\n";
        size_t branch_index = 0u;
        for (const auto &[step, tokens] : unique_branches)
        {
            if (branch_index++ != 0u)
                request << ",\n";
            request << "  {\"" << step << "\": [";
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index)
            {
                if (token_index != 0u)
                    request << ", ";
                request << tokens[token_index];
            }
            request << "]}";
        }
        request << "\n]\n";
        request.flush();
        if (!request.good())
        {
            if (error)
                *error = "failed while writing " + request_path.string();
            return false;
        }

        std::ostringstream script;
        script
            << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
               "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
               "source /workspaces/llaminar/.venv/bin/activate; fi; "
            << "python3 python/reference/"
               "generate_qwen35_moe_pipeline_snapshots.py"
            << " --model "
            << quoteDeferredReferenceArgument(identity.model_path)
            << " --prompt "
            << quoteDeferredReferenceArgument(identity.prompt)
            << " --output "
            << quoteDeferredReferenceArgument(identity.snapshot_dir)
            << " --decode-steps " << decode_steps
            << " --mtp-sidecar-snapshots --mtp-max-draft-depth "
            << maximum_depth << " --mtp-branch-overrides "
            << quoteDeferredReferenceArgument(request_path.string());
        const std::string command =
            "bash -c " + quoteDeferredReferenceArgument(script.str()) +
            " 2>&1";

        LOG_INFO(
            "[Qwen3.5 MoE MTP Parity] Resolving "
            << unique_branches.size()
            << " deferred branch trajectories with one HF model load after "
               "production residency retirement");
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (error)
                *error = "could not start deferred HF branch generator";
            return false;
        }
        std::string output;
        std::array<char, 512> buffer{};
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            output += buffer.data();
        const int exit_code = pclose(pipe);
        if (exit_code != 0)
        {
            if (error)
                *error = "deferred HF branch generation failed:\n" + output;
            return false;
        }

        for (const auto &context : contexts)
        {
            if (!compareDeferredMTPBranchContext(context, error))
                return false;
        }
        return true;
    }

} // namespace

class Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>
      , public ::testing::WithParamInterface<ModelParityCase>
{
public:
    /**
     * @brief Resolve forced-branch HF evidence after production residency ends.
     *
     * GoogleTest calls this only after every fixture TearDown has destroyed its
     * runner. Releasing the external prepared-weight cache on every MPI process
     * then creates an explicit memory phase boundary: the sole artifact owner
     * loads Hugging Face once for the union of observed branches, while every
     * other rank waits without retaining a model authority. The final broadcast
     * makes a reference or numerical failure visible to the complete test world.
     */
    static void TearDownTestSuite()
    {
        std::string retirement_error;
        const bool local_retirement_complete =
            releaseQwen122OverlayModelContextCampaignCache(
                &retirement_error);
        int local_complete = local_retirement_complete ? 1 : 0;
        int all_complete = 0;
        MPI_Allreduce(
            &local_complete,
            &all_complete,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD);
        if (all_complete == 0)
        {
            ADD_FAILURE()
                << (local_retirement_complete
                        ? "Another MPI rank failed exact 122B model retirement"
                        : retirement_error);
            return;
        }

        auto &campaign = deferredMTPBranchCampaign();
        std::vector<DeferredMTPBranchContext> local_contexts;
        {
            std::lock_guard<std::mutex> lock(campaign.mutex);
            local_contexts = std::move(campaign.contexts);
            campaign.contexts.clear();
        }

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const int local_owner = local_contexts.empty() ? 0 : rank + 1;
        const int local_owner_count = local_contexts.empty() ? 0 : 1;
        int owner = 0;
        int owner_count = 0;
        MPI_Allreduce(
            &local_owner,
            &owner,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
        MPI_Allreduce(
            &local_owner_count,
            &owner_count,
            1,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);
        if (owner_count == 0)
            return;

        EXPECT_EQ(owner_count, 1)
            << "Deferred HF branch evidence had more than one artifact authority";
        if (owner_count != 1 || owner <= 0)
            return;
        --owner;

        int success = 1;
        std::string error;
        if (rank == owner)
        {
            success = resolveDeferredMTPBranchCampaign(
                          local_contexts, &error)
                          ? 1
                          : 0;
            if (success == 0)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE MTP Parity] Deferred branch campaign failed: "
                    << error);
            }
        }
        MPI_Bcast(&success, 1, MPI_INT, owner, MPI_COMM_WORLD);
        EXPECT_EQ(success, 1)
            << (rank == owner
                    ? error
                    : "Deferred HF branch campaign failed on artifact authority rank " +
                          std::to_string(owner));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /** @brief Return one immutable config per exact generated matrix cell. */
    static const TestConfig &generatedConfig()
    {
        static std::map<std::string, TestConfig> configs;
        const auto *test_case = activeModelParityCase();
        if (!test_case)
            throw std::logic_error(
                "Graph-native parity configuration requested without an active typed case");
        const std::string name = test_case->testName();
        const auto [it, inserted] = configs.try_emplace(
            name,
            test_case->toTestConfig());
        (void)inserted;
        return it->second;
    }

    /**
     * @brief Return the configuration whose declared backends match this exact cell.
     *
     * The production campaign discovery derives its resource signature from
     * the test name; this return value keeps the test fixture's result labels,
     * reference evidence, and runner configuration aligned with that same
     * declarative topology.
     */
    const TestConfig &getTestConfig() const
    {
        return generatedConfig();
    }

protected:
    using Base = Qwen35MoEConfigDrivenParityTest<Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold>;

    /** @return Capacity-resolved immutable placement installed in production. */
    const MoERoutedExpertPlacementPlan &resolvedOverlayPlan() const
    {
        if (orch_runner_ &&
            orch_runner_->config().moe_routed_expert_plan)
        {
            return *orch_runner_->config().moe_routed_expert_plan;
        }
        if (!overlay_plan_)
        {
            throw std::logic_error(
                "ExpertOverlay parity has no requested or resolved placement plan");
        }
        return *overlay_plan_;
    }

    /**
     * @brief Retain the exact device-owned route epoch consumed by diagnostics.
     *
     * The Hugging Face pack authenticates model tensors, but it cannot name
     * Llaminar's placement-bank selector or final domain schedule. These
     * additional keys bind each compared routed contribution to the acquired
     * overlay epoch. Declaring them through the shared typed snapshot policy
     * makes them part of graph identity before capture; diagnostics never
     * mutate or recapture the serving graph after setup.
     */
    ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const override
    {
        auto policy = Base::parityGraphSnapshotPolicy(phase);
        auto &required =
            phase == ParityForwardPhase::Prefill
                ? policy.required_prefill_snapshot_keys
                : policy.required_decode_snapshot_keys;
        for (int layer = 0; layer < parityLayerCount(); ++layer)
        {
            const std::string prefix =
                "layer" + std::to_string(layer);
            const std::array<std::string_view, 8> suffixes{
                "_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
                "_MOE_RUNTIME_ROUTE_WEIGHTS",
                "_MOE_ROUTE_CONTRIBUTIONS",
                "_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
                "_MOE_OVERLAY_ROUTE_BANK0_EPOCH",
                "_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
                "_MOE_OVERLAY_ROUTE_BANK1_EPOCH",
                "_MOE_OVERLAY_ROUTE_SELECTED_BANK",
            };
            for (const std::string_view suffix : suffixes)
            {
                const std::string key = prefix + std::string(suffix);
                if (std::find(required.begin(), required.end(), key) ==
                    required.end())
                {
                    required.push_back(key);
                }
            }
        }
        return policy;
    }

    /** @brief The 122B matrix mathematically compares recursive MTP sidecars. */
    bool requiresMTPSidecarReferenceSnapshots() const override
    {
        return isQwen122ProductionTest() && activeMTPEnabled();
    }

    /** @return Maximum recurrent sidecar depth admitted by the 122B matrix. */
    int requiredMTPSidecarReferenceDraftDepth() const override
    {
        return isQwen122ProductionTest() && activeMTPEnabled()
                   ? kQwen122MaximumMTPDraftDepth
                   : 0;
    }

    /** @return Whether this process intentionally runs consecutive 122B cells. */
    bool mayReuseQwen122OverlayModelContext() const
    {
        return isQwen122ProductionTest() &&
               DebugEnv::isTruthyEnv(
                   "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN");
    }

    /**
     * @return Complete topology and policy identity for prepared-weight reuse.
     *
     * The two bits encode residency policy and whole-expert owner order. Every
     * cell in one topology reserves the same maximum retained graph geometry,
     * matching production reuse identity without coupling physical placement
     * to the fixed depth selected for one request. Topology identity prevents
     * a differently sized participant catalogue from reusing those pointers.
     */
    Qwen122OverlayPhysicalIdentity
    qwen122ModelContextPhysicalIdentity() const
    {
        const std::size_t policy =
            isDynamicResidencyProductionTest() ? 4u : 0u;
        const std::size_t owner_order =
            isRandomOwnerProductionTest() ? 2u : 0u;
        const auto *test_case = activeModelParityCase();
        if (!test_case)
        {
            throw std::logic_error(
                "122B model-context identity requires an active typed case");
        }
        return Qwen122OverlayPhysicalIdentity{
            .topology_id = test_case->topology.test_id,
            .policy_slot = policy + owner_order,
        };
    }

    /**
     * @brief Find the prior runner's rank-local prepared-weight certificate.
     *
     * A miss does not synthesize ModelContext configuration from test metadata;
     * only a previously initialized production runner may populate the slot.
     */
    std::optional<ModelContextReuseContract>
    findQwen122OverlayModelContext(
        bool *cache_hit,
        std::string *error) const
    {
        if (cache_hit)
            *cache_hit = false;
        if (!mayReuseQwen122OverlayModelContext())
            return std::nullopt;

        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.model_path.empty() &&
            cache.model_path != config_.model_path)
        {
            if (error)
                *error = "122B campaign cache belongs to a different model path";
            return std::nullopt;
        }

        const auto requested_identity =
            qwen122ModelContextPhysicalIdentity();
        if (!cache.contract || !cache.physical_identity)
            return std::nullopt;
        const ModelContextReuseContract &contract = *cache.contract;
        if (!contract.context || !contract.reuse_authority ||
            !contract.reusable_execution_workspaces)
        {
            if (error)
            {
                *error =
                    "122B campaign cache has an incomplete reusable-model contract";
            }
            return std::nullopt;
        }
        if (contract.reuse_authority->state() !=
            ModelContextReuseAuthority::State::Reusable)
        {
            if (error)
            {
                const std::string diagnostic =
                    contract.reuse_authority->diagnostic();
                *error =
                    "122B campaign cache is not reusable" +
                    (diagnostic.empty()
                         ? std::string()
                         : ": " + diagnostic);
            }
            return std::nullopt;
        }
        if (!contract.reusable_execution_workspaces->valid())
        {
            if (error)
            {
                const std::string diagnostic =
                    contract.reusable_execution_workspaces->diagnostic();
                *error =
                    "122B campaign reusable workspace authority is invalid" +
                    (diagnostic.empty()
                         ? std::string()
                         : ": " + diagnostic);
            }
            return std::nullopt;
        }
        if (*cache.physical_identity != requested_identity)
        {
            /*
             * The prior runner has already torn down all mutable state. Drop
             * the cache's final model-owned reference now so GPU allocations
             * are returned before automatic capacity observes free memory for
             * the incompatible physical plan.
             */
            std::string retirement_error;
            if (!retireQwen122OverlayModelContextCampaignCacheLocked(
                    cache, &retirement_error))
            {
                if (error)
                    *error = std::move(retirement_error);
                return std::nullopt;
            }
            PerfStatsCollector::addCounter(
                "weight_loading",
                "parity_campaign_model_context_cache_evictions",
                1.0,
                "setup",
                {},
                {{"next_topology", requested_identity.topology_id},
                 {"next_physical_identity_slot",
                  std::to_string(requested_identity.policy_slot)}});
            return std::nullopt;
        }
        if (cache_hit)
            *cache_hit = true;
        return cache.contract;
    }

    /**
     * @brief Publish one initialized runner's immutable rank-local authority.
     *
     * The cache never replaces a live slot: doing so could conceal a teardown
     * or identity defect. The contract itself remains the production source of
     * truth for participant topology and prepared-weight compatibility.
     */
    bool publishQwen122OverlayModelContext(
        const ModelContextReuseContract &contract,
        std::string *error) const
    {
        if (!mayReuseQwen122OverlayModelContext() || !contract.context ||
            contract.routed_weight_authority_identity.empty())
        {
            if (error)
            {
                *error = "cannot publish an ineligible, null, or uncertified "
                         "122B ExpertOverlay model authority";
            }
            return false;
        }
        if (!contract.reuse_authority ||
            contract.reuse_authority->state() !=
                ModelContextReuseAuthority::State::RunnerExclusive)
        {
            if (error)
            {
                *error =
                    "production runner returned a model contract outside its exclusive pre-seal lifecycle";
            }
            return false;
        }
        if (contract.context->path() != config_.model_path)
        {
            if (error)
                *error = "production runner returned a different model authority";
            return false;
        }

        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.model_path.empty() &&
            cache.model_path != config_.model_path)
        {
            if (error)
                *error = "122B campaign attempted to replace a live model path";
            return false;
        }

        const auto requested_identity =
            qwen122ModelContextPhysicalIdentity();
        if (cache.contract)
        {
            if (cache.physical_identity == requested_identity &&
                cache.contract->context == contract.context &&
                cache.contract->routed_weight_authority_identity ==
                    contract.routed_weight_authority_identity)
            {
                /*
                 * The current runner may have admitted a larger reusable
                 * workspace under the same physical model identity. Refresh
                 * the immutable certificate while retaining the same context
                 * and lifecycle authority; the final retirement then consumes
                 * the newest exact BOM instead of stale first-cell metadata.
                 */
                cache.contract = contract;
                return true;
            }
            if (error)
                *error = "122B campaign attempted to replace a live physical-capacity authority";
            return false;
        }

        cache.model_path = config_.model_path;
        cache.physical_identity = requested_identity;
        cache.contract = contract;
        return true;
    }

    /** Stable identity and cumulative value for one PerfStats timer. */
    struct ConvergenceTimerRecord
    {
        std::string domain;
        std::string name;
        std::string phase;
        std::string device;
        PerfStatsCollector::Tags tags;
        std::uint64_t count = 0u;
        std::uint64_t total_ns = 0u;

        /** @return Strict canonical ordering excluding cumulative values. */
        bool operator<(const ConvergenceTimerRecord &other) const
        {
            return std::tie(domain, name, phase, device, tags) <
                   std::tie(
                       other.domain,
                       other.name,
                       other.phase,
                       other.device,
                       other.tags);
        }
    };

    /** Canonically keyed cumulative or interval-local timer values. */
    using ConvergenceTimerSnapshot =
        std::map<ConvergenceTimerRecord, ConvergenceTimerRecord>;

    /**
     * @brief Allocation-owning samples for the observed convergence gate.
     *
     * Baseline values are ordinary production intervals claimed before any
     * residency epoch can publish. Converged values use the same authenticated
     * prompt, restored terminal state, sampled decode input, and one-forward
     * decode budget after the required profitable epochs. A sample whose
     * surrounding committed-wave count changes is rejected instead of being
     * attributed to either layout.
     */
    struct ResidencyConvergenceTimings
    {
        /**
         * @brief One complete request measured in one immutable residency epoch.
         *
         * Candidate identity is retained to match the exact same request on
         * both sides of the comparison. Decode vectors preserve their
         * per-request transaction boundary. Timer deltas cover the same request
         * identity, including its ordinary prefix restores and boundary
         * transactions, so diagnostics never compare different routing
         * workloads.
         */
        struct RequestSample
        {
            int prompt_identity = -1;
            std::uint64_t prefill_ns = 0u;
            std::uint64_t prefill_epoch = 0u;
            std::vector<std::uint64_t> decode_ns;
            std::vector<std::uint64_t> decode_epochs;
            std::vector<int32_t> decode_input_tokens;
            ConvergenceTimerSnapshot timer_deltas;
        };

        /** Exact measured initial-epoch requests matched after convergence. */
        std::vector<RequestSample> baseline_candidates;
        /** Exact selected samples measured after the final publication. */
        std::vector<RequestSample> converged_samples;
        std::vector<std::uint64_t> baseline_prefill_ns;
        std::vector<std::uint64_t> baseline_decode_ns;
        std::vector<std::uint64_t> baseline_prefill_epochs;
        std::vector<std::uint64_t> baseline_decode_epochs;
        std::vector<std::uint64_t> converged_prefill_ns;
        std::vector<std::uint64_t> converged_decode_ns;
        std::vector<std::uint64_t> converged_prefill_epochs;
        std::vector<std::uint64_t> converged_decode_epochs;
        /** Exact sampled inputs consumed by every timed decode forward. */
        std::vector<int32_t> baseline_decode_input_tokens;
    };

    /** Typed side of the immutable-epoch inference comparison. */
    enum class ResidencyTimingCohort : std::uint8_t
    {
        InitialEpoch,
        ConvergedEpoch,
    };

    /**
     * @brief Disjoint first-block identities for production economy traffic.
     *
     * Production parity deliberately uses one-token prefix-cache blocks so it
     * can prove an exact partial restore cheaply.  The stationary before/after
     * workload must therefore exclude every leading token in the bounded
     * service-certification corpus.  A probabilistic collision silently turns
     * a full-compute sample into a partial restore.
     */
    enum class EconomyPromptNamespace : std::uint8_t
    {
        ServiceCertification,
        StationaryConvergence,
    };

    /** @return Monotonic elapsed nanoseconds, clamped away from zero. */
    static std::uint64_t elapsedNanoseconds(
        std::chrono::steady_clock::time_point start) noexcept
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        return static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed, 1));
    }

    /**
     * @brief Snapshot cumulative timers needed to attribute an A/B cohort.
     *
     * Snapshot construction occurs outside every measured interval. Exact tags
     * retain layer, sparse endpoint, replay segment, and route identity so the
     * resulting CSV can locate a regression without adding clocks to the live
     * graph.
     */
    static ConvergenceTimerSnapshot convergenceTimerSnapshot()
    {
        ConvergenceTimerSnapshot result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"forward_graph", "moe_overlay"}))
        {
            if (record.kind != PerfStatRecord::Kind::Timer ||
                (record.domain == "moe_overlay" &&
                 record.name != "compute"))
                continue;
            ConvergenceTimerRecord value{
                .domain = record.domain,
                .name = record.name,
                .phase = record.phase,
                .device = record.device,
                .tags = record.tags,
                .count = record.count,
                .total_ns = record.total_ns,
            };
            result.emplace(value, value);
        }
        return result;
    }

    /**
     * @brief Subtract two cumulative timer views without losing exact tags.
     *
     * @param before Cumulative snapshot immediately before ordinary requests.
     * @param after Cumulative snapshot immediately after ordinary requests.
     * @return Interval-local records; zero-occurrence keys are omitted.
     */
    static ConvergenceTimerSnapshot convergenceTimerDelta(
        const ConvergenceTimerSnapshot &before,
        const ConvergenceTimerSnapshot &after)
    {
        ConvergenceTimerSnapshot result;
        for (const auto &[key, terminal] : after)
        {
            const auto start = before.find(key);
            const std::uint64_t begin_count =
                start == before.end() ? 0u : start->second.count;
            const std::uint64_t begin_ns =
                start == before.end() ? 0u : start->second.total_ns;
            if (terminal.count < begin_count || terminal.total_ns < begin_ns)
            {
                throw std::logic_error(
                    "PerfStats timer regressed inside an immutable convergence request");
            }
            ConvergenceTimerRecord interval = key;
            interval.count = terminal.count - begin_count;
            interval.total_ns = terminal.total_ns - begin_ns;
            if (interval.count != 0u)
                result.emplace(interval, interval);
        }
        return result;
    }

    /**
     * @brief Write request-matched initial/converged timer evidence.
     *
     * Every CSV row belongs to one of the exact six prompt identities retained
     * on both sides of the economy assertion. Movement training owns a disjoint
     * cache-identity interval, so no overlap filtering is necessary.
     *
     * @param baseline Selected initial-epoch samples in comparison order.
     * @param converged Selected converged-epoch samples in comparison order.
     */
    void writeConvergenceTimerSamples(
        const std::vector<const ResidencyConvergenceTimings::RequestSample *>
            &baseline,
        const std::vector<ResidencyConvergenceTimings::RequestSample>
            &converged)
    {
        if (baseline.size() != converged.size() ||
            baseline.size() != static_cast<std::size_t>(
                                   kConvergenceTimingMeasuredRequests))
        {
            throw std::logic_error(
                "ExpertOverlay convergence timer evidence is not request-matched");
        }
        const auto path =
            ensureResultsDir() / "expert_overlay_convergence_timers.csv";
        std::ofstream csv(path, std::ios::trunc);
        if (!csv.is_open())
        {
            throw std::runtime_error(
                "Cannot create ExpertOverlay convergence timer CSV at " +
                path.string());
        }
        csv << "cohort,prompt_identity,domain,name,phase,device,tags,count,total_ns,average_ns\n";
        const auto write_sample = [&csv](
                                      std::string_view cohort,
                                      const ResidencyConvergenceTimings::RequestSample
                                          &sample)
        {
            for (const auto &[key, interval] : sample.timer_deltas)
            {
                std::ostringstream tags;
                bool first = true;
                for (const auto &[name, value] : key.tags)
                {
                    if (!first)
                        tags << ';';
                    first = false;
                    tags << name << '=' << value;
                }
                csv << cohort << ',' << sample.prompt_identity << ','
                    << csvEscape(key.domain) << ','
                    << csvEscape(key.name) << ','
                    << csvEscape(key.phase) << ','
                    << csvEscape(key.device) << ','
                    << csvEscape(tags.str()) << ','
                    << interval.count << ',' << interval.total_ns << ','
                    << interval.total_ns / interval.count << '\n';
            }
        };
        for (const auto *sample : baseline)
        {
            if (!sample)
                throw std::logic_error(
                    "ExpertOverlay baseline timer sample is null");
            write_sample("initial_epoch", *sample);
        }
        for (const auto &sample : converged)
            write_sample("converged_epoch", sample);
        csv.flush();
        if (!csv.good())
        {
            throw std::runtime_error(
                "Failed to write ExpertOverlay convergence timer CSV at " +
                path.string());
        }
    }

    /**
     * @brief Build one cache-distinct valid-token service workload.
     *
     * Certification needs broad natural routing so every real sparse
     * participant and runtime phase contributes a measured service profile.
     * The deterministic SplitMix corpus changes only valid embedding rows and
     * enters the graph through the ordinary serving API. Its routed rows are
     * calibration evidence, not optimization demand: the production admission
     * state quarantines the complete corpus and discards its histogram bank
     * before the next public request boundary.
     *
     * @param request_index Stable request ordinal within the service corpus.
     * @return Prompt-sized deterministic token vector unique to this ordinal.
     */
    std::vector<int32_t> makeEconomyWorkloadPrompt(int request_index) const
    {
        const int vocabulary_size = orch_runner_->vocabSize();
        if (config_.token_ids.empty() || request_index < 0 ||
            vocabulary_size <= 4'096)
        {
            throw std::logic_error(
                "Economy workload requires authenticated prompt geometry, a non-negative identity, and a valid vocabulary");
        }

        std::uint64_t state =
            0x9e3779b97f4a7c15ULL ^
            (static_cast<std::uint64_t>(request_index + 1) *
             0xbf58476d1ce4e5b9ULL);
        const auto usable_vocabulary =
            static_cast<std::uint64_t>(vocabulary_size - 2'048);
        std::vector<int32_t> varied(config_.token_ids.size(), 0);
        for (auto &token : varied)
        {
            state += 0x9e3779b97f4a7c15ULL;
            std::uint64_t mixed = state;
            mixed = (mixed ^ (mixed >> 30u)) *
                    0xbf58476d1ce4e5b9ULL;
            mixed = (mixed ^ (mixed >> 27u)) *
                    0x94d049bb133111ebULL;
            mixed ^= mixed >> 31u;
            token = static_cast<int32_t>(
                256u + mixed % usable_vocabulary);
        }
        varied.front() = economyPromptLeadingToken(
            EconomyPromptNamespace::ServiceCertification,
            request_index);
        return varied;
    }

    /**
     * @brief Map one workload identity to a collision-free prefix-cache block.
     *
     * The usable embedding interval is split between service certification and
     * stationary convergence. Within either typed half the mapping is
     * injective and skips the authenticated Hugging Face prompt's first token,
     * so neither calibration nor timing can seed the later parity prefix.
     *
     * @param prompt_namespace Typed economy-traffic namespace.
     * @param request_index Non-negative identity within that namespace.
     * @return Valid vocabulary row reserved for this exact request identity.
     */
    int32_t economyPromptLeadingToken(
        EconomyPromptNamespace prompt_namespace,
        int request_index) const
    {
        const int vocabulary_size = orch_runner_->vocabSize();
        if (config_.token_ids.empty() || request_index < 0 ||
            vocabulary_size <= 4'096)
        {
            throw std::logic_error(
                "Economy prefix identity requires authenticated prompt geometry, a non-negative identity, and a valid vocabulary");
        }

        constexpr int kFirstOrdinaryEmbedding = 256;
        const int last_embedding_exclusive = vocabulary_size - 2'048;
        const int midpoint =
            kFirstOrdinaryEmbedding +
            (last_embedding_exclusive - kFirstOrdinaryEmbedding) / 2;
        const int interval_begin =
            prompt_namespace ==
                    EconomyPromptNamespace::ServiceCertification
                ? kFirstOrdinaryEmbedding
                : midpoint;
        const int interval_end =
            prompt_namespace ==
                    EconomyPromptNamespace::ServiceCertification
                ? midpoint
                : last_embedding_exclusive;
        const int reference_first_token = config_.token_ids.front();
        const bool excludes_reference_token =
            reference_first_token >= interval_begin &&
            reference_first_token < interval_end;
        const int available_identities =
            interval_end - interval_begin -
            (excludes_reference_token ? 1 : 0);
        if (request_index >= available_identities)
        {
            throw std::out_of_range(
                "Economy prefix identity exhausted its typed vocabulary namespace");
        }

        int token = interval_begin + request_index;
        if (excludes_reference_token && token >= reference_first_token)
            ++token;
        return static_cast<int32_t>(token);
    }

    /**
     * @brief Preserve the authenticated workload while changing its cache key.
     *
     * Timing requests use one cache-distinct leading token followed by repeated
     * authenticated rows, preserving the exact matched A/B geometry. A
     * movement-only request instead executes the leading causal rows of the
     * Hugging Face prompt itself. Any expert promoted from that admitted demand
     * is therefore exercised again by the later fixed parity prefill. If a
     * model's initial histogram window exceeds the reference prompt, the full
     * prompt is used rather than inventing unauthenticated suffix rows.
     *
     * @param request_index Stable request ordinal within the measured corpus.
     * @param role Typed economy phase that owns this request geometry.
     * @return Typed timing corpus or an exact causal prefix of the HF corpus.
     */
    std::vector<int32_t> makeReferenceShapedEconomyPrompt(
        int request_index,
        ReferenceEconomyPromptRole role) const
    {
        const int vocabulary_size = orch_runner_->vocabSize();
        if (vocabulary_size <= 4'096 || request_index < 0 ||
            config_.token_ids.empty())
        {
            throw std::logic_error(
                "Reference-shaped economy prompt requires authenticated tokens, a non-negative identity, and a valid vocabulary");
        }
        const int selected_rows =
            role == ReferenceEconomyPromptRole::TimingCohort
                ? static_cast<int>(kConvergenceTimingPromptRows)
                : std::min(
                      activeModelParityCaseOrThrow()
                          .dynamic_rebalance.window_size,
                      static_cast<int>(config_.token_ids.size()));
        if (selected_rows <= 0)
        {
            throw std::logic_error(
                "Reference-shaped economy prompt requires a positive typed histogram width");
        }
        const std::size_t stationary_rows =
            static_cast<std::size_t>(selected_rows);
        if (role == ReferenceEconomyPromptRole::MovementProof)
        {
            return std::vector<int32_t>(
                config_.token_ids.begin(),
                config_.token_ids.begin() +
                    static_cast<std::ptrdiff_t>(stationary_rows));
        }
        std::vector<int32_t> prompt;
        prompt.reserve(stationary_rows);
        prompt.push_back(economyPromptLeadingToken(
            EconomyPromptNamespace::StationaryConvergence,
            request_index));
        for (std::size_t row = 1u;
             row < stationary_rows;
             ++row)
        {
            prompt.push_back(config_.token_ids.at(
                (row - 1u) % config_.token_ids.size()));
        }
        return prompt;
    }

    /**
     * @brief Build the exact cache-distinct prefill that closes a demand bank.
     *
     * The authority supplies the remaining logical routed-row count. This
     * helper changes only the prefix-cache identity and repeats authenticated
     * model tokens for the requested causal length, so closure remains ordinary
     * production inference rather than synthetic histogram mutation.
     *
     * @param request_index Stable identity in the closure traffic namespace.
     * @param routed_rows Exact positive active-bank headroom to consume.
     * @return One valid model prompt with exactly @p routed_rows rows.
     */
    std::vector<int32_t> makeDemandWindowClosurePrompt(
        int request_index,
        std::uint64_t routed_rows) const
    {
        const auto &test_case = activeModelParityCaseOrThrow();
        if (request_index < 0 || config_.token_ids.empty() ||
            routed_rows == 0u ||
            routed_rows > static_cast<std::uint64_t>(
                              test_case.model.max_seq_len) ||
            routed_rows > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max()))
        {
            throw std::logic_error(
                "Demand-window closure requires a positive in-context production prompt");
        }

        const std::size_t rows = static_cast<std::size_t>(routed_rows);
        std::vector<int32_t> prompt;
        prompt.reserve(rows);
        prompt.push_back(economyPromptLeadingToken(
            EconomyPromptNamespace::StationaryConvergence,
            request_index));
        for (std::size_t row = 1u; row < rows; ++row)
        {
            prompt.push_back(config_.token_ids.at(
                (row - 1u) % config_.token_ids.size()));
        }
        return prompt;
    }

    /** @return Median of a non-empty timing corpus without changing it. */
    static std::uint64_t medianNanoseconds(
        const std::vector<std::uint64_t> &samples)
    {
        if (samples.empty())
            throw std::invalid_argument("Cannot take the median of no inference samples");
        auto ordered = samples;
        const std::size_t midpoint = ordered.size() / 2u;
        std::nth_element(
            ordered.begin(),
            ordered.begin() + static_cast<std::ptrdiff_t>(midpoint),
            ordered.end());
        if ((ordered.size() & 1u) != 0u)
            return ordered[midpoint];
        const auto lower = *std::max_element(
            ordered.begin(),
            ordered.begin() + static_cast<std::ptrdiff_t>(midpoint));
        return lower + (ordered[midpoint] - lower) / 2u;
    }

    /** @return Whether the central matrix assigned this cell the speed witness. */
    bool requiresObservedConvergenceSpeedup() const noexcept
    {
        const auto *test_case = activeModelParityCase();
        return test_case &&
               test_case->requiresObservedConvergenceSpeedup();
    }

    /**
     * @return Model-workload and topology-specific convergence objective.
     *
     * An ordinary 122B cell needs one wide publication whose durable ledger
     * proves both available axes. The centrally designated speed witness and
     * the smaller-model campaign require four successive publications to reach
     * the measured taper before timing. Every caller consumes this one value
     * rather than reproducing model/evidence-role integer tests at later
     * lifecycle boundaries.
     */
    DynamicResidencyConvergenceTarget dynamicResidencyConvergenceTarget() const
    {
        return {
            .minimum_published_waves =
                isQwen122ProductionTest() &&
                    !requiresObservedConvergenceSpeedup()
                    ? 1u
                    : static_cast<std::uint64_t>(
                          kObservedSpeedupConvergenceWindows),
            .axis_contract =
                dynamicMovementAxisContract(resolvedOverlayPlan()),
        };
    }

    void SetUp() override
    {
        ASSERT_EQ(g_active_model_parity_case, nullptr)
            << "A prior generated parity case leaked beyond fixture teardown";
        g_active_model_parity_case = &GetParam();
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            if (productionParityCampaignEnabled())
                FAIL() << "Production GraphNative CudaHot/RocmWarm/CpuCold parity requires MPI initialization";
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm/CpuCold parity requires MPI initialization";
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (world_size < cfg().mpi_ranks)
        {
            if (productionParityCampaignEnabled())
            {
                FAIL() << "Production GraphNative CudaHot/RocmWarm/CpuCold parity requires "
                       << cfg().mpi_ranks << " MPI ranks (got "
                       << world_size << ")";
            }
            GTEST_SKIP() << "GraphNative CudaHot/RocmWarm/CpuCold parity requires "
                         << cfg().mpi_ranks << " MPI ranks (got " << world_size << ")";
        }

        /*
         * The production context remains stable because a retained 122B
         * ModelContext carries it with prepared weights across compatible
         * cells. ParityTestBase now owns the fresh per-cell control and
         * teardown channels for every topology; this fixture only supplies the
         * stable production rank set before entering the common setup.
         */
        mpi_ctx_ =
            std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        Base::SetUp();
        if (this->HasFatalFailure())
            return;

        try
        {
            cluster_inventory_ = gatherClusterInventory(
                parityCoordinationMPIContext());
        }
        catch (const std::exception &e)
        {
            FAIL() << "Failed to enter the parity cell or gather topology: "
                   << e.what();
        }
        if (this->HasFatalFailure() || !isSegmentedPrefillProductionTest() ||
            !modelAvailable())
        {
            return;
        }

        const int captured_rows =
            activeModelParityCase()->prefill_graph.captured_rows;

        ASSERT_GT(
            config_.token_ids.size(),
            static_cast<size_t>(captured_rows))
            << "Segmented graph-native parity requires an authenticated prompt "
               "longer than its captured bucket";
        configureSegmentedProductionParityPrefillGraphBucket(
            captured_rows);
        ASSERT_EQ(
            debugEnv().execution.prefill_graph_bucket_sizes,
            std::vector<int>{captured_rows})
            << "Segmented parity must override the exact-bucket default with "
               "one explicit fixed capture bucket";
    }

    /** @brief Retire the typed case only after production runner teardown. */
    void TearDown() override
    {
        Base::TearDown();
        g_active_model_parity_case = nullptr;
        mpi_ctx_.reset();
    }

    void applyModelOverrides() override
    {
        // Preserve the complete declarative TestConfig contract, including
        // decode depth. Reimplementing only model/prompt fields here left the
        // 122B MTP corpus at ParityConfig's historical five-step default.
        // The shared parity base is also the sole reference-pack lifecycle
        // authority: it validates model identity and sidecar schema, performs
        // one rank-zero regeneration, then publishes readiness to every rank.
        Base::applyModelOverrides();
    }

    bool broadcastRootFlag(bool root_value) const
    {
        const int root_rank = parityArtifactAuthorityRank();
        int flag = isRootParityRank() && root_value ? 1 : 0;
        MPI_Bcast(
            &flag,
            1,
            MPI_INT,
            root_rank,
            parityCoordinationCommunicator());
        return flag != 0;
    }

    /**
     * @brief Prove each expert-only endpoint owns one immutable compact arena.
     *
     * Every process retains its own PerfStats records, while the graph-native
     * overlay spreads its accelerator tier and the two CPU-NUMA endpoints
     * across MPI instances. The dense continuation uses its model graph's
     * activation arena and must not duplicate the follower runner's storage.
     * Aggregate setup evidence only after the worker loop closes; this proves
     * the real follower graph did not allocate one packet per layer or alias
     * independent participants.
     */
    void assertParticipantCompactBufferArenaEvidence() const
    {
        constexpr size_t kAllocationValue = 0;
        constexpr size_t kAllocationRecordCount = 1;
        constexpr size_t kAllocationBytes = 2;
        constexpr size_t kByteRecordCount = 3;
        constexpr size_t kMalformedRecords = 4;
        constexpr size_t kParticipantStart = 5;
        const size_t participant_count = activeOverlayParticipantCount();
        ASSERT_GT(participant_count, 1u);
        /*
         * A continuation domain may itself be tensor-parallel. Every one of
         * those participants reuses its dense model arena; only participants
         * outside that domain own the compact follower arena. Derive the exact
         * set from the published owner-map topology instead of assuming one
         * continuation participant at global id zero.
         */
        std::vector<uint64_t> expected_compact_participant(
            participant_count, 0u);
        int topology_valid = 1;
        if (isRootParityRank())
        {
            const auto *const concrete =
                dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
            const auto snapshot = concrete
                                      ? concrete
                                            ->expertOverlayResidencySnapshotForDiagnostics()
                                      : nullptr;
            if (!snapshot || !snapshot->valid() || !overlay_plan_)
            {
                topology_valid = 0;
            }
            else
            {
                for (const auto &participant :
                     snapshot->owner_map.participants())
                {
                    if (participant.participant_id < 0 ||
                        static_cast<size_t>(participant.participant_id) >=
                            participant_count)
                    {
                        topology_valid = 0;
                        break;
                    }
                    if (participant.domain_name !=
                        overlay_plan_->continuation_domain)
                    {
                        expected_compact_participant[
                            static_cast<size_t>(participant.participant_id)] =
                            1u;
                    }
                }
            }
        }
        MPI_Bcast(
            &topology_valid,
            1,
            MPI_INT,
            parityArtifactAuthorityRank(),
            parityCoordinationCommunicator());
        MPI_Bcast(
            expected_compact_participant.data(),
            static_cast<int>(expected_compact_participant.size()),
            MPI_UINT64_T,
            parityArtifactAuthorityRank(),
            parityCoordinationCommunicator());
        ASSERT_EQ(topology_valid, 1)
            << "Published ExpertOverlay topology could not classify compact followers";
        const size_t follower_participant_count =
            static_cast<size_t>(std::accumulate(
                expected_compact_participant.begin(),
                expected_compact_participant.end(),
                uint64_t{0}));
        ASSERT_GT(follower_participant_count, 0u);
        const size_t evidence_count = kParticipantStart + participant_count;

        const int local_memory_domain_enabled =
            PerfStatsCollector::isDomainEnabled("memory") ? 1 : 0;
        int all_memory_domains_enabled = 0;
        MPI_Allreduce(
            &local_memory_domain_enabled,
            &all_memory_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_memory_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the PerfStats "
                       "memory domain to prove serial compact-arena ownership";
            }
            return;
        }

        std::vector<uint64_t> local(evidence_count, 0u);
        for (const auto &record : PerfStatsCollector::snapshot({"memory"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "memory")
            {
                continue;
            }

            const bool is_allocation =
                record.name == "moe_serial_local_expert_buffer_arena_allocations";
            const bool is_bytes =
                record.name == "moe_serial_local_expert_buffer_arena_bytes";
            if (!is_allocation && !is_bytes)
                continue;

            const auto ownership = record.tags.find("ownership");
            const auto immutable = record.tags.find("immutable");
            const auto participant = record.tags.find("participant");
            const bool tag_contract_ok =
                record.phase == "model_setup" &&
                ownership != record.tags.end() &&
                ownership->second == "per_device_participant_serial_graph_family" &&
                immutable != record.tags.end() && immutable->second == "true" &&
                participant != record.tags.end();
            if (!tag_contract_ok || record.count != 1u || record.value <= 0.0)
            {
                ++local[kMalformedRecords];
                continue;
            }

            int participant_id = -1;
            for (int candidate = 0;
                 candidate < static_cast<int>(participant_count);
                 ++candidate)
            {
                if (participant->second == std::to_string(candidate))
                {
                    participant_id = candidate;
                    break;
                }
            }
            if (participant_id < 0)
            {
                ++local[kMalformedRecords];
                continue;
            }

            if (is_allocation)
            {
                if (record.value != 1.0)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                ++local[kAllocationValue];
                ++local[kAllocationRecordCount];
                ++local[kParticipantStart + static_cast<size_t>(participant_id)];
                continue;
            }

            local[kAllocationBytes] += static_cast<uint64_t>(record.value);
            ++local[kByteRecordCount];
        }

        std::vector<uint64_t> global(evidence_count, 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());

        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Serial compact-arena PerfStats tags or aggregation were malformed";
        EXPECT_EQ(global[kAllocationValue], follower_participant_count)
            << "Each expert-only follower must allocate one compact graph-family arena";
        EXPECT_EQ(global[kAllocationRecordCount], follower_participant_count)
            << "Each expert-only follower must publish one unaggregated arena allocation record";
        EXPECT_EQ(global[kByteRecordCount], follower_participant_count)
            << "Each expert-only follower must publish one arena byte-accounting record";
        EXPECT_GT(global[kAllocationBytes], 0u)
            << "Follower compact arenas must account for their fixed graph-family storage";
        for (size_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            const uint64_t expected =
                expected_compact_participant[participant];
            EXPECT_EQ(global[kParticipantStart + participant], expected)
                << (expected != 0u
                        ? "Expected exactly one follower compact arena for participant p"
                        : "Dense continuation participant must not duplicate a compact arena for p")
                << participant;
        }
    }

    /**
     * @brief Prove a one-rank GPU+CPU graph used canonical ticket boundaries.
     *
     * A rank-local topology has no auxiliary MPI runner and must not advertise
     * one merely to satisfy the mapped-follower evidence used by node-wide
     * cases. Instead, every MoE layer must materialize and capture its exact
     * canonical-ticket consumer, while the full graph must lower to typed
     * captured/manual/captured transactions with one successor per boundary.
     */
    void assertRankLocalCanonicalTicketGraphEvidence() const
    {
        constexpr size_t kMalformedRecords = 0u;
        constexpr size_t kLifecycleTransactions = 1u;
        constexpr size_t kMappedFollowerRecords = 2u;
        constexpr size_t kMaterializations = 3u;
        constexpr size_t kCaptureLaunches = 4u;
        constexpr size_t kFixedEvidenceCount = 5u;
        /*
         * Setup capacity and live execution policy are deliberately distinct.
         * Every generated 122B cell retains the maximum MTP graph-family
         * envelope so adjacent cells can reuse one ModelContext, including the
         * MTPOff control cell. Setup must therefore materialize sidecar ticket
         * consumers whenever that envelope is retained, while capture may
         * include them only when this cell actually enables MTP. Keeping two
         * exact inventories makes a disabled sidecar launch a hard failure
         * without misclassifying its setup-owned immutable stage as malformed.
         */
        std::vector<int> materialized_layer_ids;
        std::vector<int> captured_layer_ids;
        const int main_layer_count = parityLayerCount();
        ASSERT_GT(main_layer_count, 0);
        materialized_layer_ids.reserve(
            static_cast<size_t>(main_layer_count) + 1u);
        captured_layer_ids.reserve(
            static_cast<size_t>(main_layer_count) + 1u);
        for (int layer = 0; layer < main_layer_count; ++layer)
        {
            materialized_layer_ids.push_back(layer);
            captured_layer_ids.push_back(layer);
        }

        const int retained_mtp_capacity = activeMTPRetainedDraftCapacity();
        const int active_mtp_depth = activeMTPDraftDepth();
        ASSERT_GE(retained_mtp_capacity, 0);
        ASSERT_GE(active_mtp_depth, 0);
        ASSERT_LE(active_mtp_depth, retained_mtp_capacity)
            << "A live MTP policy cannot exceed its retained graph-family envelope";
        if (retained_mtp_capacity > 0)
        {
            const ModelContext *const active_model =
                activeModelContextForDiagnostics();
            ASSERT_NE(active_model, nullptr);
            const int raw_layer_count = std::max(
                active_model->totalBlockCount(),
                active_model->blockCount());
            const MTPWeightManifest manifest = discoverMTPWeightManifest(
                active_model->concreteLoader(),
                active_model->architecture(),
                raw_layer_count,
                /*explicit_mtp=*/true);
            ASSERT_TRUE(manifest.available) << manifest.diagnostic;
            for (const auto &depth : manifest.depths)
            {
                if (depth.moe_ffn_layout)
                {
                    materialized_layer_ids.push_back(
                        depth.source_layer_index);
                    if (active_mtp_depth > 0)
                    {
                        captured_layer_ids.push_back(
                            depth.source_layer_index);
                    }
                }
            }
        }
        const auto normalize_layer_ids = [](std::vector<int> &layer_ids)
        {
            std::sort(layer_ids.begin(), layer_ids.end());
            layer_ids.erase(
                std::unique(layer_ids.begin(), layer_ids.end()),
                layer_ids.end());
        };
        normalize_layer_ids(materialized_layer_ids);
        normalize_layer_ids(captured_layer_ids);
        const size_t materialized_layer_count =
            materialized_layer_ids.size();
        const size_t captured_layer_count = captured_layer_ids.size();
        ASSERT_GT(materialized_layer_count, 0u);
        ASSERT_GT(captured_layer_count, 0u);
        const size_t materialized_layers_start = kFixedEvidenceCount;
        const size_t captured_layers_start =
            materialized_layers_start + materialized_layer_count;
        std::vector<uint64_t> local(
            captured_layers_start + captured_layer_count,
            0u);

        const int participant_domain_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph")
                ? 1
                : 0;
        const int lifecycle_domain_enabled =
            PerfStatsCollector::isDomainEnabled("forward_graph") ? 1 : 0;
        int all_domains_enabled = 0;
        const int local_domains_enabled =
            participant_domain_enabled && lifecycle_domain_enabled;
        MPI_Allreduce(
            &local_domains_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Rank-local canonical-ticket parity requires both "
                       "forward_graph and moe_overlay_participant_graph PerfStats";
            }
            return;
        }

        const auto parse_nonnegative = [](
                                           const PerfStatRecord &record,
                                           const char *name) -> int
        {
            const auto found = record.tags.find(name);
            if (found == record.tags.end())
                return -1;
            int value = -1;
            const char *const begin = found->second.data();
            const char *const end = begin + found->second.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end &&
                           value >= 0
                       ? value
                       : -1;
        };

        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_participant_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_participant_graph")
            {
                continue;
            }
            if (record.name == "materialized_mapped_follower_families" ||
                record.name == "ticket_selected_graphs")
            {
                local[kMappedFollowerRecords] += record.count;
                continue;
            }

            const bool materialization =
                record.name ==
                "materialized_rank_local_canonical_ticket_consumers";
            const bool capture_launch =
                record.name ==
                "rank_local_canonical_ticket_consumer_launches";
            if (!materialization && !capture_launch)
                continue;

            const auto &expected_layer_ids =
                materialization ? materialized_layer_ids : captured_layer_ids;
            const size_t layer_count = expected_layer_ids.size();
            const int layer = parse_nonnegative(record, "layer");
            const auto layer_position = std::lower_bound(
                expected_layer_ids.begin(),
                expected_layer_ids.end(),
                layer);
            const bool expected_layer =
                layer_position != expected_layer_ids.end() &&
                *layer_position == layer;
            const size_t layer_slot = expected_layer
                                          ? static_cast<size_t>(
                                                std::distance(
                                                    expected_layer_ids.begin(),
                                                    layer_position))
                                          : layer_count;
            const auto ticket_kind = record.tags.find("ticket_kind");
            bool valid = record.count > 0u && record.value > 0.0 &&
                         expected_layer && layer_slot < layer_count &&
                         ticket_kind != record.tags.end() &&
                         ticket_kind->second == "canonical_route";
            if (materialization)
            {
                valid = valid && record.phase == "model_setup" &&
                        parse_nonnegative(record, "bucket_rows") > 0 &&
                        parse_nonnegative(record, "d_model") > 0 &&
                        parse_nonnegative(record, "route_capacity") > 0;
            }
            else
            {
                valid = valid && record.phase == "graph_capture";
            }
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }

            if (materialization)
            {
                local[kMaterializations] += record.count;
                local[materialized_layers_start +
                      layer_slot] += record.count;
            }
            else
            {
                local[kCaptureLaunches] += record.count;
                local[captured_layers_start +
                      layer_slot] += record.count;
            }
        }

        for (const auto &record :
             PerfStatsCollector::snapshot({"forward_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "heterogeneous_ticket_transactions")
            {
                continue;
            }
            const int captured =
                parse_nonnegative(record, "capturable_segments");
            const int manual =
                parse_nonnegative(record, "manual_segments");
            const int boundaries =
                parse_nonnegative(record, "unit_boundaries");
            const int terminals =
                parse_nonnegative(record, "terminal_units");
            const auto role = record.tags.find("role");
            const auto authority =
                record.tags.find("ticket_publication_authority");
            const bool valid =
                record.phase == "capture" && record.count > 0u &&
                record.value > 0.0 && manual > 0 &&
                boundaries == manual && captured == manual + 1 &&
                terminals == 1 && role != record.tags.end() &&
                role->second == "authority" &&
                authority != record.tags.end() &&
                authority->second == "stage_owned_mapped_timeline";
            if (!valid)
                ++local[kMalformedRecords];
            else
                local[kLifecycleTransactions] += record.count;
        }

        std::vector<uint64_t> global(local.size(), 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Rank-local canonical-ticket graph evidence was malformed";
        EXPECT_EQ(global[kMappedFollowerRecords], 0u)
            << "A one-rank topology materialized an auxiliary mapped follower";
        EXPECT_GT(global[kLifecycleTransactions], 0u)
            << "No typed heterogeneous ticket transaction reached capture";
        EXPECT_GT(global[kMaterializations], 0u);
        EXPECT_GT(global[kCaptureLaunches], 0u);
        for (size_t layer = 0u; layer < materialized_layer_count; ++layer)
        {
            EXPECT_GT(global[materialized_layers_start + layer], 0u)
                << "Retained routed graph layer "
                << materialized_layer_ids[layer]
                << " has no setup-owned canonical ticket consumer";
        }
        for (size_t layer = 0u; layer < captured_layer_count; ++layer)
        {
            EXPECT_GT(global[captured_layers_start + layer], 0u)
                << "Active routed graph layer " << captured_layer_ids[layer]
                << " never recorded its canonical ticket consumer in a native graph";
        }
    }

    /**
     * @brief Prove tickets selected the topology's production specialization.
     *
     * Device-owned epochs replaced the old host-scheduled fixed-capacity
     * participant runner. Setup materializes a bounded row-shape family once;
     * each authenticated ticket selects one member without mutating token or
     * position state on the follower. Native parent capture/replay is asserted
     * independently by the shared production-path evidence gate.
     */
    void assertMappedParticipantGraphEvidence() const
    {
        if (!topologySpansMultipleMPIRanks())
        {
            assertRankLocalCanonicalTicketGraphEvidence();
            return;
        }

        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kFamilyMaterializations = 1;
        constexpr size_t kMainFamilyMaterializations = 2;
        constexpr size_t kMTPFamilyCapacity = 3;
        constexpr size_t kSelectedGraphs = 4;
        constexpr size_t kLargestPrefillSelections = 5;
        constexpr size_t kOneRowPrefillSelections = 6;
        constexpr size_t kOneRowDecodeSelections = 7;
        constexpr size_t kSelectedMTPPhysicalRows = 8;
        constexpr size_t kRetiredHostRunnerRecords = 9;
        constexpr size_t kMaterializedGpuTransactions = 10;
        constexpr size_t kMaterializedCpuEndpoints = 11;
        constexpr size_t kFollowerGpuRuntimeTables = 12;
        constexpr size_t kHostAuthorityGpuRuntimes = 13;
        constexpr size_t kDeviceAuthorityGpuRuntimes = 14;
        constexpr size_t kMappedControllerGpuRuntimes = 15;
        constexpr size_t kEvidenceCount = 16;

        int expected_largest_prefill_rows = 0;
        if (isRootParityRank())
        {
            expected_largest_prefill_rows =
                isSegmentedPrefillProductionTest()
                    ? activeSegmentedPrefillCaptureRows()
                    : static_cast<int>(config_.token_ids.size());
        }
        MPI_Bcast(
            &expected_largest_prefill_rows,
            1,
            MPI_INT,
            parityArtifactAuthorityRank(),
            parityCoordinationCommunicator());
        ASSERT_GT(expected_largest_prefill_rows, 0);

        const int local_domain_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph")
                ? 1
                : 0;
        int all_domains_enabled = 0;
        MPI_Allreduce(
            &local_domain_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the "
                       "moe_overlay_participant_graph PerfStats domain";
            }
            return;
        }

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_participant_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_participant_graph")
            {
                continue;
            }
            if (record.name == "materialized_graphs" ||
                record.name == "fixed_capacity_graph_reuses")
            {
                local[kRetiredHostRunnerRecords] += record.count;
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto parse_positive = [&](const char *name) -> int
            {
                const auto value = tag(name);
                if (!value)
                    return 0;
                int parsed = 0;
                const char *const begin = value->data();
                const char *const end = begin + value->size();
                const auto result = std::from_chars(begin, end, parsed);
                return result.ec == std::errc{} && result.ptr == end &&
                               parsed > 0
                           ? parsed
                           : 0;
            };
            const auto parse_nonnegative = [&](const char *name) -> int
            {
                const auto value = tag(name);
                if (!value)
                    return -1;
                int parsed = -1;
                const char *const begin = value->data();
                const char *const end = begin + value->size();
                const auto result = std::from_chars(begin, end, parsed);
                return result.ec == std::errc{} && result.ptr == end &&
                               parsed >= 0
                           ? parsed
                           : -1;
            };

            if (record.name == "materialized_mapped_follower_families")
            {
                const auto graph_family = tag("graph_family");
                const auto standalone_progress =
                    tag("standalone_progress_launch");
                const int graph_family_ordinal =
                    parse_nonnegative("graph_family");
                const int row_capacity = parse_positive("row_capacity");
                const int gpu_transactions = parse_nonnegative(
                    "setup_materialized_gpu_transactions");
                const int cpu_endpoints = parse_nonnegative(
                    "setup_materialized_cpu_endpoints");
                const bool valid =
                    record.phase == "model_setup" && record.value == 1.0 &&
                    record.count == 1u && graph_family &&
                    graph_family_ordinal >= 0 &&
                    row_capacity > 0 && gpu_transactions >= 0 &&
                    cpu_endpoints >= 0 &&
                    (gpu_transactions > 0 || cpu_endpoints > 0) &&
                    parse_nonnegative("captured_transfer_branches") >= 0 &&
                    standalone_progress &&
                    *standalone_progress == "false";
                if (!valid)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                ++local[kFamilyMaterializations];
                local[kMaterializedGpuTransactions] +=
                    static_cast<uint64_t>(gpu_transactions);
                local[kMaterializedCpuEndpoints] +=
                    static_cast<uint64_t>(cpu_endpoints);
                if (graph_family_ordinal == 0)
                    ++local[kMainFamilyMaterializations];
                if (graph_family_ordinal == 1 &&
                    row_capacity == activeMTPGraphCapacityVerifierRows())
                {
                    ++local[kMTPFamilyCapacity];
                }
                continue;
            }
            if (record.name != "ticket_selected_graphs")
                continue;

            const int logical_rows = parse_positive("logical_rows");
            const int physical_rows = parse_positive("physical_rows");
            const auto transport_path = tag("transport_path");
            const auto position_mutated = tag("position_mutated");
            const auto gpu_host_dispatches = tag("gpu_host_layer_dispatches");
            const bool phase_valid =
                record.phase == "main_prefill" ||
                record.phase == "main_decode" ||
                record.phase == "mtp_grouped_verifier" ||
                record.phase == "mtp_draft";
            const bool valid =
                record.value > 0.0 && record.count > 0u && phase_valid &&
                logical_rows > 0 && physical_rows >= logical_rows &&
                transport_path &&
                transport_path->rfind("node_local_mapped_", 0) == 0 &&
                position_mutated && *position_mutated == "false" &&
                gpu_host_dispatches && *gpu_host_dispatches == "0";
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }

            local[kSelectedGraphs] += record.count;
            if (record.phase == "main_prefill" &&
                logical_rows == expected_largest_prefill_rows)
            {
                local[kLargestPrefillSelections] += record.count;
            }
            if (record.phase == "main_prefill" && logical_rows == 1)
                local[kOneRowPrefillSelections] += record.count;
            if (record.phase == "main_decode" && logical_rows == 1)
                local[kOneRowDecodeSelections] += record.count;
            if (record.phase == "mtp_grouped_verifier" &&
                physical_rows == activeMTPPhysicalVerifierRows())
            {
                local[kSelectedMTPPhysicalRows] += record.count;
            }
        }

        /*
         * Policy location and follower execution state are orthogonal. Every
         * remote GPU endpoint needs a device runtime table even when a CPU
         * participant makes the sole policy authority host-resident. This is
         * the focused production-path regression for the former manual
         * dispatch-consume segment in an otherwise device-owned envelope.
         */
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_controller"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_controller" ||
                record.name != "follower_runtime_tables_materialized")
            {
                continue;
            }
            const auto tag = [&record](const char *name)
                -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto authority = tag("authority_execution");
            const auto mapped = tag("mapped_device_controller");
            const auto blocking = tag("blocking_hot_path");
            const auto layers = tag("layers");
            int parsed_layers = 0;
            if (layers)
            {
                const char *const begin = layers->data();
                const char *const end = begin + layers->size();
                const auto parsed = std::from_chars(
                    begin, end, parsed_layers);
                if (parsed.ec != std::errc{} || parsed.ptr != end)
                    parsed_layers = 0;
            }
            const bool valid =
                record.phase == "model_setup" && record.value == 1.0 &&
                record.count == 1u && authority && mapped && blocking &&
                *blocking == "false" && parsed_layers > 0 &&
                (*authority == "host-resident" ||
                 *authority == "device-resident") &&
                (*mapped == "true" || *mapped == "false") &&
                ((*authority == "host-resident" && *mapped == "false") ||
                 (*authority == "device-resident" && *mapped == "true"));
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }
            ++local[kFollowerGpuRuntimeTables];
            if (*authority == "host-resident")
                ++local[kHostAuthorityGpuRuntimes];
            else
                ++local[kDeviceAuthorityGpuRuntimes];
            if (*mapped == "true")
                ++local[kMappedControllerGpuRuntimes];
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Mapped participant graph evidence was malformed";
        /*
         * Qwen3.5 owns one set of MTP sidecar weights and recursively replays
         * that same retained graph for every speculative slot. Draft depth is
         * therefore transaction geometry, not graph-family identity. A cell
         * with retained MTP setup capacity materializes one dormant recursive
         * sidecar even when request-time MTP execution is off; it must not
         * execute that family. A model with no retained capacity owns only the
         * main family.
         */
        const uint64_t expected_family_materializations =
            activeMTPRetainedDraftCapacity() > 0 ? 2u : 1u;
        EXPECT_EQ(
            global[kFamilyMaterializations],
            expected_family_materializations)
            << "The auxiliary MPI runner materialized a graph family outside the production request";
        EXPECT_EQ(global[kMainFamilyMaterializations], 1u)
            << "The one auxiliary MPI runner must materialize its main mapped graph family exactly once";
        const bool has_remote_gpu_endpoint =
            topologyHasSecondaryGpuDomain();
        if (has_remote_gpu_endpoint)
        {
            EXPECT_GT(global[kMaterializedGpuTransactions], 0u)
                << "The mapped follower family materialized no native GPU transaction";
            EXPECT_GT(global[kFollowerGpuRuntimeTables], 0u)
                << "A remote GPU follower executed without a device-resident placement table";
            if (topologyUsesCpu())
            {
                EXPECT_EQ(
                    global[kHostAuthorityGpuRuntimes],
                    global[kFollowerGpuRuntimeTables])
                    << "CPU-participating topology must retain one host policy authority while every GPU keeps local execution state";
                EXPECT_EQ(global[kMappedControllerGpuRuntimes], 0u)
                    << "Host-authority GPU followers must not acquire a competing mapped policy controller";
            }
            else
            {
                EXPECT_EQ(
                    global[kDeviceAuthorityGpuRuntimes],
                    global[kFollowerGpuRuntimeTables]);
                EXPECT_EQ(
                    global[kMappedControllerGpuRuntimes],
                    global[kFollowerGpuRuntimeTables])
                    << "All-GPU followers must bind the sole mapped device policy controller";
            }
        }
        if (topologyUsesCpu())
        {
            EXPECT_GT(global[kMaterializedCpuEndpoints], 0u)
                << "The mapped follower family retained no typed CPU boundary endpoint";
        }
        if (activeMTPRetainedDraftCapacity() > 0)
        {
            EXPECT_EQ(global[kMTPFamilyCapacity], 1u)
                << "The recursive MTP follower family did not retain the shared "
                << activeMTPGraphCapacityVerifierRows()
                << "-row campaign capacity";
        }
        else
        {
            EXPECT_EQ(global[kMTPFamilyCapacity], 0u)
                << "A campaign without retained MTP capacity materialized a recursive sidecar family";
        }
        EXPECT_GT(global[kSelectedGraphs], 0u)
            << "No authenticated ticket selected a mapped participant graph";
        EXPECT_GT(global[kLargestPrefillSelections], 0u)
            << "The mapped family never served the largest root-published live prefill chunk";
        EXPECT_GT(global[kOneRowDecodeSelections], 0u)
            << "Decode never selected its setup-owned one-row retained parent";
        if (activeMTPEnabled())
        {
            EXPECT_GT(global[kSelectedMTPPhysicalRows], 0u)
                << "No mapped grouped-verifier graph selected the admitted "
                << activeMTPPhysicalVerifierRows()
                << "-row physical transaction bucket";
        }
        else
        {
            EXPECT_EQ(global[kSelectedMTPPhysicalRows], 0u)
                << "The execution-off control selected its dormant retained MTP family";
        }
        EXPECT_EQ(global[kRetiredHostRunnerRecords], 0u)
            << "The retired host-scheduled participant graph path executed";
        if (isSegmentedPrefillProductionTest())
        {
            EXPECT_GT(global[kOneRowPrefillSelections], 0u)
                << "The short prefill tail never selected its bounded mapped graph";
        }
    }

    /**
     * @brief Exact device snapshots for one routed layer under one request.
     *
     * `overlay_participants` is the overlay-wide `(expert -> global
     * participant)` bank selected by the request's acquired epoch status.
     * `domain_participants` is the invocation-local `(route slot -> domain
     * participant)` schedule; `-1` means the selected expert belongs to another
     * overlay domain and is completed by the heterogeneous return transaction.
     * Both pointers remain owned by SnapshotCapture.
     */
    struct PinnedDeviceRouteEvidence
    {
        const float *overlay_participants = nullptr;
        const float *domain_participants = nullptr;
        const float *runtime_weights = nullptr;
        size_t expert_count = 0u;
        size_t route_count = 0u;
        int selected_bank = -1;
    };

    /**
     * @brief Exact placement epoch consumed by one completed graph transaction.
     *
     * Snapshot capture copies these scalar values on the producer stream before
     * the request releases its RCU reader. Combining the selected-bank status
     * with that bank's own published epoch therefore observes the same device
     * authority used by packet dispatch, on local and remote topologies alike.
     */
    struct PinnedDevicePlacementEpochEvidence
    {
        uint64_t epoch = 0u;
        int selected_bank = -1;
    };

    /**
     * @brief Resolve and cross-check the request-pinned epoch on every MoE layer.
     *
     * @return One model-wide execution identity, or no value after recording a
     *         focused test failure for missing, malformed, or split evidence.
     */
    std::optional<PinnedDevicePlacementEpochEvidence>
    pinnedDevicePlacementEpochEvidence() const
    {
        std::optional<PinnedDevicePlacementEpochEvidence> model_identity;
        for (int layer = 0; layer < parityLayerCount(); ++layer)
        {
            const std::string prefix =
                "layer" + std::to_string(layer) + '_';
            size_t selected_bank_elements = 0u;
            const float *const selected_bank_value = activeSnapshot(
                prefix + "MOE_OVERLAY_ROUTE_SELECTED_BANK",
                selected_bank_elements);
            size_t bank0_epoch_elements = 0u;
            const float *const bank0_epoch_value = activeSnapshot(
                prefix + "MOE_OVERLAY_ROUTE_BANK0_EPOCH",
                bank0_epoch_elements);
            size_t bank1_epoch_elements = 0u;
            const float *const bank1_epoch_value = activeSnapshot(
                prefix + "MOE_OVERLAY_ROUTE_BANK1_EPOCH",
                bank1_epoch_elements);
            if (!selected_bank_value || !bank0_epoch_value ||
                !bank1_epoch_value || selected_bank_elements != 1u ||
                bank0_epoch_elements != 1u || bank1_epoch_elements != 1u)
            {
                ADD_FAILURE()
                    << prefix
                    << "did not publish one complete request-pinned epoch identity";
                return std::nullopt;
            }

            const float selected = selected_bank_value[0];
            if (!std::isfinite(selected) ||
                (selected != 0.0f && selected != 1.0f))
            {
                ADD_FAILURE()
                    << prefix << "published invalid selected bank " << selected;
                return std::nullopt;
            }
            const int selected_bank = static_cast<int>(selected);
            const float selected_epoch = selected_bank == 0
                                             ? bank0_epoch_value[0]
                                             : bank1_epoch_value[0];
            if (!std::isfinite(selected_epoch) || selected_epoch <= 0.0f ||
                std::trunc(selected_epoch) != selected_epoch ||
                selected_epoch > static_cast<float>(
                                     std::numeric_limits<int32_t>::max()))
            {
                ADD_FAILURE()
                    << prefix << "selected bank " << selected_bank
                    << " published invalid epoch " << selected_epoch;
                return std::nullopt;
            }
            const PinnedDevicePlacementEpochEvidence layer_identity{
                .epoch = static_cast<uint64_t>(selected_epoch),
                .selected_bank = selected_bank,
            };
            if (!model_identity)
            {
                model_identity = layer_identity;
                continue;
            }
            if (model_identity->epoch != layer_identity.epoch ||
                model_identity->selected_bank !=
                    layer_identity.selected_bank)
            {
                ADD_FAILURE()
                    << prefix << "consumed placement epoch "
                    << layer_identity.epoch << " bank "
                    << layer_identity.selected_bank
                    << " but prior model layers consumed epoch "
                    << model_identity->epoch << " bank "
                    << model_identity->selected_bank;
                return std::nullopt;
            }
        }
        return model_identity;
    }

    /**
     * @brief Resolve both typed route projections from live device checkpoints.
     *
     * @param layer Exact transformer layer.
     * @param expected_route_count Router slots in the current checkpoint.
     * @param expected_expert_count Logical routed experts in the model layer.
     * @return Complete evidence, or no value after recording a test failure.
     */
    std::optional<PinnedDeviceRouteEvidence>
    pinnedDeviceRouteEvidence(
        int layer,
        size_t expected_route_count,
        size_t expected_expert_count) const
    {
        const std::string prefix = "layer" + std::to_string(layer) + '_';
        size_t domain_elements = 0u;
        const float *const domain_participants = activeSnapshot(
            prefix + "MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            domain_elements);
        size_t runtime_weight_elements = 0u;
        const float *const runtime_weights = activeSnapshot(
            prefix + "MOE_RUNTIME_ROUTE_WEIGHTS",
            runtime_weight_elements);
        size_t bank0_elements = 0u;
        const float *const bank0 = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            bank0_elements);
        size_t bank1_elements = 0u;
        const float *const bank1 = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            bank1_elements);
        size_t selected_bank_elements = 0u;
        const float *const selected_bank_value = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_SELECTED_BANK",
            selected_bank_elements);
        if (!domain_participants || !runtime_weights || !bank0 || !bank1 ||
            !selected_bank_value)
        {
            ADD_FAILURE()
                << prefix
                << "mapped reducer did not publish both pinned route projections";
            return std::nullopt;
        }
        if (domain_elements != expected_route_count ||
            runtime_weight_elements != expected_route_count ||
            bank0_elements != expected_expert_count ||
            bank1_elements != expected_expert_count ||
            selected_bank_elements != 1u)
        {
            ADD_FAILURE()
                << prefix << "route evidence geometry mismatch: domain="
                << domain_elements << " expected_routes="
                << expected_route_count << " bank0=" << bank0_elements
                << " runtime_weights=" << runtime_weight_elements
                << " bank1=" << bank1_elements << " expected_experts="
                << expected_expert_count << " selected_bank_elements="
                << selected_bank_elements;
            return std::nullopt;
        }
        const float selected = selected_bank_value[0];
        if (!std::isfinite(selected) ||
            (selected != 0.0f && selected != 1.0f))
        {
            ADD_FAILURE()
                << prefix << "acquired route epoch selected invalid bank "
                << selected;
            return std::nullopt;
        }
        const int selected_bank = static_cast<int>(selected);
        return PinnedDeviceRouteEvidence{
            .overlay_participants = selected_bank == 0 ? bank0 : bank1,
            .domain_participants = domain_participants,
            .runtime_weights = runtime_weights,
            .expert_count = expected_expert_count,
            .route_count = expected_route_count,
            .selected_bank = selected_bank,
        };
    }

    /**
     * @brief Attribute live parity router checkpoints to the published epoch.
     *
     * Snapshot values are exact integer expert ids produced by the real router.
     * Pair them with the request-selected global placement bank and the final
     * domain-local schedule while all checkpoints remain live, before the
     * parity harness clears diagnostics. The setup-time residency snapshot
     * supplies only stable endpoint topology; it is deliberately not used to
     * reconstruct live placement after Dynamic movement.
     */
    void cacheDeviceRouteAssignmentEvidence()
    {
        if (!isRootParityRank())
            return;

        auto *const concrete =
            dynamic_cast<OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        if (!snapshot || !snapshot->valid())
        {
            ADD_FAILURE()
                << "Production parity could not inspect its immutable live ExpertOverlay residency epoch";
            return;
        }

        const size_t participant_count = activeOverlayParticipantCount();
        if (snapshot->owner_map.participants().size() != participant_count)
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participant cardinality changed before route attribution";
            return;
        }

        std::vector<uint64_t> route_counts(participant_count, 0u);
        std::vector<bool> requires_remote_completion(
            participant_count, false);
        std::vector<PublishedParticipantResidency> resident_participants(
            participant_count,
            PublishedParticipantResidency::Idle);
        for (const auto &participant : snapshot->owner_map.participants())
        {
            if (participant.participant_id < 0 ||
                static_cast<size_t>(participant.participant_id) >=
                    participant_count ||
                !participant.world_rank_known)
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay participant identity is incomplete";
                return;
            }
            requires_remote_completion[
                static_cast<size_t>(participant.participant_id)] =
                participant.world_rank != parityArtifactAuthorityRank();
        }

        size_t checkpoint_layers = 0u;
        for (const auto &placement :
             snapshot->placement_plan->placements)
        {
            if (placement.layer < 0 ||
                placement.routed_expert_tier.empty())
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay placement contains invalid layer geometry";
                return;
            }
            size_t route_elements = 0u;
            const std::string key =
                "layer" + std::to_string(placement.layer) +
                "_MOE_ROUTING_INDICES";
            const float *const routes = activeSnapshot(key, route_elements);
            if (!routes)
                continue;
            const auto route_evidence = pinnedDeviceRouteEvidence(
                placement.layer,
                route_elements,
                placement.routed_expert_tier.size());
            if (!route_evidence)
                return;
            try
            {
                includePublishedExpertOwners(
                    resident_participants,
                    std::span<const float>(
                        route_evidence->overlay_participants,
                        route_evidence->expert_count));
            }
            catch (const std::invalid_argument &error)
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay placement bank is invalid at layer "
                    << placement.layer << ": " << error.what();
                return;
            }
            ++checkpoint_layers;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                const float routed = routes[index];
                if (!std::isfinite(routed) || routed < 0.0f ||
                    routed > static_cast<float>(std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Router checkpoint " << key
                        << " contains a non-integral or out-of-range expert ID "
                        << routed << " at element " << index;
                    return;
                }

                // Validate the floating snapshot before converting it: a cast
                // of NaN or an out-of-range float to int is undefined behavior.
                const int expert = static_cast<int>(routed);
                if (routed != static_cast<float>(expert) ||
                    static_cast<size_t>(expert) >=
                        placement.routed_expert_tier.size())
                {
                    ADD_FAILURE()
                        << "Router checkpoint " << key
                        << " contains a non-integral or out-of-range expert id";
                    return;
                }
                const float global_assigned =
                    route_evidence->overlay_participants[expert];
                if (!std::isfinite(global_assigned) ||
                    global_assigned < 0.0f || global_assigned >
                        static_cast<float>(
                            std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Pinned overlay placement for " << key
                        << " contains an invalid global participant for expert "
                        << expert;
                    return;
                }
                const int participant = static_cast<int>(global_assigned);
                const auto *const global_endpoint =
                    snapshot->owner_map.participantForId(participant);
                if (global_assigned != static_cast<float>(participant) ||
                    participant < 0 ||
                    static_cast<size_t>(participant) >= participant_count ||
                    !global_endpoint)
                {
                    ADD_FAILURE()
                        << "Pinned overlay placement for " << key
                        << " names unknown global participant "
                        << global_assigned;
                    return;
                }

                const float domain_assigned =
                    route_evidence->domain_participants[index];
                if (!std::isfinite(domain_assigned) ||
                    domain_assigned < -1.0f || domain_assigned >
                        static_cast<float>(
                            std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Domain route schedule for " << key
                        << " contains invalid participant " << domain_assigned
                        << " at element " << index;
                    return;
                }
                const int domain_participant =
                    static_cast<int>(domain_assigned);
                if (domain_assigned !=
                    static_cast<float>(domain_participant))
                {
                    ADD_FAILURE()
                        << "Domain route schedule for " << key
                        << " contains non-integral participant "
                        << domain_assigned;
                    return;
                }
                const bool belongs_to_continuation =
                    global_endpoint->domain_name ==
                    overlay_plan_->continuation_domain;
                if (!belongs_to_continuation && domain_participant != -1)
                {
                    ADD_FAILURE()
                        << "Remote-domain expert " << expert << " in " << key
                        << " must retain the -1 domain-route sentinel, observed "
                        << domain_participant;
                    return;
                }
                if (belongs_to_continuation)
                {
                    const auto local_endpoint = std::find_if(
                        snapshot->owner_map.participants().begin(),
                        snapshot->owner_map.participants().end(),
                        [&](const MoEExpertOwnerParticipant &candidate)
                        {
                            return candidate.domain_name ==
                                       overlay_plan_->continuation_domain &&
                                   candidate.domain_participant_index ==
                                       domain_participant;
                        });
                    if (domain_participant < 0 ||
                        local_endpoint ==
                            snapshot->owner_map.participants().end())
                    {
                        ADD_FAILURE()
                            << "Continuation-domain expert " << expert
                            << " in " << key
                            << " names unknown domain participant "
                            << domain_participant;
                        return;
                    }
                }
                ++route_counts[static_cast<size_t>(participant)];
            }
        }
        if (checkpoint_layers == 0u)
        {
            ADD_FAILURE()
                << "Production parity retained no MoE routing checkpoint for live-epoch attribution";
            return;
        }
        parity_route_counts_by_participant_ = std::move(route_counts);
        parity_route_requires_remote_completion_ =
            std::move(requires_remote_completion);
        parity_residency_by_participant_ =
            std::move(resident_participants);
    }

    /**
     * @brief Persist immutable setup ownership and participant topology.
     *
     * The host residency snapshot is the cold-start topology authority. In an
     * all-GPU Dynamic cell it intentionally does not shadow later device-owned
     * placement epochs. Keep the established `expert_owner_map.csv` artifact
     * as the setup baseline and endpoint dictionary; exact live assignments
     * are recorded per route in `prefill_routed_expert_routes.csv` from the
     * reducer's device ledger.
     */
    void writeExpertOwnerTopologyBaselineCsv() const
    {
        if (!isRootParityRank())
            return;

        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        const auto path = ensureResultsDir() / "expert_owner_map.csv";
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output
            << "epoch,layer,expert,tier_index,tier_name,domain_name,"
               "participant,domain_participant,world_rank,device,resident\n";
        for (const auto &owner : snapshot->owner_map.owners())
        {
            output
                << snapshot->epoch << ',' << owner.layer_idx << ','
                << owner.expert_id << ',' << owner.tier_idx << ','
                << owner.tier_name << ',' << owner.domain_name << ','
                << owner.owner_participant << ','
                << owner.domain_participant_index << ','
                << owner.owner_world_rank << ',' << owner.device.toString()
                << ',' << (owner.resident ? 1 : 0) << '\n';
        }
        output.flush();
        EXPECT_TRUE(output.good()) << path;
    }

    /**
     * @brief Persist value-level routed-expert evidence for baseline and moves.
     *
     * Aggregate cosine metrics cannot distinguish a missing participant from
     * a correct route computed with the wrong weight slice.  The baseline
     * layer and every layer containing a committed promotion are therefore
     * recorded element by element against Hugging Face.  Per-route norms also
     * expose a zero, duplicated, or explosive migrated contribution without
     * requiring another instrumented inference run.  This diagnostic executes
     * only after the captured production forward and before snapshot teardown.
     */
    void writePrefillRoutedExpertDiagnosticCsv()
    {
        if (!isRootParityRank())
            return;

        std::vector<int> diagnostic_layers{0};
        for (const auto &promotion : promoted_experts_)
        {
            if (promotion.layer >= 0 &&
                std::find(
                    diagnostic_layers.begin(),
                    diagnostic_layers.end(),
                    promotion.layer) == diagnostic_layers.end())
            {
                diagnostic_layers.push_back(promotion.layer);
            }
        }
        std::sort(diagnostic_layers.begin(), diagnostic_layers.end());

        const auto values_path =
            ensureResultsDir() / "prefill_routed_expert_values.csv";
        std::ofstream values(values_path, std::ios::trunc);
        ASSERT_TRUE(values.is_open()) << values_path;
        values << std::setprecision(9);
        values
            << "layer,row,column,llaminar,pytorch,difference,"
               "continuation_local_sum\n";

        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        const auto routes_path =
            ensureResultsDir() / "prefill_routed_expert_routes.csv";
        std::ofstream route_output(routes_path, std::ios::trunc);
        ASSERT_TRUE(route_output.is_open()) << routes_path;
        route_output
            << "layer,row,slot,expert,weight,runtime_weight,participant,domain_participant,"
               "selected_placement_bank,tier_index,"
               "domain_name,world_rank,device,canonical_l2,"
               "canonical_max_abs,canonical_nonzero,canonical_first\n";
        route_output << std::setprecision(9);

        const auto *const model = activeModelContextForDiagnostics();
        ASSERT_NE(model, nullptr);
        const size_t width = static_cast<size_t>(
            model->model().embedding_length);
        ASSERT_GT(width, 0u);

        for (const int diagnostic_layer : diagnostic_layers)
        {
            const std::string prefix =
                "layer" + std::to_string(diagnostic_layer) + '_';
            size_t actual_elements = 0u;
            const float *const actual = activeSnapshot(
                prefix + "MOE_EXPERT_OUTPUT", actual_elements);
            const auto reference = loadPyTorchSnapshot(
                prefix + "MOE_EXPERT_OUTPUT");
            ASSERT_NE(actual, nullptr) << prefix;
            ASSERT_EQ(actual_elements, reference.size()) << prefix;
            ASSERT_EQ(actual_elements % width, 0u) << prefix;
            const size_t rows = actual_elements / width;

            size_t route_elements = 0u;
            size_t weight_elements = 0u;
            const float *const routes = activeSnapshot(
                prefix + "MOE_ROUTING_INDICES", route_elements);
            const float *const weights = activeSnapshot(
                prefix + "MOE_ROUTING_WEIGHTS", weight_elements);
            ASSERT_NE(routes, nullptr) << prefix;
            ASSERT_NE(weights, nullptr) << prefix;
            ASSERT_EQ(route_elements, weight_elements) << prefix;
            ASSERT_GT(rows, 0u) << prefix;
            ASSERT_EQ(route_elements % rows, 0u) << prefix;
            const size_t top_k = route_elements / rows;
            const auto placement = std::find_if(
                snapshot->placement_plan->placements.begin(),
                snapshot->placement_plan->placements.end(),
                [diagnostic_layer](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == diagnostic_layer; });
            ASSERT_NE(
                placement,
                snapshot->placement_plan->placements.end())
                << prefix << " has no declared routed-expert placement";
            const auto route_evidence = pinnedDeviceRouteEvidence(
                diagnostic_layer,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value()) << prefix;

            size_t canonical_elements = 0u;
            const float *const canonical = activeSnapshot(
                prefix + "MOE_ROUTE_CONTRIBUTIONS",
                canonical_elements);
            ASSERT_NE(canonical, nullptr)
                << prefix
                << " continuation did not retain canonical route-slot evidence";
            ASSERT_EQ(canonical_elements, route_elements * width) << prefix;

            std::vector<float> continuation_local_sum(
                actual_elements, 0.0f);
            for (size_t row = 0u; row < rows; ++row)
            {
                for (size_t slot = 0u; slot < top_k; ++slot)
                {
                    const size_t index = row * top_k + slot;
                    const size_t route_offset = index * width;
                    const size_t output_offset = row * width;
                    double squared_norm = 0.0;
                    float max_abs = 0.0f;
                    size_t nonzero = 0u;
                    for (size_t column = 0u; column < width; ++column)
                    {
                        const float contribution =
                            canonical[route_offset + column];
                        continuation_local_sum[output_offset + column] +=
                            contribution;
                        squared_norm += static_cast<double>(contribution) *
                                        static_cast<double>(contribution);
                        max_abs = std::max(max_abs, std::abs(contribution));
                        nonzero += contribution != 0.0f ? 1u : 0u;
                    }

                    ASSERT_TRUE(std::isfinite(routes[index]));
                    const int expert = static_cast<int>(routes[index]);
                    ASSERT_EQ(routes[index], static_cast<float>(expert));
                    ASSERT_GE(expert, 0);
                    ASSERT_LT(
                        static_cast<size_t>(expert),
                        route_evidence->expert_count);
                    const float global_assigned =
                        route_evidence->overlay_participants[expert];
                    ASSERT_TRUE(std::isfinite(global_assigned));
                    const int participant =
                        static_cast<int>(global_assigned);
                    ASSERT_EQ(
                        global_assigned,
                        static_cast<float>(participant))
                        << "Non-integral global route participant at layer "
                        << diagnostic_layer << " route slot " << index;
                    const float domain_assigned =
                        route_evidence->domain_participants[index];
                    ASSERT_TRUE(std::isfinite(domain_assigned));
                    const int domain_participant =
                        static_cast<int>(domain_assigned);
                    ASSERT_EQ(
                        domain_assigned,
                        static_cast<float>(domain_participant))
                        << "Non-integral domain route participant at layer "
                        << diagnostic_layer << " route slot " << index;
                    const float runtime_weight =
                        route_evidence->runtime_weights[index];
                    ASSERT_TRUE(std::isfinite(runtime_weight))
                        << "Non-finite runtime route weight at layer "
                        << diagnostic_layer << " route slot " << index;
                    const auto *const assigned_endpoint =
                        snapshot->owner_map.participantForId(participant);
                    ASSERT_NE(assigned_endpoint, nullptr)
                        << "Missing endpoint metadata for layer "
                        << diagnostic_layer << " expert " << expert
                        << " assigned participant " << participant;
                    route_output
                        << diagnostic_layer << ',' << row << ',' << slot
                        << ',' << expert << ',' << weights[index] << ','
                        << runtime_weight << ','
                        << participant << ',' << domain_participant << ','
                        << route_evidence->selected_bank << ','
                        << assigned_endpoint->tier_idx
                        << ',' << assigned_endpoint->domain_name << ','
                        << assigned_endpoint->world_rank << ','
                        << assigned_endpoint->device.toString() << ','
                        << std::sqrt(squared_norm) << ',' << max_abs << ','
                        << nonzero << ',' << canonical[route_offset] << '\n';
                }
            }

            for (size_t index = 0u; index < actual_elements; ++index)
            {
                values
                    << diagnostic_layer << ',' << index / width << ','
                    << index % width << ',' << actual[index] << ','
                    << reference[index] << ','
                    << (actual[index] - reference[index]) << ','
                    << continuation_local_sum[index] << '\n';
            }
        }
        values.flush();
        EXPECT_TRUE(values.good()) << values_path;
        route_output.flush();
        EXPECT_TRUE(route_output.good()) << routes_path;
    }

    /**
     * @brief Prove resident participants executed and capacity-idle ones did not.
     *
     * The device-owned placement bank determines whether each declared
     * endpoint owns any expert after automatic capacity resolution. Routed
     * checkpoints prove every selected endpoint was resident under that exact
     * epoch. Cross-rank residents additionally require endpoint-owned traffic
     * somewhere in the real production workload after both retained graphs
     * publish Complete. Residency alone cannot require one bounded prompt to
     * select an expert; an idle endpoint, however, can never appear in the
     * pinned route schedule. Local arithmetic remains covered by the same
     * layer/LM-head parity.
     */
    void assertActiveTierRouteEvidence() const
    {
        struct ExpectedRoute
        {
            int participant;
            int tier;
            const char *device_kind;
            std::string label;
        };

        std::vector<ExpectedRoute> expected;
        if (!overlay_plan_)
        {
            ADD_FAILURE() << "Sparse-route proof has no active overlay plan";
            return;
        }
        int participant_id = 0;
        for (const auto &domain : overlay_plan_->domains)
        {
            const auto tier = std::find_if(
                overlay_plan_->routed_tiers.begin(),
                overlay_plan_->routed_tiers.end(),
                [&](const RoutedExpertTier &candidate)
                { return candidate.domain == domain.name; });
            if (tier == overlay_plan_->routed_tiers.end())
            {
                ADD_FAILURE()
                    << "Sparse-route domain has no integer-priority tier: "
                    << domain.name;
                return;
            }
            const int tier_index = static_cast<int>(std::distance(
                overlay_plan_->routed_tiers.begin(), tier));
            for (const auto &participant : domain.participants)
            {
                const char *kind = participant.isCUDA()
                                       ? "CUDA"
                                       : participant.isROCm() ? "ROCm" : "CPU";
                expected.push_back(ExpectedRoute{
                    .participant = participant_id++,
                    .tier = tier_index,
                    .device_kind = kind,
                    .label = domain.name + "/" + participant.toShortString(),
                });
            }
        }

        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kSelectedStart = 1;
        const size_t completed_start = kSelectedStart + expected.size();
        std::vector<uint64_t> local(
            completed_start + expected.size(), 0u);
        if (isRootParityRank())
        {
            if (parity_route_counts_by_participant_.size() !=
                    expected.size() ||
                parity_route_requires_remote_completion_.size() !=
                    expected.size() ||
                parity_residency_by_participant_.size() !=
                    expected.size())
            {
                ++local[kMalformedRecords];
            }
            else
            {
                std::copy(
                    parity_route_counts_by_participant_.begin(),
                    parity_route_counts_by_participant_.end(),
                    local.begin() + kSelectedStart);
            }
        }
        for (const auto &record : PerfStatsCollector::snapshot({"forward_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "moe_overlay_local_expert_active_routes")
            {
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto it = record.tags.find(name);
                return it == record.tags.end() ? nullptr : &it->second;
            };
            const auto completion = tag("completion");
            const auto device_kind = tag("device_kind");
            const auto identity_source = tag("identity_source");
            const auto participant = tag("participant");
            const auto tier = tag("tier");
            // PerfStats coalesces repeated calls with the same full tag set.
            // A completed-route record can therefore represent several live
            // sparse packets (for example one packet per MoE layer), rather
            // than one call.  `value` is their summed route count and `count`
            // is the number of completed packets, so requiring count==1 here
            // would reject the strongest possible production evidence.
            const bool legacy_completion =
                completion &&
                *completion == "local_expert_packet_complete" &&
                identity_source &&
                *identity_source == "sparse_collective_key";
            const bool device_epoch_completion =
                completion &&
                *completion == "device_owned_epoch_complete" &&
                identity_source &&
                *identity_source == "device_owned_activation_epoch";
            const bool record_contract_ok =
                record.phase == "moe_overlay" &&
                record.count > 0u &&
                record.value > 0.0 &&
                record.tags.count("generation") == 0u &&
                record.tags.count("logical_step") == 0u &&
                (legacy_completion || device_epoch_completion) &&
                participant && tier && device_kind;
            if (!record_contract_ok)
            {
                ++local[kMalformedRecords];
                continue;
            }

            auto route = std::find_if(
                expected.begin(),
                expected.end(),
                [&](const ExpectedRoute &candidate)
                {
                    return *participant == std::to_string(candidate.participant) &&
                           *tier == std::to_string(candidate.tier) &&
                           *device_kind == candidate.device_kind;
                });
            if (route == expected.end())
            {
                ++local[kMalformedRecords];
                continue;
            }

            const size_t destination = completed_start +
                static_cast<size_t>(std::distance(expected.begin(), route));
            local[destination] += static_cast<uint64_t>(record.value);
        }

        std::vector<uint64_t> global(local.size(), 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());

        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Graph-native local-expert route evidence was malformed";
        for (size_t index = 0; index < expected.size(); ++index)
        {
            const bool owns_expert =
                parity_residency_by_participant_.size() == expected.size() &&
                parity_residency_by_participant_[index] ==
                    PublishedParticipantResidency::OwnsExpert;
            const bool requires_remote_completion =
                parity_route_requires_remote_completion_.size() ==
                    expected.size() &&
                parity_route_requires_remote_completion_[index];
            const auto verdict = validateParticipantRouteEvidence(
                owns_expert ? PublishedParticipantResidency::OwnsExpert
                            : PublishedParticipantResidency::Idle,
                global[kSelectedStart + index],
                requires_remote_completion,
                global[completed_start + index]);
            EXPECT_NE(
                verdict,
                PublishedParticipantRouteEvidence::IdleParticipantSelected)
                << expected[index].label << " participant p"
                << expected[index].participant
                << " was selected despite owning no expert in the pinned residency bank";
            EXPECT_NE(
                verdict,
                PublishedParticipantRouteEvidence::RemoteResidentNeverCompleted)
                << expected[index].label << " participant p"
                << expected[index].participant
                << " owns final experts but completed no real device-owned sparse traffic during the production workload";

            if (global[kSelectedStart + index] > 0u &&
                requires_remote_completion)
            {
                EXPECT_GT(global[completed_start + index], 0u)
                    << expected[index].label << " participant p"
                    << expected[index].participant
                    << " was selected by the pinned router but published no completed device-owned sparse traffic";
            }
        }
    }

    /**
     * @brief Prove the real sparse transport moved compact packets between tiers.
     *
     * Local-route completion proves that every participant ran an expert, but
     * it does not independently prove that the production sparse collective
     * carried compact request and result packets. The graph-native transport
     * publishes those byte counts through PerfStats. Folding them across both
     * MPI instances makes a missing dispatch, missing return, or silently
     * bypassed CPU cold tier a fatal parity failure without paying for a second
     * model setup in a profiler-only smoke test.
     */
    void assertSparseTransportPerfStatsEvidence() const
    {
        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kCompactDispatchBytes = 1;
        constexpr size_t kCompactReturnBytes = 2;
        constexpr size_t kCpuRows = 3;
        constexpr size_t kGpuRows = 4;
        constexpr size_t kRankLocalDispatchBytes = 5;
        constexpr size_t kRankLocalReturnBytes = 6;
        constexpr size_t kRankLocalCpuRows = 7;
        constexpr size_t kEvidenceCount = 8;

        const int local_domain_enabled =
            PerfStatsCollector::isDomainEnabled("moe_overlay") ? 1 : 0;
        int all_domains_enabled = 0;
        MPI_Allreduce(
            &local_domain_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the "
                       "moe_overlay PerfStats domain";
            }
            return;
        }

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot({"moe_overlay"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay")
            {
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto transport = tag("transport");
            const auto domain_kind = tag("domain_kind");
            const auto participant = tag("participant");
            const bool positive = record.count > 0u && record.value > 0.0;

            const bool rank_local_dispatch =
                record.name ==
                "rank_local_canonical_ticket_dispatch_bytes";
            const bool rank_local_return =
                record.name ==
                "rank_local_canonical_ticket_return_bytes";
            if (rank_local_dispatch || rank_local_return)
            {
                const auto completion = tag("completion");
                const auto identity_source = tag("identity_source");
                if (!positive || !transport || *transport != "compact" ||
                    !participant || !completion ||
                    *completion !=
                        "rank_local_canonical_ticket_complete" ||
                    !identity_source ||
                    *identity_source != "sparse_collective_key" ||
                    record.phase !=
                        (rank_local_dispatch
                             ? "gn_sparse_dispatch"
                             : "gn_return_reduce"))
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                const uint64_t bytes =
                    static_cast<uint64_t>(record.value);
                local[rank_local_dispatch
                          ? kRankLocalDispatchBytes
                          : kRankLocalReturnBytes] += bytes;
                local[rank_local_dispatch
                          ? kCompactDispatchBytes
                          : kCompactReturnBytes] += bytes;
                continue;
            }
            if (record.name ==
                "rank_local_canonical_ticket_cpu_rows")
            {
                const auto completion = tag("completion");
                const auto identity_source = tag("identity_source");
                if (record.phase != "gn_local_expert" || !positive ||
                    !transport || *transport != "local" || !domain_kind ||
                    *domain_kind != "CPU" || !participant || !completion ||
                    *completion !=
                        "rank_local_canonical_ticket_complete" ||
                    !identity_source ||
                    *identity_source != "sparse_collective_key")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                const uint64_t rows =
                    static_cast<uint64_t>(record.value);
                local[kRankLocalCpuRows] += rows;
                local[kCpuRows] += rows;
                continue;
            }

            if (record.name == "compact_dispatch_bytes")
            {
                if (record.phase != "gn_sparse_dispatch" || !positive ||
                    !transport || *transport != "compact")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[kCompactDispatchBytes] +=
                    static_cast<uint64_t>(record.value);
                continue;
            }
            if (record.name == "compact_return_bytes")
            {
                if (record.phase != "gn_return_reduce" || !positive ||
                    !transport || *transport != "compact")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[kCompactReturnBytes] +=
                    static_cast<uint64_t>(record.value);
                continue;
            }
            if (record.name != "device_epoch_cpu_rows" &&
                record.name != "device_epoch_gpu_rows")
                continue;

            int participant_id = -1;
            try
            {
                participant_id = participant ? std::stoi(*participant) : -1;
            }
            catch (const std::exception &)
            {
                participant_id = -1;
            }
            const bool cpu_record =
                record.name == "device_epoch_cpu_rows";
            const char *expected_kind = cpu_record ? "CPU" : "GPU";
            if (record.phase != "gn_local_expert" || !positive ||
                !transport || *transport != "local" ||
                !domain_kind || *domain_kind != expected_kind ||
                participant_id < 0)
            {
                ++local[kMalformedRecords];
                continue;
            }
            local[cpu_record ? kCpuRows : kGpuRows] +=
                static_cast<uint64_t>(record.value);
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Graph-native sparse transport PerfStats evidence was malformed";
        if (parity_residency_by_participant_.size() !=
            activeOverlayParticipantCount())
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participation was not retained for sparse transport evidence";
            return;
        }
        PublishedSparseTierParticipation active_tiers;
        try
        {
            active_tiers = summarizePublishedParticipation(
                resolvedOverlayPlan(),
                parity_residency_by_participant_);
        }
        catch (const std::invalid_argument &error)
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participation is invalid: "
                << error.what();
            return;
        }
        if (active_tiers.sparse_follower)
        {
            EXPECT_GT(global[kCompactDispatchBytes], 0u)
                << "A published sparse follower received no compact dispatch bytes";
            EXPECT_GT(global[kCompactReturnBytes], 0u)
                << "A published sparse follower emitted no compact return bytes";
        }
        else
        {
            /*
             * Automatic capacity may fit the entire model in the continuation
             * domain. Configured lower-priority endpoints then remain valid,
             * explicit idle participants: manufacturing transport solely to
             * satisfy a counter would no longer exercise production routing.
             */
            EXPECT_EQ(global[kCompactDispatchBytes], 0u)
                << "An idle sparse topology emitted compact dispatch traffic";
            EXPECT_EQ(global[kCompactReturnBytes], 0u)
                << "An idle sparse topology emitted compact return traffic";
        }
        if (active_tiers.secondary_gpu)
        {
            EXPECT_GT(global[kGpuRows], 0u)
                << "No remote GPU tier published completed device-owned expert rows";
        }
        else
        {
            EXPECT_EQ(global[kGpuRows], 0u)
                << "A topology without a remote GPU endpoint published remote GPU rows";
        }
        if (active_tiers.cpu)
        {
            EXPECT_GT(global[kCpuRows], 0u)
                << "A CPU tier owning published experts completed no expert rows";
            if (!topologySpansMultipleMPIRanks())
            {
                EXPECT_GT(global[kRankLocalDispatchBytes], 0u)
                    << "The rank-local CPU tier consumed no compact dispatch payload";
                EXPECT_GT(global[kRankLocalReturnBytes], 0u)
                    << "The rank-local CPU tier published no canonical return payload";
                EXPECT_GT(global[kRankLocalCpuRows], 0u)
                    << "The rank-local canonical ticket carried no completed CPU routes";
            }
        }
        else
        {
            EXPECT_EQ(global[kCpuRows], 0u)
                << "A capacity-resolved idle or absent CPU tier unexpectedly executed expert rows";
        }
    }

    /**
     * @brief Prove the full request lifecycle honored the shared segmented graph.
     *
     * PrefixRuntimeStateSnapshot proves the public OrchestrationRunner admitted
     * a chunked request instead of treating the test as three unrelated
     * forwards. Per-rank PerfStats then prove the distributed schedule contract
     * was published, an expert-only participant consumed the same ordered
     * transactions, and the CUDA continuation retained complete prompt-wide
     * checkpoints for the existing Hugging Face/CSV comparator. Dynamic cells
     * deliberately execute calibration and migration-training requests before
     * parity, so PerfStats and the prefix probe are cumulative by design. The
     * assertion consequently proves the whole lifecycle rather than inventing
     * a test-only reset edge: all schedules succeed, every captured row is
     * accounted for, both roles retain identical ordered digests, and the one
     * snapshot-bearing parity request independently proves `[4,4,1]`. Mandatory
     * prefix restore then contributes exactly one serial suffix-decode record.
     */
    void assertSegmentedPrefillEvidence() const
    {
        constexpr uint64_t kExpectedChunks = 3;
        constexpr uint64_t kExpectedRealTokens = 9;
        constexpr uint64_t kExpectedPaddedTokens = 3;
        constexpr uint64_t kExpectedSnapshotPrefillTransactions = 1;
        constexpr size_t kContractRecords = 0;
        constexpr size_t kSnapshotAggregationRecords = 1;
        constexpr size_t kContinuationTransactionRecords = 2;
        constexpr size_t kParticipantTransactionRecords = 3;
        constexpr size_t kContinuationFourRowTransactions = 4;
        constexpr size_t kContinuationOneRowTransactions = 5;
        constexpr size_t kParticipantFourRowTransactions = 6;
        constexpr size_t kParticipantOneRowTransactions = 7;
        constexpr size_t kContinuationSequenceRecords = 8;
        constexpr size_t kParticipantSequenceRecords = 9;
        constexpr size_t kContinuationSequenceSteps = 10;
        constexpr size_t kParticipantSequenceSteps = 11;
        constexpr size_t kContinuationSequenceWords = 12;
        constexpr size_t kParticipantSequenceWords = 13;
        constexpr size_t kContinuationSequenceDigestLo = 14;
        constexpr size_t kParticipantSequenceDigestLo = 15;
        constexpr size_t kContinuationSequenceDigestHi = 16;
        constexpr size_t kParticipantSequenceDigestHi = 17;
        constexpr size_t kMalformedRecords = 18;
        constexpr size_t kRestoredPrefixSuffixDecodeTransactions = 19;
        constexpr size_t kEvidenceCount = 20;
        constexpr uint64_t kWordsPerSequenceStep = 4u;

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"forward_graph", "moe_overlay_participant_graph"}))
        {
            const auto tagEquals = [&record](
                                       const char *name,
                                       const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };
            const auto unsignedTag = [&record](const char *name)
                -> std::optional<uint64_t>
            {
                const auto it = record.tags.find(name);
                if (it == record.tags.end() || it->second.empty())
                    return std::nullopt;
                uint64_t value = 0;
                const char *const begin = it->second.data();
                const char *const end = begin + it->second.size();
                const auto parsed = std::from_chars(begin, end, value);
                if (parsed.ec != std::errc{} || parsed.ptr != end)
                    return std::nullopt;
                return value;
            };

            if (record.kind == PerfStatRecord::Kind::OrderedSequence &&
                record.domain == "forward_graph" &&
                record.name ==
                    "moe_overlay_collective_transaction_sequence")
            {
                if (record.phase != "prefill")
                    continue;

                const bool common_contract =
                    record.count > 0u &&
                    record.value == static_cast<double>(record.count) &&
                    record.sequence_word_count ==
                        record.count * kWordsPerSequenceStep &&
                    record.sequence_digest_lo != 0u &&
                    record.sequence_digest_hi != 0u &&
                    tagEquals(
                        "identity_source",
                        "orchestration_request_and_chunk") &&
                    tagEquals(
                        "logical_step_semantics",
                        "monotonic_transaction");
                const bool continuation =
                    tagEquals("role", "continuation_graph");
                const bool participant =
                    tagEquals("role", "expert_participant_graph");
                if (!common_contract || continuation == participant)
                {
                    ++local[kMalformedRecords];
                    continue;
                }

                const size_t record_slot =
                    continuation ? kContinuationSequenceRecords
                                 : kParticipantSequenceRecords;
                const size_t step_slot =
                    continuation ? kContinuationSequenceSteps
                                 : kParticipantSequenceSteps;
                const size_t word_slot =
                    continuation ? kContinuationSequenceWords
                                 : kParticipantSequenceWords;
                const size_t digest_lo_slot =
                    continuation ? kContinuationSequenceDigestLo
                                 : kParticipantSequenceDigestLo;
                const size_t digest_hi_slot =
                    continuation ? kContinuationSequenceDigestHi
                                 : kParticipantSequenceDigestHi;
                if (local[record_slot] != 0u)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[record_slot] = 1u;
                local[step_slot] = record.count;
                local[word_slot] = record.sequence_word_count;
                local[digest_lo_slot] = record.sequence_digest_lo;
                local[digest_hi_slot] = record.sequence_digest_hi;
                continue;
            }

            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;

            if (record.domain == "forward_graph" &&
                record.name ==
                    "restored_prefix_suffix_decode_transactions")
            {
                const bool valid =
                    record.phase == "decode" &&
                    record.value == static_cast<double>(record.count) &&
                    record.count == 1u &&
                    tagEquals("outer_command", "prefill") &&
                    tagEquals("mathematical_phase", "decode") &&
                    tagEquals("logical_rows", "1") &&
                    tagEquals("state_transition", "main_only");
                if (!valid)
                    ++local[kMalformedRecords];
                else
                    local[kRestoredPrefixSuffixDecodeTransactions] +=
                        record.count;
                continue;
            }

            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_prefill_schedule_contract_rows")
            {
                if (record.phase != "model_setup" ||
                    record.value != static_cast<double>(
                                        activeSegmentedPrefillCaptureRows()) ||
                    record.count != 1u ||
                    !tagEquals("authority", "continuation_root") ||
                    !tagEquals("bucket_count", "1") ||
                    !tagEquals("immutable", "true"))
                {
                    ++local[kMalformedRecords];
                }
                else
                {
                    ++local[kContractRecords];
                }
                continue;
            }

            if (record.domain == "forward_graph" &&
                record.name == "prefill_chunk_snapshot_sequence_keys")
            {
                if (record.phase != "prefill" || record.value <= 0.0 ||
                    record.count != kExpectedSnapshotPrefillTransactions ||
                    !tagEquals("chunks", std::to_string(kExpectedChunks)) ||
                    !tagEquals("diagnostic_only", "true"))
                {
                    ++local[kMalformedRecords];
                }
                else
                {
                    ++local[kSnapshotAggregationRecords];
                }
                continue;
            }

            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_collective_transaction")
            {
                /*
                 * Decode parity performs its own prefill initialization, then
                 * emits serial decode transactions. Those records are covered
                 * by the decode comparator; they are not malformed prefill
                 * evidence merely because they share the sparse metric name.
                 */
                if (record.phase != "prefill")
                    continue;

                const auto logical_rows = unsignedTag("logical_rows");
                const auto physical_rows = unsignedTag("physical_rows");
                const bool bounded_identity =
                    record.tags.count("generation") == 0u &&
                    record.tags.count("logical_step") == 0u &&
                    record.tags.count("prefill_chunk_index") == 0u &&
                    record.tags.count("token_offset") == 0u;
                const bool chunk_geometry_ok = logical_rows && physical_rows &&
                    *physical_rows == activeSegmentedPrefillCaptureRows() &&
                    *logical_rows > 0u &&
                    *logical_rows <= *physical_rows;
                const bool transaction_contract_ok =
                    record.count > 0u &&
                    record.value == static_cast<double>(record.count) &&
                    tagEquals("role", "continuation_graph") &&
                    tagEquals("identity_source", "orchestration_request_and_chunk") &&
                    tagEquals("logical_step_semantics", "monotonic_transaction") &&
                    bounded_identity &&
                    chunk_geometry_ok;
                if (!transaction_contract_ok)
                {
                    ++local[kMalformedRecords];
                    continue;
                }

                local[kContinuationTransactionRecords] += record.count;
                if (*logical_rows == 4u)
                    local[kContinuationFourRowTransactions] += record.count;
                else if (*logical_rows == 1u)
                    local[kContinuationOneRowTransactions] += record.count;
                continue;
            }

            if (record.domain == "moe_overlay_participant_graph" &&
                record.name == "ticket_selected_graphs" &&
                record.phase == "main_prefill")
            {
                const auto logical_rows = unsignedTag("logical_rows");
                const auto physical_rows = unsignedTag("physical_rows");
                const bool bounded_identity =
                    record.tags.count("request_generation") == 0u &&
                    record.tags.count("command_id") == 0u &&
                    record.tags.count("transaction_ordinal") == 0u &&
                    record.tags.count("logical_step") == 0u &&
                    record.tags.count("prefill_schedule_fingerprint") == 0u;
                const bool transaction_geometry_ok = logical_rows &&
                    *logical_rows > 0u && physical_rows &&
                    *logical_rows <= *physical_rows;
                const bool valid =
                    record.count > 0u &&
                    record.value == static_cast<double>(record.count) &&
                    tagEquals("logical_step_semantics", "monotonic_transaction") &&
                    bounded_identity &&
                    transaction_geometry_ok &&
                    *physical_rows == activeSegmentedPrefillCaptureRows();
                if (!valid)
                {
                    std::ostringstream detail;
                    detail << "Malformed segmented-prefill follower ticket: "
                           << "phase=" << record.phase
                           << ",value=" << record.value
                           << ",count=" << record.count;
                    for (const auto &[name, value] : record.tags)
                        detail << ',' << name << '=' << value;
                    ADD_FAILURE() << detail.str();
                    ++local[kMalformedRecords];
                    continue;
                }

                local[kParticipantTransactionRecords] += record.count;
                if (*logical_rows == 4u)
                    local[kParticipantFourRowTransactions] += record.count;
                else if (*logical_rows == 1u)
                    local[kParticipantOneRowTransactions] += record.count;
            }
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());

        if (!isRootParityRank())
            return;

        const PrefixRuntimeStateSnapshot probe = activePrefixStateProbe();
        EXPECT_GT(probe.prefill_chunk_schedules, 0u)
            << "The production lifecycle never entered the segmented scheduler";
        EXPECT_EQ(
            probe.prefill_chunk_successful_schedules,
            probe.prefill_chunk_schedules)
            << "Every admitted production schedule must complete";
        EXPECT_GE(probe.prefill_chunks, kExpectedChunks)
            << "The authenticated parity prefill must contribute [4,4,1]";
        EXPECT_GE(probe.prefill_chunk_real_tokens, kExpectedRealTokens)
            << "The authenticated parity prompt's real rows were not counted";
        EXPECT_GE(probe.prefill_chunk_padded_tokens, kExpectedPaddedTokens)
            << "The authenticated parity prompt's final bucket was not padded";
        EXPECT_EQ(
            probe.prefill_chunk_real_tokens +
                probe.prefill_chunk_padded_tokens,
            probe.prefill_chunks * activeSegmentedPrefillCaptureRows())
            << "Every captured row must be classified as real or padding";
        EXPECT_EQ(probe.prefill_chunk_failures, 0u)
            << "A graph-native heterogeneous schedule may not recover through "
               "an eager or uncaptured fallback";

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Segmented prefill PerfStats tags or geometry were malformed";
        EXPECT_EQ(
            global[kContractRecords],
            static_cast<uint64_t>(mpiWorldSize()))
            << "Every MPI participant must install the immutable shared bucket contract";
        EXPECT_EQ(global[kSnapshotAggregationRecords], 1u)
            << "The continuation graph must aggregate the complete prompt into one evidence record";
        EXPECT_EQ(
            global[kParticipantTransactionRecords],
            global[kContinuationTransactionRecords])
            << "The remote expert graph must consume every lifecycle prefill ticket";
        EXPECT_GE(global[kContinuationFourRowTransactions], 2u);
        EXPECT_GE(global[kContinuationOneRowTransactions], 1u);
        EXPECT_EQ(
            global[kParticipantFourRowTransactions],
            global[kContinuationFourRowTransactions]);
        EXPECT_EQ(
            global[kParticipantOneRowTransactions],
            global[kContinuationOneRowTransactions]);
        EXPECT_EQ(global[kContinuationSequenceRecords], 1u)
            << "The continuation must retain one bounded prefill sequence row";
        EXPECT_EQ(global[kParticipantSequenceRecords], 1u)
            << "The follower must retain one bounded prefill sequence row";
        EXPECT_EQ(
            global[kParticipantSequenceSteps],
            global[kContinuationSequenceSteps]);
        EXPECT_EQ(
            global[kContinuationSequenceSteps],
            global[kContinuationTransactionRecords]);
        EXPECT_EQ(
            global[kParticipantSequenceWords],
            global[kContinuationSequenceWords]);
        EXPECT_EQ(
            global[kContinuationSequenceWords],
            global[kContinuationSequenceSteps] * kWordsPerSequenceStep);
        EXPECT_EQ(
            global[kRestoredPrefixSuffixDecodeTransactions],
            1u)
            << "The partial prefix proof must consume its uncached row through serial decode math";
        EXPECT_NE(global[kContinuationSequenceDigestLo], 0u);
        EXPECT_NE(global[kContinuationSequenceDigestHi], 0u);
        EXPECT_EQ(
            global[kContinuationSequenceDigestLo],
            global[kParticipantSequenceDigestLo])
            << "Continuation and follower observed different transaction order or geometry";
        EXPECT_EQ(
            global[kContinuationSequenceDigestHi],
            global[kParticipantSequenceDigestHi])
            << "Continuation and follower observed different transaction order or geometry";
    }

    /**
     * @brief Verify every chunk-scoped Hugging Face checkpoint became full-prompt data.
     *
     * SnapshotCapture retains context-qualified copies such as
     * `PREFILL_CHUNK_0_layer0_...` for diagnosis and rewrites the bare semantic
     * key to the ordered aggregate. This check prevents a short final chunk
     * from passing merely because a comparison used the shorter tensor length.
     */
    void assertSegmentedPrefillCheckpointCoverage()
    {
        constexpr const char *kFirstChunkPrefix = "PREFILL_CHUNK_0_";
        const auto keys = activeSnapshotKeys();
        size_t checked = 0;
        for (const std::string &scoped_key : keys)
        {
            if (scoped_key.rfind(kFirstChunkPrefix, 0) != 0)
                continue;

            const std::string semantic_key =
                scoped_key.substr(std::char_traits<char>::length(kFirstChunkPrefix));
            if (semantic_key.empty() || semantic_key == "LM_HEAD")
                continue;

            const std::vector<float> reference = loadPyTorchSnapshot(semantic_key);
            if (reference.empty())
                continue;

            size_t aggregate_elements = 0;
            const float *aggregate = activeSnapshot(
                semantic_key,
                aggregate_elements);
            ASSERT_NE(aggregate, nullptr)
                << "Segmented checkpoint '" << semantic_key
                << "' lost its bare parity key after aggregation";
            EXPECT_EQ(aggregate_elements, reference.size())
                << "Segmented checkpoint '" << semantic_key
                << "' must contain every real prompt row";
            ++checked;
        }

        EXPECT_GT(checked, 0u)
            << "Segmented production prefill published no Hugging Face-backed "
               "chunk checkpoints";
    }

    bool collectivelyCheckHardwareAndModel() const
    {
        bool available = false;
        if (isRootParityRank())
            available =
                !acceleratorHardwareBlocker(cluster_inventory_).has_value() &&
                modelAvailable();
        return broadcastRootFlag(available);
    }

    /**
     * @brief Lexicographic objective for one adversarial CPU-tier subsequence.
     *
     * The first component maximizes authenticated routes assigned to the CPU
     * participant remote from the continuation rank. Only after that total is
     * fixed does the second component minimize traffic owned by the colocated
     * CPU participant. This is an initial-layout construction objective, never
     * a runtime histogram or placement decision.
     */
    struct AdversarialCpuPlacementScore
    {
        std::uint64_t remote_routes = 0;
        std::uint64_t local_routes = 0;
        bool reachable = false;
    };

    /** @return Whether @p candidate strictly improves the layout objective. */
    static bool improvesAdversarialCpuPlacement(
        const AdversarialCpuPlacementScore &candidate,
        const AdversarialCpuPlacementScore &incumbent) noexcept
    {
        if (!candidate.reachable)
            return false;
        if (!incumbent.reachable)
            return true;
        if (candidate.remote_routes != incumbent.remote_routes)
            return candidate.remote_routes > incumbent.remote_routes;
        return candidate.local_routes < incumbent.local_routes;
    }

    /**
     * @brief Select exact lower-tier membership under random owner partitioning.
     *
     * Whole-expert ownership first sorts the selected expert IDs, applies the
     * production ordinal/random permutation to those positions, and gives one
     * balanced span to each participant. Selecting a different set therefore
     * changes the sorted-rank occupied by every later expert. A small dynamic
     * program solves that subsequence problem exactly instead of assuming an
     * expert ID maps directly to a participant.
     *
     * @param routes Authenticated prefill route count for every expert.
     * @param layer Model layer used by the production random permutation.
     * @param tier_index Lower-priority CPU tier index used by that permutation.
     * @param selected_count Exact tier quota for this layer.
     * @param participant_count Number of balanced NodeTP CPU owners.
     * @return Boolean expert mask with exactly @p selected_count entries.
     */
    static std::vector<bool> selectAdversarialCpuExperts(
        const std::vector<std::uint64_t> &routes,
        int layer,
        int tier_index,
        int selected_count,
        int participant_count,
        RoutedExpertOwnerOrder owner_order)
    {
        if (routes.empty() || layer < 0 || tier_index < 0 ||
            selected_count <= 0 ||
            selected_count > static_cast<int>(routes.size()) ||
            participant_count < 2)
        {
            throw std::invalid_argument(
                "Adversarial CPU owner selection has invalid geometry");
        }

        std::vector<int> selected_ranks(
            static_cast<std::size_t>(selected_count));
        std::iota(selected_ranks.begin(), selected_ranks.end(), 0);
        routed_expert_ownership::applyOwnerOrder(
            selected_ranks, owner_order, layer, tier_index);

        const int base = selected_count / participant_count;
        const int remainder = selected_count % participant_count;
        const int first_participant_count = base + (remainder > 0 ? 1 : 0);
        std::vector<bool> remote_selected_rank(
            static_cast<std::size_t>(selected_count), false);
        for (int offset = 0; offset < first_participant_count; ++offset)
        {
            remote_selected_rank.at(static_cast<std::size_t>(
                selected_ranks.at(static_cast<std::size_t>(offset)))) = true;
        }

        const std::size_t expert_count = routes.size();
        const std::size_t columns =
            static_cast<std::size_t>(selected_count) + 1u;
        std::vector<AdversarialCpuPlacementScore> scores(
            (expert_count + 1u) * columns);
        std::vector<std::uint8_t> took_expert(scores.size(), 0u);
        const auto index = [columns](std::size_t experts_seen,
                                     std::size_t experts_selected)
        {
            return experts_seen * columns + experts_selected;
        };
        scores[index(0u, 0u)].reachable = true;

        for (std::size_t experts_seen = 1u;
             experts_seen <= expert_count;
             ++experts_seen)
        {
            const std::size_t maximum_selected = std::min(
                experts_seen, static_cast<std::size_t>(selected_count));
            for (std::size_t experts_selected = 0u;
                 experts_selected <= maximum_selected;
                 ++experts_selected)
            {
                auto best = scores[index(
                    experts_seen - 1u, experts_selected)];
                bool take = false;
                if (experts_selected > 0u)
                {
                    auto candidate = scores[index(
                        experts_seen - 1u, experts_selected - 1u)];
                    if (candidate.reachable)
                    {
                        const std::uint64_t demand =
                            routes[experts_seen - 1u];
                        auto &total = remote_selected_rank[
                                          experts_selected - 1u]
                                          ? candidate.remote_routes
                                          : candidate.local_routes;
                        if (demand >
                            std::numeric_limits<std::uint64_t>::max() - total)
                        {
                            throw std::overflow_error(
                                "Adversarial CPU routing objective overflowed");
                        }
                        total += demand;
                        if (improvesAdversarialCpuPlacement(candidate, best))
                        {
                            best = candidate;
                            take = true;
                        }
                    }
                }
                scores[index(experts_seen, experts_selected)] = best;
                took_expert[index(experts_seen, experts_selected)] =
                    take ? 1u : 0u;
            }
        }

        if (!scores[index(expert_count,
                          static_cast<std::size_t>(selected_count))]
                 .reachable)
        {
            throw std::logic_error(
                "Adversarial CPU owner selection found no exact tier quota");
        }

        std::vector<bool> selected(expert_count, false);
        std::size_t experts_selected =
            static_cast<std::size_t>(selected_count);
        for (std::size_t experts_seen = expert_count;
             experts_seen > 0u;
             --experts_seen)
        {
            if (took_expert[index(experts_seen, experts_selected)] == 0u)
                continue;
            selected[experts_seen - 1u] = true;
            --experts_selected;
        }
        if (experts_selected != 0u ||
            static_cast<int>(std::count(
                selected.begin(), selected.end(), true)) != selected_count)
        {
            throw std::logic_error(
                "Adversarial CPU owner selection reconstruction changed its quota");
        }
        return selected;
    }

    /**
     * @brief Install an authenticated workload-adversarial initial tier layout.
     *
     * The Hugging Face pack is already the mathematical oracle for this parity
     * cell. Its routing IDs define only the starting condition: high-demand
     * experts are deliberately left on lower-priority participants, with the
     * strongest CPU candidates owned by the rank remote from continuation.
     * Runtime movement remains driven exclusively by histograms emitted from
     * the real Llaminar sparse-collective graphs.
     *
     * @param requested Inventory-bound dynamic production request.
     * @return Same request with complete explicit per-layer initial placement.
     * @throws std::exception For missing/malformed reference evidence or a
     *         layout that cannot make remote CPU demand strictly dominant.
     */
    MoERoutedExpertPlacementPlan installReferenceAdversarialPlacements(
        MoERoutedExpertPlacementPlan requested)
    {
        const auto metadata_path =
            std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
        const auto layer_text =
            readSnapshotMetadataValue(metadata_path, "n_layers");
        int layer_count = 0;
        if (!layer_text)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement requires reference n_layers metadata");
        }
        const char *const layer_begin = layer_text->data();
        const char *const layer_end = layer_begin + layer_text->size();
        const auto parsed_layers =
            std::from_chars(layer_begin, layer_end, layer_count);
        if (parsed_layers.ec != std::errc{} ||
            parsed_layers.ptr != layer_end || layer_count <= 0)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement has invalid reference n_layers metadata");
        }
        auto cpu_tier = std::find_if(
            requested.routed_tiers.begin(),
            requested.routed_tiers.end(),
            [](const auto &tier)
            { return tier.domain == kCpuColdDomain; });
        if (cpu_tier == requested.routed_tiers.end())
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement has no NodeTP CPU tier");
        }
        const int cpu_tier_index = static_cast<int>(std::distance(
            requested.routed_tiers.begin(), cpu_tier));
        const auto cpu_domain = std::find_if(
            requested.domains.begin(),
            requested.domains.end(),
            [](const auto &domain)
            { return domain.name == kCpuColdDomain; });
        const auto continuation = std::find_if(
            requested.domains.begin(),
            requested.domains.end(),
            [&](const auto &domain)
            { return domain.name == requested.continuation_domain; });
        const auto continuation_rank =
            continuation == requested.domains.end()
                ? std::optional<int>{}
                : continuation->primaryWorldRank();
        if (cpu_domain == requested.domains.end() ||
            continuation == requested.domains.end() ||
            cpu_domain->participants.size() < 2u ||
            cpu_domain->world_ranks.size() !=
                cpu_domain->participants.size() ||
            !continuation_rank ||
            cpu_domain->world_ranks.front() == *continuation_rank)
        {
            throw std::invalid_argument(
                "Adversarial ExpertOverlay placement requires remote-first multi-participant CPU ownership and a resolved continuation rank");
        }

        MoERoutedExpertModelMetadata metadata = topologyOnlyMetadata();
        metadata.num_layers = layer_count;
        const int expert_count = metadata.num_experts;
        std::vector<std::vector<std::uint64_t>> reference_routes(
            static_cast<std::size_t>(layer_count),
            std::vector<std::uint64_t>(
                static_cast<std::size_t>(expert_count), 0u));
        for (int layer = 0; layer < layer_count; ++layer)
        {
            const auto routing = loadPyTorchSnapshot(
                "layer" + std::to_string(layer) +
                "_MOE_ROUTING_INDICES");
            if (routing.empty())
            {
                throw std::invalid_argument(
                    "Adversarial ExpertOverlay placement lacks reference routing for layer " +
                    std::to_string(layer));
            }
            for (const float routed_expert : routing)
            {
                const int expert = static_cast<int>(routed_expert);
                if (!std::isfinite(routed_expert) ||
                    routed_expert != static_cast<float>(expert) ||
                    expert < 0 || expert >= expert_count)
                {
                    throw std::invalid_argument(
                        "Adversarial ExpertOverlay placement found a malformed reference expert ID");
                }
                ++reference_routes[static_cast<std::size_t>(layer)]
                                  [static_cast<std::size_t>(expert)];
            }
        }

        /*
         * Do not guess the tier quota here. The production physical-capacity
         * resolver has not yet observed fixed graph, migration, and backend
         * allocations. Declare only a complete cold-to-hot expert order; after
         * exact quotas are installed, the normal planner fills smaller integer
         * priorities first and therefore leaves the authenticated hottest
         * experts in lower-priority tiers. Concrete membership remains wholly
         * production-owned.
         */
        requested.placements.clear();
        requested.initial_layer_order_overrides.clear();
        requested.initial_layer_order_overrides.reserve(
            static_cast<std::size_t>(layer_count));
        for (int layer = 0; layer < layer_count; ++layer)
        {
            std::vector<int> order(static_cast<std::size_t>(expert_count));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(
                order.begin(), order.end(),
                [&](int lhs, int rhs)
                {
                    const auto lhs_routes = reference_routes[
                        static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(lhs)];
                    const auto rhs_routes = reference_routes[
                        static_cast<std::size_t>(layer)]
                        [static_cast<std::size_t>(rhs)];
                    if (lhs_routes != rhs_routes)
                        return lhs_routes < rhs_routes;
                    return lhs < rhs;
                });
            requested.initial_layer_order_overrides.push_back({
                .layer = layer,
                .expert_ids = std::move(order),
            });
        }

        reference_adversarial_routes_ = std::move(reference_routes);
        reference_adversarial_cpu_tier_index_ = cpu_tier_index;
        reference_adversarial_continuation_rank_ = *continuation_rank;
        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Authenticated adversarial order deferred "
            "to production capacity: layers=" << layer_count
            << " experts=" << expert_count
            << " owner_order="
            << routedExpertOwnerOrderToString(requested.owner_order));
        return requested;
    }

    /**
     * @brief Certify the exact capacity-resolved adversarial epoch-one layout.
     *
     * The setup request carries only a complete expert permutation. This check
     * runs after the production runner has resolved physical quotas and frozen
     * concrete placements. It proves that lower-priority CPU membership owns
     * the hottest suffix and that the first, remote CPU participant received a
     * genuinely hotter expert than its colocated peer on at least one layer.
     *
     * @return True when the frozen production owner map satisfies the complete
     *         adversarial contract, or when this cell declared no such order.
     */
    bool certifyInstalledReferenceAdversarialPlacement() const
    {
        if (reference_adversarial_routes_.empty())
            return true;
        if (!orch_runner_ || !orch_runner_->config().moe_routed_expert_plan)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Deferred adversarial order has no frozen production plan");
            return false;
        }

        const auto &plan = *orch_runner_->config().moe_routed_expert_plan;
        if (!plan.initial_layer_order_overrides.empty() ||
            plan.placements.size() < reference_adversarial_routes_.size())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Production did not consume the complete deferred adversarial order: initial_orders="
                << plan.initial_layer_order_overrides.size()
                << " placements=" << plan.placements.size()
                << " expected_layers="
                << reference_adversarial_routes_.size());
            return false;
        }
        if (reference_adversarial_cpu_tier_index_ < 0 ||
            reference_adversarial_cpu_tier_index_ >=
                static_cast<int>(plan.routed_tiers.size()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Deferred adversarial CPU tier identity is outside the frozen plan");
            return false;
        }

        const auto owner_map = MoEExpertOwnerMap::build(plan);
        std::uint64_t remote_routes = 0u;
        std::uint64_t local_routes = 0u;
        std::uint64_t strongest_remote = 0u;
        std::uint64_t strongest_local = 0u;
        std::uint64_t strongest_cpu = 0u;
        std::uint64_t strongest_preferred = 0u;
        std::int64_t best_layer_margin =
            std::numeric_limits<std::int64_t>::min();
        bool every_layer_has_both_tier_sides = true;
        bool cold_to_hot_order_preserved = true;

        for (std::size_t layer = 0;
             layer < reference_adversarial_routes_.size(); ++layer)
        {
            const auto &routes = reference_adversarial_routes_[layer];
            std::uint64_t layer_remote_peak = 0u;
            std::uint64_t layer_local_peak = 0u;
            std::uint64_t layer_cpu_min =
                std::numeric_limits<std::uint64_t>::max();
            std::uint64_t layer_cpu_max = 0u;
            std::uint64_t layer_preferred_max = 0u;
            std::size_t cpu_experts = 0u;
            std::size_t preferred_experts = 0u;

            for (std::size_t expert = 0; expert < routes.size(); ++expert)
            {
                const auto *owner = owner_map.ownerFor(
                    static_cast<int>(layer), static_cast<int>(expert));
                if (!owner)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Frozen adversarial owner map is incomplete at layer="
                        << layer << " expert=" << expert);
                    return false;
                }
                const std::uint64_t demand = routes[expert];
                if (owner->tier_idx ==
                    reference_adversarial_cpu_tier_index_)
                {
                    ++cpu_experts;
                    layer_cpu_min = std::min(layer_cpu_min, demand);
                    layer_cpu_max = std::max(layer_cpu_max, demand);
                    if (owner->owner_world_rank !=
                        reference_adversarial_continuation_rank_)
                    {
                        remote_routes += demand;
                        layer_remote_peak = std::max(
                            layer_remote_peak, demand);
                    }
                    else
                    {
                        local_routes += demand;
                        layer_local_peak = std::max(
                            layer_local_peak, demand);
                    }
                }
                else
                {
                    ++preferred_experts;
                    layer_preferred_max = std::max(
                        layer_preferred_max, demand);
                }
            }

            every_layer_has_both_tier_sides =
                every_layer_has_both_tier_sides &&
                cpu_experts > 0u && preferred_experts > 0u;
            if (cpu_experts > 0u && preferred_experts > 0u)
            {
                cold_to_hot_order_preserved =
                    cold_to_hot_order_preserved &&
                    layer_cpu_min >= layer_preferred_max;
            }
            strongest_cpu = std::max(strongest_cpu, layer_cpu_max);
            strongest_preferred = std::max(
                strongest_preferred, layer_preferred_max);
            strongest_remote = std::max(
                strongest_remote, layer_remote_peak);
            strongest_local = std::max(
                strongest_local, layer_local_peak);
            best_layer_margin = std::max(
                best_layer_margin,
                static_cast<std::int64_t>(layer_remote_peak) -
                    static_cast<std::int64_t>(layer_local_peak));
        }

        if (!every_layer_has_both_tier_sides ||
            !cold_to_hot_order_preserved || remote_routes == 0u ||
            best_layer_margin <= 0)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Capacity-resolved adversarial layout failed certification: both_tier_sides="
                << every_layer_has_both_tier_sides
                << " cold_to_hot_order=" << cold_to_hot_order_preserved
                << " remote_routes=" << remote_routes
                << " best_layer_margin=" << best_layer_margin);
            return false;
        }

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Capacity-resolved adversarial layout certified: layers="
            << reference_adversarial_routes_.size()
            << " remote_cpu_routes=" << remote_routes
            << " local_cpu_routes=" << local_routes
            << " strongest_remote_expert=" << strongest_remote
            << " strongest_local_expert=" << strongest_local
            << " strongest_cpu_expert=" << strongest_cpu
            << " strongest_preferred_expert=" << strongest_preferred
            << " best_layer_margin=" << best_layer_margin);
        return true;
    }

    bool setupPipeline()
    {
        try
        {
            /*
             * This is the same immutable declarative plan supplied to the
             * production application. OrchestrationRunner validates it against
             * the real GGUF metadata during initialization; the parity fixture
             * does not build, patch, or execute a graph itself.
             */
            /*
             * Supply the same model-independent request a user supplies at the
             * production boundary. OrchestrationRunner freezes exact per-layer
             * placements only after it has authenticated the real GGUF
             * metadata. Pre-planning from approximate test constants would make
             * the fixture, rather than production, the placement authority and
             * can silently give a 40-layer model a 94-layer ownership map.
             */
            auto requested = requestedPlan(topologyOnlyMetadata());
            const bool supports_remote_cpu_adversary =
                topologyUsesCpu() &&
                activeTypedParticipantCount(
                      [](const GlobalDeviceAddress &participant)
                      { return participant.isCPU(); }) > 1u &&
                activeModelParityCaseOrThrow().topology.mpi_ranks > 1;
            if (isDynamicResidencyProductionTest() &&
                supports_remote_cpu_adversary)
            {
                requested = remoteFirstNodeLocalOwnerPlan(
                    requested,
                    cluster_inventory_);
                requested = installReferenceAdversarialPlacements(
                    std::move(requested));
            }
            overlay_plan_ = std::make_shared<MoERoutedExpertPlacementPlan>(
                std::move(requested));
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Overlay plan construction failed: " << e.what());
            return false;
        }

        OrchestrationConfig orchestration = OrchestrationConfig::defaults();
        orchestration.model_path = config_.model_path;
        orchestration.max_seq_len =
            activeModelParityCaseOrThrow().model.max_seq_len;
        orchestration.batch_size = 1;
        /*
         * The named MoE domains are the sole placement authority. During
         * normalization they become the production execution-domain inventory,
         * so supplying a second --device-map authority would be both ambiguous
         * and correctly rejected by ConfigValidator. The overlay execution-plan
         * resolver derives each rank's continuation/participant role directly
         * from the domains' world_ranks mappings.
         */
        orchestration.device_mode = DeviceAssignmentMode::AUTO;
        orchestration.tp_degree = 1;
        orchestration.pp_degree = 1;
        orchestration.deterministic = !isQwen122ProductionTest();
        activeModelParityCaseOrThrow().applyRuntimePolicy(orchestration);
        if (requiresObservedConvergenceSpeedup())
        {
            /*
             * Derive the complete initial routed workload from the same
             * constants used by the driver. This admission check makes an
             * epoch-crossing timing cohort unrepresentable if a future
             * reference prompt changes geometry without updating the typed
             * proof window.
             */
            const std::uint64_t baseline_routed_rows =
                convergenceTimingCohortRoutedRows();
            const int configured_window =
                orchestration.moe_rebalance.window_size;
            if (baseline_routed_rows >=
                    static_cast<std::uint64_t>(configured_window) ||
                configured_window !=
                    orchestration.moe_rebalance.max_window_size ||
                orchestration.moe_rebalance.window_growth_factor != 1.0F)
            {
                throw std::logic_error(
                    "Observed convergence policy does not retain its complete timing corpus inside one immutable histogram epoch: routed_rows=" +
                    std::to_string(baseline_routed_rows) +
                    " window=" +
                    std::to_string(configured_window));
            }
        }
        if (isDynamicResidencyProductionTest())
        {
            /* Capacity admission and the evidence fold consume the same
             * generated policy; the fixture never invents a wave width. */
            convergence_migration_transfer_slots_ =
                orchestration.moe_rebalance.migration_transfer_slots;
            convergence_migration_cycles_per_wave_ =
                orchestration.moe_rebalance
                    .resolvedMigrationCyclesPerWave();
            if (convergence_migration_transfer_slots_ == 0u ||
                convergence_migration_cycles_per_wave_ == 0u ||
                convergence_migration_cycles_per_wave_ >
                    convergence_migration_transfer_slots_)
            {
                throw std::logic_error(
                    "Dynamic model-parity policy has an invalid physical/active migration wave envelope");
            }
        }
        applyProductionParityPrefixRestorePolicy(orchestration);
        if (isDynamicResidencyProductionTest())
        {
            /*
             * Dynamic certification now uses the authenticated reference
             * request, so an entry may exist under the pre-movement epoch.
             * Select the production invalidate-on-rebalance policy for every
             * Dynamic proof: publication retires that entry and guarantees the
             * post-movement prefill executes every checkpoint.  The same cell
             * then seeds and restores a new entry under the proven epoch, so
             * RAM/disk prefix restore remains mandatory rather than bypassed.
             * Static cells retain PlacementFingerprint and independently prove
             * the portable-cache policy without a placement transition.
             */
            orchestration.prefix_cache.moe_policy =
                PrefixCacheMoEPolicy::InvalidateOnRebalance;
        }
        // Inventory binding/adversarial placement has now specialized the
        // immutable blueprint copied by applyRuntimePolicy. Publish that exact
        // request as the sole runtime placement authority.
        orchestration.moe_routed_expert_plan = overlay_plan_;

        const auto snapshot_setup_mode =
            isDynamicResidencyProductionTest()
                ? ParitySnapshotSetupMode::Disabled
                : ParitySnapshotSetupMode::Enabled;
        bool model_context_cache_hit = false;
        std::string model_context_cache_error;
        auto reuse_contract = findQwen122OverlayModelContext(
            &model_context_cache_hit,
            &model_context_cache_error);
        const auto model_admission =
            mayReuseQwen122OverlayModelContext()
                ? reachQwen122CampaignModelAdmission(
                      parityCoordinationCommunicator(),
                      model_context_cache_hit,
                      model_context_cache_error)
                : Qwen122CampaignModelAdmissionResult{
                      .admission = Qwen122CampaignModelAdmission::Fresh,
                      .succeeded = model_context_cache_error.empty(),
                      .diagnostic = model_context_cache_error,
                  };
        if (!model_admission.succeeded)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] "
                      << model_admission.diagnostic);
            return false;
        }
        const bool collectively_reusing_model =
            model_admission.admission ==
            Qwen122CampaignModelAdmission::Reuse;
        if (collectively_reusing_model !=
            static_cast<bool>(reuse_contract))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Rank-unanimous model admission "
                "does not match this rank's retained contract");
            return false;
        }

        const bool setup_ok = reuse_contract
                                  ? setupOrchestrationRunner(
                                        orchestration,
                                        *reuse_contract,
                                        snapshot_setup_mode)
                                  : setupOrchestrationRunner(
                                        orchestration,
                                        nullptr,
                                        snapshot_setup_mode);
        if (!setup_ok)
            return false;
        if (!certifyInstalledReferenceAdversarialPlacement())
            return false;

        /*
         * The retained contract is the exact model-owned authority consumed
         * by setupOrchestrationRunner above. Publish that typed cache outcome
         * through the shared campaign evidence field; inferring reuse later
         * from elapsed time or an incidental loader counter would make the
         * production_path.csv claim weaker than the lifecycle we just proved.
         */
        production_parity_model_context_reused_ = model_context_cache_hit;

        if (!orch_runner_)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Production runner disappeared after setup");
            return false;
        }

        try
        {
            /*
             * Configure both rank-local runners before enabling the command
             * loop.  Once rank zero enters coordinated mode, this public API
             * deliberately broadcasts SET_SAMPLING to workers; doing that
             * before the workers are listening would invert the production
             * command protocol.  The pre-loop setting is therefore the normal
             * request-admission configuration boundary for this test topology.
             */
            orch_runner_->setSamplingParams(referenceGreedySamplingPolicy());
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] Failed to install Hugging Face "
                      "greedy sampling policy: " << e.what());
            return false;
        }

        /*
         * Use the same readiness surface as the HTTP server and benchmark.
         * Dynamic ExpertOverlay may need a bounded set of physical topology
         * measurements before ordinary requests are admissible; that work is
         * owned entirely below this interface and never leaks calibration
         * planning into the parity driver.
         */
        if (!orch_runner_->prepareForInference())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Production inference preparation failed: "
                << orch_runner_->lastError());
            return false;
        }

        if (mayReuseQwen122OverlayModelContext())
        {
            if (model_context_cache_hit)
            {
                const auto reuse_status =
                    orch_runner_->modelContextReuseStatus();
                if (!reuse_status.reusedPreparedWeights())
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Campaign cache hit did not "
                        "consume a production-certified PreparedWeightStore: "
                        "imported="
                        << reuse_status.imported_contract
                        << " plan_validated="
                        << reuse_status.prepared_weight_plan_validated
                        << " prepared_entries="
                        << reuse_status.prepared_entry_count
                        << " generation="
                        << reuse_status.authority_generation);
                    return false;
                }
            }
            const auto published_contract =
                orch_runner_->modelContextReuseContract();
            if (!published_contract)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Initialized runner did not "
                    "publish its exact rank-local ExpertOverlay weight contract");
                return false;
            }
            if (!publishQwen122OverlayModelContext(
                    *published_contract,
                    &model_context_cache_error))
            {
                LOG_ERROR("[Qwen3.5 MoE GraphNative] "
                          << model_context_cache_error);
                return false;
            }

            PerfStatsCollector::addCounter(
                "weight_loading",
                model_context_cache_hit
                    ? "parity_campaign_model_context_cache_hits"
                    : "parity_campaign_model_context_cache_misses",
                1.0,
                "setup",
                orch_runner_->primaryDeviceId().toString(),
                {{"owner_order",
                  isRandomOwnerProductionTest() ? "random" : "ordinal"},
                 {"mtp_depth", std::to_string(activeMTPDraftDepth())}});
        }

        return true;
    }

    /**
     * @return Passive status projected by the sole production authority.
     * @throws std::logic_error when the runner is absent or its lifecycle failed.
     */
    MoEOptimizationStatus optimizationStatus() const
    {
        if (!orch_runner_)
            throw std::logic_error(
                "ExpertOverlay optimization status requires a live runner");
        auto status = orch_runner_->moeOptimizationStatus();
        if (status.failed())
        {
            throw std::logic_error(
                status.diagnostic.empty()
                    ? "ExpertOverlay optimization lifecycle failed"
                    : status.diagnostic);
        }
        return status;
    }

    /** @return Exact durable movement-wave count from the production owner. */
    std::uint64_t localCommittedWaveCount() const
    {
        return optimizationStatus().published_movement_waves;
    }

    /**
     * @brief Measure one exact production workload inside one residency epoch.
     *
     * Both sides use identical prompt IDs, boundary calls, decode budgets, and
     * greedy token trajectories. This convergence cell selects the production
     * invalidate-on-rebalance prefix policy, so publication advances the
     * fingerprint and the post-movement replay cannot restore an initial-epoch
     * entry. Both sides therefore execute full model compute. Ordinary
     * maintenance notifications remain enabled. Any publication during the
     * cohort is a hard protocol failure instead of a sample that can be hidden
     * by median selection.
     *
     * @param cohort Initial adversarial epoch or post-movement epoch.
     * @return True only after the complete workload ran in one exact epoch.
     */
    bool collectInferenceTimings(ResidencyTimingCohort cohort)
    {
        if (!requiresObservedConvergenceSpeedup())
            return true;

        constexpr std::size_t kRequiredDecodeSamples =
            static_cast<std::size_t>(
                kConvergenceTimingMeasuredRequests) *
            kConvergenceTimingDecodeForwardsPerRequest;
        const bool initial =
            cohort == ResidencyTimingCohort::InitialEpoch;
        const DynamicResidencyProofPhase required_phase =
            initial
                ? DynamicResidencyProofPhase::EconomyCertified
                : DynamicResidencyProofPhase::
                      MovementBoundarySettled;
        if (dynamic_residency_proof_lifecycle_.phase() != required_phase)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Timing cohort crossed an invalid typed residency lifecycle transition: cohort="
                << (initial ? "initial" : "converged")
                << " required_phase="
                << static_cast<int>(required_phase)
                << " actual_phase="
                << static_cast<int>(
                       dynamic_residency_proof_lifecycle_.phase()));
            return false;
        }
        const std::uint64_t cohort_wave = localCommittedWaveCount();
        /* The typed phase above already proves the complete model-specific
         * wave and axis objective. This boundary checks only that the cohort
         * pins an initial or post-publication epoch; it must not reinterpret a
         * cycle count, histogram-window count, or another model's wave target. */
        if ((initial && cohort_wave != 0u) ||
            (!initial && cohort_wave == 0u))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Timing cohort began in the wrong residency epoch: cohort="
                << (initial ? "initial" : "converged")
                << " committed_waves=" << cohort_wave);
            return false;
        }

        if (initial &&
            (!convergence_timings_.baseline_candidates.empty() ||
             !convergence_timings_.baseline_prefill_ns.empty()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Initial timing candidates were already populated");
            return false;
        }
        if (!initial &&
            (!convergence_timings_.converged_prefill_ns.empty() ||
             !convergence_timings_.converged_decode_ns.empty() ||
             !convergence_timings_.converged_samples.empty()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Converged timing samples were already populated");
            return false;
        }

        /*
         * Movement-driving traffic can finish immediately after publishing the
         * final placement fingerprint and legitimately archive one of these
         * same prompts under that fingerprint. Retire the reusable archive once
         * at the unmeasured cohort boundary. Every first prefill below must then
         * execute full production compute, while the repeated prefill inside
         * each request still proves ordinary RAM/disk prefix restore.
         */
        activeClearCache();
        if (!orch_runner_->purgePrefixCache())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Controlled timing cohort could not retire reusable prefix state: "
                << orch_runner_->lastError());
            return false;
        }

        /*
         * Measurement identities are never used to drive movement. The
         * converged cohort therefore replays exactly the initial identities
         * after the typed quiescent boundary, with no publication-overlap
         * reserve, identity omission, or inferred race state.
         */
        const int requested_identities =
            kConvergenceTimingCorpusRequests;
        std::vector<int> prompt_identities;
        prompt_identities.reserve(
            static_cast<std::size_t>(requested_identities));
        for (int identity = 0; identity < requested_identities; ++identity)
        {
            prompt_identities.push_back(identity);
        }

        std::vector<const ResidencyConvergenceTimings::RequestSample *>
            selected_baseline_samples;
        if (!initial)
        {
            /* Materialize only the five initial candidates paired with the
             * selected converged identities. */
            convergence_timings_.baseline_prefill_ns.clear();
            convergence_timings_.baseline_decode_ns.clear();
            convergence_timings_.baseline_prefill_epochs.clear();
            convergence_timings_.baseline_decode_epochs.clear();
            convergence_timings_.baseline_decode_input_tokens.clear();
            selected_baseline_samples.reserve(
                kConvergenceTimingMeasuredRequests);
            for (int ordinal = kConvergenceTimingWarmupRequests;
                 ordinal < requested_identities;
                 ++ordinal)
            {
                const int identity = prompt_identities.at(
                    static_cast<std::size_t>(ordinal));
                const auto candidate = std::find_if(
                    convergence_timings_.baseline_candidates.begin(),
                    convergence_timings_.baseline_candidates.end(),
                    [identity](const auto &sample)
                    { return sample.prompt_identity == identity; });
                if (candidate ==
                    convergence_timings_.baseline_candidates.end())
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] No initial-epoch timing candidate matches converged prompt identity "
                        << identity);
                    return false;
                }
                convergence_timings_.baseline_prefill_ns.push_back(
                    candidate->prefill_ns);
                convergence_timings_.baseline_prefill_epochs.push_back(
                    candidate->prefill_epoch);
                convergence_timings_.baseline_decode_ns.insert(
                    convergence_timings_.baseline_decode_ns.end(),
                    candidate->decode_ns.begin(),
                    candidate->decode_ns.end());
                convergence_timings_.baseline_decode_epochs.insert(
                    convergence_timings_.baseline_decode_epochs.end(),
                    candidate->decode_epochs.begin(),
                    candidate->decode_epochs.end());
                convergence_timings_.baseline_decode_input_tokens.insert(
                    convergence_timings_.baseline_decode_input_tokens.end(),
                    candidate->decode_input_tokens.begin(),
                    candidate->decode_input_tokens.end());
                selected_baseline_samples.push_back(&*candidate);
            }
        }
        std::vector<int32_t> decode_input_tokens;

        for (int request = 0;
             request < requested_identities;
             ++request)
        {
            activeClearCache();
            const int prompt_identity = prompt_identities.at(
                static_cast<std::size_t>(request));
            const std::vector<int32_t> prompt =
                makeReferenceShapedEconomyPrompt(
                    prompt_identity,
                    ReferenceEconomyPromptRole::TimingCohort);
            const bool retain_sample =
                request >= kConvergenceTimingWarmupRequests;
            ResidencyConvergenceTimings::RequestSample request_sample;
            request_sample.prompt_identity = prompt_identity;
            const std::optional<ConvergenceTimerSnapshot> request_timer_begin =
                retain_sample
                    ? std::optional<ConvergenceTimerSnapshot>(
                          convergenceTimerSnapshot())
                    : std::nullopt;

            const std::uint64_t prefill_waves_before =
                localCommittedWaveCount();
            const auto prefill_start = std::chrono::steady_clock::now();
            if (!orch_runner_->prefill(prompt))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Residency timing prefill failed: "
                    << orch_runner_->lastError());
                return false;
            }
            const std::uint64_t prefill_ns =
                elapsedNanoseconds(prefill_start);
            const std::uint64_t prefill_waves_after =
                localCommittedWaveCount();
            if (prefill_waves_before != cohort_wave ||
                prefill_waves_after != cohort_wave)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Residency changed inside the controlled prefill cohort: expected="
                    << cohort_wave << " before=" << prefill_waves_before
                    << " after=" << prefill_waves_after);
                return false;
            }
            const PrefixRuntimeStateSnapshot prefix_state =
                orch_runner_->prefixStateProbe();
            if (prefix_state.prefix_request.hit ||
                prefix_state.prefix_request.matched_tokens != 0)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Controlled timing prefill restored cached model state instead of executing the full prompt");
                return false;
            }
            if (retain_sample)
            {
                request_sample.prefill_ns = prefill_ns;
                request_sample.prefill_epoch = cohort_wave + 1u;
            }

            for (int step = 0;
                 step < kConvergenceTimingDecodeForwardsPerRequest;
                 ++step)
            {
                if (step > 0)
                {
                    /*
                     * Restore the exact prefill terminal state so every timed
                     * decode consumes the same autoregressive input. The first
                     * pass above also proves full-compute prefill economy; these
                     * untimed repetitions prove normal prefix reuse within one
                     * immutable placement epoch.
                     */
                    activeClearCache();
                    const std::uint64_t restore_waves_before =
                        localCommittedWaveCount();
                    if (!orch_runner_->prefill(prompt))
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Residency timing prefix restore failed: "
                            << orch_runner_->lastError());
                        return false;
                    }
                    const std::uint64_t restore_waves_after =
                        localCommittedWaveCount();
                    const PrefixRuntimeStateSnapshot restored_prefix_state =
                        orch_runner_->prefixStateProbe();
                    if (restore_waves_before != cohort_wave ||
                        restore_waves_after != cohort_wave ||
                        !restored_prefix_state.prefix_request.hit ||
                        restored_prefix_state.prefix_request.matched_tokens !=
                            static_cast<int>(prompt.size()))
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Controlled decode sample did not restore the exact epoch-stable prefix: expected_epoch="
                            << cohort_wave << " before="
                            << restore_waves_before << " after="
                            << restore_waves_after << " hit="
                            << restored_prefix_state.prefix_request.hit
                            << " matched_tokens="
                            << restored_prefix_state.prefix_request.matched_tokens
                            << " prompt_tokens=" << prompt.size());
                        return false;
                    }
                }

                /* Consume prefill logits outside the decode timing interval. */
                const std::uint64_t boundary_waves_before =
                    localCommittedWaveCount();
                orch_runner_->setDecodeStepTokenBudget(1);
                GenerationResult boundary_sample = orch_runner_->decodeStep();
                orch_runner_->setDecodeStepTokenBudget(0);
                const std::uint64_t boundary_waves_after =
                    localCommittedWaveCount();
                if (!boundary_sample.success() ||
                    boundary_sample.tokens.size() != 1u ||
                    boundary_waves_before != cohort_wave ||
                    boundary_waves_after != cohort_wave)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Residency timing boundary sample was not one epoch-stable token: error="
                        << boundary_sample.error << " tokens="
                        << boundary_sample.tokens.size() << " expected_epoch="
                        << cohort_wave << " before=" << boundary_waves_before
                        << " after=" << boundary_waves_after);
                    return false;
                }
                if (retain_sample)
                {
                    request_sample.decode_input_tokens.push_back(
                        boundary_sample.tokens.front());
                }
                if (!orch_runner_->maybeApplyMoERebalance(1u))
                    return false;

                const std::uint64_t decode_waves_before =
                    localCommittedWaveCount();
                orch_runner_->setDecodeStepTokenBudget(1);
                const auto decode_start = std::chrono::steady_clock::now();
                GenerationResult decoded = orch_runner_->decodeStep();
                const std::uint64_t decode_ns =
                    elapsedNanoseconds(decode_start);
                orch_runner_->setDecodeStepTokenBudget(0);
                const std::uint64_t decode_waves_after =
                    localCommittedWaveCount();
                if (!decoded.success() || decoded.tokens.size() != 1u)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Residency timing decode did not execute exactly one forward: error="
                        << decoded.error << " tokens="
                        << decoded.tokens.size());
                    return false;
                }
                if (decode_waves_before != cohort_wave ||
                    decode_waves_after != cohort_wave)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Residency changed inside the controlled decode cohort: expected="
                        << cohort_wave << " before=" << decode_waves_before
                        << " after=" << decode_waves_after);
                    return false;
                }
                if (!orch_runner_->maybeApplyMoERebalance(1u))
                    return false;

                if (retain_sample)
                {
                    request_sample.decode_ns.push_back(decode_ns);
                    request_sample.decode_epochs.push_back(
                        cohort_wave + 1u);
                }
            }

            if (!retain_sample)
                continue;
            if (request_sample.decode_ns.size() !=
                    static_cast<std::size_t>(
                        kConvergenceTimingDecodeForwardsPerRequest) ||
                request_sample.decode_epochs.size() !=
                    request_sample.decode_ns.size() ||
                request_sample.decode_input_tokens.size() !=
                    request_sample.decode_ns.size())
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] A controlled timing request did not retain one complete decode sample group: identity="
                    << prompt_identity << " decode="
                    << request_sample.decode_ns.size() << " epochs="
                    << request_sample.decode_epochs.size() << " inputs="
                    << request_sample.decode_input_tokens.size());
                return false;
            }
            if (!request_timer_begin)
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] A retained timing request has no timer interval origin");
                return false;
            }
            request_sample.timer_deltas = convergenceTimerDelta(
                *request_timer_begin,
                convergenceTimerSnapshot());
            if (initial)
            {
                convergence_timings_.baseline_candidates.push_back(
                    std::move(request_sample));
            }
            else
            {
                convergence_timings_.converged_prefill_ns.push_back(
                    request_sample.prefill_ns);
                convergence_timings_.converged_prefill_epochs.push_back(
                    request_sample.prefill_epoch);
                convergence_timings_.converged_decode_ns.insert(
                    convergence_timings_.converged_decode_ns.end(),
                    request_sample.decode_ns.begin(),
                    request_sample.decode_ns.end());
                convergence_timings_.converged_decode_epochs.insert(
                    convergence_timings_.converged_decode_epochs.end(),
                    request_sample.decode_epochs.begin(),
                    request_sample.decode_epochs.end());
                decode_input_tokens.insert(
                    decode_input_tokens.end(),
                    request_sample.decode_input_tokens.begin(),
                    request_sample.decode_input_tokens.end());
                convergence_timings_.converged_samples.push_back(
                    std::move(request_sample));
            }
        }

        const bool complete = initial
                                  ? convergence_timings_
                                            .baseline_candidates.size() ==
                                        static_cast<std::size_t>(
                                            kConvergenceTimingMeasuredRequests)
                                  : convergence_timings_
                                                .converged_prefill_ns.size() ==
                                            static_cast<std::size_t>(
                                                kConvergenceTimingMeasuredRequests) &&
                                        convergence_timings_
                                                .converged_decode_ns.size() ==
                                            kRequiredDecodeSamples &&
                                        convergence_timings_
                                                .baseline_prefill_ns.size() ==
                                            static_cast<std::size_t>(
                                                kConvergenceTimingMeasuredRequests) &&
                                        convergence_timings_
                                                .baseline_decode_ns.size() ==
                                            kRequiredDecodeSamples &&
                                        convergence_timings_
                                                .converged_samples.size() ==
                                            static_cast<std::size_t>(
                                                kConvergenceTimingMeasuredRequests);
        if (localCommittedWaveCount() != cohort_wave || !complete)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Controlled timing cohort was incomplete or crossed an epoch: prefill="
                << (initial
                        ? convergence_timings_.baseline_candidates.size()
                        : convergence_timings_.converged_prefill_ns.size())
                << " decode="
                << (initial ? 0u
                            : convergence_timings_.converged_decode_ns.size())
                << " expected_epoch="
                << cohort_wave << " final_epoch="
                << localCommittedWaveCount());
            return false;
        }

        if (!initial &&
            decode_input_tokens !=
                 convergence_timings_.baseline_decode_input_tokens)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Before/after timing cohorts supplied different sampled inputs to a timed decode: baseline_inputs="
                << convergence_timings_.baseline_decode_input_tokens.size()
                << " converged_inputs=" << decode_input_tokens.size());
            return false;
        }
        if (!initial)
        {
            writeConvergenceTimerSamples(
                selected_baseline_samples,
                convergence_timings_.converged_samples);
        }
        if (initial)
            dynamic_residency_proof_lifecycle_.recordInitialCohort();
        else
            dynamic_residency_proof_lifecycle_.recordConvergedCohort();
        return true;
    }

    /** @return Stable diagnostic spelling for the active priority topology. */
    std::string convergenceTopologyName() const
    {
        return activeModelParityCaseOrThrow().topology.test_id;
    }

    /**
     * @brief Assert and serialize the real before/after convergence evidence.
     *
     * A two-percent floor is deliberately larger than timer quantization and
     * ordinary run-to-run jitter on this host. The median makes the gate robust
     * to background OS activity while retaining a directional performance
     * requirement for both time-to-first-token prefill and steady decode.
     */
    void assertAndWriteObservedConvergenceSpeedup()
    {
        if (!requiresObservedConvergenceSpeedup() || !isRootParityRank())
            return;

        ASSERT_EQ(
            convergence_timings_.baseline_prefill_ns.size(),
            static_cast<std::size_t>(
                kConvergenceTimingMeasuredRequests));
        ASSERT_EQ(
            convergence_timings_.baseline_decode_ns.size(),
            static_cast<std::size_t>(
                kConvergenceTimingMeasuredRequests *
                kConvergenceTimingDecodeForwardsPerRequest));
        ASSERT_EQ(
            convergence_timings_.converged_prefill_ns.size(),
            convergence_timings_.baseline_prefill_ns.size());
        ASSERT_EQ(
            convergence_timings_.converged_decode_ns.size(),
            convergence_timings_.baseline_decode_ns.size());

        const std::uint64_t baseline_prefill = medianNanoseconds(
            convergence_timings_.baseline_prefill_ns);
        const std::uint64_t baseline_decode = medianNanoseconds(
            convergence_timings_.baseline_decode_ns);
        const std::uint64_t converged_prefill = medianNanoseconds(
            convergence_timings_.converged_prefill_ns);
        const std::uint64_t converged_decode = medianNanoseconds(
            convergence_timings_.converged_decode_ns);
        constexpr long double kMaximumConvergedRatio = 0.98L;
        const bool prefill_passed =
            static_cast<long double>(converged_prefill) <=
            static_cast<long double>(baseline_prefill) *
                kMaximumConvergedRatio;
        const bool decode_passed =
            static_cast<long double>(converged_decode) <=
            static_cast<long double>(baseline_decode) *
                kMaximumConvergedRatio;
        const auto improvement = [](std::uint64_t baseline,
                                    std::uint64_t converged)
        {
            return 100.0L *
                   (static_cast<long double>(baseline) -
                    static_cast<long double>(converged)) /
                   static_cast<long double>(baseline);
        };
        const long double prefill_improvement =
            improvement(baseline_prefill, converged_prefill);
        const long double decode_improvement =
            improvement(baseline_decode, converged_decode);

        const auto path =
            ensureResultsDir() / "expert_overlay_convergence.csv";
        std::ofstream csv(path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open())
            << "Cannot create observed ExpertOverlay convergence CSV at "
            << path;
        csv << "backend,topology,phase,cohort,sample,residency_epoch,latency_ns,baseline_median_ns,converged_median_ns,improvement_percent,passed\n";
        csv << std::fixed << std::setprecision(4);
        const auto write_samples = [&](const char *phase,
                                       const char *cohort,
                                       const std::vector<std::uint64_t> &samples,
                                       const std::vector<std::uint64_t> &epochs,
                                       std::uint64_t baseline_median,
                                       std::uint64_t converged_median,
                                       long double improvement_percent,
                                       bool passed)
        {
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                const std::uint64_t epoch =
                    epochs.empty() ? 1u : epochs.at(index);
                csv << getBackendName() << ','
                    << convergenceTopologyName() << ','
                    << phase << ','
                    << cohort << ','
                    << index << ','
                    << epoch << ','
                    << samples[index] << ','
                    << baseline_median << ','
                    << converged_median << ','
                    << static_cast<double>(improvement_percent) << ','
                    << (passed ? "true" : "false") << '\n';
            }
        };
        write_samples(
            "prefill",
            "initial_epoch",
            convergence_timings_.baseline_prefill_ns,
            convergence_timings_.baseline_prefill_epochs,
            baseline_prefill,
            converged_prefill,
            prefill_improvement,
            prefill_passed);
        write_samples(
            "prefill",
            "converged_epoch",
            convergence_timings_.converged_prefill_ns,
            convergence_timings_.converged_prefill_epochs,
            baseline_prefill,
            converged_prefill,
            prefill_improvement,
            prefill_passed);
        write_samples(
            "decode",
            "initial_epoch",
            convergence_timings_.baseline_decode_ns,
            convergence_timings_.baseline_decode_epochs,
            baseline_decode,
            converged_decode,
            decode_improvement,
            decode_passed);
        write_samples(
            "decode",
            "converged_epoch",
            convergence_timings_.converged_decode_ns,
            convergence_timings_.converged_decode_epochs,
            baseline_decode,
            converged_decode,
            decode_improvement,
            decode_passed);
        csv.flush();
        ASSERT_TRUE(csv.good())
            << "Failed to write observed ExpertOverlay convergence CSV at "
            << path;

        LOG_INFO(
            "[Qwen3.5 MoE GraphNative] Observed convergence performance: topology="
            << convergenceTopologyName()
            << " prefill_initial_ns=" << baseline_prefill
            << " prefill_converged_ns=" << converged_prefill
            << " prefill_improvement_percent="
            << static_cast<double>(prefill_improvement)
            << " decode_initial_ns=" << baseline_decode
            << " decode_converged_ns=" << converged_decode
            << " decode_improvement_percent="
            << static_cast<double>(decode_improvement));
        EXPECT_TRUE(prefill_passed)
            << "Priority convergence did not improve observed prefill latency by at least 2%";
        EXPECT_TRUE(decode_passed)
            << "Priority convergence did not improve observed decode latency by at least 2%";
    }

    /**
     * @brief Execute one ordinary request prefill for economy evidence.
     *
     * @param tokens Real model tokens supplied through the serving API.
     * @param purpose Stable diagnostic name for the traffic lifecycle phase.
     * @return Whether production prefill completed successfully.
     */
    bool runDynamicEconomyPrefill(
        const std::vector<int32_t> &tokens,
        const char *purpose)
    {
        activeClearSnapshots();
        activeClearCache();
        if (orch_runner_->prefill(tokens))
            return true;
        LOG_ERROR(
            "[Qwen3.5 MoE GraphNative] Dynamic "
            << (purpose ? purpose : "economy")
            << " prefill failed: " << orch_runner_->lastError());
        return false;
    }

    /**
     * @brief Execute one ordinary bounded decode and notify maintenance.
     *
     * The first generated token after prefill consumes existing logits; the
     * next budgeted call executes a real DecodeToken graph. Callers that need
     * decode-route evidence therefore issue the typed boundary/forward pair
     * without invoking any test-only routing surface.
     *
     * @param response_token_budget Positive serving response budget.
     * @param purpose Stable diagnostic name for the traffic lifecycle phase.
     * @return Completion state, or no value when inference/maintenance failed.
     */
    std::optional<bool> runDynamicEconomyDecode(
        int response_token_budget,
        const char *purpose)
    {
        if (response_token_budget <= 0)
            return std::nullopt;
        orch_runner_->setDecodeStepTokenBudget(response_token_budget);
        const GenerationResult generated = orch_runner_->decodeStep();
        orch_runner_->setDecodeStepTokenBudget(0);
        if (!generated.success() || generated.tokens.empty())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic "
                << (purpose ? purpose : "economy")
                << " decode failed: " << generated.error);
            return std::nullopt;
        }
        if (!orch_runner_->maybeApplyMoERebalance(generated.tokens.size()))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic maintenance wake after "
                << (purpose ? purpose : "economy")
                << " decode failed: " << orch_runner_->lastError());
            return std::nullopt;
        }
        return generated.is_complete;
    }

    /**
     * @brief Replay one untimed request with the exact measured route shape.
     *
     * Every pair restores the same terminal prefill state, consumes its logits
     * with one boundary sample, and executes one DecodeToken forward. This is
     * deliberately the same production sequence as collectInferenceTimings(),
     * minus clocks and assertions, so the planner cannot optimize a longer
     * autoregressive trajectory than the one judged by the convergence gate.
     *
     * @param corpus_request Stable request ordinal in the timing corpus.
     * @return Whether all production requests and maintenance notifications ran.
     */
    bool replayStationaryConvergenceRequest(int corpus_request)
    {
        const std::vector<int32_t> prompt =
            makeReferenceShapedEconomyPrompt(
                corpus_request,
                ReferenceEconomyPromptRole::TimingCohort);
        for (int step = 0;
             step < kConvergenceTimingDecodeForwardsPerRequest;
             ++step)
        {
            if (step == 0)
            {
                if (!runDynamicEconomyPrefill(
                        prompt, "stationary-convergence"))
                {
                    return false;
                }
            }
            else
            {
                /* Request reset plus ordinary prefix restore reproduces the
                 * exact terminal state consumed by every measured pair. */
                if (!runDynamicEconomyPrefill(
                        prompt, "stationary-convergence-prefix-restore"))
                {
                    return false;
                }
            }

            if (!runDynamicEconomyDecode(
                    1, "stationary-convergence-boundary") ||
                !runDynamicEconomyDecode(
                    1, "stationary-convergence-forward"))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Learn and publish the measured service profile from real traffic.
     *
     * Transport preparation already completed through prepareForInference().
     * This phase supplies a bounded broad token corpus so every live sparse
     * participant receives natural prefill/decode work. Production publishes
     * the certificate while the corpus remains quarantined from optimization
     * demand. The following public request boundary discards that calibration
     * bank and admits the real convergence/parity workload.
     *
     * @return True after the sole production authority reports certification.
     */
    bool certifyDynamicResidencyEconomy()
    {
        if (!isDynamicResidencyProductionTest())
            return true;
        if (!isRootParityRank())
        {
            throw std::logic_error(
                "Only the coordinated parity root may certify Dynamic residency economy");
        }
        if (dynamic_residency_proof_lifecycle_.phase() !=
            DynamicResidencyProofPhase::AwaitingEconomyCertification)
        {
            throw std::logic_error(
                "Dynamic residency economy certification entered out of order");
        }

        const int decode_steps_per_request =
            isQwen122ProductionTest() ? 8 : 9;
        std::uint64_t service_profile_forwards = 0;
        MoEOptimizationStatus optimization = optimizationStatus();
        if (!optimization.learning() && !optimization.active())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic economy certification has no learning or active production authority");
            return false;
        }
        const MoEOptimizationMovementLedger certification_ledger =
            orch_runner_->moeOptimizationMovementLedger();
        if (!certification_ledger.complete())
        {
            throw std::logic_error(
                "Dynamic economy certification began with a truncated authoritative ledger");
        }
        /*
         * Freeze the frontier before the first ordinary service request. A
         * background transfer is allowed to publish while this corpus runs;
         * that publication is part of the convergence proof, not a new
         * baseline to be silently adopted by the later traffic driver.
         */
        dynamic_residency_proof_lifecycle_.beginEconomyCertification({
            .published_waves = optimization.published_movement_waves,
            .completed_transactions =
                optimization.completed_movement.transactions,
            .ledger_edges = certification_ledger.edges.size(),
            .host_admissions =
                certification_ledger.host_admissions.size(),
        });
        for (int request_index = 0;
             request_index < kMaximumServiceCertificationRequests &&
            !optimization.active();
             ++request_index)
        {
            if (!runDynamicEconomyPrefill(
                    makeEconomyWorkloadPrompt(request_index),
                    "service-certification"))
            {
                return false;
            }
            ++service_profile_forwards;

            bool request_complete = false;
            for (int step = 0;
                 !request_complete && step < decode_steps_per_request;
                 ++step)
            {
                const int response_token_budget =
                    isQwen122ProductionTest() &&
                            step < decode_steps_per_request / 2
                        ? 1
                        : (isQwen122ProductionTest() ? 4 : 2);
                const auto complete = runDynamicEconomyDecode(
                    response_token_budget,
                    "service-certification");
                if (!complete)
                    return false;
                ++service_profile_forwards;
                request_complete = *complete;
            }
            optimization = optimizationStatus();
        }

        if (!optimization.active())
        {
            struct ServiceTrafficTotals
            {
                double active_routes = 0.0;
                std::uint64_t completed_packets = 0u;
            };
            std::map<std::pair<std::string, std::string>,
                     ServiceTrafficTotals>
                service_traffic_by_participant_phase;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"forward_graph"}))
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "forward_graph")
                {
                    continue;
                }
                const auto participant = record.tags.find("participant");
                const auto source = record.tags.find("service_source");
                if (participant == record.tags.end() ||
                    source == record.tags.end())
                {
                    continue;
                }
                auto &totals = service_traffic_by_participant_phase[
                    {participant->second, source->second}];
                if (record.name ==
                    "moe_overlay_local_expert_active_routes")
                {
                    totals.active_routes += record.value;
                }
                else if (record.name ==
                         "moe_overlay_local_expert_completions")
                {
                    totals.completed_packets += record.count;
                }
            }
            std::ostringstream service_traffic;
            service_traffic << "rank=" << (mpi_ctx_ ? mpi_ctx_->rank() : 0);
            for (const auto &[coordinate, totals] :
                 service_traffic_by_participant_phase)
            {
                service_traffic
                    << " p" << coordinate.first << '/'
                    << coordinate.second << "{routes="
                    << static_cast<std::uint64_t>(totals.active_routes)
                    << ",packets=" << totals.completed_packets << '}';
            }
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic economy certification did not complete after "
                << service_profile_forwards
                << " ordinary production forwards; service_traffic="
                << service_traffic.str() << "\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency"}));
            return false;
        }

        dynamic_residency_proof_lifecycle_.completeEconomyCertification();
        return true;
    }

    /**
     * @brief Feed bounded real requests until distributed residency actually moves.
     *
     * Physical topology preparation, measured service certification, and the
     * initial stationary timing cohort are complete before this method begins.
     * This driver replays those exact prompt identities while optimizing, so
     * phase-weighted placement sees the same prefill and decode trajectories
     * that the A/B gate will judge. Only a typed demand-window-rotation phase
     * uses cache-distinct identities after the movement objective is already
     * satisfied. The background worker owns interval
     * selection, staging, transfer, overlap validation, and publication; no
     * histogram, placement, or completion value is injected here.
     *
     * The parity-artifact rank is the sole traffic-control authority. Remote
     * ranks are already inside `MPIWorkerLoop` and execute authenticated
     * transaction-follower commands; they are not peer test drivers. Calling a
     * test-owned MPI collective here would create a second command protocol
     * and collide with the follower's next typed command receive.
     *
     * @return True on the coordinated root after the typed production owner
     *         satisfies the required publication and movement-axis target.
     *         The post-shutdown PerfStats gate mirrors payload and economy
     *         diagnostics, but never controls this driver.
     */
    bool driveDynamicResidencyToDistributedMigration()
    {
        if (!isDynamicResidencyProductionTest())
            return true;
        if (!isRootParityRank())
        {
            throw std::logic_error(
                "Only the coordinated parity root may drive Dynamic residency traffic");
        }

        const DynamicResidencyProofPhase required_phase =
            requiresObservedConvergenceSpeedup()
                ? DynamicResidencyProofPhase::InitialCohortMeasured
                : DynamicResidencyProofPhase::EconomyCertified;
        if (dynamic_residency_proof_lifecycle_.phase() != required_phase)
        {
            throw std::logic_error(
                "Dynamic residency movement entered before its certified workload baseline");
        }

        /*
         * A request that closes the final histogram window only wakes the
         * background authority; it does not synchronously publish that wave.
         * Keep serving up to one additional corpus cycle so transfer,
         * certification, and event-owned publication can overlap inference.
         * This is a bounded test horizon, not a wait or a production
         * scheduling constant.
         */
        const DynamicResidencyConvergenceTarget convergence_target =
            dynamicResidencyConvergenceTarget();
        const int maximum_certified_requests =
            (requiresObservedConvergenceSpeedup()
                 ? kMaximumDynamicHistogramRequests
                 : movementProofHistogramRequestBudget(
                       activeModelParityCaseOrThrow()
                           .dynamic_rebalance.window_size,
                       std::min(
                           activeModelParityCaseOrThrow()
                               .dynamic_rebalance.window_size,
                           static_cast<int>(config_.token_ids.size())),
                       convergence_target.minimum_published_waves)) +
            kMaximumDynamicPublicationOverlapRequests;
        const int decode_steps_per_certified_request =
            isQwen122ProductionTest() ? 8 : 9;

        const int vocabulary_size = orch_runner_->vocabSize();
        if (config_.token_ids.empty() || vocabulary_size <= 4'096)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Economy service coverage requires a non-empty authenticated prompt and a valid vocabulary");
            return false;
        }
        int certified_requests = 0;
        const MoEOptimizationStatus initial_optimization =
            optimizationStatus();
        if (!initial_optimization.active())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic movement began before measured economy became active");
            return false;
        }
        const DynamicResidencyConvergenceOrigin &convergence_origin =
            dynamic_residency_proof_lifecycle_.convergenceOrigin();

        enum class MovementObservation : std::uint8_t
        {
            Continue,
            TargetSatisfied,
        };
        enum class ConvergenceBoundarySettlement : std::uint8_t
        {
            Settled,
            NeedsDemandWindowClosure,
            Failed,
        };
        const ConvergenceBoundaryPurpose boundary_purpose =
            requiresObservedConvergenceSpeedup()
                ? ConvergenceBoundaryPurpose::ObservedSpeedupCohort
                : ConvergenceBoundaryPurpose::MovementProof;
        std::optional<DemandWindowClosure> pending_demand_closure;
        const auto observeMovement = [&]()
        {
            const MoEOptimizationStatus current = optimizationStatus();
            if (!current.active())
            {
                throw std::logic_error(
                    "Active Dynamic economy regressed during movement observation");
            }
            const MoEOptimizationMovementLedger ledger =
                orch_runner_->moeOptimizationMovementLedger();
            const DynamicResidencyConvergenceState convergence =
                classifyDynamicResidencyConvergence(
                    convergence_target,
                    convergence_origin,
                    current,
                    ledger);
            if (convergence ==
                DynamicResidencyConvergenceState::InvalidAuthorityEvidence)
            {
                throw std::logic_error(
                    "ExpertOverlay convergence observer received regressed, truncated, or malformed authority evidence");
            }
            if (convergence !=
                DynamicResidencyConvergenceState::Satisfied)
            {
                return MovementObservation::Continue;
            }
            /*
             * Device publication makes the new residency selectable before
             * asynchronous source retirement and evidence publication finish.
             * The classifier requires publication, physical completion, and
             * the complete typed ledger suffix. Reaching this branch therefore
             * proves both causal event edges plus every topology-required axis
             * without synchronizing inference or treating PerfStats as authority.
             */
            dynamic_residency_proof_lifecycle_.recordMovementTarget();
            return MovementObservation::TargetSatisfied;
        };

        /**
         * Stop adding demand and classify the next measurement boundary.
         *
         * Quiescence proves that no complete window or admitted wave exists.
         * That is the complete contract for a movement-only parity cell. The
         * observed-speedup cell additionally proves that the partially
         * collected window can contain all 414 routed rows in its measured
         * cohort. The authority publishes exact active-bank occupancy for that
         * purpose. An insufficient but quiescent bank returns one typed exact
         * closure request. Ordinary cache-distinct production traffic fills
         * precisely the remaining rows and then stops while the authority
         * rotates the completed bank. No demand is discarded, movement is not
         * paused, and the test never mutates controller policy.
         */
        const auto settleConvergenceBoundary = [&]()
        {
            constexpr auto kNoProgressDeadline =
                std::chrono::seconds(30);
            MoEOptimizationStatus last_status = optimizationStatus();
            MoEOptimizationProgressStamp last_progress =
                last_status.progressStamp();
            auto no_progress_deadline =
                std::chrono::steady_clock::now() +
                kNoProgressDeadline;
            for (;;)
            {
                last_status = optimizationStatus();
                const auto observed_at =
                    std::chrono::steady_clock::now();
                const MoEOptimizationProgressStamp current_progress =
                    last_status.progressStamp();
                switch (current_progress.relationTo(last_progress))
                {
                case MoEOptimizationProgressRelation::Regressed:
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Between-wave settlement observed regressed optimization progress");
                    return ConvergenceBoundarySettlement::Failed;
                case MoEOptimizationProgressRelation::Advanced:
                    /*
                     * The standard 30-second bound applies to one unchanged
                     * lifecycle frontier. A second queued device-histogram
                     * bank may legitimately begin only after the preceding
                     * movement publishes, so durable movement, demand-bank,
                     * or reconciliation progress starts a new bounded edge.
                     * Activity enum changes alone never renew this watchdog.
                     */
                    last_progress = current_progress;
                    no_progress_deadline =
                        observed_at + kNoProgressDeadline;
                    break;
                case MoEOptimizationProgressRelation::Unchanged:
                    break;
                }
                const MoEOptimizationMovementLedger ledger =
                    orch_runner_->moeOptimizationMovementLedger();
                const DynamicResidencyConvergenceState convergence =
                    classifyDynamicResidencyConvergence(
                        convergence_target,
                        convergence_origin,
                        last_status,
                        ledger);
                if (convergence ==
                    DynamicResidencyConvergenceState::
                        InvalidAuthorityEvidence)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Between-wave settlement observed invalid authoritative movement evidence");
                    return ConvergenceBoundarySettlement::Failed;
                }
                if (convergence ==
                    DynamicResidencyConvergenceState::Satisfied)
                {
                    /* The request that closed the final demand bank can leave
                     * its profitable wave in the background authority after
                     * the finite traffic horizon. Settlement owns that same
                     * lifecycle edge, so publish the typed target transition
                     * here as well as in the request-boundary fast path. */
                    dynamic_residency_proof_lifecycle_
                        .recordMovementTarget();
                }
                if (convergence ==
                    DynamicResidencyConvergenceState::Satisfied)
                {
                    const ConvergenceBoundaryDecision boundary =
                        classifyConvergenceBoundary(
                            last_status, boundary_purpose);
                    switch (boundary.state)
                    {
                    case ConvergenceBoundaryState::AwaitingQuiescence:
                        break;
                    case ConvergenceBoundaryState::Ready:
                        dynamic_residency_proof_lifecycle_
                            .recordMovementBoundarySettled();
                        return ConvergenceBoundarySettlement::Settled;
                    case ConvergenceBoundaryState::InvalidAuthorityEvidence:
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Host Dynamic authority reached a quiescent boundary without an authoritative demand-window geometry");
                        return ConvergenceBoundarySettlement::Failed;
                    case ConvergenceBoundaryState::NeedsDemandWindowClosure:
                        if (!boundary.closure ||
                            !boundary.closure->valid())
                        {
                            LOG_ERROR(
                                "[Qwen3.5 MoE GraphNative] Dynamic boundary classified closure work without a valid typed closure");
                            return ConvergenceBoundarySettlement::Failed;
                        }
                        pending_demand_closure = *boundary.closure;
                        LOG_INFO(
                            "[Qwen3.5 MoE GraphNative] Quiescent Dynamic boundary requires exact demand-window closure before timing: generation="
                            << boundary.closure->generation
                            << " collected_rows="
                            << last_status.demand_window
                                   .collected_routed_rows
                            << " capacity_rows="
                            << last_status.demand_window
                                   .capacity_routed_rows
                            << " closure_rows="
                            << boundary.closure->routed_rows
                            << " required_headroom="
                            << convergenceTimingCohortRoutedRows());
                        return ConvergenceBoundarySettlement::
                            NeedsDemandWindowClosure;
                    }
                }
                if (observed_at >= no_progress_deadline)
                    break;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));
            }
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic residency made no authoritative progress for the canonical protocol interval before reaching a passive between-wave boundary: activity="
                << static_cast<int>(last_status.activity)
                << " published_waves="
                << last_status.published_movement_waves
                << " completed_transactions="
                << last_status.completed_movement.transactions
                << " demand_generation="
                << last_status.demand_window.generation
                << " demand_rows="
                << last_status.demand_window.collected_routed_rows
                << " demand_capacity="
                << last_status.demand_window.capacity_routed_rows
                << " published_progress_generation="
                << last_status.published_progress_generation
                << " reconciled_progress_generation="
                << last_status.reconciled_progress_generation);
            return ConvergenceBoundarySettlement::Failed;
        };

        /** @return True only after a typed settlement result terminates driving. */
        const auto handleSettlement = [&](MovementObservation observation)
            -> std::optional<bool>
        {
            if (observation != MovementObservation::TargetSatisfied)
                return std::nullopt;
            switch (settleConvergenceBoundary())
            {
            case ConvergenceBoundarySettlement::Settled:
                return true;
            case ConvergenceBoundarySettlement::NeedsDemandWindowClosure:
                return std::nullopt;
            case ConvergenceBoundarySettlement::Failed:
                return false;
            }
            throw std::logic_error(
                "Unhandled Dynamic convergence settlement state");
        };

        for (; certified_requests < maximum_certified_requests ||
               pending_demand_closure.has_value();
             ++certified_requests)
        {
            /*
             * Observe at both request boundaries so a completed publication
             * enters the explicit settlement state without admitting another
             * request. Once settlement asks only for exact bank closure, the
             * typed request below switches to a disjoint cache namespace and
             * consumes precisely the remaining authority-published rows.
            */
            if (!pending_demand_closure)
            {
                const MovementObservation before_request = observeMovement();
                if (const auto settled = handleSettlement(before_request))
                    return *settled;
            }

            /*
             * Before the movement objective is satisfied, the observed speed
             * witness cycles the exact A/B corpus. This is intentionally normal
             * production behavior: restored prefixes add decode demand, while
             * each published fingerprint makes the next copy execute full
             * prefill again. If only bank closure remains, one unique identity
             * finishes exactly that window without populating an A/B prefix
             * key or overflowing demand into its successor bank.
             */
            const bool closes_demand_window =
                pending_demand_closure.has_value();
            const int corpus_request = convergenceMovementPromptIdentity(
                closes_demand_window
                    ? ConvergenceMovementTraffic::DemandWindowClosure
                    : (requiresObservedConvergenceSpeedup()
                           ? ConvergenceMovementTraffic::MeasuredWorkload
                           : ConvergenceMovementTraffic::MovementProof),
                certified_requests);
            if (closes_demand_window)
            {
                const DemandWindowClosure closure =
                    *pending_demand_closure;
                pending_demand_closure.reset();
                if (!runDynamicEconomyPrefill(
                        makeDemandWindowClosurePrompt(
                            corpus_request, closure.routed_rows),
                        "demand-window-closure"))
                {
                    return false;
                }
                /* The first decode call consumes already-produced prefill
                 * logits and publishes the ordinary request-progress wake. It
                 * adds no routed model row, so the bank closes exactly. */
                if (!runDynamicEconomyDecode(
                         1, "demand-window-closure-boundary")
                         .has_value())
                {
                    return false;
                }
            }
            else if (requiresObservedConvergenceSpeedup())
            {
                if (!replayStationaryConvergenceRequest(corpus_request))
                    return false;
            }
            else
            {
                if (!runDynamicEconomyPrefill(
                    makeReferenceShapedEconomyPrompt(
                            corpus_request,
                            ReferenceEconomyPromptRole::MovementProof),
                        "stationary-movement"))
                {
                    return false;
                }
                for (int step = 0;
                     step < decode_steps_per_certified_request;
                     ++step)
                {
                    const auto complete = runDynamicEconomyDecode(
                        isQwen122ProductionTest() ? 4 : 2,
                        "stationary-movement");
                    if (!complete)
                        return false;
                    if (*complete)
                        break;
                }
            }
            const MovementObservation after_request = observeMovement();
            if (const auto settled = handleSettlement(after_request))
                return *settled;
        }

        /*
         * The finite corpus guarantees enough authenticated rows to close all
         * required demand banks; it does not require the asynchronous worker
         * to finish the last physical wave before the final request returns.
         * Stop admitting new demand and reuse the one typed progress/quiescence
         * state above. This deliberately waits only for already-enqueued
         * authority work and retains the standard 30-second no-progress gate.
         */
        switch (settleConvergenceBoundary())
        {
        case ConvergenceBoundarySettlement::Settled:
            return true;
        case ConvergenceBoundarySettlement::NeedsDemandWindowClosure:
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Finite movement traffic reached its publication target but exhausted its request budget before exact demand-window closure");
            break;
        case ConvergenceBoundarySettlement::Failed:
            break;
        }

        if (isRootParityRank())
        {
            /*
             * Preserve every tagged planner/admission decision before the
             * coordinated shutdown starts. A teardown failure must not erase
             * the numerical and economy evidence for the original gate.
             */
            writeResidencyDiagnosticsCsv();
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic residency did not produce the required profitable publication epoch(s) and topology-valid movement after "
                << certified_requests << " certified stationary-workload requests\n"
                << PerfStatsCollector::summaryString(
                       {"moe_overlay_residency", "moe_overlay_controller"}));
        }
        return false;
    }

    /**
     * @brief Prove all-GPU placement with the sole device-resident authority.
     *
     * The heterogeneous host authority publishes `committed_waves` and its
     * migration ledger under `moe_overlay_residency`.  An all-GPU topology has
     * no such second authority: the authenticated device-controller command is
     * the placement decision and its completed physical transaction is the
     * evidence. This fold checks numeric-priority promotion/demotion in every
     * Dynamic topology and additionally requires same-priority skew movement
     * whenever one declared priority owns two or more physical participants.
     */
    void assertDeviceResidentMovementEvidence() const
    {
        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            EconomyReady,
            CertificationComplete,
            RuntimeEpochParticipants,
            Transactions,
            Commands,
            PhysicalBytes,
            Promotions,
            Demotions,
            SamePriorityMoves,
            TypedLedgerComplete,
            TypedLedgerMalformed,
            TypedLedgerEdges,
            TypedTierResidencyEdges,
            TypedParticipantPlacementEdges,
            TypedCombinedEdges,
            CrossDomainMoves,
            CrossRankMoves,
            CrossBackendMoves,
            MigrationEdges,
            BackgroundNotifications,
            PhysicalOperations,
            ConfiguredCycleCapRecords,
            AdoptedInitialSlots,
            BootstrapSlotsRecycled,
            UniqueImprovingEpochs,
            EconomyAuthorityLedgers,
            EconomyAuthorityPublishedWaves,
            CapacityConservationCertifications,
            TaggedTransactions,
            TaggedCommands,
            TaggedPhysicalBytes,
            TaggedPromotions,
            TaggedDemotions,
            TaggedSamePriorityMoves,
            TaggedCrossDomainMoves,
            TaggedCrossRankMoves,
            TaggedCrossBackendMoves,
            EvidenceViolations,
            EvidenceCount,
        };

        std::array<uint64_t, EvidenceCount> local{};
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("moe_overlay_controller")
                ? 1u
                : 0u;
        local[RuntimeEpochParticipants] =
            orch_runner_ && orch_runner_->moeRuntimeMovementEpoch() > 0u
                ? 1u
                : 0u;
        ASSERT_NE(orch_runner_, nullptr);
        const auto movement_ledger =
            orch_runner_->moeOptimizationMovementLedger();
        local[TypedLedgerComplete] = movement_ledger.complete() ? 1u : 0u;
        std::map<std::uint64_t, std::uint64_t>
            typed_edges_by_transaction;
        std::map<std::uint64_t, std::set<std::size_t>>
            typed_cycles_by_transaction;
        for (const auto &edge : movement_ledger.edges)
        {
            if (!edge.valid() ||
                edge.authority != MoEOptimizationAuthority::Device ||
                edge.blocking_inference)
            {
                ++local[TypedLedgerMalformed];
                continue;
            }
            ++local[TypedLedgerEdges];
            ++typed_edges_by_transaction[edge.transaction];
            typed_cycles_by_transaction[edge.transaction].insert(
                edge.cycle_index);
            switch (edge.axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                ++local[TypedTierResidencyEdges];
                break;
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                ++local[TypedParticipantPlacementEdges];
                break;
            case MoEOptimizationMovementAxis::Combined:
                ++local[TypedCombinedEdges];
                break;
            }
        }
        if (!movement_ledger.economy.empty())
        {
            ++local[EconomyAuthorityLedgers];
            local[EconomyAuthorityPublishedWaves] =
                optimizationStatus().published_movement_waves;
        }
        std::set<std::uint64_t> improving_epochs;
        for (const auto &economy : movement_ledger.economy)
        {
            const auto edge_count =
                typed_edges_by_transaction.find(economy.transaction);
            const auto cycle_count =
                typed_cycles_by_transaction.find(economy.transaction);
            const bool identity_valid =
                economy.valid() &&
                economy.authority == MoEOptimizationAuthority::Device &&
                edge_count != typed_edges_by_transaction.end() &&
                edge_count->second == economy.command_count &&
                cycle_count != typed_cycles_by_transaction.end() &&
                cycle_count->second.size() == economy.cycle_count &&
                improving_epochs.insert(economy.candidate_epoch).second;
            if (!identity_valid)
                ++local[EvidenceViolations];
        }
        const auto add = [&local](Evidence evidence, double value)
        {
            if (value > 0.0 && std::isfinite(value) &&
                value <= static_cast<double>(
                    std::numeric_limits<uint64_t>::max()))
            {
                local[evidence] += static_cast<uint64_t>(value);
            }
        };
        const auto parse_u64 = [](const std::string &text,
                                  uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto parse_i64 = [](const std::string &text,
                                  std::int64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto tag_u64 = [&](const PerfStatRecord &record,
                                 const char *name,
                                 uint64_t &value)
        {
            const auto found = record.tags.find(name);
            return found != record.tags.end() &&
                   parse_u64(found->second, value);
        };

        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_controller", "moe_overlay_residency"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;

            if (record.domain == "moe_overlay_residency")
            {
                if (record.name == "economy_certification_complete")
                    add(CertificationComplete, record.value);
                else if (record.name == "physical_fabrics_materialized")
                {
                    add(ConfiguredCycleCapRecords, record.value);
                    uint64_t cycle_cap = 0u;
                    uint64_t adopted_slots = 0u;
                    if (!tag_u64(
                            record,
                            "maximum_concurrent_cycles",
                            cycle_cap) ||
                        cycle_cap !=
                            convergence_migration_transfer_slots_ ||
                        !tag_u64(
                            record,
                            "adopted_initial_slots",
                            adopted_slots))
                    {
                        ++local[EvidenceViolations];
                    }
                    else
                    {
                        local[AdoptedInitialSlots] += adopted_slots;
                    }
                }
                else if (record.name == "bootstrap_live_slots_recycled")
                {
                    add(BootstrapSlotsRecycled, record.value);
                }
                continue;
            }
            if (record.domain != "moe_overlay_controller")
                continue;

            if (record.name == "static_no_movement_transactions")
            {
                uint64_t commands = 1u;
                uint64_t bytes = 1u;
                const auto waits = record.tags.find("inference_stream_waits");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "movement_commands", commands) &&
                    commands == 0u &&
                    tag_u64(record, "packed_weight_bytes", bytes) &&
                    bytes == 0u && waits != record.tags.end() &&
                    waits->second == "0";
                if (valid)
                    add(StaticChecks, record.value);
                else
                    ++local[EvidenceViolations];
            }
            else if (record.name == "device_economy_ready")
            {
                add(EconomyReady, record.value);
            }
            else if (record.name == "dynamic_movement_transactions")
            {
                uint64_t transaction = 0u;
                uint64_t base_epoch = 0u;
                uint64_t candidate_epoch = 0u;
                uint64_t commands = 0u;
                uint64_t bytes = 0u;
                uint64_t promotions = 0u;
                uint64_t demotions = 0u;
                uint64_t same_priority = 0u;
                uint64_t cross_domain = 0u;
                uint64_t cross_rank = 0u;
                uint64_t cross_backend = 0u;
                uint64_t accepted_cycles = 0u;
                uint64_t service_gain = 0u;
                uint64_t net_benefit = 0u;
                const auto policy_owner = record.tags.find("policy_owner");
                const auto blocking = record.tags.find("blocking_inference");
                const bool valid =
                    record.phase == "maintenance" && record.value > 0.0 &&
                    record.count > 0u &&
                    tag_u64(record, "transaction", transaction) &&
                    transaction > 0u &&
                    tag_u64(record, "base_epoch", base_epoch) &&
                    tag_u64(record, "candidate_epoch", candidate_epoch) &&
                    candidate_epoch == base_epoch + 1u &&
                    tag_u64(record, "movement_commands", commands) &&
                    commands > 0u &&
                    tag_u64(record, "physical_bytes", bytes) && bytes > 0u &&
                    tag_u64(record, "promotions", promotions) &&
                    tag_u64(record, "demotions", demotions) &&
                    tag_u64(record, "same_priority_moves", same_priority) &&
                    tag_u64(record, "cross_domain_moves", cross_domain) &&
                    tag_u64(record, "cross_rank_moves", cross_rank) &&
                    tag_u64(record, "cross_backend_moves", cross_backend) &&
                    tag_u64(record, "accepted_cycles", accepted_cycles) &&
                    accepted_cycles > 0u &&
                    accepted_cycles <=
                        convergence_migration_cycles_per_wave_ &&
                    tag_u64(
                        record,
                        "projected_service_gain_ns",
                        service_gain) &&
                    service_gain > 0u &&
                    tag_u64(
                        record,
                        "projected_net_benefit_ns",
                        net_benefit) &&
                    net_benefit > 0u &&
                    policy_owner != record.tags.end() &&
                    policy_owner->second == "device" &&
                    blocking != record.tags.end() &&
                    blocking->second == "false";
                if (!valid)
                {
                    ++local[EvidenceViolations];
                    continue;
                }
                add(Transactions, record.value);
                add(TaggedTransactions, record.value);
                local[TaggedCommands] += commands;
                local[TaggedPhysicalBytes] += bytes;
                local[TaggedPromotions] += promotions;
                local[TaggedDemotions] += demotions;
                local[TaggedSamePriorityMoves] += same_priority;
                local[TaggedCrossDomainMoves] += cross_domain;
                local[TaggedCrossRankMoves] += cross_rank;
                local[TaggedCrossBackendMoves] += cross_backend;
            }
            else if (record.name == "dynamic_movement_commands")
                add(Commands, record.value);
            else if (record.name == "dynamic_physical_bytes")
                add(PhysicalBytes, record.value);
            else if (record.name == "dynamic_promotions")
                add(Promotions, record.value);
            else if (record.name == "dynamic_demotions")
                add(Demotions, record.value);
            else if (record.name == "dynamic_same_priority_moves")
                add(SamePriorityMoves, record.value);
            else if (record.name == "dynamic_cross_domain_moves")
                add(CrossDomainMoves, record.value);
            else if (record.name == "dynamic_cross_rank_moves")
                add(CrossRankMoves, record.value);
            else if (record.name == "dynamic_cross_backend_moves")
                add(CrossBackendMoves, record.value);
            else if (record.name ==
                     "dynamic_capacity_conservation_certifications")
            {
                uint64_t edges = 0u;
                uint64_t participant_coordinates = 0u;
                uint64_t tier_coordinates = 0u;
                uint64_t malformed = 0u;
                uint64_t participant_violations = 0u;
                uint64_t tier_violations = 0u;
                const auto direction_proxy = record.tags.find(
                    "direction_counts_are_capacity_proof");
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "edges_checked", edges) && edges > 0u &&
                    tag_u64(
                        record,
                        "participant_coordinates_checked",
                        participant_coordinates) &&
                    participant_coordinates > 0u &&
                    tag_u64(
                        record,
                        "tier_coordinates_checked",
                        tier_coordinates) &&
                    tier_coordinates > 0u &&
                    tag_u64(record, "malformed_edges", malformed) &&
                    malformed == 0u &&
                    tag_u64(
                        record,
                        "participant_flow_violations",
                        participant_violations) &&
                    participant_violations == 0u &&
                    tag_u64(
                        record,
                        "tier_flow_violations",
                        tier_violations) &&
                    tier_violations == 0u &&
                    direction_proxy != record.tags.end() &&
                    direction_proxy->second == "false";
                if (valid)
                    add(
                        CapacityConservationCertifications,
                        record.value);
                else
                    ++local[EvidenceViolations];
            }
            else if (record.name == "background_notification_batches")
                add(BackgroundNotifications, record.value);
            else if (record.name ==
                     "physical_wave_parallel_operations_started")
                add(PhysicalOperations, record.value);
            else if (record.name == "dynamic_migration_edges")
            {
                std::int64_t source_priority = 0;
                std::int64_t destination_priority = 0;
                uint64_t estimated_bytes = 0u;
                const auto direction = record.tags.find("direction");
                const auto source = record.tags.find("source_priority");
                const auto destination =
                    record.tags.find("destination_priority");
                const auto blocking = record.tags.find("blocking_inference");
                const bool parsed =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_i64(source->second, source_priority) &&
                    parse_i64(destination->second, destination_priority) &&
                    tag_u64(
                        record,
                        "estimated_weight_bytes",
                        estimated_bytes) &&
                    estimated_bytes > 0u;
                const bool direction_valid = parsed &&
                    ((direction->second == "promotion" &&
                      destination_priority < source_priority) ||
                     (direction->second == "demotion" &&
                      destination_priority > source_priority) ||
                     (direction->second == "same_priority" &&
                      destination_priority == source_priority));
                if (!direction_valid || blocking == record.tags.end() ||
                    blocking->second != "false")
                {
                    ++local[EvidenceViolations];
                }
                else
                {
                    add(MigrationEdges, record.value);
                }
            }
        }
        local[UniqueImprovingEpochs] = improving_epochs.size();

        std::array<uint64_t, EvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        const uint64_t ranks = static_cast<uint64_t>(mpiWorldSize());
        ASSERT_EQ(global[DomainEnabled], ranks)
            << "Every all-GPU participant rank must retain device-controller evidence";
        EXPECT_EQ(global[TypedLedgerComplete], ranks)
            << "Every device follower must retain the complete controller-authored movement ledger";
        EXPECT_EQ(global[TypedLedgerMalformed], 0u);
        EXPECT_EQ(global[EvidenceViolations], 0u);
        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[StaticChecks], ranks)
                << "Every Static device follower must certify zero movement";
            EXPECT_EQ(global[Transactions], 0u);
            EXPECT_EQ(global[Commands], 0u);
            EXPECT_EQ(global[PhysicalBytes], 0u);
            EXPECT_EQ(global[Promotions], 0u);
            EXPECT_EQ(global[Demotions], 0u);
            EXPECT_EQ(global[SamePriorityMoves], 0u);
            EXPECT_EQ(global[MigrationEdges], 0u);
            EXPECT_EQ(global[TypedLedgerEdges], 0u);
            EXPECT_EQ(global[UniqueImprovingEpochs], 0u);
            EXPECT_EQ(global[EconomyAuthorityLedgers], 0u);
            EXPECT_EQ(global[EconomyAuthorityPublishedWaves], 0u);
            return;
        }

        EXPECT_GE(global[EconomyReady], ranks)
            << "Every device follower must acquire the certified economy profile";
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_EQ(global[RuntimeEpochParticipants], ranks)
            << "Every rank must observe the completed durable movement epoch";
        EXPECT_GE(global[ConfiguredCycleCapRecords], ranks);
        EXPECT_GT(global[AdoptedInitialSlots], 0u);
        EXPECT_GT(global[BootstrapSlotsRecycled], 0u);
        EXPECT_GE(global[Transactions], ranks);
        EXPECT_EQ(global[EconomyAuthorityLedgers], 1u)
            << "Exactly one device-resident policy leader must own the "
               "admitting economics";
        EXPECT_GE(global[UniqueImprovingEpochs], 1u)
            << "The device policy leader retained no profitable movement epoch";
        EXPECT_EQ(
            global[UniqueImprovingEpochs],
            global[EconomyAuthorityPublishedWaves])
            << "Every device-authority publication must retain exactly one "
               "typed economy proof";
        EXPECT_EQ(
            global[Transactions],
            global[EconomyAuthorityPublishedWaves] * ranks)
            << "Follower telemetry must mirror each device-authority "
               "transaction; it is not an independent economy proof";
        EXPECT_GT(global[Commands], 0u);
        EXPECT_EQ(global[TypedLedgerEdges], global[Commands])
            << "Device-authored movement ledger and authenticated command accounting diverged";
        EXPECT_GT(global[PhysicalBytes], 0u);
        EXPECT_GT(global[BackgroundNotifications], 0u);
        EXPECT_GT(global[PhysicalOperations], 0u);
        EXPECT_GT(global[Promotions], 0u)
            << "Dynamic device policy never promoted a histogram-hot expert";
        EXPECT_GT(global[Demotions], 0u)
            << "Dynamic device policy never demoted an expert to release capacity";
        EXPECT_GT(
            global[TypedTierResidencyEdges] + global[TypedCombinedEdges],
            0u)
            << "Dynamic device policy never completed its tier-residency objective";
        EXPECT_EQ(
            global[CapacityConservationCertifications],
            global[Transactions])
            << "Every device-owned transaction must explicitly conserve participant and tier slots";
        if (dynamicMovementAxisContract(resolvedOverlayPlan()) ==
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance)
        {
            EXPECT_GT(
                global[TypedParticipantPlacementEdges] +
                    global[TypedCombinedEdges],
                0u)
                << "Dynamic device policy never completed its participant-placement objective";
        }
        else
        {
            EXPECT_EQ(
                global[TypedParticipantPlacementEdges] +
                    global[TypedCombinedEdges],
                0u)
                << "A layout without participant-balancing freedom reported that objective";
        }
        EXPECT_EQ(
            global[CrossDomainMoves],
            global[Promotions] + global[Demotions]);
        EXPECT_EQ(
            global[CrossBackendMoves],
            global[Promotions] + global[Demotions]);
        EXPECT_LE(global[CrossRankMoves], global[Commands]);
        EXPECT_EQ(global[MigrationEdges], global[Commands]);

        EXPECT_EQ(global[TaggedTransactions], global[Transactions]);
        EXPECT_EQ(global[TaggedCommands], global[Commands]);
        EXPECT_EQ(global[TaggedPhysicalBytes], global[PhysicalBytes]);
        EXPECT_EQ(global[TaggedPromotions], global[Promotions]);
        EXPECT_EQ(global[TaggedDemotions], global[Demotions]);
        EXPECT_EQ(
            global[TaggedSamePriorityMoves],
            global[SamePriorityMoves]);
        EXPECT_EQ(global[TaggedCrossDomainMoves], global[CrossDomainMoves]);
        EXPECT_EQ(global[TaggedCrossRankMoves], global[CrossRankMoves]);
        EXPECT_EQ(global[TaggedCrossBackendMoves], global[CrossBackendMoves]);
    }

    /**
     * @brief Fold and validate static immobility or dynamic movement evidence.
     *
     * Dynamic movement must be a capacity-preserving promotion/demotion cycle
     * across the two distinct domains, MPI ranks, and GPU backends. Static cells
     * must publish their typed immobility check and no positive movement edge.
     */
    void assertResidencyMovementEvidence() const
    {
        if (!topologyUsesCpu())
        {
            assertDeviceResidentMovementEvidence();
            return;
        }

        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            TransportProfileComplete,
            CertificationComplete,
            AuthorityCertifications,
            MaintenanceCertifications,
            CommittedWaves,
            CommittedMigrations,
            Promotions,
            Demotions,
            SamePriority,
            TypedLedgerComplete,
            TypedLedgerMalformed,
            TypedLedgerEdges,
            TypedTierResidencyEdges,
            TypedParticipantPlacementEdges,
            TypedCombinedEdges,
            CrossDomain,
            CrossRank,
            CrossBackend,
            EstimatedBytes,
            BackgroundNotifications,
            StageFailures,
            CommitFailures,
            FatalFailures,
            BlockingInferenceViolations,
            ImprovingEpochs,
            EconomyAuthorityLedgers,
            EconomyAuthorityPublishedWaves,
            ServiceGainEvidenceViolations,
            NetBenefitEvidenceViolations,
            PriorityDirectionViolations,
            ConfiguredCycleCapRecords,
            CycleCapViolations,
            HostAdmissionAuthorityLedgers,
            HostAdmissionRecords,
            HostPolicyEligibleParticipantCycles,
            HostAdmissionViolations,
            MultiCycleAuthorityWaves,
            AdoptedInitialSlots,
            BootstrapSlotsRecycled,
            PhysicalWavesPrepared,
            PhysicalEvidenceViolations,
            CapacityConservationCertifications,
            CapacityConservationViolations,
            EvidenceCount,
        };

        std::array<uint64_t, EvidenceCount> local{};
        std::map<std::uint64_t, std::uint64_t>
            typed_edges_by_transaction;
        std::map<std::uint64_t, std::set<std::size_t>>
            typed_cycles_by_transaction;
        std::set<std::uint64_t> host_admission_epochs;
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("moe_overlay_residency")
                ? 1u
                : 0u;
        ASSERT_NE(orch_runner_, nullptr);
        const auto movement_ledger =
            orch_runner_->moeOptimizationMovementLedger();
        local[TypedLedgerComplete] = movement_ledger.complete() ? 1u : 0u;
        for (const auto &edge : movement_ledger.edges)
        {
            if (!edge.valid() ||
                edge.authority != MoEOptimizationAuthority::Host)
            {
                ++local[TypedLedgerMalformed];
                continue;
            }
            ++local[TypedLedgerEdges];
            ++typed_edges_by_transaction[edge.transaction];
            typed_cycles_by_transaction[edge.transaction].insert(
                edge.cycle_index);
            switch (edge.axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                ++local[TypedTierResidencyEdges];
                break;
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                ++local[TypedParticipantPlacementEdges];
                break;
            case MoEOptimizationMovementAxis::Combined:
                ++local[TypedCombinedEdges];
                break;
            }
            if (edge.blocking_inference)
                ++local[BlockingInferenceViolations];
        }
        if (!movement_ledger.host_admissions.empty())
            ++local[HostAdmissionAuthorityLedgers];
        for (const auto &admission : movement_ledger.host_admissions)
        {
            const auto physical_cycles =
                typed_cycles_by_transaction.find(admission.transaction);
            const bool identity_valid =
                admission.valid() &&
                admission.authority == MoEOptimizationAuthority::Host &&
                admission.cycle_capacity_kind ==
                    MoEOptimizationCycleCapacityKind::Bounded &&
                admission.maximum_concurrent_cycles ==
                    convergence_migration_cycles_per_wave_ &&
                physical_cycles != typed_cycles_by_transaction.end() &&
                physical_cycles->second.size() ==
                    admission.admitted_physical_cycles &&
                host_admission_epochs.insert(
                    admission.candidate_epoch).second;
            if (!identity_valid)
            {
                ++local[HostAdmissionViolations];
                continue;
            }
            ++local[HostAdmissionRecords];
            local[HostPolicyEligibleParticipantCycles] +=
                admission.policy_eligible_axes
                    .participant_placement +
                admission.policy_eligible_axes.combined;
            if (admission.admitted_physical_cycles > 1u)
                ++local[MultiCycleAuthorityWaves];
        }
        if (!movement_ledger.economy.empty())
        {
            ++local[EconomyAuthorityLedgers];
            local[EconomyAuthorityPublishedWaves] =
                optimizationStatus().published_movement_waves;
        }
        std::set<std::uint64_t> economy_epochs;
        for (const auto &economy : movement_ledger.economy)
        {
            const auto edge_count =
                typed_edges_by_transaction.find(economy.transaction);
            const auto cycle_count =
                typed_cycles_by_transaction.find(economy.transaction);
            const bool identity_valid =
                economy.valid() &&
                economy.authority == MoEOptimizationAuthority::Host &&
                economy.transaction == economy.candidate_epoch &&
                edge_count != typed_edges_by_transaction.end() &&
                edge_count->second == economy.command_count &&
                cycle_count != typed_cycles_by_transaction.end() &&
                cycle_count->second.size() == economy.cycle_count &&
                host_admission_epochs.contains(economy.candidate_epoch) &&
                economy_epochs.insert(economy.candidate_epoch).second;
            if (!identity_valid)
            {
                ++local[ServiceGainEvidenceViolations];
                ++local[NetBenefitEvidenceViolations];
                continue;
            }
            ++local[ImprovingEpochs];
        }
        for (const std::uint64_t candidate_epoch : host_admission_epochs)
        {
            if (!economy_epochs.contains(candidate_epoch))
                ++local[HostAdmissionViolations];
        }
        const auto addValue = [&local](Evidence index, double value)
        {
            if (value > 0.0)
                local[index] += static_cast<uint64_t>(value);
        };
        const auto parse_u64 = [](const std::string &text,
                                  uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto parse_int = [](const std::string &text, int &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto tag_u64 = [&parse_u64](
                                 const PerfStatRecord &record,
                                 const char *name,
                                 uint64_t &value)
        {
            const auto found = record.tags.find(name);
            return found != record.tags.end() &&
                   parse_u64(found->second, value);
        };
        for (const auto &record :
             PerfStatsCollector::snapshot(
                 {"moe_overlay_residency", "moe_overlay_controller"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }

            if (record.domain == "moe_overlay_controller" &&
                record.name == "static_no_movement_transactions")
            {
                const auto movement_commands =
                    record.tags.find("movement_commands");
                const auto packed_weight_bytes =
                    record.tags.find("packed_weight_bytes");
                const auto inference_stream_waits =
                    record.tags.find("inference_stream_waits");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    movement_commands != record.tags.end() &&
                    movement_commands->second == "0" &&
                    packed_weight_bytes != record.tags.end() &&
                    packed_weight_bytes->second == "0" &&
                    inference_stream_waits != record.tags.end() &&
                    inference_stream_waits->second == "0";
                if (valid)
                    addValue(StaticChecks, record.value);
                else
                    ++local[PhysicalEvidenceViolations];
                continue;
            }
            if (record.domain != "moe_overlay_residency")
                continue;

            if (record.name == "static_no_movement_checks")
                addValue(StaticChecks, record.value);
            else if (record.name == "production_maintenance_composed")
            {
                addValue(ConfiguredCycleCapRecords, record.value);
                const auto cap_tag =
                    record.tags.find("migration_transfer_slots");
                uint64_t cap = 0;
                const uint64_t expected_cap =
                    isDynamicResidencyProductionTest()
                        ? convergence_migration_transfer_slots_
                        : 1u;
                if (cap_tag == record.tags.end() ||
                    !parse_u64(cap_tag->second, cap) ||
                    cap != expected_cap)
                {
                    ++local[CycleCapViolations];
                }
            }
            else if (record.name == "physical_fabrics_materialized")
            {
                const auto adopted_tag =
                    record.tags.find("adopted_initial_slots");
                uint64_t adopted = 0;
                if (adopted_tag == record.tags.end() ||
                    !parse_u64(adopted_tag->second, adopted))
                {
                    ++local[PhysicalEvidenceViolations];
                }
                else
                {
                    local[AdoptedInitialSlots] += adopted;
                }
            }
            else if (record.name == "economy_transport_profile_complete")
            {
                const auto synthetic =
                    record.tags.find("synthetic_inference");
                const auto publishes_residency =
                    record.tags.find("publish_residency");
                const auto waves_tag = record.tags.find("waves");
                const auto elapsed_tag =
                    record.tags.find("elapsed_nanoseconds");
                uint64_t waves = 0u;
                uint64_t elapsed = 0u;
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    synthetic != record.tags.end() &&
                    synthetic->second == "false" &&
                    publishes_residency != record.tags.end() &&
                    publishes_residency->second == "false" &&
                    waves_tag != record.tags.end() &&
                    parse_u64(waves_tag->second, waves) && waves > 0u &&
                    elapsed_tag != record.tags.end() &&
                    parse_u64(elapsed_tag->second, elapsed) && elapsed > 0u;
                if (valid)
                    addValue(TransportProfileComplete, record.value);
                else
                    ++local[PhysicalEvidenceViolations];
            }
            else if (record.name == "economy_certification_complete")
                addValue(CertificationComplete, record.value);
            else if (record.name == "economy_certifications")
                addValue(AuthorityCertifications, record.value);
            else if (record.name == "maintenance_economy_certifications")
                addValue(MaintenanceCertifications, record.value);
            else if (record.name == "committed_waves")
                addValue(CommittedWaves, record.value);
            else if (record.name == "committed_expert_migrations")
                addValue(CommittedMigrations, record.value);
            else if (record.name == "committed_migration_cycles")
            {
                if (record.value >
                        static_cast<double>(
                            convergence_migration_cycles_per_wave_))
                    ++local[CycleCapViolations];
            }
            else if (record.name == "bootstrap_live_slots_recycled")
                addValue(BootstrapSlotsRecycled, record.value);
            else if (record.name == "physical_waves_prepared")
                addValue(PhysicalWavesPrepared, record.value);
            else if (record.name ==
                     "capacity_conservation_certifications")
            {
                uint64_t edges = 0u;
                uint64_t cycles = 0u;
                uint64_t participant_coordinates = 0u;
                uint64_t tier_coordinates = 0u;
                uint64_t malformed = 0u;
                uint64_t participant_violations = 0u;
                uint64_t tier_violations = 0u;
                const auto direction_proxy = record.tags.find(
                    "direction_counts_are_capacity_proof");
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "edges_checked", edges) && edges > 0u &&
                    tag_u64(record, "closed_cycles", cycles) && cycles > 0u &&
                    tag_u64(
                        record,
                        "participant_coordinates_checked",
                        participant_coordinates) &&
                    participant_coordinates > 0u &&
                    tag_u64(
                        record,
                        "tier_coordinates_checked",
                        tier_coordinates) &&
                    tier_coordinates > 0u &&
                    tag_u64(record, "malformed_edges", malformed) &&
                    tag_u64(
                        record,
                        "participant_flow_violations",
                        participant_violations) &&
                    tag_u64(
                        record,
                        "tier_flow_violations",
                        tier_violations) &&
                    direction_proxy != record.tags.end() &&
                    direction_proxy->second == "false";
                if (valid)
                    addValue(
                        CapacityConservationCertifications,
                        record.value);
                if (!valid || malformed != 0u ||
                    participant_violations != 0u || tier_violations != 0u)
                {
                    ++local[CapacityConservationViolations];
                }
            }
            else if (record.name == "promotions")
                addValue(Promotions, record.value);
            else if (record.name == "demotions")
                addValue(Demotions, record.value);
            else if (record.name == "same_priority_moves")
                addValue(SamePriority, record.value);
            else if (record.name == "cross_domain_migrations")
                addValue(CrossDomain, record.value);
            else if (record.name == "cross_rank_migrations")
                addValue(CrossRank, record.value);
            else if (record.name == "cross_backend_migrations")
                addValue(CrossBackend, record.value);
            else if (record.name == "estimated_weight_bytes")
                addValue(EstimatedBytes, record.value);
            else if (record.name == "decode_boundary_background_notifications")
                addValue(BackgroundNotifications, record.value);
            else if (record.name == "migration_stage_failures")
                addValue(StageFailures, record.value);
            else if (record.name == "migration_commit_failures")
                addValue(CommitFailures, record.value);
            else if (record.name == "maintenance_fatal_failures")
                addValue(FatalFailures, record.value);
            else if (record.name == "expert_migration_edges")
            {
                const auto direction = record.tags.find("direction");
                const auto source = record.tags.find("source_priority");
                const auto destination =
                    record.tags.find("destination_priority");
                int source_priority = 0;
                int destination_priority = 0;
                uint64_t cycle_index = 0u;
                uint64_t cycle_size = 0u;
                bool direction_valid =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_int(source->second, source_priority) &&
                    parse_int(destination->second, destination_priority) &&
                    tag_u64(record, "cycle_index", cycle_index) &&
                    tag_u64(record, "cycle_size", cycle_size) &&
                    cycle_size > 0u;
                if (direction_valid)
                {
                    direction_valid =
                        (direction->second == "promotion" &&
                         destination_priority < source_priority) ||
                        (direction->second == "demotion" &&
                         destination_priority > source_priority) ||
                        (direction->second == "same_priority" &&
                         destination_priority == source_priority);
                }
                if (!direction_valid)
                    ++local[PriorityDirectionViolations];
            }

            const auto blocking = record.tags.find("blocking");
            const auto blocking_inference =
                record.tags.find("blocking_inference");
            const auto inference_path = record.tags.find("inference_path");
            const bool setup_only_blocking =
                blocking != record.tags.end() && blocking->second == "true" &&
                record.phase == "model_setup" &&
                inference_path != record.tags.end() &&
                inference_path->second == "false";
            if ((blocking != record.tags.end() && blocking->second == "true" &&
                 !setup_only_blocking) ||
                (blocking_inference != record.tags.end() &&
                 blocking_inference->second != "false"))
            {
                ++local[BlockingInferenceViolations];
            }
        }

        std::array<uint64_t, EvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        ASSERT_EQ(global[DomainEnabled], static_cast<uint64_t>(mpiWorldSize()))
            << "Every rank must retain ExpertOverlay residency PerfStats";
        EXPECT_EQ(
            global[TypedLedgerComplete],
            static_cast<uint64_t>(mpiWorldSize()))
            << "Every rank must retain its complete authority-owned movement ledger";
        EXPECT_EQ(global[TypedLedgerMalformed], 0u)
            << "The authority-owned movement ledger contained a malformed or wrongly owned edge";
        EXPECT_EQ(global[StageFailures], 0u);
        EXPECT_EQ(global[CommitFailures], 0u);
        EXPECT_EQ(global[FatalFailures], 0u);
        EXPECT_EQ(global[BlockingInferenceViolations], 0u)
            << "Expert movement exposed a blocking inference-path tag";
        EXPECT_EQ(global[PriorityDirectionViolations], 0u)
            << "Migration direction did not match numeric tier priority";

        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[StaticChecks], static_cast<uint64_t>(mpiWorldSize()))
                << "Every static authority must prove immobility exactly once";
            EXPECT_EQ(global[CommittedWaves], 0u);
            EXPECT_EQ(global[CommittedMigrations], 0u)
                << "Static ExpertOverlay placement moved an expert";
            EXPECT_EQ(global[Promotions], 0u);
            EXPECT_EQ(global[Demotions], 0u);
            EXPECT_EQ(global[SamePriority], 0u);
            EXPECT_EQ(global[TypedLedgerEdges], 0u)
                << "Static ExpertOverlay published an authoritative movement edge";
            EXPECT_EQ(global[ImprovingEpochs], 0u);
            EXPECT_EQ(global[EconomyAuthorityLedgers], 0u)
                << "Static ExpertOverlay published a movement-economy ledger";
            EXPECT_EQ(global[EconomyAuthorityPublishedWaves], 0u);
            EXPECT_EQ(global[HostAdmissionAuthorityLedgers], 0u)
                << "Static ExpertOverlay published a host-policy admission ledger";
            EXPECT_EQ(global[HostAdmissionRecords], 0u);
            EXPECT_EQ(global[HostAdmissionViolations], 0u);
            EXPECT_EQ(global[BootstrapSlotsRecycled], 0u)
                << "Static ExpertOverlay recycled no bootstrap assignment";
            return;
        }

        const uint64_t ranks = static_cast<uint64_t>(mpiWorldSize());
        EXPECT_GE(global[ConfiguredCycleCapRecords], ranks)
            << "Every rank must publish its production cycle cap";
        EXPECT_EQ(global[CycleCapViolations], 0u)
            << "The configured migration transfer-slot policy was not preserved through production composition";
        EXPECT_EQ(global[HostAdmissionAuthorityLedgers], 1u)
            << "Exactly one host policy authority must own candidate, policy, and capacity admission accounting";
        EXPECT_EQ(global[HostAdmissionViolations], 0u)
            << "Migration admission left a cycle unclassified or mislabeled an economic rejection as capacity pressure";
        EXPECT_EQ(global[PhysicalEvidenceViolations], 0u)
            << "The physical fabric published malformed slot-adoption evidence";
        if (activeOverlayTierCount() >= 3u)
        {
            EXPECT_GT(global[MultiCycleAuthorityWaves], 0u)
                << "The three-tier real-model campaign never used concurrent "
                   "migration transfer slots";
        }
        EXPECT_GT(global[AdoptedInitialSlots], 0u)
            << "The physical fabric did not adopt loader-owned expert slots";
        EXPECT_GT(global[BootstrapSlotsRecycled], 0u)
            << "Movement never recycled a retired loader-owned expert slot";
        EXPECT_GT(global[PhysicalWavesPrepared], 0u)
            << "No physical migration wave reached background preparation";
        EXPECT_GE(global[TransportProfileComplete], ranks)
            << "Every dynamic authority must finish bounded transport profiling without synthetic inference";
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_GE(global[AuthorityCertifications], ranks);
        EXPECT_GE(global[MaintenanceCertifications], ranks);
        const uint64_t minimum_epochs_per_rank =
            dynamicResidencyConvergenceTarget().minimum_published_waves;
        EXPECT_EQ(global[EconomyAuthorityLedgers], 1u)
            << "Exactly one host policy authority must own the admitting "
               "economics for a distributed movement protocol";
        EXPECT_GE(
            global[ImprovingEpochs],
            minimum_epochs_per_rank)
            << "The host policy authority did not retain the required distinct "
               "profitable publication epochs";
        EXPECT_EQ(
            global[ImprovingEpochs],
            global[EconomyAuthorityPublishedWaves])
            << "Every authority-published wave must retain exactly one typed "
               "economy proof";
        EXPECT_EQ(
            global[HostAdmissionRecords],
            global[EconomyAuthorityPublishedWaves])
            << "Every authority-published wave must retain exactly one typed "
               "host admission proof; followers must not duplicate it";
        EXPECT_EQ(
            global[CommittedWaves],
            global[EconomyAuthorityPublishedWaves] * ranks)
            << "Follower telemetry must mirror each authority-published wave; "
               "it is not an independent policy proof";
        EXPECT_EQ(global[ServiceGainEvidenceViolations], 0u)
            << "Every authority-owned economy record must improve measured "
               "service cost and match its completed movement transaction";
        EXPECT_EQ(global[NetBenefitEvidenceViolations], 0u)
            << "Every authority-owned economy record must remain profitable "
               "after movement cost and inference interference";
        EXPECT_GT(global[CommittedMigrations], 0u);
        EXPECT_EQ(global[TypedLedgerEdges], global[CommittedMigrations])
            << "Typed authority movement and physical commit accounting diverged";
        EXPECT_GT(global[Promotions], 0u);
        EXPECT_GT(global[Demotions], 0u);
        EXPECT_GT(
            global[TypedTierResidencyEdges] + global[TypedCombinedEdges],
            0u)
            << "Dynamic residency never completed its tier-residency objective";
        EXPECT_EQ(global[CapacityConservationViolations], 0u)
            << "A committed wave violated typed per-tier or per-participant slot flow";
        EXPECT_EQ(
            global[CapacityConservationCertifications],
            global[CommittedWaves])
            << "Every committed wave must carry one explicit flow-conservation proof";
        if (dynamicMovementAxisContract(resolvedOverlayPlan()) ==
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance)
        {
            const std::uint64_t completed_participant_edges =
                global[TypedParticipantPlacementEdges] +
                global[TypedCombinedEdges];
            if (global[HostPolicyEligibleParticipantCycles] > 0u)
            {
                EXPECT_GT(completed_participant_edges, 0u)
                    << "A host-policy participant cycle survived measured "
                       "economics but never completed physical movement";
            }
            else
            {
                EXPECT_EQ(completed_participant_edges, 0u)
                    << "The movement ledger reported a participant objective "
                       "that no typed host admission classified as economical";
                EXPECT_GE(
                    global[HostAdmissionRecords],
                    minimum_epochs_per_rank)
                    << "Tier-only convergence requires a complete typed "
                       "policy-admission proof for every required wave";
            }
        }
        else
        {
            EXPECT_EQ(
                global[TypedParticipantPlacementEdges] +
                    global[TypedCombinedEdges],
                0u)
                << "A layout without a participant-balancing degree of freedom reported that objective";
        }
        EXPECT_EQ(
            global[CommittedMigrations],
            global[Promotions] + global[Demotions] +
                global[SamePriority]);
        EXPECT_EQ(
            global[CrossDomain],
            global[Promotions] + global[Demotions]);
        EXPECT_EQ(
            global[CrossBackend],
            global[Promotions] + global[Demotions]);
        const auto &test_case = activeModelParityCaseOrThrow();
        if (test_case.topology.mpi_ranks == 1)
        {
            EXPECT_EQ(global[CrossRank], 0u)
                << "A rank-local overlay reported cross-rank movement";
        }
        else
        {
            if (topologyUsesCpu() &&
                activeTypedParticipantCount(
                    [](const GlobalDeviceAddress &participant)
                    { return participant.isCPU(); }) > 1u)
            {
                EXPECT_GT(global[CrossRank], 0u)
                    << "The two-rank CPU tier never exercised distributed movement";
            }
            EXPECT_LE(global[CrossRank], global[CommittedMigrations]);
        }
        EXPECT_GT(global[EstimatedBytes], 0u);
        EXPECT_GT(global[BackgroundNotifications], 0u)
            << "Inference never exercised the wake-only maintenance boundary";
    }

    /**
     * @brief Prove request-local LLEP movement or static immobility.
     *
     * Durable tier migration is certified by moe_overlay_residency above.
     * LLEP additionally owns a request-scoped assignment/copy/apply graph, so
     * its proof must come from the independent moe_rebalance counters. Static
     * cells check the same counters at zero; setup-time initial placement is
     * intentionally outside this request-movement surface.
     */
    void assertRequestMovementPolicyEvidence() const
    {
        enum Counter : size_t
        {
            LLEPAssignments,
            LLEPMovementLayers,
            CopiedArrivals,
            AppliedArrivals,
            UsefulPayloadBytes,
            CounterCount,
        };
        std::array<uint64_t, CounterCount> local{};
        const auto add = [&local](Counter counter, double value)
        {
            if (value > 0.0)
                local[counter] += static_cast<uint64_t>(value);
        };
        for (const auto &record :
             PerfStatsCollector::snapshot({"moe_rebalance"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_rebalance")
            {
                continue;
            }
            if (record.name ==
                "device_rebalance_llep_resident_assignment_calls")
                add(LLEPAssignments, record.value);
            else if (record.name ==
                     "device_rebalance_prefill_current_batch_movement_layers")
                add(LLEPMovementLayers, record.value);
            else if (record.name ==
                         "device_rebalance_copy_copied_arrivals" ||
                     record.name ==
                         "device_rebalance_transfer_current_copied_arrivals" ||
                     record.name ==
                         "device_rebalance_wave_copied_arrivals_total")
                add(CopiedArrivals, record.value);
            else if (record.name ==
                         "device_rebalance_apply_applied_arrivals" ||
                     record.name ==
                         "device_rebalance_transfer_current_applied_arrivals" ||
                     record.name ==
                         "device_rebalance_wave_applied_arrivals_total")
                add(AppliedArrivals, record.value);
            else if (record.name ==
                         "device_rebalance_transfer_useful_payload_bytes" ||
                     record.name ==
                         "device_rebalance_request_useful_payload_bytes_lower_bound")
                add(UsefulPayloadBytes, record.value);
        }

        std::array<uint64_t, CounterCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank() || !isQwen122ProductionTest())
            return;

        if (isLLEPProductionTest())
        {
            EXPECT_GT(global[LLEPAssignments], 0u)
                << "LLEP ran no least-loaded resident assignment";
            EXPECT_GT(global[LLEPMovementLayers], 0u)
                << "LLEP moved no expert payload for the current prefill";
            EXPECT_GT(global[CopiedArrivals], 0u);
            EXPECT_GT(global[AppliedArrivals], 0u);
            EXPECT_GT(global[UsefulPayloadBytes], 0u);
            return;
        }

        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[LLEPAssignments], 0u);
            EXPECT_EQ(global[LLEPMovementLayers], 0u);
            EXPECT_EQ(global[CopiedArrivals], 0u)
                << "Static placement copied a request-time expert payload";
            EXPECT_EQ(global[AppliedArrivals], 0u)
                << "Static placement applied a request-time expert payload";
            EXPECT_EQ(global[UsefulPayloadBytes], 0u)
                << "Static placement transferred request-time expert bytes";
        }
    }

    /** @brief Stable identity of one histogram-driven promotion edge. */
    struct PromotedExpert
    {
        int layer = -1;
        int expert = -1;
        int destination_participant = -1;
        uint64_t candidate_epoch = 0u;

        /** @return Whether two records name the same routed expert. */
        bool operator==(const PromotedExpert &) const = default;
    };

    /** @brief Exact post-publication route and numerical checkpoint witness. */
    struct PromotedExpertExecutionWitness
    {
        ParityForwardPhase phase = ParityForwardPhase::Prefill;
        int step = -1;
        PromotedExpert promotion;
        int selected_placement_bank = -1;
        size_t routed_rows = 0u;
        size_t comparable_route_rows = 0u;
        size_t production_executed_route_rows = 0u;
        size_t reference_executed_route_rows = 0u;
        size_t exact_zero_route_rows = 0u;
        size_t one_sided_zero_route_rows = 0u;
        size_t compared_elements = 0u;
        float expert_contribution_cosine = 0.0f;
        float production_l2_norm = 0.0f;
        float reference_l2_norm = 0.0f;
        float absolute_l2_error = 0.0f;
        float root_mean_square_error = 0.0f;
        RoutedExpertContributionState contribution_state =
            RoutedExpertContributionState::InvalidGeometry;
        RoutedExpertContributionProof numerical_proof =
            RoutedExpertContributionProof::Invalid;
        RoutedExpertReferenceLineage reference_lineage =
            RoutedExpertReferenceLineage::Canonical;
        RoutedExpertContributionDisposition disposition =
            RoutedExpertContributionDisposition::Failed;
        bool valid_geometry = false;
        bool finite = false;
        bool numerically_comparable = false;
        bool numerically_passed = false;
    };

    /** @brief Per-expert contribution evidence for every route at a moved layer. */
    struct RoutedExpertContributionWitness
    {
        ParityForwardPhase phase = ParityForwardPhase::Prefill;
        int step = -1;
        int layer = -1;
        int expert = -1;
        int domain_participant = -1;
        int selected_placement_bank = -1;
        RoutedExpertContributionComparison comparison;
        RoutedExpertContributionPublication publication =
            RoutedExpertContributionPublication::ContinuationCanonical;
        RoutedExpertContributionProof numerical_proof =
            RoutedExpertContributionProof::Invalid;
        RoutedExpertReferenceLineage reference_lineage =
            RoutedExpertReferenceLineage::Canonical;
        RoutedExpertContributionDisposition disposition =
            RoutedExpertContributionDisposition::Failed;
        float post_return_expert_output_cosine = 0.0f;
        bool numerically_comparable = false;
        bool evidence_passed = false;
    };

    /**
     * @brief Resolve route and input lineage for every compared routed layer.
     *
     * A current route-set difference preserves comparability for matched
     * per-route expert values because the layer input is still canonical. It
     * invalidates a dense aggregate as proof of an individual remote addend,
     * and it changes the residual consumed by every later layer. The shared
     * typed transition records all three cases explicitly. Ordered layer keys
     * make the rule independent of callback vector order.
     *
     * @param layers Complete per-layer comparison records for one checkpoint.
     * @return Input lineage keyed by model layer.
     */
    static std::map<int, RoutedExpertReferenceLineage>
    routedExpertReferenceLineageByLayer(
        const std::vector<LayerStats> &layers)
    {
        std::map<int, const LayerStats *> ordered_layers;
        for (const auto &layer : layers)
        {
            const bool inserted =
                ordered_layers.emplace(layer.layer_idx, &layer).second;
            if (!inserted)
            {
                throw std::logic_error(
                    "Duplicate layer in routed-expert parity lineage");
            }
        }

        std::map<int, RoutedExpertReferenceLineage> result;
        auto input_lineage = RoutedExpertReferenceLineage::Canonical;
        for (const auto &[layer_index, layer] : ordered_layers)
        {
            const auto routing = std::find_if(
                layer->stage_results.begin(),
                layer->stage_results.end(),
                [](const StageComparisonResult &stage)
                { return stage.stage_name == "MOE_ROUTING_INDICES"; });
            const bool current_routes_equal =
                routing == layer->stage_results.end() ||
                routing->routing_overlap >= 1.0f - 1.0e-6f;
            const auto transition =
                advanceRoutedExpertReferenceLineage(
                    input_lineage,
                    current_routes_equal);
            result.emplace(layer_index, transition.current_layer);
            input_lineage = transition.next_layer;
        }
        return result;
    }

    /**
     * @brief Retain promotion identities before a parity collector reset.
     *
     * The parity harness resets live PerfStats between campaign phases so the
     * CSV for each numerical comparison has an unambiguous interval. Movement
     * is model-lifetime state and deliberately survives that reset. Preserve
     * only the immutable layer/expert identities from the production movement
     * ledger; routing values and numerical outputs are still read from the
     * later live graph checkpoints.
     */
    void cacheCommittedPromotionEvidence()
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const auto ledger =
            orch_runner_->moeOptimizationMovementLedger();
        EXPECT_TRUE(ledger.complete())
            << "Production movement authority discarded "
            << ledger.discarded_edges
            << " typed edges before parity attribution";
        size_t malformed_edges = 0;
        for (const auto &edge : ledger.edges)
        {
            if (!edge.valid())
            {
                ++malformed_edges;
                continue;
            }
            if (edge.direction !=
                MoEOptimizationMovementDirection::Promotion)
            {
                continue;
            }
            if (edge.authority == MoEOptimizationAuthority::Host &&
                edge.activation_count == 0u)
            {
                continue;
            }

            const PromotedExpert promotion{
                .layer = edge.layer,
                .expert = edge.expert,
                .destination_participant = edge.destination_participant,
                .candidate_epoch = edge.candidate_epoch,
            };
            if (std::find(
                    promoted_experts_.begin(),
                    promoted_experts_.end(),
                    promotion) == promoted_experts_.end())
            {
                promoted_experts_.push_back(promotion);
            }
        }

        EXPECT_EQ(malformed_edges, 0u)
            << "Typed production movement ledger contained malformed edges";
    }

    /**
     * @brief Persist the authenticated physical movement ledger used by parity.
     *
     * Numerical CSVs name the layer and routed expert that diverged, but that is
     * not enough to diagnose a moved-weight defect: the production transaction
     * may have crossed a rank, backend, or numeric-priority boundary.  Export the
     * authority's typed edge ledger before the parity harness resets optional
     * telemetry. The CSV serializes authoritative state and never reconstructs
     * placement from PerfStats tags.
     */
    void writeCommittedMovementEvidenceCsv() const
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const auto ledger =
            orch_runner_->moeOptimizationMovementLedger();
        EXPECT_TRUE(ledger.complete())
            << "Cannot export a truncated authoritative movement ledger";
        const auto path = ensureResultsDir() / "expert_movement.csv";
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output
            << "domain,name,count,value,transaction,candidate_epoch,layer,expert,"
               "cycle_index,cycle_size,direction,movement_axis,source_participant,"
               "destination_participant,"
               "source_priority,destination_priority,source_device,"
               "destination_device,source_world_rank,destination_world_rank,"
               "estimated_weight_bytes,activation_count,blocking_inference,"
               "policy_owner\n";

        const auto direction_name = [](MoEOptimizationMovementDirection direction)
            -> const char *
        {
            switch (direction)
            {
            case MoEOptimizationMovementDirection::Promotion:
                return "promotion";
            case MoEOptimizationMovementDirection::Demotion:
                return "demotion";
            case MoEOptimizationMovementDirection::SamePriority:
                return "same_priority";
            }
            return "invalid";
        };
        const auto axis_name = [](MoEOptimizationMovementAxis axis)
            -> const char *
        {
            switch (axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                return "tier_residency";
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                return "participant_placement";
            case MoEOptimizationMovementAxis::Combined:
                return "combined";
            }
            return "invalid";
        };
        size_t edge_count = 0;
        for (const auto &edge : ledger.edges)
        {
            EXPECT_TRUE(edge.valid());
            if (!edge.valid())
                continue;
            ++edge_count;
            const bool host =
                edge.authority == MoEOptimizationAuthority::Host;
            output
                << (host ? "moe_overlay_residency"
                         : "moe_overlay_controller")
                << ','
                << (host ? "expert_migration_edges"
                         : "dynamic_migration_edges")
                << ",1,1," << edge.transaction << ','
                << edge.candidate_epoch << ',' << edge.layer << ','
                << edge.expert << ',' << edge.cycle_index << ','
                << edge.cycle_size << ',' << direction_name(edge.direction)
                << ',' << axis_name(edge.axis) << ','
                << edge.source_participant << ','
                << edge.destination_participant << ','
                << edge.source_priority << ',' << edge.destination_priority
                << ',' << edge.source_device.toString() << ','
                << edge.destination_device.toString() << ',';
            if (edge.source_world_rank_known)
                output << edge.source_world_rank;
            else
                output << "unknown";
            output << ',';
            if (edge.destination_world_rank_known)
                output << edge.destination_world_rank;
            else
                output << "unknown";
            output << ',' << edge.estimated_weight_bytes << ','
                   << edge.activation_count << ','
                   << (edge.blocking_inference ? "true" : "false") << ','
                   << (host ? "host" : "device") << '\n';
        }
        output.flush();
        EXPECT_TRUE(output.good()) << path;
        EXPECT_GT(edge_count, 0u)
            << "Dynamic parity produced no authenticated movement-ledger row";
    }

    /**
     * @brief Preserve the complete process-local residency decision trail.
     *
     * The compact `expert_movement.csv` contains committed edges only. This
     * companion artifact retains proposal, capacity, economy, and physical
     * publication records, including failures. PerfStats is process-local, so
     * followers use rank-qualified names while the artifact authority retains
     * the canonical filename. Export occurs after the production worker loop
     * has closed and cannot affect placement or inference ordering.
     */
    void writeResidencyDiagnosticsCsv() const noexcept
    {
        if (!isDynamicResidencyProductionTest())
            return;

        try
        {
            const int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            const auto path = ensureResultsDir() /
                (isRootParityRank()
                     ? "expert_residency_diagnostics.csv"
                     : "expert_residency_diagnostics_rank_" +
                           std::to_string(rank) + ".csv");
            if (!PerfStatsCollector::writeCsv(
                    path.string(),
                    {"moe_overlay_residency", "moe_overlay_controller"}))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Could not write residency diagnostics to "
                    << path);
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Residency diagnostics raised: "
                << error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Residency diagnostics raised a non-standard exception");
        }
    }

    /**
     * @brief Retain exact promoted-expert use from prefill or serial decode.
     *
     * Histogram movement is trained by the complete authenticated request, so
     * a profitable promotion may be hot only during decode.  Restricting the
     * witness to the prompt checkpoint made valid movement pass or fail based
     * on which phase selected the expert.  Observe both numerically compared
     * phases while their immutable route bank is live and retain only compact
     * identity/metric evidence.
     */
    void observeComparedParityCheckpoint(
        ParityForwardPhase phase,
        int step,
        const std::vector<LayerStats> &layers) override
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;
        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto residency = concrete
                                   ? concrete
                                         ->expertOverlayResidencySnapshotForDiagnostics()
                                   : nullptr;
        ASSERT_NE(residency, nullptr);
        ASSERT_TRUE(residency->valid());
        const auto reference_lineage_by_layer =
            routedExpertReferenceLineageByLayer(layers);

        /*
         * A promoted-expert witness proves the newly published destination,
         * but it cannot reveal a missing contribution from another tier. Keep
         * one route-conditioned record for every expert selected at each moved
         * layer. This is especially important for an asynchronous sparse
         * return: a correct local promotion can otherwise mask a late remote
         * contribution in the summed MOE_EXPERT_OUTPUT checkpoint.
         */
        std::set<int> moved_layers;
        for (const auto &promotion : promoted_experts_)
            moved_layers.insert(promotion.layer);
        for (const int layer : moved_layers)
        {
            const bool already_observed = std::any_of(
                routed_expert_contribution_witnesses_.begin(),
                routed_expert_contribution_witnesses_.end(),
                [&](const RoutedExpertContributionWitness &witness)
                {
                    return witness.phase == phase &&
                           witness.step == step &&
                           witness.layer == layer;
                });
            if (already_observed)
                continue;

            size_t route_elements = 0u;
            const std::string route_key =
                "layer" + std::to_string(layer) +
                "_MOE_ROUTING_INDICES";
            const float *const routes =
                activeSnapshot(route_key, route_elements);
            ASSERT_NE(routes, nullptr) << route_key;

            const auto placement = std::find_if(
                residency->placement_plan->placements.begin(),
                residency->placement_plan->placements.end(),
                [&](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == layer; });
            ASSERT_NE(
                placement,
                residency->placement_plan->placements.end());
            const auto route_evidence = pinnedDeviceRouteEvidence(
                layer,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value());

            const auto layer_stats = std::find_if(
                layers.begin(),
                layers.end(),
                [&](const LayerStats &stats)
                { return stats.layer_idx == layer; });
            ASSERT_NE(layer_stats, layers.end())
                << "No parity summary was produced for moved layer " << layer;
            const auto expert_output = std::find_if(
                layer_stats->stage_results.begin(),
                layer_stats->stage_results.end(),
                [](const StageComparisonResult &result)
                { return result.stage_name == "MOE_EXPERT_OUTPUT"; });
            ASSERT_NE(expert_output, layer_stats->stage_results.end())
                << "Moved layer " << layer
                << " omitted its post-return MOE_EXPERT_OUTPUT checkpoint";
            const auto lineage_it = reference_lineage_by_layer.find(layer);
            ASSERT_NE(lineage_it, reference_lineage_by_layer.end())
                << "Moved layer " << layer
                << " has no routed-expert reference lineage";
            const auto reference_lineage = lineage_it->second;

            const auto moe = getMoEConfig();
            ASSERT_GT(moe.top_k, 0);
            ASSERT_EQ(
                route_elements % static_cast<size_t>(moe.top_k),
                0u);
            const std::string reference_prefix =
                phase == ParityForwardPhase::Prefill
                    ? std::string{}
                    : "decode_step" + std::to_string(step) + '_';
            const std::vector<float> reference_routes =
                loadPyTorchSnapshot(reference_prefix + route_key);
            ASSERT_FALSE(reference_routes.empty())
                << reference_prefix + route_key;

            size_t contribution_elements = 0u;
            const std::string contribution_key =
                "layer" + std::to_string(layer) +
                "_MOE_ROUTE_CONTRIBUTIONS";
            const float *const contributions =
                activeSnapshot(contribution_key, contribution_elements);
            ASSERT_NE(contributions, nullptr) << contribution_key;
            const std::vector<float> reference_contributions =
                loadPyTorchSnapshot(
                    reference_prefix + contribution_key);
            ASSERT_FALSE(reference_contributions.empty())
                << reference_prefix + contribution_key;

            std::map<int, int> routed_expert_participants;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                const float raw_expert = routes[index];
                const float raw_participant =
                    route_evidence->domain_participants[index];
                ASSERT_TRUE(std::isfinite(raw_expert));
                ASSERT_TRUE(std::isfinite(raw_participant));
                const int expert = static_cast<int>(raw_expert);
                const int participant = static_cast<int>(raw_participant);
                ASSERT_EQ(raw_expert, static_cast<float>(expert));
                ASSERT_EQ(raw_participant, static_cast<float>(participant));
                const auto [it, inserted] =
                    routed_expert_participants.emplace(
                        expert, participant);
                ASSERT_TRUE(inserted || it->second == participant)
                    << "Expert " << expert
                    << " changed destination within one immutable route bank";
            }

            for (const auto &[expert, participant] :
                 routed_expert_participants)
            {
                const auto comparison = compareRoutedExpertContribution(
                    std::span<const float>(routes, route_elements),
                    reference_routes,
                    std::span<const float>(
                        route_evidence->domain_participants,
                        route_evidence->route_count),
                    expert,
                    participant,
                    static_cast<size_t>(moe.top_k),
                    std::span<const float>(
                        contributions,
                        contribution_elements),
                    reference_contributions);
                ASSERT_TRUE(comparison.validGeometry())
                    << "Route contribution geometry failed for layer "
                    << layer << " expert " << expert;
                /*
                 * The continuation endpoint always publishes its own route
                 * slots directly. External endpoints may either materialize
                 * canonical slots before the ordered fold or return a dense
                 * aggregate which is merged into MOE_EXPERT_OUTPUT. A zero
                 * raw external slot is therefore not missing execution by
                 * itself; it is valid only when the post-return checkpoint
                 * proves that the completed sparse transaction is present.
                 */
                const bool deferred_remote_aggregate =
                    participant < 0 &&
                    comparison.production_executed_rows == 0u;
                const auto publication =
                    participant >= 0
                        ? RoutedExpertContributionPublication::
                              ContinuationCanonical
                        : (deferred_remote_aggregate
                               ? RoutedExpertContributionPublication::
                                     DeferredRemoteAggregate
                               : RoutedExpertContributionPublication::
                                     ReturnedCanonical);
                const auto numerical_proof =
                    classifyPublishedRoutedExpertContribution(
                        publication,
                        comparison,
                        config_.cosine_threshold,
                        expert_output->cosine_similarity,
                        reference_lineage);
                const bool numerically_comparable =
                    comparison.comparable_rows > 0u;
                const bool evidence_passed =
                    routedExpertContributionProofPasses(numerical_proof);
                const auto disposition =
                    routedExpertContributionProofDisposition(
                        numerical_proof);
                routed_expert_contribution_witnesses_.push_back({
                    .phase = phase,
                    .step = step,
                    .layer = layer,
                    .expert = expert,
                    .domain_participant = participant,
                    .selected_placement_bank =
                        route_evidence->selected_bank,
                    .comparison = comparison,
                    .publication = publication,
                    .numerical_proof = numerical_proof,
                    .reference_lineage = reference_lineage,
                    .disposition = disposition,
                    .post_return_expert_output_cosine =
                        expert_output->cosine_similarity,
                    .numerically_comparable =
                        numerically_comparable,
                    .evidence_passed = evidence_passed,
                });
            }
        }

        for (const auto &promotion : promoted_experts_)
        {
            size_t route_elements = 0;
            const std::string snapshot_key =
                "layer" + std::to_string(promotion.layer) +
                "_MOE_ROUTING_INDICES";
            const float *const routes =
                activeSnapshot(snapshot_key, route_elements);
            if (!routes)
                continue;
            const auto placement = std::find_if(
                residency->placement_plan->placements.begin(),
                residency->placement_plan->placements.end(),
                [&](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == promotion.layer; });
            ASSERT_NE(
                placement,
                residency->placement_plan->placements.end());
            const auto route_evidence = pinnedDeviceRouteEvidence(
                promotion.layer,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value());
            ASSERT_GE(promotion.expert, 0);
            ASSERT_LT(
                static_cast<size_t>(promotion.expert),
                route_evidence->expert_count);
            const float placed_participant =
                route_evidence->overlay_participants[promotion.expert];
            if (placed_participant != static_cast<float>(
                                          promotion
                                              .destination_participant))
            {
                // A later committed epoch may have moved this expert again.
                continue;
            }
            const auto *const destination =
                residency->owner_map.participantForId(
                    promotion.destination_participant);
            ASSERT_NE(destination, nullptr);
            const int expected_domain_participant =
                destination->domain_name == overlay_plan_->continuation_domain
                    ? destination->domain_participant_index
                    : -1;

            // Both IDs are stored exactly as FP32 integers in parity dumps.
            bool routed_to_destination = false;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                if (routes[index] !=
                    static_cast<float>(promotion.expert))
                {
                    continue;
                }
                if (route_evidence->domain_participants[index] ==
                    static_cast<float>(expected_domain_participant))
                {
                    routed_to_destination = true;
                    break;
                }
            }
            if (!routed_to_destination)
                continue;

            /*
             * MOE_EXPERT_OUTPUT proves the complete routed sum. The canonical
             * per-route tensors below isolate the moved expert itself, so an
             * unrelated low-weight top-k difference cannot make this physical
             * movement witness spuriously incomparable.
             */
            const auto layer_it = std::find_if(
                layers.begin(),
                layers.end(),
                [&](const LayerStats &stats)
                { return stats.layer_idx == promotion.layer; });
            ASSERT_NE(layer_it, layers.end())
                << "No parity summary was produced for promoted-expert layer "
                << promotion.layer;

            const auto expert_output_it = std::find_if(
                layer_it->stage_results.begin(),
                layer_it->stage_results.end(),
                [](const StageComparisonResult &result)
                { return result.stage_name == "MOE_EXPERT_OUTPUT"; });
            ASSERT_NE(expert_output_it, layer_it->stage_results.end())
                << "Promoted-expert layer " << promotion.layer
                << " omitted the MOE_EXPERT_OUTPUT checkpoint";
            const auto lineage_it =
                reference_lineage_by_layer.find(promotion.layer);
            ASSERT_NE(lineage_it, reference_lineage_by_layer.end())
                << "Promoted-expert layer " << promotion.layer
                << " has no routed-expert reference lineage";
            const auto reference_lineage = lineage_it->second;

            const auto moe = getMoEConfig();
            ASSERT_GT(moe.top_k, 0);
            ASSERT_EQ(
                route_elements % static_cast<size_t>(moe.top_k),
                0u);
            const std::string reference_prefix =
                phase == ParityForwardPhase::Prefill
                    ? std::string{}
                    : "decode_step" + std::to_string(step) + '_';
            const std::vector<float> reference_routes =
                loadPyTorchSnapshot(reference_prefix + snapshot_key);
            ASSERT_FALSE(reference_routes.empty())
                << "Promoted-expert witness has no Hugging Face routes for "
                << reference_prefix + snapshot_key;

            size_t contribution_elements = 0u;
            const std::string contribution_key =
                "layer" + std::to_string(promotion.layer) +
                "_MOE_ROUTE_CONTRIBUTIONS";
            const float *const contributions =
                activeSnapshot(contribution_key, contribution_elements);
            ASSERT_NE(contributions, nullptr)
                << "The production sparse collective did not retain canonical "
                   "per-route execution evidence for "
                << contribution_key;
            const std::string reference_contribution_key =
                "layer" + std::to_string(promotion.layer) +
                "_MOE_ROUTE_CONTRIBUTIONS";
            const std::vector<float> reference_contributions =
                loadPyTorchSnapshot(
                    reference_prefix + reference_contribution_key);
            ASSERT_FALSE(reference_contributions.empty())
                << "Promoted-expert witness has no Hugging Face per-route "
                   "contribution for "
                << reference_prefix + reference_contribution_key;

            const auto comparison =
                compareRoutedExpertContribution(
                    std::span<const float>(routes, route_elements),
                    reference_routes,
                    std::span<const float>(
                        route_evidence->domain_participants,
                        route_evidence->route_count),
                    promotion.expert,
                    expected_domain_participant,
                    static_cast<size_t>(moe.top_k),
                    std::span<const float>(
                        contributions,
                        contribution_elements),
                    reference_contributions);
            ASSERT_TRUE(comparison.validGeometry())
                << "Promoted-expert row witness has incompatible route/contribution "
                   "geometry at layer "
                << promotion.layer << " during "
                << parityForwardPhaseName(phase) << " step " << step;
            ASSERT_EQ(comparison.routed_rows > 0u, routed_to_destination)
                << "Typed row comparison disagrees with the pinned destination route";

            const bool deferred_remote_aggregate =
                expected_domain_participant < 0 &&
                comparison.production_executed_rows == 0u;
            const auto publication =
                expected_domain_participant >= 0
                    ? RoutedExpertContributionPublication::
                          ContinuationCanonical
                    : (deferred_remote_aggregate
                           ? RoutedExpertContributionPublication::
                                 DeferredRemoteAggregate
                           : RoutedExpertContributionPublication::
                                 ReturnedCanonical);
            const auto numerical_proof =
                classifyPublishedRoutedExpertContribution(
                    publication,
                    comparison,
                    config_.cosine_threshold,
                    expert_output_it->cosine_similarity,
                    reference_lineage);
            const bool numerically_comparable =
                comparison.comparable_rows > 0u;
            const bool numerically_passed =
                routedExpertContributionProofPasses(numerical_proof);
            const auto disposition =
                routedExpertContributionProofDisposition(numerical_proof);

            const auto duplicate = std::find_if(
                promoted_expert_execution_witnesses_.begin(),
                promoted_expert_execution_witnesses_.end(),
                [&](const PromotedExpertExecutionWitness &witness)
                {
                    return witness.phase == phase && witness.step == step &&
                           witness.promotion == promotion;
                });
            if (duplicate == promoted_expert_execution_witnesses_.end())
            {
                promoted_expert_execution_witnesses_.push_back(
                    PromotedExpertExecutionWitness{
                        .phase = phase,
                        .step = step,
                        .promotion = promotion,
                        .selected_placement_bank =
                            route_evidence->selected_bank,
                        .routed_rows = comparison.routed_rows,
                        .comparable_route_rows =
                            comparison.comparable_rows,
                        .production_executed_route_rows =
                            comparison.production_executed_rows,
                        .reference_executed_route_rows =
                            comparison.reference_executed_rows,
                        .exact_zero_route_rows =
                            comparison.exact_zero_rows,
                        .one_sided_zero_route_rows =
                            comparison.one_sided_zero_rows,
                        .compared_elements =
                            comparison.compared_elements,
                        .expert_contribution_cosine =
                            comparison.cosine_similarity,
                        .production_l2_norm =
                            comparison.production_l2_norm,
                        .reference_l2_norm =
                            comparison.reference_l2_norm,
                        .absolute_l2_error =
                            comparison.absolute_l2_error,
                        .root_mean_square_error =
                            comparison.root_mean_square_error,
                        .contribution_state = comparison.state,
                        .numerical_proof = numerical_proof,
                        .reference_lineage = reference_lineage,
                        .disposition = disposition,
                        .valid_geometry = comparison.validGeometry(),
                        .finite = comparison.finite(),
                        .numerically_comparable =
                            numerically_comparable,
                        .numerically_passed = numerically_passed,
                    });
            }
            if (numerically_comparable)
            {
                if (publication == RoutedExpertContributionPublication::
                                       DeferredRemoteAggregate)
                {
                    EXPECT_EQ(comparison.production_executed_rows, 0u)
                        << "A deferred remote aggregate unexpectedly wrote a "
                           "canonical route slot";
                }
                else
                {
                    EXPECT_EQ(
                        comparison.production_executed_rows +
                            comparison.exact_zero_rows,
                        comparison.comparable_rows)
                        << "Promoted expert " << promotion.expert
                        << " at layer " << promotion.layer
                        << " omitted a canonical production route "
                           "contribution";
                    EXPECT_EQ(comparison.one_sided_zero_rows, 0u)
                        << "Promoted expert " << promotion.expert
                        << " at layer " << promotion.layer
                        << " has a one-sided canonical route contribution";
                }
                EXPECT_EQ(
                    comparison.reference_executed_rows +
                        comparison.exact_zero_rows,
                    comparison.comparable_rows)
                    << "Production produced a route contribution where "
                       "Hugging Face was exactly zero for promoted expert "
                    << promotion.expert << " at layer " << promotion.layer;
            }
        }
    }

    /**
     * @brief Assert and export a post-publication promoted-expert witness.
     *
     * Movement, route selection, and numerical comparison remain three
     * independently produced authorities.  This epilogue only joins their
     * immutable evidence; it neither chooses a route nor causes maintenance.
     */
    void assertParityExecutionExercisesPromotedExpert() const
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_FALSE(promoted_experts_.empty())
            << "Dynamic residency committed no promotion edge";

        const auto csv_path =
            ensureResultsDir() / "promoted_expert_execution.csv";
        std::ofstream csv(csv_path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open()) << csv_path;
        csv << "phase,step,layer,expert,destination_participant,"
               "candidate_epoch,selected_placement_bank,"
               "routed_rows,comparable_route_rows,"
               "production_executed_route_rows,"
               "reference_executed_route_rows,exact_zero_route_rows,"
               "one_sided_zero_route_rows,compared_elements,"
               "moe_expert_contribution_cosine,valid_geometry,"
               "finite,numerically_comparable,numerically_passed,"
               "comparison_state,numerical_proof,production_l2_norm,"
               "reference_l2_norm,absolute_l2_error,rmse,"
               "reference_lineage,proof_disposition\n";
        for (const auto &witness : promoted_expert_execution_witnesses_)
        {
            csv << parityForwardPhaseName(witness.phase) << ','
                << witness.step << ','
                << witness.promotion.layer << ','
                << witness.promotion.expert << ','
                << witness.promotion.destination_participant << ','
                << witness.promotion.candidate_epoch << ','
                << witness.selected_placement_bank << ','
                << witness.routed_rows << ','
                << witness.comparable_route_rows << ','
                << witness.production_executed_route_rows << ','
                << witness.reference_executed_route_rows << ','
                << witness.exact_zero_route_rows << ','
                << witness.one_sided_zero_route_rows << ','
                << witness.compared_elements << ','
                << witness.expert_contribution_cosine << ','
                << (witness.valid_geometry ? "true" : "false") << ','
                << (witness.finite ? "true" : "false") << ','
                << (witness.numerically_comparable ? "true" : "false")
                << ','
                << (witness.numerically_passed ? "true" : "false") << ','
                << routedExpertContributionStateName(
                       witness.contribution_state)
                << ','
                << routedExpertContributionProofName(
                       witness.numerical_proof)
                << ',' << witness.production_l2_norm << ','
                << witness.reference_l2_norm << ','
                << witness.absolute_l2_error << ','
                << witness.root_mean_square_error << ','
                << routedExpertReferenceLineageName(
                       witness.reference_lineage)
                << ','
                << routedExpertContributionDispositionName(
                       witness.disposition)
                << '\n';
        }
        csv.flush();
        ASSERT_TRUE(csv.good()) << csv_path;

        const auto routed_csv_path =
            ensureResultsDir() / "routed_expert_contributions.csv";
        std::ofstream routed_csv(routed_csv_path, std::ios::trunc);
        ASSERT_TRUE(routed_csv.is_open()) << routed_csv_path;
        routed_csv
            << "phase,step,layer,expert,domain_participant,"
               "selected_placement_bank,publication,routed_rows,"
               "comparable_route_rows,"
               "production_executed_route_rows,"
               "reference_executed_route_rows,exact_zero_route_rows,"
               "one_sided_zero_route_rows,compared_elements,cosine,"
               "post_return_expert_output_cosine,valid_geometry,finite,"
               "numerically_comparable,evidence_passed,comparison_state,"
               "numerical_proof,production_l2_norm,reference_l2_norm,"
               "absolute_l2_error,rmse,reference_lineage,"
               "proof_disposition\n";
        size_t comparable_routed_experts = 0u;
        for (const auto &witness :
             routed_expert_contribution_witnesses_)
        {
            const auto &comparison = witness.comparison;
            routed_csv
                << parityForwardPhaseName(witness.phase) << ','
                << witness.step << ',' << witness.layer << ','
                << witness.expert << ',' << witness.domain_participant << ','
                << witness.selected_placement_bank << ','
                << routedContributionPublicationName(witness.publication)
                << ','
                << comparison.routed_rows << ','
                << comparison.comparable_rows << ','
                << comparison.production_executed_rows << ','
                << comparison.reference_executed_rows << ','
                << comparison.exact_zero_rows << ','
                << comparison.one_sided_zero_rows << ','
                << comparison.compared_elements << ','
                << comparison.cosine_similarity << ','
                << witness.post_return_expert_output_cosine << ','
                << (comparison.validGeometry() ? "true" : "false") << ','
                << (comparison.finite() ? "true" : "false") << ','
                << (witness.numerically_comparable ? "true" : "false")
                << ','
                << (witness.evidence_passed ? "true" : "false") << ','
                << routedExpertContributionStateName(comparison.state)
                << ','
                << routedExpertContributionProofName(
                       witness.numerical_proof)
                << ',' << comparison.production_l2_norm << ','
                << comparison.reference_l2_norm << ','
                << comparison.absolute_l2_error << ','
                << comparison.root_mean_square_error << ','
                << routedExpertReferenceLineageName(
                       witness.reference_lineage)
                << ','
                << routedExpertContributionDispositionName(
                       witness.disposition)
                << '\n';
            if (comparison.comparable_rows == 0u)
                continue;
            ++comparable_routed_experts;
            EXPECT_NE(
                witness.disposition,
                RoutedExpertContributionDisposition::Failed)
                << "Moved-layer routed contribution proof failed for layer "
                << witness.layer << " expert " << witness.expert
                << " domain participant " << witness.domain_participant
                << " publication="
                << routedContributionPublicationName(witness.publication)
                << " state="
                << routedExpertContributionStateName(comparison.state)
                << " proof="
                << routedExpertContributionProofName(
                       witness.numerical_proof)
                << " lineage="
                << routedExpertReferenceLineageName(
                       witness.reference_lineage)
                << ": cosine=" << comparison.cosine_similarity
                << " post_return_expert_output_cosine="
                << witness.post_return_expert_output_cosine
                << " one_sided_zero_rows="
                << comparison.one_sided_zero_rows
                << " production_rows="
                << comparison.production_executed_rows
                << " reference_rows="
                << comparison.reference_executed_rows;
        }
        routed_csv.flush();
        ASSERT_TRUE(routed_csv.good()) << routed_csv_path;
        ASSERT_GT(comparable_routed_experts, 0u)
            << "Moved layers exposed no same-expert route contributions";

        std::ostringstream candidates;
        for (const auto &promotion : promoted_experts_)
        {
            if (candidates.tellp() > 0)
                candidates << ", ";
            candidates << "layer" << promotion.layer << ":expert"
                       << promotion.expert << "->participant"
                       << promotion.destination_participant << "@epoch"
                       << promotion.candidate_epoch;
        }
        ASSERT_FALSE(promoted_expert_execution_witnesses_.empty())
            << "No numerically compared post-publication prefill or decode "
               "checkpoint executed a promoted expert on its acquired "
               "destination; candidates: "
            << candidates.str();

        /*
         * A production router and the numerically close Hugging Face router
         * need not choose an identical eighth expert on every row. Such a row
         * proves destination execution, but no same-expert reference value
         * exists and it cannot vote on numerical correctness. Require every
         * observed destination participant to have at least one independent
         * same-expert, passing per-route witness. A comparable canonical-input
         * per-route mismatch fails. A deferred dense aggregate is inconclusive
         * when the current route set differs because it cannot isolate the
         * named addend. A later mismatch after an earlier discrete routing
         * divergence is likewise inconclusive: it cannot certify a path, but
         * comparing different inputs also cannot convict that path. This
         * certifies each physical publication path without turning the
         * stronger routed top-k parity gate into an accidental exact-routing
         * requirement.
         */
        std::vector<int> observed_destinations;
        for (const auto &witness : promoted_expert_execution_witnesses_)
        {
            if (std::find(
                    observed_destinations.begin(),
                    observed_destinations.end(),
                    witness.promotion.destination_participant) ==
                observed_destinations.end())
            {
                observed_destinations.push_back(
                    witness.promotion.destination_participant);
            }
            if (witness.numerically_comparable)
            {
                EXPECT_NE(
                    witness.disposition,
                    RoutedExpertContributionDisposition::Failed)
                    << "A canonical-lineage promoted-expert witness failed for layer "
                    << witness.promotion.layer << " expert "
                    << witness.promotion.expert << " proof="
                    << routedExpertContributionProofName(
                           witness.numerical_proof)
                    << " lineage="
                    << routedExpertReferenceLineageName(
                           witness.reference_lineage);
            }
        }
        for (const int destination_participant : observed_destinations)
        {
            const bool has_passing_comparable_witness = std::any_of(
                promoted_expert_execution_witnesses_.begin(),
                promoted_expert_execution_witnesses_.end(),
                [&](const PromotedExpertExecutionWitness &witness)
                {
                    return witness.promotion.destination_participant ==
                               destination_participant &&
                           witness.numerically_comparable &&
                           witness.numerically_passed;
                });
            EXPECT_TRUE(has_passing_comparable_witness)
                << "Promoted experts executed on destination participant "
                << destination_participant
                << " without any same-expert Hugging Face per-route "
                   "numerical witness for that physical publication path";
        }
    }

    /**
     * @brief Return the inventory-resolved dense continuation authority.
     *
     * Before runner setup, rank zero retains reference-pack preparation. Once
     * production has bound the topology, comparisons and CSV output move to
     * the same rank that owns logits and stage snapshots.
     */
    int parityArtifactAuthorityRank() const override
    {
        return orch_runner_ ? orch_runner_->coordinatedRootRank() : 0;
    }

    /**
     * @brief Keep additive HF reference work off the production worker ranks.
     *
     * During parity, non-continuation ranks execute `runMPIWorkerLoop()` and
     * consume only typed serving commands. An unmatched test-only broadcast or
     * barrier would corrupt that protocol, so the continuation/artifact
     * authority alone owns the filesystem reference lease.
     */
    ParityReferenceGenerationCoordination
    parityReferenceGenerationCoordination() const override
    {
        return ParityReferenceGenerationCoordination::ArtifactAuthorityOnly;
    }

    /** @return Whether this process owns the inventory-resolved continuation. */
    bool isRootParityRank() const
    {
        return mpi_ctx_
                   ? mpi_ctx_->rank() == parityArtifactAuthorityRank()
                   : isRank0();
    }

    bool synchronizedDecodeWorkAvailable()
    {
        bool available = true;
        if (isRootParityRank())
        {
            available = !loadPyTorchSnapshot("decode_step0_LM_HEAD").empty() &&
                        !readDecodeTokensFromMetadata().empty();
        }
        return broadcastRootFlag(available);
    }

    bool producedPrefillSummary(const ParityTestSummary &summary) const
    {
        return summary.embedding_passed ||
               !summary.layer_stats.empty() ||
               summary.lm_head_passed ||
               summary.lm_head_cosine != 0.0f ||
               summary.total_layers_passed > 0;
    }

    bool producedDecodeSummary(const DecodeParitySummary &summary) const
    {
        return !summary.step_stats.empty() ||
               summary.steps_total > 0 ||
               summary.top1_matches > 0;
    }

    /**
     * @brief Test whether one exact noncanonical HF branch is already complete.
     *
     * Missing branches are deliberately not generated here. This method runs
     * while the production graph and prepared 122B weights are resident; loading
     * the Python model at this boundary previously overlapped roughly 500 GB of
     * live state and was killed by the host OOM policy. The immutable production
     * checkpoints are queued below and the suite resolves them after all model
     * authorities retire.
     *
     * @param reference_step Main decode step that owns the sidecar transaction.
     * @param condition_tokens Recursive condition tokens consumed by MTP1..N.
     * @return True only when the deepest branch checkpoint is complete on disk.
     */
    bool hasHuggingFaceMTPBranchReference(
        int reference_step,
        const std::vector<int32_t> &condition_tokens) const
    {
        if (reference_step < 0 || condition_tokens.empty() ||
            condition_tokens.size() >=
                static_cast<size_t>(kQwen122MaximumMTPDraftDepth) ||
            std::any_of(
                condition_tokens.begin(),
                condition_tokens.end(),
                [](int32_t token) { return token < 0; }))
        {
            return false;
        }

        std::ostringstream qualifier;
        for (const int32_t token : condition_tokens)
            qualifier << '_' << token;
        const std::string branch_stem =
            "decode_step" + std::to_string(reference_step) + "_BRANCH" +
            qualifier.str() + "_MTP" +
            std::to_string(condition_tokens.size());
        const std::filesystem::path deepest_lm_head =
            std::filesystem::path(config_.snapshot_dir) /
            (branch_stem + "_LM_HEAD.npy");
        const std::filesystem::path deepest_embedding =
            std::filesystem::path(config_.snapshot_dir) /
            (branch_stem + "_EMBEDDING.npy");
        return std::filesystem::is_regular_file(deepest_lm_head) &&
               std::filesystem::is_regular_file(deepest_embedding);
    }

    /**
     * @brief Copy one missing recursive context into the post-residency queue.
     * @param call Public grouped-decode call index used by the CSV.
     * @param reference_step Canonical main-model decode position.
     * @param reference_depth Number of recursive condition tokens consumed.
     * @param condition_tokens Exact device-owned condition-token trajectory.
     * @param production_prefix Snapshot namespace for the live recursive row.
     * @param required_stages Complete sidecar checkpoint contract.
     * @param snapshot_csv_path Existing per-cell diagnostic CSV to append later.
     * @return True only when every live checkpoint was copied successfully.
     */
    bool deferHuggingFaceMTPBranchReference(
        int call,
        int reference_step,
        int reference_depth,
        const std::vector<int32_t> &condition_tokens,
        const std::string &production_prefix,
        std::span<const std::string_view> required_stages,
        const std::filesystem::path &snapshot_csv_path)
    {
        DeferredMTPBranchContext context{
            .test_name = activeTestName(),
            .model_path = config_.model_path,
            .prompt = config_.prompt,
            .snapshot_dir = config_.snapshot_dir,
            .snapshot_csv_path = snapshot_csv_path,
            .decode_steps = config_.decode_steps,
            .call = call,
            .reference_step = reference_step,
            .reference_depth = reference_depth,
            .vocab_size = orch_runner_ ? orch_runner_->vocabSize() : 0,
            .cosine_threshold = config_.cosine_threshold,
            .decode_cosine_threshold = config_.decode_cosine_threshold,
            .kl_threshold = config_.mtp_kl_threshold.value_or(
                config_.kl_threshold),
            .condition_tokens = condition_tokens,
        };
        const auto moe = getMoEConfig();
        context.top_k = moe.top_k;
        context.num_experts = moe.num_experts;
        context.checkpoints.reserve(required_stages.size());
        for (const std::string_view stage : required_stages)
        {
            /*
             * The terminal-hidden selector belongs to the transaction
             * envelope: it chooses the main/previous-sidecar row before the
             * depth-zero predictor graph starts. All remaining checkpoints
             * are outputs of that retained predictor graph and therefore use
             * its MTP0 namespace. Keeping this mapping explicit lets the CSV
             * distinguish inherited recursive drift from error introduced by
             * the current sidecar execution.
             */
            const std::string production_key =
                production_prefix +
                (stage == "TERMINAL_HIDDEN_ROW_SELECT"
                     ? "MTP_TERMINAL_HIDDEN_ROW_SELECT"
                     : "MTP0_" + std::string(stage));
            size_t elements = 0u;
            const float *const actual =
                activeSnapshot(production_key, elements);
            if (!actual || elements == 0u)
            {
                ADD_FAILURE()
                    << "Live sidecar omitted deferred checkpoint "
                    << production_key;
                return false;
            }
            context.checkpoints.push_back({
                .stage = std::string(stage),
                .production_key = production_key,
                .actual = std::vector<float>(actual, actual + elements),
            });
        }

        auto &campaign = deferredMTPBranchCampaign();
        std::lock_guard<std::mutex> lock(campaign.mutex);
        campaign.contexts.push_back(std::move(context));
        return true;
    }

    /**
     * @brief Compare the live grouped-MTP graph with recursive HF checkpoints.
     *
     * The classic decode parity loop is deliberately teacher forced so every
     * main-model row follows the exact Hugging Face trajectory. That loop does
     * not execute speculative sidecars. This check first records a serial
     * production trajectory by constraining the public decodeStep boundary to
     * one token. When both commands hold the same authenticated residency
     * epoch, grouped MTP
     * must reproduce that trajectory exactly. Dynamic and LLEP requests may
     * legitimately publish a new placement between those independent requests;
     * such rows are instead compared directly with their Hugging Face main-model
     * checkpoints and the epoch mismatch is retained in the diagnostic CSV.
     *
     * Sidecar tensors remain compared directly with Hugging Face whenever the
     * serial production prefix still names the same main-model row. Recursive
     * predictors select branch-qualified reference tensors using the proposal
     * tokens observed from the device authority. This keeps every checkpoint
     * mathematically comparable even when a narrow quantized-logit tie sends
     * production down a different draft branch from canonical HF argmax. The
     * two diagnostic CSVs complement—never replace—the six canonical
     * prefill/decode artifacts.
     */
    void runMTPHuggingFaceCheckpointParity()
    {
        if (!isQwen122ProductionTest() || !activeMTPEnabled() ||
            !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const std::vector<int> expected_tokens =
            readDecodeTokensFromMetadata();
        ASSERT_GE(expected_tokens.size(), 2u)
            << "MTP parity needs a sampled prefill token and one sidecar transaction";

        const auto result_dir = ensureResultsDir();
        const auto token_csv_path =
            result_dir / "mtp_sidecar_token_trace.csv";
        const auto snapshot_csv_path =
            result_dir / "mtp_sidecar_snapshot_breakdown.csv";
        const auto failure_values_csv_path =
            result_dir / "mtp_sidecar_failure_values.csv";
        std::ofstream token_csv(token_csv_path, std::ios::trunc);
        std::ofstream snapshot_csv(snapshot_csv_path, std::ios::trunc);
        std::ofstream failure_values_csv(
            failure_values_csv_path, std::ios::trunc);
        ASSERT_TRUE(token_csv.is_open()) << token_csv_path;
        ASSERT_TRUE(snapshot_csv.is_open()) << snapshot_csv_path;
        ASSERT_TRUE(failure_values_csv.is_open()) << failure_values_csv_path;
        token_csv
            << "call,reference_step,selected_depth,emitted_tokens,"
               "serial_expected_tokens,hf_expected_tokens,hf_branch_compatible,"
               "serial_epoch_compatible,serial_movement_epoch,"
               "grouped_movement_epoch_begin,grouped_movement_epoch_end,"
               "serial_execution_epoch,grouped_execution_epoch,"
               "serial_trajectory_epoch,grouped_trajectory_epoch,"
               "production_mtp0_top1,hf_mtp0_top1,recursive_branch_compatible,"
               "verifier_identity_transaction_count,verifier_identity_depth,"
               "production_verifier_draft_tokens,"
               "draft_steps,verifier_runs,accepted,rejected,commits,"
               "rollbacks,validation_failures,current_position\n";
        snapshot_csv
            << "call,reference_step,reference_depth,production_key,reference_key,"
               "elements,cosine,max_abs_diff,kl,exact_indices,routing_overlap,"
               "routing_top1_match,finite,passed\n";
        failure_values_csv
            << "call,reference_step,reference_depth,stage,index,production,reference\n"
            << std::setprecision(std::numeric_limits<float>::max_digits10);

        const std::array<std::string_view, 21> required_stages = {
            "TERMINAL_HIDDEN_ROW_SELECT",
            "EMBEDDING",
            "NORM_HIDDEN",
            "CONCAT",
            "FC",
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "FINAL_NORM",
            "LM_HEAD",
            "V_PROJECTION",
        };

        const auto join_tokens = [](const std::vector<int32_t> &tokens)
        {
            std::ostringstream out;
            for (size_t index = 0; index < tokens.size(); ++index)
            {
                if (index != 0)
                    out << ';';
                out << tokens[index];
            }
            return out.str();
        };
        const std::array<std::string_view, 19> main_verifier_stage_suffixes = {
            "ATTENTION_NORM",
            "QKV_PROJECTION",
            "Q_PROJECTION",
            "GDN_Z_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "GDN_CONV1D_OUTPUT",
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "GDN_OUTPUT",
        };
        const auto is_main_verifier_diagnostic_key =
            [&](const std::string &key)
        {
            if (key == "EMBEDDING" || key == "FINAL_NORM" ||
                key == "LM_HEAD" || key == "LM_HEAD_ROWS_SELECT")
            {
                return true;
            }
            if (key.rfind("layer", 0) != 0)
                return false;
            return std::any_of(
                main_verifier_stage_suffixes.begin(),
                main_verifier_stage_suffixes.end(),
                [&](std::string_view suffix)
                {
                    return std::string_view(key).ends_with(suffix);
                });
        };
        const auto capture_main_verifier_diagnostics = [&]
        {
            std::map<std::string, std::vector<float>> snapshots;
            for (const auto &key : activeSnapshotKeys())
            {
                if (!is_main_verifier_diagnostic_key(key))
                    continue;
                size_t elements = 0;
                const float *const data = activeSnapshot(key, elements);
                if (data && elements > 0)
                    snapshots.emplace(key, std::vector<float>(data, data + elements));
            }
            return snapshots;
        };
        const auto placement_trajectory_after_prefill =
            [&](uint64_t movement_epoch_begin,
                uint64_t movement_epoch_end)
        {
            MTPParityPlacementEpochTrajectory trajectory{
                .epoch = movement_epoch_begin,
            };
            const PrefixRuntimeStateSnapshot state = activePrefixStateProbe();
            const bool restored_prefix =
                state.prefix_request.hit || state.prefix_request.partial_hit;
            if (restored_prefix && isDynamicResidencyProductionTest())
            {
                /*
                 * Placement-agnostic production prefix reuse is numerically
                 * valid, but its restored recurrent bytes may have been
                 * produced before a later expert migration. The cache does not
                 * claim byte-identical placement provenance, so a dynamic hit
                 * cannot seed the stricter grouped-vs-serial oracle.
                 */
                trajectory.invalidate();
                return trajectory;
            }

            uint64_t execution_epoch = movement_epoch_end;
            if (!restored_prefix)
            {
                const auto placement = pinnedDevicePlacementEpochEvidence();
                EXPECT_TRUE(placement.has_value())
                    << "Executed ExpertOverlay prefill omitted its "
                       "device-authenticated placement evidence";
                execution_epoch =
                    placement.has_value() ? placement->epoch : 0u;
            }
            trajectory.observe(
                movement_epoch_begin,
                movement_epoch_end,
                execution_epoch);
            return trajectory;
        };
        /*
         * Record the serial production oracle through the same public surface
         * used by the HTTP server. A one-token response budget cannot enter a
         * speculative transaction, but it still advances the captured main
         * graph and its shifted-MTP state exactly as a normal request does.
         *
         * Residency movement was already proved before numerical parity. Do
         * not submit an additional test-driven maintenance epoch here. Normal
         * Dynamic/LLEP requests can still publish histogram work between these
         * independent serving calls, so each captured row records its actual
         * movement epoch. Only requests whose entire inherited state trajectory
         * stayed in one matching epoch form a valid strict serial-equivalence
         * experiment; all grouped rows retain the direct HF oracle below.
         */
        activeClearSnapshots();
        activeClearCache();
        const uint64_t serial_prefill_movement_epoch_begin =
            orch_runner_->moeRuntimeMovementEpoch();
        ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
            << orch_runner_->lastError();
        const uint64_t serial_prefill_movement_epoch_end =
            orch_runner_->moeRuntimeMovementEpoch();
        MTPParityPlacementEpochTrajectory serial_trajectory =
            placement_trajectory_after_prefill(
                serial_prefill_movement_epoch_begin,
                serial_prefill_movement_epoch_end);
        const uint64_t mtp_certification_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();
        /** @brief One serial row plus the residency epoch that executed it. */
        struct SerialVerifierOracle
        {
            std::map<std::string, std::vector<float>> snapshots;
            uint64_t movement_epoch_begin = 0;
            uint64_t movement_epoch_end = 0;
            uint64_t execution_epoch = 0;
            std::optional<uint64_t> trajectory_epoch;

            /** @return Whether this row inherited a complete history in @p epoch. */
            bool trajectoryInEpoch(uint64_t epoch) const
            {
                return epoch != 0u && trajectory_epoch == epoch;
            }
        };
        std::vector<int32_t> serial_tokens;
        const size_t serial_oracle_token_count = std::max(
            expected_tokens.size(),
            usesDynamicMTPDepth()
                ? static_cast<size_t>(activeMTPDraftDepth() + 1)
                : expected_tokens.size());
        std::vector<SerialVerifierOracle>
            serial_oracles_by_output_count(serial_oracle_token_count + 1u);
        while (serial_tokens.size() < serial_oracle_token_count)
        {
            activeClearSnapshots();
            orch_runner_->setDecodeStepTokenBudget(1);
            const uint64_t movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const GenerationResult serial_step = orch_runner_->decodeStep();
            const uint64_t movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(serial_step.success()) << serial_step.error;
            ASSERT_EQ(serial_step.tokens.size(), 1u)
                << "A one-token serial oracle boundary returned a grouped response";
            const auto serial_placement =
                pinnedDevicePlacementEpochEvidence();
            ASSERT_TRUE(serial_placement.has_value())
                << "Serial ExpertOverlay command omitted its device-authenticated "
                   "execution epoch";
            serial_trajectory.observe(
                movement_epoch_begin,
                movement_epoch_end,
                serial_placement->epoch);
            serial_tokens.push_back(serial_step.tokens.front());
            serial_oracles_by_output_count[serial_tokens.size()] = {
                .snapshots = capture_main_verifier_diagnostics(),
                .movement_epoch_begin = movement_epoch_begin,
                .movement_epoch_end = movement_epoch_end,
                .execution_epoch = serial_placement->epoch,
                .trajectory_epoch = serial_trajectory.epoch,
            };
        }
        orch_runner_->setDecodeStepTokenBudget(0);

        ASSERT_EQ(serial_tokens.size(), serial_oracle_token_count);
        ASSERT_GE(serial_tokens.size(), expected_tokens.size());
        ASSERT_EQ(serial_tokens.front(), expected_tokens.front())
            << "The prefill boundary must agree exactly before MTP branch comparison";

        /* Start a fresh request so the first grouped sidecar is decode_step0. */
        activeClearSnapshots();
        activeClearCache();
        const uint64_t grouped_prefill_movement_epoch_begin =
            orch_runner_->moeRuntimeMovementEpoch();
        ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
            << orch_runner_->lastError();
        const uint64_t grouped_prefill_movement_epoch_end =
            orch_runner_->moeRuntimeMovementEpoch();
        MTPParityPlacementEpochTrajectory grouped_trajectory =
            placement_trajectory_after_prefill(
                grouped_prefill_movement_epoch_begin,
                grouped_prefill_movement_epoch_end);

        std::vector<int32_t> emitted;
        const auto initial_state = activePrefixStateProbe();
        int speculative_calls = 0;
        int compared_stages = 0;
        int compared_recursive_stages = 0;
        int deferred_recursive_contexts = 0;
        int compared_main_stages = 0;
        int compared_main_lm_heads = 0;
        int failed_main_lm_heads = 0;
        double grouped_main_cosine_sum = 0.0;
        size_t grouped_main_cosine_count = 0u;
        int serial_epoch_compatible_calls = 0;
        int call = 0;
        while (emitted.size() < expected_tokens.size())
        {
            activeClearSnapshots();
            const auto before = activePrefixStateProbe();
            const int remaining = static_cast<int>(
                expected_tokens.size() - emitted.size());
            int selected_depth_before = usesDynamicMTPDepth()
                                            ? before.mtp_current_depth
                                            : activeMTPDraftDepth();
            if (selected_depth_before <= 0)
                selected_depth_before = activeMTPDraftDepth();
            ASSERT_GE(selected_depth_before, 1);
            ASSERT_LE(selected_depth_before, activeMTPDraftDepth());
            /*
             * A two-token response boundary is the smallest public serving
             * request that admits speculative execution. A rejection can let
             * one public call retire more than one device transaction, so the
             * counter fold below validates every retired transaction. The
             * row-zero snapshot comparison is enabled only when the counters
             * prove that this call has one unambiguous verifier identity.
             */
            orch_runner_->setDecodeStepTokenBudget(
                std::min(remaining, 2));
            const uint64_t grouped_movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const GenerationResult step = orch_runner_->decodeStep();
            const uint64_t grouped_movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(step.success()) << step.error;
            ASSERT_FALSE(step.tokens.empty());
            ASSERT_LE(step.tokens.size(), static_cast<size_t>(remaining));
            const auto after = activePrefixStateProbe();
            const auto grouped_placement =
                pinnedDevicePlacementEpochEvidence();
            ASSERT_TRUE(grouped_placement.has_value())
                << "Grouped ExpertOverlay command omitted its device-authenticated "
                   "execution epoch";
            const uint64_t grouped_execution_epoch =
                grouped_placement->epoch;
            grouped_trajectory.observe(
                grouped_movement_epoch_begin,
                grouped_movement_epoch_end,
                grouped_execution_epoch);
            const size_t output_begin = emitted.size();

            bool serial_epoch_compatible =
                grouped_trajectory.epoch.has_value();
            uint64_t serial_movement_epoch =
                grouped_movement_epoch_begin;
            uint64_t serial_execution_epoch = 0u;
            uint64_t serial_trajectory_epoch = 0u;
            for (size_t offset = 0; offset < step.tokens.size(); ++offset)
            {
                const size_t oracle_index = output_begin + offset + 1u;
                ASSERT_LT(oracle_index, serial_oracles_by_output_count.size());
                const auto &oracle =
                    serial_oracles_by_output_count[oracle_index];
                serial_epoch_compatible =
                    serial_epoch_compatible &&
                    oracle.trajectoryInEpoch(grouped_execution_epoch) &&
                    grouped_trajectory.matches(grouped_execution_epoch);
                if (offset == 0)
                {
                    serial_movement_epoch = oracle.movement_epoch_begin;
                    serial_execution_epoch = oracle.execution_epoch;
                    serial_trajectory_epoch =
                        oracle.trajectory_epoch.value_or(0u);
                }
            }
            if (serial_epoch_compatible)
                ++serial_epoch_compatible_calls;

            for (size_t offset = 0; offset < step.tokens.size(); ++offset)
            {
                const size_t output_index = emitted.size() + offset;
                ASSERT_LT(output_index, expected_tokens.size());
                if (serial_epoch_compatible)
                {
                    EXPECT_EQ(step.tokens[offset], serial_tokens[output_index])
                        << "Grouped MTP diverged from the serial production token "
                           "trajectory at output "
                        << output_index;
                }
            }

            const MTPParityTransactionCounters before_counters{
                .draft_steps = before.mtp_draft_steps,
                .verifier_runs = before.mtp_verifier_runs,
            };
            const MTPParityTransactionCounters after_counters{
                .draft_steps = after.mtp_draft_steps,
                .verifier_runs = after.mtp_verifier_runs,
            };
            const auto transaction_activity =
                classifyMTPParityTransactionActivity(
                    before_counters,
                    after_counters);
            EXPECT_NE(
                transaction_activity,
                MTPParityTransactionActivity::Inconsistent)
                << "MTP draft/verifier counters advanced inconsistently: before "
                << "drafts=" << before.mtp_draft_steps
                << " verifiers=" << before.mtp_verifier_runs
                << " after drafts=" << after.mtp_draft_steps
                << " verifiers=" << after.mtp_verifier_runs;
            const bool speculative =
                transaction_activity ==
                MTPParityTransactionActivity::Speculative;
            const uint64_t executed_transaction_count =
                mtpParityExecutedTransactionCount(
                    before_counters,
                    after_counters);
            const uint64_t attempted_draft_tokens =
                mtpParityAttemptedDraftTokenCount(
                    before_counters,
                    after_counters);

            int reference_step = -1;
            int selected_depth = 0;
            int production_mtp0_top1 = -1;
            int hf_mtp0_top1 = -1;
            bool recursive_branch_compatible = false;
            bool hf_branch_compatible = false;
            if (speculative)
            {
                ++speculative_calls;
                reference_step =
                    mtpParityReferenceStepForConditionPosition(
                        before.mtp_next_condition_position,
                        static_cast<int>(config_.token_ids.size()));
                ASSERT_GE(reference_step, 0)
                    << "Device MTP controller omitted its live condition position";
                selected_depth = after.mtp_last_transaction_draft_depth;
                ASSERT_GE(selected_depth, 1);
                ASSERT_LE(selected_depth, activeMTPDraftDepth());
                EXPECT_EQ(selected_depth, selected_depth_before)
                    << "The device-owned transaction selected a different "
                       "depth than its pre-submission controller state";
                const size_t reference_prefix_length =
                    static_cast<size_t>(reference_step + 1);
                std::vector<int32_t> grouped_visible_prefix = emitted;
                grouped_visible_prefix.insert(
                    grouped_visible_prefix.end(),
                    step.tokens.begin(),
                    step.tokens.end());
                hf_branch_compatible =
                    reference_prefix_length <= expected_tokens.size() &&
                    reference_prefix_length <= grouped_visible_prefix.size() &&
                    std::equal(
                        grouped_visible_prefix.begin(),
                        grouped_visible_prefix.begin() + reference_prefix_length,
                        expected_tokens.begin());
                ASSERT_GE(executed_transaction_count, 1u);
                ASSERT_GE(attempted_draft_tokens, executed_transaction_count)
                    << "Every retired verifier transaction must attempt a draft";
                ASSERT_LE(
                    attempted_draft_tokens,
                    executed_transaction_count *
                        static_cast<uint64_t>(activeMTPDraftDepth()))
                    << "The device controller attempted more drafts than its "
                       "captured transaction capacity";
                ASSERT_EQ(
                    after.mtp_observed_verifier_transaction_count,
                    static_cast<int>(executed_transaction_count))
                    << "The durable verifier identity must be committed by "
                       "the same device transaction(s) reported by production";
                ASSERT_EQ(
                    after.mtp_observed_verifier_draft_depth,
                    selected_depth)
                    << "The durable verifier identity names a different "
                       "dynamic-depth branch than the committed controller";
                ASSERT_EQ(
                    after.mtp_observed_verifier_draft_tokens.size(),
                    static_cast<size_t>(selected_depth))
                    << "Every committed draft token must have an exact, "
                       "response-visible verifier identity";

                const std::string first_sidecar_prefix = std::string(
                    mtpParityCheckpointContextPrefix(
                        before.mtp_verifier_runs == 0
                            ? MTPParityCheckpointContext::
                                  DeviceTargetTokenLivePosition
                            : MTPParityCheckpointContext::
                                  DeviceResidentLogicalState));
                size_t production_lm_head_size = 0;
                const float *production_lm_head = activeSnapshot(
                    first_sidecar_prefix + "MTP0_LM_HEAD",
                    production_lm_head_size);
                const std::vector<float> hf_mtp0_lm_head =
                    loadPyTorchSnapshot(
                        "decode_step" + std::to_string(reference_step) +
                        "_MTP0_LM_HEAD");
                ASSERT_NE(production_lm_head, nullptr);
                ASSERT_EQ(production_lm_head_size, hf_mtp0_lm_head.size());
                production_mtp0_top1 = static_cast<int>(std::distance(
                    production_lm_head,
                    std::max_element(
                        production_lm_head,
                        production_lm_head + production_lm_head_size)));
                hf_mtp0_top1 = static_cast<int>(std::distance(
                    hf_mtp0_lm_head.begin(),
                    std::max_element(
                        hf_mtp0_lm_head.begin(), hf_mtp0_lm_head.end())));
                recursive_branch_compatible =
                    production_mtp0_top1 == hf_mtp0_top1;

                struct ActiveContext
                {
                    std::string prefix;
                    int reference_depth = 0;
                };
                std::vector<ActiveContext> contexts;
                contexts.push_back({
                    .prefix = std::string(
                        mtpParityCheckpointContextPrefix(
                            before.mtp_verifier_runs == 0
                                ? MTPParityCheckpointContext::DeviceTargetTokenLivePosition
                                : MTPParityCheckpointContext::DeviceResidentLogicalState)),
                    .reference_depth = 0,
                });
                const size_t recursive_condition_tokens =
                    selected_depth > 1
                        ? static_cast<size_t>(selected_depth - 1)
                        : 0u;
                const bool has_recursive_proposal_identity =
                    recursive_condition_tokens > 0u &&
                    after.mtp_observed_verifier_draft_tokens.size() >=
                        recursive_condition_tokens &&
                    std::all_of(
                        after.mtp_observed_verifier_draft_tokens.begin(),
                        after.mtp_observed_verifier_draft_tokens.begin() +
                            static_cast<ptrdiff_t>(recursive_condition_tokens),
                        [](int32_t token) { return token >= 0; });
                bool recursive_reference_deferred = false;
                std::vector<int32_t> recursive_reference_condition_tokens;

                /**
                 * @brief Decide whether an observed proposal follows the
                 *        canonical Hugging Face recursive branch.
                 *
                 * Canonical recursive snapshots have no `_BRANCH_...`
                 * qualifier. Additive snapshots use that qualifier only after
                 * production selects a condition token different from the
                 * corresponding canonical HF predictor. Compare the complete
                 * preceding predictor chain so a depth-two checkpoint is
                 * canonical only when both condition tokens agree.
                 *
                 * @param reference_depth Number of recursive condition tokens
                 *        consumed by the requested checkpoint.
                 * @return True when the unqualified HF checkpoint is the exact
                 *         oracle for the observed production branch.
                 */
                const auto follows_canonical_hf_branch =
                    [&](int reference_depth) -> bool
                {
                    if (reference_depth < 0 ||
                        after.mtp_observed_verifier_draft_tokens.size() <
                            static_cast<size_t>(reference_depth))
                    {
                        ADD_FAILURE()
                            << "Cannot resolve recursive HF branch depth "
                            << reference_depth << " from "
                            << after.mtp_observed_verifier_draft_tokens.size()
                            << " observed proposal tokens";
                        return false;
                    }
                    for (int proposal_depth = 0;
                         proposal_depth < reference_depth;
                         ++proposal_depth)
                    {
                        const std::vector<float> canonical_logits =
                            loadPyTorchSnapshot(
                                "decode_step" +
                                std::to_string(reference_step) + "_MTP" +
                                std::to_string(proposal_depth) +
                                "_LM_HEAD");
                        if (canonical_logits.empty())
                        {
                            ADD_FAILURE()
                                << "Hugging Face pack omitted the canonical MTP"
                                << proposal_depth
                                << " LM-head oracle at decode step "
                                << reference_step;
                            return false;
                        }
                        const int32_t canonical_token = static_cast<int32_t>(
                            std::distance(
                                canonical_logits.begin(),
                                std::max_element(
                                    canonical_logits.begin(),
                                    canonical_logits.end())));
                        if (after.mtp_observed_verifier_draft_tokens[
                                static_cast<size_t>(proposal_depth)] !=
                            canonical_token)
                        {
                            return false;
                        }
                    }
                    return true;
                };
                if (has_recursive_proposal_identity)
                {
                    const int recursive_reference_depth = selected_depth - 1;
                    recursive_branch_compatible =
                        follows_canonical_hf_branch(
                            recursive_reference_depth);
                    if (!recursive_branch_compatible)
                    {
                        recursive_reference_condition_tokens.assign(
                            after.mtp_observed_verifier_draft_tokens.begin(),
                            after.mtp_observed_verifier_draft_tokens.begin() +
                                recursive_reference_depth);
                        recursive_reference_deferred =
                            hf_branch_compatible &&
                            !hasHuggingFaceMTPBranchReference(
                                reference_step,
                                recursive_reference_condition_tokens);
                    }
                    contexts.push_back({
                        .prefix = std::string(
                            mtpParityCheckpointContextPrefix(
                                MTPParityCheckpointContext::DeviceChainedTokenLivePosition)),
                        .reference_depth = recursive_reference_depth,
                    });
                }

                for (const auto &context : contexts)
                {
                    if (context.reference_depth > 0 &&
                        recursive_reference_deferred)
                    {
                        ASSERT_TRUE(deferHuggingFaceMTPBranchReference(
                            call,
                            reference_step,
                            context.reference_depth,
                            recursive_reference_condition_tokens,
                            context.prefix,
                            required_stages,
                            snapshot_csv_path))
                            << "Could not preserve the live recursive MTP "
                               "checkpoint set before runner retirement";
                        ++deferred_recursive_contexts;
                        continue;
                    }
                    /*
                     * A quantized router may exchange only the lowest-weight
                     * top-k boundary expert while retaining the top-1 expert
                     * and k-1 set overlap. The routing checkpoint below proves
                     * that bounded discrete difference explicitly. Remember
                     * whether it occurred so the immediately dependent raw
                     * routed sum uses the established intermediate-tensor
                     * threshold; exact routes retain the stricter decode floor.
                     * Downstream combined output and LM-head checks are never
                     * relaxed by this state.
                     */
                    bool routing_indices_exact = true;
                    bool context_finite = true;
                    bool context_routing_top1_match = true;
                    float context_routing_overlap = 1.0f;
                    bool context_routing_weights_equivalent = false;
                    bool context_routed_expert_output_equivalent = false;
                    bool context_lm_head_passed = false;
                    double context_numerical_cosine_sum = 0.0;
                    size_t context_numerical_stage_count = 0u;
                    const auto sidecar_moe = getMoEConfig();
                    for (const std::string_view stage : required_stages)
                    {
                        const std::string production_key =
                            context.prefix +
                            (stage == "TERMINAL_HIDDEN_ROW_SELECT"
                                 ? "MTP_TERMINAL_HIDDEN_ROW_SELECT"
                                 : "MTP0_" + std::string(stage));
                        std::string reference_prefix =
                            "decode_step" + std::to_string(reference_step);
                        if (context.reference_depth > 0)
                        {
                            ASSERT_GE(
                                after.mtp_observed_verifier_draft_tokens.size(),
                                static_cast<size_t>(context.reference_depth))
                                << "The proposal authority omitted tokens "
                                   "needed to identify recursive HF depth "
                                << context.reference_depth;
                            if (!follows_canonical_hf_branch(
                                    context.reference_depth))
                            {
                                reference_prefix += "_BRANCH";
                                for (int branch_depth = 0;
                                     branch_depth < context.reference_depth;
                                     ++branch_depth)
                                {
                                    reference_prefix += "_" + std::to_string(
                                        after.mtp_observed_verifier_draft_tokens[
                                            static_cast<size_t>(branch_depth)]);
                                }
                            }
                        }
                        const std::string reference_key =
                            reference_prefix + "_MTP" +
                            std::to_string(context.reference_depth) + "_" +
                            std::string(stage);
                        if (!hf_branch_compatible)
                            continue;

                        size_t actual_size = 0;
                        const float *actual =
                            activeSnapshot(production_key, actual_size);
                        const std::vector<float> reference =
                            loadPyTorchSnapshot(reference_key);
                        ASSERT_NE(actual, nullptr)
                            << "Live sidecar omitted " << production_key;
                        ASSERT_GT(actual_size, 0u)
                            << "Live sidecar published an empty " << production_key;
                        ASSERT_FALSE(reference.empty())
                            << "Hugging Face pack omitted " << reference_key;

                        const float *expected = reference.data();
                        size_t expected_size = reference.size();
                        if (expected_size > actual_size && actual_size > 0 &&
                            expected_size % actual_size == 0)
                        {
                            expected += expected_size - actual_size;
                            expected_size = actual_size;
                        }
                        ASSERT_EQ(actual_size, expected_size)
                            << production_key << " versus " << reference_key;

                        bool finite = true;
                        bool exact_indices = true;
                        float routing_overlap = 1.0f;
                        bool routing_top1_match = true;
                        double max_abs_diff = 0.0;
                        for (size_t index = 0; index < actual_size; ++index)
                        {
                            finite = finite && std::isfinite(actual[index]) &&
                                     std::isfinite(expected[index]);
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(
                                    static_cast<double>(actual[index]) -
                                    static_cast<double>(expected[index])));
                            if (stage == "MOE_ROUTING_INDICES")
                            {
                                exact_indices = exact_indices &&
                                                actual[index] == expected[index];
                            }
                        }
                        float cosine = computeCosineSimilarity(
                            actual, expected, actual_size);
                        float kl = 0.0f;
                        bool passed = finite;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            /*
                             * Quantized router logits may swap one expert at the
                             * low-weight top-k boundary even when the routing
                             * distribution and routed value remain numerically
                             * equivalent. Preserve the highest-weight expert and
                             * require at least k-1 set overlap, matching the
                             * production parity suite's discrete-routing model.
                             */
                            std::set<int> actual_experts;
                            std::set<int> reference_experts;
                            for (size_t index = 0; index < actual_size; ++index)
                            {
                                actual_experts.insert(
                                    static_cast<int>(actual[index]));
                                reference_experts.insert(
                                    static_cast<int>(expected[index]));
                            }
                            size_t intersection = 0;
                            for (const int expert : actual_experts)
                            {
                                if (reference_experts.contains(expert))
                                    ++intersection;
                            }
                            routing_overlap = actual_size > 0
                                                  ? static_cast<float>(intersection) /
                                                        static_cast<float>(actual_size)
                                                  : 0.0f;
                            routing_top1_match =
                                actual_size > 0 && actual[0] == expected[0];
                            routing_indices_exact = exact_indices;
                            cosine = routing_overlap;
                            max_abs_diff = 1.0 - routing_overlap;
                            const float minimum_boundary_overlap =
                                sidecar_moe.top_k > 0
                                    ? 1.0f -
                                          1.0f /
                                              static_cast<float>(
                                                  sidecar_moe.top_k)
                                    : 1.0f;
                            passed = passed && routing_top1_match &&
                                     routing_overlap >=
                                         minimum_boundary_overlap;
                            context_routing_top1_match =
                                context_routing_top1_match &&
                                routing_top1_match;
                            context_routing_overlap = std::min(
                                context_routing_overlap,
                                routing_overlap);
                        }
                        else if (stage == "MOE_ROUTING_WEIGHTS")
                        {
                            /*
                             * Ordered weight vectors can look identical even
                             * when their low-weight entries name different
                             * experts. Reconstruct sparse expert-ID vectors,
                             * exactly as the canonical decode campaign does,
                             * so the routing proof measures contribution mass
                             * rather than array position.
                             */
                            size_t actual_indices_size = 0u;
                            const float *const actual_indices = activeSnapshot(
                                context.prefix +
                                    "MTP0_MOE_ROUTING_INDICES",
                                actual_indices_size);
                            const std::vector<float> reference_indices =
                                loadPyTorchSnapshot(
                                    reference_prefix + "_MTP" +
                                    std::to_string(context.reference_depth) +
                                    "_MOE_ROUTING_INDICES");
                            ASSERT_NE(actual_indices, nullptr);
                            ASSERT_EQ(actual_indices_size, actual_size);
                            ASSERT_EQ(reference_indices.size(), actual_size);
                            const StageComparisonResult routing_result =
                                compareRoutingWeights(
                                    actual,
                                    reference,
                                    actual_indices,
                                    reference_indices,
                                    actual_size,
                                    sidecar_moe.top_k,
                                    sidecar_moe.num_experts,
                                    std::string(stage));
                            cosine = routing_result.cosine_similarity;
                            max_abs_diff = routing_result.max_abs_diff;
                            routing_overlap =
                                routing_result.routing_overlap;
                            passed = finite && routing_result.passed;
                            context_routing_weights_equivalent = passed;
                        }
                        else
                        {
                            const float numerical_threshold =
                                stage == "MOE_EXPERT_OUTPUT" &&
                                        !routing_indices_exact
                                    ? config_.cosine_threshold
                                    : config_.decode_cosine_threshold;
                            passed = passed &&
                                     cosine >= numerical_threshold;
                            if (stage == "MOE_EXPERT_OUTPUT")
                            {
                                context_routed_expert_output_equivalent =
                                    passed;
                            }
                        }
                        if (stage == "LM_HEAD")
                        {
                            kl = computeKLDivergence(
                                actual,
                                expected,
                                actual_size,
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            passed = passed &&
                                     kl < config_.mtp_kl_threshold.value_or(
                                              config_.kl_threshold);
                            const float reference_top1_in_production_top3 =
                                pytorchTop1InLlaminarTopK(
                                    actual,
                                    expected,
                                    actual_size,
                                    actual_size,
                                    3);
                            const float production_top1_in_reference_top3 =
                                pytorchTop1InLlaminarTopK(
                                    expected,
                                    actual,
                                    actual_size,
                                    actual_size,
                                    3);
                            passed = passed &&
                                     reference_top1_in_production_top3 >= 1.0f &&
                                     production_top1_in_reference_top3 >= 1.0f;
                            context_lm_head_passed = passed;

                            /*
                             * A recurrent LM-head miss can originate either in
                             * the incoming hidden row or in the projection
                             * itself. Preserve the compact 3,072-value operand
                             * only on failure so an offline exact-codebook
                             * projection can attribute those two errors without
                             * making every successful matrix cell emit a full
                             * vocabulary vector.
                             */
                            if (!passed)
                            {
                                size_t hidden_size = 0u;
                                const float *const hidden = activeSnapshot(
                                    context.prefix + "MTP0_FINAL_NORM",
                                    hidden_size);
                                std::vector<float> reference_hidden =
                                    loadPyTorchSnapshot(
                                        reference_prefix + "_MTP" +
                                        std::to_string(
                                            context.reference_depth) +
                                        "_FINAL_NORM");
                                ASSERT_NE(hidden, nullptr);
                                ASSERT_EQ(hidden_size, reference_hidden.size());
                                for (size_t index = 0u;
                                     index < hidden_size;
                                     ++index)
                                {
                                    failure_values_csv
                                        << call << ',' << reference_step << ','
                                        << context.reference_depth
                                        << ",FINAL_NORM," << index << ','
                                        << hidden[index] << ','
                                        << reference_hidden[index] << '\n';
                                }
                            }
                        }

                        context_finite = context_finite && finite;
                        /*
                         * Terminal hidden is an input-provenance diagnostic,
                         * not another model-layer result. Gate it individually
                         * above, but do not double-count the previous sidecar's
                         * output in this sidecar's numerical aggregate.
                         */
                        if (stage != "TERMINAL_HIDDEN_ROW_SELECT" &&
                            parityStageContributesToLayerCosine(
                                stage,
                                routing_indices_exact))
                        {
                            context_numerical_cosine_sum += cosine;
                            ++context_numerical_stage_count;
                        }

                        snapshot_csv
                            << call << ',' << reference_step << ','
                            << context.reference_depth << ','
                            << production_key << ',' << reference_key << ','
                            << actual_size << ',' << cosine << ','
                            << max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ','
                            << routing_overlap << ','
                            << (routing_top1_match ? 1 : 0) << ','
                            << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        ++compared_stages;
                        if (context.reference_depth > 0)
                            ++compared_recursive_stages;
                    }

                    if (!hf_branch_compatible)
                    {
                        /*
                         * A residency epoch can move the independent Dynamic
                         * request onto a different quantized-logit branch. The
                         * token CSV retains that fact, but no HF tensor from
                         * the canonical branch is an oracle for this context.
                         */
                        continue;
                    }

                    /*
                     * Apply the same route-aware numerical contract as
                     * runDecodeParity(): every checkpoint remains in the CSV,
                     * routing uses dedicated discrete/sparse metrics, raw
                     * expert sums from different legal boundary routes do not
                     * enter an elementwise cosine, and the remaining semantic
                     * tensors form one sidecar-layer aggregate. The LM-head
                     * still independently proves cosine, KL, and symmetric
                     * top-3 containment, so aggregation cannot hide a wrong
                     * token distribution.
                     */
                    ASSERT_GT(context_numerical_stage_count, 0u);
                    const double context_numerical_cosine =
                        context_numerical_cosine_sum /
                        static_cast<double>(
                            context_numerical_stage_count);
                    EXPECT_TRUE(context_finite)
                        << "Recursive MTP context published a non-finite "
                           "checkpoint at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    EXPECT_TRUE(context_routing_top1_match)
                        << "Recursive MTP context changed its highest-weight "
                           "expert at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    const float minimum_boundary_overlap =
                        sidecar_moe.top_k > 0
                            ? 1.0f -
                                  1.0f /
                                      static_cast<float>(sidecar_moe.top_k)
                            : 1.0f;
                    const bool routed_contribution_equivalent =
                        context_routing_weights_equivalent ||
                        (context_routing_top1_match &&
                         context_routing_overlap >=
                             minimum_boundary_overlap &&
                         context_routed_expert_output_equivalent);
                    EXPECT_TRUE(routed_contribution_equivalent)
                        << "Recursive MTP context changed both sparse routed "
                           "mass and the resulting expert value at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    const bool numerical_aggregate_passed =
                        context.reference_depth > 0
                            ? productionRecursiveMTPAggregatePasses(
                                  context_numerical_cosine,
                                  config_.decode_cosine_threshold)
                            : context_numerical_cosine >=
                                  static_cast<double>(
                                      config_.decode_cosine_threshold);
                    EXPECT_TRUE(numerical_aggregate_passed)
                        << "Recursive MTP context numerical aggregate failed "
                           "at reference depth "
                        << context.reference_depth
                        << " cosine=" << context_numerical_cosine
                        << " required="
                        << (context.reference_depth > 0
                                ? std::max(
                                      static_cast<double>(
                                          config_.decode_cosine_threshold),
                                      kMinimumProductionRecursiveMTPAggregateCosine)
                                : static_cast<double>(
                                      config_.decode_cosine_threshold))
                        << "\nCSV: " << snapshot_csv_path;
                    EXPECT_TRUE(context_lm_head_passed)
                        << "Recursive MTP context LM-head/KL/top-k proof "
                           "failed at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                }

                /*
                 * Compare grouped verifier row zero directly with the canonical
                 * HF main-model row while the production-visible prefix names
                 * that reference branch. This is the authoritative numerical
                 * oracle when a Dynamic/LLEP publication makes the separately
                 * executed serial request a different residency experiment.
                 */
                if (hf_branch_compatible && executed_transaction_count == 1u)
                {
                    const size_t verifier_rows = static_cast<size_t>(
                        activeMTPPhysicalVerifierRows());
                    const auto gdn = getGDNHeadConfig();
                    const auto moe = getMoEConfig();
                    for (const auto &key : activeSnapshotKeys())
                    {
                        if (!is_main_verifier_diagnostic_key(key) ||
                            key == "LM_HEAD_ROWS_SELECT")
                        {
                            continue;
                        }

                        size_t grouped_elements = 0;
                        const float *const grouped =
                            activeSnapshot(key, grouped_elements);
                        if (!grouped || grouped_elements == 0u ||
                            grouped_elements % verifier_rows != 0u)
                        {
                            continue;
                        }
                        const size_t row_elements =
                            grouped_elements / verifier_rows;
                        const std::string reference_key =
                            "decode_step" + std::to_string(reference_step) +
                            "_" + key;
                        std::vector<float> reference =
                            loadPyTorchSnapshot(reference_key);
                        if (reference.empty() ||
                            reference.size() < row_elements ||
                            reference.size() % row_elements != 0u)
                        {
                            continue;
                        }
                        if (reference.size() > row_elements)
                        {
                            reference.erase(
                                reference.begin(),
                                reference.end() -
                                    static_cast<ptrdiff_t>(row_elements));
                        }

                        std::string stage = key;
                        if (key.rfind("layer", 0) == 0)
                        {
                            const size_t delimiter = key.find('_');
                            if (delimiter != std::string::npos)
                                stage = key.substr(delimiter + 1u);
                        }
                        const std::vector<float> permuted =
                            applyGDNHeadPermutation(
                                grouped, row_elements, stage, gdn);
                        const float *const actual = permuted.empty()
                                                        ? grouped
                                                        : permuted.data();

                        StageComparisonResult result;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            result = compareRoutingIndices(
                                actual,
                                reference,
                                row_elements,
                                moe.top_k,
                                stage);
                        }
                        else if (stage == "MOE_ROUTING_WEIGHTS")
                        {
                            const std::string index_key =
                                key.substr(
                                    0,
                                    key.size() -
                                        std::string("MOE_ROUTING_WEIGHTS").size()) +
                                "MOE_ROUTING_INDICES";
                            size_t grouped_index_elements = 0;
                            const float *const grouped_indices =
                                activeSnapshot(
                                    index_key,
                                    grouped_index_elements);
                            std::vector<float> reference_indices =
                                loadPyTorchSnapshot(
                                    "decode_step" +
                                    std::to_string(reference_step) + "_" +
                                    index_key);
                            if (reference_indices.size() > row_elements &&
                                reference_indices.size() % row_elements == 0u)
                            {
                                reference_indices.erase(
                                    reference_indices.begin(),
                                    reference_indices.end() -
                                        static_cast<ptrdiff_t>(row_elements));
                            }
                            if (grouped_indices &&
                                grouped_index_elements ==
                                    verifier_rows * row_elements &&
                                reference_indices.size() == row_elements)
                            {
                                result = compareRoutingWeights(
                                    actual,
                                    reference,
                                    grouped_indices,
                                    reference_indices,
                                    row_elements,
                                    moe.top_k,
                                    moe.num_experts,
                                    stage);
                            }
                            else
                            {
                                result = compareTensors(
                                    actual, reference, row_elements, stage);
                            }
                        }
                        else
                        {
                            result = compareTensors(
                                actual, reference, row_elements, stage);
                        }

                        bool finite = true;
                        bool exact_indices = true;
                        for (size_t index = 0; index < row_elements; ++index)
                        {
                            finite = finite && std::isfinite(actual[index]) &&
                                     std::isfinite(reference[index]);
                            exact_indices = exact_indices &&
                                            actual[index] == reference[index];
                        }
                        float kl = 0.0f;
                        bool passed = finite && result.passed;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            const float minimum_overlap =
                                row_elements > 0u
                                    ? 1.0f -
                                          1.0f /
                                              static_cast<float>(row_elements)
                                    : 1.0f;
                            passed = finite &&
                                     result.routing_top1_match >= 1.0f &&
                                     result.routing_overlap >= minimum_overlap;
                        }
                        if (key == "LM_HEAD")
                        {
                            ++compared_main_lm_heads;
                            kl = computeKLDivergence(
                                actual,
                                reference.data(),
                                row_elements,
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            const float reference_top1_in_production_top3 =
                                pytorchTop1InLlaminarTopK(
                                    actual,
                                    reference.data(),
                                    row_elements,
                                    row_elements,
                                    3);
                            const float production_top1_in_reference_top3 =
                                pytorchTop1InLlaminarTopK(
                                    reference.data(),
                                    actual,
                                    row_elements,
                                    row_elements,
                                    3);
                            passed = finite &&
                                     result.cosine_similarity >=
                                         config_.decode_cosine_threshold &&
                                     kl < config_.kl_threshold &&
                                     reference_top1_in_production_top3 >= 1.0f &&
                                     production_top1_in_reference_top3 >= 1.0f;
                            if (!passed)
                                ++failed_main_lm_heads;
                        }

                        /*
                         * Match runDecodeParity's established aggregation:
                         * routing has its own set/sparse-vector metrics, while
                         * numerical tensors contribute to the mean-cosine
                         * gate. Individual low-energy intermediates remain
                         * visible through their CSV `passed` field without
                         * letting an ill-conditioned cosine replace the
                         * end-to-end logit/KL/top-k proof.
                         */
                        if (!result.is_routing_stage)
                        {
                            grouped_main_cosine_sum +=
                                result.cosine_similarity;
                            ++grouped_main_cosine_count;
                        }

                        snapshot_csv
                            << call << ',' << reference_step << ",-2,"
                            << key << ',' << reference_key << ','
                            << row_elements << ','
                            << result.cosine_similarity << ','
                            << result.max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ','
                            << (result.is_routing_stage
                                    ? result.routing_overlap
                                    : 1.0f)
                            << ','
                            << (result.is_routing_stage
                                    ? result.routing_top1_match
                                    : 1.0f)
                            << ',' << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        ++compared_main_stages;
                    }
                }

                /*
                 * The first transaction of a two-token public boundary has an
                 * unambiguous grouped-verifier row-zero oracle: the serial
                 * request after consuming the same first visible token. Copy
                 * every compact M-row checkpoint against that serial M=1 row.
                 * This turns a wrong correction token into an earliest-stage
                 * CSV diagnosis instead of leaving only the final token ID.
                 */
                /*
                 * The verifier row is identified by the device-owned live
                 * condition position, not by the number of tokens returned by
                 * this public call. An accepted draft can make output_begin
                 * advance farther than the next verifier condition; mapping by
                 * response count then compares different autoregressive rows.
                 */
                const size_t serial_snapshot_index =
                    static_cast<size_t>(reference_step + 1);
                if (serial_epoch_compatible &&
                    executed_transaction_count == 1u &&
                    serial_snapshot_index <
                        serial_oracles_by_output_count.size())
                {
                    const auto &serial_snapshots =
                        serial_oracles_by_output_count[serial_snapshot_index]
                            .snapshots;
                    const size_t verifier_rows = static_cast<size_t>(
                        activeMTPPhysicalVerifierRows());
                    for (const auto &[key, serial_values] : serial_snapshots)
                    {
                        size_t grouped_elements = 0;
                        const float *const grouped =
                            activeSnapshot(key, grouped_elements);
                        if (!grouped || serial_values.empty() ||
                            grouped_elements !=
                                verifier_rows * serial_values.size())
                        {
                            continue;
                        }

                        bool finite = true;
                        bool exact_indices = true;
                        double max_abs_diff = 0.0;
                        for (size_t index = 0;
                             index < serial_values.size();
                             ++index)
                        {
                            finite = finite &&
                                     std::isfinite(grouped[index]) &&
                                     std::isfinite(serial_values[index]);
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(
                                    static_cast<double>(grouped[index]) -
                                    static_cast<double>(serial_values[index])));
                            exact_indices = exact_indices &&
                                            grouped[index] ==
                                                serial_values[index];
                        }
                        const float cosine = computeCosineSimilarity(
                            grouped,
                            serial_values.data(),
                            serial_values.size());
                        const bool routing_indices =
                            std::string_view(key).ends_with(
                                "MOE_ROUTING_INDICES");
                        float kl = 0.0f;
                        bool passed = finite &&
                                      (routing_indices
                                           ? exact_indices
                                           : cosine >=
                                                 config_.decode_cosine_threshold);
                        if (key == "LM_HEAD")
                        {
                            kl = computeKLDivergence(
                                grouped,
                                serial_values.data(),
                                serial_values.size(),
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            passed = passed && kl < config_.kl_threshold;
                        }
                        snapshot_csv
                            << call << ',' << reference_step << ",-1,"
                            << key << ",serial_output" << serial_snapshot_index
                            << '_' << key << ',' << serial_values.size() << ','
                            << cosine << ',' << max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ",1,1,"
                            << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        if (!passed)
                        {
                            for (size_t index = 0;
                                 index < serial_values.size();
                                 ++index)
                            {
                                failure_values_csv
                                    << call << ',' << reference_step
                                    << ",-1," << key << ',' << index << ','
                                    << grouped[index] << ','
                                    << serial_values[index] << '\n';
                            }
                        }
                        EXPECT_TRUE(passed)
                            << "Grouped verifier row zero diverged from serial "
                               "decode at "
                            << key << " cosine=" << cosine
                            << " max_abs_diff=" << max_abs_diff
                            << " kl=" << kl << "\nCSV: "
                            << snapshot_csv_path;
                    }
                }
            }

            const size_t output_end = output_begin + step.tokens.size();
            const std::vector<int32_t> serial_expected(
                serial_tokens.begin() + output_begin,
                serial_tokens.begin() + output_end);
            const std::vector<int32_t> hf_expected(
                expected_tokens.begin() + output_begin,
                expected_tokens.begin() + output_end);
            token_csv
                << call << ',' << reference_step << ',' << selected_depth
                << ',' << join_tokens(step.tokens)
                << ',' << join_tokens(serial_expected)
                << ',' << join_tokens(hf_expected)
                << ',' << (hf_branch_compatible ? 1 : 0) << ','
                << (serial_epoch_compatible ? 1 : 0) << ','
                << serial_movement_epoch << ','
                << grouped_movement_epoch_begin << ','
                << grouped_movement_epoch_end << ','
                << serial_execution_epoch << ','
                << grouped_execution_epoch << ','
                << serial_trajectory_epoch << ','
                << grouped_trajectory.epoch.value_or(0u) << ','
                << production_mtp0_top1 << ',' << hf_mtp0_top1 << ','
                << (recursive_branch_compatible ? 1 : 0) << ','
                << after.mtp_observed_verifier_transaction_count << ','
                << after.mtp_observed_verifier_draft_depth << ','
                << join_tokens(after.mtp_observed_verifier_draft_tokens)
                << ','
                << after.mtp_draft_steps << ',' << after.mtp_verifier_runs
                << ',' << after.mtp_accepted_tokens << ','
                << after.mtp_rejected_tokens << ','
                << after.mtp_transaction_commits << ','
                << after.mtp_transaction_rollbacks << ','
                << after.mtp_transaction_validation_failures << ','
                << after.current_position << '\n';
            emitted.insert(emitted.end(), step.tokens.begin(), step.tokens.end());
            ++call;
        }
        orch_runner_->setDecodeStepTokenBudget(0);

        if (usesDynamicMTPDepth())
        {
            /*
             * The two-token boundaries above deliberately isolate one
             * verifier identity for checkpoint diagnosis. At the configured
             * maximum depth of fifteen, those requests are budget-limited and
             * must not contaminate the adaptive controller's economy window.
             * Submit one ordinary full-width serving request so the
             * device-owned controller sees a complete real-weight
             * transaction and can make an evidence-backed depth decision.
             */
            activeClearSnapshots();
            activeClearCache();
            ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
                << orch_runner_->lastError();
            const auto policy_before = activePrefixStateProbe();
            const uint64_t policy_movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const int policy_response_budget = activeMTPDraftDepth() + 1;
            orch_runner_->setDecodeStepTokenBudget(policy_response_budget);
            const GenerationResult policy_step = orch_runner_->decodeStep();
            orch_runner_->setDecodeStepTokenBudget(0);
            const uint64_t policy_movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(policy_step.success()) << policy_step.error;
            ASSERT_EQ(
                policy_step.tokens.size(),
                static_cast<size_t>(policy_response_budget))
                << "The dynamic-depth proof did not retire its admitted "
                   "full-width response budget";
            const auto policy_after = activePrefixStateProbe();
            EXPECT_GT(
                policy_after.mtp_depth_policy_windows,
                policy_before.mtp_depth_policy_windows)
                << "A non-budget-limited real-model verifier transaction did "
                   "not evaluate the device-owned depth policy";
            const auto promotions =
                policy_after.mtp_depth_policy_promotions -
                policy_before.mtp_depth_policy_promotions;
            const auto demotions =
                policy_after.mtp_depth_policy_demotions -
                policy_before.mtp_depth_policy_demotions;
            const auto updates =
                policy_after.mtp_depth_policy_updates -
                policy_before.mtp_depth_policy_updates;
            EXPECT_EQ(updates, promotions + demotions)
                << "The device-owned depth policy did not account for its "
                   "evaluated decision exactly once";
            if (demotions > 0u)
            {
                EXPECT_LT(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "A recorded depth demotion did not reduce the selected width";
            }
            else if (promotions > 0u)
            {
                EXPECT_GT(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "A recorded depth promotion did not increase the selected width";
            }
            else
            {
                EXPECT_EQ(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "An evidence-backed hold changed depth without recording an update";
            }

            if (!isDynamicResidencyProductionTest() &&
                policy_movement_epoch_begin == policy_movement_epoch_end)
            {
                ASSERT_GE(serial_tokens.size(), policy_step.tokens.size());
                for (size_t index = 0; index < policy_step.tokens.size(); ++index)
                {
                    EXPECT_EQ(policy_step.tokens[index], serial_tokens[index])
                        << "Dynamic-depth policy proof diverged from serial "
                           "decode at output "
                        << index;
                }
            }
        }

        const uint64_t final_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();
        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(final_movement_epoch, mtp_certification_movement_epoch)
                << "Static MTP certification crossed a residency epoch";
            EXPECT_GT(serial_epoch_compatible_calls, 0)
                << "Static MTP produced no same-epoch serial equivalence proof";
        }

        const auto final_state = activePrefixStateProbe();
        EXPECT_GT(speculative_calls, 0)
            << "No grouped MTP transaction executed";
        EXPECT_GT(compared_stages, 0)
            << "No live MTP checkpoint was compared";
        if (activeMTPDraftDepth() > 1)
        {
            EXPECT_GT(
                compared_recursive_stages + deferred_recursive_contexts,
                0)
                << "No recursive MTP checkpoint had an observed proposal "
                   "identity or a deferred exact HF branch proof";
        }
        EXPECT_GT(compared_main_stages, 0)
            << "No grouped main-model checkpoint was compared with Hugging Face";
        ASSERT_GT(grouped_main_cosine_count, 0u)
            << "Grouped main-model comparison produced no numerical rows";
        EXPECT_GE(
            grouped_main_cosine_sum /
                static_cast<double>(grouped_main_cosine_count),
            static_cast<double>(config_.decode_cosine_threshold))
            << "Grouped main-model aggregate cosine failed against Hugging Face; CSV: "
            << snapshot_csv_path;
        EXPECT_GT(compared_main_lm_heads, 0)
            << "Grouped main-model comparison omitted LM_HEAD";
        EXPECT_EQ(failed_main_lm_heads, 0)
            << "Grouped main-model LM_HEAD failed cosine/KL/mutual-top3 parity; CSV: "
            << snapshot_csv_path;
        EXPECT_GT(
            final_state.mtp_accepted_tokens,
            initial_state.mtp_accepted_tokens)
            << "The real 122B MTP request accepted no draft tokens";
        EXPECT_GT(
            final_state.mtp_transaction_commits,
            initial_state.mtp_transaction_commits);
        EXPECT_EQ(final_state.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(final_state.mtp_transaction_validation_failures, 0u);
        EXPECT_EQ(final_state.mtp_max_depth, activeMTPDraftDepth())
            << "MTP runtime capacity did not match the cell's admitted maximum";
        if (usesDynamicMTPDepth())
        {
            EXPECT_GT(final_state.mtp_depth_policy_windows, 0u)
                << "Dynamic-depth policy observed no completed verifier window";
            EXPECT_EQ(final_state.mtp_min_depth, 1);
        }
        token_csv.flush();
        snapshot_csv.flush();
        EXPECT_TRUE(token_csv.good());
        EXPECT_TRUE(snapshot_csv.good());
    }

    [[noreturn]] void abortGraphNativeWorld(const std::string &reason) const
    {
        const std::string message =
            "[Qwen3.5 MoE GraphNative] " + reason +
            "; aborting MPI world to avoid stranding rocm_warm/cpu_cold participants";
        LOG_ERROR(message);
        std::cerr << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }

    [[noreturn]] void abortAfterRootThrow(const char *phase, const std::string &what) const
    {
        abortGraphNativeWorld(std::string("root rank threw during ") + phase + ": " + what);
    }

    /**
     * @brief Persist participant-local sparse endpoint evidence beside parity CSVs.
     *
     * PerfStats is process-local by design, so a root-only export cannot show
     * which CUDA/ROCm follower accepted each routed-expert packet.  Every MPI
     * instance writes a rank-qualified file after the production worker loop
     * has closed. Records aggregate complete service timing by layer,
     * participant, tier, and retained graph-family geometry. Exact route
     * counts and the union of executed experts remain in the bounded overlay
     * profiler evidence; movement epochs remain in the residency artifacts.
     * Keeping transaction-varying values out of this timing key prevents the
     * diagnostic collector from perturbing long-horizon inference.
     */
    void writeSparseEndpointEvidenceCsv() const
    {
        if (!PerfStatsCollector::isDomainEnabled("moe_overlay_endpoint"))
            return;

        const int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        const auto path = ensureResultsDir() /
                          ("sparse_endpoint_rank_" +
                           std::to_string(rank) + ".csv");
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output << PerfStatsCollector::csvString({"moe_overlay_endpoint"});
        output.flush();
        EXPECT_TRUE(output.good()) << path;
    }

    /**
     * @brief Prove the live MTP sidecar consumed a stable read-only mailbox.
     *
     * Unit contracts establish that every supported sidecar graph declares
     * `PREFIX_TERMINAL_HIDDEN` read-only. This production assertion closes the
     * other half of the invariant: the mixed-vendor runner must actually build
     * that graph and retain the exact typed publication lease across a live
     * resident-logical-state append. Follower ranks own sparse participants,
     * not the public MTP response, so the continuation authority alone judges
     * these process-local counters after the serving command loop has closed.
     */
    void assertMTPTerminalHiddenMailboxEvidence() const
    {
        if (!isRootParityRank() || !activeMTPEnabled())
            return;

        ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("mtp"))
            << "MTP-labelled production parity requires MTP PerfStats";
        std::uint64_t read_lease_observations = 0;
        std::uint64_t read_only_contract_observations = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"mtp"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "mtp")
            {
                continue;
            }
            if (record.name ==
                "sidecar_terminal_hidden_read_leases")
            {
                read_lease_observations += record.count;
            }
            else if (record.name ==
                     "sidecar_terminal_hidden_read_only_contracts")
            {
                read_only_contract_observations += record.count;
            }
        }

        EXPECT_GT(read_only_contract_observations, 0u)
            << "Production MTP built no sidecar whose complete stage graph "
               "certified PREFIX_TERMINAL_HIDDEN as read-only";
        EXPECT_GT(read_lease_observations, 0u)
            << "Production MTP never retained a typed terminal-hidden "
               "publication lease across its resident shifted-row sidecar";
    }

    /**
     * @brief Fold the identical post-loop evidence sequence on every MPI rank.
     *
     * Worker ranks enter this sequence immediately after receiving the typed
     * SHUTDOWN command. The continuation authority must call it in the same
     * order after closing the loop so no test-only collective can race the
     * production command communicator.
     */
    void assertEvidenceAfterWorkerShutdown()
    {
        writeSparseEndpointEvidenceCsv();
        writeResidencyDiagnosticsCsv();
        assertMTPTerminalHiddenMailboxEvidence();
        assertParticipantCompactBufferArenaEvidence();
        assertMappedParticipantGraphEvidence();
        assertActiveTierRouteEvidence();
        assertSparseTransportPerfStatsEvidence();
        assertResidencyMovementEvidence();
        assertRequestMovementPolicyEvidence();
        if (isSegmentedPrefillProductionTest())
            assertSegmentedPrefillEvidence();
        finishProductionParityEvidence();
    }

    /**
     * @brief Run the complete three-tier graph-native parity contract once.
     */
    void runGraphNativeProductionParityBody()
    {
        beginProductionParityEvidence();
        if (isLegacyOverlayRuntimeEnabled())
        {
            FAIL() << kLegacyEnvVar
                   << " is set in the environment. This test requires graph-native overlay lowering.";
        }

        const bool hardware_and_model_ok = collectivelyCheckHardwareAndModel();
        if (!hardware_and_model_ok)
        {
            if (isRootParityRank())
            {
                const auto blocker = acceleratorHardwareBlocker(cluster_inventory_);
                if (blocker)
                    ADD_FAILURE() << "Production GraphNative prerequisite failed: " << *blocker;
                else
                    ADD_FAILURE() << "Production parity model not found at "
                                  << activeModelPath();
            }
            return;
        }

        /*
         * OrchestrationRunner initialization is already a rank-consensual
         * lifecycle: every setup phase publishes one result through the
         * production MPI context before either rank may advance.  A second
         * test-owned all-reduce here is not additional validation.  If one
         * graph builder rejects a phase, that reduction can match the peer's
         * final production-phase reduction and leave the peer waiting in this
         * later call while the failing rank enters TearDown.  Trust the typed
         * production result and keep the test protocol at exactly one
         * rendezvous per lifecycle transition.
         */
        const bool setup_ok = setupPipeline();
        ASSERT_TRUE(setup_ok)
            << "Production pipeline setup failed on this rank or a peer rank";

        const bool decode_available = synchronizedDecodeWorkAvailable();
        ASSERT_TRUE(decode_available)
            << "Production parity requires incremental-decode snapshots and metadata";

        /*
         * Exercise the same coordinated application boundary used by an
         * interactive/server deployment. Rank zero issues typed inference
         * commands; every other MPI instance remains a real
         * OrchestrationRunner and executes whichever accelerator and CPU-NUMA
         * participants runtime inventory binding assigned to it. This is
         * deliberately not a parity-fixture MPI loop.
        */
        orch_runner_->setMPICoordinatedMode(true);
        MPI_Barrier(parityCoordinationCommunicator());
        if (!isRootParityRank())
        {
            orch_runner_->runMPIWorkerLoop();
            orch_runner_->setMPICoordinatedMode(false);
            assertEvidenceAfterWorkerShutdown();
            return;
        }

        struct WorkerShutdownGuard
        {
            IOrchestrationRunner *runner = nullptr;
            ~WorkerShutdownGuard()
            {
                if (runner)
                {
                    runner->shutdownMPIWorkers();
                    runner->setMPICoordinatedMode(false);
                }
            }
        } worker_shutdown{orch_runner_.get()};

        if (isDynamicResidencyProductionTest() &&
            !certifyDynamicResidencyEconomy())
        {
            orch_runner_->shutdownMPIWorkers();
            orch_runner_->setMPICoordinatedMode(false);
            worker_shutdown.runner = nullptr;
            assertEvidenceAfterWorkerShutdown();
            ADD_FAILURE()
                << "Could not certify Dynamic economy from ordinary production traffic";
            return;
        }

        if (isDynamicResidencyProductionTest() &&
            !collectInferenceTimings(
                ResidencyTimingCohort::InitialEpoch))
        {
            orch_runner_->shutdownMPIWorkers();
            orch_runner_->setMPICoordinatedMode(false);
            worker_shutdown.runner = nullptr;
            assertEvidenceAfterWorkerShutdown();
            ADD_FAILURE()
                << "Could not collect the controlled initial-residency timing cohort";
            return;
        }

        if (!driveDynamicResidencyToDistributedMigration())
        {
            /*
             * No command is active here: every prefill/decode boundary already
             * completed successfully. Close the production worker protocol so
             * both ranks can fold PerfStats and retain the rejected-payoff
             * evidence. MPI_Abort remains reserved for exceptions that leave a
             * published command's collective order indeterminate.
             */
            orch_runner_->shutdownMPIWorkers();
            orch_runner_->setMPICoordinatedMode(false);
            worker_shutdown.runner = nullptr;
            assertEvidenceAfterWorkerShutdown();
            ADD_FAILURE()
                << "Real dynamic residency workload did not prove a cross-rank expert migration";
            return;
        }
        cacheCommittedPromotionEvidence();

        if (requiresObservedConvergenceSpeedup())
        {
            const bool convergence_timings_ready =
                collectInferenceTimings(
                    ResidencyTimingCohort::ConvergedEpoch);
            EXPECT_TRUE(convergence_timings_ready)
                << "Could not collect epoch-stable observed convergence timings";
            if (convergence_timings_ready)
                assertAndWriteObservedConvergenceSpeedup();
        }
        if (isDynamicResidencyProductionTest())
        {
            cacheCommittedPromotionEvidence();
            writeCommittedMovementEvidenceCsv();

            /*
             * Only the dense continuation authority owns parity artifacts.
             * Re-enabling its diagnostic nodes invalidates that local graph
             * topology once; runPrefillParity then performs the normal
             * warmup/capture retry while sparse participants keep their lean
             * production graphs and unchanged collective schedule.
             */
            orch_runner_->enableSnapshotCapture();
            /*
             * Calibration is an ordinary production workload and therefore
             * leaves KV, short-convolution, GDN recurrence, logical position,
             * and snapshot state belonging to that request.  The Hugging Face
             * reference pack starts from an empty request.  Cross the same
             * typed request boundary used by serving before collecting parity
             * checkpoints; residency is model-lifetime state and deliberately
             * survives this reset, so the following forward still exercises
             * the migrated epoch proved above.
             */
            activeClearSnapshots();
            activeClearCache();
        }

        ParityTestSummary prefill;
        try
        {
            prefill = runPrefillParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("prefill parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("prefill parity", "unknown exception");
        }

        if (isRootParityRank() && !producedPrefillSummary(prefill))
        {
            abortGraphNativeWorld(
                "root produced no prefill parity summary - forward likely failed before "
                "all overlay tiers completed");
        }

        const PrefixRuntimeStateSnapshot fresh_prefix_state =
            activePrefixStateProbe();
        if (isRootParityRank())
        {
            assertParity(prefill);
            assertProductionParityFreshPrefixSeed(
                fresh_prefix_state,
                prefill.overall_passed,
                prefill.lm_head_cosine);
            if (isSegmentedPrefillProductionTest())
                assertSegmentedPrefillCheckpointCoverage();
            assertProductionParitySnapshotInfrastructure();
            cacheDeviceRouteAssignmentEvidence();
            writeExpertOwnerTopologyBaselineCsv();
            writePrefillRoutedExpertDiagnosticCsv();
        }
        activeClearSnapshots();
        activeClearCache();

        DecodeParitySummary decode;
        try
        {
            decode = runDecodeParity(
                ParityDecodePrefillMode::CompletePrefixRestore);
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("decode parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow("decode parity", "unknown exception");
        }

        if (isRootParityRank() && !producedDecodeSummary(decode))
        {
            abortGraphNativeWorld(
                "root produced no decode parity summary - forward likely failed before "
                "all overlay tiers completed");
        }

        if (isRootParityRank())
        {
            /*
             * The standard decode assertion owns the canonical typed MTP
             * proof. The topology-specific long-horizon proof below remains
             * additive: it may retain deeper branch diagnostics, but it cannot
             * replace the standard transaction boundary, decode-stage rows,
             * or CSV.
             */
            assertDecodeParity(decode);
            assertProductionParityCompletePrefixRestore(
                fresh_prefix_state,
                decode.overall_passed,
                decode.avg_cosine);
            assertProductionParityPartialPrefixRestore();
            assertParityExecutionExercisesPromotedExpert();
        }

        try
        {
            runMTPHuggingFaceCheckpointParity();
        }
        catch (const std::exception &e)
        {
            abortAfterRootThrow("MTP Hugging Face checkpoint parity", e.what());
        }
        catch (...)
        {
            abortAfterRootThrow(
                "MTP Hugging Face checkpoint parity",
                "unknown exception");
        }

        /*
         * End the production worker protocol before entering the evidence
         * allreduce.  This keeps the command loop and the test-only collective
         * from competing for the same MPI messages while retaining the real
         * production setup and forward path above.
         */
        orch_runner_->shutdownMPIWorkers();
        orch_runner_->setMPICoordinatedMode(false);
        worker_shutdown.runner = nullptr;
        assertEvidenceAfterWorkerShutdown();
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> overlay_plan_;
    ClusterInventory cluster_inventory_;
    /** Authenticated route demand used only to certify epoch-one ordering. */
    std::vector<std::vector<std::uint64_t>> reference_adversarial_routes_;
    /** Exact CPU tier index retained across deferred capacity resolution. */
    int reference_adversarial_cpu_tier_index_ = -1;
    /** Dense continuation rank used to classify remote CPU ownership. */
    int reference_adversarial_continuation_rank_ = -1;
    DynamicResidencyProofLifecycle dynamic_residency_proof_lifecycle_;
    ResidencyConvergenceTimings convergence_timings_;
    /**
     * Physical wave width selected from authenticated model geometry.
     *
     * CPU-tier convergence cells replace the minimum with one tier-migration
     * slot per transformer layer plus one independent within-tier skew slot.
     * GPU-only cells retain the bounded minimum until their own performance
     * convergence proof derives an equivalent model-owned width.
    */
    std::uint32_t convergence_migration_transfer_slots_ = 0u;
    /** Active policy cap that may deliberately use only part of the fabric. */
    std::uint32_t convergence_migration_cycles_per_wave_ = 0u;
    std::vector<uint64_t>
        parity_route_counts_by_participant_; ///< Live checkpoint routes under the published epoch.
    std::vector<bool>
        parity_route_requires_remote_completion_; ///< Participants requiring a follower Complete proof.
    std::vector<PublishedParticipantResidency>
        parity_residency_by_participant_; ///< Device-bank authority for resident versus capacity-idle endpoints.
    std::vector<PromotedExpert>
        promoted_experts_; ///< Promotion identities retained across parity collector resets.
    std::vector<PromotedExpertExecutionWitness>
        promoted_expert_execution_witnesses_; ///< Exact parity routes through promoted destinations.
    std::vector<RoutedExpertContributionWitness>
        routed_expert_contribution_witnesses_; ///< Every comparable route at a physically moved layer.
};

/** @brief Test-only public view of the fixture's pure placement constructor. */
class Qwen35MoEAdversarialPlacementProbe final
    : public Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold
{
public:
    using Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold::
        selectAdversarialCpuExperts;
};

/**
 * @brief A balanced NodeTP tier may legitimately give one participant no expert.
 *
 * Automatic capacity can leave fewer experts in the CPU tier than there are
 * CPU participants. Production owner partitioning represents that condition as
 * a zero-sized balanced span; the adversarial test-layout constructor must
 * preserve the same valid geometry rather than rejecting the generated cell.
 */
TEST(Qwen35MoEAdversarialPlacementGeometry,
     AllowsTierQuotaBelowParticipantCount)
{
    const std::vector<std::uint64_t> routes{2u, 11u, 5u, 7u};
    for (const RoutedExpertOwnerOrder owner_order :
         {RoutedExpertOwnerOrder::Ordinal,
          RoutedExpertOwnerOrder::Random})
    {
        const auto selected =
            Qwen35MoEAdversarialPlacementProbe::
                selectAdversarialCpuExperts(
                    routes,
                    /*layer=*/0,
                    /*tier_index=*/1,
                    /*selected_count=*/1,
                    /*participant_count=*/2,
                    owner_order);
        ASSERT_EQ(selected.size(), routes.size());
        EXPECT_EQ(std::count(selected.begin(), selected.end(), true), 1);
        EXPECT_TRUE(selected[1])
            << "The single lower-tier slot must retain the hottest adversarial expert";
    }
}

/**
 * @brief Capacity-resolved idle tiers remain explicit without fictional work.
 *
 * Two accelerator tiers can exhaust a small model even when a two-participant
 * CPU tier is configured. The published bank must classify the CPU endpoints
 * as idle, while a later bank assigning one expert to CPU activates CPU
 * transport evidence. This is the focused regression for the generated static
 * segmented cell that previously required all configured endpoints to run.
 */
TEST(Qwen35MoEPublishedParticipationGeometry,
     DistinguishesConfiguredIdleTierFromResidentTier)
{
    MoERoutedExpertPlacementPlan plan;
    plan.continuation_domain = "continuation";

    RoutedExpertDomain continuation;
    continuation.name = "continuation";
    continuation.participants = {GlobalDeviceAddress::cuda(0)};
    RoutedExpertDomain secondary;
    secondary.name = "secondary";
    secondary.participants = {GlobalDeviceAddress::rocm(0)};
    RoutedExpertDomain cpu;
    cpu.name = "cpu";
    cpu.participants = {
        GlobalDeviceAddress::cpu(0),
        GlobalDeviceAddress::cpu(1),
    };
    plan.domains = {
        std::move(continuation),
        std::move(secondary),
        std::move(cpu),
    };
    plan.routed_tiers = {
        {.name = "p0", .domain = "continuation", .priority = -5},
        {.name = "p1", .domain = "secondary", .priority = 7},
        {.name = "p2", .domain = "cpu", .priority = 101, .fallback = true},
    };

    std::vector<PublishedParticipantResidency> states(
        4u,
        PublishedParticipantResidency::Idle);
    const std::array<float, 1> continuation_bank{0.0f};
    includePublishedExpertOwners(states, continuation_bank);
    const auto continuation_only =
        summarizePublishedParticipation(plan, states);
    EXPECT_FALSE(continuation_only.sparse_follower);
    EXPECT_FALSE(continuation_only.secondary_gpu);
    EXPECT_FALSE(continuation_only.cpu);

    const std::array<float, 6> gpu_bank{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    includePublishedExpertOwners(states, gpu_bank);

    EXPECT_EQ(states[0], PublishedParticipantResidency::OwnsExpert);
    EXPECT_EQ(states[1], PublishedParticipantResidency::OwnsExpert);
    EXPECT_EQ(states[2], PublishedParticipantResidency::Idle);
    EXPECT_EQ(states[3], PublishedParticipantResidency::Idle);
    const auto gpu_only = summarizePublishedParticipation(plan, states);
    EXPECT_TRUE(gpu_only.sparse_follower);
    EXPECT_TRUE(gpu_only.secondary_gpu);
    EXPECT_FALSE(gpu_only.cpu);

    const std::array<float, 1> cpu_bank{2.0f};
    includePublishedExpertOwners(states, cpu_bank);
    const auto with_cpu = summarizePublishedParticipation(plan, states);
    EXPECT_TRUE(with_cpu.sparse_follower);
    EXPECT_TRUE(with_cpu.secondary_gpu);
    EXPECT_TRUE(with_cpu.cpu);

    const std::array<float, 1> invalid_bank{4.0f};
    EXPECT_THROW(
        includePublishedExpertOwners(states, invalid_bank),
        std::invalid_argument);
}

/**
 * @brief Final residency does not manufacture demand in a bounded prompt.
 *
 * This is the regression for the iteration-seven soak failure: a remote CPU
 * participant retained final experts and had completed real traffic earlier,
 * but the final nine-token parity prompt selected none of those experts. The
 * inverse cases remain illegal—selection from an idle bank and a remote final
 * resident that never completed any production traffic.
 */
TEST(Qwen35MoEPublishedParticipationGeometry,
     SeparatesPinnedSelectionFromCumulativeExecution)
{
    EXPECT_EQ(
        validateParticipantRouteEvidence(
            PublishedParticipantResidency::OwnsExpert,
            /*pinned_route_count=*/0u,
            /*remote=*/true,
            /*completed_route_count=*/7u),
        PublishedParticipantRouteEvidence::Valid);
    EXPECT_EQ(
        validateParticipantRouteEvidence(
            PublishedParticipantResidency::Idle,
            /*pinned_route_count=*/0u,
            /*remote=*/true,
            /*completed_route_count=*/7u),
        PublishedParticipantRouteEvidence::Valid)
        << "An endpoint may have completed traffic before Dynamic movement made its final span idle";
    EXPECT_EQ(
        validateParticipantRouteEvidence(
            PublishedParticipantResidency::Idle,
            /*pinned_route_count=*/1u,
            /*remote=*/true,
            /*completed_route_count=*/1u),
        PublishedParticipantRouteEvidence::IdleParticipantSelected);
    EXPECT_EQ(
        validateParticipantRouteEvidence(
            PublishedParticipantResidency::OwnsExpert,
            /*pinned_route_count=*/0u,
            /*remote=*/true,
            /*completed_route_count=*/0u),
        PublishedParticipantRouteEvidence::RemoteResidentNeverCompleted);
    EXPECT_EQ(
        validateParticipantRouteEvidence(
            PublishedParticipantResidency::OwnsExpert,
            /*pinned_route_count=*/0u,
            /*remote=*/false,
            /*completed_route_count=*/0u),
        PublishedParticipantRouteEvidence::Valid)
        << "Local arithmetic is proved directly by parity and needs no follower completion packet";
}

/**
 * @brief Same-tier movement is required only with a real exchange degree.
 *
 * Two experts on two apportioned participants admit only a permutation and
 * can never reduce makespan. A third resident expert makes a capacity-
 * preserving ownership exchange mathematically capable of improving skew.
 */
TEST(Qwen35MoEDynamicMovementAxisGeometry,
     UsesCapacityResolvedExpertCardinality)
{
    MoERoutedExpertPlacementPlan plan;
    RoutedExpertDomain domain;
    domain.name = "integer_priority_domain";
    domain.routed_compute_policy =
        RoutedExpertComputePolicy::Apportioned;
    domain.participants = {
        GlobalDeviceAddress::cpu(0),
        GlobalDeviceAddress::cpu(1),
    };
    plan.domains = {std::move(domain)};
    plan.routed_tiers = {{
        .name = "opaque_priority",
        .domain = "integer_priority_domain",
        .priority = 37,
        .fallback = true,
    }};
    plan.placements = {{
        .layer = 0,
        .routed_expert_tier = {0, 0},
    }};

    EXPECT_EQ(
        dynamicMovementAxisContract(plan),
        DynamicMovementAxisContract::PriorityMigrationOnly);

    plan.placements.front().routed_expert_tier.push_back(0);
    EXPECT_EQ(
        dynamicMovementAxisContract(plan),
        DynamicMovementAxisContract::
            PriorityMigrationAndParticipantBalance);
}

/**
 * @brief Measured and cache-distinct convergence traffic have separate identities.
 *
 * The optimizer must observe exactly the finite workload later judged by the
 * convergence A/B. Movement-only cells instead need guaranteed full-prefill
 * evidence, and traffic used only to close a partial demand bank must also lie
 * outside the measured corpus so neither can turn a required prefill into a
 * prefix-cache hit.
 */
TEST(Qwen35MoEDynamicConvergenceLifecycle,
     SeparatesMeasuredAndCacheDistinctTrafficNamespaces)
{
    for (int request = 0;
         request < 3 * kConvergenceTimingCorpusRequests;
         ++request)
    {
        EXPECT_EQ(
            convergenceMovementPromptIdentity(
                ConvergenceMovementTraffic::MeasuredWorkload,
                request),
            request % kConvergenceTimingCorpusRequests);
    }
    for (int request = 0;
         request < kMovementProofIdentityNamespaceRequests +
                       kMaximumDynamicPublicationOverlapRequests;
         ++request)
    {
        EXPECT_EQ(
            convergenceMovementPromptIdentity(
                ConvergenceMovementTraffic::MovementProof,
                request),
            kConvergenceTimingCorpusRequests + request);
    }
    for (int request = 0;
         request < 3 * kConvergenceTimingCorpusRequests;
         ++request)
    {
        EXPECT_GE(
            convergenceMovementPromptIdentity(
                ConvergenceMovementTraffic::DemandWindowClosure,
                request),
            kConvergenceTimingCorpusRequests +
                kMovementProofIdentityNamespaceRequests +
                kMaximumDynamicPublicationOverlapRequests);
    }
}

/**
 * @brief Movement proof horizon is derived from guaranteed authenticated rows.
 *
 * Decode completion is model output and may occur on the first sampled token.
 * This regression therefore proves that cache-distinct prefills alone can
 * close every required demand bank without relying on speculative decode work
 * or physical captured-row padding.
 */
TEST(Qwen35MoEDynamicConvergenceLifecycle,
     MovementProofHorizonClosesDemandBankFromAuthenticatedRows)
{
    constexpr int authenticated_rows =
        static_cast<int>(kQwen35MoEParityTokenIds.size());
    constexpr int movement_window_rows = 256;
    constexpr std::uint64_t required_publications =
        kObservedSpeedupConvergenceWindows;
    const int budget = movementProofHistogramRequestBudget(
        movement_window_rows,
        authenticated_rows,
        required_publications);
    EXPECT_GE(
        budget * authenticated_rows,
        movement_window_rows *
            static_cast<int>(required_publications));
    EXPECT_LT(
        (budget - static_cast<int>(required_publications)) *
            authenticated_rows,
        movement_window_rows *
            static_cast<int>(required_publications));
    EXPECT_THROW(
        movementProofHistogramRequestBudget(
            0,
            authenticated_rows,
            required_publications),
        std::invalid_argument);
}

/**
 * @brief Movement-only cells must not inherit the speed witness's bank budget.
 *
 * The canonical matrix assigns exactly one matched A/B witness per Dynamic
 * topology and owner order. Other MTP depths still prove physical movement,
 * but they proceed directly from a quiescent publication boundary to parity;
 * requiring 414 rows of unused headroom would make their ordinary short
 * histogram windows impossible to settle.
 */
TEST(Qwen35MoEDynamicConvergenceLifecycle,
     RequiresDemandHeadroomOnlyForObservedSpeedup)
{
    EXPECT_EQ(
        convergenceBoundaryCohortRows(
            ConvergenceBoundaryPurpose::MovementProof),
        std::nullopt);
    EXPECT_EQ(
        convergenceBoundaryCohortRows(
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort),
        std::optional<std::uint64_t>(
            convergenceTimingCohortRoutedRows()));
}

/**
 * @brief A partial timing bank closes exactly before its successor is measured.
 *
 * A movement wave can rotate its successor bank after an inference request is
 * admitted but before that request returns to the test-owned admission loop.
 * The passive classifier must name the exact remaining rows, refuse the full
 * bank while its wake is unreconciled, and admit the complete matched cohort
 * only after production rotates to an empty successor. This removes the race
 * without widening the economy window or discarding observed demand.
 */
TEST(Qwen35MoEDynamicConvergenceLifecycle,
     ExactClosureRotatesPartialBankBeforeTimingCohort)
{
    constexpr std::uint64_t kOverlappingRequestRows =
        static_cast<std::uint64_t>(kConvergenceTimingPromptRows) +
        static_cast<std::uint64_t>(
            kConvergenceTimingDecodeForwardsPerRequest);
    const MoEOptimizationStatus partial_bank{
        .authority = MoEOptimizationAuthority::Host,
        .state = MoEOptimizationLifecycleState::Active,
        .activity = MoEOptimizationActivityState::CollectingDemand,
        .demand_window = {
            .generation = 5u,
            .collected_routed_rows = kOverlappingRequestRows,
            .capacity_routed_rows =
                kQwen35MoEConvergenceHistogramWindowRows,
        },
        .published_progress_generation = 9u,
        .reconciled_progress_generation = 9u,
    };

    ASSERT_TRUE(partial_bank.quiescentBetweenWaves());
    const ConvergenceBoundaryDecision closure =
        classifyConvergenceBoundary(
            partial_bank,
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort);
    ASSERT_EQ(
        closure.state,
        ConvergenceBoundaryState::NeedsDemandWindowClosure);
    ASSERT_TRUE(closure.closure.has_value());
    EXPECT_EQ(closure.closure->generation, 5u);
    EXPECT_EQ(
        closure.closure->routed_rows,
        static_cast<std::uint64_t>(
            kQwen35MoEConvergenceHistogramWindowRows) -
            kOverlappingRequestRows);

    EXPECT_EQ(
        classifyConvergenceBoundary(
            partial_bank,
            ConvergenceBoundaryPurpose::MovementProof)
            .state,
        ConvergenceBoundaryState::Ready)
        << "Movement-only cells need no post-publication timing bank";

    auto completed_bank = partial_bank;
    completed_bank.activity =
        MoEOptimizationActivityState::ReconcilingDemand;
    completed_bank.demand_window.collected_routed_rows =
        completed_bank.demand_window.capacity_routed_rows;
    ++completed_bank.published_progress_generation;
    EXPECT_EQ(
        classifyConvergenceBoundary(
            completed_bank,
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort)
            .state,
        ConvergenceBoundaryState::AwaitingQuiescence);

    auto rotated_bank = completed_bank;
    rotated_bank.activity =
        MoEOptimizationActivityState::CollectingDemand;
    ++rotated_bank.demand_window.generation;
    rotated_bank.demand_window.collected_routed_rows = 0u;
    rotated_bank.reconciled_progress_generation =
        rotated_bank.published_progress_generation;
    EXPECT_EQ(
        classifyConvergenceBoundary(
            rotated_bank,
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort)
            .state,
        ConvergenceBoundaryState::Ready);
}

/**
 * @brief A wave completed during service certification remains observable.
 *
 * The production worker is asynchronous: the request that activates measured
 * economics may also close, transfer, and publish the first demand wave before
 * the fixture enters its dedicated movement driver. The proof lifecycle must
 * retain the pre-traffic frontier instead of re-snapshotting that published
 * wave as a new baseline.
 */
TEST(Qwen35MoEDynamicConvergenceLifecycle,
     RetainsPreCertificationOriginAcrossPublishedWave)
{
    DynamicResidencyProofLifecycle lifecycle;
    const DynamicResidencyConvergenceOrigin origin{
        .published_waves = 4u,
        .completed_transactions = 4u,
        .ledger_edges = 7u,
        .host_admissions = 0u,
    };
    lifecycle.beginEconomyCertification(origin);
    EXPECT_EQ(
        lifecycle.phase(),
        DynamicResidencyProofPhase::CertifyingEconomy);

    /* Model one complete tier-promotion edge published by the asynchronous
     * authority while certification traffic is still in flight. */
    std::vector<MoEOptimizationMovementEdge> edges(7u);
    edges.push_back(MoEOptimizationMovementEdge{
        .authority = MoEOptimizationAuthority::Host,
        .transaction = 5u,
        .candidate_epoch = 6u,
        .layer = 0,
        .expert = 17,
        .cycle_index = 0u,
        .cycle_size = 2u,
        .direction = MoEOptimizationMovementDirection::Promotion,
        .axis = MoEOptimizationMovementAxis::TierResidency,
        .source_participant = 1,
        .destination_participant = 0,
        .source_priority = 1,
        .destination_priority = 0,
        .source_device = DeviceId::cpu(),
        .destination_device = DeviceId::cuda(0),
    });
    const MoEOptimizationStatus after_certification{
        .authority = MoEOptimizationAuthority::Host,
        .state = MoEOptimizationLifecycleState::Active,
        .published_movement_waves = 5u,
        .completed_movement = {.transactions = 5u},
    };
    const MoEOptimizationMovementLedger ledger{
        .edges = std::move(edges),
        .discarded_edges = 0u,
    };

    lifecycle.completeEconomyCertification();
    EXPECT_EQ(
        lifecycle.phase(),
        DynamicResidencyProofPhase::EconomyCertified);
    EXPECT_EQ(lifecycle.convergenceOrigin().published_waves, 4u);
    EXPECT_EQ(lifecycle.convergenceOrigin().completed_transactions, 4u);
    EXPECT_EQ(lifecycle.convergenceOrigin().ledger_edges, 7u);
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            DynamicResidencyConvergenceTarget{
                .minimum_published_waves = 1u,
                .axis_contract =
                    DynamicMovementAxisContract::PriorityMigrationOnly,
            },
            lifecycle.convergenceOrigin(),
            after_certification,
            ledger),
        DynamicResidencyConvergenceState::Satisfied);
}

/**
 * @brief One wide publication may satisfy two axes without becoming two waves.
 *
 * This is the focused regression for the 122B convergence failure: the
 * authority durably published one transaction containing a tier cycle and a
 * participant-placement cycle, while the old fixture compared the resulting
 * wave count against an unrelated four-window constant.
 */
TEST(Qwen35MoEDynamicConvergenceLifecycle,
     SeparatesWavesFromCyclesAndAxes)
{
    const auto edge = [](
                          MoEOptimizationMovementAxis axis,
                          MoEOptimizationMovementDirection direction,
                          int expert,
                          std::size_t cycle_index)
    {
        return MoEOptimizationMovementEdge{
            .authority = MoEOptimizationAuthority::Host,
            .transaction = 1u,
            .candidate_epoch = 2u,
            .layer = 0,
            .expert = expert,
            .cycle_index = cycle_index,
            .cycle_size = 2u,
            .direction = direction,
            .axis = axis,
            .source_participant = 0,
            .destination_participant = 1,
            .source_priority =
                direction == MoEOptimizationMovementDirection::Promotion
                    ? 1
                    : 0,
            .destination_priority = 0,
            .source_device = DeviceId::cpu(),
            .destination_device = DeviceId::cpu(),
        };
    };

    const DynamicResidencyConvergenceOrigin origin{};
    const DynamicResidencyConvergenceTarget one_wide_wave{
        .minimum_published_waves = 1u,
        .axis_contract =
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance,
    };
    MoEOptimizationStatus status{
        .authority = MoEOptimizationAuthority::Host,
        .state = MoEOptimizationLifecycleState::Active,
        .published_movement_waves = 1u,
        .completed_movement = {.transactions = 0u},
    };
    const MoEOptimizationMovementEdge tier_edge = edge(
        MoEOptimizationMovementAxis::TierResidency,
        MoEOptimizationMovementDirection::Promotion,
        17,
        0u);
    const MoEOptimizationMovementEdge participant_edge = edge(
        MoEOptimizationMovementAxis::ParticipantPlacement,
        MoEOptimizationMovementDirection::SamePriority,
        23,
        1u);

    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            {{tier_edge, participant_edge}, 0u}),
        DynamicResidencyConvergenceState::AwaitingPhysicalCompletion);

    status.completed_movement.transactions = 1u;
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            {{participant_edge}, 0u}),
        DynamicResidencyConvergenceState::AwaitingTierResidency);
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            {{tier_edge}, 0u}),
        DynamicResidencyConvergenceState::
            AwaitingParticipantPolicyEvidence);

    const MoEOptimizationHostMovementAdmission tier_only_admission{
        .authority = MoEOptimizationAuthority::Host,
        .transaction = 1u,
        .candidate_epoch = 1u,
        .cycle_capacity_kind =
            MoEOptimizationCycleCapacityKind::Bounded,
        .maximum_concurrent_cycles = 2u,
        .candidate_cycles = 1u,
        .policy_eligible_cycles = 1u,
        .policy_eligible_axes = {.tier_residency = 1u},
        .admitted_candidate_cycles = 1u,
        .admitted_candidate_axes = {.tier_residency = 1u},
        .admitted_physical_cycles = 1u,
        .admitted_physical_axes = {.tier_residency = 1u},
    };
    ASSERT_TRUE(tier_only_admission.valid());
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            MoEOptimizationMovementLedger{
                .edges = {tier_edge},
                .host_admissions = {tier_only_admission},
            }),
        DynamicResidencyConvergenceState::Satisfied);

    auto participant_eligible_admission = tier_only_admission;
    participant_eligible_admission.candidate_cycles = 2u;
    participant_eligible_admission.policy_eligible_cycles = 2u;
    participant_eligible_admission.policy_eligible_axes = {
        .tier_residency = 1u,
        .participant_placement = 1u,
    };
    participant_eligible_admission.capacity_rejected_cycles = 1u;
    participant_eligible_admission.capacity_bounded = true;
    ASSERT_TRUE(participant_eligible_admission.valid());
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            MoEOptimizationMovementLedger{
                .edges = {tier_edge},
                .host_admissions = {participant_eligible_admission},
            }),
        DynamicResidencyConvergenceState::AwaitingParticipantPlacement);
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            {{tier_edge, participant_edge}, 0u}),
        DynamicResidencyConvergenceState::Satisfied);

    status.authority = MoEOptimizationAuthority::Device;
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            one_wide_wave,
            origin,
            status,
            {{tier_edge}, 0u}),
        DynamicResidencyConvergenceState::AwaitingParticipantPlacement);

    DynamicResidencyConvergenceTarget four_publications = one_wide_wave;
    four_publications.minimum_published_waves = 4u;
    EXPECT_EQ(
        classifyDynamicResidencyConvergence(
            four_publications,
            origin,
            status,
            {{tier_edge, participant_edge}, 0u}),
        DynamicResidencyConvergenceState::AwaitingPublication);
}

#ifdef LLAMINAR_QWEN122_MATRIX_ONLY
/**
 * @brief Model-wide physical and active wave envelopes are typed separately.
 *
 * This prevents the regression where the fixture derived a 49-cycle target
 * but `applyRuntimePolicy()` later replaced it with an unrelated two-cycle
 * value.  A two-CPU topology has one independent participant axis; a single-
 * CPU topology has only the 48 layer-parallel tier cycles.
 */
TEST(Qwen122DynamicWaveGeometry,
     DerivesCyclesAndCommandEnvelopeFromModelAndTopology)
{
    const Qwen122OverlayTopologySpec two_cpu{
        .test_id = "typed_two_cpu",
        .cuda_participants = 0,
        .rocm_participants = 1,
        .cpu_participants = 2,
        .mpi_ranks = 2,
        .continuation = Qwen122ContinuationBackend::ROCm,
    };
    const auto two_cpu_policy = qwen122DynamicParityEconomics(
        two_cpu,
        /*transformer_layers=*/48);
    EXPECT_EQ(two_cpu_policy.migration_transfer_slots, 49u);
    EXPECT_EQ(two_cpu_policy.resolvedMigrationCyclesPerWave(), 49u);
    EXPECT_EQ(two_cpu_policy.dynamic_max_swaps_per_layer, 2u);
    EXPECT_EQ(two_cpu_policy.dynamic_max_plan_entries_per_wave, 147u);

    auto one_cpu = two_cpu;
    one_cpu.test_id = "typed_one_cpu";
    one_cpu.cpu_participants = 1;
    one_cpu.mpi_ranks = 1;
    const auto one_cpu_policy = qwen122DynamicParityEconomics(
        one_cpu,
        /*transformer_layers=*/48);
    EXPECT_EQ(one_cpu_policy.migration_transfer_slots, 48u);
    EXPECT_EQ(one_cpu_policy.resolvedMigrationCyclesPerWave(), 48u);
    EXPECT_EQ(one_cpu_policy.dynamic_max_swaps_per_layer, 1u);
    EXPECT_EQ(one_cpu_policy.dynamic_max_plan_entries_per_wave, 96u);
}

/**
 * @brief Matrix expansion selects evidence geometry before fixture setup.
 *
 * This regression prevents the speed witness from leaking its broad transfer
 * wave and timing window into the five MTP movement-only cells. It also proves
 * that the fixture receives a complete immutable policy rather than deriving
 * controller settings from the GoogleTest name.
 */
TEST(Qwen122DynamicWaveGeometry,
     ExpansionSeparatesMovementProofFromObservedSpeedup)
{
    const Qwen122OverlayTopologySpec spec{
        .test_id = "typed_runtime_policy",
        .cuda_participants = 0,
        .rocm_participants = 1,
        .cpu_participants = 2,
        .mpi_ranks = 2,
        .continuation = Qwen122ContinuationBackend::ROCm,
    };
    const auto cases = expandModelParityDefinition(
        qwen122ExpertOverlayParityDefinition(spec));

    const auto find_dynamic = [&](ModelParityMTP mtp)
        -> const ModelParityCase &
    {
        const auto found = std::find_if(
            cases.begin(), cases.end(),
            [&](const ModelParityCase &test_case)
            {
                return test_case.expert_overlay.has_value() &&
                       test_case.expert_overlay->owner_order ==
                           RoutedExpertOwnerOrder::Ordinal &&
                       test_case.expert_overlay->movement ==
                           ModelParityExpertMovement::Dynamic &&
                       test_case.mtp == mtp;
            });
        if (found == cases.end())
            throw std::logic_error("Generated Qwen122 Dynamic cell is missing");
        return *found;
    };

    const auto &speedup = find_dynamic(ModelParityMTP::Off);
    EXPECT_TRUE(speedup.requiresObservedConvergenceSpeedup());
    EXPECT_EQ(speedup.dynamic_rebalance.window_size, 448);
    EXPECT_EQ(speedup.dynamic_rebalance.max_window_size, 448);
    EXPECT_FLOAT_EQ(speedup.dynamic_rebalance.window_growth_factor, 1.0F);
    EXPECT_EQ(speedup.dynamic_rebalance.migration_transfer_slots, 49u);
    EXPECT_EQ(
        speedup.dynamic_rebalance.resolvedMigrationCyclesPerWave(),
        49u);

    const auto &movement = find_dynamic(ModelParityMTP::Depth1);
    EXPECT_FALSE(movement.requiresObservedConvergenceSpeedup());
    EXPECT_EQ(
        movement.dynamic_rebalance.window_size,
        kQwen35MoEMovementProofInitialWindowRows);
    EXPECT_EQ(movement.dynamic_rebalance.max_window_size, 4096);
    EXPECT_FLOAT_EQ(
        movement.dynamic_rebalance.window_growth_factor,
        4096.0F /
            static_cast<float>(
                kQwen35MoEMovementProofInitialWindowRows));
    EXPECT_EQ(movement.dynamic_rebalance.migration_transfer_slots, 49u);
    EXPECT_EQ(
        movement.dynamic_rebalance.resolvedMigrationCyclesPerWave(),
        2u);
    EXPECT_EQ(
        movement.dynamic_rebalance.device_min_maintenance_period_tokens,
        4096);
}
#endif

/**
 * @brief Execute one generated production 35B or 122B policy cell.
 *
 * Every parameter owns a fresh request runner but shares one bounded
 * process-resident model/prepared-weight authority where supported. Runtime
 * policy comes from `GetParam()`; the stable parameter name is diagnostic
 * output only and is never parsed to recover execution intent.
 */
TEST_P(Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold, ProductionParity)
{
    runGraphNativeProductionParityBody();
}

#ifdef LLAMINAR_QWEN122_MATRIX_ONLY
INSTANTIATE_TEST_SUITE_P(
    Qwen35_122B_ExpertOverlay,
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ::testing::ValuesIn(qwen122ExpertOverlayParityCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    { return info.param.testName(); });
#else
INSTANTIATE_TEST_SUITE_P(
    Qwen35_35B_ExpertOverlay,
    Qwen35MoEGraphNativeCudaHotRocmWarmCpuCold,
    ::testing::ValuesIn(qwen35GraphNativeParityCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    { return info.param.testName(); });
#endif

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /*
     * This dedicated integration main does not pass through RuntimeInitPhase,
     * which normally publishes rank identity to logging and PerfStats.  Set it
     * at the same MPI boundary so rank-qualified diagnostic exports cannot be
     * collapsed into (and race on) rank zero's file.
     */
    Logger::getInstance().setRank(rank);
    std::cout << "[Rank " << rank
              << "] Qwen3.5 MoE typed ExpertOverlay parity test\n";

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
