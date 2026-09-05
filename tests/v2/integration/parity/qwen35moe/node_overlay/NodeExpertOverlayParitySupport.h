/**
 * @file NodeExpertOverlayParitySupport.h
 * @brief Shared typed policy, topology, and reference declarations.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

#include "../../ModelParityDefinition.h"
#include "../Qwen35MoEModelParityDefinitions.h"
#include "../Qwen35MoEParityTestBase.h"
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
#include "execution/runner/ModelContextRetirement.h"
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

namespace llaminar2::test::parity::qwen35moe::node_overlay
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
    extern const ModelParityCase *g_active_model_parity_case;

    /** @return The active typed canonical case, or null outside a fixture. */
    const ModelParityCase *activeModelParityCase() noexcept;

    /** @return Active generated case; throws outside a parameterized fixture. */
    const ModelParityCase &activeModelParityCaseOrThrow();

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
        /** Promotions selected by the authenticated movement-proof prompt. */
        std::uint64_t minimum_authenticated_promotions = 1u;
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
        NumericalParityReady,
        AwaitingMTPParity,
        NumericalEvidenceComplete,
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
            case 7:
                return DynamicResidencyProofPhase::NumericalParityReady;
            case 8:
                return DynamicResidencyProofPhase::AwaitingMTPParity;
            case 9:
                return DynamicResidencyProofPhase::
                    NumericalEvidenceComplete;
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
         * @brief Publish isolation of proof traffic from numerical parity.
         *
         * Movement-only traffic deliberately uses the Hugging Face prompt so
         * the newly promoted experts execute before the movement proof closes.
         * If publication completes before that request is archived, its prefix
         * entry is valid in the new placement epoch. Numerical parity must
         * therefore cross the production prefix-purge boundary before it may
         * claim a fresh captured prefill. The economic A/B path has the same
         * requirement after its converged cohort.
         */
        void recordNumericalParityIsolation()
        {
            if (const auto *state =
                    std::get_if<MovementBoundarySettled>(&state_))
            {
                state_ = NumericalParityReady{state->origin};
                return;
            }
            if (const auto *state =
                    std::get_if<ConvergedCohortMeasured>(&state_))
            {
                state_ = NumericalParityReady{state->origin};
                return;
            }
            throw std::logic_error(
                "Dynamic numerical parity isolation completed out of order");
        }

        /**
         * @brief Publish completion of canonical prefill and decode parity.
         * @param mtp_required Whether an independent MTP checkpoint remains.
         *
         * A promoted main-model expert can be certified immediately after the
         * ordinary checkpoints. A promoted sidecar expert cannot: its exact
         * routed contribution is produced only by the primary Hugging Face MTP
         * checkpoint. Encoding that distinction prevents the epilogue from
         * consuming an empty witness set before its final producer runs.
         */
        void recordMainParityComparison(bool mtp_required)
        {
            const auto *current =
                std::get_if<NumericalParityReady>(&state_);
            if (!current)
            {
                throw std::logic_error(
                    "Dynamic main parity completed out of order");
            }
            state_ = mtp_required
                         ? State{AwaitingMTPParity{current->origin}}
                         : State{NumericalEvidenceComplete{current->origin}};
        }

        /** @brief Publish the independent primary MTP checkpoint comparison. */
        void recordMTPParityComparison()
        {
            advance<AwaitingMTPParity, NumericalEvidenceComplete>(
                "Dynamic MTP parity completed before main parity or while MTP was disabled");
        }

        /** @return Whether every enabled numerical evidence producer ran. */
        [[nodiscard]] bool numericalEvidenceComplete() const noexcept
        {
            return std::holds_alternative<NumericalEvidenceComplete>(state_);
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
        struct NumericalParityReady
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct AwaitingMTPParity
        {
            DynamicResidencyConvergenceOrigin origin;
        };
        struct NumericalEvidenceComplete
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
            ConvergedCohortMeasured,
            NumericalParityReady,
            AwaitingMTPParity,
            NumericalEvidenceComplete>;
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
        AwaitingAuthenticatedPromotion,
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
        const MoEOptimizationMovementLedger &ledger) noexcept;

    /** Exact relationship between durable promotions and the parity workload. */
    enum class AuthenticatedPromotionConvergenceState : std::uint8_t
    {
        AwaitingPromotion,
        Satisfied,
        InvalidAuthorityEvidence,
    };

    /**
     * @brief Classify whether the durable suffix moved a reference-routed expert.
     *
     * Service-economy certification deliberately exercises a broad natural
     * corpus. A promotion authored from that traffic is valid production
     * movement, but a later Hugging Face checkpoint cannot prove its
     * destination math when the authenticated prompt never selects the expert.
     * The movement-only driver subsequently replays the exact reference
     * prefill. Require its demand to author at least one promotion before the
     * proof boundary settles; this joins workload causality without forcing a
     * route or consulting optional telemetry.
     *
     * @param target Complete typed convergence objective.
     * @param origin Durable ledger frontier before convergence traffic.
     * @param ledger Complete movement ledger from the production authority.
     * @param authenticated_routes Per-layer reference prefill route counts.
     * @return Awaiting, satisfied, or malformed authoritative evidence.
     */
    AuthenticatedPromotionConvergenceState
    classifyAuthenticatedPromotionConvergence(
        const DynamicResidencyConvergenceTarget &target,
        const DynamicResidencyConvergenceOrigin &origin,
        const MoEOptimizationMovementLedger &ledger,
        const std::vector<std::vector<std::uint64_t>>
            &authenticated_routes) noexcept;

    /**
     * @brief Join topology movement and reference-workload movement evidence.
     *
     * The production authority remains the sole author of both inputs. The
     * reference histogram only decides whether a completed promotion is
     * mathematically witnessable by this test; it never changes placement.
     */
    DynamicResidencyConvergenceState requireAuthenticatedPromotion(
        DynamicResidencyConvergenceState topology_convergence,
        AuthenticatedPromotionConvergenceState promotion_convergence) noexcept;

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
        std::uint64_t completed_route_count) noexcept;

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
        std::span<const float> expert_owners);

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
        std::span<const PublishedParticipantResidency> states);

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
        const MoERoutedExpertPlacementPlan &plan);

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
     * cohort and the immediately following numerical-parity request. Keeping
     * these purposes distinct prevents movement-only cells from trying to
     * admit evidence they will never execute.
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

    /**
     * @brief Exact production traffic admitted to movement optimization.
     *
     * Movement proof replays only the authenticated Hugging Face prefill. The
     * next ordinary request retires its pending progress before submitting the
     * next captured prefill, so repeated requests still close every demand
     * bank without injecting sampled or MTP-predictor routes. The later parity
     * prefill therefore replays the same causal rows that selected promotion.
     */
    struct MovementProofTrafficPlan
    {
        std::size_t authenticated_prefill_rows =
            kQwen35MoEParityTokenIds.size();
        std::uint64_t committed_decode_rows = 0u;
        std::uint64_t speculative_predictor_rows = 0u;

        /** @return Whether every admitted row belongs to reference prefill. */
        [[nodiscard]] constexpr bool authenticatedPrefillOnly() const noexcept
        {
            return authenticated_prefill_rows ==
                       kQwen35MoEParityTokenIds.size() &&
                   committed_decode_rows == 0u &&
                   speculative_predictor_rows == 0u;
        }
    };

    /** @return The sole production movement-proof traffic contract. */
    [[nodiscard]] constexpr MovementProofTrafficPlan
    movementProofTrafficPlan() noexcept
    {
        return {};
    }

    /**
     * @brief Exact cold-prefill/decode traffic used to train the speed witness.
     *
     * The observed-speedup cell must optimize the same three request identities
     * that its matched A/B cohort later measures.  Replaying an archived prefix
     * would execute only decode rows and turn four 96-row windows into hundreds
     * of tiny distributed transactions.  The fixture therefore crosses the
     * public prefix-archive purge boundary once per training request: the first
     * prefill is guaranteed to execute all authenticated-shaped rows and the
     * second prefill attempts the ordinary same-epoch prefix restore used by
     * the measured cohort. A concurrently published movement epoch may
     * correctly invalidate that entry and execute another cold prefill; the
     * accounting below deliberately relies only on the first guaranteed miss.
     */
    struct ConvergenceTrainingTrafficPlan
    {
        std::uint64_t cold_prefill_rows =
            kQwen35MoEConvergenceTimingPromptRows;
        std::uint64_t decode_forward_rows =
            kQwen35MoEConvergenceTimingDecodeForwards;

        /** @return Routed model rows guaranteed by one training request. */
        [[nodiscard]] constexpr std::uint64_t
        guaranteedRoutedRows() const noexcept
        {
            return cold_prefill_rows + decode_forward_rows;
        }

        /** @return Whether the plan has the exact measured cohort geometry. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return cold_prefill_rows ==
                       kQwen35MoEConvergenceTimingPromptRows &&
                   decode_forward_rows ==
                       kQwen35MoEConvergenceTimingDecodeForwards &&
                   guaranteedRoutedRows() > 0u;
        }
    };

    /** @return The sole production training contract for the speed witness. */
    [[nodiscard]] constexpr ConvergenceTrainingTrafficPlan
    convergenceTrainingTrafficPlan() noexcept
    {
        return {};
    }

    /** @return Exact routed rows executed by one complete timing cohort. */
    constexpr std::uint64_t convergenceTimingCohortRoutedRows() noexcept
    {
        return qwen35MoEConvergenceTimingCohortRoutedRows();
    }

    /** @return Timing plus numerical proof rows protected after movement. */
    constexpr std::uint64_t convergenceProtectedRoutedRows() noexcept
    {
        return qwen35MoEConvergenceProtectedRoutedRows();
    }

    /**
     * @brief Return demand-bank headroom required by a typed boundary.
     *
     * A movement-only boundary has no following protected evidence interval,
     * so it requires no demand-window geometry. The observed-speedup boundary
     * owns the exact routed-row requirement used by its matched A/B corpus and
     * the subsequent canonical prefix/decode proof. A profitable fifth wave
     * may still be admitted later; it may not invalidate the prefix between
     * the seed and restore that certify the already-proven placement epoch.
     *
     * @param purpose Evidence executed immediately after settlement.
     * @return Required routed rows, or no requirement for movement-only proof.
     */
    constexpr std::optional<std::uint64_t>
    convergenceBoundaryProtectedRows(
        ConvergenceBoundaryPurpose purpose) noexcept
    {
        if (purpose ==
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort)
        {
            return convergenceProtectedRoutedRows();
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
        std::uint64_t observed_routed_rows = 0u;
        std::uint64_t routed_rows = 0u;

        /** @return Whether the closure names positive work in one bank. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return routed_rows > 0u;
        }
    };

    /** Immutable request to seed one newly rotated empty demand bank. */
    struct DemandWindowSeed
    {
        std::uint64_t generation = 0u;
    };

    /**
     * @brief Receipt for demand work submitted through ordinary inference.
     *
     * The optimization worker consumes request progress asynchronously.  A
     * seed or exact closure therefore remains outstanding until the sole
     * authority publishes increased occupancy in the named bank or rotates to
     * a successor.  Retaining this receipt prevents the test driver from
     * issuing duplicate traffic while that publication edge is in flight.
     */
    struct SubmittedDemandWindowAdmission
    {
        std::uint64_t generation = 0u;
        std::uint64_t observed_routed_rows = 0u;
    };

    /** Authoritative observation of one submitted demand admission. */
    enum class DemandWindowAdmissionObservation : std::uint8_t
    {
        AwaitingPublication,
        Published,
        InvalidAuthorityEvidence,
    };

    /**
     * @brief Classify whether submitted demand is visible to the authority.
     *
     * Occupancy is monotonic within one generation.  Rotation to a newer
     * generation also proves that the submitted work was consumed.  Neither
     * an unchanged snapshot nor unrelated activity authorizes a duplicate
     * admission.
     *
     * @param admission Bank generation and occupancy observed before submit.
     * @param status Current passive status from the optimization authority.
     * @return Pending, published, or malformed/regressed evidence.
     */
    DemandWindowAdmissionObservation observeDemandWindowAdmission(
        const SubmittedDemandWindowAdmission &admission,
        const MoEOptimizationStatus &status) noexcept;

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
        ConvergenceBoundaryPurpose purpose) noexcept;

    /** Whether the finite certified traffic horizon may still admit requests. */
    enum class ConvergenceTrafficHorizon : std::uint8_t
    {
        Open,      ///< More ordinary stationary-workload requests are budgeted.
        Exhausted, ///< Only exact closure traffic may extend the finite horizon.
    };

    /**
     * @brief Complete typed settlement state for Dynamic movement proof.
     *
     * Movement completion and the passive post-movement boundary are separate
     * lifecycle edges. Closure before the target must not masquerade as target
     * completion, while closure after the target must retain that completion
     * as the authority rotates into a measurement-safe demand bank.
     */
    enum class DynamicConvergenceSettlementState : std::uint8_t
    {
        AwaitingMovement,
        AwaitingBoundaryQuiescence,
        NeedsMovementDemandWindowSeed,
        NeedsMovementDemandWindowClosure,
        NeedsBoundaryDemandWindowClosure,
        Ready,
        InvalidAuthorityEvidence,
    };

    /** Typed result of combining movement and boundary lifecycle evidence. */
    struct DynamicConvergenceSettlementDecision
    {
        DynamicConvergenceSettlementState state =
            DynamicConvergenceSettlementState::AwaitingMovement;
        std::optional<DemandWindowSeed> seed;
        std::optional<DemandWindowClosure> closure;

        /** @return Whether the authoritative movement target is satisfied. */
        [[nodiscard]] constexpr bool movementTargetSatisfied() const noexcept
        {
            return state == DynamicConvergenceSettlementState::
                                AwaitingBoundaryQuiescence ||
                   state == DynamicConvergenceSettlementState::
                                NeedsBoundaryDemandWindowClosure ||
                   state == DynamicConvergenceSettlementState::Ready;
        }
    };

    /**
     * @brief Classify one complete Dynamic convergence settlement transition.
     *
     * During the ordinary traffic horizon a partial demand bank remains normal
     * production state: later certified requests should fill it. Once that
     * finite horizon is exhausted, a reconciled host bank cannot make further
     * publication progress by itself. An empty successor requests one
     * authenticated seed; a partial bank returns its exact remaining row count
     * as cache-distinct closure traffic. These transitions are deliberately
     * distinct from post-target closure used to prepare an immutable timing
     * cohort.
     *
     * @param convergence Authoritative movement-target classification.
     * @param status Passive status from the sole optimization authority.
     * @param purpose Evidence that follows successful settlement.
     * @param horizon Whether ordinary certified traffic remains admissible.
     * @return One explicit lifecycle transition and any exact closure work.
     */
    DynamicConvergenceSettlementDecision classifyDynamicConvergenceSettlement(
        DynamicResidencyConvergenceState convergence,
        const MoEOptimizationStatus &status,
        ConvergenceBoundaryPurpose purpose,
        ConvergenceTrafficHorizon horizon) noexcept;

    /**
     * Histogram geometry large enough to keep one complete A/B cohort inside
     * one immutable residency epoch while ordinary maintenance remains live.
     * Three initial requests contribute 3 * (17 prefill + 2 decode) = 57 routed
     * rows. A 96-row window therefore cannot publish during the baseline and
     * still admits one already-running 19-row request, the complete
     * post-movement cohort, and the following canonical parity request after
     * the fourth asynchronous publication.
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
     * Exact bounded training requests needed for the four speedup windows.
     *
     * Each request first retires the reusable prefix archive through the same
     * public boundary exposed to production operators. The ensuing cold
     * prefill and two real decode forwards therefore contribute at least
     * 17 + 2 routed rows. An asynchronous publication may invalidate the
     * attempted restore and contribute another 17 rows, but the bounded
     * horizon never depends on that race. This preserves the measured request
     * geometry while avoiding 192 tiny restored-prefix transactions.
     */
    constexpr ConvergenceTrainingTrafficPlan
        kConvergenceTrainingTraffic = convergenceTrainingTrafficPlan();
    static_assert(kConvergenceTrainingTraffic.valid());
    constexpr int kMaximumDynamicHistogramRequests =
        kObservedSpeedupConvergenceWindows *
        ((kConvergenceHistogramWindowTokens +
          static_cast<int>(
              kConvergenceTrainingTraffic.guaranteedRoutedRows()) -
          1) /
         static_cast<int>(
             kConvergenceTrainingTraffic.guaranteedRoutedRows()));
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
        std::uint64_t required_publications);
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
    SamplingParams referenceGreedySamplingPolicy();

    bool isLegacyOverlayRuntimeEnabled();

    /** @return Exact generated diagnostic identity for the active cell. */
    std::string activeTestName();

    /** @return Whether the active typed model is the 122B MTP identity. */
    bool isQwen122ProductionTest();

    /** @return Model root selected by the active real-weight cell. */
    const char *activeModelPath();

    /** @return Authenticated reference directory selected by the active cell. */
    const char *activeSnapshotDir();

    /** @return Whether every split file required by the active model exists. */
    bool modelAvailable();

    /** @return Whether this case uses current-batch least-loaded assignment. */
    bool isLLEPProductionTest();

    /** @brief Return whether the exact cell exercises persistent tier movement. */
    bool isDynamicResidencyProductionTest();

    /** @brief Return whether initial expert ownership uses seeded random order. */
    bool isRandomOwnerProductionTest();

    /**
     * @return Fixed MTP depth, the adaptive policy's maximum, or zero when off.
     *
     * Generated cells carry this policy explicitly.  Legacy topology cells do
     * not name an MTP mode and therefore must report zero; treating an absent
     * label as depth three made post-run evidence demand an MTP domain from a
     * runtime whose typed configuration correctly disabled MTP.
     */
    int activeMTPDraftDepth();

    /** @return Setup-time MTP draft capacity retained by the active cell. */
    int activeMTPRetainedDraftCapacity();

    /**
     * @brief Return the retained grouped-verifier row capacity for this cell.
     *
     * Every 122B campaign cell shares one maximum-capacity model context so
     * changing the requested fixed depth does not reload weights or rebuild
     * device graphs.  The device transaction still publishes its independent
     * logical depth through @ref activeMTPDraftDepth and selects a physical
     * bucket inside this admitted envelope.
     */
    int activeMTPGraphCapacityVerifierRows();

    /**
     * @brief Return the exact physical bucket selected by this transaction.
     *
     * Fixed-depth scalar transactions select the smallest retained power-of-two
     * bucket that contains their logical rows.  Dynamic depth instead embeds
     * the maximum envelope because active depth is device-owned replay data.
     */
    int activeMTPPhysicalVerifierRows();

    /** @return Whether the production device depth controller is adaptive. */
    bool usesDynamicMTPDepth();

    /** @return Whether the active generated cell enables MTP. */
    bool activeMTPEnabled();

    /** @brief Return whether the active topology contains a CUDA participant. */
    bool topologyUsesCuda();

    /** @brief Return whether the active topology contains a ROCm participant. */
    bool topologyUsesRocm();

    /** @brief Return whether the active topology contains NodeTP CPU cold. */
    bool topologyUsesCpu();

    /** @brief Return the exact sparse participants declared by the active topology. */
    size_t activeOverlayParticipantCount();

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
    bool topologyHasSecondaryGpuDomain();

    /** @return Whether sparse ExpertOverlay work crosses an MPI process. */
    bool topologySpansMultipleMPIRanks();

    /** @return Number of integer-priority residency tiers in this cell. */
    size_t activeOverlayTierCount();

    RoutedExpertTier makeTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        bool fallback = false);

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
        ModelParityDynamicSpeedupWitness dynamic_speedup_witness;
    };

    /** @return Every unique 122B ExpertOverlay topology in the production matrix. */
    const std::array<Qwen122OverlayTopologySpec, 9> &
    qwen122OverlayTopologySpecs();

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
        bool continuation);

    /** @brief Build a one- or two-socket CPU tier resolved by live inventory. */
    RoutedExpertDomain qwen122CpuDomain(int participant_count);

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
    qwen122OverlayBlueprint(const Qwen122OverlayTopologySpec &spec);

    MoERoutedExpertModelMetadata topologyOnlyMetadata();

    MoERoutedExpertModelMetadata metadataFromModel(const ModelContext &ctx);

    MoERoutedExpertPlacementPlan requestedPlan(
        const MoERoutedExpertModelMetadata &metadata);

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
        const ClusterInventory &inventory);

    std::optional<std::string> acceleratorHardwareBlocker(
        const ClusterInventory &inventory);

    /**
     * @brief Return whether this fixture instance must prove bucketed prefill.
     *
     * The normal production campaign deliberately uses one exact authenticated
     * bucket for speed. This named cell is the explicit complementary contract:
     * it keeps real weights and the same CSV oracle but forces an ordered
     * heterogeneous sparse-collective schedule.
     */
    bool isSegmentedPrefillProductionTest();

    /** @return Typed fixed capture rows, or zero for ordinary prefill. */
    int activeSegmentedPrefillCaptureRows();

    /**
     * @return Complete 35B topology x placement x movement x prefill matrix.
     *
     * The CUDA/ROCm/CPU topology owns both ordinary and segmented captured
     * prefill profiles. Every other topology owns the ordinary profile. All
     * cells, including the ten historical cases, are emitted only by the
     * central typed expander.
     */
    const std::vector<ModelParityCase> &qwen35GraphNativeParityCases();

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
        int transformer_layers);

    /**
     * @brief Derive the two typed Dynamic evidence policies for Qwen 122B.
     *
     * A movement-only cell needs one conflict-free priority cycle and, when
     * the topology exposes it, one same-priority participant cycle.  Its
     * observation window expands to the declared model context immediately
     * after that publication so numerical parity executes against the proven
     * epoch without inducing unrelated migration churn.  The designated A/B
     * witness retains the full layer-parallel transfer fabric and one fixed
     * window large enough for its complete timing cohort, canonical parity
     * request, and one publication-overlap request.
     *
     * @param spec Typed physical topology expanded by the canonical matrix.
     * @param transformer_layers Authenticated main-model routed layer count.
     * @param maximum_context_rows Declared request context admission.
     * @return Complete evidence-indexed policies used by matrix expansion.
     */
    ModelParityDynamicRuntimePolicies qwen122DynamicRuntimePolicies(
        const Qwen122OverlayTopologySpec &spec,
        int transformer_layers,
        int maximum_context_rows);

    /**
     * @brief Declare one canonical 122B real-model/topology parity matrix.
     *
     * Cross-rank GPU ownership is deliberately unresolved. Production cluster
     * inventory binds each ordinal to whichever MPI instance currently owns
     * it. NCCL/RCCL remain inside their named domains; the outer graph is
     * multi-domain, not a fictitious heterogeneous tensor-parallel collective.
     */
    ModelParityDefinition qwen122ExpertOverlayParityDefinition(
        const Qwen122OverlayTopologySpec &spec);

    /**
     * @return Deduplicated topology x 4 placement/movement x 6 MTP matrix.
     *
     * Topology identifiers and complete generated case names are checked here
     * because this binary joins several independently valid definitions into
     * one process-campaign catalogue. A duplicate would otherwise make GTest
     * registration order, rather than the typed source, choose the live case.
     */
    const std::vector<ModelParityCase> &qwen122ExpertOverlayParityCases();

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

    /**
     * @brief Complete immutable identity of a reusable production runner.
     *
     * Request-selectable MTP depth is deliberately absent. Every enabled 122B
     * cell owns the same depth-fifteen arena/graph envelope and selects its
     * fixed or dynamic policy through MTPRequestPolicy after reset. MTP-off is
     * not compatible: its main graphs intentionally omit shifted-cache and
     * predictor publication work, preserving the real non-MTP production path.
     */
    struct Qwen122OverlayRunnerIdentity
    {
        Qwen122OverlayPhysicalIdentity physical;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
        KVCachePrecision kv_cache_precision = KVCachePrecision::FP32;
        int max_seq_len = 0;
        int retained_mtp_draft_capacity = 0;
        int prefill_capture_rows = 0;
        /** Stable underlying value of the fixture's typed snapshot topology. */
        std::uint8_t snapshot_mode = 0u;

        friend bool operator==(
            const Qwen122OverlayRunnerIdentity &,
            const Qwen122OverlayRunnerIdentity &) = default;
    };

    /** One process-local prepared authority and its complete physical identity. */
    struct Qwen122OverlayModelContextCampaignCache
    {
        std::mutex mutex;
        std::string model_path;
        std::optional<Qwen122OverlayPhysicalIdentity> physical_identity;
        std::optional<ModelContextReuseContract> contract;
        /** One yielded runner; populated only for compatible static MTP cells. */
        std::optional<Qwen122OverlayRunnerIdentity> runner_identity;
        std::unique_ptr<IOrchestrationRunner> runner;
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

    /** Rank-wide decision to construct or adopt one exact retained runner. */
    enum class Qwen122CampaignRunnerAdmission : std::uint8_t
    {
        Fresh,
        Retained,
    };

    /** Complete consensus result for a process-resident runner probe. */
    struct Qwen122CampaignRunnerAdmissionResult
    {
        Qwen122CampaignRunnerAdmission admission{
            Qwen122CampaignRunnerAdmission::Fresh};
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
        std::string_view local_error);

    /**
     * @brief Require every MPI rank to choose the same retained-runner path.
     * @param control_communicator Per-cell test-control communicator.
     * @param local_cache_hit Whether this process owns the exact runner.
     * @param local_error Rank-local cache validation failure.
     * @return One unanimous fresh/retained decision or a fatal diagnostic.
     */
    Qwen122CampaignRunnerAdmissionResult
    reachQwen122CampaignRunnerAdmission(
        MPI_Comm control_communicator,
        bool local_cache_hit,
        std::string_view local_error);

    /** @return The sole bounded 122B prepared-weight cache in this MPI process. */
    Qwen122OverlayModelContextCampaignCache &
    qwen122OverlayModelContextCampaignCache();

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
        std::optional<float> mtp_recursive_aggregate_cosine_floor;
        float kl_threshold = 0.0f;
        int pytorch_top1_in_topk = 0;
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
    DeferredMTPBranchCampaign &deferredMTPBranchCampaign();

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
        std::string *error);

    /**
     * @brief Release the final immutable 122B authority before Python loads it.
     * @param error Receives a precise retirement failure.
     * @return True when all model-lifetime allocations were certified free.
     */
    bool releaseQwen122OverlayModelContextCampaignCache(std::string *error);

    /** @return A shell-safe single argument preserving every input byte. */
    std::string quoteDeferredReferenceArgument(const std::string &value);

    /**
     * @brief Load one FP32/FP64 NPY tensor without constructing a parity fixture.
     * @param path Exact authenticated reference tensor path.
     * @param error Receives a precise load/type failure.
     * @return FP32 values, or an empty optional on failure.
     */
    std::optional<std::vector<float>> loadDeferredReferenceTensor(
        const std::filesystem::path &path,
        std::string *error);

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
        int num_experts);

    /**
     * @brief Complete one deferred context with the same route-aware math gate.
     * @param context Immutable live evidence copied before runner teardown.
     * @param error Receives every failed invariant for campaign diagnostics.
     * @return True only when all tensors and the aggregate semantic gate pass.
     */
    bool compareDeferredMTPBranchContext(
        const DeferredMTPBranchContext &context,
        std::string *error);

    /**
     * @brief Generate all missing branches with one loaded HF model and compare.
     * @param contexts Immutable queue removed from the live campaign.
     * @param error Receives the generator output or first numerical failure.
     */
    bool resolveDeferredMTPBranchCampaign(
        const std::vector<DeferredMTPBranchContext> &contexts,
        std::string *error);


}
